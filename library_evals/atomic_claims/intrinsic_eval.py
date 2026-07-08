"""
Intrinsic evaluation of atomic-claim extraction quality.

Reads a JSONL of {question, answer, docs, predicted_claims} (as produced by
run_extractor.py) and computes, per example:

    - faithfulness:      mean NLI(answer -> claim) across claims
    - faithful_fraction: fraction of claims with NLI(answer -> claim) >= threshold
    - coverage_sentence: fraction of answer sentences entailed by some claim
    - coverage_whole:    NLI(concat(claims) -> answer)
    - atomicity:         fraction of claims with no clause-level conjunction
    - non_redundancy:    1 - fraction of claims with a bidirectional near-duplicate

Writes two files:
    - <prefix>_per_example.jsonl: one record per input row with all sub-scores
    - <prefix>_summary.json:      dataset-wide aggregates

NLI backend: a HuggingFace sequence-classification model with labels
{entailment, neutral, contradiction}. Default: cross-encoder/nli-deberta-v3-large.
The entailment label is looked up from model.config.id2label (case-insensitive).

Atomicity backend: spacy (en_core_web_sm). A claim is flagged non-atomic when
any coordinating-conjunction token (dep == "cc") connects two verbs, i.e.
there is a token with dep == "conj" whose head is a verb and which is itself
a verb. Noun-phrase conjunctions ("Paris, France and the capital") are
allowed.

Usage:
    python intrinsic_eval.py \\
        --input predicted_claims.jsonl \\
        --output-prefix results \\
        --device cuda --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from acclaim.claims.sentence import SentenceClaimExtractor
from acclaim.data_models import Answer
from acclaim.text_utils import jaccard_similarity, tokenize_text

logger = logging.getLogger(__name__)


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOPK = 3

ALL_METRICS = {
    "faithfulness",
    "coverage",
    "coverage_whole",
    "sentence_claim_coverage",
    "atomicity",
    "non_redundancy",
}
DEFAULT_METRICS = {
    "coverage",
    "coverage_whole",
    "sentence_claim_coverage",
    "atomicity",
    "non_redundancy",
}


@dataclass
class ExampleScores:
    alce_index: int | None
    n_claims: int
    n_answer_sentences: int
    faithfulness_mean: float | None
    faithful_fraction: float | None
    coverage_sentence: float | None
    coverage_whole: float | None
    atomicity: float | None
    non_redundancy: float | None
    per_claim_faithfulness: list[float] | None
    # Coverage diagnostics — populated when "coverage" is enabled.
    # per_sentence_coverage: NLI(concat(top-K claims) -> sentence)
    # per_sentence_topk: list of claim indices selected per sentence (Jaccard)
    per_sentence_coverage: list[float] | None
    per_sentence_topk: list[list[int]] | None
    # sentence_claim_coverage diagnostics: per-sentence flag, claim->sentence assignment.
    sentence_claim_coverage: float | None
    per_sentence_has_claim: list[bool] | None
    per_claim_assigned_sentence: list[int] | None
    per_claim_atomic: list[bool] | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# NLI scorer
# --------------------------------------------------------------------------- #


class NliScorer:
    """Batched bidirectional-entailment scorer backed by HuggingFace."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch  # local import so the file parses without torch installed
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # DeBERTa-v2's relative-position code triggers a JIT-fused kernel that
        # NVRTC tries to compile at runtime. On systems whose CUDA toolkit
        # libs don't match torch's bundled NVRTC (e.g. missing
        # libnvrtc-builtins.so.13.0), this throws at first forward pass.
        # Disable the TorchScript/NVFuser fusers so the model runs the eager
        # fallback path. No-op on CPU.
        try:
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_override_can_fuse_on_cpu(False)
        except Exception:  # noqa: BLE001 — older/newer torch may rename these
            pass

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.entail_idx = _find_entailment_index(self.model.config.id2label)
        logger.info(
            "NLI model loaded: %s (entailment idx=%d, device=%s)",
            model_name,
            self.entail_idx,
            device,
        )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return entailment probability for each (premise, hypothesis) pair."""
        if not pairs:
            return []

        torch = self._torch
        probs: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                premises = [p for p, _ in batch]
                hypotheses = [h for _, h in batch]
                enc = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                batch_probs = torch.softmax(logits, dim=-1)[:, self.entail_idx]
                probs.extend(batch_probs.detach().cpu().tolist())
        return probs


def _find_entailment_index(id2label: dict[int, str]) -> int:
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)
    raise ValueError(
        f"Could not locate an 'entailment' label in id2label={id2label}. "
        "Use an NLI model with a standard contradiction/neutral/entailment head."
    )


# --------------------------------------------------------------------------- #
# Atomicity (spacy rule-based)
# --------------------------------------------------------------------------- #


class AtomicityChecker:
    """Flag claims that bundle two clause-level facts via a coordinating verb."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        import spacy

        try:
            self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' not installed. "
                f"Run: python -m spacy download {spacy_model}"
            ) from exc

    def is_atomic(self, claim: str) -> bool:
        doc = self.nlp(claim)
        for tok in doc:
            if tok.dep_ == "conj" and tok.pos_ in {"VERB", "AUX"}:
                head = tok.head
                if head.pos_ in {"VERB", "AUX"}:
                    return False
        return True


# --------------------------------------------------------------------------- #
# Per-example scoring
# --------------------------------------------------------------------------- #


def _answer_sentences(answer_text: str) -> list[str]:
    """Split the answer into citation-stripped sentences using the library splitter."""
    if not answer_text:
        return []
    claims = SentenceClaimExtractor().extract(Answer(text=answer_text))
    return [c.text for c in claims if c.text.strip()]


def _score_faithfulness(nli: NliScorer, answer: str, claims: list[str]) -> list[float]:
    if not claims or not answer:
        return []
    return nli.score([(answer, c) for c in claims])


def _select_top_k_claims(sentence: str, claims: list[str], k: int) -> list[int]:
    """Pick the indices of the k claims with highest Jaccard token overlap to sentence."""
    s_toks = tokenize_text(sentence)
    if not s_toks or not claims:
        return list(range(min(k, len(claims))))
    scored = [
        (jaccard_similarity(s_toks, tokenize_text(c)), i) for i, c in enumerate(claims)
    ]
    scored.sort(reverse=True)
    return [i for _, i in scored[:k]]


def _score_sentence_coverage(
    nli: NliScorer,
    sentences: list[str],
    claims: list[str],
    k: int,
) -> tuple[list[float], list[list[int]]]:
    """
    Coverage = NLI(concat(top-k claims by Jaccard overlap) -> sentence).

    A single atomic claim cannot entail a multi-fact answer-sentence on its own;
    NLI requires the premise to imply the entire hypothesis. Concatenating the
    top-k most lexically-related claims gives the union of facts a chance to
    cover the whole sentence.

    Returns (per_sentence_prob, per_sentence_selected_claim_indices).
    """
    if not sentences:
        return [], []
    if not claims:
        return [0.0] * len(sentences), [[] for _ in sentences]

    selected = [_select_top_k_claims(s, claims, k) for s in sentences]
    pairs = [
        (" ".join(claims[i] for i in sel), s) if sel else ("", s)
        for s, sel in zip(sentences, selected)
    ]
    probs = nli.score(pairs) if pairs else []
    return list(probs), selected


def _score_sentence_claim_coverage(
    sentences: list[str], claims: list[str], min_jaccard: float = 0.0
) -> tuple[float, list[bool], list[int]]:
    """
    Lexical sentence coverage: fraction of answer-sentences that are the best
    Jaccard match for at least one claim. Mirrors JaccardCitationAligner.

    Returns (coverage, per_sentence_has_claim, per_claim_assigned_sentence).
    A sentence assignment of -1 means the claim had Jaccard <= min_jaccard
    against every sentence (e.g. empty claim or no token overlap).
    """
    if not sentences:
        return 0.0, [], []
    if not claims:
        return 0.0, [False] * len(sentences), []

    sent_token_sets = [tokenize_text(s) for s in sentences]
    has_claim = [False] * len(sentences)
    assigned: list[int] = []
    for claim in claims:
        c_toks = tokenize_text(claim)
        best_score, best_idx = min_jaccard, -1
        for i, s_toks in enumerate(sent_token_sets):
            score = jaccard_similarity(c_toks, s_toks)
            if score > best_score:
                best_score, best_idx = score, i
        assigned.append(best_idx)
        if best_idx >= 0:
            has_claim[best_idx] = True

    cov = sum(has_claim) / len(has_claim)
    return cov, has_claim, assigned


def _score_whole_coverage(nli: NliScorer, answer: str, claims: list[str]) -> float:
    if not claims or not answer:
        return 0.0
    concat = " ".join(claims)
    return nli.score([(concat, answer)])[0]


def _score_redundancy(nli: NliScorer, claims: list[str], threshold: float) -> float:
    """
    Redundancy score = fraction of claims that have at least one other claim
    p_j with bidirectional entailment >= threshold.
    """
    n = len(claims)
    if n < 2:
        return 0.0
    forward_pairs = [
        (claims[i], claims[j]) for i in range(n) for j in range(n) if i != j
    ]
    forward = nli.score(forward_pairs)

    def _lookup(i: int, j: int) -> float:
        if i == j:
            return 0.0
        idx = i * (n - 1) + (j if j < i else j - 1)
        return forward[idx]

    redundant_flags = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bidir = min(_lookup(i, j), _lookup(j, i))
            if bidir >= threshold:
                redundant_flags[i] = True
                break
    return sum(redundant_flags) / n


def score_example(
    *,
    record: dict,
    nli: NliScorer,
    atomicity: AtomicityChecker | None,
    threshold: float,
    metrics: set[str],
    topk: int = DEFAULT_TOPK,
) -> ExampleScores:
    answer = (record.get("answer") or "").strip()
    claims = [c for c in record.get("predicted_claims", []) if c and c.strip()]
    sentences = _answer_sentences(answer)

    per_claim_faith: list[float] | None = None
    faith_mean: float | None = None
    faith_frac: float | None = None
    if "faithfulness" in metrics:
        per_claim_faith = _score_faithfulness(nli, answer, claims)
        faith_mean = statistics.fmean(per_claim_faith) if per_claim_faith else 0.0
        faith_frac = (
            sum(p >= threshold for p in per_claim_faith) / len(per_claim_faith)
            if per_claim_faith
            else 0.0
        )

    per_sentence_cov: list[float] | None = None
    per_sentence_topk: list[list[int]] | None = None
    cov_sent: float | None = None
    if "coverage" in metrics:
        per_sentence_cov, per_sentence_topk = _score_sentence_coverage(
            nli, sentences, claims, topk
        )
        cov_sent = (
            sum(p >= threshold for p in per_sentence_cov) / len(per_sentence_cov)
            if per_sentence_cov
            else 0.0
        )

    whole_cov: float | None = None
    if "coverage_whole" in metrics:
        whole_cov = _score_whole_coverage(nli, answer, claims)

    sent_claim_cov: float | None = None
    per_sentence_has_claim: list[bool] | None = None
    per_claim_assigned: list[int] | None = None
    if "sentence_claim_coverage" in metrics:
        sent_claim_cov, per_sentence_has_claim, per_claim_assigned = (
            _score_sentence_claim_coverage(sentences, claims)
        )

    atomic_flags: list[bool] | None = None
    atomicity_score: float | None = None
    if "atomicity" in metrics and atomicity is not None:
        atomic_flags = [atomicity.is_atomic(c) for c in claims]
        atomicity_score = sum(atomic_flags) / len(atomic_flags) if atomic_flags else 0.0

    non_red: float | None = None
    if "non_redundancy" in metrics:
        non_red = 1.0 - _score_redundancy(nli, claims, threshold)

    return ExampleScores(
        alce_index=record.get("alce_index"),
        n_claims=len(claims),
        n_answer_sentences=len(sentences),
        faithfulness_mean=faith_mean,
        faithful_fraction=faith_frac,
        coverage_sentence=cov_sent,
        coverage_whole=whole_cov,
        atomicity=atomicity_score,
        non_redundancy=non_red,
        per_claim_faithfulness=per_claim_faith,
        per_sentence_coverage=per_sentence_cov,
        per_sentence_topk=per_sentence_topk,
        sentence_claim_coverage=sent_claim_cov,
        per_sentence_has_claim=per_sentence_has_claim,
        per_claim_assigned_sentence=per_claim_assigned,
        per_claim_atomic=atomic_flags,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _summarize(examples: list[ExampleScores]) -> dict:
    def stats(values: Iterable[float | None]) -> dict[str, float]:
        vs = [v for v in values if v is not None]
        if not vs:
            return {"mean": None, "std": None, "n": 0, "frac_ge_0_9": None}
        return {
            "mean": statistics.fmean(vs),
            "std": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
            "n": len(vs),
            "frac_ge_0_9": sum(v >= 0.9 for v in vs) / len(vs),
        }

    return {
        "n_examples": len(examples),
        "claim_count": stats([e.n_claims for e in examples]),
        "faithfulness_mean": stats([e.faithfulness_mean for e in examples]),
        "faithful_fraction": stats([e.faithful_fraction for e in examples]),
        "coverage_sentence": stats([e.coverage_sentence for e in examples]),
        "coverage_whole": stats([e.coverage_whole for e in examples]),
        "sentence_claim_coverage": stats([e.sentence_claim_coverage for e in examples]),
        "atomicity": stats([e.atomicity for e in examples]),
        "non_redundancy": stats([e.non_redundancy for e in examples]),
    }


# --------------------------------------------------------------------------- #
# IO + CLI
# --------------------------------------------------------------------------- #


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _autodevice() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Output prefix; writes <prefix>_per_example.jsonl and <prefix>_summary.json",
    )
    parser.add_argument("--nli-model", type=str, default=DEFAULT_NLI_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device", type=str, default=None, help="cpu|cuda (auto-detect)"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(ALL_METRICS),
        default=sorted(DEFAULT_METRICS),
        help=(
            "Which metrics to compute. Defaults exclude 'faithfulness' (NLI "
            "answer->claim is not load-bearing — claims are LLM rephrasings) "
            "and 'coverage_whole' (NLI 512-token cap is brittle on long answers)."
        ),
    )
    parser.add_argument(
        "--coverage-topk",
        type=int,
        default=DEFAULT_TOPK,
        help="Number of top-Jaccard claims concatenated as the coverage premise.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    metrics = set(args.metrics)
    device = args.device or _autodevice()
    nli = NliScorer(
        model_name=args.nli_model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    atomicity = AtomicityChecker(args.spacy_model) if "atomicity" in metrics else None

    records = list(_iter_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    per_example_path = args.output_prefix.parent / (
        args.output_prefix.name + "_per_example.jsonl"
    )
    summary_path = args.output_prefix.parent / (
        args.output_prefix.name + "_summary.json"
    )
    per_example_path.parent.mkdir(parents=True, exist_ok=True)

    scored: list[ExampleScores] = []
    with per_example_path.open("w", encoding="utf-8") as out:
        for rec in tqdm(records, desc="scoring", unit="ex"):
            scores = score_example(
                record=rec,
                nli=nli,
                atomicity=atomicity,
                threshold=args.threshold,
                metrics=metrics,
                topk=args.coverage_topk,
            )
            scored.append(scores)
            out.write(json.dumps(scores.to_dict(), ensure_ascii=False) + "\n")
            out.flush()

    summary = _summarize(scored)
    summary["config"] = {
        "nli_model": args.nli_model,
        "threshold": args.threshold,
        "metrics": sorted(metrics),
        "coverage_topk": args.coverage_topk,
        "input": str(args.input),
    }
    with summary_path.open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2)

    print(f"per-example -> {per_example_path}")
    print(f"summary     -> {summary_path}")
    _print_summary_table(summary)


def _print_summary_table(summary: dict) -> None:
    rows = [
        ("Faithfulness (mean ent.)", "faithfulness_mean"),
        ("Faithful fraction (>= thr)", "faithful_fraction"),
        ("Coverage (sentence-level NLI)", "coverage_sentence"),
        ("Coverage (whole-answer NLI)", "coverage_whole"),
        ("Coverage (>=1 claim/sent)", "sentence_claim_coverage"),
        ("Atomicity", "atomicity"),
        ("Non-redundancy", "non_redundancy"),
    ]
    print()
    print(f"{'metric':32s} {'mean':>8s} {'std':>8s} {'%>=0.9':>8s}")
    print("-" * 60)
    for label, key in rows:
        s = summary[key]
        if s["mean"] is None:
            print(f"{label:32s} {'-':>8s} {'-':>8s} {'-':>8s}  (not computed)")
        else:
            print(
                f"{label:32s} {s['mean']:>8.3f} {s['std']:>8.3f} {s['frac_ge_0_9']:>8.3f}"
            )
    cc = summary["claim_count"]
    print()
    print(
        f"claim count per example: mean={cc['mean']:.2f} std={cc['std']:.2f} n={cc['n']}"
    )


if __name__ == "__main__":
    main()
