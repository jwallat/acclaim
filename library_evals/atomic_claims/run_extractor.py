"""
Run the AtomicClaimExtractor over a sample of ALCE/ASQA answers and write
predicted atomic claims to JSONL for downstream intrinsic evaluation.

Usage:
    python run_extractor.py \\
        --input /home/wallat/citation_eval/ALCE/result/asqa-Meta-Llama-3-8B-Instruct-gtr-shot1-ndoc3-43.json \\
        --output predicted_claims.jsonl \\
        --n 500 \\
        --seed 42 \\
        --model openai/meta-llama/Meta-Llama-3-8B-Instruct \\
        --api-base http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from acclaim.claims.atomic import AtomicClaimExtractor
from acclaim.data_models import Answer


def _silence_litellm() -> None:
    """Suppress LiteLLM's per-call INFO chatter (and HTTPX request logs)."""
    os.environ.setdefault("LITELLM_LOG", "WARNING")
    for name in ("LiteLLM", "litellm", "litellm.utils", "httpx", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        import litellm

        litellm.suppress_debug_info = True
    except Exception:  # noqa: BLE001
        pass


logger = logging.getLogger(__name__)


DEFAULT_INPUT = Path(
    "/home/wallat/citation_eval/ALCE/result/"
    "asqa-Meta-Llama-3-8B-Instruct-gtr-shot1-ndoc3-43.json"
)
DEFAULT_OUTPUT = Path(__file__).parent / "predicted_claims.jsonl"


def _load_alce(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob["data"]


def _sample_indices(n_total: int, n_sample: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    n_sample = min(n_sample, n_total)
    return sorted(rng.sample(range(n_total), n_sample))


def _make_extractor(args: argparse.Namespace) -> AtomicClaimExtractor:
    return AtomicClaimExtractor(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )


def _extract_one(
    extractor: AtomicClaimExtractor, record: dict[str, Any], model: str
) -> dict[str, Any]:
    answer_text = (record.get("output") or "").strip()
    claims = extractor.extract(Answer(text=answer_text)) if answer_text else []
    return {
        "question": record.get("question", ""),
        "answer": answer_text,
        "docs": record.get("docs", []),
        "predicted_claims": [c.text for c in claims],
        "model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="openai/meta-llama/Meta-Llama-3-8B-Instruct",
        help="LiteLLM model id. Use 'openai/<hf-name>' for a local vLLM server.",
    )
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12196)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard cap on number of examples actually processed "
        "(useful for smoke tests).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _silence_litellm()

    data = _load_alce(args.input)
    indices = _sample_indices(len(data), args.n, args.seed)
    if args.limit is not None:
        indices = indices[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    extractor = _make_extractor(args)

    n_written = 0
    n_failed = 0
    with args.output.open("w", encoding="utf-8") as out:
        pbar = tqdm(indices, desc="extracting atomic claims", unit="ex")
        for idx in pbar:
            record = data[idx]
            try:
                row = _extract_one(extractor, record, args.model)
            except Exception as exc:  # noqa: BLE001 — log and continue
                logger.error("extraction failed on idx=%d: %s", idx, exc)
                row = {
                    "question": record.get("question", ""),
                    "answer": (record.get("output") or "").strip(),
                    "docs": record.get("docs", []),
                    "predicted_claims": [],
                    "model": args.model,
                    "error": str(exc),
                }
                n_failed += 1
            row["alce_index"] = idx
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_written += 1
            pbar.set_postfix(claims=len(row["predicted_claims"]), failed=n_failed)

    print(f"wrote {n_written} records -> {args.output}")


if __name__ == "__main__":
    main()
