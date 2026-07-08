"""
Main evaluation pipeline for acclaim.

Entry point: :func:`evaluate`.

Pipeline stages
---------------
1. **Claim extraction** — split the answer into claims
2. **Citation alignment** — map each claim to its cited document IDs
3. **Evidence judgement** — verify each claim against its documents
4. **Metric computation** — aggregate scalar scores

All intermediate outputs are stored in ``EvaluationResult.steps`` so
callers and tests can inspect them without modifying the pipeline.
"""

from __future__ import annotations

import logging

from .claims.base import ClaimExtractor
from .claims.sentence import SentenceClaimExtractor
from .claims.atomic import AtomicClaimExtractor
from .alignment.base import CitationAligner
from .alignment.sentence import SentenceCitationAligner
from .alignment.jaccard import JaccardCitationAligner
from ._version import __version__
from .concurrency import parallel_map
from .config.load import EvalConfig, config_to_dict, load_config
from .data_models import (
    AlignedClaim,
    Answer,
    ClaimResult,
    Document,
    EvaluationMetadata,
    EvaluationResult,
)
from .judges.base import EvidenceJudge
from .judges.citation_precision import LiteLLMPrecisionJudge, PrecisionJudge
from .judges.citation_recall import LiteLLMRecallJudge, RecallJudge
from .judges.litellm import LiteLLMJudge
from .judges.relevance import LiteLLMRelevanceJudge, RelevanceJudge
from .metrics.base import Metric
from .metrics.citation_f1 import CitationF1Metric
from .metrics.citation_precision import CitationPrecisionMetric
from .metrics.citation_recall import CitationRecallMetric
from .metrics.registry import METRIC_REGISTRY, compute_metrics
from .metrics.stratified import RelevanceStratifiedMetric

logger = logging.getLogger(__name__)

# ── Claim-extractor registry ──────────────────────────────────────────────────
# Maps string keys (used in config YAML) to extractor *classes*.
# When the extractor needs LLM params, they are forwarded from JudgeConfig
# so the user only needs to configure one model endpoint.
_CLAIM_EXTRACTOR_REGISTRY: dict[str, type[ClaimExtractor]] = {
    "sentence": SentenceClaimExtractor,
    "atomic": AtomicClaimExtractor,
}

# ── Default aligner per extractor type ───────────────────────────────────────
_DEFAULT_ALIGNER: dict[str, type[CitationAligner]] = {
    "sentence": SentenceCitationAligner,
    "atomic": JaccardCitationAligner,
}


def evaluate(
    answer: str,
    documents: list[Document],
    *,
    question: str | None = None,
    citation_to_doc: dict[int, str] | None = None,
    config: EvalConfig | None = None,
    # Fine-grained overrides (rarely needed — use config instead)
    claim_extractor: ClaimExtractor | None = None,
    aligner: CitationAligner | None = None,
    judge: EvidenceJudge | None = None,
    _inner_max_workers: int | None = None,
    _include_metadata: bool = True,
) -> EvaluationResult:
    """
    Evaluate citation attribution quality of an LLM-generated answer.

    Args:
        answer:          LLM-generated answer text containing ``[N]`` citation
                         markers (e.g. ``"Paris is the capital of France [1]."``).
        documents:       Source documents the answer may cite.
        question:        Optional original question the answer was responding to.
                         When provided and ``config.relevance_stratified_metrics``
                         is non-empty, relevance-stratified metrics are computed
                         and added to the result (e.g. ``"citation_correctness_core"``).
        citation_to_doc: Explicit mapping from citation number to document ID.
                         When ``None``, citation ``[N]`` is mapped to
                         ``documents[N-1]`` by position (1-indexed).
        config:          :class:`~acclaim.config.load.EvalConfig` instance.
                         Defaults to :func:`~acclaim.config.load.load_config`.
        claim_extractor: Override the extractor strategy directly.
        aligner:         Override the aligner strategy directly.
        judge:           Override the judge strategy directly.

    Returns:
        An :class:`~acclaim.data_models.EvaluationResult` with per-claim
        results, metric scores, and intermediate pipeline steps.
    """
    cfg = config or load_config()

    # ── Log inputs ─────────────────────────────────────────────
    logger.info("Evaluating answer: %s", answer)
    logger.info("Documents: %d", len(documents))
    if question:
        logger.info("Question: %s", question)

    # ── Build citation_to_doc map ─────────────────────────────────────────────
    if citation_to_doc is None:
        citation_to_doc = {i + 1: doc.doc_id for i, doc in enumerate(documents)}

    doc_by_id: dict[str, Document] = {doc.doc_id: doc for doc in documents}

    # ── Stage 1: Claim extraction ─────────────────────────────────────────────
    extractor = claim_extractor or _build_extractor(cfg)
    answer_obj = Answer(text=answer)
    claims = extractor.extract(answer_obj)
    logger.info("Extracted %d claims", len(claims))

    # ── Stage 2: Citation alignment ───────────────────────────────────────────
    aligner_obj = aligner or _build_aligner(cfg, claim_extractor)
    aligned_claims = aligner_obj.align(claims, answer, citation_to_doc)
    logger.info("Aligned %d claims to citations", len(aligned_claims))

    # ── Stage 3: Evidence judgement ───────────────────────────────────────────
    judge_obj = judge or _build_judge(cfg)
    effective_max_workers = (
        _inner_max_workers if _inner_max_workers is not None else cfg.concurrency.max_workers
    )

    def _judge_one(aligned: AlignedClaim) -> ClaimResult:
        docs_for_claim = [
            doc_by_id[did] for did in aligned.citation_doc_ids if did in doc_by_id
        ]
        support = judge_obj.evaluate(aligned.claim, docs_for_claim)
        return ClaimResult(
            claim=aligned.claim,
            citation_doc_ids=aligned.citation_doc_ids,
            support=support,
        )

    claim_results = parallel_map(_judge_one, aligned_claims, effective_max_workers)

    logger.info("Judged %d claims", len(claim_results))

    # ── Stage 4: Metrics ──────────────────────────────────────────────────────
    valid_doc_ids = list(doc_by_id.keys())
    metric_overrides = _build_metric_overrides(cfg, effective_max_workers)
    metrics = compute_metrics(
        cfg.metrics,
        claim_results,
        valid_doc_ids,
        documents=documents,
        metric_overrides=metric_overrides,
        answer_text=answer,
        question=question,
    )

    # ── Assemble result ───────────────────────────────────────────────────────
    metadata = (
        EvaluationMetadata(version=__version__, config=config_to_dict(cfg))
        if _include_metadata
        else None
    )

    return EvaluationResult(
        claims=claim_results,
        metrics=metrics,
        steps={
            "claims": claims,
            "aligned_claims": aligned_claims,
            "claim_results": claim_results,
            "citation_to_doc": citation_to_doc,
        },
        metadata=metadata,
        question=question,
        answer=answer,
        documents=documents,
    )


# ---------------------------------------------------------------------------
# Internal factory helpers
# ---------------------------------------------------------------------------


def _build_extractor(cfg: EvalConfig) -> ClaimExtractor:
    key = cfg.claim_extractor
    if key not in _CLAIM_EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown claim_extractor {key!r}. Available: {list(_CLAIM_EXTRACTOR_REGISTRY)}"
        )
    cls = _CLAIM_EXTRACTOR_REGISTRY[key]

    if key == "atomic":
        # Connection/sampling fields fall back to the main judge so users
        # only need to configure one LLM endpoint; max_tokens/system_prompt
        # are extractor-specific and never inherited from cfg.judge.
        ac = cfg.atomic_claim_extractor
        j = cfg.judge
        return AtomicClaimExtractor(
            model=ac.model if ac.model is not None else j.model,
            api_base=ac.api_base if ac.api_base is not None else j.api_base,
            api_key=ac.api_key if ac.api_key is not None else j.api_key,
            temperature=ac.temperature if ac.temperature is not None else j.temperature,
            max_tokens=ac.max_tokens,
            max_retries=ac.max_retries if ac.max_retries is not None else j.max_retries,
            thinking=ac.thinking if ac.thinking is not None else j.thinking,
            system_prompt=ac.system_prompt,
        )

    return cls()


def _build_aligner(
    cfg: EvalConfig,
    explicit_extractor: ClaimExtractor | None,
) -> CitationAligner:
    # Use extractor type to pick the sensible default aligner
    key = cfg.claim_extractor
    aligner_cls = _DEFAULT_ALIGNER.get(key, SentenceCitationAligner)
    if aligner_cls is JaccardCitationAligner:
        return JaccardCitationAligner(weighted=cfg.jaccard_aligner.weighted)
    return aligner_cls()


def _build_judge(cfg: EvalConfig) -> EvidenceJudge:
    j = cfg.judge
    return LiteLLMJudge(
        model=j.model,
        api_base=j.api_base,
        api_key=j.api_key,
        temperature=j.temperature,
        max_tokens=j.max_tokens,
        max_retries=j.max_retries,
        thinking=j.thinking,
        system_prompt=j.system_prompt,
    )


def _build_precision_judge(cfg: EvalConfig) -> PrecisionJudge:
    """Build a precision judge, using the dedicated config if provided."""
    j = cfg.citation_precision_judge or cfg.judge
    return LiteLLMPrecisionJudge(
        model=j.model,
        api_base=j.api_base,
        api_key=j.api_key,
        temperature=j.temperature,
        max_tokens=j.max_tokens,
        max_retries=j.max_retries,
        thinking=j.thinking,
    )


def _build_recall_judge(cfg: EvalConfig) -> RecallJudge:
    """Build a recall judge, using the dedicated config if provided."""
    j = cfg.citation_recall_judge or cfg.judge
    return LiteLLMRecallJudge(
        model=j.model,
        api_base=j.api_base,
        api_key=j.api_key,
        temperature=j.temperature,
        max_tokens=j.max_tokens,
        max_retries=j.max_retries,
        thinking=j.thinking,
    )


def _build_relevance_judge(cfg: EvalConfig) -> RelevanceJudge:
    """Build a relevance judge, using the dedicated config if provided."""
    j = cfg.relevance_judge or cfg.judge
    return LiteLLMRelevanceJudge(
        model=j.model,
        api_base=j.api_base,
        api_key=j.api_key,
        temperature=j.temperature,
        max_tokens=j.max_tokens,
        max_retries=j.max_retries,
        thinking=j.thinking,
    )


def _build_metric_overrides(cfg: EvalConfig, max_workers: int) -> dict[str, Metric]:
    """
    Build per-run metric instances that require runtime configuration.

    LLM-backed metrics (:class:`~acclaim.metrics.citation_recall.CitationRecallMetric`,
    :class:`~acclaim.metrics.citation_precision.CitationPrecisionMetric`, and
    :class:`~acclaim.metrics.citation_f1.CitationF1Metric`) hold judges that must
    be wired to the configured endpoint.
    """
    overrides: dict[str, Metric] = {}

    needs_precision = "citation_precision" in cfg.metrics or "citation_f1" in cfg.metrics
    needs_recall = "citation_recall" in cfg.metrics or "citation_f1" in cfg.metrics

    precision_metric = (
        CitationPrecisionMetric(judge=_build_precision_judge(cfg), max_workers=max_workers)
        if needs_precision
        else None
    )
    recall_metric = (
        CitationRecallMetric(judge=_build_recall_judge(cfg), max_workers=max_workers)
        if needs_recall
        else None
    )

    if "citation_precision" in cfg.metrics and precision_metric:
        overrides["citation_precision"] = precision_metric
    if "citation_recall" in cfg.metrics and recall_metric:
        overrides["citation_recall"] = recall_metric
    if "citation_f1" in cfg.metrics and precision_metric and recall_metric:
        overrides["citation_f1"] = CitationF1Metric(
            precision=precision_metric,
            recall=recall_metric,
        )

    if cfg.relevance_stratified_metrics:
        relevance_judge = _build_relevance_judge(cfg)
        for name in cfg.relevance_stratified_metrics:
            base = overrides.get(name) or METRIC_REGISTRY[name]
            overrides[name] = RelevanceStratifiedMetric(
                base_metric=base,
                relevance_judge=relevance_judge,
                max_workers=max_workers,
            )

    return overrides
