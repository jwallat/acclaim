"""Batch evaluation helpers."""

from __future__ import annotations

import logging
from typing import Any

from ._version import __version__
from .alignment.base import CitationAligner
from .claims.base import ClaimExtractor
from .concurrency import parallel_map
from .config.load import EvalConfig, config_to_dict, load_config
from .data_models import (
    BatchEvaluationResult,
    BatchExample,
    EvaluationMetadata,
    EvaluationResult,
)
from .evaluate import evaluate
from .filters.base import ClaimFilter
from .judges.base import EvidenceJudge

logger = logging.getLogger(__name__)


def evaluate_batch(
    examples: list[BatchExample],
    *,
    config: EvalConfig | None = None,
    claim_extractor: ClaimExtractor | None = None,
    claim_filters: list[ClaimFilter] | None = None,
    aligner: CitationAligner | None = None,
    judge: EvidenceJudge | None = None,
) -> BatchEvaluationResult:
    """
    Evaluate a dataset by running :func:`acclaim.evaluate.evaluate` per item.

    Metrics are aggregated with an arithmetic mean over examples. When claim
    filters ran, their decisions are summarised in
    :attr:`~acclaim.data_models.BatchEvaluationResult.filter_summary`.
    """
    if not examples:
        raise ValueError("examples must not be empty")

    cfg = config or load_config()
    max_workers = cfg.concurrency.max_workers
    # When batch-level parallelism is active, force inner per-claim loops to
    # run sequentially to avoid nested unbounded thread pools.
    inner_max_workers = 1 if max_workers > 1 else None

    def _eval_one(ex: BatchExample) -> EvaluationResult:
        return evaluate(
            ex.answer,
            ex.documents,
            question=ex.question,
            citation_to_doc=ex.citation_to_doc,
            config=cfg,
            claim_extractor=claim_extractor,
            claim_filters=claim_filters,
            aligner=aligner,
            judge=judge,
            _inner_max_workers=inner_max_workers,
            _include_metadata=False,
        )

    if max_workers > 1:
        logger.info(
            "Evaluating %d examples with max_workers=%d", len(examples), max_workers
        )
    items = parallel_map(_eval_one, examples, max_workers, desc="Evaluating examples")

    aggregate_metrics = _mean_metrics(items)
    metadata = EvaluationMetadata(version=__version__, config=config_to_dict(cfg))
    return BatchEvaluationResult(
        items=items,
        aggregate_metrics=aggregate_metrics,
        metadata=metadata,
        filter_summary=_filter_summary(items),
    )


def _filter_summary(items: list[EvaluationResult]) -> dict[str, Any] | None:
    """
    Summarise claim-filter decisions across *items*, or ``None`` if no
    filter ran on any item.

    A claim is dropped by at most one filter (later filters only see the
    claims earlier ones kept), so ``claims_dropped`` is a plain count of
    ``keep=False`` decisions.
    """
    if not any(item.filter_decisions for item in items):
        return None

    extracted = kept = items_with_drops = items_all_dropped = 0
    by_filter: dict[str, dict[str, Any]] = {}
    for item in items:
        n_claims = len(item.steps.get("claims", []))
        n_dropped = sum(1 for d in item.filter_decisions if not d.keep)
        extracted += n_claims
        kept += n_claims - n_dropped
        items_with_drops += n_dropped > 0
        items_all_dropped += n_claims > 0 and n_dropped == n_claims
        for d in item.filter_decisions:
            f = by_filter.setdefault(d.filter_name, {"seen": 0, "dropped": 0, "categories": {}})
            f["seen"] += 1
            f["dropped"] += not d.keep
            cat = f["categories"].setdefault(str(d.category), {"kept": 0, "dropped": 0})
            cat["kept" if d.keep else "dropped"] += 1

    dropped = extracted - kept
    return {
        "claims_extracted": extracted,
        "claims_kept": kept,
        "claims_dropped": dropped,
        "drop_rate": dropped / extracted if extracted else 0.0,
        "items_with_drops": items_with_drops,
        "items_all_dropped": items_all_dropped,
        "by_filter": by_filter,
    }


def _mean_metrics(items: list[EvaluationResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for item in items:
        for name, value in item.metrics.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    return {name: sums[name] / counts[name] for name in sums}
