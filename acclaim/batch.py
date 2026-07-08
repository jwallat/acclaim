"""Batch evaluation helpers."""

from __future__ import annotations

import logging

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
from .judges.base import EvidenceJudge

logger = logging.getLogger(__name__)


def evaluate_batch(
    examples: list[BatchExample],
    *,
    config: EvalConfig | None = None,
    claim_extractor: ClaimExtractor | None = None,
    aligner: CitationAligner | None = None,
    judge: EvidenceJudge | None = None,
) -> BatchEvaluationResult:
    """
    Evaluate a dataset by running :func:`acclaim.evaluate.evaluate` per item.

    Metrics are aggregated with an arithmetic mean over examples.
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
        items=items, aggregate_metrics=aggregate_metrics, metadata=metadata
    )


def _mean_metrics(items: list[EvaluationResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for item in items:
        for name, value in item.metrics.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    return {name: sums[name] / counts[name] for name in sums}
