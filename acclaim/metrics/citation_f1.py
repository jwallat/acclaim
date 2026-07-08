"""
Citation F1 metric.

Computes the harmonic mean of citation precision and citation recall::

    F1 = 2 * P * R / (P + R)

where P is the score from a :class:`~acclaim.metrics.base.Metric`
implementing citation precision and R is the score from a
:class:`~acclaim.metrics.base.Metric` implementing citation recall.
Both ``P`` and ``R`` are computed by delegating to their respective
:meth:`~acclaim.metrics.base.Metric.compute` methods with the same
arguments, so any compatible metric implementations can be injected.
"""

from __future__ import annotations

from ..data_models import ClaimResult, Document
from .base import Metric


class CitationF1Metric(Metric):
    """
    Harmonic mean of citation precision and citation recall.

    Args:
        precision: A :class:`~acclaim.metrics.base.Metric` instance
                   that computes citation precision (P).
        recall:    A :class:`~acclaim.metrics.base.Metric` instance
                   that computes citation recall (R).

    Example::

        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.metrics.citation_recall import CitationRecallMetric
        from acclaim.judges.citation_recall import LiteLLMRecallJudge

        f1 = CitationF1Metric(
            precision=CitationPrecisionMetric(),
            recall=CitationRecallMetric(judge=LiteLLMRecallJudge()),
        )
        score = f1.compute(claim_results, documents=documents)
    """

    def __init__(self, precision: Metric, recall: Metric) -> None:
        self._precision = precision
        self._recall = recall

    @property
    def name(self) -> str:
        """Metric registry key."""
        return "citation_f1"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        """
        Compute F1 = 2PR / (P + R).

        Returns 0.0 when both P and R are zero (avoids division by zero).

        Args:
            claim_results: Per-claim evaluation results.
            valid_doc_ids: Forwarded to both component metrics.
            documents:     Forwarded to both component metrics.

        Returns:
            Citation F1 score in ``[0.0, 1.0]``.
        """
        p = self._precision.compute(claim_results, valid_doc_ids, documents, answer_text, question)
        r = self._recall.compute(claim_results, valid_doc_ids, documents, answer_text, question)

        denom = p + r
        if denom == 0.0:
            return 0.0
        return 2.0 * p * r / denom
