"""
StratifiedMetric — abstract base class for metrics that produce per-stratum scores.

Concrete subclasses classify claims into relevance strata (e.g. CORE /
COMPLEMENTARY / IRRELEVANT) and run a wrapped base metric on each subset,
returning a ``{metric_name_stratum: score}`` dict instead of a single float.

``compute_metrics()`` in :mod:`~acclaim.metrics.registry` detects
:class:`StratifiedMetric` instances and merges all stratum scores into the
result dict, so stratified metrics integrate transparently with the existing
pipeline.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict

from ..concurrency import parallel_map
from ..data_models import ClaimResult, Document, RelevanceLabel
from ..judges.relevance import RelevanceJudge
from .base import Metric


class StratifiedMetric(Metric):
    """
    Abstract base for metrics that stratify claims before scoring.

    Subclasses must implement :meth:`compute_stratified`, which returns a
    ``{key: score}`` dict (e.g. ``{"citation_correctness_core": 0.9}``).

    :meth:`compute` is provided as a no-op fallback (returns ``0.0``) so
    that ``StratifiedMetric`` instances satisfy the ``Metric`` interface when
    called without a question.  In normal pipeline use ``compute_metrics()``
    routes to :meth:`compute_stratified` instead.
    """

    @abstractmethod
    def compute_stratified(
        self,
        claim_results: list[ClaimResult],
        question: str,
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
    ) -> dict[str, float]:
        """
        Compute per-stratum metric scores.

        Args:
            claim_results: Per-claim evaluation results.
            question:      Original question the answer was responding to.
            valid_doc_ids: Forwarded to the wrapped base metric.
            documents:     Forwarded to the wrapped base metric.
            answer_text:   Forwarded to the wrapped base metric.

        Returns:
            Dict mapping ``"{base_metric_name}_{stratum}"`` to a scalar score.
            Strata with no claims are omitted.
        """

    # ------------------------------------------------------------------
    # Metric interface — fallback when no question is available
    # ------------------------------------------------------------------

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        """
        Fallback: returns ``0.0``.

        ``compute_metrics()`` routes :class:`StratifiedMetric` instances to
        :meth:`compute_stratified` when a question is available, so this path
        is only hit when the metric is called directly without a question.
        """
        return 0.0


class RelevanceStratifiedMetric(StratifiedMetric):
    """
    Wraps any :class:`~acclaim.metrics.base.Metric` and runs it
    separately on each relevance stratum (CORE / COMPLEMENTARY / IRRELEVANT).

    Claims are classified by a :class:`~acclaim.judges.relevance.RelevanceJudge`
    on every call to :meth:`compute_stratified`.  Strata with no claims are
    omitted from the result.

    Args:
        base_metric:     The metric to apply within each stratum.
        relevance_judge: Judge used to classify each claim's relevance to the question.

    Example output keys (when base metric is ``citation_correctness``)::

        {
            "citation_correctness_core": 0.9,
            "citation_correctness_complementary": 0.5,
            "citation_correctness_irrelevant": 1.0,
        }
    """

    def __init__(
        self,
        base_metric: Metric,
        relevance_judge: RelevanceJudge,
        max_workers: int = 1,
    ) -> None:
        self._base_metric = base_metric
        self._relevance_judge = relevance_judge
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        return f"{self._base_metric.name}_stratified"

    def compute_stratified(
        self,
        claim_results: list[ClaimResult],
        question: str,
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
    ) -> dict[str, float]:
        # Classify each claim and group by label
        labels = parallel_map(
            lambda cr: self._relevance_judge.evaluate(cr.claim, question),
            claim_results,
            self._max_workers,
        )
        strata: dict[RelevanceLabel, list[ClaimResult]] = defaultdict(list)
        for cr, result in zip(claim_results, labels):
            strata[result.label].append(cr)

        # Always include the un-stratified base metric so downstream code
        # doesn't have to special-case the presence of stratification.
        scores: dict[str, float] = {
            self._base_metric.name: self._base_metric.compute(
                claim_results, valid_doc_ids, documents, answer_text
            )
        }
        for label, subset in strata.items():
            key = f"{self._base_metric.name}_{label.value.lower()}"
            scores[key] = self._base_metric.compute(
                subset, valid_doc_ids, documents, answer_text
            )
        return scores
