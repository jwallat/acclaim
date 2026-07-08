"""Abstract base class for evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models import ClaimResult, Document


class Metric(ABC):
    """Compute a scalar score over a list of :class:`~acclaim.data_models.ClaimResult` objects."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier used as the key in ``EvaluationResult.metrics``."""

    @abstractmethod
    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        """
        Compute the metric score.

        Args:
            claim_results: Per-claim evaluation results.
            valid_doc_ids: Optional list of document IDs that actually exist.
                           Used by metrics that detect hallucinated citations.
            documents:     Optional list of Document objects needed by metrics that
                           compare claim text to document text.
            answer_text:   Raw answer string (used by marker-based metrics).
            question:      Original question the answer was responding to.
                           Used by relevance-stratified metrics; ignored by others.

        Returns:
            Scalar score in [0, 1].
        """
