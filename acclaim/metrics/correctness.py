"""Citation correctness metric."""

from __future__ import annotations

from .base import Metric
from ..data_models import ClaimResult, SupportLabel


class CitationCorrectness(Metric):
    """
    Proportion of cited claims that are actually supported.

    Formula: ``supported_claims / cited_claims``

    A cited claim is one with at least one document ID in
    ``citation_doc_ids``.
    """

    @property
    def name(self) -> str:
        return "citation_correctness"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents=None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        cited = [cr for cr in claim_results if cr.citation_doc_ids]
        if not cited:
            return 0.0
        supported = sum(1 for cr in cited if cr.support.label == SupportLabel.SUPPORTED)
        return supported / len(cited)
