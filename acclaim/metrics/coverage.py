"""Citation coverage metric."""

from __future__ import annotations

from .base import Metric
from ..data_models import ClaimResult


class CitationCoverage(Metric):
    """
    Proportion of claims that have at least one citation.

    Formula: ``cited_claims / total_claims``
    """

    @property
    def name(self) -> str:
        return "coverage"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents=None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        if not claim_results:
            return 0.0
        cited = sum(1 for cr in claim_results if cr.citation_doc_ids)
        return cited / len(claim_results)
