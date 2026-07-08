"""Hallucination rate metric."""

from __future__ import annotations

from .base import Metric
from ..data_models import ClaimResult


class HallucinationRate(Metric):
    """
    Proportion of cited document references that do not exist in *valid_doc_ids*.

    Formula: ``non_existent_citations / total_citations``

    Returns ``0.0`` when ``valid_doc_ids`` is ``None`` (nothing to check
    against) or when there are no citations at all.
    """

    @property
    def name(self) -> str:
        return "hallucination_rate"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents=None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        if valid_doc_ids is None:
            return 0.0

        valid = set(valid_doc_ids)
        total = 0
        hallucinated = 0

        for cr in claim_results:
            for doc_id in cr.citation_doc_ids:
                total += 1
                if doc_id not in valid:
                    hallucinated += 1

        return hallucinated / total if total > 0 else 0.0
