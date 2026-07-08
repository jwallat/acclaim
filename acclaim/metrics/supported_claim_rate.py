"""Supported claim rate metric."""

from __future__ import annotations

from .base import Metric
from ..data_models import ClaimResult, SupportLabel


class SupportedClaimRate(Metric):
    """
    Proportion of all claims that are supported by their cited documents.

    Formula: ``supported_claims / total_claims``

    Unlike :class:`~acclaim.metrics.correctness.CitationCorrectness`,
    uncited claims are included in the denominator, so answers that leave
    claims unsupported (whether cited-but-refuted or simply uncited) receive
    a lower score.

    Returns ``0.0`` when there are no claims.
    """

    @property
    def name(self) -> str:
        return "supported_claim_rate"

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
        supported = sum(
            1 for cr in claim_results if cr.support.label == SupportLabel.SUPPORTED
        )
        return supported / len(claim_results)
