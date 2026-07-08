"""Citation number metric."""

from __future__ import annotations

from .base import Metric
from ..citations.parser import parse_citation_markers
from ..data_models import ClaimResult, Document


class CitationNumber(Metric):
    """
    Number of inline citation markers in a response.

    Counts markers like ``[1]`` and ``[2]`` directly in the raw answer text
    when available. The score can be any non-negative float and is often > 1.

    If raw answer text is unavailable, a fallback estimate is used based on the
    number of aligned citation document IDs across claims.
    """

    @property
    def name(self) -> str:
        return "citation_number"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        if answer_text is not None:
            markers = parse_citation_markers(answer_text)
            return float(sum(len(positions) for positions in markers.values()))

        # Fallback for standalone metric usage without raw answer context.
        return float(sum(len(cr.citation_doc_ids) for cr in claim_results))
