"""
Sentence-level citation aligner.

For sentence-level claims the span is known, so alignment is simple:
scan the claim's character span in the original text for ``[N]`` markers
and map them to document IDs via the supplied ``citation_to_doc`` dict.

No heuristics, no similarity scoring — just direct marker extraction.
"""

from __future__ import annotations

from .base import CitationAligner
from ..citations.parser import extract_doc_ids_for_span
from ..data_models import AlignedClaim, Claim


class SentenceCitationAligner(CitationAligner):
    """
    Align sentence-level claims by extracting ``[N]`` markers from each
    claim's character span in the original answer text.

    This is the default aligner for :class:`~acclaim.claims.sentence.SentenceClaimExtractor`.
    It requires claims to have valid (non ``-1``) spans.
    """

    def align(
        self,
        claims: list[Claim],
        text: str,
        citation_to_doc: dict[int, str],
    ) -> list[AlignedClaim]:
        """
        Align *claims* by scanning for citation markers within each claim span.

        Args:
            claims:          Sentence-level claims with valid character spans.
            text:            Full original answer text (with ``[N]`` markers).
            citation_to_doc: Mapping from citation number to document ID.

        Returns:
            One :class:`~acclaim.data_models.AlignedClaim` per claim.
        """
        aligned: list[AlignedClaim] = []
        for claim in claims:
            if claim.span == (-1, -1):
                # Sentence aligner cannot handle span-less claims
                aligned.append(AlignedClaim(claim=claim, citation_doc_ids=[]))
                continue

            doc_ids = extract_doc_ids_for_span(claim.span, text, citation_to_doc)
            aligned.append(AlignedClaim(claim=claim, citation_doc_ids=doc_ids))

        return aligned
