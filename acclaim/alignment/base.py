"""Abstract base class for citation aligners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models import AlignedClaim, Claim


class CitationAligner(ABC):
    """Map a list of claims to their cited document IDs."""

    @abstractmethod
    def align(
        self,
        claims: list[Claim],
        text: str,
        citation_to_doc: dict[int, str],
    ) -> list[AlignedClaim]:
        """
        Align *claims* to document citations.

        Args:
            claims:          Claims extracted from the answer.
            text:            The full original answer text (including ``[N]`` markers).
            citation_to_doc: Mapping from citation number to document ID.

        Returns:
            One :class:`~acclaim.data_models.AlignedClaim` per input claim.
        """
