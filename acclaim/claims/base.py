"""Abstract base class for claim extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models import Answer, Claim


class ClaimExtractor(ABC):
    """Extract a list of :class:`~acclaim.data_models.Claim` objects from an answer."""

    @abstractmethod
    def extract(self, answer: Answer) -> list[Claim]:
        """
        Extract claims from *answer*.

        Args:
            answer: The LLM-generated answer to extract claims from.

        Returns:
            A list of :class:`~acclaim.data_models.Claim` objects.
        """
