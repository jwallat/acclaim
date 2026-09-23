"""Abstract base class for claim filters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models import Claim, FilterDecision

__all__ = ["ClaimFilter", "FilterDecision"]


class ClaimFilter(ABC):
    """
    Decide which extracted claims should proceed to alignment and judgement.

    Subclasses implement :meth:`decide`; :meth:`filter` is a convenience
    wrapper returning only the kept claims. Filters must return the input
    :class:`~acclaim.data_models.Claim` objects unchanged (aligners such as
    :class:`~acclaim.alignment.sentence.SentenceCitationAligner` rely on
    ``Claim.span``).
    """

    name: str = "filter"

    @abstractmethod
    def decide(
        self,
        claims: list[Claim],
        answer: str,
        question: str | None = None,
    ) -> list[FilterDecision]:
        """
        Decide for each claim whether to keep it.

        Args:
            claims:   Claims to filter.
            answer:   The full original answer text (including ``[N]`` markers).
            question: Optional question the answer was responding to.

        Returns:
            One :class:`FilterDecision` per input claim, in input order.
        """

    def filter(
        self,
        claims: list[Claim],
        answer: str,
        question: str | None = None,
    ) -> list[Claim]:
        """Return the subset of *claims* to keep, in their original order."""
        return [d.claim for d in self.decide(claims, answer, question) if d.keep]
