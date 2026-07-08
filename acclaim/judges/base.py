"""Abstract base class for evidence judges."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models import Claim, Document, SupportResult


class EvidenceJudge(ABC):
    """Determine whether a claim is supported by a set of documents."""

    @abstractmethod
    def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult:
        """
        Judge whether *docs* support *claim*.

        Args:
            claim: The claim to verify.
            docs:  Source documents cited for this claim.

        Returns:
            A :class:`~acclaim.data_models.SupportResult`.
        """
