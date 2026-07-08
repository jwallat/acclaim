"""Claim extraction strategy base class and registry."""

from .base import ClaimExtractor
from .sentence import SentenceClaimExtractor
from .atomic import AtomicClaimExtractor

__all__ = ["ClaimExtractor", "SentenceClaimExtractor", "AtomicClaimExtractor"]
