"""Metric implementations and registry."""

from .base import Metric
from .citation_f1 import CitationF1Metric
from .citation_precision import CitationPrecisionMetric
from .citation_recall import CitationRecallMetric
from .citation_number import CitationNumber
from .correctness import CitationCorrectness
from .coverage import CitationCoverage
from .hallucination import HallucinationRate
from .supported_claim_rate import SupportedClaimRate
from .registry import METRIC_REGISTRY, compute_metrics

__all__ = [
    "Metric",
    "CitationF1Metric",
    "CitationPrecisionMetric",
    "CitationRecallMetric",
    "CitationNumber",
    "CitationCorrectness",
    "CitationCoverage",
    "HallucinationRate",
    "SupportedClaimRate",
    "METRIC_REGISTRY",
    "compute_metrics",
]
