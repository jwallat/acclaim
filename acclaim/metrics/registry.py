"""Metric registry and ``compute_metrics`` helper."""

from __future__ import annotations

from ..data_models import ClaimResult, Document
from .base import Metric
from .stratified import StratifiedMetric
from .citation_f1 import CitationF1Metric
from .citation_precision import CitationPrecisionMetric
from .citation_recall import CitationRecallMetric
from .correctness import CitationCorrectness
from .citation_lengths import CitationLengths
from .citation_number import CitationNumber
from .coverage import CitationCoverage
from .cvcp import CVCPMetric
from .hallucination import HallucinationRate
from .supported_claim_rate import SupportedClaimRate
from .token_overlap import TokenOverlap
from ..judges.citation_precision import LiteLLMPrecisionJudge
from ..judges.citation_recall import LiteLLMRecallJudge

_default_precision = CitationPrecisionMetric(judge=LiteLLMPrecisionJudge())
_default_recall = CitationRecallMetric(judge=LiteLLMRecallJudge())

METRIC_REGISTRY: dict[str, Metric] = {
    m.name: m
    for m in [
        _default_precision,
        _default_recall,
        CitationF1Metric(precision=_default_precision, recall=_default_recall),
        CitationCorrectness(),
        CitationLengths(),
        CitationNumber(),
        CitationCoverage(),
        CVCPMetric(),
        HallucinationRate(),
        SupportedClaimRate(),
        TokenOverlap(),
    ]
}


def compute_metrics(
    metric_names: list[str],
    claim_results: list[ClaimResult],
    valid_doc_ids: list[str] | None = None,
    documents: list[Document] | None = None,
    metric_overrides: dict[str, Metric] | None = None,
    answer_text: str | None = None,
    question: str | None = None,
) -> dict[str, float]:
    """
    Compute each named metric and return a ``{name: score}`` dict.

    Args:
        metric_names:     Names of metrics to compute (must be in
                          ``METRIC_REGISTRY`` or *metric_overrides*).
        claim_results:    Per-claim evaluation results.
        valid_doc_ids:    Optional set of valid document IDs for hallucination
                          detection.
        documents:        Optional list of Document objects needed by metrics
                          that compare claim text to document text.
        metric_overrides: Optional mapping of metric name → :class:`Metric`
                          instance.  Entries here take precedence over
                          ``METRIC_REGISTRY``, allowing callers to inject
                          metrics that require runtime configuration (e.g. a
                          :class:`~acclaim.metrics.citation_recall.
                          CitationRecallMetric` wired to the pipeline's own
                          LLM endpoint).
        answer_text:      Raw answer string forwarded to every metric.  Metrics
                          that operate on the answer text directly (e.g.
                          :class:`~acclaim.metrics.citation_number.CitationNumber`)
                          use this; others ignore it.
        question:         Original question the answer was responding to.
                          Forwarded to :class:`~acclaim.metrics.stratified.StratifiedMetric`
                          instances, which use it to classify claims by relevance.
                          Ignored by all other metrics.

    Raises:
        KeyError: If a requested metric name is not in the registry or overrides.
    """
    overrides = metric_overrides or {}
    results: dict[str, float] = {}
    for name in metric_names:
        if name in overrides:
            metric_instance = overrides[name]
        elif name in METRIC_REGISTRY:
            metric_instance = METRIC_REGISTRY[name]
        else:
            raise KeyError(
                f"Unknown metric {name!r}. Available: {list(METRIC_REGISTRY)}"
            )
        if isinstance(metric_instance, StratifiedMetric) and question is not None:
            stratum_scores = metric_instance.compute_stratified(
                claim_results, question, valid_doc_ids, documents, answer_text
            )
            results.update(stratum_scores)
        else:
            results[name] = metric_instance.compute(
                claim_results, valid_doc_ids, documents, answer_text, question
            )
    return results
