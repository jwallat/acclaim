"""
Coefficient of Variation of Citation Positions (CVCP).

Reference:
    ALiiCE: Evaluating Positional Fine-grained Citation Generation,
    Section 3.3.3 (Xu et al., 2024).
"""

from __future__ import annotations

import math

from ..citations.parser import CITATION_MARKER_RE as _CITATION_RE
from ..data_models import ClaimResult, Document
from ..text_utils import split_sentences_preserve_citations
from .base import Metric


class CVCPMetric(Metric):
    """
    Coefficient of Variation of Citation Positions (CVCP).

    For each sentence in the answer that contains at least one ``[N]`` marker:

    1. Normalize each marker's character position by sentence length.
    2. Compute the population standard deviation (σ) and mean (μ) of those
       normalized positions.
    3. Accumulate the per-sentence CV: σ / μ.

    CVCP is the mean of per-sentence CVs across all sentences that have
    citations.  Higher values indicate markers are spread throughout sentences
    rather than clustered at the end.  Returns ``0.0`` when no sentences
    contain citations.

    Sentences where μ = 0 (all markers at character position 0) are skipped
    to avoid division by zero.
    """

    @property
    def name(self) -> str:
        return "cvcp"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        if not answer_text:
            return 0.0

        sentences = split_sentences_preserve_citations(answer_text)

        per_sentence_cvs: list[float] = []
        for sentence in sentences:
            positions = [
                m.start() / len(sentence)
                for m in _CITATION_RE.finditer(sentence)
            ]
            if not positions:
                continue

            mu = sum(positions) / len(positions)
            if mu == 0.0:
                # All markers at position 0 — CV undefined; skip.
                continue

            variance = sum((p - mu) ** 2 for p in positions) / len(positions)
            sigma = math.sqrt(variance)
            per_sentence_cvs.append(sigma / mu)

        if not per_sentence_cvs:
            return 0.0

        return sum(per_sentence_cvs) / len(per_sentence_cvs)
