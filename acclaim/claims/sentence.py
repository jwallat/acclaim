"""
Sentence-level claim extractor.

Splits an answer into sentences using punctuation boundaries and returns
one :class:`~acclaim.data_models.Claim` per sentence.  Citation
markers (``[N]``) are stripped from claim text but the original character
span is preserved so the aligner can find associated citations.
"""

from __future__ import annotations

import re

from .base import ClaimExtractor
from ..citations.parser import CITATION_MARKER_RE as _CITATION_RE
from ..data_models import Answer, Claim
from ..text_utils import split_sentences_with_spans


class SentenceClaimExtractor(ClaimExtractor):
    """Split an answer into sentences; each sentence is one claim."""

    def extract(self, answer: Answer) -> list[Claim]:
        """
        Extract sentence-level claims from *answer*.

        Character spans reference positions in the *original* text
        (including citation markers) so that the aligner can scan the span
        for ``[N]`` occurrences.
        """
        text = answer.text
        claims: list[Claim] = []

        for sent in split_sentences_with_spans(text):
            raw_sent = sent.text
            if not raw_sent:
                continue

            span = (sent.start, sent.end)
            clean = _CITATION_RE.sub("", raw_sent)
            clean = re.sub(r"\s+([.,!?])", r"\1", clean)  # remove space before punct
            clean = re.sub(r"\s+", " ", clean).strip()

            if clean:
                claims.append(Claim(text=clean, span=span))

        return claims
