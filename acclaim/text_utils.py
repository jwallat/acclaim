"""Shared text processing utilities."""

from __future__ import annotations

import string
from typing import NamedTuple

import pysbd

from .citations.parser import CITATION_MARKER_RE as _CITATION_RE

_SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)


class SentenceSpan(NamedTuple):
    """A sentence together with its ``(start, end)`` span in the source text."""

    text: str
    start: int
    end: int


def split_sentences_with_spans(text: str) -> list[SentenceSpan]:
    """
    Split *text* into sentences, each carrying its character span in the
    original text.

    Uses pysbd (rather than naive punctuation-regex splitting) so that
    abbreviations like ``"U.S."``, ``"No. 1"``, or ``"George H. W. Bush"``
    are not mistaken for sentence boundaries. This is the canonical
    sentence-splitting implementation used across the library — by
    :class:`~acclaim.claims.sentence.SentenceClaimExtractor`,
    :class:`~acclaim.alignment.jaccard.JaccardCitationAligner`, and
    :func:`split_sentences` below.

    Args:
        text: Text to split.

    Returns:
        List of :class:`SentenceSpan`, in order. Sentence text retains any
        trailing whitespace pysbd includes within the span.
    """
    if not text:
        return []
    return [SentenceSpan(s.sent, s.start, s.end) for s in _SEGMENTER.segment(text)]


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on punctuation boundaries.

    Args:
        text: Text to split.

    Returns:
        List of sentences (empty sentences are filtered out).
    """
    sentences = []
    for sent in split_sentences_with_spans(text):
        stripped = sent.text.strip()
        if stripped:
            sentences.append(stripped)
    return sentences


def split_sentences_preserve_citations(text: str) -> list[str]:
    """
    Split *text* into sentences, then stitch any sentence that starts with
    bare ``[N]`` markers back onto the preceding sentence.

    This handles the common pattern where punctuation splitting breaks
    ``"...France. [1] The Eiffel..."`` into ``["...France.", "[1] The
    Eiffel..."]`` — the ``[1]`` belongs to ``"France"``, not the next
    sentence.
    """
    sentences = split_sentences(text)
    result: list[str] = []
    for sent in sentences:
        if result and _CITATION_RE.match(sent):
            leading: list[str] = []
            rest = sent
            while True:
                m = _CITATION_RE.match(rest)
                if not m:
                    break
                leading.append(m.group())
                rest = rest[m.end():].lstrip()
            result[-1] = result[-1] + " " + " ".join(leading)
            if rest:
                result.append(rest)
        else:
            result.append(sent)
    return result


def tokenize_text(text: str) -> set[str]:
    """
    Tokenize text by whitespace and lowercase, stripping punctuation.

    Args:
        text: Text to tokenize.

    Returns:
        Set of lowercased tokens with punctuation removed.
    """
    # Remove punctuation and lowercase
    cleaned = text.translate(str.maketrans("", "", string.punctuation)).lower()
    tokens = cleaned.split()
    return set(tokens)


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """
    Compute Jaccard similarity between two token sets.

    Formula: ``|intersection| / |union|``

    Args:
        set1: First set of tokens.
        set2: Second set of tokens.

    Returns:
        Similarity score in [0, 1]. Returns 0.0 if both sets are empty.
    """
    if not set1 and not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union
