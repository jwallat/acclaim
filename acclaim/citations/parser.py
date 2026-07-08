"""
Citation marker parsing.

Parses inline [N] citation markers from LLM-generated answer text and maps
them to document IDs supplied by the caller.

Supported formats: ``[1]``, ``[2][3]`` (consecutive), and ``[2, 3]`` /
``[2,3]`` (comma-separated within one bracket) — all common styles produced
by LLMs instructed to cite sources.

``CITATION_MARKER_RE`` is the canonical regex for a citation marker and is
reused by other modules (text_utils, claims/sentence, metrics/cvcp) that
need to detect or strip markers without parsing individual numbers.
"""

from __future__ import annotations

import re
from typing import NamedTuple

CITATION_MARKER_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_NUMBER_RE = re.compile(r"\d+")


class CitationMarker(NamedTuple):
    """A single ``[N]`` / ``[N][M]`` / ``[N, M]`` marker occurrence in text."""

    start: int
    end: int
    numbers: list[int]


def find_citation_markers(text: str, start: int = 0, end: int | None = None) -> list[CitationMarker]:
    """
    Find all citation marker spans in *text* (optionally restricted to
    *text[start:end]*, matching :meth:`re.Pattern.finditer`'s ``pos``/``endpos``).

    This is the canonical marker-finding implementation, reused by
    :func:`parse_citation_markers` and :func:`extract_doc_ids_for_span`
    below, and by callers outside this module (e.g. the prototype server)
    that need marker spans rather than just citation-number positions.
    """
    end = len(text) if end is None else end
    return [
        CitationMarker(m.start(), m.end(), [int(n) for n in _NUMBER_RE.findall(m.group())])
        for m in CITATION_MARKER_RE.finditer(text, start, end)
    ]


def parse_citation_markers(text: str) -> dict[int, list[int]]:
    """
    Find all ``[N]`` citation markers in *text* and return their positions.

    Returns a dict mapping each citation number to the list of character
    positions where its enclosing bracket group appears in *text*. A
    comma-separated group like ``[2, 3]`` contributes the same position for
    both 2 and 3.

    Example::

        parse_citation_markers("Paris [1] is in France [2][1].")
        # {1: [6, 24], 2: [20]}
    """
    result: dict[int, list[int]] = {}
    for marker in find_citation_markers(text):
        for n in marker.numbers:
            result.setdefault(n, []).append(marker.start)
    return result


def extract_doc_ids_for_span(
    span: tuple[int, int],
    text: str,
    citation_to_doc: dict[int, str],
) -> list[str]:
    """
    Return the document IDs cited within *span* of *text*.

    Scans *text[span[0]:span[1]]* for ``[N]`` markers and maps each found
    citation number to a document ID via *citation_to_doc*.

    Args:
        span:            (start, end) character offsets in *text*.
        text:            The full answer text.
        citation_to_doc: Mapping from citation number to document ID.

    Returns:
        Deduplicated list of document IDs found in the span (in order of
        first appearance).
    """
    start, end = span
    seen: dict[str, None] = {}  # ordered set via dict keys
    for marker in find_citation_markers(text, start, end):
        for n in marker.numbers:
            if n in citation_to_doc:
                seen[citation_to_doc[n]] = None
    return list(seen)
