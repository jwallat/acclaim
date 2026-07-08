"""
Unit tests for citation marker parsing in citations/parser.py.
"""

from acclaim.citations.parser import (
    extract_doc_ids_for_span,
    find_citation_markers,
    parse_citation_markers,
)


def test_find_citation_markers_single():
    markers = find_citation_markers("Paris [1] is in France [2][1].")
    assert markers == [
        (6, 9, [1]),
        (23, 26, [2]),
        (26, 29, [1]),
    ]


def test_find_citation_markers_comma_separated():
    markers = find_citation_markers("Some text [2, 3] more text.")
    assert markers == [(10, 16, [2, 3])]


def test_find_citation_markers_comma_separated_no_space():
    markers = find_citation_markers("Some text [2,3] more text.")
    assert markers == [(10, 15, [2, 3])]


def test_find_citation_markers_restricted_to_span():
    text = "Paris [1] is in France [2][1]."
    markers = find_citation_markers(text, 10, len(text))
    assert markers == [(23, 26, [2]), (26, 29, [1])]


def test_single_markers():
    text = "Paris [1] is in France [2][1]."
    markers = parse_citation_markers(text)
    assert markers == {1: [6, 26], 2: [23]}


def test_comma_separated_marker():
    markers = parse_citation_markers("Some text [2, 3] more text.")
    assert markers == {2: [10], 3: [10]}


def test_comma_separated_marker_no_space():
    markers = parse_citation_markers("Some text [2,3] more text.")
    assert markers == {2: [10], 3: [10]}


def test_mixed_grouped_and_consecutive_markers():
    markers = parse_citation_markers("A [1][2, 3] B.")
    assert markers == {1: [2], 2: [5], 3: [5]}


def test_extract_doc_ids_for_span_comma_separated():
    text = "Brigitte is 25 years his senior [1, 3]."
    citation_to_doc = {1: "doc_1", 2: "doc_2", 3: "doc_3"}
    doc_ids = extract_doc_ids_for_span((0, len(text)), text, citation_to_doc)
    assert doc_ids == ["doc_1", "doc_3"]


def test_extract_doc_ids_for_span_unknown_number_ignored():
    text = "Claim [1, 9]."
    citation_to_doc = {1: "doc_1"}
    doc_ids = extract_doc_ids_for_span((0, len(text)), text, citation_to_doc)
    assert doc_ids == ["doc_1"]
