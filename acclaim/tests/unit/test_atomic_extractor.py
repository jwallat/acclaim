"""
Unit tests for AtomicClaimExtractor (all LLM calls mocked).
"""

from unittest.mock import patch

from acclaim.claims.atomic import AtomicClaimExtractor
from acclaim.data_models import Answer


def _make_extractor(**kwargs) -> AtomicClaimExtractor:
    return AtomicClaimExtractor(model="gpt-4o-mini", **kwargs)


def test_extract_parses_json_array():
    extractor = _make_extractor()
    response = '["Claim one.", "Claim two."]'
    with patch.object(extractor.client, "call", return_value=response):
        claims = extractor.extract(Answer(text="Claim one and claim two."))
    assert [c.text for c in claims] == ["Claim one.", "Claim two."]
    assert all(c.span == (-1, -1) for c in claims)


def test_extract_strips_code_fences():
    extractor = _make_extractor()
    response = '```json\n["Claim one."]\n```'
    with patch.object(extractor.client, "call", return_value=response):
        claims = extractor.extract(Answer(text="Claim one."))
    assert [c.text for c in claims] == ["Claim one."]


def test_extract_handles_extra_prose_around_array():
    extractor = _make_extractor()
    response = 'Sure! Here is the output: ["Claim one."] Hope that helps.'
    with patch.object(extractor.client, "call", return_value=response):
        claims = extractor.extract(Answer(text="Claim one."))
    assert [c.text for c in claims] == ["Claim one."]


def test_extract_handles_dict_wrapped_list():
    extractor = _make_extractor()
    response = '{"claims": ["Claim one.", "Claim two."]}'
    with patch.object(extractor.client, "call", return_value=response):
        claims = extractor.extract(Answer(text="Claim one and claim two."))
    assert [c.text for c in claims] == ["Claim one.", "Claim two."]


def test_extract_empty_answer_returns_no_claims():
    extractor = _make_extractor()
    claims = extractor.extract(Answer(text=""))
    assert claims == []


def test_extract_returns_empty_list_on_persistent_failure():
    extractor = _make_extractor(max_retries=2)
    with patch.object(extractor.client, "call", return_value="not json at all"):
        claims = extractor.extract(Answer(text="Some answer."))
    assert claims == []


def test_extract_retries_then_succeeds():
    extractor = _make_extractor(max_retries=2)
    responses = ["not json at all", '["Claim one."]']
    with patch.object(extractor.client, "call", side_effect=responses):
        claims = extractor.extract(Answer(text="Claim one."))
    assert [c.text for c in claims] == ["Claim one."]
