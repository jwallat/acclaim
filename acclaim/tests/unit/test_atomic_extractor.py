"""
Unit tests for AtomicClaimExtractor (all LLM calls mocked).
"""

from unittest.mock import patch

import pytest

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


@pytest.mark.parametrize(
    "include_question, question, expect_question",
    [
        (False, "When was {it} built?", False),
        (True, None, False),
        (True, "When was {it} built?", True),
    ],
)
def test_question_in_prompt_only_when_enabled(include_question, question, expect_question):
    extractor = _make_extractor(include_question=include_question)
    with patch.object(extractor.client, "call", return_value='["A."]') as call:
        extractor.extract(Answer(text="In 1889.", question=question))
    prompt = call.call_args.args[0][1]["content"]
    base = extractor._USER_PROMPT.format(text="In 1889.")
    if expect_question:
        assert prompt.endswith(base)
        assert f"Question: {question}\n" in prompt
    else:
        assert prompt == base
