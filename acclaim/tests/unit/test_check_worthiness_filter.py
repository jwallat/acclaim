"""
Unit tests for the LLMCheckWorthinessFilter (all LLM calls mocked).
"""

import json
from unittest.mock import patch

import pytest

from acclaim.data_models import Claim
from acclaim.filters.check_worthiness import LLMCheckWorthinessFilter


ANSWER = "Great question! Paris is in France [1]. The Eiffel Tower stands 330 m [2]."
CLAIMS = [
    Claim(text="This is a great question.", span=(-1, -1)),
    Claim(text="Paris is in France.", span=(-1, -1)),
    Claim(text="The Eiffel Tower stands 330 m.", span=(-1, -1)),
]


def _make_filter(**kwargs) -> LLMCheckWorthinessFilter:
    return LLMCheckWorthinessFilter(model="gpt-4o-mini", **kwargs)


def _response(*pairs: tuple[int, str]) -> str:
    """Build a mock classification response from (claim_index, category) pairs."""
    return json.dumps(
        {
            "decisions": [
                {"claim_index": idx, "category": category, "reason": f"reason {idx}"}
                for idx, category in pairs
            ]
        }
    )


DEFAULT_RESPONSE = _response((0, "NOT_A_CLAIM"), (1, "COMMON_KNOWLEDGE"), (2, "QUANTITY"))


def test_drops_default_categories_and_preserves_order():
    f = _make_filter()
    with patch.object(f.client, "call", return_value=DEFAULT_RESPONSE):
        decisions = f.decide(CLAIMS, ANSWER)
    assert [d.keep for d in decisions] == [False, False, True]
    assert [d.claim for d in decisions] == CLAIMS
    assert [d.category for d in decisions] == ["NOT_A_CLAIM", "COMMON_KNOWLEDGE", "QUANTITY"]
    assert decisions[2].reason == "reason 2"
    assert all(d.filter_name == "check_worthiness" for d in decisions)


def test_filter_returns_kept_claim_objects_unchanged():
    f = _make_filter()
    with patch.object(f.client, "call", return_value=DEFAULT_RESPONSE):
        kept = f.filter(CLAIMS, ANSWER)
    assert kept == [CLAIMS[2]]
    assert kept[0] is CLAIMS[2]


def test_custom_drop_categories():
    f = _make_filter(drop_categories=["not_a_claim"])  # case-insensitive
    with patch.object(f.client, "call", return_value=DEFAULT_RESPONSE):
        kept = f.filter(CLAIMS, ANSWER)
    assert kept == CLAIMS[1:]


def test_unknown_drop_category_raises():
    with pytest.raises(ValueError, match="Unknown claim category"):
        _make_filter(drop_categories=["NOT_A_REAL_CATEGORY"])


def test_invalid_category_triggers_retry():
    f = _make_filter()
    bad = _response((0, "BOGUS"), (1, "COMMON_KNOWLEDGE"), (2, "QUANTITY"))
    with patch.object(f.client, "call", side_effect=[bad, DEFAULT_RESPONSE]) as mock_call:
        kept = f.filter(CLAIMS, ANSWER)
    assert mock_call.call_count == 2
    assert kept == [CLAIMS[2]]


def test_missing_index_triggers_retry():
    f = _make_filter()
    partial = _response((0, "NOT_A_CLAIM"), (2, "QUANTITY"))
    with patch.object(f.client, "call", side_effect=[partial, DEFAULT_RESPONSE]) as mock_call:
        kept = f.filter(CLAIMS, ANSWER)
    assert mock_call.call_count == 2
    assert kept == [CLAIMS[2]]


def test_fails_open_after_all_retries():
    f = _make_filter(max_retries=2)
    with patch.object(f.client, "call", return_value="not json") as mock_call:
        decisions = f.decide(CLAIMS, ANSWER)
    assert mock_call.call_count == 2
    assert all(d.keep for d in decisions)
    assert all(d.category is None for d in decisions)
    assert [d.claim for d in decisions] == CLAIMS


def test_empty_claims_makes_no_call():
    f = _make_filter()
    with patch.object(f.client, "call") as mock_call:
        assert f.decide([], ANSWER) == []
    mock_call.assert_not_called()


def test_question_included_in_user_message():
    f = _make_filter()
    with patch.object(f.client, "call", return_value=DEFAULT_RESPONSE) as mock_call:
        f.decide(CLAIMS, ANSWER, question="How tall is the Eiffel Tower?")
    user_msg = mock_call.call_args[0][0][1]["content"]
    assert "How tall is the Eiffel Tower?" in user_msg
    assert ANSWER in user_msg
    assert "2: The Eiffel Tower stands 330 m." in user_msg


def test_question_omitted_when_not_given():
    f = _make_filter()
    with patch.object(f.client, "call", return_value=DEFAULT_RESPONSE) as mock_call:
        f.decide(CLAIMS, ANSWER)
    user_msg = mock_call.call_args[0][0][1]["content"]
    assert "Question:" not in user_msg


def test_custom_system_prompt():
    f = _make_filter(system_prompt="custom prompt")
    with patch.object(f.client, "call", return_value=DEFAULT_RESPONSE) as mock_call:
        f.decide(CLAIMS, ANSWER)
    assert mock_call.call_args[0][0][0]["content"] == "custom prompt"
