"""
Unit tests for the LiteLLMJudge (all LLM calls mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from acclaim.data_models import Claim, Document, SupportLabel
from acclaim.judges.litellm import LiteLLMJudge


_CLAIM = Claim(text="The Eiffel Tower is in Paris.", span=(-1, -1))
_DOC = Document(doc_id="doc-1", text="The Eiffel Tower is located in Paris, France.")


def _make_judge(**kwargs) -> LiteLLMJudge:
    return LiteLLMJudge(model="gpt-4o-mini", **kwargs)


def _mock_response(
    label: str, confidence: float = 0.9, reason: str = "Test reason."
) -> str:
    return f'{{"label": "{label}", "confidence": {confidence}, "reason": "{reason}"}}'


def test_supported_verdict():
    judge = _make_judge()
    with patch.object(judge.client, "call", return_value=_mock_response("SUPPORTED")):
        result = judge.evaluate(_CLAIM, [_DOC])
    assert result.label == SupportLabel.SUPPORTED
    assert result.confidence == 0.9


def test_refuted_verdict():
    judge = _make_judge()
    with patch.object(
        judge.client, "call", return_value=_mock_response("REFUTED", 0.8)
    ):
        result = judge.evaluate(_CLAIM, [_DOC])
    assert result.label == SupportLabel.REFUTED


def test_no_docs_returns_unclear():
    judge = _make_judge()
    result = judge.evaluate(_CLAIM, [])
    assert result.label == SupportLabel.UNCLEAR
    assert result.confidence == 0.0


def test_llm_failure_returns_unclear():
    judge = _make_judge(max_retries=1)
    with patch.object(judge.client, "call", side_effect=RuntimeError("timeout")):
        result = judge.evaluate(_CLAIM, [_DOC])
    assert result.label == SupportLabel.UNCLEAR


def test_malformed_json_retries_then_unclear():
    judge = _make_judge(max_retries=2)
    with patch.object(judge.client, "call", return_value="not json at all"):
        result = judge.evaluate(_CLAIM, [_DOC])
    assert result.label == SupportLabel.UNCLEAR


def test_reason_preserved():
    judge = _make_judge()
    expected_reason = "Document clearly states the location."
    with patch.object(
        judge.client,
        "call",
        return_value=_mock_response("SUPPORTED", 0.95, expected_reason),
    ):
        result = judge.evaluate(_CLAIM, [_DOC])
    assert result.reason == expected_reason
