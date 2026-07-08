"""
Unit tests for LiteLLMPrecisionJudge and LiteLLMRelevanceJudge (all LLM calls mocked).
"""

from unittest.mock import patch

from acclaim.data_models import Claim, Document, RelevanceLabel
from acclaim.judges.citation_precision import LiteLLMPrecisionJudge, PrecisionLabel
from acclaim.judges.relevance import LiteLLMRelevanceJudge

_CLAIM = Claim(text="The Eiffel Tower is in Paris.", span=(-1, -1))
_DOC = Document(doc_id="doc-1", text="The Eiffel Tower is located in Paris, France.")


# ---------------------------------------------------------------------------
# LiteLLMPrecisionJudge
# ---------------------------------------------------------------------------


def _make_precision_judge(**kwargs) -> LiteLLMPrecisionJudge:
    return LiteLLMPrecisionJudge(model="gpt-4o-mini", **kwargs)


def test_precision_relevant_verdict():
    judge = _make_precision_judge()
    response = "Rating: [[Relevant]]\nAnalysis: The snippet confirms the claim."
    with patch.object(judge.client, "call", return_value=response):
        result = judge.judge(_CLAIM, _DOC)
    assert result.label == PrecisionLabel.RELEVANT
    assert result.score == 1.0
    assert result.reason == "The snippet confirms the claim."


def test_precision_unrelevant_verdict():
    judge = _make_precision_judge()
    response = "Rating: [[Unrelevant]]\nAnalysis: Off-topic snippet."
    with patch.object(judge.client, "call", return_value=response):
        result = judge.judge(_CLAIM, _DOC)
    assert result.label == PrecisionLabel.UNRELEVANT
    assert result.score == 0.0


def test_precision_failure_returns_unrelevant_fallback():
    judge = _make_precision_judge(max_retries=1)
    with patch.object(judge.client, "call", side_effect=RuntimeError("timeout")):
        result = judge.judge(_CLAIM, _DOC)
    assert result.label == PrecisionLabel.UNRELEVANT
    assert result.score == 0.0
    assert "Precision judge failed" in result.reason


def test_precision_malformed_response_retries_then_fallback():
    judge = _make_precision_judge(max_retries=2)
    with patch.object(judge.client, "call", return_value="no rating here"):
        result = judge.judge(_CLAIM, _DOC)
    assert result.label == PrecisionLabel.UNRELEVANT


# ---------------------------------------------------------------------------
# LiteLLMRelevanceJudge
# ---------------------------------------------------------------------------


def _make_relevance_judge(**kwargs) -> LiteLLMRelevanceJudge:
    return LiteLLMRelevanceJudge(model="gpt-4o-mini", **kwargs)


def _mock_relevance_response(label: str, confidence: float = 0.9, reason: str = "A" * 15) -> str:
    return f'{{"label": "{label}", "confidence": {confidence}, "reason": "{reason}"}}'


def test_relevance_core_verdict():
    judge = _make_relevance_judge()
    with patch.object(
        judge.client, "call", return_value=_mock_relevance_response("CORE")
    ):
        result = judge.evaluate(_CLAIM, "Where is the Eiffel Tower?")
    assert result.label == RelevanceLabel.CORE
    assert result.confidence == 0.9


def test_relevance_complementary_verdict():
    judge = _make_relevance_judge()
    with patch.object(
        judge.client, "call", return_value=_mock_relevance_response("COMPLEMENTARY")
    ):
        result = judge.evaluate(_CLAIM, "Where is the Eiffel Tower?")
    assert result.label == RelevanceLabel.COMPLEMENTARY


def test_relevance_irrelevant_verdict():
    judge = _make_relevance_judge()
    with patch.object(
        judge.client, "call", return_value=_mock_relevance_response("IRRELEVANT")
    ):
        result = judge.evaluate(_CLAIM, "Where is the Eiffel Tower?")
    assert result.label == RelevanceLabel.IRRELEVANT


def test_relevance_failure_returns_complementary_fallback():
    judge = _make_relevance_judge(max_retries=1)
    with patch.object(judge.client, "call", side_effect=RuntimeError("timeout")):
        result = judge.evaluate(_CLAIM, "Where is the Eiffel Tower?")
    assert result.label == RelevanceLabel.COMPLEMENTARY
    assert result.confidence == 0.0
    assert "Relevance judge failed" in result.reason


def test_relevance_strips_code_fences():
    judge = _make_relevance_judge()
    fenced = "```json\n" + _mock_relevance_response("CORE") + "\n```"
    with patch.object(judge.client, "call", return_value=fenced):
        result = judge.evaluate(_CLAIM, "Where is the Eiffel Tower?")
    assert result.label == RelevanceLabel.CORE
