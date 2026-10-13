"""
Unit tests for the DecisionJudge (all decision-model calls mocked).
"""

from unittest.mock import patch

import pytest

from acclaim.data_models import Claim, Document, SupportLabel
from acclaim.decision_client import DecisionResponse
from acclaim.judges.decision import DecisionJudge


_CLAIM = Claim(text="The Eiffel Tower is in Paris.", span=(-1, -1))
_DOC = Document(doc_id="doc-1", text="The Eiffel Tower is located in Paris, France.")


def _make_judge(**kwargs) -> DecisionJudge:
    return DecisionJudge(model="openrouter/typesafe/jev-1.13", retry_delay=0, **kwargs)


def _choice(choice: str, probs: dict[str, float]) -> DecisionResponse:
    return DecisionResponse(
        answers={"verdict": {"type": "choice", "choice": choice,
                             "confidence": max(probs.values()), "probabilities": probs}},
        usage={"input_tokens": 100, "output_tokens": 0, "cost": 1e-5},
    )


def _score(probs: list[float]) -> DecisionResponse:
    return DecisionResponse(
        answers={"verdict": {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probs)),
            "confidence": max(probs),
            "legend": {"0": "REFUTED", "1": "UNCLEAR", "2": "SUPPORTED"},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
        }},
    )


def _noul(p_sup: float, p_ref: float) -> DecisionResponse:
    return DecisionResponse(
        answers={
            "supported": {"type": "noul", "noul": p_sup},
            "refuted": {"type": "noul", "noul": p_ref},
        },
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown decision mode"):
        _make_judge(mode="vote")


def test_unknown_state_format_raises():
    with pytest.raises(ValueError, match="Unknown state_format"):
        _make_judge(state_format="xml")


def test_unknown_threshold_key_raises():
    with pytest.raises(ValueError, match="Unknown threshold keys"):
        _make_judge(mode="noul", thresholds={"unclear": 0.3})


def test_partial_thresholds_fill_defaults():
    judge = _make_judge(mode="noul", thresholds={"supported": 0.7})
    assert judge.thresholds == {"supported": 0.7, "refuted": 0.5}


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def test_structured_state_and_choice_question():
    judge = _make_judge()
    with patch.object(judge.client, "ask", return_value=_choice(
        "SUPPORTED", {"SUPPORTED": 0.9, "REFUTED": 0.05, "UNCLEAR": 0.05}
    )) as mock:
        judge.evaluate(_CLAIM, [_DOC])
    state, questions = mock.call_args.args
    assert state == {"claim": _CLAIM.text, "evidence": [{"doc_id": "doc-1", "text": _DOC.text}]}
    assert questions["verdict"]["type"] == "choice"
    assert set(questions["verdict"]["criteria"]) == {"SUPPORTED", "REFUTED", "UNCLEAR"}


def test_text_state_format():
    judge = _make_judge(state_format="text")
    with patch.object(judge.client, "ask", return_value=_choice(
        "SUPPORTED", {"SUPPORTED": 1.0, "REFUTED": 0.0, "UNCLEAR": 0.0}
    )) as mock:
        judge.evaluate(_CLAIM, [_DOC])
    state = mock.call_args.args[0]
    assert isinstance(state, str)
    assert "Document doc-1:" in state and _CLAIM.text in state


def test_score_question_levels_are_ordered_low_to_high():
    judge = _make_judge(mode="score")
    q = judge._build_questions()["verdict"]
    assert q["type"] == "score"
    assert [c.split(":")[0] for c in q["criteria"]] == ["REFUTED", "UNCLEAR", "SUPPORTED"]


def test_noul_mode_asks_two_questions():
    q = _make_judge(mode="noul")._build_questions()
    assert set(q) == {"supported", "refuted"}
    assert all(v["type"] == "noul" and set(v["criteria"]) == {"true", "false"} for v in q.values())


# ---------------------------------------------------------------------------
# Response parsing per mode
# ---------------------------------------------------------------------------


def test_choice_mode_label_and_confidence():
    judge = _make_judge()
    with patch.object(judge.client, "ask", return_value=_choice(
        "REFUTED", {"SUPPORTED": 0.1, "REFUTED": 0.7, "UNCLEAR": 0.2}
    )):
        detailed = judge.judge_detailed(_CLAIM, [_DOC])
    assert detailed.result.label == SupportLabel.REFUTED
    assert detailed.result.confidence == 0.7
    assert detailed.probabilities == {"SUPPORTED": 0.1, "REFUTED": 0.7, "UNCLEAR": 0.2}
    assert detailed.result.reason.startswith("choice: REFUTED=0.70")
    assert detailed.response.cost == 1e-5


def test_score_mode_uses_argmax_level():
    judge = _make_judge(mode="score")
    with patch.object(judge.client, "ask", return_value=_score([0.1, 0.6, 0.3])):
        detailed = judge.judge_detailed(_CLAIM, [_DOC])
    assert detailed.result.label == SupportLabel.UNCLEAR
    assert detailed.result.confidence == 0.6
    assert detailed.probabilities == {"REFUTED": 0.1, "UNCLEAR": 0.6, "SUPPORTED": 0.3}
    assert "score=1.20" in detailed.result.reason


@pytest.mark.parametrize(
    "p_sup, p_ref, expected",
    [
        (0.9, 0.1, SupportLabel.SUPPORTED),
        (0.5, 0.9, SupportLabel.SUPPORTED),  # supported wins ties at threshold
        (0.2, 0.8, SupportLabel.REFUTED),
        (0.2, 0.5, SupportLabel.REFUTED),
        (0.2, 0.3, SupportLabel.UNCLEAR),
    ],
)
def test_noul_mode_default_thresholds(p_sup, p_ref, expected):
    judge = _make_judge(mode="noul")
    with patch.object(judge.client, "ask", return_value=_noul(p_sup, p_ref)):
        detailed = judge.judge_detailed(_CLAIM, [_DOC])
    assert detailed.result.label == expected
    assert sum(detailed.probabilities.values()) == pytest.approx(1.0)
    assert detailed.result.confidence == detailed.probabilities[expected.value]


def test_noul_mode_custom_thresholds():
    judge = _make_judge(mode="noul", thresholds={"supported": 0.8, "refuted": 0.3})
    with patch.object(judge.client, "ask", return_value=_noul(0.7, 0.35)):
        assert judge.evaluate(_CLAIM, [_DOC]).label == SupportLabel.REFUTED


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_no_docs_returns_unclear_without_call():
    judge = _make_judge()
    with patch.object(judge.client, "ask") as mock:
        result = judge.evaluate(_CLAIM, [])
    assert result.label == SupportLabel.UNCLEAR
    assert result.confidence == 0.0
    mock.assert_not_called()


def test_call_failure_returns_unclear():
    judge = _make_judge(max_retries=2)
    with patch.object(judge.client, "ask", side_effect=RuntimeError("timeout")) as mock:
        detailed = judge.judge_detailed(_CLAIM, [_DOC])
    assert detailed.result.label == SupportLabel.UNCLEAR
    assert detailed.result.confidence == 0.0
    assert detailed.response is None
    assert mock.call_count == 2


def test_wrong_answer_type_retries_then_unclear():
    judge = _make_judge(max_retries=2)
    with patch.object(judge.client, "ask", return_value=_noul(0.9, 0.1)) as mock:
        result = judge.evaluate(_CLAIM, [_DOC])  # choice mode expects "verdict"
    assert result.label == SupportLabel.UNCLEAR
    assert mock.call_count == 2


def test_unknown_choice_label_retries_then_succeeds():
    judge = _make_judge(max_retries=2)
    with patch.object(judge.client, "ask", side_effect=[
        _choice("MAYBE", {"MAYBE": 1.0}),
        _choice("SUPPORTED", {"SUPPORTED": 0.8, "REFUTED": 0.1, "UNCLEAR": 0.1}),
    ]):
        assert judge.evaluate(_CLAIM, [_DOC]).label == SupportLabel.SUPPORTED
