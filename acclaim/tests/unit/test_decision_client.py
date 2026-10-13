"""
Unit tests for DecisionClient (litellm.systemone mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from acclaim.decision_client import DecisionClient, DecisionResponse


_STATE = {"claim": "c", "evidence": []}
_QUESTIONS = {"q": {"type": "noul", "instructions": "Is it?"}}


def _make_client(**kwargs) -> DecisionClient:
    return DecisionClient(model="openrouter/typesafe/jev-1.13", retry_delay=0, **kwargs)


def _fake_response(answers=None, usage=None, model="typesafe/jev-1.13-20260917"):
    resp = MagicMock()
    resp.model_dump.return_value = {
        "model": model,
        "answers": answers if answers is not None else {"q": {"type": "noul", "noul": 0.8}},
        "usage": usage if usage is not None else {"input_tokens": 10, "output_tokens": 0, "cost": 1e-6},
    }
    return resp


def test_ask_returns_plain_answers_and_usage():
    client = _make_client()
    with patch("acclaim.decision_client.systemone", return_value=_fake_response()) as mock:
        resp = client.ask(_STATE, _QUESTIONS)
    assert isinstance(resp, DecisionResponse)
    assert resp.answers == {"q": {"type": "noul", "noul": 0.8}}
    assert resp.model == "typesafe/jev-1.13-20260917"
    assert resp.cost == 1e-6
    assert resp.latency_s >= 0.0
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "openrouter/typesafe/jev-1.13"
    assert kwargs["state"] == _STATE
    assert kwargs["questions"] == _QUESTIONS
    assert "api_key" not in kwargs and "api_base" not in kwargs


def test_ask_forwards_connection_fields():
    client = _make_client(api_key="k", api_base="http://x", timeout=5.0)
    with patch("acclaim.decision_client.systemone", return_value=_fake_response()) as mock:
        client.ask(_STATE, _QUESTIONS)
    kwargs = mock.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["api_base"] == "http://x"
    assert kwargs["timeout"] == 5.0


def test_cost_none_when_not_reported():
    client = _make_client()
    with patch("acclaim.decision_client.systemone", return_value=_fake_response(usage={})):
        assert client.ask(_STATE, _QUESTIONS).cost is None


def test_ask_with_retry_recovers_from_call_error():
    client = _make_client(max_retries=2)
    with patch(
        "acclaim.decision_client.systemone",
        side_effect=[RuntimeError("503"), _fake_response()],
    ) as mock:
        p = client.ask_with_retry(_STATE, _QUESTIONS, lambda r: r.answers["q"]["noul"])
    assert p == 0.8
    assert mock.call_count == 2


def test_ask_with_retry_retries_on_parse_error_then_raises():
    client = _make_client(max_retries=3)

    def _bad_parse(_resp):
        raise ValueError("bad answer")

    with patch("acclaim.decision_client.systemone", return_value=_fake_response()) as mock:
        with pytest.raises(ValueError, match="bad answer"):
            client.ask_with_retry(_STATE, _QUESTIONS, _bad_parse)
    assert mock.call_count == 3
