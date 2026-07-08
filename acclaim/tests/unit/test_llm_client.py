"""
Unit tests for LiteLLMClient (litellm.completion mocked, no network calls).
"""

from unittest.mock import MagicMock, patch

import pytest

from acclaim.llm_client import LiteLLMClient


def _fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_call_returns_response_text():
    client = LiteLLMClient(model="gpt-4o-mini")
    with patch("acclaim.llm_client.completion", return_value=_fake_response("hello")) as mock_completion:
        result = client.call([{"role": "user", "content": "hi"}])
    assert result == "hello"
    mock_completion.assert_called_once()


def test_call_forwards_model_and_defaults():
    client = LiteLLMClient(model="gpt-4o-mini", temperature=0.5, max_tokens=100)
    with patch("acclaim.llm_client.completion", return_value=_fake_response("x")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}])
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 100


def test_call_per_call_overrides_take_precedence():
    client = LiteLLMClient(model="gpt-4o-mini", temperature=0.5)
    with patch("acclaim.llm_client.completion", return_value=_fake_response("x")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}], temperature=0.0)
    assert mock_completion.call_args.kwargs["temperature"] == 0.0


def test_call_includes_api_base_and_key_when_set():
    client = LiteLLMClient(model="openai/foo", api_base="http://localhost:8000/v1", api_key="dummy")
    with patch("acclaim.llm_client.completion", return_value=_fake_response("x")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}])
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_base"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "dummy"


def test_call_omits_api_base_and_key_when_unset():
    client = LiteLLMClient(model="gpt-4o-mini")
    with patch("acclaim.llm_client.completion", return_value=_fake_response("x")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}])
    kwargs = mock_completion.call_args.kwargs
    assert "api_base" not in kwargs
    assert "api_key" not in kwargs


def test_call_with_response_format_sets_json_mode():
    client = LiteLLMClient(model="gpt-4o-mini", response_format="json_object")
    with patch("acclaim.llm_client.completion", return_value=_fake_response("{}")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}])
    assert mock_completion.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_qwen_thinking_directive_appended_to_last_user_message():
    client = LiteLLMClient(model="openai/Qwen3-8B", thinking=False)
    with patch("acclaim.llm_client.completion", return_value=_fake_response("x")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}])
    messages = mock_completion.call_args.kwargs["messages"]
    assert messages[-1]["content"] == "hi /no_think"


def test_non_qwen_model_does_not_get_thinking_directive():
    client = LiteLLMClient(model="gpt-4o-mini", thinking=False)
    with patch("acclaim.llm_client.completion", return_value=_fake_response("x")) as mock_completion:
        client.call([{"role": "user", "content": "hi"}])
    messages = mock_completion.call_args.kwargs["messages"]
    assert messages[-1]["content"] == "hi"


def test_call_with_retry_returns_first_successful_parse():
    client = LiteLLMClient(model="gpt-4o-mini", max_retries=3)
    with patch.object(client, "call", return_value="42"):
        result = client.call_with_retry([{"role": "user", "content": "hi"}], parse_fn=int)
    assert result == 42


def test_call_with_retry_retries_until_parse_succeeds():
    client = LiteLLMClient(model="gpt-4o-mini", max_retries=3)
    with patch.object(client, "call", side_effect=["not a number", "7"]):
        result = client.call_with_retry([{"role": "user", "content": "hi"}], parse_fn=int)
    assert result == 7


def test_call_with_retry_raises_last_error_after_exhausting_retries():
    client = LiteLLMClient(model="gpt-4o-mini", max_retries=2)
    with patch.object(client, "call", return_value="not a number"):
        with pytest.raises(ValueError):
            client.call_with_retry([{"role": "user", "content": "hi"}], parse_fn=int)


def test_strip_code_fences_removes_json_fence():
    assert LiteLLMClient.strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_removes_plain_fence():
    assert LiteLLMClient.strip_code_fences("```\nhello\n```") == "hello"


def test_strip_code_fences_no_fence_returns_unchanged():
    assert LiteLLMClient.strip_code_fences("hello") == "hello"
