"""
Shared LiteLLM client for unified LLM access across acclaim.

All LLM calls (judges, claim extractors, etc.) go through this client
to ensure consistent configuration, retry logic, and provider support.

Supports OpenAI, Anthropic, vLLM, and any other LiteLLM provider.
"""

import logging
from typing import Any, Callable

import litellm
from litellm import completion

litellm.suppress_debug_info = True
logger = logging.getLogger(__name__)


class LiteLLMClient:
    """
    Unified LiteLLM client for all LLM calls in the project.

    Usage:
        client = LiteLLMClient(model="gpt-4o-mini")
        text = client.call([{"role": "user", "content": "Hello!"}])

    Provider examples:
        OpenAI:    model="gpt-4o-mini"
        Anthropic: model="claude-3-5-sonnet-20241022"
        vLLM:      model="openai/<model>", api_base="http://localhost:8000/v1", api_key="dummy"
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: str | None = None,
        max_retries: int = 3,
        thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the client.

        Args:
            model:           LiteLLM model identifier.
            api_base:        Custom API endpoint (for vLLM, etc.).
            api_key:         Optional API key; falls back to environment variables.
            temperature:     Sampling temperature.
            max_tokens:      Maximum tokens in a response.
            response_format: ``"json_object"`` for JSON mode, ``None`` for plain text.
            max_retries:     Number of retry attempts on transient failures.
            thinking:        Whether to enable reasoning/thinking on the model.
                             Defaults to ``False``; sends
                             ``chat_template_kwargs={"enable_thinking": <bool>}``
                             via ``extra_body`` so vLLM-served models
                             (Qwen3, etc.) honour the toggle.
            **kwargs:        Extra keyword arguments forwarded to every
                             ``litellm.completion()`` call.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.max_retries = max_retries
        self.thinking = thinking
        self.extra_kwargs = kwargs
        logger.debug("LiteLLMClient initialised: model=%s api_base=%s", model, api_base)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Make a single LLM call and return the response text.

        Per-call keyword arguments override the client defaults.

        Args:
            messages:        Chat messages in ``{"role": ..., "content": ...}`` format.
            temperature:     Per-call override.
            max_tokens:      Per-call override.
            response_format: Per-call override (``"json_object"`` or ``None``).
            **kwargs:        Additional per-call overrides.

        Returns:
            The text content of the first choice in the LLM response.
        """
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._apply_qwen_thinking_directive(messages),
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **self.extra_kwargs,
            **kwargs,
        }

        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        fmt = response_format if response_format is not None else self.response_format
        if fmt:
            call_kwargs["response_format"] = {"type": fmt}

        # Forward thinking toggle to vLLM-served models via chat_template_kwargs
        # unless the caller already supplied one explicitly.
        existing_extra_body = call_kwargs.get("extra_body") or {}
        existing_template_kwargs = existing_extra_body.get("chat_template_kwargs") or {}
        if "enable_thinking" not in existing_template_kwargs:
            call_kwargs["extra_body"] = {
                **existing_extra_body,
                "chat_template_kwargs": {
                    **existing_template_kwargs,
                    "enable_thinking": self.thinking,
                },
            }

        response = completion(**call_kwargs)
        return response.choices[0].message.content  # type: ignore[union-attr]

    def call_with_retry(
        self,
        messages: list[dict[str, str]],
        parse_fn: Callable[[str], Any],
        *,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call the LLM with retry logic driven by a parse/validation function.

        The call is retried whenever ``parse_fn`` raises an exception.

        Args:
            messages:    Chat messages.
            parse_fn:    Callable that converts the raw response string into the
                         desired return type.  Raise on parse failure to trigger retry.
            max_retries: Override the client-level retry count for this call.
            **kwargs:    Forwarded to :meth:`call`.

        Returns:
            The result of the first successful ``parse_fn`` invocation.

        Raises:
            The last exception produced by ``parse_fn`` if all retries fail.
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                return parse_fn(self.call(messages, **kwargs))
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    logger.warning(
                        "Attempt %d/%d failed (%s: %s). Retrying…",
                        attempt + 1,
                        retries,
                        type(exc).__name__,
                        exc,
                    )

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _apply_qwen_thinking_directive(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Append Qwen's ``/no_think`` (or ``/think``) directive to the last user
        message when the model is a Qwen variant.

        The ``extra_body.chat_template_kwargs.enable_thinking`` switch only
        works against native vLLM. OpenAI-compat gateways (Ollama, proxies)
        often drop ``extra_body``, so we additionally inject the in-prompt
        directive that Qwen3 templates honour directly. Non-Qwen models
        treat the directive as harmless trailing text — but we still skip
        it for them to keep prompts clean.
        """
        if "qwen" not in self.model.lower():
            return messages

        directive = "/think" if self.thinking else "/no_think"
        # Find last user message; if none, no-op.
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                if directive in content:
                    return messages
                patched = list(messages)
                patched[i] = {**messages[i], "content": f"{content} {directive}"}
                return patched
        return messages

    @staticmethod
    def strip_code_fences(text: str) -> str:
        """Strip Markdown code fences from LLM output."""
        text = text.strip()
        for prefix in ("```json", "```"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
