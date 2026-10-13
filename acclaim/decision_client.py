"""
Shared client for *decision models* (OpenRouter Decisions / System One API).

Decision models do not generate text. They take a ``state`` (the thing being
judged) plus a set of named, typed questions and return a probability-bearing
answer for each question:

  noul    — yes/no; answer carries ``noul`` = P(true)
  choice  — one of several labels; answer carries ``choice``, ``confidence``
            and per-label ``probabilities``
  score   — position on an ordered scale; answer carries ``score``
            (probability-weighted level index), ``confidence``, ``legend``
            and per-level ``probabilities``

All calls go through ``litellm.systemone``, so any provider LiteLLM supports
for decisions works, e.g. ``openrouter/typesafe/jev-1.13``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import litellm
from litellm import systemone

litellm.suppress_debug_info = True
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionResponse:
    """Plain-dict view of a decision-model response."""

    answers: dict[str, dict[str, Any]]
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0

    @property
    def cost(self) -> float | None:
        """Provider-reported cost in USD, if available."""
        return self.usage.get("cost")


class DecisionClient:
    """
    Unified client for decision-model calls.

    Usage::

        client = DecisionClient(model="openrouter/typesafe/jev-1.13")
        resp = client.ask(
            state={"claim": "...", "evidence": [...]},
            questions={"supported": {"type": "noul", "instructions": "..."}},
        )
        p_supported = resp.answers["supported"]["noul"]
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the client.

        Args:
            model:       LiteLLM decision-model identifier
                         (e.g. ``"openrouter/typesafe/jev-1.13"``).
            api_base:    Custom API endpoint.
            api_key:     Optional API key; falls back to environment variables.
            max_retries: Number of attempts before giving up.
            retry_delay: Base delay in seconds between attempts (doubles each retry).
            timeout:     Per-request timeout in seconds.
            **kwargs:    Extra keyword arguments forwarded to every
                         ``litellm.systemone()`` call.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.extra_kwargs = kwargs
        logger.debug("DecisionClient initialised: model=%s api_base=%s", model, api_base)

    def ask(
        self,
        state: str | Mapping[str, Any] | list[Any],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> DecisionResponse:
        """Send one decisions request and return the answers as plain dicts."""
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": questions,
            **self.extra_kwargs,
        }
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.timeout is not None:
            call_kwargs["timeout"] = self.timeout

        t0 = time.perf_counter()
        response = systemone(**call_kwargs)
        latency = time.perf_counter() - t0

        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return DecisionResponse(
            answers={k: dict(v) for k, v in (data.get("answers") or {}).items()},
            model=data.get("model"),
            usage=dict(data.get("usage") or {}),
            latency_s=latency,
        )

    def ask_with_retry(
        self,
        state: str | Mapping[str, Any] | list[Any],
        questions: Mapping[str, Mapping[str, Any]],
        parse_fn: Callable[[DecisionResponse], Any],
        *,
        max_retries: int | None = None,
    ) -> Any:
        """
        Call the decision model, retrying whenever the call or ``parse_fn`` raises.

        Args:
            state:       The state to judge.
            questions:   Named questions (System One format).
            parse_fn:    Converts the :class:`DecisionResponse` into the desired
                         return type. Raise to trigger a retry.
            max_retries: Override the client-level retry count for this call.

        Returns:
            The result of the first successful ``parse_fn`` invocation.

        Raises:
            The last exception if all attempts fail.
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                return parse_fn(self.ask(state, questions))
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
                    if self.retry_delay > 0:
                        time.sleep(self.retry_delay * 2**attempt)

        assert last_error is not None
        raise last_error
