"""
Atomic claim extractor backed by LiteLLM.

Uses an LLM to decompose each sentence or full answer into short,
self-contained atomic claims.  Because the LLM may rephrase content,
the returned claims carry ``span=(-1, -1)`` and must be aligned using
:class:`~acclaim.alignment.jaccard.JaccardCitationAligner`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import TypeAdapter

from .base import ClaimExtractor
from ..data_models import Answer, Claim
from ..llm_client import LiteLLMClient

logger = logging.getLogger(__name__)

_LIST_STR_ADAPTER: TypeAdapter[list[str]] = TypeAdapter(list[str])


class AtomicClaimExtractor(ClaimExtractor):
    """
    Decompose answer text into atomic claims via an LLM.

    Atomic claims have ``span=(-1, -1)`` because the LLM may rephrase
    them.  Pair this extractor with
    :class:`~acclaim.alignment.jaccard.JaccardCitationAligner`.

    Usage::

        extractor = AtomicClaimExtractor(model="gpt-4o-mini")
        claims = extractor.extract(answer)
    """

    _SYSTEM_PROMPT = (
        "You are an expert at breaking down complex text into atomic claims.\n"
        "An atomic claim is a short, simple sentence containing exactly one piece of information.\n"
        "Atomic claims must NOT contain conjunctions that join multiple facts.\n\n"
        "Always respond with ONLY a JSON array of strings.  No other text."
    )

    _USER_PROMPT = (
        "Break the following text into atomic claims.\n\n"
        'Example input: "The Eiffel Tower, completed in 1889, is in Paris and stands 330 m."\n'
        'Example output: ["The Eiffel Tower was completed in 1889.", '
        '"The Eiffel Tower is in Paris.", "The Eiffel Tower stands 330 m."]\n\n'
        "Text: {text}\n"
        "Output:"
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the atomic claim extractor.

        Args:
            model:         LiteLLM model identifier.
            api_base:      Custom API base URL (e.g. local vLLM server).
            api_key:       Optional API key.
            temperature:   Sampling temperature.
            max_tokens:    Maximum tokens in response.
            max_retries:   Retry attempts on JSON parse failure.
            system_prompt: Override the default system prompt.
            **kwargs:      Extra arguments forwarded to ``litellm.completion()``.
        """
        self.max_retries = max_retries
        self.system_prompt = system_prompt or self._SYSTEM_PROMPT
        self.client = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )
        logger.debug("AtomicClaimExtractor initialised: model=%s", model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_claims(self, response_text: str) -> list[str]:
        """Parse the raw LLM response into a validated list of claim strings."""
        cleaned = LiteLLMClient.strip_code_fences(response_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting a JSON array if the model added extra prose
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1:
                data = json.loads(cleaned[start : end + 1])
            else:
                raise

        # Some models wrap the list in an object e.g. {"claims": [...]}
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    data = value
                    break
            else:
                raise ValueError(f"Expected JSON list, got dict: {data}")

        return _LIST_STR_ADAPTER.validate_python(data)

    # ------------------------------------------------------------------
    # ClaimExtractor interface
    # ------------------------------------------------------------------

    def extract(self, answer: Answer) -> list[Claim]:
        """Extract atomic claims from *answer* using the configured LLM."""
        if not answer.text:
            return []

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._USER_PROMPT.format(text=answer.text)},
        ]

        try:
            texts = self.client.call_with_retry(messages, self._parse_claims)
        except Exception as exc:
            logger.error(
                "AtomicClaimExtractor: extraction failed after retries: %s", exc
            )
            return []

        return [Claim(text=t, span=(-1, -1)) for t in texts]
