"""
Relevance judge — classifies claims relative to the question that prompted an answer.

The judge assigns one of three labels:
  CORE          — directly relevant to the core question
  COMPLEMENTARY — related but supplementary (context, caveats, tangential detail)
  IRRELEVANT    — unrelated to the question
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..data_models import Claim, RelevanceLabel, RelevanceResult
from ..llm_client import LiteLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class RelevanceJudge(ABC):
    """Abstract base class for claim relevance judges."""

    @abstractmethod
    def evaluate(self, claim: Claim, question: str) -> RelevanceResult:
        """
        Classify how relevant *claim* is to *question*.

        Args:
            claim:    The claim to classify.
            question: The original question the answer was responding to.

        Returns:
            A :class:`~acclaim.data_models.RelevanceResult` with label,
            confidence, and reason.
        """


# ---------------------------------------------------------------------------
# LiteLLM implementation
# ---------------------------------------------------------------------------


class _RelevanceResponse(BaseModel):
    """Pydantic schema for the structured LLM relevance judge output."""

    label: RelevanceLabel = Field(..., description="CORE, COMPLEMENTARY, or IRRELEVANT")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=10)


class LiteLLMRelevanceJudge(RelevanceJudge):
    """
    Relevance judge backed by any LLM via LiteLLM.

    Classifies each claim as CORE, COMPLEMENTARY, or IRRELEVANT relative to
    the original question.

    Usage::

        judge = LiteLLMRelevanceJudge(model="gpt-4o-mini")
        result = judge.evaluate(claim, question="What causes inflation?")

    Local vLLM::

        judge = LiteLLMRelevanceJudge(
            model="openai/meta-llama-3-8b-instruct",
            api_base="http://localhost:8000/v1",
            api_key="dummy",
        )
    """

    _SYSTEM_PROMPT = """You are an expert at evaluating the relevance of claims to a given question.

Given a question and a claim extracted from an answer, classify the claim's relevance:

  CORE          — the claim directly addresses or answers the question
  COMPLEMENTARY — the claim provides useful context, caveats, or related detail, \
but does not directly answer the question
  IRRELEVANT    — the claim is unrelated to the question

Respond with a JSON object with exactly these keys:
- "label":      One of "CORE", "COMPLEMENTARY", or "IRRELEVANT"
- "confidence": Float between 0.0 and 1.0
- "reason":     Brief explanation (at least 10 characters)"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the LiteLLM relevance judge.

        Args:
            model:         LiteLLM model identifier.
            api_base:      Custom API endpoint (for vLLM, etc.).
            api_key:       Optional API key.
            temperature:   Sampling temperature.
            max_tokens:    Maximum response tokens.
            max_retries:   JSON parse retry attempts.
            system_prompt: Override the default system prompt.
            **kwargs:      Forwarded to ``litellm.completion()``.
        """
        self.system_prompt = system_prompt or self._SYSTEM_PROMPT
        self.client = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
            max_retries=max_retries,
            **kwargs,
        )
        logger.debug(
            "LiteLLMRelevanceJudge initialised: model=%s api_base=%s", model, api_base
        )

    def _build_user_message(self, claim: Claim, question: str) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Claim to classify:\n{claim.text}\n\n"
            "How relevant is this claim to the question?\n"
            "Provide your judgment as a JSON object."
        )

    def _parse_response(self, raw: str) -> RelevanceResult:
        cleaned = LiteLLMClient.strip_code_fences(raw)
        data = json.loads(cleaned)
        resp = _RelevanceResponse(**data)
        return RelevanceResult(
            label=resp.label,
            confidence=resp.confidence,
            reason=resp.reason,
        )

    def evaluate(self, claim: Claim, question: str) -> RelevanceResult:
        """
        Classify the relevance of *claim* to *question*.

        Returns a ``COMPLEMENTARY`` result with confidence 0.0 if all retries
        fail, as a safe neutral fallback.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(claim, question)},
        ]

        try:
            return self.client.call_with_retry(messages, self._parse_response)
        except Exception as exc:
            logger.error("LiteLLMRelevanceJudge: evaluation failed after retries: %s", exc)
            return RelevanceResult(
                label=RelevanceLabel.COMPLEMENTARY,
                confidence=0.0,
                reason=f"Relevance judge failed: {exc}",
            )
