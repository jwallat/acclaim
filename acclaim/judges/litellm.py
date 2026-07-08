"""
LiteLLM-based evidence judge.

The single judge implementation in acclaim v2.  All inference goes
through :class:`~acclaim.llm_client.LiteLLMClient`, which supports
OpenAI, Anthropic, a local vLLM server, and any other LiteLLM provider.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from .base import EvidenceJudge
from ..data_models import Claim, Document, SupportLabel, SupportResult
from ..llm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class _JudgmentResponse(BaseModel):
    """Pydantic schema for the structured LLM judge output."""

    label: SupportLabel = Field(
        ...,
        description="SUPPORTED, REFUTED, or UNCLEAR",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=10)


class LiteLLMJudge(EvidenceJudge):
    """
    Evidence judge backed by any LLM via LiteLLM.

    Usage::

        judge = LiteLLMJudge(model="gpt-4o-mini")
        result = judge.evaluate(claim, docs)

    Local vLLM::

        judge = LiteLLMJudge(
            model="openai/meta-llama-3-8b-instruct",
            api_base="http://localhost:8000/v1",
            api_key="dummy",
        )
    """

    _SYSTEM_PROMPT = """You are an expert fact-checker.
Determine whether the provided claim is supported by the evidence documents.

Respond with a JSON object with exactly these keys:
- "label":      One of "SUPPORTED", "REFUTED", or "UNCLEAR"
- "confidence": Float between 0.0 and 1.0
- "reason":     Brief explanation (at least 10 characters)

Labels:
  SUPPORTED — the evidence clearly supports the claim
  REFUTED   — the evidence contradicts the claim
  UNCLEAR   — evidence is insufficient, ambiguous, or irrelevant"""

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
        Initialise the LiteLLM judge.

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
        logger.debug("LiteLLMJudge initialised: model=%s api_base=%s", model, api_base)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, claim: Claim, docs: list[Document]) -> str:
        evidence = "\n\n".join(f"Document {d.doc_id}:\n{d.text}" for d in docs)
        return (
            f"Evidence Documents:\n{evidence}\n\n"
            f"Claim to verify:\n{claim.text}\n\n"
            "Provide your judgment as a JSON object."
        )

    def _parse_response(self, raw: str) -> SupportResult:
        cleaned = LiteLLMClient.strip_code_fences(raw)
        data = json.loads(cleaned)
        resp = _JudgmentResponse(**data)
        return SupportResult(
            label=resp.label,
            confidence=resp.confidence,
            reason=resp.reason,
        )

    # ------------------------------------------------------------------
    # EvidenceJudge interface
    # ------------------------------------------------------------------

    def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult:
        # TODO: Check if we want to force only having one doc at a time. 
        """
        Judge whether *docs* support *claim*.

        Returns ``SupportLabel.UNCLEAR`` (confidence 0.0) when no documents
        are provided or if all retries fail.
        """
        if not docs:
            return SupportResult(
                label=SupportLabel.UNCLEAR,
                confidence=0.0,
                reason="No documents provided for this claim.",
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(claim, docs)},
        ]

        try:
            return self.client.call_with_retry(messages, self._parse_response)
        except Exception as exc:
            logger.error("LiteLLMJudge: evaluation failed after retries: %s", exc)
            return SupportResult(
                label=SupportLabel.UNCLEAR,
                confidence=0.0,
                reason=f"Judge failed: {exc}",
            )
