"""
Citation-recall judge for the LongCite-style citation recall metric.

Inspired by:
  Zhang et al., "LongCite: Enabling LLMs to Generate Fine-grained Citations
  in Long-context QA", 2024.

The judge asks an LLM to decide whether a concatenated set of cited snippets
*fully supports* (1), *partially supports* (0.5), or *does not support* (0)
a given claim.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..data_models import Claim, Document
from ..llm_client import LiteLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class RecallLabel(str, Enum):
    """Three-way support label used by the citation recall judge."""

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"

    @property
    def score(self) -> float:
        """Map label to the LongCite scalar: FULL→1.0, PARTIAL→0.5, NONE→0.0."""
        return {
            RecallLabel.FULL: 1.0,
            RecallLabel.PARTIAL: 0.5,
            RecallLabel.NONE: 0.0,
        }[self]


@dataclass(frozen=True)
class RecallResult:
    """
    Output of a single :class:`RecallJudge` call.

    Attributes:
        label:  Recall label assigned by the judge.
        score:  Scalar score derived from *label* (FULL→1.0, PARTIAL→0.5, NONE→0.0).
        reason: Brief explanation produced by the judge.
    """

    label: RecallLabel
    score: float
    reason: str


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class RecallJudge(ABC):
    """Abstract base class for citation recall judges."""

    @abstractmethod
    def recall(self, claim: Claim, docs: list[Document]) -> RecallResult:
        """
        Score how well *docs* recall (support) *claim*.

        Args:
            claim: The claim to verify.
            docs:  Documents cited by *claim*.

        Returns:
            A :class:`RecallResult` with label, score, and reason.
        """


# ---------------------------------------------------------------------------
# LiteLLM implementation
# ---------------------------------------------------------------------------


class _RecallResponse(BaseModel):
    """Pydantic schema for the structured LLM recall judge output."""

    label: RecallLabel = Field(
        ...,
        description="FULL, PARTIAL, or NONE",
    )
    reason: str = Field(..., min_length=10)


class LiteLLMRecallJudge(RecallJudge):
    """
    Citation recall judge backed by any LLM via LiteLLM.

    Scores each claim 0 / 0.5 / 1 based on whether the concatenated
    cited snippets fully, partially, or do not support it — following
    the LongCite citation recall definition.

    Usage::

        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        result = judge.recall(claim, docs)

    Local vLLM::

        judge = LiteLLMRecallJudge(
            model="openai/meta-llama-3-8b-instruct",
            api_base="http://localhost:8000/v1",
            api_key="dummy",
        )
    """

    _SYSTEM_PROMPT = """You are an expert fact-checker evaluating citation quality.
Given a claim and the cited evidence text, judge how well the evidence \
supports the claim.

Respond with a JSON object with exactly these keys:
- "label":  One of "FULL", "PARTIAL", or "NONE"
- "reason": Brief explanation (at least 10 characters)

Labels:
  FULL    — the evidence fully and unambiguously supports the entire claim
  PARTIAL — the evidence supports part of the claim but is incomplete or \
ambiguous
  NONE    — the evidence does not support the claim"""

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
        Initialise the LiteLLM recall judge.

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
            "LiteLLMRecallJudge initialised: model=%s api_base=%s", model, api_base
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, claim: Claim, docs: list[Document]) -> str:
        evidence = "\n\n".join(f"Document {d.doc_id}:\n{d.text}" for d in docs)
        return (
            f"Evidence Documents:\n{evidence}\n\n"
            f"Claim to evaluate:\n{claim.text}\n\n"
            "Does the evidence fully support, partially support, or not support "
            "the claim?\n"
            "Provide your judgment as a JSON object."
        )

    def _parse_response(self, raw: str) -> RecallResult:
        cleaned = LiteLLMClient.strip_code_fences(raw)
        data = json.loads(cleaned)
        resp = _RecallResponse(**data)
        return RecallResult(
            label=resp.label,
            score=resp.label.score,
            reason=resp.reason,
        )

    # ------------------------------------------------------------------
    # RecallJudge interface
    # ------------------------------------------------------------------

    def recall(self, claim: Claim, docs: list[Document]) -> RecallResult:
        """
        Judge whether *docs* recall *claim*.

        Returns a ``RecallLabel.NONE`` result (score 0.0) when no documents
        are provided or if all retries fail.
        """
        if not docs:
            return RecallResult(
                label=RecallLabel.NONE,
                score=0.0,
                reason="No documents provided for this claim.",
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(claim, docs)},
        ]

        try:
            return self.client.call_with_retry(messages, self._parse_response)
        except Exception as exc:
            logger.error("LiteLLMRecallJudge: evaluation failed after retries: %s", exc)
            return RecallResult(
                label=RecallLabel.NONE,
                score=0.0,
                reason=f"Recall judge failed: {exc}",
            )
