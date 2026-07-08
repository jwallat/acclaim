"""
Citation-precision judge for the LongCite-style citation precision metric.

Inspired by:
  Zhang et al., "LongCite: Enabling LLMs to Generate Fine-grained Citations
  in Long-context QA", 2024.

For each (statement, cited-snippet) pair the judge asks an LLM whether the
snippet is *relevant* (1) or *unrelevant* (0) to the statement — i.e. whether
the snippet entails at least some key points of the statement.

Unlike the recall judge (which receives all cited documents at once), the
precision judge scores each citation *independently*, one call per
(claim, document) pair.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..data_models import Claim, Document
from ..llm_client import LiteLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class PrecisionLabel(str, Enum):
    """Binary relevance label used by the citation precision judge."""

    RELEVANT = "Relevant"
    UNRELEVANT = "Unrelevant"

    @property
    def score(self) -> float:
        """Map label to a scalar: RELEVANT→1.0, UNRELEVANT→0.0."""
        return 1.0 if self is PrecisionLabel.RELEVANT else 0.0


@dataclass(frozen=True)
class PrecisionResult:
    """
    Output of a single :class:`PrecisionJudge` call.

    Attributes:
        label:  Binary relevance label.
        score:  Scalar score derived from *label* (RELEVANT→1.0, UNRELEVANT→0.0).
        reason: Brief explanation produced by the judge.
    """

    label: PrecisionLabel
    score: float
    reason: str


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class PrecisionJudge(ABC):
    """Abstract base class for citation precision judges."""

    @abstractmethod
    def judge(self, claim: Claim, doc: Document) -> PrecisionResult:
        """
        Score whether *doc* is relevant to *claim*.

        Args:
            claim: The statement to evaluate.
            doc:   A single cited document snippet.

        Returns:
            A :class:`PrecisionResult` with label, score, and reason.
        """


# ---------------------------------------------------------------------------
# LiteLLM implementation
# ---------------------------------------------------------------------------

# Matches [[Relevant]] or [[Unrelevant]] (case-insensitive) in the response.
_LABEL_RE = re.compile(r"\[\[\s*(Relevant|Unrelevant)\s*\]\]", re.IGNORECASE)


class LiteLLMPrecisionJudge(PrecisionJudge):
    """
    Citation precision judge backed by any LLM via LiteLLM.

    Scores each (statement, snippet) pair as *relevant* (1) or *unrelevant*
    (0), following the LongCite citation precision definition.  The response
    format mirrors the paper's prompt: free-text starting with
    ``Rating: [[Relevant]]`` or ``Rating: [[Unrelevant]]``.

    Usage::

        judge = LiteLLMPrecisionJudge(model="gpt-4o-mini")
        result = judge.judge(claim, doc)

    Local vLLM::

        judge = LiteLLMPrecisionJudge(
            model="openai/meta-llama-3-8b-instruct",
            api_base="http://localhost:8000/v1",
            api_key="dummy",
        )
    """

    _SYSTEM_PROMPT = (
        "You are an expert in evaluating text quality. "
        "Your task is to assess whether a document snippet contains some key "
        "information of a factual statement. "
        "Respond with exactly one rating on the first line using the format "
        "'Rating: [[Relevant]]' or 'Rating: [[Unrelevant]]', "
        "followed by 'Analysis: <brief explanation>'.\n\n"
        "Grades:\n"
        "  [[Relevant]]   — some key points of the statement are supported by "
        "the snippet or extracted from it.\n"
        "  [[Unrelevant]] — the statement is almost unrelated to the snippet, "
        "or all key points of the statement are inconsistent with the snippet content.\n\n"
        "Do not use any information or knowledge outside of the snippet when evaluating."
    )

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
        Initialise the LiteLLM precision judge.

        Args:
            model:         LiteLLM model identifier.
            api_base:      Custom API endpoint (for vLLM, etc.).
            api_key:       Optional API key.
            temperature:   Sampling temperature.
            max_tokens:    Maximum response tokens.
            max_retries:   Parse/call retry attempts.
            system_prompt: Override the default system prompt.
            **kwargs:      Forwarded to ``litellm.completion()``.
        """
        self.system_prompt = system_prompt or self._SYSTEM_PROMPT
        # Free-text output — no JSON mode; we parse [[Relevant]] with regex.
        self.client = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            max_retries=max_retries,
            **kwargs,
        )
        logger.debug(
            "LiteLLMPrecisionJudge initialised: model=%s api_base=%s", model, api_base
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, claim: Claim, doc: Document, query: str = "") -> str:
        parts: list[str] = []
        if query:
            parts.append(f"<question>\n{query}\n</question>")
        parts.append(f"<statement>\n{claim.text}\n</statement>")
        parts.append(f"<snippet>\n{doc.text}\n</snippet>")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> PrecisionResult:
        match = _LABEL_RE.search(raw)
        if match is None:
            raise ValueError(
                f"Could not find [[Relevant]] or [[Unrelevant]] in response: {raw!r}"
            )
        label_str = match.group(1).capitalize()  # "Relevant" or "Unrelevant"
        label = PrecisionLabel(label_str)
        # Extract everything after the match as the reason
        reason = raw[match.end() :].strip().lstrip(":").strip()
        if reason.lower().startswith("analysis:"):
            reason = reason[len("analysis:") :].strip()
        reason = reason or f"Rated {label_str}."
        return PrecisionResult(label=label, score=label.score, reason=reason)

    # ------------------------------------------------------------------
    # PrecisionJudge interface
    # ------------------------------------------------------------------

    def judge(self, claim: Claim, doc: Document) -> PrecisionResult:
        """
        Judge whether *doc* is relevant to *claim*.

        Returns ``PrecisionLabel.UNRELEVANT`` (score 0.0) if all retries fail.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(claim, doc)},
        ]

        try:
            return self.client.call_with_retry(messages, self._parse_response)
        except Exception as exc:
            logger.error(
                "LiteLLMPrecisionJudge: evaluation failed after retries: %s", exc
            )
            return PrecisionResult(
                label=PrecisionLabel.UNRELEVANT,
                score=0.0,
                reason=f"Precision judge failed: {exc}",
            )
