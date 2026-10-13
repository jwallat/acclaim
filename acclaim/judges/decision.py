"""
Decision-model evidence judge.

Drop-in alternative to :class:`~acclaim.judges.litellm.LiteLLMJudge` that
uses a decision model (see :mod:`acclaim.decision_client`) instead of a chat
LLM. Decision models return calibrated probabilities rather than text, so
no JSON parsing is needed and the ``reason`` is a summary of the
probabilities.

Three question modes are supported:

  choice — one ``choice`` question over SUPPORTED / REFUTED / UNCLEAR.
  noul   — two independent yes/no questions ("supported?", "contradicted?")
           combined with configurable thresholds.
  score  — one ``score`` question on the ordinal scale
           REFUTED < UNCLEAR < SUPPORTED; the label is the most probable level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from .base import EvidenceJudge
from ..data_models import Claim, Document, SupportLabel, SupportResult
from ..decision_client import DecisionClient, DecisionResponse

logger = logging.getLogger(__name__)

DECISION_MODES = ("choice", "noul", "score")
STATE_FORMATS = ("structured", "text")
DEFAULT_THRESHOLDS: dict[str, float] = {"supported": 0.5, "refuted": 0.5}

_INSTRUCTIONS = (
    "Judge whether the evidence documents support the claim. "
    "Base the judgment only on the evidence, not on outside knowledge."
)

_LABEL_CRITERIA: dict[SupportLabel, str] = {
    SupportLabel.SUPPORTED: (
        "The evidence explicitly states or strongly implies that the claim is true."
    ),
    SupportLabel.REFUTED: (
        "The evidence contains a direct, clear contradiction of the claim. "
        "Use only for direct contradictions, not when the evidence is silent "
        "or about a different topic."
    ),
    SupportLabel.UNCLEAR: (
        "The evidence does not address the claim, is off-topic, or is ambiguous."
    ),
}

# Choice mode gets a fuller definition: explicit rules plus worked examples.
# The examples are deliberately not taken from library_evals/judges/pairs.jsonl.
_CHOICE_INSTRUCTIONS = """You are an expert fact-checker. Decide whether the evidence documents support the claim.

Rules:
1. Use only the evidence. Do not use outside knowledge, even if you know the claim is true or false.
2. SUPPORTED requires every part of the claim (who, what, when, where, how many) to be stated in or directly implied by the evidence. Paraphrases count. If any part is missing, the claim is not SUPPORTED.
3. Evidence about a related but different entity, event, time, or version (e.g. a sequel, a remake, a different year, a re-election instead of a first election) does not support the claim.
4. Evidence that only makes the claim plausible, or from which the claim could only be guessed, does not support it.
5. REFUTED requires the evidence to state something about the same entity, event, and time that cannot be true together with the claim (e.g. a different date, number, place, or person). Missing information is never a contradiction: if the evidence is silent or about something else, the label is UNCLEAR.

Examples:

Evidence: "The Golden Gate Bridge opened to vehicle traffic on May 27, 1937, after four years of construction."
Claim: "The Golden Gate Bridge opened in 1937."
Label: SUPPORTED (the year is stated).

Evidence: "Marie Curie was born in Warsaw, then part of the Russian Empire, in 1867."
Claim: "Marie Curie was born in Paris."
Label: REFUTED (the evidence gives a different birthplace for the same person).

Evidence: "The Nile flows north through eleven countries before emptying into the Mediterranean Sea."
Claim: "The Nile is the longest river in Africa and flows through eleven countries."
Label: UNCLEAR (only the second part is supported; the length is not mentioned).

Evidence: "Apollo 11 landed on the Moon on July 20, 1969."
Claim: "Apollo 12 landed on the Moon in November 1969."
Label: UNCLEAR (the evidence is about a different mission; it neither states nor contradicts the claim)."""

_CHOICE_CRITERIA: dict[SupportLabel, str] = {
    SupportLabel.SUPPORTED: (
        "Every part of the claim is stated in or directly implied by the evidence."
    ),
    SupportLabel.REFUTED: (
        "The evidence states something about the same entity, event, and time "
        "that cannot be true together with the claim. Missing information is "
        "never a contradiction."
    ),
    SupportLabel.UNCLEAR: (
        "Anything else: the evidence is silent on the claim or on part of it, is "
        "about a related but different entity, event, or time, only makes the "
        "claim plausible, or is ambiguous."
    ),
}

# Ordinal order used by score mode (index 0 = lowest).
_SCORE_LEVELS: tuple[SupportLabel, ...] = (
    SupportLabel.REFUTED,
    SupportLabel.UNCLEAR,
    SupportLabel.SUPPORTED,
)


@dataclass(frozen=True)
class DecisionJudgment:
    """A :class:`SupportResult` plus the decision-model details behind it."""

    result: SupportResult
    probabilities: dict[str, float]
    """Distribution over SUPPORTED / REFUTED / UNCLEAR (derived for ``noul``)."""
    response: DecisionResponse | None = None


class DecisionJudge(EvidenceJudge):
    """
    Evidence judge backed by a decision model via LiteLLM.

    Usage::

        judge = DecisionJudge(model="openrouter/typesafe/jev-1.13", mode="choice")
        result = judge.evaluate(claim, docs)
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        mode: str = "choice",
        thresholds: Mapping[str, float] | None = None,
        state_format: str = "structured",
        **kwargs: Any,
    ) -> None:
        """
        Initialise the decision judge.

        Args:
            model:        LiteLLM decision-model identifier.
            api_base:     Custom API endpoint.
            api_key:      Optional API key.
            max_retries:  Attempts before falling back to UNCLEAR.
            mode:         ``"choice"``, ``"noul"`` or ``"score"``.
            thresholds:   ``noul`` mode only: ``{"supported": t, "refuted": t}``
                          probability cut-offs. Missing keys default to 0.5.
            state_format: ``"structured"`` sends ``{"claim", "evidence"}`` as a
                          JSON object; ``"text"`` sends one formatted string.
            **kwargs:     Forwarded to :class:`~acclaim.decision_client.DecisionClient`.

        Raises:
            ValueError: On an unknown mode, state format or threshold key.
        """
        if mode not in DECISION_MODES:
            raise ValueError(f"Unknown decision mode {mode!r}. Available: {list(DECISION_MODES)}")
        if state_format not in STATE_FORMATS:
            raise ValueError(
                f"Unknown state_format {state_format!r}. Available: {list(STATE_FORMATS)}"
            )
        unknown = set(thresholds or {}) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise ValueError(
                f"Unknown threshold keys {sorted(unknown)}. Available: {list(DEFAULT_THRESHOLDS)}"
            )

        self.mode = mode
        self.state_format = state_format
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.client = DecisionClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_retries=max_retries,
            **kwargs,
        )
        logger.debug(
            "DecisionJudge initialised: model=%s mode=%s state_format=%s",
            model,
            mode,
            state_format,
        )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_state(self, claim: Claim, docs: list[Document]) -> str | dict[str, Any]:
        if self.state_format == "structured":
            return {
                "claim": claim.text,
                "evidence": [{"doc_id": d.doc_id, "text": d.text} for d in docs],
            }
        evidence = "\n\n".join(f"Document {d.doc_id}:\n{d.text}" for d in docs)
        return f"Evidence Documents:\n{evidence}\n\nClaim to verify:\n{claim.text}"

    def _build_questions(self) -> dict[str, dict[str, Any]]:
        if self.mode == "choice":
            return {
                "verdict": {
                    "type": "choice",
                    "instructions": _CHOICE_INSTRUCTIONS,
                    "criteria": {label.value: text for label, text in _CHOICE_CRITERIA.items()},
                }
            }
        if self.mode == "score":
            return {
                "verdict": {
                    "type": "score",
                    "instructions": (
                        f"{_INSTRUCTIONS} Place the evidence on a scale from "
                        "contradicting the claim to supporting it."
                    ),
                    "criteria": [
                        f"{label.value}: {_LABEL_CRITERIA[label]}" for label in _SCORE_LEVELS
                    ],
                }
            }
        return {
            "supported": {
                "type": "noul",
                "instructions": "Do the evidence documents support the claim?",
                "criteria": {
                    "true": _LABEL_CRITERIA[SupportLabel.SUPPORTED],
                    "false": (
                        "The evidence does not mention the claim, is about a different "
                        "topic, is ambiguous, or contradicts the claim."
                    ),
                },
            },
            "refuted": {
                "type": "noul",
                "instructions": "Do the evidence documents directly contradict the claim?",
                "criteria": {
                    "true": _LABEL_CRITERIA[SupportLabel.REFUTED],
                    "false": (
                        "The evidence agrees with the claim, does not mention it, is "
                        "about a different topic, or is ambiguous."
                    ),
                },
            },
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _answer(response: DecisionResponse, name: str, expected_type: str) -> dict[str, Any]:
        answer = response.answers.get(name)
        if answer is None:
            raise ValueError(f"Decision response missing answer {name!r}")
        if answer.get("type") != expected_type:
            raise ValueError(
                f"Answer {name!r} has type {answer.get('type')!r}, expected {expected_type!r}"
            )
        return answer

    def _parse_response(self, response: DecisionResponse) -> DecisionJudgment:
        if self.mode == "choice":
            answer = self._answer(response, "verdict", "choice")
            probs = {label.value: float(answer["probabilities"].get(label.value, 0.0))
                     for label in SupportLabel}
            label = SupportLabel(answer["choice"])
            reason = "choice: " + _format_probs(probs)
        elif self.mode == "score":
            answer = self._answer(response, "verdict", "score")
            level_probs = answer["probabilities"]
            probs = {level.value: float(level_probs.get(str(i), 0.0))
                     for i, level in enumerate(_SCORE_LEVELS)}
            label = SupportLabel(max(probs, key=probs.__getitem__))
            reason = f"score={float(answer['score']):.2f}: " + _format_probs(probs)
        else:
            p_sup = float(self._answer(response, "supported", "noul")["noul"])
            p_ref = float(self._answer(response, "refuted", "noul")["noul"])
            # Same order as the decision rule: SUPPORTED wins first, then REFUTED.
            probs = {
                SupportLabel.SUPPORTED.value: p_sup,
                SupportLabel.REFUTED.value: (1 - p_sup) * p_ref,
                SupportLabel.UNCLEAR.value: (1 - p_sup) * (1 - p_ref),
            }
            if p_sup >= self.thresholds["supported"]:
                label = SupportLabel.SUPPORTED
            elif p_ref >= self.thresholds["refuted"]:
                label = SupportLabel.REFUTED
            else:
                label = SupportLabel.UNCLEAR
            reason = f"noul: P(supported)={p_sup:.2f} P(refuted)={p_ref:.2f}"

        return DecisionJudgment(
            result=SupportResult(label=label, confidence=probs[label.value], reason=reason),
            probabilities=probs,
            response=response,
        )

    # ------------------------------------------------------------------
    # EvidenceJudge interface
    # ------------------------------------------------------------------

    def judge_detailed(self, claim: Claim, docs: list[Document]) -> DecisionJudgment:
        """
        Like :meth:`evaluate`, but also returns the label distribution and the
        raw response (usage, cost, latency). The response is ``None`` when no
        call was made or all retries failed.
        """
        if not docs:
            return _unclear("No documents provided for this claim.")
        try:
            return self.client.ask_with_retry(
                self._build_state(claim, docs),
                self._build_questions(),
                self._parse_response,
            )
        except Exception as exc:
            logger.error("DecisionJudge: evaluation failed after retries: %s", exc)
            return _unclear(f"Judge failed: {exc}")

    def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult:
        """
        Judge whether *docs* support *claim*.

        Returns ``SupportLabel.UNCLEAR`` (confidence 0.0) when no documents
        are provided or if all retries fail.
        """
        return self.judge_detailed(claim, docs).result


def _format_probs(probs: Mapping[str, float]) -> str:
    return " ".join(
        f"{label}={p:.2f}" for label, p in sorted(probs.items(), key=lambda kv: -kv[1])
    )


def _unclear(reason: str) -> DecisionJudgment:
    return DecisionJudgment(
        result=SupportResult(label=SupportLabel.UNCLEAR, confidence=0.0, reason=reason),
        probabilities={},
    )
