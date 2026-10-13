"""
Decision-model evidence judge.

Drop-in alternative to :class:`~acclaim.judges.litellm.LiteLLMJudge` that
uses a decision model (see :mod:`acclaim.decision_client`) instead of a chat
LLM. Decision models return calibrated probabilities rather than text, so
no JSON parsing is needed and the ``reason`` is a summary of the
probabilities.

All modes share one prompt that mirrors the chat judge's default
``system_prompt`` (label definitions and examples). Three question modes
are supported:

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

# The prompt mirrors the chat judge's `judge.system_prompt` in
# config/default.yaml (also used by library_evals/judges/config.yaml), minus
# the JSON output format, so chat and decision judges are compared on the same
# label definitions and examples. All three modes share it; only the question
# shape differs. Keep the two in sync.
_LABEL_CRITERIA: dict[SupportLabel, str] = {
    SupportLabel.SUPPORTED: (
        "The document explicitly states or strongly implies the claim is true."
    ),
    SupportLabel.REFUTED: "The document contains a direct, clear contradiction of the claim.",
    SupportLabel.UNCLEAR: (
        "The document does not address the claim, is off-topic, or the evidence is ambiguous."
    ),
}

_INSTRUCTIONS = f"""You are an expert fact-checker. Determine whether the provided claim is supported by the evidence document.

Label definitions:
  SUPPORTED — {_LABEL_CRITERIA[SupportLabel.SUPPORTED]}
  REFUTED   — {_LABEL_CRITERIA[SupportLabel.REFUTED]}
  UNCLEAR   — {_LABEL_CRITERIA[SupportLabel.UNCLEAR]}

Important: use REFUTED only for direct contradictions. If the document simply does not mention the claim or is about a different topic, use UNCLEAR.

---

Example 1 — SUPPORTED

Document: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889."
Claim: "The Eiffel Tower is located in Paris."
Label: SUPPORTED. The document explicitly states the Eiffel Tower is in Paris, France.

---

Example 2 — REFUTED

Document: "It was announced that the first concert would be pushed back by eight days to July 16, and three other July dates would be rescheduled for March 2010."
Claim: "The first concert was rescheduled for July 13."
Label: REFUTED. The document states the concert was moved to July 16, directly contradicting the claim of July 13.

---

Example 3 — UNCLEAR

Document: "India is the 7th largest producer of hydroelectric power in the world. As of April 2017, India's installed utility-scale hydroelectric capacity was 44,594 MW."
Claim: "The Colorado River flows through the Grand Canyon."
Label: UNCLEAR. The document is about hydroelectric power in India and does not address the claim about the Colorado River."""

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
                    "instructions": _INSTRUCTIONS,
                    "criteria": {label.value: text for label, text in _LABEL_CRITERIA.items()},
                }
            }
        if self.mode == "score":
            return {
                "verdict": {
                    "type": "score",
                    "instructions": (
                        f"{_INSTRUCTIONS}\n\nPlace the document on a scale from "
                        "contradicting the claim (REFUTED) to supporting it (SUPPORTED)."
                    ),
                    "criteria": [
                        f"{label.value}: {_LABEL_CRITERIA[label]}" for label in _SCORE_LEVELS
                    ],
                }
            }
        return {
            "supported": {
                "type": "noul",
                "instructions": f"{_INSTRUCTIONS}\n\nQuestion: is the label SUPPORTED?",
                "criteria": {
                    "true": _LABEL_CRITERIA[SupportLabel.SUPPORTED],
                    "false": (
                        "The label is REFUTED or UNCLEAR: the document contradicts the "
                        "claim, does not address it, is off-topic, or the evidence is "
                        "ambiguous."
                    ),
                },
            },
            "refuted": {
                "type": "noul",
                "instructions": f"{_INSTRUCTIONS}\n\nQuestion: is the label REFUTED?",
                "criteria": {
                    "true": _LABEL_CRITERIA[SupportLabel.REFUTED],
                    "false": (
                        "The label is SUPPORTED or UNCLEAR: the document supports the "
                        "claim, does not address it, is off-topic, or the evidence is "
                        "ambiguous."
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
