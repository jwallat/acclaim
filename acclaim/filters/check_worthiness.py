"""
LLM-based check-worthiness (citation-worthiness) claim filter.

The LLM *classifies* each claim into a fixed :class:`ClaimCategory`
taxonomy; whether a category is kept or dropped is a configuration choice
(``drop_categories``), not the model's. The taxonomy separates two
questions from the claim-detection literature:

1. Is it a verifiable factual claim at all? (ClaimBuster's non-factual vs.
   factual split; VeriScore's verifiable claims; Full Fact's "not a claim")
2. Does it need a source? (Wikipedia's *Citation Needed* taxonomy, where
   "common knowledge" is a reason a citation is not needed)

References:
    Hassan et al. 2017, "Toward Automated Fact-Checking: Detecting
    Check-worthy Factual Claims by ClaimBuster" (KDD).
    Konstantinovskiy et al. 2021, "Toward Automated Factchecking: Developing
    an Annotation Schema and Benchmark for Consistent Automated Claim
    Detection" (DTRAP).
    Redi et al. 2019, "Citation Needed: A Taxonomy and Algorithmic Assessment
    of Wikipedia's Verifiability" (WWW).
    Song et al. 2024, "VeriScore: Evaluating the factuality of verifiable
    claims in long-form text generation" (Findings of EMNLP).
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel

from .base import ClaimFilter, FilterDecision
from ..data_models import Claim
from ..llm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class ClaimCategory(str, Enum):
    """Claim types used by :class:`LLMCheckWorthinessFilter`."""

    # Not check-worthy by default
    NOT_A_CLAIM = "NOT_A_CLAIM"
    SUBJECTIVE = "SUBJECTIVE"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    COMMON_KNOWLEDGE = "COMMON_KNOWLEDGE"
    # Check-worthy by default
    QUANTITY = "QUANTITY"
    EVENT_OR_HISTORICAL = "EVENT_OR_HISTORICAL"
    ATTRIBUTION_OR_QUOTE = "ATTRIBUTION_OR_QUOTE"
    CAUSAL_OR_CORRELATION = "CAUSAL_OR_CORRELATION"
    SCIENTIFIC_OR_TECHNICAL = "SCIENTIFIC_OR_TECHNICAL"
    RULE_OR_POLICY = "RULE_OR_POLICY"
    PREDICTION = "PREDICTION"
    OTHER_FACTUAL = "OTHER_FACTUAL"


DEFAULT_DROP_CATEGORIES: tuple[ClaimCategory, ...] = (
    ClaimCategory.NOT_A_CLAIM,
    ClaimCategory.SUBJECTIVE,
    ClaimCategory.UNDERSPECIFIED,
    ClaimCategory.COMMON_KNOWLEDGE,
)


class _ClaimClassification(BaseModel):
    """One claim's category, as reported by the LLM."""

    claim_index: int
    category: ClaimCategory
    reason: str = ""


class _ClassificationResponse(BaseModel):
    """Pydantic schema for the structured LLM classification output."""

    decisions: list[_ClaimClassification]


class LLMCheckWorthinessFilter(ClaimFilter):
    """
    Drop claims that do not need citation-based verification, by asking an
    LLM to classify each claim in a single batched call per :meth:`decide`.

    Usage::

        f = LLMCheckWorthinessFilter(model="gpt-4o-mini")
        kept = f.filter(claims, answer_text, question)
    """

    name = "check_worthiness"

    _SYSTEM_PROMPT = """You classify claims extracted from an answer by whether they need to be verified against a source.

You will be given the question (if available), the full answer text, and a numbered list of claims extracted from that answer. Assign each claim exactly ONE category:

Claims that do NOT need verification:
- NOT_A_CLAIM: transitions, structural or meta-statements about the answer itself or about the provided sources, questions, filler, refusals (e.g. "Here are the key points.", "Let's look at the details.", "The answer discusses three options.", "The provided documents do not mention who won.", "No information is given about the release date.").
- SUBJECTIVE: ONLY claims whose entire content is a personal opinion, taste, value judgment, advice, or speculation about how hypothetical people might react, so that no source could confirm or refute them (e.g. "It is a beautiful city.", "You should consider both options.", "This is the best way to spend a weekend.", "These chips go well with salsa.", "Collectors may be interested in the letters."). Do NOT use SUBJECTIVE merely because a claim is hedged ("may", "can", "arguably"), phrased as advice ("should", "it is important to"), or contains an evaluative word; see rule 3.
- UNDERSPECIFIED: too vague to verify, even with the answer as context: the claim contains no concrete time, place, amount, property, or event that a source could confirm, or it is an exaggeration with no concrete content (e.g. "It has many benefits.", "Things changed a lot.", "The service has limited options.", "Nobody ever wins these contests."). NOT underspecified: hedged generalizations with concrete content ("The season typically starts in mid-September."), superlatives or rankings a source could confirm ("It is one of the largest banks in Europe."), and descriptions of a specific place, work, or entity ("The novel is set in a fishing village.").
- COMMON_KNOWLEDGE: facts that virtually any adult, anywhere, knows without thinking or can observe directly (e.g. "Paris is in France.", "Water freezes at low temperatures.", "The sun rises in the east."). NOT common knowledge: the location of a lesser-known venue, neighborhood, or landmark ("The Allianz Arena is in Munich."), or the occupation or nationality of anyone who is not world-famous ("Kelsey Grammer is an actor.").

Claims that DO need verification:
- QUANTITY: numbers, statistics, dates, amounts, measurements, rankings.
- EVENT_OR_HISTORICAL: specific events, actions, or happenings involving particular entities.
- ATTRIBUTION_OR_QUOTE: what a specific person, organization, study, or document said, found, or claims.
- CAUSAL_OR_CORRELATION: X causes, affects, prevents, or is associated with Y.
- SCIENTIFIC_OR_TECHNICAL: domain or expert knowledge, including definitions of technical terms.
- RULE_OR_POLICY: laws, regulations, policies, or how an institution or system operates.
- PREDICTION: forecasts, projections, or expected future developments.
- OTHER_FACTUAL: a specific, verifiable factual claim that fits none of the above.

Rules:
1. Judge whether the claim NEEDS verification, NOT whether it is true. A false claim can still be common knowledge in form (and vice versa); do not use your own belief about its truth.
2. Use the question and the answer as context. What counts as common knowledge depends on the topic: a fact that is obvious to everyone is COMMON_KNOWLEDGE, but a specific detail an ordinary reader would need to look up is not. If you are unsure whether most adults would know it, it is NOT common knowledge.
3. Before choosing SUBJECTIVE, strip hedges, advice framing, and evaluative words, and ask what the claim still asserts about the world. If a source could state, confirm, or contradict that remainder, classify the claim by the remainder, not as SUBJECTIVE. For example:
   - "Buyers should watch out for counterfeit versions of the product." asserts counterfeits exist -> OTHER_FACTUAL.
   - "The warranty has a catch." asserts the warranty contains a restrictive condition -> RULE_OR_POLICY.
   - "Teamwork lets wolves make up for individual weaknesses." asserts an effect of cooperative hunting -> CAUSAL_OR_CORRELATION.
   - "Some experts see the law as a threat to privacy." asserts experts hold this view -> ATTRIBUTION_OR_QUOTE.
   Only when nothing checkable remains (pure taste, praise, advice, or speculation about hypothetical reactions) is the claim SUBJECTIVE.
   This rule only decides between SUBJECTIVE and the categories that need verification. If the remainder has no concrete content (see UNDERSPECIFIED), choose UNDERSPECIFIED.
4. Statements about the content of a specific book, film, show, or game (its plot, setting, or what a character is like) can be checked against that work. Classify them as OTHER_FACTUAL or EVENT_OR_HISTORICAL, never as SUBJECTIVE or UNDERSPECIFIED.
5. When in doubt, choose one of the categories that need verification.

Respond with ONLY a JSON object of this shape, one entry per claim, in the same order given:
{"decisions": [{"claim_index": 0, "category": "QUANTITY", "reason": "<short reason>"}]}

---

Example

Question: "What is the Eiffel Tower known for?"
Answer: "Great question! The Eiffel Tower is in Paris [1]. It was completed in 1889 and stands 330 m tall [1]. Gustave Eiffel's company designed it [2]. It is arguably the most beautiful landmark in Europe."
Claims:
0: This is a great question.
1: The Eiffel Tower is in Paris.
2: The Eiffel Tower was completed in 1889.
3: The Eiffel Tower stands 330 m tall.
4: Gustave Eiffel's company designed the Eiffel Tower.
5: The Eiffel Tower is arguably the most beautiful landmark in Europe.
{"decisions": [{"claim_index": 0, "category": "NOT_A_CLAIM", "reason": "Conversational filler."}, {"claim_index": 1, "category": "COMMON_KNOWLEDGE", "reason": "Widely known location of a famous landmark."}, {"claim_index": 2, "category": "QUANTITY", "reason": "Specific completion date."}, {"claim_index": 3, "category": "QUANTITY", "reason": "Specific measurement."}, {"claim_index": 4, "category": "ATTRIBUTION_OR_QUOTE", "reason": "Attributes the design to a specific company."}, {"claim_index": 5, "category": "SUBJECTIVE", "reason": "Aesthetic judgment."}]}

---

Example — hedged or advisory phrasing with a factual core

Question: "Are detox teas worth it?"
Answer: "Detox teas often contain laxatives such as senna [1]. Buyers should be careful, since some brands' money-back guarantees have a catch [2]. Honestly, I think they are a waste of money. You should talk to a doctor first."
Claims:
0: Detox teas often contain laxatives such as senna.
1: Buyers should be careful with detox teas.
2: Some detox tea brands' money-back guarantees have a catch.
3: Detox teas are a waste of money.
4: You should talk to a doctor before using detox teas.
{"decisions": [{"claim_index": 0, "category": "SCIENTIFIC_OR_TECHNICAL", "reason": "Specific ingredient content."}, {"claim_index": 1, "category": "SUBJECTIVE", "reason": "Generic advice with no checkable remainder."}, {"claim_index": 2, "category": "RULE_OR_POLICY", "reason": "Asserts guarantees contain restrictive conditions, which a source can confirm."}, {"claim_index": 3, "category": "SUBJECTIVE", "reason": "Personal value judgment."}, {"claim_index": 4, "category": "SUBJECTIVE", "reason": "Generic advice."}]}"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 3,
        thinking: bool = False,
        system_prompt: str | None = None,
        drop_categories: Iterable[str | ClaimCategory] = DEFAULT_DROP_CATEGORIES,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the check-worthiness filter.

        Args:
            model:           LiteLLM model identifier.
            api_base:        Custom API endpoint (for vLLM, etc.).
            api_key:         Optional API key.
            temperature:     Sampling temperature.
            max_tokens:      Maximum response tokens (needs headroom for
                             multi-claim JSON output on answers with many claims).
            max_retries:     JSON parse / validation retry attempts.
            thinking:        Whether to enable reasoning/thinking on the model.
            system_prompt:   Override the default system prompt.
            drop_categories: :class:`ClaimCategory` names whose claims are
                             dropped. Defaults to :data:`DEFAULT_DROP_CATEGORIES`.
            **kwargs:        Forwarded to ``litellm.completion()``.

        Raises:
            ValueError: If *drop_categories* contains an unknown category.
        """
        self.system_prompt = system_prompt or self._SYSTEM_PROMPT
        self.drop_categories = _parse_categories(drop_categories)
        self.client = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
            max_retries=max_retries,
            thinking=thinking,
            **kwargs,
        )
        logger.debug("LLMCheckWorthinessFilter initialised: model=%s api_base=%s", model, api_base)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self, claims: list[Claim], answer: str, question: str | None
    ) -> str:
        numbered = "\n".join(f"{i}: {claim.text}" for i, claim in enumerate(claims))
        question_part = f"Question:\n{question}\n\n" if question else ""
        return (
            f"{question_part}"
            f"Answer text:\n{answer}\n\n"
            f"Claims:\n{numbered}\n\n"
            "Provide the classification as a JSON object."
        )

    def _parse_response(
        self, raw: str, n_claims: int
    ) -> dict[int, _ClaimClassification]:
        cleaned = LiteLLMClient.strip_code_fences(raw)
        data = json.loads(cleaned)
        resp = _ClassificationResponse(**data)

        by_index = {d.claim_index: d for d in resp.decisions}
        missing = set(range(n_claims)) - set(by_index)
        if missing:
            raise ValueError(f"Classification response missing claim indices: {sorted(missing)}")
        return by_index

    # ------------------------------------------------------------------
    # ClaimFilter interface
    # ------------------------------------------------------------------

    def decide(
        self,
        claims: list[Claim],
        answer: str,
        question: str | None = None,
    ) -> list[FilterDecision]:
        """
        Classify each claim and keep it unless its category is in
        :attr:`drop_categories`. Fails open: if classification fails after
        all retries, every claim is kept with ``category=None``.
        """
        if not claims:
            return []

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(claims, answer, question)},
        ]
        try:
            by_index = self.client.call_with_retry(
                messages, lambda raw: self._parse_response(raw, len(claims))
            )
        except Exception as exc:
            logger.error(
                "LLMCheckWorthinessFilter: classification failed after retries, "
                "keeping all claims: %s",
                exc,
            )
            return [
                FilterDecision(claim=c, keep=True, filter_name=self.name) for c in claims
            ]

        decisions = []
        for i, claim in enumerate(claims):
            c = by_index[i]
            decisions.append(
                FilterDecision(
                    claim=claim,
                    keep=c.category not in self.drop_categories,
                    filter_name=self.name,
                    category=c.category.value,
                    reason=c.reason or None,
                )
            )
        return decisions


def _parse_categories(names: Iterable[str | ClaimCategory]) -> frozenset[ClaimCategory]:
    valid = [c.value for c in ClaimCategory]
    parsed = set()
    for name in names:
        value = name.value if isinstance(name, ClaimCategory) else str(name).upper()
        if value not in valid:
            raise ValueError(f"Unknown claim category {name!r}. Available: {valid}")
        parsed.add(ClaimCategory(value))
    return frozenset(parsed)
