"""
LLM-based citation aligner.

For atomic claims that have ``span=(-1, -1)``, alignment is done by asking
the LLM which citation numbers support each claim directly — it already has
full context (the whole answer) from generating the claims, so this replaces
:class:`~acclaim.alignment.jaccard.JaccardCitationAligner`'s statistical,
after-the-fact token-overlap matching with the model's own attribution.

``Claim.span`` is still backfilled for callers/tooling that expect it (e.g.
human-annotation sheets), but via a citation-number-*constrained* sentence
match: the citation number the model already identified narrows the search
to just the sentence(s) that actually carry that marker, instead of scoring
every sentence in the answer the way ``JaccardCitationAligner`` does.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ._scoring import token_weights as _token_weights
from ._scoring import tokenize as _tokenize
from ._scoring import weighted_claim_overlap as _weighted_claim_overlap
from .base import CitationAligner
from ..citations.parser import extract_doc_ids_for_span, find_citation_markers, parse_citation_markers
from ..data_models import AlignedClaim, Claim
from ..llm_client import LiteLLMClient
from ..text_utils import split_sentences_with_spans

logger = logging.getLogger(__name__)


class _ClaimAlignment(BaseModel):
    """One claim's citation attribution, as reported by the LLM."""

    claim_index: int
    citation_numbers: list[int] = Field(default_factory=list)


class _AlignmentResponse(BaseModel):
    """Pydantic schema for the structured LLM alignment output."""

    alignments: list[_ClaimAlignment]


class LLMCitationAligner(CitationAligner):
    """
    Align atomic (span-less) claims by asking an LLM which citation numbers
    support each one, in a single batched call per :meth:`align` invocation.

    Usage::

        aligner = LLMCitationAligner(model="gpt-4o-mini")
        aligned = aligner.align(claims, answer_text, citation_to_doc)
    """

    _SYSTEM_PROMPT = """You are an expert at attributing claims to their supporting citations.

You will be given a numbered list of atomic claims extracted from an answer, \
and the original answer text containing inline [N] citation markers.

For each claim, identify which citation number(s) in the original text \
genuinely support that specific claim's content. Only use citation numbers \
that literally appear as [N] markers in the provided text; never invent a \
number. If a claim is not supported by any specific citation in the text, \
return an empty list for it.

Two rules that are easy to get wrong:

1. Multiple claims are often extracted from ONE source sentence that carries \
ONE citation group. When that happens, EVERY claim drawn from that sentence \
should normally get that same citation — including claims about background \
or definitional details (e.g. "X stands for Y", "X happened in year Z", "A \
is B's title"), not just the claim that states the sentence's main or most \
prominent fact. Do not drop a claim's citation just because a sibling claim \
from the same sentence looks like the more important fact — check every \
other claim drawn from the same sentence and be consistent with them.

2. The opposite mistake: when a sentence lists several DISTINCT entities, \
each with its OWN citation marker (e.g. "Entity A [1], Entity B [2], Entity \
C [3]."), a claim about one entity should get ONLY that entity's own \
marker(s) — never the union of every marker in the sentence.

Respond with ONLY a JSON object of this shape, one entry per claim, in the \
same order given:
{"alignments": [{"claim_index": 0, "citation_numbers": [1, 3]}, {"claim_index": 1, "citation_numbers": []}]}

---

Example 1 — shared citation across sub-clauses of one sentence (rule 1)

Answer: "Marie Curie discovered radium and polonium in 1898 [2]."
Claims:
0: Marie Curie discovered radium.
1: Marie Curie discovered polonium.
2: The discoveries were made in 1898.
{"alignments": [{"claim_index": 0, "citation_numbers": [2]}, {"claim_index": 1, "citation_numbers": [2]}, {"claim_index": 2, "citation_numbers": [2]}]}

---

Example 2 — distinct entities keep only their own marker (rule 2)

Answer: "Pink Panther diamond [1], Darya-ye Noor diamond [1], DeYoung Red Diamond [2]."
Claims:
0: The Pink Panther is a diamond.
1: The Darya-ye Noor is a diamond.
2: The DeYoung Red Diamond is a diamond.
{"alignments": [{"claim_index": 0, "citation_numbers": [1]}, {"claim_index": 1, "citation_numbers": [1]}, {"claim_index": 2, "citation_numbers": [2]}]}

---

Example 3 — genuinely uncited claim stays empty

Answer: "The band recorded the album in London [1]. Some critics consider it their best work."
Claims:
0: The album was recorded in London.
1: Critics consider it the band's best work.
{"alignments": [{"claim_index": 0, "citation_numbers": [1]}, {"claim_index": 1, "citation_numbers": []}]}"""

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
        **kwargs: Any,
    ) -> None:
        """
        Initialise the LLM aligner.

        Args:
            model:         LiteLLM model identifier.
            api_base:      Custom API endpoint (for vLLM, etc.).
            api_key:       Optional API key.
            temperature:   Sampling temperature.
            max_tokens:    Maximum response tokens (needs headroom for
                           multi-claim JSON output on answers with many claims).
            max_retries:   JSON parse / validation retry attempts.
            thinking:      Whether to enable reasoning/thinking on the model.
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
            thinking=thinking,
            **kwargs,
        )
        logger.debug("LLMCitationAligner initialised: model=%s api_base=%s", model, api_base)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, claims: list[Claim], text: str) -> str:
        numbered = "\n".join(f"{i}: {claim.text}" for i, claim in enumerate(claims))
        return (
            f"Claims:\n{numbered}\n\n"
            f"Answer text:\n{text}\n\n"
            "Provide the citation alignment as a JSON object."
        )

    def _parse_response(self, raw: str, n_claims: int) -> dict[int, list[int]]:
        cleaned = LiteLLMClient.strip_code_fences(raw)
        data = json.loads(cleaned)
        resp = _AlignmentResponse(**data)

        by_index = {a.claim_index: a.citation_numbers for a in resp.alignments}
        missing = set(range(n_claims)) - set(by_index)
        if missing:
            raise ValueError(f"Alignment response missing claim indices: {sorted(missing)}")
        return by_index

    def _sentence_index(
        self, text: str
    ) -> list[tuple[tuple[int, int], set[int], set[str]]]:
        """Each sentence's span, the citation numbers it carries, and its tokens."""
        index = []
        for sent in split_sentences_with_spans(text):
            if not sent.text:
                continue
            span = (sent.start, sent.end)
            numbers = {n for marker in find_citation_markers(text, *span) for n in marker.numbers}
            index.append((span, numbers, _tokenize(sent.text)))
        return index

    def _best_span_for_numbers(
        self,
        numbers: set[int],
        claim_tokens: set[str],
        sentence_index: list[tuple[tuple[int, int], set[int], set[str]]],
        weights: dict[str, float],
    ) -> tuple[int, int] | None:
        """
        Span of the sentence carrying *numbers*, constrained to candidates
        that actually contain at least one of them (see module docstring).

        With exactly one candidate sentence (the common case, since a given
        citation number usually appears once), it wins outright — no
        scoring needed. With multiple candidates (a repeated citation
        number), tie-break via the same weighted-Jaccard scoring
        ``JaccardCitationAligner`` uses, but only among this pre-filtered
        candidate set — irrelevant sentences can't compete at all, which is
        what fixes the wrong-sentence-wins failure mode.
        """
        candidates = [
            (span, sent_tokens)
            for span, sent_numbers, sent_tokens in sentence_index
            if sent_numbers & numbers
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        best_score = -1.0
        best_span = candidates[0][0]
        for span, sent_tokens in candidates:
            score = _weighted_claim_overlap(claim_tokens, sent_tokens, weights)
            if score > best_score:
                best_score = score
                best_span = span
        return best_span

    # ------------------------------------------------------------------
    # CitationAligner interface
    # ------------------------------------------------------------------

    def align(
        self,
        claims: list[Claim],
        text: str,
        citation_to_doc: dict[int, str],
    ) -> list[AlignedClaim]:
        """
        Align *claims* to citations by asking the LLM which citation
        numbers support each one, then backfilling ``Claim.span`` via a
        citation-constrained sentence match.

        Args:
            claims:          Claims (typically with ``span=(-1,-1)``).
            text:            Full original answer text.
            citation_to_doc: Mapping from citation number to document ID.

        Returns:
            One :class:`~acclaim.data_models.AlignedClaim` per claim.
        """
        # Defensive passthrough for claims that already carry a real span
        # (e.g. sentence-level claims erroneously routed here) — identical
        # to JaccardCitationAligner's behavior.
        needs_alignment = [(i, c) for i, c in enumerate(claims) if c.span == (-1, -1)]
        aligned: list[AlignedClaim | None] = [None] * len(claims)
        for i, claim in enumerate(claims):
            if claim.span != (-1, -1):
                doc_ids = extract_doc_ids_for_span(claim.span, text, citation_to_doc)
                aligned[i] = AlignedClaim(claim=claim, citation_doc_ids=doc_ids)

        if not needs_alignment:
            return [a for a in aligned if a is not None]

        batch_claims = [c for _, c in needs_alignment]
        by_index = self._call_llm(batch_claims, text)

        valid_numbers = set(parse_citation_markers(text)) & set(citation_to_doc)
        sentence_index = self._sentence_index(text)
        claim_token_sets = [_tokenize(c.text) for c in batch_claims]
        weights = _token_weights(claim_token_sets)

        for local_i, (orig_i, claim) in enumerate(needs_alignment):
            numbers = set(by_index.get(local_i, []))
            dropped = numbers - valid_numbers
            if dropped:
                logger.warning(
                    "LLMCitationAligner: dropping fabricated citation number(s) %s for claim %r",
                    sorted(dropped),
                    claim.text,
                )
            numbers &= valid_numbers

            seen: dict[str, None] = {}
            for n in sorted(numbers):
                seen[citation_to_doc[n]] = None
            doc_ids = list(seen)

            span = (-1, -1)
            if numbers:
                found_span = self._best_span_for_numbers(
                    numbers, claim_token_sets[local_i], sentence_index, weights
                )
                if found_span is not None:
                    span = found_span

            new_claim = Claim(text=claim.text, span=span)
            aligned[orig_i] = AlignedClaim(claim=new_claim, citation_doc_ids=doc_ids)

        return [a for a in aligned if a is not None]

    def _call_llm(self, batch_claims: list[Claim], text: str) -> dict[int, list[int]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(batch_claims, text)},
        ]
        try:
            return self.client.call_with_retry(
                messages, lambda raw: self._parse_response(raw, len(batch_claims))
            )
        except Exception as exc:
            logger.error("LLMCitationAligner: alignment failed after retries: %s", exc)
            return {}
