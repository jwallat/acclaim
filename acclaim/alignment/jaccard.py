"""
Jaccard-similarity citation aligner.

For atomic claims that have ``span=(-1, -1)``, alignment is done by finding
the sentence in the original text most similar to the claim (using Jaccard
token-set similarity) and inheriting that sentence's ``[N]`` citations.
"""

from __future__ import annotations

import re
from collections import Counter

from .base import CitationAligner
from ..citations.parser import extract_doc_ids_for_span
from ..data_models import AlignedClaim, Claim
from ..text_utils import split_sentences_with_spans


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_weights(claim_token_sets: list[set[str]]) -> dict[str, float]:
    """
    Inverse-claim-frequency weight per token, computed over *this answer's*
    atomic claims (no external corpus needed).

    Tokens that recur across most claims from the same answer are usually
    the coreference-resolved subject (e.g. a repeated title or name) rather
    than the fact actually being asserted, so they get down-weighted; tokens
    specific to one or two claims (the predicate) get weighted up.
    """
    n_claims = len(claim_token_sets)
    df = Counter(token for tokens in claim_token_sets for token in tokens)
    return {token: (n_claims + 1) / (count + 1) for token, count in df.items()}


def _weighted_claim_overlap(
    claim_tokens: set[str], sent_tokens: set[str], weights: dict[str, float]
) -> float:
    """
    Weighted fraction of *claim_tokens*' weight mass found in *sent_tokens*.

    Deliberately normalizes by the claim's own weight mass rather than the
    union (unlike plain Jaccard): the true source sentence for an atomic
    claim is often long because it packs in several distinct facts, and a
    union-based denominator would unfairly penalize it just for containing
    content unrelated to *this* claim.
    """
    if not claim_tokens or not sent_tokens:
        return 0.0
    default_weight = 1.0
    claim_weight = sum(weights.get(t, default_weight) for t in claim_tokens)
    if claim_weight == 0:
        return 0.0
    overlap_weight = sum(weights.get(t, default_weight) for t in claim_tokens & sent_tokens)
    return overlap_weight / claim_weight


class JaccardCitationAligner(CitationAligner):
    """
    Align atomic (span-less) claims via Jaccard token similarity.

    Each claim is matched to the most similar sentence in the original
    answer text; the sentence's ``[N]`` markers are inherited as citations.

    Use this aligner with :class:`~acclaim.claims.atomic.AtomicClaimExtractor`.
    """

    def __init__(self, weighted: bool = True) -> None:
        """
        Args:
            weighted: If ``True`` (default), weight tokens by inverse
                claim-frequency within the same answer before computing
                Jaccard similarity — this corrects a bias where claims that
                restate a coreference-resolved subject (e.g. "the song
                'X'") in every claim get matched to the sentence that
                *introduces* that subject rather than the sentence that
                actually contains the cited fact. Set to ``False`` for
                plain, unweighted Jaccard similarity.
        """
        self.weighted = weighted

    def align(
        self,
        claims: list[Claim],
        text: str,
        citation_to_doc: dict[int, str],
    ) -> list[AlignedClaim]:
        """
        Align *claims* to citations via sentence-level Jaccard similarity.

        Args:
            claims:          Claims (typically with ``span=(-1,-1)``).
            text:            Full original answer text.
            citation_to_doc: Mapping from citation number to document ID.

        Returns:
            One :class:`~acclaim.data_models.AlignedClaim` per claim.
        """
        # Build sentence index with spans
        sentences: list[dict] = [
            {"text": sent.text, "span": (sent.start, sent.end)}
            for sent in split_sentences_with_spans(text)
            if sent.text
        ]

        claim_token_sets = [_tokenize(claim.text) for claim in claims]
        weights = _token_weights(claim_token_sets) if self.weighted else None

        aligned: list[AlignedClaim] = []
        for claim, claim_tokens in zip(claims, claim_token_sets):
            # For sentence claims erroneously passed here, fall back to span-based
            if claim.span != (-1, -1):
                doc_ids = extract_doc_ids_for_span(claim.span, text, citation_to_doc)
                aligned.append(AlignedClaim(claim=claim, citation_doc_ids=doc_ids))
                continue

            best_score = 0.0
            best_span: tuple[int, int] | None = None

            for sent in sentences:
                sent_tokens = _tokenize(sent["text"])
                if weights is not None:
                    score = _weighted_claim_overlap(claim_tokens, sent_tokens, weights)
                else:
                    score = _jaccard(claim_tokens, sent_tokens)
                if score > best_score:
                    best_score = score
                    best_span = sent["span"]

            if best_span and best_score > 0.0:
                doc_ids = extract_doc_ids_for_span(best_span, text, citation_to_doc)
                claim = Claim(text=claim.text, span=best_span)
            else:
                doc_ids = []

            aligned.append(AlignedClaim(claim=claim, citation_doc_ids=doc_ids))

        return aligned
