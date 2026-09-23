"""
Shared token-overlap scoring helpers for citation aligners.

Extracted from :mod:`acclaim.alignment.jaccard` so both
:class:`~acclaim.alignment.jaccard.JaccardCitationAligner` and
:class:`~acclaim.alignment.llm.LLMCitationAligner` use identical tie-break
math instead of duplicating it.
"""

from __future__ import annotations

import re
from collections import Counter


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_weights(claim_token_sets: list[set[str]]) -> dict[str, float]:
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


def weighted_claim_overlap(
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
