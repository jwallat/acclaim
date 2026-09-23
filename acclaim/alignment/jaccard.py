"""
Jaccard-similarity citation aligner.

For atomic claims that have ``span=(-1, -1)``, alignment is done by finding
the sentence in the original text most similar to the claim (using Jaccard
token-set similarity) and inheriting that sentence's ``[N]`` citations.
"""

from __future__ import annotations

from ._scoring import jaccard as _jaccard
from ._scoring import token_weights as _token_weights
from ._scoring import tokenize as _tokenize
from ._scoring import weighted_claim_overlap as _weighted_claim_overlap
from .base import CitationAligner
from ..citations.parser import extract_doc_ids_for_span
from ..data_models import AlignedClaim, Claim
from ..text_utils import split_sentences_with_spans


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
