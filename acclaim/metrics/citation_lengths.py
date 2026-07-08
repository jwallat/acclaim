"""Citation lengths metric: average tokens per claim with citations."""

from __future__ import annotations

from .base import Metric
from ..data_models import ClaimResult


class CitationLengths(Metric):
    """
    Average number of tokens per claim with citations.

    Uses GPT-2 tokenizer to count tokens in claim text.
    Claims are deduplicated by text before computing the average.
    Only claims with at least one citation (non-empty ``citation_doc_ids``) are included.

    Formula: ``sum(tokens in unique cited claims) / count(unique cited claims)``

    Returns 0.0 if no claims have citations.
    """

    _tokenizer = None

    def _get_tokenizer(self):
        """Lazy-load tokenizer to avoid import overhead if metric not requested."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._tokenizer

    @property
    def name(self) -> str:
        return "citation_lengths"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents=None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        # Filter to claims with citations
        cited = [cr for cr in claim_results if cr.citation_doc_ids]
        if not cited:
            return 0.0

        # Deduplicate by claim text (avoid counting same claim twice)
        unique_texts = set()
        for cr in cited:
            unique_texts.add(cr.claim.text)

        # Count tokens for each unique claim text
        tokenizer = self._get_tokenizer()
        token_counts = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in unique_texts
        ]

        if not token_counts:
            return 0.0

        return sum(token_counts) / len(token_counts)
