"""Token overlap metric: Jaccard similarity between claims and cited document text."""

from __future__ import annotations

from .base import Metric
from ..data_models import ClaimResult, Document
from ..text_utils import split_sentences, tokenize_text, jaccard_similarity


class TokenOverlap(Metric):
    """
    Average token overlap (Jaccard similarity) between claims and cited documents.

    For each claim, finds the most similar sentence in each cited document using
    Jaccard similarity on whitespace-tokenized text (lowercased). Returns the
    average of max similarities across all cited documents, then averages across claims.

    When multiple documents are cited per claim, averages the max similarity found
    in each document.

    Optionally logs the most similar sentence for each claim when enabled in config.

    Formula: ``avg(avg(max_jaccard(claim, docs[*])) for claim in claims)``

    Returns 0.0 if no claims have citations or no documents are provided.
    """

    @property
    def name(self) -> str:
        return "token_overlap"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        if not documents:
            return 0.0

        # Create doc_id -> Document lookup
        doc_by_id = {doc.doc_id: doc for doc in documents}

        # Filter to claims with citations
        cited = [cr for cr in claim_results if cr.citation_doc_ids]
        if not cited:
            return 0.0

        similarities = []

        for claim_result in cited:
            claim_text = claim_result.claim.text
            claim_tokens = tokenize_text(claim_text)

            if not claim_tokens:
                # Empty claim text
                similarities.append(0.0)
                continue

            # For this claim, find max similarity in each cited document
            doc_max_sims = []

            for doc_id in claim_result.citation_doc_ids:
                if doc_id not in doc_by_id:
                    continue

                doc = doc_by_id[doc_id]
                sentences = split_sentences(doc.text)

                if not sentences:
                    doc_max_sims.append(0.0)
                    continue

                # Find most similar sentence in this document
                max_sim = 0.0
                for sentence in sentences:
                    sent_tokens = tokenize_text(sentence)
                    sim = jaccard_similarity(claim_tokens, sent_tokens)
                    max_sim = max(max_sim, sim)

                doc_max_sims.append(max_sim)

            # Average max similarity across all cited documents
            if doc_max_sims:
                claim_similarity = sum(doc_max_sims) / len(doc_max_sims)
            else:
                claim_similarity = 0.0

            similarities.append(claim_similarity)

        # Average across all claims
        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)
