"""
Citation recall metric — LongCite style.

For each claim:

* **Cited** (``citation_doc_ids`` is non-empty): an LLM judge scores
  whether the concatenated cited snippets *fully support* (1.0),
  *partially support* (0.5), or *do not support* (0.0) the claim.
* **Uncited** (``citation_doc_ids`` is empty): the claim receives a
  score of **0.0**.  (Atomic claim extractors mark every factual claim
  that lacks a citation as missing, so the 0 is always correct there.
  Sentence-level extractors may mix functional sentences in, which
  slightly underestimates recall — use atomic extraction for the most
  faithful LongCite score.)

The final score is the mean over all claims.

Reference:
    Zhang et al., "LongCite: Enabling LLMs to Generate Fine-grained
    Citations in Long-context QA", 2024.
"""

from __future__ import annotations

import logging

from ..concurrency import parallel_map
from ..data_models import Claim, ClaimResult, Document
from ..judges.citation_recall import RecallJudge
from .base import Metric

logger = logging.getLogger(__name__)


class CitationRecallMetric(Metric):
    """
    LongCite citation recall averaged over all claims.

    Each cited claim is scored 0 / 0.5 / 1 by a :class:`RecallJudge`;
    uncited claims receive 0.  The metric is the mean of those scores.

    Args:
        judge: The :class:`RecallJudge` instance to use for scoring.
               Typically a :class:`~acclaim.judges.citation_recall.LiteLLMRecallJudge`.
    """

    def __init__(self, judge: RecallJudge, max_workers: int = 1) -> None:
        self._judge = judge
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        """Metric registry key."""
        return "citation_recall"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        """
        Compute citation recall over all claims.

        Args:
            claim_results: Per-claim evaluation results from the pipeline.
            valid_doc_ids: Unused; present for interface compatibility.
            documents:     Source documents.  Required to resolve cited
                           document text for the LLM judge.  When
                           ``None`` or a cited doc ID is missing from
                           this list, that claim is scored 0.0.

        Returns:
            Mean citation recall in ``[0.0, 1.0]``.
        """
        if not claim_results:
            return 0.0

        doc_by_id: dict[str, Document] = (
            {doc.doc_id: doc for doc in documents} if documents else {}
        )

        resolved: list[float | None] = [None] * len(claim_results)
        to_judge: list[tuple[int, Claim, list[Document]]] = []
        for idx, cr in enumerate(claim_results):
            if not cr.citation_doc_ids:
                # Uncited claim → score 0
                resolved[idx] = 0.0
                continue

            cited_docs = [
                doc_by_id[did] for did in cr.citation_doc_ids if did in doc_by_id
            ]

            if not cited_docs:
                # Cited IDs are present but none could be resolved → 0
                logger.warning(
                    "CitationRecallMetric: no documents resolved for claim %r "
                    "(cited ids: %s). Scoring 0.0. Pass `documents` to evaluate().",
                    cr.claim.text[:60],
                    cr.citation_doc_ids,
                )
                resolved[idx] = 0.0
                continue

            to_judge.append((idx, cr.claim, cited_docs))

        def _recall_one(item: tuple[int, Claim, list[Document]]) -> float:
            _, claim, docs = item
            return self._judge.recall(claim, docs).score

        judged_scores = parallel_map(_recall_one, to_judge, self._max_workers)
        for (idx, _, _), score in zip(to_judge, judged_scores):
            resolved[idx] = score

        return sum(resolved) / len(resolved)
