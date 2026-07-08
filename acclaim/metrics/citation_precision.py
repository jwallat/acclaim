"""
Citation precision metric — LongCite style.

For each individual citation in the response, an LLM judge decides whether
the cited snippet is *relevant* (1) or *unrelevant* (0) to the statement it
supports.  A snippet is relevant when it entails at least some key points of
the statement (i.e. partial support is sufficient).  The metric is the
mean per-citation score averaged over all citations in the response.

Reference:
    Zhang et al., "LongCite: Enabling LLMs to Generate Fine-grained
    Citations in Long-context QA", 2024.
"""

from __future__ import annotations

import logging

from ..concurrency import parallel_map
from ..data_models import ClaimResult, Document
from ..judges.citation_precision import PrecisionJudge
from .base import Metric

logger = logging.getLogger(__name__)


class CitationPrecisionMetric(Metric):
    """
    LongCite citation precision averaged over all citations.

    Each citation is scored 0 (unrelevant) or 1 (relevant) by a
    :class:`~acclaim.judges.citation_precision.PrecisionJudge`; the
    metric is the mean of those per-citation scores.

    The unit of scoring is the **citation** (one judge call per
    (claim, document) pair), not the claim — a claim with three citations
    produces three scores.

    Args:
        judge: The :class:`PrecisionJudge` instance used to score each
               (statement, snippet) pair.  Typically a
               :class:`~acclaim.judges.citation_precision.LiteLLMPrecisionJudge`.
    """

    def __init__(self, judge: PrecisionJudge, max_workers: int = 1) -> None:
        self._judge = judge
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        """Metric registry key."""
        return "citation_precision"

    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float:
        """
        Compute citation precision over all citations.

        Args:
            claim_results: Per-claim evaluation results from the pipeline.
            valid_doc_ids: Unused; present for interface compatibility.
            documents:     Source documents.  Required to resolve cited
                           document text for the LLM judge.  When
                           ``None`` or a cited doc ID is missing, that
                           citation is scored 0.0.

        Returns:
            Mean citation precision in ``[0.0, 1.0]``.  Returns ``0.0``
            when there are no citations in any claim.
        """
        if not claim_results:
            return 0.0

        doc_by_id: dict[str, Document] = (
            {doc.doc_id: doc for doc in documents} if documents else {}
        )

        pairs: list[tuple[ClaimResult, str]] = [
            (cr, doc_id) for cr in claim_results for doc_id in cr.citation_doc_ids
        ]

        resolved: list[float | None] = [None] * len(pairs)
        to_judge: list[tuple[int, ClaimResult, str]] = []
        for idx, (cr, doc_id) in enumerate(pairs):
            if doc_id not in doc_by_id:
                logger.warning(
                    "CitationPrecisionMetric: document %r not found for claim %r. "
                    "Scoring 0.0. Pass `documents` to evaluate().",
                    doc_id,
                    cr.claim.text[:60],
                )
                resolved[idx] = 0.0
            else:
                to_judge.append((idx, cr, doc_id))

        def _judge_pair(item: tuple[int, ClaimResult, str]) -> float:
            _, cr, doc_id = item
            return self._judge.judge(cr.claim, doc_by_id[doc_id]).score

        judged_scores = parallel_map(_judge_pair, to_judge, self._max_workers)
        for (idx, _, _), score in zip(to_judge, judged_scores):
            resolved[idx] = score

        # No citations at all → precision is undefined; return 0.0
        return sum(resolved) / len(resolved) if resolved else 0.0
