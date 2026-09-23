"""
Domain dataclasses for acclaim.

All objects are immutable (frozen=True). The SupportLabel enum enforces
valid label values so metric comparisons are type-safe.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/Enums/tuples/dict-keys into JSON-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


class SupportLabel(str, Enum):
    """Judgement labels produced by an EvidenceJudge."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNCLEAR = "UNCLEAR"


class RelevanceLabel(str, Enum):
    """Relevance labels produced by a RelevanceJudge."""

    CORE = "CORE"
    COMPLEMENTARY = "COMPLEMENTARY"
    IRRELEVANT = "IRRELEVANT"


@dataclass(frozen=True)
class RelevanceResult:
    """Relevance judgement for a single claim relative to the question."""

    label: RelevanceLabel
    confidence: float
    reason: str


@dataclass(frozen=True)
class Document:
    """A source document that claims may be verified against."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Answer:
    """An LLM-generated answer containing inline citation markers like [1], [2]."""

    text: str


@dataclass(frozen=True)
class Claim:
    """
    An atomic or sentence-level claim extracted from an answer.

    Attributes:
        text:  The claim text (citation markers stripped).
        span:  Character span (start, end) in the original answer text.
               Set to (-1, -1) for atomic claims where no reliable span exists.
    """

    text: str
    span: tuple[int, int]


@dataclass(frozen=True)
class FilterDecision:
    """
    A claim filter's verdict on a single claim.

    Attributes:
        claim:       The claim the decision is about (never rewritten).
        keep:        Whether the claim passes the filter.
        filter_name: Name of the filter that decided (e.g. ``"check_worthiness"``).
        category:    Optional filter-specific label (e.g. ``"COMMON_KNOWLEDGE"``).
        reason:      Optional free-text justification.
    """

    claim: Claim
    keep: bool
    filter_name: str
    category: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class AlignedClaim:
    """A claim paired with the document IDs it cites."""

    claim: Claim
    citation_doc_ids: list[str]


@dataclass(frozen=True)
class SupportResult:
    """Judgement produced by an EvidenceJudge for a single (claim, docs) pair."""

    label: SupportLabel
    confidence: float
    reason: str


@dataclass(frozen=True)
class ClaimResult:
    """Final per-claim evaluation result combining alignment and judgement."""

    claim: Claim
    citation_doc_ids: list[str]
    support: SupportResult


@dataclass(frozen=True)
class EvaluationMetadata:
    """Run metadata: package version and the config used to produce the result."""

    version: str
    config: dict[str, Any]


@dataclass
class EvaluationResult:
    """
    Top-level output of evaluate().

    Attributes:
        claims:    Per-claim evaluation results (only claims that passed all
                   claim filters).
        metrics:   Aggregated metric scores keyed by metric name.
        steps:     Intermediate pipeline outputs for debugging/inspection.
                   Keys: "claims", "filtered_claims", "filter_decisions",
                   "aligned_claims", "claim_results", "citation_to_doc".
        metadata:  Package version and config used for this run. ``None`` when
                   this result is an item inside a ``BatchEvaluationResult``,
                   where the same metadata is instead attached once at the
                   batch level.
        question:  The original question the answer was responding to, if any.
        answer:    The raw answer text that was evaluated.
        documents: The source documents the answer was evaluated against.
        filter_decisions: One :class:`FilterDecision` per claim seen by each
                   configured claim filter (kept and dropped), in pipeline
                   order. Empty when no claim filters are configured.
    """

    claims: list[ClaimResult]
    metrics: dict[str, float]
    steps: dict[str, Any] = field(default_factory=dict)
    metadata: EvaluationMetadata | None = None
    question: str | None = None
    answer: str | None = None
    documents: list[Document] = field(default_factory=list)
    filter_decisions: list[FilterDecision] = field(default_factory=list)

    def pretty_print(self) -> None:
        """Print a human-readable summary of the evaluation result."""
        print("=== Evaluation Result ===")
        print(f"Total claims: {len(self.claims)}")
        print()
        for i, cr in enumerate(self.claims, 1):
            cited = ", ".join(cr.citation_doc_ids) if cr.citation_doc_ids else "(none)"
            print(f"  Claim {i}: {cr.claim.text!r}")
            print(f"    Citations: {cited}")
            print(
                f"    Verdict:   {cr.support.label.value}  (confidence={cr.support.confidence:.2f})"
            )
            print(f"    Reason:    {cr.support.reason}")
        dropped = [d for d in self.filter_decisions if not d.keep]
        if dropped:
            print()
            print(f"Filtered out: {len(dropped)}")
            for d in dropped:
                label = f"{d.filter_name}/{d.category}" if d.category else d.filter_name
                print(f"  {d.claim.text!r}")
                print(f"    Filter:    {label}")
                if d.reason:
                    print(f"    Reason:    {d.reason}")
        print()
        print("Metrics:")
        for name, value in self.metrics.items():
            print(f"  {name}: {value:.4f}")

    def to_dict(self, *, include_steps: bool = False) -> dict[str, Any]:
        """
        Convert to a JSON-safe dict.

        ``steps`` is excluded by default since ``steps["claim_results"]`` is
        the same data as the top-level ``claims`` field — pass
        ``include_steps=True`` for a full debug dump.
        """
        d: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "documents": _to_jsonable(self.documents),
            "claims": _to_jsonable(self.claims),
            "filter_decisions": _to_jsonable(self.filter_decisions),
            "metrics": self.metrics,
            "metadata": _to_jsonable(self.metadata) if self.metadata else None,
        }
        if include_steps:
            d["steps"] = _to_jsonable(self.steps)
        return d

    def to_jsonl(self, path: str | Path, *, include_steps: bool = False) -> None:
        """Write this result as a single-line JSONL file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(self.to_dict(include_steps=include_steps), ensure_ascii=False)
                + "\n"
            )


@dataclass(frozen=True)
class BatchExample:
    """Single dataset example for batch evaluation."""

    answer: str
    documents: list[Document]
    citation_to_doc: dict[int, str] | None = None
    question: str | None = None


@dataclass
class BatchEvaluationResult:
    """
    Batch evaluation outputs with per-item and aggregate metrics.

    ``filter_summary`` aggregates the items' claim-filter decisions (claim
    counts, drop rate, per-category kept/dropped counts). It is ``None``
    when no claim filters ran.
    """

    items: list[EvaluationResult]
    aggregate_metrics: dict[str, float]
    metadata: EvaluationMetadata | None = None
    filter_summary: dict[str, Any] | None = None

    def to_jsonl(
        self,
        path: str | Path,
        *,
        include_steps: bool = False,
        include_aggregate_line: bool = True,
    ) -> None:
        """
        Write one JSON line per item, followed by a trailing aggregate-metrics
        line (``record_type="aggregate"``) unless ``include_aggregate_line=False``.

        Each item line carries the batch-level ``metadata`` (version + config),
        since per-item ``EvaluationResult.metadata`` is ``None`` inside a batch.
        """
        metadata_dict = _to_jsonable(self.metadata) if self.metadata else None
        with open(path, "w", encoding="utf-8") as fh:
            for i, item in enumerate(self.items):
                line = {
                    "record_type": "item",
                    "index": i,
                    **item.to_dict(include_steps=include_steps),
                    "metadata": metadata_dict,
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            if include_aggregate_line:
                agg_line = {
                    "record_type": "aggregate",
                    "aggregate_metrics": self.aggregate_metrics,
                    "metadata": metadata_dict,
                }
                if self.filter_summary is not None:
                    agg_line["filter_summary"] = self.filter_summary
                fh.write(json.dumps(agg_line, ensure_ascii=False) + "\n")
