"""Unit tests for EvaluationResult / BatchEvaluationResult JSONL serialization."""

from __future__ import annotations

import json

from acclaim.data_models import (
    AlignedClaim,
    BatchEvaluationResult,
    Claim,
    ClaimResult,
    Document,
    EvaluationMetadata,
    EvaluationResult,
    SupportLabel,
    SupportResult,
)


def _make_claim_result(text: str = "Paris is the capital of France.") -> ClaimResult:
    return ClaimResult(
        claim=Claim(text=text, span=(0, len(text))),
        citation_doc_ids=["doc-1"],
        support=SupportResult(label=SupportLabel.SUPPORTED, confidence=0.9, reason="matches"),
    )


def _make_result(with_metadata: bool = True) -> EvaluationResult:
    claim_result = _make_claim_result()
    metadata = (
        EvaluationMetadata(version="0.1.0", config={"claim_extractor": "sentence"})
        if with_metadata
        else None
    )
    return EvaluationResult(
        claims=[claim_result],
        metrics={"coverage": 1.0, "hallucination_rate": 0.0},
        steps={
            "claims": [claim_result.claim],
            "aligned_claims": [
                AlignedClaim(claim=claim_result.claim, citation_doc_ids=["doc-1"])
            ],
            "claim_results": [claim_result],
            "citation_to_doc": {1: "doc-1"},
        },
        metadata=metadata,
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        documents=[Document(doc_id="doc-1", text="Paris is the capital of France.")],
    )


def test_to_dict_excludes_steps_by_default() -> None:
    result = _make_result()
    d = result.to_dict()

    assert "steps" not in d
    assert d["metrics"] == {"coverage": 1.0, "hallucination_rate": 0.0}
    assert d["claims"][0]["support"]["label"] == "SUPPORTED"
    assert d["claims"][0]["claim"]["span"] == [0, len("Paris is the capital of France.")]
    assert d["metadata"] == {"version": "0.1.0", "config": {"claim_extractor": "sentence"}}
    assert d["question"] == "What is the capital of France?"
    assert d["answer"] == "Paris is the capital of France."
    assert d["documents"] == [{"doc_id": "doc-1", "text": "Paris is the capital of France."}]


def test_to_dict_handles_missing_question_answer_documents() -> None:
    result = EvaluationResult(claims=[_make_claim_result()], metrics={})
    d = result.to_dict()

    assert d["question"] is None
    assert d["answer"] is None
    assert d["documents"] == []


def test_to_dict_include_steps_stringifies_int_keys() -> None:
    result = _make_result()
    d = result.to_dict(include_steps=True)

    assert "steps" in d
    assert d["steps"]["citation_to_doc"] == {"1": "doc-1"}
    assert d["steps"]["claim_results"][0]["support"]["label"] == "SUPPORTED"


def test_to_dict_handles_none_metadata() -> None:
    result = _make_result(with_metadata=False)
    d = result.to_dict()

    assert d["metadata"] is None


def test_evaluation_result_to_jsonl_writes_single_line(tmp_path) -> None:
    result = _make_result()
    path = tmp_path / "out.jsonl"

    result.to_jsonl(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert set(obj.keys()) == {"question", "answer", "documents", "claims", "metrics", "metadata"}
    assert obj["metrics"]["coverage"] == 1.0
    assert obj["question"] == "What is the capital of France?"
    assert obj["documents"] == [{"doc_id": "doc-1", "text": "Paris is the capital of France."}]


def test_batch_to_jsonl_writes_items_plus_aggregate_line(tmp_path) -> None:
    items = [_make_result(with_metadata=False), _make_result(with_metadata=False)]
    batch = BatchEvaluationResult(
        items=items,
        aggregate_metrics={"coverage": 1.0},
        metadata=EvaluationMetadata(version="0.1.0", config={"claim_extractor": "sentence"}),
    )
    path = tmp_path / "batch.jsonl"

    batch.to_jsonl(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(items) + 1

    for i, line in enumerate(lines[:-1]):
        obj = json.loads(line)
        assert obj["record_type"] == "item"
        assert obj["index"] == i
        assert obj["metadata"] == {"version": "0.1.0", "config": {"claim_extractor": "sentence"}}

    agg = json.loads(lines[-1])
    assert agg["record_type"] == "aggregate"
    assert agg["aggregate_metrics"] == {"coverage": 1.0}
    assert agg["metadata"] == {"version": "0.1.0", "config": {"claim_extractor": "sentence"}}


def test_batch_to_jsonl_can_omit_aggregate_line(tmp_path) -> None:
    items = [_make_result(with_metadata=False)]
    batch = BatchEvaluationResult(items=items, aggregate_metrics={"coverage": 1.0})
    path = tmp_path / "batch.jsonl"

    batch.to_jsonl(path, include_aggregate_line=False)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(items)
    assert all(json.loads(line)["record_type"] == "item" for line in lines)
