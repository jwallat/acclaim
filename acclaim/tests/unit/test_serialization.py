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
    FilterDecision,
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
    assert set(obj.keys()) == {
        "question",
        "answer",
        "documents",
        "claims",
        "filter_decisions",
        "metrics",
        "metadata",
    }
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


def _with_filter_decisions(result: EvaluationResult) -> EvaluationResult:
    dropped = Claim(text="Great question!", span=(-1, -1))
    result.filter_decisions = [
        FilterDecision(
            claim=result.claims[0].claim,
            keep=True,
            filter_name="check_worthiness",
            category="QUANTITY",
            reason="specific fact",
        ),
        FilterDecision(
            claim=dropped,
            keep=False,
            filter_name="check_worthiness",
            category="NOT_A_CLAIM",
            reason="filler",
        ),
    ]
    result.steps["claims"] = [result.claims[0].claim, dropped]
    return result


def test_to_dict_includes_filter_decisions() -> None:
    d = _with_filter_decisions(_make_result()).to_dict()
    assert d["filter_decisions"][1] == {
        "claim": {"text": "Great question!", "span": [-1, -1]},
        "keep": False,
        "filter_name": "check_worthiness",
        "category": "NOT_A_CLAIM",
        "reason": "filler",
    }


def test_to_dict_filter_decisions_empty_without_filters() -> None:
    assert _make_result().to_dict()["filter_decisions"] == []


def test_pretty_print_lists_dropped_claims(capsys) -> None:
    _with_filter_decisions(_make_result()).pretty_print()
    out = capsys.readouterr().out
    assert "Filtered out: 1" in out
    assert "'Great question!'" in out
    assert "check_worthiness/NOT_A_CLAIM" in out


def test_pretty_print_omits_filter_section_without_drops(capsys) -> None:
    _make_result().pretty_print()
    assert "Filtered out" not in capsys.readouterr().out


def test_batch_to_jsonl_writes_filter_summary(tmp_path) -> None:
    batch = BatchEvaluationResult(
        items=[_make_result(with_metadata=False)],
        aggregate_metrics={"coverage": 1.0},
        filter_summary={"claims_dropped": 1},
    )
    path = tmp_path / "batch.jsonl"
    batch.to_jsonl(path)
    agg = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert agg["filter_summary"] == {"claims_dropped": 1}


def test_batch_to_jsonl_omits_filter_summary_when_none(tmp_path) -> None:
    batch = BatchEvaluationResult(
        items=[_make_result(with_metadata=False)], aggregate_metrics={"coverage": 1.0}
    )
    path = tmp_path / "batch.jsonl"
    batch.to_jsonl(path)
    agg = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert "filter_summary" not in agg
