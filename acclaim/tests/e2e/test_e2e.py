"""
End-to-end tests for acclaim.

These tests run the full pipeline: claim extraction → alignment → judgement
→ metrics.  All LLM calls go to a local vLLM server (started via
``start_vllm.sh``).

Mark: @pytest.mark.e2e  — requires a running vLLM container.
      @pytest.mark.vllm  — skipped automatically if VLLM_API_BASE is unset.

Run with::

    # start the server first
    bash start_vllm.sh

    # then run only the e2e tests, showing intermediate step output
    pytest acclaim/tests/e2e/ -m vllm -s -v

Each test prints the contents of EvaluationResult.steps so you can
inspect every pipeline stage visually.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from acclaim import evaluate
from acclaim.config.load import EvalConfig, JudgeConfig
from acclaim.data_models import Document, SupportLabel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vllm_config(claim_extractor: str = "sentence") -> EvalConfig:
    """Build config pointing at the local vLLM server."""
    api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "openai/meta-llama-3-8b-instruct")
    return EvalConfig(
        claim_extractor=claim_extractor,
        judge=JudgeConfig(
            model=model,
            api_base=api_base,
            api_key="dummy",
            max_retries=2,
        ),
    )


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _vllm_available() -> bool:
    """Quick check: is the vLLM server URL set or at the default location?"""
    import socket
    import urllib.parse

    url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# @pytest.mark.vllm  — used for -m vllm selection
# @_vllm_skip        — skips at runtime when the server is not reachable
_vllm_skip = pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM server not reachable. Start it with: bash new/start_vllm.sh",
)
vllm = lambda f: pytest.mark.vllm(_vllm_skip(f))  # noqa: E731


# ---------------------------------------------------------------------------
# E2E test 1: Sentence claims + LiteLLM judge (happy path)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@vllm
def test_e2e_sentence_claims_full_pipeline():  # type: ignore[misc]
    """
    Full pipeline with sentence-level claims and a local LLM judge.

    Input
    -----
    Answer with two sentences, each citing one document.

    Expected
    --------
    - 2 claims extracted
    - Each claim aligned to the correct doc
    - Metrics: coverage=1.0 (both cited), hallucination_rate=0.0
    - attr_precision depends on judge (should be 1.0 for clearly supported facts)
    """
    answer = textwrap.dedent(
        """\
        The Eiffel Tower is located in Paris, France [1].
        It was constructed between 1887 and 1889 [2].
    """
    ).strip()

    documents = [
        Document(doc_id="doc-1", text="The Eiffel Tower stands in Paris, France."),
        Document(doc_id="doc-2", text="The Eiffel Tower was built from 1887 to 1889."),
    ]

    config = _vllm_config("sentence")
    result = evaluate(answer, documents, config=config)

    # ── Print intermediate steps ─────────────────────────────────────────────
    _section("STEP 1: Extracted claims")
    for i, claim in enumerate(result.steps["claims"], 1):
        print(f"  [{i}] span={claim.span}  text={claim.text!r}")

    _section("STEP 2: Aligned claims")
    for i, ac in enumerate(result.steps["aligned_claims"], 1):
        print(f"  [{i}] doc_ids={ac.citation_doc_ids}  claim={ac.claim.text!r}")

    _section("STEP 3: Claim results (with verdicts)")
    for i, cr in enumerate(result.steps["claim_results"], 1):
        print(
            f"  [{i}] {cr.support.label.value}  conf={cr.support.confidence:.2f}"
            f"  docs={cr.citation_doc_ids}"
        )
        print(f"       reason: {cr.support.reason}")

    _section("STEP 4: Metrics")
    for name, score in result.metrics.items():
        print(f"  {name}: {score:.4f}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert len(result.claims) == 2, "Expected 2 sentence-level claims"

    aligned = result.steps["aligned_claims"]
    assert aligned[0].citation_doc_ids == ["doc-1"]
    assert aligned[1].citation_doc_ids == ["doc-2"]

    assert result.metrics["coverage"] == pytest.approx(1.0)
    assert result.metrics["hallucination_rate"] == pytest.approx(0.0)
    # Both facts are clearly supported — judge should agree
    assert result.metrics["attr_precision"] >= 0.5


# ---------------------------------------------------------------------------
# E2E test 2: Atomic claims + Jaccard aligner
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@vllm
def test_e2e_atomic_claims_jaccard_alignment():  # type: ignore[misc]
    """
    Pipeline with atomic claim extraction (LLM-based) and Jaccard alignment.

    Input
    -----
    A multi-fact sentence citing one document. The extractor should break it
    into ≥2 atomic claims; all should align back to the same document.

    Expected
    --------
    - ≥1 atomic claims extracted
    - All aligned to doc-1 via Jaccard similarity
    - hallucination_rate = 0.0
    """
    answer = (
        "The Eiffel Tower, completed in 1889, stands 330 metres tall and "
        "is located in Paris [1]."
    )

    documents = [
        Document(
            doc_id="doc-1",
            text=(
                "The Eiffel Tower was completed in 1889. "
                "It is 330 metres tall. "
                "It is situated in Paris, France."
            ),
        )
    ]

    config = _vllm_config("atomic")
    result = evaluate(answer, documents, config=config)

    # ── Print intermediate steps ─────────────────────────────────────────────
    _section("STEP 1: Atomic claims extracted by LLM")
    for i, claim in enumerate(result.steps["claims"], 1):
        print(f"  [{i}] {claim.text!r}")

    _section("STEP 2: Jaccard alignment")
    for i, ac in enumerate(result.steps["aligned_claims"], 1):
        print(f"  [{i}] doc_ids={ac.citation_doc_ids}  claim={ac.claim.text!r}")

    _section("STEP 3: Verdicts")
    for i, cr in enumerate(result.steps["claim_results"], 1):
        print(f"  [{i}] {cr.support.label.value}  reason: {cr.support.reason}")

    _section("STEP 4: Metrics")
    for name, score in result.metrics.items():
        print(f"  {name}: {score:.4f}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert len(result.claims) >= 1, "Atomic extractor should produce at least one claim"
    assert result.metrics["hallucination_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# E2E test 3: Answer with no citation markers (edge case)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@vllm
def test_e2e_no_citations_edge_case():  # type: ignore[misc]
    """
    Answer that contains no [N] citation markers at all.

    Expected
    --------
    - Claims are still extracted (one per sentence)
    - All citation_doc_ids are empty
    - coverage = 0.0
    - hallucination_rate = 0.0 (nothing to hallucinate)
    - judge receives empty doc list → returns UNCLEAR for all claims
    - attr_precision = 0.0 (no cited claims, denominator is 0)
    """
    answer = "The sky is blue. Water is wet."
    documents = [
        Document(
            doc_id="doc-1", text="The sky appears blue due to Rayleigh scattering."
        ),
    ]

    config = _vllm_config("sentence")
    result = evaluate(answer, documents, config=config)

    _section("STEP 1: Claims")
    for i, c in enumerate(result.steps["claims"], 1):
        print(f"  [{i}] {c.text!r}")

    _section("STEP 2: Alignment (expect empty doc_ids)")
    for i, ac in enumerate(result.steps["aligned_claims"], 1):
        print(f"  [{i}] doc_ids={ac.citation_doc_ids}")

    _section("STEP 3: Verdicts (expect UNCLEAR due to no docs)")
    for i, cr in enumerate(result.steps["claim_results"], 1):
        print(f"  [{i}] {cr.support.label.value}  reason: {cr.support.reason}")

    _section("STEP 4: Metrics")
    for name, score in result.metrics.items():
        print(f"  {name}: {score:.4f}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert len(result.claims) >= 1

    for cr in result.claims:
        assert (
            cr.citation_doc_ids == []
        ), "No citations in answer → all doc_ids should be empty"
        assert (
            cr.support.label == SupportLabel.UNCLEAR
        ), "Judge should return UNCLEAR when no docs provided"

    assert result.metrics["coverage"] == pytest.approx(0.0)
    assert result.metrics["hallucination_rate"] == pytest.approx(0.0)
    assert result.metrics["attr_precision"] == pytest.approx(0.0)
