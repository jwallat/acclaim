"""
Unit tests for all three metrics.
"""

import pytest

from acclaim.data_models import Claim, ClaimResult, SupportLabel, SupportResult


def _make_cr(
    text: str,
    label: SupportLabel,
    doc_ids: list[str],
    confidence: float = 0.9,
) -> ClaimResult:
    return ClaimResult(
        claim=Claim(text=text, span=(-1, -1)),
        citation_doc_ids=doc_ids,
        support=SupportResult(label=label, confidence=confidence, reason="test"),
    )


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def all_supported():
    return [
        _make_cr("Claim A", SupportLabel.SUPPORTED, ["doc-1"]),
        _make_cr("Claim B", SupportLabel.SUPPORTED, ["doc-2"]),
    ]


@pytest.fixture
def mixed():
    return [
        _make_cr("Claim A", SupportLabel.SUPPORTED, ["doc-1"]),
        _make_cr("Claim B", SupportLabel.REFUTED, ["doc-2"]),
        _make_cr("Claim C", SupportLabel.UNCLEAR, []),  # uncited
    ]


@pytest.fixture
def no_citations():
    return [
        _make_cr("Claim A", SupportLabel.UNCLEAR, []),
        _make_cr("Claim B", SupportLabel.UNCLEAR, []),
    ]


# ── CitationCorrectness ───────────────────────────────────────────────────────


class TestCitationCorrectness:
    from acclaim.metrics.correctness import CitationCorrectness as _cls

    def test_all_supported(self, all_supported):
        from acclaim.metrics.correctness import CitationCorrectness

        assert CitationCorrectness().compute(all_supported) == 1.0

    def test_half_supported(self, mixed):
        from acclaim.metrics.correctness import CitationCorrectness

        # 1 supported out of 2 cited (uncited claim C is excluded from denominator)
        score = CitationCorrectness().compute(mixed)
        assert score == pytest.approx(0.5)

    def test_no_citations_returns_zero(self, no_citations):
        from acclaim.metrics.correctness import CitationCorrectness

        assert CitationCorrectness().compute(no_citations) == 0.0

    def test_empty_list(self):
        from acclaim.metrics.correctness import CitationCorrectness

        assert CitationCorrectness().compute([]) == 0.0


# ── CitationCoverage ──────────────────────────────────────────────────────────


class TestCitationCoverage:
    def test_all_cited(self, all_supported):
        from acclaim.metrics.coverage import CitationCoverage

        assert CitationCoverage().compute(all_supported) == 1.0

    def test_partial_coverage(self, mixed):
        from acclaim.metrics.coverage import CitationCoverage

        # 2 cited out of 3
        assert CitationCoverage().compute(mixed) == pytest.approx(2 / 3)

    def test_zero_coverage(self, no_citations):
        from acclaim.metrics.coverage import CitationCoverage

        assert CitationCoverage().compute(no_citations) == 0.0

    def test_empty_list(self):
        from acclaim.metrics.coverage import CitationCoverage

        assert CitationCoverage().compute([]) == 0.0


# ── HallucinationRate ─────────────────────────────────────────────────────────


class TestHallucinationRate:
    def test_no_hallucinations(self, all_supported):
        from acclaim.metrics.hallucination import HallucinationRate

        assert (
            HallucinationRate().compute(all_supported, valid_doc_ids=["doc-1", "doc-2"])
            == 0.0
        )

    def test_all_hallucinated(self, all_supported):
        from acclaim.metrics.hallucination import HallucinationRate

        # valid_doc_ids doesn't include doc-1 or doc-2
        assert (
            HallucinationRate().compute(all_supported, valid_doc_ids=["doc-99"]) == 1.0
        )

    def test_partial_hallucination(self, all_supported):
        from acclaim.metrics.hallucination import HallucinationRate

        # doc-1 valid, doc-2 hallucinated → 1/2
        score = HallucinationRate().compute(all_supported, valid_doc_ids=["doc-1"])
        assert score == pytest.approx(0.5)

    def test_no_valid_doc_ids_returns_zero(self, all_supported):
        from acclaim.metrics.hallucination import HallucinationRate

        # None means we can't check — return safe default 0.0
        assert HallucinationRate().compute(all_supported, valid_doc_ids=None) == 0.0

    def test_no_citations_returns_zero(self, no_citations):
        from acclaim.metrics.hallucination import HallucinationRate

        assert HallucinationRate().compute(no_citations, valid_doc_ids=["doc-1"]) == 0.0


# ── CitationLengths ──────────────────────────────────────────────────────────


class TestCitationLengths:
    def test_no_citations_returns_zero(self, no_citations):
        from acclaim.metrics.citation_lengths import CitationLengths

        assert CitationLengths().compute(no_citations) == 0.0

    def test_empty_list(self):
        from acclaim.metrics.citation_lengths import CitationLengths

        assert CitationLengths().compute([]) == 0.0

    def test_single_claim(self):
        from acclaim.metrics.citation_lengths import CitationLengths

        claims = [_make_cr("The quick brown fox", SupportLabel.SUPPORTED, ["doc-1"])]
        score = CitationLengths().compute(claims)
        # "The quick brown fox" should be 4 tokens in GPT-2
        assert score == pytest.approx(4.0)

    def test_multiple_unique_claims(self):
        from acclaim.metrics.citation_lengths import CitationLengths

        # "Hello world" = 2 tokens, "The quick brown fox" = 4 tokens → avg = 3
        claims = [
            _make_cr("Hello world", SupportLabel.SUPPORTED, ["doc-1"]),
            _make_cr("The quick brown fox", SupportLabel.SUPPORTED, ["doc-2"]),
        ]
        score = CitationLengths().compute(claims)
        assert score == pytest.approx(3.0)

    def test_duplicate_claim_text_counted_once(self):
        from acclaim.metrics.citation_lengths import CitationLengths

        # Same claim text twice should only count once
        claims = [
            _make_cr("Hello world", SupportLabel.SUPPORTED, ["doc-1"]),
            _make_cr("Hello world", SupportLabel.SUPPORTED, ["doc-2"]),
        ]
        score = CitationLengths().compute(claims)
        # "Hello world" = 2 tokens, counted once → avg = 2
        assert score == pytest.approx(2.0)

    def test_mixed_cited_and_uncited(self, mixed):
        from acclaim.metrics.citation_lengths import CitationLengths

        # mixed has:
        # - "Claim A" with ["doc-1"] (cited)
        # - "Claim B" with ["doc-2"] (cited)
        # - "Claim C" with [] (uncited, should be excluded)
        # Only A and B should be counted
        score = CitationLengths().compute(mixed)
        # "Claim A" = 2 tokens, "Claim B" = 2 tokens → avg = 2
        assert score == pytest.approx(2.0)

    def test_valid_doc_ids_ignored(self):
        from acclaim.metrics.citation_lengths import CitationLengths

        claims = [_make_cr("Hello world", SupportLabel.SUPPORTED, ["doc-1"])]
        # valid_doc_ids should not affect citation_lengths computation
        score = CitationLengths().compute(claims, valid_doc_ids=["doc-1", "doc-2"])
        assert score == pytest.approx(2.0)


# ── TokenOverlap ──────────────────────────────────────────────────────────────


class TestTokenOverlap:
    def test_no_documents_returns_zero(self):
        from acclaim.metrics.token_overlap import TokenOverlap

        claims = [_make_cr("Hello world", SupportLabel.SUPPORTED, ["doc-1"])]
        # Without document objects, should return 0.0
        assert TokenOverlap().compute(claims, documents=None) == 0.0

    def test_no_citations_returns_zero(self, no_citations):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        documents = [Document(doc_id="doc-1", text="Some text here.")]
        assert TokenOverlap().compute(no_citations, documents=documents) == 0.0

    def test_empty_list(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        documents = [Document(doc_id="doc-1", text="Some text here.")]
        assert TokenOverlap().compute([], documents=documents) == 0.0

    def test_exact_match_claim_to_sentence(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim exactly matches a sentence in the document
        claims = [_make_cr("Hello world", SupportLabel.SUPPORTED, ["doc-1"])]
        documents = [Document(doc_id="doc-1", text="Hello world. Other text here.")]

        score = TokenOverlap().compute(claims, documents=documents)
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim: "hello world" (2 tokens)
        # Sentence: "hello beautiful world" (3 tokens)
        # Jaccard: 2 / 3 ≈ 0.667
        claims = [_make_cr("hello world", SupportLabel.SUPPORTED, ["doc-1"])]
        documents = [
            Document(doc_id="doc-1", text="hello beautiful world. Other text.")
        ]

        score = TokenOverlap().compute(claims, documents=documents)
        assert score == pytest.approx(2.0 / 3.0)

    def test_no_overlap(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim and document have no tokens in common
        claims = [_make_cr("apple orange", SupportLabel.SUPPORTED, ["doc-1"])]
        documents = [Document(doc_id="doc-1", text="banana grape. Other text.")]

        score = TokenOverlap().compute(claims, documents=documents)
        assert score == pytest.approx(0.0)

    def test_multiple_documents_averages_max(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim: "hello world"
        # doc-1 sentence "hello world" → Jaccard = 1.0
        # doc-2 sentence "banana grape" → Jaccard = 0.0
        # Average of max: (1.0 + 0.0) / 2 = 0.5
        claims = [_make_cr("hello world", SupportLabel.SUPPORTED, ["doc-1", "doc-2"])]
        documents = [
            Document(doc_id="doc-1", text="hello world. Other text."),
            Document(doc_id="doc-2", text="banana grape. Other text."),
        ]

        score = TokenOverlap().compute(claims, documents=documents)
        assert score == pytest.approx(0.5)

    def test_multiple_claims_averages(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim 1: "hello world" matches "hello world" exactly → 1.0
        # Claim 2: "apple orange" has no match → 0.0
        # Average: (1.0 + 0.0) / 2 = 0.5
        claims = [
            _make_cr("hello world", SupportLabel.SUPPORTED, ["doc-1"]),
            _make_cr("apple orange", SupportLabel.SUPPORTED, ["doc-1"]),
        ]
        documents = [
            Document(
                doc_id="doc-1",
                text="hello world. banana grape. Other text.",
            )
        ]

        score = TokenOverlap().compute(claims, documents=documents)
        assert score == pytest.approx(0.5)

    def test_case_insensitive(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim and document with different cases should match
        claims = [_make_cr("HELLO WORLD", SupportLabel.SUPPORTED, ["doc-1"])]
        documents = [Document(doc_id="doc-1", text="hello world. Other text.")]

        score = TokenOverlap().compute(claims, documents=documents)
        assert score == pytest.approx(1.0)

    def test_missing_document_id(self):
        from acclaim.metrics.token_overlap import TokenOverlap
        from acclaim.data_models import Document

        # Claim cites doc-2 which doesn't exist
        claims = [_make_cr("hello world", SupportLabel.SUPPORTED, ["doc-2"])]
        documents = [Document(doc_id="doc-1", text="hello world. Other text.")]

        score = TokenOverlap().compute(claims, documents=documents)
        # doc-2 not found, so similarity for that doc is 0.0
        assert score == pytest.approx(0.0)


# ── CitationNumber ───────────────────────────────────────────────────────────


class TestCitationNumber:
    def test_counts_inline_markers_in_answer(self):
        from acclaim.metrics.citation_number import CitationNumber

        metric = CitationNumber()
        assert metric.compute([], answer_text="A [1]. B [1][2].") == pytest.approx(3.0)

    def test_no_markers_returns_zero(self):
        from acclaim.metrics.citation_number import CitationNumber

        metric = CitationNumber()
        assert metric.compute([], answer_text="No citations here.") == pytest.approx(0.0)

    def test_fallback_uses_aligned_claim_citation_counts(self):
        from acclaim.metrics.citation_number import CitationNumber

        claims = [
            _make_cr("Claim A", SupportLabel.SUPPORTED, ["doc-1"]),
            _make_cr("Claim B", SupportLabel.SUPPORTED, ["doc-1", "doc-2"]),
            _make_cr("Claim C", SupportLabel.UNCLEAR, []),
        ]
        assert CitationNumber().compute(claims) == pytest.approx(3.0)


# ── SupportedClaimRate ────────────────────────────────────────────────────────


class TestSupportedClaimRate:
    def test_all_supported(self, all_supported):
        from acclaim.metrics.supported_claim_rate import SupportedClaimRate

        assert SupportedClaimRate().compute(all_supported) == pytest.approx(1.0)

    def test_empty_list(self):
        from acclaim.metrics.supported_claim_rate import SupportedClaimRate

        assert SupportedClaimRate().compute([]) == 0.0

    def test_mixed_includes_uncited_in_denominator(self, mixed):
        from acclaim.metrics.supported_claim_rate import SupportedClaimRate

        # mixed: Claim A SUPPORTED, Claim B REFUTED, Claim C UNCLEAR (uncited)
        # 1 supported out of 3 total (uncited claim C counts in denominator)
        assert SupportedClaimRate().compute(mixed) == pytest.approx(1 / 3)

    def test_differs_from_correctness_when_uncited_claims_present(self, mixed):
        from acclaim.metrics.supported_claim_rate import SupportedClaimRate
        from acclaim.metrics.correctness import CitationCorrectness

        # correctness: 1/2 (only cited claims), supported_claim_rate: 1/3 (all claims)
        assert CitationCorrectness().compute(mixed) == pytest.approx(0.5)
        assert SupportedClaimRate().compute(mixed) == pytest.approx(1 / 3)

    def test_none_supported(self):
        from acclaim.metrics.supported_claim_rate import SupportedClaimRate

        claims = [
            _make_cr("Claim A", SupportLabel.REFUTED, ["doc-1"]),
            _make_cr("Claim B", SupportLabel.UNCLEAR, []),
        ]
        assert SupportedClaimRate().compute(claims) == pytest.approx(0.0)

    def test_name(self):
        from acclaim.metrics.supported_claim_rate import SupportedClaimRate

        assert SupportedClaimRate().name == "supported_claim_rate"

    def test_registered_in_metric_registry(self):
        from acclaim.metrics.registry import METRIC_REGISTRY

        assert "supported_claim_rate" in METRIC_REGISTRY
