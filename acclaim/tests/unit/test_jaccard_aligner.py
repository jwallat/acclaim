"""
Unit tests for the JaccardCitationAligner.
"""

import pytest

from acclaim.alignment.jaccard import JaccardCitationAligner
from acclaim.data_models import Claim


TEXT = "Paris is the capital of France [1]. The Eiffel Tower stands 330 m [2]."
CITATION_TO_DOC = {1: "doc-france", 2: "doc-eiffel"}


def test_atomic_claim_aligned_to_best_sentence():
    aligner = JaccardCitationAligner()
    # Claim paraphrases the first sentence — should get doc-france
    claims = [Claim(text="Paris is the capital city of France.", span=(-1, -1))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == ["doc-france"]


def test_atomic_claim_aligned_to_second_sentence():
    aligner = JaccardCitationAligner()
    claims = [Claim(text="The Eiffel Tower is 330 metres high.", span=(-1, -1))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == ["doc-eiffel"]


def test_no_match_returns_empty():
    aligner = JaccardCitationAligner()
    claims = [Claim(text="Blockchain technology.", span=(-1, -1))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []


def test_sentence_claim_with_span_uses_span_logic():
    """Claims with real spans bypass Jaccard and use direct span extraction."""
    aligner = JaccardCitationAligner()
    # span covers first sentence in TEXT (includes [1])
    claims = [Claim(text="Paris is the capital of France.", span=(0, 35))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == ["doc-france"]


def test_atomic_claim_span_set_to_best_sentence():
    """The aligned claim should expose the span of the best-matching sentence."""
    aligner = JaccardCitationAligner()
    claims = [Claim(text="The Eiffel Tower is 330 metres high.", span=(-1, -1))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    start, end = aligned[0].claim.span
    assert (start, end) != (-1, -1)
    assert TEXT[start:end].startswith("The Eiffel Tower")


def test_atomic_claim_span_unchanged_when_no_match():
    """If no sentence overlaps the claim tokens, span stays (-1, -1)."""
    aligner = JaccardCitationAligner()
    claims = [Claim(text="Blockchain technology.", span=(-1, -1))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].claim.span == (-1, -1)


def test_atomic_claim_aligned_with_comma_separated_markers():
    text = "Paris is the capital of France [1]. The Eiffel Tower stands 330 m [1, 2]."
    claims = [Claim(text="The Eiffel Tower is 330 metres high.", span=(-1, -1))]
    aligned = JaccardCitationAligner().align(claims, text, CITATION_TO_DOC)
    assert set(aligned[0].citation_doc_ids) == {"doc-france", "doc-eiffel"}


# ── Weighted vs. plain scoring (coreference-repetition regression) ──────────
#
# Real-world example (ASQA, llama-3.1-8b) surfaced during manual annotation of
# library_evals/citation_alignment/: atomic claims that restate the
# coreference-resolved subject ("the song 'I'm Coming Out'") in nearly every
# claim get mis-matched by plain Jaccard to the sentence that merely
# *introduces* that subject, instead of the later, longer, cited sentence
# that actually states the fact. weighted=True fixes most (not all — see the
# "wrote" claims below, which fail for an unrelated reason: "wrote" vs.
# "written" is a word-form mismatch that no token-weighting scheme resolves).

COREF_TEXT = (
    'The song "I\'m Coming Out" was recorded by American singer Diana Ross. '
    "It was written and produced by Chic members Bernard Edwards and Nile "
    "Rodgers, and released in August 22, 1980 as the second single from "
    "Ross' self-titled tenth album \"Diana\" (1980) [1]."
)
COREF_CITATION_TO_DOC = {1: "doc-song"}
COREF_CLAIMS = [
    Claim(text='The song "I\'m Coming Out" was recorded by Diana Ross.', span=(-1, -1)),
    Claim(text="Diana Ross is an American singer.", span=(-1, -1)),
    Claim(text="Bernard Edwards is a member of Chic.", span=(-1, -1)),
    Claim(text="Nile Rodgers is a member of Chic.", span=(-1, -1)),
    Claim(text="Bernard Edwards wrote the song 'I'm Coming Out'.", span=(-1, -1)),
    Claim(text="Nile Rodgers wrote the song 'I'm Coming Out'.", span=(-1, -1)),
    Claim(text="Bernard Edwards produced the song 'I'm Coming Out'.", span=(-1, -1)),
    Claim(text="Nile Rodgers produced the song 'I'm Coming Out'.", span=(-1, -1)),
    Claim(text="The song 'I'm Coming Out' was released on August 22, 1980.", span=(-1, -1)),
    Claim(text="The song 'I'm Coming Out' was the second single from the album 'Diana'.", span=(-1, -1)),
    Claim(text="The album 'Diana' is the tenth album by Diana Ross.", span=(-1, -1)),
]


def test_weighted_default_fixes_coreference_bias():
    """
    Claims about facts only stated in the second (cited) sentence should
    inherit its citation, not the first (uncited) sentence's, even though
    they all repeat the coreference-resolved subject.
    """
    aligner = JaccardCitationAligner()  # weighted=True by default
    aligned = aligner.align(COREF_CLAIMS, COREF_TEXT, COREF_CITATION_TO_DOC)

    # Claims that genuinely have no source in this text stay uncited.
    assert aligned[0].citation_doc_ids == []  # recorded by Diana Ross
    assert aligned[1].citation_doc_ids == []  # American singer

    # Claims whose fact lives in the second sentence get its citation.
    for i in (2, 3, 6, 7, 8, 9, 10):
        assert aligned[i].citation_doc_ids == ["doc-song"], aligned[i].claim.text

    # Known residual limitation: "wrote" vs. "written" is a word-form
    # mismatch no token-weighting scheme fixes, so these stay uncited.
    assert aligned[4].citation_doc_ids == []
    assert aligned[5].citation_doc_ids == []


def test_weighted_false_reproduces_plain_jaccard_bias():
    """weighted=False must keep today's plain-Jaccard behavior, bug and all."""
    aligner = JaccardCitationAligner(weighted=False)
    aligned = aligner.align(COREF_CLAIMS, COREF_TEXT, COREF_CITATION_TO_DOC)

    # Under plain Jaccard, these "produced"/"released" claims get
    # mis-matched to the first (uncited) sentence because they repeat the
    # song title — the exact bug weighted=True fixes.
    assert aligned[6].citation_doc_ids == []
    assert aligned[8].citation_doc_ids == []
