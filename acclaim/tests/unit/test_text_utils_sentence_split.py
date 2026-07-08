"""
Unit tests for sentence splitting in text_utils.py.

Covers easy cases (plain multi-sentence text) and the hard cases that a
naive punctuation regex (``r"(?<=[.!?])\\s+"``) gets wrong: abbreviations
like "U.S." or "No. 1", and citation markers placed either before or after
the sentence-final punctuation.
"""

from acclaim.text_utils import split_sentences, split_sentences_with_spans


# ---------------------------------------------------------------------------
# Easy cases
# ---------------------------------------------------------------------------


def test_empty_string():
    assert split_sentences_with_spans("") == []
    assert split_sentences("") == []


def test_single_sentence():
    text = "The sky is blue."
    result = split_sentences_with_spans(text)
    assert len(result) == 1
    assert result[0].text.strip() == text
    assert result[0].start == 0
    assert result[0].end == len(text)


def test_multiple_plain_sentences():
    text = "Paris is the capital of France. The Eiffel Tower was built in 1889."
    sentences = split_sentences(text)
    assert sentences == [
        "Paris is the capital of France.",
        "The Eiffel Tower was built in 1889.",
    ]


# ---------------------------------------------------------------------------
# Hard cases: abbreviations that a naive regex incorrectly splits on
# ---------------------------------------------------------------------------


def test_no_period_abbreviation_not_split():
    text = "It became Travis's third No. 1 single on the charts."
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


def test_us_abbreviation_not_split():
    text = 'It topped the U.S. "Billboard" Hot Country Singles charts in 1990.'
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


def test_middle_initial_abbreviation_not_split():
    text = (
        "In 1991, Schlitz also wrote a theme song for President George H. W. "
        "Bush's \"Points of Light\" program."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


# ---------------------------------------------------------------------------
# Citation markers before / after sentence-final punctuation
# ---------------------------------------------------------------------------


def test_citation_before_period_kept_in_same_sentence():
    text = (
        "It became Travis's third No. 1 single on the charts [1]. "
        "It won a Grammy [2]."
    )
    spans = split_sentences_with_spans(text)
    assert len(spans) == 2
    assert spans[0].text.strip() == "It became Travis's third No. 1 single on the charts [1]."
    assert spans[1].text.strip() == "It won a Grammy [2]."
    for sent in spans:
        assert text[sent.start:sent.end] == sent.text


def test_citation_after_period_in_next_sentence_span():
    text = "Paris is the capital of France. [1] The Eiffel Tower stands 330 m."
    spans = split_sentences_with_spans(text)
    assert len(spans) == 2
    assert spans[0].text.strip() == "Paris is the capital of France."
    assert spans[1].text.strip() == "[1] The Eiffel Tower stands 330 m."
    for sent in spans:
        assert text[sent.start:sent.end] == sent.text


# ---------------------------------------------------------------------------
# Regression case from the reported bug (screenshot/notebook reproduction)
# ---------------------------------------------------------------------------


def test_forever_and_ever_amen_regression():
    text = (
        'The song "Forever and Ever Amen" was written by Paul Overstreet and '
        "Don Schlitz, and recorded by Randy Travis [1][2]. It became Travis's "
        "third No. 1 single on the U.S. \"Billboard\" Hot Country Singles "
        "charts and won a Grammy for Best Country & Western Song and an "
        "Academy of Country Music award for Song of the Year [1]. Since "
        "then, the song has become a classic and has sold 966,000 digital "
        "copies [2]. In 1991, Schlitz also wrote a theme song for President "
        "George H. W. Bush's \"Points of Light\" program [3]."
    )
    spans = split_sentences_with_spans(text)
    assert len(spans) == 4

    last = spans[-1]
    assert last.text.strip() == (
        "In 1991, Schlitz also wrote a theme song for President George H. "
        "W. Bush's \"Points of Light\" program [3]."
    )
    assert text[last.start:last.end] == last.text
    assert "[3]" in text[last.start:last.end]
