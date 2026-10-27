# End-to-end walkthrough

This page follows a single answer through every stage of the `acclaim`
pipeline and shows what each stage produces. Everything below is real output
from [`end_to_end.py`](end_to_end.py), run with the bundled default config
(`acclaim/config/default.yaml`: atomic claims, check-worthiness filter, LLM
aligner, and Gemma 4 via OpenRouter as the judge). LLM outputs can vary
slightly between runs.

```bash
export OPENROUTER_API_KEY=...
python examples/end_to_end.py
```

## Input

A question, two source documents, and an answer that cites them with `[N]`
markers:

```python
question = "When was the Eiffel Tower built and how tall is it?"

documents = [
    Document(doc_id="eiffel_history", text=(
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
        "France. It was designed by Gustave Eiffel's engineering company and constructed "
        "from 1887 to 1889 as the centrepiece of the 1889 World's Fair.")),
    Document(doc_id="eiffel_facts", text=(
        "The tower is 330 metres tall, about the same height as an 81-storey building. "
        "It was the tallest man-made structure in the world until the Chrysler Building "
        "in New York City was finished in 1930.")),
]

answer = (
    "The Eiffel Tower was built between 1887 and 1889 for the World's Fair [1]. "
    "It is 300 metres tall [2]. "
    "It remained the tallest structure in the world until 1930 [3]. "
    "It is located on the Champ de Mars in Paris. "
    "It is arguably the most beautiful monument in Europe [1]."
)

result = evaluate(answer, documents, question=question, config=load_config())
```

The answer has several problems on purpose:

- It gets the height wrong: the document says 330 m, the answer says 300 m.
- It cites `[3]`, but only two documents exist.
- One sentence has no citation at all.
- One sentence is an opinion.

Every intermediate result is available in `result.steps`.

## 1. Claim extraction → `steps["claims"]`

[Module docs](../acclaim/claims/README.md)

The `atomic` extractor asks an LLM to split the answer into short claims that
each state one fact. It also resolves pronouns ("It" becomes "The Eiffel
Tower"), so each claim can be checked on its own. The LLM rewrites the text,
so the claims have no character span (`(-1, -1)`):

```
'The Eiffel Tower was built between 1887 and 1889.'                         span=(-1, -1)
"The Eiffel Tower was built for the World's Fair."                          span=(-1, -1)
'The Eiffel Tower is 300 metres tall.'                                      span=(-1, -1)
'The Eiffel Tower remained the tallest structure in the world until 1930.'  span=(-1, -1)
'The Eiffel Tower is located on the Champ de Mars.'                         span=(-1, -1)
'The Eiffel Tower is located in Paris.'                                     span=(-1, -1)
'The Eiffel Tower is arguably the most beautiful monument in Europe.'       span=(-1, -1)
```

The first sentence became two claims, and so did the fourth.

## 2. Claim filtering → `steps["filter_decisions"]`, `steps["filtered_claims"]`

[Module docs](../acclaim/filters/README.md)

The `check_worthiness` filter assigns each claim a category. Claims that need
no source are dropped and are not counted in any metric:

```
[keep] 'The Eiffel Tower was built between 1887 and 1889.'      (QUANTITY: Specific years of construction.)
[keep] "The Eiffel Tower was built for the World's Fair."       (EVENT_OR_HISTORICAL: Specific historical event/purpose.)
[keep] 'The Eiffel Tower is 300 metres tall.'                   (QUANTITY: Specific measurement.)
[keep] 'The Eiffel Tower remained the tallest structure ...'    (QUANTITY: Specific ranking and date.)
[keep] 'The Eiffel Tower is located on the Champ de Mars.'      (OTHER_FACTUAL: Specific location/landmark.)
[DROP] 'The Eiffel Tower is located in Paris.'                  (COMMON_KNOWLEDGE: Widely known location of a famous landmark.)
[DROP] 'The Eiffel Tower is arguably the most beautiful ...'    (SUBJECTIVE: Aesthetic judgment.)
```

5 of the 7 claims are kept. The opinion is dropped, so the `[1]` attached to
it doesn't count against citation correctness.

## 3. Citation alignment → `steps["aligned_claims"]`

[Module docs](../acclaim/alignment/README.md)

By default, `[N]` refers to `documents[N-1]`. You can override this with
`citation_to_doc=`. Atomic claims have no span, so the `llm` aligner asks an
LLM which markers in the original answer back each claim:

```
citation_to_doc = {1: 'eiffel_history', 2: 'eiffel_facts'}

'The Eiffel Tower was built between 1887 and 1889.'                         -> ['eiffel_history']
"The Eiffel Tower was built for the World's Fair."                          -> ['eiffel_history']
'The Eiffel Tower is 300 metres tall.'                                      -> ['eiffel_facts']
'The Eiffel Tower remained the tallest structure in the world until 1930.'  -> ['invalid:3']
'The Eiffel Tower is located on the Champ de Mars.'                         -> []
```

`[3]` has no matching document, so the aligner keeps it as `invalid:3`
instead of dropping it silently. This is what the `hallucination_rate`
metric counts. The Champ de Mars claim had no marker in the answer, so it is
aligned to no documents.

## 4. Evidence judgement → `steps["claim_results"]`

[Module docs](../acclaim/judges/README.md)

The judge reads each claim together with the documents it cites, and labels
it `SUPPORTED`, `REFUTED` or `UNCLEAR`. A claim with no valid cited document
gets `UNCLEAR` without an LLM call:

```
'The Eiffel Tower was built between 1887 and 1889.'
  SUPPORTED (confidence=1.00): The document explicitly states the Eiffel Tower was constructed from 1887 to 1889.
"The Eiffel Tower was built for the World's Fair."
  SUPPORTED (confidence=1.00): The document explicitly states the tower was constructed as the centrepiece of the 1889 World's Fair.
'The Eiffel Tower is 300 metres tall.'
  REFUTED (confidence=1.00): The document states the tower is 330 metres tall, which directly contradicts the claim that it is 300 metres tall.
'The Eiffel Tower remained the tallest structure in the world until 1930.'
  UNCLEAR (confidence=0.00): No documents provided for this claim.
'The Eiffel Tower is located on the Champ de Mars.'
  UNCLEAR (confidence=0.00): No documents provided for this claim.
```

The 1930 claim is actually supported by `eiffel_facts`, but the answer cited
`[3]` for it. `acclaim` only checks the documents a claim actually cites, so
a claim that is true but wrongly cited does not count as supported.

## 5. Metrics → `result.metrics`

[Module docs](../acclaim/metrics/README.md)

| Metric | Value | Why |
|---|---|---|
| `coverage` | 0.80 | 4 of the 5 kept claims cite something (the Champ de Mars claim cites nothing). |
| `citation_correctness` | 0.50 | 2 of those 4 cited claims are SUPPORTED. |
| `supported_claim_rate` | 0.40 | 2 of all 5 kept claims are SUPPORTED. |
| `hallucination_rate` | 0.25 | 1 of the 4 citation references (`invalid:3`) points to a document that doesn't exist. |
| `citation_number` | 4 | Number of `[N]` markers in the raw answer, including the one on the dropped opinion. |
| `citation_lengths` | 12.75 | Average token length of the cited claims. |
| `token_overlap` | 0.18 | Average Jaccard overlap between each cited claim and the closest matching sentence in its cited document. The atomic claims are rephrased, so this is low. |
| `cvcp` | 0.00 | Each sentence has only one citation, at its end, so citation positions within a sentence don't vary. |

## Output

`result.pretty_print()` prints a readable summary of the claims, the dropped
claims and the metrics. `result.to_jsonl("result.jsonl")` saves the answer,
documents, per-claim results, filter decisions, metrics and run metadata
(package version and full config). Pass `include_steps=True` to also include
every intermediate stage.

To run the pipeline on a dataset, use `evaluate_batch()` (see the
[main README](../README.md#batch-evaluation)). The resulting JSONL files can
be browsed in the [`acclaim_explorer`](../acclaim_explorer/README.md) web
app.
