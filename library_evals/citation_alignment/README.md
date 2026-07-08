# Citation Alignment Accuracy Evaluation

## What this measures

Atomic claims are LLM rephrasings of answer sentences and don't carry exact
character spans. `JaccardCitationAligner` (`acclaim/alignment/jaccard.py`)
matches each atomic claim to the answer sentence with the highest token
Jaccard overlap, then has the claim inherit that sentence's `[N]` citation
marker(s). This is a heuristic, not ground truth — a wrong sentence match
silently misattributes a claim to the wrong source document, which then
propagates into every downstream metric (precision, recall, F1, etc.).

This folder produces the human-annotated number that backs the citation
alignment paragraph in the paper: the percentage of atomic claims where the
predicted citation marker(s) actually match the marker(s) on the claim's
*true* source sentence in the answer.

## Pipeline

1. **`sample_answers.py`** → `sampled_answers.jsonl`
   Samples 50 ASQA answers with no duplicate questions. The three model
   outputs (`outputs/alce_eval/asqa_{gemma-4-e4b,llama-3.1-8b,qwen3-8b}.jsonl`)
   evaluate the identical 51 ASQA questions in the same order, so each of the
   50 sampled question indices is assigned to exactly one model (seeded
   shuffle, ~evenly split across the three), and one question is dropped to
   land on exactly 50. No extraction or alignment is re-run — the sampled
   `claims` are copied as-is from the existing evaluated outputs, since
   `JaccardCitationAligner` already overwrote each claim's `span` with its
   matched sentence's span and populated `citation_doc_ids`.

2. **`build_annotation_sheet.py`** → `claim_pairs.jsonl` + `annotation_sheet.csv`
   Expands each sampled answer into one row per atomic claim.

3. **Manual annotation** — fill in the `correct` column of
   `annotation_sheet.csv` (e.g. `y`/`n`).

4. **`compute_accuracy.py`** → `accuracy_results.json`
   Reads the filled-in sheet and reports overall accuracy plus a per-model
   breakdown (diagnostic only — the paper number is the overall accuracy).

## `annotation_sheet.csv` columns

| Column | Meaning |
|---|---|
| `pair_id` | Stable row identifier |
| `example_id`, `model`, `question` | Which sampled example this claim comes from |
| `answer` | The full model answer, with inline `[N]` markers — read this to find the claim's actual source sentence |
| `claim_text` | The extracted atomic claim |
| `matched_sentence_text` | The sentence in `answer` that `JaccardCitationAligner` matched the claim to. **This is a convenience hint, not ground truth** — it's exactly the sentence the predicted `citation_doc_ids` were inherited from |
| `predicted_citation_doc_ids` | The citation marker(s) the aligner assigned to this claim |
| `correct` | Fill in by hand: does `predicted_citation_doc_ids` match the marker(s) on the claim's *true* source sentence (found by reading `answer`, not just trusting `matched_sentence_text`)? |

To judge `correct`: read `claim_text`, find which sentence in `answer` it
actually paraphrases, note that sentence's `[N]` marker(s), and check whether
they match `predicted_citation_doc_ids`. If `matched_sentence_text` already is
the true source sentence, the check is trivial; the interesting failure mode
is when the aligner picked the *wrong* sentence (`matched_sentence_text` !=
true source) — those should be marked incorrect if the doc ids differ.

## Running it

```bash
mamba run -n attribution-eval python sample_answers.py
mamba run -n attribution-eval python build_annotation_sheet.py
# ... fill in the `correct` column in annotation_sheet.csv by hand ...
mamba run -n attribution-eval python compute_accuracy.py
```
