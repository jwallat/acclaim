# Atomic claim extractor — intrinsic eval

Quantitative evaluation of [`AtomicClaimExtractor`](../../acclaim/claims/atomic.py).
Reference-free: coverage, atomicity, non-redundancy (and optionally
faithfulness) are checked directly against the source answer, so we can run
on many ALCE answers instead of 50 hand-labelled ones.

## What gets measured

All NLI scores use `cross-encoder/nli-deberta-v3-large` with an entailment
threshold of 0.5 (`--nli-model`, `--threshold`).

| Metric (`--metrics` name)              | Default | Definition                                                                 |
|----------------------------------------|---------|----------------------------------------------------------------------------|
| Coverage, sentence (`coverage`)        | on      | fraction of answer sentences with NLI(concat(top-3 claims by Jaccard overlap) → sentence) ≥ threshold (`--coverage-topk`) |
| Coverage, lexical (`sentence_claim_coverage`) | on | fraction of answer sentences that are the best Jaccard match of at least one claim |
| Atomicity (`atomicity`)                | on      | fraction of claims with no verb–verb `conj` dependency (spaCy `en_core_web_sm`); noun-phrase conjunctions are allowed |
| Non-redundancy (`non_redundancy`)      | on      | 1 − fraction of claims bidirectionally entailed by another claim           |
| Coverage, whole answer (`coverage_whole`) | off  | NLI(concat(claims) → answer); brittle on long answers (512-token NLI cap)  |
| Faithfulness (`faithfulness`)          | off     | mean NLI(answer → claim), plus fraction of claims ≥ threshold; off because claims are LLM rephrasings, so answer→claim NLI is not load-bearing |

Answers are split into sentences with the library's
[`SentenceClaimExtractor`](../../acclaim/claims/sentence.py), which also strips
citation markers.

## Data

The answers come from the [ALCE](https://github.com/princeton-nlp/ALCE)
framework. We generated them ourselves on ASQA dev (948 questions); the result
file is not checked in because of its size (~34 MB):

`asqa-Llama-3.1-8B-Instruct-gtr-shot2-ndoc5-43.json`, produced by ALCE's
`run.py` with config `configs/asqa_modern_shot2_ndoc5_gtr_default.yaml`,
eval file `data/asqa_eval_gtr_top100.json`, model
`meta-llama/Llama-3.1-8B-Instruct`, 2-shot, 5 docs, seed 43, temperature 1.0,
top-p 0.95 (the full args are stored under `"args"` in the file).

To re-run extraction you need this file (or any ALCE result JSON with a
`data` list of `{question, docs, output}`) and pass it via `--input`.
Everything downstream of extraction does not: the question, answer and
retrieved docs of every evaluated example are embedded in the
`predictions/*.jsonl` files and in the annotation sheet.

Examples are sampled with `random.Random(seed).sample(range(948), n)`; the
default `--n 100 --seed 42` gives the same 100 `alce_index` values for every
model.

`sampled_data.json` (see [Phase 2](#phase-2-not-yet-implemented)) predates
this and was sampled by `sample_data.ipynb` from an earlier ALCE run
(`asqa-Meta-Llama-3-8B-Instruct-gtr-shot1-ndoc3-43.json`) that is no longer
available; the notebook's hardcoded path will not resolve.

## One-time setup

Dependencies are listed in the repo-root [`requirements.txt`](../../requirements.txt):

```bash
micromamba run -n acclaim pip install -r requirements.txt
micromamba run -n acclaim python -m spacy download en_core_web_sm
```

Extraction needs an LLM endpoint reachable through LiteLLM: an OpenRouter key
(`OPENROUTER_API_KEY`) for `openrouter/...` models, or a local
OpenAI-compatible server such as vLLM (`--api-base`). Scoring downloads the
deberta NLI model and is much faster on a GPU.

## End-to-end

```bash
export OPENROUTER_API_KEY=...
ALCE=path/to/asqa-Llama-3.1-8B-Instruct-gtr-shot2-ndoc5-43.json

bash library_evals/atomic_claims/run_eval.sh --input "$ALCE"              # gemma-4-26b, 100 examples
bash library_evals/atomic_claims/run_eval.sh --input "$ALCE" --n 20       # smaller
MODEL=openrouter/openai/gpt-5.4-mini bash library_evals/atomic_claims/run_eval.sh --input "$ALCE"
```

`run_eval.sh` runs extraction then scoring and writes, per model (name = last
part of the model id):

- `predictions/<model>.jsonl`
- `results/<model>_per_example.jsonl`, `results/<model>_summary.json`

Flags: `--input`, `--n`, `--seed`, `--model`, `--api-base`;
`--extract-only` / `--score-only` run one half (e.g. extract on a laptop,
score on a GPU node). Everything else is forwarded to `intrinsic_eval.py`,
e.g. `--metrics ...`. Env overrides: `MODEL`, `API_BASE`, `API_KEY` (falls
back to `OPENROUTER_API_KEY`), `N`, `SEED`, `ALCE_INPUT`, `ENV_NAME`
(default `acclaim`).

## Manual run (each step explicit)

### 1. Extract atomic claims

```bash
micromamba run -n acclaim python -m library_evals.atomic_claims.run_extractor \
    --input "$ALCE" --n 100 --seed 42 \
    --model openrouter/google/gemma-4-26b-a4b-it --api-base "" \
    --output library_evals/atomic_claims/predictions/gemma-4-26b-a4b-it.jsonl
```

For a local vLLM server use e.g. `--model openai/<hf-name> --api-base
http://localhost:8000/v1` (the default `--api-base`). Output: one JSON record
per line with `question`, `answer`, `docs`, `predicted_claims`, `model`,
`alce_index` (plus `error` if extraction failed for that example).

### 2. Score the predictions

```bash
micromamba run -n acclaim python -m library_evals.atomic_claims.intrinsic_eval \
    --input library_evals/atomic_claims/predictions/gemma-4-26b-a4b-it.jsonl \
    --output-prefix library_evals/atomic_claims/results/gemma-4-26b-a4b-it \
    --metrics atomicity coverage coverage_whole non_redundancy sentence_claim_coverage
```

Writes `<prefix>_per_example.jsonl` (per-example sub-scores) and
`<prefix>_summary.json` (mean / std / fraction ≥ 0.9 per metric, plus the run
config) and prints the summary table. Other flags: `--limit N` for smoke
tests, `--coverage-topk`, `--threshold`, `--nli-model`, `--device`,
`--batch-size`.

### 3. Sanity-check the metrics

1. predicted_claims = whole answer → faithfulness ~1.0, coverage ~1.0, atomicity low
2. predicted_claims = claims from another example → faithfulness collapses
3. NLI spot-check on 20 hand-picked probes (negation, numbers, scope)

```bash
micromamba run -n acclaim python -m library_evals.atomic_claims.sanity_check \
    --input library_evals/atomic_claims/predicted_claims.jsonl
```

Checks 1+2 use the first `--limit` (default 20) records. `--skip-1-2` runs
just the NLI probes (no predictions needed).

### 4. Browse results

```bash
micromamba run -n acclaim jupyter lab library_evals/atomic_claims/report.ipynb
```

Reads the top-level `results_per_example.jsonl`, `results_summary.json` and
`predicted_claims.jsonl` (copies of the gemma-4-26b run). Renders the summary table, histograms, metric correlations,
and the 5 worst examples by faithfulness and coverage. The faithfulness cells
only work if scoring ran with `--metrics faithfulness ...`; the outputs saved
in the notebook are from an older 50-example run.

## Comparing multiple models

`run_eval_all_models.sh` runs `run_eval.sh` for seven OpenRouter models
(claude-haiku-4.5, claude-haiku-5.5, gemma-4-26b-a4b-it, gpt-5.4-mini, gpt-6-luna, qwen3.6-35b-a3b,
llama-3.1-8b-instruct) and forwards all arguments:

```bash
bash library_evals/atomic_claims/run_eval_all_models.sh --input "$ALCE" \
    --metrics atomicity coverage coverage_whole non_redundancy sentence_claim_coverage
```

Then:

```bash
micromamba run -n acclaim python -m library_evals.atomic_claims.compare_models
micromamba run -n acclaim python -m library_evals.atomic_claims.compare_models --csv comparison.csv
```

To refresh the top-level copies the notebook reads:

```bash
cd library_evals/atomic_claims
cp predictions/gemma-4-26b-a4b-it.jsonl predicted_claims.jsonl
cp results/gemma-4-26b-a4b-it_per_example.jsonl results_per_example.jsonl
cp results/gemma-4-26b-a4b-it_summary.json results_summary.json
```

`compare_models.py` scans `results/*_summary.json` and
prints one row per model across all metrics and the mean claim count.

## Results

Run on 2026-10-28: 100 ASQA answers (`--n 100 --seed 42`) from
`asqa-Llama-3.1-8B-Instruct-gtr-shot2-ndoc5-43.json`, scored with
`--metrics atomicity coverage coverage_whole non_redundancy sentence_claim_coverage`
(faithfulness not computed). Means per model, from `compare_models.py`:

| Model                 | Coverage (sent.) | Coverage (whole) | Coverage (lexical) | Atomicity | Non-redundancy | Claims / answer |
|-----------------------|-----------------:|-----------------:|-------------------:|----------:|---------------:|----------------:|
| claude-haiku-4.5      | 0.544            | 0.803            | 0.981              | 0.999     | 0.953          | 6.53            |
| claude-haiku-5.5      | 0.539            | 0.872            | 0.990              | 1.000     | 0.953          | 7.27            |
| gemma-4-26b-a4b-it    | 0.551            | 0.863            | 0.990              | 1.000     | 0.965          | 7.09            |
| gpt-5.4-mini          | 0.609            | 0.825            | 0.990              | 0.998     | 0.968          | 6.15            |
| gpt-6-luna            | 0.615            | 0.864            | 0.997              | 0.998     | 0.961          | 6.19            |
| llama-3.1-8b-instruct | 0.496            | 0.737            | 0.980              | 0.995     | 0.949          | 6.27            |
| qwen3.6-35b-a3b       | 0.621            | 0.805            | 0.985              | 0.999     | 0.955          | 6.27            |

- Atomicity, lexical coverage and non-redundancy are near ceiling for every
  model; they catch broken extractors but do not separate good ones.
- Sentence-level NLI coverage is the only metric that varies, but its
  standard error is ~0.04 per model (95% CI ≈ ±0.08), so only the extremes
  (llama-3.1-8b vs. qwen3.6 / gpt-6-luna / gpt-5.4-mini) are clearly apart.
  Whole-answer coverage is much higher (0.74–0.87) than sentence-level
  coverage for every model; we have not yet checked how much of the
  sentence-level gap is the top-3 claim selection or NLI strictness rather
  than missing content (`inspect_coverage.py` is the tool for that).

## Manual annotations

[`manual_annotations/atomic_annotation_sheet.csv`](manual_annotations/atomic_annotation_sheet.csv)
holds 100 claims extracted by gemma-4-26b-a4b-it from 62 of the 100 sampled
examples, each next to the answer sentence it was matched to. Every claim is
labelled `correct = 1`; 18 are noted as "not selfcomplete" (they need context
from the answer to be understood). The claims come from an earlier run of the
same extractor on the same examples, so they can differ slightly from the
current `predictions/gemma-4-26b-a4b-it.jsonl`.

## Debugging a single example

`inspect_coverage.py` prints, for one `alce_index`, each answer sentence
next to the Jaccard-selected top-K claims, the concat-NLI coverage score,
and the single-best-claim NLI score. Use it when an example's
`coverage_sentence` looks surprisingly low:

```bash
micromamba run -n acclaim python -m library_evals.atomic_claims.inspect_coverage \
    --predictions library_evals/atomic_claims/predicted_claims.jsonl --alce-index 25 --topk 3
```

## Files

| File                     | Purpose                                                           |
|--------------------------|-------------------------------------------------------------------|
| `run_extractor.py`       | Sample ALCE answers, run the extractor, write predictions JSONL   |
| `intrinsic_eval.py`      | NLI + spaCy scoring → per-example JSONL + summary JSON            |
| `sanity_check.py`        | The three metric sanity checks above                              |
| `inspect_coverage.py`    | Per-example debug view of the sentence-coverage scoring           |
| `compare_models.py`      | Cross-model comparison table from `results/*_summary.json`        |
| `run_eval.sh`            | Extract → score for one model                                     |
| `run_eval_all_models.sh` | `run_eval.sh` across the seven OpenRouter models                    |
| `predictions/`           | Extracted claims per model, `<model>.jsonl` (n=100, seed 42)      |
| `results/`               | Scores per model, `<model>_per_example.jsonl` / `<model>_summary.json` |
| `predicted_claims.jsonl`, `results_per_example.jsonl`, `results_summary.json` | Copies of the gemma-4-26b run, read by `report.ipynb` and as script defaults |
| `report.ipynb`           | Summary table + plots + worst-N qualitative inspection            |
| `manual_annotations/`    | Human check of 100 extracted claims (see above)                   |
| `sampled_data.json`      | 50 answers with human-written `atomic_claims` (Phase 2)           |
| `sample_data.ipynb`      | Notebook that produced `sampled_data.json`                        |

## Phase 2 (not yet implemented)

`compare_to_human.py` — bipartite NLI-matched P/R/F1 against the 50-example
human gold in `sampled_data.json`. Sanity check, not headline.
