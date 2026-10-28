# Judge Evaluation

Evaluates how well the `LiteLLMJudge` agrees with human judgments on
(claim, document) pairs sampled from the atomic claims experiment.

Run every command below from this directory (`library_evals/judges/`) in the
`acclaim` env; `summarize_runs.py` imports `compute_agreement` as a top-level
module. Some examples use `M="micromamba run -n acclaim"` as shorthand.

## Workflow

### 1. Sample pairs

```bash
micromamba run -n acclaim python sample_pairs.py
```

Reads `../atomic_claims/predicted_claims.jsonl`, samples 100 (claim, single-doc)
pairs stratified across source examples, and writes **`data/pairs.jsonl`** and
**`annotation_sheet.csv`** (blank `human_label` column ready for labeling).

Options:
```
--input   Path to predicted_claims.jsonl  (default: ../atomic_claims/predicted_claims.jsonl)
--output  Output path for pairs           (default: data/pairs.jsonl)
--sheet   Output path for annotation CSV  (default: annotation_sheet.csv)
--n       Number of pairs to sample       (default: 100)
--seed    Random seed                     (default: 42)
```

---

> **Note:** the existing `data/pairs.jsonl` is the exact sample behind the
> committed `annotation_sheet.csv` (all 100 rows verified on claim, doc_id and
> text prefix). It was sampled from an earlier 100-example atomic-claim
> extraction over Llama-3-8B-Instruct ALCE/ASQA answers (shot1, ndoc3) that is
> not in the repo. `../atomic_claims/predicted_claims.jsonl` is now the
> gemma-4-26b run (see `../atomic_claims/README.md`), so re-running
> `sample_pairs.py` draws a different sample; don't regenerate it unless you
> also re-annotate.

### 2. Annotate by hand

```bash
micromamba run -n acclaim python annotate.py
```

A terminal annotation tool that displays each (claim, document) pair one at a time
and records your judgment in **`annotation_sheet.csv`**.

Keys:

| Key | Action |
|-----|--------|
| `S` | SUPPORTED — document clearly supports the claim |
| `R` | REFUTED — document contradicts the claim |
| `U` | UNCLEAR — document is insufficient, ambiguous, or irrelevant |
| `→` or `n` | Next pair (skip without labeling) |
| `←` or `p` | Previous pair |
| `g` | Jump to first unlabeled pair |
| `q` | Quit |

Progress is saved after every keystroke. You can quit and resume at any time —
the tool always starts at the first unlabeled pair.

Options:
```
--pairs   Path to pairs.jsonl       (default: data/pairs.jsonl)
--output  Path to annotation CSV    (default: annotation_sheet.csv)
```

---

### 3. Configure and run the judge

Edit **`config.yaml`**, uncommenting the `judge.model` line you want (several
OpenRouter models are listed, commented out, as a quick swap list):

```yaml
judge:
  model: "openrouter/openai/gpt-5.4-mini"   # any LiteLLM model string
  api_base: null
  api_key: ${OPENROUTER_API_KEY}
```

Then run:

```bash
micromamba run -n acclaim python run_judges.py
```

Reads `data/pairs.jsonl`, calls the judge on each pair, and writes
**`predictions/<model>.jsonl`** (model name sanitized from the `model`
string, e.g. `predictions/gpt-5.4-mini.jsonl`) with the label, confidence,
and reason for each pair. Because the output filename is derived from the
model, rerunning `run_judges.py` after swapping `config.yaml`'s model
accumulates one predictions file per model instead of overwriting the last
run — this is what backs the multi-model comparison in step 5.

Options:
```
--config        Path to config YAML    (default: config.yaml)
--pairs         Path to pairs.jsonl    (default: data/pairs.jsonl)
--output        Path for predictions   (default: predictions/<run>.jsonl, see below)
--model         Override judge.model
--backend       Override judge.backend: chat | decision
--mode          Override judge.decision_mode: choice | noul | score
--state-format  Override judge.decision_state_format: structured | text
--thresholds    noul thresholds as JSON, e.g. '{"supported": 0.6, "refuted": 0.4}'
--sleep         Seconds between pairs (e.g. 1 for free-tier rate limits; default 0)
```

Each prediction also records `latency_s`, `cost_usd` (OpenRouter-reported),
and for decision models the label distribution (`judge_probabilities`) and
raw answers (`decision_answers`).

#### Decision models

The judge is built the same way the library builds it, so decision models
(OpenRouter Decisions API, e.g. `typesafe/jev-1.13`) work through the
`backend: decision` setting:

```bash
M="micromamba run -n acclaim"

# Recommended: choice mode (the default, so --mode can be omitted)
$M python run_judges.py --backend decision --model openrouter/typesafe/jev-1.13 \
    --output predictions/jev-1.13_choice_v3.jsonl
$M python run_judges.py --backend decision --model openrouter/microsoft/microsoft-decision-1 \
    --output predictions/microsoft-decision-1_choice_v3.jsonl

# Alternatives, for comparison
$M python run_judges.py --backend decision --model openrouter/typesafe/jev-1.13 --mode score --output predictions/jev-1.13_score_v3.jsonl
$M python run_judges.py --backend decision --model openrouter/typesafe/jev-1.13 --mode noul --output predictions/jev-1.13_noul_v3.jsonl
$M python run_judges.py --backend decision --model openrouter/typesafe/jev-1.13 --state-format text --output predictions/jev-1.13_choice_text_v3.jsonl
```

Modes (see `acclaim/judges/decision.py`). All three use the same prompt, which
mirrors the chat judge's `system_prompt` (same label definitions and examples):

- `choice` (default, recommended): one choice question over SUPPORTED / REFUTED / UNCLEAR.
- `score`: an ordinal scale REFUTED < UNCLEAR < SUPPORTED; label = most probable level.
- `noul`: two yes/no questions (supported? contradicted?) combined with thresholds (default 0.5).

Without `--output`, decision runs are written to
`predictions/<model>_<mode>[_text].jsonl`.

> **Prompt versions and file names.** Only the current (v3) decision prompt
> is in the code, and prediction files do not record which prompt produced
> them. Committed decision files without a version suffix (`_choice`,
> `_score`, `_noul`, `_choice_text`) are **v1** runs and cannot be
> regenerated; `_v2` / `_v3` files were written with an explicit `--output`.
> Only the runs with the current prompts are in git (`*_v3.jsonl` and the
> chat runs); the v1/v2 files are gitignored, so their rows in
> `results_summary.md` and the tables below can only be rebuilt from a local
> copy.
> A new run without `--output` would silently overwrite a v1 file, so always
> pass `--output` with a version suffix, as above.

---

### 4. Compute agreement

```bash
micromamba run -n acclaim python compute_agreement.py --predictions predictions/jev-1.13_choice_v3.jsonl
micromamba run -n acclaim python compute_agreement.py --predictions-dir predictions   # every run
```

Joins your labels from `annotation_sheet.csv` with one predictions file on
`pair_id` (`--predictions` or `--predictions-dir` is required), then prints:
- Overall accuracy, Cohen's κ and macro-F1, with bootstrap 95% CIs
- Calibration of the judge's confidence (ECE, Brier)
- Per-label precision / recall / F1
- Confusion matrix (rows = human, columns = judge)

Results are also written to **`results/agreement_results.json`** (with
`--predictions-dir`: `results/agreement_<run>.json` per file).

Options:
```
--human            Path to annotated CSV               (default: annotation_sheet.csv)
--predictions      Path to one predictions JSONL
--predictions-dir  Score every *.jsonl in this directory (e.g. predictions/)
--output           Path for JSON results               (default: results/agreement_results.json)
```

---

### 5. Compare across judge models

Agreement with the human labels per run is covered by `summarize_runs.py`
(step 6). `compare_models.py` (not the same as `../atomic_claims/compare_models.py`)
reports **inter-model** agreement instead, with no human labels needed:
pairwise % match and Cohen's κ across every file in `predictions/`, plus a per-model label distribution:

```bash
micromamba run -n acclaim python compare_models.py
micromamba run -n acclaim python compare_models.py --dir path/to/predictions --output results/inter_model_agreement.json
```

---

### 6. Summary table

```bash
micromamba run -n acclaim python summarize_runs.py
```

Summarises every `predictions/*.jsonl` into **`results_summary.md`**:
accuracy, Cohen's κ and macro-F1 with bootstrap 95% CIs, calibration
(ECE / Brier of the judge's confidence in its own label), predicted label
distribution, failures, mean latency and total cost. For `noul` runs it adds
a row with thresholds tuned by stratified 5-fold cross-validation (the
0.5-threshold row remains the headline number). With only 100 pairs (4
REFUTED), CIs are wide, so differences of a few points are not meaningful.

## Results: decision models vs. chat LLMs (2026-10-13)

The full table is in [`results_summary.md`](results_summary.md). It covers 5 decision models × 4 configs, the v2/v3 choice-prompt reruns, and 3 chat baselines, all on the 100 annotated pairs.

| Judge | κ [95% CI] | Acc | ECE | Latency/pair | Cost/100 pairs |
|---|---|---|---|---|---|
| **jev-1.13, choice (v3, current default)** | **0.728** [0.60, 0.85] | 0.86 | 0.093 | 0.29 s | $0.0040 |
| jev-1.13, score (v1) | 0.727 [0.60, 0.86] | 0.86 | 0.086 | 0.26 s | $0.0025 |
| jev-1.13, noul (0.5, v1) | 0.709 [0.58, 0.84] | 0.85 | 0.076 | 0.29 s | $0.0026 |
| **microsoft-decision-1, choice (v3, current default)** | 0.703 [0.56, 0.83] | 0.85 | 0.089 | 0.37 s | $0.0028 |
| microsoft-decision-1, score (v1) | 0.702 [0.57, 0.82] | 0.84 | 0.082 | 0.31 s | $0.0013 |
| jev-1.13, choice (v1) | 0.695 [0.57, 0.82] | 0.84 | 0.101 | 0.27 s | $0.0025 |
| gpt-6-luna (chat) | 0.681 [0.54, 0.82] | 0.84 | 0.134 | 2.79 s | $0.0130 |
| gemma-4-26b-a4b-it (chat, library default) | 0.677 [0.53, 0.82] | 0.84 | 0.154 | 2.06 s | $0.0069 |
| pplx-decider-27b, choice | 0.668 [0.53, 0.80] | 0.83 | 0.078 | 0.32 s | $0.0007 |
| gpt-6-luna-decisions, choice | 0.652 [0.52, 0.79] | 0.82 | 0.144 | 0.37 s | $0.0036 |
| kev-4b, choice | 0.628 [0.49, 0.76] | 0.81 | 0.076 | 0.82 s | $0.0012 |
| claude-sonnet-5.5 (chat) | 0.604 [0.45, 0.75] | 0.80 | 0.076 | 2.13 s | $0.3056 |

Findings:
- The best decision models are about as accurate as the best chat judges, and better calibrated. The v3 choice defaults (jev-1.13, microsoft-decision-1) are 6–10× faster and 2–110× cheaper than the three chat judges. All 95% CIs in the table overlap heavily, so no ranking among the top runs is established by this data.
- Structured state (a `{claim, evidence}` object) beats a plain-text state for 4 of 5 models (microsoft-decision-1 is the exception; for jev choice κ 0.695 vs 0.619). Keep `structured` as the default.
- No mode wins for every model. With the v1 prompt, `score` was best for the two strongest (jev, microsoft); with the v3 prompt, `choice` matches it (see below). The default 0.5 thresholds in `noul` work well. Thresholds tuned by cross-validation are unstable across folds and do not help on average. gpt-6-luna-decisions returns near-hard 0/1 probabilities (83% of its `noul` answers are below 0.02 or above 0.98, vs. 7% for jev-1.13), so thresholds are meaningless for it.
- The Decisions API does not make gpt-6-luna more accurate than the same model through chat (κ 0.652 vs 0.681). It only makes it faster and cheaper.
- Prompt versions: v1 was a short definition-only prompt (all non-choice rows above use it). v2 was a rules-plus-examples choice prompt. v3, the current code default for all modes, mirrors the chat judge's `system_prompt` so that chat and decision judges are compared on identical label definitions and examples.
- v3 choice (2026-10-13) is the best configuration for microsoft-decision-1 and ties the best for jev-1.13. microsoft-decision-1 rose from κ 0.660 (v1) / 0.655 (v2) to 0.703, because the v3 prompt is not as strict as v2: SUPPORTED→UNCLEAR errors fell from 8 to 2. jev-1.13 rose from 0.695 (v1) to 0.728, close to v2 (0.737). Both are above the gemma chat baseline (0.677), which uses the same prompt, but the per-run CIs overlap widely, so none of these differences are established on 100 pairs. With choice now matching score for both models, choice is the recommended mode: it is the simplest to read, and its probabilities map directly to the three labels.
- Historical v2 result: a fuller choice prompt (explicit rules plus 4 worked examples not taken from the eval set) moved jev-1.13 choice from κ 0.695 to **0.737**, the best single configuration, and cut UNCLEAR→SUPPORTED errors from 8 to 3. microsoft-decision-1 did not change (0.660 → 0.655): it became too strict, so SUPPORTED→UNCLEAR errors rose from 1 to 8 on paraphrased or implicitly referenced support. Input cost rose about 2–3×, to $0.003–0.005 per 100 pairs. The v1 rows in `results_summary.md` use the previous, shorter prompt. Caveat: the rules were written after looking at errors on these same 100 pairs, so the v2 numbers are somewhat optimistic.
- Two of the three few-shot examples in the chat-LLM judge prompt (`config.yaml`: the July 16 concert and the Colorado River / India hydroelectric examples) are pairs 43 and 0 of this eval set. The v3 decision prompt inherits this. In practice the effect is negligible: every chat and decision run (including v1/v2, which never saw these examples) labels both pairs correctly, and excluding them lowers every κ by 0.009–0.014 without changing the ranking.
- Only 4 pairs are REFUTED, so macro-F1 depends mostly on those 4 pairs. A larger gold set is needed before choosing a default model with confidence.

## Output files

| File | Description |
|------|-------------|
| `data/pairs.jsonl` | 100 sampled (claim, doc) pairs |
| `annotation_sheet.csv` | Pairs with `human_label` column for annotation |
| `predictions/<run>.jsonl` | Judge output for each pair (label, confidence, reason, latency, cost), one file per run; see the prompt-version note in step 3 |
| `results/agreement_results.json` / `results/agreement_<run>.json` | Accuracy, kappa, per-label F1, confusion matrix vs. human labels, from `compute_agreement.py` (not committed) |
| `results_summary.md` | Comparison table over all runs, from `summarize_runs.py` |
| `results/inter_model_agreement.json` | Pairwise inter-model % match + Cohen's κ, from `compare_models.py` (not committed) |
