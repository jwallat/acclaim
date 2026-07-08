# Judge Evaluation

Evaluates how well the `LiteLLMJudge` agrees with human judgments on
(claim, document) pairs sampled from the atomic claims experiment.

## Workflow

### 1. Sample pairs

```bash
python sample_pairs.py
```

Reads `../atomic_claims/predicted_claims.jsonl`, samples 100 (claim, single-doc)
pairs stratified across source examples, and writes **`pairs.jsonl`** and
**`annotation_sheet.csv`** (blank `human_label` column ready for labeling).

Options:
```
--input   Path to predicted_claims.jsonl  (default: ../atomic_claims/predicted_claims.jsonl)
--output  Output path for pairs           (default: pairs.jsonl)
--sheet   Output path for annotation CSV  (default: annotation_sheet.csv)
--n       Number of pairs to sample       (default: 100)
--seed    Random seed                     (default: 42)
```

---

### 2. Annotate by hand

```bash
python annotate.py
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
--pairs   Path to pairs.jsonl       (default: pairs.jsonl)
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
python run_judges.py
```

Reads `pairs.jsonl`, calls the judge on each pair, and writes
**`predictions_<model>.jsonl`** (model name sanitized from the `model`
string, e.g. `predictions_gpt-5.4-mini.jsonl`) with the label, confidence,
and reason for each pair. Because the output filename is derived from the
model, rerunning `run_judges.py` after swapping `config.yaml`'s model
accumulates one predictions file per model instead of overwriting the last
run — this is what backs the multi-model comparison in step 5.

Options:
```
--config    Path to config YAML    (default: config.yaml)
--pairs     Path to pairs.jsonl    (default: pairs.jsonl)
--output    Path for predictions   (ignored — always predictions_<model>.jsonl)
```

---

### 4. Compute agreement

```bash
python compute_agreement.py
```

Joins your labels from `annotation_sheet.csv` with judge predictions from
`predictions.jsonl` on `pair_id`, then prints:
- Overall accuracy and Cohen's κ
- Per-label precision / recall / F1
- Confusion matrix (rows = human, columns = judge)

Results are also written to **`agreement_results.json`**.

Options:
```
--human        Path to annotated CSV      (default: annotation_sheet.csv)
--predictions  Path to predictions JSONL  (default: predictions.jsonl)
--output       Path for JSON results      (default: agreement_results.json)
```

---

### 5. Compare across judge models

Once `predictions_<model>.jsonl` exists for more than one model (repeat
steps 3–4 per model), two comparison angles are available:

**Agreement with humans, per model** — `compare_models.py` in `atomic_claims/`
has a same-named but different sibling here: this one reports
*inter-model* agreement (no human labels needed), computing pairwise %
match and Cohen's κ across every `predictions*.jsonl` in the directory, plus
a per-model label distribution:

```bash
python compare_models.py
python compare_models.py --dir path/to/dir --output inter_model_agreement.json
```

**Old vs. new prompt** — `plot_agreement.py` plots accuracy and κ (vs. human
labels, from each model's `agreement_predictions_<model>.json`) side by side
for two prompt versions. `old_prompt/` and `old_preds/` are frozen snapshots
of predictions/agreement results from before the judge prompt was rewritten
(see `config.yaml`'s system prompt), kept so this comparison stays
reproducible:

```bash
python plot_agreement.py                       # old_prompt/ vs current dir
python plot_agreement.py --old old_prompt/ --new . --output comparison.png
```

---

## Output files

| File | Description |
|------|-------------|
| `pairs.jsonl` | 100 sampled (claim, doc) pairs |
| `annotation_sheet.csv` | Pairs with `human_label` column for annotation |
| `predictions_<model>.jsonl` | Judge output for each pair (label, confidence, reason), one file per model run |
| `agreement_predictions_<model>.json` / `agreement_results.json` | Accuracy, kappa, per-label F1, confusion matrix vs. human labels |
| `inter_model_agreement.json` | Pairwise inter-model % match + Cohen's κ, from `compare_models.py` |
| `prompt_comparison.png` | Old-vs-new prompt accuracy/κ bar chart, from `plot_agreement.py` |
| `old_prompt/`, `old_preds/` | Frozen predictions/agreement results from before the judge prompt rewrite |
