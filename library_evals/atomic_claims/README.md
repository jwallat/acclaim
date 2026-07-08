# Atomic claim extractor — intrinsic eval

Quantitative evaluation of [`AtomicClaimExtractor`](../../acclaim/claims/atomic.py).
Reference-free: faithfulness, coverage, atomicity and non-redundancy are
checked directly against the source answer, so we can run on hundreds of
ALCE answers instead of 50 hand-labelled ones. See [PLAN.md](PLAN.md) for
the full rationale.

## What gets measured

| Metric          | Definition                                                                |
|-----------------|---------------------------------------------------------------------------|
| Faithfulness    | mean NLI(answer → claim); fraction of claims with entailment ≥ threshold  |
| Coverage (sent) | fraction of answer sentences entailed by some predicted claim             |
| Coverage (full) | NLI(concat(claims) → answer)                                              |
| Atomicity       | fraction of claims with no clause-level conjunction (spaCy rule)          |
| Non-redundancy  | 1 − fraction of claims bidirectionally entailed by another claim          |

## One-time setup

Dependencies (not yet wired into `pyproject.toml`):

```bash
micromamba run -n attribution-eval pip install transformers torch spacy matplotlib pandas
micromamba run -n attribution-eval python -m spacy download en_core_web_sm
```

A vLLM server is required for the extraction step. The repo ships
[`start_vllm.sh`](../../start_vllm.sh) (CPU) and [`start_vllm_gpu.sh`](../../start_vllm_gpu.sh)
(GPU/SLURM).

## End-to-end (minimal)

### Option A — L3S "interweb" endpoint (no local server)

The fastest path. Uses `https://interweb.l3s.uni-hannover.de/v1` with the
key already wired in `exploration.py`.

```bash
bash run_eval_l3s.sh                              # 500 examples, llama3.1:8b
bash run_eval_l3s.sh --n 50                       # smoke run
MODEL=openai/qwen3.5:4b bash run_eval_l3s.sh      # swap model
L3S_API_KEY=... bash run_eval_l3s.sh              # override key
```

### Option B — local vLLM

Start vLLM in another shell, then:

```bash
bash run_eval.sh                                  # default: 500 examples
bash run_eval.sh --n 100 --skip-whole-coverage    # smaller / faster
```

Both wrappers run extraction → intrinsic eval → print the summary table.
Open [`report.ipynb`](report.ipynb) afterwards for plots and the worst-N
qualitative inspection.

`run_eval.sh` honours `MODEL`, `API_BASE`, `API_KEY`, `N`, `SEED` env vars
and forwards extra flags to `intrinsic_eval.py`.

## Manual run (if you want each step explicit)

### 1. Extract atomic claims

Needs a vLLM server (or any LiteLLM-compatible endpoint).

```bash
micromamba run -n attribution-eval python -m library_evals.atomic_claims.run_extractor \
    --n 500 --seed 42 \
    --model openai/meta-llama/Meta-Llama-3-8B-Instruct \
    --api-base http://localhost:8000/v1 \
    --output library_evals/atomic_claims/predicted_claims.jsonl
```

Defaults sample 500 ALCE/ASQA answers from
`/home/wallat/citation_eval/ALCE/result/asqa-Meta-Llama-3-8B-Instruct-gtr-shot1-ndoc3-43.json`.
Output: one JSON record per line with `question`, `answer`, `docs`,
`predicted_claims`, `alce_index`.

### 2. Score the predictions

NLI-heavy; runs on GPU if `torch.cuda.is_available()`, otherwise CPU.

```bash
micromamba run -n attribution-eval python -m library_evals.atomic_claims.intrinsic_eval \
    --input  library_evals/atomic_claims/predicted_claims.jsonl \
    --output-prefix library_evals/atomic_claims/results \
    --batch-size 32 --verbose
```

Writes:
- `results_per_example.jsonl` — per-example sub-scores (one row per input)
- `results_summary.json` — dataset-wide aggregates + the run config

The summary table is also printed to stdout.

Useful flags: `--limit N` for smoke tests, `--skip-whole-coverage` (saves
one NLI pass per example), `--skip-redundancy` (saves the O(n²) pairwise
pass), `--nli-model <hf-id>` to swap models.

### 3. Sanity-check the metrics

Implements the three checks from `PLAN.md`:
1. predicted_claims = whole answer → faithfulness ~1.0, coverage ~1.0, atomicity low
2. predicted_claims = claims from another example → faithfulness collapses
3. NLI spot-check on 20 hand-picked probes (negation, numbers, scope)

```bash
micromamba run -n attribution-eval python -m library_evals.atomic_claims.sanity_check \
    --input library_evals/atomic_claims/predicted_claims.jsonl
```

Skip checks 1+2 with `--skip-1-2` to run just the NLI probes (no
predictions needed).

### 4. Browse results

```bash
micromamba run -n attribution-eval jupyter lab library_evals/atomic_claims/report.ipynb
```

Renders the summary table, faithfulness/coverage/claim-count histograms,
metric correlations, and the 5 worst examples by faithfulness and coverage.

## Comparing multiple models

`run_eval_all_models.sh` reruns extraction + intrinsic eval for the same set
of OpenRouter models used by the judge eval
(`library_evals/judges/config.yaml`), so both halves of the pipeline get
evaluated on comparable model strength:

```bash
OPENROUTER_API_KEY=... bash run_eval_all_models.sh          # all 5 models, 500 examples
OPENROUTER_API_KEY=... bash run_eval_all_models.sh --n 100 --seed 42
```

Each model writes its own `predicted_claims_<model>.jsonl`,
`results_<model>_per_example.jsonl`, and `results_<model>_summary.json`
(unsuffixed `predicted_claims.jsonl` / `results_*.jsonl` are the
single-model default run). `run_slurm.sh` submits the same sweep as a SLURM
job (see its `#SBATCH` header for resource requests).

Once summaries exist for multiple models:

```bash
mamba run -n attribution-eval python compare_models.py                # print table
mamba run -n attribution-eval python compare_models.py --csv comparison.csv
```

Scans `results_*_summary.json` and prints one row per model across
faithfulness, coverage, atomicity, non-redundancy and mean claim count.

## Debugging a single example

`inspect_coverage.py` prints, for one `alce_index`, each answer sentence
next to the Jaccard-selected top-K claims, the concat-NLI coverage score,
and the single-best-claim NLI score — use it when an example's
`coverage_sentence` looks surprisingly low and you want to see which
sentence/claim pairing the metric is scoring:

```bash
mamba run -n attribution-eval python -m library_evals.atomic_claims.inspect_coverage \
    --predictions predicted_claims.jsonl --alce-index 25 --topk 3
```

## Files

| File                  | Purpose                                                           |
|-----------------------|-------------------------------------------------------------------|
| `run_extractor.py`    | Sample ALCE answers, run the extractor, write `predicted_claims.jsonl` |
| `intrinsic_eval.py`   | NLI + spaCy scoring → per-example JSONL + summary JSON            |
| `sanity_check.py`     | Three verification scenarios from PLAN.md                         |
| `inspect_coverage.py` | Per-example debug view of the sentence-coverage scoring           |
| `compare_models.py`   | Cross-model comparison table from `results_*_summary.json`        |
| `run_eval.sh`         | One-shot driver for steps 1 → 2, single model                     |
| `run_eval_l3s.sh`     | Same as `run_eval.sh` but against the L3S interweb endpoint       |
| `run_eval_all_models.sh` | Reruns `run_eval.sh` across the 5 OpenRouter models compared with `judges/` |
| `run_slurm.sh`        | SLURM submission wrapper for `run_eval_all_models.sh`             |
| `report.ipynb`        | Summary table + plots + worst-N qualitative inspection            |
| `sampled_data.json`   | 50-example human-gold subset (Phase 2, not used here)             |
| `sample_data.ipynb`   | Notebook that produced `sampled_data.json`                        |
| `PLAN.md`             | Design rationale                                                  |

## Phase 2 (not yet implemented)

`compare_to_human.py` — bipartite NLI-matched P/R/F1 against the 50-example
human gold in `sampled_data.json`. Sanity check, not headline.
