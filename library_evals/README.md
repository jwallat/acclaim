# Library Evals

Standalone evaluation experiments that validate individual components of the
`acclaim` pipeline (claim extraction → citation alignment → evidence judging
→ metrics). These are not unit tests — they measure quality against human
judgment or reference-free intrinsic checks, and back the numbers reported
in the paper.

Each subfolder is a self-contained experiment: sampling script(s) →
(optional manual annotation) → scoring script → results. Intermediate and
result files (`*.jsonl`, `*_summary.json`, `*.csv`) are committed so the
numbers are reproducible without rerunning LLM calls.

## Folders

| Folder | Component under test | What it measures |
|---|---|---|
| [`atomic_claims/`](atomic_claims/README.md) | `AtomicClaimExtractor` (`acclaim/claims/atomic.py`) | Intrinsic (reference-free) quality of LLM-based claim decomposition: faithfulness, coverage, atomicity, non-redundancy. Runs across several models via OpenRouter. |
| [`citation_alignment/`](citation_alignment/README.md) | `JaccardCitationAligner` (`acclaim/alignment/jaccard.py`) | Human-annotated accuracy of matching atomic claims back to the citation marker(s) on their true source sentence. |
| [`judges/`](judges/README.md) | `LiteLLMJudge` (`acclaim/judges/litellm.py`) | Agreement between the judge's SUPPORTED/REFUTED/UNCLEAR label and human judgment on (claim, document) pairs, plus inter-model agreement across judge models. |

## Shared conventions

- Run everything with `mamba run -n attribution-eval python ...` (or
  `micromamba run -n attribution-eval ...`) — see the environment note in
  each subfolder.
- Pairwise/model comparisons pull from `predicted_claims.jsonl` (produced by
  `atomic_claims/run_extractor.py`), so `atomic_claims/` is upstream of
  `judges/` — sample the atomic claims first if starting from scratch.
- Multi-model runs write one file per model (e.g.
  `results_<model>_summary.json`, `predictions_<model>.jsonl`) so a single
  `compare_models.py` per folder can build a cross-model table without
  rerunning anything.
- Files named `old_prompt/` or `old_preds/` are prior-version snapshots kept
  for before/after comparison plots (e.g. `judges/plot_agreement.py`), not
  stale cruft — leave them in place.
