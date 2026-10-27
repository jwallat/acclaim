<div align="center">

# <img src="icon.png" alt="" height="72" align="center"> acclaim

**Evaluate citation attribution quality of LLM-generated answers.**

<p>
  <a href="https://github.com/jwallat/acclaim/actions/workflows/tests.yml"><img alt="build" src="https://github.com/jwallat/acclaim/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/jwallat/acclaim/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/jwallat/acclaim"></a>
  <a href="https://pypi.org/project/acclaim/"><img alt="release" src="https://img.shields.io/pypi/v/acclaim?label=release"></a>
  <a href="https://pypi.org/project/acclaim/"><img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <!-- Keep in sync with len(METRIC_REGISTRY) in acclaim/metrics/registry.py -->
  <a href="acclaim/metrics/README.md"><img alt="metrics" src="https://img.shields.io/badge/metrics-11-blueviolet"></a>
  <!-- DOI badge (add once a Zenodo DOI exists):
  <a href="https://doi.org/10.5281/zenodo.XXXXXXX"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg"></a>
  -->
</p>

</div>

`acclaim` takes an answer with inline citation markers (`[1]`, `[2]`, …)
and splits it into claims. It drops claims that don't need a source, works
out which documents each remaining claim cites, and checks whether those
documents actually support the claim. From that it computes attribution
metrics, which show how much of the answer is backed by its sources, where
citations are missing, wrong or point to documents that don't exist, and
which claims the sources contradict.

---

## Pipeline

```text
answer text + documents
    │
    ▼
1.  Claim extraction    ──► list[Claim]           atomic (LLM) | sentence
    │
    ▼
1b. Claim filtering     ──► list[Claim]           drops opinions, filler, common knowledge
    │
    ▼
2.  Citation alignment  ──► list[AlignedClaim]    claim → cited doc IDs
    │
    ▼
3.  Evidence judgement  ──► list[ClaimResult]     SUPPORTED / REFUTED / UNCLEAR
    │
    ▼
4.  Metrics             ──► {"citation_correctness": …, "coverage": …, "hallucination_rate": …}
```

Each stage is a pluggable module with a base class, several implementations
and its own README:

| Stage | Module | Implementations |
|---|---|---|
| 1. Claim extraction | [`acclaim/claims/`](acclaim/claims/README.md) | `AtomicClaimExtractor` (default), `SentenceClaimExtractor` |
| 1b. Claim filtering | [`acclaim/filters/`](acclaim/filters/README.md) | `LLMCheckWorthinessFilter` (default on) |
| 2. Citation alignment | [`acclaim/alignment/`](acclaim/alignment/README.md) | `LLMCitationAligner`, `SentenceCitationAligner`, `JaccardCitationAligner` |
| 3. Evidence judgement | [`acclaim/judges/`](acclaim/judges/README.md) | `LiteLLMJudge` (default), `DecisionJudge` |
| 4. Metrics | [`acclaim/metrics/`](acclaim/metrics/README.md) | 11 metrics, plus relevance-stratified variants |

**New here?** The [end-to-end walkthrough](examples/README.md) follows one
answer through every stage and shows the real output of each.

---

## Installation

```bash
pip install acclaim
```

**From source (development)**, with Python 3.10+:

```bash
mamba create -n acclaim python=3.10
mamba activate acclaim
pip install -e ".[dev]"
```

To also run the experiments in [`library_evals/`](library_evals/README.md),
install the extra evaluation and notebook dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API keys

All LLM calls go through [LiteLLM](https://github.com/BerriAI/litellm), which
reads the standard environment variables of each provider. Export the key for
the provider you use:

```bash
export OPENROUTER_API_KEY=...   # used by the default config
# or OPENAI_API_KEY, ANTHROPIC_API_KEY, ... depending on the configured models
```

Config YAML files can also reference a variable directly with `${VAR_NAME}`
(e.g. `api_key: ${ANTHROPIC_API_KEY}`). The variable is read from the
environment when the config is loaded.

If you'd rather keep keys in a file, copy `.env.example` to `.env` and install
with `pip install -e ".[dev]"`, which pulls in `python-dotenv`. The `.env` file
is then loaded automatically whenever `acclaim` is imported. Without
`python-dotenv` this does nothing, and `.env` is gitignored.

---

## Quick start

This is a minimal example on how to use the library. After installation (see above) please follow these steps:

The default config uses Gemma 4 through OpenRouter, so set
`OPENROUTER_API_KEY` first.

```python
from acclaim import evaluate, load_config, Document

documents = [
    Document(doc_id="doc1", text="The Eiffel Tower was completed in 1889 for the World's Fair."),
    Document(doc_id="doc2", text="The Eiffel Tower is 330 metres tall."),
]

answer = "The Eiffel Tower was completed in 1889 [1]. It is 300 metres tall [2]."

config = load_config()           # bundled acclaim/config/default.yaml
results = evaluate(answer, documents, config=config)

results.pretty_print()
print(results.metrics)
# {'citation_correctness': 0.5, 'citation_lengths': 10.0, 'citation_number': 2.0,
#  'coverage': 1.0, 'cvcp': 0.0, 'hallucination_rate': 0.0,
#  'supported_claim_rate': 0.5, 'token_overlap': 0.725}
```

The second claim is `REFUTED` (the document says 330 m), so only half of the
cited claims are supported.

Every intermediate result is available in `results.steps`:

```python
results.steps["claims"]            # list[Claim]           all extracted claims
results.steps["filter_decisions"]  # list[FilterDecision]  keep/drop + category + reason
results.steps["filtered_claims"]   # list[Claim]           claims that passed the filters
results.steps["aligned_claims"]    # list[AlignedClaim]
results.steps["claim_results"]     # list[ClaimResult]
results.steps["citation_to_doc"]   # dict[int, str]
```

Save the results (claims, filter decisions, metrics, and the package version
and config) to JSONL:

```python
results.to_jsonl("results.jsonl")                      # add include_steps=True for a full dump
```

### Batch evaluation

```python
from acclaim import BatchExample, evaluate_batch

examples = [
    BatchExample(answer=answer, documents=documents, question="How tall is the Eiffel Tower?"),
    # ...
]

batch_result = evaluate_batch(examples, config=config)
batch_result.aggregate_metrics   # mean of each metric over the examples
batch_result.filter_summary      # claim counts, drop rate, per-category counts
batch_result.to_jsonl("batch_results.jsonl")   # one line per item + a trailing aggregate line
```

The batch JSONL files can be browsed in the
[`acclaim_explorer`](acclaim_explorer/README.md) web app.

### Optional inputs

- **`question=`**: the question the answer responds to. Claim filtering uses
  it, and it is required for
  [relevance-stratified metrics](acclaim/metrics/README.md#relevance-stratified-metrics).
- **`citation_to_doc=`**: by default `[N]` maps to `documents[N-1]`. If your
  document IDs don't follow that order, pass an explicit mapping:

  ```python
  evaluate(answer, documents, citation_to_doc={1: "doc_paris", 2: "doc_eiffel"})
  ```

- **Component overrides**: `claim_extractor=`, `claim_filters=`, `aligner=` and
  `judge=` take instances (including your own subclasses) and replace the
  component built from the config. See the module READMEs for examples.

---

## Configuration

Configuration is in YAML. Calling `load_config()` without arguments reads the
bundled [`acclaim/config/default.yaml`](acclaim/config/default.yaml), which
documents every option. To load your own file, use
`load_config("my_config.yaml")`. The main options:

```yaml
claim_extractor: atomic          # atomic | sentence                 → acclaim/claims/
claim_filters: [check_worthiness]  # [] disables filtering          → acclaim/filters/
aligner: null                    # null = auto (atomic→llm, sentence→sentence) → acclaim/alignment/

judge:                           # evidence judge                    → acclaim/judges/
  model: openrouter/google/gemma-4-26b-a4b-it
  api_base: null                 # e.g. http://localhost:8000/v1 for local vLLM
  api_key: null                  # null = provider env var
  temperature: 0.0
  max_tokens: 2048

concurrency:
  max_workers: 10                # parallel LLM calls (per claim and per batch item)

metrics:                         # → acclaim/metrics/
  - citation_correctness
  - citation_lengths
  - citation_number
  - coverage
  - cvcp
  - hallucination_rate
  - supported_claim_rate
  - token_overlap
```

You only need to configure one LLM endpoint. The LLM-based components
(`atomic_claim_extractor`, `check_worthiness_filter`, `llm_aligner`, and the
auxiliary judges) use the `judge:` connection settings unless their own
config section overrides them.

### LLM provider

Any model that [LiteLLM](https://docs.litellm.ai/docs/providers) supports can
be used:

```yaml
# Anthropic
judge:
  model: anthropic/claude-sonnet-5-5
  api_key: ${ANTHROPIC_API_KEY}

# Local vLLM server
judge:
  model: openai/meta-llama/Meta-Llama-3-8B-Instruct
  api_base: http://localhost:8000/v1
  api_key: dummy
```

The evidence judge can also run on a **decision model** (`backend: decision`)
instead of a chat LLM. See the [judges README](acclaim/judges/README.md#decisionjudge).

---

## Metrics

| Metric key | Description |
|---|---|
| `citation_correctness` | Fraction of *cited* claims labelled SUPPORTED |
| `supported_claim_rate` | Fraction of *all* claims labelled SUPPORTED |
| `coverage` | Fraction of claims with at least one citation |
| `hallucination_rate` | Fraction of citation references that point to documents that don't exist |
| `citation_number` | Number of `[N]` markers in the answer |
| `citation_lengths` | Average token length of cited claims |
| `cvcp` | Coefficient of Variation of Citation Positions within sentences |
| `token_overlap` | Jaccard overlap between each cited claim and its best-matching document sentence |
| `citation_precision`* | LLM-judged relevance of each citation to its claim (LongCite-style) |
| `citation_recall`* | LLM-judged full / partial / no support of each claim by its citations |
| `citation_f1`* | Harmonic mean of citation precision and recall |

\* Opt-in, because they make extra LLM calls. The
[metrics README](acclaim/metrics/README.md) has exact definitions, explains
relevance stratification and shows how to add a custom metric.

---

## Running tests

Unit tests mock all LLM calls and need no external service:

```bash
pytest acclaim/tests/unit/
```

End-to-end tests run the full pipeline against an OpenAI-compatible vLLM
server. They are skipped automatically when the server can't be reached:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000 &

export VLLM_API_BASE=http://localhost:8000/v1                 # default
export VLLM_MODEL=openai/meta-llama/Meta-Llama-3-8B-Instruct  # LiteLLM model name
pytest acclaim/tests/e2e/ -m vllm -s -v
```

---

## Related tools

- [`examples/`](examples/README.md): annotated end-to-end walkthrough and a
  runnable script.
- [`acclaim_explorer/`](acclaim_explorer/README.md): a web app for browsing
  batch results claim by claim. It also has a live mode that retrieves
  documents, generates an answer and evaluates it. [`DOCKER.md`](DOCKER.md)
  covers running it in Docker.
- [`library_evals/`](library_evals/README.md): experiments that check each
  pipeline component against human judgement (atomic claims, citation
  alignment, judge agreement).

---

<!-- ## Project structure

```text
acclaim/
├── evaluate.py            evaluate(): pipeline entry point + component factories
├── batch.py               evaluate_batch(): batch evaluation and aggregation
├── data_models.py         Immutable dataclasses (Claim, Document, EvaluationResult, ...)
├── llm_client.py          LiteLLM wrapper with retries / JSON parsing
├── decision_client.py     Client for decision models (OpenRouter Decisions API)
├── concurrency.py         Thread-pool helpers for parallel LLM calls
├── text_utils.py          Sentence splitting / tokenization
├── citations/parser.py    [N] marker parsing, citation → doc ID mapping
├── claims/                Stage 1: claim extraction        (README)
├── filters/               Stage 1b: claim filtering        (README)
├── alignment/             Stage 2: citation alignment      (README)
├── judges/                Stage 3: evidence + auxiliary judges (README)
├── metrics/               Stage 4: metrics + registry      (README)
├── config/
│   ├── default.yaml       Default configuration (documents every option)
│   └── load.py            EvalConfig / JudgeConfig + load_config()
└── tests/                 unit/ (mocked) and e2e/ (vLLM)
examples/                  End-to-end walkthrough
acclaim_explorer/          Results browser web app
library_evals/             Component evaluation experiments
```

--- -->

## License

[MIT](LICENSE)
