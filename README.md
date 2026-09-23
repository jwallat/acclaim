# acclaim

**Evaluate citation attribution quality of LLM-generated answers.**

`acclaim` extracts claims from an answer, aligns them to the inline citation markers (`[1]`, `[2]`, …), judges whether each claim is actually supported by the cited document, and computes attribution metrics. The result tells you how much of an answer is grounded in its sources and where hallucinations occur.

---

## Pipeline

```
answer text
    │
    ▼
Claim extraction  ──► list[Claim]
    │
    ▼
Claim filtering   ──► list[Claim]           (optional, e.g. drop non-check-worthy claims)
    │
    ▼
Citation alignment ──► list[AlignedClaim]   (claim → cited doc IDs)
    │
    ▼
Evidence judgement ──► list[ClaimResult]    (SUPPORTED / REFUTED / UNCLEAR)
    │
    ▼
Metric computation ──► {"coverage": …, "hallucination_rate": …, "citation_correctness": …, …}
```

---

## Installation


**Simple usage:**

```bash
pip install acclaim
```

**From source (development):**

```bash
pip install -e .
```

### Environment setup

Python 3.10+ is required. Using mamba or conda:

```bash
mamba create -n acclaim python=3.10
mamba activate acclaim
pip install -e .
```

### API keys

`acclaim` doesn't require any special setup for API keys — just export the
credential your provider expects before running, e.g.:

```bash
export OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY, OPENROUTER_API_KEY, etc., depending on judge.model
```

All LLM calls go through LiteLLM, which reads these standard provider env
vars automatically, so a plain `pip install acclaim` works with nothing
else to configure. Config YAML files can also reference a var explicitly
with `${VAR_NAME}` (see `judge.api_key: ${ANTHROPIC_API_KEY}` below),
resolved from the environment at load time.

If you'd rather keep keys in a file instead of exporting them each session,
copy `.env.example` to `.env` and install with `pip install -e ".[dev]"`
(which pulls in `python-dotenv`) — `.env` is then loaded automatically
whenever `acclaim` is imported. This is purely a convenience for local
development; it's a no-op if `python-dotenv` isn't installed, and `.env` is
gitignored and never committed.

---

## Quick start

The default config uses a Gemma 4 model from OpenRouter, so export the according API key:

```shell
export OPENROUTER_API_KEY=...
```

```python
from acclaim import evaluate, load_config, Document

documents = [
    Document(doc_id="doc1", text="Paris is the capital of France."),
    Document(doc_id="doc2", text="The Eiffel Tower was completed in 1889."),
]

answer = "Paris is the capital of France [1]. The Eiffel Tower was built in 1799 [2]."

config = load_config()           # uses config/default.yaml
result = evaluate(answer, documents, config=config)

result.pretty_print()
print(result.metrics)
# {"citation_correctness": 0.5, "coverage": 1.0, "hallucination_rate": 0.0, ...}
```

The pipeline stages are available for inspection via `result.steps`:

```python
result.steps["claims"]          # list[Claim]
result.steps["aligned_claims"]  # list[AlignedClaim]
result.steps["claim_results"]   # list[ClaimResult]
```

Save the result (claims, metrics, and run metadata) to JSONL:

```python
result.to_jsonl("results.jsonl")
```

`evaluate_batch()` results support the same method — one line per item plus a
trailing aggregate-metrics summary line:

```python
batch_result = evaluate_batch(examples, config=config)
batch_result.to_jsonl("batch_results.jsonl")
```

---

## Configuration

Configuration is YAML-based. `load_config()` with no arguments reads the bundled `config/default.yaml`:

```yaml
# Claim extraction strategy
# "sentence"  — one claim per sentence, no LLM call (fast)
# "atomic"    — LLM decomposes sentences into sub-claims (more precise, slower)
claim_extractor: sentence

# Evidence judge (always LiteLLM-based)
judge:
  model: gpt-4o-mini
  api_base: null          # e.g. "http://localhost:8000/v1" for a local vLLM server
  api_key: null           # falls back to OPENAI_API_KEY env var when null
  temperature: 0.0
  max_tokens: 512
  max_retries: 3

# Metrics to compute. citation_precision / citation_recall / citation_f1
# are opt-in because they run extra LLM judges; add them here to enable.
metrics:
  - citation_correctness
  - citation_lengths
  - citation_number
  - coverage
  - cvcp
  - hallucination_rate
  - supported_claim_rate
  - token_overlap
```

Load a custom config file:

```python
config = load_config("config/my_config.yaml")
```

### LLM provider

The judge uses [LiteLLM](https://github.com/BerriAI/litellm) and therefore supports 100+ model providers (OpenAI, Anthropic, Ollama, local vLLM, etc.) through a unified interface.

```yaml
# Anthropic
judge:
  model: claude-3-5-sonnet-20241022
  api_key: ${ANTHROPIC_API_KEY}

# Local vLLM server
judge:
  model: openai/meta-llama/Meta-Llama-3-8B-Instruct
  api_base: http://localhost:8000/v1
  api_key: dummy
```

### Citation mapping

By default, citation marker `[N]` maps to `documents[N-1]` (1-indexed position).  Override with an explicit mapping if your document IDs don't follow positional numbering:

```python
result = evaluate(
    answer,
    documents,
    citation_to_doc={1: "doc_paris", 2: "doc_eiffel"},
    config=config,
)
```

### Claim filtering

Optionally drop claims before alignment, so they don't count toward any metric. Filters run in the order listed:

```yaml
claim_filters:
  - check_worthiness

check_worthiness_filter:
  model: null              # null = reuse judge.model
  drop_categories: [NOT_A_CLAIM, SUBJECTIVE, UNDERSPECIFIED, COMMON_KNOWLEDGE]
```

`check_worthiness` asks an LLM to put each claim into one category:

- **Dropped by default:** `NOT_A_CLAIM`, `SUBJECTIVE`, `UNDERSPECIFIED`, `COMMON_KNOWLEDGE`
- **Kept:** `QUANTITY`, `EVENT_OR_HISTORICAL`, `ATTRIBUTION_OR_QUOTE`, `CAUSAL_OR_CORRELATION`, `SCIENTIFIC_OR_TECHNICAL`, `RULE_OR_POLICY`, `PREDICTION`, `OTHER_FACTUAL`

The categories draw on ClaimBuster, the Full Fact claim schema, Wikipedia's *Citation Needed* taxonomy, and VeriScore. If the LLM call fails, the filter keeps all claims.

Every decision (kept and dropped, with category and reason) is available as `result.filter_decisions` and is written to `to_dict()` / `to_jsonl()` output by default. `pretty_print()` lists the dropped claims. For `evaluate_batch()`, `result.filter_summary` (also written to the JSONL aggregate line) holds claim counts, the drop rate, and kept/dropped counts per category:

```python
result = evaluate_batch(examples, config=config)
result.filter_summary
# {"claims_extracted": 595, "claims_kept": 545, "claims_dropped": 50, "drop_rate": 0.084,
#  "items_with_drops": 24, "items_all_dropped": 1,
#  "by_filter": {"check_worthiness": {"seen": 595, "dropped": 50,
#                "categories": {"SUBJECTIVE": {"kept": 0, "dropped": 33}, ...}}}}
```

To write a custom filter, subclass `ClaimFilter` and pass it directly:

```python
from acclaim.filters import ClaimFilter, FilterDecision

class MinLengthFilter(ClaimFilter):
    name = "min_length"

    def decide(self, claims, answer, question=None):
        return [
            FilterDecision(claim=c, keep=len(c.text.split()) >= 4, filter_name=self.name)
            for c in claims
        ]

result = evaluate(answer, documents, claim_filters=[MinLengthFilter()])
```

---

## Running tests

### Unit tests

Unit tests mock all LLM calls and run without any external service:

```bash
pytest acclaim/tests/unit/
```

### End-to-end tests (local vLLM)

E2E tests run the full pipeline against a local vLLM server.

**Option 1 – CPU (for quick smoke-testing):**

```bash
bash start_vllm.sh
pytest acclaim/tests/e2e/ -m vllm -s -v
```

**Option 2 – GPU via SLURM (recommended):**

The `run_e2e_tests.sh` script starts a vLLM server inside the SLURM job, polls the `/health` endpoint until ready, runs pytest, then tears the server down automatically:

```bash
# Interactive
srun --gpus=1 --pty bash run_e2e_tests.sh

# Batch
sbatch --gpus=1 run_e2e_tests.sh
```

**Optional env vars for the E2E script:**

| Variable | Default | Description |
|---|---|---|
| `VLLM_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Model to load |
| `VLLM_PORT` | `8000` | Server port |
| `PYTEST_ARGS` | `-s -v` | Extra pytest arguments |

---

## Metrics

`citation_precision`, `citation_recall` and `citation_f1` are not in the default `metrics:` list because they make extra LLM calls on top of the evidence judge. Add them to your config to enable them.

| Metric key | Description |
|---|---|
| `citation_precision` | Mean per-citation score from an LLM judge assessing whether each cited snippet is relevant to the claim it supports (LongCite-style) |
| `citation_recall` | Mean per-claim score from an LLM judge assessing whether cited snippets fully, partially, or not at all support the claim |
| `citation_f1` | Harmonic mean of `citation_precision` and `citation_recall` |
| `citation_correctness` | Fraction of *cited* claims whose support label is SUPPORTED |
| `citation_lengths` | Average token count of unique claim texts that have at least one citation |
| `citation_number` | Total count of inline `[N]` citation markers found in the answer text |
| `coverage` | Fraction of all claims that have at least one citation |
| `cvcp` | Coefficient of Variation of Citation Positions — whether citations cluster at sentence ends vs. spread throughout |
| `hallucination_rate` | Fraction of citation references pointing to document IDs not present among the valid documents |
| `supported_claim_rate` | Fraction of *all* claims (cited or not) whose support label is SUPPORTED |
| `token_overlap` | Average Jaccard token overlap between each claim and its best-matching sentence in each cited document |

---

## Project structure

```
acclaim/
├── evaluate.py          Main entry point (evaluate(), evaluate_batch())
├── batch.py             Batch evaluation over multiple examples
├── concurrency.py       Thread-pool helpers for parallel LLM calls
├── text_utils.py        Sentence splitting / tokenization utilities
├── data_models.py       Immutable domain dataclasses
├── claims/              Claim extraction strategies
│   ├── sentence.py      SentenceClaimExtractor (default, no LLM)
│   └── atomic.py        AtomicClaimExtractor (LLM-based)
├── filters/             Optional claim filters (between extraction and alignment)
│   ├── base.py          ClaimFilter / FilterDecision
│   └── check_worthiness.py  LLMCheckWorthinessFilter + ClaimCategory taxonomy
├── alignment/           Citation alignment strategies
│   ├── sentence.py      SentenceCitationAligner (auto-selected for sentence claims)
│   └── jaccard.py       JaccardCitationAligner (auto-selected for atomic claims)
├── judges/
│   ├── litellm.py       LiteLLMJudge (evidence support judge)
│   ├── citation_precision.py  LiteLLMPrecisionJudge
│   ├── citation_recall.py     LiteLLMRecallJudge
│   └── relevance.py     LiteLLMRelevanceJudge (claim relevance stratification)
├── metrics/
│   ├── citation_precision.py  CitationPrecisionMetric
│   ├── citation_recall.py     CitationRecallMetric
│   ├── citation_f1.py         CitationF1Metric
│   ├── correctness.py         CitationCorrectness
│   ├── citation_lengths.py    CitationLengths
│   ├── citation_number.py     CitationNumber
│   ├── coverage.py            CitationCoverage
│   ├── cvcp.py                CVCPMetric
│   ├── hallucination.py       HallucinationRate
│   ├── supported_claim_rate.py SupportedClaimRate
│   ├── token_overlap.py       TokenOverlap
│   ├── stratified.py          StratifiedMetric (relevance-stratified wrapper)
│   └── registry.py            Metric registry
├── citations/
│   └── parser.py        Citation marker parsing utilities
└── config/
    ├── default.yaml     Default configuration
    └── load.py          EvalConfig / JudgeConfig dataclasses + load_config()
```
