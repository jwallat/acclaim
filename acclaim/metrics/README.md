# Metrics

**Pipeline stage 4.** This stage turns the per-claim results into scores.
Metrics are chosen by name in the `metrics:` config list, and the results end
up in `result.metrics` (and in `aggregate_metrics` for `evaluate_batch()`,
which averages them over the items). Claims removed by a
[filter](../filters/README.md) are left out of every metric.

## Base class

[`Metric`](base.py):

```python
class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute(
        self,
        claim_results: list[ClaimResult],
        valid_doc_ids: list[str] | None = None,
        documents: list[Document] | None = None,
        answer_text: str | None = None,
        question: str | None = None,
    ) -> float: ...
```

## Available metrics

The 11 metrics in [`METRIC_REGISTRY`](registry.py):

| Metric key | LLM calls | Description |
|---|---|---|
| `citation_correctness` | — | Fraction of *cited* claims labelled SUPPORTED |
| `supported_claim_rate` | — | Fraction of *all* claims (cited or not) labelled SUPPORTED |
| `coverage` | — | Fraction of claims that cite at least one document |
| `hallucination_rate` | — | Fraction of citation references that point to a document that doesn't exist (`invalid:<N>`) |
| `citation_number` | — | Total number of `[N]` markers in the raw answer |
| `citation_lengths` | — | Average token count of the distinct cited claim texts |
| `cvcp` | — | Coefficient of Variation of Citation Positions: whether the citations in a sentence cluster at its end or are spread through it |
| `token_overlap` | — | Average Jaccard token overlap between each cited claim and the closest matching sentence in each document it cites |
| `citation_precision` | 1 per claim–citation pair | LongCite-style: how many cited documents are relevant to the claim they are attached to |
| `citation_recall` | 1 per claim | How well a claim's cited documents support it: fully (1), partially (0.5) or not at all (0) |
| `citation_f1` | precision + recall | Harmonic mean of `citation_precision` and `citation_recall` |

`citation_correctness` and `supported_claim_rate` are based on the evidence
judge's labels, which were already computed in stage 3. Only the last three
metrics make their own LLM calls. They are left out of the default list
because they are by far the most expensive. To enable them, add them to the
list:

```yaml
metrics:
  - citation_correctness
  - coverage
  - hallucination_rate
  - citation_precision     # opt-in
  - citation_recall        # opt-in
  - citation_f1            # opt-in
```

Each one uses `citation_precision_judge:` / `citation_recall_judge:` if that
section is set, and otherwise falls back to `judge:`. See the
[judges README](../judges/README.md#auxiliary-judges).

## Relevance-stratified metrics

If you call `evaluate()` with a `question`, you can also break metrics down
by how relevant each claim is to the question. A
[`LiteLLMRelevanceJudge`](../judges/relevance.py) labels each claim `CORE`,
`COMPLEMENTARY` or `IRRELEVANT`, and the base metric is then computed
separately for each group:

```yaml
relevance_stratified_metrics:
  - citation_correctness   # must also be listed under `metrics:`
```

```python
result = evaluate(answer, documents, question="...", config=config)
result.metrics
# {"citation_correctness": 0.8, "citation_correctness_core": 1.0,
#  "citation_correctness_complementary": 0.5, ...}
```

Groups with no claims don't appear in the output. Without a `question`,
stratification is skipped.

## Custom metric

Subclass `Metric`, register an instance in `METRIC_REGISTRY` and add its name
to `metrics:`:

```python
import dataclasses
from acclaim import evaluate, load_config
from acclaim.data_models import SupportLabel
from acclaim.metrics import METRIC_REGISTRY, Metric

class RefutedRate(Metric):
    name = "refuted_rate"

    def compute(self, claim_results, valid_doc_ids=None, documents=None,
                answer_text=None, question=None):
        if not claim_results:
            return 0.0
        refuted = sum(cr.support.label == SupportLabel.REFUTED for cr in claim_results)
        return refuted / len(claim_results)

METRIC_REGISTRY["refuted_rate"] = RefutedRate()

config = load_config()
config = dataclasses.replace(config, metrics=[*config.metrics, "refuted_rate"])
result = evaluate(answer, documents, config=config)
```
