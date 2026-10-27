# Claim filtering

**Pipeline stage 1b (optional).** This stage drops claims that don't need a
source, such as opinions, filler sentences and common knowledge. It runs after
extraction and before alignment, so dropped claims are left out of alignment,
judgement and all metrics. Filters run in the order they are configured.

## Base class

[`ClaimFilter`](base.py):

```python
class ClaimFilter(ABC):
    name: str = "filter"

    @abstractmethod
    def decide(
        self, claims: list[Claim], answer: str, question: str | None = None
    ) -> list[FilterDecision]: ...

    def filter(self, claims, answer, question=None) -> list[Claim]:  # kept claims only
```

`decide()` returns one [`FilterDecision`](../data_models.py) per input claim,
in the same order. A decision has the fields `claim`, `keep`, `filter_name`,
`category` (optional) and `reason` (optional). Filters must return the claim
objects unchanged, because span-based aligners depend on `Claim.span`.

## Implementations

| Class | Config key | LLM call |
|---|---|---|
| [`LLMCheckWorthinessFilter`](check_worthiness.py) | `check_worthiness` (**on by default**) | 1 batched call per answer |

### `check_worthiness`

An LLM assigns each claim exactly one [`ClaimCategory`](check_worthiness.py):

- **Dropped by default:** `NOT_A_CLAIM`, `SUBJECTIVE`, `UNDERSPECIFIED`, `COMMON_KNOWLEDGE`
- **Kept:** `QUANTITY`, `EVENT_OR_HISTORICAL`, `ATTRIBUTION_OR_QUOTE`,
  `CAUSAL_OR_CORRELATION`, `SCIENTIFIC_OR_TECHNICAL`, `RULE_OR_POLICY`,
  `PREDICTION`, `OTHER_FACTUAL`

The categories are based on ClaimBuster, the Full Fact claim schema,
Wikipedia's *Citation Needed* taxonomy and VeriScore. If the LLM call fails,
the filter keeps every claim.

## Configuration

```yaml
claim_filters:
  - check_worthiness      # [] disables filtering

# api_base/api_key/temperature/max_retries/thinking set to null fall back to
# the `judge:` section; model: null reuses judge.model.
check_worthiness_filter:
  model: null
  max_tokens: 2048
  system_prompt: null
  drop_categories: [NOT_A_CLAIM, SUBJECTIVE, UNDERSPECIFIED, COMMON_KNOWLEDGE]
```

To turn filtering off for a single call, pass `evaluate(..., claim_filters=[])`.

## Inspecting decisions

Every decision is stored in `result.filter_decisions`, with its category and
reason, whether the claim was kept or dropped. The decisions are included in
`to_dict()` / `to_jsonl()`, and `pretty_print()` lists the dropped claims. For
`evaluate_batch()`, `result.filter_summary` (also written to the JSONL
aggregate line) collects claim counts, the drop rate, and per-category
kept/dropped counts:

```python
result.filter_summary
# {"claims_extracted": 595, "claims_kept": 545, "claims_dropped": 50, "drop_rate": 0.084,
#  "items_with_drops": 24, "items_all_dropped": 1,
#  "by_filter": {"check_worthiness": {"seen": 595, "dropped": 50,
#                "categories": {"SUBJECTIVE": {"kept": 0, "dropped": 33}, ...}}}}
```

## Custom filter

```python
from acclaim import evaluate
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

Passing `claim_filters=` replaces the configured filters. To keep the
check-worthiness filter as well, include an `LLMCheckWorthinessFilter(...)`
instance in the list.
