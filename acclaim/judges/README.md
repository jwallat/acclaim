# Judges

**Pipeline stage 3.** The *evidence judge* checks each claim against the
documents it cites and labels it `SUPPORTED`, `REFUTED` or `UNCLEAR`. This
directory also contains the judges used by the opt-in LLM-based metrics
(citation precision/recall) and by relevance stratification.

## Base class

[`EvidenceJudge`](base.py):

```python
class EvidenceJudge(ABC):
    @abstractmethod
    def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult: ...
```

A [`SupportResult`](../data_models.py) has a `label` (a `SupportLabel`), a
`confidence` in [0, 1] and a one-sentence `reason`. A claim with no valid
cited document is labelled `UNCLEAR` with confidence 0.0, and no model call
is made.

Labels:

- **SUPPORTED**: the documents state the claim explicitly or strongly imply it.
- **REFUTED**: the documents directly contradict the claim.
- **UNCLEAR**: the documents don't address the claim, are about something
  else, or are ambiguous. "Not mentioned" counts as `UNCLEAR`, not `REFUTED`.

## Evidence judges

| Class | `judge.backend` | Model type |
|---|---|---|
| [`LiteLLMJudge`](litellm.py) | `chat` (**default**) | Any chat LLM supported by LiteLLM, prompted for a JSON verdict |
| [`DecisionJudge`](decision.py) | `decision` | A decision model (OpenRouter Decisions API) that returns probabilities instead of text |

### `LiteLLMJudge`

The default judge. It uses `judge.system_prompt` from the config, which has
few-shot examples. It works with any provider LiteLLM supports: OpenAI,
Anthropic, OpenRouter, a local vLLM server, and others.

```yaml
judge:
  model: openrouter/google/gemma-4-26b-a4b-it
  api_base: null        # e.g. http://localhost:8000/v1 for vLLM
  api_key: null         # null = provider env var (OPENROUTER_API_KEY, ...)
  temperature: 0.0
  max_tokens: 2048
  max_retries: 3
  thinking: false       # true sends chat_template_kwargs.enable_thinking (vLLM)
```

### `DecisionJudge`

Asks a decision model and turns its probability distribution into a label.
There are three question modes:

- `choice`: one question with three options: SUPPORTED, REFUTED, UNCLEAR.
- `noul`: two independent yes/no questions ("supported?" and
  "contradicted?"). Their answers are combined using `decision_thresholds`.
- `score`: a question on an ordered scale, REFUTED < UNCLEAR < SUPPORTED.

```yaml
judge:
  backend: decision
  model: openrouter/typesafe/jev-1.13
  api_key: ${OPENROUTER_API_KEY}
  decision_mode: choice               # choice | noul | score
  decision_state_format: structured   # structured | text
```

A decision model can't produce text, so the components that need a chat LLM
(`atomic_claim_extractor`, `check_worthiness_filter` and `llm_aligner`) can no
longer fall back to `judge.model`. Set a `model` in each of their config
sections.

Agreement of both judge types with human labels is measured in
[`library_evals/judges/`](../../library_evals/judges/README.md).

## Auxiliary judges

These judges are used only when the corresponding feature is turned on. Each
one reuses `judge:` unless it has its own config section.

| Class | Used by | Labels | Config section |
|---|---|---|---|
| [`LiteLLMPrecisionJudge`](citation_precision.py) | `citation_precision`, `citation_f1` metrics | `Relevant` / `Unrelevant` per claim–citation pair | `citation_precision_judge` |
| [`LiteLLMRecallJudge`](citation_recall.py) | `citation_recall`, `citation_f1` metrics | `FULL` / `PARTIAL` / `NONE` per claim | `citation_recall_judge` |
| [`LiteLLMRelevanceJudge`](relevance.py) | `relevance_stratified_metrics` | `CORE` / `COMPLEMENTARY` / `IRRELEVANT` per claim relative to the question | `relevance_judge` |

These three judges require a chat backend.

## Custom judge

```python
from acclaim import evaluate
from acclaim.data_models import SupportLabel, SupportResult
from acclaim.judges import EvidenceJudge

class SubstringJudge(EvidenceJudge):
    def evaluate(self, claim, docs):
        hit = any(claim.text.rstrip(".").lower() in d.text.lower() for d in docs)
        label = SupportLabel.SUPPORTED if hit else SupportLabel.UNCLEAR
        return SupportResult(label=label, confidence=1.0, reason="Exact substring match.")

result = evaluate(answer, documents, judge=SubstringJudge())
```
