# Claim extraction

**Pipeline stage 1.** This stage splits an answer into the claims that are
later aligned to citations and judged. The rest of the pipeline works on these
claims, so the choice of extractor affects every metric.

## Base class

[`ClaimExtractor`](base.py):

```python
class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, answer: Answer) -> list[Claim]: ...
```

A [`Claim`](../data_models.py) has `text` (with the citation markers removed)
and `span`, the `(start, end)` character offsets in the original answer. The
span is `(-1, -1)` when the claim has no exact location in the answer.

## Implementations

| Class | Config key | LLM call | Spans | Default aligner |
|---|---|---|---|---|
| [`AtomicClaimExtractor`](atomic.py) | `atomic` (**default**) | 1 per answer | `(-1, -1)` | `LLMCitationAligner` |
| [`SentenceClaimExtractor`](sentence.py) | `sentence` | none | exact | `SentenceCitationAligner` |

- **`atomic`** asks an LLM to break the answer into short claims that each
  state one fact and stand on their own. "The tower, completed in 1889, is in
  Paris [1]." becomes two claims. Because the LLM rewrites the text, these
  claims have no spans. The citation aligner (by default the
  [LLM aligner](../alignment/README.md)) then works out which citations belong
  to each claim. Use this for fine-grained evaluation where each fact counts
  on its own.
- **`sentence`** treats each sentence as one claim. It needs no LLM call, so
  it is fast and free, and the result is deterministic. The trade-off is
  granularity: if one fact in a sentence is wrong, the whole sentence fails.

## Configuration

```yaml
claim_extractor: atomic   # or: sentence

# Only used for `atomic`. api_base/api_key/temperature/max_retries/thinking
# set to null fall back to the `judge:` section.
atomic_claim_extractor:
  model: openrouter/google/gemma-4-26b-a4b-it
  max_tokens: 12196
  system_prompt: null     # null = built-in decomposition prompt
```

See [`config/default.yaml`](../config/default.yaml) for all fields. The
quality of the atomic extractor across models is measured in
[`library_evals/atomic_claims/`](../../library_evals/atomic_claims/README.md).

## Custom extractor

Subclass `ClaimExtractor` and pass an instance to `evaluate()`. If your claims
have no spans, also pass an aligner that doesn't need them, such as
`LLMCitationAligner` or `JaccardCitationAligner`. If you pass only the
extractor, `acclaim` picks the aligner for the `claim_extractor` set in the
config.

```python
from acclaim import evaluate
from acclaim.citations.parser import CITATION_MARKER_RE
from acclaim.claims import ClaimExtractor
from acclaim.data_models import Answer, Claim

class ParagraphClaimExtractor(ClaimExtractor):
    def extract(self, answer: Answer) -> list[Claim]:
        claims, pos = [], 0
        for para in answer.text.split("\n\n"):
            start = answer.text.index(para, pos)
            pos = start + len(para)
            text = CITATION_MARKER_RE.sub("", para).strip()  # span keeps the markers
            claims.append(Claim(text=text, span=(start, pos)))
        return claims

result = evaluate(answer, documents, claim_extractor=ParagraphClaimExtractor())
```
