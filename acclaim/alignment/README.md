# Citation alignment

**Pipeline stage 2.** This stage maps each claim to the documents it cites. A
citation marker `[N]` resolves to a document ID through `citation_to_doc`. By
default `[N]` maps to `documents[N-1]`; you can override this with
`evaluate(..., citation_to_doc={1: "doc_a", ...})`. A marker with no matching
document is kept as `invalid:<N>` rather than dropped, and the
`hallucination_rate` metric counts it.

## Base class

[`CitationAligner`](base.py):

```python
class CitationAligner(ABC):
    @abstractmethod
    def align(
        self, claims: list[Claim], text: str, citation_to_doc: dict[int, str]
    ) -> list[AlignedClaim]: ...
```

`text` is the original answer, including the `[N]` markers. The aligner
returns one [`AlignedClaim`](../data_models.py) (`claim`, `citation_doc_ids`)
per input claim.

## Implementations

| Class | Config key | LLM call | Works with |
|---|---|---|---|
| [`SentenceCitationAligner`](sentence.py) | `sentence` | none | Claims with spans (`sentence` extractor) |
| [`LLMCitationAligner`](llm.py) | `llm` | 1 batched call per answer | Claims without spans (`atomic` extractor) |
| [`JaccardCitationAligner`](jaccard.py) | `jaccard` | none | Claims without spans (legacy) |

- **`sentence`** reads the `[N]` markers inside each claim's span. Alignment
  is exact, with no guessing involved. Claims without a span get no
  citations.
- **`llm`** gives the LLM the numbered claims and the original answer, and
  asks which citation numbers support each claim. Afterwards it fills in
  `Claim.span` by matching each claim to a sentence that contains one of
  those markers. This is the default for atomic claims.
- **`jaccard`** matches each claim to the most similar sentence in the answer
  (token-set Jaccard overlap, optionally weighted by inverse claim frequency),
  and the claim takes over that sentence's markers. It is free and works
  offline, but it is less accurate when several sentences share wording.
  Its accuracy is measured in
  [`library_evals/citation_alignment/`](../../library_evals/citation_alignment/README.md).

## Configuration

`acclaim` picks the aligner from `claim_extractor` (`sentence` → `sentence`,
`atomic` → `llm`) unless you set `aligner:` yourself:

```yaml
aligner: null            # or: sentence | llm | jaccard

# Used by `llm`. Fields set to null fall back to `judge:`;
# max_tokens/system_prompt never do.
llm_aligner:
  model: null
  max_tokens: 2048
  system_prompt: null

# Used by `jaccard`.
jaccard_aligner:
  weighted: true
```

## Custom aligner

Subclass `CitationAligner` and pass it with `evaluate(..., aligner=MyAligner())`.
To turn a citation number into a doc ID (or `invalid:<N>`), use
`acclaim.citations.parser.citation_doc_id`. To read the markers in a span,
use `extract_doc_ids_for_span`.
