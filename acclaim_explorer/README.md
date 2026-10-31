# Demo Tool for the Library

A FastAPI-backed webapp that visualizes the outputs of an `acclaim` batch run,
plus a live "Ask question" mode that retrieves documents, generates an
answer, and evaluates it on the fly.

Loads batch results from `outputs/alce_eval/<dataset>_<model>.jsonl` (e.g.
`asqa_qwen3-8b.jsonl`) and renders each evaluated answer as a chat-style page
with metrics, claims, and cited documents side-by-side.

## Run

From the **repo root**:

```bash
uvicorn acclaim_explorer.server.main:app --port 8000 --reload
```

Then open <http://localhost:8000/>.

The server exposes two JSON endpoints the frontend uses to load batch runs:

- `GET /api/runs` — sorted list of `.jsonl` filenames in `outputs/alce_eval/`.
- `GET /api/runs/{name}` — parses that file's `record_type: "item"` lines and
  its trailing `record_type: "aggregate"` line, computes display-only fields
  (sentence boundaries, citation marker spans, a citation-number → doc-id
  map), and returns `{items, aggregate_metrics, metadata}` in one response.

## Controls

- **Run dropdown** — lists every `.jsonl` file in `outputs/alce_eval/`.
- **Example dropdown / Prev / Next** — switch between records in the current run. Arrow keys also work.
- **Highlight toggle**
  - `Off` — plain text
  - `Support` — color each answer sentence by the aggregated support label of
    the claims whose `claim.span` falls inside that sentence
    (green=supported, red=refuted, amber=unclear, striped=mixed)
  - `Citations` — highlight only the `[N]` citation markers in the answer
- **Hover linkage**
  - Hovering an answer sentence outlines every claim that maps to it.
  - Hovering a claim outlines the sentence it maps to.
  - Mapping uses `claims[*].claim.span`. Claims with `span == [-1, -1]`
    (typically uncited atomic claims, which the aligner has no citation number
    to place by) are placed by the server via weighted word overlap with the
    answer's sentences (display only — `claim.span` and metrics are
    unchanged). Those claims show an "≈ sentence" hint.
- **"+N filtered" toggle** (claims panel header; only shown when
  the example has filtered claims) — off: only the claims that survived the
  claim filter (e.g. check-worthiness), i.e. exactly what the metrics are
  computed over. On: also lists the dropped claims (dashed, grey
  `FILTERED · <category>` pill, filter reason) and links them to their most
  likely sentence; a sentence whose claims were *all* filtered gets a dashed
  underline. Filtered claims are never judged and count toward no metric.
- **Theme button** — toggles light/dark. Defaults to your OS preference; saved in `localStorage`.
- Clicking a `[N]` marker scrolls to the matching cited document and flashes its border.
- **Ask question** — switches the answer panel into a live query: retrieves
  documents (via `acclaim_explorer/server/retrieval.py`), generates an
  answer, evaluates it, and streams progress via Server-Sent Events. The
  retriever is set by `retriever.default` in `acclaim_explorer_config.yaml`:
  `keenable` (live web search, default; works without a key, optional
  `KEENABLE_API_KEY` for higher limits) or `bm25` (MSMARCO via pyserini; see
  that file for local-index vs. prebuilt-index options). A single query can
  override it with `/query?q=...&retriever=bm25`. Also needs an LLM API key
  for the generator/judge models, resolved from the repo-root `.env` file.

## Known limitations

- **No within-document highlighting** beyond token-overlap keyword highlighting
  on pin — the cited document text is otherwise shown in full.

## Files

- `index.html` — page skeleton
- `styles.css` — layout and highlight colors
- `app.js` — batch-run loading, rendering, navigation
- `query.js` — live "Ask question" mode (SSE streaming into the same panels)
- `server/main.py` — FastAPI app, `/api/runs` endpoints, `/query` SSE pipeline
- `server/serialization.py` — `augment_record()`, shared by both the batch and live-query paths
- `server/config.py` — loads `eval_config.yaml` + `acclaim_explorer_config.yaml`
- `server/retrieval.py` — retrieval backends (Keenable web search, BM25)
- `server/generation.py` — RAG answer generation for the live-query pipeline
- `server/build_index.py` — CLI to pre-build a local BM25 index
- `PLAN.md` — design notes for the original v1 prototype (predates the FastAPI server and current output format; kept for historical context)
