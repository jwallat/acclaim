"""FastAPI server for the acclaim explorer.

Start with:
    uvicorn acclaim_explorer.server.main:app --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from acclaim.evaluate import evaluate

from .config import AcclaimExplorerConfig, load_acclaim_explorer_config
from .generation import RagGenerator
from .retrieval import PyseriniRetriever, RetrieverBase
from .serialization import augment_record

logger = logging.getLogger(__name__)

# acclaim_explorer/server/main.py → go up two levels to reach the repo root
ROOT = Path(__file__).parent.parent.parent
RUNS_DIR = ROOT / "outputs" / "alce_eval"


def _build_retriever(cfg: AcclaimExplorerConfig, name: str) -> RetrieverBase:
    if name == "bm25":
        if not cfg.retriever.bm25_index_path and not cfg.retriever.bm25_prebuilt_index:
            raise RuntimeError(
                "Configure bm25.index_path (local) or bm25.prebuilt_index in acclaim_explorer_config.yaml"
            )
        return PyseriniRetriever(
            index_path=cfg.retriever.bm25_index_path,
            prebuilt_index=cfg.retriever.bm25_prebuilt_index,
            top_k=cfg.retriever.top_k,
        )
    raise ValueError(f"Unknown retriever: {name!r}. Choose 'bm25'.")


cfg = load_acclaim_explorer_config()
_generator = RagGenerator(cfg.generator)

# Retrievers are built lazily so a missing API key doesn't crash startup.
_retriever_cache: dict[str, RetrieverBase] = {}


def _get_retriever(name: str) -> RetrieverBase:
    if name not in _retriever_cache:
        _retriever_cache[name] = _build_retriever(cfg, name)
    return _retriever_cache[name]


app = FastAPI(title="acclaim explorer")

app.mount(
    "/acclaim_explorer",
    StaticFiles(directory=str(ROOT / "acclaim_explorer"), html=True),
    name="acclaim_explorer",
)
app.mount(
    "/outputs",
    StaticFiles(directory=str(ROOT / "outputs"), html=True),
    name="outputs",
)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/acclaim_explorer/index.html")


@app.get("/api/runs")
def list_runs() -> list[str]:
    """Return sorted list of .jsonl filenames available in outputs/alce_eval/."""
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.glob("*.jsonl"))


def _load_run(path: Path) -> dict:
    """Parse a BatchEvaluationResult.to_jsonl() file into a frontend-ready shape."""
    items: list[dict] = []
    aggregate_metrics: dict = {}
    metadata: dict | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            record_type = rec.pop("record_type", "item")
            if record_type == "aggregate":
                aggregate_metrics = rec.get("aggregate_metrics", {})
                metadata = rec.get("metadata")
                continue
            index = rec.pop("index", len(items))
            item = augment_record(rec)
            item["index"] = index
            items.append(item)
    return {"items": items, "aggregate_metrics": aggregate_metrics, "metadata": metadata}


@app.get("/api/runs/{name}")
def get_run(name: str) -> dict:
    """Return a single batch run's items + aggregate metrics as one JSON payload."""
    safe_name = Path(name).name  # strip any directory components
    path = (RUNS_DIR / safe_name).resolve()
    if (
        not safe_name.endswith(".jsonl")
        or not path.is_relative_to(RUNS_DIR.resolve())
        or not path.is_file()
    ):
        raise HTTPException(status_code=404, detail=f"Run not found: {name!r}")
    return _load_run(path)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _pipeline(question: str, retriever_name: str):
    loop = asyncio.get_event_loop()

    try:
        retriever = _get_retriever(retriever_name)
    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
        return

    yield _sse("retrieving", {})
    try:
        docs = await loop.run_in_executor(None, retriever.retrieve, question)
    except Exception as exc:
        logger.exception("Retrieval failed")
        yield _sse("error", {"message": f"Retrieval failed: {exc}"})
        return

    # Send docs immediately so the UI can render them before generation starts.
    yield _sse("retrieved", {"documents": [{"doc_id": d.doc_id, "text": d.text} for d in docs]})

    yield _sse("generating", {})
    try:
        answer = await loop.run_in_executor(None, _generator.generate, question, docs)
    except Exception as exc:
        logger.exception("Generation failed")
        yield _sse("error", {"message": f"Generation failed: {exc}"})
        return

    # Send answer immediately so the UI can show it before evaluation.
    yield _sse("generated", {"answer": answer})

    yield _sse("evaluating", {})
    try:
        citation_to_doc = {i: doc.doc_id for i, doc in enumerate(docs, 1)}
        result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                answer,
                docs,
                question=question,
                citation_to_doc=citation_to_doc,
                config=cfg.eval,
            ),
        )
    except Exception as exc:
        logger.exception("Evaluation failed")
        yield _sse("error", {"message": f"Evaluation failed: {exc}"})
        return

    yield _sse("done", {"result": augment_record(result.to_dict())})


@app.get("/query")
async def query_endpoint(
    q: str = Query(..., min_length=1, description="The question to evaluate"),
    retriever: str = Query(default="bm25", description="Retriever backend"),
) -> StreamingResponse:
    return StreamingResponse(
        _pipeline(q, retriever),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
