"""Document retrieval backends for the acclaim explorer server."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from acclaim.data_models import Document

logger = logging.getLogger(__name__)


def _ensure_java_home() -> None:
    """Set JAVA_HOME if not already set; pyjnius requires it to find the JVM."""
    if os.environ.get("JAVA_HOME"):
        return
    try:
        java_bin = subprocess.check_output(
            ["readlink", "-f", subprocess.check_output(["which", "java"], text=True).strip()],
            text=True,
        ).strip()
        java_home = str(Path(java_bin).parent.parent)
        os.environ["JAVA_HOME"] = java_home
        logger.debug("Auto-set JAVA_HOME=%s", java_home)
    except Exception:
        raise RuntimeError(
            "Could not locate Java. Set JAVA_HOME or install Java 11+: https://www.java.com/en/download/"
        )


class RetrieverBase(ABC):
    @abstractmethod
    def retrieve(self, question: str) -> list[Document]:
        ...


class PyseriniRetriever(RetrieverBase):
    """BM25 retrieval via a Lucene index (local path or pyserini prebuilt).

    Local index:
        PyseriniRetriever(index_path="./data/bm25_index")

    Prebuilt MSMARCO (downloads to ~/.cache/pyserini/ on first use):
        PyseriniRetriever(prebuilt_index="msmarco-v1-passage")
    """

    def __init__(
        self,
        *,
        index_path: str | None = None,
        prebuilt_index: str | None = None,
        top_k: int = 7,
    ) -> None:
        _ensure_java_home()
        from pyserini.search.lucene import LuceneSearcher  # type: ignore[import]

        if prebuilt_index:
            logger.info("Loading prebuilt index %r (downloads on first use)", prebuilt_index)
            self._searcher = LuceneSearcher.from_prebuilt_index(prebuilt_index)
        elif index_path:
            self._searcher = LuceneSearcher(index_path)
        else:
            raise ValueError("Provide either index_path or prebuilt_index")

        self._searcher.set_bm25(k1=0.9, b=0.4)  # MSMARCO-tuned defaults
        self._top_k = top_k

    def retrieve(self, question: str) -> list[Document]:
        hits = self._searcher.search(question, k=self._top_k)
        docs: list[Document] = []
        for i, hit in enumerate(hits, 1):
            raw = json.loads(self._searcher.doc(hit.docid).raw())
            # MSMARCO passages use "contents"; BEIR/custom may use "text" or "passage"
            text = raw.get("contents") or raw.get("text") or raw.get("passage") or ""
            title = raw.get("title", "")
            full_text = f"{title}\n\n{text}".strip() if title else text
            docs.append(Document(doc_id=f"doc_{i}", text=full_text))
        logger.debug("PyseriniRetriever returned %d docs for %r", len(docs), question)
        return docs
