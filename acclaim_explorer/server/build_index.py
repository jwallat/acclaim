"""Build a local Lucene BM25 index from a JSONL corpus for offline retrieval.

Each line of the corpus must be:  {"id": "...", "contents": "..."}

Usage
-----
# Full MSMARCO passage collection (~3.8 GB download, builds ~1.7 GB index):
python -m acclaim_explorer.server.build_index --corpus ./data/collection.tsv --output ./data/bm25_index

# From a JSONL corpus you already have:
python -m acclaim_explorer.server.build_index --corpus ./data/corpus.jsonl --output ./data/bm25_index

Then set in acclaim_explorer/acclaim_explorer_config.yaml:
  bm25:
    index_path: ./data/bm25_index

Tip: For a quick start without building a local index, use the prebuilt option instead:
  bm25:
    prebuilt_index: msmarco-v1-passage   # downloads ~1.7 GB on first search
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# MSMARCO passage collection (TSV: id\tpassage)
_MSMARCO_URL = (
    "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz"
)


def _check_java() -> None:
    if shutil.which("java") is None:
        sys.exit(
            "Java not found. pyserini requires Java 11+.\n"
            "Install: https://www.java.com/en/download/"
        )


def _tsv_to_jsonl(tsv_path: Path, jsonl_path: Path, max_docs: int | None) -> None:
    """Convert MSMARCO TSV (id<TAB>passage) to pyserini JSONL format."""
    print(f"Converting {tsv_path} → {jsonl_path} …")
    opener = gzip.open if str(tsv_path).endswith(".gz") else open
    count = 0
    with opener(tsv_path, "rt", encoding="utf-8") as fin, open(jsonl_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            doc_id, contents = parts
            fout.write(json.dumps({"id": doc_id, "contents": contents}) + "\n")
            count += 1
            if max_docs and count >= max_docs:
                break
            if count % 500_000 == 0:
                print(f"  {count:,} passages …")
    print(f"  Wrote {count:,} passages.")


def _build_index(corpus_jsonl: Path, output_dir: Path) -> None:
    """Call pyserini's Lucene indexer on a JSONL corpus directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # pyserini expects a directory of JSONL files
    corpus_dir = corpus_jsonl.parent

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(corpus_dir),
        "--index", str(output_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(min(8, os.cpu_count() or 4)),
        "--storeRaw",
    ]
    print("Indexing …")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Index written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Path to corpus file (.jsonl, .tsv, or .tsv.gz). "
             "TSV format: id<TAB>passage per line. "
             "JSONL format: {\"id\": \"...\", \"contents\": \"...\"} per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/bm25_index"),
        help="Directory to write the Lucene index to (default: ./data/bm25_index).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit to the first N passages (useful for building a small subset).",
    )
    parser.add_argument(
        "--download-msmarco",
        action="store_true",
        help="Download the full MSMARCO passage collection (~3.8 GB) before indexing.",
    )
    args = parser.parse_args()

    _check_java()

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = args.corpus

        if args.download_msmarco:
            tar_path = Path(tmp) / "collection.tar.gz"
            print(f"Downloading MSMARCO passage collection (~3.8 GB) …")
            urllib.request.urlretrieve(_MSMARCO_URL, tar_path)
            import tarfile
            with tarfile.open(tar_path) as tf:
                tf.extractall(tmp)
            corpus_path = Path(tmp) / "collection.tsv"

        if corpus_path is None:
            parser.error("Provide --corpus or --download-msmarco")

        suffix = "".join(corpus_path.suffixes)
        if suffix in (".tsv", ".tsv.gz"):
            jsonl_path = Path(tmp) / "corpus.jsonl"
            _tsv_to_jsonl(corpus_path, jsonl_path, args.max_docs)
        else:
            jsonl_path = corpus_path

        _build_index(jsonl_path, args.output)

    print("\nDone. Update acclaim_explorer/acclaim_explorer_config.yaml:")
    print(f"  bm25:")
    print(f"    index_path: {args.output}")


if __name__ == "__main__":
    main()
