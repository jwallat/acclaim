"""
Run the configured LiteLLMJudge on each (claim, doc) pair in pairs.jsonl
and write judge predictions to predictions.jsonl.

Usage:
    python run_judges.py                     # uses config.yaml in this directory
    python run_judges.py --config my.yaml    # custom config path
    python run_judges.py --pairs my_pairs.jsonl --output my_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import litellm
import yaml

# Allow running from this directory without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acclaim.data_models import Claim, Document
from acclaim.judges.litellm import LiteLLMJudge

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
litellm.set_verbose = False
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_CONFIG = _HERE / "config.yaml"
_DEFAULT_PAIRS = _HERE / "pairs.jsonl"
_DEFAULT_OUTPUT = _HERE / "predictions.jsonl"


def _resolve(value: object) -> object:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value


def _load_judge(config_path: Path) -> Tuple[str, LiteLLMJudge]:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    judge_raw = raw.get("judge", {})
    model = str(_resolve(judge_raw.get("model", "gpt-4o-mini")))
    return model, LiteLLMJudge(
        model=model,
        api_base=_resolve(judge_raw.get("api_base")) or None,
        api_key=_resolve(judge_raw.get("api_key")) or None,
        temperature=judge_raw.get("temperature", 0.0),
        max_tokens=judge_raw.get("max_tokens", 512),
        max_retries=judge_raw.get("max_retries", 3),
        system_prompt=judge_raw.get("system_prompt") or None,
    )


def run_judges(
    config_path: Path = _DEFAULT_CONFIG,
    pairs_path: Path = _DEFAULT_PAIRS,
    output_path: Path = _DEFAULT_OUTPUT,
) -> None:
    model, judge = _load_judge(config_path)

    model_sanitized = model.split("/")[-1].split(":")[0]  # e.g. "claude-haiku-4.5"

    output_path = _HERE / f"predictions_{model_sanitized}.jsonl"

    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    predictions = []
    for i, pair in enumerate(pairs):
        logger.info("Judging pair %d/%d (pair_id=%s)", i + 1, len(pairs), pair["pair_id"])
        claim = Claim(text=pair["claim"], span=(-1, -1))
        doc = Document(doc_id=pair["doc_id"], text=pair["doc_text"])
        result = judge.evaluate(claim, [doc])
        time.sleep(1)  # stay under 16 req/min free-tier limit
        predictions.append(
            {
                "pair_id": pair["pair_id"],
                "alce_index": pair["alce_index"],
                "claim": pair["claim"],
                "doc_id": pair["doc_id"],
                "doc_title": pair["doc_title"],
                "judge_label": result.label.value,
                "judge_confidence": result.confidence,
                "judge_reason": result.reason,
            }
        )

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiteLLMJudge on sampled pairs.")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--pairs", type=Path, default=_DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_judges(args.config, args.pairs, args.output)


if __name__ == "__main__":
    main()
