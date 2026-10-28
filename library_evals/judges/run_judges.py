"""
Run the configured evidence judge on each (claim, doc) pair in data/pairs.jsonl
and write judge predictions to predictions/<run>.jsonl.

The judge is built exactly as the library builds it from the ``judge:``
section of the config, so both chat LLMs (``backend: chat``) and decision
models (``backend: decision``) are supported. CLI flags override the config.

Usage:
    python run_judges.py                     # uses config.yaml in this directory
    python run_judges.py --config my.yaml    # custom config path
    python run_judges.py --model openrouter/typesafe/jev-1.13 --backend decision --mode noul
    python run_judges.py --pairs my_pairs.jsonl --output my_preds.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import litellm
import yaml

# Allow running from this directory without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import acclaim.llm_client as llm_client  # noqa: E402
from acclaim.config.load import EvalConfig, JudgeConfig, _parse_config  # noqa: E402
from acclaim.data_models import Claim, Document  # noqa: E402
from acclaim.evaluate import _build_judge  # noqa: E402
from acclaim.judges.base import EvidenceJudge  # noqa: E402
from acclaim.judges.decision import DecisionJudge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
litellm.set_verbose = False
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_CONFIG = _HERE / "config.yaml"
_DEFAULT_PAIRS = _HERE / "data" / "pairs.jsonl"


def _load_judge_config(config_path: Path, args: argparse.Namespace) -> JudgeConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    judge_cfg = _parse_config({"judge": raw.get("judge", {})}).judge

    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.backend:
        overrides["backend"] = args.backend
    if args.mode:
        overrides["decision_mode"] = args.mode
    if args.state_format:
        overrides["decision_state_format"] = args.state_format
    if args.thresholds:
        overrides["decision_thresholds"] = json.loads(args.thresholds)
    return dataclasses.replace(judge_cfg, **overrides)


def _run_name(cfg: JudgeConfig) -> str:
    name = cfg.model.split("/")[-1].split(":")[0]  # e.g. "claude-haiku-4.5"
    if cfg.backend == "decision":
        name += f"_{cfg.decision_mode}"
        if cfg.decision_state_format != "structured":
            name += f"_{cfg.decision_state_format}"
    return name


class _ChatCostTracker:
    """
    Records the cost of each chat completion made through LiteLLMClient by
    wrapping ``acclaim.llm_client.completion``. Sequential use only.
    """

    def __init__(self) -> None:
        self.cost = 0.0
        self._original = llm_client.completion
        llm_client.completion = self._completion

    def _completion(self, **kwargs: Any) -> Any:
        response = self._original(**kwargs)
        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        if cost is None:
            cost = (getattr(response, "_hidden_params", {}) or {}).get("response_cost")
        self.cost += float(cost or 0.0)
        return response

    def take(self) -> float:
        cost, self.cost = self.cost, 0.0
        return cost


def _judge_pair(
    judge: EvidenceJudge, claim: Claim, doc: Document, tracker: _ChatCostTracker | None
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if isinstance(judge, DecisionJudge):
        detailed = judge.judge_detailed(claim, [doc])
        result, probabilities = detailed.result, detailed.probabilities
        response = detailed.response
        cost = response.cost if response is not None else None
        answers = response.answers if response is not None else None
    else:
        result, probabilities, answers = judge.evaluate(claim, [doc]), None, None
        cost = tracker.take() if tracker is not None else None
    return {
        "judge_label": result.label.value,
        "judge_confidence": result.confidence,
        "judge_reason": result.reason,
        "judge_probabilities": probabilities,
        "decision_answers": answers,
        "latency_s": round(time.perf_counter() - t0, 4),
        "cost_usd": cost,
    }


def run_judges(
    judge_cfg: JudgeConfig,
    pairs_path: Path = _DEFAULT_PAIRS,
    output_path: Path | None = None,
    sleep: float = 0.0,
) -> Path:
    judge = _build_judge(EvalConfig(judge=judge_cfg))
    tracker = _ChatCostTracker() if judge_cfg.backend == "chat" else None
    output_path = output_path or _HERE / "predictions" / f"{_run_name(judge_cfg)}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    predictions = []
    for i, pair in enumerate(pairs):
        logger.info("Judging pair %d/%d (pair_id=%s)", i + 1, len(pairs), pair["pair_id"])
        claim = Claim(text=pair["claim"], span=(-1, -1))
        doc = Document(doc_id=pair["doc_id"], text=pair["doc_text"])
        predictions.append(
            {
                "pair_id": pair["pair_id"],
                "alce_index": pair["alce_index"],
                "claim": pair["claim"],
                "doc_id": pair["doc_id"],
                "doc_title": pair["doc_title"],
                **_judge_pair(judge, claim, doc, tracker),
            }
        )
        if sleep:
            time.sleep(sleep)  # e.g. 1.0 to stay under free-tier rate limits

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an evidence judge on sampled pairs.")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--pairs", type=Path, default=_DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: predictions/<model>[_<mode>[_<state_format>]].jsonl")
    parser.add_argument("--model", help="Override judge.model")
    parser.add_argument("--backend", choices=["chat", "decision"], help="Override judge.backend")
    parser.add_argument("--mode", choices=["choice", "noul", "score"],
                        help="Override judge.decision_mode")
    parser.add_argument("--state-format", choices=["structured", "text"],
                        help="Override judge.decision_state_format")
    parser.add_argument("--thresholds", help='noul thresholds as JSON, e.g. \'{"supported": 0.6}\'')
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between pairs")
    args = parser.parse_args()

    judge_cfg = _load_judge_config(args.config, args)
    run_judges(judge_cfg, args.pairs, args.output, args.sleep)


if __name__ == "__main__":
    main()
