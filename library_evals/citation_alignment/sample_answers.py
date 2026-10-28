"""
Sample 50 ASQA answers (no duplicate questions) across the three evaluated models.

The three tooling/outputs/alce_eval/asqa_*.jsonl files (gitignored, not in the
repo) all evaluate the exact same set of ASQA questions (same order, same
indices) with three different models. A seeded shuffle picks n question
indices, and the selected indices (in sorted order) are assigned to the three
models round-robin, so each question appears for exactly one model.

Usage:
    python sample_answers.py                      # writes data/sampled_answers.jsonl
    python sample_answers.py --n 50 --seed 42      # defaults shown
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_HERE = Path(__file__).parent
_ALCE_EVAL_DIR = _HERE.parent.parent / "tooling" / "outputs" / "alce_eval"
_MODELS = ["gemma-4-e4b", "llama-3.1-8b", "qwen3-8b"]
_DEFAULT_OUTPUT = _HERE / "data" / "sampled_answers.jsonl"


def _load_items(path: Path) -> list[dict]:
    """Load per-item records from a BatchEvaluationResult jsonl, skipping the aggregate line."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "item":
                items.append(rec)
    return sorted(items, key=lambda r: r["index"])


def sample_answers(
    alce_eval_dir: Path = _ALCE_EVAL_DIR,
    n: int = 50,
    seed: int = 42,
) -> list[dict]:
    per_model = {model: _load_items(alce_eval_dir / f"asqa_{model}.jsonl") for model in _MODELS}

    questions_by_model = {model: [rec["question"] for rec in items] for model, items in per_model.items()}
    reference = questions_by_model[_MODELS[0]]
    for model in _MODELS[1:]:
        if questions_by_model[model] != reference:
            raise ValueError(f"Question mismatch between {_MODELS[0]} and {model} — expected identical question sets/order.")

    n_questions = len(reference)
    if n > n_questions:
        raise ValueError(f"Requested n={n} but only {n_questions} questions are available.")

    rng = random.Random(seed)
    indices = list(range(n_questions))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:n])

    # Assign the selected question indices to the three models round-robin.
    assignment_order = list(_MODELS)
    examples = []
    for i, q_idx in enumerate(selected_indices):
        model = assignment_order[i % len(assignment_order)]
        rec = per_model[model][q_idx]
        examples.append({
            "example_id": i,
            "question_index": q_idx,
            "model": model,
            "question": rec["question"],
            "answer": rec["answer"],
            "documents": rec["documents"],
            "claims": rec["claims"],
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample ASQA answers for citation alignment eval.")
    parser.add_argument("--alce-eval-dir", type=Path, default=_ALCE_EVAL_DIR)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = sample_answers(args.alce_eval_dir, args.n, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts = {model: sum(1 for ex in examples if ex["model"] == model) for model in _MODELS}
    print(f"Wrote {len(examples)} examples to {args.output}")
    print(f"Model split: {counts}")


if __name__ == "__main__":
    main()
