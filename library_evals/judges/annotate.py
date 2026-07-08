"""
Terminal annotation tool for labeling (claim, doc) pairs.

Reads pairs from pairs.jsonl, loads any existing labels from annotation_sheet.csv,
and saves labels back to annotation_sheet.csv after each keystroke.

Usage:
    python annotate.py
    python annotate.py --pairs my_pairs.jsonl --output my_labels.csv

Keys:
    S  — SUPPORTED
    R  — REFUTED
    U  — UNCLEAR
    →/n — next pair (skip without labeling)
    ←/p — previous pair
    g   — jump to first unlabeled
    q   — quit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import termios
import textwrap
import tty
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_PAIRS = _HERE / "pairs.jsonl"
_DEFAULT_OUTPUT = _HERE / "annotation_sheet.csv"

LABELS = {"s": "SUPPORTED", "r": "REFUTED", "u": "UNCLEAR"}

# ANSI helpers
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
CLEAR = "\033[2J\033[H"

LABEL_COLOR = {"SUPPORTED": GREEN, "REFUTED": RED, "UNCLEAR": YELLOW, "": DIM}


def _getch() -> str:
    """Read a single keypress (including arrow keys) without Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode("utf-8", errors="replace")
        if ch == "\x1b":  # escape sequence
            rest = os.read(fd, 2).decode("utf-8", errors="replace")
            ch = ch + rest
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {
            int(row["pair_id"]): row["human_label"].strip().upper()
            for row in csv.DictReader(f)
        }


def _save_labels(pairs: list[dict], labels: dict[int, str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "claim", "doc_id", "doc_title", "doc_text_truncated", "human_label"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "pair_id": pair["pair_id"],
                "claim": pair["claim"],
                "doc_id": pair["doc_id"],
                "doc_title": pair["doc_title"],
                "doc_text_truncated": pair["doc_text"][:300],
                "human_label": labels.get(pair["pair_id"], ""),
            })


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total}"


def _wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    lines = textwrap.wrap(text, width - len(indent))
    return "\n".join(indent + l for l in lines)


def _render(pair: dict, label: str, idx: int, total: int, labels: dict) -> str:
    done = sum(1 for v in labels.values() if v)
    col = LABEL_COLOR.get(label, DIM)
    term_width = os.get_terminal_size().columns

    sep = "─" * min(term_width, 80)
    lines = [
        CLEAR,
        f"{BOLD}Attribution Annotator{RESET}  {DIM}{_progress_bar(done, total)}{RESET}",
        sep,
        f"{BOLD}Pair {idx + 1} / {total}{RESET}  {DIM}(pair_id={pair['pair_id']}){RESET}",
        "",
        f"{BOLD}CLAIM{RESET}",
        _wrap(pair["claim"]),
        "",
        f"{BOLD}DOCUMENT{RESET}  {DIM}{pair['doc_title']}  (id: {pair['doc_id']}){RESET}",
        _wrap(pair["doc_text"], width=min(term_width, 80)),
        "",
        sep,
        (
            f"Current label: {col}{BOLD}{label or '—'}{RESET}"
            if label else
            f"Current label: {DIM}—{RESET}"
        ),
        "",
        (
            f"  {GREEN}[S]{RESET} SUPPORTED   "
            f"{RED}[R]{RESET} REFUTED   "
            f"{YELLOW}[U]{RESET} UNCLEAR   "
            f"{DIM}[←/p] prev  [→/n] next  [g] first unlabeled  [q] quit{RESET}"
        ),
    ]
    return "\n".join(lines)


def run(pairs_path: Path, output_path: Path) -> None:
    pairs = _load_pairs(pairs_path)
    labels = _load_labels(output_path)
    idx = 0

    # Start at first unlabeled pair
    for i, p in enumerate(pairs):
        if not labels.get(p["pair_id"]):
            idx = i
            break

    while True:
        pair = pairs[idx]
        label = labels.get(pair["pair_id"], "")
        print(_render(pair, label, idx, len(pairs), labels), end="", flush=True)

        ch = _getch().lower()

        if ch in LABELS:
            labels[pair["pair_id"]] = LABELS[ch]
            _save_labels(pairs, labels, output_path)
            if idx < len(pairs) - 1:
                idx += 1
        elif ch in ("n", "\x1b[c", "\x1b[a"):  # n, →, ↑
            if idx < len(pairs) - 1:
                idx += 1
        elif ch in ("p", "\x1b[d", "\x1b[b"):  # p, ←, ↓
            if idx > 0:
                idx -= 1
        elif ch == "g":
            for i, p in enumerate(pairs):
                if not labels.get(p["pair_id"]):
                    idx = i
                    break
        elif ch in ("q", "\x03"):  # q or Ctrl-C
            done = sum(1 for v in labels.values() if v)
            print(f"\n{BOLD}Saved.{RESET} {done}/{len(pairs)} labeled → {output_path}\n")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal annotator for claim-doc pairs.")
    parser.add_argument("--pairs", type=Path, default=_DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.pairs.exists():
        print(f"Error: {args.pairs} not found. Run sample_pairs.py first.")
        sys.exit(1)

    run(args.pairs, args.output)


if __name__ == "__main__":
    main()
