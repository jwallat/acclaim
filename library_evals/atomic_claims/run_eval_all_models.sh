#!/usr/bin/env bash
# Run run_eval.sh for each OpenRouter model below. All arguments are forwarded,
# so --input, --n, --extract-only, --score-only and intrinsic_eval flags work here too.
#
# Usage:
#   OPENROUTER_API_KEY=... bash run_eval_all_models.sh --input <alce-result.json>
#   bash run_eval_all_models.sh --score-only --metrics atomicity coverage coverage_whole non_redundancy sentence_claim_coverage

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    "openrouter/anthropic/claude-haiku-4.5"
    "openrouter/anthropic/claude-haiku-5.5"
    "openrouter/google/gemma-4-26b-a4b-it"
    "openrouter/openai/gpt-5.4-mini"
    "openrouter/openai/gpt-6-luna"
    "openrouter/qwen/qwen3.6-35b-a3b"
    "openrouter/meta-llama/llama-3.1-8b-instruct"
)

for m in "${MODELS[@]}"; do
    echo "=== $m ==="
    MODEL="$m" bash "$HERE/run_eval.sh" "$@"
done
