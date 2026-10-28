#!/usr/bin/env bash
# Run the atomic-claim intrinsic eval for one model: extract -> score.
#
# Usage:
#   bash run_eval.sh --input <alce-result.json>                  # 100 examples, default model
#   bash run_eval.sh --input <alce-result.json> --n 20           # smaller run
#   bash run_eval.sh --input <alce-result.json> --extract-only   # LLM step only
#   bash run_eval.sh --score-only --metrics atomicity coverage   # NLI step only, on existing predictions
#
# Environment overrides: MODEL, API_BASE, N, SEED, ALCE_INPUT, ENV_NAME. The API key is read
# from API_KEY / OPENROUTER_API_KEY by run_extractor.py (never passed on the command line).
# Unrecognised flags are forwarded to intrinsic_eval.py.

set -euo pipefail

ORIG_PWD="$PWD"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

ENV_NAME="${ENV_NAME:-acclaim}"
MODEL="${MODEL:-openrouter/google/gemma-4-26b-a4b-it}"
API_BASE="${API_BASE:-}"
N="${N:-100}"
SEED="${SEED:-42}"
INPUT="${ALCE_INPUT:-}"
DO_EXTRACT=1
DO_SCORE=1

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n) N="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api-base) API_BASE="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --extract-only) DO_SCORE=0; shift ;;
        --score-only) DO_EXTRACT=0; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# e.g. "openrouter/anthropic/claude-haiku-4.5" -> "claude-haiku-4.5"
MODEL_SANITIZED="$(echo "$MODEL" | sed -E 's#.*/##; s/:.*//')"
PREDICTIONS="$HERE/predictions/${MODEL_SANITIZED}.jsonl"
RESULTS_PREFIX="$HERE/results/${MODEL_SANITIZED}"

if [[ $DO_EXTRACT -eq 1 ]]; then
    if [[ -z "$INPUT" ]]; then
        echo "error: pass --input <alce-result.json> (or set ALCE_INPUT); see README 'Data'" >&2
        exit 1
    fi
    # Resolve a relative --input against the caller's cwd, since we cd to REPO_ROOT below.
    [[ "$INPUT" != /* ]] && INPUT="$ORIG_PWD/$INPUT"
fi

cd "$REPO_ROOT"

if [[ $DO_EXTRACT -eq 1 ]]; then
    echo "[extract] n=$N seed=$SEED model=$MODEL"
    # An empty --api-base lets LiteLLM route on the model prefix (e.g. "openrouter/...");
    # set API_BASE only for a local server such as vLLM.
    micromamba run -n "$ENV_NAME" python -m library_evals.atomic_claims.run_extractor \
        --input "$INPUT" --n "$N" --seed "$SEED" \
        --model "$MODEL" --api-base "$API_BASE" \
        --output "$PREDICTIONS"
fi

if [[ $DO_SCORE -eq 1 ]]; then
    echo "[score] $PREDICTIONS"
    micromamba run -n "$ENV_NAME" python -m library_evals.atomic_claims.intrinsic_eval \
        --input "$PREDICTIONS" \
        --output-prefix "$RESULTS_PREFIX" \
        "${EXTRA_ARGS[@]}"
fi

echo "done: $MODEL_SANITIZED"
echo "  predictions: $PREDICTIONS"
[[ $DO_SCORE -eq 1 ]] && echo "  summary:     ${RESULTS_PREFIX}_summary.json"
exit 0
