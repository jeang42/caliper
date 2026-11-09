#!/usr/bin/env bash
# Run one approved benchmark stage from docs/CANDIDATES.md.
#
# The stages below name the models installed on the host this project was
# developed against, and their --context/--filler sizes are chosen for a
# 32 GiB card. On another machine they are a worked example rather than a
# recipe: edit the model lists, or call `caliper run` directly.
#
# Each stage runs preflight first and aborts if the GPU is busy, prints the
# plan, runs the benchmark, then writes a Markdown report and JSON/CSV exports
# into reports/. Nothing is downloaded. Models already resident on the GPU are
# never unloaded.
#
#   ./scripts/benchmark-stage.sh 2        # run stage 2
#
# A busy GPU is waited out for CALIPER_WAIT seconds (default 180) rather than
# aborting, because a stage queued after another often starts while the
# previous one is still settling.
#   ./scripts/benchmark-stage.sh 2 --dry  # show the plan only, no inference
#   ./scripts/benchmark-stage.sh list     # list the stages
set -euo pipefail

cd "$(dirname "$0")/.."
CALIPER="${CALIPER:-.venv/bin/caliper}"

if [[ ! -x "$CALIPER" ]]; then
    echo "error: $CALIPER not found. Create the venv first:" >&2
    echo "  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'" >&2
    exit 1
fi

# stage -> "description|caliper run arguments"
declare -A STAGES=(
  [1]="smoke test: gpt-oss:20b, instruction following only|--model gpt-oss:20b --category instruction_following --repetitions 2 --warmup 1"
  [2]="baseline: gpt-oss:20b, full suite, code execution on|--model gpt-oss:20b --repetitions 3 --warmup 1 --allow-code-execution --timeout 900"
  [3]="comparison: the three fast installed candidates, full suite|--model gpt-oss:20b --model qwen3-coder:30b --model qwen3.5:35b --repetitions 3 --warmup 1 --allow-code-execution --timeout 900"
  [4]="controls: add a different architecture and a dense model|--model glm-4.7-flash:latest --model qwen3.8:27b --repetitions 3 --warmup 1 --allow-code-execution --timeout 900"
  [5a]="context scaling at 16k|--model gpt-oss:20b --model qwen3-coder:30b --category long_context --repetitions 2 --warmup 1 --context 16384 --timeout 900"
  [5b]="context scaling at 32k|--model gpt-oss:20b --model qwen3-coder:30b --category long_context --repetitions 2 --warmup 1 --context 32768 --timeout 900"
  [5c]="context scaling at 64k|--model gpt-oss:20b --model qwen3-coder:30b --category long_context --repetitions 2 --warmup 1 --context 65536 --timeout 1200"
  # Suite v2 replaces the long-context category with eight harder tasks. The
  # window stays at the tasks' own 16384 so this measures difficulty, not
  # offloading; 8b re-measures the same tasks at 32k for the scaling curve.
  [8a]="v2 long context: all five measured models, harder tasks|--suite v2 --category long_context --model gpt-oss:20b --model qwen3-coder:30b --model qwen3.5:35b --model glm-4.7-flash:latest --model qwen3.8:27b --repetitions 3 --warmup 1 --timeout 1200"
  [8b]="v2 long context at 32k|--suite v2 --category long_context --model gpt-oss:20b --model qwen3-coder:30b --repetitions 2 --warmup 1 --context 32768 --timeout 1200"
  # Stage 9 varies the one axis no run has ever moved: the size of the
  # document itself. --context only moves num_ctx; --filler moves the
  # document. 9a doubles v2's measured ~9,200 prompt tokens; 9b takes the two
  # models whose weights leave room for a 64k window up to roughly five times
  # it. Peak GPU memory measured at num_ctx 65536 was 19,447 MiB for
  # gpt-oss:20b and 31,195 MiB for qwen3-coder:30b of 32,760 total, so 9b is
  # close to the card's limit and may partially offload - which the report
  # records rather than hides.
  [9a]="long-document sweep: ~16k-token documents, all five models|--suite v2 --category long_context --model gpt-oss:20b --model qwen3-coder:30b --model qwen3.5:35b --model glm-4.7-flash:latest --model qwen3.8:27b --repetitions 2 --warmup 1 --filler 16000 --context 32768 --timeout 1800"
  [9b]="long-document sweep: ~40k-token documents, the two models that fit a 64k window|--suite v2 --category long_context --model gpt-oss:20b --model qwen3-coder:30b --repetitions 2 --warmup 1 --filler 40000 --context 65536 --timeout 2400"
)

if [[ "${1:-}" == "list" || -z "${1:-}" ]]; then
    echo "Stages (see docs/CANDIDATES.md for the rationale):"
    for key in 1 2 3 4 5a 5b 5c 8a 8b 9a 9b; do
        printf "  %-3s %s\n" "$key" "${STAGES[$key]%%|*}"
    done
    echo
    echo "Stages 6 and 7 need a download and are deliberately not scripted."
    exit 0
fi

STAGE="$1"; shift || true
if [[ -z "${STAGES[$STAGE]:-}" ]]; then
    echo "error: unknown stage '$STAGE'. Try: $0 list" >&2
    exit 1
fi

DESC="${STAGES[$STAGE]%%|*}"
ARGS="${STAGES[$STAGE]#*|}"
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1

echo "=== Stage $STAGE: $DESC ==="
echo

echo "--- preflight ---"
# Stages are often queued back to back, and a run that has just finished leaves
# the GPU busy for a few seconds. Wait for it to settle rather than failing, or
# worse, lowering the safety threshold.
if ! $CALIPER preflight --disk / --wait "${CALIPER_WAIT:-180}"; then
    echo
    echo "ABORTED: the machine is too busy to benchmark safely." >&2
    echo "Re-run later, or override deliberately by adding --force to the run command." >&2
    exit 2
fi
echo

echo "--- plan ---"
# shellcheck disable=SC2086
$CALIPER plan --engine ollama $ARGS
echo

if [[ "$DRY" == "1" ]]; then
    echo "--dry given: stopping before inference."
    exit 0
fi

echo "--- run ---"
# shellcheck disable=SC2086
$CALIPER run --engine ollama $ARGS --wait "${CALIPER_WAIT:-180}" \
    --label "stage $STAGE: $DESC"
echo

RUN_ID="$($CALIPER runs --limit 1 | awk 'NR==2 {print $1}')"
REPORT="reports/stage${STAGE}-${RUN_ID}.md"
echo "--- report ---"
$CALIPER report --run "$RUN_ID" -o "$REPORT"
$CALIPER export --run "$RUN_ID" --out-dir reports --responses
echo
echo "Stage $STAGE complete."
echo "  report:  $REPORT"
echo "  run id:  $RUN_ID"
