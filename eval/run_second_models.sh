#!/usr/bin/env bash
# Second-model sweep for the ablation study (paper item A).
#
# Per model: B2 s8 (400 calls, the production-evidence arm the paper's claim rests on) and B3
# (400 calls). The s40 arm is deliberately NOT repeated - it exists only to show that 5x evidence
# does not close the gap, and running it on every model doubles the bill for a column nobody reads.
#
# Model order matters: each model finishes completely before the next starts, so a token expiry
# leaves whole models done rather than all of them half done. Every batch is resumable - re-run
# this script after `~/bin/opencode auth login -p llm_gateway` and it picks up where it stopped.
#
#   bash eval/run_second_models.sh            # all three
#   bash eval/run_second_models.sh claude-sonnet-5
# Each batch runs up to PASSES times. A pass only re-requests signatures whose last cache record is
# _ok=False, so the extra passes cost nothing when a batch already completed - and they are what
# recovers the transport losses (dropped payloads, 429s) that a single pass leaves behind.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=../.venv/bin/python
PASSES=${PASSES:-3}
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(claude-sonnet-5 gpt-5.4-2026-03-05 deepseekv4-flash)

for m in "${MODELS[@]}"; do
  for study in b2 b3; do
    for pass in $(seq 1 "$PASSES"); do
      echo "=== $(date +%H:%M:%S)  $study  $m  pass $pass/$PASSES ==="
      if [ "$study" = "b2" ]; then
        out=$($PY b2_llm_numbers.py --run s8 --model "$m" --workers 8 2>&1) || exit 1
      else
        out=$($PY b3_single_call.py --run --model "$m" --workers 8 2>&1) || exit 1
      fi
      echo "$out" | tail -2
      echo "$out" | grep -q "0 permanent failure" && break
    done
  done
done
echo "=== $(date +%H:%M:%S)  sweep complete ==="
