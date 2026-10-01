#!/usr/bin/env bash
# Run an answer eval to completion, resuming across deaths.
#
# Why: a 500-question run takes ~66 h on this host and the process dies
# repeatedly — MPS/system OOM (the machine runs near memory exhaustion) and
# proxy 502s from the cloud reader/judge. The answer eval checkpoints after
# every question (atomically), so restarting with --resume loses at most the
# in-flight question. This wrapper does the restarting.
#
# It stops when the eval exits 0 (all questions done) and gives up after
# --max-stall consecutive attempts that make NO forward progress, so a genuine
# data/config error cannot spin forever.
#
# Usage:
#   scripts/run_answer_eval_resumable.sh <dataset> <output.json> [extra args...]
#
# Example:
#   scripts/run_answer_eval_resumable.sh data/longmemeval_s_cleaned.json \
#     results/eval_answer_s500_factdigest.json \
#     --root /tmp/coremem-factdigest-s500 \
#     --answer-model ollama:gpt-oss:120b-cloud \
#     --judge-model ollama:gpt-oss:120b-cloud

set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <dataset> <output.json> [--root DIR] [--answer-model M] [--judge-model M]" >&2
  exit 2
fi

DATASET=$1
OUTPUT=$2
shift 2

MAX_STALL=${MAX_STALL:-5}
# The 56-question gate run saw 8 MPS OOMs and one proxy 502 across ~3 days.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-200}
# MPS refuses allocations when the rest of the system is near-full; this lifts
# the watermark and was worth 18 -> 45 questions per process life.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

progress() {
  # Row count of the checkpoint, or 0 if it does not exist / is unreadable.
  python3 - "$OUTPUT" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        print(len(json.load(fh).get("results", [])))
except Exception:
    print(0)
PY
}

total=$(python3 - "$DATASET" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
except Exception:
    print(0)
PY
)

best=-1
stall=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  before=$(progress)
  echo "=== [$(date '+%F %T')] attempt $attempt/$MAX_ATTEMPTS | $before/$total done ==="

  uv run --no-sync python3 scripts/eval_answer_longmemeval.py \
    "$DATASET" --output "$OUTPUT" --resume "$@"
  status=$?
  after=$(progress)

  if [ "$status" -eq 0 ] && [ "$after" -ge "$total" ] && [ "$total" -gt 0 ]; then
    echo "=== [$(date '+%F %T')] COMPLETE: $after/$total ==="
    exit 0
  fi

  if [ "$after" -gt "$best" ]; then
    best=$after
    stall=0
    echo "--- progress $after/$total (exit $status); continuing ---"
  else
    stall=$((stall + 1))
    echo "--- NO progress (still $after/$total, exit $status); stall $stall/$MAX_STALL ---"
    if [ "$stall" -ge "$MAX_STALL" ]; then
      echo "=== giving up: $stall consecutive attempts without progress ===" >&2
      exit 1
    fi
  fi
  sleep 20
done

echo "=== giving up: exhausted $MAX_ATTEMPTS attempts ===" >&2
exit 1