#!/usr/bin/env bash
# The three SEAMS campaigns (GPU training, CPU evaluation), then the summary.
# One campaign at a time: three hybrid trainings do not fit in 12 GB of GPU memory.
# Usage: scripts/run_seams_comparison.sh [OUT] ; PROFILE=smoke for a quick check.
set -u
OUT=${1:-generated/seams/comparison}
PROFILE=${PROFILE:-compact-cpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Campaigns resume from their checkpoints, so a crash (e.g. GPU memory) is retried.
run() {
  local code=1
  for attempt in 1 2 3 4 5; do
    uv run --no-sync python -m examples.seams "$@" --profile "$PROFILE" --device cuda \
      --budget-seconds 86400
    code=$?
    [ "$code" -eq 0 ] && return 0
    echo "attempt $attempt exited with $code" >&2
    sleep 30
  done
  return "$code"
}

mkdir -p "$OUT/logs"
run space-fluid --stage all --out "$OUT/main" >> "$OUT/logs/main.log" 2>&1 &&
  run space-fluid-scenarios --out "$OUT/scenarios" >> "$OUT/logs/scenarios.log" 2>&1 &&
  run space-fluid-clusters --out "$OUT/clusters" >> "$OUT/logs/clusters.log" 2>&1 ||
  { echo "a campaign failed, see $OUT/logs" >&2; exit 1; }
uv run --no-sync python -m examples.seams space-fluid-summary --out "$OUT"
