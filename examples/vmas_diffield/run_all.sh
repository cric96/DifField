#!/usr/bin/env bash
# Launch the whole VMAS experiment suite (SHAC sweep + imitation + summary).
#
#   bash examples/vmas_diffield/run_all.sh            # full paper sweep (~6-7 h)
#   bash examples/vmas_diffield/run_all.sh --pilot    # quick sanity pass (~15 min)
#
# Results land under <repo>/generated/vmas-<scenario>-{shac,imitation}/ and
# <repo>/generated/vmas-comparison/ (override with OUT_ROOT=...).
#
# Crash resilience: VMAS occasionally dies with a native SIGSEGV/SIGILL from
# numpy RNG-state swapping under long runs (environmental, not our code — see
# the per-(policy,seed) checkpoint note in main.py). Each unit is therefore
# retried, and checkpoints make every retry a resume, so a crash costs one
# training unit instead of the sweep. Trust only the printed RESULT=DONE /
# RESULT=FAIL sentinels, never the exit code of intermediate steps.

set -u
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/generated}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

SCENARIOS=(flocking flocking_beacon navigation discovery sampling)
EXTRA_ARGS=()
SHAC_RETRIES=6
IMIT_RETRIES=4

if [[ "${1:-}" == "--pilot" ]]; then
  SCENARIOS=(flocking_beacon sampling)
  EXTRA_ARGS+=(--updates 20 --seeds 0 --num-envs 32)
  echo "PILOT MODE: scenarios=${SCENARIOS[*]}, 1 seed, 20 updates"
fi

run_with_retries() {
  local label="$1" max="$2"
  shift 2
  local attempt=0
  until "$@"; do
    attempt=$((attempt + 1))
    if (( attempt >= max )); then
      echo "RESULT=FAIL scenario=$label"
      exit 1
    fi
    echo "RETRY $label attempt=$attempt"
  done
}

for s in "${SCENARIOS[@]}"; do
  run_with_retries "$s" "$SHAC_RETRIES" \
    uv run python main.py --mode shac --scenarios "$s" \
      --out-root "$OUT_ROOT" "${EXTRA_ARGS[@]}"
  echo "SCENARIO_DONE=$s"
done

ALL_SCENARIOS="$(IFS=,; echo "${SCENARIOS[*]}")"
run_with_retries imitation "$IMIT_RETRIES" \
  uv run python main.py --mode imitation --scenarios "$ALL_SCENARIOS" \
    --out-root "$OUT_ROOT" "${EXTRA_ARGS[@]}"
echo "SCENARIO_DONE=imitation"

uv run python summarize.py --out-root "$OUT_ROOT"
echo "RESULT=DONE"
