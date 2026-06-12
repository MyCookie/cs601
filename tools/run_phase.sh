#!/usr/bin/env bash
# Licensed under Apache 2.0
# Run all tasks in a phase concurrently via background jobs.
#
# Usage:  ./run_phase.sh <PHASE> [MAX_PARALLEL]
#
# Phases: textbook | appendix | lecture | labs | assessment | infrastructure | all
# MAX_PARALLEL defaults to 5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

PHASE="${1:-}"
MAX_PAR="${2:-5}"

if [ -z "${PHASE}" ]; then
  err "Usage: $0 <PHASE> [MAX_PARALLEL]"
  echo "  Phases: textbook appendix lecture labs assessment infrastructure all"
  exit 2
fi

ensure_log_dir
init_status

# ---------------------------------------------------------------------------
# Get task list
# ---------------------------------------------------------------------------
TASKS="$(phase_range "$PHASE")"
TOTAL=$(echo "$TASKS" | wc -w | tr -d ' ')
info "Phase '${PHASE}': ${TOTAL} tasks (max parallel: ${MAX_PAR})"

# ---------------------------------------------------------------------------
# Semaphore: limit concurrent jobs
# ---------------------------------------------------------------------------
ACTIVE=0
PASSED=0
FAILED=0
PIDS=()

wait_slot() {
  while (( ACTIVE >= MAX_PAR )); do
    # Reap finished jobs
    local new_active=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_active=$((new_active + 1))
      fi
    done
    ACTIVE=$new_active
    PIDS=($(printf '%s\n' "${PIDS[@]}" | while read p; do
      kill -0 "$p" 2>/dev/null && echo "$p"
    done))

    if (( ACTIVE >= MAX_PAR )); then
      sleep 5
    fi
  done
}

for tid in $TASKS; do
  wait_slot

  # Skip already-passed tasks
  local_status="$(task_status "$tid")"
  if [[ "$local_status" == "passed" ]]; then
    info "${tid} already passed — skipping"
    PASSED=$((PASSED + 1))
    continue
  fi

  info "Launching ${tid}..."
  "${SCRIPT_DIR}/run_task.sh" "$tid" &
  PIDS+=($!)
  ACTIVE=$((ACTIVE + 1))
done

# ---------------------------------------------------------------------------
# Wait for all
# ---------------------------------------------------------------------------
info "Waiting for all jobs to complete..."
for pid in "${PIDS[@]}"; do
  if wait "$pid" 2>/dev/null; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo -e "${BOLD}=== Phase '${PHASE}' Complete ===${RESET}"
echo -e "  Total:  ${TOTAL}"
echo -e "  ${GREEN}Passed: ${PASSED}${RESET}"
echo -e "  ${RED}Failed: ${FAILED}${RESET}"
echo ""

if (( FAILED > 0 )); then
  echo "Failed tasks:"
  for tid in $TASKS; do
    s="$(task_status "$tid")"
    if [[ "$s" == "failed" ]]; then
      echo "  ${RED}${tid}${RESET}"
    fi
  done
  exit 1
fi

exit 0
