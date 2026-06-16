#!/usr/bin/env bash
# Licensed under Apache 2.0
# Run all tasks in a phase concurrently via background jobs.
#
# Usage:  ./run_phase.sh [-j NUM] <PHASE> [MAX_PARALLEL]
#
#   -j NUM, --jobs NUM   Maximum concurrent Claude Code invocations (1-5, default: 5)
#   PHASE                textbook | appendix | lecture | labs | assessment | infrastructure | all
#   MAX_PARALLEL         Deprecated — use -j / --jobs instead (still accepted)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
MAX_PAR="${MAX_JOBS}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)
      if [[ -z "${2:-}" ]] || [[ "$2" == -* ]]; then
        err "--jobs requires a numeric argument (1-5)"
        exit 2
      fi
      validate_jobs "$2" || exit 2
      MAX_PAR="$2"; shift 2
      ;;
    -*)
      err "Unknown option: $1"
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

PHASE="${1:-}"
# Backward compat: second positional arg overrides if provided
if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
  validate_jobs "$2" || exit 2
  MAX_PAR="$2"
  warn "Positional MAX_PARALLEL is deprecated — use --jobs instead"
fi

if [ -z "${PHASE}" ]; then
  err "Usage: $0 [-j NUM] <PHASE>"
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
    # Filter out dead PIDs — the while-read subshell exits non-zero when
    # all PIDs are dead, so we need || true to prevent set -e from killing us.
    PIDS=($(printf '%s\n' "${PIDS[@]}" | while read p; do
      kill -0 "$p" 2>/dev/null && echo "$p"
    done || true))

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
if (( ${#PIDS[@]} > 0 )); then
  for pid in "${PIDS[@]}"; do
    if wait "$pid" 2>/dev/null; then
      PASSED=$((PASSED + 1))
    else
      FAILED=$((FAILED + 1))
    fi
  done
fi

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
