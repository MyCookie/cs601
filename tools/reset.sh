#!/usr/bin/env bash
# Licensed under Apache 2.0
# Reset one or all task statuses so they can be re-run.
#
# Usage:  ./reset.sh [TASK_ID|PHASE]
#   ./reset.sh T01           reset single task
#   ./reset.sh textbook      reset entire phase
#   ./reset.sh               reset all tasks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

init_status

TARGET="${1:-all}"

reset_ids() {
  local ids="$1"
  for tid in $ids; do
    set_status "$tid" "pending"
    # Clean up log
    rm -f "${LOG_DIR}/run_${tid^^}.log" "${LOG_DIR}/verify_${tid^^}.log"
    info "Reset ${tid} → pending"
  done
}

case "$TARGET" in
  textbook|appendix|lecture|labs|assessment|infrastructure|all)
    reset_ids "$(phase_range "$TARGET")"
    ;;
  T*)
    reset_ids "$TARGET"
    ;;
  *)
    err "Unknown target: $TARGET"
    echo "  Pass a task ID (T01…T63) or phase name."
    exit 1
    ;;
esac

info "Done. Run ./status.sh to verify."
