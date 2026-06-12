#!/usr/bin/env bash
# Licensed under Apache 2.0
# Show the execution status of all 63 tasks.
#
# Usage:  ./status.sh [PHASE]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

PHASE="${1:-all}"
TASKS="$(phase_range "$PHASE")"

init_status

echo ""
echo -e "${BOLD}  CS601 Task Status — Phase: ${PHASE}${RESET}"
echo -e "${BOLD}  ─────────────────────────────────────────${RESET}"

TOTAL=0; PENDING=0; RUNNING=0; PASSED=0; FAILED=0

for tid in $TASKS; do
  TOTAL=$((TOTAL + 1))
  s="$(task_status "$tid")"
  title="$(task_title "$tid" 2>/dev/null || echo "$tid")"
  case "$s" in
    passed)  PASSED=$((PASSED+1));  icon="${GREEN}✓${RESET}"; ;;
    running) RUNNING=$((RUNNING+1)); icon="${BLUE}◐${RESET}"; ;;
    failed)  FAILED=$((FAILED+1));  icon="${RED}✗${RESET}"; ;;
    *)       PENDING=$((PENDING+1)); icon="${YELLOW}○${RESET}"; ;;
  esac
  printf "  ${icon} %-5s %s\n" "$tid" "$title"
done

echo -e "${BOLD}  ─────────────────────────────────────────${RESET}"
echo -e "  Total: ${TOTAL}  ${GREEN}Passed: ${PASSED}${RESET}  ${BLUE}Running: ${RUNNING}${RESET}  ${YELLOW}Pending: ${PENDING}${RESET}  ${RED}Failed: ${FAILED}${RESET}"
echo ""
