#!/usr/bin/env bash
# Licensed under Apache 2.0
# Print the Claude Code prompt for one or more tasks without running them.
#
# Usage:  ./print_prompt.sh <TASK_ID> [<TASK_ID> ...]
#
# Examples:
#   ./print_prompt.sh T01
#   ./print_prompt.sh T01 T02 T03
#   ./print_prompt.sh T01 > prompt_t01.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ $# -eq 0 ]; then
  err "Usage: $0 <TASK_ID> [<TASK_ID> ...]"
  echo "  e.g.  $0 T01 T02"
  echo "  Or use run_task.sh -p for a single task."
  exit 2
fi

for tid in "$@"; do
  "${SCRIPT_DIR}/run_task.sh" --print-prompt "$(echo "$tid" | tr '[:upper:]' '[:lower:]')"
done
