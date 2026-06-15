#!/usr/bin/env bash
# Licensed under Apache 2.0
# Shared configuration and helper functions for the CS601 toolchain.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASKS_FILE="${REPO_ROOT}/TASKS.md"
TOOLS_DIR="${REPO_ROOT}/tools"
LOG_DIR="${REPO_ROOT}/.task_logs"
STATUS_FILE="${REPO_ROOT}/.task_status.json"

# ---------------------------------------------------------------------------
# Claude CLI
# ---------------------------------------------------------------------------
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_FLAGS=(
  --dangerously-skip-permissions
  --effort max
  --disable-slash-commands
)

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
# Default max concurrent Claude Code invocations.
# Override via CS601_JOBS env var or --jobs / -j flag on run_phase.sh.
MAX_JOBS="${CS601_JOBS:-1}"

validate_jobs() {
  local n="$1"
  # Must be an integer between 1 and 5
  if ! [[ "$n" =~ ^[0-9]+$ ]] || (( n < 1 || n > 5 )); then
    err "--jobs must be an integer between 1 and 5 (got: ${n})"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Colours (disable when stdout is not a TTY)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
  BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; RESET=''
fi

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { printf "${GREEN}[TASK]%s${RESET} $*\n" "$(date +%H:%M:%S)" >&2; }
info() { printf "${BLUE}[INFO]%s${RESET} $*\n" "$(date +%H:%M:%S)" >&2; }
warn() { printf "${YELLOW}[WARN]%s${RESET} $*\n" "$(date +%H:%M:%S)" >&2; }
err()  { printf "${RED}[ERR]%s${RESET}  $*\n" "$(date +%H:%M:%S)" >&2; }

ensure_log_dir() { mkdir -p "${LOG_DIR}"; }

# ---------------------------------------------------------------------------
# Portable string helpers (no bash 4.0+ parameter expansion)
# ---------------------------------------------------------------------------
to_upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

# ---------------------------------------------------------------------------
# Task metadata parser  (extracts fields from TASKS.md by task id)
# ---------------------------------------------------------------------------
# Returns the full task block for a given ID (e.g. "T01", "t01").
# The block is delimited by "---" lines.
task_block() {
  local id
  id="$(to_upper "$1")"
  # awk: grab everything between "### ${id}" and the next "---"
  awk -v id="### ${id}" '
    $0 ~ id { found=1 }
    found { print }
    found && /^---/ { exit }
  ' "${TASKS_FILE}"
}

# Extract branch name from a task block.
task_branch() {
  task_block "$1" | grep -o 'task/[^`]*' | sed 's|^task/||' | head -1
}

# Extract expected files from a task block.
# Stops at the first line that starts with "- ** (a new bold key).
task_files() {
  local _block
  _block="$(task_block "$1")"
  echo "${_block}" | awk '
    /\*\*Files (created|modified):\*\*/ { in_section=1; next }
    in_section && /^[[:space:]]*-[[:space:]]*\*\*/ { exit }
    in_section && /^[[:space:]]*-[[:space:]]/ {
      sub(/^[[:space:]]*-[[:space:]]*/, "")
      gsub(/`/, "")
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      if (length($0) > 0) print
    }
  '
}

# Extract the definition-of-done bullets.
task_dod() {
  local _block
  _block="$(task_block "$1")"
  echo "${_block}" | awk '
    /\*\*Definition of done:\*\*/ { in_section=1; next }
    in_section && /^---/ { exit }
    in_section && /^[[:space:]]*-[[:space:]]/ {
      sub(/^[[:space:]]*-[[:space:]]*/, "")
      gsub(/`/, "")
      if (length($0) > 0) print
    }
  '
}

# Get the task title line.
task_title() {
  task_block "$1" | sed -n 's/^### //p' | head -1 | sed 's/^[A-Z][0-9][0-9][A-Z]*[[:space:]]*—[[:space:]]*//'
}

# ---------------------------------------------------------------------------
# Phase -> task id range
# ---------------------------------------------------------------------------
phase_range() {
  case "$1" in
    textbook)       echo "T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 T11 T12 T13 T14" ;;
    appendix)       echo "T15 T16 T17 T18 T19" ;;
    lecture)        echo "T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 T32 T33" ;;
    labs)           echo "T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47" ;;
    assessment)     echo "T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61" ;;
    infrastructure) echo "T62 T63" ;;
    all)            seq -f "T%02g" 1 63 | tr '\n' ' ' ;;
    *)              err "Unknown phase: $1"; return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# JSON status helpers  (lightweight, no jq dependency)
# ---------------------------------------------------------------------------
init_status() {
  if [ ! -f "${STATUS_FILE}" ]; then
    echo '{}' > "${STATUS_FILE}"
  fi
}

task_status() {
  local id
  id="$(to_upper "$1")"
  if [ -f "${STATUS_FILE}" ]; then
    python3 -c "
import json, sys
try:
    d = json.load(open('${STATUS_FILE}'))
    print(d.get('${id}', 'pending'))
except: print('pending')
" 2>/dev/null || echo "pending"
  else
    echo "pending"
  fi
}

set_status() {
  local id status
  id="$(to_upper "$1")"
  status="$2"
  python3 -c "
import json
path = '${STATUS_FILE}'
try:
    d = json.load(open(path))
except FileNotFoundError:
    d = {}
d['${id}'] = '${status}'
json.dump(d, open(path, 'w'), indent=2)
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
git_create_branch() {
  local branch="$1"
  local _existing
  _existing="$(git branch --list "$branch" 2>/dev/null || true)"
  if [ -n "$_existing" ]; then
    git branch -D "$branch" >/dev/null 2>&1 || true
  fi
  git checkout -b "$branch" main 2>/dev/null || git checkout -b "$branch"
}

git_commit_task() {
  local id="$1"
  local title
  title="$(task_title "$id")"
  git add -A
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "task/${id}: ${title}"
  fi
}
