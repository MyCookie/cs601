#!/usr/bin/env bash
# Licensed under Apache 2.0
# Run a single CS601 task via Claude Code.
#
# Usage:  ./run_task.sh [-p] <TASK_ID>
#
#   -p, --print-prompt   Print the prompt instead of running Claude
#
# The script:
#   1. Extracts task metadata from TASKS.md
#   2. Creates a new git branch
#   3. Invokes Claude Code to create all required content
#   4. Runs the verification suite
#   5. Commits on success (or leaves branch for manual fix)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
PRINT_PROMPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--print-prompt)
      PRINT_PROMPT=1; shift
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

TASK_ID="${1:-}"
if [ -z "${TASK_ID}" ]; then
  err "Usage: $0 [-p] <TASK_ID>"
  echo "  e.g.  $0 T01"
  exit 2
fi

TASK_ID="$(to_upper "${TASK_ID}")"
LOG_FILE="${LOG_DIR}/run_${TASK_ID}.log"
ensure_log_dir
init_status

# ---------------------------------------------------------------------------
# Extract metadata
# ---------------------------------------------------------------------------
BRANCH="task/$(task_branch "$TASK_ID" || echo "${TASK_ID}-work")"
TITLE="$(task_title "$TASK_ID" || echo "Task ${TASK_ID}")"
DOD="$(task_dod "$TASK_ID")"
FILES="$(task_files "$TASK_ID")"
BLOCK="$(task_block "$TASK_ID")"

# ---------------------------------------------------------------------------
# Build the Claude prompt
# ---------------------------------------------------------------------------
PROMPT="You are authoring content for the CS601 course.

Read \`${TASKS_FILE}\` and locate the section for **${TASK_ID}**.
Also read \`${REPO_ROOT}/textbook/OUTLINE.md\` for the authoring convention.

Execute task **${TASK_ID}** completely:

1. Create every file listed under 'Files created' in the task description.
2. Each file must have \`Licensed under Apache 2.0\` as the first line.
3. For every 'Diagram to render' bullet, create the diagram INLINE — use a
   Mermaid code block (\`\`\`mermaid ... \`\`\`) or an ASCII-art figure.
   Never describe a diagram without also rendering it.
4. For every 'Code blocks' bullet, include a complete, runnable fenced code
   block (Python with imports, or a shell script).
5. Follow the textbook authoring convention (Learning Objectives, Prerequisites,
   Core Content with all sections, Worked Example, Exercises ≥3 per section,
   Summary, Further Reading).
6. For lab .py files: each script must run without errors and include ≥3
   exercises with increasing difficulty.
7. For assessment files: quiz.md needs ≥15 questions, answer_sheet.md has
   full solutions, grading_guide.md has rubric levels (excellent/good/fair/poor).

Definition of done:
${DOD}

Required files:
${FILES}

Write every file in full. Do not leave TODOs or placeholders. When finished,
list all files you created."

info "Task:     ${TASK_ID} — ${TITLE}"
info "Branch:   ${BRANCH}"
info "Log:      ${LOG_FILE}"

# ---------------------------------------------------------------------------
# Print prompt mode: dump prompt and exit
# ---------------------------------------------------------------------------
if (( PRINT_PROMPT )); then
  echo "=== Prompt for ${TASK_ID} ==="
  echo ""
  echo "${PROMPT}"
  echo ""
  echo "=== End of prompt (${#PROMPT} chars) ==="
  exit 0
fi

# ---------------------------------------------------------------------------
# Skip if already passed
# ---------------------------------------------------------------------------
CURRENT_STATUS="$(task_status "$TASK_ID")"
if [[ "$CURRENT_STATUS" == "passed" ]]; then
  info "Task ${TASK_ID} already passed. Use --force to re-run."
  exit 0
fi

# ---------------------------------------------------------------------------
# Create branch
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
git_create_branch "$BRANCH"
info "Created branch ${BRANCH}"

# ---------------------------------------------------------------------------
# Invoke Claude Code
# ---------------------------------------------------------------------------
set_status "$TASK_ID" "running"
info "Invoking Claude Code (this may take several minutes)..."
info "Output → ${LOG_FILE}"

# Run claude; capture output to log.  The process blocks until done.
"${CLAUDE_BIN}" "${CLAUDE_FLAGS[@]}" "${PROMPT}" > "${LOG_FILE}" 2>&1
CLAUDE_EXIT=$?

if [[ ${CLAUDE_EXIT} -ne 0 ]]; then
  err "Claude exited with code ${CLAUDE_EXIT}. Log: ${LOG_FILE}"
  set_status "$TASK_ID" "failed"
  info "Branch ${BRANCH} left for manual inspection."
  exit 1
fi

info "Claude completed successfully."

# ---------------------------------------------------------------------------
# Stage changes
# ---------------------------------------------------------------------------
git add -A
if git diff --cached --quiet 2>/dev/null; then
  err "No changes detected on branch ${BRANCH}. Did Claude create files?"
  set_status "$TASK_ID" "failed"
  exit 1
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
info "Running verification..."
if "${SCRIPT_DIR}/verify_task.sh" "$TASK_ID" 2>&1 | tee -a "${LOG_FILE}"; then
  info "Verification passed for ${TASK_ID}."
  git_commit_task "$TASK_ID"
  info "Committed to ${BRANCH}."

  # -----------------------------------------------------------------------
  # Merge into main
  # -----------------------------------------------------------------------
  info "Merging ${BRANCH} into main..."
  git checkout main >/dev/null 2>&1 || git checkout -b main >/dev/null 2>&1
  if git merge --no-ff "${BRANCH}" -m "merge: ${BRANCH}" >/dev/null 2>&1; then
    info "Merged into main."
  else
    err "Merge conflict on ${BRANCH}. Leaving branch for manual resolution."
    set_status "$TASK_ID" "failed"
    exit 1
  fi

  set_status "$TASK_ID" "passed"
else
  err "Verification failed for ${TASK_ID}. Branch ${BRANCH} left for fixes."
  set_status "$TASK_ID" "failed"
  # Switch back to main so the caller isn't stuck on the bad branch
  git checkout main >/dev/null 2>&1 || true
  exit 1
fi

info "Task ${TASK_ID} complete. Merged into main from ${BRANCH}."
exit 0
