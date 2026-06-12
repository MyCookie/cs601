#!/usr/bin/env bash
# Licensed under Apache 2.0
# Verification and grading for a completed task.
#
# Usage:  ./verify_task.sh <TASK_ID> [strict]
#   TASK_ID  : e.g. T01, t15, T48
#   strict   : if set, fail on any warning (optional)
#
# Exit 0 = pass, 1 = fail, 2 = error running the verifier itself.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

TASK_ID="${1:-}"
STRICT="${2:-}"

if [ -z "${TASK_ID}" ]; then
  err "Usage: $0 <TASK_ID> [strict]"; exit 2
fi

TASK_ID="$(to_upper "${TASK_ID}")"  # normalise to uppercase
LOG_FILE="${LOG_DIR}/verify_${TASK_ID}.log"

PASS=0; FAIL=0; WARN=0

pass() { PASS=$((PASS+1)); echo "  ${GREEN}PASS${RESET} $*"; }
fail() { FAIL=$((FAIL+1)); echo "  ${RED}FAIL${RESET} $*"; }
warn() { WARN=$((WARN+1)); echo "  ${YELLOW}WARN${RESET} $*"; }

ensure_log_dir
init_status

# ---------------------------------------------------------------------------
# Determine task phase
# ---------------------------------------------------------------------------
task_phase() {
  local id_num
  id_num=$(echo "$TASK_ID" | grep -oE '[0-9]+')
  if (( id_num >= 1 && id_num <= 14 )); then echo "textbook"
  elif (( id_num >= 15 && id_num <= 19 )); then echo "appendix"
  elif (( id_num >= 20 && id_num <= 33 )); then echo "lecture"
  elif (( id_num >= 34 && id_num <= 47 )); then echo "labs"
  elif (( id_num >= 48 && id_num <= 61 )); then echo "assessment"
  elif (( id_num >= 62 && id_num <= 63 )); then echo "infrastructure"
  else echo "unknown"; fi
}

PHASE="$(task_phase "$TASK_ID")"

# ---------------------------------------------------------------------------
# Check 1: All expected files exist and are non-empty
# ---------------------------------------------------------------------------
check_files_exist() {
  info "Check 1: Expected files exist and are non-empty"
  local files
  files="$(task_files "$TASK_ID")"
  if [ -z "$files" ]; then
    warn "Could not parse expected files for ${TASK_ID}"
    return
  fi
  while IFS= read -r f; do
    # Strip leading whitespace / bullet chars
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    local fp="${REPO_ROOT}/${f}"
    if [ -f "$fp" ] && [ -s "$fp" ]; then
      pass "$f ($(wc -c < "$fp") bytes)"
    else
      fail "$f  (missing or empty)"
    fi
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Check 2: Markdown content quality
# ---------------------------------------------------------------------------
check_markdown_quality() {
  info "Check 2: Markdown content quality"
  local files
  files="$(task_files "$TASK_ID")"
  [ -z "$files" ] && return

  while IFS= read -r f; do
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    [[ "$f" != *.md ]] && continue
    local fp="${REPO_ROOT}/${f}"
    [ ! -f "$fp" ] && continue

    local word_count
    word_count=$(wc -w < "$fp")

    # Phase-specific minimums
    local min_words=0
    case "$PHASE" in
      textbook)       min_words=2000 ;;
      lecture)        if [[ "$f" == *"lecture.md" ]]; then min_words=3500; fi ;;
      slides)         if [[ "$f" == *"slides.md" ]]; then min_words=800; fi ;;
      assessment)
        if [[ "$f" == *"quiz.md" ]]; then min_words=400; fi
        if [[ "$f" == *"assignment.md" ]]; then min_words=600; fi
        if [[ "$f" == *"answer_sheet.md" ]]; then min_words=500; fi
        if [[ "$f" == *"grading_guide.md" ]]; then min_words=300; fi
        ;;
    esac

    if (( min_words > 0 )); then
      if (( word_count >= min_words )); then
        pass "$f word count: ${word_count} >= ${min_words}"
      else
        fail "$f word count: ${word_count} < ${min_words}"
      fi
    else
      pass "$f exists with ${word_count} words"
    fi

    # Check for Apache 2.0 license header
    if head -3 "$fp" | grep -qi "apache 2.0\|licensed under"; then
      pass "$f has license header"
    else
      warn "$f missing license header"
    fi
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Check 3: Diagrams present (mermaid blocks or ASCII art)
# ---------------------------------------------------------------------------
check_diagrams() {
  info "Check 3: Diagrams rendered inline"
  local files
  files="$(task_files "$TASK_ID")"
  [ -z "$files" ] && return

  while IFS= read -r f; do
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    [[ "$f" != *.md ]] && continue
    local fp="${REPO_ROOT}/${f}"
    [ ! -f "$fp" ] && continue

    local mermaid_count
    mermaid_count=$(grep -c '^\`\`\`mermaid' "$fp" 2>/dev/null || echo 0)
    # ASCII diagrams: fenced code blocks that are NOT python/bash/json/yaml
    local code_blocks
    code_blocks=$(grep -c '^\`\`\`' "$fp" 2>/dev/null || echo 0)
    local lang_blocks
    lang_blocks=$(grep -cE '^\`\`\`(python|bash|sh|json|yaml|yml|toml)' "$fp" 2>/dev/null || echo 0)
    local ascii_candidates=$(( (code_blocks - lang_blocks) / 2 ))

    if (( mermaid_count > 0 || ascii_candidates > 0 )); then
      pass "$f has diagrams (mermaid: ${mermaid_count}, ascii blocks: ${ascii_candidates})"
    else
      # Only flag if the task description mentions diagrams
      local block
      block="$(task_block "$TASK_ID")"
      if echo "$block" | grep -qi "diagram\|figure\|mermaid\|ascii"; then
        fail "$f missing diagrams (task requires diagrams)"
      else
        pass "$f no diagrams required"
      fi
    fi
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Check 4: Python syntax
# ---------------------------------------------------------------------------
check_python_syntax() {
  info "Check 4: Python files have valid syntax"
  local files
  files="$(task_files "$TASK_ID")"
  [ -z "$files" ] && return

  while IFS= read -r f; do
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    [[ "$f" != *.py ]] && continue
    local fp="${REPO_ROOT}/${f}"
    [ ! -f "$fp" ] && continue

    if python3 -c "import py_compile; py_compile.compile('${fp}', doraise=True)" 2>/dev/null; then
      pass "$f valid Python syntax"
    else
      fail "$f Python syntax error"
    fi
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Check 5: Textbook-specific — section headers and structure
# ---------------------------------------------------------------------------
check_textbook_structure() {
  info "Check 5: Textbook chapter structure"
  local files
  files="$(task_files "$TASK_ID")"
  [ -z "$files" ] && return

  while IFS= read -r f; do
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    [[ "$f" != chapter_* ]] && continue
    local fp="${REPO_ROOT}/${f}"
    [ ! -f "$fp" ] && continue

    # Check for required sections
    for section in "Learning Objective" "Prerequisite" "Worked Example" "Exercise" "Summary" "Further Reading"; do
      if grep -qi "$section" "$fp"; then
        pass "$f has '${section}' section"
      else
        fail "$f missing '${section}' section"
      fi
    done
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Check 6: Assessment-specific — quiz question count
# ---------------------------------------------------------------------------
check_assessment_structure() {
  info "Check 6: Assessment structure"
  local files
  files="$(task_files "$TASK_ID")"
  [ -z "$files" ] && return

  while IFS= read -r f; do
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    [[ "$f" != *quiz.md ]] && continue
    local fp="${REPO_ROOT}/${f}"
    [ ! -f "$fp" ] && continue

    # Count questions (lines starting with number + period or "### Question")
    local q_count
    q_count=$(grep -cE '^(##|\d+\.|Q\d+)' "$fp" 2>/dev/null || echo 0)
    if (( q_count >= 15 )); then
      pass "$f has ${q_count} questions (>= 15)"
    else
      fail "$f has ${q_count} questions (< 15)"
    fi
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Check 7: Lab-specific — exercise count
# ---------------------------------------------------------------------------
check_lab_exercises() {
  info "Check 7: Lab exercises"
  local files
  files="$(task_files "$TASK_ID")"
  [ -z "$files" ] && return

  while IFS= read -r f; do
    f="$(echo "$f" | sed 's/^[[:space:]`]*//;s/[[:space:]`]*$//')"
    [ -z "$f" ] && continue
    [[ "$f" != *.py ]] && continue
    local fp="${REPO_ROOT}/${f}"
    [ ! -f "$fp" ] && continue

    local ex_count
    ex_count=$(grep -ci "exercise\|# TODO\|# YOUR TURN\|def exercise" "$fp" 2>/dev/null || echo 0)
    if (( ex_count >= 3 )); then
      pass "$f has ${ex_count} exercise markers (>= 3)"
    else
      warn "$f has ${ex_count} exercise markers (< 3)"
    fi
  done <<< "$files"
}

# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=== Verification: ${TASK_ID} ($(task_title "$TASK_ID")) ===${RESET}"
echo ""

check_files_exist
echo ""
check_markdown_quality
echo ""
check_diagrams
echo ""
check_python_syntax
echo ""
if [[ "$PHASE" == "textbook" ]]; then check_textbook_structure; fi
echo ""
if [[ "$PHASE" == "assessment" ]]; then check_assessment_structure; fi
echo ""
if [[ "$PHASE" == "labs" ]]; then check_lab_exercises; fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=== Summary ===${RESET}"
echo -e "  ${GREEN}PASS: ${PASS}${RESET}  ${RED}FAIL: ${FAIL}${RESET}  ${YELLOW}WARN: ${WARN}${RESET}"
echo ""

TOTAL_CHECKS=$((PASS + FAIL))
if (( TOTAL_CHECKS == 0 )); then
  warn "No checks ran — task metadata may be incomplete"
  set_status "$TASK_ID" "failed"
  exit 2
fi

GRADE=$(( (PASS * 100) / TOTAL_CHECKS ))
echo "  Grade: ${GRADE}%"

if (( FAIL == 0 )); then
  echo -e "${GREEN}✓ VERIFIED${RESET}"
  set_status "$TASK_ID" "passed"
  exit 0
elif (( GRADE >= 80 )); then
  if [[ "$STRICT" == "strict" ]]; then
    echo -e "${YELLOW}⚠ PARTIAL (strict mode — warnings are errors)${RESET}"
    set_status "$TASK_ID" "failed"
    exit 1
  fi
  echo -e "${YELLOW}⚠ PARTIAL (>= 80% but has failures)${RESET}"
  set_status "$TASK_ID" "passed"
  exit 0
else
  echo -e "${RED}✗ FAILED (< 80%)${RESET}"
  set_status "$TASK_ID" "failed"
  exit 1
fi
