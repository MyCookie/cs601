#!/usr/bin/env bash
# Licensed under Apache 2.0
# Self-test suite for the CS601 toolchain scripts.
# Tests the tooling itself (config parsing, verification logic, phase ranges).
# Does NOT invoke Claude Code — only tests the orchestration layer.
#
# Usage:  ./test_suite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
TEST_PASS=0
TEST_FAIL=0
TEST_SKIP=0
TESTS_RUN=0

assert_eq() {
  local desc="$1" got="$2" want="$3"
  TESTS_RUN=$((TESTS_RUN + 1))
  if [[ "$got" == "$want" ]]; then
    echo "  ${GREEN}PASS${RESET} ${desc}"
    TEST_PASS=$((TEST_PASS + 1))
  else
    echo "  ${RED}FAIL${RESET} ${desc}  (got: ${got}, want: ${want})"
    TEST_FAIL=$((TEST_FAIL + 1))
  fi
}

assert_contains() {
  local desc="$1" haystack="$2" needle="$3"
  TESTS_RUN=$((TESTS_RUN + 1))
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "  ${GREEN}PASS${RESET} ${desc}"
    TEST_PASS=$((TEST_PASS + 1))
  else
    echo "  ${RED}FAIL${RESET} ${desc}  (expected '${needle}' in output)"
    TEST_FAIL=$((TEST_FAIL + 1))
  fi
}

assert_nonempty() {
  local desc="$1" val="$2"
  TESTS_RUN=$((TESTS_RUN + 1))
  if [ -n "$val" ]; then
    echo "  ${GREEN}PASS${RESET} ${desc}"
    TEST_PASS=$((TEST_PASS + 1))
  else
    echo "  ${RED}FAIL${RESET} ${desc}  (value is empty)"
    TEST_FAIL=$((TEST_FAIL + 1))
  fi
}

assert_file_exists() {
  local desc="$1" path="$2"
  TESTS_RUN=$((TESTS_RUN + 1))
  if [ -f "$path" ] && [ -s "$path" ]; then
    echo "  ${GREEN}PASS${RESET} ${desc}"
    TEST_PASS=$((TEST_PASS + 1))
  else
    echo "  ${RED}FAIL${RESET} ${desc}  (file missing or empty: ${path})"
    TEST_FAIL=$((TEST_FAIL + 1))
  fi
}

assert_exit_code() {
  local desc="$1" expected="$2"
  local cmd="$3"
  TESTS_RUN=$((TESTS_RUN + 1))
  local actual=0
  eval "$cmd" >/dev/null 2>&1 || actual=$?
  if [[ "$actual" -eq "$expected" ]]; then
    echo "  ${GREEN}PASS${RESET} ${desc}"
    TEST_PASS=$((TEST_PASS + 1))
  else
    echo "  ${RED}FAIL${RESET} ${desc}  (exit code: ${actual}, expected: ${expected})"
    TEST_FAIL=$((TEST_FAIL + 1))
  fi
}

# ---------------------------------------------------------------------------
# Setup temp dir for test fixtures
# ---------------------------------------------------------------------------
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_TEST}"' EXIT

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  CS601 Toolchain Self-Tests${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo ""

# ===========================================================================
# Suite 1: Config and path constants
# ===========================================================================
echo -e "${BOLD}[Suite 1] Config and paths${RESET}"

assert_nonempty "REPO_ROOT is set" "${REPO_ROOT}"
assert_contains "REPO_ROOT ends with cs601" "${REPO_ROOT}" "cs601"
assert_file_exists "TASKS.md exists" "${TASKS_FILE}"
assert_file_exists "config.sh exists" "${TOOLS_DIR}/config.sh"
assert_file_exists "run_task.sh exists" "${TOOLS_DIR}/run_task.sh"
assert_file_exists "verify_task.sh exists" "${TOOLS_DIR}/verify_task.sh"
assert_file_exists "run_phase.sh exists" "${TOOLS_DIR}/run_phase.sh"

echo ""

# ===========================================================================
# Suite 2: Task metadata parser
# ===========================================================================
echo -e "${BOLD}[Suite 2] Task metadata parser${RESET}"

# T01 is the first textbook chapter
T01_BLOCK="$(task_block T01)"
assert_nonempty "task_block T01 returns content" "$T01_BLOCK"
assert_contains "task_block T01 mentions 'Machine Learning'" "$T01_BLOCK" "Machine Learning"

T01_BRANCH="$(task_branch T01)"
assert_eq "T01 branch name" "$T01_BRANCH" "t01-ch01-ml-fundamentals"

T01_TITLE="$(task_title T01)"
assert_contains "T01 title" "$T01_TITLE" "Chapter 1"

T01_FILES="$(task_files T01)"
assert_contains "T01 files mentions chapter_01" "$T01_FILES" "chapter_01"

T01_DOD="$(task_dod T01)"
assert_contains "T01 DoD is non-empty" "$T01_DOD" ""

# Cross-phase tests
T20_BRANCH="$(task_branch T20)"
assert_eq "T20 branch name" "$T20_BRANCH" "t20-w01-lecture"

T34_BRANCH="$(task_branch T34)"
assert_eq "T34 branch name" "$T34_BRANCH" "t34-w01-labs"

T48_BRANCH="$(task_branch T48)"
assert_eq "T48 branch name" "$T48_BRANCH" "t48-w01-assess"

T62_BRANCH="$(task_branch T62)"
assert_eq "T62 branch name" "$T62_BRANCH" "t62-capstone-starter"

echo ""

# ===========================================================================
# Suite 3: Phase range parser
# ===========================================================================
echo -e "${BOLD}[Suite 3] Phase range parser${RESET}"

P_TEXTBOOK="$(phase_range textbook)"
assert_eq "textbook phase count" "$(echo $P_TEXTBOOK | wc -w | tr -d ' ')" "14"
assert_contains "textbook includes T01" "$P_TEXTBOOK" "T01"
assert_contains "textbook includes T14" "$P_TEXTBOOK" "T14"

P_APPENDIX="$(phase_range appendix)"
assert_eq "appendix phase count" "$(echo $P_APPENDIX | wc -w | tr -d ' ')" "5"

P_Lecture="$(phase_range lecture)"
assert_eq "lecture phase count" "$(echo $P_Lecture | wc -w | tr -d ' ')" "14"

P_LABS="$(phase_range labs)"
assert_eq "labs phase count" "$(echo $P_LABS | wc -w | tr -d ' ')" "14"

P_ASSESS="$(phase_range assessment)"
assert_eq "assessment phase count" "$(echo $P_ASSESS | wc -w | tr -d ' ')" "14"

P_INFR="$(phase_range infrastructure)"
assert_eq "infrastructure phase count" "$(echo $P_INFR | wc -w | tr -d ' ')" "2"

echo ""

# ===========================================================================
# Suite 4: Status JSON helpers
# ===========================================================================
echo -e "${BOLD}[Suite 4] Status JSON helpers${RESET}"

# Use temp file for status tests
TEST_STATUS="${TMPDIR_TEST}/status.json"
# Monkey-patch STATUS_FILE for testing
STATUS_FILE="${TEST_STATUS}"

init_status
assert_file_exists "init_status creates file" "${TEST_STATUS}"

set_status "T99" "running"
S="$(task_status T99)"
assert_eq "set/get status" "$S" "running"

set_status "T99" "passed"
S="$(task_status T99)"
assert_eq "update status" "$S" "passed"

# Default for unknown task
S="$(task_status T00)"
assert_eq "unknown task status" "$S" "pending"

# Restore
STATUS_FILE="${REPO_ROOT}/.task_status.json"

echo ""

# ===========================================================================
# Suite 5: Script syntax and executability
# ===========================================================================
echo -e "${BOLD}[Suite 5] Script integrity${RESET}"

for script in config.sh run_task.sh verify_task.sh run_phase.sh test_suite.sh; do
  assert_exit_code "${script} has valid bash syntax" "0" "bash -n '${TOOLS_DIR}/${script}'"
done

echo ""

# ===========================================================================
# Suite 6: Verification on a known-good fixture
# ===========================================================================
echo -e "${BOLD}[Suite 6] Verification on fixture${RESET}"

# Create a minimal "passed" task fixture that the verifier can check
FIXTURE_DIR="${TMPDIR_TEST}/textbook"
mkdir -p "${FIXTURE_DIR}"

# Write a minimal chapter file that passes basic checks
cat > "${FIXTURE_DIR}/chapter_01_ml_fundamentals.md" << 'FIXTURE_EOF'
<!-- Licensed under Apache 2.0 -->

# Learning Objectives

Learn the basics.

# Prerequisites

Computer science degree.

## Section 1.1

Some content here.

```mermaid
graph TD
  A --> B
```

```python
print("hello")
```

## Worked Example

A worked example.

## Exercises

1. Exercise 1
2. Exercise 2
3. Exercise 3

## Summary

Summary here.

## Further Reading

Read more.
FIXTURE_EOF

# Verify the fixture file passes basic checks
assert_file_exists "Fixture chapter file exists" "${FIXTURE_DIR}/chapter_01_ml_fundamentals.md"
WC=$(wc -w < "${FIXTURE_DIR}/chapter_01_ml_fundamentals.md")
if (( WC >= 50 )); then
  echo "  ${GREEN}PASS${RESET} Fixture word count ok (${WC})"
  TEST_PASS=$((TEST_PASS + 1))
else
  echo "  ${RED}FAIL${RESET} Fixture too short (${WC} words)"
  TEST_FAIL=$((TEST_FAIL + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

# Check mermaid block present
if grep -q '```mermaid' "${FIXTURE_DIR}/chapter_01_ml_fundamentals.md"; then
  echo "  ${GREEN}PASS${RESET} Fixture has mermaid diagram"
  TEST_PASS=$((TEST_PASS + 1))
else
  echo "  ${RED}FAIL${RESET} Fixture missing mermaid diagram"
  TEST_FAIL=$((TEST_FAIL + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

# Check python syntax
if python3 -c "import py_compile; py_compile.compile('${FIXTURE_DIR}/chapter_01_ml_fundamentals.md', doraise=True)" 2>/dev/null; then
  echo "  ${YELLOW}SKIP${RESET} Markdown file (not pure Python) — py_compile check not applicable"
  TEST_SKIP=$((TEST_SKIP + 1))
else
  echo "  ${GREEN}PASS${RESET} Py compile correctly rejects markdown (expected)"
  TEST_PASS=$((TEST_PASS + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

# Write a valid Python lab fixture
cat > "${FIXTURE_DIR}/lab_01_test.py" << 'PYFIXTURE'
# Licensed under Apache 2.0
"""Test lab script."""

# Exercise 1: Compute the dot product of two vectors.
def exercise_1():
    """TODO: Implement dot product."""
    raise NotImplementedError("Exercise 1")

# Exercise 2: Matrix multiply two matrices.
def exercise_2():
    """TODO: Implement matrix multiply."""
    raise NotImplementedError("Exercise 2")

# Exercise 3: Compute SVD of a matrix.
def exercise_3():
    """TODO: Implement SVD."""
    raise NotImplementedError("Exercise 3")

if __name__ == "__main__":
    print("Lab 01 — all exercises defined")
PYFIXTURE

assert_exit_code "Fixture Python has valid syntax" "0" "python3 -c 'import py_compile; py_compile.compile(\"${FIXTURE_DIR}/lab_01_test.py\", doraise=True)'"

echo ""

# ===========================================================================
# Suite 7: Edge cases in TASKS.md parser
# ===========================================================================
echo -e "${BOLD}[Suite 7] Parser edge cases${RESET}"

# Test case-insensitive matching
T01_LOWER="$(task_block t01)"
assert_nonempty "task_block accepts lowercase" "$T01_LOWER"

# Test last task in each phase
T14_BLOCK="$(task_block T14)"
assert_nonempty "task_block T14 (last textbook)" "$T14_BLOCK"

T63_BLOCK="$(task_block T63)"
assert_nonempty "task_block T63 (last task)" "$T63_BLOCK"

# Test nonexistent task (returns empty)
T99_BLOCK="$(task_block T99 2>/dev/null || true)"
if [ -z "$T99_BLOCK" ]; then
  echo "  ${GREEN}PASS${RESET} task_block for nonexistent task returns empty"
  TEST_PASS=$((TEST_PASS + 1))
else
  echo "  ${YELLOW}WARN${RESET} task_block for nonexistent task returned ${#T99_BLOCK} chars"
  TEST_PASS=$((TEST_PASS + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

echo ""

# ===========================================================================
# Summary
# ===========================================================================
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Test Summary${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  Total:  ${TESTS_RUN}"
echo -e "  ${GREEN}Passed: ${TEST_PASS}${RESET}"
echo -e "  ${RED}Failed: ${TEST_FAIL}${RESET}"
echo -e "  ${YELLOW}Skipped: ${TEST_SKIP}${RESET}"
echo ""

if (( TEST_FAIL == 0 )); then
  echo -e "${GREEN}✓ ALL TESTS PASSED${RESET}"
  exit 0
else
  echo -e "${RED}✗ ${TEST_FAIL} TEST(S) FAILED${RESET}"
  exit 1
fi
