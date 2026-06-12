# CS601 Toolchain

Bash scripts for orchestrating the generation of all 63 CS601 course tasks via Claude Code. Each task runs on an isolated git branch, is verified automatically, and commits on success.

## Quick Start

```bash
# Verify the toolchain itself (43 tests, ~instant)
cd tools && ./test_suite.sh

# Check what's pending
./status.sh all

# Run a single task
./run_task.sh T01

# Run all 14 textbook chapters concurrently (max 5 at a time)
./run_phase.sh textbook

# See progress
./status.sh textbook
```

## Scripts

| Script | What it does |
|---|---|
| `config.sh` | Shared library — never run directly. Sourced by all other scripts. |
| `run_task.sh` | Execute one task: branch → Claude Code → verify → commit. |
| `run_phase.sh` | Execute all tasks in a phase concurrently with a job-limit semaphore. |
| `verify_task.sh` | Grade a completed task against 7 automated checks. |
| `test_suite.sh` | Self-test the toolchain (parsers, status, fixtures, syntax). |
| `status.sh` | Show pass/fail/pending status for all tasks in a phase. |
| `reset.sh` | Reset task status so it can be re-run. |

## Usage

### Run a Single Task

```bash
./run_task.sh T01
```

What happens:
1. Parses `TASKS.md` for the task's metadata (branch name, files, definition of done).
2. Creates a new git branch (`task/t01-ch01-ml-fundamentals`).
3. Invokes Claude Code with `--dangerously-skip-permissions --effort max --disable-slash-commands`.
4. Runs `verify_task.sh` on the created files.
5. If verification passes (≥80%), commits and marks the task as passed.
6. If it fails, leaves the branch for manual inspection and marks the task as failed.

Full logs are written to `.task_logs/run_<TASK_ID>.log`.

```bash
./run_task.sh T15    # any task ID, case-insensitive
./run_task.sh t03    # lowercase also works
```

### Run a Full Phase

```bash
./run_phase.sh textbook 5
```

Runs all tasks in the specified phase concurrently, limiting to 5 simultaneous Claude Code invocations. Phases:

| Phase | Tasks | Count |
|---|---|---|
| `textbook` | T01–T14 | 14 |
| `appendix` | T15–T19 | 5 |
| `lecture` | T20–T33 | 14 |
| `labs` | T34–T47 | 14 |
| `assessment` | T48–T61 | 14 |
| `infrastructure` | T62–T63 | 2 |
| `all` | T01–T63 | 63 |

Already-passed tasks are skipped.

### Check Status

```bash
./status.sh all          # all 63 tasks
./status.sh textbook     # T01–T14 only
```

Symbols: `✓` passed, `◐` running, `✗` failed, `○` pending.

### Verify a Task

```bash
./verify_task.sh T01              # standard mode (pass at ≥80%)
./verify_task.sh T01 strict       # strict mode (fail on any warning)
```

Runs 7 checks:
1. **Files exist** — every expected file is present and non-empty.
2. **Content quality** — phase-specific word count minimums, Apache 2.0 license header.
3. **Diagrams rendered** — Mermaid blocks or ASCII art present when the task requires them.
4. **Python syntax** — `py_compile` validates all `.py` files.
5. **Textbook structure** — required sections (Learning Objectives, Exercises, Summary, etc.).
6. **Assessment structure** — ≥15 quiz questions per `quiz.md`.
7. **Lab exercises** — ≥3 exercise markers per `.py` file.

Exits 0 = pass, 1 = fail, 2 = internal error.

### Reset a Task

```bash
./reset.sh T01              # single task
./reset.sh textbook         # entire phase
./reset.sh                  # all tasks
```

Clears the status and removes log files so the task can be re-run.

### Run the Self-Tests

```bash
./test_suite.sh
```

43 tests across 7 suites: config constants, task parser, phase ranges, status JSON, script syntax, verification on fixtures, and parser edge cases.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_BIN` | `claude` | Path to the Claude Code binary. |

Override for a non-standard installation:

```bash
CLAUDE_BIN=/opt/homebrew/bin/claude ./run_task.sh T01
```

## Execution Order

Tasks within a phase are independent and can run in any order or in parallel. Recommended sequence:

1. **Textbook** (T01–T14)
2. **Appendix** (T15–T19)
3. **Lectures** (T20–T33)
4. **Labs** (T34–T47)
5. **Assessments** (T48–T61)
6. **Infrastructure** (T62–T63, run T63 last)

## State Files

| File | Purpose |
|---|---|
| `.task_status.json` | Tracks pass/fail/running/pending for each task. |
| `.task_logs/` | Per-task stdout/stderr from Claude and the verifier. |

Both are git-ignored. Delete them and start fresh at any time:

```bash
rm -f .task_status.json && rm -rf .task_logs
```

## Troubleshooting

**Claude exits with an error.** Check the log:
```bash
cat .task_logs/run_T01.log
```

**Verification fails.** Re-run in strict mode to see all warnings:
```bash
./verify_task.sh T01 strict
```

**Task stuck on "running".** Reset it and re-run:
```bash
./reset.sh T01 && ./run_task.sh T01
```

**Branch conflict.** `run_task.sh` deletes stale local branches before creating a new one. If a remote branch exists with the same name, rename or delete it manually:
```bash
git push origin --delete task/t01-ch01-ml-fundamentals
```

**macOS `grep` errors.** All scripts use POSIX-compatible `grep` (no `-P` flag). If you see issues, confirm your `grep` supports `-E`:
```bash
grep --version   # should not error
```
