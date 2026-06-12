# Prompts Used

Prompts from the CS601 project planning session (June 12, 2026). Saved for reproducibility and context when spinning up sub-agent work.

---

## Prompt 1 — Initial Course Setup

> This project is a self-guided graduate-level course on post-training (fine tuning) Large Language Models for enterprise-grade applications. The prerequisite is a Bachelors in Computer Science (no specific specialization). The student is not expected to have any familiarity in Machine Learning before starting this course.
> At the end of the course, students should be able to create an end-to-end project: creating pipelines to ingest data, improving the quality of the dataset, tokenizing the data, performing post-training/fine-tuning, quantizing, then finally deploying the quantized model behind an API. The project will be created on local hardware—such as the DGX Spark—but it should also be deployable in the cloud to create enterprise-grade, production-ready, fine-tuned models for any variety of tasks.
> The course will cover the full stack, including but not limited to: understanding datasets, data quality, data engineering, tokenization, Transformer model architectures, mixture-of-experts, base model training, instruction tuning, quantization, speculative decoding.
> Create a summary of this course in README.md, and include any details you may need to remember in there as well. Create a detailed schedule for this course in SYLLABUS.md.
> Inside a textbook folder, create an outline for a textbook for this course, named OUTLINE.md. Include any elements you will need to create the individual sections later.

**Model:** qwen3.6-27b
**Output:** `README.md`, `SYLLABUS.md`, `textbook/OUTLINE.md`

---

## Prompt 2 — Task Breakdown

> Analyze our goal: Create the full set of learning materials for this course (defined in README, SYLLABUS), including the course textbook (defined in textbook/OUTLINE) as well as any and all lecture slides (and notes), labs, assignments, tests and grading material. Break this down into an ordered list of isolated, independent subtasks. For each subtask, specify the exact files it will touch and its definition of done. Each subtask will be created on a new git branch which will be merged into main when the subtask is finished. Save this entire plan to TASKS.md in the root directory. For each task in TASKS.md make sure to create any diagrams, do not simply describe them. Do not write any material yet.

**Model:** qwen3.6-27b
**Output:** `TASKS.md`, updated `README.md` project layout

---

## Prompt 3 — Save Prompts

> Save the prompts used so far in PROMPTS.md.

**Model:** qwen3.6-27b
**Output:** `PROMPTS.md` (this file)

---

## Prompt 4 — Automation Tooling

> Inside a 'tools' directory, create a set of bash scripts that will automate the execution of each task defined in TASKS.md. Create a test suite to verify and grade the conclusion of each task. Create a test suite within the script(s) to test its execution, and debug errors. For example `claude "Read TASKS.md. Execute task $TASK_NUM completely. Verify your work."` (this is just an example to convey my intent). Since Claude Code will need to work without user input, make sure it has all the permissive options, flags or env vars needed to operate autonomously. For example `clade --allow-dangerously-skip-permissions "Read TASKS.md. Execute task $TASK_NUM completely. Verify your work."`

**Model:** qwen3.6-27b
**Output:** `tools/config.sh`, `tools/run_task.sh`, `tools/run_phase.sh`, `tools/verify_task.sh`, `tools/test_suite.sh`, `tools/status.sh`, `tools/reset.sh`
