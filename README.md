# CS601: Post-Training of Large Language Models

A self-guided, graduate-level course on fine-tuning Large Language Models (LLMs) for enterprise-grade applications. This course covers the full stack: from data engineering through model deployment, with a capstone project that produces a production-ready, fine-tuned LLM.

## Course Overview

| | |
|---|---|
| **Level** | Graduate (self-guided) |
| **Prerequisites** | Bachelor of Science in Computer Science (no prior machine learning required) |
| **Hardware** | Local GPU workstation (e.g., NVIDIA DGX Spark); cloud-deployable |
| **License** | Apache 2.0 |

## Learning Outcomes

By the end of this course, you will be able to:

- Build end-to-end data pipelines: ingest, clean, de-duplicate, and quality-score datasets for LLM training
- Tokenize text corpora and understand the impact of tokenization on model behavior
- Explain and reason about Transformer architectures, including attention variants and Mixture-of-Experts
- Perform instruction tuning with parameter-efficient methods (LoRA, QLoRA, DoRA)
- Align model outputs via reinforcement learning (DPO, PPO, RLHF)
- Quantize models to INT8/INT4/FP8 without significant quality degradation
- Apply speculative decoding to accelerate inference
- Package, serve, and monitor a fine-tuned model behind a production API

## Capstone Project

You will design, train, quantize, and deploy a fine-tuned LLM for a real-world task. The project spans the entire lifecycle:

1. **Data** - Curate and improve a domain-specific dataset
2. **Tokenize** - Build a vocabulary and tokenize the corpus
3. **Train** - Instruction-tune a base model on your dataset
4. **Quantize** - Compress the model for efficient inference
5. **Deploy** - Serve the quantized model behind an API with monitoring

The project runs on local hardware but is structured to deploy to any cloud provider.

## Course Structure

The course is organized into three parts:

- **Part I - Foundations** (Weeks 1-4): Machine learning fundamentals, linear algebra refresh, Transformers from first principles, scaling laws, and Mixture-of-Experts
- **Part II - Data & Training** (Weeks 5-9): Data engineering, tokenization, base model training concepts, instruction tuning, and alignment
- **Part III - Optimization & Deployment** (Weeks 10-14): Quantization, speculative decoding, API serving, monitoring, and the capstone project

Each module includes lecture notes, hands-on labs, assignments, and a quiz.

## Project Layout

```
cs601/
├── README.md          # This file
├── SYLLABUS.md        # Detailed week-by-week schedule
├── TASKS.md           # Implementation task breakdown
├── textbook/          # Course textbook
│   ├── OUTLINE.md     # Textbook structure and section briefs
│   └── appendix/      # Reference appendices (A–E)
└── modules/           # Per-week learning materials
    └── week_XX_*/     # Lecture notes, slides, labs, assessments
```

## Getting Started

Begin with `SYLLABUS.md` for the detailed course schedule, then work through the textbook chapter by chapter. Each chapter builds on the last -- the prerequisite assumption is a CS degree, not ML experience. Foundational math and statistics are reviewed where needed.
