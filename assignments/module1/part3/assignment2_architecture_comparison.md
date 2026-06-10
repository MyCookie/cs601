# Assignment 2: Architecture Comparison and Optimization

## Objective
Analyze the trade-offs between Encoder-only, Decoder-only, and Encoder-Decoder architectures, and evaluate the impact of memory optimizations like FlashAttention.

---

## Prerequisites
Before starting this assignment, ensure you have:
- Completed **Assignment 1: The Mechanics of Attention**.
- Read **Lecture Notes for Module 1, Part 3**, specifically the sections on Transformer Flavors and FlashAttention.

---

## 1. Task 1: The "Optimal Architecture" Matrix
You are a consultant for an AI company. You must recommend an architecture for three different product requirements. For each, select the architecture (Encoder-only, Decoder-only, or Encoder-Decoder) and justify your choice based on the **attention mechanism** (Bi-directional vs Causal) and the **training objective** (MLM vs CLM).

1. **High-Precision Sentiment Analysis:** A system that must analyze the same sentence multiple times to determine if a subtle nuance in a word choice indicates a positive or negative sentiment.
    - **Architecture:** 
    - **Justification:** 
2. **Real-time Translation Engine:** A system that takes English text and produces a corresponding French translation in real-time.
    - **Architecture:** 
    - **Justification:** 
3. **Causal Storytelling AI:** A tool that creates an expanding story based on a user's prompt, where the next word must logically follow from all previous words.
    - **Architecture:** 
    - **Justification:** 

---

## 2. Task 2: FlashAttention Analysis
The lecture notes describe FlashAttention as a "memory optimization, not a change to the math."

**Questions:**
1. Why does standard attention have a complexity of $O(N^2)$ and why does this lead to a "Memory Wall" for long context windows (e.g., 100k tokens)?
2. Describe the "Tiling" process in FlashAttention. Explain how it reduces the number of read/write operations between High Bandwidth Memory (HBM) and on-chip SRAM.
3. How does the use of a "re-computation" strategy in the backward pass of FlashAttention trade off compute for memory?
4. In your own words, explain why FlashAttention is considered "IO-Aware."

---

## 3. Submission Guidelines
- Submit your architectural recommendations in a table or list format.
- For Task 2, include a diagram or a detailed step-by-step description of the tiling process.
- Ensure your explanation of the $O(N^2)$ complexity specifically mentions the size of the attention matrix.