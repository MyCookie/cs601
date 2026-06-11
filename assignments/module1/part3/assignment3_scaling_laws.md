# Assignment 3: Scaling Laws and FlashAttention

**Name:** ___________________________  
**Date:** ____________________________

---

## Overview
As models grow in size, the computational and memory requirements increase. This assignment explores the "Memory Wall," the quadratic complexity of attention, and how FlashAttention optimizes the process through IO-awareness.

---

## Task 1: The Quadratic Bottleneck

The standard Attention mechanism has a computational and memory complexity of $O(N^2)$, where $N$ is the sequence length.

**Question:**
1. Explain mathematically why the complexity is $O(N^2)$. Which specific operations in the attention formula $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$ contribute to this?
2. If you increase the sequence length from 1,024 tokens to 102,400 tokens (a $100\times$ increase), how much more memory is required for the attention matrix? 
3. What is the "Memory Wall," and why does it become the primary bottleneck for Large Language Models (LLMs) rather than the raw floating-point operations (FLOPs) of the GPU?

**Your Response:**
\
\
\
\
\
\
\
\
\
\
\
\
\
\

---

## Task 2: FlashAttention and IO-Awareness

FlashAttention is described as an "IO-aware" algorithm. Instead of focusing on reducing the number of FLOPs, it focuses on reducing the number of memory reads/writes.

**Questions:**
1. **Tiling:** Describe the concept of "Tiling" in FlashAttention. How does it allow the algorithm to compute attention without writing the large $N \times N$ intermediate matrix back to High Bandwidth Memory (HBM)?
2. **SRAM vs. HBM:** Compare the roles of HBM and SRAM in the context of FlashAttention. Why is moving data between them so expensive?
3. **Recomputation:** FlashAttention avoids storing the large attention matrix for the backward pass by using "recomputation." Explain the trade-off being made here (Compute vs. Memory).
4. **Impact:** In your own words, explain how FlashAttention enables the training of models with significantly longer context windows (e.g., 128k tokens) compared to standard attention.

**Your Response:**
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

---

## Task 3: Scaling Analysis

Assume you have a GPU with 80GB of VRAM. You are using FP16 precision (2 bytes per parameter/activation).

**Scenario:**
You are calculating the memory required to store the attention matrix $A = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})$ for a single head.

1. For $N = 2,048$, how many megabytes (MB) does the attention matrix occupy?
2. For $N = 32,768$, how many gigabytes (GB) does the attention matrix occupy?
3. Based on your answer to #2, why is the standard attention implementation impossible for very long sequences even on high-end GPUs?

**Your Calculations:**
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\