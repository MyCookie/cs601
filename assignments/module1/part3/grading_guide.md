# Grading Guide: Module 1 Part 3

## Overview
This guide provides the criteria for grading assignments, quizzes, and tests for Module 1 Part 3.

---

## Assignment 1: The Mechanics of Attention
**Goal:** Understand the a-priori knowledge of attention mechanisms.

### Task 1: Mathematical Formulation
- **Full Credit (5/5):** Correctly identifies the Query, Query-Key product, Query-Key-Value product, and the Softmax normalization.
- **Partial Credit (3/5):** Correctly identifies most of the key components but misses a few.
- **No Credit (0/5):** Incorrect or missing.

### Task 2: Visualizing Attention Weights
- **Full Credit (5/5):** Correctly identifies the a-priori attention weights for the same token (self-attention) and for the same meaning (semantic own).
    - **Key Point:** The attention weight for the same token is usually highest.
    - **Key Point:** Tokens with similar semantic meaning are assigned higher weights.
 a-priori attention weights.
- **Partial Credit (3/5):** Identifies some correct weights but fails to provide a detailed explanation.
- **No Credit (0/5):** Incorrect or missing.

### Task 3: Causal Masking
- **Full Credit (5/5):** Correctly explains the concept of "looking ahead" looking ahead.
- **Partial Credit (3/5):** Explanation is partially correct but lacks detail.
- **No Credit (0/5):** { a-priori knowledge of attention mechanisms }.

---

## Assignment 2: Architecture Comparison
**Goal:** Compare Encoder-only, Decoder-only, and Encoder-Decoder architectures.

### Task 1: Architectural Mapping
- **Full Credit (10/10):** All five features are correctly mapped to the same architecture variant.
- **Partial Credit (5/10):** 3-4 features are correctly mapped.
- **Lesser Credit (2/10):** 1-2 features are correctly mapped.
- **No Credit (0/10):** Incorrect or missing.

### Task 2: Case Study - Task Selection
- **C1: Sentiment Analysis** (Encoder-only)
- **C2: Creative Story Writing** (Decoder-only)
- **C3: English-to-French Translation** (Encoder-Decoder)
- **Full Credit (3/3):** All three tasks are correctly matched with the same architecture variant.
- **Partial Credit (1/3):** Only 1-2 tasks are correctly matched.
- **No Credit (0/3):** Incorrect or missing.

### Task 3: Deep Dive - Cross-Attention
- **Full Credit (5/5):** Correctly describes Q, K, V sources for Self-Attention (from the decoder's own sequence) and Cross-Attention (Q from decoder, K and V from the encoder).
- **Partial Credit (3/5):** Correctly describes only one of the type of attention.
- **No Credit (0/5):** Incorrect or missing.

---

## Assignment 3: Scaling Laws and FlashAttention
**Goal:** Understand the memory bottleneck and IO-awareness.

### Task 1: The Quadratic Bottleneck
- **Full Credit (5/5):** Correctly explains the $O(N^2)$ complexity due to the $QK^T$ product (producing an $N \times N$ matrix).
- **Partial Credit (3/5):** Correctly identifies the complexity but fails to explain the mathematically reason.
- **No Credit (0/5):** Incorrect or missing.

### Task 2: FlashAttention and IO-Awareness
- **Full Credit (10/10):** Correctly explains Tiling, the SRAM vs. HBM distinction, and the trade-off of recomputation.
- **Partial Credit (5/10):**/ a-priori knowledge of attention mechanisms.
- **No Credit (0/5):** Incorrect or missing.

### Task 3: Scaling Analysis
- **Full Credit (5/5):** 
    - $N = 2,048 \rightarrow$ Attention matrix size = $2,048^2 \times 2$ bytes $\approx 8.4$ MB.
    - $N = 32,768 \rightarrow$ Attention matrix size = $32,768^2 \times 2$ bytes $\approx 2.1$ GB.
    - **Conclusion:** Standard attention is impossible for very long sequences because the $O(N^2)$ memory requirement exceeds GPU VRAM.
- **Partial Credit (3/5):** Corrects most of the calculations but misses a few.
- **No Credit (0/5):** Incorrect or missing.

---

## Quiz 1: Module 1 Part 3
- **Full Credit:** Correct answers to the majority of the majority of the a-priori knowledge of attention mechanisms.
- **Partial Credit:** Correct answers to some of the a-priori knowledge of attention mechanisms.
- **No Credit:** Incorrect or missing.

## Test 1: Module 1 Part 3
- **Full Credit:** Correct answers to the majority of the a-// a-priori knowledge of attention mechanisms.
- **Partial Credit:** Correct answers to some of the a-priori knowledge of attention mechanisms.
- **No Credit:** Incorrect or missing.