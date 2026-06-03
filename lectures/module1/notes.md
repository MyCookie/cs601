# Lecture Notes: Module 1 - Foundations of the Modern LLM

## Overview
This module establishes the theoretical and mechanical foundations of Large Language Models (LLMs). We move from the basic concept of how a machine "reads" text to the complex architectures that enable modern AI.

---

## Prerequisites Refresher
*Assume audience knows Basic CS, but may need reminders on:*
- **Linear Algebra:** Vectors, Matrices, and Matrix Multiplication (The core of all Transformer operations).
- **Calculus:** Derivatives and Gradient Descent (How models learn by minimizing error).
- **Probability:** Softmax functions and probability distributions (How models choose the next word).

---

## Chapter 1: Transformer Architectures

### 1.1 Introduction to the Transformer
The Transformer, introduced in "Attention is All You Need" (2017), replaced Recurrent Neural Networks (RNNs) by allowing the model to process all tokens in a sequence simultaneously (parallelization) rather than one by one.

### 1.2 Encoder-only Models (e.g., BERT)
**Concept:** These models "read" the entire sequence and produce a rich representation of each token based on its context (both left and right).
- **Mechanism:** Bi-directional attention.
- **Use Case:** Natural Language Understanding (NLU) tasks like sentiment analysis, named entity recognition, and classification.
- **Key Example:** BERT (Bidirectional Encoder Representations from Transformers).

### 1.3 Decoder-only Models (e.g., GPT)
**Concept:** These models are designed for generation. They read the sequence and predict the *next* token.
- **Mechanism:** Causal (masked) attention. A token can only "attend" to tokens that came before it.
- **Use Case:** Natural Language Generation (NLG), chat assistants, story writing.
- **Key Example:** GPT-4, Llama-3, Mistral.

### 1.4 Encoder-Decoder Models (e.g., T5)
**Concept:** A hybrid approach where an encoder processes the input and a decoder generates the output.
- **Mechanism:** The encoder provides a context vector that the decoder uses to generate a sequence.
- **Use Case:** Translation (English $\rightarrow$ French), Summarization.
- **Key Example:** T5 (Text-to-Text Transfer Transformer).

### 1.5 Comparative Analysis
| Feature | Encoder-only | Decoder-only | Encoder-Decoder |
| :--- | :--- | :--- | :--- |
| **Attention** | Bi-directional | Causal (Masked) | Bi-directional $\rightarrow$ Causal |
| **Primary Goal** | Understanding | Generation | Transformation |
| **Input/Output** | Sequence $\rightarrow$ Embeddings | Sequence $\rightarrow$ Next Token | Sequence $\rightarrow$ Sequence |

---

## Chapter 2: Attention Mechanisms

### 2.1 The Mechanics of Self-Attention
Self-attention allows a model to weigh the importance of different words in a sentence regardless of their distance.

**The QKV Framework:**
For every token, the model generates three vectors via learned linear projections:
1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What information do I contain?"
3. **Value (V):** "What is the actual content I provide?"

**The Process:**
1. **Score:** Compute the dot product of $Q$ and $K^T$. This determines how much the current token relates to others.
2. **Scale:** Divide by $\sqrt{d_k}$ to prevent gradients from becoming too small.
3. **Softmax:** Convert scores into probabilities (summing to 1).
4. **Weight:** Multiply the Softmax result by $V$ to get the final weighted representation.

**Formula:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### 2.2 Multi-Head Attention (MHA)
Instead of one attention pass, the model performs several in parallel ("heads"). Each head can attend to different aspects of the text (e.g., one head focuses on grammar, another on entity relationships).
- **Process:** Run $N$ attention heads $\rightarrow$ Concatenate results $\rightarrow$ Linear projection back to original dimension.

### 2.3 FlashAttention and Computational Efficiency
Standard attention has $O(n^2)$ complexity relative to sequence length $n$. This makes long documents extremely expensive.
- **FlashAttention:** An IO-aware algorithm that optimizes memory access. It avoids writing the large $n \times n$ attention matrix to the slow GPU HBM (High Bandwidth Memory) and instead computes it in blocks within the fast SRAM.
- **Impact:** Significant speed-up and the ability to handle much longer context windows.

### 2.4 Attention Bottlenecks
- **Memory Wall:** The KV Cache (Key-Value cache) grows linearly with sequence length, consuming massive VRAM.
- **Compute Wall:** The quadratic cost of attention calculations.

---

## Chapter 3: Scaling Laws & Mixture of Experts (MoE)

### 3.1 Understanding Scaling Laws
Research (e.g., Kaplan et al.) shows that model performance follows a power-law relationship with three variables:
1. **Compute (C):** Total FLOPs used.
2. **Dataset Size (D):** Number of tokens trained on.
3. **Parameters (N):** Size of the model.
- **Insight:** To improve performance, you must scale all three in tandem. Increasing only parameters without increasing data leads to diminishing returns.

### 3.2 Sparse Activation and Routing (MoE)
Dense models activate every parameter for every token. MoE introduces "Sparsity."
- **Mechanism:** Instead of one large Feed-Forward Network (FFN), the model has multiple "experts" (small FFNs).
- **Router:** A learned gating network that decides which 1 or 2 experts should handle a specific token.
- **Benefit:** You can have a model with 1 Trillion parameters, but only activate 10 Billion per token, keeping inference costs low while maintaining high capacity.

### 3.3 Case Study: Mixtral Architecture
Mixtral is a prime example of a Sparse MoE. It uses a "Router" to send tokens to a subset of experts, allowing it to outperform larger dense models while remaining faster.

---

## Chapter 4: Base Model Training

### 4.1 The Objective: Next-Token Prediction
Base models are trained via **Causal Language Modeling (CLM)**.
- **The Goal:** Given a sequence of tokens $t_1, t_2, ..., t_n$, predict $t_{n+1}$.
- **Loss Function:** Cross-Entropy Loss. The model compares its predicted probability distribution against the actual next token in the training set.

### 4.2 Data Requirements for Pre-training
- **Volume:** Trillions of tokens (Common Crawl, Wikipedia, GitHub, Books).
- **Diversity:** High-quality code, mathematical reasoning, and natural language.
- **Cleaning:** Removing duplicates, boilerplate, and low-quality "web-junk."

### 4.3 Compute Costs and Training Infrastructure
Pre-training is the most expensive phase.
- **Hardware:** Thousands of GPUs (H100s/A100s) connected via NVLink.
- **Costs:** Millions of dollars in electricity and hardware.
- **Parallelism:**
    - **Data Parallelism:** Different GPUs process different batches of data.
    - **Model Parallelism:** The model is split across GPUs because it's too large for one.

### 4.4 Evaluation of Base Models
Base models are not "chatbots"; they are "document completers."
- **Perplexity:** A measure of how well the model predicts the sample. Lower perplexity = better model.
- **Benchmarks:** MMLU (Massive Multitask Language Understanding), HumanEval (for code).