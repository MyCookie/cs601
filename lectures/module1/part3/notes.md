# Module 1, Part 3: Deep Dive into Architectures and Attention - Lecture Notes

## Introduction
In Part 1, we got a high-level overview of the LLM landscape. In Part 2, we looked at the "Macro" view—scaling laws, the efficiency of Mixture of Experts, and the brute-force process of base model training. 

Now, we shift to the "Micro" view. We are opening the hood of the Transformer to understand the actual engine: **Architectures and Attention**. If Scaling Laws tell us how many GPUs we need, the Architecture tells us how those GPUs are actually calculating the next word.

---

## 1. The Mathematical Primer (Prerequisites)
To understand the Transformer, we need to refresh three fundamental concepts from linear algebra and probability. If these feel obscure, here is a quick summary.

### 1.1 Vectors and the Dot Product
A **Vector** is simply a list of numbers. In LLMs, every word (token) is represented as a vector (an embedding).
The **Dot Product** is a way to measure how similar two vectors are. 
- If two vectors point in the same direction, their dot product is high.
- If they are perpendicular, the dot product is zero.
- If they point in opposite directions, it is negative.
**In LLMs:** We use the dot product to determine how much "attention" one word should pay to another.

### 1.2 The Softmax Function
The **Softmax** function takes a list of raw scores (logits) and turns them into probabilities that sum to 1.0.
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
**In LLMs:** After calculating similarity scores between words, we use softmax to decide the percentage of focus to assign to each word.

### 1.3 Matrix Multiplication
Almost every operation in a Transformer is a matrix multiplication. If we have a vector $x$ and a weight matrix $W$, the operation $xW$ transforms the input into a new space.
**In LLMs:** We use these matrices to project tokens into "Query", "Key", and "Value" spaces.

---

## 2. Deep Dive: Transformer Architectures
The original Transformer (Vaswani et al., 2017) had two halves: an Encoder and a Decoder. Modern LLMs have evolved into three distinct "flavors."

### 2.1 Encoder-only Models (The "Understanders")
**Example: BERT (Bidirectional Encoder Representations from Transformers)**

Encoder-only models are designed to read the entire sequence at once. They are **Bidirectional**.

- **How it works:** Every token can "see" every other token in the sentence, both to its left and to its right.
- **Training Objective: Masked Language Modeling (MLM).** Instead of predicting the next word, the model is given a sentence with some words hidden (e.g., "The [MASK] sat on the mat") and must predict the hidden word using context from both sides.
- **Strength:** Exceptional at understanding nuance, sentiment, and relationship extraction.
- **Weakness:** Terrible at generating long-form text.

### 2.2 Decoder-only Models (The "Generators")
**Example: GPT (Generative Pre-trained Transformer), Llama, Mistral**

The vast majority of today's LLMs are decoder-only. They are **Unidirectional** (or Causal).

- **How it works:** A token can only "see" tokens that came before it. This is enforced by **Causal Masking**, which mathematically blocks the model from looking at future tokens during training.
- **Training Objective: Causal Language Modeling (CLM).** The goal is strictly next-token prediction: $P(x_{t+1} | x_1, \dots, x_t)$.
- **Strength:** Fluid, coherent text generation and zero-shot task performance.
- **Weakness:** Slightly less efficient at "understanding" a static block of text compared to a bidirectional encoder.

### 2.3 Encoder-Decoder Models (The "Translators")
**Example: T5 (Text-to-Text Transfer Transformer), BART**

These models use the full original architecture. An encoder processes the input, and a decoder generates the output.

- **How it works:** The encoder creates a rich representation of the source text. The decoder then uses **Cross-Attention** to "peek" at the encoder's representation while generating the target sequence.
- **Strength:** Superior for tasks where the input and output are fundamentally different (e.g., English $\to$ French, or Long Article $\to$ Short Summary).
- **Weakness:** More computationally expensive than decoder-only models for simple generation.

### Architecture Comparison Summary

| Feature | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
| :--- | :--- | :--- | :--- |
| **Direction** | Bidirectional | Unidirectional (Causal) | Both |
| **Core Goal** | Understanding | Generation | Mapping (Seq2Seq) |
| **Key Mechanism**| Self-Attention | Causal Masking | Cross-Attention |
| **Best Use Case** | Sentiment, NER | Chatbots, Coding | Translation, Summary |

---

## 3. Deep Dive: The Attention Mechanism
If the architecture is the "skeleton," Attention is the "brain." It solves the biggest problem in old AI: **Long-term dependency.** (i.e., remembering a subject from the beginning of a paragraph by the time you reach the end).

### 3.1 The Intuition
In the sentence: *"The bank of the river was muddy, so the hiker didn't go to the bank to deposit his check,"* the word "bank" appears twice with two different meanings.
Attention allows the first "bank" to link to "river" and "muddy," and the second "bank" to link to "deposit" and "check."

### 3.2 The QKV Mechanism (The Mathematical Dance)
To achieve this, the model creates three vectors for every token by multiplying the token embedding by three learned weight matrices ($W_Q, W_K, W_V$):

1. **Query (Q):** "What am I looking for?" (The current token's search criteria).
2. **Key (K):** "What do I contain?" (The labels other tokens use to find this token).
3. **Value (V):** "What information do I actually hold?" (The content to be extracted).

#### The Step-by-Step Calculation:
1. **Calculate Similarity:** We take the dot product of the Query of the current word with the Keys of all other words. 
   $$\text{Scores} = Q K^T$$
2. **Scale for Stability:** As the vector dimension ($d_k$) grows, dot products can become massive, pushing the softmax into regions where gradients are tiny (the "vanishing gradient" problem). We divide by $\sqrt{d_k}$.
   $$\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}$$
3. **Normalize to Probabilities:** We apply **Softmax** to get a percentage (e.g., 90% focus on "river", 10% on "muddy").
   $$\text{Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$
4. **Extract Information:** We multiply these weights by the Value vectors.
   $$\text{Attention}(Q, K, V) = \text{Weights} \cdot V$$

### 3.3 Multi-Head Attention (MHA)
Single-head attention can only focus on one "thing" at a time. But language is complex. A word might have a grammatical relationship with one word and a semantic relationship with another.

**Multi-Head Attention** runs the QKV process $N$ times in parallel (heads). 
- Head 1 might focus on syntax (Who is the subject?).
- Head 2 might focus on entities (Where is the location?).
- Head 3 might focus on sentiment (Is this positive or negative?).

The outputs of all heads are concatenated and projected back to the original dimension.

### 3.4 FlashAttention: Solving the Memory Wall
Standard attention has a complexity of $O(n^2)$. For a context window of 100,000 tokens, the attention matrix is 10 billion elements. This creates a **Memory Bottleneck**.

**The Problem:** GPUs have fast memory (**SRAM**) and slow memory (**HBM**). Standard attention moves the massive $Q K^T$ matrix back and forth between them constantly, which is slow.

**The Solution: Tiling.** 
FlashAttention doesn't calculate the whole matrix at once. It breaks the matrix into small blocks (tiles) that fit entirely within the fast SRAM. It computes the softmax and weighted sum in chunks, meaning it only writes to the slow HBM once at the very end.

**Result:** Massive speedups and the ability to handle context windows of 128k, 200k, or even 1M tokens.

---

## Summary Checklist
- [ ] Do I understand why BERT is bidirectional and GPT is causal?
- [ ] Can I explain the difference between a Query, a Key, and a Value?
- [ ] Do I know why we divide by $\sqrt{d_k}$ in the attention formula?
- [ ] Can I describe how Multi-Head Attention allows a model to "see" multiple relationships?
- [ ] Do I understand that FlashAttention is a memory optimization, not a change to the math?