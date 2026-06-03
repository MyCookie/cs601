# Module 1, Part 1: Foundations of the Modern LLM - Lecture Notes

## Introduction
Welcome to the first module of CS601. To build, adapt, and deploy Large Language Models (LLMs), we must first understand the architecture that made them possible: the **Transformer**. 

Before we dive in, it is important to understand that an LLM is essentially a massive mathematical function. It takes text as input, converts it into numbers, performs billions of calculations, and outputs a probability distribution over possible next words.

### Prerequisites Refresher
For those who may be rusty:
- **Neural Network:** A system of interconnected "neurons" (mathematical layers) that learn patterns in data.
- **Linear Algebra:** LLMs operate almost entirely on **Tensors** (multi-dimensional arrays). Most operations are matrix multiplications.
- **Probability:** The model doesn't "know" the answer; it calculates the most likely next token based on the patterns it saw during training.

---

## Chapter 1: Transformer Architectures

The "Transformer" is not a single model, but a design pattern. There are three primary "flavors" based on which parts of the original Transformer architecture (Vaswani et al., 2017) are used.

### 1.1 Encoder-only Models (e.g., BERT)
**The "Reader"**
Encoder-only models are designed to "understand" the relationship between all words in a sequence simultaneously.

- **Bidirectional Processing:** Unlike humans who read left-to-right, BERT looks at the entire sentence at once. This allows it to understand that in the phrase "bank of the river," the word "bank" refers to land, not a financial institution.
- **Training Objective: Masked Language Modeling (MLM):** The model is trained by hiding random words (masking) and forcing the model to predict them using the surrounding context.
- **Best Use Cases:** Sentiment analysis, Named Entity Recognition (NER), and extractive question answering.

### 1.2 Decoder-only Models (e.g., GPT)
**The "Writer"**
Decoder-only models are designed for generation. They are **Autoregressive**, meaning they generate one token at a time and feed that token back into the input for the next step.

- **Causal Language Modeling:** These models are strictly unidirectional. They can only "see" the past. To prevent the model from "cheating" during training by looking at the answer, we use **Causal Masking**.
- **The Next-Token Game:** The sole goal is to predict the next token: $P(x_{t+1} | x_1, \dots, x_t)$.
- **Best Use Cases:** Chatbots, creative writing, and code generation.

### 1.3 Encoder-Decoder Models (e.g., T5)
**The "Translator"**
These models combine both architectures. An Encoder processes the source text into a "meaning vector" (latent space), and a Decoder uses that vector to generate a target sequence.

- **Cross-Attention:** The Decoder doesn't just look at what it has already written; it uses "cross-attention" to look back at the Encoder's output to ensure the generation is grounded in the original input.
- **Text-to-Text Framework:** T5 treats every problem (summarization, translation, classification) as a string-to-string mapping.
- **Best Use Cases:** Machine translation, abstractive summarization, and paraphrasing.

---

## Chapter 2: Attention Mechanisms

Attention is the "secret sauce" of the Transformer. It allows the model to focus on the most relevant parts of the input regardless of their distance.

### 2.1 Self-Attention (The Core)
In a sentence, not all words are equally important. In "The cat sat on the mat because it was tired," the word "it" refers to "cat." Self-attention is the mechanism that allows the model to mathematically link "it" to "cat."

**The QKV Mechanism:**
To calculate attention, the model creates three vectors for every token:
1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I contain?"
3. **Value (V):** "What information do I provide if I am relevant?"

The attention score is calculated as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2.2 Multi-Head Attention (MHA)
One attention head might focus on grammar, another on entity relationships, and another on sentiment. **Multi-Head Attention** runs several attention processes in parallel, allowing the model to capture multiple dimensions of the text.

### 2.3 FlashAttention
Standard attention has a computational complexity of $O(n^2)$, meaning if you double the sequence length, the compute cost quadruples. **FlashAttention** optimizes this by using "tiling" to reduce the number of times data is moved between slow High-Bandwidth Memory (HBM) and fast SRAM on the GPU. This is what allows modern LLMs to have massive context windows (e.g., 128k+ tokens).

---

## Chapter 3: Scaling Laws & Mixture of Experts (MoE)

### 3.1 Scaling Laws
Research (Kaplan et al., 2020) showed that model performance follows a predictable power law relative to three factors:
1. **Parameter Count (N):** The number of weights in the model.
2. **Dataset Size (D):** The number of tokens trained on.
3. **Compute (C):** The total FLOPs used for training.

As you increase these, the "loss" (error) decreases predictably. This led to the "bigger is better" era of LLMs.

### 3.2 Mixture of Experts (MoE)
As models grew, they became too expensive to run. **MoE** solves this by introducing **Sparsity**.

Instead of every token passing through every parameter, an MoE model has several "expert" sub-networks. A **Router** determines which expert is best suited for the current token.
- **Dense Model:** Every token $\to$ All parameters.
- **Sparse Model (MoE):** Every token $\to$ Router $\to$ Top-k Experts.

**Example: Mixtral**
Mixtral uses a MoE architecture where only a fraction of its total parameters are active for any given token, providing the performance of a massive model with the inference speed of a smaller one.

---

## Chapter 4: Base Model Training

### 4.1 The Objective: Next-Token Prediction
Base models are trained using **Self-Supervised Learning**. They don't need labeled data (like "Positive/Negative"); they use the internet itself as the label. The task is simple: *Given this text, predict the next word.*

This objective is powerful because to predict the next word in a medical paper, the model must implicitly learn medicine. To predict the next word in a Python file, it must learn logic and syntax.

### 4.2 Pre-training Costs
Pre-training is the most expensive part of the LLM lifecycle.
- **Compute Cost:** Thousands of GPUs (H100s) running for months.
- **Memory Bottlenecks:** The primary limit is VRAM. Model weights, gradients, and optimizer states must all fit in GPU memory.
- **Energy:** Massive electrical requirements for data center cooling and power.

This is why "Base Models" (Llama, Mistral) are released for the community—so that individual researchers don't have to spend millions of dollars on the initial pre-training phase.

### 4.3 Summary Table: The Big Picture

| Concept | Role | Key Takeaway |
| :--- | :--- | :--- |
| **Transformer** | Architecture | Parallel processing of sequences via Attention. |
| **Encoder** | Understanding | Bidirectional, good for NLU (e.g., BERT). |
| **Decoder** | Generation | Causal, good for NLG (e.g., GPT). |
| **Attention** | Weighting | QKV mechanism to find relevance. |
| **MoE** | Efficiency | Sparse activation using a Router. |
| **Next-Token** | Objective | Self-supervised learning from raw text. |