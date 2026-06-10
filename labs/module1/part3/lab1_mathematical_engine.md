# Lab 1: The Mathematical Engine of LLMs

## Introduction
Before we can understand how a Transformer "thinks," we must understand the mathematical language it speaks. Large Language Models (LLMs) do not process words as text; they process them as high-dimensional vectors (lists of numbers). The "intelligence" of the model emerges from how these vectors are multiplied, scaled, and summed.

In this lab, you will implement the three fundamental operations that power every single layer of a modern LLM: the **Dot Product**, the **Softmax Function**, and **Matrix Multiplication**.

---

## 1. Prerequisites: Vectors and Embeddings
**What is a Vector?**
In the context of LLMs, a word (or token) is converted into a vector called an **Embedding**. For example, the word "Apple" might be represented as `[0.12, -0.59, 0.88, ...]`. This vector represents the "meaning" of the word in a multi-dimensional space.

### Concept: The Dot Product
The dot product is the primary way LLMs measure **similarity**. 
If two vectors point in the same direction, their dot product is a large positive number. If they are perpendicular (unrelated), the dot product is zero.

**Formula:** 
For two vectors $\mathbf{a}$ and $\mathbf{b}$ of length $n$:
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

### 🛠️ Task 1: Measuring Token Similarity
Given two hypothetical embeddings for the words "King" and "Queen":
- $\mathbf{v}_{king} = [1.0, 0.5, 0.1]$
- $\mathbf{v}_{queen} = [0.9, 0.6, 0.2]$
- $\mathbf{v}_{apple} = [-0.2, 0.1, 0.8]$

**Exercise:**
1. Calculate the dot product of $\mathbf{v}_{king}$ and $\mathbf{v}_{queen}$.
2. Calculate the dot product of $\mathbf{v}_{king}$ and $\mathbf{v}_{apple}$.
3. Which pair is more "similar"? Why?

---

## 2. The Softmax Function: Turning Scores into Probabilities
The dot product gives us "raw scores" (called **logits**). However, a model needs to make a decision based on probabilities (which must sum to 1.0). The **Softmax** function transforms these raw scores into a probability distribution.

**Formula:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Visualizing Softmax:**
Imagine raw scores: `[2.0, 1.0, 0.1]`
1. **Exponentiate:** $e^{2.0} \approx 7.39$, $e^{1.0} \approx 2.72$, $e^{0.1} \approx 1.11$
2. **Sum:** $7.39 + 2.72 + 1.11 = 11.22$
3. **Normalize:** 
   - $7.39 / 11.22 \approx 0.66$
   - $2.72 / 11.22 \approx 0.24$
   - $1.11 / 11.22 \approx 0.10$
**Result:** `[0.66, 0.24, 0.10]` (These sum to 1.0!)

### 🛠️ Task 2: Normalizing Attention Scores
You have calculated the similarity scores between a target word and three other words in a sentence: `[4.5, 1.2, -0.5]`.

**Exercise:**
Apply the softmax function to these scores. Which word is the model most likely to "attend" to?

---

## 3. Matrix Multiplication: The Projection Engine
While dot products compare two vectors, **Matrix Multiplication** allows the model to transform a vector into a completely different "space." This is how LLMs create the **Queries, Keys, and Values** used in attention.

If you have an input vector $\mathbf{x}$ and a weight matrix $W$, the operation $\mathbf{y} = \mathbf{x}W$ creates a new vector $\mathbf{y}$ that represents the input from a different perspective.

**Visual Representation:**
```text
Input Vector (1x3)    Weight Matrix (3x2)    Output Vector (1x2)
[ x1, x2, x3 ]   X   [ w11, w12 ]   =   [ y1, y2 ]
                     [ w21, w22 ]
                     [ w31, w32 ]
```

### 🛠️ Task 3: Projecting a Token
You have a token embedding $\mathbf{x} = [1, 0, 1]$ and a projection matrix $W$ that transforms tokens into "Query" space:
$$W = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.8 \\ 0.3 & 0.4 \end{bmatrix}$$

**Exercise:**
Calculate the resulting Query vector $\mathbf{q} = \mathbf{x}W$.

---

## Lab Summary Checklist
- [ ] I can calculate a dot product and explain how it relates to similarity.
- [ ] I can apply the softmax function to turn raw logits into probabilities.
- [ ] I can perform matrix-vector multiplication to project a token into a new space.