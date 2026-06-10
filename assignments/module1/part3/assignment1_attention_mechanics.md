# Assignment 1: The Mechanics of Attention

## Objective
Develop a deep mathematical and conceptual understanding of the Scaled Dot-Product Attention mechanism and Multi-Head Attention (MHA).

---

## Prerequisites
Before starting this assignment, ensure you have:
- Completed **Lab 1: Transformer Architecture Mapping**.
- Read **Lecture Notes for Module 1, Part 3**.
- Reviewed the **Mathematical Primer** (Vectors, Dot Products, and Softmax).

---

## 1. Task 1: Manual Attention Trace
You are given a simplified scenario with a small embedding dimension $d_k = 2$. 

**Inputs:**
- **Query (Q):** $\begin{bmatrix} 1 & 0 \end{bmatrix}$
- **Keys (K):** $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ (Two tokens)
- **Values (V):** $\begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix}$

**Your Task:**
Calculate the final Attention output step-by-step. Show all your work.

1. **Compute Raw Scores ($QK^T$):** Calculate the dot product of $Q$ with each row of $K$.
2. **Scale the Scores:** Divide the scores by $\sqrt{d_k}$ (where $d_k = 2$).
3. **Apply Softmax:** Convert the scaled scores into probabilities.
4. **Compute Final Output:** Multiply the softmax weights by the Value matrix $V$.

**Final Result:** $\begin{bmatrix} ? & ? \end{bmatrix}$

---

## 2. Task 2: The "Vanishing Gradient" Problem
In the lecture notes, we discussed the scaling factor $\frac{1}{\sqrt{d_k}}$. 

**Questions:**
1. If the embedding dimension $d_k$ is very large (e.g., $d_k = 1024$), what happens to the magnitude of the dot product $QK^T$ if the values in $Q$ and $K$ are independently sampled from a normal distribution with mean 0 and variance 1?
2. Why does a very large dot product cause the Softmax function to "saturate"?
3. How does this saturation affect the gradients during backpropagation (the "vanishing gradient" problem)?
4. Provide a technical justification for why $\sqrt{d_k}$ is the mathematically appropriate scaling factor to keep the variance of the dot product equal to 1.

---

## 3. Task 3: Multi-Head Attention (MHA) Design
Imagine you are designing a Transformer for a highly specialized task: **Legal Document Analysis**. The model must simultaneously track:
- **Syntactic Structure:** (Who is the actor? Who is the object?)
- **Legal Precedence:** (Does this clause refer to a previously mentioned statute?)
- **Sentiment/Tone:** (Is this clause restrictive or permissive?)

**Your Task:**
1. Propose the number of attention heads ($h$) you would use and explain why.
2. Describe what you expect "Head 1," "Head 2," and "Head 3" to specialize in based on the requirements above.
3. If the final model dimension is $d_{model} = 512$, what would be the dimension of the Query, Key, and Value vectors ($d_k, d_v$) for each head? Explain the formula used to derive this.

---

## 4. Submission Guidelines
- Submit your mathematical derivations in a Markdown file using LaTeX for equations.
- For Task 3, provide a brief architectural justification for your head count and dimensions.
- Ensure your explanation of the vanishing gradient problem references the properties of the $e^x$ function in the Softmax denominator.