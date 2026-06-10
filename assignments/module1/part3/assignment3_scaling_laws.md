# Assignment 3: Scaling Laws and Compute-Optimal Training

## Objective
Understand the relationship between model size, dataset size, and training compute, and apply the principles of Chinchilla scaling to determine compute-optimal configurations.

---

## Prerequisites
Before starting this assignment, ensure you have:
- Completed **Assignment 2: Architecture Comparison and Optimization**.
- Read **Lecture Notes for Module 1, Part 3**, specifically the section on Scaling Laws.

---

## 1. Task 1: The Scaling Law Relationship
Based on the scaling laws discussed in the course, answer the following:

1. **The Three Variables:** Identify the three primary variables that govern the final test loss of a Transformer model. How does each variable independently affect performance?
2. **Diminishing Returns:** Explain why increasing the number of parameters ($N$) without increasing the dataset size ($D$) eventually leads to diminishing returns in loss reduction.
3. **The Power Law:** Scaling laws are typically described as power laws. If the loss follows $L(N, D) \propto N^{-\alpha}$, what does a larger $\alpha$ imply about the efficiency of adding parameters to the model?

---

## 2. Task 2: Chinchilla Optimality Calculation
You are given a fixed compute budget of $C = 2 \times 10^{22}$ FLOPs. 

Assume the compute approximation $C \approx 6ND$ (where $N$ is the number of parameters and $D$ is the number of training tokens).

1. **The Chinchilla Principle:** According to the Chinchilla study, for a compute-optimal model, the number of parameters $N$ and the number of tokens $D$ should scale equally. If $N \propto C^{0.5}$ and $D \propto C^{0.5}$, calculate the optimal $N$ and $D$ for your given budget $C$.
2. **Over-training vs. Under-training:** 
    - If you decided to use a model with $N = 10$ billion parameters for this budget, would you be over-training or under-training the model relative to the Chinchilla optimum?
    - Explain the impact of this decision on the final model's inference cost versus its performance.
3. **Data Constraints:** Suppose you only have access to 1 trillion high-quality tokens. If you still want to spend your entire compute budget $C$, how must you adjust your model size $N$? Will the resulting model be "compute-optimal" in the Chinchilla sense? Why or why not?

---

## 3. Submission Guidelines
- Show all calculations for Task 2 clearly.
- For Task 1, provide concise but technically accurate explanations.
- Use LaTeX formatting for all mathematical expressions.