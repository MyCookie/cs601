# Test 1: Scaling, MoE, and Training Infrastructure (Module 1, Part 2)

**Student Name:** ____________________  
**Date:** ____________________  
**Total Score:** /100

---

## Section 1: Scaling Laws & Compute (30 Points)

**Q1. (10 pts) The Chinchilla Scaling Rule**
Explain the core intuition behind the Chinchilla scaling rule. If you have a fixed compute budget, why is it often better to train a smaller model on more data rather than a larger model on less data?

**Q2. (10 pts) Compute Calculation**
You are given a compute budget of $C = 5 \times 10^{23}$ FLOPs. 
Using the simplified formula $C \approx 6ND$:
1. If you choose a model with $N = 2 \times 10^9$ (2 Billion) parameters, how many tokens ($D$) can you train on?
2. Is this model "optimally trained" according to the Chinchilla rule ($\text{Tokens} \approx 20 \times \text{Params}$)? Explain.

**Q3. (10 pts) Visualizing Scaling**
Describe in detail a graph that illustrates the relationship between Compute ($C$) and Loss ($L$). 
- What are the X and Y axes?
- What does the shape of the curve represent?
- Where do "diminishing returns" occur on this graph?
- **Requirement:** Provide a Mermaid diagram representation of this relationship.

---

## Section 2: Mixture of Experts (MoE) (35 Points)

**Q4. (10 pts) Sparse vs. Dense Architectures**
Compare a Dense Transformer to a Sparse MoE Transformer. In your explanation, specifically address the "Compute per Token" versus "Total Model Capacity."

**Q5. (15 pts) MoE Parameter Analysis**
Consider an MoE model with:
- Total Experts ($E$): 32
- Top-k ($k$): 2
- Expert Size: 200 Million parameters per expert.
- Shared Parameters: 100 Million.

**Calculate:**
1. Total Parameter Count.
2. Active Parameter Count per token.
3. Sparsity Ratio.

**Q6. (10 pts) Routing Dynamics**
Describe the process of **Top-k Routing**. What happens if the router assigns nearly equal scores to all experts? How does this impact the "Expert Load" across the model?
- **Requirement:** Provide a Mermaid diagram illustrating the flow from Input Token $\rightarrow$ Router $\rightarrow$ Expert Selection $\rightarrow$ Aggregation.

---

## Section 3: Training Infrastructure & VRAM (35 Points)

**Q7. (15 pts) Memory Footprint Estimation**
You are pre-training a model with **10 Billion parameters** using the **Adam Optimizer** in **BF16 precision**.
Calculate the static VRAM required for:
1. Model Weights (2 bytes/param).
2. Gradients (2 bytes/param).
3. Optimizer States (Momentum + Variance, 4 bytes each).
4. **Total Static VRAM** (in GB).

**Q8. (10 pts) Distributed Training Strategies**
Contrast **Data Parallelism (DP)** and **Model Parallelism (MP)**. 
- Which one is used when the model is too large to fit on a single GPU?
- Which one is used primarily to increase the throughput of training by processing more batches in parallel?

**Q9. (10 pts) VRAM Bottlenecks**
Beyond weights and optimizer states, what other factors consume VRAM during training? Describe how **Activation Checkpointing** helps mitigate these costs.
- **Requirement:** Provide a Mermaid diagram comparing the memory layout of a single GPU using Data Parallelism vs. Model Parallelism.

---