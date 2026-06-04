# Assignment 3: Training Infrastructure & Resource Planning

## Objective
Estimate the hardware requirements for pre-training a large-scale model and plan a distributed training strategy to overcome memory limitations.

---

## 1. Concept Review: The Memory Footprint
When training an LLM, the GPU memory (VRAM) is consumed by more than just the model weights. To update the weights, the system must keep track of the gradients and the optimizer's internal state.

**Prerequisite: Adam Optimizer States**
The Adam optimizer is the most common choice for LLM training. It maintains two buffers for every single parameter in the model:
1. **Momentum (First Moment):** A moving average of gradients.
2. **Variance (Second Moment):** A moving average of squared gradients.
These are typically stored in FP32 (4 bytes per value) to maintain precision, regardless of whether the model weights are in BF16.

---

## 2. Task 1: VRAM Estimation
You are planning to train a model with **7 Billion parameters** using the **Adam Optimizer** in **BF16 precision**.

**Calculate the memory required for the following (in GB):**
1. **Model Weights:** (7B parameters $\times$ 2 bytes)
2. **Gradients:** (7B parameters $\times$ 2 bytes)
3. **Optimizer States:** (7B parameters $\times$ 2 states $\times$ 4 bytes)

**Total Static Memory:** Sum the results above.

---

## 3. Task 2: Distributed Strategy
Your hardware consists of **8 GPUs, each with 80 GB of VRAM**. 
You discover that including **Activations** and **KV Cache**, the total memory requirement per GPU for a single replica of the model is **95 GB**.

**Questions:**
1. Can you use **Data Parallelism (DP)** (where the model is replicated on every GPU)? Why or why not?
2. Propose a **Model Parallelism (MP)** strategy (e.g., Tensor Parallelism or Pipeline Parallelism) to fit this model across your 8 GPUs.
3. How does your proposed strategy reduce the VRAM pressure on each individual GPU?

---

## 4. Task 3: Infrastructure Diagram
Create a Mermaid diagram showing the difference between **Data Parallelism** (replicating the model) and **Model Parallelism** (splitting the model).

**Your diagram should illustrate:**
- **Data Parallelism:** Multiple GPUs $\rightarrow$ Each has Full Model $\rightarrow$ Different Data batches.
- **Model Parallelism:** Multiple GPUs $\rightarrow$ Each has a Slice of the Model $\rightarrow$ Same Data batch.