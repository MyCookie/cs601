# Grading Guide: Module 1, Part 2

This guide provides the expected answers and grading criteria for the assignments in Module 1, Part 2.

---

## Assignment 1: Scaling Law Analysis

### Task 1: Chinchilla Optimization
1. **Which option is closer to "optimally trained"?**
   - **Answer:** Option B.
   - **Reasoning:** Chinchilla says $\text{Tokens} \approx 20 \times \text{Params}$.
     - Option A: $10\text{B} \times 20 = 200\text{B}$ tokens. (It only has 100B, so it is under-trained).
     - Option B: $5\text{B} \times 20 = 100\text{B}$ tokens. (It has 200B, so it is over-trained/compute-heavy).
     - Comparison: Option B is closer to the optimal ratio for its size, although it exceeds it.
2. **Optimal tokens for Option A:** $10\text{B} \times 20 = 200\text{ Billion tokens}$.
3. **Optimal params for Option B's token count (200B):** $200\text{B} / 20 = 10\text{ Billion parameters}$.
4. **Generalization:** Option B is more likely to generalize better because it has seen significantly more data relative to its model capacity (reducing the risk of overfitting).

### Task 2: Compute Budgeting
- **Formula:** $C = 6ND$
- **Calculation:** $10^{24} = 6 \times 10^{10} \times D$
- $D = 10^{24} / (6 \times 10^{10}) \approx 1.66 \times 10^{13}$ tokens.
- **Answer:** $\approx 16.6$ Trillion tokens.

### Task 3: Visual Analysis
- **Criteria:**
  - X-axis correctly labeled as Compute (log).
  - Y-axis correctly labeled as Loss (log).
  - Curve shows a power-law decay (downward sloping, flattening).
  - "Diminishing Returns" identified at the flatten point.

---

## Assignment 2: Mixture of Experts (MoE) Design

### Task 1: Parameter Calculation
1. **Total Parameter Count:**
   - $16 \text{ experts} \times 100\text{M} + 50\text{M shared} = 1.6\text{B} + 50\text{M} = 1.65\text{ Billion parameters}$.
2. **Active Parameter Count:**
   - $2 \text{ active experts} \times 100\text{M} + 50\text{M shared} = 200\text{M} + 50\text{M} = 250\text{ Million parameters}$.
3. **Sparsity Ratio:**
   - $250\text{M} / 1.65\text{B} \approx 15.1\%$.
4. **Increase $k$ to 4:**
   - Active Params: $4 \times 100\text{M} + 50\text{M} = 450\text{M}$. (Increases).
   - Total Params: No change (still $1.65\text{B}$).

### Task 2: Routing Logic
- **Token A:** Top 2 scores are $0.8$ and $0.4$. $\rightarrow$ **Experts 2 and 4**.
- **Token B:** Top 2 scores are $0.5$ and $0.4$. $\rightarrow$ **Experts 3 and 1**.

### Task 3: Architectural Diagram
- **Criteria:** Logical flow from Token $\rightarrow$ Router $\rightarrow$ Top-k Selection $\rightarrow$ Expert Compute $\rightarrow$ Aggregation $\rightarrow$ Output.

---

## Assignment 3: Training Infrastructure

### Task 1: VRAM Estimation
1. **Weights:** $7\text{B} \times 2\text{ bytes} = 14\text{ GB}$
2. **Gradients:** $7\text{B} \times 2\text{ bytes} = 14\text{ GB}$
3. **Optimizer States:** $7\text{B} \times 2 \text{ states} \times 4\text{ bytes} = 56\text{ GB}$
- **Total Static Memory:** $14 + 14 + 56 = 84\text{ GB}$.

### Task 2: Distributed Strategy
1. **Data Parallelism (DP):** No. 95 GB required > 80 GB available. The model cannot fit in one GPU's memory.
2. **Proposed Strategy:** Model Parallelism (Tensor Parallelism). Split the layers/tensors across multiple GPUs (e.g., across 2 or 4 GPUs).
3. **VRAM Reduction:** By splitting the weights and optimizer states across multiple GPUs, each GPU only holds a fraction (e.g., 1/2 or 1/4) of the total model state, bringing the requirements below 80 GB.

### Task 3: Infrastructure Diagram
- **Criteria:**
  - DP: Shows multiple GPUs, each with a full copy of the model.
  - MP: Shows multiple GPUs, each with a unique slice of the model.