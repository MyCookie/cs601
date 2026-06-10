# Lab 3: Scaling Laws and Mixture of Experts (MoE)

## Introduction
In the previous labs, we explored the mathematical foundations and the different architectural "flavors" of Transformers. Now, we move to the "Scale" dimension. In the modern era of LLMs, performance is not just about the architecture, but about how we scale the number of parameters, the amount of data, and the the total compute.

In this lab, you will investigate the relationship between model size, compute budget, and the concept of **Sparse Activation** (MoE), which allows models to be massive in terms of total parameters, but efficient in terms of active parameters during a single forward pass.

---

## 1. The Scaling Laws
The "Chinchilla" scaling laws suggest that for a given compute budget, the most efficient way to increase model performance is to scale the number of parameters ($N$) and the number of training tokens ($D$) in equal proportions.

### Concept: Compute-Optimal Models
A model is "compute-optimal" when it is trained on enough data to reach its maximum potential for its size. Many early models were "under-trained" (too many parameters, not enough data).

**The Scaling Equation (Simplified):**
$$\text{Loss} \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
Where:
- $N$: Number of parameters
- $D$: Number of training tokens
- $\alpha, \beta$: Scaling coefficients

### 🛠️ Task 1: Estimating Compute Budget
You are designing a model with $11$ billion parameters ($N = 11\text{B}$) and you want it to follow the Chinchilla optimality rule (roughly $20$ tokens per parameter).

**Exercise:**
1. Calculate the total number of training tokens ($D$) required for this model to be compute-optimal.
2. If you increase the model size to $70\text{B}$ parameters, how many tokens would you need if you maintain the same token-to-parameter ratio?
3. Why do we say that "more data" is often more important than "more parameters" for models of a certain size?

---

## 2. Mixture of Experts (MoE)
Standard Transformers are **Dense** models. Every single parameter is used for every single token. In a MoE model, we replace the dense Feed-Forward Network (FFN) with a **Sparse MoE layer**.

### Concept: Sparse Activation and Routing
Instead of one giant FFN, we have $E$ "experts" (smaller FFNs). For each token, a **Router** (a small linear layer) decides which expert(s) to send the token to.

**The Routing Logic:**
1. **Input Token:** $\mathbf{x}$
2. **Router Calculation:** $\text{score} = \mathbf{x} W_{router}$
3. **Top-K Selection:** The router selects the top $K$ experts (e.g., $K=2$)
4. **Output:** The final output is a weighted sum of the outputs of the others.

**Visual Representation:**
```text
Input Token -> [ Router ] -> [ Expert 1 ] [ Expert 2 ] [ Expert 3 ] [ Expert 4 ] [ Expert 5 ] [ Expert 6 ] [ Expert 7 ] [ Expert 8 ]
                                  ^              ^
                                  |              |
                                  (Selected Experts)
```

### 🛠️ Task 2: Understanding the Routing Process
Imagine you have an MoE model with $8$ experts, and for a token "Python" (which represents a coding concept), the router produces the following scores:
`[0.1, 0.05, 0.8, 0.1, 0.2, 0.05, 0.1, 0.1]`

**Exercise:**
1. If the model uses **Top-1 Routing**, which expert is selected?
2. If the model uses **Top-2 Routing**, which two experts are selected?
3. In terms of compute, why is Top-2 routing more "expensive" than Top-1 routing, but potentially more accurate?
4. **The Load Balancing Problem:** If the router always picks Expert 3, what happens to the rest of the experts? Why is this a problem for training?

---

## 3. MoE vs. Dense Models: The VRAM Trade-off
The key advantage of MoE is that it allows a massive total parameter count while keeping the **Active Parameters** (the number of parameters used per token) low.

### Concept: Memory vs. Compute
- **Dense Model (7B):**
    - Total Parameters: $7\text{B}$
    - Active Parameters: $7\text{B}$
    - Compute Cost per Token: Proportional to $7\text{B}$
- **MoE Model (Mixtral 8x7B):**
    - Total Parameters: $\sim 47\text{B}$
    - Active Parameters: $\sim 13\text{B}$ (due to shared components and shared experts)
- **VRAM Requirement:** You must load the *entire* model into VRAM to avoid slow disk swapping. Therefore, MoE models require significantly more VRAM than dense models of the same "active" size.

### 🛠️ Task 3: VRAM Planning
You are deploying an MoE model with $47\text{B}$ total parameters.
Assuming $16\text{bit}$ precision (2 bytes per parameter), a dense $7\text{B}$ model would take $14\text{GB}$ of VRAM.

**Exercise:**
1. Calculate the total VRAM needed to load this $47\text{B}$ MoE model in $16\text{bit}$ precision.
2. If you use $4\text{bit}$ quantization (0.5 bytes per parameter), how much VRAM is needed?
3. Compare this to a dense $13\text{B}$ model (which has the same active parameters as the MoE model). How much more VRAM does the MoE model use?

---

## Lab Summary Checklist
- [ ] I can explain the Chinchilla scaling laws and compute-optimality.
- [ ] I understand the concept of Sparse Activation and the Routing mechanism.
- [ ] I can identify the difference between Total Parameters vs. Active Parameters.
- [ ] I can estimate the VRAM requirements for MoE models based on total parameters.