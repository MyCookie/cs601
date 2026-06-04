# Assignment 2: Mixture of Experts (MoE) Design

## Objective
Design a sparse MoE architecture and calculate the relationship between total parameters and active parameters.

---

## 1. Concept Review: Sparsity and Routing
In a standard "dense" model, every input token passes through every single parameter in the model. In a **Sparse MoE (Mixture of Experts)** model, the network is divided into multiple "expert" sub-networks. A **Router** determines which few experts are best suited to handle a specific token.

**Prerequisite: Top-k Routing**
Top-k routing is the process where the router calculates a score for every expert and selects the $k$ experts with the highest scores. For example, if $k=2$, only 2 experts are activated per token, regardless of how many total experts exist in the model.

---

## 2. Task 1: Parameter Calculation
You are designing an MoE model with the following specifications:
- **Number of Experts ($E$):** 16
- **Top-k ($k$):** 2
- **Expert Size:** 100 Million parameters per expert.
- **Shared Parameters (Non-expert):** 50 Million parameters (e.g., embedding layer, final output layer).

**Questions:**
1. What is the **Total Parameter Count** of this model?
2. What is the **Active Parameter Count** per token?
3. What is the **Sparsity Ratio** (Active Parameters / Total Parameters)?
4. If you increase $k$ to 4, how does the Active Parameter Count change? Does the Total Parameter Count change?

---

## 3. Task 2: Routing Logic
Imagine a router that uses a simple linear layer to produce scores for experts.
- Token A produces scores: `[0.1, 0.8, 0.2, 0.4, 0.1, ...]`
- Token B produces scores: `[0.4, 0.3, 0.5, 0.2, 0.1, ...]`

**Exercise:**
Using **Top-2 routing**, which experts are selected for Token A and Token B? Show your work.

---

## 4. Task 3: Architectural Diagram
Create a Mermaid diagram representing the flow of a token through an MoE layer. 

**Your diagram should include:**
- Input Token $\rightarrow$ Router $\rightarrow$ Selection of Top-k Experts $\rightarrow$ Expert Computation $\rightarrow$ Weighted Sum $\rightarrow$ Output.