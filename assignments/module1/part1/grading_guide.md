# Grading Guide: Module 1, Part 1 (Foundations of Modern LLMs)

**Course:** CS601 - Advanced LLM Adaptation & Deployment  
**Module:** 1, Part 1

---

## 1. Test 1 Grading Rubric (100 Marks Total)

### Section A: Architecture & Design (40 Marks)
**Question 1: Comparative Analysis (20 Marks)**
- **Correct Objectives (6 Marks):** 2 marks each for correctly identifying the goals of Encoder-only (Understanding), Decoder-only (Generation), and Encoder-Decoder (Seq2Seq).
- **Masking Mechanism (6 Marks):** 6 marks if the student correctly explains the causal mask and its purpose in preventing "looking ahead".
- **Scenario (4 Marks):** 4 marks for a valid scenario (e.g., Translation) where Encoder-Decoder is superior.
- **Conceptual Diagram (4 Marks):** 4 marks if the diagram logically shows the flow of data for the three architectures.

**Question 2: Mechanics of Attention (20 Marks)**
- **Q, K, V Definitions (6 Marks):** 2 marks each for correct definitions.
- **Multi-Head Purpose (6 Marks):** 6 marks for explaining that MHA allows attention to different semantic/syntactic features.
- **Scaling Factor $\sqrt{d_k}$ (4 Marks):** 4 marks for explaining the prevention of softmax saturation.
- **Complexity (4 Marks):** 4 marks for correctly identifying $O(N^2)$ complexity.

### Section B: Scaling Laws & MoE (30 Marks)
**Question 3: Scaling Laws (15 Marks)**
- **Compute-Optimal (5 Marks):** 5 marks for explaining the balance of model size and data for a given compute budget.
- **Trade-off (5 Marks):** 5 marks for discussing the scaling of data vs model size.
- **Prediction (5 Marks):** 5 marks for mentioning power-law relationships.

**Question 4: MoE & Routing (15 Marks)**
- **Dense vs Sparse (5 Marks):** 5 marks for explaining that only a subset of parameters is active per token.
- **Router Role (5 Marks):** 5 marks for explaining the gating network's role in mapping tokens to experts.
- **Routing Diagram (5 Marks):** 5 marks for a diagram illustrating Token $\rightarrow$ Router $\rightarrow$ Experts $\rightarrow$ Aggregator.

### Section C: Base Model Training (30 Marks)
**Question 5: Next-Token Prediction (15 Marks)**
- **CLM Objective (5 Marks):** 5 marks for explaining $P(x_t | x_{1...t-1})$.
- **Pre-training vs Fine-tuning (5 Marks):** 5 marks for distinguishing self-supervised pre-training from supervised fine-tuning.
- **Vocab Impact (5 Marks):** 5 marks for discussing the trade-off between sequence length and softmax layer cost.

**Question 6: Compute & Infrastructure (15 Marks)**
- **VRAM/Batch Size (5 Marks):** 5 marks for explaining how VRAM limits dictate max batch size.
- **Memory Usage (5 Marks):** 5 marks for identifying that Optimizer states (Adam) usually dominate memory.
- **Parallelism (5 Marks):** 5 marks for distinguishing Data Parallelism from Model Parallelism.

---

## 2. Quiz 1 Grading Rubric

### Multiple Choice Questions
- **Q1:** B (Encoder-only) - 5 Marks
- **Q2:** B (Prevent look-ahead) - 5 Marks
- **Q4:** B (Prevent saturation) - 5 Marks
- **Q7:** B (CLM) - 5 Marks
- **Q8:** B (Optimizer states) - 5 Marks

### Short Answer / True-False
- **Q3:** $O(N^2)$ time and space. (5 Marks)
- **Q5:** False. Only top-k experts are activated. (5 Marks)
- **Q6:** Router is a gating network. If poorly trained, it leads to expert collapse (all tokens go to one expert). (10 Marks)

---

## 3. Final Grade Calculation
- **Test Score:** Sum of Section A, B, and C.
- **Quiz Score:** Sum of Q1-Q8.
- **Total Part 1 Grade:** $\frac{\text{Test Score} + \text{Quiz Score}}{150} \times 100$