# Grading Guide: Module 1, Part 1

This document provides the evaluation criteria and expected answers for the assignments in Module 1, Part 1.

---

## Assignment 1: Transformer Architecture Analysis

### Task 1: The AI Architect's Dilemma
**Grading Scale:** 2 points per correct selection and justification (Total: 8 points).

| Scenario | Expected Architecture | Key Justification Points |
| :--- | :--- | :--- |
| Medical Entity Extraction | **Encoder-only** | Needs bi-directional context to understand the entity in relation to surrounding text. Not generating new text, just labeling/extracting. |
| Automated Code Doc | **Encoder-Decoder** | Takes one sequence (code) and transforms it into another sequence (natural language). Sequence-to-sequence task. |
| Story Continuation | **Decoder-only** | Causal attention is required to generate tokens one by one. The focus is on predicting the next token based on a prefix. |
| Legal Summarization | **Encoder-Decoder** | Requires encoding a large context (the contract) and decoding a condensed version (the summary). |

**Evaluation:**
- **Full Credit:** Correct architecture + justification referencing Bi-directional/Causal attention.
- **Partial Credit:** Correct architecture but weak/generic justification.
- **No Credit:** Incorrect architecture.

### Task 2: Comparative Analysis Table
**Grading Scale:** 1 point per cell (Total: 16 points).

| Feature | Encoder-only | Decoder-only | Encoder-Decoder |
| :--- | :--- | :--- | :--- |
| **Attention Type** | Bi-directional | Causal (Masked) | Bi-directional (Enc) / Causal (Dec) |
| **Primary Goal** | Understanding/Labeling | Generation | Translation/Transformation |
| **Input/Output Ratio** | 1:1 (approx) | 1:Many | Many:Many |
| **Best Use Case** | Classification/NER | Chatbots/Storytelling | Translation/Summarization |

---

## Assignment 2: The Mechanics of Attention

### Task 1: Manual Attention Trace
**Grading Scale:** 10 points.

**Step-by-Step Key:**
1. **Raw Scores ($Q \cdot K$):**
   - Cloud $\cdot$ Cloud: $(1*1) + (2*2) = 5$
   - Cloud $\cdot$ Compute: $(1*2) + (2*1) = 4$
2. **Scaled Scores ($\text{Score} / 2$):**
   - Cloud: $5 / 2 = 2.5$
   - Compute: $4 / 2 = 2.0$
3. **Softmax Weights:**
   - $e^{2.5} \approx 12.18$
   - $e^{2.0} \approx 7.39$
   - Total $\approx 19.57$
   - Weight (Cloud) $\approx 12.18 / 19.57 \approx 0.62$
   - Weight (Compute) $\approx 7.39 / 19.57 \approx 0.38$
4. **Final Vector:**
   - $(0.62 * [10, 0]) + (0.38 * [0, 20]) = [6.2, 7.6]$

**Evaluation:**
- **Full Credit:** All steps correct.
- **Partial Credit:** Correct logic but arithmetic errors.
- **No Credit:** Incorrect approach to QKV.

### Task 2: Analytical Reflection
**Grading Scale:** 5 points per answer (Total: 10 points).

1. **Orthogonal Q and K:** The dot product is 0. This results in a baseline attention score before softmax. The model effectively "ignores" the token or treats it as unrelated.
2. **Multi-Head Attention:** Allows the model to attend to different types of information (e.g., one head for syntax, one for semantics). This mirrors MoE in that different "heads" become specialists for different feature subspaces.

---

## Assignment 3: Scaling and Sparse Activation

### Task 1: Dense vs. Sparse Analysis
**Grading Scale:** 5 points per answer (Total: 15 points).

1. **Inference Cost:** Model B is 10x more efficient in terms of active FLOPs per token.
2. **Capacity vs. Compute:** Increase the total number of experts. The active parameter count per token remains constant, but the model's overall knowledge capacity increases.
3. **Training Stability:** Expert Collapse. If the router is not regularized, it may favor one expert, causing others to never be updated.

### Task 2: Specialized MoE Router
**Grading Scale:** 10 points.
- **Experts (4 pts):** Logical medical specializations (e.g., Oncology, Pediatrics, etc.).
- **Routing (3 pts):** Correct mapping (e.g., "MRI" $\rightarrow$ Radiology Expert).
- **Conflict Resolution (3 pts):** Mention of weighted distribution or top-k routing.

---

## Assignment 4: Causal Modeling and Prediction

### Task 1: Dataset Design
**Grading Scale:** 2 points per pair (Total: 10 points).
- Correct logic for Prompt $\rightarrow$ Target.
- Plausible distractor tokens.

### Task 2: "Mask-Off" Analysis
**Grading Scale:** 5 points per answer (Total: 15 points).

1. **Training Performance:** Loss drops significantly faster. The model simply "looks ahead" to the answer in the training set.
2. **Inference Failure:** Total failure. The model never learned to predict based on a prefix; it learned to copy from the future. In inference, there is no future.
3. **Cheating vs. Predicting:** Predicting requires learning the underlying distribution; copying is just an identity function mapping.