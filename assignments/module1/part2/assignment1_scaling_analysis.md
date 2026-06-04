# Assignment 1: Scaling Law Analysis

## Objective
Apply the principles of Scaling Laws to analyze model performance and determine optimal training configurations.

---

## 1. Concept Review: The Power Law
In the context of LLMs, a **Power Law** describes how loss decreases as resources increase. Unlike a linear relationship (where adding 1 unit of resource always gives the same amount of improvement), a power law means that as the model gets larger, you need exponentially more data or compute to get the same absolute decrease in loss.

**Prerequisite: Cross-Entropy Loss**
Loss is a measure of how "wrong" the model's prediction is compared to the actual next token. A lower loss generally indicates a more accurate model.

---

## 2. Task 1: Chinchilla Optimization
You are tasked with designing a model using a fixed compute budget. You have two options:
- **Option A:** A model with 10 Billion parameters trained on 100 Billion tokens.
- **Option B:** A model with 5 Billion parameters trained on 200 Billion tokens.

**Questions:**
1. Based on the Chinchilla scaling rule ($\text{Tokens} \approx 20 \times \text{Parameters}$), which of these options is closer to being "optimally trained"?
2. Calculate the optimal number of tokens for Option A.
3. Calculate the optimal number of parameters for the token count in Option B.
4. Which option is more likely to exhibit better generalization on a test set, and why?

---

## 3. Task 2: Compute Budgeting
You have a compute budget of $C = 10^{24}$ FLOPs. 
The compute for a transformer is roughly estimated as $C \approx 6ND$ (where $N$ is parameters and $D$ is tokens).

**Exercise:**
If you decide to build a model with 10 Billion parameters ($N = 10^{10}$), how many tokens ($D$) can you afford to train on with your budget?

---

## 4. Task 3: Visual Analysis
Create a sketch or a Mermaid diagram showing the relationship between Compute ($C$) and Loss ($L$). Label the axis and indicate where "Diminishing Returns" begin to occur.

**Expected format for the diagram:**
- X-axis: Compute (Log scale)
- Y-axis: Loss (Log scale)
- Curve: A downward sloping line that flattens over time.