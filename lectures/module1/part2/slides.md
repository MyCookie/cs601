# Module 1, Part 2: Scaling Laws, MoE, and Base Model Training - Lecture Slides

---

## Slide 1: Title
**Course:** CS601 - Advanced LLM Adaptation & Deployment
**Module 1, Part 2:** Scaling Laws, Mixture of Experts (MoE), and Base Model Training
**Goal:** Transition from *how* the Transformer works to *how it scales* and *how it is trained*.

---

## Slide 2: The Core Theme
**Efficiency vs. Performance**
- As models grow, we hit a "Wall."
- Diminishing returns on compute.
- Massive increase in financial and energy costs.
- **The Question:** How do we get the intelligence of a 1T parameter model without the cost of running 1T parameters?

---

## Slide 3: Scaling Laws
**What are Scaling Laws?**
- Empirical observations of model performance.
- **Performance Measure:** Loss (Cross-Entropy Loss).
- **Key Concept: The Power Law**
  - Performance improvement is not linear.
  - To get a linear gain in quality, you need an exponential increase in resources.

---

## Slide 4: The Three Pillars of Scaling
**How we measure "Size"**
1. **Parameter Count ($N$):** The "Capacity." How many weights are in the network?
2. **Dataset Size ($D$):** The "Experience." How many tokens was it trained on?
3. **Compute ($C$):** The "Energy." Total FLOPs (Floating Point Operations) used.

---

## Slide 5: Kaplan vs. Chinchilla
**The Evolution of Scaling Theory**
- **Kaplan (OpenAI, 2020):** Focus on $N$. Bigger models are more sample-efficient. (Led to the "Bigger is Better" era).
- **Chinchilla (DeepMind, 2022):** Focus on $N$ and $D$ equally.
- **The Insight:** Most models were under-trained.
- **Optimal Ratio:** $N \approx 20 \times D$.

---

## Slide 6: Visualizing Scaling
**The Scaling Relationship**
- [Insert Diagram: Scaling Flow]
- Compute Budget $\to$ Balances $N$ and $D$ $\to$ Performance.
- Note the "Diminishing Returns" curve.

---

## Slide 7: Mixture of Experts (MoE)
**The Problem with Dense Models**
- **Dense Model:** Every token activates every parameter.
- **The Bottleneck:** 1T parameters = 1T calculations per token.
- **Result:** High latency, extreme compute cost.

---

## Slide 8: MoE Mechanics
**Introducing Sparsity**
- **Experts:** Specialized sub-networks (FFNs) within the model.
- **The Router:** A gating network that decides which expert gets the token.
- **Top-k Routing:** Only 1 or 2 experts are activated per token.
- **Result:** High Total Parameters, Low Active Parameters.

---

## Slide 9: MoE Workflow
**Token Path in MoE**
- [Insert Diagram: MoE Sequence]
- Token $\to$ Router $\to$ (Top-k Expert) $\to$ Output.
- Only a fraction of the model "wakes up" for any given token.

---

## Slide 10: MoE Case Study: Mixtral 8x7B
**Performance vs. Efficiency**
- Total Parameters: $\approx 47B$ (based on 8 experts of 7B, though shared layers exist).
- Active Parameters: Much lower per token.
- **The Win:** Intelligence of a larger model, inference speed of a smaller model.

---

## Slide 11: Base Model Training
**The Objective: Next-Token Prediction**
- **Self-Supervised Learning:** No human labels needed.
- **Task:** Predict the next token given the previous sequence.
- $P(x_{t+1} | x_1, \dots, x_t)$
- By predicting the next word, the model implicitly learns grammar, facts, and logic.

---

## Slide 12: Pre-training Infrastructure
**The VRAM Wall**
- GPU memory must hold:
  - **Model Weights:** The static parameters.
  - **Gradients:** The "delta" for updates.
  - **Optimizer States:** Memory for Adam/SGD.
- If Model > VRAM $\to$ We must distribute.

---

## Slide 13: Distributed Training
**Scaling across GPUs**
- **Data Parallelism:**
  - Model is copied to all GPUs.
  - Each GPU gets a different batch of data.
- **Model Parallelism:**
  - Model is split (tensor/pipeline) across GPUs.
  - A single token travels through multiple GPUs.

---

## Slide 14: The Cost of Pre-training
**Engineering Challenges**
- **Compute:** Measured in FLOPs.
- **Data Curation:**
  - Deduplication (Remove duplicates).
  - Filtering (Remove low-quality/toxic content).
- **Stability:** Managing "Loss Spikes" and training divergence.

---

## Slide 15: Summary: Dense vs. Sparse
| Feature | Dense | MoE (Sparse) |
| :--- | :--- | :--- |
| **Activation** | All params | Subset of params |
| **Speed** | Slower | Faster |
| **VRAM** | High | Very High |
| **Stability** | Stable | Harder to balance |

---

## Slide 16: Conclusion & Next Steps
- We have seen how scaling laws guide model size.
- We have seen how MoE provides efficiency.
- We have seen how base models are pre-trained on massive data.
- **Next Part:** Data Engineering for LLMs (Module 2).