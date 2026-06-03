# Answer Sheet: Module 1, Part 1 (Foundations of Modern LLMs)

---

## Test 1: Foundations of Modern LLMs

### Section A: Architecture & Design

**Question 1: Comparative Analysis of Transformers**
- **Encoder-only (BERT):** Bi-directional context. Primary objective: Understanding/Classification (e.g., Sentiment Analysis).
- **Decoder-only (GPT):** Uni-directional (Causal) context. Primary objective: Generation (Next-token prediction).
- **Encoder-Decoder (T5):** Bi-directional encoder, Causal decoder. Primary objective: Sequence-to-Sequence tasks (e.g., Translation).
- **Masking:** Decoder-only models use a causal mask (triangular matrix) to set attention scores of future tokens to $-\infty$, ensuring the model only attends to previous positions.
- **Scenario:** Translation. The encoder processes the source sentence in its entirety (bi-directional), and the decoder generates the target sentence token-by-token.

**Question 2: The Mechanics of Attention**
- **Q, K, V:** 
    - **Query (Q):** What I am looking for.
    - **Key (K):** What I contain.
    - **Value (V):** The actual information to be extracted.
- **Multi-Head Attention:** Allows the model to attend to different parts of the sequence simultaneously (e.g., one head for syntax, one for semantics).
- **Formula:** $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- **Scaling Factor $\sqrt{d_k}$:** Prevents the dot product from growing too large in high dimensions, which would push the softmax into regions with vanishing gradients.
- **Complexity:** $O(N^2)$ time and space complexity relative to sequence length $N$.

### Section B: Scaling Laws & MoE

**Question 3: Scaling Laws**
- **Compute-Optimal:** The point where the model size and training data are balanced to achieve the lowest loss for a given compute budget.
- **Trade-off:** Increasing data is more beneficial for smaller models; increasing model size is more beneficial for larger datasets.
- **Prediction:** Scaling laws provide power-law relationships between Loss, Model Size, and Data, allowing researchers to predict performance based on small-scale runs.

**Question 4: MoE & Routing**
- **Dense vs Sparse:** In dense models, every parameter is used for every token. In sparse MoE, only a subset of parameters (experts) is activated per token.
- **Router:** A learned gating network that maps input tokens to the top-k experts.
- **Routing Flow:** Token $\rightarrow$ Gating Network $\rightarrow$ Selection of Top-K Experts $\rightarrow$ Processing by Experts $\rightarrow$ Weighted Sum of Outputs.

### Section C: Base Model Training

**Question 5: Next-Token Prediction**
- **CLM:** Predicts $P(x_t | x_1, ..., x_{t-1})$.
- **Pre-training vs Fine-tuning:** Pre-training uses a self-supervised objective on massive unlabeled data. Fine-tuning uses supervised learning (Instruction Tuning) on curated pairs.
- **Vocabulary Size:** Larger vocab reduces sequence length but increases the cost of the final linear layer and softmax.

**Question 6: Compute and Infrastructure**
- **VRAM Limits:** The amount of GPU memory. Dictates the max batch size and sequence length before an "Out of Memory" (OOM) error occurs.
- **Memory usage:** Optimizer states (e.g., Adam) usually consume the most memory (storing two moments per parameter), followed by gradients and weights.
- **Parallelism:** 
    - **Data Parallelism:** Same model on multiple GPUs, different data.
    - **Model Parallelism:** Model split across GPUs (Tensor Parallelism or Pipeline Parallelism).