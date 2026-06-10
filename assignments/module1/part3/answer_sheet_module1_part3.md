# Answer Sheet: Module 1, Part 3

## Assignment 1: The Mechanics of Attention

### Task 1: Manual Attention Trace
1. **Compute Raw Scores ($QK^T$):**
   $\begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}^T = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \end{bmatrix}$
   
   *Wait, checking the matrix multiplication:*
   $Q = [1, 0], K = [[1, 1], [0, 1]]$
   $Q \times K_1 = (1*1) + (0*1) = 1$
   $Q \times K_2 = (1*0) + (0*1) = 0$
   Scores = $[1, 0]$

2. **Scale the Scores:**
   $\sqrt{d_k} = \sqrt{2} \approx 1.414$
   Scaled Scores = $[1/1.414, 0/1.414] \approx [0.707, 0]$

3. **Apply Softmax:**
   $\text{softmax}([0.707, 0]) = \frac{[e^{0.707}, e^0]}{e^{0.707} + e^0} = \frac{[2.028, 1]}{2.028 + 1} = \frac{[2.028, 1]}{3.028} \approx [0.67, 0.33]$

4. **Compute Final Output:**
   $\text{Output} = [0.67, 0.33] \begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix} = [ (0.67*10 + 0.33*30), (0.67*20 + 0.33*40) ]$
   $\text{Output} = [ (6.7 + 9.9), (13.4 + 13.2) ] = [16.6, 26.6]$

**Final Result:** $\begin{bmatrix} 16.6 & 26.6 \end{bmatrix}$

---

## Assignment 2: Architecture Comparison and Optimization

### Task 1: The "Optimal Architecture" Matrix
1. **High-Precision Sentiment Analysis:**
    - **Architecture:** Encoder-only (e.g., BERT).
    - **Justification:** Bi-directional attention allows the model to see the entire sentence simultaneously, which is crucial for detecting subtle nuances and dependencies between words regardless of their position.
2. **Real-time Translation Engine:**
    - **Architecture:** Encoder-Decoder (e.g., T5, BART).
    - **Justification:** This architecture is designed for sequence-to-sequence mapping where the source (English) is fully encoded into a representation, and the target (French) is generated auto-regressively.
3. **Causal Storytelling AI:**
    - **Architecture:** Decoder-only (e.g., GPT series).
    - **Justification:** Causal (masked) attention ensures that the prediction of the next token is based only on previous tokens, preventing "leakage" from the future and enabling generative storytelling.

### Task 2: FlashAttention Analysis
1. **$O(N^2)$ Complexity:** Standard attention computes an $N \times N$ attention matrix. For $N=100\text{k}$, the matrix size is $10^{10}$ elements. If each element is 4 bytes (float32), this requires $\approx 40\text{GB}$ of VRAM just for one head in one layer. This "Memory Wall" occurs because VRAM capacity is much smaller than the total compute power of the GPU.
2. **Tiling:** FlashAttention breaks the large attention matrix into small blocks (tiles) that fit into the GPU's on-chip SRAM. It computes the softmax in blocks and updates the running normalization factor, allowing it to avoid writing the full $N \times N$ matrix to HBM.
3. **Re-computation:** In the backward pass, standard attention stores the $N \times N$ matrix from the forward pass. FlashAttention instead re-computes the necessary attention blocks on-the-fly from the $Q, K, V$ matrices in SRAM, trading off a small amount of extra compute to drastically reduce memory storage requirements.
4. **IO-Aware:** It is "IO-Aware" because its primary goal is to minimize the data movement (reads/writes) between the slow High Bandwidth Memory (HBM) and the fast on-chip SRAM.

---

## Assignment 3: Scaling Laws and Compute-Optimal Training

### Task 1: The Scaling Law Relationship
1. **The Three Variables:** 
    - **Model Size ($N$):** Increasing $N$ generally reduces loss, but with diminishing returns.
    - **Dataset Size ($D$):** Increasing $D$ reduces loss, provided the model has enough capacity to learn the patterns.
    - **Compute Budget ($C$):** The total FLOPs available for training.
2. **Diminishing Returns:** As $N$ increases, the model becomes more expressive. However, if $D$ is constant, the model eventually reaches a limit of what can be learned from that specific dataset. It may start to overfit or simply stop improving as it has already captured all the patterns available in $D$.
3. **The Power Law:** A larger $\alpha$ implies that the model is more "parameter-efficient" - meaning a more significant drop in loss is achieved for every doubling of parameters.

---

## Assignment 2: Chinchilla Optimality Calculation

### Task 2: Chinchilla Optimality Calculation
1. **Optimal $N$ and $D$:**
   Given $C = 2 \times 10^{22}$ FLOPs and $C \approx 6ND$.
   According to Chinchilla, $N$ and $D$ should scale equally.
   $2 \times 10^{22} = 6 \times N \times D$
   If $N = D/20$ (using the standard Chinchilla ratio for $N:D$ as $1:20$),
   $2 \times 10^{22} = 6 \times (D/20) \times D = 0.3 \times D^2$
   $D^2 = (2 \times 10^{22}) / 0.3 \approx 6.67 \times 10^{21}$
   $D \approx \sqrt{6.67 \times 10^{21}} \approx 8.16 \times 10^{10}$ tokens.
   $N = D/20 \approx 4.08 \times 10^9$ parameters.
   
   *Wait, recalculating using the $1:20$ ratio:*
   $C = 6ND$. If $N=k D$, $C = 6kD^2$.
   $2 \times 10^{22} = 6 \times (1/20) \times D^2 \Rightarrow 0.3 D^2 = 2 \times 10^{22} \Rightarrow D^2 = 6.67 \times 10^{22} \Rightarrow D = 8.16 \times 10^{11}$ tokens.
   $N = 4.08 \times 10^{10}$ parameters.

2. **Over-training vs. Under-training:**
   If $N = 10 \times 10^9$ (10 billion), $D = C / (6N) = (2 \times 10^{22}) / (6 \times 10^{10}) \approx 3.33 \times 10^{11}$ tokens.
   Compared to the optimal $N \approx 40$ billion and $D \approx 816$ billion, this model is **under-parameterized** (too small) and is therefore **over-trained** on its available compute budget.
   Impact: The final model will have higher inference cost per token compared to a larger model trained on fewer tokens for the same loss, but it will be more efficient at inference time (smaller model = faster inference). However, it will not reach the minimum possible loss for that compute budget.

3. **Data Constraints:**
   If $D = 1 \times 10^{12}$ (1 trillion tokens), $N = C / (6D) = (2 \times 10^{22}) / (6 \times 10^{12}) \approx 3.33 \times 10^9$ parameters.
   This model is **not compute-optimal** in the Chinchilla sense because the model size $N$ was forced to be smaller than the optimal $N$ for that budget $C$, because we maximized $D$ based on available data. In this case, the model is "over-trained" on the data it has.