# Grading Guide: Module 1, Part 3

This guide provides the criteria for evaluating the assignments in Module 1, Part 3.

## Assignment 1: The Mechanics of Attention

### Task 1: Manual Attention Trace
- **Full Credit (10/10):** 
    - Correctly calculated $QK^T = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \end{bmatrix}$ (or equivalent).
    - Correctly scaled by $1/\sqrt{2} \approx 0.707$.
    - Correctly applied Softmax: $\text{softmax}([0.707, 0]) \approx [0.67, 0.33]$.
    - Correctly multiplied by $V$: $[0.67, 0.33] \begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix} = [16.7, 26.7]$.
    - All intermediate steps shown clearly.
- **Partial Credit (5/10):** 
    - Correct calculation of $QK^T$, but errors in scaling or softmax.
    - Correct scaling, but errors in softmax or multiplication by $V$.
    - Correct final result, but no intermediate steps shown.
- **No Credit (0/10):** 
    - Calculation is entirely incorrect or missing.

### Task 2: The "Vanishing Gradient" Problem
- **Full Credit (10/10):** 
    - Explains that for large $d_k$, dot products can be very large, leading to softmax saturation (the largest value becomes $\approx 1$ and others $\approx 0$).
    - Correctly identifies the "vanishing gradient" problem: in the saturated regions of softmax, the gradient is nearly zero, making backpropagation ineffective.
    - Technically justifies $\sqrt{d_k}$ scaling: variance of dot product of two independent $N(0, 1)$ vectors is $d_k$. Scaling by $1/\sqrt{d_k}$ brings the variance back to 1.
    - Reference to $e^x$ in the softmax denominator.
- **Partial Credit (5/10):** 
    - Conceptual explanation is correct, but lacks technical depth or mathematical justification for $\sqrt{d_k}$.
- **No Credit (0/10):** 
    - Correctly identifies the "vanishing gradient" problem but cannot explain why.

### Assignment 2: Architecture Comparison and Optimization

### Task 1: The "Optimal Architecture" Matrix
- **Full Credit (10/10):** 
    - **Sentiment Analysis:** Encoder-only (Bi-directional attention; MLM objective; captures context from both sides).
    - **Translation:** Encoder-Decoder (Cross-attention; allows mapping of source sequence to target sequence of different lengths).
    - **Translation (Alt):** Decoder-only (Causal attention; but typically less efficient for translation than Encoder-Decoder).
    - **Storytelling:** Decoder-only (Causal attention; CLM objective; ensures the model doesn't "cheat" by seeing the future).
    - Justifications are technically sound.
- **Partial Credit (5/10):** 
    - Some architecture recommendations are correct, but justifications are lacking or technically incorrect.
- **No Credit (0/10):** 
    - Recommendations are entirely incorrect.

### Task 2: FlashAttention Analysis
- **Full Credit (10/10):** 
    - Explains $O(N^2)$ complexity as the size of the attention matrix $N \times N$.
    - Describes Tiling: Breaking the $Q, K, V$ matrices into blocks (tiles) and computing attention in SRAM own-chip memory, reducing HBM reads/writes.
    - Explains re-computation in the backward pass: instead of storing the large attention matrix for the backward pass, FlashAttention re-computes it from the blocks in SRAM.
    - Correctly identifies FlashAttention as "IO-Aware" because it optimizes the memory movement between HBM and SRAM.
- **Partial Credit (5/10):** 
    - Some concepts are correct, but basic explanation of tiling tiling process is missing.
- **No Credit (0/10):** 
    - Incorrect explanation of memory optimization.

### Assignment 3: Scaling Laws and Compute-Optimal Training

### Task 1: The Scaling Law Relationship
- **Full Credit (10/10):** 
    - Identifies variables: Model size ($N$), Dataset size ($D$), and Compute budget ($C$).
    - Explains diminishing returns: as $N$ increases, the model can memorize the same data $D$ over and over, but the model doesn't learn new patterns from the same data.
    - Explains power law: larger $\alpha$ means that adding parameters is more effective at reducing loss.
    - Technical accuracy is high.
- **Partial Credit (5/10):** 
    - Some explanations are correct, but basic concepts are missing.
- **No Credit (0/10):** 
    - **Incorrect explanations of scaling laws.**

### Task 2: Chinchilla Optimality Calculation
- **Full Credit (10/10):** 
    - Calculation of optimal $N$ and $D$ for $C = 2 \times 10^{22}$ FLOPs.
    - Using $C \approx 6ND$: $2 \times 10^{22} = 6ND$.
    - For Chinchilla optimality, $N \approx D / 20$ (approximate rule of thumb, although the Chinchilla paper suggests a $1:1$ scaling ratio for $N$ and $D$ in terms of power laws).
    - Correctly handles the "Over-training" vs "Under-training" scenario: If $N = 10$ billion, $D = (2 \times 10^{22}) / (6 \times 10^{10}) = 3.33 \times 10^{11}$ tokens.
    - Correctly identifies that if $N$ is smaller than the Chinchilla optimum, the model is "over-trained" (trained on more tokens than compute-optimally required).
    - Correctly handles the "Data Constraints" a case: if $D = 1$ trillion, then $N = (2 \times 10^{22}) / (6 \times 10^{12}) = 3.33 \times 10^9$ parameters.
    - All mathematical derivations are clearly shown.
- **Partial Credit (5/10):** 
    - **Incorrect calculations but correct conceptual understanding.**
- **No Credit (0/10):** 
    - **Incorrect calculations and conceptual understanding.**