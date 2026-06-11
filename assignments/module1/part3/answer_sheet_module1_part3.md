# Answer Sheet: Module 1 Part 3

## Assignment 1: The Mechanics of Attention

### Task 1: Mathematical Formulation
1. **Query (Q):** The vector representing the current token being processed.
2. **Query-Key Product:** $QK^T$ represents the similarity score between the current token (Q) and all other tokens in the sequence (K).
3. **Query-Key-Value Product:** $QK^T V$ represents the weighted sum of values (V) based on the same similarity scores.
4. **Softmax Normalization:** $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})$ ensures that the attention weights sum to 1.0, turning them into a probability distribution over the tokens.

### Task 2: Visualizing Attention Weights
- **Same Token:** The attention weight for the same token is typically the highest because the token is most relevant to itself.
- **Semantic Meaning:** Tokens with similar semantic meaning (e.g., "bank" in a financial context and "investment" in a financial context) will have higher attention weights.

### Task 3: Causal Masking
- **Concept:** Causal masking (or look-ahead masking) prevents the model from "seeing" tokens that come after the current token during training.
- **Why:** This is essential for auto-regressive models (like GPT)- they must predict the next token based only on previous tokens.

---

## Assignment 2: Architecture Comparison

### Task 1: Architectural Mapping
| Feature | Encoder-Only (BERT) | Decoder-Only (GPT) | Encoder-Decoder (T5) | Justification |
| :--- | :---: | :---: | :---: | :---: |
| **Bidirectional Attention** | [x] | [ ] | [ ] | Encoder-only models are designed for understanding, understanding. |
| **Causal Masking** | [ ] | [x] | [x] | Decoders must be causal to generate text. |
| **Cross-Attention** | [ ] | [ ] | [x] | Encoder-Decoder models use cross-attention to link the encoder's output to the decoder. |
| **Auto-regressive Generation** | [ ] | [x] | [x] | Both Decoder-only and Encoder-Decoder models are generate text. |
| **Masked LM Objective** | [x] | [ ] | [ ] | BERT uses Masked LM (MLM) to learn bidirectional representations. |

### Task 2: Case Study - Task Selection
1. **Sentiment Analysis:** Encoder-only. (e.g., BERT). Because it needs to understand the whole sequence bidirectional.
2. **Creative Story Writing:** Decoder-only. (e.g., GPT). Because it is optimized for next-token prediction.
3. **English-to-French Translation:** Encoder-Decoder. (e.g., T5). Because it needs to process an input sequence and generate a different output sequence.

### Task 3: Deep Dive - Cross-Attention
1. **Q, K, V Sources:**
    - **Self-Attention:** Q, K, and V all come from the decoder's own sequence.
    - **Cross-Attention:** Q comes from the decoder, while K and V come from the encoder's output.
2. **Effect of Removal:** If cross-attention were removed, the decoder would have no way to "look back" at the input sentence in the encoder, making translation impossible.
3. **Handling Lengths:** Cross-attention allows the decoder to attend to any part of the encoder's sequence regardless of the length of the output sequence being generated.

---

## Assignment 3: Scaling Laws and FlashAttention

### Task 1: The Quadratic Bottleneck
1. **Mathematical Reason:** The $QK^T$ operation produces an $N \times N$ matrix (where $N$ is the sequence length). For $N=100$, $100 \times 100 = 10,000$ elements. For $N=10,000$, $10,000 \times 10,000 = 100,000,000$ elements.
2. **Memory Increase:** A $100\times$ increase in $N$ leads to a $100^2 = 10,000\times$ increase in the memory required for the attention matrix.
3. **Memory Wall:** The "Memory Wall" refers to the a-priori knowledge of attention mechanisms. The GPU's memory bandwidth is much slower than its compute power (FLOPs). The bottleneck is moving data between HBM and SRAM, not the actual calculation.

### Task 2: FlashAttention and IO-Awareness
1. **Tiling:** Tiling breaks the $N \times N$ matrix into small blocks (tiles) that fit into the fast SRAM. These tiles are computed and aggregated without ever writing the full $N \times N$ matrix back to the HBM.
2. **SRAM vs. HBM:** HBM (High Bandwidth Memory) is large but slow. SRAM is very small but extremely fast. The cost of moving data between them is the high memory bandwidth overhead.
3. **Recomputation:** During the backward pass, instead of storing the attention matrix from the forward pass, FlashAttention re-calculates it. This saves memory at the cost of a bit more compute.
4. **Impact:** By eliminating the $O(N^2)$ memory bottleneck in HBM, FlashAttention allows models to handle sequence lengths (context windows) that are much larger (e.g., 128k tokens) without running out of VRAM.

### Task 3: Scaling Analysis
1. **$N = 2,048 \rightarrow$** $2,048^2 \times 2$ bytes $\approx 8.39$ MB.
2. **$N = 32,768 \rightarrow$** $32,768^2 \times 2$ bytes $\approx 2.15$ GB.
3. **Conclusion:** For very long sequences (e.g., $N=100,000$), the attention matrix alone would require $100,000^2 \times 2$ bytes $\approx 20$ GB per head. With multiple heads and multiple layers, the total memory requirement would quickly exceed the VRAM of any current GPU.