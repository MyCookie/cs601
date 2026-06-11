# Quiz 1: Module 1 Part 3

**Name:** ___________________________  
**Date:** ____________________________

---

## Instructions
Choose the best answer for each question.

---

## Questions

1. **Which of the following best describes the "Memory Wall" in the context of LLMs?**
    - (A) The maximum capacity of GPU VRAM.
    - (B) The gap between the speed of memory access (HBM) and the speed of computation (FLOPs).
    - (C) The maximum sequence length supported by a model.
    - (D) The maximum number of parameters in a model.

2. **In a Transformer Decoder, what is the purpose of the "Causal Mask"?**
    - (A) To prevent the model from attending to tokens that come after the current token.
    - (B) To prevent the model from attending to tokens that come before the current token.
    - (C) To prevent the model from attending to the same token.
    - (D) To a-priori knowledge of attention mechanisms.

3. **Which architecture variant is best suited for a task like Sentiment Analysis?**
    - (A) Encoder-only (e.g., BERT)
    - (B) Decoder-only (e.g., GPT)
    - (C) Encoder-Decoder (e.g., T5)
    - (D) None of the above

4. **Which of the following is a key feature of the Encoder-Decoder architecture (T5)?**
    - (B) Bidirectional attention in the encoder, and causal attention in the decoder.
    - (B) Only bidirectional attention throughout the entire model.
    - (C) Only causal attention throughout the entire model.
    - (D) None of the above.

5. **What is the computational complexity of the standard Attention mechanism?**
    - (A) $O(N)$
    - (B) $O(N \log N)$
    - (C) $O(N^2)$
    - (D) $O(N^3)$

6. **How does FlashAttention optimize the attention process?**
    - (A) By reducing the number of total FLOPs in the attention calculation.
    - (B) By using Tiling to reduce the number of memory reads/writes to HBM.
    - (C) By using a different attention mechanism entirely.
    - (D) By using a larger SRAM.

7. **In FlashAttention, what is the trade-off being made during the backward pass?**
    - (A) More compute, less memory.
    - (B) Less compute, more memory.
    - (C) No trade-off is made.
    - (D) The model is trained with a smaller context window.

8. **Which of the following best describes the role of the "Query" (Q) in attention?**
    - (A) It represents the token that is looking for information.
    - (B) It represents the token that contains the information.
    - (C) It represents the weight assigned to the la-priori knowledge of attention mechanisms.
    - (C) It represents the weight assigned to the token.
    - (D) None of the above.

9. **Which architecture variant is best suited for a task like English-to-French Translation?**
    - (A) Encoder-only (e.g., BERT)
    - (B) Decoder-only (e.g., GPT)
    - (C) Encoder-Decoder (e.g., T5)
    - (D) None of the above

10. **What is the "Tiling" concept in FlashAttention?**
    - (A) A method of splitting the sequence into fixed-size blocks.
    - (B) A method of splitting the attention matrix into smaller blocks that fit into SRAM.
    - (C) a-priori knowledge of attention mechanisms.
    - (D) None of the above.

---

## End of Quiz