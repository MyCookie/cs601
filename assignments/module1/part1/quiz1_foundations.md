# Quiz 1: Foundations of Modern LLMs (Module 1, Part 1)

**Course:** CS601 - Advanced LLM Adaptation & Deployment  
**Format:** Mixed (Multiple Choice & Short Answer)  
**Instructions:** Complete all sections. Show your work for conceptual questions.

---

## Part 1: Transformer Architectures (Quick Check)

**Q1. Which architecture is most suitable for a "fill-in-the-blank" task where the model can see both the left and right context?**
- A) Decoder-only (GPT)
- B) Encoder-only (BERT)
- C) Encoder-Decoder (T5)
- D) Sparse MoE

**Q2. In a Decoder-only model, what is the primary purpose of the causal mask?**
- A) To reduce the memory footprint of the KV cache.
- B) To prevent the model from "looking ahead" at future tokens.
- C) To allow the model to attend to the encoder output.
- D) To speed up the softmax calculation.

---

## Part 2: Attention Mechanics (Conceptual)

**Q3. Given a sequence length $N$ and hidden dimension $d$, what is the time and space complexity of the standard self-attention mechanism?**
- Answer: 

**Q4. Why is the dot product of Q and K divided by $\sqrt{d_k}$?**
- A) To normalize the output to a range between 0 and 1.
- B) To prevent the softmax function from entering regions with extremely small gradients (saturation).
- C) To ensure the attention weights sum to 1.
- D) To reduce the number of parameters in the attention head.

---

## Part 3: Scaling & MoE (Advanced)

**Q5. True or False: In a Sparse MoE model, every token is processed by every expert in the model.**
- Answer: 

**Q6. Describe the "Router" in a Mixture of Experts architecture. What happens if the router is poorly trained?**
- Answer:

---

## Part 4: Base Model Training (Practical)

**Q7. Which of the following is the primary training objective for a base GPT-style model?**
- A) Masked Language Modeling (MLM)
- B) Causal Language Modeling (CLM)
- C) Sequence-to-Sequence Mapping
- D) Contrastive Learning

**Q8. In terms of VRAM, which of these typically consumes the most memory during training?**
- A) Model weights (FP16)
- B) Optimizer states (e.g., Adam)
- C) Input embeddings
- D) Activation caches