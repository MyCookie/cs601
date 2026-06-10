# Module 1, Part 3: Deep Dive into Architectures and Attention - Lecture Slides

## Slide 1: Title Slide
**Title:** Module 1, Part 3: The Engine of the LLM
**Subtitle:** Deep Dive into Transformer Architectures and Attention Mechanisms
**Presenter Notes:** 
- Welcome back. In Part 1 we saw the "What", in Part 2 we saw the "How Much" (Scaling). Today we see the "How".
- Goal: Move from intuition to the actual mathematical mechanics.

---

## Slide 2: Roadmap
**Content:**
- **The Math Primer:** Vectors, Dot Products, and Softmax.
- **Architectural Flavors:** Encoder-only, Decoder-only, and Encoder-Decoder.
- **The Attention Mechanism:** The QKV Dance.
- **Scaling the Engine:** Multi-Head Attention and FlashAttention.

---

## Slide 3: Math Primer - The Dot Product
**Content:**
- **What is it?** A measure of similarity between two vectors.
- **Visual:** [Image/Diagram of two vectors with a small angle (High dot product) vs large angle (Low dot product)]
- **Formula:** $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$
- **LLM Context:** If Word A's Query is similar to Word B's Key $\rightarrow$ High Attention.

---

## Slide 4: Math Primer - Softmax
**Content:**
- **Purpose:** Converts raw scores $\rightarrow$ Probabilities.
- **Key Properties:**
  - All outputs are between 0 and 1.
  - Sum of all outputs = 1.0.
- **Visual:** [Diagram showing raw logits (e.g., [2.0, 1.0, 0.1]) $\rightarrow$ Softmax $\rightarrow$ [0.7, 0.2, 0.1]]
- **LLM Context:** This is how the model decides "I will give 70% of my focus to this word."

---

## Slide 5: Architectural Flavors - Encoder-Only
**Title:** Encoder-Only (The "Understanders")
**Example:** BERT
**Content:**
- **Bidirectional:** Every token sees every other token.
- **Training:** Masked Language Modeling (MLM) - "Fill in the blank."
- **Best for:** Sentiment Analysis, Named Entity Recognition (NER), Classification.
- **Presenter Notes:** Explain that BERT is like reading a whole sentence and then answering a question about it.

---

## Slide 6: Architectural Flavors - Decoder-Only
**Title:** Decoder-Only (The "Generators")
**Example:** GPT, Llama, Mistral
**Content:**
- **Unidirectional (Causal):** Tokens only see the past.
- **Training:** Causal Language Modeling (CLM) - "Predict the next word."
- **Best for:** Chatbots, Creative Writing, Coding.
- **Presenter Notes:** Mention "Causal Masking" - the mathematical blindfold that prevents the model from cheating by looking ahead.

---

## Slide 7: Architectural Flavors - Encoder-Decoder
**Title:** Encoder-Decoder (The "Translators")
**Example:** T5, BART
**Content:**
- **Hybrid:** Encoder reads the source; Decoder generates the target.
- **Mechanism:** Cross-Attention (Decoder peeks at Encoder).
- **Best for:** Translation, Summarization.
- **Presenter Notes:** This is the original 2017 design. It's like a translator listening to a sentence in English (Encoder) and then speaking it in French (Decoder).

---

## Slide 8: The Attention Mechanism - Intuition
**Title:** Why Attention?
**Content:**
- **The Problem:** Long-term dependency.
- **Example:** *"The bank of the river was muddy... hiker went to the bank to deposit a check."*
- **The Solution:** Dynamic weighting.
- **Visual:** [Diagram showing "bank" connecting to "river" in the first instance and "deposit" in the second].

---

## Slide 9: The QKV Dance - The Components
**Title:** Query, Key, and Value
**Content:**
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I hold?"
- **Process:** $Token \times W_Q = Q$, $Token \times W_K = K$, $Token \times W_V = V$.
- **Presenter Notes:** Use an analogy: Q is a search term, K is the page title, V is the page content.

---

## Slide 10: The Attention Formula
**Title:** The Mathematical Step-by-Step
**Content:**
1. **Score:** $QK^T$ (Dot product similarity)
2. **Scale:** $\frac{QK^T}{\sqrt{d_k}}$ (Prevents vanishing gradients)
3. **Normalize:** $\text{softmax}(\dots)$ (Turn scores into percentages)
4. **Extract:** $\text{softmax}(\dots)V$ (Weighted sum of values)
- **Formula:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

---

## Slide 11: Multi-Head Attention (MHA)
**Title:** Seeing the World in Parallel
**Content:**
- **The Limitation:** A single head can only focus on one relationship.
- **The Solution:** Run $N$ attention heads simultaneously.
- **Division of Labor:**
  - Head 1: Syntax (Grammar)
  - Head 2: Semantics (Meaning)
  - Head 3: Coreference (Who is "he"?)
- **Final Step:** Concatenate all heads $\rightarrow$ Project back.

---

## Slide 12: The Memory Wall & FlashAttention
**Title:** FlashAttention: Breaking the Bottleneck
**Content:**
- **The Bottleneck:** $O(n^2)$ complexity. Moving massive matrices between HBM (slow) and SRAM (fast).
- **The Solution: Tiling.**
  - Break the matrix into blocks.
  - Compute everything in SRAM.
  - Only write the final result to HBM.
- **Visual:** [Comparison diagram: Standard Attention (Many HBM reads/writes) vs FlashAttention (Few HBM reads/writes)].
- **Result:** Faster speeds, longer context windows (128k+).

---

## Slide 13: Summary & Review
**Content:**
- **BERT** $\rightarrow$ Bidirectional $\rightarrow$ Understanding.
- **GPT** $\rightarrow$ Causal $\rightarrow$ Generation.
- **T5** $\rightarrow$ Seq2Seq $\rightarrow$ Mapping.
- **Attention** $\rightarrow$ $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
- **MHA** $\rightarrow$ Parallel relationships.
- **FlashAttention** $\rightarrow$ Memory efficiency (SRAM Tiling).