# Lecture Slides: Module 1 - Foundations of the Modern LLM

---

## Slide 1: Welcome to CS601
**Title: Foundations of the Modern LLM**

- **Objective:** Understand the "engine" inside an LLM.
- **What we are covering today:**
    - How Transformers work.
    - Why "Attention" is the secret sauce.
    - How we scale models to trillions of parameters (MoE).
    - How a base model is trained from scratch.

**[Diagram Suggestion: A high-level evolution timeline showing RNN $\rightarrow$ Transformer $\rightarrow$ GPT $\rightarrow$ MoE]**

---

## Slide 2: Prerequisites Refresher
**Title: The Tools We'll Use**

- **Linear Algebra:** We treat words as vectors (lists of numbers).
- **Matrix Multiplication:** How the model "combines" information.
- **Softmax:** Turning raw scores into probabilities.
- **Gradient Descent:** The process of adjusting weights to reduce error.

**[Summary Box: "Think of a Vector as a coordinate in a high-dimensional space where similar words are placed close together."]**

---

## Slide 3: The Transformer Revolution
**Title: Why Transformers?**

- **Before:** RNNs processed text linearly (one word at a time).
- **The Problem:** "Forgetting" the beginning of a long sentence; slow training.
- **The Solution:** The Transformer (2017) processes the *entire* sequence at once.
- **Result:** Massive parallelization $\rightarrow$ Faster training $\rightarrow$ Larger models.

---

## Slide 4: Architecture 1 - Encoder-only
**Title: Understanding (The BERT Style)**

- **Goal:** Natural Language Understanding (NLU).
- **Key Feature:** Bi-directional context.
- **How it works:** Reads the whole sentence. Each token "sees" everything to its left and right.
- **Best for:** Sentiment analysis, classification.

**[Diagram Suggestion: A sentence where arrows point both ways between all words]**

---

## Slide 5: Architecture 2 - Decoder-only
**Title: Generating (The GPT Style)**

- **Goal:** Natural Language Generation (NLG).
- **Key Feature:** Causal (Masked) Attention.
- **How it works:** Reads the sequence and predicts the *next* token.
- **Constraint:** Tokens can only "see" what came before them.
- **Best for:** Chatbots, storytelling, code completion.

**[Diagram Suggestion: A sentence where arrows only point from left to right]**

---

## Slide 6: Architecture 3 - Encoder-Decoder
**Title: Transforming (The T5 Style)**

- **Goal:** Sequence-to-Sequence transformation.
- **How it works:**
    1. **Encoder:** Understands the input (Bi-directional).
    2. **Decoder:** Generates the output (Causal).
- **Best for:** Translation (Eng $\rightarrow$ Spa), Summarization.

**[Diagram Suggestion: Two separate blocks (Encoder and Decoder) connected by a "Context Vector" arrow]**

---

## Slide 7: The Magic of Attention
**Title: What is Self-Attention?**

- **Concept:** Not all words are equally important.
- **Example:** "The animal didn't cross the street because **it** was too tired."
- **Question:** What does "**it**" refer to?
- **Attention's Job:** Weigh the connection between "it" and "animal" more heavily than "it" and "street".

---

## Slide 8: The QKV Framework
**Title: Query, Key, and Value**

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What information do I have?"
- **Value (V):** "The actual content."

**The Mathematical Dance:**
1. $\text{Score} = Q \cdot K$ (How well do they match?)
2. $\text{Probability} = \text{softmax}(\text{Score} / \sqrt{d_k})$
3. $\text{Output} = \text{Probability} \cdot V$

**[Diagram Suggestion: A table showing Q, K, and V vectors for three different words, with a dot-product operation showing the matching score]**

---

## Slide 9: Multi-Head Attention (MHA)
**Title: Looking at the World in Different Ways**

- **The Idea:** One "attention pass" isn't enough.
- **Multi-Head:** The model runs several attention processes (Heads) in parallel.
- **Example:** 
    - Head 1: Focuses on Subject-Verb agreement.
    - Head 2: Focuses on Entity relationships.
    - Head 3: Focuses on Punctuation/Structure.
- **Result:** A much richer understanding of the text.

---

## Slide 10: Efficiency & FlashAttention
**Title: The Bottleneck: Quadratic Complexity**

- **The Problem:** $O(n^2)$ complexity. If you double the sequence length, you quadruple the compute cost.
- **FlashAttention:** 
    - Optimizes memory access (IO-aware).
    - Uses fast SRAM instead of slow HBM.
- **Impact:** Longer context windows (e.g., 128k tokens) become feasible.

---

## Slide 11: Scaling Laws
**Title: How Big is Enough?**

- **The Power Law:** Model performance increases predictably as we scale.
- **The Three Pillars:**
    1. **Compute:** More FLOPs.
    2. **Data:** More tokens.
    3. **Parameters:** Larger model.
- **Key Lesson:** Scaling only one (e.g., parameters) without the others is a waste of compute.

**[Diagram Suggestion: A graph showing the logarithmic relationship between Model Size and Loss]**

---

## Slide 12: Mixture of Experts (MoE)
**Title: Smarter, Not Harder**

- **Dense Models:** Every token activates every parameter. (Expensive!)
- **MoE Models:** Only a few "experts" (sub-networks) are activated per token.
- **The Router:** A gatekeeper that sends the token to the best expert.
- **Benefit:** High capacity (trillions of parameters) with low inference cost.

**[Diagram Suggestion: A "Router" block sending tokens to one of eight "Expert" blocks]**

---

## Slide 13: Base Model Training
**Title: Next-Token Prediction**

- **The Task:** "The cat sat on the ___" $\rightarrow$ "mat".
- **The Objective:** Minimize Cross-Entropy Loss.
- **Training Data:** The "Internet" (Common Crawl, GitHub, Books).
- **The Goal:** To learn the statistical structure of human language.

---

## Slide 14: Pre-training Infrastructure
**Title: The Cost of Intelligence**

- **Hardware:** Thousands of GPUs (A100/H100).
- **Communication:** NVLink (ultra-fast interconnects).
- **Parallelism:**
    - **Data Parallelism:** Same model, different data.
    - **Model Parallelism:** Different parts of the model, same data.

---

## Slide 15: Evaluation
**Title: Is the Model actually good?**

- **Perplexity:** How "surprised" is the model by the next token? (Lower is better).
- **Benchmarks:**
    - **MMLU:** General knowledge/reasoning.
    - **HumanEval:** Python coding ability.
- **Reminder:** A Base Model is a "completer," not a "chatbot." (SFT comes later!)