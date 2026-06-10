# Lab 2: Transformer Flavors - Encoder vs. Decoder

## Introduction
In the previous lab, we learned the mathematical primitives that power LLMs. Now, we move from the "micro" math to the "macro" architecture. While the original Transformer was a hybrid, modern LLMs have diverged into three distinct flavors of architecture. 

Understanding which flavor is used for which task is critical for choosing the right model for your specific domain adaptation.

---

## 1. The Encoder-only Model (The "Understander")
Encoder-only models (like BERT) are designed to process the entire input sequence simultaneously. They are **Bidirectional**, meaning every token can "see" every other token in the sentence.

### Concept: Bidirectional Context
In a sentence like *"The bank of the river was muddy,"* a BERT-style model processes "bank" by looking at "The", "of", "the", "river", and "was muddy" all at once. It uses the surrounding context from both sides to determine that "bank" refers to land, not a financial institution.

### 🛠️ Task 1: Masked Language Modeling (MLM)
The primary training objective for encoder-only models is **Masked Language Modeling**. The model is given a sentence with some words hidden (masked) and must predict the hidden word.

**Scenario:**
You are training a model for medical entity recognition. You provide the following input:
*"The patient presented with [MASK] hypertension."*

**Exercise:**
1. List three possible tokens the model might predict for the `[MASK]`.
2. Why is the bidirectional nature of the model helpful here? (Hint: Think about the words "patient" and "hypertension").
3. Contrast this with a model that could only see words to the left. If the model only saw *"The patient presented with..."*, would it be as accurate in its prediction?

---

## 2. The Decoder-only Model (The "Generator")
Decoder-only models (like GPT-4, Llama 3, Mistral) are the foundation of modern generative AI. They are **Unidirectional (Causal)**, meaning a token can only "see" tokens that came before it.

### Concept: Causal Masking
To prevent the model from "cheating" during training, we use **Causal Masking**. This mathematically blocks the model from seeing future tokens.

**Visualizing the Mask:**
If the input is: `"The cat sat on the mat"`
- When predicting the word "sat", the model can only see: `["The", "cat"]`
- When predicting the word "on", the model can only see: `["The", "cat", "sat"]`
- It is **blind** to everything to the right of the current token.

### Concept: Autoregressive Generation
Generative models work in a loop:
1. Input: `"The cat"` $\rightarrow$ Model predicts: `"sat"`
2. Input: `"The cat sat"` $\rightarrow$ Model predicts: `"on"`
3. Input: `"The cat sat on"` $\rightarrow$ Model predicts: `"the"`
This process is called **Autoregressive Generation**.

### 🛠️ Task 2: Causal Prediction
Given the sequence: *"The capital of France is [NEXT]"*

**Exercise:**
1. What is the most likely token for `[NEXT]`?
2. If the model was bidirectional (encoder-only) and the input was *"The capital of France is [MASK] and it is famous for the Eiffel Tower,"* how does the "clue" about the Eiffel Tower change the way the model predicts the mask compared to the causal (decoder-only) approach?

---

## 3. The Encoder-Decoder Model (The "Translator")
The original Transformer architecture was a hybrid. The **Encoder** processes the source sequence (understanding), and the **Decoder** generates the target sequence (generating).

### Use Case: Sequence-to-Sequence (Seq2Seq)
This is ideal for translation. The Encoder reads the English sentence and creates a "thought vector." The Decoder then takes that vector and generates the French translation one word at a time.

### 🛠️ Task 3: The Translation Flow
Imagine translating *"Hello world"* $\rightarrow$ *"Bonjour le monde"*.

**Exercise:**
1. Which part of the model (Encoder or Decoder) sees the full English phrase *"Hello world"* all at once?
2. Which part of the model generates *"Bonjour"*, then *"le"*, then *"monde"* sequentially?
3. Why would using a Decoder-only model for this be different from using an Encoder-Decoder model?

---

## Lab Summary Checklist
- [ ] I can explain the difference between Bidirectional and Causal context.
- [ ] I understand why Masked Language Modeling is used for Encoders.
- [ ] I can describe the autoregressive loop of a Decoder.
- [ ] I can identify which architecture is best for understanding vs. generating text.