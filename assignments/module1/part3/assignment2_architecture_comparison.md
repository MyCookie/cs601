# Assignment 2: Architecture Comparison - Encoder, Decoder, and Encoder-Decoder

**Name:** ___________________________  
**Date:** ____________________________

---

## Overview
The Transformer architecture has evolved into three primary variants: Encoder-only (e.g., BERT), Decoder-only (e.g., GPT), and Encoder-Decoder (e.g., T5). This assignment focuses on comparing these architectures from a structural and functional perspective.

---

## Task 1: Architectural Mapping

Complete the following table by checking the appropriate box and providing a brief justification.

| Feature | Encoder-Only (BERT) | Decoder-Only (GPT) | Encoder-Decoder (T5) | Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Bidirectional Attention** | [ ] | [ ] | [ ] | |
| **Causal Masking** | [ ] | [ ] | [ ] | |
| **Cross-Attention** | [ ] | [ ] | [ ] | |
| **Auto-regressive Generation** | [ ] | [ ] | [ ] | |
| **Masked LM Objective** | [ ] | [ ] | [ ] | |

---

## Task 2: Case Study - Task Selection

For each of the following tasks, identify which of the three architectures is **best suited** for the job and **explain why**.

### 1. Sentiment Analysis (Classifying a movie review as Positive or Negative)
**Best Architecture:** ___________________________  
**Reasoning:**
\
\
\

### 2. Creative Story Writing (Given a prompt, generate a 500-word story)
**Best Architecture:** ___________________________  
**Reasoning:**
\
\
\

### 3. English-to-French Translation
**Best Architecture:** ___________________________  
**Reasoning:**
\
\
\

---

## Task 3: Deep Dive - The Role of Cross-Attention

In an Encoder-Decoder architecture (like T5), there are two types of attention in the decoder: **Self-Attention** and **Cross-Attention**.

**Question:**
1. Describe the difference between the Query (Q), Key (K), and Value (V) sources for Self-Attention vs. Cross-Attention in the decoder.
2. What would happen to the model's ability to translate if we removed the Cross-Attention layers but kept the Self-Attention layers?
3. How does Cross-Attention allow the model to handle sequences of different lengths (e.g., a short English sentence translating to a longer French sentence)?

**Your Response:**
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\