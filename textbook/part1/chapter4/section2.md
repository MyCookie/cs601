# 4.2 Data Requirements for Pre-training

## Peer-to-Peer Guide
Okay, so we know the model is just trying to predict the next token. But for that to work, it needs to see *everything*. If you only feed a model cookbooks, it will be great at making soufflés but will have no idea how to write a Python script or explain the French Revolution. 

This is where the "Large" in Large Language Model comes from. Pre-training data isn't just about size; it's about **diversity** and **quality**.

### The "Diet" of an LLM

Think of pre-training data as the model's education. To build a general-purpose intelligence, the model needs a diverse diet:

1. **The Web (Common Crawl):** This is the bulk of the data. It's like the model reading the entire public internet. It learns how people talk, how forums work, and general world knowledge. But there's a catch: the internet is messy. It's full of spam, ads, and toxic content.
2. **Code (GitHub):** This is the "secret sauce." Even if you aren't building a coding assistant, training on code teaches the model **logical reasoning**, structured thinking, and how to follow strict rules. 
3. **Books (Project Gutenberg/Books3):** Books provide long-form coherence. While a tweet is a snippet, a book teaches the model how to maintain a consistent narrative or argument over thousands of words.
4. **Academic Papers (arXiv):** This is where the model learns formal language, mathematics, and cutting-edge scientific concepts.

### The Quality Problem: "Garbage In, Garbage Out"

If you train a model on a billion pages of spam, you get a model that speaks like a spam bot. Data engineers spend a huge amount of time on **Curation**.

**How we clean the data:**
- **Deduplication:** If the same article appears 1,000 times on the web, the model might "overfit" on it and just memorize it verbatim. We use algorithms to remove near-duplicate text.
- **Filtering:** We use "heuristic filters" (e.g., "Does this page have too many symbols?") or "quality classifiers" (small models trained to distinguish between a high-quality Wikipedia article and a low-quality forum post).
- **Toxicity Removal:** Removing hate speech or dangerous instructions before the model ever sees them.

### Scaling the Data: Tokens vs. Bytes

When we talk about data size, we don't usually say "Terabytes." We talk about **Tokens**. 

If a model is trained on "2 Trillion Tokens," it means it has seen 2 trillion units of text. Because tokens are usually smaller than words (e.g., "apple" is 1 token, but "unbelievable" might be 3 tokens), the actual amount of text is massive.

**The Scaling Law Intuition:**
Generally, the more high-quality tokens you feed a model (up to a point), the lower its "loss" (the penalty score we talked about in 4.1) becomes. This is the fundamental driver behind the race to scrape more of the web!

---

## Technical Summary

**Data Composition and Diversity**
Effective pre-training requires a balanced mixture of corpora to ensure generalization across domains. A typical mixture includes:
- **Web Corpora:** (e.g., Common Crawl, C4) for general knowledge and linguistic variety.
- **Code:** (e.g., The Stack, GitHub) for improving logical reasoning and structural coherence.
- **Formal Text:** (e.g., Wikipedia, arXiv) for factual accuracy and technical terminology.
- **Books:** For long-range dependency modeling and narrative flow.

**The Curation Pipeline**
To prevent model degradation (e.g., "hallucinations" caused by noisy data or "memorization" due to duplicates), a rigorous pipeline is employed:
1. **Deduplication:** Implementation of MinHash or LSH (Locality Sensitive Hashing) to remove overlapping documents.
2. **Quality Filtering:** Application of fastText classifiers or regex-based heuristics to discard "low-signal" content (e.g., boilerplate, gibberish).
3. **Safety Filtering:** Removal of PII (Personally Identifiable Information) and toxic content using blacklist-based or model-based filters.

**Data Volume and Scaling**
The relationship between data volume ($D$), model parameters ($N$), and compute ($C$) is governed by **Scaling Laws**. Chinchilla scaling suggests that for a given compute budget, most models are under-trained. The optimal ratio is approximately 20 tokens per parameter. 

For a 7B parameter model, this implies a minimum of $\sim 140$ billion tokens for optimal convergence, though modern "over-training" (e.g., Llama-3) uses trillions of tokens to further push the performance ceiling.