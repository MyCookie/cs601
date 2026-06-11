# 4.4 Evaluation of Base Models

## Peer-to-Peer Guide
So, you've spent millions of dollars and months of compute time. Your model is finally done training. But how do you actually *know* if it's any good? You can't just ask it a few questions and say, "Yeah, it looks pretty smart," because that's called **cherry-picking** (picking only the examples that work and ignoring the la ones that don't).

Evaluating a base model is tricky because base models aren't "chatbots" yet—they are "document completers." They don't know they are supposed to answer a question; they just want to complete the pattern.

### The "Perplexity" Metric
The most common way to measure a model's performance during training is **Perplexity (PPL)**. 

Think of perplexity as a measure of "uncertainty." If a model is perfectly predicting every single token in a test set, its perplexity is 1. If it's completely confused, the perplexity is very high.

**The Intuition:**
If a model has a perplexity of 10, it means that when the model is predicting the next token, it is as confused as if it were choosing uniformly among 10 possible tokens. Lower is better.

### Zero-Shot and Few-Shot Evaluation
Since we want to know if the model has actually learned useful knowledge, we use **Benchmarks**. 

1. **Zero-Shot:** We give the model a prompt and ask for an answer without any examples.
   - *Prompt:* "The capital of France is"
   - *Completion:* "Paris."
2. **Few-Shot:** We give the model a few examples of the task first, then ask it to do the same. This "primes" the model to understand the pattern.
   - *Prompt:* "Q: What is the capital of Germany? A: Berlin. Q: What is the capital of Italy? A: Rome. Q: What is the capital of France? A:"
   - *Completion:* "Paris."

### Common Benchmarks
You'll see these names all over the place in research papers:

- **MMLU (Massive Multitask Language Understanding):** A giant test covering 57 subjects across STEM, humanities, and other fields. It's the gold standard for "general intelligence."
- **HumanEval / MBPP:** Tests specifically for Python coding ability.
- **GSM8K:** A test of the model's ability to do multi-step grade-school math problems.

### The Trap: Data Contamination
The biggest danger in evaluation is **Data Contamination**. This happens when the training data (the internet) accidentally includes the test questions and answers from the benchmarks.

If a model gets 100% on MMLU, it might not be "smart"—it might have just memorized the MMLU test from a website it read during pre-training. This is why researchers are now fighting for "private" benchmarks that the model has never seen before.

---

## Technical Summary

**Quantitative Metrics**
The primary internal metric for evaluating causal language models is **Perplexity**, defined as the exponentiated average negative log-likelihood:
$$\text{PPL}(X) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i}) \right)$$
A lower perplexity indicates the model has a better probability distribution over the test corpus, representing higher predictive confidence and lower entropy.

**Evaluation Paradigms**
Base models are evaluated using **In-Context Learning (ICL)**:
1. **Zero-Shot Learning:** The model generates a completion based solely on the la prompt.
2. **Few-Shot Learning:** The model is provided with a small set of examples $\mathcal{E}$ to constrain the output distribution to a specific task format.

**Standardized Benchmarks**
General capability is measured across multiple axes:
- **World Knowledge:** MMLU (Massive Multitask Language Understanding) measures cross-domain factual recall.
- **Reasoning:** GSM8K (Grade School Math 8K) evaluates chain-of-thought reasoning in mathematics.
- **Causal/Logical Reasoning:** HumanEval and MBPP evaluate the synthesis of functional code.

**The Contamination Problem**
**Data Contamination** occurs when the test set $\mathcal{T}$ is present in the pre-training corpus $\mathcal{D}_{train}$. This results in an artificial inflation of benchmark scores due to memorization rather than generalization. Mitigation strategies include:
- **n-gram overlap analysis:** Comparing test sets against the training corpus to detect verbatim leaks.
- **Canary strings:** Inserting unique identifiers into test sets to detect their presence in training data.
- **Evaluation on "held-out" or synthetic datasets.**