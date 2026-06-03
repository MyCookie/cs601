# Chapter 1: Transformer Architectures

Welcome to the start of the journey. If you've used ChatGPT, Claude, or Gemini, you've already interacted with the output of a Transformer. But if you're here, you're probably looking to move beyond just prompting these models and start understanding how they actually work under the hood.

At its core, a Transformer is just a specific way of designing a neural network that is exceptionally good at handling sequences of data—like text. Before Transformers, we relied on architectures that processed text one word at a time, which was slow and often "forgot" the beginning of a sentence by the time it reached the end.

> **Neural Network:** A computing system inspired by the biological neural networks that constitute animal brains. In the context of LLMs, it's a series of mathematical layers that learn patterns in data to make predictions.

The "magic" of the Transformer is that it can look at an entire sequence of text all at once. This allows the model to understand the context of a word based on everything else around it.

In this chapter, we're going to look at the three primary "flavors" of Transformer architectures. You'll see that while they all share the same DNA, they are built for very different purposes:

1.  **Encoder-only models (like BERT):** These are designed to "understand" and categorize text. Think of them as the great readers.
2.  **Decoder-only models (like GPT):** These are the architects of generation. They are designed to predict the next word in a sequence.
3.  **Encoder-Decoder models (like T5):** These are the translators. They take one sequence as input and transform it into a completely different sequence as output.

Understanding these distinctions is crucial. Depending on whether you are building a sentiment analyzer, a chatbot, or a translation tool, your choice of architecture will fundamentally change how you handle your data and how you tune your model.

Let's dive into the first one: the Encoder.