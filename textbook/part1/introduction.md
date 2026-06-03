# Part I: Foundations of Modern LLMs

To effectively adapt and deploy Large Language Models (LLMs), we must first understand the architectural decisions that enable their emergent capabilities. This part of the textbook focuses on the structural foundations of modern LLMs, moving from the core Transformer architecture to the optimizations that allow models to scale.

We begin by analyzing the three primary Transformer variants: encoder-only, decoder-only, and encoder-decoder architectures. Understanding these distinctions is critical, as the choice of architecture dictates whether a model is optimized for representation, generation, or translation.

From there, we dive into the attention mechanism. We will explore how self-attention and multi-head attention allow models to capture complex dependencies across long sequences, and how innovations like FlashAttention address the quadratic computational complexity that typically limits context windows.

We then examine scaling laws and the shift toward sparse architectures. We will discuss how Mixture of Experts (MoE) allows for an increase in model capacity without a proportional increase in compute cost per token, using architectures like Mixtral as primary case studies.

Finally, we cover the fundamentals of base model training. We will examine the next-token prediction objective, the data requirements for pre-training, and the massive compute infrastructure required to produce a high-quality base model.

By the end of this section, you will have the theoretical grounding necessary to move from using LLMs as black-box APIs to engineering them as malleable systems.