Now that we've looked at the overall blueprints of Transformers in Chapter 1, it's time to dive into the engine that actually makes them work: Attention.

If you're coming into this fresh, "attention" might sound like a human psychological term, but in our context, it's a mathematical trick. Imagine reading a sentence: your brain doesn't give equal weight to every single word. If I say "The bank of the river," you know "bank" refers to land, not a financial institution, because you're paying attention to the word "river."

> **Contextual Weighting:** The process of assigning different levels of importance to different parts of an input to determine the meaning of a specific element.

In this chapter, we're going to break down exactly how the model does this. We'll start with the basic mechanics of Self-Attention, then see how Multi-Head Attention lets the model "look" at the data from different perspectives. We'll also discuss FlashAttention—a clever optimization that makes these models actually feasible to run—and finally, we'll look at the bottlenecks that make attention so computationally expensive.

Let's get into it.