# Quiz 1: Scaling, MoE, and Training Infrastructure (Module 1, Part 2)

**Student Name:** ____________________  
**Date:** ____________________  
**Score:** /20

---

## Instructions
Answer the following questions briefly. Show calculations where necessary.

**Q1. (5 pts) Chinchilla Rule Quick-Check**
According to the Chinchilla scaling rule, what is the optimal number of tokens for a model with 7 Billion parameters?

**Q2. (5 pts) MoE Parameters**
In a Mixture of Experts model with 16 experts, where each expert has 100M parameters, and $k=2$ (top-2 routing), how many expert parameters are active for a single token?

**Q3. (5 pts) VRAM Calculation**
If a model has 1 Billion parameters and you are using the Adam optimizer (which stores 2 states per parameter in FP32), how much VRAM is consumed by the optimizer states alone?

**Q4. (5 pts) Distributed Training**
Which distributed training strategy is strictly required if a model's weights alone exceed the VRAM of a single GPU?
- A) Data Parallelism (DP)
- B) Model Parallelism (MP)
- C) Gradient Accumulation
- D) Weight Tieing

---