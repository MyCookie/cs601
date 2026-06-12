# Textbook Outline: Post-Training of Large Language Models

This document defines the structure of the course textbook. Each section includes a title, description, prerequisite knowledge, key concepts, and the elements needed to write the content (figures, code examples, exercises, further reading).

## Authoring Convention

Every chapter follows this structure:

1. **Learning Objectives** - What the reader will know by the end
2. **Prerequisites** - What the reader should already know
3. **Core Content** - Sections and subsections with prose, equations, figures, and code
4. **Worked Example** - A complete, end-to-end code example
5. **Exercises** - Practice problems with increasing difficulty
6. **Summary** - Key takeaways
7. **Further Reading** - Papers and resources for deeper study

Each section below lists the elements required to author it.

---

## Part I: Foundations

### Chapter 1: Machine Learning Fundamentals

**Audience:** CS graduates with no ML background. Assumes proficiency in at least one programming language, data structures, and algorithms.

#### Section 1.1: What Is Machine Learning?
- **Content:** Definition of ML, types of learning (supervised, unsupervised, reinforcement), the bias-variance tradeoff, generalization
- **Elements needed:**
  - Figure: taxonomy of ML tasks
  - Code: zero-to-hero linear regression on a toy dataset
  - Exercise: classify a dataset into supervised/unsupervised/RL

#### Section 1.2: Linear Algebra for Machine Learning
- **Content:** Vectors, matrices, tensors, operations (dot product, matrix multiply, outer product), norms, eigenvalues/eigenvectors, SVD
- **Elements needed:**
  - Figure: visualizing matrix multiplication as linear transformation
  - Code: implement matrix multiply, SVD decomposition in NumPy
  - Exercise: compute eigenvalues of a covariance matrix

#### Section 1.3: Probability and Statistics
- **Content:** Random variables, distributions (Gaussian, Bernoulli, categorical), expectation, variance, covariance, Bayes' theorem, maximum likelihood estimation, maximum a posteriori
- **Elements needed:**
  - Figure: common distributions and their shapes
  - Code: implement MLE for Gaussian parameters
  - Exercise: apply Bayes' theorem to a spam classification problem

#### Section 1.4: Optimization
- **Content:** Loss functions, gradient descent, stochastic gradient descent, momentum, RMSProp, Adam, learning rate schedules (warmup, cosine decay), gradient clipping, vanishing/exploding gradients
- **Elements needed:**
  - Figure: loss landscape with optimizer trajectories
  - Code: implement SGD, Momentum, and Adam from scratch
  - Exercise: tune learning rate for a small neural network

#### Section 1.5: Model Evaluation
- **Content:** Train/validation/test splits, k-fold cross-validation, overfitting/underfitting diagnosis, regularization (L1, L2, dropout, early stopping), metrics (accuracy, precision, recall, F1, perplexity)
- **Elements needed:**
  - Figure: learning curves showing over/underfitting
  - Code: implement k-fold cross-validation
  - Exercise: diagnose and fix an overfitting model

**Worked Example:** Build a regularized logistic regression classifier with cross-validation and early stopping, evaluated on a held-out test set.

---

### Chapter 2: Deep Learning Building Blocks

**Prerequisites:** Chapter 1

#### Section 2.1: Neural Networks
- **Content:** Perceptrons, multi-layer perceptrons, activation functions (sigmoid, tanh, ReLU, LeakyReLU, GeLU, SiLU), universal approximation theorem, why depth matters
- **Elements needed:**
  - Figure: MLP forward pass with activation at each layer
  - Code: implement an MLP forward and backward pass
  - Exercise: compare activation functions on a non-linear regression task

#### Section 2.2: Convolutional Neural Networks
- **Content:** Convolution operation, filters/kernels, stride, padding, pooling, 1D/2D/3D convolutions, why CNNs dominate vision but not text
- **Elements needed:**
  - Figure: convolution operation step-by-step
  - Code: implement 1D convolution from scratch
  - Exercise: apply a CNN to a text classification task and observe limitations

#### Section 2.3: Recurrent Neural Networks
- **Content:** RNN architecture, shared weights across time, backpropagation through time, vanishing gradient problem, why RNNs cannot parallelize
- **Elements needed:**
  - Figure: RNN unrolled through time steps
  - Code: implement an RNN language model
  - Exercise: demonstrate vanishing gradients with a deep RNN

#### Section 2.4: LSTMs and GRUs
- **Content:** Gate mechanisms (input, forget, output), cell state, residual connections within LSTM, GRU as a simplified variant
- **Elements needed:**
  - Figure: LSTM cell diagram with gates
  - Code: implement LSTM from scratch
  - Exercise: compare RNN vs LSTM on a long-sequence task

#### Section 2.5: Sequence-to-Sequence Models
- **Content:** Encoder-decoder architecture, teacher forcing, beam search, the attention mechanism as a fix for bottleneck representations
- **Elements needed:**
  - Figure: seq2seq with and without attention
  - Code: build a seq2seq model with attention for a simple translation task
  - Exercise: compare teacher forcing vs scheduled sampling

#### Section 2.6: Normalization
- **Content:** BatchNorm, LayerNorm, RMSNorm, weight normalization, why LayerNorm is standard for Transformers
- **Elements needed:**
  - Figure: BatchNorm vs LayerNorm computation graphs
  - Code: implement LayerNorm and RMSNorm
  - Exercise: ablate normalization layers on a Transformer training run

**Worked Example:** Train an LSTM-based machine translation model, then add attention and measure the improvement in BLEU score.

---

### Chapter 3: The Transformer Architecture

**Prerequisites:** Chapter 2

#### Section 3.1: The Transformer Paper
- **Content:** Full walkthrough of Vaswani et al. (2017), encoder-decoder design, why the paper changed everything
- **Elements needed:**
  - Figure: full Transformer architecture diagram
  - Code: minimal Transformer implementation following the paper
  - Exercise: trace a forward pass through the architecture diagram

#### Section 3.2: Scaled Dot-Product Attention
- **Content:** Query, Key, Value matrices, the attention function, why divide by sqrt(d_k), softmax, masking
- **Elements needed:**
  - Figure: attention computation as matrix operations
  - Code: implement scaled dot-product attention
  - Exercise: visualize attention weights on a sentence pair

#### Section 3.3: Multi-Head Attention
- **Content:** Why multiple heads, parallel computation, projection matrices, concatenation and final projection, interpretability of heads
- **Elements needed:**
  - Figure: multi-head attention with head specialization
  - Code: implement multi-head attention
  - Exercise: analyze what different attention heads attend to

#### Section 3.4: Positional Encodings
- **Content:** Why position matters, sinusoidal encodings, learned positional embeddings, RoPE (Rotary Positional Embeddings), ALiBi, relative vs absolute position
- **Elements needed:**
  - Figure: sinusoidal vs RoPE positional encodings
  - Code: implement sinusoidal, learned, and RoPE positional encodings
  - Exercise: extrapolate a model trained with each encoding to longer sequences

#### Section 3.5: Transformer Blocks
- **Content:** Encoder block, decoder block, cross-attention, residual connections, feed-forward network (FFN), why GELU and why two hidden dimensions
- **Elements needed:**
  - Figure: decoder block with mask and cross-attention
  - Code: implement a full Transformer decoder block
  - Exercise: remove residual connections and observe training degradation

#### Section 3.6: Attention Variants and Optimizations
- **Content:** Sparse attention patterns (local, stride, low-rank, random), FlashAttention (IO-aware algorithm, tiling, reordering), memory bandwidth bottleneck, streaming attention
- **Elements needed:**
  - Figure: FlashAttention tiling and computation reordering
  - Code: benchmark naive attention vs FlashAttention
  - Exercise: profile attention memory I/O to understand the bottleneck

**Worked Example:** Build a GPT-2 style decoder-only Transformer from scratch, train on the WikiText-2 dataset, and report perplexity.

---

### Chapter 4: Scaling Laws, Mixture-of-Experts, and Pre-Training

**Prerequisites:** Chapter 3

#### Section 4.1: Scaling Laws
- **Content:** Kaplan scaling laws (loss vs N, d, T), power-law fit, compute-optimal regime, extrapolation to 100T+ scales, Chinchilla laws and the token-to-parameter ratio, implications for model design
- **Elements needed:**
  - Figure: scaling law curves from GPT-3 and Chinchilla papers
  - Code: fit a power law to published benchmark data
  - Exercise: compute the optimal model size for a given compute budget

#### Section 4.2: The Emergent Abilities of Scale
- **Content:** In-context learning, chain-of-thought, few-shot prompting, phase transitions, why capabilities appear discontinuously despite continuous loss curves
- **Elements needed:**
  - Figure: emergent ability curves vs model size
  - Code: demonstrate in-context learning with a small model family
  - Exercise: identify an emergent behavior in a scaling experiment

#### Section 4.3: Mixture-of-Experts Fundamentals
- **Content:** Dense vs sparse activation, the core MoE idea, gating network, top-k routing, expert capacity, auxiliary load-balancing loss
- **Elements needed:**
  - Figure: MoE layer vs dense FFN layer
  - Code: implement a MoE layer with top-k gating
  - Exercise: measure compute-vs-memory tradeoff of MoE vs dense

#### Section 4.4: MoE Architectures
- **Content:** Switch Transformer (one expert per token), GShard (combining MoE with hierarchical routing), Mixtral 8x7B (sparse MoE with dense attention), Grok (diagram-of-thought), training stability challenges (expert collapse, load imbalance)
- **Elements needed:**
  - Figure: Mixtral architecture with expert routing
  - Code: implement a Switch Transformer layer
  - Exercise: diagnose expert collapse in a training run

#### Section 4.5: Base Model Pre-Training
- **Content:** Next-token prediction objective, causal vs masked language modeling, data mixture design (web, books, code, academic), curriculum learning, context length during pre-training, multi-modal pre-training
- **Elements needed:**
  - Figure: pre-training data mixture pie chart
  - Code: inspect and analyze a pre-training data mixture
  - Exercise: design a data mixture for a domain-specific model

#### Section 4.6: Training Infrastructure
- **Content:** Data parallelism, model parallelism (tensor, pipeline, sequence), ZeRO optimization, distributed training frameworks (DeepSpeed, FSDP), checkpointing, training monitoring
- **Elements needed:**
  - Figure: data vs tensor vs pipeline parallelism
  - Code: configure a multi-GPU training run with FSDP
  - Exercise: estimate training time for a given model size and GPU budget

**Worked Example:** Design a pre-training plan for a 3B-parameter model given 8x A100 GPUs. Include data mixture, training schedule, parallelism strategy, and expected training time.

---

## Part II: Data & Training

### Chapter 5: Datasets for Large Language Models

**Prerequisites:** Chapter 1 (machine learning fundamentals)

#### Section 5.1: Pre-Training Corpora
- **Content:** Common Crawl (web crawling, dedup, language filtering), The Pile (22-domain mixture), RefinedWeb (quality-filtered web data), C4 (colossal cleaned corpus), RedPajama (open GPT-3 replica)
- **Elements needed:**
  - Table: pre-training corpus comparison (size, domains, license)
  - Code: download and inspect a subset of a pre-training corpus
  - Exercise: estimate the storage and compute needed to process a corpus

#### Section 5.2: Instruction Datasets
- **Content:** Alpaca (52K self-instruct), Stanford Alpaca extensions, flan collection, Dolly (hermes), UltraChat (web collected), Orca (teacherforced reasoning), instruction format (prompt/instruction/output vs system/user/assistant)
- **Elements needed:**
  - Table: instruction dataset comparison (size, format, license, domain)
  - Code: convert between instruction formats
  - Exercise: design an instruction template for a domain-specific task

#### Section 5.3: Preference Datasets
- **Content:** Anthropic's helpful-harmless dataset, HH-RLHF, DPO-style preference pairs, chosen/rejected format, annotator guidelines, inter-annotator agreement
- **Elements needed:**
  - Figure: preference data collection pipeline
  - Code: explore a preference dataset distribution
  - Exercise: create preference pairs from instruction outputs

#### Section 5.4: Dataset Quality Assessment
- **Content:** Quality dimensions (toxicity, factual correctness, instruction diversity, language, domain balance), automated quality scoring (perplexity-based, classifier-based, LLM-as-judge), manual review processes, dataset cards
- **Elements needed:**
  - Figure: quality scoring pipeline
  - Code: build a quality scoring report for a dataset
  - Exercise: identify and remove toxic samples from a dataset

#### Section 5.5: Synthetic Data
- **Content:** Self-Instruct (data generation from a seed set), Evol-Instruct (evolutionary prompt complexity), data augmentation via LLM APIs, quality control for synthetic data, contamination risks
- **Elements needed:**
  - Figure: self-instruct generation loop
  - Code: generate synthetic instructions using an API model
  - Exercise: evaluate the quality of synthetic vs human-written instructions

#### Section 5.6: Data Licensing and Provenance
- **Content:** Open data licenses (CC-BY, CC0, ODC-BY), copyleft concerns, copyright and training data lawsuits, data provenance tracking, the open weights vs open data distinction
- **Elements needed:**
  - Table: common data licenses and their permissions
  - Exercise: audit a dataset mixture for license compatibility

**Worked Example:** Profile three instruction-tuning datasets (Alpaca, Dolly, UltraChat subset) for quality. Produce a dataset card with duplication rate, toxicity score, instruction diversity, and domain distribution.

---

### Chapter 6: Data Engineering Pipelines

**Prerequisites:** Chapter 5

#### Section 6.1: Pipeline Architecture
- **Content:** ETL for ML data, batch vs streaming pipelines, DAG-based orchestration, fault tolerance, idempotency, scalability
- **Elements needed:**
  - Figure: data pipeline DAG with stages
  - Code: design a data pipeline with error handling and retries
  - Exercise: implement a fault-tolerant data processing step

#### Section 6.2: Text Cleaning
- **Content:** HTML stripping, Unicode normalization (NFC, NFD), encoding detection and fix, whitespace normalization, PII redaction, language identification
- **Elements needed:**
  - Code: implement a comprehensive text cleaning pipeline
  - Table: cleaning operations and their impact on data size
  - Exercise: measure the proportion of documents removed by each cleaning step

#### Section 6.3: Deduplication
- **Content:** Exact dedup (hash-based), fuzzy dedup (minhash, SimHash, LSH), document-level vs sentence-level dedup, cross-lingual dedup, why dedup improves generalization
- **Elements needed:**
  - Figure: minhash + LSH algorithm diagram
  - Code: implement minhash-based near-dedup
  - Exercise: measure the impact of dedup on held-out perplexity

#### Section 6.4: Quality Filtering
- **Content:** Heuristic filters (character/word ratio, line count, entropy, alphanumeric ratio), perplexity-based filtering (using a reference model), classifier-based filtering, model-based scoring, quality thresholds
- **Elements needed:**
  - Figure: quality filter cascade with data loss at each stage
  - Code: implement a multi-stage quality filter
  - Exercise: tune filtering thresholds to maximize quality while retaining coverage

#### Section 6.5: Formatting for Fine-Tuning
- **Content:** Instruction format templates (Alpaca, ChatML, ShareGPT), system/user/assistant role formatting, conversation history formatting, special tokens, EOS placement
- **Elements needed:**
  - Table: instruction format comparison with examples
  - Code: implement formatters for three instruction formats
  - Exercise: measure token count impact of different formatting choices

#### Section 6.6: Data Versioning and Reproducibility
- **Content:** DVC (Data Version Control), dataset cards, provenance tracking, reproducibility in data pipelines, hashing and checksums
- **Elements needed:**
  - Code: set up DVC for a dataset with versioning
  - Exercise: create a dataset card following the Hugging Face template

**Worked Example:** Build an end-to-end data pipeline that takes a raw Common Crawl dump and produces a cleaned, deduplicated, quality-scored instruction dataset. Document data volume at each stage.

---

### Chapter 7: Tokenization

**Prerequisites:** Chapters 1, 5

#### Section 7.1: The Tokenization Problem
- **Content:** Why tokenization matters, word-level tokenization, open-vocabulary problem, out-of-vocabulary (OOV) tokens, subword tokenization as a compromise
- **Elements needed:**
  - Figure: word-level vs subword tokenization on the same text
  - Code: implement a word-level tokenizer and measure OOV rate
  - Exercise: quantify the OOV problem on a domain-specific corpus

#### Section 7.2: Byte-Pair Encoding
- **Content:** BPE algorithm, merge rules, training a BPE vocabulary, vocabulary size tradeoffs, byte-level BPE, GPT tokenization
- **Elements needed:**
  - Figure: BPE merge steps visualized
  - Code: implement BPE training from scratch
  - Exercise: train BPE tokenizers at different vocabulary sizes and compare

#### Section 7.3: WordPiece, Unigram, and SentencePiece
- **Content:** WordPiece (BERT tokenizer), Unigram model, SentencePiece framework, comparison of algorithms, when to choose which
- **Elements needed:**
  - Table: BPE vs WordPiece vs Unigram comparison
  - Code: train a WordPiece tokenizer and compare to BPE
  - Exercise: benchmark tokenization speed across algorithms

#### Section 7.4: Tokenizer Metrics and Evaluation
- **Content:** Vocabulary size, byte coverage, compression ratio (tokens per character), OOV rate, language coverage, domain adaptation score
- **Elements needed:**
  - Code: implement a tokenizer benchmark suite
  - Figure: radar chart of tokenizer metrics
  - Exercise: evaluate five tokenizers on a multilingual corpus

#### Section 7.5: Domain-Adaptive Tokenization
- **Content:** Vocabulary extension, continued pre-training with new tokens, code tokenization (whitespace sensitivity), math symbol handling, special token design
- **Elements needed:**
  - Code: extend a base tokenizer with domain-specific tokens
  - Exercise: measure OOV reduction after vocabulary extension

#### Section 7.6: Tokenization Edge Cases
- **Content:** Multilingual tokenization, CJK character handling, emoji, code formatting (whitespace, indentation), mathematical notation, special characters, BOS/EOS/BEG/END tokens
- **Elements needed:**
  - Code: stress-test a tokenizer on edge cases
  - Exercise: design a special token scheme for a code-generation model

**Worked Example:** Train a custom BPE tokenizer on a biomedical corpus. Compare efficiency to GPT-4 tokenizer. Document the impact of vocabulary size on model size and downstream perplexity.

---

### Chapter 8: Instruction Tuning

**Prerequisites:** Chapters 3, 6, 7

#### Section 8.1: From Base Model to Assistant
- **Content:** What instruction tuning is, why base models don't follow instructions, the chat template, supervised fine-tuning (SFT), the loss function for instruction data
- **Elements needed:**
  - Figure: base model vs instruction-tuned model behavior
  - Code: prepare instruction data for SFT
  - Exercise: compare base model and fine-tuned model responses

#### Section 8.2: Full Fine-Tuning
- **Content:** Training all parameters, compute requirements, catastrophic forgetting, when full FT makes sense, checkpointing, merging adapters
- **Elements needed:**
  - Table: memory requirements for full FT at different model sizes
  - Code: full fine-tune a small model and track GPU memory
  - Exercise: estimate the cost of full FT for a 70B model

#### Section 8.3: Low-Rank Adaptation (LoRA)
- **Content:** LoRA motivation and derivation, low-rank decomposition, target modules (attention vs FFN), rank r, scaling alpha, dropout, why LoRA works, training only adapters, merging LoRA weights
- **Elements needed:**
  - Figure: LoRA weight perturbation diagram
  - Code: apply LoRA to a Transformer and fine-tune
  - Exercise: ablate LoRA rank from 4 to 256 and measure quality

#### Section 8.4: Quantized LoRA (QLoRA)
- **Content:** 4-bit precision (NF4), why NF4 over INT4, paged optimizers, double quantization, memory savings, quality retention, when QLoRA is sufficient vs when you need full precision
- **Elements needed:**
  - Figure: QLoRA memory breakdown vs full FT
  - Code: fine-tune a 7B model with QLoRA on a single GPU
  - Exercise: compare QLoRA to LoRA quality on a benchmark

#### Section 8.5: Beyond LoRA
- **Content:** DoRA (weight-decomposed LoRA), AdaLoRA (adaptive rank allocation), BoFT (bias fine-tuning), IA³ (infused adapter by invisible tuning), prefix tuning, prompt tuning, comparative analysis
- **Elements needed:**
  - Table: PEFT method comparison (params, quality, speed, memory)
  - Code: compare LoRA vs DoRA on the same task
  - Exercise: choose the right PEFT method for a given constraint

#### Section 8.6: Training Configuration
- **Content:** Learning rate selection (rule of thumb for FT vs PEFT), batch size and gradient accumulation, number of epochs, warmup, weight decay, optimizer choice (AdamW), gradient checkpointing
- **Elements needed:**
  - Code: hyperparameter sweep on LoRA rank and learning rate
  - Figure: learning curves for different configurations
  - Exercise: find the optimal learning rate for a fine-tuning run

#### Section 8.7: Evaluation
- **Content:** Instruction-following benchmarks (IFEval, AlpacaEval, MT-Bench), automated evaluation with LLM-as-judge, perplexity on held-out data, human evaluation, leaderboard pitfalls
- **Elements needed:**
  - Code: evaluate a fine-tuned model on IFEval
  - Exercise: design an evaluation suite for a domain-specific model

**Worked Example:** Instruction-tune a Mistral-7B model using QLoRA on a customer service dataset. Compare to the base model on instruction-following benchmarks and ablate key hyperparameters.

---

### Chapter 9: Alignment

**Prerequisites:** Chapter 8

#### Section 9.1: The Alignment Problem
- **Content:** Why instruction tuning is insufficient, specification gaming, sycophancy, the need for preference learning, capability vs alignment tradeoff
- **Elements needed:**
  - Figure: instruction-tuned vs aligned model behavior examples
  - Exercise: identify misaligned behavior in a fine-tuned model

#### Section 9.2: Reward Modeling
- **Content:** Training a reward model from preference pairs, the Bradley-Terry model, reward model architecture, reward hacking, why reward models generalize poorly
- **Elements needed:**
  - Figure: reward model training pipeline
  - Code: train a reward model on a preference dataset
  - Exercise: evaluate reward model correlation with human judgments

#### Section 9.3: Reinforcement Learning from Human Feedback
- **Content:** RLHF overview, the four-model system (policy, reference, reward, value), PPO for language models, KL divergence penalty, clipping, advantage estimation, why RLHF is complex and expensive
- **Elements needed:**
  - Figure: full RLHF pipeline diagram
  - Code: walk through a PPO fine-tuning implementation
  - Exercise: tune the KL penalty and observe behavior changes

#### Section 9.4: Direct Preference Optimization
- **Content:** DPO derivation, why DPO replaces the reward model, the DPO loss, reference model, beta parameter, BRDPO (best-of-N), IPO (identity preference optimization), IPO vs DPO tradeoffs
- **Elements needed:**
  - Figure: DPO vs RLHF architecture comparison (simpler pipeline)
  - Code: apply DPO to a fine-tuned model
  - Exercise: tune the beta parameter and measure alignment

#### Section 9.5: Constitutional AI and Self-Alignment
- **Content:** Constitutional AI (Bai et al.), self-critique with principles, self-instruct for alignment, red-teaming, automated safety evaluation, guardrail models
- **Elements needed:**
  - Code: implement a self-critique loop with a constitution
  - Exercise: design a constitution for a domain-specific model

#### Section 9.6: Evaluating Alignment
- **Content:** Harmlessness benchmarks (XSTest, TruthfulQA, MMLU-Safety), refusal behavior, over-refusal, helpfulness-harmlessness tradeoff, adversarial evaluation, jailbreak resistance
- **Elements needed:**
  - Code: benchmark alignment on XSTest and TruthfulQA
  - Exercise: red-team a fine-tuned model and document vulnerabilities

**Worked Example:** Take the model from Chapter 8's worked example and align it using DPO. Evaluate on harmlessness and helpfulness benchmarks. Compare to the unaligned model and to a PPO-aligned baseline.

---

## Part III: Optimization & Deployment

### Chapter 10: Quantization

**Prerequisites:** Chapter 8

#### Section 10.1: Why Quantize?
- **Content:** Memory footprint, latency, throughput, cost, precision formats (FP32, FP16, BF16, FP8, INT8, INT4, NF4), the precision-vs-quality curve, when quantization is (and is not) worth it
- **Elements needed:**
  - Table: model size at different precisions for common model sizes
  - Code: measure model memory and latency at different precisions
  - Exercise: compute the cost savings of INT4 vs FP16 inference

#### Section 10.2: Post-Training Quantization
- **Content:** Weight-only quantization, activation quantization, per-channel vs per-tensor, symmetric vs asymmetric quantization, calibration data, accuracy retention
- **Elements needed:**
  - Figure: quantization mapping from float to int
  - Code: apply PTQ to a model and measure accuracy drop
  - Exercise: find the calibration dataset size needed for stable PTQ

#### Section 10.3: Quantization-Aware Training
- **Content:** QAT motivation, fake quantization, straight-through estimator, QAT vs PTQ accuracy, when QAT is worth the training cost
- **Elements needed:**
  - Figure: QAT training loop with fake quant nodes
  - Code: run QAT on a small model and compare to PTQ
  - Exercise: measure the compute cost of QAT vs the accuracy gain

#### Section 10.4: Algorithmic Quantization for Transformers
- **Content:** GPTQ (layer-by-layer quantization with outlier handling), AWQ (activation-aware weight quantization), SmoothQuant (moving quantization difficulty from activations to weights), comparison and when to use each
- **Elements needed:**
  - Figure: GPTQ layer-by-layer quantization with loss minimization
  - Code: quantize a 7B model with GPTQ and AWQ
  - Exercise: compare GPTQ and AWQ on a perplexity benchmark

#### Section 10.5: The 4-Bit Ecosystem
- **Content:** GGUF format, llama.cpp, EXL2, bitsandbytes, AWQ toolkit, quantization presets (q4_0, q4_1, q4_k_m, q5_0, q5_1, q8_0), choosing a quant preset
- **Elements needed:**
  - Table: GGUF quant preset comparison (size, speed, quality)
  - Code: convert a model to GGUF and serve with llama.cpp
  - Exercise: benchmark inference speed across quant presets

#### Section 10.6: Quantized Fine-Tuning
- **Content:** Training on quantized weights (QLoRA), quantize-then-fine-tune vs fine-tune-then-quantize, quality comparison, when to quantize in the pipeline
- **Elements needed:**
  - Figure: quantize-then-FT vs FT-then-quant pipeline comparison
  - Code: compare both quantization pipelines
  - Exercise: recommend a quantization strategy for a given quality constraint

**Worked Example:** Quantize a 7B instruction-tuned model to INT8 and INT4 using PTQ, GPTQ, and AWQ. Benchmark accuracy (perplexity on validation set), latency (tokens/sec), and memory for each. Recommend a production configuration.

---

### Chapter 11: Speculative Decoding and Inference Optimization

**Prerequisites:** Chapters 3, 10

#### Section 11.1: The Inference Bottleneck
- **Content:** Why autoregressive decoding is sequential, latency breakdown (prefill vs decode), memory bandwidth bound, why inference is the dominant cost in production
- **Elements needed:**
  - Figure: latency breakdown of autoregressive generation
  - Code: profile a model's inference latency at different sequence lengths
  - Exercise: identify the bottleneck (compute-bound vs memory-bound)

#### Section 11.2: Speculative Decoding
- **Content:** Draft model, verification step, token acceptance, acceptance rate, theoretical speedup, why speculative decoding works, the draft-target model size ratio
- **Elements needed:**
  - Figure: speculative decoding step-by-step
  - Code: implement speculative decoding with a small draft model
  - Exercise: measure speedup at different draft model sizes

#### Section 11.3: Draft Model Strategies
- **Content:** Smaller model as draft, echo decoding (using the target model's own history), lookback decoding, training a dedicated draft model, self-speculative decoding
- **Elements needed:**
  - Code: compare three draft model strategies
  - Figure: acceptance rate vs speedup for each strategy
  - Exercise: choose the best draft strategy for a given target model

#### Section 11.4: Beyond Speculative Decoding
- **Content:** Speculative token sampling, Medusa (multi-token prediction heads), parallel decoding, look-ahead decoding, chunked generation, early exiting
- **Elements needed:**
  - Figure: Medusa multi-head prediction
  - Code: benchmark multi-token prediction vs speculative decoding
  - Exercise: combine speculative decoding with quantization and measure compound speedup

#### Section 11.5: KV Cache Optimization
- **Content:** KV cache and memory growth, paged attention (vLLM), continuous batching, chunked prefill, RadixAttention (cache eviction and reuse), maximum throughput under concurrency
- **Elements needed:**
  - Figure: paged attention vs contiguous KV cache
  - Code: serve a model with vLLM and benchmark concurrent throughput
  - Exercise: tune batch size and sequence length for maximum throughput

#### Section 11.6: Compilation-Based Optimization
- **Content:** torch.compile (inductor), Triton kernels, TensorRT-LLM, ONNX Runtime, graph fusion, kernel auto-tuning
- **Elements needed:**
  - Code: compile a model with torch.compile and benchmark
  - Exercise: profile a model to find the top-3 kernels to optimize

**Worked Example:** Implement speculative decoding for a quantized 7B model. Benchmark tokens-per-second at different temperatures and sequence lengths. Compare to baseline autoregressive decoding and to a compiled baseline.

---

### Chapter 12: Deployment

**Prerequisites:** Chapter 10, 11

#### Section 12.1: API Design for LLMs
- **Content:** REST API design, request/response schemas, OpenAI-compatible API, streaming responses (SSE), batch endpoints, function calling interfaces
- **Elements needed:**
  - Code: build a FastAPI server for a local model
  - Figure: request/response flow for streaming inference
  - Exercise: implement an OpenAI-compatible API for a local model

#### Section 12.2: Serving Frameworks
- **Content:** vLLM (PagedAttention, high throughput), TGI (Text Generation Inference), Ollama (local inference), litserve (PyTorch-native), feature comparison, when to choose which
- **Elements needed:**
  - Table: serving framework comparison (features, supported models, quantization)
  - Code: deploy the same model with vLLM and TGI
  - Exercise: benchmark serving frameworks under concurrent load

#### Section 12.3: Containerization
- **Content:** Docker for ML, multi-stage builds, GPU-enabled containers (NVIDIA Container Toolkit), image size optimization, reproducibility
- **Elements needed:**
  - Code: Dockerfile for a model serving stack with CUDA
  - Exercise: optimize a Docker image size for a model deployment

#### Section 12.4: Cloud Deployment
- **Content:** AWS SageMaker, GCP Vertex AI, Azure ML, Hugging Face Inference Endpoints, bare metal vs managed, cost comparison, spot instances for inference
- **Elements needed:**
  - Code: deploy a model to a cloud inference endpoint
  - Table: cloud provider cost comparison for model serving
  - Exercise: estimate monthly hosting cost for a given throughput requirement

#### Section 12.5: Production Patterns
- **Content:** Load balancing, autoscaling, canary deployments, blue-green deployments, circuit breakers, retry policies, rate limiting, authentication, API key management, TLS
- **Elements needed:**
  - Figure: canary deployment rollout
  - Code: add auth and rate limiting to a model API
  - Exercise: design a rollout strategy for a model update

#### Section 12.6: Load Testing
- **Content:** Load testing methodologies, metrics (throughput, p50/p95/p99 latency, error rate), stress testing, soak testing, tools (k6, Locust, wrk)
- **Elements needed:**
  - Code: write a load test for a model API
  - Figure: latency percentile chart under increasing load
  - Exercise: find the breaking point of a model serving deployment

**Worked Example:** Containerize a quantized, instruction-tuned model and deploy behind an authenticated, rate-limited API. Run a load test simulating 100 concurrent users. Report throughput, p99 latency, and error rate.

---

### Chapter 13: Monitoring, Evaluation, and Iteration

**Prerequisites:** Chapter 12

#### Section 13.1: Production Evaluation
- **Content:** Online vs offline metrics, A/B testing for models, shadow deployments, champion-challenger, statistical significance, evaluation latency
- **Elements needed:**
  - Figure: A/B test architecture for model endpoints
  - Code: design an A/B test infrastructure
  - Exercise: calculate the sample size needed for a statistically significant A/B test

#### Section 13.2: Drift Detection
- **Content:** Prompt drift, output drift, data drift, concept drift, detection methods (KL divergence, PSI, population stability index), alerting thresholds
- **Elements needed:**
  - Code: implement drift detection on API request logs
  - Figure: drift detection dashboard
  - Exercise: simulate a drift scenario and verify detection

#### Section 13.3: Observability
- **Content:** Request logging, latency percentiles, token usage metrics, cost tracking, error tracking, distributed tracing for model pipelines, dashboard design (Grafana, Datadog)
- **Elements needed:**
  - Code: build a metrics dashboard for a model serving stack
  - Exercise: design an alerting rule for latency degradation

#### Section 13.4: Continuous Learning
- **Content:** Feedback collection from production, human-in-the-loop, active learning, incremental fine-tuning, online learning, data flywheel
- **Elements needed:**
  - Figure: data flywheel diagram
  - Code: design a feedback collection pipeline
  - Exercise: implement an active learning loop for data collection

#### Section 13.5: Catastrophic Forgetting
- **Content:** Why models forget, continual learning, replay buffers, elastic weight consolidation, incremental fine-tuning, multi-task fine-tuning
- **Elements needed:**
  - Code: measure forgetting after incremental fine-tuning
  - Exercise: apply replay-based continual learning to mitigate forgetting

#### Section 13.6: Experiment Tracking
- **Content:** MLflow, Weights & Biases, experiment metadata, parameter tracking, artifact storage, model registry, rollback
- **Elements needed:**
  - Code: set up MLflow for tracking fine-tuning experiments
  - Exercise: compare three experiment tracking runs side-by-side

**Worked Example:** Build a monitoring dashboard for a deployed model. Implement drift detection on real API traffic. Design a retraining trigger based on quality degradation thresholds.

---

### Chapter 14: Capstone

**Prerequisites:** All preceding chapters

#### Section 14.1: Project Design
- **Content:** Choosing a domain, defining success criteria, selecting a base model, scoping the dataset, risk assessment, timeline
- **Elements needed:**
  - Template: project proposal template
  - Exercise: write a project proposal

#### Section 14.2: End-to-End Implementation Guide
- **Content:** Integrating data pipeline, training, quantization, and deployment into a single workflow, CI/CD for ML models, reproducibility, documentation
- **Elements needed:**
  - Figure: end-to-end pipeline architecture
  - Code: skeleton repository structure for the capstone
  - Exercise: implement one stage of the pipeline

#### Section 14.3: Evaluation and Reporting
- **Content:** Comprehensive evaluation methodology, benchmarking against baseline, writing a technical report, presenting findings, open-sourcing considerations
- **Elements needed:**
  - Template: technical report structure
  - Exercise: write an abstract for the capstone report

#### Section 14.4: Production Readiness Checklist
- **Content:** Model card, security audit, performance benchmark, disaster recovery, cost estimation, monitoring plan, rollback procedure
- **Elements needed:**
  - Checklist: production readiness items
  - Exercise: complete a production readiness checklist for the capstone

**Worked Example:** A complete capstone project demonstrating a fine-tuned model for a real-world task, from data to deployment, with full documentation and evaluation.

---

## Appendix

### A: Math Reference
- Vector and matrix operations, probability distributions, calculus basics, information theory (entropy, cross-entropy, KL divergence)

### B: PyTorch Quick Reference
- Tensor operations, autograd, nn.Module, DataLoader, common layers, GPU memory management

### C: GPU Hardware Guide
- GPU architecture overview, VRAM vs system memory, CUDA cores, tensor cores, interconnect (NVLink, PCIe), choosing a GPU for training vs inference

### D: Glossary
- Terms defined across all chapters, organized alphabetically

### E: Commands and Tooling Quick Reference
- Common commands for training, quantization, serving, and debugging with output examples
