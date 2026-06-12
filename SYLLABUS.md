# CS601 Syllabus: Post-Training of Large Language Models

## Schedule Overview

This is a 14-week self-guided course. Each week is a module with lecture notes, labs, assignments, and a quiz. Work at your own pace, but maintain the order -- each module builds on the last.

### Part I: Foundations (Weeks 1-4)

#### Week 1: Machine Learning Refresher

**Goal:** Bring a general CS background up to speed on ML fundamentals needed for the rest of the course.

| Day | Topic | Lab |
|---|---|---|
| 1 | Course orientation; what is machine learning; supervised vs unsupervised vs reinforcement learning | Set up the development environment (Python, PyTorch, CUDA) |
| 2 | Linear algebra for ML: vectors, matrices, tensor operations, eigenvalues | NumPy/PyTorch tensor manipulation exercises |
| 3 | Probability and statistics: distributions, expectation, variance, Bayes' rule, maximum likelihood | Implement MLE and MAP estimation from scratch |
| 4 | Gradient descent and variants: SGD, Momentum, Adam; loss landscapes, learning rate schedules | Train a linear regression and logistic regression from scratch |
| 5 | Overfitting, underfitting, regularization; train/val/test splits; cross-validation | Experiment with L2 regularization and early stopping |
| 6 | Backpropagation and automatic differentiation | Implement backprop through a small computation graph |
| 7 | Quiz 1 | -- |

**Assignment 1:** Implement a multi-layer perceptron from scratch (no frameworks) that classifies MNIST digits, with Adam optimizer, L2 regularization, and early stopping.

---

#### Week 2: Deep Learning & Neural Networks

**Goal:** Understand the building blocks of deep neural networks that feed into Transformer architectures.

| Day | Topic | Lab |
|---|---|---|
| 1 | Multi-layer perceptrons; activation functions (ReLU, GeLU, SiLU); why depth matters | Build and compare activation functions on a simple task |
| 2 | Convolutional neural networks; why CNNs are not the default for text | Implement a 1D conv layer and compare to linear |
| 3 | Recurrent neural networks; vanishing gradients; why RNNs struggle with long sequences | Implement an RNN language model on a small corpus |
| 4 | Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) | Train an LSTM on sequence prediction |
| 5 | Sequence-to-sequence models; encoder-decoder architecture; the attention breakthrough | Build a seq2seq model with and without attention |
| 6 | Normalization layers: LayerNorm, BatchNorm, RMSNorm and when to use each | Compare normalization strategies on a small network |
| 7 | Quiz 2 | -- |

**Assignment 2:** Build a character-level language model using LSTM, then replace with a positional-encoding + self-attention layer and compare perplexity.

---

#### Week 3: The Transformer Architecture

**Goal:** Derive the Transformer from first principles and understand every component.

| Day | Topic | Lab |
|---|---|---|
| 1 | The Transformer paper (Vaswani et al., 2017); why self-attention replaced recurrence | Walk through the original paper's equations |
| 2 | Scaled dot-product attention: derivation, the scaling factor, softmax | Implement attention from scratch and visualize attention weights |
| 3 | Multi-head attention; why parallel attention heads help; projection matrices | Implement multi-head attention and compare single vs multi-head |
| 4 | Positional encodings: sinusoidal, learned, RoPE, ALiBi | Compare positional encoding strategies on a small model |
| 5 | Encoder and decoder blocks; residual connections; feed-forward layers | Build a minimal Transformer from scratch |
| 6 | Attention optimization: sparse attention, FlashAttention, memory I/O bottlenecks | Run a FlashAttention vs naive attention benchmark |
| 7 | Quiz 3 | -- |

**Assignment 3:** Implement a minimal GPT-style Transformer (decoder-only) and train it on a small text corpus. Report perplexity vs number of layers and attention heads.

---

#### Week 4: Scaling Laws, Mixture-of-Experts, and Base Model Training

**Goal:** Understand how Transformers scale to billions of parameters and the paradigm shift introduced by MoE.

| Day | Topic | Lab |
|---|---|---|
| 1 | Compute-optimal scaling laws (Kaplan et al.); loss vs model size, data, compute | Plot scaling law curves from published data |
| 2 | Chinchilla laws (Hoffmann et al.); token-to-parameter ratio; training compute-optimal models | Analyze the Chinchilla dataset and find optimal model size for a given budget |
| 3 | Mixture-of-Experts motivation; dense vs sparse; routing, gating, capacity factor | Implement a toy MoE layer and compare to dense |
| 4 | MoE architectures: Switch Transformer, GShard, Mixtral, Grok | Compare Mixtral expert routing to Switch Transformers |
| 5 | Base model pre-training: objectives (next-token prediction), data mixtures, curriculum learning | Inspect a pre-training data mixture and its class distribution |
| 6 | Loss landscapes at scale; learning rate warmup, cosine decay, gradient clipping | Run a learning rate sweep on a small pre-training run |
| 7 | Transformers wrap-up and Part I review | -- |
| 8 | Quiz 4 (covers Weeks 1-4) | -- |

**Assignment 4:** Write a technical report comparing dense and MoE architectures. Include a working MoE layer, ablation studies on expert count and routing strategy, and a discussion of when MoE is (and is not) worth the complexity.

---

### Part II: Data & Training (Weeks 5-9)

#### Week 5: Datasets for LLMs

**Goal:** Understand the datasets used to train and fine-tune LLMs, and how to evaluate their quality.

| Day | Topic | Lab |
|---|---|---|
| 1 | Pre-training corpora: Common Crawl, The Pile, RefinedWeb, C4 | Download and inspect a subset of The Pile |
| 2 | Instruction datasets: Alpaca, flan, Dolly, UltraChat, Helpfulness/Harmlessness | Compare instruction formats across three datasets |
| 3 | Preference datasets: HH-RLHF, Anthropic's helpful-harmless, DPO datasets | Explore the structure of a preference dataset |
| 4 | Dataset quality metrics: toxicity, duplication, language mix, domain distribution | Build a quality report for a given dataset |
| 5 | Synthetic data generation: self-instruct, evol-instruct, data augmentation with LLMs | Generate a synthetic instruction dataset using an API model |
| 6 | Data licensing and provenance; open weights vs open data; copyright considerations | Audit the license compatibility of a dataset mixture |
| 7 | Quiz 5 | -- |

**Assignment 5:** Profile three instruction-tuning datasets for quality. Produce a report with duplication rates, toxicity scores, instruction diversity, and a recommendation for which to use for a given domain.

---

#### Week 6: Data Engineering Pipelines

**Goal:** Build production-grade data pipelines that ingest, clean, filter, and format training data.

| Day | Topic | Lab |
|---|---|---|
| 1 | Data pipeline architecture: extraction, transformation, loading; batch vs streaming | Design a data pipeline DAG for a fine-tuning dataset |
| 2 | Text cleaning: HTML stripping, normalization, encoding fixes, PII redaction | Implement a text cleaning pipeline and measure its impact |
| 3 | Document deduplication: exact (hash-based), fuzzy (minhash, SimHash), near-dedup | Implement minhash-based dedup and measure duplicate removal rate |
| 4 | Quality filtering: heuristic filters, perplexity-based filtering, model-based scoring | Build a quality filter using a pre-trained model's perplexity as a signal |
| 5 | Dataset formatting: instruction format templates, system/user/assistant roles | Convert a raw dataset into three instruction formats and compare token counts |
| 6 | Data versioning and reproducibility: DVC, dataset cards, provenance tracking | Set up DVC for a dataset and create a dataset card |
| 7 | Quiz 6 | -- |

**Assignment 6:** Build an end-to-end data pipeline that takes a raw web crawl dump and produces a cleaned, deduplicated, quality-scored instruction dataset. Document the data loss at each pipeline stage.

---

#### Week 7: Tokenization

**Goal:** Understand how text becomes numbers, the tradeoffs in tokenization strategies, and how to build a vocabulary for a domain-specific model.

| Day | Topic | Lab |
|---|---|---|
| 1 | Word-level tokenization and the open-vocabulary problem | Compare word-level vs subword on a held-out corpus |
| 2 | Byte-Pair Encoding (BPE): algorithm, merge rules, vocabulary size tradeoffs | Implement BPE from scratch |
| 3 | WordPiece, Unigram, SentencePiece; comparator | Train a WordPiece tokenizer and compare to BPE on the same corpus |
| 4 | Tokenizer metrics: vocabulary size, byte coverage, bpe/character ratio, OOV rate | Benchmark five tokenizers on a domain-specific corpus |
| 5 | Domain-adaptive pre-training and vocabulary extension | Extend a base tokenizer with domain-specific tokens and measure OOV reduction |
| 6 | Tokenization edge cases: multilingual text, code, math, special tokens | Stress-test a tokenizer on code, math notation, and emoji |
| 7 | Quiz 7 | -- |

**Assignment 7:** Train a custom BPE tokenizer on a domain-specific corpus. Compare its efficiency to the base model's tokenizer. Document the impact of vocabulary size on model size and perplexity.

---

#### Week 8: Instruction Tuning

**Goal:** Learn the primary methods for adapting a base model to follow instructions, with a focus on parameter-efficient fine-tuning.

| Day | Topic | Lab |
|---|---|---|
| 1 | Full fine-tuning vs parameter-efficient fine-tuning; compute and storage tradeoffs | Compare full FT memory usage to PEFT on a small model |
| 2 | LoRA: low-rank adaptation, target modules, rank r, alpha, dropout | Apply LoRA to a 1B-parameter model and measure task performance |
| 3 | QLoRA: 4-bit quantized LoRA; NF4 precision, paged optimizers, double quantization | Fine-tune a 7B model with QLoRA on a single GPU |
| 4 | DoRA, AdaLoRA, BoFT, IA3, prefix tuning, prompt tuning | Compare LoRA vs DoRA on the same task and budget |
| 5 | Training configuration: learning rate, batch size, gradient accumulation, epochs | Run a hyperparameter sweep on LoRA rank and learning rate |
| 6 | Evaluation: instruction-following benchmarks (IFEval, AlpacaEval), perplexity on held-out data | Evaluate a fine-tuned model on IFEval and compare to the base model |
| 7 | Quiz 8 | -- |

**Assignment 8:** Instruction-tune a 7B model using QLoRA on a custom dataset. Compare the fine-tuned model to the base model on instruction-following benchmarks. Ablate LoRA rank and learning rate.

---

#### Week 9: Alignment

**Goal:** Align model outputs to human preferences using reinforcement learning and preference optimization.

| Day | Topic | Lab |
|---|---|---|
| 1 | The alignment problem; why instruction tuning is not enough; reward modeling | Train a reward model on a preference dataset |
| 2 | Reinforcement Learning from Human Feedback (RLHF): overview, PPO, the four-model system | Walk through the RLHF pipeline with a reference implementation |
| 3 | Proximal Policy Optimization (PPO): clipping, advantage estimation, KL penalty | Fine-tune a model with PPO on a simple preference task |
| 4 | Direct Preference Optimization (DPO): derivation, simplifications, BRDPO, IPO | Apply DPO to a fine-tuned model and compare to PPO |
| 5 | Constitutional AI; self-alignment; red-teaming and safety fine-tuning | Apply a constitutional AI prompt to self-critique model outputs |
| 6 | Evaluating alignment: harmlessness benchmarks (XSTest, TruthfulQA), refusal behavior | Benchmark alignment before and after DPO fine-tuning |
| 7 | Quiz 9 | -- |

**Assignment 9:** Take the model from Assignment 8 and align it using DPO. Evaluate on harmlessness and helpfulness benchmarks. Write a comparison of DPO vs PPO for this use case.

---

### Part III: Optimization & Deployment (Weeks 10-14)

#### Week 10: Quantization

**Goal:** Compress fine-tuned models for efficient inference without sacrificing quality.

| Day | Topic | Lab |
|---|---|---|
| 1 | Why quantize: memory, latency, throughput, cost; precision formats (FP32, FP16, BF16, FP8, INT8, INT4) | Measure the memory and latency of a model at different precisions |
| 2 | Post-training quantization (PTQ): weight-only, activations, per-channel, per-tensor | Apply PTQ to a model and measure accuracy retention |
| 3 | Quantization-aware training (QAT): fake quantization, straight-through estimator | Run QAT on a small model and compare to PTQ |
| 4 | GPTQ, AWQ, SmoothQuant; algorithmic quantization for Transformers | Quantize a 7B model with GPTQ and benchmark quality/latency |
| 5 | 4-bit inference: GGUF, EXL2, bitsandbytes; ecosystem and tooling | Serve a 4-bit quantized model and compare to FP16 |
| 6 | Quantized fine-tuning: training on quantized weights; QLoRA revisited at inference time | Fine-tune a 4-bit model and evaluate the quantize-then-fine-tune vs fine-tune-then-quantize pipeline |
| 7 | Quiz 10 | -- |

**Assignment 10:** Quantize the aligned model from Assignment 9 to INT8 and INT4 using three methods (PTQ, GPTQ, AWQ). Benchmark each for accuracy, latency, and memory. Recommend a quantization strategy for production.

---

#### Week 11: Speculative Decoding and Inference Optimization

**Goal:** Speed up model inference using speculative decoding and other optimization techniques.

| Day | Topic | Lab |
|---|---|---|
| 1 | Autoregressive decoding bottlenecks; why inference is sequential and slow | Measure the latency breakdown of autoregressive generation |
| 2 | Speculative decoding: draft model, verification, acceptance rate, speedup theory | Implement speculative decoding with a small draft model |
| 3 | Draft model selection: smaller model, echo decoding, lookback decoding | Compare three draft model strategies on the same target model |
| 4 | Speculative decoding variants: speculative token sampling, medusa, multi-token prediction | Benchmark multi-token prediction vs standard speculative decoding |
| 5 | KV cache optimization: paged attention (vLLM), continuous batching, chunked prefill | Serve a model with vLLM and benchmark throughput under concurrent load |
| 6 | Compilation-based optimization: torch.compile, Triton, TensorRT-LLM | Compile a model with torch.compile and measure latency improvement |
| 7 | Quiz 11 | -- |

**Assignment 11:** Implement speculative decoding for the quantized model from Assignment 10. Benchmark tokens-per-second and latency at different temperatures and sequence lengths. Compare to baseline autoregressive decoding.

---

#### Week 12: Deployment

**Goal:** Package and serve a fine-tuned, quantized model as a production API.

| Day | Topic | Lab |
|---|---|---|
| 1 | Serving architectures: REST API, gRPC, streaming SSE; request/response design | Build a REST API for a local model using FastAPI |
| 2 | Serving frameworks: vLLM, TGI, Ollama, litserve; feature comparison | Deploy the same model with vLLM and TGI and compare |
| 3 | Containerization: Docker, multi-stage builds, GPU-enabled containers | Containerize a model serving stack with CUDA dependencies |
| 4 | Cloud deployment: AWS SageMaker, GCP Vertex AI, Hugging Face Inference Endpoints | Deploy a model to a cloud inference endpoint |
| 5 | Load balancing, autoscaling, canary deployments | Set up autoscaling for a model endpoint under variable load |
| 6 | Rate limiting, authentication, API key management | Add auth and rate limiting to a model API |
| 7 | Quiz 12 | -- |

**Assignment 12:** Containerize the optimized model and deploy it behind an authenticated, rate-limited API on a cloud provider. Write a load test report showing throughput, p99 latency, and error rate under concurrent users.

---

#### Week 13: Monitoring, Evaluation, and Iteration

**Goal:** Maintain model quality in production and iterate based on real-world feedback.

| Day | Topic | Lab |
|---|---|---|
| 1 | Production evaluation: online vs offline metrics, A/B testing, shadow deployments | Design an A/B test for two model versions |
| 2 | drift detection: prompt drift, output drift, data drift | Implement a drift detection pipeline on API logs |
| 3 | Logging and observability: request logging, latency percentiles, token usage metrics | Build a dashboard for model serving metrics |
| 4 | Feedback collection and continuous learning: collecting human feedback in production | Design a feedback collection pipeline from API users |
| 5 | Catastrophic forgetting; continual learning; incremental fine-tuning | Fine-tune on a new domain and measure forgetting on the original task |
| 6 | Model versioning and rollback; experiment tracking with MLflow | Set up MLflow for tracking fine-tuning experiments |
| 7 | Quiz 13 | -- |

**Assignment 13:** Build a monitoring dashboard for the deployed model. Implement drift detection on real API traffic and design a retraining trigger based on quality degradation.

---

#### Week 14: Capstone Project

**Goal:** Integrate everything into a single, end-to-end project.

| Day | Topic | Deliverable |
|---|---|---|
| 1 | Project scoping: choose a domain, define the task, select a base model | Project proposal with scope, base model, and evaluation criteria |
| 2 | Data pipeline: ingest, clean, deduplicate, quality-score, format | Cleaned and formatted training dataset with dataset card |
| 3 | Tokenization: evaluate base tokenizer, optionally extend vocabulary | Tokenization report with OOV analysis |
| 4 | Instruction tuning: fine-tune with LoRA/QLoRA | Fine-tuned model checkpoint with training logs |
| 5 | Alignment: apply DPO or constitutional AI | Aligned model with before/after evaluation |
| 6 | Quantize and optimize: quantize, add speculative decoding | Quantized model with benchmark report |
| 7 | Deploy: containerize, serve behind API, monitor | Live API endpoint with monitoring dashboard |

**Capstone Deliverable:** A complete repository containing:
- Data pipeline code and processed dataset
- Training configuration and model checkpoint
- Quantization and benchmarking scripts
- Dockerfile and deployment configuration
- API endpoint with authentication and rate limiting
- Monitoring dashboard
- Technical report documenting all design decisions, tradeoffs, and evaluation results

---

## Grading

This is a self-guided course. Each module has:

- **Lecture Notes** - Read and understand
- **Labs** - Complete all hands-on exercises
- **Assignment** - Implement and write up
- **Quiz** - Self-assessment

Aim to complete all labs and assignments before moving to the next module. Quizzes are self-graded; if you score below 80%, revisit the material.

## Environment

- **Language:** Python 3.11+
- **Framework:** PyTorch 2.x with CUDA support
- **Hardware:** Single GPU with 16GB+ VRAM minimum (DGX Spark or equivalent recommended)
- **Libraries:** transformers, peft, bitsandbytes, accelerate, vLLM, trl, datasets, tokenizers

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
- Tobin et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- Rafailov et al., "Direct Preference Optimization" (2023)
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022)
- Jiang et al., "Mixtral of Experts" (2024)
- Levi et al., "Fast Inference from Transformers through Speculative Decoding" (2023)
