# CS601 — Master Task Plan

56 independent subtasks, each created on its own git branch and merged into `main` when done.
Tasks are ordered so that a later task never depends on an earlier one being merged first — they only share content references (e.g., a module references its textbook chapter).

---

## Conventions

| Item | Rule |
|---|---|
| **Branch naming** | `task/<id>-<short-slug>` (e.g. `task/t01-ch01-ml-fundamentals`) |
| **Diagrams** | Every figure callout is rendered as an inline **Mermaid** block (```mermaid ... ```) or a hand-drawn **ASCII-art** figure. Never describe a diagram without also rendering it. |
| **Code** | Inline fenced Python blocks inside the markdown; lab scripts live as standalone `.py` files in the `labs/` directory. |
| **Definition of Done** | Listed per task below. A task is not merged until every bullet is satisfied. |
| **License** | Every new file includes `Licensed under Apache 2.0` in the header comment. |

---

## Phase 1 — Textbook (T01–T14)

The textbook is authored chapter-by-chapter. Each chapter task creates a single large markdown file following the authoring convention in `textbook/OUTLINE.md`: Learning Objectives, Prerequisites, Core Content (all sections), Worked Example, Exercises, Summary, Further Reading. All figures are rendered; all code blocks are complete and runnable.

### T01 — Chapter 1: Machine Learning Fundamentals

- **Branch:** `task/t01-ch01-ml-fundamentals`
- **Files created:**
  - `textbook/chapter_01_ml_fundamentals.md`
- **Sections authored:** 1.1 What Is ML, 1.2 Linear Algebra, 1.3 Probability & Statistics, 1.4 Optimization, 1.5 Model Evaluation
- **Diagrams to render:**
  - Mermaid taxonomy tree of ML tasks (supervised / unsupervised / RL with sub-types)
  - ASCII figure: matrix multiplication as linear transformation (2D grid → warped grid)
  - ASCII figure: common distributions (Gaussian, Bernoulli, Categorical) bell/shape sketches with labeled mean/variance
  - Mermaid flowchart: loss landscape with optimizer trajectories (SGD zigzag → Momentum → Adam smooth convergence)
  - ASCII figure: learning curves (train vs val loss) for underfit / good fit / overfit
- **Code blocks:** linear regression on toy dataset; NumPy SVD; MLE for Gaussian; SGD/Momentum/Adam from scratch; k-fold CV
- **Definition of done:**
  - All 5 sections written with prose, equations, and code
  - All 5 diagrams rendered inline (Mermaid or ASCII)
  - Worked Example: regularized logistic regression with CV and early stopping — complete, runnable code
  - Exercises section with ≥3 problems per section (easy/medium/hard)
  - Summary and Further Reading present

---

### T02 — Chapter 2: Deep Learning Building Blocks

- **Branch:** `task/t02-ch02-dl-building-blocks`
- **Files created:**
  - `textbook/chapter_02_dl_building_blocks.md`
- **Sections authored:** 2.1 Neural Networks, 2.2 CNNs, 2.3 RNNs, 2.4 LSTMs & GRUs, 2.5 Seq2Seq, 2.6 Normalization
- **Diagrams to render:**
  - ASCII figure: MLP forward pass (input → hidden1 → hidden2 → output) with activation labels at each layer
  - ASCII figure: 1D convolution step-by-step (kernel sliding over input vector with stride/padding annotations)
  - Mermaid state diagram: RNN unrolled through 6 time steps with shared weight arrows
  - ASCII figure: LSTM cell with input/forget/output gates, cell state highway, and gate equations
  - ASCII figure: seq2seq encoder-decoder with and without attention (side-by-side)
  - ASCII figure: BatchNorm vs LayerNorm computation graphs (which axes are normalized)
- **Code blocks:** MLP forward/backward from scratch; 1D conv; RNN language model; LSTM from scratch; seq2seq with attention; LayerNorm and RMSNorm
- **Definition of done:**
  - All 6 sections written
  - All 6 diagrams rendered
  - Worked Example: LSTM machine translation model → add attention → BLEU comparison
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T03 — Chapter 3: The Transformer Architecture

- **Branch:** `task/t03-ch03-transformer`
- **Files created:**
  - `textbook/chapter_03_transformer.md`
- **Sections authored:** 3.1 Transformer Paper, 3.2 Scaled Dot-Product Attention, 3.3 Multi-Head Attention, 3.4 Positional Encodings, 3.5 Transformer Blocks, 3.6 Attention Variants & FlashAttention
- **Diagrams to render:**
  - Mermaid diagram: full Transformer architecture (stacked encoder → stacked decoder → linear + softmax)
  - ASCII figure: attention computation as matrix ops (Q × K^T → scale → softmax → × V)
  - ASCII figure: multi-head attention with 4 heads showing parallel projection → attention → concatenate → output projection
  - ASCII figure: sinusoidal positional encoding table (row = position, col = dimension, sine/cosine waves)
  - ASCII figure: RoPE rotation visualization (query/key vectors rotated by position)
  - ASCII figure: decoder block with self-attention mask, cross-attention, FFN, and residual connections
  - ASCII figure: FlashAttention tiling (input split into blocks, compute order: S_ij → softmax → softmax_sum → accumulate)
- **Code blocks:** minimal Transformer; scaled dot-product attention; multi-head attention; sinusoidal/learned/RoPE positional encodings; Transformer decoder block; FlashAttention vs naive benchmark
- **Definition of done:**
  - All 6 sections written
  - All 7 diagrams rendered
  - Worked Example: GPT-2 style decoder trained on WikiText-2 with perplexity report
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T04 — Chapter 4: Scaling Laws, MoE, Pre-Training

- **Branch:** `task/t04-ch04-scaling-moe`
- **Files created:**
  - `textbook/chapter_04_scaling_moe.md`
- **Sections authored:** 4.1 Scaling Laws, 4.2 Emergent Abilities, 4.3 MoE Fundamentals, 4.4 MoE Architectures, 4.5 Base Model Pre-Training, 4.6 Training Infrastructure
- **Diagrams to render:**
  - ASCII figure: scaling law curves (log-log plot of loss vs parameters, loss vs tokens, loss vs compute) with power-law fit lines
  - ASCII figure: emergent ability S-curves (capability vs model size showing discontinuous jumps)
  - ASCII figure: MoE layer vs dense FFN side-by-side (token → gate → top-k experts → weighted sum)
  - ASCII figure: Mixtral architecture (dense attention layers interleaved with MoE FFN layers, 8 experts, 2 active)
  - ASCII figure: Switch Transformer (single expert per token, auxiliary loss)
  - Mermaid pie chart: pre-training data mixture (web 45%, books 15%, code 15%, academic 15%, other 10%)
  - Mermaid diagram: data parallelism vs tensor parallelism vs pipeline parallelism (3-column comparison)
- **Code blocks:** power-law fit to published data; MoE layer with top-k gating; Switch Transformer layer; pre-training data mixture analysis; FSDP multi-GPU config
- **Definition of done:**
  - All 6 sections written
  - All 7 diagrams rendered
  - Worked Example: 3B model pre-training plan on 8× A100
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T05 — Chapter 5: Datasets for LLMs

- **Branch:** `task/t05-ch05-datasets`
- **Files created:**
  - `textbook/chapter_05_datasets.md`
- **Sections authored:** 5.1 Pre-Training Corpora, 5.2 Instruction Datasets, 5.3 Preference Datasets, 5.4 Quality Assessment, 5.5 Synthetic Data, 5.6 Data Licensing
- **Diagrams to render:**
  - Mermaid diagram: preference data collection pipeline (annotators → pairwise comparison → chosen/rejected → dataset)
  - ASCII figure: quality scoring pipeline (raw docs → toxicity filter → duplication filter → language filter → quality score → ranked output)
  - ASCII figure: self-instruct generation loop (seed tasks → API model → generated instructions → filter → augment seed)
- **Tables:** pre-training corpus comparison (5 corpora × size/domains/license); instruction dataset comparison (5 datasets × size/format/license); data license permissions matrix (CC-BY, CC0, ODC-BY, Apache 2.0)
- **Code blocks:** download/inspect Pile subset; convert instruction formats; quality scoring report; synthetic instruction generation; license audit script
- **Definition of done:**
  - All 6 sections written
  - All 3 diagrams rendered; all 3 tables rendered
  - Worked Example: quality profile of Alpaca, Dolly, UltraChat with dataset card
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T06 — Chapter 6: Data Engineering Pipelines

- **Branch:** `task/t06-ch06-data-pipelines`
- **Files created:**
  - `textbook/chapter_06_data_pipelines.md`
- **Sections authored:** 6.1 Pipeline Architecture, 6.2 Text Cleaning, 6.3 Deduplication, 6.4 Quality Filtering, 6.5 Formatting, 6.6 Versioning
- **Diagrams to render:**
  - Mermaid DAG: data pipeline stages (extract → clean → dedup → filter → format → version) with error handling branches
  - ASCII figure: minhash + LSH algorithm (document → shingles → minhash signatures → band comparison → candidate pairs → verified duplicates)
  - ASCII figure: quality filter cascade (funnel diagram showing document count at each stage: raw → cleaned → deduped → filtered → formatted)
- **Tables:** cleaning operations and data size impact; instruction format comparison (Alpaca, ChatML, ShareGPT) with example output
- **Code blocks:** fault-tolerant pipeline with retries; text cleaning pipeline; minhash near-dedup; multi-stage quality filter; instruction formatters; DVC setup
- **Definition of done:**
  - All 6 sections written
  - All 3 diagrams rendered; all 2 tables rendered
  - Worked Example: end-to-end pipeline from Common Crawl dump to instruction dataset with stage-by-stage data volume
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T07 — Chapter 7: Tokenization

- **Branch:** `task/t07-ch07-tokenization`
- **Files created:**
  - `textbook/chapter_07_tokenization.md`
- **Sections authored:** 7.1 Tokenization Problem, 7.2 BPE, 7.3 WordPiece/Unigram/SentencePiece, 7.4 Metrics, 7.5 Domain-Adaptive, 7.6 Edge Cases
- **Diagrams to render:**
  - ASCII figure: word-level vs subword tokenization on the same sentence ("transformers are amazing!") showing token count and OOV
  - ASCII figure: BPE merge steps (character-level → first merge → ... → final tokens) with merge table
  - Mermaid radar chart: tokenizer metrics (vocab size, byte coverage, compression, OOV rate, speed, multilingual)
- **Tables:** BPE vs WordPiece vs Unigram comparison (algorithm, training objective, pros/cons)
- **Code blocks:** word-level tokenizer with OOV measurement; BPE from scratch; WordPiece training and comparison; tokenizer benchmark suite; vocabulary extension; edge-case stress test
- **Definition of done:**
  - All 6 sections written
  - All 3 diagrams rendered; all 1 table rendered
  - Worked Example: custom BPE on biomedical corpus vs GPT-4 tokenizer
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T08 — Chapter 8: Instruction Tuning

- **Branch:** `task/t08-ch08-instruction-tuning`
- **Files created:**
  - `textbook/chapter_08_instruction_tuning.md`
- **Sections authored:** 8.1 Base Model to Assistant, 8.2 Full FT, 8.3 LoRA, 8.4 QLoRA, 8.5 Beyond LoRA, 8.6 Training Config, 8.7 Evaluation
- **Diagrams to render:**
  - ASCII figure: base model continuation vs instruction-tuned completion on the same prompt
  - ASCII figure: LoRA weight perturbation (pre-trained W → W + ΔW = W + BA, where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k))
  - ASCII figure: QLoRA memory breakdown (4-bit weights + paged optimizer + double quantization scalars) vs full FT
  - ASCII figure: learning curves for different LoRA ranks and learning rates (5 curves on one plot)
- **Tables:** memory requirements for full FT (1B, 7B, 13B, 70B × FP16/FP32 optimizer states); PEFT method comparison (LoRA, QLoRA, DoRA, IA³, prefix, prompt × trainable params/quality/speed/memory)
- **Code blocks:** SFT data preparation; full FT with GPU memory tracking; LoRA fine-tuning; QLoRA on 7B; LoRA vs DoRA comparison; hyperparameter sweep; IFEval evaluation
- **Definition of done:**
  - All 7 sections written
  - All 4 diagrams rendered; all 2 tables rendered
  - Worked Example: QLoRA fine-tune Mistral-7B on customer service dataset with benchmark comparison
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T09 — Chapter 9: Alignment

- **Branch:** `task/t09-ch09-alignment`
- **Files created:**
  - `textbook/chapter_09_alignment.md`
- **Sections authored:** 9.1 Alignment Problem, 9.2 Reward Modeling, 9.3 RLHF, 9.4 DPO, 9.5 Constitutional AI, 9.6 Evaluating Alignment
- **Diagrams to render:**
  - ASCII figure: instruction-tuned vs aligned model behavior (side-by-side response comparison on sensitive prompts)
  - ASCII figure: reward model training pipeline (preference pairs → Bradley-Terry loss → reward model)
  - Mermaid diagram: full RLHF pipeline (policy model ← PPO ← reward model + value model → environment → human feedback)
  - ASCII figure: DPO vs RLHF architecture comparison (DPO: 2 models vs RLHF: 4 models)
- **Code blocks:** reward model training; PPO fine-tuning walkthrough; DPO application; self-critique constitution loop; XSTest/TruthfulQA benchmark
- **Definition of done:**
  - All 6 sections written
  - All 4 diagrams rendered
  - Worked Example: DPO alignment on Ch8 model with harmlessness/helpfulness comparison
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T10 — Chapter 10: Quantization

- **Branch:** `task/t10-ch10-quantization`
- **Files created:**
  - `textbook/chapter_10_quantization.md`
- **Sections authored:** 10.1 Why Quantize, 10.2 PTQ, 10.3 QAT, 10.4 Algorithmic Quantization, 10.5 4-Bit Ecosystem, 10.6 Quantized Fine-Tuning
- **Diagrams to render:**
  - ASCII figure: quantization mapping (float32 weight distribution → min/max clipping → uniform int8 buckets → dequantized)
  - ASCII figure: QAT training loop (forward → fake quant → loss → straight-through estimator → backward)
  - ASCII figure: GPTQ layer-by-layer quantization (Hessian computation → greedy ordering → quantize → residual compensation → next layer)
  - ASCII figure: quantize-then-FT vs FT-then-quant pipeline (two parallel timelines showing when quantization happens)
- **Tables:** model size at different precisions (1B, 7B, 13B, 70B × FP32/FP16/INT8/INT4); GGUF quant preset comparison (q4_0, q4_1, q4_k_m, q5_0, q8_0 × size/speed/quality)
- **Code blocks:** memory/latency measurement at precisions; PTQ with accuracy measurement; QAT vs PTQ; GPTQ and AWQ quantization; GGUF conversion; quantize-then-FT vs FT-then-quant comparison
- **Definition of done:**
  - All 6 sections written
  - All 4 diagrams rendered; all 2 tables rendered
  - Worked Example: 7B model quantized to INT8/INT4 via PTQ/GPTQ/AWQ with benchmark report
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T11 — Chapter 11: Speculative Decoding & Inference Optimization

- **Branch:** `task/t11-ch11-speculative-decoding`
- **Files created:**
  - `textbook/chapter_11_speculative_decoding.md`
- **Sections authored:** 11.1 Inference Bottleneck, 11.2 Speculative Decoding, 11.3 Draft Models, 11.4 Beyond Speculative, 11.5 KV Cache, 11.6 Compilation
- **Diagrams to render:**
  - ASCII figure: latency breakdown bar chart (prefill tokens 1-5: compute bound; decode tokens 6-N: memory bound per-token)
  - ASCII figure: speculative decoding step-by-step (draft model proposes 3 tokens → target verifies token 1 ✓, token 2 ✗ → accept token 1 → regenerate from token 2)
  - ASCII figure: acceptance rate vs speedup scatter plot (higher acceptance → higher speedup, with theoretical max line)
  - ASCII figure: Medusa multi-head prediction (single forward pass → 5 prediction heads → fan-out token candidates)
  - ASCII figure: paged attention vs contiguous KV cache (contiguous: fragmented blocks vs paged: non-contiguous physical blocks with block table)
- **Code blocks:** inference latency profiler; speculative decoding with small draft; 3 draft strategy comparison; multi-token prediction benchmark; vLLM concurrent throughput; torch.compile benchmark
- **Definition of done:**
  - All 6 sections written
  - All 5 diagrams rendered
  - Worked Example: speculative decoding on quantized 7B with tokens/sec benchmark at varying temperatures
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T12 — Chapter 12: Deployment

- **Branch:** `task/t12-ch12-deployment`
- **Files created:**
  - `textbook/chapter_12_deployment.md`
- **Sections authored:** 12.1 API Design, 12.2 Serving Frameworks, 12.3 Containerization, 12.4 Cloud Deployment, 12.5 Production Patterns, 12.6 Load Testing
- **Diagrams to render:**
  - Mermaid sequence diagram: streaming inference request/response flow (client → load balancer → serving container → model → SSE stream back)
  - ASCII figure: canary deployment rollout (100% v1 → 90% v1 + 10% v2 → 50/50 → 100% v2) with traffic arrows
  - ASCII figure: latency percentile chart under increasing load (p50/p95/p99 curves vs concurrent users)
- **Tables:** serving framework comparison (vLLM, TGI, Ollama, litserve × features/models/quantization); cloud provider cost comparison (SageMaker, Vertex AI, HF Endpoints × on-demand/spot/GPU types)
- **Code blocks:** FastAPI server; vLLM and TGI deployment; Dockerfile with CUDA; cloud endpoint deployment; auth and rate limiting; load test script
- **Definition of done:**
  - All 6 sections written
  - All 3 diagrams rendered; all 2 tables rendered
  - Worked Example: containerized quantized model behind authenticated API with 100-concurrent-user load test
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T13 — Chapter 13: Monitoring, Evaluation, and Iteration

- **Branch:** `task/t13-ch13-monitoring`
- **Files created:**
  - `textbook/chapter_13_monitoring.md`
- **Sections authored:** 13.1 Production Evaluation, 13.2 Drift Detection, 13.3 Observability, 13.4 Continuous Learning, 13.5 Catastrophic Forgetting, 13.6 Experiment Tracking
- **Diagrams to render:**
  - ASCII figure: A/B test architecture (traffic splitter → v1 endpoint, v2 endpoint → metrics collector → statistical analysis)
  - ASCII figure: drift detection dashboard (time series of prompt distribution KL divergence with alert threshold line)
  - Mermaid diagram: data flywheel (production → feedback collection → data pipeline → fine-tuning → deployment → production)
- **Code blocks:** A/B test infrastructure; drift detection on API logs; metrics dashboard; feedback collection pipeline; forgetting measurement; MLflow experiment tracking
- **Definition of done:**
  - All 6 sections written
  - All 3 diagrams rendered
  - Worked Example: monitoring dashboard with drift detection and retraining trigger
  - Exercises ≥3 per section
  - Summary and Further Reading

---

### T14 — Chapter 14: Capstone

- **Branch:** `task/t14-ch14-capstone`
- **Files created:**
  - `textbook/chapter_14_capstone.md`
- **Sections authored:** 14.1 Project Design, 14.2 End-to-End Implementation, 14.3 Evaluation & Reporting, 14.4 Production Readiness
- **Diagrams to render:**
  - Mermaid diagram: end-to-end pipeline architecture (data → tokenize → train → align → quantize → optimize → deploy → monitor)
  - ASCII figure: skeleton repository tree for the capstone project
- **Templates:** project proposal template; technical report structure; production readiness checklist
- **Definition of done:**
  - All 4 sections written
  - All 2 diagrams rendered
  - Project proposal template, technical report template, and production readiness checklist included
  - Exercises for each section
  - Summary and Further Reading

---

## Phase 2 — Appendices (T15–T19)

Appendix files are reference documents. They are independent of each other and of the module tasks.

### T15 — Appendix A: Math Reference

- **Branch:** `task/t15-appendix-math`
- **Files created:**
  - `textbook/appendix/math_reference.md`
- **Content:** Vector/matrix operations table, probability distributions table, calculus rules, information theory (entropy, cross-entropy, KL divergence) with derivations
- **Diagrams to render:** ASCII figure of entropy vs cross-entropy visualization (two distributions, shaded KL divergence region)
- **Definition of done:**
  - All four topics covered with formulas and worked examples
  - 1 diagram rendered
  - Cross-references to textbook chapters where each concept is first introduced

---

### T16 — Appendix B: PyTorch Quick Reference

- **Branch:** `task/t16-appendix-pytorch`
- **Files created:**
  - `textbook/appendix/pytorch_quick_reference.md`
- **Content:** Tensor operations cheat sheet, autograd essentials, nn.Module patterns, DataLoader usage, common layer constructors, GPU memory management tips
- **Code blocks:** runnable snippet for every operation/pattern documented
- **Definition of done:**
  - All six topics covered with code snippets
  - Every snippet is self-contained and runnable
  - Cross-references to textbook chapters

---

### T17 — Appendix C: GPU Hardware Guide

- **Branch:** `task/t17-appendix-gpu`
- **Files created:**
  - `textbook/appendix/gpu_hardware_guide.md`
- **Content:** GPU architecture overview, VRAM vs system memory, CUDA cores, tensor cores, NVLink vs PCIe, choosing GPU for training vs inference
- **Diagrams to render:**
  - ASCII figure: GPU architecture (SM → CUDA cores + tensor cores + shared memory + L2 cache → global memory)
  - ASCII figure: NVLink vs PCIe bandwidth comparison bar chart
- **Tables:** GPU comparison table (A100, H100, L40S, RTX 4090, DGX Spark × VRAM/bandwidth/TFLOPS)
- **Definition of done:**
  - All topics covered
  - 2 diagrams rendered; 1 comparison table
  - Cross-references to textbook chapters

---

### T18 — Appendix D: Glossary

- **Branch:** `task/t18-appendix-glossary`
- **Files created:**
  - `textbook/appendix/glossary.md`
- **Content:** Alphabetical glossary of all terms defined across Chapters 1–14. Each entry: term, one-sentence definition, chapter reference
- **Definition of done:**
  - ≥150 terms defined (collectively covering all chapters)
  - Every term has a chapter cross-reference
  - Sorted alphabetically

---

### T19 — Appendix E: Commands and Tooling Quick Reference

- **Branch:** `task/t19-appendix-commands`
- **Files created:**
  - `textbook/appendix/commands_tooling_reference.md`
- **Content:** Common commands for training (`torchrun`, `accelerate launch`), quantization (`optimum-cli`, `llama.cpp`), serving (`vllm serve`, `text-generation-server`), debugging (`nvtop`, `nvidia-smi dmon`, `py-spy`) with expected output examples
- **Definition of done:**
  - All four categories covered with ≥5 commands each
  - Each command includes flags, expected output, and common error messages
  - Cross-references to textbook chapters and module labs

---

## Phase 3 — Module Lecture Materials (T20–T33)

Each task creates lecture notes (`lecture.md`) and slides (`slides.md`) for one week. Lecture notes are detailed, prose-heavy reading material. Slides are a condensed, bulleted presentation format suitable for delivery.

### T20 — Week 1 Lecture: Machine Learning Refresher

- **Branch:** `task/t20-w01-lecture`
- **Files created:**
  - `modules/week_01_ml_refresher/lecture.md`
  - `modules/week_01_ml_refresher/slides.md`
- **Topics:** What is ML, linear algebra, probability, optimization, evaluation, backprop
- **Diagrams to render:**
  - Mermaid taxonomy of ML tasks (supervised/unsupervised/RL)
  - ASCII figure: gradient descent on a 2D loss surface (contour plot with descent arrows)
  - ASCII figure: computation graph for backprop (nodes = operations, edges = tensors with forward/backward arrows)
  - ASCII figure: learning curves (underfit/good/overfit)
- **Definition of done:**
  - `lecture.md`: ≥4000 words covering all 6 syllabus days with equations and code snippets
  - `slides.md`: ≥40 slides in bullet format, each slide ≤6 bullets, with speaker notes for complex topics
  - All 4 diagrams rendered in both files (inline in lecture, simplified in slides)
  - Cross-references to Textbook Chapter 1

---

### T21 — Week 2 Lecture: Deep Learning & Neural Networks

- **Branch:** `task/t21-w02-lecture`
- **Files created:**
  - `modules/week_02_deep_learning/lecture.md`
  - `modules/week_02_deep_learning/slides.md`
- **Topics:** MLPs, CNNs, RNNs, LSTMs/GRUs, seq2seq, normalization
- **Diagrams to render:**
  - ASCII figure: activation function zoo (sigmoid, tanh, ReLU, LeakyReLU, GeLU, SiLU curves on one plot)
  - ASCII figure: RNN unrolled through time with vanishing gradient annotation
  - ASCII figure: LSTM gate mechanics (input/forget/output with equations)
  - ASCII figure: seq2seq encoder-decoder with attention
- **Definition of done:**
  - `lecture.md`: ≥4000 words, all 6 days covered
  - `slides.md`: ≥40 slides with speaker notes
  - All 4 diagrams rendered
  - Cross-references to Textbook Chapter 2

---

### T22 — Week 3 Lecture: The Transformer Architecture

- **Branch:** `task/t22-w03-lecture`
- **Files created:**
  - `modules/week_03_transformer/lecture.md`
  - `modules/week_03_transformer/slides.md`
- **Topics:** Transformer paper, attention, multi-head, positional encodings, blocks, FlashAttention
- **Diagrams to render:**
  - Mermaid diagram: full Transformer architecture (encoder stack → decoder stack)
  - ASCII figure: scaled dot-product attention computation flow
  - ASCII figure: multi-head attention with parallel heads
  - ASCII figure: positional encoding comparison (sinusoidal vs learned vs RoPE)
  - ASCII figure: FlashAttention tiling computation order
- **Definition of done:**
  - `lecture.md`: ≥5000 words, all 6 days covered with equations
  - `slides.md`: ≥50 slides with speaker notes
  - All 5 diagrams rendered
  - Cross-references to Textbook Chapter 3

---

### T23 — Week 4 Lecture: Scaling Laws, MoE, Pre-Training

- **Branch:** `task/t23-w04-lecture`
- **Files created:**
  - `modules/week_04_scaling_moe/lecture.md`
  - `modules/week_04_scaling_moe/slides.md`
- **Topics:** Scaling laws, Chinchilla, MoE, MoE architectures, pre-training, loss landscapes
- **Diagrams to render:**
  - ASCII figure: Kaplan vs Chinchilla scaling law comparison (loss vs compute with different token/param ratios)
  - ASCII figure: MoE gating and routing (token → softmax gate → top-2 experts → weighted residual sum)
  - ASCII figure: Mixtral expert routing diagram (8 experts, 2 activated per token, with load balancing)
  - ASCII figure: learning rate schedule (warmup → peak → cosine decay → gradient clipping events)
- **Definition of done:**
  - `lecture.md`: ≥4000 words, all 7 days covered
  - `slides.md`: ≥45 slides with speaker notes
  - All 4 diagrams rendered
  - Cross-references to Textbook Chapter 4

---

### T24 — Week 5 Lecture: Datasets for LLMs

- **Branch:** `task/t24-w05-lecture`
- **Files created:**
  - `modules/week_05_datasets/lecture.md`
  - `modules/week_05_datasets/slides.md`
- **Topics:** Pre-training corpora, instruction datasets, preference datasets, quality, synthetic data, licensing
- **Diagrams to render:**
  - ASCII figure: data quality funnel (raw web → language-filtered → deduped → quality-scored → instruction-formatted)
  - ASCII figure: self-instruct generation loop
  - ASCII figure: chosen/rejected preference pair examples
- **Tables:** pre-training corpus comparison; instruction dataset comparison; data license matrix
- **Definition of done:**
  - `lecture.md`: ≥3500 words, all 6 days covered
  - `slides.md`: ≥35 slides with speaker notes
  - All 3 diagrams rendered; all 3 tables rendered
  - Cross-references to Textbook Chapter 5

---

### T25 — Week 6 Lecture: Data Engineering Pipelines

- **Branch:** `task/t25-w06-lecture`
- **Files created:**
  - `modules/week_06_data_pipelines/lecture.md`
  - `modules/week_06_data_pipelines/slides.md`
- **Topics:** Pipeline architecture, text cleaning, dedup, quality filtering, formatting, versioning
- **Diagrams to render:**
  - Mermaid DAG: data pipeline with 6 stages and error/retry branches
  - ASCII figure: minhash + LSH algorithm (shingles → hash matrix → minhash → banding → candidates)
  - ASCII figure: quality filter cascade funnel with document count at each stage
- **Definition of done:**
  - `lecture.md`: ≥3500 words, all 6 days covered
  - `slides.md`: ≥35 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 6

---

### T26 — Week 7 Lecture: Tokenization

- **Branch:** `task/t26-w07-lecture`
- **Files created:**
  - `modules/week_07_tokenization/lecture.md`
  - `modules/week_07_tokenization/slides.md`
- **Topics:** Tokenization problem, BPE, WordPiece/Unigram, metrics, domain-adaptive, edge cases
- **Diagrams to render:**
  - ASCII figure: word-level vs subword on same text with OOV highlighting
  - ASCII figure: BPE merge steps (5-step progression from chars to tokens)
  - ASCII figure: tokenizer comparison bar chart (tokens-per-character for BPE, WordPiece, Unigram on same corpus)
- **Definition of done:**
  - `lecture.md`: ≥3500 words, all 6 days covered
  - `slides.md`: ≥35 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 7

---

### T27 — Week 8 Lecture: Instruction Tuning

- **Branch:** `task/t27-w08-lecture`
- **Files created:**
  - `modules/week_08_instruction_tuning/lecture.md`
  - `modules/week_08_instruction_tuning/slides.md`
- **Topics:** Full FT vs PEFT, LoRA, QLoRA, beyond LoRA, training config, evaluation
- **Diagrams to render:**
  - ASCII figure: LoRA decomposition (W → W + BA) with dimensions labeled
  - ASCII figure: QLoRA memory breakdown pie chart (4-bit weights, optimizer states, activations, grad states)
  - ASCII figure: learning rate ablation curves (3 learning rates, train/val loss)
- **Definition of done:**
  - `lecture.md`: ≥4000 words, all 6 days covered
  - `slides.md`: ≥40 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 8

---

### T28 — Week 9 Lecture: Alignment

- **Branch:** `task/t28-w09-lecture`
- **Files created:**
  - `modules/week_09_alignment/lecture.md`
  - `modules/week_09_alignment/slides.md`
- **Topics:** Alignment problem, reward modeling, RLHF, DPO, constitutional AI, evaluating alignment
- **Diagrams to render:**
  - Mermaid diagram: RLHF four-model system (policy, reference, reward, value) with data flow
  - ASCII figure: DPO simplification (RLHF 4-model → DPO 2-model)
  - ASCII figure: constitutional AI self-critique loop
- **Definition of done:**
  - `lecture.md`: ≥4000 words, all 6 days covered
  - `slides.md`: ≥40 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 9

---

### T29 — Week 10 Lecture: Quantization

- **Branch:** `task/t29-w10-lecture`
- **Files created:**
  - `modules/week_10_quantization/lecture.md`
  - `modules/week_10_quantization/slides.md`
- **Topics:** Why quantize, PTQ, QAT, GPTQ/AWQ/SmoothQuant, 4-bit ecosystem, quantized FT
- **Diagrams to render:**
  - ASCII figure: float-to-int quantization mapping with clipping
  - ASCII figure: GPTQ layer-by-layer quantization pipeline
  - ASCII figure: GGUF quant preset size/quality tradeoff scatter plot
- **Tables:** model size at different precisions; GGUF preset comparison
- **Definition of done:**
  - `lecture.md`: ≥4000 words, all 6 days covered
  - `slides.md`: ≥40 slides with speaker notes
  - All 3 diagrams rendered; all 2 tables rendered
  - Cross-references to Textbook Chapter 10

---

### T30 — Week 11 Lecture: Speculative Decoding & Inference Optimization

- **Branch:** `task/t30-w11-lecture`
- **Files created:**
  - `modules/week_11_inference_optimization/lecture.md`
  - `modules/week_11_inference_optimization/slides.md`
- **Topics:** Inference bottleneck, speculative decoding, draft models, beyond speculative, KV cache, compilation
- **Diagrams to render:**
  - ASCII figure: speculative decoding step-by-step (3-token draft → verify → accept/reject → regenerate)
  - ASCII figure: paged attention block table vs contiguous allocation
  - ASCII figure: Medusa multi-head prediction architecture
- **Definition of done:**
  - `lecture.md`: ≥4000 words, all 6 days covered
  - `slides.md`: ≥40 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 11

---

### T31 — Week 12 Lecture: Deployment

- **Branch:** `task/t31-w12-lecture`
- **Files created:**
  - `modules/week_12_deployment/lecture.md`
  - `modules/week_12_deployment/slides.md`
- **Topics:** API design, serving frameworks, containerization, cloud deployment, production patterns, load testing
- **Diagrams to render:**
  - Mermaid sequence diagram: streaming inference request/response
  - ASCII figure: canary deployment rollout timeline
  - ASCII figure: latency percentiles under load (p50/p95/p99 vs concurrent users)
- **Definition of done:**
  - `lecture.md`: ≥3500 words, all 6 days covered
  - `slides.md`: ≥35 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 12

---

### T32 — Week 13 Lecture: Monitoring, Evaluation, and Iteration

- **Branch:** `task/t32-w13-lecture`
- **Files created:**
  - `modules/week_13_monitoring/lecture.md`
  - `modules/week_13_monitoring/slides.md`
- **Topics:** Production eval, drift detection, observability, continuous learning, catastrophic forgetting, experiment tracking
- **Diagrams to render:**
  - ASCII figure: A/B test traffic splitting architecture
  - ASCII figure: drift detection time series with alert threshold
  - ASCII figure: data flywheel cycle (produce → collect feedback → retrain → deploy → produce)
- **Definition of done:**
  - `lecture.md`: ≥3500 words, all 6 days covered
  - `slides.md`: ≥35 slides with speaker notes
  - All 3 diagrams rendered
  - Cross-references to Textbook Chapter 13

---

### T33 — Week 14 Lecture: Capstone Project

- **Branch:** `task/t33-w14-lecture`
- **Files created:**
  - `modules/week_14_capstone/lecture.md`
  - `modules/week_14_capstone/slides.md`
- **Topics:** Project scoping, data pipeline, tokenization, instruction tuning, alignment, quantization, deployment
- **Diagrams to render:**
  - Mermaid diagram: end-to-end capstone pipeline (7 stages with deliverables at each)
  - ASCII figure: capstone project timeline (Gantt chart of 7 days with dependencies)
- **Definition of done:**
  - `lecture.md`: ≥3000 words, all 7 days covered with deliverables
  - `slides.md`: ≥30 slides with speaker notes
  - All 2 diagrams rendered
  - Cross-references to Textbook Chapter 14

---

## Phase 4 — Module Labs (T34–T47)

Each task creates a `labs/` directory under the week's module folder. Every lab is a standalone Python file (`.py`) with inline documentation, runnable from the repository root. Labs include setup instructions, expected outputs, and discussion questions.

### T34 — Week 1 Labs

- **Branch:** `task/t34-w01-labs`
- **Files created:**
  - `modules/week_01_ml_refresher/labs/lab_01_env_setup.md` — environment setup guide (Python, PyTorch, CUDA verification)
  - `modules/week_01_ml_refresher/labs/lab_02_tensor_ops.py` — NumPy/PyTorch tensor manipulation exercises
  - `modules/week_01_ml_refresher/labs/lab_03_probability.py` — MLE and MAP estimation from scratch
  - `modules/week_01_ml_refresher/labs/lab_04_optimization.py` — train linear and logistic regression from scratch
  - `modules/week_01_ml_refresher/labs/lab_05_evaluation.py` — L2 regularization and early stopping experiments
  - `modules/week_01_ml_refresher/labs/lab_06_backprop.py` — implement backprop through a computation graph
- **Definition of done:**
  - 1 markdown guide + 5 Python lab scripts
  - Each script runs without errors and prints expected output
  - Scripts include ≥3 exercises with increasing difficulty
  - Inline comments explain the "why" behind each step

---

### T35 — Week 2 Labs

- **Branch:** `task/t35-w02-labs`
- **Files created:**
  - `modules/week_02_deep_learning/labs/lab_01_activation_functions.py`
  - `modules/week_02_deep_learning/labs/lab_02_1d_conv.py`
  - `modules/week_02_deep_learning/labs/lab_03_rnn_lm.py`
  - `modules/week_02_deep_learning/labs/lab_04_lstm.py`
  - `modules/week_02_deep_learning/labs/lab_05_seq2seq.py`
  - `modules/week_02_deep_learning/labs/lab_06_normalization.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Each script runs without errors
  - Lab 5 includes seq2seq with and without attention for comparison
  - All scripts include ≥3 exercises

---

### T36 — Week 3 Labs

- **Branch:** `task/t36-w03-labs`
- **Files created:**
  - `modules/week_03_transformer/labs/lab_01_transformer_paper.py`
  - `modules/week_03_transformer/labs/lab_02_attention.py`
  - `modules/week_03_transformer/labs/lab_03_multihead.py`
  - `modules/week_03_transformer/labs/lab_04_positional.py`
  - `modules/week_03_transformer/labs/lab_05_decoder_block.py`
  - `modules/week_03_transformer/labs/lab_06_flash_attention.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 2 visualizes attention weights (saves to file or prints text heatmap)
  - Lab 6 benchmarks FlashAttention vs naive attention with timing output
  - All scripts include ≥3 exercises

---

### T37 — Week 4 Labs

- **Branch:** `task/t37-w04-labs`
- **Files created:**
  - `modules/week_04_scaling_moe/labs/lab_01_scaling_laws.py`
  - `modules/week_04_scaling_moe/labs/lab_02_chinchilla.py`
  - `modules/week_04_scaling_moe/labs/lab_03_moe_layer.py`
  - `modules/week_04_scaling_moe/labs/lab_04_mixtral_compare.py`
  - `modules/week_04_scaling_moe/labs/lab_05_pretraining_mixture.py`
  - `modules/week_04_scaling_moe/labs/lab_06_lr_sweep.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 1 plots scaling curves (saves figure or prints ASCII chart)
  - Lab 3 implements MoE layer and compares to dense FFN
  - Lab 6 runs a learning rate sweep and plots results
  - All scripts include ≥3 exercises

---

### T38 — Week 5 Labs

- **Branch:** `task/t38-w05-labs`
- **Files created:**
  - `modules/week_05_datasets/labs/lab_01_pile_inspect.py`
  - `modules/week_05_datasets/labs/lab_02_instruction_formats.py`
  - `modules/week_05_datasets/labs/lab_03_preference_data.py`
  - `modules/week_05_datasets/labs/lab_04_quality_report.py`
  - `modules/week_05_datasets/labs/lab_05_synthetic_data.py`
  - `modules/week_05_datasets/labs/lab_06_license_audit.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 4 produces a quality report (print to stdout or save to JSON)
  - Lab 5 generates synthetic instructions (mock API if no key available)
  - All scripts include ≥3 exercises

---

### T39 — Week 6 Labs

- **Branch:** `task/t39-w06-labs`
- **Files created:**
  - `modules/week_06_data_pipelines/labs/lab_01_pipeline_dag.py`
  - `modules/week_06_data_pipelines/labs/lab_02_text_cleaning.py`
  - `modules/week_06_data_pipelines/labs/lab_03_minhash_dedup.py`
  - `modules/week_06_data_pipelines/labs/lab_04_quality_filter.py`
  - `modules/week_06_data_pipelines/labs/lab_05_instruction_format.py`
  - `modules/week_06_data_pipelines/labs/lab_06_dvc_versioning.md`
- **Definition of done:**
  - 5 Python lab scripts + 1 markdown guide (DVC is tool-setup, not code)
  - Lab 3 implements minhash from scratch and measures dedup rate
  - Lab 4 uses a pre-trained model's perplexity as quality signal
  - All scripts include ≥3 exercises

---

### T40 — Week 7 Labs

- **Branch:** `task/t40-w07-labs`
- **Files created:**
  - `modules/week_07_tokenization/labs/lab_01_word_level.py`
  - `modules/week_07_tokenization/labs/lab_02_bpe.py`
  - `modules/week_07_tokenization/labs/lab_03_wordpiece.py`
  - `modules/week_07_tokenization/labs/lab_04_benchmark.py`
  - `modules/week_07_tokenization/labs/lab_05_vocab_extend.py`
  - `modules/week_07_tokenization/labs/lab_06_edge_cases.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 2 implements BPE training from scratch (no external tokenizer library)
  - Lab 4 benchmarks ≥5 tokenizers on a domain-specific corpus
  - Lab 6 stress-tests on code, math notation, emoji, and CJK
  - All scripts include ≥3 exercises

---

### T41 — Week 8 Labs

- **Branch:** `task/t41-w08-labs`
- **Files created:**
  - `modules/week_08_instruction_tuning/labs/lab_01_ft_vs_peft.py`
  - `modules/week_08_instruction_tuning/labs/lab_02_lora.py`
  - `modules/week_08_instruction_tuning/labs/lab_03_qlora.py`
  - `modules/week_08_instruction_tuning/labs/lab_04_dora_compare.py`
  - `modules/week_08_instruction_tuning/labs/lab_05_hyperparameter_sweep.py`
  - `modules/week_08_instruction_tuning/labs/lab_06_ifeval.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 1 compares full FT memory to PEFT on a small model
  - Lab 3 fine-tunes a 7B model with QLoRA (graceful fallback for <16GB VRAM)
  - Lab 6 evaluates on IFEval
  - All scripts include ≥3 exercises

---

### T42 — Week 9 Labs

- **Branch:** `task/t42-w09-labs`
- **Files created:**
  - `modules/week_09_alignment/labs/lab_01_reward_model.py`
  - `modules/week_09_alignment/labs/lab_02_rlhf_walkthrough.py`
  - `modules/week_09_alignment/labs/lab_03_ppo.py`
  - `modules/week_09_alignment/labs/lab_04_dpo.py`
  - `modules/week_09_alignment/labs/lab_05_constitutional_ai.py`
  - `modules/week_09_alignment/labs/lab_06_alignment_benchmark.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 2 is a walkthrough with annotated code (not a full PPO training run)
  - Lab 4 applies DPO and compares to PPO results
  - Lab 6 benchmarks on XSTest and TruthfulQA
  - All scripts include ≥3 exercises

---

### T43 — Week 10 Labs

- **Branch:** `task/t43-w10-labs`
- **Files created:**
  - `modules/week_10_quantization/labs/lab_01_precision_compare.py`
  - `modules/week_10_quantization/labs/lab_02_ptq.py`
  - `modules/week_10_quantization/labs/lab_03_qat.py`
  - `modules/week_10_quantization/labs/lab_04_gptq_awq.py`
  - `modules/week_10_quantization/labs/lab_05_gguf.py`
  - `modules/week_10_quantization/labs/lab_06_quant_ft_compare.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 1 measures memory and latency at FP32/FP16/BF16/INT8/INT4
  - Lab 4 quantizes a 7B model with GPTQ and AWQ
  - Lab 5 converts to GGUF and serves with llama.cpp
  - All scripts include ≥3 exercises

---

### T44 — Week 11 Labs

- **Branch:** `task/t44-w11-labs`
- **Files created:**
  - `modules/week_11_inference_optimization/labs/lab_01_latency_profile.py`
  - `modules/week_11_inference_optimization/labs/lab_02_speculative.py`
  - `modules/week_11_inference_optimization/labs/lab_03_draft_strategies.py`
  - `modules/week_11_inference_optimization/labs/lab_04_medusa.py`
  - `modules/week_11_inference_optimization/labs/lab_05_kv_cache.py`
  - `modules/week_11_inference_optimization/labs/lab_06_compile.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 2 implements speculative decoding from scratch
  - Lab 3 compares 3 draft strategies (smaller model, echo, lookback)
  - Lab 5 benchmarks vLLM concurrent throughput
  - All scripts include ≥3 exercises

---

### T45 — Week 12 Labs

- **Branch:** `task/t45-w12-labs`
- **Files created:**
  - `modules/week_12_deployment/labs/lab_01_fastapi.py`
  - `modules/week_12_deployment/labs/lab_02_serving_frameworks.md`
  - `modules/week_12_deployment/labs/lab_03_docker.md`
  - `modules/week_12_deployment/labs/lab_04_cloud_deploy.md`
  - `modules/week_12_deployment/labs/lab_05_autoscaling.py`
  - `modules/week_12_deployment/labs/lab_06_auth_ratelimit.py`
- **Definition of done:**
  - 2 Python scripts + 4 markdown guides (Docker/cloud are infrastructure, not pure code)
  - Lab 1 builds a working FastAPI server for local model
  - Lab 3 includes a complete, multi-stage Dockerfile with CUDA
  - Lab 6 adds JWT auth and token bucket rate limiting
  - All include ≥3 exercises

---

### T46 — Week 13 Labs

- **Branch:** `task/t46-w13-labs`
- **Files created:**
  - `modules/week_13_monitoring/labs/lab_01_ab_test.py`
  - `modules/week_13_monitoring/labs/lab_02_drift_detection.py`
  - `modules/week_13_monitoring/labs/lab_03_dashboard.py`
  - `modules/week_13_monitoring/labs/lab_04_feedback_pipeline.py`
  - `modules/week_13_monitoring/labs/lab_05_forgetting.py`
  - `modules/week_13_monitoring/labs/lab_06_mlflow.py`
- **Definition of done:**
  - 6 Python lab scripts
  - Lab 2 implements drift detection with PSI and KL divergence
  - Lab 3 builds a Grafana-compatible metrics dashboard
  - Lab 6 sets up MLflow tracking for experiments
  - All scripts include ≥3 exercises

---

### T47 — Week 14 Labs

- **Branch:** `task/t47-w14-labs`
- **Files created:**
  - `modules/week_14_capstone/labs/lab_01_proposal_template.md`
  - `modules/week_14_capstone/labs/lab_02_data_pipeline.py`
  - `modules/week_14_capstone/labs/lab_03_tokenize.py`
  - `modules/week_14_capstone/labs/lab_04_finetune.py`
  - `modules/week_14_capstone/labs/lab_05_align.py`
  - `modules/week_14_capstone/labs/lab_06_quantize.py`
  - `modules/week_14_capstone/labs/lab_07_deploy.py`
- **Definition of done:**
  - 1 markdown template + 6 Python scaffold scripts
  - Each scaffold script is a complete but minimal implementation the student extends
  - Lab 1 template includes sections for domain, base model, dataset, evaluation criteria, timeline
  - Scripts chain together (output of lab N feeds into lab N+1)

---

## Phase 5 — Module Assessments (T48–T56)

Each task creates the assignment description, quiz questions, answer sheet, and grading guide for one week. These are four distinct files per task.

### T48 — Week 1 Assessments

- **Branch:** `task/t48-w01-assess`
- **Files created:**
  - `modules/week_01_ml_refresher/assignment.md`
  - `modules/week_01_ml_refresher/quiz.md`
  - `modules/week_01_ml_refresher/answer_sheet.md`
  - `modules/week_01_ml_refresher/grading_guide.md`
- **Assignment:** Implement MLP from scratch classifying MNIST with Adam, L2 reg, early stopping
- **Quiz:** 15 questions (5 MC, 5 short answer, 5 calculation)
- **Answer sheet:** full solutions for all quiz questions + assignment rubric with starter code
- **Grading guide:** point breakdown (assignment 70%, quiz 30%), partial credit rubric, common mistakes
- **Definition of done:**
  - All 4 files created
  - Assignment includes starter code skeleton, evaluation criteria, and submission instructions
  - Quiz has ≥15 questions with answer key in answer_sheet.md
  - Grading guide has point breakdown, rubric levels (excellent/good/fair/poor), and common mistake callouts

---

### T49 — Week 2 Assessments

- **Branch:** `task/t49-w02-assess`
- **Files created:**
  - `modules/week_02_deep_learning/assignment.md`
  - `modules/week_02_deep_learning/quiz.md`
  - `modules/week_02_deep_learning/answer_sheet.md`
  - `modules/week_02_deep_learning/grading_guide.md`
- **Assignment:** Char-level LM with LSTM, then replace with attention + compare perplexity
- **Quiz:** 15 questions covering MLPs, CNNs, RNNs, LSTMs, seq2seq, normalization
- **Definition of done:** Same 4-file deliverable; assignment with starter code; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T50 — Week 3 Assessments

- **Branch:** `task/t50-w03-assess`
- **Files created:**
  - `modules/week_03_transformer/assignment.md`
  - `modules/week_03_transformer/quiz.md`
  - `modules/week_03_transformer/answer_sheet.md`
  - `modules/week_03_transformer/grading_guide.md`
- **Assignment:** Minimal GPT-style decoder trained on small corpus; report perplexity vs layers/heads
- **Quiz:** 15 questions covering attention, multi-head, positional encodings, blocks, FlashAttention
- **Definition of done:** Same 4-file deliverable; assignment requires ablation study; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T51 — Week 4 Assessments

- **Branch:** `task/t51-w04-assess`
- **Files created:**
  - `modules/week_04_scaling_moe/assignment.md`
  - `modules/week_04_scaling_moe/quiz.md`
  - `modules/week_04_scaling_moe/answer_sheet.md`
  - `modules/week_04_scaling_moe/grading_guide.md`
- **Assignment:** Technical report comparing dense and MoE with working MoE layer and ablation studies
- **Quiz:** 20 questions (comprehensive across Weeks 1-4)
- **Definition of done:** Same 4-file deliverable; assignment is a technical report (not just code); ≥20 quiz questions; answer sheet; grading rubric

---

### T52 — Week 5 Assessments

- **Branch:** `task/t52-w05-assess`
- **Files created:**
  - `modules/week_05_datasets/assignment.md`
  - `modules/week_05_datasets/quiz.md`
  - `modules/week_05_datasets/answer_sheet.md`
  - `modules/week_05_datasets/grading_guide.md`
- **Assignment:** Profile 3 instruction-tuning datasets; report with duplication, toxicity, diversity, recommendation
- **Quiz:** 15 questions covering corpora, instruction formats, preference data, quality, synthetic data, licensing
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T53 — Week 6 Assessments

- **Branch:** `task/t53-w06-assess`
- **Files created:**
  - `modules/week_06_data_pipelines/assignment.md`
  - `modules/week_06_data_pipelines/quiz.md`
  - `modules/week_06_data_pipelines/answer_sheet.md`
  - `modules/week_06_data_pipelines/grading_guide.md`
- **Assignment:** End-to-end data pipeline from raw crawl to instruction dataset with stage-by-stage documentation
- **Quiz:** 15 questions covering pipeline design, cleaning, dedup, filtering, formatting, versioning
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T54 — Week 7 Assessments

- **Branch:** `task/t54-w07-assess`
- **Files created:**
  - `modules/week_07_tokenization/assignment.md`
  - `modules/week_07_tokenization/quiz.md`
  - `modules/week_07_tokenization/answer_sheet.md`
  - `modules/week_07_tokenization/grading_guide.md`
- **Assignment:** Custom BPE tokenizer on domain corpus; compare to base tokenizer; document vocab size impact
- **Quiz:** 15 questions covering BPE, WordPiece, Unigram, metrics, domain adaptation, edge cases
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T55 — Week 8 Assessments

- **Branch:** `task/t55-w08-assess`
- **Files created:**
  - `modules/week_08_instruction_tuning/assignment.md`
  - `modules/week_08_instruction_tuning/quiz.md`
  - `modules/week_08_instruction_tuning/answer_sheet.md`
  - `modules/week_08_instruction_tuning/grading_guide.md`
- **Assignment:** QLoRA fine-tune 7B model; compare to base on benchmarks; ablate rank and learning rate
- **Quiz:** 15 questions covering FT vs PEFT, LoRA, QLoRA, DoRA, training config, evaluation
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T56 — Week 9 Assessments

- **Branch:** `task/t56-w09-assess`
- **Files created:**
  - `modules/week_09_alignment/assignment.md`
  - `modules/week_09_alignment/quiz.md`
  - `modules/week_09_alignment/answer_sheet.md`
  - `modules/week_09_alignment/grading_guide.md`
- **Assignment:** DPO alignment on Assignment 8 model; evaluate on harmlessness/helpfulness; DPO vs PPO comparison
- **Quiz:** 15 questions covering alignment, reward modeling, RLHF, DPO, constitutional AI, evaluation
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

## Phase 6 — Module Assessments: Part III (T57–T63)

### T57 — Week 10 Assessments

- **Branch:** `task/t57-w10-assess`
- **Files created:**
  - `modules/week_10_quantization/assignment.md`
  - `modules/week_10_quantization/quiz.md`
  - `modules/week_10_quantization/answer_sheet.md`
  - `modules/week_10_quantization/grading_guide.md`
- **Assignment:** Quantize aligned model to INT8/INT4 via PTQ, GPTQ, AWQ; benchmark accuracy/latency/memory; recommend strategy
- **Quiz:** 15 questions covering precision formats, PTQ, QAT, GPTQ, AWQ, GGUF, quantized FT
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T58 — Week 11 Assessments

- **Branch:** `task/t58-w11-assess`
- **Files created:**
  - `modules/week_11_inference_optimization/assignment.md`
  - `modules/week_11_inference_optimization/quiz.md`
  - `modules/week_11_inference_optimization/answer_sheet.md`
  - `modules/week_11_inference_optimization/grading_guide.md`
- **Assignment:** Implement speculative decoding for quantized model; benchmark tps/latency at temperatures and lengths
- **Quiz:** 15 questions covering inference bottleneck, speculative decoding, draft models, KV cache, compilation
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T59 — Week 12 Assessments

- **Branch:** `task/t59-w12-assess`
- **Files created:**
  - `modules/week_12_deployment/assignment.md`
  - `modules/week_12_deployment/quiz.md`
  - `modules/week_12_deployment/answer_sheet.md`
  - `modules/week_12_deployment/grading_guide.md`
- **Assignment:** Containerize optimized model; deploy behind authenticated, rate-limited API; load test report
- **Quiz:** 15 questions covering API design, serving frameworks, Docker, cloud, production patterns, load testing
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T60 — Week 13 Assessments

- **Branch:** `task/t60-w13-assess`
- **Files created:**
  - `modules/week_13_monitoring/assignment.md`
  - `modules/week_13_monitoring/quiz.md`
  - `modules/week_13_monitoring/answer_sheet.md`
  - `modules/week_13_monitoring/grading_guide.md`
- **Assignment:** Monitoring dashboard with drift detection and retraining trigger on quality degradation
- **Quiz:** 15 questions covering production eval, drift, observability, continuous learning, forgetting, experiment tracking
- **Definition of done:** Same 4-file deliverable; ≥15 quiz questions; complete answer sheet; grading rubric

---

### T61 — Week 14 Assessments

- **Branch:** `task/t61-w14-assess`
- **Files created:**
  - `modules/week_14_capstone/assignment.md`
  - `modules/week_14_capstone/quiz.md`
  - `modules/week_14_capstone/answer_sheet.md`
  - `modules/week_14_capstone/grading_guide.md`
- **Assignment:** Complete capstone project (data → tokenize → train → align → quantize → deploy → monitor)
- **Quiz:** 20 comprehensive questions spanning the entire course
- **Definition of done:**
  - `assignment.md`: full capstone specification with deliverables, milestones, and evaluation criteria
  - `quiz.md`: ≥20 questions spanning all 14 weeks
  - `answer_sheet.md`: complete solutions
  - `grading_guide.md`: capstone rubric covering all 7 pipeline stages with scoring criteria (excellent/good/fair/poor) per stage

---

## Phase 7 — Project Infrastructure (T62–T63)

### T62 — Capstone Starter Repository

- **Branch:** `task/t62-capstone-starter`
- **Files created:**
  - `capstone/README.md` — project overview, setup instructions, milestone checklist
  - `capstone/pyproject.toml` — Python project configuration with dependencies
  - `capstone/src/pipeline/__init__.py`
  - `capstone/src/pipeline/data.py` — scaffold: data ingestion and cleaning
  - `capstone/src/pipeline/tokenize.py` — scaffold: tokenization
  - `capstone/src/pipeline/train.py` — scaffold: instruction tuning with QLoRA
  - `capstone/src/pipeline/align.py` — scaffold: DPO alignment
  - `capstone/src/pipeline/quantize.py` — scaffold: quantization
  - `capstone/src/pipeline/serve.py` — scaffold: API server with auth and rate limiting
  - `capstone/src/pipeline/monitor.py` — scaffold: drift detection and metrics
  - `capstone/Dockerfile` — multi-stage build for model serving
  - `capstone/configs/base.yaml` — default training configuration
  - `capstone/tests/test_pipeline.py` — skeleton tests
- **Definition of done:**
  - All 13 files created
  - Each scaffold module has typed function signatures, docstrings, and `NotImplementedError` bodies
  - `pyproject.toml` lists all required dependencies
  - `Dockerfile` builds successfully (syntax check)
  - `README.md` has setup instructions and milestone checklist matching Week 14 syllabus

---

### T63 — Update README and SYLLABUS

- **Branch:** `task/t63-update-readme`
- **Files modified:**
  - `README.md` — update project layout section to reflect final directory structure
  - `SYLLABUS.md` — add cross-references to textbook chapters and module file paths
- **Definition of done:**
  - `README.md` project layout matches the actual directory structure after all tasks are merged
  - `SYLLABUS.md` each week includes a "Materials" line linking to the module directory and textbook chapter
  - No broken internal links

---

## Task Dependency Summary

```
Phase 1 (T01-T14):   Textbook chapters — all independent of each other
Phase 2 (T15-T19):   Appendices — independent of each other; reference textbook
Phase 3 (T20-T33):   Lecture materials — 1 per week, independent of each other
Phase 4 (T34-T47):   Labs — 1 per week, independent of each other
Phase 5 (T48-T61):   Assessments — 1 per week, independent of each other
Phase 7 (T62-T63):   Infrastructure — T62 independent; T63 runs last (updates file layout)
```

Within each week N:
- Textbook chapter N (T01–T14) can be created before or after module materials (T20+, T34+, T48+)
- Lecture materials (T20–T33) and labs (T34–T47) and assessments (T48–T61) for the same week are independent of each other
- All three module artifacts reference the same textbook chapter but do not depend on each other's files

**Recommended execution order for parallelism:**
1. Spawn all 14 textbook tasks (T01–T14) in parallel
2. Spawn all 5 appendix tasks (T15–T19) in parallel
3. Spawn all 14 lecture tasks (T20–T33) in parallel
4. Spawn all 14 lab tasks (T34–T47) in parallel
5. Spawn all 14 assessment tasks (T48–T61) in parallel
6. Run T62 (capstone starter) after phase 4 completes
7. Run T63 (README update) last, after all other tasks are merged

## File Count Summary

| Category | Files |
|---|---|
| Textbook chapters | 14 |
| Appendices | 5 |
| Lecture notes | 14 |
| Slides | 14 |
| Lab scripts (Python) | ~78 |
| Lab guides (markdown) | ~6 |
| Assignments | 14 |
| Quizzes | 14 |
| Answer sheets | 14 |
| Grading guides | 14 |
| Capstone starter | 13 |
| **Total** | **~218 files** |
