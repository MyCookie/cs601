# 4.3 Compute Costs and Training Infrastructure

## Peer-to-Peer Guide
Alright, let's get real. Training a base model isn't something you do on a laptop. It's one of the most computationally expensive tasks in all of computer science. When you hear that a model "cost $100 million to train," it's not because the software is expensive—it's because of the hardware and the electricity.

### The GPU: The Engine of LLMs
You've probably heard of NVIDIA. Their GPUs (Graphics Processing Units) are the gold standard for LLM training. Why? Because training a Transformer is basically just doing billions of matrix multiplications (linear algebra) over and over again. CPUs are like a few geniuses who can solve complex problems one by one; GPUs are like ten thousand students who can all do simple multiplication at the exact same time.

For the big models, we use the **H100** or **A100** GPUs. These aren't just cards you plug into a motherboard; they are often clustered into "nodes" (like a DGX system) and then linked together into "clusters" of thousands of GPUs.

### The Bottlenecks: Memory and Bandwidth
The biggest headache isn't actually the "math" (compute); it's the **movement of data**.

1. **VRAM (Video RAM):** The model weights have to live inside the GPU's memory. If a model has 7 billion parameters, and each parameter is a 16-bit float (2 bytes), the model alone takes 14GB of VRAM. But during training, we also store "gradients" and "optimizer states," which can multiply that memory requirement by 4x or more.
2. **Interconnect (NVLink):** When you have 1,000 GPUs, they need to talk to each other to sync their learning. If they use standard cables, the training slows to a crawl. NVIDIA uses **NVLink**, which is like a super-highway for data between GPUs.

### Parallelism: Splitting the Work
Since one GPU can't hold a massive model, we have to split the work. There are three main ways we do this:

- **Data Parallelism:** Every GPU has a copy of the model, but they each look at a *different* piece of the dataset. They then "average" their findings at the end of each step.
- **Tensor Parallelism:** We split a single layer (a big matrix) across multiple GPUs. One GPU does the left half of the multiplication, another does the right half.
- **Pipeline Parallelism:** We split the model by layers. GPU 1 handles layers 1-10, GPU 2 handles 11-20, and so on.

### The Bill: Compute Budget
Training is usually measured in **GPU-hours**. If you use 1,000 H100s for 30 days, that's $1,000 \times 24 \times 30 = 720,000$ GPU-hours. At cloud prices, this is where the millions of dollars come from.

---

## Technical Summary

**Computational Requirements**
Pre-training a Large Language Model is a compute-bound and memory-bound problem. The total compute required ($C$) can be approximated as:
$$C \approx 6ND$$
Where $N$ is the number of parameters and $D$ is the number of tokens. This indicates that compute scales linearly with both model size and dataset size.

**Hardware Constraints**
1. **Memory Capacity:** The memory required for training exceeds the model size due to:
   - **Model Weights:** $\sim 2$ bytes per parameter (FP16/BF16).
   - **Gradients:** $\sim 2$ bytes per parameter.
   - **Optimizer States:** (e.g., Adam optimizer) $\sim 12$ bytes per parameter.
   - **Total:** Approximately 16–20 bytes per parameter for full fine-tuning.
2. **Memory Bandwidth:** The time spent moving weights from VRAM to the Tensor Cores is often the primary bottleneck (Memory Wall).

**Distributed Training Strategies**
To overcome single-device limits, distributed training is utilized:
- **Data Parallelism (DP/DDP):** Replicates the model across $N$ devices; each processes a unique mini-batch.
- **Tensor Parallelism (TP):** Shards individual tensors across devices, requiring high-bandwidth interconnects (e.g., NVLink) due to frequent synchronization.
- **Pipeline Parallelism (PP):** Partitions the model's layers across devices, introducing "bubbles" (idle time) that are mitigated using micro-batching (e.g., GPipe).
- **ZeRO (Zero Redundancy Optimizer):** A technique used in DeepSpeed to shard optimizer states and gradients across GPUs, reducing memory redundancy without the communication overhead of full TP.

**Infrastructure Stack**
Modern clusters rely on a hierarchy of communication:
- **Intra-node:** NVLink/NVSwitch for high-speed GPU-to-GPU transfer.
- **Inter-node:** InfiniBand or RoCE (RDMA over Converged Ethernet) to minimize latency across different server racks.