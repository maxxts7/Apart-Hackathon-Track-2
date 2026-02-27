# Why Multi-GPU Inference Works the Way It Does

*The motivation and rationale behind every design decision, derived from first principles*

---

## The Chain of Constraints

Every design decision in multi-GPU inference follows from a chain of physical and mathematical constraints. This document traces that chain: each section identifies a constraint, explains why it exists, and shows how the engineering solution follows inevitably from the physics.

```
Constraint 1: Models are collections of numbers
    ↓
Constraint 2: Numbers must live in fast memory (VRAM)
    ↓
Constraint 3: VRAM is finite and expensive
    ↓
Constraint 4: So we split the model across GPUs
    ↓
Constraint 5: Splitting requires communication
    ↓
Constraint 6: Communication has bandwidth limits
    ↓
Constraint 7: Bandwidth limits dictate HOW we split
    ↓
Every design decision follows from these 7 constraints.
```

---

## 1. Why Can't We Just Use One GPU?

### The Physical Constraint

A neural network is a sequence of matrix multiplications. Each matrix (a "weight matrix") is a 2D grid of numbers. During inference, the input passes through these matrices one by one, transforming at each step until it becomes an output.

These weight matrices must be in **GPU VRAM** (not CPU RAM, not disk) because GPUs are the only hardware fast enough to do the matrix multiplications at acceptable speed. A GPU can do ~300 trillion floating-point operations per second. A CPU can do ~1 trillion. That's a 300x difference.

But VRAM is physically small. It's fabricated on the same package as the GPU chip, using expensive high-bandwidth memory (HBM) stacks. Manufacturing constraints cap it at 80-180 GB per chip in 2026.

### The Arithmetic That Forces Multi-GPU

A model with P parameters at B bytes per parameter needs P × B bytes of VRAM for weights alone:

```
Qwen3.5-397B-A17B:
  397,000,000,000 parameters × 2 bytes (BF16) = 794 GB

Best single GPU (H100):
  80 GB VRAM

794 GB ÷ 80 GB = 9.9 GPUs minimum — just for weights
```

There is no engineering trick to avoid this. The numbers must be stored somewhere the GPU can access them at HBM speed. The only option is to spread them across multiple GPUs.

**Further reading:**
- [How to Choose the Right GPU for vLLM (DigitalOcean)](https://www.digitalocean.com/community/conceptual-articles/vllm-gpu-sizing-configuration-guide) — GPU sizing calculations from first principles
- [Serving LLMs with vLLM: A Practical Guide (Nebius)](https://nebius.com/blog/posts/serving-llms-with-vllm-practical-guide) — Memory breakdown for inference

---

## 2. Why Three Different Splitting Strategies?

Once you accept that the model must be split, the question is: **along which dimension?** A neural network is a 3D object — it has width (hidden dimension), depth (number of layers), and, for MoE models, a sparsity dimension (number of experts). Each dimension gives a different splitting strategy.

### The Three Dimensions of a Transformer

```
           Width (hidden dimension)
           ◄──────────────────────►
     ┌──── ┌────────────────────────┐
     │     │ W W W W W W W W W W W W│  Layer 0
     │     ├────────────────────────┤
     │     │ W W W W W W W W W W W W│  Layer 1
  Depth    ├────────────────────────┤
 (layers)  │ W W W W W W W W W W W W│  Layer 2
     │     ├────────────────────────┤
     │     │       ...              │
     │     ├────────────────────────┤
     └──── │ W W W W W W W W W W W W│  Layer 59
           └────────────────────────┘

For MoE: each layer also has a THIRD dimension — 512 experts,
of which only 11 are active per token.
```

Each dimension suggests a natural split:

| Split Along | Strategy | Name |
|---|---|---|
| **Width** (columns of each weight matrix) | Every GPU gets a thin slice of every layer | **Tensor Parallelism (TP)** |
| **Depth** (groups of layers) | Each GPU gets a few complete layers | **Pipeline Parallelism (PP)** |
| **Sparsity** (experts in MoE) | Each GPU gets a subset of experts | **Expert Parallelism (EP)** |

There are exactly three dimensions, so there are exactly three fundamental strategies. Every hybrid approach is a combination of these three.

**Further reading:**
- [Data, Tensor, Pipeline, Expert and Hybrid Parallelisms (BentoML)](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism) — All strategies on one page with diagrams
- [NVIDIA Megatron Parallelisms Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) — NVIDIA's canonical reference

---

## 3. Why Is Tensor Parallelism the Default?

### The Motivation: Minimize Latency for a Single Request

When you split along the width dimension, every GPU participates in every token's computation. No GPU is ever idle. This gives you the **lowest possible latency** for a single request.

### Why It Works (The Linear Algebra)

A matrix multiplication `Y = X × W` can be split column-wise:

```
W = [W₁ | W₂ | W₃ | W₄]    (split across 4 GPUs)

GPU 0 computes: Y₁ = X × W₁
GPU 1 computes: Y₂ = X × W₂
GPU 2 computes: Y₃ = X × W₃
GPU 3 computes: Y₄ = X × W₄

Then: Y = [Y₁ | Y₂ | Y₃ | Y₄]    (concatenate results)
```

Each GPU does 1/4 of the multiply in parallel. The results are combined via an **AllReduce** operation — every GPU sends its partial result to every other GPU, and they all end up with the complete answer.

### The Cost: Communication After Every Layer

AllReduce must happen after every layer. For a 60-layer model, that's 60 synchronization points per token. Each one requires moving data between all GPUs.

This is why tensor parallelism **demands high-bandwidth interconnects**. If the GPUs can't synchronize fast enough, the communication time dominates and adding more GPUs makes things *slower*, not faster.

### The Rationale for Making It the Default

Despite the communication cost, tensor parallelism is preferred because:

1. **No idle GPUs**: Every GPU works on every token. No pipeline bubbles.
2. **Linear memory reduction**: 8 GPUs = 1/8 the memory per GPU. Simple.
3. **No code changes**: The same forward pass logic works — just on smaller matrices.
4. **Latency matters**: For serving, users care about how fast they get *their* response, not aggregate throughput.

**Further reading:**
- [Tensor Parallelism in Three Levels of Difficulty (Determined AI)](https://www.determined.ai/blog/tp) — From intuition to the actual AllReduce math
- [Tensor Parallel LLM Inferencing (Thomson Reuters)](https://medium.com/tr-labs-ml-engineering-blog/tensor-parallel-llm-inferencing-09138daf0ba7) — Code-level walkthrough
- [Analyzing Tensor Parallelism Configurations (ROCm)](https://rocm.blogs.amd.com/artificial-intelligence/tensor-parallelism/README.html) — What happens to performance as you increase TP size

---

## 4. Why Does Pipeline Parallelism Exist If Tensor Parallelism Is Better?

### The Constraint That Creates PP: Inter-Node Bandwidth

Tensor parallelism requires AllReduce after every layer — 60 rounds of every-GPU-talks-to-every-GPU per token. This works when GPUs are connected by NVLink (900 GB/s). It fails catastrophically when GPUs are on different physical machines connected by InfiniBand (50 GB/s) — an 18x bandwidth penalty.

Pipeline parallelism solves this by splitting along depth instead of width. Only **one** message passes between pipeline stages per token (the activation from the last layer on GPU N to the first layer on GPU N+1). Instead of 60 AllReduces, you get 1 point-to-point send.

### The Cost: Pipeline Bubbles

The tradeoff is brutal. When processing a single request:

```
Time →

GPU 0: [compute layers 0-29] [idle...................]
GPU 1: [idle...............] [compute layers 30-59]
```

GPU 0 finishes its work and goes idle while GPU 1 processes. With 2 pipeline stages, each GPU is idle 50% of the time. With 4 stages, 75% idle.

This can be partially mitigated by processing multiple requests simultaneously (microbatching), but it never fully goes away.

### The Design Decision

```
Same node (NVLink):     Use Tensor Parallelism  → all GPUs active
Across nodes (IB):      Use Pipeline Parallelism → fewer communication rounds

Combined (2 nodes × 8 GPUs):
  TP=8 within each node (NVLink handles the AllReduce)
  PP=2 between nodes (InfiniBand only carries one activation per stage)
```

This is why every multi-node deployment uses TP inside nodes and PP between nodes. It's not a preference — it's dictated by the physics of the interconnect hierarchy.

**Further reading:**
- [vLLM Parallelism & Scaling Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — When to use TP vs PP
- [Multi-node & Multi-GPU Inference with vLLM (MeluXina)](https://docs.lxp.lu/howto/llama3-vllm/) — Practical multi-node setup
- [vLLM Distributed Serving Docs](https://docs.vllm.ai/en/v0.6.0/serving/distributed_serving.html) — Ray cluster for PP across nodes

---

## 5. Why Does Expert Parallelism Exist for MoE?

### The Constraint: MoE's Memory-Computation Paradox

In a dense model, every parameter participates in every token. Memory used = compute used. Simple.

In a Mixture-of-Experts model, this relationship breaks:

```
Qwen3.5-397B-A17B:
  Memory needed: 397B parameters (all must be stored)
  Compute per token: 17B parameters (only 11 of 512 experts activate)

  Memory : Compute ratio = 397 : 17 = 23:1
```

You need 23x more memory than compute. This means if you use tensor parallelism (which splits both memory and compute equally), you're using 23x more compute capacity than necessary. Every GPU has tons of idle compute while being full on memory.

### Why Expert Parallelism Is Natural

Experts are **independent sub-networks**. Expert 47 and Expert 320 share no weights. They can live on different GPUs without any need to synchronize their internal computations.

The only coordination needed is **routing**: the router decides which experts each token needs, tokens travel to the GPU holding that expert, get processed, and results travel back. This is an All-to-All communication pattern.

```
Tensor Parallelism approach to MoE:
  Every GPU holds 1/8 of EVERY expert's weights
  Every GPU computes 1/8 of every activated expert
  AllReduce after every expert → massive wasted communication

Expert Parallelism approach to MoE:
  Each GPU holds ALL weights for 64 complete experts (512 ÷ 8)
  Only GPUs holding activated experts do computation
  All-to-All just routes tokens to the right GPU → natural, minimal
```

Expert parallelism maps the model's sparsity structure onto the hardware. Tensor parallelism ignores it. That's why EP is better for MoE.

### The Cost: All-to-All Communication

All-to-All is expensive because every GPU potentially sends data to every other GPU simultaneously. For 60 MoE layers with 2 All-to-All ops each (dispatch + combine), that's 120 communication rounds per token.

Benchmarks show All-to-All can consume **40-77% of total MoE layer processing time**. This is why the communication backend matters enormously — the `deepep_high_throughput` backend is 40-60% faster than the naive implementation.

**Further reading:**
- [The vLLM MoE Playbook (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) — Complete guide to EP for MoE
- [Mixture of Experts Explained (HuggingFace)](https://huggingface.co/blog/moe) — MoE architecture fundamentals
- [vLLM Expert Parallel Deployment Guide](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) — Official EP documentation
- [Optimizing Communication for MoE (NVIDIA)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel) — The All-to-All overhead problem

---

## 6. Why Do We Need NVLink? Why Isn't PCIe Enough?

### The Physics of the Bottleneck

GPU computation is fast. A single H100 can perform 990 trillion FP16 operations per second. At that speed, a 397B-parameter matrix multiplication finishes in milliseconds.

But the result of each layer must be shared with other GPUs before the next layer can begin. The speed of this sharing is limited by the physical interconnect between GPUs.

### The Numbers

Consider tensor parallelism with 8 GPUs across 60 layers. Each AllReduce moves roughly 100 MB of data:

```
                    Bandwidth       Time per        Total for 60
Interconnect        (GB/s)          AllReduce       layers
─────────────       ─────────       ──────────      ────────────
HBM (on-chip)       3,350           0.03 ms         1.8 ms
NVLink 4.0          900             0.11 ms         6.7 ms        ← acceptable
PCIe Gen 5          128             0.78 ms         46.9 ms       ← too slow
InfiniBand NDR      50              2.00 ms         120 ms        ← far too slow
```

The actual computation per layer takes about 0.1-0.3 ms. With NVLink, communication (0.11 ms) is comparable to computation — acceptable. With PCIe, communication (0.78 ms) is 3-8x longer than computation — the GPUs spend most of their time waiting to talk, not computing.

### The Design Consequence

This is why data center GPUs come in two variants:

- **SXM form factor**: Plugs into an NVLink-connected baseboard (NVSwitch). Used for multi-GPU inference and training. This is what you want.
- **PCIe form factor**: Plugs into a standard PCIe slot. Fine for single-GPU workloads. Terrible for multi-GPU parallelism.

When you rent an "8x H100" instance from Lambda, you're getting 8 SXM GPUs on an NVSwitch baseboard — all connected at 900 GB/s. This is not optional. It's a hard requirement for the parallelism to work.

### NVSwitch: Why Full-Bisection Matters

Without NVSwitch, NVLink connections are point-to-point: GPU 0 connects to GPU 1, GPU 1 connects to GPU 2, etc. If GPU 0 needs to talk to GPU 7, the data must hop through intermediate GPUs — adding latency and contention.

NVSwitch provides a **full crossbar**: every GPU connects directly to every other GPU at full bandwidth, simultaneously. For All-to-All communication (which expert parallelism needs), this is essential — all 8 GPUs are sending to all 8 GPUs at the same time.

**Further reading:**
- [Scaling DeepSeek with Wide-EP (Red Hat)](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep) — Why NVLink bandwidth determines EP feasibility
- [vLLM Large Scale Serving Blog](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) — Real bandwidth measurements at scale

---

## 7. Why Quantize? And Why Is FP8 the Sweet Spot?

### The Motivation: Memory Is the Binding Constraint

From first principles, inference speed is limited by whichever is slower:

1. **Compute**: How fast the GPU can multiply matrices
2. **Memory bandwidth**: How fast the GPU can read weights from VRAM
3. **Communication**: How fast GPUs can sync (discussed above)

For large models, the bottleneck is almost always **memory bandwidth**, not compute. The GPU can multiply faster than it can read the operands. This means:

- Making weights smaller → they load faster from memory → inference speeds up
- The compute savings from smaller numbers are a bonus, not the primary benefit

### Why Not Just Always Quantize to INT4?

Each reduction in precision loses information:

```
BF16 (16 bits):  Can distinguish 65,536 different values
FP8  (8 bits):   Can distinguish 256 different values
INT4 (4 bits):   Can distinguish 16 different values
```

The question is: does the model's behavior change when you can only distinguish 256 values instead of 65,536?

For most weights: no. Neural network weights cluster in narrow ranges, and the difference between 0.00238 and 0.00234 rarely matters.

But for critical weights (attention heads, router logits in MoE), precision matters more. This is why FP8 works well — 256 distinct values is enough to preserve the important distinctions — while INT4 (16 values) starts to blur things that matter.

### Why FP8 Is Especially Good for MoE

MoE models have a unique property: **95% of expert weights are inactive per token**. Quantizing an inactive expert's weights from BF16 to FP8 has **zero quality impact** — those weights don't participate in the computation. The only precision that matters is in the 11 activated experts and the shared attention layers.

This makes MoE models more quantization-friendly than dense models of the same total size. A 397B MoE at FP8 loses less quality than a 70B dense model at FP8, because the MoE's active computation path is much smaller.

**Further reading:**
- [Mixture-of-Experts LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — Why MoE models behave differently under quantization
- [MoE Architecture Deep Dive (AWS Neuron)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html) — Expert activation patterns
- [Comprehensive Survey of MoE](https://arxiv.org/html/2503.07137v1) — Quantization for MoE in the academic literature

---

## 8. Why Does Load Balancing Matter for MoE?

### The Constraint: Synchronized Execution

In tensor parallelism, every GPU does exactly 1/N of the work. The split is perfectly even by construction (each gets the same number of columns).

In expert parallelism, the split is **not even**. The router dynamically decides which experts each token uses. Some experts are popular (selected frequently), others are rarely used. If the popular experts happen to be on the same GPU, that GPU becomes a **straggler** — all other GPUs finish and wait for it.

```
Without load balancing:

GPU 0: ████████████████████████████  (Experts 12, 47 are hot)
GPU 1: ████                          (cold experts)
GPU 2: ██████                        (moderate)
GPU 7: ██                            (dead experts)

Total time = time of slowest GPU = GPU 0's time
GPU 1-7 are mostly idle. Massive waste.
```

### Why This Is Fundamental to MoE (Not Just an Engineering Bug)

Expert popularity imbalance is an inherent property of MoE models, not a flaw. During training, different experts specialize on different types of input. Some input patterns are more common than others, so some experts are needed more often. The router learns to send tokens where they'll be best processed — which means uneven distribution.

Forcing perfectly even distribution (auxiliary loss during training) hurts model quality because it prevents specialization. The imbalance is the price you pay for the model being good.

### The Solution: Dynamic Rebalancing (EPLB)

Expert Parallel Load Balancing monitors which experts are hot and cold at runtime, then:

1. **Replicates** hot experts onto multiple GPUs (so the load spreads)
2. **Migrates** cold experts to make room for replicas
3. **Periodically re-checks** because popularity shifts with different inputs

This doesn't eliminate the imbalance — it adapts the placement to match whatever the current imbalance is.

**Further reading:**
- [MoE Load Balancing Review (HuggingFace Blog)](https://huggingface.co/blog/NormalUhr/moe-balance) — The imbalance problem and all known solutions
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — How DeepSeek handles load balancing at scale
- [A Visual Guide to Mixture of Experts (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — Visual explanation of routing and balance

---

## 9. Why TP Inside Nodes and PP Between Nodes?

### The Decision Tree from First Principles

```
Q: Do I need multiple GPUs?
   (Model doesn't fit on one GPU)
   → Yes

Q: Are all GPUs in the same node with NVLink?
   ├── YES → Use Tensor Parallelism (TP)
   │         Rationale: NVLink can handle AllReduce at every layer
   │         All GPUs active on every token (no idle time)
   │
   └── NO → Use Pipeline Parallelism (PP) between nodes
             Rationale: InfiniBand can't handle AllReduce at every layer
             PP only sends one activation between stages (manageable)
             Use TP within each node for the layers assigned to it

Q: Is it an MoE model?
   ├── YES → Add Expert Parallelism (EP) within the node
   │         Rationale: Experts are independent, natural to split
   │         All-to-All needs NVLink bandwidth (same node only)
   │
   └── NO → TP alone is sufficient
```

This isn't a recommendation — it's the only viable option given the bandwidth constraints. Every other combination violates a physical bandwidth limit somewhere in the stack.

### Why Not EP Across Nodes?

Expert parallelism uses All-to-All: every GPU sends tokens to every other GPU at every MoE layer (120 rounds per forward pass). If some of those GPUs are on different nodes:

```
120 All-to-All rounds × inter-node data per round ÷ InfiniBand bandwidth
= unacceptable latency
```

You'd need dual InfiniBand (100+ GB/s) at minimum, and even then it's marginal. Within a single NVLink node, the same data moves 9-18x faster. So EP stays within the node.

**Further reading:**
- [vLLM Parallelism & Scaling Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — TP + PP composition rules
- [Deploying DeepSeek with PD Disaggregation (LMSYS)](https://lmsys.org/blog/2025-05-05-large-scale-ep/) — How LMSYS solved multi-node MoE serving
- [What is Inference Parallelism and How it Works (InfraCloud)](https://www.infracloud.io/blogs/inference-parallelism/) — The decision framework

---

## 10. Why vLLM? (And When SGLang?)

### The Motivation Behind vLLM

Before vLLM, serving LLMs meant naive batch processing: collect N requests, run them all, return results. If one request has 10 tokens of context and another has 10,000, the short one waits for the long one. Memory allocation was wasteful — reserved for maximum possible length even if most requests were short.

vLLM's core innovation is **PagedAttention**: it manages KV cache memory like an operating system manages RAM pages. Memory is allocated in small blocks, assigned on demand, and reclaimed when no longer needed. This single idea increased serving throughput by 2-24x over the previous state of the art.

On top of that, vLLM handles all the parallelism complexity described in this document:
- Tensor parallelism with NCCL AllReduce
- Pipeline parallelism with Ray orchestration
- Expert parallelism with configurable All-to-All backends
- Automatic memory management across all GPUs

### Why SGLang Is Sometimes Better for MoE

SGLang adds **RadixAttention**: a tree-based cache that automatically reuses KV cache across requests sharing common prefixes. If 100 requests all start with the same system prompt, the KV cache for that prefix is computed once and shared.

For our experiment (monitor + verifier hitting the same model with different system prompts but similar code inputs), RadixAttention provides free speedup. SGLang also has more aggressively optimized MoE kernels, benchmarking 29-33% faster on similar models.

The tradeoff: vLLM has a larger community, better documentation, and more stable multi-node support. SGLang is faster but newer.

**Further reading:**
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) — Source code and architecture
- [Mastering vLLM Multi-GPU for Faster Inference (DejaFlow)](https://www.dejaflow.com/blog/2025/03/02/mastering-vllm-multi-gpu-for-faster-ai-inference/) — Practical guide
- [vLLM Distributed Inference Optimization Guide (DatabaseMart)](https://www.databasemart.com/blog/vllm-distributed-inference-optimization-guide) — Pitfalls and solutions

---

## Summary: The Design Decision Map

Every decision traces back to a physical constraint:

| Decision | Constraint | Rationale |
|---|---|---|
| Use multiple GPUs | Model > VRAM | 794 GB doesn't fit in 80 GB |
| Use Tensor Parallelism | Need lowest latency | All GPUs work on every token, no idle time |
| Use NVLink (not PCIe) | TP needs AllReduce every layer | 900 GB/s vs 128 GB/s — 7x difference |
| Use Pipeline Parallelism between nodes | InfiniBand too slow for AllReduce | PP sends one activation per stage, not 60 AllReduces |
| Use Expert Parallelism for MoE | Experts are independent sub-networks | Natural split; avoids wasting compute on inactive experts |
| Use NVSwitch | EP needs All-to-All | Full crossbar so all GPUs talk simultaneously |
| Quantize to FP8 | Reduce memory → fit on fewer GPUs | MoE is especially forgiving (95% weights inactive) |
| Use EPLB | Expert popularity is uneven | Prevent straggler GPUs from bottlenecking the whole system |
| Use vLLM/SGLang | Need efficient KV cache + parallelism | PagedAttention / RadixAttention manage memory intelligently |
| TP within node, PP between nodes | Bandwidth hierarchy (NVLink > IB) | Match communication pattern to available bandwidth |

None of these are arbitrary choices. They are the minimum necessary responses to physical and mathematical constraints. If GPU VRAM were unlimited, we'd use one GPU. If interconnect bandwidth were infinite, we'd use tensor parallelism everywhere. If expert popularity were uniform, we wouldn't need load balancing. Each constraint creates exactly one design decision.
