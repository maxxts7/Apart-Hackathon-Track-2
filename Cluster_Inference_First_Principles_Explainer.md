# Deploying Large Models on GPU Clusters: A First-Principles Explainer

*From "why doesn't my model fit on one GPU?" to running inference across a cluster*

---

## Part 1: The Fundamental Problem — Models Are Bigger Than GPUs

A neural network is, at its core, a massive collection of numbers (called **parameters** or **weights**). During inference, these numbers must all be accessible in fast memory (GPU VRAM) so the GPU can do matrix multiplications on them.

Here's the math that creates the problem:

### How Much Memory Does a Model Need?

Each parameter is stored in a numeric format:

| Format | Bytes per Parameter | What It Means |
|---|---|---|
| FP32 (full precision) | 4 bytes | 32-bit floating point — maximum accuracy |
| BF16 / FP16 (half precision) | 2 bytes | 16-bit — standard for inference, negligible quality loss |
| FP8 (quarter precision) | 1 byte | 8-bit — slight quality loss, much faster |
| INT4 (4-bit quantization) | 0.5 bytes | 4-bit integers — noticeable quality loss, very compact |

So for **Qwen3.5-397B-A17B** (397 billion parameters):

| Format | Memory for Weights |
|---|---|
| BF16 | 397B × 2 bytes = **794 GB** |
| FP8 | 397B × 1 byte = **397 GB** |
| INT4 | 397B × 0.5 bytes = **~199 GB** |

### How Much VRAM Does a GPU Have?

| GPU | VRAM | Price (Cloud) |
|---|---|---|
| RTX 4090 (consumer) | 24 GB | N/A (buy it) |
| A100 SXM | 80 GB | ~$1.79/hr |
| H100 SXM | 80 GB | ~$2.99/hr |
| B200 | 180 GB | ~$4.99/hr |

The biggest single GPU has 180 GB. The model needs 794 GB in BF16. **It doesn't fit.** Not even close.

This is like trying to load a 794 GB file into a computer with 80 GB of RAM. You need to split it across multiple machines.

### But Wait — Inference Needs More Than Just Weights

During inference, the GPU also needs memory for:

1. **KV Cache**: Stores intermediate attention states for each token in the sequence. Grows linearly with context length and batch size.
2. **Activations**: Temporary computation results during the forward pass.
3. **Framework overhead**: CUDA contexts, memory allocators, etc.

Rule of thumb: reserve **20-40% of VRAM beyond model weights** for KV cache and overhead.

For Qwen3.5-397B at 131K context in BF16:
- Weights: ~794 GB
- KV cache (1 request): ~80 GB
- Overhead: ~50 GB
- **Total: ~924 GB**

That's **12 GPUs worth** of A100 80GB memory, minimum.

---

## Part 2: The Solution — Split the Model Across GPUs

If the model doesn't fit on one GPU, split it across many. There are three ways to do this, each splitting along a different dimension.

Think of a neural network as a sequence of layers, where each layer is a big matrix multiplication:

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → ... → [Layer 60] → Output
         ┌────┐      ┌────┐      ┌────┐             ┌────┐
         │WWWW│      │WWWW│      │WWWW│             │WWWW│
         │WWWW│      │WWWW│      │WWWW│             │WWWW│
         │WWWW│      │WWWW│      │WWWW│             │WWWW│
         │WWWW│      │WWWW│      │WWWW│             │WWWW│
         └────┘      └────┘      └────┘             └────┘
        (weights)   (weights)   (weights)          (weights)
```

### Strategy 1: Tensor Parallelism (TP) — Split Each Layer Horizontally

Slice every weight matrix into chunks, one per GPU. Each GPU holds a piece of every layer.

```
                Layer 1
GPU 0:  [WW]    ← top half of weight matrix
GPU 1:  [WW]    ← bottom half of weight matrix

Both GPUs compute their part simultaneously, then combine results.
```

**How it works step by step:**
1. The input tensor is broadcast to all GPUs
2. Each GPU multiplies the input by its slice of the weight matrix
3. GPUs communicate to combine their partial results (AllReduce operation)
4. The combined result becomes the input to the next layer
5. Repeat for every layer

**Pros:**
- All GPUs are active on every token (no idle time)
- Reduces per-GPU memory linearly with GPU count
- Low latency (single token processed in parallel)

**Cons:**
- Requires **AllReduce communication after every layer** — GPUs must constantly synchronize
- Needs high-bandwidth interconnect (NVLink, not PCIe)
- Communication overhead grows with GPU count

**When to use**: Always the default for inference. It's the bread and butter.

```bash
# vLLM: tensor parallel across 8 GPUs
vllm serve Qwen/Qwen3.5-397B-A17B --tensor-parallel-size 8
```

### Strategy 2: Pipeline Parallelism (PP) — Split Layers Across GPUs

Instead of splitting each layer, assign entire layers to different GPUs. GPU 0 gets layers 0-7, GPU 1 gets layers 8-15, etc.

```
GPU 0:  [Layer 0] → [Layer 1] → ... → [Layer 7]
                                           │
                                           ▼ (send activation to GPU 1)
GPU 1:  [Layer 8] → [Layer 9] → ... → [Layer 15]
                                           │
                                           ▼ (send activation to GPU 2)
GPU 2:  [Layer 16] → [Layer 17] → ... → [Layer 23]
                                           │
                                           ▼
                                         ...
```

**How it works:**
1. Input enters GPU 0, which runs layers 0-7
2. GPU 0 sends the output activation to GPU 1
3. GPU 1 runs layers 8-15 and sends to GPU 2
4. And so on until the final GPU produces the output

**Pros:**
- Minimal communication (only sends activations between pipeline stages, not at every layer)
- Works across nodes (only needs inter-node bandwidth between stages)
- Each GPU holds fewer layers → less memory per GPU

**Cons:**
- **Pipeline bubble**: At any moment, only one GPU in the pipeline is active for a given token. The others are idle, waiting for data. With 4 pipeline stages, each GPU is idle 75% of the time for a single request.
- Higher latency per token (sequential processing)

**When to use**: When you need to span multiple nodes (inter-node bandwidth is too slow for tensor parallelism's AllReduce).

```bash
# vLLM: 2 nodes × 8 GPUs each, pipeline parallel across nodes
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

### Strategy 3: Expert Parallelism (EP) — For MoE Models Only

This is specific to **Mixture-of-Experts** (MoE) models like Qwen3.5-397B-A17B. Here's why it exists and what makes MoE special.

---

## Part 3: What Is Mixture-of-Experts and Why Does It Change Everything?

### The MoE Architecture

A standard (dense) transformer runs every token through every parameter. If the model has 397B parameters, all 397B activate for each token.

A MoE transformer has many parallel "expert" sub-networks in each layer, and a **router** that selects only a few experts per token:

```
Standard Dense Layer:
   Input → [One Big FFN with all parameters] → Output
   (All parameters active)

MoE Layer:
   Input → [Router] → selects Expert 3, Expert 47, Expert 201
                          │          │            │
                          ▼          ▼            ▼
                      [Expert 3] [Expert 47] [Expert 201]  ← Only these 3 activate
                          │          │            │
                          └──── combine ──────────┘
                                   │
                                   ▼
                                Output
   (512 experts exist, but only 11 activate per token)
```

For Qwen3.5-397B-A17B:
- **512 total experts** per MoE layer
- **11 experts activated** per token (10 routed + 1 shared)
- This means only **17B out of 397B parameters** fire per token
- But all 512 experts must be **stored in memory** because any token might route to any expert

This is the paradox of MoE: **computation is cheap (17B active) but memory is expensive (397B stored)**.

### Expert Parallelism: The Natural Solution

Since each GPU holds a subset of experts, and the router decides which experts each token needs, tokens must be **sent to the GPU that holds their selected expert**:

```
GPU 0 holds: Experts 0-63          GPU 1 holds: Experts 64-127
GPU 2 holds: Experts 128-191       GPU 3 holds: Experts 192-255
GPU 4 holds: Experts 256-319       GPU 5 holds: Experts 320-383
GPU 6 holds: Experts 384-447       GPU 7 holds: Experts 448-511

Token X → Router says: use Expert 47 and Expert 320
  → Token X is sent to GPU 0 (for Expert 47)
  → Token X is sent to GPU 5 (for Expert 320)
  → Results are sent back and combined
```

This requires **All-to-All communication**: every GPU potentially sends tokens to every other GPU, and receives processed results back. This happens at every MoE layer (60 layers × 2 All-to-All ops = 120 communication rounds per forward pass).

**Pros:**
- Naturally matches MoE architecture (experts are independent, so splitting them is clean)
- Each GPU only computes its own experts (work distribution matches data distribution)
- Scales well with more GPUs (more GPUs = fewer experts per GPU = less memory each)

**Cons:**
- All-to-All communication is expensive — can be **40-77% of total MoE layer time**
- Requires very high bandwidth between GPUs
- Load imbalance: some experts are more popular than others, creating "straggler" GPUs

```bash
# vLLM: enable expert parallelism for MoE
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --all2all-backend deepep_high_throughput
```

---

## Part 4: GPU Interconnects — The Highway System Between GPUs

Parallelism strategies require GPUs to talk to each other constantly. The speed of this communication determines whether parallelism actually helps or becomes a bottleneck.

### The Interconnect Hierarchy

Think of GPU communication like a highway system:

```
Within a GPU:  HBM (High Bandwidth Memory)
               ████████████████████████████████
               3,350 GB/s (H100) — the autobahn

Between GPUs   NVLink
(same node):   ═══════════════════════════
               900 GB/s (H100) — a 16-lane highway

Between GPUs   PCIe Gen 5
(no NVLink):   ────────────────
               128 GB/s — a 2-lane road

Between nodes  InfiniBand NDR
(network):     ·········
               50 GB/s — a dirt path (relatively)
```

### Why NVLink Matters

Tensor parallelism requires AllReduce after every layer. If you have 60 layers and each AllReduce moves ~100 MB of data:

| Interconnect | Time per AllReduce | Total for 60 Layers |
|---|---|---|
| NVLink (900 GB/s) | 0.11 ms | 6.7 ms |
| PCIe (128 GB/s) | 0.78 ms | 46.9 ms |
| InfiniBand (50 GB/s) | 2.0 ms | 120 ms |

NVLink is **7x faster** than PCIe and **18x faster** than InfiniBand for this pattern. This is why tensor parallelism only works within a single node connected by NVLink.

For expert parallelism, the All-to-All pattern is even more bandwidth-hungry (every GPU sends to every other GPU simultaneously). Without NVLink, MoE inference on multiple GPUs is impractical.

### NVSwitch: Full-Bisection Bandwidth

In an 8-GPU node (like Lambda's 8x H100 SXM), NVSwitch connects all GPUs with full-bisection bandwidth. This means every GPU can talk to every other GPU at full speed simultaneously — there's no contention.

```
     ┌─────────────────────────┐
     │       NVSwitch           │
     │   (Full Crossbar)        │
     └┬──┬──┬──┬──┬──┬──┬──┬──┘
      │  │  │  │  │  │  │  │
     GPU GPU GPU GPU GPU GPU GPU GPU
      0   1   2   3   4   5   6   7

     Each pair connected at 900 GB/s
     No contention — all pairs can communicate simultaneously
```

This is essential for expert parallelism: when 120 All-to-All operations happen per forward pass, any bottleneck is catastrophic.

### Verifying Your Interconnect

When you SSH into a multi-GPU instance, always verify:

```bash
nvidia-smi topo -m
```

Good output (NVLink everywhere):
```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
...
```

Bad output (PCIe fallback):
```
        GPU0  GPU1  GPU2  GPU3
GPU0     X    PHB   SYS   SYS
GPU1    PHB    X    SYS   SYS
...
```

`NV18` = 18 NVLink connections (H100). `PHB`/`SYS` = PCIe. If you see PCIe between GPUs you need for tensor or expert parallelism, stop — your inference will be 7x slower than expected.

---

## Part 5: Putting It All Together — The Optimal Strategy for Qwen3.5-397B

### Step 1: Calculate Memory Requirements

```
Model weights (BF16):     794 GB
KV cache (131K, batch=1): ~80 GB
Overhead:                 ~50 GB
Total:                    ~924 GB
```

### Step 2: Choose GPU Count

```
924 GB ÷ 80 GB per H100 = 11.6 GPUs minimum
Round up to 8 GPUs (standard node size) — won't fit with full context.

Options:
A) 8x H100 with reduced context (65K instead of 131K)  → fits
B) 8x H100 with FP8 weights (halves weight memory)     → fits comfortably
C) 8x H100 with FP8 weights + FP8 KV cache             → fits with room for batching
```

### Step 3: Choose Parallelism Strategy

For 8 GPUs on a single node with NVLink:

```
Attention layers:  Tensor Parallelism (TP=8)
  → AllReduce at each attention layer
  → NVLink provides sufficient bandwidth

MoE FFN layers:    Expert Parallelism (EP=8)
  → 512 experts ÷ 8 GPUs = 64 experts per GPU
  → All-to-All communication for token routing
  → NVLink provides sufficient bandwidth

Combined: TP=8 + EP=8 on single 8-GPU node
```

### Step 4: The Full Command

```bash
# SSH into Lambda 8x H100 instance
ssh -i ~/.ssh/key.pem ubuntu@<IP>

# Verify NVLink
nvidia-smi topo -m

# Download model (takes ~1-2 hours)
pip install vllm huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3.5-397B-A17B

# Serve
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --all2all-backend deepep_high_throughput \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.8 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### Step 5: What Happens During a Single Inference Request

Let's trace what happens when you send "Review this code for backdoors" to the model:

```
1. TOKENIZATION (CPU)
   "Review this code..." → [15496, 420, 2191, ...]

2. EMBEDDING (GPU 0, broadcast to all)
   Token IDs → embedding vectors
   Broadcast to all 8 GPUs

3. FOR EACH OF 60 LAYERS:

   3a. ATTENTION (Tensor Parallel across 8 GPUs)
       Each GPU computes attention for its slice of heads
       AllReduce to combine results (NVLink, ~0.1ms)

   3b. MoE ROUTER (computed on each GPU)
       For each token, router selects 11 out of 512 experts

   3c. ALL-TO-ALL DISPATCH (Expert Parallel, NVLink)
       Tokens sent to the GPU holding their selected experts
       GPU 0 sends tokens needing experts 64-511 to other GPUs
       GPU 0 receives tokens from other GPUs needing experts 0-63

   3d. EXPERT COMPUTATION (Local to each GPU)
       Each GPU runs its experts on the tokens routed to it
       Only 11 experts activate per token (out of 64 on each GPU)

   3e. ALL-TO-ALL COMBINE (Expert Parallel, NVLink)
       Results sent back to the originating GPU
       Combined with shared expert output

   3f. RESIDUAL + NORM (Local)
       Add residual connection, apply layer norm

4. FINAL LOGITS (GPU 0)
   Last hidden state → vocabulary logits
   Sample next token

5. REPEAT from step 3 for each output token (autoregressive)
```

Each output token requires this full forward pass. At ~45 tokens/second on 8x H100, the entire cycle completes in ~22 milliseconds.

---

## Part 6: Quantization — Trading Quality for Speed and Memory

When full-precision weights don't fit, or you want more room for batching, quantization compresses the weights by using fewer bits per number.

### The Intuition

A weight value like `0.0023847562` stored in BF16 uses 16 bits. But does the model really need all those decimal places? Often, rounding to `0.00238` (or even mapping to a small integer) barely changes the output.

### Quantization Levels

```
BF16:  ████████████████  (16 bits)  — Full quality, full size
FP8:   ████████          (8 bits)   — ~0.25% quality loss, half the size
INT4:  ████              (4 bits)   — ~1-3% quality loss, quarter size
```

For Qwen3.5-397B-A17B:

| Format | Weight Memory | GPUs Needed | Quality Impact |
|---|---|---|---|
| BF16 | 794 GB | 8x H100 (tight) | Baseline |
| FP8 | 397 GB | 4x H100 (comfortable) | < 0.25% loss |
| INT4 (NVFP4) | ~199 GB | 2x H100 (comfortable) | ~1-3% loss |

### FP8: The Sweet Spot for Large MoE Models

FP8 quantization is particularly effective for MoE models:

- MoE models have many experts, most of which are inactive per token. Quantizing inactive expert weights costs almost nothing in quality because they don't participate.
- The 17B active parameters still get FP8 precision, which is nearly lossless for inference.
- Official FP8 checkpoints exist for Qwen3.5-397B (`Qwen/Qwen3.5-397B-A17B-FP8`).

```bash
# FP8 — halves memory, fits on 4x H100
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8
```

### KV Cache Quantization

Separately from weight quantization, you can quantize the KV cache (the stored attention states):

```bash
--kv-cache-dtype fp8  # Halves KV cache memory
```

This is independent of weight quantization and is almost always worth enabling for long-context inference. The quality impact is minimal because KV cache values have relatively low precision requirements.

---

## Part 7: The Data Flow — From Your Python Script to 8 GPUs and Back

Here's the full stack from your application code to the hardware:

```
┌──────────────────────────────────────────┐
│  Your Python Script                       │
│  client = OpenAI(base_url="...")          │
│  client.chat.completions.create(...)     │
└──────────────────┬───────────────────────┘
                   │ HTTP POST (JSON)
                   ▼
┌──────────────────────────────────────────┐
│  vLLM API Server (FastAPI)               │
│  OpenAI-compatible endpoint              │
│  Tokenizes input, manages request queue  │
└──────────────────┬───────────────────────┘
                   │ Internal
                   ▼
┌──────────────────────────────────────────┐
│  vLLM Scheduler                          │
│  Batches requests for throughput         │
│  Manages KV cache allocation             │
│  Decides which requests to process next  │
└──────────────────┬───────────────────────┘
                   │ Batch of token IDs
                   ▼
┌──────────────────────────────────────────┐
│  vLLM Model Runner                       │
│  Distributes computation across GPUs     │
│  Manages tensor parallel / expert parallel│
│                                          │
│  GPU 0 ←NVLink→ GPU 1 ←NVLink→ GPU 2   │
│  GPU 3 ←NVLink→ GPU 4 ←NVLink→ GPU 5   │
│  GPU 6 ←NVLink→ GPU 7                   │
│           (via NVSwitch)                 │
│                                          │
│  Each GPU: ~99 GB weights (BF16/8 GPUs) │
│           + KV cache slice              │
│           + activation memory           │
└──────────────────┬───────────────────────┘
                   │ Logits → sampled token
                   ▼
┌──────────────────────────────────────────┐
│  vLLM Detokenizer                        │
│  Token IDs → text                        │
│  Streams back via SSE or returns JSON    │
└──────────────────┬───────────────────────┘
                   │ HTTP Response (JSON)
                   ▼
┌──────────────────────────────────────────┐
│  Your Python Script                       │
│  response.choices[0].message.content     │
└──────────────────────────────────────────┘
```

The key insight: from your code's perspective, it's just an HTTP call to an OpenAI-compatible API. All the multi-GPU complexity is hidden behind vLLM.

```python
from openai import OpenAI

# This works identically whether the model is on 1 GPU or 8
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=[
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": "Check this code for backdoors..."},
    ],
    temperature=0.3,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

---

## Part 8: Multi-Node Inference — When One Machine Isn't Enough

Sometimes even 8 GPUs on one machine aren't enough (e.g., running the model at full BF16 with very long context). You need to span multiple physical servers.

### The Challenge: Network Bandwidth

Within a single node, NVLink provides 900 GB/s between GPUs. Between nodes, even the best InfiniBand provides only 50-72 GB/s — **12-18x slower**.

This means you must minimize inter-node communication:

```
Node 0 (8x H100, NVLink)              Node 1 (8x H100, NVLink)
┌─────────────────────────┐           ┌─────────────────────────┐
│ GPU0 GPU1 GPU2 GPU3     │           │ GPU0 GPU1 GPU2 GPU3     │
│ GPU4 GPU5 GPU6 GPU7     │           │ GPU4 GPU5 GPU6 GPU7     │
│                         │           │                         │
│ Layers 0-29             │    IB     │ Layers 30-59            │
│ (Tensor Parallel inside)│◄────────► │ (Tensor Parallel inside)│
│                         │  50 GB/s  │                         │
└─────────────────────────┘           └─────────────────────────┘
      Pipeline Stage 0                      Pipeline Stage 1
```

**The strategy**: Tensor parallelism within each node (uses NVLink), pipeline parallelism between nodes (uses InfiniBand). This way, only activations pass between nodes (once per pipeline stage), not the constant AllReduce traffic.

### Ray Cluster Setup

vLLM uses Ray for multi-node coordination:

```bash
# On Node 0 (head node)
ray start --head --port=6379

# On Node 1 (worker)
ray start --address=<NODE_0_IP>:6379

# Then on Node 0, launch vLLM
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

### When Do You Actually Need Multi-Node?

For Qwen3.5-397B-A17B:

| Scenario | Single Node (8x H100) | Multi-Node Needed? |
|---|---|---|
| FP8, context ≤ 131K | Fits | No |
| BF16, context ≤ 65K | Fits (tight) | No |
| BF16, context = 262K | Doesn't fit | Yes (2 nodes) |
| BF16, context = 1M (YaRN) | Doesn't fit | Yes (4+ nodes) |

**For our hackathon experiment**: Single node with FP8 is sufficient. We're doing code review (short context), not document analysis.

---

## Part 9: Load Balancing and the Expert Imbalance Problem

MoE models have a unique challenge: **expert popularity is not uniform**. Some experts get selected by the router much more often than others.

### The Problem

```
GPU 0 holds Experts 0-63:    ██████████████████████ (Experts 12 and 47 are very popular)
GPU 1 holds Experts 64-127:  ████ (few popular experts)
GPU 2 holds Experts 128-191: ██████ (moderate)
...
GPU 7 holds Experts 448-511: ██ (rarely used experts)
```

GPU 0 is doing 5x more work than GPU 7. The entire system waits for GPU 0 to finish before moving to the next layer. GPU 7 sits idle. This is the **straggler problem**.

### The Solution: Expert Parallel Load Balancing (EPLB)

vLLM's EPLB dynamically rebalances expert placement:

1. **Monitor**: Track which experts are hot (frequently selected) and which are cold
2. **Replicate**: Copy hot experts to additional GPUs
3. **Rebalance**: Move experts between GPUs to equalize load
4. **Repeat**: Periodically recheck and adjust

```bash
--enable-eplb  # Enable dynamic expert load balancing
```

After rebalancing:
```
GPU 0: Experts 0-55 + copy of Expert 200   ████████████ (balanced)
GPU 1: Experts 56-120 + copy of Expert 12  ████████████ (balanced)
GPU 2: Experts 121-190                      ████████████ (balanced)
...
```

The hot experts get replicated across multiple GPUs so no single GPU is a bottleneck.

---

## Part 10: Monitoring and Debugging Your Cluster

### During Model Loading

Model loading for a 397B model takes 5-15 minutes. Watch progress:

```bash
# Terminal 1: GPU memory usage (should fill gradually)
watch -n 2 nvidia-smi

# Terminal 2: vLLM logs
# (vLLM prints loading progress to stdout)
```

### During Inference

```bash
# Real-time GPU utilization (should be 80-95% during inference)
nvidia-smi dmon -s pucvmet

# Columns: power, utilization, memory, encoder, clock, temperature
```

**Healthy indicators:**
- GPU utilization: 80-95% (some idle time is normal during communication)
- Memory used: close to `--gpu-memory-utilization` fraction
- All 8 GPUs at similar utilization (no stragglers)
- Temperature: < 80°C

**Unhealthy indicators:**
- One GPU at 100%, others at 10% → expert imbalance, enable EPLB
- All GPUs at < 50% → communication bottleneck, check NVLink topology
- Frequent OOM errors → reduce `--max-model-len` or `--gpu-memory-utilization`
- Increasing latency over time → KV cache fragmentation, may need restart

### Throughput Testing

```bash
# Send a test request and measure time
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-397B-A17B","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Tokens: {r[\"usage\"][\"completion_tokens\"]}')"
```

---

## Part 11: Decision Flowchart

```
START: I need to serve a large model
  │
  ├─ Does it fit on 1 GPU in BF16?
  │   │
  │   YES → Use 1 GPU, done.
  │   │
  │   NO ↓
  │
  ├─ Does it fit on 1 GPU with INT4 quantization?
  │   │
  │   YES → Is quality loss acceptable?
  │   │      YES → Use 1 GPU + quantization
  │   │      NO  ↓
  │   NO ↓
  │
  ├─ Is it an MoE model?
  │   │
  │   YES → Use Tensor Parallel + Expert Parallel
  │   │      with --enable-expert-parallel --enable-eplb
  │   │
  │   NO  → Use Tensor Parallel only
  │          with --tensor-parallel-size N
  │
  ├─ Does it fit on 1 node (8 GPUs)?
  │   │
  │   YES → Single node, TP=8 (+ EP if MoE)
  │   │      NVLink required between all GPUs
  │   │
  │   NO  → Multi-node: TP within node, PP between nodes
  │          InfiniBand between nodes
  │          Ray for orchestration
  │
  └─ Can I use FP8 to make it fit on fewer GPUs?
      │
      YES → Use FP8 checkpoint (< 0.25% quality loss)
      │      --kv-cache-dtype fp8 for extra savings
      │
      NO  → Need more GPUs or accept longer context limits
```

---

## Appendix: Glossary

| Term | Definition |
|---|---|
| **VRAM** | Video RAM — the GPU's fast local memory where model weights and computations live |
| **Tensor Parallelism (TP)** | Splitting each layer's weight matrix across GPUs; requires AllReduce after every layer |
| **Pipeline Parallelism (PP)** | Assigning entire layers to different GPUs; sequential processing with pipeline bubbles |
| **Expert Parallelism (EP)** | Assigning MoE experts to different GPUs; requires All-to-All communication |
| **AllReduce** | Communication pattern where all GPUs combine their partial results into a single result shared by all |
| **All-to-All** | Communication pattern where every GPU sends data to every other GPU (used in EP) |
| **NVLink** | High-bandwidth interconnect between GPUs on the same node (900 GB/s on H100) |
| **NVSwitch** | Chip that enables full-bisection NVLink bandwidth between all GPUs in a node |
| **InfiniBand** | High-bandwidth network between nodes (50-72 GB/s) |
| **PCIe** | Standard expansion bus; too slow for tensor/expert parallelism (128 GB/s) |
| **KV Cache** | Stored attention key-value pairs from previous tokens; grows with context length |
| **MoE** | Mixture of Experts — architecture where only a subset of parameters activate per token |
| **Router** | The component in MoE that decides which experts process each token |
| **EPLB** | Expert Parallel Load Balancing — dynamically rebalancing expert placement across GPUs |
| **Quantization** | Reducing the precision of model weights (BF16→FP8→INT4) to save memory |
| **vLLM** | Open-source inference engine with built-in support for all parallelism strategies |
| **SGLang** | Alternative inference engine; often faster for MoE models due to RadixAttention |
| **Ray** | Distributed computing framework used by vLLM for multi-node orchestration |
| **Pipeline bubble** | Idle time in pipeline parallelism where GPUs wait for data from the previous stage |
| **Straggler** | A GPU that takes longer than others, causing all GPUs to wait for it |
