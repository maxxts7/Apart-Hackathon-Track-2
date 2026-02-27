# Deep Dive: Cluster Inference for Qwen3.5-397B-A17B

## Model Architecture at a Glance

| Property | Value |
|---|---|
| Total parameters | 397B |
| Active parameters per token | 17B (4.3% activation) |
| Architecture | Sparse MoE with hybrid attention |
| Total experts | 512 |
| Experts activated per token | 11 (10 routed + 1 shared) |
| Attention | Gated Delta Networks (linear, 3:1 ratio) + Full Attention (softmax, 1:1 ratio) |
| Layers | 60 MoE layers |
| KV heads | 32 |
| Head dim | 128 |
| Native context | 262K tokens (extendable to 1M with YaRN) |
| License | Apache 2.0 |

The key thing about MoE: **all 397B parameters must be resident in GPU memory** even though only 17B fire per token. The routing is dynamic — any token could select any expert — so you can't just load a subset.

---

## 1. Memory Requirements

### Model Weights

| Precision | Total Size | Per GPU (8 GPUs) | Per GPU (4 GPUs) |
|---|---|---|---|
| **BF16** | ~794 GB | ~99 GB | ~199 GB |
| **FP8** | ~397 GB | ~50 GB | ~99 GB |
| **INT4 (NVFP4)** | ~214-224 GB | ~28 GB | ~56 GB |

### KV Cache (per request, per context length)

| Context Length | FP16 | FP8 |
|---|---|---|
| 32K | ~20 GB | ~10 GB |
| 64K | ~40 GB | ~20 GB |
| 131K | ~80 GB | ~40 GB |
| 262K (native max) | ~160 GB | ~80 GB |

KV cache scales linearly with batch size: batch_size=4 at 32K in FP8 = 40 GB total.

### What Actually Fits Where

| GPU Config | Total VRAM | BF16 weights fit? | FP8 weights fit? | Room for KV cache? |
|---|---|---|---|---|
| **4x H100 80GB** | 320 GB | No (794 GB) | Barely (397 GB) | Very little |
| **8x H100 80GB** | 640 GB | Yes (99 GB/GPU) | Yes (50 GB/GPU) | ~30 GB/GPU for KV |
| **8x A100 80GB** | 640 GB | Yes (99 GB/GPU) | Yes (50 GB/GPU) | ~30 GB/GPU for KV |
| **4x B200 180GB** | 720 GB | Yes (199 GB/GPU) | Yes, with room | Plenty |

**Bottom line: 8x H100 80GB is the minimum practical config for BF16. 4x H100 can work with FP8 but tight.**

---

## 2. GPU Configurations and Pricing

### Lambda Labs

| Instance | GPUs | Total VRAM | NVLink | Price/hr | 12 hrs |
|---|---|---|---|---|---|
| 8x A100 SXM 80GB | 8 | 640 GB | 600 GB/s (NVLink 3.0) | ~$14.32 | ~$172 |
| **8x H100 SXM 80GB** | 8 | 640 GB | 900 GB/s (NVLink 4.0) | **~$23.92** | **~$287** |
| 8x B200 | 8 | 1440 GB | Next-gen | ~$39.92 | ~$479 |

### Alternatives

| Provider | 8x H100 Price/hr | Notes |
|---|---|---|
| Lambda Labs | ~$23.92 | Zero egress, NVLink |
| RunPod | ~$21.52 | Per-second billing |
| Together.ai (API) | $0.60/$3.60 per M tokens | No infra needed |
| Alibaba Cloud (API) | $0.40/$2.40 per M tokens | Cheapest managed API |

### Cost for Our Experiment (~5,000 calls)

| Approach | Estimate |
|---|---|
| Self-hosted 8x H100, ~10 hrs | **~$240** |
| Together.ai API, ~5K calls × ~1K tokens | **~$20** |
| Alibaba Cloud API, same | **~$14** |

**The managed API is 10-15x cheaper for our volume.** Self-hosting only makes sense if we need full control over prompting, batching, or want to avoid rate limits.

---

## 3. Parallelism Strategies for MoE

There are three types of parallelism, and MoE models benefit from a specific combination:

### Tensor Parallelism (TP)

Slices each layer's weight matrices across GPUs. Every GPU participates in every token's computation.

- **Good for**: Dense attention layers
- **Bad for**: MoE layers (doesn't exploit sparsity)
- **Communication**: AllReduce after every layer (~moderate bandwidth)

### Pipeline Parallelism (PP)

Assigns different layers to different GPUs. GPU 0 does layers 0-7, GPU 1 does layers 8-15, etc.

- **Good for**: Reducing per-GPU memory
- **Bad for**: Latency (sequential pipeline bubbles)
- **Communication**: Point-to-point between pipeline stages

### Expert Parallelism (EP) — Best for MoE

Assigns complete experts to different GPUs. Each GPU holds a subset of all 512 experts. Tokens get routed to the GPU holding their selected expert.

- **Good for**: MoE layers specifically
- **Bad for**: Dense attention layers (doesn't help)
- **Communication**: All-to-All at every MoE layer (tokens dispatched to expert-hosting GPU, results returned)

### The Optimal Combo

DeepSeek-V3's approach (proven at scale):
```
Dense Attention layers  →  Data Parallel (DP) or Tensor Parallel (TP)
MoE FFN layers          →  Expert Parallel (EP)
```

For Qwen3.5-397B on 8x H100:
```
TP=8 for attention + EP=8 for MoE layers
```

vLLM handles this automatically when you enable expert parallelism.

---

## 4. vLLM Deployment Commands

### Recommended: 8x H100, BF16, Full Precision

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.8 \
  --language-model-only \
  --enable-prefix-caching \
  --reasoning-parser qwen3
```

### With Expert Parallelism (Higher Throughput)

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --all2all-backend deepep_high_throughput \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.8 \
  --language-model-only \
  --enable-prefix-caching
```

### FP8 on 4x H100 (Cheaper, Slightly Lower Quality)

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### INT4 on 4x H100 (Most Memory Efficient)

```bash
vllm serve nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --language-model-only
```

### Multi-Node: 2 Nodes × 8x H100 (16 GPUs)

```bash
# Requires Ray cluster setup
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.8 \
  --language-model-only
```

### Key Flags Explained

| Flag | Purpose |
|---|---|
| `--tensor-parallel-size 8` | Split model across 8 GPUs |
| `--enable-expert-parallel` | Route experts to specific GPUs (MoE optimization) |
| `--enable-eplb` | Dynamic expert load balancing (rebalances hot experts) |
| `--all2all-backend deepep_high_throughput` | Optimized all-to-all communication kernel |
| `--language-model-only` | Skip vision encoder (~20-30 GB savings) |
| `--kv-cache-dtype fp8` | Halve KV cache memory |
| `--enable-prefix-caching` | Reuse KV cache across requests sharing prefixes |
| `--gpu-memory-utilization 0.8` | Reserve 20% for overhead (increase to 0.9 if stable) |
| `--max-model-len 131072` | Cap context to save KV memory; 131K enough for our use case |
| `--reasoning-parser qwen3` | Enable reasoning/thinking token extraction |

---

## 5. SGLang Alternative (Faster for MoE)

SGLang benchmarks show **29-33% higher throughput** than vLLM on similar MoE models, thanks to RadixAttention and fused MoE kernels.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B \
  --port 30000 \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 131072 \
  --reasoning-parser qwen3
```

| | vLLM | SGLang |
|---|---|---|
| Throughput | 12,553 tok/s | 16,215 tok/s |
| MoE optimization | Good | 33% faster |
| Prefix caching | Manual flag | Automatic (RadixAttention) |
| Maturity | Stable, large community | Newer, rapid development |
| Multi-turn benefit | Minimal | Significant (auto KV reuse) |

**Recommendation**: SGLang if throughput is priority. vLLM if you need stability and ecosystem compatibility.

---

## 6. All-to-All Communication Backends

Expert parallelism requires all-to-all communication at every MoE layer (60 layers × 2 ops each = 120 rounds per forward pass). The backend matters a lot:

| Backend | Best For | Performance | Notes |
|---|---|---|---|
| `naive` | Testing only | Slow | Basic broadcast |
| `allgather_reducescatter` | Default fallback | Good | Reliable |
| `deepep_high_throughput` | Batch size > 128 | **Best** | 40-60% faster than naive |
| `deepep_low_latency` | Batch size < 128 | **Best** | Optimized for low concurrency |
| `flashinfer_all2allv` | New hardware | Good | FlashInfer kernel optimized |

For our use case (~5K sequential calls, batch ~32): use `deepep_high_throughput`.

**Communication overhead can be 40-77% of total MoE layer time.** This is why NVLink is non-negotiable — PCIe (128 GB/s) is 7x slower than H100 NVLink (900 GB/s).

---

## 7. Network Requirements

### Intra-Node (GPU-to-GPU)

| Interconnect | Bandwidth | Sufficient for 397B MoE? |
|---|---|---|
| **H100 NVLink 4.0** | 900 GB/s per GPU | Yes — recommended |
| **A100 NVLink 3.0** | 600 GB/s per GPU | Yes but 33% slower |
| **PCIe Gen 5** | 128 GB/s | No — communication bottleneck |

**Lambda's 8x H100 SXM instances use NVLink via NVSwitch**, providing full-bisection bandwidth between all 8 GPUs. This is the correct topology.

### Inter-Node (Node-to-Node, if multi-node)

| Interconnect | Bandwidth | Sufficient? |
|---|---|---|
| **InfiniBand NDR** | 400 Gbps (50 GB/s) | Marginal — pipeline parallel only |
| **InfiniBand XDR** | 576 Gbps (72 GB/s) | Better |
| **Dual InfiniBand** | 100+ GB/s | Recommended for EP across nodes |

**For our experiment**: Single-node 8x H100 is sufficient. No multi-node needed.

---

## 8. Step-by-Step Lambda Setup

### Step 1: Provision Instance

1. Go to [lambda.ai/instances](https://lambda.ai/instances)
2. Select **8x H100 SXM** (on-demand)
3. OS: Ubuntu 22.04 LTS (Lambda Stack pre-installed)
4. Add SSH key
5. Launch

### Step 2: SSH In

```bash
ssh -i ~/.ssh/lambda_key.pem ubuntu@<LAMBDA_IP>
```

### Step 3: Verify GPUs and Interconnect

```bash
# Check all 8 GPUs visible
nvidia-smi

# Verify NVLink topology (should show NV18 between all GPU pairs)
nvidia-smi topo -m

# Expected output for 8x H100 NVLink:
# GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
#  X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
# NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
# ...
```

### Step 4: Install Dependencies

```bash
python3 -m venv env && source env/bin/activate
pip install vllm huggingface_hub

# Or for SGLang:
pip install sglang[all]
```

### Step 5: Download Model (~794 GB for BF16, ~397 GB for FP8)

```bash
# Fast parallel download
export HF_HUB_NUM_WORKERS=16
huggingface-cli download Qwen/Qwen3.5-397B-A17B

# Estimated time at 100 MB/s: ~2.2 hours (BF16) or ~1.1 hours (FP8)
```

**Disk space needed**: At least 1 TB free. Lambda instances typically come with NVMe storage — verify with `df -h`.

### Step 6: Launch Server

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.8 \
  --language-model-only \
  --enable-prefix-caching
```

Model loading takes ~5-10 minutes for 794 GB across 8 GPUs.

### Step 7: Verify

```bash
curl http://localhost:8000/v1/models
# Should return model name

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-397B-A17B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'
```

### Step 8: Port Forward to Local Machine

```bash
# From your local machine
ssh -i ~/.ssh/lambda_key.pem -L 8000:localhost:8000 ubuntu@<LAMBDA_IP>

# Now http://localhost:8000 on your machine hits the Lambda vLLM server
```

### Step 9: Monitor During Experiments

```bash
# In a separate SSH session
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet -i 0,1,2,3,4,5,6,7
```

---

## 9. Python Client for Monitor + Verifier

Same as before, but now hitting the 397B model:

```python
from openai import OpenAI, AsyncOpenAI
import asyncio

MODEL = "Qwen/Qwen3.5-397B-A17B"

# Sync client (simple)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def call_monitor(code: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MONITOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this code:\n```python\n{code}\n```"},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return parse_monitor_response(resp.choices[0].message.content)

def call_verifier(code: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Check consistency:\n```python\n{code}\n```"},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return parse_verifier_response(resp.choices[0].message.content)

# Async client (for 5000 calls)
async_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

async def run_both(code: str) -> tuple:
    """Run monitor and verifier in parallel on same code."""
    monitor_task = async_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MONITOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this code:\n```python\n{code}\n```"},
        ],
        temperature=0.3, max_tokens=512,
    )
    verifier_task = async_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Check consistency:\n```python\n{code}\n```"},
        ],
        temperature=0.3, max_tokens=512,
    )
    monitor_resp, verifier_resp = await asyncio.gather(monitor_task, verifier_task)
    return (
        parse_monitor_response(monitor_resp.choices[0].message.content),
        parse_verifier_response(verifier_resp.choices[0].message.content),
    )

async def batch_process(codes: list[str], batch_size: int = 16) -> list:
    """Process all codes through both monitor and verifier."""
    results = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        batch_results = await asyncio.gather(*[run_both(code) for code in batch])
        results.extend(batch_results)
        print(f"Processed {min(i+batch_size, len(codes))}/{len(codes)}")
    return results
```

---

## 10. Expected Performance for Our Experiment

### Throughput Estimates (8x H100, BF16)

| Metric | Value |
|---|---|
| Output tokens/sec (batch=1) | ~45 tok/s |
| Output tokens/sec (batch=16) | ~120-150 tok/s |
| Output tokens/sec (batch=32) | ~180+ tok/s |
| Time per monitor+verifier call (~512 tokens out) | ~3-10 sec |
| Time for 5,000 dual calls (batch=16) | ~2-3 hours |
| Time for 5,000 dual calls (batch=32) | ~1-2 hours |

### Cost Estimate

| Component | Hours | Rate | Cost |
|---|---|---|---|
| 8x H100 instance | ~6 hrs (3 download + 3 inference) | $23.92/hr | ~$144 |
| o4-mini API (untrusted coder) | ~3,000 calls | ~$0.003/call | ~$10 |
| Buffer/reruns | | | ~$30 |
| **Total** | | | **~$184** |

### vs. Managed API Alternative

| Component | Cost |
|---|---|
| Together.ai: 10,000 calls × ~1K tokens | ~$40 |
| o4-mini API | ~$10 |
| **Total** | **~$50** |

---

## 11. Common Pitfalls and How to Avoid Them

### Expert Load Imbalance

**Problem**: Some GPUs get more tokens routed to their experts, creating stragglers.
**Fix**: Enable dynamic load balancing:
```bash
--enable-eplb
```

### OOM During Serving

**Problem**: KV cache grows with concurrent requests and context length.
**Fix**: Reduce context or memory utilization:
```bash
--max-model-len 65536          # Reduce from 131K to 64K
--gpu-memory-utilization 0.75  # Reserve more headroom
--kv-cache-dtype fp8           # Halve KV cache size
```

### Slow Model Download

**Problem**: 794 GB takes hours at normal speeds.
**Fix**: Parallel download:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3.5-397B-A17B
```

### NCCL Timeout on Large Models

**Problem**: Model loading or first inference exceeds default timeout.
**Fix**:
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
export VLLM_RPC_TIMEOUT=300000  # 5 minutes
```

### Routing Instability

**Problem**: All-to-all communication consumes 40-77% of MoE layer time.
**Fix**: Use optimized backend:
```bash
--all2all-backend deepep_high_throughput  # For batch > 128
--all2all-backend deepep_low_latency      # For batch < 128
```

---

## 12. Decision Matrix: Which Config to Use

### For Our Hackathon Experiment

| Option | Config | Cost | Quality | Throughput | Recommendation |
|---|---|---|---|---|---|
| **A: Self-host BF16** | 8x H100, TP=8 | ~$184 | Best | ~150 tok/s | Best if budget allows |
| **B: Self-host FP8** | 8x H100, TP=8 | ~$144 | Near-best | ~180 tok/s | Good balance |
| **C: Managed API** | Together.ai | ~$50 | Best (their infra) | Rate-limited | Cheapest, simplest |
| **D: Self-host 27B** | 1x H100, BF16 | ~$36 | Good (72.4% SWE) | ~1500 tok/s | Original plan |

**My recommendation**: **Option C (Together.ai API) or Option D (27B self-hosted)** for the hackathon.

- The 397B model is much more capable but costs 4-5x more
- For a 3-day hackathon, API simplicity matters — no debugging vLLM OOM issues
- The research contribution is the **protocol design**, not the model size
- Run the 27B experiments first, then optionally validate key results on 397B via API

If you want to go big: **Option B (FP8 on 8x H100)** gives near-full quality at reasonable cost.
