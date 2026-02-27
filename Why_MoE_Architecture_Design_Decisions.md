# Why Mixture-of-Experts: Design Decisions from First Principles

Every design decision in MoE architecture traces back to a single tension: **how do you make a model know more without making every prediction cost more?** This document follows the chain of constraints, from the fundamental scaling problem to the specific engineering solutions used in models like DeepSeek-V3 and Qwen3.5-397B.

Each section starts with the physical or mathematical constraint that forces the decision, then explains the solution and why alternatives were rejected.

---

## Table of Contents

1. [The Scaling Wall: Why Dense Models Hit a Ceiling](#1-the-scaling-wall-why-dense-models-hit-a-ceiling)
2. [Conditional Computation: The Core Insight](#2-conditional-computation-the-core-insight)
3. [The Router: Why Gating Is the Hardest Problem](#3-the-router-why-gating-is-the-hardest-problem)
4. [Top-K Selection: How Many Experts Per Token?](#4-top-k-selection-how-many-experts-per-token)
5. [The Load Balancing Crisis: Why Experts Collapse](#5-the-load-balancing-crisis-why-experts-collapse)
6. [Auxiliary Losses: Forcing Balance Without Destroying Specialization](#6-auxiliary-losses-forcing-balance-without-destroying-specialization)
7. [Token Dropping and Expert Capacity: The Static Graph Problem](#7-token-dropping-and-expert-capacity-the-static-graph-problem)
8. [Training Instability: Why MoE Models Break](#8-training-instability-why-moe-models-break)
9. [The Memory Paradox: More Parameters, Same Compute, But More VRAM](#9-the-memory-paradox-more-parameters-same-compute-but-more-vram)
10. [Many Small Experts vs. Few Large Experts](#10-many-small-experts-vs-few-large-experts)
11. [Shared Experts: The Safety Net](#11-shared-experts-the-safety-net)
12. [DeepSeek-V3's Revolution: Auxiliary-Loss-Free Balancing](#12-deepseek-v3s-revolution-auxiliary-loss-free-balancing)
13. [Fine-Tuning MoE: Why It's Harder Than Dense Models](#13-fine-tuning-moe-why-its-harder-than-dense-models)
14. [Inference Challenges: The Deployment Tax](#14-inference-challenges-the-deployment-tax)
15. [The Historical Arc: How We Got Here](#15-the-historical-arc-how-we-got-here)
16. [Summary: Every Decision Traces to a Constraint](#16-summary-every-decision-traces-to-a-constraint)

---

## 1. The Scaling Wall: Why Dense Models Hit a Ceiling

### The constraint

In a dense transformer, **every parameter participates in every forward pass**. If a model has 70B parameters, processing a single token requires ~70 billion multiply-accumulate operations. Double the parameters, double the compute per token.

This creates two linked problems:

**Problem 1 — Training cost scales quadratically.** Scaling laws (Kaplan et al., 2020) showed that loss decreases as a power law in compute, but compute grows proportionally with parameters for dense models. To train a model twice as large, you need roughly 4× the budget: 2× for the larger model, 2× for the additional training data it needs.

**Problem 2 — Inference cost scales linearly with parameters.** Every user query against a 70B model costs 70B FLOPs per token. You cannot escape this. A 70B model is always ~3.5× slower per token than a 20B model.

### The fundamental question

Can we build a model that **knows as much as a 400B-parameter model** but **computes as fast as a 20B-parameter model**?

Dense architectures say no — knowledge capacity and compute cost are the same thing.

### Why this matters for our project

We need a capable monitor/verifier for the Verified Legibility protocol. More capable models are better at detecting backdoors. But more capable models are also slower and more expensive per query. If we could decouple capability from cost, we could afford a much stronger monitor.

> **Further reading:** [Mixture of Experts Explained (HuggingFace)](https://huggingface.co/blog/moe) — Section on scaling motivation. [MoE LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — Scaling laws context.

---

## 2. Conditional Computation: The Core Insight

### The solution

MoE introduces **conditional computation**: instead of activating all parameters for every token, activate only the relevant subset. Replace the monolithic feed-forward network (FFN) in each transformer layer with a collection of **N independent expert FFNs**, and use a learned **router** (gating network) to select which experts process each token.

The mathematical formulation:

```
y(x) = Σᵢ g_i(x) · E_i(x)
```

Where `g_i(x)` is the gating weight (how much to trust expert i for this token) and `E_i(x)` is expert i's output. Crucially, most `g_i` values are zero — only K experts are selected out of N total.

### Why this works

The key asymmetry:

| Property | Determined by... |
|---|---|
| **Model quality** (what it knows) | Total parameters across all experts |
| **Inference speed** (cost per token) | Active parameters per token (K experts) |
| **Memory requirement** | Total parameters (all experts must be loaded) |

A model with 400B total parameters but 20B active parameters per token has the knowledge capacity of a 400B model but the inference speed of a 20B model. Quality comes from total parameters; speed comes from active parameters. Dense models bundle these together. MoE separates them.

### Concrete evidence

| Model | Total Params | Active Params | Performance |
|---|---|---|---|
| Mixtral 8×7B | 47B | ~13B | Outperformed LLaMA-2 70B |
| DeepSeek-V2 | 236B | 21B | Far better than a 21B dense model |
| DeepSeek-V3 | 671B | 37B | Matched frontier closed-source models |
| Qwen3.5-397B | 397B | 17B | Frontier open-source performance |

GLaM matched GPT-3 quality with **1/3 the energy**. Switch Transformers achieved a **7× pretraining speedup** over T5-Base with equivalent quality.

> **Further reading:** [A Visual Guide to MoE (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — Excellent diagrams of the sparse activation concept. [HuggingFace MoE blog](https://huggingface.co/blog/moe) — Building blocks section.

---

## 3. The Router: Why Gating Is the Hardest Problem

### The constraint

The router must make a **discrete decision** (which K experts?) but must be **trainable by gradient descent** (which requires continuous, differentiable functions). Discrete selection is inherently non-differentiable.

### The solution: Softmax + Top-K

The standard approach uses a two-step trick:

**Step 1 — Compute continuous scores.** A linear layer maps the token's hidden state to N scores (one per expert):

```
scores = W_router · x      (linear projection, d_model → N)
```

**Step 2 — Apply softmax to get a probability distribution:**

```
probabilities = softmax(scores)
```

**Step 3 — Select the top K, zero out the rest.** This is the discrete step. It is non-differentiable, but the softmax probabilities that flow into the output weighting remain differentiable, allowing gradients to flow backward through the system.

### Why softmax?

Softmax ensures probabilities sum to 1, creating a valid distribution. This matters for two reasons:
1. **Gradient flow:** Backpropagation through softmax is well-understood and stable.
2. **Interpretable weights:** The output `y = Σ g_i · E_i(x)` is a proper weighted average when `Σ g_i = 1`.

### Why a linear layer (not something more complex)?

The router runs on **every single token**. It must be computationally trivial. A linear layer is the minimal transformation that can learn token-to-expert mappings. Anything more complex (multi-layer networks, attention mechanisms) would add overhead to every token's processing, partially defeating the efficiency purpose of MoE.

### Why this is the hardest problem

The router is "arguably the most important component of any MoE" because it determines the quality of expert utilization. A bad router means:
- Some experts are overloaded, others idle (wasted parameters)
- Tokens go to irrelevant experts (wrong computation applied)
- Training becomes unstable (routing collapse)

The router must simultaneously learn **which expert is best for each token** and maintain **balanced utilization across experts** — two objectives that directly conflict.

### The DeepSeek-V3 departure: Sigmoid gating

DeepSeek-V3 broke from convention by using **sigmoid** instead of softmax for computing individual expert scores:

```
s_{i,t} = σ(u_t^T · e_i)
```

The scores are then normalized only across the selected top-K experts. This allows each expert's score to be computed independently of all other experts, which is better suited to very large expert counts (256+) where softmax over all experts becomes expensive and creates strong coupling between expert scores.

> **Further reading:** [MoE Architecture Deep Dive (AWS Neuron)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html) — Router mechanics in detail. [MoE LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — Gating variants comparison.

---

## 4. Top-K Selection: How Many Experts Per Token?

### The constraint

Each additional active expert **multiplies compute cost linearly** but provides **diminishing quality returns**. The choice of K directly determines the compute-quality tradeoff.

### The options and their tradeoffs

| K | Compute Cost | Expert Combinations (from 8 experts) | Expert Combinations (from 256 experts) | Training Stability |
|---|---|---|---|---|
| 1 | Lowest | 8 | 256 | Fragile |
| 2 | 2× K=1 | 28 | 32,640 | Good |
| 4 | 4× K=1 | 70 | ~174 million | Best |
| 8 | 8× K=1 | 1 (from 8) | ~4.4 × 10¹³ (from 256) | Best |

### K=1 (Switch Transformer, 2022)

**Why try it?** Maximum sparsity. Halves the effective batch size per expert compared to K=2, reducing communication costs. Simpler routing logic.

**Why it's risky:** Extremely high variance in token routing. No redundancy — if the selected expert is poorly suited, there's no fallback. Routing collapse can be severe because the router gets less gradient signal (only one expert's contribution to differentiate against).

**When it works:** Large expert counts (hundreds+) where each expert is small, so the penalty for a suboptimal single choice is limited.

### K=2 (Shazeer 2017, GShard, Mixtral)

**Why it became the default:** Shazeer et al. conjectured that the router needs to see at least two experts' outputs to learn effective routing — it needs a comparison signal. K=2 provides redundancy without excessive compute. Two experts' contributions are weighted and integrated, giving the model a richer representation per token.

**The sweet spot for 8-expert models.** With 8 experts and K=2, each token has 28 possible expert pairs. This provides enough routing diversity for specialization while keeping compute at 25% of what a dense model of the same total size would require.

### K=4+ (DBRX, DeepSeek-V3)

**Why go higher?** With fine-grained experts (many small experts), each individual expert contributes less. You need more of them active to maintain quality. DBRX uses 16 experts with K=4, providing "65× more expert combinations" than 8-expert K=2.

**The modern pattern:** More experts + higher K = fine-grained specialization:
- DeepSeek-V3: 256 experts, K=8
- Qwen3: 128 experts, K=8
- Qwen3.5-397B: 512 experts, K=10+1 shared

### The historical oscillation

The field has swung back and forth:

```
Shazeer 2017 (K≥2) → Switch 2022 (K=1, simplified) → Mixtral 2023 (K=2)
→ DBRX/DeepSeek 2024 (K=4-8, fine-grained) → Qwen3.5 2026 (K=10+1)
```

The trend is clear: **more experts, higher K, finer granularity**. But the fundamental tradeoff — more active experts = more compute = slower inference — never goes away.

> **Further reading:** [HuggingFace MoE blog](https://huggingface.co/blog/moe) — Top-K comparisons. [Comprehensive Survey of MoE](https://arxiv.org/html/2503.07137v1) — Historical analysis of K choices.

---

## 5. The Load Balancing Crisis: Why Experts Collapse

### The constraint

The router is trained by gradient descent to minimize language modeling loss. Language modeling loss only rewards correct predictions — it has **no inherent pressure toward balanced expert utilization**. The router is free to achieve good predictions by using just 2-3 experts for everything.

### The collapse mechanism

This creates a self-reinforcing positive feedback loop:

```
1. Router slightly favors Expert A (due to random initialization)
2. Expert A receives more training tokens
3. Expert A improves faster (more data = better weights)
4. Improved Expert A produces better outputs
5. Router learns to favor Expert A even more
6. Repeat until 2-3 experts handle all tokens, others stagnate
```

This is called **routing collapse**. It effectively reduces an N-expert model to a 2-3 expert model, wasting the vast majority of parameters. The model pays the memory cost of N experts but gets the quality of 2-3.

### Why the gradient signal doesn't fix this

The language modeling loss gradient tells the router "Expert A gave a good answer for this token" or "Expert A gave a bad answer." It never says "you should use Expert B more because Expert B is underutilized." There is simply no signal in the standard training objective that cares about balance.

### Why this is catastrophic in distributed settings

When experts are distributed across GPUs (expert parallelism), unbalanced routing means:
- GPU hosting the popular expert is saturated (100% utilization, becomes the bottleneck)
- GPUs hosting unpopular experts are idle (0-10% utilization, wasted money)
- The slowest GPU determines throughput for the entire system
- You're paying for 8 GPUs but getting the throughput of 1

### What causes initial imbalance

Even without routing collapse, several factors create natural imbalance:
1. **Initialization randomness:** Some experts start with slightly better weights by chance
2. **Data distribution:** Natural language has inherent frequency biases (common tokens are far more frequent than rare ones)
3. **Temporal locality:** Consecutive tokens in a sequence tend to route to the same experts (found in Mixtral analysis)
4. **Context-independent routing:** Tokens route based on surface patterns (token ID, position) more than semantic meaning (OpenMoE finding)

> **Further reading:** [MoE Load Balancing Review (HuggingFace)](https://huggingface.co/blog/NormalUhr/moe-balance) — The expert imbalance problem and solutions. [A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — Routing collapse diagrams.

---

## 6. Auxiliary Losses: Forcing Balance Without Destroying Specialization

### The constraint

We need to add a gradient signal that pushes toward balanced expert utilization, but we must not overpower the language modeling signal that pushes toward good predictions. If the balance loss dominates, experts are forced into generalism and the whole point of MoE (specialization) is lost.

### The fundamental tension

```
Strong auxiliary loss → Balanced but generic experts → MoE adds little value
Weak auxiliary loss → Specialized but imbalanced → Routing collapse risk
No auxiliary loss → Best specialization but guaranteed collapse
```

Every auxiliary loss design is an attempt to thread this needle.

### Solution 1: Importance Loss (Shazeer et al., 2017)

Compute each expert's "importance" as the sum of router probabilities across all tokens in a batch, then minimize the coefficient of variation:

```
CV = σ(importance) / μ(importance)
```

**Limitation:** Experts can have equal importance but wildly unequal token loads. Expert A gets 1 token with 99% probability. Expert B gets 1000 tokens with 0.1% each. Both have similar importance scores but vastly different computational loads.

### Solution 2: Switch Transformer Load Balancing Loss (2022)

Combines two metrics to fix the importance loss blind spot:

```
L_aux = α · N · Σᵢ (fᵢ · Pᵢ)
```

Where:
- `fᵢ` = fraction of tokens actually dispatched to expert i (the real load)
- `Pᵢ` = mean router probability assigned to expert i (the intended load)
- `α` = loss coefficient (typically 10⁻² to 10⁻¹)

This product is minimized when both `f` and `P` are uniform across experts. It explicitly penalizes the case where one expert gets many tokens AND has high probability — precisely the collapse pattern.

### Solution 3: Router Z-Loss (ST-MoE, 2022)

A different kind of instability: router logits become very large during training, causing numerical overflow in the softmax exponential:

```
L_z = (1/B) · Σ_x [ log( Σᵢ exp(logit_i(x)) ) ]²
```

This penalizes large logits, keeping the softmax numerically stable. It improved training stability without degrading quality.

### The total training objective

Modern MoE training optimizes:

```
L_total = L_language_modeling + λ_balance · L_load_balance + λ_z · L_z
```

Tuning `λ_balance` is critical. Too high destroys specialization. Too low allows collapse. There is no universally correct value — it depends on expert count, K, dataset, and model scale.

### The problem with all auxiliary losses

Any gradient-based auxiliary loss creates a **conflicting optimization pressure**. The language modeling loss says "send this token to Expert A because Expert A gives the best answer." The auxiliary loss says "send this token to Expert B because Expert B is underutilized." These gradients literally point in opposite directions, and the model must compromise between quality and balance.

This fundamental limitation motivated DeepSeek-V3's completely different approach (Section 12).

> **Further reading:** [MoE Load Balancing Review (HuggingFace)](https://huggingface.co/blog/NormalUhr/moe-balance) — Mathematical details of all balance losses. [HuggingFace MoE blog](https://huggingface.co/blog/moe) — Practical impact of auxiliary loss tuning.

---

## 7. Token Dropping and Expert Capacity: The Static Graph Problem

### The constraint

GPU hardware requires **static computation graphs** with **fixed tensor shapes** at compile time. But MoE routing is **dynamic** — the number of tokens sent to each expert varies per batch, per sequence, per layer. These two requirements directly conflict.

### The solution: Expert capacity

Set a hard ceiling on how many tokens any single expert can process:

```
Expert_Capacity = (tokens_per_batch / num_experts) × capacity_factor
```

With capacity_factor = 1.25 (Switch Transformer's sweet spot):
- 1024 tokens per batch, 8 experts
- Capacity = (1024 / 8) × 1.25 = 160 tokens per expert maximum

This gives the GPU a fixed tensor size to compile against, regardless of routing dynamics.

### What happens to excess tokens: Token dropping

When more tokens are routed to an expert than its capacity allows, the excess tokens are **dropped** — they bypass the expert entirely and flow to the next layer only via the residual connection. They receive zero computation from that MoE layer.

### Capacity factor tradeoffs

| Capacity Factor | Behavior |
|---|---|
| 1.0 | Assumes perfect balance (unrealistic). Heavy token dropping. |
| 1.25 | Switch Transformer's sweet spot. 25% buffer for imbalance. |
| 1.5-2.0 | Used by GLaM. Few drops but high memory overhead from padding. |
| ∞ (no limit) | No dropping but unpredictable memory usage, cannot compile static graphs. |

### Why token dropping is dangerous

- **Drops accumulate across layers:** A token dropped in layer N enters layer N+1 with stale features, causing cascading quality degradation.
- **Drops are systematic, not random:** Later tokens in a sequence are dropped more frequently because earlier tokens exhaust expert capacity first (OpenMoE finding). This creates blind spots for long-context tasks.
- **Drop patterns lock in during pretraining** and cannot be corrected by fine-tuning.

### The regularization counterargument

Surprisingly, Switch Transformer authors found that ~11% token dropping during training did not significantly hurt quality. They hypothesized that token dropping acts as **implicit regularization** similar to dropout — randomly degrading some tokens' representations forces the model to be more robust.

ST-MoE authors confirmed: auxiliary loss can be turned off during fine-tuning with minimal impact, partly because the resulting token dropping provides beneficial regularization.

### Modern solutions: Dropless MoE

Several approaches eliminate token dropping entirely:

1. **MegaBlocks (Nov 2022):** Block-sparse GPU kernels that handle variable expert sizes natively. No fixed capacity, no dropping.
2. **DeepSpeed-MoE:** Dynamically redistributes overflow tokens to less-loaded experts instead of dropping them.
3. **DeepSeek-V3:** Uses no token dropping in either training or inference, enabled by bias-based load balancing that keeps routing balanced enough to avoid overflow.

> **Further reading:** [HuggingFace MoE blog](https://huggingface.co/blog/moe) — Expert capacity and token dropping experiments. [Comprehensive Survey of MoE](https://arxiv.org/html/2503.07137v1) — Dropless approaches.

---

## 8. Training Instability: Why MoE Models Break

### The constraint

MoE introduces multiple interacting failure modes that don't exist in dense models. Each stems from a different aspect of the sparse architecture.

### Failure mode 1: Routing collapse (primary)

The self-reinforcing loop described in Section 5. Popular experts improve faster, get selected more, and dominate. This is the most fundamental instability in MoE training.

**Solution stack:** Auxiliary load balancing loss + noisy gating + capacity factors + dynamic bias terms (DeepSeek-V3).

### Failure mode 2: Numerical instability in the router

Router logits grow during training. When passed through softmax's exponential function:

```
exp(very_large_number) → numerical overflow
```

Even in float32, large logits cause precision loss. In bfloat16, the problem is catastrophic.

**Solution:** Router z-loss to constrain logit magnitudes, plus selective precision — router computations in float32, expert FFNs in bfloat16 (Switch Transformer approach).

### Failure mode 3: Communication bottleneck feedback

In distributed training with expert parallelism, unbalanced routing causes some GPUs to be overloaded while others idle. The overloaded GPU becomes the bottleneck, slowing the entire step. Slow steps mean the model trains on fewer tokens per hour, which can compound with other instabilities.

**Solution:** Device-level load balancing losses (DeepSeek-V2) that specifically target cross-GPU balance, not just cross-expert balance.

### Failure mode 4: Domain shift during post-training

Routing patterns established during pretraining on web text become "sporadic" when the data distribution shifts to instruction-following data. Token dropping increases and quality degrades. Critically, routing patterns are **locked in early during pretraining** and cannot be fundamentally altered by fine-tuning.

**Solution:** Mix instruction data into pretraining (OpenMoE recommendation). This exposes the router to diverse data patterns from the start.

### Failure mode 5: Loss spikes

MoE models exhibit sudden, large increases in training loss that dense models don't. These spikes can be irrecoverable, requiring training rollback.

**DeepSeek-V3's achievement:** "Did not experience any irrecoverable loss spikes or perform rollbacks" during training of a 671B model. This was enabled by their combination of auxiliary-loss-free balancing, FP8 mixed precision with careful quantization, and the DualPipe algorithm for compute-communication overlap.

> **Further reading:** [MoE LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — Training instability analysis. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — Zero loss spikes achievement.

---

## 9. The Memory Paradox: More Parameters, Same Compute, But More VRAM

### The constraint

All experts must reside in GPU memory at all times, because **routing is determined at runtime**. You cannot predict in advance which experts a future token will need. The computation graph must be ready for any routing decision.

### The paradox in numbers

| Model | Total Params | Active Params | FLOPs/token | Memory (BF16) |
|---|---|---|---|---|
| Dense 47B | 47B | 47B | ~47B | ~94 GB |
| Mixtral 8×7B | 47B | ~13B | ~13B | ~94 GB |
| Dense 37B | 37B | 37B | ~37B | ~74 GB |
| DeepSeek-V3 | 671B | 37B | ~37B | ~1.3 TB |

Mixtral is ~3.5× faster than a dense model of the same memory footprint. But DeepSeek-V3 computes at the speed of a 37B model while requiring ~18× the memory of a 37B dense model.

### Why this is the fundamental deployment challenge

The memory paradox means MoE models need more GPUs than their compute cost suggests. A 671B model that only does 37B FLOPs per token still needs enough GPU VRAM to hold 671B parameters. You're paying for **memory capacity** (buying GPUs) but only utilizing **compute capacity** (FLOPs per GPU) at a fraction of peak.

This is the core reason MoE models require multi-GPU deployment even when their per-token compute doesn't demand it. The GPU memory holds mostly idle experts at any given moment.

### Mitigation strategies

1. **Quantization:** Compress all weights (including idle experts) to use less memory per parameter. FP8 halves memory vs BF16. INT4 quarters it.
2. **Expert offloading:** Keep inactive experts in CPU RAM or SSD, load on demand. Works for batch inference but adds latency for real-time serving.
3. **Expert merging:** Post-training, merge similar experts to reduce total count. Recovers 30-40% of the sparsity benefit.
4. **Distillation:** Compress the entire MoE into a smaller dense model. Faster serving but loses much of the MoE quality advantage.

> **Further reading:** [HuggingFace MoE blog](https://huggingface.co/blog/moe) — Memory analysis and serving strategies. [Dynamic Expert Quantization for MoE (DynaExq)](https://arxiv.org/html/2511.15015v2) — MoE-specific quantization.

---

## 10. Many Small Experts vs. Few Large Experts

### The constraint

The number and size of experts determines the **granularity of specialization**. Few large experts (8 × 7B) give coarse specialization. Many small experts (256 × 0.3B) give fine-grained specialization. Both have the same total parameter count but very different routing dynamics.

### The combinatorial argument

With K experts selected from N total, the number of unique expert combinations is C(N, K):

| Configuration | Experts (N) | Active (K) | Possible Combinations |
|---|---|---|---|
| Mixtral | 8 | 2 | 28 |
| DBRX | 16 | 4 | 1,820 |
| DeepSeek-V3 | 256 | 8 | ~4.4 × 10¹³ |
| Qwen3.5-397B | 512 | 10 | ~2.7 × 10²⁰ |

More expert combinations means the model can represent more distinct "expert assemblies" — different combinations of capabilities that are brought together for different token types. Each combination is essentially a different sub-network tailored to that specific input.

### The trend: Many small experts won

The historical progression is clear:

```
Shazeer 2017:  2048 experts (many small) — original vision
GShard 2020:   hundreds of experts
Switch 2022:   2048 experts, top-1
Mixtral 2023:  8 experts (few large) — simplified
DBRX 2024:     16 experts (medium)
DeepSeek-V3:   256 experts (many small) — back to original vision
Qwen3.5:       512 experts (many small) — current state of the art
```

Mixtral's 8-expert design was a temporary simplification that proved the open-source MoE concept. The field has since returned to Shazeer's original insight that many small experts provide richer combinatorial capacity.

### DeepSeekMoE's formalization

DeepSeekMoE (Jan 2024) formalized the principle: instead of N experts of size d_ffn, use mN experts of size d_ffn/m, activating mK of them. Their 16B model matched LLaMA-2 7B at 40% of the compute, validating that fine-grained experts are more parameter-efficient.

> **Further reading:** [Comprehensive Survey of MoE](https://arxiv.org/html/2503.07137v1) — Expert granularity analysis. [MoE LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — DBRX and DeepSeek architecture comparison.

---

## 11. Shared Experts: The Safety Net

### The constraint

If all experts are routed (selected by the router), every token depends entirely on the router making good decisions. If the router makes a bad choice, the token receives irrelevant computation with no fallback.

### The solution: Dedicated shared experts

Some models designate one or more experts as **shared** — they process every token unconditionally, regardless of routing:

```
output = residual + Σ shared_experts(x) + Σ g_i · routed_expert_i(x)
```

### Why shared experts help

1. **Quality floor:** Even if the router fails completely for a token, the shared expert provides a baseline computation. No token is left with zero useful processing.
2. **Common knowledge capture:** Shared experts learn patterns that are universally useful (grammar, common phrases, structural syntax). This frees routed experts from redundantly learning these patterns, allowing them to specialize more aggressively.
3. **Gradient stability:** Shared experts receive gradients from every token, making their training dynamics smooth and stable. They act as an anchor that stabilizes the overall optimization.

### Who uses shared experts

| Model | Shared Experts | Routed Experts | Total Active |
|---|---|---|---|
| Mixtral 8×7B | 0 | 2 of 8 | 2 |
| DeepSeek-V3 | 1 | 8 of 256 | 9 |
| Qwen3 | 0 | 8 of 128 | 8 |
| Qwen3.5-397B | 1 | 10 of 512 | 11 |

The DeepSeek and Qwen3.5 models both use shared experts. Mixtral and Qwen3 don't. The trend toward shared experts correlates with the trend toward many small routed experts — when each routed expert is small and specialized, the shared expert becomes more important as a universal backbone.

> **Further reading:** [MoE Architecture Deep Dive (AWS Neuron)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html) — Shared vs routed expert architecture.

---

## 12. DeepSeek-V3's Revolution: Auxiliary-Loss-Free Balancing

### The constraint

All auxiliary losses create a fundamental conflict: the language modeling gradient says "route to the best expert" while the balancing gradient says "route to the least-loaded expert." These gradients literally oppose each other, and any fixed loss weight `λ` is a compromise that sacrifices either quality or balance.

### DeepSeek-V3's insight: Separate routing from weighting

Instead of adding a loss term, DeepSeek-V3 adds a **dynamic bias term** to each expert that affects routing decisions but NOT output weighting:

```
Routing decision:  select expert if (s_{i,t} + b_i) is in TopK
Output weighting:  g_{i,t} = s_{i,t} / Σ_selected s_{j,t}   (bias NOT included)
```

The bias `b_i` steers which experts get selected (load balancing) but is excluded from the output weights (preserving quality). This cleanly separates the two concerns that auxiliary losses conflate.

### How the bias adapts

The bias is adjusted dynamically during training, not by gradient descent:

```
After each training step, for each expert i:
    if expert_i received too many tokens:  b_i ← b_i - γ
    if expert_i received too few tokens:   b_i ← b_i + γ
```

Where `γ` (gamma) is the bias update speed — the only critical hyperparameter. It starts at 0.001 and is reduced to 0.0 late in training.

### Why this is better than auxiliary losses

| Aspect | Auxiliary Loss | Dynamic Bias |
|---|---|---|
| Optimization conflict | Yes — two gradients oppose | No — bias doesn't affect the loss |
| Specialization preservation | Compromised by balance pressure | Fully preserved |
| Adaptability | Fixed weight throughout training | Adapts per expert per batch |
| Hyperparameter sensitivity | Very sensitive to λ | Only γ to tune |
| Implementation | Changes the loss function | Changes the routing logic |

### The safety net: Minimal complementary loss

DeepSeek-V3 still uses a tiny auxiliary loss as a safety net against extreme within-sequence imbalance, but with α set very small — it does not dominate the training objective.

### Results

DeepSeek-V3 trained a 671B model on 14.8T tokens for ~$5.6M with zero irrecoverable loss spikes and zero training rollbacks. This stability is unprecedented for a model of this scale and is directly attributed to the auxiliary-loss-free approach.

> **Further reading:** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — Full details of the bias-based approach. [MoE Load Balancing Review (HuggingFace)](https://huggingface.co/blog/NormalUhr/moe-balance) — Comparison with prior approaches.

---

## 13. Fine-Tuning MoE: Why It's Harder Than Dense Models

### The constraint

MoE models have far more total parameters than their effective compute suggests. When fine-tuning on small datasets, this massive parameter count creates severe **overfitting**: the model has the capacity to memorize the entire fine-tuning dataset.

### The asymmetry

| Task Type | Dense Model | MoE Model |
|---|---|---|
| Knowledge-heavy (TriviaQA) | Good | **Much better** |
| Reasoning-heavy (SuperGLUE) | Good | **Worse** |
| Small datasets | Gradual overfitting | **Sharp overfitting** |
| Large datasets | Good | **Better** |

MoE models excel at tasks that benefit from their larger knowledge capacity (more parameters = more stored knowledge). They struggle on tasks that require reasoning (where the active parameter count determines quality, not total parameters) and small datasets (where the total parameter count enables memorization).

### Routing pattern rigidity

The most fundamental fine-tuning challenge: **routing patterns are established during pretraining and resist change during fine-tuning.** If the router learned to handle web text during pretraining, it may route instruction-following tokens suboptimally. And there's no way to "retrain" the router without a pretraining-scale intervention.

### What works

1. **Higher dropout in expert layers:** 0.2-0.3 for expert FFNs vs 0.1 for dense layers. More parameters need more regularization.
2. **Freeze non-expert weights:** Only update MoE layers during fine-tuning. ~80% of parameters live in MoE layers (ST-MoE), so this barely reduces quality while dramatically reducing memory and instability.
3. **Turn off auxiliary loss during fine-tuning:** The resulting ~11% token dropping provides beneficial regularization.
4. **Multi-task instruction tuning (the breakthrough):** MoE models benefit more from instruction tuning than dense models. Flan-MoE dramatically outperforms base MoE on downstream tasks. More diverse fine-tuning tasks = better routing diversity.

> **Further reading:** [HuggingFace MoE blog](https://huggingface.co/blog/moe) — Fine-tuning challenges section. [DeepSpeed Mixture of Experts Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/) — Practical fine-tuning configuration.

---

## 14. Inference Challenges: The Deployment Tax

### The constraint

MoE's training efficiency (fewer FLOPs per token) doesn't fully translate to inference efficiency because of three deployment-specific problems.

### Problem 1: Memory bandwidth is the bottleneck, not compute

During autoregressive generation, each token requires loading the weights of K expert FFNs from GPU HBM to compute cores. The latency is dominated by **memory bandwidth** (how fast you can read weights), not **compute throughput** (how fast you can multiply). Since all experts must be in memory but only K are read per token, most of the memory holds idle weights that still consume precious VRAM capacity.

### Problem 2: All-to-All communication for distributed serving

When experts are distributed across GPUs (expert parallelism), every token must be physically shipped to the GPU holding its selected expert, processed, and shipped back. This All-to-All communication pattern can account for **>50% of total inference time** without optimization (NVIDIA finding).

The communication is also **irregular** — different tokens go to different GPUs based on routing decisions, creating unpredictable traffic patterns that are hard to optimize.

### Problem 3: Load imbalance at inference

During training, auxiliary losses keep routing balanced. At inference, there are no losses — the router just runs. Token distribution can become highly skewed, with popular experts creating hotspots that bottleneck the entire system.

### Solutions

| Strategy | Tradeoff |
|---|---|
| **Quantization** (FP8, INT4) | Reduces memory per parameter, fits more experts per GPU. Quality degradation. |
| **Expert offloading** to CPU/SSD | Reduces VRAM needs, adds latency per expert load. |
| **Expert merging** | Fewer experts to serve, but loses specialization. |
| **Distillation** to dense model | Simplest deployment, but loses 60-70% of MoE quality advantage. |
| **Expert parallelism** | Distributes memory cost across GPUs, but adds communication overhead. |
| **Hybrid-EP** (NVIDIA) | Optimizes All-to-All with hierarchical NVLink/InfiniBand communication. |

> **Further reading:** [Optimizing Communication for MoE Training (NVIDIA)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel) — Hybrid-EP architecture. [The vLLM MoE Playbook (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) — Practical MoE serving guide.

---

## 15. The Historical Arc: How We Got Here

The evolution of MoE is a story of the same fundamental ideas being rediscovered and refined as hardware caught up with theory.

| Year | Milestone | Key Decision | Why |
|---|---|---|---|
| **1991** | Jacobs et al. — Adaptive Mixtures of Local Experts | Multiple experts + gating network | Partition input space for specialization |
| **1994** | Jordan & Jacobs — Hierarchical MoE | Tree-structured gating | Recursive problem decomposition |
| **2017** | Shazeer et al. — Sparsely-Gated MoE | Noisy top-K gating, 137B LSTM | Scale language models 1000× with sublinear compute |
| **2020** | GShard | Top-2 + expert capacity + expert parallelism | Scale to 600B+ across TPU pods |
| **2021** | Switch Transformer | Top-1 routing, 1.6T params, 2048 experts | Simplify routing, maximize sparsity |
| **2022** | ST-MoE | Router z-loss for numerical stability | Fix training instability at scale |
| **2023** | Mixtral 8×7B | 8 experts, top-2, open-source | Prove MoE works for practical deployment |
| **2024** | DeepSeekMoE | Fine-grained experts + shared experts | More combinatorial flexibility |
| **2024** | DeepSeek-V3 | Auxiliary-loss-free balancing, sigmoid gating, FP8 training | Eliminate the quality-balance tradeoff |
| **2025** | Qwen3 | 128 experts, top-8 | Continue fine-grained trend |
| **2026** | Qwen3.5-397B | 512 experts, 10+1 active, hybrid architecture | Push granularity further with linear attention |

### The pattern

Each generation solves the previous generation's biggest problem:
- **2017:** "Can we make MoE work at all?" → Yes, with noisy gating.
- **2020-2022:** "Can we scale it?" → Yes, with expert capacity and z-loss.
- **2023:** "Can we deploy it practically?" → Yes, with smaller expert counts and open-source weights.
- **2024:** "Can we eliminate the quality-balance tradeoff?" → Yes, with bias-based balancing.
- **2025-2026:** "Can we push granularity to the limit?" → 512 experts with hybrid attention architectures.

> **Further reading:** [Mixture of Experts (Wikipedia)](https://en.wikipedia.org/wiki/Mixture_of_experts) — Historical foundations. [MoE LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — Full historical timeline.

---

## 16. Summary: Every Decision Traces to a Constraint

| Design Decision | Physical/Mathematical Constraint | Solution |
|---|---|---|
| Use MoE at all | Dense models couple knowledge capacity to compute cost | Conditional computation: activate subsets |
| Softmax + Top-K router | Need differentiable training but discrete expert selection | Continuous softmax scores, discrete top-K selection |
| Linear router layer | Router runs on every token, must be cheap | Minimal overhead projection |
| K=2 (or K=8-10 for fine-grained) | More active experts = more compute but more quality | Tradeoff point depends on expert granularity |
| Auxiliary load balancing loss | Router has no inherent balance incentive | Extra loss term penalizing imbalance |
| Router z-loss | Large logits cause softmax numerical overflow | Penalize logit magnitudes |
| Expert capacity factor | GPUs need static tensor shapes but routing is dynamic | Hard ceiling on tokens per expert |
| Token dropping | Capacity exceeded → tokens must go somewhere | Bypass via residual connection |
| Selective precision (FP32 router) | Exponentials in gating need precision | Router in FP32, experts in BF16/FP8 |
| Many small experts (256-512) | Combinatorial flexibility scales as C(N,K) | Fine-grained experts enable richer routing |
| Shared experts | Bad routing → token gets no useful computation | Unconditional backbone processing |
| Dynamic bias (DeepSeek-V3) | Auxiliary loss creates optimization conflict | Separate routing adjustment from output weighting |
| All experts in VRAM | Cannot predict which experts future tokens need | Pay memory cost for all, compute cost for K |
| Expert parallelism | One GPU can't hold all experts | Distribute experts, All-to-All communication |
| Hybrid-EP (NVIDIA) | NVLink ≠ InfiniBand bandwidth | Hierarchical communication matching hardware topology |

**The through-line:** Every MoE design decision is a response to the tension between wanting more model capacity (quality) and needing bounded compute cost (speed) and bounded memory (hardware reality). Dense models have none of these tensions because they don't try to separate capacity from cost. MoE models accept the architectural complexity in exchange for the fundamental advantage of decoupling what the model knows from what each prediction costs.

---

## References

All links below are from the [Learning Resources](./Learning_Resources.md) document, Section 3 (Mixture-of-Experts Architecture):

### Conceptual Understanding
- [Mixture of Experts Explained (HuggingFace)](https://huggingface.co/blog/moe)
- [A Visual Guide to Mixture of Experts (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [Mixture-of-Experts LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms)
- [What is Mixture of Experts? (IBM)](https://www.ibm.com/think/topics/mixture-of-experts)
- [What is MoE? How It Works, Use Cases (DataCamp)](https://www.datacamp.com/blog/mixture-of-experts-moe)

### Technical Deep Dives
- [Comprehensive Survey of MoE: Algorithms, Theory, Applications](https://arxiv.org/html/2503.07137v1)
- [MoE Architecture Deep Dive (AWS Neuron)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [MoE Load Balancing Review (HuggingFace Blog)](https://huggingface.co/blog/NormalUhr/moe-balance)
- [Optimizing Communication for MoE Training (NVIDIA)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel)

### Implementation
- [DeepSpeed Mixture of Experts Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
- [Mixture of Experts (Wikipedia)](https://en.wikipedia.org/wiki/Mixture_of_experts)

### MoE-Specific Inference (from Section 2)
- [The vLLM MoE Playbook (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [Dynamic Expert Quantization for MoE (DynaExq)](https://arxiv.org/html/2511.15015v2)
