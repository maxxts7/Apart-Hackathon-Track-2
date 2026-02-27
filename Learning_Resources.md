# Learning Resources

A curated reading list for the Verified Legibility project, organized by topic with a suggested learning order.

---

## Suggested Learning Order

1. HuggingFace MoE blog + Visual Guide to MoE (understand the architecture)
2. BentoML parallelism page + Tensor Parallelism in Three Levels (understand why we split models)
3. Redwood reading list + Greenblatt 2024 paper (understand AI control)
4. Legibility paper arXiv:2602.10153 (understand what we're extending)
5. vLLM practical guide + vLLM MoE playbook (understand deployment)

---

## 1. AI Control & Safety

### Start Here (Essentials)

- [Redwood Research AI Control Reading List](https://www.redwoodresearch.org/reading-list) — The definitive reading list, organized from quick intro to deep understanding
- [An Overview of Areas of Control Work](https://blog.redwoodresearch.org/p/an-overview-of-areas-of-control-work) — Ryan Greenblatt's map of the field
- [AXRP Podcast: AI Control with Buck Shlegeris & Ryan Greenblatt](https://axrp.net/episode/2024/04/11/episode-27-ai-control-buck-shlegeris-and-ryan-greenblatt.html) — Great audio introduction to the core ideas

### The Key Papers (Read in This Order)

1. [AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942) — Greenblatt et al. (2024). The foundational paper that established the control agenda.
2. [Ctrl-Z: Controlling AI Agents via Resampling](https://arxiv.org/abs/2504.10374) — Bhatt et al. (2025). Extends control to multi-step agent settings.
3. [Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols](https://arxiv.org/abs/2510.09462) — Terekhov et al. (2025). Shows how prompt injection and adaptive attacks break monitors.
4. [Async Control: Stress-testing Asynchronous Control Measures](https://arxiv.org/abs/2512.13526) — Stickland et al. (2025). Multi-barrier architecture: two barriers take safety from 0% to 91%.
5. [Basic Legibility Protocols Improve Trusted Monitoring](https://arxiv.org/abs/2602.10153) — Sreevatsa et al. (2026). The paper we are directly extending.
6. [Evaluating Control Protocols for Untrusted AI Agents](https://arxiv.org/abs/2511.02997) — Kutasov et al. (2025). Systematic evaluation across diverse agentic environments.

### Infrastructure

- [ControlArena Documentation](https://control-arena.aisi.org.uk/) — The evaluation framework we are building on
- [ControlArena GitHub](https://github.com/UKGovernmentBEIS/control-arena) — Source code and examples
- [ControlArena Tutorial](https://control-arena.aisi.org.uk/tutorial.html) — Hands-on getting started guide
- [ControlArena Micro-Protocols](https://control-arena.aisi.org.uk/micro_protocols.html) — How protocols are implemented
- [ControlArena Monitors](https://control-arena.aisi.org.uk/monitors.html) — How monitors work
- [ControlArena Policies](https://control-arena.aisi.org.uk/policies.html) — How attack/honest policies work
- [ControlArena Evaluation](https://control-arena.aisi.org.uk/evaluation.html) — Running evaluations
- [ControlArena Analysis](https://control-arena.aisi.org.uk/analysis.html) — Safety/usefulness metrics and Pareto frontiers

### Broader Context

- [AI Control tag on LessWrong](https://www.lesswrong.com/tag/ai-control) — All control research discussions in one place
- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026/) — Global policy landscape
- [Detecting and Reducing Scheming in AI Models](https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/) — OpenAI's findings on deceptive model behavior
- [Research Areas in AI Control — Alignment Forum](https://www.alignmentforum.org/posts/rGcg4XDPDzBFuqNJz/research-areas-in-ai-control-the-alignment-project-by-uk) — UK AISI's research priorities
- [Legibility Protocols Discussion on LessWrong](https://www.greaterwrong.com/posts/veqcY4Q6vadE2gv5j/basic-legibility-protocols-improve-trusted-monitoring-1) — Community discussion of the paper we extend

---

## 2. Multi-GPU & Distributed Inference

### Parallelism Fundamentals (Start Here)

- [Data, Tensor, Pipeline, Expert and Hybrid Parallelisms](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism) — Best single page covering all parallelism strategies with clear diagrams
- [Tensor Parallelism in Three Levels of Difficulty](https://www.determined.ai/blog/tp) — Builds from intuition to math to implementation
- [What is Inference Parallelism and How it Works](https://www.infracloud.io/blogs/inference-parallelism/) — Beginner-friendly overview
- [Tensor Parallel LLM Inferencing (Thomson Reuters)](https://medium.com/tr-labs-ml-engineering-blog/tensor-parallel-llm-inferencing-09138daf0ba7) — Practical walkthrough with code
- [Analyzing Tensor Parallelism Configurations (ROCm)](https://rocm.blogs.amd.com/artificial-intelligence/tensor-parallelism/README.html) — Impact of TP size on performance

### vLLM

- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) — Source code
- [vLLM Parallelism & Scaling Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — Official reference for TP, PP, EP
- [Serving LLMs with vLLM: A Practical Guide (Nebius)](https://nebius.com/blog/posts/serving-llms-with-vllm-practical-guide) — From transformers basics to production deployment
- [How to Choose the Right GPU for vLLM (DigitalOcean)](https://www.digitalocean.com/community/conceptual-articles/vllm-gpu-sizing-configuration-guide) — GPU sizing and memory calculations
- [vLLM Distributed Inference Optimization Guide (DatabaseMart)](https://www.databasemart.com/blog/vllm-distributed-inference-optimization-guide) — Common pitfalls and how to avoid them
- [Mastering vLLM Multi-GPU for Faster Inference (DejaFlow)](https://www.dejaflow.com/blog/2025/03/02/mastering-vllm-multi-gpu-for-faster-ai-inference/) — Practical multi-GPU walkthrough
- [vLLM on Google Cloud Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm) — Cloud deployment tutorial

### MoE-Specific Inference

- [The vLLM MoE Playbook (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) — Complete guide to TP, DP, PP, EP for MoE models
- [vLLM Expert Parallel Deployment Guide](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) — Official EP documentation
- [Scaling DeepSeek with Wide-EP (Red Hat)](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep) — Real-world MoE scaling case study
- [vLLM Large Scale Serving Blog](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) — DeepSeek-V3 at production scale
- [Deploying DeepSeek with PD Disaggregation (LMSYS)](https://lmsys.org/blog/2025-05-05-large-scale-ep/) — Advanced prefill-decode separation

### Multi-Node

- [Multi-node & Multi-GPU Inference with vLLM (MeluXina)](https://docs.lxp.lu/howto/llama3-vllm/) — Step-by-step multi-node tutorial
- [vLLM Distributed Serving Docs](https://docs.vllm.ai/en/v0.6.0/serving/distributed_serving.html) — Ray cluster setup for multi-node
- [NVIDIA Megatron Parallelisms Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) — NVIDIA's reference on all parallelism types

---

## 3. Mixture-of-Experts Architecture

### Conceptual Understanding

- [Mixture of Experts Explained (HuggingFace)](https://huggingface.co/blog/moe) — The best starting point. Covers building blocks, training, and serving tradeoffs.
- [A Visual Guide to Mixture of Experts (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — Excellent diagrams and intuitive explanations
- [Mixture-of-Experts LLMs (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/moe-llms) — Deep technical dive through the timeline of MoE in LLMs
- [What is Mixture of Experts? (IBM)](https://www.ibm.com/think/topics/mixture-of-experts) — Quick conceptual overview
- [What is MoE? How It Works, Use Cases (DataCamp)](https://www.datacamp.com/blog/mixture-of-experts-moe) — Accessible introduction with use cases

### Technical Deep Dives

- [Comprehensive Survey of MoE: Algorithms, Theory, Applications](https://arxiv.org/html/2503.07137v1) — Academic survey covering the full landscape
- [MoE Architecture Deep Dive (AWS Neuron)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html) — Router mechanics, expert MLPs, shared experts
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — How the biggest open MoE model was designed and trained
- [MoE Load Balancing Review (HuggingFace Blog)](https://huggingface.co/blog/NormalUhr/moe-balance) — The expert imbalance problem and solutions
- [Optimizing Communication for MoE Training (NVIDIA)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel) — All-to-All communication overhead deep dive

### Implementation

- [DeepSpeed Mixture of Experts Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/) — Microsoft's MoE training framework
- [Mixture of Experts (Wikipedia)](https://en.wikipedia.org/wiki/Mixture_of_experts) — Historical context and mathematical foundations

---

## 4. Qwen3.5 (The Model We're Using)

- [Qwen3.5-397B-A17B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) — Architecture details, benchmarks, usage
- [Qwen3.5-27B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-27B) — The dense model variant
- [Qwen3.5 Official Blog](https://qwen.ai/blog?id=qwen3.5) — Design decisions and benchmark results
- [Qwen3.5 GitHub Repository](https://github.com/QwenLM/Qwen3.5) — Code and documentation
- [Qwen3.5 vLLM Deployment Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) — Official vLLM recipes
- [Qwen3.5 SGLang Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/sglang.html) — Alternative serving framework
- [Qwen vLLM Deployment Docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) — Qwen team's own deployment guide
- [Qwen3.5 vs Claude Opus 4.6 Comparison](https://help.apiyi.com/en/minimax-m2-5-vs-claude-opus-4-6-coding-comparison-en.html) — Benchmark analysis

---

## 5. GPU Hardware & Cloud Infrastructure

### Hardware

- [NVIDIA Hopper Architecture In Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — H100, NVLink 4.0, NVSwitch explained
- [H100 vs A100 Comparison (Northflank)](https://northflank.com/blog/h100-vs-a100) — Performance and bandwidth differences
- [H100 Price Guide 2026 (JarvisLabs)](https://docs.jarvislabs.ai/blog/h100-price) — Cloud pricing comparison
- [H100 Rental Prices Compared (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison) — 15+ providers compared

### Lambda Labs

- [Lambda Labs Instances](https://lambda.ai/instances) — Available GPU configurations
- [Lambda Labs Pricing](https://lambda.ai/pricing) — Current rates
- [Lambda Cloud Getting Started](https://docs.lambda.ai/public-cloud/on-demand/getting-started/) — SSH setup, environment
- [Lambda Multi-Node Distributed Training Guide](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide) — Multi-node setup

### Alternatives

- [RunPod Pricing](https://docs.runpod.io/pods/pricing) — Per-second billing alternative
- [Together.ai Qwen3.5 Inference](https://www.together.ai/models/qwen3-5-397b-a17b) — Managed API option
- [OpenRouter Qwen3.5](https://openrouter.ai/qwen/qwen3.5-397b-a17b) — Multi-provider routing
- [Cheapest Cloud GPU Providers 2026 (Northflank)](https://northflank.com/blog/cheapest-cloud-gpu-providers) — Full comparison

---

## 6. Quantization

- [LLM Quantization Comprehensive Guide (AI Multiple)](https://research.aimultiple.com/llm-quantization/) — Overview of all quantization methods
- [vLLM Quantized KV Cache Docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) — FP8 KV cache quantization
- [Dynamic Expert Quantization for MoE (DynaExq)](https://arxiv.org/html/2511.15015v2) — MoE-specific quantization strategies
- [Qwen3.5-27B AWQ 4-bit (HuggingFace)](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit) — Quantized checkpoint we may use
- [NVIDIA Qwen3.5-397B NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4) — 4-bit NVIDIA-optimized checkpoint
