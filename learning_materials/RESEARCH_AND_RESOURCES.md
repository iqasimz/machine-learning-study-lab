# Research basis and source guide

Updated 1 November 2026; original source map checked 28 October. This is a source map, not compulsory homework to finish before starting. The lessons are original explanations; linked papers establish particular methods and documentation establishes implementation details. Source access does not certify every paper's claims or make one recipe universal.

## 1. How the scope was researched

The curriculum was cross-checked against Stanford's CS336 language-model lifecycle and CS224N NLP foundations, runnable textbook/framework teaching, original papers for major algorithms, and maintained documentation for serving, retrieval, MCP and safety. We deliberately cover both model research and the surrounding product system.

Source hierarchy:

1. Original papers for a method's objective and assumptions.
2. Maintained official project/protocol documentation for implementation.
3. Authors' interactive explanations for visual reinforcement.
4. Our own toy calculations and design scenarios to test understanding.

We did not use anonymous summaries, model rankings or generated lecture notes as technical authority. No advertised API price or leaderboard rank is frozen into the course. The source check verifies that the named material exists and supports its assigned role; it is not an exhaustive audit of every linked repository or every notebook's executable compatibility.

## 2. The learning spines

| Source | Use it for | Exactly what to do |
| --- | --- | --- |
| [Stanford CS336](https://cs336.stanford.edu/) | Whole lifecycle: tokenizers, architectures, systems, data, scaling, evaluation and post-training | After T1–T3, follow the matching lecture/assignment for a deeper implementation. Do not try to complete all cluster exercises immediately |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | Linguistic/NLP and neural-network foundations | Use word representations, backpropagation and Transformer material after P5–T3. Some current recordings are enrolled-student-only; the site links public earlier courses |
| [Dive into Deep Learning](https://d2l.ai/) | Maths with executable examples | Use preliminaries, softmax, MLP, attention and optimization alongside corresponding lessons; select one framework |
| [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Tensor/autograd/training mechanics | Work from tensor shapes to one full training loop. Change data and debug one error yourself |
| [Hugging Face LLM course](https://huggingface.co/learn/llm-course/en/chapter1/1) | Tokenizers, datasets and model interfaces | Study the matching tokenizer/data/training chapter after understanding the operation in our lessons |

These are supplements, not missing core explanations. University course academic-integrity rules still apply to their assessed assignments; this folder awards no university credit.

## 3. Static worked checks in the course

The website uses full, static diagrams rather than animated controls. The examples below are reading/calculation tasks, not prerequisites for opening another page.

| Topic | Read | Check your understanding |
| --- | --- | --- |
| Tensor storage and gradients | [P1](pre_llm/01_COMPUTERS_TENSORS_AND_SOFTWARE.md), [P2](pre_llm/02_LINEAR_ALGEBRA_AND_CALCULUS.md) | Distinguish view/copy/detach and derive the shapes of a linear-layer backward pass |
| Statistical evidence | [P3](pre_llm/03_PROBABILITY_INFORMATION_AND_STATISTICS.md), [P4](pre_llm/04_LEARNING_AND_GENERALIZATION.md) | Specify the paired resampling unit and what a transfer result does not establish |
| Attention and positions | [T3](llm_training/03_TRANSFORMER_FROM_FIRST_PRINCIPLES.md), [T4](llm_training/04_ARCHITECTURES_AND_MODALITIES.md) | Recalculate one attention row, GQA shapes and the difference between cache sharing and sparse visibility |
| Training updates | [T0](llm_training/00_THE_COMPLETE_TRAINING_LOOP.md), [T5](llm_training/05_STABLE_TRAINING_AND_DEBUGGING.md), [T7](llm_training/07_DISTRIBUTED_TRAINING_AND_KERNELS.md) | Derive the valid-token denominator, then the DDP averaging factor |
| Adaptation and preferences | [R1](post_training/01_SFT_AND_PARAMETER_EFFICIENT_ADAPTATION.md), [R2](post_training/02_PREFERENCES_RLHF_DPO_PPO_AND_GRPO.md) | Count LoRA parameters and explain a DPO gradient versus a rollout-policy objective |
| Serving and compression | [R6](post_training/06_INFERENCE_AND_SERVING.md), [R7](post_training/07_COMPRESSION_EDGE_AND_DEPLOYMENT.md) | Separate KV state, packed weights, metadata, workspace and measured latency |

Official documentation and original papers below provide deeper implementation details. Do not send private data to external demos. Advanced privacy, distributed systems and mechanistic interpretability still require specialist study; covering their role here does not certify mastery.

## 4. Original methods and official implementation sources

### Architecture, data and pretraining

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): original Transformer architecture.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): empirical compute/data/parameter trade-off; not a universal fixed ratio.
- [FineWeb](https://arxiv.org/abs/2406.17557): documented filtering/deduplication/data studies.
- [OLMo 2](https://arxiv.org/abs/2501.00656): open model/data/training recipe and stability choices.
- [LLaMA](https://arxiv.org/abs/2302.13971): representative dense autoregressive architecture/training context.
- [Switch Transformers](https://arxiv.org/abs/2101.03961): sparse expert routing and engineering concerns.
- [Mamba](https://arxiv.org/abs/2312.00752): selective state-space alternative.
- [Gemini report](https://arxiv.org/abs/2312.11805): example multimodal model family; not independent validation of all reported capabilities.

### Systems

- [FlashAttention](https://arxiv.org/abs/2205.14135): IO-aware exact attention.
- [NVIDIA parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html): concrete DP/TP/PP/EP strategies and constraints.
- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html): version-sensitive state sharding.
- [PagedAttention](https://arxiv.org/abs/2309.06180): inference cache management.
- [vLLM paged attention documentation](https://docs.vllm.ai/en/latest/design/paged_attention/): implementation-oriented view.
- [Speculative decoding](https://arxiv.org/abs/2211.17192): exact target-distribution sampling with proposal/verification machinery.

### Adaptation, feedback and RL

- [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314): low-rank adaptation and frozen quantized-base training.
- [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora): current conceptual implementation reference. The old stable URL returned a version-availability notice; use a guide matching your installed release.
- [InstructGPT](https://arxiv.org/abs/2203.02155): influential SFT/reward/preference pipeline.
- [PPO](https://arxiv.org/abs/1707.06347): original clipped policy objective.
- [DPO](https://arxiv.org/abs/2305.18290): preference-pair objective and regularized-policy connection.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): GRPO's original context.
- [TRL GRPO trainer](https://huggingface.co/docs/trl/en/grpo_trainer): current objective/configuration variants; a trainer's default is not the definition of all GRPO.
- [Constitutional AI](https://arxiv.org/abs/2212.08073): principle-guided critique and AI feedback.

### Retrieval, tools and safety

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401): original learned retrieval/generation formulation.
- [Sentence Transformers](https://www.sbert.net/): bi-encoders, rerankers, sparse and multi-vector interfaces.
- [Faiss index guide](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes): exact/ANN/index compression trade-offs.
- [Qdrant filtering](https://qdrant.tech/documentation/search/filtering/): example database filtering concepts; not universal product behaviour.
- [ReAct](https://arxiv.org/abs/2210.03629): interaction between generated reasoning/actions.
- [MCP architecture](https://modelcontextprotocol.io/docs/learn/architecture), [security](https://modelcontextprotocol.io/specification/latest/basic/security_best_practices): checked pages resolve to the 2026-07-28 documentation. Do not teach an older lifecycle as the only current protocol.
- [OWASP LLM risks](https://owasp.org/www-project-top-10-for-large-language-model-applications/), [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework): risk/threat/release frameworks.

### Compression and advanced research

- [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438): distinct quantization strategies.
- [Knowledge distillation](https://arxiv.org/abs/1503.02531): teacher/student formulation.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLX-LM](https://github.com/ml-explore/mlx-lm): maintained local-inference ecosystems; support depends on hardware/build/model.
- [ROME](https://arxiv.org/abs/2202.05262): factual-association editing example.
- [Differentially private deep learning](https://arxiv.org/abs/1607.00133): privacy-aware optimization/accounting foundations.
- [FrugalGPT](https://arxiv.org/abs/2305.05176): system-level model selection/cascades, distinct from token-level MoE.
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness): reproducible evaluation implementation, with task/configuration choices still needing inspection.

## 5. Further depth without pretending it is already taught

For an advanced method, use the bibliography/citations of the relevant original paper and the current official implementation. Before extending the course, write the missing skill explicitly: e.g. “derive a discrete denoising objective,” “implement an HNSW construction,” or “audit a privacy accountant.” Do not turn a new acronym into a claimed mastered skill.

The main specialist boundaries are recorded in [Coverage](COVERAGE_AND_LIMITS.md). Each expansion should retain the same structure: problem, plain-language mechanism, defined mathematics, worked example, diagram, failure case, comparison and checkpoint.

## 6. Access and security notes

Public page access was checked, not every repository dependency downloaded. No external notebook or pretrained artifact was executed as part of creating the lessons. Seeing Theory is archived. Browser/WebGL demos may not work on every machine; the local lab and arithmetic exercises remain available. One very large interpretability article could not be fetched reliably, so it was not used as the authority for a detailed claim.

Recheck URLs and versions before studying in 2027. Do not paste access tokens into interactive demos; do not enable unreviewed remote code; do not assume an online notebook's compute is free. Every core course explanation is readable offline from these files.
