# Course coverage and limits

## Depth labels

- **Derived core:** the course explains the mechanism, defines mathematics, works a small example and tests understanding.
- **Operational core:** explains design, failure modes and decision-making; meaningful practice needs a small implementation or system exercise.
- **Survey:** explains the family and its boundary, but not every proof/algorithm/implementation.

Self-contained means the main explanations are here. It does not mean an advanced research specialty has been reduced to one chapter or that reading replaces practice.

## Exercises and interview practice

The local labs support model-building, training, adaptation and evaluation. Systems exercises explore retrieval, tools and operational decisions without claiming a deployed assistant is included. [The interview coverage matrix](interview_preparation/COVERAGE_MATRIX.md) links coding, mathematics, debugging and system-design practice, including data structures, SQL, HTTP, concurrency, capacity, observability and recovery.

There are 44 guided technical questions, six worked design cases and 16 practical assessments. These are starting answer/check frameworks plus the existing detailed lessons, not an exhaustive oracle for every interview. All implementation/readiness status remains unattempted. Hardware-dependent and specialist topics have a minimum literacy/design exercise and a clearly separate optional implementation gate; they are not falsely labelled practically mastered.

## Lifecycle coverage

| Area | Location | Depth |
| --- | --- | --- |
| Python/environment/process/SSH/Linux/SLURM relationships | P1, T7 | Operational core; full beginner programming stays in programming foundations |
| Tensor shapes, dtypes, memory, algorithmic cost | P1, P2, T3, T6 | Derived core |
| Linear algebra, rank, chain rule, gradients | P2 | Derived core; SVD/eigenvalues introduced, not a full proof course |
| Probability, likelihood, entropy, KL, perplexity | P3, T1 | Derived core |
| Bootstrap, uncertainty, Holm, calibration/abstention | P3, P4 | Derived core with assumptions; advanced statistical theory not exhaustive |
| ML learning types, baselines, regularization, generalization | P4, P5 | Derived/operational core |
| NLP tasks, linguistic context, historical sequence models | P6 | Operational core; computational linguistics surveyed |
| Tokenizers, BPE, WordPiece/Unigram, Unicode, templates | P6, T2 | BPE derived; other tokenizer families surveyed |
| Embeddings, contrastive objectives, lexical/dense/sparse/late interaction | P7, R4 | Derived core for key objectives; index internals surveyed |
| Causal/masked/denoising training and target masks | T1, T2, T5 | Derived core |
| Data provenance, rights, filtering, deduplication, contamination | T2, T8 | Operational core |
| Synthetic data, mixtures, curriculum and forgetting | T2, R9 | Operational core; model-collapse theory surveyed |
| Deterministic IDs, hashes, manifests and source records | T2, R4, R8 | Operational core |
| Attention, residuals, MLPs, positions, normalization | T3, T4, P5 | Derived core; RoPE explained with a rotation |
| MoE, GQA/MQA, latent attention | T4, R6 | Core mechanisms; specialised routing/cache internals surveyed |
| State-space, hybrid and diffusion-language models | T4, R9 | Survey; not a complete denoising/scan derivation |
| Vision/audio/video/multimodal alignment | T4 | Survey/operational issues; not image/audio generation training |
| Optimizers, schedules, precision, clipping, accumulation | P5, T5 | Derived core with debugging |
| Parameter/FLOP/memory/scaling estimates | T6 | Derived core; not a hardware performance guarantee |
| DP/FSDP/ZeRO/TP/PP/EP/sequence parallelism | T7 | Operational core; implementations need authorised hardware |
| Kernels, FlashAttention, profiling | T7 | Mechanism core; custom CUDA/Triton programming not fully taught |
| SFT/full tuning/adapters/prefix/LoRA/QLoRA | R1 | LoRA/SFT derived; other PEFT branches surveyed |
| Reward model, critic, reference and rollout policy | R2 | Derived core |
| Policy gradient, KL regularization, PPO, DPO, GRPO | R2 | Derived core with numerical examples; full convergence proofs excluded |
| RLAIF/RLVR/process reward/RLOO/IPO/KTO/ORPO | R2 | Family map/survey; specific losses require specialist expansion |
| Prompting, few-shot, test-time compute, search and reasoning | R3, R9 | Operational core |
| Sampling, temperature, top-k/top-p, beam, stopping | R3, local lab | Derived core for principal decoding operations |
| Schema, reference and semantic validation | R3, R8 | Operational core |
| RAG ingestion/chunking/retrieval/reranking/citations | R4 | Operational core with design exercises |
| Vector databases, HNSW/IVF/PQ, ANN versus task recall | R4 | Core decisions; algorithm construction surveyed |
| GraphRAG/multi-hop/query rewrite/corrective retrieval | R4 | Survey with failure boundaries |
| Tools/agents/memory/multi-agent orchestration | R5 | Operational core |
| MCP architecture/protocol versions/security | R5 | Operational core; wire-level implementations use current spec |
| Prefill/decode/KV/batching/caching/speculation | R6 | Derived core for memory; serving design operational |
| Quantization/distillation/pruning | R7 | Derived core for quantization; method families surveyed |
| Cloud/self-hosted/on-device/hybrid deployment | R7 | Operational core |
| Evaluation, judges, source faithfulness, drift and shift | T8, R8 | Operational core grounded in P3/P4 |
| Security/privacy/permissions/supply chain/incident response | R5, R8 | Operational core; not professional security certification |
| Monitoring/traces/versioning/cost/capacity/rollback | R6–R8 | Operational core |
| Editing/merging/unlearning/continual/federated/DP | R9 | Survey with key mathematical definitions |
| Probes/patching/SAEs/steering and claim boundaries | T8, R9 | Survey with SAE/steering formulation |

## What would require a further specialist course?

Full mathematical proofs of generalization/concentration; measure-theoretic probability; optimization convergence; complete policy-gradient/PPO theory; every modern preference objective; privacy accountants; detailed HNSW/PQ implementation; distributed storage consensus; kernel/compiler programming; full speech/vision/diffusion generation; formal verification; mechanistic circuit discovery; multilingual linguistic theory; domain-specific law, medicine and safety certification.

Those are not silently “covered” by an acronym. The relevant lessons identify the next question and the primary source to begin with. Future expansion should add worked lessons rather than inflate a resource list.

## Fast-changing details to recheck

1. MCP specification and SDK versions; the checked architecture differs from many older tutorials.
2. Library trainer defaults, especially GRPO loss normalization, KL and importance weights.
3. Model/tokenizer/chat-template revisions and model/data licenses.
4. Quantization formats, operators and accelerator support.
5. Provider pricing, retention, regions and rate limits.
6. Benchmark integrity, task definitions and evaluation harness versions.
7. Applicable legal/regulatory requirements for real deployment.

## Maintain the course

For each update, record date, changed assumption/source, affected lesson, revised example/test and whether an old statement is now obsolete. Keep stable mathematics separate from release-specific instructions. If a new model claims a breakthrough, add a controlled comparison or explicit uncertainty rather than replacing the core with marketing.
