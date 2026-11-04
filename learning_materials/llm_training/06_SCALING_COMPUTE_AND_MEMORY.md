# T6. LLM parameter counts, memory and compute

A model's parameter count, raw weight storage, training-state memory and compute budget measure different things. Training can require gradients, optimizer history, activations and temporary workspaces beyond the weights. Resource estimates are conditional accounting tools; measured peak memory and throughput must establish whether a real configuration fits and runs usefully.

Select one, ten or one hundred million hypothetical parameters. The simple convention stores FP32 weights, FP32 gradients and two FP32 Adam moments. Its static state uses sixteen bytes per parameter before activations or runtime overhead; other recipes need different accounting.

The training loop explains what values exist and when. This lesson counts their storage and work before scaling, while keeping the detailed prepared-model count, scaling laws and hardware bottlenecks in the deeper explanations.

## 1. Parameter counting

**Key terms.** **Parameters N** are learned numbers; **training tokens D** count exposures; **d** is hidden width; **d_ff** is MLP width; **V** is vocabulary size.

**Process.** Count each matrix from its dimensions before estimating storage or compute. Add shared parameters only once, then use the total in the separate memory and work calculations below.

Parameters, training tokens, context length, update count, hardware time and money are different quantities. A model with fewer parameters can cost more overall if trained on many more tokens or served millions of times. Total lifetime cost includes data preparation, experiments, training, evaluation and inference.

In a simple dense decoder block, attention projections contribute roughly `4d²` weights and a non-gated MLP roughly `2 d d_ff`, ignoring biases/norms. With `d_ff=4d`, that is about `12d²` per layer. Gated MLPs, GQA, tied embeddings and expert layers change the count. Add embeddings `Vd` and any untied output projection.

### Count the prepared model instead of guessing from its name

For the configured bias-free SmolLM2 model: vocabulary 49,152; width 960; 32 layers; KV projection width 320; gated intermediate width 2560; tied input/output embeddings.

1. The shared embedding/output matrix contains `49,152×960 = 47,185,920` parameters. Do not count it twice when weights are tied.
2. Each block's attention matrices contain `960×960 + 960×320 + 960×320 + 960×960 = 2,457,600` parameters: query, key, value and output projections.
3. Its three gated-MLP matrices contain `3×960×2560 = 7,372,800` parameters. A two-matrix GELU formula would undercount this model.
4. Two RMSNorm scale vectors add `2×960 = 1,920` per block. Across 32 blocks the subtotal is `32×9,832,320 = 314,634,240`.
5. Add the shared embedding and final 960-element RMSNorm: total **361,821,120**. This is a parameter count, not the peak-memory requirement.

The recipe follows the [model configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json). Another architecture can have biases, untied heads or expert weights, so reuse the counting procedure, not the total. The compute-optimal scaling study in [Chinchilla](https://arxiv.org/abs/2203.15556) addresses a separate question: how to allocate a training-compute budget between parameters and tokens.

## 2. Training memory

**Key terms.** **Bits per value** determine raw storage; **activations** are forward intermediates; **optimizer state** stores update history; **GiB** means 2³⁰ bytes.

**Process.** Calculate raw weight bytes first, then add gradients, optimizer state, activations and temporary buffers. Compare the complete peak estimate with available device and host memory before launching training.

For `N` parameters stored at `b` bits, raw weights need `Nb/8` bytes. A 7-billion-parameter model at 16 bits needs about 14 GB decimal, or 13.04 GiB. That is **not** a complete training or serving requirement.

An illustrative mixed-precision Adam setup might store:

| Component | Bytes/parameter in this example |
| --- | ---: |
| Low-precision parameter copy | 2 |
| Low-precision gradients | 2 |
| FP32 master weights | 4 |
| Two FP32 Adam moments | 8 |
| **Subtotal** | **16** |

For 7B parameters, subtotal is 112 GB before activations, workspace, communication buffers and allocator overhead. Other implementations use different gradient/master-state choices. Do not present “16 bytes” as a universal constant.

Activations depend on batch, sequence length, width, layers and kernels. Gradient checkpointing discards selected intermediate activations and recomputes them during backward, trading compute for memory. Optimizer sharding distributes state but adds communication. Offloading moves state to CPU/disk and can create a bandwidth bottleneck.

## 3. FLOPs and time estimates

**Key terms.** A **FLOP** is a floating-point operation; **throughput** is operations per second; **sustained throughput** is achieved rather than advertised speed.

**Process.** Estimate total arithmetic from model and token counts, divide by measured or explicitly assumed sustained speed, then add non-compute overhead. Treat the result as an estimate, not a benchmark.

A common rough estimate for dense Transformer training is `C≈6ND`, where `N` is parameter count and `D` is training-token count. It approximates forward/backward matrix operations; attention at long context, embeddings, architecture and implementation can change the total.

For `N=10^9`, `D=20×10^9`, estimate `C≈1.2×10²⁰` FLOPs. If an assumed **sustained**, not peak, throughput is `10^14` FLOPs/s, time is `1.2×10^6` seconds, about 13.9 days. This is an arithmetic illustration, not a hardware benchmark. Add evaluation, checkpoints, failures and orchestration overhead.

Model FLOPs utilization compares useful model compute with a stated hardware peak, but definitions and precision conventions vary. GPU “100% utilization” is not equivalent to peak useful model FLOPs.

## 4. Scaling laws

**Key terms.** A **scaling law** is an empirical fitted relation; **N/D** are parameter/token counts; **A/B/α/β/E** are fitted constants; **C** is compute budget.

**Process.** Fit loss against model and data sizes, impose a budget, then examine its preferred allocation. Check whether the new architecture, corpus and deployment regime match the fit before using its recommendation.

A stylized loss model is:

\[
L(N,D)\approx E+A N^{-\alpha}+B D^{-\beta}.
\]

`E` is an estimated irreducible term within this fit; other constants are fitted empirically. Holding compute `C≈kND` fixed gives `D=C/(kN)`. Differentiating the resulting two-term expression yields approximately:

\[
N^*\propto C^{\beta/(\alpha+\beta)},\qquad
D^*\propto C^{\alpha/(\alpha+\beta)}.
\]

This explains a trade-off: more parameters leave fewer training tokens under a fixed budget. The fitted exponents and constants depend on the regime and recipe. A popular tokens-per-parameter heuristic is not a law for every dataset, MoE, context length or inference-heavy product. A smaller model trained longer can be preferable when serving cost dominates.

## 5. Compute and bandwidth limits

**Key terms.** **Arithmetic intensity** is work per byte moved; **bandwidth** is bytes moved per second; a **roofline** bounds speed using compute and bandwidth limits.

**Process.** Estimate the operation's data movement and arithmetic, then identify the tighter hardware limit. Profile the real workload next to determine whether computation or memory traffic deserves optimization.

Compute-bound work waits for arithmetic throughput; memory-bound work waits for data movement. **Arithmetic intensity** is FLOPs per byte moved. A large matrix multiplication can reuse loaded values and achieve high intensity. A small-batch decode step may move many weights for relatively few new tokens.

If memory bandwidth doubles but compute stays fixed, a memory-bound task may improve substantially; a compute-bound task may not. These are useful limiting models, not labels permanently attached to all prefill/decode workloads.

## 6. Experiment budgeting

**Key terms.** A **comparison budget** includes all runs and checks; **accelerator hours** measure occupied devices over time; **energy** integrates power over time.

**Process.** Allocate resources to baselines, seeds, pilots, evaluation and failures before choosing run sizes. Report the same budget unit for competing methods so equal updates are not mistaken for equal cost.

Budget the entire comparison: pilot runs, seeds, baselines, failed runs, data generation and evaluation. State whether equal budget means equal tokens, optimizer steps, elapsed time, accelerator hours or money. Those equalities are not interchangeable.

Energy is power integrated over time; carbon estimates also need location/time-dependent factors and system boundaries. Parameter count alone does not establish sustainability. Report assumptions rather than inventing precise emissions.

## Exercises and checks

1. 1B parameters at 4 bits: raw weights about 0.5 GB. What is missing? Quantization metadata, selected higher-precision tensors, activations/cache and runtime overhead.
2. `N=2B,D=10B`: rough dense-training compute is `1.2×10²⁰` FLOPs, the same product as the earlier example but not necessarily the same quality or actual runtime.
3. Why can lowering weight precision fail to speed up inference? Hardware/kernel support, dequantization, launch overhead, cache or another bottleneck.

Primary reading: [compute-optimal scaling](https://arxiv.org/abs/2203.15556). Systems teaching: [CS336](https://cs336.stanford.edu/). Separate raw weights, optimizer state, activations and inference cache before estimating the complete resource budget.
