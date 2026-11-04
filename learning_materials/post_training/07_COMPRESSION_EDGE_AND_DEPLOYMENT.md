# R7. LLM quantization, compression and deployment

Compression changes a model or its representation to reduce selected storage or execution costs. Quantization approximates numerical values using a declared code range and scale; pruning and distillation use different mechanisms. Smaller values or files do not automatically produce a faster, fitting or equally capable deployed model.

The weights [−1.2,0.4,0.8] are mapped to the full signed four-bit range [−8,7] with zero point zero. Change scale to inspect rounding, clipping and reconstruction error. The browser's ordinary arrays are not packed four-bit storage or a low-bit inference kernel.

Serving separated weights, cache and runtime memory. This lesson approximates stored numerical values while keeping value error, actual memory savings, hardware support and task quality as separate outcomes.

## 1. Compression methods

**Key terms.** **Quantization** lowers precision; **pruning** removes components; **distillation** trains a smaller student; **compression** targets storage or computation cost.

**Process.** Identify the expensive resource, choose a matching compression method and preserve a baseline. Re-evaluate quality and runtime rather than assuming smaller means faster.

**Quantization** reduces numerical precision. **Pruning** removes selected weights/components. **Distillation** trains a student using a teacher's outputs or representations. **Low-rank approximation** restricts transformations. **Architecture selection** starts with a different model. Each can reduce cost, but they do not preserve exactly the same capabilities or require the same runtime support.

## 2. Quantization by hand

**Key terms.** A **scale s** maps integer steps to real values; a **zero-point z** represents real zero; **clipping** bounds integer values.

**Process.** Choose the range and scale, map weights to integers and reconstruct approximate values. Compare reconstruction error with the downstream model's output change.

**Calculation.** This separate symmetric example uses integer values from −7 to 7; the live fixture uses the full signed four-bit range −8 to 7. They are explicitly different quantizers. With the symmetric range and scale s=0.1, weight 0.26 rounds to q=3 and reconstructs as 0.30, an error of +0.04. Weight 1.0 clips to q=7 and reconstructs as 0.70. Real 4-bit formats can use different ranges and codebooks.

For affine integer quantization:

\[
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),
\qquad \hat x=s(q-z).
\]

`x` is the original value, `q` the stored integer, `s>0` a scale and `z` a zero point. Symmetric quantization commonly sets zero point to zero under an appropriate integer range. Per-tensor, per-channel and groupwise scales trade metadata for accuracy.

With signed range `[-7,7]`, scale `1/7`, zero point 0 and `x=0.6`, stored integer is 4 and reconstructed value is `4/7≈0.5714`. Error is about −0.0286. Real formats may reserve a code, use asymmetric ranges or nonlinear codebooks; this is a toy calculation.

**Weight-only** quantization leaves some activations at higher precision. **W8A8** denotes 8-bit weights and activations in the affected operations. **KV quantization** targets cache state. **PTQ** quantizes after training, often with calibration data; **QAT** trains while simulating/handling quantization effects. “4-bit model” does not mean every operation/state is 4-bit.

## 3. Calibration and runtime support

**Key terms.** **Calibration data** estimates quantization settings; a **kernel** executes a numeric format; **dequantization** reconstructs usable values.

**Process.** Calibrate on representative inputs, encode weights and load a compatible runtime. Benchmark next: unsupported low-bit formats may save storage without reducing latency.

Activation outliers can make a single scale waste precision on common values. GPTQ uses reconstruction/approximate curvature information for weight quantization; AWQ uses activation-informed weight scaling/protection; SmoothQuant redistributes scale between weights and activations to ease quantization. These are different methods, not interchangeable file extensions.

Quantized weights may require dequantization or specialized kernels. Without supported kernels, smaller storage can be slower. Calibration data should represent intended use without leaking the final test. Evaluate rare numbers, code, long contexts, languages and refusal behaviour, not only average loss.

### Work out quantized storage, then test the runtime separately

Consider one `4096×4096` weight matrix quantized to four bits, with groups of 128 weights and one two-byte scale per group.

1. There are 16,777,216 weights. Packed four-bit values need 8,388,608 bytes: 8 MiB.
2. There are 131,072 groups. Their scales need 262,144 additional bytes: 0.25 MiB.
3. This simplified format therefore uses 8.25 MiB, before zero points, alignment, headers or other tensors. The comparison FP16 matrix uses 32 MiB.
4. A compute kernel may dequantize tiles into a higher precision. Temporary workspace, activations and KV cache remain separate memory terms.
5. Benchmark the actual supported hardware and kernel after warmup, including sustained performance and output quality. A smaller file is not proof of lower latency or acceptable error.

Group size changes the quality/metadata trade-off. Calibration examples should cover intended activation distributions but stay separate from the final quality test. Test numerical entities, code and long contexts rather than only average perplexity. Sources: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978) and [QLoRA](https://arxiv.org/abs/2305.14314). Their quantization/optimization recipes differ; the arithmetic example is not a description of every four-bit format.

## 4. Distillation and pruning

**Key terms.** A **teacher** supplies behaviour; a **student** learns it; **structured pruning** removes whole components instead of arbitrary entries.

**Process.** For distillation, produce teacher targets and train the student. For pruning, remove components and recover/test quality. Both need evaluation beyond the original model's score.

A student can learn from hard teacher outputs, soft probability distributions, intermediate representations or generated reasoning traces. A common soft-target term is `T² KL(p_teacher^T || p_student^T)`, combined with a task loss. Here `T` is distillation temperature, not sequence length. The `T²` factor is conventional gradient-scale compensation under that formulation.

Teacher mistakes can transfer; a student's good benchmark score does not prove faithful reasoning. Trace distillation may imitate persuasive explanations without the intended computation. Maintain independent data and teacher-free evaluation.

Unstructured pruning removes individual weights; structured pruning removes blocks, channels, heads or layers. Sparse weights speed up only if the runtime/hardware exploits their pattern. A smaller dense architecture can outperform an awkward sparse one in deployment despite similar nominal parameter savings.

## 5. Cloud, on-device and hybrid

**Key terms.** **On-device** runs locally; **cloud** runs remotely; **hybrid** divides work; a **privacy boundary** limits where data goes.

**Process.** Map latency, privacy and memory requirements to a deployment location. Measure the whole path including network/fallbacks before choosing a smaller local or larger remote model.

| Choice | Benefits | Real constraints |
| --- | --- | --- |
| Cloud API | Managed capacity and fast integration | Network, provider changes, cost, retention and data jurisdiction |
| Self-hosted server | Control over weights, logs and infrastructure | Operations, security, GPU cost and upgrades |
| On-device/edge | Offline use, low network dependence, local processing | RAM, bandwidth, battery, heat, model distribution and device compromise |
| Hybrid | Local first with controlled escalation | Routing quality, consent, inconsistent behaviour and network failure |

Local processing can reduce data exposure but is not automatically private: apps may log/sync content, devices can be compromised and model outputs can reveal information. Unified memory is shared capacity, not unlimited memory or bandwidth. Mobile thermal throttling makes short benchmark bursts misleading.

## 6. Deployment packaging

**Key terms.** An **artifact bundle** includes weights/tokenizer/configuration; a **runtime** executes it; a **revision** fixes compatibility.

**Process.** Package matching artifacts, load in the target runtime and run fixed regression prompts. Include rollback and license information before distribution.

Package model weights, tokenizer, configuration, template, quantization metadata and runtime compatibility. Formats such as GGUF are ecosystem choices; format conversion is not a guarantee of equivalent output or licensing permission. Verify downloaded artifacts and avoid executing arbitrary remote model code.

Select a runtime for actual hardware: CPU instruction support, GPU backend, Apple Silicon, mobile accelerator or embedded device. Check operator support and context limits. Tools such as llama.cpp and MLX-LM cover particular ecosystems; they are not interchangeable across every device.

Use a staged release: offline evaluation → local/integration tests → limited canary → monitored rollout → rollback-ready deployment. An edge update needs signed artifacts, compatibility checks, interrupted-download recovery and a safe previous version. Preserve user privacy during telemetry collection.

## 7. A deployment decision example

**Key terms.** A **criterion** is a required quality/cost limit; a **trade-off** improves one requirement at another's expense.

**Process.** Measure candidates on the same task and constraints, then select a deployment meeting the full requirements. Keep alternatives and their measured costs available for review.

A field researcher needs offline abstract search. First try lexical retrieval plus a small local model and explicit sources. Test the weakest supported device, actual document lengths, cold start, sustained latency and battery. If a query escalates to cloud, ask permission and send only necessary approved data. Do not make the offline promise depend on an invisible network call.

## Exercises and checks

1. Why can a 4-bit 7B model still fail on an 8 GB device? Weights are not total memory; cache, workspace, app/OS memory and allocation peaks matter.
2. What is a proper compression comparison? Same tasks/prompts, precision/runtime stated, quality slices, memory peaks, sustained latency, energy assumptions and failure cases.
3. Can pruning 50% of weights guarantee 2× speed? **No:** pattern, kernels, hardware and bottlenecks determine speed.

Sources: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438), [distillation](https://arxiv.org/abs/1503.02531), [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLX-LM](https://github.com/ml-explore/mlx-lm). Do not use old performance numbers as current hardware recommendations.
