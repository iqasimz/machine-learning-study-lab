# T5. Gradient accumulation and stable LLM training

Gradient accumulation combines several small forward/backward calculations before one optimizer update. The loss weighting must match the intended objective, especially when microbatches contain different numbers of valid targets. Numerical stability, clipping, scheduling and diagnostics are additional training choices, not substitutes for correctly aligned data and targets.

Microbatch A has two scored tokens with mean loss one; B has a selectable token count with mean loss three. At six B tokens the combined token mean is 2.5, while averaging the two microbatch means gives 2.0. The control changes this accounting, not a live run.

The training loop identified where gradients are accumulated and applied. This lesson derives their weighting on one worker; the distributed lesson later adds the separate convention used to combine workers.

## 1. Minimal training loop

**Key terms.** An **optimizer** applies updates; **zero_grad** clears stored gradients; **backward** accumulates derivatives; **evaluation mode** controls behaviours such as dropout.

**Process.** Fetch a batch, clear gradients at the update boundary, compute loss, backpropagate and update parameters. Validation is a separate branch with no optimizer step; verify the smallest loop before extending it.

Before scaling, run a single process, one device and a tiny clean batch. For a causal model, confirm that the last input position predicts the next token, not the token already at that position. Decide whether the framework or your code shifts labels.

Pseudocode, not a copy-paste library recipe:

```text
load configuration, tokenizer and document-level splits
initialize model and optimizer; record all versions
repeat:
    fetch token batch and visibility/loss masks
    compute logits
    compute SUM of valid-token negative log probabilities
    divide by the intended token count for the update
    backpropagate
    at the update boundary: synchronize as required, clip, optimizer step
    periodically evaluate a fixed held-out set without weight updates
    save complete resumable state and inspect samples
```

## 2. Token-weighted gradient accumulation

**Key terms.** A **microbatch mean** averages its scored tokens; an **update mean** averages all scored tokens across accumulation; **K** is the number of microbatches.

**Process.** Count valid targets before scaling the losses. Accumulate token-weighted contributions, then unscale and clip once at the update boundary before changing weights. Equal microbatch weights only match equal target counts.

Accumulation processes several microbatches before an optimizer step. Effective batch size in examples is approximately `microbatch × accumulation × data-parallel workers`, assuming equal batch sizes and a standard synchronization convention. Token counts vary with length/masking.

If microbatch A contains 10 valid targets and B contains 90, averaging their two mean losses gives each microbatch half the influence. Averaging all 100 token losses gives weights 0.1 and 0.9. Neither is automatically the intended objective. For token-mean training, accumulate sums with the corresponding global valid-token denominator. Distributed reduction semantics can introduce another scale factor.

Accumulation is not always exactly equivalent to one large forward batch: dropout, normalization variants, clipping, optimizer scheduling and rounding can differ. Define learning-rate schedule units—optimizer updates versus microsteps versus tokens.

### A two-microbatch update, with the denominator made explicit

Suppose the next update uses two microbatches containing 10 and 90 valid targets. Their mean losses are 2 and 4.

1. Convert means to sums: `S₁=20`, `S₂=360`; total target count is 100.
2. Backward `S₁/100`, retaining its parameter gradients but freeing that forward graph. Backward `S₂/100` on the second graph. The accumulated gradient matches the derivative of `(S₁+S₂)/100` under the same deterministic objective.
3. The combined scalar loss for reporting is 3.8. Averaging the two means would report 3.0 and optimize a different weighting.
4. Unscale if required, check finite gradients, clip the accumulated norm, then call the optimizer once. Clear gradient buffers at the boundary so the next update does not include this one unintentionally.
5. For a final incomplete accumulation window, use its actual intended denominator. Dividing every microbatch by a fixed accumulation count can incorrectly shrink the last update.

Knowing the total count before the backward calls may require preparing the small group first or a carefully specified equivalent normalization strategy. Avoid retaining every microbatch's full computation graph just to normalize later. Distributed averaging adds a world-size convention; T7 derives it separately. Reference: [PyTorch AMP accumulation and clipping](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html).

## 3. Numerical stability

**Key terms.** **FP16/BF16/FP32** are floating-point formats; **loss scaling** protects small gradients in some mixed-precision recipes; **non-finite** means NaN or infinity.

**Process.** Compute in supported precision, inspect losses and gradients, then convert or unscale before clipping and updating. Invalid arithmetic must be diagnosed before it contaminates subsequent optimizer state.

Use stable log-softmax/cross-entropy implementations rather than taking logs of rounded probabilities. Compute sensitive reductions in suitable precision. FP16 may require loss scaling to preserve small gradients; BF16 has a wider exponent range but not FP32 precision. Hardware and kernels determine supported operations.

Initialize weights to avoid exploding or vanishing activation scales. Monitor gradient norm, activation norms, attention-logit scale and embedding/output norms. Warmup, normalization, clipping and cautious optimizer settings are tools, not replacements for finding data/label errors.

## 4. Diagnosing training failures

**Key terms.** **Divergence** is unstable worsening; **overfitting** improves training but harms held-out behaviour; a **bottleneck** limits throughput.

**Process.** Observe the failure pattern, check data and numerics, and change one plausible cause at a time. Re-run a controlled small case before committing another full training budget.

| Symptom | First checks | Do not immediately conclude |
| --- | --- | --- |
| Loss implausibly near zero | Target leakage, wrong shift, padding-only scoring | “The architecture is amazing” |
| Loss flat near random baseline | Gradients, learning rate, trainable parameters, targets | “Need a larger model” |
| NaNs after a few steps | Overflow, invalid inputs, normalization, rate, division by zero | “Mixed precision never works” |
| Train improves, validation worsens | Overfitting, duplicates, mixture and distribution | “Validation is broken” |
| Repetitive generation | Data repetition, decoding, EOS/template, model quality | “Temperature alone fixes it” |
| GPU underused | Data loader, transfers, small kernels, synchronization | “Need more GPUs” |
| Resume changes behaviour | Missing optimizer/scheduler/RNG/sampler state | “Seeds guarantee equivalence” |

## 5. Efficient experiments

**Key terms.** A **pilot** is a short feasibility run; an **ablation** isolates a component; **early stopping** ends a run under a declared validation rule.

**Process.** Establish a working baseline, run a small controlled change and inspect its evidence. Scale the promising comparison only after correctness and resource costs are understood.

Start with a reference implementation and inspect its assumptions. Establish a fixed baseline, then change one meaningful component. Keep a run manifest with model/tokenizer hashes, config, data version, code revision, seed, hardware, package versions and valid-token exposure.

Run a **smoke test** for execution and shapes; a **tiny overfit test** for learning; a **short validation run** for obvious instability; then the budgeted comparison. A short run's ranking may reverse at scale, so call it a screening result, not the final conclusion.

Checkpoint state includes model, optimizer, scheduler, mixed-precision scaler when relevant, random states, data position and configuration. Checkpoint writes can be interrupted: write to a temporary destination and publish a complete artifact atomically where the storage system supports it. Verify restoration on a small run.

## 6. Optimizer and schedule settings

**Key terms.** A **schedule** changes learning rate over training; **warmup** increases it initially; **weight decay** shrinks parameters under its own rule.

**Process.** Choose the objective and effective batch first, then set optimizer, rate schedule and stopping policy. Record schedule units so resume or accumulation changes do not silently alter the recipe.

Batch by similar length when it reduces wasted padding without introducing unwanted ordering effects. Cache deterministic preprocessing. Use streaming/sharded reads when corpus size exceeds RAM. Measure tokens/second, not only batches/second. Long sequences and short sequences can have very different compute per batch.

Use held-out diagnostics by domain/language/length, not just one average. If the training mixture changes, compare exposure as well as update counts. Interpret early stopping against a predefined criterion; repeated hand selection based on generated samples is another source of selection bias.

## Exercises and checks

1. Microbatch means are 2 and 4 with 10 and 90 valid tokens. Token-mean loss? **3.8**, not 3.
2. What is the first experiment when a model cannot learn? A tiny clean overfit test with audited labels and gradients.
3. List the exact state needed to resume your proposed toy experiment; distinguish “approximately continue” from “bitwise reproduce.”

References: [PyTorch training basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [Stanford CS336](https://cs336.stanford.edu/), [OLMo 2 training recipe](https://arxiv.org/abs/2501.00656). The diagnostic table is a general engineering checklist, not a guarantee of one-to-one causes.
