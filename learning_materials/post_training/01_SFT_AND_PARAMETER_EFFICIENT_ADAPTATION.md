# R1. Supervised fine-tuning and LoRA

Supervised fine-tuning adapts a pretrained model using demonstrations of desired outputs. Low-rank adaptation, or LoRA, changes selected projections through small trainable matrix factors while keeping their base weights frozen. These choices concern the training signal and trainable parameters; they do not remove the base model or guarantee better behavior.

A two-feature input [2,1] follows an identity base projection and a separate supplied LoRA correction. Change rank to inspect their addition and parameter count. The adapter values are fixed examples, not learned updates or a fair benchmark comparing trained ranks.

A trained base can be adapted rather than learned from scratch. First decide whether the failure needs changed weights, retrieved information or an external action; this lesson then explains the parameter boundary of one adaptation method.

## 1. Choosing an adaptation method

**Key terms.** A **base model** supplies pretrained capabilities; **SFT** learns from demonstrations; **retrieval** supplies evidence without updating weights.

**Process.** Identify whether the failure is missing information, formatting or learned behaviour before choosing adaptation. Establish the unchanged model's performance first, then select an intervention and a held-out check.

New changing facts often call for retrieval. An external action needs a tool. Consistent style, output structure or task behaviour may benefit from supervised fine-tuning. Domain vocabulary/knowledge may justify continued pretraining, followed by task adaptation if needed. Start with a strong prompt baseline so you know whether training is necessary.

**SFT**, supervised fine-tuning, updates weights using demonstrations `(instruction, desired response)`. It is usually next-token learning on selected response positions, not a new neural architecture.

\[
L_{SFT}=-\frac1{N_{response}}\sum_{i,t}m_{it}\log\pi_\theta(y_{it}\mid x_i,y_{i,<t}).
\]

`x` is prompt/context, `y` demonstrated response, `m` marks included targets. Some setups include prompt loss; others mask it. State your choice. A schema-valid but factually wrong demonstration trains the wrong behaviour very efficiently.

## 2. Full fine-tuning versus PEFT

**Key terms.** **Full fine-tuning** updates all intended weights; **PEFT** updates a smaller parameter set; **frozen** weights receive no update.

**Process.** Load a compatible model and tokenizer, select trainable parameters and run supervised loss. Compare cost and behaviour with the same starting base.

Full tuning changes most/all selected model parameters. It has broad capacity but substantial optimizer memory and potential forgetting. **PEFT** changes a smaller parameter subset or adds small modules:

- **Adapters:** small trainable modules inside the network.
- **Soft prompt tuning:** learned input embeddings, not ordinary human-written prompt text.
- **Prefix tuning:** learned prefix-like internal representations, implementation-dependent across layers.
- **Bias-only / selective tuning:** update a restricted existing subset.
- **LoRA:** factorize trainable weight changes into low-rank matrices.

Fewer trainable parameters can save optimizer memory, not all activation memory. The frozen base still participates in forward/backward computation as needed for gradients through the trainable components.

## 3. LoRA mathematics

**Key terms.** **LoRA** represents a weight change as two smaller matrices; **r** is their inner rank; **α/r** scales the change; **W₀** is the frozen base matrix.

**Process.** Input follows the original projection and a trainable low-rank branch. Add both outputs, compute loss and backpropagate into the adapters; base weights remain fixed.

For row-vector convention, a base layer `xW` with `W:[d_in,d_out]` becomes:

\[
y=x\left(W+\frac\alpha r AB\right),\qquad
A:[d_{in},r],\ B:[r,d_{out}].
\]

`r` is rank/bottleneck width and `α` a scaling hyperparameter. Some libraries use transposed conventions or different rank scaling; verify implementation. Train `A,B` while keeping `W` fixed. Often initialization makes the initial update zero so the model begins at the base behaviour.

Full `4096×4096` update: 16,777,216 parameters. LoRA rank 8: `4096×8+8×4096=65,536`, or 1/256 as many for that matrix. That ratio is not the whole model's memory reduction.

Target modules matter: query/value projections only, all attention projections or MLP layers give different capacity. Rank is not the number of tasks or reasoning steps. Higher rank can help but may overfit or increase cost. Match data and training budgets when comparing ranks.

### Why zero initial adapter output does not stop learning

Use the row-vector convention above and initialize `A` randomly and `B` to zero. Then `AB=0`, so the added branch initially changes nothing.

1. The base model supplies the initial prediction. Response masks determine which known target tokens contribute to the SFT loss.
2. Backward computes a gradient for `B` through `XA`; that can be nonzero even though `B` is zero.
3. The first gradient for `A` contains the initially zero `B` factor, so it can be zero. After `B` changes, later updates can train both factors. Freezing both because one first-step gradient is zero would be an incorrect diagnosis.
4. The base weight `W` stays frozen, but its operation can still participate in propagating gradients through the network. LoRA saves parameter-state memory; it does not delete the base model from the computation.
5. Save the exact base revision, rank, scaling, target modules and template with the adapter. The same rank can yield different whole-model parameter counts when different projections are targeted.

For the local rank-8 query/value setup, each layer contributes `8×(960+960)+8×(960+320)=25,600` adapter parameters. Across 32 layers, that is **819,200**. References: [LoRA](https://arxiv.org/abs/2106.09685) and [PEFT LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora). Initialization/scaling variants exist, so inspect the selected implementation rather than assuming every adapter starts this way.

## 4. QLoRA and quantized bases

**Key terms.** **QLoRA** combines a quantized frozen base with trainable adapters; **quantization** stores an approximate low-bit representation.

**Process.** Load the supported quantized base, compute through compatible kernels and train the higher-precision adapters. Measure memory and speed including activations and dequantization.

QLoRA trains adapters through a frozen quantized base, with selected computations performed at higher precision. Its recipe includes quantization choices and memory-saving techniques. It does **not** mean freely updating arbitrary 4-bit integer weights with ordinary Adam.

Frozen base quantization error, adapter precision, dequantization, optimizer state and activation memory all matter. Merge/export carefully: a trained adapter must be paired with its exact compatible base and tokenizer. A merged high-precision model and a re-quantized export may not produce identical outputs.

## 5. Data design and evaluation

**Key terms.** A **demonstration** pairs input with desired output; a **response-only mask** scores answer tokens; **contamination** compromises a holdout.

**Process.** Curate and split demonstrations by source, serialize them with the correct template and score only declared targets. Compare base and adapted outputs on untouched local holdouts while acknowledging pretraining overlap.

Use representative instructions with correct responses, failures and appropriate abstentions. Split by source/document before creating related demonstrations. Keep a held-out task set and general-capability regression set. Add difficult counterexamples instead of simply duplicating easy examples.

Chat-template errors, unmasked padding and accidental prompt leakage are common. Inspect the actual rendered token sequence and label mask for at least several examples. A training library accepting the dataset does not guarantee that it learned only the intended response tokens.

Learning rate, epochs, sequence length, effective valid-token batch, regularization and checkpoint selection are all experimental variables. Smaller datasets can overfit quickly. Lower SFT loss on demonstrations does not imply improved preferences or safety.

## 6. Checkpoint selection and deployment

**Key terms.** A **checkpoint** fixes saved parameters; **forgetting** is lost prior behaviour; an **adapter** must match its base revision.

**Process.** Choose rank and budget on development evidence, save the selected adapter and verify base compatibility. Check useful gains and regressions before using it.

**Multiple customers:** separate adapters can be cheaper than full model copies, but adapter routing and tenant isolation must be correct. **Changing policy:** version both adapter and policy data. **No improvement:** check whether the base already solves the task with correct prompting; whether demonstrations are inconsistent; whether evaluation uses a different template; whether the target modules can express the change.

**Model merging** is not automatically safe: adapters trained from different bases or targeting different modules may not combine meaningfully. R9 discusses merging and interference.

## Exercises and checks

1. Matrix `1024×2048`, rank 4: adapter has `4×(1024+2048)=12,288` parameters, compared with 2,097,152 full weights.
2. When is RAG preferable? When the main issue is accessing current, attributable or permission-controlled information, while the model can already use retrieved evidence adequately.
3. What must accompany a saved adapter? Base revision, target modules/config, tokenizer/template, training configuration, data version and evaluation.

Primary references: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [PEFT documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). Practical task: compute parameter savings yourself before using a PEFT library.
