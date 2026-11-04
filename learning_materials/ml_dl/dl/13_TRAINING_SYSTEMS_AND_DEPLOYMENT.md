# DL13. The neural training loop and its system state

Training is a repeated procedure: fetch a batch, calculate predictions and loss, compute gradients and update trainable parameters. The data loader, device, optimizer state and checkpoint rules are part of that procedure, not extra neural-network layers. Validation checks a model without fitting on those cases; saving state makes selected results or later resumption possible.

We have 32 examples with 16 float32 features each. Change batch size to see input-memory bytes and updates per epoch under one-update-per-batch training. These calculations exclude other memory costs and do not launch a job.

The earlier lessons explain individual model operations. Here we connect them to data movement, training/evaluation modes, gradient clearing, periodic validation and recoverable state.

[course map](../COURSE_MAP.md). Required background: DL04, DL05.

## 1. Training state, devices and data movement

**Key terms.** **Parameters** are learned state; **gradients** hold derivatives; **optimizer state** stores update history; **activations** are forward intermediates. A **checkpoint** may contain only weights or a complete resumable state; these are different contracts.

**Process.** Load permitted data and a configuration, allocate compatible tensors, run bounded forward/backward/update steps and record actual progress. Validate separately, save complete state if resuming is supported, then measure inference without training.

Dataset → DataLoader → tensor batch → model → loss → backward → optimizer is the core loop. A scheduler has declared step units. Gradient accumulation combines microbatches before an update and needs the correct denominator; clipping occurs at the specified update boundary. Train/eval modes and gradient-recording modes solve different problems.

CPU is the correctness reference; Mac MPS is a possible accelerator, not CUDA. Unsupported operations or mixed-precision paths need explicit handling, not silent fallback that invalidates timing. DataLoader workers, thread counts and device transfers can dominate small models. Synchronize correctly when timing asynchronous work, and distinguish startup from steady-state latency.

## 2. A worked calculation

**Symbols and assumptions.** N is parameter count and b bytes per value. Raw parameter storage is N×b, before all other states. Effective batch is microbatch size times accumulation count under the declared equal-size single-device convention.

One million FP32 parameters occupy 4,000,000 bytes, about 3.81 MiB. FP32 parameters plus gradients plus two FP32 Adam moments alone occupy about 16,000,000 bytes, before activations, buffers and overhead; other precision recipes differ. Microbatch 16 with accumulation 4 supplies 64 examples per update only if all four batches contain 16 included examples.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Batching, throughput and deployment metadata

More batching can improve throughput and increase memory while changing optimization. A small model can run faster on CPU than GPU when transfer/launch costs dominate. Do not make speed claims before a representative bounded pilot. Deployment also needs feature/token mappings, preprocessing and task metadata.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: save enough state to resume

Use existing app logging concepts, but keep the new ML/DL artifacts in their own versioned task namespace. A classical estimator's final fit and a neural optimizer update must not be given fabricated identical progress curves.

**Check.** Why are weights alone insufficient to reproduce the next training update?

**Answer.** The optimizer, scheduler, random generators, sampler/data position and compatible configuration may all affect it. Save/restore the complete state and test continuation under declared tolerances.

## 5. Microbatch weighting and memory accounting

**Key terms.** A microbatch is one forward/backward calculation. An optimizer update can accumulate several such calculations. Effective batch size counts examples contributing before that update.

**Process.** Zero gradients, accumulate correctly weighted losses over microbatches, clip once after accumulation and then update once. If using mixed precision, unscale gradients before clipping.

For microbatch target counts 12 and 4, loss means L₁ and L₂ combine as (12L₁+4L₂)/16. Simply averaging the two means gives the smaller microbatch disproportionate influence. For equal-sized classification batches, division by the number of accumulated microbatches is equivalent; the final partial group needs its actual denominator.

### Memory has several owners

Weights, gradients, optimizer moments and saved activations are different allocations. For P float32 parameters, weights alone use 4P bytes; gradients add about 4P, and two Adam moments add about 8P, before activations, temporary buffers and allocator overhead. Freezing parameters removes their gradient/moment needs but not necessarily all activations used by trainable downstream layers.

The current image lab runs one batch per update in float32 with bounded width and rows. It does not pretend to implement accumulation, mixed precision, distributed execution or compiled kernels. The LLM scratch path separately exposes accumulation. Record actual hardware and elapsed time rather than extrapolating a theoretical FLOP estimate into a speed claim.

## 6. Export a prediction pipeline, not an isolated weight file

A deployable prediction pipeline includes everything needed to turn an allowed input into the intended output: parsing, feature order, resizing/scaling, vocabularies or category maps, model parameters, output interpretation and any decision threshold. The serving code must use compatible versions and the same relevant conventions as evaluation.

First save a small non-sensitive set of input/output fixtures from the approved pipeline. Export the declared artifact, load it in an isolated supported runtime, and compare outputs within a chosen tolerance. Record artifact hash, data/model revision, dependencies, input schema, expected units and supported hardware. A hash detects changed bytes but does not authenticate an untrusted publisher by itself.

Training mode and inference mode remain separate: disable unwanted dropout and state updates, and choose appropriate derivative-recording settings. A model export format such as ONNX can support another runtime but does not guarantee that every operator, dynamic shape or preprocessing step transfers correctly. Loading arbitrary pickle/joblib files can execute code; accept only trusted artifacts under an explicit policy, not anonymous uploads.

Measure an end-to-end boundary as well as model-only time. If parsing/preprocessing takes 8 ms, model computation 12 ms and postprocessing 5 ms, serial service work totals 25 ms before network and queueing. That is not automatically 40 requests/second under concurrent traffic: batching, overlap, hardware contention and tail effects must be measured. Report input sizes, batch, precision, warm-up, synchronization and measurement distribution.

**Practice.** Exported weights give the same tensor outputs on a synthetic input, but live predictions change. What else should be inspected? **Answer.** Feature ordering, normalization, token/label maps, resize alignment, thresholds, mode and runtime precision. Test the full raw-input path before blaming the model architecture.

The next section handles changes after deployment. Existing LLM serving lessons add prefill/decoding/KV specifics; they do not replace these general pipeline checks. [scikit-learn persistence/security](https://scikit-learn.org/stable/model_persistence.html) · [PyTorch export documentation](https://docs.pytorch.org/docs/stable/export.html).

## 7. Monitor data, predictions and outcomes; recover deliberately

Monitoring asks whether the deployed system still meets its intended contract. **Data drift** is a change in input distribution; **prediction drift** is a change in outputs; **performance degradation** concerns quality against outcomes. A shift in an input histogram does not by itself prove failure, and unchanged inputs do not prove the target relationship stayed stable.

Log permitted aggregate distributions, schema failures, latency/errors, confidence or score distributions and outcomes when available. Real labels may arrive late or only for selected cases. Preserve that selection context and group-level counts; do not report apparent accuracy from a convenient unrepresentative slice as the whole service's performance.

If 10 of 1,000 baseline requests fail but 40 of 1,000 later requests fail, the failure rate rose from 1% to 4%. That is +3 percentage points and a fourfold rate, not +3%. Explain denominators and time windows. Investigate request mix, data sources, dependencies and resource saturation before automatically retraining. Drift alarms need validation and accountable response rules to avoid alert fatigue.

A **shadow** deployment compares a candidate on copied permitted inputs without affecting decisions; ensure it does not duplicate external side effects. A **canary** routes a limited portion of eligible live traffic to a candidate under a defined allocation policy. Neither is automatically a randomized quality experiment. Maintain an approved prior version, compatible data/schema state and a rollback path; replacing the model file alone may not undo downstream effects.

**Practice.** When an alarm fires, should the pipeline immediately fit on all new requests? **Answer.** No. First validate labels, privacy/consent, source changes and evaluation boundaries. Quarantine suspect data, stop unsafe actions where appropriate, investigate and re-run the agreed regression checks before approving a new version.

These are system-design exercises, not a production service running behind this local application. Use ML16's risk/reporting criteria and R8 for the additional risks of LLM-generated instructions and tool actions. [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).

## Sources

- [PyTorch training workflow](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
- [CS231n practical neural-network training notes](https://cs231n.github.io/neural-networks-3/).
