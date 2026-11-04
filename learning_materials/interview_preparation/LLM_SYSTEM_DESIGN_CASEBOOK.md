# LLM system-design interview scenarios

These are example answers to design discussions, not deployed systems. Start by clarifying the user's problem and constraints. The numbers below are illustrative calculations, not measurements from this laptop. A simple design with an explained trade-off is more useful than an unexplained collection of frameworks.

## D01 — Design a support assistant that uses the current policy

Suppose employees ask about support incidents, and policy limits change over time. The assistant must use the applicable version and show where its answer came from. I would first clarify who may access which incidents, whether the assistant only answers or also performs actions, and how quickly a policy update must appear.

I would start with permission-aware incident lookup and lexical search over versioned documents. The answer model receives selected evidence; the application validates source references and support. Current policies belong in externally managed evidence, not only in fine-tuned weights. Dense retrieval or reranking should address a measured baseline weakness rather than appear by default.

For a concrete failure, incident 104 concerns a 28 MiB attachment under policy version 2. Version 1 allowed 30 MiB; version 2 allows 20 MiB. An answer using the old limit is wrong for this case even if its citation is genuine. Missing version information should lead to clarification or an explicitly conditional answer.

Ten thousand 768-dimensional float32 vectors need 30.72 MB of raw values before text, metadata and index overhead. At two requests per second and eight seconds average response time, stable average in-flight work is 16 by Little's law; this does not prove a laptop can run 16 generations concurrently.

I would evaluate current-version correctness, supported answers, useful clarification, access violations and latency. The first release can be read-only. A good follow-up is what happens when an old source is withdrawn: index, cache and retained summaries all need a compatible policy.

## D02 — Design a human-approved action that can survive a restart

The assistant proposes a ticket, a person approves it, and execution may happen after an application restart. I would store the exact proposal, requester, required approval, expiry and resource version before acknowledging that the work is waiting. Approval must bind to the payload that will actually execute.

For a local database action, the execution transaction rechecks current authority and proposal state, then uses a unique execution key to prevent two concurrent attempts creating two tickets. “We check whether a ticket exists first” is insufficient if both attempts can pass the check before either writes.

A remote service changes the problem. The local transaction cannot atomically include an arbitrary remote API. Use a supported idempotency key and status lookup, record durable intent and reconcile an uncertain timeout. If the provider cannot establish whether an irreversible action completed, expose that uncertainty instead of blindly retrying.

I would test duplicate clicks, a changed proposal, expired approval, lost responses and crashes before and after dispatch. Success means the correct authorized final state without an unintended duplicate, not a prescribed sequence of tool calls. A workflow checkpoint helps resumption but is not an exactly-once guarantee for external effects.

## D03 — Serve a language model on limited local hardware

I would ask for actual available memory, supported runtimes, privacy constraints and representative prompt/output lengths before choosing a model. Nominal parameter storage is only a lower bound: 1.7 billion parameters at four bits represent 0.85 GB of raw bits, before scales, unquantized tensors, cache, temporary buffers and runtime overhead.

The design can begin with one compatible local runtime, bounded admission and explicit context/output caps. Queueing is preferable to launching unlimited simultaneous generations, but a full queue must have a defined rejection or timeout response. A cloud fallback is optional only if its cost and data policy are authorized.

I would measure first-token time, later token gaps, total latency, throughput, peak memory and task quality at the intended workload. Compare cold and warm runs explicitly. Native Metal/MLX support and CUDA on a Linux GPU are different execution profiles; a container does not automatically provide Metal acceleration.

If the workload fails, determine whether the bottleneck is memory, prompt processing, output generation or queueing. Quantization, batching, a smaller model or another replica can address different limits and must be checked for quality regressions. Neither a parameter-count estimate nor a cache formula is a measured speed claim.

## D04 — Plan a training study before committing substantial compute

The first question is whether we want to demonstrate a mechanism or produce a useful model. I would define the task, allowed corpus, evaluation split, reference model and success measure before selecting size. Unique training data and repeated token exposures are different quantities.

A rough dense-training estimate for five million parameters and one million token exposures is 6×5 million×1 million = 3×10^13 FLOPs. It omits important architecture/context details and does not predict elapsed time. A stated 16-byte-per-parameter state recipe would account for 80 MB before activations and other allocations; actual precision and optimizer choices can change it.

I would inspect a small batch, verify loss/masks and trainable parameters, overfit a tiny clean case, then run a bounded throughput pilot. Only that pilot can inform a realistic resource request. Record checkpoint/resumption behavior and selection rules before a longer run.

A lower validation text loss but worse support answers could indicate an objective mismatch, masking/template error or forgetting. The conclusion should follow the task evidence, not just the training curve. Distributed execution comes after a correct single-device reference and a reasoned state/communication budget.

## D05 — Keep an assistant usable during outages without exceeding its budget

Suppose a provider becomes slow during a traffic spike and there is a hard spending ceiling. I would clarify which features can degrade, what users should see, and which alternative backends are permitted. A visible unavailable response can be safer than an unauthorized fallback.

Use bounded queues, request deadlines, concurrency limits and a finite retry allowance for eligible failures. A circuit breaker temporarily avoids a failing dependency; a rate limiter controls traffic. They do not solve the same problem, and permission denials should not be retried as temporary model failures.

Concurrent requests need atomic budget reservation before dispatch, based on allowed input and maximum output cost under current approved rates. Settle actual usage afterward. A client cancellation does not guarantee upstream work or billing stopped, so release reservations under a documented reconciliation policy rather than assuming everything was free.

I would inspect queue age, latency distribution, errors, usage and supported task outcomes. Test slow responses, uncertain completion, throttling and failure during recovery with local fake services. If both primary and fallback depend on the same network or database, they share a failure risk; two backend names do not prove independent availability.

## D06 — Release a new model and index without leaking another team's data

A release must carry compatible code, model, tokenizer/template, index and policy versions. I would establish trusted artifact sources, permitted data use, identity and access rules before opening traffic. The first design question is which threats and mistakes the system must resist, not whether it can be called “secure.”

Server-derived identity should constrain retrieval and tools. Retrieved text, tool metadata and model output remain untrusted inputs. A malicious document asking for secrets must not expand network destinations, source access or execution authority. Trace data also needs redaction and retention rules.

Before release, test wrong-base adapters, mismatched tokenizers, stale indexes, cross-team searches and secret-like dummy values in logs. Keep a recoverable approved version and test restoration. A new code version may require a database migration; rolling it back may not undo that migration or an already completed external action.

If another team's content appears in search, contain the access path, preserve necessary evidence safely and review other exposure surfaces before reopening. Delete/restore actions alone do not prove that downstream summaries, caches or exported logs are clean. Add a regression test for the actual failure. Specialist security/privacy review is distinct from demonstrating a safe local prototype.
