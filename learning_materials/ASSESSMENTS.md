# Assessments, practical projects and oral defence

## How to use this file

Attempt before reading the checks. You may use a calculator and your own notes for projects, but not a generated full answer for the first attempt. Record assistance honestly. These are educational gates, not validated hiring tests or an accredited examination.

For each answer, score yourself 0–3: 0 = cannot explain; 1 = recognises words; 2 = explains and works a small example; 3 = independently applies it and states a limitation. A high total does not compensate for a critical gap in data leakage, authorization or the objective you actually trained.

## Gate A — foundations

1. Trace text from a terminal command through tokenizer, tensors and GPU. Separate environment, process, model state and hardware.
2. Multiply `[2,1]` by `[[1,3],[2,0]]` and state the shape.
3. Derive the gradient of `½(wx−y)²`; update `w=1,x=2,y=1` at rate 0.1.
4. Compute loss when the correct token has probability 0.25; explain perplexity if every included position has that loss.
5. Explain why confidence 0.9 on a token is not a 90% guarantee that a complete scientific claim is correct.
6. Plan a paired bootstrap comparing two systems on 500 questions from 50 documents. What is the resampling unit and why?
7. Give one example each of changed input distribution and changed target definition.
8. Explain why a simple lexical baseline can beat a dense retriever on error codes.

### Answer checks

2: `[4,6]`, shape `[1,2]`. 3: gradient 2, new weight 0.8. 4: loss about 1.3863 nats, perplexity 4. 6: document-level or otherwise dependence-aware resampling is a sensible starting point; keep system predictions paired and specify aggregation. A random question bootstrap may understate uncertainty if questions within documents are dependent.

Pass only when you can explain all eight without copying lesson wording. If algebra or code tracing remains difficult, review programming and mathematical foundations alongside these lessons.

## Gate B — model and training

1. Construct inputs/targets from `[BOS,A,B,C,EOS]`; explain where labels originate.
2. Draw the causal visibility mask for four tokens. Why can all four positions be trained in one pass?
3. Calculate the T3 attention example and identify the dimensions of `QKᵀ` and `AU`.
4. Compare MHA/GQA, dense/MoE and attention/state-space models without saying one is always superior.
5. Two microbatches contain 10/90 valid tokens with losses 2/4. Compute the token-average loss.
6. Estimate raw 7B FP16 weight storage, then explain why that is not training memory.
7. Explain DP, FSDP, TP and PP by drawing where weights, activations and optimizer states live.
8. You discover validation questions in training data. What conclusions must change, and how would you repair the evaluation?

### Answer checks

1: inputs `[BOS,A,B,C]`, targets `[A,B,C,EOS]`, subject to API shift convention. 2: token i sees positions up to i, not future tokens. 3: single query output approximately `[1.3395,1.3210]`; score matrix compares positions. 5: 3.8. 6: 14 GB decimal raw weights, plus the other states described in T6. 8: disclose contamination; create a genuinely independent evaluation or clearly limit claims; do not quietly delete known overlaps and declare universal cleanliness.

## Gate C — adaptation and systems

1. Choose prompt, RAG, SFT or tool use for: changing facts; fixed response format; sending a message; a new domain corpus. Explain combinations where needed.
2. Calculate LoRA rank-8 parameters for a `4096×4096` matrix. What memory is not removed?
3. Explain reward model, critic, reference and rollout policy as separate objects.
4. Calculate the PPO clipping example for positive and negative advantages in R2.
5. Explain DPO's preference margin and why equal GRPO rewards provide no relative signal.
6. Diagnose a RAG answer with a valid citation that does not support its claim.
7. Draw an MCP call with authentication, authorization, side effects, timeout and error handling.
8. Calculate the KV-cache example in R6. What if concurrency doubles?
9. Quantize 0.6 using the signed `[-7,7]` toy scale in R7.
10. State a production release gate and a rollback condition without treating a system prompt as the security control.

### Answer checks

2: 65,536 adapter parameters; base/activations/runtime still matter. 5: reference-relative winning minus losing sequence log probability, scaled by β; equal rewards centre to zero. 6: source existence is not entailment; inspect chunk scope, negation, version and answer construction. 8: 1 GiB; equal independent sequences at double concurrency roughly double that component. 9: integer 4, reconstructed 4/7≈0.5714.

## Project 1 — LLM Training Lab, beginning with a tiny model

This capstone connects model building, training, evaluation and inspection. Use the [16 practical assessments](interview_preparation/PRACTICAL_ASSESSMENTS.md) to practise the individual mechanisms.

**Goal:** demonstrate the training mechanism, not compete with a commercial model.

Use a small approved public-domain corpus. First build a character/unigram/bigram baseline. Then implement or carefully reconstruct a tiny causal decoder with a small vocabulary/context. Keep architecture and data small enough for your authorised machine. CPU execution is acceptable; use a reduced dataset if necessary.

Required artifacts:

- A data manifest, document split and tokenizer specification.
- An architecture drawing with shapes and parameter count.
- Tests for target shift, causal mask, padding and reproducible loading.
- A tiny-overfit test, training/validation curves and a baseline comparison.
- A short report separating measured results from expected limitations.
- An assistance log identifying code you wrote, adapted or generated and independently checked.

**Defence:** explain one backward path, one bug and one failed idea. Do not claim that memorising short text is general reasoning.

## Optional capstone — grounded document answers

This application exercise can start with a pretrained model or an extractive baseline. It does not require training a language model first and is not an additional implemented application in this repository.

Use a small collection of approved public documents. Build lexical retrieval first; add dense/hybrid retrieval only when you can explain the baseline. An optional generator can be replaced initially with an extractive answer that returns matched passages, so no paid inference is required.

Create questions covering exact identifiers, paraphrases, negation, missing evidence, conflicting versions and inaccessible documents. Keep development and final questions separate.

Required comparisons: no retrieval; lexical; dense/hybrid if available; optional oracle evidence. Record retrieval relevance, source support, answer completeness, abstention coverage, latency and memory. No invented scores: a blank result sheet is preferable to guessed success.

Required failures: one parser error, one missed relevant chunk, one misleading near-neighbour, one permission denial and one source update. Explain the repair at the correct layer.

## Application exercise — a bounded tool workflow

Build a read-only local workflow: retrieve a document and calculate a simple statistic. Define typed inputs/outputs and a permission allowlist. Tools must reject unknown file paths/operations. A fake/test tool can demonstrate timeouts safely.

Compare a deterministic workflow with a model-selected tool sequence if a local model is available. Do not add multi-agent debate unless there is a measurable question it addresses. MCP is optional for the first implementation; demonstrate its host/client/server boundary in the design.

Required tests: malformed input, empty results, timeout, repeated call, untrusted instruction inside a document and an action beyond the allowlist. No email, payment or destructive filesystem integration is needed.

## Shared deployment assessment — decision, not a benchmark screenshot

Use model experiments to evaluate training trade-offs and a document-answering exercise to evaluate complete request/service trade-offs. The exercises can share data and evaluation cases without becoming four separate applications.

Choose an intended machine and workload. Compute weight/cache lower bounds, then measure actual memory if you can run it. Compare two safe configurations, such as context limits or supported precisions, under the same task set. Record cold start, sustained performance, quality and failures.

If you lack hardware, submit a design with clearly labelled estimates and an executable arithmetic check. Never present estimated token/s as a measurement.

## Six real-world oral scenarios

### “The fine-tuned model is worse than the base.”

Check template, target masks, data quality, learning rate/epochs, base/adapter compatibility, evaluation changes and forgetting. Compare a correctly prompted base under the same input/budget. Do not immediately increase rank.

### “The RAG system invents facts even with sources.”

Trace ingestion → retrieval → selection → prompt → generation → support check. Distinguish missing evidence from misuse of present evidence. A citation validator should verify source IDs and claim support, not only URL syntax.

### “We need 10× more traffic.”

Measure the bottleneck and quality/latency budget first. Consider batching, caps, caching, replicas, quantization or routing. Include queueing and tail latency; more concurrency can exhaust KV memory. Do not promise linear GPU scaling.

### “We want the assistant to act autonomously.”

Define allowed actions and consequences, then authorization, human confirmation, idempotency, budgets, auditability and rollback/compensation. Model capability does not confer authority.

### “The reward rose, but users complain.”

Suspect reward mismatch/hacking, verbosity, sycophancy, changed population or a poor proxy. Inspect independent quality and failure cases. Fixing the reward is a hypothesis to test, not automatic proof that RL is wrong.

### “Can we deploy offline on a phone?”

Ask model/task size, worst-case memory/cache, supported kernels, sustained thermals, battery, update mechanism, privacy and network fallbacks. Test the weakest supported device; a laptop demo is not mobile validation.

## Final capstone marking guide

| Dimension | What strong evidence looks like |
| --- | --- |
| Mechanism | You trace each step and its inputs/outputs |
| Mathematics | Shapes, denominators, units and assumptions are correct |
| Code ownership | You can change/debug a small component unaided |
| Evaluation | Independent cases, matched comparisons, honest uncertainty |
| Reliability | Failure handling, permissions, provenance and rollback |
| Communication | Clear explanations with no inflated claims |

Finish by writing a one-page “what I can do independently / with documentation / need supervision for” inventory. That inventory, not the course title, should determine what enters your CV.
