# Retrieval and LLM systems interview questions

Try a short answer to the situation, then use the explanation and follow-up to check your reasoning. These are examples, not scripts or claims about your experience. Full mechanisms and calculations remain in the course; describe only work you actually did.

## S01 — An employee asks about an error code. How does a retrieval assistant answer?

Documents first need permission review, parsing and versioned source spans. When an employee asks about an error code, the server identifies the allowed sources, searches them, selects relevant passages within the context budget and supplies them to the answer model. The response needs citations to accessible passages that actually support its claims. Retrieved content is evidence, not authority to change tool permissions.

Retrieval-augmented generation (RAG) can improve access to current evidence; it does not guarantee the retriever finds the right source or that the generator uses it correctly. Separate failures at each stage. A perfect answer given oracle evidence can help diagnose a retrieval bottleneck, but oracle performance is not deployable performance.

**Follow-up example.** If manual oracle evidence fixes the answer, investigate parsing/retrieval before retraining generation. Oracle quality diagnoses a bottleneck; it is not deployable performance.

## S02 — When would you use lexical search, vector search or reranking?

Lexical retrieval matches words/identifiers with weighting; it is strong on exact error codes. Dense retrieval compares learned vector representations and may match paraphrases. Hybrid combines complementary scores/ranks using an explicit rule rather than adding incomparable raw numbers blindly. A cross-encoder reranks a smaller candidate list by reading query/document together, at extra latency.

An ANN index trades resource use and approximate vector-neighbour recall; high vector recall is not proof of task relevance. Normalization and cosine/dot-product/Euclidean choices must match the embedding model and index settings.

**Follow-up example.** Fuse scores or ranks using an explicit validated rule. Raw scores can have incompatible scales; vector-neighbor recall is not the same as relevance to a user's task.

## S03 — A policy exception is split from its rule. How would you fix chunks and citations?

Preserve semantic units and important caveats: a condition/negation/table header separated from its value can invert meaning. Small chunks improve specificity but can lose context; large chunks add noise/cost. Overlap helps some boundaries but can multiply near-duplicates. Choose by labelled task outcomes, not a universal token count.

Store immutable source version and exact span; validate citation references against sources the caller may access. A real source ID can still be attached to a claim it does not support. A source update must affect index selection, caches and retained summaries—not just the raw file.

**Follow-up example.** A genuine source ID is insufficient if the cited span does not support the claim. Preserve applicable version, surrounding qualifications and the caller's access rights.

## S04 — A user asks for another team's private document. Where must access be checked?

A vector store supplies similarity search and metadata/index management. A relational store supplies durable structured state and constraints; some systems also support vectors. Start with the simplest adequate deployment. You do not need several databases to demonstrate breadth.

Server-derived identity determines source scope. Metadata filters help enforce that policy but are not a replacement for authentication or authorization. Never let the model choose tenant IDs or arbitrary search filters. Unauthorized content must not leak through hits, previews, counts, cache entries, traces or judge requests.

**Follow-up example.** The server's authenticated identity defines scope. A vector database or metadata filter alone does not secure results, counts, previews, caches and traces.

## S05 — A request exceeds its context window. What would you change?

Weights store learned parameters; context is the current serialized input; retrieval brings external evidence; persistent memory is application-owned stored state. A growing chat transcript is not unlimited reliable memory. Count system instructions, messages, evidence, tools, special tokens and reserved output against the actual model/template limit.

For example, a 4,096-token budget minus 600 for system/tools, 300 for the user, 400 for history, 800 reserved for output and a 200-token margin leaves 1,796 for evidence. Treat it as a planning calculation; tokenize the actual serialized request. Removing evidence to fit may change answerability, so ask/abstain when needed.

**Follow-up example.** Summarizing history can lose conditions or preserve stale/private text. Keep provenance and retention rules; clarify or abstain if necessary evidence cannot fit.

## S06 — When would you choose a fixed workflow rather than an agent?

A fixed workflow follows code-defined steps; an agent lets a model choose some next actions based on state. Use dynamic choice only where it solves meaningful branching; it adds error, cost and policy risk. Bound allowed tools, steps, deadlines and spend; define terminal states and escalation.

Multiple agents may provide complementary checks, or merely repeat correlated errors. Compare against one agent/fixed workflow at a fair total budget. Do not judge only whether the tool order matches one script: enforce required safety ordering and actual outcomes while permitting valid alternatives.

**Follow-up example.** Multiple agents can repeat correlated errors at extra cost. Compare outcomes and required safety ordering at a matched total budget, allowing valid alternative tool paths.

## S07 — An approved action retries after a timeout. How do you avoid a duplicate?

Persist a proposal's exact payload/hash, requester, required approver, expiry and resource versions. Authenticate/authorize the reviewer independently. At execution recheck the approval and current state; edited/expired/stale proposals need review again. Use a unique execution key and database transaction for the local side effect.

If the network times out after a remote write, you may not know whether it succeeded. Use provider-supported idempotency/status lookup or reconciliation before retrying; otherwise expose an uncertain outcome. A graph checkpoint alone does not guarantee exactly-once external action. An outbox records intended work with the local transaction, but consumers still need duplicate-safe handling.

**Follow-up example.** If its payload or resource state changed after approval, obtain approval again. A durable workflow checkpoint does not guarantee exactly-once remote effects.

## S08 — A retrieved runbook asks the assistant to send credentials. What prevents it?

A tool schema defines inputs/outputs; the backend enforces allowed functions and resource scope. MCP provides a protocol boundary for a host/client to discover/use server capabilities. It does not make a server trustworthy or grant its tools authority. Pin/review the actual protocol/SDK revision and discovered tool definitions.

Prompt injection occurs when untrusted text tries to redirect system behaviour. Treat retrieved/tool text as data, enforce least privilege outside the model, isolate secrets and constrain outbound destinations. Valid JSON or a regex that removes suspicious phrases cannot prove safety. SSRF occurs when server-side requests are manipulated toward unintended/internal destinations; validate hosts/schemes/redirects and network policy.

**Follow-up example.** Schemas constrain shape; MCP supplies a protocol boundary. Neither authorizes an action. Backend scope, destination and secret-handling controls must reject the harmful request.

## S09 — A remembered preference is stale. How should persistent memory work?

Stored summaries/preferences can be stale, wrong, sensitive or scoped to the wrong person/tenant. Give memory a purpose, ownership, provenance, expiry/deletion rule and version dependencies. Separate session state from durable user preference and approved knowledge. Revalidate before using a remembered authorization or current fact; do not treat text memory as permission.

Retrieval caches, prefix/KV caches and answer caches are different objects with different invalidation/security constraints. Cross-session reuse must not expose private content or silently ignore source withdrawal.

**Follow-up example.** Application memory, caches and model weights are separate stores. Deleting a vector cannot promise erasure of weight memorization; explain what deletion and retention actually cover.

## S10 — How would you evaluate an assistant and its LLM judge?

Use known fixture outcomes for facts/actions and an explicit human rubric for semantic support/usefulness. Track supported resolution, completeness, clarification/abstention coverage, forbidden actions, duplicate effects and latency/cost separately. A generator grading its own outputs is not independent evidence.

An LLM judge is optional: blind model identity, calibrate against reviewed cases, examine disagreement, order/verbosity bias and prompt injection in evaluated text. Keep development tuning distinct from final cases. A synthetic teacher answer is not gold merely because it sounds fluent.

**Follow-up example.** A correct clarification can be preferable to a confident unsupported answer. Review judge false positives/negatives and order, verbosity and injection biases against an explicit rubric.

## S11 — p95 latency doubled but average tokens per second did not. What would you inspect?

p95 latency is the latency at or below which 95% of the measured requests fall under a stated window/workload. A few slow requests can worsen it even when average throughput is stable. Measure queueing, retrieval, prefill, decode and tool time separately. Bound admission/concurrency and propagate deadlines. Use backpressure/load shedding rather than infinite retries. A timeout/cancellation may leave work or billed usage in flight, so reserve worst-case allowed cost before dispatch and settle actual usage afterward; atomic reservations prevent races.

Routing to a smaller/local model can reduce some cost but may harm quality; escalation/fallback needs evidence, data-region permission and remaining budget. A provider-compatible API can differ in templates, capabilities, rates and usage fields. Do not silently switch to a paid or privacy-incompatible backend.

**Follow-up example.** Queueing, long outputs, retries and tools can worsen tail latency while average throughput is stable. A cheaper fallback must also satisfy quality, data policy and remaining budget.

## S12 — A new index produces wrong answers after release. How would you recover?

Logs record events, metrics aggregate measurements, traces connect request steps. Capture relevant versions, latency, errors, queue saturation and usage without exporting secrets/private source text by default. Quality drift needs labelled review/evaluation, not just uptime. Define a service-level indicator/objective with a population/window; “99% reliable” without a denominator is incomplete.

Release compatible code/model/template/index/policy versions together when required. Keep a tested fallback; rehearse restore from backup on synthetic data. Rolling back code may not undo database migrations, external side effects or a poisoned source—plan those separately. Local Compose or Kubernetes exercise is not proof of multi-region high availability.

**Follow-up example.** Rolling back application code may not restore an old index, reverse a migration or undo an external action. Recover a compatible approved state and test the repair.
