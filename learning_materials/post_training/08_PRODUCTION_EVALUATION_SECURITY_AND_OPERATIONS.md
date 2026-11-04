# R8. LLM system evaluation, security and operations

A deployed LLM system must satisfy task-quality, access, reliability and resource requirements across its components, not just produce fluent model outputs. Release evaluation, authorization, budget enforcement, monitoring and recovery address different failure paths. They are coordinated controls rather than one model-forward chain or a certificate of complete safety.

The budget control illustrates how many whole 0.2-unit operations fit when spending starts at zero. It is not a current API price. Other sections discuss permission checks and staged releases as separate operational decisions, with no traffic or external effects executed.

Generation, retrieval and tools introduced different sources of failure. This lesson joins their evaluation and operating boundaries while preserving independent enforcement outside the model and honest reporting of incomplete tasks.

## 1. End-to-end system quality

**Key terms.** **System quality** includes model, retrieval, tools and delivery; **reliability** means meeting declared requirements under expected conditions.

**Process.** Define a successful user task, trace its contributing components and measure the full path. A model-only benchmark cannot certify the surrounding system.

A correct model can fail because retrieval used the wrong tenant, a parser dropped a negation, an API timed out or a user interpreted a suggestion as advice. Start with users, allowed uses, consequences of error and a human escalation path. The research assistant example is not automatically suitable for medical decisions merely because it reads biomedical papers.

## 2. Release criteria

**Key terms.** A **release specification** lists acceptance limits; a **regression** worsens accepted behaviour; **rollback** restores a prior version.

**Process.** Set quality, latency, cost and safety thresholds, test a fixed package and deploy only if it passes the declared checks.

Define measurable requirements before deployment:

- Correctness and evidence support on representative tasks.
- Coverage/completeness, appropriate refusal and usability.
- Worst-case slices: languages, long inputs, rare entities, ambiguous labels.
- Security/privacy requirements and prohibited actions.
- Latency percentiles, capacity, availability and cost ceilings.
- Ownership, monitoring, escalation and rollback.

Set numerical thresholds from application needs and evidence. This course does not invent a universal “95% means safe” rule. Document who accepts residual risk.

## 3. Offline and online evaluation

**Key terms.** **Offline evaluation** uses fixed cases; **online evaluation** observes usage; a **canary** limits exposure to a new version.

**Process.** Test reproducible cases, run a controlled limited rollout and monitor outcomes. Compare with the previous version before expanding exposure.

Use fixed regression sets, independent held-out cases, adversarial probes and human-reviewed samples. Separate retrieval failures from generation failures. Test model, prompt, retriever, parser and database changes independently where possible, then evaluate the combined system.

Online A/B testing compares assigned treatments under ethical and statistical constraints. Shadow testing processes copies without affecting decisions, subject to privacy rules. Canary releases expose a small permitted fraction before broad deployment. User clicks/thumbs-up measure behaviour or satisfaction, not necessarily truth. Logging only successful requests biases evaluation.

Track base rates: a rare high-impact error may not appear in a small random sample. Test known critical cases deliberately and describe that sampling. Human evaluation needs clear instructions and agreement checks; domain experts may be needed for factual interpretation.

## 4. Threat model

**Key terms.** A **threat model** states attackers/assets/capabilities; **prompt injection** brings instructions through untrusted input; an **approval gate** controls consequential actions.

**Process.** Identify protected actions and untrusted entry points, enforce authority in code and test boundary crossings. Turn each exploit into a regression case before release.

```text
assets: private data, credentials, money, integrity and availability
attack surfaces: user inputs, retrieved pages, tools, models, dependencies, logs
controls: least privilege, validation, isolation, provenance, monitoring, humans
residual risk: what remains possible despite these controls?
```

**Prompt injection** places adversarial instructions in inputs or data; **jailbreaking** seeks disallowed model behaviour. A system prompt is not a dependable security sandbox. **Data poisoning** changes training/retrieval data; **backdoors** cause targeted hidden behaviour. **Supply-chain attacks** compromise packages, model artifacts or servers. **Exfiltration** leaks information through answers, tools or network calls.

Protect secrets outside prompts, apply real authorization at tools/data services, restrict network/file access, sandbox untrusted execution and validate outputs before using them in SQL, shell, HTML or downstream actions. Never execute model-generated code automatically just because it appears in a fenced block.

Multi-agent systems add shared-memory and communication attacks. Tool descriptions can be poisoned. Caches can leak data across users. Logging full prompts can create a sensitive database even if the model provider stores nothing.

### Turn one failure into a release test

Suppose a retrieved document contains an instruction to send private text to an external address. The important outcome is whether protected information or an unauthorized action escapes, not whether the answer contains a reassuring safety phrase.

1. Define the protected asset, attacker-controlled input and prohibited action. Keep the test fixture synthetic rather than using a real secret.
2. Test the trust boundary: retrieved text is evidence, not authority. The tool service should deny the action independently of the model's willingness.
3. Assert that no external side effect occurred. An “attack detected” log after the send is not prevention.
4. Check permitted normal use too. A change that blocks every tool call may prevent this attack while breaking the product.
5. Add the case to a versioned regression set, along with logging-redaction and cross-tenant cache tests. Re-run it after changing the model, prompt, retriever or tool integration.

For online measurements, record the experimental unit and assignment: requests from one user can be correlated, and shadow traffic must not duplicate write actions. Report refused/failed/timed-out cases in their declared denominators. Engineering frameworks: [OWASP LLM risks](https://owasp.org/www-project-top-10-for-large-language-model-applications/) and [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework). These organize risk analysis; passing a small regression suite does not certify complete security.

## 5. Privacy and governance

**Key terms.** **PII** is personally identifying information; **retention** controls storage duration; **governance** assigns rules and accountability.

**Process.** Determine needed data and authority, then minimize collection. Enforce access and deletion across logs, indexes and backups rather than only the visible interface.

Data minimization means collecting/retaining only what is needed. Encryption protects data in transit/at rest under its assumptions; it does not stop an authorised application from misusing plaintext. Pseudonymization changes identifiers; it does not necessarily prevent re-identification. Differential privacy is a mathematical property with a specified unit and privacy budget, not a synonym for hiding names.

Membership inference asks whether data was included; extraction tries to recover content. An absence of known successful attacks is not proof of privacy. Federated learning can keep raw data distributed while updates still leak information; privacy protection requires additional design.

For regulated/high-stakes settings, obtain current domain/legal review. This is engineering education, not legal or medical advice. Record model/data licenses, provider retention/region policies, intended use and impact assessments where relevant. Recheck rules when deploying; they change.

## 6. Metrics, logs and traces

**Key terms.** A **metric** aggregates quantities; a **log** records events; a **trace** connects a request's steps; **redaction** removes sensitive content.

**Process.** Instrument components, correlate request events and record only needed fields. Diagnose failures without turning debugging into unrestricted data collection.

A useful request trace records model/prompt/retriever versions, timings, token counts, error categories, evidence IDs and policy decisions. Redact sensitive content. Use access-controlled logs and retention/deletion policies. Logs, traces and metrics should let you reconstruct a failure without unnecessarily copying every private document.

Monitor task success, refusal/coverage, unsupported claims, drift, cost, queue length, memory, timeout/retry rate and tool failures. Distribution drift concerns changed inputs; concept/policy drift concerns changed meaning of a correct output. Automatically retraining on feedback without auditing it invites poisoning and bias.

## 7. Reliability mechanisms

**Key terms.** A **timeout** bounds waiting; a **retry** repeats work; a **circuit breaker** pauses a failing dependency; **idempotency** avoids duplicate effects.

**Process.** Detect failure, determine whether retry is safe and apply bounded recovery. Escalate or degrade gracefully when the budget ends, preserving confirmed state.

Timeouts bound waiting; circuit breakers stop repeated calls to a failing dependency; backpressure limits load; fallbacks degrade gracefully. A fallback must preserve permissions and critical safety constraints. A cheaper model is not a safe fallback merely because it responds quickly.

Use immutable versioned artifacts and rollback. An incident response needs detection, containment, impact assessment, communication, repair and a regression test. If a model leaked private data, simply tweaking the prompt is not a complete response.

## 8. Cost accounting

**Key terms.** **Token cost** uses priced input/output counts; **request cost** also includes retrieval/tools/hosting; a **budget guard** caps spend.

**Process.** Record component usage, aggregate complete request cost and enforce limits before dispatching more work. Report cost together with successful-task quality.

**Calculation.** With illustrative prices of USD 0.20 per million input tokens and USD 0.80 per million output tokens, 2,000 input plus 500 output tokens cost USD 0.0008 for the model call. Retrieval, retries, tool use and hosting add to that total. These prices demonstrate units, not a current provider quote.

Illustrative per-request model cost is `input_tokens×input_rate + output_tokens×output_rate`, under compatible units, plus retrieval/reranking/tools/storage/observability and infrastructure. Price per token differs from total cost per successfully completed task. Cache discounts, minimum billing and provisioned capacity depend on the provider; do not hardcode current rates into reasoning.

Rate limits, output caps, bounded agent loops and budget-aware routing control runaway cost. Evaluate whether truncation/fallback reduces task success. Cost optimization that makes the assistant unusable is not a win.

## Exercises and checks

1. A prompt injection is detected after a tool sent an email. Was detection sufficient? **No:** the consequential action had already occurred; prevention/authorization and incident handling are needed.
2. User satisfaction rises while source support falls. What might explain it? Verbosity/confidence or agreement with the user; test these rather than celebrating one metric.
3. Define a release gate and rollback trigger for an abstract assistant, including a no-answer case and a permission-restricted document.

Guidance: [OWASP LLM application risks](https://owasp.org/www-project-top-10-for-large-language-model-applications/), [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), [MCP security](https://modelcontextprotocol.io/specification/latest/basic/security_best_practices). These are frameworks for identifying/handling risks, not certificates that a system is safe.
