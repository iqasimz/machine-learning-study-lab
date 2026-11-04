# Practical interview exercises

These are small interview-style tasks with expected reasoning, not tests already passed or software already deployed. Attempt the task before reading the answer. Use disposable examples; no exercise requires paid APIs, cluster access or a real customer's data. Record honestly which parts you completed unaided and which used assistance.

## P01 — Return a permitted record without mutating the input

Implement lookup(records, record_id, caller_team). The caller's team comes from trusted authentication, not a field supplied by the language model. For a record with id=t1, team=a and status=open, caller a may receive id and status. A caller outside the allowed team must not receive private fields or confirmation that distinguishes a forbidden record from an unknown one under this policy.

A good solution separates lookup from authorization and returns only allowed output fields. Decide how malformed and duplicate records are handled rather than silently selecting an arbitrary duplicate. Test that changing the returned object cannot unexpectedly mutate the input record. Dictionary lookup can be efficient, but building the index has a cost and duplicate policy.

A useful follow-up is adding a new optional field. Explain whether the function should expose it automatically; an explicit output contract normally should not.

## P02 — Rank tied results and stay within a context budget

Given scored IDs [(b,0.8),(a,0.8),(c,0.2)], return the top two using score descending and ID ascending for ties. The expected IDs are [a,b]. State the behavior for k=0, k larger than the list and duplicate IDs.

A full sort is straightforward; a size-k heap can reduce asymptotic work for large n and small k. Either solution needs deterministic final ordering. Ranking is not authorization, so candidates must already be within the allowed scope.

For a separate context-allocation task, a total budget of 4,096 minus 600 system/tool, 300 user, 400 history, 800 output reserve and 200 safety margin leaves 1,796 evidence tokens. Reject an impossible negative budget instead of dropping safeguards. Verify the real serialized token count later; the arithmetic is a plan, not exact tokenizer output.

## P03 — Explain and implement a stable softmax

For logits [1000,1001], subtract the maximum before exponentiating. The result is proportional to [exp(-1),1], giving probabilities approximately [0.2689,0.7311]. Direct exponentiation can overflow without changing the underlying intended probabilities.

Explain why subtracting a common constant cancels in the normalization. Test equal logits and very large magnitudes. State the policy for invalid inputs rather than silently producing NaN. Temperature scales logits before normalization; a greedy temperature-zero path should choose a maximum directly instead of dividing by zero.

A follow-up is deterministic tie-breaking or seeded sampling. Numerical stability and a reproducible random sequence do not make a generated answer factual.

## P04 — Find a future-token leak in a small decoder

Use a small supplied attention block in evaluation mode. Feed [A,B,C], change only input C to another token, and compare outputs at positions A and B. Under the intended causal mask, those earlier logits should not change beyond declared numerical tolerance.

If they change, inspect mask orientation, the softmax axis and any operation mixing future positions. Check positions and batch axes before blaming the tokenizer. Training-mode dropout can make this diagnostic noisy, so control stochastic behavior.

A pretrained wrapper can help inspect a model but is not evidence that you implemented a decoder from scratch. If asked to implement one, identify embeddings, positions, Q/K/V, masking, residual paths and the output head explicitly, using autograd where allowed.

## P05 — Calculate the correct loss for unequal microbatches

One microbatch contains 10 scored targets with mean loss 2; another has 90 with mean loss 4. The correct combined token mean is (10×2+90×4)/100=3.8, not the unweighted average 3.

Explain which tokens are scored before implementing the calculation. Padding and excluded user-prompt targets do not belong in that denominator. With input [A,B,C] and next-token labels [B,C,D], ensure the library does not apply a second unwanted shift.

A good gradient test compares weighted microbatch contributions against one concatenated deterministic batch, then performs one update. Define what happens when there are no valid targets; do not divide by zero. Correct weighting alone does not make stochastic or batch-normalized executions identical.

## P06 — Diagnose a training failure and restore the right state

Suppose loss becomes nonfinite after a checkpoint, and restarting from its weights changes the next update. Explain the two separate investigations.

For the failure, locate the first nonfinite input, activation, loss or gradient and inspect the relevant mask, operation, dtype and update scale. A small clean batch helps distinguish a broken implementation from a difficult dataset.

For resumption, weights are not enough: optimizer/scheduler, applicable scaler, random states and data/sampler position can affect the next update. Compare uninterrupted and restored runs on a small controlled case. Reject a partial save and document numerical tolerances and version compatibility. Do not promise exact replay across arbitrary hardware changes.

## P07 — Compare two prediction lists without overstating the gain

Gold labels are [1,0,0,1]. Model A predicts [1,0,1,0]; B predicts [1,0,0,0]. A gets 2/4 correct and B gets 3/4, an accuracy gain of 0.25. These four cases are a teaching fixture, not strong evidence of general superiority.

A paired bootstrap resamples case indices and carries both predictions for each chosen case together. If several cases share a source, use an appropriate group-level unit. This resamples fixed predictions; it does not include retraining variation.

For a multiple-testing follow-up, sorted p-values 0.01,0.03,0.04 with alpha=0.05 face Holm thresholds 0.05/3,0.05/2,0.05. The first passes and the second fails, so stop; do not independently accept the third. Corrections do not repair leaked or repeatedly tuned evaluation data.

## P08 — Diagnose a stale citation instead of changing the prompt blindly

A request concerns an attachment of 28 MiB under policy version 2. An answer cites a genuine version-1 rule allowing 30 MiB, but version 2 allows 20 MiB. The source exists, yet the answer uses the wrong applicable rule.

Trace where the mismatch arose: source metadata/parsing, candidate filtering, retrieval, evidence selection or generation. A correct response should identify that 28 exceeds the applicable limit, or clarify the version if it is unknown. It should not perform an external action merely because it found a source.

Build a small relevance/support fixture with exact codes, paraphrases, exceptions and withdrawn versions. An extractive baseline can reveal whether an LLM adds value. A real citation ID is not proof that its span supports the claim.

## P09 — Handle a slow fake API without an infinite retry loop

A local fake service produces one timeout, then a rate-limit response, then success. Another endpoint returns a permission denial. Describe a finite retry policy that respects deadlines and does not treat these failures identically.

Bound concurrency and total attempts. Honor relevant retry guidance within the remaining deadline; use backoff/jitter where suitable. A permission denial needs a policy or credential correction, not fallback to a more permissive backend. Validate response shape before passing it onward.

If a request could write remotely, a timeout may conceal successful completion. Use supported idempotency/status handling or surface uncertainty. Cancelling the waiting caller does not prove upstream work stopped. No paid provider is needed to demonstrate this reasoning.

## P10 — Prevent two concurrent local executions

Two workers both read an approved proposal as unexecuted before either writes a ticket. A check-then-write approach can create duplicates.

Use a transaction and a unique execution key, with an appropriate isolation/locking strategy. Recheck the exact approved payload, expiry, resource version and executing authority. Race both attempts in a disposable database and show one permitted local side effect, not merely one successful UI message.

For a remote service, explain the remaining boundary: a local SQL transaction cannot include arbitrary remote effects atomically. A durable intent/outbox, compatible provider idempotency and reconciliation address different parts of the problem. Unknown completion must remain visible where it cannot be resolved.

## P11 — Reject a harmful action even when its JSON is valid

A retrieved document tells the assistant to ignore approval and send a secret to an external URL. The model emits a schema-valid tool call. Explain where execution must be prevented.

The backend checks authenticated identity, action scope, approved destinations and current authorization independently of the model. Document text cannot grant permission. A schema validates structure, not whether an action is allowed. A hidden button or stronger prompt is not the enforcing control.

Use synthetic secret-like text and fake tools to test cross-team lookup, cached previews, malicious metadata and stale approval. Inspect traces too: rejecting execution does not help if logging already leaked private content. No real secret or external transmission is needed.

## P12 — Distinguish a numerical preference example from actual RL training

Start with a DPO scaled margin of zero. Its pair loss is -log(sigmoid(0))=log 2≈0.6931. At scaled margin one the loss is about 0.3133. Explain which answer is labelled preferred and which reference/policy log probabilities form that margin.

For PPO, calculate the clipped policy surrogate for both advantage signs; do not call it clipping rewards. For a group-relative example, equal rewards provide no within-group ranking, and division by zero must be avoided.

These arithmetic tasks are not a trained policy. A toy action-policy update and a real language-model adapter run are additional evidence levels with different data, compute and evaluation requirements. A good answer names what was actually calculated or run, including masks, reference roles and independent quality checks.

## P13 — Interpret a serving measurement before optimizing it

Run A has first-token time 100 ms and nine later token gaps of 20 ms. Run B has first-token time 50 ms and nine gaps of 40 ms. For ten output tokens, totals under this simplified timing boundary are 280 ms and 410 ms. B starts sooner but finishes later.

State whether measurements include queueing, retrieval and postprocessing. Match prompt/output lengths, concurrency, runtime, cache warmth and model version before attributing a difference to compression. Quantization can reduce storage without speeding unsupported kernels or preserving all task quality.

A memory estimate is useful before loading a model, but is not a measured peak. If hardware is unavailable, label the design unmeasured instead of inventing latency numbers.

## P14 — Explain a capacity estimate and a budget race

At a stable average arrival rate of two requests per second and mean end-to-end time eight seconds, Little's law gives average in-flight work 16. This does not describe peak traffic, p95 latency or prove that one GPU can support that concurrency.

For a budget race, two requests both see enough money before either reserves it. A non-atomic check can overspend. Reserve permitted worst-case usage atomically before dispatch, then settle actual usage under the documented policy. Include allowed retries/candidates in planning and account for uncertain in-flight work.

A local simulated workload can demonstrate queue saturation, rate limiting, cancellation and recovery. Review traffic, errors, latency, saturation and task quality together; high tokens per second alone can hide unacceptable waiting or wrong answers.

## P15 — Adapt a system design when the interviewer changes a constraint

Choose one design scenario, explain its user/task and simplest viable data flow, then respond to a changed condition: offline only, ten times the peak traffic, a revoked source, an outage or stricter privacy.

A good answer identifies which assumptions broke and what changes as a result. Offline operation may remove cloud fallback; source revocation affects retrieval and caches; more traffic may expose queueing rather than model-size limits. Do not add agents, Kubernetes or fine-tuning solely to sound sophisticated.

Calculate one relevant bound, walk a successful request and a failure, and name what you would measure before committing. Explain what you deliberately leave out and when specialist help is needed. There is no single architecture diagram correct for every version of the problem.

## P16 — Reproduce a run when a required artifact is missing

A colleague receives weights but no tokenizer, preprocessing configuration or dependency versions. Explain why loading the tensors is not the same as reproducing the prediction pipeline.

Identify the compatible base/template or feature map, input/output contract, code revision and permitted data. Reproduce a small non-sensitive fixture first, with known expected outputs and declared tolerance. A hash detects changed bytes but does not establish a publisher is trusted.

In a disposable repository, inspect the diff and make a small tested change on a branch. Do not rewrite shared history casually or practice destructive commands on real work. A release should exclude secrets/private data and document its limitations. Another reader should be able to locate the exact code, settings and result, not just see a successful screenshot.

## Rehearsal

Explain the main idea in your own words before adding specialist detail. Score theoretical explanation, calculation, coding, debugging and design separately. A small correct function you can change is stronger implementation evidence than a large generated module you cannot explain. These exercises do not automatically certify readiness.
