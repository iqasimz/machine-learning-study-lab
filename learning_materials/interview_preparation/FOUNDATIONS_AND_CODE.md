# Python, mathematics and research interview questions

Try a short answer to the situation, then use the explanation and follow-up to check your reasoning. These are examples, not scripts or claims about your experience. Full mechanisms and calculations remain in the course; describe only work you actually did.

## F01 — The package installed, but your script cannot import it. What would you check?

I would first check whether installation and execution use the same Python interpreter. A successful installation in another environment does not make a package available to this script.

The shell starts a Python process. That interpreter loads your modules and dependencies, executes code and asks the operating system for files, memory or network access. A virtual environment selects an interpreter/package set; it is not a separate computer or a security sandbox. `pip` installs packages; imports use Python's module search rules. Use `python -m pip` to reduce interpreter/pip mismatch.

**Follow-up example.** Use the executable running the script with its -m pip command. Compare interpreter paths and the environment before reinstalling; installation elsewhere does not fix this process.

## F02 — Why does calling this function twice append to the same list?

The likely cause is a shared mutable object, such as a default list created once when the function was defined. A fresh per-call list prevents accidental accumulation.

A function transforms inputs or performs an operation; a class groups state and related behaviour. Abstraction exposes a clear contract while hiding irrelevant detail—not necessarily inheritance. A mutable list can change in place; assigning it to a second name does not copy its contents. Python passes object references by assignment; saying simply “pass by reference” can mislead about rebinding.

Use pure functions for budget maths/token shifts, a record/dataclass for a run manifest, and a small adapter interface for model backends. Prefer composition over a deep class tree without a reason.

**Follow-up example.** A mutable default list may be reused across calls. Default to None and create a fresh list when no list was supplied. A shallow copy still shares nested objects, so choose copying deliberately.

## F03 — You need the highest-scoring k results. Would you sort everything?

Lists are ordered dynamic arrays; dictionary/set lookup is expected average O(1), not a universal worst-case guarantee. A heap keeps the smallest/largest-priority item accessible in O(log k) update work; a deque supports efficient operations at both ends. Sorting n items usually costs O(n log n); binary search needs sorted data and costs O(log n) comparisons.

For top-k chunks among n scores, a size-k heap uses O(n log k) time and O(k) extra storage; full sort is simpler and often fine for small n. Complexity describes growth, not exact milliseconds.

**Follow-up example.** Define ties and boundary cases such as k=0 first. A size-k heap can save work for small k; full sorting is often clearer for a small list. Complexity is not a measured latency.

## F04 — How would you follow document links without looping forever?

A tree is a connected acyclic graph; a rooted tree can describe a hierarchy. General graphs represent relationships that can include cycles. BFS explores unweighted shortest paths using a queue; DFS explores depth-first using a stack/recursion. A visited set prevents repeated traversal/cycles. An LRU cache evicts the least recently used entry; a TTL expires old entries. Neither policy defines whether the cached answer remains authorized or true.

**Follow-up example.** Keep a visited set and a defined stopping rule. If you cache results, include access scope and source version: expiry alone does not establish that a cached answer is true or permitted.

## F05 — A tensor operation runs but gives the wrong loss. What could be happening?

I would check shapes and the meaning of each axis before trusting the scalar loss. A valid broadcast can produce the wrong comparison without raising an exception.

A tensor is an array with shape/dtype/device. Matrix multiplication combines aligned dimensions: `[B,T,d] @ [d,V] → [B,T,V]`. Broadcasting can expand compatible dimensions without your intending it; correct execution is not proof of correct maths. A gradient gives the local sensitivity of a scalar objective to each parameter.

For `L=½(wx−y)²`, `dL/dw=(wx−y)x`. At `w=1,x=2,y=1`, gradient=2; learning rate .1 gives new w=.8. The chain rule connects such local derivatives through layers.

**Follow-up example.** Subtracting [B,1] and [B] can broadcast into [B,B]. Inspect the intended axes and unreduced values on two known cases; averaging a wrong-shaped loss can hide the bug.

## F06 — Can you average two microbatch losses with unequal valid-token counts?

Gradients add until cleared. Accumulation approximates a larger batch only if loss weighting and optimizer-step timing are correct; stochastic layers and other details can still differ. Ten valid tokens with mean loss 2 and ninety with mean loss 4 give `(10×2+90×4)/100=3.8`, not 3. Masked/padding tokens must not inflate the denominator.

**Follow-up example.** Accumulate correctly weighted gradients, then clip under the declared update rule and step once. Compare with a deterministic full-batch reference; dropout or batch-dependent layers can still differ.

## F07 — Why is next-token cross-entropy not confidence that an answer is true?

Cross-entropy punishes assigning low probability to the observed target. For one correct token with p=.25, loss is `−ln(.25)≈1.386` nats. KL compares distributions: `Σ p log(p/q)`; it is directional, generally not symmetric, and can diverge when q gives zero probability to p-supported events.

Calibration asks whether predicted confidence matches frequency for a defined event. A .9 token probability is not a .9 probability that a full answer is factually correct. Define the event, labels and sample population before making a confidence claim.

**Follow-up example.** A token probability of 0.9 concerns that token under its context, not verified factual correctness of the full answer. Calibration needs reviewed labels for the actual event.

## F08 — A new system gains a few points on the same questions. Is that convincing?

Bootstrap resampling builds many alternative evaluation lists by sampling the original units with replacement. For two systems on the same cases, resample the cases **as pairs**, recompute the score difference many times, and use an interval procedure such as percentile bounds. If questions share a document/scenario, resample that cluster where appropriate rather than pretending all questions are independent. The interval estimates uncertainty under the resampling assumptions. A 95% confidence procedure aims for repeated-sampling coverage; it is not a posterior probability assigned to this particular fixed population difference.

Example: an estimated gain .03 with interval [−.01,.07] does not clearly distinguish a gain from no gain under that procedure. Holm adjusts a declared family of p-value tests to control family-wise error: sort p-values and compare successively to `α/(m−i+1)`, stopping at the first failure. It cannot rescue leaked data or uncontrolled post-hoc searching.

**Follow-up example.** A bootstrap of fixed predictions estimates case-sampling uncertainty, not variation across retrained models. Repeat seeds separately when that source of variation matters.

## F09 — How would you know which dataset produced a checkpoint?

An ID names a record; a content hash fingerprints bytes under a defined hash/serialization rule. Different line endings or JSON ordering can change bytes, so canonicalize deliberately. Hashes help identify exact artifacts and detect changes; they do not establish truth, license permission or trust in the publisher.

**Follow-up example.** Record stable source IDs, raw and processed hashes, and configuration revisions. Row numbers can change after sorting; hashes detect changes but do not establish data rights or publisher trust.

## F10 — A request timed out. Is it safe to retry?

Not automatically: the server might have completed the work before its response was lost. The retry decision depends on whether the operation is safe to repeat or has reliable idempotency/status handling.

DNS resolves a hostname; transport establishes communication; TLS can protect data in transit and authenticate the server; HTTP carries the request/response. None alone authorizes a user to see a tenant's job. Your backend validates inputs, authenticates identity, checks permission, calls bounded services and returns a defined status/body.

Streaming delivers partial response events, not automatic durable completion. A timeout means the caller stopped waiting, not proof the server did nothing. Distinguish retryable rate/temporary failures from invalid requests and authorization denials; respect provider retry guidance within your budget/deadline.

**Follow-up example.** A remote write may already have completed. Use supported idempotency or status lookup and reconcile uncertain outcomes. Do not treat an authorization failure as a transient error.

## F11 — Why does an async API stall during preprocessing?

Async lets a task yield while waiting on I/O; it does not make CPU-heavy Python or model inference free. Blocking work inside the event loop can stall unrelated requests. Threads can help I/O and some native operations; Python runtime/GIL details vary, so measure rather than assume arbitrary parallel Python speedup. Processes isolate interpreters/state but add memory and communication costs.

A durable worker queue persists jobs across API restarts. An in-process background task does not automatically survive a crash. Bound concurrency, propagate cancellation/deadlines, and lock/atomically update shared budget/state where races matter.

**Follow-up example.** A semaphore limits in-flight work but cannot make a blocking CPU loop nonblocking. Use a suitable worker boundary; durable job storage is separate from concurrency control.

## F12 — Two reviewers click execute together. How do you prevent duplicate tickets?

Two workers can both pass a preliminary check before either writes. Enforce the allowed local execution with a transaction and unique key, rather than relying on the UI or a check-then-write sequence. Tables model proposals/approvals/executions; primary/foreign keys and uniqueness enforce relationships. An index speeds some reads at storage/write cost; it is not automatically useful for every query. A transaction groups operations under a chosen isolation model. ACID means atomicity, consistency constraints, isolation and durability—not that arbitrary business logic is correct.

For a synthetic ticket, a unique execution key plus a transaction prevents duplicate local writes. A remote API write cannot be made atomic merely by starting your local SQL transaction. Use provider idempotency, an outbox/reconciliation strategy and explicit uncertain outcomes where needed.

**Follow-up example.** Use a tested transaction and unique execution key for local state. A remote API is outside that transaction and needs compatible idempotency or reconciliation.

## F13 — How would you run an experiment on a cluster and collect its logs?

The terminal displays a shell; the shell launches commands. macOS and Linux have related Unix tooling but are not identical. SSH opens an authenticated encrypted connection to another machine. On a cluster, the login node is usually for light work; SLURM requests scheduled resources on compute nodes. Git versions source history, not running processes or hidden datasets.

Know `pwd`, `ls`, `cd`, `less`, `rg`, `head`, `tail`, `ps`, `df`, safe permissions, `ssh`, `scp/rsync` with reviewed paths, `tmux`, and cluster-approved `sbatch/squeue/sacct/scancel`. For Git: status/diff/add/commit/branch/switch/log, merge/rebase conflicts and safe pushing. Never practice destructive shell commands on your real thesis; use a disposable fixture directory.

**Follow-up example.** Test and version locally, connect with SSH, prepare the remote environment and submit a bounded scheduler job. Inspect status/logs and cancel only your own job; a login node is not a free training machine.

## F14 — Your classifier finds more positives but its F1 falls. How is that possible?

More true positives can improve recall while extra false alarms reduce precision enough to lower F1. I would compare both error types, not just recovered positives.

Linear regression predicts continuous values; logistic regression models class probabilities. Trees split feature space; random forests average/vote across randomized trees to reduce variance, without universally eliminating overfitting. Neural networks learn representations through differentiable layers. A strong lexical/logistic/tree baseline can beat a complicated model for the task and data at hand.

Precision asks how many predicted positives are correct; recall asks how many actual positives were found. F1 trades them off; macro averaging gives classes equal weight, not instances equal weight. Thresholds, prevalence and error costs matter. Use validation for choices and a held-out test for final reporting.

**Follow-up example.** TP=8, FP=2, FN=2 gives binary F1=16/20=0.80. TP=9, FP=9, FN=1 gives 18/28=0.6429: more recovered positives, but many more false alarms. Macro-F1 additionally averages class-level scores.

## F15 — Why might lexical search beat embeddings on a support query?

An exact product identifier or error code can be better served by lexical matching than broad learned similarity. I would inspect those queries before switching retrieval methods.

Classification, sequence labelling, retrieval, extraction, generation and entailment ask different questions. Words/tokens are not interchangeable. Negation, references, discourse, pragmatic intent and annotation guidelines can change a label even when vocabulary is similar. TF-IDF/BM25 use lexical matching/weighting; embeddings represent learned similarity; a cross-encoder jointly reads a query-document pair to score it.

RNN/LSTM models maintain sequential state; Transformers use attention with positional information. Bidirectional encoders suit many representation/classification tasks; causal decoders generate next tokens. This does not make one universally best.

**Follow-up example.** Exact error codes can favor lexical matching; paraphrases can favor learned similarity. Compare the actual relevance cases. Similarity does not prove entailment or source support.

## F16 — Which parts of this project did you implement yourself?

Explain your contribution using concrete decisions, code changes, tests and results. Distinguish original work from adapted components, name relevant tools or sources honestly, and be ready to explain and modify a small part of the implementation. Research design and independent coding are different kinds of evidence; do not claim either without an example.

For an unfamiliar question: clarify the task, state assumptions, derive a small case, propose a simple baseline, identify uncertainty and describe how you would test it. Avoid guessing a framework API as if certain.

**Follow-up example.** Separate research decisions, adapted/generated code, testing and unaided implementation. If a follow-up is unfamiliar, state assumptions and reason through a small case rather than invent experience.
