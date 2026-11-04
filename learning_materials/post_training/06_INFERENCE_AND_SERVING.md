# R6. LLM inference, KV caching and serving

Inference uses a model's parameters to calculate outputs, while serving adds request admission, scheduling, state management and delivery. Prefill processes the known prompt; autoregressive decoding processes successive chosen tokens. A key/value cache can reuse earlier attention projections, but its memory, compatibility and privacy constraints remain separate from model-weight storage.

The fixed cache example has two layers, four query heads, two KV heads, head width four and two-byte values. Change cached length to calculate value bytes. A separate hypothetical timing example uses first-token time 100 ms and nine later 20 ms intervals, not measured hardware performance.

The Transformer explained autoregressive generation with fixed parameters. Serving turns that computation into a resource-bounded service; cache accounting, throughput and user-visible latency should not be confused with the forward layers themselves.

## 1. Request latency and throughput

**Key terms.** **TTFT** is time to first token; **inter-token latency** measures later gaps; **throughput** is output per time; **queue time** is waiting.

**Process.** A request waits, tokenizes, processes its prompt and generates tokens before delivery. Measure stages separately so a faster kernel is not confused with a faster complete response.

```text
request → authentication/queue → tokenization/retrieval → prefill
→ first token → repeated decode → output checks → response complete
```

**Time to first token (TTFT)** includes waiting and initial computation under a stated measurement boundary. **Inter-token latency (ITL)** measures spacing between emitted tokens. **Throughput** is work per second—state whether tokens, requests or successful tasks. **End-to-end latency** includes the complete user-visible process. p95 latency is a tail percentile, not the average.

Streaming can make an application feel faster while leaving total computation unchanged. Streaming unvalidated output can also expose unsupported or unsafe content before a later check catches it.

## 2. Prefill versus decode

**Key terms.** **Prefill** processes a known prompt; **decode** generates later tokens; the **KV cache** retains earlier attention keys/values.

**Process.** Prefill produces representations and cached state. Each decode step processes a new token using that cache, appends its result and repeats until stopping.

Prefill processes the prompt, often across many positions in parallel, and builds state. Autoregressive decode produces successive tokens; each step depends on earlier choices. The workload changes with batch, context and architecture. Prefill is often relatively compute-heavy; small-batch decode is often bandwidth-sensitive. Neither description is universal.

Long prompts increase prefill cost and cache use. Long outputs increase repeated decode cost and tie up serving capacity. Advertised tokens/second without prompt/output length, concurrency and hardware is not an informative comparison.

## 3. KV-cache memory

**Key terms.** Here **B** is batch size, **T** cached length, **L** layers, **h_KV** KV heads, **d_h** head width and **b** bytes per cached value.

**Process.** Store keys and values for each cached position at each layer. Multiply their dimensions before admitting requests; add weights, activations and runtime memory separately.

For a conventional decoder storing keys and values per token:

\[
M_{KV}=2 L B T h_{kv} d_{head} b.
\]

`L`: layers; `B`: concurrent sequences under equal-length simplification; `T`: cached tokens per sequence; `h_kv`: KV heads; `d_head`: head width; `b`: bytes per cached value. Factor 2 is for keys and values. For unequal lengths, replace `BT` with the sum of cached sequence lengths. Sliding windows, latent attention, quantized caches and shared prefixes change the accounting.

Example: `L=32,B=1,T=8192,h_kv=8,d_head=128,b=2` gives **1,073,741,824 bytes = 1 GiB**. Eight such independent sequences need about 8 GiB just for this cache, before weights/workspace. GQA head count belongs here, not automatically the number of query heads.

A cache is conditional on the exact preceding computation. Reusing incompatible prefixes, adapters, position settings or model revisions can be incorrect. Cache isolation matters for privacy and tenant separation.

## 4. Batching and memory management

**Key terms.** **Continuous batching** admits/removes requests during service; **paged cache** manages KV blocks; **prefix caching** reuses compatible prefix work.

**Process.** Batch eligible requests, allocate cache and execute shared model steps. Release completed requests and admit new ones while checking latency and permissions.

Static batching waits for a fixed group; continuous batching admits/completes requests dynamically. This improves utilization but introduces scheduling trade-offs. Very long requests can harm short requests' latency. Admission control rejects/queues work rather than allowing memory exhaustion.

PagedAttention organizes KV state in blocks to reduce fragmentation and support flexible allocation/sharing. It is a cache-management design, not the same as FlashAttention's attention-IO algorithm. A server may use both.

Prefix caching reuses compatible computed prompt prefixes. It mainly avoids repeated prefill work; it does not make all future decode steps free. Chunked prefill processes long prompts in pieces to coexist with decode work. Scheduler policy affects TTFT, ITL, fairness and throughput differently.

## 5. Speculative decoding

**Key terms.** A **draft model** proposes tokens; a **target model** verifies them; an **acceptance rule** selects retained draft tokens.

**Process.** Generate a draft block, verify with the target and accept or correct it. Measure full latency including rejected drafts and extra cache work.

A cheaper draft model proposes several tokens; the target verifies them in a more parallel operation. With an appropriate accept/reject and correction procedure, exact speculative sampling preserves the target distribution. For a proposed token from `q`, a basic acceptance term is `min(1,p(token)/q(token))`; rejections require sampling from a corrected residual distribution, not simply choosing another arbitrary draft token.

Speedup depends on acceptance rate, proposal cost, verification efficiency and hardware. An inaccurate drafter can make things slower. Not every method marketed as “speculation” preserves exactly the same distribution, and finite-precision/runtime choices can still affect outputs.

### Measure a serving change without hiding its cost

Use the same model revision, prompt/output distribution, quality protocol and hardware before changing batching, caching or speculation.

1. Include queue time in the user-visible measurement. Separate cold startup from warmed service, and state whether network time is included.
2. Report TTFT, inter-token latency, complete-request latency, successful requests/second and generated tokens/second. They can move in opposite directions.
3. For `n` output tokens, a rough sequential latency model is `TTFT + (n−1)×mean inter-token latency`, plus any completion-only checks outside that boundary. It is an accounting approximation, not a constant-time prediction for variable workloads.
4. Increase concurrency gradually and plot tail latency, failures and memory against throughput. Stop interpreting a saturated system's successful-request subset as representative of all submitted work.
5. For speculation, count draft generation, target verification, rejection/correction and cache handling. High acceptance alone is insufficient if proposal or verification overhead dominates.

Exact speculative sampling corrects rejected proposals so the resulting distribution matches the target in the ideal algorithm. Accepting every draft token or sampling arbitrarily after a rejection changes that guarantee. Likewise, prefix caching requires compatible tokens, model, adapter and positional state; identical-looking text is not enough. Sources: [speculative decoding](https://arxiv.org/abs/2211.17192), [PagedAttention](https://arxiv.org/abs/2309.06180) and [vLLM serving documentation](https://docs.vllm.ai/).

## 6. Parallel serving and disaggregation

**Key terms.** **Disaggregation** separates stages such as prefill/decode; **parallel serving** divides work across devices; **transfer cost** moves state.

**Process.** Route requests to workers, execute their stages and transfer required state. Compare the whole path with a simpler deployment, including network and queue overhead.

Tensor/pipeline parallelism can make large models fit and share work, at communication cost. Separate prefill/decode workers can tune hardware/scheduling for different workloads but need state transfer. Replicas handle more requests; sharding partitions one model's work/state. They are not synonymous.

Route easy queries to smaller models and escalate difficult cases if a validated policy supports it. Routing quality, fallback cost and failure detection determine total value. A router's confidence needs held-out evaluation; “small model first” can add latency if most requests escalate.

## 7. Queues and capacity

**Key terms.** An **arrival rate λ** counts incoming requests per time; **capacity** is completed work per time; **Little's law** relates average work in system to arrival rate and time.

**Process.** Measure arrivals and service time, then reserve capacity or apply backpressure before queues grow. Check tail latency and cancellation under realistic traffic, not just empty-system averages.

**Calculation.** Little's law uses L=λW for long-run averages in a stable system with matching boundaries. At 2 completed requests/second and 3 seconds average time in the system, average requests present is 6. This identity alone does not prove sufficient capacity or a tail-latency guarantee.

The long-run average identity is not permission to exceed memory or ignore bursts. Near saturation, queueing and tail latency can grow sharply; the capacity and admission policy still need measurements under realistic traffic.

Set rate limits, per-request token caps, concurrency limits and deadlines. Backpressure means slowing/refusing incoming work when capacity is constrained. Retry with bounded backoff and respect idempotency; uncontrolled retries can worsen an outage.

## Exercises and checks

1. Double cached length: conventional KV memory doubles. Dense prefill attention arithmetic may grow more steeply.
2. Why might batching improve throughput but worsen TTFT? Waiting, scheduling and sharing compute add delay for individual requests.
3. Which measurements are needed to compare servers? Model/tokenizer/revision, hardware, precision, prompt/output distributions, concurrency, quality, TTFT/ITL/end-to-end percentiles, throughput and failure rate.

Sources: [PagedAttention](https://arxiv.org/abs/2309.06180), [vLLM attention design](https://docs.vllm.ai/en/latest/design/paged_attention/), [speculative decoding](https://arxiv.org/abs/2211.17192). The static KV-cache diagram and equations above separate cache bytes from the rest of the runtime.
