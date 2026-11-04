# Language-model training interview questions

Try a short answer to the situation, then use the explanation and follow-up to check your reasoning. These are examples, not scripts or claims about your experience. Full mechanisms and calculations remain in the course; describe only work you actually did.

## M01 — Where do next-token training labels come from?

In causal pretraining, the text supplies the target. For `[BOS,A,B,EOS]`, an explicit shifted convention uses inputs `[BOS,A,B]` and targets `[A,B,EOS]`. The objective averages negative log probability over included target positions. Some libraries accept unshifted labels and shift internally: check the implementation so you do not shift twice.

This differs from supervised classification, where annotators/dataset rules supply an external class label. A generated SFT answer is also not automatically a gold answer; it needs the chosen quality/provenance review.

**Follow-up example.** Some libraries shift labels internally. Check one explicit input/label pair so your own shift does not shift the target twice.

## M02 — Why did changing the tokenizer or chat template break the model?

A tokenizer converts text to IDs that the model's embeddings understand. BPE repeatedly merges frequent pairs under a learned rule; byte-based choices help represent arbitrary input but may fragment languages unevenly. Vocabulary size trades longer sequences against larger embedding/output tables. The exact tokenizer is part of the checkpoint contract.

A chat template turns roles/messages/tool results into the token sequence expected by an instruction model. Wrong role delimiters, special tokens or thinking-mode settings can change behaviour without changing weights. Do not “fix” a mismatched tokenizer by substituting IDs from another model.

**Follow-up example.** Token IDs belong to a particular vocabulary and embedding table. Inspect the serialized roles, special tokens and response mask instead of copying another model's IDs.

## M03 — How can a causal model train in parallel but generate one token at a time?

Causal training predicts the next token from preceding tokens. Masked training predicts deliberately hidden tokens using permitted bidirectional context. Encoder–decoder models encode an input and generate a target using decoder self-attention and cross-attention to the encoded input. They differ in objective and information flow, not only model size.

Teacher forcing provides true preceding target tokens during training. At generation time, a causal decoder conditions on its own prior outputs, so mistakes can affect later tokens. The causal mask prevents future-token leakage while allowing many training positions in parallel.

**Follow-up example.** Training knows the correct prefix at each position; generation must wait for its own earlier choices. Test the causal mask by changing a future input and checking earlier logits in evaluation mode.

## M04 — What do queries, keys and values actually do in attention?

Input states X are projected into queries Q, keys K and values V. For one head, `softmax(QKᵀ/√d_k + mask)V` forms a weighted sum of value vectors. The query/key comparison determines weights; values supply the mixed content. Scaling stabilizes score magnitude as key dimension grows. Multi-head attention learns several such projections; heads are concatenated/projected.

Residual connections preserve an additive path for information/gradients; normalization controls activation scales; the position-wise MLP transforms features. Positions matter because content-only attention does not by itself distinguish arbitrary token order. Learned/sinusoidal positions and RoPE encode order differently.

**Follow-up example.** The largest attention coefficient is a mixing weight, not proof of why an answer is true. Other heads, layers and residual paths also affect the result.

## M05 — What changes when you call backward and then the optimizer?

Read a batch → forward pass → compute correctly masked loss → backward gradients → optional accumulation/clipping → optimizer update → schedule/record/checkpoint. AdamW keeps moving gradient statistics and decouples weight decay; its extra state costs memory. Dropout, weight decay and early stopping solve different problems.

Loss decreasing does not prove useful generalization. Compare a tiny overfit sanity check, held-out loss and task outcomes. Mixed precision can reduce cost but introduces backend/dtype/operator constraints; FP16/BF16 are not interchangeable guarantees of stability.

**Follow-up example.** Backward computes gradients; the optimizer applies its update. If learning fails, check finite gradients, parameter connectivity, optimizer membership and whether a step actually occurs.

## M06 — Training loss becomes NaN after 200 steps. What would you inspect?

NaN means “not a number,” often produced by an invalid numerical operation. For this failure, locate the first nonfinite input/activation/loss/gradient; inspect data, masks, log/softmax stability, dtype and learning rate before randomly changing everything. An all-masked target batch can cause an invalid reduction. For OOM: identify weights, optimizer, activations, cache and concurrent processes; then reduce the actual limiting component. Training KV cache is not necessarily an inference-style persistent cache.

A checkpoint needs model/optimizer/scheduler and, where applicable, scaler, RNG and sampler/data position. Weights alone do not reproduce the next update. Resume compatibility and deterministic tolerances must be stated.

**Follow-up example.** A memory failure needs allocation/workload checks, not the same response as a nonfinite loss. A bad restart needs model, optimizer, random and data-order state checks. Diagnose the observed failure first.

## M07 — Why might training not fit when FP16 weights occupy only 2 GB?

Raw FP16 weights for 1B parameters occupy about 2 GB decimal before overhead. Training also needs gradients, optimizer states and activations. A common rough dense-training FLOP estimate is `6×parameters×training tokens`, under simplifying assumptions; attention/context, architecture and implementation can make it inaccurate. It is an order-of-magnitude budget, not a timing prediction.

Replicated data-parallel training copies model state across devices and aggregates gradients; FSDP/ZeRO shard selected state; TP splits tensor operations; PP splits layers/stages. Communication bandwidth, synchronization and pipeline bubbles can dominate. Gradient checkpointing recomputes activations to save memory at compute cost.

**Follow-up example.** Choose a distributed strategy only after identifying whether replicated state, activations or an operation is limiting. Sharding and recomputation solve different problems and add communication or compute.

## M08 — Why does grouped-query attention reduce serving memory?

GQA shares key/value heads among query heads, reducing KV storage relative to standard MHA at fixed other dimensions; MQA uses a single KV head. MoE routes tokens to a subset of expert networks: active computation and total parameter/storage requirements are different, and routing/load balance/communication add costs. RoPE rotates query/key components with position; it is not unlimited reliable context extrapolation.

An efficient attention kernel can avoid materializing large intermediate matrices and improve memory movement without changing the mathematical dense-attention function, up to numerical differences. State-space/hybrid/diffusion-language approaches change other assumptions and need their own objective/context evaluation.

**Follow-up example.** GQA shares KV heads; MoE instead selects expert subnetworks. Efficient kernels change execution and memory movement. Compare the resource each mechanism actually changes.

## M09 — An assistant gives outdated policy answers and inconsistent JSON. Would you fine-tune it?

Prompting changes the input; RAG supplies selected external evidence without necessarily changing weights. Continued pretraining extends a language-model objective on new text. SFT trains on demonstrations of desired input/output behaviour. None substitutes for permission to execute an external action.

Changing service limits belong in versioned evidence, not solely in a weight update. Consistent response/tool format may benefit from SFT, but first test a well-prompted baseline. A model can learn behaviour without reliably memorising exact current facts, and can memorize private training content despite retrieval ACLs.

**Follow-up example.** Use current permitted evidence for changing facts and a schema/validation baseline for formatting. SFT may improve consistent behaviour but does not replace fresh policy or authorization.

## M10 — Why can LoRA still run out of memory with few trainable parameters?

Full tuning updates all selected base parameters. LoRA learns a low-rank update: `W' = W + sBA`, with A `[r,d_in]` and B `[d_out,r]`; parameter count is `r(d_in+d_out)` instead of `d_in×d_out` for that matrix. At 4096×4096,r=8, this is 65,536 adapter parameters. Low trainable-parameter count does not remove base-weight or activation memory.

QLoRA combines low-bit base storage with adapter learning and specific implementation choices; it is not simply “LoRA has quantized gradients.” Hardware/operator support matters, so CUDA-oriented tooling is not assumed to run natively on M4. The adapter needs its compatible base/template/config and evaluated conversion path.

**Follow-up example.** The frozen base and required activations still occupy memory. QLoRA adds a particular low-bit base/adaptation recipe; it is not simply quantized gradients and needs supported operators.

## M11 — What changes during one LLM reinforcement-learning iteration?

The policy generates a response/action sequence; an environment/verifier/reward model assigns feedback; an advantage/baseline helps evaluate choices; an objective updates policy parameters. Repeat with controlled sampling and evaluate independently. Yes, LLM RL updates weights and is a form of post-training, but its signal/objective differ from matching a fixed demonstrated answer in SFT.

Separate roles: policy generates; reward model scores preferences; critic estimates value; reference policy anchors behaviour. The old rollout policy used for a PPO probability ratio need not be the fixed reference used for KL. Some methods omit a learned critic/reward model.

**Follow-up example.** Policy, reward model, critic and reference have different roles. The old sampling policy in a probability ratio need not be the fixed regularization reference.

## M12 — What does DPO encourage in a preferred versus rejected answer pair?

DPO encourages the preferred completion to gain relative likelihood against the rejected one, compared with a reference model. It learns from preferred/rejected response pairs without running a separate on-policy rollout-and-critic loop for that objective. A standard form is `−log σ(β[(log π(y+|x)−log π_ref(y+|x))−(log π(y−|x)−log π_ref(y−|x))])`. Sequence log probabilities sum included response-token log probabilities; masking and the actual chosen objective variant matter.

The margin compares the policy's relative preference against a reference, not just whether the winner's raw probability is large. At margin zero the loss is ln2≈.693; at scaled margin 1 it is about .313. Lower training loss does not guarantee truthful answers if pairs reward confidence, length or sycophancy.

**Follow-up example.** Decreasing pair loss does not establish truthfulness. Labels may reward verbosity or confidence; check held-out task quality and response masking independently.

## M13 — What does PPO clip, and why does advantage sign matter?

An advantage estimates how much better or worse a sampled action was than a baseline. PPO's clipped policy surrogate uses ratio `r=π_new(a|s)/π_old(a|s)` and advantage A: maximize `min(rA, clip(r,1−ε,1+ε)A)`, with value/entropy/KL terms depending on the implementation. For A=1,r=1.5,ε=.2, the clipped surrogate is 1.2 rather than 1.5. Negative advantage changes which term limits the objective; calculate instead of saying “always clip rewards.”

GRPO-family implementations use multiple sampled responses to a prompt to form a group-relative advantage, often using group reward mean/std instead of a learned value critic. Equal rewards give no centred relative signal; standard deviation handling must avoid division by zero. Loss normalization, KL and importance weighting vary by implementation/version.

**Follow-up example.** For A=-1, ratio=0.5 and band [0.8,1.2], the surrogate is min(-0.5,-0.8)=-0.8. It is maximized; a loss commonly negates it. GRPO normalization is not clipping rewards.

## M14 — Why does temperature zero not guarantee factual or identical answers?

At each step the model produces logits. Temperature rescales them before softmax; top-k/top-p restrict candidates under defined rules; sampling chooses a token; stopping rules terminate. Greedy decoding selects a maximum rather than naively dividing by zero for temperature=0. Low temperature does not prove factuality or universal reproducibility.

Constrained generation can enforce a grammar/schema, but a valid structure can contain false facts or unauthorized actions. Free-form reasoning adds computation/text and may help some tasks; an explanation is not automatically a faithful account of internal causation.

**Follow-up example.** A schema can ensure output structure without ensuring true facts or authorized actions. Decoding, semantic validation and backend permissions solve different problems.

## M15 — Why does a long prompt delay the first token and require more memory?

Prefill processes prompt tokens; decode produces further tokens while reusing eligible cached keys/values. Approximate KV bytes are `2×layers×batch×tokens×kv_heads×head_dim×bytes_per_element`, excluding allocator/paging/other overhead. For 12 layers, one sequence,1024 tokens,4 KV heads,64 dimensions,2-byte elements, the cache is 12 MiB. Doubling sequence count roughly doubles that component.

Quantization reduces precision/storage, with metadata, kernel and quality trade-offs. Distillation trains a student using teacher signal; pruning removes selected structure/weights. None guarantees speedup on unsupported kernels. Prefix caching, batching and speculative decoding affect different portions of latency; exactness depends on the algorithm used.

**Follow-up example.** Time to first token, inter-token gaps and total latency measure different things. State prompt/output lengths, concurrency, cache warmth and runtime before comparing.

## M16 — A new model scores better. How would you establish that the change helped?

Define the task, baseline, selection data, final evaluation and primary metric before comparing. Keep model/input/evidence/budget fixed except the intended factor; where not possible, call it an operational comparison rather than isolated causality. Report support/coverage, failures, cost and uncertainty—not only a selected best score.

Generalization can fail because inputs changed, label definitions changed or a shortcut stopped working. Test those separately. A probe can reveal decodable information without proving the model used it; an activation intervention needs controls for collateral damage and distribution effects.

**Follow-up example.** A probe can reveal decodable information without proving it caused a gain. Mechanistic claims need controlled interventions and checks beyond a better task score.
