# Language models, training and systems

## T0. The language-model training loop

Language-model training repeatedly asks the network to predict known tokens, measures their probability, and changes weights to reduce an objective. The text supplies the targets. Generating a continuation is different: the target is unknown and the model's chosen token becomes part of the next input.

### Construct the prediction problem

For a token sequence `[a,b,c,d]`, a three-position input can be `[a,b,c]` with targets `[b,c,d]`. At each position, causal attention permits only the prefix ending there. The final input c has a target d, even though d is outside the input slice. Padding masks and ignored labels remove artificial or deliberately unscored targets. Attention masks separately control what each position can read.

### Predict, score, differentiate, update

Embedding lookup maps IDs to vectors. Decoder blocks transform them. A vocabulary projection produces logits at each position. Softmax describes the next-token distribution, while a stable cross-entropy implementation can consume logits directly. If probabilities assigned to three realised targets are 0.5, 0.25, and 0.8, total negative log-likelihood is about 2.303 and mean is about 0.768 nats over three targets.

Backpropagation computes gradients of that loss through trainable operations. The optimizer applies an update using those gradients and any stored history. Clear gradient buffers at the intended update boundary, not indiscriminately between microbatches that should accumulate. Repeat with training examples. Teacher-forced training scores many positions in parallel under the causal mask; it does not sample a new sequence for every ordinary next-token update.

### Check progress without fitting the answers

Periodically evaluate development loss with frozen weights and evaluation behaviour. Use it to select a checkpoint. Test evaluates the selected model, not every trial until a pleasing score appears. Save compatible tokenizer, config, weights, and the additional optimizer/sampler state needed for continuation. A best checkpoint and the latest checkpoint answer different questions.

In this lab, scratch runs use real optimizer steps. Development loss is measured at step 1, each 20 steps, and the final step. `last.pt` is overwritten; it is not a history of model states. The standard export contains selected weights and final evidence, not a replay of all earlier token probabilities. The architecture diagram's head-only derivative is an illustration; the trainer differentiates the full trainable scratch network.

Sources: [D2L: language-model training](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html), [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html), [PyTorch: checkpointing](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## T1. Language-model training objectives

An objective defines which information the model receives and what it must predict. A causal next-token objective predicts each successor from its prefix. A masked-language objective predicts selected hidden tokens using permitted surrounding text. An encoder–decoder denoising objective reconstructs target text from a corrupted source. These are related learning problems, not interchangeable prompt templates.

For causal training, the loss is the mean negative log-probability of the supplied successors over scored positions. A correct token assigned probability 0.5 contributes $-\ln0.5\approx0.693$ nats. With two valid targets and one ignored padding position, divide by two, not three. Specifying the denominator is part of specifying the objective.

In **pretraining**, a broad corpus supplies self-supervised targets. Continued pretraining adapts the same kind of objective to another distribution. **Supervised fine-tuning** typically scores desired assistant responses, sometimes masking user/prompt tokens. At **inference**, parameters are fixed and decoding chooses outputs; no optimizer step is implied by a longer answer.

The objective rewards a measurable prediction behaviour, not all qualities a user wants. Lower token loss can coexist with factual errors, poor instruction following, or unsafe actions. Evaluate the intended behaviour separately and keep context visibility matched when comparing objectives.

Sources: [SLP: Transformers and pretraining](https://web.stanford.edu/~jurafsky/slp3/7.pdf), [BERT: masked pretraining](https://arxiv.org/abs/1810.04805), [T5: text-to-text objectives](https://arxiv.org/abs/1910.10683).

## T2. LLM training data, tokenization and packing

A training corpus is a collection of examples with provenance, permissions, boundaries, and a sampling policy. More tokens do not automatically mean better evidence. Duplicate documents can overweight one source or leak near-identical answers across splits. Reserve evaluation groups before fitting the tokenizer or selecting data filters.

Tokenisation maps text to model-compatible IDs. A larger vocabulary can shorten sequences while increasing embedding and output-table size; a smaller vocabulary can increase the number of positions needed. Different tokenizers make raw token perplexity difficult to compare. Preserve exact tokenizer files and special-token conventions with the model.

**Packing** combines short sequences to reduce padding waste. It also forces a choice about cross-document attention. Resetting position IDs alone does not prevent one document reading another. Use a suitable attention boundary mask or independent windows if isolation is intended. An end-of-document token marks a boundary but does not itself enforce invisibility. Some training setups intentionally allow cross-document context; document that choice rather than assuming it.

Inspect one real batch: input IDs, shifted targets, attention visibility, padding, loss mask, and document identity. The scratch lab's examples remain within individual documents and use shifted targets. Changing window length changes available context and computational work; it does not alter what a target label means.

Sources: [Hugging Face: tokenizer algorithms](https://huggingface.co/learn/llm-course/chapter6/5), [D2L: language-model datasets](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html), [PyTorch: causal attention masks](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html).

## T3. Decoder Transformer architecture

A decoder Transformer converts a token prefix into a distribution over the next token at each position. Each block updates a vector for every position; the vector width is not the vocabulary size. The small diagram uses hidden width four so the intermediate numbers can be inspected. It is not a scale model of every detail in a pretrained checkpoint.

### What happens inside one pre-norm block

Token IDs select learned embeddings. Position information distinguishes order. Layer normalization rescales the features at each position before a branch. Learned query, key, and value projections let each position compare with earlier permitted positions. Attention scores are scaled, masked, and normalised over keys; their weighted values are combined and projected back to the residual-stream width.

The first residual addition combines that branch output with the block input. Another normalization feeds a position-wise MLP, which expands features, applies a nonlinearity, and contracts them. A second residual addition preserves a direct path around the MLP. Stacking blocks repeats these transformations with different parameters. Neither residual route is generation or backpropagation.

### From representations to a token

A final normalization and vocabulary projection produce logits. Softmax over the vocabulary yields probabilities. Training compares the known successor with these scores. Generation instead selects a token from the final prefix position, appends it, and repeats with fixed weights.

Attention heads divide the representation into projected subspaces; their outputs are concatenated and mixed, not averaged as class votes. The original Transformer paper used an encoder–decoder design; this lesson demonstrates a later causal pre-norm decoder. Pretrained models can use RMSNorm, rotary positions, gated MLPs, grouped-query attention, and tied embeddings. Follow their actual equations and dimensions rather than assuming the teaching diagram's choices apply universally.

Sources: [D2L: Transformer architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html), [Vaswani et al.](https://arxiv.org/abs/1706.03762), [SLP: decoder language models](https://web.stanford.edu/~jurafsky/slp3/7.pdf).

## T4. Language-model architecture families

Architecture determines which input positions may influence an output. An encoder-only model usually builds bidirectional representations of a supplied sequence. A decoder-only language model predicts successors using a causal prefix. An encoder–decoder separately represents a source and generates a target that can cross-attend to it.

For translation, the whole source sentence can be visible while the next target token may use only the already supplied or generated target prefix. Cross-attention queries come from the decoder; keys and values come from the source representation. This differs from the decoder's causal self-attention. For complete-document classification, bidirectional access is often appropriate because the entire document exists at prediction time.

Changing modality changes the input representation and training task: a vision Transformer embeds image patches, while speech models may encode acoustic frames. Calling everything a token does not make patch, text, and audio boundaries semantically interchangeable.

Mixture-of-experts routes representations through selected learned expert branches, often replacing a feed-forward sublayer. Total parameters and active parameters per token then differ. This is not simply an ensemble of finished answers. Choose a family by permitted evidence, output requirements, and budget, then inspect the actual implementation; the name “Transformer” leaves many details unspecified.

Sources: [T5](https://arxiv.org/abs/1910.10683), [Vision Transformer](https://arxiv.org/abs/2010.11929), [Switch Transformers](https://arxiv.org/abs/2101.03961).

## T5. Gradient accumulation and stable LLM training

Gradient accumulation combines several forward/backward microbatches before one optimizer update. It helps when a desired batch does not fit in memory. It is equivalent to a larger-batch objective only when loss scaling, model behaviour, and update boundaries match; repeated optimizer steps on small batches are a different procedure.

Suppose three microbatches contain 1, 1, and 100 valid target tokens. Averaging their three mean losses gives the two tiny batches two-thirds of the vote. A token-average objective gives those two tokens only $2/102$ of the weight. Recover each summed token loss, add the sums, and divide by the total number of scored tokens. Padding and ignored prompt tokens must not enter that count.

Clip gradients after accumulating the intended update, and update the optimizer and scheduler at their defined boundary. A zero-target microbatch needs deliberate handling rather than division by zero. Compare against a single-batch reference on a small deterministic case when checking the implementation.

Mixed precision changes which numerical operations use lower precision. FP16 may need loss scaling; BF16 has a different range/precision trade-off. A master-weight copy is implementation-dependent, not a universal property of every autocast setup. If loss becomes non-finite, inspect logits, masks, reduction, data, and update scale before launching a larger run. A correct small loop is the reference for debugging a large one.

Sources: [PyTorch: gradient accumulation with AMP](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#gradient-accumulation), [Hugging Face: token-loss normalisation](https://huggingface.co/blog/gradient_accumulation).

## T6. LLM parameter counts, memory and compute

Parameter count is the number of learned entries, not the memory footprint of an experiment. A dense matrix from d inputs to h outputs has dh weights, plus h biases if used. A conventional attention block with four d×d projections has approximately $4d^2$ weights; a d→4d→d MLP has approximately $8d^2$. Gated MLPs, grouped-query projections, and tied vocabulary tables change that count.

Raw storage is entries times bytes per entry. For 100 million parameters in two-byte storage, weights alone require 200 million bytes, about 191 MiB. Training additionally needs gradients, optimizer state, activations, and temporary buffers. Inference adds cache and workspace. A model file's size is therefore not the GPU memory needed for a chosen workload.

Batch size and sequence length change activation costs. Dense attention can create a quadratic position-pair object, though memory-efficient kernels avoid storing it in the ordinary way. A rough dense-model training estimate is about 6ND FLOPs for N parameters and D tokens, with simplifying assumptions; it is not a hardware runtime prediction and can miss attention or other costs.

Measure actual peak memory, throughput, and wall time on the intended device. More parameters with too little data or too little training can be a worse use of a fixed compute budget. The small lab teaches these accounting choices; it does not reproduce large-model scaling experiments.

Sources: [Hoffmann et al.: compute-optimal training](https://arxiv.org/abs/2203.15556), [PyTorch: CUDA memory accounting](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-memory-management).

## T7. Distributed LLM training and gradient reduction

Data parallelism gives each worker a model replica and different data, then combines gradients so replicas can apply a consistent update. Tensor or pipeline parallelism instead splits parts of the computation. Sharding optimizer or parameter state changes memory ownership. These are different reasons to use several devices.

The reference objective matters before choosing a reduction. If worker A sees one token with loss 2 and worker B sees three tokens with total loss 9, the global token mean is $(2+9)/4=2.75$. Averaging their local means gives $(2+3)/2=2.5$, which weights workers equally rather than tokens equally. Uneven token counts require scaling compatible with whether the reducer sums or averages gradients.

Synchronise at the intended accumulation boundary. All replicas must agree about the update; clipping and skipped non-finite updates need coherent handling. Also partition data so workers do not unintentionally repeat the same examples, and record how checkpoint state is assembled.

The local course illustrates these calculations but does not offer multi-GPU distributed training. Use a small deterministic single-process reference when validating a distributed implementation. A speedup claim requires measured communication, compute, and workload; doubling workers does not guarantee halving time.

Sources: [PyTorch: DistributedDataParallel](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html), [PyTorch: fully sharded data parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

## T8. LLM evaluation and interpretability evidence

Begin with the comparison: what system changed, what data were fixed, and what outcome would support the claim? A prompt, tokenizer, generator version, label policy, or checkpoint change can alter the meaning of a result. A controlled ablation changes the relevant factor while preserving the intended comparison.

Token loss measures prediction of declared targets. Task accuracy, faithfulness, instruction following, latency, and security test different behaviours. Evaluate important cases and subgroups rather than treating one aggregate as general capability. Repeatedly choosing prompts using the test set makes it part of development, even when no weights are updated.

Interpretability adds a question about evidence. An attention pattern or probe can reveal an association. An intervention changes a component and measures an effect. A changed output after intervention is stronger evidence of causal influence in that setup, but does not automatically establish a unique mechanism or broad generalisation. Random-direction and location controls can help distinguish specificity from generic disruption.

Save versions, conditions, predictions, and limitations so another person can reconstruct the comparison. A fixed-prediction bootstrap addresses evaluation-sample sensitivity, not all model-training uncertainty. Model cards, introduced by Mitchell and colleagues, organise intended use and measured limitations; they do not certify safety or replace evidence.

Sources: [Mitchell et al.: model cards](https://arxiv.org/abs/1810.03993), [CheckList behavioural evaluation](https://aclanthology.org/2020.acl-main.442/), [Attention is not Explanation](https://aclanthology.org/N19-1357/).

## R1. Supervised fine-tuning and LoRA

Supervised fine-tuning trains on examples of desired behaviour, usually by predicting target response tokens. The prompt can be input without being scored. Full fine-tuning updates selected existing weights; parameter-efficient methods restrict which new or existing parameters change. Both still require correct targets, gradients, and held-out evaluation.

LoRA adds a low-rank correction to a frozen projection: $W_{effective}=W_0+(\alpha/r)BA$. For a d-input, h-output map, A has shape `[r,d]` and B `[h,r]`, adding $r(d+h)$ parameters instead of hd. With d=h=100 and r=4, that is 800 instead of 10,000 entries for this projection. The base weights still need storage and a forward computation.

A common initialisation sets one adapter matrix to zero, making the initial correction zero while permitting learning through the other matrix's nonzero values. Rank, alpha, target modules, dropout, data, and optimizer settings jointly affect the update. Higher rank is not automatically better.

Adapters can learn task patterns and factual associations, but low training loss does not prove reliable knowledge acquisition. Check both the target task and capabilities that should remain intact. Save adapters with the exact compatible base model and tokenizer. The lab's LoRA path adapts a pretrained model; it is not pretraining a new assistant from scratch.

Sources: [Hu et al.: LoRA](https://arxiv.org/abs/2106.09685), [Hugging Face PEFT: LoRA configuration](https://huggingface.co/docs/peft/developer_guides/lora).

## R2. Preference learning: DPO, PPO and GRPO

Preference learning changes which responses a language model tends to produce. A **policy** is its response distribution; a **reward** or preference label supplies an evaluation signal. Improving that signal is not necessarily improving truth or usefulness, especially when the signal is incomplete.

### DPO uses preferred and rejected examples

For the same prompt, DPO compares a chosen response with a rejected one. It increases their relative log-probability margin against a reference policy using a logistic objective. If the policy/reference ratios for the two responses are equal, the scaled margin is zero and the pair loss is $\ln2$. A larger desired margin reduces this pair loss. It does not guarantee that the preferred response's absolute probability rises in every update, because probabilities and other training examples interact. Standard offline DPO does not need on-policy generation and a separately trained reward model inside each update.

### PPO uses sampled responses and advantages

PPO samples actions from an old policy and updates a policy using an **advantage**, an estimate of how good the action was relative to a baseline. Its clipped surrogate limits the incentive for certain large probability-ratio changes; it is not a hard guarantee of a small KL divergence. In LLM RLHF, rewards and penalties can include a learned reward and a reference-policy KL term. A value model commonly supports advantage estimation.

### GRPO compares a group of responses

The original GRPO samples multiple responses for one prompt and uses their group rewards to form relative advantages, avoiding a separate learned critic in that formulation. Rewards `[1,1,0,0]` have mean 0.5 and population standard deviation 0.5, giving standardised advantages `[1,1,-1,-1]`. State the variance convention and epsilon. Equal rewards remove this relative signal for that group, but other groups, KL terms, or optimizer state may still affect updates.

All methods depend on data quality and evaluation. A KL penalty can restrain drift but does not guarantee prevention of reward hacking. The application demonstrates objective arithmetic; it does not run DPO, PPO, or GRPO training loops.

Sources: [Rafailov et al.: DPO](https://arxiv.org/abs/2305.18290), [Schulman et al.: PPO](https://arxiv.org/abs/1707.06347), [Shao et al.: GRPO in DeepSeekMath](https://arxiv.org/abs/2402.03300).

## R3. Prompting, reasoning and output decoding

A prompt supplies instructions, examples, and context to a model whose weights remain fixed during ordinary inference. Few-shot examples can change how it interprets the task, called in-context learning. This is legitimate learning terminology, but it is not an optimizer updating model parameters.

The model produces logits for the next token. Positive temperature T rescales them before softmax: $p_i\propto e^{z_i/T}$. For logits `[2,0]` at T=1, probabilities are about `[0.881,0.119]`. Lower positive temperature sharpens the distribution; it does not verify factuality. Greedy decoding chooses the highest score. Top-k and top-p restrict sampling candidates under different rules and also do not prove correctness.

Asking for reasoning or a structured intermediate record can change task performance. A fluent explanation may nevertheless be irrelevant, unsupported, or a post-hoc account. Compare direct prediction, free-form reasoning, and constrained structure under matched inputs and evaluation before attributing a gain to reasoning itself.

Validate final outputs against both schema and task meaning. JSON validity prevents one class of parsing failure, not hallucinated facts. Repeatedly tuning prompts on evaluation answers consumes that set as development evidence even when the model weights never move.

Sources: [SLP: generation and decoding](https://web.stanford.edu/~jurafsky/slp3/7.pdf), [Holtzman et al.: neural text degeneration and nucleus sampling](https://arxiv.org/abs/1904.09751).

## R4. Retrieval-augmented generation and vector indexes

RAG retrieves external information and conditions generation on it. Its attraction is that evidence can be updated or inspected without retraining all model weights. Its weakness is that failures can occur at retrieval, evidence selection, generation, or validation; adding a vector database does not remove hallucination.

Index documents with provenance, version, boundaries, and access permissions. At query time, retrieve eligible candidates using lexical, dense, or hybrid signals. Optionally rerank them, then assemble a prompt within a token budget. Chunk size trades local detail against context and retrieval granularity; overlap repeats material and must not be mistaken for independent evidence.

Evaluate retrieval recall separately from answer support. If the relevant passage is missing, the generator may lack the needed information. If it is present, the generator may still ignore a qualification or cite a nearby unsupported passage. A valid citation URL is not proof that its contents support the claim.

Keep source text untrusted as instructions. A retrieved document can contain malicious requests or stale data. Enforce permissions and tool boundaries outside generation, and abstain when evidence is insufficient. Record the index and retrieved evidence used for a result so comparisons remain reproducible.

Sources: [Lewis et al.: RAG](https://arxiv.org/abs/2005.11401), [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906), [OWASP: prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).

## R5. Tool-using agents and MCP boundaries

A tool-using system allows a model to propose actions that a host application may execute. The model's proposal, schema validation, permission check, actual execution, and confirmed result are separate stages. Keeping them separate prevents a plausible sentence from being mistaken for a completed action.

Suppose the model proposes sending an email. A schema can verify that recipient, subject, and body fields exist. It cannot decide whether the user authorised the recipient, whether the content leaks private data, or whether sending is appropriate. The host must enforce those rules and obtain approval when required before calling the tool.

MCP standardises exchanges among a host, clients, and servers exposing tools, resources, and related capabilities. Discovery tells the client what a server advertises; it is not a transfer of unlimited authority. Authenticate services, restrict accessible resources, and validate returned data. An error response is evidence of failure, not permission to pretend completion.

For retries, distinguish safe repetition from duplicate side effects. Reading a document again is different from charging a payment twice. Use bounded loops, deadlines, budgets, idempotency where appropriate, and durable state around approvals. A multi-agent arrangement adds handoff boundaries; it does not remove these responsibilities.

Sources: [MCP architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture), [MCP security guidance](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices).

## R6. LLM inference, KV caching and serving

Serving turns fixed model weights into responses under a workload. **Prefill** processes the prompt and builds intermediate key/value states. **Decode** repeatedly generates subsequent tokens. A KV cache reuses earlier attention keys and values rather than recomputing the entire prefix at every step. It consumes memory but does not store an optimizer or learn from the conversation.

Approximate raw cache storage is $2BLTH_{kv}d_hs$ bytes: keys plus values, batch B, layers L, cached positions T, KV heads, head width, and bytes per entry. Metadata, allocation policy, padding, and temporary buffers add overhead. Grouped-query attention reduces the number of KV heads relative to query heads.

If a prefix changes, dependent cached positions must be recomputed; an unchanged leading prefix may still be reusable. Cache validity also depends on model state and positional conventions. A cached response and a KV cache are different mechanisms.

Measure time to first token, per-token latency, throughput, and memory under declared prompt/output lengths and concurrency. Batching can improve throughput while affecting latency. Queueing can dominate first-token delay, so no one latency relationship is universal. Quantisation, kernel support, and hardware matter; a smaller model is not guaranteed faster under every serving configuration.

Sources: [Hugging Face: KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation), [vLLM: paged attention paper](https://arxiv.org/abs/2309.06180).

## R7. LLM quantization, compression and deployment

Quantisation represents numerical values with fewer bits using a declared mapping. In an affine scheme, a real value is rounded to an integer code with a scale and zero point, then clipped to the supported range. Dequantisation reconstructs an approximation, not the original value exactly.

With scale 0.1 and zero point 0, value 0.26 rounds to code 3 and reconstructs as 0.3. Its absolute error is 0.04. Values outside the code range clip, potentially causing larger error. Per-channel or grouped scales can reduce error for uneven distributions but add metadata. A four-bit weight format is not automatically a fourfold reduction in total runtime memory relative to sixteen-bit training.

Post-training quantisation transforms an already fitted model, sometimes using calibration data. Quantisation-aware training exposes numerical effects during fitting. QLoRA trains adapters through a quantised frozen base under its specific design; it is not the claim that every training tensor becomes four-bit.

Check task performance, rare-case regressions, supported kernels, speed, and actual memory. Compression can preserve average behaviour while damaging a sensitive task. Package tokenizer, model configuration, licence obligations, and runtime requirements as carefully as weight files; a compact file that cannot run reproducibly is not a successful deployment.

Sources: [QLoRA](https://arxiv.org/abs/2305.14314), [Hugging Face: quantisation overview](https://huggingface.co/docs/transformers/quantization/overview).

## R8. LLM system evaluation, security and operations

An LLM application's success includes the task outcome and the path taken to reach it. A correct final answer reached by exposing private records or skipping approval is still a system failure. Evaluate tools, retrieval, permissions, and costs as well as generated language.

Build cases for expected tasks and critical failures. A model judge can help scale some assessments, but its prompt, biases, and agreement with human criteria require checking. Do not replace an objective tool-success record or source-support check with whether another model likes the answer. Regression tests should include boundary cases and allowed alternative paths, not insist that every valid agent follow one exact trajectory.

Prompt injection tries to turn untrusted content into authority. Separating instructions and data helps, but is not a guarantee. Enforce resource access, action allowlists, approval, and sandbox boundaries outside the model. Redact secrets in logs and traces while retaining enough provenance to debug failures.

Set iteration, token, latency, and cost limits before executing a loop. Retry transient failures within those bounds; stop on permission failures rather than silently bypassing them. Roll out a change with observable evidence and a rollback path. Shadow evaluation must not duplicate external side effects or leak user data. Good operations make failures visible and recoverable, not impossible by declaration.

Sources: [OWASP: prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/), [NIST: Generative AI risk profile](https://doi.org/10.6028/NIST.AI.600-1).

## R9. LLM research methods and activation steering

Activation steering changes internal values during inference rather than merely rewriting the input or fine-tuning parameters. A simple intervention adds a direction v at a specified layer and position: $h'=h+\alpha v$. The strength alpha, location, and construction of v are parts of the method, not incidental settings.

For a fixed linear readout $s=w^Th$, adding the direction changes the score by $\alpha w^Tv$. If w and v are perpendicular, this particular linear score does not change. In a full nonlinear network, later computation can alter the effect, so a local calculation is not a guarantee about generated behaviour.

Construct directions using an explicit dataset and separation from evaluation. Compare no intervention, several strengths, random or irrelevant directions, and different locations. Measure the target behaviour and side effects such as fluency, refusal, factuality, or task accuracy. A direction that changes a probe's score may not change the intended semantic behaviour.

Distinguish observing an association, predicting a property with a probe, and intervening on an activation. A successful intervention supports influence in that setup, not a unique human-like concept representation. Repeat across prompts and domains before claiming transfer. The lab's steering arithmetic is a teaching example, not a trained steering model or evidence of universal control.

Sources: [Turner et al.: activation engineering](https://arxiv.org/abs/2308.10248), [Geiger et al.: causal abstractions of neural networks](https://arxiv.org/abs/2106.02997).
