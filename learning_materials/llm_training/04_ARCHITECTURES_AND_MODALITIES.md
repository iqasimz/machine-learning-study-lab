# T4. Language-model architecture families

An architecture defines which representations interact and what state is carried through a model. Encoders, causal decoders and encoder–decoders differ in information access and output interface; expert layers and state-space operations change other parts of the computation. These categories can overlap within one model, so architecture names are not an ordered quality ranking.

Select a family to inspect its structure. The encoder–decoder example has three source positions and two target-prefix positions. Separate supplied examples illustrate two weighted experts or a scalar recurrent state; none is a trained comparison between model families.

The detailed decoder provided one concrete architecture. This lesson varies the structure while keeping visibility, shape, state, active computation and cache costs distinct, with positional and multimodal extensions in the deeper notes.

## 1. Architecture design choices

**Key terms.** An **architecture** defines the computation graph; **dense** layers use the full relevant matrix; **sparse** computation activates only selected components.

**Process.** Classify what a design changes—visibility, state, computation or input modality—before comparing its scores. That classification determines which costs and controls the experiment must measure.

Do not memorise a list of model brands. Ask: which positions interact, what state is retained, how many parameters are active, and what objective trains the representation?

| Family | Main property | Main trade-off |
| --- | --- | --- |
| Encoder-only | Bidirectional input representation | Strong classification/retrieval interface, not ordinary left-to-right generation |
| Decoder-only | Causal continuation | Flexible generation; sequential output steps |
| Encoder–decoder | Separate source processing and target generation | Natural conditional-generation interface; two interacting stacks |
| Sparse/sliding attention | Restricted token interactions | Longer sequences cheaply; some global connections need special handling |
| Recurrent/state-space | Compress history into state | Efficient long sequences; different access to exact past content |
| Hybrid | Mix attention and other sequence operations | Balance benefits, but more complex implementation and evaluation |
| Diffusion/masked-generation language models | Iterative refinement rather than only left-to-right generation | Alternative parallel/refinement opportunities; objective and serving differ |

These categories overlap. A mixture-of-experts decoder can also use grouped-query and sliding-window attention.

## 2. Positional methods

**Key terms.** **Absolute positions** label locations; **relative positions** describe offsets; **RoPE** uses rotations; **extrapolation** uses lengths beyond the training regime.

**Process.** Encode position in the chosen way before attention computes interactions. Test changed lengths next, because a larger configured window does not guarantee reliable use of distant information.

Learned absolute positions assign a vector per position. Sinusoidal positions use fixed functions. Relative biases modify pair scores using distance. ALiBi uses head-dependent linear distance biases. RoPE rotates query/key coordinate pairs.

For a 2D pair, a rotation is:

\[
R(\phi)\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}a\cos\phi-b\sin\phi\\a\sin\phi+b\cos\phi\end{bmatrix}.
\]

RoPE applies position-dependent angles at several frequencies. Rescaling positions/frequencies can extend supported lengths, but changes the model's operating conditions. Test long-context retrieval and reasoning, not only whether the runtime accepts a larger number.

## 3. MHA, MQA, GQA and latent attention

**Key terms.** **MHA** has separate keys/values per query head; **MQA** shares one set; **GQA** shares groups; **latent attention** stores a compressed representation.

**Process.** Project hidden states into the selected query/key/value layout, compute attention and retain the needed cache. The sharing choice changes projection and cache costs, not automatically the allowed token pairs.

**Calculation.** At head width 64, 15 query heads with 15 KV heads need 2×15×64=1,920 cached values per token per layer. Sharing 5 KV heads needs 2×5×64=640: one third as many before dtype and runtime overhead.

Multi-head attention uses distinct key/value projections per head. Multi-query attention shares keys/values across query heads. Grouped-query attention shares them within groups. This reduces KV-cache storage and bandwidth at possible quality/architecture trade-offs. Query-head count is not the KV-head count in the memory equation.

Latent-attention approaches compress stored information and reconstruct needed components. They change how cache memory is represented; do not apply a standard uncompressed KV formula blindly to them. Implementation compatibility matters as much as nominal architecture support.

### Separate three ways of reducing attention cost

They change different objects and cannot be compared using one headline memory ratio.

1. **GQA/MQA** reduce the number of stored key/value heads. The number of query positions and allowed token interactions can remain unchanged.
2. **Sliding-window or sparse attention** removes some query–key connections. A token may no longer directly access evidence outside its permitted window; special global paths or multiple layers change that access pattern.
3. **Latent KV representations** compress the cached information. Reconstructing or absorbing projections into other operations changes the runtime arithmetic; the ordinary uncompressed cache formula no longer describes the complete implementation.
4. **FlashAttention** reorganizes the same dense attention operation to reduce IO/materialization. It need not remove interactions or change head sharing.

For RoPE, the identity `(R_m q)ᵀ(R_n k)=qᵀR_(n−m)k` explains why a relative position offset enters the score: rotations preserve lengths and compose by angle differences. It does not prove that a model trained on short sequences can use arbitrarily long contexts.

Compare quality on long evidence retrieval as well as bytes, throughput and latency. Relevant original methods: [GQA](https://arxiv.org/abs/2305.13245), [RoFormer](https://arxiv.org/abs/2104.09864), [FlashAttention](https://arxiv.org/abs/2205.14135) and [DeepSeek-V2 latent attention](https://arxiv.org/abs/2405.04434).

## 4. Mixture of experts

**Key terms.** A **mixture of experts (MoE)** has alternative learned sub-networks; a **router** selects them; **active parameters** are those used for a token.

**Process.** Each token reaches the router, which assigns selected expert computations. Combine their outputs and continue the block; measure routing balance and communication as well as total parameter count.

An MoE layer has several expert networks and a router. For input representation `x`, router scores select a subset; output combines selected expert outputs:

\[
y=\sum_{e\in TopK(g(x))}a_e(x)f_e(x).
\]

Sparse activation increases total available parameters without evaluating every expert for every token. Total parameters still occupy storage or require offloading. Routing adds communication, load imbalance and capacity decisions. Load-balancing losses discourage all tokens selecting the same expert; overflow policies may drop or reroute tokens.

**Routing distinction:** choosing one complete classifier per dataset is a system-level routing decision. Token-level MoE routes internal hidden states through expert sublayers. Shared terminology does not make them the same implementation.

## 5. State spaces and alternative sequence models

**Key terms.** A **state-space model** updates a compact sequence state; **recurrence** carries that state forward; an **attention cache** instead retains token-specific keys and values.

**Process.** A sequence model consumes earlier information through its chosen state mechanism. Compare how that state updates and scales before assuming attention's cache and cost formulas apply.

A simplified state-space recurrence is `h_t=A h_(t−1)+B x_t`, `y_t=C h_t`. Modern selective/state-space models make parts input-dependent and use efficient scans. The finite state summarises prior input differently from attention's explicit access to stored token keys/values. This can improve certain length/resource trade-offs but is not a universal replacement for exact retrieval.

Diffusion language models iteratively refine corrupted/masked tokens. They involve a corruption schedule, denoising objective and generation schedule. Distinguish discrete token diffusion from image diffusion; architecture and likelihood accounting vary. This course surveys the branch; deriving each diffusion estimator requires the specialist route in the resource guide.

## 6. Multimodal language systems

**Key terms.** A **modality** is an input type such as text or audio; an **encoder** turns it into features; a **projector** maps features into a compatible space.

**Process.** Encode the non-text input, align its representation with the language system and fuse it through the chosen architecture. Test whether answers use that input rather than a text-only shortcut.

A vision-language system may encode image patches, map them through a projector and combine them with text representations. Audio systems can use acoustic encoders, discrete audio tokens or streaming representations. Video introduces time, frame sampling and synchronization. OCR is not equivalent to complete image understanding.

**Contrastive alignment** learns matched text/image representations; **generative alignment** learns to produce text conditioned on another modality; **joint training** may update several components. A frozen encoder plus learned projector differs from end-to-end multimodal pretraining.

Real issues: unreadably small images, incorrect crop/resizing, missing frames, audio accents, hallucinated chart values, private biometric data and poor grounding. Evaluate whether the answer is supported by the modality, not merely plausible from text priors.

## Exercises and checks

1. An MoE has 64 experts and activates 2. Does it need storage only for 2 experts? **No**, unless a separately designed loading/offloading arrangement provides others when needed.
2. Does GQA reduce all model memory by the query/KV head ratio? **No:** it mainly changes affected attention components and cache, not every weight/activation.
3. Why might a long-context state model still fail an exact quotation query? Compressed state can lose exact content; test the task instead of inferring capability from complexity.

Primary examples: [Switch Transformers](https://arxiv.org/abs/2101.03961), [Mamba](https://arxiv.org/abs/2312.00752), [LLaMA](https://arxiv.org/abs/2302.13971), [Gemini technical report](https://arxiv.org/abs/2312.11805). These establish example families, not a current leaderboard.
