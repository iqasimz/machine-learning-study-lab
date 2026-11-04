# DL08. Attention and Transformer building blocks

Attention updates a representation by taking a weighted combination of value vectors. A query is compared with keys to calculate matching scores; after masking forbidden positions, softmax turns those scores into mixing weights. The values supply the content being combined. Attention weights are not automatically explanations of truth or model correctness.

We update one query using two candidate key/value pairs. Allow both positions or block the later one with a causal mask, then see how the allowed scores, weights and output change. The model weights and vectors are authored examples, not trained measurements.

Recurrent cells carry information through a sequence state. Attention provides another way to combine information across positions. The LLM Transformer lesson shows how heads, projections, residuals and feed-forward layers surround this operation.

[course map](../COURSE_MAP.md). Required background: DL07, P2.

## 1. Matching and mixing across positions

**Key terms.** A **query Q** asks for a match; a **key K** supplies matching features; a **value V** carries mixed content; **attention weights** normalize allowed matches. A **causal mask** excludes future input; **position information** represents order.

**Process.** Project token/sequence features into Q/K/V, compare queries with permitted keys, normalize scores and mix values. Multi-head outputs return to hidden width, pass through residual/feed-forward structure and reach the task head.

Attention permits content-dependent interactions across positions without a recurrent state transition for every input step. This can parallelize training positions, but exact dense attention can involve quadratic pair interactions. Efficient kernels change memory traffic without necessarily changing the function. A causal mask still matters when every training token is stored in the same batch.

Transformers add position design, feed-forward layers, residuals and normalization. They are not attention alone. Encoder, decoder and encoder–decoder forms permit different information flow. The existing T3 lesson supplies the complete modern decoder and real GQA shapes; this lesson is the bridge from general DL, not a duplicate full LLM textbook.

## 2. A worked calculation

**Symbols and assumptions.** For one head, S=QK^T/sqrt(d_k)+M; row-wise softmax gives A; output O=AV. d_k is key width and M is 0 on allowed scores and negative infinity on blocked ones.

With query [1,0], allowed keys [1,0]/[0,1], key width 2 and values [2,0]/[0,4], scores are [0.7071,0]. Softmax weights are approximately [0.6698,0.3302], producing [1.3395,1.3210]. Blocking the second key before softmax instead gives output [2,0]. A large weight is a mixing coefficient, not truth confidence.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Attention and architecture alternatives

Compare recurrence's compact evolving state with attention's position-specific interactions/cache. A small Transformer can be worse than a simpler model when data, compute or task structure differ. Head count at fixed hidden width changes per-head width; it does not automatically increase total representation width.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare allowed information

Use a tiny authored sequence task to verify future-token invariance, then continue into T3 and T0. No new language-model training backend is required to write this bridge.

**Check.** Does masking only the loss on a future token stop earlier positions attending to it?

**Answer.** No. The attention mask controls information flow; the loss mask controls scoring. They must be constructed and tested separately.

## 5. Attention dimensions, masking and memory

**Key terms.** Queries ask which keys to read. Keys determine match scores; values carry the content mixed by those scores. A mask excludes unavailable positions before normalization.

**Process.** Project input vectors into Q, K and V, score permitted position pairs, scale by the square root of key width, normalize each query row and combine value vectors. Concatenate head outputs and project them onward.

In these equations, T is the number of query positions, S the number of key/value positions and d_k the width used to compare a query with a key. Each row of A describes one query's weights over S possible sources. Multiplying A by V combines source content into output features.

\[
A=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M),\qquad O=AV.
\]

With Q shaped B×T×d_k and K shaped B×S×d_k, the score matrix is B×T×S. Each row sums to one after softmax over S permitted keys. Masked scores behave like negative infinity, yielding zero weight. An entirely masked row needs special handling rather than an undefined softmax.

### A training-system consequence

Standard attention materializes a quadratic position matrix when T=S. Memory-efficient kernels change the computation order to avoid storing that full matrix; they are not a different semantic license to read future tokens. Positional information remains necessary because content-only self-attention cannot distinguish arbitrary input order by itself.

The existing LLM course supplies the full Transformer walkthrough and the scratch trainer. The image lab's CNN does not become a Transformer simply because both architectures use matrices and backpropagation.

## Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [Dive into Deep Learning](https://d2l.ai/).
