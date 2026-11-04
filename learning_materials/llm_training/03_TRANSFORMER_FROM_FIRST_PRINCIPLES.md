# T3. Decoder Transformer architecture

A decoder Transformer predicts a token from a permitted prefix by repeatedly transforming one vector per input position. Attention mixes information between allowed positions, while feed-forward layers transform features within each position. Residual connections preserve direct paths around these transformations. Training compares predictions with known targets; generation instead chooses and appends tokens with fixed parameters.

The walkthrough uses a small supplied vocabulary and fixed weights: hidden width four, additive positions, LayerNorm and a 4→8→4 GELU feed-forward branch. Select a head, position or block to inspect the actual arithmetic. This is not the architecture or learned competence of the locally prepared pretrained model.

The data lesson produced aligned token inputs. This walkthrough follows their full computation while keeping tensor axes, residual paths and attention/vocabulary normalizations distinct. The deeper notes separately explain real grouped-query dimensions and other positional methods.

## 1. Decoder input and output

**Key terms.** Here **B** is batch size, **T** sequence length, **d** hidden width and **V** vocabulary size. A **hidden state** is the vector representing one position inside the model.

**Process.** IDs enter an embedding lookup, blocks refine the resulting vectors, and a vocabulary projection produces logits. Training sends them to a loss; generation instead selects a token and repeats.

```text
text → tokenizer → token IDs [B,T]
→ embedding lookup [B,T,d]
→ positional information + repeated Transformer blocks
→ final normalization → vocabulary projection [B,T,V]
→ logits → probabilities → selected next token
```

`B` is batch size, `T` sequence length, `d` hidden width and `V` vocabulary size. An embedding table `[V,d]` maps each integer ID to a row. During training, loss compares the vocabulary scores at a position with the next target token. During inference, the selected token extends the input.

We explain a common pre-normalized decoder block. Exact models differ in normalization, biases, positions, activations and attention variants. The original 2017 Transformer was encoder–decoder, not this exact modern decoder recipe.

## 2. Query, key and value

**Key terms.** A **query Q** is a matching request; a **key K** is what it matches against; **values U** are mixed information. **dₖ/dᵥ** are head widths; **M** is the attention mask.

**Process.** Token states enter three learned projections to produce Q, K and U. Query–key products become masked, normalized attention weights; those weights combine U and pass a new representation to the next operation.

Given token representations `X:[T,d]` for one sequence, compute:

\[
Q=XW_Q,\quad K=XW_K,\quad U=XW_V.
\]

We call the value matrix `U` here to avoid confusing it with vocabulary size `V`. For a single head, projections yield `[T,d_k]`, `[T,d_k]`, `[T,d_v]`.

A **query** represents what a position seeks; a **key** represents how a position can match that request; a **value** carries information to aggregate. These are useful analogies, not literal questions stored in the weights.

Attention is:

\[
S=QK^\top/\sqrt{d_k}+M,\quad
A=\operatorname{softmax}_{row}(S),\quad O=AU.
\]

`S:[T,T]` contains pairwise scores. `M` is zero for allowed positions and negative infinity for forbidden positions. Each row of `A` sums to one over allowed keys. Scaling controls score growth with feature width under useful initialization assumptions. Softmax operates over keys for each query, not across every entry of the matrix at once.

## 3. Calculate one attention row

**Key terms.** An **attention row** holds one query's weights over keys; **softmax** normalizes allowed scores to sum to 1; a **weighted sum** combines values using those weights.

**Process.** Compute dot products, divide by the square root of key width, mask forbidden keys, then apply softmax. Multiply the resulting row by the value vectors to obtain this query position's output.

Let the current query be `[1,0]`, two permitted keys be `[1,0]` and `[0,1]`, and values be `[2,0]` and `[0,4]`. With `d_k=2`, scaled scores are `[1/√2,0]≈[0.7071,0]`. Softmax gives approximately `[0.6698,0.3302]`.

The weighted value is:

```text
0.6698 × [2,0] + 0.3302 × [0,4] = [1.3395,1.3210]
```

Attention produced a mixture, not a selected source token. If the second key is causally forbidden, its probability becomes zero and the output becomes `[2,0]`. Mask **before** softmax; setting an output probability to zero afterwards without renormalizing is not the same calculation.

The static attention diagram shows this calculation's complete path. Compare the allowed two-key result with the one-key masked result above; no playback or parameter slider is required.

## 4. Multi-head attention and shapes

**Key terms.** An **attention head** is one learned projection subspace; **h** counts heads; **concatenation** joins features; **Wₒ** projects joined outputs back to hidden width.

**Process.** Split projected features into heads, compute each head's attention, join their outputs and project back. That final width must match the residual stream before the block can add the result.

With `h` heads, projected features are reshaped to `[B,h,T,d_k]`. Each head computes its own interactions. Concatenate head outputs, then apply an output projection `W_O` to return to `[B,T,d]`. In a common choice, `h d_k=d` and `d_v=d_k`.

Multiple heads give multiple learned interaction subspaces. They do not have guaranteed roles such as “grammar head” and “truth head.” Some behaviours may be studied empirically, and redundancy can occur.

For `B=2,T=4,d=8,h=2`, each head width is 4. Scores have shape `[2,2,4,4]`; concatenated outputs have shape `[2,4,8]`. Distinguish number of tokens from feature width at every reshape.

## 5. One decoder block

**Key terms.** **Norm** rescales each token's features; **MLP** is a feed-forward network; a **residual** adds the input back; **d_ff** is the MLP's intermediate width.

**Process.** Normalize the input, apply attention and add its residual. Normalize that result, apply the MLP and add the second residual. The unchanged outer shape lets the next decoder block repeat this sequence.

A common pre-norm block is:

\[
H=X+\operatorname{Attention}(\operatorname{Norm}(X)),
\qquad Y=H+\operatorname{MLP}(\operatorname{Norm}(H)).
\]

The attention mixes positions. The MLP applies a learned nonlinear transformation separately at each position using shared weights. Residual additions preserve the shape. For a simple MLP, expand width `d→d_ff`, activate, then contract `d_ff→d`. Gated variants use another branch.

The residual stream is the evolving vector at each position across layers. Adding two same-shape tensors is not concatenation. Normalization generally acts over features per token, not over the batch's labels.

### Read a real grouped-query block by its dimensions

The locally prepared SmolLM2-360M configuration has width `d=960`, 15 query heads, 5 KV heads and head width `960/15=64`.

1. For input `[B,T,960]`, the query projection produces `[B,T,960]`, then reshapes to `[B,15,T,64]`.
2. Each key and value projection produces only `5×64=320` features: `[B,5,T,64]` after reshaping. Three query heads share each KV head. A kernel need not physically store three copies of the same cache entry.
3. Rotary position transformations modify Q/K coordinate pairs before their dot products; they are not learned absolute vectors added to the token embedding in this model.
4. Attention returns 15 head outputs and an output projection returns width 960. A gated feed-forward branch then uses intermediate width 2560 and projects back to 960.
5. The gated branch has three matrices: two input projections and one down projection. In row convention it is `SiLU(XW_gate) ⊙ (XW_up)`, followed by `W_down`. Here `⊙` is elementwise multiplication, not another matrix contraction.

Both residual additions preserve `[B,T,960]`. Check exact architecture fields instead of assuming every modern model uses a four-times-width GELU MLP. Sources: [local model's upstream configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json), [GQA](https://arxiv.org/abs/2305.13245), [RoPE](https://arxiv.org/abs/2104.09864) and [gated feed-forward variants](https://arxiv.org/abs/2002.05202).

## 6. Position information

**Key terms.** A **position embedding** represents token location; **RoPE** rotates query/key coordinate pairs; **relative offset** is the distance between positions.

**Process.** Supply position information before or within token interactions. In RoPE models, transform Q and K before their dot products; the resulting scores can depend on relative position as well as content.

Without positional information, plain self-attention has strong permutation symmetry: it cannot generally distinguish reordered tokens in the desired way. Absolute positional embeddings add position vectors. Relative methods influence interactions based on distance. Rotary embeddings rotate pairs of query/key coordinates as a function of position, allowing their dot products to encode relative offsets. A causal mask adds directionality but does not replace every use of explicit position design.

Longer nominal context does not ensure reliable use of distant evidence. Position extrapolation, training-length distribution, distractors and retrieval all affect performance.

## 7. Output and generation

**Key terms.** A **vocabulary projection** maps hidden features to token scores; **weight tying** shares compatible input/output weights; the **KV cache** retains earlier keys and values.

**Process.** The final hidden state becomes vocabulary logits. Select a next token, append it, reuse cached earlier state where supported, and repeat until the stop condition.

After the blocks, a linear projection maps each hidden state to `V` logits. Weight tying can share embedding and output weights under compatible shapes. During generation, use the last relevant position's distribution, choose a token, append it and repeat until a stop rule or budget is reached.

The **KV cache** stores earlier keys and values so they need not be recomputed for every generated token. It is temporary inference state, not new model knowledge. R6 derives its memory cost.

## 8. Limits of attention visualization

**Key terms.** An **attention heatmap** displays mixing weights; a **causal intervention** changes computation to test an effect; a **control** rules out alternative explanations.

**Process.** Inspect a heatmap to form a hypothesis, then intervene under matched conditions and measure behaviour. Do not jump from a dark cell to a claim about why the complete model answered correctly.

An attention heatmap shows a component of computation, not a complete causal explanation. Residuals, MLPs, many heads and multiple layers all contribute. High weight on a passage does not establish that the final answer faithfully uses it. Intervention tests need carefully matched controls.

## Exercises and checks

1. Why is `QKᵀ` `[T,T]`? Each query compares with each key; the inner feature axis is summed out.
2. During causal training, can token 2 attend to token 4? **No**, even though token 4 is present in the batch.
3. What changes at inference versus training? Gold next tokens are not supplied; generation uses chosen outputs; cached state can avoid repeated work; dropout/gradient tracking settings differ.
4. Reproduce the numerical attention row above within 0.001 and explain every axis.

Primary sources: [original Transformer](https://arxiv.org/abs/1706.03762), [GQA](https://arxiv.org/abs/2305.13245), [RoPE](https://arxiv.org/abs/2104.09864) and [RMSNorm](https://arxiv.org/abs/1910.07467). The numerical attention example is illustrative; the separately labelled checkpoint inspector displays actual model values.
