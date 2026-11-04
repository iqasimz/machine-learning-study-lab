# Equations and terms: a compact reference

Read the lessons before using this as a revision sheet. A remembered equation without its denominator, units or assumptions is easy to misuse.

## 1. Notation

| Symbol | Meaning in this course | Caution |
| --- | --- | --- |
| `B` | Batch/concurrent sequence count | Training and serving batches differ |
| `T` | Sequence/token length | In distillation, temperature is explicitly named instead |
| `V` | Vocabulary size | T3 calls the attention value matrix `U` to avoid collision |
| `d` | Hidden representation width | Not sequence length |
| `h`, `h_kv` | Query heads, KV heads | Can differ under GQA/MQA |
| `N` | Parameter count or sample count as locally defined | Always read local definition |
| `D` | Training-token exposure in scaling equations | Not automatically unique tokens |
| `θ` | Model/policy parameters | Distinct from temporary activations |
| `πθ` | Policy distribution | A distribution, not one generated answer |
| `η` | Learning rate | Update units matter |
| `β` | Objective-specific coefficient | DPO/PPO/Adam symbols do not all mean the same thing |
| `ε` | Small numerical term or clipping/privacy parameter | Local definition is essential |

## 2. Core calculation sheet

| Quantity | Expression | Use and limitation |
| --- | --- | --- |
| Dot product | `Σxᵢyᵢ` | Magnitude affects the result |
| Cosine | `(x·y)/(||x||||y||)` | Undefined for a zero vector |
| Softmax | `exp(zᵢ−max z)/Σexp(zⱼ−max z)` | Converts scores, not truth, to probabilities |
| Sequence log likelihood | `Σ_t log p(x_t|x_<t)` | Conditional on template/context/tokenizer |
| Token loss | `−log p(target)` | Zero target probability gives infinite ideal loss |
| Perplexity | `exp(mean valid-token loss)` | Tokenizer/corpus dependent |
| Entropy | `−Σp log p` | Uncertainty of a distribution |
| KL | `Σp log(p/q)` | Asymmetric, support-sensitive |
| SGD | `θ←θ−ηg` | Local gradient step, not global guarantee |
| Softmax/CE gradient | `p−y` | One-hot/categorical CE setup |
| Attention | `softmax(QKᵀ/√d_k+M)U` | Row-wise softmax; mask before softmax |
| LoRA parameters | `r(d_in+d_out)` | For one adapted matrix |
| Dense training FLOPs | approximately `6ND` | Rough, architecture/context dependent |
| Raw weight bytes | `N×bits/8` | Excludes metadata, cache, workspace |
| Conventional KV bytes | `2LBTh_kv d_head×bytes_per_value` | Equal-length uncompressed cache assumption |
| Precision/recall | `TP/(TP+FP)`, `TP/(TP+FN)` | Define zero-denominator handling |
| Positive F1 | `2TP/(2TP+FP+FN)` | Not automatically macro-F1 |
| Little's law | `L=λW` | Appropriate stable queueing conditions |

## 3. Explain these pairs aloud

- **Parameter / hyperparameter:** learned number / externally selected design or training setting.
- **Forward / backward:** compute outputs / propagate gradients of a scalar objective.
- **Activation / checkpoint:** input-dependent temporary computation / saved model and possibly training state.
- **Pretraining / SFT:** broad language objective / adaptation to desired demonstrations; objectives can share next-token mathematics.
- **Training / inference:** change weights from a learning signal / compute using selected weights.
- **Context / memory / learned knowledge:** current input / stored application state / information represented in parameters.
- **Token / word / byte:** tokenizer unit / linguistic unit / encoding unit.
- **Logit / probability / confidence:** raw score / normalized distribution value / claim-specific uncertainty estimate that needs a definition.
- **Attention mask / loss mask:** permitted input visibility / scored target positions.
- **Validation / test:** selection and development / final held-out estimation after freezing choices.
- **Error / uncertainty:** observed incorrectness / limited knowledge about an outcome or estimate.
- **Aleatoric / epistemic uncertainty:** variability/ambiguity in the specified process / uncertainty due to limited knowledge or model; decomposition is modelling-dependent.
- **Factuality / faithfulness:** true in the relevant world / supported by the specified source. Neither implies the other automatically.
- **Alignment / safety / security:** matching intended objectives/preferences / avoiding unacceptable harm / resisting adversarial misuse and unauthorized access.
- **Authentication / authorization:** who calls / what they may do.
- **RAG / fine-tuning:** retrieve information into input / update model parameters.
- **Embedding / vector database:** representation / storage/search/management system using such representations.
- **ANN recall / relevance recall:** recover metric nearest neighbours / recover truly useful evidence.
- **Tool calling / MCP:** proposed structured external action / protocol standard for exposing/connecting capabilities.
- **Agent / workflow:** policy-driven action selection / primarily predetermined orchestration; not a strict binary.
- **RL reward model / critic:** scores behaviour / predicts expected return for advantage estimation.
- **Reference / old rollout policy:** regularization anchor / policy that produced sampled actions for importance ratios.
- **PPO / DPO:** policy-gradient-style rollout optimization / standard direct preference-pair objective.
- **GRPO / MoE:** group-relative policy optimization / sparse internal expert routing. Unrelated acronyms.
- **Quantization / LoRA:** numeric compression / low-rank adaptation. QLoRA combines particular forms.
- **FlashAttention / PagedAttention:** attention IO computation / serving KV-cache organization.
- **Latency / throughput:** time per request/step / completed work per time.
- **Replica / shard:** copy serving more work / partition of state or computation.
- **Open weights / reproducible model:** available parameters / sufficiently available recipe, data, code and state to reproduce under stated conditions.
- **Abstention / warning:** no decision under a policy / a limitation accompanying an output, if the schema defines it that way.

## 4. The universal answer pattern

For an interview or real design question:

> “The goal is ___. The mechanism changes ___. It assumes ___. The main cost is ___. It can fail when ___. I would compare it with ___ and measure ___ under a matched budget.”

Example: “RAG supplies current source evidence without necessarily retraining the model. It needs authorised, well-parsed and retrievable documents. It can fail in retrieval, context construction or evidence use. I would compare direct generation, lexical retrieval and dense/hybrid retrieval, then score support and answerability separately from latency.”

This is a structure for your own reasoning, not a script that substitutes for understanding.
