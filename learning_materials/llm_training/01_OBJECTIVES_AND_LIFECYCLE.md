# T1. Language-model training objectives

A training objective defines what the model predicts, which information it may use and how mistakes are scored. Causal language modelling predicts successors from prefixes; masked modelling reconstructs selected tokens from permitted surrounding text; encoder–decoder learning predicts a target sequence conditioned on a source. These are different prediction problems, not interchangeable names for pretraining.

Choose an objective to compare visibility and targets. A separate fixed set of three correct-target probabilities [0.5,0.25,0.5] illustrates mean negative log-likelihood. The numbers do not establish which objective trains a better model.

The complete training loop can optimize different objectives. Define their information access before comparing losses or downstream behavior; a language-model objective alone does not establish instruction following or safe deployment.

## 1. Next-token prediction

**Key terms.** An **autoregressive objective** predicts the next token from preceding tokens; **θ** denotes learned parameters; **x₍<t₎** is the prefix before position t.

**Process.** Tokenized text supplies prefixes and known next tokens. The model assigns target probabilities; negative log likelihood turns them into the loss used to adjust θ.

For an autoregressive model:

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\qquad
L(\theta)=-\frac1{N_{valid}}\sum_{b,t}m_{bt}\log p_\theta(x_{bt}\mid x_{b,<t}).
\]

`θ` is the trainable parameter collection; `m` is a 0/1 loss mask; `N_valid=Σm` counts included target positions. The loss increases when the model assigns low probability to observed targets. It does not explicitly reward truthfulness, helpfulness or obedience unless the data/objective supplies an appropriate signal.

**Where labels come from:** take token sequence `[BOS,the,cat,sat,EOS]`. Inputs are `[BOS,the,cat,sat]`; targets are `[the,cat,sat,EOS]`. At each position the correct next token is in the original text. This is self-supervision, not an absence of targets. In real code, some model APIs shift labels internally; shifting again is a bug.

Example target probabilities `[0.5,0.25,0.5,0.8]` give average loss `(0.6931+1.3863+0.6931+0.2231)/4≈0.7489` nats and perplexity about 2.115. Average across valid tokens, not padded positions.

## 2. Three major objective families

**Key terms.** An **objective** specifies the prediction problem; a **mask** hides or excludes information; **teacher forcing** supplies observed preceding tokens during training.

**Process.** Choose which text is visible and which tokens must be predicted, then construct inputs and targets accordingly. Causal, masked-token and encoder–decoder objectives therefore learn from different information patterns.

**Causal/next-token:** predict from earlier tokens. Appropriate for left-to-right generation. A causal mask prevents access to future tokens during training, even though many positions are computed in parallel.

**Masked language modelling:** hide selected input tokens and recover them from permitted surrounding text. BERT-style encoders can see both sides of unmasked context. They are useful for representations/classification but are not the same generation interface as an autoregressive decoder.

**Encoder–decoder / denoising:** encode a source, then decode a target, often reconstructing corrupted spans or performing translation. The encoder attends to source input; the decoder attends to earlier target tokens and the source representation. A prefix-LM can instead use different visibility rules within one stack.

Teacher forcing provides gold previous target tokens during training. During generation, the model receives its own earlier outputs. That mismatch is one source of compounding errors; it does not imply teacher forcing is inherently invalid.

### Why one forward pass can learn many next tokens

Suppose the text contains four input positions. The model produces vocabulary scores at all four positions, but a triangular attention mask gives each position a different permitted prefix.

1. Position 1 predicts token 2 using only position 1. Position 2 predicts token 3 using positions 1–2, and so on.
2. Gold tokens are available as later *input positions*, but the mask blocks those positions from an earlier query. Parallel computation does not imply permission to read future answers.
3. Gather the score of each correct next token, compute its negative log probability, apply the scoring mask and aggregate.
4. Backpropagate the aggregated loss once. Shared parameters receive contributions from many target positions in that pass.

At generation time there is no gold future sequence. The program must append a chosen token before computing the next output distribution. This distinction explains why training over positions can be parallel while ordinary output generation remains sequential. The statistical objective is conditional likelihood, not a direct test of whether a generated paragraph is true. Original objective examples: [Transformer](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805) and [T5](https://www.jmlr.org/papers/v21/20-074.html).

## 3. Pretraining, adaptation and inference

**Key terms.** **Pretraining** learns broad patterns; **continued pretraining** adds domain exposure; **SFT** imitates demonstrations; **inference** uses saved weights.

**Process.** First acquire general representations, then adapt them if the task requires it. Keep each stage's data and objective separate so a later behaviour change is attributed to the correct intervention.

```text
raw sources → governed corpus → tokenizer/packing → initialized architecture
→ pretraining → base checkpoint → optional continued/domain pretraining
→ supervised adaptation → optional preference/RL alignment
→ model evaluation → system integration → system evaluation → deployment
```

**Pretraining** learns broad statistical structure. **Continued pretraining** keeps a language-model objective on new data, often for a domain/language. **SFT** learns desired responses from demonstrations. **Preference optimization** changes which outputs are favoured. **Inference** computes outputs using fixed weights; an agent's memory or tool use is not automatically weight learning.

These are not mandatory steps in one universal recipe. A classifier can be adapted from an encoder without chat alignment. Retrieval can improve freshness without retraining. RL is not required to make every useful assistant.

## 4. Limits of the training objective

**Key terms.** **Fluency** is well-formed language; **factuality** concerns correctness; **alignment** concerns the desired behaviour under stated criteria.

**Process.** A training objective improves its measured prediction target. Independent behavioural tests must then establish any stronger claim about useful, truthful or safe answers.

Lower validation language-model loss demonstrates better prediction on that validation distribution. It does not alone establish factual support, robust instruction following, causality, reasoning faithfulness or successful deployment. A model may learn source style, reproduce private sequences, or inherit the corpus's stereotypes.

**Base versus instruction model:** a base model continues patterns; an instruction-tuned model is trained to respond in a particular conversational format. Comparing one as a plain completion and the other with a correct template can conflate interface and capability. Save the complete input format.

## 5. Experiment design

**Key terms.** A **factor** is the intended experimental change; a **control** fixes an alternative explanation; a **budget** specifies allowed work.

**Process.** Declare the hypothesis, data, objective and comparison budget first. Run the conditions with recorded settings, then assess held-out outcomes rather than choosing the story after seeing scores.

For a toy model, choose a small licensed/public-domain corpus, a fixed tokenizer, document-level splits, a context length, a small architecture and a compute ceiling. Record tokens processed, updates, elapsed time and validation loss. Run a trivial n-gram or unigram baseline. Inspect actual generated samples but do not choose checkpoints only because samples look entertaining.

## Exercises and answers

1. Does an unlabelled web document supply training labels? **Yes:** its next tokens become targets under a causal objective.
2. A batch has 100 valid targets and 100 padding positions. Is dividing the valid loss sum by 200 correct? **No:** it changes the scale with padding; divide by the intended valid-token count.
3. Which stage solves a request for today's weather? **An authorised current source/tool**, not merely more old-text pretraining.

Original architecture/objective context: [Transformer paper](https://arxiv.org/abs/1706.03762). Full lifecycle case study: [OLMo 2](https://arxiv.org/abs/2501.00656). Read these as examples of recipes, not evidence that all models use one pipeline.
