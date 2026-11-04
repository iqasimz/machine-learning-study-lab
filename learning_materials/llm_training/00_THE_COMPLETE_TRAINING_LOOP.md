# T0. The language-model training loop

Training changes a language model's parameters using examples. A batch supplies input tokens and known targets; the model predicts, a loss measures mistakes, backpropagation computes gradients and an optimizer updates the parameters. Validation and generation use the model differently: they do not apply these training updates.

A sequence [3,7,2,9] supplies inputs [3,7,2] and next-token targets [7,2,9]. One supplied target probability illustrates loss, while a separate scalar gradient illustrates an update. They explain the stages but are not one fully differentiated numerical model.

The foundations and DL course introduced loss, gradients and updates. This lesson assembles them for token sequences, keeping the repeated training path separate from periodic validation, checkpoint selection and fixed-weight use.

## 1. Training, validation and generation

**Key terms.** A **batch** is a group of training examples; **weights** are learned numbers; an **update** changes them; **validation** checks separate examples without updates.

**Process.** Preparation produces batches and starting weights. Training repeats prediction, loss, gradients and updates; periodic validation selects a saved checkpoint. Generation then uses that checkpoint without the training loop.

There are three parts: prepare once, repeat training, then check/save/use. They are connected but are not the same loop.

```text
Prepare: sources → clean and split → tokenizer → batches → initial weights

Repeat:  batch → predictions → loss → gradients → optimizer update
            ↑                                      |
            +--------------- next batch ------------+

Check:   validation text → measure without updates → choose checkpoint
Use:     chosen checkpoint → prompt → generate → inspect the output
```

A batch supplies many next-token targets. After enough batches the program checks progress, saves state and continues if the planned budget allows it. Validation is periodic, not necessarily after every update.

## 2. Text preparation

**Key terms.** A **corpus** is a collection of text; a **split** assigns independent sources to training, development or test; a **tokenizer** converts text to IDs.

**Process.** Choose permitted sources, clean and split them, then fit a tokenizer only on training text or reuse the pretrained model's tokenizer. The output is compatible token sequences for building batches.

Choose permitted text that suits the goal. A story model needs many varied stories; 24 repeated teaching paragraphs are not a substitute. Remove broken documents and obvious duplicates, then assign whole documents or related groups to training, validation and test splits.

Training supplies weight updates. Validation, also called development text, helps choose settings/checkpoints. Test is the final reserved check. If we keep changing the design after seeing test, it is no longer independent.

A tokenizer turns text into pieces and gives each piece a number. Fit a new tokenizer on training text only. A pretrained model instead needs its compatible existing tokenizer: changing IDs without changing their learned meanings breaks that compatibility.

## 3. Inputs and next-token targets

**Key terms.** A **target** is the answer to predict; a **causal mask** blocks future input; a **loss mask** excludes positions from scoring.

**Process.** Take a token sequence and align each input position with its next token. Mask forbidden attention and unscored targets, then send the aligned inputs to the model and targets to the loss.

```text
Original: <start> The cat sat <end>
Input:    <start> The cat sat
Target:   The     cat sat <end>
```

The target at each position is the next piece. That is where labels come from; someone does not label every word by hand. Some libraries shift internally, so our code must not shift twice.

Training processes many positions together. A causal attention mask still blocks future input: when predicting `cat`, the model cannot read it from a future position. A separate loss mask decides which targets count towards the grade. Padding is ignored. In our conversation adapter, the prompt provides context but only response targets contribute to loss.

## 4. Forward pass

**Key terms.** An **embedding** maps an ID to a vector; **attention** mixes allowed positions; an **MLP** transforms features; a **logit** is a raw vocabulary score.

**Process.** Token IDs become vectors, Transformer blocks transform those vectors, and the output layer produces vocabulary scores. These forward calculations precede loss and do not themselves update weights.

IDs select learned vectors, called embeddings. Position information represents order. Transformer blocks mix permitted earlier information through attention and transform it through feed-forward networks. Residual connections carry earlier information around a sublayer; normalization controls the scale of intermediate values.

The final layer produces a raw score for each vocabulary token: a logit. Softmax turns logits into probabilities summing to one. This is the forward pass. Predictions exist, but weights have not changed yet.

## 5. Cross-entropy loss

**Key terms.** **Cross-entropy loss** is the negative log probability of the correct target; a **nat** uses the natural logarithm; **N** counts scored targets; **pᵢ** is each correct-target probability.

**Process.** Combine vocabulary scores with the known targets, calculate a penalty at each included position and average those penalties. This scalar loss is the starting point for the backward calculation.

If the correct token gets probability 0.5, its penalty is about 0.693 nats. At probability 0.1, the penalty is about 2.303. Average over valid target tokens, not padding.

\[
L=-\frac{1}{N}\sum_{i=1}^{N}\log p_i
\]

`N` is the included target count; `p_i` is the probability of correct target `i`. A nat is the unit from a natural logarithm. Lower loss means better prediction on that text/tokenization, not automatic truthfulness or useful conversation.

## 6. Backpropagation

**Key terms.** A **gradient** is a loss derivative; **backpropagation** applies the chain rule backwards; **sigmoid** maps a scalar score to a probability between 0 and 1.

**Process.** Start from the scalar loss and follow recorded operations backwards to each trainable parameter. The result is a set of gradients waiting for an optimizer, not a new set of weights.

Backpropagation follows the calculation backwards to find how a small change in each trainable weight would locally affect loss. These derivatives are gradients. They describe an adjustment direction; they are not updated weights yet.

As a separate numerical check, let a two-choice model use probability `sigmoid(w)` for a correct target of 1. At scalar weight zero the probability is 0.5 and the cross-entropy gradient is −0.5. Increasing that weight would locally reduce this loss. A real LLM has many interacting weights, with gradients through the complete computation.

## 7. Accumulation and optimizer updates

**Key terms.** A **microbatch** fits one forward/backward pass; **accumulation** combines gradients; **clipping** limits their overall length, called the gradient norm; **η** denotes learning rate; **g** denotes gradient.

**Process.** Accumulate correctly weighted microbatch gradients, unscale if needed, clip if configured, then apply one optimizer step. Clear gradients at the declared boundary before collecting the next update's evidence.

A microbatch is a small group that fits in memory. Gradient accumulation combines several microbatches before one optimizer update. It does not apply an update after each small batch.

With 10 valid targets in one microbatch and 90 in another, a token-average objective weights their mean losses by 0.1 and 0.9, not half each. The denominator matters. Gradient clipping can then reduce an excessively large overall gradient norm; it does not normally clip every weight separately.

The simple SGD update is:

\[
w_{\text{new}}=w-\eta g
\]

`w` is the learned number, `g` its gradient and `η` the learning rate. Weight 0, gradient −0.5 and rate 0.2 give new weight 0.1. In the two-choice example, correct-token probability becomes about 0.525 and loss falls from 0.693 to 0.644. This checked calculation is not a promise that every large-model update improves validation. AdamW also uses running statistics and weight decay rather than this bare SGD rule.

Clear gradients at the appropriate boundary so unrelated updates do not accidentally accumulate. Advance the schedule in its declared units, fetch another batch and repeat. An update is not an epoch: an epoch is a pass over a dataset, whereas our sampler can practise examples repeatedly.

### Check the complete update boundary

Before calling an experiment reproducible, trace one update with these checks:

1. Save the exact input IDs, target IDs and scoring mask of a small non-sensitive test batch. Count valid targets after shifting.
2. Compute a summed cross-entropy and divide by the intended update's target count. A response-only objective and an all-token objective are different even with the same input text.
3. Confirm that expected trainable parameters have finite gradients, frozen parameters stay frozen, and no training graph is retained accidentally in logging.
4. Apply one optimizer update and verify that the expected weights changed. Record learning rate and gradient norm, with an explicit before/after-clipping convention.
5. Restore a complete checkpoint and compare the next few CPU updates against uninterrupted training. Approximate continuation and identical replay are different acceptance criteria.

The saved-run inspector's optional scratch update is a separate diagnostic: it uses its own short text batch and SGD, not the original run's optimizer state. Do not treat it as a replay of the training trajectory. For implementation details, consult [PyTorch autograd](https://docs.pytorch.org/docs/2.9/notes/autograd.html) and [checkpoint restoration](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## 8. Validation and checkpoints

**Key terms.** **Evaluation mode** changes behaviours such as dropout; **no-grad mode** stops recording derivative history; a **checkpoint** saves model state.

**Process.** Pause training updates, measure separate validation text and choose whether to save the current state. Resume training only afterwards; validation results select checkpoints but supply no training gradient.

Periodically use separate validation text in evaluation mode, with gradient recording disabled. Dropout is off in evaluation mode. There is no backward pass or optimizer update. Validation helps choose a checkpoint, not the training gradient.

Training loss can fall while validation rises. That can indicate overfitting, although noisy checks, data changes and evaluation mistakes also matter. The lab's saved-run curves show recorded observations; the teaching process diagram is not a measured validation curve.

A checkpoint saves weights. Proper training resume also needs optimizer/schedule position, random/sampler state and compatible data/configuration. Our scratch path supports this continuation. The adapter path saves selected/final adapters but does not yet promise resumable optimizer state.

## 9. Generation with saved weights

**Key terms.** A **prompt** is starting text; **decoding** selects tokens from predicted probabilities; a **stop rule** ends generation.

**Process.** Load a chosen checkpoint, process the prompt, choose and append a token, then predict again. Generated tokens become the next inputs; no correct future targets or optimizer steps are supplied.

Process a prompt, choose a next token, append it and repeat until an end token or length limit. No weights change. Temperature and top-k affect selection, not learned knowledge. A story-completion model is not automatically an instruction-following assistant.

Training has known targets and uses gradients. Generation does not receive the true future answer and normally performs no backward pass. Inspect unedited outputs as well as loss: our adapter example became wrong about melting despite a better average loss.

## 10. Scratch training versus adaptation

**Key terms.** **Scratch training** starts from random weights; **adaptation** starts from pretrained weights; an **adapter** is a small trainable addition to a frozen base.

**Process.** Choose the learning goal before the training path. That choice determines the starting knowledge and updated parameters; evaluate success against the same starting point rather than comparing unlike training budgets.

| Path | Starting weights | Updated weights | Intended success |
| --- | --- | --- | --- |
| Scratch mechanics | Random | All weights | Correct, inspectable masking, gradients and controlled tests |
| Current language adapter | Pretrained SmolLM2 | Small LoRA matrices | Audited change against that same starting model |
| Proposed story model | Random | All weights of a roughly 14M decoder | Held-out prediction gains plus coherent completions on fixed new story beginnings |

The story model is a proposed next run, not trained already. It needs a tokenizer/corpus pipeline, throughput benchmark and approved budget. More parameters on the same tiny fixture would not supply missing language exposure.

## Quick checks

1. Does a forward pass change weights? No: it makes predictions.
2. Does backward apply the update? No: it computes gradients; the optimizer updates.
3. Does validation call the optimizer? No: that would train on the check.
4. Is lower training loss enough? No: check held-out prediction and intended behaviour.
5. Is adapting pretrained weights equivalent to learning from random weights? No: starting knowledge and compute requirements differ.

References: [PyTorch optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [TinyStories paper](https://arxiv.org/abs/2305.07759), [TinyStories data/license](https://huggingface.co/datasets/roneneldan/TinyStories).
