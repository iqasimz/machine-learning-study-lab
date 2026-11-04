# DL03. Optimizers, initialization and training stability

An optimizer turns gradients into parameter updates. Plain stochastic gradient descent, or SGD, subtracts a learning-rate-scaled gradient. Adam keeps moving averages of gradients and squared gradients to adapt update sizes. Initialization chooses the starting weights; a schedule changes learning rate over time. These are related choices, not interchangeable methods.

We compare one SGD step with the first Adam step from weight 1 and gradient 2 at learning rate 0.1. The Adam-state sections explain Adam even when SGD is selected; the final update uses the selected rule. No weight decay is included in this arithmetic.

Backpropagation supplies gradients. This lesson explains how optimizer rules use them, then the deeper notes cover initialization, clipping, normalization, schedules and failure diagnosis.

[course map](../COURSE_MAP.md). Required background: DL02.

## 1. Starting weights, normalization and update size

**Key terms.** **Initialization** sets starting weights; **momentum** retains gradient history; **Adam** estimates first/second gradient moments; **normalization** rescales activations. **BatchNorm** uses batch/spatial statistics during training and typically stored statistics during evaluation.

**Process.** Initialize compatible weight scales, make a forward/backward pass and inspect finite gradients. Apply the configured optimizer/schedule, update required running statistics and evaluate in the correct mode before judging stability.

Bad scales can saturate nonlinearities or amplify signals across depth. Xavier/Glorot and He initialization use different variance assumptions; the choice depends on layer/activation. BatchNorm, LayerNorm and RMSNorm reduce across different axes and have different train/eval semantics. Do not describe BatchNorm's original proposed explanation as the only established reason it helps.

Adam adapts updates through moment estimates; its first bias corrections matter because moments begin at zero. AdamW applies decoupled decay rather than equating all adaptive decay with ordinary L2 loss. Epsilon stabilizes a denominator; it is not the same control as learning rate. Schedules must state whether units are batches, updates or epochs.

## 2. A worked calculation

**Symbols and assumptions.** For scalar gradient g_t, Adam uses m_t=beta1 m_(t−1)+(1−beta1)g_t and v_t=beta2 v_(t−1)+(1−beta2)g_t², then bias-corrects both. eta is learning rate and epsilon is a stabilizer.

Starting from m=v=0 with g=2, beta1=0.9 and beta2=0.999 gives m=0.2,v=0.004. First-step corrections give m-hat=2,v-hat=4. With eta=0.1 and negligible epsilon, w=1 becomes 0.9 under the gradient update. A separate AdamW decay of eta×lambda×w with lambda=0.01 subtracts 0.001 under this stated order, giving about 0.899.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Optimizer and initialization choices

Compare SGD with momentum and AdamW under separately selected reasonable rates and matched data exposure. An optimizer comparison at one arbitrary shared rate can favour one through tuning. Normalization can change behaviour beyond speeding convergence; small-batch BatchNorm can be noisy.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: diagnosing unstable training

Inspect loss, update magnitudes and activation/gradient scales on a small digits MLP. Reuse the existing training-loop lesson rather than invent a separate meaning for update or checkpoint.

**Check.** Why can calling model.train() versus model.eval() change outputs without an optimizer step?

**Answer.** Dropout and normalization can change computation or use of running statistics. Weight updates and mode-dependent forward behaviour are distinct.

## 5. Adam, weight decay, clipping and schedules

**Key terms.** A first moment tracks a moving average of gradients; a second moment tracks squared gradients. Bias correction compensates for zero-initialized moment estimates. A schedule changes the global learning rate.

**Process.** Calculate the gradient, optionally clip its norm, update optimizer state, calculate a parameter step and apply the chosen decay rule. The next batch sees the updated parameter values.

In this calculation, beta values control how much previous optimizer history is retained. The gradient is fixed so we can inspect the rule rather than compare trained models. The decay contribution described below is a separate operation from the gradient-based adaptive step.

For Adam's first gradient g=0.2, β₁=0.9 gives m₁=0.02 and β₂=0.999 gives v₁=0.00004. Correcting gives m̂₁=0.2 and v̂₁=0.04. Ignoring epsilon, their ratio is 1, so learning rate 0.1 gives a step of 0.1—not 0.02 as ordinary SGD would. AdamW decay with λ=0.01 on parameter w=1 adds a separate shrinkage 0.001 under this illustrative step.

### What does clipping change?

A gradient vector [3,4] has norm 5. Clipping the maximum norm to 1 rescales it to [0.6,0.8], preserving direction before the optimizer transforms it. The lab logs the pre-clipping norm; it is not an update norm.

A cosine schedule lowers the common step scale across epochs. It does not replace Adam's moment estimates. Compare optimizer, rate, batch size and total updates together, and retain a simple constant-rate baseline.

## Sources

- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101).
- [Adam](https://arxiv.org/abs/1412.6980).
- [Batch Normalization](https://arxiv.org/abs/1502.03167).
- [CS231n practical neural-network training notes](https://cs231n.github.io/neural-networks-3/).
