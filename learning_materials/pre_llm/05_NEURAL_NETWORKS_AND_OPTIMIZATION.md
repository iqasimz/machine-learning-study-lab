# P5. How models learn: loss and optimization

Training adjusts a model's learned numbers so its predictions better match examples. The usual loop is: calculate a prediction, measure its error with a loss function, compute how each weight affects the loss, and use an optimizer to update the weights. A neural network repeats this idea across many connected layers.

Our smallest model predicts y = w × x. The input is 2, the target is 4 and only one weight can change. One explicit update lets you see what happens without hiding it inside a training library.

The vectors and gradients lesson explains the derivative behind the gradient. This lesson uses it to update a model; DL later adds nonlinear layers and many weights without changing the basic loop.

## 1. From a linear score to a network

**Key terms.** A **linear layer** forms weighted sums; an **activation function** adds nonlinearity; **parameters** are the learned weights and biases.

**Process.** Input features pass through a linear map and activation, then further layers produce predictions. The training objective later sends gradients back through this same chain.

A linear layer calculates `z=xW+b`. A nonlinear activation gives `h=φ(z)`. A second layer combines these transformed features. Without nonlinearities, stacked linear layers collapse into one linear transformation. Depth then adds no new function family of that kind.

ReLU is `max(0,z)`; GELU smoothly weights inputs using a Gaussian-related function; SiLU is `z·sigmoid(z)`. These choices affect gradients and computation. A gated feed-forward layer, such as SwiGLU, multiplies a transformed branch by an activated gate. A gate controls information flow; it is not necessarily a binary decision.

```text
input → linear → activation → linear → prediction → loss
          ↑          ↑          ↑                 │
          └──────── gradients via chain rule ─────┘
```

## 2. Forward, backward, update

**Key terms.** The **forward pass** computes predictions; the **backward pass** computes gradients; an **optimizer step** changes trainable parameters.

**Process.** A batch enters the model, predictions meet targets in a loss, backward computes sensitivities, then the optimizer updates weights. The next forward pass uses those changed weights.

The forward pass computes activations and loss. Backpropagation applies the chain rule in reverse to calculate parameter gradients. The optimizer uses gradients and possibly stored state to change parameters. These are separate operations.

Autograd records a computational graph. A detached tensor breaks a gradient path. Converting a tensor into a plain Python number does not preserve that path. Gradient buffers often accumulate, so failing to clear them changes the effective update. Training mode enables behaviours such as dropout; evaluation mode changes those behaviours but is not the same as disabling gradients.

Conceptual loop, with one batch per update:

```python
optimizer.zero_grad()
logits = model(inputs)
loss = objective(logits, targets)
loss.backward()
optimizer.step()
```

This omits device placement, masks, mixed precision, clipping and distributed synchronization deliberately. Understand the four operations before adding machinery.

## 3. SGD, momentum and AdamW

**Key terms.** **SGD** uses the current gradient; **momentum** retains an update history; **AdamW** uses running gradient statistics and decoupled weight decay.

**Process.** After backward, the optimizer combines gradients with its stored state to choose a parameter update. Its state and learning-rate position must also be saved if training will resume.

Stochastic gradient descent uses an estimated minibatch gradient `g`: `θ←θ−ηg`. Momentum tracks an exponentially weighted gradient, smoothing updates. Adam also tracks squared gradients:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

Bias-correct these initial zero-based averages, then update using `m̂/(√v̂+ε)`. AdamW separates weight decay from the adaptive gradient transformation. In a simple convention, `θ←θ−η m̂/(√v̂+ε)−ηλθ`. `η` is learning rate; `λ` controls decay. Layer exclusions and exact implementation details matter.

Learning-rate **warmup** gradually increases step size at the beginning. Cosine decay gradually decreases it. Schedules are choices to validate, not universal laws. A smaller rate may improve stability while requiring more steps; it can also fail to learn enough within the budget.

## 4. Residuals and normalization

**Key terms.** A **residual connection** adds a sublayer's input to its output; **normalization** rescales activations; **activations** are intermediate computed values.

**Process.** A sublayer transforms token features. The residual path carries the earlier representation around it; normalization stabilizes the scale seen by a neighbouring operation. Exact order depends on the architecture.

A residual connection adds an input back: `y=x+F(x)`. It creates a direct information/gradient route; it does not ensure every deep network trains well. Layer normalization centres/scales feature values per example/token. RMSNorm scales by root mean square without the same centring step. Their learned scales and normalization placement affect stability.

Dropout randomly removes selected activations during training using appropriate scaling. It is ordinarily disabled for evaluation. Weight decay and dropout are not interchangeable; one constrains parameters, the other perturbs intermediate computation.

## 5. Numerical reality

**Key terms.** **Precision** controls numeric resolution; **overflow** exceeds the available range; **underflow** loses very small values; **mixed precision** uses different formats for different operations.

**Process.** Choose numeric formats, run a small forward/backward calculation and check finite values. Scaling or safer arithmetic can repair representation problems before an optimizer consumes invalid gradients.

Floating-point numbers have finite range and precision. FP16 can overflow at values BF16 can represent; BF16 has fewer mantissa bits than FP32. Mixed precision combines lower-precision computation with selected higher-precision operations/state. It does not mean everything is stored in half the memory.

Gradient clipping limits update-driving gradient norm: `g←g·min(1,c/||g||)`. It can prevent destructive steps but cannot fix invalid labels or a fundamentally wrong loss. Check for NaNs/Infs, abnormal norms, loss spikes and inconsistent dtype/device operations.

## 6. Debugging story

**Key terms.** A **symptom** is an observed failure; a **diagnostic** isolates its cause; a **sanity check** tests a deliberately simple case.

**Process.** Begin with a tiny batch whose behaviour you can inspect. Check labels, masks, gradients and updates in order, then scale only after the small case behaves as intended.

Your model cannot fit even 16 training examples. Before changing architecture: verify targets, shift, padding mask, trainable parameters, learning rate, gradients and whether the optimizer owns the same parameters the forward pass uses. Try disabling regularization and overfitting that tiny batch. Failure to memorise a small clean batch often signals an implementation problem; success does not establish generalization.

### Read an optimizer step as a sequence of state changes

AdamW is more than multiplying a gradient by a learning rate. After computing the gradient of the intended objective:

1. Combine the required microbatches and, where relevant, distributed contributions. Do not clip every partial gradient and assume the result matches clipping their sum.
2. If gradients were scaled for FP16 training, unscale before measuring or clipping their norm. Otherwise the clipping threshold applies to the wrong scale.
3. Update first and second moment estimates. Bias correction compensates for their zero initialization; epsilon stabilizes division, not the underlying statistical objective.
4. Apply the adaptive update and decoupled weight decay using the implementation's parameter groups. Some recipes exclude normalization scales or biases from decay; that is a recipe choice to record.
5. Advance the scheduler only when its declared optimizer update occurred. If a scaler skips a non-finite update, blindly advancing the schedule can desynchronize the intended recipe.

Freezing a weight removes its own parameter update but does not necessarily remove the need to differentiate through its operation to reach earlier trainable adapters. Conversely, evaluation mode changes dropout/module behaviour without disabling autograd. See the [AdamW paper](https://arxiv.org/abs/1711.05101) and [PyTorch AMP examples](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html) for these distinct mechanisms.

## Practice and checks

1. `θ=2,g=3,η=0.1`: an SGD step gives 1.7.
2. `g=[3,4]`, clipping threshold 1: gradient becomes `[0.6,0.8]`.
3. Why isn't `model.eval()` enough to save autograd memory? **Answer:** it controls module behaviour; gradient tracking is a separate mechanism.
4. What would you log besides loss? Gradient norm, learning rate, valid-token count, memory, throughput, held-out loss and task metrics.

Visual experiment: [TensorFlow Playground](https://playground.tensorflow.org/). Choose a nonlinear dataset, predict whether a linear network can separate it, then add one hidden layer. Change noise and compare training/test loss. You are testing representation and overfitting, not training an LLM. Framework reinforcement: [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
