# DL02. Backpropagation and automatic differentiation

Backpropagation calculates how each trainable weight affects a loss by following the model's operations backwards. It applies the chain rule and adds contributions when a weight affects the loss through several paths. Automatic differentiation records these operations for you; the optimizer, not backpropagation itself, changes the weights.

The calculation has two weights: a = w × 2, then prediction = 0.5 × a, with target 3. Change the first weight and inspect the prediction, loss and the separate gradients for both weights.

The foundations introduced one gradient. This example shows why different weights receive different gradients, and why clearing or accumulating gradient buffers is a separate programming decision.

[course map](../COURSE_MAP.md). Required background: DL01, P2.

## 1. How autograd follows differentiable operations

**Key terms.** A **computation graph** records operations; a **gradient** measures local loss sensitivity; **backpropagation** applies the chain rule backwards. **Autograd** computes derivatives of recorded operations; an **optimizer** applies updates.

**Process.** Compute intermediate values during forward, build a scalar loss, propagate its sensitivities backwards and accumulate parameter gradients. Only an optimizer step changes weights; clear gradients at the intended update boundary.

An operation receives an upstream sensitivity from later computation and converts it into sensitivities for its inputs and parameters. A matrix layer reuses efficient matrix products for this calculation. Shared parameters collect contributions from every use. Autograd cannot know whether your labels, masks or objective match your scientific intention.

Detaching a tensor breaks its gradient path without necessarily copying storage. Retaining loss tensors in a log can retain whole graphs. Finite-difference checks compare analytic and numerical sensitivity on small deterministic differentiable examples; noise, rounding and activation kinks need care. Never diagnose the first broken training loop by merely enlarging the model.

## 2. A worked calculation

**Symbols and assumptions.** For prediction a=wx+b, target y and L=(a−y)²/2, the local derivatives are dL/da=a−y, da/dw=x and da/db=1. eta is learning rate.

At x=2,y=1,w=1,b=0, a=2 and L=0.5. Backward gives dL/dw=2 and dL/db=1. Updating both at eta=0.1 yields w=0.8,b=−0.1, new a=1.5 and L=0.125. Updating only w instead yields L=0.18: the distinction is which parameters changed. A centred finite difference uses [L(w+epsilon)−L(w−epsilon)]/(2epsilon) with b fixed for that check.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Autograd, manual derivatives and graph retention

Reverse-mode differentiation is efficient for many parameters leading to one scalar loss; forward-mode addresses other shapes of derivative problems. Numerical differentiation is a useful diagnostic, not a practical substitute for full-network training gradients. Clipping limits an update signal's norm; it does not repair a wrong derivative graph.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: verify a small gradient

Use an authored scalar case and a small linear layer, compare hand gradients with autograd and check parameter changes after exactly one step. Freeze randomness and disable stochastic layers for the numerical check.

**Check.** Loss decreases but one layer's weights never move. What do you inspect?

**Answer.** Trainable flags, optimizer parameter membership, detached paths, gradient values, zeroing order and learning rate. A missing update is not automatically an architecture limitation.

## 5. Gradients from multiple paths

**Key terms.** A computation graph records differentiable operations. The chain rule multiplies local derivatives along a path and sums contributions from multiple paths. Gradient accumulation adds contributions until explicitly cleared.

**Process.** Run the forward operations, obtain a scalar loss, seed its derivative with one and traverse the graph backward. Each parameter receives a derivative; an optimizer later uses it to change the parameter.

Let a=wx, z=a+a and L=z²/2. With w=3 and x=2, a=6, z=12 and L=72. dL/dz=12; the two uses of a contribute 12+12=24 to dL/da. Multiplying by x gives dL/dw=48. Forgetting one branch gives the wrong derivative even though the forward answer was correct.

### Verify without trusting the training curve

A finite-difference check slightly perturbs a weight in each direction, recomputes the loss and estimates its slope. It compares [L(w+ε)−L(w−ε)]/(2ε) with the analytic derivative. Very large ε gives approximation error; very small ε gives floating-point cancellation. Use a smooth small example away from ReLU's zero kink.

Autograd does not automatically update weights. zero_grad, forward, loss, backward and optimizer.step are distinct operations. The training lab clears gradients once per batch; the LLM scratch path separately exposes accumulation. Evaluation runs without graph construction and with dropout disabled.

## Sources

- [PyTorch training workflow](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
- [CS231n practical neural-network training notes](https://cs231n.github.io/neural-networks-3/).
