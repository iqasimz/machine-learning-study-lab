# P2. Vectors, matrices and gradients

Models work with numbers arranged in lists and tables. A vector is an ordered list of features, such as a flower's length and width. A matrix holds several such lists or the weights that combine them. Calculus tells us how changing one weight would change a prediction or its error.

Use two input features, [2, 1], to predict the number 4. We will calculate one prediction, measure its error and work backwards to find the effect of its first weight.

This introduces the arithmetic used by linear regression and neural networks. The gradient calculated here becomes an input to an optimizer in the next training lesson.

## 1. Vectors: numbers with a shared meaning

**Key terms.** A **vector** is an ordered list of numbers; a **coordinate** is one entry; a **dot product** multiplies corresponding entries and sums them.

**Process.** A token's vector enters a weighted sum. The dot product gives one score; several such scores form the output of a linear layer. Start with the two-number example before the matrix version.

Write a token representation as `x=[2,1]`. The two coordinates are learned features, not necessarily human-readable categories. The dot product with `w=[3,-1]` is `x·w=2×3+1×(-1)=5`. A linear layer combines input features using these weighted sums.

Vector length, or Euclidean norm, is `||x||₂=√Σxᵢ²`. For `[3,4]`, the length is 5. Dividing by length gives a unit vector `[0.6,0.8]`. Its direction is unchanged. The dot product between unit vectors is their cosine similarity.

A **basis** is a set of directions with which we express vectors. Changing coordinates can preserve the represented information. Consequently, “dimension 17 means honesty” is generally not justified just by looking at an embedding coordinate.

## 2. Matrices are transformations

**Key terms.** A **matrix** is a two-axis array; **transpose** swaps its axes; **rank** counts independent output directions.

**Process.** Input features enter a matrix multiplication. Matching inner dimensions are summed out, producing new features. Write the input and output shapes before using the same operation inside attention.

Use row-vector convention throughout the course. Let:

```text
x = [2, 1]              W = [[1, 3, 0],
                            [2, 0, 4]]
xW = [4, 6, 4]
```

Shape rule: `[m,n] @ [n,p] → [m,p]`. Here `[1,2] @ [2,3] → [1,3]`. Each column of `W` defines one output feature. Transpose swaps axes: `Wᵀ` has shape `[3,2]`.

Matrix multiplication is usually not commutative: `AB` may differ from `BA`, and the second product might not even be defined. Elementwise multiplication does not replace it. In attention, `QKᵀ` compares every query row with every key row.

**Rank** measures the number of independent directions a matrix can produce. A low-rank product `AB`, with intermediate width `r`, has rank at most `r`. This will explain LoRA: a large weight update can be represented through a smaller bottleneck when that restriction is useful.

**Eigenvectors** are directions preserved by a square transformation up to scaling. **Singular value decomposition** writes a matrix as `UΣVᵀ`: rotate/change basis, scale along directions, rotate again. Truncating small singular values approximates the matrix with fewer independent directions. This is a mathematical explanation of low-rank approximation, not proof that every neural update is low rank.

## 3. Derivatives: sensitivity to a change

**Key terms.** A **derivative** measures local sensitivity; a **gradient** collects sensitivities of one scalar result to many variables; a **learning rate** scales an update.

**Process.** First compute a prediction and loss. Work backwards through those operations to obtain a gradient, then apply an update. The numerical example keeps these three stages separate.

For `f(w)=w²`, the derivative is `2w`: near `w=3`, a small change `δ` changes the output by approximately `6δ`. A **partial derivative** changes one variable while holding the others fixed. A **gradient** collects all partial derivatives of a scalar output.

Suppose prediction `ŷ=wx`, target `y`, and loss `L=½(ŷ-y)²`. By the chain rule:

\[
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial\hat y}\frac{\partial\hat y}{\partial w}
=(wx-y)x.
\]

With `x=2,y=1,w=1`, prediction is 2, loss is 0.5 and gradient is 2. A gradient-descent update with learning rate `η=0.1` gives `w←1−0.1×2=0.8`. New prediction is 1.6 and loss is 0.18. The update improved this example; arbitrary large updates need not improve it.

## 4. The matrix derivative you will actually use

**Key terms.** An **upstream gradient** is sensitivity arriving from a later operation; a **Jacobian** contains output-to-input derivatives; **autograd** computes derivatives through recorded operations.

**Process.** The forward layer maps X to Y. Later, loss supplies a gradient with respect to Y; the layer converts it into gradients for X and its weights. The input gradient continues to the previous layer.

For a linear layer `Y=XW`, suppose `G=∂L/∂Y`. Then:

\[
\frac{\partial L}{\partial W}=X^\top G,
\qquad \frac{\partial L}{\partial X}=GW^\top.
\]

Check shapes: `X:[B,d]`, `W:[d,h]`, `G:[B,h]`; therefore `XᵀG:[d,h]` and `GWᵀ:[B,d]`. Bias gradients sum across examples. This is why backpropagation can reuse efficient matrix multiplication.

A **Jacobian** contains derivatives of each vector output with respect to each input. Frameworks usually propagate vector–Jacobian products rather than materialising every enormous Jacobian. Autograd follows recorded operations; it does not understand your intended scientific meaning.

### Follow a gradient through a complete linear sublayer

Take `X:[2,3]`, `W:[3,4]`, bias `b:[4]` and `Y=XW+b`. Suppose a later loss supplies `G=∂L/∂Y:[2,4]`.

1. Each output feature receives contributions from three input features. That is why the forward contraction removes the shared size-3 axis.
2. The weight gradient `XᵀG:[3,4]` sums contributions from both examples. For sequence data, the batch and scored-position axes also contribute to this sum.
3. The input gradient `GWᵀ:[2,3]` sends sensitivity back to the preceding sublayer. The framework does not have to build a giant full-network Jacobian.
4. The bias gradient sums `G` over the two examples. If the loss is averaged over targets, that averaging already affects `G`; dividing again changes the update.
5. Check the result on a small differentiable function using a centred finite difference, `(L(w+ε)−L(w−ε))/(2ε)`. Compare with autograd using double precision and several sensible ε values. Extremely small ε amplifies rounding; discontinuities require care.

This is a **vector–Jacobian product**: reverse-mode differentiation propagates the loss's upstream sensitivity through an operation. Its efficiency is why scalar-loss neural training is practical. See [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html) and [gradient checking](https://docs.pytorch.org/docs/2.9/notes/gradcheck.html).

## 5. Softmax and cross-entropy

**Key terms.** A **logit** is a raw score; **softmax** converts a score vector into probabilities; a **one-hot target** is 1 at the correct class and 0 elsewhere.

**Process.** Logits enter softmax, target probabilities determine cross-entropy, and the combined derivative returns a signal to the score-producing layer. Define the target class before calculating its penalty.

For logits `z`, define `pᵢ=exp(zᵢ)/Σⱼexp(zⱼ)`. A logit is an unrestricted score, not a probability. Subtracting the maximum logit before exponentiation leaves probabilities unchanged and avoids numerical overflow.

For one-hot target `y`, cross-entropy is `L=−Σᵢyᵢ log pᵢ`. The useful combined derivative is:

\[
\frac{\partial L}{\partial z_i}=p_i-y_i.
\]

Derivation sketch: `log p_k=z_k−log Σⱼexp(zⱼ)`. Differentiating its negative gives `−1` at the correct class plus the softmax probability at every class. If `p=[0.7,0.2,0.1]` and the target is class 2, gradients are `[0.7,−0.8,0.1]`. Gradient descent raises the target logit relative to the others.

## 6. Curvature and optimisation intuition

**Key terms.** **Curvature** describes how a gradient changes; a **Hessian** contains second derivatives; a **local approximation** describes only a small neighbourhood.

**Process.** A gradient suggests a direction, but curvature determines how quickly that direction becomes outdated. Use this distinction to reason about update size before trusting a large step.

A **Hessian** collects second derivatives; it describes how gradients change. A narrow, steep valley can make a single learning rate awkward. A **condition number** describes imbalance between directions in appropriate settings. These ideas motivate normalization and adaptive optimizers, but do not make nonconvex neural optimization globally solved.

A **Taylor approximation** replaces a smooth function locally by simpler terms: `f(w+δ)≈f(w)+∇f(w)·δ`. This is local, not a promise about large changes. Gradients are local sensitivity, not automatic causal explanations of model behaviour.

## Exercises and checks

1. Compute `[1,2] @ [[2,0],[1,3]]`. **Answer:** `[4,6]`.
2. If `X:[8,16]`, `W:[16,32]`, what is `∂L/∂W`? **Answer:** `[16,32]`, not `[8,32]`.
3. For probabilities `[0.25,0.75]` and target class 1, calculate loss and logit gradients. **Answer:** `−ln(0.25)≈1.3863`; `[-0.75,0.75]`.
4. Why can a rank-4 update be much cheaper than a full `4096×4096` update? Explain parameter counts before R1.

Further study: [D2L linear algebra, calculus and autograd](https://d2l.ai/chapter_preliminaries/index.html). Verify softmax's invariance to a constant logit shift algebraically; the static attention diagram in T3 applies the same normalization to attention scores.
