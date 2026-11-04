# DL01. Neural networks and multilayer perceptrons

A neural network combines learned weighted sums with nonlinear functions. A multilayer perceptron, or MLP, uses dense layers: each output feature reads all features in the preceding layer. Hidden layers create intermediate representations that can support a more flexible prediction rule than one linear transformation.

Our network has two inputs, two hidden features and one numerical output. Its weights stay fixed while you switch between ReLU and tanh, so you can see how the nonlinear function changes the prediction before studying training.

Linear regression combines inputs once. An MLP inserts intermediate features and nonlinearities; the prediction, loss and gradient-update principles from the foundations still apply.

[course map](../COURSE_MAP.md). Required background: ML04, ML05.

## 1. Dense layers, nonlinearities and representations

**Key terms.** A **neuron** computes a weighted sum and activation; an **MLP** stacks fully connected layers; a **hidden layer** builds intermediate features. **Depth** counts successive transformations; **width** counts features in a layer.

**Process.** Prepared input enters a linear layer, an activation introduces nonlinearity, and later layers turn learned features into task scores. A loss compares scores with targets; backpropagation later adjusts all trainable layers together.

Stacking only linear maps still produces one linear map, regardless of depth. Nonlinear activations let compositions represent different boundaries. This is the key transition from fixed features with a learned classifier to a jointly learned representation and classifier—not a guarantee that extra layers generalize.

A network's inductive assumptions come from architecture, data, optimization and regularization. An MLP on pixels ignores explicit spatial structure; a CNN adds sharing/locality later. For small tabular data, a boosted tree can outperform an MLP without invalidating deep learning. The relevant question is which assumptions and resources suit the task.

## 2. A worked calculation

**Symbols and assumptions.** W_l and b_l are a layer's weights and bias; phi is an activation; h_l=phi(h_(l−1)W_l+b_l) in row convention. ReLU(a)=max(0,a).

For binary inputs, XOR is 1 at [1,0]/[0,1] and 0 at [0,0]/[1,1]. Define h1=ReLU(x1+x2), h2=ReLU(x1+x2−1), output=h1−2h2. The four outputs are [0,1,1,0]. These hand-set weights demonstrate representability, not successful training. A 2-input, 3-hidden, 2-output dense network has 2×3+3+3×2+2=17 trainable weights/biases.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing width, depth and activation

Linear models need nonlinear supplied features to express XOR; a tree can partition it; an MLP can learn a hidden representation. More width adds parameters; more depth changes composition and optimization. Compare fixed data, objective and selection budget rather than assume that deep implies intelligent.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: when deeper is not better

First verify XOR exactly, then compare a small MLP with a regularized linear model on digits. Plot held-out quality and fit cost, not just whether training loss reached zero.

**Check.** Would ten linear layers without any nonlinear operations express a boundary that one linear layer cannot?

**Answer.** No, under compatible ordinary affine compositions they collapse to a single affine transformation. Intermediate factorization can change optimization or constraints, but not that unrestricted function family.

## 5. Dense-layer dimensions and parameter counts

**Key terms.** B is batch size, d the input width and h the hidden width. A bias broadcasts across batch rows; a trainable parameter is stored once, not once per example.

**Process.** A B×d batch multiplies a d×h weight matrix, receives an h-element bias and passes through a nonlinear activation. The resulting B×h matrix is the next layer's input.

For each dense layer, count one weight for every input–output feature pair plus one bias per output feature. The same parameters are reused for every row in a batch.

A 784→64→10 network has 784×64+64+64×10+10=50,890 parameters with one hidden layer. With two 64-wide hidden layers it adds 64×64+64=4,160, giving 55,050. Increasing the batch size changes activation memory and update statistics, not this parameter count.

### Capacity is not the only difference

The local MLP flattens images. Its first layer treats neighbouring and far-apart pixels with the same unconstrained connectivity. A CNN instead shares a filter across spatial positions. This changes inductive bias, not simply width.

An MLP's hidden values are learned representations; they need not be human-readable categories. Useful representations are tested by held-out tasks or transfer, not inferred from a colourful projection. Compare the same data, preprocessing and checkpoint rule before attributing a gain to depth.

## Sources

- [Deep Learning, author site](https://www.deeplearningbook.org/).
- [Dive into Deep Learning](https://d2l.ai/).
- [PyTorch training workflow](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
