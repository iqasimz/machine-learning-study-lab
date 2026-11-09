# Deep learning code

Prerequisites: arrays, classes, functions and the difference between model construction and fitting. PyTorch supplies tensors, automatic differentiation and components for neural networks. It does not choose a sensible task, loss or evaluation protocol for you. All numerical examples below are fixed calculations or inspection, not trained-model quality results.

## PT01. PyTorch

PyTorch is the library this course uses for neural networks, and it is the library behind most current research code and behind the Hugging Face models in the LLM lessons. People keep using it because the numbers, the derivatives, and the network live in one Python process.

### Why it is widely used

A paper can publish a model as ordinary Python. A reader can run the same functions, print a value in the middle, and change one layer. The deep-learning trainer in this lab is written against the same objects: tensors and `nn.Module`. When a later lesson loads a language model, that model is a PyTorch module with stored weights. Learning these objects once is what makes the rest of the code readable.

Three pieces do the work:

- A **tensor** holds the numbers. Arithmetic and indexing look like the NumPy arrays from the data lessons.
- **Autograd** stores how those numbers were computed. You write the forward calculation. `backward` fills in the gradient. You do not differentiate each layer by hand.
- A **module** is a class. `forward` is a method. A new layer is the same `class` and `def` you already wrote in the Python chapter.

The library will also add two tensors of the wrong shape, score the wrong tokens, or train on a leaked split. It runs the calculation you wrote. It does not know which task you meant, and it does not decide that a result is good.

### Why the first lines stay short

The operations are the math. `+` adds. `@` multiplies and adds, which for two vectors is a dot product. A mistake is an ordinary Python traceback, so you can print the tensor that caused it. Moving the same tensor to a GPU, when the machine has one, changes where the arithmetic runs. The formula in the next line stays the same. These lessons stay on the CPU so the printed numbers do not depend on a graphics chip.

<!-- example: PT01-add -->
```python
import torch

first = torch.tensor([1., 2.])
second = torch.tensor([3., 4.])
print((first + second).tolist())
print((first @ second).item())
```

**Output**
```text
[4.0, 6.0]
11.0
```

`1+3` and `2+4` are the pair `[4, 6]`. The dot product is `1×3 + 2×4 = 11`. Those are the same operations you would write for an array. These two tensors only demonstrate that arithmetic. `requires_grad` is false unless you ask for tracking, so this addition does not record a graph and `backward` has nothing to fill in. The gradient lesson turns tracking on.

<!-- example: PT01-track -->
```python
import torch

plain = torch.tensor([1., 2.])
tracked = plain.clone().requires_grad_(True)
print(plain.requires_grad)
print(tracked.requires_grad)
```

**Output**
```text
False
True
```

`plain` is the same kind of tensor as the addition above. `clone` makes another tensor with the same numbers. `requires_grad_(True)` marks that copy so a later `backward` can fill its `.grad`. The original stays unmarked. No weight is updated here.

### A tensor

A tensor carries a shape, a storage type, and a device. `shape` says how many values sit along each axis. `dtype` says how each value is stored. `device` says whether those values live on the CPU or on a GPU.

<!-- example: PT01-shape -->
```python
import torch

values = torch.tensor([1., 2., 3.])
print(tuple(values.shape), str(values.dtype), values.device.type)
print(values[0].item())
```

**Output**
```text
(3,) torch.float32 cpu
1.0
```

Three numbers along one axis give shape `(3,)`. The dtype is a 32-bit float. This tensor is on the CPU. `.item()` reads one stored number back out as an ordinary Python float.

A tensor can share storage with a NumPy array. Automatic differentiation can additionally record how some tensors were calculated.

<!-- example: PT01-main -->
```python
import numpy as np
import torch

array = np.array([1., 2.], dtype=np.float32)
shared = torch.from_numpy(array)
independent = shared.clone()
shared[0] = 9
print(array.tolist())
print(independent.tolist())
print(tuple(shared.shape), str(shared.dtype), shared.device.type)
```

**Output**
```text
[9.0, 2.0]
[1.0, 2.0]
(2,) torch.float32 cpu
```

`from_numpy` shares compatible CPU storage; `clone` makes separate tensor storage. Moving a tensor with `.to(device)` returns a tensor on the chosen device; do not assume every conversion mutates the original in place. Tensor shape alone does not tell you whether an axis is batch, time or channels.

Use integer tensors for token/class IDs and floating tensors for features or probabilities as required by the operation. CUDA is NVIDIA's execution platform; MPS is Apple's Metal backend; CPU works without either. Availability and available memory are different questions. These examples deliberately use CPU so interpreting a shape does not require a GPU.

Sources: [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html), [torch.from_numpy](https://docs.pytorch.org/docs/2.9/generated/torch.from_numpy.html).

## PT02. Gradients

A **gradient** tells us how a small local change in a weight would change a result. `requires_grad=True` asks PyTorch to remember the operations that used the weight. `backward` writes the gradient into `.grad`. It does not change the weight.

<!-- example: PT02-scale -->
```python
import torch

weight = torch.tensor(2., requires_grad=True)
result = weight * 3
result.backward()
print(result.item(), weight.grad.item())
```

**Output**
```text
6.0 3.0
```

`2 × 3` is 6. The derivative of `3w` with respect to `w` is 3, so `.grad` is 3. The weight is still 2, because nothing subtracted the gradient from it.

The next calculation uses a squared error, which is the usual shape of a loss. Autograd applies the chain rule. Calling `backward` still only calculates the gradient.

<!-- example: PT02-main -->
```python
import torch

weight = torch.tensor(2., requires_grad=True)
prediction = 3 * weight
loss = (prediction - 1) ** 2
loss.backward()
print(loss.item(), weight.grad.item())
print(prediction.detach().requires_grad)
```

**Output**
```text
25.0 30.0
False
```

The prediction is 6, so the squared error is `(6−1)²=25`. The derivative is `2×(3w−1)×3`, which equals 30 at `w=2`. The weight remains 2: no optimizer step ran. `detach` creates a tensor disconnected from this gradient history; it does not train or improve the prediction.

Gradients accumulate by default. Repeating backward calculations without clearing or intentionally accumulating them changes the meaning. An optimizer later uses the gradients, but a gradient is not itself a claim of feature importance or causal influence in the real world. This isolated calculation teaches the mechanism; PT05 explains how the actual loop uses it.

Sources: [PyTorch autograd](https://docs.pytorch.org/docs/2.9/notes/autograd.html), [D2L automatic differentiation](https://d2l.ai/chapter_preliminaries/autograd.html).

## PT03. Modules

`nn.Linear` stores a weight and a bias and computes one weighted sum per output. The numbers below are written in by hand, so the result is arithmetic you can check.

<!-- example: PT03-linear -->
```python
import torch
from torch import nn

layer = nn.Linear(2, 1)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1., 2.]]))
    layer.bias.fill_(0.5)
with torch.inference_mode():
    print(layer(torch.tensor([[1., 1.]])).tolist())
```

**Output**
```text
[[3.5]]
```

`1×1 + 1×2 + 0.5 = 3.5`. `nn.Module` organizes a neural component and its parameters. A new module starts in training mode. `.eval()` switches it to evaluation mode. That switch changes layers such as dropout. It does not run the layer and it does not change the weights.

<!-- example: PT03-mode -->
```python
from torch import nn

model = nn.Linear(2, 1)
print(model.training)
model.eval()
print(model.training)
```

**Output**
```text
True
False
```

`model.training` is true immediately after construction. After `.eval()` it is false. Several layers are used in order. An activation adds a nonlinear transformation. Without nonlinearities, several linear layers can collapse into one linear map.

<!-- example: PT03-main -->
```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
with torch.no_grad():
    for parameter in model.parameters():
        parameter.fill_(0.5)
model.eval()
with torch.inference_mode():
    output = model(torch.tensor([[1., 2.]]))
print(output.tolist())
print(sum(parameter.numel() for parameter in model.parameters()))
```

**Output**
```text
[[3.5]]
13
```

All weights/biases are explicitly set to 0.5; none was learned. Each first-layer unit computes `1×.5+2×.5+.5=2`. ReLU keeps it positive. The final layer computes `3×2×.5+.5=3.5`. Parameter count is `(2×3+3)+(3×1+1)=13`.

A custom model subclasses `nn.Module`, calls `super().__init__()`, stores submodules as attributes and defines `forward`. Invoke `model(inputs)`, not a direct `forward` call that bypasses module call machinery/hooks. Parameters are learned tensors registered with the module; buffers are state such as running statistics that are not ordinary optimizer parameters. `.eval()` changes mode-dependent layers; it does not, by itself, disable gradient recording.

Sources: [PyTorch model tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), [D2L layers/modules](https://d2l.ai/chapter_builders-guide/model-construction.html).

### Define your own module when the computation needs a name

<!-- example: PT03-custom -->
```python
import torch
from torch import nn

class MeanEncoder(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.feature_count = feature_count

    def forward(self, inputs):
        if inputs.shape[-1] != self.feature_count:
            raise ValueError("unexpected feature width")
        return inputs.mean(dim=1)

encoder = MeanEncoder(2)
print(encoder(torch.tensor([[[1., 2.], [3., 4.]]])).tolist())
```

**Output**
```text
[[2.0, 3.0]]
```

The input is one example, two positions and two features. Reducing `dim=1` averages positions and retains both features. This module has no learned parameters, demonstrating that `nn.Module` is an organizational interface, not a synonym for “trained neural layer.” A padded sequence would additionally need masked averaging as shown in NC03.

## PT04. Batches

A batch is several examples kept together with their own targets. The feature row and the label at the same position belong to one example.

<!-- example: PT04-pair -->
```python
import torch

features = torch.tensor([[1., 2.], [3., 4.]])
labels = torch.tensor([0, 1])
print(tuple(features.shape), labels.tolist())
print(features[0].tolist(), int(labels[0]))
```

**Output**
```text
(2, 2) [0, 1]
[1.0, 2.0] 0
```

The feature table has two examples and two measurements. The first example's features are `[1, 2]` and its label is 0. A **Dataset** defines how to obtain those pairs. A **DataLoader** groups them into batches and can schedule loading work. Shuffling the feature rows without shuffling the labels in the same order would attach the wrong answer to an example.

<!-- example: PT04-main -->
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
labels = torch.tensor([0, 1, 0])
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
for inputs, targets in loader:
    print(tuple(inputs.shape), targets.tolist())
```

**Output**
```text
(2, 2) [0, 1]
(1, 2) [0]
```

The final batch contains one example, not two. Averaging batch means equally would overweight that batch when calculating an example-weighted epoch mean. Variable-length text additionally needs a `collate_fn` that pads/organizes several examples and creates masks/targets correctly.

Map-style datasets support indexed access; iterable datasets supply a stream. More loader workers can improve throughput but add process/memory complexity, and streaming workers must avoid duplicating data. Samplers determine which indices are visited; they are not a substitute for defining independent train/validation/test groups before loading. This code only inspects batches and performs no training.

Sources: [PyTorch data tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), [PyTorch data loading](https://docs.pytorch.org/docs/2.9/data.html).

## PT05. The training loop

This **read-only walkthrough** describes the order used by the supported neural lab. It is intentionally not executed and has no training-output panel. The caller supplies a model, optimizer, loaders, loss and device. It is a simplified classification example, not a replacement for the full lab worker/checkpoint/cancellation code.

```python walkthrough
def supervised_epoch(model, optimizer, train_loader, validation_loader, device):
    import torch
    from torch.nn import functional as F
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    validation_sum, validation_count = 0.0, 0
    with torch.inference_mode():
        for inputs, targets in validation_loader:
            logits = model(inputs.to(device))
            validation_sum += F.cross_entropy(logits, targets.to(device), reduction="sum").item()
            validation_count += len(targets)
    if validation_count == 0:
        raise ValueError("validation needs at least one example")
    return validation_sum / validation_count
```

Mode first: training enables behavior such as dropout. Clearing gradients prevents accidental carry-over; forward produces logits (raw class scores); cross-entropy compares those scores with the correct class IDs; backward calculates gradients; the optimizer changes parameters. PT06 works through this loss numerically. Validation changes mode and suppresses gradient recording. Its summed loss is divided by the number of examples, not the number of unequal batches.

The real DL lab adds settings, logs, best-development checkpoint selection, early stopping and saved evaluation. The test set must not choose the checkpoint. A weights-only save supports inference; exact resume additionally needs optimizer/scheduler/random/sampling state and cannot be assumed from a weights file.

**Run it in the existing lab:** **Deep learning → Training lab** (`dl-lab`) supports linear/MLP, CNN/residual CNN and its dense image autoencoder under compatible tasks. **Saved results** and **Compare models** own measured outputs. Local installation is required; reading this function never starts a job.

Sources: [PyTorch optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [saving/loading](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html). Local reference: `src/training_lab/model_lab.py`.

## PT06. Losses

A **logit** is an unnormalized score. Softmax turns one row of logits into fractions that add to 1. Equal logits become equal fractions.

<!-- example: PT06-softmax -->
```python
import torch
from torch.nn import functional as F

logits = torch.zeros(1, 3)
print([round(value, 3) for value in F.softmax(logits, dim=-1)[0].tolist()])
```

**Output**
```text
[0.333, 0.333, 0.333]
```

Three equal scores become about one third each. A classification head produces logits, and the loss compares them with the target under a specific convention. Applying softmax yourself before `cross_entropy` is unnecessary for the usual class-index case, because that loss already includes the normalization.

<!-- example: PT06-main -->
```python
import torch
from torch.nn import functional as F

class_logits = torch.zeros(2, 3)
class_targets = torch.tensor([0, 2])
binary_logits = torch.zeros(2)
binary_targets = torch.tensor([0., 1.])
print(f"class loss={F.cross_entropy(class_logits, class_targets).item():.6f}")
print(f"binary loss={F.binary_cross_entropy_with_logits(binary_logits, binary_targets).item():.6f}")
```

**Output**
```text
class loss=1.098612
binary loss=0.693147
```

Equal three-class logits assign each class probability `1/3`, giving loss `−log(1/3)`. A zero binary logit assigns probability one half. Multiclass class-index targets are typically integer IDs; BCE targets are floating values aligned to logits. Multilabel classification uses independent binary targets, not a single mutually exclusive class ID.

For token/segmentation losses, keep the class axis and ignored/padded targets correct. Regression uses numerical targets and a suitable loss such as MSE. Initialization, normalization, learning rate and optimizer affect training; gradient clipping cannot fix a wrong target or leaked feature. A useful debugging procedure is checking whether a supported model can overfit a very small subset—but that experiment belongs in the lab, not a fabricated output here.

Sources: [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html).

## PT07. Convolution

An image tensor in this lesson is arranged as batch, channels, height and width. A filter looks at a small window of that image.

<!-- example: PT07-window -->
```python
import torch

image = torch.arange(1., 10.).reshape(1, 1, 3, 3)
print(tuple(image.shape))
print(image[0, 0, :2, :2].tolist())
```

**Output**
```text
(1, 1, 3, 3)
[[1.0, 2.0], [4.0, 5.0]]
```

Shape `(1, 1, 3, 3)` is one image, one channel, three rows and three columns. The upper-left 2×2 window holds 1, 2, 4 and 5. A **convolutional filter** reuses the same numbers on every window. Stride controls how far the window moves. Padding controls the boundary. Pooling summarizes nearby outputs.

<!-- example: PT07-main -->
```python
import torch
from torch import nn

image = torch.arange(1., 10.).reshape(1, 1, 3, 3)
convolution = nn.Conv2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    convolution.weight.fill_(1.)
with torch.inference_mode():
    features = convolution(image)
    pooled = nn.MaxPool2d(2)(features)
print(features[0, 0].tolist())
print(tuple(features.shape), pooled.item())
```

**Output**
```text
[[12.0, 16.0], [24.0, 28.0]]
(1, 1, 2, 2) 28.0
```

The upper-left window sums `1+2+4+5=12`. The fixed all-one filter is a teaching choice, not a trained detector. Pooling selects the maximum from the 2×2 feature map. Increasing stride reduces output resolution; flattening later turns spatial features into classifier inputs, while adaptive pooling can produce a defined spatial size.

A residual block adds a compatible input path to a learned correction, helping information/gradients travel through deeper networks. A pretrained torchvision model needs its declared input transforms and weight revision; constructing an architecture with random weights does not give pretrained visual competence. Freezing parameters and switching evaluation mode do different jobs. Current lab CNN/residual training is supported; arbitrary torchvision transfer-learning training is not.

Sources: [PyTorch Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html), [D2L convolutions](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html), [torchvision models](https://docs.pytorch.org/vision/stable/models.html).

## PT08. Recurrent models

An RNN reads a sequence one position at a time and carries a hidden vector forward. With `batch_first=True`, the input shape is batch, positions, features.

<!-- example: PT08-rnn -->
```python
import torch
from torch import nn

inputs = torch.zeros(1, 3, 2)
model = nn.RNN(2, 4, batch_first=True)
model.eval()
with torch.inference_mode():
    output, hidden = model(inputs)
print(tuple(output.shape), tuple(hidden.shape))
```

**Output**
```text
(1, 3, 4) (1, 1, 4)
```

One example, three positions and two features go in. The output has four features at each of the three positions. The hidden state has one layer, one example and four features. The values are zeros in and an untrained network out, so only the shapes are the lesson. A GRU adds gates. An LSTM also carries a cell state. Bidirectionality runs a second pass from the other end of the sequence.

<!-- example: PT08-main -->
```python
import torch
from torch import nn

inputs = torch.zeros(2, 5, 4)
for kind in [nn.RNN, nn.GRU, nn.LSTM]:
    model = kind(4, 3, batch_first=True, bidirectional=True)
    model.eval()
    with torch.inference_mode():
        output, state = model(inputs)
    hidden = state[0] if isinstance(state, tuple) else state
    print(kind.__name__, tuple(output.shape), tuple(hidden.shape))
```

**Output**
```text
RNN (2, 5, 6) (2, 2, 3)
GRU (2, 5, 6) (2, 2, 3)
LSTM (2, 5, 6) (2, 2, 3)
```

The input is two examples, five positions and four features. Two directions each provide three hidden features, giving six features per output position. The returned hidden state's first axis indexes layer×direction, not batch; LSTM also returns a matching cell-state tensor. Random initialization affects numerical values, but the displayed dimensions are fixed by this configuration. No meaningful labels are predicted.

For padded sequences, length handling/packing prevents padding from being mistaken for real sequence content. In a bidirectional model, `output[:, -1]` is not generally the desired concatenation of both complete-sequence final states. A backward direction also sees future tokens and is unsuitable for an ordinary causal next-token predictor. The current DL lab has no recurrent trainer; these are forward-interface examples.

Sources: [PyTorch LSTM](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html), [GRU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRU.html), [D2L bidirectional RNNs](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html).

## PT09. Attention

Attention replaces each position with a mix of the value vectors it is allowed to read. With a causal rule, the first position can read only itself. The second can read both. When those two are allowed equally, the mix is their average.

<!-- example: PT09-mix -->
```python
values = [[2.0, 0.0], [0.0, 4.0]]
print(values[0])
print([(values[0][i] + values[1][i]) / 2 for i in range(2)])
```

**Output**
```text
[2.0, 0.0]
[1.0, 2.0]
```

The first value stays `[2, 0]`. The average of `[2, 0]` and `[0, 4]` is `[1, 2]`. Queries and keys produce the scores that decide those weights. The output is mixed information, not automatically a class prediction.

<!-- example: PT09-main -->
```python
import torch
from torch.nn import functional as F

queries = torch.zeros(1, 1, 2, 2)
keys = torch.zeros(1, 1, 2, 2)
values = torch.tensor([[[[2., 0.], [0., 4.]]]])
output = F.scaled_dot_product_attention(queries, keys, values,
                                        is_causal=True, dropout_p=0.0)
print(output[0, 0].tolist())
```

**Output**
```text
[[2.0, 0.0], [1.0, 2.0]]
```

All allowed scores are equal. Position zero can read only the first value. Position one can read both and averages them: `([2,0]+[0,4])/2=[1,2]`. Multiple heads perform several projections/mixes before combining results. Residual paths preserve earlier representations, normalization controls feature scale and a feed-forward block transforms each position's features.

Encoder self-attention can read both directions; decoder causal attention cannot. Encoder–decoder cross-attention uses decoder queries with source-side keys/values. PyTorch attention APIs differ in Boolean-mask conventions: check the exact function instead of reusing a mask blindly. SDPA applies its specified dropout probability, so pass zero explicitly for evaluation. The existing architecture diagram owns the detailed Transformer explanation; this code links its mask/mix operations to actual tensors.

Sources: [PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html), [D2L attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html).

## PT10. Generative models

An autoencoder reconstructs input through a bottleneck. A VAE represents uncertainty in a latent distribution; a GAN compares generated samples through an adversarial objective; diffusion/denoising methods reverse a corruption process. These are different tasks/objectives, not names for one universal training loop.

<!-- example: PT10-main -->
```python
import torch

mean = torch.tensor([1., 2.])
log_variance = torch.zeros(2)
fixed_noise = torch.tensor([-1., 1.])
latent = mean + torch.exp(0.5 * log_variance) * fixed_noise
original = torch.tensor([0., 1.])
reconstruction = torch.tensor([0.2, 0.8])
print(latent.tolist())
print(f"reconstruction MSE={((original - reconstruction) ** 2).mean().item():.2f}")
```

**Output**
```text
[0.0, 3.0]
reconstruction MSE=0.04
```

The VAE expression separates supplied noise from mean/scale, allowing differentiation through the distribution parameters. Unit variance means unit standard deviation here; the factor `.5` converts log variance to log standard deviation. This is a fixed latent calculation, not a generated-image quality result.

A VAE balances reconstruction with a distribution regularizer, a GAN alternates distinct generator/discriminator objectives, and denoising predicts a defined target at a specified noise level. Contrastive learning instead changes similarity relations between representations using pairs/negatives under a chosen objective. The lab currently trains its dense image autoencoder, not VAE/GAN/diffusion/contrastive models. Additional forward/objective recipes are tracked separately rather than pretending this one snippet implements all four.

Sources: [D2L autoencoders/representation context](https://d2l.ai/), [VAE paper](https://arxiv.org/abs/1312.6114), [DDPM paper](https://arxiv.org/abs/2006.11239).

### Separate adversarial objectives

In a common non-saturating GAN formulation, a discriminator distinguishes real from generated inputs and the generator tries to make generated inputs score as real. We can inspect the two objectives using fixed discriminator logits without updating either network.

<!-- example: PT10-gan -->
```python
import torch
from torch.nn import functional as F

real_logit = torch.tensor(2.)
fake_logit = torch.tensor(-1.)
discriminator_loss = F.softplus(-real_logit) + F.softplus(fake_logit)
generator_loss = F.softplus(-fake_logit)
print(f"discriminator loss={discriminator_loss.item():.6f}")
print(f"generator loss={generator_loss.item():.6f}")
```

**Output**
```text
discriminator loss=0.440190
generator loss=1.313262
```

`softplus(x)=log(1+exp(x))` supplies numerically stable logistic-loss terms. The discriminator's real example is already scored positively and fake example negatively, so its loss is relatively small. The generator faces that same negative fake score and has a larger loss. A real adversarial procedure controls which parameters receive gradients during separate phases; no such training is performed here. Source: [GAN paper](https://arxiv.org/abs/1406.2661).

### Follow one denoising target

<!-- example: PT10-denoising -->
```python
import math

alpha_bar = 0.64
clean_value, noise = 2., 1.
noisy_value = math.sqrt(alpha_bar) * clean_value + math.sqrt(1-alpha_bar) * noise
predicted_noise = 0.5
reconstructed = (noisy_value - math.sqrt(1-alpha_bar) * noise) / math.sqrt(alpha_bar)
print(f"noisy={noisy_value:.1f}, noise-prediction error={(predicted_noise-noise)**2:.2f}")
print(f"recovered with true noise={reconstructed:.1f}")
```

**Output**
```text
noisy=2.2, noise-prediction error=0.25
recovered with true noise=2.0
```

The forward corruption mixes clean signal and supplied noise. A noise-prediction objective compares the predicted noise against that supplied target. Recovering a clean estimate with the true noise is an algebra check, not a complete reverse diffusion sampler; sampling also needs a schedule, model predictions and a defined update rule. Source: [DDPM](https://arxiv.org/abs/2006.11239).

### A contrastive objective compares alternatives

<!-- example: PT10-contrastive -->
```python
import torch
from torch.nn import functional as F

similarity_logits = torch.tensor([[2., 0.]])
positive_candidate_index = torch.tensor([0])
print(f"contrastive classification loss={F.cross_entropy(similarity_logits, positive_candidate_index).item():.6f}")
```

**Output**
```text
contrastive classification loss=0.126928
```

For this row, candidate zero is the designated positive. Increasing its score relative to the negative reduces this loss. Temperature scaling changes score sharpness; negative construction and false negatives affect what the representation learns. This one row illustrates a contrastive classification objective, not every self-supervised method or a trained encoder. Source: [SimCLR](https://arxiv.org/abs/2002.05709).

## PT11. Graph layers

A graph contains **nodes** and **edges**. Node features form a matrix; edge indices specify which source sends information to which target. A message-passing calculation aggregates permitted neighbours and may then transform the result.

<!-- example: PT11-main -->
```python
import torch

node_features = torch.tensor([[1.], [2.], [4.]])
source = torch.tensor([0, 2])
target = torch.tensor([1, 1])
received = torch.zeros_like(node_features)
received.index_add_(0, target, node_features[source])
print(received.squeeze(1).tolist())
print((node_features + received).squeeze(1).tolist())
```

**Output**
```text
[0.0, 5.0, 0.0]
[1.0, 7.0, 4.0]
```

Nodes zero and two send features to node one, which receives `1+4=5`. Adding its own feature produces 7. This is an unnormalized sum illustration, not the precise definition of every GCN/GAT layer. An undirected relation often requires explicit edges in both directions; adding a self-loop changes the aggregation too.

PyG's Data convention uses `x` for node features and `edge_index` shaped `(2, edges)`. Graph classification pools node states into one graph representation; node classification predicts per node. Batching several graphs offsets their node indices and records graph membership. Never let test labels enter the feature graph merely because the graph is shared. The lab has no GNN trainer; the PyTorch arithmetic above requires no PyG installation.

Sources: [PyG introduction](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html), [PyTorch index_add](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.index_add_.html).

## PT12. Saved weights

A **state dictionary** maps parameter/buffer names to tensors. Saving/loading it can reproduce a fixed model's forward behavior when architecture/configuration match. It does not by itself preserve optimizer state, training data, a split protocol or exact resume history.

<!-- example: PT12-main -->
```python
import io
import torch
from torch import nn

model = nn.Linear(2, 1)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[2., -1.]]))
    model.bias.fill_(1.)
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)
restored = nn.Linear(2, 1)
restored.load_state_dict(torch.load(buffer, weights_only=True))
restored.eval()
with torch.inference_mode():
    print(restored(torch.tensor([[3., 2.]])).tolist())
print(sorted(restored.state_dict()))
```

**Output**
```text
[[5.0]]
['bias', 'weight']
```

The parameters were deliberately supplied, not trained. The output is `3×2−2+1=5`. The in-memory save avoids creating an unexplained binary fixture. Load only trusted artifacts; `weights_only` reduces loading scope but is not a universal safety certificate for arbitrary files or every resource-exhaustion risk.

TensorBoard displays recorded scalar events; a profiler measures operation costs. Mixed precision changes numerical/storage choices; gradient accumulation changes how several microbatches contribute to one update. Compilation and distributed execution introduce their own constraints. These are not automatically enabled by saving a model, and no new training/benchmark is run here. Use existing DL results/comparison pages for measured curves and compatible model comparisons, with the same data/split/budget boundaries.

Sources: [PyTorch save/load tutorial](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html), [PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html).
