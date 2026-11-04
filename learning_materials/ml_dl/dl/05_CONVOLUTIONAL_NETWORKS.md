# DL05. Convolutional neural networks

A convolutional neural network, or CNN, learns local pattern detectors and reuses their weights across image positions. Each filter produces a feature map. Stacking convolutions and nonlinearities lets later features combine earlier patterns; pooling can reduce spatial resolution, and a final head maps the remaining features to task outputs.

The walkthrough uses one 8×8 grayscale image, two convolution blocks with ReLU (zeroing negative values) and max pooling (keeping local maxima), and three illustrative classes. Kernel size sets filter width; stride sets its movement; padding supplies boundary zeros here. All weights are fixed teaching values, so the example explains computation rather than measuring image-recognition accuracy.

A dense layer combines all input features at once; a convolution first combines local patches with shared weights. The final dense head still uses the weighted-sum and classification ideas introduced earlier.

[course map](../COURSE_MAP.md). Required background: DL01.

## 1. Local filters, channels and receptive fields

**Key terms.** A **convolution kernel** is a shared local filter; **channels** are feature maps; **stride** controls movement; **padding** extends boundaries; **dilation** spaces kernel samples. A **receptive field** is the input region that can influence an output.

**Process.** An image tensor enters local shared filtering, activation and optional spatial downsampling. Later layers combine larger regions, then a task head converts features to class or structured outputs. Track spatial sizes and channels at each transition.

A dense layer connects every flattened pixel to each output, while a convolution reuses a local weight pattern at different positions. This encodes locality and sharing assumptions, not perfect invariance. Pooling/downsampling changes spatial resolution; padding and stride affect boundary and alignment behaviour. Standard deep-learning Conv2d commonly performs cross-correlation rather than reversing the kernel as a mathematical convolution would.

Convolution arithmetic is worth a static patch/filter diagram. A decorative image of a robot is not. Visualize one patch, shared weights and the corresponding output cell, then show how layers expand receptive fields.

## 2. A worked calculation

**Symbols and assumptions.** For one spatial dimension, output length is floor((H+2p−d(k−1)−1)/s+1). Here H is input size, k kernel size, p padding, d dilation and s stride. groups divides channel connectivity.

For H=28,k=3,p=1,d=1,s=1, output size is 28. With stride 2 it is floor(27/2+1)=14. A biased Conv2d with 1 input channel, 16 output channels and a 3×3 kernel has 16×1×3×3+16=160 parameters. Two stride-1, dilation-1 3×3 layers have a theoretical 5×5 receptive field, not 6×6.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Kernel, stride and pooling choices

Larger filters, more channels, more depth and dilation enlarge different aspects of capacity/context with different costs. A 1×1 convolution mixes channels without enlarging spatial neighbourhood. Groups/depthwise convolution change connectivity and weight count; they are not merely fewer output channels.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare image architectures

Compare a linear classifier, MLP and small CNN on Fashion-MNIST with the official test reserved and a separate training-derived development split. Start with a bounded subset for a pilot, but label subset results distinctly from full-data results.

**Check.** Why must the classifier head change if spatial pooling changes the flattened feature size?

**Answer.** Its input dimension must match the new tensor shape. Global average pooling is an alternative design, not a reason to silently reshape incompatible arrays.

## 5. Shapes through the actual local CNN

**Key terms.** Channels are learned feature maps. Padding preserves boundary access; stride controls movement. Pooling reduces spatial resolution. Flattening concatenates spatial positions before the classification head.

**Process.** The lab's CNN applies a padded 3×3 convolution, normalization/activation and 2×2 pooling twice. It then flattens the retained map, applies dropout and a linear head. The residual variant inserts same-shape shortcut blocks.

For a 28×28 input, first width 32 and ten classes without normalization: Conv 1→32 has 32×1×9+32=320 parameters. The second Conv 32→64 has 64×32×9+64=18,496. Pooling twice gives 7×7; the linear head uses 64×7×7×10+10=31,370. Total: 50,186. On 8×8 digits, the final map is 2×2, so the same channel choices produce 21,386 parameters. The interface records the actual count.

### Do not confuse this with global pooling

Global average pooling would average each final channel into one number and use a smaller head. That is an alternative architecture, not the current local implementation. A fixed 3×3 filter is shared at every location; this reduces free parameters relative to an equally wide dense spatial map but does not make the model fully translation invariant.

## 6. Vision Transformers: turn image patches into a sequence

**Read after DL08 attention.** A Vision Transformer (ViT) represents an image as a sequence of patch vectors and mixes them with Transformer blocks. It is an alternative to the CNN's built-in local filtering pattern, not a promise that attention is always a better image model.

Begin with an image of height H, width W and C channels. Divide it into non-overlapping P×P patches, assuming both dimensions are divisible by P. Flatten each patch into P²C values and multiply by a learned projection to hidden width D. Add the architecture's position information so identical patch content at different locations is distinguishable. The original ViT adds a learned class token; other designs pool or use dense prediction heads differently.

\[
N=HW/P^2,\qquad X_{patch}\in\mathbb R^{B\times N\times(P^2C)},\qquad E=X_{patch}W_E.
\]

N counts patches and B counts images. A 224×224 RGB image with 16×16 patches has 196 patches, each containing 768 raw values. With D=192, the projected patch tensor is [B,196,192]; adding a class token gives [B,197,192]. Projection weights alone number 768×192=147,456, or 147,648 with a 192-value bias. Raw patch width 768 does not force hidden width to be 768.

Image → patch vectors → learned embedding/positions → attention/MLP blocks → task head is the computation. With global attention, each head forms a 197×197 score map in this example. Halving patch width to eight quadruples patch count to 784; the pairwise score count grows approximately sixteen-fold, before considering special tokens. Windowed/hierarchical attention restricts connectivity or changes resolution to control this cost. Patch size also changes spatial detail and learned inductive assumptions.

A CNN already shares local filters; a ViT learns interactions over its permitted patches. Compare data exposure, pretraining, augmentation, resolution, compute and adaptation, not just architecture labels. The original ViT's strong transfer results depended on substantial pretraining. A small scratch ViT is not guaranteed to outperform a small CNN on Fashion-MNIST.

**Practice.** Does a class token represent a ground-truth category supplied at the input? **Answer.** No. It is a learned input vector whose final representation can feed a classifier. It must not encode the correct test label. Continue to DL12 for changing an image representation into boxes or pixel maps; keep the original CNN walkthrough as the local-filter reference.

[Original Vision Transformer](https://arxiv.org/abs/2010.11929) · [Swin Transformer](https://arxiv.org/abs/2103.14030).

## Sources

- [PyTorch 2.9 Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html).
- [Computing Receptive Fields of CNNs](https://distill.pub/2019/computing-receptive-fields/).
- [Stanford CS231n, Spring 2026](https://cs231n.stanford.edu/).
