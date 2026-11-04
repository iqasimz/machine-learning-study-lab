# DL06. Residual networks and transfer learning

A residual block adds its input to a transformed version of that input: output = input + correction. This shortcut lets information and gradients use a direct path around the transformation. Transfer learning is a separate choice: reuse a pretrained model and decide which parameters to freeze or adapt. Residual architectures can be trained from scratch or transferred.

The input is 2 and the transformed branch multiplies it by a chosen scale. We calculate the addition and its derivative. A separate section then compares a frozen 1,000-parameter backbone with a trainable 20-parameter head; it is not another operation on the scalar output.

The CNN lesson stacks transformations. A residual connection changes how those transformations are connected; freezing changes which weights an optimizer may update. Keep these two decisions separate.

[course map](../COURSE_MAP.md). Required background: DL05, DL03.

## 1. Shortcut structure versus adaptation policy

**Key terms.** A **residual block** adds a learned change to a skip path; a **projection shortcut** aligns incompatible shapes. A **backbone** supplies features; a **head** maps them to a task. **Transfer learning** reuses pretrained representations.

**Process.** Choose scratch weights or a compatible pretrained backbone, replace the task head if needed and declare which layers are trainable. Forward through the residual network, fit the allowed parameters and compare against the unchanged starting representation.

Writing a block as x+F(x) supplies a direct additive path. It is not concatenation and does not guarantee stable learning at arbitrary depth. If channel or spatial shapes change, use a justified projection or another specified shortcut. The normalization/order must match the actual architecture.

A frozen feature extractor still needs forward computation, memory and a trained new head. Full fine-tuning adds optimizer/gradient requirements and may damage useful prior behaviour. Training/evaluation modes of BatchNorm require particular care when freezing a backbone. Pretrained weights can contain evaluation-related exposure; source and revision are part of the comparison.

## 2. A worked calculation

**Symbols and assumptions.** x is the block input and F(x) its learned branch. y=x+F(x) requires equal shapes; otherwise use y=P(x)+F(x) with a declared projection P.

For x=[1,2] and F(x)=[0.1,−0.2], residual output is [1.1,1.8]. Adding [1,2] to a three-feature vector is undefined under the intended residual contract. A [batch,64] feature extractor with a ten-class linear head adds 64×10+10=650 trainable parameters if only that head is trained.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Scratch, frozen and fine-tuned comparisons

Compare scratch training, frozen features and full fine-tuning in separate rows. Their starting information differs; a pretrained gain is not solely an architecture effect. Use smaller rates or staged unfreezing only when development evidence and controlled budgets justify them.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: avoid training-data reuse

The first local lab should use a small scratch residual model or an explicitly supplied compatible feature extractor. Do not silently resize grayscale clothing images into a pretrained RGB pipeline and call the result a fair scratch-vs-transfer study.

**Check.** Why is requiring_grad=False not always enough to freeze every behavioural aspect of a backbone?

**Answer.** Non-parameter running state and mode-dependent layers can still change. Specify BatchNorm/dropout behaviour and verify the saved state as well as parameter gradients.

## 5. Residual derivatives and frozen-module state

**Key terms.** A residual block returns x+F(x). An identity shortcut requires equal shapes. A projection shortcut learns a shape-changing map. Freezing a parameter disables its gradient update, not necessarily all state changes.

**Process.** Send the input through a learned correction branch and a shortcut, add the equal-shaped outputs and continue. During backward propagation both paths contribute.

For y=x+F(x), dy/dx=I+∂F/∂x. The identity term supplies a direct path even if the learned branch initially contributes little. It does not guarantee that every gradient stays well-scaled or that an arbitrarily deep network will optimize.

### Transfer learning needs an access and state policy

A frozen backbone can still change BatchNorm running means if left in training mode. A proper frozen-feature comparison fixes both learned weights and relevant state; dropout also changes behaviour between training and evaluation. A pretrained encoder may already have seen evaluation data, so a new task split does not prove an untouched representation.

The local residual CNN starts from random weights and uses same-channel identity blocks. It is not pretrained transfer learning. Comparing it to the plain CNN changes depth, parameter count and compute; the saved settings make those differences visible. External-backbone transfer remains a course topic rather than a hidden automatic download.

## 6. Contrastive image–text learning and multimodal transfer

**Read after DL08 attention and the embedding/retrieval foundations.** A multimodal model connects information from different kinds of input, such as images and text. In a CLIP-style contrastive system, separate encoders map an image and its paired text into a shared comparison space. This is representation learning, not by itself an image generator or a chatbot that can inspect arbitrary images.

A training batch contains B matched image–text pairs. Normalize their embeddings, form all B×B similarities, and train the matching pair to score above the other batch entries. A common row loss is:

\[
L_i=-\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B}\exp(s_{ij}/\tau)}.
\]

s is a similarity and tau a positive temperature. The symmetric objective also predicts the matching image from each text. A real batch may contain near-duplicate or equally valid descriptions: treating every off-diagonal pair as negative is an assumption, not ground truth about meaning.

For one image with two text similarities 1 and 0, temperature 1 assigns its matched text probability e/(e+1)=0.7311 and row loss 0.3133 nats. Temperature 0.5 gives probability 0.8808 and loss 0.1269 for these fixed scores. This sharper distribution does not establish improved recognition or calibration; changing temperature in training also changes gradients.

For zero-shot classification, encode descriptions of candidate classes and compare each with the image embedding. “Zero-shot” here means no supervised adaptation to that target task under the stated protocol, not that the pretrained model saw no relevant data. Prompt wording, class-set choice, source exposure and image preprocessing affect the result. A frozen encoder can also support retrieval or a fitted downstream head.

**Practice.** An image–text model finds a matching caption. Has it generated an image or learned the target task online? **Answer.** Neither follows. Similarity retrieval, supervised adaptation and conditional generation are different operations. This distinction leads naturally to conditioning in DL10 without duplicating the Transformer internals owned by T3.

[CLIP primary paper](https://arxiv.org/abs/2103.00020).

## Sources

- [Deep Residual Learning](https://arxiv.org/abs/1512.03385).
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
