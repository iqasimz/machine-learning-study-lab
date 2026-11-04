# DL09. Autoencoders and self-supervised representations

An autoencoder learns to encode an input into a representation and decode it back into a reconstruction. A bottleneck can limit how much information passes through, but accurate copying alone does not prove the representation is useful for another task. Self-supervised learning is broader: its training targets are derived from the data rather than supplied as separate human labels.

For input [1,0], a one-feature code averages the coordinates and reconstructs [0.5,0.5]. A two-feature identity map copies it exactly. These are fixed maps, not trained encoders. A separate final section introduces contrastive learning as a different objective.

PCA also compresses and reconstructs using a linear subspace. Autoencoders can learn nonlinear mappings; contrastive and masked-prediction methods ask them to preserve different information.

[course map](../COURSE_MAP.md). Required background: DL04.

## 1. Reconstruction and representation learning

**Key terms.** An **encoder** maps inputs to features; a **decoder** reconstructs them; a **latent bottleneck** limits representation size. **Self-supervision** obtains targets from data itself; **contrastive learning** compares matched and mismatched views.

**Process.** Construct a learning signal from permitted training data, encode inputs and optimize reconstruction or representation agreement. Freeze or adapt the representation for a separate downstream task, then evaluate on independent cases.

An autoencoder can learn useful compression, but a powerful unconstrained decoder may reconstruct without a useful task representation. Denoising asks it to reconstruct clean input from a corrupted view. PCA is a linear reference; a neural reconstruction gain does not automatically imply a classification gain.

Contrastive learning uses positive views believed to preserve relevant identity and other views as negatives under a stated recipe. Augmentation choices define invariances: a transformation that changes the desired label is not harmless. Unlabelled test data still constitutes extra exposure in a transductive setting; do not secretly use it in an inductive comparison.

## 2. A worked calculation

**Symbols and assumptions.** z=encoder(x) is a latent representation; x-hat=decoder(z) is reconstruction. Squared reconstruction error can be summed or averaged over coordinates; state the reduction. The live fixed-map walkthrough sums two squared coordinate errors, while the following examples explicitly report their mean. In a contrastive softmax, temperature tau divides similarity scores before normalization.

For input [1,0] and reconstruction [0.8,0.1], mean squared reconstruction error is (0.2²+0.1²)/2=0.025. For one positive similarity 1 and one negative similarity 0 at tau=1, the positive softmax probability is exp(1)/(exp(1)+1)≈0.7311 and its log loss is 0.3133. This two-candidate calculation is not the complete SimCLR batch construction.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Linear probes, PCA and fine-tuning

Compare raw pixels, train-fitted PCA and an autoencoder with a fixed downstream classifier. A linear probe tests information accessible to a linear head; full fine-tuning changes the representation as well. Reconstruction, embedding geometry and downstream utility are distinct measurements.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: evaluate more than reconstructions

Use digits first, then a declared Fashion-MNIST training subset. Save reconstruction examples selected by a fixed rule, including bad ones; do not report only attractive outputs.

**Check.** Why is making every pair of examples a positive pair dangerous?

**Answer.** The signal no longer distinguishes cases and can favour collapsed or uninformative features. The exact objective and negative/variance controls determine whether collapse is prevented.

## 5. Reconstruction versus contrastive objectives

**Key terms.** An encoder maps inputs into representations. A decoder reconstructs inputs. A positive pair shares a chosen semantic relation; negatives contrast with it. Temperature scales a contrastive softmax.

**Process.** Decide which information should be preserved, construct the training signal, update the representation and evaluate its downstream usefulness separately from its training objective.

An autoencoder minimizes a declared reconstruction objective. For two coordinates, the live walkthrough's summed error 0.5 is mean squared error 0.25 after division by two; they describe the same errors under different reductions. With target pixel vector [1,0] and reconstruction [0.8,0.3], mean squared error is (0.04+0.09)/2=0.065. A narrow bottleneck restricts capacity but does not guarantee meaningful factors: background texture may dominate the easiest reconstruction signal.

### Contrastive arithmetic

If a positive similarity is 0.8, one negative similarity is 0.2 and temperature is 0.2, the two softmax logits are 4 and 1. Positive probability is exp(4)/(exp(4)+exp(1))≈0.9526 and its log loss is about 0.0486. With temperature 1, probability is about 0.6457. Sharper probabilities change gradient emphasis; they do not by themselves prove better embeddings.

The local autoencoder learns image reconstruction only and is compared with a mean training image. It is not a contrastive trainer. Never report lower reconstruction error as a measured gain in class prediction, retrieval or transfer without actually evaluating those tasks.

## Sources

- [SimCLR](https://arxiv.org/abs/2002.05709).
- [Deep Learning, author site](https://www.deeplearningbook.org/).
