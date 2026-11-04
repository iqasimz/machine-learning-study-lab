# Deep learning

Read in order unless the prerequisites are already familiar. All fourteen lessons are available in Model Lab and share the sidebar/Interactive reader. Definitions come before calculations; architecture diagrams show actual dependencies rather than forcing unrelated decisions into a chain. The local training lab supports MLPs, CNNs, residual CNNs and image autoencoders. Other families have worked teaching calculations, not implemented trainers.

- **DL01** — [Neural networks and multilayer perceptrons](01_FROM_LINEAR_MODELS_TO_MLPS.md). Show why stacked linear maps remain linear. Calculate an XOR representation and MLP parameter count.
- **DL02** — [Backpropagation and automatic differentiation](02_BACKPROPAGATION_AND_AUTOGRAD.md). Compute a complete chain-rule update. Check gradients without changing saved model state.
- **DL03** — [Optimizers, initialization and training stability](03_OPTIMIZATION_AND_INITIALIZATION.md). Trace SGD/momentum/Adam update state. Explain initialization scale and BatchNorm train/eval behaviour.
- **DL04** — [Regularization, dropout and training diagnostics](04_REGULARIZATION_AND_TRAINING_DIAGNOSTICS.md). Diagnose underfitting, overfitting and numerical faults. Explain dropout, label smoothing and update budgets.
- **DL05** — [Convolutional neural networks](05_CONVOLUTIONAL_NETWORKS.md). Calculate output shape, weights and receptive field. Explain parameter sharing and spatial assumptions.
- **DL06** — [Residual networks and transfer learning](06_RESIDUAL_NETWORKS_AND_TRANSFER.md). Trace a skip connection and projection. Separate scratch, frozen features and pretrained fine-tuning.
- **DL07** — [Recurrent networks, GRUs and LSTMs](07_RECURRENT_SEQUENCE_MODELS.md). Trace recurrent state and truncated backpropagation. Compare RNN, GRU and LSTM mechanisms.
- **DL08** — [Attention and Transformer building blocks](08_ATTENTION_AND_TRANSFORMERS.md). Calculate a masked attention row. Compare recurrence, attention and existing LLM training.
- **DL09** — [Autoencoders and self-supervised representations](09_AUTOENCODERS_AND_SELF_SUPERVISION.md). Calculate reconstruction loss and contrastive scoring. Choose augmentations that preserve the downstream task.
- **DL10** — [Generative models: VAEs, GANs and diffusion](10_GENERATIVE_MODEL_FAMILIES.md). Compare VAE, GAN and diffusion objectives. Distinguish a mechanism demonstration from generative quality.
- **DL11** — [Graph neural networks](11_GRAPH_NEURAL_NETWORKS.md). Calculate one permutation-invariant message-passing update. Distinguish transductive information from label leakage.
- **DL12** — [Prediction heads, targets and losses](12_TASK_HEADS_AND_LOSSES.md). Match output shapes and losses to task semantics. Calculate intersection-over-union and separate classification from segmentation.
- **DL13** — [The neural training loop and its system state](13_TRAINING_SYSTEMS_AND_DEPLOYMENT.md). Account for model/gradient/optimizer/activation memory. Save and restore state and measure actual latency.
- **DL14** — [Compare neural and classical models](14_ML_VERSUS_DL_CAPSTONE.md). Run a matched method comparison without assuming a winner. Separate model-class effects from exposure and tuning-budget effects.

Return to [the combined course map](../COURSE_MAP.md).
