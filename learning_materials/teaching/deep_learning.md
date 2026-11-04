# Deep learning

## DL01. Neural networks and multilayer perceptrons

A neural network composes adjustable functions. A dense layer forms weighted sums; an activation changes their shape; another layer combines the resulting features. The intermediate values are called **activations**, while weights and biases are learned parameters. Increasing the batch size creates more activations, not more model parameters.

Without nonlinear activations, stacking affine layers still gives one affine transformation. A hidden nonlinearity lets the composition represent rules that one straight boundary cannot. XOR is a small example: the two different input bits should be positive, while equal bits should be negative. One line cannot separate these four cases. Hand-built hidden units $h_1=\operatorname{ReLU}(x_1+x_2)$ and $h_2=\operatorname{ReLU}(x_1+x_2-1)$ give output $h_1-2h_2$ equal to 0, 1, 1, 0. This demonstrates representational ability, not proof that training will find those weights.

A layer mapping d features to h outputs has $dh+h$ parameters when it includes biases. A 2→3→2 network therefore has $2(3)+3+3(2)+2=17$. Width changes how many features a layer can form; depth changes how many transformations are composed. Neither guarantees better held-out performance. Choose an output head and loss matching the task, then evaluate against simpler models.

Sources: [D2L: multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html), [Deep Learning: deep feedforward networks](https://www.deeplearningbook.org/contents/mlp.html).

## DL02. Backpropagation and automatic differentiation

Backpropagation efficiently calculates how each parameter affects a scalar loss. It traverses the operations used in the forward pass in reverse, multiplying local sensitivities and adding contributions where a value was used along several paths. The optimizer is a separate operation that consumes those gradients.

Take $a=w_1x$, $\hat y=w_2a$, and $L=(\hat y-y)^2$. With $x=2$, $w_1=1$, $w_2=3$, and target 4, the forward values are a=2, prediction 6, and loss 4. Starting backwards, $\partial L/\partial\hat y=4$. Then $\partial L/\partial w_2=4a=8$, and $\partial L/\partial w_1=4w_2x=24$. The first weight matters through both later operations. Updating only the last layer is not equivalent to training the whole network.

Automatic differentiation records supported operations and applies their derivative rules; it does not estimate gradients by repeatedly perturbing every weight. A finite-difference check is a separate diagnostic for a small function. In PyTorch, gradient buffers accumulate unless cleared. `eval()` changes behaviours such as dropout; `no_grad()` disables gradient recording. They solve different problems, so validation often uses both. A detached tensor can interrupt the intended path even when its numerical value looks correct.

Sources: [D2L: computational graphs and backpropagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html), [PyTorch: autograd mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html).

## DL03. Optimizers, initialization and training stability

Once gradients exist, an optimizer decides how to change parameters. SGD uses the current estimated gradient; momentum carries a running direction; Adam tracks moving averages of gradients and squared gradients to adapt update scales by coordinate. These history variables are optimizer state, not extra predictions.

For a first scalar gradient 2 with Adam beta values 0.9 and 0.999, the uncorrected moments are 0.2 and 0.004. Bias correction divides by $1-0.9$ and $1-0.999$, giving 2 and 4. Ignoring epsilon, their ratio is $2/\sqrt4=1$, so a learning rate of 0.01 produces an approximately 0.01 update, not SGD's 0.02. Later updates depend on accumulated history. Epsilon stabilises a small denominator.

Initialisation sets the starting signal scale and breaks symmetry between hidden units. Identically initialised hidden units can receive identical updates and fail to learn diverse features. This is different from a simple convex linear model where zero initialisation may be fine. Normalisation controls activation statistics; clipping limits a gradient norm; a schedule changes learning rate over time. These are distinct interventions.

Check non-finite values, data scale, loss reduction, and gradient norms before blaming the optimizer. AdamW's decoupled weight decay is not generally identical to putting an L2 penalty inside Adam's gradient calculation.

Sources: [D2L: Adam](https://d2l.ai/chapter_optimization/adam.html), [D2L: numerical stability and initialisation](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html), [AdamW paper](https://arxiv.org/abs/1711.05101).

## DL04. Regularization, dropout and training diagnostics

A network can learn patterns specific to its training sample. Regularisation changes fitting so those patterns have less influence. Dropout randomly removes selected activations during training; weight decay discourages large weights; early stopping chooses a checkpoint before further fitting damages validation performance. These techniques address different mechanisms and are not substitutes for correct data splits.

With **inverted dropout**, let q be the probability of keeping an activation. A kept value is divided by q, while a dropped value becomes zero. For activation 4 and q=0.5, the output is either 8 or 0, each with probability 0.5. Its expectation is 4. This preserves the individual activation's mean, not every nonlinear downstream output. At evaluation, ordinary dropout is disabled and no such random rescaling is applied.

Keep “drop probability” and “keep probability” distinct: $q=1-p_{drop}$. `eval()` selects evaluation behaviour; it does not prevent gradients. Conversely, `no_grad()` does not automatically disable dropout.

Interpret curves alongside the objective and data. Rising validation error while training error falls can suggest overfitting; high error on both may indicate underfitting or a bug. Adding dropout to an already struggling fit can make it worse. Verify masks, targets, and learning rate before adding constraints.

Sources: [D2L: dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html), [PyTorch: Dropout](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Dropout.html).

## DL05. Convolutional neural networks

A convolutional layer searches for local patterns using the same filter weights at different positions. For images, nearby pixels often form meaningful edges or textures, and similar patterns can occur in different places. This **inductive bias** is useful when the input's spatial arrangement matters; it is usually inappropriate for an arbitrary ordering of unrelated table columns.

An image tensor commonly has axes `[batch, channels, height, width]`. A filter combines a spatial patch across all input channels to produce one output value. Sliding it forms a feature map. With one input channel and a 3×3 filter, the filter has nine weights plus a bias. Ten such filters produce ten channels without needing a separate filter for every image location.

Stride sets how far the filter moves; padding extends the boundary. Without dilation, output size per axis is $\lfloor(H+2P-K)/S\rfloor+1$. For H=8, K=3, P=1, S=2, the result is 4. ReLU changes values without learning additional weights. Pooling summarises local regions; flattening changes representation shape before a dense classifier.

The diagram follows two convolutions and a class head, showing actual small arithmetic. Its gradient example updates the head only. A real CNN fit in Model Lab also differentiates through trainable convolutional layers. Do not mistake a worked forward trace for a measured trained classifier.

Sources: [D2L: convolutions for images](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html), [D2L: padding and stride](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html).

## DL06. Residual networks and transfer learning

A residual block learns a correction to an existing representation: $y=x+F(x)$. The bypass carries x unchanged; F is a learned branch. If both have shape `[batch, features]`, they can be added element by element. If dimensions differ, a projection must make them compatible. A bypass is not a backwards pass or a generation loop.

For x=2 and a branch output of −0.3, the result is 1.7. During differentiation, the input affects the output through both paths: locally, $dy/dx=1+F'(x)$. The direct contribution helps gradients reach earlier layers, but does not guarantee perfect optimisation or permit incompatible shapes. Deep residual networks use many such blocks.

Transfer learning starts with parameters learned on another task or dataset. A frozen encoder supplies features for a new head; fine-tuning allows some or all encoder weights to change. Freezing saves gradient and optimizer-state work for those weights, but the forward computation and activations needed by later trainable layers still have costs.

Compare a pretrained frozen baseline, selective fine-tuning, and full fine-tuning only when the data and budget justify them. Similarity between source and target tasks influences usefulness. A pretrained model can carry source biases, and a good source score does not establish target-domain reliability.

Sources: [D2L: residual networks](https://d2l.ai/chapter_convolutional-modern/resnet.html), [D2L: fine-tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html).

## DL07. Recurrent networks, GRUs and LSTMs

A recurrent network processes a sequence by updating a hidden state. At each position it combines the new input with the previous state using the same learned weights. The state summarises earlier inputs for a later prediction. Sequence length changes how many times the update runs, not how many copies of its parameters are learned.

A basic update is $h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$. Repeated multiplication through time can shrink or amplify gradients. A GRU adds gates that control how much old state to retain and how a candidate is constructed. Under the convention used here, $h_t=z_th_{t-1}+(1-z_t)\tilde h_t$: an update gate near one mostly retains the previous state. Other presentations reverse the naming, so follow the equation.

An LSTM maintains a cell state as well as a hidden output. Forget and input gates scale old cell memory and a candidate update; an output gate controls how much cell content is exposed. Gates are learned continuous values, not hard-coded decisions about which word matters.

The worked trace compares recurrent cells on a fixed small sequence. It does not train them. A bidirectional variant also reads from the right, which is legal for complete-sequence classification but would leak future tokens in ordinary causal next-token prediction.

Sources: [D2L: RNNs](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html), [D2L: GRUs](https://d2l.ai/chapter_recurrent-modern/gru.html), [D2L: LSTMs](https://d2l.ai/chapter_recurrent-modern/lstm.html).

## DL08. Attention and Transformer building blocks

Attention lets one position retrieve a weighted mixture of information from other permitted positions. Instead of compressing the whole past into one recurrent state, it constructs a query for the current position, keys used for comparison, and values carrying information to mix. All three are learned projections, not separate natural-language questions and answers.

For a query q and key $k_j$, a scaled dot product gives score $q^Tk_j/\sqrt{d_k}$. A mask removes disallowed positions before softmax normalises the scores over keys. With two allowed scores 0 and $\ln3$, the weights are 0.25 and 0.75. Values 2 and 6 then produce $0.25(2)+0.75(6)=5$. The weights describe this mixture; they do not alone prove an explanation of the final prediction.

Self-attention takes queries, keys, and values from the same sequence. Cross-attention takes queries from one stream and keys/values from another. Causal masking permits only the prefix. Multiple heads learn different projections and combine their outputs; they do not simply vote on labels.

A Transformer block adds position information, normalization, residual paths, and a position-wise feed-forward network. The attention mixture communicates across positions; the feed-forward branch transforms each position's features. The decoder lesson follows the complete block and distinguishes forward inference, repeated generation, and weight updates.

Sources: [D2L: self-attention and positions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html), [Vaswani et al.: Transformer](https://arxiv.org/abs/1706.03762).

## DL09. Autoencoders and self-supervised representations

An autoencoder maps an input into a code and reconstructs it. The target is available from the input itself, so no separate class annotation is required. A bottleneck restricts what can be retained, making the choice of representation important.

If input `[1,0]` is reconstructed as `[0.5,0.5]`, summed squared error is 0.5 while mean squared error across its two features is 0.25. State the reduction before comparing numbers. A more accurate reconstruction does not automatically create a representation better for classification: the model may preserve irrelevant detail or copy the input too easily.

For centred data and squared reconstruction error, a linear autoencoder with an appropriate bottleneck can recover a principal subspace, connecting it to PCA. Nonlinear encoders can represent more complicated mappings. A denoising autoencoder instead reconstructs a clean input from a corrupted version, forcing the model to use surviving structure.

Other self-supervised objectives include predicting masked content and bringing two views of an example together in representation space. Each objective defines what information is rewarded. Test the learned representation on the intended downstream task rather than inferring usefulness from reconstruction loss alone. The local lab includes an image autoencoder, not a general contrastive-training backend.

Sources: [Deep Learning: autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html), [SimCLR paper](https://arxiv.org/abs/2002.05709).

## DL10. Generative models: VAEs, GANs and diffusion

A generative model produces new samples resembling a learned distribution. VAE, GAN, and diffusion methods use different training signals; their losses are not scores to place in one universal leaderboard.

### A VAE learns a distribution over latent codes

An encoder gives an approximate posterior $q(z\mid x)$; a decoder gives $p(x\mid z)$. The encoder predicts distribution parameters, then a latent code is sampled and decoded. Reparameterising a Gaussian as $z=\mu+\sigma\epsilon$, with standard-normal epsilon, permits gradients through mu and sigma. The usual objective minimises expected reconstruction negative log-likelihood plus $\operatorname{KL}(q(z\mid x)\|p(z))$.

This objective is negative ELBO. The correct bound is $\mathrm{ELBO}\leq\log p(x)$, or $-\mathrm{ELBO}\geq-\log p(x)$. Reconstruction MSE is appropriate only under a stated observation model and scale; it is not the definition of every VAE. Too much pressure toward the prior can make the latent code uninformative; too little can damage useful prior sampling.

### A GAN learns through an opponent

A generator maps noise to a sample. A discriminator learns to distinguish real from generated samples; the generator is updated through the discriminator's response. This adversarial game supplies a learning signal without an ordinary tractable sample likelihood. Instability and missing modes are possible. A discriminator's current success is not a final measure of sample quality.

### Diffusion separates corruption from generation

During training, known noise corrupts a real example at a selected noise level. A network predicts noise or a related clean-data target from that corrupted example and the noise level. Sampling starts from noise and repeatedly applies learned reverse updates. One training denoising prediction is not the full generation loop. The worked variants below illustrate these mechanisms; the local application does not train these generative families.

Sources: [Kingma and Welling: VAE](https://arxiv.org/abs/1312.6114), [Goodfellow et al.: GAN](https://arxiv.org/abs/1406.2661), [Ho et al.: diffusion](https://arxiv.org/abs/2006.11239), [Deep Learning: generative models](https://www.deeplearningbook.org/contents/generative_models.html).

## DL11. Graph neural networks

A graph represents entities as nodes and relations as edges. Node features describe each entity; adjacency describes which entities can exchange messages. A graph network updates a node using its neighbours' representations, usually with shared learned transformations and a permutation-invariant aggregation such as sum or mean.

Suppose node B has value 2 and neighbours with values 1 and 4. Their mean is 2.5. A simple update $h'_B=\operatorname{ReLU}(0.5(h_B+\operatorname{mean}_{j\in N(B)}h_j))$ gives 2.25. Using a sum instead gives 3.5. These are different choices: sum retains information about neighbour count, while mean normalises it away. This teaching update is not the exact normalisation rule of every GCN.

One message-passing layer reaches one-hop neighbours; two can use neighbours of neighbours. Repeated mixing may make representations too similar, called oversmoothing. Graph-level prediction additionally needs a readout pooling node representations; node classification predicts separately at each node.

The graph's edge meaning is a modelling decision, not proof that those relations are causal. Check whether evaluation nodes or labels were available during fitting and whether the intended setting is transductive or inductive. The site computes a small graph example but has no GNN-training button.

Sources: [Kipf and Welling: GCN](https://arxiv.org/abs/1609.02907), [PyTorch Geometric: message passing](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html).

## DL12. Prediction heads, targets and losses

A representation becomes a prediction only after an output head assigns the required meaning. First specify the target unit and which outcomes may coexist. Then choose a head and objective, rather than selecting a loss because its name is familiar.

For one of K mutually exclusive classes, produce K logits and use categorical cross-entropy. For independent binary labels that can coexist, produce one logit per label and use binary cross-entropy. A sequence with one tag at every position is still categorical at each position, with logits shaped conceptually `[batch, positions, tags]`; it is not automatically multilabel. A CRF additionally models a joint score over allowed tag sequences.

For a numerical target, a linear output can be trained with squared or absolute error. For language modelling, each scored position predicts one vocabulary token. Ignore masks determine which targets contribute. For reconstruction, the observation model determines an appropriate output distribution and loss.

Training loss and task metric answer related but different questions. Cross-entropy can improve without changing any thresholded labels. Segmentation overlap metrics count regions: if intersection is 3 and union is 9, IoU is 1/3. Dice instead divides twice the intersection by the total sizes of both regions. Define the counted sets before comparing them.

Sources: [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html), [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html).

## DL13. The neural training loop and its system state

A training loop changes a model in a controlled sequence. Load a batch, clear or deliberately retain gradient buffers, run the forward computation, calculate the intended scalar loss, backpropagate, and apply one optimizer update. An epoch counts a pass through training data; a step here means an optimizer update. Gradient accumulation can combine several microbatches into one such step.

The system contains more than weights. Adam has moment estimates; the data sampler has a position and random state; a scheduler has progress. A checkpoint for exact continuation needs this state and the compatible model, tokenizer, preprocessing, and data. Saving only weights can support inference without supporting faithful resumption.

Validation should use evaluation behaviour and avoid parameter updates. The local neural and scratch trainers select their best checkpoint using development loss before reporting test performance. Best and last checkpoints can differ: last means most recent, not necessarily best on validation.

Memory holds parameters, gradients, optimizer state, activations, and temporary buffers. Hardware availability does not imply every operation or precision is supported efficiently. The lab probes a small neural forward/backward path when choosing a device; classical scikit-learn fits remain CPU-based. A live loss chart reports saved measurements, not a promise that the next point will fall.

Sources: [PyTorch: saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [D2L: parameter management](https://d2l.ai/chapter_builders-guide/parameters.html).

## DL14. Compare neural and classical models

Choosing a neural model is an empirical decision, not a reward for using a more advanced architecture. A small numerical table may favour a tree ensemble; a spatial image task may benefit from convolution. Compare methods on the same target and held-out cases, while documenting what each fit is allowed to learn.

Start with a baseline, then compare plausible candidates with a declared tuning budget. A linear classifier on scaled features and a neural network on raw features differ in both learner and representation. That may be a useful system comparison, but it does not isolate architecture alone. Decide which question you mean.

For a one-point accuracy gain, inspect changed predictions, sample size, seed variation, cost, and important subgroups. A conditional bootstrap interval over test predictions addresses sample sensitivity for that fitted pair; it does not include every training or search choice. Fit time and inference latency are also measurements with hardware and workload attached.

Save unsuccessful fits and explain what they ruled out. A larger network that increases training accuracy while reducing validation accuracy is evidence about that configuration, not evidence that deep learning fails. Likewise, a neural win on one dataset is not proof of a general ranking. The comparison pages should help answer which fitted system is useful for this task and budget.

Sources: [scikit-learn: statistical comparisons of models](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_grid_search_stats.html), [D2L: generalisation in deep learning](https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html).
