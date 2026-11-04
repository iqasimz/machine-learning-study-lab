# ML and DL interview practice

These are predicted practice questions, not questions attributed to an employer. All numerical examples are illustrative, not results from this lab. Start with the short answer, then explain the example and limitation in your own words. The source links support the methods; they do not establish that a particular answer guarantees interview success.

## IML01 — Several records belong to each customer. How would you split the data?

**Answer.** A split separates examples used to learn from examples used to judge the model. Leakage means the learning procedure receives information that would not be available when making the intended real prediction. I first define what “new” means: a new row, customer, hospital, or future date. That determines the split, before I fit preprocessing or search for model settings.

For example, 1,000 images from 100 patients are not necessarily 1,000 independent cases. If the deployment target is new patients, I keep every image from one patient in a single partition. Otherwise the model may recognize a familiar patient rather than learn the disease signal. I fit the imputer and scaler inside each training fold, then apply those fitted transformations to validation rows. A final test partition stays outside model selection.

I also inspect features: a discharge code recorded after a diagnosis cannot support a prediction required at admission. A perfectly separated split cannot repair that unavailable feature. Group splitting can still leave hospital-specific shortcuts, so the evaluation must match the actual deployment setting.

**Follow-up: Is unsupervised preprocessing allowed before splitting?** Not if it learns from held-out examples. Fit PCA, imputation, and scaling within the training partition too.

**Source links:** [scikit-learn leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html), [group-aware validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

## IML02 — Your model fits training data but fails on new cases. Could regularization help?

**Answer.** Regularization adds a preference for simpler parameter values to the training objective. In a mean-loss convention, an L2 objective is the average prediction loss plus `lambda` times the sum of squared weights. Increasing `lambda` makes large weights more expensive; it does not directly force the test score to improve. L1 uses absolute weights and can make some coefficients exactly zero.

Consider two candidate solutions with equal training error. Their squared-weight totals are 4 and 40. With `lambda = 0.1`, their penalties are 0.4 and 4, so the objective prefers the first. If the useful relationship genuinely needs a large coefficient, excessive regularization can also erase signal and cause underfitting.

Some implementations expose inverse strength `C`: a smaller `C` means stronger regularization. I check the exact objective rather than copying `C = 1/lambda` between libraries. Averaging versus summing losses, sample weights, intercept treatment, and factors such as one half can change the conversion. Scaling matters too: changing a feature's units changes the coefficient needed to represent the same relationship.

**Follow-up: Does doubling the dataset require doubling lambda?** Not automatically. First inspect whether the implementation averages or sums its loss and how its penalty is normalized.

**Source links:** [logistic regression objective](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), [Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## IML03 — A fraud classifier has high accuracy. Is it ready to use?

**Answer.** I start with the decision and the cost of each error. Precision asks how many alerts are correct; recall asks how many real events are found. A classification threshold turns a score into an action. The correct threshold depends on review capacity, missed-event cost, and false-alarm cost, not a universal value of 0.5.

Suppose 20 of 1,000 transactions are fraud. Predicting “not fraud” everywhere gives 98% accuracy but zero fraud recall. A detector that finds 15 frauds and also flags 45 legitimate transactions has precision `15/60 = 25%` and recall `15/20 = 75%`. Whether that is useful depends on what investigators can handle. I would report these counts, a precision–recall curve, and performance at a threshold selected on validation data. Average precision summarizes ranking across thresholds; it is not precision at the deployed threshold.

If probabilities drive financial decisions, I separately examine calibration: among cases scored near 0.2, roughly one fifth should be positive in a comparable population. Changing class prevalence can change precision even when conditional error rates stay similar.

**Follow-up: Should I oversample before splitting?** No. Keep validation and test representative; any training resampling belongs inside the training fold.

**Source links:** [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics), [threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).

## IML04 — Would you start with a neural network for a small tabular dataset?

**Answer.** I compare the structure of the data, the amount of labeled data, operational constraints, and simple baselines. A decision tree makes conditional rules by splitting features. A neural network learns combinations of features through parameterized layers. Neither family is universally better or automatically more interpretable in its larger forms.

For 8,000 customers with 25 numeric and categorical fields, I would try a regularized linear model and a tree ensemble before building a deep network. Trees can express rules such as “recent spending matters differently for new customers” without manually writing that interaction. For images, a convolutional network has a useful assumption: local visual patterns can occur in different positions. That structure is harder to exploit by treating every pixel as an unrelated table column.

The comparison uses the same splits, available features, and a stated tuning budget. I also measure prediction time, model size, missing-value behavior, and calibration. A small accuracy gain may not justify ten times the serving cost. Conversely, a pretrained neural representation may win with few labels because it brings useful information learned elsewhere; that is not an equal-from-scratch comparison.

**Follow-up: Are tree ensembles explainable?** Individual splits are readable, but hundreds of interacting trees need careful inspection; feature importance alone is not a causal explanation.

**Source links:** [decision trees](https://scikit-learn.org/stable/modules/tree.html), [neural network models](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## IML05 — Why choose gradient boosting rather than a random forest?

**Answer.** Both combine trees, but they construct the ensemble differently. A random forest fits varied trees using randomized training samples and feature choices, then combines their predictions. This reduces the instability of a single tree. Gradient boosting builds the prediction sequentially: the next tree follows the direction that reduces the current objective's errors.

For squared-error regression, imagine a current prediction of 7 for a target of 10. The residual is 3. A new tree may suggest a correction of 2; with learning rate 0.1, the ensemble adds only 0.2, giving 7.2. For other losses, the training signal is a negative gradient, not necessarily the ordinary residual. The learning rate controls each added tree's contribution; the number and complexity of trees control how much correction is available.

I tune those choices together and monitor held-out performance. Very deep trees, many rounds, or poorly selected stopping points can overfit. Forest trees are comparatively independent to train; standard boosting rounds depend on preceding rounds. This is an algorithmic distinction, not a guarantee that one implementation will always train faster.

**Follow-up: Does adding trees always hurt a forest?** More trees generally stabilize the ensemble average; depth and leaf-size settings still determine the individual trees' behavior and resource cost.

**Source link:** [scikit-learn ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html).

## IML06 — Why does converting income units change a nearest-neighbour model?

**Answer.** Scaling changes feature units so that a large numerical range does not automatically dominate a model's geometry or penalty. Standardization usually subtracts a training mean and divides by a training standard deviation. It does not make the data normally distributed or remove every outlier.

Suppose age varies by 10 years while annual income varies by 10,000 euros. In an unscaled Euclidean distance, an income difference can overwhelm age simply because of the units. A nearest-neighbor model may then choose neighbors almost entirely by income. Gradient-based linear models can also become harder to optimize when feature scales differ greatly. For axis-aligned decision trees, ordinary positive rescaling preserves the order of values and usually preserves the possible partitions, so scaling is less important.

My pipeline fits scaling on training rows, transforms validation rows with those saved statistics, and ships the same transformation with the model. I inspect outliers before selecting a standard or robust scaler. Sparse count matrices need special care: subtracting a mean can turn zeros into nonzero values and destroy sparse storage.

**Follow-up: What if a feature is constant?** It carries no within-training variation. Handle the zero-variance case explicitly rather than dividing by zero or assuming it is informative.

**Source link:** [scikit-learn preprocessing and standardization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling).

## IML07 — Model B scores slightly higher than A. Is the improvement convincing?

**Answer.** I distinguish uncertainty from the sampled test cases from variation caused by training randomness. A confidence interval describes a procedure's uncertainty about a defined population quantity; it is not a promise about every future example. A paired comparison evaluates both models on the same cases, preserving which examples were easy or difficult for both.

Suppose models A and B score 0.81 and 0.82 on 500 reports. I would save their per-report predictions and repeatedly resample reports with replacement, recomputing the difference each time. If reports come from 50 patients and the target is new patients, I resample patients with all their reports rather than pretend the 500 reports are independent. The metric and weighting must also match the question: average report performance and average patient performance are different targets.

I then repeat training seeds when feasible and report that variation separately. A test-case bootstrap of one fitted model does not measure all training uncertainty. A small difference with a wide interval warrants caution, not a declaration of equivalence. Repeatedly testing many settings also creates selection bias.

**Follow-up: Can I bootstrap five cross-validation scores as independent cases?** Usually not safely. Their training data overlap; define the dependence structure and comparison procedure first.

**Source links:** [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [dependence and validation](https://scikit-learn.org/stable/modules/cross_validation.html).

## IML08 — PCA keeps most of the variance. Could it still hurt classification?

**Answer.** Ordinary PCA first centers its input features; scaling their variances is a separate choice. Principal component analysis, or PCA, finds orthogonal directions that explain variation in the input features. Orthogonal means perpendicular in the chosen coordinate system. It projects rows into fewer coordinates. PCA does not use the target label, so preserving a large fraction of input variance does not guarantee preserving the information needed for prediction.

Imagine a machine sensor table with a large, harmless temperature fluctuation and a tiny pressure change that signals failure. A one-component projection may preserve temperature variation while discarding the important pressure signal. I therefore compare the downstream predictor with and without PCA, fitting PCA only on training folds. I also decide whether to standardize first: otherwise large-unit features can dominate the variance.

For a simple calculation, reducing a 100-feature row to 10 components changes the next dense layer with 20 outputs from `100 × 20 + 20 = 2,020` parameters to `10 × 20 + 20 = 220`. That can reduce one layer's work, but computing the PCA projection itself adds work. Components can mix many original features and become harder to interpret.

**Follow-up: Is PCA the same as feature selection?** No. Selection retains chosen original columns; PCA constructs new combinations of columns.

**Source link:** [PCA and dimensionality reduction](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca).

## IML09 — K-means found customer groups. How would you decide whether they help?

**Answer.** Clustering groups examples without requiring target labels. Its result depends on the representation, distance, algorithm, and scale of the features. A cluster is a pattern under those choices, not proof of a real category or a useful business segment.

For example, customer coordinates measured as purchases per month and income in euros may produce income-dominated clusters unless I choose an appropriate representation. K-means repeatedly assigns points to their nearest center and updates each center to the assigned points' average. It favors compact groups under squared Euclidean distance. Density-based methods can identify differently shaped groups and noise, but depend on their own density and neighborhood settings.

I inspect internal measures, stability under resampling, representative examples, and an external purpose. A silhouette score compares within-cluster distances with distances to another cluster; a good score does not tell me that the grouping is fair or actionable. If four customer clusters change completely when a few rows are removed, I would not attach strong permanent labels to them. Any downstream campaign needs its own evaluation.

**Follow-up: How do I select the number of clusters?** Combine objective diagnostics with stability and the intended use; an elbow plot is evidence to inspect, not an automatic answer.

**Source link:** [clustering assumptions and evaluation](https://scikit-learn.org/stable/modules/clustering.html).

## IML10 — How would you evaluate a model that forecasts next week's sales?

**Answer.** Forecasting predicts values after a decision time. A forecast horizon is how far ahead the prediction reaches; a lag is a previously observed value used as an input. I define both before constructing features, then evaluate in chronological order so each prediction uses only information available at that time.

To forecast Monday sales, last Monday's sales may be available, but the final weekly total containing that Monday is not. I build rolling features from past observations only. A rolling-origin evaluation repeatedly trains on an earlier interval and tests on a later interval. For a seven-day horizon, I verify that feature construction and label windows do not accidentally overlap unavailable future information. Any gap depends on the actual label and data-availability process, not a fixed rule.

I compare against a naive baseline, such as the previous week's corresponding day. A complex model with mean absolute error 12 is not impressive if that baseline achieves 10. I also inspect holidays, changing behavior, and performance by horizon. Ordinary tree predictions can struggle to extrapolate a trend beyond the target values seen in training.

**Follow-up: Is a random split always forbidden for temporal data?** The split must reflect the prediction question. For actual future forecasting, random row splits generally answer the wrong question.

**Source link:** [time-ordered validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## IML11 — Escalated support tickets take longer. Does that mean escalation causes delays?

**Answer.** Not necessarily. Prediction estimates an outcome from observed information; causal inference asks what would change after an intervention. A confounder is a variable that influences both an apparent cause and the outcome. A feature can be highly predictive because it records consequences, shared causes, or selection rules rather than an effect we can change.

Suppose hospitalized patients receiving a treatment have worse outcomes. The treatment may have been given to the sickest patients. A predictor can use treatment as a risk signal without demonstrating that withholding it would improve health. Similarly, a support ticket's escalation flag may predict delays because difficult tickets are escalated, not because escalation causes the delay.

I would not convert feature importance into policy advice. I first specify the intervention, outcome, eligible population, and timing. Randomized experiments can help identify effects when ethical and feasible; observational approaches need explicit assumptions, suitable data, and sensitivity analysis. Controlling for every available variable is not automatically correct: some variables are consequences of treatment or selection mechanisms. A model's predictive score cannot validate those causal assumptions.

**Follow-up: Does permutation importance solve causality?** No. It measures how a fitted predictor depends on a feature under a particular perturbation; correlated features can complicate even that interpretation.

**Source links:** [Hernán and Robins, Causal Inference](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/), [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## IML12 — The offline model was strong, but production quality dropped. What would you investigate?

**Answer.** I separate a broken implementation from a changed prediction problem. Training–serving skew means that the production input or transformation differs from the training setup. Distribution shift means the relevant data distribution has changed. Both can produce a drop without any error in the model's saved weights.

For a concrete example, a training pipeline represents missing income as an explicit missing value, but production replaces it with zero. A tree can now follow entirely different branches. I replay saved, non-sensitive example inputs through both pipelines and compare schema, units, categories, intermediate features, and final scores. I then check delayed or incorrect labels, population composition, prediction-time availability, and subgroup results. A dashboard showing only overall accuracy may hide a collapse on new customers.

I also compare the deployed version with the intended artifact and threshold. If the business now reviews only 50 alerts instead of 200, the operational metric has changed even if scores are identical. Retraining is not my first reflex: it can hide a pipeline defect and create another unverifiable version. Rollback and bounded monitoring can be safer immediate responses.

**Follow-up: Should every drift alarm trigger retraining?** No. Investigate whether the changed inputs affect outcome quality and whether reliable new labels are available.

**Source link:** [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).

## IDL01 — Backpropagation runs, but the weights do not change. What could be wrong?

**Answer.** Autograd records supported operations involving tensors that require gradients, then applies the chain rule to calculate derivatives. A derivative tells us how a small parameter change would affect the loss. Backpropagation computes those derivatives; the optimizer uses them to change parameters. These are separate operations with separate ownership in the loop.

For a scalar example, let prediction be `w × x`, with `w = 2`, input `x = 3`, target 5, and loss `(prediction − target)²`. The prediction is 6, the loss is 1, and the weight gradient is `2 × (6 − 5) × 3 = 6`. A basic SGD step with learning rate 0.1 changes `w` to `2 − 0.1 × 6 = 1.4`. That large step need not be optimal; the example illustrates the mechanism.

In a normal update I clear old gradients, run the forward pass, compute loss, call backward, then step the optimizer. Gradients accumulate unless cleared. Accidentally detaching a tensor can break the intended gradient path; calculating a gradient does not itself apply an update. Frozen parameters and optimizer membership also need to agree.

**Follow-up: Does `model.eval()` disable autograd?** No. Evaluation mode changes modules such as dropout and BatchNorm; use a separate gradient-disabled context when appropriate.

**Source link:** [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html).

## IDL02 — What outputs, labels and loss would you use for a ten-class classifier?

**Answer.** I write the meaning of each axis before debugging values. A batch of 32 examples and 10 mutually exclusive classes usually produces a logits tensor shaped `[32, 10]`. Logits are unnormalized class scores. For the standard integer-target form of PyTorch cross-entropy, targets have shape `[32]` and contain class indices from 0 to 9.

The loss converts each row's scores into a penalty for the correct class and reduces the per-example losses, commonly by their mean. If the correct class receives probability 0.25, its negative log-likelihood is `−log(0.25) ≈ 1.386`. Moving that probability to 0.5 lowers the loss to about 0.693, even if the top predicted class has not changed yet. This explains why loss can improve before accuracy changes.

I do not apply softmax before that loss's logits interface. For independent multi-label decisions, I normally use one logit per label with a suitable sigmoid-based binary loss instead. I check target type, label mapping, reduction, masks, and class weights. Broadcasting can silently turn an intended one-to-one regression comparison into comparisons across an entire batch.

**Follow-up: Can cross-entropy accept probability targets?** Some interfaces can, but that is a different target representation with its own validity requirements; do not infer it from a coincidental shape match.

**Source link:** [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## IDL03 — Your CNN uses much more memory after adding channels. Why?

**Answer.** A convolution applies learned local filters across spatial locations. Input channels are the features entering each location; output channels are the number of learned filter responses. Weight sharing means one filter is reused across the image, so its parameter count does not grow with image width or height.

For a standard convolution with 3 input channels, 16 output channels, a `3 × 3` kernel, and one bias per output channel, the count is `16 × 3 × 3 × 3 + 16 = 448`. With input width 32, padding 1, stride 1, and dilation 1, output width is `floor((32 + 2 − 2 − 1)/1 + 1) = 32`. Stride 2 instead gives 16. The same calculation applies independently to height.

Next, a nonlinearity lets stacked layers represent more than a single linear transformation; later layers combine earlier local patterns. The receptive field is the input region that can affect an output location, and it grows across layers. More channels increase representational capacity but also computation and activation memory. Grouped convolutions change the parameter formula because each output sees only a subset of input channels.

**Follow-up: Does padding add trainable parameters?** No. Padding adds boundary values; it changes spatial behavior and output size, not filter count.

**Source link:** [PyTorch Conv2d definitions and output formula](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

## IDL04 — Training becomes unstable after a learning-rate change. What would you inspect?

**Answer.** The learning rate controls the scale of an update, not the number of examples learned. Momentum smooths directions using past gradients. Adam maintains moving estimates of gradients and squared gradients, using them to adjust the scale of each parameter's update. AdamW separates weight decay from that adaptive gradient transformation.

With basic SGD, a parameter of 1 and gradient 0.2 becomes 0.98 at learning rate 0.1. That simple arithmetic does not describe Adam's full step because its stored estimates and bias correction also matter. If loss suddenly becomes unstable after a learning-rate increase, I examine gradient norms, finite values, and the update schedule before assuming the architecture is wrong.

I select learning rate together with batch size, warmup, total updates, and optimizer. Warmup starts with smaller steps; a scheduler changes the learning rate over training. A schedule advancing once per microbatch instead of once per actual update can finish too early under gradient accumulation. Resuming only weights without optimizer state changes the trajectory of adaptive methods. Neither AdamW nor SGD is universally superior across tasks and budgets.

**Follow-up: Is L2 regularization identical to AdamW decay?** No. Adding an L2 term to a gradient that is adaptively rescaled is not generally the same as a separate decay step.

**Source links:** [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

## IDL05 — Why can the same input produce different predictions in training and evaluation?

**Answer.** Some layers deliberately behave differently while learning. Dropout randomly removes selected activations during training as regularization, then disables that randomness during standard evaluation. Batch normalization, or BatchNorm, normalizes channel activations using batch statistics during training and usually uses saved running statistics during evaluation. Layer normalization instead normalizes within each example over its specified feature dimensions.

Imagine a BatchNorm layer whose saved mean is 10 and standard deviation is 2. Ignoring its numerical stabilizer, an activation of 12 normalizes to 1 before the learned scale and shift. A tiny current batch with a different mean could produce another value in training mode. That means validation in the wrong mode can change predictions and contaminate running statistics with held-out data.

My validation routine sets evaluation mode, runs without gradient tracking when no derivatives are needed, then restores training mode before continuing optimization. These switches solve different problems: mode controls layer behavior; gradient mode controls graph recording. Very small or non-representative training batches can make BatchNorm statistics unreliable. LayerNorm avoids dependence on other examples but is not a universally better replacement.

**Follow-up: Does freezing BatchNorm's weights freeze its statistics?** Not necessarily. Disabling parameter gradients and stopping running-statistic updates are separate decisions.

**Source links:** [BatchNorm behavior](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), [LayerNorm dimensions](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

## IDL06 — A large batch will not fit in memory. Can accumulation replace it?

**Answer.** Gradient accumulation combines gradients from several microbatches before one optimizer step. A microbatch is the group processed in one forward/backward pass; the effective batch is the total contributing to the update. This can reduce peak activation memory, but it does not make every aspect of training identical to a single larger forward pass.

For four equally sized microbatches of eight examples, dividing each mean loss by four and accumulating gradients gives the mean-loss gradient over 32 examples, assuming equivalent per-example computation. I clear gradients once before the group, backpropagate four times, then clip and update once. With unequal numbers of valid targets, averaging microbatch means equally is wrong: losses 2 and 4 over 10 and 90 targets have overall mean `(10 × 2 + 90 × 4)/100 = 3.8`, not 3.

BatchNorm depends on each physical microbatch, and dropout randomness or floating-point summation can differ. Accumulation also does not remove optimizer-state or weight memory. Under mixed precision, I follow the scaler's required order and unscale before clipping. The last incomplete accumulation group needs the intended normalization too.

**Follow-up: When should a learning-rate scheduler advance?** Usually at its defined optimizer-update boundary, not automatically after every microbatch.

**Source link:** [PyTorch mixed-precision accumulation guidance](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation).

## IDL07 — Training loss is flat. How would you find out what is wrong?

**Answer.** I start with a small controlled case before changing the architecture. “Not learning” can mean flat training loss, poor validation performance, invalid numbers, or a metric bug; those are different failures. I verify labels, input ranges, output shapes, and a simple baseline first, then inspect whether gradients and parameter updates actually exist.

For example, I take 16 clearly labeled examples, disable unnecessary augmentation and regularization, and try to fit them. If the model cannot substantially reduce their training loss, I inspect shuffled label mappings, a detached output, frozen parameters, a missing optimizer step, or an unsuitable learning rate. I compare selected parameter values before and after one update. A nonzero gradient does not prove the intended parameter changed: it may be absent from the optimizer.

If the small case fits but validation remains poor, I move to overfitting, split quality, distribution mismatch, and data coverage. If loss becomes non-finite, I inspect the first failing batch, unstable operations, and numerical precision. A model that memorizes 16 examples has passed a useful mechanical check; it has not demonstrated generalization or ruled out leakage elsewhere.

**Follow-up: Should I immediately add layers?** No. More capacity will not repair incorrect targets, broken gradients, or an evaluation pipeline that measures the wrong thing.

**Source link:** [PyTorch optimization workflow](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

## IDL08 — With limited labelled images, would you train from scratch or adapt a model?

**Answer.** Training from scratch starts with newly initialized weights. Transfer learning reuses representations learned on another task or dataset. A frozen-feature approach trains only a new prediction head; fine-tuning also changes some or all pretrained parameters. I choose based on available labels, domain similarity, compute, licensing, and the acceptable risk of inherited behavior.

Suppose I have 2,000 labeled images and a suitable pretrained image encoder. I first train a small head on frozen features, then compare partial or full fine-tuning at a conservative learning rate. With a 512-dimensional representation and 5 classes, a linear head has `512 × 5 + 5 = 2,565` parameters. That is a tractable baseline, but the frozen encoder still has storage and inference costs.

I keep a clean validation set and record exactly which weights were pretrained, frozen, or updated. Large upstream data can contain evaluation examples, so pretraining overlap is a limitation to investigate. A model from a very different domain may transfer poorly. Comparing fine-tuning with a scratch model also compares access to prior data, not merely optimizer quality.

**Follow-up: Does frozen mean computationally free?** No. Frozen weights avoid their training updates, but the encoder still performs a forward pass and occupies memory.

**Source link:** [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## IDL09 — The sequence model predicts padding. Which masks and targets would you check?

**Answer.** A mask specifies which positions may participate in a particular computation. A padding mask identifies artificial filler positions used to batch unequal-length sequences. A causal attention mask prevents a position from reading future tokens. A loss mask excludes chosen positions from the training objective. These masks solve different problems and cannot be substituted blindly.

Suppose sequences have lengths 3 and 5 and are padded to length 5: eight real tokens and two padding positions. Attention must not treat padding as meaningful evidence, and a token-level loss should not reward predicting padding unless that is intentional. In next-token prediction, input positions are paired with the following target token, so shifting and masking must agree.

With input `[A,B,C]` and targets `[B,C,D]`, the output at input position B predicts C and may attend to A and B, not input C. Removing a target from the loss alone does not stop attention from reading future inputs. Conversely, an attention mask does not automatically remove a target from loss averaging. Incorrect normalization can make heavily padded batches appear artificially easy. Mask conventions vary across APIs, so I test a tiny example rather than guess whether `True` means allowed or blocked.

**Follow-up: Are all sequence tasks causal?** No. Bidirectional classification can use the full observed input when the prediction setting permits it.

**Source links:** [MultiheadAttention masks](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), [Transformer masks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html).

## IDL10 — Two documents have similar embeddings. Does one necessarily support the other?

**Answer.** An embedding maps an item into a numeric vector. Similarity between vectors is meaningful only relative to the representation's training objective and intended task. Contrastive learning trains representations so chosen positive pairs are relatively close and selected negative pairs are relatively far apart. The choice of pairs teaches the model what “related” means.

For document search, I might treat a question and its relevant passage as a positive pair. A paraphrase objective can teach a different notion of similarity from one focused on relevance. A passage that contradicts a question can share many topic words, so high semantic similarity alone does not establish supporting evidence. I evaluate retrieval with held-out queries and relevance judgments before evaluating the complete answer-producing system.

If 80 of 100 queries retrieve a relevant passage among their first five results, query-level recall-at-five is 0.8 in a single-relevant-passage setup. That does not mean answer accuracy is 0.8: the reader may ignore or misuse the evidence. I also verify pooling, normalization, indexing, and exact versus approximate search. In-batch negatives can accidentally include other valid positives.

**Follow-up: Is cosine similarity a probability?** No. It measures vector alignment; interpreting it as a probability requires an additional justified model or calibration procedure.

**Source link:** [SimCLR contrastive learning paper](https://arxiv.org/abs/2002.05709).

## IDL11 — A smaller model is slower in production. How could that happen?

**Answer.** Latency is how long one request waits; throughput is completed work per unit time. Peak memory includes more than saved parameter files: activations, temporary buffers, framework allocations, and sequence caches where the model uses them also matter. Optimizer state normally belongs to training, not an inference-only service. I measure the actual deployment path rather than infer performance from parameter count alone.

For example, 10 million parameters stored in float32 occupy about 40 million bytes before additional runtime state. That is roughly 38.1 MiB, not 40 MiB, because MiB uses powers of two. A model can fit its weights in memory but fail on a large image batch because intermediate activations are much larger. Quantization can shrink weights without reducing all other allocations equally.

I report hardware, precision, input sizes, batch size, warmup, and whether tokenization, transfers, and queuing are included. On accelerators with asynchronous execution, I synchronize appropriately when timing; otherwise I may measure only work submission. I inspect tail latency, such as the 95th percentile, not just an average. Larger batches can improve throughput while making an individual request wait longer.

**Follow-up: Does a two-times smaller model guarantee two-times lower latency?** No. Kernel efficiency, memory bandwidth, overhead, batching, and hardware support can dominate the result.

**Source links:** [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html), [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).

## IDL12 — A better-scoring model used twice the training budget. What can you conclude?

**Answer.** I state the claim before deciding which factors to hold fixed. A fair comparison isolates the change relevant to that claim; reproducibility records enough information to repeat the procedure and inspect the outcome. Matching everything is impossible when methods differ, so unmatched budgets and data access must be explicit.

Suppose model A trains for 10,000 updates and model B for 20,000, both at batch size 32. Their example exposure is about 320,000 versus 640,000 before incomplete batches or masking. A better B score does not isolate architecture quality. Equal updates also do not ensure equal tokens, wall-clock time, energy, or tuning effort. I choose the relevant budget and report the others where practical.

I save dataset versions, split identifiers, preprocessing, model and tokenizer revisions, effective batch sizes, seeds, optimizer state, selection rules, and per-example predictions. I compare seeds when affordable and select checkpoints using validation rather than test outcomes. Ablations change one meaningful component while keeping the comparison understandable. Deterministic settings can reduce variability, but do not guarantee bitwise equality across hardware or software versions.

**Follow-up: If one method gets more hyperparameter trials, is its win invalid?** Not automatically, but the result includes that search advantage. Report it and use a controlled search budget for claims about the method itself.

**Source link:** [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
