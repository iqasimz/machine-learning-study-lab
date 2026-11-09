# Consolidated interview bank

Generated from the canonical authored answers. Do not edit this snapshot directly.

## B01 — Bias and variance: underfitting and overfitting

### What underfitting and overfitting mean

**Underfitting** means the fitted model has not learned enough of the useful pattern. It performs poorly even on training examples. A straight line cannot fully describe a genuinely curved relationship, no matter how carefully you fit that line.

**Overfitting** means the model learns details of its training sample that do not carry over to new examples. A deep tree might learn that a particular customer ID predicts fraud perfectly, but that rule is useless for unseen customers.

**Bias** is systematic error in the average fitted prediction across hypothetical training samples. **Variance** is how much fitted predictions change when the training sample changes. An overly restrictive model often has high bias; an overly sensitive fit often has high variance. These are explanations of behaviour, not two scores read directly from one run.

### How to recognise the problem

| Observed pattern | Possible explanation | What to try |
|---|---|---|
| Training and validation are both poor | Underfitting, poor inputs, or training not working | Check the pipeline; then improve features, capacity or optimisation |
| Training is strong, validation much worse | Overfitting or a mismatch between the splits | Check leakage and label consistency; try regularisation or more representative data |
| Training improves, validation peaks and then worsens | Later training may be fitting noise | Select an earlier checkpoint using validation data |

These patterns are clues, not proofs. A shifted validation set can create a gap even without classic overfitting. Learning curves plot training and validation scores as training time, data size or model complexity changes.

### One useful qualification

Zero training error does not automatically mean failure on new data. Some large models generalise after fitting the training set exactly; test error can even fall again beyond the interpolation threshold, a pattern called **double descent**. Choose from measured held-out behaviour, not parameter count alone.



### Further reading

[scikit-learn: learning and validation curves](https://scikit-learn.org/stable/modules/learning_curve.html)

### How an interviewer might ask about this

- **Both training and validation scores are poor. Why?** Check underfitting or an optimisation/data problem.
- **Training is excellent but validation is poor. What would you change?** Check overfitting, leakage and split differences before choosing a remedy.

## B03 — Types of learning, parameters and hyperparameters

### What the model learns from

In **supervised learning**, examples include the desired target: text plus a spam label, or a house plus its price. **Unsupervised learning** looks for structure without those targets, such as customer clusters. **Self-supervised learning** creates targets from the data itself: hide a word and predict it, or predict the next token. **Semi-supervised learning** uses both labelled and unlabelled examples.

Classification predicts a category; regression predicts a number; ranking orders candidates. These describe the task, not the model architecture.

### Parameters versus hyperparameters

A **parameter** is fitted from data, such as a regression weight. A **hyperparameter** is a choice controlling the model or training, such as tree depth, learning rate or regularisation strength. Tune these using validation data, not the final test labels.

For logistic regression, the optimiser learns the weights. You choose how strongly to penalise them. A decision threshold is another choice; it is not normally learned by the same weight-fitting objective.

### Other labels you may hear

**Parametric** models have a fixed-size parameterisation once the architecture is chosen. **Non-parametric** methods can grow their representation with the data; nearest neighbours stores examples. A **generative** model represents how observations could be produced; a **discriminative** model focuses on predicting the target from an input. A generative LLM can still be used to classify text. These distinctions are separate axes, not competing lists of model types.

**Inductive** evaluation concerns new examples beyond the fitted sample. **Transductive** methods can use the specific unlabelled target set while fitting. State that information access when comparing results.


### How an interviewer might ask about this

- **Is predicting the next token supervised or unsupervised?** Explain how the targets are constructed.
- **What does the optimiser learn, and what do you choose?** Distinguish weights from settings such as learning rate.

## B05 — Preprocessing: missing values, scaling and categories

### Why preprocessing matters

Preprocessing changes raw records into usable model inputs. Common operations fill missing features, scale numeric columns and encode categories. They can change what a model learns even when the algorithm stays the same.

**Scaling:** a distance-based model may treat income values near 50,000 as more important than ages near 30 simply because of their units. Standardisation uses training-column means and standard deviations to put features on comparable scales. Trees usually do not need this for ordinary threshold splits.

**Missing values:** an empty income field is not an income of zero. A median fill is a baseline, not a guarantee of an unbiased estimate. Missingness may itself carry information, so a missing-value indicator can help.

**Categories:** one-hot encoding represents an unordered category using separate indicators. Encoding red, blue and green as 1, 2 and 3 can accidentally suggest an order or numeric distance.

Standardisation centres and rescales a feature; min–max scaling maps a training range to a chosen range; robust scaling uses quantities such as the median and interquartile range. Choose a convention that suits outliers and the model, then keep it fixed at evaluation.

### Fit on training data, apply elsewhere

Learn imputation values, scaling statistics and vocabularies from the training split. Apply those fitted transformations to validation and test data. In cross-validation, fit them again inside each training fold; a pipeline helps enforce this.

### Why values are missing

**MCAR:** missingness is unrelated to values. **MAR:** observed information can explain it. **MNAR:** it still depends on the missing value itself, for example very high earners withholding income. The observed table alone generally cannot tell you which story is true. Investigate collection rather than assuming missing rows are a random sample.


### How an interviewer might ask about this

- **Why did changing income from euros to cents change k-nearest neighbours?** Distance depends on feature scale.
- **Can I fill missing values before splitting the table?** Learned preprocessing must be fitted on training data only.

## B06 — Train, validation and test splits; data leakage

### What each split is for

**Training data** fits the model. **Validation data** chooses settings and checkpoints. **Test data** estimates performance after those choices are fixed. Cross-validation repeats training/validation splits within the development data; it is not permission to tune repeatedly on the final test.

**Leakage** is information entering fitting or selection that would not be available in the intended prediction setting. It can make a model look much better than it will be in use.

### How leakage happens

- The same patient, passage or near-duplicate occurs on both sides of a split.
- A feature contains an outcome recorded after the prediction time.
- Scaling, feature selection or threshold tuning uses final test data.

If the goal is prediction for new patients, keep each patient in one split. If the goal is forecasting, train on earlier periods and validate on later ones. Stratification preserves class proportions; it does not solve patient overlap or time leakage.

### Sampling and duplicates

A dataset can be imbalanced without being incorrectly sampled, and balanced without being representative. Check how it was collected. Content hashes find exact duplicates; near-duplicate detection needs additional rules. Similar wording alone is not proof that two records are the same observation. Group genuinely linked records before splitting.


### How an interviewer might ask about this

- **Several rows belong to the same patient. Is a random split safe?** Keep the patient together to match the intended new-patient evaluation.
- **How would you validate next month's demand forecast?** Respect time order and information available at prediction time.

## B15 — Linear and logistic regression

### Predicting a number or a class

**Linear regression** predicts a number by adding weighted input features and a bias. For example, `predicted price = 50 + 2 × area` predicts 250 when area is 100, in the chosen units. Fitting chooses weights that reduce an error such as squared prediction error.

**Logistic regression** starts with a linear score too, but passes it through a sigmoid to model a binary probability. A score of zero gives probability 0.5. A threshold then turns that probability into a class label. It is a classifier despite its name.

### Why correlated features can be difficult

If two income columns are almost identical, many pairs of weights can make nearly the same prediction. Individual coefficients can therefore change substantially between fitted samples. This is called **multicollinearity**. Removing a redundant feature or regularising the weights can stabilise the fit.

### Prediction versus interpretation

A coefficient describes the fitted relationship under that model, not automatically the effect of intervening on a feature. Classical standard errors need additional assumptions about the data and errors; normal errors are not required merely to compute an ordinary least-squares fit. Compare validation performance and residuals rather than judging a model only by coefficient size.


### How an interviewer might ask about this

- **Why is logistic regression used for classification?** Its sigmoid turns a linear score into a binary probability model.
- **Two features are almost identical. What can go wrong?** Coefficients can become unstable even if predictions remain similar.

## B16 — Decision trees, random forests and boosting

### One tree makes a sequence of decisions

A decision tree splits examples with questions such as “is income below this threshold?” Each split tries to make the resulting groups easier to predict. For classification, **impurity** measures how mixed the class labels are; Gini impurity and entropy are two common split criteria. Leaves provide the final prediction.

A deep tree can fit small accidental patterns. A **random forest** trains varied trees using resampled rows and random feature choices, then averages probabilities or votes. Averaging helps when their errors are not identical. **Bagging** is this broad idea of aggregating fits from resampled data.

### Boosting corrects the current ensemble

Boosting adds learners sequentially. With squared-error regression, a current prediction of 7 for a target of 10 leaves residual 3. The next small tree learns a correction; a learning-rate factor controls how much is added. XGBoost, LightGBM and CatBoost are boosting systems with different engineering and feature-handling choices, not three fundamentally unrelated tasks.

**Stacking** trains a combiner on expert predictions. Use out-of-fold predictions so it does not learn from unrealistically strong in-sample expert results. Forests and boosting can both overfit; inspect validation results and tune complexity rather than assuming an ensemble cannot fail.


### How an interviewer might ask about this

- **Why use a random forest instead of one tree?** Averaging varied trees reduces sensitivity to individual samples.
- **How is boosting different from bagging?** New boosting learners correct the current ensemble rather than fitting independently.

## B17 — Support vector machines and kernels

### What it means

An SVM looks for a decision boundary with a large margin. A hard-margin formulation permits no violations and requires separability; a soft margin trades violations against margin width. A kernel evaluates inner products in a feature representation without explicitly constructing all its coordinates.

### Example

A linear boundary in a transformed feature space can become a curved boundary in the original two-dimensional input. A linear SVM can also be a strong sparse-text baseline without any nonlinear kernel.

### What to check

A kernel is not free computation: forming or using pairwise relationships can be expensive in memory and time as the training set grows. Tune the violation penalty and kernel settings on training/validation splits.

### How an interviewer might ask about this

- **What is an SVM trying to maximise?** Explain the margin and the cost of violations.
- **How can a linear separator create a curved boundary?** The boundary can be linear in a transformed feature space.

## B18 — Clustering and dimensionality reduction

### Clustering asks which examples belong together

**K-means** alternates between assigning points to a nearby centre and updating each centre from its assigned points. It favours compact groups under squared Euclidean distance. You choose the number of clusters. **DBSCAN** instead links sufficiently dense regions and can mark isolated points as noise. **Hierarchical clustering** builds nested groupings.

### Dimensionality reduction asks how to represent fewer features

**PCA** finds linear directions that preserve as much input variance as possible. It does not use the class labels in ordinary unsupervised fitting. Suppose lighting causes most variation in an image collection, while a small feature distinguishes the classes. Keeping high-variance directions could preserve lighting and discard useful class information.

**t-SNE** and **UMAP** make low-dimensional embeddings emphasising particular neighbourhood relationships. A convenient two-dimensional picture is not proof that the original data has clean categories; apparent islands and distances can depend on settings.

### How to judge usefulness

Scale features appropriately, inspect sensitivity to settings/seeds and ask whether groups support a real task. Internal cluster scores can describe a chosen geometry without establishing meaningful customer categories. For PCA before a classifier, evaluate the full training-only-fitted pipeline on held-out data.


### How an interviewer might ask about this

- **PCA retains 95% of variance. Must classification improve?** Large input variation need not be the useful label signal.
- **K-means returned five groups. Are they meaningful?** Check stability, geometry and usefulness, not just whether fitting succeeded.

## B67 — Distribution shift and transfer learning

### Why performance changes on new data

**Distribution shift** means the data encountered in use differs from the development data. **Transfer learning** reuses learned representations or weights for another task/domain. **Domain adaptation** deliberately tries to handle changed input conditions.

Several changes are possible. **Covariate shift** changes input frequencies while assuming the conditional target rule stays fixed. **Label shift** changes class prevalence while assuming class-conditional inputs stay fixed. **Concept drift** changes the relationship between input and label. Real data can violate all these clean categories.

### Example

A news classifier meets tweets. Shorter text, slang and missing context may hurt it. In argument identification, a second dataset might also count different things as Argument. More tweet training examples cannot settle a changed label definition unless the model is told or learns that definition.

### What to inspect

Compare collection, labels, vocabulary, lengths, groups and error types. Use appropriately labelled target data when permitted, or state clearly when only unlabelled adaptation is allowed. Overall accuracy can hide poor results for a small group. Worst-group evaluation reports that risk; robust optimisation may target it, but depends on reliable group definitions and can trade average performance for protection of difficult groups.


### How an interviewer might ask about this

- **A news classifier fails on tweets. Why?** Check vocabulary, style, coverage and the definition of the label.
- **Overall accuracy is good but one group fails. What now?** Inspect group-specific performance and the training/evaluation populations.

## B07 — Probability, Bayes' rule and likelihood

### Probability and likelihood

**Probability** asks what outcomes a fixed model would produce. If a coin's heads probability is 0.5, how likely is a particular sequence? **Likelihood** keeps the observed sequence fixed and asks which possible coin bias explains it better.

After eight heads in ten flips, **maximum likelihood estimation (MLE)** chooses a heads probability of 0.8. This is the best-fitting value under that model, not proof that the true probability is exactly 0.8.

### Bayes' rule

Bayes updates a prior belief using evidence: posterior is proportional to likelihood times prior. A **prior** describes uncertainty before the current observations; the **posterior** describes it after conditioning on them. **MAP** chooses the parameter with the largest posterior density. A prior favouring moderate coin biases can move the estimate away from an extreme supported by little data.

### A base-rate example

Of 10,000 people, suppose 100 have a condition. A test finds 90 of those, but also flags 99 healthy people. A positive result therefore corresponds to 90 real cases out of 189 positives: about 48%, not 90%. Sensitivity alone did not answer the question; prevalence and false positives mattered too.

A random variable is a numerical outcome before it is observed, such as the number of heads. Its distribution describes possible values and their probabilities. A continuous density is not the probability of one exact value; probability comes from an interval's area.

Frequentist inference treats the unknown parameter as fixed and studies sampling behaviour. Bayesian inference assigns uncertainty over parameters through a prior and posterior. Neither removes the need to justify data and model assumptions.


### How an interviewer might ask about this

- **What changes between probability and likelihood?** Fix parameters to predict data, or fix the data to compare parameters.
- **What is the difference between MLE and MAP?** MAP also uses a prior.

## B08 — Mean, variance, correlation and causation

### Describing values

The **mean** is the sum divided by the count. The **median** is the middle of sorted observations. **Variance** measures spread using squared distances from the mean; standard deviation puts that spread back in the original units.

For latencies 1, 1, 1 and 97 milliseconds, the mean is 25 ms and the median is 1 ms. Neither alone describes the slow tail. A latency percentile, such as p95, tells you a value at or below which about 95% of observations fall under the stated convention.

### Correlation is not a cause

Correlation describes a linear association. Difficult support tickets may both take longer and be more likely to be escalated. That does not establish that escalation caused their delay. You need a design or additional assumptions that address the competing explanations. Zero linear correlation also does not exclude a curved relationship.

### Averages and uncertainty

The **standard error** describes how an estimated quantity, such as a sample mean, would vary between samples. It is not the spread of individual observations. Under suitable independence and finite-variance conditions, sample means become approximately normal as sample size increases: the central limit theorem concerns means, not a promise that the raw data is normal.


### How an interviewer might ask about this

- **Why can average latency hide a bad user experience?** A mean hides the shape and tail of the distribution.
- **Escalated tickets take longer. Does escalation cause delay?** Difficult tickets may be selected for escalation: association alone does not settle cause.

## B79 — Bootstrapping and confidence intervals

### Why resample?

A score is calculated on a particular test sample. Another sample could contain easier or harder cases and produce a different score. **Bootstrapping** estimates this sampling uncertainty by repeatedly drawing from the observed sample itself.

**With replacement** means putting an item back after drawing it. One item can appear several times in a resample; another may not appear at all. Each resample normally has the same number of units as the original sample.

### A small calculation

Suppose four observed values are `[2, 4, 6, 8]`; their mean is 5. One bootstrap draw `[2, 2, 6, 8]` has mean 4.5. Another `[4, 6, 8, 8]` has mean 6.5. Repeat many times to see how the estimated mean moves. A percentile interval uses selected endpoints of that distribution, such as its 2.5th and 97.5th percentiles.

For two classifiers, draw the same example IDs for both, recompute both scores and save **B minus A**. This is a **paired bootstrap**. For macro-F1, recompute F1 from the resampled labels and predictions; do not average per-example F1 values. If several targets share a passage, draw whole passages together: a **cluster bootstrap**.

For an illustrative output, an interval from +.01 to +.05 suggests a positive difference under the analysis assumptions. An interval from −.02 to +.04 leaves the direction unresolved; it does not prove the systems are equivalent.

### What a confidence interval means

A 95% confidence procedure aims to cover the fixed population quantity in about 95% of repeated studies under its assumptions. It is not a 95% probability assigned to this one fixed parameter after observing the interval. The bootstrap approximates sampling from the available cases; it cannot repair an unrepresentative dataset or include all uncertainty from retraining and new domains.

A **prediction interval** concerns a future individual observation, rather than uncertainty about a population mean or score difference. Predicting one person's response time is a different uncertainty question from estimating the average response time.



### Further reading

[SciPy: bootstrap procedure and interval methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

### How an interviewer might ask about this

- **Model B is two points better. How uncertain is that difference?** Resample appropriate evaluation units and compare the models on the same draws.
- **Why resample whole passages instead of individual sentences?** Sentences sharing a passage are dependent observations.

## B09 — p-values, significance and Holm correction

### What a p-value asks

Start with a **null hypothesis**, such as no difference between two systems under a specified testing model. A p-value is the probability, under that null model, of a result at least as extreme as the observed one. Small values indicate tension with the null assumptions; they do not give the probability that your research claim is true.

A permutation test builds a null reference by rearrangements allowed by the design, for example swapping the two systems within valid paired units. This is different from bootstrapping the observed sample to estimate uncertainty.

### Why many tests need care

If you try many variants, more chances exist for a false-positive result. **Holm correction** controls the chance of at least one false rejection within a declared family of tests, assuming the underlying p-values are valid.

For three sorted p-values `.010, .030, .040` at level `.05`:

1. Compare `.010` with `.05/3 ≈ .0167`: it passes.
2. Compare `.030` with `.05/2 = .025`: it fails.
3. Stop; the final comparison is not declared significant.

### What significance does not establish

A tiny gain can be significant but unimportant in practice. A non-significant difference does not prove equivalence. Correcting p-values does not turn a post-hoc analysis into a preregistered one. State the effect size, uncertainty, test family and how the comparison was selected.


### How an interviewer might ask about this

- **Does p = 0.03 mean a 97% chance my claim is true?** No: the probability is calculated under a specified null model.
- **Why correct when testing many model variants?** More tests create more opportunities for false positives.

## B10 — Entropy, cross-entropy and KL divergence

### Measuring uncertainty and prediction cost

**Entropy** is the average uncertainty in an outcome distribution. A fair coin is more uncertain than a coin known always to land heads. With base-2 logs, those entropies are one bit and zero bits respectively.

**Cross-entropy** is the average log cost of using a model's probabilities to predict outcomes from the data distribution. For an observed correct class assigned probability .9, the natural-log cost is `−ln(.9) ≈ .105`; assigning .1 costs about `2.303`. Confident mistakes are expensive.

### What KL adds

If the true distribution is p and the model uses q, `cross-entropy(p,q) = entropy(p) + KL(p||q)`. KL is the extra average cost from using q rather than p. It is directional because p decides how much each possible outcome counts. If q gives zero probability to an outcome possible under p, that direction's KL can be infinite.

**Jensen–Shannon divergence** compares each distribution with their mixture. It is symmetric and bounded for a chosen log base, unlike ordinary KL. None of these probability measures is the same as accuracy or F1: changing confidence can change cross-entropy without changing the predicted class.


### How an interviewer might ask about this

- **Why does a confidently wrong answer get a large loss?** Log loss strongly penalises assigning little probability to the observed target.
- **Why is KL divergence directional?** The first distribution determines which outcomes receive weight.

## B11 — Accuracy, precision, recall and F1

### Start with the positive class

A **true positive** is a real positive found by the classifier. A **false positive** is a false alarm. A **false negative** is a missed positive. A **true negative** is a correctly rejected negative.

**Accuracy** is the fraction of all predictions that are correct. **Precision** asks how many predicted positives were right. **Recall** asks how many real positives were found. **Specificity** is recall for the negative class.

### Work through the counts

Suppose there are eight true positives, four false positives, two false negatives and six true negatives:

- Accuracy: `(8 + 6)/20 = 0.70`.
- Precision: `8/(8 + 4) = 0.667`.
- Recall: `8/(8 + 2) = 0.80`.
- Positive F1: `2×8/(2×8 + 4 + 2) = 0.727`.

F1 is the harmonic mean of precision and recall. Catching more positives can raise recall while enough extra false alarms reduce precision and F1.

### Which average?

**Macro-F1** computes F1 per class, then gives each class equal weight. Here negative-class F1 is `12/(12 + 2 + 4) = 0.667`, so macro-F1 is about `0.697`. **Weighted macro** weights class scores by their true counts. **Micro** pools counts first; in ordinary single-label multiclass classification, micro-F1 equals accuracy. **Sample averaging** is a separate multilabel convention.

Use the metric that reflects the task's errors. A detector that calls every case negative can be highly accurate on a rare-positive dataset and still be useless at finding positives.


### How an interviewer might ask about this

- **A fraud detector is 99% accurate. Is it useful?** Ask about prevalence, missed fraud and false alarms.
- **Recall increased but macro-F1 fell. Is that possible?** More positive predictions can catch positives and create many false positives.

## B12 — Thresholds, ROC-AUC and precision–recall curves

### The threshold chooses how many positives you predict

A model may score an example .7. A threshold of .5 labels it positive; a threshold of .8 does not. Lowering the threshold usually finds more positives but also creates more false alarms. Choose using validation data and the costs of the task, not because .5 is always correct.

### Two ways to view the trade-off

An **ROC curve** plots true-positive rate (recall) against false-positive rate as the threshold changes. False-positive rate is false positives divided by all actual negatives. **ROC-AUC** summarises ranking across thresholds.

A **precision–recall curve** plots precision against recall. It makes the number of false alerts relative to true alerts more visible, especially when positives are rare.

### Why a small rate can still be a large problem

With 10 positives and 9,990 negatives, a 1% false-positive rate means about 100 false alerts. Finding eight positives then gives precision about `8/108 ≈ 7.4%`. That may overwhelm a review team despite the small false-positive rate.

Report an operating point the team can use, not only an area score. Precision depends on prevalence; compare curves on the same population. Average precision and trapezoidal PR-AUC are related summaries, not identical calculations.


### How an interviewer might ask about this

- **Which threshold should an alerting system use?** Match the precision/recall trade-off to the cost and review capacity.
- **Why can ROC-AUC look good for a poor fraud detector?** A small false-positive rate can still mean many false alerts.

## B13 — Calibration: can I trust the model's probabilities?

### What it means

Accuracy asks whether labels are right. Calibration asks whether stated probabilities match observed frequencies. Cross-entropy punishes confident mistakes logarithmically; Brier score uses squared probability errors. Selective prediction lets the system abstain on some cases and evaluates error together with the proportion it answers.

### Example

If 100 comparable cases are scored near 0.8, about 80 should be positive for that probability to be calibrated. Two systems can predict the same labels while one says 0.6 and the other 0.999; their accuracy matches but probability losses differ. Tune an F1 threshold or abstention rule on validation data.

### What to check

A next-token probability is not a calibrated probability that a whole answer is factually correct. Calibration on yesterday's population need not survive a prevalence or task change.

### How an interviewer might ask about this

- **When the model says 80%, what should that mean?** About 80% of comparable cases should be positive.
- **Can two equally accurate models have different probability losses?** Yes: their confidence can differ even when labels match.

## B19 — Neural networks, loss and backpropagation

### What a neural network computes

A layer takes numbers, multiplies them by learned weights, adds a bias and often applies a nonlinear function. Stacking nonlinear layers lets the model learn relationships that a single linear map cannot represent. ReLU keeps positive values and replaces negative ones with zero; sigmoid maps a number to between zero and one.

### One training step

1. **Forward pass:** use current weights to produce predictions.
2. **Loss:** calculate how wrong those predictions are under the chosen objective.
3. **Backpropagation:** use the chain rule to calculate how the loss changes with each trainable weight.
4. **Optimiser update:** change weights using those gradients and the learning rate.

For one weight `w`, gradient descent uses `w_new = w − learning_rate × gradient`. If `w = 2`, the gradient is `3`, and the learning rate is `.1`, the new weight is `1.7`. Backward calculates the gradient; it does not itself perform this update.

### Match outputs to the task

For a ten-class single-label classifier, produce ten logits per example and use integer class targets with a suitable cross-entropy implementation. Do not apply softmax twice. For regression, choose a numeric output and an appropriate regression loss. Clear accumulated gradients at the intended update boundary; evaluation should not silently change training state.


### How an interviewer might ask about this

- **What happens during one training step?** Trace input, prediction, loss, gradient and weight update.
- **Why do networks need nonlinear activation functions?** Otherwise stacked linear layers collapse to one linear map.

## B21 — Normalisation, initialisation and residual connections

### Why control the scale of internal values?

Very large or tiny activations and gradients can make optimisation difficult. **Initialisation** sets the starting weights to useful scales. **Normalisation** rescales selected groups of activations during computation. It is not the same as scaling dataset columns before training.

### Which values are grouped together?

- **BatchNorm:** uses batch-related statistics for each feature/channel during training, with running statistics commonly used at evaluation.
- **LayerNorm:** uses the selected feature dimensions within an individual example or token, not other examples in the batch.
- **RMSNorm:** rescales by root-mean-square magnitude without subtracting the mean.
- **GroupNorm:** normalises groups of channels, commonly in image models.

The exact dimensions matter. Saying “normalise the layer” without naming them is incomplete. BatchNorm can be sensitive to small or unrepresentative batches; LayerNorm does not rely on those batch statistics.

### Residual connections

A residual block adds its input back: `output = input + learned_change(input)`. The learned branch can adjust rather than reconstruct the full representation, and gradients have an additive route through the block. This helps deep networks train, but it does not make depth free or guarantee improved validation performance.


### How an interviewer might ask about this

- **What is the difference between BatchNorm and LayerNorm?** State which dimensions supply the normalisation statistics.
- **Why do residual connections help deep networks?** They give information and gradients a shorter additive path.

## B23 — Regularisation, dropout and early stopping

### Why regularise?

Regularisation discourages a model from depending too strongly on accidental details of its training set. It is useful when training performance is strong but new-example performance is poor; it will not repair every data or optimisation bug.

**L2** penalises large squared weights. **L1** penalises absolute weights and can push some to zero. **Dropout** randomly removes selected activations during training so later layers cannot always rely on one feature. **Early stopping** selects a checkpoint before further training harms validation performance.

### Example

Training accuracy rises from 90% to 99%, while validation peaks at 84% and falls to 78%. Keeping the checkpoint at the validation peak may generalise better than using the last epoch. Select that point using validation, not test data.

During ordinary inference, dropout is disabled. Inverted dropout scales retained activations during training to preserve their expectation, so no matching random removal is needed at evaluation. `model.eval()` changes these behaviours; it is not itself a command to disable gradient recording.

### Choose the remedy from the problem

If both training and validation are poor, stronger regularisation may worsen underfitting. First check the inputs and whether optimisation works. AdamW's decoupled weight decay is not identical to adding an L2 term to Adam's loss.


### How an interviewer might ask about this

- **Training improves while validation worsens. What should I try?** This is a common overfitting pattern; regularisation or early stopping may help.
- **Why do predictions change between train and eval mode?** Dropout and some normalisation layers behave differently.

## B24 — Learning rate, SGD, Adam and batch size

### What the optimiser controls

The gradient indicates a local direction for reducing loss. The **learning rate** sets the size of an update. Too large can overshoot or diverge; too small can make progress very slow.

**SGD** uses a minibatch gradient. **Momentum** also carries a running direction, helping move through noisy or narrow regions. **Adam** tracks averages of gradients and squared gradients to adapt updates per parameter. **AdamW** separates weight decay from that adaptive gradient update.

### Batch size and schedules

A larger batch generally gives a less noisy gradient estimate, but uses more memory and gives fewer updates per fixed number of examples. It is not automatically better. Changing it may require retuning the learning rate.

A **schedule** changes the learning rate over training. **Warmup** starts with smaller updates; decay reduces them later. Validation-based stopping chooses when to stop fitting. These are different controls, even if all affect the loss curve.

### When loss behaves badly

If loss suddenly becomes unstable after a learning-rate increase, compare the update size, gradient norms and precision. If it never decreases, also check targets, masks, frozen weights and the optimiser. Do not blame the learning rate before confirming that the training step is correct.


### How an interviewer might ask about this

- **Loss oscillates after increasing the learning rate. Why?** Updates may overshoot useful regions.
- **Can a bigger batch require different settings?** It changes gradient noise, update frequency and memory use.

## B26 — Training memory, precision and gradient accumulation

### Why weights are only part of training memory

Training needs model weights, gradients, optimiser state, saved activations and temporary buffers. A model that fits for inference may therefore fail during training. Batch size and sequence length can dominate activation memory.

**Mixed precision** uses lower-precision arithmetic for suitable operations while preserving more precision where needed. **Activation checkpointing** discards selected intermediate activations and recomputes them during backward: less memory, more computation.

### Gradient accumulation

Process several small **microbatches**, accumulate their gradients, then update once. This can approximate a larger batch without storing all its activations simultaneously. It may not exactly reproduce a large batch when dropout or batch-dependent operations differ.

Weight losses by the number of valid targets. If one microbatch has 10 tokens with mean loss 2, and another has 90 with mean loss 4, the combined mean is `(10×2 + 90×4)/100 = 3.8`, not `(2+4)/2 = 3`. Clear gradients at the update boundary, not between microbatches you intend to combine.

### More devices

Data parallelism gives devices different data with replicated weights; tensor or pipeline parallelism splits model computation. Communication can outweigh useful work on small jobs. More GPUs therefore need not mean proportionally faster training.


### How an interviewer might ask about this

- **Weights fit in GPU memory. Why does training run out?** Include activations, gradients, optimiser state and temporary buffers.
- **How do unequal microbatches affect the loss?** Weight by the number of valid training targets, not just the number of microbatches.

## B80 — CNNs: filters, channels and pooling

### What a convolution does

A convolution slides learned filters across an image. A filter combines values in a local patch to produce an output value. Reusing the same filter at many positions is **weight sharing**: a useful edge pattern need not be relearned separately at every pixel.

**Channels** are separate input or output feature maps. **Kernel size** controls the local window. **Stride** controls the distance between positions. **Padding** adds border values so the output shape can be controlled. **Pooling** summarises nearby values, often reducing spatial size.

### A shape and parameter example

A 3×3 convolution with three input channels and sixteen output channels has `3×3×3×16 = 432` weights, plus sixteen biases if used. On a 32×32 image, stride 1 and padding 1 preserve the 32×32 spatial shape. The output has sixteen channels.

### Why memory grows

More channels mean more activation values and more input/output-channel combinations in later convolutions. Doubling both input and output channels roughly quadruples that layer's convolution arithmetic, with other factors fixed. An autoencoder instead learns an encoding and reconstruction; low reconstruction error does not by itself prove useful classification features.


### How an interviewer might ask about this

- **Why does a CNN use the same filter across an image?** Weight sharing detects a local pattern at multiple positions.
- **Why did doubling channels greatly increase compute?** A convolution mixes input and output channels at every spatial location.

## B81 — RNNs, LSTMs, GRUs and bidirectionality

### Remembering earlier input

An **RNN** reads a sequence one step at a time. At each step it combines the new input with a hidden state summarising earlier steps. Repeated transformations can make distant information or its gradient fade or grow too much.

An **LSTM** adds a cell state and gates controlling what to forget, write and read out. Think of carrying a note along the sequence while deciding what to erase or update. A **GRU** uses update and reset gates with a simpler state arrangement. These mechanisms help learning long dependencies; they do not guarantee perfect memory.

### One direction or two?

A forward RNN reads left to right. A **bidirectional** RNN has separate forward and backward models and combines their states. This is useful when the full sentence is available, such as tagging words in an existing document. It cannot use the actual future words for a next-word prediction that happens while someone is typing.

### Example

In “The keys to the cabinet are missing,” a sequence model should preserve information about “keys” when processing “are.” Gates can help retain it through the intervening phrase. When batching different lengths, masking or packing must stop padding from becoming meaningful input or a training target. Choose GRU versus LSTM by task results and resource cost, not a universal ranking.


### How an interviewer might ask about this

- **How does an LSTM preserve information longer than a basic RNN?** Explain the cell state and gates.
- **Can a bidirectional model predict the next typed word?** Future input is unavailable in that setting.

## B27 — Text features: n-grams, TF–IDF and word embeddings

### Three ways to represent text

**Bag of words** counts word occurrences and discards order. **N-grams** count short sequences, so “not good” can remain different from separate occurrences of “not” and “good.” **TF–IDF** gives more weight to terms distinctive within the document collection and less to very common ones.

These sparse vectors work well with a linear classifier and are a useful cheap baseline. Fit the vocabulary and IDF statistics using the training data, not the held-out test collection unless the evaluation explicitly permits that information.

**Naive Bayes** is another inexpensive text baseline. It combines class priors and feature evidence under a conditional-independence assumption: features are treated as independent given the class. Real words violate that simplification, but the resulting classifier can still be useful.

### Word embeddings

A static embedding gives each vocabulary item one learned dense vector. A contextual encoder gives a token a representation that changes with the surrounding sentence: “bank” can mean something different near “river” and “loan.” A sentence embedding represents a whole text for a chosen purpose, such as similarity.

### Cleaning is a task choice

Stemming mechanically trims word forms; lemmatisation uses linguistic information to identify a base form. Aggressive cleaning can remove useful distinctions, punctuation or negation. A traditional n-gram language model predicts a token from a short preceding history; smoothing assigns some probability to combinations absent from training. Transformer tokenisation normally replaces, rather than requires, a separate stemming pipeline.


### How an interviewer might ask about this

- **What is a sensible first text-classification baseline?** Sparse features plus a linear model give a cheap, inspectable baseline.
- **How do stemming and lemmatisation differ?** Stemming trims forms; lemmatisation uses linguistic information to recover a base form.

## B28 — Tokenisation, padding and context length

### Text becomes IDs

A **tokenizer** splits text into units and maps them to numeric IDs. Units may be words, subwords, characters or bytes. A **vocabulary** is the set of token types; **context length** is how many token positions a model accepts together. One rare name can take several tokens, so word count is not token count.

The tokenizer and model must agree about ID meanings. Changing the vocabulary without matching the embeddings can silently give the model different inputs. Chat templates also insert role markers and boundaries the model may have learned to expect.

### Batching different lengths

**Padding** adds artificial positions so sequences form a rectangular batch. A padding mask identifies positions the model should not treat as real text. Exclude padded targets from the loss too. **Truncation** cuts input to a length limit; always check whether it removed the target sentence or needed evidence.

### Long documents

Options include selecting relevant passages, splitting into chunks, combining chunk representations, or using a model designed for the required length. Chunking can separate an exception from its rule. A longer window costs memory and does not guarantee the model uses every passage reliably. Character offsets, normalisation and special tokens also matter when aligning token predictions back to source text.


### How an interviewer might ask about this

- **Why do 1,000 words not always mean the same token count?** Tokenisation depends on the vocabulary, language and text.
- **What if a document is longer than the model accepts?** Choose truncation, chunking or a suitable architecture, and check what information is lost.

## B30 — Classification, sequence labelling and extraction

### Name what gets a prediction

**Text classification** assigns a label to a sentence or document. **Sequence labelling** assigns labels to tokens, such as person/entity tags. **Span extraction** identifies a stretch of text. **Relation extraction** predicts a connection between identified items. The required output determines the model head, loss and evaluation.

### Example: named entities

In “Ada works in Paris,” a system might mark Ada as a person and Paris as a location. Most tokens in a dataset may be outside entities. Predicting O everywhere can then get high token accuracy while finding no complete entities. Entity-level precision/recall/F1 checks boundaries and types, not just individual token agreement.

### Language and task definitions

Syntax concerns grammatical arrangement; semantics concerns meaning; pragmatics concerns intended use in context; discourse concerns connections across sentences. “Great, another delay” may be a complaint despite its positive word.

Before adapting a model to another dataset, inspect what its labels mean. One argument dataset may count claims, another may also count supporting material. The same text can legitimately receive different labels. Start a classification study with a simple baseline, inspect errors, and justify a larger encoder from the needed evidence and budget.


### How an interviewer might ask about this

- **NER has high token accuracy but poor entity F1. How?** Many O labels can hide incorrect entity boundaries.
- **What exactly does this dataset label?** Identify the prediction unit and annotation definition before choosing a model.

## B31 — HMMs, CRFs and valid label sequences

### Why labels should sometimes depend on their neighbours

In sequence labelling, classifying every token independently can create an invalid whole sequence. **BIO** tags mark the beginning of an entity (B), its continuation (I), or outside text (O). An I-PER immediately after O is invalid under the usual strict scheme.

A **CRF** scores a label sequence using local input evidence and relationships between neighbouring labels. A neural encoder can supply token scores, while transition scores or explicit constraints discourage invalid combinations. It models labels conditioned on the observed input.

### HMM and decoding

An **HMM** instead models hidden-label transitions and how observations are emitted from hidden states. A sequence of hidden entity labels is one possible application. This generative formulation differs from a CRF's conditional one.

**Viterbi decoding** finds the highest-scoring path by storing the best partial score ending in each state, then extending those partial paths. It avoids trying every full sequence separately. The model still needs the right transition constraints: using a CRF does not automatically enforce every desired annotation rule or fix missing evidence.


### How an interviewer might ask about this

- **Why put a CRF on top of an encoder?** Combine local evidence with scores for neighbouring labels.
- **How do you find the best whole label sequence?** Dynamic programming reuses partial solutions instead of enumerating every sequence.

## B29 — Search: BM25, embeddings and reranking

### Retrieval before reranking

**BM25** scores lexical matches using term frequency, document length and how common a term is in the collection. **Dense retrieval** compares learned query/document vectors and can find related wording without exact matches. Neither is always better: an error code may need exact matching, while a paraphrase may benefit from semantic representations.

An **approximate nearest-neighbour index**, such as HNSW or an IVF-based index, trades some retrieval accuracy for speed and memory. A **reranker** then scores a small candidate set more carefully, often reading query and document together. It cannot recover an item absent from that candidate set.

### Similarity and ranking metrics

Cosine compares vector direction; dot product also depends on magnitude. With unit-normalised vectors they give the same ranking. Finding the wrong sense of a word may call for better query context or embedding training, not just a larger index.

**Recall@k** asks how many relevant items were found in the first k. If one of two useful papers appears in the top five, recall@5 is 1/2. **MRR** uses the first relevant rank: rank four contributes 1/4. **MAP** averages precision at relevant hits; **NDCG** also handles graded relevance and position discounts. Choose the metric that matches whether you need one early answer or comprehensive evidence.


### How an interviewer might ask about this

- **Why did keyword search beat an embedding on an error code?** Exact identifiers can matter more than general semantic similarity.
- **How do you search a million documents efficiently?** Retrieve candidates first, then rerank a small set.

## B36 — Sentence embeddings and contrastive learning

### What a sentence embedding should preserve

A sentence embedding compresses text into a vector so a chosen notion of similarity can be compared cheaply. A contextual encoder is not automatically trained to make its pooled vectors good for semantic search. The training objective and pooling method matter.

### How contrastive learning teaches similarity

A **positive pair** should be close, such as a question and its correct answer. **Negative pairs** should be farther apart. The loss rewards distinguishing the positive from alternatives. The temperature controls how sharply similarity scores affect that choice.

If an alternative is also a correct answer, treating it as a negative sends the wrong training signal. This is a **false negative** in contrastive training, distinct from a classifier missing a positive test case.

### What similarity does not prove

“The drug improves survival” and “The drug does not improve survival” share much wording and may have similar vectors. Similarity does not establish entailment, factual support or the same argument label. For retrieval, test relevant-query/document pairs; for support, use an evaluation that checks the actual relation. Compare cosine, dot product and normalisation under the intended model rather than assuming their rankings are identical.


### How an interviewer might ask about this

- **Can I use any BERT CLS vector for semantic search?** A sentence-similarity training objective matters.
- **What if a negative example is actually another valid answer?** False negatives push genuinely related representations apart.

## B33 — Evaluating summaries and generated text

### A good answer is not always the same string

A summary or translation can be correct in several different wordings. Automatic text metrics compare a generated answer with a reference, but each captures only part of quality.

**BLEU** rewards matching word sequences, with a penalty for answers that are too short. **ROUGE** variants measure reference overlap, often emphasising how much reference content was covered. **chrF** compares character n-grams; **METEOR** includes additional matching/alignment rules. **BERTScore** uses contextual representations to compare text. State the variant and implementation when reporting results.

### Why overlap is not enough

“Treatment reduced mortality” and “Treatment did not reduce mortality” share most words but reverse the conclusion. A correct paraphrase can instead share few words. Check important claims, numbers, negation and attribution, not just a similarity score.

### Distinguish the quality questions

**Factuality** concerns whether a statement is true. **Source faithfulness** concerns whether it accurately reflects the supplied source. **Helpfulness** concerns whether it meets the user's need. A fluent, source-faithful summary can still repeat an incorrect source; a true answer can be unsupported by its claimed citation. Use task-specific checks and representative human judgements for the dimensions that matter.


### How an interviewer might ask about this

- **A summary has high ROUGE but reverses the conclusion. Why?** Word overlap does not establish factual consistency.
- **What would you measure besides fluency?** Check task correctness, source support, coverage and harmful omissions separately.

## B38 — Attention: queries, keys, values and softmax

### Attention mixes information using learned relevance

Each token has a **query** vector describing what it is looking for, a **key** vector used for matching and a **value** vector carrying information to combine. Compare a query with allowed keys, turn those scores into weights, then take a weighted sum of values.

For one head: `Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V`. Here d_k is the key-vector dimension. Scaling helps control score magnitude. **Softmax** exponentiates scores and divides by their sum to produce nonnegative weights adding to one.

### A small calculation

Two equal scores give weights .5 and .5. If the values are 2 and 6, the output is `.5×2 + .5×6 = 4`. With vector values, the same weighted sum operates coordinate by coordinate. Several heads learn different projections, then combine their outputs.

### What can be attended to?

Self-attention reads the same sequence; cross-attention reads another representation. A causal mask blocks future positions, while padding masks block artificial padding. To compute softmax stably, subtract the largest score before exponentiating; the resulting probabilities are unchanged. Attention weights show information mixing, not automatically why the final answer is correct.


### How an interviewer might ask about this

- **What do Q, K and V do?** Explain matching, weighting and information mixing.
- **How do you compute softmax without overflow?** Subtract the largest logit before exponentiating; the probabilities do not change.

## B39 — Encoders, decoders and causal masks

### What each architecture can read

An **encoder** can usually attend to the whole provided input and is useful for representations or classification. A **causal decoder** predicts a continuation using earlier tokens only. An **encoder–decoder** first represents a source, then generates an output while attending to that source. Architecture and training objective together determine behaviour.

### Parallel training, sequential generation

For the sequence “cats chase mice,” training can supply “cats chase” as input and “chase mice” as targets. A causal mask prevents the prediction at “cats” from reading “chase.” Because the whole training sequence is known, predictions at all allowed positions can be computed together.

At generation time the next actual token is not known. Generate one, append it and repeat. **Teacher forcing** means using the known training prefix; **exposure bias** is the mismatch when generation must instead continue from its own imperfect outputs.

### Different masks

A causal mask hides future tokens. A padding mask identifies artificial padding. An attention mask controls what can be read; a loss mask controls which target positions count in optimisation. They are not interchangeable. A model that can read future targets during training may get a low loss by using information unavailable at generation time.


### How an interviewer might ask about this

- **Why can training run in parallel but generation is sequential?** During training the target sequence is known, but future targets must be masked.
- **What happens if the decoder can read its future targets?** It can copy information that will not exist at generation time.

## B40 — Position embeddings and RoPE

### What it means

Attention alone treats a set of vectors without an inherent left-to-right order. Position mechanisms tell the model which positions or relative offsets are involved. Learned/sinusoidal embeddings add positional signals; RoPE rotates query/key coordinates to encode relative-position relationships; ALiBi adds distance-related attention biases.

### Example

The words 'dog bites man' and 'man bites dog' use the same vocabulary but need different representations. A causal mask already supplies some asymmetric positional structure, so 'no explicit position embedding' does not always mean 'no position information whatsoever'.

### What to check

A position scheme supporting a long window does not establish reliable use of every location. Test extrapolation, truncation and task quality rather than assuming a context-length setting proves comprehension.

### How an interviewer might ask about this

- **How does attention know token order?** Positional information must be supplied through a chosen mechanism.
- **Does extending the context window guarantee good long-document reasoning?** Accepting more tokens and using them reliably are different.

## B41 — Efficient attention: GQA, FlashAttention and caching

### Why attention becomes expensive

With n tokens, ordinary dense self-attention considers roughly n×n query/key relationships per head. Long inputs can therefore be expensive even when the weights are unchanged.

### Three different improvements

**Grouped-query attention (GQA)** lets several query heads share fewer key/value heads. This reduces stored key/value state during generation; it is a model architecture choice, not just a serving flag.

**FlashAttention** computes dense attention using tiles and less expensive movement between fast and slower memory. It avoids storing the full attention matrix in the ordinary way, but does not turn dense attention into a linear-time algorithm.

**PagedAttention** manages the storage of growing KV caches in blocks, helping serving systems use memory efficiently. It addresses a different issue from FlashAttention's attention computation.

### Another decoding option

Speculative decoding lets a cheaper draft model propose tokens for verification by the target model. Exact speculative sampling schemes preserve the target distribution under their assumptions; not every heuristic draft/accept system does. Actual speed depends on acceptance rate and overhead, so measure the full system.


### How an interviewer might ask about this

- **Why does grouped-query attention save KV memory?** Several query heads share key/value heads.
- **Does FlashAttention remove attention's quadratic arithmetic?** It reduces expensive memory traffic without changing dense attention into sparse attention.

## B42 — Next-token training, cross-entropy and perplexity

### The training target comes from the text

For “the cat sleeps,” a causal language model learns to predict “cat” after “the” and “sleeps” after “the cat.” No person needs to label those next tokens separately. The model produces a vocabulary distribution and is penalised when it assigns too little probability to the observed next token.

### Cross-entropy and perplexity

For a correct token assigned probability p, the negative log loss is `−ln(p)`. Probability `.5` costs about `.693`; `.1` costs about `2.303`. Average over the specified valid tokens. **Perplexity** is the exponential of that mean loss when natural logs are used; mean loss `.693` gives perplexity about 2.

Lower perplexity means better prediction of that tokenised text under that evaluation. It does not directly measure factuality, reasoning or usefulness. Different tokenizers change the units, so their raw perplexities are not automatically comparable.

### Data and compute

Larger models need sufficient useful data and compute, not just more parameters. Compute-optimal training asks how to allocate a budget between model size and training tokens. Duplicate, low-quality or contaminated text can make the apparent scale misleading. Report the data, tokenizer, loss mask and budget before comparing runs.


### How an interviewer might ask about this

- **Where do the labels for language-model pretraining come from?** The next observed tokens are the targets.
- **Can I compare perplexity across different tokenizers directly?** The unit being averaged has changed.

## B44 — Pretraining, instruction tuning and forgetting

### What it means

Continued pretraining keeps learning the broad language objective on new text. Supervised fine-tuning learns from task demonstrations or labelled targets. Catastrophic forgetting means adaptation damages capabilities learned earlier; low new-domain training loss does not reveal that damage.

### Example

Technical manuals may help a base model learn terminology through continued pretraining. Example question/answer pairs teach a response behaviour through SFT. A small encoder classifier may be the simpler choice when the only output is a label.

### What to check

Keep a held-out task set and a relevant retention set. Diagnose prompt formatting, overfitting and objective mismatch before concluding that more finetuning is needed. Retrieval supplies evidence without changing weights.

### How an interviewer might ask about this

- **Should I fine-tune to give the assistant today's policy?** Retrieval handles changing evidence; tuning can teach behaviour or format.
- **Why did specialist tuning harm general ability?** Updates can overwrite useful behaviour or narrow the training distribution.

## B45 — Knowledge distillation

### What it means

Distillation trains a student to match information supplied by a teacher: distributions, labels, responses or solution traces. The student still needs a training objective; merely calling a teacher API is not distilling a model. Self-distillation uses a model or related checkpoint as its own teacher in a specified procedure.

### Example

A larger model can label training texts, then a smaller encoder learns those labels. If the teacher consistently mistakes premises for claims, the student may inherit that bias. High agreement demonstrates imitation, not independent semantic correctness.

### What to check

State the supervision, data rights, filtering and independent evaluation. A smaller distilled model's success on selected tasks does not establish that it copied all of the teacher's abilities.

### How an interviewer might ask about this

- **How can a smaller model learn from a bigger one?** Train on the teacher's outputs or distributions.
- **Does agreement with the teacher establish correctness?** The student can inherit the teacher's mistakes.

## B46 — Ensembles, routers and mixture of experts

### What it means

A static ensemble combines predictions using a fixed rule. A stacker learns from expert predictions. A router chooses or weights experts for an input, and a jointly trained mixture can learn experts and gate together. Sparse LLM MoE layers activate only selected experts per token, unlike simply running several complete chatbots.

### Example

A model can have many total expert parameters but use only two experts per token, reducing active arithmetic relative to activating them all. Loading, communication, routing and shared layers still cost resources. In a classifier ensemble, correlated errors or miscalibrated probabilities can make an average worse than the strongest expert.

### What to check

Name what is selected—prompt, classifier or internal expert—and at what granularity. A dataset-level route using labelled development data is not an oracle for an unseen individual sentence.

### How an interviewer might ask about this

- **How is a router different from averaging predictions?** It chooses or weights experts rather than applying a fixed combination.
- **Does a sparse MoE use all its parameters for every token?** Only selected experts are activated, although shared computation and communication remain.

## B47 — RL post-training: rewards, PPO, DPO and GRPO

### From imitation to preference or reward

Supervised fine-tuning imitates example answers. Preference training instead favours responses judged better than alternatives. A **reward** is a scoring signal, not a guarantee of truth. A **policy** is the model's distribution over possible next tokens/responses.

### The basic RL flow

Generate candidate responses, score them, estimate which actions were better than expected, and update the policy to favour those actions. That update changes model parameters: it is a form of post-training. Online methods repeat collection and updating as the policy changes.

- **PPO** commonly uses a learned value estimate and a clipped probability-ratio objective to discourage overly large policy changes. Clipping limits parts of the incentive; it does not guarantee every update is small.
- **GRPO** commonly compares rewards within a group of responses to the same prompt, avoiding a separate learned value critic. If every reward is identical, the group provides no relative preference signal under the usual normalisation.
- **DPO** learns from preferred/rejected pairs relative to a reference formulation, without the same online rollout/critic loop. It increases the relative preference for chosen answers, not simply their length or fluency.

### What can go wrong?

A model can exploit the scoring rule, become overly agreeable or improve judged style without improving correctness. Check outcomes separately. IPO changes the pairwise preference objective; ORPO combines supervised learning with an odds-ratio preference term. Knowing these distinctions is not the same as having implemented each trainer.



### Further reading

[DPO](https://arxiv.org/abs/2305.18290), [PPO](https://arxiv.org/abs/1707.06347), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

### How an interviewer might ask about this

- **What happens in one RL update for an LLM?** Generate responses, score them, estimate an update signal and change policy weights.
- **Why use DPO instead of an online RL loop?** Available preference pairs can train a different objective without the same rollout/critic loop.

## B50 — Can LLMs reason? Models, tools and explanations

### Capability is not the same as human thought

LLMs can solve some new multi-step problems without external tools. It is therefore too strong to say they cannot reason at all. But fluent explanations and correct answers do not establish human-like thought, consciousness or reliable general reasoning. Next-token training describes an objective; it does not by itself settle which computations the trained model can perform.

### What is the harness doing?

A **harness** is the surrounding software: prompting, search, memory, tools, retries or verification. Compare the model answering alone, the model with extra sampling/scratchpad text, and the full tool-assisted system. A calculator can explain an arithmetic improvement in the last condition, but not every success of the model without tools.

**Chain of thought** is generated intermediate text. **Self-consistency** combines answers from several samples. A **verifier** checks or ranks candidates. These spend additional computation, and more steps can still amplify mistakes. Evaluate on changed tasks under explicit budgets.

### A natural interview position

“I would separate the model's own ability from what its tools and workflow add. I would not infer human-like thinking from fluent explanations, but I also would not claim it cannot reason. I would test generalisation and whether changing the apparent reasoning changes the answer.” This last issue is **rationale faithfulness**: the printed explanation may omit or invent influences rather than reveal the actual decision process.



### Further reading

[DeepSeek-R1](https://arxiv.org/abs/2501.12948), [rationale-faithfulness interventions](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)

### How an interviewer might ask about this

- **Is the model reasoning, or are tools doing the work?** Separate the model alone from scaffolding and tool-assisted performance.
- **Does a correct chain of thought explain the actual decision?** A plausible rationale is not automatically faithful evidence of the internal process.

## B51 — LoRA and QLoRA

### What it means

Full fine-tuning updates the chosen base weights. LoRA freezes a matrix W and learns a low-rank update BA, often scaled by alpha/r. For a d_out by d_in matrix, the adapter has r(d_in + d_out) weights rather than d_in·d_out. Prompt tuning learns input-side embeddings instead of those weight updates.

### Example

For a 1,024 by 1,024 matrix and rank 8, the adapter uses 8(1024 + 1024) = 16,384 weights versus 1,048,576 in the matrix. QLoRA stores the frozen base in a low-bit representation and trains adapters in suitable compute precision. NF4 is a nonuniform 4-bit codebook; double quantisation also compresses quantisation constants.

### What to check

Activations still use memory. Rank, target modules, scaling, data and objective affect quality; low rank may miss a useful full-rank change. Attention-only and MLP adaptation are different interventions. QLoRA does not train every base weight in four-bit arithmetic.

### How an interviewer might ask about this

- **Why can LoRA still run out of memory?** Fewer trainable weights do not remove activation or base-model memory.
- **How many parameters does a rank-8 adapter add?** Count the two low-rank matrices, not the frozen base matrix.

## B52 — Attention masks, loss masks and chat templates

### What it means

An attention mask controls what information a token can read; a loss mask controls which targets contribute to optimisation. Packing combines sequences to reduce wasted padding; dynamic padding pads only as much as the current batch requires. Their boundary rules must match the intended objective.

### Example

For assistant-only SFT, user text can be visible as context while its tokens are excluded from the target loss. A bad EOS or chat template can teach the model to continue user turns or fail to stop. Packing unrelated examples without the intended document boundaries can leak context between them.

### What to check

Inspect actual IDs, masks and decoded target tokens on a tiny batch. 'The loss decreased' does not certify the formatting or target shift is correct.

### How an interviewer might ask about this

- **Why exclude user tokens from the SFT loss?** They may supply context without being outputs the assistant should imitate.
- **What should I inspect before trusting a training run?** Decode actual IDs and check targets, masks, boundaries and template.

## B53 — Decoding: greedy, beam search and sampling

### What it means

Greedy decoding takes the highest-scoring next token. Beam search retains several partial sequences under a scoring rule. Top-k sampling samples from k candidates; top-p samples from a probability-mass-selected set. Temperature rescales logits before sampling; penalties modify scores to discourage repeats or change length preference.

### Example

A model assigning probabilities 0.6 and 0.4 can always choose the first token greedily or sometimes choose the second under sampling. A low-temperature answer can still confidently be false. Beam search optimises its sequence scoring rule, not factual correctness.

### What to check

Temperature zero does not guarantee identical outputs across hardware, kernels, backend versions or ties. Record the complete decoding and serving setup, not just one parameter.

### How an interviewer might ask about this

- **Does temperature zero make an answer true?** Decoding controls selection, not factual correctness.
- **How are top-k and top-p different?** One limits candidate count; the other limits cumulative probability mass.

## B54 — Serving: prefill, decode, latency and throughput

### A request has more than one stage

**Queueing** is waiting for service. **Prefill** processes the prompt and builds cached keys/values. **Decode** generates later tokens using that cache. A long input can make the first token slow even when later tokens arrive quickly.

**Time to first token (TTFT)** measures the initial wait under the stated measurement boundary. **Inter-token latency** measures the gaps between later tokens. **Throughput** measures completed work per time; it is not the same as one user's response time.

### A calculation

Suppose TTFT is 100 ms, and the next nine tokens take 20 ms each. Ten output tokens finish after `100 + 9×20 = 280 ms`. Another system with 50 ms TTFT but 40 ms later-token gaps finishes in 410 ms. “Faster first token” did not mean “faster complete answer.”

### Find the bottleneck

If p95 latency doubles while average generation rate is stable, check queues, long prompts and concurrency. If GPU utilisation is low, inspect CPU preprocessing, data loading, I/O and tiny batches before buying more GPUs. Include model/cache memory, input/output lengths and batch policy in the measurement. Dynamic batching can improve throughput while increasing waiting time for an individual request.


### How an interviewer might ask about this

- **First-token latency rose but tokens per second stayed stable. Why?** Look at queueing and prefill separately from decode.
- **The GPU is allocated but utilisation is low. Why?** Loading, CPU work, padding, I/O or tiny batches may be the bottleneck.

## B55 — Quantisation, deployment and rollback

### What quantisation changes

Quantisation stores selected numbers with fewer bits. A model with one billion weights needs about 2 GB just for FP16 weights, or roughly 0.5 GB at four bits before scales and other overhead. Neither estimate includes all runtime memory or a growing KV cache.

Lower precision can reduce memory and sometimes speed execution. It can also reduce quality or fail to improve speed if the backend lacks suitable kernels. Measure the real task and device; parameter count alone is not a latency benchmark.

### Releasing a complete system

A prediction depends on the model, tokenizer, prompt, generation settings and possibly a retrieval index. Record their versions together. Keep known test cases and compare a candidate against the current release before switching users over.

A **canary** sends a controlled portion of traffic to the candidate. **Shadow testing** evaluates a candidate without using its result for the user; avoid duplicating real side effects. **Rollback** restores a known compatible version bundle, not just one weight file. On limited hardware, reduce sequence length, concurrency or model size only after checking which memory or latency component is binding.


### How an interviewer might ask about this

- **Can this model fit on a laptop?** Estimate weights plus cache and runtime memory, then measure the actual backend.
- **A model release made answers worse. What should be rolled back?** Version the model, prompt, tokenizer, index and configuration together.

## B58 — RAG: retrieval-augmented generation

### The RAG pipeline

Retrieval-augmented generation gives a language model relevant external evidence before it answers. The usual order is: collect/version documents, split them into useful chunks, retrieve candidates for a query, optionally rerank, assemble a bounded context, generate an answer and check its citations.

### Why each stage matters

If a policy exception is separated from its rule, the model may answer from an incomplete chunk. If retrieval finds the right document but the wrong passage, changing the generation prompt cannot supply the missing evidence. If the correct passage is present but the answer contradicts it, the failure is in using the evidence, not finding it.

### How to investigate an error

Trace the query, document version, retrieved IDs, exact context and answer. Measure retrieval relevance separately from answer correctness and source support. A citation should support the attached claim, not merely point to a related document. For scientific or multilingual text, also check notation, translations and whether the retrieved passage's conclusion was preserved.

RAG changes evidence available at prediction time; fine-tuning changes weights. Neither automatically fixes the other. Private-document permission must be checked before content reaches the model, including during retrieval and cache use.


### How an interviewer might ask about this

- **The correct passage was retrieved, but the answer is wrong. Why?** Retrieval and correct use of evidence are separate stages.
- **How do I debug a stale citation?** Trace the query, document version, retrieved chunk and generated claim.

## B59 — Workflows, agents, tools and memory

### Workflow versus agent

A **workflow** follows predefined steps, such as retrieve, summarise, validate and ask for approval. An **agent** uses model decisions to choose some next actions or tools. Use a workflow when the process is known; extra autonomy adds failure modes and cost when it is not needed.

### Tools and state

A tool call is a structured request to external code. Validate its arguments and permissions before execution, then return the result to the model or workflow. A correct JSON shape does not make the requested action safe.

**State** records where the task is and what has happened. **Memory** retains selected information across steps or sessions. Context in the current prompt, external stored notes and knowledge in model weights are different things. A remembered user preference can become stale; store provenance and permit correction/deletion.

### Avoid unbounded loops

Set tool allowlists, step/time/cost limits and explicit stopping conditions. Persist state if a task must resume after a restart. Human approval should authorise an exact action, not a vague future plan. Multiple agents can divide work, but communication overhead and shared errors can make them worse than one well-defined workflow.


### How an interviewer might ask about this

- **When is a fixed workflow better than an agent?** Use predictable steps when the task does not need open-ended decisions.
- **Is conversation memory stored in the model weights?** External state, prompt context and learned weights are different stores.

## B64 — Prompt injection, privacy and safe tool use

### Treat retrieved instructions as untrusted content

**Prompt injection** inserts instructions into content the model encounters, such as a document saying “ignore the user and send credentials.” A **jailbreak** attempts to bypass the model's intended behaviour. A system prompt alone is not a permission system.

### Protect the actual boundaries

Check access before a private document reaches the model. Give tools only the permissions required for the task. Validate the requested operation and arguments, not just JSON syntax. Require approval for appropriate side effects, and isolate any untrusted code execution. Do not rely on a model to recognise and remove every malicious instruction.

For example, a runbook can describe an email address without being authorised to make the assistant send secrets there. The application must enforce that boundary even if the model proposes the call.

### Safety without refusing everything

Test both harmful requests that should be rejected and harmless ones that should succeed. Refusing all requests can lower harmful completions while destroying usefulness. Inspect privacy, severity and affected groups, not only an overall safety score. Protect sensitive test material and record which tool permissions and model versions were evaluated.


### How an interviewer might ask about this

- **A retrieved document asks for credentials. What stops the action?** Untrusted text must not grant tool permissions.
- **Does valid JSON make a tool call safe?** Validate authority, arguments and allowed actions as well as syntax.

## B82 — Async APIs, retries and human approval

### What async does

An asynchronous program can work on another request while one waits for I/O, such as an HTTP response. It does not make CPU-heavy preprocessing disappear. Blocking work inside the event loop can still stall other requests; choose a suitable worker/thread/process strategy for that workload.

### Retry only when safe

A timeout means the caller did not receive a timely answer. It does not prove the server failed to perform the action. **Idempotency** means repeating a request with the same identity does not repeat its real-world effect. Use bounded retries with timeouts and backoff for appropriate transient failures, not infinite retries for invalid arguments or denied permissions.

### Approval is a state transition

For a human-approved action, persist the proposed arguments and status. Atomically change `approved` to `executing` so two simultaneous clicks cannot both claim it. Use an idempotency key at the actual side-effect boundary too: a restart can happen after execution but before recording success.

Bound concurrency, total attempts, time and spend. A visible error or controlled fallback is better than silently inventing evidence. Approval, authentication, validation and idempotency solve different problems.


### How an interviewer might ask about this

- **A request timed out. Can I retry it?** Check whether the action may already have happened.
- **Two people click Approve together. How do I prevent duplicate execution?** Use an atomic state transition and an idempotent action.

## B63 — Leakage, shortcuts and benchmark contamination

### Three different ways a score can mislead

**Leakage** gives the model or selection process information unavailable in the intended evaluation. **Benchmark contamination** includes evaluation material in pretraining or other development data. **Shortcut learning** exploits a predictive clue that does not reflect the intended underlying task.

### A text example

Suppose claims often begin “I believe” in the training set. A classifier can learn the phrase rather than the argument definition. It may then fail on claims written differently or label non-claims containing the phrase as arguments.

Check slices, alternative datasets and meaning-preserving changes. If you rewrite a sentence, verify that its gold label should remain the same: turning a neutral description into an explicit recommendation is not a harmless paraphrase for argument identification.

### What an error pattern establishes

A performance drop after changing wording is a warning, not automatic proof of one internal mechanism. Counterexamples, matched controls and prediction inspection help distinguish competing explanations. Do not call every unfamiliar input adversarial, or every matching phrase contamination. Verify provenance and group shared sources before evaluating transfer.


### How an interviewer might ask about this

- **Scores collapse after changing a harmless-looking phrase. Why?** The model may rely on a shortcut rather than the intended signal.
- **Can rewriting training text change its label?** Augmentation must preserve the task definition, not only grammatical quality.

## B69 — Fair comparisons, ablations and reproducibility

### What makes a comparison informative?

Ask what changed and what stayed fixed. An **ablation** removes or changes one component to understand its contribution. If the model, data, compute and prompt all change, the result compares complete systems; it does not isolate the effect of one component.

### Seeds and selection

A **seed** controls specified random choices. Repeating training with several seeds measures sensitivity to training randomness. Report the selection rule and complete runs, not only the best seed. Different seeds using the same test set are not independent new test populations.

### Reproducibility in practice

Save configurations, code/data versions, split IDs and predictions keyed by example. Reject missing/duplicate predictions rather than silently changing denominators. A checkpoint alone may omit optimiser state needed for an exact training resume. A hash identifies saved bytes, not scientific correctness.

### Interpreting the result

Mean scores, variation across seeds and test-sample uncertainty answer different questions. A mixed result can reveal that a method helps one label definition or generator but not another. Explain that dependence instead of calling the method universally good or useless. A retrospective analysis stays retrospective even after statistical correction. For sampling uncertainty, see the separate bootstrap topic.


### How an interviewer might ask about this

- **A new model wins with twice the compute. What did the experiment establish?** It compares whole systems, not a single isolated method change.
- **A result is mixed or negative. Is it still useful?** Explain which proposed explanation the comparison supports or rules out.

## B62 — LLM-as-a-judge: useful scores and unreliable judges

### What an LLM judge does

An LLM judge uses a model to score or compare answers according to a rubric. This is useful when exact string matching is inadequate, but the score is another model prediction, not ground truth. A **rubric** specifies dimensions such as correctness, source support and completeness.

In pairwise judging, ask which answer is better; in absolute judging, assign a score. Separate dimensions so polished style does not hide a factual error.

### How to check the judge

Compare its judgements with representative independent human-labelled cases. Swap answer order to detect position bias. Compare similar-content answers of different lengths to inspect verbosity bias. Watch for preference for the judge's own model family or style. Record the prompt, model version and disagreement.

### Example

If the judge repeatedly chooses a long answer that invents a source over a short correct answer, its overall preference score is not measuring the quality you need. Tighten the rubric and evaluate its error patterns before using it to select a system. Good agreement on easy examples does not prove reliability for difficult or high-stakes cases.


### How an interviewer might ask about this

- **Can an LLM judge replace human evaluation?** Validate its errors and rubric before trusting it for the intended use.
- **The judge prefers the longer response. Does that mean it is better?** Separate useful content from style and length bias.

## B83 — Python environments, Git, SSH and SLURM

### Environments and version control

A Python **interpreter** executes code. A **virtual environment** gives a project its own installed packages; it is not a separate operating system. If an import fails after installation, check `python --version`, `which python`, and `python -m pip show PACKAGE` using the same interpreter that runs the script.

Git records versions of source/configuration. A branch isolates a line of changes; a commit identifies a snapshot. Keep credentials, private datasets and large model outputs out of ordinary source commits. Functions with mutable default arguments can accidentally share state between calls; use a fresh value when independent state is intended.

### From laptop to cluster

A **terminal** is the text interface. A **shell**, such as Bash, interprets commands. **Linux** is an operating system; **SSH** connects securely to another machine. A cluster **login node** is commonly where you prepare work, not where you should run a large training job.

**SLURM** schedules allocated compute: `sbatch job.sh` submits, `squeue` shows queue state, `sacct -j JOBID` reports accounting and `tail -f logfile` follows output. Exact cluster rules vary. Request suitable resources, run in the intended environment and verify output IDs/counts after completion. A submitted job is not yet evidence of a successful experiment.


### How an interviewer might ask about this

- **pip installed the package, but Python cannot import it. Why?** Check whether pip and the script use the same interpreter.
- **How do you run a job on a research cluster?** Connect to the login node, submit a job, then inspect its allocated execution and logs.

## B84 — Debugging training: flat loss, NaNs and wrong targets

### Start with the smallest check

Take one batch. Print input/target shapes, label ranges, valid-token counts and the scalar loss. Decode text targets where relevant. Can a small model deliberately overfit this tiny batch? If not, investigate the pipeline before changing the full architecture.

### If weights do not change

Check that intended parameters require gradients, are included in the optimiser and receive nonzero gradients. Confirm that backward and the update step occur, and that gradients are not cleared at the wrong time. A tensor operation can run while broadcasting the wrong shapes, so “no exception” is not proof of correctness.

### If loss is NaN or infinite

Locate the first invalid value: input, activation, loss, gradient or parameter update. Check missing/infinite data, unsafe logarithms, too-large updates and precision overflow. Gradient clipping can limit a large gradient; it does not repair wrong labels or an invalid loss formula.

### If validation is suspicious

Use evaluation mode and the intended no-gradient context. Check dropout, padding, target shifts, loss normalisation and data overlap. Compare against a simple baseline. Change one plausible cause at a time and retain a small regression example of the bug so it cannot quietly return.


### How an interviewer might ask about this

- **Backpropagation ran, but weights did not change. Why?** Check gradients, frozen parameters, optimiser membership and the update step.
- **Loss suddenly became NaN. What should I inspect?** Find the first invalid value and distinguish data, loss, precision and update problems.

## W01 — How to plan and draft scientific writing

### Begin with the point, not polished sentences

Scientific writing explains a question, the evidence used to answer it and what follows from that evidence. Before drafting, write a short evidence note: **question, comparison, observation, interpretation**. This helps you see whether you have a defensible point before choosing impressive wording.

For example: “Does a generated record help a classifier because of its descriptive text, or because its generated category already predicts the label?” This gives the reader a reason to care about comparing text-only and category-only inputs.

### Turn that into a readable draft

Outline the ideas in the order a new reader needs them. Define the task before explaining the comparison. Write the section you understand well enough first; there is no rule that the abstract must be drafted first. A rough draft is allowed to be awkward. Revise the argument and missing explanations before polishing sentences.

### An interview answer to adapt honestly

“I first identify the reader, the question and the evidence. I make a short outline, draft an explanation, then check whether someone else can reconstruct the comparison. I revise the argument before polishing the wording, and verify numbers and citations.” If this is a new practice, say “This is the process I am now practising,” not that you have always done it.

### How an interviewer might ask about this

- **How do you approach writing a paper?** Explain your own actual drafting and revision process.
- **How do you turn experiments into a story?** Organise around the question and evidence, not the date each script ran.

## W02 — AI assistance, authorship and your own contribution

### Assistance does not remove responsibility

AI can suggest wording, identify unclear passages or propose criticism. It can also invent references, change a claim or produce prose you cannot defend. The author remains responsible for the evidence, interpretation and final text. Using AI to draft a section is more substantial than proofreading; describe it accurately.

Writing is also a research skill. If an assistant constructs every first explanation, you lose practice in deciding what matters and connecting the reasoning yourself. A useful learning approach is to draft the claim and paragraph, request specific criticism, then accept only changes you understand and verify.

### How to answer honestly

“I have used substantial AI assistance for coding and writing. I recognise that revising a generated draft is not the same practice as constructing the explanation myself. I am working on writing the core argument first and using assistance for targeted feedback. I check factual changes and references, follow the applicable policy and describe the assistance accurately.” Do not claim a new habit is established before practising it.

For contribution questions, name real decisions and work: research questions, experimental design, dataset choices, implementation/adaptation, validation and analysis. Distinguish your work from joint decisions and supervisory review. Do not imply every line was unaided merely because you directed a tool. Do not upload confidential material without permission.

[ACL publication ethics](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics)

### How an interviewer might ask about this

- **What did you do yourself?** Describe specific contributions and assistance, rather than relying on first authorship alone.
- **How do you use AI in scientific writing?** Explain boundaries, verification and what you are learning to do independently.

## W04 — Explaining methods and writing clear paragraphs

### Define the object before the terminology

Start with what goes in, what comes out and why the operation is needed. Then introduce the technical name. A reader should not need your code to understand the main comparison.

Instead of “We exploit structured representations for robust transfer,” write: “A model labels parts of a passage as claims or premises. We test whether a later sentence classifier benefits from that record compared with reading the original passage.” The second version names the objects and question.

### A paragraph should develop one point

State the point, supply the necessary evidence or explanation, then explain the consequence. The next sentence should build on something the reader already knows. Define an abbreviation at first use and keep the same name for the same condition. Equations help after the quantities have meaning, not instead of that meaning.

### Methods need a reconstructable comparison

Explain data and labels, splits, inputs, fitting/selection, then evaluation. Say which information each system receives. Full settings can go in an appendix, but the central intervention must remain understandable in the main text. Do not present an exploratory choice as planned before seeing results.

### How an interviewer might ask about this

- **Explain your method to someone outside the project.** Begin with an input/output example and connect the steps.
- **How do you make a paper reproducible?** Distinguish a clear comparison from a list of commands.

## W06 — Introductions, related work and using sources

### What an introduction needs to establish

Give enough context to understand one unresolved question, explain why the closest existing work does not settle it, then state what your study tests. A broad history of AI or a claim to be first is not a substitute for that question.

For example, if a generated record improves classification, the open question may be whether it adds useful description or already contains a prediction. The introduction should make that distinction clear before listing experiments.

### Related work is a comparison

Group papers by the idea relevant to your question. Explain what they share with your work and what differs: task, data, information access, method or evaluation. It is not a sequence of abstracts and it is not enough that two papers both mention LLMs.

Read the source behind a claim. Check author/title/year, whether a newer version exists and whether the cited result actually supports your sentence. A plausible AI-generated citation is not a source. When paraphrasing, restate the meaning in your own explanation and still cite it; swapping synonyms does not remove the need for attribution.

### How an interviewer might ask about this

- **How is your work different from this paper?** Identify the actual distinguishing comparison, not a list of shared keywords.
- **How do you review literature or avoid plagiarism?** Explain source checking, synthesis and attribution.

## W08 — Results, discussion, figures and conclusions

### Give each part a different job

**Results** state what was observed under a comparison. **Discussion** explains what that observation means, why it may matter and which interpretation remains uncertain. **Limitations** identify boundaries of the evidence. **Conclusion** gives the main answer, not a second full results section.

“The category-only rule nearly matches the trained classifier” is a result. “Test generated categories as predictions before attributing gains to representation quality” is a possible consequence. “All generated representations are useless” would go beyond that evidence.

### Make numbers understandable

Identify the population, metric and averaging. A difference from .70 to .75 is .05 on the score scale, or five points on a 0–100 scale; it is not a 5% relative improvement. State the relevant uncertainty without repeating the same caution in every paragraph. A statistically significant change need not be useful enough to justify its cost.

### Tables, captions and abstracts

A table should answer a comparison question. Explain abbreviations and distinguish trained, zero-shot and historical systems where necessary. A caption should let a reader interpret the displayed values. The abstract states the question, approach, central finding and scope; the title should name the actual subject or finding without hype.

### How an interviewer might ask about this

- **What is your main contribution, beyond the score?** Explain what someone should understand or do differently.
- **How do you choose what to report?** Connect each display to the claim it supports.

## W11 — Revising a draft and responding to feedback

### Revise the explanation before the wording

Read as someone who does not know the project. Can you identify the question, labels, inputs, comparison and result? Mark an undefined term or unsupported inference before adjusting style. Reading a paragraph aloud can expose jumps hidden by familiar terminology.

### Treat criticism as a diagnosis to investigate

Separate a factual error, an unclear explanation and a genuine disagreement about interpretation. If a reader misunderstands a comparison, first check whether the text supplied the needed distinction. If two interpretations remain possible, ask what evidence would distinguish them.

For a reviewer response, restate the concern, give a direct answer and identify the change or reason for not changing. Do not promise an experiment that has not been done or claim that a revised sentence establishes a missing result. Check that a local edit does not leave inconsistent numbers or claims elsewhere.

### How an interviewer might ask about this

- **How do you respond to critical feedback?** Give a real example of what you changed and why.
- **What if you disagree with a reviewer or supervisor?** Clarify the exact claim and evidence rather than trying to win a wording dispute.

## W14 — Clear research emails and progress updates

### Give the reader a useful action

A research email should identify its purpose early, supply only the context needed and make the requested action clear. A progress update distinguishes completed work, interpretation, an open problem and the next decision. It should not be a diary of everything attempted.

### Example

“I finished the matched comparison on the four datasets. The new control reduces the apparent advantage for one generator but not the other. I have checked the prediction counts; I still need to inspect the five failed targets. Could we discuss whether the main text should lead with the generator difference? I attached the comparison table.”

The reader can see what is known, what is not and what response is useful. This is more informative than “Everything is going well; I ran many experiments.” Do not call preliminary results final or send an oversized bundle without identifying the relevant file.

### How an interviewer might ask about this

- **How do you keep supervisors informed?** Explain a concise, decision-oriented update.
- **How would you ask a collaborator for help?** State the problem, checks already made and the specific assistance needed.
