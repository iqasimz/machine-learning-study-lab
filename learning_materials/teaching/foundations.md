# Shared foundations

## P2. Vectors, matrices and gradients

A model needs a numerical description of each example. If a flower is described by length and width, its two measurements form a **vector**. Put several flowers in rows and the measurements form a **matrix**. The shape `[batch, features]` records which axis represents examples and which represents measurements; it is not just a storage detail.

### From measurements to one prediction

Take one input row $x=[2,1]$, weights $w=[1,-1]$, and target $y=4$. The measurements and target are supplied data. The weights are adjustable model parameters. A dot product multiplies matching entries and adds them: $\hat y=2(1)+1(-1)=1$. A separate intercept, if present, would add a constant. Here it is zero.

For squared loss $L=(\hat y-y)^2$, this prediction gives $L=9$. The **residual** is $1-4=-3$; squaring removes its sign. A derivative asks how a small change in a parameter changes the loss, with the other values held fixed. The chain rule follows the two effects: the first weight changes the prediction at rate 2, and the prediction changes squared loss at rate $2(1-4)=-6$. Their product gives $\partial L/\partial w_1=-12$.

The negative sign tells us that a sufficiently small increase in this weight lowers the loss locally. It does not specify the step size. Gradient descent with learning rate 0.1 changes this weight to $1-0.1(-12)=2.2$. Keeping the other weight fixed gives a new prediction of 3.4 and loss 0.36. An excessively large update can overshoot; a gradient is a local slope, not a guarantee for an arbitrary jump.

### The same operation for many examples

With $X$ shaped $[B,d]$ and $W$ shaped $[d,h]$, the product $Y=XW$ has shape $[B,h]$. Each row is processed with the **same** weights. The shared dimension $d$ is summed over. Changing batch size changes the number of examples, not the number of learned weights.

The matrix-gradient note below extends this scalar calculation. A Jacobian records output-to-input sensitivities; reverse-mode differentiation propagates the loss's sensitivity without constructing a huge full-network Jacobian. A Hessian describes how gradients change. Neither is needed to guess a weight update: start with the forward values, loss, local slope, and an actual checked step.

**Check:** doubling a feature's numerical unit while keeping its weight fixed changes the prediction. To represent the same function, divide that feature's weight by two.

Sources: [D2L: linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html), [D2L: calculus](https://d2l.ai/chapter_preliminaries/calculus.html), [PyTorch: autograd mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html).

## P3. Probability, uncertainty and statistics

Two questions often get called uncertainty. “How much probability does this model assign to spam?” concerns one prediction. “How accurately have we estimated this model's success rate?” concerns evidence from a sample. A model can be confidently wrong on individual messages while a large test set estimates that failure rate precisely. We need different mathematical objects for the two questions.

### Outcomes, events and random variables

Imagine drawing one message from a stated population. The message is the **outcome** of this random experiment. An **event** is a set of outcomes, such as all spam messages. A **random variable** maps outcomes to values: let $Y=1$ for spam and $Y=0$ otherwise. The capital letter names a variable before observing a draw; a lower-case value such as $y=1$ names a realised value.

A **distribution** specifies how probability is allocated. For this binary variable, $P(Y=1)=0.2$ and $P(Y=0)=0.8$ form a probability mass function: nonnegative masses summing to one. For a continuous model, a **density** assigns probability through area, not the height of a single point. A uniform density on $[0,0.5]$ has height 2 and total area 1. Thus a density may exceed 1. Under that continuous model, the probability of any one exact point is zero; an interval can have positive probability. A recorded rounded measurement corresponds to an interval, not infinite measurement precision.

The **cumulative distribution function**, $F(a)=P(Y\leq a)$, accumulates probability up to a value. A percentile locates a point in this cumulative mass. This connects the distribution of resampled scores to the percentile endpoints used later.

### Conditioning changes the reference set

$P(A\mid B)$ means the probability of event A among outcomes satisfying B. It is not generally $P(B\mid A)$. Suppose a population of 10,000 people contains 100 with a condition. A test detects 90 of them and falsely flags 495 of the 9,900 without it. Among the 585 positive tests, only 90 have the condition: $90/585\approx0.154$. Sensitivity is $90/100=0.9$; it is a different fraction. These are illustrative counts, not medical guidance.

Bayes' rule formalises that reversal:

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.
\]

The prior $P(A)$ describes the base rate before this evidence. The likelihood $P(B\mid A)$ describes the evidence under the hypothesis. The denominator accounts for all ways the evidence can occur. For continuous observations, likelihood can use a density rather than a point probability. As a function of parameters with observed data fixed, likelihood is not a probability distribution over those parameters.

### Averages and variation

The **expectation** is a probability-weighted average. For a payoff that is 1 with probability 0.7 and 0 otherwise, $E[X]=0.7$. It need not be a possible single outcome. Variance averages squared distance from this mean: $\operatorname{Var}(X)=E[(X-E[X])^2]$. Here it is $0.7(0.3)^2+0.3(0.7)^2=0.21$; standard deviation is its square root and has the original units.

Covariance describes how two variables vary together. Correlation rescales covariance by their standard deviations when these are nonzero. It does not establish causation, and zero correlation does not generally imply independence. Independence means knowing one variable does not change the other's distribution. It justifies some convenient products, but must not be assumed for repeated measurements of the same person or document.

### A score is also a random quantity

A **population parameter** is a quantity of interest, such as a fixed model's accuracy on a specified population. An **estimator** is the rule used to estimate it from data. Its observed numerical output is an **estimate**. For accuracy, the estimator averages correctness indicators.

Imagine repeatedly collecting new test sets of 100 independent messages. Each set gives a different accuracy. The distribution of those possible estimates is the **sampling distribution**. It is not the distribution of class probabilities on one message. Its standard deviation is the **standard error**. For an independent sample mean, standard error is $\sigma/\sqrt n$, commonly estimated by replacing the unknown population standard deviation with a sample estimate. Four times as many independent cases roughly halves this error; four copies of each case do not.

For 82 correct predictions out of 100, accuracy is 0.82. An illustrative binomial plug-in standard error is $\sqrt{0.82(0.18)/100}\approx0.0384$. This number describes sampling variation under those assumptions, not a model's confidence in the next message. Approximation quality depends on sample size and where the parameter lies; a normal interval is not universally appropriate.

### Why bootstrap resampling is used

We do not have the population or many fresh datasets. The observed sample defines an **empirical distribution**: each of n observed rows receives mass $1/n$. The nonparametric bootstrap temporarily uses that distribution as a stand-in for the unknown population. Drawing n times from it is sampling the observed rows **with replacement**. Some rows repeat and others disappear. Recomputing the statistic approximates how it varies across possible samples, when this approximation is appropriate.

For comparing two fitted models on the same cases, sample the same indices for both. This **paired bootstrap** preserves which case each model found easy or hard. Recompute the whole metric on each draw, including F1's numerator and denominator. A sentence-level F1 is not the ingredient of corpus macro-F1. If messages are grouped by person, resample the independent people and keep their messages together. A row bootstrap would otherwise pretend correlated rows were independent.

A 95% **confidence-interval procedure** is designed to contain the fixed population quantity in about 95% of repeated suitable samples. This is not a claim that 95% of predictions are correct or, without a Bayesian model, a 95% probability about the fixed parameter after observing this one interval. Coverage is approximate and assumption-dependent.

The lab uses 200 percentile resamples for fixed-model accuracy or MAE, with duplicate-feature groups where supplied. It does not refit on every resample. It therefore omits training-seed, model-selection, and future-distribution uncertainty. More bootstrap repeats smooth the simulated distribution but do not repair unrepresentative data. A toy with four rows illustrates the operation, not a trustworthy real interval. The existing longer note below covers paired comparisons in more detail.

### Loss, entropy and statistical tests

For a known correct label, negative log-likelihood is $-\ln q(y)$. Assigning it probability 0.8 gives about 0.223 nats; assigning 0.2 gives about 1.609. **Entropy** instead averages surprise over the model's entire distribution: $-\sum_c q(c)\ln q(c)$. It requires no correct label. Low entropy can coexist with a confidently wrong answer. Perplexity exponentiates mean observed-token negative log-likelihood; it is not a probability of being correct.

A p-value asks how extreme a stated statistic would be under a null model and its assumptions. It is not the probability that the null is true, the effect size, or a replication probability. Testing many hypotheses increases opportunities for false rejection. With 20 independent true-null tests at level 0.05, the expected number of false rejections is 1, while the probability of at least one is $1-0.95^{20}\approx0.642$. Holm's procedure controls family-wise error for a declared family of valid tests: sort p-values, compare successively to $\alpha/m,\alpha/(m-1),\ldots$, and stop rejecting at the first failure. Report the comparison, effect, and uncertainty before using significance as a summary.

Sources: [D2L: probability and statistics](https://d2l.ai/chapter_preliminaries/probability.html), [Harvard Stat 110 and its probability textbook](https://stat110.hsites.harvard.edu/), [Think Stats: CDFs](https://allendowney.github.io/ThinkStats/chap04.html), [Think Stats: estimation](https://allendowney.github.io/ThinkStats/chap08.html), [Think Stats: hypothesis testing](https://allendowney.github.io/ThinkStats/chap09.html), [SciPy: bootstrap methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [statsmodels: multiple testing](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## P4. Training, validation and generalization

Suppose a spam detector scores well because training spam often contains a particular sender name. It may fail when that name disappears. Generalisation is performance on new examples under a stated task and population; it is not simply a small training loss. First define what counts as spam and which information the detector may see at decision time.

### Three datasets have different jobs

Training data fit parameters. Development or validation data choose settings, preprocessing variants, checkpoints, and thresholds. A test set evaluates the frozen choice. Trying ten models and reporting the best development score overstates what an untouched evaluation would show, because selection benefited from that sample's noise. Cross-validation repeats fitting and validation splits; nested cross-validation places this selection inside an outer evaluation loop.

The unit held out should match the claim. For new patients, separate patients rather than visits. For next-month forecasting, train on the past and evaluate the future. Fit means, scalers, vocabularies, and feature selection on training data only. Otherwise evaluation information influences the fitted system: **leakage**. Even an unsupervised preprocessing step can leak.

### A changed population is a different question

If only observed feature frequencies change while the conditional label mechanism stays fixed, importance weighting may estimate a new population's average error, provided the original data cover its cases. If label definitions change, that assumption is not satisfied. A tweet and a formal essay may use similar words but follow different annotation policies. More data from the old policy do not identify the new policy automatically.

Compare against simple baselines on the same cases, then inspect disagreement and subgroup failures. Looking at test errors is useful for learning. Once those errors guide another choice, call the analysis exploratory or obtain a fresh final evaluation. The problem is not curiosity; it is claiming independence after using the answers.

Sources: [D2L: generalisation](https://d2l.ai/chapter_linear-regression/generalization.html), [scikit-learn: leakage and preprocessing](https://scikit-learn.org/1.7/common_pitfalls.html), [scikit-learn: cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html).

## P5. How models learn: loss and optimization

A model first makes a prediction using its current weights. A loss measures the error against a supplied target. Differentiation measures how the weights influence that loss. An optimizer then changes the weights. These are four separate operations: evaluating a loss does not itself learn anything.

### One complete update

Use the deliberately small model $\hat y=wx$ with $x=2$, target 4, and starting weight 1. Prediction is 2; squared loss is $(2-4)^2=4$. By the chain rule, the weight gradient is $2(wx-y)x=-8$. With learning rate $\eta=0.1$, gradient descent gives $w_{new}=w-\eta\nabla L=1.8$. The next prediction is 3.6 and loss is 0.16. If the learning rate were 1, the new weight would be 9, prediction 18, and loss 196. Following a downhill local slope with too large a step can make things worse.

### What a training loop repeats

A batch estimates a gradient from several examples; an epoch is a pass through the training set. Batch size affects gradient noise, memory, and the number of updates in an epoch. Learning rate sets update scale. Adam also uses gradient-history estimates, so its update is not just the current gradient times a scalar.

For classification, train on a suitable differentiable objective such as cross-entropy. A thresholded class decision discards probability information and does not supply the usual useful gradient. The training objective and reported metric need not be identical: lowering cross-entropy does not guarantee higher F1 at an unchanged threshold.

Validation checks whether improvement survives beyond fitted examples. Decreasing training error with rising validation error is a reason to investigate overfitting. Before adding regularisation, verify the data, labels, masks, and implementation. Regularisation cannot correct a target shifted to the wrong position.

Sources: [D2L: gradient descent](https://d2l.ai/chapter_optimization/gd.html), [D2L: backpropagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html), [PyTorch: optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

## P8. Evaluation metrics

A metric is a particular question asked of predictions and known answers. It is not a universal quality score. Decide what the output means and which mistakes matter before selecting the summary. The worked view below lets you inspect several task families; these are alternative measurements, not successive stages of a model.

### Classification: keep track of the denominator

Call the class being detected “positive,” such as spam. Suppose 100 messages contain 20 spam. A system flags 15 messages, of which 12 are spam. Then there are 12 True Positives, 3 False Positives, 8 False Negatives, and 77 True Negatives.

- Accuracy asks what fraction of all decisions were correct: $(12+77)/100=0.89$.
- Precision asks how many flagged cases were truly positive: $12/(12+3)=0.8$.
- Recall asks how many actual positives were found: $12/(12+8)=0.6$.
- F1 takes their harmonic mean: $2(12)/(2(12)+3+8)=24/35\approx0.686$.

An always-negative rule scores 80% accuracy but finds none of the spam. F1 ignores True Negatives, so it is not a summary of every mistake. A zero denominator needs an explicit policy; a library's returned zero is a convention, not evidence from absent cases. Changing the score threshold changes the confusion counts without retraining.

### Several classes, scores, and probabilities

For multiclass F1, compute a one-versus-rest F1 for each class. **Macro** averages these class scores equally. **Weighted** averages them by observed class support. **Micro** pools the counts before taking the ratio. On single-label classification over all classes, micro-F1 equals accuracy; this identity does not extend unchanged to multilabel tasks. Balanced accuracy averages per-class recall, not per-class F1.

ROC curves compare True Positive rate against False Positive rate across thresholds. AUC evaluates ranking, not the quality of probabilities or any chosen cutoff. Precision–recall summaries put positive detections in focus and depend on prevalence. Average precision is a specific discrete summary, not automatically every implementation's trapezoidal PR area. Log loss and Brier score instead evaluate probability forecasts. Calibration asks whether stated probabilities correspond to observed frequencies; discrimination asks whether positives rank above negatives. A system can be good at one and poor at the other.

### Numerical predictions and other output units

For targets 3 and 7 with predictions 5 and 5, absolute errors are 2 and 2. MAE is 2, MSE is 4, and RMSE is 2. MAE and RMSE have target units; MSE has squared units. Standard $R^2=1-\sum(y-\hat y)^2/\sum(y-\bar y_{eval})^2$ compares to the evaluated targets' mean in its denominator. It can be negative. That mean is not a deployable training-fitted baseline, and constant evaluated targets require special handling.

For language models, average negative log-probability over the declared scored tokens; padding and ignored prompt tokens do not enter that denominator. Perplexity depends on the tokenizer and target construction. Retrieval metrics score ranked relevant items; sequence tagging may require exact entity boundaries rather than merely correct token tags. State the evaluated unit before comparing numbers. A good aggregate can conceal the cases that matter most.

Sources: [scikit-learn: classification and regression metrics](https://scikit-learn.org/1.7/modules/model_evaluation.html), [scikit-learn: R²](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.r2_score.html), [SLP: logistic regression and evaluation](https://web.stanford.edu/~jurafsky/slp3/4.pdf).
