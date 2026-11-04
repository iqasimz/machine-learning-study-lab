# Machine learning

## ML01. Machine learning tasks and baselines

Machine learning fits a rule from examples rather than requiring every decision to be hand-written. Start by specifying the output: a class, a numerical value, a ranking, or a grouping. Supervised learning has supplied targets; clustering seeks structure without those class targets. Neither formulation automatically says what is useful to a person.

An input feature is information available when predicting. A target is the quantity being predicted. A future outcome accidentally included in the inputs makes the task artificially easy. Write down when the prediction is made before choosing columns.

A **baseline** is a deliberately simple comparison. On 100 examples with 90 negative labels, predicting negative always achieves 90% accuracy and zero positive recall. For numerical targets 2, 4, and 9, the constant training mean is 5; its squared errors are 9, 1, and 16. The training median 4 instead minimises absolute error among constants. Baselines expose whether a sophisticated model earns its complexity under the chosen loss.

Compare all candidates on the same task, split, and metric. A low score on a difficult held-out domain may reveal more than a near-perfect score on leaked data. The Model Lab runs actual fitted models; its numerical outcome is separate from the fixed examples used in these explanations.

Sources: [ISLP: book and chapter labs](https://www.statlearning.com/), [scikit-learn: dummy estimators](https://scikit-learn.org/1.7/modules/generated/sklearn.dummy.DummyClassifier.html).

## ML02. Features and preprocessing

A model sees the representation we supply, not the original meaning of a column. Consider age in years and income in currency units. A distance calculation can be dominated by income's larger numbers even when age matters more. **Standardisation** subtracts a learned training mean and divides by a learned training standard deviation. It changes units; it does not make the feature causally important.

For training values 2, 4, and 6, the mean is 4. Under population-variance scaling, variance is $8/3$ and standard deviation is about 1.633. A new value 8 becomes $(8-4)/1.633\approx2.449$. Do not recompute the mean using that new example. Robust scaling instead uses a median and an interquartile range, reducing sensitivity to extreme values.

Missing-value imputation also learns quantities. Categorical encoding establishes which categories are known and what an unseen category means. Unordered colours should not acquire an invented ordinal distance simply by numbering them 1, 2, and 3.

A pipeline fits each learned preprocessing step only on the current training fold, then applies the stored transformation to validation or test. Putting preprocessing inside the pipeline is what keeps cross-validation honest. Some tree methods need little scaling; nearest-neighbour and penalised linear methods are much more sensitive to it.

Sources: [scikit-learn: preprocessing](https://scikit-learn.org/1.7/modules/preprocessing.html), [scikit-learn: common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html).

## ML03. Linear regression

Linear regression predicts a numerical target using a weighted sum, $\hat y=w_1x_1+\cdots+w_dx_d+b$. A weight describes a slope while the other features remain fixed. Its magnitude depends on feature units; a large coefficient is not automatically a large real-world effect.

Two points, $(0,1)$ and $(2,5)$, define the line $\hat y=2x+1$. Add $(1,4)$: that same line predicts 3 and misses by 1. Least squares fits a compromise across all rows by minimising summed or averaged squared residuals. These reductions have the same unregularised minimiser, but their scale matters when a penalty or a learning rate is introduced.

For many features, matrix notation writes predictions as $Xw+b$. Least-squares solvers do not require explicitly computing a matrix inverse. If columns are dependent, multiple coefficient vectors may make identical predictions; regularisation can make the solution more stable. A linear model can also use nonlinear input features such as $x^2$: “linear” refers to how coefficients enter the model.

Inspect held-out residuals, not only training fit. Curved residual patterns suggest missing structure; unusually large errors deserve investigation. Neither a good fit nor a coefficient alone establishes causation.

Sources: [D2L: linear regression](https://d2l.ai/chapter_linear-regression/linear-regression.html), [scikit-learn: ordinary least squares](https://scikit-learn.org/1.7/modules/linear_model.html#ordinary-least-squares).

## ML04. Logistic regression and classification

A class decision is discrete, but training benefits from a smooth score. Binary logistic regression computes $s=w^Tx+b$ and maps it to a probability with $p=1/(1+e^{-s})$. At score 0, probability is 0.5; at score 2 it is about 0.881. The resulting decision boundary is still linear in the supplied features.

There are two branches after this probability. To make a decision, compare it with a chosen threshold. To train against a known label, calculate binary cross-entropy: $-y\ln p-(1-y)\ln(1-p)$. If the label is 1, changing p from 0.5 to 0.9 reduces the loss from about 0.693 to 0.105. Thresholding first would throw away this distinction.

A low threshold may catch more positives while increasing false alarms. Select it using validation data and relevant costs, not because 0.5 is universally right. Training weights and selecting a decision threshold solve different problems.

For one of K mutually exclusive classes, softmax normalises K logits into a categorical distribution. Independent sigmoid outputs instead allow multiple labels at once. Use the formulation matching the target, then check both discrimination and probability quality.

Sources: [SLP: logistic regression](https://web.stanford.edu/~jurafsky/slp3/4.pdf), [scikit-learn: logistic regression](https://scikit-learn.org/1.7/modules/linear_model.html#logistic-regression).

## ML05. Regularization and model selection

A flexible model can fit accidental details of a sample. Regularisation restricts the fit so those details have less influence. Ridge penalises squared coefficient size; lasso penalises absolute size and can set coefficients exactly to zero; elastic net combines both. The penalty strength is a choice about fit, not an extra observation.

Suppose two nearly duplicate features receive large opposite coefficients. Their training predictions can look sensible while small input changes cause instability. Ridge discourages such large coefficients. Increasing its strength generally sacrifices some training fit; held-out performance can improve or worsen, so validation must decide.

Compare a training curve with a validation curve. Poor performance on both can suggest an unsuitable representation, too much restriction, or a fitting bug. A large training–validation gap suggests sensitivity to the fitted sample. These are diagnostic patterns, not infallible labels.

Select penalty, depth, and model family inside the validation procedure. Report the selected configuration and the untouched evaluation. Trying many settings also consumes statistical evidence: a winning validation score is partly a result of selection. Scaling matters because a penalty on coefficients depends on feature units.

Sources: [scikit-learn: ridge, lasso and elastic net](https://scikit-learn.org/1.7/modules/linear_model.html#ridge-regression-and-classification), [D2L: weight decay](https://d2l.ai/chapter_linear-regression/weight-decay.html).

## ML06. Nearest neighbours, kernels and SVMs

Nearest-neighbour prediction asks which stored training cases resemble the new input. The similarity definition is part of the model. For a one-dimensional point 2.2 and training locations 1, 2, and 5, absolute distances are 1.2, 0.2, and 2.8. With k=1 the label at 2 decides; with k=3 all three labels vote. Increasing k smooths decisions but may mix distinct groups.

Euclidean distance sums squared numerical differences across features. Changing metres to millimetres multiplies a length difference by 1,000 and its squared contribution by a million. Training-fitted scaling can therefore change neighbours and predictions. It does not guarantee a higher score. Nearest-neighbour methods retain training examples and can be expensive at prediction time.

An **SVM** takes a different route: learn a separating decision surface with a margin. Support vectors are cases constraining this surface. A soft-margin penalty permits violations; the C parameter trades these penalties against the margin objective. A kernel computes a permitted inner-product similarity in an implicit feature space, enabling nonlinear boundaries. It is not an arbitrary similarity function with no mathematical restrictions.

Both methods depend on representation and scale, but “find nearby examples” and “fit a margin” are different learning rules. The worked view compares their mechanics rather than pretending one flows into the other.

Sources: [scikit-learn: nearest neighbours](https://scikit-learn.org/1.7/modules/neighbors.html), [scikit-learn: support vector machines](https://scikit-learn.org/1.7/modules/svm.html).

## ML07. Naive Bayes classification

Naive Bayes combines how common a class is with how likely the observed features are within that class. Its simplification is **conditional independence**: once the class is specified, the feature likelihood factors into separate contributions. Features need not be independent in the population. The assumption is often imperfect, especially for neighbouring words.

Consider 20 messages, ten spam and ten non-spam. “Prize” appears in eight spam and one non-spam message. Using only its presence as a binary feature, the two unnormalised class scores are $(10/20)(8/10)=0.4$ and $(10/20)(1/10)=0.05$. Divide each by their sum. The spam posterior is $0.4/0.45=8/9\approx0.889$, not 0.8. The larger score chooses a class; normalisation gives a probability distribution under the model.

With many features, log likelihoods avoid underflow from multiplying tiny numbers. Smoothing prevents an unseen training count from forcing an entire class score to zero, but requires a declared finite feature/vocabulary model.

Bernoulli naive Bayes models feature presence and absence. Multinomial naive Bayes models count contributions. Gaussian naive Bayes models continuous feature densities within classes; this is the variant available for numerical data in Model Lab. These are not interchangeable likelihoods, and a strong classifier can still have poorly calibrated probabilities.

Sources: [scikit-learn: naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html), [SLP: naive Bayes appendix](https://web.stanford.edu/~jurafsky/slp3/B.pdf).

## ML08. Decision trees and random forests

A decision tree asks a sequence of feature questions, such as whether a measurement is below a cutoff. Each split sends examples to smaller groups. A leaf stores a prediction based on the training labels that reached it. The model learns both the questions and their ordering.

For a group with half class A and half class B, Gini impurity is $1-(0.5^2+0.5^2)=0.5$. A split producing two pure groups has weighted child impurity 0. A pure but tiny leaf can fit noise, so maximum depth and minimum leaf size control how detailed the partition becomes. A greedy split is locally preferred; it is not guaranteed to produce the globally best tree.

A random forest fits many trees using resampled training cases and random feature subsets. Averaging reduces instability most effectively when trees do not make identical errors. This is why randomness helps an ensemble rather than merely making its individual trees stronger. More trees reduce Monte Carlo variability at increased cost; they do not fix shared bias or leakage.

Tree feature importance measures a particular use within a fitted model. Correlated features can share or distort this importance. It is not a causal attribution. Compare held-out performance, stability, and interpretation requirements before choosing one tree or a forest.

Sources: [scikit-learn: decision trees](https://scikit-learn.org/1.7/modules/tree.html), [scikit-learn: random forests](https://scikit-learn.org/1.7/modules/ensemble.html#forest).

## ML09. Gradient boosting

Boosting builds an additive model in stages. Rather than fitting many independent models and averaging them, it fits the next component to improve the current ensemble. For squared-error regression, the correction targets are residuals: target minus current prediction.

If the target is 10 and the current prediction is 6, the residual is 4. Suppose the next small tree predicts a correction of 3. With learning rate 0.1, the ensemble prediction becomes $6+0.1(3)=6.3$, not 9. The smaller step can make many-stage fitting less aggressive, at the cost of needing more stages. For other differentiable objectives, the corrections follow negative gradients rather than literal numerical residuals.

Tree depth controls the complexity of each correction. Learning rate, number of stages, and early stopping interact: more stages are not automatically better after validation performance stops improving. Histogram methods group feature values into bins to make split search more efficient; they are an implementation strategy, not a new definition of the target.

Compare boosting with a forest using the same split and meaningful budget. Forests primarily average variable fits; boosting sequentially corrects a fit. Neither method is universally preferable, and both can exploit a spurious feature.

Sources: [scikit-learn: gradient boosting](https://scikit-learn.org/1.7/modules/ensemble.html#gradient-boosting), [D2L: optimisation perspective](https://d2l.ai/chapter_optimization/optimization-intro.html).

## ML10. Probability calibration and decision costs

A probability forecast should mean something beyond the ordering of cases. For binary positive-class calibration, among comparable cases assigned probability 0.8, about 80% should be positive. For **top-label confidence calibration**, compare the largest predicted class probability with the frequency of that selected class being correct. These are related but different constructions; the lab's ECE summary uses top-label confidence bins.

Log loss strongly penalises tiny probabilities on realised labels. Brier score measures squared probability error. Calibration plots group predictions and compare forecasts with frequencies; bin choices and small samples affect the picture. A low binned calibration error does not certify every subgroup or future population.

Decision costs are another question. If a false alarm costs 1 and a missed positive costs 9, under calibrated binary probabilities and zero cost for correct decisions, predict positive when $p>1/(1+9)=0.1$. This follows by comparing expected costs: $1(1-p)$ versus $9p$. Different costs or unreliable probabilities change the reasoning.

Fit a calibration mapping on data not used to fit the base predictor, or use an appropriate cross-validation scheme. A strictly monotonic mapping preserves ranking but can change probability quality and decisions at a numerical threshold. Calibration on yesterday's population need not survive tomorrow's shift.

Sources: [scikit-learn: calibration](https://scikit-learn.org/1.7/modules/calibration.html), [scikit-learn: decision-threshold tuning](https://scikit-learn.org/1.7/modules/classification_threshold.html).

## ML11. Clustering and K-means

Clustering groups examples using an imposed notion of similarity; no supplied class label tells it which groups are meaningful. K-means chooses k centres, assigns each point to its nearest centre, then moves each centre to its assigned points' mean. Repeating these operations reduces or preserves its within-cluster squared-distance objective until the procedure stops.

For points 0, 1, and 10 with initial centres 0 and 10, the first assignments are {0,1} and {10}. Updated centres are 0.5 and 10. The squared-distance sum is $0.5^2+0.5^2+0=0.5$. Choosing more centres can reduce this training objective even when the extra groups have no useful meaning. Different initialisations can lead to different local solutions, but that must be demonstrated on a suitable example rather than asserted for every toy dataset.

K-means favours compact groups under its chosen distance and is sensitive to scale and outliers. Gaussian mixtures instead model overlapping components probabilistically. DBSCAN uses local density and can mark points as noise; its scale parameters also matter. A clustering score measures a property of the grouping, not proof that natural categories were discovered. Fit any preprocessing on the intended training population before interpreting a new point's assignment.

Sources: [scikit-learn: clustering methods](https://scikit-learn.org/1.7/modules/clustering.html), [scikit-learn: Gaussian mixtures](https://scikit-learn.org/1.7/modules/mixture.html).

## ML12. PCA, dimensionality reduction and anomalies

Principal component analysis finds orthogonal directions capturing variation in centred training data. Projecting onto fewer directions compresses the representation; projecting back gives a reconstruction. The first component is the direction of greatest variance, not necessarily the direction most predictive of a label.

Imagine centred points near the line $x_2=x_1$. A unit direction $v=[1,1]/\sqrt2$ maps a point $[2,2]$ to coordinate $x^Tv=2\sqrt2$. Reconstructing from this one coordinate gives $[2,2]$. A point $[2,0]$ projects to $\sqrt2$ and reconstructs as $[1,1]$, losing the component perpendicular to the retained line. Feature scaling changes which variations dominate this fit.

Reconstruction error can flag an unusual example relative to learned regularities, but a rare valid case can have large error and an abnormal case can lie along a common direction. Isolation forest takes a different approach by isolating examples through random partitions. Neither score becomes a calibrated probability of wrongdoing or failure without additional modelling and validation.

Choose dimensionality and anomaly thresholds using an appropriate held-out protocol. Explain whether the aim is compression, visualisation, denoising, or downstream prediction; the best representation may differ for each.

Sources: [scikit-learn: PCA](https://scikit-learn.org/1.7/modules/decomposition.html#pca), [scikit-learn: novelty and outlier detection](https://scikit-learn.org/1.7/modules/outlier_detection.html).

## ML13. Time-series forecasting

A forecast predicts a future value using information available at a stated time. Ordering matters: training on tomorrow and testing on yesterday answers the wrong deployment question. A rolling-origin evaluation repeatedly fits or updates using the past, then scores the next held-out period.

If the last three observed demands were 8, 10, and 12, a three-period moving average forecasts 10. A persistence baseline forecasts the latest value, 12. Neither knows the next observation. A rising trend or weekly seasonality can make these baselines fail differently, so compare against the relevant structure rather than a randomly shuffled split.

A lag is an earlier observed value used as a feature. A rolling statistic summarises a window of those values. It must stop at the prediction boundary; a centred rolling mean can accidentally use future observations. Missing values, delayed labels, and revisions to historical data also affect what was truly known at the time.

Forecast error depends on horizon: tomorrow and next month are different targets. Compare methods on matching origins and horizons, report errors in useful units, and treat forecast intervals as model-based uncertainty rather than a guarantee against structural change.

Sources: [Think Stats: time-series analysis](https://allendowney.github.io/ThinkStats/chap12.html), [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## ML14. Recommendation and sequential decisions

A recommender ranks eligible items for a user. In matrix factorisation, a user vector and item vector produce a score by dot product. A user $[1,2]$ and item $[3,1]$ give score 5. This is a compatibility score, not automatically a calibrated click probability. Unobserved interactions are not necessarily dislikes: the user may never have seen the item.

Evaluate the recommendation among a stated candidate set and time period. Randomly sampled easy negatives can make ranking look better than ranking a realistic catalogue. Include cold-start users/items if those are part of deployment.

A sequential decision problem adds consequences over time. An action changes what happens next, and a policy chooses actions using available state. A discounted return $r_0+\gamma r_1+\gamma^2r_2+\cdots$ values future rewards according to gamma. Rewards 1 now and 2 next with gamma 0.9 give return 2.8. A larger immediate reward need not maximise long-term return.

Recommendation and reinforcement learning can interact, but fitting item scores is not automatically training an RL agent. Logged feedback reflects past exposure and policy choices. Evaluation must consider this selection bias before claiming a new policy is better.

Sources: [D2L: matrix factorisation](https://d2l.ai/chapter_recommender-systems/mf.html), [D2L: value iteration](https://d2l.ai/chapter_reinforcement-learning/value-iter.html).

## ML15. Generalization and distribution shift

A model can perform well because a feature happens to agree with the label in its training environment. If that shortcut disappears, the learned rule may fail despite a high ordinary test score. First ask what changed: feature frequencies, class frequencies, the label definition, or what information is available.

Suppose group A has error 0.1 and group B error 0.4. In a population with 80% A and 20% B, average error is $0.8(0.1)+0.2(0.4)=0.16$. Changing proportions to 20% A and 80% B gives error 0.34 even if the model's within-group behaviour is unchanged. Reweighting the original observations can estimate that new mixture if the relevant groups are represented and their conditional behaviour stays stable.

If B's definition of a positive label changes, the old error rate is no longer the needed quantity. No mixture weighting alone repairs this. Similarly, unsupported regions of input space cannot be reliably estimated by enormous weights on unrelated examples.

Compare ordinary and shifted evaluations, subgroup errors, and simple controls. A confidence threshold may remove uncertain errors but keep confident shortcut-driven mistakes. The present worked view illustrates reweighting; it is not a measured shortcut-training experiment. Distinguishing an authored mechanism from an actual fitted result is part of understanding the evidence.

Sources: [D2L: environment and distribution shift](https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html), [Geirhos et al.: shortcut learning](https://arxiv.org/abs/2004.07780).

## ML16. Run and report an ML experiment

A useful experiment answers a specific question, such as whether a train-fitted scaler improves a fixed nearest-neighbour classifier on a specified split. “Try many models and keep the largest number” leaves the comparison and selection process unclear.

Write down the target, permitted inputs, split unit, baseline, candidate configurations, selection metric, and evaluation budget. Run preprocessing inside each training fit. Compare development results first, choose a configuration, then report its reserved test result. Save the configuration, seed, data identity, software versions, fit duration, and predictions so an apparent gain can be investigated.

For example, comparing unscaled and standardised k=5 neighbours should hold the rows and distance convention fixed. If several k values are tried as well, document that additional selection. A saved test score alone does not explain which choices produced it.

Read results in task units: how many important cases changed, whether the baseline was beaten, and what extra time or complexity was required. The lab's bootstrap interval conditions on one fitted model; it does not include the entire search process. Repeated training across seeds addresses a different variation source. Keep an unexpected negative result when it answers the question; it can reveal a useful limitation rather than a failed experiment.

Sources: [scikit-learn: pipelines](https://scikit-learn.org/1.7/modules/compose.html), [scikit-learn: model selection](https://scikit-learn.org/1.7/modules/grid_search.html).
