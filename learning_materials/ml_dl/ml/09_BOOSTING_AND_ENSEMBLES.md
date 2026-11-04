# ML09. Gradient boosting

Gradient boosting builds a predictor in stages. Each new small model tries to correct errors made by the current combined model. For squared-error regression, the correction can be trained on the remaining differences between targets and predictions. This differs from a random forest, whose trees are trained without sequentially correcting a running prediction.

One example has target 3 and current prediction 1, leaving residual 2. We assume a small correction model returns 2 at this point. The learning-rate control changes how much of that correction is added; this demonstration does not fit the correction model.

A forest combines varied trees; boosting combines a sequence of corrections. Smaller corrections often need more stages, so learning rate and stage count should be considered together.

[course map](../COURSE_MAP.md). Required background: ML08.

## 1. How ensemble methods combine models

**Key terms.** **Boosting** adds models sequentially to improve an objective; a **weak learner** is one component; **shrinkage** scales its contribution. **Voting** combines predictions; **stacking** learns a combining model from out-of-fold predictions.

**Process.** Fit an initial prediction, calculate how current predictions fail, fit another learner to an appropriate correction signal and add a scaled contribution. Select the stopping point on development data, then assess the complete fitted ensemble.

For squared loss, fitting residuals gives an intuitive boosting example. Other losses use their own gradients and sometimes curvature; do not describe all boosting as literally predicting numeric residuals. Histogram methods bin feature values to reduce split-search cost. Missing-value handling, categorical support and constraints vary by implementation.

Random forests fit randomized trees largely independently; boosting uses a sequence whose later models depend on current errors. More boosting rounds can overfit or improve depending on rate and regularization. A small learning rate with many rounds is not the same computational budget as a large rate with few rounds.

## 2. A worked calculation

**Symbols and assumptions.** F_m is the ensemble after round m; h_m is a fitted correction; eta is shrinkage. The additive update is F_m(x)=F_(m−1)(x)+eta h_m(x).

For targets [3,7] and current predictions [2,5], residuals are [1,2]. Suppose a fitted correction predicts [1,2] and eta=0.1. New predictions are [2.1,5.2]; mean squared error drops from (1²+2²)/2=2.5 to (0.9²+1.8²)/2=2.025. This arithmetic assumes that particular correction; it is not a claim that every round helps validation.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Boosting settings and stacking

Control rounds together with learning rate, leaf count/depth, minimum leaf size and optional row subsampling. For stacking, train the combiner on predictions from base models that did not train on those same rows. Equal-weight averaging is a useful baseline before learning a complicated combiner.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare strong tabular baselines

Compare ridge, random forest and histogram gradient boosting on Wine Quality using the same splits and search budget. Save chosen configurations, not only a winner's score. If the boosted model wins tabular data, the later neural comparison must keep that baseline.

**Check.** Why is training a stacker on in-sample predictions optimistic?

**Answer.** Its inputs are unrealistically easy because base learners already saw those labels. Generate out-of-fold training predictions for the combiner; keep the final test separate.

## 5. A boosting update, not just a collection of trees

**Key terms.** A residual is observed minus predicted value in this squared-error example. A negative gradient is the correction direction for the chosen loss. Shrinkage scales the added learner.

**Process.** Start with a simple prediction. Compute the current loss gradient, fit a shallow learner to the correction and add a scaled version to the running prediction. Repeat, checking generalization rather than only reducing training loss.

For targets [1,3] and initial prediction [2,2], residuals are [−1,+1]. A tree reproducing these residuals, added at rate 0.1, produces [1.9,2.1]. Mean squared error falls from 1 to 0.81. This example is a squared-loss gradient step; logistic boosting uses a different gradient and second-order variants also use curvature.

### Distinguish the ensemble mechanisms

Bagging fits randomized models largely independently and averages them. Boosting is sequential. Stacking fits a combiner on out-of-fold base-model predictions; training that combiner on each model's in-sample predictions is a common leakage route.

The executable histogram booster uses a fixed round budget and disables its automatic internal early stopping. A smaller boosting rate may need more rounds; changing both is a budget-aware system comparison, not a one-variable ablation.

## Sources

- [scikit-learn ensembles](https://scikit-learn.org/stable/modules/ensemble.html).
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
