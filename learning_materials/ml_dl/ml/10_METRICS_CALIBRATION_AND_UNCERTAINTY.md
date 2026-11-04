# ML10. Probability calibration and decision costs

Two models can make the same yes/no decisions but assign different probabilities. Calibration asks whether those probabilities agree with observed frequencies: among predictions made with about 80% confidence, roughly 80% should be correct in an appropriate evaluation population. Decision costs ask a separate question: how harmful are false alarms and missed cases?

The four positive-class probabilities are [0.9, 0.6, 0.4, 0.1]. Changing their sharpness leaves their order and 0.5-threshold decisions unchanged, but changes probability errors. A separate cost example then derives a decision threshold.

The shared metrics lesson owns precision, recall and F1. Here we compare probability quality, a reliability estimate and a cost-based action rule; these are related questions, not one calculation feeding the next.

[course map](../COURSE_MAP.md). Required background: ML05.

## 1. Calibration and error consequences

**Key terms.** A **true positive** is a correct positive decision; a **false positive** is an incorrect one; a **false negative** misses a positive. **Calibration** compares confidence with correctness; **coverage** is the fraction answered or the inclusion rate of a prediction set, depending on context.

**Process.** Freeze predictions on the intended evaluation units, calculate metrics and inspect error groups. Estimate uncertainty by an appropriate paired or grouped procedure, then compare effects with cost and decision requirements rather than selecting a flattering metric.

Accuracy can hide failure on a rare class. Precision/recall measure a declared positive class; macro-F1 averages per-class F1 equally. AUROC and average precision describe ranking under different constructions, not correctness of predicted probabilities. Brier score and log loss assess probability predictions. Regression needs an explicit unit and objective: MAE, RMSE and R² are not interchangeable.

Thresholds and calibrators must be fitted or selected without the final test. A paired bootstrap resamples the same independent units for both systems and recalculates the entire metric. Training-seed variation is another uncertainty source. Multiple comparisons and repeated test peeking need their own design controls.

## 2. A worked calculation

**Symbols and assumptions.** TP/FP/FN are counts, precision=TP/(TP+FP), recall=TP/(TP+FN), and F1=2PR/(P+R) where defined. Binary Brier score averages (p−y)².

With TP=18, FP=2 and FN=12, precision=0.9, recall=0.6 and F1=0.72. For probabilities [0.8,0.3] and labels [1,0], Brier score=(0.2²+0.3²)/2=0.065. This two-case number is a hand example, not sufficient evidence of calibration.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Prediction intervals and refusal policies

Confidence intervals for a fixed-model average are not prediction intervals for a future case. Conformal prediction uses calibration scores to build sets/intervals with marginal coverage under assumptions such as exchangeability; it does not promise conditional coverage for every subgroup or arbitrary shifts. Accuracy, abstention error and answer coverage should be reported together when refusing cases.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: separate confidence from test uncertainty

Compare per-class confusion counts and proper scoring on Wine or Iris. For a high-cost decision, specify the cost trade-off rather than automatically use threshold 0.5. The course does not validate medical or legal deployment.

**Check.** Why is a single accuracy standard deviation across three seeds not a full confidence interval for deployment performance?

**Answer.** It describes limited training randomness on one protocol. It omits many possible changes in sampled cases, future domains, labels and selection. State precisely which source of variation was measured.

## 5. Two different meanings of uncertainty

**Key terms.** A confidence interval describes uncertainty about an estimated quantity under a sampling procedure. A prediction set concerns possible outcomes for a new example. Neither is simply the model's largest probability.

**Process.** State the uncertain quantity, identify independent resampling units and choose a procedure appropriate to that quantity. Report assumptions and failure modes alongside the numerical interval.

For an accuracy bootstrap, sample n test-row indices with replacement, score those sampled rows using fixed saved predictions and repeat. The 2.5th and 97.5th percentiles give a percentile interval. Repeated rows are intentional. This measures conditional test-sample variability, not sensitivity to retraining, model selection, dataset shift or unseen user groups. Compare two systems with the same sampled indices, rather than two unrelated resamples.

### Split conformal prediction

Fit a predictor on training data and compute nonconformity scores on a separate calibration set of m examples. With α=0.1, use the sorted score at rank ceil((m+1)(1−α)); for m=19 that is rank 18. For absolute-residual scores, predict an interval around the new prediction using that quantile. Exchangeability supports marginal coverage, not a guarantee for each subgroup or shifted future distribution. Extreme α can require an infinite boundary under the standard finite-sample convention.

The current lab implements the conditional row-bootstrap illustration. Conformal calibration is taught here but is not an executable switch in this release.

## Sources

- [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [Angelopoulos and Bates, conformal prediction introduction](https://arxiv.org/abs/2107.07511).
