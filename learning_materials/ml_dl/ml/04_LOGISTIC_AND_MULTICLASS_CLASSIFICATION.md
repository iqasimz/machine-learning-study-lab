# ML04. Logistic regression and classification

Classification predicts a category rather than a numerical measurement. Binary logistic regression first calculates a weighted score, then maps it to a probability for one of two classes. A separate decision threshold turns that probability into a yes/no prediction.

Features [1, 2] and weights [0.2, 0.1] give score 0.4 and positive-class probability about 0.60. A threshold of 0.5 predicts positive; a threshold of 0.7 predicts negative, without changing the model.

Linear regression in the linear regression lesson predicts a number directly. Logistic regression adds a probability mapping and a classification loss. The threshold controls the action, while the loss can be used to fit the probabilities.

[course map](../COURSE_MAP.md). Required background: ML03, P3.

## 1. Binary, multiclass and multilabel tasks

**Key terms.** A **logit** is a raw score; **sigmoid** converts one score into a binary probability; **softmax** normalizes several class scores; a **threshold** turns a probability into a decision. **Multilabel** permits several simultaneous labels.

**Process.** Transform features into logits, calculate target log loss and fit weights. At prediction, output probabilities first and apply the chosen decision rule afterwards. Changing the threshold alone does not retrain the model.

Logistic regression has a linear boundary in its supplied feature space, not a linear probability. Binary and mutually exclusive multiclass predictions need different output shapes; multilabel tasks commonly use independent binary outputs. Stable loss functions take logits directly rather than rounded probabilities. In PyTorch, standard CrossEntropyLoss expects logits and class indices for the common hard-label case.

Positive class labels and metric averaging must be stated. A threshold chosen to catch more spam can also increase legitimate-message false alarms. Class weights alter fitting, while threshold changes alter decisions from already fitted scores. They can interact but are not substitutes.

## 2. A worked calculation

**Symbols and assumptions.** z=x·w+b is a binary logit; p=sigmoid(z); y is 0 or 1. Natural logarithms make the loss unit a nat. C in a library's regularized estimator is an inverse-regularization control, not this probability.

\[
p=\frac1{1+e^{-z}},\qquad
\ell=-y\log p-(1-y)\log(1-p).
\]

At z=0, p=0.5. With y=1, loss is −ln(0.5)=0.6931 and the logit derivative p−y=−0.5. A positive target with predicted p=0.8 has loss 0.2231. Threshold 0.9 rejects it despite the good probability; threshold 0.5 accepts it. No weight changed between those decisions.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing the probability model

A linear probability model fits a number directly to binary labels and may return values outside [0,1]. Logistic regression instead maps a linear score through sigmoid; a nonlinear classifier can learn a more complex boundary. Compare these as different probability models, not three steps in one model. Softmax models an exclusive choice; independent sigmoids need not sum to one. Well-ranked scores can be miscalibrated, so accuracy or AUROC alone is not probability validation.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: three-class prediction

For three Iris classes, fit multinomial logistic regression on scaled measurements and inspect a three-entry probability vector. Class order must match the saved label mapping. Do not train a binary sigmoid and quietly discard one species.

**Check.** A network outputs [batch,10] scores for a single-label ten-class task. Should you apply sigmoid and expect the outputs to sum to one?

**Answer.** No. Use the appropriate mutually exclusive classification objective and softmax for reported probabilities. Sigmoid is appropriate for separate binary labels under a different task definition.

## 5. Why cross-entropy gives a useful update

**Key terms.** A logit z is an unnormalized class score. Softmax turns K logits into K probabilities; a one-hot target has one entry equal to one.

**Process.** Compute logits, normalize with softmax, select the true-class probability and take its negative logarithm. Backpropagation passes the probability-minus-target error into each score's producing weights.

In this equation, K is the number of mutually exclusive classes, z_k is the score for class k, and y identifies the observed correct class. The indicator is 1 for that class and 0 for the others. Normalizing means dividing by the sum across classes so the probabilities add to one.

\[
p_k=\frac{\exp(z_k)}{\sum_j\exp(z_j)},\qquad
\ell=-\log p_y,\qquad
\frac{\partial\ell}{\partial z_k}=p_k-\mathbf1[k=y].
\]

For probabilities [0.7,0.2,0.1] and true class two, the logit gradients are [0.7,−0.8,0.1]. A gradient-descent update tends to raise the true-class score and lower the others. Numerically stable implementations subtract the largest logit and combine log-softmax with the loss; computing exp on arbitrary large scores can overflow.

### Prediction is a separate decision

The default multiclass rule selects the largest probability. A binary decision threshold or cost matrix can change decisions without refitting weights; select it using development data, never the final test. The current runnable datasets are multiclass, so the lab does not pretend to offer a generic binary threshold control. Calibration is about whether probabilities agree with frequencies, not simply whether the winning class is correct.

## Sources

- [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html).
- [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [PyTorch 2.9 CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html).
