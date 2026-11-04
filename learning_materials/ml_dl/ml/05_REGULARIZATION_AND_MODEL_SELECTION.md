# ML05. Regularization and model selection

A model can fit its training examples closely but perform poorly on new data. Regularization discourages some kinds of overly flexible fit, for example by penalizing large coefficients. Model selection uses validation data to choose how much regularization or complexity to allow.

The example holds one candidate weight at 3 and its data loss at 0.4. Changing lambda shows how the L2 penalty contributes to the objective. It does not refit the candidate, so it is not evidence that one lambda trains a better model.

The linear regression lesson showed a coefficient actually changing when ridge strength changes. Here we separate the data-loss term, penalty term and validation choice so they are not confused.

[course map](../COURSE_MAP.md). Required background: ML04.

## 1. Why complexity can overfit

**Key terms.** **Bias** is systematic limitation; **variance** is sensitivity to the sampled training data; **regularization** restricts fitting; a **hyperparameter** is selected rather than learned by the inner fit. **Cross-validation** repeats fitting on held-out folds.

**Process.** Define a small configuration search, fit each complete pipeline on training folds and compare validation scores. Choose one recipe, refit on the allowed training data and evaluate the untouched final test only after selection.

A complex model can fit noise; too much shrinkage can erase useful signal. **L2** penalizes a sum of squared coefficients; **L1** penalizes their absolute magnitudes and can induce zeros; **elastic net** mixes both. Their effects depend on feature scaling and objective normalization. A fitted coefficient is a model parameter; the penalty strength is a hyperparameter. Do not equate similarly named alpha, lambda and C across libraries.

### Learning curves and validation curves

A learning curve asks whether more training data helps. Fit the same procedure on different training sizes, such as 100, 500 and 1,000 cases, and evaluate on a fixed held-out population. Keep the evaluation data outside each fit. A large gap between training and validation performance can suggest sensitivity to the available training data; it is not a complete diagnosis on its own.

A validation curve asks whether a setting helps at a fixed data size. For example, fit trees at depths 1, 3 and 8 and compare training and validation scores. Improving training fit while validation worsens suggests that the added complexity is not helping on these held-out cases. Neither curve is a universal rule about all deep networks.

Nested cross-validation uses an inner search to choose settings and an outer evaluation to test that selection procedure when no separate final holdout is available. The course's local comparison page does not run an automatic nested search.

## 2. A worked calculation

**Symbols and assumptions.** w contains coefficients, lambda is nonnegative penalty strength and L_data is a declared average loss. The simple objective here is L_data+lambda||w||²/2; it is not a claim about every library's exact scaling.

For a one-coefficient penalty with w=3 and lambda=0.2, added loss is 0.2×9/2=0.9 and added gradient is lambda×w=0.6. A polynomial expansion of d inputs through degree 2, including constant and cross terms, has (d+2)(d+1)/2 features: for d=3, that is 10, before any policy to drop the constant.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Ways to control model complexity

Limit polynomial degree, increase penalty, use early stopping or collect more independent data. These interventions act differently. A tree's depth limit is not numerically comparable to ridge lambda. Compare validation quality and cost rather than setting both controls to the same number.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: tune inside the training folds

On Wine Quality, compare linear and degree-2 ridge models using fixed inner folds and matched search counts. If a fitted scaler or feature selector saw outer test rows, the score is not a valid nested comparison.

**Check.** You test 100 settings and report the best test score. Why is a narrow bootstrap interval around that score insufficient?

**Answer.** The test set was used for model selection; uncertainty from resampling its cases does not undo selection bias. Use a new independent assessment or a design that nests selection inside evaluation.

## 5. Why choosing the best result is itself learning

**Key terms.** Model selection chooses an algorithm or setting; estimation fits its weights. Nested cross-validation separates the outer evaluation from inner selection.

**Process.** Reserve a final test first. Within the remaining data, fit every preprocessing operation and candidate model inside each training fold. Use inner validation to choose a candidate, refit only on the permitted data, then evaluate once at the outer boundary.

Trying 100 settings gives 100 chances to exploit validation noise. The smallest observed error is therefore an optimistic estimate of the chosen procedure's future error. More folds do not repair leakage: a scaler fitted on all rows before cross-validation still sees validation values.

### A numerical regularization check

The following small problem shows why L1 can remove a feature entirely. Think of a as the coefficient preferred by the unpenalized fit and w as the coefficient we will choose after adding a penalty. The sign function preserves whether a is positive or negative; max(…,0) prevents the remaining magnitude going below zero.

For the one-coefficient objective ½(w−a)²+λ|w|, the solution is sign(a)·max(|a|−λ,0). If a=0.6 and λ=0.8, the fitted coefficient is zero; if a=2 it becomes 1.2. L1 creates exact zeros. L2 shrinkage for ½(w−a)²+λw²/2 instead gives w=a/(1+λ), usually nonzero.

The interface uses fixed development/test partitions, not an automatic nested search. Its comparison screen is a learning instrument; repeatedly selecting from the displayed test scores does not become a valid nested experiment.

## Sources

- [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html).
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- [An Introduction to Statistical Learning, author site](https://www.statlearning.com/).
