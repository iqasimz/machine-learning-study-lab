# ML03. Linear regression

Linear regression predicts a number by adding weighted feature contributions and, usually, a constant intercept. Fitting chooses weights that reduce prediction errors on training examples. Ridge regression adds a cost for large weights, which can make the fit less sensitive to limited or correlated data.

Here one feature predicts three targets: x = [1, 2, 3], y = [1, 2, 2]. To keep the solution visible we omit the intercept. Changing the ridge penalty changes the fitted coefficient, predictions and training error.

The diagram shows the small closed-form calculation, not the only way to fit regression. The loss and optimization lesson explains gradient-based fitting; the regularization lesson explains how validation chooses penalty strength.

[course map](../COURSE_MAP.md). Required background: ML02, P2.

## 1. Regression assumptions and intercepts

**Key terms.** A **linear model** combines features with learned weights; an **intercept b** is a constant offset; a **residual** is prediction minus target. **Least squares** chooses weights to minimize squared residuals; it does not establish causal effects.

**Process.** Prepared feature rows enter a weighted sum, predictions meet targets in a loss, and a solver finds better weights. Once fitted, new rows use the same preprocessing and weighted sum without another optimization.

The model is linear in its coefficients, even if the input includes polynomial features. Large residuals receive more influence under squared loss than under absolute loss. A closed-form mathematical solution can explain the optimum, but numerical implementations should use stable least-squares routines rather than explicitly invert a poorly conditioned matrix.

Independent, well-specified errors are important for classical inference claims, not a prerequisite for computing a fitted line. Collinear features can make coefficients unstable even when predictions remain similar. Poisson regression models positive count means through a log link; robust or quantile regression changes the fitting target. Do not label every numeric task as identical Gaussian least squares.

## 2. A worked calculation

**Symbols and assumptions.** x_i is feature row i; w is its coefficient vector; b is intercept; n is sample count; y_i is the observed target. L below uses a 1/(2n) convention so the gradient has no factor 2.

\[
\hat y_i=x_i^\top w+b,\qquad L=\frac1{2n}\sum_i(\hat y_i-y_i)^2.
\]

For one point x=2,y=1,w=1,b=0, prediction is 2 and L=0.5. The derivative with respect to w is (2−1)×2=2. Updating only w at learning rate 0.1 gives w=0.8, prediction 1.6 and L=0.18. If b is trainable, it also has a gradient; this scalar example deliberately holds it fixed.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Regression alternatives and solver settings

A mean predictor ignores features. Ordinary least squares estimates a conditional mean under its objective; ridge adds shrinkage; a tree permits piecewise predictions. Fit intercept and scaling consistently before comparing them. Solver tolerance controls optimization accuracy, not statistical generalization.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: choose the task for an ordered rating

Wine Quality uses an ordered rating. Regression asks how far a prediction is from that rating; multiclass classification asks whether the exact category is right. Choose the task before fitting, retain the original unit and report MAE/RMSE rather than pretending one setup answers both questions.

**Check.** Why can a model with a lower training MSE still be a worse choice?

**Answer.** It may overfit, exploit leakage or violate deployment constraints. Compare held-out error, stability and cost under the same protocol; inspect large residuals and groups.

## 5. From the loss to a stable solver

**Key terms.** A design matrix X stacks feature rows; a gradient gives one partial derivative per coefficient. Full column rank means no feature column is an exact linear combination of the others.

**Process.** Stack observations, compute predictions and residuals, differentiate the scalar objective and solve the resulting linear system. Apply the fitted preprocessing and coefficients unchanged to a new row.

In the formulas below, n is the number of training rows, w contains the coefficients and y contains observed targets. The norm squared means add all squared residuals. Dividing by 2n averages the errors and simplifies the derivative; it changes the numerical gradient scale, not the unregularized least-squares minimizer.

For L=‖Xw−y‖²/(2n), the gradient is Xᵀ(Xw−y)/n. Setting it to zero gives XᵀXw=Xᵀy. The familiar inverse formula requires invertibility, and forming XᵀX squares the condition number. QR and singular value decomposition (SVD) are stable matrix-factorization methods used to solve the least-squares problem. They are usually preferable to explicitly computing an inverse. Ridge instead solves (XᵀX+λI)w=Xᵀy under the unnormalized squared-error convention used by sklearn Ridge; compare objective conventions before transferring λ from a paper.

With X=[1,2]ᵀ and y=[1,2]ᵀ, no intercept and λ=1, XᵀX=5 and Xᵀy=5, giving w=5/6. Unregularized w=1 fits perfectly; ridge deliberately accepts some training error. An intercept is normally excluded from the shrinkage penalty.

### What to inspect in the lab

Compare λ=0, 1 and 10 on identical rows. Look at development error and coefficient stability, not training fit alone. The regression report retains original target units even when a neural comparator internally standardizes its target.

## Sources

- [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html).
- [An Introduction to Statistical Learning, author site](https://www.statlearning.com/).
