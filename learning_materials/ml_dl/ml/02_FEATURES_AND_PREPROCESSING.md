# ML02. Features and preprocessing

Most models need inputs in a consistent numerical form. Preprocessing handles missing values, represents categories and puts numerical measurements on suitable scales. Some transformations learn statistics from data, so they must be fitted on training data and reused unchanged on new cases.

Training values [1, 3] have mean 2 and population standard deviation 1. A new value 4 becomes (4 − 2) / 1 = 2. It is two training standard deviations above the training mean.

This example focuses on standardization. The additional notes explain missing-value handling, categories and complete pipelines; distance-based and regularized methods are especially sensitive to feature units.

[course map](../COURSE_MAP.md). Required background: ML01.

## 1. Representing inputs and missing values

**Key terms.** **Preprocessing** converts raw values into model inputs; **imputation** fills missing entries; **scaling** changes numeric units; **one-hot encoding** represents categories as indicator columns. A **pipeline** fits and applies these steps together.

**Process.** Split raw records first. Fit imputation, scaling, vocabularies and feature selection on training records only, transform validation/test without re-fitting, then fit the estimator. Inside cross-validation, repeat the entire fitted pipeline separately in each training fold.

A feature must be available at prediction time and have a defined unit. Numeric codes for colours do not imply meaningful distances between colours. One-hot encoding avoids that assumption, while unknown-category handling must be explicit. A missingness indicator can preserve the fact that a measurement was absent; imputation cannot reconstruct its true value.

Text can become token counts or TF-IDF features. The vocabulary and document frequencies are learned data, so even an unsupervised vectorizer belongs inside the fitting boundary. Sparse text matrices should not be blindly centred into dense arrays. For trees, numeric standardization is often unnecessary; for distances, regularization and gradient methods it can be central.

## 2. A worked calculation

**Symbols and assumptions.** mu is a training-feature mean; s is its nonzero training standard deviation; z=(x−mu)/s is a standardized value. A constant feature needs explicit safe handling.

Suppose a training column has mu=10 and s=2. Validation value 14 becomes (14−10)/2=2. Recomputing mu from the validation set changes the fitted transformation and exposes holdout information. For categories [red,blue], red can become [1,0]; an unknown green needs a declared policy, not an arbitrary third integer with misleading numeric order.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Scaling, encoding and feature choices

Standard scaling suits many distance/linear methods; robust median/quantile scaling is less sensitive to outliers but changes geometry. One-hot, ordinal and learned embeddings encode different assumptions. Polynomial features give linear models nonlinear inputs, but rapidly increase feature counts. Target encoding needs training-fold/out-of-fold safeguards because it directly uses y.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: test the effect of scaling

Wine measurements have very different scales. Compare scaled and unscaled k-NN with unchanged split, neighbours and distance rule. Fit a tree on the same data as a contrast. Treat any ranking as a measured outcome, not a guaranteed scaling win.

**Check.** You select the 20 features most correlated with the label using the full dataset and then cross-validate. Is the score clean?

**Answer.** No. Test-fold labels influenced selection. Put feature selection inside the pipeline so every fold chooses its features only from that fold's training data.

## Sources

- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html).
- [scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html).
