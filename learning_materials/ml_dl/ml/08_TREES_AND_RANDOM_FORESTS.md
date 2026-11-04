# ML08. Decision trees and random forests

A decision tree predicts by asking feature questions, such as whether a measurement is below a cutoff. Each answer sends an example down one branch until a leaf returns a prediction. A random forest trains many varied trees and combines their predictions, making the result less dependent on one particular tree.

The four training examples have feature values [1, 2, 3, 4] and labels [0, 0, 1, 1]. Change the cutoff to see which groups it creates and how mixed their labels are.

This diagram evaluates a candidate training split. Its two branches meet only to calculate split quality; a prediction for one new case would follow one branch, not average the two leaves. The forest note explains how multiple trees are combined.

[course map](../COURSE_MAP.md). Required background: ML05.

## 1. How trees differ from random forests

**Key terms.** A **decision tree** partitions features with tests; a **leaf** makes a prediction; **impurity** measures label mixing. **Bagging** fits models on resampled data; a **random forest** adds feature randomness to many trees.

**Process.** At each tree node, evaluate candidate splits using training labels, choose a permitted gain and repeat until stopping rules apply. Predict by following one path to a leaf; a forest combines the predictions of many fitted trees.

A deep tree can memorize small leaves and react strongly to training changes. Minimum leaf size, depth and pruning restrict this. Numeric scaling generally preserves order-based splits, unlike distance methods. Category encoding still matters: an arbitrary ordinal code may impose an unhelpful partition ordering.

A forest averages less-correlated errors. More trees can reduce Monte Carlo variation but do not create new independent training data. Out-of-bag observations can supply an internal estimate when bootstrap sampling is used; tuning repeatedly against that estimate still involves selection. Impurity-based feature importance can favour high-cardinality variables, and correlated features complicate interpretation.

## 2. A worked calculation

**Symbols and assumptions.** p_c is a node's class proportion, N is its sample count and G=1−Σp_c² is Gini impurity. Split gain is parent Gini minus the child-size-weighted Gini average.

A parent has 4 positives and 4 negatives, so G=0.5. Split into two four-case children with label counts 3/1 and 1/3. Each child has G=1−0.75²−0.25²=0.375. Weighted child impurity is 0.375 and gain is 0.125. A completely pure split would have larger training gain, but may depend on a leaked feature.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Tree depth, leaf size and forest size

Increase leaf size to reduce highly local fitting; cap depth or prune to restrict partitions. More forest trees chiefly change averaging cost/variation; increasing features per split changes tree correlation and strength. Compare a single constrained tree with the forest before attributing a gain to every ingredient.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: feature importance is not causality

On Wine Quality, inspect residuals and prediction stability across seeds. Do not identify an input as a causal driver just because it received high importance. Held-out permutation importance is another predictive diagnostic, with its own correlation caveats.

**Check.** Can growing 1,000 trees fix a target column accidentally included in X?

**Answer.** No. It can make the leaked result more stable. Remove forbidden features and rebuild a clean split/pipeline before comparing algorithms.

## Sources

- [scikit-learn ensembles](https://scikit-learn.org/stable/modules/ensemble.html).
- [An Introduction to Statistical Learning, author site](https://www.statlearning.com/).
