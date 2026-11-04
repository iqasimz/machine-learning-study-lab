# ML06. Nearest neighbours, kernels and SVMs

Nearest-neighbour prediction compares a new example with stored training examples and uses the answers of the closest ones. It is a local rule: the prediction depends on nearby cases rather than one fitted line. A support vector machine, or SVM, is a different method that learns a separating boundary with a margin.

For nearest neighbours we classify query x = 3 using labelled points at 0, 1, 2, 4 and 5. The control changes how many neighbours vote. A separate final section explains how a kernel-based SVM differs; it is not a later stage of nearest-neighbour prediction.

Feature scaling from the features and preprocessing lesson affects distances. This lesson compares two uses of similarity while keeping their prediction mechanisms distinct.

[course map](../COURSE_MAP.md). Required background: ML05.

## 1. Local voting versus margin classification

**Key terms.** **k-NN** predicts from nearby training examples; a **distance metric** defines nearby; a **support vector** constrains a margin solution; a **kernel** calculates an implicit feature similarity. **Hinge loss** penalizes insufficient signed margin.

**Process.** For k-NN, store training representations, measure query distances and aggregate neighbours. For an SVM, fit a margin-based decision function first, then apply it to queries. Scaling belongs before both fitting and distance/kernel computation.

Small k can follow local noise; larger k smooths decisions and can miss small classes. Distance weighting gives close neighbours more influence. In high dimensions, irrelevant coordinates and concentrated distances can make neighbourhoods less useful.

An RBF kernel can represent nonlinear boundaries without constructing all transformed features explicitly. Its gamma controls locality; C controls the trade-off between fitting violations and regularization under the estimator's objective. Kernel SVM training/storage can become expensive as the number of cases grows. An explicit-feature linear classifier is often the stronger operational baseline for sparse large text.

## 2. A worked calculation

**Symbols and assumptions.** For classification labels y∈{−1,+1}, f(x) is a real decision score. Hinge loss is max(0,1−y f(x)). The RBF kernel uses gamma≥0 and squared Euclidean distance.

With neighbour labels [A,A,B], an unweighted k=3 vote selects A. At distances [2,2,0.2], inverse-distance weights are [0.5,0.5,5], so B wins. For a positive example with f(x)=0.3, hinge loss is 0.7. For two points at squared distance 4 and gamma=0.5, RBF similarity is exp(−2)≈0.1353.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing k, kernel and regularization

A linear kernel tests whether nonlinear geometry is needed. Increasing gamma and C together can create a highly local, overfit boundary; neither guarantees improvement. k-NN requires retained training data at prediction, while a linear model summarizes it in weights.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare nearest neighbours and SVMs

Compare standardized k-NN, linear SVM and RBF SVM on the same digits split with equal numbers of search candidates. Record prediction latency and fit time as well as macro-F1. Do not rank their speed using unmeasured complexity slogans.

**Check.** A feature measured in thousands dominates one measured in fractions. Can changing k alone repair this geometry?

**Answer.** Not reliably. Inspect units and fit a justified scaler on training data, then retune the distance method within the same validation design.

## 5. Margin, kernel and regularization work together

**Key terms.** A margin measures a labelled example's signed distance-related score. Hinge loss penalizes examples whose signed score is below one. A support vector influences the fitted margin boundary.

**Process.** Fit a preprocessing transform on training data, compute linear or kernel similarities and solve the regularized margin objective. A new row uses that same transform and similarities to the stored support vectors.

The equation balances two costs. The first term discourages large weights; the second adds a penalty whenever a training example is wrong or not far enough on the correct side of the boundary. The chosen constant C controls how much these violations count. Inputs are x_i, labels y_i are −1 or +1, w are learned weights and b is the bias.

\[
\min_{w,b}\ \frac12\|w\|^2+C\sum_i\max(0,1-y_i(w^\top x_i+b)).
\]

This binary formulation uses labels −1 and +1 and an unaveraged loss sum. Multiclass SVC uses a reduction to binary problems. C is not λ: its numerical meaning changes with the loss normalization and number of training rows.

### A similarity calculation

With RBF gamma=0.5, distance squared 2 gives similarity exp(−1)≈0.368. If one unscaled feature makes distance squared 200, similarity is exp(−100), effectively zero. Changing feature units can therefore overwhelm kernel behaviour before C has any useful interpretation.

In the lab, gamma affects RBF only and is disabled for a linear kernel. SVC does not automatically supply calibrated probabilities here, so the results show class metrics without inventing log loss or a confidence calibration score.

## Sources

- [scikit-learn nearest neighbours](https://scikit-learn.org/stable/modules/neighbors.html).
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html).
