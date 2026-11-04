# ML12. PCA, dimensionality reduction and anomalies

Dimensionality reduction replaces many input features with fewer coordinates. Principal component analysis, or PCA, chooses directions that preserve as much training-data variance as possible. Reconstructing an input from those fewer coordinates shows information that was lost. A large reconstruction error can be an anomaly signal, but does not prove that an example is invalid.

Training points lie on a diagonal line. We keep that one direction, represent the new point [2, 0] with one coordinate, then reconstruct it as [1, 1]. The control changes only the illustrative anomaly cutoff, not the fitted PCA direction.

PCA compresses inputs without using class labels. Low-variance information can still matter for prediction, so preserving variance is not the same as preserving everything useful to a target task.

[course map](../COURSE_MAP.md). Required background: ML11, P2.

## 1. What compression preserves and loses

**Key terms.** **PCA** finds high-variance linear directions; a **component** is one direction; **reconstruction** maps a compressed representation back. An **anomaly score** measures unusualness under a model, not confirmed harmful behaviour.

**Process.** Fit the feature transformation on training inputs, centre using training statistics, learn a reduced representation and apply it to held-out cases. For anomaly detection, choose the threshold under a separate labelled or explicitly assumed validation policy.

PCA can remove redundancy and reduce downstream cost. Large variance is not necessarily predictive signal: a low-variance coordinate might contain the label. Standardizing first changes the optimization because it changes the variance units. Whitening additionally rescales retained component scores and can amplify low-variance noise.

t-SNE and UMAP are alternatives for visualizing high-dimensional examples in two or three coordinates. They emphasize neighbourhood relationships rather than PCA's maximum-variance linear directions. A clear-looking plot can distort distances between groups and their apparent areas, so it is not proof of true categories. Isolation forests instead score how easily examples can be isolated by random splits; local outlier methods compare an example's density with its neighbours. Detecting unusual new cases is different from finding outliers already present in a contaminated fitting set.

## 2. A worked calculation

**Symbols and assumptions.** mu is training mean, v is a unit component, z=(x−mu)·v is a score and x-hat=mu+zv is one-component reconstruction. Explained variance is the fraction of fitted variance retained.

For centred points [−1,0] and [1,0], the first component is [1,0]. Their one-component reconstruction is exact. A new point [0,2] projects to z=0 and reconstructs as [0,0], giving squared error 4. That error reflects distance from the learned line, not proof that the new point is invalid.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. PCA versus feature selection and anomaly methods

Compare no reduction, PCA and supervised feature selection using a fixed downstream model. For anomaly methods, contamination is a threshold/assumption control, not a known population fact unless independently established. Report detection false alarms and subgroup effects where labels exist.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: separate compression from prediction quality

Reduce digits or Wine inputs using train-fitted PCA, then compare a fixed linear classifier at several component counts. Fit reduction inside cross-validation. Save reconstruction and classifier quality separately so a compact representation is not mistaken for better predictions.

**Check.** Can labels be safely used to choose a PCA dimension and then evaluated on those same rows?

**Answer.** Not for an independent final claim. That choice is supervised selection even though PCA's internal fit does not use labels.

## 5. PCA as projection and reconstruction

**Key terms.** A covariance matrix summarizes centered feature co-variation. An eigenvector names a direction; its eigenvalue gives variance along that direction. An orthonormal basis has unit, mutually perpendicular columns.

**Process.** Fit training centering/scaling, obtain principal directions by a numerically stable decomposition, project rows into the retained coordinates, reconstruct and measure what was lost.

Read the two equations as compression and reconstruction. X holds examples in rows, mu is the training mean, and the columns of V_k are the k retained directions. Z stores the smaller set of coordinates. Multiplying back by the transposed directions restores an approximation in the original feature space, then adding mu restores its location.

\[
Z=(X-\mu)V_k,\qquad
\widehat X=ZV_k^\top+\mu.
\]

If centered covariance is diag(9,1), the first principal direction retains variance 9/10=90%. Dropping the second direction loses expected squared error 1 per row across the two coordinates, or mean-per-feature error 0.5. Reporting reconstruction “MSE” therefore requires stating whether the sum or average over features is used.

### Units can change the answer

Standardizing each original feature makes both variances one in this example, so the original 90% story no longer applies. Whitening additionally rescales retained coordinates; it does not create more information.

The lab reports mean squared error in the selected preprocessed units. An autoencoder uses 0–1 pixel units instead. Their numerical reconstruction errors cannot be ranked until the coordinate system and averaging convention are matched. Isolation Forest is a different task: it scores isolation depth, not projection error.

## Sources

- [scikit-learn decomposition](https://scikit-learn.org/stable/modules/decomposition.html).
- [scikit-learn novelty/outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html).
