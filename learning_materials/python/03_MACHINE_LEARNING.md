# Machine learning code

Prerequisites: Python functions/containers and NumPy array shapes. These lessons teach scikit-learn interfaces and the calculations around them. They do not fit models to produce a new set of lesson scores. Constructed weights/predictions are identified as such. SK12 explains the existing lab's fitting workflow once, without a duplicate training-output panel.

## SK01. Estimators

### Why scikit-learn

scikit-learn is the library this course uses for classical machine learning. It is widely used because many learning objects share one shape: you construct the object with settings, `fit` learns from fitting rows, and `predict` or `transform` applies what was learned. A logistic regression, a tree, and a scaler are that kind of object. A scaler is a transformer: `fit` learns a statistic and `transform` applies it. A metric function is a different kind of tool. `f1_score(actual, predicted)` is a function call on labels you already have. It is not constructed, fitted, and then asked to predict. The same estimator shape is why a pipeline can hold the scaler and the model together. After `import sklearn`, the classes live under short module names such as `sklearn.linear_model`. Constructing `LogisticRegression(C=0.5)` only stores the setting. The library will also let you call `predict` before `fit`, and it will raise an error rather than invent coefficients. It does not choose the split, the metric, or whether a higher score is the decision you wanted. Fitting stays in the training lab. Where a later example writes `coef_` or `mean_` by hand, that is an inspection fixture: a constructed state used so the method can be shown. It is not the supported way to prepare an arbitrary estimator.

In scikit-learn, an **estimator** is an object configured to learn something from data. `X` is the table of examples. Its shape is `(examples, features)`: each row is one example, and each column is one measurement. `y` contains the target for each row. The confusion-matrix lesson already used one metrics function from this same library.

<!-- example: SK01-shape -->
```python
import numpy as np

examples = np.array([[1.0, 0.2], [0.0, 1.5], [1.0, 0.4]])
targets = np.array([1, 0, 1])
print(examples.shape)
print(targets.shape)
```

**Output**
```text
(3, 2)
(3,)
```

Three rows mean three examples. Two columns mean two features. `targets` has one answer per row, so its shape is `(3,)`.

Constructing an estimator only stores settings. `fit` is the method that learns from examples and writes the learned numbers onto the object. `predict` uses those numbers. Calling `predict` first fails, because the learned numbers are not there yet.

<!-- example: SK01-unfitted -->
```python
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
try:
    model.predict(np.array([[1.0, 0.2]]))
except NotFittedError as error:
    print(type(error).__name__)
```

**Output**
```text
NotFittedError
```

`LogisticRegression()` created the object. It has not seen any examples. `predict` refuses to invent a decision, and the error names the missing step: `fit`.

This lesson does not call `fit`. The next example places the learned numbers by hand, so `predict` and `predict_proba` can show what they return. `coef_` holds one weight per feature. `intercept_` holds the added constant. The trailing underscore is scikit-learn’s mark for a value produced by learning. `classes_` records which column of `predict_proba` belongs to which class.

<!-- example: SK01-predict -->
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.classes_ = np.array([0, 1])
model.coef_ = np.array([[1.0, -1.0]])
model.intercept_ = np.array([0.0])
model.n_features_in_ = 2
features = np.array([[2.0, 0.0], [0.0, 2.0]])
print(model.predict(features).tolist())
print(np.round(model.predict_proba(features), 3).tolist())
```

**Output**
```text
[1, 0]
[[0.119, 0.881], [0.881, 0.119]]
```

The first row has a larger score for class 1, so `predict` returns 1. The second row has a larger score for class 0. `predict_proba` returns a probability for each class, and each pair adds to 1. Those are class probabilities for the two rows, not a count of how many rows belong to each class. Column 0 belongs to class 0 because that is the order in `classes_`. The weights were written into the object so the method could be inspected. A real run would obtain them from `fit` on fitting rows only. Copying this assignment into a project is not how an estimator is trained. The existing training lab is where that call belongs.

<!-- example: SK01-main -->
```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression

models = [DummyClassifier(strategy="most_frequent"),
          DummyRegressor(strategy="mean"), LogisticRegression(C=0.5)]
print([type(model).__name__ for model in models])
print(models[2].get_params()["C"])
training_targets = [2., 4., 6.]
mean_baseline = sum(training_targets) / len(training_targets)
print([mean_baseline] * 2)
```

**Output**
```text
['DummyClassifier', 'DummyRegressor', 'LogisticRegression']
0.5
[4.0, 4.0]
```

No estimator was fitted. The final line manually applies the mean rule to two future examples. A baseline asks whether a more complex model improves on a simple defensible rule. For classification, “always predict the most common fitting class” can have high accuracy on imbalanced data while finding none of the rare class.

Settings such as `C` are hyperparameters; learned coefficients are model parameters. Scikit-learn often names learned attributes with a trailing underscore, such as `coef_`. An unfitted estimator should not be treated as a trained predictor. Calling `predict` before fitting usually raises a not-fitted error. The distribution is installed as `scikit-learn`, but imported as `sklearn`.

Sources: [sklearn getting started](https://scikit-learn.org/1.7/getting_started.html), [dummy estimators](https://scikit-learn.org/1.7/modules/model_evaluation.html#dummy-estimators).

## SK02. Pipelines

**Preprocessing** converts raw features into the representation a model expects. A scaler learns statistics; an imputer learns replacement values; an encoder may learn categories. `transform` applies statistics that were already learned. Like `predict`, it fails before that learning has happened.

<!-- example: SK02-transform -->
```python
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
try:
    scaler.transform(np.array([[5.0, 2.0]]))
except NotFittedError as error:
    print(type(error).__name__)
scaler.mean_ = np.array([1.0, 2.0])
scaler.scale_ = np.array([2.0, 2.0])
scaler.var_ = np.array([4.0, 4.0])
scaler.n_features_in_ = 2
print(scaler.transform(np.array([[5.0, 2.0]])).tolist())
```

**Output**
```text
NotFittedError
[[2.0, 0.0]]
```

The first call has no stored mean, so it stops. The lesson then supplies the mean and the scale as an inspection fixture, the same kind of constructed state as the classifier weights above. `(5 − 1) / 2` is 2, and `(2 − 2) / 2` is 0. `fit` would have calculated that mean and scale from fitting rows. Using the same stored values on later rows is what keeps the representation stable. Recomputing them from the rows you are about to score would mix those rows into the transformation.

If these operations see test data while learning, evaluation can become optimistic. A Pipeline expresses the order and keeps those learned transformations with the estimator.

<!-- example: SK02-main -->
```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

numeric = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
prepare = ColumnTransformer([("numbers", numeric, [0, 1])])
model = Pipeline([("prepare", prepare), ("classify", LogisticRegression())])
print(list(model.named_steps))
frozen_mean = np.array([4., 6.])
frozen_scale = np.array([2., 2.])
print(((np.array([[6., 4.]]) - frozen_mean) / frozen_scale).tolist())
```

**Output**
```text
['prepare', 'classify']
[[1.0, -1.0]]
```

The pipeline is configured, not fitted. The numerical output uses explicitly supplied statistics: `(6−4)/2 = 1`, `(4−6)/2 = −1`. In a real experiment, those statistics are learned only from fitting rows, then reused unchanged for validation/test/new inputs. Recomputing a mean on each prediction batch would change the representation and leak information between examples.

For mixed tables, ColumnTransformer applies different operations to numeric and categorical columns. OneHotEncoder creates indicator columns for categories; choose a policy for unseen categories rather than failing unexpectedly. During cross-validation, the entire learned preprocessing chain must be fitted separately inside each training fold. SK12 shows the final fitting boundary; this lesson does not run it.

Sources: [sklearn pipelines](https://scikit-learn.org/1.7/modules/compose.html), [common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html).

## SK03. Linear regression

Linear regression predicts a weighted sum of features plus an intercept. `predict` on a `LinearRegression` object does that arithmetic with the stored `coef_` and `intercept_`. The weights below are an inspection fixture: the same arithmetic a fitted model would apply, written in by hand so the lesson can show it without calling `fit`. They are not a saved training result and not a recipe for rebuilding an estimator from a notebook.

<!-- example: SK03-predict -->
```python
import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.coef_ = np.array([2.0, -1.0])
model.intercept_ = np.array([1.0])
model.n_features_in_ = 2
features = np.array([[1.0, 2.0], [2.0, 1.0]])
print(model.predict(features).tolist())
```

**Output**
```text
[1.0, 4.0]
```

The first row is `2×1 + (−1)×2 + 1 = 1`. The second is `2×2 + (−1)×1 + 1 = 4`. `predict` returns one number per row. Ridge discourages large squared weights; lasso uses an absolute-value penalty that can set weights to zero; elastic net combines both. These penalties affect fitting, not the arithmetic used to apply a fixed linear predictor.

<!-- example: SK03-main -->
```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

features = np.array([[1., 2.], [2., 1.]])
fixed_weights = np.array([2., -1.])
predicted = features @ fixed_weights + 1
actual = np.array([2., 3.])
print(predicted.tolist())
print(f"MAE={mean_absolute_error(actual, predicted):.1f}")
print(f"RMSE={root_mean_squared_error(actual, predicted):.1f}")
print([type(m).__name__ for m in [Ridge(alpha=1), Lasso(alpha=1), ElasticNet(alpha=1)]])
```

**Output**
```text
[1.0, 4.0]
MAE=1.0
RMSE=1.0
['Ridge', 'Lasso', 'ElasticNet']
```

The weights are teaching values, not fitted coefficients. Both predictions differ from their targets by one unit. MAE averages absolute errors; RMSE square-roots the mean squared error and penalizes large errors more strongly. R² compares squared error with a mean-target reference and can be negative; it is not a percentage accuracy.

Coefficient magnitudes depend on feature units. Standardizing a feature changes the coefficient scale and the effective regularization. PolynomialFeatures can create powers/interactions, but high degree can amplify overfitting and numerical problems. Inspect residuals—actual minus predicted values—and compare against a simple baseline rather than interpreting a complex equation as evidence of quality.

Sources: [sklearn linear models](https://scikit-learn.org/1.7/modules/linear_model.html), [ISLP book](https://www.statlearning.com/).

## SK04. Thresholds

Logistic regression produces class scores that can be mapped to probabilities. `predict_proba` lines those probabilities up in the order stored in `classes_`. The positive class is whichever position matches the label you care about. It is not automatically the last column.

<!-- example: SK04-column -->
```python
import numpy as np

classes = np.array(["premise", "claim"])
probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
claim_column = int(np.where(classes == "claim")[0][0])
print(claim_column)
print(probabilities[:, claim_column].tolist())
```

**Output**
```text
1
[0.2, 0.7]
```

`"claim"` is position 1 in `classes`. The claim probabilities are therefore 0.2 and 0.7, not the first column. A **threshold** then turns one of those probabilities into a label. Evaluation must distinguish the numerical scores from the decisions made using them.

<!-- example: SK04-main -->
```python
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

actual = [0, 1, 0, 1]
fixed_positive_probabilities = np.array([0.1, 0.9, 0.6, 0.4])
predicted = (fixed_positive_probabilities >= 0.5).astype(int)
print(predicted.tolist())
print(confusion_matrix(actual, predicted, labels=[0, 1]).tolist())
print(precision_score(actual, predicted), recall_score(actual, predicted), f1_score(actual, predicted))
```

**Output**
```text
[0, 1, 1, 0]
[[1, 1], [1, 1]]
0.5 0.5 0.5
```

These are constructed probabilities. There is one true positive, one false positive and one false negative. Precision is `1/(1+1)` and recall is `1/(1+1)`; F1 is their harmonic mean. In actual `predict_proba` output, locate the positive-class column through the estimator's `classes_`; do not assume column 1 always means the concept you want.

For several classes, macro F1 gives each class equal weight, weighted F1 weights by support and micro averaging pools counts. Multilabel tasks allow several simultaneous labels and differ from multiclass tasks with exactly one class per example. A higher F1 after changing a threshold does not prove better calibrated probabilities. Threshold choice belongs to validation, not repeated optimization on the final test set.

Sources: [sklearn classification metrics](https://scikit-learn.org/1.7/modules/model_evaluation.html#classification-metrics), [threshold tuning](https://scikit-learn.org/1.7/modules/classification_threshold.html).

## SK05. Neighbours

A nearest-neighbour rule stores examples and predicts the label of the closest one. Distance is the absolute difference here.

<!-- example: SK05-near -->
```python
reference = [0.0, 2.0, 5.0]
labels = [0, 1, 0]
query = 1.8
distances = [abs(point - query) for point in reference]
nearest = distances.index(min(distances))
print([round(distance, 1) for distance in distances])
print(labels[nearest])
```

**Output**
```text
[1.8, 0.2, 3.2]
1
```

The query 1.8 is 0.2 away from the stored value 2, and farther from 0 and 5. The label stored with 2 is 1, so that is the prediction. No model was fitted. An SVM learns a separating rule instead of storing every neighbour. Naive Bayes combines class evidence under a conditional-independence assumption. Similar method names do not make those mechanisms the same.

<!-- example: SK05-main -->
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB

reference = np.array([0., 2., 5.])
labels = np.array([0, 1, 0])
distance = np.abs(reference - 1.8)
print(labels[distance.argmin()].item())
models = [KNeighborsClassifier(n_neighbors=1), SVC(kernel="rbf", C=1), GaussianNB(), MultinomialNB()]
print([type(model).__name__ for model in models])
```

**Output**
```text
1
['KNeighborsClassifier', 'SVC', 'GaussianNB', 'MultinomialNB']
```

The nearest-reference calculation is manual and performs no fitting. Changing feature units can change which point is nearest; this is why distance-based methods often need scaling. Larger `k` smooths local decisions. For an RBF SVM, `gamma` controls locality and `C` the regularization trade-off; tune them together under a valid selection protocol.

GaussianNB models numerical class-conditional feature distributions. MultinomialNB is appropriate for nonnegative count-like features, not arbitrary centered measurements. A large predicted probability from an assumption-heavy model is not automatically a reliable confidence estimate. Current lab training includes Gaussian NB, kNN classification and SVM classification/regression—not every variant constructed here.

Sources: [sklearn neighbours](https://scikit-learn.org/1.7/modules/neighbors.html), [SVM](https://scikit-learn.org/1.7/modules/svm.html), [Naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html).

## SK06. Trees

A decision tree asks a yes-or-no question about one feature. Several trees can vote, and the majority wins.

<!-- example: SK06-vote -->
```python
feature = 1.5
print("claim" if feature >= 1 else "premise")
votes = [0, 1, 1]
print(1 if sum(votes) > len(votes) / 2 else 0)
```

**Output**
```text
claim
1
```

`1.5` is at least 1, so this one question answers `"claim"`. The three votes contain two 1s, which is more than half, so the vote is 1. A random forest combines randomized trees. Stacking learns another model from predictions that were kept separate from the rows used to make them.

<!-- example: SK06-main -->
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

fixed_tree_votes = np.array([[0, 1], [1, 1], [1, 0]])
print((fixed_tree_votes.mean(axis=0) > 0.5).astype(int).tolist())
models = [DecisionTreeClassifier(max_depth=3), RandomForestClassifier(n_estimators=30), ExtraTreesClassifier()]
print([type(model).__name__ for model in models])
```

**Output**
```text
[1, 1]
['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier']
```

Rows here are three fixed trees' binary votes; columns are two examples. Both examples receive two positive votes. This illustrates hard voting, not necessarily a forest library's probability-averaging implementation. A tree's depth and minimum leaf size limit complexity; the number of trees controls ensemble size and cost. More trees are not the same as deeper trees.

Bagging fits models to randomized samples and aggregates them. Stacking needs out-of-fold predictions to train the combining model without giving it unrealistically easy fitting-set outputs. Tree importance values can favor some feature types and split credit across correlated features. The ML lab supports its defined tree and forest routes; ExtraTrees and stacking remain code/theory comparisons, not advertised training options.

Sources: [sklearn trees](https://scikit-learn.org/1.7/modules/tree.html), [sklearn ensembles](https://scikit-learn.org/1.7/modules/ensemble.html).

## SK07. Boosting

Boosting starts with a prediction and then adds a small correction. The learning rate shrinks that correction before it is added.

<!-- example: SK07-step -->
```python
prediction = 2.0
correction = 1.0
learning_rate = 0.2
print(prediction + learning_rate * correction)
```

**Output**
```text
2.2
```

The correction is 1. Only a fifth of it is added, so the prediction moves from 2.0 to 2.2. The next component would look at the error that remains. The number of rounds and the size of each tree both change how far the model can move, and they are not the same setting.

<!-- example: SK07-main -->
```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostClassifier

initial_prediction = np.array([2., 2.])
fixed_corrections = np.array([[1., -1.], [0.5, 0.5]])
print((initial_prediction + 0.2 * fixed_corrections.sum(axis=0)).tolist())
models = [HistGradientBoostingRegressor(max_iter=20), GradientBoostingRegressor(), AdaBoostClassifier()]
print([type(model).__name__ for model in models])
```

**Output**
```text
[2.3, 1.9]
['HistGradientBoostingRegressor', 'GradientBoostingRegressor', 'AdaBoostClassifier']
```

The corrections are supplied numbers; they were not learned here. The first example receives `0.2×(1+0.5)=0.3`, while the second receives `0.2×(−1+0.5)=−0.1`. Classification boosting combines scores according to its objective; do not blindly apply a regression calculation to class labels.

Histogram boosting bins numerical values to make split search efficient. AdaBoost uses a different reweighting mechanism from gradient boosting. XGBoost, LightGBM and CatBoost offer additional implementations/features but are not needed to understand the additive idea. The existing lab implements histogram boosting with a fixed round budget, not an exposed early-stopping or every-library comparison service.

Sources: [sklearn gradient boosting](https://scikit-learn.org/1.7/modules/ensemble.html#gradient-boosting), [ISLP tree methods](https://www.statlearning.com/).

## SK08. Splits

A score answers the question defined by which rows were used to learn and which rows were held out. If two rows come from the same person, putting one in each side lets the model see that person before the test.

<!-- example: SK08-group -->
```python
groups = ["a", "a", "b", "b"]
train = [0, 1]
valid = [2, 3]
train_groups = {groups[i] for i in train}
valid_groups = {groups[i] for i in valid}
print(sorted(train_groups), sorted(valid_groups))
print(train_groups.isdisjoint(valid_groups))
```

**Output**
```text
['a'] ['b']
True
```

Rows 0 and 1 both belong to person `a` and stay in training. Rows 2 and 3 belong to person `b` and stay in validation. The two sets of people do not overlap. Time splitting instead keeps earlier rows on the fitting side. Cross-validation rotates those subsets. It does not repair a split that mixes one person across both sides.

<!-- example: SK08-main -->
```python
import numpy as np
from sklearn.model_selection import GroupKFold, ParameterGrid

features = np.zeros((6, 2))
groups = np.array(["a", "a", "b", "b", "c", "c"])
folds = list(GroupKFold(n_splits=3).split(features, groups=groups))
print([(len(train), len(valid)) for train, valid in folds])
print(all(set(groups[train]).isdisjoint(groups[valid]) for train, valid in folds))
print(len(ParameterGrid({"C": [0.1, 1.0], "kernel": ["linear", "rbf"]})))
```

**Output**
```text
[(4, 2), (4, 2), (4, 2)]
True
4
```

The splitter only creates indices; no model is fitted. The grid enumerates four configurations; it does not run four experiments. A search would fit each candidate inside valid training folds. Any learned preprocessing must be refitted within those folds too. If a final performance estimate is required after tuning, use an untouched test set or an appropriately nested procedure.

Repeatedly selecting a configuration from test scores changes the test set's role into development data. Training seeds assess fitted-model variation; bootstrap resampling assesses specified sampling variability. Neither justifies inventing tiny error bars from repeated visits to the same table. The current lab supports manual compatible-run comparisons, not a full automated cross-validation/search interface.

Sources: [sklearn cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html), [parameter search](https://scikit-learn.org/1.7/modules/grid_search.html).

## SK09. Clustering

Clustering assigns a point to the nearest center. Those centers are not class labels supplied by a teacher.

<!-- example: SK09-center -->
```python
point = 1.0
centers = [0.0, 10.0]
distances = [abs(point - center) for center in centers]
print(distances)
print(distances.index(min(distances)))
```

**Output**
```text
[1.0, 9.0]
0
```

The point is 1 away from center 0 and 9 away from center 10, so it joins center 0. PCA instead returns coordinates along directions of large spread. Anomaly detection returns an unusualness score. A cluster number, a coordinate and a score are different kinds of output.

<!-- example: SK09-main -->
```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

points = np.array([[1., 0.], [8., 0.]])
fixed_centers = np.array([[0., 0.], [10., 0.]])
distances = ((points[:, None, :] - fixed_centers[None, :, :]) ** 2).sum(axis=2)
print(distances.argmin(axis=1).tolist())
print((points @ np.array([[1.], [0.]])).tolist())
print([type(m).__name__ for m in [KMeans(), DBSCAN(), AgglomerativeClustering(), PCA(), TruncatedSVD(), GaussianMixture(), IsolationForest()]])
```

**Output**
```text
[0, 1]
[[1.0], [8.0]]
['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'PCA', 'TruncatedSVD', 'GaussianMixture', 'IsolationForest']
```

The centers and projection are supplied for illustration, not fitted K-means/PCA outputs. K-means minimizes within-cluster squared distance; DBSCAN uses density connectivity and may mark noise as −1. A Gaussian mixture offers component probabilities, not known semantic classes. Cluster IDs can be permuted without changing the grouping.

PCA's high-variance directions are not guaranteed to contain the best predictive information. TruncatedSVD is useful with sparse features without the same centering requirement. Isolation Forest's contamination setting helps set a decision threshold, not discover a verified true anomaly prevalence. DBSCAN lacks a native held-out prediction method in this lab: its fitting-set diagnostics must not be presented as equivalent to a held-out supervised score.

Sources: [sklearn clustering](https://scikit-learn.org/1.7/modules/clustering.html), [decomposition](https://scikit-learn.org/1.7/modules/decomposition.html), [outlier detection](https://scikit-learn.org/1.7/modules/outlier_detection.html).

## SK10. Calibration

**Calibration** asks whether events assigned a probability occur at about that frequency. One way to score a single probability is the square of its distance from what happened: 1 if the event happened, 0 if it did not.

<!-- example: SK10-one -->
```python
probability = 0.8
happened = 0
print(round((probability - happened) ** 2, 2))
```

**Output**
```text
0.64
```

The model said 0.8 and the event did not happen. `(0.8 − 0)²` is 0.64. A lower number would mean the stated probability was closer to the outcome. A useful ranking can still have badly calibrated probabilities. A decision also depends on what a false alarm or a miss costs.

<!-- example: SK10-main -->
```python
import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix

actual = np.array([0, 1, 0, 1])
probabilities = np.array([0.2, 0.4, 0.8, 0.9])
print(f"Brier={brier_score_loss(actual, probabilities):.4f}")
for threshold in [0.3, 0.8]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    print(threshold, "false alarms", fp, "misses", fn)
```

**Output**
```text
Brier=0.2625
0.3 false alarms 1 misses 0
0.8 false alarms 1 misses 1
```

The Brier score averages squared probability errors: `(.04+.36+.64+.01)/4`. Lower is better for the same task/protocol. Raising this threshold creates an extra miss without reducing false alarms in these four constructed cases. That is an observed trade-off for this fixture, not a general theorem about all models.

Calibration fitting and threshold selection are learned choices and require a proper development protocol. Slice metrics can expose group/domain failures hidden by an overall mean, but tiny groups have unstable estimates. Permutation importance changes a feature in a frozen model's input and measures the resulting difference; correlation can complicate interpretation, and it is not a causal-effect estimate. These concepts do not imply dedicated calibration/threshold-fitting controls already exist in the lab.

Sources: [sklearn calibration](https://scikit-learn.org/1.7/modules/calibration.html), [permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html).

## SK11. Time and ranking

A forecast predicts a later value from information available earlier. The input is the past. The target is the next observation.

<!-- example: SK11-lag -->
```python
observations = [10.0, 12.0, 11.0]
print(observations[:-1], observations[1:])
```

**Output**
```text
[10.0, 12.0] [12.0, 11.0]
```

`10` is used to predict `12`, and `12` is used to predict `11`. The future is not used to predict the past. A recommender instead ranks candidate items. Shuffling rows as if they were ordinary classification examples answers a different question.

<!-- example: SK11-main -->
```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

observations = np.array([10., 12., 11., 15., 14., 18.])
lagged_input = observations[:-1]
next_value = observations[1:]
print(lagged_input.tolist(), next_value.tolist())
print(all(train.max() < test.min() for train, test in TimeSeriesSplit(n_splits=2).split(observations)))
relevance = [1, 0, 1]
print(f"precision@2={sum(relevance[:2]) / 2:.1f}")
```

**Output**
```text
[10.0, 12.0, 11.0, 15.0, 14.0] [12.0, 11.0, 15.0, 14.0, 18.0]
True
precision@2=0.5
```

The first pair uses time `t` to predict `t+1`, not the future to predict the past. Real forecasts may require a gap, horizon-aware features and rolling evaluation. The recommendation example counts relevant items in the first two ranks; it needs known relevance judgments and a declared candidate set.

`partial_fit` updates supported estimators incrementally; it is still learning and is not called here. Semi-supervised methods use unlabeled as well as labeled data, while Gaussian processes represent functions with a probabilistic prior/kernel. Sequential decision problems additionally need actions, transitions and rewards, not only target labels. These are scope extensions, not currently runnable time-series/recommender/RL lab trainers. See the extension register for code still to be expanded.

Sources: [sklearn TimeSeriesSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [scaling strategies](https://scikit-learn.org/1.7/computing/scaling_strategies.html).

### A sequential decision adds transitions and rewards

In a Markov decision model, an action can change the next state as well as produce an immediate reward. Given fixed transition probabilities and state values, calculate one action's expected return without learning a policy:

<!-- example: SK11-return -->
```python
import numpy as np

next_state_probabilities = np.array([0.25, 0.75])
fixed_next_state_values = np.array([2., 5.])
immediate_reward, discount = 1., 0.9
expected_return = immediate_reward + discount * (next_state_probabilities @ fixed_next_state_values)
print(f"expected return={expected_return:.3f}")
```

**Output**
```text
expected return=4.825
```

The expected next value is `.25×2+.75×5=4.25`; discounting and adding reward gives 4.825. These values are provided, not learned. A classification label does not encode this action-dependent future. The existing ML lab has no sequential-decision trainer. Source: [Sutton and Barto, author's book page](http://incompleteideas.net/book/the-book-2nd.html).

## SK12. The fitting workflow

This is a **read-only walkthrough**, not an output-bearing example or another training interface. The existing ML lab selects a supported dataset/task/model, obtains fixed fitting/development/test indices, fits preprocessing only on fitting rows, fits the estimator and evaluates under its task-specific protocol.

The function below is a simplified supervised illustration of that order, not a verbatim replacement for `model_lab.py`. Its caller supplies the configured pipeline and already separated arrays. It intentionally has no call and no training-output block.

```python walkthrough
def fit_and_evaluate_supervised(pipeline, x_train, y_train, x_dev, y_dev, x_test, y_test):
    from sklearn.metrics import accuracy_score
    pipeline.fit(x_train, y_train)
    development_predictions = pipeline.predict(x_dev)
    test_predictions = pipeline.predict(x_test)
    return {
        "development_accuracy": accuracy_score(y_dev, development_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
    }
```

Here `pipeline.fit` learns every preprocessing step and then the estimator from `x_train/y_train`. Prediction reuses that learned state. Accuracy is only an illustrative classification metric; regression, clustering, PCA and anomaly methods need different outputs and checks. The real lab records richer artifacts and enforces method/dataset compatibility. It also has a simple baseline and comparisons; it does not automatically run the cross-validation/search described in SK08.

**Run supported experiments:** open **Machine learning → Training lab** (`ml-lab`). **Saved results** (`ml-results`) owns real metrics/predictions, and **Compare models** (`ml-compare`) owns compatible-run comparisons. On the hosted preview, training is unavailable; install locally to fit models. No lesson click should start a job.

Save a complete fitted preprocessing/model pipeline together with the environment and input schema. Loading unknown pickle/joblib objects can execute code, and cross-version compatibility is not guaranteed. An inference example must use a reviewed artifact or a transparent fixed-weight calculation, not an unexplained binary downloaded from a stranger.

Sources: [sklearn pipeline](https://scikit-learn.org/1.7/modules/compose.html), [model persistence](https://scikit-learn.org/1.7/model_persistence.html). Local implementation reference: `src/training_lab/model_lab.py`; internal page names above are navigation targets, not assumed URL routes.
