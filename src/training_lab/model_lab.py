"""Real local ML/DL experiments with bounded settings and auditable split artifacts."""
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from . import model_data
from .hardware import choose_device, hardware_info
from .artifacts import ROOT, read_json, run_path, seal, sha256, source_hashes, verify, write_json

RUNS = ROOT / "model_runs"


def control(label, choices, default, definition, effect, equation=""):
    return {"label": label, "choices": choices, "default": default, "definition": definition, "effect": effect, "equation": equation}


CONTROLS = {
 "train_limit": control("Maximum training rows", [500, 1500, 5000, 15000, 60000], 5000, "Cap only fitting examples; development and test remain fixed.", "More examples cost more computation. Different caps are data-budget comparisons, not isolated model changes."),
 "seed": control("Model seed", [7, 17, 29], 17, "Initializes model weights, fitting randomness and minibatch order, not the split.", "Repeat seeds to measure variation. The split seed stays fixed at 20261102."),
 "scaling": control("Feature scaling", ["none", "standard", "robust"], "standard", "Fit a feature transform using training rows only, then apply it unchanged to development/test.", "Scaling changes distances and the effective strength of regularization; trees usually do not need it.", "standard: z = (x − training mean) / training standard deviation"),
 "strength": control("Regularization strength λ", [0.0, .01, .1, 1.0, 10.0], 1.0, "A penalty discouraging large coefficients; the exact objective differs for ridge, lasso and elastic net.", "Larger penalties can reduce variance but also erase useful signal. This λ is not the SVM/logistic C."),
 "C": control("Inverse regularization C", [.01, .1, 1.0, 10.0, 100.0], 1.0, "scikit-learn's inverse penalty control for logistic regression and SVM.", "Larger C weakens regularization; small C can underfit. Scaling changes the trade-off."),
 "l1_ratio": control("L1 fraction", [.1, .5, .9], .5, "Mixes sparse absolute-value and squared-weight penalties in elastic net.", "Near one behaves more like lasso; near zero more like ridge."),
 "neighbors": control("Number of neighbours k", [1, 3, 5, 11, 25], 5, "The closest training examples voting on a new example.", "Small k is flexible and noise-sensitive; larger k smooths boundaries."),
 "neighbor_weights": control("Neighbour voting", ["uniform", "distance"], "uniform", "Equal votes or votes weighted by inverse distance.", "Distance weighting makes nearby examples more influential; duplicate points can dominate."),
 "kernel": control("SVM kernel", ["linear", "rbf"], "rbf", "Linear uses a flat separator; RBF permits local curved boundaries.", "RBF flexibility depends jointly on C, feature scaling and gamma."),
 "gamma": control("RBF gamma", ["scale", .01, .1, 1.0], "scale", "How quickly RBF similarity decays with squared distance.", "Larger gamma makes more local decisions and can overfit.", "K(x,z) = exp(−γ ‖x−z‖²)"),
 "epsilon": control("Regression error tube ε", [.01, .1, .3], .1, "SVR ignores prediction errors inside an epsilon-wide tube in target units.", "Larger epsilon permits more small errors; it changes the fitting objective, not the reported MAE."),
 "smoothing": control("Variance smoothing", [1e-11, 1e-9, 1e-6, .001], 1e-9, "A stabilizing addition to Gaussian Naive Bayes feature variances.", "Too little can be numerically fragile; too much can blur discrimination."),
 "depth": control("Maximum tree depth", [2, 4, 8, 16, None], 8, "The longest allowed sequence of tree decisions. None allows growth until other rules stop it.", "Deeper trees fit finer patterns but can memorize small training sets."),
 "leaf": control("Minimum rows per leaf", [1, 3, 10, 20], 3, "The minimum number of training rows in a final tree region.", "Larger leaves smooth predictions and reduce variance."),
 "trees": control("Number of trees", [30, 100, 200], 100, "Independent bootstrapped trees whose predictions are combined.", "More trees reduce sampling noise but increase memory and fitting time."),
 "rounds": control("Boosting rounds", [30, 100, 200], 100, "Sequential fitted trees correcting the current model's objective.", "More rounds increase capacity. This implementation disables internal early stopping for a transparent fixed budget."),
 "boost_rate": control("Boosting learning rate", [.03, .1, .3], .1, "Scales each newly added tree's contribution.", "Smaller values generally need more rounds; do not compare learning rate without its round budget."),
 "clusters": control("Number of clusters", [2, 3, 4, 6, 10], 3, "The requested number of K-means groups or Gaussian mixture components.", "A requested group count is an assumption, not a discovered truth."),
 "radius": control("DBSCAN neighbourhood radius", [.15, .3, .5, .8, 1.2], .3, "Maximum distance for neighbours after the selected scaling.", "Small radius labels more points as noise; large radius can merge distinct groups."),
 "min_samples": control("DBSCAN minimum neighbours", [3, 5, 10, 20], 5, "The local density requirement for a core point, including the point itself.", "Larger requirements reject sparse regions. This lab reports DBSCAN fitting-set diagnostics only."),
 "components": control("PCA components", [2, 3, 5, 10], 2, "The number of orthogonal directions retained from training-set variance.", "More components reduce reconstruction error but compress less. High variance is not necessarily predictive signal."),
 "contamination": control("Expected outlier fraction", [.02, .05, .1, .2], .1, "Isolation Forest's training threshold assumption.", "The chosen fraction sets a cutoff, not an independently estimated real-world prevalence."),
 "epochs": control("Epochs", [1, 3, 5, 10, 20, 30], 10, "One epoch visits every selected training row once.", "More epochs can improve fitting or overfit. The best development checkpoint is restored before the final test."),
 "batch_size": control("Batch size", [16, 32, 64, 128, 256], 64, "Examples contributing to one backward calculation and weight update.", "Larger batches use more activation memory and yield fewer updates per epoch."),
 "learning_rate": control("Learning rate", [.0001, .0003, .001, .003, .01, .03], .001, "The scale used by SGD or AdamW to change weights.", "Too large can destabilize learning; too small may appear not to learn. Compare with batch size and schedule."),
 "optimizer": control("Optimizer", ["adamw", "sgd"], "adamw", "AdamW adapts per-parameter updates; SGD here uses momentum 0.9.", "The same numerical learning rate is not equally suitable for both methods."),
 "width": control("Hidden width / CNN channels", [16, 32, 64, 128], 64, "MLP hidden units, or the channel count in the first convolution block.", "Wider layers increase capacity, parameter count and computation. CNN block two has twice this width."),
 "layers": control("Hidden layers", [1, 2, 3], 2, "The number of nonlinear hidden layers in an MLP.", "Depth increases repeated transformations and may make optimization harder."),
 "activation": control("Activation", ["relu", "gelu", "tanh"], "relu", "The nonlinear transformation between learned linear layers.", "Without nonlinearity stacked linear layers collapse into one linear map."),
 "dropout": control("Dropout fraction", [0.0, .1, .3, .5], .1, "Randomly zeros this fraction of hidden activations during training only.", "Can reduce co-adaptation; excessive dropout can prevent fitting. Evaluation disables it."),
 "weight_decay": control("Weight decay", [0.0, .0001, .001, .01], .0001, "Optimizer-level shrinkage on weights.", "AdamW uses decoupled decay; SGD implements L2-style decay. They are not numerically identical objectives."),
 "normalization": control("Hidden normalization", ["none", "layer", "batch"], "none", "Layer uses LayerNorm in an MLP and one-group GroupNorm over channel/spatial values in a CNN (with channel-wise affine weights). Batch uses BatchNorm and running evaluation statistics.", "Normalization can stabilize optimization. Very small batches make batch norm noisy; the layer and spatial group variants are not identical parameterizations."),
 "clip": control("Maximum gradient norm", [0.0, 1.0, 5.0], 1.0, "Rescales a gradient if its total norm exceeds this value. Zero disables clipping.", "Clipping limits large updates; it does not repair a wrong loss or leaked data."),
 "schedule": control("Learning-rate schedule", ["constant", "cosine"], "constant", "Keep the rate fixed or reduce it by a cosine across epochs.", "A falling rate can refine fitting; a short schedule may reduce it too early."),
 "patience": control("Early-stopping patience", [0, 3, 5], 0, "Stop after this many epochs without better development loss. Zero runs the full budget.", "Uses development data only. The final model is the best development checkpoint even when early stopping is disabled."),
 "device": control("Local processor", ["auto", "cpu", "cuda", "mps"], "auto", "Auto checks NVIDIA CUDA, Apple Metal, then CPU using installed PyTorch. Classical ML always uses CPU.", "Small workloads may be slower on a GPU. Detection is not a guarantee that your batch fits in memory. This lab never starts a cloud job."),
}

ML_COMMON = ["train_limit", "seed", "scaling"]
DL_COMMON = ["train_limit", "seed", "epochs", "batch_size", "learning_rate", "optimizer", "weight_decay", "clip", "schedule", "patience", "device"]


def method(name, area, tasks, keys, meaning):
    return {"name": name, "area": area, "tasks": tasks, "controls": (ML_COMMON if area == "ml" else DL_COMMON) + keys, "meaning": meaning}


METHODS = {
 "dummy": method("Constant baseline", "ml", ["classification", "regression"], [], "Most frequent training class, or mean training rating. Checks whether learned features add value."),
 "logistic": method("Logistic regression", "ml", ["classification"], ["C"], "A linear score per class becomes normalized class probabilities. Minimize regularized log loss."),
 "ridge": method("Ridge regression", "ml", ["regression"], ["strength"], "A weighted feature sum fitted with squared error and an L2 penalty. λ=0 removes the penalty."),
 "lasso": method("Lasso regression", "ml", ["regression"], ["strength"], "Squared error plus an L1 penalty that can set coefficients to zero."),
 "elastic": method("Elastic net", "ml", ["regression"], ["strength", "l1_ratio"], "Combine L1 sparsity and L2 shrinkage in a linear predictor."),
 "knn": method("Nearest neighbours", "ml", ["classification"], ["neighbors", "neighbor_weights"], "Store training rows; classify a new row using nearby examples in the chosen feature space."),
 "svm": method("Support vector machine", "ml", ["classification", "regression"], ["C", "kernel", "gamma", "epsilon"], "Fit a margin-based decision rule or epsilon-insensitive regression; an RBF kernel allows nonlinear patterns."),
 "naive_bayes": method("Gaussian Naive Bayes", "ml", ["classification"], ["smoothing"], "Combine class-conditional Gaussian feature likelihoods under a conditional-independence assumption."),
 "tree": method("Decision tree", "ml", ["classification", "regression"], ["depth", "leaf"], "Repeatedly split feature values to create locally constant prediction regions."),
 "forest": method("Random forest", "ml", ["classification", "regression"], ["depth", "leaf", "trees"], "Average many randomized, bootstrapped trees to reduce individual-tree variance."),
 "boost": method("Histogram gradient boosting", "ml", ["classification", "regression"], ["depth", "leaf", "rounds", "boost_rate"], "Add trees sequentially to reduce the current loss. A fixed round budget is used here."),
 "kmeans": method("K-means", "ml", ["clustering"], ["clusters"], "Assign each row to a nearby fitted center, minimizing within-group squared distance. Labels never enter fitting."),
 "gmm": method("Gaussian mixture", "ml", ["clustering"], ["clusters"], "Fit a probabilistic mixture of Gaussian components by expectation maximization."),
 "dbscan": method("DBSCAN", "ml", ["clustering"], ["radius", "min_samples"], "Connect dense neighbourhoods and mark sparse points as noise. No native held-out prediction; fitting-set diagnostics only."),
 "pca": method("Principal component analysis", "ml", ["representation"], ["components"], "Fit high-variance directions on training rows, project and reconstruct separate rows. No target labels are used."),
 "isolation": method("Isolation Forest", "ml", ["anomaly"], ["contamination", "trees"], "Random partitions isolate unusual examples in fewer splits. Generated noise labels are available only for diagnostics."),
 "linear_head": method("Linear neural baseline", "dl", ["classification", "regression"], [], "One learned linear layer; a useful same-training-loop baseline, not a deep architecture."),
 "mlp": method("Multilayer perceptron", "dl", ["classification", "regression"], ["width", "layers", "activation", "dropout", "normalization"], "Alternate learned dense layers and nonlinear activations. Flatten images; standardize numeric features using training rows."),
 "cnn": method("Convolutional network", "dl", ["classification"], ["width", "activation", "dropout", "normalization"], "Two spatial convolution blocks share filters across image positions, pool locally, then classify the retained feature map."),
 "residual": method("Residual convolutional network", "dl", ["classification"], ["width", "activation", "dropout", "normalization"], "Add an identity shortcut around two same-shape convolutions so each block can learn a correction."),
 "autoencoder": method("Dense autoencoder", "dl", ["representation"], ["width", "activation"], "Compress images into a learned bottleneck and reconstruct the original pixel values. Targets are input pixels, not class labels."),
}


def control_keys(model, task):
    keys = list(METHODS[model]["controls"])
    if model == "svm" and task != "regression":
        keys.remove("epsilon")
    return keys


def catalog():
    import torch
    return {"methods": [{"id": k, **v} for k, v in METHODS.items()], "controls": CONTROLS,
            "datasets": model_data.catalog(), "hardware": hardware_info(),
            "scope": "Executable: tabular supervised models, clustering/PCA/anomaly diagnostics, dense networks, CNNs, residual CNNs and image autoencoders. RNNs, GNNs, diffusion, transfer learning and RL are course topics, not runnable trainers in this release.",
            "protocol": "One local worker. Fixed split seed independent of model seed. Train-only preprocessing. Development selects neural checkpoints; test never selects weights. Repeated test viewing is exploratory."}


def validate(settings, require_ready=True):
    if not isinstance(settings, dict) or set(settings) != {"area", "dataset", "task", "model", "controls"}:
        raise ValueError("Expected area, dataset, task, model and controls only.")
    area, dataset, task, model, selected = (settings[k] for k in ["area", "dataset", "task", "model", "controls"])
    if not all(isinstance(v, str) for v in [area, dataset, task, model]) or model not in METHODS or dataset not in model_data.CATALOG:
        raise ValueError("Unknown model, area or dataset.")
    spec = METHODS[model]
    if area != spec["area"] or task not in spec["tasks"]:
        raise ValueError("This method does not implement the requested task.")
    dataset_task = model_data.CATALOG[dataset]["task"]
    if task in {"classification", "regression"} and task != dataset_task:
        raise ValueError("This dataset has a different target type.")
    if task in {"clustering", "representation"} and dataset_task == "regression":
        raise ValueError("Choose a classification image/tabular dataset or the authored clusters.")
    if task == "anomaly" and dataset != "clusters":
        raise ValueError("Anomaly reference labels are available only in the authored noise fixture.")
    image = dataset in {"digits", "fashion_mnist"}
    if model in {"cnn", "residual", "autoencoder"} and not image:
        raise ValueError("This architecture requires an image dataset.")
    if not isinstance(selected, dict) or set(selected) - set(control_keys(model, task)):
        raise ValueError("An irrelevant or unknown control was supplied.")
    controls = {k: CONTROLS[k]["default"] for k in control_keys(model, task)}
    for key, value in selected.items():
        if not any(((type(value) is type(choice)) or (type(value) in {int,float} and type(choice) in {int,float})) and value == choice for choice in CONTROLS[key]["choices"]):
            raise ValueError("Unsupported value for " + key)
        controls[key] = value
    if model in {"lasso", "elastic"} and controls["strength"] == 0:
        raise ValueError("Use ridge with λ=0 for ordinary least squares; coordinate descent needs a positive penalty here.")
    if model == "pca" and controls["components"] > {"iris": 4, "clusters": 2}.get(dataset, 11 if dataset.startswith("wine_") else 64):
        raise ValueError("PCA components cannot exceed the input dimension.")
    # Quadratic kernel/distance workloads on 50k image rows are not a safe local default.
    if dataset == "fashion_mnist" and model in {"svm", "knn", "gmm", "dbscan"} and controls["train_limit"] > 5000:
        raise ValueError("This method is limited to 5,000 Fashion-MNIST fitting rows; use a scalable method for more.")
    if require_ready and not model_data.ready(dataset):
        raise ValueError("Prepare this dataset first; training never downloads silently.")
    return {"area": area, "dataset": dataset, "task": task, "model": model, "controls": controls}


def estimator(model, task, p):
    regression = task == "regression"
    if model == "dummy": return DummyRegressor() if regression else DummyClassifier(strategy="most_frequent")
    if model == "logistic": return LogisticRegression(C=p["C"], max_iter=1500, random_state=p["seed"])
    if model == "ridge": return Ridge(alpha=p["strength"])
    if model == "lasso": return Lasso(alpha=p["strength"], max_iter=3000)
    if model == "elastic": return ElasticNet(alpha=p["strength"], l1_ratio=p["l1_ratio"], max_iter=3000)
    if model == "knn": return KNeighborsClassifier(n_neighbors=p["neighbors"], weights=p["neighbor_weights"], n_jobs=1)
    if model == "svm":
        args = dict(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], cache_size=256)
        return SVR(**args, epsilon=p["epsilon"]) if regression else SVC(**args, probability=False, random_state=p["seed"])
    if model == "naive_bayes": return GaussianNB(var_smoothing=p["smoothing"])
    if model in {"tree", "forest"}:
        cls = {("tree", False): DecisionTreeClassifier, ("tree", True): DecisionTreeRegressor,
               ("forest", False): RandomForestClassifier, ("forest", True): RandomForestRegressor}[model, regression]
        args = dict(max_depth=p["depth"], min_samples_leaf=p["leaf"], random_state=p["seed"])
        if model == "forest": args.update(n_estimators=p["trees"], n_jobs=1)
        return cls(**args)
    if model == "boost":
        cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
        return cls(max_iter=p["rounds"], learning_rate=p["boost_rate"], max_depth=p["depth"], min_samples_leaf=p["leaf"], early_stopping=False, random_state=p["seed"])
    if model == "kmeans": return KMeans(n_clusters=p["clusters"], n_init=10, random_state=p["seed"])
    if model == "gmm": return GaussianMixture(n_components=p["clusters"], reg_covar=1e-5, random_state=p["seed"])
    if model == "dbscan": return DBSCAN(eps=p["radius"], min_samples=p["min_samples"], n_jobs=1)
    if model == "pca": return PCA(n_components=p["components"], svd_solver="full")
    if model == "isolation": return IsolationForest(n_estimators=p["trees"], contamination=p["contamination"], n_jobs=1, random_state=p["seed"])
    raise ValueError("No classical estimator for this method.")


def scores(y, pred, task, probabilities=None, labels=None):
    if task == "regression":
        return {"mae": float(metrics.mean_absolute_error(y, pred)), "rmse": float(np.sqrt(metrics.mean_squared_error(y, pred))), "r2": float(metrics.r2_score(y, pred))}
    result = {"accuracy": float(metrics.accuracy_score(y, pred)), "macro_f1": float(metrics.f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "balanced_accuracy": float(metrics.balanced_accuracy_score(y, pred))}
    if probabilities is not None:
        prob = np.clip(probabilities, 1e-12, 1); prob /= prob.sum(axis=1, keepdims=True)
        result["log_loss"] = float(metrics.log_loss(y, prob, labels=labels))
        one_hot = np.eye(len(labels))[np.searchsorted(labels, y)]
        result["brier_multiclass"] = float(np.mean(np.sum((prob - one_hot)**2, axis=1)))
        confidence = prob.max(axis=1); correct = np.asarray(pred) == y
        result["ece_10_bins"] = float(sum(np.mean(mask) * abs(correct[mask].mean() - confidence[mask].mean()) for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]) if (mask := (confidence > a) & (confidence <= b)).any()))
    return result


def interval(y, pred, task, seed=44, groups=None):
    # A conditional uncertainty illustration, not a substitute for repeat training/group-aware inference.
    rng = np.random.default_rng(seed); values = []
    units=None if groups is None else [np.flatnonzero(groups==g) for g in np.unique(groups)]
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y)) if units is None else np.concatenate([units[i] for i in rng.integers(0,len(units),len(units))])
        values.append(float(np.mean(np.abs(y[idx] - pred[idx]))) if task == "regression" else float(np.mean(y[idx] == pred[idx])))
    low, high = np.quantile(values, [.025, .975])
    return {"metric": "mae" if task == "regression" else "accuracy", "low": float(low), "high": float(high), "resamples": 200,
            "resampling_unit":"row" if units is None else "duplicate-feature group",
            "note": "95% percentile bootstrap conditional on this fitted model. "+("Exact duplicate-feature groups are sampled together." if units is not None else "Rows are sampled independently; unmodelled group/time dependence needs a different resampler.")+" It omits retraining, model-selection and split uncertainty."}


def _prediction_artifact(data, outputs, task):
    result = {}; labels = np.unique(data["y"]) if task == "classification" else None
    for split, output in outputs.items():
        ids = data["indices"][split]; y = data["y"][ids]; pred, prob = output
        result[split] = {"row_ids": ids.tolist(), "truth": y.tolist(), "prediction": pred.tolist(),
                         "scores": scores(y, pred, task, prob, labels)}
        if task == "classification":
            result[split]["confusion"] = metrics.confusion_matrix(y, pred, labels=labels).tolist()
            result[split]["labels"] = labels.tolist()
            precision,recall,f1,support=metrics.precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
            result[split]["per_class"]=[{"label":int(label),"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"support":int(support[i])} for i,label in enumerate(labels)]
            if prob is not None:
                result[split]["confidence"] = prob.max(axis=1).tolist()
        if split == "test": result[split]["interval"] = interval(y, pred, task, groups=None if data.get("groups") is None else data["groups"][ids])
    return result


def _classical(settings, data, directory, update):
    p = settings["controls"]; task = settings["task"]; name = settings["model"]
    transform = {"none": None, "standard": StandardScaler(), "robust": RobustScaler()}[p["scaling"]]
    train = data["indices"]["train"]; x, y = data["x"], data["y"]
    update(phase="Fitting training-only preprocessing and estimator")
    model = estimator(name, task, p)
    pipeline = Pipeline([("scale", transform or "passthrough"), ("model", model)])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter("always")
        pipeline.fit(x[train], y[train] if task in {"classification", "regression"} else None)
        if task in {"classification", "regression"}:
            outputs = {}
            update(phase="Evaluating fixed development and test splits")
            for split in ["train", "dev", "test"]:
                values = x[data["indices"][split]]
                prob = pipeline.predict_proba(values) if hasattr(model, "predict_proba") and task == "classification" else None
                outputs[split] = (pipeline.predict(values), prob)
            result = {"predictions": _prediction_artifact(data, outputs, task)}
        else:
            scaled = pipeline[:-1].transform(x[train])
            reference = y[train]
            if name == "pca":
                result = {"representation": {"explained_variance_ratio": model.explained_variance_ratio_.tolist(), "projection": model.transform(scaled[:500])[:, :2].tolist(), "labels": reference[:500].tolist(), "reconstruction_mse": {}}}
                for split, ids in data["indices"].items():
                    z = pipeline[:-1].transform(x[ids]); restored = model.inverse_transform(model.transform(z))
                    result["representation"]["reconstruction_mse"][split] = float(np.mean((z - restored)**2))
                result["representation"]["units"] = "Squared error in the chosen preprocessed feature units; not necessarily pixel units."
            elif name == "isolation":
                result = {"anomaly": {}}
                for split, ids in data["indices"].items():
                    score = -pipeline.decision_function(x[ids]); flagged = pipeline.predict(x[ids]) == -1
                    result["anomaly"][split] = {"roc_auc": float(metrics.roc_auc_score(y[ids] == -1, score)), "flagged_fraction": float(flagged.mean())}
            else:
                assignments = model.labels_ if name == "dbscan" else model.predict(scaled)
                unique = np.unique(assignments); meaningful = assignments != -1
                silhouette = None
                if meaningful.sum() > 2 and 1 < len(np.unique(assignments[meaningful])) < meaningful.sum():
                    silhouette = float(metrics.silhouette_score(scaled[meaningful], assignments[meaningful], sample_size=min(1000, meaningful.sum()), random_state=11))
                projection = PCA(n_components=2).fit_transform(scaled) if scaled.shape[1] > 2 else scaled
                result = {"clustering": {"fit_silhouette_excluding_noise": silhouette, "fit_adjusted_rand_reference": float(metrics.adjusted_rand_score(reference, assignments)), "noise_fraction": float(np.mean(assignments == -1)), "groups_excluding_noise": int(sum(k != -1 for k in unique)), "projection": projection[:500].tolist(), "assignments": assignments[:500].tolist(), "note": "Fitting-set geometry and reference-label diagnostics, not supervised test accuracy. Labels were withheld from fitting; a 2D projection can hide distances."}}
                if name != "dbscan":
                    result["clustering"]["test_adjusted_rand_reference"] = float(metrics.adjusted_rand_score(y[data["indices"]["test"]], pipeline.predict(x[data["indices"]["test"]])) )
        result["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    # Private, locally generated persistence only. Never loaded through a web request.
    import joblib
    joblib.dump(pipeline, directory / "estimator.joblib", compress=3)
    result["estimator_parameters"]=model.get_params(deep=False)
    result["fitted_preprocessing"]={"kind":p["scaling"],"center":None if transform is None else getattr(transform,"mean_",getattr(transform,"center_",np.array([]))).tolist(),"scale":None if transform is None else transform.scale_.tolist()}
    result["persistence_note"] = "Trusted local sklearn/joblib artifact. Pickle-based files can execute code when loaded; never load a file from an untrusted source. The web API does not deserialize it."
    if hasattr(model, "feature_importances_"):
        result["feature_importance"] = model.feature_importances_.tolist()
        result["importance_note"] = "Training impurity-based importance; correlated/high-cardinality features can distort this. Not a causal explanation."
    elif hasattr(model, "coef_"):
        result["coefficients"] = np.asarray(model.coef_).tolist()
        result["intercept"] = np.asarray(model.intercept_).tolist()
    return result


def network(name, p, features, outputs, image_shape):
    import torch
    from torch import nn
    activation = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[p.get("activation", "relu")]
    width = p.get("width", 64)
    def norm(n, spatial=False):
        kind = p.get("normalization", "none")
        if kind == "batch": return nn.BatchNorm2d(n) if spatial else nn.BatchNorm1d(n)
        if kind == "layer": return nn.GroupNorm(1, n) if spatial else nn.LayerNorm(n)
        return nn.Identity()
    if name == "linear_head": return nn.Sequential(nn.Flatten(), nn.Linear(features, outputs))
    if name == "mlp":
        layers = [nn.Flatten()]; previous = features
        for _ in range(p["layers"]):
            layers.extend([nn.Linear(previous, width), norm(width), activation(), nn.Dropout(p["dropout"])])
            previous = width
        return nn.Sequential(*layers, nn.Linear(width, outputs))
    if name == "autoencoder":
        return nn.Sequential(nn.Flatten(), nn.Linear(features, width), activation(), nn.Linear(width, max(2, width // 4)), activation(), nn.Linear(max(2, width // 4), width), activation(), nn.Linear(width, features), nn.Sigmoid())
    if not image_shape: raise ValueError("Convolution requires spatial images.")
    class Residual(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.path = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), norm(channels, True), activation(), nn.Conv2d(channels, channels, 3, padding=1), norm(channels, True))
            self.act = activation()
        def forward(self, x): return self.act(x + self.path(x))
    side = image_shape[-1] // 4
    blocks = [nn.Unflatten(1, tuple(image_shape)), nn.Conv2d(1, width, 3, padding=1), norm(width, True), activation()]
    if name == "residual": blocks.append(Residual(width))
    blocks.extend([nn.MaxPool2d(2), nn.Conv2d(width, width * 2, 3, padding=1), norm(width * 2, True), activation()])
    if name == "residual": blocks.append(Residual(width * 2))
    blocks.extend([nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(p["dropout"]), nn.Linear(width * 2 * side * side, outputs)])
    return nn.Sequential(*blocks)


def _neural(settings, data, directory, update):
    import torch
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    p = settings["controls"]; task = settings["task"]; ids = data["indices"]
    torch.set_num_threads(2); torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    device = choose_device(p["device"])
    x = data["x"].copy(); y = data["y"].copy(); shape = data["info"]["image_shape"]
    preprocess = {"kind": "fixed image range 0–1"}
    if shape is None:
        scaler = StandardScaler().fit(x[ids["train"]]); x = scaler.transform(x).astype(np.float32)
        preprocess = {"kind": "training-only standardization", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    target_mean, target_scale = 0.0, 1.0
    if task == "regression":
        target_mean, target_scale = float(y[ids["train"]].mean()), max(float(y[ids["train"]].std()), 1e-6)
        y_train = ((y - target_mean) / target_scale).astype(np.float32)
    else: y_train = y
    labels = np.unique(y) if task == "classification" else None
    n_outputs = len(labels) if labels is not None else 1
    model = network(settings["model"], p, x.shape[1], n_outputs, shape).to(device)
    count = sum(t.numel() for t in model.parameters())
    if count > 3_000_000: raise ValueError("This local configuration exceeds three million parameters.")
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y_train.astype(np.int64 if task == "classification" else np.float32))
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]) if p["optimizer"] == "adamw" else torch.optim.SGD(model.parameters(), lr=p["learning_rate"], momentum=.9, weight_decay=p["weight_decay"])
    history = []; best = math.inf; best_state = None; best_epoch = 0; stale = 0; updates = 0
    writer = SummaryWriter(str(directory / "tensorboard"))
    def evaluate(split):
        model.eval(); predictions = []; total = 0.; seen = 0
        with torch.no_grad():
            for start in range(0, len(ids[split]), 256):
                rows = ids[split][start:start+256]; bx = x_tensor[rows].to(device)
                target = bx if task == "representation" else y_tensor[rows].to(device)
                output = model(bx)
                if task == "regression": output = output.squeeze(-1)
                loss = criterion(output, target)
                total += float(loss) * len(rows); seen += len(rows)
                predictions.append(output.detach().cpu().numpy())
        return total / seen, np.concatenate(predictions)
    update(phase="Training neural network", model_parameters=count, architecture=str(model), hardware={"device": device}, versions={"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"torch":str(torch.__version__)}, preprocessing=preprocess,
           target_transform={"mean": target_mean, "scale": target_scale}, requested_epochs=p["epochs"], completed_epochs=0)
    started = time.monotonic()
    try:
        for epoch in range(1, p["epochs"] + 1):
            model.train(); total = 0.; seen = 0; norms = []
            lr = p["learning_rate"] * (.5 * (1 + math.cos(math.pi * (epoch-1) / p["epochs"])) if p["schedule"] == "cosine" else 1)
            for group in optimizer.param_groups: group["lr"] = lr
            order = torch.randperm(len(ids["train"])).numpy(); batches = [order[i:i+p["batch_size"]] for i in range(0, len(order), p["batch_size"])]
            if len(batches) > 1 and len(batches[-1]) == 1 and p.get("normalization") == "batch":
                batches[-2] = np.concatenate([batches[-2], batches[-1]]); batches.pop()
            for batch_no, batch in enumerate(batches):
                rows = ids["train"][batch]; bx = x_tensor[rows].to(device)
                target = bx if task == "representation" else y_tensor[rows].to(device)
                optimizer.zero_grad(set_to_none=True); output = model(bx)
                if task == "regression": output = output.squeeze(-1)
                loss = criterion(output, target)
                if not torch.isfinite(loss): raise ValueError("Loss became non-finite; lower the learning rate or inspect the configuration.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p["clip"] if p["clip"] else math.inf)
                if not torch.isfinite(norm): raise ValueError("Gradient became non-finite.")
                norms.append(float(norm)); optimizer.step(); updates += 1
                total += float(loss.detach()) * len(rows); seen += len(rows)
                if batch_no % 10 == 0:
                    update(phase=f"Epoch {epoch}: batch {batch_no+1}/{len(batches)}", updates=updates, epoch_fraction=(epoch-1)+(batch_no+1)/len(batches), elapsed_seconds=time.monotonic()-started)
            dev_loss, _ = evaluate("dev"); improved = dev_loss < best - 1e-8
            if improved:
                best, best_epoch, stale = dev_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else: stale += 1
            row = {"epoch": epoch, "train_loss": total/seen, "dev_loss": dev_loss, "learning_rate": lr, "mean_gradient_norm_before_clipping": float(np.mean(norms)), "elapsed_seconds": time.monotonic()-started, "updates": updates}
            history.append(row); write_json(directory / "history.json", history)
            for key in ["train_loss", "dev_loss", "learning_rate", "mean_gradient_norm_before_clipping"]: writer.add_scalar(key, row[key], epoch)
            writer.flush()
            update(phase=f"Completed epoch {epoch}", completed_epochs=epoch, epoch_fraction=epoch, best_epoch=best_epoch, elapsed_seconds=row["elapsed_seconds"], updates=updates)
            if p["patience"] and stale >= p["patience"]: break
        model.load_state_dict(best_state); model.to(device)
        torch.save({"state_dict": best_state, "settings": settings, "features": x.shape[1], "outputs": n_outputs, "image_shape": shape,
                    "preprocessing": preprocess, "target_transform": {"mean": target_mean, "scale": target_scale}}, directory / "weights.pt")
        update(phase="Evaluating the best development checkpoint")
        result = {"history": history, "best_epoch": best_epoch, "parameters": count, "device": device,
                  "loss_units": "Class cross-entropy (natural-log units)" if task == "classification" else "Mean squared pixel error in 0–1 units" if task == "representation" else "Mean squared error of standardized training targets; final MAE/RMSE are restored to original target units"}
        if task in {"classification", "regression"}:
            outputs = {}
            for split in ids:
                _, raw = evaluate(split)
                if task == "classification":
                    shifted = raw - raw.max(axis=1, keepdims=True); prob = np.exp(shifted); prob /= prob.sum(axis=1, keepdims=True)
                    pred = labels[prob.argmax(axis=1)]
                else: pred, prob = raw * target_scale + target_mean, None
                outputs[split] = (pred, prob)
            result["predictions"] = _prediction_artifact(data, outputs, task)
        else:
            result["representation"] = {"reconstruction_mse": {}, "units": "Mean squared pixel error in fixed 0–1 units."}
            for split in ids:
                value, reconstructed = evaluate(split)
                result["representation"]["reconstruction_mse"][split] = value
                if split == "test":
                    result["image_examples"] = [{"row_id": int(idx), "truth": int(y[idx]), "pixels": x[idx].tolist(), "reconstruction": reconstructed[i].tolist()} for i, idx in enumerate(ids[split][:8])]
            mean_image = x[ids["train"]].mean(axis=0)
            result["representation"]["constant_image_test_mse"] = float(np.mean((x[ids["test"]] - mean_image)**2))
        return result
    finally:
        writer.close()


def fit(identifier, settings, root=RUNS, data_root=model_data.DATA):
    settings = validate(settings); directory = run_path(identifier, root)
    if directory.exists(): raise ValueError("Run already exists; completed runs are immutable.")
    directory.mkdir(parents=True)
    manifest = {"id": identifier, "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
                "phase": "Loading and checking local data", "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
                "source_hashes": source_hashes()}
    started = time.monotonic()
    def update(**values):
        manifest.update(values); write_json(directory / "manifest.json", manifest)
    update()
    try:
        data = model_data.load(settings["dataset"], settings["controls"]["train_limit"], data_root)
        write_json(directory / "data.json", data["info"]); write_json(directory / "splits.json", data["split_ids"])
        update(data_fingerprint=data["info"]["fingerprint"], split_fingerprint=data["info"]["split_fingerprint"])
        result = _classical(settings, data, directory, update) if settings["area"] == "ml" else _neural(settings, data, directory, update)
        task = settings["task"]
        if task in {"classification", "regression"}:
            baseline = DummyRegressor() if task == "regression" else DummyClassifier(strategy="most_frequent")
            ids = data["indices"]; baseline.fit(data["x"][ids["train"]], data["y"][ids["train"]])
            result["baseline"] = {split: scores(data["y"][rows], baseline.predict(data["x"][rows]), task, labels=np.unique(data["y"]) if task == "classification" else None) for split, rows in ids.items()}
            # Deterministic, unedited test examples, including model errors when present.
            test = result["predictions"]["test"]; truth = np.array(test["truth"]); pred = np.array(test["prediction"])
            ranked = np.argsort(-np.abs(truth-pred)) if task == "regression" else np.concatenate([np.flatnonzero(truth != pred), np.flatnonzero(truth == pred)])
            result["examples"] = [{"row_id": int(ids["test"][i]), "truth": float(truth[i]), "prediction": float(pred[i]),
                                   "features": data["x"][ids["test"][i]].tolist()} for i in ranked[:12]]
        write_json(directory / "result.json", result)
        update(status="MEASURED", phase="Completed", elapsed_seconds=time.monotonic()-started, completed_at=datetime.now(timezone.utc).isoformat())
        seal(directory)
    except BaseException as exc:
        update(status="CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAILED", phase="Stopped", error=str(exc), elapsed_seconds=time.monotonic()-started)
        if not isinstance(exc, KeyboardInterrupt): raise
    return manifest


def listing(root=RUNS):
    rows = []
    if Path(root).exists():
        for path in Path(root).iterdir():
            if path.is_dir() and not path.is_symlink() and (path / "manifest.json").is_file():
                try:
                    m = read_json(path / "manifest.json")
                    rows.append({k: m.get(k) for k in ["id", "status", "started_at", "settings", "elapsed_seconds", "phase"]})
                except (OSError, ValueError): pass
    return sorted(rows, key=lambda r: r["started_at"], reverse=True)


def detail(identifier, root=RUNS):
    directory = run_path(identifier, root); manifest = read_json(directory / "manifest.json")
    result = {"manifest": manifest}
    for name in ["history", "data", "result"]:
        path = directory / (name + ".json")
        if path.is_file() and not path.is_symlink(): result[name] = read_json(path)
    if result.get("data",{}).get("id") in {"wine_red","wine_white"}:
        old=result.get("result",{}).get("predictions",{}).get("test",{}).get("interval",{})
        if old and "resampling_unit" not in old:
            result["uncertainty_warning"]="Historical pilot uses a row bootstrap despite duplicate-feature groups. Treat its interval as illustrative only; new Wine Quality runs resample whole duplicate-feature groups. Saved artifacts remain unchanged."
    result["integrity"] = "Not sealed while incomplete"
    if manifest["status"] == "MEASURED":
        try: verify(directory); result["integrity"] = "PASS"
        except (OSError, ValueError) as exc: result["integrity"] = "FAIL: " + str(exc)
    return result


def compare(identifiers, root=RUNS):
    if not isinstance(identifiers, list) or not 2 <= len(identifiers) <= 4 or any(not isinstance(i, str) for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Choose two to four different runs.")
    rows = []; signature = None; warnings_out = []
    for identifier in identifiers:
        run = detail(identifier, root); m = run["manifest"]; s = m["settings"]
        if m["status"] != "MEASURED" or run["integrity"] != "PASS": raise ValueError("Only completed, integrity-verified runs can be compared.")
        current = (s["dataset"], s["task"], m["data_fingerprint"], m["split_fingerprint"])
        if signature is None: signature = current
        if current != signature: raise ValueError("Comparison requires the same task, dataset bytes and exact training/development/test rows. Match the training-row limit first.")
        r = run["result"]
        if "predictions" in r:
            values = {"dev": r["predictions"]["dev"]["scores"], "test": r["predictions"]["test"]["scores"]}
        elif "representation" in r:
            rep=r["representation"]
            values = {"reconstruction_mse":rep["reconstruction_mse"],"units":rep["units"]}
            if "explained_variance_ratio" in rep:values["total_variance_retained"]=sum(rep["explained_variance_ratio"])
            warnings_out.append("PCA reconstruction can use standardized feature units; autoencoders use fixed pixel units. Do not rank their errors unless units match.")
        elif "clustering" in r:values={k:v for k,v in r["clustering"].items() if k not in {"projection","assignments"}}
        else: values = r.get("anomaly", {})
        rows.append({"id": identifier, "settings": s, "scores": values, "seconds": m["elapsed_seconds"], "parameters": r.get("parameters"), "history": run.get("history", [])})
    warnings_out += ["These are exploratory local comparisons, not claims that any algorithm wins everywhere.", "Check preprocessing, fitting budget, seed, checkpoint-selection rule and uncertainty before attributing differences to architecture."]
    return {"rows": rows, "warnings": list(dict.fromkeys(warnings_out)), "dataset": signature[0], "task": signature[1]}
