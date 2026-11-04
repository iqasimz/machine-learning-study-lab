# ML → DL → LLMs

## Starting point

Review existing [software/tensors](../pre_llm/01_COMPUTERS_TENSORS_AND_SOFTWARE.md), [linear algebra/calculus](../pre_llm/02_LINEAR_ALGEBRA_AND_CALCULUS.md) and [probability](../pre_llm/03_PROBABILITY_INFORMATION_AND_STATISTICS.md). You need to multiply a small matrix, read a Python function and interpret a conditional probability—not already know every algorithm.

The IDs below are stable integration references. Each lesson teaches a mechanism, calculates a small example, compares alternatives and sets a check. No test is automatically passed because these files exist.

## Machine learning

| Lesson | Main question resolved |
| --- | --- |
| [ML01 · Problems, data and baselines](ml/01_PROBLEMS_DATA_AND_BASELINES.md) | Define available inputs and target before fitting. |
| [ML02 · Features and preprocessing](ml/02_FEATURES_AND_PREPROCESSING.md) | Fit preprocessing within each training fold. |
| [ML03 · Linear regression](ml/03_LINEAR_REGRESSION.md) | Derive squared-loss gradients and least squares. |
| [ML04 · Logistic and multiclass classification](ml/04_LOGISTIC_AND_MULTICLASS_CLASSIFICATION.md) | Calculate log loss and a threshold decision. |
| [ML05 · Regularization and model selection](ml/05_REGULARIZATION_AND_MODEL_SELECTION.md) | Explain bias/variance and selection error. |
| [ML06 · Neighbours, kernels and SVMs](ml/06_NEIGHBOURS_KERNELS_AND_SVM.md) | Compute neighbour votes and a margin penalty. |
| [ML07 · Probabilistic classifiers](ml/07_PROBABILISTIC_CLASSIFIERS.md) | Calculate a naive Bayes posterior ratio. |
| [ML08 · Trees and random forests](ml/08_TREES_AND_RANDOM_FORESTS.md) | Calculate a split's impurity gain. |
| [ML09 · Boosting and model combinations](ml/09_BOOSTING_AND_ENSEMBLES.md) | Explain additive fitting versus independent bagging. |
| [ML10 · Metrics, calibration and uncertainty](ml/10_METRICS_CALIBRATION_AND_UNCERTAINTY.md) | Calculate class metrics and paired uncertainty. |
| [ML11 · Clustering and mixture models](ml/11_CLUSTERING_AND_MIXTURES.md) | Carry out a k-means assignment/update. |
| [ML12 · Dimension reduction and anomaly detection](ml/12_DIMENSION_REDUCTION_AND_ANOMALIES.md) | Calculate PCA projection and reconstruction. |
| [ML13 · Time series and forecast evaluation](ml/13_TIME_SERIES_AND_FORECASTING.md) | Build causal lag features and a seasonal baseline. |
| [ML14 · Recommendation and sequential decisions](ml/14_RECOMMENDATION_AND_SEQUENTIAL_DECISIONS.md) | Calculate a factorized recommendation score. |
| [ML15 · Generalization and probabilistic prediction](ml/15_LEARNING_THEORY_AND_PROBABILISTIC_PREDICTION.md) | Interpret a finite-class generalization bound. |
| [ML16 · A complete ML experiment](ml/16_COMPLETE_ML_EXPERIMENT.md) | Design and defend a reproducible baseline comparison. |

## Deep learning

| Lesson | Main question resolved |
| --- | --- |
| [DL01 · From linear models to neural networks](dl/01_FROM_LINEAR_MODELS_TO_MLPS.md) | Show why stacked linear maps remain linear. |
| [DL02 · Backpropagation and autograd](dl/02_BACKPROPAGATION_AND_AUTOGRAD.md) | Compute a complete chain-rule update. |
| [DL03 · Optimization, initialization and normalization](dl/03_OPTIMIZATION_AND_INITIALIZATION.md) | Trace SGD/momentum/Adam update state. |
| [DL04 · Regularization and training diagnostics](dl/04_REGULARIZATION_AND_TRAINING_DIAGNOSTICS.md) | Diagnose underfitting, overfitting and numerical faults. |
| [DL05 · Convolutional networks](dl/05_CONVOLUTIONAL_NETWORKS.md) | Calculate output shape, weights and receptive field. |
| [DL06 · Residual networks and transfer learning](dl/06_RESIDUAL_NETWORKS_AND_TRANSFER.md) | Trace a skip connection and projection. |
| [DL07 · Recurrent sequence models](dl/07_RECURRENT_SEQUENCE_MODELS.md) | Trace recurrent state and truncated backpropagation. |
| [DL08 · Attention and Transformers](dl/08_ATTENTION_AND_TRANSFORMERS.md) | Calculate a masked attention row. |
| [DL09 · Autoencoders and self-supervised learning](dl/09_AUTOENCODERS_AND_SELF_SUPERVISION.md) | Calculate reconstruction loss and contrastive scoring. |
| [DL10 · Generative model families](dl/10_GENERATIVE_MODEL_FAMILIES.md) | Compare VAE, GAN and diffusion objectives. |
| [DL11 · Graph neural networks](dl/11_GRAPH_NEURAL_NETWORKS.md) | Calculate one permutation-invariant message-passing update. |
| [DL12 · Task heads and structured prediction](dl/12_TASK_HEADS_AND_LOSSES.md) | Match output shapes and losses to task semantics. |
| [DL13 · Training systems and deployment](dl/13_TRAINING_SYSTEMS_AND_DEPLOYMENT.md) | Account for model/gradient/optimizer/activation memory. |
| [DL14 · Comparing ML and DL fairly](dl/14_ML_VERSUS_DL_CAPSTONE.md) | Run a matched method comparison without assuming a winner. |

## Transition into the existing LLM course

DL02 connects to [P2](../pre_llm/02_LINEAR_ALGEBRA_AND_CALCULUS.md); DL03/DL04 connect to [the full training loop T0](../llm_training/00_THE_COMPLETE_TRAINING_LOOP.md). DL08 introduces why attention helps and then uses [T3](../llm_training/03_TRANSFORMER_FROM_FIRST_PRINCIPLES.md) for the full decoder details. DL06's transfer-learning comparison prepares the distinction between scratch language training and [R1 adaptation](../post_training/01_SFT_AND_PARAMETER_EFFICIENT_ADAPTATION.md).

The website sidebar follows Shared foundations → ML → DL → NLP → LLMs. Tokenization/embedding foundations have one owner under NLP; Transformer internals and LLM-specific adaptation/serving retain their existing homes. General deployment belongs with the ML/DL experiment pipeline; LLM-specific additions link back rather than repeat it.

## Advanced study inside the existing lectures

The basic lecture remains the entry point. Its **Advanced study** guide entries extend that topic; they do not add a compulsory new top-level course. Each section starts with a definition, follows inputs through the mechanism, works a numerical example and ends with practice/limitations. All numbers are teaching examples, not new benchmark results.

| Route | Location and order |
| --- | --- |
| Probability | ML07: Bayesian update → probabilistic graphs. Then ML15: Monte Carlo/MCMC versus variational inference. Return here before VAE derivations if needed. |
| Recommendation | ML14: factorization training → two-tower retrieval/ranking → exposure and evaluation boundaries. |
| Decision learning | Continue within ML14: contextual bandits → MDP/Bellman/Q-learning → policy gradient/actor–critic. Then R2 for PPO and the distinct LLM preference methods. |
| Vision | After DL08 attention, revisit DL05 for ViT patch/feature axes, then DL06 for contrastive image–text transfer. DL12 adds detection → segmentation → tracking. |
| Generation | After DL09 and probability: DL10 VAE → GAN → diffusion training → sampling/guidance → latent diffusion → evaluation. |
| Graphs | DL11 GCN foundation → GraphSAGE → GAT → graph/link tasks → graph Transformers; graph access/leakage limits accompany the task. |
| Responsible systems | ML16 group metrics/oversight → robustness/release documentation. DL13 export/full-pipeline tests → monitoring/recovery. R8 retains the additional LLM/tool-specific issues. |

These are learning extensions, not newly implemented model-training families. Search/symbolic reasoning, causal-inference coursework and robotics are deliberately outside this update. There is no meaningful “90% of AI” mastery claim; the goal is to explain a mechanism, reproduce its example and identify an appropriate test.

## Workload and depth

No arbitrary week count is assigned. A lesson is complete when the learner can explain its inputs and outputs, reproduce its hand calculation, implement or inspect its core operation, diagnose a planted error and defend the comparison. First-course coverage is broad; [specialist limits](COVERAGE_AND_LIMITS.md) remain explicit.
