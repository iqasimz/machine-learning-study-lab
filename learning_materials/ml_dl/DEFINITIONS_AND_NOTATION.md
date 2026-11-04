# Definitions and notation

Use this as a compact reference, not a substitute for definitions immediately before a lesson's equations. The new lessons define terms locally; controls have their own [cards](metadata/parameters.json).

## Data and decisions

- **Example/sample:** one observation under a stated unit, such as an image or a document. Rows, people, pixels and measurements are different units.
- **Feature/input x:** information available to the model at prediction time.
- **Target y:** desired quantity or label; its definition is part of the task.
- **Label:** a category or annotation supplied by a rule, annotator or data-generating process; not automatically objective truth.
- **Dataset:** a collection plus schema, provenance and scope.
- **Training/development/test:** fitting, selection and reserved final assessment roles. A set repeatedly used for selection is no longer an independent final check.
- **Leakage:** unintended access to target, future or evaluation information during fitting/selection.
- **Baseline:** a simple or established comparison using the same allowed information.
- **Prediction:** output from a fitted rule; **decision:** action chosen from it under a threshold/cost policy.
- **Supervised / unsupervised / self-supervised / reinforcement learning:** labelled targets, structure without external labels, targets derived from data, and feedback from selected behaviour respectively. These families can be combined; they do not have identical evidence requirements.

## Model and fitting

- **Parameter:** number learned in fitting, such as a coefficient or neural weight.
- **Hyperparameter:** configuration chosen outside the inner fit, such as tree depth or learning rate.
- **Architecture:** permitted computational structure and shapes.
- **Representation:** numerical description used for prediction; may be engineered or learned.
- **Inductive bias:** assumptions/preferences making some rules easier or more plausible to fit.
- **Loss:** penalty used for fitting; **metric:** quantity used to report a defined outcome. They need not be identical.
- **Objective:** loss aggregation plus declared penalties/constraints.
- **Regularization:** restriction or preference discouraging unsuitable fitting; not every method shares one numeric strength scale.
- **Generalization:** useful performance beyond fitting examples under specified conditions.
- **Calibration:** agreement between probabilistic confidence and observed frequencies, not merely correct ranking.
- **Causality:** consequences of interventions; observational predictive associations do not identify these by themselves.

## Neural computation

- **Logit:** unrestricted output score; **probability:** nonnegative normalized mass under a specified model.
- **Activation:** computed intermediate value; also, by context, the nonlinear activation function.
- **Gradient:** local derivative of scalar loss with respect to a variable.
- **Forward / backward / update:** calculate predictions, calculate gradients, then change parameters.
- **Batch / microbatch:** examples processed together; **accumulation:** combine small-batch gradients before one update.
- **Epoch:** full pass through a specified dataset. Do not call a sampled window count an exact epoch without justification.
- **Optimizer state:** stored update history, such as Adam moments.
- **Checkpoint:** saved model state; **resumable checkpoint:** additionally contains the compatible optimizer/schedule/random/data position.
- **Inference:** using fitted weights, normally without updates.
- **Pretraining / fine-tuning:** initial broad fitting and subsequent adaptation; their exact objectives/data must still be stated.
- **CPU / GPU / MPS:** processor/backend choices; MPS is Apple's accelerator interface and not NVIDIA CUDA.

## Mathematics conventions

| Symbol | Meaning | Shape or unit |
| --- | --- | --- |
| n | Examples | Count under a declared unit |
| d | Input features | Features per example |
| X | Feature matrix | [n,d], rows are examples |
| x_i | One example | d features; dot-product notation may write it as a column |
| y / y-hat | Target / prediction | Task-specific |
| w, b | Coefficients, intercept | Shapes match the map |
| L, ell | Aggregated loss, one-case loss | Units depend on objective |
| theta | All trainable parameters | A structured collection, not one vector shape in every implementation |
| eta | Learning rate | Meaning depends on optimizer and loss scaling |
| lambda | Penalty coefficient | Objective-specific |
| C | Common inverse-regularization control | Estimator-specific, not universally 1/lambda |
| B,T,H,W | Batch, time, image height/width | Context-dependent; always redefined locally |
| mu, sigma² | Mean, variance | Do not confuse variance with standard deviation |
| ||x||₂ | Euclidean vector length | sqrt(sum coordinates²) |
| x·z | Dot product | Sum of coordinate products |
| A^T | Matrix transpose | Swaps axes |
| @ or adjacent matrices | Matrix product | Matching inner dimensions sum out |
| elementwise product | Coordinate-wise multiplication | Different from matrix contraction |
| E | Expectation | Probability-weighted average |
| log, ln | Natural logarithm unless otherwise declared | Loss unit nats; base-2 logs give bits |

Greek letters are not global variable names. Gamma can mean an RBF locality control or an RL discount; alpha can mean a penalty or a miscoverage target. The lesson/control must define it before using it.

## Reading a calculation

Identify the quantities and shapes, substitute a small example, perform one operation at a time, check the output's units and shape, then ask which next operation consumes it. A loss is not a parameter; a gradient is not an update; a model probability is not a guarantee of factual correctness.
