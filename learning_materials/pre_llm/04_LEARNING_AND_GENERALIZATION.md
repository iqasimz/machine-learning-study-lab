# P4. Training, validation and generalization

A model learns from training examples, but we want it to work on new examples. That is generalization. To check it honestly, separate data used to learn weights, data used to choose settings, and data reserved for the final evaluation.

With 100 labelled cases, we keep 10 for testing. The control changes how many of the other 90 are used for fitting and how many remain for development, also called validation.

This is a study plan, not a model architecture. The complete-experiment lesson later applies these boundaries to a complete experiment with saved settings, predictions and results.

## 1. Learning from data

**Key terms.** A **model** is a parameterized prediction rule; an **objective** measures what training tries to improve; **generalization** is performance beyond fitting examples.

**Process.** Examples and an objective determine the fitted rule. Held-out evaluation then asks whether its useful behaviour extends beyond those examples, rather than assuming low training loss is enough.

Let `fθ(x)` predict an output from input `x` using parameters `θ`. Supervised learning uses observed input–target pairs. Self-supervised learning constructs targets from the data, such as the next token. Unsupervised learning is a broader family concerned with structure without externally supplied task labels. Reinforcement learning uses rewards from interaction or sampled behaviour, not a correct target token for every step.

**Empirical risk minimization** chooses parameters to reduce average training loss:

\[
\hat\theta=\arg\min_\theta\frac1n\sum_{i=1}^{n}\ell(f_\theta(x_i),y_i).
\]

This does not guarantee small future error. A sufficiently flexible system can memorise examples or exploit a shortcut that fails elsewhere. An **inductive bias** is a preference imposed by representation, model, objective or optimization. There is no completely assumption-free learner.

## 2. Baselines

**Key terms.** A **baseline** is a simple reference method; a **shortcut** is an easy predictive cue that may fail outside the current data.

**Process.** Run a simple method on the same split first. Its score establishes what extra complexity must improve on; inspect errors before deciding what larger model to train.

For text classification, compare majority class, bag-of-words with logistic regression, a pretrained encoder and your proposed method. Logistic regression maps a linear score to class probabilities. A decision tree splits examples using feature rules; a random forest averages many decorrelated trees, reducing some variance. Neither automatically understands language, but simple baselines reveal whether an expensive representation earns its cost.

Regularization constrains fitting: weight penalties, early stopping, smaller models or data augmentation. The bias–variance picture is useful intuition about systematic restriction versus sample sensitivity; modern overparameterized models do not always follow a simple U-shaped textbook curve.

## 3. Training, development and test splits

**Key terms.** The **training split** supplies updates; **development** data guides choices; the **test split** is reserved for final assessment.

**Process.** Split independent source groups before fitting anything. Train on one part, select settings on another, then freeze choices before testing. Reusing test scores for selection changes their role.

```text
training set → update model weights
development set → choose model, prompt, threshold and other settings
test set → estimate performance AFTER choices are frozen
```

Repeatedly checking the test and changing the method adapts to the test, even with no gradient update. Split by document/source/person/time where needed. A random sentence split may leak near-identical text or author style into evaluation. Deduplicate before deciding that a random split is safe.

**Cross-validation** repeats train/validation splits for estimation or selection. **Nested cross-validation** separates hyperparameter selection from outer evaluation. **Leave-one-dataset-out** trains/selects using other datasets, then evaluates on the held-out dataset. It tests a particular transfer protocol, not universal generalization.

## 4. Domain and label-policy shift

**Key terms.** A **domain** is a data setting; **distribution shift** changes the data-generating conditions; **label policy** defines what the target means.

**Process.** After an in-domain result, change the evaluation domain or label policy deliberately. Separate changed language from changed definitions before concluding that a model transferred or failed.

- **Covariate shift:** input distribution changes, while the relevant conditional relationship is assumed stable.
- **Label/prior shift:** class proportions change under a particular stability assumption about class-conditional inputs.
- **Concept shift:** the relationship between input and target changes.
- **Annotation-policy shift:** people apply different definitions, for example whether an unsupported assertion counts as an argument.

A different guideline can change the correct answer for the same text. That is not necessarily noisy annotation. A model that ignores the guideline may perform well on one dataset and systematically fail on another.

If a role-based rule works on claim-oriented corpora, test whether those corpora's definitions favour that mapping before claiming the rule captures argumentation generally. A successful rule on one annotation policy is not evidence that it transfers to every definition of the task.

## 5. Metrics answer different questions

**Key terms.** A **true positive** is a correct positive prediction; a **false positive** is an incorrect one; a **false negative** is a missed positive. **Precision** measures correctness among positive predictions; **recall** measures positives recovered; **F1** combines them; **macro** means averaging class scores equally.

**Process.** Convert predictions and labels into counts, calculate class metrics, then choose an aggregation matching the research question. A single average cannot show every change in error type.

**Calculation.** For the positive class, 18 true positives, 2 false positives and 12 false negatives give precision 18/20=0.90, recall 18/30=0.60 and F1=2×0.90×0.60/(0.90+0.60)=0.72. Macro-F1 additionally needs the other class scores; it is not this single positive-class F1.

For a positive class: precision `TP/(TP+FP)`; recall `TP/(TP+FN)`; F1 `2TP/(2TP+FP+FN)`. Macro-F1 averages per-class F1; micro-averaging pools decisions. Accuracy can conceal minority failure. State how undefined zero-denominator cases are handled.

Example: 90 negatives and 10 positives; predicting all negative yields 90% accuracy but zero positive recall. An improved positive recall can coexist with lower macro-F1 if false positives or negative-class performance worsen. A single count does not determine an aggregated score.

For generation, exact match is strict, overlap metrics test surface similarity and semantic metrics use representations. A fluent but unsupported answer can score well on some of them. Choose metrics from the actual task and error costs.

## 6. Controlled comparisons

**Key terms.** A **controlled comparison** changes one intended factor; a **confound** changes alongside it; an **ablation** removes a component to test its contribution.

**Process.** State the claim, hold its competing explanations fixed, run the contrasted conditions and compare paired outcomes. Only then attribute a difference to the changed component.

If you compare direct classification with generated structure, hold constant the split, allowed information, target definition and selection budget where feasible. Report additional generation/training compute. If one condition gets a passage and guideline while the other gets only a sentence, the difference is not solely “structure versus no structure.”

An **ablation** removes or changes a component. A **matched control** changes the factor of interest while matching important alternatives. Neither is automatically a causal experiment: changes can alter multiple properties simultaneously. Strong claims need strong controls.

### A transfer experiment that separates style from definition

Suppose source data calls an unsupported assertion an argument, but the target guideline requires explicit supporting evidence. The same sentence can legitimately have different labels. Describe the task as predicting `y` from text `x`, context `c` and guideline `g`, rather than silently assuming `y` depends on `x` alone.

1. Freeze dataset-level splits before selecting prompts or checkpoints. Choose the target dataset only for final transfer testing under a declared leave-one-dataset-out protocol.
2. Compare sentence-only, sentence-plus-guideline, and sentence-plus-context-plus-guideline conditions. These isolate different questions; do not call every gain a better representation.
3. Keep model, selection data and generation budget matched where possible. Treat a generated abstraction as another representation with its own error and cost, not as automatically lossless information.
4. Inspect disagreements by target definition, negation, source style and missing context. A few examples suggest hypotheses; they do not establish the cause of the whole score difference.
5. If you retune a rule using target labels, report a target-adapted condition separately. It is no longer the same untouched transfer test.

This makes a failed transfer informative: it can reveal changed task semantics instead of simply a “weak model.” For systematic dataset-shift evaluation, see [WILDS](https://arxiv.org/abs/2012.07421); its benchmark is an example of evaluating shifts, not a guarantee for every NLP definition shift.

## Practice and checks

1. `TP=8,FP=4,FN=2`: precision 0.6667, recall 0.8, F1 about 0.7273.
2. A prompt was selected because it scored highest on the test. Is the final test score untouched? **No.** The test has influenced selection.
3. An in-domain improvement vanishes on a new dataset. List three alternatives: shortcut reliance, changed definition, changed input style/data quality. Do not assert which caused it without tests.
4. Design one study in which you separately vary the sentence, passage and guideline. State exactly which information every condition sees.

Reinforcement: [D2L generalization and environment shift](https://d2l.ai/). Use its generalization material after the definitions here; the course's matched-control exercises remain your main assessment.
