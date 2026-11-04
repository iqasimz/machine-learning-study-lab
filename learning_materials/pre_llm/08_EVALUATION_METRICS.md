# P8. Evaluation metrics

A metric turns predictions and known answers into a number that summarizes one aspect of performance. Different metrics answer different questions: accuracy counts correct decisions, recall counts recovered positives, and a numerical error measures distance from a target value. Choose a metric to match the task and the consequences of mistakes, not because it gives the largest-looking score.

Choose a task below to inspect its cases and calculations. For binary classification, 'positive' means the class we want to detect, such as spam; 'negative' means the other class.

The examples show what is counted and divided. Metrics on the same cases are not separate processing stages: precision does not become an input to recall. The calibration lesson separately studies whether probabilities are reliable and how decision costs affect a cutoff.

## 1. Choose the quantity before comparing models

The main explanation in [the shared foundations](../teaching/foundations.md) defines the counted cases and denominators before interpreting a metric. The accompanying interactive examples let you change a threshold or task family and inspect those calculations. The sidebar and Interactive mode serve the same explanation; the notes below extend it with threshold curves and aggregation conventions.

## 2. Reading threshold curves

An ROC curve plots true-positive rate against false-positive rate as a score threshold changes. ROC-AUC measures ranking discrimination, not the correctness of a chosen probability scale. A PR curve instead plots precision against recall. Its apparent quality depends on positive prevalence, so compare it on a stated population.

Average precision sums recall increments weighted by precision at each score cutoff. It is not universally identical to trapezoidal area under a plotted PR curve. A decision threshold must be chosen using development data and consequences, not retrospectively optimized on the final test.

## 3. Probability quality and decision costs

Log loss and Brier score use probabilities, while F1 and accuracy use decisions. A monotonic recalibration can change probability quality without changing ranking. Calibration is studied in ML10; uncertainty and resampling assumptions are studied in P3 rather than redefined here.

### Aggregation conventions are part of the result

For multilabel tasks, “samples” averaging and subset accuracy have meanings different from single-label multiclass accuracy. State the label set, averaging rule, exclusions, sample weights and zero-division convention. Compare confidence intervals on the independent units: rows, documents, patients or groups as appropriate.

## Sources

- [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [scikit-learn F1 and averaging conventions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
- [Language-model perplexity and context](https://huggingface.co/docs/transformers/perplexity).
