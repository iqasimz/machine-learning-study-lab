# ML16. Run and report an ML experiment

A complete experiment compares models under a clear task, data split and metric. Fit a simple baseline, choose candidate pipelines using validation evidence, and evaluate the final choice on reserved test data. Save enough information for someone else to check how the result was produced.

The worked plan compares a baseline and two candidates. Their displayed scores are invented examples used to explain selection. They are not measured rankings of the model families taught in this course.

The training/validation split lesson introduced the split boundaries. This capstone turns them into concrete outputs: data-split IDs, fitted preprocessing/model, chosen settings, per-example predictions and a result report.

[course map](../COURSE_MAP.md). Required background: ML09, ML10, ML12.

## 1. What makes an experiment reproducible

**Key terms.** A **protocol** fixes the comparison; a **manifest** records exact inputs and settings; a **pipeline artifact** includes preprocessing plus estimator. A **baseline** is the reference against which an added cost must be justified.

**Process.** Define the task, reserve the holdout, build the full fitted pipeline, validate and select within training data, refit the chosen recipe and assess once. Save per-example outputs and artifacts so another reader can reconstruct the result.

The model is only part of a reproducible experiment. Record data revision, independent split units, row identities, source exclusions, metric definition, search budget, estimator/library versions, seeds, selected settings and hardware. Hashes detect changed bytes; they do not prove a source is true or a comparison is fair.

Classical methods often fit in one solver call. Report measured elapsed work or available iterations; do not fabricate per-epoch curves for algorithms that expose no such trace. A final score is not a learning curve. Save source-unit errors, model-size/latency estimates and important failure groups alongside averages.

## 2. A worked calculation

**Symbols and assumptions.** For paired predictions A and B on the same cases, delta=metric(A)−metric(B). Its sign depends on whether larger or smaller metric values are better. A budget is the counted resource, not an adjective such as fair.

Suppose baseline MAE=0.60 and candidate MAE=0.57 in target units. Candidate minus baseline is −0.03; relative reduction is 0.03/0.60=5%. A bootstrap interval overlapping zero or much larger inference cost can change the interpretation. These numbers are illustrative, not results from a newly run experiment.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Fair comparison budgets and saved artifacts

Report one controlled factor before a large model zoo. Equal search counts, equal fit time and equal parameter counts are different constraints. For deployment, save fitted encoders/scalers, feature order and label map, not just estimator weights. Untrusted pickle/joblib files can execute code; never load arbitrary uploaded models.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare classical and neural methods

Choose Wine Quality regression or digits classification, state the task and compare dummy, linear, kernel/tree and ensemble candidates. Reserve the exact same protocol for the DL capstone where compatible. Do not drop a strong classical baseline to make DL appear successful.

**Check.** What evidence would allow a first reader to decide that a neural model genuinely helped?

**Answer.** Matched data/splits, baseline tuning, independent outcomes, per-case errors, uncertainty appropriate to the units, and measured compute/latency. If the neural model does not help, a defensible negative result is a successful experiment.

## 5. Responsible evaluation: groups, harms and human oversight

A model can have good average performance while distributing mistakes unevenly. Responsible evaluation starts with the intended use, who is affected, what a false positive/negative costs and who can challenge or stop a decision. One fairness statistic does not answer all those questions.

**Selection rate** is the fraction receiving a positive prediction. **Demographic parity** compares that rate across groups. **Equal opportunity** compares true-positive rates among actual positives; **equalized odds** also compares false-positive rates among actual negatives. Define reference labels and group membership carefully: historical labels can already reflect exclusion, and grouping choices can hide important intersections. Removing a group column does not remove correlated proxy information.

Suppose group A has 100 cases: 50 actual positives, 40 true positives and 10 false positives. Group B has 100: 20 actual positives, 16 true positives and 4 false positives. Both true-positive rates are 0.8. Selection rates are 0.50 and 0.20. False-positive rates are 10/50=0.20 and 4/80=0.05. This fixture has equal opportunity on the supplied labels, not demographic parity or equalized odds. The metric names describe different comparisons, not different spellings of “fair”.

Use the shared metrics lesson for the underlying confusion counts rather than relearning F1 here. In a real report, give group sample sizes and uncertainty, examine missing labels and choose thresholds/mitigations within validation data. Averages from tiny groups can be unstable; incompatible criteria can require a context-dependent trade-off. Do not present a selected metric as legal compliance or moral certification.

**Practice.** What must a human-oversight plan specify beyond an “approve” button? **Answer.** Who is responsible, what evidence they receive, when the system abstains/escalates, how mistakes are corrected, and whether reviewers have time and authority to disagree. Privacy-aware data minimization, access control and retention policies apply to these audit records too.

Use risk management as an ongoing process of defining context, measuring failures and managing responses, not a final accuracy threshold. [Fairlearn metric definitions](https://fairlearn.org/main/user_guide/assessment/) · [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).

## 6. Robustness tests, documentation and release decisions

Robustness asks whether useful behaviour survives a specified change. Test realistic missing values, subgroup shifts, input corruptions or new sources separately from an adversarial threat model in which an attacker deliberately chooses inputs. An adversarial example, poisoned training data and a leaked private record are different failures and need different tests. No finite stress suite proves universal robustness or privacy.

Start from the frozen development protocol, define each stress condition and preserve the original independent unit. Compare matched predictions with the same metric and report where the model fails. If validation accuracy is 0.90 and a corruption score is 0.72, the drop is 0.18, or 18 percentage points; the relative drop is 20%. Neither number identifies why the drop occurred. Inspect errors before attributing a mechanism.

The release record should connect data and license/provenance, intended users, prohibited uses, training procedure, evaluations, group failures, uncertainty, operational limits and an accountable owner. Model cards describe a model; dataset documentation describes the data. They complement runnable artifacts rather than replacing them. Do not put private training examples or credentials into an open-source report.

An executable acceptance plan might require: finite outputs for malformed-but-allowed inputs, exact preprocessing agreement on saved fixtures, bounded resource use, a specified minimum subgroup performance, and a documented fallback. Choose thresholds from the actual use case, not from the examples in this lesson. A high-risk system may need external review that this local course cannot provide.

**Practice.** A repaired model passes the new stress test but loses original-task recall. What should happen? **Answer.** Re-run the complete agreed regression suite and examine the trade-off before release. Keep the prior approved version recoverable, document the decision and avoid silently redefining success around the repair.

Continue to DL13 for exporting a compatible prediction pipeline and monitoring it; the existing R8 lesson adds LLM-specific prompt/tool risks rather than repeating these general metrics. [Model Cards](https://arxiv.org/abs/1810.03993) · [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) · [NIST risk-management core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/).

## Sources

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html).
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
