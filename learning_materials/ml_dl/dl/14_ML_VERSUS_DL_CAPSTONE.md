# DL14. Compare neural and classical models

A useful neural-model experiment asks whether it improves a real outcome over a strong classical alternative. Both candidates need the same target definition and held-out evaluation cases. Their preprocessing and fitting procedures may differ, but each must use legitimate training information and a declared selection budget.

The worked comparison uses hypothetical final scores 0.82 and 0.80 to illustrate a difference of 0.02. They are not measured evidence that deep learning wins. The actual lab's saved results keep their own dataset, protocol and configuration information.

The ML capstone defined a complete experiment. This DL capstone applies that plan to a cross-family comparison, including tuning effort, errors, uncertainty and compute cost rather than just architecture labels.

[course map](../COURSE_MAP.md). Required background: ML16, DL06, DL13.

## 1. Evidence for a cross-family comparison

**Key terms.** A **matched comparison** controls the intended question; an **inductive bias** is a model's structural preference; a **tuning budget** is allowed selection work; **pretraining exposure** is extra prior data, not free architecture capacity.

**Process.** Choose a shared task/split, freeze preprocessing and evidence rules, tune each model within its allowed budget and save selected configurations. Compare independent outcomes, errors and measured costs, then explain which change the evidence actually supports.

On images, linear classifiers, MLPs and CNNs test different representation assumptions. On tabular data, ridge or boosted trees remain necessary competitors. Comparing only weakly tuned classical methods with an extensively tuned neural network measures more than model family.

Equal parameter counts are not equal training work; equal epochs are not equal updates; equal wall time is not equal hyperparameter exploration. Report these trade-offs instead of claiming one universal fairness definition. Repeat selected conditions across seeds where relevant and use paired independent-unit comparisons. A null or reversed result is informative if the protocol is sound.

## 2. A worked calculation

**Symbols and assumptions.** For N training cases, batch size B and E full epochs without dropping cases, optimizer steps per epoch are ceil(N/B) without accumulation. Parameter count, exposure, latency and metric difference are separate measurements.

For 1,000 cases and batch size 64, one full epoch needs 16 batches, with a final batch of 40. Ten epochs give 160 optimizer updates without accumulation. With accumulation 4 and a defined partial-window rule, they give 40 updates; the same epoch count no longer implies the same optimizer schedule. Count actual cases rather than assume 16×64 unique cases.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Comparison budgets and baselines

A prediction-quality study, resource-efficiency study and teaching demonstration deserve different result tables. Scratch and pretrained models need separate exposure labels. Model-size or architecture changes that alter compute should be documented, not hidden by one averaged score.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: report a useful negative result

Run the future digits progression first; only then move to Fashion-MNIST and a tabular contrast. Show confusion matrices, selected examples, measured curves where available, and a compact comparison table. No outcome is predetermined and no new results are claimed by these drafts.

**Check.** If a CNN fails to beat a linear baseline on a small subset, is the project a failure?

**Answer.** No. Check correctness, uncertainty, tuning and sample-size effects, then report the supported limitation. The learning goal is to understand the comparison, not manufacture a deep-learning victory.

## Sources

- [Stanford CS230, Fall 2026 syllabus](https://cs230.stanford.edu/syllabus/).
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- [CS231n practical neural-network training notes](https://cs231n.github.io/neural-networks-3/).
