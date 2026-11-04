# ML01. Machine learning tasks and baselines

Machine learning fits a rule from examples rather than writing every decision by hand. In supervised learning, each training example has input features and a target: classification predicts a category, while regression predicts a number. Unsupervised learning instead looks for patterns without these target labels.

Suppose 20 of 100 cases are positive and 80 are negative. A system that always answers 'negative' gets 80% accuracy while finding none of the positive cases. This simple baseline exposes what a score can hide.

Before choosing a model, decide what must be predicted and what mistakes matter. The next lesson prepares the inputs; later lessons compare methods against a baseline on the same task.

[course map](../COURSE_MAP.md). Required background: P1, P2, P3.

## 1. Supervised and unsupervised tasks

**Key terms.** A **sample** is one observation; **features x** are available inputs; **target y** is the quantity to predict. **Supervised learning** uses targets; **unsupervised learning** searches for structure without them. **Generalization** means useful performance beyond fitting examples.

**Process.** Begin with a decision and its prediction time. Identify available features and target, group related records, split independent units, then fit a simple baseline. Later models must improve on that baseline under the same protocol.

For bike demand, predicting tomorrow's total is different from estimating today's total after both component counts are known. The first task must not use future observations. For flower classification, a row of measurements and a species label can be a supervised example, but repeated measurements from one plant are not automatically independent plants.

A model is a learned rule; training is the procedure that chooses it; evaluation tests a declared use. Establish a mean/median baseline for regression and a majority/prior baseline for classification. These expose the task's difficulty without architecture complexity. A good score from an easier, leaked task does not answer the harder question.

## 2. A worked calculation

**Symbols and assumptions.** n counts examples; d counts features; X has shape [n,d]; y has n targets; the baseline prediction is written as y-hat.

Suppose training targets are 2, 4 and 9. Their mean is 5, the constant that minimizes training squared error. On separate targets 3 and 7, predictions [5,5] have MSE=((5−3)²+(5−7)²)/2=4 and MAE=(2+2)/2=2. The median training target is 4 and minimizes absolute training error instead. Different objectives imply different baselines.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing a useful baseline

Classification predicts categories; regression predicts numeric outcomes; ranking orders alternatives. Unsupervised clusters need another validation criterion, not invented class accuracy. Reinforcement learning changes behaviour through sequential feedback rather than a fixed row label. Start with the smallest task definition that answers the actual question.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: splitting related observations

Iris is a quick correctness exercise, not proof of general robustness. Reserve a fixed stratified holdout if class counts permit it. For temporal bike data, train on earlier observations and evaluate later ones instead of shuffling all hours.

**Check.** A dataset contains several visits per person. Why can a random row split exaggerate performance? What should be split instead?

**Answer.** The model may see the same person's distinctive features in training and test. Split by person for new-person evaluation; choose a temporal policy instead if the real task is future visits of known people. State which claim the split supports.

## Sources

- [Stanford CS229, Summer 2026 overview](https://cs229.stanford.edu/).
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
