# P3. Probability, uncertainty and statistics

Probability describes how likely different outcomes are. Statistics uses a finite set of observations to estimate quantities such as a model's accuracy. These are related but different: a model's confidence in one prediction is not the same as our uncertainty about an accuracy measured on a small test set.

Imagine a model assigns probability 0.8 to class 1 and 0.2 to class 0. We can ask whether it predicted an observed label well, how uncertain its distribution is, and how a sample-based statistic varies.

First distinguish probability, prediction loss and entropy. Then study resampling and confidence intervals as a separate question about the evidence in a dataset.

## 1. Probability is a distribution over possibilities

**Key terms.** A **distribution** assigns probabilities to outcomes; **conditional probability** uses stated information; **likelihood** scores observed data under a model.

**Process.** A prefix conditions the next-token distribution. Multiplying successive conditional probabilities gives a sequence probability; taking logarithms turns that product into the sum used for training.

A discrete distribution assigns nonnegative probabilities that sum to one. A language model predicts a distribution over its vocabulary for the next position. It does not produce a single word until a decoding rule chooses one.

Conditional probability `P(A|B)` means probability of A after conditioning on B. The chain rule gives:

\[
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t}).
\]

If `P(the)=0.2`, `P(cat|the)=0.1` and `P(sat|the cat)=0.4`, the three-token sequence probability is `0.008`. Taking logs turns multiplication into addition: `log P(sequence)=Σ log P(token|prefix)`. This avoids numerical underflow and leads directly to language-model training.

**Likelihood** treats observed data as fixed and asks which parameter choices assign it high probability. It is not the same as the probability that a theory is true. Bayes' rule combines likelihood and a prior: `P(H|D)∝P(D|H)P(H)`. A model's next-token probability is not automatically a calibrated probability that its sentence is factually correct.

## 2. Expectation, variance and sampling

**Key terms.** **Expectation** is a probability-weighted average; **variance** is average squared deviation from it; a **sample** is a finite set of observations.

**Process.** A distribution describes possible outcomes. Sampling produces observed outcomes, whose average estimates the expectation. The dependence between observations determines how much independent evidence that average contains.

Expected value is a probability-weighted average: `E[X]=Σpᵢxᵢ`. A reward taking values 1 with probability 0.7 and 0 otherwise has expectation 0.7 and variance `0.7×0.3=0.21`. A finite sample average fluctuates around the population expectation under suitable sampling assumptions.

Independent examples carry more new information than repeated copies. If 100 questions are slight variations of one source document, treating them as 100 independent pieces of evidence makes uncertainty look artificially small.

## 3. Entropy, cross-entropy and KL

**Key terms.** **Entropy** measures uncertainty in one distribution; **cross-entropy** measures prediction cost under another; **KL divergence** is their difference under matching conventions.

**Process.** Start with a target distribution and predicted probabilities. Compute log penalties and average them, then separate unavoidable uncertainty from mismatch. Specify the logarithm base before interpreting the units.

Entropy `H(p)=−Σpᵢ log pᵢ` describes uncertainty within a distribution. A fair binary distribution has entropy `ln 2≈0.6931` nats. A deterministic distribution has zero entropy. “Nats” means natural logarithms; bits use base 2.

Cross-entropy `H(p,q)=−Σpᵢ log qᵢ` measures the log loss of predicting `q` when outcomes follow `p`. KL divergence is:

\[
D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}=H(p,q)-H(p).
\]

It is nonnegative but not symmetric and not an ordinary geometric distance. If `qᵢ=0` where `pᵢ>0`, KL is infinite. With `p=[0.5,0.5]` and `q=[0.9,0.1]`, KL is about 0.5108 nats. High-confidence wrong predictions can be punished heavily by log loss.

**Perplexity** is `exp(mean token negative log likelihood)`. If mean loss is `ln 4`, perplexity is 4. It behaves like an effective uncertainty count, not “the model is confused about exactly four words.” Perplexities using different tokenizers or corpora are not directly comparable.

## 4. Confidence intervals and bootstrap

**Key terms.** An **estimate** is calculated from a sample; a **confidence interval** quantifies uncertainty under a sampling procedure; **bootstrap** resamples observed units with replacement.

**Process.** Freeze paired predictions, resample the independent cases, recompute the metric difference and repeat. The resulting distribution supplies interval bounds; it does not include every possible source of uncertainty.

Suppose a classifier gets 82 of 100 test cases right. Its observed accuracy is 82%, but another sample from the same population could give a different score. A **confidence interval** expresses uncertainty about the population score using a specified sampling procedure. For a 95% procedure, intervals would contain the fixed population quantity in about 95% of repeated suitable samples. A frequentist 95% interval is not, strictly, a 95% posterior probability that the fixed parameter lies in this particular observed interval.

The **bootstrap** approximates sampling variability by resampling observed units with replacement. For two models tested on the same examples:

```text
fixed paired predictions and labels
→ sample example indices WITH replacement
→ compute score A and score B on those SAME indices
→ save difference A−B
→ repeat many times
→ inspect the difference distribution and percentile bounds
```

Example: A is correct on `[1,1,0,1]`; B on `[1,0,0,1]`. Accuracy difference is 0.25. A bootstrap sample of indices `[0,1,1,3]` produces a difference of 0.5; `[0,2,2,3]` gives 0. Four examples are too few for trustworthy real conclusions; the arithmetic illustrates the procedure only.

For macro-F1, recalculate macro-F1 inside each resample; do not bootstrap only the final score. Resample documents/patients/conversations if those are the independent units. Bootstrap uncertainty for a fixed trained model does not include all training-seed variation, future distribution shift or label-policy disagreement.

### Decide which uncertainty your experiment needs

A confidence interval is only meaningful after its target quantity and resampling unit are clear.

1. Freeze both systems' predictions on the same cases. Define whether the target is accuracy on one population, macro-F1, or an average of per-dataset scores.
2. Identify independent units. For five sentences from each document, resample documents and keep their sentences together; resampling individual sentences treats correlated evidence as independent.
3. In each paired resample, apply the same sampled units to A and B, recompute the complete metric, and record `metric(A)−metric(B)`. For F1, do not average sentence-level “F1 values.”
4. Repeat and calculate the stated interval method. Percentile and bias-corrected/accelerated intervals are different procedures; tiny or degenerate samples can make either unreliable.
5. Repeat training across seeds separately when training variability matters. A test-case bootstrap of one seed does not include that variability, and neither procedure certifies performance under a new annotation policy.

Also distinguish **micro-averaged token loss**, which gives longer texts more weight, from a mean of per-document losses. Neither is universally right; they estimate different quantities. Report the denominator, effect size and uncertainty instead of using a p-value as a quality ranking. Implementation references: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) and [multiple-testing procedures](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## 5. Hypothesis tests and many comparisons

**Key terms.** A **null hypothesis** is the claim a test assesses; a **p-value** measures compatibility with that null under its assumptions; **Holm correction** controls family-wise error across tests.

**Process.** Define comparisons before examining results. Calculate their test statistics and p-values, then apply the declared multiple-testing rule before interpreting evidence. Effect sizes and intervals still matter.

**Calculation.** Holm step-down example: with three sorted p-values 0.01, 0.03 and 0.04 and family-wise α=0.05, compare against 0.05/3≈0.0167, then 0.05/2=0.025. Reject the first null, stop at the second failure, and do not reject the third. These are illustrative test outputs, not evidence from this lab.

A p-value describes how surprising a statistic at least as extreme would be under the specified null model. It is not the probability that your hypothesis is false, nor an effect-size measure.

Testing many variants increases the chance of declaring at least one apparent improvement just by chance. A family-wise error rule limits the probability of one or more such false rejections across the declared collection of tests. **Holm's correction** controls family-wise error under its assumptions: sort `m` p-values ascending; compare the first with `α/m`, the next with `α/(m−1)`, and continue until a comparison fails. Stop rejecting from that point. For `p=[0.01,0.03,0.04]` and `α=0.05`, reject 0.01 against 0.0167; 0.03 fails against 0.025; the remaining tests are not rejected. Predefine the comparison family. Do not select only favourable tests and then correct that subset.

## 6. Calibration and abstention

**Key terms.** **Calibration** compares stated confidence with observed correctness; **abstention** means declining to answer; **coverage** is the proportion answered.

**Process.** A model supplies scores or confidence. Check them against held-out outcomes, choose any refusal threshold on development data, then measure both error and coverage on an independent set.

Calibration asks whether cases assigned confidence 0.8 are correct about 80% of the time. A model can rank examples well but be badly calibrated. Reliability diagrams bin predictions; expected calibration error depends on bins and sample size. Brier score averages squared probability error; log loss uses logarithms.

**Abstention** means declining to predict under a specified policy. Coverage is the fraction answered; selective risk is error among answered cases. Reporting lower error while hiding that coverage fell from 100% to 10% is misleading. Confidence thresholds must be selected on development data, not the held-out test.

## Exercises and checks

1. Target probability rises from 0.25 to 0.5. How much does loss fall? **Answer:** `ln 2≈0.6931` nats.
2. Why pair bootstrap indices? **Answer:** preserve the shared-example dependence so the difference is estimated on comparable resamples.
3. Can a narrow interval establish generalization to Twitter? **Answer:** no; it describes sampling variability for the sampled population under assumptions, not a new domain.
4. What happens to KL regularization when you reverse `KL(policy||reference)`? **Answer:** it becomes a different objective, with different sensitivity to support and low-probability regions.

Visual reinforcement: [Seeing Theory](https://seeing-theory.brown.edu/) → Conditional Probability, then Frequentist Inference → Bootstrap. The site is archived; it remains a visual aid, not current software guidance. The worked examples above supply the corresponding entropy, loss and resampling calculations without an animated interface.
