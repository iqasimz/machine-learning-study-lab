# ML07. Naive Bayes classification

Naive Bayes predicts a class by combining how common the class is with how likely the observed features are within that class. For example, a word can be more common in spam than ordinary email. With several features, the model simplifies the calculation by treating them as independent once the class is known; that is the 'naive' assumption.

We use eight training cases: four positive and four negative. A binary feature is present in three positive cases but only one negative case. We will use this evidence to calculate a class probability for a new case with that feature.

Unlike logistic regression, which directly fits a class probability from features, Naive Bayes models feature probabilities within each class. Both can classify; they make different modelling assumptions.

[course map](../COURSE_MAP.md). Required background: ML04.

## 1. Feature distributions and independence assumptions

**Key terms.** A **generative classifier** models class priors and feature distributions given class; a **discriminative classifier** models the class decision directly. **Naive Bayes** assumes conditional feature independence. **Covariance** describes joint feature variation. **Linear/quadratic discriminant analysis (LDA/QDA)** use Gaussian class models with shared/separate covariances.

**Process.** Estimate class priors and class-conditional feature statistics from training data. Score a new example under each class, combine with priors, normalize if probabilities are needed and choose a label or threshold.

The word generative here describes a statistical factorization; it does not mean a modern text-generation assistant. Multinomial naive Bayes suits nonnegative count-like features. Gaussian naive Bayes uses a per-feature Gaussian model. Bernoulli naive Bayes models presence/absence. Feature representation and likelihood must agree.

Conditional independence means features are independent given the class, not independent in all data. Repeated correlated words can make predicted confidence too strong. Smoothing avoids impossible unseen-event probabilities, but an arbitrary large smoothing value can wash out useful evidence. LDA shares covariance across classes; QDA estimates more flexible class-specific covariance and needs more data or regularization.

## 2. A worked calculation

**Symbols and assumptions.** P(c) is class prior and P(x_j|c) is a feature likelihood. The naive Bayes score is P(c) times the product of feature likelihoods. The normalization sums scores over classes.

For two classes spam/ham with priors 0.2/0.8, suppose two observed binary features have likelihoods [0.6,0.5] under spam and [0.1,0.2] under ham. Scores are 0.2×0.6×0.5=0.06 and 0.8×0.1×0.2=0.016. Normalized spam probability is 0.06/0.076≈0.7895 under these assumptions. This is not a calibrated guarantee.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Naive Bayes, LDA and QDA

Compare a naive Bayes model with logistic regression under the same vocabulary/split, or LDA with QDA on scaled numeric data. QDA can express curved boundaries but estimates more covariance terms. A simple model's small-data advantage is a hypothesis to measure, not a law.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: classification versus probability quality

Wine gives a bounded numeric comparison. SMS is only a conditional future dataset: audit privacy and duplicate groups before displaying examples. No real contact details should appear in a public screenshot or unrestricted log.

**Check.** Why can naive Bayes classify accurately while its probability estimates are poor?

**Answer.** Its independence and likelihood assumptions can preserve class ordering while overstating combined evidence. Evaluate calibration separately from classification accuracy.

## 5. Bayesian updating: keep a distribution over a probability

Bayesian updating combines an initial model of uncertainty with observed evidence. A **prior** describes an unknown parameter before these observations; a **likelihood** describes how observations arise for a proposed parameter; a **posterior** is the updated distribution. This differs from using Bayes' rule only to classify one row, as in the core Naive Bayes example.

Suppose a binary trial has one unknown success probability p. Assume trials are independent conditional on this fixed p. A Beta distribution is a convenient prior because updating it with successes and failures gives another Beta distribution. Its two positive parameters control the prior's shape; they are not observed data counts.

\[
p\sim\operatorname{Beta}(a,b),\qquad p\mid D\sim\operatorname{Beta}(a+s,b+f).
\]

Here s and f count observed successes and failures. Start with Beta(2,2), then observe three successes and one failure. The posterior is Beta(5,3). Its mean, 5/8=0.625, is also the model's predictive success probability for the next trial. The empirical success fraction is 3/4=0.75; these differ because the posterior still incorporates the prior. The posterior remains a distribution, not certainty that p equals 0.625.

Follow the information: specify the trial and prior → observe permitted training outcomes → update both parameters → calculate a predictive probability or a credible interval. A credible interval describes posterior uncertainty under the model. It is not the same construction as the bootstrap interval introduced in the statistics lesson.

**Practice.** Starting from Beta(1,1), observe two successes and no failures. What is the next-trial success probability? **Answer.** Beta(3,1) gives 3/4, not certainty of success. More observations narrow uncertainty only when the assumed fixed-probability model is appropriate. Repeated observations from correlated users or a changing environment require reconsidering that model.

Continue to graphical models below to see how several uncertain quantities can be connected. [Conjugate updating reference](https://stanford.edu/~ashlearn/RLForFinanceBook/appendix7.pdf).

## 6. Probabilistic graphical models: factor a joint distribution

A probabilistic graphical model expresses how a joint distribution can be assembled from smaller distributions. Its nodes represent **random variables**, unlike the data entities in the graph-neural-network lesson. A directed acyclic graph specifies one factorization; its arrows do not, by themselves, justify causal claims.

Consider hidden topic Z and two observed word indicators X and Y. The model assumes the indicators are independent once Z is known:

\[
p(z,x,y)=p(z)p(x\mid z)p(y\mid z).
\]

This is the same conditional-independence pattern underlying a Naive Bayes classifier, but the topic here is hidden. It is not a statement that words are independent in the complete population. Suppose P(Z=1)=0.5, and the probabilities of X=1 and Y=1 are 0.8 and 0.6 in topic 1, versus 0.2 and 0.1 in topic 0. After observing both words, the two unnormalized weights are 0.5×0.8×0.6=0.24 and 0.5×0.2×0.1=0.01. Normalizing gives P(Z=1|X=Y=1)=0.96.

**Inference** asks a probability query in a specified model, such as that posterior topic probability. **Learning** estimates model parameters from data. To obtain P(X,Y), sum the joint over hidden Z. In larger graphs, variable elimination reorders these sums and products to reuse intermediate factors; intermediate factor size can still make exact inference expensive. An undirected model instead specifies nonnegative factors with a global normalizer and needs its own independence interpretation.

**Practice.** If Y is unobserved, should its probability simply be set to zero? **Answer.** No. Sum over its possible values; because its conditional probabilities sum to one, the posterior topic query uses the remaining observed evidence. Missing is not the same event as Y=0.

The advanced uncertainty section in ML15 explains approximate inference when exact sums or integrals become impractical. [Graphical-model lecture](https://www.cs.cmu.edu/~rsalakhu/10417/Lectures/Lecture_GM.pdf).

## Sources

- [scikit-learn naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
- [scikit-learn LDA/QDA](https://scikit-learn.org/stable/modules/lda_qda.html).
