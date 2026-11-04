# ML15. Generalization and distribution shift

Generalization means performing well on new examples, not only examples used for fitting. A score may change because the new population contains a different mixture of easy and difficult cases. Distribution shift is the broader name for a change between the data used to develop a model and the data where it is evaluated or used.

Group A has average loss 0.1 and group B has average loss 0.4. The original population is 80% A and 20% B. Change their proportions while keeping these within-group losses fixed to see why the overall average changes.

The direct weighted average is the simplest explanation. Importance weighting is an equivalent way to express it using the original population. Learning-theory and uncertainty notes discuss other questions; they are not later operations on this average.

[course map](../COURSE_MAP.md). Required background: ML05, ML10, ML06.

## 1. Learning-theory assumptions and model uncertainty

**Key terms.** **Empirical risk** is average observed loss; **population risk** is expected loss under a target distribution; a **hypothesis class** is the candidate rule family. A **Gaussian process (GP)** models a distribution over functions through a covariance kernel.

**Process.** State sampling and function-class assumptions before invoking a generalization claim. Fit a model under those assumptions, quantify the uncertainty it actually represents, then test whether deployment changes invalidate that interpretation.

A theoretical bound is an upper limit justified by assumptions, not a measured error bar. It asks questions such as how much data would be enough to keep all models in a stated candidate family close to their true average loss. Real data dependence or a changed population can invalidate the assumptions.

Learning theory explains conditions under which fitting evidence constrains unseen risk; it is not a formula granting safety to arbitrary learned systems. **PAC (probably approximately correct)** formalizes learning to a chosen error tolerance with a chosen probability under stated sampling assumptions. **VC dimension** measures how many points a binary hypothesis class can classify in every possible labelling. These help describe capacity and sample requirements; optimization separately asks whether a fitting procedure finds a suitable solution.

A GP combines a covariance model, observations and noise assumptions to predict a mean and variance. Its uncertainty is conditional on that modelling choice, not automatic awareness of all unknown domains. Exact GP factorization has cubic training cost in case count and quadratic covariance storage; approximations trade other properties. Prediction from observational data also does not identify causal effects of intervening on a feature.

## 2. A worked calculation

**Symbols and assumptions.** For a finite fixed class H and iid bounded losses in [0,1], epsilon is an error tolerance, delta a failure probability and n sample size. A simple uniform bound has epsilon=sqrt(log(2|H|/delta)/(2n)). For GP regression, k is a kernel and sigma_n² is observation-noise variance.

For |H|=10, delta=0.05 and n=1000, the displayed bound gives epsilon=sqrt(ln(400)/2000)≈0.0547. It is a bound under its assumptions, not a measured standard error. In a one-point GP with zero prior mean, prior variance 1, noise variance 0.25 and observation y=2, posterior latent mean at that same point is 1×(1.25)⁻¹×2=1.6 and variance is 1−1/1.25=0.2.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Different sources of uncertainty

Bayesian model uncertainty, test-case bootstrap, training-seed variation and conformal marginal coverage answer different questions. A model can be confidently wrong under shift. Predictive importance of weather does not establish what an intervention on weather would cause.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: evaluate uncertainty outside observed regions

Use an authored one-dimensional noisy function to compare linear, kernel and GP fits. Evaluate withheld regions separately and record the assumed noise/kernel. Do not portray a smooth GP band as a distribution-free guarantee.

**Check.** Why does doubling model parameters not directly substitute into the finite-|H| formula above?

**Answer.** A continuous neural parameterization is not a finite class with one hypothesis per parameter. Its capacity analysis and optimization behaviour require additional theory; the simple bound cannot be reused by changing the label on |H|.

## 5. A Gaussian-process prediction is a conditional distribution

**Key terms.** A kernel matrix K stores pairwise covariance assumptions. Observation noise σ² separates noisy measurements from the latent function. A posterior conditions the prior on observed training data.

**Process.** Choose a mean/kernel/noise model, assemble the training covariance, solve the positive-definite linear system and condition on a new input. Prediction returns both a mean and an uncertainty under these assumptions.

A Gaussian process starts with assumptions about how function values vary together, then uses observed examples to update a prediction and its uncertainty. For example, a smoothness assumption says nearby input points should usually have similar outputs. The kernel encodes this relationship; it is not the probability that the prediction is correct.

In the equations, K is the training covariance matrix, k_* relates the new point to training points, k_** is its prior variance, and sigma squared is the assumed observation-noise variance. The first expression adjusts the predicted mean using observed targets y; the second reduces prior uncertainty by the information supplied by those observations.

\[
\mu_* = k_*^\top(K+\sigma^2I)^{-1}y,\qquad
v_* = k_{**}-k_*^\top(K+\sigma^2I)^{-1}k_*.
\]

For one centered observation y=2 with K=1, noise variance 0.25, k_*=1 and k_**=1, posterior mean is 2/1.25=1.6 and latent variance is 1−1/1.25=0.2. A noisy future observation adds 0.25 to that variance. Use a Cholesky solve, not an explicitly formed inverse.

### Assumptions remain part of the result

A smooth kernel can be confidently wrong about abrupt changes. Exact training is typically cubic in row count and quadratic in memory, motivating sparse/approximate variants. A fitted Bayesian uncertainty model is not automatically robust to distribution shift.

Gaussian processes are a theory branch here, not an implemented lab estimator. The executable bootstrap interval answers a different question and must not be described as a GP posterior.

## 6. Approximate inference: samples versus an optimized distribution

**Read after the Bayesian updating and graphical-model extensions in ML07.** Exact inference calculates the desired sum or integral without replacing it with a simpler approximation. The small topic model there can enumerate two possibilities. Many continuous latent variables or densely connected variables make that approach impractical.

Monte Carlo estimates a quantity by averaging samples. If z samples really come from the posterior, the average of a function g(z) approximates its posterior expectation. Markov chain Monte Carlo (MCMC) constructs a dependent sequence with the desired stationary distribution under stated conditions. Warm-up, mixing and effective sample size matter: ten thousand nearly identical chain states are not ten thousand independent pieces of evidence.

**Variational inference** instead chooses a manageable family q(z) and optimizes its parameters. In a common formulation it maximizes an evidence lower bound (ELBO):

\[
\mathcal L(q)=\mathbb E_q[\log p(x,z)-\log q(z)],\qquad
\log p(x)-\mathcal L(q)=\operatorname{KL}(q(z)\|p(z\mid x)).
\]

Here x is observed evidence, z hidden state, p the model and q the approximation. The identity holds for normalized q with suitable support. Since the KL divergence is nonnegative, the ELBO cannot exceed log evidence. Optimizing within a restricted family need not recover the true posterior. A mean-field family factors q into independent parts even when the posterior has dependencies; it can miss correlations or modes. The VAE lesson later uses an encoder to predict such approximate-posterior parameters for each input.

For a separate, fully enumerable example, let a binary posterior assign p(Z=1|x)=0.8 and choose q(Z=1)=0.5. Then KL(q||p)=0.5 log(0.5/0.8)+0.5 log(0.5/0.2)=0.2231 nats. If log evidence were −2, this q would have ELBO −2.2231. Both distributions are normalized; the number is not a classification error rate.

**Practice.** An MCMC run and a variational fit return different uncertainty. Which is automatically correct? **Answer.** Neither. Check convergence and sample dependence for MCMC, approximation/optimization limits for VI, and whether the underlying likelihood/prior fit the problem. Compare on an enumerable case first. Do not turn this computational choice into a claim about robustness under distribution shift.

[Variational-inference review by its authors](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf).

## Sources

- [Understanding Machine Learning, author site](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). Author-hosted reading is personal-use only; do not redistribute the book.
- [Gaussian Processes for Machine Learning, author site](https://gaussianprocess.org/gpml/).
- [scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html).
- [Angelopoulos and Bates, conformal prediction introduction](https://arxiv.org/abs/2107.07511).
- [An Introduction to Statistical Learning, author site](https://www.statlearning.com/).
