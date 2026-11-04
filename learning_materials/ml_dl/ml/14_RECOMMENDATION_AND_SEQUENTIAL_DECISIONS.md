# ML14. Recommendation and sequential decisions

A recommender ranks items, such as films or products, for a user. A simple model scores how well learned user and item vectors match. When today's choice also affects future outcomes, we need a sequential-decision view rather than treating each recommendation as an isolated ranking problem.

The first example scores two items and ranks them. The second, separate example adds an immediate reward of 1 to a discounted future reward of 2. The recommendation score is not an input to the reward calculation.

The common question is which action to choose. Ranking predicts relative preference; sequential methods additionally model feedback and consequences over time. These examples share a page for comparison, not because they form one pipeline.

[course map](../COURSE_MAP.md). Required background: ML10, P3.

## 1. Recommendation, bandits and action sequences

**Key terms.** A **recommender** ranks items; an **embedding** is a learned vector; a **bandit** observes feedback for selected actions; an **MDP** models multi-step state transitions; **return** discounts future rewards.

**Process.** For supervised recommendation, learn scores from historical interactions and evaluate ranking on an appropriate later/user split. For interactive decisions, choose an action, observe its limited feedback, update the policy and account for which actions were actually observed.

A missing user–item rating is not automatically a negative preference. Exposure determines which labels can exist. Popularity and user/item bias are useful baselines before matrix factorization. Cold-start users or items need a policy because their embeddings have no training history.

Bandits trade exploration against immediate reward. Multi-step RL additionally accounts for effects on future states. This lesson introduces that distinction, not a complete control course or a claim that an offline classifier can optimize real-world interventions. The existing LLM preference lesson later shows another policy-learning setting.

## 2. A worked calculation

**Symbols and assumptions.** u_i and v_j are user/item vectors; their dot product is a score. gamma in [0,1) discounts future rewards; G_t=sum_k gamma^k r_(t+k+1) is a return. Epsilon-greedy exploration chooses a random available action with probability epsilon.

For user vector [1,2] and item vectors [2,0] and [0,2], scores are 2 and 4, so the second item ranks first before any bias terms. For rewards [1,2] and gamma=0.5, the two-step return is 1+0.5×2=2. With epsilon=0.1, roughly one tenth of choices explore in expectation, not exactly every tenth action.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. What ranking and reward evaluation measure

Popularity, content similarity and factorized models use different available information. A deeper model may learn features but does not remove exposure bias or feedback loops. Offline ranking quality and online reward are different measurements; counterfactual evaluation needs assumptions beyond ordinary held-out accuracy.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: feedback depends on what was displayed

Use an authored interaction matrix and simulated bandit rewards first. No external user data or live recommendation actions are required. Keep a real public interaction dataset on a later rights/privacy review rather than import one for breadth alone.

**Check.** Why is evaluating only items that the old system displayed potentially misleading?

**Answer.** The logged outcomes are selected by the old policy. Unshown items lack comparable feedback; treating that selection as random can bias estimates.

## 5. Learn recommendation scores with matrix factorization

**First follow the recommendation branch; the decision-learning branch starts in section 7.** Matrix factorization learns a short vector for each known user and item. Their dot product, optionally with user/item bias terms, predicts an interaction score. The vectors are fitted parameters, not necessarily named traits such as humour or price preference.

For explicit ratings, one possible training loss is squared error over observed user–item pairs plus a norm penalty:

\[
\hat r_{ui}=b+b_u+b_i+u^\top v_i,\qquad
L=\sum_{(u,i)\in\Omega}(r_{ui}-\hat r_{ui})^2+\lambda\left(\sum_u\|u\|^2+\sum_i\|v_i\|^2\right).
\]

Omega is the set of observed pairs; r is an observed rating, b terms are biases and lambda sets regularization strength. Implicit feedback such as clicks needs a declared objective, weighting and negative-sampling policy instead of silently calling every missing rating zero.

For a single observed pair with target 5, predicted score 3 and half-squared error, the loss is 0.5×(5−3)²=2. If u=[1,1] and v=[1,2], the gradient with respect to u, without regularization, is (3−5)v=[−2,−4]. A step of size 0.1 gives u'=[1.2,1.4]. Holding v fixed for this illustrative step changes the score to 4.0. Actual joint training updates many factors; this step is not its final result.

Data preparation → observed-pair sampling → score → declared loss → factor updates is the training path. At prediction time the saved factors score candidates without learning from an unseen user's future feedback. New users/items lack fitted identity vectors; content features or explicit fallback policies address this **cold-start** case.

**Practice.** A movie was never shown to a user. Is its missing rating evidence of dislike? **Answer.** No. Exposure and preference are different. Test a popularity baseline, use an appropriate time/user split and state which feedback was available.

[Matrix-factorization explanation](https://developers.google.com/machine-learning/recommendation/collaborative/matrix).

## 6. Retrieve candidates, rank them and audit exposure

A large catalogue cannot always be processed with the most expensive scorer for every request. **Retrieval** selects a manageable candidate set. **Ranking** scores that set more carefully. **Reranking** applies declared constraints such as diversity or freshness. These are different stages; the final ranker cannot recover an item that retrieval excluded.

A **two-tower model** encodes user/query features with one network and item features with another, producing compatible vectors. Precompute item vectors where the architecture allows it, then retrieve by a specified similarity. A cross-feature ranker can use richer interactions afterward; unlike independent towers, it may not permit one reusable vector per item. Training objectives and negative sampling determine what the similarity learns.

For query vector [1,2] and items A=[2,0], B=[0,2], C=[1,1], dot products are 2, 4 and 3. Top-two retrieval returns B and C. A separate ranker might assign click probabilities 0.30 and 0.45, producing final order C,B. The dot products were not click probabilities, and A is unavailable to this ranker. Offline retrieval recall measures whether useful items survive; ranking metrics then assess ordering under the declared candidate protocol. Their denominators and negative pools must be stated.

Logged data reflect which items the previous system exposed. Users may also inspect high positions more often. This **exposure/position bias** can make “not clicked” ambiguous and reinforce earlier selection. Log what was eligible, shown and observed where permitted. Corrections based on exposure probabilities require suitable support and assumptions; they are not a universal fix. Keep this as a data/evaluation limitation rather than claiming an experiment on user behaviour that was never run.

**Practice.** The ranker improves on its fixed candidate set but overall useful-item recall falls. What should be checked? **Answer.** Retrieval coverage, filtering and candidate construction first. A conditional ranking gain does not establish an end-to-end improvement.

Continue to bandits only when the system must learn from the feedback its own choices produce. [Recommendation stages](https://developers.google.com/machine-learning/recommendation/overview/types) · [Retrieval](https://developers.google.com/machine-learning/recommendation/dnn/retrieval) · [Exposure-bias study](https://proceedings.mlr.press/v48/schnabel16.html).

## 7. Contextual bandits: learn from selected actions

A contextual bandit observes a context, chooses an action and receives feedback for that action. It does not observe every alternative's reward. This is different from a fully labelled classifier and from a multi-step environment in which actions change future states. **Exploration** tries alternatives to learn about them; **exploitation** chooses an action currently estimated to be good.

Suppose estimated rewards for A and B are 0.6 and 0.4. An epsilon-greedy rule exploits with probability 0.8 and otherwise samples uniformly from both actions. Thus P(A)=0.8+0.2/2=0.9 and P(B)=0.1. Exploration includes A, so its 0.2 probability is not the probability of selecting B.

If A produces reward 1, a supplied incremental step of size 0.25 updates its estimate to 0.6+0.25×(1−0.6)=0.7. Nothing was observed about B on this round. A real contextual estimator conditions on context and shares information under a specified function class; this scalar update only illustrates feedback handling.

Keep the context, eligible actions, selected action, selection probability and observed outcome under a privacy-aware logging policy. These probabilities are necessary for some offline-policy estimators, but reliable evaluation also needs support: a policy cannot be confidently evaluated on actions the logging policy never tried in the relevant contexts. Delayed, censored or adversarial feedback further changes the problem.

**Practice.** Can a deterministic historical policy provide an ordinary labelled reward for every unchosen item? **Answer.** No. The missing rewards are not observed negatives. Evaluate first in a bounded simulator with known action rewards and compare several seeds; do not explore on real users merely to complete a course exercise.

The next section introduces state transitions and future consequences, which bandits do not model in this formulation. [Contextual-bandit tutorial](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html).

## 8. MDPs, Bellman equations and tabular Q-learning

A Markov decision process (MDP) models repeated interaction: observe state s, choose action a, receive reward r and enter next state s'. A **policy** specifies action probabilities. A state representation is Markov when it contains the information needed for the next-state/reward distribution given the action. Real environments can violate this representation assumption.

The **return** sums future rewards with discount gamma. A value V describes expected return from a state under a policy; Q describes expected return after a state–action pair. For a fixed policy, the Bellman relationship averages over its next actions. Optimality instead uses the best available next-action value. Do not substitute a maximum into a policy-evaluation equation without changing its meaning.

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right].
\]

This tabular Q-learning rule uses one sampled transition. Alpha is a learning step, not the discount. If Q(s,a)=2, r=1, gamma=0.9 and next-action estimates are 4 and 3, the target is 4.6 and a step alpha=0.25 gives 2+0.25(4.6−2)=2.65. The target is a bootstrap estimate using current Q values, not a known final return. At a genuinely terminal next state, its future contribution is zero, giving target 1 instead. A time limit is not automatically a terminal state of the underlying task; define truncation handling.

Q-learning is **off-policy** because its maximum evaluates a greedy next action even if exploratory behaviour chose something else. Tabular convergence requires sampling and step-size conditions; replacing the table with a neural network does not inherit that guarantee. Replay buffers reuse transitions; target networks can stabilize a moving target, but neither creates missing coverage or guarantees safe behaviour.

**Practice.** With the same values but a terminal transition, what is the update? **Answer.** 2+0.25(1−2)=1.75. Explain why bootstrapping to 4.6 would be wrong for that terminal transition.

The next section learns the policy directly rather than extracting it from a Q table. [MDP/value definitions](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) · [Original Q-learning analysis](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).

## 9. Policy gradients and actor–critic learning

A policy-gradient method changes the parameters of an action distribution to improve expected return. It uses the log probability of an action that was actually sampled, weighted by a return or **advantage** estimate. An advantage compares an outcome with a baseline; a reward alone is not that comparison.

In actor–critic, the **actor** is the policy and the **critic** estimates value. A one-step temporal-difference estimate is:

\[
\delta=r+\gamma V(s')-V(s),\qquad
\theta\leftarrow\theta+\eta\,\delta\,\nabla_\theta\log\pi_\theta(a\mid s).
\]

For this illustration, treat delta as fixed during the actor update. Terminal states need zero future contribution as above. The critic learns a value target; the actor increases or decreases the probability of the sampled action. They optimize different quantities, even when they share some representation parameters.

Let P(A)=sigmoid(theta), initially theta=0 so P(A)=0.5. Choose A; suppose V(s)=2, reward=1, V(s')=3 and gamma=0.9. Delta is 1.7. For sampled A, the derivative of log P(A) is 1−P(A)=0.5. An actor step eta=0.1 gives theta'=0.085 and P(A)≈0.5212. A separate tabular critic step of size 0.2 gives V'(s)=2+0.2×1.7=2.34. The critic's accuracy affects the advantage signal; these are one-step authored values, not a converged agent.

Policy gradients can have high variance; baselines and advantage estimation help under stated assumptions but can introduce other estimation trade-offs. Continuous actions need suitable distributions and bounded-action handling. Compare evaluation episodes with fixed rules, multiple training seeds and disclosed interaction budgets; reward hacking can improve the number while failing the intended task.

**Practice.** If the estimated advantage is negative, should this sampled action become more probable? **Answer.** The same ascent rule lowers its log-probability locally, subject to parameter interactions. No claim follows that every other aspect of behaviour improves.

Continue to the existing **R2 preference-learning lesson** for PPO's clipped surrogate and LLM-specific rollout/reference roles. DPO is a separate preference-pair objective, not the next ordinary actor–critic update. [Policy-gradient derivation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) · [Vanilla policy gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html).

## Sources

- [Google recommendation course: matrix factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix).
- [Stanford CS229, Summer 2026 overview](https://cs229.stanford.edu/).
