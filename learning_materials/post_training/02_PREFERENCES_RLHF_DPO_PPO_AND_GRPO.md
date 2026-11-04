# R2. Preference learning: DPO, PPO and GRPO

Preference and reward methods adapt which responses a language model favors. Direct preference optimization, or DPO, fits labelled response pairs relative to a reference model. Proximal policy optimization, or PPO, uses sampled behavior and an advantage-based surrogate. Group relative policy optimization, or GRPO, derives a comparison signal from responses sampled for the same prompt. Their signals and complete objectives differ.

Choose a method. The DPO fixture calculates one pair loss; PPO illustrates a positive-advantage clipped surrogate; GRPO normalizes rewards [0,1,2]. These are separate teaching calculations, not three stages of a training run or complete trainers.

SFT imitates demonstrated answers. Preference training compares or scores alternatives and can still use backpropagation to change policy weights. Feedback quality, rollout/reference roles and independent evaluation matter as much as the objective name.

## 1. Demonstrations, preferences and rewards

**Key terms.** A **demonstration** shows an answer; a **preference** ranks alternatives; a **reward** scores behaviour; **RLHF** uses human feedback in an RL-oriented pipeline.

**Process.** Begin with an imitation-trained model, collect feedback and choose its corresponding objective. Test usefulness independently because feedback can favour the wrong behaviour.

Demonstrations show one desirable answer, but several answers can be valid. Preferences compare alternatives. Rewards score behaviour. Optimizing either can improve usefulness, but it can also exploit flaws in the feedback. A long, confident answer may win preference comparisons while containing unsupported claims.

**RLHF** uses human feedback in a reinforcement-learning-oriented pipeline. **RLAIF** uses AI feedback. **RLVR** uses verifiable rewards, such as unit-test success or a checked final answer. A verifier can still be incomplete or exploitable; “verifiable” does not mean impossible to game.

## 2. The language model as a policy

**Key terms.** A **policy πθ** assigns action probabilities; **state** is available context; **trajectory** is sampled behaviour; **advantage** compares return with a baseline.

**Process.** The policy observes state and samples actions. A trajectory receives feedback, which becomes an update signal for those actions. A value critic estimates return rather than supplying ground-truth reward.

In RL terminology:

| RL concept | Language-model example |
| --- | --- |
| State | Prompt, prior generated tokens and available environment state |
| Action | Next token, or a tool/action in a larger policy |
| Policy `πθ` | Distribution over actions given the state |
| Trajectory | Generated response and possibly tool interactions |
| Reward | Human/model/verifier score, perhaps with penalties |
| Return | Accumulated future reward under a convention |
| Value | Expected return from a state |
| Advantage | How much better an action did than an appropriate baseline |

In many LLM tasks, a reward arrives only at response completion. Credit assignment asks which earlier choices deserve the change. A critic predicts expected return; it is not the reward model itself.

## 3. Policy-gradient training

**Key terms.** A **policy gradient** estimates how expected reward changes with weights **θ**; **R(y)** scores response y; **E** denotes expectation.

**Process.** Sample responses, score them and construct a differentiable objective from log probabilities. Backpropagate to update policy weights; the external judge and discrete sampling need not be differentiable.

```text
current policy → sample answers → score/reward
→ estimate advantages → calculate differentiable objective
→ backpropagate → optimizer changes policy weights → repeat
```

Yes, weights are still updated with gradients. The difference from SFT is how training signals and trajectories are obtained. You usually do not differentiate through a sampled discrete token choice or an external judge. The policy-gradient estimator provides the learning rule.

For response `y` and reward `R`, the score-function identity gives:

\[
\nabla_\theta E_{y\sim\pi_\theta}[R(y)]
=E[R(y)\nabla_\theta\log\pi_\theta(y)].
\]

Since `log π(y|x)=Σ_t log π(y_t|x,y_<t)`, sequence reward can influence each sampled token. Subtracting a state-dependent baseline can reduce variance without changing the ideal expectation under the needed assumptions. Practical estimators, clipping and normalization introduce further choices.

## 4. Reward learning

**Key terms.** A **reward model rφ** predicts preference scores; **σ** is sigmoid; **y_w/y_l** are preferred/rejected responses; **β** scales a reference-policy KL penalty.

**Process.** Fit the reward model on comparisons, score outputs and combine reward with a deviation penalty. Send this signal to policy optimization, keeping the reward model distinct from a value critic.

Given preferred response `y_w` and less-preferred `y_l`, a common preference model is:

\[
P(y_w\succ y_l\mid x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
\]

`σ` is sigmoid and `rφ` the reward model. Train by negative log likelihood of observed comparisons. A preference is conditional on annotator instructions and context; it is not a universal fact. Model annotator disagreement rather than pretending every pair has one objectively correct preference.

The policy is often regularized against a fixed reference:

\[
J(\pi)=E_{y\sim\pi}[r(x,y)]-\beta D_{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x)).
\]

The penalty discourages drifting too far from the reference under this KL direction. `β` trades reward against deviation. It does not guarantee safety or preservation of every capability.

## 5. PPO: careful updates from sampled behaviour

**Key terms.** **PPO** is proximal policy optimization; **π_old** generated samples; **ρ** is a new-to-old probability ratio; **A_t** is advantage; **ε** sets a clipping band.

**Process.** Collect rollouts under the old policy, estimate advantages and compute the clipped surrogate under the current policy. Optimize, then refresh rollouts; a separate reference can still penalize drift.

Let `π_old` be the policy that generated the rollout, `πθ` the policy being updated, and `A_t` an advantage estimate. The importance ratio is:

\[
\rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}.
\]

PPO's clipped policy surrogate is maximized:

\[
E[\min(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)].
\]

For positive advantage 2, ratio 1.5 and `ε=0.2`, the terms are 3 and 2.4, so the surrogate uses 2.4. The objective stops rewarding that particular excessive increase. For negative advantage, the sign matters: do not describe clipping as simply multiplying every advantage by a capped ratio. It is not a hard global guarantee on the entire policy's change.

**Crucial distinction:** `π_old` is for rollout importance ratios. `π_ref` is the reference used for regularization. They may be different models/checkpoints and serve different purposes.

For a negative-advantage contrast, let `A=−2`, ratio `ρ=0.5` and the same clipping band `[0.8,1.2]`. The terms are `−1` and `−1.6`, so the minimum is `−1.6`. Capping the ratio and always selecting that term without the minimum rule would misdescribe the objective. This signed case complements the live positive-advantage example.

LLM PPO commonly also trains a value function, uses an advantage estimator such as GAE, and adds KL/entropy/value terms according to the recipe. GAE forms a discounted weighted sum of temporal-difference residuals `δ_t=r_t+γV(s_(t+1))−V(s_t)`; its `λ` trades estimator bias and variance. Sequence truncation and terminal handling matter.

## 6. DPO: learn directly from preference pairs

**Key terms.** **DPO** is direct preference optimization; **π_ref** is fixed; a **log-ratio margin** contrasts policy-versus-reference scores of preferred/rejected completions; **Z(x)** normalizes a distribution.

**Process.** Score a fixed preference pair under policy and reference, compute the margin and minimize its logistic loss. Standard offline DPO backpropagates into the policy without a separate learned reward model.

DPO avoids training an explicit separate reward model in its standard offline setup. The regularized optimal policy has the form:

\[
\pi^*(y|x)=\frac1{Z(x)}\pi_{ref}(y|x)\exp(r(x,y)/\beta).
\]

This follows by optimizing expected reward minus the stated KL divergence from the policy distribution to the reference distribution, subject to probabilities summing to one. Naming the direction as `KL(policy || reference)` avoids forward/reverse terminology that can change across contexts. Rearranging gives `r=β log(π*/πref)+β log Z`. When comparing two responses for the same prompt, the `log Z` term cancels. Substitute into the preference likelihood:

\[
L_{DPO}=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right).
\]

These are **sequence** log probabilities under a defined token/masking convention. Length normalization changes the objective; do not insert it silently.

Separate numerical DPO example: the winning response's log-ratio is 0.4, the losing one's is −0.2, and `β=1`. These are different supplied values from the live pair fixture, not conflicting results. The margin is 0.6; preference probability is `sigmoid(0.6)≈0.6457`; loss is about 0.4375. At zero margin, loss is `ln 2≈0.6931`. This calculation does not establish that the model's real answers are now better.

DPO still requires quality preference data and a suitable reference. Offline data can become poorly matched to current outputs. It is not “PPO with fewer lines” and does not need online rollout sampling in its standard training objective.

### Follow one DPO pair through the actual differentiable quantities

Let `d` be the chosen-versus-rejected **log-ratio margin**, before multiplying by `β`.

1. Teacher-force each completion under the trainable policy and fixed reference. Sum log probabilities over the declared completion tokens; prompts and padding must not accidentally enter the sum.
2. Subtract each reference sequence log probability from its policy log probability, then subtract the rejected log ratio from the chosen one. The result is `d`.
3. Compute `L=−log sigmoid(βd)` using a numerically stable log-sigmoid. The derivative is:

\[
\frac{\partial L}{\partial d}=-\beta\,\sigma(-\beta d).
\]

4. At `d=0`, the loss is `ln 2` and the derivative is `−β/2`. Minimization pushes the relative margin upward; it does not independently guarantee that the chosen response's absolute probability rises on every update.
5. Backpropagate only through the trainable policy or its adapters. Reference scores can be precomputed for fixed data/reference, but become invalid if that reference or tokenization changes.

The objective fits observed preferences. If the pairs reward verbosity or contain incorrect “chosen” answers, successful optimization can reinforce those problems. In PPO/GRPO, additionally version the rollout policy and control stale trajectories: the ratio denominator belongs to the sampling policy, not automatically the regularization reference. Primary sources: [DPO](https://arxiv.org/abs/2305.18290), [PPO](https://arxiv.org/abs/1707.06347) and [current GRPO implementation choices](https://huggingface.co/docs/trl/grpo_trainer).

## 7. GRPO: compare a group of responses

**Key terms.** **GRPO** is group relative policy optimization; **G** counts responses; **rᵢ** is a reward; **r̄** is group mean; **ε** stabilizes division.

**Process.** Sample a response group, score it and centre rewards against its baseline. Use these advantages in the declared policy objective, then evaluate quality independently of reward.

For each prompt, sample `G` responses and obtain rewards. A simple group-relative advantage is:

\[
A_i=\frac{r_i-\bar r}{\operatorname{std}(r_1,\ldots,r_G)+\varepsilon}.
\]

These advantages can enter a PPO-like clipped token objective, often with a KL term depending on the version. GRPO avoids a separate learned value critic in its basic form; group rollouts still cost inference memory/time. Reference-model and rollout-policy roles remain distinct.

Separate four-response example rewards `[0,0,1,1]`: mean 0.5 and **population** standard deviation 0.5, so normalized advantages are approximately `[-1,-1,1,1]`, ignoring the tiny stabilizer. The live fixture instead uses three rewards `[0,1,2]`. A sample-standard-deviation implementation gives different magnitudes. If all rewards match, the centred signal is zero; group-relative normalization cannot invent a quality ranking.

Modern trainers expose alternative loss normalizations, importance weighting, KL choices and variance controls. Some variants alter length bias or handle stale/off-policy rollouts differently. **“GRPO” alone is not a complete experiment specification.** Save the actual loss type and configuration.

## 8. Other preference and reward objectives

**Key terms.** An **objective family** is defined by training signal and loss, not its acronym; **outcome rewards** score final results; **process rewards** score intermediate steps.

**Process.** Identify the feedback and exact loss before choosing a variant. Trace what is sampled, scored and differentiated so superficially similar methods are not treated as interchangeable.

- **Rejection sampling / best-of-N distillation:** sample, filter/choose, then imitate selected outputs; not necessarily policy-gradient RL.
- **RLOO:** use leave-one-out reward baselines for policy-gradient estimation.
- **IPO:** a preference-optimization alternative with a different objective/regularization interpretation; not just a DPO hyperparameter.
- **KTO:** uses desirable/undesirable examples rather than requiring every training item to be a matched preference pair.
- **ORPO:** combines preference-related training with a supervised objective through its own odds-based formulation.
- **Process rewards:** score intermediate steps; **outcome rewards:** score final results. Step labels can be costly and judging steps can still reward convincing but invalid reasoning.
- **Constitutional/AI-feedback approaches:** use explicit principles and AI critique/preferences; they inherit principle-selection and judge limitations.

This is a map of objective families. Before using a named variant, read its actual loss in the matching trainer/paper; similar acronyms conceal different assumptions.

## 9. Reward-optimization failure modes

**Key terms.** **Reward hacking** improves a score without its intended benefit; **sycophancy** favours pleasing agreement over warranted answers; **over-refusal** rejects legitimate requests.

**Process.** Monitor reward with independent quality, length, refusals and drift. When these disagree, inspect feedback and the update rule before spending more compute on the same objective.

Reward hacking, excessive verbosity, sycophancy, over-refusal, style over truth, length bias, forgetting, judge leakage, verifier exploitation and unstable updates all need tests. Keep a held-out human-reviewed set. Monitor reward **and** independent task quality, KL, response length, invalid outputs, group reward variance and cost.

Do not train on the same examples used to certify improvement. Test benign and difficult prompts across settings. A higher learned reward is evidence that optimization found higher reward, not proof it achieved the human objective.

## Exercises and checks

1. Explain reward model versus critic versus reference versus rollout policy with four separate sentences.
2. PPO: `A=−2,ρ=0.5,ε=0.2`. Terms are −1 and −1.6; minimum is **−1.6**.
3. DPO margin 0: preference probability 0.5, loss about 0.6931.
4. GRPO equal rewards: no within-group relative quality signal. What might help? Better reward discrimination, more informative prompts or a changed learning setup—not merely dividing by a smaller number.

Primary sources: [InstructGPT](https://arxiv.org/abs/2203.02155), [PPO](https://arxiv.org/abs/1707.06347), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300), [Constitutional AI](https://arxiv.org/abs/2212.08073). Implementation variants: [TRL GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer). Recalculate the pairwise margin and group advantage examples above, stating their masking and normalization conventions.
