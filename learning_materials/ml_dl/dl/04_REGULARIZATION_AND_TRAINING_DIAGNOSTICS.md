# DL04. Regularization, dropout and training diagnostics

Regularization changes training to discourage a model from relying too strongly on limited or accidental patterns. Dropout is one method: during training it randomly masks activation values, encouraging the network to cope without any particular selected feature. It does not permanently delete weights, and it is not a guaranteed improvement.

The features are [2,4]. Choose the probability of keeping a feature, q. A declared mask shows the arithmetic: retained values are divided by q. This fixed demonstration mask is not a randomly sampled training run.

Regularization was introduced in ML. Here we distinguish a particular dropout sample, its expectation over many masks and evaluation-mode behaviour; those are three related explanations, not a serial architecture.

[course map](../COURSE_MAP.md). Required background: DL03, ML10.

## 1. Different ways to control overfitting

**Key terms.** **Overfitting** improves fitting data while worsening new data; **dropout** randomly removes activations during training; **augmentation** changes examples under an assumed label-preserving transformation; **early stopping** selects a training duration using development data.

**Process.** Establish a small learning sanity check, train with a declared regularization recipe, inspect train/development behaviour and stop/select only under the declared rule. Evaluate the selected checkpoint separately from the final training state.

A failure to overfit a tiny clean batch can expose bugs, but is not proof that generalization is impossible. Once the mechanics work, compare learning curves, class errors and probability quality. High loss on both splits suggests different first checks from low train/high development loss. NaNs and out-of-memory errors are execution/numerical failures, not ordinary underfitting.

Dropout changes the training function stochastically. Weight decay changes parameter updates. Augmentation changes observed inputs. Label smoothing replaces a completely one-hot target with a distribution that leaves some probability on other classes. Early stopping changes the selected training duration. These are not equivalent controls even if all can limit brittle fitting. For a batch-size change, record whether epochs, updates, examples and scheduler timing remain matched.

## 2. A worked calculation

**Symbols and assumptions.** p_drop is the probability of dropping a feature and q_keep=1−p_drop is its keep probability. Inverted dropout divides a retained activation by q_keep during training so its expectation matches the original. PyTorch's p argument specifies p_drop; the walkthrough control specifies q_keep. Evaluation disables this masking and scaling.

For activation a=2 and p_drop=0.5 (q_keep=0.5), the training output is either 0 or 4, each with probability 0.5, so its expected value is 2. It is not 2 on every forward pass. Under K-class label smoothing convention, target distribution is (1−epsilon)one_hot+epsilon/K: for K=4 and epsilon=0.1, target class is 0.925 and each other class 0.025.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing a response to training failures

Try a smaller model, more representative data, justified augmentation or a changed regularizer based on observed failure. A drop probability of 0.9 (keep probability 0.1) can remove useful signal; no smooth monotonic improvement is promised. Loss reduction and class weighting affect gradient scale, so record their interaction with learning rate.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare dropout with the same split

Train a small image MLP under identical splits with dropout off/on. Save the best development checkpoint including an unchanged baseline where appropriate. Keep transformations used for training out of deterministic evaluation unless a separate test-time augmentation protocol is declared.

**Check.** Development loss is noisy. May you choose whichever checkpoint gives the prettiest test samples?

**Answer.** That uses test evidence for selection. Use a declared development criterion and reserve test prompts/scores for the final check.

## Sources

- [CS231n practical neural-network training notes](https://cs231n.github.io/neural-networks-3/).
- [PyTorch 2.9 CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html).
- [Deep Learning, author site](https://www.deeplearningbook.org/).
