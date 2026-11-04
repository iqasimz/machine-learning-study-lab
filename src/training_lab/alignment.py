"""Differentiable teaching objectives, NOT claims of a completed LLM RLHF run."""
import torch
from torch.nn import functional as F


def response_example(tokenizer, prompt, response):
    """Explicit boundary: tokenize separately so a BPE merge cannot cross into labels."""
    prefix = [1] + tokenizer.encode(prompt)
    answer = tokenizer.encode(response) + [2]
    sequence = prefix + answer
    # Index i predicts sequence[i+1]; first response label follows the last prompt input.
    targets = [-100] * (len(prefix) - 1) + answer
    return sequence[:-1], targets


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Inputs are RESPONSE sequence log-probability sums, not raw token logits."""
    if beta <= 0:
        raise ValueError("beta must be positive.")
    margin = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margin).mean()


def ppo_policy_loss(new_logp, old_logp, advantage, epsilon=0.2):
    """Clipped policy term only. No critic, reward model, KL term or LLM rollout here."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1).")
    ratio = (new_logp - old_logp.detach()).exp()
    return -torch.minimum(ratio * advantage.detach(), ratio.clamp(1-epsilon, 1+epsilon) * advantage.detach()).mean()


def group_advantages(rewards):
    """Population-standardized group rewards; equal rewards have zero relative signal."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("At least two rewards required.")
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)
