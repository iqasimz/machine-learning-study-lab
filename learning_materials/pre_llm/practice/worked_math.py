"""Inspectable arithmetic for the LLMs A to Z lessons; standard library only."""

import math
import random


def softmax(logits, temperature=1.0):
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Use nonempty logits and a finite positive temperature")
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("Logits must be finite")
    scaled = [x / temperature for x in logits]
    largest = max(scaled)
    values = [math.exp(x - largest) for x in scaled]
    total = sum(values)
    return [x / total for x in values]


def validate_distribution(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Probabilities must sum to one")


def cross_entropy_target(probabilities, target):
    validate_distribution(probabilities)
    if not 0 <= target < len(probabilities):
        raise ValueError("Target is outside the distribution")
    value = probabilities[target]
    return -math.log(value) if value else math.inf


def entropy(probabilities):
    validate_distribution(probabilities)
    return -sum(p * math.log(p) for p in probabilities if p)


def kl_divergence(p, q):
    validate_distribution(p)
    validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("Distributions must have the same support size")
    if any(a > 0 and b == 0 for a, b in zip(p, q)):
        return math.inf
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a)


def dot(a, b):
    if not a or len(a) != len(b):
        raise ValueError("Vectors must be nonempty and equally sized")
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    numerator = dot(a, b)
    denominator = math.sqrt(dot(a, a) * dot(b, b))
    if denominator == 0:
        raise ValueError("Cosine is undefined for a zero vector")
    return numerator / denominator


def attention_row(query, keys, values, allowed=None):
    if not keys or len(keys) != len(values):
        raise ValueError("Provide one value vector per key")
    if not query or any(len(k) != len(query) for k in keys):
        raise ValueError("Keys must match query width")
    width = len(values[0])
    if width == 0 or any(len(v) != width for v in values):
        raise ValueError("Value widths must agree")
    if allowed is None:
        allowed = [True] * len(keys)
    if len(allowed) != len(keys) or not any(allowed):
        raise ValueError("At least one key must be allowed")
    included = [i for i, ok in enumerate(allowed) if ok]
    scores = [dot(query, keys[i]) / math.sqrt(len(query)) for i in included]
    local = softmax(scores)
    weights = [0.0] * len(keys)
    for i, probability in zip(included, local):
        weights[i] = probability
    output = [sum(w * v[j] for w, v in zip(weights, values)) for j in range(width)]
    return weights, output


def dpo_loss(winning_log_ratio, losing_log_ratio, beta=1.0):
    if beta <= 0 or not all(math.isfinite(x) for x in
                            (winning_log_ratio, losing_log_ratio, beta)):
        raise ValueError("Use finite log ratios and a positive beta")
    margin = beta * (winning_log_ratio - losing_log_ratio)
    # softplus(-margin), written without unstable exponentiation.
    return max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))


def grpo_advantages(rewards, epsilon=1e-8):
    if not rewards or epsilon <= 0 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("Use finite rewards and a positive epsilon")
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance)
    return [(r - mean) / (std + epsilon) for r in rewards]


def ppo_clipped_term(ratio, advantage, epsilon=0.2):
    if ratio < 0 or not 0 <= epsilon < 1:
        raise ValueError("Ratio must be nonnegative; epsilon must be in [0,1)")
    clipped = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped * advantage)


def kv_cache_bytes(layers, sequences, tokens, kv_heads, head_width, bytes_per_value):
    values = (layers, sequences, tokens, kv_heads, head_width, bytes_per_value)
    if any(not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("Cache dimensions must be positive integers")
    return 2 * math.prod(values)


def holm_rejections(p_values, alpha=0.05):
    if not 0 < alpha < 1 or any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("Use valid probabilities and alpha")
    rejected = []
    for rank, (original_index, p) in enumerate(sorted(enumerate(p_values), key=lambda x: x[1])):
        if p <= alpha / (len(p_values) - rank):
            rejected.append(original_index)
        else:
            break
    return rejected


def paired_bootstrap_accuracy_differences(correct_a, correct_b, repeats=1000, seed=17):
    if not correct_a or len(correct_a) != len(correct_b) or repeats <= 0:
        raise ValueError("Provide equally sized nonempty paired outcomes")
    if any(x not in (0, 1) for x in list(correct_a) + list(correct_b)):
        raise ValueError("Correctness values must be 0 or 1")
    rng = random.Random(seed)
    n = len(correct_a)
    differences = []
    for _ in range(repeats):
        indices = [rng.randrange(n) for _ in range(n)]
        differences.append(sum(correct_a[i] - correct_b[i] for i in indices) / n)
    return differences


def main():
    p = softmax([2.0, 1.0, 0.0])
    weights, output = attention_row([1, 0], [[1, 0], [0, 1]], [[2, 0], [0, 4]])
    print("Softmax [2,1,0]:", [round(x, 6) for x in p])
    print("Loss for target index 1:", round(cross_entropy_target(p, 1), 6))
    print("Entropy in nats:", round(entropy(p), 6))
    print("KL([.5,.5] || [.9,.1]):", round(kl_divergence([.5, .5], [.9, .1]), 6))
    print("Attention weights/output:", weights, output)
    print("DPO loss for log ratios .4 and -.2:", round(dpo_loss(.4, -.2), 6))
    print("GRPO population-normalized advantages:", grpo_advantages([0, 0, 1, 1]))
    print("PPO positive / negative examples:", ppo_clipped_term(1.5, 2), ppo_clipped_term(.5, -2))
    print("Conventional KV bytes:", kv_cache_bytes(32, 1, 8192, 8, 128, 2))
    print("Holm rejected original indices:", holm_rejections([.01, .03, .04]))
    print("Toy bootstrap first 8 differences:", paired_bootstrap_accuracy_differences(
        [1, 1, 0, 1], [1, 0, 0, 1], repeats=8))


if __name__ == "__main__":
    main()
