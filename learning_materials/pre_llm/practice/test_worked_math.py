"""Checks for the small worked examples; no model or third-party dependencies."""

import math
import unittest

from worked_math import (attention_row, cosine, cross_entropy_target, dpo_loss,
                         entropy, grpo_advantages, holm_rejections,
                         kl_divergence, kv_cache_bytes,
                         paired_bootstrap_accuracy_differences,
                         ppo_clipped_term, softmax)


class WorkedMathTests(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([2, 1, 0])), 1)

    def test_softmax_shift_invariance(self):
        for a, b in zip(softmax([1, 2]), softmax([1001, 1002])):
            self.assertAlmostEqual(a, b)

    def test_lower_temperature_sharpens(self):
        self.assertGreater(softmax([2, 0], .5)[0], softmax([2, 0], 2)[0])

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            softmax([1, 2], 0)

    def test_cross_entropy(self):
        self.assertAlmostEqual(cross_entropy_target([.25, .75], 0), math.log(4))

    def test_invalid_distribution(self):
        with self.assertRaises(ValueError):
            entropy([.2, .2])

    def test_entropy(self):
        self.assertAlmostEqual(entropy([.5, .5]), math.log(2))
        self.assertEqual(entropy([1, 0]), 0)

    def test_kl(self):
        self.assertAlmostEqual(kl_divergence([.5, .5], [.9, .1]), .5108256237659907)
        self.assertTrue(math.isinf(kl_divergence([.5, .5], [1, 0])))

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 1], [1, 0]), 1 / math.sqrt(2))
        with self.assertRaises(ValueError):
            cosine([0, 0], [1, 1])

    def test_attention(self):
        weights, output = attention_row([1, 0], [[1, 0], [0, 1]], [[2, 0], [0, 4]])
        self.assertAlmostEqual(weights[0], .6697615493266569)
        self.assertAlmostEqual(output[0], 1.3395230986533138)
        self.assertAlmostEqual(output[1], 1.3209538026933724)

    def test_attention_mask(self):
        weights, output = attention_row([1, 0], [[1, 0], [0, 1]], [[2, 0], [0, 4]], [True, False])
        self.assertEqual(weights, [1, 0])
        self.assertEqual(output, [2, 0])

    def test_all_masked_rejected(self):
        with self.assertRaises(ValueError):
            attention_row([1], [[1]], [[2]], [False])

    def test_dpo(self):
        self.assertAlmostEqual(dpo_loss(.4, -.2), .4374879504858856)
        self.assertAlmostEqual(dpo_loss(0, 0), math.log(2))
        self.assertLess(dpo_loss(2, -2), dpo_loss(0, 0))

    def test_grpo(self):
        a = grpo_advantages([0, 0, 1, 1])
        self.assertAlmostEqual(a[0], -1, places=6)
        self.assertAlmostEqual(sum(a), 0)
        self.assertEqual(grpo_advantages([3, 3, 3]), [0, 0, 0])

    def test_ppo_sign(self):
        self.assertAlmostEqual(ppo_clipped_term(1.5, 2), 2.4)
        self.assertAlmostEqual(ppo_clipped_term(.5, -2), -1.6)

    def test_cache(self):
        self.assertEqual(kv_cache_bytes(32, 1, 8192, 8, 128, 2), 1024**3)
        self.assertEqual(kv_cache_bytes(32, 2, 8192, 8, 128, 2), 2 * 1024**3)

    def test_holm(self):
        self.assertEqual(holm_rejections([.01, .03, .04]), [0])
        self.assertEqual(holm_rejections([]), [])
        self.assertEqual(holm_rejections([.04, .001]), [1, 0])

    def test_bootstrap_pairing_and_seed(self):
        a, b = [1, 1, 0, 1], [1, 0, 0, 1]
        results = paired_bootstrap_accuracy_differences(a, b, repeats=100)
        self.assertEqual(results, paired_bootstrap_accuracy_differences(a, b, repeats=100))
        self.assertTrue(all(0 <= x <= 1 for x in results))
        self.assertEqual(paired_bootstrap_accuracy_differences(a, a, repeats=3), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
