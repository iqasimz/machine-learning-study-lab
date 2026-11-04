"""Advanced sections stay inside existing lectures; arithmetic is independently checked."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import curriculum as c, guided_course as g
from training_lab.lesson_reader import ADVANCED


class AdvancedSectionsTests(unittest.TestCase):
    def test_exact_scope_and_canonical_reader(self):
        self.assertEqual(set(ADVANCED), {"ML07", "ML14", "ML15", "ML16", "DL05", "DL06", "DL10", "DL11", "DL12", "DL13"})
        self.assertEqual(sum(map(len, ADVANCED.values())), 27)
        self.assertEqual(len(c.LESSONS), 79)
        self.assertEqual(len({row[0] for row in c.LESSONS}), 79)
        all_ids = set()
        for identifier, declared in ADVANCED.items():
            doc = c.document(identifier)
            notes = [n for n in doc["reader"]["notes"] if n.get("level") == "advanced"]
            self.assertEqual([n["advanced_id"] for n in notes], [v["id"] for v in declared.values()])
            for note in notes:
                self.assertNotIn(note["advanced_id"], all_ids)
                all_ids.add(note["advanced_id"])
                self.assertGreater(len(note["markdown"].split()), 180)
                self.assertIn("**Practice.**", note["markdown"])
                self.assertIn("**Answer.**", note["markdown"])
                self.assertIn("https://", note["markdown"])
                self.assertIn(note["markdown"], doc["markdown"])
            self.assertEqual(g.lesson(identifier)["steps"][0]["document"]["reader"], doc["reader"])
        self.assertEqual(len(all_ids), 27)

    def test_advanced_checks_and_preserved_base_step_ids(self):
        additions = g._json("advanced_checks.json")["lessons"]
        self.assertEqual(set(additions), set(ADVANCED))
        self.assertEqual(sum(map(len, additions.values())), 25)
        self.assertEqual(sum(map(len, g.checks().values())), 183)
        for identifier in ADVANCED:
            steps = [s for s in g.lesson(identifier)["steps"] if s["kind"] == "question"]
            self.assertEqual(steps[0]["id"], identifier + ":check-a")
            self.assertEqual(steps[1]["id"], identifier + ":check-b")
            self.assertTrue(all(s["title"] == "Advanced check" for s in steps[2:]))

    def test_bayesian_update_graph_query_and_vi_gap(self):
        posterior = torch.distributions.Beta(torch.tensor(5.), torch.tensor(3.))
        self.assertAlmostEqual(float(posterior.mean), .625)
        weights = np.array([.5*.2*.1, .5*.8*.6])
        self.assertAlmostEqual((weights/weights.sum())[1], .96)
        q = torch.distributions.Bernoulli(torch.tensor(.5, dtype=torch.float64))
        p = torch.distributions.Bernoulli(torch.tensor(.8, dtype=torch.float64))
        gap = float(torch.distributions.kl_divergence(q, p))
        self.assertAlmostEqual(gap, .2231435513142)
        self.assertAlmostEqual(-2-gap, -2.2231435513142)

    def test_factorization_bandit_q_and_actor_critic(self):
        u = torch.tensor([1., 1.], requires_grad=True)
        v = torch.tensor([1., 2.])
        loss = .5*(u@v-5).square()
        loss.backward()
        torch.testing.assert_close(u.grad, torch.tensor([-2., -4.]))
        self.assertAlmostEqual(float(((u-.1*u.grad)@v).detach()), 4.)
        self.assertAlmostEqual(.8+.2/2, .9)
        self.assertAlmostEqual(2+.25*(1+.9*4-2), 2.65)
        self.assertAlmostEqual(2+.25*(1-2), 1.75)
        theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        log_prob = torch.nn.functional.logsigmoid(theta)
        log_prob.backward()
        td = 1+.9*3-2
        self.assertAlmostEqual(td, 1.7)
        updated = theta.detach()+.1*td*theta.grad
        self.assertAlmostEqual(float(updated), .085)
        self.assertAlmostEqual(float(updated.sigmoid()), .521237214966)
        self.assertAlmostEqual(2+.2*td, 2.34)

    def test_vision_shapes_contrastive_and_detection(self):
        image = torch.zeros(1, 3, 224, 224)
        patches = torch.nn.Unfold(kernel_size=16, stride=16)(image).transpose(1, 2)
        self.assertEqual(tuple(patches.shape), (1, 196, 768))
        self.assertEqual(tuple(torch.nn.Linear(768, 192)(patches).shape), (1, 196, 192))
        self.assertEqual(sum(p.numel() for p in torch.nn.Linear(768, 192).parameters()), 147648)
        self.assertAlmostEqual(float(torch.softmax(torch.tensor([1., 0.]), 0)[0]), .7310586, places=6)
        a = np.array([0., 0., 2., 2.]); b = np.array([.5, 0., 2.5, 2.])
        overlap = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
        self.assertAlmostEqual(overlap/(4+4-overlap), .6)
        costs = np.array([[1., 2.], [1.1, 100.]])
        self.assertAlmostEqual(float(costs[0, 1]+costs[1, 0]), 3.1)

    def test_generative_and_graph_arithmetic(self):
        q = torch.distributions.Normal(torch.tensor(1., dtype=torch.float64), torch.tensor(.5, dtype=torch.float64))
        p = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
        self.assertAlmostEqual(float(torch.distributions.kl_divergence(q,p)), .81814718056)
        self.assertAlmostEqual(-2*math.log(.8), .446287102628)
        self.assertAlmostEqual(-math.log(.2), 1.60943791243)
        xt = math.sqrt(.64)*1+math.sqrt(.36)*(-.5)
        self.assertAlmostEqual(xt, .5)
        self.assertAlmostEqual((-.5-(-.4))**2, .01)
        self.assertAlmostEqual((xt-math.sqrt(.36)*(-.4))/math.sqrt(.64), .925)
        self.assertAlmostEqual(.2+2*(.5-.2), .8)
        self.assertEqual((256*256*3)/(32*32*4), 48)
        self.assertEqual((0-1)**2+1+4-2*1*2, 2)
        weights = torch.softmax(torch.tensor([math.log(3),0.], dtype=torch.float64),0)
        self.assertAlmostEqual(float(weights@torch.tensor([2.,6.], dtype=torch.float64)), 3.)
        self.assertAlmostEqual(float(torch.sigmoid(torch.tensor(.5, dtype=torch.float64))), .6224593312)


if __name__ == "__main__":
    unittest.main()
