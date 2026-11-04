"""LLM reader boundaries, independently checked teaching arithmetic and retained depth."""
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_lab import curriculum as c


class LLMReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = r'''
const fs=require('fs'),vm=require('vm'),v=vm.createContext({});
for(const f of ['architecture_math','explorer_lessons','explorer_llm'])
 vm.runInContext(fs.readFileSync(process.argv[1]+'/'+f+'.js','utf8'),v);
console.log(vm.runInContext(`JSON.stringify((()=>{
 const b=(id,c)=>ConceptLessons.build(id,c).steps.map(s=>({calculation:s.calculation,equation:s.equation}));
 return {dpo:b('R2',{method:'dpo'}),ppo:b('R2',{method:'ppo'}),grpo:b('R2',{method:'grpo'}),
 lora:[1,2].map(rank=>b('R1',{rank})),
 accumulation:[2,6,10].map(second=>b('T5',{second})),
 distributed:[2,6,10].map(second=>b('T7',{second})),
 cache:[16,32,64].map(tokens=>b('R6',{tokens})),
 quant:[.1,.2,.4].map(scale=>b('R7',{scale})),
 steering:[0,.5,1].map(alpha=>b('R9',{alpha})),
 denied:b('R5',{approval:'deny'}),json:b('R3',{})};
})())`,v));
'''
        cls.data = json.loads(subprocess.check_output(
            ["node", "-e", script, str(ROOT / "src/training_lab/web")], text=True))

    def test_graphs_separate_execution_accounting_and_denial(self):
        p = c.READER_PROFILES
        self.assertEqual(len(p), 79)
        self.assertEqual(p["T3"]["view"], "architecture")
        self.assertEqual(len(p["T3"]["steps"]), 25)
        for identifier in ("T1", "T6", "T8", "R2", "R8"):
            self.assertEqual(p[identifier]["view"], "reading")
        self.assertEqual(p["R1"]["diagram"]["nodes"], [0, 1, 2, 3])
        denied = p["R5"]["variants"]["deny"]["diagram"]
        self.assertNotIn(3, denied["nodes"])
        self.assertIn((2, 4), {(a, b) for a, b, *_ in denied["links"]})
        cache_edges = {(a, b) for a, b, *_ in p["R6"]["diagram"]["links"]}
        self.assertIn((1, 3), cache_edges)
        self.assertNotIn((2, 3), cache_edges)
        training = {(a, b) for a, b, *_ in p["T0"]["diagram"]["links"]}
        self.assertIn((4, 0), training)
        self.assertNotIn((6, 0), training)
        self.assertEqual(set(p["T4"]["variants"]), {"encoder", "decoder", "experts", "state"})
        self.assertEqual(set(p["R2"]["variants"]), {"ppo", "grpo"})
        self.assertEqual(self.data["denied"][3]["calculation"], "No execution result is produced.")

    def test_preference_examples_against_torch_objectives(self):
        d = torch.tensor(math.log(3), dtype=torch.float64, requires_grad=True)
        loss = -torch.nn.functional.logsigmoid(.1 * d)
        actual = float(re.search(r"loss=([0-9]+(?:\.[0-9]+)?)", self.data["dpo"][2]["calculation"])[1])
        self.assertAlmostEqual(actual, float(loss.detach()), delta=5.1e-5)
        loss.backward()
        self.assertAlmostEqual(float(d.grad), -.1 * float(torch.sigmoid(-.1*d.detach())))
        for ratio, advantage, expected in ((1.5, 2., 2.4), (.5, -2., -1.6)):
            r = torch.tensor(ratio, dtype=torch.float64)
            surrogate = torch.minimum(r * advantage, r.clamp(.8, 1.2) * advantage)
            self.assertAlmostEqual(float(surrogate), expected)
        self.assertIn("min(3,2.4)=2.4", self.data["ppo"][2]["calculation"])
        rewards = torch.tensor([0., 1., 2.], dtype=torch.float64)
        advantages = (rewards-rewards.mean()) / rewards.std(correction=0)
        displayed = json.loads(self.data["grpo"][2]["calculation"].split("Advantages ")[1].rstrip("."))
        np.testing.assert_allclose(displayed, advantages.numpy(), atol=5.1e-5)

    def test_lora_and_token_weighted_reduction(self):
        x = torch.tensor([[2., 1.]], dtype=torch.float64)
        for rank, row in zip([1, 2], self.data["lora"]):
            a = torch.tensor([[1.], [-1.]], dtype=torch.float64) if rank == 1 else torch.eye(2, dtype=torch.float64)
            b = torch.tensor([[.1, .2]], dtype=torch.float64) if rank == 1 else torch.tensor([[.1, .2], [-.1, 0.]], dtype=torch.float64)
            expected = x + (2/rank)*x@a@b
            actual = json.loads(row[3]["calculation"].split("Adapted output=")[1].rstrip("."))
            np.testing.assert_allclose(actual, expected.numpy()[0], atol=5.1e-5)
        a = torch.tensor([[1.], [-1.]], dtype=torch.float64, requires_grad=True)
        b = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        (x + 2*x@a@b).square().sum().backward()
        self.assertEqual(float(a.grad.abs().sum()), 0.)
        self.assertGreater(float(b.grad.abs().sum()), 0.)
        for second, accum, distributed in zip([2, 6, 10], self.data["accumulation"], self.data["distributed"]):
            theta = torch.tensor(1., dtype=torch.float64, requires_grad=True)
            losses = torch.cat([theta.repeat(2), (3*theta).repeat(second)])
            losses.mean().backward()
            target = float(theta.grad)
            actual = float(re.search(r"Correct mean ([0-9.]+)", accum[2]["calculation"])[1])
            self.assertAlmostEqual(actual, target, delta=5.1e-5)
            actual = float(re.search(r"Result ([0-9.]+)", distributed[3]["calculation"])[1])
            self.assertAlmostEqual(actual, target, delta=5.1e-5)

    def test_cache_quantization_and_steering_arithmetic(self):
        for tokens, row in zip([16, 32, 64], self.data["cache"]):
            actual = int(re.search(r"= ([0-9]+) bytes", row[2]["calculation"])[1])
            cache = np.empty((2, 1, 2, tokens, 2, 4), dtype=np.float16)
            self.assertEqual(actual, cache.nbytes)
        for scale, row in zip([.1, .2, .4], self.data["quant"]):
            w = torch.tensor([-1.2, .4, .8], dtype=torch.float64)
            reconstructed = (w/scale).round().clamp(-8, 7)*scale
            mse = float(torch.nn.functional.mse_loss(reconstructed, w))
            actual = float(re.search(r"MSE=([0-9]+(?:\.[0-9]+)?)", row[2]["calculation"])[1])
            self.assertAlmostEqual(actual, mse, delta=5.1e-5)
        for alpha, row in zip([0, .5, 1], self.data["steering"]):
            h = torch.tensor([[.2, -.1]], dtype=torch.float64)
            direction = torch.tensor([[1., 0.]], dtype=torch.float64)
            projection = torch.tensor([[1.], [-1.]], dtype=torch.float64)
            expected = float(((h+alpha*direction)@projection).item())
            actual = float(re.search(r"Score = ([0-9]+(?:\.[0-9]+)?)", row[2]["calculation"])[1])
            self.assertAlmostEqual(actual, expected, delta=5.1e-5)
            self.assertEqual(row[2]["equation"], "s'=h'[1,-1]^T")

    def test_visible_depth_and_precise_source_conventions(self):
        notes = lambda identifier: "\n".join(n.get("title", "") + "\n" + n["markdown"] for n in c.document(identifier)["reader"]["notes"])
        for identifier, phrase in (("T0", "Check the complete update boundary"),
                                   ("T1", "Why one forward pass can learn many next tokens"),
                                   ("T5", "A two-microbatch update"),
                                   ("T6", "361,821,120"), ("T7", "ordinary DDP averaging"),
                                   ("R1", "zero initial adapter output"),
                                   ("R2", "KL(policy || reference)"),
                                   ("R8", "circuit breakers"), ("R9", "sparsity objective")):
            self.assertIn(phrase, notes(identifier))
        r9 = notes("R9")
        self.assertEqual(r9.count("### Evaluate a steering direction as an intervention, not a label"), 1)
        r6 = c.document("R6")["markdown"]
        self.assertNotIn("At 10 requests/s and 2 s", r6)
        self.assertIn('{"answer":5}', self.data["json"][3]["calculation"])
        self.assertNotIn("{answer:5}", self.data["json"][3]["calculation"])


if __name__ == "__main__":
    unittest.main()
