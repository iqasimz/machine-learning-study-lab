"""DL reader semantics and independent checks for the teaching calculations."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_lab import curriculum as c


class DLReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = r"""
const fs=require('fs'),vm=require('vm'),v=vm.createContext({});
for(const f of ['architecture_math','explorer_lessons','explorer_dl'])vm.runInContext(fs.readFileSync(process.argv[1]+'/'+f+'.js','utf8'),v);
const result=vm.runInContext("(()=>{const b=(id,c)=>ConceptLessons.build(id,c),number=(s,re)=>parseFloat(s.match(re)[1]);return {recurrent:['rnn','gru','lstm'].map(variant=>({variant,states:b('DL07',{variant}).steps.slice(1,4).map(s=>number(s.calculation,/h[₁₂₃]=([0-9.-]+)/)),cells:variant==='lstm'?b('DL07',{variant}).steps.slice(1,4).map(s=>number(s.calculation,/c[₁₂₃]=([0-9.-]+)/)):null})),dropout:[.5,.75,1].map(keep=>({keep,values:JSON.parse(b('DL04',{keep}).steps[1].calculation.match(/\\[[^\\]]+\\]/)[0])})),graph:['mean','sum'].map(aggregation=>({aggregation,node:number(b('DL11',{aggregation}).steps[3].calculation,/Updated B = ([0-9.]+)/),readout:number(b('DL11',{aggregation}).steps[4].calculation,/Sum readout=([0-9.]+)/)})),auto:[1,2].map(width=>number(b('DL09',{width}).steps[2].calculation,/Loss = ([0-9.]+)/)),vae:number(b('DL10',{family:'vae'}).steps[3].calculation,/KL = ([0-9.]+)/),gan:b('DL10',{family:'gan'}).steps[4].calculation,diffusion:b('DL10',{family:'diffusion'}).steps.map(s=>s.calculation)}})()",v);
console.log(JSON.stringify(result));
"""
        cls.data = json.loads(subprocess.check_output(
            ["node", "-e", script, str(ROOT / "src/training_lab/web")], text=True))

    def test_dl_scope_and_separate_computations(self):
        self.assertTrue(all(f"DL{i:02}" in c.READER_PROFILES for i in range(1, 15)))
        p = c.READER_PROFILES
        self.assertEqual(p["DL05"]["view"], "architecture")
        for identifier in ("DL03", "DL04", "DL12", "DL14"):
            self.assertEqual(p[identifier]["view"], "reading")
        for identifier, last in (("DL06", 2), ("DL07", 3), ("DL09", 2), ("DL11", 3)):
            self.assertEqual(max(p[identifier]["diagram"]["nodes"]), last)
        vae = {(a, b) for a, b, *_ in p["DL10"]["variants"]["vae"]["diagram"]["links"]}
        self.assertEqual(vae, {(0, 1), (1, 2), (0, 3)})
        training = {(a, b) for a, b, *_ in p["DL13"]["diagram"]["links"]}
        self.assertEqual(training, {(0, 1), (1, 2), (2, 0), (2, 3)})

    def test_consistent_source_conventions_and_retained_metrics(self):
        dropout = c.document("DL04")["markdown"]
        self.assertIn("q_keep=1−p_drop", dropout)
        gru = c.document("DL07")["markdown"]
        self.assertIn("0.75×0.2+0.25×0.8=0.35", gru)
        self.assertNotIn("h_t=0.65", gru)
        self.assertFalse(any("alternative convention below" in n["markdown"]
                             for n in c.document("DL07")["reader"]["notes"]))
        self.assertIn("live fixed-map walkthrough sums", c.document("DL09")["markdown"])
        notes = c.document("DL12")["reader"]["notes"]
        overlap = next(n for n in notes if n["title"] == "Region overlap: IoU and Dice")
        self.assertIn("IoU=3/9=0.3333", overlap["markdown"])
        self.assertIn("=0.5", overlap["markdown"])

    def test_recurrent_values_match_pytorch_cells(self):
        for row in self.data["recurrent"]:
            variant = row["variant"]
            cell = {"rnn": torch.nn.RNNCell, "gru": torch.nn.GRUCell,
                    "lstm": torch.nn.LSTMCell}[variant](1, 1, dtype=torch.float64)
            with torch.no_grad():
                for param in cell.parameters():
                    param.zero_()
                if variant == "rnn":
                    cell.weight_ih[0, 0] = .5
                    cell.weight_hh[0, 0] = .8
                elif variant == "gru":
                    cell.weight_ih[2, 0] = .5
                    cell.weight_hh[2, 0] = .8
                    cell.bias_ih[1] = math.log(.25 / .75)
                else:
                    cell.weight_ih[2, 0] = .5
                    cell.bias_ih[1] = math.log(.8 / .2)
                    cell.bias_ih[3] = math.log(.7 / .3)
                h = torch.zeros((1, 1), dtype=torch.float64)
                memory = h.clone()
                values, cells = [], []
                for x in [1., 0., 1.]:
                    value = torch.tensor([[x]], dtype=torch.float64)
                    if variant == "lstm":
                        h, memory = cell(value, (h, memory))
                        cells.append(float(memory[0, 0]))
                    else:
                        h = cell(value, h)
                    values.append(float(h[0, 0]))
            np.testing.assert_allclose(values, row["states"], atol=5.1e-5)
            if variant == "lstm":
                np.testing.assert_allclose(cells, row["cells"], atol=5.1e-5)

    def test_dropout_graph_updates_and_reconstruction_reduction(self):
        features = np.array([2., 4.])
        for row in self.data["dropout"]:
            q = row["keep"]
            mask = np.array([1, 1] if q == 1 else [1, 0])
            np.testing.assert_allclose(mask * features / q, row["values"], atol=5.1e-5)
            expectation = np.zeros(2)
            for bits in itertools.product([0, 1], repeat=2):
                probability = np.prod([q if bit else 1-q for bit in bits])
                expectation += probability * np.array(bits) * features / q
            np.testing.assert_allclose(expectation, features)
            layer = torch.nn.Dropout(p=1-q).eval()
            np.testing.assert_allclose(layer(torch.tensor(features)).numpy(), features)
        adjacency = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
        h = torch.tensor([1.,2.,4.])
        for row in self.data["graph"]:
            messages = adjacency @ h
            if row["aggregation"] == "mean":
                messages /= adjacency.sum(1)
            updated = torch.relu(.5 * (h + messages))
            self.assertAlmostEqual(float(updated[1]), row["node"])
            self.assertAlmostEqual(float(updated.sum()), row["readout"])
        x = torch.tensor([1., 0.])
        for width, actual in zip([1, 2], self.data["auto"]):
            reconstruction = torch.tensor([.5, .5]) if width == 1 else x
            summed = torch.nn.functional.mse_loss(reconstruction, x, reduction="sum")
            mean = torch.nn.functional.mse_loss(reconstruction, x, reduction="mean")
            self.assertAlmostEqual(float(summed), actual)
            self.assertAlmostEqual(float(summed), 2 * float(mean))

    def test_generative_objectives_are_separate(self):
        normal = torch.distributions.Normal
        kl = torch.distributions.kl_divergence(normal(.5, .5), normal(0., 1.))
        self.assertAlmostEqual(float(kl), self.data["vae"], delta=5.1e-5)
        self.assertIn(f"{-math.log(.8):.4f}", self.data["gan"])
        self.assertIn(f"{-math.log(.9)-math.log(.2):.4f}", self.data["gan"])
        noisy = math.sqrt(.64) * 1 + math.sqrt(1-.64) * .5
        restored = (noisy - math.sqrt(1-.64) * .5) / math.sqrt(.64)
        self.assertAlmostEqual(noisy, 1.1)
        self.assertAlmostEqual(restored, 1.)
        self.assertIn("= 1.1", self.data["diffusion"][1])
        self.assertIn("= 1.", self.data["diffusion"][3])


if __name__ == "__main__":
    unittest.main()

