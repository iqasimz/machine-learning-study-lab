"""Independent PyTorch references for the browser's teaching calculations."""
import json
from pathlib import Path
import subprocess
import unittest

import torch
import torch.nn.functional as F

WEB = Path(__file__).resolve().parents[1] / "src/training_lab/web"


def trace(kind, options=None):
    code = """const fs=require('fs'),vm=require('vm');
    const c=vm.createContext({});
    vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),c);
    console.log(vm.runInContext('JSON.stringify(ArchitectureMath.'+process.argv[2]+'('+process.argv[3]+'))',c));"""
    result = subprocess.run(
        ["node", "-e", code, str(WEB / "architecture_math.js"), kind, json.dumps(options or {})],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def tensor(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


class ArchitectureMathTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(tensor(actual), expected.detach(), atol=1e-11, rtol=1e-10)

    def check_transformer(self, options):
        r = trace("transformer", options)
        x = tensor(r["embedding"]) + tensor(r["positions"])
        self.close(r["start"], x)
        n, heads = len(r["ids"]), r["cfg"]["heads"]
        width = 4 // heads
        for b in r["blocks"]:
            self.close(b["input"], x)
            z = F.layer_norm(x, (4,), eps=1e-5)
            self.close(b["normalized"], z)
            w = {key: tensor(value) for key, value in b["weights"].items()}
            q, k, v = (z @ w[key] for key in ("wq", "wk", "wv"))
            for name, value in (("q", q), ("k", k), ("v", v)):
                self.close(b[name], value)
            pieces = []
            for h in range(heads):
                sl = slice(h * width, (h + 1) * width)
                scores = q[:, sl] @ k[:, sl].T / width**.5
                self.close(b["scores"][h], scores)
                masked = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), -torch.inf)
                # JSON's null encodes the intentional -Infinity mask entries.
                for row, values in enumerate(b["masked"][h]):
                    for col, value in enumerate(values):
                        if col > row:
                            self.assertIsNone(value)
                        else:
                            self.assertAlmostEqual(value, float(masked[row, col]), places=12)
                attention = masked.softmax(-1)
                self.close(b["attention"][h], attention)
                mixed = attention @ v[:, sl]
                self.close(b["heads"][h], mixed)
                pieces.append(mixed)
            joined = torch.cat(pieces, -1)
            projected = joined @ w["wo"]
            residual = x + projected
            normalized = F.layer_norm(residual, (4,), eps=1e-5)
            expanded = normalized @ w["w1"]
            activated = F.gelu(expanded, approximate="tanh")
            mlp = activated @ w["w2"]
            x = residual + mlp
            for name, value in (("joined", joined), ("projected", projected), ("residual", residual),
                                ("norm2", normalized), ("expanded", expanded), ("activated", activated),
                                ("mlp", mlp), ("output", x)):
                self.close(b[name], value)
        final = F.layer_norm(x, (4,), eps=1e-5)
        self.close(r["final"], final)
        head = tensor(r["headWeights"], grad=True)
        logits = final @ head
        self.close(r["logits"], logits)
        self.close(r["probabilities"], logits.softmax(-1))
        position = r["cfg"]["position"]
        loss = F.cross_entropy(logits[position:position+1], torch.tensor([r["target"]]))
        self.assertAlmostEqual(r["loss"], loss.item(), places=12)
        loss.backward()
        self.close(r["gradient"], head.grad)
        after = head.detach() - .1 * head.grad
        self.close(r["updatedHead"], after)
        new_loss = F.cross_entropy((final @ after)[position:position+1], torch.tensor([r["target"]]))
        self.assertAlmostEqual(r["updatedLoss"], new_loss.item(), places=12)
        self.assertLess(r["updatedLoss"], r["loss"])
        self.assertEqual(r["nextId"], int(logits[-1].argmax()))

    def test_transformer_heads_layers_and_positions(self):
        for heads in (1, 2, 4):
            for layers in (1, 2, 3):
                with self.subTest(heads=heads, layers=layers):
                    self.check_transformer(dict(heads=heads, layers=layers, block=layers-1,
                                                head=heads-1, position=0))

    def test_transformer_long_sequence_and_middle_loss(self):
        self.check_transformer(dict(ids=[0, 1, 2, 3, 4, 5], position=3, block=1))
        self.check_transformer({})

    def test_future_tokens_do_not_change_earlier_outputs(self):
        a = trace("transformer", dict(ids=[0, 1, 2], position=0))
        b = trace("transformer", dict(ids=[0, 4, 5], position=0))
        self.assertEqual(a["logits"][0], b["logits"][0])
        self.assertNotEqual(a["logits"][-1], b["logits"][-1])
        # Inspecting a different block/head is a view, not a different network.
        c = trace("transformer", dict(block=1, head=1))
        self.assertEqual(a["logits"], c["logits"])

    def check_cnn(self, options):
        r = trace("cnn", options)
        x = tensor([r["input"]])
        c = r["cfg"]
        z1 = F.conv2d(x, tensor(r["f1"]), tensor(r["b1"]),
                      stride=c["stride"], padding=r["padding"])
        a1 = z1.relu()
        p1 = F.max_pool2d(a1, 2)
        z2 = F.conv2d(p1, tensor(r["f2"]), tensor(r["b2"]), padding=1)
        a2 = z2.relu()
        p2 = F.max_pool2d(a2, 2)
        flat = p2.flatten(1)
        for name, value in (("z1", z1), ("a1", a1), ("p1", p1),
                            ("z2", z2), ("a2", a2), ("p2", p2), ("flat", flat)):
            self.close(r[name], value[0])
        products = tensor(r["patch"]) * tensor(r["f1"][c["filter"]][0])
        self.close(r["products"], products)
        expected = products.sum().item() + r["b1"][c["filter"]]
        self.assertAlmostEqual(expected, r["z1"][c["filter"]][c["row"]][c["col"]], places=12)
        w, bias = tensor(r["w"], grad=True), tensor(r["bias"], grad=True)
        logits = flat @ w + bias
        self.close(r["logits"], logits[0])
        self.close(r["probabilities"], logits.softmax(-1)[0])
        loss = F.cross_entropy(logits, torch.tensor([r["target"]]))
        self.assertAlmostEqual(r["loss"], loss.item(), places=12)
        loss.backward()
        self.close(r["gradient"], w.grad)
        after = flat @ (w.detach()-.1*w.grad) + bias.detach()-.1*bias.grad
        self.assertAlmostEqual(r["updatedLoss"], F.cross_entropy(after, torch.tensor([r["target"]])).item(), places=12)
        self.assertLess(r["updatedLoss"], r["loss"])
        counts = dict(conv1=tensor(r["f1"]).numel()+len(r["b1"]),
                      conv2=tensor(r["f2"]).numel()+len(r["b2"]),
                      head=w.numel()+bias.numel())
        self.assertEqual(r["parameters"], counts)
        self.assertEqual(r["totalParameters"], sum(counts.values()))

    def test_cnn_patterns_filters_stride_kernel_and_boundary_cells(self):
        for pattern in ("vertical", "horizontal", "cross"):
            for stride in (1, 2):
                for kernel in (3, 5):
                    for filt in (0, 1):
                        with self.subTest(pattern=pattern, stride=stride, kernel=kernel, filt=filt):
                            self.check_cnn(dict(pattern=pattern, stride=stride, kernel=kernel,
                                                filter=filt, row=0, col=8//stride-1))
        self.check_cnn({})

    def test_selected_cnn_cell_changes_inspection_not_predictions(self):
        a, b = trace("cnn"), trace("cnn", dict(row=0, col=0, filter=1))
        self.assertEqual(a["logits"], b["logits"])
        self.assertNotEqual(a["patch"], b["patch"])

    def test_invalid_settings_are_rejected(self):
        for kind, opts in (("transformer", dict(heads=3)), ("transformer", dict(block=5)),
                           ("transformer", dict(ids=[0], position=2)), ("cnn", dict(row=9)),
                           ("cnn", dict(stride=0)), ("cnn", dict(kernel=4))):
            with self.subTest(kind=kind, opts=opts):
                with self.assertRaises(subprocess.CalledProcessError):
                    trace(kind, opts)


if __name__ == "__main__":
    unittest.main()
