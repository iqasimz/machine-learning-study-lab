"""Executable sequence/vision checks, using authored fixtures without network calls."""
from datetime import datetime, timedelta
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import model_data, model_lab, sequence_data
from training_lab.artifacts import read_json, sha256, write_json
from training_lab.sequence_models import SEQUENCE_MODELS, TemporalBlock, probe_training_device


def config(model, dataset="digits", task="classification", area="dl", **controls):
    return dict(area=area, dataset=dataset, task=task, model=model, controls=controls)


def fixture(root, name):
    folder = Path(root) / name; folder.mkdir()
    file = folder / "source.zip"
    with zipfile.ZipFile(file, "w", zipfile.ZIP_DEFLATED) as archive:
        if name == "sms_spam":
            archive.writestr("SMSSpamCollection", "\n".join(("spam" if i % 3 == 0 else "ham") + "\t" + ("Free prize" if i % 3 == 0 else "Meet later") + f" unique{i}" for i in range(120)))
        elif name == "uci_har":
            for split, subjects in [("train", range(1, 9)), ("test", range(9, 13))]:
                people = np.repeat(list(subjects), 12); n = len(people)
                prefix = "UCI HAR Dataset/" + split + "/"
                archive.writestr(prefix + "subject_" + split + ".txt", "\n".join(map(str, people)))
                archive.writestr(prefix + "y_" + split + ".txt", "\n".join(map(str, np.tile(np.arange(1, 7), n//6))))
                rng = np.random.default_rng(22)
                for kind in ["body_acc", "body_gyro", "total_acc"]:
                    for axis in "xyz":
                        output = io.StringIO(); np.savetxt(output, rng.normal(size=(n, 128)), fmt="%.5f")
                        archive.writestr(prefix + "Inertial Signals/" + kind + "_" + axis + "_" + split + ".txt", output.getvalue())
        else:
            rows = ["dteday,hr,cnt,casual,registered"]
            for i in range(720):
                if i == 150: continue  # Do not treat this gap as one adjacent hour.
                t = datetime(2011, 1, 1) + timedelta(hours=i)
                rows.append(f"{t:%Y-%m-%d},{t.hour},{10+i%24},999999,999999")
            archive.writestr("hour.csv", "\n".join(rows))
    write_json(folder / "receipt.json", {"files":{"source.zip":sha256(file)}})


class SequenceLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_compatibility_and_patch_controls(self):
        for bad in [config("lstm", "iris"), config("vit", "sms_spam"), config("mlp", "sms_spam"),
                    config("vit", "fashion_mnist", patch_size=8), config("embedding_mean", "digits"),
                    config("tfidf_nb", "digits", area="ml"), config("vit", "fashion_mnist", patch_size=2, heads=8, batch_size=256)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError): model_lab.validate(bad, require_ready=False)
        s = model_lab.validate(config("bilstm", "sms_spam"), require_ready=False)
        self.assertIn("embedding_dim", s["controls"])
        self.assertNotIn("embedding_dim", model_lab.validate(config("lstm"))["controls"])
        self.assertIn("lookback", model_lab.validate(config("ridge", "bike_sharing", "regression", "ml"), require_ready=False)["controls"])
        self.assertIs(type(model_lab.validate(config("vit", layers=2.0))["controls"]["layers"]), int)

    def test_every_encoder_forward_backward_and_mask_invariance(self):
        spec = {"kind":"text", "sequence_shape":[12, 1], "vocabulary_size":20}
        for name in sorted(SEQUENCE_MODELS):
            s = model_lab.validate(config(name, "sms_spam", width=16) if name != "embedding_mean" else config(name, "sms_spam"), require_ready=False)
            model = model_lab.network(name, s["controls"], 12, 2, None, spec)
            a = torch.tensor([[2, 3, 4, 0, 0, 0], [2, 4, 0, 0, 0, 0]])
            model.eval()
            output = model(a); padded = model(torch.nn.functional.pad(a, (0, 4)))
            self.assertEqual(tuple(output.shape), (2, 2), name)
            torch.testing.assert_close(output, padded, atol=1e-6, rtol=1e-5, msg=name)
            output.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), name)
            self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0, name)
        # Bidirectional readout is exactly the final forward/backward states.
        name = "bilstm"; p = model_lab.validate(config(name, "sms_spam", dropout=0), require_ready=False)["controls"]
        model = model_lab.network(name, p, 12, 2, None, spec).eval()
        a = torch.tensor([[2, 3, 4, 0], [4, 5, 0, 0]])
        packed = torch.nn.utils.rnn.pack_padded_sequence(model.embedding(a), torch.tensor([3, 2]), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = model.encoder(packed)
        torch.testing.assert_close(model(a), model.head(torch.cat([hidden[-2], hidden[-1]], -1)))

    def test_tcn_causality_and_vit_shapes(self):
        block = TemporalBlock(4, 3, 2, 0).eval()
        x = torch.randn(2, 4, 12); changed = x.clone(); changed[:, :, 7:] += 100
        valid = torch.ones(2, 12, dtype=torch.bool)
        torch.testing.assert_close(block(x, valid)[:, :, :7], block(changed, valid)[:, :, :7])
        for shape, dataset in [([1, 8, 8], "digits"), ([1, 28, 28], "fashion_mnist"), ([3, 32, 32], "cifar10")]:
            s = model_lab.validate(config("vit", dataset, width=16, layers=1), require_ready=False)
            model = model_lab.network("vit", s["controls"], int(np.prod(shape)), 10, shape)
            output = model(torch.randn(3, int(np.prod(shape)))); output.sum().backward()
            self.assertEqual(tuple(output.shape), (3, 10))
            self.assertLess(sum(p.numel() for p in model.parameters()), 3_000_000)

    def test_device_probe_restores_weights_mode_and_rng(self):
        p = model_lab.validate(config("gru", dropout=.5))["controls"]
        model = model_lab.network("gru", p, 64, 10, [1, 8, 8]).eval()
        weights = {k:v.clone() for k,v in model.state_dict().items()}
        state = torch.get_rng_state()
        probe_training_device(model, torch.zeros(2, 64), "cpu")
        self.assertFalse(model.training)
        torch.testing.assert_close(torch.get_rng_state(), state)
        for key, value in model.state_dict().items(): torch.testing.assert_close(value, weights[key])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_data_protocols_and_training_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as root:
            for name in sequence_data.SOURCES: fixture(root, name)
            a = model_data.load("sms_spam", 500, root, {"max_tokens":32})
            b = model_data.load("sms_spam", 500, root, {"max_tokens":64})
            self.assertEqual(a["info"]["fingerprint"], b["info"]["fingerprint"])
            self.assertEqual(a["split_ids"], b["split_ids"])
            words = {t for i in a["indices"]["train"] for t in sequence_data.word_tokens(a["texts"][i])}
            self.assertEqual(set(a["vocabulary"]), words)
            x, _, _ = sequence_data.neural_inputs(a); self.assertEqual(x.dtype, np.int64)
            self.assertTrue((x[:, 10:] == 0).all())
            har = model_data.load("uci_har", 500, root)
            groups = [set(har["subjects"][ids]) for ids in har["indices"].values()]
            for i in range(3):
                for j in range(i): self.assertFalse(groups[i] & groups[j])
            self.assertEqual(har["info"]["sequence_shape"], [128, 9])
            x, pre, _ = sequence_data.neural_inputs(har)
            np.testing.assert_allclose(x[har["indices"]["train"]].reshape(-1, 9).mean(0), 0, atol=1e-6)
            bike = model_data.load("bike_sharing", 500, root, {"lookback":12})
            longer = model_data.load("bike_sharing", 500, root, {"lookback":96})
            self.assertEqual(bike["split_ids"], longer["split_ids"])
            self.assertEqual(bike["info"]["fingerprint"], longer["info"]["fingerprint"])
            np.testing.assert_array_equal(bike["x"][:, -1], bike["persistence"])
            self.assertLess(bike["indices"]["train"].max(), bike["indices"]["dev"].min())
            self.assertLess(bike["indices"]["dev"].max(), bike["indices"]["test"].min())
            times = [datetime.fromisoformat(t) for t in bike["target_times"]]
            self.assertNotIn(datetime(2011, 1, 7, 7), times)  # Inside missing-hour history.
            self.assertLess(float(bike["x"].max()), 100)  # Components never used.

    def test_real_cpu_training_all_new_families_and_reload(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); data_root = root / "data"; data_root.mkdir()
            for name in sequence_data.SOURCES: fixture(data_root, name)
            for name in sorted(SEQUENCE_MODELS | {"vit"}):
                dataset = "sms_spam" if name == "embedding_mean" else "digits"
                controls = dict(epochs=1, train_limit=500, batch_size=128, device="cpu", dropout=0)
                if name != "embedding_mean": controls.update(width=16, layers=1)
                model_lab.fit(name, config(name, dataset, **controls), root / "runs", data_root)
                run = model_lab.detail(name, root / "runs")
                self.assertEqual(run["integrity"], "PASS", name)
                self.assertTrue(np.isfinite(run["history"][0]["dev_loss"]), name)
                saved = torch.load(root / "runs" / name / "weights.pt", weights_only=True)
                model = model_lab.network(name, saved["settings"]["controls"], saved["features"], saved["outputs"], saved["image_shape"], saved["input_spec"])
                model.load_state_dict(saved["state_dict"])
            for model, dataset, task, area in [("bilstm", "sms_spam", "classification", "dl"), ("tcn", "uci_har", "classification", "dl"), ("gru", "bike_sharing", "regression", "dl"), ("tfidf_logistic", "sms_spam", "classification", "ml"), ("tfidf_nb", "sms_spam", "classification", "ml")]:
                controls = dict(train_limit=500)
                if area == "dl": controls.update(epochs=1, width=16, layers=1, device="cpu")
                identifier = "task-" + model
                model_lab.fit(identifier, config(model, dataset, task, area, **controls), root / "runs", data_root)
                run = model_lab.detail(identifier, root / "runs")
                self.assertEqual(run["integrity"], "PASS")
                if dataset == "sms_spam": self.assertIn("text", run["result"]["examples"][0])
                if dataset == "bike_sharing":
                    self.assertIn("persistence_baseline", run["result"])
                    self.assertEqual(run["result"]["predictions"]["test"]["interval"]["resamples"], 0)
                if dataset == "uci_har": self.assertEqual(run["result"]["predictions"]["test"]["interval"]["resampling_unit"], "participant")
            self.assertEqual(len(model_lab.compare(["task-bilstm", "task-tfidf_logistic"], root / "runs")["rows"]), 2)
            model_lab.fit("short-history", config("ridge", "bike_sharing", "regression", "ml", lookback=12, train_limit=500), root / "runs", data_root)
            self.assertEqual(len(model_lab.compare(["task-gru", "short-history"], root / "runs")["rows"]), 2)

    def test_downloads_explicit_and_members_bounded(self):
        with tempfile.TemporaryDirectory() as root, patch("training_lab.model_data._download") as download:
            with self.assertRaises(ValueError): model_data.load("sms_spam", 500, root)
            download.assert_not_called()
            fixture(root, "sms_spam")
            (Path(root) / "sms_spam" / "source.zip").write_bytes(b"changed")
            with self.assertRaises(ValueError): model_data.load("sms_spam", 500, root)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z: z.writestr("../SMSSpamCollection", "bad")
        with zipfile.ZipFile(buffer) as z, self.assertRaises(KeyError): sequence_data.member(z, "SMSSpamCollection", 10)


if __name__ == "__main__": unittest.main()
