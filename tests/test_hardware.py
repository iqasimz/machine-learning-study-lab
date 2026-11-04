"""Device policies are mocked; the CPU smoke check uses real operations."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import hardware
from training_lab import model_lab, adaptation, experiments


class HardwareTests(unittest.TestCase):
    def test_real_cpu_backward_probe(self):
        self.assertIsNone(hardware.probe("cpu"))

    def test_auto_prioritizes_cuda_then_mps_then_cpu(self):
        for available, expected in [({"cpu": True, "cuda": True, "mps": True}, "cuda"),
                                    ({"cpu": True, "cuda": False, "mps": True}, "mps"),
                                    ({"cpu": True, "cuda": False, "mps": False}, "cpu")]:
            with self.subTest(expected=expected), patch.object(hardware, "available_devices", return_value=available), patch.object(hardware, "probe", return_value=None):
                self.assertEqual(hardware.choose_device("auto"), expected)

    def test_broken_gpu_falls_back_only_in_auto(self):
        with patch.object(hardware, "available_devices", return_value={"cpu": True, "cuda": True, "mps": False}), patch.object(hardware, "probe", side_effect=lambda d: "driver failure" if d=="cuda" else None):
            self.assertEqual(hardware.choose_device(), "cpu")
            with self.assertRaisesRegex(ValueError, "failed"):
                hardware.choose_device("cuda")

    def test_explicit_unavailable_devices_fail_instead_of_silent_cpu(self):
        with patch.object(hardware, "available_devices", return_value={"cpu": True, "cuda": False, "mps": False}):
            for device in ["cuda", "mps", "arbitrary"]:
                with self.subTest(device=device), self.assertRaises(ValueError): hardware.choose_device(device)

    def test_all_neural_registries_offer_auto_and_cuda(self):
        self.assertEqual(model_lab.CONTROLS["device"]["default"], "auto")
        self.assertEqual(experiments.DEFAULTS["device"], "auto")
        self.assertEqual(adaptation.DEFAULTS["device"], "auto")
        for choices in [model_lab.CONTROLS["device"]["choices"], experiments.PARAMETERS["device"]["choices"], adaptation.CHOICES["device"]]:
            self.assertEqual(set(choices), {"auto", "cpu", "cuda", "mps"})


if __name__ == "__main__": unittest.main()
