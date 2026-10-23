"""Kiểm thử Golden Model điểm cố định và bộ xuất file .mem."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.config import load_config  # noqa: E402
from ocsvm_fpga.export.fpga_exporter import export_all  # noqa: E402
from ocsvm_fpga.models.training import calibrate_threshold, fit_detector  # noqa: E402
from ocsvm_fpga.quantization.fixed_point import to_fixed, to_hex  # noqa: E402
from ocsvm_fpga.quantization.golden_model import FixedPointOCSVM  # noqa: E402


class TestFixedPoint(unittest.TestCase):
    def test_round_and_saturate(self):
        q = to_fixed([0.5, -0.5, 100.0, -100.0], 16, 10)
        self.assertEqual(q.tolist(), [512, -512, 32767, -32768])

    def test_twos_complement_hex(self):
        self.assertEqual(to_hex([-1, 1, -32768], 16), ["FFFF", "0001", "8000"])
        self.assertEqual(to_hex([-2], 48), ["FFFFFFFFFFFE"])


class TestGoldenModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.cfg = load_config(overrides=["model.n_prototypes=400", "model.max_support_vectors=64",
                                         "threshold.temporal_smoothing=1"])
        cls.x = rng.normal(size=(1500, 36))
        cls.det, _ = fit_detector(cls.x, cls.cfg, rng)
        calibrate_threshold(cls.det, cls.cfg, None)
        cls.fx = FixedPointOCSVM(cls.det, cls.cfg.fpga)

    def test_sv_budget(self):
        self.assertLessEqual(self.det.model.n_sv, 64)

    def test_fixed_close_to_float(self):
        x = np.vstack([self.x[:200], np.random.default_rng(1).normal(size=(200, 36)) * 3])
        f_float = self.det.decision(x)
        xq, _ = self.fx.quantize_input(x)
        f_fix = self.fx.decision_float_equiv(self.fx.decision_q(xq))
        self.assertLess(np.abs(f_fix - f_float).max(), 0.02 * np.abs(f_float).max() + 1e-3)
        agree = np.mean((-f_float > self.det.threshold) == self.fx.anomaly_q(self.fx.decision_q(xq)))
        self.assertGreater(agree, 0.97)

    def test_export_files(self):
        with tempfile.TemporaryDirectory() as d:
            export_all(self.fx, self.cfg, Path(d), self.x[:30], None, np.random.default_rng(0))
            for f in ["sv.mem", "alpha.mem", "exp_lut.mem", "test_input.mem", "test_expected_score.mem"]:
                self.assertTrue((Path(d) / "mem" / f).exists(), f)
            n_lines = len((Path(d) / "mem" / "sv.mem").read_text().split())
            self.assertEqual(n_lines, self.fx.sv_q.size)
            self.assertIn("N_SV", (Path(d) / "ocsvm_params.vh").read_text())


if __name__ == "__main__":
    unittest.main()
