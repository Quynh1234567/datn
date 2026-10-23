"""Kiểm thử đơn vị cho Horn–Schunck và ma trận hiệp biến.   python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocsvm_fpga.config import load_config  # noqa: E402
from ocsvm_fpga.features.covariance import (FEATURE_SETS, covariance_matrix, descriptor_dim,  # noqa: E402
                                            pixel_feature_matrix, spd_logm, unvech_scaled,
                                            vech_scaled, covariance_descriptor)
from ocsvm_fpga.features.optical_flow import horn_schunck  # noqa: E402


def textured(h=140, w=180, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 3)


class TestHornSchunck(unittest.TestCase):
    def test_translation_x(self):
        base = textured()
        i1, i2 = base[10:130, 10:170], base[10:130, 9:169]      # dịch phải 1 px
        u, v = horn_schunck(i1, i2, alpha=2.0, num_iter=100)
        self.assertAlmostEqual(float(u[10:-10, 10:-10].mean()), 1.0, delta=0.15)
        self.assertAlmostEqual(float(v[10:-10, 10:-10].mean()), 0.0, delta=0.05)

    def test_translation_y(self):
        base = textured()
        i1, i2 = base[10:130, 10:170], base[9:129, 10:170]      # dịch xuống 1 px
        u, v = horn_schunck(i1, i2, alpha=2.0, num_iter=100)
        self.assertAlmostEqual(float(v[10:-10, 10:-10].mean()), 1.0, delta=0.15)

    def test_static_scene_zero_flow(self):
        a = textured()
        u, v = horn_schunck(a, a, 2.0, 32)
        self.assertLess(float(np.abs(u).max() + np.abs(v).max()), 1e-6)


class TestCovariance(unittest.TestCase):
    def test_matches_numpy_cov(self):
        z = np.random.default_rng(1).normal(size=(500, 8))
        np.testing.assert_allclose(covariance_matrix(z, eps=0.0), np.cov(z, rowvar=False), atol=1e-12)

    def test_feature_dims(self):
        u = np.random.default_rng(2).normal(size=(60, 80))
        v = np.random.default_rng(3).normal(size=(60, 80))
        for name, cols in FEATURE_SETS.items():
            z = pixel_feature_matrix(u, v, name, stride=2)
            self.assertEqual(z.shape[1], len(cols))
            self.assertEqual(descriptor_dim(name), len(cols) * (len(cols) + 1) // 2)

    def test_log_euclidean_isometry(self):
        """‖vech(logC1) − vech(logC2)‖ = ‖logC1 − logC2‖_F  → RBF ≡ kernel Log-Euclidean."""
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=(300, 8)), rng.normal(size=(300, 8)) * 1.7
        l1, l2 = spd_logm(covariance_matrix(a)), spd_logm(covariance_matrix(b))
        d_vec = np.linalg.norm(vech_scaled(l1) - vech_scaled(l2))
        self.assertAlmostEqual(d_vec, np.linalg.norm(l1 - l2, "fro"), places=10)
        np.testing.assert_allclose(unvech_scaled(vech_scaled(l1), 8), l1, atol=1e-12)

    def test_descriptor_pipeline(self):
        cfg = load_config()
        u = np.random.default_rng(5).normal(size=(120, 160)).astype(np.float32)
        v = np.random.default_rng(6).normal(size=(120, 160)).astype(np.float32)
        vec, c = covariance_descriptor(u, v, cfg.features)
        self.assertEqual(vec.shape, (descriptor_dim(cfg.features.feature_set),))
        self.assertTrue(np.all(np.linalg.eigvalsh(c.astype(np.float64)) > 0))


if __name__ == "__main__":
    unittest.main()
