"""Kiểm thử đặc trưng lưới (đường cơ sở) và tính ổn định của cache."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.config import load_config  # noqa: E402
from ocsvm_fpga.features.grid_flow import cell_edges, grid_descriptor_dim, grid_flow_descriptor  # noqa: E402
from ocsvm_fpga.features.pipeline import feature_signature  # noqa: E402


class TestGridFlow(unittest.TestCase):
    def test_dims(self):
        u = np.zeros((120, 160), np.float32)
        self.assertEqual(grid_flow_descriptor(u, u, 6, 8, "basic").shape, (144,))
        self.assertEqual(grid_flow_descriptor(u, u, 6, 8, "rich").shape, (240,))
        self.assertEqual(grid_descriptor_dim(4, 4, "rich"), 80)

    def test_values_and_order(self):
        """Ô (r, c) phải nằm ở vị trí (r*cols + c)*n_stats, thống kê đúng giá trị."""
        u = np.zeros((120, 160), np.float32)
        v = np.zeros((120, 160), np.float32)
        u[20:40, 60:80] = 3.0                     # đúng ô hàng 1, cột 3 của lưới 6×8 (ô 20×20)
        v[20:40, 60:80] = 4.0
        d = grid_flow_descriptor(u, v, 6, 8, "rich").reshape(6, 8, 5)
        np.testing.assert_allclose(d[1, 3], [3, 4, 5, 5, 0], atol=1e-6)
        self.assertEqual(float(np.abs(np.delete(d.reshape(48, 5), 1 * 8 + 3, 0)).sum()), 0.0)

    def test_uneven_edges(self):
        e = cell_edges(100, 6)
        self.assertEqual((e[0], e[-1], len(e)), (0, 100, 7))
        self.assertTrue(np.all(np.diff(e) > 0))

    def test_signature_backward_compatible(self):
        """Thêm tùy chọn lưới/Farneback KHÔNG được làm đổi hash cache hiệp biến đã có."""
        cfg = load_config()
        self.assertEqual(feature_signature(cfg), "4b93299830")
        g = load_config(overrides=["features.descriptor=grid_flow"])
        self.assertNotEqual(feature_signature(g), feature_signature(cfg))
        # đổi tham số chỉ của hiệp biến không ảnh hưởng cache lưới
        g2 = load_config(overrides=["features.descriptor=grid_flow", "features.feature_set=F4"])
        self.assertEqual(feature_signature(g), feature_signature(g2))


if __name__ == "__main__":
    unittest.main()
