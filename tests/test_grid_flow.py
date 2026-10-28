"""Kiểm thử đặc trưng lưới (đường cơ sở) và tính ổn định của cache."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.config import load_config  # noqa: E402
from ocsvm_fpga.features.grid_flow import (block_stats, cell_edges, from_blocks, grid_descriptor_dim,  # noqa: E402
                                           grid_flow_descriptor)
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

    def test_max_is_rich_without_std(self):
        rng = np.random.default_rng(0)
        u, v = rng.normal(size=(120, 160)).astype(np.float32), rng.normal(size=(120, 160)).astype(np.float32)
        rich = grid_flow_descriptor(u, v, 4, 4, "rich").reshape(16, 5)
        mx = grid_flow_descriptor(u, v, 4, 4, "max").reshape(16, 4)
        np.testing.assert_array_equal(mx, rich[:, :4])

    def test_blocks_equal_direct(self):
        """Lưới ghép từ cache khối 10×10 phải trùng với tính trực tiếp (đều / phối cảnh, có diff)."""
        rng = np.random.default_rng(1)
        u, v = (rng.normal(size=(120, 160)).astype(np.float32) for _ in range(2))
        d = np.abs(rng.normal(size=(120, 160))).astype(np.float32)
        B = block_stats(u, v, d)[None]
        for ye, xe in ((cell_edges(120, 4), cell_edges(160, 4)), ([0, 20, 40, 70, 120], [0, 50, 80, 110, 160])):
            for stats in ("basic", "max", "rich"):
                for diff in (False, True):
                    direct = grid_flow_descriptor(u, v, stats=stats, ye=ye, xe=xe, diff_img=d if diff else None)
                    np.testing.assert_allclose(from_blocks(B, ye, xe, stats, diff)[0], direct, rtol=1e-4, atol=1e-5)

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
