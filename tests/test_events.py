"""Kiểm thử cảnh báo có trễ và chỉ số theo sự kiện."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ocsvm_fpga.evaluation.events import event_metrics, hysteresis, runs  # noqa: E402


class TestEvents(unittest.TestCase):
    def test_hysteresis_bridges_flicker(self):
        s = np.array([0, 0, 1, 0.5, 1, 0.5, 1, 0, 0, 0, 0, 0])   # on = 0.8, off = 0.2, hold = 2
        a = hysteresis(s, 0.8, 0.2, 2)
        np.testing.assert_array_equal(a, [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

    def test_runs(self):
        self.assertEqual(runs(np.array([0, 1, 1, 0, 1])), [(1, 2), (4, 4)])

    def test_event_metrics(self):
        y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        al = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])          # phát hiện trễ 1 cặp + 1 báo giả
        vid = np.zeros(10, np.int32)
        fidx = np.arange(10) * 2
        r = event_metrics(y, al, vid, fidx, fps=25.0)
        self.assertEqual((r["events"], r["detected"], r["false_alarm_events"]), (1, 1, 1))
        self.assertAlmostEqual(r["delay_median_s"], 2 / 25.0)


if __name__ == "__main__":
    unittest.main()
