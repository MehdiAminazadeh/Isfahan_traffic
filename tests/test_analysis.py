import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isfahan_traffic.analysis import _dtw_distance


class AnalysisTests(unittest.TestCase):
    def test_dtw_identical_series_is_zero(self):
        values = np.array([0.0, 1.0, 2.0, 1.0])
        self.assertEqual(_dtw_distance(values, values), 0.0)

    def test_dtw_phase_shift_is_smaller_than_pointwise_distance(self):
        left = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        right = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        pointwise = float(np.abs(left - right).mean())
        self.assertLess(_dtw_distance(left, right, window=2), pointwise)


if __name__ == "__main__":
    unittest.main()
