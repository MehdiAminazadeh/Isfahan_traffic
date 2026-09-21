import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isfahan_traffic.dashboard import _paired_change


class DashboardAnalysisTests(unittest.TestCase):
    def test_paired_change_keeps_only_common_intersections(self):
        frame = pd.DataFrame(
            [
                {"year_month": "2020-01", "intersection_id": 1, "mean_15min_volume": 100.0},
                {"year_month": "2020-01", "intersection_id": 2, "mean_15min_volume": 200.0},
                {"year_month": "2020-02", "intersection_id": 2, "mean_15min_volume": 150.0},
                {"year_month": "2020-02", "intersection_id": 3, "mean_15min_volume": 300.0},
            ]
        )
        paired = _paired_change(frame, "2020-01", "2020-02", "change")
        self.assertEqual(paired["intersection_id"].tolist(), [2])
        self.assertAlmostEqual(paired.iloc[0]["change"], -0.25)


if __name__ == "__main__":
    unittest.main()
