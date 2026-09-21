import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isfahan_traffic.context import build_iran_calendar, load_manual_events


class ContextTests(unittest.TestCase):
    def test_nowruz_conversion_and_friday_flag(self):
        calendar = build_iran_calendar(pd.date_range("2020-03-19", "2020-03-21"))
        row = calendar.loc[calendar["date"] == pd.Timestamp("2020-03-20")].iloc[0]
        self.assertEqual(row["jalali_date"], "1399-01-01")
        self.assertEqual(row["is_nowruz_window"], 1)
        self.assertEqual(row["is_official_holiday"], 1)
        self.assertEqual(row["is_friday_weekend"], 1)

    def test_manual_event_interval_expands_without_duplicating_calendar(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.csv"
            pd.DataFrame(
                [
                    {
                        "date_start": "2020-01-02",
                        "date_end": "2020-01-03",
                        "event_type": "test",
                        "event_name": "two day thing",
                        "event_scope": "test",
                        "location": "test",
                        "source_url": "https://example.com",
                        "source_confidence": "high",
                        "notes": "only a unit test",
                    }
                ]
            ).to_csv(event_path, index=False)
            daily, _ = load_manual_events(
                pd.date_range("2020-01-01", "2020-01-04"), path=event_path
            )
        self.assertEqual(len(daily), 4)
        self.assertEqual(daily["manual_event_count"].sum(), 2)
        self.assertEqual(daily["date"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
