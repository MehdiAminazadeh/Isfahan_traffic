import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isfahan_traffic.parser import iter_scats_records


class ParserTests(unittest.TestCase):
    def test_multiline_record_and_sentinels(self):
        text = """Thursday 01 November 2018 00:15
Int 1081    1=14    2=51    3=2046    4=0
 5=2047  6=NA  7=9
Int 1093    1=24 2=52
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(text, encoding="utf-8")
            records = list(iter_scats_records(path))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].intersection_id, 1081)
        self.assertEqual(records[0].channels[1], 14)
        self.assertEqual(records[0].channels[4], 0)
        self.assertIsNone(records[0].channels[3])
        self.assertIsNone(records[0].channels[5])
        self.assertIsNone(records[0].channels[6])


if __name__ == "__main__":
    unittest.main()
