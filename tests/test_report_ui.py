import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isfahan_traffic.report_ui import build_explorer_payload


class TrafficExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads((ROOT / "outputs/dashboard/artifact.json").read_text(encoding="utf-8"))
        cls.summary = json.loads((ROOT / "outputs/tables/analysis_summary.json").read_text(encoding="utf-8"))
        cls.insights = json.loads((ROOT / "outputs/tables/dashboard_insights.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((ROOT / "outputs/tables/dashboard_validation.json").read_text(encoding="utf-8"))
        cls.settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
        cls.payload = build_explorer_payload(cls.artifact, cls.summary, cls.insights, cls.validation, cls.settings)

    def test_charts_remain_small_and_reachable_from_their_section(self):
        charts = self.payload["charts"]
        self.assertEqual(len(charts), 8)
        links = [chart_id for section in self.payload["sections"] for chart_id in section["chartIds"]]
        self.assertEqual(sorted(links), sorted(charts))
        for section in self.payload["sections"]:
            self.assertTrue(all(charts[key]["section"] == section["id"] for key in section["chartIds"]))

    def test_monthly_line_does_not_bridge_missing_or_partial_months(self):
        trace = self.payload["charts"]["monthly"]["figure"]["data"][0]
        values = {str(x)[:7]: y for x, y in zip(trace["x"], trace["y"])}
        self.assertEqual(len(values), 28)
        self.assertIsNone(values["2020-11"])
        self.assertIsNone(values["2020-12"])
        self.assertFalse(trace.get("connectgaps", False))

    def test_approach_lines_keep_partial_december_separate(self):
        traces = self.payload["charts"]["approaches"]["figure"]["data"]
        lines = [trace for trace in traces if "lines" in trace.get("mode", "")]
        self.assertEqual(len(lines), 4)
        for trace in lines:
            values = {str(x)[:7]: y for x, y in zip(trace["x"], trace["y"])}
            self.assertIsNone(values["2020-11"])
            self.assertIsNone(values["2020-12"])
            self.assertFalse(trace.get("connectgaps", False))
        partial = [trace for trace in traces if trace.get("mode") == "markers"]
        self.assertEqual(len(partial), 4)
        self.assertTrue(all(any(str(x).startswith("2020-12") for x in trace["x"]) for trace in partial))

    def test_site_chart_uses_only_the_common_comparison_set(self):
        expected = self.artifact["snapshot"]["datasets"]["shock_recovery"]
        traces = self.payload["charts"]["sites"]["figure"]["data"]
        actual = [(x, y) for trace in traces for x, y in zip(trace["x"], trace["y"])]
        self.assertEqual(len(actual), 39)
        self.assertEqual(sorted(actual), sorted((row["shock_change"], row["recovery_change"]) for row in expected))

    def test_build_does_not_change_source_analysis(self):
        artifact = copy.deepcopy(self.artifact)
        build_explorer_payload(artifact, self.summary, self.insights, self.validation, self.settings)
        self.assertEqual(artifact, self.artifact)

    def test_developer_view_uses_real_dataset_sizes(self):
        actual = {row["id"]: row["count"] for row in self.payload["developer"]["datasets"]}
        expected = {key: len(rows) for key, rows in self.artifact["snapshot"]["datasets"].items()}
        self.assertEqual(actual, expected)
        self.assertEqual(self.payload["developer"]["validation"], self.validation)


if __name__ == "__main__":
    unittest.main()
