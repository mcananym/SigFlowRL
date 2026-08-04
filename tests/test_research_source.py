from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sigflow_v4_research.py"


class ResearchSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def test_source_is_valid_and_import_is_not_auto_run(self) -> None:
        self.assertIn('if __name__ == "__main__":', self.text)
        self.assertNotIn("RESULTS = main(CFG", self.text)

    def test_research_protocol_features_are_present(self) -> None:
        required = (
            "run_multi_horizon_rolling_research",
            "fit_interval_tail_calibration",
            "run_separate_ticker_validation_comparison",
            "build_full_regime_diagnostics",
            "apply_validation_selected_gate",
        )
        functions = {
            node.name for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(set(required) <= functions)

    def test_smoothing_default_is_false(self) -> None:
        self.assertIn("apply_regime_smoothing: bool = False", self.text)


if __name__ == "__main__":
    unittest.main()
