from __future__ import annotations

import unittest
from datetime import date, timedelta

from sigflow_v4.protocol import (
    DEFAULT_RESEARCH_SEEDS,
    make_rolling_origin_windows,
)


class RollingOriginTests(unittest.TestCase):
    def test_required_five_research_seeds(self) -> None:
        self.assertEqual(DEFAULT_RESEARCH_SEEDS, (42, 123, 456, 789, 2026))

    def test_windows_expand_and_tests_do_not_overlap(self) -> None:
        start = date(2020, 1, 1)
        dates = [start + timedelta(days=index) for index in range(80)]
        windows = make_rolling_origin_windows(
            dates,
            minimum_train_dates=20,
            validation_dates=5,
            calibration_dates=5,
            test_dates=8,
            step_dates=8,
            horizon_dates=3,
        )
        self.assertGreaterEqual(len(windows), 2)
        for previous, current in zip(windows[:-1], windows[1:]):
            self.assertLess(previous.test_end, current.test_start)
            self.assertLess(previous.train_end, current.train_end)
        for window in windows:
            self.assertLess(window.train_end, window.validation_end)
            self.assertLess(window.validation_end, window.calibration_end)
            self.assertLess(window.calibration_end, window.test_start)

    def test_too_few_dates_fails(self) -> None:
        with self.assertRaises(ValueError):
            make_rolling_origin_windows(
                [date(2020, 1, day) for day in range(1, 10)],
                minimum_train_dates=5,
                validation_dates=2,
                calibration_dates=2,
                test_dates=2,
                horizon_dates=2,
            )


if __name__ == "__main__":
    unittest.main()
