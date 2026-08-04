from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sigflow_v4_matplotlib_tests")

import sigflow_v4_research as research


ROOT = Path(__file__).resolve().parents[1]


def _price_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    trend = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Open": 100.0 + trend,
            "High": 102.0 + trend,
            "Low": 99.0 + trend,
            "Close": 101.0 + trend,
            "Volume": 1_000.0 + 100.0 * trend,
        },
        index=dates,
    )


def _valid_price_frame() -> pd.DataFrame:
    return _price_frame("2025-01-02", 4)


class CacheContractTests(unittest.TestCase):
    def test_verified_cache_round_trip_preserves_origin_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-01-12",
            )
            frame = _valid_price_frame()
            frame.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            research.save_cached_frame(frame, "AAPL", "yahoo_direct", cfg)

            loaded = research.load_cached_frame("AAPL", cfg)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.attrs["provider"], "yahoo_direct")
            self.assertTrue(loaded.attrs["cache_hit"])
            self.assertEqual(len(loaded), len(frame))

    def test_corrupt_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-01-12",
            )
            frame = _valid_price_frame()
            frame.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            research.save_cached_frame(frame, "AAPL", "yahoo_direct", cfg)
            metadata_path = research.cache_metadata_path_for("AAPL", cfg)
            metadata = json.loads(metadata_path.read_text())
            path = metadata_path.parent / metadata["content_file"]
            path.write_bytes(path.read_bytes() + b"corruption")
            self.assertIsNone(research.load_cached_frame("AAPL", cfg))

    def test_unverified_legacy_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-01-12",
            )
            _valid_price_frame().to_csv(research.cache_path_for("AAPL", cfg))
            self.assertIsNone(research.load_cached_frame("AAPL", cfg))

    def test_cache_with_missing_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-01-12",
            )
            frame = _valid_price_frame()
            frame.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            research.save_cached_frame(frame, "AAPL", "yahoo_direct", cfg)
            metadata_path = research.cache_metadata_path_for("AAPL", cfg)
            metadata = json.loads(metadata_path.read_text())
            metadata.pop("downloaded_at_utc")
            metadata_path.write_text(json.dumps(metadata))
            self.assertIsNone(research.load_cached_frame("AAPL", cfg))

    def test_truncated_tail_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-03-01",
            )
            frame = _valid_price_frame()
            frame.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            with self.assertRaisesRegex(ValueError, "truncated tail"):
                research.save_cached_frame(frame, "AAPL", "yahoo_direct", cfg)

    def test_next_day_snapshot_reuses_and_extends_prior_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior_cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-03-01",
            )
            prior = _price_frame("2025-01-02", 41)
            prior.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            research.save_cached_frame(prior, "AAPL", "yahoo_direct", prior_cfg)

            current_cfg = replace(prior_cfg, end_date="2025-03-04")
            complete = _price_frame("2025-01-02", 43)
            tail = complete.loc["2025-02-20":].copy()
            tail.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            requested_starts: list[str] = []

            def downloader(_symbol, start, _end, _timeout, _cfg):
                requested_starts.append(start)
                return tail

            with mock.patch.object(
                research, "download_yahoo_frame", side_effect=downloader
            ):
                combined, provider = research.download_symbol("AAPL", current_cfg)

            self.assertEqual(provider, "yahoo_direct")
            self.assertGreater(pd.Timestamp(requested_starts[0]), pd.Timestamp("2025-01-01"))
            self.assertGreater(len(combined), len(prior))
            self.assertTrue(research.cache_path_for("AAPL", current_cfg).exists())
            manifest = research.build_manifest(
                {"AAPL": combined}, {"AAPL": provider}
            )
            self.assertEqual(
                manifest.iloc[0]["incremental_base_sha256"],
                prior.attrs["csv_sha256"],
            )
            self.assertEqual(
                manifest.iloc[0]["incremental_fetch_start_date"],
                requested_starts[0],
            )

    def test_inconsistent_incremental_overlap_forces_full_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior_cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-03-01",
            )
            prior = _price_frame("2025-01-02", 41)
            source_url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            prior.attrs["source_url"] = source_url
            research.save_cached_frame(prior, "AAPL", "yahoo_direct", prior_cfg)

            current_cfg = replace(prior_cfg, end_date="2025-03-04")
            inconsistent_tail = _price_frame("2025-02-20", 8)
            inconsistent_tail.attrs["source_url"] = source_url
            full_refresh = _price_frame("2025-01-02", 43)
            full_refresh.attrs["source_url"] = source_url
            requested_starts: list[str] = []

            def downloader(_symbol, start, _end, _timeout, _cfg):
                requested_starts.append(start)
                return inconsistent_tail if len(requested_starts) == 1 else full_refresh

            with mock.patch.object(
                research, "download_yahoo_frame", side_effect=downloader
            ):
                combined, provider = research.download_symbol("AAPL", current_cfg)

            self.assertEqual(provider, "yahoo_direct")
            self.assertEqual(len(requested_starts), 2)
            self.assertGreater(
                pd.Timestamp(requested_starts[0]), pd.Timestamp(current_cfg.start_date)
            )
            self.assertEqual(requested_starts[1], current_cfg.start_date)
            pd.testing.assert_frame_equal(combined, full_refresh)

    def test_impossible_ohlc_row_is_removed(self) -> None:
        frame = _valid_price_frame()
        frame.iloc[1, frame.columns.get_loc("High")] = 50.0
        cleaned = research.clean_ohlcv(frame, "AAPL")
        self.assertEqual(len(cleaned), len(frame) - 1)

    def test_daily_market_alignment_ignores_provider_clock_time(self) -> None:
        dates = pd.date_range("2025-01-02", periods=75, freq="B")
        equity = _price_frame("2025-01-02", 75)
        equity.index = dates + pd.Timedelta(hours=14, minutes=30)
        vix = _price_frame("2025-01-02", 75)
        vix.index = dates + pd.Timedelta(hours=8)
        equity = research.clean_ohlcv(equity, "SPY")
        vix = research.clean_ohlcv(vix, "^VIX")
        self.assertTrue((equity.index.hour == 0).all())
        self.assertTrue((vix.index.hour == 0).all())

        target_dates = equity.index[1:]
        aligned = research.align_market_data_forward_only(
            target_dates,
            {"SPY": equity, "^VIX": vix},
            ("SPY", "^VIX"),
            forward_fill_limit=3,
        )
        cfg = research.config_for_profile("smoke")
        values = research.market_features_or_none(
            aligned, 61, ("SPY", "^VIX"), cfg
        )
        self.assertIsNotNone(values)

    def test_yahoo_application_error_falls_through_to_stooq(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"), cache_dir=directory
            )
            stooq_frame = _valid_price_frame()
            with (
                mock.patch.object(
                    research,
                    "download_yahoo_frame",
                    side_effect=research.DataDownloadError("Yahoo chart error"),
                ),
                mock.patch.object(
                    research, "download_stooq_frame", return_value=stooq_frame
                ),
                mock.patch.object(research, "save_cached_frame"),
            ):
                loaded, provider = research.download_symbol("AAPL", cfg)
            self.assertIs(loaded, stooq_frame)
            self.assertEqual(provider, "stooq_direct")

    def test_interrupted_pointer_update_preserves_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-01-12",
            )
            first = _valid_price_frame()
            first.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            research.save_cached_frame(first, "AAPL", "yahoo_direct", cfg)

            replacement = first.copy()
            replacement.loc[replacement.index[-1], "Close"] += 0.25
            replacement.attrs["source_url"] = first.attrs["source_url"]
            with (
                mock.patch.object(
                    research,
                    "write_json_atomic",
                    side_effect=RuntimeError("simulated pointer interruption"),
                ),
                self.assertRaisesRegex(RuntimeError, "pointer interruption"),
            ):
                research.save_cached_frame(
                    replacement, "AAPL", "yahoo_direct", cfg
                )

            loaded = research.load_cached_frame("AAPL", cfg)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertAlmostEqual(
                float(loaded.iloc[-1]["Close"]),
                float(first.iloc[-1]["Close"]),
            )

    def test_optional_hardlink_failure_does_not_invalidate_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(
                research.config_for_profile("smoke"),
                cache_dir=directory,
                start_date="2025-01-01",
                end_date="2025-01-12",
            )
            frame = _valid_price_frame()
            frame.attrs["source_url"] = (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            )
            with (
                mock.patch.object(
                    research.os,
                    "link",
                    side_effect=OSError("hard links unsupported"),
                ),
                self.assertWarns(RuntimeWarning),
            ):
                research.save_cached_frame(
                    frame, "AAPL", "yahoo_direct", cfg
                )
            self.assertIsNotNone(research.load_cached_frame("AAPL", cfg))


class ModelContractTests(unittest.TestCase):
    def test_vectorised_signature_matches_chen_identity_reference(self) -> None:
        rng = np.random.default_rng(20260804)
        path = np.cumsum(rng.normal(size=(19, 4)), axis=0)

        def reference(values: np.ndarray, depth: int) -> np.ndarray:
            increments = np.diff(values, axis=0)
            dimension = values.shape[1]
            level1 = np.zeros(dimension)
            level2 = np.zeros((dimension, dimension))
            level3 = np.zeros((dimension, dimension, dimension))
            for increment in increments:
                old1 = level1.copy()
                old2 = level2.copy()
                segment2 = np.outer(increment, increment) / 2.0
                segment3 = np.einsum(
                    "i,j,k->ijk", increment, increment, increment
                ) / 6.0
                level1 = old1 + increment
                if depth >= 2:
                    level2 = old2 + np.outer(old1, increment) + segment2
                if depth >= 3:
                    level3 = (
                        level3
                        + np.einsum("ij,k->ijk", old2, increment)
                        + np.einsum("i,jk->ijk", old1, segment2)
                        + segment3
                    )
            levels = [level1.ravel()]
            if depth >= 2:
                levels.append(level2.ravel())
            if depth >= 3:
                levels.append(level3.ravel())
            return np.concatenate(levels)

        for depth in (1, 2, 3):
            np.testing.assert_allclose(
                research.truncated_signature(path, depth),
                reference(path, depth),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_regime_weights_are_capped_and_use_training_rows_only(self) -> None:
        cfg = research.config_for_profile("smoke")
        labels = np.array([0] * 98 + [1, 2] + [2] * 50, dtype=int)
        metadata = pd.DataFrame(
            {
                "ticker_id": np.zeros(len(labels), dtype=int),
                "origin_date": pd.date_range("2020-01-01", periods=len(labels)),
            }
        )
        training = np.arange(100, dtype=int)
        weights, *_ = research.build_training_regime_statistics(
            metadata, labels, training, cfg
        )
        self.assertGreaterEqual(float(weights.min()), cfg.minimum_regime_class_weight)
        self.assertLessEqual(float(weights.max()), cfg.maximum_regime_class_weight)

        changed = labels.copy()
        changed[100:] = 0
        changed_weights, *_ = research.build_training_regime_statistics(
            metadata, changed, training, cfg
        )
        np.testing.assert_array_equal(weights, changed_weights)

    def test_ordinal_probabilities_are_valid(self) -> None:
        cfg = replace(
            research.config_for_profile("smoke"),
            hidden_size=8,
            regime_classifier_type="ordinal",
            use_ticker_embeddings=False,
        )
        model = research.RegimeFlowMixture(5, cfg)
        bundle = model.forward_bundle(
            torch.zeros((7, 5)), ticker_ids=torch.zeros(7, dtype=torch.long)
        )
        probabilities = bundle.probabilities.detach().numpy()
        self.assertTrue(np.all(probabilities >= 0.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert bundle.ordinal_logits is not None
        ordinal_logits = bundle.ordinal_logits.detach().numpy()
        self.assertTrue(np.all(ordinal_logits[:, 0] >= ordinal_logits[:, 1]))

    def test_auxiliary_gate_cannot_change_mixture_weights(self) -> None:
        cfg = replace(
            research.config_for_profile("smoke"),
            hidden_size=8,
            regime_gate_mode="auxiliary",
            use_ticker_embeddings=False,
        )
        model = research.RegimeFlowMixture(5, cfg)
        bundle = model.forward_bundle(
            torch.randn((9, 5)), ticker_ids=torch.zeros(9, dtype=torch.long)
        )
        expected = np.full((9, cfg.regimes), 1.0 / cfg.regimes)
        np.testing.assert_allclose(
            bundle.mixture_probabilities.detach().numpy(), expected, atol=1e-7
        )

    def test_single_distribution_ablation_does_not_duplicate_nll(self) -> None:
        spec = research.GROUP_ABLATIONS["single_distribution"]
        overrides = dict(spec.overrides)
        self.assertFalse(overrides["use_mixture_experts"])
        self.assertEqual(overrides["expert_alignment_weight"], 0.0)


class CliIsolationTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "sigflow_v4_research.py"), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_pipeline_mode_rejects_research_profile_before_network(self) -> None:
        completed = self._run(
            "--mode", "pipeline_test", "--profile", "gpu_long", "--no-save"
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires the smoke profile", completed.stderr + completed.stdout)

    def test_research_mode_rejects_synthetic_profile(self) -> None:
        completed = self._run(
            "--mode", "development", "--profile", "smoke", "--no-save"
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("research profile", completed.stderr + completed.stdout)

    def test_final_evaluation_cannot_disable_artifact_ledger(self) -> None:
        completed = self._run(
            "--mode", "final_evaluation", "--profile", "gpu_long",
            "--confirm-final-test", "--no-save",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must save", completed.stderr + completed.stdout)

    def test_prospective_protocol_hash_is_frozen_and_current(self) -> None:
        cfg = research.prospective_final_config(
            research.config_for_profile("gpu_long"), authorized=True
        )
        self.assertNotIn(
            "TO_BE_FROZEN", research.PREREGISTERED_FINAL_PROTOCOL_SHA256
        )
        self.assertEqual(
            research.final_protocol_hash(cfg),
            research.PREREGISTERED_FINAL_PROTOCOL_SHA256,
        )


class BootstrapContractTests(unittest.TestCase):
    def test_primary_accuracy_metric_tracks_configured_tolerance(self) -> None:
        cfg = replace(
            research.config_for_profile("smoke"),
            primary_success_tolerance=0.30,
        )
        frame = pd.DataFrame({
            "actual_vol": [0.20, 0.30],
            "crps": [0.01, 0.02],
            "sigflow_har_blend_crps": [0.02, 0.03],
        })
        for column in set(research.MODEL_COLUMNS.values()) | set(
            research.QLIKE_MODEL_COLUMNS.values()
        ):
            frame[column] = [0.21, 0.29]
        differences = research.paired_difference_arrays(
            frame, "Log-HAR transferable", cfg
        )
        self.assertIn("within_30pct_accuracy", differences)
        self.assertNotIn("within_20pct_accuracy", differences)

    def test_resample_hash_binds_draws_to_the_calendar(self) -> None:
        draws = np.array([[0, 1], [1, 0]], dtype=int)
        first = research.resample_plan_hash(
            {"scheme": "date_block", "dates": ["2025-01-02", "2025-01-03"]},
            draws,
        )
        second = research.resample_plan_hash(
            {"scheme": "date_block", "dates": ["2026-01-02", "2026-01-05"]},
            draws,
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
