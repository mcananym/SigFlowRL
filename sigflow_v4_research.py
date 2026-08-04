#!/usr/bin/env python3
"""Canonical Python runner exported from SigFlow-Sim v4.

Edit this module for research protocol changes; the notebook remains the interactive report.
"""

from __future__ import annotations


# %% [markdown cell 0]
# # SigFlow-Sim v4 — Leakage-Controlled Cross-Ticker Evaluation
# 
# This notebook preserves the existing SigFlow conditional-flow model and its objective, while making the experimental protocol deeper, configuration-driven, and suitable for honest generalisation tests.
# 
# ## Experimental guarantees
# 
# - **Separated execution paths:** development stops at validation; calibration and the prospective test are reached only by explicit evaluation modes.
# - **Two final generalisation cohorts:** trained tickers on unseen days, and completely held-out tickers on the same unseen days.
# - **Purging at every chronological boundary:** no target window crosses from an earlier fitting section into a later one.
# - **Held-out tickers never influence fitting:** preprocessing, early stopping, calibration, regime thresholds, transition priors, and the transferable HAR fallback use training tickers only.
# - **No backward filling:** market context is forward-filled for at most three sessions; incomplete samples are discarded.
# - **Canonical feature matrix:** multiscale shape, amplitude, OHLC, joint-market, and lead-lag signatures are built once, then masks select matched ablations.
# - **Sector-balanced transfer test:** 22 training and 22 held-out tickers cover the same 11 sectors.
# - **Prespecified inference:** unseen-ticker QLIKE versus HAR is primary; tolerance accuracy, other baselines, sectors, and days are secondary.
# - **Matched model comparisons:** gate, regime loss, ordinal, mixture, ticker, HAR, and signature variants share the same data folds, seeds, budgets, and Monte Carlo streams.
# - **Result-only reporting:** the notebook displays and saves test forecasts, errors, success flags, intervals, regime outputs, and summaries—not raw OHLCV or feature rows.
# 
# ## Success definition
# 
# A forecast is successful when its median volatility forecast is within 20% of realised volatility. Each row is issued on `origin_date` and predicts realised volatility over the following `horizon` sessions, shown by `target_start_date` and `target_end_date`.
# 
# ## First run
# 
# The CLI defaults to the synthetic `smoke` pipeline test. Real development and rolling-origin research require an explicit real-data profile; `gpu_long` is intended for a normal CUDA GPU and can run for many hours depending on cache coverage and hardware.

# %% [cell 1]
# Environment diagnostic — no package installation is attempted.

import importlib.util
import os
import sys

# Keep Matplotlib's generated font/config cache beside the project.  This is
# writable in notebook, CLI, and headless runs even when ~/.config is not.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib-cache"),
)

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "torch": "torch",
}

missing = [
    package
    for package, import_name in REQUIRED_IMPORTS.items()
    if importlib.util.find_spec(import_name) is None
]

VERBOSE_IMPORT = os.environ.get("SIGFLOW_VERBOSE_IMPORT", "0") == "1"
if VERBOSE_IMPORT:
    print("Python executable:", sys.executable)
if missing:
    raise ModuleNotFoundError(
        "Missing core scientific packages: " + ", ".join(missing)
    )

if VERBOSE_IMPORT:
    print("Core packages are available.")
    print("No yfinance, iisignature, or nflows installation is required.")

# %% [cell 2]

import copy
import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import resource
import time
import urllib.parse
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from IPython.display import HTML, display
except ImportError:  # The CLI does not need IPython.
    HTML = lambda value: value
    display = print
from torch.utils.data import DataLoader, TensorDataset

from sigflow_v4.data import (
    DataDownloadError,
    NetworkUnavailableError,
    RateLimitError,
    RetryPolicy,
    cache_metadata,
    fsync_directory,
    request_bytes_with_retry,
    sha256_bytes,
    write_json_atomic,
)
from sigflow_v4.protocol import (
    DEFAULT_RESEARCH_SEEDS,
    collect_reproducibility_manifest,
    make_rolling_origin_windows,
    serialise_windows,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
REGIME_NAMES = np.array(["Low", "Medium", "High"])

if VERBOSE_IMPORT:
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    print("Data route: verified cache → retrying Yahoo → retrying Stooq")

# %% [markdown cell 3]
# ## Configuration
# 
# The next cell is the notebook control panel. The most useful controls are:
# 
# - `PROFILE_NAME`: `smoke`, `gpu_balanced`, `gpu_long`, or `deep`. The default `gpu_long` profile uses five seeds, at least 100 epochs per seed, and a 320-epoch cap.
# - `training_tickers` and `unseen_test_tickers`: disjoint, sector-balanced 22-name universes.
# - signature-family/window switches: shape and amplitude at 10/20/60 sessions plus OHLC, joint-market, and lead-lag paths.
# - significance controls: primary cohort/baseline/metric, block sensitivities, practical effect size, and seed-consistency threshold.
# - feature-family switches plus `disabled_feature_groups` and `disabled_feature_names`; every canonical feature can be removed by its exact printed name.
# - `RUN_GROUP_ABLATIONS` and `RUN_INDIVIDUAL_FEATURE_ABLATIONS`. Ablations are averaged across expanding inner folds wholly inside training, so outer validation, calibration, and final test retain their roles.
# - `DISPLAY_ALL_TEST_RESULTS`: renders every result row in a scrollable table; the same rows are saved to `detailed_test_results.csv`.
# 
# Importing this module never launches an experiment. Use the CLI or thin runner notebook; start with `pipeline_test` before committing GPU time to a real-data profile.

# %% [cell 4]
# ---------------------------------------------------------------------
# Primary controls: edit these, then use Restart and Run All.
# ---------------------------------------------------------------------
PROFILE_NAME = os.environ.get("SIGFLOW_PROFILE", "gpu_long").strip().lower()
RUN_GROUP_ABLATIONS = False
RUN_INDIVIDUAL_FEATURE_ABLATIONS = False
DISPLAY_ALL_TEST_RESULTS = True

SELECTED_GROUP_ABLATIONS = (
    "no_signature",
    "no_shape_signatures",
    "no_amplitude_signatures",
    "no_ohlc_signatures",
    "no_joint_market_signatures",
    "no_lead_lag_signatures",
    "no_signature_w10",
    "no_signature_w20",
    "no_signature_w60",
    "no_return_statistics",
    "no_ohlcv",
    "no_market",
    "no_regime_classification",
    "no_regimes",
    "regime_auxiliary_only",
    "soft_regime_gate",
    "plain_regime_cross_entropy",
    "weighted_regime_cross_entropy",
    "focal_regime_loss",
    "ordinal_regimes",
    "single_distribution",
    "with_ticker_embeddings",
    "without_ticker_embeddings",
    "with_ticker_specific_heads",
    "pooled_feature_scaling",
    "without_har_feature",
    "har_residual",
    "simple_neural",
    "single_seed",
    "regime_weight_030",
    "absolute_regime_labels",
    "with_ticker_identity",
    "no_auxiliary_losses",
)

# Empty means every active feature when RUN_INDIVIDUAL_FEATURE_ABLATIONS=True.
# Otherwise enter exact feature names printed by the active-feature manifest.
INDIVIDUAL_FEATURES_TO_ABLATE: tuple[str, ...] = ()


TRAINING_TICKERS_BY_SECTOR = {
    "Information Technology": ("AAPL", "MSFT"),
    "Communication Services": ("GOOGL", "DIS"),
    "Consumer Discretionary": ("AMZN", "HD"),
    "Consumer Staples": ("PG", "KO"),
    "Financials": ("JPM", "BAC"),
    "Health Care": ("JNJ", "PFE"),
    "Industrials": ("CAT", "HON"),
    "Energy": ("XOM", "CVX"),
    "Materials": ("LIN", "APD"),
    "Utilities": ("NEE", "DUK"),
    "Real Estate": ("PLD", "AMT"),
}
UNSEEN_TEST_TICKERS_BY_SECTOR = {
    "Information Technology": ("NVDA", "AMD"),
    "Communication Services": ("META", "NFLX"),
    "Consumer Discretionary": ("TSLA", "NKE"),
    "Consumer Staples": ("PEP", "COST"),
    "Financials": ("GS", "MS"),
    "Health Care": ("UNH", "MRK"),
    "Industrials": ("GE", "UPS"),
    "Energy": ("COP", "SLB"),
    "Materials": ("FCX", "NEM"),
    "Utilities": ("AEP", "SO"),
    "Real Estate": ("O", "SPG"),
}


def flatten_sector_universe(
    universe: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(
        ticker
        for tickers in universe.values()
        for ticker in tickers
    )


TRAINING_TICKERS = flatten_sector_universe(TRAINING_TICKERS_BY_SECTOR)
UNSEEN_TEST_TICKERS = flatten_sector_universe(
    UNSEEN_TEST_TICKERS_BY_SECTOR
)
TICKER_SECTORS = {
    ticker: sector
    for universe in (
        TRAINING_TICKERS_BY_SECTOR,
        UNSEEN_TEST_TICKERS_BY_SECTOR,
    )
    for sector, tickers in universe.items()
    for ticker in tickers
}


PROFILE_SETTINGS = {
    "smoke": {
        "quick_mode": True,
        "training_tickers": ("AAPL", "MSFT", "JPM"),
        "unseen_test_tickers": ("TSLA", "AMD"),
        "use_real_market_data": False,
        "allow_synthetic_fallback": False,
        "start_date": "2020-01-01",
        "epochs": 2,
        "batch_size": 256,
        "hidden_size": 48,
        "patience": 2,
        "minimum_epochs": 1,
        "scheduler_patience": 1,
        "ensemble_seeds": (42,),
        "prediction_samples": 96,
        "bootstrap_repetitions": 50,
        "run_block_bootstrap": False,
        "minimum_test_origins_per_ticker": 60,
    },
    "gpu_balanced": {
        "quick_mode": False,
        "training_tickers": TRAINING_TICKERS,
        "unseen_test_tickers": UNSEEN_TEST_TICKERS,
        "use_real_market_data": True,
        "allow_synthetic_fallback": False,
        "start_date": "2012-01-01",
        "epochs": 150,
        "batch_size": 512,
        "hidden_size": 96,
        "patience": 18,
        "minimum_epochs": 40,
        "scheduler_patience": 8,
        "ensemble_seeds": DEFAULT_RESEARCH_SEEDS,
        "prediction_samples": 256,
        "bootstrap_repetitions": 750,
        "run_block_bootstrap": True,
        "minimum_test_origins_per_ticker": 400,
    },
    "gpu_long": {
        "quick_mode": False,
        "training_tickers": TRAINING_TICKERS,
        "unseen_test_tickers": UNSEEN_TEST_TICKERS,
        "use_real_market_data": True,
        "allow_synthetic_fallback": False,
        "start_date": "2010-01-01",
        "epochs": 320,
        "batch_size": 256,
        "hidden_size": 128,
        "patience": 40,
        "minimum_epochs": 100,
        "scheduler_patience": 8,
        "ensemble_seeds": DEFAULT_RESEARCH_SEEDS,
        "prediction_samples": 384,
        "bootstrap_repetitions": 2000,
        "run_block_bootstrap": True,
        "minimum_test_origins_per_ticker": 500,
    },
    "deep": {
        "quick_mode": False,
        "training_tickers": TRAINING_TICKERS,
        "unseen_test_tickers": UNSEEN_TEST_TICKERS,
        "use_real_market_data": True,
        "allow_synthetic_fallback": False,
        "start_date": "2010-01-01",
        "epochs": 480,
        "batch_size": 256,
        "hidden_size": 160,
        "patience": 55,
        "minimum_epochs": 150,
        "scheduler_patience": 10,
        "ensemble_seeds": DEFAULT_RESEARCH_SEEDS + (31415, 27182),
        "prediction_samples": 512,
        "bootstrap_repetitions": 3000,
        "run_block_bootstrap": True,
        "minimum_test_origins_per_ticker": 500,
    },
}

if PROFILE_NAME not in PROFILE_SETTINGS:
    raise ValueError(
        f"Unknown PROFILE_NAME={PROFILE_NAME!r}; choose from "
        f"{tuple(PROFILE_SETTINGS)}."
    )

PROFILE = PROFILE_SETTINGS[PROFILE_NAME]


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    overrides: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class Config:
    profile_name: str = PROFILE_NAME
    quick_mode: bool = bool(PROFILE["quick_mode"])
    experiment_mode: str = (
        "pipeline_test" if PROFILE["quick_mode"] else "research"
    )
    run_name: str = f"{PROFILE_NAME}_base"

    # Target universes. Held-out tickers are never used for fitting.
    training_tickers: tuple[str, ...] = PROFILE["training_tickers"]
    unseen_test_tickers: tuple[str, ...] = PROFILE["unseen_test_tickers"]

    # Data and provenance
    use_real_market_data: bool = bool(PROFILE["use_real_market_data"])
    allow_synthetic_fallback: bool = bool(PROFILE["allow_synthetic_fallback"])
    refresh_data: bool = False
    download_attempts: int = 5
    download_initial_backoff_seconds: float = 1.0
    download_maximum_backoff_seconds: float = 60.0
    download_jitter_fraction: float = 0.15
    provider_request_spacing_seconds: float = 0.25
    allow_stale_cache_after_refresh_failure: bool = True
    network_preflight_timeout_seconds: int = 3
    network_request_timeout_seconds: int = 8
    run_network_preflight: bool = True
    market_symbols: tuple[str, ...] = ("SPY", "QQQ", "^VIX")
    include_market_context: bool = True
    start_date: str = str(PROFILE["start_date"])
    end_date: str = "2026-07-25"
    cache_dir: str = "sigflow_v4_cache"
    output_root: str = "sigflow_v4_outputs"

    # Forecast construction and configurable signature lifts.
    window: int = 60
    horizon: int = 10
    forecast_horizons: tuple[int, ...] = (1, 5, 10, 20)
    annualisation: float = 252.0
    ewma_lambda: float = 0.94
    signature_windows: tuple[int, ...] = (10, 20, 60)
    secondary_signature_windows: tuple[int, ...] = (20,)
    signature_depth: int = 3
    short_signature_depth: int = 2
    amplitude_signature_depth: int = 2
    ohlc_signature_depth: int = 2
    joint_market_signature_depth: int = 2
    lead_lag_signature_depth: int = 2
    signature_return_scale: float = 0.01
    signature_vix_change_scale: float = 0.05
    joint_signature_symbols: tuple[str, str] = ("SPY", "^VIX")
    market_forward_fill_limit: int = 3

    # Model-input feature switches. The canonical feature matrix is still
    # built once, so ablations share exactly the same rows and values.
    include_signature_features: bool = True
    include_shape_signatures: bool = True
    include_amplitude_signatures: bool = True
    include_ohlc_signatures: bool = True
    include_joint_market_signatures: bool = True
    include_lead_lag_signatures: bool = True
    include_return_statistics: bool = True
    include_ohlcv_features: bool = True
    include_market_features: bool = True
    include_ticker_identity: bool = False
    disabled_feature_groups: tuple[str, ...] = ()
    disabled_feature_names: tuple[str, ...] = ()

    # Strict chronological sections, calculated from training tickers only.
    train_fraction: float = 0.65
    validation_fraction: float = 0.10
    calibration_fraction: float = 0.10
    explicit_train_end_date: str | None = None
    explicit_validation_end_date: str | None = None
    explicit_calibration_end_date: str | None = None
    explicit_test_start_date: str | None = None
    explicit_test_origin_end_date: str | None = None
    explicit_test_origin_dates: tuple[str, ...] = ()
    explicit_test_end_date: str | None = None
    test_window_id: str = "fixed_outer"
    inner_validation_folds: int = 3
    inner_validation_fraction_of_training: float = 0.30

    # Training-only preprocessing and transferable regime definitions.
    winsor_lower_quantile: float = 0.001
    winsor_upper_quantile: float = 0.999
    standardised_clip: float = 8.0
    regime_threshold_mode: str = "pooled_training"
    regime_target_mode: str = "relative_to_recent_volatility"
    ticker_specific_regime_smoothing: bool = False
    apply_regime_smoothing: bool = False
    regime_loss_type: str = "weighted_cross_entropy"
    regime_weighting: str = "sqrt_inverse_frequency"
    minimum_regime_class_weight: float = 0.50
    maximum_regime_class_weight: float = 3.00
    focal_gamma: float = 2.0
    regime_classifier_type: str = "softmax"
    regime_gate_mode: str = "soft"

    # Model capacity (the conditional-flow equations are unchanged).
    regimes: int = 3
    use_mixture_experts: bool = True
    include_har_feature: bool = True
    forecast_mode: str = "direct"
    use_ticker_embeddings: bool = not bool(PROFILE["quick_mode"])
    ticker_embedding_dimension: int = 8
    use_ticker_specific_heads: bool = False
    normalise_features_by_ticker: bool = True
    hidden_size: int = int(PROFILE["hidden_size"])
    dropout: float = 0.20
    minimum_scale: float = 0.03
    maximum_scale: float = 1.50
    maximum_skew: float = 1.50
    minimum_tail: float = 0.70
    maximum_tail: float = 2.00

    # Original full objective and its independently ablatable terms.
    expert_alignment_weight: float = 0.30
    regime_classification_weight: float = 0.10
    medium_class_multiplier: float = 1.00
    gate_balance_weight: float = 0.005
    qlike_weight: float = 0.05
    quantile_weight: float = 0.10
    label_smoothing: float = 0.02

    # Deeper normal-GPU optimisation profile.
    epochs: int = int(PROFILE["epochs"])
    batch_size: int = int(PROFILE["batch_size"])
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = int(PROFILE["patience"])
    minimum_epochs: int = int(PROFILE["minimum_epochs"])
    early_stopping_min_delta: float = 1e-4
    scheduler_patience: int = int(PROFILE["scheduler_patience"])
    scheduler_factor: float = 0.50
    scheduler_cooldown: int = 2
    minimum_learning_rate: float = 2.5e-6
    checkpoint_metric: str = "validation_qlike"
    print_every: int = 1 if PROFILE["quick_mode"] else 5
    pin_memory: bool = True
    matmul_precision: str = "high"

    # Ensemble, deterministic evaluation sampling, and predictions.
    ensemble_seeds: tuple[int, ...] = PROFILE["ensemble_seeds"]
    single_seed_ablation: bool = False
    prediction_seed: int = 20260803
    prediction_samples: int = int(PROFILE["prediction_samples"])
    prediction_batch_size: int = 512
    minimum_reported_volatility: float = 0.01
    maximum_reported_volatility: float = 5.00

    # Calibration-only searches. Toggle these manually for a prespecified
    # final-protocol sensitivity run; validation feature ablations do not
    # pretend to score calibration that has deliberately not been fitted.
    calibrate_intervals: bool = True
    calibrate_regime_probabilities: bool = True
    interval_coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    asymmetric_interval_calibration: bool = True
    ticker_interval_calibration: bool = True
    regime_conditional_interval_calibration: bool = True
    minimum_group_calibration_rows: int = 80
    blend_weight_candidates: int = 101
    blend_selection_metric: str = "qlike"
    interval_scale_min: float = 0.50
    interval_scale_max: float = 2.00
    interval_scale_candidates: int = 76
    temperature_min: float = 0.25
    temperature_max: float = 4.00
    temperature_candidates: int = 151

    # Result accuracy and dependence-aware evaluation.
    success_relative_tolerances: tuple[float, ...] = (
        0.05, 0.10, 0.20, 0.30, 0.50,
    )
    primary_success_tolerance: float = 0.20
    minimum_test_origins_per_ticker: int = int(
        PROFILE["minimum_test_origins_per_ticker"]
    )
    primary_evaluation_cohort: str = "unseen_ticker_unseen_day"
    primary_baseline: str = "Log-HAR transferable"
    primary_metric: str = "qlike"
    primary_min_relative_improvement: float = 0.05
    minimum_seed_improvement_fraction: float = 0.80
    significance_alpha: float = 0.05
    run_block_bootstrap: bool = bool(PROFILE["run_block_bootstrap"])
    bootstrap_repetitions: int = int(PROFILE["bootstrap_repetitions"])
    bootstrap_sensitivity_repetitions: int = 500
    bootstrap_block_days: int = 20
    bootstrap_block_sensitivity_days: tuple[int, ...] = (20, 40, 60, 120)
    ticker_date_bootstrap_repetitions: int = 1000
    display_all_test_results: bool = DISPLAY_ALL_TEST_RESULTS
    plot_each_ticker: bool = False
    save_plots: bool = True
    save_audit_artifacts: bool = False
    overwrite_existing_outputs: bool = False

    # Development-only ablation budget. Set reduced=False in the runner
    # for fully matched budgets after the shortlist is stable.
    ablation_reduced_budget: bool = bool(PROFILE["quick_mode"])
    ablation_epochs: int = 60
    ablation_patience: int = 8
    ablation_ensemble_seeds: tuple[int, ...] = PROFILE["ensemble_seeds"]

    # Expanding rolling-origin research protocol. The final-test runner creates
    # explicit chronological windows; a single pipeline test bypasses it.
    rolling_origin_windows: int = 4
    rolling_validation_dates: int = 126
    rolling_calibration_dates: int = 126
    rolling_test_dates: int = 126
    rolling_step_dates: int = 126
    decision_freeze_date: str = "2026-08-04"
    # Frozen on 2026-08-04 after the current redesign.  Earlier observations
    # are development evidence and can never support the confirmatory claim.
    prospective_test_start_date: str = "2026-08-05"
    prospective_test_end_date: str = "2028-08-31"
    prospective_data_end_date: str = "2028-09-15"
    final_train_end_date: str = "2025-06-30"
    final_validation_end_date: str = "2025-12-31"
    final_calibration_end_date: str = "2026-07-24"
    final_evaluation_authorized: bool = False

    seed: int = 42

    @property
    def target_tickers(self) -> tuple[str, ...]:
        return self.training_tickers + self.unseen_test_tickers

    @property
    def output_dir(self) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.run_name)
        return str(
            Path(self.output_root)
            / self.experiment_mode
            / f"horizon_{self.horizon}"
            / self.test_window_id
            / safe_name
        )


PREREGISTERED_FINAL_PROTOCOL_SHA256 = (
    "2adf58a4db7650bdd8323e32a64be954eba59dc3cffd3de02a6867d615ba03b1"
)


GROUP_ABLATIONS = {
    spec.name: spec
    for spec in (
        AblationSpec("no_signature", "Remove every path-signature coordinate.", (("include_signature_features", False),)),
        AblationSpec("no_signature_l3", "Remove signature level 3 only.", (("disabled_feature_groups", ("signature_l3",)),)),
        AblationSpec("no_shape_signatures", "Remove locally normalised shape signatures.", (("include_shape_signatures", False),)),
        AblationSpec("no_amplitude_signatures", "Remove scale-preserving amplitude signatures.", (("include_amplitude_signatures", False),)),
        AblationSpec("no_ohlc_signatures", "Remove OHLC range/gap path signatures.", (("include_ohlc_signatures", False),)),
        AblationSpec("no_joint_market_signatures", "Remove joint asset-SPY-VIX signatures.", (("include_joint_market_signatures", False),)),
        AblationSpec("no_lead_lag_signatures", "Remove lead-lag signatures.", (("include_lead_lag_signatures", False),)),
        AblationSpec("no_signature_w10", "Remove every 10-session signature block.", (("disabled_feature_groups", ("signature_w10",)),)),
        AblationSpec("no_signature_w20", "Remove every 20-session signature block.", (("disabled_feature_groups", ("signature_w20",)),)),
        AblationSpec("no_signature_w60", "Remove every 60-session signature block.", (("disabled_feature_groups", ("signature_w60",)),)),
        AblationSpec("no_return_statistics", "Remove all return-statistic features.", (("include_return_statistics", False),)),
        AblationSpec("no_ohlcv", "Remove all OHLCV-derived features.", (("include_ohlcv_features", False),)),
        AblationSpec("no_market", "Remove all SPY, QQQ, and VIX context features.", (("include_market_features", False),)),
        AblationSpec("no_spy", "Remove SPY context only.", (("disabled_feature_groups", ("market_spy",)),)),
        AblationSpec("no_qqq", "Remove QQQ context only.", (("disabled_feature_groups", ("market_qqq",)),)),
        AblationSpec("no_vix", "Remove VIX context only.", (("disabled_feature_groups", ("market_vix",)),)),
        AblationSpec("with_ticker_identity", "Add seen-ticker identity; unseen tickers receive a neutral all-zero code.", (("include_ticker_identity", True),)),
        AblationSpec("with_ticker_embeddings", "Use learned ticker embeddings with an unknown-ticker bucket.", (("use_ticker_embeddings", True),)),
        AblationSpec("without_ticker_embeddings", "Remove learned ticker embeddings.", (("use_ticker_embeddings", False),)),
        AblationSpec("with_ticker_specific_heads", "Use shared layers with seen-ticker output adapters.", (("use_ticker_specific_heads", True),)),
        AblationSpec("pooled_feature_scaling", "Replace past-only per-ticker scaling with pooled training scaling.", (("normalise_features_by_ticker", False),)),
        AblationSpec("no_expert_alignment", "Set the expert-alignment weight to zero.", (("expert_alignment_weight", 0.0),)),
        AblationSpec("no_regime_classification", "Set the regime-classification weight to zero.", (("regime_classification_weight", 0.0),)),
        AblationSpec("plain_regime_cross_entropy", "Use unweighted regime cross-entropy.", (("regime_loss_type", "cross_entropy"),)),
        AblationSpec("weighted_regime_cross_entropy", "Use capped square-root inverse-frequency weights.", (("regime_loss_type", "weighted_cross_entropy"),)),
        AblationSpec("focal_regime_loss", "Use capped class-weighted focal loss.", (("regime_loss_type", "focal"),)),
        AblationSpec("ordinal_regimes", "Predict ordered exceedance thresholds instead of a three-way softmax.", (("regime_classifier_type", "ordinal"),)),
        AblationSpec("no_regimes", "Remove regime classification, alignment and forecasting gate.", (
            ("regime_gate_mode", "none"),
            ("regime_classification_weight", 0.0),
            ("expert_alignment_weight", 0.0),
            ("gate_balance_weight", 0.0),
        )),
        AblationSpec("regime_auxiliary_only", "Train the regime task without using it for forecasting.", (
            ("regime_gate_mode", "auxiliary"),
            ("expert_alignment_weight", 0.0),
            ("gate_balance_weight", 0.0),
        )),
        AblationSpec("soft_regime_gate", "Use probabilistic regime gating for forecasting.", (
            ("regime_gate_mode", "soft"),
            # Keep the isolated gate comparison matched to auxiliary-only;
            # expert alignment and balance are evaluated separately/full-model.
            ("expert_alignment_weight", 0.0),
            ("gate_balance_weight", 0.0),
        )),
        AblationSpec("with_regime_smoothing", "Apply the prespecified past-only regime smoother.", (("apply_regime_smoothing", True),)),
        AblationSpec("single_distribution", "Use one forecast distribution instead of mixture experts.", (
            ("use_mixture_experts", False),
            # With one component, expert alignment would duplicate the same
            # likelihood term and unfairly rescale this ablation's objective.
            ("expert_alignment_weight", 0.0),
            ("gate_balance_weight", 0.0),
        )),
        AblationSpec("without_har_feature", "Remove the training-only HAR forecast from model context.", (("include_har_feature", False),)),
        AblationSpec("har_residual", "Predict an additive correction to the training-only HAR forecast.", (("forecast_mode", "har_residual"),)),
        AblationSpec("simple_neural", "Neural control without signatures, regimes, or mixture experts.", (
            ("include_signature_features", False),
            ("regime_gate_mode", "none"),
            ("regime_classification_weight", 0.0),
            ("expert_alignment_weight", 0.0),
            ("gate_balance_weight", 0.0),
            ("use_mixture_experts", False),
        )),
        AblationSpec("single_seed", "Compare seed 42 against the distributional ensemble.", (
            ("ensemble_seeds", (42,)),
            ("single_seed_ablation", True),
        )),
        AblationSpec("regime_weight_030", "Restore the stronger 0.30 classification weight.", (("regime_classification_weight", 0.30),)),
        AblationSpec("absolute_regime_labels", "Use pooled absolute-volatility regime labels.", (("regime_target_mode", "absolute_volatility"),)),
        AblationSpec("no_gate_balance", "Set the gate-balance weight to zero.", (("gate_balance_weight", 0.0),)),
        AblationSpec("no_qlike_term", "Set the QLIKE auxiliary weight to zero.", (("qlike_weight", 0.0),)),
        AblationSpec("no_quantile_term", "Set the quantile auxiliary weight to zero.", (("quantile_weight", 0.0),)),
        AblationSpec("no_auxiliary_losses", "Train with mixture NLL only.", (
            ("expert_alignment_weight", 0.0),
            ("regime_classification_weight", 0.0),
            ("gate_balance_weight", 0.0),
            ("qlike_weight", 0.0),
            ("quantile_weight", 0.0),
        )),
    )
}


def validate_config(cfg: Config) -> None:
    if not cfg.training_tickers or not cfg.unseen_test_tickers:
        raise ValueError("Both training and unseen-test ticker lists must be non-empty.")
    if len(set(cfg.target_tickers)) != len(cfg.target_tickers):
        raise ValueError("Training and unseen-test tickers must be unique and disjoint.")
    if set(cfg.target_tickers) & set(cfg.market_symbols):
        raise ValueError("Target tickers and market-context symbols must be disjoint.")
    if cfg.train_fraction + cfg.validation_fraction + cfg.calibration_fraction >= 1.0:
        raise ValueError("Train, validation, and calibration fractions must sum below 1.")
    if cfg.regime_threshold_mode not in {"pooled_training", "per_training_ticker"}:
        raise ValueError("regime_threshold_mode must be pooled_training or per_training_ticker.")
    if cfg.regime_target_mode not in {
        "relative_to_recent_volatility", "absolute_volatility"
    }:
        raise ValueError("Unknown regime_target_mode.")
    if tuple(sorted(set(cfg.signature_windows))) != cfg.signature_windows:
        raise ValueError("signature_windows must be sorted and unique.")
    if tuple(sorted(set(cfg.secondary_signature_windows))) != cfg.secondary_signature_windows:
        raise ValueError("secondary_signature_windows must be sorted and unique.")
    if not cfg.signature_windows or max(cfg.signature_windows) > cfg.window:
        raise ValueError("Signature windows must be non-empty and at most window.")
    if not set(cfg.secondary_signature_windows) <= set(cfg.signature_windows):
        raise ValueError("Secondary signature windows must be signature windows.")
    signature_depths = (
        cfg.signature_depth,
        cfg.short_signature_depth,
        cfg.amplitude_signature_depth,
        cfg.ohlc_signature_depth,
        cfg.joint_market_signature_depth,
        cfg.lead_lag_signature_depth,
    )
    if any(depth not in (1, 2, 3) for depth in signature_depths):
        raise ValueError("Every signature depth must be 1, 2, or 3.")
    if cfg.signature_return_scale <= 0.0 or cfg.signature_vix_change_scale <= 0.0:
        raise ValueError("Signature scales must be positive.")
    if not set(cfg.joint_signature_symbols) <= set(cfg.market_symbols):
        raise ValueError("Joint-signature symbols must be configured market symbols.")
    if not set(cfg.target_tickers) <= set(TICKER_SECTORS):
        raise ValueError("Every configured target ticker needs exactly one sector.")
    if cfg.experiment_mode not in {"pipeline_test", "research"}:
        raise ValueError("experiment_mode must be pipeline_test or research.")
    if cfg.experiment_mode == "research":
        training_sectors = {
            TICKER_SECTORS[ticker] for ticker in cfg.training_tickers
        }
        unseen_sectors = {
            TICKER_SECTORS[ticker] for ticker in cfg.unseen_test_tickers
        }
        if training_sectors != unseen_sectors or len(unseen_sectors) < 10:
            raise ValueError(
                "Training and unseen universes must cover the same sectors."
            )
        if len(cfg.unseen_test_tickers) < 15:
            raise ValueError(
                "A real cross-ticker claim requires at least 15 held-out tickers."
            )
    frozen_dates = [
        pd.Timestamp(cfg.final_train_end_date),
        pd.Timestamp(cfg.final_validation_end_date),
        pd.Timestamp(cfg.final_calibration_end_date),
        pd.Timestamp(cfg.decision_freeze_date),
        pd.Timestamp(cfg.prospective_test_start_date),
        pd.Timestamp(cfg.prospective_test_end_date),
        pd.Timestamp(cfg.prospective_data_end_date),
    ]
    if frozen_dates != sorted(frozen_dates) or len(set(frozen_dates)) != len(
        frozen_dates
    ):
        raise ValueError(
            "Frozen fitting, decision, prospective-test, and data-end dates "
            "must be strictly increasing."
        )
    if cfg.checkpoint_metric not in {"validation_objective", "validation_qlike"}:
        raise ValueError("checkpoint_metric must be validation_objective or validation_qlike.")
    required_tolerances = {0.05, 0.10, 0.20, 0.30, 0.50}
    if not required_tolerances <= set(cfg.success_relative_tolerances):
        raise ValueError(
            "success_relative_tolerances must include 5%, 10%, 20%, 30%, and 50%."
        )
    if cfg.primary_success_tolerance not in cfg.success_relative_tolerances:
        raise ValueError(
            "primary_success_tolerance must appear in success_relative_tolerances."
        )
    if cfg.bootstrap_block_days not in cfg.bootstrap_block_sensitivity_days:
        raise ValueError("The primary bootstrap block must appear in sensitivity blocks.")
    if any(days < cfg.horizon for days in cfg.bootstrap_block_sensitivity_days):
        raise ValueError("Bootstrap blocks cannot be shorter than the forecast horizon.")
    if not 0.0 < cfg.primary_min_relative_improvement < 1.0:
        raise ValueError("primary_min_relative_improvement must be in (0, 1).")
    if not 0.0 < cfg.minimum_seed_improvement_fraction <= 1.0:
        raise ValueError("minimum_seed_improvement_fraction must be in (0, 1].")
    if not 1 <= cfg.minimum_epochs <= cfg.epochs:
        raise ValueError("minimum_epochs must be between 1 and epochs.")
    if not 1 <= cfg.scheduler_patience < cfg.patience:
        raise ValueError("scheduler_patience must be positive and below patience.")
    if not 0.0 < cfg.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be strictly between zero and one.")
    if not 0.0 < cfg.minimum_learning_rate < cfg.learning_rate:
        raise ValueError("minimum_learning_rate must be positive and below learning_rate.")
    if cfg.scheduler_cooldown < 0 or cfg.early_stopping_min_delta < 0.0:
        raise ValueError("Scheduler cooldown and stopping delta cannot be negative.")
    if cfg.experiment_mode == "research" and cfg.allow_synthetic_fallback:
        raise ValueError("Synthetic fallback must be disabled in research mode.")
    if cfg.experiment_mode == "research" and not cfg.use_real_market_data:
        raise ValueError("Research mode requires real market data.")
    if cfg.experiment_mode == "pipeline_test" and cfg.use_real_market_data:
        raise ValueError("pipeline_test must use synthetic data only.")
    if cfg.experiment_mode == "pipeline_test" and cfg.allow_synthetic_fallback:
        raise ValueError("pipeline_test selects synthetic data directly; fallback is invalid.")
    if cfg.regime_loss_type not in {
        "cross_entropy", "weighted_cross_entropy", "focal"
    }:
        raise ValueError("Unknown regime_loss_type.")
    if cfg.regime_weighting != "sqrt_inverse_frequency":
        raise ValueError("Only sqrt_inverse_frequency weighting is supported.")
    if cfg.regime_classifier_type not in {"softmax", "ordinal"}:
        raise ValueError("regime_classifier_type must be softmax or ordinal.")
    if cfg.regime_gate_mode not in {"none", "auxiliary", "soft"}:
        raise ValueError("regime_gate_mode must be none, auxiliary, or soft.")
    if cfg.forecast_mode not in {"direct", "har_residual"}:
        raise ValueError("forecast_mode must be direct or har_residual.")
    if tuple(sorted(set(cfg.forecast_horizons))) != cfg.forecast_horizons:
        raise ValueError("forecast_horizons must be sorted and unique.")
    if any(horizon < 1 for horizon in cfg.forecast_horizons):
        raise ValueError("Every forecast horizon must be positive.")
    if cfg.horizon not in cfg.forecast_horizons:
        raise ValueError("The active horizon must appear in forecast_horizons.")
    if len(cfg.ensemble_seeds) < (
        1 if (cfg.quick_mode or cfg.single_seed_ablation) else 5
    ):
        raise ValueError("Research mode requires at least five ensemble seeds.")
    if cfg.interval_coverage_levels != (0.50, 0.80, 0.90, 0.95):
        raise ValueError(
            "interval_coverage_levels must be exactly (0.50, 0.80, 0.90, 0.95)."
        )
    if cfg.minimum_regime_class_weight <= 0.0:
        raise ValueError("Regime class weights must be positive.")
    if cfg.maximum_regime_class_weight < cfg.minimum_regime_class_weight:
        raise ValueError("Maximum class weight must not be below the minimum.")


CFG = Config()
validate_config(CFG)
if VERBOSE_IMPORT:
    print(json.dumps({
        **asdict(CFG),
        "target_tickers": CFG.target_tickers,
        "output_dir": CFG.output_dir,
    }, indent=2))

# %% [markdown cell 5]
# ## Reproducibility, robust preprocessing, and data containers

# %% [cell 6]
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


def model_state_hash(models: list[nn.Module]) -> str:
    digest = hashlib.sha256()
    for model_index, model in enumerate(models):
        digest.update(str(model_index).encode("ascii"))
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def append_audit_check(
    split: "PreparedSplit",
    description: str,
    passed: bool,
    observed: object,
    boundary: object,
) -> None:
    row = pd.DataFrame([{
        "check": description,
        "passed": bool(passed),
        "observed": observed,
        "boundary": boundary,
    }])
    split.leakage_audit = pd.concat(
        [split.leakage_audit, row], ignore_index=True
    )
    if not passed:
        raise AssertionError(description)


@torch.no_grad()
def audit_target_invariance(
    model: nn.Module,
    dataset: "MarketDataset",
    split: "PreparedSplit",
    indices: np.ndarray,
    cfg: Config,
) -> bool:
    selected = indices[: min(16, len(indices))]
    context = torch.from_numpy(split.context[selected]).float().to(DEVICE)
    targets = torch.from_numpy(split.model_targets[selected]).float().unsqueeze(1).to(DEVICE)
    ticker_ids = torch.from_numpy(
        dataset.metadata.iloc[selected]["ticker_id"].to_numpy(
            dtype=np.int64, copy=True
        )
    ).long().to(DEVICE)
    original = model.forward_bundle(
        context, targets, ticker_ids=ticker_ids
    )
    permuted = model.forward_bundle(
        context, targets.flip(0), ticker_ids=ticker_ids
    )
    return bool(
        torch.equal(original.logits, permuted.logits)
        and torch.equal(original.locations, permuted.locations)
        and torch.equal(original.scales, permuted.scales)
    )


@dataclass
class RobustStandardiser:
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    z_clip: float

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        lower_quantile: float,
        upper_quantile: float,
        z_clip: float,
    ) -> "RobustStandardiser":
        lower = np.nanquantile(values, lower_quantile, axis=0)
        upper = np.nanquantile(values, upper_quantile, axis=0)
        upper = np.maximum(upper, lower)

        clipped = np.clip(values, lower, upper)
        mean = np.nanmean(clipped, axis=0)
        scale = np.nanstd(clipped, axis=0)
        scale = np.where(
            np.isfinite(scale) & (scale > 1e-6),
            scale,
            1.0,
        )

        return cls(
            lower=lower.astype(np.float32),
            upper=upper.astype(np.float32),
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            z_clip=float(z_clip),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.lower, self.upper)
        transformed = (clipped - self.mean) / self.scale
        transformed = np.clip(
            transformed,
            -self.z_clip,
            self.z_clip,
        )
        return np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=self.z_clip,
            neginf=-self.z_clip,
        ).astype(np.float32)


@dataclass
class MarketDataset:
    features: np.ndarray
    targets_log_vol: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]
    data_source: str
    available_market_symbols: tuple[str, ...]
    data_manifest: pd.DataFrame
    skipped_samples: pd.DataFrame
    construction_config: dict[str, object]


@dataclass
class PreparedSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    seen_ticker_unseen_day_indices: np.ndarray
    unseen_ticker_unseen_day_indices: np.ndarray
    context: np.ndarray
    model_targets: np.ndarray
    har_predictions: np.ndarray
    regime_labels: np.ndarray
    ticker_thresholds: np.ndarray
    class_weights: np.ndarray
    train_regime_proportions: np.ndarray
    transition_matrices: np.ndarray
    initial_regime_probabilities: np.ndarray
    standardiser: RobustStandardiser
    ticker_standardisers: dict[str, RobustStandardiser]
    active_feature_indices: np.ndarray
    active_feature_names: list[str]
    dropped_feature_names: list[str]
    context_feature_names: list[str]
    train_cutoff: pd.Timestamp
    validation_cutoff: pd.Timestamp
    calibration_cutoff: pd.Timestamp
    leakage_audit: pd.DataFrame


set_seed(CFG.seed)

# %% [markdown cell 7]
# ## Multiscale, amplitude-preserving path signatures
# 
# The canonical representation separates **shape** from **amplitude**. Shape paths retain local standardisation; amplitude paths use a fixed 1% return unit so volatility scale is not erased. Signatures are generated over 10, 20, and 60 sessions, with additional 20-session OHLC range/gap, joint asset–SPY–VIX, and lead-lag paths. Short and secondary paths use depth two for ordinary-GPU practicality; the 60-session shape path retains depth three.
# 
# Every family and window has a named zero-mask ablation. The full signature tensor is dependency-free and validated against amplitude/shape invariants during testing.

# %% [cell 8]
def truncated_signature(path: np.ndarray, depth: int = 3) -> np.ndarray:
    if depth not in (1, 2, 3):
        raise ValueError("Supported signature depths are 1, 2, and 3.")

    increments = np.diff(np.asarray(path, dtype=np.float64), axis=0)
    level1 = increments.sum(axis=0)
    levels = [level1.ravel()]
    if depth >= 2:
        # Chen's identity can be evaluated for every segment at once.  The
        # prefix tensors are precisely the old level-1/level-2 states used by
        # the former Python loop, but NumPy performs the time-axis reductions.
        prefix1 = np.cumsum(increments, axis=0) - increments
        segment2 = np.einsum(
            "ti,tj->tij", increments, increments
        ) / 2.0
        level2_contributions = (
            np.einsum("ti,tj->tij", prefix1, increments)
            + segment2
        )
        level2 = level2_contributions.sum(axis=0)
        levels.append(level2.ravel())
    if depth >= 3:
        prefix2 = (
            np.cumsum(level2_contributions, axis=0)
            - level2_contributions
        )
        level3 = (
            np.einsum("tij,tk->ijk", prefix2, increments)
            + np.einsum("ti,tjk->ijk", prefix1, segment2)
            + np.einsum(
                "ti,tj,tk->ijk", increments, increments, increments
            ) / 6.0
        )
        levels.append(level3.ravel())
    return np.concatenate(levels)


def signature_feature_names(
    prefix: str,
    path_dimension: int,
    depth: int,
) -> list[str]:
    names = []
    for level in range(1, depth + 1):
        for index in range(path_dimension**level):
            names.append(f"signature_{prefix}_L{level}_{index}")
    return names

@dataclass(frozen=True)
class SignatureBlockSpec:
    family: str
    window: int
    path_dimension: int
    depth: int

    @property
    def prefix(self) -> str:
        return f"{self.family}_w{self.window}"

# %% [markdown cell 9]
# ## Direct OHLCV loading with provenance and no invented prices

# %% [cell 10]
REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
ALLOWED_REAL_DATA_PROVIDERS = frozenset({"yahoo_direct", "stooq_direct"})
PROVIDER_HOSTS = {
    "yahoo_direct": frozenset({"query1.finance.yahoo.com"}),
    "stooq_direct": frozenset({"stooq.com", "www.stooq.com"}),
}
KNOWN_LISTING_DATES = {
    "TSLA": "2010-06-29",
    "META": "2012-05-18",
}
MAXIMUM_CACHE_TAIL_GAP_DAYS = 10
MINIMUM_OBSERVED_BUSINESS_DAY_COVERAGE = 0.85


def validate_market_frame_coverage(
    frame: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
) -> None:
    """Reject truncated or unusually sparse provider/cache responses."""
    first = pd.Timestamp(frame.index.min()).normalize()
    last = pd.Timestamp(frame.index.max()).normalize()
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    listing = max(
        requested_start,
        pd.Timestamp(KNOWN_LISTING_DATES.get(symbol, start_date)).normalize(),
    )
    if first > listing + pd.Timedelta(days=31):
        raise ValueError(
            f"{symbol} history starts at {first.date()}, too late for the "
            f"requested/listing boundary {listing.date()}."
        )
    tail_gap = int((requested_end - last).days)
    if tail_gap > MAXIMUM_CACHE_TAIL_GAP_DAYS:
        raise ValueError(
            f"{symbol} history ends at {last.date()}, {tail_gap} days before "
            f"requested end {requested_end.date()}; refusing a truncated tail."
        )
    expected_weekdays = pd.bdate_range(first, last)
    observed_weekdays = pd.DatetimeIndex(frame.index).normalize().intersection(
        expected_weekdays
    )
    coverage = len(observed_weekdays) / max(len(expected_weekdays), 1)
    if coverage < MINIMUM_OBSERVED_BUSINESS_DAY_COVERAGE:
        raise ValueError(
            f"{symbol} business-day coverage is {coverage:.1%}, below "
            f"{MINIMUM_OBSERVED_BUSINESS_DAY_COVERAGE:.0%}."
        )


def validate_cache_provenance(
    metadata: dict[str, object],
    symbol: str,
    cfg: Config,
) -> None:
    if metadata.get("schema_version") != 2:
        raise ValueError("unsupported or missing cache schema_version")
    if metadata.get("source_kind") != "real_market":
        raise ValueError("cache source_kind is not real_market")
    if metadata.get("symbol") != symbol:
        raise ValueError("cache symbol metadata mismatch")
    if metadata.get("requested_start_date") != cfg.start_date:
        raise ValueError("cache start-date metadata mismatch")
    if metadata.get("requested_end_date") != cfg.end_date:
        raise ValueError("cache end-date metadata mismatch")
    provider = str(metadata.get("provider", ""))
    if provider not in ALLOWED_REAL_DATA_PROVIDERS:
        raise ValueError(f"unrecognized cache provider {provider!r}")
    source_url = str(metadata.get("source_url", ""))
    parsed_url = urllib.parse.urlparse(source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in PROVIDER_HOSTS[provider]
    ):
        raise ValueError("cache source URL does not match its provider")
    downloaded = pd.Timestamp(metadata.get("downloaded_at_utc"))
    if (
        pd.isna(downloaded)
        or downloaded.tzinfo is None
        or downloaded.utcoffset() != pd.Timedelta(0)
    ):
        raise ValueError("cache download timestamp must be parseable and UTC")
    expected_hash = str(metadata.get("csv_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("cache metadata has no valid SHA-256 checksum")
    content_file = str(metadata.get("content_file", ""))
    content_parts = Path(content_file).parts
    if (
        not content_file
        or Path(content_file).is_absolute()
        or ".." in content_parts
        or not content_parts
        or content_parts[0] != ".objects"
    ):
        raise ValueError("cache metadata has no safe content-addressed object")
    incremental_hash = metadata.get("incremental_base_sha256")
    incremental_start = metadata.get("incremental_fetch_start_date")
    incremental_path = metadata.get("incremental_base_cache_path")
    lineage_values = (incremental_hash, incremental_start, incremental_path)
    if any(value is not None for value in lineage_values):
        if not all(value is not None for value in lineage_values):
            raise ValueError("incremental cache lineage is incomplete")
        if re.fullmatch(r"[0-9a-f]{64}", str(incremental_hash)) is None:
            raise ValueError("incremental base cache has no valid SHA-256")
        fetch_start = pd.Timestamp(incremental_start)
        if (
            pd.isna(fetch_start)
            or fetch_start < pd.Timestamp(cfg.start_date)
            or fetch_start > pd.Timestamp(cfg.end_date)
        ):
            raise ValueError("incremental fetch start is outside the requested range")
        if not str(incremental_path).strip():
            raise ValueError("incremental base cache path is empty")


def clean_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    # Providers encode daily bars at different UTC clock times (for example,
    # Yahoo equities at the US open and VIX near midnight).  The observation
    # key is the trading date, not that provider-specific transport timestamp.
    # Normalising here makes cross-market alignment exact without permitting
    # backward filling or using any future observation.
    frame.index = frame.index.normalize()

    frame = frame[~frame.index.isna()]
    for column in REQUIRED_PRICE_COLUMNS + ["Volume"]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Negative volume and impossible OHLC ranges indicate a malformed or
    # truncated provider response.  They must not silently become features.
    frame.loc[frame["Volume"] < 0.0, "Volume"] = np.nan

    # Never replace missing Open/High/Low with Close. Those rows are dropped.
    frame = frame.dropna(subset=REQUIRED_PRICE_COLUMNS)
    frame = frame[
        (frame["Open"] > 0)
        & (frame["High"] > 0)
        & (frame["Low"] > 0)
        & (frame["Close"] > 0)
        & (frame["High"] >= frame["Open"])
        & (frame["High"] >= frame["Close"])
        & (frame["High"] >= frame["Low"])
        & (frame["Low"] <= frame["Open"])
        & (frame["Low"] <= frame["Close"])
    ]

    current_volume_missing = frame["Volume"].isna()
    if "VolumeMissing" in frame:
        cached_volume_missing = pd.to_numeric(
            frame["VolumeMissing"], errors="coerce"
        ).fillna(0.0).gt(0.0)
    else:
        cached_volume_missing = pd.Series(False, index=frame.index)
    frame["VolumeMissing"] = (
        current_volume_missing | cached_volume_missing
    ).astype(float)
    frame["Volume"] = frame["Volume"].fillna(0.0)

    frame = frame[
        ["Open", "High", "Low", "Close", "Volume", "VolumeMissing"]
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=REQUIRED_PRICE_COLUMNS)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if frame.empty:
        raise ValueError(f"No usable OHLCV observations for {symbol}.")
    return frame


def request_bytes(
    url: str,
    timeout: int,
    cfg: Config,
) -> bytes:
    policy = RetryPolicy(
        attempts=cfg.download_attempts,
        initial_delay_seconds=cfg.download_initial_backoff_seconds,
        maximum_delay_seconds=cfg.download_maximum_backoff_seconds,
        jitter_fraction=cfg.download_jitter_fraction,
    )

    def report_retry(
        attempt: int,
        status: int | None,
        delay: float,
        reason: str,
    ) -> None:
        detail = f"HTTP {status}" if status is not None else reason
        print(
            f"    retry {attempt}/{policy.attempts - 1} after {detail}; "
            f"backing off {delay:.2f}s",
            flush=True,
        )

    return request_bytes_with_retry(
        url,
        timeout_seconds=timeout,
        policy=policy,
        on_retry=report_retry,
    )


def network_preflight(cfg: Config) -> tuple[str, str]:
    if not cfg.run_network_preflight:
        return "reachable", "preflight disabled"

    probes = (
        (
            "Yahoo",
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            "AAPL?range=5d&interval=1d",
        ),
        (
            "Stooq",
            "https://stooq.com/q/d/l/?s=aapl.us&i=d",
        ),
    )
    print(
        "Checking external market-data connectivity "
        f"(timeout={cfg.network_preflight_timeout_seconds}s)...",
        flush=True,
    )

    failures = []
    for provider, url in probes:
        try:
            payload = request_bytes(
                url,
                timeout=cfg.network_preflight_timeout_seconds,
                cfg=cfg,
            )
            if not payload:
                raise ValueError("empty response")
            print(
                f"  {provider} connectivity check succeeded.",
                flush=True,
            )
            return "reachable", f"{provider} reachable"
        except RateLimitError as exc:
            message = f"{provider}: rate_limited: {exc}"
            failures.append(message)
            print("  Connectivity check was rate limited: " + message, flush=True)
        except NetworkUnavailableError as exc:
            message = f"{provider}: network_unavailable: {exc}"
            failures.append(message)
            print("  Connectivity check could not connect: " + message, flush=True)
        except DataDownloadError as exc:
            message = f"{provider}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(
                "  Provider check failed: " + message,
                flush=True,
            )
    joined = " | ".join(failures)
    if failures and all("rate_limited" in failure for failure in failures):
        return "rate_limited", joined
    if failures and all("network_unavailable" in failure for failure in failures):
        return "network_unavailable", joined
    return "provider_error", joined


def download_yahoo_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    timeout: int,
    cfg: Config,
) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    period2 = int(pd.Timestamp(end_date, tz="UTC").timestamp())

    parameters = urllib.parse.urlencode({
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    })
    quoted_symbol = urllib.parse.quote(symbol, safe="")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quoted_symbol}?{parameters}"
    )

    payload = json.loads(
        request_bytes(url, timeout=timeout, cfg=cfg).decode("utf-8")
    )
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise DataDownloadError(
            f"Yahoo application-level error for {symbol}: {chart['error']}"
        )

    results = chart.get("result")
    if not results:
        raise ValueError(f"Yahoo returned no result for {symbol}.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_blocks = result.get("indicators", {}).get("quote") or []
    if not timestamps or not quote_blocks:
        raise ValueError(f"Yahoo returned incomplete OHLCV for {symbol}.")

    quote = quote_blocks[0]
    dates = pd.to_datetime(
        timestamps,
        unit="s",
        utc=True,
    ).tz_localize(None)

    frame = pd.DataFrame({
        "Open": quote.get("open"),
        "High": quote.get("high"),
        "Low": quote.get("low"),
        "Close": quote.get("close"),
        "Volume": quote.get("volume"),
    }, index=dates)

    adjusted_blocks = (
        result.get("indicators", {}).get("adjclose") or []
    )
    adjusted = (
        adjusted_blocks[0].get("adjclose")
        if adjusted_blocks else None
    )

    if adjusted is not None:
        raw_close = pd.to_numeric(
            frame["Close"], errors="coerce"
        )
        adjustment = (
            pd.Series(adjusted, index=dates) / raw_close
        )
        adjustment = adjustment.replace(
            [np.inf, -np.inf], np.nan
        )

        for column in REQUIRED_PRICE_COLUMNS:
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                * adjustment
            )

    cleaned = clean_ohlcv(frame, symbol)
    cleaned.attrs["source_url"] = url
    return cleaned


def stooq_symbol(symbol: str) -> str:
    if symbol == "^VIX":
        return "vix"
    return symbol.lower().replace(".", "-") + ".us"


def download_stooq_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    timeout: int,
    cfg: Config,
) -> pd.DataFrame:
    parameters = urllib.parse.urlencode({
        "s": stooq_symbol(symbol),
        "d1": pd.Timestamp(start_date).strftime("%Y%m%d"),
        "d2": pd.Timestamp(end_date).strftime("%Y%m%d"),
        "i": "d",
    })
    url = f"https://stooq.com/q/d/l/?{parameters}"
    raw = request_bytes(
        url,
        timeout=timeout,
        cfg=cfg,
    )
    frame = pd.read_csv(io.BytesIO(raw))

    if frame.empty or "Date" not in frame or "Close" not in frame:
        preview = raw[:150].decode("utf-8", errors="replace")
        raise ValueError(f"Stooq returned no useful data: {preview!r}")

    frame.index = pd.to_datetime(frame.pop("Date"), errors="coerce")
    cleaned = clean_ohlcv(frame, symbol)
    cleaned.attrs["source_url"] = url
    return cleaned


def synthetic_frames(cfg: Config) -> dict[str, pd.DataFrame]:
    dates = pd.date_range(
        cfg.start_date,
        cfg.end_date,
        freq="B",
        inclusive="left",
    )
    rng = np.random.default_rng(cfg.seed + 9000)

    regimes = np.zeros(len(dates), dtype=int)
    transition = np.array([
        [0.975, 0.023, 0.002],
        [0.030, 0.940, 0.030],
        [0.010, 0.080, 0.910],
    ])
    for index in range(1, len(dates)):
        regimes[index] = rng.choice(
            3,
            p=transition[regimes[index - 1]],
        )

    common_volatility = np.array(
        [0.007, 0.014, 0.030]
    )[regimes]
    market_return = rng.normal(0.0002, common_volatility)

    frames = {}
    symbols = list(cfg.target_tickers) + (
        list(cfg.market_symbols)
        if cfg.include_market_context else []
    )

    for symbol_id, symbol in enumerate(symbols):
        local_rng = np.random.default_rng(
            cfg.seed + 1000 * (symbol_id + 1)
        )

        if symbol == "^VIX":
            close = (
                13.0
                + 7.0 * regimes
                + 250.0 * np.abs(market_return)
                + local_rng.normal(0.0, 1.4, len(dates))
            )
            close = np.maximum(close, 8.0)
            open_price = close * np.exp(
                local_rng.normal(0.0, 0.01, len(dates))
            )
        else:
            beta = 0.75 + 0.12 * symbol_id
            idiosyncratic = local_rng.normal(
                0.0001,
                np.array([0.006, 0.010, 0.020])[regimes],
            )
            returns = beta * market_return + idiosyncratic
            close = 100.0 * np.exp(np.cumsum(returns))
            previous_close = np.concatenate([[close[0]], close[:-1]])
            open_price = previous_close * np.exp(
                local_rng.normal(
                    0.0,
                    common_volatility * 0.25,
                )
            )

        range_scale = (
            0.004
            + 0.8 * np.abs(
                np.log(close / np.maximum(open_price, EPS))
            )
            + 0.5 * common_volatility
        )
        high = np.maximum(open_price, close) * np.exp(
            range_scale / 2.0
        )
        low = np.minimum(open_price, close) * np.exp(
            -range_scale / 2.0
        )
        volume = local_rng.lognormal(
            mean=16.0 + 7.0 * common_volatility,
            sigma=0.35,
            size=len(dates),
        )

        frames[symbol] = clean_ohlcv(
            pd.DataFrame({
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }, index=dates),
            symbol,
        )

    return frames


def safe_cache_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace("/", "_")


def cache_path_for(symbol: str, cfg: Config) -> Path:
    safe_symbol = safe_cache_symbol(symbol)
    directory = Path(cfg.cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (
        f"{safe_symbol}_{cfg.start_date}_{cfg.end_date}.csv"
    )


def cache_metadata_path_for(symbol: str, cfg: Config) -> Path:
    return cache_path_for(symbol, cfg).with_suffix(".metadata.json")


def load_latest_prior_cache(
    symbol: str,
    cfg: Config,
) -> tuple[pd.DataFrame, str] | None:
    """Find the newest verified snapshot that can seed an incremental update."""
    directory = Path(cfg.cache_dir)
    if not directory.exists():
        return None
    prefix = f"{safe_cache_symbol(symbol)}_{cfg.start_date}_"
    candidates: list[tuple[pd.Timestamp, Config]] = []
    for metadata_path in directory.glob(f"{prefix}*.metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_end = pd.Timestamp(metadata.get("requested_end_date"))
            if (
                pd.isna(candidate_end)
                or candidate_end > pd.Timestamp(cfg.end_date)
            ):
                continue
            candidate_cfg = replace(
                cfg,
                end_date=candidate_end.date().isoformat(),
                refresh_data=False,
            )
            if cache_metadata_path_for(symbol, candidate_cfg) != metadata_path:
                continue
            candidates.append((candidate_end, candidate_cfg))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    for _, candidate_cfg in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        frame = load_cached_frame(symbol, candidate_cfg, ignore_refresh=True)
        if frame is not None:
            return frame, str(frame.attrs["provider"])
    return None


def load_cached_frame(
    symbol: str,
    cfg: Config,
    *,
    ignore_refresh: bool = False,
) -> pd.DataFrame | None:
    path = cache_path_for(symbol, cfg)
    if cfg.refresh_data and not ignore_refresh:
        return None

    metadata_path = cache_metadata_path_for(symbol, cfg)
    if not metadata_path.exists():
        if path.exists():
            print(
                f"  {symbol}: ignoring unverified legacy cache without metadata",
                flush=True,
            )
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_cache_provenance(metadata, symbol, cfg)
        expected_hash = str(metadata.get("csv_sha256", ""))
        content_path = metadata_path.parent / str(metadata["content_file"])
        actual_hash = sha256_bytes(content_path.read_bytes())
        if expected_hash != actual_hash:
            raise ValueError("cache checksum mismatch")

        frame = pd.read_csv(content_path, index_col=0, parse_dates=True)
        if frame.empty:
            raise ValueError("cache contains no rows")
        cleaned = clean_ohlcv(frame, symbol)
        expected_rows = int(metadata.get("row_count", -1))
        if expected_rows != len(cleaned):
            raise ValueError(
                f"cache row-count mismatch: {len(cleaned)} != {expected_rows}"
            )
        if str(cleaned.index.min().date()) != str(metadata.get("first_date")):
            raise ValueError("cache first-date metadata mismatch")
        if str(cleaned.index.max().date()) != str(metadata.get("last_date")):
            raise ValueError("cache last-date metadata mismatch")
        if cleaned.index.min() < pd.Timestamp(cfg.start_date):
            raise ValueError("cache contains rows before the requested range")
        if cleaned.index.max() > pd.Timestamp(cfg.end_date):
            raise ValueError("cache contains rows after the requested range")
        validate_market_frame_coverage(
            cleaned, symbol, cfg.start_date, cfg.end_date
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        print(f"  {symbol}: ignoring invalid cache: {exc}", flush=True)
        return None

    cleaned.attrs.update({
        "provider": metadata.get("provider", "legacy_cache"),
        "downloaded_at_utc": metadata.get("downloaded_at_utc"),
        "source_url": metadata.get("source_url"),
        "cache_path": str(content_path),
        "cache_pointer_path": str(path),
        "csv_sha256": metadata.get("csv_sha256"),
        "requested_start_date": metadata.get("requested_start_date"),
        "requested_end_date": metadata.get("requested_end_date"),
        "incremental_base_cache_path": metadata.get(
            "incremental_base_cache_path"
        ),
        "incremental_base_sha256": metadata.get("incremental_base_sha256"),
        "incremental_fetch_start_date": metadata.get(
            "incremental_fetch_start_date"
        ),
        "cache_hit": True,
    })
    print(
        f"  {symbol}: loaded verified cache "
        f"(origin={cleaned.attrs['provider']}, "
        f"downloaded={cleaned.attrs['downloaded_at_utc'] or 'unknown'})",
        flush=True,
    )
    return cleaned


def save_cached_frame(
    frame: pd.DataFrame,
    symbol: str,
    provider: str,
    cfg: Config,
) -> None:
    path = cache_path_for(symbol, cfg)
    validate_market_frame_coverage(frame, symbol, cfg.start_date, cfg.end_date)
    csv_payload = frame.to_csv().encode("utf-8")
    policy = RetryPolicy(
        attempts=cfg.download_attempts,
        initial_delay_seconds=cfg.download_initial_backoff_seconds,
        maximum_delay_seconds=cfg.download_maximum_backoff_seconds,
        jitter_fraction=cfg.download_jitter_fraction,
    )
    metadata = cache_metadata(
        symbol=symbol,
        provider=provider,
        source_url=str(frame.attrs.get("source_url", "unknown")),
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        row_count=len(frame),
        first_date=str(frame.index.min().date()),
        last_date=str(frame.index.max().date()),
        csv_payload=csv_payload,
        retry_policy=policy,
    )
    object_directory = path.parent / ".objects"
    object_directory.mkdir(parents=True, exist_ok=True)
    content_path = object_directory / f"{metadata['csv_sha256']}.csv"
    metadata["content_file"] = str(content_path.relative_to(path.parent))
    for key in (
        "incremental_base_cache_path",
        "incremental_base_sha256",
        "incremental_fetch_start_date",
    ):
        if frame.attrs.get(key) is not None:
            metadata[key] = frame.attrs[key]
    validate_cache_provenance(metadata, symbol, cfg)
    temporary = object_directory / f".{content_path.name}.{os.getpid()}.tmp"
    content_is_valid = (
        content_path.exists()
        and sha256_bytes(content_path.read_bytes()) == metadata["csv_sha256"]
    )
    if not content_is_valid:
        try:
            with temporary.open("wb") as stream:
                stream.write(csv_payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, content_path)
            fsync_directory(object_directory)
        finally:
            if temporary.exists():
                temporary.unlink()
    # The metadata file is the atomic snapshot pointer. A crash before this
    # replacement leaves the previous pointer and immutable object valid.
    write_json_atomic(cache_metadata_path_for(symbol, cfg), metadata)
    # Maintain the historical CSV path as a convenience hard link only; cache
    # reads use the verified content-addressed object referenced by metadata.
    pointer_temporary = path.with_suffix(f".csv.{os.getpid()}.tmp")
    try:
        if pointer_temporary.exists():
            pointer_temporary.unlink()
        os.link(content_path, pointer_temporary)
        os.replace(pointer_temporary, path)
    except OSError as exc:
        # This link exists only for people/tools expecting the historical CSV
        # filename. The committed metadata pointer and immutable object are the
        # authoritative cache, so unsupported hard links must not report a
        # false save failure.
        warnings.warn(
            f"Cache committed for {symbol}, but convenience link creation "
            f"failed: {exc}",
            RuntimeWarning,
        )
    finally:
        if pointer_temporary.exists():
            pointer_temporary.unlink()
    frame.attrs.update({
        **metadata,
        "cache_path": str(content_path),
        "cache_pointer_path": str(path),
        "cache_hit": False,
    })


def validate_incremental_overlap(
    prior: pd.DataFrame,
    tail: pd.DataFrame,
    symbol: str,
    *,
    relative_tolerance: float = 1e-3,
) -> None:
    """Reject a tail whose adjusted OHLC scale disagrees with its base cache."""

    overlap = prior.index.intersection(tail.index)
    if overlap.empty:
        raise ValueError(
            f"{symbol} incremental response has no overlap with its base cache"
        )
    previous = prior.loc[overlap, REQUIRED_PRICE_COLUMNS].to_numpy(dtype=float)
    updated = tail.loc[overlap, REQUIRED_PRICE_COLUMNS].to_numpy(dtype=float)
    relative_error = np.abs(updated - previous) / np.maximum(
        np.abs(previous), EPS
    )
    maximum_error = float(np.max(relative_error))
    if not np.all(relative_error <= relative_tolerance):
        raise ValueError(
            f"{symbol} incremental overlap changed adjusted OHLC values by up "
            f"to {maximum_error:.3%}; forcing a full-history provider refresh"
        )


def download_symbol(
    symbol: str,
    cfg: Config,
) -> tuple[pd.DataFrame, str]:
    cached = load_cached_frame(symbol, cfg)
    if cached is not None:
        origin = str(cached.attrs.get("provider", "legacy_cache"))
        return cached, f"cache:{origin}"

    stale_cache = (
        load_cached_frame(symbol, cfg, ignore_refresh=True)
        if cfg.refresh_data and cfg.allow_stale_cache_after_refresh_failure
        else None
    )

    errors: list[tuple[str, BaseException]] = []
    providers = [
        ("yahoo_direct", download_yahoo_frame),
        ("stooq_direct", download_stooq_frame),
    ]

    prior_cache = load_latest_prior_cache(symbol, cfg)
    if prior_cache is not None:
        prior_frame, prior_provider = prior_cache
        matching_downloader = dict(providers).get(prior_provider)
        if matching_downloader is not None:
            incremental_start = max(
                pd.Timestamp(cfg.start_date),
                pd.Timestamp(prior_frame.index.max()) - pd.Timedelta(days=10),
            ).date().isoformat()
            print(
                f"  {symbol}: extending verified {prior_provider} cache "
                f"from {incremental_start}",
                flush=True,
            )
            try:
                tail = matching_downloader(
                    symbol,
                    incremental_start,
                    cfg.end_date,
                    cfg.network_request_timeout_seconds,
                    cfg,
                )
                validate_incremental_overlap(prior_frame, tail, symbol)
                source_url = tail.attrs.get("source_url")
                combined = clean_ohlcv(
                    pd.concat([prior_frame, tail]).loc[
                        lambda value: ~value.index.duplicated(keep="last")
                    ],
                    symbol,
                )
                combined.attrs.update({
                    "source_url": source_url,
                    "incremental_base_cache_path": prior_frame.attrs.get(
                        "cache_path"
                    ),
                    "incremental_base_sha256": prior_frame.attrs.get(
                        "csv_sha256"
                    ),
                    "incremental_fetch_start_date": incremental_start,
                })
                save_cached_frame(combined, symbol, prior_provider, cfg)
                print(
                    f"  {symbol}: extended cache to {len(combined):,} rows",
                    flush=True,
                )
                return combined, prior_provider
            except (
                DataDownloadError, ValueError, KeyError, json.JSONDecodeError
            ) as exc:
                message = (
                    f"incremental_{prior_provider}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append((message, exc))
                print(
                    f"  {symbol}: {message}; retrying a full provider fetch",
                    flush=True,
                )

    for provider, downloader in providers:
        print(
            f"  {symbol}: trying {provider} "
            f"(timeout={cfg.network_request_timeout_seconds}s)...",
            flush=True,
        )
        try:
            frame = downloader(
                symbol,
                cfg.start_date,
                cfg.end_date,
                cfg.network_request_timeout_seconds,
                cfg,
            )
            save_cached_frame(frame, symbol, provider, cfg)
            print(
                f"  {symbol}: {len(frame):,} rows via {provider}",
                flush=True,
            )
            return frame, provider
        except (DataDownloadError, ValueError, KeyError, json.JSONDecodeError) as exc:
            message = (
                f"{provider}: {type(exc).__name__}: {exc}"
            )
            errors.append((message, exc))
            print(f"  {symbol}: {message}", flush=True)

    if stale_cache is not None:
        print(
            f"  {symbol}: refresh failed; using the prior verified cache. "
            "The manifest records this as stale-cache fallback.",
            flush=True,
        )
        return stale_cache, f"stale_cache:{stale_cache.attrs.get('provider', 'unknown')}"
    if errors and all(isinstance(error, RateLimitError) for _, error in errors):
        raise errors[0][1]
    raise DataDownloadError(" | ".join(message for message, _ in errors))


def build_manifest(
    frames: dict[str, pd.DataFrame],
    providers: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for symbol, frame in frames.items():
        provider_label = providers.get(symbol, "unknown")
        if provider_label.startswith("stale_cache:"):
            transport = "stale_cache"
        elif provider_label.startswith("cache:"):
            transport = "cache"
        elif provider_label == "synthetic_requested":
            transport = "synthetic"
        else:
            transport = "download"
        source_kind = frame.attrs.get(
            "source_kind",
            "synthetic" if transport == "synthetic" else "real_market",
        )
        rows.append({
            "symbol": symbol,
            "source_kind": source_kind,
            "provider": frame.attrs.get("provider", provider_label),
            "transport": transport,
            "provider_label": provider_label,
            "rows": len(frame),
            "first_date": frame.index.min(),
            "last_date": frame.index.max(),
            "missing_volume_rows": int(
                frame["VolumeMissing"].sum()
            ),
            "downloaded_at_utc": frame.attrs.get("downloaded_at_utc"),
            "source_url": frame.attrs.get("source_url"),
            "cache_path": frame.attrs.get("cache_path"),
            "csv_sha256": frame.attrs.get("csv_sha256"),
            "requested_start_date": frame.attrs.get("requested_start_date"),
            "requested_end_date": frame.attrs.get("requested_end_date"),
            "incremental_base_cache_path": frame.attrs.get(
                "incremental_base_cache_path"
            ),
            "incremental_base_sha256": frame.attrs.get(
                "incremental_base_sha256"
            ),
            "incremental_fetch_start_date": frame.attrs.get(
                "incremental_fetch_start_date"
            ),
        })
    return pd.DataFrame(rows)


def validate_research_data_manifest(
    manifest: pd.DataFrame,
    cfg: Config,
) -> None:
    required_columns = {
        "symbol", "source_kind", "provider", "transport", "rows",
        "first_date", "last_date", "downloaded_at_utc", "source_url",
        "csv_sha256", "requested_start_date", "requested_end_date",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Research data manifest is missing {sorted(missing_columns)}."
        )
    expected_symbols = set(cfg.target_tickers)
    if cfg.include_market_context:
        expected_symbols.update(cfg.market_symbols)
    observed_symbols = set(manifest["symbol"].astype(str))
    if observed_symbols != expected_symbols or manifest["symbol"].duplicated().any():
        raise ValueError(
            "Research manifest must contain every configured symbol exactly once."
        )
    if not manifest["source_kind"].eq("real_market").all():
        raise ValueError("Every research manifest row must be real_market data.")
    if not manifest["provider"].isin(ALLOWED_REAL_DATA_PROVIDERS).all():
        raise ValueError("Research manifest contains an unrecognized provider.")
    if not manifest["transport"].isin(
        ["download", "cache", "stale_cache"]
    ).all():
        raise ValueError("Research manifest contains an invalid transport.")
    if not manifest["requested_start_date"].eq(cfg.start_date).all():
        raise ValueError("Research manifest start dates do not match the config.")
    if not manifest["requested_end_date"].eq(cfg.end_date).all():
        raise ValueError("Research manifest end dates do not match the config.")
    downloaded = pd.to_datetime(
        manifest["downloaded_at_utc"], errors="coerce", utc=True
    )
    if downloaded.isna().any():
        raise ValueError("Research manifest has a missing download timestamp.")
    if not manifest["csv_sha256"].astype(str).map(
        lambda value: re.fullmatch(r"[0-9a-f]{64}", value) is not None
    ).all():
        raise ValueError("Research manifest has an invalid content checksum.")
    for row in manifest.to_dict("records"):
        provider = str(row["provider"])
        parsed = urllib.parse.urlparse(str(row["source_url"]))
        if (
            parsed.scheme != "https"
            or parsed.hostname not in PROVIDER_HOSTS[provider]
        ):
            raise ValueError(
                f"Manifest source URL does not match provider for {row['symbol']}."
            )
        last = pd.Timestamp(row["last_date"])
        if (pd.Timestamp(cfg.end_date) - last).days > MAXIMUM_CACHE_TAIL_GAP_DAYS:
            raise ValueError(
                f"Manifest tail coverage is incomplete for {row['symbol']}."
            )


def load_all_frames(
    cfg: Config,
) -> tuple[
    dict[str, pd.DataFrame],
    tuple[str, ...],
    str,
    pd.DataFrame,
]:
    if not cfg.use_real_market_data:
        print(
            "Real market-data requests are disabled; "
            "using synthetic pipeline-test data immediately.",
            flush=True,
        )
        frames = synthetic_frames(cfg)
        providers = {
            symbol: "synthetic_requested"
            for symbol in frames
        }
        return (
            frames,
            tuple(
                cfg.market_symbols
                if cfg.include_market_context else ()
            ),
            "synthetic_requested",
            build_manifest(frames, providers),
        )

    # Cached target data can be used without making a network request.
    target_cache_complete = all(
        cache_metadata_path_for(symbol, cfg).exists()
        and not cfg.refresh_data
        for symbol in cfg.target_tickers
    )

    if not target_cache_complete:
        connectivity, reason = network_preflight(cfg)
        if connectivity != "reachable":
            if connectivity == "rate_limited":
                raise RateLimitError(
                    "market-data preflight",
                    cfg.download_attempts,
                    None,
                ) from DataDownloadError(reason)
            if connectivity == "network_unavailable":
                raise NetworkUnavailableError(
                    "Market-data endpoints could not be reached. Cached data "
                    f"remain usable. Preflight result: {reason}"
                )
            raise DataDownloadError(
                "Market-data providers were reachable but did not return a "
                f"usable preflight response: {reason}"
            )
    else:
        print(
            "Target caches are available; skipping connectivity preflight.",
            flush=True,
        )

    frames = {}
    providers = {}

    try:
        print("Loading target assets...", flush=True)
        for symbol in cfg.target_tickers:
            frame, provider = download_symbol(symbol, cfg)
            frames[symbol] = frame
            providers[symbol] = provider
            if not provider.startswith("cache"):
                time.sleep(cfg.provider_request_spacing_seconds)
    except (RateLimitError, NetworkUnavailableError):
        # Preserve the operational category for callers and automation.
        raise
    except (DataDownloadError, ValueError) as exc:
        raise DataDownloadError(
            "Research data loading failed; synthetic substitution is disabled. "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    available_market_symbols = []
    if cfg.include_market_context:
        print("Loading market context...", flush=True)
        for symbol in cfg.market_symbols:
            try:
                frame, provider = download_symbol(symbol, cfg)
                frames[symbol] = frame
                providers[symbol] = provider
                available_market_symbols.append(symbol)
                if not provider.startswith("cache"):
                    time.sleep(cfg.provider_request_spacing_seconds)
            except (RateLimitError, NetworkUnavailableError):
                raise
            except (DataDownloadError, ValueError) as exc:
                raise DataDownloadError(
                    "Research market-context loading failed; the prespecified "
                    f"feature universe cannot change silently. {symbol}: {exc}"
                ) from exc

    source_name = "real_" + "+".join(
        sorted(set(providers.values()))
    )
    return (
        frames,
        tuple(available_market_symbols),
        source_name,
        build_manifest(frames, providers),
    )

# %% [markdown cell 11]
# ## Feature engineering, forward-only alignment, and dataset construction

# %% [cell 12]
STATISTIC_NAMES = [
    "annualised_mean_return",
    "log_annualised_volatility",
    "mean_absolute_return",
    "downside_annualised_volatility",
    "upside_annualised_volatility",
    "maximum_absolute_return",
    "last_return",
    "cumulative_return",
    "lag1_return_autocorrelation",
    "skewness",
    "excess_kurtosis",
    "volatility_of_volatility",
    "squared_return_trend",
    "log_realised_volatility_5",
    "log_realised_volatility_10",
    "log_realised_volatility_20",
    "log_realised_volatility_60",
    "volatility_ratio_5_20",
    "volatility_ratio_20_60",
    "absolute_return_autocorrelation",
    "squared_return_autocorrelation",
    "large_return_fraction",
    "maximum_drawdown",
    "current_return_streak",
]

OHLCV_FEATURE_NAMES = [
    "mean_log_intraday_range",
    "last_log_intraday_range",
    "log_parkinson_volatility_5",
    "log_parkinson_volatility_20",
    "log_parkinson_volatility_60",
    "mean_absolute_overnight_gap",
    "last_overnight_gap",
    "last_log_relative_volume",
    "mean_log_relative_volume",
    "volume_volatility",
    "missing_volume_fraction",
]


def safe_autocorrelation(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0

    left = values[:-1]
    right = values[1:]
    if np.std(left) < EPS or np.std(right) < EPS:
        return 0.0

    result = np.corrcoef(left, right)[0, 1]
    return float(result) if np.isfinite(result) else 0.0


def recent_volatility(
    values: np.ndarray,
    length: int,
    annualisation: float,
) -> float:
    selected = values[-min(length, len(values)):]
    return float(
        np.sqrt(annualisation * np.mean(selected**2))
    )


def volatility_of_volatility(
    values: np.ndarray,
    subwindow: int = 5,
) -> float:
    rolling_mean_square = np.convolve(
        values**2,
        np.ones(subwindow) / subwindow,
        mode="valid",
    )
    return float(
        np.std(np.sqrt(np.maximum(rolling_mean_square, 0.0)))
    )


def maximum_drawdown_from_returns(
    values: np.ndarray,
) -> float:
    cumulative_prices = np.exp(np.cumsum(values))
    running_maximum = np.maximum.accumulate(cumulative_prices)
    return float(
        np.min(cumulative_prices / running_maximum - 1.0)
    )


def current_streak(values: np.ndarray) -> float:
    if len(values) == 0 or values[-1] == 0:
        return 0.0

    final_sign = np.sign(values[-1])
    length = 0
    for value in values[::-1]:
        if np.sign(value) == final_sign:
            length += 1
        else:
            break

    return float(length * final_sign)


def statistical_features(
    past: np.ndarray,
    annualisation: float,
) -> np.ndarray:
    mean_return = float(np.mean(past))
    sample_std = max(float(np.std(past, ddof=1)), EPS)
    annualised_volatility = (
        math.sqrt(annualisation) * sample_std
    )

    downside = np.minimum(past, 0.0)
    upside = np.maximum(past, 0.0)
    standardised = (past - mean_return) / sample_std

    time = np.arange(len(past), dtype=float)
    centred_time = time - time.mean()
    denominator = max(
        float(np.sum(centred_time**2)),
        EPS,
    )
    squared_return_trend = float(
        np.sum(
            centred_time
            * (past**2 - np.mean(past**2))
        )
        / denominator
    )

    vol5 = recent_volatility(past, 5, annualisation)
    vol10 = recent_volatility(past, 10, annualisation)
    vol20 = recent_volatility(past, 20, annualisation)
    vol60 = recent_volatility(past, 60, annualisation)

    large_return_fraction = float(
        np.mean(
            np.abs(past - mean_return)
            > 2.0 * sample_std
        )
    )

    return np.array([
        annualisation * mean_return,
        math.log(annualised_volatility + EPS),
        np.mean(np.abs(past)),
        math.sqrt(
            annualisation * np.mean(downside**2)
        ),
        math.sqrt(
            annualisation * np.mean(upside**2)
        ),
        np.max(np.abs(past)),
        past[-1],
        np.sum(past),
        safe_autocorrelation(past),
        np.mean(standardised**3),
        np.mean(standardised**4) - 3.0,
        volatility_of_volatility(past),
        squared_return_trend,
        math.log(vol5 + EPS),
        math.log(vol10 + EPS),
        math.log(vol20 + EPS),
        math.log(vol60 + EPS),
        vol5 / (vol20 + EPS),
        vol20 / (vol60 + EPS),
        safe_autocorrelation(np.abs(past)),
        safe_autocorrelation(past**2),
        large_return_fraction,
        maximum_drawdown_from_returns(past),
        current_streak(past),
    ], dtype=np.float64)


def ohlcv_features(
    past_frame: pd.DataFrame,
    annualisation: float,
) -> np.ndarray:
    open_price = past_frame["Open"].to_numpy(dtype=float)
    high = past_frame["High"].to_numpy(dtype=float)
    low = past_frame["Low"].to_numpy(dtype=float)
    close = past_frame["Close"].to_numpy(dtype=float)
    volume = past_frame["Volume"].to_numpy(dtype=float)
    volume_missing = past_frame[
        "VolumeMissing"
    ].to_numpy(dtype=float)

    log_range = np.log(
        np.maximum(high, EPS) / np.maximum(low, EPS)
    )
    previous_close = np.concatenate(
        [[close[0]], close[:-1]]
    )
    overnight_gap = np.log(
        np.maximum(open_price, EPS)
        / np.maximum(previous_close, EPS)
    )

    def parkinson(length: int) -> float:
        selected = log_range[-min(length, len(log_range)):]
        variance = (
            np.mean(selected**2)
            / (4.0 * math.log(2.0))
        )
        return math.sqrt(
            annualisation * max(variance, EPS)
        )

    volume_series = pd.Series(volume)
    rolling_volume = volume_series.rolling(
        20,
        min_periods=5,
    ).mean().to_numpy()

    positive_volume = volume[volume > 0]
    fallback_volume = (
        float(np.median(positive_volume))
        if len(positive_volume) else 1.0
    )
    rolling_volume = np.where(
        np.isfinite(rolling_volume)
        & (rolling_volume > 0),
        rolling_volume,
        fallback_volume,
    )
    log_relative_volume = np.log(
        (volume + 1.0) / (rolling_volume + 1.0)
    )

    return np.array([
        np.mean(log_range),
        log_range[-1],
        math.log(parkinson(5) + EPS),
        math.log(parkinson(20) + EPS),
        math.log(parkinson(60) + EPS),
        np.mean(np.abs(overnight_gap)),
        overnight_gap[-1],
        log_relative_volume[-1],
        np.mean(log_relative_volume),
        np.std(log_relative_volume),
        np.mean(volume_missing),
    ], dtype=np.float64)


def market_feature_names(
    symbols: tuple[str, ...],
) -> list[str]:
    names = []
    for symbol in symbols:
        safe_symbol = symbol.lower().replace("^", "")
        if symbol == "^VIX":
            names.extend([
                f"{safe_symbol}_level",
                f"{safe_symbol}_change_5",
                f"{safe_symbol}_mean_20",
                f"{safe_symbol}_volatility_20",
            ])
        else:
            names.extend([
                f"{safe_symbol}_log_volatility_5",
                f"{safe_symbol}_log_volatility_20",
                f"{safe_symbol}_log_volatility_60",
                f"{safe_symbol}_return_5",
                f"{safe_symbol}_return_20",
                f"{safe_symbol}_downside_volatility_20",
            ])
    return names


def align_market_data_forward_only(
    target_dates: pd.DatetimeIndex,
    frames: dict[str, pd.DataFrame],
    symbols: tuple[str, ...],
    forward_fill_limit: int,
) -> dict[str, dict[str, np.ndarray]]:
    aligned = {}

    for symbol in symbols:
        raw_close = frames[symbol]["Close"].reindex(target_dates)
        close = raw_close.ffill(limit=forward_fill_limit)

        # No backward fill is permitted.
        returns = np.log(close).diff()

        aligned[symbol] = {
            "close": close.to_numpy(dtype=float),
            "returns": returns.to_numpy(dtype=float),
            "originally_missing": raw_close.isna().to_numpy(),
        }

    return aligned


def market_features_or_none(
    aligned_market_data: dict[str, dict[str, np.ndarray]],
    end_index: int,
    symbols: tuple[str, ...],
    cfg: Config,
) -> np.ndarray | None:
    values = []

    for symbol in symbols:
        information = aligned_market_data[symbol]
        close = information["close"][
            end_index - cfg.window:end_index
        ]
        returns = information["returns"][
            end_index - cfg.window:end_index
        ]

        if (
            len(close) != cfg.window
            or len(returns) != cfg.window
            or not np.all(np.isfinite(close))
            or not np.all(np.isfinite(returns))
        ):
            return None

        if symbol == "^VIX":
            values.extend([
                close[-1],
                close[-1] / (close[-6] + EPS) - 1.0,
                np.mean(close[-20:]),
                np.std(close[-20:]),
            ])
        else:
            downside = np.minimum(returns[-20:], 0.0)
            values.extend([
                math.log(
                    recent_volatility(
                        returns,
                        5,
                        cfg.annualisation,
                    )
                    + EPS
                ),
                math.log(
                    recent_volatility(
                        returns,
                        20,
                        cfg.annualisation,
                    )
                    + EPS
                ),
                math.log(
                    recent_volatility(
                        returns,
                        60,
                        cfg.annualisation,
                    )
                    + EPS
                ),
                np.sum(returns[-5:]),
                np.sum(returns[-20:]),
                math.sqrt(
                    cfg.annualisation
                    * np.mean(downside**2)
                ),
            ])

    return np.asarray(values, dtype=np.float64)


def cumulative_channel_path(channels: np.ndarray) -> np.ndarray:
    channels = np.asarray(channels, dtype=np.float64)
    if channels.ndim == 1:
        channels = channels[:, None]
    cumulative = np.vstack([
        np.zeros((1, channels.shape[1]), dtype=np.float64),
        np.cumsum(channels, axis=0),
    ])
    time_values = np.linspace(0.0, 1.0, len(channels) + 1)[:, None]
    return np.concatenate([time_values, cumulative], axis=1)


def make_shape_signature_path(past: np.ndarray) -> np.ndarray:
    mean = np.mean(past)
    scale = max(float(np.std(past, ddof=1)), EPS)
    standardised = (past - mean) / scale
    channels = np.column_stack([
        standardised / math.sqrt(len(past)),
        standardised**2 / len(past),
    ])
    return cumulative_channel_path(channels)


def make_amplitude_signature_path(
    past: np.ndarray,
    return_scale: float,
) -> np.ndarray:
    scaled = past / return_scale
    channels = np.column_stack([
        scaled / math.sqrt(len(past)),
        scaled**2 / len(past),
    ])
    return cumulative_channel_path(channels)


def make_ohlc_signature_path(
    past: np.ndarray,
    past_frame: pd.DataFrame,
    return_scale: float,
) -> np.ndarray:
    open_price = past_frame["Open"].to_numpy(dtype=float)
    high = past_frame["High"].to_numpy(dtype=float)
    low = past_frame["Low"].to_numpy(dtype=float)
    close = past_frame["Close"].to_numpy(dtype=float)
    intraday_return = np.log(
        np.maximum(close, EPS) / np.maximum(open_price, EPS)
    )
    # Total close-to-close return minus the same-day intraday return equals
    # the overnight component and does not require a pre-window close.
    overnight_return = past - intraday_return
    range_variance = (
        np.log(np.maximum(high, EPS) / np.maximum(low, EPS)) ** 2
        / (4.0 * math.log(2.0))
    )
    channels = np.column_stack([
        (past / return_scale) / math.sqrt(len(past)),
        (range_variance / return_scale**2) / len(past),
        (overnight_return / return_scale) ** 2 / len(past),
    ])
    return cumulative_channel_path(channels)


def make_joint_market_signature_path(
    asset_returns: np.ndarray,
    spy_returns: np.ndarray,
    vix_changes: np.ndarray,
    return_scale: float,
    vix_change_scale: float,
) -> np.ndarray:
    length = len(asset_returns)
    channels = np.column_stack([
        (asset_returns / return_scale) / math.sqrt(length),
        (spy_returns / return_scale) / math.sqrt(length),
        (vix_changes / vix_change_scale) / math.sqrt(length),
    ])
    return cumulative_channel_path(channels)


def make_lead_lag_signature_path(
    past: np.ndarray,
    return_scale: float,
) -> np.ndarray:
    stream = np.concatenate([
        [0.0],
        np.cumsum(past / return_scale) / math.sqrt(len(past)),
    ])
    path = np.empty((2 * len(past) + 1, 2), dtype=np.float64)
    path[0] = (stream[0], stream[0])
    path[1::2, 0] = stream[1:]
    path[1::2, 1] = stream[:-1]
    path[2::2, 0] = stream[1:]
    path[2::2, 1] = stream[1:]
    return path


def signature_block_specs(cfg: Config) -> tuple[SignatureBlockSpec, ...]:
    if not cfg.include_signature_features:
        return ()
    specs = []
    longest_window = max(cfg.signature_windows)
    for window in cfg.signature_windows:
        shape_depth = (
            cfg.signature_depth
            if window == longest_window
            else cfg.short_signature_depth
        )
        if cfg.include_shape_signatures:
            specs.append(
                SignatureBlockSpec("shape", window, 3, shape_depth)
            )
        if cfg.include_amplitude_signatures:
            specs.append(SignatureBlockSpec(
                "amplitude", window, 3, cfg.amplitude_signature_depth
            ))
    for window in cfg.secondary_signature_windows:
        if cfg.include_ohlc_signatures:
            specs.append(SignatureBlockSpec(
                "ohlc", window, 4, cfg.ohlc_signature_depth
            ))
        if cfg.include_joint_market_signatures:
            specs.append(SignatureBlockSpec(
                "joint_market", window, 4, cfg.joint_market_signature_depth
            ))
        if cfg.include_lead_lag_signatures:
            specs.append(SignatureBlockSpec(
                "lead_lag", window, 2, cfg.lead_lag_signature_depth
            ))
    return tuple(specs)


def all_signature_feature_names(cfg: Config) -> list[str]:
    names = []
    for spec in signature_block_specs(cfg):
        names.extend(signature_feature_names(
            spec.prefix,
            spec.path_dimension,
            spec.depth,
        ))
    return names


def signature_features_or_none(
    past: np.ndarray,
    past_frame: pd.DataFrame,
    aligned_market_data: dict[str, dict[str, np.ndarray]],
    end_index: int,
    cfg: Config,
) -> np.ndarray | None:
    pieces = []
    spy_symbol, vix_symbol = cfg.joint_signature_symbols
    for spec in signature_block_specs(cfg):
        selected_returns = past[-spec.window:]
        if spec.family == "shape":
            path = make_shape_signature_path(selected_returns)
        elif spec.family == "amplitude":
            path = make_amplitude_signature_path(
                selected_returns, cfg.signature_return_scale
            )
        elif spec.family == "ohlc":
            path = make_ohlc_signature_path(
                selected_returns,
                past_frame.iloc[-spec.window:],
                cfg.signature_return_scale,
            )
        elif spec.family == "joint_market":
            spy_returns = aligned_market_data[spy_symbol]["returns"][
                end_index - spec.window:end_index
            ]
            vix_changes = aligned_market_data[vix_symbol]["returns"][
                end_index - spec.window:end_index
            ]
            if (
                len(spy_returns) != spec.window
                or len(vix_changes) != spec.window
                or not np.all(np.isfinite(spy_returns))
                or not np.all(np.isfinite(vix_changes))
            ):
                return None
            path = make_joint_market_signature_path(
                selected_returns,
                spy_returns,
                vix_changes,
                cfg.signature_return_scale,
                cfg.signature_vix_change_scale,
            )
        elif spec.family == "lead_lag":
            path = make_lead_lag_signature_path(
                selected_returns, cfg.signature_return_scale
            )
        else:
            raise RuntimeError(f"Unknown signature family: {spec.family}")

        values = truncated_signature(path, spec.depth)
        if not np.all(np.isfinite(values)):
            return None
        pieces.append(values)
    return (
        np.concatenate(pieces)
        if pieces
        else np.empty(0, dtype=np.float64)
    )


def realised_volatility(
    values: np.ndarray,
    annualisation: float,
) -> float:
    return float(
        np.sqrt(annualisation * np.mean(values**2))
    )


def ewma_volatility(
    values: np.ndarray,
    annualisation: float,
    decay: float,
) -> float:
    variance = max(float(np.var(values, ddof=1)), EPS)

    for value in values:
        variance = (
            decay * variance
            + (1.0 - decay) * float(value**2)
        )

    return math.sqrt(
        annualisation * max(variance, EPS)
    )


def feature_groups_for(
    feature_names: list[str],
) -> dict[str, tuple[str, ...]]:
    """Return overlapping, named masks for group and exact-feature ablations."""
    names = set(feature_names)

    def present(candidates: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in candidates if name in names)

    signature = tuple(name for name in feature_names if name.startswith("signature_"))
    signature_shape = tuple(
        name for name in signature if name.startswith("signature_shape_")
    )
    signature_amplitude = tuple(
        name for name in signature if name.startswith("signature_amplitude_")
    )
    signature_ohlc = tuple(
        name for name in signature if name.startswith("signature_ohlc_")
    )
    signature_joint_market = tuple(
        name for name in signature
        if name.startswith("signature_joint_market_")
    )
    signature_lead_lag = tuple(
        name for name in signature if name.startswith("signature_lead_lag_")
    )
    scalar_market = tuple(
        name for name in feature_names
        if name.startswith(("spy_", "qqq_", "vix_"))
    )
    market = tuple(dict.fromkeys(scalar_market + signature_joint_market))
    scalar_ohlcv = present(OHLCV_FEATURE_NAMES)
    ohlcv = tuple(dict.fromkeys(scalar_ohlcv + signature_ohlc))

    return {
        "signature": signature,
        "signature_shape": signature_shape,
        "signature_amplitude": signature_amplitude,
        "signature_ohlc": signature_ohlc,
        "signature_joint_market": signature_joint_market,
        "signature_lead_lag": signature_lead_lag,
        "signature_w10": tuple(name for name in signature if "_w10_" in name),
        "signature_w20": tuple(name for name in signature if "_w20_" in name),
        "signature_w60": tuple(name for name in signature if "_w60_" in name),
        "signature_l1": tuple(name for name in signature if "_L1_" in name),
        "signature_l2": tuple(name for name in signature if "_L2_" in name),
        "signature_l3": tuple(name for name in signature if "_L3_" in name),
        "return_statistics": present(STATISTIC_NAMES),
        "return_location_scale": present((
            "annualised_mean_return", "log_annualised_volatility",
            "mean_absolute_return", "downside_annualised_volatility",
            "upside_annualised_volatility",
        )),
        "return_tail_shape": present((
            "maximum_absolute_return", "skewness", "excess_kurtosis",
            "large_return_fraction", "maximum_drawdown",
        )),
        "return_dynamics": present((
            "last_return", "cumulative_return", "lag1_return_autocorrelation",
            "volatility_of_volatility", "squared_return_trend",
            "absolute_return_autocorrelation", "squared_return_autocorrelation",
            "current_return_streak",
        )),
        "realised_volatility_horizons": present((
            "log_realised_volatility_5", "log_realised_volatility_10",
            "log_realised_volatility_20", "log_realised_volatility_60",
            "volatility_ratio_5_20", "volatility_ratio_20_60",
        )),
        "ohlcv": ohlcv,
        "ohlcv_range": present((
            "mean_log_intraday_range", "last_log_intraday_range",
            "log_parkinson_volatility_5", "log_parkinson_volatility_20",
            "log_parkinson_volatility_60",
        )),
        "ohlcv_overnight": present((
            "mean_absolute_overnight_gap", "last_overnight_gap",
        )),
        "ohlcv_volume": present((
            "last_log_relative_volume", "mean_log_relative_volume",
            "volume_volatility", "missing_volume_fraction",
        )),
        "market": market,
        "market_spy": tuple(name for name in scalar_market if name.startswith("spy_")) + signature_joint_market,
        "market_qqq": tuple(name for name in scalar_market if name.startswith("qqq_")),
        "market_vix": tuple(name for name in scalar_market if name.startswith("vix_")) + signature_joint_market,
    }


def dataset_construction_spec(cfg: Config) -> dict[str, object]:
    """Fields that change source rows or canonical feature values/width."""
    return {
        "target_tickers": tuple(cfg.target_tickers),
        "window": cfg.window,
        "horizon": cfg.horizon,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "market_symbols": tuple(cfg.market_symbols),
        "include_market_context": cfg.include_market_context,
        "market_forward_fill_limit": cfg.market_forward_fill_limit,
        "signature_windows": tuple(cfg.signature_windows),
        "secondary_signature_windows": tuple(cfg.secondary_signature_windows),
        "signature_depth": cfg.signature_depth,
        "short_signature_depth": cfg.short_signature_depth,
        "amplitude_signature_depth": cfg.amplitude_signature_depth,
        "ohlc_signature_depth": cfg.ohlc_signature_depth,
        "joint_market_signature_depth": cfg.joint_market_signature_depth,
        "lead_lag_signature_depth": cfg.lead_lag_signature_depth,
        "signature_return_scale": cfg.signature_return_scale,
        "signature_vix_change_scale": cfg.signature_vix_change_scale,
        "joint_signature_symbols": tuple(cfg.joint_signature_symbols),
        "annualisation": cfg.annualisation,
        "ewma_lambda": cfg.ewma_lambda,
        "synthetic_seed": cfg.seed if cfg.experiment_mode == "pipeline_test" else None,
    }


def build_market_dataset(cfg: Config) -> MarketDataset:
    (
        frames,
        available_market_symbols,
        data_source,
        data_manifest,
    ) = load_all_frames(cfg)

    data_manifest = data_manifest.copy()
    data_manifest["role"] = data_manifest["symbol"].map(
        lambda symbol: (
            "training_ticker" if symbol in cfg.training_tickers
            else "unseen_test_ticker" if symbol in cfg.unseen_test_tickers
            else "market_context"
        )
    )
    data_manifest["sector"] = data_manifest["symbol"].map(TICKER_SECTORS).fillna(
        "market_context"
    )
    provenance_valid = False

    if (
        cfg.experiment_mode == "research"
        and data_source.startswith("synthetic")
    ):
        raise RuntimeError(
            "A final real experiment may not use synthetic data."
        )
    if (
        cfg.experiment_mode == "research"
        and cfg.include_market_context
        and set(available_market_symbols) != set(cfg.market_symbols)
    ):
        raise RuntimeError(
            "Research mode requires the complete prespecified market-context "
            f"universe; missing {sorted(set(cfg.market_symbols) - set(available_market_symbols))}."
        )
    if cfg.experiment_mode == "research":
        validate_research_data_manifest(data_manifest, cfg)
        provenance_valid = True
    data_manifest["provenance_valid"] = provenance_valid

    feature_rows = []
    target_rows = []
    metadata_rows = []
    skipped_rows = []

    required_joint_symbols = set(cfg.joint_signature_symbols)
    if (
        cfg.include_signature_features
        and cfg.include_joint_market_signatures
        and not required_joint_symbols <= set(available_market_symbols)
    ):
        raise RuntimeError(
            "Joint market signatures require all configured symbols; missing: "
            f"{sorted(required_joint_symbols - set(available_market_symbols))}"
        )

    feature_names = (
        all_signature_feature_names(cfg)
        + STATISTIC_NAMES
        + OHLCV_FEATURE_NAMES
        + market_feature_names(available_market_symbols)
    )

    print("\nBuilding leakage-controlled samples...")

    for ticker_id, ticker in enumerate(cfg.target_tickers):
        frame = frames[ticker].copy()

        close_returns = np.log(
            frame["Close"]
        ).diff().dropna()
        dates = pd.DatetimeIndex(close_returns.index)
        returns = close_returns.to_numpy(dtype=np.float64)
        aligned_asset_frame = frame.reindex(dates)

        aligned_markets = align_market_data_forward_only(
            dates,
            frames,
            available_market_symbols,
            cfg.market_forward_fill_limit,
        )

        accepted = 0
        skipped = 0

        for index in range(
            cfg.window,
            len(returns) - cfg.horizon + 1,
        ):
            past = returns[index - cfg.window:index]
            future = returns[index:index + cfg.horizon]
            past_frame = aligned_asset_frame.iloc[
                index - cfg.window:index
            ]

            if past_frame[REQUIRED_PRICE_COLUMNS].isna().any().any():
                skipped_rows.append({
                    "ticker": ticker,
                    "origin_date": dates[index - 1],
                    "reason": "missing_target_ohlc",
                })
                skipped += 1
                continue

            market_values = market_features_or_none(
                aligned_markets,
                index,
                available_market_symbols,
                cfg,
            )
            if market_values is None:
                skipped_rows.append({
                    "ticker": ticker,
                    "origin_date": dates[index - 1],
                    "reason": "insufficient_forward_only_market_context",
                })
                skipped += 1
                continue

            signature_values = signature_features_or_none(
                past,
                past_frame,
                aligned_markets,
                index,
                cfg,
            )
            if signature_values is None:
                skipped_rows.append({
                    "ticker": ticker,
                    "origin_date": dates[index - 1],
                    "reason": "insufficient_signature_context",
                })
                skipped += 1
                continue

            feature = np.concatenate([
                signature_values,
                statistical_features(
                    past,
                    cfg.annualisation,
                ),
                ohlcv_features(
                    past_frame,
                    cfg.annualisation,
                ),
                market_values,
            ])

            actual_volatility = realised_volatility(
                future,
                cfg.annualisation,
            )
            rolling_baseline = recent_volatility(
                past, 20, cfg.annualisation
            )

            feature_rows.append(feature)
            target_rows.append(
                math.log(actual_volatility + EPS)
            )
            metadata_rows.append({
                "ticker_id": ticker_id,
                "ticker": ticker,
                "ticker_role": (
                    "training" if ticker in cfg.training_tickers
                    else "unseen_test"
                ),
                "origin_date": dates[index - 1],
                "feature_start_date": dates[index - cfg.window],
                "feature_end_date": dates[index - 1],
                "forecast_issued_at": f"{dates[index - 1].date()} after_market_close",
                "target_start_date": dates[index],
                "target_end_date": dates[
                    index + cfg.horizon - 1
                ],
                "horizon": cfg.horizon,
                "actual_vol": actual_volatility,
                "rolling_vol_baseline": rolling_baseline,
                "future_to_recent_vol_ratio": (
                    actual_volatility / max(rolling_baseline, EPS)
                ),
                "sector": TICKER_SECTORS[ticker],
                "ewma_vol_baseline": ewma_volatility(
                    past,
                    cfg.annualisation,
                    cfg.ewma_lambda,
                ),
                "data_source": data_source,
                "data_provenance_valid": provenance_valid,
            })
            accepted += 1

        print(
            f"  {ticker}: accepted={accepted:,}, skipped={skipped:,}"
        )

    if not feature_rows:
        raise RuntimeError("No samples survived the leakage controls.")

    unsorted_metadata = pd.DataFrame(metadata_rows)
    ordering = (
        unsorted_metadata.assign(
            _position=np.arange(len(unsorted_metadata))
        )
        .sort_values(["origin_date", "ticker"])["_position"]
        .to_numpy(dtype=int)
    )

    metadata = (
        unsorted_metadata
        .iloc[ordering]
        .reset_index(drop=True)
    )
    features = np.asarray(
        feature_rows,
        dtype=np.float32,
    )[ordering]
    targets = np.asarray(
        target_rows,
        dtype=np.float32,
    )[ordering]

    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("Canonical feature names are not unique.")
    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature mismatch: values={features.shape[1]}, "
            f"names={len(feature_names)}"
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite feature values remain.")

    if not np.all(np.isfinite(targets)):
        raise ValueError("Non-finite target values remain.")

    skipped_samples = pd.DataFrame(
        skipped_rows,
        columns=["ticker", "origin_date", "reason"],
    )

    print(
        f"Dataset complete: {len(metadata):,} samples, "
        f"{features.shape[1]} features, source={data_source}"
    )

    return MarketDataset(
        features=features,
        targets_log_vol=targets,
        metadata=metadata,
        feature_names=feature_names,
        data_source=data_source,
        available_market_symbols=available_market_symbols,
        data_manifest=data_manifest,
        skipped_samples=skipped_samples,
        construction_config=dataset_construction_spec(cfg),
    )

# %% [markdown cell 13]
# ## Purged outer splits, inner development folds, and transferable regimes
# 
# The outer protocol reserves training, validation, calibration, and a final 15% time section. Feature or objective selection uses three expanding folds entirely inside the outer training section. The untouched outer validation selects checkpoints, calibration alone tunes intervals/temperature, and the final test is opened once.
# 
# Regimes are Contracting, Stable, and Expanding relative to each row's past-only 20-session volatility. Their symmetric boundaries are fitted from training targets only; held-out tickers receive pooled training boundaries.

# %% [cell 14]
def cutoff_from_fraction(
    unique_dates: np.ndarray,
    fraction: float,
) -> pd.Timestamp:
    position = int(len(unique_dates) * fraction) - 1
    position = max(0, min(position, len(unique_dates) - 1))
    return pd.Timestamp(unique_dates[position])


def resolve_active_feature_set(
    dataset: MarketDataset,
    cfg: Config,
) -> tuple[np.ndarray, list[str], list[str]]:
    feature_names = dataset.feature_names
    groups = feature_groups_for(feature_names)
    unknown_groups = sorted(set(cfg.disabled_feature_groups) - set(groups))
    if unknown_groups:
        raise ValueError(
            f"Unknown disabled feature groups: {unknown_groups}. "
            f"Valid groups: {sorted(groups)}"
        )

    disabled = set(cfg.disabled_feature_names)
    unknown_features = sorted(disabled - set(feature_names))
    if unknown_features:
        raise ValueError(
            f"Unknown disabled features: {unknown_features}. "
            "Use the exact names printed in the feature manifest."
        )

    if not cfg.include_signature_features:
        disabled.update(groups["signature"])
    if not cfg.include_shape_signatures:
        disabled.update(groups["signature_shape"])
    if not cfg.include_amplitude_signatures:
        disabled.update(groups["signature_amplitude"])
    if not cfg.include_ohlc_signatures:
        disabled.update(groups["signature_ohlc"])
    if not cfg.include_joint_market_signatures:
        disabled.update(groups["signature_joint_market"])
    if not cfg.include_lead_lag_signatures:
        disabled.update(groups["signature_lead_lag"])
    if not cfg.include_return_statistics:
        disabled.update(groups["return_statistics"])
    if not cfg.include_ohlcv_features:
        disabled.update(groups["ohlcv"])
    if not cfg.include_market_features:
        disabled.update(groups["market"])

    for group_name in cfg.disabled_feature_groups:
        disabled.update(groups[group_name])

    active_indices = np.array(
        [index for index, name in enumerate(feature_names) if name not in disabled],
        dtype=int,
    )
    active_names = [feature_names[index] for index in active_indices]
    dropped_names = [name for name in feature_names if name in disabled]

    if len(active_indices) == 0:
        raise ValueError("Every canonical feature was disabled.")

    print("\nModel-input feature manifest:")
    print(f"  Canonical engineered features: {len(feature_names)}")
    print(f"  Active engineered features:    {len(active_names)}")
    print(f"  Dropped engineered features:   {len(dropped_names)}")
    print(f"  Seen-ticker identity information enabled: {cfg.include_ticker_identity}")
    print("  Valid feature groups:", ", ".join(sorted(groups)))
    print("  Active feature names:")
    print("   ", ", ".join(active_names))
    if dropped_names:
        print("  Dropped feature names:")
        print("   ", ", ".join(dropped_names))

    return active_indices, active_names, dropped_names


def regime_score_values(
    dataset: MarketDataset,
    cfg: Config,
) -> np.ndarray:
    if cfg.regime_target_mode == "absolute_volatility":
        return dataset.targets_log_vol.astype(float)
    recent = dataset.metadata["rolling_vol_baseline"].to_numpy(dtype=float)
    return (
        dataset.targets_log_vol.astype(float)
        - np.log(np.maximum(recent, EPS))
    )


def regime_boundaries(values: np.ndarray, relative: bool) -> np.ndarray:
    if relative:
        half_width = float(np.quantile(np.abs(values), 1.0 / 3.0))
        return np.array([-half_width, half_width], dtype=np.float32)
    return np.quantile(values, [1 / 3, 2 / 3]).astype(np.float32)


def build_ticker_thresholds_and_labels(
    dataset: MarketDataset,
    train_indices: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = dataset.metadata
    scores = regime_score_values(dataset, cfg)
    relative = cfg.regime_target_mode == "relative_to_recent_volatility"
    pooled_thresholds = regime_boundaries(scores[train_indices], relative)

    thresholds = np.tile(
        pooled_thresholds[None, :],
        (len(cfg.target_tickers), 1),
    ).astype(np.float32)
    sources = np.full(
        len(cfg.target_tickers),
        f"pooled_training_tickers::{cfg.regime_target_mode}",
        dtype=object,
    )
    labels = np.zeros(len(metadata), dtype=np.int64)

    for ticker_id, ticker in enumerate(cfg.target_tickers):
        ticker_train_indices = train_indices[
            metadata.iloc[train_indices]["ticker_id"].to_numpy(dtype=int)
            == ticker_id
        ]

        if ticker in cfg.training_tickers and len(ticker_train_indices) < 30:
            raise ValueError(f"Too few training samples for {ticker}.")

        if (
            cfg.regime_threshold_mode == "per_training_ticker"
            and ticker in cfg.training_tickers
        ):
            thresholds[ticker_id] = regime_boundaries(
                scores[ticker_train_indices], relative
            )
            sources[ticker_id] = (
                f"{ticker}_training_only::{cfg.regime_target_mode}"
            )

        ticker_indices = np.flatnonzero(
            metadata["ticker_id"].to_numpy(dtype=int) == ticker_id
        )
        labels[ticker_indices] = np.digitize(
            scores[ticker_indices],
            thresholds[ticker_id],
        )

    return thresholds, labels, sources


def build_training_regime_statistics(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    train_indices: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(
        labels[train_indices],
        minlength=cfg.regimes,
    ).astype(float)
    proportions = counts / counts.sum()

    # Square-root inverse frequency is intentionally gentler than full inverse
    # frequency; explicit clipping prevents rare extreme regimes from
    # dominating the classifier and being overpredicted.
    class_weights = np.sqrt(
        counts.sum() / (cfg.regimes * np.maximum(counts, 1.0))
    )
    class_weights[1] *= cfg.medium_class_multiplier
    class_weights /= class_weights.mean()
    class_weights = np.clip(
        class_weights,
        cfg.minimum_regime_class_weight,
        cfg.maximum_regime_class_weight,
    )

    training_set = set(train_indices.tolist())
    pooled_transition = np.ones((cfg.regimes, cfg.regimes), dtype=float)
    pooled_initial = np.ones(cfg.regimes, dtype=float)
    ordered_training_indices: dict[int, np.ndarray] = {}

    for ticker_id, ticker in enumerate(cfg.target_tickers):
        if ticker not in cfg.training_tickers:
            continue
        ordered = (
            metadata[metadata["ticker_id"] == ticker_id]
            .sort_values("origin_date")
            .index.to_numpy(dtype=int)
        )
        ordered = np.array(
            [index for index in ordered if index in training_set],
            dtype=int,
        )
        ordered_training_indices[ticker_id] = ordered
        if len(ordered) == 0:
            continue
        pooled_initial[labels[ordered[0]]] += 1.0
        for previous, current in zip(labels[ordered[:-1]], labels[ordered[1:]]):
            pooled_transition[previous, current] += 1.0

    pooled_transition /= pooled_transition.sum(axis=1, keepdims=True)
    pooled_initial /= pooled_initial.sum()

    transitions = np.tile(
        pooled_transition[None, :, :],
        (len(cfg.target_tickers), 1, 1),
    )
    initial = np.tile(
        pooled_initial[None, :],
        (len(cfg.target_tickers), 1),
    )

    if cfg.ticker_specific_regime_smoothing:
        for ticker_id, ordered in ordered_training_indices.items():
            local_transition = np.ones((cfg.regimes, cfg.regimes), dtype=float)
            local_initial = np.ones(cfg.regimes, dtype=float)
            if len(ordered):
                local_initial[labels[ordered[0]]] += 1.0
                for previous, current in zip(
                    labels[ordered[:-1]], labels[ordered[1:]]
                ):
                    local_transition[previous, current] += 1.0
            local_transition /= local_transition.sum(axis=1, keepdims=True)
            local_initial /= local_initial.sum()
            transitions[ticker_id] = local_transition
            initial[ticker_id] = local_initial

    return (
        class_weights.astype(np.float32),
        proportions.astype(np.float32),
        transitions.astype(np.float32),
        initial.astype(np.float32),
    )


def assert_split_integrity(
    dataset: MarketDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    calibration_indices: np.ndarray,
    seen_ticker_unseen_day_indices: np.ndarray,
    unseen_ticker_unseen_day_indices: np.ndarray,
    train_cutoff: pd.Timestamp,
    validation_cutoff: pd.Timestamp,
    calibration_cutoff: pd.Timestamp,
    cfg: Config,
) -> pd.DataFrame:
    metadata = dataset.metadata
    sections = {
        "train": train_indices,
        "validation": validation_indices,
        "calibration": calibration_indices,
        "seen_ticker_unseen_day": seen_ticker_unseen_day_indices,
        "unseen_ticker_unseen_day": unseen_ticker_unseen_day_indices,
    }
    all_indices = np.concatenate(list(sections.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise AssertionError("Split index sets overlap.")

    checks: list[dict[str, object]] = []

    def check(description: str, passed: bool, observed: object, boundary: object) -> None:
        checks.append({
            "check": description,
            "passed": bool(passed),
            "observed": observed,
            "boundary": boundary,
        })
        if not passed:
            raise AssertionError(description)

    fit_indices = np.concatenate([
        train_indices, validation_indices, calibration_indices,
    ])
    fit_tickers = set(metadata.iloc[fit_indices]["ticker"])
    feature_end = pd.to_datetime(metadata["feature_end_date"])
    origin_all = pd.to_datetime(metadata["origin_date"])
    target_start_all = pd.to_datetime(metadata["target_start_date"])
    target_end_all = pd.to_datetime(metadata["target_end_date"])
    check(
        "every feature timestamp ends by its forecast origin",
        bool((feature_end <= origin_all).all()),
        int((feature_end > origin_all).sum()),
        0,
    )
    check(
        "every forecast origin precedes its target window",
        bool((origin_all < target_start_all).all()),
        int((origin_all >= target_start_all).sum()),
        0,
    )
    check(
        "every target has the configured non-overlapping horizon length",
        bool((metadata["horizon"].to_numpy(dtype=int) == cfg.horizon).all()),
        sorted(set(metadata["horizon"])),
        cfg.horizon,
    )
    check(
        "held-out tickers absent from train, validation, and calibration",
        fit_tickers.isdisjoint(cfg.unseen_test_tickers),
        sorted(fit_tickers & set(cfg.unseen_test_tickers)),
        "empty",
    )
    check(
        "training targets end by training cutoff",
        pd.to_datetime(metadata.iloc[train_indices]["target_end_date"]).max()
        <= train_cutoff,
        pd.to_datetime(metadata.iloc[train_indices]["target_end_date"]).max(),
        train_cutoff,
    )
    check(
        "validation origins begin after training cutoff",
        pd.to_datetime(metadata.iloc[validation_indices]["origin_date"]).min()
        > train_cutoff,
        pd.to_datetime(metadata.iloc[validation_indices]["origin_date"]).min(),
        train_cutoff,
    )
    check(
        "validation targets end by validation cutoff",
        pd.to_datetime(metadata.iloc[validation_indices]["target_end_date"]).max()
        <= validation_cutoff,
        pd.to_datetime(metadata.iloc[validation_indices]["target_end_date"]).max(),
        validation_cutoff,
    )
    check(
        "calibration origins begin after validation cutoff",
        pd.to_datetime(metadata.iloc[calibration_indices]["origin_date"]).min()
        > validation_cutoff,
        pd.to_datetime(metadata.iloc[calibration_indices]["origin_date"]).min(),
        validation_cutoff,
    )
    check(
        "calibration targets end by calibration cutoff",
        pd.to_datetime(metadata.iloc[calibration_indices]["target_end_date"]).max()
        <= calibration_cutoff,
        pd.to_datetime(metadata.iloc[calibration_indices]["target_end_date"]).max(),
        calibration_cutoff,
    )
    ordered_sections = (
        ("train", train_indices),
        ("validation", validation_indices),
        ("calibration", calibration_indices),
        ("test", np.concatenate([
            seen_ticker_unseen_day_indices,
            unseen_ticker_unseen_day_indices,
        ])),
    )
    for (previous_name, previous), (next_name, following) in zip(
        ordered_sections[:-1], ordered_sections[1:]
    ):
        previous_end = target_end_all.iloc[previous].max()
        next_start = target_start_all.iloc[following].min()
        check(
            f"target intervals do not cross {previous_name}-to-{next_name} boundary",
            previous_end < next_start,
            previous_end,
            next_start,
        )

    for cohort_name, indices, allowed_tickers in (
        (
            "seen-ticker unseen-day",
            seen_ticker_unseen_day_indices,
            set(cfg.training_tickers),
        ),
        (
            "unseen-ticker unseen-day",
            unseen_ticker_unseen_day_indices,
            set(cfg.unseen_test_tickers),
        ),
    ):
        cohort_tickers = set(metadata.iloc[indices]["ticker"])
        cohort_origin_min = pd.to_datetime(
            metadata.iloc[indices]["origin_date"]
        ).min()
        check(
            f"{cohort_name} origins begin after calibration cutoff",
            cohort_origin_min > calibration_cutoff,
            cohort_origin_min,
            calibration_cutoff,
        )
        check(
            f"{cohort_name} contains every configured ticker in its role",
            cohort_tickers == allowed_tickers,
            sorted(cohort_tickers),
            sorted(allowed_tickers),
        )

    seen_dates = set(pd.to_datetime(
        metadata.iloc[seen_ticker_unseen_day_indices]["origin_date"]
    ))
    unseen_dates = set(pd.to_datetime(
        metadata.iloc[unseen_ticker_unseen_day_indices]["origin_date"]
    ))
    check(
        "seen and unseen ticker cohorts use the identical origin-date calendar",
        seen_dates == unseen_dates,
        len(seen_dates),
        len(unseen_dates),
    )

    return pd.DataFrame(checks)


def prepare_split(
    dataset: MarketDataset,
    cfg: Config,
) -> PreparedSplit:
    metadata = dataset.metadata
    training_ticker_mask = metadata["ticker"].isin(cfg.training_tickers)
    unseen_ticker_mask = metadata["ticker"].isin(cfg.unseen_test_tickers)
    training_dates = np.array(sorted(pd.to_datetime(
        metadata.loc[training_ticker_mask, "origin_date"]
    ).unique()))

    explicit_boundaries = (
        cfg.explicit_train_end_date,
        cfg.explicit_validation_end_date,
        cfg.explicit_calibration_end_date,
    )
    if any(value is not None for value in explicit_boundaries):
        if not all(value is not None for value in explicit_boundaries):
            raise ValueError("All three explicit fitting cutoffs must be supplied.")
        train_cutoff = pd.Timestamp(cfg.explicit_train_end_date)
        validation_cutoff = pd.Timestamp(cfg.explicit_validation_end_date)
        calibration_cutoff = pd.Timestamp(cfg.explicit_calibration_end_date)
        if not train_cutoff < validation_cutoff < calibration_cutoff:
            raise ValueError("Explicit fitting cutoffs must increase chronologically.")
    else:
        train_cutoff = cutoff_from_fraction(training_dates, cfg.train_fraction)
        validation_cutoff = cutoff_from_fraction(
            training_dates, cfg.train_fraction + cfg.validation_fraction
        )
        calibration_cutoff = cutoff_from_fraction(
            training_dates,
            cfg.train_fraction + cfg.validation_fraction + cfg.calibration_fraction,
        )

    origin = pd.to_datetime(metadata["origin_date"])
    target_end = pd.to_datetime(metadata["target_end_date"])

    train_indices = np.flatnonzero(
        (training_ticker_mask & (target_end <= train_cutoff)).to_numpy()
    )
    validation_indices = np.flatnonzero((
        training_ticker_mask
        & (origin > train_cutoff)
        & (target_end <= validation_cutoff)
    ).to_numpy())
    calibration_indices = np.flatnonzero((
        training_ticker_mask
        & (origin > validation_cutoff)
        & (target_end <= calibration_cutoff)
    ).to_numpy())
    post_calibration_date_sets = []
    for ticker in cfg.target_tickers:
        eligible_test = (origin > calibration_cutoff)
        if cfg.explicit_test_start_date is not None:
            eligible_test &= origin >= pd.Timestamp(cfg.explicit_test_start_date)
        if cfg.explicit_test_origin_end_date is not None:
            eligible_test &= origin <= pd.Timestamp(
                cfg.explicit_test_origin_end_date
            )
        if cfg.explicit_test_origin_dates:
            eligible_test &= origin.isin(pd.to_datetime(
                list(cfg.explicit_test_origin_dates)
            ))
        if cfg.explicit_test_end_date is not None:
            eligible_test &= target_end <= pd.Timestamp(cfg.explicit_test_end_date)
        ticker_dates = set(pd.to_datetime(metadata.loc[
            (metadata["ticker"] == ticker) & eligible_test,
            "origin_date",
        ]))
        post_calibration_date_sets.append(ticker_dates)
    common_test_dates = set.intersection(*post_calibration_date_sets)
    if len(common_test_dates) < cfg.minimum_test_origins_per_ticker:
        raise ValueError(
            "Too few common post-calibration origins across every configured "
            f"ticker: {len(common_test_dates)} < "
            f"{cfg.minimum_test_origins_per_ticker}. Check cache coverage."
        )
    common_test_date_mask = origin.isin(common_test_dates)
    seen_ticker_unseen_day_indices = np.flatnonzero((
        training_ticker_mask & common_test_date_mask
    ).to_numpy())
    unseen_ticker_unseen_day_indices = np.flatnonzero((
        unseen_ticker_mask & common_test_date_mask
    ).to_numpy())
    test_indices = np.sort(np.concatenate([
        seen_ticker_unseen_day_indices,
        unseen_ticker_unseen_day_indices,
    ]))

    if min(
        len(train_indices),
        len(validation_indices),
        len(calibration_indices),
        len(seen_ticker_unseen_day_indices),
        len(unseen_ticker_unseen_day_indices),
    ) == 0:
        raise ValueError("At least one fitting section or test cohort is empty.")

    leakage_audit = assert_split_integrity(
        dataset,
        train_indices,
        validation_indices,
        calibration_indices,
        seen_ticker_unseen_day_indices,
        unseen_ticker_unseen_day_indices,
        train_cutoff,
        validation_cutoff,
        calibration_cutoff,
        cfg,
    )

    active_indices, active_names, dropped_names = resolve_active_feature_set(
        dataset, cfg
    )
    standardiser = RobustStandardiser.fit(
        dataset.features[train_indices],
        cfg.winsor_lower_quantile,
        cfg.winsor_upper_quantile,
        cfg.standardised_clip,
    )
    scaled_features = standardiser.transform(dataset.features)
    ticker_standardisers: dict[str, RobustStandardiser] = {}
    if cfg.normalise_features_by_ticker:
        ticker_values = metadata["ticker"].to_numpy()
        for ticker in cfg.training_tickers:
            ticker_train = train_indices[ticker_values[train_indices] == ticker]
            if len(ticker_train) == 0:
                raise ValueError(f"No training rows available to scale {ticker}.")
            local_standardiser = RobustStandardiser.fit(
                dataset.features[ticker_train],
                cfg.winsor_lower_quantile,
                cfg.winsor_upper_quantile,
                cfg.standardised_clip,
            )
            ticker_standardisers[ticker] = local_standardiser
            ticker_rows = np.flatnonzero(ticker_values == ticker)
            scaled_features[ticker_rows] = local_standardiser.transform(
                dataset.features[ticker_rows]
            )
    active_mask = np.zeros(len(dataset.feature_names), dtype=bool)
    active_mask[active_indices] = True
    # Zero masking keeps canonical width, fan-in, parameter count, and seed
    # behaviour fixed across every engineered-feature ablation.
    scaled_features[:, ~active_mask] = 0.0

    ticker_identity = np.zeros(
        (len(metadata), len(cfg.training_tickers)),
        dtype=np.float32,
    )
    if cfg.include_ticker_identity:
        ticker_values = metadata["ticker"].to_numpy()
        for identity_index, ticker in enumerate(cfg.training_tickers):
            ticker_identity[ticker_values == ticker, identity_index] = 1.0

    identity_names = [
        f"ticker_identity_{ticker}" for ticker in cfg.training_tickers
    ]
    context = np.concatenate(
        [scaled_features, ticker_identity],
        axis=1,
    ).astype(np.float32)
    context_feature_names = list(dataset.feature_names) + identity_names

    ticker_thresholds, labels, threshold_sources = (
        build_ticker_thresholds_and_labels(dataset, train_indices, cfg)
    )
    (
        class_weights,
        train_proportions,
        transition_matrices,
        initial_probabilities,
    ) = build_training_regime_statistics(
        metadata, labels, train_indices, cfg
    )

    metadata.loc[:, "split"] = "not_used_for_fitting_or_final_test"
    metadata.loc[train_indices, "split"] = "train"
    metadata.loc[validation_indices, "split"] = "validation"
    metadata.loc[calibration_indices, "split"] = "calibration"
    metadata.loc[test_indices, "split"] = "test"
    metadata.loc[:, "evaluation_cohort"] = "not_evaluated"
    metadata.loc[
        seen_ticker_unseen_day_indices, "evaluation_cohort"
    ] = "seen_ticker_unseen_day"
    metadata.loc[
        unseen_ticker_unseen_day_indices, "evaluation_cohort"
    ] = "unseen_ticker_unseen_day"
    metadata.loc[:, "regime"] = labels
    metadata.loc[:, "regime_name"] = REGIME_NAMES[labels]
    metadata.loc[:, "regime_threshold_source"] = threshold_sources[
        metadata["ticker_id"].to_numpy(dtype=int)
    ]

    print("\nStrict fitting sections and final generalisation cohorts:")
    print(f"  Train:                     {len(train_indices):,}")
    print(f"  Validation:                {len(validation_indices):,}")
    print(f"  Calibration:               {len(calibration_indices):,}")
    print(f"  Seen ticker / unseen day:  {len(seen_ticker_unseen_day_indices):,}")
    print(f"  Unseen ticker / unseen day:{len(unseen_ticker_unseen_day_indices):,}")
    print(f"  Train cutoff:       {train_cutoff.date()}")
    print(f"  Validation cutoff:  {validation_cutoff.date()}")
    print(f"  Calibration cutoff: {calibration_cutoff.date()}")
    print(f"  Common final-test origin dates: {len(common_test_dates):,}")

    boundary_units = (
        "future/recent volatility ratio"
        if cfg.regime_target_mode == "relative_to_recent_volatility"
        else "annualised volatility"
    )
    print(f"\nTraining-only regime boundaries ({boundary_units}):")
    for ticker_id, ticker in enumerate(cfg.target_tickers):
        print(
            f"  {ticker}: {math.exp(ticker_thresholds[ticker_id, 0]):.3f}, "
            f"{math.exp(ticker_thresholds[ticker_id, 1]):.3f} "
            f"[{threshold_sources[ticker_id]}]"
        )

    prepared = PreparedSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
        seen_ticker_unseen_day_indices=seen_ticker_unseen_day_indices,
        unseen_ticker_unseen_day_indices=unseen_ticker_unseen_day_indices,
        context=context,
        model_targets=dataset.targets_log_vol.copy(),
        har_predictions=np.full(len(dataset.metadata), np.nan, dtype=np.float32),
        regime_labels=labels,
        ticker_thresholds=ticker_thresholds,
        class_weights=class_weights,
        train_regime_proportions=train_proportions,
        transition_matrices=transition_matrices,
        initial_regime_probabilities=initial_probabilities,
        standardiser=standardiser,
        ticker_standardisers=ticker_standardisers,
        active_feature_indices=active_indices,
        active_feature_names=active_names,
        dropped_feature_names=dropped_names,
        context_feature_names=context_feature_names,
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
        calibration_cutoff=calibration_cutoff,
        leakage_audit=leakage_audit,
    )
    fit_tickers = set(metadata.iloc[train_indices]["ticker"])
    append_audit_check(
        prepared,
        "pooled scaler fitting cohort contains training tickers only",
        fit_tickers <= set(cfg.training_tickers),
        sorted(fit_tickers),
        sorted(cfg.training_tickers),
    )
    expected_local_scalers = (
        set(cfg.training_tickers) if cfg.normalise_features_by_ticker else set()
    )
    append_audit_check(
        prepared,
        "per-ticker scalers are fitted only for training tickers",
        set(ticker_standardisers) == expected_local_scalers,
        sorted(ticker_standardisers),
        sorted(expected_local_scalers),
    )
    heldout_source_ids = range(
        len(cfg.training_tickers), len(cfg.target_tickers)
    )
    heldout_threshold_sources = {
        str(threshold_sources[index]) for index in heldout_source_ids
    }
    append_audit_check(
        prepared,
        "held-out regime thresholds use pooled training-ticker rows only",
        all(
            source.startswith("pooled_training_tickers::")
            for source in heldout_threshold_sources
        ),
        sorted(heldout_threshold_sources),
        [f"pooled_training_tickers::{cfg.regime_target_mode}"],
    )

    print("\nLeakage and cohort audit:")
    print(prepared.leakage_audit.to_string(index=False))
    return prepared

# %% [markdown cell 15]
# ## Single-pass conditional flow mixture

# %% [cell 16]
def stable_log_cosh(value: torch.Tensor) -> torch.Tensor:
    return (
        torch.logaddexp(value, -value)
        - math.log(2.0)
    )


@dataclass
class ForwardBundle:
    logits: torch.Tensor
    probabilities: torch.Tensor
    mixture_probabilities: torch.Tensor
    ordinal_logits: torch.Tensor | None
    locations: torch.Tensor
    scales: torch.Tensor
    skews: torch.Tensor
    tails: torch.Tensor
    expert_medians: torch.Tensor
    mixture_median_approximation: torch.Tensor
    quantile_lower: torch.Tensor
    quantile_median: torch.Tensor
    quantile_upper: torch.Tensor
    component_log_probabilities: torch.Tensor | None
    mixture_log_probability: torch.Tensor | None


class RegimeFlowMixture(nn.Module):
    def __init__(
        self,
        context_dimension: int,
        cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        self.expert_count = cfg.regimes if cfg.use_mixture_experts else 1
        self.unknown_ticker_index = len(cfg.training_tickers)
        encoder_input_dimension = context_dimension
        if cfg.use_ticker_embeddings:
            self.ticker_embedding = nn.Embedding(
                len(cfg.training_tickers) + 1,
                cfg.ticker_embedding_dimension,
            )
            nn.init.zeros_(self.ticker_embedding.weight[self.unknown_ticker_index])
            encoder_input_dimension += cfg.ticker_embedding_dimension
        else:
            self.ticker_embedding = None

        self.encoder = nn.Sequential(
            nn.Linear(
                encoder_input_dimension,
                cfg.hidden_size,
            ),
            nn.LayerNorm(cfg.hidden_size),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(
                cfg.hidden_size,
                cfg.hidden_size,
            ),
            nn.SiLU(),
        )

        self.gate_head = nn.Linear(
            cfg.hidden_size,
            1 if cfg.regime_classifier_type == "ordinal" else cfg.regimes,
        )
        if cfg.regime_classifier_type == "ordinal":
            # One latent volatility score with globally ordered cutpoints.
            # The positive gap guarantees P(y>0) >= P(y>1) by construction.
            self.ordinal_threshold_base = nn.Parameter(torch.tensor(-0.5))
            self.ordinal_threshold_gap_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(1.0)))
            )
        else:
            self.register_parameter("ordinal_threshold_base", None)
            self.register_parameter("ordinal_threshold_gap_raw", None)
        self.quantile_head = nn.Linear(
            cfg.hidden_size,
            3,
        )
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    cfg.hidden_size,
                    cfg.hidden_size,
                ),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size, 4),
            )
            for _ in range(self.expert_count)
        ])
        if cfg.use_ticker_specific_heads:
            self.ticker_output_heads = nn.ModuleList([
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
                for _ in range(len(cfg.training_tickers) + 1)
            ])
            for adapter in self.ticker_output_heads:
                nn.init.zeros_(adapter.weight)
                nn.init.zeros_(adapter.bias)
        else:
            self.ticker_output_heads = None

        for head in self.expert_heads:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            with torch.no_grad():
                final_layer.bias[:] = torch.tensor([
                    0.0 if cfg.forecast_mode == "har_residual" else -1.8,
                    0.0,
                    0.0,
                    0.0,
                ])

    def forward_bundle(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
        ticker_ids: torch.Tensor | None = None,
    ) -> ForwardBundle:
        if ticker_ids is None:
            ticker_lookup = torch.full(
                (len(context),),
                self.unknown_ticker_index,
                dtype=torch.long,
                device=context.device,
            )
        else:
            ticker_lookup = torch.where(
                ticker_ids < len(self.cfg.training_tickers),
                ticker_ids,
                torch.full_like(ticker_ids, self.unknown_ticker_index),
            )
        if self.ticker_embedding is not None:
            context = torch.cat(
                [context, self.ticker_embedding(ticker_lookup)], dim=1
            )

        # Exactly one stochastic encoder pass per batch.
        hidden = self.encoder(context)
        if self.ticker_output_heads is not None:
            adapters = torch.stack(
                [head(hidden) for head in self.ticker_output_heads],
                dim=1,
            )
            row = torch.arange(len(hidden), device=hidden.device)
            hidden = hidden + adapters[row, ticker_lookup]

        classifier_output = self.gate_head(hidden)
        ordinal_logits = None
        if self.cfg.regime_classifier_type == "ordinal":
            if (
                self.ordinal_threshold_base is None
                or self.ordinal_threshold_gap_raw is None
            ):
                raise RuntimeError("Ordinal thresholds were not initialized.")
            first_threshold = self.ordinal_threshold_base
            second_threshold = (
                first_threshold
                + F.softplus(self.ordinal_threshold_gap_raw)
                + 1e-4
            )
            ordinal_logits = torch.cat(
                [
                    classifier_output - first_threshold,
                    classifier_output - second_threshold,
                ],
                dim=1,
            )
            exceedance = torch.sigmoid(ordinal_logits)
            probabilities = torch.stack((
                1.0 - exceedance[:, 0],
                exceedance[:, 0] - exceedance[:, 1],
                exceedance[:, 1],
            ), dim=1).clamp_min(EPS)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            logits = torch.log(probabilities)
        else:
            logits = classifier_output
            probabilities = F.softmax(logits, dim=1)

        if self.expert_count == 1:
            mixture_probabilities = torch.ones(
                (len(context), 1), device=context.device, dtype=context.dtype
            )
        elif self.cfg.regime_gate_mode == "soft":
            mixture_probabilities = probabilities
        else:
            # `none` removes all regime terms through its ablation overrides;
            # `auxiliary` keeps classification but cannot affect forecasts.
            mixture_probabilities = torch.full_like(
                probabilities, 1.0 / self.expert_count
            )

        raw_quantiles = self.quantile_head(hidden)
        quantile_median = raw_quantiles[:, 0]
        quantile_lower = (
            quantile_median
            - F.softplus(raw_quantiles[:, 1])
        )
        quantile_upper = (
            quantile_median
            + F.softplus(raw_quantiles[:, 2])
        )

        raw_parameters = torch.stack(
            [head(hidden) for head in self.expert_heads],
            dim=1,
        )

        locations = raw_parameters[:, :, 0]
        scales = (
            self.cfg.minimum_scale
            + (
                self.cfg.maximum_scale
                - self.cfg.minimum_scale
            )
            * torch.sigmoid(raw_parameters[:, :, 1])
        )
        skews = (
            self.cfg.maximum_skew
            * torch.tanh(raw_parameters[:, :, 2])
        )
        tails = (
            self.cfg.minimum_tail
            + (
                self.cfg.maximum_tail
                - self.cfg.minimum_tail
            )
            * torch.sigmoid(raw_parameters[:, :, 3])
        )

        expert_medians = (
            locations
            + scales * torch.sinh(skews / tails)
        )
        mixture_median_approximation = torch.sum(
            mixture_probabilities * expert_medians,
            dim=1,
        )

        component_log_probabilities = None
        mixture_log_probability = None

        if target is not None:
            target_values = target.squeeze(-1).unsqueeze(1)
            standardised = (
                target_values - locations
            ) / scales

            inverse_argument = (
                tails * torch.asinh(standardised)
                - skews
            )
            inverse_argument = torch.clamp(
                inverse_argument,
                -12.0,
                12.0,
            )
            base = torch.sinh(inverse_argument)

            base_log_density = (
                -0.5 * base**2
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_jacobian = (
                torch.log(tails)
                - torch.log(scales)
                + stable_log_cosh(inverse_argument)
                - 0.5 * torch.log1p(standardised**2)
            )

            component_log_probabilities = (
                base_log_density + log_jacobian
            )
            mixture_log_probability = torch.logsumexp(
                torch.log(mixture_probabilities.clamp_min(EPS))
                + component_log_probabilities,
                dim=1,
            )

        return ForwardBundle(
            logits=logits,
            probabilities=probabilities,
            mixture_probabilities=mixture_probabilities,
            ordinal_logits=ordinal_logits,
            locations=locations,
            scales=scales,
            skews=skews,
            tails=tails,
            expert_medians=expert_medians,
            mixture_median_approximation=(
                mixture_median_approximation
            ),
            quantile_lower=quantile_lower,
            quantile_median=quantile_median,
            quantile_upper=quantile_upper,
            component_log_probabilities=(
                component_log_probabilities
            ),
            mixture_log_probability=mixture_log_probability,
        )

    def sample_from_bundle(
        self,
        bundle: ForwardBundle,
        number_of_samples: int,
    ) -> torch.Tensor:
        batch_size = bundle.locations.shape[0]

        noise = torch.randn(
            batch_size,
            self.expert_count,
            number_of_samples,
            device=bundle.locations.device,
            dtype=bundle.locations.dtype,
        )

        transformed = torch.sinh(
            (
                torch.asinh(noise)
                + bundle.skews.unsqueeze(-1)
            )
            / bundle.tails.unsqueeze(-1)
        )

        component_samples = (
            bundle.locations.unsqueeze(-1)
            + bundle.scales.unsqueeze(-1)
            * transformed
        )

        selected_components = torch.distributions.Categorical(
            probs=bundle.mixture_probabilities
        ).sample((number_of_samples,)).transpose(0, 1)

        selected = torch.gather(
            component_samples.permute(0, 2, 1),
            dim=2,
            index=selected_components.unsqueeze(-1),
        ).squeeze(-1)

        return selected

# %% [markdown cell 17]
# ## Consistent training objective, guarded optimisation, and early stopping
# 
# The conditional-flow objective is unchanged. Configuration now controls a minimum training phase, the learning-rate schedule, and early stopping independently. Every seed restores its best validation checkpoint; learning rate, pre-clipping gradient norm, epoch time, and total runtime are reported and saved.

# %% [cell 18]
def quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(
        quantile * error,
        (quantile - 1.0) * error,
    ).mean()


def regime_classification_loss(
    bundle: ForwardBundle,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    sample_weights = class_weights[labels]
    use_weights = cfg.regime_loss_type in {"weighted_cross_entropy", "focal"}

    if cfg.regime_classifier_type == "ordinal":
        if bundle.ordinal_logits is None:
            raise RuntimeError("Ordinal classifier did not return threshold logits.")
        thresholds = torch.arange(
            cfg.regimes - 1, device=labels.device
        ).unsqueeze(0)
        exceedance_targets = (labels.unsqueeze(1) > thresholds).float()
        per_sample = F.binary_cross_entropy_with_logits(
            bundle.ordinal_logits,
            exceedance_targets,
            reduction="none",
        ).mean(dim=1)
    else:
        per_sample = F.cross_entropy(
            bundle.logits,
            labels,
            reduction="none",
            label_smoothing=cfg.label_smoothing,
        )

    if cfg.regime_loss_type == "focal":
        true_probability = bundle.probabilities.gather(
            1, labels.unsqueeze(1)
        ).squeeze(1).clamp(EPS, 1.0 - EPS)
        per_sample = (1.0 - true_probability) ** cfg.focal_gamma * per_sample
    if use_weights:
        return torch.sum(per_sample * sample_weights) / sample_weights.sum().clamp_min(EPS)
    return per_sample.mean()


def objective(
    model: RegimeFlowMixture,
    context: torch.Tensor,
    target: torch.Tensor,
    actual_log_vol: torch.Tensor,
    har_forecast: torch.Tensor,
    labels: torch.Tensor,
    ticker_ids: torch.Tensor,
    class_weights: torch.Tensor,
    target_gate_usage: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    bundle = model.forward_bundle(context, target, ticker_ids=ticker_ids)

    if (
        bundle.mixture_log_probability is None
        or bundle.component_log_probabilities is None
    ):
        raise RuntimeError("Training bundle lacks log probabilities.")

    mixture_nll = -bundle.mixture_log_probability.mean()

    if model.expert_count == cfg.regimes:
        selected_expert_nll = -bundle.component_log_probabilities.gather(
            1, labels.unsqueeze(1)
        ).mean()
    else:
        selected_expert_nll = -bundle.component_log_probabilities[:, 0].mean()

    classification = regime_classification_loss(
        bundle,
        labels,
        class_weights,
        cfg,
    )

    average_gate_usage = bundle.probabilities.mean(dim=0)
    gate_balance = torch.mean(
        (
            average_gate_usage
            - target_gate_usage
        ) ** 2
    )

    model_target = target.squeeze(1)
    actual_log_volatility = actual_log_vol.squeeze(1)
    if cfg.forecast_mode == "har_residual":
        predicted_volatility = (
            har_forecast.squeeze(1) + bundle.mixture_median_approximation
        ).clamp_min(cfg.minimum_reported_volatility)
    else:
        predicted_volatility = torch.exp(
            bundle.mixture_median_approximation
        )
    predicted_variance = predicted_volatility.square().clamp_min(EPS)
    actual_variance = torch.exp(
        2.0 * actual_log_volatility
    ).clamp_min(EPS)

    variance_ratio = (
        actual_variance / predicted_variance
    )
    qlike = torch.mean(
        variance_ratio
        - torch.log(variance_ratio)
        - 1.0
    )

    quantile_objective = (
        quantile_loss(
            bundle.quantile_lower,
            model_target,
            0.05,
        )
        + quantile_loss(
            bundle.quantile_median,
            model_target,
            0.50,
        )
        + quantile_loss(
            bundle.quantile_upper,
            model_target,
            0.95,
        )
    )

    total = (
        mixture_nll
        + cfg.expert_alignment_weight
        * selected_expert_nll
        + cfg.regime_classification_weight
        * classification
        + cfg.gate_balance_weight
        * gate_balance
        + cfg.qlike_weight * qlike
        + cfg.quantile_weight
        * quantile_objective
    )

    metrics = {
        "total": float(total.detach().cpu()),
        "nll": float(mixture_nll.detach().cpu()),
        "classification": float(
            classification.detach().cpu()
        ),
        "qlike": float(qlike.detach().cpu()),
        "accuracy": float(
            (
                bundle.logits.argmax(dim=1)
                == labels
            )
            .float()
            .mean()
            .detach()
            .cpu()
        ),
        "medium_probability": float(
            bundle.probabilities[:, 1]
            .mean()
            .detach()
            .cpu()
        ),
    }

    return total, metrics


def make_loader(
    split: PreparedSplit,
    dataset: MarketDataset,
    indices: np.ndarray,
    cfg: Config,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    data = TensorDataset(
        torch.from_numpy(
            split.context[indices]
        ).float(),
        torch.from_numpy(
            split.model_targets[indices]
        ).float().unsqueeze(1),
        torch.from_numpy(
            dataset.targets_log_vol[indices]
        ).float().unsqueeze(1),
        torch.from_numpy(
            split.har_predictions[indices]
        ).float().unsqueeze(1),
        torch.from_numpy(
            split.regime_labels[indices]
        ).long(),
        torch.from_numpy(
            dataset.metadata.iloc[indices]["ticker_id"].to_numpy(
                dtype=np.int64, copy=True
            )
        ).long(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        data,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        num_workers=0,
        pin_memory=(cfg.pin_memory and DEVICE.type == "cuda"),
    )


def run_epoch(
    model: RegimeFlowMixture,
    loader: DataLoader,
    cfg: Config,
    class_weights: torch.Tensor,
    target_gate_usage: torch.Tensor,
    optimiser: optim.Optimizer | None,
) -> dict[str, float]:
    training = optimiser is not None
    model.train(training)

    totals = {}
    observation_count = 0

    for context, target, actual_log_vol, har_forecast, labels, ticker_ids in loader:
        non_blocking = cfg.pin_memory and DEVICE.type == "cuda"
        context = context.to(DEVICE, non_blocking=non_blocking)
        target = target.to(DEVICE, non_blocking=non_blocking)
        actual_log_vol = actual_log_vol.to(DEVICE, non_blocking=non_blocking)
        har_forecast = har_forecast.to(DEVICE, non_blocking=non_blocking)
        labels = labels.to(DEVICE, non_blocking=non_blocking)
        ticker_ids = ticker_ids.to(DEVICE, non_blocking=non_blocking)
        batch_size = len(context)

        if training:
            optimiser.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            total, metrics = objective(
                model,
                context,
                target,
                actual_log_vol,
                har_forecast,
                labels,
                ticker_ids,
                class_weights,
                target_gate_usage,
                cfg,
            )

            if not torch.isfinite(total):
                raise FloatingPointError(
                    "Non-finite objective encountered; training stopped before "
                    "the optimiser could corrupt the checkpoint."
                )

            if training:
                total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.gradient_clip,
                    error_if_nonfinite=True,
                )
                metrics["gradient_norm"] = float(
                    gradient_norm.detach().cpu()
                )
                optimiser.step()

        for key, value in metrics.items():
            totals[key] = (
                totals.get(key, 0.0)
                + value * batch_size
            )
        observation_count += batch_size

    return {
        key: value / observation_count
        for key, value in totals.items()
    }


def fit_single_model(
    dataset: MarketDataset,
    split: PreparedSplit,
    cfg: Config,
    seed: int,
) -> tuple[
    RegimeFlowMixture,
    dict[str, list[float]],
    int,
]:
    set_seed(seed)

    train_loader = make_loader(
        split,
        dataset,
        split.train_indices,
        cfg,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        split,
        dataset,
        split.validation_indices,
        cfg,
        shuffle=False,
        seed=seed,
    )

    model = RegimeFlowMixture(
        split.context.shape[1],
        cfg,
    ).to(DEVICE)

    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        threshold=cfg.early_stopping_min_delta,
        threshold_mode="abs",
        cooldown=cfg.scheduler_cooldown,
        min_lr=cfg.minimum_learning_rate,
    )

    class_weights = torch.from_numpy(
        split.class_weights
    ).float().to(DEVICE)
    target_gate_usage = torch.from_numpy(
        split.train_regime_proportions
    ).float().to(DEVICE)

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_nll": [],
        "validation_qlike": [],
        "checkpoint_value": [],
        "validation_accuracy": [],
        "validation_medium_probability": [],
        "learning_rate": [],
        "gradient_norm": [],
        "epoch_seconds": [],
    }

    best_checkpoint_value = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"\nTraining seed {seed} | parameters={parameter_count:,} | "
        f"minimum/cap epochs={cfg.minimum_epochs}/{cfg.epochs}..."
    )

    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            cfg,
            class_weights,
            target_gate_usage,
            optimiser,
        )

        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                cfg,
                class_weights,
                target_gate_usage,
                optimiser=None,
            )

        checkpoint_value = (
            validation_metrics["qlike"]
            if cfg.checkpoint_metric == "validation_qlike"
            else validation_metrics["total"]
        )
        scheduler.step(checkpoint_value)
        current_learning_rate = float(optimiser.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_started

        history["train_loss"].append(
            train_metrics["total"]
        )
        history["validation_loss"].append(
            validation_metrics["total"]
        )
        history["validation_nll"].append(
            validation_metrics["nll"]
        )
        history["validation_qlike"].append(
            validation_metrics["qlike"]
        )
        history["checkpoint_value"].append(checkpoint_value)
        history["validation_accuracy"].append(
            validation_metrics["accuracy"]
        )
        history[
            "validation_medium_probability"
        ].append(
            validation_metrics["medium_probability"]
        )
        history["learning_rate"].append(current_learning_rate)
        history["gradient_norm"].append(train_metrics["gradient_norm"])
        history["epoch_seconds"].append(epoch_seconds)

        if (
            checkpoint_value
            < best_checkpoint_value - cfg.early_stopping_min_delta
        ):
            best_checkpoint_value = checkpoint_value
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % cfg.print_every == 0:
            print(
                f"  Epoch {epoch:03d}/{cfg.epochs} | "
                f"train={train_metrics['total']:.4f} | "
                f"validation={validation_metrics['total']:.4f} | "
                f"qlike={validation_metrics['qlike']:.4f} | "
                f"checkpoint={checkpoint_value:.4f} | "
                f"accuracy={validation_metrics['accuracy']:.3f} | "
                f"P(medium)={validation_metrics['medium_probability']:.3f} | "
                f"lr={current_learning_rate:.2e} | "
                f"epoch={epoch_seconds:.1f}s | "
                f"max ETA={np.mean(history['epoch_seconds'][-5:]) * (cfg.epochs - epoch) / 60.0:.1f}m"
            )

        if epoch >= cfg.minimum_epochs and stale_epochs >= cfg.patience:
            print(
                f"  Early stopping; best epoch={best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training failed to create a checkpoint."
        )

    model.load_state_dict(best_state)
    model.eval()

    return model, history, best_epoch


def fit_ensemble(
    dataset: MarketDataset,
    split: PreparedSplit,
    cfg: Config,
) -> tuple[
    list[RegimeFlowMixture],
    list[dict[str, list[float]]],
    list[int],
]:
    models = []
    histories = []
    best_epochs = []

    for seed in cfg.ensemble_seeds:
        model, history, best_epoch = fit_single_model(
            dataset,
            split,
            cfg,
            seed,
        )
        models.append(model)
        histories.append(history)
        best_epochs.append(best_epoch)

    return models, histories, best_epochs


def build_training_summary(
    histories: list[dict[str, list[float]]],
    best_epochs: list[int],
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    for seed, history, best_epoch in zip(
        cfg.ensemble_seeds, histories, best_epochs
    ):
        epochs_run = len(history["validation_loss"])
        best_index = best_epoch - 1
        rows.append({
            "seed": seed,
            "configured_minimum_epochs": cfg.minimum_epochs,
            "configured_epoch_cap": cfg.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "checkpoint_metric": cfg.checkpoint_metric,
            "best_checkpoint_value": history["checkpoint_value"][best_index],
            "best_validation_objective": history["validation_loss"][best_index],
            "best_validation_qlike": history["validation_qlike"][best_index],
            "final_validation_objective": history["validation_loss"][-1],
            "final_validation_qlike": history["validation_qlike"][-1],
            "final_learning_rate": history["learning_rate"][-1],
            "final_gradient_norm": history["gradient_norm"][-1],
            "total_seconds": float(np.sum(history["epoch_seconds"])),
            "total_minutes": float(np.sum(history["epoch_seconds"]) / 60.0),
            "stopped_early": epochs_run < cfg.epochs,
        })
    return pd.DataFrame(rows)


def build_complexity_summary(
    models: list[RegimeFlowMixture],
    histories: list[dict[str, list[float]]],
    inference_seconds: float,
    inference_rows: int,
    point_metrics_table: pd.DataFrame,
    per_seed_metrics: pd.DataFrame,
    baseline_predictions: dict[str, object],
    cfg: Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pooled_seed_metrics = per_seed_metrics.query(
        "group_level == 'pooled'"
    ).set_index("seed")
    for seed, model, history in zip(cfg.ensemble_seeds, models, histories):
        parameters = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        rows.append({
            "model": "SigFlow v4",
            "seed": seed,
            "measurement_scope": "ensemble_member_estimate",
            "ensemble_members": 1,
            "trainable_parameters": parameters,
            "model_coefficients": parameters,
            "training_seconds": float(np.sum(history["epoch_seconds"])),
            "inference_seconds": inference_seconds / max(len(models), 1),
            "combined_fit_and_forecast_seconds": np.nan,
            "inference_milliseconds_per_1000_rows": (
                1000.0 * inference_seconds / max(inference_rows, 1)
                / max(len(models), 1) * 1000.0
            ),
            "peak_cpu_rss_bytes": np.nan,
            "peak_gpu_memory_bytes": np.nan,
            "pooled_qlike": float(pooled_seed_metrics.loc[seed, "qlike"]),
        })
    member_parameters = [
        int(row["trainable_parameters"]) for row in rows
    ]
    rows.append({
        "model": "SigFlow v4",
        "seed": "ensemble_total",
        "measurement_scope": "measured_ensemble_total",
        "ensemble_members": len(models),
        "trainable_parameters": int(sum(member_parameters)),
        "model_coefficients": int(sum(member_parameters)),
        "training_seconds": float(sum(
            np.sum(history["epoch_seconds"]) for history in histories
        )),
        "inference_seconds": inference_seconds,
        "combined_fit_and_forecast_seconds": np.nan,
        "inference_milliseconds_per_1000_rows": (
            inference_seconds / max(inference_rows, 1) * 1_000_000.0
        ),
        "peak_cpu_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ) * (1024 if sys.platform != "darwin" else 1),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else 0
        ),
    })
    baseline_complexity = baseline_predictions.get("_complexity", {})
    baseline_parameters = {
        "Rolling volatility": (0, np.nan, "feature_construction_not_timed"),
        "EWMA volatility": (1, np.nan, "feature_construction_not_timed"),
        "Log-HAR transferable": (
            int(baseline_complexity["Log-HAR transferable"]["model_coefficients"]),
            float(baseline_complexity["Log-HAR transferable"]["fit_and_forecast_seconds"]),
            "combined_fit_and_full_dataset_forecast",
        ),
        "Level HAR transferable": (
            int(baseline_complexity["Level HAR transferable"]["model_coefficients"]),
            float(baseline_complexity["Level HAR transferable"]["fit_and_forecast_seconds"]),
            "combined_fit_and_full_dataset_forecast",
        ),
        "Log-HAR leverage transferable": (
            int(baseline_complexity["Log-HAR leverage transferable"]["model_coefficients"]),
            float(baseline_complexity["Log-HAR leverage transferable"]["fit_and_forecast_seconds"]),
            "combined_fit_and_full_dataset_forecast",
        ),
        "GARCH(1,1)": (
            int(baseline_complexity["GARCH(1,1)"]["model_coefficients"]),
            float(baseline_complexity["GARCH(1,1)"]["fit_and_forecast_seconds"]),
            "combined_fit_and_full_dataset_forecast",
        ),
        "SigFlow + Log-HAR blend": (
            1,
            float(np.asarray(baseline_predictions.get(
                "blend_selection_seconds", [0.0]
            ))[0]),
            "calibration_weight_selection_only",
        ),
    }
    for model, (coefficients, elapsed, scope) in baseline_parameters.items():
        rows.append({
            "model": model,
            "seed": "deterministic",
            "measurement_scope": scope,
            "ensemble_members": 1,
            "trainable_parameters": 0,
            "model_coefficients": coefficients,
            "training_seconds": np.nan,
            "inference_seconds": np.nan,
            "combined_fit_and_forecast_seconds": elapsed,
            "inference_milliseconds_per_1000_rows": np.nan,
            "peak_cpu_rss_bytes": np.nan,
            "peak_gpu_memory_bytes": np.nan,
        })
    log_har_coefficients, log_har_seconds, _ = baseline_parameters[
        "Log-HAR transferable"
    ]
    blend_seconds = float(np.asarray(baseline_predictions.get(
        "blend_selection_seconds", [0.0]
    ))[0])
    ensemble_total = rows[len(models)]
    blend_row = next(
        row for row in rows if row["model"] == "SigFlow + Log-HAR blend"
    )
    blend_row.update({
        "measurement_scope": (
            "composite_ensemble_plus_log_har_and_calibration_selection"
        ),
        "ensemble_members": len(models),
        "trainable_parameters": int(ensemble_total["trainable_parameters"]),
        "model_coefficients": int(
            ensemble_total["model_coefficients"] + log_har_coefficients + 1
        ),
        "training_seconds": float(
            ensemble_total["training_seconds"] + blend_seconds
        ),
        "inference_seconds": float(ensemble_total["inference_seconds"]),
        "combined_fit_and_forecast_seconds": log_har_seconds,
        "inference_milliseconds_per_1000_rows": float(
            ensemble_total["inference_milliseconds_per_1000_rows"]
        ),
        "peak_cpu_rss_bytes": ensemble_total["peak_cpu_rss_bytes"],
        "peak_gpu_memory_bytes": ensemble_total["peak_gpu_memory_bytes"],
    })
    table = pd.DataFrame(rows)
    pooled = point_metrics_table[
        point_metrics_table["group_level"] == "pooled"
    ].set_index("model")
    har_qlike = (
        float(pooled.loc[cfg.primary_baseline, "qlike"])
        if cfg.primary_baseline in pooled.index else float("nan")
    )
    model_qlike = pooled["qlike"].to_dict()
    table["pooled_qlike"] = table["pooled_qlike"].fillna(
        table["model"].map(model_qlike)
    )
    table["qlike_improvement_over_log_har"] = (
        har_qlike - table["pooled_qlike"]
    )
    return table


def build_structured_experiment_summary(
    evaluation: dict[str, object],
    complexity: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    point = evaluation["point_metrics"]
    point = point[point["group_level"] == "cohort_ticker"].copy()
    point["metric_family"] = "point"
    point["complexity_seed"] = np.where(
        point["model"] == "SigFlow v4", "ensemble_total", "deterministic"
    )
    seeds = evaluation["per_seed_metrics"]
    seeds = seeds[seeds["group_level"] == "cohort_ticker"].copy()
    seeds["model"] = "SigFlow v4"
    seeds["metric_family"] = "point_per_seed"
    seeds["complexity_seed"] = seeds["seed"]
    combined = pd.concat([point, seeds], ignore_index=True, sort=False)
    combined["model_variant"] = cfg.run_name
    combined["horizon"] = cfg.horizon
    combined["test_window_id"] = cfg.test_window_id
    combined["experiment_mode"] = cfg.experiment_mode
    complexity_columns = [
        "model",
        "seed",
        "measurement_scope",
        "ensemble_members",
        "trainable_parameters",
        "model_coefficients",
        "training_seconds",
        "inference_seconds",
        "combined_fit_and_forecast_seconds",
        "inference_milliseconds_per_1000_rows",
        "peak_cpu_rss_bytes",
        "peak_gpu_memory_bytes",
    ]
    complexity_join = complexity[complexity_columns].rename(
        columns={"seed": "complexity_seed"}
    )
    return combined.merge(
        complexity_join,
        on=["model", "complexity_seed"],
        how="left",
        validate="many_to_one",
    )

# %% [markdown cell 19]
# ## Training-only HAR baselines and ensemble outputs

# %% [cell 20]
def _fit_ridge_coefficients(
    design: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(design)), design.astype(float)])
    penalty = 1e-4 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )


def _predict_ridge(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(design)), design.astype(float)])
    return design @ coefficients


def har_design_and_target(
    dataset: MarketDataset,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    feature_columns = [
        dataset.feature_names.index("log_realised_volatility_5"),
        dataset.feature_names.index("log_realised_volatility_20"),
        dataset.feature_names.index("log_realised_volatility_60"),
    ]
    log_design = dataset.features[:, feature_columns].astype(float)
    if variant == "log_har":
        return log_design, dataset.targets_log_vol.astype(float), True
    if variant == "level_har":
        return (
            np.exp(log_design),
            dataset.metadata["actual_vol"].to_numpy(dtype=float),
            False,
        )
    if variant == "log_har_leverage":
        last_return = dataset.features[:, [
            dataset.feature_names.index("last_return")
        ]].astype(float)
        downside = np.minimum(last_return, 0.0)
        return (
            np.column_stack([log_design, downside, downside**2]),
            dataset.targets_log_vol.astype(float),
            True,
        )
    raise ValueError(f"Unknown HAR variant: {variant}")


def fit_ticker_specific_har(
    dataset: MarketDataset,
    split: PreparedSplit,
    cfg: Config,
    variant: str = "log_har",
) -> np.ndarray:
    """Fit HAR with chronological OOF train forecasts and frozen future fits."""
    design, target, log_output = har_design_and_target(dataset, variant)
    pooled_coefficients = _fit_ridge_coefficients(
        design[split.train_indices],
        target[split.train_indices],
    )
    predictions = np.full(len(dataset.metadata), np.nan, dtype=float)
    ticker_ids = dataset.metadata["ticker_id"].to_numpy(dtype=int)
    origin_dates = pd.to_datetime(dataset.metadata["origin_date"])
    target_end_dates = pd.to_datetime(dataset.metadata["target_end_date"])
    fallback = dataset.metadata["rolling_vol_baseline"].to_numpy(dtype=float)

    for ticker_id, ticker in enumerate(cfg.target_tickers):
        ticker_all = np.flatnonzero(ticker_ids == ticker_id)
        if ticker in cfg.training_tickers:
            ticker_train = split.train_indices[
                ticker_ids[split.train_indices] == ticker_id
            ]
            ticker_train = ticker_train[
                np.argsort(origin_dates.iloc[ticker_train].to_numpy())
            ]
            coefficients = _fit_ridge_coefficients(
                design[ticker_train], target[ticker_train]
            )
            minimum_history = min(60, max(20, len(ticker_train) // 5))
            oof_coefficients: np.ndarray | None = None
            for position, row_index in enumerate(ticker_train):
                # A merely earlier forecast row is not necessarily observable:
                # for multi-session targets its realised target may still extend
                # beyond this row's forecast origin. Purge those overlapping
                # outcomes before every chronological OOF HAR fit.
                earlier = ticker_train[:position]
                history = earlier[
                    target_end_dates.iloc[earlier].to_numpy()
                    <= origin_dates.iloc[row_index]
                ]
                if len(history) < minimum_history:
                    predictions[row_index] = fallback[row_index]
                    continue
                if oof_coefficients is None or position % 20 == 0:
                    oof_coefficients = _fit_ridge_coefficients(
                        design[history], target[history]
                    )
                raw = _predict_ridge(
                    design[[row_index]], oof_coefficients
                )[0]
                predictions[row_index] = math.exp(raw) if log_output else raw
        else:
            # No held-out outcome is used to fit this transferable baseline.
            coefficients = pooled_coefficients

        remaining = ticker_all[~np.isfinite(predictions[ticker_all])]
        raw = _predict_ridge(design[remaining], coefficients)
        predictions[remaining] = np.exp(raw) if log_output else raw

    predictions = np.clip(
        predictions,
        cfg.minimum_reported_volatility,
        cfg.maximum_reported_volatility,
    )
    if not np.all(np.isfinite(predictions)) or np.any(predictions <= 0.0):
        raise ValueError("HAR predictions are incomplete.")
    return predictions


def _fit_garch_parameters(returns: np.ndarray) -> tuple[float, float, float]:
    returns = np.asarray(returns, dtype=float)
    unconditional = max(float(np.mean(returns**2)), 1e-8)
    best = (unconditional * 0.05, 0.05, 0.90)
    best_nll = float("inf")
    for alpha in (0.03, 0.05, 0.08, 0.12, 0.16):
        for beta in (0.70, 0.80, 0.86, 0.90, 0.94):
            if alpha + beta >= 0.985:
                continue
            omega = unconditional * (1.0 - alpha - beta)
            variance = unconditional
            nll = 0.0
            for value in returns:
                variance = omega + alpha * value**2 + beta * variance
                nll += math.log(variance) + value**2 / variance
            if nll < best_nll:
                best_nll = nll
                best = (omega, alpha, beta)
    return best


def fit_garch_baseline(
    dataset: MarketDataset,
    split: PreparedSplit,
    cfg: Config,
) -> np.ndarray:
    """Dependency-free GARCH(1,1) QMLE grid baseline."""
    return_column = dataset.feature_names.index("last_return")
    last_returns = dataset.features[:, return_column].astype(float)
    ticker_ids = dataset.metadata["ticker_id"].to_numpy(dtype=int)
    pooled_parameters = _fit_garch_parameters(last_returns[split.train_indices])
    pooled_variance = max(
        float(np.var(last_returns[split.train_indices])), 1e-8
    )
    predictions = np.empty(len(last_returns), dtype=float)

    for ticker_id, ticker in enumerate(cfg.target_tickers):
        rows = np.flatnonzero(ticker_ids == ticker_id)
        dates = pd.to_datetime(dataset.metadata.iloc[rows]["origin_date"])
        rows = rows[np.argsort(dates.to_numpy())]
        ticker_train = split.train_indices[
            ticker_ids[split.train_indices] == ticker_id
        ]
        if ticker in cfg.training_tickers:
            parameters = _fit_garch_parameters(last_returns[ticker_train])
            variance = max(float(np.var(last_returns[ticker_train])), 1e-8)
        else:
            parameters = pooled_parameters
            variance = pooled_variance
        omega, alpha, beta = parameters
        for row_index in rows:
            variance = (
                omega
                + alpha * last_returns[row_index] ** 2
                + beta * variance
            )
            predictions[row_index] = math.sqrt(cfg.annualisation * variance)
    return np.clip(
        predictions,
        cfg.minimum_reported_volatility,
        cfg.maximum_reported_volatility,
    )


def fit_stronger_baselines(
    dataset: MarketDataset,
    split: PreparedSplit,
    cfg: Config,
) -> dict[str, object]:
    predictions: dict[str, object] = {}
    complexity: dict[str, dict[str, float | int]] = {}
    specifications = (
        ("log_har", "Log-HAR transferable", 4),
        ("level_har", "Level HAR transferable", 4),
        ("log_har_leverage", "Log-HAR leverage transferable", 6),
    )
    coefficient_sets = len(cfg.training_tickers) + 1
    for key, model_name, coefficients_per_fit in specifications:
        started = time.perf_counter()
        predictions[key] = fit_ticker_specific_har(
            dataset, split, cfg, key
        )
        complexity[model_name] = {
            "model_coefficients": coefficient_sets * coefficients_per_fit,
            "fit_and_forecast_seconds": time.perf_counter() - started,
        }
    started = time.perf_counter()
    predictions["garch"] = fit_garch_baseline(dataset, split, cfg)
    complexity["GARCH(1,1)"] = {
        "model_coefficients": coefficient_sets * 3,
        "fit_and_forecast_seconds": time.perf_counter() - started,
    }
    predictions["_complexity"] = complexity
    return predictions


def configure_har_forecasting_problem(
    dataset: MarketDataset,
    split: PreparedSplit,
    har_predictions: np.ndarray,
    cfg: Config,
) -> None:
    """Attach a safe HAR feature and select direct versus additive residual target."""
    split.har_predictions = har_predictions.astype(np.float32)
    log_har = np.log(np.clip(har_predictions, EPS, None))
    train_values = log_har[split.train_indices]
    mean = float(np.mean(train_values))
    scale = max(float(np.std(train_values)), 1e-6)
    if cfg.include_har_feature:
        standardised = np.clip(
            (log_har - mean) / scale,
            -cfg.standardised_clip,
            cfg.standardised_clip,
        ).astype(np.float32)
    else:
        # Preserve canonical width, fan-in, parameter count and initialization
        # for the matched HAR-input ablation.
        standardised = np.zeros(len(har_predictions), dtype=np.float32)
    split.context = np.column_stack([
        split.context, standardised,
    ]).astype(np.float32)
    split.context_feature_names.append(
        "chronological_oof_log_har_forecast"
    )

    if cfg.forecast_mode == "har_residual":
        actual = np.exp(dataset.targets_log_vol)
        split.model_targets = (actual - har_predictions).astype(np.float32)
    else:
        split.model_targets = dataset.targets_log_vol.astype(np.float32).copy()
    append_audit_check(
        split,
        "HAR forecasts align one-to-one with dataset rows",
        len(har_predictions) == len(dataset.metadata)
        and bool(np.all(np.isfinite(har_predictions))),
        len(har_predictions),
        len(dataset.metadata),
    )
    append_audit_check(
        split,
        "HAR coefficients and HAR-feature scaling use training rows only",
        pd.to_datetime(
            dataset.metadata.iloc[split.train_indices]["target_end_date"]
        ).max() <= split.train_cutoff,
        pd.to_datetime(
            dataset.metadata.iloc[split.train_indices]["target_end_date"]
        ).max(),
        split.train_cutoff,
    )


def softmax_numpy(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


@torch.no_grad()
def collect_ensemble_outputs(
    models: list[RegimeFlowMixture],
    dataset: MarketDataset,
    split: PreparedSplit,
    indices: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    # Each seed receives the same fixed Monte-Carlo budget. A single-seed
    # ablation therefore reuses exactly that member's stream and distribution.
    samples_by_model = []
    logits_by_model = []
    log_probabilities_by_model = []
    samples_per_model = max(32, cfg.prediction_samples)

    for model, model_seed in zip(models, cfg.ensemble_seeds):
        set_seed(cfg.prediction_seed + int(model_seed))
        model_samples = []
        model_logits = []
        model_log_probabilities = []

        for start in range(0, len(indices), cfg.prediction_batch_size):
            batch_indices = indices[start:start + cfg.prediction_batch_size]
            context = torch.from_numpy(
                split.context[batch_indices]
            ).float().to(DEVICE)
            target = torch.from_numpy(
                split.model_targets[batch_indices]
            ).float().unsqueeze(1).to(DEVICE)
            ticker_ids = torch.from_numpy(
                dataset.metadata.iloc[batch_indices]["ticker_id"].to_numpy(
                    dtype=np.int64, copy=True
                )
            ).long().to(DEVICE)

            bundle = model.forward_bundle(
                context, target, ticker_ids=ticker_ids
            )
            log_samples = model.sample_from_bundle(bundle, samples_per_model)
            if bundle.mixture_log_probability is None:
                raise RuntimeError("Evaluation bundle lacks log probability.")

            model_samples.append(log_samples.cpu().numpy())
            model_logits.append(bundle.logits.cpu().numpy())
            model_log_probabilities.append(
                bundle.mixture_log_probability.cpu().numpy()
            )

        samples_by_model.append(np.concatenate(model_samples, axis=0))
        logits_by_model.append(np.concatenate(model_logits, axis=0))
        log_probabilities_by_model.append(
            np.concatenate(model_log_probabilities, axis=0)
        )

    log_samples = np.concatenate(samples_by_model, axis=1)
    logits = np.stack(logits_by_model, axis=0)
    stacked_log_probabilities = np.stack(
        log_probabilities_by_model,
        axis=1,
    )
    maximum = np.max(stacked_log_probabilities, axis=1, keepdims=True)
    ensemble_log_probability = maximum[:, 0] + np.log(
        np.mean(
            np.exp(stacked_log_probabilities - maximum),
            axis=1,
        ) + EPS
    )

    return {
        "log_samples": log_samples,
        "model_log_samples": np.stack(samples_by_model, axis=0),
        "model_logits": logits,
        "model_log_probabilities": np.stack(
            log_probabilities_by_model, axis=0
        ),
        "uncalibrated_log_probability": ensemble_log_probability,
    }

# %% [markdown cell 21]
# ## Calibration-only interval scale and regime temperature

# %% [cell 22]
def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    score = upper - lower
    score += (
        2.0 / alpha
        * (lower - actual)
        * (actual < lower)
    )
    score += (
        2.0 / alpha
        * (actual - upper)
        * (actual > upper)
    )
    return score


def model_samples_to_volatility(
    model_samples: np.ndarray,
    har_predictions: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if cfg.forecast_mode == "har_residual":
        values = har_predictions[:, None] + model_samples
    else:
        values = np.exp(model_samples)
    return np.clip(
        values,
        cfg.minimum_reported_volatility,
        cfg.maximum_reported_volatility,
    )


def calibrate_interval_scale(
    model_samples: np.ndarray,
    actual_log_volatility: np.ndarray,
    har_predictions: np.ndarray,
    cfg: Config,
) -> float:
    median = np.median(
        model_samples,
        axis=1,
        keepdims=True,
    )
    actual = np.exp(actual_log_volatility)

    candidates = np.linspace(
        cfg.interval_scale_min,
        cfg.interval_scale_max,
        cfg.interval_scale_candidates,
    )

    best_scale = 1.0
    best_score = float("inf")

    for scale in candidates:
        adjusted = (
            median
            + scale * (model_samples - median)
        )
        samples = model_samples_to_volatility(
            adjusted,
            har_predictions,
            cfg,
        )

        q05, q25, q75, q95 = np.quantile(
            samples,
            [0.05, 0.25, 0.75, 0.95],
            axis=1,
        )

        score = np.mean(
            interval_score(
                q05,
                q95,
                actual,
                alpha=0.10,
            )
            + interval_score(
                q25,
                q75,
                actual,
                alpha=0.50,
            )
        )

        if score < best_score:
            best_score = score
            best_scale = float(scale)

    print(
        f"Calibration interval scale: {best_scale:.3f}"
    )
    return best_scale


def select_har_blend_weight(
    model_samples: np.ndarray,
    har_predictions: np.ndarray,
    actual_log_volatility: np.ndarray,
    interval_scale: float,
    cfg: Config,
) -> float:
    """Choose the SigFlow share using calibration rows only."""
    centre = np.median(model_samples, axis=1, keepdims=True)
    scaled = centre + interval_scale * (model_samples - centre)
    sigflow_samples = model_samples_to_volatility(
        scaled, har_predictions, cfg
    )
    actual = np.exp(actual_log_volatility)
    best_weight = 1.0
    best_score = float("inf")
    for weight in np.linspace(0.0, 1.0, cfg.blend_weight_candidates):
        blended = weight * sigflow_samples + (
            1.0 - weight
        ) * har_predictions[:, None]
        if cfg.blend_selection_metric == "mae":
            score = float(np.mean(
                np.abs(np.median(blended, axis=1) - actual)
            ))
        elif cfg.blend_selection_metric == "qlike":
            score = qlike(
                actual,
                np.sqrt(np.mean(blended**2, axis=1)),
            )
        else:
            raise ValueError("blend_selection_metric must be mae or qlike.")
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    print(
        f"Calibration-selected SigFlow blend weight: {best_weight:.3f} "
        f"({cfg.blend_selection_metric}={best_score:.6f})"
    )
    return best_weight


@dataclass
class IntervalTailCalibration:
    scale: float
    corrections: dict[str, dict[str, tuple[float, float]]]


def finite_sample_conformal_quantile(
    values: np.ndarray,
    miscoverage: float,
) -> float:
    """Higher order statistic at ceil((n+1)(1-alpha)), capped at n."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) == 0:
        raise ValueError("Conformal calibration group is empty.")
    rank = min(
        len(ordered),
        int(math.ceil((len(ordered) + 1) * (1.0 - miscoverage))),
    )
    return float(ordered[max(rank - 1, 0)])


def fit_interval_tail_calibration(
    model_samples: np.ndarray,
    har_predictions: np.ndarray,
    actual_log_volatility: np.ndarray,
    metadata: pd.DataFrame,
    regime_probabilities: np.ndarray,
    interval_scale: float,
    cfg: Config,
) -> IntervalTailCalibration:
    centre = np.median(model_samples, axis=1, keepdims=True)
    scaled = centre + interval_scale * (model_samples - centre)
    samples = model_samples_to_volatility(scaled, har_predictions, cfg)
    actual = np.exp(actual_log_volatility)
    labels = regime_probabilities.argmax(axis=1)
    tickers = metadata["ticker"].to_numpy(dtype=str)
    masks: dict[str, np.ndarray] = {
        "pooled": np.ones(len(actual), dtype=bool)
    }
    if cfg.ticker_interval_calibration:
        for ticker in sorted(set(tickers)):
            masks[f"ticker:{ticker}"] = tickers == ticker
    if cfg.regime_conditional_interval_calibration:
        for regime in range(cfg.regimes):
            masks[f"regime:{regime}"] = labels == regime
        if cfg.ticker_interval_calibration:
            for ticker in sorted(set(tickers)):
                for regime in range(cfg.regimes):
                    masks[f"ticker:{ticker}|regime:{regime}"] = (
                        (tickers == ticker) & (labels == regime)
                    )

    corrections: dict[str, dict[str, tuple[float, float]]] = {}
    for key, mask in masks.items():
        minimum = 1 if key == "pooled" else cfg.minimum_group_calibration_rows
        if int(mask.sum()) < minimum:
            continue
        group: dict[str, tuple[float, float]] = {}
        for level in cfg.interval_coverage_levels:
            alpha = (1.0 - level) / 2.0
            lower = np.quantile(samples[mask], alpha, axis=1)
            upper = np.quantile(samples[mask], 1.0 - alpha, axis=1)
            # Separate one-sided conformal corrections. Non-negative clipping
            # avoids selecting artificially narrow intervals on finite samples.
            lower_correction = max(0.0, finite_sample_conformal_quantile(
                lower - actual[mask], alpha
            ))
            upper_correction = max(0.0, finite_sample_conformal_quantile(
                actual[mask] - upper, alpha
            ))
            if not cfg.asymmetric_interval_calibration:
                shared = max(lower_correction, upper_correction)
                lower_correction = shared
                upper_correction = shared
            group[str(int(round(100.0 * level)))] = (
                lower_correction,
                upper_correction,
            )
        corrections[key] = group
    return IntervalTailCalibration(scale=interval_scale, corrections=corrections)


def apply_interval_tail_calibration(
    frame: pd.DataFrame,
    calibration: IntervalTailCalibration,
    cfg: Config,
) -> pd.DataFrame:
    interval_columns = {
        50: ("predicted_q25_vol", "predicted_q75_vol"),
        80: ("predicted_q10_vol", "predicted_q90_vol"),
        90: ("predicted_q05_vol", "predicted_q95_vol"),
        95: ("predicted_q025_vol", "predicted_q975_vol"),
    }
    for row_index, row in frame.iterrows():
        ticker = str(row["ticker"])
        regime = int(row["predicted_regime"])
        candidate_keys = (
            f"ticker:{ticker}|regime:{regime}",
            f"ticker:{ticker}",
            f"regime:{regime}",
            "pooled",
        )
        key = next(
            (candidate for candidate in candidate_keys
             if candidate in calibration.corrections),
            "pooled",
        )
        frame.loc[row_index, "interval_calibration_group"] = key
        for level, (lower_column, upper_column) in interval_columns.items():
            lower_correction, upper_correction = calibration.corrections.get(
                key, calibration.corrections.get("pooled", {})
            ).get(str(level), (0.0, 0.0))
            frame.loc[row_index, lower_column] = max(
                cfg.minimum_reported_volatility,
                float(row[lower_column]) - lower_correction,
            )
            frame.loc[row_index, upper_column] = min(
                cfg.maximum_reported_volatility,
                float(row[upper_column]) + upper_correction,
            )
        # Enforce nested central intervals after group-specific corrections.
        ordered_intervals = [50, 80, 90, 95]
        for narrower, wider in zip(
            ordered_intervals[:-1], ordered_intervals[1:]
        ):
            narrow_lower, narrow_upper = interval_columns[narrower]
            wide_lower, wide_upper = interval_columns[wider]
            frame.loc[row_index, wide_lower] = min(
                float(frame.loc[row_index, wide_lower]),
                float(frame.loc[row_index, narrow_lower]),
            )
            frame.loc[row_index, wide_upper] = max(
                float(frame.loc[row_index, wide_upper]),
                float(frame.loc[row_index, narrow_upper]),
            )
    return frame


def multiclass_log_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    selected = probabilities[
        np.arange(len(labels)),
        labels,
    ]
    return float(
        -np.mean(np.log(selected + EPS))
    )


def calibrate_regime_temperature(
    model_logits: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
) -> float:
    candidates = np.linspace(
        cfg.temperature_min,
        cfg.temperature_max,
        cfg.temperature_candidates,
    )

    best_temperature = 1.0
    best_loss = float("inf")

    for temperature in candidates:
        probabilities = np.mean(
            softmax_numpy(
                model_logits,
                temperature,
            ),
            axis=0,
        )
        loss = multiclass_log_loss(
            probabilities,
            labels,
        )

        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)

    print(
        f"Calibration regime temperature: "
        f"{best_temperature:.3f}"
    )
    return best_temperature


def calibrated_probabilities(
    model_logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    return np.mean(
        softmax_numpy(
            model_logits,
            temperature,
        ),
        axis=0,
    )


@torch.no_grad()
def collect_regime_probabilities(
    models: list[RegimeFlowMixture],
    dataset: MarketDataset,
    split: PreparedSplit,
    indices: np.ndarray,
    temperature: float,
    cfg: Config,
) -> np.ndarray:
    logits_by_model = []
    for model in models:
        batches = []
        for start in range(0, len(indices), cfg.prediction_batch_size):
            batch_indices = indices[start:start + cfg.prediction_batch_size]
            context = torch.from_numpy(
                split.context[batch_indices]
            ).float().to(DEVICE)
            ticker_ids = torch.from_numpy(
                dataset.metadata.iloc[batch_indices]["ticker_id"].to_numpy(
                    dtype=np.int64, copy=True
                )
            ).long().to(DEVICE)
            bundle = model.forward_bundle(
                context, ticker_ids=ticker_ids
            )
            batches.append(bundle.logits.cpu().numpy())
        logits_by_model.append(np.concatenate(batches, axis=0))
    return calibrated_probabilities(
        np.stack(logits_by_model, axis=0), temperature
    )


def build_full_regime_diagnostics(
    models: list[RegimeFlowMixture],
    dataset: MarketDataset,
    split: PreparedSplit,
    regime_temperature: float,
    cfg: Config,
    sections_to_report: tuple[str, ...] = (
        "train", "validation", "calibration", "test"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sections = {
        "train": split.train_indices,
        "validation": split.validation_indices,
        "calibration": split.calibration_indices,
        "test": split.test_indices,
    }
    diagnostic_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for section in sections_to_report:
        indices = sections[section]
        probabilities = collect_regime_probabilities(
            models, dataset, split, indices, regime_temperature, cfg
        )
        section_frame = dataset.metadata.iloc[indices][
            ["ticker", "ticker_id"]
        ].copy().reset_index(drop=True)
        section_frame["true_regime"] = split.regime_labels[indices]
        section_frame["predicted_regime"] = probabilities.argmax(axis=1)
        for regime_id, regime_name in enumerate(REGIME_NAMES):
            section_frame[f"probability_{regime_name.lower()}"] = probabilities[
                :, regime_id
            ]

        groups = [("all", section_frame)] + list(
            section_frame.groupby("ticker", sort=True)
        )
        for ticker, selected in groups:
            labels = selected["true_regime"].to_numpy(dtype=int)
            predicted = selected["predicted_regime"].to_numpy(dtype=int)
            row: dict[str, object] = {
                "section": section,
                "ticker": str(ticker),
                "observations": len(selected),
            }
            for regime_id, regime_name in enumerate(REGIME_NAMES):
                slug = regime_name.lower()
                row[f"true_count_{slug}"] = int(np.sum(labels == regime_id))
                row[f"predicted_count_{slug}"] = int(
                    np.sum(predicted == regime_id)
                )
                row[f"average_probability_{slug}"] = float(
                    selected[f"probability_{slug}"].mean()
                )
            diagnostic_rows.append(row)
            matrix = confusion_matrix_numpy(
                labels, predicted, cfg.regimes
            )
            for actual_id, actual_name in enumerate(REGIME_NAMES):
                for predicted_id, predicted_name in enumerate(REGIME_NAMES):
                    confusion_rows.append({
                        "section": section,
                        "ticker": str(ticker),
                        "actual_regime": actual_name,
                        "predicted_regime": predicted_name,
                        "count": int(matrix[actual_id, predicted_id]),
                    })
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(confusion_rows)

# %% [markdown cell 23]
# ## Predictions and past-only regime smoothing

# %% [cell 24]
def empirical_crps_numpy(
    samples: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    first_term = np.mean(
        np.abs(samples - actual[:, None]),
        axis=1,
    )
    sorted_samples = np.sort(samples, axis=1)
    sample_count = samples.shape[1]
    ranks = np.arange(1, sample_count + 1, dtype=float)
    coefficients = 2.0 * ranks - sample_count - 1.0
    second_term = (
        sorted_samples * coefficients[None, :]
    ).sum(axis=1) / (sample_count**2)
    return first_term - second_term


def regime_probability_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_{name.lower()}"
        for name in REGIME_NAMES
    ]


def apply_regime_smoothing(
    predictions: pd.DataFrame,
    split: PreparedSplit,
) -> pd.DataFrame:
    predictions = predictions.copy()
    raw_columns = regime_probability_columns("probability")
    smoothed_columns = regime_probability_columns("smoothed_probability")
    for column in smoothed_columns:
        predictions[column] = 0.0

    for ticker_id, ticker_rows in predictions.groupby("ticker_id", sort=False):
        ordered = ticker_rows.sort_values("origin_date")
        posterior = split.initial_regime_probabilities[
            int(ticker_id)
        ].astype(float).copy()
        transition = split.transition_matrices[int(ticker_id)]

        for row_index in ordered.index:
            gate_probability = predictions.loc[
                row_index,
                raw_columns,
            ].to_numpy(dtype=float)
            # Only the training-derived transition, previous filtered state,
            # and current model probability are used.
            prior = posterior @ transition
            posterior = prior * gate_probability
            posterior /= posterior.sum() + EPS
            predictions.loc[row_index, smoothed_columns] = posterior

    predictions["smoothed_regime"] = predictions[
        smoothed_columns
    ].to_numpy().argmax(axis=1)
    predictions["smoothed_regime_name"] = REGIME_NAMES[
        predictions["smoothed_regime"].to_numpy(dtype=int)
    ]
    return predictions


def tolerance_percent_label(tolerance: float) -> int:
    return int(round(100.0 * tolerance))


def add_detailed_result_columns(
    frame: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    frame = frame.copy()
    actual = frame["actual_vol"].to_numpy(dtype=float)
    denominator = np.maximum(np.abs(actual), EPS)

    for model_name, prediction_column in MODEL_COLUMNS.items():
        prefix = MODEL_RESULT_PREFIXES[model_name]
        predicted = frame[prediction_column].to_numpy(dtype=float)
        signed_error = predicted - actual
        absolute_error = np.abs(signed_error)
        absolute_relative_error = absolute_error / denominator
        frame[f"{prefix}_signed_error"] = signed_error
        frame[f"{prefix}_absolute_error"] = absolute_error
        frame[f"{prefix}_absolute_percentage_error"] = (
            100.0 * absolute_relative_error
        )
        frame[f"{prefix}_squared_error"] = signed_error**2
        qlike_prediction = frame[
            QLIKE_MODEL_COLUMNS[model_name]
        ].to_numpy(dtype=float)
        frame[f"{prefix}_qlike_loss"] = qlike_loss_values(
            actual, qlike_prediction
        )
        for tolerance in cfg.success_relative_tolerances:
            label = tolerance_percent_label(tolerance)
            frame[f"{prefix}_within_{label}pct"] = (
                absolute_relative_error <= tolerance
            )

    primary_label = tolerance_percent_label(cfg.primary_success_tolerance)
    frame["success_within_20pct"] = frame["sigflow_within_20pct"]
    frame["success_within_configured_tolerance"] = frame[
        f"sigflow_within_{primary_label}pct"
    ]
    # Consolidate the wide result table before adding interval and diagnostic
    # columns. This keeps pandas from emitting fragmentation warnings during
    # normal research runs without changing any values.
    frame = frame.copy()
    interval_columns = {
        50: ("predicted_q25_vol", "predicted_q75_vol"),
        80: ("predicted_q10_vol", "predicted_q90_vol"),
        90: ("predicted_q05_vol", "predicted_q95_vol"),
        95: ("predicted_q025_vol", "predicted_q975_vol"),
    }
    for level, (lower_column, upper_column) in interval_columns.items():
        lower = frame[lower_column].to_numpy(dtype=float)
        upper = frame[upper_column].to_numpy(dtype=float)
        frame[f"inside_{level}pct_interval"] = (actual >= lower) & (actual <= upper)
        frame[f"below_{level}pct_interval"] = actual < lower
        frame[f"above_{level}pct_interval"] = actual > upper
        frame[f"interval_width_{level}"] = upper - lower
    frame["raw_regime_correct"] = (
        frame["predicted_regime"].to_numpy(dtype=int)
        == frame["regime"].to_numpy(dtype=int)
    )
    if cfg.apply_regime_smoothing:
        frame["smoothed_regime_correct"] = (
            frame["smoothed_regime"].to_numpy(dtype=int)
            == frame["regime"].to_numpy(dtype=int)
        )

    recent = frame["rolling_vol_baseline"].to_numpy(dtype=float)
    frame["actual_direction_vs_recent"] = np.sign(actual - recent).astype(int)
    frame["predicted_direction_vs_recent"] = np.sign(
        frame["predicted_median_vol"].to_numpy(dtype=float) - recent
    ).astype(int)
    frame["direction_correct_vs_recent"] = (
        frame["actual_direction_vs_recent"]
        == frame["predicted_direction_vs_recent"]
    )

    for baseline_prefix in (
        "rolling", "ewma", "har", "level_har", "har_leverage", "garch", "blend"
    ):
        frame[f"sigflow_beats_{baseline_prefix}_absolute_error"] = (
            frame["sigflow_absolute_error"]
            < frame[f"{baseline_prefix}_absolute_error"]
        )

    return frame.copy()


def build_prediction_frame(
    ensemble_outputs: dict[str, np.ndarray],
    dataset: MarketDataset,
    split: PreparedSplit,
    indices: np.ndarray,
    interval_scale: float,
    interval_tail_calibration: IntervalTailCalibration,
    regime_temperature: float,
    baseline_predictions: dict[str, np.ndarray],
    blend_weight: float,
    cfg: Config,
    cohort_override: str | None = None,
) -> pd.DataFrame:
    raw_model_samples = ensemble_outputs["log_samples"]
    model_median = np.median(raw_model_samples, axis=1, keepdims=True)
    calibrated_model_samples = model_median + interval_scale * (
        raw_model_samples - model_median
    )
    har_predictions = baseline_predictions["log_har"]
    evaluation_har = har_predictions[indices]
    samples = model_samples_to_volatility(
        calibrated_model_samples,
        evaluation_har,
        cfg,
    )
    actual = np.exp(dataset.targets_log_vol[indices])
    quantiles = np.quantile(
        samples,
        [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975],
        axis=1,
    )
    probabilities = calibrated_probabilities(
        ensemble_outputs["model_logits"],
        regime_temperature,
    )

    frame = dataset.metadata.iloc[indices].copy().reset_index(drop=True)
    frame["dataset_row_index"] = np.asarray(indices, dtype=int)
    frame["horizon"] = cfg.horizon
    if cohort_override is not None:
        frame["evaluation_cohort"] = cohort_override
    frame["predicted_mean_vol"] = samples.mean(axis=1)
    # The median is the Bayes action for absolute error. QLIKE instead
    # requires expected conditional variance, reported back in vol units.
    frame["predicted_qlike_vol"] = np.sqrt(
        np.mean(samples**2, axis=1)
    )
    for label, values in zip(
        ("q025", "q05", "q10", "q25", "median", "q75", "q90", "q95", "q975"),
        quantiles,
    ):
        frame[f"predicted_{label}_vol"] = values
    frame["model_space_uncalibrated_negative_log_likelihood"] = -ensemble_outputs[
        "uncalibrated_log_probability"
    ]
    # Backward-compatible name is explicit that no density calibration occurs.
    frame["uncalibrated_negative_log_likelihood"] = frame[
        "model_space_uncalibrated_negative_log_likelihood"
    ]
    frame["crps"] = empirical_crps_numpy(samples, actual)
    frame["pit"] = np.mean(samples <= actual[:, None], axis=1)
    frame["log_har_baseline"] = evaluation_har
    frame["har_style_baseline"] = evaluation_har
    frame["level_har_baseline"] = baseline_predictions["level_har"][indices]
    frame["log_har_leverage_baseline"] = baseline_predictions[
        "log_har_leverage"
    ][indices]
    frame["garch_baseline"] = baseline_predictions["garch"][indices]
    blended_samples = (
        blend_weight * samples
        + (1.0 - blend_weight) * evaluation_har[:, None]
    )
    frame["sigflow_har_blend"] = np.median(blended_samples, axis=1)
    frame["sigflow_har_blend_qlike"] = np.sqrt(
        np.mean(blended_samples**2, axis=1)
    )
    frame["sigflow_har_blend_crps"] = empirical_crps_numpy(
        blended_samples, actual
    )
    frame["har_blend_weight"] = blend_weight
    frame["har_fit_scope"] = np.where(
        frame["ticker"].isin(cfg.training_tickers),
        "ticker_training_only",
        "pooled_training_tickers_only",
    )

    model_log_samples = ensemble_outputs["model_log_samples"]
    for model_index, seed in enumerate(cfg.ensemble_seeds):
        seed_logs = model_log_samples[model_index]
        seed_median = np.median(seed_logs, axis=1, keepdims=True)
        seed_logs = seed_median + interval_scale * (
            seed_logs - seed_median
        )
        seed_samples = model_samples_to_volatility(
            seed_logs,
            evaluation_har,
            cfg,
        )
        frame[f"seed_{seed}_median_vol"] = np.median(
            seed_samples, axis=1
        )
        frame[f"seed_{seed}_qlike_vol"] = np.sqrt(
            np.mean(seed_samples**2, axis=1)
        )
        frame[f"seed_{seed}_crps"] = empirical_crps_numpy(
            seed_samples, actual
        )
        frame[f"seed_{seed}_nll"] = -ensemble_outputs[
            "model_log_probabilities"
        ][model_index]
        seed_q05, seed_q95 = np.quantile(
            seed_samples, [0.05, 0.95], axis=1
        )
        frame[f"seed_{seed}_coverage_90"] = (
            (actual >= seed_q05) & (actual <= seed_q95)
        )

    for regime_id, regime_name in enumerate(REGIME_NAMES):
        frame[f"probability_{regime_name.lower()}"] = probabilities[:, regime_id]
    frame["predicted_regime"] = probabilities.argmax(axis=1)
    frame["predicted_regime_name"] = REGIME_NAMES[
        frame["predicted_regime"].to_numpy(dtype=int)
    ]
    frame = apply_interval_tail_calibration(
        frame, interval_tail_calibration, cfg
    )

    if cfg.apply_regime_smoothing:
        frame = apply_regime_smoothing(frame, split)
    return add_detailed_result_columns(frame, cfg)

# %% [markdown cell 25]
# ## Metric-correct accuracy and dependence-aware significance
# 
# Median volatility is used for MAE and tolerance accuracy. QLIKE uses `sqrt(E[volatility²])`, the correct action for conditional variance loss. Inference includes paired moving-date blocks at 20/40/60/120 sessions, a prespecified 20-session primary block, a two-way ticker/date bootstrap, paired MAE/RMSE/QLIKE/CRPS/within-20% differences, Holm-corrected secondary tests, and per-seed stability. More bootstrap repetitions stabilise estimates; they do not manufacture power.

# %% [cell 26]
MODEL_COLUMNS = {
    "SigFlow v4": "predicted_median_vol",
    "SigFlow + Log-HAR blend": "sigflow_har_blend",
    "Rolling volatility": "rolling_vol_baseline",
    "EWMA volatility": "ewma_vol_baseline",
    "Log-HAR transferable": "log_har_baseline",
    "Level HAR transferable": "level_har_baseline",
    "Log-HAR leverage transferable": "log_har_leverage_baseline",
    "GARCH(1,1)": "garch_baseline",
}
MODEL_RESULT_PREFIXES = {
    "SigFlow v4": "sigflow",
    "SigFlow + Log-HAR blend": "blend",
    "Rolling volatility": "rolling",
    "EWMA volatility": "ewma",
    "Log-HAR transferable": "har",
    "Level HAR transferable": "level_har",
    "Log-HAR leverage transferable": "har_leverage",
    "GARCH(1,1)": "garch",
}
QLIKE_MODEL_COLUMNS = {
    "SigFlow v4": "predicted_qlike_vol",
    "SigFlow + Log-HAR blend": "sigflow_har_blend_qlike",
    "Rolling volatility": "rolling_vol_baseline",
    "EWMA volatility": "ewma_vol_baseline",
    "Log-HAR transferable": "log_har_baseline",
    "Level HAR transferable": "level_har_baseline",
    "Log-HAR leverage transferable": "log_har_leverage_baseline",
    "GARCH(1,1)": "garch_baseline",
}


def qlike_loss_values(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    actual_variance = np.maximum(actual**2, EPS)
    predicted_variance = np.maximum(predicted**2, EPS)
    ratio = actual_variance / predicted_variance
    return ratio - np.log(ratio) - 1.0


def qlike(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(np.mean(qlike_loss_values(actual, predicted)))


def point_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    cfg: Config,
    qlike_predicted: np.ndarray | None = None,
) -> dict[str, float]:
    if qlike_predicted is None:
        qlike_predicted = predicted
    error = predicted - actual
    absolute_error = np.abs(error)
    absolute_relative_error = absolute_error / np.maximum(np.abs(actual), EPS)
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > EPS and np.std(predicted) > EPS
        else float("nan")
    )
    total_square = float(np.sum((actual - np.mean(actual)) ** 2))
    actual_std = float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0
    predicted_std = float(np.std(predicted, ddof=1)) if len(predicted) > 1 else 0.0
    calibration_slope = (
        float(np.cov(actual, predicted, ddof=1)[0, 1] / max(np.var(predicted, ddof=1), EPS))
        if len(actual) > 1 else float("nan")
    )
    best_lag = 0
    best_lagged_correlation = correlation
    for lag in range(-5, 6):
        if lag < 0:
            lag_actual, lag_predicted = actual[-lag:], predicted[:lag]
        elif lag > 0:
            lag_actual, lag_predicted = actual[:-lag], predicted[lag:]
        else:
            lag_actual, lag_predicted = actual, predicted
        if len(lag_actual) < 3 or np.std(lag_actual) <= EPS or np.std(lag_predicted) <= EPS:
            candidate = float("nan")
        else:
            candidate = float(np.corrcoef(lag_actual, lag_predicted)[0, 1])
        if np.isfinite(candidate) and (
            not np.isfinite(best_lagged_correlation)
            or candidate > best_lagged_correlation
        ):
            best_lag = lag
            best_lagged_correlation = candidate

    metrics = {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "mape_percent": float(100.0 * np.mean(absolute_relative_error)),
        "median_ape_percent": float(100.0 * np.median(absolute_relative_error)),
        "smape_percent": float(100.0 * np.mean(
            2.0 * absolute_error
            / np.maximum(np.abs(actual) + np.abs(predicted), EPS)
        )),
        "correlation": correlation,
        "actual_standard_deviation": actual_std,
        "predicted_standard_deviation": predicted_std,
        "prediction_to_actual_std_ratio": predicted_std / max(actual_std, EPS),
        "calibration_slope_actual_on_predicted": calibration_slope,
        "best_prediction_lag_sessions": best_lag,
        "best_lagged_correlation": best_lagged_correlation,
        "r_squared": (
            float(1.0 - np.sum(error**2) / total_square)
            if total_square > EPS else float("nan")
        ),
        "qlike": qlike(actual, qlike_predicted),
    }
    for tolerance in cfg.success_relative_tolerances:
        label = tolerance_percent_label(tolerance)
        metrics[f"within_{label}pct_accuracy"] = float(
            np.mean(absolute_relative_error <= tolerance)
        )
    return metrics


def confusion_matrix_numpy(
    actual: np.ndarray,
    predicted: np.ndarray,
    classes: int,
) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=int)
    for actual_value, predicted_value in zip(actual, predicted):
        matrix[int(actual_value), int(predicted_value)] += 1
    return matrix


def regime_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    matrix = confusion_matrix_numpy(labels, predictions, classes)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, EPS)
    one_hot = np.eye(classes)[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    metrics = {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "brier_score": brier,
        "log_loss": multiclass_log_loss(probabilities, labels),
    }
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        metrics[f"recall_{regime_name.lower()}"] = float(recall[regime_id])
    return metrics, matrix


def evaluation_slices(
    predictions: pd.DataFrame,
    include_tickers: bool = True,
):
    yield {
        "group_level": "pooled",
        "group": "Pooled",
        "cohort": "all",
        "ticker": "all",
        "rows": predictions,
    }
    for cohort, cohort_rows in predictions.groupby(
        "evaluation_cohort", sort=True
    ):
        yield {
            "group_level": "cohort",
            "group": cohort,
            "cohort": cohort,
            "ticker": "all",
            "rows": cohort_rows,
        }
        for sector, sector_rows in cohort_rows.groupby("sector", sort=True):
            yield {
                "group_level": "cohort_sector",
                "group": f"{cohort}::{sector}",
                "cohort": cohort,
                "ticker": "all",
                "rows": sector_rows,
            }
        if include_tickers:
            for ticker, ticker_rows in cohort_rows.groupby("ticker", sort=True):
                yield {
                    "group_level": "cohort_ticker",
                    "group": f"{cohort}::{ticker}",
                    "cohort": cohort,
                    "ticker": ticker,
                    "rows": ticker_rows,
                }


def evaluate_point_models(
    predictions: pd.DataFrame,
    group_information: dict[str, object],
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    actual = predictions["actual_vol"].to_numpy(dtype=float)
    for model_name, column in MODEL_COLUMNS.items():
        row = {
            key: group_information[key]
            for key in ("group_level", "group", "cohort", "ticker")
        }
        row.update({
            "model": model_name,
            "seed": (
                "ensemble"
                if model_name.startswith("SigFlow")
                else "deterministic"
            ),
            "observations": len(predictions),
        })
        row.update(point_metrics(
            actual,
            predictions[column].to_numpy(dtype=float),
            cfg,
            qlike_predicted=predictions[
                QLIKE_MODEL_COLUMNS[model_name]
            ].to_numpy(dtype=float),
        ))
        row["point_forecast_column"] = column
        row["qlike_forecast_column"] = QLIKE_MODEL_COLUMNS[model_name]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_all_groups(
    predictions: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    tables = []
    for information in evaluation_slices(predictions, include_tickers=True):
        tables.append(evaluate_point_models(
            information["rows"], information, cfg
        ))
    return pd.concat(tables, ignore_index=True)


def build_accuracy_summary(
    predictions: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    for information in evaluation_slices(predictions, include_tickers=True):
        selected = information["rows"]
        row = {
            key: information[key]
            for key in ("group_level", "group", "cohort", "ticker")
        }
        row["observations"] = len(selected)
        row["successful_forecasts_20pct"] = int(
            selected["success_within_20pct"].sum()
        )
        for model_name in MODEL_COLUMNS:
            prefix = MODEL_RESULT_PREFIXES[model_name]
            for tolerance in cfg.success_relative_tolerances:
                label = tolerance_percent_label(tolerance)
                row[f"{prefix}_within_{label}pct_accuracy"] = float(
                    selected[f"{prefix}_within_{label}pct"].mean()
                )
        row["sigflow_direction_accuracy"] = float(
            selected["direction_correct_vs_recent"].mean()
        )
        for level in (50, 80, 90, 95):
            row[f"coverage_{level}"] = float(
                selected[f"inside_{level}pct_interval"].mean()
            )
            row[f"interval_width_{level}"] = float(
                selected[f"interval_width_{level}"].mean()
            )
            row[f"lower_miss_rate_{level}"] = float(
                selected[f"below_{level}pct_interval"].mean()
            )
            row[f"upper_miss_rate_{level}"] = float(
                selected[f"above_{level}pct_interval"].mean()
            )
        row["raw_regime_accuracy"] = float(selected["raw_regime_correct"].mean())
        row["smoothed_regime_accuracy"] = (
            float(selected["smoothed_regime_correct"].mean())
            if cfg.apply_regime_smoothing else float("nan")
        )
        for baseline_prefix in (
            "rolling", "ewma", "har", "level_har", "har_leverage", "garch", "blend"
        ):
            row[f"sigflow_beats_{baseline_prefix}_rate"] = float(
                selected[
                    f"sigflow_beats_{baseline_prefix}_absolute_error"
                ].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_success_summary(
    predictions: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    for (cohort, origin_date), selected in predictions.groupby(
        ["evaluation_cohort", "origin_date"], sort=True
    ):
        successful = selected.loc[
            selected["success_within_20pct"], "ticker"
        ].astype(str).tolist()
        failed = selected.loc[
            ~selected["success_within_20pct"], "ticker"
        ].astype(str).tolist()
        row = {
            "cohort": cohort,
            "origin_date": pd.Timestamp(origin_date),
            "forecast_count": len(selected),
            "success_count_20pct": len(successful),
            "success_rate_20pct": float(
                selected["success_within_20pct"].mean()
            ),
            "any_success_20pct": bool(successful),
            "all_success_20pct": not bool(failed),
            "successful_tickers": ", ".join(successful),
            "failed_tickers": ", ".join(failed),
            "sigflow_mae": float(selected["sigflow_absolute_error"].mean()),
            "sigflow_mape_percent": float(
                selected["sigflow_absolute_percentage_error"].mean()
            ),
            "coverage_90": float(selected["inside_90pct_interval"].mean()),
            "raw_regime_accuracy": float(selected["raw_regime_correct"].mean()),
            "smoothed_regime_accuracy": (
                float(selected["smoothed_regime_correct"].mean())
                if cfg.apply_regime_smoothing else float("nan")
            ),
        }
        for tolerance in cfg.success_relative_tolerances:
            label = tolerance_percent_label(tolerance)
            row[f"within_{label}pct_accuracy"] = float(
                selected[f"sigflow_within_{label}pct"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["origin_date", "cohort"]
    ).reset_index(drop=True)


def probabilistic_summary_by_group(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for information in evaluation_slices(predictions, include_tickers=True):
        selected = information["rows"]
        row = {
            key: information[key]
            for key in ("group_level", "group", "cohort", "ticker")
        }
        row.update({
            "observations": len(selected),
            "model_space_uncalibrated_nll": float(
                selected["uncalibrated_negative_log_likelihood"].mean()
            ),
            "crps": float(selected["crps"].mean()),
        })
        for level in (50, 80, 90, 95):
            row[f"coverage_{level}"] = float(
                selected[f"inside_{level}pct_interval"].mean()
            )
            row[f"interval_width_{level}"] = float(
                selected[f"interval_width_{level}"].mean()
            )
            row[f"lower_miss_rate_{level}"] = float(
                selected[f"below_{level}pct_interval"].mean()
            )
            row[f"upper_miss_rate_{level}"] = float(
                selected[f"above_{level}pct_interval"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def regime_summary_by_group(
    predictions: pd.DataFrame,
    cfg: Config,
    smoothed: bool,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    rows = []
    matrices = {}
    probability_prefix = "smoothed_probability" if smoothed else "probability"
    prediction_column = "smoothed_regime" if smoothed else "predicted_regime"
    pooled_matrix = np.zeros((cfg.regimes, cfg.regimes), dtype=int)

    for information in evaluation_slices(predictions, include_tickers=True):
        selected = information["rows"]
        labels = selected["regime"].to_numpy(dtype=int)
        predicted = selected[prediction_column].to_numpy(dtype=int)
        probabilities = selected[
            regime_probability_columns(probability_prefix)
        ].to_numpy(dtype=float)
        metrics, matrix = regime_metrics(
            labels, predicted, probabilities, cfg.regimes
        )
        row = {
            key: information[key]
            for key in ("group_level", "group", "cohort", "ticker")
        }
        row["observations"] = len(selected)
        row.update(metrics)
        rows.append(row)
        matrices[str(information["group"])] = matrix
        if information["group_level"] == "pooled":
            pooled_matrix = matrix

    return pd.DataFrame(rows), pooled_matrix, matrices


def non_overlapping_phase_metrics(
    predictions: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    tables = []
    for information in evaluation_slices(predictions, include_tickers=False):
        cohort_rows = information["rows"]
        for phase in range(cfg.horizon):
            selected_pieces = []
            for _, ticker_rows in cohort_rows.groupby("ticker", sort=False):
                ordered = ticker_rows.sort_values("origin_date")
                selected_pieces.append(ordered.iloc[phase::cfg.horizon])
            selected = pd.concat(selected_pieces, ignore_index=True)
            phase_information = dict(information)
            phase_information["rows"] = selected
            table = evaluate_point_models(selected, phase_information, cfg)
            table["phase"] = phase
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def moving_block_draw(
    date_count: int,
    requested_block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    effective = min(requested_block_length, date_count)
    possible_starts = np.arange(max(1, date_count - effective + 1))
    selected = []
    while len(selected) < date_count:
        start = int(rng.choice(possible_starts))
        selected.extend(range(start, start + effective))
    return np.asarray(selected[:date_count], dtype=int), effective


def primary_accuracy_metric_name(cfg: Config) -> str:
    return (
        f"within_{tolerance_percent_label(cfg.primary_success_tolerance)}"
        "pct_accuracy"
    )


def resample_plan_hash(
    metadata: dict[str, object],
    *draw_arrays: np.ndarray,
) -> str:
    """Hash both bootstrap draws and the calendar/panel they index."""
    digest = hashlib.sha256()
    digest.update(json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8"))
    for values in draw_arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def paired_difference_arrays(
    predictions: pd.DataFrame,
    baseline_name: str,
    cfg: Config,
) -> dict[str, np.ndarray]:
    actual = predictions["actual_vol"].to_numpy(dtype=float)
    sigflow_point = predictions[MODEL_COLUMNS["SigFlow v4"]].to_numpy(dtype=float)
    sigflow_qlike = predictions[QLIKE_MODEL_COLUMNS["SigFlow v4"]].to_numpy(dtype=float)
    baseline_point = predictions[MODEL_COLUMNS[baseline_name]].to_numpy(dtype=float)
    baseline_qlike = predictions[QLIKE_MODEL_COLUMNS[baseline_name]].to_numpy(dtype=float)
    tolerance = cfg.primary_success_tolerance
    denominator = np.maximum(np.abs(actual), EPS)
    return {
        "mae": np.abs(sigflow_point - actual) - np.abs(baseline_point - actual),
        "crps": (
            predictions["crps"].to_numpy(dtype=float)
            - (
                predictions["sigflow_har_blend_crps"].to_numpy(dtype=float)
                if baseline_name == "SigFlow + Log-HAR blend"
                else np.abs(baseline_point - actual)
            )
        ),
        "qlike": (
            qlike_loss_values(actual, sigflow_qlike)
            - qlike_loss_values(actual, baseline_qlike)
        ),
        primary_accuracy_metric_name(cfg): (
            (np.abs(sigflow_point - actual) / denominator <= tolerance).astype(float)
            - (np.abs(baseline_point - actual) / denominator <= tolerance).astype(float)
        ),
    }


def bootstrap_result_row(
    values: np.ndarray,
    observed: float,
    cohort: str,
    baseline: str,
    metric: str,
    scheme: str,
    requested_block: int,
    effective_block: int,
    date_count: int,
    ticker_count: int,
    repetitions: int,
    cfg: Config,
) -> dict[str, object]:
    higher_is_better = metric == primary_accuracy_metric_name(cfg)
    lower_quantile = cfg.significance_alpha / 2.0
    upper_quantile = 1.0 - lower_quantile
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return {
            "cohort": cohort,
            "comparison": f"SigFlow v4 minus {baseline}",
            "metric": metric,
            "resampling_scheme": scheme,
            "requested_block_length": requested_block,
            "effective_block_length": effective_block,
            "unique_dates": date_count,
            "ticker_clusters": ticker_count,
            "approximate_effective_blocks": date_count / max(effective_block, 1),
            "repetitions": repetitions,
            "observed_difference": observed,
            "mean_difference": float("nan"),
            "standard_error": float("nan"),
            "lower_95": float("nan"),
            "upper_95": float("nan"),
            "raw_p_value": float("nan"),
            "sigflow_better_if": "difference > 0" if higher_is_better else "difference < 0",
            "inference_available": False,
            "unavailable_reason": "fewer than two valid bootstrap replicates",
            "is_primary_block_length": requested_block == cfg.bootstrap_block_days,
        }
    centered = values - np.mean(values)
    raw_p = (
        1.0 + np.sum(np.abs(centered) >= abs(observed))
    ) / (len(values) + 1.0)
    return {
        "cohort": cohort,
        "comparison": f"SigFlow v4 minus {baseline}",
        "metric": metric,
        "resampling_scheme": scheme,
        "requested_block_length": requested_block,
        "effective_block_length": effective_block,
        "unique_dates": date_count,
        "ticker_clusters": ticker_count,
        "approximate_effective_blocks": date_count / max(effective_block, 1),
        "repetitions": repetitions,
        "observed_difference": observed,
        "mean_difference": float(np.mean(values)),
        "standard_error": float(np.std(values, ddof=1)),
        "lower_95": float(np.quantile(values, lower_quantile)),
        "upper_95": float(np.quantile(values, upper_quantile)),
        "raw_p_value": float(raw_p),
        "sigflow_better_if": "difference > 0" if higher_is_better else "difference < 0",
        "inference_available": True,
        "unavailable_reason": "",
        "is_primary_block_length": requested_block == cfg.bootstrap_block_days,
    }


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    cfg: Config,
    cohort: str,
    requested_block_length: int,
    repetitions: int,
) -> pd.DataFrame:
    ordered_dates = np.array(sorted(pd.to_datetime(
        predictions["origin_date"]
    ).unique()))
    date_codes = pd.Categorical(
        pd.to_datetime(predictions["origin_date"]),
        categories=ordered_dates,
        ordered=True,
    ).codes
    rng = np.random.default_rng(
        cfg.seed + 1009 * requested_block_length + len(predictions)
    )
    draws = []
    effective_block = min(requested_block_length, len(ordered_dates))
    for _ in range(repetitions):
        draw, effective_block = moving_block_draw(
            len(ordered_dates), requested_block_length, rng
        )
        draws.append(draw)
    draws = np.asarray(draws, dtype=int)
    plan_hash = resample_plan_hash(
        {
            "schema_version": 2,
            "scheme": "date_block",
            "requested_block_length": requested_block_length,
            "effective_block_length": effective_block,
            "repetitions": repetitions,
            "rng_seed": cfg.seed + 1009 * requested_block_length + len(predictions),
            "ordered_origin_dates": [
                pd.Timestamp(value).date().isoformat() for value in ordered_dates
            ],
            "ordered_tickers": sorted(predictions["ticker"].astype(str).unique()),
            "observations_per_date": np.bincount(
                date_codes, minlength=len(ordered_dates)
            ).astype(int).tolist(),
        },
        draws,
    )

    rows = []
    baseline_names = [name for name in MODEL_COLUMNS if name != "SigFlow v4"]
    counts = np.bincount(date_codes, minlength=len(ordered_dates)).astype(float)
    for baseline in baseline_names:
        differences = paired_difference_arrays(predictions, baseline, cfg)
        for metric, row_differences in differences.items():
            sums = np.bincount(
                date_codes,
                weights=row_differences,
                minlength=len(ordered_dates),
            )
            bootstrap_values = np.array([
                sums[draw].sum() / max(counts[draw].sum(), 1.0)
                for draw in draws
            ])
            rows.append(bootstrap_result_row(
                bootstrap_values,
                float(np.mean(row_differences)),
                cohort,
                baseline,
                metric,
                "date_block",
                requested_block_length,
                effective_block,
                len(ordered_dates),
                predictions["ticker"].nunique(),
                repetitions,
                cfg,
            ))
            rows[-1]["resample_plan_hash"] = plan_hash

        actual = predictions["actual_vol"].to_numpy(dtype=float)
        sigflow = predictions[MODEL_COLUMNS["SigFlow v4"]].to_numpy(dtype=float)
        baseline_values = predictions[MODEL_COLUMNS[baseline]].to_numpy(dtype=float)
        sigflow_sums = np.bincount(
            date_codes, weights=(sigflow - actual) ** 2,
            minlength=len(ordered_dates),
        )
        baseline_sums = np.bincount(
            date_codes, weights=(baseline_values - actual) ** 2,
            minlength=len(ordered_dates),
        )
        rmse_values = np.array([
            math.sqrt(sigflow_sums[draw].sum() / max(counts[draw].sum(), 1.0))
            - math.sqrt(baseline_sums[draw].sum() / max(counts[draw].sum(), 1.0))
            for draw in draws
        ])
        observed_rmse = (
            math.sqrt(float(np.mean((sigflow - actual) ** 2)))
            - math.sqrt(float(np.mean((baseline_values - actual) ** 2)))
        )
        rows.append(bootstrap_result_row(
            rmse_values,
            observed_rmse,
            cohort,
            baseline,
            "rmse",
            "date_block",
            requested_block_length,
            effective_block,
            len(ordered_dates),
            predictions["ticker"].nunique(),
            repetitions,
            cfg,
        ))
        rows[-1]["resample_plan_hash"] = plan_hash
    return pd.DataFrame(rows)


def paired_ticker_date_bootstrap(
    predictions: pd.DataFrame,
    cfg: Config,
    cohort: str,
) -> pd.DataFrame:
    tickers = sorted(predictions["ticker"].astype(str).unique())
    dates = np.array(sorted(pd.to_datetime(
        predictions["origin_date"]
    ).unique()))
    panel_counts = predictions.groupby(["ticker", "origin_date"]).size()
    complete = (
        len(panel_counts) == len(tickers) * len(dates)
        and int(panel_counts.min()) == 1
        and int(panel_counts.max()) == 1
    )
    if not complete or len(tickers) < 2 or len(dates) < 2:
        return pd.DataFrame([{
            "cohort": cohort,
            "comparison": f"SigFlow v4 minus {cfg.primary_baseline}",
            "metric": cfg.primary_metric,
            "resampling_scheme": "ticker_date_two_way",
            "requested_block_length": cfg.bootstrap_block_days,
            "effective_block_length": min(cfg.bootstrap_block_days, len(dates)),
            "unique_dates": len(dates),
            "ticker_clusters": len(tickers),
            "approximate_effective_blocks": len(dates) / max(cfg.bootstrap_block_days, 1),
            "repetitions": 0,
            "observed_difference": float("nan"),
            "mean_difference": float("nan"),
            "standard_error": float("nan"),
            "lower_95": float("nan"),
            "upper_95": float("nan"),
            "raw_p_value": float("nan"),
            "sigflow_better_if": "difference < 0",
            "inference_available": False,
            "unavailable_reason": "ticker/date panel is incomplete or too small",
            "is_primary_block_length": True,
        }])

    ordered = predictions.copy()
    ordered["origin_date"] = pd.to_datetime(ordered["origin_date"])
    ordered = ordered.set_index(["ticker", "origin_date"])
    rng = np.random.default_rng(cfg.seed + 987654)
    ticker_draws = rng.integers(
        0, len(tickers),
        size=(cfg.ticker_date_bootstrap_repetitions, len(tickers)),
    )
    date_draws = []
    effective_block = min(cfg.bootstrap_block_days, len(dates))
    for _ in range(cfg.ticker_date_bootstrap_repetitions):
        draw, effective_block = moving_block_draw(
            len(dates), cfg.bootstrap_block_days, rng
        )
        date_draws.append(draw)
    date_draws = np.asarray(date_draws, dtype=int)
    plan_hash = resample_plan_hash(
        {
            "schema_version": 2,
            "scheme": "ticker_date_two_way",
            "requested_block_length": cfg.bootstrap_block_days,
            "effective_block_length": effective_block,
            "repetitions": cfg.ticker_date_bootstrap_repetitions,
            "rng_seed": cfg.seed + 987654,
            "ordered_origin_dates": [
                pd.Timestamp(value).date().isoformat() for value in dates
            ],
            "ordered_tickers": tickers,
            "panel_order": "ticker_then_origin_date",
        },
        ticker_draws,
        date_draws,
    )

    rows = []
    for baseline in [name for name in MODEL_COLUMNS if name != "SigFlow v4"]:
        flat = predictions.copy()
        differences = paired_difference_arrays(flat, baseline, cfg)
        for metric, values in differences.items():
            value_frame = flat[["ticker", "origin_date"]].copy()
            value_frame["difference"] = values
            matrix = value_frame.pivot(
                index="ticker", columns="origin_date", values="difference"
            ).reindex(index=tickers, columns=dates).to_numpy(dtype=float)
            if not np.all(np.isfinite(matrix)):
                bootstrap_values = np.array([], dtype=float)
            else:
                bootstrap_values = np.array([
                    matrix[np.ix_(ticker_draws[index], date_draws[index])].mean()
                    for index in range(cfg.ticker_date_bootstrap_repetitions)
                ])
            rows.append(bootstrap_result_row(
                bootstrap_values,
                float(np.mean(values)),
                cohort,
                baseline,
                metric,
                "ticker_date_two_way",
                cfg.bootstrap_block_days,
                effective_block,
                len(dates),
                len(tickers),
                cfg.ticker_date_bootstrap_repetitions,
                cfg,
            ))
            rows[-1]["resample_plan_hash"] = plan_hash

        actual_frame = flat[["ticker", "origin_date", "actual_vol"]].copy()
        actual_frame["sigflow_squared_error"] = (
            flat[MODEL_COLUMNS["SigFlow v4"]].to_numpy(dtype=float)
            - flat["actual_vol"].to_numpy(dtype=float)
        ) ** 2
        actual_frame["baseline_squared_error"] = (
            flat[MODEL_COLUMNS[baseline]].to_numpy(dtype=float)
            - flat["actual_vol"].to_numpy(dtype=float)
        ) ** 2
        sigflow_matrix = actual_frame.pivot(
            index="ticker", columns="origin_date", values="sigflow_squared_error"
        ).reindex(index=tickers, columns=dates).to_numpy(dtype=float)
        baseline_matrix = actual_frame.pivot(
            index="ticker", columns="origin_date", values="baseline_squared_error"
        ).reindex(index=tickers, columns=dates).to_numpy(dtype=float)
        if np.all(np.isfinite(sigflow_matrix)) and np.all(np.isfinite(baseline_matrix)):
            rmse_values = np.array([
                math.sqrt(sigflow_matrix[np.ix_(ticker_draws[index], date_draws[index])].mean())
                - math.sqrt(baseline_matrix[np.ix_(ticker_draws[index], date_draws[index])].mean())
                for index in range(cfg.ticker_date_bootstrap_repetitions)
            ])
        else:
            rmse_values = np.array([], dtype=float)
        rows.append(bootstrap_result_row(
            rmse_values,
            math.sqrt(float(np.mean(sigflow_matrix)))
            - math.sqrt(float(np.mean(baseline_matrix))),
            cohort,
            baseline,
            "rmse",
            "ticker_date_two_way",
            cfg.bootstrap_block_days,
            effective_block,
            len(dates),
            len(tickers),
            cfg.ticker_date_bootstrap_repetitions,
            cfg,
        ))
        rows[-1]["resample_plan_hash"] = plan_hash
    return pd.DataFrame(rows)


def apply_holm_correction(
    inference: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    inference = inference.copy()
    inference["holm_adjusted_p_value"] = np.nan
    inference["raw_significant_0_05"] = (
        inference["raw_p_value"] < cfg.significance_alpha
    )
    inference["holm_significant_0_05"] = False
    inference["holm_family"] = "outside_prespecified_secondary_family"
    family_mask = (
        (inference["resampling_scheme"] == "ticker_date_two_way")
        & inference["cohort"].isin([
            "seen_ticker_unseen_day", "unseen_ticker_unseen_day"
        ])
        & inference["metric"].isin([
            "mae", "rmse", "qlike", "crps", primary_accuracy_metric_name(cfg)
        ])
        & np.isfinite(inference["raw_p_value"])
    )
    family_indices = inference.index[family_mask].tolist()
    if family_indices:
        ordered = sorted(
            family_indices,
            key=lambda index: float(inference.loc[index, "raw_p_value"]),
        )
        family_size = len(ordered)
        running = 0.0
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0,
                (family_size - rank)
                * float(inference.loc[index, "raw_p_value"]),
            )
            running = max(running, adjusted)
            inference.loc[index, "holm_adjusted_p_value"] = running
        inference.loc[family_indices, "holm_family"] = (
            "two_final_cohorts_x_all_baselines_x_five_metrics"
        )
        inference.loc[family_indices, "holm_significant_0_05"] = (
            inference.loc[family_indices, "holm_adjusted_p_value"]
            < cfg.significance_alpha
        )
    return inference


def bootstrap_evidence_label(
    mean_difference: float,
    lower_95: float,
    upper_95: float,
    higher_is_better: bool = False,
) -> str:
    if not np.all(np.isfinite([mean_difference, lower_95, upper_95])):
        return "bootstrap_not_run"
    if higher_is_better:
        if lower_95 > 0.0:
            return "strong_evidence_sigflow_better"
        if upper_95 < 0.0:
            return "strong_evidence_sigflow_worse"
        return (
            "mixed_ci_direction_sigflow"
            if mean_difference > 0.0
            else "mixed_ci_direction_baseline"
        )
    if upper_95 < 0.0:
        return "strong_evidence_sigflow_better"
    if lower_95 > 0.0:
        return "strong_evidence_sigflow_worse"
    return (
        "mixed_ci_direction_sigflow"
        if mean_difference < 0.0
        else "mixed_ci_direction_baseline"
    )


def build_per_seed_metrics(
    predictions: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    primary_label = tolerance_percent_label(cfg.primary_success_tolerance)
    for information in evaluation_slices(predictions, include_tickers=True):
        selected = information["rows"]
        actual = selected["actual_vol"].to_numpy(dtype=float)
        har = selected["har_style_baseline"].to_numpy(dtype=float)
        har_metrics = point_metrics(actual, har, cfg, qlike_predicted=har)
        for seed in cfg.ensemble_seeds:
            median = selected[f"seed_{seed}_median_vol"].to_numpy(dtype=float)
            qlike_forecast = selected[f"seed_{seed}_qlike_vol"].to_numpy(dtype=float)
            metrics = point_metrics(
                actual, median, cfg, qlike_predicted=qlike_forecast
            )
            qlike_difference = metrics["qlike"] - har_metrics["qlike"]
            rows.append({
                **{
                    key: information[key]
                    for key in ("group_level", "group", "cohort", "ticker")
                },
                "seed": seed,
                "observations": len(selected),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "qlike": metrics["qlike"],
                "correlation": metrics["correlation"],
                "crps": float(selected[f"seed_{seed}_crps"].mean()),
                "model_space_uncalibrated_nll": float(
                    selected[f"seed_{seed}_nll"].mean()
                ),
                "coverage_90": float(
                    selected[f"seed_{seed}_coverage_90"].mean()
                ),
                f"within_{primary_label}pct_accuracy": metrics[
                    f"within_{primary_label}pct_accuracy"
                ],
                "mae_difference_vs_har": metrics["mae"] - har_metrics["mae"],
                "qlike_difference_vs_har": qlike_difference,
                "qlike_relative_improvement_vs_har": (
                    (har_metrics["qlike"] - metrics["qlike"])
                    / max(abs(har_metrics["qlike"]), EPS)
                ),
                "within_primary_difference_vs_har": (
                    metrics[f"within_{primary_label}pct_accuracy"]
                    - har_metrics[f"within_{primary_label}pct_accuracy"]
                ),
                "qlike_favours_sigflow": qlike_difference < 0.0,
            })
    per_seed = pd.DataFrame(rows)
    stability_rows = []
    stability_source = per_seed[
        per_seed["group_level"].isin(["pooled", "cohort"])
    ]
    for (group_level, cohort), selected in stability_source.groupby(
        ["group_level", "cohort"], sort=True
    ):
        prediction_rows = (
            predictions
            if group_level == "pooled"
            else predictions[predictions["evaluation_cohort"] == cohort]
        )
        seed_columns = [
            f"seed_{seed}_median_vol" for seed in cfg.ensemble_seeds
        ]
        row_dispersion = prediction_rows[seed_columns].std(axis=1, ddof=0)
        qlike_differences = selected["qlike_difference_vs_har"].to_numpy(dtype=float)
        stability_rows.append({
            "group_level": group_level,
            "cohort": cohort,
            "seed_count": len(selected),
            "fraction_seeds_qlike_favours_sigflow": float(
                np.mean(qlike_differences < 0.0)
            ),
            "all_seeds_qlike_favour_sigflow": bool(
                np.all(qlike_differences < 0.0)
            ),
            "qlike_difference_mean": float(np.mean(qlike_differences)),
            "qlike_difference_std": float(np.std(qlike_differences)),
            "qlike_difference_min": float(np.min(qlike_differences)),
            "qlike_difference_max": float(np.max(qlike_differences)),
            "row_forecast_dispersion_mean": float(row_dispersion.mean()),
            "row_forecast_dispersion_p90": float(row_dispersion.quantile(0.90)),
            "stability_sufficient": len(selected) >= (1 if cfg.quick_mode else 5),
        })
        for metric in (
            "mae", "rmse", "qlike", "crps",
            "model_space_uncalibrated_nll", "coverage_90", "correlation",
        ):
            values = selected[metric].to_numpy(dtype=float)
            stability_rows[-1][f"{metric}_mean"] = float(np.nanmean(values))
            stability_rows[-1][f"{metric}_std"] = float(np.nanstd(values))
            stability_rows[-1][f"{metric}_min"] = float(np.nanmin(values))
            stability_rows[-1][f"{metric}_max"] = float(np.nanmax(values))
    return per_seed, pd.DataFrame(stability_rows)


def build_primary_claim_summary(
    point_table: pd.DataFrame,
    inference: pd.DataFrame | None,
    seed_stability: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    data_source: str,
    evaluation_split: str,
) -> pd.DataFrame:
    cohort = cfg.primary_evaluation_cohort
    selected_points = point_table[
        (point_table["group_level"] == "cohort")
        & (point_table["group"] == cohort)
    ]
    sigflow = selected_points[selected_points["model"] == "SigFlow v4"]
    baseline = selected_points[selected_points["model"] == cfg.primary_baseline]
    relative_improvement = float("nan")
    if not sigflow.empty and not baseline.empty:
        relative_improvement = float(
            (baseline.iloc[0]["qlike"] - sigflow.iloc[0]["qlike"])
            / max(abs(float(baseline.iloc[0]["qlike"])), EPS)
        )
    primary_row = pd.DataFrame()
    if inference is not None:
        primary_row = inference[
            (inference["cohort"] == cohort)
            & (inference["comparison"] == f"SigFlow v4 minus {cfg.primary_baseline}")
            & (inference["metric"] == cfg.primary_metric)
            & (inference["resampling_scheme"] == "ticker_date_two_way")
        ]
    if primary_row.empty:
        observed = lower = upper = raw_p = holm_p = float("nan")
        inference_available = False
    else:
        selected = primary_row.iloc[0]
        observed = float(selected["observed_difference"])
        lower = float(selected["lower_95"])
        upper = float(selected["upper_95"])
        raw_p = float(selected["raw_p_value"])
        holm_p = float(selected["holm_adjusted_p_value"])
        inference_available = bool(selected["inference_available"])
    sensitivity_rows = pd.DataFrame()
    if inference is not None:
        sensitivity_rows = inference[
            (inference["cohort"] == cohort)
            & (inference["comparison"] == f"SigFlow v4 minus {cfg.primary_baseline}")
            & (inference["metric"] == cfg.primary_metric)
            & (inference["resampling_scheme"] == "date_block")
        ]
    block_sensitivity_robust = (
        len(sensitivity_rows) == len(cfg.bootstrap_block_sensitivity_days)
        and bool(np.all(
            sensitivity_rows["inference_available"].to_numpy(dtype=bool)
        ))
        and bool(np.all(
            sensitivity_rows["upper_95"].to_numpy(dtype=float) < 0.0
        ))
    )
    seed_row = seed_stability[seed_stability["cohort"] == cohort]
    seed_fraction = (
        float(seed_row.iloc[0]["fraction_seeds_qlike_favours_sigflow"])
        if not seed_row.empty else float("nan")
    )
    cohort_rows = predictions[predictions["evaluation_cohort"] == cohort]
    unique_dates = int(cohort_rows["origin_date"].nunique())
    ticker_count = int(cohort_rows["ticker"].nunique())
    real_final = (
        evaluation_split == "test"
        and cfg.experiment_mode == "research"
        and cfg.final_evaluation_authorized
        and final_protocol_hash(cfg) == PREREGISTERED_FINAL_PROTOCOL_SHA256
        and not str(data_source).startswith("synthetic")
        and not cohort_rows.empty
        and "data_provenance_valid" in cohort_rows
        and bool(cohort_rows["data_provenance_valid"].astype(bool).all())
        and pd.to_datetime(cohort_rows["origin_date"]).min()
        >= pd.Timestamp(cfg.prospective_test_start_date)
        and pd.to_datetime(cohort_rows["origin_date"]).max()
        <= pd.Timestamp(cfg.prospective_test_end_date)
        and pd.to_datetime(cohort_rows["target_end_date"]).max()
        <= pd.Timestamp(cfg.prospective_data_end_date)
    )
    adequate_panel = (
        unique_dates >= cfg.minimum_test_origins_per_ticker
        and ticker_count >= 15
    )
    statistical = inference_available and upper < 0.0 and raw_p < cfg.significance_alpha
    practical = relative_improvement >= cfg.primary_min_relative_improvement
    stable = seed_fraction >= cfg.minimum_seed_improvement_fraction
    if not real_final:
        verdict = (
            "pipeline_validation_only"
            if cfg.experiment_mode == "pipeline_test"
            else "development_evidence_only_prospective_holdout_required"
        )
    elif not adequate_panel:
        verdict = "underpowered_panel"
    elif statistical and practical and stable and block_sensitivity_robust:
        verdict = "supported"
    else:
        verdict = "not_supported"
    return pd.DataFrame([{
        "claim": "SigFlow improves unseen-ticker QLIKE versus Log-HAR",
        "evaluation_cohort": cohort,
        "baseline": cfg.primary_baseline,
        "metric": cfg.primary_metric,
        "data_source": data_source,
        "evaluation_split": evaluation_split,
        "unique_test_origins": unique_dates,
        "unseen_ticker_count": ticker_count,
        "observed_difference": observed,
        "lower_95": lower,
        "upper_95": upper,
        "raw_p_value": raw_p,
        "holm_adjusted_p_value_secondary_family": holm_p,
        "relative_qlike_improvement": relative_improvement,
        "required_relative_improvement": cfg.primary_min_relative_improvement,
        "fraction_seeds_favouring_sigflow": seed_fraction,
        "required_seed_fraction": cfg.minimum_seed_improvement_fraction,
        "real_untouched_final_test": real_final,
        "adequately_powered_panel": adequate_panel,
        "statistical_requirement_met": statistical,
        "practical_requirement_met": practical,
        "seed_stability_requirement_met": stable,
        "all_block_sensitivities_favour_sigflow": block_sensitivity_robust,
        "verdict": verdict,
    }])


def build_baseline_comparison_summary(
    point_table: pd.DataFrame,
    accuracy_summary: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    primary_label = tolerance_percent_label(cfg.primary_success_tolerance)
    group_keys = point_table.loc[
        point_table["group_level"].isin(["pooled", "cohort"]),
        ["group_level", "group", "cohort", "ticker"],
    ].drop_duplicates()

    for group in group_keys.to_dict("records"):
        mask = (
            (point_table["group_level"] == group["group_level"])
            & (point_table["group"] == group["group"])
        )
        group_points = point_table.loc[mask]
        sigflow = group_points.loc[
            group_points["model"] == "SigFlow v4"
        ].iloc[0]
        accuracy = accuracy_summary.loc[
            (accuracy_summary["group_level"] == group["group_level"])
            & (accuracy_summary["group"] == group["group"])
        ].iloc[0]
        bootstrap_cohort = "all" if group["group_level"] == "pooled" else group["group"]

        for baseline_name in MODEL_COLUMNS:
            if baseline_name == "SigFlow v4":
                continue
            baseline = group_points.loc[
                group_points["model"] == baseline_name
            ].iloc[0]
            prefix = MODEL_RESULT_PREFIXES[baseline_name]
            row = {
                **group,
                "observations": int(sigflow["observations"]),
                "baseline": baseline_name,
                "sigflow_mae": float(sigflow["mae"]),
                "baseline_mae": float(baseline["mae"]),
                "mae_difference": float(sigflow["mae"] - baseline["mae"]),
                "mae_relative_improvement_percent": float(
                    100.0 * (baseline["mae"] - sigflow["mae"])
                    / max(abs(float(baseline["mae"])), EPS)
                ),
                "sigflow_qlike": float(sigflow["qlike"]),
                "baseline_qlike": float(baseline["qlike"]),
                "qlike_difference": float(sigflow["qlike"] - baseline["qlike"]),
                "qlike_relative_improvement_percent": float(
                    100.0 * (baseline["qlike"] - sigflow["qlike"])
                    / max(abs(float(baseline["qlike"])), EPS)
                ),
            }
            for tolerance in cfg.success_relative_tolerances:
                label = tolerance_percent_label(tolerance)
                sigflow_accuracy = float(accuracy[f"sigflow_within_{label}pct_accuracy"])
                baseline_accuracy = float(accuracy[f"{prefix}_within_{label}pct_accuracy"])
                row[f"sigflow_within_{label}pct_accuracy"] = sigflow_accuracy
                row[f"baseline_within_{label}pct_accuracy"] = baseline_accuracy
                row[f"accuracy_difference_{label}pct_percentage_points"] = (
                    100.0 * (sigflow_accuracy - baseline_accuracy)
                )

            for metric in ("mae", "qlike"):
                bootstrap_row = pd.DataFrame()
                if bootstrap is not None:
                    preferred_scheme = (
                        "date_block"
                        if group["group_level"] == "pooled"
                        else "ticker_date_two_way"
                    )
                    bootstrap_row = bootstrap.loc[
                        (bootstrap["cohort"] == bootstrap_cohort)
                        & (bootstrap["comparison"] == f"SigFlow v4 minus {baseline_name}")
                        & (bootstrap["metric"] == metric)
                        & (bootstrap["resampling_scheme"] == preferred_scheme)
                        & (bootstrap["requested_block_length"] == cfg.bootstrap_block_days)
                    ]
                if bootstrap_row.empty:
                    mean_difference = lower_95 = upper_95 = float("nan")
                else:
                    selected = bootstrap_row.iloc[0]
                    mean_difference = float(selected["mean_difference"])
                    lower_95 = float(selected["lower_95"])
                    upper_95 = float(selected["upper_95"])
                row[f"{metric}_bootstrap_mean_difference"] = mean_difference
                row[f"{metric}_bootstrap_lower_95"] = lower_95
                row[f"{metric}_bootstrap_upper_95"] = upper_95
                row[f"{metric}_evidence"] = bootstrap_evidence_label(
                    mean_difference, lower_95, upper_95
                )

            accuracy_bootstrap = pd.DataFrame()
            if bootstrap is not None:
                preferred_scheme = (
                    "date_block"
                    if group["group_level"] == "pooled"
                    else "ticker_date_two_way"
                )
                accuracy_bootstrap = bootstrap.loc[
                    (bootstrap["cohort"] == bootstrap_cohort)
                    & (bootstrap["comparison"] == f"SigFlow v4 minus {baseline_name}")
                    & (bootstrap["metric"] == primary_accuracy_metric_name(cfg))
                    & (bootstrap["resampling_scheme"] == preferred_scheme)
                    & (bootstrap["requested_block_length"] == cfg.bootstrap_block_days)
                ]
            if accuracy_bootstrap.empty:
                acc_mean = acc_lower = acc_upper = float("nan")
            else:
                selected_accuracy = accuracy_bootstrap.iloc[0]
                acc_mean = float(selected_accuracy["mean_difference"])
                acc_lower = float(selected_accuracy["lower_95"])
                acc_upper = float(selected_accuracy["upper_95"])
            row["primary_accuracy_bootstrap_mean_difference"] = acc_mean
            row["primary_accuracy_bootstrap_lower_95"] = acc_lower
            row["primary_accuracy_bootstrap_upper_95"] = acc_upper
            row["primary_accuracy_evidence"] = bootstrap_evidence_label(
                acc_mean, acc_lower, acc_upper, higher_is_better=True
            )
            row["primary_success_tolerance"] = cfg.primary_success_tolerance
            row["primary_accuracy_difference_percentage_points"] = row[
                f"accuracy_difference_{primary_label}pct_percentage_points"
            ]
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    cfg: Config,
    data_source: str,
    evaluation_split: str,
) -> dict[str, object]:
    predictions = predictions.sort_values(
        ["origin_date", "evaluation_cohort", "ticker"]
    ).reset_index(drop=True)
    required_positive = [
        "actual_vol", "predicted_median_vol", "predicted_qlike_vol",
        "rolling_vol_baseline", "ewma_vol_baseline", "har_style_baseline",
    ]
    if predictions.duplicated(["ticker", "origin_date"]).any():
        raise ValueError("Every ticker/origin result must be unique.")
    values = predictions[required_positive].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Inference requires finite, positive forecasts and outcomes.")

    point_table = evaluate_all_groups(predictions, cfg)
    accuracy_summary = build_accuracy_summary(predictions, cfg)
    daily_summary = build_daily_success_summary(predictions, cfg)
    probabilistic_summary = probabilistic_summary_by_group(predictions)
    raw_metrics, raw_matrix, raw_matrices = regime_summary_by_group(
        predictions, cfg, smoothed=False
    )
    if cfg.apply_regime_smoothing:
        smoothed_metrics, smoothed_matrix, smoothed_matrices = (
            regime_summary_by_group(predictions, cfg, smoothed=True)
        )
    else:
        smoothed_metrics = pd.DataFrame()
        smoothed_matrix = np.zeros((cfg.regimes, cfg.regimes), dtype=int)
        smoothed_matrices = {}
    non_overlapping = non_overlapping_phase_metrics(predictions, cfg)
    per_seed_metrics, seed_stability = build_per_seed_metrics(predictions, cfg)

    date_bootstrap = None
    ticker_date_bootstrap = None
    inference = None
    if cfg.run_block_bootstrap:
        date_tables = []
        groups = [("all", predictions)] + list(
            predictions.groupby("evaluation_cohort", sort=True)
        )
        for cohort, selected in groups:
            for block_days in cfg.bootstrap_block_sensitivity_days:
                repetitions = (
                    cfg.bootstrap_repetitions
                    if block_days == cfg.bootstrap_block_days
                    else cfg.bootstrap_sensitivity_repetitions
                )
                date_tables.append(paired_block_bootstrap(
                    selected,
                    cfg,
                    str(cohort),
                    block_days,
                    repetitions,
                ))
        date_bootstrap = pd.concat(date_tables, ignore_index=True)
        two_way_tables = [
            paired_ticker_date_bootstrap(selected, cfg, str(cohort))
            for cohort, selected in predictions.groupby(
                "evaluation_cohort", sort=True
            )
        ]
        ticker_date_bootstrap = pd.concat(two_way_tables, ignore_index=True)
        inference = apply_holm_correction(pd.concat(
            [date_bootstrap, ticker_date_bootstrap],
            ignore_index=True,
        ), cfg)
        date_bootstrap = inference[
            inference["resampling_scheme"] == "date_block"
        ].reset_index(drop=True)
        ticker_date_bootstrap = inference[
            inference["resampling_scheme"] == "ticker_date_two_way"
        ].reset_index(drop=True)

    baseline_comparison = build_baseline_comparison_summary(
        point_table, accuracy_summary, inference, cfg
    )
    primary_claim = build_primary_claim_summary(
        point_table,
        inference,
        seed_stability,
        predictions,
        cfg,
        data_source,
        evaluation_split,
    )

    detailed_results = predictions.drop(
        columns=["ticker_id", "split"],
        errors="ignore",
    )
    responsiveness = point_table[
        (point_table["group_level"] == "cohort_ticker")
        & (point_table["model"] == "SigFlow v4")
    ][[
        "cohort", "ticker", "observations", "correlation", "bias",
        "actual_standard_deviation", "predicted_standard_deviation",
        "prediction_to_actual_std_ratio",
        "calibration_slope_actual_on_predicted",
        "best_prediction_lag_sessions", "best_lagged_correlation",
    ]].reset_index(drop=True)
    return {
        "detailed_test_results": detailed_results,
        "successful_forecasts": detailed_results[
            detailed_results["success_within_20pct"]
        ].reset_index(drop=True),
        "failed_forecasts": detailed_results[
            ~detailed_results["success_within_20pct"]
        ].reset_index(drop=True),
        "accuracy_summary": accuracy_summary,
        "daily_success_summary": daily_summary,
        "point_metrics": point_table,
        "baseline_comparison_summary": baseline_comparison,
        "probabilistic_summary": probabilistic_summary,
        "coverage_diagnostics": probabilistic_summary.copy(),
        "raw_regime_metrics": raw_metrics,
        "smoothed_regime_metrics": smoothed_metrics,
        "raw_confusion_matrix": raw_matrix,
        "smoothed_confusion_matrix": smoothed_matrix,
        "raw_confusion_matrices_by_cohort": raw_matrices,
        "smoothed_confusion_matrices_by_cohort": smoothed_matrices,
        "non_overlapping_phase_metrics": non_overlapping,
        "paired_bootstrap": date_bootstrap,
        "ticker_date_bootstrap": ticker_date_bootstrap,
        "inference_summary": inference,
        "per_seed_metrics": per_seed_metrics,
        "per_seed_stability": seed_stability,
        "primary_claim_summary": primary_claim,
        "responsiveness_diagnostics": responsiveness,
    }

# %% [markdown cell 27]
# ## Generalisation plots and result-only artifact saving
# 
# The notebook saves all detailed forecast values plus `training_summary.csv`, `baseline_comparison_summary.csv`, `per_seed_metrics.csv`, `per_seed_stability.csv`, `paired_block_bootstrap.csv`, `ticker_date_bootstrap.csv`, `inference_summary.csv`, and `primary_claim_summary.csv`. No raw OHLCV or model-input values are placed in result tables.

# %% [cell 28]
def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.xticks(np.arange(3), REGIME_NAMES)
    plt.yticks(np.arange(3), REGIME_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    for row in range(3):
        for column in range(3):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def save_diagnostic_plots(
    predictions: pd.DataFrame,
    evaluation: dict[str, object],
    cfg: Config,
    output_directory: Path | None = None,
) -> None:
    plot_directory = (
        output_directory if output_directory is not None else Path(cfg.output_dir)
    ) / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)

    for ticker, selected in predictions.groupby("ticker", sort=True):
        selected = selected.sort_values("origin_date")
        dates = pd.to_datetime(selected["origin_date"])
        figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(dates, selected["actual_vol"], label="Actual", linewidth=1.5)
        axes[0].plot(dates, selected["predicted_median_vol"], label="SigFlow")
        axes[0].plot(dates, selected["log_har_baseline"], label="Log-HAR", alpha=0.8)
        axes[0].plot(dates, selected["ewma_vol_baseline"], label="EWMA", alpha=0.7)
        axes[0].plot(dates, selected["rolling_vol_baseline"], label="Rolling RV", alpha=0.7)
        axes[0].set_ylabel("Annualised volatility")
        axes[0].legend(ncol=5, fontsize=8)
        axes[0].set_title(f"{ticker}: actual and competing forecasts")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].plot(
            dates,
            selected["predicted_median_vol"] - selected["actual_vol"],
            label="SigFlow residual",
        )
        axes[1].set_ylabel("Forecast error")
        axes[1].legend()
        for regime_name in REGIME_NAMES:
            column = f"probability_{regime_name.lower()}"
            axes[2].plot(dates, selected[column], label=str(regime_name))
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_ylabel("Regime probability")
        axes[2].legend(ncol=3)
        axes[2].set_xlabel("Forecast origin")
        figure.tight_layout()
        figure.savefig(
            plot_directory / f"{ticker}_forecast_residual_regimes.png",
            dpi=160,
        )
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 7))
    for cohort, selected in predictions.groupby("evaluation_cohort", sort=True):
        axis.scatter(
            selected["actual_vol"],
            selected["predicted_median_vol"],
            s=10,
            alpha=0.35,
            label=cohort,
        )
    bounds = [
        float(min(predictions["actual_vol"].min(), predictions["predicted_median_vol"].min())),
        float(max(predictions["actual_vol"].max(), predictions["predicted_median_vol"].max())),
    ]
    axis.plot(bounds, bounds, "k--", linewidth=1, label="identity")
    axis.set_xlabel("Actual volatility")
    axis.set_ylabel("SigFlow forecast")
    axis.set_title("Forecast scatter")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_directory / "forecast_scatter.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    error_groups = [
        selected["sigflow_absolute_error"].to_numpy(dtype=float)
        for _, selected in predictions.groupby("regime", sort=True)
    ]
    axis.boxplot(error_groups, labels=REGIME_NAMES, showfliers=False)
    axis.set_ylabel("Absolute forecast error")
    axis.set_title("SigFlow error by true volatility regime")
    figure.tight_layout()
    figure.savefig(plot_directory / "error_by_regime.png", dpi=160)
    plt.close(figure)

    nominal = np.asarray(cfg.interval_coverage_levels, dtype=float)
    empirical = np.asarray([
        predictions[f"inside_{int(round(level * 100))}pct_interval"].mean()
        for level in nominal
    ])
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(nominal, empirical, marker="o", label="Empirical")
    axis.plot([0, 1], [0, 1], "k--", label="Ideal")
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_title("Interval calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_directory / "coverage_calibration.png", dpi=160)
    plt.close(figure)

    ticker_metrics = evaluation["point_metrics"].query(
        "group_level == 'cohort_ticker' and model == 'SigFlow v4'"
    ).sort_values("prediction_to_actual_std_ratio")
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(ticker_metrics["ticker"], ticker_metrics["prediction_to_actual_std_ratio"])
    axis.axhline(1.0, color="black", linestyle="--")
    axis.tick_params(axis="x", rotation=75)
    axis.set_ylabel("Prediction SD / actual SD")
    axis.set_title(f"Forecast responsiveness by ticker, horizon={cfg.horizon}")
    figure.tight_layout()
    figure.savefig(plot_directory / "responsiveness_by_ticker.png", dpi=160)
    plt.close(figure)


def plot_results(
    histories: list[dict[str, list[float]]],
    predictions: pd.DataFrame,
    evaluation: dict[str, object],
    cfg: Config,
) -> None:
    plt.figure(figsize=(10, 4))
    for model_index, history in enumerate(histories, start=1):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        plt.plot(
            epochs,
            history["train_loss"],
            alpha=0.55,
            label=f"Train {model_index}",
        )
        plt.plot(
            epochs,
            history["validation_loss"],
            label=f"Validation {model_index}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("Training and validation history")
    plt.legend()
    plt.tight_layout()
    plt.show()

    cohort_accuracy = evaluation["accuracy_summary"].query(
        "group_level == 'cohort'"
    ).copy()
    accuracy_columns = [
        "sigflow_within_5pct_accuracy",
        "sigflow_within_10pct_accuracy",
        "sigflow_within_20pct_accuracy",
        "sigflow_within_30pct_accuracy",
    ]
    x = np.arange(len(cohort_accuracy))
    width = 0.18
    plt.figure(figsize=(11, 4.5))
    for offset, column in enumerate(accuracy_columns):
        plt.bar(
            x + (offset - 1.5) * width,
            cohort_accuracy[column],
            width=width,
            label=column.replace("sigflow_within_", "≤").replace("_accuracy", ""),
        )
    plt.xticks(x, cohort_accuracy["cohort"], rotation=10)
    plt.ylim(0, 1)
    plt.ylabel("Forecast accuracy")
    plt.title("SigFlow accuracy by generalisation cohort")
    plt.legend()
    plt.tight_layout()
    plt.show()

    daily = evaluation["daily_success_summary"].copy()
    plt.figure(figsize=(13, 4.5))
    for cohort, selected in daily.groupby("cohort", sort=True):
        selected = selected.sort_values("origin_date")
        smoothed_accuracy = selected["success_rate_20pct"].rolling(
            20, min_periods=1
        ).mean()
        plt.plot(
            pd.to_datetime(selected["origin_date"]),
            smoothed_accuracy,
            label=f"{cohort} (20-origin rolling mean)",
        )
    plt.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    plt.ylim(0, 1)
    plt.ylabel("Within-20% accuracy")
    plt.title("Success through evaluation origins")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if cfg.plot_each_ticker:
        for ticker in cfg.target_tickers:
            ticker_data = predictions[
                predictions["ticker"] == ticker
            ].sort_values("origin_date").copy()
            if ticker_data.empty:
                continue
            dates = pd.to_datetime(ticker_data["origin_date"])
            cohort = str(ticker_data["evaluation_cohort"].iloc[0])
            success = ticker_data["success_within_20pct"].to_numpy(dtype=bool)

            plt.figure(figsize=(13, 5))
            plt.plot(dates, ticker_data["actual_vol"], label="Actual future volatility")
            plt.plot(dates, ticker_data["predicted_median_vol"], label="SigFlow v4")
            plt.fill_between(
                dates,
                ticker_data["predicted_q05_vol"],
                ticker_data["predicted_q95_vol"],
                alpha=0.20,
                label="Calibration-tuned 90% interval",
            )
            plt.scatter(
                dates[success],
                ticker_data.loc[success, "actual_vol"],
                color="green",
                s=14,
                label="Within 20%",
            )
            plt.scatter(
                dates[~success],
                ticker_data.loc[~success, "actual_vol"],
                color="red",
                marker="x",
                s=18,
                label="Outside 20%",
            )
            plt.title(
                f"{ticker}: {cfg.horizon}-session volatility forecast | {cohort}"
            )
            plt.ylabel("Annualised volatility")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(predictions["pit"], bins=np.linspace(0, 1, 11))
    plt.axhline(len(predictions) / 10, linestyle="--")
    plt.title("Evaluation PIT calibration")
    plt.xlabel("PIT")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plot_confusion_matrix(
        evaluation["raw_confusion_matrix"],
        "Raw evaluation regime confusion matrix",
    )
    if cfg.apply_regime_smoothing:
        plot_confusion_matrix(
            evaluation["smoothed_confusion_matrix"],
            "Past-only smoothed evaluation regime confusion matrix",
        )
def save_results(
    models: list[RegimeFlowMixture],
    histories: list[dict[str, list[float]]],
    predictions: pd.DataFrame,
    evaluation: dict[str, object],
    dataset: MarketDataset,
    split: PreparedSplit,
    interval_scale: float,
    interval_tail_calibration: IntervalTailCalibration,
    regime_temperature: float,
    blend_weight: float,
    cfg: Config,
) -> None:
    destination_directory = Path(cfg.output_dir)
    if (
        cfg.experiment_mode == "research"
        and destination_directory.exists()
        and not cfg.overwrite_existing_outputs
    ):
        raise FileExistsError(
            "Refusing to overwrite or merge into an immutable research run: "
            f"{destination_directory}"
        )
    transactional_research_save = (
        cfg.experiment_mode == "research"
        and not cfg.overwrite_existing_outputs
    )
    if transactional_research_save:
        destination_directory.parent.mkdir(parents=True, exist_ok=True)
        output_directory = destination_directory.parent / (
            f".{destination_directory.name}.staging.{os.getpid()}.{time.time_ns()}"
        )
        output_directory.mkdir(parents=False, exist_ok=False)
    else:
        output_directory = destination_directory
        output_directory.mkdir(parents=True, exist_ok=True)

    table_files = {
        "detailed_test_results.csv": evaluation["detailed_test_results"],
        "successful_test_forecasts.csv": evaluation["successful_forecasts"],
        "failed_test_forecasts.csv": evaluation["failed_forecasts"],
        "accuracy_summary.csv": evaluation["accuracy_summary"],
        "daily_success_summary.csv": evaluation["daily_success_summary"],
        "point_metrics.csv": evaluation["point_metrics"],
        "baseline_comparison_summary.csv": (
            evaluation["baseline_comparison_summary"]
        ),
        "primary_claim_summary.csv": evaluation["primary_claim_summary"],
        "per_seed_metrics.csv": evaluation["per_seed_metrics"],
        "per_seed_stability.csv": evaluation["per_seed_stability"],
        "training_summary.csv": evaluation["training_summary"],
        "complexity_summary.csv": evaluation["complexity_summary"],
        "responsiveness_diagnostics.csv": evaluation[
            "responsiveness_diagnostics"
        ],
        "structured_experiment_summary.csv": evaluation[
            "structured_experiment_summary"
        ],
        "probabilistic_summary.csv": evaluation["probabilistic_summary"],
        "coverage_diagnostics.csv": evaluation["coverage_diagnostics"],
        "calibration_partition.csv": evaluation["calibration_partition"],
        "raw_regime_metrics.csv": evaluation["raw_regime_metrics"],
        "regime_diagnostics.csv": evaluation["regime_diagnostics"],
        "regime_confusion_matrices.csv": evaluation[
            "regime_confusion_matrices"
        ],
        "non_overlapping_phase_metrics.csv": (
            evaluation["non_overlapping_phase_metrics"]
        ),
    }
    if cfg.apply_regime_smoothing:
        table_files["smoothed_regime_metrics.csv"] = evaluation[
            "smoothed_regime_metrics"
        ]
    if evaluation["paired_bootstrap"] is not None:
        table_files["paired_block_bootstrap.csv"] = evaluation[
            "paired_bootstrap"
        ]
    if evaluation["ticker_date_bootstrap"] is not None:
        table_files["ticker_date_bootstrap.csv"] = evaluation[
            "ticker_date_bootstrap"
        ]
    if evaluation["inference_summary"] is not None:
        table_files["inference_summary.csv"] = evaluation[
            "inference_summary"
        ]
    for filename, table in table_files.items():
        table.to_csv(output_directory / filename, index=False)

    if cfg.save_audit_artifacts or cfg.experiment_mode == "research":
        dataset.data_manifest.to_csv(
            output_directory / "data_manifest.csv", index=False
        )
        dataset.skipped_samples.to_csv(
            output_directory / "skipped_samples.csv", index=False
        )
        split.leakage_audit.to_csv(
            output_directory / "leakage_audit.csv", index=False
        )
        pd.DataFrame([{
            "train_end": split.train_cutoff,
            "validation_end": split.validation_cutoff,
            "calibration_end": split.calibration_cutoff,
            "test_start": predictions["origin_date"].min(),
            "test_end": predictions["target_end_date"].max(),
            "horizon": cfg.horizon,
            "test_window_id": cfg.test_window_id,
        }]).to_csv(output_directory / "split_boundaries.csv", index=False)

    for model_index, history in enumerate(histories, start=1):
        pd.DataFrame(history).to_csv(
            output_directory / f"training_history_model_{model_index}.csv",
            index=False,
        )

    torch.save({
        "model_state_dicts": [model.state_dict() for model in models],
        "config": asdict(cfg),
        "training_tickers": cfg.training_tickers,
        "unseen_test_tickers": cfg.unseen_test_tickers,
        "canonical_feature_names": dataset.feature_names,
        "active_feature_indices": split.active_feature_indices,
        "active_feature_names": split.active_feature_names,
        "dropped_feature_names": split.dropped_feature_names,
        "context_feature_names": split.context_feature_names,
        "data_source": dataset.data_source,
        "available_market_symbols": dataset.available_market_symbols,
        "winsor_lower": split.standardiser.lower,
        "winsor_upper": split.standardiser.upper,
        "feature_mean": split.standardiser.mean,
        "feature_scale": split.standardiser.scale,
        "ticker_standardisers": {
            ticker: asdict(standardiser)
            for ticker, standardiser in split.ticker_standardisers.items()
        },
        "ticker_thresholds": split.ticker_thresholds,
        "class_weights": split.class_weights,
        "train_regime_proportions": split.train_regime_proportions,
        "transition_matrices": split.transition_matrices,
        "initial_regime_probabilities": split.initial_regime_probabilities,
        "interval_scale": interval_scale,
        "interval_tail_calibration": asdict(interval_tail_calibration),
        "regime_temperature": regime_temperature,
        "blend_weight": blend_weight,
        "split_cutoffs": {
            "train": str(split.train_cutoff),
            "validation": str(split.validation_cutoff),
            "calibration": str(split.calibration_cutoff),
        },
    }, output_directory / "sigflow_v4_ensemble.pt")

    data_manifest_csv = dataset.data_manifest.to_csv(index=False).encode("utf-8")
    reproducibility = collect_reproducibility_manifest(Path.cwd())
    source_hashes = {}
    for relative_path in (
        "sigflow_v4_research.py",
        "sigflow_v4/data.py",
        "sigflow_v4/protocol.py",
        "requirements.txt",
        "SigFlow_v4_Research_Runner.ipynb",
    ):
        source_path = Path(relative_path)
        if source_path.is_file():
            source_hashes[relative_path] = sha256_bytes(source_path.read_bytes())
    reproducibility.update({
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "data_manifest_sha256": hashlib.sha256(data_manifest_csv).hexdigest(),
        "model_state_sha256": model_state_hash(models),
        "source_file_sha256": source_hashes,
        "ensemble_seeds": list(cfg.ensemble_seeds),
        "best_epochs": evaluation["training_summary"]["best_epoch"].tolist(),
    })
    write_json_atomic(
        output_directory / "reproducibility_manifest.json", reproducibility
    )
    write_json_atomic(
        output_directory / "run_config.json",
        {
            **asdict(cfg),
            "target_tickers": list(cfg.target_tickers),
            "output_dir": cfg.output_dir,
            "actual_data_source": dataset.data_source,
            "available_market_symbols": list(dataset.available_market_symbols),
            "active_feature_names": split.active_feature_names,
            "dropped_feature_names": split.dropped_feature_names,
            "interval_scale": interval_scale,
            "interval_tail_calibration": asdict(interval_tail_calibration),
            "regime_temperature": regime_temperature,
            "blend_weight": blend_weight,
            "data_manifest_sha256": reproducibility["data_manifest_sha256"],
            "git_commit": reproducibility["git_commit"],
        },
    )
    if cfg.save_plots:
        save_diagnostic_plots(
            predictions,
            evaluation,
            cfg,
            output_directory=output_directory,
        )

    if transactional_research_save:
        # Every canonical numeric artifact is complete before this same-filesystem
        # directory rename. Interrupted writes remain visibly quarantined under
        # the hidden staging name and never masquerade as a completed run.
        if destination_directory.exists():
            raise FileExistsError(
                f"Research output appeared during staging: {destination_directory}"
            )
        os.replace(output_directory, destination_directory)
        fsync_directory(destination_directory.parent)

    print("\nSaved result artifacts to:", destination_directory.resolve())

# %% [markdown cell 29]
# ## Final-test discipline
# 
# The final evaluation contains only dates after the calibration cutoff:
# 
# - `seen_ticker_unseen_day`: trained tickers evaluated on future dates;
# - `unseen_ticker_unseen_day`: completely held-out tickers evaluated on the identical future-date calendar.
# 
# The 22 training and 22 unseen tickers contain two names from each of 11 sectors. Held-out tickers never enter training, checkpoint selection, preprocessing, calibration, regime boundaries, transition priors, or HAR fitting. Optional feature selection uses three expanding inner folds that end no later than the canonical training cutoff.
# 
# The prespecified confirmatory claim is unseen-ticker QLIKE versus the transferable HAR baseline. Support requires real final-test data, the configured minimum of common origins, at least a 5% relative QLIKE improvement, a two-way-bootstrap 95% interval wholly below zero, consistent seed direction, and favourable effect direction at every prespecified date-block length. Pooled, sector, individual-day, and alternative-block results remain descriptive or robustness evidence.
# 
# Adding a modelling decision after inspecting final-test results turns this test into development evidence; create a later holdout period before making another final claim.

# %% [markdown cell 30]
# ## Reading the result tables
# 
# `detailed_test_results.csv` contains one result row per ticker and forecast origin. It includes the forecast window, sector, actual volatility, median and QLIKE-optimal forecasts, quantiles, per-seed forecasts, baselines, errors, 5/10/20/30/50% accuracy flags, interval coverage, relative-volatility regimes, CRPS, PIT, and NLL.
# 
# A “successful day” means the median forecast issued on `origin_date` is within the configured 20% tolerance for realised volatility over `target_start_date` through `target_end_date`. It is not a one-session target and individual days are not significance tests.
# 
# `primary_claim_summary.csv` is the concise verdict. `inference_summary.csv` contains all formal and secondary comparisons, while the two bootstrap files expose block-length and ticker-population sensitivity. Synthetic runs are always labelled `pipeline_validation_only` regardless of their numerical scores.

# %% [markdown cell 31]
# # Run the configured watertight experiment

# %% [cell 32]
def display_scrollable_results(
    table: pd.DataFrame,
    title: str,
    show_every_row: bool,
    height_px: int = 620,
) -> None:
    display(HTML(f"<h3>{title}</h3>"))
    if not show_every_row:
        display(table.head(100))
        print(
            f"Showing the first {min(100, len(table)):,} of {len(table):,} rows."
        )
        return
    rendered = table.to_html(
        index=False,
        border=0,
        float_format=lambda value: f"{value:.6g}",
    )
    display(HTML(
        f'<div style="max-height:{height_px}px; overflow:auto; '
        'border:1px solid #bbb; padding:4px">'
        f"{rendered}</div>"
    ))
    print(f"Rendered all {len(table):,} result rows in the scrollable table.")


def attach_run_identifiers(
    evaluation: dict[str, object],
    cfg: Config,
    evaluation_split: str,
) -> None:
    for value in evaluation.values():
        if isinstance(value, pd.DataFrame):
            if "run_name" not in value.columns:
                value.insert(0, "run_name", cfg.run_name)
            if "evaluation_split" not in value.columns:
                value.insert(1, "evaluation_split", evaluation_split)
            for column, value_to_add in (
                ("experiment_mode", cfg.experiment_mode),
                ("horizon", cfg.horizon),
                ("test_window_id", cfg.test_window_id),
            ):
                if column not in value.columns:
                    value.insert(len(value.columns), column, value_to_add)


def validate_dataset_compatibility(
    cfg: Config,
    dataset: MarketDataset,
) -> None:
    expected = dataset_construction_spec(cfg)
    mismatches = {
        key: (dataset.construction_config.get(key), expected_value)
        for key, expected_value in expected.items()
        if dataset.construction_config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Injected dataset is incompatible with config: {mismatches}")
    observed_tickers = set(dataset.metadata["ticker"])
    if observed_tickers != set(cfg.target_tickers):
        raise ValueError("Dataset ticker universe does not match the configuration.")
    if not (dataset.metadata["horizon"].to_numpy(dtype=int) == cfg.horizon).all():
        raise ValueError("Dataset contains the wrong forecast horizon.")
    synthetic = dataset.data_source.startswith("synthetic")
    if cfg.experiment_mode == "research" and synthetic:
        raise ValueError("Research mode cannot accept a synthetic dataset.")
    if cfg.experiment_mode == "pipeline_test" and not synthetic:
        raise ValueError("pipeline_test cannot accept a real-market dataset.")
    if cfg.experiment_mode == "research":
        validate_research_data_manifest(dataset.data_manifest, cfg)
        if (
            "data_provenance_valid" not in dataset.metadata
            or not dataset.metadata["data_provenance_valid"].astype(bool).all()
        ):
            raise ValueError(
                "Research dataset rows are not linked to validated provenance."
            )


def prospective_final_config(
    cfg: Config,
    *,
    authorized: bool,
) -> Config:
    return replace(
        cfg,
        run_name=f"prospective_final_{cfg.prospective_test_start_date}_{cfg.prospective_test_end_date}",
        end_date=cfg.prospective_data_end_date,
        explicit_train_end_date=cfg.final_train_end_date,
        explicit_validation_end_date=cfg.final_validation_end_date,
        explicit_calibration_end_date=cfg.final_calibration_end_date,
        explicit_test_start_date=cfg.prospective_test_start_date,
        explicit_test_origin_end_date=cfg.prospective_test_end_date,
        explicit_test_origin_dates=(),
        # The frozen test end is the last forecast origin. The later data-end
        # buffer lets every origin's multi-session target fully mature.
        explicit_test_end_date=cfg.prospective_data_end_date,
        test_window_id=(
            f"prospective_{cfg.prospective_test_start_date}_"
            f"{cfg.prospective_test_end_date}"
        ),
        final_evaluation_authorized=authorized,
        overwrite_existing_outputs=False,
    )


def final_protocol_hash(cfg: Config) -> str:
    payload = asdict(cfg)
    # The confirmation bit is operational authorization, not a model choice.
    payload.pop("final_evaluation_authorized", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def validate_final_evaluation_protocol(
    cfg: Config,
    *,
    save_artifacts: bool,
) -> str:
    if not cfg.final_evaluation_authorized:
        raise PermissionError(
            "Prospective test evaluation requires explicit authorization."
        )
    if not save_artifacts:
        raise ValueError(
            "Prospective evaluation must save its immutable audit ledger."
        )
    if datetime.now(timezone.utc).date() < pd.Timestamp(
        cfg.prospective_data_end_date
    ).date():
        raise RuntimeError(
            "The preregistered prospective window is not complete; do not "
            f"evaluate before {cfg.prospective_data_end_date}."
        )
    observed_hash = final_protocol_hash(cfg)
    if observed_hash != PREREGISTERED_FINAL_PROTOCOL_SHA256:
        raise ValueError(
            "Prospective configuration does not match the preregistered "
            f"protocol hash: {observed_hash}."
        )
    return observed_hash


def final_evaluation_ledger_path(cfg: Config) -> Path:
    return (
        Path(cfg.output_root)
        / "research"
        / "prospective_ledgers"
        / (
            f"{cfg.prospective_test_start_date}_"
            f"{cfg.prospective_test_end_date}.json"
        )
    )


def acquire_final_evaluation_ledger(
    cfg: Config,
    protocol_hash: str,
) -> Path:
    path = final_evaluation_ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "started_and_locked",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "prospective_test_start_date": cfg.prospective_test_start_date,
        "prospective_test_end_date": cfg.prospective_test_end_date,
        "process_id": os.getpid(),
        "note": (
            "A failed or interrupted run remains locked for manual audit; "
            "never delete this ledger merely to obtain another test look."
        ),
    }
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            "The prospective window has already been started or evaluated; "
            f"refusing another look. Ledger: {path.resolve()}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def finalize_final_evaluation_ledger(
    path: Path,
    cfg: Config,
    predictions: pd.DataFrame,
    models: list[nn.Module],
) -> None:
    write_json_atomic(path, {
        "status": "completed_write_once",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": final_protocol_hash(cfg),
        "model_state_sha256": model_state_hash(models),
        "output_dir": str(Path(cfg.output_dir).resolve()),
        "prediction_rows": len(predictions),
        "first_origin_date": str(
            pd.to_datetime(predictions["origin_date"]).min().date()
        ),
        "last_origin_date": str(
            pd.to_datetime(predictions["origin_date"]).max().date()
        ),
        "last_target_end_date": str(
            pd.to_datetime(predictions["target_end_date"]).max().date()
        ),
    })


def main(
    cfg: Config = CFG,
    dataset: MarketDataset | None = None,
    evaluation_split: str = "test",
    show_plots: bool = True,
    display_tables: bool = True,
    save_artifacts: bool = True,
) -> dict[str, object]:
    validate_config(cfg)
    if evaluation_split not in {"test", "rolling_test", "validation"}:
        raise ValueError(
            "evaluation_split must be test, rolling_test, or validation."
        )
    final_ledger: Path | None = None
    if cfg.experiment_mode == "research" and evaluation_split == "test":
        protocol_hash = validate_final_evaluation_protocol(
            cfg, save_artifacts=save_artifacts
        )
        # This exclusive lock is acquired before data construction, target
        # inference, metric computation, printing, or plotting.
        final_ledger = acquire_final_evaluation_ledger(cfg, protocol_hash)
    elif (
        cfg.experiment_mode == "research"
        and pd.Timestamp(cfg.end_date) > pd.Timestamp(cfg.decision_freeze_date)
    ):
        raise ValueError(
            "Development and rolling research are frozen at "
            f"{cfg.decision_freeze_date}; later data belong to the reserved "
            "prospective holdout."
        )

    print("=" * 78, flush=True)
    print(f"Starting SigFlow-Sim v4 | run={cfg.run_name}", flush=True)
    print("Mode:", cfg.experiment_mode, flush=True)
    print("Evaluation:", evaluation_split, flush=True)
    print("=" * 78, flush=True)
    if DEVICE.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0), flush=True)
    elif not cfg.quick_mode:
        print(
            "WARNING: CUDA is unavailable; this real profile will run on CPU "
            "and can take substantially longer.",
            flush=True,
        )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    set_seed(cfg.seed)

    if dataset is None:
        dataset = build_market_dataset(cfg)
    validate_dataset_compatibility(cfg, dataset)
    split = prepare_split(dataset, cfg)
    print(
        "Prepared rows | "
        f"train={len(split.train_indices):,}, "
        f"validation={len(split.validation_indices):,}, "
        f"calibration={len(split.calibration_indices):,}, "
        f"test={len(split.test_indices):,}",
        flush=True,
    )
    baseline_predictions = fit_stronger_baselines(dataset, split, cfg)
    har_predictions = np.asarray(baseline_predictions["log_har"], dtype=float)
    configure_har_forecasting_problem(
        dataset, split, har_predictions, cfg
    )
    models, histories, best_epochs = fit_ensemble(dataset, split, cfg)
    pre_calibration_model_hash = model_state_hash(models)
    print("\nBest validation epochs:", best_epochs)

    if evaluation_split in {"test", "rolling_test"}:
        interval_scale = 1.0
        interval_tail_calibration = IntervalTailCalibration(
            scale=1.0, corrections={"pooled": {}}
        )
        regime_temperature = 1.0
        blend_weight = 1.0
        print(
            "\nGenerating calibration forecasts "
            "(seen tickers only; never used for model fitting)..."
        )
        calibration_metadata = dataset.metadata.iloc[
            split.calibration_indices
        ].copy().reset_index(drop=True)
        calibration_dates = np.array(sorted(pd.to_datetime(
            calibration_metadata["origin_date"].unique()
        )))
        if len(calibration_dates) < 2:
            raise ValueError(
                "Need at least two calibration origin dates to separate "
                "selection from conformal correction."
            )
        partition_at = max(1, len(calibration_dates) // 2)
        selection_dates = set(calibration_dates[:partition_at])
        provisional_selection_mask = pd.to_datetime(
            calibration_metadata["origin_date"]
        ).isin(selection_dates).to_numpy()
        conformal_mask = ~provisional_selection_mask
        first_conformal_target_start = pd.to_datetime(
            calibration_metadata.loc[conformal_mask, "target_start_date"]
        ).min()
        selection_mask = (
            provisional_selection_mask
            & (
                pd.to_datetime(calibration_metadata["target_end_date"])
                < first_conformal_target_start
            ).to_numpy()
        )
        selection_indices = split.calibration_indices[selection_mask]
        conformal_indices = split.calibration_indices[conformal_mask]
        if len(selection_indices) == 0 or len(conformal_indices) == 0:
            raise ValueError("Purged calibration partition produced an empty role.")
        calibration_partition = calibration_metadata[[
            "ticker", "origin_date", "target_start_date", "target_end_date"
        ]].copy()
        calibration_partition["calibration_role"] = np.select(
            [selection_mask, conformal_mask],
            [
                "scale_temperature_blend_selection",
                "tail_conformal_correction",
            ],
            default="purged_internal_calibration_embargo",
        )
        append_audit_check(
            split,
            "calibration selection targets end before conformal targets begin",
            pd.to_datetime(
                calibration_metadata.loc[selection_mask, "target_end_date"]
            ).max() < first_conformal_target_start,
            pd.to_datetime(
                calibration_metadata.loc[selection_mask, "target_end_date"]
            ).max(),
            first_conformal_target_start,
        )
        print(
            "  Calibration partition: "
            f"selection={len(selection_indices):,}, "
            f"purged={int((~selection_mask & ~conformal_mask).sum()):,}, "
            f"tail correction={len(conformal_indices):,}",
            flush=True,
        )
        selection_outputs = collect_ensemble_outputs(
            models,
            dataset,
            split,
            selection_indices,
            cfg,
        )
        if cfg.calibrate_intervals:
            interval_scale = calibrate_interval_scale(
                selection_outputs["log_samples"],
                dataset.targets_log_vol[selection_indices],
                har_predictions[selection_indices],
                cfg,
            )
        if cfg.calibrate_regime_probabilities:
            regime_temperature = calibrate_regime_temperature(
                selection_outputs["model_logits"],
                split.regime_labels[selection_indices],
                cfg,
            )
        if cfg.calibrate_intervals:
            conformal_outputs = collect_ensemble_outputs(
                models,
                dataset,
                split,
                conformal_indices,
                cfg,
            )
            conformal_probabilities = calibrated_probabilities(
                conformal_outputs["model_logits"], regime_temperature
            )
            interval_tail_calibration = fit_interval_tail_calibration(
                conformal_outputs["log_samples"],
                har_predictions[conformal_indices],
                dataset.targets_log_vol[conformal_indices],
                dataset.metadata.iloc[conformal_indices].reset_index(drop=True),
                conformal_probabilities,
                interval_scale,
                cfg,
            )
        blend_started = time.perf_counter()
        blend_weight = select_har_blend_weight(
            selection_outputs["log_samples"],
            har_predictions[selection_indices],
            dataset.targets_log_vol[selection_indices],
            interval_scale,
            cfg,
        )
        baseline_predictions["blend_selection_seconds"] = np.asarray([
            time.perf_counter() - blend_started
        ])

        print(
            "\nGenerating both evaluation cohorts on future origins..."
        )
        evaluation_indices = split.test_indices
        cohort_override = None
    else:
        # Development ablations use no calibration and never inspect test rows.
        interval_scale = 1.0
        interval_tail_calibration = IntervalTailCalibration(
            scale=1.0, corrections={"pooled": {}}
        )
        regime_temperature = 1.0
        blend_weight = 1.0
        calibration_partition = pd.DataFrame(columns=[
            "ticker", "origin_date", "target_start_date", "target_end_date",
            "calibration_role",
        ])
        evaluation_indices = split.validation_indices
        cohort_override = "development_validation_seen_tickers"
        print("\nGenerating development-validation forecasts only...")

    post_calibration_model_hash = model_state_hash(models)
    append_audit_check(
        split,
        "calibration does not modify model parameters",
        pre_calibration_model_hash == post_calibration_model_hash,
        post_calibration_model_hash,
        pre_calibration_model_hash,
    )
    append_audit_check(
        split,
        "forecast parameters are invariant to evaluation-target permutation",
        audit_target_invariance(
            models[0], dataset, split, evaluation_indices, cfg
        ),
        "identical logits/locations/scales",
        "identical logits/locations/scales",
    )

    inference_started = time.perf_counter()
    outputs = collect_ensemble_outputs(
        models,
        dataset,
        split,
        evaluation_indices,
        cfg,
    )
    inference_seconds = time.perf_counter() - inference_started
    predictions = build_prediction_frame(
        outputs,
        dataset,
        split,
        evaluation_indices,
        interval_scale,
        interval_tail_calibration,
        regime_temperature,
        baseline_predictions,
        blend_weight,
        cfg,
        cohort_override=cohort_override,
    )
    evaluation = evaluate_predictions(
        predictions,
        cfg,
        data_source=dataset.data_source,
        evaluation_split=evaluation_split,
    )
    evaluation["calibration_partition"] = calibration_partition
    regime_diagnostics, regime_confusions = build_full_regime_diagnostics(
        models,
        dataset,
        split,
        regime_temperature,
        cfg,
        sections_to_report=(
            ("train", "validation")
            if evaluation_split == "validation"
            else ("train", "validation", "calibration", "test")
        ),
    )
    evaluation["regime_diagnostics"] = regime_diagnostics
    evaluation["regime_confusion_matrices"] = regime_confusions
    evaluation["training_summary"] = build_training_summary(
        histories, best_epochs, cfg
    )
    evaluation["complexity_summary"] = build_complexity_summary(
        models,
        histories,
        inference_seconds,
        len(evaluation_indices),
        evaluation["point_metrics"],
        evaluation["per_seed_metrics"],
        baseline_predictions,
        cfg,
    )
    evaluation["structured_experiment_summary"] = (
        build_structured_experiment_summary(
            evaluation, evaluation["complexity_summary"], cfg
        )
    )
    attach_run_identifiers(evaluation, cfg, evaluation_split)

    print("\nPrespecified primary claim:")
    print(evaluation["primary_claim_summary"].round(5).to_string(index=False))
    print("\nPer-seed stability:")
    print(evaluation["per_seed_stability"].round(5).to_string(index=False))
    print("\nTraining runtime and checkpoint summary:")
    print(evaluation["training_summary"].round(5).to_string(index=False))
    print("\nModel complexity and inference cost:")
    print(evaluation["complexity_summary"].round(5).to_string(index=False))
    print("\nSigFlow point metrics (pooled, cohort, sector, and ticker):")
    print(
        evaluation["point_metrics"].query(
            "model == 'SigFlow v4'"
        ).round(5).to_string(index=False)
    )
    print("\nMultiple accuracy definitions:")
    print(
        evaluation["accuracy_summary"].round(5).to_string(index=False)
    )
    print(
        "\nBaseline comparison scorecard "
        "(negative error differences favour SigFlow):"
    )
    print(
        evaluation["baseline_comparison_summary"].round(5).to_string(index=False)
    )
    print("\nProbabilistic accuracy and interval coverage:")
    print(
        evaluation["probabilistic_summary"].round(5).to_string(index=False)
    )
    print("\nRaw regime accuracy:")
    print(
        evaluation["raw_regime_metrics"].round(5).to_string(index=False)
    )
    print("\nTrue/predicted regime counts and average probabilities:")
    print(regime_diagnostics.round(5).to_string(index=False))
    msft = evaluation["point_metrics"].query(
        "ticker == 'MSFT' and model == 'SigFlow v4'"
    )
    if not msft.empty:
        print("\nMSFT under-response diagnostic:")
        print(msft[[
            "correlation", "actual_standard_deviation",
            "predicted_standard_deviation", "prediction_to_actual_std_ratio",
            "best_prediction_lag_sessions", "best_lagged_correlation",
        ]].round(5).to_string(index=False))
    if cfg.apply_regime_smoothing:
        print("\nPast-only smoothed regime accuracy (optional ablation):")
        print(
            evaluation["smoothed_regime_metrics"].round(5).to_string(index=False)
        )
    if evaluation["paired_bootstrap"] is not None:
        print(
            "\nPaired date-block bootstrap differences "
            "(negative favours SigFlow):"
        )
        print(
            evaluation["paired_bootstrap"].round(5).to_string(index=False)
        )

    print("\nActual data source:", dataset.data_source)
    if dataset.data_source.startswith("synthetic"):
        print("WARNING: synthetic results validate the pipeline only.")

    if save_artifacts:
        save_results(
            models,
            histories,
            predictions,
            evaluation,
            dataset,
            split,
            interval_scale,
            interval_tail_calibration,
            regime_temperature,
            blend_weight,
            cfg,
        )
    if show_plots:
        plot_results(histories, predictions, evaluation, cfg)

    if display_tables:
        cohort_rows = evaluation["accuracy_summary"].query(
            "group_level in ['cohort', 'cohort_ticker']"
        )
        display_scrollable_results(
            cohort_rows,
            "Accuracy by unseen-data cohort and ticker",
            show_every_row=True,
            height_px=520,
        )
        display_scrollable_results(
            evaluation["daily_success_summary"],
            "Which forecast-origin days succeeded within 20%",
            show_every_row=True,
            height_px=620,
        )
        display_scrollable_results(
            evaluation["detailed_test_results"],
            "Every test result value (result fields only; no raw OHLCV/features)",
            show_every_row=(
                cfg.display_all_test_results and evaluation_split == "test"
            ),
            height_px=720,
        )

    if final_ledger is not None:
        finalize_final_evaluation_ledger(
            final_ledger, cfg, predictions, models
        )
    print("\nComplete.")
    return {
        "dataset": dataset,
        "split": split,
        "models": models,
        "histories": histories,
        "best_epochs": best_epochs,
        "predictions": predictions,
        "evaluation": evaluation,
        "interval_scale": interval_scale,
        "interval_tail_calibration": interval_tail_calibration,
        "regime_temperature": regime_temperature,
        "blend_weight": blend_weight,
        "evaluation_split": evaluation_split,
    }


def make_ablation_config(
    base_cfg: Config,
    name: str,
    overrides: dict[str, object],
) -> Config:
    updates = dict(overrides)
    if "disabled_feature_groups" in updates:
        updates["disabled_feature_groups"] = tuple(dict.fromkeys(
            base_cfg.disabled_feature_groups
            + tuple(updates["disabled_feature_groups"])
        ))
    if "disabled_feature_names" in updates:
        updates["disabled_feature_names"] = tuple(dict.fromkeys(
            base_cfg.disabled_feature_names
            + tuple(updates["disabled_feature_names"])
        ))
    updates["run_name"] = f"{base_cfg.profile_name}_ablation_{name}"
    updates["run_block_bootstrap"] = False
    if base_cfg.ablation_reduced_budget:
        reduced_epochs = min(base_cfg.epochs, base_cfg.ablation_epochs)
        reduced_patience = min(base_cfg.patience, base_cfg.ablation_patience)
        updates.update({
            "epochs": reduced_epochs,
            "minimum_epochs": min(base_cfg.minimum_epochs, reduced_epochs),
            "patience": reduced_patience,
            "scheduler_patience": min(
                base_cfg.scheduler_patience,
                max(1, reduced_patience // 2),
            ),
            "ensemble_seeds": base_cfg.ablation_ensemble_seeds,
            "prediction_samples": min(256, base_cfg.prediction_samples),
        })
    return replace(base_cfg, **updates)


def inner_validation_fold_configs(
    base_cfg: Config,
) -> list[Config]:
    heldout_fraction = (
        base_cfg.train_fraction
        * base_cfg.inner_validation_fraction_of_training
    )
    fold_width = heldout_fraction / base_cfg.inner_validation_folds
    initial_train_fraction = base_cfg.train_fraction - heldout_fraction
    folds = []
    for fold_index in range(base_cfg.inner_validation_folds):
        fold_train_fraction = initial_train_fraction + fold_index * fold_width
        fold_cfg = replace(
            base_cfg,
            train_fraction=fold_train_fraction,
            validation_fraction=fold_width,
            run_name=f"{base_cfg.run_name}_inner_fold_{fold_index + 1}",
        )
        validate_config(fold_cfg)
        if (
            fold_cfg.train_fraction + fold_cfg.validation_fraction
            > base_cfg.train_fraction + 1e-12
        ):
            raise AssertionError("Inner validation crossed outer training cutoff.")
        folds.append(fold_cfg)
    return folds


def summarise_validation_ablation(
    name: str,
    description: str,
    fold_index: int,
    result: dict[str, object],
) -> dict[str, object]:
    point = result["evaluation"]["point_metrics"].query(
        "group_level == 'pooled' and model == 'SigFlow v4'"
    ).iloc[0]
    accuracy = result["evaluation"]["accuracy_summary"].query(
        "group_level == 'pooled'"
    ).iloc[0]
    return {
        "ablation": name,
        "description": description,
        "inner_fold": fold_index,
        "fold_train_cutoff": result["split"].train_cutoff,
        "fold_validation_cutoff": result["split"].validation_cutoff,
        "active_engineered_features": len(
            result["split"].active_feature_names
        ),
        "context_dimension": result["split"].context.shape[1],
        "validation_mae": float(point["mae"]),
        "validation_rmse": float(point["rmse"]),
        "validation_mape_percent": float(point["mape_percent"]),
        "validation_qlike": float(point["qlike"]),
        "validation_within_10pct_accuracy": float(
            accuracy["sigflow_within_10pct_accuracy"]
        ),
        "validation_within_20pct_accuracy": float(
            accuracy["sigflow_within_20pct_accuracy"]
        ),
        "validation_within_30pct_accuracy": float(
            accuracy["sigflow_within_30pct_accuracy"]
        ),
        "validation_raw_regime_accuracy": float(
            accuracy["raw_regime_accuracy"]
        ),
    }


def run_configured_validation_ablations(
    base_cfg: Config,
    dataset: MarketDataset,
) -> pd.DataFrame:
    jobs: list[tuple[str, str, dict[str, object]]] = [
        ("full_reference", "Full configured feature and objective reference.", {})
    ]

    if RUN_GROUP_ABLATIONS:
        unknown = sorted(set(SELECTED_GROUP_ABLATIONS) - set(GROUP_ABLATIONS))
        if unknown:
            raise ValueError(
                f"Unknown selected group ablations: {unknown}. "
                f"Valid names: {sorted(GROUP_ABLATIONS)}"
            )
        for name in SELECTED_GROUP_ABLATIONS:
            spec = GROUP_ABLATIONS[name]
            jobs.append((spec.name, spec.description, dict(spec.overrides)))

    if RUN_INDIVIDUAL_FEATURE_ABLATIONS:
        _, base_active_names, _ = resolve_active_feature_set(dataset, base_cfg)
        feature_names = (
            INDIVIDUAL_FEATURES_TO_ABLATE
            if INDIVIDUAL_FEATURES_TO_ABLATE
            else tuple(base_active_names)
        )
        unknown_features = sorted(set(feature_names) - set(dataset.feature_names))
        if unknown_features:
            raise ValueError(
                f"Unknown individual feature ablations: {unknown_features}"
            )
        already_inactive = sorted(set(feature_names) - set(base_active_names))
        if already_inactive:
            raise ValueError(
                "Individual ablations must name currently active features; "
                f"already inactive: {already_inactive}"
            )
        for feature_name in feature_names:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_name)
            jobs.append((
                f"without_{safe_name}",
                f"Remove only canonical feature {feature_name}.",
                {"disabled_feature_names": (feature_name,)},
            ))

    fold_count = base_cfg.inner_validation_folds
    print(
        f"\nRunning {len(jobs):,} configurations across {fold_count} "
        "expanding inner-validation folds. Every fold ends at or before the "
        "canonical training cutoff; outer validation and final test are not "
        "forecast in this selection loop."
    )
    summaries = []
    structured_summaries: list[pd.DataFrame] = []
    for job_index, (name, description, overrides) in enumerate(jobs, start=1):
        print(f"\nAblation {job_index}/{len(jobs)}: {name} — {description}")
        ablation_cfg = make_ablation_config(base_cfg, name, overrides)
        fold_configs = inner_validation_fold_configs(ablation_cfg)
        for fold_index, fold_cfg in enumerate(fold_configs, start=1):
            print(
                f"  Inner fold {fold_index}/{len(fold_configs)} | "
                f"train fraction={fold_cfg.train_fraction:.3f} | "
                f"validation fraction={fold_cfg.validation_fraction:.3f}"
            )
            result = main(
                fold_cfg,
                dataset=dataset,
                evaluation_split="validation",
                show_plots=False,
                display_tables=False,
                save_artifacts=False,
            )
            summaries.append(summarise_validation_ablation(
                name, description, fold_index, result
            ))
            structured = result["evaluation"][
                "structured_experiment_summary"
            ].copy()
            structured.insert(0, "ablation", name)
            structured.insert(1, "ablation_description", description)
            structured.insert(2, "inner_fold", fold_index)
            structured["fold_train_cutoff"] = result["split"].train_cutoff
            structured["fold_validation_cutoff"] = (
                result["split"].validation_cutoff
            )
            structured_summaries.append(structured)
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    detailed = pd.DataFrame(summaries)
    metric_columns = [
        "validation_mae",
        "validation_rmse",
        "validation_mape_percent",
        "validation_qlike",
        "validation_within_10pct_accuracy",
        "validation_within_20pct_accuracy",
        "validation_within_30pct_accuracy",
        "validation_raw_regime_accuracy",
    ]
    identity_columns = [
        "ablation", "description", "active_engineered_features",
        "context_dimension",
    ]
    aggregate = detailed.groupby(
        identity_columns, as_index=False, dropna=False
    )[metric_columns].agg(["mean", "std"])
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple) else str(column)
        for column in aggregate.columns
    ]
    aggregate["inner_fold_count"] = fold_count
    aggregate = aggregate.sort_values(
        ["validation_qlike_mean", "validation_mae_mean"]
    ).reset_index(drop=True)

    selection_destination = (
        Path(base_cfg.output_root)
        / base_cfg.experiment_mode
        / "development_selection"
        / base_cfg.run_name
    )
    transactional_selection_save = (
        base_cfg.experiment_mode == "research"
        and not base_cfg.overwrite_existing_outputs
    )
    if transactional_selection_save:
        if selection_destination.exists():
            raise FileExistsError(
                "Refusing to overwrite immutable development selection: "
                f"{selection_destination}"
            )
        selection_destination.parent.mkdir(parents=True, exist_ok=True)
        output_directory = selection_destination.parent / (
            f".{selection_destination.name}.staging."
            f"{os.getpid()}.{time.time_ns()}"
        )
        output_directory.mkdir(exist_ok=False)
    else:
        output_directory = selection_destination
        output_directory.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(
        output_directory / "development_inner_fold_ablation_details.csv",
        index=False,
    )
    aggregate.to_csv(
        output_directory / "development_validation_ablation_summary.csv",
        index=False,
    )
    pd.concat(structured_summaries, ignore_index=True, sort=False).to_csv(
        output_directory
        / "development_validation_ablation_structured_summary.csv",
        index=False,
    )
    gate_candidates = aggregate[
        aggregate["ablation"].isin([
            "no_regimes", "regime_auxiliary_only", "soft_regime_gate"
        ])
    ].copy()
    if len(gate_candidates) == 3:
        best = gate_candidates.sort_values(
            ["validation_qlike_mean", "validation_mae_mean"]
        ).iloc[0]
        soft = gate_candidates[
            gate_candidates["ablation"] == "soft_regime_gate"
        ].iloc[0]
        recommendation = pd.DataFrame([{
            "selected_on": "inner development validation only",
            "best_gate_variant": best["ablation"],
            "best_validation_qlike": best["validation_qlike_mean"],
            "soft_gate_validation_qlike": soft["validation_qlike_mean"],
            "remove_regime_gate": best["ablation"] != "soft_regime_gate",
            "test_results_consulted": False,
        }])
        recommendation.to_csv(
            output_directory / "regime_gate_validation_recommendation.csv",
            index=False,
        )
    if transactional_selection_save:
        os.replace(output_directory, selection_destination)
        fsync_directory(selection_destination.parent)
    display_scrollable_results(
        aggregate,
        "Inner-fold ablation comparison ranked by mean validation QLIKE",
        show_every_row=True,
        height_px=620,
    )
    return aggregate


def apply_validation_selected_gate(
    cfg: Config,
    ablation_summary: pd.DataFrame,
) -> Config:
    candidates = ablation_summary[
        ablation_summary["ablation"].isin([
            "no_regimes", "regime_auxiliary_only", "soft_regime_gate"
        ])
    ].sort_values(["validation_qlike_mean", "validation_mae_mean"])
    if candidates.empty:
        return cfg
    selected = str(candidates.iloc[0]["ablation"])
    if selected == "soft_regime_gate":
        return replace(
            cfg,
            regime_gate_mode="soft",
            expert_alignment_weight=0.0,
            gate_balance_weight=0.0,
        )
    if selected == "regime_auxiliary_only":
        print(
            "Validation did not support regime gating; retaining only the "
            "auxiliary regime task for subsequent runs."
        )
        return replace(
            cfg,
            regime_gate_mode="auxiliary",
            expert_alignment_weight=0.0,
            gate_balance_weight=0.0,
        )
    print(
        "Validation did not support regimes; removing regime loss and gate "
        "from subsequent runs."
    )
    return replace(
        cfg,
        regime_gate_mode="none",
        regime_classification_weight=0.0,
        expert_alignment_weight=0.0,
        gate_balance_weight=0.0,
    )


def config_for_profile(
    profile_name: str,
    *,
    run_name: str | None = None,
) -> Config:
    if profile_name not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown profile {profile_name!r}.")
    settings = PROFILE_SETTINGS[profile_name]
    quick = bool(settings["quick_mode"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = replace(
        Config(),
        profile_name=profile_name,
        quick_mode=quick,
        experiment_mode="pipeline_test" if quick else "research",
        run_name=run_name or f"{profile_name}_{timestamp}",
        training_tickers=tuple(settings["training_tickers"]),
        unseen_test_tickers=tuple(settings["unseen_test_tickers"]),
        use_real_market_data=bool(settings["use_real_market_data"]),
        allow_synthetic_fallback=bool(settings["allow_synthetic_fallback"]),
        start_date=str(settings["start_date"]),
        end_date=(
            str(settings.get("end_date", "2026-07-25"))
            if quick else datetime.now(timezone.utc).date().isoformat()
        ),
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        hidden_size=int(settings["hidden_size"]),
        patience=int(settings["patience"]),
        minimum_epochs=int(settings["minimum_epochs"]),
        scheduler_patience=int(settings["scheduler_patience"]),
        ensemble_seeds=tuple(settings["ensemble_seeds"]),
        prediction_samples=int(settings["prediction_samples"]),
        bootstrap_repetitions=int(settings["bootstrap_repetitions"]),
        run_block_bootstrap=bool(settings["run_block_bootstrap"]),
        minimum_test_origins_per_ticker=int(
            settings["minimum_test_origins_per_ticker"]
        ),
        print_every=1 if quick else 5,
        use_ticker_embeddings=not quick,
        ablation_reduced_budget=quick,
        ablation_ensemble_seeds=tuple(settings["ensemble_seeds"]),
    )
    validate_config(cfg)
    return cfg


def run_multi_horizon_rolling_research(
    cfg: Config,
    *,
    show_plots: bool = False,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    if cfg.experiment_mode != "research":
        raise ValueError("Rolling-origin evaluation is a real-data research run.")
    if pd.Timestamp(cfg.end_date) > pd.Timestamp(cfg.decision_freeze_date):
        cfg = replace(cfg, end_date=cfg.decision_freeze_date)
    maximum_horizon = max(cfg.forecast_horizons)
    anchor_cfg = replace(
        cfg,
        horizon=maximum_horizon,
        explicit_train_end_date=None,
        explicit_validation_end_date=None,
        explicit_calibration_end_date=None,
        explicit_test_start_date=None,
        explicit_test_origin_end_date=None,
        explicit_test_origin_dates=(),
        explicit_test_end_date=None,
        test_window_id="rolling_anchor",
    )
    anchor_dataset = build_market_dataset(anchor_cfg)
    per_ticker_dates = [
        set(pd.to_datetime(anchor_dataset.metadata.loc[
            anchor_dataset.metadata["ticker"] == ticker, "origin_date"
        ]))
        for ticker in cfg.target_tickers
    ]
    common_dates = sorted(set.intersection(*per_ticker_dates))
    maximum_embargo = maximum_horizon - 1
    effective_step = cfg.rolling_step_dates + maximum_embargo
    fixed_dates = (
        cfg.rolling_validation_dates
        + cfg.rolling_calibration_dates
        + cfg.rolling_test_dates
        + 3 * maximum_embargo
        + (cfg.rolling_origin_windows - 1) * effective_step
    )
    minimum_train_dates = max(504, len(common_dates) - fixed_dates)
    available_windows = make_rolling_origin_windows(
        common_dates,
        minimum_train_dates=minimum_train_dates,
        validation_dates=cfg.rolling_validation_dates,
        calibration_dates=cfg.rolling_calibration_dates,
        test_dates=cfg.rolling_test_dates,
        step_dates=effective_step,
        horizon_dates=maximum_horizon,
    )
    if len(available_windows) < cfg.rolling_origin_windows:
        raise ValueError(
            f"Rolling protocol requires {cfg.rolling_origin_windows} windows; "
            f"only {len(available_windows)} fit before the decision freeze."
        )
    windows = available_windows[-cfg.rolling_origin_windows:]
    if len(windows) != cfg.rolling_origin_windows:
        raise AssertionError("Rolling planner did not return the exact frozen window count.")
    anchor_target_ends = (
        anchor_dataset.metadata.assign(
            origin_date=pd.to_datetime(anchor_dataset.metadata["origin_date"]),
            target_end_date=pd.to_datetime(
                anchor_dataset.metadata["target_end_date"]
            ),
        )
        .groupby("origin_date")["target_end_date"]
        .max()
    )

    summaries: list[pd.DataFrame] = []
    summary_directory = (
        Path(cfg.output_root) / "research" / "rolling_origin_summaries"
    )
    if save_artifacts:
        summary_directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            summary_directory / f"{cfg.run_name}_shared_window_plan.json",
            {
                "anchor_horizon": maximum_horizon,
                "effective_step_dates": effective_step,
                "decision_freeze_date": cfg.decision_freeze_date,
                "windows": serialise_windows(windows),
            },
        )

    for horizon in cfg.forecast_horizons:
        horizon_cfg = replace(
            cfg,
            horizon=horizon,
            explicit_train_end_date=None,
            explicit_validation_end_date=None,
            explicit_calibration_end_date=None,
            explicit_test_start_date=None,
            explicit_test_origin_end_date=None,
            explicit_test_origin_dates=(),
            explicit_test_end_date=None,
            test_window_id="rolling_plan",
        )
        dataset = (
            anchor_dataset
            if horizon == maximum_horizon
            else build_market_dataset(horizon_cfg)
        )
        horizon_predictions: list[pd.DataFrame] = []
        horizon_complexities: list[pd.DataFrame] = []

        for window in windows:
            window_id = f"rolling_{window.window_id:02d}_{window.test_start}_{window.test_end}"
            planned_origin_dates = tuple(
                pd.Timestamp(value).date().isoformat()
                for value in common_dates
                if window.test_start <= pd.Timestamp(value).date() <= window.test_end
            )
            if len(planned_origin_dates) != cfg.rolling_test_dates:
                raise AssertionError(
                    f"Shared rolling window {window_id} has "
                    f"{len(planned_origin_dates)} origins, expected "
                    f"{cfg.rolling_test_dates}."
                )
            target_end = pd.Timestamp(anchor_target_ends.loc[pd.Timestamp(
                window.test_end
            )])
            window_cfg = replace(
                horizon_cfg,
                run_name=f"{cfg.run_name}_h{horizon}_{window_id}",
                explicit_train_end_date=window.train_end.isoformat(),
                explicit_validation_end_date=window.validation_end.isoformat(),
                explicit_calibration_end_date=window.calibration_end.isoformat(),
                explicit_test_start_date=window.test_start.isoformat(),
                explicit_test_origin_end_date=window.test_end.isoformat(),
                explicit_test_origin_dates=planned_origin_dates,
                explicit_test_end_date=target_end.date().isoformat(),
                test_window_id=window_id,
                minimum_test_origins_per_ticker=cfg.rolling_test_dates,
            )
            result = main(
                window_cfg,
                dataset=dataset,
                evaluation_split="rolling_test",
                show_plots=show_plots,
                display_tables=False,
                save_artifacts=save_artifacts,
            )
            observed_origins = set(
                pd.to_datetime(result["predictions"]["origin_date"])
                .dt.date.astype(str)
            )
            if observed_origins != set(planned_origin_dates):
                raise AssertionError(
                    f"{window_id} horizon {horizon} did not evaluate the exact "
                    "shared origin-date plan."
                )
            origins_per_ticker = result["predictions"].groupby(
                "ticker"
            )["origin_date"].nunique()
            if not (
                len(origins_per_ticker) == len(cfg.target_tickers)
                and (origins_per_ticker == len(planned_origin_dates)).all()
            ):
                raise AssertionError(
                    f"{window_id} horizon {horizon} has an incomplete ticker/date panel."
                )
            summary = result["evaluation"][
                "structured_experiment_summary"
            ].copy()
            summary["rolling_train_end"] = window.train_end
            summary["rolling_validation_end"] = window.validation_end
            summary["rolling_calibration_end"] = window.calibration_end
            summary["rolling_test_start"] = window.test_start
            summary["rolling_test_end"] = window.test_end
            summary["rolling_target_end"] = target_end
            summaries.append(summary)
            predictions = result["predictions"].copy()
            predictions["rolling_window_id"] = window_id
            horizon_predictions.append(predictions)
            horizon_complexities.append(
                result["evaluation"]["complexity_summary"].copy()
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregate_predictions = pd.concat(
            horizon_predictions, ignore_index=True, sort=False
        )
        aggregate_cfg = replace(
            horizon_cfg,
            run_name=f"{cfg.run_name}_h{horizon}_rolling_aggregate",
            test_window_id="rolling_aggregate",
        )
        aggregate_evaluation = evaluate_predictions(
            aggregate_predictions,
            aggregate_cfg,
            data_source=dataset.data_source,
            evaluation_split="rolling_test",
        )
        aggregate_complexity = (
            pd.concat(horizon_complexities, ignore_index=True, sort=False)
            .groupby(["model", "seed"], as_index=False, sort=False, dropna=False)
            .agg({
                "ensemble_members": "max",
                "trainable_parameters": "max",
                "model_coefficients": "max",
                "training_seconds": lambda values: values.sum(min_count=1),
                "inference_seconds": lambda values: values.sum(min_count=1),
                "combined_fit_and_forecast_seconds": (
                    lambda values: values.sum(min_count=1)
                ),
                "inference_milliseconds_per_1000_rows": "mean",
                "peak_cpu_rss_bytes": "max",
                "peak_gpu_memory_bytes": "max",
            })
        )
        aggregate_complexity["measurement_scope"] = (
            "rolling_windows_aggregate"
        )
        pooled = aggregate_evaluation["point_metrics"].query(
            "group_level == 'pooled'"
        ).set_index("model")
        aggregate_complexity["pooled_qlike"] = aggregate_complexity[
            "model"
        ].map(pooled["qlike"].to_dict())
        har_qlike = float(pooled.loc[cfg.primary_baseline, "qlike"])
        aggregate_complexity["qlike_improvement_over_log_har"] = (
            har_qlike - aggregate_complexity["pooled_qlike"]
        )
        aggregate_structured = build_structured_experiment_summary(
            aggregate_evaluation, aggregate_complexity, aggregate_cfg
        )
        aggregate_structured.insert(0, "run_name", aggregate_cfg.run_name)
        aggregate_structured.insert(1, "evaluation_split", "rolling_test")
        attach_run_identifiers(
            aggregate_evaluation, aggregate_cfg, "rolling_test"
        )
        aggregate_structured["rolling_train_end"] = "expanding_multiple"
        aggregate_structured["rolling_validation_end"] = "multiple"
        aggregate_structured["rolling_calibration_end"] = "multiple"
        aggregate_structured["rolling_test_start"] = min(
            window.test_start for window in windows
        )
        aggregate_structured["rolling_test_end"] = max(
            window.test_end for window in windows
        )
        summaries.append(aggregate_structured)
        if save_artifacts:
            aggregate_tables = {
                "point_metrics": aggregate_evaluation["point_metrics"],
                "probabilistic_metrics": aggregate_evaluation[
                    "probabilistic_summary"
                ],
                "regime_metrics": aggregate_evaluation["raw_regime_metrics"],
                "per_seed_metrics": aggregate_evaluation["per_seed_metrics"],
                "baseline_comparisons": aggregate_evaluation[
                    "baseline_comparison_summary"
                ],
                "complexity": aggregate_complexity,
            }
            if aggregate_evaluation["inference_summary"] is not None:
                aggregate_tables["inference"] = aggregate_evaluation[
                    "inference_summary"
                ]
            for table_name, table in aggregate_tables.items():
                table.to_csv(
                    summary_directory
                    / f"{cfg.run_name}_h{horizon}_aggregate_{table_name}.csv",
                    index=False,
                )

    combined = pd.concat(summaries, ignore_index=True, sort=False)
    if save_artifacts:
        combined.to_csv(
            summary_directory / f"{cfg.run_name}_all_horizons_windows.csv",
            index=False,
        )
    return combined


def run_separate_ticker_validation_comparison(
    cfg: Config,
    dataset: MarketDataset,
    *,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """Compare pooled and separate models on the seen-ticker validation set."""
    pooled_result = main(
        cfg,
        dataset=dataset,
        evaluation_split="validation",
        show_plots=False,
        display_tables=False,
        save_artifacts=False,
    )
    split = pooled_result["split"]
    pooled_metrics = pooled_result["evaluation"]["point_metrics"].query(
        "group_level == 'cohort_ticker' and model == 'SigFlow v4'"
    ).set_index("ticker")
    ticker_ids = dataset.metadata["ticker_id"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for ticker_id, ticker in enumerate(cfg.training_tickers):
        local_train = split.train_indices[
            ticker_ids[split.train_indices] == ticker_id
        ]
        local_validation = split.validation_indices[
            ticker_ids[split.validation_indices] == ticker_id
        ]
        local_calibration = split.calibration_indices[
            ticker_ids[split.calibration_indices] == ticker_id
        ]
        (
            local_weights,
            local_proportions,
            local_transitions,
            local_initial,
        ) = build_training_regime_statistics(
            dataset.metadata,
            split.regime_labels,
            local_train,
            cfg,
        )
        local_split = replace(
            split,
            train_indices=local_train,
            validation_indices=local_validation,
            calibration_indices=local_calibration,
            class_weights=local_weights,
            train_regime_proportions=local_proportions,
            transition_matrices=local_transitions,
            initial_regime_probabilities=local_initial,
        )
        models, _, _ = fit_ensemble(dataset, local_split, cfg)
        outputs = collect_ensemble_outputs(
            models, dataset, local_split, local_validation, cfg
        )
        model_samples = outputs["log_samples"]
        samples = model_samples_to_volatility(
            model_samples,
            local_split.har_predictions[local_validation],
            cfg,
        )
        actual = np.exp(dataset.targets_log_vol[local_validation])
        separate = point_metrics(
            actual,
            np.median(samples, axis=1),
            cfg,
            qlike_predicted=np.sqrt(np.mean(samples**2, axis=1)),
        )
        pooled = pooled_metrics.loc[ticker]
        rows.append({
            "ticker": ticker,
            "horizon": cfg.horizon,
            "seeds": ",".join(map(str, cfg.ensemble_seeds)),
            "observations": len(local_validation),
            "pooled_mae": pooled["mae"],
            "separate_mae": separate["mae"],
            "separate_minus_pooled_mae": separate["mae"] - pooled["mae"],
            "pooled_rmse": pooled["rmse"],
            "separate_rmse": separate["rmse"],
            "separate_minus_pooled_rmse": separate["rmse"] - pooled["rmse"],
            "pooled_qlike": pooled["qlike"],
            "separate_qlike": separate["qlike"],
            "separate_minus_pooled_qlike": separate["qlike"] - pooled["qlike"],
            "selection_data": "development_validation_only",
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    comparison = pd.DataFrame(rows)
    if save_artifacts:
        directory = Path(cfg.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(
            directory / "pooled_vs_separate_ticker_models.csv", index=False
        )
    return comparison


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled SigFlow v4 experiment runner."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "pipeline_test", "development", "rolling_research", "final_evaluation"
        ),
        default="pipeline_test",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SETTINGS),
        default=None,
    )
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--individual-feature-ablations", action="store_true")
    parser.add_argument("--ticker-model-comparison", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required acknowledgement for the one-time prospective test.",
    )
    return parser.parse_args()


def cli() -> None:
    global RUN_GROUP_ABLATIONS, RUN_INDIVIDUAL_FEATURE_ABLATIONS
    arguments = parse_cli_arguments()
    profile = arguments.profile or (
        "smoke" if arguments.mode == "pipeline_test" else "gpu_long"
    )
    cfg = config_for_profile(profile)
    if (
        arguments.mode == "pipeline_test"
        and cfg.experiment_mode != "pipeline_test"
    ):
        raise SystemExit(
            "pipeline_test requires the smoke profile; research profiles may "
            "not be relabelled as pipeline tests."
        )
    if (
        arguments.mode != "pipeline_test"
        and cfg.experiment_mode != "research"
    ):
        raise SystemExit(
            f"{arguments.mode} requires a real-data research profile; the "
            "synthetic smoke profile is pipeline-test only."
        )
    if arguments.mode == "final_evaluation":
        if profile != "gpu_long":
            raise SystemExit(
                "The preregistered final evaluation is frozen to gpu_long; "
                "alternative profiles remain development variants."
            )
        if arguments.no_save:
            raise SystemExit(
                "The prospective final evaluation must save its immutable "
                "configuration, audit, and result ledger."
            )
        if (
            arguments.run_ablations
            or arguments.individual_feature_ablations
            or arguments.ticker_model_comparison
        ):
            raise SystemExit(
                "Ablations and model selection are development-only and may "
                "not be combined with the prospective final evaluation."
            )
        if not arguments.confirm_final_test:
            raise SystemExit(
                "Final evaluation is write-once and prospective. Re-run with "
                "--confirm-final-test after freezing the configuration."
            )
        cfg = prospective_final_config(cfg, authorized=True)
        evaluation_split = "test"
    else:
        evaluation_split = "validation"
        if cfg.experiment_mode == "research":
            cfg = replace(
                cfg,
                end_date=min(
                    pd.Timestamp(cfg.end_date),
                    pd.Timestamp(cfg.decision_freeze_date),
                ).date().isoformat(),
            )
    RUN_GROUP_ABLATIONS = bool(arguments.run_ablations)
    RUN_INDIVIDUAL_FEATURE_ABLATIONS = bool(
        arguments.individual_feature_ablations
    )

    if arguments.mode == "rolling_research":
        run_multi_horizon_rolling_research(
            cfg,
            show_plots=arguments.show_plots,
            save_artifacts=not arguments.no_save,
        )
        return

    # The prospective path lets main acquire its exclusive ledger before it
    # constructs or inspects any holdout targets.
    dataset = None if evaluation_split == "test" else build_market_dataset(cfg)
    if arguments.ticker_model_comparison:
        run_separate_ticker_validation_comparison(
            cfg, dataset, save_artifacts=not arguments.no_save
        )
        return
    if RUN_GROUP_ABLATIONS or RUN_INDIVIDUAL_FEATURE_ABLATIONS:
        ablation_summary = run_configured_validation_ablations(cfg, dataset)
        cfg = apply_validation_selected_gate(cfg, ablation_summary)

    main(
        cfg,
        dataset=dataset,
        evaluation_split=evaluation_split,
        show_plots=arguments.show_plots,
        display_tables=False,
        save_artifacts=not arguments.no_save,
    )


SHARED_DATASET: MarketDataset | None = None
ABLATION_SUMMARY = pd.DataFrame()
RESULTS: dict[str, object] | None = None

if __name__ == "__main__":
    cli()
