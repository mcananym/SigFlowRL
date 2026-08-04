# SigFlow v4

The canonical implementation is [sigflow_v4_research.py](sigflow_v4_research.py). The thin [research runner notebook](SigFlow_v4_Research_Runner.ipynb) provides an interactive entry point; the older notebooks are retained as historical work.

## Execution modes

Install the scientific dependencies from `requirements.txt`, then use one of these explicit modes:

```bash
# Synthetic data, one seed, two epochs, no network calls
python sigflow_v4_research.py --mode pipeline_test --profile smoke

# Real cached/downloaded data, five seeds, validation only
python sigflow_v4_research.py --mode development --profile gpu_long

# 1/5/10/20-day forecasts over expanding rolling-origin windows
python sigflow_v4_research.py --mode rolling_research --profile gpu_long

# Complete matched-budget validation ablations on real data
python sigflow_v4_research.py --mode development --profile gpu_long --run-ablations

# Pooled versus separate seen-ticker models on validation data
python sigflow_v4_research.py --mode development --profile gpu_long --ticker-model-comparison
```

The one-time prospective test is intentionally gated and write-once:

```bash
python sigflow_v4_research.py \
  --mode final_evaluation \
  --profile gpu_long \
  --confirm-final-test
```

The decision protocol is frozen on **2026-08-04**. The prospective test origins
are exactly **2026-08-05 through 2028-08-31**; August 31 is the last forecast
origin, not the last target observation. Data through **2028-09-15** are reserved
so every multi-session target can mature, and the runner refuses to execute the
final evaluation before that date. Frozen fitting cutoffs are 2025-06-30
(training), 2025-12-31 (validation), and 2026-07-24 (calibration).

The final command must match preregistered SHA-256
`2adf58a4db7650bdd8323e32a64be954eba59dc3cffd3de02a6867d615ba03b1`. Before it
downloads data or computes a forecast, it creates an exclusive one-time ledger.
A completed, failed, or interrupted attempt remains locked for audit and cannot
be silently repeated. Result artifacts are written to a staging directory and
atomically promoted to the fixed output path only after the complete artifact
set has been written.

## What changed

- Typed HTTP failures, correct 429 semantics, `Retry-After`, bounded exponential backoff, provider pacing, atomic verified caches, and retrieval provenance.
- Strict `pipeline_test` versus `research` isolation; research never substitutes synthetic prices.
- Raw regimes by default; smoothing is an optional validation ablation.
- Per-section and per-ticker class counts, probabilities, confusion matrices, responsiveness and lag diagnostics.
- Unweighted, capped square-root-weighted, focal, and ordinal regime losses.
- Matched no-regime, auxiliary-only, soft-gate, single-distribution, simple-neural, signature, mixture, ticker, seed, HAR-input, and HAR-residual ablations.
- Five-seed distributional ensembles using `(42, 123, 456, 789, 2026)`.
- Chronological out-of-fold Log-HAR input forecasts, additive HAR residual modelling, and calibration-only HAR/SigFlow blending.
- Rolling RV, EWMA, level HAR, Log-HAR, leverage Log-HAR, dependency-free GARCH(1,1), and simple-neural controls.
- Ticker-specific past-only scaling, learned ticker embeddings, shared layers with ticker adapters, and a separate-ticker comparison runner.
- Separate calibration period, asymmetric tail corrections, ticker/regime fallbacks, and 50/80/90/95% coverage plus width and tail misses.
- Multi-horizon, purged rolling-origin evaluation; paired shared date-block/ticker-date bootstraps for MAE, RMSE, QLIKE, CRPS, and Holm-adjusted secondary comparisons.
- Reproducibility, provenance, leakage, complexity, plots, checkpoints, and tidy per-model/seed/ticker/horizon/window artifacts.
- Development ablations also save fold-level structured summaries; their architecture decisions never inspect calibration or test rows.

Run the standard-library regression suite with:

```bash
python -m unittest discover -s tests -v
```
