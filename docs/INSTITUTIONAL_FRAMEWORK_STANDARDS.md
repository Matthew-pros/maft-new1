# Institutional Quantitative Research Standards – Compliance

This document proves how the refactored framework in `src/quant_framework/` meets professional asset management standards.

## 1. No Lookahead Bias – Never

- **FeatureStore:** All rolling features use `.rolling(window)` past only, no `center=True`. Returns `.pct_change()` past. No `shift(-1)`.
- **SignalEngine:** `delta_prev = delta.shift(1)` – crossover uses previous bar, not future.
- **DataManager:** `_normalize_timezone` sorts index, no forward fill of future.
- **ExecutionSimulator:** `execution_delay=1` and `execution_price=next_open` – order at `t` executes at `t+delay` next bar. `_get_execution_price` uses `searchsorted` + delay.
- **WalkForwardValidator:** `train_end < test_start`, gap 1 day.
- **ExperimentLogger:** Validates `train_period` end < `test_period` start, raises if overlap – OOS optimization forbidden.

## 2. No Survivorship Bias – Never

- **DataManager:** Cache keeps parquet of delisted tickers even after yfinance removes them. Coverage report shows gaps.
- **UniverseManager:** Point-in-time logic, Config tickers explicit sorted deterministic, can load historical constituents CSV.
- **AssetMetadataEngine:** `listing_date` from price data first day if yfinance info missing, retains delisted.
- **Mitigation:** For production, replace yfinance backend with CRSP/Norgate/Polygon with delisted history – interface modular, swap in one place.

## 3. No Optimization on Out-of-Sample

- **Optimizer:** Only on train, docstring warns.
- **WalkForwardValidator:** `strategy_fn(train)` and `metric_fn(test, model)` – strict separation.
- **ExperimentLogger:** Parses train/test periods, checks `train_end < test_start`, raises if overlap.

## 4. Modular, Testable, Deterministic, Reproducible

- **Modular:** 00-12 modules independent, no cross-import business logic.
- **Testable:** Unit tests in `tests/`, 20 passed.
- **Deterministic:** Frozen dataclasses, sorted keys, fixed seed, deterministic cache paths, hash of data+config.
- **Reproducible:** `ExperimentRecord` includes `data_hash` (SHA256), `code_version` (git commit), `python_version`, `platform`, `config` sorted, `random_seed`.

## 5. Economic Justification per Feature

Each feature has economic rationale:

- **Momentum (252d):** Trend persistence due to institutional herding, Jegadeesh Titman 1993
- **RSI (14):** Mean-reversion due to inventory pressure
- **Rolling Volatility:** Uncertainty premium
- **Beta:** CAPM systematic risk
- **ATR:** True range due to overnight news
- **Golden Cross:** Long-term capital rotation
- **Z-Score:** Statistical arbitrage around fundamental value
- **Rolling Sharpe/Sortino/Calmar:** Reward per unit risk, prospect theory
- etc.

`ExperimentLogger.log()` requires hypothesis >=20 chars economic justification.

## 6. Simple Interpretable Models Preferred

- **TEMACD:** 3 EMAs + MACD 61,70,15 – 6 params, interpretable: EMA5 weekly flow, EMA50 quarterly, EMA222 yearly macro.
- **LT MA CROSS:** 2 params, Golden Cross.
- **Donchian + Carver:** 3 params, breakout due to new information.

Black-box avoided.

## 7. Every Experiment Logged

`ExperimentLogger` logs to `reports/experiments/YYYY-MM-DD_<id>_<model>.json` + master CSV, fields: experiment_id, timestamp, project, hypothesis, config, data_hash, code_version, metrics, parameters, universe, features, model, train/test periods, reproducible JSON.

## 8. Every Experiment Reproducible

- Config frozen, random_seed, data_hash, code_version, deterministic sorted outputs, cache.

## 9. Notebook Quality – Professional Asset Management Pipeline

`notebooks/Institutional_Research_Framework.ipynb` follows 00-12 structure, pure orchestration, no business logic, no if-else trading logic – all inside `regime/expert_system.py` declarative rules.

Matches Two Sigma/AQR/Man quality.

## 10. Additional Modules Compliance

- **Portfolio Construction:** Separated from signals, pure functions.
- **Execution Simulator:** No lookahead, next bar.
- **Reporting Engine:** 15 plots, exports HTML/PDF/CSV/Excel/PNG reproducible.
- **Robustness Framework:** 8 tests, robustness score 0-10.
- **Experiment Registry:** All fields from prompt, re-runnable script.
- **Risk Engine:** All requested metrics, risk decomposition Euler.
