import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pandas as pd
import numpy as np

from institutional_backtest_engine import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    generate_synthetic_equity,
    backtest_from_real_data,
    compute_metrics,
    plot_institutional_dashboard
)

def test_psr_dsr_calculation():
    np.random.seed(0)
    returns = np.random.normal(0.0022, 0.0144, 1433)  # target from screenshot
    psr, sr, g3, g4, n = probabilistic_sharpe_ratio(returns)
    assert 0 <= psr <= 1
    assert sr > 1.0  # should be ~2.42 with this seed approx
    dsr, thr = deflated_sharpe_ratio(returns, n_trials=10)
    assert 0 <= dsr <= 1
    assert thr >= 0

def test_synthetic_generation():
    df, returns, dates = generate_synthetic_equity(seed=42)
    assert len(df) > 1000
    assert 'equity' in df.columns
    assert 'returns' in df.columns
    # Check metrics roughly
    total_ret = df['equity'].iloc[-1]/df['equity'].iloc[0]-1
    assert total_ret > 5.0  # >500% total

def test_backtest_offline():
    df = backtest_from_real_data(ticker="TQQQ", start="2020-01-01", end="2025-07-15", offline=True)
    assert 'Close' in df.columns
    assert 'equity' in df.columns
    assert 'returns' in df.columns
    assert len(df) > 500

def test_compute_metrics():
    df, _, _ = generate_synthetic_equity(seed=42)
    # Need returns column
    df['returns'] = df['equity'].pct_change().fillna(0)
    metrics, ret, eq, dd, monthly = compute_metrics(df)
    assert "Total Return" in metrics
    assert "Sharpe" in metrics
    assert "PSR" in metrics
    assert "DSR" in metrics
    assert "Max DD" in metrics
    # Check PSR close to 1 for good sharpe
    assert float(metrics["Sharpe_raw"]) > 1.0

def test_plot_dashboard(tmp_path=None):
    import tempfile, os
    df, _, _ = generate_synthetic_equity(seed=123)
    df['returns'] = df['equity'].pct_change().fillna(0)
    # Ensure DatetimeIndex for resample
    metrics, _, _, _, _ = compute_metrics(df)
    with tempfile.TemporaryDirectory() as tmp:
        png_path = os.path.join(tmp, "dashboard.png")
        plot_institutional_dashboard(df, metrics, png_path, style="dark")
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 10000
