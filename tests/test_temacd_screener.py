import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pandas as pd
from temacd_screener import calc_temacd, detect_signals, generate_demo_data, ema

def test_ema_calculation():
    s = pd.Series([1,2,3,4,5,6,7,8,9,10])
    e = ema(s, 3)
    assert len(e) == len(s)
    assert not pd.isna(e.iloc[-1])

def test_temacd_columns():
    df = generate_demo_data("TQQQ", days=300)
    df = calc_temacd(df)
    for col in ["EMA1","EMA2","EMA3","MACD","aMACD","delta"]:
        assert col in df.columns
        assert not pd.isna(df[col].iloc[-1])

def test_signal_detection():
    df = generate_demo_data("TEST", days=400)
    df = calc_temacd(df)
    sig = detect_signals(df)
    assert sig["valid"] == True
    assert "signal" in sig
    assert sig["signal"] in ["LONG ENTRY","SHORT/EXIT","HOLD"]
    assert "trend" in sig

def test_not_enough_data():
    df = generate_demo_data("TEST", days=50)
    df = calc_temacd(df)
    sig = detect_signals(df)
    assert sig["valid"] == False

def test_demo_universe_scan():
    from temacd_screener import scan_tickers
    results = scan_tickers(["TQQQ","QQQ","SPY"], offline=True)
    assert len(results) == 3
    for r in results:
        assert "symbol" in r
        assert "close" in r
