import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from regime_engine_6 import fetch_macro, classify_regime, REGIMES

def test_fetch_offline():
    m = fetch_macro(offline=True)
    assert "vix" in m
    assert "xlu_spy" in m
    assert "spy_price" in m

def test_classify_black_swan():
    m = {"vix":35, "xlu_spy":0.01, "xly_xlp":0, "xlb_gld":0, "hyg_ief":0, "xlk_xlv":0, "xle_xlu":0, "spy_price":500, "spy_ema200":530}
    assert classify_regime(m) == 6

def test_classify_credit_stress():
    m = {"vix":20, "xlu_spy":0.009, "xly_xlp":0, "xlb_gld":0, "hyg_ief":-0.02, "xlk_xlv":0, "xle_xlu":0, "spy_price":500, "spy_ema200":530}
    assert classify_regime(m) == 5

def test_classify_risk_on():
    m = {"vix":14, "xlu_spy":-0.01, "xly_xlp":0.01, "xlb_gld":0, "hyg_ief":0.005, "xlk_xlv":0, "xle_xlu":0, "spy_price":580, "spy_ema200":530}
    assert classify_regime(m) == 1

def test_regimes_defined():
    assert len(REGIMES) == 6
    for i in range(1,7):
        assert i in REGIMES
        assert "name" in REGIMES[i]
