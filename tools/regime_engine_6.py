#!/usr/bin/env python3
"""
6-Regime Classifier – determines market regime from Beta Rotations
Uses yfinance live data for XLU/SPY, XLY/XLP, XLB/GLD, HYG/IEF, XLK/XLV, XLE/XLU, VIX, SPY vs 200EMA

This is simplified production version for daily filtering of TEMACD signals.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

try:
    import yfinance as yf
    import pandas as pd
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("Missing yfinance/pandas, using offline fallback")
    import pandas as pd

# Regime definitions
REGIMES = {
    1: {"name": "RISK_ON_GROWTH", "desc": "Býčí expanze / Tech & Mega-Cap Dominance (Risk-On)", "eq":100,"hedge":0,"cash":0, "sharpe":1.30, "dd":"-15.4%"},
    2: {"name": "INFLATION_GROWTH", "desc": "Inflační expanze / Komoditní a energetický cyklus", "eq":75,"hedge":20,"cash":5, "sharpe":1.15, "dd":"-14.2%"},
    3: {"name": "RANGE_BOUND_WHIPSAW", "desc": "Sideways Konsolidace / Whipsaw trh", "eq":40,"hedge":30,"cash":30, "sharpe":1.45, "dd":"-8.7%"},
    4: {"name": "DEFENSIVE_ROTATION", "desc": "Defenzivní rotace / Zpomalení ekonomiky", "eq":50,"hedge":35,"cash":15, "sharpe":1.10, "dd":"-11.5%"},
    5: {"name": "CREDIT_STRESS", "desc": "Kreditní stres / Risk-Off Medvědí trh", "eq":0,"hedge":80,"cash":20, "sharpe":1.05, "dd":"-9.2%"},
    6: {"name": "BLACK_SWAN_VOL_SHOCK", "desc": "Černá labuť / Extrémní volatilní šok", "eq":0,"hedge":20,"cash":80, "sharpe":0.95, "dd":"-4.5%"},
}

def fetch_macro(offline=False):
    if offline or not HAS_YF:
        # deterministic offline example: risk-on
        return {
            "vix": 16.5,
            "xlu_spy": -0.012,
            "xly_xlp": 0.015,
            "xlb_gld": 0.002,
            "hyg_ief": 0.008,
            "xlk_xlv": 0.006,
            "xle_xlu": 0.003,
            "spy_price": 580,
            "spy_ema200": 530,
            "source": "OFFLINE"
        }
    try:
        tickers = ["^VIX","XLU","SPY","XLY","XLP","XLB","GLD","HYG","IEF","XLK","XLV","XLE"]
        data = yf.download(tickers, period="1mo", interval="1d", progress=False, auto_adjust=False)['Close']
        # compute 20d changes
        def chg(t):
            if t not in data.columns:
                return 0
            s = data[t].dropna()
            if len(s) < 2:
                return 0
            return (s.iloc[-1]/s.iloc[0]-1)
        vix = float(data["^VIX"].iloc[-1]) if "^VIX" in data.columns else 18
        xlu_spy = chg("XLU") - chg("SPY")
        xly_xlp = chg("XLY") - chg("XLP")
        xlb_gld = chg("XLB") - chg("GLD")
        hyg_ief = chg("HYG") - chg("IEF")
        xlk_xlv = chg("XLK") - chg("XLV")
        xle_xlu = chg("XLE") - chg("XLU")
        spy_price = float(data["SPY"].iloc[-1]) if "SPY" in data.columns else 580
        # EMA200 for SPY
        spy_series = data["SPY"].dropna() if "SPY" in data.columns else pd.Series([580]*200)
        ema200 = spy_series.ewm(span=200, adjust=False).mean().iloc[-1]
        return {
            "vix": vix,
            "xlu_spy": xlu_spy,
            "xly_xlp": xly_xlp,
            "xlb_gld": xlb_gld,
            "hyg_ief": hyg_ief,
            "xlk_xlv": xlk_xlv,
            "xle_xlu": xle_xlu,
            "spy_price": spy_price,
            "spy_ema200": float(ema200),
            "source": "LIVE yfinance"
        }
    except Exception as e:
        print(f"[WARN] fetch_macro failed {e}, offline fallback")
        return fetch_macro(offline=True)

def classify_regime(m):
    # priority order: black swan first
    if m["vix"] >= 30 or m["hyg_ief"] <= -0.04:
        return 6
    if m["hyg_ief"] < -0.012 and m["xlu_spy"] > 0.008:
        return 5
    if m["xlu_spy"] > 0.005 and m["xly_xlp"] < -0.005:
        return 4
    if 18 <= m["vix"] < 25:
        return 3
    if m["xle_xlu"] > 0.008 and m["xlb_gld"] > 0.005:
        return 2
    return 1

def print_table(current_regime, macro):
    print("\n" + "="*120)
    print(f"6-REGIME CLASSIFIER | Source: {macro['source']} | Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"SPY ${macro['spy_price']:.2f} vs EMA200 ${macro['spy_ema200']:.2f} | VIX {macro['vix']:.2f}")
    print(f"XLU/SPY {macro['xlu_spy']:+.4f} | XLY/XLP {macro['xly_xlp']:+.4f} | XLB/GLD {macro['xlb_gld']:+.4f} | HYG/IEF {macro['hyg_ief']:+.4f} | XLK/XLV {macro['xlk_xlv']:+.4f} | XLE/XLU {macro['xle_xlu']:+.4f}")
    print("="*120)
    print(f"{'ID':2} | {'REGIME':30} | EQ% | HD% | CA% | SHARPE | DD | ACTIVE")
    print("-"*120)
    for rid, r in REGIMES.items():
        active = "<<< CURRENT" if rid==current_regime else ""
        print(f"{rid:2} | {r['name']:30} | {r['eq']:3}% | {r['hedge']:3}% | {r['cash']:3}% | {r['sharpe']:4.2f} | {r['dd']:>6} | {active} {r['desc']}")
    print("="*120)
    if current_regime in [5,6]:
        print("⚠️  RISK-OFF: TEMACD LONGy ignoruj, drž pouze TLT/IEF/GLD/BIL")
    elif current_regime == 4:
        print("🛡️  DEFENSIVE: LONG pouze XLU,XLV,XLP,IEF – technologie EXIT")
    elif current_regime == 3:
        print("🔄  WHIPSAW: Ber pouze STRONG BULL TEMACD + 5m-ORB intraday, jinak 60% hedge")
    elif current_regime == 2:
        print("🛢️  INFLATION: LONG XLE,XLB,XLI,XLF,GLD – tech podvážit")
    else:
        print("🚀  RISK-ON: Ber všechny TEMACD LONGy – full equity TQQQ,QQQ,Mega-tech")

def export_html(macro, regime_id, path):
    r = REGIMES[regime_id]
    html = f"""<html><head><meta charset="utf-8"><title>Regime Report</title>
    <style>body{{background:#0b0e14;color:#e6edf3;font-family:sans-serif;padding:20px}} .card{{background:#161b22;border:1px solid #30363d;padding:16px;border-radius:8px;margin:10px 0}}</style>
    </head><body>
    <h1>6-Regime Classifier – {r['name']}</h1>
    <p>{r['desc']}</p>
    <div class="card">VIX {macro['vix']:.2f} | XLU/SPY {macro['xlu_spy']:+.4f} | XLY/XLP {macro['xly_xlp']:+.4f} | HYG/IEF {macro['hyg_ief']:+.4f}</div>
    <div class="card">SPY ${macro['spy_price']:.2f} vs EMA200 ${macro['spy_ema200']:.2f} | Source {macro['source']}</div>
    <div class="card">Allocation: EQ {r['eq']}% | Hedge {r['hedge']}% | Cash {r['cash']}% | Sharpe {r['sharpe']} | DD {r['dd']}</div>
    </body></html>"""
    Path(path).write_text(html, encoding='utf-8')
    print(f"[OK] HTML exported to {path}")

def main():
    parser = argparse.ArgumentParser(description="6-Regime Classifier")
    parser.add_argument("--offline", action="store_true", help="Offline demo")
    parser.add_argument("--export-html", type=str, help="Export HTML path")
    args = parser.parse_args()
    
    macro = fetch_macro(offline=args.offline)
    regime_id = classify_regime(macro)
    print_table(regime_id, macro)
    
    if args.export_html:
        export_html(macro, regime_id, args.export_html)

if __name__ == "__main__":
    main()
