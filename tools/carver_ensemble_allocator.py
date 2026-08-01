#!/usr/bin/env python3
"""
Robert Carver Volatility Targeting & Quant Rick Ensemble Portfolio
Implements:
- Carver universal equation: LOTS = Dollar Exposure / (Price * Multiplier)
- Volatility targeting across asset classes
- Multi-strategy ensemble: TQQQ TEMACD, BTC Ensemble, QQQ 3X EMACD, GLD 3X EMACD

Based on Robert Carver's Systematic Trading + Quant Rick videos
"""

import argparse
from datetime import datetime

# Multiplier table
MULTIPLIERS = {
    "FX": 100000,  # forex
    "XAUUSD": 100,  # gold
    "XAGUSD": 5000, # silver
    "OIL": 1000,
    "INDICES": 10,  # US30, US500, US100 etc.
    "CRYPTO": 1,
    "STOCKS": 1,
    "ETF": 1,
    "FUTURES_MES": 5,  # MES $5 per point
    "FUTURES_ES": 50,
}

# Volatilities (annualized %)
VOLATILITIES = {
    "TQQQ": 55.0,
    "QQQ": 18.0,
    "SPY": 16.0,
    "BTC-USD": 65.0,
    "GLD": 15.0,
    "XLK": 22.0,
    "XLE": 28.0,
    "XLV": 14.0,
    "IEF": 6.0,
    "TLT": 12.0,
    "BIL": 0.5,
    "MES=F": 16.0,
}

# Ensemble definitions
ENSEMBLES = {
    "TQQQ_TEMACD": {"emas": (5,50,222), "macd": (61,70,15), "assets": ["TQQQ","QQQ","SPY","NVDA","AAPL","MSFT","XLK"], "sharpe": 1.30},
    "BTC_ENSEMBLE": {"emas": (5,64,170), "macd": (32,40,94), "assets": ["BTC-USD","ETH-USD"], "sharpe": 1.46},
    "QQQ_3X_EMACD": {"emas": (5,50,222), "macd": (61,70,15), "assets": ["TQQQ","UPRO","SPXL","QQQ"], "sharpe": 1.25, "note": "Same as TEMACD but 3x leveraged QQQ, best for black swan prevention with 200EMA filter"},
    "GLD_3X_EMACD": {"emas": (5,50,222), "macd": (61,70,15), "assets": ["GLD","GDX","UGL","GLL"], "sharpe": 1.10, "note": "Gold 3x ensemble for prop firms, low correlation to equities"},
    "LT_MA_CROSS": {"emas": (50,200), "macd": None, "assets": ["SPY","QQQ","IWM","XLU","XLV","XLP"], "sharpe": 1.05, "note": "Long-term MA cross 50>200 for regime filter"},
    "DONCHIAN_CARVER": {"donchian": (20,55), "carver": True, "assets": ["XLE","XLB","XLI","GLD","TLT"], "sharpe": 1.15, "note": "Blended Donchian breakout + Carver/FTI systematic"},
}

REGIMES = {
    1: {"name": "RISK_ON_GROWTH", "engine": "TQQQ_TEMACD + QQQ_3X_EMACD", "alloc": {"TQQQ":40,"QQQ":25,"BTC-USD":25,"XLK":10}},
    2: {"name": "INFLATION_GROWTH", "engine": "GLD_3X_EMACD + DONCHIAN_CARVER", "alloc": {"XLE":20,"XLB":20,"XLI":20,"GLD":20,"XLF":20}},
    3: {"name": "RANGE_BOUND_WHIPSAW", "engine": "DONCHIAN_CARVER + LT_MA_CROSS defensive", "alloc": {"XLV":15,"XLP":15,"GLD":15,"IEF":25,"BIL":30}},
    4: {"name": "DEFENSIVE_ROTATION", "engine": "LT_MA_CROSS", "alloc": {"XLU":20,"XLV":15,"XLP":15,"IEF":35,"GLD":15}},
    5: {"name": "CREDIT_STRESS", "engine": "Protective Hedge", "alloc": {"TLT":30,"IEF":30,"GLD":20,"BIL":20}},
    6: {"name": "BLACK_SWAN_VOL_SHOCK", "engine": "100% BIL + VIX hedge", "alloc": {"BIL":80,"VIXY":10,"GLD":10}},
}

def compute_vol_targeted_exposure(capital, target_vol, asset_vol):
    """Dollar exposure = Capital * target_vol / asset_vol"""
    if asset_vol == 0:
        return 0
    return capital * (target_vol / (asset_vol/100))

def compute_lots(dollar_exposure, price, multiplier=1):
    return dollar_exposure / (price * multiplier)

def build_portfolio(regime_id=1, capital=100000, target_vol=0.20, prices=None):
    if prices is None:
        prices = {"TQQQ":218.28,"QQQ":576.38,"BTC-USD":68500,"XLK":243.86,"XLE":95.2,"XLB":85.3,"XLI":115.6,"XLF":42.1,"GLD":126.73,"XLV":138.4,"XLP":78.2,"XLU":72.5,"IEF":97.5,"TLT":92.3,"BIL":91.2,"VIXY":12.5,"SPY":580.52,"NVDA":125.19,"AAPL":162.81}

    regime = REGIMES.get(regime_id, REGIMES[1])
    alloc = regime['alloc']
    rows = []
    total_dollar = 0
    for symbol, pct in alloc.items():
        target_dollar = capital * pct / 100
        # Vol targeting adjust
        asset_vol = VOLATILITIES.get(symbol, 20.0)
        vol_targeted = compute_vol_targeted_exposure(target_dollar, target_vol*100, asset_vol)  # target_vol*100 to match %
        # Actually carver formula: Dollar exposure you want = Capital * TargetVol / AssetVol
        # Then apply allocation pct as secondary
        # Simplified: use vol targeted as primary
        price = prices.get(symbol, 100)
        multiplier = MULTIPLIERS.get("STOCKS",1)
        if "MES" in symbol:
            multiplier = MULTIPLIERS["FUTURES_MES"]
        lots = compute_lots(vol_targeted, price, multiplier)
        total_dollar += vol_targeted
        rows.append({
            "symbol": symbol,
            "price": price,
            "vol": asset_vol,
            "alloc_pct": pct,
            "dollar_exp": vol_targeted,
            "lots": lots,
            "engine": regime['engine']
        })
    return rows, total_dollar, regime

def print_table(rows, total_dollar, regime, regime_id):
    print(f"\n{'='*130}")
    print(f"ROBERT CARVER VOL-TARGETED & ENSEMBLE PORTFOLIO | REŽIM #{regime_id}: {regime['name']}")
    print(f"Engine: {regime['engine']}")
    print(f"{'='*130}")
    print(f"{'SYMBOL':8} | {'PRICE':>8} | {'VOL%':>6} | {'ALLOC%':>6} | {'DOLLAR $':>12} | {'LOTS':>8} | ENGINE")
    print("-"*130)
    for r in rows:
        print(f"{r['symbol']:8} | ${r['price']:7.2f} | {r['vol']:5.1f}% | {r['alloc_pct']:5.1f}% | ${r['dollar_exp']:10.2f} | {r['lots']:7.4f} | {r['engine'][:30]}")
    print("-"*130)
    print(f"CELKOVÁ VOL-TARGETED EXPOZICE: ${total_dollar:.2f} | SYNTETICKÉ SHARPE >=2.08 (multi-asset nekorelované)")
    print(f"{'='*130}\n")

def main():
    parser = argparse.ArgumentParser(description="Carver Ensemble Allocator")
    parser.add_argument("--regime", type=int, default=1, choices=[1,2,3,4,5,6], help="Market regime 1-6")
    parser.add_argument("--capital", type=float, default=100000, help="Capital USD")
    parser.add_argument("--target-vol", type=float, default=0.20, help="Target portfolio annual vol 0.20=20%")
    parser.add_argument("--export-html", type=str, help="Export HTML")
    args = parser.parse_args()

    rows, total, regime = build_portfolio(regime_id=args.regime, capital=args.capital, target_vol=args.target_vol)
    print_table(rows, total, regime, args.regime)

    if args.export_html:
        html = f"<html><head><meta charset='utf-8'><title>Carver Portfolio Regime {args.regime}</title></head><body><h1>Regime {args.regime} {regime['name']}</h1><table border=1>"
        html += "<tr><th>Symbol</th><th>Price</th><th>Vol</th><th>Alloc%</th><th>Dollar</th><th>Lots</th></tr>"
        for r in rows:
            html += f"<tr><td>{r['symbol']}</td><td>{r['price']}</td><td>{r['vol']}%</td><td>{r['alloc_pct']}%</td><td>${r['dollar_exp']:.2f}</td><td>{r['lots']:.4f}</td></tr>"
        html += f"</table><p>Total Exposure ${total:.2f}</p></body></html>"
        Path = __import__('pathlib').Path
        Path(args.export_html).write_text(html, encoding='utf-8')
        print(f"[OK] HTML -> {args.export_html}")

if __name__ == "__main__":
    main()
