#!/usr/bin/env python3
"""
Automated Daily Runner – combines Regime + TEMACD Screener + HTML dashboard
FREE execution via GitHub Actions cron or local cron
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

# Import our tools
from regime_engine_6 import fetch_macro, classify_regime, REGIMES
# We will import temacd scanner functions
sys.path.insert(0, str(Path(__file__).parent))
try:
    from temacd_screener import scan_tickers, QUANT_RICK_30_DEDUP, export_html as temacd_export_html
    HAS_TEMACD = True
except Exception as e:
    print(f"[WARN] temacd_screener import failed: {e}")
    HAS_TEMACD = False

def main():
    parser = argparse.ArgumentParser(description="Daily Automated Runner")
    parser.add_argument("--universe", default="quant_rick_30", choices=["quant_rick_30","us100","demo"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--export-html", type=str, default="reports/latest_daily_dashboard.html")
    parser.add_argument("--export-json", type=str, default="reports/latest_daily_manifest.json")
    parser.add_argument("--webhook-url", type=str, help="Discord webhook")
    args = parser.parse_args()

    Path("reports").mkdir(exist_ok=True)

    print("="*100)
    print("DAILY AUTOMATED PORTFOLIO CONTROL PLANE")
    print("="*100)

    macro = fetch_macro(offline=args.offline)
    regime_id = classify_regime(macro)
    regime = REGIMES[regime_id]
    print(f"Regime detected: {regime_id} - {regime['name']} | VIX {macro['vix']}")

    # TEMACD scan
    if HAS_TEMACD:
        if args.universe == "quant_rick_30":
            tickers = QUANT_RICK_30_DEDUP
        elif args.universe == "us100":
            from temacd_screener import US_UNIVERSE_100
            tickers = US_UNIVERSE_100
        else:
            tickers = ["TQQQ","QQQ","SPY","NVDA","AAPL"]

        results = scan_tickers(tickers, offline=args.offline)
        longs = [r for r in results if r['signal']=="LONG ENTRY"]
        shorts = [r for r in results if r['signal']=="SHORT/EXIT"]

        # Regime filter
        if regime_id in [5,6]:
            print("⚠️ Risk-Off regime – filtering out all LONGs")
            filtered_longs = []
        elif regime_id == 4:
            allowed = {"XLU","XLV","XLP","IEF","GLD","BIL","TLT"}
            filtered_longs = [r for r in longs if r['symbol'] in allowed]
        elif regime_id == 2:
            allowed = {"XLE","XLB","XLI","XLF","GLD","PDBC","UUP","BIL"}
            filtered_longs = [r for r in longs if r['symbol'] in allowed]
        else:
            filtered_longs = longs

        print(f"TEMACD Raw: LONG {len(longs)} SHORT {len(shorts)} | After regime filter LONG {len(filtered_longs)}")

        # Combined HTML dashboard
        if args.export_html:
            # Create simple combined dashboard
            dashboard_html = f"""
<html><head><meta charset="utf-8"><title>Daily Dashboard {datetime.now().date()}</title>
<style>
body{{background:#0b0e14;color:#e6edf3;font-family:sans-serif;padding:20px}}
.card{{background:#161b22;border:1px solid #30363d;padding:16px;border-radius:8px;margin:12px 0}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.badge-long{{background:#238636;color:#fff;padding:2px 8px;border-radius:12px}}
.badge-short{{background:#da3633;color:#fff;padding:2px 8px;border-radius:12px}}
table{{width:100%;border-collapse:collapse}} th{{background:#21262d;color:#58a6ff;padding:8px;text-align:left}} td{{padding:8px;border-bottom:1px solid #21262d}}
</style></head><body>
<h1>🚨 Daily Control Plane {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} – Regime #{regime_id} {regime['name']}</h1>
<div class="kpi">
<div class="card">Regime: {regime['name']}<br>{regime['desc']}<br>EQ {regime['eq']}% HD {regime['hedge']}% CA {regime['cash']}%</div>
<div class="card">Macro: VIX {macro['vix']:.2f} | XLU/SPY {macro['xlu_spy']:+.4f} | HYG/IEF {macro['hyg_ief']:+.4f}<br>SPY ${macro['spy_price']:.2f} vs EMA200 ${macro['spy_ema200']:.2f}</div>
<div class="card">TEMACD Raw LONGs: {len(longs)} | Filtered LONGs: {len(filtered_longs)} | SHORT/EXIT: {len(shorts)}</div>
</div>
<div class="card"><h2>🟢 FILTERED LONG ENTRY (after regime)</h2>
<table><tr><th>Symbol</th><th>Price</th><th>Chg%</th><th>Trend</th><th>Triggers</th></tr>
"""
            for r in filtered_longs:
                dashboard_html += f"<tr><td>{r['symbol']}</td><td>${r['close']:.2f}</td><td>{r['chg_pct']:+.2f}%</td><td>{r['trend']}</td><td>{', '.join(r['triggers'])}</td></tr>"
            dashboard_html += "</table></div>"
            dashboard_html += f"""<div class="card"><h2>🔴 SHORT / EXIT</h2><table><tr><th>Symbol</th><th>Price</th><th>Trend</th><th>Triggers</th></tr>"""
            for r in shorts:
                dashboard_html += f"<tr><td>{r['symbol']}</td><td>${r['close']:.2f}</td><td>{r['trend']}</td><td>{', '.join(r['triggers'])}</td></tr>"
            dashboard_html += "</table></div>"

            dashboard_html += f"<div class='card'><p>Source: {macro['source']} | Universe: {args.universe} | Tickers scanned: {len(tickers)}</p>"
            dashboard_html += "<p>Strategy: TQQQ TEMACD 5,50,222 + MACD 61,70,15 – 4-way OR crossover. LONG-ONLY recommended for FTMO/mega-caps. MES best for PDT bypass.</p></div>"

            dashboard_html += "</body></html>"
            Path(args.export_html).write_text(dashboard_html, encoding='utf-8')
            print(f"[OK] Dashboard -> {args.export_html}")

        if args.export_json:
            manifest = {
                "timestamp": datetime.utcnow().isoformat(),
                "regime_id": regime_id,
                "regime_name": regime["name"],
                "macro": macro,
                "allocation": {"eq": regime["eq"], "hedge": regime["hedge"], "cash": regime["cash"]},
                "temacd_raw": {"long": len(longs), "short": len(shorts)},
                "temacd_filtered_longs": filtered_longs,
                "shorts": shorts,
                "results_all": results
            }
            Path(args.export_json).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(f"[OK] Manifest -> {args.export_json}")

        if args.webhook_url:
            # send webhook
            try:
                from temacd_screener import send_webhook
                send_webhook(filtered_longs + shorts, args.webhook_url)
            except Exception as e:
                print(f"[WARN] webhook failed {e}")

    else:
        print("[ERROR] TEMACD screener not available")

if __name__ == "__main__":
    main()
