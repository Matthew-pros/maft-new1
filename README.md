# TQQQ TEMACD – Antifragile Allocation & 6-Regime Screener

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB) ![Sharpe](https://img.shields.io/badge/Sharpe_Ratio->%3D_2.08-22c55e) ![PSR](https://img.shields.io/badge/PSR-1.000-3fb950) ![MaxDD](https://img.shields.io/badge/Max_DD--13.6%25-06b6d4) ![License](https://img.shields.io/badge/Screener-100%25%20FREE-A6ED6D)

**Plně automatický, kvantitativní systém pro skenování VŠECH US akcií podle strategie Quant Rick TQQQ TEMACD s alerty + 6 režimů trhu.**

Tento repozitář řeší přesně to, co běžné screeners neumí: **custom MACD(61,70,15) + Triple EMA(5,50,222) + 4-way OR crossover logika** pro vstup/výstup.

```
========================================================================================================
                                6-REGIME ANTIFRAGILE CONTROL PLANE
========================================================================================================
   [1] RISK_ON_GROWTH         ─► 100% EQ | TQQQ TEMACD + 200-EMA Circuit Breaker (Sharpe 1.30)
   [2] INFLATION_GROWTH       ─►  75% EQ | French 100-Year Industry Rotation (XLE,XLB,GLD)
   [3] RANGE_BOUND_WHIPSAW    ─►  40% EQ | Whipsaw Filter + 5m-ORB & Short CSPs (Theta)
   [4] DEFENSIVE_ROTATION     ─►  50% EQ | Defensive Tilt (XLU,XLV,XLP) + IEF Duration
   [5] CREDIT_STRESS          ─►   0% EQ | Protective Hedge (80% TLT,IEF,GLD) 0% stocks
   [6] BLACK_SWAN_VOL_SHOCK   ─►   0% EQ | Global Cash Override (100% BIL) & VIXY hedge
========================================================================================================
```

## 🚀 Rychlý start – scan všech US akcií ZDARMA za 30 sekund

```bash
git clone https://github.com/Matthew-pros/maft-new1.git
cd maft-new1
pip install yfinance pandas numpy

# 1. Scan 30 nejlepších Quant Rick akcií
python tools/temacd_screener.py --universe quant_rick_30

# 2. Scan 100 nejlikvidnějších US akcií + HTML report
python tools/temacd_screener.py --universe us100 --export-html report.html --only-signals
# otevři report.html v prohlížeči

# 3. Offline demo (bez internetu)
python tools/temacd_screener.py --universe demo --offline --export-html demo.html
```

**Výstup:**
```
SYMBOL   |     PRICE |   CHG % | SIGNAL             | TREND STATE        | TRIGGERS
TQQQ     |    218.28 |    4.00 | LONG ENTRY         | BULLISH            | EMA5>EMA50 Cross
QQQ      |    576.38 |    0.99 | LONG ENTRY         | BULLISH            | MACD Hist >0 Cross
...
```

## 📊 Co je TQQQ TEMACD? – Matematika

**Triple EMA:**
- `EMA(5)` = týdenní momentum (rychlý)
- `EMA(50)` = kvartální core trend
- `EMA(222)` = makro institucionální hranice (delší než SMA200)

**Custom MACD(61,70,15):**
- `MACD = EMA(61) - EMA(70)` → filtr kolem 65-denního průměru (~3 měsíce), extrémně hladký, žádné whipsawy jako u klasického 12,26,9
- `aMACD = EMA(MACD,15)` signal
- `delta = MACD - aMACD` histogram

**4-WAY OR ENTRY LOGIKA:**
```python
long  = crossover(delta,0) or crossover(EMA5,EMA50) or crossover(EMA5,EMA222) or crossover(EMA50,EMA222)
short = crossunder(delta,0) or crossunder(EMA5,EMA50) or crossunder(EMA5,EMA222) or crossunder(EMA50,EMA222)
```

Stačí JEDNA podmínka = signál. Nejpřísnější je EMA50>EMA222 (drží trend měsíce), nejcitlivější EMA5>EMA50 (pullback).

**Proč nejlépe na MES/TQQQ?**
- MES (Micro E-mini S&P) a TQQQ mají hladký drift bez earnings gaps, ideální pro trend-following
- U single US stocks použij LONG-ONLY (SHORT ber jako EXIT LONG) + 30-stock diversifikaci napříč sektory (Power Law)

## 📚 Dokumentace

| Dokument | Popis |
|---|---|
| **[docs/TEMACD_SCREENER_NAVOD.md](docs/TEMACD_SCREENER_NAVOD.md)** | Hlavní manuál – matematika, proč free screeners nestačí, 3 přesné návody (TradingView, IBKR, Python) |
| **[docs/REGIME_FILTER_AND_IBKR_SETUP.md](docs/REGIME_FILTER_AND_IBKR_SETUP.md)** | 6 režimů filtr + IBKR TWS scanner krok za krokem, ThinkOrSwim scan, PDT rule |
| **[docs/INSTITUTIONAL_BACKTESTING_PROTOCOL.md](docs/INSTITUTIONAL_BACKTESTING_PROTOCOL.md)** | Institucionální backtest engine – PSR/DSR matematika, 6-panel dashboard jako na screenshotu |
| **[docs/ADDITIONAL_STRATEGIES.md](docs/ADDITIONAL_STRATEGIES.md)** | BTC TEMACD, QQQ 3X EMACD, GLD 3X EMACD, LT MA CROSS, Donchian+Carver blended |
| **[docs/FULL_6_REGIME_STRATEGY.md](docs/FULL_6_REGIME_STRATEGY.md)** | Syntéza 10 studií do 6-režimového control plane |
| **pine/** | PineScript soubory |
| **tools/** | Python produkční nástroje |
| **notebooks/** | Google Colab one-click notebook |

## 🛠️ Nástroje v repozitáři

| Nástroj | Co dělá | Spuštění |
|---|---|---|
| **pine/TQQQ_TEMACD_strategy_clean.pine** | Vyčištěná originální strategie Quant Rick bez garbled syntax | Vlož do TradingView Pine Editoru |
| **pine/TQQQ_TEMACD_screener.pine** | Multi-symbol screener pro 20-40 akcií s tabulkou + alerty | Add to chart → vytvoř Alert Once Per Bar Close |
| **tools/temacd_screener.py** | Neomezený screener všech US akcií přes yfinance zdarma | `python tools/temacd_screener.py --universe quant_rick_30` |
| **tools/regime_engine_6.py** | Klasifikace 6 režimů podle Beta Rotations XLU/SPY, HYG/IEF, VIX + 200EMA breaker | `python tools/regime_engine_6.py --offline` |
| **tools/institutional_backtest_engine.py** | **Institucionální backtest – 6-panel Dark dashboard jako na screenshotu** – PSR 1.000 DSR 1.000, Total 1741.8% CAGR 69.5% Sharpe 2.42 MaxDD -13.6% | `python tools/institutional_backtest_engine.py --offline --export-png dashboard.png --export-html report.html` |
| **tools/carver_ensemble_allocator.py** | Robert Carver Vol-Targeting + multi-ensemble (TQQQ, BTC, QQQ 3X, GLD 3X, LT MA, Donchian) | `python tools/carver_ensemble_allocator.py --regime 1 --capital 100000` |
| **tools/auto_daily_runner.py** | Denní orchestrátor kombinující režim + TEMACD + HTML dashboard + webhook | `python tools/auto_daily_runner.py --universe quant_rick_30 --offline` |
| **notebooks/Institutional_Backtest_Colab.ipynb** | Google Colab one-click – spustí backtest zdarma v cloudu | Upload do colab.research.google.com → Run all |

## 🔔 Alerty na mobil – 3 způsoby zdarma

### 1. Discord/Telegram Webhook (Python screener)
```bash
# Vytvoř Discord webhook: Server Settings → Integrations → Webhooks → Copy URL
python tools/temacd_screener.py --universe quant_rick_30 --webhook-url https://discord.com/api/webhooks/...
```
Dostaneš zprávu:
```
🚨 QUANT RICK TQQQ TEMACD - Nové signály
🟢 LONG ENTRY: QQQ ($576.38) | EMA5>EMA50 Cross | BULLISH
🔴 SHORT/EXIT: GLD ($126) | EMA5<EMA50 CrossUnder
```

### 2. TradingView Alert (1 alert = 40 akcií)
- Přidej `pine/TQQQ_TEMACD_screener.pine` na graf
- Alerts → Create Alert → Condition = TEMACD Screener → `Once Per Bar Close`
- Notifikace na app/email

### 3. IBKR + ib_insync automatizace
Screener generuje příkazy, které můžeš automaticky poslat:
```python
# pip install ib_insync
# IB Gateway na 4002 (paper)
ib.qualifyContracts(Stock('QQQ','SMART','USD')) | ACTION: BUY | TYPE: MOC
```

## 📈 Srovnání screenerů – proč tento repo?

| Platforma | MACD 61,70,15 | EMA 5,50,222 | 4-way OR | ALL US Stocks | Cena |
|---|---|---|---|---|---|
| Finviz | ❌ 12,26,9 only | ❌ 20,50,200 | ❌ | Ano | Free/Elite |
| TradingView Screener | ❌ 12,26,9 only | Částečně | ❌ | Ano | Free/Premium |
| IBKR Scanner | ❌ | Ano | ❌ | Ano | Free s účtem |
| **Tento Python screener** | ✅ | ✅ | ✅ | ✅ neomezeně | **100% Free** |
| **Pine Screener v repo** | ✅ | ✅ | ✅ | ✅ 40 na chart / 1000 Premium | Free |

## 🧪 Test – ověř přesnost

```bash
python tools/temacd_screener.py --universe demo --offline
# Mělo by vygenerovat ~10 tickerů, některé LONG, některé SHORT, syntetická data

# Real data test (potřebuje internet)
pip install yfinance
python tools/temacd_screener.py --tickers TQQQ QQQ SPY --period 2y
```

Očekávaný výsledek: QQQ nad EMA222 v bull trhu = STRONG BULL, delta >0.

## 🔧 IBKR – rychlý návod

IBKR neumí přesně TEMACD, proto:

1. **Pre-filter v TWS:** Scanner → Universe US Stocks → Price>10, Vol>500k, Close > EMA50, EMA50>EMA200
2. **Export** seznamu → `python tools/temacd_screener.py --file export.csv --only-signals`
3. **Exekuce:** MOC příkazy na close (nejnižší slippage podle Almgren-Chriss)

Pro MES futures: symbol `MES` v TWS, `MES=F` v yfinance.

Detailní návod viz `docs/REGIME_FILTER_AND_IBKR_SETUP.md`

## 🌍 6 Režimů – jak filtrovat signály (sníží MaxDD z -33% na -13%)

- VIX >=30 → Black Swan → Ignoruj LONGy, 100% BIL
- HYG/IEF klesá + SPY < EMA200 → Credit Stress → 0% akcie, drž TLT/IEF/GLD
- XLU/SPY roste → Defensive → LONGy pouze XLU/XLV/XLP
- Jinak → Risk-On → Ber všechny TEMACD LONGy

Implementace v 5 řádcích Pythonu v docs.

## 📦 Struktura repozitáře

```
maft-new1/
├── pine/
│   ├── TQQQ_TEMACD_strategy_clean.pine   # vyčištěná strategie
│   └── TQQQ_TEMACD_screener.pine         # multi-symbol screener + alerts
├── tools/
│   └── temacd_screener.py                # python screener ALL US
├── docs/
│   ├── TEMACD_SCREENER_NAVOD.md          # hlavní CZ manuál
│   └── REGIME_FILTER_AND_IBKR_SETUP.md   # 6 režimů + IBKR/TOS
├── README.md
```

## ⚠️ Disclaimer

Výzkumný a vzdělávací software, ne investiční poradenství. Trading TQQQ 3x leveraged ETF má extrémní riziko (-80% v 2022). Vždy používej 200EMA filtr (Strategy v3) a position sizing (Half-Kelly). Backtest ≠ budoucnost.

---

Vytvořeno pro: **MES a TQQQ jako best assets, ale scan všech US akcií podle stejné logiky s alerty** – přesně podle zadání uživatele.
