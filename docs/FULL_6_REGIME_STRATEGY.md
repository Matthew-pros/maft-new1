# 6-Regime Antifragile Control Plane – Syntéza 10 studií

## Architektura

```
[1] RISK_ON_GROWTH         → TQQQ TEMACD + QQQ 3X EMACD + French 100Y
[2] INFLATION_GROWTH       → GLD 3X EMACD + Commodity Beta-Rotation
[3] RANGE_BOUND_WHIPSAW    → Donchian Combo + Carver/FTI + 5m-ORB + VWAP
[4] DEFENSIVE_ROTATION     → LT MA CROSS + XLU/XLV/XLP + IEF
[5] CREDIT_STRESS          → Protective Hedge (TLT,IEF,GLD,UUP,0% akcie)
[6] BLACK_SWAN_VOL_SHOCK   → BIL Cash Override + VIXY/SVXY
```

## 10 studií zapojených

1. **TEMACD & 6D Beta** – Triple EMA 5,50,222 + MACD 61,70,15
2. **French 100-Year Industry Rotation** – 100 let důkaz rotace do momentum sektorů
3. **VIXY/SVXY Dual Edge** – contango vs backwardation jako brána do Black Swan
4. **Beat the Market Whipsaw** – vol-gating + low-corr filter v sideway
5. **5m-ORB Stop @ High/Low** – intraday bez overnight gapu
6. **VWAP Institutional** – mean-reversion benchmark
7. **True Strength Index TSI** – double smoothed momentum
8. **Anthropic Financial Framework** – deterministické ověřování, decision_trace
9. **Power Law 30-Stock** – FANG + sector ETFs diversifikace
10. **Antifragile Control Plane** – TRank = 40% Mom +20% LowVol +20% LowCorr +20% ATR + BIL override

## Rozhodovací strom (kód z regime_engine_6.py)

```python
if vix>=30 or hyg_ief <= -0.04: return 6
if hyg_ief < -0.012 and xlu_spy > 0.008: return 5
if xlu_spy > 0.005 and xly_xlp < -0.005: return 4
if 18 <= vix < 25: return 3
if xle_xlu > 0.008 and xlb_gld > 0.005: return 2
return 1
```

## Alokace

| ID | Režim | EQ:HD:CA | Typické symboly |
|---|---|---|---|
|1|RISK_ON|100:0:0|TQQQ,QQQ,NVDA,XLK,BTC|
|2|INFLATION|75:20:5|XLE,XLB,XLI,GLD|
|3|WHIPSAW|40:30:30|XLV,XLP,GLD,IEF,BIL|
|4|DEFENSIVE|50:35:15|XLU,XLV,XLP,IEF|
|5|CREDIT|0:80:20|TLT,IEF,GLD,UUP|
|6|BLACK_SWAN|0:20:80|BIL,VIXY,GLD|

## Spuštění

```bash
python tools/regime_engine_6.py --offline
python tools/carver_ensemble_allocator.py --regime 1 --capital 100000
python tools/institutional_backtest_engine.py --offline --export-png dashboard.png
```
