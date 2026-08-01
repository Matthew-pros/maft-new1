# Další strategie z Quant Rick videí – BTC TEMACD, LT MA CROSS, QQQ 3X EMACD, GLD 3X EMACD, Blended Donchian + Carver

Tento dokument rozšiřuje hlavní TEMACD strategii o další 5 ensemble systémů, které se ve videích objevují jako komplementární. Všechny fungují na stejném principu **binary queue** (EMA crossover bez fixed target) ale s jinými parametry optimalizovanými pro dané aktivum.

## 1. BTC TEMACD – Crypto Engine

**Původ:** Quant Rick BTC-USD Ensemble report  
**Parametry:**
- EMA: **5, 64, 170** (místo 5,50,222 pro akcie) – delší střední EMA 64 filtruje crypto whipsaw
- MACD: **32, 40, 94** (místo 61,70,15) – rychlejší než equity verze, protože crypto trendy jsou kratší ale silnější

**Proč jiné parametry?**
- Crypto 24/7, vyšší volatilita 65% annual vs 18% QQQ, trend-following musí být citlivější na krátké momentum
- Backtest z reportu: CAGR 72.85% vs Buy&Hold 31.48%, Sharpe 1.462 vs 0.750, MaxDD -34.52% vs -81.53%, WinRate 38.1% (asymetrický – málo výher ale obrovské), Profit Factor 7.39

**Pine kód:**
```pine
//@version=5
strategy("BTC TEMACD", overlay=true)
len1=5; len2=64; len3=170; fast=32; slow=40; sig=94
ema1=ta.ema(close,len1); ema2=ta.ema(close,len2); ema3=ta.ema(close,len3)
macd=ta.ema(close,fast)-ta.ema(close,slow); aMacd=ta.ema(macd,sig); delta=macd-aMacd
long = ta.crossover(delta,0) or ta.crossover(ema1,ema2) or ta.crossover(ema1,ema3) or ta.crossover(ema2,ema3)
short = ta.crossunder(delta,0) or ta.crossunder(ema1,ema2) or ta.crossunder(ema1,ema3) or ta.crossunder(ema2,ema3)
if long
    strategy.entry("long", strategy.long)
if short
    strategy.close("long")
```

**Použití v 6-režimovém portfoliu:** Režim 1 RISK_ON_GROWTH – 25% alokace BTC-USD (vol-targeted ~30% nominálu kvůli 65% vol). Korelace s QQQ jen ~0.18 → zvyšuje Sharpe portfolia na >=2.08.

---

## 2. LT MA CROSS – Long-Term Moving Average Cross

**Parametry:** EMA(50) x EMA(200) nebo SMA(50)xSMA(200) – klasický Golden/Death Cross, ale použitý jako **regime filter** ne jako standalone.

**Logika:**
- Long-only když close > EMA200 a EMA50 > EMA200
- Slouží jako Circuit Breaker pro TQQQ – pokud SPY < EMA200, zavři všechny equity (Strategy v3 – Full Sample Results: Total 241.5%, CAGR 24.67%, Sharpe 1.301, MaxDD -15.36%, WinRate 73.7%, Profit Factor 9.17, Trades/year 2.3)

**Pine:**
```pine
ema50=ta.ema(close,50); ema200=ta.ema(close,200)
long = ta.crossover(ema50,ema200) and close>ema200
short = ta.crossunder(ema50,ema200) or close<ema200
```

**Použití:** Režim 4 DEFENSIVE_ROTATION – filtruje které defenzivní ETF smí být držena (XLU,XLV,XLP musí mít 50>200).

---

## 3. QQQ 3X EMACD – super na prevent black swan

**Stejné parametry jako TQQQ TEMACD:** EMA 5,50,222 + MACD 61,70,15
**Rozdíl:** Aplikováno na QQQ (ne TQQQ) s 3x pákovým sizingem, ale s tvrdým 200EMA filtrem který zabrání -80% DD v roce 2022.

**Proč "prevent black swan"?**
- QQQ má 18% vol vs TQQQ 55% – méně dramatický pád při VIX spike
- Když SPY < EMA200 → celá pozice do BIL (cash override) – backtest ukazuje MaxDD sražen z -33.7% na -13.4% (viz equity curve red shading = below 200EMA bear regime)
- V Black Swan režimu (VIX>=30) funguje jako hedge – QQQ klesne méně než TQQQ, a s BIL override ztratíš -4.5% místo -30%

**Sizing:** Half-Kelly pro 3x ETF: `f* = 0.5 * mu / sigma^2`. Pro TQQQ doporučeno 0.3-0.5x full equity.

---

## 4. GLD 3X EMACD – dobrý na props

**Parametry:** Stejné 5,50,222 + 61,70,15 ale na GLD (gold) s 3x pákou (UGL pro 2x, UGLD pro 3x, nebo futures GC).

**Proč funguje:**
- Gold má annual vol 15% vs 55% TQQQ – nízká korelace k akciím -0.35 (deflation hedge), ideální pro prop firm challenge kde potřebuješ nízký DD
- V inflačním režimu (XLE/XLU nahoru) GLD trenduje silně – 3X EMACD zachytí trend bez whipsaw díky pomalému MACD 61,70
- Sharpe v commodity režimu ~1.10-1.15, ale v kombinaci s equities zvedne celkové Sharpe na 2.08

**Použití:** Režim 2 INFLATION_GROWTH – 20% GLD, Režim 5 CREDIT_STRESS – 20% GLD jako ochrana.

---

## 5. Blended – Donchian Combo + Carver/FTI

**Kombinace dvou klasických trend systémů:**

- **Donchian Channel Breakout:** Buy když close > 20-day high, Sell když close < 10-day low (nebo 55-day high pro pomalejší). Bez targetu, trailing.
- **Carver/FTI (Fast Trend Indicator):** EWMAC – Exponentially Weighted Moving Average Crossover – definice: `price > EMA(fast)` a `EMA(fast) > EMA(slow)`. Carver používá 4 rychlosti EWMAC (2-512 dní) jako ensemble hlasování.

**Blended logika:** Donchian dává breakout entry, Carver dává vol-targeted sizing + trend filter. Společně tvoří **Toyota, Not Ferrari** – low parametrization, robustní, funguje na všech aktivech s různými periodami.

**Pine Donchian:**
```pine
lenEntry=20; lenExit=10
upper=ta.highest(high,lenEntry); lower=ta.lowest(low,lenExit)
long = ta.crossover(close,upper[1])
short = ta.crossunder(close,lower[1])
```

**Použití:** Režim 3 RANGE_BOUND_WHIPSAW – místo držení akcií overnight, obchoduj 5m-ORB (intraday Opening Range Breakout) nebo Donchian na 20/55 s VWAP filtrem. Commission 0.0035/share, žádné gap riziko.

---

## 6. Jak vše zapadá do super-robust 6-režimového portfolia

| Režim | Hlavní engine | Další ensemble pro diverzifikaci | Alokace |
|---|---|---|---|
| 1 RISK_ON_GROWTH | TQQQ TEMACD | QQQ 3X EMACD + BTC TEMACD (korelace 0.18) | 100% EQ (40 TQQQ,25 QQQ,25 BTC,10 XLK) |
| 2 INFLATION_GROWTH | GLD 3X EMACD + Commodity Beta-Rotation | Donchian Combo na XLE/XLB | 75% EQ (XLE,XLB,XLI) 20% GLD 5% BIL |
| 3 RANGE_BOUND_WHIPSAW | Donchian + Carver + 5m-ORB VWAP | LT MA CROSS filter pro defenzivu | 40% EQ (XLV,XLP) 30% IEF/GLD 30% Cash/ORB |
| 4 DEFENSIVE_ROTATION | LT MA CROSS on XLU/XLV/XLP | – | 50% XLU/XLV/XLP 35% IEF 15% BIL |
| 5 CREDIT_STRESS | Protective Hedge (0% akcie) | GLD 3X EMACD jako safe haven | 0% EQ 80% TLT/IEF/GLD 20% BIL |
| 6 BLACK_SWAN | BIL Cash Override + VIXY long | QQQ 3X EMACD s 200EMA breaker jako hedge | 0% EQ 20% VIXY/GLD 80% BIL |

**Výsledné Sharpe multi-asset:**
- TQQQ TEMACD Sharpe 1.30
- BTC Ensemble Sharpe 1.46 (korelace 0.18)
- GLD Sharpe 1.10 (korelace -0.35)
- Donchian/Carver Sharpe 1.15 (korelace 0.22)
- **Kombinované Sharpe >=2.08** díky vol-targeting `Dollar Exposure = Capital * TargetVol / AssetVol` a nízké korelaci (Robert Carver formula LOTS = Dollar Exposure / (Price*Multiplier))

**FTMO tipy:** Mega-cap FANG (AAPL, MSFT, NVDA, AMZN, META, GOOGL) + US100/US500/US30 indexy, avoid leverage na FTMO (1:30 + 5% daily loss = blowup risk), use Toyota Not Ferrari (low params).

**Beta rotace jako meta-layer:** 6 dimenzí XLU/SPY, XLY/XLP, XLB/GLD, HYG/IEF, XLK/XLV, XLE/XLU jako voters, ne dictators – z-score vs historie → composite risk-on/off → weight adjustment mezi momentum/volatility v TRank (40% Mom +20% LowVol +20% LowCorr +20% ATR).

**VIX-adaptive:** Když VIX>30, zkrať RSI lookback na 21-42 dní pro rychlejší rotaci do nových lídrů.

Tím máš super-robust plně automatický systém pro všechny 6 režimů, testovaný PSR 1.000 / DSR 1.000 v institutional backtest enginu.
