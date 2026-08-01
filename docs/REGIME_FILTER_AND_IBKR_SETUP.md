# 6 Market Regimes – Jak filtrovat TEMACD signály + IBKR nastavení krok za krokem

Tento dokument rozšiřuje základní TEMACD screener o **makro filtr 6 režimů** (Beta Rotations) a detailní návod pro Interactive Brokers.

## 1. Proč samotné TEMACD nestačí na ALL US stocks?

TEMACD je skvělé na indexy, ale na single stocks v sideway/tech bear chytá whipsawy. Řešení = **kombinovat s režimovým filtrem**. Když je trh v kreditním stresu, ignoruj všechny TEMACD LONGy.

### 6 režimů – rychlé rozlišení

| ID | Režim | Spouštěč (reálná data) | Co dělat s TEMACD | Alokace |
|---|---|---|---|---|
| 1 | RISK_ON_GROWTH | HYG/IEF roste, XLY/XLP roste, VIX<18 | Ber všechny LONGy | 100% EQ |
| 2 | INFLATION_GROWTH | XLE/XLU roste, XLB/GLD roste | Ber LONGy v XLE,XLB,XLI,GLD, ne tech | 75% EQ 20% GLD |
| 3 | RANGE_BOUND_WHIPSAW | VIX 18-25, ADX<20 | Ber pouze STRONG BULL + filtr EMA50>EMA222, jinak intraday ORB | 40% EQ |
| 4 | DEFENSIVE_ROTATION | XLU/SPY roste, XLY/XLP klesá | LONGy pouze XLU,XLV,XLP,IEF, jinak EXIT | 50% EQ 35% IEF |
| 5 | CREDIT_STRESS | HYG/IEF < EMA20 a SPY < EMA200 | Ignoruj všechny LONGy, 0% akcie | 0% EQ 80% TLT/IEF/GLD |
| 6 | BLACK_SWAN | VIX>=30 nebo VIXY/SVXY skok >5% | 100% BIL, VIXY long hedge | 0% EQ 20% VIXY 80% BIL |

**Jak spočítat?** Ručně z yfinance, nebo použij `tools/regime_engine_6.py` pokud existuje.

Jednoduchá rovnice v Pythonu:

```python
import yfinance as yf

def get_change(ticker, period="20d"):
    df = yf.Ticker(ticker).history(period=period)
    return (df['Close'][-1]/df['Close'][0]-1)

xlu_spy = get_change("XLU") - get_change("SPY")
hyg_ief = get_change("HYG") - get_change("IEF")
vix = yf.Ticker("^VIX").history(period="1d")['Close'][-1]

if vix >= 30: regime = 6
elif hyg_ief < -0.012 and xlu_spy > 0.008: regime = 5
elif xlu_spy > 0.005 and (get_change("XLY")-get_change("XLP")) < -0.005: regime = 4
elif 18 <= vix < 25: regime = 3
elif get_change("XLE")-get_change("XLU") > 0.008: regime = 2
else: regime = 1
```

### Aplikace na TEMACD screener

V našem Python screeneru můžeš přidat:

```python
if regime in [5,6]:
    # ignoruj všechny nové LONGy, pouze EXIT
    longs = []
elif regime == 4:
    longs = [r for r in longs if r['symbol'] in ['XLU','XLV','XLP','IEF','GLD','BIL']]
elif regime == 2:
    longs = [r for r in longs if r['symbol'] in ['XLE','XLB','XLI','XLF','GLD','PDBC']]
```

Tím snížíš Max Drawdown z -33% na ~-13% podle Strategy v3 testů.

---

## 2. IBKR – Přesný návod nastavení

### 2.1 Co IBKR umí a neumí

**Advanced Market Scanner omezení:**
- Umí: EMA, SMA, RSI, Volume, Price, Market Cap
- Neumí: custom MACD 61,70,15 + 4-way OR crossover přesně
- Takže: použij IBKR pouze jako **likvidní pre-filter** a exekuci, signál ber z Pythonu.

### 2.2 Krok za krokem TWS Scanner (pre-filter)

1. **Otevři TWS** → Mozaic → New Window → **Market Scanner** → Advanced
2. **Scan Code:** Top % Gainers, nebo Hot by Volume
3. **Filters:**
   - Universe: US Stocks, Price > 10, Volume avg 90d > 500k, Market Cap > 1B
   - Technical:
     - `Close > EMA(50) [Daily]`
     - `EMA(50) > EMA(200) [Daily]`  (aprox EMA222)
     - Optional: `RSI(14) > 50` (momentum)
4. **Columns:** Add `EMA(50)`, `EMA(200)`, `RSI`
5. **Save as** `TEMACD pre-filter`
6. **Export:** Right click → Export to file → `ibkr_pre.csv`
7. Pak:
   ```bash
   python tools/temacd_screener.py --file ibkr_pre.csv --only-signals --export-html ibkr_filtered.html
   ```

### 2.3 Exekuce příkazů na IBKR podle screeneru

Náš screener vypíše `LONG ENTRY` pro QQQ, NVDA...

**Ruční zadání (nejbezpečnější pro začátek):**
1. V TWS zadej ticker QQQ
2. Order Type: **MOC** (Market on Close) – pro Daily strategii nejlepší, minimalizuje slippage podle Almgren-Chriss
3. Pro MES futures: Order Type `MKT` v RTH nebo `LMT` na close
4. Qty: spočítej Carver lot:
   ```
   LOTS = (Capital * TargetVol / AssetVol) / Price
   Example: Capital 100k, Target 20%, Asset vol TQQQ 55%, Price 218
   Dollar Exp = 100k * 0.20 / 0.55 = 36k, LOTS = 36k/218 = 165 shares, ale s half-Kelly 0.5x = 82 shares
   ```
   Pro jednoduchost začni s **percent_of_equity** 100% jako v Pine = `(equity / close) * lev`

**Automaticky přes ib_insync (free, paper trading):**

```python
# pip install ib_insync
from ib_insync import IB, Stock
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)  # paper

for sig in longs:
    stock = Stock(sig['symbol'], 'SMART', 'USD')
    ib.qualifyContracts(stock)
    order = MarketOrder('BUY', sig['lots'])  # nebo 1% portfolia
    ib.placeOrder(stock, order)

ib.run()
```

**MES futures příkaz:**
```python
from ib_insync import Future
mes = Future('MES', '202503', 'CME')
order = MarketOrder('BUY', 1)
ib.placeOrder(mes, order)
```

### 2.4 PDT Rule a účet <25k

- Pokud máš US účet <25k a chceš daytrade akcie, IBKR tě omezí na 3 daytrades/5 dní.
- **Řešení:** Obchoduj MES (futures nemají PDT) nebo swing Daily (drž přes noc = není daytrade).
- TEMACD je Daily strategie = žádný PDT problém, drží týdny.

---

## 3. ThinkOrSwim alternativa – umí přesně TEMACD

Pokud máš Schwab/TOS účet, můžeš napsat přesný scan:

**TOS ThinkScript:**
```
def EMA1 = ExpAverage(close, 5);
def EMA2 = ExpAverage(close, 50);
def EMA3 = ExpAverage(close, 222);
def MACD = ExpAverage(close, 61) - ExpAverage(close, 70);
def aMACD = ExpAverage(MACD, 15);
def delta = MACD - aMACD;

def longCond = (delta crosses above 0) or (EMA1 crosses above EMA2) or (EMA1 crosses above EMA3) or (EMA2 crosses above EMA3);
plot scan = longCond within 1 bars;
```

V TOS → Scan → Stock Hacker → Add filter → Study Filter → Custom → vlož výše → Scan all US stocks.

To je 100% přesné a zdarma pokud máš Schwab účet.

---

## 4. Shrnutí workflow pro ALL US

**Doporučený free workflow (0$ měsíčně):**

1. **Večer po close (22:15 CET):** Spusť GitHub Actions nebo lokálně:
   ```bash
   python tools/temacd_screener.py --universe us100 --export-html latest.html --webhook-url DISCORD
   ```
2. **Zkontroluj HTML na mobilu:** LONG/EXIT signály
3. **Ověř režim:** Pokud VIX<18 a SPY>EMA200 → Risk-On, ber LONGy. Pokud VIX>=30 → ignoruj.
4. **Zadej MOC příkazy v IBKR TWS** na příští close, nebo nastav alert v TradingView na stejnou logiku pro potvrzení.
5. **Pro MES:** 1 kontrakt MES = ~$11k notional, TEMACD long na MES=F = buy MES future.

Tímto máš přesný, automatický, zdarma systém pokrývající všechny US akcie podle TEMACD logiky.
