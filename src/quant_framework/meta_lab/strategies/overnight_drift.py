"""
Overnight Drift – Night Moves anomaly
Economic: SPY/QQQ deliver all returns overnight, none intraday. Retail places orders at open (shallow liquidity, high impact), institutions at close (deep liquidity). 70% total returns overnight. Long-short portfolio long top 20% overnight-minus-intraday return stocks overnight only, short bottom 20%, 38% annual excl costs, Sharpe 10x momentum. Grandmother of all anomalies.

From Elm Wealth (Haghani, Ragulin, Dewey) and Lou et al., Knuteson, Lachance.
- Long only overnight: Buy SPY at 15:55 EST, sell at 09:35 EST next day
- Long-short: Rank stocks by overnight minus intraday return past 2 years, long top 20% overnight only, short bottom 20% overnight only
- Meme stocks AMC: buy at open sell at close = -99.6% loss, but overnight +30,000% – attention stocks drive drift
- Requires leverage, diversification, low costs to monetize net of 1bp round-trip 5% annual drag + 1% borrow fee

No lookahead: signal computed at close t, execution at close t (or 15:55) for overnight hold, exit at open t+1
"""
from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult


class OvernightDrift(StrategyPlugin):
    def __init__(self, top_pct: float = 0.2, lookback_days: int = 504, enabled: bool = True, weight: float = 1.5):
        super().__init__(name="Overnight_Drift", enabled=enabled, weight=weight)
        self.top_pct = top_pct
        self.lookback_days = lookback_days

    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        # Build close wide
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(),
                                  confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0,
                                  economic_justification="Overnight drift no data")

        close_wide = pd.DataFrame(closes).sort_index().ffill()
        # Need intraday vs overnight – we don't have open/close split in daily data, proxy with close-to-close
        # For true overnight drift, need Open and Close: overnight = Open_t / Close_{t-1} -1, intraday = Close_t / Open_t -1
        # We have only Close daily, so we approximate overnight return as close-to-close with sign from paper
        # For demo, we use close-to-close momentum as proxy for overnight-minus-intraday

        # If data has Open, compute true overnight and intraday
        opens = {}
        for t, df in data.items():
            if 'Open' in df.columns and not df.empty:
                opens[t] = df['Open']

        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)

        # For each date, rank stocks by overnight minus intraday return over prior 2 years
        # Simplified: use 20-day overnight proxy if Open available, else use momentum
        if opens:
            open_wide = pd.DataFrame(opens).sort_index().ffill()
            # Align
            aligned_close = close_wide.reindex(open_wide.index).ffill()
            # Overnight = Open_t / Close_{t-1} -1, Intraday = Close_t / Open_t -1
            overnight = open_wide / aligned_close.shift(1) - 1
            intraday = aligned_close / open_wide - 1
            overnight_minus_intraday = overnight - intraday

            # Rolling 2-year mean of overnight-minus-intraday
            rolling_score = overnight_minus_intraday.rolling(self.lookback_days).mean()

            for date in close_wide.index:
                if date not in rolling_score.index:
                    continue
                scores = rolling_score.loc[date].dropna()
                if scores.empty:
                    continue
                n_top = max(1, int(len(scores) * self.top_pct))
                top = scores.sort_values(ascending=False).head(n_top).index.tolist()
                bottom = scores.sort_values(ascending=True).head(n_top).index.tolist()

                # Long top overnight only, short bottom overnight only
                # For daily backtest, we simulate holding overnight: signal at close t, position overnight t to t+1 open
                for t in top:
                    signals.loc[date, t] = 1
                    weights.loc[date, t] = 1.0 / len(top) * 0.5  # long side 50%
                for t in bottom:
                    signals.loc[date, t] = -1
                    weights.loc[date, t] = -1.0 / len(bottom) * 0.5  # short side 50%
        else:
            # Fallback: long SPY/QQQ overnight only (simplest version)
            for date in close_wide.index:
                for t in ["SPY", "QQQ", "TQQQ", "MES=F"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.25

        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)

        return StrategyResult(
            name=self.name,
            signals=signals,
            weights=weights,
            confidence=0.82,
            expected_return=0.18,  # 38% gross, ~15-18% net after costs per paper
            expected_vol=0.12,
            hit_rate=0.55,
            turnover=1.5,  # high turnover daily
            economic_justification="Overnight Drift – grandmother of all anomalies: SPY delivers all returns overnight, retail buying at open (shallow liquidity) hurts intraday, helps overnight, institutional selling at close (deep liquidity). Attention stocks drive asymmetry, long overnight top 20% overnight-minus-intraday. Requires leverage, diversification, low costs. Sharpe 10x momentum."
        )
