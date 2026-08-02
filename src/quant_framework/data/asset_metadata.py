"""
Asset Metadata Engine – institutional level
Each ticker: asset class, sector, industry, country, currency, beta, avg vol, avg CAGR, listing date, ETF issuer, theme, macro bucket, risk bucket
Automatic classification: XLU=Utilities Defensive Low etc.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


@dataclass(frozen=True)
class AssetMetadata:
    ticker: str
    asset_class: str
    sector: str
    industry: str
    country: str
    currency: str
    beta: float
    avg_volatility: float
    avg_cagr: float
    listing_date: Optional[pd.Timestamp]
    etf_issuer: str
    theme: str
    macro_bucket: str
    risk_bucket: str
    description: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get('listing_date') and isinstance(d['listing_date'], pd.Timestamp):
            d['listing_date'] = d['listing_date'].isoformat()
        return d


KNOWN_ASSETS: Dict[str, Dict] = {
    "XLU": {"asset_class": "ETF", "sector": "Utilities", "industry": "Utilities", "macro_bucket": "Defensive", "risk_bucket": "Low", "beta": 0.5, "theme": "Defensive Yield", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLP": {"asset_class": "ETF", "sector": "Consumer Staples", "industry": "Consumer Staples", "macro_bucket": "Defensive", "risk_bucket": "Low", "beta": 0.6, "theme": "Staples", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLV": {"asset_class": "ETF", "sector": "Health Care", "industry": "Health Care", "macro_bucket": "Defensive", "risk_bucket": "Low", "beta": 0.7, "theme": "Healthcare", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLE": {"asset_class": "ETF", "sector": "Energy", "industry": "Oil & Gas", "macro_bucket": "Inflation", "risk_bucket": "High", "beta": 1.3, "theme": "Energy", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLB": {"asset_class": "ETF", "sector": "Materials", "industry": "Basic Materials", "macro_bucket": "Inflation", "risk_bucket": "Medium", "beta": 1.1, "theme": "Materials", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLI": {"asset_class": "ETF", "sector": "Industrials", "industry": "Industrials", "macro_bucket": "Real Economy", "risk_bucket": "Medium", "beta": 1.1, "theme": "Industrials", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLK": {"asset_class": "ETF", "sector": "Technology", "industry": "Technology", "macro_bucket": "Risk On", "risk_bucket": "Medium", "beta": 1.2, "theme": "Technology Growth", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLF": {"asset_class": "ETF", "sector": "Financials", "industry": "Financials", "macro_bucket": "Rate", "risk_bucket": "Medium", "beta": 1.2, "theme": "Financials", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "XLY": {"asset_class": "ETF", "sector": "Consumer Discretionary", "industry": "Consumer Discretionary", "macro_bucket": "Risk On", "risk_bucket": "Medium", "beta": 1.2, "theme": "Discretionary", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "SPY": {"asset_class": "ETF", "sector": "Broad Market", "industry": "Broad Market", "macro_bucket": "Risk On", "risk_bucket": "Medium", "beta": 1.0, "theme": "S&P 500", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "QQQ": {"asset_class": "ETF", "sector": "Technology", "industry": "Nasdaq 100", "macro_bucket": "Risk On", "risk_bucket": "Medium", "beta": 1.2, "theme": "Growth", "etf_issuer": "Invesco", "country": "US", "currency": "USD"},
    "TQQQ": {"asset_class": "ETF", "sector": "Technology", "industry": "Nasdaq 100 3x Leveraged", "macro_bucket": "Risk On", "risk_bucket": "Very High", "beta": 3.6, "theme": "Leveraged Growth", "etf_issuer": "ProShares", "country": "US", "currency": "USD"},
    "GLD": {"asset_class": "ETF", "sector": "Commodities", "industry": "Gold", "macro_bucket": "Safe Haven", "risk_bucket": "Low", "beta": 0.1, "theme": "Gold Commodity", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "BIL": {"asset_class": "ETF", "sector": "Fixed Income", "industry": "T-Bills 1-3M", "macro_bucket": "Cash", "risk_bucket": "Low", "beta": 0.0, "theme": "Cash Equivalent", "etf_issuer": "SPDR", "country": "US", "currency": "USD"},
    "IEF": {"asset_class": "ETF", "sector": "Fixed Income", "industry": "Treasury 7-10Y", "macro_bucket": "Defensive", "risk_bucket": "Low", "beta": 0.2, "theme": "Intermediate Treasuries", "etf_issuer": "iShares", "country": "US", "currency": "USD"},
    "TLT": {"asset_class": "ETF", "sector": "Fixed Income", "industry": "Treasury 20Y+", "macro_bucket": "Safe Haven", "risk_bucket": "Medium", "beta": -0.2, "theme": "Long Duration", "etf_issuer": "iShares", "country": "US", "currency": "USD"},
    "HYG": {"asset_class": "ETF", "sector": "Fixed Income", "industry": "High Yield", "macro_bucket": "Credit", "risk_bucket": "Medium", "beta": 0.5, "theme": "High Yield Credit", "etf_issuer": "iShares", "country": "US", "currency": "USD"},
    "VIXY": {"asset_class": "ETF", "sector": "Volatility", "industry": "VIX Futures", "macro_bucket": "Fear", "risk_bucket": "Very High", "beta": -3.0, "theme": "Long Volatility", "etf_issuer": "ProShares", "country": "US", "currency": "USD"},
    "UUP": {"asset_class": "ETF", "sector": "Currency", "industry": "Dollar", "macro_bucket": "Dollar", "risk_bucket": "Low", "beta": 0.1, "theme": "US Dollar", "etf_issuer": "Invesco", "country": "US", "currency": "USD"},
    "DBMF": {"asset_class": "ETF", "sector": "Alternative", "industry": "Managed Futures", "macro_bucket": "Alternative", "risk_bucket": "Medium", "beta": 0.2, "theme": "Managed Futures Replication", "etf_issuer": "iMGP", "country": "US", "currency": "USD"},
    "BTC-USD": {"asset_class": "Crypto", "sector": "Crypto", "industry": "Bitcoin", "macro_bucket": "Risk On", "risk_bucket": "Very High", "beta": 2.5, "theme": "Bitcoin", "etf_issuer": "N/A", "country": "Global", "currency": "USD"},
    "MES=F": {"asset_class": "Futures", "sector": "Broad Market", "industry": "Micro E-mini S&P 500", "macro_bucket": "Risk On", "risk_bucket": "Medium", "beta": 1.0, "theme": "Micro ES", "etf_issuer": "CME", "country": "US", "currency": "USD"},
}


class AssetMetadataEngine:
    def __init__(self, cache_data: Dict[str, pd.DataFrame] = None):
        self.cache_data = cache_data or {}
        self._metadata_cache: Dict[str, AssetMetadata] = {}

    @staticmethod
    def _classify_macro_bucket(sector: str, asset_class: str, ticker: str) -> str:
        sector_lower = sector.lower()
        t = ticker.upper()
        if t in ["XLU", "XLP", "XLV", "BIL", "IEF"] or "utilities" in sector_lower or "staples" in sector_lower or "health" in sector_lower:
            return "Defensive"
        if t in ["XLE", "XLB", "USO"] or "energy" in sector_lower or "materials" in sector_lower:
            return "Inflation"
        if t in ["GLD", "SLV", "TLT"] or "gold" in sector_lower:
            return "Safe Haven"
        if t in ["UUP"]:
            return "Dollar"
        if t in ["VIXY"]:
            return "Fear"
        if t in ["SVXY", "QQQ", "TQQQ", "XLK", "XLY", "BTC-USD", "MES=F", "SPY", "IWM"]:
            return "Risk On"
        if t in ["HYG", "LQD"]:
            return "Credit"
        if t in ["DBMF"]:
            return "Alternative"
        if asset_class == "Crypto":
            return "Risk On"
        return "Balanced"

    @staticmethod
    def _classify_risk_bucket(beta: float, volatility: float, asset_class: str) -> str:
        if asset_class == "Crypto" or abs(beta) >= 3 or volatility >= 50:
            return "Very High"
        if abs(beta) >= 1.3 or volatility >= 28:
            return "High"
        if abs(beta) >= 0.8 or volatility >= 16:
            return "Medium"
        return "Low"

    @staticmethod
    def _classify_asset_class(ticker: str) -> str:
        t = ticker.upper()
        if "-USD" in t:
            return "Crypto"
        if "=F" in t:
            return "Futures"
        if t in ["DBMF"]:
            return "Managed Futures"
        if t in KNOWN_ASSETS:
            return KNOWN_ASSETS[t].get("asset_class", "ETF")
        return "Equity"

    def get_metadata(self, ticker: str) -> AssetMetadata:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be str")
        t = ticker.strip().upper()
        if not t:
            raise ValueError("ticker cannot be empty")
        if t in self._metadata_cache:
            return self._metadata_cache[t]

        known = KNOWN_ASSETS.get(t, {})
        asset_class = known.get("asset_class", self._classify_asset_class(t))
        sector = known.get("sector", "Unknown")
        industry = known.get("industry", "Unknown")
        country = known.get("country", "US")
        currency = known.get("currency", "USD")
        beta = float(known.get("beta", 1.0))
        etf_issuer = known.get("etf_issuer", "Unknown")
        theme = known.get("theme", sector)
        macro_bucket = known.get("macro_bucket", self._classify_macro_bucket(sector, asset_class, t))
        risk_bucket = known.get("risk_bucket", "Medium")

        avg_vol = 20.0
        avg_cagr = 8.0
        listing_date = None

        df = self.cache_data.get(t)
        if df is not None and not df.empty and 'Close' in df.columns:
            try:
                close = df['Close'].dropna()
                if len(close) > 30:
                    daily_ret = close.pct_change().dropna()
                    avg_vol = float(daily_ret.std() * np.sqrt(252) * 100)
                    years = (close.index[-1] - close.index[0]).days / 365.25
                    if years > 0:
                        total_return = close.iloc[-1] / close.iloc[0] - 1
                        avg_cagr = float(((1 + total_return) ** (1 / years) - 1) * 100)
                    listing_date = close.index.min()
            except Exception:
                pass

        if t not in KNOWN_ASSETS or "risk_bucket" not in known:
            risk_bucket = self._classify_risk_bucket(beta, avg_vol, asset_class)

        meta = AssetMetadata(
            ticker=t, asset_class=asset_class, sector=sector, industry=industry, country=country,
            currency=currency, beta=round(float(beta), 2), avg_volatility=round(float(avg_vol), 2),
            avg_cagr=round(float(avg_cagr), 2), listing_date=listing_date, etf_issuer=etf_issuer,
            theme=theme, macro_bucket=macro_bucket, risk_bucket=risk_bucket,
            description=f"{t} – {sector} / {theme} – {macro_bucket} – {risk_bucket} risk"
        )
        self._metadata_cache[t] = meta
        return meta

    def get_all_metadata(self, tickers: List[str]) -> Dict[str, AssetMetadata]:
        clean = sorted(list(dict.fromkeys([t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()])))
        result = {}
        for t in clean:
            try:
                result[t] = self.get_metadata(t)
            except Exception:
                result[t] = AssetMetadata(
                    ticker=t, asset_class="Unknown", sector="Unknown", industry="Unknown",
                    country="US", currency="USD", beta=1.0, avg_volatility=20.0, avg_cagr=5.0,
                    listing_date=None, etf_issuer="Unknown", theme="Unknown",
                    macro_bucket="Balanced", risk_bucket="Medium", description=f"{t} unknown"
                )
        return result

    def get_by_macro_bucket(self, macro_bucket: str, tickers: List[str]) -> List[str]:
        all_meta = self.get_all_metadata(tickers)
        filtered = [t for t, m in all_meta.items() if m.macro_bucket.lower() == macro_bucket.lower()]
        return sorted(filtered)

    def get_by_risk_bucket(self, risk_bucket: str, tickers: List[str]) -> List[str]:
        all_meta = self.get_all_metadata(tickers)
        filtered = [t for t, m in all_meta.items() if m.risk_bucket.lower() == risk_bucket.lower()]
        return sorted(filtered)

    def to_dataframe(self, tickers: List[str]) -> pd.DataFrame:
        meta_dict = self.get_all_metadata(tickers)
        rows = [m.to_dict() for m in meta_dict.values()]
        df = pd.DataFrame(rows)
        return df.sort_values("ticker").reset_index(drop=True)
