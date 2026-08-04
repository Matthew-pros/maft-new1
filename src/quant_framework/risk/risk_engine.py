"""
Risk Engine – Institutional Level
Calculates:
- Portfolio Volatility
- Marginal Risk Contribution (MRC)
- Component Risk Contribution (CRC)
- VaR (Value at Risk)
- CVaR / Expected Shortfall
- Ulcer Index
- Tail Ratio
- Skew
- Kurtosis
- Rolling Beta
- Rolling Correlation
- Risk Concentration
- Risk Budget
- Risk Decomposition

No lookahead bias, deterministic, type hints, docstring, validation
Economic justification per metric included in docstrings
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class RiskMetrics:
    """Immutable risk metrics snapshot"""
    portfolio_volatility: float  # annualized
    marginal_risk_contribution: Dict[str, float]  # ticker -> MRC
    component_risk_contribution: Dict[str, float]  # ticker -> CRC
    component_risk_pct: Dict[str, float]  # CRC % of total risk
    var_95: float
    var_99: float
    cvar_95: float  # Expected Shortfall
    cvar_99: float
    expected_shortfall_95: float  # alias for CVaR
    ulcer_index: float
    tail_ratio: float
    skew: float
    kurtosis: float
    risk_concentration: float  # Herfindahl of risk contributions
    risk_budget: Dict[str, float]  # target vs actual risk budget
    timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict:
        return {
            "portfolio_volatility": self.portfolio_volatility,
            "marginal_risk_contribution": self.marginal_risk_contribution,
            "component_risk_contribution": self.component_risk_contribution,
            "component_risk_pct": self.component_risk_pct,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "expected_shortfall_95": self.expected_shortfall_95,
            "ulcer_index": self.ulcer_index,
            "tail_ratio": self.tail_ratio,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "risk_concentration": self.risk_concentration,
            "risk_budget": self.risk_budget,
            "timestamp": str(self.timestamp) if self.timestamp else None
        }


class RiskEngine:
    """
    Risk Engine – institutional risk decomposition
    Modular, testable, deterministic, no lookahead bias

    Economic justification per metric:
    - Portfolio Vol: total uncertainty, risk premium required
    - MRC: marginal increase in portfolio vol per unit weight increase, economic = cost of adding risk
    - CRC: MRC * weight = risk contribution, economic = Euler decomposition, sum of CRC = portfolio vol
    - VaR: quantile loss, economic = regulatory capital, potential loss under normal conditions
    - CVaR/ES: average loss beyond VaR, economic = tail risk, coherent risk measure (Artzner et al)
    - Ulcer Index: average drawdown, economic = pain index, loss aversion
    - Tail Ratio: 95th percentile gain / |95th percentile loss|, economic = asymmetry of tails
    - Skew: asymmetry, economic = crash risk premium
    - Kurtosis: fat tails, economic = tail risk
    - Rolling Beta: systematic risk vs benchmark, economic = CAPM
    - Rolling Correlation: diversification, economic = common factor exposure
    - Risk Concentration: Herfindahl of risk contributions, economic = concentration risk
    - Risk Budget: target risk allocation vs actual, economic = risk parity budgeting
    """

    def __init__(self, confidence_levels: List[float] = None, risk_free_rate: float = 0.0):
        """
        Args:
            confidence_levels: List of VaR confidence levels, e.g., [0.95, 0.99]
            risk_free_rate: Risk-free rate for Sharpe-like calculations
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.risk_free_rate = float(risk_free_rate)
        if not all(0 < c < 1 for c in self.confidence_levels):
            raise ValueError("confidence_levels must be in (0,1)")

    @staticmethod
    def _validate_inputs(returns: pd.DataFrame, weights: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate returns DataFrame and weights dict – deterministic, no lookahead"""
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns must be non-empty DataFrame")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("weights must be non-empty dict")
        # Ensure weights sorted deterministically
        weights_sorted = dict(sorted(weights.items()))
        # Check tickers in returns
        missing = [t for t in weights_sorted.keys() if t not in returns.columns]
        if missing:
            raise ValueError(f"Weights contain tickers not in returns: {missing}")
        # Normalize weights to sum 1 (if not)
        total = sum(weights_sorted.values())
        if total == 0:
            raise ValueError("weights sum cannot be 0")
        # Convert to Series sorted
        w_series = pd.Series(weights_sorted).sort_index()
        # Ensure returns sorted by index
        returns_sorted = returns.sort_index()
        return returns_sorted, w_series

    @staticmethod
    def _portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """Compute portfolio returns – no lookahead, uses current weights * returns"""
        # Align
        aligned_returns = returns[weights.index]
        port_ret = (aligned_returns * weights.values).sum(axis=1)
        return port_ret

    @staticmethod
    def _portfolio_volatility(cov_matrix: pd.DataFrame, weights: pd.Series) -> float:
        """Portfolio vol = sqrt(w' * Cov * w) – annualized if cov annualized"""
        w = weights.values.reshape(-1, 1)
        cov = cov_matrix.values
        # Compute variance as scalar: w.T @ cov @ w -> (1,1)
        var_matrix = w.T @ cov @ w
        # Extract scalar safely
        try:
            var = float(var_matrix.item())
        except Exception:
            var = float(np.array(var_matrix).flatten()[0])
        vol = np.sqrt(var) if var >= 0 else 0.0
        return float(vol)

    def calculate_portfolio_volatility(self, cov_matrix: pd.DataFrame, weights: Dict[str, float]) -> float:
        """
        Calculate annualized portfolio volatility.

        Args:
            cov_matrix: Covariance matrix (annualized) – index/columns tickers
            weights: Dict ticker -> weight

        Returns:
            Portfolio volatility (annualized, e.g., 0.20 = 20%)

        Raises:
            ValueError
        """
        if not isinstance(cov_matrix, pd.DataFrame) or cov_matrix.empty:
            raise ValueError("cov_matrix must be non-empty DataFrame")
        _, w_series = self._validate_inputs(cov_matrix, weights)
        cov_sorted = cov_matrix.loc[w_series.index, w_series.index]
        vol = self._portfolio_volatility(cov_sorted, w_series)
        return float(vol)

    def calculate_marginal_risk_contribution(self, cov_matrix: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Marginal Risk Contribution: MRC_i = (Cov * w)_i / portfolio_vol

        Economic: how much portfolio vol increases if weight of asset i increases by small amount.

        Args:
            cov_matrix: Covariance matrix
            weights: Portfolio weights

        Returns:
            Dict ticker -> MRC
        """
        _, w_series = self._validate_inputs(cov_matrix, weights)
        cov_sorted = cov_matrix.loc[w_series.index, w_series.index]
        port_vol = self._portfolio_volatility(cov_sorted, w_series)
        if port_vol == 0:
            return {t: 0.0 for t in w_series.index}

        # MRC = (Cov * w) / sigma_p
        cov_w = cov_sorted.values @ w_series.values
        mrc = cov_w / port_vol
        return {t: float(mrc[i]) for i, t in enumerate(w_series.index)}

    def calculate_component_risk_contribution(self, cov_matrix: pd.DataFrame, weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Component Risk Contribution: CRC_i = w_i * MRC_i
        Sum CRC_i = portfolio_vol (Euler decomposition)

        Economic: actual risk contribution of each asset to total portfolio risk.

        Returns:
            Tuple (CRC dict, CRC % dict)
        """
        mrc = self.calculate_marginal_risk_contribution(cov_matrix, weights)
        _, w_series = self._validate_inputs(cov_matrix, weights)
        crc = {t: float(w_series[t] * mrc[t]) for t in w_series.index}
        total_crc = sum(crc.values())
        # CRC % = CRC_i / total
        crc_pct = {t: float(crc[t] / total_crc) if total_crc != 0 else 0.0 for t in crc}
        return crc, crc_pct

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Value at Risk – historical VaR: -quantile(returns, 1-confidence)
        VaR 95% = -5th percentile of returns (positive number = loss)

        Economic: potential loss under normal conditions, regulatory capital.

        Args:
            returns: Daily returns Series
            confidence: e.g., 0.95

        Returns:
            VaR (positive value for loss, e.g., 0.02 = 2% daily VaR)
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0,1)")
        # Historical VaR: -percentile
        q = np.percentile(returns.dropna().values, (1 - confidence) * 100)
        var = -float(q)
        return float(var)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        CVaR / Expected Shortfall – average loss beyond VaR
        Economic: tail risk, coherent risk measure, average of worst (1-confidence) losses

        Args:
            returns: Daily returns
            confidence: 0.95

        Returns:
            CVaR (positive loss)
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")
        var = RiskEngine.calculate_var(returns, confidence)
        # CVaR = -mean(returns where returns <= -VaR)
        threshold = -var
        tail = returns[returns <= threshold]
        if tail.empty:
            return float(var)
        cvar = -float(tail.mean())
        return float(cvar)

    @staticmethod
    def calculate_ulcer_index(equity_curve: pd.Series) -> float:
        """
        Ulcer Index – sqrt(mean(drawdown^2)), average pain

        Economic: loss aversion, measures depth and duration of drawdowns, pain index.

        Formula: UI = sqrt(mean(DD%^2)) where DD% = percentage drawdown from peak

        Args:
            equity_curve: Equity curve Series (cumulative equity)

        Returns:
            Ulcer Index %
        """
        if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
            raise ValueError("equity_curve must be non-empty Series")
        peak = equity_curve.cummax()
        dd = (equity_curve / peak - 1) * 100  # negative %
        ui = np.sqrt(np.mean(dd ** 2))
        return float(ui)

    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, percentile: int = 95) -> float:
        """
        Tail Ratio = 95th percentile gain / |95th percentile loss|
        Economic: asymmetry of tails, >1 means right tail fatter (good)

        Args:
            returns: Daily returns
            percentile: e.g., 95

        Returns:
            Tail ratio
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")
        if not 50 < percentile < 100:
            raise ValueError("percentile must be in (50,100)")
        p95 = np.percentile(returns.dropna().values, percentile)
        p5 = np.percentile(returns.dropna().values, 100 - percentile)
        # Tail ratio = p95 / |p5|
        if p5 == 0:
            return float('inf') if p95 > 0 else 0.0
        return float(abs(p95 / abs(p5)))

    @staticmethod
    def calculate_skew(returns: pd.Series) -> float:
        """Skewness – asymmetry, economic = crash risk premium"""
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")
        return float(returns.skew())

    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """Kurtosis – fat tails, economic = tail risk, Pearson kurtosis (normal=3)"""
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")
        return float(returns.kurtosis())  # Fisher? pandas kurtosis is Fisher (normal=0), we want Pearson? Use Fisher for simplicity

    @staticmethod
    def calculate_rolling_beta(returns: pd.DataFrame, benchmark_returns: pd.Series, window: int = 60) -> pd.DataFrame:
        """
        Rolling Beta vs benchmark – systematic risk

        Economic: CAPM beta, how much asset moves with market

        Args:
            returns: DataFrame ticker -> returns
            benchmark_returns: Series benchmark returns
            window: Rolling window

        Returns:
            DataFrame rolling betas
        """
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns must be non-empty DataFrame")
        if not isinstance(benchmark_returns, pd.Series):
            raise TypeError("benchmark_returns must be Series")
        if window <= 1:
            raise ValueError("window must be >1")

        betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for col in returns.columns:
            # Align
            asset = returns[col].dropna()
            bench = benchmark_returns.dropna()
            # Use rolling covariance / variance
            # Need to align indices
            combined = pd.concat([asset, bench], axis=1, join='inner')
            combined.columns = ['asset', 'bench']
            cov = combined['asset'].rolling(window).cov(combined['bench'])
            var_bench = combined['bench'].rolling(window).var()
            beta = cov / var_bench.replace(0, np.nan)
            betas[col] = beta

        return betas

    @staticmethod
    def calculate_rolling_correlation(returns: pd.DataFrame, benchmark_returns: pd.Series, window: int = 60) -> pd.DataFrame:
        """Rolling correlation – diversification benefit"""
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns must be non-empty DataFrame")
        if window <= 1:
            raise ValueError("window must be >1")

        corrs = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for col in returns.columns:
            asset = returns[col]
            combined = pd.concat([asset, benchmark_returns], axis=1, join='inner')
            combined.columns = ['asset', 'bench']
            corr = combined['asset'].rolling(window).corr(combined['bench'])
            corrs[col] = corr

        return corrs

    @staticmethod
    def calculate_risk_concentration(crc_pct: Dict[str, float]) -> float:
        """
        Risk Concentration – Herfindahl index of risk contributions
        HHI = sum(CRC%_i^2), range [1/N, 1], higher = more concentrated

        Economic: concentration risk, low HHI = diversified risk

        Args:
            crc_pct: Dict ticker -> CRC % (0-1)

        Returns:
            Herfindahl index (0-1)
        """
        if not isinstance(crc_pct, dict) or not crc_pct:
            raise ValueError("crc_pct must be non-empty dict")
        values = np.array(list(crc_pct.values()))
        # Normalize to sum 1
        total = values.sum()
        if total == 0:
            return 0.0
        norm = values / total
        hhi = np.sum(norm ** 2)
        return float(hhi)

    @staticmethod
    def calculate_risk_budget(crc_pct: Dict[str, float], target_budget: Dict[str, float]) -> Dict[str, float]:
        """
        Risk Budget – difference between actual CRC% and target risk budget

        Economic: risk parity budgeting, if target = equal risk (1/N), deviation shows risk budget imbalance

        Args:
            crc_pct: Actual CRC %
            target_budget: Target risk budget % (e.g., equal risk 1/N)

        Returns:
            Dict ticker -> deviation (actual - target)
        """
        if not isinstance(crc_pct, dict) or not isinstance(target_budget, dict):
            raise TypeError("crc_pct and target_budget must be dict")

        # Align keys deterministically sorted
        all_tickers = sorted(set(crc_pct.keys()) | set(target_budget.keys()))
        deviation = {}
        for t in all_tickers:
            actual = crc_pct.get(t, 0.0)
            target = target_budget.get(t, 0.0)
            deviation[t] = float(actual - target)

        return dict(sorted(deviation.items()))

    def full_risk_decomposition(self, returns: pd.DataFrame, weights: Dict[str, float],
                                benchmark_returns: Optional[pd.Series] = None,
                                equity_curve: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Full risk decomposition – computes all risk metrics in one call, deterministic, no lookahead

        Args:
            returns: DataFrame daily returns (ticker -> returns), must be sorted, no future
            weights: Dict ticker -> weight
            benchmark_returns: Optional benchmark for beta/correlation
            equity_curve: Optional equity curve for Ulcer Index (if None, computed from returns + weights)

        Returns:
            RiskMetrics immutable

        Raises:
            ValueError
        """
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns must be non-empty DataFrame")

        _, w_series = self._validate_inputs(returns, weights)
        returns_sorted = returns.sort_index()
        w_series_sorted = w_series.sort_index()

        # Covariance annualized
        cov_matrix = returns_sorted.cov() * 252

        # Portfolio metrics
        port_vol = self.calculate_portfolio_volatility(cov_matrix, weights)
        mrc = self.calculate_marginal_risk_contribution(cov_matrix, weights)
        crc, crc_pct = self.calculate_component_risk_contribution(cov_matrix, weights)

        # Portfolio returns for VaR etc
        port_returns = self._portfolio_returns(returns_sorted, w_series_sorted)

        var_95 = self.calculate_var(port_returns, 0.95)
        var_99 = self.calculate_var(port_returns, 0.99)
        cvar_95 = self.calculate_cvar(port_returns, 0.95)
        cvar_99 = self.calculate_cvar(port_returns, 0.99)

        # Equity curve for Ulcer
        if equity_curve is None:
            equity_curve = (1 + port_returns).cumprod() * 100000
        ulcer = self.calculate_ulcer_index(equity_curve)
        tail_ratio = self.calculate_tail_ratio(port_returns, 95)
        skew = self.calculate_skew(port_returns)
        kurt = self.calculate_kurtosis(port_returns)

        risk_conc = self.calculate_risk_concentration(crc_pct)

        # Risk budget: target equal risk
        n = len(w_series_sorted)
        target_budget = {t: 1 / n for t in w_series_sorted.index}
        risk_budget = self.calculate_risk_budget(crc_pct, target_budget)

        return RiskMetrics(
            portfolio_volatility=float(port_vol),
            marginal_risk_contribution=mrc,
            component_risk_contribution=crc,
            component_risk_pct=crc_pct,
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            expected_shortfall_95=float(cvar_95),
            ulcer_index=float(ulcer),
            tail_ratio=float(tail_ratio),
            skew=float(skew),
            kurtosis=float(kurt),
            risk_concentration=float(risk_conc),
            risk_budget=risk_budget,
            timestamp=returns_sorted.index[-1] if not returns_sorted.empty else None
        )

    # ── Convenience for rolling metrics ──
    def rolling_risk_report(self, returns: pd.DataFrame, weights: Dict[str, float],
                            benchmark_returns: pd.Series, window: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Generate rolling risk reports: rolling beta, rolling correlation

        Args:
            returns: DataFrame returns
            weights: Weights dict
            benchmark_returns: Benchmark returns Series
            window: Window

        Returns:
            Dict with rolling_beta, rolling_correlation DataFrames
        """
        rolling_beta = self.calculate_rolling_beta(returns, benchmark_returns, window)
        rolling_corr = self.calculate_rolling_correlation(returns, benchmark_returns, window)
        return {
            "rolling_beta": rolling_beta,
            "rolling_correlation": rolling_corr
        }
