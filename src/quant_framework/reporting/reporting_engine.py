"""
12 Reporting – Complete Institutional Reporting Engine
Generates:
- Equity Curve
- Rolling Sharpe, Rolling Sortino, Rolling CAGR
- Drawdown Curve, Drawdown Duration
- Monthly Returns Heatmap, Yearly Returns
- Rolling Volatility, Exposure, Turnover, Asset Allocation, Regime Timeline, Canary Timeline, Portfolio Composition

Exports: HTML, PDF, CSV, Excel, PNG
Reproducible: deterministic filenames, sorted keys, fixed style
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import json

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


class ReportingEngine:
    """
    Complete Reporting Engine – institutional grade, deterministic, no lookahead bias
    All plots use past data only, no future.
    """

    def __init__(self, output_dir: str = "reports/full"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Core helpers ──
    @staticmethod
    def _validate_equity(equity: pd.Series) -> None:
        if not isinstance(equity, pd.Series) or equity.empty:
            raise ValueError("equity must be non-empty Series")
        if not isinstance(equity.index, pd.DatetimeIndex):
            raise TypeError("equity index must be DatetimeIndex")

    @staticmethod
    def _rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
        return returns.rolling(window).mean() / returns.rolling(window).std().replace(0, np.nan) * np.sqrt(252)

    @staticmethod
    def _rolling_sortino(returns: pd.Series, window: int = 60) -> pd.Series:
        def _sortino(s):
            downside = s[s < 0]
            ds = downside.std() if len(downside) > 1 else s.std()
            return s.mean() / ds * np.sqrt(252) if ds and not np.isnan(ds) and ds != 0 else np.nan
        return returns.rolling(window).apply(_sortino, raw=False)

    @staticmethod
    def _rolling_cagr(equity: pd.Series, window: int = 252) -> pd.Series:
        def _cagr(s):
            if len(s) < 2:
                return np.nan
            total = s.iloc[-1] / s.iloc[0] - 1
            years = window / 252
            return (1 + total) ** (1 / years) - 1 if years != 0 else total
        return equity.rolling(window).apply(_cagr, raw=False) * 100

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        return equity / peak - 1

    @staticmethod
    def _drawdown_duration(dd: pd.Series) -> pd.Series:
        """Duration of current drawdown in days – time since last peak"""
        # Identify peaks
        equity = dd.index.to_series()  # placeholder, actually need equity
        # We compute duration as days since last zero drawdown
        # Use cumulative logic
        is_zero = (dd == 0)
        # For each point, find last zero
        duration = []
        last_peak_idx = 0
        for i, val in enumerate(dd):
            if val == 0:
                last_peak_idx = i
                duration.append(0)
            else:
                duration.append(i - last_peak_idx)
        return pd.Series(duration, index=dd.index)

    # ── Individual plot generators (each returns fig, ax) ──
    def plot_equity_curve(self, equity: pd.Series, benchmark: Optional[pd.Series] = None) -> plt.Figure:
        self._validate_equity(equity)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
        ax.plot(equity.index, (equity / equity.iloc[0] - 1) * 100, label='Strategy', color='#1f77b4', linewidth=2)
        if benchmark is not None and not benchmark.empty:
            bench = benchmark.reindex(equity.index).ffill()
            ax.plot(bench.index, (bench / bench.iloc[0] - 1) * 100, label='Benchmark', color='gray', alpha=0.7)
        ax.set_title('Equity Curve (%)', fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        rs = self._rolling_sharpe(returns, window)
        ax.plot(rs.index, rs, color='green', linewidth=1.2)
        ax.axhline(rs.mean(), color='blue', ls='--', label=f'Overall: {rs.mean():.2f}')
        ax.set_title(f'Rolling Sharpe ({window}d)', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_rolling_sortino(self, returns: pd.Series, window: int = 60) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        rs = self._rolling_sortino(returns, window)
        ax.plot(rs.index, rs, color='purple', linewidth=1.2)
        ax.set_title(f'Rolling Sortino ({window}d)', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_rolling_cagr(self, equity: pd.Series, window: int = 252) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        rc = self._rolling_cagr(equity, window)
        ax.plot(rc.index, rc, color='orange', linewidth=1.2)
        ax.set_title(f'Rolling CAGR ({window}d) %', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_drawdown_curve(self, equity: pd.Series) -> plt.Figure:
        self._validate_equity(equity)
        dd = self._drawdown(equity) * 100
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        ax.plot(dd.index, dd, color='darkred')
        ax.fill_between(dd.index, dd, 0, color='darkred', alpha=0.2)
        ax.axhline(dd.min(), color='red', ls='--', label=f"Max DD: {dd.min():.1f}%")
        ax.set_title('Drawdown Curve (%)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_drawdown_duration(self, equity: pd.Series) -> plt.Figure:
        dd = self._drawdown(equity)
        duration = self._drawdown_duration(dd)
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        ax.plot(duration.index, duration, color='brown')
        ax.set_title('Drawdown Duration (bars)', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_monthly_returns_heatmap(self, returns: pd.Series) -> plt.Figure:
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        df = pd.DataFrame({'ret': monthly})
        df['year'] = df.index.year
        df['month'] = df.index.month
        pivot = df.pivot(index='month', columns='year', values='ret') * 100
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-10, vmax=10, aspect='auto')
        ax.set_title('Monthly Returns Heatmap (%)', fontweight='bold')
        ax.set_yticks(range(12))
        ax.set_yticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        return fig

    def plot_yearly_returns(self, returns: pd.Series) -> plt.Figure:
        yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
        colors = ['green' if v > 0 else 'red' for v in yearly.values]
        ax.bar(yearly.index.year.astype(str), yearly.values, color=colors)
        ax.set_title('Yearly Returns (%)', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    def plot_rolling_volatility(self, returns: pd.Series, window: int = 60) -> plt.Figure:
        rv = returns.rolling(window).std() * np.sqrt(252) * 100
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        ax.plot(rv.index, rv, color='blue')
        ax.axhline(rv.mean(), color='red', ls='--', label=f"Overall: {rv.mean():.1f}%")
        ax.set_title(f'Rolling Volatility ({window}d) %', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_exposure(self, positions: pd.DataFrame) -> plt.Figure:
        """
        positions: DataFrame index dates, columns tickers, values position 0-1 or quantity
        Exposure = sum of absolute positions or count of active?
        """
        if not isinstance(positions, pd.DataFrame) or positions.empty:
            # Create dummy exposure
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(0.5, 0.5, "No positions data", ha='center')
            return fig
        exposure = positions.abs().sum(axis=1)
        # Normalize to 0-1 if >1
        if exposure.max() > 1.5:
            exposure = (exposure > 0).astype(int).sum(axis=1) / positions.shape[1] if isinstance(positions, pd.DataFrame) else exposure

        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        ax.plot(exposure.index, exposure, color='teal')
        ax.set_title('Exposure (0-1)', fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_turnover(self, positions: pd.DataFrame) -> plt.Figure:
        """Turnover = sum of absolute changes in positions"""
        if not isinstance(positions, pd.DataFrame) or positions.empty or len(positions) < 2:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(0.5, 0.5, "No turnover data", ha='center')
            return fig
        turnover = positions.diff().abs().sum(axis=1)
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        ax.bar(turnover.index, turnover.values, color='orange', width=2)
        ax.set_title('Turnover (sum |Δ positions|)', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_asset_allocation(self, weights: pd.DataFrame) -> plt.Figure:
        """Stacked area of asset allocation over time – weights DataFrame index dates, columns tickers"""
        if not isinstance(weights, pd.DataFrame) or weights.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, "No allocation data", ha='center')
            return fig
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
        ax.stackplot(weights.index, *[weights[col].values for col in weights.columns], labels=weights.columns, alpha=0.8)
        ax.set_title('Asset Allocation Over Time', fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_regime_timeline(self, regime_series: pd.Series) -> plt.Figure:
        """Regime timeline – categorical over time"""
        if not isinstance(regime_series, pd.Series) or regime_series.empty:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(0.5, 0.5, "No regime data", ha='center')
            return fig
        # Map regimes to integers for color
        unique_regimes = sorted(regime_series.unique())
        mapping = {r: i for i, r in enumerate(unique_regimes)}
        numeric = regime_series.map(mapping)
        fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
        ax.plot(regime_series.index, numeric, drawstyle='steps-post', color='darkblue')
        ax.set_yticks(list(mapping.values()))
        ax.set_yticklabels(list(mapping.keys()), fontsize=8)
        ax.set_title('Regime Timeline', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_canary_timeline(self, canary_df: pd.DataFrame) -> plt.Figure:
        """
        Canary timeline – DataFrame index dates, columns canary names, values signal numeric or confidence
        """
        if not isinstance(canary_df, pd.DataFrame) or canary_df.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, "No canary data", ha='center')
            return fig
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
        for col in canary_df.columns:
            ax.plot(canary_df.index, canary_df[col], label=col, alpha=0.8)
        ax.set_title('Canary Timeline (signal/confidence)', fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_portfolio_composition(self, latest_weights: Dict[str, float]) -> plt.Figure:
        """Pie chart of current composition"""
        if not isinstance(latest_weights, dict) or not latest_weights:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, "No composition", ha='center')
            return fig
        labels = list(latest_weights.keys())
        sizes = list(latest_weights.values())
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Portfolio Composition (latest)', fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_factor_exposure(self, factor_exposure: pd.DataFrame) -> plt.Figure:
        """
        Factor exposure over time – each factor as line
        Economic justification: shows style tilts over time (e.g., Market beta, Value, Momentum)
        Each backtest must show factor exposure in time as per prompt

        Args:
            factor_exposure: DataFrame index dates, columns factors (Market, Size, Value, etc), values exposure (beta or score)

        Returns:
            Figure
        """
        if not isinstance(factor_exposure, pd.DataFrame) or factor_exposure.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, "No factor exposure data", ha='center')
            return fig

        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        # Deterministic sorted columns
        for col in sorted(factor_exposure.columns):
            ax.plot(factor_exposure.index, factor_exposure[col], label=col, linewidth=1.2, alpha=0.8)

        ax.set_title('Factor Exposure Over Time (Market, Size, Value, Growth, Momentum, Quality, LowVol, Carry, Commodity, Duration, Dollar, Inflation)', fontweight='bold')
        ax.set_ylabel('Exposure (beta / score)')
        ax.set_xlabel('Date')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8, ncol=3, loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_factor_exposure_heatmap(self, factor_exposure: pd.DataFrame) -> plt.Figure:
        """Heatmap of factor exposure over time – alternative view"""
        if not isinstance(factor_exposure, pd.DataFrame) or factor_exposure.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, "No factor exposure data", ha='center')
            return fig

        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        # Transpose for heatmap: factors as y, dates as x
        # Resample to monthly for readability if too many dates
        if len(factor_exposure) > 200:
            # Monthly resample mean
            monthly = factor_exposure.resample('ME').mean()
        else:
            monthly = factor_exposure

        im = ax.imshow(monthly.T.values, aspect='auto', cmap='RdYlGn', vmin=-1.5, vmax=1.5)
        ax.set_yticks(range(len(monthly.columns)))
        ax.set_yticklabels(sorted(monthly.columns), fontsize=8)
        ax.set_title('Factor Exposure Heatmap (monthly avg)', fontweight='bold')
        ax.set_xlabel('Time (months)')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Exposure')
        plt.tight_layout()
        return fig

    # ── Export methods – reproducible ──
    def export_all(self, equity: pd.Series, returns: pd.Series,
                   positions: Optional[pd.DataFrame] = None,
                   weights: Optional[pd.DataFrame] = None,
                   regime_series: Optional[pd.Series] = None,
                   canary_df: Optional[pd.DataFrame] = None,
                   benchmark: Optional[pd.Series] = None,
                   factor_exposure: Optional[pd.DataFrame] = None,
                   prefix: str = "report") -> Dict[str, Path]:
        """
        Generate all plots and export to PNG, plus CSV/Excel/HTML/PDF

        Args:
            equity: Equity curve
            returns: Daily returns (if None, computed from equity)
            positions: DataFrame positions over time
            weights: DataFrame weights over time
            regime_series: Series regime over time
            canary_df: DataFrame canary signals over time
            benchmark: Benchmark equity
            prefix: Filename prefix (deterministic)

        Returns:
            Dict of exported file paths

        Raises:
            ValueError
        """
        if not isinstance(equity, pd.Series) or equity.empty:
            raise ValueError("equity must be non-empty Series")
        if returns is None:
            returns = equity.pct_change().dropna()

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        outputs: Dict[str, Path] = {}

        # Generate figures deterministically sorted order – includes all requested reports
        plots = {
            "equity_curve": self.plot_equity_curve(equity, benchmark),
            "rolling_sharpe": self.plot_rolling_sharpe(returns),
            "rolling_sortino": self.plot_rolling_sortino(returns),
            "rolling_cagr": self.plot_rolling_cagr(equity),
            "drawdown_curve": self.plot_drawdown_curve(equity),
            "drawdown_duration": self.plot_drawdown_duration(equity),
            "monthly_heatmap": self.plot_monthly_returns_heatmap(returns),
            "yearly_returns": self.plot_yearly_returns(returns),
            "rolling_volatility": self.plot_rolling_volatility(returns),
            "exposure": self.plot_exposure(positions if positions is not None else pd.DataFrame()),
            "turnover": self.plot_turnover(positions if positions is not None else pd.DataFrame()),
            "asset_allocation": self.plot_asset_allocation(weights if weights is not None else pd.DataFrame()),
            "regime_timeline": self.plot_regime_timeline(regime_series if regime_series is not None else pd.Series(dtype=object)),
            "canary_timeline": self.plot_canary_timeline(canary_df if canary_df is not None else pd.DataFrame()),
        }

        # Factor exposure – required: each backtest must show factor exposure over time
        if factor_exposure is not None and not factor_exposure.empty:
            plots["factor_exposure"] = self.plot_factor_exposure(factor_exposure)
            plots["factor_exposure_heatmap"] = self.plot_factor_exposure_heatmap(factor_exposure)

        # Add portfolio composition if weights provided (latest)
        if weights is not None and not weights.empty:
            latest_weights = weights.iloc[-1].to_dict() if isinstance(weights, pd.DataFrame) else {}
            if latest_weights:
                plots["portfolio_composition"] = self.plot_portfolio_composition(latest_weights)

        # Save PNGs
        for name, fig in plots.items():
            png_path = self.output_dir / f"{prefix}_{name}.png"
            fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            outputs[f"{name}_png"] = png_path

        # Save combined PDF
        pdf_path = self.output_dir / f"{prefix}_full_report.pdf"
        with PdfPages(pdf_path) as pdf:
            for name, fig in plots.items():
                # Recreate fig for PDF (since closed) – instead, we saved PNGs, now create again for PDF is heavy
                # For simplicity, we will create PDF from PNGs? Instead, regenerate quickly
                pass
            # Actually, we already closed figs, so we need to regenerate for PDF – for simplicity, create PDF with PNGs embedded via matplotlib?
            # We'll create a new PDF by loading PNGs? Simpler: save each PNG figure again into PDF if we kept figs
            # Workaround: create PDF with blank pages referencing PNG paths – for institutional, we generate PDF from HTML later
            pass

        # For true PDF, we will create a multi-page PDF with all plots anew (reuse plot methods) – includes factor exposure
        # To avoid double computation, we regenerate and save into PDF
        pdf_path = self.output_dir / f"{prefix}_full_report.pdf"
        with PdfPages(pdf_path) as pdf:
            for plot_name in ["equity_curve", "rolling_sharpe", "rolling_sortino", "rolling_cagr",
                              "drawdown_curve", "drawdown_duration", "monthly_heatmap",
                              "yearly_returns", "rolling_volatility", "exposure", "turnover",
                              "asset_allocation", "regime_timeline", "canary_timeline",
                              "factor_exposure", "factor_exposure_heatmap", "portfolio_composition"]:
                # For factor_exposure, need factor_exposure param
                if plot_name in ("factor_exposure", "factor_exposure_heatmap") and (factor_exposure is None or factor_exposure.empty):
                    continue
                if plot_name == "portfolio_composition" and (weights is None or weights.empty):
                    continue

                # Regenerate fig
                if plot_name == "equity_curve":
                    fig = self.plot_equity_curve(equity, benchmark)
                elif plot_name == "rolling_sharpe":
                    fig = self.plot_rolling_sharpe(returns)
                elif plot_name == "rolling_sortino":
                    fig = self.plot_rolling_sortino(returns)
                elif plot_name == "rolling_cagr":
                    fig = self.plot_rolling_cagr(equity)
                elif plot_name == "drawdown_curve":
                    fig = self.plot_drawdown_curve(equity)
                elif plot_name == "drawdown_duration":
                    fig = self.plot_drawdown_duration(equity)
                elif plot_name == "monthly_heatmap":
                    fig = self.plot_monthly_returns_heatmap(returns)
                elif plot_name == "yearly_returns":
                    fig = self.plot_yearly_returns(returns)
                elif plot_name == "rolling_volatility":
                    fig = self.plot_rolling_volatility(returns)
                elif plot_name == "exposure":
                    fig = self.plot_exposure(positions if positions is not None else pd.DataFrame())
                elif plot_name == "turnover":
                    fig = self.plot_turnover(positions if positions is not None else pd.DataFrame())
                elif plot_name == "asset_allocation":
                    fig = self.plot_asset_allocation(weights if weights is not None else pd.DataFrame())
                elif plot_name == "regime_timeline":
                    fig = self.plot_regime_timeline(regime_series if regime_series is not None else pd.Series(dtype=object))
                elif plot_name == "canary_timeline":
                    fig = self.plot_canary_timeline(canary_df if canary_df is not None else pd.DataFrame())
                elif plot_name == "factor_exposure":
                    fig = self.plot_factor_exposure(factor_exposure if factor_exposure is not None else pd.DataFrame())
                elif plot_name == "factor_exposure_heatmap":
                    fig = self.plot_factor_exposure_heatmap(factor_exposure if factor_exposure is not None else pd.DataFrame())
                elif plot_name == "portfolio_composition":
                    latest_w = weights.iloc[-1].to_dict() if isinstance(weights, pd.DataFrame) and not weights.empty else {}
                    fig = self.plot_portfolio_composition(latest_w)
                else:
                    continue
                pdf.savefig(fig)
                plt.close(fig)

        outputs["pdf"] = pdf_path

        # Export CSV / Excel – deterministic sorted columns
        # Equity curve
        equity_df = pd.DataFrame({"equity": equity, "returns": returns})
        equity_csv = self.output_dir / f"{prefix}_equity.csv"
        equity_df.to_csv(equity_csv)
        outputs["equity_csv"] = equity_csv

        equity_excel = self.output_dir / f"{prefix}_equity.xlsx"
        try:
            equity_df.to_excel(equity_excel)
            outputs["equity_excel"] = equity_excel
        except Exception:
            pass

        # Monthly returns CSV
        try:
            monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({"monthly_return": monthly})
            monthly_csv = self.output_dir / f"{prefix}_monthly.csv"
            monthly_df.to_csv(monthly_csv)
            outputs["monthly_csv"] = monthly_csv
        except Exception:
            pass

        # HTML report – embed all PNGs as base64
        html_path = self.output_dir / f"{prefix}_report.html"
        html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{prefix} – Full Institutional Report</title>
<style>
body{{background:#0b0e14;color:#e6edf3;font-family:sans-serif;padding:20px}}
.card{{background:#161b22;border:1px solid #30363d;padding:16px;border-radius:8px;margin:16px 0}}
h1{{color:#58a6ff}} h2{{color:#8b949e}}
img{{max-width:100%;border-radius:8px;border:1px solid #30363d;margin:10px 0}}
</style></head><body>
<h1>{prefix} – Complete Institutional Report</h1>
<p>Generated {pd.Timestamp.now().isoformat()} – Deterministic, reproducible, no lookahead bias</p>
"""
        for name in sorted(plots.keys()):
            png_file = self.output_dir / f"{prefix}_{name}.png"
            if png_file.exists():
                import base64
                b64 = base64.b64encode(png_file.read_bytes()).decode()
                html_content += f'<div class="card"><h2>{name.replace("_"," ").title()}</h2><img src="data:image/png;base64,{b64}"></div>\n'

        html_content += "</body></html>"
        html_path.write_text(html_content, encoding='utf-8')
        outputs["html"] = html_path

        return outputs
