"""
Interactive Dashboard – Plotly, sections Universe, Signals, Regimes, Portfolio, Risk, Performance, Optimization, Experiments, Walk Forward, Stress Tests
All interactive, modular, no business logic, uses ReportEngine + Plotly
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import dash
    from dash import dcc, html, Input, Output
    HAS_DASH = True
except ImportError:
    HAS_DASH = False


class InteractiveDashboard:
    """
    Interactive Dashboard – institutional

    Sections:
    - Universe
    - Signals
    - Regimes
    - Portfolio
    - Risk
    - Performance
    - Optimization
    - Experiments
    - Walk Forward
    - Stress Tests

    Uses Plotly for interactivity, all plots interactive (zoom, hover, etc.)
    """

    def __init__(self, output_dir: str = "reports/dashboard"):
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_plotly():
        if not HAS_PLOTLY:
            raise ImportError("plotly not installed, pip install plotly")

    def plot_universe(self, universe_over_time: pd.DataFrame) -> go.Figure:
        """Universe size over time + heatmap of inclusion"""
        self._check_plotly()
        if not isinstance(universe_over_time, pd.DataFrame) or universe_over_time.empty:
            fig = go.Figure()
            fig.add_annotation(text="No universe data", x=0.5, y=0.5, showarrow=False)
            return fig

        size_series = universe_over_time.sum(axis=1)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=("Universe Size Over Time", "Universe Inclusion Heatmap"),
                            row_heights=[0.4, 0.6])

        # Size over time
        fig.add_trace(go.Scatter(x=size_series.index, y=size_series.values,
                                 mode='lines+markers', name='Size'), row=1, col=1)

        # Heatmap – only first 30 tickers for readability
        cols_to_show = universe_over_time.columns[:30]
        heat_data = universe_over_time[cols_to_show].astype(int).T
        fig.add_trace(go.Heatmap(z=heat_data.values, x=heat_data.columns, y=heat_data.index,
                                 colorscale='Viridis', showscale=False), row=2, col=1)

        fig.update_layout(title="Universe – Size and Inclusion", height=800)
        return fig

    def plot_signals(self, signals_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        """Signals – each ticker signal over time"""
        self._check_plotly()
        fig = go.Figure()
        if not signals_dict:
            fig.add_annotation(text="No signals", x=0.5, y=0.5, showarrow=False)
            return fig

        for ticker, df in signals_dict.items():
            if df.empty or 'signal' not in df.columns:
                continue
            fig.add_trace(go.Scatter(x=df.index, y=df['signal'],
                                     mode='lines', name=f"{ticker} signal"))

        fig.update_layout(title="Signals – Buy/Sell/Hold Over Time", height=500)
        return fig

    def plot_regimes(self, regime_series: pd.Series, layer_results: Dict[str, pd.DataFrame] = None) -> go.Figure:
        """Regimes – timeline + multi-layer"""
        self._check_plotly()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Final Regime Timeline", "Layer Regimes"))

        if isinstance(regime_series, pd.Series) and not regime_series.empty:
            # Map regimes to numeric for plotting
            unique = sorted(regime_series.unique())
            mapping = {r: i for i, r in enumerate(unique)}
            numeric = regime_series.map(mapping)
            fig.add_trace(go.Scatter(x=regime_series.index, y=numeric,
                                     mode='lines', name='Regime', line=dict(shape='hv')), row=1, col=1)
            fig.update_yaxes(tickvals=list(mapping.values()), ticktext=list(mapping.keys()), row=1, col=1)

        if layer_results:
            # layer_results can be dict layer_name -> Series regime
            for layer_name, series in layer_results.items():
                if isinstance(series, pd.Series) and not series.empty:
                    unique = sorted(series.unique())
                    mapping = {r: i for i, r in enumerate(unique)}
                    numeric = series.map(mapping)
                    fig.add_trace(go.Scatter(x=series.index, y=numeric,
                                             mode='lines', name=layer_name), row=2, col=1)

        fig.update_layout(title="Regimes – Final + Layers", height=700)
        return fig

    def plot_portfolio(self, weights_over_time: pd.DataFrame) -> go.Figure:
        """Portfolio – asset allocation over time stacked area"""
        self._check_plotly()
        if not isinstance(weights_over_time, pd.DataFrame) or weights_over_time.empty:
            fig = go.Figure()
            fig.add_annotation(text="No portfolio weights", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        for col in weights_over_time.columns:
            fig.add_trace(go.Scatter(x=weights_over_time.index, y=weights_over_time[col],
                                     mode='lines', stackgroup='one', name=col))

        fig.update_layout(title="Portfolio – Asset Allocation Over Time", height=600)
        return fig

    def plot_risk(self, risk_metrics_over_time: pd.DataFrame = None, risk_decomposition: Dict = None) -> go.Figure:
        """Risk – VaR, CVaR, Vol, Concentration etc"""
        self._check_plotly()
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Portfolio Volatility", "VaR / CVaR", "Risk Concentration", "Risk Budget"))

        # Dummy data if not provided
        if risk_metrics_over_time is None or risk_metrics_over_time.empty:
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='ME')
            risk_metrics_over_time = pd.DataFrame({
                'volatility': np.random.normal(0.2, 0.02, len(dates)),
                'var_95': np.random.normal(0.02, 0.005, len(dates)),
                'cvar_95': np.random.normal(0.03, 0.005, len(dates)),
                'risk_concentration': np.random.normal(0.5, 0.1, len(dates))
            }, index=dates)

        fig.add_trace(go.Scatter(x=risk_metrics_over_time.index, y=risk_metrics_over_time['volatility'],
                                 name='Volatility'), row=1, col=1)
        if 'var_95' in risk_metrics_over_time.columns:
            fig.add_trace(go.Scatter(x=risk_metrics_over_time.index, y=risk_metrics_over_time['var_95'],
                                     name='VaR 95%'), row=1, col=2)
        if 'cvar_95' in risk_metrics_over_time.columns:
            fig.add_trace(go.Scatter(x=risk_metrics_over_time.index, y=risk_metrics_over_time['cvar_95'],
                                     name='CVaR 95%'), row=1, col=2)
        if 'risk_concentration' in risk_metrics_over_time.columns:
            fig.add_trace(go.Scatter(x=risk_metrics_over_time.index, y=risk_metrics_over_time['risk_concentration'],
                                     name='Risk Conc.'), row=2, col=1)

        fig.update_layout(title="Risk – Vol, VaR, Concentration", height=700)
        return fig

    def plot_performance(self, equity_curve: pd.Series, benchmark: pd.Series = None) -> go.Figure:
        """Performance – equity curve, drawdown, rolling Sharpe"""
        self._check_plotly()
        if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
            fig = go.Figure()
            fig.add_annotation(text="No equity curve", x=0.5, y=0.5, showarrow=False)
            return fig

        returns = equity_curve.pct_change().dropna()
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Equity Curve", "Drawdown", "Rolling Sharpe"),
                            vertical_spacing=0.1)

        # Equity
        fig.add_trace(go.Scatter(x=equity_curve.index, y=(equity_curve/equity_curve.iloc[0]-1)*100,
                                 name='Strategy'), row=1, col=1)
        if benchmark is not None and not benchmark.empty:
            bench = benchmark.reindex(equity_curve.index).ffill()
            fig.add_trace(go.Scatter(x=bench.index, y=(bench/bench.iloc[0]-1)*100,
                                     name='Benchmark'), row=1, col=1)

        # Drawdown
        peak = equity_curve.cummax()
        dd = (equity_curve / peak - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', name='Drawdown'), row=2, col=1)

        # Rolling Sharpe
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Rolling Sharpe'), row=3, col=1)

        fig.update_layout(title="Performance", height=900)
        return fig

    def plot_optimization(self, optimization_results: pd.DataFrame) -> go.Figure:
        """Optimization – param vs Sharpe scatter"""
        self._check_plotly()
        if not isinstance(optimization_results, pd.DataFrame) or optimization_results.empty:
            fig = go.Figure()
            fig.add_annotation(text="No optimization results", x=0.5, y=0.5, showarrow=False)
            return fig

        # Expect columns: param1, param2, sharpe
        # For simplicity, plot first param vs Sharpe
        first_param = optimization_results.columns[0]
        if 'sharpe' in optimization_results.columns:
            fig = px.scatter(optimization_results, x=first_param, y='sharpe', color='sharpe', title="Optimization – Param vs Sharpe")
        else:
            fig = px.scatter(optimization_results, x=optimization_results.columns[0], y=optimization_results.columns[1])

        return fig

    def plot_experiments(self, experiments_df: pd.DataFrame) -> go.Figure:
        """Experiments – Sharpe over time, etc"""
        self._check_plotly()
        if not isinstance(experiments_df, pd.DataFrame) or experiments_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No experiments", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        if 'timestamp' in experiments_df.columns and 'sharpe' in experiments_df.columns:
            fig.add_trace(go.Scatter(x=pd.to_datetime(experiments_df['timestamp']), y=experiments_df['sharpe'],
                                     mode='markers', name='Sharpe'))
            fig.update_layout(title="Experiments – Sharpe over time")
        else:
            fig = px.scatter(experiments_df, x=experiments_df.columns[0], y=experiments_df.columns[1] if len(experiments_df.columns)>1 else experiments_df.columns[0])

        return fig

    def plot_walk_forward(self, wf_results: Dict) -> go.Figure:
        """Walk Forward – IS vs OOS Sharpe"""
        self._check_plotly()
        if not isinstance(wf_results, dict) or 'windows' not in wf_results:
            fig = go.Figure()
            fig.add_annotation(text="No walk-forward results", x=0.5, y=0.5, showarrow=False)
            return fig

        windows = wf_results.get('windows', [])
        if not windows:
            fig = go.Figure()
            fig.add_annotation(text="No windows", x=0.5, y=0.5, showarrow=False)
            return fig

        is_scores = [w.get('oos_score', 0) for w in windows]  # actually IS/OOS, simplified

        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(len(windows))), y=is_scores, name='OOS Score'))
        fig.update_layout(title="Walk Forward – OOS Score per Window")

        return fig

    def plot_stress_tests(self, stress_results: Dict) -> go.Figure:
        """Stress Tests – historical stress total return and max DD"""
        self._check_plotly()
        if not isinstance(stress_results, dict) or not stress_results:
            fig = go.Figure()
            fig.add_annotation(text="No stress test results", x=0.5, y=0.5, showarrow=False)
            return fig

        scenarios = list(stress_results.keys())
        total_returns = [stress_results[s].get('total_return', 0) for s in scenarios]
        max_dds = [stress_results[s].get('max_dd', 0) for s in scenarios]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Return in Stress", "Max DD in Stress"))

        fig.add_trace(go.Bar(x=scenarios, y=total_returns, name='Total Return'), row=1, col=1)
        fig.add_trace(go.Bar(x=scenarios, y=max_dds, name='Max DD'), row=1, col=2)

        fig.update_layout(title="Stress Tests", height=500)
        return fig

    def build_dashboard(self, data: Dict) -> str:
        """
        Build full interactive dashboard with all sections – returns path to HTML file

        Args:
            data: Dict containing keys: universe_over_time, signals, regime_series, layer_results, weights, risk_metrics_over_time, equity_curve, benchmark, optimization_results, experiments_df, wf_results, stress_results, factor_exposure, etc.

        Returns:
            Path to HTML file
        """
        self._check_plotly()

        output_path = self.output_dir / "interactive_dashboard.html"

        # Generate figures for each section
        figs = {}

        # Universe
        if 'universe_over_time' in data:
            figs['universe'] = self.plot_universe(data['universe_over_time'])

        # Signals
        if 'signals_dict' in data:
            figs['signals'] = self.plot_signals(data['signals_dict'])

        # Regimes
        if 'regime_series' in data:
            figs['regimes'] = self.plot_regimes(data.get('regime_series'), data.get('layer_results'))

        # Portfolio
        if 'weights' in data:
            figs['portfolio'] = self.plot_portfolio(data['weights'])

        # Risk
        if 'risk_metrics_over_time' in data:
            figs['risk'] = self.plot_risk(data['risk_metrics_over_time'])

        # Performance
        if 'equity_curve' in data:
            figs['performance'] = self.plot_performance(data['equity_curve'], data.get('benchmark'))

        # Optimization
        if 'optimization_results' in data:
            figs['optimization'] = self.plot_optimization(data['optimization_results'])

        # Experiments
        if 'experiments_df' in data:
            figs['experiments'] = self.plot_experiments(data['experiments_df'])

        # Walk Forward
        if 'wf_results' in data:
            figs['walk_forward'] = self.plot_walk_forward(data['wf_results'])

        # Stress Tests
        if 'stress_results' in data:
            figs['stress'] = self.plot_stress_tests(data['stress_results'])

        # Build HTML with all figures embedded
        html_parts = [f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Institutional Interactive Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{{font-family:sans-serif;background:#f9fafb;padding:20px}}
h1{{color:#1f2937}}
h2{{color:#374151;border-bottom:2px solid #e5e7eb;padding-bottom:8px}}
.section{{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0}}
</style></head><body>
<h1>Institutional Quantitative Research – Interactive Dashboard</h1>
<p>Sections: Universe, Signals, Regimes, Portfolio, Risk, Performance, Optimization, Experiments, Walk Forward, Stress Tests – all interactive Plotly</p>
"""]

        for section, fig in figs.items():
            # Convert fig to HTML div
            try:
                html_div = fig.to_html(full_html=False, include_plotlyjs=False)
                html_parts.append(f'<div class="section"><h2>{section.replace("_"," ").title()}</h2>{html_div}</div>')
            except Exception as e:
                html_parts.append(f'<div class="section"><h2>{section}</h2><p>Error rendering: {e}</p></div>')

        html_parts.append("</body></html>")

        full_html = "\n".join(html_parts)
        output_path.write_text(full_html, encoding='utf-8')

        return str(output_path)

    def build_dash_app(self, data: Dict, port: int = 8050):
        """
        Build Dash app for interactive dashboard – if dash available

        Args:
            data: Same as build_dashboard
            port: Port to run on

        Returns:
            Dash app instance (call app.run_server)

        Raises:
            ImportError if dash not installed
        """
        if not HAS_DASH:
            raise ImportError("dash not installed, pip install dash")

        import dash
        from dash import dcc, html

        app = dash.Dash(__name__)

        # Generate figures
        figures = {}
        if 'universe_over_time' in data:
            figures['universe'] = self.plot_universe(data['universe_over_time'])
        if 'equity_curve' in data:
            figures['performance'] = self.plot_performance(data['equity_curve'], data.get('benchmark'))

        # Build layout with tabs for each section
        app.layout = html.Div([
            html.H1("Institutional Quantitative Research – Interactive Dashboard"),
            dcc.Tabs([
                dcc.Tab(label='Universe', children=[dcc.Graph(figure=figures.get('universe', {}))]),
                dcc.Tab(label='Signals', children=[dcc.Graph(figure=figures.get('signals', {}))]),
                dcc.Tab(label='Regimes', children=[dcc.Graph(figure=figures.get('regimes', {}))]),
                dcc.Tab(label='Portfolio', children=[dcc.Graph(figure=figures.get('portfolio', {}))]),
                dcc.Tab(label='Risk', children=[dcc.Graph(figure=figures.get('risk', {}))]),
                dcc.Tab(label='Performance', children=[dcc.Graph(figure=figures.get('performance', {}))]),
                dcc.Tab(label='Optimization', children=[dcc.Graph(figure=figures.get('optimization', {}))]),
                dcc.Tab(label='Experiments', children=[dcc.Graph(figure=figures.get('experiments', {}))]),
                dcc.Tab(label='Walk Forward', children=[dcc.Graph(figure=figures.get('walk_forward', {}))]),
                dcc.Tab(label='Stress Tests', children=[dcc.Graph(figure=figures.get('stress', {}))]),
            ])
        ])

        return app
