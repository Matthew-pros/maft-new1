import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pandas as pd
import numpy as np
from quant_engine_institutional import (
    TEMACD_Strategy,
    BTC_TEMACD,
    LT_MA_CROSS,
    Donchian_Carver,
    Kalman_Trend,
    run_backtest,
    compute_validation_scores,
    plot_comprehensive_with_underwater,
    plot_bull_market_barometer,
    plot_equity_log_scale,
    plot_master_leaderboard
)

def test_temacd_strategy():
    dates=pd.date_range("2020-01-01","2020-12-31",freq='B')
    close=pd.Series(100+np.cumsum(np.random.normal(0,1,len(dates))),index=dates)
    df=pd.DataFrame({"Close":close})
    strat=TEMACD_Strategy()
    out=strat.generate_signals(df)
    assert 'long' in out.columns and 'short' in out.columns

def test_btc_temacd():
    dates=pd.date_range("2020-01-01","2020-12-31",freq='B')
    close=pd.Series(10000+np.cumsum(np.random.normal(0,100,len(dates))),index=dates)
    df=pd.DataFrame({"Close":close})
    strat=BTC_TEMACD()
    out=strat.generate_signals(df)
    assert 'long' in out.columns

def test_run_backtest():
    dates=pd.date_range("2020-01-01","2021-12-31",freq='B')
    close=pd.Series(100+np.cumsum(np.random.normal(0.1,1,len(dates))),index=dates)
    df=pd.DataFrame({"Close":close})
    strat=TEMACD_Strategy()
    df_bt,trades=run_backtest(df,strat,capital=100000)
    assert 'equity' in df_bt.columns
    assert 'returns' in df_bt.columns
    assert len(df_bt)==len(dates)

def test_validation_scores():
    m_is={"Sharpe_raw":1.8}
    m_oos={"Sharpe_raw":1.2}
    scores=compute_validation_scores(m_is,m_oos,walk_forward_sharpe=1.1,stress_score=0.9,temporal_robust=0.7,cross_asset=0,sensibilite=0)
    assert scores["Score brut"]==8.2
    assert scores["SCORE NORMALISÉ /10"]==9.65
    scores2=compute_validation_scores(m_is,m_oos,1.1,0.9,1.0,1.0,0.5)
    assert scores2["Score brut"]==10.0
    assert scores2["SCORE NORMALISÉ /10"]==10.0

def test_plots(tmp_path=None):
    import tempfile, os
    dates=pd.date_range("2020-01-01","2025-07-15",freq='B')
    np.random.seed(42)
    ret=np.random.normal(0.002,0.014,len(dates))
    equity=100000*np.cumprod(1+ret)
    df=pd.DataFrame({"Close":equity,"returns":ret,"equity":equity},index=dates)
    metrics={"Total Return":"1741.8%","CAGR":"69.5%","Sharpe":"2.42","Sortino":"3.51","Max DD":"-13.6%","Win Rate":"50.1%","Profit Factor":"1.62","PSR":"1.000","DSR":"1.000","Start":"2020-01-01","End":"2025-07-15","Total Days":len(dates),"Sharpe_raw":2.42,"Annualized Volatility_raw":0.229,"Annualized Volatility":"22.9%"}
    with tempfile.TemporaryDirectory() as tmp:
        p1=os.path.join(tmp,"comp.png")
        plot_comprehensive_with_underwater(df,metrics,p1,style="light")
        assert os.path.exists(p1)
        p2=os.path.join(tmp,"bar.png")
        plot_bull_market_barometer(None,p2)
        assert os.path.exists(p2)
        p3=os.path.join(tmp,"log.png")
        eq_dict={"Rotational":pd.Series(equity,index=dates),"Max Sharpe":pd.Series(equity*1.1,index=dates)}
        plot_equity_log_scale(eq_dict,p3)
        assert os.path.exists(p3)
        p4=os.path.join(tmp,"lead.png")
        results=[{"ticker":"AAPL","ensemble":"KALMAN OR MACD","IS_Sharpe":1.35,"OOS_Sharpe":0.32,"Boruta_Score":94,"Total_Return":"475.74%","WinRate":"49%","MaxDD":"-33%","Num_Trades":100}]
        plot_master_leaderboard(results,p4)
        assert os.path.exists(p4)
