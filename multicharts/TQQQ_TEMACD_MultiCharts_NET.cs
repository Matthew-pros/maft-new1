// MultiCharts .NET C# Strategy – TQQQ TEMACD
// For MultiCharts .NET SE / 64 – compile as Strategy

using System;
using System.Drawing;
using PowerLanguage.Strategy;
using PowerLanguage.Indicator;

namespace PowerLanguage.Strategy
{
    [SameAsSymbol(true)]
    public class TQQQ_TEMACD_NET : SignalObject
    {
        [Input] public int len1 { get; set; }
        [Input] public int len2 { get; set; }
        [Input] public int len3 { get; set; }
        [Input] public int fastLen { get; set; }
        [Input] public int slowLen { get; set; }
        [Input] public int macdLen { get; set; }
        [Input] public double Leverage { get; set; }
        [Input] public string TradeDirection { get; set; }
        [Input] public int StartDate { get; set; } // yyyMMdd
        [Input] public int EndDate { get; set; }

        private XAverage EMA1, EMA2, EMA3, EMA61, EMA70, aMACD;
        private VariableSeries<double> MACDLine, delta;
        private IPlotObject PlotEMA1, PlotEMA2, PlotEMA3;

        public TQQQ_TEMACD_NET(object _ctx) : base(_ctx)
        {
            len1 = 5;
            len2 = 50;
            len3 = 222;
            fastLen = 61;
            slowLen = 70;
            macdLen = 15;
            Leverage = 1.0;
            TradeDirection = "Long";
            StartDate = 20180101;
            EndDate = 21701231;
        }

        protected override void Create()
        {
            EMA1 = new XAverage(this);
            EMA2 = new XAverage(this);
            EMA3 = new XAverage(this);
            EMA61 = new XAverage(this);
            EMA70 = new XAverage(this);
            aMACD = new XAverage(this);
            MACDLine = new VariableSeries<double>(this);
            delta = new VariableSeries<double>(this);

            PlotEMA1 = AddPlot(new PlotAttributes("EMA1", EPlotShapes.Line, Color.Blue));
            PlotEMA2 = AddPlot(new PlotAttributes("EMA2", EPlotShapes.Line, Color.White));
            PlotEMA3 = AddPlot(new PlotAttributes("EMA3", EPlotShapes.Line, Color.Orange));
        }

        protected override void StartCalc()
        {
            EMA1.Length = len1;
            EMA2.Length = len2;
            EMA3.Length = len3;
            EMA61.Length = fastLen;
            EMA70.Length = slowLen;
            aMACD.Length = macdLen;
        }

        protected override void CalcBar()
        {
            EMA1.Value = Bars.Close.Value;
            EMA2.Value = Bars.Close.Value;
            EMA3.Value = Bars.Close.Value;
            EMA61.Value = Bars.Close.Value;
            EMA70.Value = Bars.Close.Value;

            double ema1 = EMA1[0];
            double ema2 = EMA2[0];
            double ema3 = EMA3[0];
            double ema61 = EMA61[0];
            double ema70 = EMA70[0];

            MACDLine.Value = ema61 - ema70;
            aMACD.Value = MACDLine.Value;
            delta.Value = MACDLine.Value - aMACD[0];

            // Cross detection (manual, since CrossOver helper not in .NET by default)
            bool deltaCrossUp = delta[0] > 0 && delta[1] <= 0;
            bool deltaCrossDown = delta[0] < 0 && delta[1] >= 0;
            bool e1e2Up = ema1 > ema2 && EMA1[1] <= EMA2[1];
            bool e1e2Down = ema1 < ema2 && EMA1[1] >= EMA2[1];
            bool e1e3Up = ema1 > ema3 && EMA1[1] <= EMA3[1];
            bool e1e3Down = ema1 < ema3 && EMA1[1] >= EMA3[1];
            bool e2e3Up = ema2 > ema3 && EMA2[1] <= EMA3[1];
            bool e2e3Down = ema2 < ema3 && EMA2[1] >= EMA3[1];

            bool longCond = deltaCrossUp || e1e2Up || e1e3Up || e2e3Up;
            bool shortCond = deltaCrossDown || e1e2Down || e1e3Down || e2e3Down;

            int date = (int)(Bars.Time[0].Year * 10000 + Bars.Time[0].Month * 100 + Bars.Time[0].Day);
            bool inRange = date >= StartDate && date <= EndDate;
            bool longOK = TradeDirection == "Long" || TradeDirection == "Both";
            bool shortOK = TradeDirection == "Short" || TradeDirection == "Both";

            double equity = 100000 + StrategyInfo.NetProfit;
            double contracts = equity / Bars.Close[0] * Leverage;
            if (contracts < 1) contracts = 1;

            // For MES futures override
            string sym = Bars.Info.Name;
            if (sym.Contains("MES"))
                contracts = 1;

            if (longCond && inRange && longOK && StrategyInfo.MarketPosition <= 0)
            {
                GenerateEntrySignal(1, "LE", (int)contracts, MarketType.Market, "Long_Entry");
            }
            if (shortCond && inRange && shortOK && StrategyInfo.MarketPosition >= 0)
            {
                GenerateEntrySignal(-1, "SE", (int)contracts, MarketType.Market, "Short_Entry");
            }
            if (StrategyInfo.MarketPosition > 0 && shortCond)
            {
                GenerateExitSignal(1, "LX", MarketType.Market, "Exit_Long");
            }
            if (StrategyInfo.MarketPosition < 0 && longCond)
            {
                GenerateExitSignal(-1, "SX", MarketType.Market, "Exit_Short");
            }

            PlotEMA1.Set(ema1);
            PlotEMA2.Set(ema2);
            PlotEMA3.Set(ema3);
        }
    }
}
