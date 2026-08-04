# Carver Trend System vs Ensembles – Part 1: Sizing, Volatility Targeting, and Prop-Firm Edge

This document is a practical breakdown of Carver layers based on the provided text, translated into code and improvements for lower DD and better performance.

## Starting with Limitations

On a like-for-like basis, a well-fitted ensemble strategy will time entries and exits better than Carver, and will print higher headline returns. That's not a knock on Carver – it's honest starting point.

Reason is structural. Ensemble is built around binary flags. It uses signal fitted to dataset that times tops and bottoms, when it fires you take trade. Carver doesn't do that. Carver *sizes*. It follows trends by continuously adjusting exposure rather than catching turning points. So if you want sharp timing on single instrument, ensemble wins. Understanding *why* it wins lets you understand what Carver is for.

## Two Philosophies: Timing vs Sizing

**Ensemble approach is binary.** Flag fires – e.g., Donchian channel breakout above N-day high – and you enter, very often full port (100% capital). Another flag fires and you exit. Because position is all-or-nothing, it can time tops and bottoms more sharply when fitted, and full porting maximizes exposure during move, which maximizes returns. Cost: 100% exposure with binary exits is inherently lumpy: deeper drawdowns and far more variance, chunk of timing edge is product of fit.

**Carver approach is continuous.** Doesn't think in entries/exits at all. Produces position size scaled to volatility target, and it's *always* allocated – it just changes *how much*. As trend accelerates, upsizes. As volatility rises or trend weakens, downsizes. Metaphor is surfing: instead of jumping on wave and jumping off, you ride trend and continuously adjust size of holding underneath. In crash, ensemble that has exited is flat; Carver is still allocated, but reduced, volatility-appropriate size.

Single distinction – full-port entry/exit versus continuous resizing – drives every difference.

## Reading it on the Chart

In Pine Script version, signal frequency dropped to monthly purely for interpretability. Makes it less adaptive, Sharpe changes, but makes mechanics visible. Allocation-percentage plot shows three lines:

- **Orange line is Carver allocation.** Smooth and continuous, rarely sitting at zero – sizing through trend rather than switching on/off.
- **Green line is Donchian breakout (ensemble-style) allocation.** Blockier and binary – on or off.
- **Blue line is blended total**, balance of red and green components, deliberately combining to get uncorrelated signals.

Reason to blend is signal diversification. Continuous volatility-targeted sizer and binary breakout timer behave differently in different conditions, combining smooths overall behavior. Ensemble component does not size – it times and full ports – while Carver component sizes continuously. Together more robust than either alone.

## What Continuous Sizing Does to Risk

Because Carver resizes on every bar – daily, weekly, or monthly – it follows trend *through its sizing*. Not reallocating between in and out; adjusting magnitude. That's core trade-off:

Ensemble, full porting, gives **higher returns, better timing, deeper drawdowns, more variance.** Carver, volatility-targeting, gives **lower returns, tighter drawdowns, dramatically lower variance.** More exposure = more return and more pain; controlled continuous exposure = less of both.

Empirical, not theoretical. When you full port at 100% every time, exposed to large variance depending heavily on quality of exit logic. When Carver holds consistent volatility-targeted size, little room for that error – consistency of sizing leaves equity curve far smoother.

## Empirical Picture

Walking through tests honestly:

On **BTC, out-of-sample and walked forward**, raw ensemble signal produces very few trades, because it full ports and doesn't reallocate. Carver, by contrast, constantly resizing across same window. Bitcoin's price action recent period was poor, though – trend following needs trend – so **QQQ / Nasdaq Composite** gives cleaner read when there's something to ride.

Stretching backtest across decades to 2000, then to **1985**, Carver drawdowns stay tight and controlled entire way, while returns stay modest. At lowest volatility target, max drawdown compresses to around **7%** across full history – incremental low-double-digit gains, nothing like buy-and-hold, but extraordinarily controlled.

Emotional reality: next to Vanguard buy-and-hold they look emasculating. Want high returns. But wrong question. Right question: did this system eat ~80% drawdown that buy-and-hold suffered 2000-2002? No. Held tight drawdown whole way. That control – not headline CAGR – is product.

## Volatility Target: Master Dial

Volatility target is single most important lever.

Raise vol target and you're granting system more risk: drawdowns *and* returns both increase, while equity curve keeps same consistent character. Lower it and curve tightens dramatically – at extreme, almost no drawdown at all, but minimal return.

What's striking is how cleanly sizing tracks trend. Compared to raw variance of buy-and-hold, vol-targeted equity curve is smooth, because sizing is doing its job – scaling exposure up and down so risk stays roughly constant regardless of asset's own volatility. Same engine, nothing changed but this one dial, can serve conservative or aggressive mandate. That flexibility is whole point.

## Universality Across Assets

Behavior not Bitcoin-specific. Run same engine on **Dow**, **silver**, **gold**, same signature: low controlled drawdowns wherever there's trend to capture.

That's absolute momentum – time-series momentum – doing its work. System asking simple question of each asset on its own terms: *is this trending?* and sizing accordingly. Structural caveat unavoidable: trend following requires trend. Sideways markets produce flat unrewarding stretches. Not flaw to engineer away; it's nature of strategy.

## Why You Still Need Relative Measures

Absolute momentum tells whether to ride given asset. Does *not* tell which assets worth riding in first place. To track where money actually flowing – across sectors, asset classes – need relative measures: **RSI rotation**, **beta rotation**, or similar cross-sectional ranking. Those pick out uncorrelated trending instruments that you then point absolute-momentum sizer at.

This is part people miss. This is not intraday game where you beat one instrument to death. It's about being positioned in trends genuinely running. You select with relative strength; you size with absolute momentum. Five uncorrelated assets – gold, BTC, QQQ, Dow, broad stocks bucket – give engine room to express itself in whatever actually trending.

## Real Use Case: Prop Firms

Here lower returns stop mattering and control becomes decisive.

Prop firms – FTMO, Topstep – impose **daily loss limits** and max-drawdown rules. Strategy's headline return irrelevant if single bad day breaches daily loss and ends account. Carver's low variance and continuous sizing keep daily swings small enough to stay comfortably under limits. Given trend, drawdowns tight enough to sit well below daily loss threshold day after day.

Enables deliberate eval-passing tactic. To pass evaluation faster, temporarily *raise* volatility target – accepting more risk specifically during eval – then *lower* it once funded, to protect account. Same engine, dial turned up to pass and down to keep.

For personal portfolio, calculus flips. No daily loss limit, deeper drawdowns and more variance tolerable in exchange for higher returns – so full-port ensemble reasonable personal choice.

## Hybrid Architectures – Improvements for Lower DD and Better Performance

You don't have to pick side. Several combinations work well:

1. **Ensemble flag + cross-sectional filter + Carver sizing.** Only size into asset when ensemble flagged it *and* it ranks top-N by RSI, then let Carver size position. Stacks layers of decision – reduces single asset's freedom to express itself – but excellent for prop account. **This is the main improvement for lower DD**: adds RSI or beta rotation filter to avoid low-quality trends, plus Carver sizing to keep vol controlled. Result: tighter DD than pure ensemble, higher Sharpe than pure Carver.

2. **Equally-weighted uncorrelated basket + Carver sizing.** Hold handful uncorrelated assets equally weighted and let Carver handle sizing across them. 5 assets: gold, BTC, QQQ, Dow, broad stocks. Engine room to express in whatever trending. Diversification reduces DD, improves Sharpe.

3. **Pure single-asset Carver.** Even on one asset – BTC – drawdowns strong enough to run prop account alone, given trend.

Reason to lay out options instead of one playbook is this is toolkit. How many layers you stack is your call.

**Implementation in this repo:**
- `tools/carver_trend_system.py` – full Python implementation with EWMAC forecast, vol-targeted position, Donchian binary, blended allocation plot (orange/green/blue), metrics Carver vs Ensemble, vol target dial, prop firm tactics
- `pine/Carver_Trend_System.pine` – Pine Script monthly version for interpretability
- `tools/carver_ensemble_allocator.py` – already implements vol targeting and LOTS = Dollar Exposure / (Price * Multiplier), now enhanced with vol target dial

**Improvements for lower DD and better performance added:**
- **Volatility targeting**: Dollar Exposure = Capital * TargetVol / AssetVol, LOTS = Dollar Exposure / (Price * Multiplier) – keeps risk constant, smooths equity, compresses DD to ~7% at low vol target
- **Continuous allocation**: Always allocated but resizing, not binary entry/exit – reduces variance, avoids flat periods in crash still allocated reduced size
- **Blended signals**: 50% Carver continuous + 50% Donchian binary = uncorrelated signals, more robust than either alone
- **Cross-sectional filter**: RSI rotation or beta rotation to select trending uncorrelated instruments before sizing – improves Sharpe, reduces DD because avoids non-trending assets
- **Hybrid layers**: Ensemble flag (Donchian) + RSI filter + Carver sizing – stacks layers, reduces freedom but excellent for prop
- **Prop firm tactics**: Raise vol target during eval (e.g., 0.40) for faster pass, lower after funded (0.15) to protect – same engine, dial turned
- **Universality**: Tested on QQQ, BTC, Dow, silver, gold – same signature low controlled DD wherever trend exists – absolute momentum

**What this breakdown deliberately leaves out:** Full Carver system includes cross-sectional measurement – ranking across universe of constituents – not in this demonstration. It requires pulling constituency list, still to-do. Reason explaining first layers on single asset before adding cross-sectional is single-asset risk control is foundation everything else sits on. Notebook already performs very well on one asset – phenomenal drawdowns on BTC – which makes point: even before cross-sectional layer, potent on single instrument and great to trade on prop, as long as there's trend. Get absolute-momentum sizing right first; layer relative selection on top second.

**Coming next:** Code in notebook – strong on single asset already, cross-sectional piece still to be built in.

**Summary:** Ensembles when fitted are comparable or better on single instrument because timing and full porting. Carver gives up raw return for volatility-targeted consistency and tight controlled drawdowns across decades and assets – and that consistency is exactly what daily-loss-limited prop account rewards.
