# Multi-Agent Trading Strategy Framework

This is a research and simulation framework. Five specialist agents vote, a Boss Orchestrator aggregates
their votes, and a cost-aware backtester executes the result causally. On top of that sit agent
attribution, leakage-free adaptive weights and walk-forward validation.

> **Status:** this is a tested, reproducible *baseline*. It is not a profitable or production-ready trading
> system. All bundled data is **synthetic**. No real orders can be sent.

## Install

* Python **3.11+** (developed and tested on 3.13; syntax-checked on 3.11)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # numpy, pandas, scipy, pydantic, pyyaml only
```

## Project structure

```
trading_agents/
├── agents/                 voting agents, boss, adversarial shadow, agent factory
│   ├── base_agent.py       BaseAgent contract + explicit "unavailable" HOLD
│   ├── market_intel.py     OFI + CVD + institutional-style momentum
│   ├── risk_engine.py      market risk score (vol, ATR%, drawdown) -> veto
│   ├── liquidity_engine.py confirmed liquidity sweeps
│   ├── sentiment_engine.py social + news + contrarian funding
│   ├── regime_detector.py  EMA20/100 + Wilder ADX + RV -> regime
│   ├── on_chain.py         optional 6th agent (feature flag)
│   ├── adversarial_agent.py shadow counter-consensus books (never votes)
│   └── boss_orchestrator.py weighted vote, diagnostics, overrides, explanation
├── core/                   shared contracts and utilities (added for clarity)
│   ├── models.py           dataclasses: MarketState, AgentSignal, BossDecision, Order, Fill, Trade, ...
│   ├── config.py           Pydantic config schema + loader (unknown keys rejected)
│   ├── feature_bus.py      causal indicator computation -> MarketState per bar
│   ├── kalman.py           1-D Kalman filter
│   ├── portfolio_risk.py   drawdown/Sharpe kill switch
│   └── reasoner.py         Reasoner protocol + deterministic TemplateReasoner
├── indicators/             pure functions: order_flow, volatility, liquidity, sentiment, trend (EMA/ADX)
├── execution/
│   ├── backtester.py       event loop, position management, benchmarks, ablation, sensitivity
│   ├── executor.py         Portfolio accounting, SimulatedExecutor, PaperExecutor, NullBroker
│   ├── evaluator.py        metrics, agent scoring, weight suggestions, calibration
│   ├── feedback.py         periodic training-only reweighting
│   └── walk_forward.py     chronological train/val/test with embargo
├── data/feed.py            synthetic feed, validation, LiveFeedStub
├── data/providers.py       NSE real data: Yahoo (free), Kite Connect, CSV + cache
├── data/universes/         symbol lists: nifty50 (current), nifty50_former (ex-members), nifty50_ever (both)
├── execution/universe.py   multi-symbol runs, remove-one-agent verdicts with sign test
├── execution/designs.py    Step 3 role-based designs (trend-only, stay-invested) vs ensemble
├── execution/portfolio_test.py  Step 3c equal-weight basket simulator (hold / trend / Nifty filter), cash interest, Indian tax book
├── execution/robustness.py     Step 5 robustness test of the Nifty crash filter (settings grid, split periods, real money)
├── execution/daily_signal.py   Step 7 daily paper-trading signal, paper book replay and journal (paper/)
├── execution/agent_panel.py    advisory agent panel (every agent + boss view daily) and its live scorecard
├── run_daily_signal.bat        Windows: run the daily signal, output to logs\signal_<date>.txt
├── execution/dashboard.py      builds the web dashboard site/index.html (charts, agent panel, journal) on every daily run
├── .github/workflows/daily-signal.yml  free cloud run on GitHub every weekday + website (see docs/WEBSITE_SETUP.md)
├── setup_daily_task.bat        Windows: one-time setup of the weekday 4:30 pm scheduled run (remove_daily_task.bat undoes it)
├── tests/                  unittest suite (217 tests)
├── docs/                   design, assumptions, next steps, captured transcript
├── reporting.py            CLI text formatting (library code never prints)
├── main.py                 CLI
├── config.yaml             crypto/synthetic defaults
├── config_nse.yaml         NSE defaults (IST, 250 sessions/yr, Indian costs)
└── requirements.txt
```

## Commands

```bash
python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic
python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --walk-forward
python main.py --mode paper --symbol BTCUSDT --synthetic
python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --seed 42
python main.py --help
python -m unittest discover -s tests -v
```

Runtime on the build machine: backtest about 7 s (including 5 ablation re-runs), walk-forward about 9 s,
tests about 5 s.

### CLI arguments

| Argument | Default | Behaviour |
|---|---|---|
| `--mode` | `backtest` | `backtest` or `paper` (bar-by-bar replay through `PaperExecutor`, simulated fills only) |
| `--market` | `crypto` | `crypto` or `nse`; selects `config.yaml` or `config_nse.yaml` |
| `--symbol` | `BTCUSDT` (crypto) | instrument; required for NSE (`RELIANCE`, `NIFTY50`, `BANKNIFTY`); a label only in synthetic mode |
| `--symbols` | none | comma list (real data, plain backtest): per-symbol table, pooled trades, agent verdicts |
| `--universe` | none | `nifty50` (from `data/universes/`) or a path to your own symbol list; same output as `--symbols` |
| `--compare-designs` | off | Step 3: run the ensemble, trend-only and stay-invested designs plus buy-and-hold on every symbol; prints the winner by a rule fixed in advance |
| `--portfolio-test` | off | Step 3c: simulate an equal-weight basket of the universe three ways (hold, per-stock trend, Nifty crash filter) with monthly rebalancing |
| `--robustness` | off | Step 5: test option C against holding across nearby filter settings, two halves of history, with cash interest and capital-gains tax; pass rule fixed in advance |
| `--daily-signal` | off | Step 7: after the close, print tomorrow's action for the Nifty-filter basket (INVESTED / IN CASH, orders), update the paper book and append to `paper/journal.csv`. No orders are sent. |
| `--paper-dir` | `paper/` | folder for the paper book (`state.json`), journal and order files; delete `state.json` to restart the paper book |
| `--jobs` | CPU cores minus 1 (max 4) | parallel worker processes for `--symbols` / `--universe` |
| `--real` | | real historical data (NSE); one of `--synthetic` / `--real` / `--live-feed` is required |
| `--source` | `yahoo` | `yahoo` (free), `kite` (env credentials), `csv` (with `--csv PATH`) |
| `--cache-dir` | `data_cache/` | where downloads are cached |
| `--refresh-data` | off | re-download, ignoring the cache |
| `--start` | `2023-01-01` | midnight in the market timezone (UTC / IST), **inclusive** |
| `--end` | `2024-01-01` | midnight in the market timezone, **exclusive** |
| `--synthetic` / `--live-feed` | | data source; `--live-feed` is a stub that exits with code 2 and never falls back |
| `--walk-forward` | off | walk-forward validation (backtest mode only) |
| `--seed` | `app.seed` (42) | synthetic seed |
| `--config` | `config.yaml` next to `main.py` | YAML config |
| `--log-level` | `app.log_level` | DEBUG shows every forced-HOLD rule per bar |
| `--no-ablation` | off | skip the 5 execution-level ablation re-runs |
| `--sensitivity` | off | 18-run grid over min_confidence x floor x regime weights |
| `--show-decision N` | first non-HOLD | print the full bar-level decision at bar offset N |
| `--paper-bars` | 300 | bars replayed in paper mode |
| `--output-dir` | none | write equity.csv, fills.csv, trades.csv, decisions.csv, leaderboard.json |

Exit codes: 0 means success; 2 means a config, data, date-range or live-feed error (with an actionable
message).

## Configuration guide (`config.yaml`)

The config is validated at startup. Unknown keys and inconsistent values are rejected (for example
embargo < horizon, `long_only` together with `allow_short`, `max_exposure > max_leverage`, or weights
outside the bounds). Settings that most affect risk and performance:

* `strategy.min_confidence` (0.35) and `directional_score_floor` (0.15): together these decide how often
  the strategy trades. The sensitivity grid shows results swing widely with them, on very few trades.
* `execution.fee_bps` 10 / `slippage_bps` 5: charged on every fill, including exits and reversals.
* `execution.max_leverage` 1.0, `max_exposure` 1.0: no leverage by default.
* `strategy.on_risk_veto: reduce` with `veto_exposure_cap` 0.5: what happens to an open position when the
  veto fires.
* `risk.*`: kill switch at 20% drawdown or rolling Sharpe < -1 (after 60 bars). By default it flattens
  positions, then recovers after a 60-bar cooldown.
* `evaluation.*`: weight bounds [0.3, 2.0], smoothing 0.25, shrinkage 0.5, neutral |t| < 1.
* `regime_weights`: illustrative multipliers only, not fitted.
* `synthetic.flow_signal_strength` 0.25: how strongly synthetic flow and sentiment depend on the hidden
  regime. Set it to 0 for a pure-noise control.

## Boss math (summary)

* `S = sum(w_i * x_i) / sum(w_i)` over eligible (enabled and available) agents, where
  `x_i = c_i * d_i` (Kalman-smoothed if enabled). HOLD votes count in the denominator. Unavailable
  agents count in neither part.
* `agreement` = aligned directional weight / directional weight
* `coverage` = directional weight / eligible weight
* `data_coverage` = eligible weight / enabled weight
* `strength = agreement * mean_conf(aligned) * sqrt(coverage)`. This is labelled
  `uncalibrated_ensemble_strength`.
* HOLD rules, in order: `kill_switch`, `risk_unavailable`, `risk_veto`, `data_coverage`,
  `directional_score_floor`, `min_confidence`.
* Effective weight = learned x regime factor x family factor. All three appear in the decision breakdown.

## Performance metric definitions

All metrics are net of fees and slippage, computed from the bar-close mark-to-market equity curve.

* **Total return:** final equity / initial equity - 1.
* **CAGR:** only shown if the window is at least 0.25 years.
* **Sharpe:** mean / std of per-bar returns x sqrt(bars_per_year), with risk-free rate 0. "n/a" if fewer
  than 10 returns or zero variance.
* **Max drawdown:** maximum of (1 - equity / running peak).
* **Trades:** closed round trips (flat to flat). Win rate and profit factor use per-trade net PnL. Profit
  factor is n/a when there are no losses.
* **Turnover:** traded notional / mean equity.
* **Fees / slippage:** sums over all fills. Slippage is already inside the fill prices; it is reported for
  transparency.
* **Average exposure:** mean |position value| / equity per bar.
* **Agent hit rate:** share of matured BUY/SELL predictions whose 20-bar forward return has the predicted
  sign. HOLD is excluded. This is not the trade win rate.

## Sample output (actual, captured; seed 42; full transcript in `docs/sample_run_transcript.txt`)

```
== Benchmark comparison (same window, same fees/slippage, mark-to-market) ==
  name              return   sharpe    maxDD  trades       fees
  strategy          +6.17%     0.43   19.21%       5        383
  no_trade          +0.00%      n/a    0.00%       0          0
  buy_and_hold     -24.72%     0.28   68.91%       0        100
  ema_trend        -28.83%    -0.01   61.29%      14      2,081
```

How to read this honestly: the strategy was in the market about 10% of the time and made 5 trades. The
in-sample sensitivity grid ranges from -12% to +16% across nearby thresholds. The adversarial counter-book
beat the consensus book. This is **not** evidence of an edge.

## NSE real data (India cash equities)

```bash
# one stock, free Yahoo daily data (downloaded once, then cached in data_cache/)
python main.py --market nse --real --symbol RELIANCE --start 2015-01-01 --end 2026-01-01
python main.py --market nse --real --symbol RELIANCE --start 2015-01-01 --end 2026-01-01 --walk-forward
python main.py --market nse --real --symbol RELIANCE --start 2025-01-01 --end 2026-01-01 --mode paper --paper-bars 200
# several stocks at once -> per-stock table, pooled stats and agent verdicts
python main.py --market nse --real --symbols RELIANCE,TCS,HDFCBANK,INFY,ITC --start 2015-01-01 --end 2026-01-01
# all Nifty 50 stocks (list in data/universes/nifty50.txt); results also saved to reports/*.csv
python main.py --market nse --real --universe nifty50 --start 2015-01-01 --end 2026-01-01
# Step 3: compare designs (ensemble vs trend-only vs stay-invested) incl. the 2008 and 2020 crashes
python main.py --market nse --real --universe nifty50 --start 2006-01-01 --end 2026-01-01 --compare-designs
# Step 3b: survivorship check on former Nifty 50 members, and on everything ever in the index
python main.py --market nse --real --universe nifty50_former --start 2006-01-01 --end 2026-01-01 --compare-designs
python main.py --market nse --real --universe nifty50_ever --start 2006-01-01 --end 2026-01-01 --compare-designs
# Step 3c: portfolio test - equal-weight basket: hold vs per-stock trend vs Nifty crash filter
python main.py --market nse --real --universe nifty50 --start 2006-01-01 --end 2026-01-01 --portfolio-test

# Step 5: robustness test of the Nifty crash filter (C) vs holding the basket (A)
python main.py --market nse --real --universe nifty50_ever --start 2006-01-01 --end 2026-01-01 --robustness

# Step 7: daily paper signal - run once a day after 4 pm IST (current Nifty 50 members only)
python main.py --market nse --real --universe nifty50 --daily-signal
# The daily signal also prints an advisory AGENT PANEL (every agent's and the boss's view on the Nifty and each
# stock). It is logged to paper/agent_panel.csv and scored live after 20 sessions; it does NOT drive the paper trades.
# Windows automation: double-click setup_daily_task.bat once; daily output then lands in logs\signal_<date>.txt
# index (no volume data, so volume-based agents abstain)
python main.py --market nse --real --symbol NIFTY50 --start 2015-01-01 --end 2026-01-01
```

* `--market nse` loads `config_nse.yaml`: IST dates, 250 sessions per year, long-only, about 12 bps cost
  per side (approximate delivery STT/exchange/stamp/GST; check it against your broker) and 5 bps
  slippage. Equity-scale risk and regime thresholds were set **before** looking at any NSE results.
* `--source yahoo` (default) is free and needs no account. `--source kite` uses your Zerodha Kite
  Connect app: set `KITE_API_KEY` and `KITE_ACCESS_TOKEN` as environment variables first. Kite
  historical data needs an active Kite Connect subscription. `--source csv --csv file.csv` loads your own
  file (date, open, high, low, close, volume).
* `--refresh-data` re-downloads instead of using the cache.
* **What is real vs unavailable:** only OHLCV exists in these feeds. OFI, depth, social/news sentiment
  and funding are marked UNAVAILABLE.
  * MarketIntel runs in a clearly labelled **proxy mode**: a tick-rule CVD built from price direction,
    not real buyer/seller data, with confidence halved.
  * The crypto Sentiment agent is replaced on NSE by a **Market Mood** agent
    (`features.sentiment_slot: market_mood`). It uses India VIX at its extremes (contrarian when panic
    recedes, risk-off while fear rises), the Nifty 50 trend vs its 100-day EMA, and the stock's 3-month
    relative strength vs Nifty. The VIX and Nifty series are downloaded with the stock and joined
    without look-ahead. If either is missing, the agent runs on what it has (minimum 2 of 3) or abstains.
  * The Liquidity agent can only confirm sweeps with the CVD proxy.
* **Expect very few trades:** with fewer independent inputs, the ensemble often stays below its
  thresholds. Offline checks showed only 3–5 trades over 10 years. Do not loosen thresholds by looking
  at the backtest; use walk-forward.
* Taxes (STCG/LTCG), DP charges and corporate-action edge cases are not modelled. Yahoo prices are
  split- and dividend-adjusted; Kite prices are used as returned.

## Connecting a real market-data provider

1. Implement a class with `get_bars(symbol, start, end) -> MarketData` (see `MarketDataProvider` in
   `data/feed.py`). Return a UTC-indexed DataFrame with at least `open, high, low, close, volume`.
2. Label every column you really receive `FeatureStatus.OBSERVED`. Label the rest `UNAVAILABLE`, and never
   fill them with synthetic values. Agents whose essential inputs are unavailable then abstain
   automatically, and the boss excludes them.
3. Include `warmup_bars` of history before `start` and set `first_trade_index`.
4. Call `validate_bars`, then wire the provider into `main._load_data` in place of `LiveFeedStub`.

## Paper executor and broker integration

`PaperExecutor` logs every order (`PAPER ORDER (simulated, no broker)`) and fills it with the same
next-open, fee and slippage model. Live routing is deliberately impossible: `NullBroker.submit_order`
raises, and `PaperExecutor(live_orders_enabled=True)` raises. To integrate a broker, implement
`BrokerAdapter.submit_order`, add reconciliation of real fills into `Portfolio`, and add a separate,
explicitly opted-in executor. Keep exchange-specific code out of `agents/` and `core/`.

## Known limitations

See `docs/assumptions.md`. In short:

* The data is synthetic.
* Execution is a single instrument with fixed bps costs and no market impact.
* Confidence is uncalibrated.
* The ensemble-of-bosses and options-flow agent are not implemented.
* The test suite has been run on Python 3.13 only.
