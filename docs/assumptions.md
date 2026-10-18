# Assumptions and limitations

## Implemented

* All five core agents, the optional OnChain agent (feature flag), the boss with weighted vote,
  agreement/coverage/strength, the risk veto, the kill switch and a template reasoner.
* Regime-conditional weights, optional family normalization, Kalman confidence smoothing.
* Backtester with next-bar-open fills, fees and slippage on every fill, a long-only default with an optional
  short mode, SELL close/reduce policies, risk-veto exposure reduction, and an unfilled-order report at
  window end.
* Benchmarks: no-trade, buy-and-hold and EMA trend, all on the identical window and costs.
* Evaluator (matured-only hit rate, confidence-weighted return, standalone PnL, signal-level incremental
  contribution) and execution-level ablation. Sensitivity grid via `--sensitivity`.
* Feedback loop (training only), walk-forward with embargo, validation threshold selection, train-fitted
  logistic calibration scored on test.
* Adversarial shadow agent, paper executor that logs simulated orders, live-feed stub, config and data
  validation, CLI and unittest tests.

## Simplified approximations

* **Synthetic data:** regime-switching random walk. Order flow, depth, news and on-chain series are linked
  to the hidden regime drift (`flow_signal_strength=0.25`). Social sentiment and funding react to past
  returns. This link is a design choice, so any edge seen on synthetic data partly reflects how the
  generator was written.
* **CVD:** with synthetic data, `trade_side` is a net aggressor fraction. Without it, CVD falls back to a
  tick-rule proxy, flagged `cvd_is_proxy`.
* **Absorption ratio:** relative volume / relative range. It cannot separate absorption from two-sided churn
  without real aggressor data.
* **Costs:** fixed bps slippage and fees. There is no spread model, no market impact, no funding payments on
  positions and no borrow cost for shorts.
* **Mark-to-market:** final equity is marked at the last close with no liquidation cost. Buy-and-hold uses
  the same convention.
* **Walk-forward:** each segment starts flat with fresh capital and fresh Kalman/kill-switch state.
  Out-of-sample results are chained test-window returns, so a position still open at the end of a test
  window is marked, not closed. That is why OOS "closed trades" can be 0.
* **Hit rate:** uses close-to-close forward returns from the decision bar, not the executed fill price.
* **Incremental t-stat:** uses n_eff = n/h as a simple overlap correction, not a HAC estimator.
* **Kill switch:** the rolling Sharpe is undefined while flat (zero variance), so it can only trigger
  through drawdown then.
* **Confidence:** reported as an uncalibrated ensemble strength. Calibration is a diagnostic in the
  walk-forward report and is not used for sizing.
* **Symbol:** in synthetic mode, `--symbol` is a label only.

## Optional features not implemented

* Options-flow agent (on-chain was chosen instead).
* Ensemble of regime-specific bosses with a router (future extension).
* External LLM reasoner. The `Reasoner` protocol exists, but no LLM implementation is included.

## NSE real-data mode (added)

* **Sources:** Yahoo daily/hourly via its public chart endpoint (unofficial, no SLA, may change or
  rate-limit), Kite Connect historical API (your credentials), or CSV. Only OHLCV is observed.
* **Prices:** Yahoo OHLC are dividend- and split-adjusted with the adjclose/close factor; volume is left
  unadjusted.
* **Calendar:** bars are labelled 09:15 IST of their session. Exchange holidays simply have no bar. The
  synthetic NSE mode uses weekdays only and ignores holidays.
* **Costs:** a flat ~12 bps per side approximates delivery charges. DP charges, taxes and broker-specific
  fees are not modelled.
* **Market Mood agent (NSE):** its rules and constants (VIX 90th/10th percentile over 250 sessions, EMA100,
  63-day relative strength, weights 0.4/0.4/0.2) were fixed before any NSE backtest. They are hypotheses to
  validate with walk-forward testing. Context series are aligned to stock bars with a backward as-of fill.
* **Universe runs:** `data/universes/nifty50.txt` is the index as of the 30-Sep-2026 rebalance, applied to
  all past years (survivorship bias). Stocks listed after `--start` (e.g. JIOFIN, ETERNAL) have shorter
  histories. Agent verdicts compare Sharpe with and without each agent on every stock, using a two-sided
  sign test; they are in-sample and descriptive.
* **Step 3 designs:** the trend-only and stay-invested rules, 30% volatility target, 10% rebalance band, 50%
  caps and crash windows were fixed before running on NSE data. Crash-window returns need at least 120 bars of
  history before the crash, so a strategy that has only just started (and so is still in cash) is not credited.
  India VIX / Nifty context on Yahoo begins around late 2007 to 2008, so market mood is partly unavailable
  during the 2008 crash.
* **Step 3b former-member list:** compiled from NSE index-change history (exit years approximate). Stocks that
  no longer trade (Satyam, Ranbaxy, Cairn India, HDFC Ltd, Reliance Capital and others) cannot be tested, so some
  survivorship bias remains. Each former member is tested over the full window, not only while it was in the
  index.
* **Step 3c portfolio test:** equal-weight slots across eligible stocks, monthly rebalance; per-stock trend and
  Nifty-filter switches trade at the next open. Idle cash earns nothing (a real account could earn liquid-fund
  interest, which would favour options B and C). A stock with no price on a day is neither valued nor traded.
  Taxes are not modelled.
* **Step 5 robustness test:** settings grid (EMA 20/100, 10/50, 30/150, 50/200 x ADX 15/20/25), the 2016-01-01
  split, the 75% settings pass share and the "beat A on both Sharpe and worst fall" rule were fixed before running on
  real data. Idle cash earns 6.5%/yr (liquid-fund proxy, constant, no fund expense or tax on the interest). Capital-gains
  tax uses today's listed-equity rules for every year (STCG 20%, LTCG 12.5% above Rs 1.25 lakh per financial year,
  4% cess, FIFO lots, 365-day holding); the older, lower rates are not modelled. Tax is paid from cash on the first
  session of each April; losses are set off and carried forward without the 8-year limit. Dividends are inside the
  adjusted prices and are not taxed separately. Capital is Rs 10 lakh because the exemption is a rupee amount.
  Sharpe uses a zero risk-free rate. The halves start fresh with new capital.
* **Step 7 daily paper signal:** uses the base filter setting (EMA 20/100, ADX 20), not the best setting from Step 5.
  Each run downloads fresh data from 2014, drops a still-forming bar if run before 15:45 IST, and replays the paper book
  from its start date with the backtest rules (next-open fills, costs, 6.5% cash, tax). The journal is append-only (one
  row per completed close) so later data revisions cannot rewrite what the signal said. "Next session" is the next
  weekday; exchange holidays are not known. Order sizes are rupee targets; the share counts are rounded down at the
  last close and fractional shares are allowed in the paper book. Use the current-members list (nifty50) for paper
  trading, since you can only buy stocks that exist today.
* **Agent panel (advisory):** each daily run records every agent's and the boss's BUY/SELL/HOLD on the Nifty and each
  stock in `paper/agent_panel.csv` (one block per close, never rewritten). Scoring rule fixed before any live data:
  only BUY/SELL calls count; outcome = close 20 sessions later vs close on the call day, no costs; baseline = share of
  matured stock-days that rose. The boss's smoothing is warmed on the last 60 bars. The panel never changes the paper
  trades; an agent can earn a vote only after the review, under a rule fixed before it.
* **Not verified live from the build environment:** downloads could not be tested there (network
  blocked). Parsing, chunking, caching and the end-to-end pipeline are tested offline with recorded-format
  payloads.

## Requires external market data

* Genuine OFI/CVD (aggressor-flagged trades), order-book depth, social/news sentiment, funding history, and
  on-chain netflow/stablecoin supply. `LiveFeedStub` raises until a provider is implemented.

## Requires an exchange or broker integration

* Live order routing. `BrokerAdapter` is defined and `NullBroker` refuses all orders; `PaperExecutor` cannot
  be constructed with live orders enabled.
* Real fills, partial fills, order rejections, exchange fees/tiers and latency.

## Verification status

* The test suite (235 tests) and every documented CLI command were run on Python 3.13.16 with numpy 2.5.3,
  pandas 3.0.5, pydantic 2.13.5. All tests pass.
* All files compile under Python 3.11. The test suite could not be run on 3.11 in the build sandbox because
  package downloads were blocked there.
