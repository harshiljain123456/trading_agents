# Design write-up

## Architecture

```
data/feed.py (synthetic | live stub)  ->  validate_bars
        |
core/feature_bus.py   causal indicators (indicators/*) + provenance labels -> MarketState per bar
        |
agents/*  five voting specialists (+ optional on_chain)  -> AgentSignal (signal, confidence, reason, features)
        |
agents/boss_orchestrator.py  weighted vote, agreement/coverage/strength, risk veto, kill switch, template reasoner
        |
execution/backtester.py  position management -> Order (decided at close t)
        |
execution/executor.py    fill at open t+1 with slippage + fees, cash/position/trade ledger (Backtest or Paper)
        |
execution/evaluator.py   metrics, agent scoring on matured outcomes, signal & execution ablation, calibration
        |
execution/feedback.py    every 60 bars (training only): bounded, shrunk, smoothed learned weights
        |
execution/walk_forward.py  train -> embargo -> validation -> embargo -> frozen test, chained OOS report
```

Side components: `core/portfolio_risk.py` (kill switch on the strategy's own equity), `core/kalman.py`
(per-agent confidence smoothing), `core/reasoner.py` (explanations only), `agents/adversarial_agent.py`
(shadow books, never votes).

## Data flow for one bar t

1. The executor fills the order decided at t-1 at bar t's open.
2. Equity is marked at t's close and fed to the kill switch.
3. In training runs only, the feedback loop may re-weight agents using predictions whose 20-bar outcome has
   fully matured by t.
4. `FeatureFrame.state(t)` builds a `MarketState` from bars <= t. Every indicator is trailing-window only.
   A test recomputes features on truncated history and checks they match.
5. Each agent returns an `AgentSignal`. An agent missing essential inputs returns an *unavailable* HOLD.
6. The boss computes S, agreement, coverage, data coverage and strength, then applies the override rules
   and writes the analyst note.
7. Position management turns the decision into a target exposure, and the order is queued for t+1.

## Why each agent exists

| Agent | Contribution | Inputs |
|---|---|---|
| MarketIntel | short-horizon flow pressure: OFI, CVD change, volume-confirmed momentum (an institutional-*style* proxy) | bid/ask aggressor volume, trade side, OHLCV |
| RiskEngine | market-risk score (vol, ATR%, market drawdown) driving a hard veto; never votes direction, so it also dampens S | OHLC |
| LiquidityEngine | event signal: confirmed sweeps of prior rolling lows/highs, qualified by CVD, depth and absorption | OHLCV, trade side, depth |
| SentimentEngine | social/news momentum plus contrarian funding | social, news, funding |
| RegimeDetector | regime label (EMA20/100, Wilder ADX, RV) that selects regime weights; votes only in trends | OHLC |
| OnChain (optional) | exchange netflow and stablecoin supply proxy | on-chain feed |

## Why the adversarial agent is separate

If it voted, it would cancel the ensemble by construction. As a shadow it answers one question with the same
timing and costs: would the negation of our vote have done better? It never feeds the weight updates, so its
verdict is not re-using the outcomes that tuned the strategy.

## Risk veto vs. kill switch

* **Risk veto (market):** RiskEngine score > 0.80 forces HOLD for new entries. Existing exposure is capped at
  `veto_exposure_cap` (`on_risk_veto: reduce`). If risk data is missing, the boss also forces HOLD
  (`risk_unavailable`, fail-safe).
* **Kill switch (portfolio):** this watches the strategy's net-of-cost equity. It triggers on drawdown >
  `max_drawdown` or rolling Sharpe < threshold once `min_history` returns exist. It blocks entries and
  flattens by default. Recovery happens after a cooldown, which resets the history.

## Adaptive weights without leakage

* Predictions at bar t are scored only when `t + h <= now`.
* Weight updates happen only in training windows. Validation and test runs are frozen, and a test asserts
  there are no updates.
* Each update applies: an insufficient-sample guard, a neutral band on the overlap-adjusted t-stat
  (n_eff = n/h), shrinkage toward the base weight, smoothing (0.25), and clipping to [0.3, 2.0].
* Embargo >= horizon is enforced by config validation. Thresholds are chosen on validation, never on test.
  Calibration is fitted on training data and scored on test.

## Key assumptions (details in Deliverable 8)

The data is synthetic toy dynamics with a deliberately weak flow/regime link. Execution is one instrument,
next-bar-open fills, and fixed bps costs with no market impact. Ensemble "confidence" is an uncalibrated
strength, not a probability.
