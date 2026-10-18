"""Event-driven backtester, benchmarks, ablation and sensitivity analysis.

Loop for each bar t in ``[start, end)``:

1. Fill the order decided at t-1's close at bar t's open (strategy and shadow books).
2. Mark equity at t's close; update the portfolio kill switch with it.
3. (training only) feedback loop may update learned weights using outcomes
   matured by t.
4. Build ``MarketState`` for t, run every agent, aggregate with the boss.
5. Convert the decision into a target exposure; queue an order for t+1.

The decision at the final bar of a window cannot be filled inside that
window and is reported as unfilled.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from agents import build_agents
from agents.adversarial_agent import AdversarialAgent, AdversarialReport
from agents.base_agent import BaseAgent
from agents.boss_orchestrator import BossOrchestrator
from core.config import TIMEFRAME_BARS_PER_YEAR, FrameworkConfig

NSE_BARS_PER_DAY: dict[str, float] = {"1d": 1.0, "1h": 7.0}
from core.feature_bus import FeatureFrame
from core.models import AgentSignal, BossDecision, MarketState, Order, PerformanceMetrics, SignalType, WeightUpdate
from core.portfolio_risk import KillSwitch, KillSwitchEvent
from execution.evaluator import DecisionRecord, Evaluator, SignalRecord, compute_metrics
from execution.executor import PaperExecutor, Portfolio, SimulatedExecutor
from execution.feedback import FeedbackController

logger = logging.getLogger(__name__)


def bars_per_day(config: FrameworkConfig) -> float:
    """Bars per day: calendar days for 24/7 crypto, trading sessions for NSE."""
    if config.data.market == "nse":
        return NSE_BARS_PER_DAY[config.data.timeframe]
    return TIMEFRAME_BARS_PER_YEAR[config.data.timeframe] / 365.0


def target_from_decision(
    decision: BossDecision, current_exposure: float, state: MarketState, config: FrameworkConfig, kill_policy: str
) -> tuple[float | None, str]:
    """Translate a boss decision into a target exposure (None = no trade).

    Position-management rules (applied even when the boss returned HOLD):
      * kill switch active: ``flatten`` -> 0, ``hold`` -> keep;
      * risk veto with ``on_risk_veto=reduce``: cap |exposure| at ``veto_exposure_cap``;
      * BUY: target = min(max_position_size, strength * exposure_cap) (optionally
        scaled by target_vol / realized_vol); only increases a long;
      * SELL: closes (``sell_policy=close``) or reduces a long; opens/increases a
        short only when ``allow_short``;
      * HOLD: keep current units.
    """
    s, ex = config.strategy, config.execution
    cur = current_exposure
    if decision.kill_switch_active:
        if kill_policy == "flatten" and abs(cur) > 1e-9:
            return 0.0, "kill switch active: flatten existing position"
        return None, "kill switch active: no new entries"
    if decision.risk_veto and s.on_risk_veto == "reduce" and abs(cur) > s.veto_exposure_cap + 1e-9:
        return math.copysign(s.veto_exposure_cap, cur), f"risk veto: reduce exposure to {s.veto_exposure_cap:.2f}"

    def sized() -> float:
        size = min(ex.max_position_size, decision.confidence * ex.exposure_cap)
        if ex.volatility_aware_sizing and state.is_valid("realized_vol") and state.get("realized_vol") > 0:
            size *= min(1.0, ex.target_volatility / state.get("realized_vol"))
        return min(size, ex.max_exposure)

    if decision.action == SignalType.BUY:
        size = sized()
        if cur < 0:
            return size, "BUY reverses short"
        if size > cur + s.min_rebalance_exposure:
            return size, f"BUY: raise exposure {cur:.2f} -> {size:.2f}"
        return None, "BUY: already at or above target exposure"
    if decision.action == SignalType.SELL:
        if cur > 1e-9:
            if s.sell_policy == "close":
                return (-sized() if s.allow_short else 0.0), "SELL: close long" + (" and reverse" if s.allow_short else "")
            new = cur * (1 - s.sell_reduce_fraction)
            return (new if new > s.min_rebalance_exposure else 0.0), f"SELL: reduce long by {s.sell_reduce_fraction:.0%}"
        if s.allow_short:
            size = -sized()
            if size < cur - s.min_rebalance_exposure:
                return size, f"SELL: short exposure {cur:.2f} -> {size:.2f}"
            return None, "SELL: already at or beyond short target"
        return None, "SELL while flat: long-only mode does not open shorts"
    return None, "HOLD"


@dataclass
class BacktestResult:
    """Everything produced by one backtest run."""

    start_index: int
    end_index: int
    equity: pd.Series
    exposures: pd.Series
    portfolio: Portfolio
    metrics: PerformanceMetrics
    decisions: list[BossDecision]
    signals: list[list[AgentSignal]]
    signal_records: list[SignalRecord]
    decision_records: list[DecisionRecord]
    weight_updates: list[WeightUpdate]
    kill_switch_events: list[KillSwitchEvent]
    adversarial: AdversarialReport | None
    unfilled_orders: list[Order]
    final_weights: dict[str, float]
    rule_counts: dict[str, int] = field(default_factory=dict)


class Backtester:
    """Runs the multi-agent strategy over a slice of a feature frame."""

    def __init__(
        self,
        config: FrameworkConfig,
        features: FeatureFrame,
        exclude_agents: tuple[str, ...] = (),
        agents: list[BaseAgent] | None = None,
        boss: BossOrchestrator | None = None,
        paper: bool = False,
    ) -> None:
        """Create a backtester (``paper=True`` uses the logging PaperExecutor)."""
        self.config = config
        self.ff = features
        self.agents = agents if agents is not None else build_agents(config, exclude=exclude_agents)
        names = [a.name for a in self.agents]
        self.boss = boss or BossOrchestrator(config, names)
        self.paper = paper
        self.evaluator = Evaluator(config, self.boss.base_weights)

    def run(
        self,
        start: int,
        end: int,
        learn: bool = False,
        initial_weights: dict[str, float] | None = None,
        on_decision: Callable[[int, BossDecision, list[AgentSignal]], None] | None = None,
    ) -> BacktestResult:
        """Run bars ``[start, end)``. ``learn=True`` enables the feedback loop."""
        cfg = self.config
        if not 0 <= start < end <= len(self.ff):
            raise ValueError(f"invalid window [{start}, {end}) for {len(self.ff)} bars")
        if end - start < 2:
            raise ValueError("window must contain at least 2 bars")
        ex, rk = cfg.execution, cfg.risk
        self.boss.reset()
        self.boss.learned_weights = dict(self.boss.base_weights)
        if initial_weights:
            self.boss.set_learned_weights(initial_weights)
        portfolio = Portfolio(ex.initial_capital, ex.fee_bps, ex.slippage_bps, allow_short=cfg.strategy.allow_short, max_leverage=ex.max_leverage)
        executor: SimulatedExecutor = PaperExecutor(portfolio) if self.paper else SimulatedExecutor(portfolio)
        ks = KillSwitch(
            rk.max_drawdown, rk.rolling_sharpe_threshold, rk.rolling_sharpe_window, rk.kill_switch_min_history,
            cfg.bars_per_year, rk.kill_switch_recovery, rk.kill_switch_cooldown_bars, rk.kill_switch_position_policy,
        )
        adv = (
            AdversarialAgent(ex.initial_capital, ex.fee_bps, ex.slippage_bps, 1.0, cfg.strategy.prediction_horizon)
            if cfg.features.enable_adversarial else None
        )
        fb = FeedbackController(self.evaluator, self.boss, cfg.evaluation.feedback_interval, start, frozen=not learn)
        opens, closes = self.ff.column("open"), self.ff.column("close")
        idx = self.ff.index
        eq_vals, ex_vals = [], []
        decisions, all_signals = [], []
        srecs: list[SignalRecord] = []
        drecs: list[DecisionRecord] = []
        rule_counts: dict[str, int] = {}

        for i in range(start, end):
            ts = self.ff.timestamp(i)
            if i > start:
                executor.on_bar_open(i, ts, opens[i])
                if adv:
                    adv.on_open(i, ts, opens[i])
            equity = portfolio.equity(closes[i])
            if equity <= 0:
                raise RuntimeError(f"equity non-positive at bar {i}; check leverage settings")
            eq_vals.append(equity)
            ex_vals.append(portfolio.exposure(closes[i]))
            if adv:
                adv.on_close(closes[i])
            ks_active = ks.update(equity, ts, i)
            fb.maybe_update(i, ts, srecs, closes)

            state = self.ff.state(i)
            signals = [a.analyze(state) for a in self.agents]
            decision = self.boss.decide(signals, kill_switch_active=ks_active, timestamp=ts)
            decisions.append(decision)
            all_signals.append(signals)
            if decision.forced_hold_rule:
                for r in decision.forced_hold_rule.split(";"):
                    rule_counts[r] = rule_counts.get(r, 0) + 1
            for c in decision.breakdown:
                srecs.append(SignalRecord(i, c.agent_name, c.signal.direction, c.confidence, c.available,
                                          c.signed_input, c.effective_weight, c.eligible))
            drecs.append(DecisionRecord(i, decision.raw_action.direction, decision.action.direction,
                                        decision.confidence, decision.directional_score))
            if on_decision:
                on_decision(i, decision, signals)
            if adv:
                adv.observe(i, decision.raw_action.direction)
            target, why = target_from_decision(decision, portfolio.exposure(closes[i]), state, cfg, ks.position_policy)
            if target is not None:
                executor.submit(Order(ts, i, target, f"{why} | {decision.reason}"))
        executor.finalize()

        equity_s = pd.Series(eq_vals, index=idx[start:end], name="equity")
        expo_s = pd.Series(ex_vals, index=idx[start:end], name="exposure")
        metrics = compute_metrics(
            equity_s, portfolio.trades, expo_s, portfolio.total_fees, portfolio.total_slippage,
            portfolio.traded_notional, cfg.bars_per_year, bars_per_day(cfg),
        )
        return BacktestResult(
            start_index=start, end_index=end, equity=equity_s, exposures=expo_s, portfolio=portfolio,
            metrics=metrics, decisions=decisions, signals=all_signals, signal_records=srecs,
            decision_records=drecs, weight_updates=fb.updates, kill_switch_events=ks.events,
            adversarial=adv.report(closes, end - 1) if adv else None, unfilled_orders=executor.unfilled,
            final_weights=dict(self.boss.learned_weights), rule_counts=rule_counts,
        )


def _rule_backtest(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int, rule: Callable[[int, float], float | None],
                   tag: str = "benchmark") -> BacktestResult:
    """Run a simple rule (bar index, current exposure) -> target exposure with the same execution model."""
    ex = config.execution
    portfolio = Portfolio(ex.initial_capital, ex.fee_bps, ex.slippage_bps, allow_short=False, max_leverage=ex.max_leverage)
    executor = SimulatedExecutor(portfolio)
    opens, closes, idx = ff.column("open"), ff.column("close"), ff.index
    eq, xp = [], []
    for i in range(start, end):
        ts = ff.timestamp(i)
        if i > start:
            executor.on_bar_open(i, ts, opens[i])
        eq.append(portfolio.equity(closes[i]))
        xp.append(portfolio.exposure(closes[i]))
        tgt = rule(i, portfolio.exposure(closes[i]))
        if tgt is not None:
            executor.submit(Order(ts, i, tgt, tag))
    executor.finalize()
    es = pd.Series(eq, index=idx[start:end])
    xs = pd.Series(xp, index=idx[start:end])
    m = compute_metrics(es, portfolio.trades, xs, portfolio.total_fees, portfolio.total_slippage,
                        portfolio.traded_notional, config.bars_per_year, bars_per_day(config))
    return BacktestResult(start, end, es, xs, portfolio, m, [], [], [], [], [], [], None, executor.unfilled, {})


rule_backtest = _rule_backtest  # public alias


def run_benchmarks(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int) -> dict[str, PerformanceMetrics]:
    """No-trade, buy-and-hold and EMA trend-following benchmarks on the same window and costs.

    Buy-and-hold enters at the same first possible fill as the strategy (open
    of ``start + 1``) at ``max_position_size`` and is marked to market at the
    final close (no exit fee), like the strategy.
    """
    size = config.execution.max_position_size
    ema_f, ema_s = ff.column("ema_fast"), ff.column("ema_slow")
    no_trade = _rule_backtest(config, ff, start, end, lambda i, cur: None)
    bh = _rule_backtest(config, ff, start, end, lambda i, cur: size if (i == start and cur == 0) else None)

    def trend(i: int, cur: float) -> float | None:
        if not (np.isfinite(ema_f[i]) and np.isfinite(ema_s[i])):
            return None
        want = size if ema_f[i] > ema_s[i] else 0.0
        if want > 0 and cur < 1e-9:
            return want
        if want == 0 and cur > 1e-9:
            return 0.0
        return None

    tf = _rule_backtest(config, ff, start, end, trend)
    return {"no_trade": no_trade.metrics, "buy_and_hold": bh.metrics, "ema_trend": tf.metrics}


@dataclass(frozen=True)
class AblationRow:
    """Execution-level ablation of one agent."""

    agent_name: str
    total_return_without: float
    sharpe_without: float | None
    delta_return: float
    delta_sharpe: float | None


def run_ablation(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int, full: BacktestResult) -> list[AblationRow]:
    """Re-run the identical backtest with each agent removed (same window, costs, rules).

    delta = full - without; positive delta means the agent added value *in this
    sample*. This is descriptive and in-sample unless run on held-out data.
    """
    rows = []
    for name in [c.agent_name for c in full.decisions[0].breakdown] if full.decisions else []:
        res = Backtester(config, ff, exclude_agents=(name,)).run(start, end)
        m = res.metrics
        ds = None if (m.sharpe is None or full.metrics.sharpe is None) else full.metrics.sharpe - m.sharpe
        rows.append(AblationRow(name, m.total_return, m.sharpe, full.metrics.total_return - m.total_return, ds))
    return rows


@dataclass(frozen=True)
class SensitivityRow:
    """One cell of the threshold sensitivity grid."""

    min_confidence: float
    score_floor: float
    regime_weights: bool
    total_return: float
    sharpe: float | None
    max_drawdown: float
    n_trades: int


def run_sensitivity(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int) -> list[SensitivityRow]:
    """Grid over min_confidence x directional floor x regime weights on/off (in-sample, descriptive)."""
    rows = []
    for mc in (0.25, 0.35, 0.45):
        for fl in (0.10, 0.15, 0.25):
            for rw in (True, False):
                c = config.model_copy(deep=True)
                c.strategy.min_confidence = mc
                c.strategy.directional_score_floor = fl
                c.features.enable_regime_weights = rw
                c.features.enable_adversarial = False
                m = Backtester(c, ff).run(start, end).metrics
                rows.append(SensitivityRow(mc, fl, rw, m.total_return, m.sharpe, m.max_drawdown, m.n_trades))
    return rows
