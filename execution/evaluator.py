"""Performance metrics, agent scoring/attribution and confidence calibration.

Agent scoring uses **matured outcomes only**: a prediction made at bar t with
horizon h is scored only when ``t + h <= as_of_index`` (the close of bar
t + h has been observed).

Definitions
-----------
* forward return  r_t^h = close[t+h] / close[t] - 1 (close-to-close; used for
  directional evaluation only, not for trade PnL).
* hit: a BUY with r > 0 or a SELL with r < 0. HOLD and unavailable signals
  are excluded from the hit rate; r == 0 counts as a miss.
* confidence-weighted return: mean of c_t * d_t * r_t^h over directional
  matured predictions.
* standalone PnL: hypothetical unit position = agent direction held from
  close t to close t+1, minus (fee+slippage) on every position change; summed
  (non-compounded) over matured bars. Independent of the ensemble.
* incremental contribution (signal-level ablation): for each matured bar,
  (S_full - S_without_agent) * r_t^h using the effective weights actually in
  force at that bar. Reported in basis points; its t-stat uses an
  overlap-adjusted sample size n_eff = n / h. The execution-level ablation
  (full re-run without the agent) is in ``execution.backtester.run_ablation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.config import FrameworkConfig
from core.models import AgentScore, PerformanceMetrics, Trade


@dataclass(frozen=True)
class SignalRecord:
    """One agent's logged output at one bar (enough to replay the vote)."""

    bar_index: int
    agent_name: str
    direction: int
    confidence: float
    available: bool
    signed_input: float
    effective_weight: float
    eligible: bool


@dataclass(frozen=True)
class DecisionRecord:
    """Boss decision summary for calibration and reporting."""

    bar_index: int
    raw_direction: int
    action_direction: int
    strength: float
    directional_score: float


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    exposures: pd.Series,
    total_fees: float,
    total_slippage: float,
    traded_notional: float,
    bars_per_year: int,
    bars_per_day: float,
) -> PerformanceMetrics:
    """Compute net-of-cost metrics from a marked-to-market equity curve.

    Unavailable metrics are returned as ``None`` with an explanation in ``notes``.
    """
    notes: dict[str, str] = {}
    if len(equity) < 2:
        raise ValueError("equity curve needs at least 2 points")
    e0, e1 = float(equity.iloc[0]), float(equity.iloc[-1])
    rets = equity.pct_change().dropna()
    total = e1 / e0 - 1.0
    years = len(rets) / bars_per_year
    cagr: Optional[float]
    if years >= 0.25 and e1 > 0:
        cagr = (e1 / e0) ** (1.0 / years) - 1.0
    else:
        cagr = None
        notes["cagr"] = f"unavailable: window {years:.2f}y < 0.25y"
    sd = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    sharpe: Optional[float]
    if len(rets) < 10 or not np.isfinite(sd) or sd < 1e-15:
        sharpe = None
        notes["sharpe"] = "unavailable: too few returns or zero variance"
    else:
        sharpe = float(rets.mean() / sd * math.sqrt(bars_per_year))
    peak = equity.cummax()
    mdd = float((1.0 - equity / peak).max())
    closed = [t for t in trades if t.is_closed]
    n = len(closed)
    wins = [t.net_pnl for t in closed if t.net_pnl > 0]
    losses = [t.net_pnl for t in closed if t.net_pnl <= 0]
    win_rate = len(wins) / n if n else None
    if n == 0:
        notes["win_rate"] = "unavailable: no closed trades"
    pf: Optional[float]
    if n == 0:
        pf = None
        notes["profit_factor"] = "unavailable: no closed trades"
    elif not losses or sum(losses) == 0:
        pf = None
        notes["profit_factor"] = "unavailable: no losing trades (would be infinite)"
    else:
        pf = sum(wins) / abs(sum(losses))
    hold = [t.bars_held for t in closed if t.bars_held is not None]
    avg_hold = float(np.mean(hold)) if hold else None
    if avg_hold is None:
        notes["avg_holding"] = "unavailable: no closed trades"
    mean_eq = float(equity.mean())
    return PerformanceMetrics(
        initial_equity=e0,
        final_equity=e1,
        total_return=total,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=mdd,
        n_trades=n,
        win_rate=win_rate,
        profit_factor=pf,
        turnover=traded_notional / mean_eq if mean_eq > 0 else 0.0,
        total_fees=total_fees,
        total_slippage=total_slippage,
        avg_holding_bars=avg_hold,
        avg_exposure=float(exposures.abs().mean()) if len(exposures) else 0.0,
        avg_long_exposure=float(exposures.clip(lower=0).mean()) if len(exposures) else 0.0,
        avg_short_exposure=float((-exposures.clip(upper=0)).mean()) if len(exposures) else 0.0,
        n_bars=len(equity),
        notes=notes,
    )


class Evaluator:
    """Scores agents on matured outcomes and suggests bounded weights."""

    def __init__(self, config: FrameworkConfig, base_weights: dict[str, float]) -> None:
        """Create an evaluator using evaluation settings and base weights."""
        self.config = config
        self.ev = config.evaluation
        self.base_weights = dict(base_weights)
        self.cost_rate = (config.execution.fee_bps + config.execution.slippage_bps) / 1e4

    def score_agents(
        self,
        history: Sequence[SignalRecord],
        prices: np.ndarray,
        lookahead: int | None = None,
        as_of_index: int | None = None,
        current_weights: dict[str, float] | None = None,
        min_index: int = 0,
    ) -> dict[str, AgentScore]:
        """Score every agent in ``history`` using outcomes matured by ``as_of_index``.

        Args:
            history: Logged signal records.
            prices: Close prices indexed by bar index (whole feature frame).
            lookahead: Forward horizon h (defaults to config).
            as_of_index: Last bar whose close is known; defaults to the last price.
            current_weights: Weights to report as current (defaults to base).
            min_index: Ignore records before this bar (e.g. training-window start).
        """
        h = lookahead or self.ev.lookahead
        as_of = len(prices) - 1 if as_of_index is None else as_of_index
        current = current_weights or self.base_weights
        recs = [r for r in history if min_index <= r.bar_index <= as_of]
        by_bar: dict[int, list[SignalRecord]] = {}
        for r in recs:
            by_bar.setdefault(r.bar_index, []).append(r)
        agents = sorted({r.agent_name for r in recs})
        out: dict[str, AgentScore] = {}
        for name in agents:
            mine = sorted((r for r in recs if r.agent_name == name), key=lambda r: r.bar_index)
            directional = [r for r in mine if r.available and r.direction != 0]
            matured = [r for r in directional if r.bar_index + h <= as_of]
            hits, cw = [], []
            for r in matured:
                fwd = prices[r.bar_index + h] / prices[r.bar_index] - 1.0
                hits.append(1.0 if r.direction * fwd > 0 else 0.0)
                cw.append(r.confidence * r.direction * fwd)
            pnl = 0.0
            prev_pos = 0
            n_pnl = 0
            for r in mine:
                if r.bar_index + 1 > as_of:
                    break
                pos = r.direction if r.available else 0
                pnl += pos * (prices[r.bar_index + 1] / prices[r.bar_index] - 1.0) - abs(pos - prev_pos) * self.cost_rate
                prev_pos = pos
                n_pnl += 1
            inc_vals = []
            for t, rows in by_bar.items():
                if t + h > as_of:
                    continue
                elig = [x for x in rows if x.eligible]
                w_all = sum(x.effective_weight for x in elig)
                me = [x for x in elig if x.agent_name == name]
                if not me or w_all <= 0:
                    continue
                s_full = sum(x.effective_weight * x.signed_input for x in elig) / w_all
                rest = [x for x in elig if x.agent_name != name]
                w_rest = sum(x.effective_weight for x in rest)
                s_wo = sum(x.effective_weight * x.signed_input for x in rest) / w_rest if w_rest > 0 else 0.0
                fwd = prices[t + h] / prices[t] - 1.0
                inc_vals.append((s_full - s_wo) * fwd)
            inc = float(np.mean(inc_vals)) * 1e4 if inc_vals else None
            tstat = None
            note = ""
            if len(inc_vals) >= 2:
                sd = float(np.std(inc_vals, ddof=1))
                n_eff = max(1.0, len(inc_vals) / h)
                tstat = float(np.mean(inc_vals) / (sd / math.sqrt(n_eff))) if sd > 1e-15 else 0.0
            if len(matured) < self.ev.min_agent_samples:
                note = f"insufficient matured directional samples ({len(matured)} < {self.ev.min_agent_samples})"
            out[name] = AgentScore(
                agent_name=name,
                prediction_count=len(mine),
                directional_count=len(directional),
                matured_count=len(matured),
                hit_rate=float(np.mean(hits)) if hits else None,
                confidence_weighted_return=float(np.mean(cw)) if cw else None,
                standalone_pnl=pnl if n_pnl else None,
                incremental_contribution=inc,
                incremental_tstat=tstat,
                current_weight=current.get(name, self.base_weights.get(name, 1.0)),
                suggested_weight=current.get(name, self.base_weights.get(name, 1.0)),
                note=note,
            )
        return out

    def suggest_weights(self, scores: dict[str, AgentScore]) -> dict[str, float]:
        """Bounded, shrunk, smoothed weight suggestions.

        * Insufficient matured samples or no t-stat -> keep the current weight.
        * |t| < ``tstat_neutral_band`` (statistically unreliable) -> target is
          the base weight (evidence-free drift back toward the prior).
        * Otherwise target = shrink*base + (1-shrink)*current*exp(sens*clip(t, -3, 3)).
        * new = (1 - smoothing)*current + smoothing*target, clipped to
          [min_weight, max_weight].
        """
        ev = self.ev
        out: dict[str, float] = {}
        for name, s in scores.items():
            base = self.base_weights.get(name, s.current_weight)
            cur = s.current_weight
            if s.matured_count < ev.min_agent_samples or s.incremental_tstat is None:
                out[name] = cur
                continue
            t = max(-3.0, min(3.0, s.incremental_tstat))
            if abs(t) < ev.tstat_neutral_band:
                target = base
            else:
                target = ev.shrinkage_to_base * base + (1 - ev.shrinkage_to_base) * cur * math.exp(ev.weight_sensitivity * t)
            new = (1 - ev.weight_smoothing) * cur + ev.weight_smoothing * target
            out[name] = min(ev.max_weight, max(ev.min_weight, new))
        return out


class LogisticCalibrator:
    """Maps ensemble strength to P(direction correct) with a 1-D logistic fit.

    Fit only on training-window matured decisions; evaluate on separate
    out-of-sample decisions with the Brier score.
    """

    def __init__(self) -> None:
        """Create an unfitted calibrator."""
        self.a: float | None = None
        self.b: float | None = None

    @property
    def fitted(self) -> bool:
        """True after a successful fit."""
        return self.a is not None

    def fit(self, x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> bool:
        """Fit P(y=1|x) = sigmoid(a + b x). Returns False if data are insufficient."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 20 or len(np.unique(y)) < 2:
            self.a = self.b = None
            return False

        def nll(p: np.ndarray) -> float:
            z = p[0] + p[1] * x
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + l2 * (p @ p))

        res = minimize(nll, np.zeros(2), method="BFGS")
        if not res.success and not np.isfinite(res.fun):
            self.a = self.b = None
            return False
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (raises if unfitted)."""
        if not self.fitted:
            raise RuntimeError("calibrator not fitted")
        z = self.a + self.b * np.asarray(x, dtype=float)  # type: ignore[operator]
        return 1.0 / (1.0 + np.exp(-z))


def matured_decision_outcomes(
    decisions: Sequence[DecisionRecord], prices: np.ndarray, horizon: int, as_of_index: int, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """(strength, hit) arrays for directional raw votes in [lo, hi) matured by ``as_of_index``."""
    xs, ys = [], []
    for d in decisions:
        if lo <= d.bar_index < hi and d.raw_direction != 0 and d.bar_index + horizon <= as_of_index:
            fwd = prices[d.bar_index + horizon] / prices[d.bar_index] - 1.0
            xs.append(d.strength)
            ys.append(1.0 if d.raw_direction * fwd > 0 else 0.0)
    return np.asarray(xs), np.asarray(ys)


def brier(p: np.ndarray, y: np.ndarray) -> float | None:
    """Mean squared error of probabilities; None if empty."""
    return float(np.mean((p - y) ** 2)) if len(y) else None
