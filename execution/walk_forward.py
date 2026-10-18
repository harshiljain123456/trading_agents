"""Chronological walk-forward validation.

Fold layout (bar indices, all chronological, no shuffling)::

    [ train ][ embargo ][ validation ][ embargo ][ test ]
    -- step forward by ``step_bars`` (default = test_bars) --

* Train: strategy runs with the feedback loop on; weights adapt using only
  outcomes matured inside the training window.
* Validation: learned weights frozen; ``min_confidence`` is chosen from
  ``threshold_grid`` by validation Sharpe.
* Test: weights and threshold frozen; results are out-of-sample.
* Embargo >= outcome horizon (enforced by config validation) so labels that
  overlap a boundary cannot leak into the next segment.
* Indicators are causal, so precomputing them on the full series does not
  leak (tested in ``tests/test_integration.py``). Each segment starts flat
  with fresh capital and fresh Kalman/kill-switch state.
* Confidence calibration is fitted on training decisions and scored on test
  decisions (Brier score vs. a constant base-rate forecast).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import FrameworkConfig
from core.feature_bus import FeatureFrame
from core.models import PerformanceMetrics, Trade
from execution.backtester import Backtester, BacktestResult, _rule_backtest, bars_per_day
from execution.evaluator import LogisticCalibrator, brier, compute_metrics, matured_decision_outcomes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldWindows:
    """Index boundaries of one fold (half-open intervals)."""

    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


@dataclass
class FoldResult:
    """Results of one fold."""

    fold: int
    windows: FoldWindows
    train_metrics: PerformanceMetrics
    validation_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    test_buy_and_hold: PerformanceMetrics
    chosen_min_confidence: float
    learned_weights: dict[str, float]
    weight_updates: int
    calibration_brier: float | None
    baseline_brier: float | None
    test_returns: pd.Series
    test_bh_returns: pd.Series
    test_trades: list[Trade]
    test_exposures: pd.Series


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward output (in-sample and out-of-sample kept separate)."""

    folds: list[FoldResult]
    oos_metrics: PerformanceMetrics
    oos_buy_and_hold: PerformanceMetrics
    in_sample_mean_return: float
    validation_mean_return: float


def make_folds(config: FrameworkConfig, n_bars: int, first_index: int) -> list[FoldWindows]:
    """Generate chronological, non-overlapping-by-construction fold windows."""
    wf = config.walk_forward
    step = wf.step_bars or wf.test_bars
    folds = []
    s = first_index
    while True:
        tr = (s, s + wf.train_bars)
        va = (tr[1] + wf.embargo_bars, tr[1] + wf.embargo_bars + wf.validation_bars)
        te = (va[1] + wf.embargo_bars, va[1] + wf.embargo_bars + wf.test_bars)
        if te[1] > n_bars:
            break
        assert tr[1] + wf.embargo_bars <= va[0] and va[1] + wf.embargo_bars <= te[0]
        folds.append(FoldWindows(tr, va, te))
        s += step
    return folds


def _chain(returns: list[pd.Series], initial: float) -> pd.Series:
    """Compound per-bar returns from consecutive test windows into one equity curve."""
    r = pd.concat(returns)
    eq = initial * (1.0 + r).cumprod()
    first = pd.Series([initial], index=[r.index[0] - pd.Timedelta(seconds=1)])
    return pd.concat([first, eq])


def run_walk_forward(config: FrameworkConfig, ff: FeatureFrame) -> WalkForwardReport:
    """Execute the full walk-forward procedure and aggregate out-of-sample results."""
    folds = make_folds(config, len(ff), ff.first_trade_index)
    if not folds:
        wf = config.walk_forward
        need = wf.train_bars + wf.validation_bars + wf.test_bars + 2 * wf.embargo_bars
        raise ValueError(
            f"date range too short for walk-forward: need >= {need} tradable bars, have {len(ff) - ff.first_trade_index}"
        )
    closes = ff.column("close")
    h = config.strategy.prediction_horizon
    results: list[FoldResult] = []
    for k, fw in enumerate(folds):
        logger.info("walk-forward fold %d: train %s val %s test %s", k, fw.train, fw.validation, fw.test)
        train = Backtester(config, ff).run(*fw.train, learn=True)
        weights = train.final_weights
        best = None
        for mc in config.walk_forward.threshold_grid:
            c = config.model_copy(deep=True)
            c.strategy.min_confidence = mc
            c.features.enable_adversarial = False
            vr = Backtester(c, ff).run(*fw.validation, initial_weights=weights)
            key = (vr.metrics.sharpe if vr.metrics.sharpe is not None else -np.inf, vr.metrics.total_return)
            if best is None or key > best[0]:
                best = (key, mc, vr)
        assert best is not None
        _, mc, val = best
        c = config.model_copy(deep=True)
        c.strategy.min_confidence = mc
        test = Backtester(c, ff).run(*fw.test, initial_weights=weights)
        bh_run = _bh_run(config, ff, *fw.test)
        bh = bh_run.metrics

        cal_b = base_b = None
        if config.evaluation.calibration:
            x_tr, y_tr = matured_decision_outcomes(train.decision_records, closes, h, fw.train[1] - 1, *fw.train)
            x_te, y_te = matured_decision_outcomes(test.decision_records, closes, h, fw.test[1] - 1, *fw.test)
            cal = LogisticCalibrator()
            if cal.fit(x_tr, y_tr) and len(y_te):
                cal_b = brier(cal.predict(x_te), y_te)
                base_b = brier(np.full(len(y_te), y_tr.mean()), y_te)
        bh_eq = bh_run.equity
        results.append(
            FoldResult(
                fold=k, windows=fw, train_metrics=train.metrics, validation_metrics=val.metrics,
                test_metrics=test.metrics, test_buy_and_hold=bh, chosen_min_confidence=mc,
                learned_weights=weights, weight_updates=len(train.weight_updates),
                calibration_brier=cal_b, baseline_brier=base_b,
                test_returns=test.equity.pct_change().dropna(), test_bh_returns=bh_eq.pct_change().dropna(),
                test_trades=list(test.portfolio.trades),
                test_exposures=test.exposures,
            )
        )
    init = config.execution.initial_capital
    oos_eq = _chain([f.test_returns for f in results], init)
    oos_bh = _chain([f.test_bh_returns for f in results], init)
    zero = pd.Series(0.0, index=oos_bh.index)
    oos_expo = pd.concat([f.test_exposures for f in results])
    trades = [t for f in results for t in f.test_trades]
    fees = sum(f.test_metrics.total_fees for f in results)
    slip = sum(f.test_metrics.total_slippage for f in results)
    turn_notional = sum(f.test_metrics.turnover * f.test_metrics.initial_equity for f in results)
    oos = compute_metrics(oos_eq, trades, oos_expo, fees, slip, turn_notional, config.bars_per_year, bars_per_day(config))
    oos_bh_m = compute_metrics(oos_bh, [], zero, 0.0, 0.0, 0.0, config.bars_per_year, bars_per_day(config))
    return WalkForwardReport(
        folds=results, oos_metrics=oos, oos_buy_and_hold=oos_bh_m,
        in_sample_mean_return=float(np.mean([f.train_metrics.total_return for f in results])),
        validation_mean_return=float(np.mean([f.validation_metrics.total_return for f in results])),
    )


def _bh_run(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int) -> BacktestResult:
    """Buy-and-hold run for a window (same entry convention as run_benchmarks)."""
    size = config.execution.max_position_size
    return _rule_backtest(config, ff, start, end, lambda i, cur: size if (i == start and cur == 0) else None)
