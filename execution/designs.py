"""Step 3: compare role-based strategy designs on the same data, costs and timing.

Designs (rules fixed BEFORE running them on NSE data):

* ``ensemble``   - the original five-agent weighted vote (baseline).
* ``trend_only`` - the trend agent decides *when* to hold the stock:
                   regime BUY (trend_up) -> invest; regime SELL (trend_down) -> exit;
                   range / volatile / warm-up -> keep the current position.
* ``defensive``  - "stay invested, step aside on danger": invested by default;
                   regime SELL -> exit; market-mood SELL (weak market / fear rising)
                   -> cap at ``mood_cap``; otherwise fully invested.

Roles shared by ``trend_only`` and ``defensive`` (agents no longer vote):
* risk agent = position sizer: size = min(1, target_vol / realized_vol), and while
  its veto is active exposure is capped at ``veto_cap`` and no new entry is made;
* market intel and liquidity agents are switched off (they need real order-flow
  data that free NSE data does not have);
* trades only when the target differs from the current exposure by more than
  ``rebalance_band`` (or to exit completely), to avoid churning costs;
* no portfolio kill switch (the trend exit is the stop).

Every decision uses only the bar's close; orders fill at the next open with the
same fees and slippage as every other strategy in the framework.

Winner rule (also fixed in advance): the design that beats buy-and-hold on Sharpe
on the most stocks; ties broken by median Sharpe. Crash windows report each
design's return through known market falls versus buy-and-hold.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from agents.market_mood import MarketMoodAgent
from agents.regime_detector import RegimeDetectorAgent
from agents.risk_engine import RiskEngineAgent
from core.config import FrameworkConfig
from core.feature_bus import FeatureFrame, compute_features
from core.models import SignalType
from data.feed import MarketData
from execution.backtester import Backtester, BacktestResult, rule_backtest, run_benchmarks

logger = logging.getLogger(__name__)

DESIGNS: tuple[str, ...] = ("ensemble", "trend_only", "defensive")


def design_rule(cfg: FrameworkConfig, ff: FeatureFrame, design: str) -> Callable[[int, float], float | None]:
    """Return the causal rule (bar index, current exposure) -> target exposure (None = no trade)."""
    if design not in ("trend_only", "defensive"):
        raise ValueError(f"no rule for design '{design}'")
    a = cfg.agents
    regime = RegimeDetectorAgent(1.0, a.regime_detector)
    risk = RiskEngineAgent(1.0, a.risk_engine, cfg.risk.risk_veto_threshold)
    mood = MarketMoodAgent(1.0, a.market_mood)
    d = cfg.designs

    def rule(i: int, cur: float) -> float | None:
        st = ff.state(i)
        reg = regime.analyze(st)
        rsk = risk.analyze(st)
        size = 1.0
        if st.is_valid("realized_vol") and st.get("realized_vol") > 0:
            size = min(1.0, d.target_volatility / st.get("realized_vol"))
        veto = bool(rsk.available and rsk.features.get("risk_veto"))
        if design == "trend_only":
            if reg.available and reg.signal == SignalType.BUY:
                want = size
            elif reg.available and reg.signal == SignalType.SELL:
                want = 0.0
            else:
                want = cur  # range / volatile / warm-up: keep what we hold
        else:  # defensive
            want = size
            if reg.available and reg.signal == SignalType.SELL:
                want = 0.0
            else:
                md = mood.analyze(st)
                if md.available and md.signal == SignalType.SELL:
                    want = min(want, d.mood_cap)
        if veto:
            want = min(want, d.veto_cap) if cur > 1e-9 else 0.0  # cap existing, block new entries
        want = max(0.0, min(want, cfg.execution.max_exposure))
        if want < 1e-9 and cur > 1e-9:
            return 0.0
        if abs(want - cur) > d.rebalance_band:
            return want
        return None

    return rule


def run_design(cfg: FrameworkConfig, ff: FeatureFrame, start: int, end: int, design: str) -> BacktestResult:
    """Backtest one design over ``[start, end)``."""
    if design == "ensemble":
        return Backtester(cfg, ff).run(start, end)
    return rule_backtest(cfg, ff, start, end, design_rule(cfg, ff, design), tag=design)


def window_return(equity: pd.Series, start: str, end: str, tz: str, min_lead_bars: int = 120) -> float | None:
    """Return over [start, end] from the equity curve.

    None if the window is not covered, or if fewer than ``min_lead_bars`` bars
    precede it: a strategy that has only just started is still in cash, which
    would make it look artificially safe through a crash that begins right away.
    """
    s, e = pd.Timestamp(start).tz_localize(tz), pd.Timestamp(end).tz_localize(tz) + pd.Timedelta(days=1)
    if len(equity) == 0 or equity.index[-1] < e - pd.Timedelta(days=1):
        return None
    before = equity[equity.index <= s]
    if len(before) < min_lead_bars:
        return None
    upto = equity[equity.index < e]
    if before.empty or upto.empty:
        return None
    return float(upto.iloc[-1] / before.iloc[-1] - 1.0)


@dataclass
class DesignRow:
    """Metrics of one design (or buy-and-hold) on one symbol."""

    ret: float
    sharpe: float | None
    mdd: float
    trades: int
    expo: float
    crash: dict[str, float | None] = field(default_factory=dict)


@dataclass
class SymbolDesigns:
    """All designs plus buy-and-hold for one symbol."""

    symbol: str
    bars: int = 0
    rows: dict[str, DesignRow] = field(default_factory=dict)
    error: str | None = None


def evaluate_designs(cfg: FrameworkConfig, md: MarketData) -> SymbolDesigns:
    """Run every design and buy-and-hold on one symbol."""
    ff = compute_features(md, cfg)
    start, end = ff.first_trade_index, len(ff)
    out = SymbolDesigns(symbol=md.symbol, bars=end - start)
    tz = cfg.app.timezone
    for name in DESIGNS:
        r = run_design(cfg, ff, start, end, name)
        m = r.metrics
        out.rows[name] = DesignRow(m.total_return, m.sharpe, m.max_drawdown, m.n_trades, m.avg_exposure,
                                   {k: window_return(r.equity, a, b, tz) for k, (a, b) in cfg.designs.crash_windows.items()})
    size = cfg.execution.max_position_size
    bh = rule_backtest(cfg, ff, start, end, lambda i, cur: size if (i == start and cur == 0) else None)
    m = bh.metrics
    out.rows["buy_and_hold"] = DesignRow(m.total_return, m.sharpe, m.max_drawdown, m.n_trades, m.avg_exposure,
                                         {k: window_return(bh.equity, a, b, tz) for k, (a, b) in cfg.designs.crash_windows.items()})
    return out


@dataclass(frozen=True)
class DesignSummary:
    """Cross-sectional summary of one design."""

    design: str
    n: int
    beat_bh_sharpe: int
    beat_bh_return: int
    median_return: float
    median_sharpe: float | None
    median_mdd: float
    median_expo: float
    median_trades: float
    crash: dict[str, float | None]
    vs_ensemble_wins: int | None
    vs_ensemble_p: float | None


@dataclass
class DesignReport:
    """Per-symbol rows, per-design summaries and the pre-registered winner."""

    symbols: list[SymbolDesigns]
    summaries: list[DesignSummary]
    winner: str | None


def _med(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def summarize(symbols: list[SymbolDesigns], crash_names: list[str]) -> DesignReport:
    """Aggregate designs across symbols and pick the winner by the pre-registered rule."""
    ok = [s for s in symbols if s.error is None]
    sums = []
    for name in list(DESIGNS) + ["buy_and_hold"]:
        rows = [s.rows[name] for s in ok]
        bh = [s.rows["buy_and_hold"] for s in ok]
        sh = lambda x: x if x is not None else -9.0  # noqa: E731
        beat_sh = sum(sh(r.sharpe) > sh(b.sharpe) for r, b in zip(rows, bh)) if name != "buy_and_hold" else 0
        beat_ret = sum(r.ret > b.ret for r, b in zip(rows, bh)) if name != "buy_and_hold" else 0
        wins = p = None
        if name not in ("ensemble", "buy_and_hold"):
            ens = [s.rows["ensemble"] for s in ok]
            w = sum(sh(r.sharpe) > sh(e.sharpe) for r, e in zip(rows, ens))
            l = sum(sh(r.sharpe) < sh(e.sharpe) for r, e in zip(rows, ens))
            wins, p = w, (binomtest(w, w + l, 0.5).pvalue if w + l else None)
        crash = {c: _med([r.crash[c] for r in rows if r.crash.get(c) is not None]) for c in crash_names}
        sums.append(DesignSummary(
            design=name, n=len(rows), beat_bh_sharpe=beat_sh, beat_bh_return=beat_ret,
            median_return=_med([r.ret for r in rows]) or 0.0,
            median_sharpe=_med([r.sharpe for r in rows if r.sharpe is not None]),
            median_mdd=_med([r.mdd for r in rows]) or 0.0, median_expo=_med([r.expo for r in rows]) or 0.0,
            median_trades=_med([float(r.trades) for r in rows]) or 0.0, crash=crash,
            vs_ensemble_wins=wins, vs_ensemble_p=p,
        ))
    cands = [s for s in sums if s.design != "buy_and_hold"]
    winner = max(cands, key=lambda s: (s.beat_bh_sharpe, s.median_sharpe or -9)).design if ok else None
    return DesignReport(symbols=symbols, summaries=sums, winner=winner)


def compare_designs(
    cfg: FrameworkConfig,
    symbols: list[str],
    loader: Callable[[str], MarketData],
    progress: Callable[[int, int, str], None] | None = None,
    jobs: int = 1,
) -> DesignReport:
    """Run all designs on every symbol (data loaded here, backtests in ``jobs`` processes)."""
    results: dict[int, SymbolDesigns] = {}
    futs = {}
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        for i, sym in enumerate(symbols):
            if progress:
                progress(i + 1, len(symbols), sym)
            try:
                md = loader(sym)
            except Exception as exc:  # one bad symbol must not abort the comparison
                logger.warning("%s skipped: %s", sym, exc)
                results[i] = SymbolDesigns(symbol=sym, error=f"{type(exc).__name__}: {exc}")
                continue
            if pool is None:
                try:
                    results[i] = evaluate_designs(cfg, md)
                except Exception as exc:
                    results[i] = SymbolDesigns(symbol=sym, error=f"{type(exc).__name__}: {exc}")
            else:
                futs[i] = pool.submit(evaluate_designs, cfg, md)
        for i, f in futs.items():
            try:
                results[i] = f.result()
            except Exception as exc:
                results[i] = SymbolDesigns(symbol=symbols[i], error=f"{type(exc).__name__}: {exc}")
    finally:
        if pool is not None:
            pool.shutdown()
    ordered = [results[i] for i in range(len(symbols))]
    return summarize(ordered, list(cfg.designs.crash_windows))


def write_designs_csv(rep: DesignReport, path: Path) -> None:
    """One row per symbol x design."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in rep.symbols:
        if s.error:
            rows.append({"symbol": s.symbol, "error": s.error})
            continue
        for name, r in s.rows.items():
            rows.append({"symbol": s.symbol, "design": name, "bars": s.bars, "return": r.ret, "sharpe": r.sharpe,
                         "max_dd": r.mdd, "trades": r.trades, "avg_exposure": r.expo,
                         **{f"crash_{k}": v for k, v in r.crash.items()}})
    pd.DataFrame(rows).to_csv(path, index=False)
