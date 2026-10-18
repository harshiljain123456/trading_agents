"""Multi-symbol (universe) evaluation with cross-sectional agent verdicts.

For every symbol the identical strategy, window and cost model is run, plus
(optionally) one execution-level ablation per agent (the backtest re-run with
that agent removed). Results are then pooled:

* per agent: mean / median change in return and Sharpe when the agent is
  present vs removed, the number of symbols where it helped, a two-sided
  sign test (binomial, p = 0.5) on "helped vs hurt", and its pooled
  matured 20-bar directional hit rate;
* pooled trades: count, win rate, profit factor across all symbols;
* strategy vs buy-and-hold: how many symbols the strategy beat on return
  and on Sharpe.

Verdict rule (fixed in advance, descriptive): an agent "HELPS" if it raised
Sharpe on >= 60% of symbols with sign-test p < 0.05; "HURTS" if it lowered
Sharpe on >= 60% with p < 0.05; otherwise "UNCLEAR". All of this is in-sample
for the chosen window; confirm with walk-forward before acting on it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import binomtest

from core.config import FrameworkConfig
from core.feature_bus import compute_features
from data.feed import MarketData
from execution.backtester import Backtester, run_ablation, run_benchmarks
from execution.evaluator import Evaluator

logger = logging.getLogger(__name__)


def load_universe(name_or_path: str) -> list[str]:
    """Read a universe file (built-in name like ``nifty50`` or a path); '#' starts a comment."""
    p = Path(name_or_path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "data" / "universes" / f"{name_or_path.lower()}.txt"
    if not p.exists():
        raise ValueError(f"universe '{name_or_path}' not found (looked for {p})")
    syms = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip().upper()
        if s and s not in syms:
            syms.append(s)
    if not syms:
        raise ValueError(f"universe file {p} contains no symbols")
    return syms


@dataclass
class SymbolResult:
    """Outcome for one symbol (``error`` set when it could not be run)."""

    symbol: str
    bars: int = 0
    ret: float = 0.0
    sharpe: float | None = None
    mdd: float = 0.0
    trades: int = 0
    win: float | None = None
    expo: float = 0.0
    bh_ret: float = 0.0
    bh_sharpe: float | None = None
    bh_mdd: float = 0.0
    trade_pnls: list[float] = field(default_factory=list)
    ablation: dict[str, tuple[float, float | None]] = field(default_factory=dict)  # agent -> (d_ret, d_sharpe)
    hits: dict[str, tuple[int, int]] = field(default_factory=dict)  # agent -> (hits, matured)
    error: str | None = None


@dataclass(frozen=True)
class AgentVerdict:
    """Cross-sectional summary for one agent."""

    agent: str
    n_symbols: int
    mean_d_ret: float
    median_d_ret: float
    mean_d_sharpe: float | None
    helped: int
    hurt: int
    sign_p: float | None
    hit_rate: float | None
    matured: int
    verdict: str


@dataclass
class UniverseReport:
    """Everything produced by :func:`run_universe`."""

    results: list[SymbolResult]
    verdicts: list[AgentVerdict]
    pooled_trades: int
    pooled_win_rate: float | None
    pooled_profit_factor: float | None


def evaluate_symbol(cfg: FrameworkConfig, md: MarketData, ablation: bool) -> SymbolResult:
    """Backtest one symbol, its buy-and-hold benchmark, agent hit rates and (optionally) ablations."""
    ff = compute_features(md, cfg)
    start, end = ff.first_trade_index, len(ff)
    bt = Backtester(cfg, ff)
    res = bt.run(start, end)
    bh = run_benchmarks(cfg, ff, start, end)["buy_and_hold"]
    m = res.metrics
    out = SymbolResult(
        symbol=md.symbol, bars=end - start, ret=m.total_return, sharpe=m.sharpe, mdd=m.max_drawdown,
        trades=m.n_trades, win=m.win_rate, expo=m.avg_exposure, bh_ret=bh.total_return, bh_sharpe=bh.sharpe,
        bh_mdd=bh.max_drawdown, trade_pnls=[t.net_pnl for t in res.portfolio.trades if t.is_closed],
    )
    scores = Evaluator(cfg, bt.boss.base_weights).score_agents(
        res.signal_records, ff.column("close"), as_of_index=end - 1, min_index=start
    )
    for name, s in scores.items():
        if s.hit_rate is not None:
            out.hits[name] = (int(round(s.hit_rate * s.matured_count)), s.matured_count)
    if ablation:
        for row in run_ablation(cfg, ff, start, end, res):
            out.ablation[row.agent_name] = (row.delta_return, row.delta_sharpe)
    return out


def _verdicts(results: list[SymbolResult]) -> list[AgentVerdict]:
    agents = sorted({a for r in results for a in r.ablation})
    out = []
    for a in agents:
        rows = [r.ablation[a] for r in results if a in r.ablation]
        d_ret = np.array([x[0] for x in rows])
        d_sh = np.array([x[1] for x in rows if x[1] is not None])
        helped = int((d_sh > 1e-9).sum())
        hurt = int((d_sh < -1e-9).sum())
        p = binomtest(helped, helped + hurt, 0.5).pvalue if helped + hurt > 0 else None
        n_dec = helped + hurt
        verdict = "UNCLEAR"
        if p is not None and p < 0.05 and n_dec:
            if helped / n_dec >= 0.6:
                verdict = "HELPS"
            elif hurt / n_dec >= 0.6:
                verdict = "HURTS"
        h = sum(r.hits.get(a, (0, 0))[0] for r in results)
        n = sum(r.hits.get(a, (0, 0))[1] for r in results)
        out.append(AgentVerdict(
            agent=a, n_symbols=len(rows), mean_d_ret=float(d_ret.mean()), median_d_ret=float(np.median(d_ret)),
            mean_d_sharpe=float(d_sh.mean()) if len(d_sh) else None, helped=helped, hurt=hurt, sign_p=p,
            hit_rate=h / n if n else None, matured=n, verdict=verdict,
        ))
    return out


def run_universe(
    cfg: FrameworkConfig,
    symbols: list[str],
    loader: Callable[[str], MarketData],
    ablation: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    jobs: int = 1,
) -> UniverseReport:
    """Run every symbol; failures are recorded per symbol and never stop the run.

    Data is loaded sequentially in this process (network + cache); the
    backtests then run in ``jobs`` worker processes. Results are identical to
    ``jobs=1`` because every run is deterministic and independent.
    """
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    results: dict[int, SymbolResult] = {}
    pending: dict[int, object] = {}
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        for i, sym in enumerate(symbols):
            if progress:
                progress(i + 1, len(symbols), sym)
            try:
                md = loader(sym)
            except Exception as exc:  # one bad symbol (delisted, no data) must not abort the universe
                logger.warning("%s skipped: %s", sym, exc)
                results[i] = SymbolResult(symbol=sym, error=f"{type(exc).__name__}: {exc}")
                continue
            if pool is None:
                try:
                    results[i] = evaluate_symbol(cfg, md, ablation)
                except Exception as exc:
                    logger.warning("%s failed: %s", sym, exc)
                    results[i] = SymbolResult(symbol=sym, error=f"{type(exc).__name__}: {exc}")
            else:
                pending[i] = pool.submit(evaluate_symbol, cfg, md, ablation)
        for i, fut in pending.items():
            try:
                results[i] = fut.result()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("%s failed: %s", symbols[i], exc)
                results[i] = SymbolResult(symbol=symbols[i], error=f"{type(exc).__name__}: {exc}")
    finally:
        if pool is not None:
            pool.shutdown()
    ordered = [results[i] for i in range(len(symbols))]
    ok = [r for r in ordered if r.error is None]
    pnls = [p for r in ok for p in r.trade_pnls]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else None
    return UniverseReport(
        results=ordered, verdicts=_verdicts(ok), pooled_trades=len(pnls),
        pooled_win_rate=len(wins) / len(pnls) if pnls else None, pooled_profit_factor=pf,
    )


def write_universe_csv(report: UniverseReport, path: Path) -> None:
    """Save per-symbol rows and agent verdicts as two CSV files (``path`` and ``*_agents.csv``)."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in report.results:
        row = {k: v for k, v in r.__dict__.items() if k not in ("trade_pnls", "ablation", "hits")}
        for a, (dr, ds) in r.ablation.items():
            row[f"d_ret_{a}"] = dr
            row[f"d_sharpe_{a}"] = ds
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame([v.__dict__ for v in report.verdicts]).to_csv(path.with_name(path.stem + "_agents.csv"), index=False)
