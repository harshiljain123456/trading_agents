"""Step 5: robustness test of option C (equal-weight basket + Nifty crash filter) against A (hold).

Three questions, each with a pass rule fixed BEFORE running on real data:

1. Settings   - does C still beat A when the Nifty trend filter uses nearby settings?
                Grid: EMA pairs ``designs.ema_grid`` x ADX thresholds ``designs.adx_grid``.
                PASS if C beats A on BOTH Sharpe and worst fall in at least
                ``designs.param_pass_share`` (75%) of the settings.
2. Periods    - does it work in each half separately (start .. split_date, split_date .. end),
                each half started fresh with new capital, base setting?
                PASS if C beats A on BOTH Sharpe and worst fall in EVERY half.
3. Real money - idle cash earns a liquid-fund rate and capital-gains tax is paid each
                financial year. All of 1 and 2 are judged with this realism switched ON.
                The "plain" and "cash interest only" rows are shown for comparison only.

Overall: C passes only if 1 and 2 both pass. This is still in-sample on survivorship-affected
data; passing means "worth paper trading", not "profitable".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from core.config import FrameworkConfig
from data.feed import MarketData
from execution.evaluator import compute_metrics
from execution.portfolio_test import OptionResult, StockSeries, TaxRules, market_state, simulate

logger = logging.getLogger(__name__)


def _beats(c: OptionResult, a: OptionResult) -> bool:
    sc, sa = c.metrics.sharpe, a.metrics.sharpe
    return sc is not None and sa is not None and sc > sa and c.metrics.max_drawdown < a.metrics.max_drawdown


@dataclass
class Row:
    """One A-vs-C comparison."""

    label: str
    a: OptionResult
    c: OptionResult
    c_invested: float = 0.0

    @property
    def c_wins(self) -> bool:
        return _beats(self.c, self.a)


@dataclass
class RobustnessReport:
    symbols_used: list[str]
    symbols_skipped: dict[str, str]
    capital: float
    realism: list[Row] = field(default_factory=list)   # plain / +cash interest / +interest and tax (base setting)
    settings: list[Row] = field(default_factory=list)  # grid, full realism
    periods: list[Row] = field(default_factory=list)   # halves, full realism, base setting
    settings_pass: bool = False
    periods_pass: bool = False
    settings_wins: int = 0
    pass_share: float = 0.75

    @property
    def passed(self) -> bool:
        return self.settings_pass and self.periods_pass


def price_series(md: MarketData) -> StockSeries:
    """Prices only (options A and C do not need the per-stock trend path)."""
    df = md.bars
    return StockSeries(md.symbol, df.index, df["open"].to_numpy(float), df["close"].to_numpy(float),
                       pd.Series(0.0, index=df.index).to_numpy())


def filter_config(cfg: FrameworkConfig, ema_fast: int, ema_slow: int, adx: float) -> FrameworkConfig:
    """Copy of cfg with a different Nifty trend-filter setting (everything else unchanged)."""
    c = cfg.model_copy(deep=True)
    c.indicators.ema_fast, c.indicators.ema_slow = int(ema_fast), int(ema_slow)
    rd = c.agents.regime_detector
    rd.adx_trend = float(adx)
    if rd.adx_strong <= rd.adx_trend:
        rd.adx_strong = rd.adx_trend + 10.0
    return c


def _setting_label(f: int, s: int, adx: float) -> str:
    return f"EMA {f}/{s}, ADX {adx:g}"


def run_robustness(
    cfg: FrameworkConfig,
    symbols: list[str],
    loader: Callable[[str], MarketData],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    index_symbol: str = "NIFTY50",
) -> RobustnessReport:
    d = cfg.designs
    stocks: list[StockSeries] = []
    skipped: dict[str, str] = {}
    for i, sym in enumerate(symbols):
        if progress:
            progress(i + 1, len(symbols), sym)
        try:
            stocks.append(price_series(loader(sym)))
        except Exception as exc:  # one bad symbol must not abort the test
            skipped[sym] = f"{type(exc).__name__}: {exc}"
    if not stocks:
        raise ValueError("no symbols could be loaded for the robustness test")
    nifty = loader(index_symbol)  # required: the filter IS the test
    tax = TaxRules.from_config(cfg)
    cap = d.robustness_capital
    tz = cfg.app.timezone
    split = pd.Timestamp(d.split_date, tz=tz) if pd.Timestamp(d.split_date).tzinfo is None else pd.Timestamp(d.split_date)

    def run(option: str, mkt, s, e, yld, tx) -> OptionResult:
        r = simulate(cfg, stocks, mkt, s, option, end=e, cash_yield=yld, tax=tx, capital=cap)
        r.metrics = compute_metrics(r.equity, [], r.invested, r.costs, 0.0, r.traded, cfg.bars_per_year, 1.0)
        return r

    base_f, base_s = cfg.indicators.ema_fast, cfg.indicators.ema_slow
    base_adx = cfg.agents.regime_detector.adx_trend
    markets: dict[tuple[int, int, float], pd.Series] = {}

    def mkt_for(f: int, s: int, a: float) -> pd.Series:
        key = (int(f), int(s), float(a))
        if key not in markets:
            markets[key] = market_state(filter_config(cfg, f, s, a), nifty)
        return markets[key]

    rep = RobustnessReport([s.symbol for s in stocks], skipped, cap, pass_share=d.param_pass_share)
    base_m = mkt_for(base_f, base_s, base_adx)

    # 3. realism ladder on the base setting, full period
    for label, yld, tx in (("plain (as Step 3c)", 0.0, None),
                           (f"+ cash earns {d.cash_yield:.1%}", d.cash_yield, None),
                           ("+ cash interest + tax", d.cash_yield, tax)):
        a = run("A_hold", None, start, end, yld, tx)
        c = run("C_market", base_m, start, end, yld, tx)
        rep.realism.append(Row(label, a, c, float(c.invested.mean())))

    # 1. settings grid, full realism (A does not depend on the filter)
    a_full = rep.realism[-1].a
    for f, s in d.ema_grid:
        for adx in d.adx_grid:
            c = run("C_market", mkt_for(f, s, adx), start, end, d.cash_yield, tax)
            rep.settings.append(Row(_setting_label(f, s, adx), a_full, c, float(c.invested.mean())))
    rep.settings_wins = sum(r.c_wins for r in rep.settings)
    rep.settings_pass = rep.settings_wins >= d.param_pass_share * len(rep.settings)

    # 2. split periods, full realism, base setting, fresh capital in each half
    halves = []
    if start < split and (end is None or split < end):
        halves = [(f"{start.date()} to {split.date()}", start, split), (f"{split.date()} to {'end' if end is None else end.date()}", split, end)]
    for label, s, e in halves:
        a = run("A_hold", None, s, e, d.cash_yield, tax)
        c = run("C_market", base_m, s, e, d.cash_yield, tax)
        rep.periods.append(Row(label, a, c, float(c.invested.mean())))
    rep.periods_pass = bool(rep.periods) and all(r.c_wins for r in rep.periods)
    return rep


def write_robustness_csv(rep: RobustnessReport, path) -> None:
    rows = []
    for section, items in (("realism", rep.realism), ("settings", rep.settings), ("periods", rep.periods)):
        for r in items:
            for opt in (r.a, r.c):
                m = opt.metrics
                rows.append({"section": section, "case": r.label, "option": opt.name, "final_value": m.final_equity,
                             "cagr": m.cagr, "sharpe": m.sharpe, "max_drawdown": m.max_drawdown,
                             "taxes_paid": opt.taxes_paid, "interest": opt.interest, "costs": opt.costs,
                             "after_tax_exit_value": opt.after_tax_exit_value,
                             "invested_avg": float(opt.invested.mean()), "c_beats_a": r.c_wins})
    pd.DataFrame(rows).to_csv(path, index=False)
