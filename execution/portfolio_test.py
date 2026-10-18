"""Step 3c: portfolio-level test on an equal-weight basket of stocks.

Three options, all starting with the same capital on the same dates and paying
the same costs (fee + slippage on every rupee traded):

* ``A_hold``    - equal-weight basket of every eligible stock, rebalanced to equal
                  weights on the first trading day of each month (index-fund-like).
* ``B_trend``   - same slots, but each stock's slot is invested only while the
                  per-stock trend rule (the Step 3 "trend only" design, with risk
                  sizing and veto) says so; otherwise that slot sits in cash.
* ``C_market``  - basket A, but everything goes to cash while the Nifty 50 itself
                  is in a downtrend (trend agent on the index: EMA20 < EMA100 with
                  ADX >= 20) and returns when the index trend turns up.

Timing is causal and matches the rest of the framework: a decision taken at
day t's close is executed at day t+1's open. Overnight gaps are earned by the
positions held before the trade, the day's move by the positions after it.

Eligibility (identical for all options): a stock joins the basket once it has
``warmup_bars`` sessions of history, so the trend rule is defined for it; new
stocks enter (and all slots are re-equalised) at the next monthly rebalance. Missing
prices (suspensions, gaps) earn zero return and are not traded.

Winner rule (fixed before running): highest portfolio Sharpe; an option counts as
better than A only if it beats A on BOTH Sharpe and maximum drawdown.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from agents.regime_detector import RegimeDetectorAgent
from core.config import FrameworkConfig
from core.feature_bus import compute_features
from core.models import SignalType
from data.feed import MarketData
from execution.designs import design_rule, window_return
from execution.evaluator import compute_metrics

logger = logging.getLogger(__name__)

OPTIONS: tuple[str, ...] = ("A_hold", "B_trend", "C_market")


@dataclass
class StockSeries:
    """Per-stock prices plus the trend rule's target exposure decided at each close."""

    symbol: str
    index: pd.DatetimeIndex
    open: np.ndarray
    close: np.ndarray
    trend_target: np.ndarray


def stock_series(cfg: FrameworkConfig, md: MarketData) -> StockSeries:
    """Compute the per-stock trend-rule target path (causal state machine, no price drift)."""
    ff = compute_features(md, cfg)
    rule = design_rule(cfg, ff, "trend_only")
    tgt = np.zeros(len(ff))
    cur = 0.0
    for i in range(len(ff)):
        new = rule(i, cur)
        if new is not None:
            cur = new
        tgt[i] = cur
    return StockSeries(md.symbol, ff.index, ff.column("open").copy(), ff.column("close").copy(), tgt)


def market_state(cfg: FrameworkConfig, md: MarketData) -> pd.Series:
    """1.0 = Nifty in uptrend/neutral (invested), 0.0 = downtrend (cash); range/volatile keep the last state."""
    ff = compute_features(md, cfg)
    agent = RegimeDetectorAgent(1.0, cfg.agents.regime_detector)
    state, out = 1.0, np.ones(len(ff))
    for i in range(len(ff)):
        s = agent.analyze(ff.state(i))
        if s.available and s.signal == SignalType.SELL:
            state = 0.0
        elif s.available and s.signal == SignalType.BUY:
            state = 1.0
        out[i] = state
    return pd.Series(out, index=ff.index)


@dataclass
class TaxRules:
    """Indian listed-equity capital-gains rules (current rates applied to every year; see assumptions)."""

    stcg_rate: float = 0.20
    ltcg_rate: float = 0.125
    ltcg_exemption: float = 125_000.0
    cess: float = 0.04
    holding_days: int = 365

    @classmethod
    def from_config(cls, cfg: FrameworkConfig) -> "TaxRules":
        d = cfg.designs
        return cls(d.stcg_rate, d.ltcg_rate, d.ltcg_exemption, d.tax_cess, d.ltcg_holding_days)


class _TaxBook:
    """FIFO lots per stock, realised gains per financial year (April-March), loss carry-forward.

    Set-off: short-term losses offset short- or long-term gains; long-term losses offset only
    long-term gains; unused losses carry forward (the 8-year limit is ignored).
    """

    def __init__(self, n: int, rules: TaxRules):
        self.rules = rules
        self.lots: list[list[list[float]]] = [[] for _ in range(n)]  # [entry_day_ordinal, units, cost_per_unit]
        self.st = self.lt = 0.0
        self.cf_st = self.cf_lt = 0.0
        self.paid = 0.0

    def buy(self, j: int, day: int, rupees: float, price: float) -> None:
        if rupees > 0 and price > 0:
            self.lots[j].append([day, rupees / price, price])

    def _gain(self, j: int, day: int, units: float, price: float) -> tuple[float, float]:
        st = lt = 0.0
        lots = self.lots[j]
        while units > 1e-12 and lots:
            lot = lots[0]
            take = min(units, lot[1])
            g = take * (price - lot[2])
            if day - lot[0] >= self.rules.holding_days:
                lt += g
            else:
                st += g
            lot[1] -= take
            units -= take
            if lot[1] <= 1e-12:
                lots.pop(0)
        return st, lt

    def sell(self, j: int, day: int, rupees: float, price: float) -> None:
        if rupees > 0 and price > 0:
            st, lt = self._gain(j, day, rupees / price, price)
            self.st += st
            self.lt += lt

    def _tax(self, st: float, lt: float, cf_st: float, cf_lt: float) -> tuple[float, float, float]:
        r = self.rules
        if st < 0 and lt > 0:  # current-year short-term loss offsets long-term gain
            use = min(-st, lt)
            st, lt = st + use, lt - use
        if lt < 0:
            cf_lt, lt = cf_lt - lt, 0.0
        if st < 0:
            cf_st, st = cf_st - st, 0.0
        use = min(cf_lt, lt)
        lt, cf_lt = lt - use, cf_lt - use
        use = min(cf_st, st)
        st, cf_st = st - use, cf_st - use
        use = min(cf_st, lt)
        lt, cf_st = lt - use, cf_st - use
        tax = (st * r.stcg_rate + max(0.0, lt - r.ltcg_exemption) * r.ltcg_rate) * (1 + r.cess)
        return tax, cf_st, cf_lt

    def close_year(self) -> float:
        """Tax due for the financial year just ended; resets the year's gains."""
        tax, self.cf_st, self.cf_lt = self._tax(self.st, self.lt, self.cf_st, self.cf_lt)
        self.st = self.lt = 0.0
        self.paid += tax
        return tax

    def liquidation_tax(self, day: int, prices: np.ndarray) -> float:
        """Tax if everything were sold at ``prices`` today (current-year gains included); no state change."""
        st, lt = self.st, self.lt
        for j, lots in enumerate(self.lots):
            p = prices[j]
            if np.isnan(p):
                continue
            for d0, u, c in lots:
                g = u * (p - c)
                if day - d0 >= self.rules.holding_days:
                    lt += g
                else:
                    st += g
        return self._tax(st, lt, self.cf_st, self.cf_lt)[0]


@dataclass
class OptionResult:
    """Simulated equity path and statistics for one option."""

    name: str
    equity: pd.Series
    invested: pd.Series
    costs: float
    traded: float
    n_stocks_avg: float
    metrics: object = None
    crash: dict[str, float | None] = field(default_factory=dict)
    yearly: dict[int, float] = field(default_factory=dict)
    taxes_paid: float = 0.0
    interest: float = 0.0
    after_tax_exit_value: float | None = None  # equity minus exit costs and tax if all sold on the last day
    final_holdings: np.ndarray | None = None   # rupee value per stock at the last close
    final_cash: float = 0.0


@dataclass
class PortfolioReport:
    """All options plus the pre-registered verdict."""

    options: list[OptionResult]
    symbols_used: list[str]
    symbols_skipped: dict[str, str]
    winner: str
    beats_hold: list[str]


def _fin_year(ts: pd.Timestamp) -> int:
    return ts.year if ts.month >= 4 else ts.year - 1


def simulate(
    cfg: FrameworkConfig,
    stocks: list[StockSeries],
    market: pd.Series | None,
    start: pd.Timestamp,
    option: str,
    end: pd.Timestamp | None = None,
    cash_yield: float = 0.0,
    tax: TaxRules | None = None,
    capital: float | None = None,
) -> OptionResult:
    """Run one option over the union calendar from ``start`` (to ``end``, exclusive, if given).

    ``cash_yield``: annual rate earned by positive idle cash, accrued per session.
    ``tax``: if given, capital-gains tax on realised gains is paid from cash on the first session
    of each new financial year (the monthly rebalance that day re-equalises the basket).
    """
    cal = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in stocks])))
    n = len(stocks)
    O = np.full((len(cal), n), np.nan)
    C = np.full((len(cal), n), np.nan)
    TT = np.zeros((len(cal), n))
    ELIG = np.zeros((len(cal), n), dtype=bool)
    warm = cfg.data.warmup_bars
    for j, s in enumerate(stocks):
        pos = cal.get_indexer(s.index)
        O[pos, j] = s.open
        C[pos, j] = s.close
        tt = pd.Series(s.trend_target, index=s.index).reindex(cal).ffill().fillna(0.0).to_numpy()
        TT[:, j] = tt
        count = np.zeros(len(cal))
        count[pos] = 1
        ELIG[:, j] = np.cumsum(count) >= warm
    M = np.ones(len(cal))
    if market is not None:
        M = market.reindex(cal, method="ffill").fillna(1.0).to_numpy()
    cost_rate = (cfg.execution.fee_bps + cfg.execution.slippage_bps) / 1e4
    daily_yield = (1.0 + cash_yield) ** (1.0 / cfg.bars_per_year) - 1.0
    t0 = int(np.searchsorted(cal, start))
    if t0 < 1:
        t0 = 1
    t_end = len(cal) if end is None else int(np.searchsorted(cal, end))
    if t_end <= t0:
        raise ValueError("simulation window is empty")
    cash, h = float(capital if capital is not None else cfg.execution.initial_capital), np.zeros(n)
    book = _TaxBook(n, tax) if tax is not None else None
    ordinal = np.array([ts.toordinal() for ts in cal])
    costs = traded = interest = 0.0
    eq_vals, inv_vals, nst = [], [], []
    last_close = C[t0 - 1].copy()
    for t in range(t0, t_end):
        # carry the last known close forward for stocks without a print today
        prev = np.where(np.isnan(C[t - 1]), last_close, C[t - 1])
        last_close = prev
        if cash > 0 and daily_yield > 0:
            earned = cash * daily_yield
            cash += earned
            interest += earned
        if book is not None and t > t0 and _fin_year(cal[t]) != _fin_year(cal[t - 1]):
            cash -= book.close_year()
        o = O[t]
        has_open = ~np.isnan(o) & ~np.isnan(prev) & (prev > 0)
        gap = np.where(has_open, o / np.where(has_open, prev, 1.0), 1.0)
        h = h * gap
        # targets decided at the previous close
        elig = ELIG[t - 1] & has_open
        k = int(elig.sum())
        slot = 1.0 / k if k else 0.0
        if option == "A_hold":
            w = np.where(elig, slot, 0.0)
        elif option == "B_trend":
            w = np.where(elig, slot * TT[t - 1], 0.0)
        elif option == "C_market":
            w = np.where(elig, slot * M[t - 1], 0.0)
        else:
            raise ValueError(f"unknown option {option}")
        new_month = t == t0 or cal[t].month != cal[t - 1].month  # first day: initial purchase
        if option == "B_trend":
            switched = has_open & ((TT[t - 1] > 0) != (TT[t - 2] > 0 if t >= 2 else TT[t - 1] > 0))
        elif option == "C_market":
            switched = has_open & (t >= 2 and M[t - 1] != M[t - 2])
        else:
            switched = np.zeros(n, dtype=bool)
        trade_mask = (has_open & new_month) | switched | (has_open & (h > 0) & ~elig)
        if trade_mask.any():
            equity = cash + h.sum()
            desired = w * equity
            delta = np.where(trade_mask, desired - h, 0.0)
            if book is not None:
                for j in np.flatnonzero(delta):
                    if delta[j] > 0:
                        book.buy(j, ordinal[t], delta[j], o[j])
                    else:
                        book.sell(j, ordinal[t], -delta[j], o[j])
            c = np.abs(delta).sum() * cost_rate
            h = h + delta
            cash -= delta.sum() + c
            costs += c
            traded += np.abs(delta).sum()
        cl = C[t]
        has_close = ~np.isnan(cl) & ~np.isnan(o)
        intraday = np.where(has_close, cl / np.where(has_close, o, 1.0), 1.0)
        h = h * intraday
        equity = cash + h.sum()
        eq_vals.append(equity)
        inv_vals.append(h.sum() / equity if equity > 0 else 0.0)
        nst.append(k)
    eq = pd.Series(eq_vals, index=cal[t0:t_end])
    inv = pd.Series(inv_vals, index=cal[t0:t_end])
    res = OptionResult(option, eq, inv, costs, traded, float(np.mean(nst)) if nst else 0.0, interest=interest,
                       final_holdings=h.copy(), final_cash=cash)
    exit_cost = h.sum() * cost_rate
    if book is not None:
        res.taxes_paid = book.paid
        last = np.where(np.isnan(C[t_end - 1]), last_close, C[t_end - 1])
        res.after_tax_exit_value = float(eq.iloc[-1] - exit_cost - book.liquidation_tax(ordinal[t_end - 1], last))
    else:
        res.after_tax_exit_value = float(eq.iloc[-1] - exit_cost)
    return res


def _finish(cfg: FrameworkConfig, r: OptionResult) -> None:
    """Attach metrics, crash-window and calendar-year returns."""
    r.metrics = compute_metrics(r.equity, [], r.invested, r.costs, 0.0, r.traded, cfg.bars_per_year, 1.0)
    tz = cfg.app.timezone
    r.crash = {k: window_return(r.equity, a, b, tz) for k, (a, b) in cfg.designs.crash_windows.items()}
    years = r.equity.groupby(r.equity.index.year)
    prev = None
    for y, s in years:
        base = prev if prev is not None else s.iloc[0]
        r.yearly[int(y)] = float(s.iloc[-1] / base - 1.0)
        prev = s.iloc[-1]


def run_portfolio_test(
    cfg: FrameworkConfig,
    symbols: list[str],
    loader: Callable[[str], MarketData],
    start: pd.Timestamp,
    progress: Callable[[int, int, str], None] | None = None,
    jobs: int = 1,
    index_symbol: str = "NIFTY50",
) -> PortfolioReport:
    """Load every stock, compute trend paths (in parallel), simulate the three options."""
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    futs: dict[str, object] = {}
    series: dict[str, StockSeries] = {}
    skipped: dict[str, str] = {}
    try:
        for i, sym in enumerate(symbols):
            if progress:
                progress(i + 1, len(symbols), sym)
            try:
                md = loader(sym)
            except Exception as exc:  # one bad symbol must not abort the test
                skipped[sym] = f"{type(exc).__name__}: {exc}"
                continue
            if pool is None:
                try:
                    series[sym] = stock_series(cfg, md)
                except Exception as exc:
                    skipped[sym] = f"{type(exc).__name__}: {exc}"
            else:
                futs[sym] = pool.submit(stock_series, cfg, md)
        for sym, f in futs.items():
            try:
                series[sym] = f.result()  # type: ignore[attr-defined]
            except Exception as exc:
                skipped[sym] = f"{type(exc).__name__}: {exc}"
    finally:
        if pool is not None:
            pool.shutdown()
    if not series:
        raise ValueError("no symbols could be loaded for the portfolio test")
    try:
        mkt = market_state(cfg, loader(index_symbol))
    except Exception as exc:
        logger.warning("Nifty index unavailable (%s); option C will stay invested", exc)
        mkt = None
    stocks = [series[s] for s in symbols if s in series]
    results = []
    for opt in OPTIONS:
        r = simulate(cfg, stocks, mkt, start, opt)
        _finish(cfg, r)
        results.append(r)
    hold = results[0].metrics
    sh = lambda m: m.sharpe if m.sharpe is not None else -9.0  # noqa: E731
    winner = max(results, key=lambda r: (sh(r.metrics), -r.metrics.max_drawdown)).name
    beats = [r.name for r in results[1:] if sh(r.metrics) > sh(hold) and r.metrics.max_drawdown < hold.max_drawdown]
    return PortfolioReport(results, [s.symbol for s in stocks], skipped, winner, beats)
