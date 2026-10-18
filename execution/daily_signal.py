"""Step 7: daily paper-trading signal for option C (equal-weight basket + Nifty crash filter).

Run once per trading day AFTER the close (from ~4 pm IST). It:

1. downloads fresh daily bars for the basket and the Nifty 50 (a still-forming bar for today is
   dropped if the run happens before 15:45 IST),
2. reads the Nifty trend filter at the last completed close (base setting EMA 20/100, ADX 20 -
   NOT re-tuned after the Step 5 results),
3. replays the paper portfolio from its start date with the exact backtest rules (next-open
   fills, costs, cash interest, capital-gains tax), so the paper book cannot drift from the rules,
4. prints tomorrow's action and orders, and appends one line to an append-only journal.

Nothing is sent to any broker. The journal records what the signal said ON THE DAY; if Yahoo later
revises data, the replayed value can differ slightly from the journal - that is visible, not hidden.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from agents.regime_detector import RegimeDetectorAgent
from core.config import FrameworkConfig
from core.feature_bus import compute_features
from data.feed import MarketData
from execution.portfolio_test import TaxRules, market_state, simulate
from execution.robustness import price_series
from execution.agent_panel import append_panel, panel_for, score_panel

CLOSE_CUTOFF = (15, 45)  # IST: before this time today's bar is still forming
JOURNAL_FIELDS = ["run_time", "as_of_close", "nifty_close", "ema_fast", "ema_slow", "adx", "regime",
                  "filter", "action_next_session", "orders", "paper_value", "paper_invested", "paper_start"]


@dataclass
class Order:
    symbol: str
    side: str
    rupees: float
    approx_shares: int
    ref_price: float


@dataclass
class SignalReport:
    as_of: pd.Timestamp
    next_session: pd.Timestamp
    filter_now: float
    filter_prev: float
    nifty_close: float
    ema_fast: float
    ema_slow: float
    adx: float
    regime_summary: str
    action: str
    orders: list[Order]
    paper_start: pd.Timestamp
    capital: float
    paper_value: float
    paper_cash: float
    paper_invested: float
    sessions_traded: int
    taxes_paid: float
    symbols_used: list[str]
    symbols_skipped: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    panel: list[dict] = field(default_factory=list)   # advisory agent views (do not drive trades)
    scorecard: object = None
    nifty_hist: list[dict] = field(default_factory=list)  # for the dashboard chart
    paper_hist: list[dict] = field(default_factory=list)


def completed_bars(md: MarketData, now: pd.Timestamp) -> MarketData:
    """Drop today's bar if the market has not closed yet (``now`` in the market timezone)."""
    if md.bars.empty:
        return md
    last = md.bars.index[-1]
    still_open = last.date() == now.date() and (now.hour, now.minute) < CLOSE_CUTOFF
    return dataclasses.replace(md, bars=md.bars.iloc[:-1]) if still_open else md


def _load_state(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def run_daily_signal(
    cfg: FrameworkConfig,
    symbols: list[str],
    loader: Callable[[str], MarketData],
    paper_dir: Path,
    universe_label: str,
    now: pd.Timestamp | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    index_symbol: str = "NIFTY50",
    panel: bool = True,
) -> SignalReport:
    tz = cfg.app.timezone
    now = pd.Timestamp.now(tz=tz) if now is None else now.tz_convert(tz)
    warnings: list[str] = []
    stocks, skipped, mds = [], {}, []
    for i, sym in enumerate(symbols):
        if progress:
            progress(i + 1, len(symbols), sym)
        try:
            md = completed_bars(loader(sym), now)
            stocks.append(price_series(md))
            mds.append(md)
        except Exception as exc:  # one bad symbol must not stop the daily run
            skipped[sym] = f"{type(exc).__name__}: {exc}"
    if not stocks:
        raise ValueError("no symbols could be loaded for the daily signal")
    nifty = completed_bars(loader(index_symbol), now)
    mkt = market_state(cfg, nifty)
    ff = compute_features(nifty, cfg)
    last = len(ff) - 1
    sig = RegimeDetectorAgent(1.0, cfg.agents.regime_detector).analyze(ff.state(last))
    as_of = ff.index[last]
    if (now.normalize() - as_of.normalize()).days > 4:
        warnings.append(f"latest Nifty close is {as_of.date()} - data may be stale (holiday or download problem)")
    lagging = [s.symbol for s in stocks if s.index[-1].normalize() < as_of.normalize()]
    if lagging:
        warnings.append(f"{len(lagging)} stock(s) have no bar for {as_of.date()}: {', '.join(lagging[:8])}"
                        + (" ..." if len(lagging) > 8 else ""))

    # paper book: start date fixed on the first run, stored in paper_dir/state.json
    paper_dir.mkdir(parents=True, exist_ok=True)
    state_path = paper_dir / "state.json"
    state = _load_state(state_path)
    next_session = as_of + pd.offsets.BDay(1)  # exchange holidays are not known here
    if state is None:
        state = {"paper_start": str(next_session.date()), "capital": cfg.designs.robustness_capital,
                 "universe": universe_label, "created": str(now)}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    elif state.get("universe") != universe_label:
        warnings.append(f"paper book was started with universe '{state.get('universe')}', now running '{universe_label}'")
    paper_start = pd.Timestamp(state["paper_start"], tz=tz)
    capital = float(state["capital"])

    cal = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in stocks])))
    started = bool((cal >= paper_start).any())
    tax = TaxRules.from_config(cfg)
    if started:
        r = simulate(cfg, stocks, mkt, paper_start, "C_market", cash_yield=cfg.designs.cash_yield, tax=tax, capital=capital)
        h, cash, taxes, sessions = r.final_holdings, r.final_cash, r.taxes_paid, len(r.equity)
    else:
        h, cash, taxes, sessions = np.zeros(len(stocks)), capital, 0.0, 0
    equity = cash + float(h.sum())

    # tomorrow's targets, decided at today's close (same rules as the simulator)
    m_now = float(mkt.iloc[-1])
    m_prev = float(mkt.iloc[-2]) if len(mkt) > 1 else m_now
    elig = np.array([len(s.index) >= cfg.data.warmup_bars and s.index[-1].normalize() == as_of.normalize()
                     for s in stocks])
    k = int(elig.sum())
    w = np.where(elig, (1.0 / k if k else 0.0) * m_now, 0.0)
    if not started:
        action = "START: BUY THE BASKET" if m_now > 0 else "START: STAY IN CASH (Nifty filter is off)"
        trade = elig.copy()
    elif m_now != m_prev:
        action = "SWITCH: BUY THE BASKET (filter turned on)" if m_now > 0 else "SWITCH: SELL EVERYTHING (filter turned off)"
        trade = elig | (h > 0)
    elif next_session.month != as_of.month:
        action = "MONTHLY REBALANCE (back to equal weights)" if m_now > 0 else "NO TRADE (in cash, month change)"
        trade = elig | (h > 0)
    else:
        action = "NO TRADE (hold current positions)" if m_now > 0 else "NO TRADE (stay in cash)"
        trade = (h > 0) & ~elig  # only drop stocks that stopped printing
    delta = np.where(trade, w * equity - h, 0.0)
    orders = []
    for j in np.argsort(-np.abs(delta)):
        if abs(delta[j]) < 1.0:
            continue
        px = float(stocks[j].close[-1])
        orders.append(Order(stocks[j].symbol, "BUY" if delta[j] > 0 else "SELL", float(abs(delta[j])),
                            int(math.floor(abs(delta[j]) / px)) if px > 0 else 0, px))
    if orders and all(o.approx_shares == 0 for o in orders if o.side == "BUY"):
        warnings.append("order sizes are below one share at today's prices; consider a larger paper capital")

    rep = SignalReport(as_of, next_session, m_now, m_prev, float(ff.column("close")[last]),
                       float(ff.column("ema_fast")[last]), float(ff.column("ema_slow")[last]),
                       float(ff.column("adx")[last]), sig.summary, action, orders, paper_start, capital,
                       equity, cash, float(h.sum()) / equity if equity > 0 else 0.0, sessions, taxes,
                       [s.symbol for s in stocks], skipped, warnings)
    n0 = max(0, len(ff) - 260)
    cl, ef, es = ff.column("close"), ff.column("ema_fast"), ff.column("ema_slow")
    mk = mkt.reindex(ff.index).ffill().fillna(1.0).to_numpy()
    rep.nifty_hist = [{"d": str(ff.index[i].date()), "c": round(float(cl[i]), 2),
                       "f": None if np.isnan(ef[i]) else round(float(ef[i]), 2),
                       "s": None if np.isnan(es[i]) else round(float(es[i]), 2), "m": int(mk[i])}
                      for i in range(n0, len(ff))]
    if started:
        rep.paper_hist = [{"d": str(d.date()), "v": round(float(v), 2), "i": round(float(x), 4)}
                          for d, v, x in zip(r.equity.index, r.equity.to_numpy(), r.invested.to_numpy())]
    _write_journal(rep, paper_dir, now)
    if panel:
        rep.panel = _panel(cfg, [nifty] + mds, as_of, rep.warnings)
        append_panel(rep.panel, paper_dir / "agent_panel.csv")
        closes = {m.symbol: m.bars["close"] for m in [nifty] + mds}
        rep.scorecard = score_panel(paper_dir / "agent_panel.csv", closes, cfg.strategy.prediction_horizon)
    return rep


def _panel(cfg: FrameworkConfig, mds: list[MarketData], as_of: pd.Timestamp, warnings: list[str]) -> list[dict]:
    rows, failed = [], []
    for md in mds:
        if md.bars.index[-1].normalize() != as_of.normalize():
            continue  # no bar today: no view
        try:
            rows += panel_for(cfg, md)
        except Exception as exc:  # the advisory panel must never stop the signal
            failed.append(f"{md.symbol} ({type(exc).__name__})")
    if failed:
        warnings.append("agent panel skipped: " + ", ".join(failed[:6]))
    return rows


def _write_journal(rep: SignalReport, paper_dir: Path, now: pd.Timestamp) -> None:
    """Append one line per completed session (a re-run on the same close is not duplicated)."""
    path = paper_dir / "journal.csv"
    done = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            done = {row["as_of_close"] for row in csv.DictReader(f)}
    key = str(rep.as_of.date())
    if key not in done:
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow({"run_time": now.strftime("%Y-%m-%d %H:%M"), "as_of_close": key,
                         "nifty_close": round(rep.nifty_close, 2), "ema_fast": round(rep.ema_fast, 2),
                         "ema_slow": round(rep.ema_slow, 2), "adx": round(rep.adx, 1), "regime": rep.regime_summary,
                         "filter": "INVESTED" if rep.filter_now > 0 else "CASH", "action_next_session": rep.action,
                         "orders": len(rep.orders), "paper_value": round(rep.paper_value, 2),
                         "paper_invested": round(rep.paper_invested, 4), "paper_start": str(rep.paper_start.date())})
    if rep.orders:
        pd.DataFrame([dataclasses.asdict(o) for o in rep.orders]).to_csv(
            paper_dir / f"orders_for_{rep.next_session:%Y%m%d}.csv", index=False)
