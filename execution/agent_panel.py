"""Agent panel: every agent's view on the Nifty and on each stock, logged daily and scored live.

The panel is ADVISORY. It does not drive the paper trades (those follow the Nifty filter only).
Its purpose is a clean forward test: every day each agent (and the boss that combines them)
states BUY / SELL / HOLD on data it has never seen; after ``horizon`` sessions we check whether
the price moved the way it said. An agent earns a vote in the strategy only if it beats the
simple baseline on this live record (rule to be fixed before the review).

Scoring rule (fixed now, before any live data):
* only BUY and SELL calls are scored (HOLD and "no data" are not calls);
* outcome = close ``horizon`` sessions later / close on the call day - 1 (no costs);
* a BUY is right if the outcome > 0, a SELL is right if the outcome < 0;
* baseline = share of all matured (stock, day) pairs that went up, i.e. what "always BUY" scores.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agents import build_agents
from agents.boss_orchestrator import BossOrchestrator
from core.config import FrameworkConfig
from core.feature_bus import compute_features
from data.feed import MarketData

PANEL_FIELDS = ["as_of", "symbol", "close", "agent", "signal", "confidence", "available", "summary"]
AGENT_LABELS = {
    "market_intel": "Market flow",
    "risk_engine": "Risk",
    "liquidity_engine": "Liquidity",
    "market_mood": "Market mood",
    "sentiment_engine": "Sentiment",
    "regime_detector": "Trend",
    "on_chain": "On-chain",
    "boss": "BOSS (combined)",
}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _inputs(features: dict) -> dict:
    """Compact numeric inputs for the dashboard tooltip."""
    out = {}
    for k, v in (features or {}).items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v
        else:
            n = _num(v)
            if n is not None:
                out[k] = round(n, 4)
    return out


def explain(agent: str, signal: str, available: bool, f: dict, cfg: FrameworkConfig) -> str:
    """Plain-English reason built from the agent's real inputs (no new judgement is added)."""
    if not available:
        return "No view: the data this agent needs is not available for this symbol."
    g = lambda k: _num(f.get(k))  # noqa: E731
    if agent == "regime_detector":
        ef, es, adx, rv = g("ema_fast"), g("ema_slow"), g("adx"), g("realized_vol")
        reg = f.get("regime", "")
        p = cfg.agents.regime_detector
        if None in (ef, es, adx):
            return "Trend inputs incomplete."
        gap = ef / es - 1 if es else 0.0
        side = "below" if ef < es else "above"
        if reg == "trend_down":
            return (f"Bearish: the 20-day average ({ef:,.2f}) is {abs(gap):.1%} {side} the 100-day average ({es:,.2f}) "
                    f"and the trend is strong (ADX {adx:.0f} >= {p.adx_trend:.0f}).")
        if reg == "trend_up":
            return (f"Bullish: the 20-day average ({ef:,.2f}) is {abs(gap):.1%} {side} the 100-day average ({es:,.2f}) "
                    f"and the trend is strong (ADX {adx:.0f} >= {p.adx_trend:.0f}).")
        if reg == "volatile":
            return (f"Neutral: prices are too jumpy to trust the trend (realised volatility "
                    f"{(rv or 0):.0%} >= {p.volatile_rv:.0%}).")
        return (f"Neutral: no clear trend - ADX {adx:.0f} is below {p.adx_trend:.0f} "
                f"(20-day average {abs(gap):.1%} {side} the 100-day).")
    if agent == "market_intel":
        mom, cvd, sc = g("inst_momentum"), g("cvd_norm"), g("score")
        th = cfg.agents.market_intel.signal_threshold
        bits = []
        if mom is not None:
            bits.append(f"recent momentum is {'positive' if mom > 0 else 'negative'} ({mom:+.2f})")
        if cvd is not None:
            bits.append(f"buying-vs-selling pressure (estimated from price ticks) is {cvd:+.2f}")
        lead = {"BUY": "Bullish", "SELL": "Bearish"}.get(signal, "Neutral")
        tail = (f"combined score {sc:+.2f} is beyond +/-{th:.2f}" if signal != "HOLD"
                else f"combined score {(sc or 0):+.2f} is inside +/-{th:.2f}")
        return f"{lead}: " + " and ".join(bits) + f"; {tail}. (NSE has no real order-flow data, so this is an estimate.)"
    if agent == "market_mood":
        bits = []
        t, r, vx, vp = g("index_trend"), g("rel_strength"), g("india_vix"), g("vix_pct")
        if t is not None:
            bits.append(f"the Nifty is {'above' if t > 0 else 'below'} its 100-day average")
        if r is not None:
            bits.append(f"the stock is {'beating' if r > 0 else 'lagging'} the Nifty over 3 months")
        if vx is not None and vp is not None:
            bits.append(f"India VIX is {vx:.1f} ({vp:.0%} of the past year's readings were lower)")
        lead = {"BUY": "Positive mood", "SELL": "Negative mood"}.get(signal, "Mixed mood")
        return f"{lead}: " + "; ".join(bits) + f" (score {(g('score') or 0):+.2f})."
    if agent == "risk_engine":
        rs, rv, dd = g("risk_score"), g("realized_vol"), g("market_drawdown")
        veto = cfg.risk.risk_veto_threshold
        level = "extreme" if (rs or 0) > veto else "elevated" if (rs or 0) > 0.5 else "moderate" if (rs or 0) > 0.25 else "low"
        return (f"Risk is {level} (score {(rs or 0):.2f}; volatility {(rv or 0):.0%}, fall from recent high "
                f"{abs(dd or 0):.1%}). This agent never votes buy/sell; above {veto:.2f} it blocks new trades.")
    if agent == "liquidity_engine":
        if signal == "HOLD":
            return "Neutral: no stop-run (a quick break of a recent high/low that reverses) was detected."
        return f"{'Bullish' if signal == 'BUY' else 'Bearish'}: price swept a recent {'low' if signal == 'BUY' else 'high'} and reversed."
    return ""


def boss_math(dec, cfg: FrameworkConfig) -> dict:
    """The numbers behind the boss decision, for the dashboard's Explain panel."""
    rows = [{"agent": c.agent_name, "signal": c.signal.value, "confidence": round(c.confidence, 3),
             "input": round(c.signed_input, 3), "weight": round(c.effective_weight, 3),
             "contribution": round(c.contribution, 3), "eligible": c.eligible} for c in dec.breakdown]
    return {"rows": rows, "S": round(dec.directional_score, 3), "agreement": round(dec.agreement, 3),
            "coverage": round(dec.coverage, 3), "strength": round(dec.confidence, 3),
            "min_confidence": cfg.strategy.min_confidence, "score_floor": cfg.strategy.directional_score_floor,
            "raw": dec.raw_action.value, "action": dec.action.value, "forced": dec.forced_hold_rule,
            "risk_score": None if dec.risk_score is None else round(dec.risk_score, 3), "risk_veto": dec.risk_veto,
            "regime": dec.regime.value}


def panel_for(cfg: FrameworkConfig, md: MarketData, warm_bars: int = 60) -> list[dict]:
    """Every agent's view at the last bar, plus the boss decision (boss smoothing warmed on recent bars)."""
    ff = compute_features(md, cfg)
    agents = build_agents(cfg)
    boss = BossOrchestrator(cfg, [a.name for a in agents])
    sigs, dec = [], None
    for i in range(max(0, len(ff) - warm_bars), len(ff)):
        sigs = [a.analyze(ff.state(i)) for a in agents]
        dec = boss.decide(sigs, timestamp=ff.timestamp(i))
    last = len(ff) - 1
    as_of, close = str(ff.index[last].date()), float(ff.column("close")[last])
    rows = [{"as_of": as_of, "symbol": md.symbol, "close": close, "agent": s.agent_name, "signal": s.signal.value,
             "confidence": round(float(s.confidence), 3), "available": bool(s.available),
             "summary": (s.summary or s.reason)[:140],
             "why": explain(s.agent_name, s.signal.value, bool(s.available), s.features, cfg),
             "inputs": _inputs(s.features)} for s in sigs]
    if dec is not None:
        m = boss_math(dec, cfg)
        if dec.action.value != "HOLD":
            why = (f"{dec.action.value}: the weighted vote is {dec.directional_score:+.2f} and strength "
                   f"{dec.confidence:.2f} clears the {m['min_confidence']:.2f} bar.")
        elif dec.forced_hold_rule:
            why = (f"HOLD: the vote leans {dec.raw_action.value if dec.raw_action.value != 'HOLD' else ('up' if dec.directional_score > 0 else 'down')}"
                   f" ({dec.directional_score:+.2f}) but it is held back by: "
                   + ", ".join({"min_confidence": f"strength {dec.confidence:.2f} below {m['min_confidence']:.2f}",
                                "directional_score_floor": f"vote smaller than {m['score_floor']:.2f}",
                                "risk_veto": "risk veto", "risk_unavailable": "risk not measurable",
                                "data_coverage": "too little data", "kill_switch": "kill switch"}.get(r, r)
                               for r in dec.forced_hold_rule.split(";")) + ".")
        else:
            why = "HOLD: the agents do not lean either way."
        rows.append({"as_of": as_of, "symbol": md.symbol, "close": close, "agent": "boss", "signal": dec.action.value,
                     "confidence": round(float(dec.confidence), 3), "available": True,
                     "summary": (dec.forced_hold_rule or f"score {dec.directional_score:+.2f}, agreement {dec.agreement:.0%}")[:140],
                     "why": why, "math": m})
    return rows


def append_panel(rows: list[dict], path: Path) -> bool:
    """Append rows for a new close date; returns False if that date is already logged."""
    if not rows:
        return False
    day = rows[0]["as_of"]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            if any(r["as_of"] == day for r in csv.DictReader(f)):
                return False
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=PANEL_FIELDS, extrasaction="ignore")
        if new:
            wr.writeheader()
        wr.writerows(rows)
    return True


@dataclass
class AgentScore:
    agent: str
    calls: int
    buys: int
    sells: int
    hit_rate: float | None
    avg_signed_return: float | None  # mean of outcome for BUY, minus outcome for SELL


@dataclass
class Scorecard:
    horizon: int
    matured_days: int
    baseline_up_share: float | None
    scores: list[AgentScore]


def score_panel(path: Path, closes: dict[str, pd.Series], horizon: int = 20) -> Scorecard:
    """Score logged calls whose outcome is known (``horizon`` sessions have passed)."""
    if not path.exists():
        return Scorecard(horizon, 0, None, [])
    log = pd.read_csv(path)
    outcome = {}
    for (sym, day), _ in log.groupby(["symbol", "as_of"]):
        s = closes.get(sym)
        if s is None:
            continue
        dates = pd.Index([d.date().isoformat() for d in s.index])
        pos = dates.get_indexer([day])[0]
        if pos < 0 or pos + horizon >= len(s):
            continue
        outcome[(sym, day)] = float(s.iloc[pos + horizon] / s.iloc[pos] - 1.0)
    days = {d for _, d in outcome}
    base = float(np.mean([v > 0 for v in outcome.values()])) if outcome else None
    scores = []
    for agent in [a for a in AGENT_LABELS if a in set(log["agent"])]:
        sub = log[(log["agent"] == agent) & (log["signal"].isin(["BUY", "SELL"]))]
        hits, signed, nb, ns = [], [], 0, 0
        for r in sub.itertuples():
            o = outcome.get((r.symbol, r.as_of))
            if o is None:
                continue
            d = 1.0 if r.signal == "BUY" else -1.0
            nb += r.signal == "BUY"
            ns += r.signal == "SELL"
            hits.append(d * o > 0)
            signed.append(d * o)
        scores.append(AgentScore(agent, len(hits), nb, ns, float(np.mean(hits)) if hits else None,
                                 float(np.mean(signed)) if signed else None))
    return Scorecard(horizon, len(days), base, scores)


def format_panel(rows: list[dict], card: Scorecard, index_symbol: str = "NIFTY50", top: int = 5) -> str:
    """Advisory panel section of the daily report."""
    out = ["== AGENT PANEL (advisory only - does not change the paper trades) =="]
    nifty = [r for r in rows if r["symbol"] == index_symbol]
    if nifty:
        out.append(f"  Views on the Nifty 50:")
        for r in nifty:
            view = r["signal"] if r["available"] else "no data"
            out.append(f"    {AGENT_LABELS.get(r['agent'], r['agent']):<17}{view:<9}{r['confidence']:>5.2f}  {r['summary']}")
    stocks = [r for r in rows if r["symbol"] != index_symbol]
    if stocks:
        out += ["", f"  Views on the {len({r['symbol'] for r in stocks})} stocks:   {'BUY':>5}{'SELL':>6}{'HOLD':>6}{'no data':>9}"]
        for a in AGENT_LABELS:
            sub = [r for r in stocks if r["agent"] == a]
            if not sub:
                continue
            nd = sum(not r["available"] for r in sub)
            c = lambda x: sum(r["available"] and r["signal"] == x for r in sub)  # noqa: E731
            out.append(f"    {AGENT_LABELS[a]:<26}{c('BUY'):>5}{c('SELL'):>6}{c('HOLD'):>6}{nd:>9}")
        boss = [r for r in stocks if r["agent"] == "boss"]
        for side in ("BUY", "SELL"):
            pick = sorted([r for r in boss if r["signal"] == side], key=lambda r: -r["confidence"])[:top]
            if pick:
                out.append(f"    Boss strongest {side}: " + ", ".join(f"{r['symbol']} ({r['confidence']:.2f})" for r in pick))
    out += ["", f"  LIVE SCORECARD ({card.horizon}-session outcome, calls made since the panel started):"]
    if not any(s.calls for s in card.scores):
        out.append(f"    no calls have matured yet - the first scores appear {card.horizon} trading sessions after the first panel day")
    else:
        base = "n/a" if card.baseline_up_share is None else f"{card.baseline_up_share:.0%}"
        out.append(f"    matured days {card.matured_days}; baseline (share of stock-days that went up) {base}")
        out.append(f"    {'agent':<17}{'calls':>7}{'BUY':>6}{'SELL':>6}{'right':>8}{'avg result':>12}")
        for s in card.scores:
            hr = "n/a" if s.hit_rate is None else f"{s.hit_rate:.0%}"
            av = "n/a" if s.avg_signed_return is None else f"{s.avg_signed_return:+.2%}"
            out.append(f"    {AGENT_LABELS.get(s.agent, s.agent):<17}{s.calls:>7}{s.buys:>6}{s.sells:>6}{hr:>8}{av:>12}")
        out.append("    A few weeks of calls are mostly noise; judge only at the 8-12 week review.")
    return "\n".join(out)
