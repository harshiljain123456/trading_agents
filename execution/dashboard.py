"""Build the self-contained web dashboard (site/index.html) from a daily signal run.

The page needs no server and no external scripts: all data is embedded as JSON and the
charts are drawn with inline SVG. It is safe to publish (paper data only, no account data).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from execution.agent_panel import AGENT_LABELS

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dashboard_data(rep, paper_dir: Path, universe: str, index_symbol: str = "NIFTY50") -> dict:
    panel = rep.panel or []
    stocks: dict[str, dict] = {}
    for r in panel:
        if r["symbol"] == index_symbol:
            continue
        s = stocks.setdefault(r["symbol"], {"close": r["close"], "views": {}})
        s["views"][r["agent"]] = [r["signal"] if r["available"] else "n/a", r["confidence"], r["summary"]]
    card = rep.scorecard
    pnl = rep.paper_value / rep.capital - 1.0 if rep.capital else 0.0
    return {
        "universe": universe,
        "updated": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d-%b-%Y %H:%M IST"),
        "as_of": f"{rep.as_of:%a %d-%b-%Y}",
        "next_session": f"{rep.next_session:%a %d-%b}",
        "filter": "INVESTED" if rep.filter_now > 0 else "IN CASH",
        "filter_changed": rep.filter_now != rep.filter_prev,
        "action": rep.action,
        "nifty": {"close": rep.nifty_close, "ema_fast": rep.ema_fast, "ema_slow": rep.ema_slow, "adx": rep.adx,
                  "regime": rep.regime_summary},
        "paper": {"start": f"{rep.paper_start:%d-%b-%Y}", "capital": rep.capital, "value": rep.paper_value,
                  "ret": pnl, "invested": rep.paper_invested, "cash": rep.paper_cash, "tax": rep.taxes_paid,
                  "sessions": rep.sessions_traded},
        "orders": [asdict(o) for o in rep.orders],
        "warnings": list(rep.warnings),
        "skipped": list(rep.symbols_skipped),
        "nifty_hist": rep.nifty_hist,
        "paper_hist": rep.paper_hist,
        "agent_labels": AGENT_LABELS,
        "panel_nifty": [r for r in panel if r["symbol"] == index_symbol],
        "panel_stocks": stocks,
        "scorecard": None if card is None else {
            "horizon": card.horizon, "matured_days": card.matured_days, "baseline": card.baseline_up_share,
            "scores": [asdict(s) for s in card.scores]},
        "journal": _read_csv(paper_dir / "journal.csv")[::-1],
    }


def build_dashboard(rep, paper_dir: Path, out_dir: Path, universe: str) -> Path:
    """Write ``out_dir/index.html`` and ``out_dir/data.json``; returns the HTML path."""
    data = dashboard_data(rep, paper_dir, universe)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, default=str, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    (out_dir / "data.json").write_text(json.dumps(data, default=str, indent=1), encoding="utf-8")
    return path
