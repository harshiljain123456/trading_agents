"""Agent panel: views per agent + boss, append-only log, live scoring rule."""

import os
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT

import pandas as pd

from core.config import load_config
from data.feed import SyntheticFeed
from execution.agent_panel import append_panel, format_panel, panel_for, score_panel, PANEL_FIELDS

NSE = os.path.join(ROOT, "config_nse.yaml")
IST = "Asia/Kolkata"


def closes(values, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=len(values), tz=IST) + pd.Timedelta(hours=9, minutes=15)
    return pd.Series(values, index=idx, dtype=float)


class TestPanel(unittest.TestCase):
    def test_panel_has_every_agent_and_boss(self):
        cfg = load_config(NSE)
        md = SyntheticFeed(cfg).get_bars("AAA", pd.Timestamp("2019-01-01", tz=IST).to_pydatetime(),
                                         pd.Timestamp("2020-01-01", tz=IST).to_pydatetime())
        rows = panel_for(cfg, md)
        agents = {r["agent"] for r in rows}
        self.assertEqual(agents, {"market_intel", "risk_engine", "liquidity_engine", "market_mood", "regime_detector", "boss"})
        self.assertTrue(all(set(PANEL_FIELDS) <= set(r) for r in rows))
        self.assertTrue(all(r["why"] for r in rows))
        boss = [r for r in rows if r["agent"] == "boss"][0]
        self.assertEqual({x["agent"] for x in boss["math"]["rows"]}, agents - {"boss"})
        self.assertIn("min_confidence", boss["math"])
        self.assertTrue(all(r["signal"] in ("BUY", "SELL", "HOLD") for r in rows))
        self.assertIn("AGENT PANEL", format_panel(rows, score_panel(Path("/nonexistent"), {})))

    def test_append_is_once_per_day(self):
        p = Path(tempfile.mkdtemp()) / "panel.csv"
        row = {"as_of": "2026-01-01", "symbol": "X", "close": 1.0, "agent": "boss", "signal": "BUY",
               "confidence": 0.5, "available": True, "summary": "s"}
        self.assertTrue(append_panel([row], p))
        self.assertFalse(append_panel([row], p))
        self.assertEqual(len(pd.read_csv(p)), 1)

    def test_scoring_rule(self):
        p = Path(tempfile.mkdtemp()) / "panel.csv"
        s_up = closes([100 + i for i in range(30)])        # rises
        s_dn = closes([100 - i for i in range(30)])        # falls
        day = str(s_up.index[0].date())
        late = str(s_up.index[25].date())                  # not matured (needs 20 more sessions)
        rows = [
            {"as_of": day, "symbol": "UP", "close": 100, "agent": "regime_detector", "signal": "BUY", "confidence": 1, "available": True, "summary": ""},
            {"as_of": day, "symbol": "DN", "close": 100, "agent": "regime_detector", "signal": "BUY", "confidence": 1, "available": True, "summary": ""},
            {"as_of": day, "symbol": "DN", "close": 100, "agent": "risk_engine", "signal": "SELL", "confidence": 1, "available": True, "summary": ""},
            {"as_of": day, "symbol": "UP", "close": 100, "agent": "risk_engine", "signal": "HOLD", "confidence": 1, "available": True, "summary": ""},
        ]
        append_panel(rows, p)
        append_panel([dict(rows[0], as_of=late)], p)
        card = score_panel(p, {"UP": s_up, "DN": s_dn}, horizon=20)
        sc = {s.agent: s for s in card.scores}
        self.assertEqual(sc["regime_detector"].calls, 2)          # late call not matured
        self.assertAlmostEqual(sc["regime_detector"].hit_rate, 0.5)
        self.assertEqual(sc["risk_engine"].calls, 1)              # HOLD not scored
        self.assertAlmostEqual(sc["risk_engine"].hit_rate, 1.0)
        self.assertAlmostEqual(sc["risk_engine"].avg_signed_return, 20 / 100)
        self.assertAlmostEqual(card.baseline_up_share, 0.5)
        self.assertEqual(card.matured_days, 1)


if __name__ == "__main__":
    unittest.main()
