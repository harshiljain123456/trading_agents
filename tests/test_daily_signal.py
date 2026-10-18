"""Step 7: daily paper signal (offline, synthetic data, fixed clock)."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT

import pandas as pd

from core.config import load_config
from data.feed import SyntheticFeed
from execution.daily_signal import completed_bars, run_daily_signal

NSE = os.path.join(ROOT, "config_nse.yaml")
IST = "Asia/Kolkata"


class TestDailySignal(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(NSE)
        feed = SyntheticFeed(self.cfg)
        start, end = pd.Timestamp("2018-01-01", tz=IST), pd.Timestamp("2020-06-30", tz=IST)
        self.loader = lambda sym: feed.get_bars(sym, start.to_pydatetime(), end.to_pydatetime())
        self.tmp = Path(tempfile.mkdtemp())

    def test_drops_forming_bar_only_before_close(self):
        md = self.loader("AAA")
        last = md.bars.index[-1]
        during = last.normalize() + pd.Timedelta(hours=11)
        after = last.normalize() + pd.Timedelta(hours=16)
        self.assertEqual(len(completed_bars(md, during).bars), len(md.bars) - 1)
        self.assertEqual(len(completed_bars(md, after).bars), len(md.bars))
        self.assertEqual(len(completed_bars(md, last + pd.Timedelta(days=1)).bars), len(md.bars))

    def test_first_run_starts_book_and_orders_sum_to_capital(self):
        now = pd.Timestamp("2020-06-30 18:00", tz=IST)
        rep = run_daily_signal(self.cfg, ["AAA", "BBB", "CCC"], self.loader, self.tmp, "test", now=now)
        state = json.loads((self.tmp / "state.json").read_text())
        self.assertEqual(state["paper_start"], str(rep.next_session.date()))
        self.assertEqual(rep.sessions_traded, 0)
        self.assertEqual(rep.paper_value, self.cfg.designs.robustness_capital)
        if rep.filter_now > 0:
            self.assertTrue(rep.action.startswith("START: BUY"))
            self.assertAlmostEqual(sum(o.rupees for o in rep.orders), rep.capital, places=2)
            self.assertTrue(all(o.side == "BUY" for o in rep.orders))
        else:
            self.assertEqual(rep.orders, [])

    def test_journal_not_duplicated_and_book_replays(self):
        (self.tmp / "state.json").write_text(json.dumps(
            {"paper_start": "2020-01-01", "capital": 1_000_000.0, "universe": "test"}))
        now = pd.Timestamp("2020-06-30 18:00", tz=IST)
        r1 = run_daily_signal(self.cfg, ["AAA", "BBB"], self.loader, self.tmp, "test", now=now)
        r2 = run_daily_signal(self.cfg, ["AAA", "BBB"], self.loader, self.tmp, "test", now=now)
        self.assertGreater(r1.sessions_traded, 50)
        self.assertAlmostEqual(r1.paper_value, r2.paper_value)
        with (self.tmp / "journal.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0]["filter"], ("INVESTED", "CASH"))

    def test_universe_change_warns(self):
        (self.tmp / "state.json").write_text(json.dumps(
            {"paper_start": "2020-01-01", "capital": 1_000_000.0, "universe": "other"}))
        rep = run_daily_signal(self.cfg, ["AAA"], self.loader, self.tmp, "test",
                               now=pd.Timestamp("2020-06-30 18:00", tz=IST))
        self.assertTrue(any("universe" in w for w in rep.warnings))


if __name__ == "__main__":
    unittest.main()


class TestDashboard(unittest.TestCase):
    def test_dashboard_builds_with_embedded_data(self):
        from execution.dashboard import build_dashboard
        cfg = load_config(NSE)
        feed = SyntheticFeed(cfg)
        start, end = pd.Timestamp("2018-01-01", tz=IST), pd.Timestamp("2020-06-30", tz=IST)
        loader = lambda sym: feed.get_bars(sym, start.to_pydatetime(), end.to_pydatetime())  # noqa: E731
        tmp = Path(tempfile.mkdtemp())
        (tmp / "state.json").write_text(json.dumps({"paper_start": "2020-01-01", "capital": 1_000_000.0, "universe": "t"}))
        rep = run_daily_signal(cfg, ["AAA", "BBB"], loader, tmp, "t", now=pd.Timestamp("2020-06-30 18:00", tz=IST))
        self.assertGreater(len(rep.nifty_hist), 100)
        self.assertGreater(len(rep.paper_hist), 50)
        html = build_dashboard(rep, tmp, tmp / "site", "t").read_text(encoding="utf-8")
        self.assertNotIn("/*__DATA__*/null", html)
        data = json.loads((tmp / "site" / "data.json").read_text())
        self.assertEqual(set(data["panel_stocks"]), {"AAA", "BBB"})
        self.assertIn(data["filter"], ("INVESTED", "IN CASH"))
