"""Step 3 designs: rule behaviour, crash windows, winner rule, end-to-end (offline)."""

import os
import unittest

from helpers import ROOT, make_state

import pandas as pd

from core.config import load_config
from core.feature_bus import compute_features
from data.feed import SyntheticFeed
from execution.backtester import Backtester
from execution.designs import (
    DESIGNS, DesignRow, SymbolDesigns, design_rule, evaluate_designs, summarize, window_return,
)
from main import parse_market_date

NSE = os.path.join(ROOT, "config_nse.yaml")


def st(trend="up", rv=0.25, risk_high=False, mood="neutral"):
    ef, es, adx = {"up": (110, 100, 30), "down": (90, 100, 30), "range": (101, 100, 10)}[trend]
    ind = {"ema_fast": ef, "ema_slow": es, "adx": adx, "realized_vol": rv,
           "atr_pct": 0.1 if risk_high else 0.01, "market_drawdown": -0.5 if risk_high else 0.0,
           "vix_pct": 0.5, "vix_change": 0.0, "india_vix": 15.0}
    ind["index_trend"], ind["rel_strength"] = {"neutral": (0.1, 0.0), "bear": (-0.8, -0.8)}[mood]
    if risk_high:
        ind["realized_vol"] = 3.0
    return make_state(ind)


class FakeFF:
    def __init__(self, states):
        self.states = states

    def state(self, i):
        return self.states[i]


class TestRules(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(NSE)

    def rule(self, design, states):
        return design_rule(self.cfg, FakeFF(states), design)

    def test_trend_only_enters_exits_and_holds(self):
        r = self.rule("trend_only", [st("up"), st("range"), st("down")])
        self.assertAlmostEqual(r(0, 0.0), 1.0)          # vol 25% < 30% target -> full size
        self.assertIsNone(r(1, 1.0))                     # range: keep position
        self.assertIsNone(r(1, 0.0))                     # range: stay out
        self.assertEqual(r(2, 1.0), 0.0)                 # downtrend: exit

    def test_risk_sizing(self):
        r = self.rule("trend_only", [st("up", rv=0.40)])
        self.assertAlmostEqual(r(0, 0.0), 0.75)          # 0.30 / 0.40 (rv below the 0.45 'volatile' regime line)

    def test_rebalance_band(self):
        r = self.rule("trend_only", [st("up", rv=0.32)])
        self.assertIsNone(r(0, 1.0))                     # 0.9375 vs 1.0 -> inside the 10% band

    def test_defensive_default_invested_and_steps_aside(self):
        r = self.rule("defensive", [st("range"), st("down"), st("up", mood="bear")])
        self.assertAlmostEqual(r(0, 0.0), 1.0)           # invested by default
        self.assertEqual(r(1, 1.0), 0.0)                 # downtrend: exit
        self.assertAlmostEqual(r(2, 1.0), self.cfg.designs.mood_cap)  # bearish market: cap

    def test_risk_veto_blocks_entry_and_caps(self):
        r = self.rule("defensive", [st("up", risk_high=True)])
        self.assertIsNone(r(0, 0.0))                     # flat stays flat
        r2 = self.rule("trend_only", [st("up", risk_high=True)])
        self.assertAlmostEqual(r2(0, 1.0), self.cfg.designs.veto_cap)

    def test_unknown_design(self):
        with self.assertRaises(ValueError):
            design_rule(self.cfg, FakeFF([]), "ensemble")


class TestWindowAndWinner(unittest.TestCase):
    def test_window_return(self):
        idx = pd.date_range("2019-06-01", periods=300, freq="D", tz="Asia/Kolkata")
        eq = pd.Series(range(100, 400), index=idx, dtype=float)
        r = window_return(eq, "2020-01-17", "2020-03-23", "Asia/Kolkata")
        self.assertAlmostEqual(r, eq.loc[:"2020-03-23"].iloc[-1] / eq.loc[:"2020-01-17"].iloc[-1] - 1)
        self.assertIsNone(window_return(eq, "2008-01-08", "2008-10-27", "Asia/Kolkata"))
        late = eq[eq.index >= "2020-01-01"]   # strategy started only ~2 weeks before the crash
        self.assertIsNone(window_return(late, "2020-01-17", "2020-03-23", "Asia/Kolkata"))

    def test_winner_rule(self):
        def row(sh):
            return DesignRow(0.1, sh, 0.2, 3, 0.5, {"c": -0.1})
        syms = []
        for i in range(10):
            syms.append(SymbolDesigns(f"S{i}", 100, {"ensemble": row(0.2), "trend_only": row(0.6),
                                                     "defensive": row(0.7 if i < 3 else 0.4), "buy_and_hold": row(0.5)}))
        rep = summarize(syms, ["c"])
        by = {s.design: s for s in rep.summaries}
        self.assertEqual(by["trend_only"].beat_bh_sharpe, 10)
        self.assertEqual(by["defensive"].beat_bh_sharpe, 3)
        self.assertEqual(rep.winner, "trend_only")
        self.assertEqual(by["trend_only"].vs_ensemble_wins, 10)
        self.assertLess(by["trend_only"].vs_ensemble_p, 0.01)


class TestEndToEnd(unittest.TestCase):
    def test_evaluate_designs_on_synthetic(self):
        cfg = load_config(NSE)
        md = SyntheticFeed(cfg, seed=5).get_bars("T", parse_market_date("2018-01-01", cfg.app.timezone),
                                                 parse_market_date("2020-06-01", cfg.app.timezone))
        out = evaluate_designs(cfg, md)
        self.assertEqual(set(out.rows), set(DESIGNS) | {"buy_and_hold"})
        ff = compute_features(md, cfg)
        ens = Backtester(cfg, ff).run(ff.first_trade_index, len(ff)).metrics.total_return
        self.assertAlmostEqual(out.rows["ensemble"].ret, ens)
        for r in out.rows.values():
            self.assertTrue(0 <= r.expo <= 1.0 + 1e-9)
        self.assertIsNotNone(out.rows["buy_and_hold"].crash["2020 covid"])   # ~2 years of lead-in
        self.assertIsNone(out.rows["buy_and_hold"].crash["2008 crash"])      # outside the data


if __name__ == "__main__":
    unittest.main()
