"""Step 3c portfolio simulator: accounting, timing, costs and option logic (offline)."""

import os
import unittest

from helpers import ROOT

import numpy as np
import pandas as pd

from core.config import load_config
from execution.portfolio_test import StockSeries, simulate

NSE = os.path.join(ROOT, "config_nse.yaml")
IST = "Asia/Kolkata"


def cfg(fee=0.0, slip=0.0, warm=2):
    return load_config(NSE, {"execution": {"fee_bps": fee, "slippage_bps": slip}, "data": {"warmup_bars": warm}})


def series(sym, closes, opens=None, tt=None, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes), tz=IST) + pd.Timedelta(hours=9, minutes=15)
    c = np.asarray(closes, float)
    o = np.asarray(opens if opens is not None else c, float)
    t = np.asarray(tt if tt is not None else np.ones(len(c)), float)
    return StockSeries(sym, idx, o, c, t)


class TestSimulator(unittest.TestCase):
    def test_single_stock_hold_matches_price_path(self):
        closes = [100, 100, 100, 110, 121, 133.1]
        s = series("X", closes)
        start = s.index[3]  # eligible after 2 bars; first trade at open of bar 3
        r = simulate(cfg(), [s], None, start, "A_hold")
        self.assertAlmostEqual(r.equity.iloc[-1], 100_000 * 133.1 / 110, places=4)

    def test_trend_all_on_equals_hold_and_all_off_is_flat(self):
        closes = [100, 101, 99, 105, 107, 103, 111]
        a = simulate(cfg(), [series("X", closes)], None, series("X", closes).index[3], "A_hold")
        b = simulate(cfg(), [series("X", closes, tt=np.ones(7))], None, series("X", closes).index[3], "B_trend")
        off = simulate(cfg(), [series("X", closes, tt=np.zeros(7))], None, series("X", closes).index[3], "B_trend")
        pd.testing.assert_series_equal(a.equity, b.equity)
        self.assertTrue((off.equity == 100_000).all())

    def test_market_filter(self):
        closes = [100, 101, 99, 105, 107, 103, 111]
        s = series("X", closes)
        on = pd.Series(1.0, index=s.index)
        off = pd.Series(0.0, index=s.index)
        a = simulate(cfg(), [s], None, s.index[3], "A_hold")
        c_on = simulate(cfg(), [s], on, s.index[3], "C_market")
        c_off = simulate(cfg(), [s], off, s.index[3], "C_market")
        pd.testing.assert_series_equal(a.equity, c_on.equity)
        self.assertTrue((c_off.equity == 100_000).all())

    def test_decision_executes_next_open(self):
        # trend turns on at the close of bar 4 -> bought at the open of bar 5 (price 200), not at bar 4's close (150)
        closes = [100, 100, 100, 100, 150, 200, 220]
        opens = [100, 100, 100, 100, 100, 200, 210]
        tt = [0, 0, 0, 0, 1, 1, 1]
        s = series("X", closes, opens, tt)
        r = simulate(cfg(), [s], None, s.index[3], "B_trend")
        self.assertAlmostEqual(r.equity.iloc[-1], 100_000 * 220 / 200, places=4)
        self.assertEqual(r.equity.iloc[1], 100_000)   # bar 4: still in cash

    def test_costs_charged_on_trades(self):
        closes = [100] * 8
        s = series("X", closes)
        r = simulate(cfg(fee=10, slip=5), [s], None, s.index[3], "A_hold")
        self.assertAlmostEqual(r.costs, 100_000 * 0.0015, places=2)
        self.assertAlmostEqual(r.equity.iloc[-1], 100_000 - r.costs, places=2)

    def test_two_stocks_equal_weight(self):
        a = series("A", [100, 100, 100, 100, 200])
        b = series("B", [100, 100, 100, 100, 100])
        r = simulate(cfg(), [a, b], None, a.index[3], "A_hold")
        self.assertAlmostEqual(r.equity.iloc[-1], 150_000, places=4)

    def test_late_listing_joins_after_warmup(self):
        a = series("A", [100] * 40)                                   # 2024-01-01 .. late Feb
        late = series("L", [50] * 10 + [100] * 15, start="2024-01-22")  # eligible in Jan, joins at the Feb rebalance
        r = simulate(cfg(), [a, late], None, a.index[3], "A_hold")
        feb = r.equity[r.equity.index.month == 2]
        self.assertAlmostEqual(feb.iloc[0], 100_000)                 # joined at Feb's first open (price still 50)
        self.assertGreater(r.equity.iloc[-1], 100_000)               # then L doubled on its half of the money
        self.assertLess(r.equity.iloc[-1], 200_000)


if __name__ == "__main__":
    unittest.main()
