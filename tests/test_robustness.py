"""Step 5: cash interest, Indian capital-gains tax book, robustness runner (offline)."""

import os
import unittest

from helpers import ROOT

import numpy as np
import pandas as pd

from core.config import load_config
from execution.portfolio_test import StockSeries, TaxRules, _TaxBook, simulate
from execution.robustness import filter_config, run_robustness
from data.feed import SyntheticFeed

NSE = os.path.join(ROOT, "config_nse.yaml")
IST = "Asia/Kolkata"


def cfg(fee=0.0, slip=0.0, warm=2):
    return load_config(NSE, {"execution": {"fee_bps": fee, "slippage_bps": slip}, "data": {"warmup_bars": warm}})


def series(closes, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes), tz=IST) + pd.Timedelta(hours=9, minutes=15)
    c = np.asarray(closes, float)
    return StockSeries("X", idx, c, c, np.ones(len(c)))


class TestTaxBook(unittest.TestCase):
    rules = TaxRules(0.20, 0.125, 100.0, 0.0, 365)

    def test_short_term_gain_taxed_at_stcg(self):
        b = _TaxBook(1, self.rules)
        b.buy(0, 0, 1000.0, 10.0)
        b.sell(0, 100, 500.0, 20.0)  # sells 25 units bought at 10 -> gain 250, short term
        self.assertAlmostEqual(b.close_year(), 250 * 0.20)

    def test_long_term_gain_exemption_and_fifo(self):
        b = _TaxBook(1, self.rules)
        b.buy(0, 0, 1000.0, 10.0)      # 100 units @10 (old lot)
        b.buy(0, 300, 2000.0, 20.0)    # 100 units @20 (new lot)
        b.sell(0, 400, 3000.0, 30.0)   # FIFO: 100 old units (LT gain 2000, 400 days), no new units
        self.assertAlmostEqual(b.close_year(), (2000 - 100) * 0.125)

    def test_losses_offset_and_carry_forward(self):
        b = _TaxBook(2, self.rules)
        b.buy(0, 0, 1000.0, 10.0)
        b.sell(0, 10, 500.0, 5.0)      # short-term loss 500
        self.assertEqual(b.close_year(), 0.0)
        b.buy(1, 20, 1000.0, 10.0)
        b.sell(1, 30, 1500.0, 15.0)    # short-term gain 500, offset by carried loss
        self.assertAlmostEqual(b.close_year(), 0.0)

    def test_long_term_loss_cannot_offset_short_term_gain(self):
        b = _TaxBook(2, self.rules)
        b.buy(0, 0, 1000.0, 10.0)
        b.sell(0, 400, 500.0, 5.0)     # long-term loss 500
        b.buy(1, 390, 1000.0, 10.0)
        b.sell(1, 395, 1500.0, 15.0)   # short-term gain 500
        self.assertAlmostEqual(b.close_year(), 500 * 0.20)

    def test_cess_applied(self):
        b = _TaxBook(1, TaxRules(0.20, 0.125, 0.0, 0.04, 365))
        b.buy(0, 0, 100.0, 10.0)
        b.sell(0, 1, 200.0, 20.0)
        self.assertAlmostEqual(b.close_year(), 100 * 0.20 * 1.04)


class TestRealism(unittest.TestCase):
    def test_cash_yield_on_idle_cash(self):
        s = series([100] * 260)
        off = pd.Series(0.0, index=s.index)
        c = cfg()
        r = simulate(c, [s], off, s.index[3], "C_market", cash_yield=0.10)
        n = len(r.equity)
        self.assertAlmostEqual(r.equity.iloc[-1], 100_000 * 1.10 ** (n / c.bars_per_year), places=4)
        self.assertGreater(r.interest, 0)

    def test_tax_paid_after_financial_year_end(self):
        # price doubles in Feb, position sold by the filter in Mar (short-term), tax due in April
        idx = pd.bdate_range("2024-01-01", "2024-05-31", tz=IST) + pd.Timedelta(hours=9, minutes=15)
        px = np.where(idx < pd.Timestamp("2024-02-01", tz=IST), 100.0, 200.0)
        s = StockSeries("X", idx, px, px, np.ones(len(idx)))
        m = pd.Series(np.where(idx < pd.Timestamp("2024-03-01", tz=IST), 1.0, 0.0), index=idx)
        rules = TaxRules(0.20, 0.125, 0.0, 0.0, 365)
        r = simulate(cfg(), [s], m, idx[3], "C_market", tax=rules)
        self.assertAlmostEqual(r.taxes_paid, 100_000 * 0.20, places=2)
        before = r.equity[r.equity.index < pd.Timestamp("2024-04-01", tz=IST)].iloc[-1]
        self.assertAlmostEqual(before, 200_000, places=2)
        self.assertAlmostEqual(r.equity.iloc[-1], 180_000, places=2)

    def test_after_tax_exit_value_for_hold(self):
        s = series([100, 100, 100, 100, 150])
        rules = TaxRules(0.20, 0.125, 0.0, 0.0, 365)
        r = simulate(cfg(), [s], None, s.index[3], "A_hold", tax=rules)
        self.assertAlmostEqual(r.equity.iloc[-1], 150_000, places=2)
        self.assertAlmostEqual(r.after_tax_exit_value, 150_000 - 50_000 * 0.20, places=2)

    def test_defaults_unchanged(self):
        s = series([100, 101, 99, 105, 107, 103, 111])
        a = simulate(cfg(), [s], None, s.index[3], "A_hold")
        self.assertEqual(a.taxes_paid, 0.0)
        self.assertEqual(a.interest, 0.0)


class TestRunner(unittest.TestCase):
    def test_filter_config_copies(self):
        c = cfg()
        f = filter_config(c, 50, 200, 25)
        self.assertEqual((f.indicators.ema_fast, f.indicators.ema_slow), (50, 200))
        self.assertEqual(f.agents.regime_detector.adx_trend, 25.0)
        self.assertEqual(c.indicators.ema_fast, 20)
        self.assertGreater(f.agents.regime_detector.adx_strong, 25.0)

    def test_runs_end_to_end_on_synthetic(self):
        c = load_config(NSE)
        feed = SyntheticFeed(c)
        start = pd.Timestamp("2017-01-02", tz=IST)
        end = pd.Timestamp("2019-12-31", tz=IST)

        def loader(sym):
            return feed.get_bars(sym, start.to_pydatetime(), end.to_pydatetime())

        c2 = c.model_copy(deep=True)
        c2.designs.split_date = "2018-07-01"
        c2.designs.ema_grid = [(20, 100), (50, 200)]
        c2.designs.adx_grid = [20.0]
        rep = run_robustness(c2, ["AAA", "BBB", "CCC"], loader, start, end)
        self.assertEqual(len(rep.realism), 3)
        self.assertEqual(len(rep.settings), 2)
        self.assertEqual(len(rep.periods), 2)
        self.assertEqual(rep.realism[0].a.taxes_paid, 0.0)
        self.assertIsInstance(rep.passed, bool)
        from reporting import format_robustness
        txt = format_robustness(rep, "synthetic")
        self.assertIn("OVERALL", txt)


if __name__ == "__main__":
    unittest.main()
