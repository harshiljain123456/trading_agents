"""Hand-calculated tests for indicators/volatility.py and indicators/trend.py."""

import math
import unittest

import helpers  # noqa: F401
import numpy as np
import pandas as pd

from indicators.trend import adx, ema
from indicators.volatility import (
    atr, atr_percent, garman_klass_volatility, parkinson_volatility, realized_volatility, rolling_drawdown,
    true_range, validate_ohlc, wilder_smooth,
)


class TestRealizedVol(unittest.TestCase):
    def test_hand_calculated(self):
        rv = realized_volatility(pd.Series([100.0, 110.0, 99.0]), 2, 1)
        r1, r2 = math.log(1.1), math.log(0.9)
        self.assertAlmostEqual(rv.iloc[2], abs(r1 - r2) / math.sqrt(2))
        self.assertTrue(math.isnan(rv.iloc[1]))

    def test_annualization(self):
        rv1 = realized_volatility(pd.Series([100.0, 110.0, 99.0]), 2, 1).iloc[2]
        rv4 = realized_volatility(pd.Series([100.0, 110.0, 99.0]), 2, 4).iloc[2]
        self.assertAlmostEqual(rv4, 2 * rv1)

    def test_constant_zero_variance(self):
        self.assertEqual(realized_volatility(pd.Series([5.0] * 5), 3, 365).iloc[-1], 0.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            realized_volatility(pd.Series([1.0, 0.0, 1.0]), 2, 1)
        with self.assertRaises(ValueError):
            realized_volatility(pd.Series([1.0, 2.0]), 1, 1)


class TestATR(unittest.TestCase):
    def setUp(self):
        self.h = pd.Series([10.0, 11, 12, 13])
        self.l = pd.Series([9.0, 10, 11, 12])
        self.c = pd.Series([9.5, 10.5, 11.5, 12.5])

    def test_true_range(self):
        self.assertEqual(list(true_range(self.h, self.l, self.c)), [1.0, 1.5, 1.5, 1.5])

    def test_wilder_hand_calculated(self):
        a = atr(self.h, self.l, self.c, 2)
        self.assertTrue(math.isnan(a.iloc[0]))
        self.assertAlmostEqual(a.iloc[1], 1.25)
        self.assertAlmostEqual(a.iloc[2], 1.375)
        self.assertAlmostEqual(a.iloc[3], 1.4375)
        self.assertAlmostEqual(atr_percent(self.h, self.l, self.c, 2).iloc[3], 1.4375 / 12.5)

    def test_wilder_insufficient(self):
        self.assertTrue(wilder_smooth(pd.Series([1.0]), 3).isna().all())


class TestRangeEstimators(unittest.TestCase):
    def test_parkinson(self):
        low = pd.Series([10.0, 10.0])
        high = low * math.e
        v = parkinson_volatility(high, low, 2, 4).iloc[1]
        self.assertAlmostEqual(v, math.sqrt(1 / (4 * math.log(2))) * 2)

    def test_garman_klass(self):
        low = pd.Series([10.0, 10.0])
        high = low * math.e
        o = c = pd.Series([15.0, 15.0])
        self.assertAlmostEqual(garman_klass_volatility(o, high, low, c, 2, 1).iloc[1], math.sqrt(0.5))

    def test_zero_range_is_zero(self):
        p = pd.Series([10.0, 10.0, 10.0])
        self.assertEqual(parkinson_volatility(p, p, 2, 1).iloc[-1], 0.0)
        self.assertEqual(garman_klass_volatility(p, p, p, p, 2, 1).iloc[-1], 0.0)


class TestDrawdownAndValidation(unittest.TestCase):
    def test_drawdown(self):
        d = rolling_drawdown(pd.Series([100.0, 120.0, 90.0]), 3)
        self.assertAlmostEqual(d.iloc[2], -0.25)
        self.assertTrue(math.isnan(d.iloc[1]))

    def test_validate_ohlc(self):
        o = pd.Series([10.0])
        with self.assertRaises(ValueError):
            validate_ohlc(o, pd.Series([9.0]), pd.Series([8.0]), pd.Series([10.0]))
        with self.assertRaises(ValueError):
            validate_ohlc(o, pd.Series([11.0]), pd.Series([10.5]), pd.Series([10.0]))
        validate_ohlc(o, pd.Series([11.0]), pd.Series([9.0]), pd.Series([10.5]))


class TestTrend(unittest.TestCase):
    def test_ema_warmup_and_constant(self):
        e = ema(pd.Series([5.0] * 6), 3)
        self.assertTrue(e.iloc[:2].isna().all())
        self.assertAlmostEqual(e.iloc[-1], 5.0)

    def test_adx_strong_uptrend(self):
        n = 60
        c = pd.Series(100 + np.arange(n, dtype=float))
        out = adx(c + 0.5, c - 0.5, c, 14)
        self.assertTrue(out["adx"].iloc[: 2 * 14 - 1].isna().all())
        self.assertFalse(math.isnan(out["adx"].iloc[2 * 14 - 1]))
        self.assertGreater(out["adx"].iloc[-1], 90)
        self.assertGreater(out["plus_di"].iloc[-1], out["minus_di"].iloc[-1])

    def test_adx_flat_zero(self):
        c = pd.Series([100.0] * 40)
        out = adx(c + 1, c - 1, c, 5)
        self.assertEqual(out["adx"].iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
