"""Hand-calculated tests for indicators/order_flow.py."""

import math
import unittest

import helpers  # noqa: F401  (sets sys.path)
import numpy as np
import pandas as pd

from indicators.order_flow import (
    cvd, cvd_change_normalized, institutional_momentum, order_flow_imbalance, price_volume_correlation,
)


class TestOFI(unittest.TestCase):
    def test_hand_calculated(self):
        # window 2: t1 = (1+2 - (1+0)) / (3+1) = 0.5 ; t2 = (2+3 - (0+1)) / 6 = 0.6667
        o = order_flow_imbalance(pd.Series([1.0, 2, 3]), pd.Series([1.0, 0, 1]), 2)
        self.assertTrue(math.isnan(o.iloc[0]))
        self.assertAlmostEqual(o.iloc[1], 0.5)
        self.assertAlmostEqual(o.iloc[2], 4 / 6)

    def test_sign_convention_and_bounds(self):
        self.assertEqual(order_flow_imbalance(pd.Series([5.0]), pd.Series([0.0]), 1).iloc[0], 1.0)
        self.assertEqual(order_flow_imbalance(pd.Series([0.0]), pd.Series([5.0]), 1).iloc[0], -1.0)
        self.assertEqual(order_flow_imbalance(pd.Series([3.0]), pd.Series([3.0]), 1).iloc[0], 0.0)

    def test_zero_volume_is_nan(self):
        self.assertTrue(math.isnan(order_flow_imbalance(pd.Series([0.0, 0]), pd.Series([0.0, 0]), 2).iloc[1]))

    def test_missing_value_propagates_nan(self):
        o = order_flow_imbalance(pd.Series([1.0, np.nan, 1.0]), pd.Series([1.0, 1.0, 1.0]), 2)
        self.assertTrue(math.isnan(o.iloc[1]) and math.isnan(o.iloc[2]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            order_flow_imbalance(pd.Series([-1.0]), pd.Series([1.0]), 1)
        with self.assertRaises(ValueError):
            order_flow_imbalance(pd.Series([1.0]), pd.Series([1.0]), 0)
        with self.assertRaises(ValueError):
            order_flow_imbalance(pd.Series([1.0, 2]), pd.Series([1.0]), 1)

    def test_causal(self):
        b = pd.Series(np.arange(1.0, 11))
        a = pd.Series(np.ones(10))
        full = order_flow_imbalance(b, a, 3)
        b2 = b.copy()
        b2.iloc[-1] = 1000.0
        part = order_flow_imbalance(b2, a, 3)
        pd.testing.assert_series_equal(full.iloc[:-1], part.iloc[:-1])


class TestCVD(unittest.TestCase):
    def test_with_side(self):
        c = cvd(pd.Series([1.0, 1, 1]), pd.Series([10.0, 20, 30]), pd.Series([1.0, -0.5, np.nan]))
        self.assertEqual(list(c), [10.0, 0.0, 0.0])  # NaN side contributes zero

    def test_tick_rule_proxy(self):
        # signs: [0, +1, +1 (carried through unchanged price), -1] -> cumsum of [0, 2, 3, -4]
        c = cvd(pd.Series([10.0, 11, 11, 10]), pd.Series([1.0, 2, 3, 4]), None)
        self.assertEqual(list(c), [0.0, 2.0, 5.0, 1.0])

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            cvd(pd.Series([1.0]), pd.Series([1.0]), pd.Series([1.5]))

    def test_normalized_change(self):
        cv = pd.Series([0.0, 10.0, 0.0])
        n = cvd_change_normalized(cv, pd.Series([10.0, 10.0, 10.0]), 1)
        self.assertAlmostEqual(n.iloc[1], 1.0)
        self.assertAlmostEqual(n.iloc[2], -1.0)
        z = cvd_change_normalized(cv, pd.Series([0.0, 0.0, 0.0]), 1)
        self.assertTrue(math.isnan(z.iloc[1]))


class TestInstitutionalMomentum(unittest.TestCase):
    def test_constant_price_is_zero(self):
        s = institutional_momentum(pd.Series([100.0] * 10), pd.Series([1.0] * 10), 3)
        self.assertEqual(s.iloc[-1], 0.0)

    def test_zero_variance_trend(self):
        # constant 1% steps -> sigma 0, R > 0 -> sign 1; equal volume -> factor 0.5
        p = pd.Series(100 * 1.01 ** np.arange(10))
        s = institutional_momentum(p, pd.Series([1.0] * 10), 3)
        self.assertAlmostEqual(s.iloc[-1], 0.5)

    def test_insufficient_history(self):
        s = institutional_momentum(pd.Series(100 * 1.01 ** np.arange(10)), pd.Series([1.0] * 10), 3)
        self.assertTrue(s.iloc[:5].isna().all())
        self.assertFalse(math.isnan(s.iloc[5]))

    def test_matches_formula(self):
        rng = np.random.default_rng(0)
        p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))))
        v = pd.Series(rng.uniform(1, 2, 30))
        L = 5
        s = institutional_momentum(p, v, L)
        t = 29
        lp = np.log(p.to_numpy())
        R = lp[t] - lp[t - L]
        sig = np.std(np.diff(lp)[t - L : t], ddof=1)
        vr = v.iloc[t - L + 1 : t + 1].mean() / v.iloc[t - 2 * L + 1 : t - L + 1].mean()
        expect = math.tanh(R / (sig * math.sqrt(L))) * min(vr, 2) / 2
        self.assertAlmostEqual(s.iloc[t], expect, places=10)

    def test_zero_prior_volume_nan(self):
        s = institutional_momentum(pd.Series(100 * 1.01 ** np.arange(6)), pd.Series([0.0, 0, 0, 1, 1, 1]), 3)
        self.assertTrue(math.isnan(s.iloc[5]))

    def test_reproducible(self):
        p = pd.Series(100 * 1.01 ** np.arange(12))
        v = pd.Series(np.arange(1.0, 13))
        pd.testing.assert_series_equal(institutional_momentum(p, v, 3), institutional_momentum(p, v, 3))


class TestPVCorr(unittest.TestCase):
    def test_constant_is_nan(self):
        c = price_volume_correlation(pd.Series([100.0] * 10), pd.Series([1.0] * 10), 3)
        self.assertTrue(c.isna().all())

    def test_perfect_positive(self):
        p = pd.Series(np.exp(np.cumsum([0, 0.01, 0.02, 0.03, 0.04, 0.05])))
        v = pd.Series(np.exp(np.cumsum([0, 0.01, 0.02, 0.03, 0.04, 0.05])))
        self.assertAlmostEqual(price_volume_correlation(p, v, 3).iloc[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
