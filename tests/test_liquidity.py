"""Hand-calculated tests for indicators/liquidity.py."""

import math
import unittest

import helpers  # noqa: F401
import pandas as pd

from indicators.liquidity import absorption_ratio, depth_imbalance, detect_sweeps


class TestSweeps(unittest.TestCase):
    def frame(self, last_low, last_high, last_close):
        h = pd.Series([11.0, 11.0, 11.0, last_high])
        l = pd.Series([10.0, 10.0, 10.0, last_low])
        c = pd.Series([10.5, 10.5, 10.5, last_close])
        return detect_sweeps(h, l, c, 3)

    def test_bullish_confirmed(self):
        s = self.frame(9.0, 10.8, 10.5)
        self.assertTrue(s["bullish_sweep"].iloc[3])
        self.assertFalse(s["bearish_sweep"].iloc[3])
        self.assertEqual(s["prior_low"].iloc[3], 10.0)

    def test_no_reversal_no_sweep(self):
        s = self.frame(9.0, 10.8, 9.5)
        self.assertFalse(s["bullish_sweep"].iloc[3])

    def test_bearish_confirmed(self):
        s = self.frame(10.2, 12.0, 10.6)
        self.assertTrue(s["bearish_sweep"].iloc[3])
        self.assertEqual(s["prior_high"].iloc[3], 11.0)

    def test_warmup_false(self):
        s = self.frame(9.0, 10.8, 10.5)
        self.assertFalse(s["bullish_sweep"].iloc[:3].any())
        self.assertTrue(s["prior_low"].iloc[:3].isna().all())

    def test_prior_level_excludes_current_and_is_causal(self):
        h = pd.Series([11.0, 11, 11, 10.8, 11])
        l = pd.Series([10.0, 10, 10, 9.0, 10])
        c = pd.Series([10.5, 10.5, 10.5, 10.5, 10.5])
        a = detect_sweeps(h, l, c, 3)
        l2 = l.copy()
        l2.iloc[4] = 1.0  # change only the future bar
        b = detect_sweeps(h, l2, c, 3)
        pd.testing.assert_frame_equal(a.iloc[:4], b.iloc[:4])


class TestAbsorption(unittest.TestCase):
    def test_hand_calculated(self):
        # range = [0.1, 0.1, 0.05]; rel_vol = 3; rel_range = 0.5 -> 6
        v = pd.Series([1.0, 1.0, 3.0])
        c = pd.Series([10.0, 10.0, 10.0])
        h = pd.Series([10.5, 10.5, 10.25])
        l = pd.Series([9.5, 9.5, 9.75])
        self.assertAlmostEqual(absorption_ratio(v, h, l, c, 2).iloc[2], 6.0)

    def test_zero_range_capped(self):
        v = pd.Series([1.0, 1.0, 2.0])
        c = pd.Series([10.0, 10.0, 10.0])
        h = pd.Series([10.5, 10.5, 10.0])
        l = pd.Series([9.5, 9.5, 10.0])
        self.assertAlmostEqual(absorption_ratio(v, h, l, c, 2, 0.1).iloc[2], 20.0)

    def test_zero_baseline_volume_nan(self):
        v = pd.Series([0.0, 0.0, 2.0])
        c = pd.Series([10.0] * 3)
        h = pd.Series([10.5] * 3)
        l = pd.Series([9.5] * 3)
        self.assertTrue(math.isnan(absorption_ratio(v, h, l, c, 2).iloc[2]))


class TestDepth(unittest.TestCase):
    def test_scalar(self):
        self.assertAlmostEqual(depth_imbalance(30.0, 10.0), 0.5)
        self.assertTrue(math.isnan(depth_imbalance(0.0, 0.0)))

    def test_series(self):
        d = depth_imbalance(pd.Series([10.0, 0.0]), pd.Series([30.0, 0.0]))
        self.assertAlmostEqual(d.iloc[0], -0.5)
        self.assertTrue(math.isnan(d.iloc[1]))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            depth_imbalance(-1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
