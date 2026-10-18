"""LiquidityEngineAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.liquidity_engine import LiquidityEngineAgent
from core.models import SignalType

BASE = {"prior_low": 95.0, "prior_high": 105.0, "absorption_ratio": 1.0}


def st(bull=0.0, bear=0.0, cvd=None, depth=None, ar=1.0, unavailable=()):
    ind = dict(BASE, bullish_sweep=bull, bearish_sweep=bear, absorption_ratio=ar)
    if cvd is not None:
        ind["cvd_norm_short"] = cvd
    if depth is not None:
        ind["depth_imbalance"] = depth
    return make_state(ind, unavailable=unavailable)


class TestLiquidityEngine(unittest.TestCase):
    def setUp(self):
        self.agent = LiquidityEngineAgent(1.0, config().agents.liquidity_engine)

    def test_bullish_confirmed_by_cvd_and_depth(self):
        s = self.agent.analyze(st(bull=1.0, cvd=0.3, depth=0.2))
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, 0.8)  # 0.4 + 0.2 cvd + 0.2 depth
        self.assertIn("95.00", s.reason)
        self.assertIn("not proof", s.reason)

    def test_absorption_adds_confidence(self):
        s = self.agent.analyze(st(bull=1.0, cvd=0.3, depth=0.2, ar=2.0))
        self.assertAlmostEqual(s.confidence, 1.0)
        self.assertIn("absorption of selling", s.reason)

    def test_bullish_cvd_divergence_hold(self):
        s = self.agent.analyze(st(bull=1.0, cvd=-0.3, depth=0.2))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("diverges", s.reason)

    def test_bullish_without_flow_data_unconfirmed(self):
        s = self.agent.analyze(st(bull=1.0, unavailable=("cvd_norm_short", "depth_imbalance")))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertTrue(s.available)
        self.assertIn("unconfirmed", s.reason)

    def test_bearish_confirmed(self):
        s = self.agent.analyze(st(bear=1.0, cvd=-0.2, depth=-0.3))
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("105.00", s.reason)
        self.assertIn("above", s.reason)

    def test_no_sweep(self):
        s = self.agent.analyze(st(cvd=0.5, depth=0.5))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("no sweep", s.reason)

    def test_two_sided_ambiguous(self):
        s = self.agent.analyze(st(bull=1.0, bear=1.0, cvd=0.5, depth=0.5))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("ambiguous", s.reason)

    def test_warmup_unavailable(self):
        s = self.agent.analyze(make_state({"bullish_sweep": float("nan"), "bearish_sweep": float("nan"),
                                           "prior_low": float("nan"), "prior_high": float("nan")}))
        self.assertFalse(s.available)


if __name__ == "__main__":
    unittest.main()
