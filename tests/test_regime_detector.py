"""RegimeDetectorAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.regime_detector import RegimeDetectorAgent
from core.models import SignalType


def st(ef, es, adx, rv):
    return make_state({"ema_fast": ef, "ema_slow": es, "adx": adx, "realized_vol": rv})


class TestRegimeDetector(unittest.TestCase):
    def setUp(self):
        self.p = config().agents.regime_detector
        self.agent = RegimeDetectorAgent(1.0, self.p)

    def test_trend_up(self):
        s = self.agent.analyze(st(110, 100, 30, 0.5))
        self.assertEqual(s.features["regime"], "trend_up")
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, (30 - self.p.adx_trend) / (self.p.adx_strong - self.p.adx_trend))
        self.assertIn("ADX 30.0", s.reason)

    def test_trend_down(self):
        s = self.agent.analyze(st(90, 100, 45, 0.5))
        self.assertEqual(s.features["regime"], "trend_down")
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertEqual(s.confidence, 1.0)

    def test_range(self):
        s = self.agent.analyze(st(101, 100, 12, 0.5))
        self.assertEqual(s.features["regime"], "range")
        self.assertEqual(s.signal, SignalType.HOLD)

    def test_volatile_overrides_trend(self):
        s = self.agent.analyze(st(110, 100, 35, 2.0))
        self.assertEqual(s.features["regime"], "volatile")
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("volatility", s.reason)

    def test_warmup_unknown(self):
        s = self.agent.analyze(st(110, float("nan"), float("nan"), 0.5))
        self.assertFalse(s.available)
        self.assertEqual(s.features["regime"], "unknown")


if __name__ == "__main__":
    unittest.main()
