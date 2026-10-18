"""MarketIntelAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.market_intel import MarketIntelAgent
from core.models import SignalType


class TestMarketIntel(unittest.TestCase):
    def setUp(self):
        cfg = config()
        self.p = cfg.agents.market_intel
        self.agent = MarketIntelAgent(1.0, self.p)

    def expected(self, ofi, cvd, mom):
        p = self.p
        return (p.w_ofi * ofi + p.w_cvd * cvd + p.w_momentum * mom) / (p.w_ofi + p.w_cvd + p.w_momentum)

    def test_positive_flow_buy(self):
        s = self.agent.analyze(make_state({"ofi": 0.5, "cvd_norm": 0.4, "inst_momentum": 0.6, "pv_corr": 0.2}))
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, abs(self.expected(0.5, 0.4, 0.6)))
        self.assertIn("buying pressure", s.reason)
        self.assertIn("OFI +0.500", s.reason)

    def test_negative_flow_sell(self):
        s = self.agent.analyze(make_state({"ofi": -0.5, "cvd_norm": -0.3, "inst_momentum": -0.4}))
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("selling pressure", s.reason)

    def test_neutral_hold_explained(self):
        s = self.agent.analyze(make_state({"ofi": 0.05, "cvd_norm": -0.02, "inst_momentum": 0.03}))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertTrue(s.available)
        self.assertIn("within", s.reason)

    def test_missing_input_unavailable(self):
        s = self.agent.analyze(make_state({"ofi": float("nan"), "cvd_norm": 0.1, "inst_momentum": 0.1}))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("ofi", s.reason)

    def test_feed_not_supplied(self):
        s = self.agent.analyze(make_state({"cvd_norm": 0.1, "inst_momentum": 0.1}, unavailable=("ofi",)))
        self.assertFalse(s.available)
        self.assertIn("feeds not supplied", s.reason)

    def test_deterministic(self):
        st = make_state({"ofi": 0.3, "cvd_norm": 0.2, "inst_momentum": 0.1})
        self.assertEqual(self.agent.analyze(st), self.agent.analyze(st))


if __name__ == "__main__":
    unittest.main()
