"""RiskEngineAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.risk_engine import RiskEngineAgent
from core.models import SignalType


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.agent = RiskEngineAgent(1.0, self.cfg.agents.risk_engine, 0.8)

    def test_low_risk(self):
        s = self.agent.analyze(make_state({"realized_vol": 0.2, "atr_pct": 0.004, "market_drawdown": 0.0}))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertEqual(s.features["risk_score"], 0.0)
        self.assertFalse(s.features["risk_veto"])
        self.assertEqual(s.confidence, 1.0)

    def test_hand_calculated_components(self):
        p = self.cfg.agents.risk_engine
        # vol: (0.9-0.3)/1.2 = 0.5; atr: (0.0225-0.005)/0.035 = 0.5; dd: 0.175/0.35 = 0.5
        s = self.agent.analyze(make_state({"realized_vol": 0.9, "atr_pct": 0.0225, "market_drawdown": -0.175}))
        for k in ("vol_score", "atr_score", "drawdown_score"):
            self.assertAlmostEqual(s.features[k], 0.5)
        self.assertAlmostEqual(s.features["risk_score"], 0.5)
        self.assertAlmostEqual(p.w_vol + p.w_atr + p.w_drawdown, 1.0)

    def test_extreme_risk_veto(self):
        s = self.agent.analyze(make_state({"realized_vol": 3.0, "atr_pct": 0.1, "market_drawdown": -0.5}))
        self.assertAlmostEqual(s.features["risk_score"], 1.0)
        self.assertTrue(s.features["risk_veto"])
        self.assertIn("ABOVE", s.reason)

    def test_missing_fails_safe(self):
        s = self.agent.analyze(make_state({"realized_vol": 0.5, "atr_pct": float("nan"), "market_drawdown": 0.0}))
        self.assertFalse(s.available)
        self.assertIn("fail-safe", s.reason)


if __name__ == "__main__":
    unittest.main()
