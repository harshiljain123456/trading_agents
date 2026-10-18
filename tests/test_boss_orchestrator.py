"""BossOrchestrator aggregation, overrides and diagnostics."""

import math
import unittest

from helpers import TS, config

from agents.boss_orchestrator import BossOrchestrator
from core.models import AgentSignal, SignalType

NAMES = ["market_intel", "risk_engine", "liquidity_engine", "sentiment_engine", "regime_detector"]


def sig(name, s, c, available=True, **feat):
    return AgentSignal(name, s, c, f"{name} test reason", dict(feat), TS, available, f"{name} summary")


def risk(score=0.3):
    return sig("risk_engine", SignalType.HOLD, 1 - score, risk_score=score)


def regime(r="range", s=SignalType.HOLD, c=0.0):
    return sig("regime_detector", s, c, regime=r)


def plain_cfg(**extra):
    o = {"features": {"enable_kalman": False, "enable_regime_weights": False}}
    o.update(extra)
    return config(**o)


class TestBoss(unittest.TestCase):
    def test_weighted_score_hand_calculated(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 0.8), risk(0.3), sig("liquidity_engine", SignalType.HOLD, 0.0),
                sig("sentiment_engine", SignalType.BUY, 0.6), regime()]
        d = boss.decide(sigs)
        # equal weights: S = (0.8 + 0.6) / 5 = 0.28 ; agreement 1 ; coverage 2/5
        self.assertAlmostEqual(d.directional_score, 0.28)
        self.assertAlmostEqual(d.agreement, 1.0)
        self.assertAlmostEqual(d.coverage, 0.4)
        self.assertAlmostEqual(d.confidence, 1.0 * 0.7 * math.sqrt(0.4))
        self.assertEqual(d.action, SignalType.BUY)
        self.assertIsNone(d.forced_hold_rule)

    def test_disagreement_reduces_agreement(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 0.9), risk(), sig("liquidity_engine", SignalType.SELL, 0.4),
                sig("sentiment_engine", SignalType.BUY, 0.9), regime()]
        d = boss.decide(sigs)
        self.assertAlmostEqual(d.directional_score, (0.9 - 0.4 + 0.9) / 5)
        self.assertAlmostEqual(d.agreement, 2 / 3)

    def test_risk_veto_overrides_buy(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 1.0), risk(0.95), sig("liquidity_engine", SignalType.BUY, 1.0),
                sig("sentiment_engine", SignalType.BUY, 1.0), regime("trend_up", SignalType.BUY, 1.0)]
        d = boss.decide(sigs)
        self.assertEqual(d.raw_action, SignalType.BUY)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertTrue(d.risk_veto)
        self.assertIn("risk_veto", d.forced_hold_rule)
        self.assertIn("overridden", d.reason)

    def test_risk_unavailable_fails_safe(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 1.0), sig("risk_engine", SignalType.HOLD, 0.0, available=False),
                sig("liquidity_engine", SignalType.BUY, 1.0), sig("sentiment_engine", SignalType.BUY, 1.0), regime()]
        d = boss.decide(sigs)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertIn("risk_unavailable", d.forced_hold_rule)

    def test_floors(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        d = boss.decide([sig("market_intel", SignalType.BUY, 0.5), risk(), sig("liquidity_engine", SignalType.HOLD, 0),
                         sig("sentiment_engine", SignalType.HOLD, 0), regime()])
        self.assertAlmostEqual(d.directional_score, 0.1)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertIn("directional_score_floor", d.forced_hold_rule)
        self.assertIn("min_confidence", d.forced_hold_rule)

    def test_kill_switch_forces_hold(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 1.0), risk(), sig("liquidity_engine", SignalType.BUY, 1.0),
                sig("sentiment_engine", SignalType.BUY, 1.0), regime()]
        d = boss.decide(sigs, kill_switch_active=True)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertTrue(d.kill_switch_active)
        self.assertIn("kill_switch", d.forced_hold_rule)

    def test_unavailable_agent_excluded_not_confirming(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        base = [sig("market_intel", SignalType.BUY, 0.8), risk(), sig("liquidity_engine", SignalType.HOLD, 0),
                sig("sentiment_engine", SignalType.BUY, 0.8)]
        d1 = boss.decide(base + [sig("regime_detector", SignalType.HOLD, 0.0, available=False, regime="unknown")])
        # denominator excludes the unavailable agent: S = 1.6 / 4
        self.assertAlmostEqual(d1.directional_score, 0.4)
        self.assertAlmostEqual(d1.data_coverage, 0.8)
        elig = {c.agent_name: c.eligible for c in d1.breakdown}
        self.assertFalse(elig["regime_detector"])

    def test_data_coverage_rule(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        un = lambda n: sig(n, SignalType.HOLD, 0.0, available=False)  # noqa: E731
        d = boss.decide([sig("market_intel", SignalType.BUY, 1.0), risk(), un("liquidity_engine"),
                         un("sentiment_engine"), un("regime_detector")])
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertIn("data_coverage", d.forced_hold_rule)

    def test_disabled_agent_rejected(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        with self.assertRaises(ValueError):
            boss.decide([sig("on_chain", SignalType.BUY, 1.0)])

    def test_regime_weights_applied(self):
        cfg = config(features={"enable_kalman": False, "enable_regime_weights": True})
        boss = BossOrchestrator(cfg, NAMES)
        d = boss.decide([sig("market_intel", SignalType.BUY, 1.0), risk(), sig("liquidity_engine", SignalType.HOLD, 0),
                         sig("sentiment_engine", SignalType.HOLD, 0), regime("trend_up", SignalType.BUY, 0.5)])
        w = {c.agent_name: c.effective_weight for c in d.breakdown}
        tbl = cfg.regime_weights["trend_up"]
        for n in NAMES:
            self.assertAlmostEqual(w[n], tbl[n])
        self.assertAlmostEqual(d.directional_score, (tbl["market_intel"] * 1.0 + tbl["regime_detector"] * 0.5) / sum(tbl[n] for n in NAMES))

    def test_learned_weights_clipped(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        boss.set_learned_weights({"market_intel": 10.0, "sentiment_engine": 0.01})
        self.assertEqual(boss.learned_weights["market_intel"], 2.0)
        self.assertEqual(boss.learned_weights["sentiment_engine"], 0.3)

    def test_family_normalization(self):
        cfg = config(features={"enable_kalman": False, "enable_regime_weights": False, "enable_family_normalization": True})
        boss = BossOrchestrator(cfg, NAMES)
        self.assertEqual(boss.family_factor["market_intel"], 0.5)
        self.assertEqual(boss.family_factor["regime_detector"], 0.5)
        self.assertEqual(boss.family_factor["sentiment_engine"], 1.0)

    def test_kalman_smoothing_responds_to_reversal(self):
        cfg = config(features={"enable_kalman": True, "enable_regime_weights": False})
        boss = BossOrchestrator(cfg, NAMES)
        rest = [risk(), sig("liquidity_engine", SignalType.HOLD, 0), sig("sentiment_engine", SignalType.HOLD, 0), regime()]
        scores = [boss.decide([sig("market_intel", SignalType.BUY, 1.0)] + rest).directional_score for _ in range(10)]
        after = [boss.decide([sig("market_intel", SignalType.SELL, 1.0)] + rest).directional_score for _ in range(15)]
        self.assertAlmostEqual(scores[-1], 0.2, places=3)
        self.assertGreater(after[0], -0.2)            # lagged response...
        self.assertLess(after[0], scores[-1])         # ...but moving toward the new direction
        self.assertLess(after[-1], -0.15)             # converges after the reversal

    def test_reasoner_mentions_actual_values(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        d = boss.decide([sig("market_intel", SignalType.BUY, 0.8), risk(0.31), sig("liquidity_engine", SignalType.HOLD, 0),
                         sig("sentiment_engine", SignalType.BUY, 0.6), regime()])
        self.assertIn("risk_score 0.31", d.reason)
        self.assertIn("market_intel summary", d.reason.lower())


if __name__ == "__main__":
    unittest.main()
