"""Feedback loop: interval, freezing, maturity and bounds."""

import unittest

from helpers import TS, config

import numpy as np

from agents.boss_orchestrator import BossOrchestrator
from execution.evaluator import Evaluator, SignalRecord
from execution.feedback import FeedbackController

NAMES = ["market_intel", "risk_engine", "liquidity_engine", "sentiment_engine", "regime_detector"]


def history(n, good="market_intel"):
    """market_intel always right on a rising series, sentiment always wrong."""
    out = []
    for i in range(n):
        for name in NAMES:
            d = 1 if name == good else (-1 if name == "sentiment_engine" else 0)
            c = 0.8 if d else 0.0
            out.append(SignalRecord(i, name, d, c, True, c * d, 1.0, True))
    return out


class TestFeedback(unittest.TestCase):
    def setUp(self):
        self.cfg = config(evaluation={"min_agent_samples": 10, "feedback_interval": 50})
        self.boss = BossOrchestrator(self.cfg, NAMES)
        self.ev = Evaluator(self.cfg, self.boss.base_weights)
        rng = np.random.default_rng(1)
        self.prices = 100 * np.exp(np.cumsum(0.002 + 0.01 * rng.standard_normal(400)))

    def test_only_on_interval(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=0)
        h = history(300)
        self.assertFalse(fb.maybe_update(49, TS, h, self.prices))
        self.assertTrue(fb.maybe_update(100, TS, h, self.prices))

    def test_frozen_never_updates(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=0, frozen=True)
        before = dict(self.boss.learned_weights)
        self.assertFalse(fb.maybe_update(100, TS, history(300), self.prices))
        self.assertEqual(before, self.boss.learned_weights)

    def test_uses_only_matured_data(self):
        h = history(300)
        fb1 = FeedbackController(self.ev, self.boss, 50, window_start=0)
        fb1.maybe_update(150, TS, h, self.prices)
        w1 = dict(self.boss.learned_weights)
        boss2 = BossOrchestrator(self.cfg, NAMES)
        p2 = self.prices.copy()
        p2[151:] = p2[151:] * 0.01  # future collapse must not matter
        fb2 = FeedbackController(Evaluator(self.cfg, boss2.base_weights), boss2, 50, window_start=0)
        fb2.maybe_update(150, TS, h, p2)
        self.assertEqual(w1, boss2.learned_weights)

    def test_ignores_records_before_window(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=200)
        h = history(300)
        fb.maybe_update(250, TS, h, self.prices)  # only bars 200..230 mature (h=20)
        self.assertTrue(all(abs(w - 1.0) < 1.0 for w in self.boss.learned_weights.values()))

    def test_weights_bounded_and_logged(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=0)
        h = history(399)
        for t in range(50, 400, 50):
            fb.maybe_update(t, TS, h, self.prices)
        ev = self.cfg.evaluation
        for w in self.boss.learned_weights.values():
            self.assertTrue(ev.min_weight <= w <= ev.max_weight)
        self.assertGreater(len(fb.updates), 0)
        self.assertTrue(all(u.evidence for u in fb.updates))
        # risk_engine never votes directionally -> weight preserved
        self.assertEqual(self.boss.learned_weights["risk_engine"], 1.0)


if __name__ == "__main__":
    unittest.main()
