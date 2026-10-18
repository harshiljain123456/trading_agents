"""Evaluator scoring, weight suggestions, metrics and calibration."""

import unittest

from helpers import TS, config

import numpy as np
import pandas as pd

from core.models import AgentScore, Trade
from execution.evaluator import Evaluator, LogisticCalibrator, SignalRecord, compute_metrics


def rec(i, name, d, c=1.0, available=True, w=1.0):
    return SignalRecord(i, name, d, c, available, c * d if available else 0.0, w, available)


class TestScoreAgents(unittest.TestCase):
    def setUp(self):
        self.cfg = config(evaluation={"min_agent_samples": 2})
        self.ev = Evaluator(self.cfg, {"a": 1.0, "b": 1.0})
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)

    def test_only_matured_outcomes(self):
        hist = [rec(i, "a", 1) for i in range(10)] + [rec(i, "b", 0) for i in range(10)]
        s = self.ev.score_agents(hist, self.prices, lookahead=3, as_of_index=5)
        self.assertEqual(s["a"].matured_count, 3)  # bars 0,1,2 (2+3 <= 5)
        self.assertEqual(s["a"].directional_count, 6)  # bars 0..5 visible
        self.assertEqual(s["a"].hit_rate, 1.0)

    def test_future_prices_do_not_matter(self):
        hist = [rec(i, "a", 1) for i in range(10)] + [rec(i, "b", -1) for i in range(10)]
        s1 = self.ev.score_agents(hist, self.prices, lookahead=3, as_of_index=5)
        p2 = self.prices.copy()
        p2[6:] = 1.0  # crash after as_of
        s2 = self.ev.score_agents(hist, p2, lookahead=3, as_of_index=5)
        self.assertEqual(s1, s2)

    def test_hold_excluded_from_hit_rate(self):
        hist = [rec(i, "a", 1 if i % 2 else 0) for i in range(10)]
        s = self.ev.score_agents(hist, self.prices, lookahead=1)
        self.assertEqual(s["a"].directional_count, 5)
        self.assertEqual(s["a"].hit_rate, 1.0)

    def test_wrong_direction_misses(self):
        hist = [rec(i, "a", -1) for i in range(10)]
        self.assertEqual(self.ev.score_agents(hist, self.prices, lookahead=1)["a"].hit_rate, 0.0)

    def test_unavailable_excluded(self):
        hist = [rec(i, "a", 0, available=False) for i in range(10)]
        s = self.ev.score_agents(hist, self.prices, lookahead=1)["a"]
        self.assertEqual(s.directional_count, 0)
        self.assertIsNone(s.hit_rate)

    def test_standalone_pnl_with_costs(self):
        ev = Evaluator(config(execution={"fee_bps": 0, "slippage_bps": 0}), {"a": 1.0})
        hist = [rec(i, "a", 1) for i in range(3)]
        p = np.array([100.0, 110.0, 121.0])
        s = ev.score_agents(hist, p, lookahead=1)["a"]
        self.assertAlmostEqual(s.standalone_pnl, 0.2)  # 10% + 10%, non-compounded


class TestSuggestWeights(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.ev = Evaluator(self.cfg, {"a": 1.0})

    def score(self, matured, t, current=1.0):
        return AgentScore("a", 100, 100, matured, 0.6, 0.01, 0.1, 1.0, t, current, current)

    def test_insufficient_samples_keep_weight(self):
        self.assertEqual(self.ev.suggest_weights({"a": self.score(5, 3.0, 1.3)})["a"], 1.3)

    def test_positive_evidence_increases_bounded(self):
        w = self.ev.suggest_weights({"a": self.score(100, 2.5)})["a"]
        self.assertGreater(w, 1.0)
        for _ in range(200):
            w = self.ev.suggest_weights({"a": self.score(100, 3.0, w)})["a"]
        self.assertLessEqual(w, self.cfg.evaluation.max_weight)

    def test_negative_evidence_decreases_bounded(self):
        w = 1.0
        for _ in range(200):
            w = self.ev.suggest_weights({"a": self.score(100, -3.0, w)})["a"]
        self.assertGreaterEqual(w, self.cfg.evaluation.min_weight)
        self.assertLess(w, 1.0)

    def test_neutral_evidence_shrinks_to_base(self):
        w = self.ev.suggest_weights({"a": self.score(100, 0.2, 1.8)})["a"]
        self.assertAlmostEqual(w, 0.75 * 1.8 + 0.25 * 1.0)


class TestMetrics(unittest.TestCase):
    def test_zero_variance_and_no_trades(self):
        eq = pd.Series([100.0] * 20)
        m = compute_metrics(eq, [], pd.Series([0.0] * 20), 0, 0, 0, 2190, 6)
        self.assertIsNone(m.sharpe)
        self.assertIsNone(m.win_rate)
        self.assertIsNone(m.profit_factor)
        self.assertIsNone(m.cagr)
        self.assertIn("sharpe", m.notes)

    def test_no_losses_profit_factor_unavailable(self):
        t = Trade(1, TS, 0, 100, 1, "x", TS, 5, 110, "y", gross_pnl=10, fees=1)
        m = compute_metrics(pd.Series(np.linspace(100, 110, 20)), [t], pd.Series([1.0] * 20), 1, 0, 100, 2190, 6)
        self.assertEqual(m.win_rate, 1.0)
        self.assertIsNone(m.profit_factor)
        self.assertEqual(m.avg_holding_bars, 5)

    def test_drawdown(self):
        m = compute_metrics(pd.Series([100.0, 120, 90, 130]), [], pd.Series([0.0] * 4), 0, 0, 0, 2190, 6)
        self.assertAlmostEqual(m.max_drawdown, 0.25)


class TestCalibrator(unittest.TestCase):
    def test_fit_monotone(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < 0.3 + 0.4 * x).astype(float)
        c = LogisticCalibrator()
        self.assertTrue(c.fit(x, y))
        p = c.predict(np.array([0.1, 0.9]))
        self.assertLess(p[0], p[1])

    def test_insufficient(self):
        self.assertFalse(LogisticCalibrator().fit(np.array([0.1, 0.2]), np.array([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
