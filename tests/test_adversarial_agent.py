"""AdversarialAgent shadow accounting."""

import unittest

from helpers import TS

import numpy as np

from agents.adversarial_agent import AdversarialAgent


class TestAdversarial(unittest.TestCase):
    def test_takes_opposite_side_and_pnl(self):
        a = AdversarialAgent(10_000, fee_bps=0, slippage_bps=0, exposure=1.0, horizon=1)
        a.observe(0, +1)                     # consensus BUY at bar 0 close
        a.on_open(1, TS, 100.0)              # filled at bar 1 open
        self.assertGreater(a.consensus.units, 0)
        self.assertLess(a.counter.units, 0)
        a.on_close(90.0)                     # price falls 10%
        self.assertAlmostEqual(a.consensus_equity[-1], 9_000.0)
        self.assertAlmostEqual(a.counter_equity[-1], 11_000.0)
        rep = a.report(np.array([100.0, 90.0]), as_of_index=1)
        self.assertTrue(rep.counter_outperformed)
        self.assertEqual(rep.observations, 1)
        self.assertEqual(rep.matured_outcomes, 1)
        self.assertEqual(rep.counter_hit_rate, 1.0)

    def test_costs_applied_equally(self):
        a = AdversarialAgent(10_000, fee_bps=10, slippage_bps=5, exposure=1.0)
        a.observe(0, -1)
        a.on_open(1, TS, 100.0)
        a.on_close(100.0)
        self.assertLess(a.consensus_equity[-1], 10_000)
        self.assertLess(a.counter_equity[-1], 10_000)
        # long side is cash-constrained for costs, so notional differs by ~costs only
        self.assertAlmostEqual(a.consensus.total_fees, a.counter.total_fees, delta=0.05)

    def test_no_same_bar_execution(self):
        a = AdversarialAgent(10_000, 0, 0)
        a.observe(5, 1)
        with self.assertRaises(RuntimeError):
            a.on_open(5, TS, 100.0)

    def test_hold_is_not_an_observation(self):
        a = AdversarialAgent(10_000, 0, 0, horizon=20)
        a.observe(0, 0)
        a.observe(1, 1)
        rep = a.report(np.ones(10), as_of_index=9)
        self.assertEqual(rep.observations, 1)
        self.assertEqual(rep.matured_outcomes, 0)  # 1 + 20 > 9: not matured


if __name__ == "__main__":
    unittest.main()
