"""Accounting, execution timing, position management and kill switch."""

import unittest
from datetime import timedelta

from helpers import TS, config, make_state

from core.models import BossDecision, Regime, SignalType
from core.portfolio_risk import KillSwitch
from data.feed import SyntheticFeed, parse_utc_date
from core.feature_bus import compute_features
from execution.backtester import Backtester, run_benchmarks, target_from_decision
from execution.executor import NullBroker, PaperExecutor, Portfolio, SimulatedExecutor
from core.models import Order


def decision(action=SignalType.BUY, strength=0.5, veto=False, kill=False):
    return BossDecision(action, action, strength, 0.3, 1.0, 0.5, 1.0, "test", [], Regime.RANGE, 0.3, veto, kill, None, TS)


class TestPortfolioAccounting(unittest.TestCase):
    def test_buy_then_sell_hand_calculated(self):
        p = Portfolio(10_000, fee_bps=10, slippage_bps=5)
        f = p.rebalance_to(0.5, 100.0, TS, 1, "entry")[0]
        self.assertAlmostEqual(f.exec_price, 100.05)
        self.assertAlmostEqual(f.quantity, 50.0)
        self.assertAlmostEqual(f.fee, 50 * 100.05 * 0.001)
        self.assertAlmostEqual(p.cash, 10_000 - 5002.5 - 5.0025)
        self.assertAlmostEqual(p.equity(100.0), 10_000 - 2.5 - 5.0025)
        p.rebalance_to(0.0, 110.0, TS + timedelta(hours=4), 2, "exit")
        t = p.trades[0]
        self.assertAlmostEqual(t.exit_price, 110 * 0.9995)
        self.assertAlmostEqual(t.gross_pnl, 50 * (109.945 - 100.05))
        self.assertAlmostEqual(t.fees, 5.0025 + 50 * 109.945 * 0.001)
        self.assertAlmostEqual(p.equity(110.0), 10_000 + t.net_pnl)
        self.assertEqual(p.units, 0.0)

    def test_no_unintended_leverage(self):
        p = Portfolio(10_000, fee_bps=10, slippage_bps=5)
        p.rebalance_to(1.0, 100.0, TS, 1, "all-in")
        self.assertGreaterEqual(p.cash, -1e-9)
        self.assertLessEqual(p.exposure(100.0), 1.0 + 1e-9)

    def test_partial_exit_realizes_proportionally(self):
        p = Portfolio(10_000, 0, 0)
        p.rebalance_to(1.0, 100.0, TS, 1, "in")
        p.rebalance_to(0.5, 120.0, TS, 2, "half")  # equity 12000 -> keep 50 units (6000/120)
        self.assertAlmostEqual(p.units, 50.0)
        self.assertAlmostEqual(p.open_trade.gross_pnl, 50 * 20)
        self.assertEqual(len(p.trades), 0)

    def test_reversal_two_legs_both_charged(self):
        p = Portfolio(10_000, 10, 0, allow_short=True)
        p.rebalance_to(0.5, 100.0, TS, 1, "long")
        fills = p.rebalance_to(-0.5, 100.0, TS, 2, "reverse")
        self.assertEqual(len(fills), 2)
        self.assertTrue(all(f.fee > 0 for f in fills))
        self.assertEqual(len(p.trades), 1)
        self.assertLess(p.units, 0)

    def test_short_rejected_when_long_only(self):
        with self.assertRaises(ValueError):
            Portfolio(10_000, 0, 0).rebalance_to(-0.5, 100.0, TS, 1, "short")

    def test_equity_identity(self):
        p = Portfolio(10_000, 10, 5)
        p.rebalance_to(0.8, 100.0, TS, 1, "a")
        p.rebalance_to(0.3, 105.0, TS, 2, "b")
        p.rebalance_to(0.9, 98.0, TS, 3, "c")
        price = 101.0
        realized = sum(t.net_pnl for t in p.trades) + p.open_trade.gross_pnl - p.open_trade.fees
        self.assertAlmostEqual(p.equity(price) - 10_000, realized + p.unrealized_pnl(price), places=6)


class TestExecutor(unittest.TestCase):
    def test_fill_only_after_decision_bar(self):
        ex = SimulatedExecutor(Portfolio(10_000, 0, 0))
        ex.submit(Order(TS, 5, 0.5, "x"))
        with self.assertRaises(RuntimeError):
            ex.on_bar_open(5, TS, 100.0)
        fills = ex.on_bar_open(6, TS, 100.0)
        self.assertEqual(fills[0].bar_index, 6)

    def test_unfilled_at_end(self):
        ex = SimulatedExecutor(Portfolio(10_000, 0, 0))
        ex.submit(Order(TS, 9, 0.5, "x"))
        ex.finalize()
        self.assertEqual(len(ex.unfilled), 1)
        self.assertEqual(ex.portfolio.units, 0)

    def test_paper_never_routes_live(self):
        with self.assertRaises(RuntimeError):
            PaperExecutor(Portfolio(10_000, 0, 0), live_orders_enabled=True)
        with self.assertRaises(RuntimeError):
            NullBroker().submit_order(Order(TS, 0, 1.0, "x"), 1.0, None)


class TestPositionManagement(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.state = make_state({"realized_vol": 0.5})

    def test_buy_sizing_bounded(self):
        t, _ = target_from_decision(decision(SignalType.BUY, 0.6), 0.0, self.state, self.cfg, "flatten")
        self.assertAlmostEqual(t, min(self.cfg.execution.max_position_size, 0.6 * self.cfg.execution.exposure_cap))

    def test_long_only_sell_closes_without_short(self):
        t, why = target_from_decision(decision(SignalType.SELL), 0.6, self.state, self.cfg, "flatten")
        self.assertEqual(t, 0.0)
        t2, why2 = target_from_decision(decision(SignalType.SELL), 0.0, self.state, self.cfg, "flatten")
        self.assertIsNone(t2)
        self.assertIn("does not open shorts", why2)

    def test_sell_reduce_policy(self):
        cfg = config(strategy={"sell_policy": "reduce", "sell_reduce_fraction": 0.5})
        t, _ = target_from_decision(decision(SignalType.SELL), 0.8, self.state, cfg, "flatten")
        self.assertAlmostEqual(t, 0.4)

    def test_hold_keeps_position(self):
        t, _ = target_from_decision(decision(SignalType.HOLD), 0.7, self.state, self.cfg, "flatten")
        self.assertIsNone(t)

    def test_risk_veto_reduces_existing(self):
        t, why = target_from_decision(decision(SignalType.HOLD, veto=True), 0.9, self.state, self.cfg, "flatten")
        self.assertAlmostEqual(t, self.cfg.strategy.veto_exposure_cap)
        self.assertIn("risk veto", why)

    def test_kill_switch_flatten_and_hold_policies(self):
        t, _ = target_from_decision(decision(SignalType.HOLD, kill=True), 0.5, self.state, self.cfg, "flatten")
        self.assertEqual(t, 0.0)
        t2, _ = target_from_decision(decision(SignalType.HOLD, kill=True), 0.5, self.state, self.cfg, "hold")
        self.assertIsNone(t2)

    def test_volatility_aware_sizing(self):
        cfg = config(execution={"volatility_aware_sizing": True, "target_volatility": 0.25})
        t, _ = target_from_decision(decision(SignalType.BUY, 0.8), 0.0, self.state, cfg, "flatten")
        self.assertAlmostEqual(t, 0.8 * 0.25 / 0.5)


class TestKillSwitch(unittest.TestCase):
    def ks(self, **kw):
        base = dict(max_drawdown=0.1, sharpe_threshold=-100.0, sharpe_window=10, min_history=5, bars_per_year=2190,
                    recovery="cooldown", cooldown_bars=3)
        base.update(kw)
        return KillSwitch(**base)

    def test_min_history_then_drawdown_trigger(self):
        k = self.ks()
        eq = [100, 100, 100, 85]  # 15% drawdown but only 3 returns
        self.assertFalse(any(k.update(e, TS, i) for i, e in enumerate(eq)))
        for i, e in enumerate([85, 85, 85], start=4):
            k.update(e, TS, i)
        self.assertTrue(k.active)
        self.assertEqual(k.events[0].kind, "trigger")

    def test_cooldown_recovery(self):
        k = self.ks(min_history=1)
        k.update(100, TS, 0)
        k.update(80, TS, 1)
        self.assertTrue(k.active)
        k.update(80, TS, 2)
        k.update(80, TS, 3)
        self.assertTrue(k.active)
        self.assertFalse(k.update(80, TS, 4))
        self.assertEqual(k.events[-1].kind, "recovery")

    def test_manual_stays_active(self):
        k = self.ks(min_history=1, recovery="manual")
        k.update(100, TS, 0)
        k.update(80, TS, 1)
        for i in range(2, 20):
            self.assertTrue(k.update(80, TS, i))
        k.reset()
        self.assertFalse(k.active)

    def test_sharpe_trigger(self):
        k = self.ks(max_drawdown=0.99, sharpe_threshold=-1.0, sharpe_window=10, min_history=10)
        e = 100.0
        for i in range(12):
            e *= 0.999 if i % 2 else 0.998
            k.update(e, TS, i)
        self.assertTrue(k.active)
        self.assertIn("Sharpe", k.events[0].detail)


class TestBacktestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        md = SyntheticFeed(cls.cfg, seed=7).get_bars("T", parse_utc_date("2023-01-01"), parse_utc_date("2023-05-01"))
        cls.ff = compute_features(md, cls.cfg)
        cls.res = Backtester(cls.cfg, cls.ff).run(cls.ff.first_trade_index, len(cls.ff))

    def test_fills_at_next_bar_open_with_costs(self):
        opens = self.ff.column("open")
        slip = self.cfg.execution.slippage_bps / 1e4
        fee = self.cfg.execution.fee_bps / 1e4
        self.assertGreater(len(self.res.portfolio.fills), 0)
        for f in self.res.portfolio.fills:
            self.assertEqual(f.reference_price, opens[f.bar_index])
            self.assertEqual(f.timestamp, self.ff.index[f.bar_index].to_pydatetime())
            expect = f.reference_price * (1 + slip if f.side.value == "BUY" else 1 - slip)
            self.assertAlmostEqual(f.exec_price, expect)
            self.assertAlmostEqual(f.fee, f.notional * fee)

    def test_orders_reference_previous_decision(self):
        for f in self.res.portfolio.fills:
            k = f.bar_index - 1 - self.res.start_index
            self.assertGreaterEqual(k, 0)

    def test_long_only_never_short(self):
        self.assertTrue((self.res.exposures >= -1e-12).all())
        self.assertTrue((self.res.exposures <= self.cfg.execution.max_exposure + 1e-9).all())

    def test_equity_consistency(self):
        p = self.res.portfolio
        last_close = self.ff.column("close")[self.res.end_index - 1]
        self.assertAlmostEqual(self.res.equity.iloc[-1], p.cash + p.units * last_close, places=6)

    def test_benchmarks_same_window(self):
        b = run_benchmarks(self.cfg, self.ff, self.ff.first_trade_index, len(self.ff))
        for m in b.values():
            self.assertEqual(m.n_bars, self.res.metrics.n_bars)
        self.assertEqual(b["no_trade"].total_return, 0.0)
        self.assertGreater(b["buy_and_hold"].total_fees, 0)


if __name__ == "__main__":
    unittest.main()
