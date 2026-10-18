"""End-to-end integration: data -> features -> agents -> boss -> execution -> evaluation."""

import unittest

from helpers import CONFIG_PATH, config

import numpy as np
import pandas as pd

from agents import build_agents
from agents.boss_orchestrator import BossOrchestrator
from core.config import ConfigError, load_config
from core.feature_bus import compute_features
from core.models import FeatureStatus, SignalType
from data.feed import DataValidationError, LiveFeedStub, LiveFeedUnavailableError, MarketData, SyntheticFeed, parse_utc_date, validate_bars
from execution.backtester import Backtester
from execution.walk_forward import make_folds, run_walk_forward

START, END = parse_utc_date("2023-01-01"), parse_utc_date("2023-04-01")


def feed(cfg, seed=11, start=START, end=END):
    return SyntheticFeed(cfg, seed=seed).get_bars("TEST", start, end)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.md = feed(cls.cfg)
        cls.ff = compute_features(cls.md, cls.cfg)

    def test_all_agents_analyze_same_state(self):
        st = self.ff.state(self.ff.first_trade_index + 10)
        sigs = [a.analyze(st) for a in build_agents(self.cfg)]
        self.assertEqual(len(sigs), 5)
        for s in sigs:
            self.assertTrue(s.reason)
            self.assertTrue(0 <= s.confidence <= 1)
            self.assertEqual(s.timestamp, st.timestamp)
        d = BossOrchestrator(self.cfg, [a.name for a in build_agents(self.cfg)]).decide(sigs)
        self.assertTrue(-1 <= d.directional_score <= 1)

    def test_synthetic_labelled(self):
        self.assertEqual(self.md.availability["close"], FeatureStatus.SYNTHETIC)
        self.assertEqual(self.md.availability["put_call_skew"], FeatureStatus.UNAVAILABLE)
        self.assertEqual(self.ff.availability["ofi"], FeatureStatus.DERIVED)

    def test_features_are_causal(self):
        t = self.ff.first_trade_index + 50
        trunc = MarketData(self.md.symbol, self.md.bars.iloc[: t + 1].copy(), self.md.availability, "trunc", self.md.first_trade_index)
        ff2 = compute_features(trunc, self.cfg)
        a = self.ff.frame.iloc[t][self.ff.indicator_names].astype(float)
        b = ff2.frame.iloc[t][ff2.indicator_names].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_reproducible_with_seed(self):
        r1 = Backtester(self.cfg, compute_features(feed(self.cfg, 11), self.cfg)).run(self.ff.first_trade_index, len(self.ff))
        r2 = Backtester(self.cfg, compute_features(feed(self.cfg, 11), self.cfg)).run(self.ff.first_trade_index, len(self.ff))
        pd.testing.assert_series_equal(r1.equity, r2.equity)
        other = feed(self.cfg, 12).bars["close"]
        self.assertFalse(np.allclose(other.to_numpy(), self.md.bars["close"].to_numpy()))

    def test_frozen_run_has_no_weight_updates(self):
        w = {"market_intel": 1.5}
        r = Backtester(self.cfg, self.ff).run(self.ff.first_trade_index, len(self.ff), learn=False, initial_weights=w)
        self.assertEqual(r.weight_updates, [])
        self.assertEqual(r.final_weights["market_intel"], 1.5)

    def test_missing_sentiment_feed_is_unavailable_not_neutral(self):
        bars = self.md.bars.drop(columns=["social_sentiment", "news_sentiment"])
        av = dict(self.md.availability)
        av["social_sentiment"] = av["news_sentiment"] = FeatureStatus.UNAVAILABLE
        ff = compute_features(MarketData("T", bars, av, "x", self.md.first_trade_index), self.cfg)
        st = ff.state(ff.first_trade_index + 5)
        self.assertIsNone(st.social_sentiment)
        s = [a for a in build_agents(self.cfg) if a.name == "sentiment_engine"][0].analyze(st)
        self.assertFalse(s.available)

    def test_missing_order_flow_disables_ofi(self):
        bars = self.md.bars.drop(columns=["bid_volume", "ask_volume"])
        av = dict(self.md.availability)
        av["bid_volume"] = av["ask_volume"] = FeatureStatus.UNAVAILABLE
        ff = compute_features(MarketData("T", bars, av, "x", self.md.first_trade_index), self.cfg)
        self.assertEqual(ff.availability["ofi"], FeatureStatus.UNAVAILABLE)
        s = [a for a in build_agents(self.cfg) if a.name == "market_intel"][0].analyze(ff.state(ff.first_trade_index))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)

    def test_optional_agent_flag(self):
        self.assertNotIn("on_chain", [a.name for a in build_agents(self.cfg)])
        cfg = config(features={"enable_optional_agent": True})
        agents = build_agents(cfg)
        self.assertIn("on_chain", [a.name for a in agents])
        r = Backtester(cfg, compute_features(feed(cfg), cfg)).run(self.ff.first_trade_index, self.ff.first_trade_index + 50)
        self.assertIn("on_chain", [c.agent_name for c in r.decisions[-1].breakdown])

    def test_optional_agent_unavailable_without_data(self):
        cfg = config(features={"enable_optional_agent": True}, synthetic={"generate_optional": False})
        ff = compute_features(feed(cfg), cfg)
        s = [a for a in build_agents(cfg) if a.name == "on_chain"][0].analyze(ff.state(ff.first_trade_index + 5))
        self.assertFalse(s.available)


class TestWalkForward(unittest.TestCase):
    def test_folds_chronological_with_embargo(self):
        cfg = config()
        folds = make_folds(cfg, 2440, 250)
        self.assertGreater(len(folds), 1)
        e = cfg.walk_forward.embargo_bars
        for f in folds:
            self.assertLessEqual(f.train[1] + e, f.validation[0])
            self.assertLessEqual(f.validation[1] + e, f.test[0])
            self.assertLess(f.train[0], f.train[1])
        for a, b in zip(folds, folds[1:]):
            self.assertLess(a.test[0], b.test[0])
            self.assertLessEqual(a.test[1], b.test[0] + cfg.walk_forward.test_bars)

    def test_walk_forward_runs_and_separates_results(self):
        cfg = config(walk_forward={"train_bars": 200, "validation_bars": 50, "test_bars": 50, "embargo_bars": 20,
                                   "threshold_grid": [0.35]})
        ff = compute_features(feed(cfg, start=START, end=parse_utc_date("2023-06-01")), cfg)
        rep = run_walk_forward(cfg, ff)
        self.assertGreater(len(rep.folds), 0)
        for f in rep.folds:
            self.assertEqual(f.test_metrics.n_bars, 50)
            self.assertEqual(f.train_metrics.n_bars, 200)
            self.assertLess(f.windows.train[1], f.windows.test[0])
        self.assertEqual(rep.oos_metrics.n_bars, 1 + sum(len(f.test_returns) for f in rep.folds))

    def test_too_short_raises(self):
        cfg = config()
        ff = compute_features(feed(cfg), cfg)
        with self.assertRaises(ValueError):
            run_walk_forward(cfg, ff)


class TestValidation(unittest.TestCase):
    def bars(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="4h", tz="UTC")
        return pd.DataFrame({"open": [10.0] * 5, "high": [11.0] * 5, "low": [9.0] * 5, "close": [10.0] * 5, "volume": [1.0] * 5}, index=idx)

    def test_valid(self):
        validate_bars(self.bars())

    def test_duplicates(self):
        b = self.bars()
        b.index = b.index[[0, 1, 1, 3, 4]]
        with self.assertRaises(DataValidationError):
            validate_bars(b)

    def test_naive_timestamps(self):
        b = self.bars()
        b.index = b.index.tz_localize(None)
        with self.assertRaises(DataValidationError):
            validate_bars(b)

    def test_missing_and_invalid_prices(self):
        b = self.bars()
        b.iloc[2, 3] = np.nan
        with self.assertRaises(DataValidationError):
            validate_bars(b)
        b = self.bars()
        b.iloc[2, 1] = 9.5  # high < close
        with self.assertRaises(DataValidationError):
            validate_bars(b)
        b = self.bars()
        b.iloc[0, 0] = -1
        with self.assertRaises(DataValidationError):
            validate_bars(b)

    def test_insufficient_history(self):
        with self.assertRaises(DataValidationError):
            validate_bars(self.bars(), min_bars=10)

    def test_invalid_date_range(self):
        with self.assertRaises(DataValidationError):
            SyntheticFeed(config()).get_bars("T", END, START)

    def test_live_feed_stub_never_falls_back(self):
        with self.assertRaises(LiveFeedUnavailableError):
            LiveFeedStub().get_bars("BTCUSDT", START, END)

    def test_config_errors(self):
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"walk_forward": {"embargo_bars": 5}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"strategy": {"min_confidnce": 0.3}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"strategy": {"allow_short": True}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"execution": {"max_exposure": 2.0}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"agents": {"base_weights": {"market_intel": 5.0}}})


if __name__ == "__main__":
    unittest.main()


class TestKalman(unittest.TestCase):
    def test_stable_reversal_missing_bounds(self):
        from core.kalman import Kalman1D

        k = Kalman1D(0.02, 0.1)
        self.assertIsNone(k.update(None))          # uninitialized + missing
        self.assertEqual(k.update(0.5), 0.5)       # first observation initializes
        for _ in range(20):
            x = k.update(0.5)
        self.assertAlmostEqual(x, 0.5)             # stable signal stays put
        held = k.update(float("nan"))
        self.assertAlmostEqual(held, 0.5)          # missing obs: predict only
        path = [k.update(-0.8) for _ in range(30)]
        self.assertGreater(path[0], -0.8)          # lag on abrupt reversal
        self.assertLess(path[0], 0.5)
        self.assertAlmostEqual(path[-1], -0.8, places=3)
        k2 = Kalman1D(0.02, 0.1)
        self.assertEqual(k2.update(5.0), 1.0)      # clipped to bounds
        with self.assertRaises(ValueError):
            Kalman1D(0.0, 0.1)
