"""MarketMoodAgent, its features and the market-context join (offline)."""

import json
import math
import os
import unittest

from helpers import ROOT, make_state

import numpy as np
import pandas as pd

from agents import build_agents
from agents.market_mood import MarketMoodAgent
from core.config import load_config
from core.feature_bus import compute_features
from core.models import FeatureStatus, SignalType
from data.providers import DataSourceError, YahooProvider, attach_market_context
from execution.backtester import Backtester
from test_providers import fake_daily, ist, yahoo_payload

NSE_CFG = os.path.join(ROOT, "config_nse.yaml")


def mood_state(trend=None, rs=None, vix_pct=None, vix_chg=None, vix=18.0):
    ind = {}
    unavailable = []
    for k, v in (("index_trend", trend), ("rel_strength", rs), ("vix_pct", vix_pct), ("vix_change", vix_chg)):
        if v is None:
            unavailable.append(k)
        else:
            ind[k] = v
    ind["india_vix"] = vix
    return make_state(ind, unavailable=tuple(unavailable))


class TestMarketMoodAgent(unittest.TestCase):
    def setUp(self):
        self.p = load_config(NSE_CFG).agents.market_mood
        self.agent = MarketMoodAgent(1.0, self.p)

    def test_vix_term_rules(self):
        self.assertEqual(self.agent.vix_term(0.95, -0.1), 1.0)     # panic receding
        self.assertEqual(self.agent.vix_term(0.95, 0.1), -1.0)     # fear escalating
        self.assertEqual(self.agent.vix_term(0.05, 0.0), -self.p.complacency_penalty)
        self.assertEqual(self.agent.vix_term(0.5, 0.3), 0.0)

    def test_bullish_market_and_leader_buy(self):
        s = self.agent.analyze(mood_state(0.6, 0.5, 0.5, 0.0))
        expect = (0.4 * 0.6 + 0.4 * 0.5 + 0.2 * 0.0) / 1.0
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.features["score"], expect)
        self.assertAlmostEqual(s.confidence, expect)
        self.assertIn("outperforming", s.reason)

    def test_bearish_market_and_laggard_sell(self):
        s = self.agent.analyze(mood_state(-0.7, -0.6, 0.95, 0.2))
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("fear escalating", s.reason)

    def test_panic_receding_offsets_weak_trend(self):
        s = self.agent.analyze(mood_state(-0.2, 0.0, 0.97, -0.15))
        self.assertAlmostEqual(s.features["score"], (0.4 * -0.2 + 0.2 * 1.0) / 1.0)
        self.assertIn("panic receding", s.reason)

    def test_partial_inputs_reduce_confidence(self):
        s = self.agent.analyze(mood_state(0.8, 0.8))   # no VIX
        self.assertAlmostEqual(s.features["coverage"], 0.8)
        self.assertAlmostEqual(s.confidence, 0.8 * 0.8)
        self.assertIn("missing: india_vix", s.reason)

    def test_no_context_unavailable(self):
        s = self.agent.analyze(mood_state(rs=0.5))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("India VIX / Nifty 50", s.reason)

    def test_nse_config_uses_market_mood(self):
        cfg = load_config(NSE_CFG)
        names = [a.name for a in build_agents(cfg)]
        self.assertIn("market_mood", names)
        self.assertNotIn("sentiment_engine", names)
        crypto = [a.name for a in build_agents(load_config(os.path.join(ROOT, "config.yaml")))]
        self.assertIn("sentiment_engine", crypto)


class TestContextJoinAndFeatures(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(NSE_CFG)
        days, o, h, l, c, v = fake_daily(900, "2020-01-01")
        self.stock = (days, o, h, l, c, v)
        rng = np.random.default_rng(9)
        vix = 15 * np.exp(np.cumsum(rng.normal(0, 0.04, len(days))))
        idx = 15000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days))))
        # VIX missing on one stock day to exercise the as-of fill
        vdays = days.delete(700)
        self.payloads = {
            "TEST.NS": yahoo_payload(days, o, h, l, c, v),
            "%5EINDIAVIX": yahoo_payload(vdays, *[np.delete(vix, 700)] * 4, np.zeros(len(vdays))),
            "%5ENSEI": yahoo_payload(days, idx, idx * 1.005, idx * 0.995, idx, np.zeros(len(days))),
        }
        self.vix, self.idx = vix, idx

    def fetch(self, url, headers):
        for key, payload in self.payloads.items():
            if f"/chart/{key}?" in url:
                return json.dumps(payload).encode()
        raise DataSourceError("unknown ticker in test")

    def test_join_is_asof_and_causal(self):
        prov = YahooProvider(self.cfg, None, fetcher=self.fetch)
        start, end = ist(2021, 6, 1), ist(2023, 6, 1)
        md = attach_market_context(prov.get_bars("TEST", start, end), prov, start, end)
        self.assertEqual(md.availability["india_vix"], FeatureStatus.OBSERVED)
        b = md.bars
        days = self.stock[0]
        pos_missing = b.index.get_indexer([pd.Timestamp(days[700]).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)])[0]
        self.assertGreaterEqual(pos_missing, 1)
        # the stock bar whose VIX is missing takes the previous day's VIX (backward as-of fill)
        self.assertAlmostEqual(b["india_vix"].iloc[pos_missing], b["india_vix"].iloc[pos_missing - 1])

    def test_missing_context_marked_unavailable(self):
        del self.payloads["%5EINDIAVIX"]
        prov = YahooProvider(self.cfg, None, fetcher=self.fetch)
        start, end = ist(2021, 6, 1), ist(2023, 6, 1)
        md = attach_market_context(prov.get_bars("TEST", start, end), prov, start, end)
        self.assertEqual(md.availability["india_vix"], FeatureStatus.UNAVAILABLE)
        self.assertEqual(md.availability["index_close"], FeatureStatus.OBSERVED)
        ff = compute_features(md, self.cfg)
        s = MarketMoodAgent(1.0, self.cfg.agents.market_mood).analyze(ff.state(len(ff) - 1))
        self.assertTrue(s.available)                      # 2 of 3 inputs still enough
        self.assertIn("missing: india_vix", s.reason)

    def test_feature_formulas_and_end_to_end(self):
        prov = YahooProvider(self.cfg, None, fetcher=self.fetch)
        start, end = ist(2021, 6, 1), ist(2023, 6, 1)
        md = attach_market_context(prov.get_bars("TEST", start, end), prov, start, end)
        ff = compute_features(md, self.cfg)
        t = len(ff) - 1
        c = ff.column("close")
        ix = ff.column("index_close")
        L, sc = self.cfg.indicators.rs_lookback, self.cfg.indicators.rs_scale
        expect_rs = math.tanh(((c[t] / c[t - L]) - (ix[t] / ix[t - L])) / sc)
        self.assertAlmostEqual(ff.state(t).get("rel_strength"), expect_rs)
        vix = ff.column("india_vix")
        w = self.cfg.indicators.vix_rank_window
        self.assertAlmostEqual(ff.state(t).get("vix_pct"), float((vix[t - w + 1 : t + 1] <= vix[t]).mean()))
        self.assertEqual(ff.availability["india_vix"], FeatureStatus.OBSERVED)
        res = Backtester(self.cfg, ff).run(ff.first_trade_index, len(ff))
        mm = [s for s in res.signals[-1] if s.agent_name == "market_mood"][0]
        self.assertTrue(mm.available)


if __name__ == "__main__":
    unittest.main()
