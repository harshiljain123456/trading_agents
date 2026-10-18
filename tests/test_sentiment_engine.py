"""SentimentEngineAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.sentiment_engine import SentimentEngineAgent
from core.models import SignalType


class TestSentimentEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.agent = SentimentEngineAgent(1.0, self.cfg.agents.sentiment_engine)

    def test_positive_news_social_buy(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 90, "news_sentiment": 0.8, "funding_rate": 0.0001}))
        self.assertEqual(s.signal, SignalType.BUY)
        # social 0.8, news 0.8, funding term 0 -> (0.35*0.8+0.35*0.8+0)/1.0 = 0.56
        self.assertAlmostEqual(s.features["composite"], 0.56)
        self.assertAlmostEqual(s.confidence, 0.56)

    def test_negative_news_social_sell(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 10, "news_sentiment": -0.8, "funding_rate": 0.0}))
        self.assertEqual(s.signal, SignalType.SELL)

    def test_extreme_positive_funding_contrarian_sell(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 50, "news_sentiment": 0.0, "funding_rate": 0.003}))
        self.assertEqual(s.features["funding_term"], -1.0)
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("contrarian bearish", s.reason)

    def test_extreme_negative_funding_contrarian_buy(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 50, "news_sentiment": 0.0, "funding_rate": -0.003}))
        self.assertEqual(s.signal, SignalType.BUY)

    def test_missing_feed_not_neutral(self):
        # news missing: composite uses social + funding only and coverage reduces confidence
        s = self.agent.analyze(make_state(raw={"social_sentiment": 90, "funding_rate": 0.0001}, unavailable=("news_sentiment",)))
        self.assertAlmostEqual(s.features["composite"], (0.35 * 0.8) / 0.65)
        self.assertAlmostEqual(s.features["weight_coverage"], 0.65)
        self.assertIn("missing: news_sentiment", s.reason)

    def test_too_few_components_unavailable(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 90}, unavailable=("news_sentiment", "funding_rate")))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("not treated as neutral", s.reason)


if __name__ == "__main__":
    unittest.main()
