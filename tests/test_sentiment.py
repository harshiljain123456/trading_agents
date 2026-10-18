"""Tests for indicators/sentiment.py."""

import math
import unittest

import helpers  # noqa: F401

from indicators.sentiment import composite_sentiment, funding_contrarian, normalize_score


class TestNormalize(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(normalize_score(75, 0, 100), 0.5)
        self.assertAlmostEqual(normalize_score(50, 0, 100), 0.0)
        self.assertEqual(normalize_score(150, 0, 100), 1.0)
        self.assertEqual(normalize_score(-5, 0, 100), -1.0)

    def test_missing_and_invalid(self):
        self.assertTrue(math.isnan(normalize_score(None, 0, 1)))
        self.assertTrue(math.isnan(normalize_score(float("nan"), 0, 1)))
        with self.assertRaises(ValueError):
            normalize_score(1, 1, 1)


class TestFunding(unittest.TestCase):
    def test_inside_threshold_zero(self):
        self.assertEqual(funding_contrarian(0.0003, 0.0003, 0.001), 0.0)
        self.assertEqual(funding_contrarian(-0.0002, 0.0003, 0.001), 0.0)

    def test_positive_is_bearish(self):
        self.assertAlmostEqual(funding_contrarian(0.00065, 0.0003, 0.001), -0.5)

    def test_negative_extreme_is_bullish_and_saturates(self):
        self.assertEqual(funding_contrarian(-0.01, 0.0003, 0.001), 1.0)

    def test_missing(self):
        self.assertTrue(math.isnan(funding_contrarian(None, 0.0003, 0.001)))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            funding_contrarian(0.0, 0.001, 0.001)


class TestComposite(unittest.TestCase):
    def test_hand_calculated_with_missing(self):
        c = composite_sentiment(0.5, float("nan"), -1.0, 0.35, 0.35, 0.30)
        self.assertAlmostEqual(c.score, (0.175 - 0.30) / 0.65)
        self.assertEqual(c.available_components, 2)
        self.assertAlmostEqual(c.weight_coverage, 0.65)

    def test_component_directions(self):
        self.assertGreater(composite_sentiment(1.0, None, None, 1, 1, 1).score, 0)
        self.assertLess(composite_sentiment(None, -1.0, None, 1, 1, 1).score, 0)
        self.assertLess(composite_sentiment(None, None, funding_contrarian(0.002, 0.0003, 0.001), 1, 1, 1).score, 0)

    def test_all_missing_nan(self):
        c = composite_sentiment(None, None, None, 1, 1, 1)
        self.assertTrue(math.isnan(c.score))
        self.assertEqual(c.available_components, 0)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            composite_sentiment(0, 0, 0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
