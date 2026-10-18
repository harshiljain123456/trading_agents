"""Shared test fixtures: hand-crafted market states and small configs."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import FrameworkConfig, load_config  # noqa: E402
from core.models import RAW_FIELDS, FeatureStatus, MarketState  # noqa: E402

TS = datetime(2023, 6, 1, tzinfo=timezone.utc)
logging.basicConfig(level=logging.ERROR)  # keep test output readable
CONFIG_PATH = os.path.join(ROOT, "config.yaml")


def config(**overrides: dict) -> FrameworkConfig:
    """Load the project config with nested overrides."""
    return load_config(CONFIG_PATH, overrides or None)


def make_state(
    indicators: dict[str, float] | None = None,
    raw: dict[str, float | None] | None = None,
    unavailable: tuple[str, ...] = (),
    close: float = 100.0,
    bar_index: int = 500,
) -> MarketState:
    """Build a MarketState; every supplied feature is valid unless listed in ``unavailable``."""
    ind = dict(indicators or {})
    r = {"social_sentiment": 50.0, "news_sentiment": 0.0, "funding_rate": 0.0001,
         "bid_volume": 500.0, "ask_volume": 500.0, "trade_side": 0.0, "bid_depth": 1000.0, "ask_depth": 1000.0}
    r.update(raw or {})
    av: dict[str, FeatureStatus] = {}
    for k in RAW_FIELDS:
        av[k] = FeatureStatus.SYNTHETIC
    for k in ind:
        av[k] = FeatureStatus.DERIVED
    for k in unavailable:
        av[k] = FeatureStatus.UNAVAILABLE
        if k in r:
            r[k] = None
    return MarketState(
        timestamp=TS, symbol="TEST", open=close, high=close * 1.01, low=close * 0.99, close=close, volume=1000.0,
        bid_volume=r["bid_volume"], ask_volume=r["ask_volume"], trade_side=r["trade_side"],
        bid_depth=r["bid_depth"], ask_depth=r["ask_depth"], social_sentiment=r["social_sentiment"],
        news_sentiment=r["news_sentiment"], funding_rate=r["funding_rate"],
        indicators=ind, availability=av, bar_index=bar_index,
    )
