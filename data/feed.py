"""Market-data ingestion: deterministic synthetic feed, validation and live stub.

Synthetic schema (every column labelled ``FeatureStatus.SYNTHETIC``)
-------------------------------------------------------------------
=================  ==========================================================
column             meaning / units
=================  ==========================================================
open/high/low/close  price in quote currency (> 0, OHLC-consistent)
volume             base-asset volume per bar (>= 0)
bid_volume         buyer-initiated volume  = volume * (1 + trade_side) / 2
ask_volume         seller-initiated volume = volume * (1 - trade_side) / 2
trade_side         net aggressor fraction in [-1, 1]
bid_depth          resting bid size near the touch (base units)
ask_depth          resting ask size near the touch (base units)
social_sentiment   social index on [0, 100] (50 = neutral)
news_sentiment     news tone on [-1, 1]
funding_rate       perpetual funding per 8h period (0.0001 = 1 bp)
exchange_netflow   optional on-chain: net coins flowing *into* exchanges
stablecoin_supply_change  optional on-chain: fractional change in supply
=================  ==========================================================

The generator stitches together segments of five regimes (trend_up,
trend_down, range, volatile, noisy) with seeded random lengths. Order-flow,
depth, sentiment and funding are generated with a configurable, deliberately
*weak* dependence on the hidden regime drift (``flow_signal_strength``), plus
noise; social sentiment and funding react to *past* returns. These are toy
dynamics for exercising the pipeline - not a model of any real market.

Timestamps are UTC. The requested range is ``[start, end)`` (end exclusive);
``warmup_bars`` extra bars are generated *before* ``start`` so indicators can
warm up. Those bars are never traded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_PANDAS_FREQ, FrameworkConfig
from core.models import FeatureStatus

logger = logging.getLogger(__name__)

OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")
MICROSTRUCTURE: tuple[str, ...] = ("bid_volume", "ask_volume", "trade_side", "bid_depth", "ask_depth")
SENTIMENT: tuple[str, ...] = ("social_sentiment", "news_sentiment", "funding_rate")
ONCHAIN: tuple[str, ...] = ("exchange_netflow", "stablecoin_supply_change")
OPTIONS: tuple[str, ...] = ("put_call_skew",)
CONTEXT: tuple[str, ...] = ("india_vix", "index_close")  # market-wide series joined onto a stock's bars


class DataValidationError(ValueError):
    """Raised when input bars violate the data contract."""


class LiveFeedUnavailableError(RuntimeError):
    """Raised by the live-feed stub: no real provider is connected."""


@dataclass
class MarketData:
    """Validated bar data plus per-column provenance.

    ``first_trade_index`` is the position of the first bar at or after the
    requested start; earlier rows are warm-up history only.
    """

    symbol: str
    bars: pd.DataFrame
    availability: dict[str, FeatureStatus]
    source: str
    first_trade_index: int = 0
    meta: dict[str, str] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Interface for any data source (synthetic, CSV, exchange REST/WebSocket)."""

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Return validated bars for ``[start, end)`` plus warm-up history."""
        ...


def parse_utc_date(text: str) -> datetime:
    """Parse ``YYYY-MM-DD`` (or ISO datetime) as a UTC timestamp.

    Naive inputs are interpreted as UTC; aware inputs are converted to UTC.
    """
    ts = pd.Timestamp(text)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def validate_bars(bars: pd.DataFrame, min_bars: int = 1, required: tuple[str, ...] = OHLCV) -> None:
    """Validate a bar DataFrame against the data contract.

    Checks: tz-aware DatetimeIndex, strictly increasing, no duplicates,
    required columns present and non-missing, positive prices, OHLC
    consistency, non-negative volumes, and minimum length.

    Raises:
        DataValidationError: With the first violated rule.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise DataValidationError("bars must be indexed by a DatetimeIndex")
    if bars.index.tz is None:
        raise DataValidationError("bar timestamps must be timezone-aware (UTC)")
    if bars.index.has_duplicates:
        dups = bars.index[bars.index.duplicated()].unique()[:3]
        raise DataValidationError(f"duplicate bar timestamps: {list(map(str, dups))}")
    if not bars.index.is_monotonic_increasing:
        raise DataValidationError("bar timestamps must be strictly increasing")
    missing_cols = [c for c in required if c not in bars.columns]
    if missing_cols:
        raise DataValidationError(f"missing required columns: {missing_cols}")
    if len(bars) < min_bars:
        raise DataValidationError(f"insufficient history: {len(bars)} bars < required {min_bars}")
    na = bars[list(required)].isna().sum()
    if na.any():
        raise DataValidationError(f"missing values in required columns: {na[na > 0].to_dict()}")
    prices = bars[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or not np.isfinite(prices.to_numpy()).all():
        raise DataValidationError("prices must be finite and strictly positive")
    if (bars["high"] < bars[["open", "close"]].max(axis=1) - 1e-9).any():
        raise DataValidationError("high below max(open, close)")
    if (bars["low"] > bars[["open", "close"]].min(axis=1) + 1e-9).any():
        raise DataValidationError("low above min(open, close)")
    if (bars["volume"] < 0).any():
        raise DataValidationError("volume must be non-negative")
    for col in ("bid_volume", "ask_volume", "bid_depth", "ask_depth"):
        if col in bars.columns and (bars[col].dropna() < 0).any():
            raise DataValidationError(f"{col} must be non-negative")
    if "trade_side" in bars.columns and (bars["trade_side"].dropna().abs() > 1 + 1e-9).any():
        raise DataValidationError("trade_side must lie in [-1, 1]")


class SyntheticFeed:
    """Deterministic multi-regime synthetic feed (labelled SYNTHETIC)."""

    REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range", "volatile", "noisy")

    def __init__(self, config: FrameworkConfig, seed: int | None = None) -> None:
        """Create a feed; ``seed`` overrides ``config.app.seed``."""
        self.config = config
        self.seed = config.app.seed if seed is None else seed

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Generate bars for ``[start, end)`` plus warm-up history."""
        if start.tzinfo is None or end.tzinfo is None:
            raise DataValidationError("start and end must be timezone-aware")
        if end <= start:
            raise DataValidationError(f"end ({end}) must be after start ({start})")
        warm = self.config.data.warmup_bars
        if self.config.data.market == "nse":
            # Weekday sessions labelled 09:15 IST (exchange holidays are not modelled).
            if self.config.data.timeframe != "1d":
                raise DataValidationError("synthetic NSE data supports timeframe 1d only")
            tz = self.config.app.timezone
            s0 = pd.Timestamp(start).tz_convert(tz).normalize()
            days = pd.bdate_range(s0 - pd.offsets.BDay(warm), pd.Timestamp(end).tz_convert(tz), tz=tz)
            index = days + pd.Timedelta(hours=9, minutes=15)
            index = index[index < pd.Timestamp(end)]
        else:
            freq = TIMEFRAME_PANDAS_FREQ[self.config.data.timeframe]
            offset = pd.tseries.frequencies.to_offset(freq)
            first = pd.Timestamp(start) - warm * offset
            index = pd.date_range(first, pd.Timestamp(end), freq=freq, inclusive="left", tz=timezone.utc)
        n_trade = int((index >= pd.Timestamp(start)).sum())
        if n_trade < 2:
            raise DataValidationError("requested range produces fewer than 2 tradable bars")
        bars, regimes = self._generate(len(index))
        bars.index = index
        bars["hidden_regime"] = regimes  # ground truth for diagnostics only; agents never read it
        validate_bars(bars, min_bars=self.config.data.min_history_bars)
        availability = {c: FeatureStatus.SYNTHETIC for c in OHLCV + MICROSTRUCTURE + SENTIMENT}
        for c in ONCHAIN:
            availability[c] = (
                FeatureStatus.SYNTHETIC if self.config.synthetic.generate_optional else FeatureStatus.UNAVAILABLE
            )
        for c in OPTIONS + CONTEXT:
            availability[c] = FeatureStatus.UNAVAILABLE
        logger.info(
            "synthetic feed: symbol=%s seed=%d bars=%d (warm-up %d, tradable %d) %s..%s",
            symbol, self.seed, len(bars), len(bars) - n_trade, n_trade, index[0], index[-1],
        )
        return MarketData(
            symbol=symbol,
            bars=bars,
            availability=availability,
            source=f"synthetic(seed={self.seed})",
            first_trade_index=len(bars) - n_trade,
            meta={"timeframe": self.config.data.timeframe, "seed": str(self.seed)},
        )

    def _generate(self, n: int) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate ``n`` bars of synthetic data and the hidden regime labels."""
        sc = self.config.synthetic
        rng = np.random.default_rng(self.seed)
        bpy = self.config.bars_per_year
        sigma0 = sc.base_vol_annual / np.sqrt(bpy)

        regimes = np.empty(n, dtype=object)
        pos, prev = 0, None
        while pos < n:
            choices = [r for r in self.REGIMES if r != prev]
            reg = choices[int(rng.integers(len(choices)))]
            length = int(rng.integers(sc.min_segment_bars, sc.max_segment_bars + 1))
            regimes[pos : pos + length] = reg
            pos += length
            prev = reg
        regimes = regimes[:n]

        sig_mult = {"trend_up": 1.0, "trend_down": 1.0, "range": 0.7, "volatile": 2.6, "noisy": 1.5}
        drift_k = {"trend_up": 0.15, "trend_down": -0.15, "range": 0.0, "volatile": 0.0, "noisy": 0.0}
        sigma = np.array([sigma0 * sig_mult[r] for r in regimes])
        drift_sign = np.array([np.sign(drift_k[r]) for r in regimes])

        logp = np.empty(n)
        r = np.empty(n)
        lp = np.log(sc.start_price)
        anchor = lp
        for i in range(n):
            reg = regimes[i]
            if i > 0 and regimes[i] != regimes[i - 1]:
                anchor = lp
            eps = rng.standard_normal()
            if reg == "volatile" and rng.random() < 0.03:
                eps += rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)  # jump
            ret = drift_k[reg] * sigma0 + sigma[i] * eps
            if reg == "range":
                ret += -0.05 * (lp - anchor)
            lp += ret
            logp[i] = lp
            r[i] = ret
        close = np.exp(logp)
        prev_close = np.concatenate([[sc.start_price], close[:-1]])
        open_ = prev_close * np.exp(0.1 * sigma * rng.standard_normal(n))
        hi_ext = np.abs(rng.standard_normal(n)) * 0.6 * sigma
        lo_ext = np.abs(rng.standard_normal(n)) * 0.6 * sigma
        high = np.maximum(open_, close) * np.exp(hi_ext)
        low = np.minimum(open_, close) * np.exp(-lo_ext)

        vol_mult = np.where(regimes == "volatile", 1.8, np.where(regimes == "range", 0.8, 1.0))
        volume = sc.base_volume * vol_mult * np.exp(0.3 * rng.standard_normal(n)) * (1 + 0.3 * np.abs(r) / sigma)

        k = sc.flow_signal_strength
        side = np.clip(
            0.5 * np.tanh(r / sigma) + k * drift_sign + 0.25 * rng.standard_normal(n), -0.95, 0.95
        )
        bid_volume = volume * (1 + side) / 2
        ask_volume = volume * (1 - side) / 2
        depth_base = sc.base_volume * 2.0
        tilt = k * drift_sign + 0.15 * rng.standard_normal(n)
        bid_depth = depth_base * np.exp(0.25 * rng.standard_normal(n)) * np.clip(1 + tilt, 0.2, None)
        ask_depth = depth_base * np.exp(0.25 * rng.standard_normal(n)) * np.clip(1 - tilt, 0.2, None)

        past_ret = pd.Series(logp).diff(20).shift(1).fillna(0.0).to_numpy()  # strictly past
        social = np.empty(n)
        news = np.empty(n)
        funding = np.empty(n)
        s, nw, f = 50.0, 0.0, 0.0001
        for i in range(n):
            s = 0.92 * s + 0.08 * (50 + 250 * past_ret[i]) + 2.5 * rng.standard_normal()
            s = float(np.clip(s, 0, 100))
            nw = 0.9 * nw + 0.1 * (0.6 * k * 4 * drift_sign[i]) + 0.12 * rng.standard_normal()
            nw = float(np.clip(nw, -1, 1))
            f = 0.9 * f + 0.1 * (0.0001 + 0.0015 * np.tanh(8 * past_ret[i])) + 0.00008 * rng.standard_normal()
            social[i], news[i], funding[i] = s, nw, f
        netflow = -k * 2 * drift_sign * sc.base_volume * 0.1 + sc.base_volume * 0.1 * rng.standard_normal(n)
        stable = 0.0005 * k * 2 * drift_sign + 0.001 * rng.standard_normal(n)

        df = pd.DataFrame(
            {
                "open": open_, "high": high, "low": low, "close": close, "volume": volume,
                "bid_volume": bid_volume, "ask_volume": ask_volume, "trade_side": side,
                "bid_depth": bid_depth, "ask_depth": ask_depth,
                "social_sentiment": social, "news_sentiment": news, "funding_rate": funding,
            }
        )
        if sc.generate_optional:
            df["exchange_netflow"] = netflow
            df["stablecoin_supply_change"] = stable
        return df, regimes


class LiveFeedStub:
    """Placeholder for a real market-data provider. **Not a working connection.**

    Implement ``get_bars`` against an exchange/vendor API, label real columns
    ``FeatureStatus.OBSERVED`` and missing ones ``UNAVAILABLE`` (never fill
    them with synthetic values), then call :func:`validate_bars`.
    """

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Always raises: no provider is connected and synthetic fallback is forbidden."""
        raise LiveFeedUnavailableError(
            f"live feed for {symbol} is a stub: no market-data provider is connected. "
            "Implement data.feed.LiveFeedStub.get_bars (or a new MarketDataProvider) "
            "or run with --synthetic. The framework never falls back to synthetic data silently."
        )
