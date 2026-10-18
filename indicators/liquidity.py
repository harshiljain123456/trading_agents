"""Liquidity indicators: sweeps, absorption and depth imbalance.

Sweep definition (causal, same-bar confirmation)
-------------------------------------------------
Let ``prior_low_t = min(low_{t-L}, ..., low_{t-1})`` (the trailing window
*excluding* bar t) and ``prior_high_t`` analogously.

* Bullish sweep at t: ``low_t < prior_low_t`` and ``close_t > prior_low_t``.
  Price traded through the prior low and closed back above it.
* Bearish sweep at t: ``high_t > prior_high_t`` and ``close_t < prior_high_t``.

The reversal is confirmed by bar t's close, which is known when the decision
is made after bar t closes, so no future bar is used. A bar that sweeps both
sides is flagged as both (the agent treats that as ambiguous).

A sweep is a price pattern; it does not prove institutional participation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_sweeps(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.DataFrame:
    """Flag bullish/bearish liquidity sweeps using prior-window levels only.

    Returns:
        DataFrame with bool columns ``bullish_sweep``, ``bearish_sweep`` and
        float columns ``prior_low``, ``prior_high`` (NaN during warm-up; sweeps
        are False there).
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    prior_low = low.shift(1).rolling(lookback, min_periods=lookback).min()
    prior_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
    bull = (low < prior_low) & (close > prior_low)
    bear = (high > prior_high) & (close < prior_high)
    return pd.DataFrame(
        {
            "bullish_sweep": bull.fillna(False).astype(bool),
            "bearish_sweep": bear.fillna(False).astype(bool),
            "prior_low": prior_low,
            "prior_high": prior_high,
        },
        index=close.index,
    )


def absorption_ratio(
    volume: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, min_range_ratio: float = 0.1
) -> pd.Series:
    """Relative volume divided by relative price range.

    Definition at bar t (baseline = mean of the *previous* ``window`` bars)::

        rel_vol   = V_t / mean(V_{t-window..t-1})
        range_t   = (H_t - L_t) / C_t
        rel_range = range_t / mean(range_{t-window..t-1})
        AR_t      = rel_vol / max(rel_range, min_range_ratio)

    AR > 1 means more volume than usual was needed per unit of price movement:
    consistent with passive orders absorbing aggressive flow. Without genuine
    aggressor-side and order-book data this cannot distinguish absorption from
    simple two-sided churn; the direction must come from CVD/depth.
    ``min_range_ratio`` caps the ratio when a bar has (near) zero range.
    NaN when the baseline volume or range is zero or history is insufficient.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_range_ratio <= 0:
        raise ValueError("min_range_ratio must be > 0")
    v = volume.astype(float)
    rng = (high - low) / close
    base_v = v.shift(1).rolling(window, min_periods=window).mean()
    base_r = rng.shift(1).rolling(window, min_periods=window).mean()
    rel_v = v / base_v.where(base_v > 0)
    rel_r = rng / base_r.where(base_r > 0)
    return (rel_v / np.maximum(rel_r, min_range_ratio)).rename("absorption_ratio")


def depth_imbalance(bid_depth: pd.Series | float, ask_depth: pd.Series | float) -> pd.Series | float:
    """Resting-depth imbalance (bid - ask) / (bid + ask) in [-1, 1].

    Positive = more resting bids than asks near the touch. NaN if both depths
    are zero or either is missing. Accepts scalars or Series.
    """
    b = bid_depth if isinstance(bid_depth, pd.Series) else np.float64(bid_depth)
    a = ask_depth if isinstance(ask_depth, pd.Series) else np.float64(ask_depth)
    if np.any(np.asarray(b) < 0) or np.any(np.asarray(a) < 0):
        raise ValueError("depth must be non-negative")
    total = b + a
    if isinstance(total, pd.Series):
        return ((b - a) / total.where(total > 0)).rename("depth_imbalance")
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float((b - a) / total)
