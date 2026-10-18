"""Volatility and drawdown indicators.

Return conventions
------------------
* ``realized_volatility``: sample stdev (ddof=1) of 1-bar **log** returns,
  annualized by ``sqrt(bars_per_year)``.
* ``parkinson_volatility`` / ``garman_klass_volatility``: range estimators on
  **log** price ratios, annualized the same way.
* ``atr_percent``: Wilder ATR divided by the close (a simple-price ratio).

``bars_per_year`` is supplied by the caller (derived from the configured
timeframe; crypto trades 24/7 so 4h bars -> 2190 bars per year).
Every window is trailing; insufficient history yields NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _check_positive(*series: pd.Series) -> None:
    """Raise if any non-NaN price is <= 0."""
    for s in series:
        if ((s <= 0) & s.notna()).any():
            raise ValueError("prices must be strictly positive")


def validate_ohlc(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    """Validate OHLC consistency (positive, high >= max(open, close), low <= min).

    Raises:
        ValueError: Describing the first violated rule.
    """
    _check_positive(open_, high, low, close)
    tol = 1e-9
    if (high + tol < np.maximum(open_, close)).any():
        raise ValueError("high must be >= max(open, close)")
    if (low - tol > np.minimum(open_, close)).any():
        raise ValueError("low must be <= min(open, close)")
    if (high + tol < low).any():
        raise ValueError("high must be >= low")


def realized_volatility(close: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    """Annualized realized volatility of log returns over a trailing window.

    Returns 0.0 for a constant price series (zero variance) once the window is
    full, and NaN before that.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")
    c = close.astype(float)
    _check_positive(c)
    r = np.log(c).diff()
    sd = r.rolling(window, min_periods=window).std(ddof=1)
    return (sd * math.sqrt(bars_per_year)).rename("realized_vol")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(H-L, |H-C_prev|, |L-C_prev|); first bar uses H-L."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[:1] = (high - low).iloc[:1]
    return tr.rename("true_range")


def wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: seed = SMA of the first ``period`` valid values, then
    ``s_t = s_{t-1} + (x_t - s_{t-1}) / period``.

    NaN inputs after seeding propagate the previous value (no update).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid_idx = np.flatnonzero(~np.isnan(x))
    if len(valid_idx) < period:
        return pd.Series(out, index=values.index)
    seed_pos = valid_idx[period - 1]
    out[seed_pos] = np.mean(x[valid_idx[:period]])
    prev = out[seed_pos]
    for i in range(seed_pos + 1, len(x)):
        if not np.isnan(x[i]):
            prev = prev + (x[i] - prev) / period
        out[i] = prev
    return pd.Series(out, index=values.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average true range with Wilder smoothing."""
    _check_positive(high, low, close)
    return wilder_smooth(true_range(high, low, close), window).rename("atr")


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """ATR as a fraction of the current close (0.02 = 2%)."""
    return (atr(high, low, close, window) / close).rename("atr_pct")


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    """Parkinson range volatility: sqrt(mean(ln(H/L)^2) / (4 ln 2)), annualized."""
    if window < 1:
        raise ValueError("window must be >= 1")
    _check_positive(high, low)
    hl = np.log(high / low) ** 2
    var = hl.rolling(window, min_periods=window).mean() / (4.0 * math.log(2.0))
    return (np.sqrt(var) * math.sqrt(bars_per_year)).rename("parkinson_vol")


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, bars_per_year: int
) -> pd.Series:
    """Garman-Klass volatility, annualized.

    Per-bar variance: 0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2, averaged over
    the window. A negative window mean (possible with tiny ranges) is floored at
    zero before the square root.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    _check_positive(open_, high, low, close)
    term = 0.5 * np.log(high / low) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(close / open_) ** 2
    var = term.rolling(window, min_periods=window).mean().clip(lower=0.0)
    return (np.sqrt(var) * math.sqrt(bars_per_year)).rename("garman_klass_vol")


def rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Drawdown from the trailing ``window``-bar high: close / max - 1 (<= 0)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    _check_positive(close)
    peak = close.rolling(window, min_periods=window).max()
    return (close / peak - 1.0).rename("market_drawdown")
