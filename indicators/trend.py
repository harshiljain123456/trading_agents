"""Trend indicators: EMA and Wilder ADX.

ADX convention (Wilder, 1978)
-----------------------------
* +DM_t = H_t - H_{t-1} if it exceeds L_{t-1} - L_t and is > 0, else 0.
* -DM_t = L_{t-1} - L_t if it exceeds H_t - H_{t-1} and is > 0, else 0.
* TR, +DM and -DM are Wilder-smoothed over ``period`` (seed = SMA).
* +DI = 100 * sm(+DM) / sm(TR), -DI likewise.
* DX = 100 * |+DI - -DI| / (+DI + -DI) (0 when both are 0).
* ADX = Wilder smoothing of DX over ``period``.

Warm-up: the first ADX value appears at bar ``2 * period - 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.volatility import true_range, wilder_smooth


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2/(span+1)), NaN until ``span`` bars."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return series.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI and -DI.

    Returns:
        DataFrame with columns ``adx``, ``plus_di``, ``minus_di``.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    tr = true_range(high, low, close)
    tr.iloc[0] = np.nan
    s_tr = wilder_smooth(tr, period)
    s_p = wilder_smooth(plus_dm, period)
    s_m = wilder_smooth(minus_dm, period)
    safe_tr = s_tr.where(s_tr > 0)
    plus_di = 100.0 * s_p / safe_tr
    minus_di = 100.0 * s_m / safe_tr
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum > 0)
    dx = dx.where(~((di_sum == 0) & s_tr.notna()), 0.0)
    out = pd.DataFrame(
        {"adx": wilder_smooth(dx, period), "plus_di": plus_di, "minus_di": minus_di}, index=high.index
    )
    return out
