"""Order-flow indicators.

Conventions
-----------
* ``bid_vol`` is **buyer-initiated** (aggressive buy) volume and ``ask_vol`` is
  **seller-initiated** (aggressive sell) volume in the same bar. With real data
  these come from trade prints classified by the exchange's aggressor flag.
* Order-flow imbalance over a trailing window ``W`` ending at bar t::

      OFI_t = (sum_{t-W+1..t} bid_vol - sum ask_vol) / (sum bid_vol + sum ask_vol)

  OFI lies in [-1, 1]; +1 means every traded unit was buyer-initiated.
  Positive OFI = buying pressure. Undefined (NaN) when the window has zero
  total volume or fewer than W observations.
* All rolling windows are trailing (end at the current bar); no value at t
  depends on bars after t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x: pd.Series | np.ndarray | list, name: str) -> pd.Series:
    """Coerce to float Series, raising on non-numeric input."""
    s = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype=float))
    try:
        return s.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def order_flow_imbalance(bid_vol: pd.Series, ask_vol: pd.Series, window: int) -> pd.Series:
    """Trailing-window order-flow imbalance in [-1, 1].

    Args:
        bid_vol: Buyer-initiated volume per bar (>= 0).
        ask_vol: Seller-initiated volume per bar (>= 0).
        window: Number of trailing bars (>= 1).

    Returns:
        Series aligned to the input. NaN during the first ``window - 1`` bars,
        when any input in the window is NaN, or when window volume is zero.

    Raises:
        ValueError: On negative volumes, mismatched lengths or window < 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    b = _as_series(bid_vol, "bid_vol")
    a = _as_series(ask_vol, "ask_vol")
    if len(b) != len(a):
        raise ValueError("bid_vol and ask_vol must have equal length")
    if (b < 0).any() or (a < 0).any():
        raise ValueError("volumes must be non-negative")
    a.index = b.index
    sb = b.rolling(window, min_periods=window).sum()
    sa = a.rolling(window, min_periods=window).sum()
    total = sb + sa
    ofi = (sb - sa) / total.where(total > 0)
    return ofi.rename("ofi")


def cvd(price: pd.Series, volume: pd.Series, side: pd.Series | None) -> pd.Series:
    """Cumulative volume delta.

    ``side`` is the **net aggressor fraction** of each bar in [-1, 1]:
    +1 = all buyer-initiated, -1 = all seller-initiated, 0 = balanced or
    ambiguous. Signed volume is ``side * volume`` so that, when ``volume`` is
    total traded volume, the per-bar delta equals buy volume minus sell volume.

    NaN ``side`` values (unknown aggressor) contribute **zero** delta; they are
    not guessed.

    If ``side`` is ``None`` the classic *tick rule* is used as an explicit
    **proxy**: the sign of the close-to-close price change, carrying the last
    non-zero sign through unchanged prices (first bar = 0). This is not
    genuine aggressor data and callers must label the result as a proxy.

    Args:
        price: Close prices (used only by the tick-rule proxy and for alignment).
        volume: Traded volume per bar (>= 0, NaN treated as 0).
        side: Net aggressor fraction or ``None`` for the tick-rule proxy.

    Returns:
        Cumulative sum of signed volume (Series named ``cvd``).
    """
    p = _as_series(price, "price")
    v = _as_series(volume, "volume")
    if len(p) != len(v):
        raise ValueError("price and volume must have equal length")
    v.index = p.index
    if (v < 0).any():
        raise ValueError("volume must be non-negative")
    if side is None:
        sign = np.sign(p.diff()).replace(0.0, np.nan).ffill().fillna(0.0)
    else:
        s = _as_series(side, "side")
        if len(s) != len(p):
            raise ValueError("side must match price length")
        s.index = p.index
        if ((s.abs() > 1.0 + 1e-12) & s.notna()).any():
            raise ValueError("side must lie in [-1, 1]")
        sign = s.fillna(0.0)
    delta = (sign * v.fillna(0.0))
    return delta.cumsum().rename("cvd")


def cvd_change_normalized(cvd_series: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Change of CVD over ``window`` bars divided by volume traded in that span.

    Lies in [-1, 1]; NaN when insufficient history or zero window volume.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = _as_series(cvd_series, "cvd")
    v = _as_series(volume, "volume")
    v.index = c.index
    dc = c - c.shift(window)
    vol = v.rolling(window, min_periods=window).sum()
    return (dc / vol.where(vol > 0)).clip(-1.0, 1.0).rename("cvd_norm")


def institutional_momentum(price: pd.Series, volume: pd.Series, lookback: int) -> pd.Series:
    """Volume-confirmed momentum score in [-1, 1] (an institutional-*style* proxy).

    Definition at bar t with lookback L::

        R_t   = ln(P_t / P_{t-L})
        sigma = stdev(1-bar log returns over the last L bars, ddof=1)
        z_t   = R_t / (sigma * sqrt(L))              (trend strength in sd units)
        rv_t  = mean(V over last L bars) / mean(V over the L bars before that)
        score = tanh(z_t) * min(rv_t, 2) / 2

    The volume factor scales the score by participation: equal volume gives
    half weight, doubled (or more) volume gives full weight. High volume alone
    does not imply institutional activity; this is only a proxy.

    Edge cases: needs ``2 * L`` bars; zero prior volume -> NaN;
    zero return variance -> score 0 if R_t == 0 else sign(R_t) * volume factor.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    p = _as_series(price, "price")
    v = _as_series(volume, "volume")
    v.index = p.index
    if (p <= 0).any():
        raise ValueError("prices must be positive")
    logp = np.log(p)
    r = logp.diff()
    R = logp - logp.shift(lookback)
    sigma = r.rolling(lookback, min_periods=lookback).std(ddof=1)
    vol_recent = v.rolling(lookback, min_periods=lookback).mean()
    vol_prior = vol_recent.shift(lookback)
    vratio = vol_recent / vol_prior.where(vol_prior > 0)
    vfac = vratio.clip(upper=2.0) / 2.0
    denom = sigma * np.sqrt(lookback)
    z = R / denom.where(denom > 0)
    trend = np.tanh(z)
    zero_var = (denom == 0) & R.notna()
    trend = trend.where(~zero_var, np.sign(R))
    score = trend * vfac
    return score.rename("inst_momentum")


def price_volume_correlation(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between 1-bar log returns and log volume change.

    Positive values mean volume expands with up moves (and contracts on down
    moves). NaN when either series is constant in the window.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    p = _as_series(price, "price")
    v = _as_series(volume, "volume")
    v.index = p.index
    r = np.log(p).diff()
    dv = np.log(v.where(v > 0)).diff()
    corr = r.rolling(window, min_periods=window).corr(dv)
    corr = corr.where(np.isfinite(corr))
    return corr.rename("pv_corr")
