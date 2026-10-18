"""Indicator feature bus.

Computes every indicator once over the whole bar history using strictly
trailing (causal) functions, then serves a ``MarketState`` per bar. Because
each indicator value at bar t depends only on bars <= t, precomputing over the
full series is equivalent to recomputing on a truncated history (this is
verified in ``tests/test_integration.py``).

Indicator availability is DERIVED when all inputs are available and
UNAVAILABLE otherwise, so a missing raw feed can never surface as a valid
derived feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import FrameworkConfig
from core.models import FeatureStatus, MarketState, RAW_FIELDS
from data.feed import MarketData, ONCHAIN, OPTIONS
from indicators import liquidity as liq
from indicators import order_flow as of
from indicators import trend as tr
from indicators import volatility as vol

logger = logging.getLogger(__name__)


@dataclass
class FeatureFrame:
    """Bars joined with indicators plus provenance for every column."""

    symbol: str
    frame: pd.DataFrame
    availability: dict[str, FeatureStatus]
    first_trade_index: int
    indicator_names: list[str]

    def __post_init__(self) -> None:
        self._arrays = {c: self.frame[c].to_numpy(dtype=float) for c in self.frame.columns if c != "hidden_regime"}
        self._index = self.frame.index
        self._timestamps = list(self.frame.index.to_pydatetime())  # boxed once; per-bar boxing is slow

    def __len__(self) -> int:
        return len(self.frame)

    def timestamp(self, i: int) -> "datetime":
        """Python datetime of bar ``i`` (cached)."""
        return self._timestamps[i]

    @property
    def index(self) -> pd.DatetimeIndex:
        """Bar timestamps."""
        return self._index

    def column(self, name: str) -> np.ndarray:
        """Numpy view of a numeric column."""
        return self._arrays[name]

    def state(self, i: int) -> MarketState:
        """Build the MarketState visible at the close of bar ``i``."""
        if not 0 <= i < len(self.frame):
            raise IndexError(f"bar index {i} out of range")
        a = self._arrays

        def raw(name: str) -> float | None:
            if self.availability.get(name, FeatureStatus.UNAVAILABLE) == FeatureStatus.UNAVAILABLE:
                return None
            return float(a[name][i]) if name in a else None

        indicators = {n: float(a[n][i]) for n in self.indicator_names}
        onchain = {n: float(a[n][i]) for n in ONCHAIN if n in a and self.availability.get(n) != FeatureStatus.UNAVAILABLE}
        options = {n: float(a[n][i]) for n in OPTIONS if n in a and self.availability.get(n) != FeatureStatus.UNAVAILABLE}
        return MarketState(
            timestamp=self._timestamps[i],
            symbol=self.symbol,
            open=float(a["open"][i]),
            high=float(a["high"][i]),
            low=float(a["low"][i]),
            close=float(a["close"][i]),
            volume=float(a["volume"][i]),
            bid_volume=raw("bid_volume"),
            ask_volume=raw("ask_volume"),
            trade_side=raw("trade_side"),
            bid_depth=raw("bid_depth"),
            ask_depth=raw("ask_depth"),
            social_sentiment=raw("social_sentiment"),
            news_sentiment=raw("news_sentiment"),
            funding_rate=raw("funding_rate"),
            onchain=onchain,
            options=options,
            indicators=indicators,
            availability=self.availability,
            bar_index=i,
        )


def compute_features(md: MarketData, config: FrameworkConfig) -> FeatureFrame:
    """Run every indicator over ``md.bars`` and return a :class:`FeatureFrame`."""
    ic = config.indicators
    bpy = config.bars_per_year
    b = md.bars
    av = dict(md.availability)
    for col in RAW_FIELDS:
        av.setdefault(col, FeatureStatus.UNAVAILABLE if col not in b.columns else av.get(col, FeatureStatus.OBSERVED))

    def ok(*cols: str) -> bool:
        return all(av.get(c, FeatureStatus.UNAVAILABLE) != FeatureStatus.UNAVAILABLE and c in b.columns for c in cols)

    nan = pd.Series(np.nan, index=b.index)
    feats: dict[str, pd.Series] = {}
    status: dict[str, FeatureStatus] = {}

    def put(name: str, series: pd.Series | None, inputs_ok: bool) -> None:
        feats[name] = series.astype(float) if (series is not None and inputs_ok) else nan
        status[name] = FeatureStatus.DERIVED if inputs_ok else FeatureStatus.UNAVAILABLE

    c, h, l, o, v = b["close"], b["high"], b["low"], b["open"], b["volume"]

    has_flow = ok("bid_volume", "ask_volume")
    put("ofi", of.order_flow_imbalance(b["bid_volume"], b["ask_volume"], ic.ofi_window) if has_flow else None, has_flow)
    has_side = ok("trade_side")
    cvd_series = of.cvd(c, v, b["trade_side"] if has_side else None)
    # CVD from the tick rule is a proxy; it is still derived but flagged.
    put("cvd_norm", of.cvd_change_normalized(cvd_series, v, ic.cvd_window), True)
    put("cvd_norm_short", of.cvd_change_normalized(cvd_series, v, 3), True)
    put("cvd_is_proxy", pd.Series(0.0 if has_side else 1.0, index=b.index), True)
    put("inst_momentum", of.institutional_momentum(c, v, ic.momentum_lookback), True)
    put("pv_corr", of.price_volume_correlation(c, v, ic.pv_corr_window), True)

    put("realized_vol", vol.realized_volatility(c, ic.rv_window, bpy), True)
    put("atr_pct", vol.atr_percent(h, l, c, ic.atr_window), True)
    put("parkinson_vol", vol.parkinson_volatility(h, l, ic.rv_window, bpy), True)
    put("garman_klass_vol", vol.garman_klass_volatility(o, h, l, c, ic.rv_window, bpy), True)
    put("market_drawdown", vol.rolling_drawdown(c, ic.drawdown_window), True)

    put("ema_fast", tr.ema(c, ic.ema_fast), True)
    put("ema_slow", tr.ema(c, ic.ema_slow), True)
    adx_df = tr.adx(h, l, c, ic.adx_period)
    for col in ("adx", "plus_di", "minus_di"):
        put(col, adx_df[col], True)

    sw = liq.detect_sweeps(h, l, c, ic.sweep_lookback)
    warm = sw["prior_low"].isna()
    put("bullish_sweep", sw["bullish_sweep"].astype(float).where(~warm), True)
    put("bearish_sweep", sw["bearish_sweep"].astype(float).where(~warm), True)
    put("prior_low", sw["prior_low"], True)
    put("prior_high", sw["prior_high"], True)
    put("absorption_ratio", liq.absorption_ratio(v, h, l, c, ic.absorption_window), True)
    has_depth = ok("bid_depth", "ask_depth")
    put("depth_imbalance", liq.depth_imbalance(b["bid_depth"], b["ask_depth"]) if has_depth else None, has_depth)

    has_netflow = ok("exchange_netflow")
    if has_netflow:
        nf = b["exchange_netflow"]
        mu = nf.rolling(ic.netflow_window, min_periods=ic.netflow_window).mean()
        sd = nf.rolling(ic.netflow_window, min_periods=ic.netflow_window).std(ddof=1)
        put("netflow_z", (nf - mu) / sd.where(sd > 0), True)
    else:
        put("netflow_z", None, False)
    has_stable = ok("stablecoin_supply_change")
    put("stablecoin_change", b["stablecoin_supply_change"] if has_stable else None, has_stable)

    # Market context (India VIX, Nifty 50) joined onto the stock's bars by the provider.
    has_vix = ok("india_vix")
    if has_vix:
        vix = b["india_vix"].astype(float)
        w = ic.vix_rank_window
        rank = vix.rolling(w, min_periods=w).apply(lambda a: float((a <= a[-1]).mean()), raw=True)
        put("india_vix", vix, True)
        status["india_vix"] = av["india_vix"]  # raw observed series, not derived
        put("vix_pct", rank, True)
        put("vix_change", np.log(vix / vix.shift(ic.vix_change_window)), True)
    else:
        for n in ("india_vix", "vix_pct", "vix_change"):
            put(n, None, False)
    has_idx = ok("index_close")
    if has_idx:
        idx = b["index_close"].astype(float)
        put("index_trend", np.tanh((idx / tr.ema(idx, ic.index_ema) - 1.0) / ic.index_trend_scale), True)
        L = ic.rs_lookback
        rs = (c / c.shift(L)) - (idx / idx.shift(L))
        put("rel_strength", np.tanh(rs / ic.rs_scale), True)
    else:
        put("index_trend", None, False)
        put("rel_strength", None, False)

    frame = b.copy()
    for name, series in feats.items():
        frame[name] = series.to_numpy()
    av.update(status)
    logger.debug("feature bus computed %d indicators over %d bars", len(feats), len(frame))
    return FeatureFrame(
        symbol=md.symbol,
        frame=frame,
        availability=av,
        first_trade_index=md.first_trade_index,
        indicator_names=list(feats.keys()),
    )
