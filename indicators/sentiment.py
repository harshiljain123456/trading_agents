"""Sentiment indicators.

Every component is mapped to [-1, 1] where +1 is bullish:

* ``normalize_score``: linear map of a raw range [lo, hi] onto [-1, 1]
  (midpoint -> 0), clipped.
* ``funding_contrarian``: perpetual funding is treated contrarian. Inside
  +-threshold the term is 0; beyond it, the term grows linearly to -sign(f)
  at ``saturation``. Very positive funding (crowded longs) -> bearish term.
* ``composite_sentiment``: weighted average over the *available* components
  with weights renormalized; a missing component is excluded, never assumed
  neutral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _is_num(x: float | None) -> bool:
    """True for finite floats."""
    return x is not None and not isinstance(x, bool) and math.isfinite(float(x))


def normalize_score(value: float | None, lo: float, hi: float) -> float:
    """Map ``value`` from [lo, hi] to [-1, 1]; NaN if missing.

    Raises:
        ValueError: If ``lo >= hi``.
    """
    if lo >= hi:
        raise ValueError("lo must be < hi")
    if not _is_num(value):
        return float("nan")
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return max(-1.0, min(1.0, (float(value) - mid) / half))


def funding_contrarian(funding: float | None, threshold: float, saturation: float) -> float:
    """Contrarian funding term in [-1, 1]; NaN if funding is missing.

    term = -sign(f) * min(1, (|f| - threshold) / (saturation - threshold)) when
    |f| > threshold, else 0.
    """
    if threshold < 0 or saturation <= threshold:
        raise ValueError("require 0 <= threshold < saturation")
    if not _is_num(funding):
        return float("nan")
    f = float(funding)
    excess = abs(f) - threshold
    if excess <= 0:
        return 0.0
    mag = min(1.0, excess / (saturation - threshold))
    return -math.copysign(mag, f)


@dataclass(frozen=True)
class CompositeSentiment:
    """Result of :func:`composite_sentiment`."""

    score: float
    available_components: int
    weight_coverage: float


def composite_sentiment(
    social: float | None,
    news: float | None,
    funding_term: float | None,
    w_social: float,
    w_news: float,
    w_funding: float,
) -> CompositeSentiment:
    """Weighted average of the available normalized components.

    Args:
        social, news, funding_term: Components already in [-1, 1] (NaN/None = missing).
        w_social, w_news, w_funding: Non-negative component weights.

    Returns:
        ``CompositeSentiment`` with score (NaN if no component is available),
        count of available components and the share of total weight they cover.
    """
    weights = (w_social, w_news, w_funding)
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    num = 0.0
    den = 0.0
    n = 0
    for comp, w in zip((social, news, funding_term), weights):
        if _is_num(comp):
            num += w * float(comp)  # type: ignore[arg-type]
            den += w
            n += 1
    if den <= 0:
        return CompositeSentiment(float("nan"), n, 0.0)
    return CompositeSentiment(max(-1.0, min(1.0, num / den)), n, den / sum(weights))
