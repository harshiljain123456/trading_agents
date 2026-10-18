"""RegimeDetectorAgent: classify the market regime from EMA, ADX and volatility."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import RegimeParams
from core.models import AgentSignal, MarketState, Regime, SignalType


class RegimeDetectorAgent(BaseAgent):
    """Rule-based regime classifier (checked in this order).

    1. realized vol (annualized) >= ``volatile_rv``           -> volatile
    2. ADX >= adx_trend and EMA_fast > EMA_slow               -> trend_up
    3. ADX >= adx_trend and EMA_fast < EMA_slow               -> trend_down
    4. otherwise                                              -> range

    Trading vote (if ``directional_vote``): BUY in trend_up, SELL in
    trend_down, HOLD otherwise, with confidence
    ``min(1, (ADX - adx_trend) / (adx_strong - adx_trend))``, floored at 0.1 so
    a confirmed trend never votes with zero confidence.
    During warm-up (EMA_slow/ADX not yet defined) the regime is ``unknown``
    and the agent is unavailable.
    """

    name = "regime_detector"
    essential_features = ("ema_fast", "ema_slow", "adx", "realized_vol")

    def __init__(self, weight: float, params: RegimeParams) -> None:
        """Create the agent with weight and threshold parameters."""
        super().__init__(weight)
        self.params = params

    def classify(self, ema_fast: float, ema_slow: float, adx_value: float, rv: float) -> Regime:
        """Return the regime for the given readings."""
        p = self.params
        if rv >= p.volatile_rv:
            return Regime.VOLATILE
        if adx_value >= p.adx_trend and ema_fast > ema_slow:
            return Regime.TREND_UP
        if adx_value >= p.adx_trend and ema_fast < ema_slow:
            return Regime.TREND_DOWN
        return Regime.RANGE

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Classify the regime at ``market_state``."""
        missing = self.check_inputs(market_state)
        if missing:
            sig = self.unavailable(market_state, missing, "Regime is 'unknown' during indicator warm-up.")
            sig.features["regime"] = Regime.UNKNOWN.value
            return sig
        p = self.params
        ef, es = market_state.get("ema_fast"), market_state.get("ema_slow")
        a, rv = market_state.get("adx"), market_state.get("realized_vol")
        regime = self.classify(ef, es, a, rv)
        rel = ">" if ef > es else "<" if ef < es else "="
        gap = ef / es - 1.0
        readings = f"EMA_fast {ef:,.2f} {rel} EMA_slow {es:,.2f} ({gap:+.2%}), ADX {a:.1f} (trend >= {p.adx_trend:.0f}), realized vol {rv:.0%} (volatile >= {p.volatile_rv:.0%})"
        why = {
            Regime.VOLATILE: "volatility above the volatile threshold dominates",
            Regime.TREND_UP: "ADX confirms a trend and the fast EMA is above the slow EMA",
            Regime.TREND_DOWN: "ADX confirms a trend and the fast EMA is below the slow EMA",
            Regime.RANGE: "ADX below the trend threshold (no persistent direction)",
        }[regime]
        conf = 0.0
        sig = SignalType.HOLD
        if p.directional_vote and regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            sig = SignalType.BUY if regime == Regime.TREND_UP else SignalType.SELL
            conf = max(0.1, min(1.0, (a - p.adx_trend) / (p.adx_strong - p.adx_trend)))
        prefix = f"{sig.value}: " if sig != SignalType.HOLD else "HOLD (no directional vote in this regime): "
        reason = f"{prefix}regime '{regime.value}' because {why}; {readings}"
        return AgentSignal(
            agent_name=self.name,
            signal=sig,
            confidence=conf,
            reason=reason,
            features={"regime": regime.value, "ema_fast": ef, "ema_slow": es, "adx": a, "realized_vol": rv,
                      "plus_di": market_state.value("plus_di"), "minus_di": market_state.value("minus_di")},
            timestamp=market_state.timestamp,
            summary=f"regime {regime.value} (ADX {a:.0f}, EMA gap {gap:+.1%})",
        )
