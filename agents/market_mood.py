"""MarketMoodAgent: NSE replacement for the crypto sentiment agent.

Uses only free, point-in-time daily market data (each value is known at the
close of the decision day):

* index_trend   - tanh((Nifty / EMA(Nifty, 100) - 1) / 0.05): broad market
                  direction (+ = market above its trend). Momentum reading.
* rel_strength  - tanh((stock 63-day return - Nifty 63-day return) / 0.10):
                  is this stock leading or lagging the market?
* vix_term      - India VIX (fear gauge) read at its extremes only, using its
                  percentile over the trailing 250 sessions:
                    pct >= vix_high_pct and VIX falling over 5 days -> +1
                      (panic peaking / receding: contrarian bullish)
                    pct >= vix_high_pct and VIX rising            -> -1
                      (fear still escalating: risk-off)
                    pct <= vix_low_pct                            -> -complacency_penalty
                      (complacency)
                    otherwise                                     -> 0

score = weighted mean of the available components (missing inputs are
excluded, never treated as neutral); needs ``min_components``.
BUY / SELL beyond +-signal_threshold; confidence = |score| * weight coverage.

These rules and constants were fixed before looking at any NSE backtest; they
are hypotheses to validate with walk-forward testing, not proven signals.
"""

from __future__ import annotations

import math

from agents.base_agent import BaseAgent
from core.config import MarketMoodParams
from core.models import AgentSignal, MarketState, SignalType


class MarketMoodAgent(BaseAgent):
    """Market regime/mood from India VIX, Nifty trend and relative strength."""

    name = "market_mood"
    essential_features: tuple[str, ...] = ()

    def __init__(self, weight: float, params: MarketMoodParams) -> None:
        """Create the agent with weight and scoring parameters."""
        super().__init__(weight)
        self.params = params

    def vix_term(self, pct: float, change: float) -> float:
        """Contrarian/risk-off reading of India VIX extremes (see module docstring)."""
        p = self.params
        if pct >= p.vix_high_pct:
            return 1.0 if change < 0 else -1.0
        if pct <= p.vix_low_pct:
            return -p.complacency_penalty
        return 0.0

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Score market mood at ``market_state``."""
        p = self.params
        comps: dict[str, float] = {}
        if market_state.is_valid("index_trend"):
            comps["index_trend"] = market_state.get("index_trend")
        if market_state.is_valid("rel_strength"):
            comps["rel_strength"] = market_state.get("rel_strength")
        vix_ok = market_state.is_valid("vix_pct") and market_state.is_valid("vix_change")
        if vix_ok:
            comps["vix_term"] = self.vix_term(market_state.get("vix_pct"), market_state.get("vix_change"))
        weights = {"index_trend": p.w_index_trend, "rel_strength": p.w_rel_strength, "vix_term": p.w_vix}
        present = {"index_trend": "index_trend" in comps, "rel_strength": "rel_strength" in comps, "india_vix": vix_ok}
        missing = [n for n, ok in present.items() if not ok]
        if len(comps) < p.min_components:
            return self.unavailable(
                market_state, missing,
                f"Only {len(comps)} of 3 market-mood inputs available (min {p.min_components}); "
                "India VIX / Nifty 50 context data is needed.",
            )
        wsum = sum(weights[k] for k in comps)
        score = max(-1.0, min(1.0, sum(weights[k] * v for k, v in comps.items()) / wsum))
        coverage = wsum / sum(weights.values())
        if score > p.signal_threshold:
            sig = SignalType.BUY
        elif score < -p.signal_threshold:
            sig = SignalType.SELL
        else:
            sig = SignalType.HOLD

        parts = []
        if "index_trend" in comps:
            t = comps["index_trend"]
            parts.append(f"Nifty trend {t:+.2f} ({'above' if t > 0 else 'below'} its 100-day EMA)")
        if "rel_strength" in comps:
            r = comps["rel_strength"]
            parts.append(f"relative strength {r:+.2f} ({'outperforming' if r > 0 else 'lagging'} Nifty over 3 months)")
        if vix_ok:
            vix = market_state.value("india_vix")
            pct = market_state.get("vix_pct")
            chg = market_state.get("vix_change")
            vix_txt = "n/a" if vix is None or not math.isfinite(vix) else f"{vix:.1f}"
            mood = ("panic receding (contrarian bullish)" if comps["vix_term"] > 0 else
                    "fear escalating (risk-off)" if comps["vix_term"] == -1.0 else
                    "complacency" if comps["vix_term"] < 0 else "normal range")
            parts.append(f"India VIX {vix_txt} at {pct:.0%} percentile, 5-day change {chg:+.1%} -> {mood}")
        lead = f"{sig.value}: " if sig != SignalType.HOLD else f"HOLD: mood score within +/-{p.signal_threshold:.2f}; "
        reason = lead + "; ".join(parts) + f"; score {score:+.3f} from {len(comps)}/3 inputs"
        if missing:
            reason += f" (missing: {', '.join(missing)})"
        return AgentSignal(
            agent_name=self.name,
            signal=sig,
            confidence=min(1.0, abs(score) * coverage),
            reason=reason,
            features={**comps, "india_vix": market_state.value("india_vix"),
                      "vix_pct": market_state.value("vix_pct"), "score": score, "coverage": coverage},
            timestamp=market_state.timestamp,
            summary=f"market mood {score:+.2f} ({len(comps)}/3 inputs)",
        )
