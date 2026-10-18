"""RiskEngineAgent: market-level risk score and hard veto input.

This agent never votes directionally (always HOLD). Inside the boss its weight
therefore enters the denominator of the directional score as a deliberate
*dampener*, and its ``risk_score`` drives the hard veto. Portfolio drawdown
and the kill switch are handled separately in ``core.portfolio_risk``.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import RiskEngineParams
from core.models import AgentSignal, MarketState, SignalType


def _scale(x: float, lo: float, hi: float) -> float:
    """Linear map of x from [lo, hi] onto [0, 1], clipped."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


class RiskEngineAgent(BaseAgent):
    """Composite market risk in [0, 1].

    Components (each clipped to [0, 1]):
      * vol_score  = (RV_annual - rv_low) / (rv_high - rv_low)
      * atr_score  = (ATR% - atr_pct_low) / (atr_pct_high - atr_pct_low)
      * dd_score   = |rolling market drawdown| / drawdown_full
    risk_score = weighted mean with (w_vol, w_atr, w_drawdown).
    confidence = 1 - risk_score (how comfortable the agent is).
    """

    name = "risk_engine"
    essential_features = ("realized_vol", "atr_pct", "market_drawdown")

    def __init__(self, weight: float, params: RiskEngineParams, veto_threshold: float) -> None:
        """Create the agent with normalization params and the boss veto threshold."""
        super().__init__(weight)
        self.params = params
        self.veto_threshold = veto_threshold

    def risk_components(self, rv: float, atr_pct: float, drawdown: float) -> dict[str, float]:
        """Return the individual component scores and the composite risk score."""
        p = self.params
        vol_s = _scale(rv, p.rv_low, p.rv_high)
        atr_s = _scale(atr_pct, p.atr_pct_low, p.atr_pct_high)
        dd_s = max(0.0, min(1.0, abs(min(drawdown, 0.0)) / p.drawdown_full))
        wsum = p.w_vol + p.w_atr + p.w_drawdown
        score = (p.w_vol * vol_s + p.w_atr * atr_s + p.w_drawdown * dd_s) / wsum
        return {"vol_score": vol_s, "atr_score": atr_s, "drawdown_score": dd_s, "risk_score": score}

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Compute the market risk score; always HOLD, with veto flag in features."""
        missing = self.check_inputs(market_state)
        if missing:
            return self.unavailable(
                market_state, missing, "Risk cannot be scored; the boss treats missing risk as a veto (fail-safe)."
            )
        rv = market_state.get("realized_vol")
        atrp = market_state.get("atr_pct")
        dd = market_state.get("market_drawdown")
        comp = self.risk_components(rv, atrp, dd)
        rs = comp["risk_score"]
        veto = rs > self.veto_threshold
        level = "extreme" if veto else "elevated" if rs > 0.5 else "moderate" if rs > 0.25 else "low"
        reason = (
            f"HOLD (risk agent does not vote direction): {level} market risk, risk_score {rs:.2f} "
            f"({'ABOVE' if veto else 'below'} veto {self.veto_threshold:.2f}); components: realized vol "
            f"{rv:.0%} -> {comp['vol_score']:.2f}, ATR {atrp:.2%} -> {comp['atr_score']:.2f}, "
            f"market drawdown {dd:.1%} -> {comp['drawdown_score']:.2f}"
        )
        return AgentSignal(
            agent_name=self.name,
            signal=SignalType.HOLD,
            confidence=1.0 - rs,
            reason=reason,
            features={
                "realized_vol": rv, "atr_pct": atrp, "market_drawdown": dd, **comp, "risk_veto": veto,
                "parkinson_vol": market_state.value("parkinson_vol"),
                "garman_klass_vol": market_state.value("garman_klass_vol"),
            },
            timestamp=market_state.timestamp,
            summary=f"risk_score {rs:.2f} ({level})",
        )
