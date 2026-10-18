"""Optional OnChainAgent (feature flag ``features.enable_optional_agent``)."""

from __future__ import annotations

import math

from agents.base_agent import BaseAgent
from core.config import OnChainParams
from core.models import AgentSignal, MarketState, SignalType


class OnChainAgent(BaseAgent):
    """Exchange netflow and stablecoin supply proxy.

    * netflow_z: z-score of net coins flowing *into* exchanges (trailing
      window). Inflows are treated as bearish (potential sell supply):
      term_netflow = -tanh(netflow_z).
    * stablecoin_change: fractional change in stablecoin supply, treated as
      bullish dry powder: term_stable = tanh(change / 0.001).
    score = (w_netflow*term_netflow + w_stable*term_stable) / (w_netflow + w_stable)
    BUY/SELL when |netflow_z| >= netflow_z_threshold and score agrees in sign.
    In synthetic mode these inputs are SYNTHETIC and say nothing about a real chain.
    """

    name = "on_chain"
    essential_features = ("netflow_z", "stablecoin_change")

    def __init__(self, weight: float, params: OnChainParams) -> None:
        """Create the agent with weight and scoring parameters."""
        super().__init__(weight)
        self.params = params

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Score on-chain flows at ``market_state``."""
        missing = self.check_inputs(market_state)
        if missing:
            return self.unavailable(market_state, missing, "No on-chain provider connected.")
        p = self.params
        z = market_state.get("netflow_z")
        sc = market_state.get("stablecoin_change")
        t_nf = -math.tanh(z)
        t_sc = math.tanh(sc / 0.001)
        score = (p.w_netflow * t_nf + p.w_stablecoin * t_sc) / (p.w_netflow + p.w_stablecoin)
        sig = SignalType.HOLD
        if abs(z) >= p.netflow_z_threshold and math.copysign(1, score) == math.copysign(1, t_nf) and score != 0:
            sig = SignalType.BUY if score > 0 else SignalType.SELL
        src = self.provenance(market_state, ("exchange_netflow", "stablecoin_supply_change"))
        flow = "outflow from" if z < 0 else "inflow to"
        reason = (
            f"{sig.value}: " if sig != SignalType.HOLD else f"HOLD: |netflow z| below {p.netflow_z_threshold:.2f} or terms disagree; "
        ) + f"exchange netflow z {z:+.2f} ({flow} exchanges), stablecoin supply change {sc:+.4%}, score {score:+.2f} [inputs: {src}]"
        return AgentSignal(
            self.name, sig, min(1.0, abs(score)), reason,
            {"netflow_z": z, "stablecoin_change": sc, "score": score, "input_provenance": src},
            market_state.timestamp, True, f"on-chain netflow z {z:+.2f}",
        )
