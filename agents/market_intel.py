"""MarketIntelAgent: institutional-style momentum from order flow and price/volume."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import MarketIntelParams
from core.models import AgentSignal, MarketState, SignalType


class MarketIntelAgent(BaseAgent):
    """Combines OFI, normalized CVD change and volume-confirmed momentum.

    score = (w_ofi*OFI + w_cvd*CVD_norm + w_mom*InstMom) / (w_ofi + w_cvd + w_mom)

    Each input lies in [-1, 1], so ``score`` does too. BUY if score >
    ``signal_threshold``, SELL if score < -threshold, else HOLD.
    confidence = |score| (the agent's evidence strength, not a probability).

    The rolling price/volume correlation is reported for context only.
    Note: with synthetic data CVD_norm is close to a longer-window OFI, so the
    two inputs are correlated; their weights should be read jointly.

    Proxy mode (``allow_proxy_flow``): when genuine aggressor volume is not
    available (OFI unavailable, e.g. NSE OHLCV-only data), the agent may run
    on the tick-rule CVD proxy plus momentum::

        score = (w_cvd*CVD_proxy + w_mom*InstMom) / (w_cvd + w_mom)
        confidence = |score| * proxy_confidence_scale

    and states explicitly that no real order-flow data was used.
    """

    name = "market_intel"
    essential_features = ("ofi", "cvd_norm", "inst_momentum")

    def __init__(self, weight: float, params: MarketIntelParams) -> None:
        """Create the agent with weight and scoring parameters."""
        super().__init__(weight)
        self.params = params

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Score order-flow momentum at ``market_state``."""
        missing = self.check_inputs(market_state)
        p = self.params
        if missing == ["ofi"] and p.allow_proxy_flow:
            return self._analyze_proxy(market_state)
        if missing:
            return self.unavailable(market_state, missing, "Order-flow momentum cannot be assessed.")
        ofi = market_state.get("ofi")
        cvdn = market_state.get("cvd_norm")
        mom = market_state.get("inst_momentum")
        pv = market_state.value("pv_corr")
        wsum = p.w_ofi + p.w_cvd + p.w_momentum
        score = (p.w_ofi * ofi + p.w_cvd * cvdn + p.w_momentum * mom) / wsum
        score = max(-1.0, min(1.0, score))
        conf = abs(score)
        proxy = market_state.value("cvd_is_proxy") == 1.0
        src = self.provenance(market_state, ("bid_volume", "ask_volume", "trade_side"))
        pressure = "buying" if ofi > 0 else "selling" if ofi < 0 else "balanced"
        mom_word = "positive" if mom > 0 else "negative" if mom < 0 else "flat"
        if score > p.signal_threshold:
            signal = SignalType.BUY
        elif score < -p.signal_threshold:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        pv_txt = f", price/volume corr {pv:+.2f}" if pv is not None and pv == pv else ""
        summary = f"OFI {ofi:+.2f} and {mom_word} institutional-style momentum ({mom:+.2f}) indicate {pressure} pressure"
        if signal == SignalType.HOLD:
            reason = (
                f"HOLD: composite flow score {score:+.3f} within +/-{p.signal_threshold:.2f}; "
                f"OFI {ofi:+.3f} ({pressure} pressure), CVD change {cvdn:+.3f}{' (tick-rule proxy)' if proxy else ''}, "
                f"institutional-style momentum {mom:+.3f}{pv_txt} do not agree strongly enough"
            )
        else:
            reason = (
                f"{signal.value}: {pressure} pressure dominates - OFI {ofi:+.3f}, CVD change "
                f"{cvdn:+.3f}{' (tick-rule proxy)' if proxy else ''}, institutional-style momentum {mom:+.3f}"
                f"{pv_txt}; composite {score:+.3f} beyond +/-{p.signal_threshold:.2f}"
            )
        reason += f" [inputs: {src}]"
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=conf,
            reason=reason,
            features={
                "ofi": ofi, "cvd_norm": cvdn, "inst_momentum": mom, "pv_corr": pv,
                "score": score, "cvd_is_proxy": proxy, "input_provenance": src,
            },
            timestamp=market_state.timestamp,
            summary=summary,
        )

    def _analyze_proxy(self, market_state: MarketState) -> AgentSignal:
        """Score momentum from the tick-rule CVD proxy when OFI is unavailable."""
        p = self.params
        cvdn = market_state.get("cvd_norm")
        mom = market_state.get("inst_momentum")
        proxy = market_state.value("cvd_is_proxy") == 1.0
        score = max(-1.0, min(1.0, (p.w_cvd * cvdn + p.w_momentum * mom) / (p.w_cvd + p.w_momentum)))
        conf = abs(score) * p.proxy_confidence_scale
        if score > p.signal_threshold:
            signal = SignalType.BUY
        elif score < -p.signal_threshold:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        bias = "up-tick" if cvdn > 0 else "down-tick" if cvdn < 0 else "balanced"
        src = "tick-rule CVD proxy (price direction, NOT real aggressor data)" if proxy else "aggressor-side CVD"
        lead = f"{signal.value}: " if signal != SignalType.HOLD else f"HOLD: composite within +/-{p.signal_threshold:.2f}; "
        reason = (
            f"{lead}OFI unavailable (no buyer/seller-initiated volume in this feed); using {src} {cvdn:+.3f} "
            f"({bias} volume dominates) and institutional-style momentum {mom:+.3f}; composite {score:+.3f}, "
            f"confidence scaled by {p.proxy_confidence_scale:.2f} for proxy data"
        )
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=min(1.0, conf),
            reason=reason,
            features={"ofi": None, "cvd_norm": cvdn, "inst_momentum": mom, "score": score,
                      "cvd_is_proxy": proxy, "proxy_mode": True},
            timestamp=market_state.timestamp,
            summary=f"proxy flow {cvdn:+.2f} and momentum {mom:+.2f} (no real order-flow data)",
        )
