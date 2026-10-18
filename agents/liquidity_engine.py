"""LiquidityEngineAgent: liquidity sweeps confirmed by order flow and depth."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import LiquidityParams
from core.models import AgentSignal, MarketState, SignalType


class LiquidityEngineAgent(BaseAgent):
    """Trades confirmed sweeps of prior rolling lows/highs.

    Rules (all inputs observable at the bar close):
      * No sweep, or both sides swept in one bar -> HOLD.
      * Bullish sweep: CVD confirms if short-window CVD change > cvd_threshold;
        depth confirms if depth imbalance > depth_threshold. Bearish mirrors.
      * CVD *opposing* the sweep beyond the threshold = divergence -> HOLD.
      * Need at least ``min_confirmations`` confirmations, else HOLD
        (unconfirmed sweep).
    confidence = 0.4 + 0.2*[cvd confirms] + 0.2*[depth confirms]
                 + 0.2*[absorption_ratio >= absorption_threshold]
    Unavailable CVD/depth/absorption simply cannot confirm (never assumed).
    """

    name = "liquidity_engine"
    essential_features = ("bullish_sweep", "bearish_sweep", "prior_low", "prior_high")

    def __init__(self, weight: float, params: LiquidityParams) -> None:
        """Create the agent with weight and confirmation parameters."""
        super().__init__(weight)
        self.params = params

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Detect and qualify a sweep at ``market_state``."""
        missing = self.check_inputs(market_state)
        if missing:
            return self.unavailable(market_state, missing, "Sweep levels need a full lookback window.")
        p = self.params
        bull = market_state.get("bullish_sweep") >= 0.5
        bear = market_state.get("bearish_sweep") >= 0.5
        lo = market_state.get("prior_low")
        hi = market_state.get("prior_high")
        cvd_s = market_state.value("cvd_norm_short") if market_state.is_valid("cvd_norm_short") else None
        depth = market_state.value("depth_imbalance") if market_state.is_valid("depth_imbalance") else None
        ar = market_state.value("absorption_ratio") if market_state.is_valid("absorption_ratio") else None
        feats = {
            "bullish_sweep": bull, "bearish_sweep": bear, "prior_low": lo, "prior_high": hi,
            "cvd_norm_short": cvd_s, "depth_imbalance": depth, "absorption_ratio": ar,
            "cvd_is_proxy": market_state.value("cvd_is_proxy") == 1.0,
        }
        proxy_txt = " (tick-rule proxy)" if feats["cvd_is_proxy"] else ""
        ctx = (
            f"CVD(3){proxy_txt} {'n/a' if cvd_s is None else f'{cvd_s:+.2f}'}, depth imbalance "
            f"{'n/a' if depth is None else f'{depth:+.2f}'}, absorption ratio {'n/a' if ar is None else f'{ar:.2f}'}"
        )

        def hold(reason: str, summary: str) -> AgentSignal:
            return AgentSignal(self.name, SignalType.HOLD, 0.0, reason, feats, market_state.timestamp, True, summary)

        if bull and bear:
            return hold(
                f"HOLD: bar swept both the prior low {lo:,.2f} and prior high {hi:,.2f}; direction ambiguous ({ctx})",
                "two-sided sweep (ambiguous)",
            )
        if not bull and not bear:
            return hold(
                f"HOLD: no sweep - price stayed within the prior {lo:,.2f}-{hi:,.2f} range or failed to reverse; {ctx}",
                "no liquidity sweep",
            )
        d = 1 if bull else -1
        level = lo if bull else hi
        kind = "below" if bull else "above"
        cvd_conf = cvd_s is not None and d * cvd_s > p.cvd_threshold
        cvd_div = cvd_s is not None and d * cvd_s < -p.cvd_threshold
        depth_conf = depth is not None and d * depth > p.depth_threshold
        absorb = ar is not None and ar >= p.absorption_threshold
        n_conf = int(cvd_conf) + int(depth_conf)
        flow_word = ("absorption of selling" if bull else "distribution into buying") if absorb else "no clear absorption"
        feats.update({"cvd_confirms": cvd_conf, "cvd_divergence": cvd_div, "depth_confirms": depth_conf,
                      "absorption_supportive": absorb, "confirmations": n_conf, "swept_level": level})
        if cvd_div:
            return hold(
                f"HOLD: {'bullish' if bull else 'bearish'} sweep {kind} {level:,.2f} but CVD diverges "
                f"({cvd_s:+.2f} against the reversal); {ctx}",
                f"sweep {kind} {level:,.2f} with CVD divergence",
            )
        if n_conf < p.min_confirmations:
            return hold(
                f"HOLD: unconfirmed {'bullish' if bull else 'bearish'} sweep {kind} {level:,.2f} "
                f"({n_conf}/{p.min_confirmations} confirmations; unavailable inputs cannot confirm); {ctx}",
                f"unconfirmed sweep {kind} {level:,.2f}",
            )
        conf = 0.4 + 0.2 * cvd_conf + 0.2 * depth_conf + 0.2 * absorb
        sig = SignalType.BUY if bull else SignalType.SELL
        reason = (
            f"{sig.value}: {'bullish' if bull else 'bearish'} sweep {kind} prior level {level:,.2f} reversed by the "
            f"close; confirmations: CVD {'yes' if cvd_conf else 'no'}, depth {'yes' if depth_conf else 'no'}; "
            f"{flow_word}; {ctx}. A sweep is a price pattern, not proof of institutional activity"
        )
        summary = f"a sweep {kind} {level:,.2f} with {flow_word} ({n_conf} flow confirmations)"
        return AgentSignal(self.name, sig, min(1.0, conf), reason, feats, market_state.timestamp, True, summary)
