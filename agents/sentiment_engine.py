"""SentimentEngineAgent: social + news momentum with contrarian funding."""

from __future__ import annotations

import math

from agents.base_agent import BaseAgent
from core.config import SentimentParams
from core.models import AgentSignal, MarketState, SignalType
from indicators.sentiment import composite_sentiment, funding_contrarian, normalize_score


class SentimentEngineAgent(BaseAgent):
    """Composite sentiment in [-1, 1].

    * social, news: normalized to [-1, 1] from their configured raw ranges
      (momentum interpretation: positive = bullish).
    * funding: contrarian term (crowded longs -> bearish).
    * composite: weighted mean of *available* components (missing feeds are
      excluded, not treated as neutral, unless ``treat_missing_as_neutral``).
    At least ``min_components`` components must be available, else the agent
    is unavailable. BUY/SELL beyond +-signal_threshold.
    confidence = |composite| * weight_coverage (penalizes partial data).
    """

    name = "sentiment_engine"
    essential_features: tuple[str, ...] = ()

    def __init__(self, weight: float, params: SentimentParams) -> None:
        """Create the agent with weight and normalization parameters."""
        super().__init__(weight)
        self.params = params

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Score sentiment at ``market_state``."""
        p = self.params
        names = ("social_sentiment", "news_sentiment", "funding_rate")
        valid = {n: market_state.is_valid(n) for n in names}
        social = normalize_score(market_state.value("social_sentiment"), p.social_min, p.social_max) if valid["social_sentiment"] else float("nan")
        news = normalize_score(market_state.value("news_sentiment"), p.news_min, p.news_max) if valid["news_sentiment"] else float("nan")
        fund_raw = market_state.value("funding_rate") if valid["funding_rate"] else None
        fund = funding_contrarian(fund_raw, p.funding_threshold, p.funding_saturation) if valid["funding_rate"] else float("nan")
        if p.treat_missing_as_neutral:
            social = 0.0 if math.isnan(social) else social
            news = 0.0 if math.isnan(news) else news
            fund = 0.0 if math.isnan(fund) else fund
        comp = composite_sentiment(social, news, fund, p.w_social, p.w_news, p.w_funding)
        missing = [n for n in names if not valid[n]]
        if comp.available_components < p.min_components:
            return self.unavailable(
                market_state, missing,
                f"Only {comp.available_components} of 3 sentiment components available "
                f"(min {p.min_components}); missing feeds are not treated as neutral.",
            )
        s = comp.score
        if s > p.signal_threshold:
            sig = SignalType.BUY
        elif s < -p.signal_threshold:
            sig = SignalType.SELL
        else:
            sig = SignalType.HOLD

        def fmt(x: float) -> str:
            return "n/a" if math.isnan(x) else f"{x:+.2f}"

        fund_txt = "n/a" if fund_raw is None else f"{fund_raw * 1e4:+.2f}bp/8h"
        fund_note = ""
        if not math.isnan(fund) and fund != 0.0:
            fund_note = " (extreme funding -> contrarian " + ("bearish" if fund < 0 else "bullish") + ")"
        detail = (
            f"social {fmt(social)}, news {fmt(news)}, funding {fund_txt} -> contrarian term {fmt(fund)}{fund_note}; "
            f"composite {s:+.3f} using {comp.available_components}/3 components"
        )
        if missing:
            detail += f" (missing: {', '.join(missing)})"
        reason = (f"{sig.value}: " if sig != SignalType.HOLD else f"HOLD: composite within +/-{p.signal_threshold:.2f}; ") + detail
        mood = "bullish" if s > 0 else "bearish" if s < 0 else "neutral"
        return AgentSignal(
            agent_name=self.name,
            signal=sig,
            confidence=min(1.0, abs(s) * comp.weight_coverage),
            reason=reason,
            features={
                "social_norm": None if math.isnan(social) else social,
                "news_norm": None if math.isnan(news) else news,
                "funding_rate": fund_raw,
                "funding_term": None if math.isnan(fund) else fund,
                "composite": s,
                "components_available": comp.available_components,
                "weight_coverage": comp.weight_coverage,
                "input_provenance": self.provenance(market_state, names),
            },
            timestamp=market_state.timestamp,
            summary=f"{mood} sentiment composite {s:+.2f} (funding {fund_txt})",
        )
