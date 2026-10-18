"""Specialist agents, the boss orchestrator and agent factories."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from agents.liquidity_engine import LiquidityEngineAgent
from agents.market_intel import MarketIntelAgent
from agents.market_mood import MarketMoodAgent
from agents.on_chain import OnChainAgent
from agents.regime_detector import RegimeDetectorAgent
from agents.risk_engine import RiskEngineAgent
from agents.sentiment_engine import SentimentEngineAgent
from core.config import FrameworkConfig


def build_agents(config: FrameworkConfig, exclude: tuple[str, ...] = ()) -> list[BaseAgent]:
    """Instantiate every enabled voting agent except those in ``exclude``.

    The optional agent is created only when ``features.enable_optional_agent``
    is true, so a disabled agent can never affect votes or denominators.
    """
    a = config.agents
    w = a.base_weights
    agents: list[BaseAgent] = [
        MarketIntelAgent(w["market_intel"], a.market_intel),
        RiskEngineAgent(w["risk_engine"], a.risk_engine, config.risk.risk_veto_threshold),
        LiquidityEngineAgent(w["liquidity_engine"], a.liquidity_engine),
        (
            MarketMoodAgent(w["market_mood"], a.market_mood)
            if config.features.sentiment_slot == "market_mood"
            else SentimentEngineAgent(w["sentiment_engine"], a.sentiment_engine)
        ),
        RegimeDetectorAgent(w["regime_detector"], a.regime_detector),
    ]
    if config.features.enable_optional_agent and config.features.optional_agent_type == "on_chain":
        agents.append(OnChainAgent(w["on_chain"], a.on_chain))
    return [ag for ag in agents if ag.name not in exclude]


__all__ = [
    "BaseAgent",
    "LiquidityEngineAgent",
    "MarketIntelAgent",
    "MarketMoodAgent",
    "OnChainAgent",
    "RegimeDetectorAgent",
    "RiskEngineAgent",
    "SentimentEngineAgent",
    "build_agents",
]
