"""Base class for voting agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.models import AgentSignal, FeatureStatus, FeatureValue, MarketState, SignalType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract voting agent.

    Subclasses declare ``essential_features``; if any is unavailable or not
    finite, :meth:`analyze` must return an *unavailable* HOLD (see
    :meth:`unavailable`) so the boss excludes the agent from the vote.
    """

    name: str = "base"
    essential_features: tuple[str, ...] = ()

    def __init__(self, weight: float = 1.0) -> None:
        """Create the agent with its base voting weight."""
        if weight <= 0:
            raise ValueError(f"{self.name}: weight must be positive")
        self.weight = float(weight)

    @abstractmethod
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Return this agent's signal for ``market_state``."""

    def check_inputs(self, market_state: MarketState) -> list[str]:
        """Names of essential features that are unavailable/invalid at this bar."""
        return market_state.missing(list(self.essential_features))

    def provenance(self, market_state: MarketState, names: tuple[str, ...]) -> str:
        """Comma-separated provenance labels of raw inputs (e.g. 'synthetic')."""
        labels = sorted({market_state.status(n).value for n in names})
        return ",".join(labels)

    def unavailable(self, market_state: MarketState, missing: list[str], extra: str = "") -> AgentSignal:
        """Build an explicit unavailable/HOLD signal naming the missing inputs."""
        warm = [m for m in missing if market_state.status(m) != FeatureStatus.UNAVAILABLE]
        absent = [m for m in missing if market_state.status(m) == FeatureStatus.UNAVAILABLE]
        bits = []
        if absent:
            bits.append(f"feeds not supplied: {', '.join(absent)}")
        if warm:
            bits.append(f"insufficient history / non-finite values for: {', '.join(warm)}")
        reason = f"{self.name}: HOLD (unavailable) - " + "; ".join(bits) + (f". {extra}" if extra else "")
        logger.debug(reason)
        feats: dict[str, FeatureValue] = {"missing": ",".join(missing)}
        return AgentSignal(
            agent_name=self.name,
            signal=SignalType.HOLD,
            confidence=0.0,
            reason=reason,
            features=feats,
            timestamp=market_state.timestamp,
            available=False,
            summary=f"{self.name} unavailable",
        )
