"""Explanation layer for boss decisions.

The ``Reasoner`` only writes text. It receives the already-final numerical
decision and cannot change the action, the scores or any risk control. An
external LLM reasoner can implement the same protocol; the default is a
deterministic template reasoner that needs no network or API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.models import AgentContribution, AgentSignal, SignalType


@dataclass(frozen=True)
class ReasoningContext:
    """Inputs the reasoner may describe (read-only)."""

    action: SignalType
    raw_action: SignalType
    directional_score: float
    strength: float
    agreement: float
    coverage: float
    data_coverage: float
    regime: str
    risk_score: float | None
    risk_veto_threshold: float
    forced_hold_rule: str | None
    contributions: list[AgentContribution]
    signals: dict[str, AgentSignal]


class Reasoner(Protocol):
    """Anything that turns a decision context into an analyst note."""

    def explain(self, ctx: ReasoningContext) -> str:
        """Return a human-readable explanation."""
        ...


def _strength_word(x: float) -> str:
    return "strong" if x >= 0.75 else "moderate" if x >= 0.5 else "weak"


class TemplateReasoner:
    """Deterministic analyst-style note built from actual feature values."""

    def explain(self, ctx: ReasoningContext) -> str:
        """Compose the note: top-3 contributors, risk status, ensemble metrics, overrides."""
        directional = [c for c in ctx.contributions if c.eligible and c.signal != SignalType.HOLD]
        directional.sort(key=lambda c: abs(c.contribution), reverse=True)
        parts: list[str] = []
        for c in directional[:3]:
            sig = ctx.signals.get(c.agent_name)
            summary = sig.summary if sig and sig.summary else sig.reason if sig else c.reason
            parts.append(f"{summary} ({c.agent_name} {c.signal.value}, contribution {c.contribution:+.3f})")
        if not parts:
            parts.append("no agent produced a directional vote")
        if ctx.risk_score is not None:
            rel = "above" if ctx.risk_score > ctx.risk_veto_threshold else "below"
            parts.append(f"risk_score {ctx.risk_score:.2f} is {rel} the veto threshold {ctx.risk_veto_threshold:.2f}")
        else:
            parts.append("risk_score unavailable")
        unavailable = [c.agent_name for c in ctx.contributions if not c.available]
        if unavailable:
            parts.append(f"excluded for missing data: {', '.join(unavailable)}")
        direction = "positive" if ctx.directional_score > 0 else "negative" if ctx.directional_score < 0 else "flat"
        parts.append(
            f"overall weighted direction is {direction} (S={ctx.directional_score:+.3f}) with "
            f"{_strength_word(ctx.agreement)} agreement {ctx.agreement:.2f}, coverage {ctx.coverage:.2f}, "
            f"ensemble strength {ctx.strength:.2f} (uncalibrated) in regime '{ctx.regime}'"
        )
        if ctx.forced_hold_rule:
            if ctx.raw_action != SignalType.HOLD:
                parts.append(f"the {ctx.raw_action.value} vote was overridden to HOLD by rule '{ctx.forced_hold_rule}'")
            else:
                parts.append(f"HOLD because of rule '{ctx.forced_hold_rule}'")
        parts.append(f"decision: {ctx.action.value}")
        text = "; ".join(parts) + "."
        return text[0].upper() + text[1:]
