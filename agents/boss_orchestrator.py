"""Boss Orchestrator: weighted ensemble vote with risk overrides and explanations.

Directional score
-----------------
For eligible agents i (enabled *and* ``available``)::

    S = sum_i w_i * x_i / sum_i w_i,     x_i = c_i * d_i  (or its Kalman-smoothed value)

The denominator is the total effective weight of **eligible** agents,
including those voting HOLD, so lack of conviction shrinks |S|. Unavailable
agents are excluded from both numerator and denominator (they cannot confirm
anything); their absence is reported through ``data_coverage``.

Effective weight: ``w_i = learned_i * regime_factor_i * family_factor_i`` where
``learned_i`` starts at the base weight and is changed only by the feedback
loop (clipped to [min_weight, max_weight]); ``regime_factor_i`` comes from the
regime table (1.0 if disabled or regime unknown); ``family_factor_i`` is
1/(enabled members in the agent's signal family) when family normalization
is on, else 1.

Diagnostics
-----------
* agreement     = weight of directional votes aligned with sign(S) / weight of all directional votes
* coverage      = weight of directional votes / weight of eligible agents
* data_coverage = weight of eligible agents / weight of enabled agents
* strength      = agreement * (weighted mean confidence of aligned votes) * sqrt(coverage)

``strength`` is an *uncalibrated ensemble strength* in [0, 1], not a
probability of profit.

Rules producing HOLD (all triggered rules are recorded, in this order):
kill_switch, risk_unavailable, risk_veto, data_coverage,
directional_score_floor, min_confidence.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from core.config import FrameworkConfig
from core.kalman import Kalman1D
from core.models import AgentContribution, AgentSignal, BossDecision, Regime, SignalType
from core.reasoner import Reasoner, ReasoningContext, TemplateReasoner

logger = logging.getLogger(__name__)


class BossOrchestrator:
    """Aggregates specialist signals into a :class:`BossDecision`."""

    def __init__(self, config: FrameworkConfig, agent_names: list[str], reasoner: Reasoner | None = None) -> None:
        """Create the boss for the given enabled agents."""
        self.config = config
        self.agent_names = list(agent_names)
        self.base_weights = {n: config.agents.base_weights[n] for n in self.agent_names}
        self.learned_weights = dict(self.base_weights)
        self.reasoner: Reasoner = reasoner or TemplateReasoner()
        self.min_confidence = config.strategy.min_confidence
        self.score_floor = config.strategy.directional_score_floor
        fams = config.agents.families
        self.family_factor = {}
        for n in self.agent_names:
            fam = fams.get(n, n)
            members = sum(1 for m in self.agent_names if fams.get(m, m) == fam)
            self.family_factor[n] = 1.0 / members if config.features.enable_family_normalization else 1.0
        self._kalman: dict[str, Kalman1D] = {}
        self.reset()

    def reset(self) -> None:
        """Reset Kalman smoothing state (call at the start of every run)."""
        k = self.config.kalman
        self._kalman = {
            n: Kalman1D(k.process_noise, k.observation_noise, k.initial_variance) for n in self.agent_names
        }

    def set_learned_weights(self, weights: dict[str, float]) -> None:
        """Replace learned weights (clipped to configured bounds)."""
        ev = self.config.evaluation
        for n, w in weights.items():
            if n in self.learned_weights:
                self.learned_weights[n] = min(ev.max_weight, max(ev.min_weight, float(w)))

    def regime_factor(self, agent: str, regime: Regime) -> float:
        """Regime multiplier for ``agent`` (1.0 when disabled/unknown/unspecified)."""
        if not self.config.features.enable_regime_weights or regime == Regime.UNKNOWN:
            return 1.0
        return self.config.regime_weights.get(regime.value, {}).get(agent, 1.0)

    def effective_weights(self, regime: Regime) -> dict[str, float]:
        """Effective weight for every enabled agent in ``regime``."""
        return {
            n: self.learned_weights[n] * self.regime_factor(n, regime) * self.family_factor[n]
            for n in self.agent_names
        }

    def decide(
        self,
        signals: list[AgentSignal],
        kill_switch_active: bool = False,
        timestamp: datetime | None = None,
    ) -> BossDecision:
        """Aggregate ``signals`` (one per enabled agent) into a decision."""
        by_name = {s.agent_name: s for s in signals}
        unknown = set(by_name) - set(self.agent_names)
        if unknown:
            raise ValueError(f"signals from agents that are not enabled: {sorted(unknown)}")
        ts = timestamp or (signals[0].timestamp if signals else None)
        if ts is None:
            raise ValueError("timestamp required when no signals are given")

        regime = Regime.UNKNOWN
        rd = by_name.get("regime_detector")
        if rd is not None and rd.available:
            regime = Regime(str(rd.features.get("regime", "unknown")))
        risk = by_name.get("risk_engine")
        risk_score = float(risk.features["risk_score"]) if risk is not None and risk.available else None  # type: ignore[arg-type]

        weights = self.effective_weights(regime)
        inputs: dict[str, float] = {}
        for n in self.agent_names:
            s = by_name.get(n)
            raw = s.signed_confidence if (s is not None and s.available) else None
            if self.config.features.enable_kalman:
                sm = self._kalman[n].update(raw)
                inputs[n] = sm if (raw is not None and sm is not None) else 0.0
            else:
                inputs[n] = raw if raw is not None else 0.0

        eligible = [n for n in self.agent_names if n in by_name and by_name[n].available]
        w_enabled = sum(weights.values())
        w_elig = sum(weights[n] for n in eligible)
        S = sum(weights[n] * inputs[n] for n in eligible) / w_elig if w_elig > 0 else 0.0
        S = max(-1.0, min(1.0, S))
        directional = [n for n in eligible if by_name[n].signal != SignalType.HOLD]
        w_dir = sum(weights[n] for n in directional)
        dom = 1 if S > 0 else -1 if S < 0 else 0
        aligned = [n for n in directional if by_name[n].signal.direction == dom] if dom else []
        w_al = sum(weights[n] for n in aligned)
        agreement = w_al / w_dir if w_dir > 0 else 0.0
        coverage = w_dir / w_elig if w_elig > 0 else 0.0
        data_cov = w_elig / w_enabled if w_enabled > 0 else 0.0
        mean_conf = sum(weights[n] * by_name[n].confidence for n in aligned) / w_al if w_al > 0 else 0.0
        strength = max(0.0, min(1.0, agreement * mean_conf * math.sqrt(coverage)))

        candidate = SignalType.BUY if dom > 0 else SignalType.SELL if dom < 0 else SignalType.HOLD
        rules: list[str] = []
        floors_ok = abs(S) >= self.score_floor and strength >= self.min_confidence
        if abs(S) < self.score_floor:
            rules.append("directional_score_floor")
        if strength < self.min_confidence:
            rules.append("min_confidence")
        raw_action = candidate if floors_ok else SignalType.HOLD
        overrides: list[str] = []
        if kill_switch_active:
            overrides.append("kill_switch")
        if "risk_engine" in self.agent_names and risk_score is None:
            overrides.append("risk_unavailable")
        risk_veto = risk_score is not None and risk_score > self.config.risk.risk_veto_threshold
        if risk_veto:
            overrides.append("risk_veto")
        if data_cov < self.config.strategy.min_data_coverage:
            overrides.append("data_coverage")
        all_rules = overrides + rules
        action = raw_action if not overrides else SignalType.HOLD
        forced = ";".join(all_rules) if (all_rules and action == SignalType.HOLD) else None
        if action == SignalType.HOLD and forced:
            logger.debug("%s forced HOLD by %s (raw %s, S=%+.3f, strength=%.2f)", ts, forced, raw_action.value, S, strength)

        breakdown = []
        for n in self.agent_names:
            s = by_name.get(n)
            elig = n in eligible
            breakdown.append(
                AgentContribution(
                    agent_name=n,
                    signal=s.signal if s else SignalType.HOLD,
                    confidence=s.confidence if s else 0.0,
                    signed_input=inputs[n] if elig else 0.0,
                    base_weight=self.base_weights[n],
                    learned_weight=self.learned_weights[n],
                    regime_factor=self.regime_factor(n, regime),
                    family_factor=self.family_factor[n],
                    effective_weight=weights[n],
                    contribution=(weights[n] * inputs[n] / w_elig) if (elig and w_elig > 0) else 0.0,
                    eligible=elig,
                    available=bool(s and s.available),
                    reason=s.reason if s else f"{n}: no signal supplied",
                )
            )
        ctx = ReasoningContext(
            action=action, raw_action=raw_action, directional_score=S, strength=strength, agreement=agreement,
            coverage=coverage, data_coverage=data_cov, regime=regime.value, risk_score=risk_score,
            risk_veto_threshold=self.config.risk.risk_veto_threshold, forced_hold_rule=forced,
            contributions=breakdown, signals=by_name,
        )
        reason = self.reasoner.explain(ctx)
        return BossDecision(
            action=action, raw_action=raw_action, confidence=strength, directional_score=S,
            agreement=agreement, coverage=coverage, data_coverage=data_cov, reason=reason,
            breakdown=breakdown, regime=regime, risk_score=risk_score, risk_veto=risk_veto,
            kill_switch_active=kill_switch_active, forced_hold_rule=forced, timestamp=ts,
        )
