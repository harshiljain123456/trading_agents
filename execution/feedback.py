"""Controlled feedback loop: periodic, leakage-free agent reweighting.

Every ``interval`` bars (default 60) during a *training* run, the controller
scores agents on outcomes that have matured by the current bar (prediction
bar + horizon <= current bar) and within the training window, asks the
evaluator for bounded weight suggestions and applies them to the boss's
learned weights. When ``frozen`` is True (validation/test), nothing changes.
Every change is logged with its evidence.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import numpy as np

from agents.boss_orchestrator import BossOrchestrator
from core.models import WeightUpdate
from execution.evaluator import Evaluator, SignalRecord

logger = logging.getLogger(__name__)


class FeedbackController:
    """Applies evaluator suggestions to the boss on a fixed bar interval."""

    def __init__(self, evaluator: Evaluator, boss: BossOrchestrator, interval: int, window_start: int, frozen: bool = False) -> None:
        """Create the controller for a run that starts at bar ``window_start``."""
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.evaluator = evaluator
        self.boss = boss
        self.interval = interval
        self.window_start = window_start
        self.frozen = frozen
        self.updates: list[WeightUpdate] = []

    def maybe_update(self, bar_index: int, timestamp: datetime, history: Sequence[SignalRecord], prices: np.ndarray) -> bool:
        """Update weights if this bar is a feedback point. Returns True if evaluated.

        Only records with ``window_start <= bar_index`` and outcomes matured by
        ``bar_index`` (whose close is already known) are used.
        """
        if self.frozen:
            return False
        elapsed = bar_index - self.window_start
        if elapsed <= 0 or elapsed % self.interval != 0:
            return False
        scores = self.evaluator.score_agents(
            history, prices, as_of_index=bar_index, current_weights=dict(self.boss.learned_weights),
            min_index=self.window_start,
        )
        suggested = self.evaluator.suggest_weights(scores)
        for name, new in suggested.items():
            old = self.boss.learned_weights.get(name)
            if old is None:
                continue
            s = scores[name]
            if s.note:
                evidence = s.note + " -> weight preserved"
            else:
                evidence = (
                    f"matured={s.matured_count} hit_rate={s.hit_rate if s.hit_rate is None else round(s.hit_rate, 3)} "
                    f"incremental={s.incremental_contribution if s.incremental_contribution is None else round(s.incremental_contribution, 3)}bp "
                    f"t={s.incremental_tstat if s.incremental_tstat is None else round(s.incremental_tstat, 2)}"
                )
            if abs(new - old) > 1e-12:
                self.updates.append(WeightUpdate(bar_index, timestamp, name, old, new, evidence))
                logger.info("feedback @%s %s: weight %.3f -> %.3f (%s)", timestamp, name, old, new, evidence)
            else:
                logger.debug("feedback @%s %s: weight unchanged %.3f (%s)", timestamp, name, old, evidence)
        self.boss.set_learned_weights(suggested)
        return True
