"""Portfolio-level risk: drawdown/Sharpe kill switch.

This is deliberately separate from the RiskEngineAgent. The agent scores
*market* risk (volatility, ATR, market drawdown) from prices; the kill switch
watches the *strategy's own* net-of-cost equity curve.

Trigger: once ``min_history`` net returns exist, the switch activates when
  * drawdown from the equity peak exceeds ``max_drawdown``; or
  * the annualized rolling Sharpe over ``sharpe_window`` returns falls below
    ``sharpe_threshold`` (only evaluated when the window is full and the
    return stdev is non-zero).

While active, new entries are blocked. Existing positions follow
``position_policy`` (``flatten`` closes them, ``hold`` keeps them).

Recovery: ``cooldown`` deactivates after ``cooldown_bars`` and resets the
peak and the return history, so the switch must re-accumulate
``min_history`` returns before it can trigger again; ``manual`` stays active
until :meth:`reset` is called.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchEvent:
    """Audit record of a trigger or recovery."""

    timestamp: datetime
    bar_index: int
    kind: str
    detail: str


@dataclass
class KillSwitch:
    """Drawdown- and Sharpe-aware portfolio kill switch."""

    max_drawdown: float
    sharpe_threshold: float
    sharpe_window: int
    min_history: int
    bars_per_year: int
    recovery: str = "cooldown"
    cooldown_bars: int = 60
    position_policy: str = "flatten"
    active: bool = False
    events: list[KillSwitchEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.recovery not in ("cooldown", "manual"):
            raise ValueError("recovery must be 'cooldown' or 'manual'")
        if self.position_policy not in ("flatten", "hold"):
            raise ValueError("position_policy must be 'flatten' or 'hold'")
        self._returns: deque[float] = deque(maxlen=max(self.sharpe_window, self.min_history))
        self._n = 0
        self._peak: float | None = None
        self._last_equity: float | None = None
        self._active_since: int | None = None
        self.last_drawdown = 0.0
        self.last_sharpe: float | None = None

    def reset(self) -> None:
        """Manually clear the switch and its history."""
        self.active = False
        self._returns.clear()
        self._n = 0
        self._peak = None
        self._last_equity = None
        self._active_since = None

    def update(self, equity: float, timestamp: datetime, bar_index: int) -> bool:
        """Feed the latest marked-to-market equity; return the active state."""
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError(f"equity must be positive and finite, got {equity}")
        if self._last_equity is not None:
            self._returns.append(equity / self._last_equity - 1.0)
            self._n += 1
        self._last_equity = equity
        self._peak = equity if self._peak is None else max(self._peak, equity)
        self.last_drawdown = 1.0 - equity / self._peak
        self.last_sharpe = self._rolling_sharpe()

        if self.active:
            if self.recovery == "cooldown" and bar_index - (self._active_since or bar_index) >= self.cooldown_bars:
                self.active = False
                self._returns.clear()
                self._n = 0
                self._peak = equity
                self.last_drawdown = 0.0
                self._active_since = None
                self.events.append(KillSwitchEvent(timestamp, bar_index, "recovery", "cooldown elapsed; history reset"))
                logger.warning("kill switch RECOVERED at %s (cooldown elapsed)", timestamp)
            return self.active

        if self._n < self.min_history:
            return False
        reasons = []
        if self.last_drawdown > self.max_drawdown:
            reasons.append(f"drawdown {self.last_drawdown:.1%} > cap {self.max_drawdown:.1%}")
        if self.last_sharpe is not None and self.last_sharpe < self.sharpe_threshold:
            reasons.append(f"rolling Sharpe {self.last_sharpe:.2f} < {self.sharpe_threshold:.2f}")
        if reasons:
            self.active = True
            self._active_since = bar_index
            detail = "; ".join(reasons)
            self.events.append(KillSwitchEvent(timestamp, bar_index, "trigger", detail))
            logger.warning("kill switch TRIGGERED at %s: %s", timestamp, detail)
        return self.active

    def _rolling_sharpe(self) -> float | None:
        """Annualized Sharpe of the last ``sharpe_window`` returns, or None."""
        if len(self._returns) < self.sharpe_window:
            return None
        r = np.fromiter(list(self._returns)[-self.sharpe_window :], dtype=float)
        sd = r.std(ddof=1)
        if sd <= 1e-15:
            return None
        return float(r.mean() / sd * math.sqrt(self.bars_per_year))
