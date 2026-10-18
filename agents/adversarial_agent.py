"""Adversarial shadow agent: trades against the primary consensus, never votes.

It keeps two hypothetical books with *identical* timing and costs to the
primary strategy (decision at close t, fill at open t+1, same fee/slippage):

* ``consensus`` book: follows the ensemble's raw vote (BUY -> long,
  SELL -> short, HOLD -> keep) at a fixed exposure.
* ``counter`` book: takes the opposite side of the same vote.

Both ignore risk controls so the comparison isolates the *signal*. If the
counter book beats the consensus book, the ensemble's directional view is
worse than its own negation over that sample.

Disagreement observations (directional votes) are counted separately from
matured outcomes (votes whose ``horizon`` bars have elapsed). The adversary
never feeds back into weights, so its results are an independent diagnostic
rather than a re-use of the outcomes that tuned the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from execution.executor import Portfolio


@dataclass(frozen=True)
class AdversarialReport:
    """Summary of the shadow books."""

    observations: int
    matured_outcomes: int
    counter_hit_rate: float | None
    consensus_final_equity: float
    counter_final_equity: float
    consensus_max_drawdown: float
    counter_max_drawdown: float
    counter_outperformed: bool


class AdversarialAgent:
    """Shadow evaluator holding consensus and counter-consensus hypothetical books."""

    name = "adversarial"

    def __init__(self, initial_capital: float, fee_bps: float, slippage_bps: float, exposure: float = 1.0, horizon: int = 20) -> None:
        """Create the shadow books with the same cost model as the strategy."""
        if not 0 < exposure <= 1:
            raise ValueError("exposure must be in (0, 1]")
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.exposure = exposure
        self.horizon = horizon
        self.reset()

    def reset(self) -> None:
        """Clear both books and all observations."""
        mk = lambda: Portfolio(self.initial_capital, self.fee_bps, self.slippage_bps, allow_short=True)  # noqa: E731
        self.consensus = mk()
        self.counter = mk()
        self._pending: tuple[int, float] | None = None
        self.observations: list[tuple[int, int]] = []  # (bar_index, consensus direction)
        self.consensus_equity: list[float] = []
        self.counter_equity: list[float] = []

    def observe(self, bar_index: int, consensus_direction: int) -> None:
        """Record the consensus vote at bar close; schedule shadow orders for the next open."""
        if consensus_direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or 1")
        if consensus_direction != 0:
            self.observations.append((bar_index, consensus_direction))
            self._pending = (bar_index, consensus_direction * self.exposure)

    def on_open(self, bar_index: int, timestamp: datetime, open_price: float) -> None:
        """Execute pending shadow orders at this bar's open (strictly after the decision bar)."""
        if self._pending is None:
            return
        dec_idx, target = self._pending
        if bar_index <= dec_idx:
            raise RuntimeError("look-ahead in adversarial shadow execution")
        self._pending = None
        reason = "shadow"
        if abs(self.consensus.exposure(open_price) - target) > 1e-9:
            self.consensus.rebalance_to(target, open_price, timestamp, bar_index, reason)
        if abs(self.counter.exposure(open_price) + target) > 1e-9:
            self.counter.rebalance_to(-target, open_price, timestamp, bar_index, reason)

    def on_close(self, close_price: float) -> None:
        """Mark both books to market."""
        self.consensus_equity.append(self.consensus.equity(close_price))
        self.counter_equity.append(self.counter.equity(close_price))

    def report(self, prices: np.ndarray, as_of_index: int) -> AdversarialReport:
        """Summarize shadow performance with outcomes matured by ``as_of_index``."""
        matured = [(i, d) for i, d in self.observations if i + self.horizon <= as_of_index]
        hits = [1.0 if -d * (prices[i + self.horizon] / prices[i] - 1.0) > 0 else 0.0 for i, d in matured]

        def mdd(eq: list[float]) -> float:
            if not eq:
                return 0.0
            a = np.asarray(eq)
            return float((1 - a / np.maximum.accumulate(a)).max())

        ce = self.consensus_equity[-1] if self.consensus_equity else self.initial_capital
        ke = self.counter_equity[-1] if self.counter_equity else self.initial_capital
        return AdversarialReport(
            observations=len(self.observations),
            matured_outcomes=len(matured),
            counter_hit_rate=float(np.mean(hits)) if hits else None,
            consensus_final_equity=ce,
            counter_final_equity=ke,
            consensus_max_drawdown=mdd(self.consensus_equity),
            counter_max_drawdown=mdd(self.counter_equity),
            counter_outperformed=ke > ce,
        )
