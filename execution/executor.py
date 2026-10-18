"""Portfolio accounting, simulated execution and the paper executor.

Execution model (default ``next_bar_open``): an order created from a decision
at bar t's close is filled at bar t+1's open. Buys fill at
``open * (1 + slippage)``, sells at ``open * (1 - slippage)``; a fee of
``fee_bps`` is charged on the executed notional of every fill (entries,
exits and both legs of a reversal). If bar t+1 does not exist the order stays
unfilled - no price is invented.

No real broker is ever called. ``PaperExecutor`` logs every simulated order;
``NullBroker`` raises if anything tries to route a live order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from core.models import Fill, Order, OrderSide, Trade

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class Portfolio:
    """Single-instrument cash + position accounting with round-trip trade ledger."""

    initial_capital: float
    fee_bps: float
    slippage_bps: float
    allow_short: bool = False
    max_leverage: float = 1.0
    cash: float = field(init=False)
    units: float = field(init=False, default=0.0)
    fills: list[Fill] = field(init=False, default_factory=list)
    trades: list[Trade] = field(init=False, default_factory=list)
    open_trade: Trade | None = field(init=False, default=None)
    total_fees: float = field(init=False, default=0.0)
    total_slippage: float = field(init=False, default=0.0)
    traded_notional: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.cash = float(self.initial_capital)
        self._avg_price = 0.0

    @property
    def fee_rate(self) -> float:
        """Fee as a fraction of notional."""
        return self.fee_bps / 1e4

    @property
    def slip_rate(self) -> float:
        """Slippage as a fraction of the reference price."""
        return self.slippage_bps / 1e4

    def equity(self, price: float) -> float:
        """Mark-to-market equity at ``price``."""
        return self.cash + self.units * price

    def exposure(self, price: float) -> float:
        """Signed position value / equity."""
        eq = self.equity(price)
        return self.units * price / eq if eq > 0 else 0.0

    def unrealized_pnl(self, price: float) -> float:
        """Unrealized PnL of the open position at ``price`` (before exit costs)."""
        if self.open_trade is None:
            return 0.0
        return self.units * (price - self._avg_price)

    def rebalance_to(self, target_exposure: float, ref_price: float, timestamp: datetime, bar_index: int, reason: str) -> list[Fill]:
        """Trade to ``target_exposure`` of current equity at ``ref_price``.

        A reversal is executed as a close followed by an open so both legs pay
        costs and the trade ledger stays correct. Long buys are capped by the
        cash available after fees and slippage (no unintended leverage).
        """
        if not math.isfinite(ref_price) or ref_price <= 0:
            raise ValueError(f"invalid reference price {ref_price}")
        if target_exposure < 0 and not self.allow_short:
            raise ValueError("negative target exposure requires allow_short")
        if abs(target_exposure) > self.max_leverage + 1e-9:
            raise ValueError(f"target exposure {target_exposure} exceeds max_leverage {self.max_leverage}")
        fills: list[Fill] = []
        if self.units != 0 and (target_exposure == 0 or math.copysign(1, target_exposure) != math.copysign(1, self.units)):
            fills.append(self._execute(-self.units, ref_price, timestamp, bar_index, reason))
        eq = self.equity(ref_price)
        if eq <= 0:
            return fills
        target_units = target_exposure * eq / ref_price
        delta = target_units - self.units
        if delta > 0 and self.units >= 0:
            # Cash-constrained long buy: units * price * (1+slip) * (1+fee) <= cash * max_leverage headroom.
            unit_cost = ref_price * (1 + self.slip_rate) * (1 + self.fee_rate)
            affordable = (self.cash - (1.0 - self.max_leverage) * eq) / unit_cost
            delta = min(delta, max(0.0, affordable))
        if delta < 0 and self.units <= 0 and target_exposure < 0:
            # Short sale: cap so that |position| * price * (1+costs) <= leverage * equity.
            max_units = self.max_leverage * eq / (ref_price * (1 + self.slip_rate) * (1 + self.fee_rate))
            delta = max(delta, -max_units - self.units)
        if abs(delta) * ref_price > _EPS * max(1.0, eq) and abs(delta) > _EPS:
            fills.append(self._execute(delta, ref_price, timestamp, bar_index, reason))
        return fills

    def _execute(self, qty: float, ref_price: float, timestamp: datetime, bar_index: int, reason: str) -> Fill:
        """Execute a signed quantity with slippage and fees and update the ledger."""
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        exec_price = ref_price * (1 + self.slip_rate) if qty > 0 else ref_price * (1 - self.slip_rate)
        notional = abs(qty) * exec_price
        fee = notional * self.fee_rate
        slip_cost = abs(qty) * ref_price * self.slip_rate
        self.cash -= qty * exec_price + fee
        self.total_fees += fee
        self.total_slippage += slip_cost
        self.traded_notional += notional
        prev_units = self.units
        new_units = prev_units + qty
        if abs(new_units) < 1e-12:
            new_units = 0.0
        if prev_units == 0:
            self.open_trade = Trade(
                direction=1 if qty > 0 else -1, entry_time=timestamp, entry_index=bar_index,
                entry_price=exec_price, quantity=abs(qty), entry_reason=reason, max_quantity=abs(qty),
            )
            self._avg_price = exec_price
            self.open_trade.fees += fee
            self.open_trade.slippage += slip_cost
        elif math.copysign(1, qty) == math.copysign(1, prev_units):
            tot = abs(prev_units) + abs(qty)
            self._avg_price = (self._avg_price * abs(prev_units) + exec_price * abs(qty)) / tot
            assert self.open_trade is not None
            self.open_trade.entry_price = self._avg_price
            self.open_trade.quantity = tot
            self.open_trade.max_quantity = max(self.open_trade.max_quantity, tot)
            self.open_trade.fees += fee
            self.open_trade.slippage += slip_cost
        else:
            if abs(qty) > abs(prev_units) + 1e-12:
                raise RuntimeError("reversal must be split into close + open")
            t = self.open_trade
            assert t is not None
            closed = abs(qty)
            t.gross_pnl += closed * (exec_price - self._avg_price) * t.direction
            t.fees += fee
            t.slippage += slip_cost
            t.exit_quantity += closed
            t.exit_notional += closed * exec_price
            t.quantity = abs(new_units)
            if new_units == 0.0:
                t.exit_time = timestamp
                t.exit_index = bar_index
                t.exit_price = t.exit_notional / t.exit_quantity
                t.exit_reason = reason
                self.trades.append(t)
                self.open_trade = None
                self._avg_price = 0.0
        self.units = new_units
        fill = Fill(timestamp, bar_index, side, abs(qty), ref_price, exec_price, notional, fee, slip_cost, reason)
        self.fills.append(fill)
        return fill


class BrokerAdapter(Protocol):
    """Interface a real broker integration would implement."""

    def submit_order(self, order: Order, quantity: float, side: OrderSide) -> str:
        """Submit a live order and return the broker order id."""
        ...


class NullBroker:
    """Default broker: refuses to send anything anywhere."""

    def submit_order(self, order: Order, quantity: float, side: OrderSide) -> str:
        """Always raises - live order routing is not implemented."""
        raise RuntimeError("live order routing is disabled: no broker integration is configured")


class SimulatedExecutor:
    """Holds at most one pending target-exposure order and fills it at the next open."""

    def __init__(self, portfolio: Portfolio) -> None:
        """Wrap ``portfolio``."""
        self.portfolio = portfolio
        self.pending: Order | None = None
        self.unfilled: list[Order] = []

    def submit(self, order: Order) -> None:
        """Queue ``order`` for the next bar open (replaces any earlier pending order)."""
        self.pending = order

    def on_bar_open(self, bar_index: int, timestamp: datetime, open_price: float) -> list[Fill]:
        """Fill the pending order at this bar's open (must be after the decision bar)."""
        order = self.pending
        if order is None:
            return []
        if bar_index <= order.decision_index:
            raise RuntimeError("look-ahead: attempted to fill an order on or before its decision bar")
        self.pending = None
        return self.portfolio.rebalance_to(order.target_exposure, open_price, timestamp, bar_index, order.reason)

    def finalize(self) -> None:
        """Record a still-pending order as unfilled (no next bar was available)."""
        if self.pending is not None:
            logger.info("order from bar %d left unfilled: no next bar in window", self.pending.decision_index)
            self.unfilled.append(self.pending)
            self.pending = None


class PaperExecutor(SimulatedExecutor):
    """Simulated executor that logs every order as a paper trade. Never routes live."""

    def __init__(self, portfolio: Portfolio, broker: BrokerAdapter | None = None, live_orders_enabled: bool = False) -> None:
        """Create a paper executor; ``live_orders_enabled`` must stay False (no live support)."""
        super().__init__(portfolio)
        if live_orders_enabled:
            raise RuntimeError("live order routing is not implemented; paper mode only")
        self.broker = broker or NullBroker()
        self.order_log: list[dict[str, object]] = []

    def submit(self, order: Order) -> None:
        """Log and queue a simulated order."""
        logger.info(
            "PAPER ORDER (simulated, no broker) decided %s -> target exposure %.3f | %s",
            order.decision_time, order.target_exposure, order.reason[:160],
        )
        self.order_log.append({"decision_time": order.decision_time, "target_exposure": order.target_exposure})
        super().submit(order)

    def on_bar_open(self, bar_index: int, timestamp: datetime, open_price: float) -> list[Fill]:
        """Fill at the open and log simulated fills."""
        fills = super().on_bar_open(bar_index, timestamp, open_price)
        for f in fills:
            logger.info(
                "PAPER FILL (simulated) %s %s %.6f @ %.2f (ref %.2f) fee %.2f",
                f.timestamp, f.side.value, f.quantity, f.exec_price, f.reference_price, f.fee,
            )
        return fills
