# Multi-Agent Trading Framework — Deliverables

Deliverables 1–8 in the requested order. Every file is reproduced in full from the tested source tree.

## Deliverable 1 — Design write-up


## Architecture

```
data/feed.py (synthetic | live stub)  ->  validate_bars
        |
core/feature_bus.py   causal indicators (indicators/*) + provenance labels -> MarketState per bar
        |
agents/*  five voting specialists (+ optional on_chain)  -> AgentSignal (signal, confidence, reason, features)
        |
agents/boss_orchestrator.py  weighted vote, agreement/coverage/strength, risk veto, kill switch, template reasoner
        |
execution/backtester.py  position management -> Order (decided at close t)
        |
execution/executor.py    fill at open t+1 with slippage + fees, cash/position/trade ledger (Backtest or Paper)
        |
execution/evaluator.py   metrics, agent scoring on matured outcomes, signal & execution ablation, calibration
        |
execution/feedback.py    every 60 bars (training only): bounded, shrunk, smoothed learned weights
        |
execution/walk_forward.py  train -> embargo -> validation -> embargo -> frozen test, chained OOS report
```

Side components: `core/portfolio_risk.py` (kill switch on the strategy's own equity), `core/kalman.py`
(per-agent confidence smoothing), `core/reasoner.py` (explanations only), `agents/adversarial_agent.py`
(shadow books, never votes).

## Data flow for one bar t

1. The executor fills the order decided at t-1 at bar t's open.
2. Equity is marked at t's close and fed to the kill switch.
3. In training runs only, the feedback loop may re-weight agents using predictions whose 20-bar outcome has
   fully matured by t.
4. `FeatureFrame.state(t)` builds a `MarketState` from bars <= t. Every indicator is trailing-window only.
   A test recomputes features on truncated history and checks they match.
5. Each agent returns an `AgentSignal`. An agent missing essential inputs returns an *unavailable* HOLD.
6. The boss computes S, agreement, coverage, data coverage and strength, then applies the override rules
   and writes the analyst note.
7. Position management turns the decision into a target exposure, and the order is queued for t+1.

## Why each agent exists

| Agent | Contribution | Inputs |
|---|---|---|
| MarketIntel | short-horizon flow pressure: OFI, CVD change, volume-confirmed momentum (an institutional-*style* proxy) | bid/ask aggressor volume, trade side, OHLCV |
| RiskEngine | market-risk score (vol, ATR%, market drawdown) driving a hard veto; never votes direction, so it also dampens S | OHLC |
| LiquidityEngine | event signal: confirmed sweeps of prior rolling lows/highs, qualified by CVD, depth and absorption | OHLCV, trade side, depth |
| SentimentEngine | social/news momentum plus contrarian funding | social, news, funding |
| RegimeDetector | regime label (EMA20/100, Wilder ADX, RV) that selects regime weights; votes only in trends | OHLC |
| OnChain (optional) | exchange netflow and stablecoin supply proxy | on-chain feed |

## Why the adversarial agent is separate

If it voted, it would cancel the ensemble by construction. As a shadow it answers one question with the same
timing and costs: would the negation of our vote have done better? It never feeds the weight updates, so its
verdict is not re-using the outcomes that tuned the strategy.

## Risk veto vs. kill switch

* **Risk veto (market):** RiskEngine score > 0.80 forces HOLD for new entries. Existing exposure is capped at
  `veto_exposure_cap` (`on_risk_veto: reduce`). If risk data is missing, the boss also forces HOLD
  (`risk_unavailable`, fail-safe).
* **Kill switch (portfolio):** this watches the strategy's net-of-cost equity. It triggers on drawdown >
  `max_drawdown` or rolling Sharpe < threshold once `min_history` returns exist. It blocks entries and
  flattens by default. Recovery happens after a cooldown, which resets the history.

## Adaptive weights without leakage

* Predictions at bar t are scored only when `t + h <= now`.
* Weight updates happen only in training windows. Validation and test runs are frozen, and a test asserts
  there are no updates.
* Each update applies: an insufficient-sample guard, a neutral band on the overlap-adjusted t-stat
  (n_eff = n/h), shrinkage toward the base weight, smoothing (0.25), and clipping to [0.3, 2.0].
* Embargo >= horizon is enforced by config validation. Thresholds are chosen on validation, never on test.
  Calibration is fitted on training data and scored on test.

## Key assumptions (details in Deliverable 8)

The data is synthetic toy dynamics with a deliberately weak flow/regime link. Execution is one instrument,
next-bar-open fills, and fixed bps costs with no market impact. Ensemble "confidence" is an uncalibrated
strength, not a probability.


## Deliverable 2 — Complete source code

### `core/__init__.py`

```python
"""Shared contracts, configuration, feature bus and risk utilities."""
```

### `core/models.py`

```python
"""Typed data contracts shared by every layer of the framework.

All structured objects that cross a module boundary (market state, agent
signals, boss decisions, orders, fills, trades, evaluation results) are
defined here as dataclasses so that the whole system speaks one vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional

FeatureValue = float | str | bool | None


class SignalType(str, Enum):
    """Directional vote of an agent or of the boss."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    @property
    def direction(self) -> int:
        """Return +1 for BUY, -1 for SELL and 0 for HOLD."""
        return {"BUY": 1, "SELL": -1, "HOLD": 0}[self.value]


class FeatureStatus(str, Enum):
    """Provenance label attached to every market-state feature.

    OBSERVED    - came from a real market-data provider.
    SYNTHETIC   - generated by the deterministic synthetic feed (never real data).
    DERIVED     - computed by the feature bus from other available inputs.
    UNAVAILABLE - not supplied; consumers must not treat it as a neutral value.
    """

    OBSERVED = "observed"
    SYNTHETIC = "synthetic"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"


class Regime(str, Enum):
    """Market regime labels produced by the RegimeDetectorAgent."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"  # only during indicator warm-up / missing inputs


RAW_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_volume",
    "ask_volume",
    "trade_side",
    "bid_depth",
    "ask_depth",
    "social_sentiment",
    "news_sentiment",
    "funding_rate",
)


def _finite(value: object) -> bool:
    """Return True when ``value`` is a real, finite number (bools excluded)."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class MarketState:
    """Everything an agent is allowed to know at one decision timestamp.

    The state is built by the feature bus from bars up to and including the
    decision bar ``bar_index``; it never contains information from later bars.
    ``availability`` maps every feature name (raw fields, indicator names and
    optional on-chain/options keys) to its provenance label.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    trade_side: Optional[float] = None
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None
    social_sentiment: Optional[float] = None
    news_sentiment: Optional[float] = None
    funding_rate: Optional[float] = None
    onchain: Mapping[str, float] = field(default_factory=dict)
    options: Mapping[str, float] = field(default_factory=dict)
    indicators: Mapping[str, float] = field(default_factory=dict)
    availability: Mapping[str, FeatureStatus] = field(default_factory=dict)
    bar_index: int = 0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("MarketState.timestamp must be timezone-aware")

    def status(self, name: str) -> FeatureStatus:
        """Return the provenance label of ``name`` (UNAVAILABLE if unknown)."""
        return self.availability.get(name, FeatureStatus.UNAVAILABLE)

    def value(self, name: str) -> Optional[float]:
        """Return the raw value of a feature from any section of the state."""
        if name in RAW_FIELDS:
            return getattr(self, name)
        if name in self.indicators:
            return self.indicators[name]
        if name in self.onchain:
            return self.onchain[name]
        if name in self.options:
            return self.options[name]
        return None

    def is_valid(self, name: str) -> bool:
        """True when the feature is not UNAVAILABLE and holds a finite number."""
        if self.status(name) == FeatureStatus.UNAVAILABLE:
            return False
        return _finite(self.value(name))

    def missing(self, names: tuple[str, ...] | list[str]) -> list[str]:
        """Return the subset of ``names`` that are unavailable or invalid."""
        return [n for n in names if not self.is_valid(n)]

    def get(self, name: str) -> float:
        """Return a feature known to be valid as ``float`` (raises otherwise)."""
        if not self.is_valid(name):
            raise KeyError(f"feature '{name}' is unavailable or not finite")
        return float(self.value(name))  # type: ignore[arg-type]


@dataclass(frozen=True)
class AgentSignal:
    """Output contract for every voting agent.

    ``available`` is False when the agent lacked essential inputs; such a
    signal is always HOLD and is excluded from the boss vote so that it can
    never masquerade as an independent confirmation.
    ``summary`` is a short phrase (built from real feature values) used by the
    template reasoner; ``reason`` is the full explanation.
    """

    agent_name: str
    signal: SignalType
    confidence: float
    reason: str
    features: dict[str, FeatureValue]
    timestamp: datetime
    available: bool = True
    summary: str = ""

    def __post_init__(self) -> None:
        if not _finite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.agent_name}: confidence must be in [0, 1], got {self.confidence}")
        if not self.available and self.signal != SignalType.HOLD:
            raise ValueError(f"{self.agent_name}: unavailable agents must return HOLD")
        if not self.reason:
            raise ValueError(f"{self.agent_name}: reason must be non-empty")

    @property
    def signed_confidence(self) -> float:
        """Confidence multiplied by direction (+c, 0 or -c)."""
        return self.confidence * self.signal.direction


@dataclass(frozen=True)
class AgentContribution:
    """Per-agent line of the boss decision breakdown."""

    agent_name: str
    signal: SignalType
    confidence: float
    signed_input: float
    base_weight: float
    learned_weight: float
    regime_factor: float
    family_factor: float
    effective_weight: float
    contribution: float
    eligible: bool
    available: bool
    reason: str


@dataclass(frozen=True)
class BossDecision:
    """Aggregated ensemble decision with full diagnostics.

    ``confidence`` is an *uncalibrated ensemble strength* in [0, 1]; it is not a
    probability of profit (see ``confidence_label``).
    ``raw_action`` is the vote before risk veto / kill switch / floors.
    """

    action: SignalType
    raw_action: SignalType
    confidence: float
    directional_score: float
    agreement: float
    coverage: float
    data_coverage: float
    reason: str
    breakdown: list[AgentContribution]
    regime: Regime
    risk_score: Optional[float]
    risk_veto: bool
    kill_switch_active: bool
    forced_hold_rule: Optional[str]
    timestamp: datetime
    confidence_label: str = "uncalibrated_ensemble_strength"


class OrderSide(str, Enum):
    """Side of a simulated order."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    """A target-exposure order generated at a decision bar's close."""

    decision_time: datetime
    decision_index: int
    target_exposure: float
    reason: str
    tag: str = "strategy"


@dataclass(frozen=True)
class Fill:
    """A simulated execution."""

    timestamp: datetime
    bar_index: int
    side: OrderSide
    quantity: float
    reference_price: float
    exec_price: float
    notional: float
    fee: float
    slippage_cost: float
    reason: str


@dataclass
class Trade:
    """A round-trip position from flat to flat (or reversal)."""

    direction: int
    entry_time: datetime
    entry_index: int
    entry_price: float
    quantity: float
    entry_reason: str
    exit_time: Optional[datetime] = None
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    max_quantity: float = 0.0
    exit_quantity: float = 0.0
    exit_notional: float = 0.0

    @property
    def net_pnl(self) -> float:
        """Realized PnL after fees (slippage is already inside exec prices)."""
        return self.gross_pnl - self.fees

    @property
    def is_closed(self) -> bool:
        """True once the position has returned to flat."""
        return self.exit_index is not None

    @property
    def bars_held(self) -> Optional[int]:
        """Number of bars between entry and exit fills."""
        if self.exit_index is None:
            return None
        return self.exit_index - self.entry_index


@dataclass(frozen=True)
class PerformanceMetrics:
    """Net-of-cost performance summary. ``None`` means unavailable (see notes)."""

    initial_equity: float
    final_equity: float
    total_return: float
    cagr: Optional[float]
    sharpe: Optional[float]
    max_drawdown: float
    n_trades: int
    win_rate: Optional[float]
    profit_factor: Optional[float]
    turnover: float
    total_fees: float
    total_slippage: float
    avg_holding_bars: Optional[float]
    avg_exposure: float
    avg_long_exposure: float
    avg_short_exposure: float
    n_bars: int
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentScore:
    """Evaluator output for one agent, using matured outcomes only."""

    agent_name: str
    prediction_count: int
    directional_count: int
    matured_count: int
    hit_rate: Optional[float]
    confidence_weighted_return: Optional[float]
    standalone_pnl: Optional[float]
    incremental_contribution: Optional[float]
    incremental_tstat: Optional[float]
    current_weight: float
    suggested_weight: float
    note: str = ""


@dataclass(frozen=True)
class WeightUpdate:
    """Audit record of one feedback-loop weight change."""

    bar_index: int
    timestamp: datetime
    agent_name: str
    old_weight: float
    new_weight: float
    evidence: str
```

### `core/config.py`

```python
"""Configuration schema and loader.

The YAML file is validated with Pydantic at startup; unknown keys are rejected
(``extra="forbid"``) so that typos never silently become no-ops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CORE_AGENTS: tuple[str, ...] = (
    "market_intel",
    "risk_engine",
    "liquidity_engine",
    "sentiment_engine",
    "regime_detector",
)
OPTIONAL_AGENTS: tuple[str, ...] = ("on_chain",)
REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range", "volatile")
TIMEFRAME_BARS_PER_YEAR: dict[str, int] = {
    # Crypto trades 24/7, so bars per year = 365 days * bars per day.
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}
TIMEFRAME_PANDAS_FREQ: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1D"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppSection(_Strict):
    seed: int = 42
    timezone: Literal["UTC"] = "UTC"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class DataSection(_Strict):
    timeframe: Literal["1h", "4h", "1d"] = "4h"
    warmup_bars: int = Field(250, ge=0)
    min_history_bars: int = Field(120, ge=1)


class StrategySection(_Strict):
    min_confidence: float = Field(0.35, ge=0.0, le=1.0)
    directional_score_floor: float = Field(0.15, ge=0.0, le=1.0)
    min_data_coverage: float = Field(0.5, ge=0.0, le=1.0)
    prediction_horizon: int = Field(20, ge=1)
    long_only: bool = True
    allow_short: bool = False
    sell_policy: Literal["close", "reduce"] = "close"
    sell_reduce_fraction: float = Field(0.5, gt=0.0, le=1.0)
    min_rebalance_exposure: float = Field(0.05, ge=0.0, le=1.0)
    on_risk_veto: Literal["hold_position", "reduce"] = "reduce"
    veto_exposure_cap: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_short(self) -> "StrategySection":
        if self.long_only and self.allow_short:
            raise ValueError("strategy.long_only=true conflicts with strategy.allow_short=true")
        return self


class ExecutionSection(_Strict):
    initial_capital: float = Field(100_000.0, gt=0)
    fee_bps: float = Field(10.0, ge=0, le=500)
    slippage_bps: float = Field(5.0, ge=0, le=500)
    execution_price: Literal["next_bar_open"] = "next_bar_open"
    max_position_size: float = Field(1.0, gt=0)
    max_exposure: float = Field(1.0, gt=0)
    exposure_cap: float = Field(1.0, gt=0)
    max_leverage: float = Field(1.0, gt=0)
    volatility_aware_sizing: bool = False
    target_volatility: float = Field(0.5, gt=0)

    @model_validator(mode="after")
    def _check_leverage(self) -> "ExecutionSection":
        if self.max_exposure > self.max_leverage:
            raise ValueError(
                f"execution.max_exposure ({self.max_exposure}) exceeds execution.max_leverage "
                f"({self.max_leverage}); raise max_leverage explicitly if leverage is intended"
            )
        if self.max_position_size > self.max_exposure:
            raise ValueError("execution.max_position_size must be <= execution.max_exposure")
        return self


class RiskSection(_Strict):
    risk_veto_threshold: float = Field(0.80, gt=0.0, le=1.0)
    max_drawdown: float = Field(0.20, gt=0.0, lt=1.0)
    rolling_sharpe_threshold: float = -1.0
    rolling_sharpe_window: int = Field(120, ge=10)
    kill_switch_min_history: int = Field(60, ge=2)
    kill_switch_recovery: Literal["cooldown", "manual"] = "cooldown"
    kill_switch_cooldown_bars: int = Field(60, ge=1)
    kill_switch_position_policy: Literal["flatten", "hold"] = "flatten"


class EvaluationSection(_Strict):
    lookahead: int = Field(20, ge=1)
    feedback_interval: int = Field(60, ge=1)
    min_agent_samples: int = Field(30, ge=1)
    min_weight: float = Field(0.3, gt=0)
    max_weight: float = Field(2.0, gt=0)
    weight_smoothing: float = Field(0.25, gt=0, le=1.0)
    shrinkage_to_base: float = Field(0.5, ge=0, le=1.0)
    tstat_neutral_band: float = Field(1.0, ge=0)
    weight_sensitivity: float = Field(0.15, ge=0)
    calibration: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> "EvaluationSection":
        if self.min_weight >= self.max_weight:
            raise ValueError("evaluation.min_weight must be < evaluation.max_weight")
        return self


class WalkForwardSection(_Strict):
    enabled: bool = False
    train_bars: int = Field(500, ge=50)
    validation_bars: int = Field(100, ge=10)
    test_bars: int = Field(100, ge=10)
    embargo_bars: int = Field(20, ge=0)
    step_bars: int | None = None
    threshold_grid: list[float] = Field(default_factory=lambda: [0.25, 0.35, 0.45])

    @field_validator("threshold_grid")
    @classmethod
    def _grid(cls, v: list[float]) -> list[float]:
        if not v or any(not 0 <= x <= 1 for x in v):
            raise ValueError("walk_forward.threshold_grid must be non-empty values in [0, 1]")
        return v


class FeaturesSection(_Strict):
    enable_optional_agent: bool = False
    optional_agent_type: Literal["on_chain"] = "on_chain"
    enable_kalman: bool = True
    enable_regime_weights: bool = True
    enable_adversarial: bool = True
    enable_family_normalization: bool = False


class IndicatorSection(_Strict):
    ofi_window: int = Field(20, ge=1)
    cvd_window: int = Field(50, ge=2)
    momentum_lookback: int = Field(20, ge=2)
    pv_corr_window: int = Field(50, ge=3)
    rv_window: int = Field(30, ge=2)
    atr_window: int = Field(14, ge=1)
    drawdown_window: int = Field(120, ge=2)
    ema_fast: int = Field(20, ge=2)
    ema_slow: int = Field(100, ge=3)
    adx_period: int = Field(14, ge=2)
    sweep_lookback: int = Field(20, ge=2)
    absorption_window: int = Field(20, ge=2)
    netflow_window: int = Field(50, ge=5)

    @model_validator(mode="after")
    def _ema(self) -> "IndicatorSection":
        if self.ema_fast >= self.ema_slow:
            raise ValueError("indicators.ema_fast must be < indicators.ema_slow")
        return self


class MarketIntelParams(_Strict):
    w_ofi: float = 0.4
    w_cvd: float = 0.2
    w_momentum: float = 0.4
    signal_threshold: float = Field(0.15, gt=0, lt=1)


class RiskEngineParams(_Strict):
    rv_low: float = Field(0.30, ge=0)
    rv_high: float = Field(1.50, gt=0)
    atr_pct_low: float = Field(0.005, ge=0)
    atr_pct_high: float = Field(0.04, gt=0)
    drawdown_full: float = Field(0.35, gt=0, le=1)
    w_vol: float = Field(0.4, ge=0)
    w_atr: float = Field(0.3, ge=0)
    w_drawdown: float = Field(0.3, ge=0)

    @model_validator(mode="after")
    def _ranges(self) -> "RiskEngineParams":
        if self.rv_low >= self.rv_high or self.atr_pct_low >= self.atr_pct_high:
            raise ValueError("risk_engine normalization: *_low must be < *_high")
        if self.w_vol + self.w_atr + self.w_drawdown <= 0:
            raise ValueError("risk_engine component weights must sum to > 0")
        return self


class LiquidityParams(_Strict):
    min_confirmations: int = Field(1, ge=0, le=2)
    absorption_threshold: float = Field(1.5, gt=0)
    depth_threshold: float = Field(0.05, ge=0, lt=1)
    cvd_threshold: float = Field(0.0, ge=0, lt=1)


class SentimentParams(_Strict):
    social_min: float = 0.0
    social_max: float = 100.0
    news_min: float = -1.0
    news_max: float = 1.0
    funding_threshold: float = Field(0.0003, ge=0)
    funding_saturation: float = Field(0.0010, gt=0)
    w_social: float = Field(0.35, ge=0)
    w_news: float = Field(0.35, ge=0)
    w_funding: float = Field(0.30, ge=0)
    min_components: int = Field(2, ge=1, le=3)
    signal_threshold: float = Field(0.20, gt=0, lt=1)
    treat_missing_as_neutral: bool = False

    @model_validator(mode="after")
    def _check(self) -> "SentimentParams":
        if self.social_min >= self.social_max or self.news_min >= self.news_max:
            raise ValueError("sentiment normalization ranges must have min < max")
        if self.funding_saturation <= self.funding_threshold:
            raise ValueError("sentiment.funding_saturation must exceed funding_threshold")
        return self


class RegimeParams(_Strict):
    adx_trend: float = Field(20.0, ge=0)
    adx_strong: float = Field(40.0, gt=0)
    volatile_rv: float = Field(1.2, gt=0)
    directional_vote: bool = True

    @model_validator(mode="after")
    def _check(self) -> "RegimeParams":
        if self.adx_strong <= self.adx_trend:
            raise ValueError("regime_detector.adx_strong must exceed adx_trend")
        return self


class OnChainParams(_Strict):
    netflow_z_threshold: float = Field(0.5, gt=0)
    w_netflow: float = 0.6
    w_stablecoin: float = 0.4


class AgentsSection(_Strict):
    base_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "market_intel": 1.0,
            "risk_engine": 1.0,
            "liquidity_engine": 1.0,
            "sentiment_engine": 1.0,
            "regime_detector": 1.0,
            "on_chain": 0.8,
        }
    )
    families: dict[str, str] = Field(
        default_factory=lambda: {
            "market_intel": "price_flow",
            "regime_detector": "price_flow",
            "liquidity_engine": "microstructure",
            "sentiment_engine": "sentiment",
            "risk_engine": "risk",
            "on_chain": "on_chain",
        }
    )
    market_intel: MarketIntelParams = Field(default_factory=MarketIntelParams)
    risk_engine: RiskEngineParams = Field(default_factory=RiskEngineParams)
    liquidity_engine: LiquidityParams = Field(default_factory=LiquidityParams)
    sentiment_engine: SentimentParams = Field(default_factory=SentimentParams)
    regime_detector: RegimeParams = Field(default_factory=RegimeParams)
    on_chain: OnChainParams = Field(default_factory=OnChainParams)


class KalmanSection(_Strict):
    process_noise: float = Field(0.02, gt=0)
    observation_noise: float = Field(0.10, gt=0)
    initial_variance: float = Field(1.0, gt=0)


class SyntheticSection(_Strict):
    start_price: float = Field(20_000.0, gt=0)
    base_volume: float = Field(1_000.0, gt=0)
    base_vol_annual: float = Field(0.55, gt=0)
    min_segment_bars: int = Field(60, ge=5)
    max_segment_bars: int = Field(240, ge=5)
    flow_signal_strength: float = Field(0.25, ge=0, le=1)
    generate_optional: bool = True

    @model_validator(mode="after")
    def _seg(self) -> "SyntheticSection":
        if self.min_segment_bars > self.max_segment_bars:
            raise ValueError("synthetic.min_segment_bars must be <= max_segment_bars")
        return self


class FrameworkConfig(_Strict):
    """Root configuration object."""

    app: AppSection = Field(default_factory=AppSection)
    data: DataSection = Field(default_factory=DataSection)
    strategy: StrategySection = Field(default_factory=StrategySection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    risk: RiskSection = Field(default_factory=RiskSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    walk_forward: WalkForwardSection = Field(default_factory=WalkForwardSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    indicators: IndicatorSection = Field(default_factory=IndicatorSection)
    agents: AgentsSection = Field(default_factory=AgentsSection)
    regime_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    kalman: KalmanSection = Field(default_factory=KalmanSection)
    synthetic: SyntheticSection = Field(default_factory=SyntheticSection)

    @model_validator(mode="after")
    def _cross_checks(self) -> "FrameworkConfig":
        ev = self.evaluation
        known = set(CORE_AGENTS) | set(OPTIONAL_AGENTS)
        for name in CORE_AGENTS:
            if name not in self.agents.base_weights:
                raise ValueError(f"agents.base_weights is missing core agent '{name}'")
        for name, w in self.agents.base_weights.items():
            if name not in known:
                raise ValueError(f"agents.base_weights has unknown agent '{name}'")
            if not ev.min_weight <= w <= ev.max_weight:
                raise ValueError(
                    f"agents.base_weights.{name}={w} outside [{ev.min_weight}, {ev.max_weight}]"
                )
        if self.features.enable_optional_agent and self.features.optional_agent_type not in self.agents.base_weights:
            raise ValueError("optional agent enabled but has no base weight in agents.base_weights")
        for regime, table in self.regime_weights.items():
            if regime not in REGIMES:
                raise ValueError(f"regime_weights has unknown regime '{regime}' (expected {REGIMES})")
            for name, factor in table.items():
                if name not in known:
                    raise ValueError(f"regime_weights.{regime} has unknown agent '{name}'")
                if factor <= 0:
                    raise ValueError(f"regime_weights.{regime}.{name} must be > 0")
        if self.features.enable_regime_weights:
            for regime in REGIMES:
                if regime not in self.regime_weights:
                    raise ValueError(f"enable_regime_weights=true but regime_weights.{regime} missing")
        wf = self.walk_forward
        horizon = max(self.strategy.prediction_horizon, ev.lookahead)
        if wf.embargo_bars < horizon:
            raise ValueError(
                f"walk_forward.embargo_bars ({wf.embargo_bars}) must be >= the outcome horizon "
                f"({horizon}) to prevent overlapping-label leakage"
            )
        if ev.lookahead != self.strategy.prediction_horizon:
            raise ValueError("evaluation.lookahead must equal strategy.prediction_horizon")
        if not self.strategy.long_only and not self.strategy.allow_short:
            raise ValueError("set strategy.allow_short=true when strategy.long_only=false")
        return self

    @property
    def bars_per_year(self) -> int:
        """Annualization factor implied by ``data.timeframe``."""
        return TIMEFRAME_BARS_PER_YEAR[self.data.timeframe]

    def enabled_agents(self) -> list[str]:
        """Names of agents that participate in the vote."""
        names = list(CORE_AGENTS)
        if self.features.enable_optional_agent:
            names.append(self.features.optional_agent_type)
        return names


class ConfigError(ValueError):
    """Raised when the configuration file cannot be loaded or validated."""


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> FrameworkConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: YAML file; ``None`` uses built-in defaults only.
        overrides: Nested dict merged over the file contents before validation.

    Returns:
        A validated ``FrameworkConfig``.

    Raises:
        ConfigError: With an actionable message for any invalid value.
    """
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"config file {p} must contain a mapping at top level")
        raw = loaded or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    try:
        return FrameworkConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError -> actionable message
        raise ConfigError(f"invalid configuration: {exc}") from exc


def _deep_merge(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` into a copy of ``base``."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
```

### `core/feature_bus.py`

```python
"""Indicator feature bus.

Computes every indicator once over the whole bar history using strictly
trailing (causal) functions, then serves a ``MarketState`` per bar. Because
each indicator value at bar t depends only on bars <= t, precomputing over the
full series is equivalent to recomputing on a truncated history (this is
verified in ``tests/test_integration.py``).

Indicator availability is DERIVED when all inputs are available and
UNAVAILABLE otherwise, so a missing raw feed can never surface as a valid
derived feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import FrameworkConfig
from core.models import FeatureStatus, MarketState, RAW_FIELDS
from data.feed import MarketData, ONCHAIN, OPTIONS
from indicators import liquidity as liq
from indicators import order_flow as of
from indicators import trend as tr
from indicators import volatility as vol

logger = logging.getLogger(__name__)


@dataclass
class FeatureFrame:
    """Bars joined with indicators plus provenance for every column."""

    symbol: str
    frame: pd.DataFrame
    availability: dict[str, FeatureStatus]
    first_trade_index: int
    indicator_names: list[str]

    def __post_init__(self) -> None:
        self._arrays = {c: self.frame[c].to_numpy(dtype=float) for c in self.frame.columns if c != "hidden_regime"}
        self._index = self.frame.index

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def index(self) -> pd.DatetimeIndex:
        """Bar timestamps."""
        return self._index

    def column(self, name: str) -> np.ndarray:
        """Numpy view of a numeric column."""
        return self._arrays[name]

    def state(self, i: int) -> MarketState:
        """Build the MarketState visible at the close of bar ``i``."""
        if not 0 <= i < len(self.frame):
            raise IndexError(f"bar index {i} out of range")
        a = self._arrays

        def raw(name: str) -> float | None:
            if self.availability.get(name, FeatureStatus.UNAVAILABLE) == FeatureStatus.UNAVAILABLE:
                return None
            return float(a[name][i]) if name in a else None

        indicators = {n: float(a[n][i]) for n in self.indicator_names}
        onchain = {n: float(a[n][i]) for n in ONCHAIN if n in a and self.availability.get(n) != FeatureStatus.UNAVAILABLE}
        options = {n: float(a[n][i]) for n in OPTIONS if n in a and self.availability.get(n) != FeatureStatus.UNAVAILABLE}
        return MarketState(
            timestamp=self._index[i].to_pydatetime(),
            symbol=self.symbol,
            open=float(a["open"][i]),
            high=float(a["high"][i]),
            low=float(a["low"][i]),
            close=float(a["close"][i]),
            volume=float(a["volume"][i]),
            bid_volume=raw("bid_volume"),
            ask_volume=raw("ask_volume"),
            trade_side=raw("trade_side"),
            bid_depth=raw("bid_depth"),
            ask_depth=raw("ask_depth"),
            social_sentiment=raw("social_sentiment"),
            news_sentiment=raw("news_sentiment"),
            funding_rate=raw("funding_rate"),
            onchain=onchain,
            options=options,
            indicators=indicators,
            availability=self.availability,
            bar_index=i,
        )


def compute_features(md: MarketData, config: FrameworkConfig) -> FeatureFrame:
    """Run every indicator over ``md.bars`` and return a :class:`FeatureFrame`."""
    ic = config.indicators
    bpy = config.bars_per_year
    b = md.bars
    av = dict(md.availability)
    for col in RAW_FIELDS:
        av.setdefault(col, FeatureStatus.UNAVAILABLE if col not in b.columns else av.get(col, FeatureStatus.OBSERVED))

    def ok(*cols: str) -> bool:
        return all(av.get(c, FeatureStatus.UNAVAILABLE) != FeatureStatus.UNAVAILABLE and c in b.columns for c in cols)

    nan = pd.Series(np.nan, index=b.index)
    feats: dict[str, pd.Series] = {}
    status: dict[str, FeatureStatus] = {}

    def put(name: str, series: pd.Series | None, inputs_ok: bool) -> None:
        feats[name] = series.astype(float) if (series is not None and inputs_ok) else nan
        status[name] = FeatureStatus.DERIVED if inputs_ok else FeatureStatus.UNAVAILABLE

    c, h, l, o, v = b["close"], b["high"], b["low"], b["open"], b["volume"]

    has_flow = ok("bid_volume", "ask_volume")
    put("ofi", of.order_flow_imbalance(b["bid_volume"], b["ask_volume"], ic.ofi_window) if has_flow else None, has_flow)
    has_side = ok("trade_side")
    cvd_series = of.cvd(c, v, b["trade_side"] if has_side else None)
    # CVD from the tick rule is a proxy; it is still derived but flagged.
    put("cvd_norm", of.cvd_change_normalized(cvd_series, v, ic.cvd_window), True)
    put("cvd_norm_short", of.cvd_change_normalized(cvd_series, v, 3), True)
    put("cvd_is_proxy", pd.Series(0.0 if has_side else 1.0, index=b.index), True)
    put("inst_momentum", of.institutional_momentum(c, v, ic.momentum_lookback), True)
    put("pv_corr", of.price_volume_correlation(c, v, ic.pv_corr_window), True)

    put("realized_vol", vol.realized_volatility(c, ic.rv_window, bpy), True)
    put("atr_pct", vol.atr_percent(h, l, c, ic.atr_window), True)
    put("parkinson_vol", vol.parkinson_volatility(h, l, ic.rv_window, bpy), True)
    put("garman_klass_vol", vol.garman_klass_volatility(o, h, l, c, ic.rv_window, bpy), True)
    put("market_drawdown", vol.rolling_drawdown(c, ic.drawdown_window), True)

    put("ema_fast", tr.ema(c, ic.ema_fast), True)
    put("ema_slow", tr.ema(c, ic.ema_slow), True)
    adx_df = tr.adx(h, l, c, ic.adx_period)
    for col in ("adx", "plus_di", "minus_di"):
        put(col, adx_df[col], True)

    sw = liq.detect_sweeps(h, l, c, ic.sweep_lookback)
    warm = sw["prior_low"].isna()
    put("bullish_sweep", sw["bullish_sweep"].astype(float).where(~warm), True)
    put("bearish_sweep", sw["bearish_sweep"].astype(float).where(~warm), True)
    put("prior_low", sw["prior_low"], True)
    put("prior_high", sw["prior_high"], True)
    put("absorption_ratio", liq.absorption_ratio(v, h, l, c, ic.absorption_window), True)
    has_depth = ok("bid_depth", "ask_depth")
    put("depth_imbalance", liq.depth_imbalance(b["bid_depth"], b["ask_depth"]) if has_depth else None, has_depth)

    has_netflow = ok("exchange_netflow")
    if has_netflow:
        nf = b["exchange_netflow"]
        mu = nf.rolling(ic.netflow_window, min_periods=ic.netflow_window).mean()
        sd = nf.rolling(ic.netflow_window, min_periods=ic.netflow_window).std(ddof=1)
        put("netflow_z", (nf - mu) / sd.where(sd > 0), True)
    else:
        put("netflow_z", None, False)
    has_stable = ok("stablecoin_supply_change")
    put("stablecoin_change", b["stablecoin_supply_change"] if has_stable else None, has_stable)

    frame = b.copy()
    for name, series in feats.items():
        frame[name] = series.to_numpy()
    av.update(status)
    logger.debug("feature bus computed %d indicators over %d bars", len(feats), len(frame))
    return FeatureFrame(
        symbol=md.symbol,
        frame=frame,
        availability=av,
        first_trade_index=md.first_trade_index,
        indicator_names=list(feats.keys()),
    )
```

### `core/kalman.py`

```python
"""One-dimensional Kalman filter for smoothing agent signed confidence.

Model: random walk  x_t = x_{t-1} + w,  w ~ N(0, q);  z_t = x_t + v,  v ~ N(0, r).

Latency vs smoothing: the steady-state gain is K = P/(P + r) with P solving
P = (P + q) r / (P + q + r). A larger ``q/r`` ratio gives a larger gain (faster
reaction to reversals, less smoothing); a smaller ratio smooths noise but
delays response to genuine flips. Missing observations run the predict step
only, so the estimate holds while its uncertainty grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Kalman1D:
    """Scalar Kalman filter with optional output clipping."""

    process_noise: float
    observation_noise: float
    initial_variance: float = 1.0
    lower: float = -1.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if self.process_noise <= 0 or self.observation_noise <= 0 or self.initial_variance <= 0:
            raise ValueError("Kalman noise parameters must be positive")
        if self.lower >= self.upper:
            raise ValueError("lower bound must be < upper bound")
        self.reset()

    def reset(self) -> None:
        """Forget all history (state becomes uninitialized)."""
        self.x: float | None = None
        self.p: float = self.initial_variance

    def update(self, z: float | None) -> float | None:
        """Assimilate observation ``z`` (None/NaN = missing) and return the estimate.

        The first valid observation initializes the state directly.
        Returns ``None`` until the filter has been initialized.
        """
        missing = z is None or not math.isfinite(z)
        if self.x is None:
            if missing:
                return None
            self.x = min(self.upper, max(self.lower, float(z)))  # type: ignore[arg-type]
            return self.x
        self.p += self.process_noise
        if not missing:
            k = self.p / (self.p + self.observation_noise)
            self.x = self.x + k * (float(z) - self.x)  # type: ignore[arg-type]
            self.p = (1.0 - k) * self.p
        self.x = min(self.upper, max(self.lower, self.x))
        return self.x
```

### `core/portfolio_risk.py`

```python
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
```

### `core/reasoner.py`

```python
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
```

### `indicators/__init__.py`

```python
"""Pure, causal indicator functions (no I/O, no state)."""
```

### `indicators/order_flow.py`

```python
"""Order-flow indicators.

Conventions
-----------
* ``bid_vol`` is **buyer-initiated** (aggressive buy) volume and ``ask_vol`` is
  **seller-initiated** (aggressive sell) volume in the same bar. With real data
  these come from trade prints classified by the exchange's aggressor flag.
* Order-flow imbalance over a trailing window ``W`` ending at bar t::

      OFI_t = (sum_{t-W+1..t} bid_vol - sum ask_vol) / (sum bid_vol + sum ask_vol)

  OFI lies in [-1, 1]; +1 means every traded unit was buyer-initiated.
  Positive OFI = buying pressure. Undefined (NaN) when the window has zero
  total volume or fewer than W observations.
* All rolling windows are trailing (end at the current bar); no value at t
  depends on bars after t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x: pd.Series | np.ndarray | list, name: str) -> pd.Series:
    """Coerce to float Series, raising on non-numeric input."""
    s = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype=float))
    try:
        return s.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def order_flow_imbalance(bid_vol: pd.Series, ask_vol: pd.Series, window: int) -> pd.Series:
    """Trailing-window order-flow imbalance in [-1, 1].

    Args:
        bid_vol: Buyer-initiated volume per bar (>= 0).
        ask_vol: Seller-initiated volume per bar (>= 0).
        window: Number of trailing bars (>= 1).

    Returns:
        Series aligned to the input. NaN during the first ``window - 1`` bars,
        when any input in the window is NaN, or when window volume is zero.

    Raises:
        ValueError: On negative volumes, mismatched lengths or window < 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    b = _as_series(bid_vol, "bid_vol")
    a = _as_series(ask_vol, "ask_vol")
    if len(b) != len(a):
        raise ValueError("bid_vol and ask_vol must have equal length")
    if (b < 0).any() or (a < 0).any():
        raise ValueError("volumes must be non-negative")
    a.index = b.index
    sb = b.rolling(window, min_periods=window).sum()
    sa = a.rolling(window, min_periods=window).sum()
    total = sb + sa
    ofi = (sb - sa) / total.where(total > 0)
    return ofi.rename("ofi")


def cvd(price: pd.Series, volume: pd.Series, side: pd.Series | None) -> pd.Series:
    """Cumulative volume delta.

    ``side`` is the **net aggressor fraction** of each bar in [-1, 1]:
    +1 = all buyer-initiated, -1 = all seller-initiated, 0 = balanced or
    ambiguous. Signed volume is ``side * volume`` so that, when ``volume`` is
    total traded volume, the per-bar delta equals buy volume minus sell volume.

    NaN ``side`` values (unknown aggressor) contribute **zero** delta; they are
    not guessed.

    If ``side`` is ``None`` the classic *tick rule* is used as an explicit
    **proxy**: the sign of the close-to-close price change, carrying the last
    non-zero sign through unchanged prices (first bar = 0). This is not
    genuine aggressor data and callers must label the result as a proxy.

    Args:
        price: Close prices (used only by the tick-rule proxy and for alignment).
        volume: Traded volume per bar (>= 0, NaN treated as 0).
        side: Net aggressor fraction or ``None`` for the tick-rule proxy.

    Returns:
        Cumulative sum of signed volume (Series named ``cvd``).
    """
    p = _as_series(price, "price")
    v = _as_series(volume, "volume")
    if len(p) != len(v):
        raise ValueError("price and volume must have equal length")
    v.index = p.index
    if (v < 0).any():
        raise ValueError("volume must be non-negative")
    if side is None:
        sign = np.sign(p.diff()).replace(0.0, np.nan).ffill().fillna(0.0)
    else:
        s = _as_series(side, "side")
        if len(s) != len(p):
            raise ValueError("side must match price length")
        s.index = p.index
        if ((s.abs() > 1.0 + 1e-12) & s.notna()).any():
            raise ValueError("side must lie in [-1, 1]")
        sign = s.fillna(0.0)
    delta = (sign * v.fillna(0.0))
    return delta.cumsum().rename("cvd")


def cvd_change_normalized(cvd_series: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Change of CVD over ``window`` bars divided by volume traded in that span.

    Lies in [-1, 1]; NaN when insufficient history or zero window volume.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = _as_series(cvd_series, "cvd")
    v = _as_series(volume, "volume")
    v.index = c.index
    dc = c - c.shift(window)
    vol = v.rolling(window, min_periods=window).sum()
    return (dc / vol.where(vol > 0)).clip(-1.0, 1.0).rename("cvd_norm")


def institutional_momentum(price: pd.Series, volume: pd.Series, lookback: int) -> pd.Series:
    """Volume-confirmed momentum score in [-1, 1] (an institutional-*style* proxy).

    Definition at bar t with lookback L::

        R_t   = ln(P_t / P_{t-L})
        sigma = stdev(1-bar log returns over the last L bars, ddof=1)
        z_t   = R_t / (sigma * sqrt(L))              (trend strength in sd units)
        rv_t  = mean(V over last L bars) / mean(V over the L bars before that)
        score = tanh(z_t) * min(rv_t, 2) / 2

    The volume factor scales the score by participation: equal volume gives
    half weight, doubled (or more) volume gives full weight. High volume alone
    does not imply institutional activity; this is only a proxy.

    Edge cases: needs ``2 * L`` bars; zero prior volume -> NaN;
    zero return variance -> score 0 if R_t == 0 else sign(R_t) * volume factor.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    p = _as_series(price, "price")
    v = _as_series(volume, "volume")
    v.index = p.index
    if (p <= 0).any():
        raise ValueError("prices must be positive")
    logp = np.log(p)
    r = logp.diff()
    R = logp - logp.shift(lookback)
    sigma = r.rolling(lookback, min_periods=lookback).std(ddof=1)
    vol_recent = v.rolling(lookback, min_periods=lookback).mean()
    vol_prior = vol_recent.shift(lookback)
    vratio = vol_recent / vol_prior.where(vol_prior > 0)
    vfac = vratio.clip(upper=2.0) / 2.0
    denom = sigma * np.sqrt(lookback)
    z = R / denom.where(denom > 0)
    trend = np.tanh(z)
    zero_var = (denom == 0) & R.notna()
    trend = trend.where(~zero_var, np.sign(R))
    score = trend * vfac
    return score.rename("inst_momentum")


def price_volume_correlation(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between 1-bar log returns and log volume change.

    Positive values mean volume expands with up moves (and contracts on down
    moves). NaN when either series is constant in the window.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    p = _as_series(price, "price")
    v = _as_series(volume, "volume")
    v.index = p.index
    r = np.log(p).diff()
    dv = np.log(v.where(v > 0)).diff()
    corr = r.rolling(window, min_periods=window).corr(dv)
    corr = corr.where(np.isfinite(corr))
    return corr.rename("pv_corr")
```

### `indicators/volatility.py`

```python
"""Volatility and drawdown indicators.

Return conventions
------------------
* ``realized_volatility``: sample stdev (ddof=1) of 1-bar **log** returns,
  annualized by ``sqrt(bars_per_year)``.
* ``parkinson_volatility`` / ``garman_klass_volatility``: range estimators on
  **log** price ratios, annualized the same way.
* ``atr_percent``: Wilder ATR divided by the close (a simple-price ratio).

``bars_per_year`` is supplied by the caller (derived from the configured
timeframe; crypto trades 24/7 so 4h bars -> 2190 bars per year).
Every window is trailing; insufficient history yields NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _check_positive(*series: pd.Series) -> None:
    """Raise if any non-NaN price is <= 0."""
    for s in series:
        if ((s <= 0) & s.notna()).any():
            raise ValueError("prices must be strictly positive")


def validate_ohlc(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    """Validate OHLC consistency (positive, high >= max(open, close), low <= min).

    Raises:
        ValueError: Describing the first violated rule.
    """
    _check_positive(open_, high, low, close)
    tol = 1e-9
    if (high + tol < np.maximum(open_, close)).any():
        raise ValueError("high must be >= max(open, close)")
    if (low - tol > np.minimum(open_, close)).any():
        raise ValueError("low must be <= min(open, close)")
    if (high + tol < low).any():
        raise ValueError("high must be >= low")


def realized_volatility(close: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    """Annualized realized volatility of log returns over a trailing window.

    Returns 0.0 for a constant price series (zero variance) once the window is
    full, and NaN before that.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")
    c = close.astype(float)
    _check_positive(c)
    r = np.log(c).diff()
    sd = r.rolling(window, min_periods=window).std(ddof=1)
    return (sd * math.sqrt(bars_per_year)).rename("realized_vol")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(H-L, |H-C_prev|, |L-C_prev|); first bar uses H-L."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[:1] = (high - low).iloc[:1]
    return tr.rename("true_range")


def wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: seed = SMA of the first ``period`` valid values, then
    ``s_t = s_{t-1} + (x_t - s_{t-1}) / period``.

    NaN inputs after seeding propagate the previous value (no update).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid_idx = np.flatnonzero(~np.isnan(x))
    if len(valid_idx) < period:
        return pd.Series(out, index=values.index)
    seed_pos = valid_idx[period - 1]
    out[seed_pos] = np.mean(x[valid_idx[:period]])
    prev = out[seed_pos]
    for i in range(seed_pos + 1, len(x)):
        if not np.isnan(x[i]):
            prev = prev + (x[i] - prev) / period
        out[i] = prev
    return pd.Series(out, index=values.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average true range with Wilder smoothing."""
    _check_positive(high, low, close)
    return wilder_smooth(true_range(high, low, close), window).rename("atr")


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """ATR as a fraction of the current close (0.02 = 2%)."""
    return (atr(high, low, close, window) / close).rename("atr_pct")


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    """Parkinson range volatility: sqrt(mean(ln(H/L)^2) / (4 ln 2)), annualized."""
    if window < 1:
        raise ValueError("window must be >= 1")
    _check_positive(high, low)
    hl = np.log(high / low) ** 2
    var = hl.rolling(window, min_periods=window).mean() / (4.0 * math.log(2.0))
    return (np.sqrt(var) * math.sqrt(bars_per_year)).rename("parkinson_vol")


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, bars_per_year: int
) -> pd.Series:
    """Garman-Klass volatility, annualized.

    Per-bar variance: 0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2, averaged over
    the window. A negative window mean (possible with tiny ranges) is floored at
    zero before the square root.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    _check_positive(open_, high, low, close)
    term = 0.5 * np.log(high / low) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(close / open_) ** 2
    var = term.rolling(window, min_periods=window).mean().clip(lower=0.0)
    return (np.sqrt(var) * math.sqrt(bars_per_year)).rename("garman_klass_vol")


def rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Drawdown from the trailing ``window``-bar high: close / max - 1 (<= 0)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    _check_positive(close)
    peak = close.rolling(window, min_periods=window).max()
    return (close / peak - 1.0).rename("market_drawdown")
```

### `indicators/trend.py`

```python
"""Trend indicators: EMA and Wilder ADX.

ADX convention (Wilder, 1978)
-----------------------------
* +DM_t = H_t - H_{t-1} if it exceeds L_{t-1} - L_t and is > 0, else 0.
* -DM_t = L_{t-1} - L_t if it exceeds H_t - H_{t-1} and is > 0, else 0.
* TR, +DM and -DM are Wilder-smoothed over ``period`` (seed = SMA).
* +DI = 100 * sm(+DM) / sm(TR), -DI likewise.
* DX = 100 * |+DI - -DI| / (+DI + -DI) (0 when both are 0).
* ADX = Wilder smoothing of DX over ``period``.

Warm-up: the first ADX value appears at bar ``2 * period - 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.volatility import true_range, wilder_smooth


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2/(span+1)), NaN until ``span`` bars."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return series.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI and -DI.

    Returns:
        DataFrame with columns ``adx``, ``plus_di``, ``minus_di``.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    tr = true_range(high, low, close)
    tr.iloc[0] = np.nan
    s_tr = wilder_smooth(tr, period)
    s_p = wilder_smooth(plus_dm, period)
    s_m = wilder_smooth(minus_dm, period)
    safe_tr = s_tr.where(s_tr > 0)
    plus_di = 100.0 * s_p / safe_tr
    minus_di = 100.0 * s_m / safe_tr
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum > 0)
    dx = dx.where(~((di_sum == 0) & s_tr.notna()), 0.0)
    out = pd.DataFrame(
        {"adx": wilder_smooth(dx, period), "plus_di": plus_di, "minus_di": minus_di}, index=high.index
    )
    return out
```

### `indicators/liquidity.py`

```python
"""Liquidity indicators: sweeps, absorption and depth imbalance.

Sweep definition (causal, same-bar confirmation)
-------------------------------------------------
Let ``prior_low_t = min(low_{t-L}, ..., low_{t-1})`` (the trailing window
*excluding* bar t) and ``prior_high_t`` analogously.

* Bullish sweep at t: ``low_t < prior_low_t`` and ``close_t > prior_low_t``.
  Price traded through the prior low and closed back above it.
* Bearish sweep at t: ``high_t > prior_high_t`` and ``close_t < prior_high_t``.

The reversal is confirmed by bar t's close, which is known when the decision
is made after bar t closes, so no future bar is used. A bar that sweeps both
sides is flagged as both (the agent treats that as ambiguous).

A sweep is a price pattern; it does not prove institutional participation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_sweeps(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.DataFrame:
    """Flag bullish/bearish liquidity sweeps using prior-window levels only.

    Returns:
        DataFrame with bool columns ``bullish_sweep``, ``bearish_sweep`` and
        float columns ``prior_low``, ``prior_high`` (NaN during warm-up; sweeps
        are False there).
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    prior_low = low.shift(1).rolling(lookback, min_periods=lookback).min()
    prior_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
    bull = (low < prior_low) & (close > prior_low)
    bear = (high > prior_high) & (close < prior_high)
    return pd.DataFrame(
        {
            "bullish_sweep": bull.fillna(False).astype(bool),
            "bearish_sweep": bear.fillna(False).astype(bool),
            "prior_low": prior_low,
            "prior_high": prior_high,
        },
        index=close.index,
    )


def absorption_ratio(
    volume: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, min_range_ratio: float = 0.1
) -> pd.Series:
    """Relative volume divided by relative price range.

    Definition at bar t (baseline = mean of the *previous* ``window`` bars)::

        rel_vol   = V_t / mean(V_{t-window..t-1})
        range_t   = (H_t - L_t) / C_t
        rel_range = range_t / mean(range_{t-window..t-1})
        AR_t      = rel_vol / max(rel_range, min_range_ratio)

    AR > 1 means more volume than usual was needed per unit of price movement:
    consistent with passive orders absorbing aggressive flow. Without genuine
    aggressor-side and order-book data this cannot distinguish absorption from
    simple two-sided churn; the direction must come from CVD/depth.
    ``min_range_ratio`` caps the ratio when a bar has (near) zero range.
    NaN when the baseline volume or range is zero or history is insufficient.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_range_ratio <= 0:
        raise ValueError("min_range_ratio must be > 0")
    v = volume.astype(float)
    rng = (high - low) / close
    base_v = v.shift(1).rolling(window, min_periods=window).mean()
    base_r = rng.shift(1).rolling(window, min_periods=window).mean()
    rel_v = v / base_v.where(base_v > 0)
    rel_r = rng / base_r.where(base_r > 0)
    return (rel_v / np.maximum(rel_r, min_range_ratio)).rename("absorption_ratio")


def depth_imbalance(bid_depth: pd.Series | float, ask_depth: pd.Series | float) -> pd.Series | float:
    """Resting-depth imbalance (bid - ask) / (bid + ask) in [-1, 1].

    Positive = more resting bids than asks near the touch. NaN if both depths
    are zero or either is missing. Accepts scalars or Series.
    """
    b = bid_depth if isinstance(bid_depth, pd.Series) else np.float64(bid_depth)
    a = ask_depth if isinstance(ask_depth, pd.Series) else np.float64(ask_depth)
    if np.any(np.asarray(b) < 0) or np.any(np.asarray(a) < 0):
        raise ValueError("depth must be non-negative")
    total = b + a
    if isinstance(total, pd.Series):
        return ((b - a) / total.where(total > 0)).rename("depth_imbalance")
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float((b - a) / total)
```

### `indicators/sentiment.py`

```python
"""Sentiment indicators.

Every component is mapped to [-1, 1] where +1 is bullish:

* ``normalize_score``: linear map of a raw range [lo, hi] onto [-1, 1]
  (midpoint -> 0), clipped.
* ``funding_contrarian``: perpetual funding is treated contrarian. Inside
  +-threshold the term is 0; beyond it, the term grows linearly to -sign(f)
  at ``saturation``. Very positive funding (crowded longs) -> bearish term.
* ``composite_sentiment``: weighted average over the *available* components
  with weights renormalized; a missing component is excluded, never assumed
  neutral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _is_num(x: float | None) -> bool:
    """True for finite floats."""
    return x is not None and not isinstance(x, bool) and math.isfinite(float(x))


def normalize_score(value: float | None, lo: float, hi: float) -> float:
    """Map ``value`` from [lo, hi] to [-1, 1]; NaN if missing.

    Raises:
        ValueError: If ``lo >= hi``.
    """
    if lo >= hi:
        raise ValueError("lo must be < hi")
    if not _is_num(value):
        return float("nan")
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return max(-1.0, min(1.0, (float(value) - mid) / half))


def funding_contrarian(funding: float | None, threshold: float, saturation: float) -> float:
    """Contrarian funding term in [-1, 1]; NaN if funding is missing.

    term = -sign(f) * min(1, (|f| - threshold) / (saturation - threshold)) when
    |f| > threshold, else 0.
    """
    if threshold < 0 or saturation <= threshold:
        raise ValueError("require 0 <= threshold < saturation")
    if not _is_num(funding):
        return float("nan")
    f = float(funding)
    excess = abs(f) - threshold
    if excess <= 0:
        return 0.0
    mag = min(1.0, excess / (saturation - threshold))
    return -math.copysign(mag, f)


@dataclass(frozen=True)
class CompositeSentiment:
    """Result of :func:`composite_sentiment`."""

    score: float
    available_components: int
    weight_coverage: float


def composite_sentiment(
    social: float | None,
    news: float | None,
    funding_term: float | None,
    w_social: float,
    w_news: float,
    w_funding: float,
) -> CompositeSentiment:
    """Weighted average of the available normalized components.

    Args:
        social, news, funding_term: Components already in [-1, 1] (NaN/None = missing).
        w_social, w_news, w_funding: Non-negative component weights.

    Returns:
        ``CompositeSentiment`` with score (NaN if no component is available),
        count of available components and the share of total weight they cover.
    """
    weights = (w_social, w_news, w_funding)
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    num = 0.0
    den = 0.0
    n = 0
    for comp, w in zip((social, news, funding_term), weights):
        if _is_num(comp):
            num += w * float(comp)  # type: ignore[arg-type]
            den += w
            n += 1
    if den <= 0:
        return CompositeSentiment(float("nan"), n, 0.0)
    return CompositeSentiment(max(-1.0, min(1.0, num / den)), n, den / sum(weights))
```

### `data/__init__.py`

```python

```

### `data/feed.py`

```python
"""Market-data ingestion: deterministic synthetic feed, validation and live stub.

Synthetic schema (every column labelled ``FeatureStatus.SYNTHETIC``)
-------------------------------------------------------------------
=================  ==========================================================
column             meaning / units
=================  ==========================================================
open/high/low/close  price in quote currency (> 0, OHLC-consistent)
volume             base-asset volume per bar (>= 0)
bid_volume         buyer-initiated volume  = volume * (1 + trade_side) / 2
ask_volume         seller-initiated volume = volume * (1 - trade_side) / 2
trade_side         net aggressor fraction in [-1, 1]
bid_depth          resting bid size near the touch (base units)
ask_depth          resting ask size near the touch (base units)
social_sentiment   social index on [0, 100] (50 = neutral)
news_sentiment     news tone on [-1, 1]
funding_rate       perpetual funding per 8h period (0.0001 = 1 bp)
exchange_netflow   optional on-chain: net coins flowing *into* exchanges
stablecoin_supply_change  optional on-chain: fractional change in supply
=================  ==========================================================

The generator stitches together segments of five regimes (trend_up,
trend_down, range, volatile, noisy) with seeded random lengths. Order-flow,
depth, sentiment and funding are generated with a configurable, deliberately
*weak* dependence on the hidden regime drift (``flow_signal_strength``), plus
noise; social sentiment and funding react to *past* returns. These are toy
dynamics for exercising the pipeline - not a model of any real market.

Timestamps are UTC. The requested range is ``[start, end)`` (end exclusive);
``warmup_bars`` extra bars are generated *before* ``start`` so indicators can
warm up. Those bars are never traded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_PANDAS_FREQ, FrameworkConfig
from core.models import FeatureStatus

logger = logging.getLogger(__name__)

OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")
MICROSTRUCTURE: tuple[str, ...] = ("bid_volume", "ask_volume", "trade_side", "bid_depth", "ask_depth")
SENTIMENT: tuple[str, ...] = ("social_sentiment", "news_sentiment", "funding_rate")
ONCHAIN: tuple[str, ...] = ("exchange_netflow", "stablecoin_supply_change")
OPTIONS: tuple[str, ...] = ("put_call_skew",)


class DataValidationError(ValueError):
    """Raised when input bars violate the data contract."""


class LiveFeedUnavailableError(RuntimeError):
    """Raised by the live-feed stub: no real provider is connected."""


@dataclass
class MarketData:
    """Validated bar data plus per-column provenance.

    ``first_trade_index`` is the position of the first bar at or after the
    requested start; earlier rows are warm-up history only.
    """

    symbol: str
    bars: pd.DataFrame
    availability: dict[str, FeatureStatus]
    source: str
    first_trade_index: int = 0
    meta: dict[str, str] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Interface for any data source (synthetic, CSV, exchange REST/WebSocket)."""

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Return validated bars for ``[start, end)`` plus warm-up history."""
        ...


def parse_utc_date(text: str) -> datetime:
    """Parse ``YYYY-MM-DD`` (or ISO datetime) as a UTC timestamp.

    Naive inputs are interpreted as UTC; aware inputs are converted to UTC.
    """
    ts = pd.Timestamp(text)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def validate_bars(bars: pd.DataFrame, min_bars: int = 1, required: tuple[str, ...] = OHLCV) -> None:
    """Validate a bar DataFrame against the data contract.

    Checks: tz-aware DatetimeIndex, strictly increasing, no duplicates,
    required columns present and non-missing, positive prices, OHLC
    consistency, non-negative volumes, and minimum length.

    Raises:
        DataValidationError: With the first violated rule.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise DataValidationError("bars must be indexed by a DatetimeIndex")
    if bars.index.tz is None:
        raise DataValidationError("bar timestamps must be timezone-aware (UTC)")
    if bars.index.has_duplicates:
        dups = bars.index[bars.index.duplicated()].unique()[:3]
        raise DataValidationError(f"duplicate bar timestamps: {list(map(str, dups))}")
    if not bars.index.is_monotonic_increasing:
        raise DataValidationError("bar timestamps must be strictly increasing")
    missing_cols = [c for c in required if c not in bars.columns]
    if missing_cols:
        raise DataValidationError(f"missing required columns: {missing_cols}")
    if len(bars) < min_bars:
        raise DataValidationError(f"insufficient history: {len(bars)} bars < required {min_bars}")
    na = bars[list(required)].isna().sum()
    if na.any():
        raise DataValidationError(f"missing values in required columns: {na[na > 0].to_dict()}")
    prices = bars[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or not np.isfinite(prices.to_numpy()).all():
        raise DataValidationError("prices must be finite and strictly positive")
    if (bars["high"] < bars[["open", "close"]].max(axis=1) - 1e-9).any():
        raise DataValidationError("high below max(open, close)")
    if (bars["low"] > bars[["open", "close"]].min(axis=1) + 1e-9).any():
        raise DataValidationError("low above min(open, close)")
    if (bars["volume"] < 0).any():
        raise DataValidationError("volume must be non-negative")
    for col in ("bid_volume", "ask_volume", "bid_depth", "ask_depth"):
        if col in bars.columns and (bars[col].dropna() < 0).any():
            raise DataValidationError(f"{col} must be non-negative")
    if "trade_side" in bars.columns and (bars["trade_side"].dropna().abs() > 1 + 1e-9).any():
        raise DataValidationError("trade_side must lie in [-1, 1]")


class SyntheticFeed:
    """Deterministic multi-regime synthetic feed (labelled SYNTHETIC)."""

    REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range", "volatile", "noisy")

    def __init__(self, config: FrameworkConfig, seed: int | None = None) -> None:
        """Create a feed; ``seed`` overrides ``config.app.seed``."""
        self.config = config
        self.seed = config.app.seed if seed is None else seed

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Generate bars for ``[start, end)`` plus warm-up history."""
        if start.tzinfo is None or end.tzinfo is None:
            raise DataValidationError("start and end must be timezone-aware")
        if end <= start:
            raise DataValidationError(f"end ({end}) must be after start ({start})")
        freq = TIMEFRAME_PANDAS_FREQ[self.config.data.timeframe]
        offset = pd.tseries.frequencies.to_offset(freq)
        warm = self.config.data.warmup_bars
        first = pd.Timestamp(start) - warm * offset
        index = pd.date_range(first, pd.Timestamp(end), freq=freq, inclusive="left", tz=timezone.utc)
        n_trade = int((index >= pd.Timestamp(start)).sum())
        if n_trade < 2:
            raise DataValidationError("requested range produces fewer than 2 tradable bars")
        bars, regimes = self._generate(len(index))
        bars.index = index
        bars["hidden_regime"] = regimes  # ground truth for diagnostics only; agents never read it
        validate_bars(bars, min_bars=self.config.data.min_history_bars)
        availability = {c: FeatureStatus.SYNTHETIC for c in OHLCV + MICROSTRUCTURE + SENTIMENT}
        for c in ONCHAIN:
            availability[c] = (
                FeatureStatus.SYNTHETIC if self.config.synthetic.generate_optional else FeatureStatus.UNAVAILABLE
            )
        for c in OPTIONS:
            availability[c] = FeatureStatus.UNAVAILABLE
        logger.info(
            "synthetic feed: symbol=%s seed=%d bars=%d (warm-up %d, tradable %d) %s..%s",
            symbol, self.seed, len(bars), len(bars) - n_trade, n_trade, index[0], index[-1],
        )
        return MarketData(
            symbol=symbol,
            bars=bars,
            availability=availability,
            source=f"synthetic(seed={self.seed})",
            first_trade_index=len(bars) - n_trade,
            meta={"timeframe": self.config.data.timeframe, "seed": str(self.seed)},
        )

    def _generate(self, n: int) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate ``n`` bars of synthetic data and the hidden regime labels."""
        sc = self.config.synthetic
        rng = np.random.default_rng(self.seed)
        bpy = self.config.bars_per_year
        sigma0 = sc.base_vol_annual / np.sqrt(bpy)

        regimes = np.empty(n, dtype=object)
        pos, prev = 0, None
        while pos < n:
            choices = [r for r in self.REGIMES if r != prev]
            reg = choices[int(rng.integers(len(choices)))]
            length = int(rng.integers(sc.min_segment_bars, sc.max_segment_bars + 1))
            regimes[pos : pos + length] = reg
            pos += length
            prev = reg
        regimes = regimes[:n]

        sig_mult = {"trend_up": 1.0, "trend_down": 1.0, "range": 0.7, "volatile": 2.6, "noisy": 1.5}
        drift_k = {"trend_up": 0.15, "trend_down": -0.15, "range": 0.0, "volatile": 0.0, "noisy": 0.0}
        sigma = np.array([sigma0 * sig_mult[r] for r in regimes])
        drift_sign = np.array([np.sign(drift_k[r]) for r in regimes])

        logp = np.empty(n)
        r = np.empty(n)
        lp = np.log(sc.start_price)
        anchor = lp
        for i in range(n):
            reg = regimes[i]
            if i > 0 and regimes[i] != regimes[i - 1]:
                anchor = lp
            eps = rng.standard_normal()
            if reg == "volatile" and rng.random() < 0.03:
                eps += rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)  # jump
            ret = drift_k[reg] * sigma0 + sigma[i] * eps
            if reg == "range":
                ret += -0.05 * (lp - anchor)
            lp += ret
            logp[i] = lp
            r[i] = ret
        close = np.exp(logp)
        prev_close = np.concatenate([[sc.start_price], close[:-1]])
        open_ = prev_close * np.exp(0.1 * sigma * rng.standard_normal(n))
        hi_ext = np.abs(rng.standard_normal(n)) * 0.6 * sigma
        lo_ext = np.abs(rng.standard_normal(n)) * 0.6 * sigma
        high = np.maximum(open_, close) * np.exp(hi_ext)
        low = np.minimum(open_, close) * np.exp(-lo_ext)

        vol_mult = np.where(regimes == "volatile", 1.8, np.where(regimes == "range", 0.8, 1.0))
        volume = sc.base_volume * vol_mult * np.exp(0.3 * rng.standard_normal(n)) * (1 + 0.3 * np.abs(r) / sigma)

        k = sc.flow_signal_strength
        side = np.clip(
            0.5 * np.tanh(r / sigma) + k * drift_sign + 0.25 * rng.standard_normal(n), -0.95, 0.95
        )
        bid_volume = volume * (1 + side) / 2
        ask_volume = volume * (1 - side) / 2
        depth_base = sc.base_volume * 2.0
        tilt = k * drift_sign + 0.15 * rng.standard_normal(n)
        bid_depth = depth_base * np.exp(0.25 * rng.standard_normal(n)) * np.clip(1 + tilt, 0.2, None)
        ask_depth = depth_base * np.exp(0.25 * rng.standard_normal(n)) * np.clip(1 - tilt, 0.2, None)

        past_ret = pd.Series(logp).diff(20).shift(1).fillna(0.0).to_numpy()  # strictly past
        social = np.empty(n)
        news = np.empty(n)
        funding = np.empty(n)
        s, nw, f = 50.0, 0.0, 0.0001
        for i in range(n):
            s = 0.92 * s + 0.08 * (50 + 250 * past_ret[i]) + 2.5 * rng.standard_normal()
            s = float(np.clip(s, 0, 100))
            nw = 0.9 * nw + 0.1 * (0.6 * k * 4 * drift_sign[i]) + 0.12 * rng.standard_normal()
            nw = float(np.clip(nw, -1, 1))
            f = 0.9 * f + 0.1 * (0.0001 + 0.0015 * np.tanh(8 * past_ret[i])) + 0.00008 * rng.standard_normal()
            social[i], news[i], funding[i] = s, nw, f
        netflow = -k * 2 * drift_sign * sc.base_volume * 0.1 + sc.base_volume * 0.1 * rng.standard_normal(n)
        stable = 0.0005 * k * 2 * drift_sign + 0.001 * rng.standard_normal(n)

        df = pd.DataFrame(
            {
                "open": open_, "high": high, "low": low, "close": close, "volume": volume,
                "bid_volume": bid_volume, "ask_volume": ask_volume, "trade_side": side,
                "bid_depth": bid_depth, "ask_depth": ask_depth,
                "social_sentiment": social, "news_sentiment": news, "funding_rate": funding,
            }
        )
        if sc.generate_optional:
            df["exchange_netflow"] = netflow
            df["stablecoin_supply_change"] = stable
        return df, regimes


class LiveFeedStub:
    """Placeholder for a real market-data provider. **Not a working connection.**

    Implement ``get_bars`` against an exchange/vendor API, label real columns
    ``FeatureStatus.OBSERVED`` and missing ones ``UNAVAILABLE`` (never fill
    them with synthetic values), then call :func:`validate_bars`.
    """

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> MarketData:
        """Always raises: no provider is connected and synthetic fallback is forbidden."""
        raise LiveFeedUnavailableError(
            f"live feed for {symbol} is a stub: no market-data provider is connected. "
            "Implement data.feed.LiveFeedStub.get_bars (or a new MarketDataProvider) "
            "or run with --synthetic. The framework never falls back to synthetic data silently."
        )
```

### `agents/__init__.py`

```python
"""Specialist agents, the boss orchestrator and agent factories."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from agents.liquidity_engine import LiquidityEngineAgent
from agents.market_intel import MarketIntelAgent
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
        SentimentEngineAgent(w["sentiment_engine"], a.sentiment_engine),
        RegimeDetectorAgent(w["regime_detector"], a.regime_detector),
    ]
    if config.features.enable_optional_agent and config.features.optional_agent_type == "on_chain":
        agents.append(OnChainAgent(w["on_chain"], a.on_chain))
    return [ag for ag in agents if ag.name not in exclude]


__all__ = [
    "BaseAgent",
    "LiquidityEngineAgent",
    "MarketIntelAgent",
    "OnChainAgent",
    "RegimeDetectorAgent",
    "RiskEngineAgent",
    "SentimentEngineAgent",
    "build_agents",
]
```

### `agents/base_agent.py`

```python
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
```

### `agents/market_intel.py`

```python
"""MarketIntelAgent: institutional-style momentum from order flow and price/volume."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import MarketIntelParams
from core.models import AgentSignal, MarketState, SignalType


class MarketIntelAgent(BaseAgent):
    """Combines OFI, normalized CVD change and volume-confirmed momentum.

    score = (w_ofi*OFI + w_cvd*CVD_norm + w_mom*InstMom) / (w_ofi + w_cvd + w_mom)

    Each input lies in [-1, 1], so ``score`` does too. BUY if score >
    ``signal_threshold``, SELL if score < -threshold, else HOLD.
    confidence = |score| (the agent's evidence strength, not a probability).

    The rolling price/volume correlation is reported for context only.
    Note: with synthetic data CVD_norm is close to a longer-window OFI, so the
    two inputs are correlated; their weights should be read jointly.
    """

    name = "market_intel"
    essential_features = ("ofi", "cvd_norm", "inst_momentum")

    def __init__(self, weight: float, params: MarketIntelParams) -> None:
        """Create the agent with weight and scoring parameters."""
        super().__init__(weight)
        self.params = params

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Score order-flow momentum at ``market_state``."""
        missing = self.check_inputs(market_state)
        if missing:
            return self.unavailable(market_state, missing, "Order-flow momentum cannot be assessed.")
        p = self.params
        ofi = market_state.get("ofi")
        cvdn = market_state.get("cvd_norm")
        mom = market_state.get("inst_momentum")
        pv = market_state.value("pv_corr")
        wsum = p.w_ofi + p.w_cvd + p.w_momentum
        score = (p.w_ofi * ofi + p.w_cvd * cvdn + p.w_momentum * mom) / wsum
        score = max(-1.0, min(1.0, score))
        conf = abs(score)
        proxy = market_state.value("cvd_is_proxy") == 1.0
        src = self.provenance(market_state, ("bid_volume", "ask_volume", "trade_side"))
        pressure = "buying" if ofi > 0 else "selling" if ofi < 0 else "balanced"
        mom_word = "positive" if mom > 0 else "negative" if mom < 0 else "flat"
        if score > p.signal_threshold:
            signal = SignalType.BUY
        elif score < -p.signal_threshold:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        pv_txt = f", price/volume corr {pv:+.2f}" if pv is not None and pv == pv else ""
        summary = f"OFI {ofi:+.2f} and {mom_word} institutional-style momentum ({mom:+.2f}) indicate {pressure} pressure"
        if signal == SignalType.HOLD:
            reason = (
                f"HOLD: composite flow score {score:+.3f} within +/-{p.signal_threshold:.2f}; "
                f"OFI {ofi:+.3f} ({pressure} pressure), CVD change {cvdn:+.3f}{' (tick-rule proxy)' if proxy else ''}, "
                f"institutional-style momentum {mom:+.3f}{pv_txt} do not agree strongly enough"
            )
        else:
            reason = (
                f"{signal.value}: {pressure} pressure dominates - OFI {ofi:+.3f}, CVD change "
                f"{cvdn:+.3f}{' (tick-rule proxy)' if proxy else ''}, institutional-style momentum {mom:+.3f}"
                f"{pv_txt}; composite {score:+.3f} beyond +/-{p.signal_threshold:.2f}"
            )
        reason += f" [inputs: {src}]"
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=conf,
            reason=reason,
            features={
                "ofi": ofi, "cvd_norm": cvdn, "inst_momentum": mom, "pv_corr": pv,
                "score": score, "cvd_is_proxy": proxy, "input_provenance": src,
            },
            timestamp=market_state.timestamp,
            summary=summary,
        )
```

### `agents/risk_engine.py`

```python
"""RiskEngineAgent: market-level risk score and hard veto input.

This agent never votes directionally (always HOLD). Inside the boss its weight
therefore enters the denominator of the directional score as a deliberate
*dampener*, and its ``risk_score`` drives the hard veto. Portfolio drawdown
and the kill switch are handled separately in ``core.portfolio_risk``.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import RiskEngineParams
from core.models import AgentSignal, MarketState, SignalType


def _scale(x: float, lo: float, hi: float) -> float:
    """Linear map of x from [lo, hi] onto [0, 1], clipped."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


class RiskEngineAgent(BaseAgent):
    """Composite market risk in [0, 1].

    Components (each clipped to [0, 1]):
      * vol_score  = (RV_annual - rv_low) / (rv_high - rv_low)
      * atr_score  = (ATR% - atr_pct_low) / (atr_pct_high - atr_pct_low)
      * dd_score   = |rolling market drawdown| / drawdown_full
    risk_score = weighted mean with (w_vol, w_atr, w_drawdown).
    confidence = 1 - risk_score (how comfortable the agent is).
    """

    name = "risk_engine"
    essential_features = ("realized_vol", "atr_pct", "market_drawdown")

    def __init__(self, weight: float, params: RiskEngineParams, veto_threshold: float) -> None:
        """Create the agent with normalization params and the boss veto threshold."""
        super().__init__(weight)
        self.params = params
        self.veto_threshold = veto_threshold

    def risk_components(self, rv: float, atr_pct: float, drawdown: float) -> dict[str, float]:
        """Return the individual component scores and the composite risk score."""
        p = self.params
        vol_s = _scale(rv, p.rv_low, p.rv_high)
        atr_s = _scale(atr_pct, p.atr_pct_low, p.atr_pct_high)
        dd_s = max(0.0, min(1.0, abs(min(drawdown, 0.0)) / p.drawdown_full))
        wsum = p.w_vol + p.w_atr + p.w_drawdown
        score = (p.w_vol * vol_s + p.w_atr * atr_s + p.w_drawdown * dd_s) / wsum
        return {"vol_score": vol_s, "atr_score": atr_s, "drawdown_score": dd_s, "risk_score": score}

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Compute the market risk score; always HOLD, with veto flag in features."""
        missing = self.check_inputs(market_state)
        if missing:
            return self.unavailable(
                market_state, missing, "Risk cannot be scored; the boss treats missing risk as a veto (fail-safe)."
            )
        rv = market_state.get("realized_vol")
        atrp = market_state.get("atr_pct")
        dd = market_state.get("market_drawdown")
        comp = self.risk_components(rv, atrp, dd)
        rs = comp["risk_score"]
        veto = rs > self.veto_threshold
        level = "extreme" if veto else "elevated" if rs > 0.5 else "moderate" if rs > 0.25 else "low"
        reason = (
            f"HOLD (risk agent does not vote direction): {level} market risk, risk_score {rs:.2f} "
            f"({'ABOVE' if veto else 'below'} veto {self.veto_threshold:.2f}); components: realized vol "
            f"{rv:.0%} -> {comp['vol_score']:.2f}, ATR {atrp:.2%} -> {comp['atr_score']:.2f}, "
            f"market drawdown {dd:.1%} -> {comp['drawdown_score']:.2f}"
        )
        return AgentSignal(
            agent_name=self.name,
            signal=SignalType.HOLD,
            confidence=1.0 - rs,
            reason=reason,
            features={
                "realized_vol": rv, "atr_pct": atrp, "market_drawdown": dd, **comp, "risk_veto": veto,
                "parkinson_vol": market_state.value("parkinson_vol"),
                "garman_klass_vol": market_state.value("garman_klass_vol"),
            },
            timestamp=market_state.timestamp,
            summary=f"risk_score {rs:.2f} ({level})",
        )
```

### `agents/liquidity_engine.py`

```python
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
        ctx = (
            f"CVD(3) {'n/a' if cvd_s is None else f'{cvd_s:+.2f}'}, depth imbalance "
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
```

### `agents/sentiment_engine.py`

```python
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
```

### `agents/regime_detector.py`

```python
"""RegimeDetectorAgent: classify the market regime from EMA, ADX and volatility."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.config import RegimeParams
from core.models import AgentSignal, MarketState, Regime, SignalType


class RegimeDetectorAgent(BaseAgent):
    """Rule-based regime classifier (checked in this order).

    1. realized vol (annualized) >= ``volatile_rv``           -> volatile
    2. ADX >= adx_trend and EMA_fast > EMA_slow               -> trend_up
    3. ADX >= adx_trend and EMA_fast < EMA_slow               -> trend_down
    4. otherwise                                              -> range

    Trading vote (if ``directional_vote``): BUY in trend_up, SELL in
    trend_down, HOLD otherwise, with confidence
    ``min(1, (ADX - adx_trend) / (adx_strong - adx_trend))``, floored at 0.1 so
    a confirmed trend never votes with zero confidence.
    During warm-up (EMA_slow/ADX not yet defined) the regime is ``unknown``
    and the agent is unavailable.
    """

    name = "regime_detector"
    essential_features = ("ema_fast", "ema_slow", "adx", "realized_vol")

    def __init__(self, weight: float, params: RegimeParams) -> None:
        """Create the agent with weight and threshold parameters."""
        super().__init__(weight)
        self.params = params

    def classify(self, ema_fast: float, ema_slow: float, adx_value: float, rv: float) -> Regime:
        """Return the regime for the given readings."""
        p = self.params
        if rv >= p.volatile_rv:
            return Regime.VOLATILE
        if adx_value >= p.adx_trend and ema_fast > ema_slow:
            return Regime.TREND_UP
        if adx_value >= p.adx_trend and ema_fast < ema_slow:
            return Regime.TREND_DOWN
        return Regime.RANGE

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Classify the regime at ``market_state``."""
        missing = self.check_inputs(market_state)
        if missing:
            sig = self.unavailable(market_state, missing, "Regime is 'unknown' during indicator warm-up.")
            sig.features["regime"] = Regime.UNKNOWN.value
            return sig
        p = self.params
        ef, es = market_state.get("ema_fast"), market_state.get("ema_slow")
        a, rv = market_state.get("adx"), market_state.get("realized_vol")
        regime = self.classify(ef, es, a, rv)
        rel = ">" if ef > es else "<" if ef < es else "="
        gap = ef / es - 1.0
        readings = f"EMA_fast {ef:,.2f} {rel} EMA_slow {es:,.2f} ({gap:+.2%}), ADX {a:.1f} (trend >= {p.adx_trend:.0f}), realized vol {rv:.0%} (volatile >= {p.volatile_rv:.0%})"
        why = {
            Regime.VOLATILE: "volatility above the volatile threshold dominates",
            Regime.TREND_UP: "ADX confirms a trend and the fast EMA is above the slow EMA",
            Regime.TREND_DOWN: "ADX confirms a trend and the fast EMA is below the slow EMA",
            Regime.RANGE: "ADX below the trend threshold (no persistent direction)",
        }[regime]
        conf = 0.0
        sig = SignalType.HOLD
        if p.directional_vote and regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            sig = SignalType.BUY if regime == Regime.TREND_UP else SignalType.SELL
            conf = max(0.1, min(1.0, (a - p.adx_trend) / (p.adx_strong - p.adx_trend)))
        prefix = f"{sig.value}: " if sig != SignalType.HOLD else "HOLD (no directional vote in this regime): "
        reason = f"{prefix}regime '{regime.value}' because {why}; {readings}"
        return AgentSignal(
            agent_name=self.name,
            signal=sig,
            confidence=conf,
            reason=reason,
            features={"regime": regime.value, "ema_fast": ef, "ema_slow": es, "adx": a, "realized_vol": rv,
                      "plus_di": market_state.value("plus_di"), "minus_di": market_state.value("minus_di")},
            timestamp=market_state.timestamp,
            summary=f"regime {regime.value} (ADX {a:.0f}, EMA gap {gap:+.1%})",
        )
```

### `agents/on_chain.py`

```python
"""Optional OnChainAgent (feature flag ``features.enable_optional_agent``)."""

from __future__ import annotations

import math

from agents.base_agent import BaseAgent
from core.config import OnChainParams
from core.models import AgentSignal, MarketState, SignalType


class OnChainAgent(BaseAgent):
    """Exchange netflow and stablecoin supply proxy.

    * netflow_z: z-score of net coins flowing *into* exchanges (trailing
      window). Inflows are treated as bearish (potential sell supply):
      term_netflow = -tanh(netflow_z).
    * stablecoin_change: fractional change in stablecoin supply, treated as
      bullish dry powder: term_stable = tanh(change / 0.001).
    score = (w_netflow*term_netflow + w_stable*term_stable) / (w_netflow + w_stable)
    BUY/SELL when |netflow_z| >= netflow_z_threshold and score agrees in sign.
    In synthetic mode these inputs are SYNTHETIC and say nothing about a real chain.
    """

    name = "on_chain"
    essential_features = ("netflow_z", "stablecoin_change")

    def __init__(self, weight: float, params: OnChainParams) -> None:
        """Create the agent with weight and scoring parameters."""
        super().__init__(weight)
        self.params = params

    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Score on-chain flows at ``market_state``."""
        missing = self.check_inputs(market_state)
        if missing:
            return self.unavailable(market_state, missing, "No on-chain provider connected.")
        p = self.params
        z = market_state.get("netflow_z")
        sc = market_state.get("stablecoin_change")
        t_nf = -math.tanh(z)
        t_sc = math.tanh(sc / 0.001)
        score = (p.w_netflow * t_nf + p.w_stablecoin * t_sc) / (p.w_netflow + p.w_stablecoin)
        sig = SignalType.HOLD
        if abs(z) >= p.netflow_z_threshold and math.copysign(1, score) == math.copysign(1, t_nf) and score != 0:
            sig = SignalType.BUY if score > 0 else SignalType.SELL
        src = self.provenance(market_state, ("exchange_netflow", "stablecoin_supply_change"))
        flow = "outflow from" if z < 0 else "inflow to"
        reason = (
            f"{sig.value}: " if sig != SignalType.HOLD else f"HOLD: |netflow z| below {p.netflow_z_threshold:.2f} or terms disagree; "
        ) + f"exchange netflow z {z:+.2f} ({flow} exchanges), stablecoin supply change {sc:+.4%}, score {score:+.2f} [inputs: {src}]"
        return AgentSignal(
            self.name, sig, min(1.0, abs(score)), reason,
            {"netflow_z": z, "stablecoin_change": sc, "score": score, "input_provenance": src},
            market_state.timestamp, True, f"on-chain netflow z {z:+.2f}",
        )
```

### `agents/adversarial_agent.py`

```python
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
```

### `agents/boss_orchestrator.py`

```python
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
```

### `execution/__init__.py`

```python
"""Execution, accounting, evaluation, feedback and walk-forward validation."""
```

### `execution/executor.py`

```python
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
```

### `execution/evaluator.py`

```python
"""Performance metrics, agent scoring/attribution and confidence calibration.

Agent scoring uses **matured outcomes only**: a prediction made at bar t with
horizon h is scored only when ``t + h <= as_of_index`` (the close of bar
t + h has been observed).

Definitions
-----------
* forward return  r_t^h = close[t+h] / close[t] - 1 (close-to-close; used for
  directional evaluation only, not for trade PnL).
* hit: a BUY with r > 0 or a SELL with r < 0. HOLD and unavailable signals
  are excluded from the hit rate; r == 0 counts as a miss.
* confidence-weighted return: mean of c_t * d_t * r_t^h over directional
  matured predictions.
* standalone PnL: hypothetical unit position = agent direction held from
  close t to close t+1, minus (fee+slippage) on every position change; summed
  (non-compounded) over matured bars. Independent of the ensemble.
* incremental contribution (signal-level ablation): for each matured bar,
  (S_full - S_without_agent) * r_t^h using the effective weights actually in
  force at that bar. Reported in basis points; its t-stat uses an
  overlap-adjusted sample size n_eff = n / h. The execution-level ablation
  (full re-run without the agent) is in ``execution.backtester.run_ablation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.config import FrameworkConfig
from core.models import AgentScore, PerformanceMetrics, Trade


@dataclass(frozen=True)
class SignalRecord:
    """One agent's logged output at one bar (enough to replay the vote)."""

    bar_index: int
    agent_name: str
    direction: int
    confidence: float
    available: bool
    signed_input: float
    effective_weight: float
    eligible: bool


@dataclass(frozen=True)
class DecisionRecord:
    """Boss decision summary for calibration and reporting."""

    bar_index: int
    raw_direction: int
    action_direction: int
    strength: float
    directional_score: float


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    exposures: pd.Series,
    total_fees: float,
    total_slippage: float,
    traded_notional: float,
    bars_per_year: int,
    bars_per_day: float,
) -> PerformanceMetrics:
    """Compute net-of-cost metrics from a marked-to-market equity curve.

    Unavailable metrics are returned as ``None`` with an explanation in ``notes``.
    """
    notes: dict[str, str] = {}
    if len(equity) < 2:
        raise ValueError("equity curve needs at least 2 points")
    e0, e1 = float(equity.iloc[0]), float(equity.iloc[-1])
    rets = equity.pct_change().dropna()
    total = e1 / e0 - 1.0
    years = len(rets) / bars_per_year
    cagr: Optional[float]
    if years >= 0.25 and e1 > 0:
        cagr = (e1 / e0) ** (1.0 / years) - 1.0
    else:
        cagr = None
        notes["cagr"] = f"unavailable: window {years:.2f}y < 0.25y"
    sd = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    sharpe: Optional[float]
    if len(rets) < 10 or not np.isfinite(sd) or sd < 1e-15:
        sharpe = None
        notes["sharpe"] = "unavailable: too few returns or zero variance"
    else:
        sharpe = float(rets.mean() / sd * math.sqrt(bars_per_year))
    peak = equity.cummax()
    mdd = float((1.0 - equity / peak).max())
    closed = [t for t in trades if t.is_closed]
    n = len(closed)
    wins = [t.net_pnl for t in closed if t.net_pnl > 0]
    losses = [t.net_pnl for t in closed if t.net_pnl <= 0]
    win_rate = len(wins) / n if n else None
    if n == 0:
        notes["win_rate"] = "unavailable: no closed trades"
    pf: Optional[float]
    if n == 0:
        pf = None
        notes["profit_factor"] = "unavailable: no closed trades"
    elif not losses or sum(losses) == 0:
        pf = None
        notes["profit_factor"] = "unavailable: no losing trades (would be infinite)"
    else:
        pf = sum(wins) / abs(sum(losses))
    hold = [t.bars_held for t in closed if t.bars_held is not None]
    avg_hold = float(np.mean(hold)) if hold else None
    if avg_hold is None:
        notes["avg_holding"] = "unavailable: no closed trades"
    mean_eq = float(equity.mean())
    return PerformanceMetrics(
        initial_equity=e0,
        final_equity=e1,
        total_return=total,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=mdd,
        n_trades=n,
        win_rate=win_rate,
        profit_factor=pf,
        turnover=traded_notional / mean_eq if mean_eq > 0 else 0.0,
        total_fees=total_fees,
        total_slippage=total_slippage,
        avg_holding_bars=avg_hold,
        avg_exposure=float(exposures.abs().mean()) if len(exposures) else 0.0,
        avg_long_exposure=float(exposures.clip(lower=0).mean()) if len(exposures) else 0.0,
        avg_short_exposure=float((-exposures.clip(upper=0)).mean()) if len(exposures) else 0.0,
        n_bars=len(equity),
        notes=notes,
    )


class Evaluator:
    """Scores agents on matured outcomes and suggests bounded weights."""

    def __init__(self, config: FrameworkConfig, base_weights: dict[str, float]) -> None:
        """Create an evaluator using evaluation settings and base weights."""
        self.config = config
        self.ev = config.evaluation
        self.base_weights = dict(base_weights)
        self.cost_rate = (config.execution.fee_bps + config.execution.slippage_bps) / 1e4

    def score_agents(
        self,
        history: Sequence[SignalRecord],
        prices: np.ndarray,
        lookahead: int | None = None,
        as_of_index: int | None = None,
        current_weights: dict[str, float] | None = None,
        min_index: int = 0,
    ) -> dict[str, AgentScore]:
        """Score every agent in ``history`` using outcomes matured by ``as_of_index``.

        Args:
            history: Logged signal records.
            prices: Close prices indexed by bar index (whole feature frame).
            lookahead: Forward horizon h (defaults to config).
            as_of_index: Last bar whose close is known; defaults to the last price.
            current_weights: Weights to report as current (defaults to base).
            min_index: Ignore records before this bar (e.g. training-window start).
        """
        h = lookahead or self.ev.lookahead
        as_of = len(prices) - 1 if as_of_index is None else as_of_index
        current = current_weights or self.base_weights
        recs = [r for r in history if min_index <= r.bar_index <= as_of]
        by_bar: dict[int, list[SignalRecord]] = {}
        for r in recs:
            by_bar.setdefault(r.bar_index, []).append(r)
        agents = sorted({r.agent_name for r in recs})
        out: dict[str, AgentScore] = {}
        for name in agents:
            mine = sorted((r for r in recs if r.agent_name == name), key=lambda r: r.bar_index)
            directional = [r for r in mine if r.available and r.direction != 0]
            matured = [r for r in directional if r.bar_index + h <= as_of]
            hits, cw = [], []
            for r in matured:
                fwd = prices[r.bar_index + h] / prices[r.bar_index] - 1.0
                hits.append(1.0 if r.direction * fwd > 0 else 0.0)
                cw.append(r.confidence * r.direction * fwd)
            pnl = 0.0
            prev_pos = 0
            n_pnl = 0
            for r in mine:
                if r.bar_index + 1 > as_of:
                    break
                pos = r.direction if r.available else 0
                pnl += pos * (prices[r.bar_index + 1] / prices[r.bar_index] - 1.0) - abs(pos - prev_pos) * self.cost_rate
                prev_pos = pos
                n_pnl += 1
            inc_vals = []
            for t, rows in by_bar.items():
                if t + h > as_of:
                    continue
                elig = [x for x in rows if x.eligible]
                w_all = sum(x.effective_weight for x in elig)
                me = [x for x in elig if x.agent_name == name]
                if not me or w_all <= 0:
                    continue
                s_full = sum(x.effective_weight * x.signed_input for x in elig) / w_all
                rest = [x for x in elig if x.agent_name != name]
                w_rest = sum(x.effective_weight for x in rest)
                s_wo = sum(x.effective_weight * x.signed_input for x in rest) / w_rest if w_rest > 0 else 0.0
                fwd = prices[t + h] / prices[t] - 1.0
                inc_vals.append((s_full - s_wo) * fwd)
            inc = float(np.mean(inc_vals)) * 1e4 if inc_vals else None
            tstat = None
            note = ""
            if len(inc_vals) >= 2:
                sd = float(np.std(inc_vals, ddof=1))
                n_eff = max(1.0, len(inc_vals) / h)
                tstat = float(np.mean(inc_vals) / (sd / math.sqrt(n_eff))) if sd > 1e-15 else 0.0
            if len(matured) < self.ev.min_agent_samples:
                note = f"insufficient matured directional samples ({len(matured)} < {self.ev.min_agent_samples})"
            out[name] = AgentScore(
                agent_name=name,
                prediction_count=len(mine),
                directional_count=len(directional),
                matured_count=len(matured),
                hit_rate=float(np.mean(hits)) if hits else None,
                confidence_weighted_return=float(np.mean(cw)) if cw else None,
                standalone_pnl=pnl if n_pnl else None,
                incremental_contribution=inc,
                incremental_tstat=tstat,
                current_weight=current.get(name, self.base_weights.get(name, 1.0)),
                suggested_weight=current.get(name, self.base_weights.get(name, 1.0)),
                note=note,
            )
        return out

    def suggest_weights(self, scores: dict[str, AgentScore]) -> dict[str, float]:
        """Bounded, shrunk, smoothed weight suggestions.

        * Insufficient matured samples or no t-stat -> keep the current weight.
        * |t| < ``tstat_neutral_band`` (statistically unreliable) -> target is
          the base weight (evidence-free drift back toward the prior).
        * Otherwise target = shrink*base + (1-shrink)*current*exp(sens*clip(t, -3, 3)).
        * new = (1 - smoothing)*current + smoothing*target, clipped to
          [min_weight, max_weight].
        """
        ev = self.ev
        out: dict[str, float] = {}
        for name, s in scores.items():
            base = self.base_weights.get(name, s.current_weight)
            cur = s.current_weight
            if s.matured_count < ev.min_agent_samples or s.incremental_tstat is None:
                out[name] = cur
                continue
            t = max(-3.0, min(3.0, s.incremental_tstat))
            if abs(t) < ev.tstat_neutral_band:
                target = base
            else:
                target = ev.shrinkage_to_base * base + (1 - ev.shrinkage_to_base) * cur * math.exp(ev.weight_sensitivity * t)
            new = (1 - ev.weight_smoothing) * cur + ev.weight_smoothing * target
            out[name] = min(ev.max_weight, max(ev.min_weight, new))
        return out


class LogisticCalibrator:
    """Maps ensemble strength to P(direction correct) with a 1-D logistic fit.

    Fit only on training-window matured decisions; evaluate on separate
    out-of-sample decisions with the Brier score.
    """

    def __init__(self) -> None:
        """Create an unfitted calibrator."""
        self.a: float | None = None
        self.b: float | None = None

    @property
    def fitted(self) -> bool:
        """True after a successful fit."""
        return self.a is not None

    def fit(self, x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> bool:
        """Fit P(y=1|x) = sigmoid(a + b x). Returns False if data are insufficient."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 20 or len(np.unique(y)) < 2:
            self.a = self.b = None
            return False

        def nll(p: np.ndarray) -> float:
            z = p[0] + p[1] * x
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + l2 * (p @ p))

        res = minimize(nll, np.zeros(2), method="BFGS")
        if not res.success and not np.isfinite(res.fun):
            self.a = self.b = None
            return False
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (raises if unfitted)."""
        if not self.fitted:
            raise RuntimeError("calibrator not fitted")
        z = self.a + self.b * np.asarray(x, dtype=float)  # type: ignore[operator]
        return 1.0 / (1.0 + np.exp(-z))


def matured_decision_outcomes(
    decisions: Sequence[DecisionRecord], prices: np.ndarray, horizon: int, as_of_index: int, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """(strength, hit) arrays for directional raw votes in [lo, hi) matured by ``as_of_index``."""
    xs, ys = [], []
    for d in decisions:
        if lo <= d.bar_index < hi and d.raw_direction != 0 and d.bar_index + horizon <= as_of_index:
            fwd = prices[d.bar_index + horizon] / prices[d.bar_index] - 1.0
            xs.append(d.strength)
            ys.append(1.0 if d.raw_direction * fwd > 0 else 0.0)
    return np.asarray(xs), np.asarray(ys)


def brier(p: np.ndarray, y: np.ndarray) -> float | None:
    """Mean squared error of probabilities; None if empty."""
    return float(np.mean((p - y) ** 2)) if len(y) else None
```

### `execution/feedback.py`

```python
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
```

### `execution/backtester.py`

```python
"""Event-driven backtester, benchmarks, ablation and sensitivity analysis.

Loop for each bar t in ``[start, end)``:

1. Fill the order decided at t-1's close at bar t's open (strategy and shadow books).
2. Mark equity at t's close; update the portfolio kill switch with it.
3. (training only) feedback loop may update learned weights using outcomes
   matured by t.
4. Build ``MarketState`` for t, run every agent, aggregate with the boss.
5. Convert the decision into a target exposure; queue an order for t+1.

The decision at the final bar of a window cannot be filled inside that
window and is reported as unfilled.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from agents import build_agents
from agents.adversarial_agent import AdversarialAgent, AdversarialReport
from agents.base_agent import BaseAgent
from agents.boss_orchestrator import BossOrchestrator
from core.config import TIMEFRAME_BARS_PER_YEAR, FrameworkConfig
from core.feature_bus import FeatureFrame
from core.models import AgentSignal, BossDecision, MarketState, Order, PerformanceMetrics, SignalType, WeightUpdate
from core.portfolio_risk import KillSwitch, KillSwitchEvent
from execution.evaluator import DecisionRecord, Evaluator, SignalRecord, compute_metrics
from execution.executor import PaperExecutor, Portfolio, SimulatedExecutor
from execution.feedback import FeedbackController

logger = logging.getLogger(__name__)


def bars_per_day(config: FrameworkConfig) -> float:
    """Bars per calendar day for the configured timeframe."""
    return TIMEFRAME_BARS_PER_YEAR[config.data.timeframe] / 365.0


def target_from_decision(
    decision: BossDecision, current_exposure: float, state: MarketState, config: FrameworkConfig, kill_policy: str
) -> tuple[float | None, str]:
    """Translate a boss decision into a target exposure (None = no trade).

    Position-management rules (applied even when the boss returned HOLD):
      * kill switch active: ``flatten`` -> 0, ``hold`` -> keep;
      * risk veto with ``on_risk_veto=reduce``: cap |exposure| at ``veto_exposure_cap``;
      * BUY: target = min(max_position_size, strength * exposure_cap) (optionally
        scaled by target_vol / realized_vol); only increases a long;
      * SELL: closes (``sell_policy=close``) or reduces a long; opens/increases a
        short only when ``allow_short``;
      * HOLD: keep current units.
    """
    s, ex = config.strategy, config.execution
    cur = current_exposure
    if decision.kill_switch_active:
        if kill_policy == "flatten" and abs(cur) > 1e-9:
            return 0.0, "kill switch active: flatten existing position"
        return None, "kill switch active: no new entries"
    if decision.risk_veto and s.on_risk_veto == "reduce" and abs(cur) > s.veto_exposure_cap + 1e-9:
        return math.copysign(s.veto_exposure_cap, cur), f"risk veto: reduce exposure to {s.veto_exposure_cap:.2f}"

    def sized() -> float:
        size = min(ex.max_position_size, decision.confidence * ex.exposure_cap)
        if ex.volatility_aware_sizing and state.is_valid("realized_vol") and state.get("realized_vol") > 0:
            size *= min(1.0, ex.target_volatility / state.get("realized_vol"))
        return min(size, ex.max_exposure)

    if decision.action == SignalType.BUY:
        size = sized()
        if cur < 0:
            return size, "BUY reverses short"
        if size > cur + s.min_rebalance_exposure:
            return size, f"BUY: raise exposure {cur:.2f} -> {size:.2f}"
        return None, "BUY: already at or above target exposure"
    if decision.action == SignalType.SELL:
        if cur > 1e-9:
            if s.sell_policy == "close":
                return (-sized() if s.allow_short else 0.0), "SELL: close long" + (" and reverse" if s.allow_short else "")
            new = cur * (1 - s.sell_reduce_fraction)
            return (new if new > s.min_rebalance_exposure else 0.0), f"SELL: reduce long by {s.sell_reduce_fraction:.0%}"
        if s.allow_short:
            size = -sized()
            if size < cur - s.min_rebalance_exposure:
                return size, f"SELL: short exposure {cur:.2f} -> {size:.2f}"
            return None, "SELL: already at or beyond short target"
        return None, "SELL while flat: long-only mode does not open shorts"
    return None, "HOLD"


@dataclass
class BacktestResult:
    """Everything produced by one backtest run."""

    start_index: int
    end_index: int
    equity: pd.Series
    exposures: pd.Series
    portfolio: Portfolio
    metrics: PerformanceMetrics
    decisions: list[BossDecision]
    signals: list[list[AgentSignal]]
    signal_records: list[SignalRecord]
    decision_records: list[DecisionRecord]
    weight_updates: list[WeightUpdate]
    kill_switch_events: list[KillSwitchEvent]
    adversarial: AdversarialReport | None
    unfilled_orders: list[Order]
    final_weights: dict[str, float]
    rule_counts: dict[str, int] = field(default_factory=dict)


class Backtester:
    """Runs the multi-agent strategy over a slice of a feature frame."""

    def __init__(
        self,
        config: FrameworkConfig,
        features: FeatureFrame,
        exclude_agents: tuple[str, ...] = (),
        agents: list[BaseAgent] | None = None,
        boss: BossOrchestrator | None = None,
        paper: bool = False,
    ) -> None:
        """Create a backtester (``paper=True`` uses the logging PaperExecutor)."""
        self.config = config
        self.ff = features
        self.agents = agents if agents is not None else build_agents(config, exclude=exclude_agents)
        names = [a.name for a in self.agents]
        self.boss = boss or BossOrchestrator(config, names)
        self.paper = paper
        self.evaluator = Evaluator(config, self.boss.base_weights)

    def run(
        self,
        start: int,
        end: int,
        learn: bool = False,
        initial_weights: dict[str, float] | None = None,
        on_decision: Callable[[int, BossDecision, list[AgentSignal]], None] | None = None,
    ) -> BacktestResult:
        """Run bars ``[start, end)``. ``learn=True`` enables the feedback loop."""
        cfg = self.config
        if not 0 <= start < end <= len(self.ff):
            raise ValueError(f"invalid window [{start}, {end}) for {len(self.ff)} bars")
        if end - start < 2:
            raise ValueError("window must contain at least 2 bars")
        ex, rk = cfg.execution, cfg.risk
        self.boss.reset()
        self.boss.learned_weights = dict(self.boss.base_weights)
        if initial_weights:
            self.boss.set_learned_weights(initial_weights)
        portfolio = Portfolio(ex.initial_capital, ex.fee_bps, ex.slippage_bps, allow_short=cfg.strategy.allow_short, max_leverage=ex.max_leverage)
        executor: SimulatedExecutor = PaperExecutor(portfolio) if self.paper else SimulatedExecutor(portfolio)
        ks = KillSwitch(
            rk.max_drawdown, rk.rolling_sharpe_threshold, rk.rolling_sharpe_window, rk.kill_switch_min_history,
            cfg.bars_per_year, rk.kill_switch_recovery, rk.kill_switch_cooldown_bars, rk.kill_switch_position_policy,
        )
        adv = (
            AdversarialAgent(ex.initial_capital, ex.fee_bps, ex.slippage_bps, 1.0, cfg.strategy.prediction_horizon)
            if cfg.features.enable_adversarial else None
        )
        fb = FeedbackController(self.evaluator, self.boss, cfg.evaluation.feedback_interval, start, frozen=not learn)
        opens, closes = self.ff.column("open"), self.ff.column("close")
        idx = self.ff.index
        eq_vals, ex_vals = [], []
        decisions, all_signals = [], []
        srecs: list[SignalRecord] = []
        drecs: list[DecisionRecord] = []
        rule_counts: dict[str, int] = {}

        for i in range(start, end):
            ts = idx[i].to_pydatetime()
            if i > start:
                executor.on_bar_open(i, ts, opens[i])
                if adv:
                    adv.on_open(i, ts, opens[i])
            equity = portfolio.equity(closes[i])
            if equity <= 0:
                raise RuntimeError(f"equity non-positive at bar {i}; check leverage settings")
            eq_vals.append(equity)
            ex_vals.append(portfolio.exposure(closes[i]))
            if adv:
                adv.on_close(closes[i])
            ks_active = ks.update(equity, ts, i)
            fb.maybe_update(i, ts, srecs, closes)

            state = self.ff.state(i)
            signals = [a.analyze(state) for a in self.agents]
            decision = self.boss.decide(signals, kill_switch_active=ks_active, timestamp=ts)
            decisions.append(decision)
            all_signals.append(signals)
            if decision.forced_hold_rule:
                for r in decision.forced_hold_rule.split(";"):
                    rule_counts[r] = rule_counts.get(r, 0) + 1
            for c in decision.breakdown:
                srecs.append(SignalRecord(i, c.agent_name, c.signal.direction, c.confidence, c.available,
                                          c.signed_input, c.effective_weight, c.eligible))
            drecs.append(DecisionRecord(i, decision.raw_action.direction, decision.action.direction,
                                        decision.confidence, decision.directional_score))
            if on_decision:
                on_decision(i, decision, signals)
            if adv:
                adv.observe(i, decision.raw_action.direction)
            target, why = target_from_decision(decision, portfolio.exposure(closes[i]), state, cfg, ks.position_policy)
            if target is not None:
                executor.submit(Order(ts, i, target, f"{why} | {decision.reason}"))
        executor.finalize()

        equity_s = pd.Series(eq_vals, index=idx[start:end], name="equity")
        expo_s = pd.Series(ex_vals, index=idx[start:end], name="exposure")
        metrics = compute_metrics(
            equity_s, portfolio.trades, expo_s, portfolio.total_fees, portfolio.total_slippage,
            portfolio.traded_notional, cfg.bars_per_year, bars_per_day(cfg),
        )
        return BacktestResult(
            start_index=start, end_index=end, equity=equity_s, exposures=expo_s, portfolio=portfolio,
            metrics=metrics, decisions=decisions, signals=all_signals, signal_records=srecs,
            decision_records=drecs, weight_updates=fb.updates, kill_switch_events=ks.events,
            adversarial=adv.report(closes, end - 1) if adv else None, unfilled_orders=executor.unfilled,
            final_weights=dict(self.boss.learned_weights), rule_counts=rule_counts,
        )


def _rule_backtest(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int, rule: Callable[[int, float], float | None]) -> BacktestResult:
    """Run a simple rule (bar index, current exposure) -> target exposure with the same execution model."""
    ex = config.execution
    portfolio = Portfolio(ex.initial_capital, ex.fee_bps, ex.slippage_bps, allow_short=False, max_leverage=ex.max_leverage)
    executor = SimulatedExecutor(portfolio)
    opens, closes, idx = ff.column("open"), ff.column("close"), ff.index
    eq, xp = [], []
    for i in range(start, end):
        ts = idx[i].to_pydatetime()
        if i > start:
            executor.on_bar_open(i, ts, opens[i])
        eq.append(portfolio.equity(closes[i]))
        xp.append(portfolio.exposure(closes[i]))
        tgt = rule(i, portfolio.exposure(closes[i]))
        if tgt is not None:
            executor.submit(Order(ts, i, tgt, "benchmark"))
    executor.finalize()
    es = pd.Series(eq, index=idx[start:end])
    xs = pd.Series(xp, index=idx[start:end])
    m = compute_metrics(es, portfolio.trades, xs, portfolio.total_fees, portfolio.total_slippage,
                        portfolio.traded_notional, config.bars_per_year, bars_per_day(config))
    return BacktestResult(start, end, es, xs, portfolio, m, [], [], [], [], [], [], None, executor.unfilled, {})


def run_benchmarks(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int) -> dict[str, PerformanceMetrics]:
    """No-trade, buy-and-hold and EMA trend-following benchmarks on the same window and costs.

    Buy-and-hold enters at the same first possible fill as the strategy (open
    of ``start + 1``) at ``max_position_size`` and is marked to market at the
    final close (no exit fee), like the strategy.
    """
    size = config.execution.max_position_size
    ema_f, ema_s = ff.column("ema_fast"), ff.column("ema_slow")
    no_trade = _rule_backtest(config, ff, start, end, lambda i, cur: None)
    bh = _rule_backtest(config, ff, start, end, lambda i, cur: size if (i == start and cur == 0) else None)

    def trend(i: int, cur: float) -> float | None:
        if not (np.isfinite(ema_f[i]) and np.isfinite(ema_s[i])):
            return None
        want = size if ema_f[i] > ema_s[i] else 0.0
        if want > 0 and cur < 1e-9:
            return want
        if want == 0 and cur > 1e-9:
            return 0.0
        return None

    tf = _rule_backtest(config, ff, start, end, trend)
    return {"no_trade": no_trade.metrics, "buy_and_hold": bh.metrics, "ema_trend": tf.metrics}


@dataclass(frozen=True)
class AblationRow:
    """Execution-level ablation of one agent."""

    agent_name: str
    total_return_without: float
    sharpe_without: float | None
    delta_return: float
    delta_sharpe: float | None


def run_ablation(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int, full: BacktestResult) -> list[AblationRow]:
    """Re-run the identical backtest with each agent removed (same window, costs, rules).

    delta = full - without; positive delta means the agent added value *in this
    sample*. This is descriptive and in-sample unless run on held-out data.
    """
    rows = []
    for name in [c.agent_name for c in full.decisions[0].breakdown] if full.decisions else []:
        res = Backtester(config, ff, exclude_agents=(name,)).run(start, end)
        m = res.metrics
        ds = None if (m.sharpe is None or full.metrics.sharpe is None) else full.metrics.sharpe - m.sharpe
        rows.append(AblationRow(name, m.total_return, m.sharpe, full.metrics.total_return - m.total_return, ds))
    return rows


@dataclass(frozen=True)
class SensitivityRow:
    """One cell of the threshold sensitivity grid."""

    min_confidence: float
    score_floor: float
    regime_weights: bool
    total_return: float
    sharpe: float | None
    max_drawdown: float
    n_trades: int


def run_sensitivity(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int) -> list[SensitivityRow]:
    """Grid over min_confidence x directional floor x regime weights on/off (in-sample, descriptive)."""
    rows = []
    for mc in (0.25, 0.35, 0.45):
        for fl in (0.10, 0.15, 0.25):
            for rw in (True, False):
                c = config.model_copy(deep=True)
                c.strategy.min_confidence = mc
                c.strategy.directional_score_floor = fl
                c.features.enable_regime_weights = rw
                c.features.enable_adversarial = False
                m = Backtester(c, ff).run(start, end).metrics
                rows.append(SensitivityRow(mc, fl, rw, m.total_return, m.sharpe, m.max_drawdown, m.n_trades))
    return rows
```

### `execution/walk_forward.py`

```python
"""Chronological walk-forward validation.

Fold layout (bar indices, all chronological, no shuffling)::

    [ train ][ embargo ][ validation ][ embargo ][ test ]
    -- step forward by ``step_bars`` (default = test_bars) --

* Train: strategy runs with the feedback loop on; weights adapt using only
  outcomes matured inside the training window.
* Validation: learned weights frozen; ``min_confidence`` is chosen from
  ``threshold_grid`` by validation Sharpe.
* Test: weights and threshold frozen; results are out-of-sample.
* Embargo >= outcome horizon (enforced by config validation) so labels that
  overlap a boundary cannot leak into the next segment.
* Indicators are causal, so precomputing them on the full series does not
  leak (tested in ``tests/test_integration.py``). Each segment starts flat
  with fresh capital and fresh Kalman/kill-switch state.
* Confidence calibration is fitted on training decisions and scored on test
  decisions (Brier score vs. a constant base-rate forecast).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import FrameworkConfig
from core.feature_bus import FeatureFrame
from core.models import PerformanceMetrics, Trade
from execution.backtester import Backtester, BacktestResult, _rule_backtest, bars_per_day
from execution.evaluator import LogisticCalibrator, brier, compute_metrics, matured_decision_outcomes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldWindows:
    """Index boundaries of one fold (half-open intervals)."""

    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


@dataclass
class FoldResult:
    """Results of one fold."""

    fold: int
    windows: FoldWindows
    train_metrics: PerformanceMetrics
    validation_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    test_buy_and_hold: PerformanceMetrics
    chosen_min_confidence: float
    learned_weights: dict[str, float]
    weight_updates: int
    calibration_brier: float | None
    baseline_brier: float | None
    test_returns: pd.Series
    test_bh_returns: pd.Series
    test_trades: list[Trade]
    test_exposures: pd.Series


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward output (in-sample and out-of-sample kept separate)."""

    folds: list[FoldResult]
    oos_metrics: PerformanceMetrics
    oos_buy_and_hold: PerformanceMetrics
    in_sample_mean_return: float
    validation_mean_return: float


def make_folds(config: FrameworkConfig, n_bars: int, first_index: int) -> list[FoldWindows]:
    """Generate chronological, non-overlapping-by-construction fold windows."""
    wf = config.walk_forward
    step = wf.step_bars or wf.test_bars
    folds = []
    s = first_index
    while True:
        tr = (s, s + wf.train_bars)
        va = (tr[1] + wf.embargo_bars, tr[1] + wf.embargo_bars + wf.validation_bars)
        te = (va[1] + wf.embargo_bars, va[1] + wf.embargo_bars + wf.test_bars)
        if te[1] > n_bars:
            break
        assert tr[1] + wf.embargo_bars <= va[0] and va[1] + wf.embargo_bars <= te[0]
        folds.append(FoldWindows(tr, va, te))
        s += step
    return folds


def _chain(returns: list[pd.Series], initial: float) -> pd.Series:
    """Compound per-bar returns from consecutive test windows into one equity curve."""
    r = pd.concat(returns)
    eq = initial * (1.0 + r).cumprod()
    first = pd.Series([initial], index=[r.index[0] - pd.Timedelta(seconds=1)])
    return pd.concat([first, eq])


def run_walk_forward(config: FrameworkConfig, ff: FeatureFrame) -> WalkForwardReport:
    """Execute the full walk-forward procedure and aggregate out-of-sample results."""
    folds = make_folds(config, len(ff), ff.first_trade_index)
    if not folds:
        wf = config.walk_forward
        need = wf.train_bars + wf.validation_bars + wf.test_bars + 2 * wf.embargo_bars
        raise ValueError(
            f"date range too short for walk-forward: need >= {need} tradable bars, have {len(ff) - ff.first_trade_index}"
        )
    closes = ff.column("close")
    h = config.strategy.prediction_horizon
    results: list[FoldResult] = []
    for k, fw in enumerate(folds):
        logger.info("walk-forward fold %d: train %s val %s test %s", k, fw.train, fw.validation, fw.test)
        train = Backtester(config, ff).run(*fw.train, learn=True)
        weights = train.final_weights
        best = None
        for mc in config.walk_forward.threshold_grid:
            c = config.model_copy(deep=True)
            c.strategy.min_confidence = mc
            c.features.enable_adversarial = False
            vr = Backtester(c, ff).run(*fw.validation, initial_weights=weights)
            key = (vr.metrics.sharpe if vr.metrics.sharpe is not None else -np.inf, vr.metrics.total_return)
            if best is None or key > best[0]:
                best = (key, mc, vr)
        assert best is not None
        _, mc, val = best
        c = config.model_copy(deep=True)
        c.strategy.min_confidence = mc
        test = Backtester(c, ff).run(*fw.test, initial_weights=weights)
        bh_run = _bh_run(config, ff, *fw.test)
        bh = bh_run.metrics

        cal_b = base_b = None
        if config.evaluation.calibration:
            x_tr, y_tr = matured_decision_outcomes(train.decision_records, closes, h, fw.train[1] - 1, *fw.train)
            x_te, y_te = matured_decision_outcomes(test.decision_records, closes, h, fw.test[1] - 1, *fw.test)
            cal = LogisticCalibrator()
            if cal.fit(x_tr, y_tr) and len(y_te):
                cal_b = brier(cal.predict(x_te), y_te)
                base_b = brier(np.full(len(y_te), y_tr.mean()), y_te)
        bh_eq = bh_run.equity
        results.append(
            FoldResult(
                fold=k, windows=fw, train_metrics=train.metrics, validation_metrics=val.metrics,
                test_metrics=test.metrics, test_buy_and_hold=bh, chosen_min_confidence=mc,
                learned_weights=weights, weight_updates=len(train.weight_updates),
                calibration_brier=cal_b, baseline_brier=base_b,
                test_returns=test.equity.pct_change().dropna(), test_bh_returns=bh_eq.pct_change().dropna(),
                test_trades=list(test.portfolio.trades),
                test_exposures=test.exposures,
            )
        )
    init = config.execution.initial_capital
    oos_eq = _chain([f.test_returns for f in results], init)
    oos_bh = _chain([f.test_bh_returns for f in results], init)
    zero = pd.Series(0.0, index=oos_bh.index)
    oos_expo = pd.concat([f.test_exposures for f in results])
    trades = [t for f in results for t in f.test_trades]
    fees = sum(f.test_metrics.total_fees for f in results)
    slip = sum(f.test_metrics.total_slippage for f in results)
    turn_notional = sum(f.test_metrics.turnover * f.test_metrics.initial_equity for f in results)
    oos = compute_metrics(oos_eq, trades, oos_expo, fees, slip, turn_notional, config.bars_per_year, bars_per_day(config))
    oos_bh_m = compute_metrics(oos_bh, [], zero, 0.0, 0.0, 0.0, config.bars_per_year, bars_per_day(config))
    return WalkForwardReport(
        folds=results, oos_metrics=oos, oos_buy_and_hold=oos_bh_m,
        in_sample_mean_return=float(np.mean([f.train_metrics.total_return for f in results])),
        validation_mean_return=float(np.mean([f.validation_metrics.total_return for f in results])),
    )


def _bh_run(config: FrameworkConfig, ff: FeatureFrame, start: int, end: int) -> BacktestResult:
    """Buy-and-hold run for a window (same entry convention as run_benchmarks)."""
    size = config.execution.max_position_size
    return _rule_backtest(config, ff, start, end, lambda i, cur: size if (i == start and cur == 0) else None)
```

### `reporting.py`

```python
"""Plain-text report formatting for the CLI (no printing here; main.py prints)."""

from __future__ import annotations

from typing import Optional

from agents.adversarial_agent import AdversarialReport
from core.models import AgentScore, AgentSignal, BossDecision, PerformanceMetrics, Trade
from execution.backtester import AblationRow, SensitivityRow
from execution.walk_forward import WalkForwardReport


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def _num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def format_metrics(title: str, m: PerformanceMetrics, bars_per_day: float) -> str:
    """Render performance metrics with explicit 'n/a' for unavailable values."""
    hold = "n/a" if m.avg_holding_bars is None else f"{m.avg_holding_bars:.1f} bars ({m.avg_holding_bars / bars_per_day:.1f} days)"
    lines = [
        f"== {title} ==",
        f"  initial equity      {m.initial_equity:,.2f}",
        f"  final equity        {m.final_equity:,.2f}",
        f"  total net return    {_pct(m.total_return)}",
        f"  CAGR                {_pct(m.cagr)}",
        f"  Sharpe (annualized) {_num(m.sharpe)}",
        f"  max drawdown        {m.max_drawdown:.2%}",
        f"  closed trades       {m.n_trades}",
        f"  win rate            {_num(m.win_rate, '{:.1%}')}",
        f"  profit factor       {_num(m.profit_factor)}",
        f"  turnover            {m.turnover:.2f}x mean equity",
        f"  total fees          {m.total_fees:,.2f}",
        f"  est. slippage       {m.total_slippage:,.2f}",
        f"  avg holding period  {hold}",
        f"  avg |exposure|      {m.avg_exposure:.2%}  (long {m.avg_long_exposure:.2%}, short {m.avg_short_exposure:.2%})",
        f"  bars                {m.n_bars}",
    ]
    for k, v in m.notes.items():
        lines.append(f"  note[{k}]: {v}")
    return "\n".join(lines)


def format_benchmarks(strategy: PerformanceMetrics, bench: dict[str, PerformanceMetrics]) -> str:
    """Side-by-side comparison on the identical window and cost model."""
    rows = [("strategy", strategy)] + list(bench.items())
    out = ["== Benchmark comparison (same window, same fees/slippage, mark-to-market) ==",
           f"  {'name':<14}{'return':>10}{'sharpe':>9}{'maxDD':>9}{'trades':>8}{'fees':>11}"]
    for name, m in rows:
        out.append(f"  {name:<14}{_pct(m.total_return):>10}{_num(m.sharpe):>9}{m.max_drawdown:>9.2%}{m.n_trades:>8}{m.total_fees:>11,.0f}")
    return "\n".join(out)


def format_leaderboard(scores: dict[str, AgentScore], suggested: dict[str, float]) -> str:
    """Agent leaderboard from matured outcomes."""
    out = ["== Agent leaderboard (matured outcomes only; hit = direction of 20-bar forward return) ==",
           f"  {'agent':<18}{'preds':>6}{'dir':>6}{'mat':>6}{'hit':>7}{'cw_ret':>10}{'standalone':>11}{'incr(bp)':>10}{'t':>7}{'w_now':>7}{'w_sugg':>8}"]
    ordered = sorted(scores.values(), key=lambda s: (s.incremental_contribution is None, -(s.incremental_contribution or 0)))
    for s in ordered:
        out.append(
            f"  {s.agent_name:<18}{s.prediction_count:>6}{s.directional_count:>6}{s.matured_count:>6}"
            f"{_num(s.hit_rate, '{:.1%}'):>7}{_num(s.confidence_weighted_return, '{:+.4f}'):>10}"
            f"{_num(s.standalone_pnl, '{:+.3f}'):>11}{_num(s.incremental_contribution, '{:+.2f}'):>10}"
            f"{_num(s.incremental_tstat, '{:+.2f}'):>7}{s.current_weight:>7.2f}{suggested.get(s.agent_name, s.current_weight):>8.2f}"
        )
        if s.note:
            out.append(f"      note: {s.note}")
    out.append("  (standalone = non-compounded unit-position return sum after costs; incr = signal-level ablation)")
    return "\n".join(out)


def format_ablation(rows: list[AblationRow]) -> str:
    """Execution-level ablation table."""
    out = ["== Execution-level ablation (full minus without-agent; in-sample, descriptive) ==",
           f"  {'removed agent':<18}{'return w/o':>12}{'sharpe w/o':>12}{'d_return':>11}{'d_sharpe':>10}"]
    for r in rows:
        out.append(f"  {r.agent_name:<18}{_pct(r.total_return_without):>12}{_num(r.sharpe_without):>12}"
                   f"{_pct(r.delta_return):>11}{_num(r.delta_sharpe, '{:+.2f}'):>10}")
    return "\n".join(out)


def format_sensitivity(rows: list[SensitivityRow]) -> str:
    """Threshold/regime-weight sensitivity grid."""
    out = ["== Sensitivity (in-sample; do not pick parameters from this table) ==",
           f"  {'min_conf':>9}{'floor':>7}{'regimeW':>9}{'return':>10}{'sharpe':>8}{'maxDD':>8}{'trades':>7}"]
    for r in rows:
        out.append(f"  {r.min_confidence:>9.2f}{r.score_floor:>7.2f}{str(r.regime_weights):>9}{_pct(r.total_return):>10}"
                   f"{_num(r.sharpe):>8}{r.max_drawdown:>8.2%}{r.n_trades:>7}")
    return "\n".join(out)


def format_decision(decision: BossDecision, signals: list[AgentSignal]) -> str:
    """Bar-level view: every agent's signal plus the boss aggregation."""
    out = [f"== Bar-level decision @ {decision.timestamp.isoformat()} =="]
    for s in signals:
        feats = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.features.items())
        out.append(f"  [{s.agent_name}] {s.signal.value} conf={s.confidence:.3f} available={s.available}")
        out.append(f"      reason: {s.reason}")
        out.append(f"      features: {feats}")
    out.append("  weights (base x regime x family -> learned/effective):")
    for c in decision.breakdown:
        out.append(f"      {c.agent_name:<18} base {c.base_weight:.2f} learned {c.learned_weight:.2f} regime x{c.regime_factor:.2f} "
                   f"family x{c.family_factor:.2f} -> eff {c.effective_weight:.2f}; input {c.signed_input:+.3f} contrib {c.contribution:+.3f}")
    out += [
        f"  regime={decision.regime.value} directional_score={decision.directional_score:+.4f} agreement={decision.agreement:.3f} "
        f"coverage={decision.coverage:.3f} data_coverage={decision.data_coverage:.3f}",
        f"  ensemble strength={decision.confidence:.3f} ({decision.confidence_label})",
        f"  risk_score={_num(decision.risk_score)} risk_veto={decision.risk_veto} kill_switch={decision.kill_switch_active}",
        f"  raw_action={decision.raw_action.value} action={decision.action.value} forced_hold_rule={decision.forced_hold_rule}",
        f"  analyst note: {decision.reason}",
    ]
    return "\n".join(out)


def format_trade(t: Trade) -> str:
    """One round-trip trade."""
    side = "LONG" if t.direction > 0 else "SHORT"
    return (
        f"== Sample trade ==\n  {side} entry {t.entry_time} @ {t.entry_price:,.2f} (qty peak {t.max_quantity:.4f})\n"
        f"  exit {t.exit_time} @ {_num(t.exit_price, '{:,.2f}')} after {t.bars_held} bars\n"
        f"  gross PnL {t.gross_pnl:,.2f} (slippage inside fill prices: {t.slippage:,.2f}) fees {t.fees:,.2f} net PnL {t.net_pnl:,.2f}\n"
        f"  entry reason: {t.entry_reason[:300]}\n  exit reason: {t.exit_reason[:300]}"
    )


def format_adversarial(r: AdversarialReport) -> str:
    """Shadow-agent summary."""
    verdict = ("counter-consensus OUTPERFORMED consensus in this sample" if r.counter_outperformed
               else "consensus outperformed counter-consensus in this sample")
    return (
        "== Adversarial shadow agent (never votes; same timing/costs; ignores risk controls) ==\n"
        f"  disagreement observations {r.observations}, matured outcomes {r.matured_outcomes}, "
        f"counter hit rate {_num(r.counter_hit_rate, '{:.1%}')}\n"
        f"  consensus book: final {r.consensus_final_equity:,.0f} maxDD {r.consensus_max_drawdown:.1%} | "
        f"counter book: final {r.counter_final_equity:,.0f} maxDD {r.counter_max_drawdown:.1%}\n  {verdict}"
    )


def format_walk_forward(rep: WalkForwardReport, bars_per_day: float) -> str:
    """Per-fold table plus aggregated out-of-sample metrics."""
    out = ["== Walk-forward folds (IS = train with feedback, VAL = threshold selection, OOS = frozen test) ==",
           f"  {'fold':>4}  {'train':>11} {'val':>11} {'test':>11}  {'IS ret':>8} {'VAL ret':>8} {'OOS ret':>8} {'B&H OOS':>8} {'min_conf':>8} {'w_upd':>5} {'Brier cal/base':>15}"]
    for f in rep.folds:
        w = f.windows
        cal = "n/a" if f.calibration_brier is None else f"{f.calibration_brier:.3f}/{f.baseline_brier:.3f}"
        out.append(
            f"  {f.fold:>4}  {w.train[0]:>5}-{w.train[1]:<5} {w.validation[0]:>5}-{w.validation[1]:<5} {w.test[0]:>5}-{w.test[1]:<5}  "
            f"{_pct(f.train_metrics.total_return):>8} {_pct(f.validation_metrics.total_return):>8} {_pct(f.test_metrics.total_return):>8} "
            f"{_pct(f.test_buy_and_hold.total_return):>8} {f.chosen_min_confidence:>8.2f} {f.weight_updates:>5} {cal:>15}"
        )
    out.append(f"  mean in-sample (train) return {rep.in_sample_mean_return:+.2%}; mean validation return {rep.validation_mean_return:+.2%}")
    last = rep.folds[-1].learned_weights
    out.append("  learned weights after last training window: " + ", ".join(f"{k}={v:.2f}" for k, v in last.items()))
    out.append(format_metrics("Aggregated OUT-OF-SAMPLE (chained test windows)", rep.oos_metrics, bars_per_day))
    b = rep.oos_buy_and_hold
    out.append(f"  buy-and-hold over the same chained test windows: return {_pct(b.total_return)}, Sharpe {_num(b.sharpe)}, maxDD {b.max_drawdown:.2%}")
    return "\n".join(out)
```

### `main.py`

```python
"""Command-line interface for the multi-agent trading framework.

Examples::

    python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic
    python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --walk-forward
    python main.py --mode paper --symbol BTCUSDT --synthetic

Dates are interpreted as UTC midnight. The range is [start, end): end is exclusive.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from core.config import ConfigError, FrameworkConfig, load_config
from core.feature_bus import compute_features
from data.feed import DataValidationError, LiveFeedStub, LiveFeedUnavailableError, MarketData, SyntheticFeed, parse_utc_date
from core.models import AgentScore
from execution.backtester import BacktestResult, Backtester, bars_per_day, run_ablation, run_benchmarks, run_sensitivity
from execution.evaluator import Evaluator
from execution.walk_forward import run_walk_forward
import reporting

logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (all defaults documented in --help)."""
    p = argparse.ArgumentParser(
        description="Multi-agent trading strategy framework (research/simulation only; never sends real orders).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["backtest", "paper"], default="backtest", help="backtest or paper (simulated) execution")
    p.add_argument("--symbol", default="BTCUSDT", help="instrument label (synthetic mode: label only, does not change data)")
    p.add_argument("--start", default="2023-01-01", help="UTC start date, inclusive (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-01-01", help="UTC end date, exclusive (YYYY-MM-DD)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use the deterministic synthetic feed (all data labelled SYNTHETIC)")
    src.add_argument("--live-feed", action="store_true", help="use the live-feed STUB (raises: no provider connected)")
    p.add_argument("--walk-forward", action="store_true", help="run chronological walk-forward validation (backtest mode)")
    p.add_argument("--seed", type=int, default=None, help="synthetic seed (default: app.seed from config)")
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")), help="YAML config path")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None, help="override app.log_level")
    p.add_argument("--no-ablation", action="store_true", help="skip execution-level agent ablation (5 extra backtests)")
    p.add_argument("--sensitivity", action="store_true", help="run threshold/regime-weight sensitivity grid (18 backtests)")
    p.add_argument("--show-decision", type=int, default=None,
                   help="bar offset (from --start) to print in full; default: first directional decision")
    p.add_argument("--paper-bars", type=int, default=300, help="paper mode: number of bars to replay from --start")
    p.add_argument("--output-dir", default=None, help="write equity/trades/decisions CSV and leaderboard JSON here")
    return p


def _load_data(args: argparse.Namespace, cfg: FrameworkConfig) -> MarketData:
    """Fetch data from the selected source (never silently falls back)."""
    start, end = parse_utc_date(args.start), parse_utc_date(args.end)
    if end <= start:
        raise DataValidationError(f"--end ({args.end}) must be after --start ({args.start}); end is exclusive")
    if args.live_feed:
        return LiveFeedStub().get_bars(args.symbol, start, end)
    return SyntheticFeed(cfg, seed=args.seed).get_bars(args.symbol, start, end)


def _run_backtest(args: argparse.Namespace, cfg: FrameworkConfig, md: MarketData) -> list[str]:
    """Run a single backtest with benchmarks, leaderboard, ablation and adversarial report."""
    ff = compute_features(md, cfg)
    start, end = ff.first_trade_index, len(ff)
    bt = Backtester(cfg, ff)
    res = bt.run(start, end)
    bpd = bars_per_day(cfg)
    out = [f"Data source: {md.source} | symbol {md.symbol} | timeframe {cfg.data.timeframe} | "
           f"bars traded {end - start} ({ff.index[start]} .. {ff.index[end - 1]}, UTC) | warm-up bars {start}",
           "NOTE: synthetic data - results say nothing about real-market profitability."]
    pick = args.show_decision
    if pick is None:
        pick = next((k for k, d in enumerate(res.decisions) if d.action.value != "HOLD"), 0)
    if not 0 <= pick < len(res.decisions):
        raise ValueError(f"--show-decision must be in [0, {len(res.decisions) - 1}]")
    out.append(reporting.format_decision(res.decisions[pick], res.signals[pick]))
    closed = [t for t in res.portfolio.trades if t.is_closed]
    out.append(reporting.format_trade(closed[0]) if closed else "== Sample trade ==\n  no closed trades")
    ev = Evaluator(cfg, bt.boss.base_weights)
    scores = ev.score_agents(res.signal_records, ff.column("close"), as_of_index=end - 1, current_weights=res.final_weights, min_index=start)
    out.append(reporting.format_leaderboard(scores, ev.suggest_weights(scores)))
    if not args.no_ablation:
        out.append(reporting.format_ablation(run_ablation(cfg, ff, start, end, res)))
    if args.sensitivity:
        out.append(reporting.format_sensitivity(run_sensitivity(cfg, ff, start, end)))
    if res.adversarial:
        out.append(reporting.format_adversarial(res.adversarial))
    rc = ", ".join(f"{k}={v}" for k, v in sorted(res.rule_counts.items()))
    out.append(f"== Forced-HOLD rule counts (a bar can trigger several) ==\n  {rc or 'none'}")
    ks = "; ".join(f"{e.kind}@{e.timestamp:%Y-%m-%d %H:%M} ({e.detail})" for e in res.kill_switch_events)
    out.append(f"== Kill-switch events ==\n  {ks or 'none'}")
    out.append(reporting.format_benchmarks(res.metrics, run_benchmarks(cfg, ff, start, end)))
    out.append(reporting.format_metrics("Strategy performance (net of fees and slippage)", res.metrics, bpd))
    if res.unfilled_orders:
        out.append(f"  unfilled orders at window end: {len(res.unfilled_orders)} (no next bar; no fill invented)")
    if args.output_dir:
        _write_outputs(Path(args.output_dir), res, scores)
        out.append(f"Outputs written to {args.output_dir}")
    return out


def _write_outputs(path: Path, res: BacktestResult, scores: dict[str, AgentScore]) -> None:
    """Persist equity curve, fills, trades, decisions and leaderboard."""
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"equity": res.equity, "exposure": res.exposures}).to_csv(path / "equity.csv")
    pd.DataFrame([asdict(f) for f in res.portfolio.fills]).to_csv(path / "fills.csv", index=False)
    pd.DataFrame([{**asdict(t), "net_pnl": t.net_pnl} for t in res.portfolio.trades]).to_csv(path / "trades.csv", index=False)
    pd.DataFrame([{
        "timestamp": d.timestamp, "action": d.action.value, "raw_action": d.raw_action.value, "strength": d.confidence,
        "directional_score": d.directional_score, "agreement": d.agreement, "coverage": d.coverage, "regime": d.regime.value,
        "risk_score": d.risk_score, "risk_veto": d.risk_veto, "kill_switch": d.kill_switch_active,
        "forced_hold_rule": d.forced_hold_rule, "reason": d.reason} for d in res.decisions]).to_csv(path / "decisions.csv", index=False)
    (path / "leaderboard.json").write_text(json.dumps({k: asdict(v) for k, v in scores.items()}, indent=2, default=str))


def _run_paper(args: argparse.Namespace, cfg: FrameworkConfig, md: MarketData) -> list[str]:
    """Replay bars one at a time through the PaperExecutor (simulated fills only)."""
    if args.paper_bars < 2:
        raise ValueError("--paper-bars must be >= 2")
    ff = compute_features(md, cfg)
    start = ff.first_trade_index
    end = min(len(ff), start + args.paper_bars)
    bt = Backtester(cfg, ff, paper=True)
    res = bt.run(start, end)
    out = [
        f"PAPER MODE (simulated fills, NO broker connection, no real orders) | source {md.source} | bars {end - start}",
        f"  simulated fills: {len(res.portfolio.fills)}; closed trades: {res.metrics.n_trades}; "
        f"unfilled at end: {len(res.unfilled_orders)}",
    ]
    for f in res.portfolio.fills[:10]:
        out.append(f"  SIM FILL {f.timestamp} {f.side.value} {f.quantity:.4f} @ {f.exec_price:,.2f} fee {f.fee:,.2f}")
    if len(res.portfolio.fills) > 10:
        out.append(f"  ... {len(res.portfolio.fills) - 10} more simulated fills (see log)")
    out.append(reporting.format_metrics("Paper session performance", res.metrics, bars_per_day(cfg)))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        overrides: dict = {}
        if args.seed is not None:
            overrides.setdefault("app", {})["seed"] = args.seed
        if args.walk_forward:
            if args.mode != "backtest":
                raise ConfigError("--walk-forward is only valid with --mode backtest")
            overrides.setdefault("walk_forward", {})["enabled"] = True
        if args.log_level:
            overrides.setdefault("app", {})["log_level"] = args.log_level
        cfg = load_config(args.config, overrides)
        logging.basicConfig(level=cfg.app.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
        md = _load_data(args, cfg)
        if args.mode == "paper":
            lines = _run_paper(args, cfg, md)
        elif cfg.walk_forward.enabled:
            ff = compute_features(md, cfg)
            rep = run_walk_forward(cfg, ff)
            lines = [f"Data source: {md.source} | {len(ff) - ff.first_trade_index} tradable bars (UTC, end exclusive)",
                     "NOTE: synthetic data - out-of-sample here means unseen *synthetic* bars, not evidence of real alpha.",
                     reporting.format_walk_forward(rep, bars_per_day(cfg))]
        else:
            lines = _run_backtest(args, cfg, md)
    except LiveFeedUnavailableError as exc:
        print(f"LIVE FEED UNAVAILABLE: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, DataValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\n\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```


## Deliverable 3 — config.yaml and requirements.txt

### `config.yaml`

```yaml
# Multi-agent trading framework configuration.
# All values are starting defaults for a synthetic research baseline. None of
# them has been empirically optimized; do not treat them as validated settings.

app:
  seed: 42                 # synthetic data seed (CLI --seed overrides)
  timezone: UTC            # all timestamps are UTC; CLI dates are UTC midnight
  log_level: INFO

data:
  timeframe: 4h            # 1h | 4h | 1d ; sets bars_per_year (24/7 crypto: 4h -> 2190)
  warmup_bars: 250         # history generated/loaded before --start; never traded
  min_history_bars: 120    # reject datasets shorter than this

strategy:
  min_confidence: 0.35           # min ensemble strength (uncalibrated) to act
  directional_score_floor: 0.15  # min |S| to act
  min_data_coverage: 0.5         # min share of enabled agent weight with valid data
  prediction_horizon: 20         # bars; directional evaluation only (not holding period)
  long_only: true
  allow_short: false             # must be false when long_only is true
  sell_policy: close             # close | reduce  (what SELL does to an existing long)
  sell_reduce_fraction: 0.5      # used when sell_policy = reduce
  min_rebalance_exposure: 0.05   # ignore target changes smaller than this
  on_risk_veto: reduce           # hold_position | reduce  (existing position on veto)
  veto_exposure_cap: 0.5         # max |exposure| kept while risk veto is active

execution:
  initial_capital: 100000
  fee_bps: 10                    # charged on every fill's notional
  slippage_bps: 5                # adverse price adjustment on every fill
  execution_price: next_bar_open # only supported model: decide at close t, fill at open t+1
  max_position_size: 1.0         # max target exposure from one BUY (fraction of equity)
  max_exposure: 1.0              # hard cap on |exposure|
  exposure_cap: 1.0              # target = min(max_position_size, strength * exposure_cap)
  max_leverage: 1.0              # 1.0 = no leverage
  volatility_aware_sizing: false
  target_volatility: 0.5         # annualized; size *= min(1, target_vol / realized_vol)

risk:
  risk_veto_threshold: 0.80      # market risk score above this forces HOLD
  max_drawdown: 0.20             # portfolio kill switch: drawdown cap
  rolling_sharpe_threshold: -1.0 # portfolio kill switch: annualized rolling Sharpe floor
  rolling_sharpe_window: 120     # bars
  kill_switch_min_history: 60    # bars of returns before the switch can trigger
  kill_switch_recovery: cooldown # cooldown | manual
  kill_switch_cooldown_bars: 60
  kill_switch_position_policy: flatten  # flatten | hold existing positions when triggered

evaluation:
  lookahead: 20                  # must equal strategy.prediction_horizon
  feedback_interval: 60          # bars between weight updates (training only)
  min_agent_samples: 30          # matured directional predictions required to update
  min_weight: 0.3
  max_weight: 2.0
  weight_smoothing: 0.25         # new = (1-s)*old + s*target
  shrinkage_to_base: 0.5         # target pulled toward the base weight
  tstat_neutral_band: 1.0        # |t| below this is treated as no evidence
  weight_sensitivity: 0.15       # target multiplier exp(sensitivity * t)
  calibration: true              # fit strength->hit calibration on train, score on test

walk_forward:
  enabled: false                 # CLI --walk-forward enables
  train_bars: 500
  validation_bars: 100
  test_bars: 100
  embargo_bars: 20               # must be >= prediction_horizon
  step_bars: null                # null = test_bars
  threshold_grid: [0.25, 0.35, 0.45]  # min_confidence candidates chosen on validation

features:
  enable_optional_agent: false
  optional_agent_type: on_chain
  enable_kalman: true
  enable_regime_weights: true
  enable_adversarial: true
  enable_family_normalization: false

indicators:
  ofi_window: 20
  cvd_window: 50
  momentum_lookback: 20
  pv_corr_window: 50
  rv_window: 30
  atr_window: 14
  drawdown_window: 120
  ema_fast: 20
  ema_slow: 100
  adx_period: 14
  sweep_lookback: 20
  absorption_window: 20
  netflow_window: 50

agents:
  base_weights:
    market_intel: 1.0
    risk_engine: 1.0
    liquidity_engine: 1.0
    sentiment_engine: 1.0
    regime_detector: 1.0
    on_chain: 0.8
  families:                      # signal families (for optional family normalization)
    market_intel: price_flow
    regime_detector: price_flow
    liquidity_engine: microstructure
    sentiment_engine: sentiment
    risk_engine: risk
    on_chain: on_chain
  market_intel:
    w_ofi: 0.4
    w_cvd: 0.2
    w_momentum: 0.4
    signal_threshold: 0.15
  risk_engine:
    rv_low: 0.30                 # annualized vol mapped to score 0
    rv_high: 1.50                # ... mapped to score 1
    atr_pct_low: 0.005
    atr_pct_high: 0.04
    drawdown_full: 0.35          # market drawdown mapped to score 1
    w_vol: 0.4
    w_atr: 0.3
    w_drawdown: 0.3
  liquidity_engine:
    min_confirmations: 1
    absorption_threshold: 1.5
    depth_threshold: 0.05
    cvd_threshold: 0.0
  sentiment_engine:
    social_min: 0
    social_max: 100
    news_min: -1
    news_max: 1
    funding_threshold: 0.0003    # per 8h; |funding| below this is ignored
    funding_saturation: 0.0010   # contrarian term reaches +-1 here
    w_social: 0.35
    w_news: 0.35
    w_funding: 0.30
    min_components: 2
    signal_threshold: 0.20
    treat_missing_as_neutral: false
  regime_detector:
    adx_trend: 20
    adx_strong: 40
    volatile_rv: 1.2
    directional_vote: true
  on_chain:
    netflow_z_threshold: 0.5
    w_netflow: 0.6
    w_stablecoin: 0.4

# Multiplicative factors applied to learned weights in each regime.
# Illustrative initial values, NOT empirically validated.
regime_weights:
  trend_up:   {market_intel: 1.4, risk_engine: 1.0, liquidity_engine: 0.9, sentiment_engine: 1.1, regime_detector: 0.8, on_chain: 1.0}
  trend_down: {market_intel: 1.3, risk_engine: 1.2, liquidity_engine: 1.0, sentiment_engine: 1.0, regime_detector: 0.8, on_chain: 1.0}
  range:      {market_intel: 0.8, risk_engine: 1.0, liquidity_engine: 1.3, sentiment_engine: 0.9, regime_detector: 1.0, on_chain: 1.0}
  volatile:   {market_intel: 0.7, risk_engine: 1.5, liquidity_engine: 1.0, sentiment_engine: 0.7, regime_detector: 1.2, on_chain: 1.0}

kalman:
  process_noise: 0.02            # higher = faster reaction, less smoothing
  observation_noise: 0.10
  initial_variance: 1.0

synthetic:
  start_price: 20000
  base_volume: 1000
  base_vol_annual: 0.55
  min_segment_bars: 60
  max_segment_bars: 240
  flow_signal_strength: 0.25     # weak regime dependence of flow/sentiment (0 = pure noise)
  generate_optional: true        # synthetic on-chain columns
```

### `requirements.txt`

```text
numpy>=1.26
pandas>=2.1
scipy>=1.11
pydantic>=2.5
pyyaml>=6.0
```

Defaults that materially affect risk/performance: `min_confidence` 0.35 and `directional_score_floor` 0.15 (trade frequency; results are very sensitive to them), `fee_bps` 10 + `slippage_bps` 5 on every fill, `max_leverage`/`max_exposure` 1.0 (no leverage), `on_risk_veto: reduce` to 0.5 exposure, kill switch at 20% drawdown or rolling Sharpe < -1 with flatten + 60-bar cooldown, weight bounds [0.3, 2.0] with 0.25 smoothing and 0.5 shrinkage, `flow_signal_strength` 0.25 (how much signal the synthetic generator plants).


## Deliverable 4 — README.md

````markdown
# Multi-Agent Trading Strategy Framework

This is a research and simulation framework. Five specialist agents vote, a Boss Orchestrator aggregates
their votes, and a cost-aware backtester executes the result causally. On top of that sit agent
attribution, leakage-free adaptive weights and walk-forward validation.

> **Status:** this is a tested, reproducible *baseline*. It is not a profitable or production-ready trading
> system. All bundled data is **synthetic**. No real orders can be sent.

## Install

* Python **3.11+** (developed and tested on 3.13; syntax-checked on 3.11)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # numpy, pandas, scipy, pydantic, pyyaml only
```

## Project structure

```
trading_agents/
├── agents/                 voting agents, boss, adversarial shadow, agent factory
│   ├── base_agent.py       BaseAgent contract + explicit "unavailable" HOLD
│   ├── market_intel.py     OFI + CVD + institutional-style momentum
│   ├── risk_engine.py      market risk score (vol, ATR%, drawdown) -> veto
│   ├── liquidity_engine.py confirmed liquidity sweeps
│   ├── sentiment_engine.py social + news + contrarian funding
│   ├── regime_detector.py  EMA20/100 + Wilder ADX + RV -> regime
│   ├── on_chain.py         optional 6th agent (feature flag)
│   ├── adversarial_agent.py shadow counter-consensus books (never votes)
│   └── boss_orchestrator.py weighted vote, diagnostics, overrides, explanation
├── core/                   shared contracts and utilities (added for clarity)
│   ├── models.py           dataclasses: MarketState, AgentSignal, BossDecision, Order, Fill, Trade, ...
│   ├── config.py           Pydantic config schema + loader (unknown keys rejected)
│   ├── feature_bus.py      causal indicator computation -> MarketState per bar
│   ├── kalman.py           1-D Kalman filter
│   ├── portfolio_risk.py   drawdown/Sharpe kill switch
│   └── reasoner.py         Reasoner protocol + deterministic TemplateReasoner
├── indicators/             pure functions: order_flow, volatility, liquidity, sentiment, trend (EMA/ADX)
├── execution/
│   ├── backtester.py       event loop, position management, benchmarks, ablation, sensitivity
│   ├── executor.py         Portfolio accounting, SimulatedExecutor, PaperExecutor, NullBroker
│   ├── evaluator.py        metrics, agent scoring, weight suggestions, calibration
│   ├── feedback.py         periodic training-only reweighting
│   └── walk_forward.py     chronological train/val/test with embargo
├── data/feed.py            synthetic feed, validation, LiveFeedStub
├── tests/                  unittest suite (168 tests)
├── docs/                   design, assumptions, next steps, captured transcript
├── reporting.py            CLI text formatting (library code never prints)
├── main.py                 CLI
├── config.yaml
└── requirements.txt
```

## Commands

```bash
python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic
python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --walk-forward
python main.py --mode paper --symbol BTCUSDT --synthetic
python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --seed 42
python main.py --help
python -m unittest discover -s tests -v
```

Runtime on the build machine: backtest about 7 s (including 5 ablation re-runs), walk-forward about 9 s,
tests about 5 s.

### CLI arguments

| Argument | Default | Behaviour |
|---|---|---|
| `--mode` | `backtest` | `backtest` or `paper` (bar-by-bar replay through `PaperExecutor`, simulated fills only) |
| `--symbol` | `BTCUSDT` | label; in synthetic mode it does not change the data |
| `--start` | `2023-01-01` | UTC midnight, **inclusive** |
| `--end` | `2024-01-01` | UTC midnight, **exclusive** |
| `--synthetic` / `--live-feed` | one required | data source; `--live-feed` is a stub that exits with code 2 and never falls back |
| `--walk-forward` | off | walk-forward validation (backtest mode only) |
| `--seed` | `app.seed` (42) | synthetic seed |
| `--config` | `config.yaml` next to `main.py` | YAML config |
| `--log-level` | `app.log_level` | DEBUG shows every forced-HOLD rule per bar |
| `--no-ablation` | off | skip the 5 execution-level ablation re-runs |
| `--sensitivity` | off | 18-run grid over min_confidence x floor x regime weights |
| `--show-decision N` | first non-HOLD | print the full bar-level decision at bar offset N |
| `--paper-bars` | 300 | bars replayed in paper mode |
| `--output-dir` | none | write equity.csv, fills.csv, trades.csv, decisions.csv, leaderboard.json |

Exit codes: 0 means success; 2 means a config, data, date-range or live-feed error (with an actionable
message).

## Configuration guide (`config.yaml`)

The config is validated at startup. Unknown keys and inconsistent values are rejected (for example
embargo < horizon, `long_only` together with `allow_short`, `max_exposure > max_leverage`, or weights
outside the bounds). Settings that most affect risk and performance:

* `strategy.min_confidence` (0.35) and `directional_score_floor` (0.15): together these decide how often
  the strategy trades. The sensitivity grid shows results swing widely with them, on very few trades.
* `execution.fee_bps` 10 / `slippage_bps` 5: charged on every fill, including exits and reversals.
* `execution.max_leverage` 1.0, `max_exposure` 1.0: no leverage by default.
* `strategy.on_risk_veto: reduce` with `veto_exposure_cap` 0.5: what happens to an open position when the
  veto fires.
* `risk.*`: kill switch at 20% drawdown or rolling Sharpe < -1 (after 60 bars). By default it flattens
  positions, then recovers after a 60-bar cooldown.
* `evaluation.*`: weight bounds [0.3, 2.0], smoothing 0.25, shrinkage 0.5, neutral |t| < 1.
* `regime_weights`: illustrative multipliers only, not fitted.
* `synthetic.flow_signal_strength` 0.25: how strongly synthetic flow and sentiment depend on the hidden
  regime. Set it to 0 for a pure-noise control.

## Boss math (summary)

* `S = sum(w_i * x_i) / sum(w_i)` over eligible (enabled and available) agents, where
  `x_i = c_i * d_i` (Kalman-smoothed if enabled). HOLD votes count in the denominator. Unavailable
  agents count in neither part.
* `agreement` = aligned directional weight / directional weight
* `coverage` = directional weight / eligible weight
* `data_coverage` = eligible weight / enabled weight
* `strength = agreement * mean_conf(aligned) * sqrt(coverage)`. This is labelled
  `uncalibrated_ensemble_strength`.
* HOLD rules, in order: `kill_switch`, `risk_unavailable`, `risk_veto`, `data_coverage`,
  `directional_score_floor`, `min_confidence`.
* Effective weight = learned x regime factor x family factor. All three appear in the decision breakdown.

## Performance metric definitions

All metrics are net of fees and slippage, computed from the bar-close mark-to-market equity curve.

* **Total return:** final equity / initial equity - 1.
* **CAGR:** only shown if the window is at least 0.25 years.
* **Sharpe:** mean / std of per-bar returns x sqrt(bars_per_year), with risk-free rate 0. "n/a" if fewer
  than 10 returns or zero variance.
* **Max drawdown:** maximum of (1 - equity / running peak).
* **Trades:** closed round trips (flat to flat). Win rate and profit factor use per-trade net PnL. Profit
  factor is n/a when there are no losses.
* **Turnover:** traded notional / mean equity.
* **Fees / slippage:** sums over all fills. Slippage is already inside the fill prices; it is reported for
  transparency.
* **Average exposure:** mean |position value| / equity per bar.
* **Agent hit rate:** share of matured BUY/SELL predictions whose 20-bar forward return has the predicted
  sign. HOLD is excluded. This is not the trade win rate.

## Sample output (actual, captured; seed 42; full transcript in `docs/sample_run_transcript.txt`)

```
== Benchmark comparison (same window, same fees/slippage, mark-to-market) ==
  name              return   sharpe    maxDD  trades       fees
  strategy          +6.17%     0.43   19.21%       5        383
  no_trade          +0.00%      n/a    0.00%       0          0
  buy_and_hold     -24.72%     0.28   68.91%       0        100
  ema_trend        -28.83%    -0.01   61.29%      14      2,081
```

How to read this honestly: the strategy was in the market about 10% of the time and made 5 trades. The
in-sample sensitivity grid ranges from -12% to +16% across nearby thresholds. The adversarial counter-book
beat the consensus book. This is **not** evidence of an edge.

## Connecting a real market-data provider

1. Implement a class with `get_bars(symbol, start, end) -> MarketData` (see `MarketDataProvider` in
   `data/feed.py`). Return a UTC-indexed DataFrame with at least `open, high, low, close, volume`.
2. Label every column you really receive `FeatureStatus.OBSERVED`. Label the rest `UNAVAILABLE`, and never
   fill them with synthetic values. Agents whose essential inputs are unavailable then abstain
   automatically, and the boss excludes them.
3. Include `warmup_bars` of history before `start` and set `first_trade_index`.
4. Call `validate_bars`, then wire the provider into `main._load_data` in place of `LiveFeedStub`.

## Paper executor and broker integration

`PaperExecutor` logs every order (`PAPER ORDER (simulated, no broker)`) and fills it with the same
next-open, fee and slippage model. Live routing is deliberately impossible: `NullBroker.submit_order`
raises, and `PaperExecutor(live_orders_enabled=True)` raises. To integrate a broker, implement
`BrokerAdapter.submit_order`, add reconciliation of real fills into `Portfolio`, and add a separate,
explicitly opted-in executor. Keep exchange-specific code out of `agents/` and `core/`.

## Known limitations

See `docs/assumptions.md`. In short:

* The data is synthetic.
* Execution is a single instrument with fixed bps costs and no market impact.
* Confidence is uncalibrated.
* The ensemble-of-bosses and options-flow agent are not implemented.
* The test suite has been run on Python 3.13 only.
````


## Deliverable 5 — Sample run transcript (actual captured output)

Captured by running the commands below in the build environment (Python 3.13.16). Timestamps in log lines are wall-clock.

```text
$ python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --seed 42
2026-10-12 06:29:18,290 INFO data.feed: synthetic feed: symbol=BTCUSDT seed=42 bars=2440 (warm-up 250, tradable 2190) 2022-11-20 08:00:00+00:00..2023-12-31 20:00:00+00:00
2026-10-12 06:29:18,391 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-01-21 00:00:00+00:00: rolling Sharpe -2.66 < -1.00
2026-10-12 06:29:18,419 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-01-31 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:18,553 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-03-28 04:00:00+00:00: rolling Sharpe -1.23 < -1.00
2026-10-12 06:29:18,577 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-04-07 04:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:18,739 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-06-16 00:00:00+00:00: rolling Sharpe -1.29 < -1.00
2026-10-12 06:29:18,763 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-06-26 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:18,976 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-09-25 08:00:00+00:00: rolling Sharpe -1.84 < -1.00
2026-10-12 06:29:19,002 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-10-05 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,071 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-11-08 08:00:00+00:00: rolling Sharpe -1.11 < -1.00
2026-10-12 06:29:19,095 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-11-18 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,324 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-01-11 00:00:00+00:00: drawdown 20.2% > cap 20.0%
2026-10-12 06:29:19,339 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-01-21 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,416 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-03-04 04:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:19,438 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-03-14 04:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,470 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-04-03 20:00:00+00:00: rolling Sharpe -1.52 < -1.00
2026-10-12 06:29:19,509 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-04-13 20:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,665 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-06-11 08:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:19,688 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-06-21 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,879 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-09-25 08:00:00+00:00: rolling Sharpe -2.62 < -1.00
2026-10-12 06:29:19,900 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-10-05 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:19,963 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-11-08 08:00:00+00:00: rolling Sharpe -1.11 < -1.00
2026-10-12 06:29:20,084 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-11-18 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:20,189 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-01-11 00:00:00+00:00: drawdown 20.0% > cap 20.0%
2026-10-12 06:29:20,210 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-01-21 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:20,350 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-03-26 00:00:00+00:00: rolling Sharpe -1.07 < -1.00
2026-10-12 06:29:20,372 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-04-05 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:20,538 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-06-11 20:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:20,565 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-06-21 20:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:20,760 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-09-24 12:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:20,783 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-10-04 12:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:20,857 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-11-08 08:00:00+00:00: rolling Sharpe -1.09 < -1.00
2026-10-12 06:29:20,877 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-11-18 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:20,994 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-01-11 04:00:00+00:00: drawdown 20.2% > cap 20.0%
2026-10-12 06:29:21,009 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-01-21 04:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:21,176 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-03-26 00:00:00+00:00: rolling Sharpe -1.07 < -1.00
2026-10-12 06:29:21,196 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-04-05 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:21,318 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-06-12 04:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:21,352 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-06-22 04:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:21,532 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-09-24 12:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:21,553 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-10-04 12:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:21,613 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-11-08 08:00:00+00:00: rolling Sharpe -1.09 < -1.00
2026-10-12 06:29:21,633 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-11-18 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:21,746 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-01-11 00:00:00+00:00: drawdown 20.3% > cap 20.0%
2026-10-12 06:29:21,771 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-01-21 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:21,896 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-03-26 00:00:00+00:00: rolling Sharpe -1.07 < -1.00
2026-10-12 06:29:21,920 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-04-05 00:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:22,044 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-06-11 20:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:22,067 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-06-21 20:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:22,276 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-09-24 12:00:00+00:00: rolling Sharpe -4.27 < -1.00
2026-10-12 06:29:22,299 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-10-04 12:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:22,359 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-11-08 08:00:00+00:00: rolling Sharpe -1.09 < -1.00
2026-10-12 06:29:22,379 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-11-18 08:00:00+00:00 (cooldown elapsed)
2026-10-12 06:29:22,656 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-03-26 20:00:00+00:00: rolling Sharpe -1.18 < -1.00
2026-10-12 06:29:22,687 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-04-05 20:00:00+00:00 (cooldown elapsed)
Data source: synthetic(seed=42) | symbol BTCUSDT | timeframe 4h | bars traded 2190 (2023-01-01 00:00:00+00:00 .. 2023-12-31 20:00:00+00:00, UTC) | warm-up bars 250

NOTE: synthetic data - results say nothing about real-market profitability.

== Bar-level decision @ 2023-01-01T00:00:00+00:00 ==
  [market_intel] BUY conf=0.280 available=True
      reason: BUY: buying pressure dominates - OFI +0.145, CVD change +0.091, institutional-style momentum +0.510, price/volume corr +0.05; composite +0.280 beyond +/-0.15 [inputs: synthetic]
      features: ofi=0.1448, cvd_norm=0.09126, inst_momentum=0.5099, pv_corr=0.05401, score=0.2801, cvd_is_proxy=False, input_provenance=synthetic
  [risk_engine] HOLD conf=0.400 available=True
      reason: HOLD (risk agent does not vote direction): elevated market risk, risk_score 0.60 (below veto 0.80); components: realized vol 120% -> 0.75, ATR 4.57% -> 1.00, market drawdown 0.0% -> 0.00
      features: realized_vol=1.199, atr_pct=0.04567, market_drawdown=0, vol_score=0.7492, atr_score=1, drawdown_score=0, risk_score=0.5997, risk_veto=False, parkinson_vol=1.53, garman_klass_vol=1.63
  [liquidity_engine] HOLD conf=0.000 available=True
      reason: HOLD: no sweep - price stayed within the prior 35,185.67-45,310.92 range or failed to reverse; CVD(3) +0.38, depth imbalance -0.09, absorption ratio 1.55
      features: bullish_sweep=False, bearish_sweep=False, prior_low=3.519e+04, prior_high=4.531e+04, cvd_norm_short=0.3754, depth_imbalance=-0.08517, absorption_ratio=1.553, cvd_is_proxy=False
  [sentiment_engine] BUY conf=0.236 available=True
      reason: BUY: social +0.74, news +0.73, funding +9.53bp/8h -> contrarian term -0.93 (extreme funding -> contrarian bearish); composite +0.236 using 3/3 components
      features: social_norm=0.7421, news_norm=0.7323, funding_rate=0.0009532, funding_term=-0.9331, composite=0.2361, components_available=3, weight_coverage=1, input_provenance=synthetic
  [regime_detector] BUY conf=1.000 available=True
      reason: BUY: regime 'trend_up' because ADX confirms a trend and the fast EMA is above the slow EMA; EMA_fast 39,969.12 > EMA_slow 32,236.58 (+23.99%), ADX 41.8 (trend >= 20), realized vol 120% (volatile >= 120%)
      features: regime=trend_up, ema_fast=3.997e+04, ema_slow=3.224e+04, adx=41.75, realized_vol=1.199, plus_di=29.01, minus_di=8.607
  weights (base x regime x family -> learned/effective):
      market_intel       base 1.00 learned 1.00 regime x1.40 family x1.00 -> eff 1.40; input +0.280 contrib +0.075
      risk_engine        base 1.00 learned 1.00 regime x1.00 family x1.00 -> eff 1.00; input +0.000 contrib +0.000
      liquidity_engine   base 1.00 learned 1.00 regime x0.90 family x1.00 -> eff 0.90; input +0.000 contrib +0.000
      sentiment_engine   base 1.00 learned 1.00 regime x1.10 family x1.00 -> eff 1.10; input +0.236 contrib +0.050
      regime_detector    base 1.00 learned 1.00 regime x0.80 family x1.00 -> eff 0.80; input +1.000 contrib +0.154
  regime=trend_up directional_score=+0.2792 agreement=1.000 coverage=0.635 data_coverage=1.000
  ensemble strength=0.350 (uncalibrated_ensemble_strength)
  risk_score=0.60 risk_veto=False kill_switch=False
  raw_action=BUY action=BUY forced_hold_rule=None
  analyst note: Regime trend_up (ADX 42, EMA gap +24.0%) (regime_detector BUY, contribution +0.154); OFI +0.14 and positive institutional-style momentum (+0.51) indicate buying pressure (market_intel BUY, contribution +0.075); bullish sentiment composite +0.24 (funding +9.53bp/8h) (sentiment_engine BUY, contribution +0.050); risk_score 0.60 is below the veto threshold 0.80; overall weighted direction is positive (S=+0.279) with strong agreement 1.00, coverage 0.63, ensemble strength 0.35 (uncalibrated) in regime 'trend_up'; decision: BUY.

== Sample trade ==
  LONG entry 2023-01-01 04:00:00+00:00 @ 45,937.12 (qty peak 0.7634)
  exit 2023-01-21 04:00:00+00:00 @ 35,673.35 after 120 bars
  gross PnL -7,834.99 (slippage inside fill prices: 31.15) fees 62.30 net PnL -7,897.29
  entry reason: BUY: raise exposure 0.00 -> 0.35 | Regime trend_up (ADX 42, EMA gap +24.0%) (regime_detector BUY, contribution +0.154); OFI +0.14 and positive institutional-style momentum (+0.51) indicate buying pressure (market_intel BUY, contribution +0.075); bullish sentiment composite +0.24 (funding +9.53bp/8h)
  exit reason: kill switch active: flatten existing position | OFI -0.41 and negative institutional-style momentum (-0.31) indicate selling pressure (market_intel SELL, contribution -0.067); regime trend_down (ADX 26, EMA gap -5.5%) (regime_detector SELL, contribution -0.050); risk_score 0.59 is below the veto thr

== Agent leaderboard (matured outcomes only; hit = direction of 20-bar forward return) ==
  agent              preds   dir   mat    hit    cw_ret standalone  incr(bp)      t  w_now  w_sugg
  market_intel        2190  1224  1208  63.2%   +0.0061     -0.087     +5.21  +1.99   1.00    1.04
  regime_detector     2190  1024  1024  57.0%   +0.0075     +0.624     +2.55  +0.57   1.00    1.00
  sentiment_engine    2190   503   493  66.9%   +0.0075     +0.262     -0.43  -0.21   1.00    1.00
  liquidity_engine    2190    75    74  62.2%   +0.0260     -0.051     -2.38  -0.98   1.00    1.00
  risk_engine         2190     0     0    n/a       n/a     +0.000     -4.96  -2.21   1.00    1.00
      note: insufficient matured directional samples (0 < 30)
  (standalone = non-compounded unit-position return sum after costs; incr = signal-level ablation)

== Execution-level ablation (full minus without-agent; in-sample, descriptive) ==
  removed agent       return w/o  sharpe w/o   d_return  d_sharpe
  market_intel            +2.94%        0.26     +3.23%     +0.17
  risk_engine             +8.44%        0.53     -2.27%     -0.10
  liquidity_engine        +8.05%        0.51     -1.88%     -0.08
  sentiment_engine        +8.46%        0.52     -2.29%     -0.09
  regime_detector         -1.43%       -0.29     +7.59%     +0.72

== Adversarial shadow agent (never votes; same timing/costs; ignores risk controls) ==
  disagreement observations 243, matured outcomes 243, counter hit rate 27.6%
  consensus book: final 15,818 maxDD 99.2% | counter book: final 65,553 maxDD 62.5%
  counter-consensus OUTPERFORMED consensus in this sample

== Forced-HOLD rule counts (a bar can trigger several) ==
  directional_score_floor=1677, kill_switch=300, min_confidence=1927, risk_veto=308

== Kill-switch events ==
  trigger@2023-01-21 00:00 (rolling Sharpe -2.66 < -1.00); recovery@2023-01-31 00:00 (cooldown elapsed; history reset); trigger@2023-03-28 04:00 (rolling Sharpe -1.23 < -1.00); recovery@2023-04-07 04:00 (cooldown elapsed; history reset); trigger@2023-06-16 00:00 (rolling Sharpe -1.29 < -1.00); recovery@2023-06-26 00:00 (cooldown elapsed; history reset); trigger@2023-09-25 08:00 (rolling Sharpe -1.84 < -1.00); recovery@2023-10-05 08:00 (cooldown elapsed; history reset); trigger@2023-11-08 08:00 (rolling Sharpe -1.11 < -1.00); recovery@2023-11-18 08:00 (cooldown elapsed; history reset)

== Benchmark comparison (same window, same fees/slippage, mark-to-market) ==
  name              return   sharpe    maxDD  trades       fees
  strategy          +6.17%     0.43   19.21%       5        383
  no_trade          +0.00%      n/a    0.00%       0          0
  buy_and_hold     -24.72%     0.28   68.91%       0        100
  ema_trend        -28.83%    -0.01   61.29%      14      2,081

== Strategy performance (net of fees and slippage) ==
  initial equity      100,000.00
  final equity        106,167.63
  total net return    +6.17%
  CAGR                +6.17%
  Sharpe (annualized) 0.43
  max drawdown        19.21%
  closed trades       5
  win rate            40.0%
  profit factor       1.70
  turnover            3.97x mean equity
  total fees          383.08
  est. slippage       191.54
  avg holding period  97.6 bars (16.3 days)
  avg |exposure|      10.05%  (long 10.05%, short 0.00%)
  bars                2190

$ python main.py --mode backtest --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01 --synthetic --walk-forward --log-level WARNING   # stderr kill-switch warnings omitted
Data source: synthetic(seed=42) | 2190 tradable bars (UTC, end exclusive)

NOTE: synthetic data - out-of-sample here means unseen *synthetic* bars, not evidence of real alpha.

== Walk-forward folds (IS = train with feedback, VAL = threshold selection, OOS = frozen test) ==
  fold        train         val        test    IS ret  VAL ret  OOS ret  B&H OOS min_conf w_upd  Brier cal/base
     0    250-750     770-870     890-990      -6.42%   +0.00%   +0.00%   +7.17%     0.25     4             n/a
     1    350-850     870-970     990-1090     +1.82%   +0.00%   +0.00%  +37.03%     0.25    22             n/a
     2    450-950     970-1070   1090-1190     +1.78%   +0.00%   +0.00%   -6.43%     0.25    22             n/a
     3    550-1050   1070-1170   1190-1290     +1.75%   +0.00%   -5.62%  -10.10%     0.25    18     0.250/0.257
     4    650-1150   1170-1270   1290-1390     -0.91%   -2.04%   +0.00%   -1.61%     0.25    15             n/a
     5    750-1250   1270-1370   1390-1490     -0.48%   +0.00%   +0.00%  -15.56%     0.25     4             n/a
     6    850-1350   1370-1470   1490-1590     -0.61%   +0.00%   +0.00%   +4.65%     0.25     0             n/a
     7    950-1450   1470-1570   1590-1690     -0.61%   +0.00%   +0.00%  -16.69%     0.25     7             n/a
     8   1050-1550   1570-1670   1690-1790     -0.61%   +0.00%   +0.00%   -7.04%     0.25     0             n/a
     9   1150-1650   1670-1770   1790-1890     -3.35%   +0.00%   -0.30%   +4.22%     0.25     0             n/a
    10   1250-1750   1770-1870   1890-1990     -7.16%   -1.79%  +21.80%  +63.77%     0.35     0             n/a
    11   1350-1850   1870-1970   1990-2090     +0.00%  +16.61%   -2.30%   -5.64%     0.25     3             n/a
    12   1450-1950   1970-2070   2090-2190     +2.08%   +5.73%   +0.00%   +7.93%     0.25     7             n/a
    13   1550-2050   2070-2170   2190-2290    +22.11%   +0.00%   +0.00%  -13.02%     0.25     7     0.596/0.631
    14   1650-2150   2170-2270   2290-2390    +13.98%   +0.00%   +0.00%   -6.05%     0.25    10     0.209/0.231
  mean in-sample (train) return +1.56%; mean validation return +1.23%
  learned weights after last training window: market_intel=1.11, risk_engine=1.00, liquidity_engine=1.00, sentiment_engine=0.94, regime_detector=1.09
== Aggregated OUT-OF-SAMPLE (chained test windows) ==
  initial equity      100,000.00
  final equity        111,977.25
  total net return    +11.98%
  CAGR                +18.16%
  Sharpe (annualized) 1.13
  max drawdown        14.11%
  closed trades       0
  win rate            n/a
  profit factor       n/a
  turnover            1.55x mean equity
  total fees          160.55
  est. slippage       80.24
  avg holding period  n/a
  avg |exposure|      8.27%  (long 8.27%, short 0.00%)
  bars                1486
  note[win_rate]: unavailable: no closed trades
  note[profit_factor]: unavailable: no closed trades
  note[avg_holding]: unavailable: no closed trades
  buy-and-hold over the same chained test windows: return +18.14%, Sharpe 0.78, maxDD 54.97%

$ python main.py --mode paper --symbol BTCUSDT --synthetic
2026-10-12 06:29:33,084 INFO data.feed: synthetic feed: symbol=BTCUSDT seed=42 bars=2440 (warm-up 250, tradable 2190) 2022-11-20 08:00:00+00:00..2023-12-31 20:00:00+00:00
2026-10-12 06:29:33,152 INFO execution.executor: PAPER ORDER (simulated, no broker) decided 2023-01-01 00:00:00+00:00 -> target exposure 0.350 | BUY: raise exposure 0.00 -> 0.35 | Regime trend_up (ADX 42, EMA gap +24.0%) (regime_detector BUY, contribution +0.154); OFI +0.14 and positive institutional-sty
2026-10-12 06:29:33,153 INFO execution.executor: PAPER FILL (simulated) 2023-01-01 04:00:00+00:00 BUY 0.763364 @ 45937.12 (ref 45914.16) fee 35.07
2026-10-12 06:29:33,189 WARNING core.portfolio_risk: kill switch TRIGGERED at 2023-01-21 00:00:00+00:00: rolling Sharpe -2.66 < -1.00
2026-10-12 06:29:33,190 INFO execution.executor: PAPER ORDER (simulated, no broker) decided 2023-01-21 00:00:00+00:00 -> target exposure 0.000 | kill switch active: flatten existing position | OFI -0.41 and negative institutional-style momentum (-0.31) indicate selling pressure (market_intel SELL, contri
2026-10-12 06:29:33,190 INFO execution.executor: PAPER FILL (simulated) 2023-01-21 04:00:00+00:00 SELL 0.763364 @ 35673.35 (ref 35691.20) fee 27.23
2026-10-12 06:29:33,230 WARNING core.portfolio_risk: kill switch RECOVERED at 2023-01-31 00:00:00+00:00 (cooldown elapsed)
PAPER MODE (simulated fills, NO broker connection, no real orders) | source synthetic(seed=42) | bars 300

  simulated fills: 2; closed trades: 1; unfilled at end: 0

  SIM FILL 2023-01-01 04:00:00+00:00 BUY 0.7634 @ 45,937.12 fee 35.07

  SIM FILL 2023-01-21 04:00:00+00:00 SELL 0.7634 @ 35,673.35 fee 27.23

== Paper session performance ==
  initial equity      100,000.00
  final equity        92,102.71
  total net return    -7.90%
  CAGR                n/a
  Sharpe (annualized) -1.70
  max drawdown        19.21%
  closed trades       1
  win rate            0.0%
  profit factor       0.00
  turnover            0.66x mean equity
  total fees          62.30
  est. slippage       31.15
  avg holding period  120.0 bars (20.0 days)
  avg |exposure|      13.17%  (long 13.17%, short 0.00%)
  bars                300
  note[cagr]: unavailable: window 0.14y < 0.25y

$ python main.py --mode backtest --symbol BTCUSDT --live-feed
LIVE FEED UNAVAILABLE: live feed for BTCUSDT is a stub: no market-data provider is connected. Implement data.feed.LiveFeedStub.get_bars (or a new MarketDataProvider) or run with --synthetic. The framework never falls back to synthetic data silently.
(exit code 2)

$ python -m unittest discover -s tests -v
test_costs_applied_equally (test_adversarial_agent.TestAdversarial.test_costs_applied_equally) ... ok
test_hold_is_not_an_observation (test_adversarial_agent.TestAdversarial.test_hold_is_not_an_observation) ... ok
test_no_same_bar_execution (test_adversarial_agent.TestAdversarial.test_no_same_bar_execution) ... ok
test_takes_opposite_side_and_pnl (test_adversarial_agent.TestAdversarial.test_takes_opposite_side_and_pnl) ... ok
test_benchmarks_same_window (test_backtester.TestBacktestRun.test_benchmarks_same_window) ... ok
test_equity_consistency (test_backtester.TestBacktestRun.test_equity_consistency) ... ok
test_fills_at_next_bar_open_with_costs (test_backtester.TestBacktestRun.test_fills_at_next_bar_open_with_costs) ... ok
test_long_only_never_short (test_backtester.TestBacktestRun.test_long_only_never_short) ... ok
test_orders_reference_previous_decision (test_backtester.TestBacktestRun.test_orders_reference_previous_decision) ... ok
test_fill_only_after_decision_bar (test_backtester.TestExecutor.test_fill_only_after_decision_bar) ... ok
test_paper_never_routes_live (test_backtester.TestExecutor.test_paper_never_routes_live) ... ok
test_unfilled_at_end (test_backtester.TestExecutor.test_unfilled_at_end) ... ok
test_cooldown_recovery (test_backtester.TestKillSwitch.test_cooldown_recovery) ... ok
test_manual_stays_active (test_backtester.TestKillSwitch.test_manual_stays_active) ... ok
test_min_history_then_drawdown_trigger (test_backtester.TestKillSwitch.test_min_history_then_drawdown_trigger) ... ok
test_sharpe_trigger (test_backtester.TestKillSwitch.test_sharpe_trigger) ... ok
test_buy_then_sell_hand_calculated (test_backtester.TestPortfolioAccounting.test_buy_then_sell_hand_calculated) ... ok
test_equity_identity (test_backtester.TestPortfolioAccounting.test_equity_identity) ... ok
test_no_unintended_leverage (test_backtester.TestPortfolioAccounting.test_no_unintended_leverage) ... ok
test_partial_exit_realizes_proportionally (test_backtester.TestPortfolioAccounting.test_partial_exit_realizes_proportionally) ... ok
test_reversal_two_legs_both_charged (test_backtester.TestPortfolioAccounting.test_reversal_two_legs_both_charged) ... ok
test_short_rejected_when_long_only (test_backtester.TestPortfolioAccounting.test_short_rejected_when_long_only) ... ok
test_buy_sizing_bounded (test_backtester.TestPositionManagement.test_buy_sizing_bounded) ... ok
test_hold_keeps_position (test_backtester.TestPositionManagement.test_hold_keeps_position) ... ok
test_kill_switch_flatten_and_hold_policies (test_backtester.TestPositionManagement.test_kill_switch_flatten_and_hold_policies) ... ok
test_long_only_sell_closes_without_short (test_backtester.TestPositionManagement.test_long_only_sell_closes_without_short) ... ok
test_risk_veto_reduces_existing (test_backtester.TestPositionManagement.test_risk_veto_reduces_existing) ... ok
test_sell_reduce_policy (test_backtester.TestPositionManagement.test_sell_reduce_policy) ... ok
test_volatility_aware_sizing (test_backtester.TestPositionManagement.test_volatility_aware_sizing) ... ok
test_data_coverage_rule (test_boss_orchestrator.TestBoss.test_data_coverage_rule) ... ok
test_disabled_agent_rejected (test_boss_orchestrator.TestBoss.test_disabled_agent_rejected) ... ok
test_disagreement_reduces_agreement (test_boss_orchestrator.TestBoss.test_disagreement_reduces_agreement) ... ok
test_family_normalization (test_boss_orchestrator.TestBoss.test_family_normalization) ... ok
test_floors (test_boss_orchestrator.TestBoss.test_floors) ... ok
test_kalman_smoothing_responds_to_reversal (test_boss_orchestrator.TestBoss.test_kalman_smoothing_responds_to_reversal) ... ok
test_kill_switch_forces_hold (test_boss_orchestrator.TestBoss.test_kill_switch_forces_hold) ... ok
test_learned_weights_clipped (test_boss_orchestrator.TestBoss.test_learned_weights_clipped) ... ok
test_reasoner_mentions_actual_values (test_boss_orchestrator.TestBoss.test_reasoner_mentions_actual_values) ... ok
test_regime_weights_applied (test_boss_orchestrator.TestBoss.test_regime_weights_applied) ... ok
test_risk_unavailable_fails_safe (test_boss_orchestrator.TestBoss.test_risk_unavailable_fails_safe) ... ok
test_risk_veto_overrides_buy (test_boss_orchestrator.TestBoss.test_risk_veto_overrides_buy) ... ok
test_unavailable_agent_excluded_not_confirming (test_boss_orchestrator.TestBoss.test_unavailable_agent_excluded_not_confirming) ... ok
test_weighted_score_hand_calculated (test_boss_orchestrator.TestBoss.test_weighted_score_hand_calculated) ... ok
test_fit_monotone (test_evaluator.TestCalibrator.test_fit_monotone) ... ok
test_insufficient (test_evaluator.TestCalibrator.test_insufficient) ... ok
test_drawdown (test_evaluator.TestMetrics.test_drawdown) ... ok
test_no_losses_profit_factor_unavailable (test_evaluator.TestMetrics.test_no_losses_profit_factor_unavailable) ... ok
test_zero_variance_and_no_trades (test_evaluator.TestMetrics.test_zero_variance_and_no_trades) ... ok
test_future_prices_do_not_matter (test_evaluator.TestScoreAgents.test_future_prices_do_not_matter) ... ok
test_hold_excluded_from_hit_rate (test_evaluator.TestScoreAgents.test_hold_excluded_from_hit_rate) ... ok
test_only_matured_outcomes (test_evaluator.TestScoreAgents.test_only_matured_outcomes) ... ok
test_standalone_pnl_with_costs (test_evaluator.TestScoreAgents.test_standalone_pnl_with_costs) ... ok
test_unavailable_excluded (test_evaluator.TestScoreAgents.test_unavailable_excluded) ... ok
test_wrong_direction_misses (test_evaluator.TestScoreAgents.test_wrong_direction_misses) ... ok
test_insufficient_samples_keep_weight (test_evaluator.TestSuggestWeights.test_insufficient_samples_keep_weight) ... ok
test_negative_evidence_decreases_bounded (test_evaluator.TestSuggestWeights.test_negative_evidence_decreases_bounded) ... ok
test_neutral_evidence_shrinks_to_base (test_evaluator.TestSuggestWeights.test_neutral_evidence_shrinks_to_base) ... ok
test_positive_evidence_increases_bounded (test_evaluator.TestSuggestWeights.test_positive_evidence_increases_bounded) ... ok
test_frozen_never_updates (test_feedback.TestFeedback.test_frozen_never_updates) ... ok
test_ignores_records_before_window (test_feedback.TestFeedback.test_ignores_records_before_window) ... ok
test_only_on_interval (test_feedback.TestFeedback.test_only_on_interval) ... ok
test_uses_only_matured_data (test_feedback.TestFeedback.test_uses_only_matured_data) ... ok
test_weights_bounded_and_logged (test_feedback.TestFeedback.test_weights_bounded_and_logged) ... ok
test_stable_reversal_missing_bounds (test_integration.TestKalman.test_stable_reversal_missing_bounds) ... ok
test_all_agents_analyze_same_state (test_integration.TestPipeline.test_all_agents_analyze_same_state) ... ok
test_features_are_causal (test_integration.TestPipeline.test_features_are_causal) ... ok
test_frozen_run_has_no_weight_updates (test_integration.TestPipeline.test_frozen_run_has_no_weight_updates) ... ok
test_missing_order_flow_disables_ofi (test_integration.TestPipeline.test_missing_order_flow_disables_ofi) ... ok
test_missing_sentiment_feed_is_unavailable_not_neutral (test_integration.TestPipeline.test_missing_sentiment_feed_is_unavailable_not_neutral) ... ok
test_optional_agent_flag (test_integration.TestPipeline.test_optional_agent_flag) ... ok
test_optional_agent_unavailable_without_data (test_integration.TestPipeline.test_optional_agent_unavailable_without_data) ... ok
test_reproducible_with_seed (test_integration.TestPipeline.test_reproducible_with_seed) ... ok
test_synthetic_labelled (test_integration.TestPipeline.test_synthetic_labelled) ... ok
test_config_errors (test_integration.TestValidation.test_config_errors) ... ok
test_duplicates (test_integration.TestValidation.test_duplicates) ... ok
test_insufficient_history (test_integration.TestValidation.test_insufficient_history) ... ok
test_invalid_date_range (test_integration.TestValidation.test_invalid_date_range) ... ok
test_live_feed_stub_never_falls_back (test_integration.TestValidation.test_live_feed_stub_never_falls_back) ... ok
test_missing_and_invalid_prices (test_integration.TestValidation.test_missing_and_invalid_prices) ... ok
test_naive_timestamps (test_integration.TestValidation.test_naive_timestamps) ... ok
test_valid (test_integration.TestValidation.test_valid) ... ok
test_folds_chronological_with_embargo (test_integration.TestWalkForward.test_folds_chronological_with_embargo) ... ok
test_too_short_raises (test_integration.TestWalkForward.test_too_short_raises) ... ok
test_walk_forward_runs_and_separates_results (test_integration.TestWalkForward.test_walk_forward_runs_and_separates_results) ... ok
test_hand_calculated (test_liquidity.TestAbsorption.test_hand_calculated) ... ok
test_zero_baseline_volume_nan (test_liquidity.TestAbsorption.test_zero_baseline_volume_nan) ... ok
test_zero_range_capped (test_liquidity.TestAbsorption.test_zero_range_capped) ... ok
test_negative_raises (test_liquidity.TestDepth.test_negative_raises) ... ok
test_scalar (test_liquidity.TestDepth.test_scalar) ... ok
test_series (test_liquidity.TestDepth.test_series) ... ok
test_bearish_confirmed (test_liquidity.TestSweeps.test_bearish_confirmed) ... ok
test_bullish_confirmed (test_liquidity.TestSweeps.test_bullish_confirmed) ... ok
test_no_reversal_no_sweep (test_liquidity.TestSweeps.test_no_reversal_no_sweep) ... ok
test_prior_level_excludes_current_and_is_causal (test_liquidity.TestSweeps.test_prior_level_excludes_current_and_is_causal) ... ok
test_warmup_false (test_liquidity.TestSweeps.test_warmup_false) ... ok
test_absorption_adds_confidence (test_liquidity_engine.TestLiquidityEngine.test_absorption_adds_confidence) ... ok
test_bearish_confirmed (test_liquidity_engine.TestLiquidityEngine.test_bearish_confirmed) ... ok
test_bullish_confirmed_by_cvd_and_depth (test_liquidity_engine.TestLiquidityEngine.test_bullish_confirmed_by_cvd_and_depth) ... ok
test_bullish_cvd_divergence_hold (test_liquidity_engine.TestLiquidityEngine.test_bullish_cvd_divergence_hold) ... ok
test_bullish_without_flow_data_unconfirmed (test_liquidity_engine.TestLiquidityEngine.test_bullish_without_flow_data_unconfirmed) ... ok
test_no_sweep (test_liquidity_engine.TestLiquidityEngine.test_no_sweep) ... ok
test_two_sided_ambiguous (test_liquidity_engine.TestLiquidityEngine.test_two_sided_ambiguous) ... ok
test_warmup_unavailable (test_liquidity_engine.TestLiquidityEngine.test_warmup_unavailable) ... ok
test_deterministic (test_market_intel.TestMarketIntel.test_deterministic) ... ok
test_feed_not_supplied (test_market_intel.TestMarketIntel.test_feed_not_supplied) ... ok
test_missing_input_unavailable (test_market_intel.TestMarketIntel.test_missing_input_unavailable) ... ok
test_negative_flow_sell (test_market_intel.TestMarketIntel.test_negative_flow_sell) ... ok
test_neutral_hold_explained (test_market_intel.TestMarketIntel.test_neutral_hold_explained) ... ok
test_positive_flow_buy (test_market_intel.TestMarketIntel.test_positive_flow_buy) ... ok
test_invalid_side (test_order_flow.TestCVD.test_invalid_side) ... ok
test_normalized_change (test_order_flow.TestCVD.test_normalized_change) ... ok
test_tick_rule_proxy (test_order_flow.TestCVD.test_tick_rule_proxy) ... ok
test_with_side (test_order_flow.TestCVD.test_with_side) ... ok
test_constant_price_is_zero (test_order_flow.TestInstitutionalMomentum.test_constant_price_is_zero) ... ok
test_insufficient_history (test_order_flow.TestInstitutionalMomentum.test_insufficient_history) ... ok
test_matches_formula (test_order_flow.TestInstitutionalMomentum.test_matches_formula) ... ok
test_reproducible (test_order_flow.TestInstitutionalMomentum.test_reproducible) ... ok
test_zero_prior_volume_nan (test_order_flow.TestInstitutionalMomentum.test_zero_prior_volume_nan) ... ok
test_zero_variance_trend (test_order_flow.TestInstitutionalMomentum.test_zero_variance_trend) ... ok
test_causal (test_order_flow.TestOFI.test_causal) ... ok
test_hand_calculated (test_order_flow.TestOFI.test_hand_calculated) ... ok
test_invalid_inputs (test_order_flow.TestOFI.test_invalid_inputs) ... ok
test_missing_value_propagates_nan (test_order_flow.TestOFI.test_missing_value_propagates_nan) ... ok
test_sign_convention_and_bounds (test_order_flow.TestOFI.test_sign_convention_and_bounds) ... ok
test_zero_volume_is_nan (test_order_flow.TestOFI.test_zero_volume_is_nan) ... ok
test_constant_is_nan (test_order_flow.TestPVCorr.test_constant_is_nan) ... ok
test_perfect_positive (test_order_flow.TestPVCorr.test_perfect_positive) ... ok
test_range (test_regime_detector.TestRegimeDetector.test_range) ... ok
test_trend_down (test_regime_detector.TestRegimeDetector.test_trend_down) ... ok
test_trend_up (test_regime_detector.TestRegimeDetector.test_trend_up) ... ok
test_volatile_overrides_trend (test_regime_detector.TestRegimeDetector.test_volatile_overrides_trend) ... ok
test_warmup_unknown (test_regime_detector.TestRegimeDetector.test_warmup_unknown) ... ok
test_extreme_risk_veto (test_risk_engine.TestRiskEngine.test_extreme_risk_veto) ... ok
test_hand_calculated_components (test_risk_engine.TestRiskEngine.test_hand_calculated_components) ... ok
test_low_risk (test_risk_engine.TestRiskEngine.test_low_risk) ... ok
test_missing_fails_safe (test_risk_engine.TestRiskEngine.test_missing_fails_safe) ... ok
test_all_missing_nan (test_sentiment.TestComposite.test_all_missing_nan) ... ok
test_component_directions (test_sentiment.TestComposite.test_component_directions) ... ok
test_hand_calculated_with_missing (test_sentiment.TestComposite.test_hand_calculated_with_missing) ... ok
test_invalid_weights (test_sentiment.TestComposite.test_invalid_weights) ... ok
test_inside_threshold_zero (test_sentiment.TestFunding.test_inside_threshold_zero) ... ok
test_invalid (test_sentiment.TestFunding.test_invalid) ... ok
test_missing (test_sentiment.TestFunding.test_missing) ... ok
test_negative_extreme_is_bullish_and_saturates (test_sentiment.TestFunding.test_negative_extreme_is_bullish_and_saturates) ... ok
test_positive_is_bearish (test_sentiment.TestFunding.test_positive_is_bearish) ... ok
test_missing_and_invalid (test_sentiment.TestNormalize.test_missing_and_invalid) ... ok
test_values (test_sentiment.TestNormalize.test_values) ... ok
test_extreme_negative_funding_contrarian_buy (test_sentiment_engine.TestSentimentEngine.test_extreme_negative_funding_contrarian_buy) ... ok
test_extreme_positive_funding_contrarian_sell (test_sentiment_engine.TestSentimentEngine.test_extreme_positive_funding_contrarian_sell) ... ok
test_missing_feed_not_neutral (test_sentiment_engine.TestSentimentEngine.test_missing_feed_not_neutral) ... ok
test_negative_news_social_sell (test_sentiment_engine.TestSentimentEngine.test_negative_news_social_sell) ... ok
test_positive_news_social_buy (test_sentiment_engine.TestSentimentEngine.test_positive_news_social_buy) ... ok
test_too_few_components_unavailable (test_sentiment_engine.TestSentimentEngine.test_too_few_components_unavailable) ... ok
test_true_range (test_volatility.TestATR.test_true_range) ... ok
test_wilder_hand_calculated (test_volatility.TestATR.test_wilder_hand_calculated) ... ok
test_wilder_insufficient (test_volatility.TestATR.test_wilder_insufficient) ... ok
test_drawdown (test_volatility.TestDrawdownAndValidation.test_drawdown) ... ok
test_validate_ohlc (test_volatility.TestDrawdownAndValidation.test_validate_ohlc) ... ok
test_garman_klass (test_volatility.TestRangeEstimators.test_garman_klass) ... ok
test_parkinson (test_volatility.TestRangeEstimators.test_parkinson) ... ok
test_zero_range_is_zero (test_volatility.TestRangeEstimators.test_zero_range_is_zero) ... ok
test_annualization (test_volatility.TestRealizedVol.test_annualization) ... ok
test_constant_zero_variance (test_volatility.TestRealizedVol.test_constant_zero_variance) ... ok
test_hand_calculated (test_volatility.TestRealizedVol.test_hand_calculated) ... ok
test_invalid (test_volatility.TestRealizedVol.test_invalid) ... ok
test_adx_flat_zero (test_volatility.TestTrend.test_adx_flat_zero) ... ok
test_adx_strong_uptrend (test_volatility.TestTrend.test_adx_strong_uptrend) ... ok
test_ema_warmup_and_constant (test_volatility.TestTrend.test_ema_warmup_and_constant) ... ok

----------------------------------------------------------------------
Ran 168 tests in 5.238s

OK
```


## Deliverable 6 — Unit and integration tests

Run: `python -m unittest discover -s tests -v` (168 tests, ~5 s).

| File | Verifies |
|---|---|
| test_order_flow / test_volatility / test_liquidity / test_sentiment | hand-calculated indicator values, NaN/zero-volume/zero-variance/insufficient-history/invalid-input handling, causality, ADX/EMA warm-up |
| test_market_intel / test_risk_engine / test_liquidity_engine / test_sentiment_engine / test_regime_detector | agents on hand-crafted states: each signal direction, documented confidence formulas, explicit unavailable HOLDs |
| test_boss_orchestrator | S/agreement/coverage/strength formulas, risk veto & fail-safe, floors, kill switch, unavailable/disabled agents, regime & family weights, Kalman reversal, reasoner content |
| test_adversarial_agent | opposite-side shadow PnL, equal costs, no same-bar fills, observation vs matured counts |
| test_backtester | fill price/fee hand calc, partial exits, reversals, no leverage, equity identity, next-open timing, long-only SELL, veto/kill-switch position policies, kill-switch trigger/recovery, benchmarks on the same window |
| test_evaluator | matured-only scoring, future prices ignored, HOLD excluded from hit rate, standalone PnL, bounded/shrunk weight suggestions, metric edge cases, calibration |
| test_feedback | interval, frozen mode, no future data, bounds, logged evidence |
| test_integration | full pipeline, feature causality via truncation, seed reproducibility, missing feeds -> unavailable, optional agent flag, walk-forward chronology/embargo/separation, data & config validation, live stub, Kalman |

### `tests/helpers.py`

```python
"""Shared test fixtures: hand-crafted market states and small configs."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import FrameworkConfig, load_config  # noqa: E402
from core.models import RAW_FIELDS, FeatureStatus, MarketState  # noqa: E402

TS = datetime(2023, 6, 1, tzinfo=timezone.utc)
logging.basicConfig(level=logging.ERROR)  # keep test output readable
CONFIG_PATH = os.path.join(ROOT, "config.yaml")


def config(**overrides: dict) -> FrameworkConfig:
    """Load the project config with nested overrides."""
    return load_config(CONFIG_PATH, overrides or None)


def make_state(
    indicators: dict[str, float] | None = None,
    raw: dict[str, float | None] | None = None,
    unavailable: tuple[str, ...] = (),
    close: float = 100.0,
    bar_index: int = 500,
) -> MarketState:
    """Build a MarketState; every supplied feature is valid unless listed in ``unavailable``."""
    ind = dict(indicators or {})
    r = {"social_sentiment": 50.0, "news_sentiment": 0.0, "funding_rate": 0.0001,
         "bid_volume": 500.0, "ask_volume": 500.0, "trade_side": 0.0, "bid_depth": 1000.0, "ask_depth": 1000.0}
    r.update(raw or {})
    av: dict[str, FeatureStatus] = {}
    for k in RAW_FIELDS:
        av[k] = FeatureStatus.SYNTHETIC
    for k in ind:
        av[k] = FeatureStatus.DERIVED
    for k in unavailable:
        av[k] = FeatureStatus.UNAVAILABLE
        if k in r:
            r[k] = None
    return MarketState(
        timestamp=TS, symbol="TEST", open=close, high=close * 1.01, low=close * 0.99, close=close, volume=1000.0,
        bid_volume=r["bid_volume"], ask_volume=r["ask_volume"], trade_side=r["trade_side"],
        bid_depth=r["bid_depth"], ask_depth=r["ask_depth"], social_sentiment=r["social_sentiment"],
        news_sentiment=r["news_sentiment"], funding_rate=r["funding_rate"],
        indicators=ind, availability=av, bar_index=bar_index,
    )
```

### `tests/test_adversarial_agent.py`

```python
"""AdversarialAgent shadow accounting."""

import unittest

from helpers import TS

import numpy as np

from agents.adversarial_agent import AdversarialAgent


class TestAdversarial(unittest.TestCase):
    def test_takes_opposite_side_and_pnl(self):
        a = AdversarialAgent(10_000, fee_bps=0, slippage_bps=0, exposure=1.0, horizon=1)
        a.observe(0, +1)                     # consensus BUY at bar 0 close
        a.on_open(1, TS, 100.0)              # filled at bar 1 open
        self.assertGreater(a.consensus.units, 0)
        self.assertLess(a.counter.units, 0)
        a.on_close(90.0)                     # price falls 10%
        self.assertAlmostEqual(a.consensus_equity[-1], 9_000.0)
        self.assertAlmostEqual(a.counter_equity[-1], 11_000.0)
        rep = a.report(np.array([100.0, 90.0]), as_of_index=1)
        self.assertTrue(rep.counter_outperformed)
        self.assertEqual(rep.observations, 1)
        self.assertEqual(rep.matured_outcomes, 1)
        self.assertEqual(rep.counter_hit_rate, 1.0)

    def test_costs_applied_equally(self):
        a = AdversarialAgent(10_000, fee_bps=10, slippage_bps=5, exposure=1.0)
        a.observe(0, -1)
        a.on_open(1, TS, 100.0)
        a.on_close(100.0)
        self.assertLess(a.consensus_equity[-1], 10_000)
        self.assertLess(a.counter_equity[-1], 10_000)
        # long side is cash-constrained for costs, so notional differs by ~costs only
        self.assertAlmostEqual(a.consensus.total_fees, a.counter.total_fees, delta=0.05)

    def test_no_same_bar_execution(self):
        a = AdversarialAgent(10_000, 0, 0)
        a.observe(5, 1)
        with self.assertRaises(RuntimeError):
            a.on_open(5, TS, 100.0)

    def test_hold_is_not_an_observation(self):
        a = AdversarialAgent(10_000, 0, 0, horizon=20)
        a.observe(0, 0)
        a.observe(1, 1)
        rep = a.report(np.ones(10), as_of_index=9)
        self.assertEqual(rep.observations, 1)
        self.assertEqual(rep.matured_outcomes, 0)  # 1 + 20 > 9: not matured


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_backtester.py`

```python
"""Accounting, execution timing, position management and kill switch."""

import unittest
from datetime import timedelta

from helpers import TS, config, make_state

from core.models import BossDecision, Regime, SignalType
from core.portfolio_risk import KillSwitch
from data.feed import SyntheticFeed, parse_utc_date
from core.feature_bus import compute_features
from execution.backtester import Backtester, run_benchmarks, target_from_decision
from execution.executor import NullBroker, PaperExecutor, Portfolio, SimulatedExecutor
from core.models import Order


def decision(action=SignalType.BUY, strength=0.5, veto=False, kill=False):
    return BossDecision(action, action, strength, 0.3, 1.0, 0.5, 1.0, "test", [], Regime.RANGE, 0.3, veto, kill, None, TS)


class TestPortfolioAccounting(unittest.TestCase):
    def test_buy_then_sell_hand_calculated(self):
        p = Portfolio(10_000, fee_bps=10, slippage_bps=5)
        f = p.rebalance_to(0.5, 100.0, TS, 1, "entry")[0]
        self.assertAlmostEqual(f.exec_price, 100.05)
        self.assertAlmostEqual(f.quantity, 50.0)
        self.assertAlmostEqual(f.fee, 50 * 100.05 * 0.001)
        self.assertAlmostEqual(p.cash, 10_000 - 5002.5 - 5.0025)
        self.assertAlmostEqual(p.equity(100.0), 10_000 - 2.5 - 5.0025)
        p.rebalance_to(0.0, 110.0, TS + timedelta(hours=4), 2, "exit")
        t = p.trades[0]
        self.assertAlmostEqual(t.exit_price, 110 * 0.9995)
        self.assertAlmostEqual(t.gross_pnl, 50 * (109.945 - 100.05))
        self.assertAlmostEqual(t.fees, 5.0025 + 50 * 109.945 * 0.001)
        self.assertAlmostEqual(p.equity(110.0), 10_000 + t.net_pnl)
        self.assertEqual(p.units, 0.0)

    def test_no_unintended_leverage(self):
        p = Portfolio(10_000, fee_bps=10, slippage_bps=5)
        p.rebalance_to(1.0, 100.0, TS, 1, "all-in")
        self.assertGreaterEqual(p.cash, -1e-9)
        self.assertLessEqual(p.exposure(100.0), 1.0 + 1e-9)

    def test_partial_exit_realizes_proportionally(self):
        p = Portfolio(10_000, 0, 0)
        p.rebalance_to(1.0, 100.0, TS, 1, "in")
        p.rebalance_to(0.5, 120.0, TS, 2, "half")  # equity 12000 -> keep 50 units (6000/120)
        self.assertAlmostEqual(p.units, 50.0)
        self.assertAlmostEqual(p.open_trade.gross_pnl, 50 * 20)
        self.assertEqual(len(p.trades), 0)

    def test_reversal_two_legs_both_charged(self):
        p = Portfolio(10_000, 10, 0, allow_short=True)
        p.rebalance_to(0.5, 100.0, TS, 1, "long")
        fills = p.rebalance_to(-0.5, 100.0, TS, 2, "reverse")
        self.assertEqual(len(fills), 2)
        self.assertTrue(all(f.fee > 0 for f in fills))
        self.assertEqual(len(p.trades), 1)
        self.assertLess(p.units, 0)

    def test_short_rejected_when_long_only(self):
        with self.assertRaises(ValueError):
            Portfolio(10_000, 0, 0).rebalance_to(-0.5, 100.0, TS, 1, "short")

    def test_equity_identity(self):
        p = Portfolio(10_000, 10, 5)
        p.rebalance_to(0.8, 100.0, TS, 1, "a")
        p.rebalance_to(0.3, 105.0, TS, 2, "b")
        p.rebalance_to(0.9, 98.0, TS, 3, "c")
        price = 101.0
        realized = sum(t.net_pnl for t in p.trades) + p.open_trade.gross_pnl - p.open_trade.fees
        self.assertAlmostEqual(p.equity(price) - 10_000, realized + p.unrealized_pnl(price), places=6)


class TestExecutor(unittest.TestCase):
    def test_fill_only_after_decision_bar(self):
        ex = SimulatedExecutor(Portfolio(10_000, 0, 0))
        ex.submit(Order(TS, 5, 0.5, "x"))
        with self.assertRaises(RuntimeError):
            ex.on_bar_open(5, TS, 100.0)
        fills = ex.on_bar_open(6, TS, 100.0)
        self.assertEqual(fills[0].bar_index, 6)

    def test_unfilled_at_end(self):
        ex = SimulatedExecutor(Portfolio(10_000, 0, 0))
        ex.submit(Order(TS, 9, 0.5, "x"))
        ex.finalize()
        self.assertEqual(len(ex.unfilled), 1)
        self.assertEqual(ex.portfolio.units, 0)

    def test_paper_never_routes_live(self):
        with self.assertRaises(RuntimeError):
            PaperExecutor(Portfolio(10_000, 0, 0), live_orders_enabled=True)
        with self.assertRaises(RuntimeError):
            NullBroker().submit_order(Order(TS, 0, 1.0, "x"), 1.0, None)


class TestPositionManagement(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.state = make_state({"realized_vol": 0.5})

    def test_buy_sizing_bounded(self):
        t, _ = target_from_decision(decision(SignalType.BUY, 0.6), 0.0, self.state, self.cfg, "flatten")
        self.assertAlmostEqual(t, min(self.cfg.execution.max_position_size, 0.6 * self.cfg.execution.exposure_cap))

    def test_long_only_sell_closes_without_short(self):
        t, why = target_from_decision(decision(SignalType.SELL), 0.6, self.state, self.cfg, "flatten")
        self.assertEqual(t, 0.0)
        t2, why2 = target_from_decision(decision(SignalType.SELL), 0.0, self.state, self.cfg, "flatten")
        self.assertIsNone(t2)
        self.assertIn("does not open shorts", why2)

    def test_sell_reduce_policy(self):
        cfg = config(strategy={"sell_policy": "reduce", "sell_reduce_fraction": 0.5})
        t, _ = target_from_decision(decision(SignalType.SELL), 0.8, self.state, cfg, "flatten")
        self.assertAlmostEqual(t, 0.4)

    def test_hold_keeps_position(self):
        t, _ = target_from_decision(decision(SignalType.HOLD), 0.7, self.state, self.cfg, "flatten")
        self.assertIsNone(t)

    def test_risk_veto_reduces_existing(self):
        t, why = target_from_decision(decision(SignalType.HOLD, veto=True), 0.9, self.state, self.cfg, "flatten")
        self.assertAlmostEqual(t, self.cfg.strategy.veto_exposure_cap)
        self.assertIn("risk veto", why)

    def test_kill_switch_flatten_and_hold_policies(self):
        t, _ = target_from_decision(decision(SignalType.HOLD, kill=True), 0.5, self.state, self.cfg, "flatten")
        self.assertEqual(t, 0.0)
        t2, _ = target_from_decision(decision(SignalType.HOLD, kill=True), 0.5, self.state, self.cfg, "hold")
        self.assertIsNone(t2)

    def test_volatility_aware_sizing(self):
        cfg = config(execution={"volatility_aware_sizing": True, "target_volatility": 0.25})
        t, _ = target_from_decision(decision(SignalType.BUY, 0.8), 0.0, self.state, cfg, "flatten")
        self.assertAlmostEqual(t, 0.8 * 0.25 / 0.5)


class TestKillSwitch(unittest.TestCase):
    def ks(self, **kw):
        base = dict(max_drawdown=0.1, sharpe_threshold=-100.0, sharpe_window=10, min_history=5, bars_per_year=2190,
                    recovery="cooldown", cooldown_bars=3)
        base.update(kw)
        return KillSwitch(**base)

    def test_min_history_then_drawdown_trigger(self):
        k = self.ks()
        eq = [100, 100, 100, 85]  # 15% drawdown but only 3 returns
        self.assertFalse(any(k.update(e, TS, i) for i, e in enumerate(eq)))
        for i, e in enumerate([85, 85, 85], start=4):
            k.update(e, TS, i)
        self.assertTrue(k.active)
        self.assertEqual(k.events[0].kind, "trigger")

    def test_cooldown_recovery(self):
        k = self.ks(min_history=1)
        k.update(100, TS, 0)
        k.update(80, TS, 1)
        self.assertTrue(k.active)
        k.update(80, TS, 2)
        k.update(80, TS, 3)
        self.assertTrue(k.active)
        self.assertFalse(k.update(80, TS, 4))
        self.assertEqual(k.events[-1].kind, "recovery")

    def test_manual_stays_active(self):
        k = self.ks(min_history=1, recovery="manual")
        k.update(100, TS, 0)
        k.update(80, TS, 1)
        for i in range(2, 20):
            self.assertTrue(k.update(80, TS, i))
        k.reset()
        self.assertFalse(k.active)

    def test_sharpe_trigger(self):
        k = self.ks(max_drawdown=0.99, sharpe_threshold=-1.0, sharpe_window=10, min_history=10)
        e = 100.0
        for i in range(12):
            e *= 0.999 if i % 2 else 0.998
            k.update(e, TS, i)
        self.assertTrue(k.active)
        self.assertIn("Sharpe", k.events[0].detail)


class TestBacktestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        md = SyntheticFeed(cls.cfg, seed=7).get_bars("T", parse_utc_date("2023-01-01"), parse_utc_date("2023-05-01"))
        cls.ff = compute_features(md, cls.cfg)
        cls.res = Backtester(cls.cfg, cls.ff).run(cls.ff.first_trade_index, len(cls.ff))

    def test_fills_at_next_bar_open_with_costs(self):
        opens = self.ff.column("open")
        slip = self.cfg.execution.slippage_bps / 1e4
        fee = self.cfg.execution.fee_bps / 1e4
        self.assertGreater(len(self.res.portfolio.fills), 0)
        for f in self.res.portfolio.fills:
            self.assertEqual(f.reference_price, opens[f.bar_index])
            self.assertEqual(f.timestamp, self.ff.index[f.bar_index].to_pydatetime())
            expect = f.reference_price * (1 + slip if f.side.value == "BUY" else 1 - slip)
            self.assertAlmostEqual(f.exec_price, expect)
            self.assertAlmostEqual(f.fee, f.notional * fee)

    def test_orders_reference_previous_decision(self):
        for f in self.res.portfolio.fills:
            k = f.bar_index - 1 - self.res.start_index
            self.assertGreaterEqual(k, 0)

    def test_long_only_never_short(self):
        self.assertTrue((self.res.exposures >= -1e-12).all())
        self.assertTrue((self.res.exposures <= self.cfg.execution.max_exposure + 1e-9).all())

    def test_equity_consistency(self):
        p = self.res.portfolio
        last_close = self.ff.column("close")[self.res.end_index - 1]
        self.assertAlmostEqual(self.res.equity.iloc[-1], p.cash + p.units * last_close, places=6)

    def test_benchmarks_same_window(self):
        b = run_benchmarks(self.cfg, self.ff, self.ff.first_trade_index, len(self.ff))
        for m in b.values():
            self.assertEqual(m.n_bars, self.res.metrics.n_bars)
        self.assertEqual(b["no_trade"].total_return, 0.0)
        self.assertGreater(b["buy_and_hold"].total_fees, 0)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_boss_orchestrator.py`

```python
"""BossOrchestrator aggregation, overrides and diagnostics."""

import math
import unittest

from helpers import TS, config

from agents.boss_orchestrator import BossOrchestrator
from core.models import AgentSignal, SignalType

NAMES = ["market_intel", "risk_engine", "liquidity_engine", "sentiment_engine", "regime_detector"]


def sig(name, s, c, available=True, **feat):
    return AgentSignal(name, s, c, f"{name} test reason", dict(feat), TS, available, f"{name} summary")


def risk(score=0.3):
    return sig("risk_engine", SignalType.HOLD, 1 - score, risk_score=score)


def regime(r="range", s=SignalType.HOLD, c=0.0):
    return sig("regime_detector", s, c, regime=r)


def plain_cfg(**extra):
    o = {"features": {"enable_kalman": False, "enable_regime_weights": False}}
    o.update(extra)
    return config(**o)


class TestBoss(unittest.TestCase):
    def test_weighted_score_hand_calculated(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 0.8), risk(0.3), sig("liquidity_engine", SignalType.HOLD, 0.0),
                sig("sentiment_engine", SignalType.BUY, 0.6), regime()]
        d = boss.decide(sigs)
        # equal weights: S = (0.8 + 0.6) / 5 = 0.28 ; agreement 1 ; coverage 2/5
        self.assertAlmostEqual(d.directional_score, 0.28)
        self.assertAlmostEqual(d.agreement, 1.0)
        self.assertAlmostEqual(d.coverage, 0.4)
        self.assertAlmostEqual(d.confidence, 1.0 * 0.7 * math.sqrt(0.4))
        self.assertEqual(d.action, SignalType.BUY)
        self.assertIsNone(d.forced_hold_rule)

    def test_disagreement_reduces_agreement(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 0.9), risk(), sig("liquidity_engine", SignalType.SELL, 0.4),
                sig("sentiment_engine", SignalType.BUY, 0.9), regime()]
        d = boss.decide(sigs)
        self.assertAlmostEqual(d.directional_score, (0.9 - 0.4 + 0.9) / 5)
        self.assertAlmostEqual(d.agreement, 2 / 3)

    def test_risk_veto_overrides_buy(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 1.0), risk(0.95), sig("liquidity_engine", SignalType.BUY, 1.0),
                sig("sentiment_engine", SignalType.BUY, 1.0), regime("trend_up", SignalType.BUY, 1.0)]
        d = boss.decide(sigs)
        self.assertEqual(d.raw_action, SignalType.BUY)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertTrue(d.risk_veto)
        self.assertIn("risk_veto", d.forced_hold_rule)
        self.assertIn("overridden", d.reason)

    def test_risk_unavailable_fails_safe(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 1.0), sig("risk_engine", SignalType.HOLD, 0.0, available=False),
                sig("liquidity_engine", SignalType.BUY, 1.0), sig("sentiment_engine", SignalType.BUY, 1.0), regime()]
        d = boss.decide(sigs)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertIn("risk_unavailable", d.forced_hold_rule)

    def test_floors(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        d = boss.decide([sig("market_intel", SignalType.BUY, 0.5), risk(), sig("liquidity_engine", SignalType.HOLD, 0),
                         sig("sentiment_engine", SignalType.HOLD, 0), regime()])
        self.assertAlmostEqual(d.directional_score, 0.1)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertIn("directional_score_floor", d.forced_hold_rule)
        self.assertIn("min_confidence", d.forced_hold_rule)

    def test_kill_switch_forces_hold(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        sigs = [sig("market_intel", SignalType.BUY, 1.0), risk(), sig("liquidity_engine", SignalType.BUY, 1.0),
                sig("sentiment_engine", SignalType.BUY, 1.0), regime()]
        d = boss.decide(sigs, kill_switch_active=True)
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertTrue(d.kill_switch_active)
        self.assertIn("kill_switch", d.forced_hold_rule)

    def test_unavailable_agent_excluded_not_confirming(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        base = [sig("market_intel", SignalType.BUY, 0.8), risk(), sig("liquidity_engine", SignalType.HOLD, 0),
                sig("sentiment_engine", SignalType.BUY, 0.8)]
        d1 = boss.decide(base + [sig("regime_detector", SignalType.HOLD, 0.0, available=False, regime="unknown")])
        # denominator excludes the unavailable agent: S = 1.6 / 4
        self.assertAlmostEqual(d1.directional_score, 0.4)
        self.assertAlmostEqual(d1.data_coverage, 0.8)
        elig = {c.agent_name: c.eligible for c in d1.breakdown}
        self.assertFalse(elig["regime_detector"])

    def test_data_coverage_rule(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        un = lambda n: sig(n, SignalType.HOLD, 0.0, available=False)  # noqa: E731
        d = boss.decide([sig("market_intel", SignalType.BUY, 1.0), risk(), un("liquidity_engine"),
                         un("sentiment_engine"), un("regime_detector")])
        self.assertEqual(d.action, SignalType.HOLD)
        self.assertIn("data_coverage", d.forced_hold_rule)

    def test_disabled_agent_rejected(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        with self.assertRaises(ValueError):
            boss.decide([sig("on_chain", SignalType.BUY, 1.0)])

    def test_regime_weights_applied(self):
        cfg = config(features={"enable_kalman": False, "enable_regime_weights": True})
        boss = BossOrchestrator(cfg, NAMES)
        d = boss.decide([sig("market_intel", SignalType.BUY, 1.0), risk(), sig("liquidity_engine", SignalType.HOLD, 0),
                         sig("sentiment_engine", SignalType.HOLD, 0), regime("trend_up", SignalType.BUY, 0.5)])
        w = {c.agent_name: c.effective_weight for c in d.breakdown}
        tbl = cfg.regime_weights["trend_up"]
        for n in NAMES:
            self.assertAlmostEqual(w[n], tbl[n])
        self.assertAlmostEqual(d.directional_score, (tbl["market_intel"] * 1.0 + tbl["regime_detector"] * 0.5) / sum(tbl[n] for n in NAMES))

    def test_learned_weights_clipped(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        boss.set_learned_weights({"market_intel": 10.0, "sentiment_engine": 0.01})
        self.assertEqual(boss.learned_weights["market_intel"], 2.0)
        self.assertEqual(boss.learned_weights["sentiment_engine"], 0.3)

    def test_family_normalization(self):
        cfg = config(features={"enable_kalman": False, "enable_regime_weights": False, "enable_family_normalization": True})
        boss = BossOrchestrator(cfg, NAMES)
        self.assertEqual(boss.family_factor["market_intel"], 0.5)
        self.assertEqual(boss.family_factor["regime_detector"], 0.5)
        self.assertEqual(boss.family_factor["sentiment_engine"], 1.0)

    def test_kalman_smoothing_responds_to_reversal(self):
        cfg = config(features={"enable_kalman": True, "enable_regime_weights": False})
        boss = BossOrchestrator(cfg, NAMES)
        rest = [risk(), sig("liquidity_engine", SignalType.HOLD, 0), sig("sentiment_engine", SignalType.HOLD, 0), regime()]
        scores = [boss.decide([sig("market_intel", SignalType.BUY, 1.0)] + rest).directional_score for _ in range(10)]
        after = [boss.decide([sig("market_intel", SignalType.SELL, 1.0)] + rest).directional_score for _ in range(15)]
        self.assertAlmostEqual(scores[-1], 0.2, places=3)
        self.assertGreater(after[0], -0.2)            # lagged response...
        self.assertLess(after[0], scores[-1])         # ...but moving toward the new direction
        self.assertLess(after[-1], -0.15)             # converges after the reversal

    def test_reasoner_mentions_actual_values(self):
        boss = BossOrchestrator(plain_cfg(), NAMES)
        d = boss.decide([sig("market_intel", SignalType.BUY, 0.8), risk(0.31), sig("liquidity_engine", SignalType.HOLD, 0),
                         sig("sentiment_engine", SignalType.BUY, 0.6), regime()])
        self.assertIn("risk_score 0.31", d.reason)
        self.assertIn("market_intel summary", d.reason.lower())


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_evaluator.py`

```python
"""Evaluator scoring, weight suggestions, metrics and calibration."""

import unittest

from helpers import TS, config

import numpy as np
import pandas as pd

from core.models import AgentScore, Trade
from execution.evaluator import Evaluator, LogisticCalibrator, SignalRecord, compute_metrics


def rec(i, name, d, c=1.0, available=True, w=1.0):
    return SignalRecord(i, name, d, c, available, c * d if available else 0.0, w, available)


class TestScoreAgents(unittest.TestCase):
    def setUp(self):
        self.cfg = config(evaluation={"min_agent_samples": 2})
        self.ev = Evaluator(self.cfg, {"a": 1.0, "b": 1.0})
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)

    def test_only_matured_outcomes(self):
        hist = [rec(i, "a", 1) for i in range(10)] + [rec(i, "b", 0) for i in range(10)]
        s = self.ev.score_agents(hist, self.prices, lookahead=3, as_of_index=5)
        self.assertEqual(s["a"].matured_count, 3)  # bars 0,1,2 (2+3 <= 5)
        self.assertEqual(s["a"].directional_count, 6)  # bars 0..5 visible
        self.assertEqual(s["a"].hit_rate, 1.0)

    def test_future_prices_do_not_matter(self):
        hist = [rec(i, "a", 1) for i in range(10)] + [rec(i, "b", -1) for i in range(10)]
        s1 = self.ev.score_agents(hist, self.prices, lookahead=3, as_of_index=5)
        p2 = self.prices.copy()
        p2[6:] = 1.0  # crash after as_of
        s2 = self.ev.score_agents(hist, p2, lookahead=3, as_of_index=5)
        self.assertEqual(s1, s2)

    def test_hold_excluded_from_hit_rate(self):
        hist = [rec(i, "a", 1 if i % 2 else 0) for i in range(10)]
        s = self.ev.score_agents(hist, self.prices, lookahead=1)
        self.assertEqual(s["a"].directional_count, 5)
        self.assertEqual(s["a"].hit_rate, 1.0)

    def test_wrong_direction_misses(self):
        hist = [rec(i, "a", -1) for i in range(10)]
        self.assertEqual(self.ev.score_agents(hist, self.prices, lookahead=1)["a"].hit_rate, 0.0)

    def test_unavailable_excluded(self):
        hist = [rec(i, "a", 0, available=False) for i in range(10)]
        s = self.ev.score_agents(hist, self.prices, lookahead=1)["a"]
        self.assertEqual(s.directional_count, 0)
        self.assertIsNone(s.hit_rate)

    def test_standalone_pnl_with_costs(self):
        ev = Evaluator(config(execution={"fee_bps": 0, "slippage_bps": 0}), {"a": 1.0})
        hist = [rec(i, "a", 1) for i in range(3)]
        p = np.array([100.0, 110.0, 121.0])
        s = ev.score_agents(hist, p, lookahead=1)["a"]
        self.assertAlmostEqual(s.standalone_pnl, 0.2)  # 10% + 10%, non-compounded


class TestSuggestWeights(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.ev = Evaluator(self.cfg, {"a": 1.0})

    def score(self, matured, t, current=1.0):
        return AgentScore("a", 100, 100, matured, 0.6, 0.01, 0.1, 1.0, t, current, current)

    def test_insufficient_samples_keep_weight(self):
        self.assertEqual(self.ev.suggest_weights({"a": self.score(5, 3.0, 1.3)})["a"], 1.3)

    def test_positive_evidence_increases_bounded(self):
        w = self.ev.suggest_weights({"a": self.score(100, 2.5)})["a"]
        self.assertGreater(w, 1.0)
        for _ in range(200):
            w = self.ev.suggest_weights({"a": self.score(100, 3.0, w)})["a"]
        self.assertLessEqual(w, self.cfg.evaluation.max_weight)

    def test_negative_evidence_decreases_bounded(self):
        w = 1.0
        for _ in range(200):
            w = self.ev.suggest_weights({"a": self.score(100, -3.0, w)})["a"]
        self.assertGreaterEqual(w, self.cfg.evaluation.min_weight)
        self.assertLess(w, 1.0)

    def test_neutral_evidence_shrinks_to_base(self):
        w = self.ev.suggest_weights({"a": self.score(100, 0.2, 1.8)})["a"]
        self.assertAlmostEqual(w, 0.75 * 1.8 + 0.25 * 1.0)


class TestMetrics(unittest.TestCase):
    def test_zero_variance_and_no_trades(self):
        eq = pd.Series([100.0] * 20)
        m = compute_metrics(eq, [], pd.Series([0.0] * 20), 0, 0, 0, 2190, 6)
        self.assertIsNone(m.sharpe)
        self.assertIsNone(m.win_rate)
        self.assertIsNone(m.profit_factor)
        self.assertIsNone(m.cagr)
        self.assertIn("sharpe", m.notes)

    def test_no_losses_profit_factor_unavailable(self):
        t = Trade(1, TS, 0, 100, 1, "x", TS, 5, 110, "y", gross_pnl=10, fees=1)
        m = compute_metrics(pd.Series(np.linspace(100, 110, 20)), [t], pd.Series([1.0] * 20), 1, 0, 100, 2190, 6)
        self.assertEqual(m.win_rate, 1.0)
        self.assertIsNone(m.profit_factor)
        self.assertEqual(m.avg_holding_bars, 5)

    def test_drawdown(self):
        m = compute_metrics(pd.Series([100.0, 120, 90, 130]), [], pd.Series([0.0] * 4), 0, 0, 0, 2190, 6)
        self.assertAlmostEqual(m.max_drawdown, 0.25)


class TestCalibrator(unittest.TestCase):
    def test_fit_monotone(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < 0.3 + 0.4 * x).astype(float)
        c = LogisticCalibrator()
        self.assertTrue(c.fit(x, y))
        p = c.predict(np.array([0.1, 0.9]))
        self.assertLess(p[0], p[1])

    def test_insufficient(self):
        self.assertFalse(LogisticCalibrator().fit(np.array([0.1, 0.2]), np.array([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_feedback.py`

```python
"""Feedback loop: interval, freezing, maturity and bounds."""

import unittest

from helpers import TS, config

import numpy as np

from agents.boss_orchestrator import BossOrchestrator
from execution.evaluator import Evaluator, SignalRecord
from execution.feedback import FeedbackController

NAMES = ["market_intel", "risk_engine", "liquidity_engine", "sentiment_engine", "regime_detector"]


def history(n, good="market_intel"):
    """market_intel always right on a rising series, sentiment always wrong."""
    out = []
    for i in range(n):
        for name in NAMES:
            d = 1 if name == good else (-1 if name == "sentiment_engine" else 0)
            c = 0.8 if d else 0.0
            out.append(SignalRecord(i, name, d, c, True, c * d, 1.0, True))
    return out


class TestFeedback(unittest.TestCase):
    def setUp(self):
        self.cfg = config(evaluation={"min_agent_samples": 10, "feedback_interval": 50})
        self.boss = BossOrchestrator(self.cfg, NAMES)
        self.ev = Evaluator(self.cfg, self.boss.base_weights)
        rng = np.random.default_rng(1)
        self.prices = 100 * np.exp(np.cumsum(0.002 + 0.01 * rng.standard_normal(400)))

    def test_only_on_interval(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=0)
        h = history(300)
        self.assertFalse(fb.maybe_update(49, TS, h, self.prices))
        self.assertTrue(fb.maybe_update(100, TS, h, self.prices))

    def test_frozen_never_updates(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=0, frozen=True)
        before = dict(self.boss.learned_weights)
        self.assertFalse(fb.maybe_update(100, TS, history(300), self.prices))
        self.assertEqual(before, self.boss.learned_weights)

    def test_uses_only_matured_data(self):
        h = history(300)
        fb1 = FeedbackController(self.ev, self.boss, 50, window_start=0)
        fb1.maybe_update(150, TS, h, self.prices)
        w1 = dict(self.boss.learned_weights)
        boss2 = BossOrchestrator(self.cfg, NAMES)
        p2 = self.prices.copy()
        p2[151:] = p2[151:] * 0.01  # future collapse must not matter
        fb2 = FeedbackController(Evaluator(self.cfg, boss2.base_weights), boss2, 50, window_start=0)
        fb2.maybe_update(150, TS, h, p2)
        self.assertEqual(w1, boss2.learned_weights)

    def test_ignores_records_before_window(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=200)
        h = history(300)
        fb.maybe_update(250, TS, h, self.prices)  # only bars 200..230 mature (h=20)
        self.assertTrue(all(abs(w - 1.0) < 1.0 for w in self.boss.learned_weights.values()))

    def test_weights_bounded_and_logged(self):
        fb = FeedbackController(self.ev, self.boss, 50, window_start=0)
        h = history(399)
        for t in range(50, 400, 50):
            fb.maybe_update(t, TS, h, self.prices)
        ev = self.cfg.evaluation
        for w in self.boss.learned_weights.values():
            self.assertTrue(ev.min_weight <= w <= ev.max_weight)
        self.assertGreater(len(fb.updates), 0)
        self.assertTrue(all(u.evidence for u in fb.updates))
        # risk_engine never votes directionally -> weight preserved
        self.assertEqual(self.boss.learned_weights["risk_engine"], 1.0)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_integration.py`

```python
"""End-to-end integration: data -> features -> agents -> boss -> execution -> evaluation."""

import unittest

from helpers import CONFIG_PATH, config

import numpy as np
import pandas as pd

from agents import build_agents
from agents.boss_orchestrator import BossOrchestrator
from core.config import ConfigError, load_config
from core.feature_bus import compute_features
from core.models import FeatureStatus, SignalType
from data.feed import DataValidationError, LiveFeedStub, LiveFeedUnavailableError, MarketData, SyntheticFeed, parse_utc_date, validate_bars
from execution.backtester import Backtester
from execution.walk_forward import make_folds, run_walk_forward

START, END = parse_utc_date("2023-01-01"), parse_utc_date("2023-04-01")


def feed(cfg, seed=11, start=START, end=END):
    return SyntheticFeed(cfg, seed=seed).get_bars("TEST", start, end)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.md = feed(cls.cfg)
        cls.ff = compute_features(cls.md, cls.cfg)

    def test_all_agents_analyze_same_state(self):
        st = self.ff.state(self.ff.first_trade_index + 10)
        sigs = [a.analyze(st) for a in build_agents(self.cfg)]
        self.assertEqual(len(sigs), 5)
        for s in sigs:
            self.assertTrue(s.reason)
            self.assertTrue(0 <= s.confidence <= 1)
            self.assertEqual(s.timestamp, st.timestamp)
        d = BossOrchestrator(self.cfg, [a.name for a in build_agents(self.cfg)]).decide(sigs)
        self.assertTrue(-1 <= d.directional_score <= 1)

    def test_synthetic_labelled(self):
        self.assertEqual(self.md.availability["close"], FeatureStatus.SYNTHETIC)
        self.assertEqual(self.md.availability["put_call_skew"], FeatureStatus.UNAVAILABLE)
        self.assertEqual(self.ff.availability["ofi"], FeatureStatus.DERIVED)

    def test_features_are_causal(self):
        t = self.ff.first_trade_index + 50
        trunc = MarketData(self.md.symbol, self.md.bars.iloc[: t + 1].copy(), self.md.availability, "trunc", self.md.first_trade_index)
        ff2 = compute_features(trunc, self.cfg)
        a = self.ff.frame.iloc[t][self.ff.indicator_names].astype(float)
        b = ff2.frame.iloc[t][ff2.indicator_names].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_reproducible_with_seed(self):
        r1 = Backtester(self.cfg, compute_features(feed(self.cfg, 11), self.cfg)).run(self.ff.first_trade_index, len(self.ff))
        r2 = Backtester(self.cfg, compute_features(feed(self.cfg, 11), self.cfg)).run(self.ff.first_trade_index, len(self.ff))
        pd.testing.assert_series_equal(r1.equity, r2.equity)
        other = feed(self.cfg, 12).bars["close"]
        self.assertFalse(np.allclose(other.to_numpy(), self.md.bars["close"].to_numpy()))

    def test_frozen_run_has_no_weight_updates(self):
        w = {"market_intel": 1.5}
        r = Backtester(self.cfg, self.ff).run(self.ff.first_trade_index, len(self.ff), learn=False, initial_weights=w)
        self.assertEqual(r.weight_updates, [])
        self.assertEqual(r.final_weights["market_intel"], 1.5)

    def test_missing_sentiment_feed_is_unavailable_not_neutral(self):
        bars = self.md.bars.drop(columns=["social_sentiment", "news_sentiment"])
        av = dict(self.md.availability)
        av["social_sentiment"] = av["news_sentiment"] = FeatureStatus.UNAVAILABLE
        ff = compute_features(MarketData("T", bars, av, "x", self.md.first_trade_index), self.cfg)
        st = ff.state(ff.first_trade_index + 5)
        self.assertIsNone(st.social_sentiment)
        s = [a for a in build_agents(self.cfg) if a.name == "sentiment_engine"][0].analyze(st)
        self.assertFalse(s.available)

    def test_missing_order_flow_disables_ofi(self):
        bars = self.md.bars.drop(columns=["bid_volume", "ask_volume"])
        av = dict(self.md.availability)
        av["bid_volume"] = av["ask_volume"] = FeatureStatus.UNAVAILABLE
        ff = compute_features(MarketData("T", bars, av, "x", self.md.first_trade_index), self.cfg)
        self.assertEqual(ff.availability["ofi"], FeatureStatus.UNAVAILABLE)
        s = [a for a in build_agents(self.cfg) if a.name == "market_intel"][0].analyze(ff.state(ff.first_trade_index))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)

    def test_optional_agent_flag(self):
        self.assertNotIn("on_chain", [a.name for a in build_agents(self.cfg)])
        cfg = config(features={"enable_optional_agent": True})
        agents = build_agents(cfg)
        self.assertIn("on_chain", [a.name for a in agents])
        r = Backtester(cfg, compute_features(feed(cfg), cfg)).run(self.ff.first_trade_index, self.ff.first_trade_index + 50)
        self.assertIn("on_chain", [c.agent_name for c in r.decisions[-1].breakdown])

    def test_optional_agent_unavailable_without_data(self):
        cfg = config(features={"enable_optional_agent": True}, synthetic={"generate_optional": False})
        ff = compute_features(feed(cfg), cfg)
        s = [a for a in build_agents(cfg) if a.name == "on_chain"][0].analyze(ff.state(ff.first_trade_index + 5))
        self.assertFalse(s.available)


class TestWalkForward(unittest.TestCase):
    def test_folds_chronological_with_embargo(self):
        cfg = config()
        folds = make_folds(cfg, 2440, 250)
        self.assertGreater(len(folds), 1)
        e = cfg.walk_forward.embargo_bars
        for f in folds:
            self.assertLessEqual(f.train[1] + e, f.validation[0])
            self.assertLessEqual(f.validation[1] + e, f.test[0])
            self.assertLess(f.train[0], f.train[1])
        for a, b in zip(folds, folds[1:]):
            self.assertLess(a.test[0], b.test[0])
            self.assertLessEqual(a.test[1], b.test[0] + cfg.walk_forward.test_bars)

    def test_walk_forward_runs_and_separates_results(self):
        cfg = config(walk_forward={"train_bars": 200, "validation_bars": 50, "test_bars": 50, "embargo_bars": 20,
                                   "threshold_grid": [0.35]})
        ff = compute_features(feed(cfg, start=START, end=parse_utc_date("2023-06-01")), cfg)
        rep = run_walk_forward(cfg, ff)
        self.assertGreater(len(rep.folds), 0)
        for f in rep.folds:
            self.assertEqual(f.test_metrics.n_bars, 50)
            self.assertEqual(f.train_metrics.n_bars, 200)
            self.assertLess(f.windows.train[1], f.windows.test[0])
        self.assertEqual(rep.oos_metrics.n_bars, 1 + sum(len(f.test_returns) for f in rep.folds))

    def test_too_short_raises(self):
        cfg = config()
        ff = compute_features(feed(cfg), cfg)
        with self.assertRaises(ValueError):
            run_walk_forward(cfg, ff)


class TestValidation(unittest.TestCase):
    def bars(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="4h", tz="UTC")
        return pd.DataFrame({"open": [10.0] * 5, "high": [11.0] * 5, "low": [9.0] * 5, "close": [10.0] * 5, "volume": [1.0] * 5}, index=idx)

    def test_valid(self):
        validate_bars(self.bars())

    def test_duplicates(self):
        b = self.bars()
        b.index = b.index[[0, 1, 1, 3, 4]]
        with self.assertRaises(DataValidationError):
            validate_bars(b)

    def test_naive_timestamps(self):
        b = self.bars()
        b.index = b.index.tz_localize(None)
        with self.assertRaises(DataValidationError):
            validate_bars(b)

    def test_missing_and_invalid_prices(self):
        b = self.bars()
        b.iloc[2, 3] = np.nan
        with self.assertRaises(DataValidationError):
            validate_bars(b)
        b = self.bars()
        b.iloc[2, 1] = 9.5  # high < close
        with self.assertRaises(DataValidationError):
            validate_bars(b)
        b = self.bars()
        b.iloc[0, 0] = -1
        with self.assertRaises(DataValidationError):
            validate_bars(b)

    def test_insufficient_history(self):
        with self.assertRaises(DataValidationError):
            validate_bars(self.bars(), min_bars=10)

    def test_invalid_date_range(self):
        with self.assertRaises(DataValidationError):
            SyntheticFeed(config()).get_bars("T", END, START)

    def test_live_feed_stub_never_falls_back(self):
        with self.assertRaises(LiveFeedUnavailableError):
            LiveFeedStub().get_bars("BTCUSDT", START, END)

    def test_config_errors(self):
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"walk_forward": {"embargo_bars": 5}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"strategy": {"min_confidnce": 0.3}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"strategy": {"allow_short": True}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"execution": {"max_exposure": 2.0}})
        with self.assertRaises(ConfigError):
            load_config(CONFIG_PATH, {"agents": {"base_weights": {"market_intel": 5.0}}})


if __name__ == "__main__":
    unittest.main()


class TestKalman(unittest.TestCase):
    def test_stable_reversal_missing_bounds(self):
        from core.kalman import Kalman1D

        k = Kalman1D(0.02, 0.1)
        self.assertIsNone(k.update(None))          # uninitialized + missing
        self.assertEqual(k.update(0.5), 0.5)       # first observation initializes
        for _ in range(20):
            x = k.update(0.5)
        self.assertAlmostEqual(x, 0.5)             # stable signal stays put
        held = k.update(float("nan"))
        self.assertAlmostEqual(held, 0.5)          # missing obs: predict only
        path = [k.update(-0.8) for _ in range(30)]
        self.assertGreater(path[0], -0.8)          # lag on abrupt reversal
        self.assertLess(path[0], 0.5)
        self.assertAlmostEqual(path[-1], -0.8, places=3)
        k2 = Kalman1D(0.02, 0.1)
        self.assertEqual(k2.update(5.0), 1.0)      # clipped to bounds
        with self.assertRaises(ValueError):
            Kalman1D(0.0, 0.1)
```

### `tests/test_liquidity.py`

```python
"""Hand-calculated tests for indicators/liquidity.py."""

import math
import unittest

import helpers  # noqa: F401
import pandas as pd

from indicators.liquidity import absorption_ratio, depth_imbalance, detect_sweeps


class TestSweeps(unittest.TestCase):
    def frame(self, last_low, last_high, last_close):
        h = pd.Series([11.0, 11.0, 11.0, last_high])
        l = pd.Series([10.0, 10.0, 10.0, last_low])
        c = pd.Series([10.5, 10.5, 10.5, last_close])
        return detect_sweeps(h, l, c, 3)

    def test_bullish_confirmed(self):
        s = self.frame(9.0, 10.8, 10.5)
        self.assertTrue(s["bullish_sweep"].iloc[3])
        self.assertFalse(s["bearish_sweep"].iloc[3])
        self.assertEqual(s["prior_low"].iloc[3], 10.0)

    def test_no_reversal_no_sweep(self):
        s = self.frame(9.0, 10.8, 9.5)
        self.assertFalse(s["bullish_sweep"].iloc[3])

    def test_bearish_confirmed(self):
        s = self.frame(10.2, 12.0, 10.6)
        self.assertTrue(s["bearish_sweep"].iloc[3])
        self.assertEqual(s["prior_high"].iloc[3], 11.0)

    def test_warmup_false(self):
        s = self.frame(9.0, 10.8, 10.5)
        self.assertFalse(s["bullish_sweep"].iloc[:3].any())
        self.assertTrue(s["prior_low"].iloc[:3].isna().all())

    def test_prior_level_excludes_current_and_is_causal(self):
        h = pd.Series([11.0, 11, 11, 10.8, 11])
        l = pd.Series([10.0, 10, 10, 9.0, 10])
        c = pd.Series([10.5, 10.5, 10.5, 10.5, 10.5])
        a = detect_sweeps(h, l, c, 3)
        l2 = l.copy()
        l2.iloc[4] = 1.0  # change only the future bar
        b = detect_sweeps(h, l2, c, 3)
        pd.testing.assert_frame_equal(a.iloc[:4], b.iloc[:4])


class TestAbsorption(unittest.TestCase):
    def test_hand_calculated(self):
        # range = [0.1, 0.1, 0.05]; rel_vol = 3; rel_range = 0.5 -> 6
        v = pd.Series([1.0, 1.0, 3.0])
        c = pd.Series([10.0, 10.0, 10.0])
        h = pd.Series([10.5, 10.5, 10.25])
        l = pd.Series([9.5, 9.5, 9.75])
        self.assertAlmostEqual(absorption_ratio(v, h, l, c, 2).iloc[2], 6.0)

    def test_zero_range_capped(self):
        v = pd.Series([1.0, 1.0, 2.0])
        c = pd.Series([10.0, 10.0, 10.0])
        h = pd.Series([10.5, 10.5, 10.0])
        l = pd.Series([9.5, 9.5, 10.0])
        self.assertAlmostEqual(absorption_ratio(v, h, l, c, 2, 0.1).iloc[2], 20.0)

    def test_zero_baseline_volume_nan(self):
        v = pd.Series([0.0, 0.0, 2.0])
        c = pd.Series([10.0] * 3)
        h = pd.Series([10.5] * 3)
        l = pd.Series([9.5] * 3)
        self.assertTrue(math.isnan(absorption_ratio(v, h, l, c, 2).iloc[2]))


class TestDepth(unittest.TestCase):
    def test_scalar(self):
        self.assertAlmostEqual(depth_imbalance(30.0, 10.0), 0.5)
        self.assertTrue(math.isnan(depth_imbalance(0.0, 0.0)))

    def test_series(self):
        d = depth_imbalance(pd.Series([10.0, 0.0]), pd.Series([30.0, 0.0]))
        self.assertAlmostEqual(d.iloc[0], -0.5)
        self.assertTrue(math.isnan(d.iloc[1]))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            depth_imbalance(-1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_liquidity_engine.py`

```python
"""LiquidityEngineAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.liquidity_engine import LiquidityEngineAgent
from core.models import SignalType

BASE = {"prior_low": 95.0, "prior_high": 105.0, "absorption_ratio": 1.0}


def st(bull=0.0, bear=0.0, cvd=None, depth=None, ar=1.0, unavailable=()):
    ind = dict(BASE, bullish_sweep=bull, bearish_sweep=bear, absorption_ratio=ar)
    if cvd is not None:
        ind["cvd_norm_short"] = cvd
    if depth is not None:
        ind["depth_imbalance"] = depth
    return make_state(ind, unavailable=unavailable)


class TestLiquidityEngine(unittest.TestCase):
    def setUp(self):
        self.agent = LiquidityEngineAgent(1.0, config().agents.liquidity_engine)

    def test_bullish_confirmed_by_cvd_and_depth(self):
        s = self.agent.analyze(st(bull=1.0, cvd=0.3, depth=0.2))
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, 0.8)  # 0.4 + 0.2 cvd + 0.2 depth
        self.assertIn("95.00", s.reason)
        self.assertIn("not proof", s.reason)

    def test_absorption_adds_confidence(self):
        s = self.agent.analyze(st(bull=1.0, cvd=0.3, depth=0.2, ar=2.0))
        self.assertAlmostEqual(s.confidence, 1.0)
        self.assertIn("absorption of selling", s.reason)

    def test_bullish_cvd_divergence_hold(self):
        s = self.agent.analyze(st(bull=1.0, cvd=-0.3, depth=0.2))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("diverges", s.reason)

    def test_bullish_without_flow_data_unconfirmed(self):
        s = self.agent.analyze(st(bull=1.0, unavailable=("cvd_norm_short", "depth_imbalance")))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertTrue(s.available)
        self.assertIn("unconfirmed", s.reason)

    def test_bearish_confirmed(self):
        s = self.agent.analyze(st(bear=1.0, cvd=-0.2, depth=-0.3))
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("105.00", s.reason)
        self.assertIn("above", s.reason)

    def test_no_sweep(self):
        s = self.agent.analyze(st(cvd=0.5, depth=0.5))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("no sweep", s.reason)

    def test_two_sided_ambiguous(self):
        s = self.agent.analyze(st(bull=1.0, bear=1.0, cvd=0.5, depth=0.5))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("ambiguous", s.reason)

    def test_warmup_unavailable(self):
        s = self.agent.analyze(make_state({"bullish_sweep": float("nan"), "bearish_sweep": float("nan"),
                                           "prior_low": float("nan"), "prior_high": float("nan")}))
        self.assertFalse(s.available)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_market_intel.py`

```python
"""MarketIntelAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.market_intel import MarketIntelAgent
from core.models import SignalType


class TestMarketIntel(unittest.TestCase):
    def setUp(self):
        cfg = config()
        self.p = cfg.agents.market_intel
        self.agent = MarketIntelAgent(1.0, self.p)

    def expected(self, ofi, cvd, mom):
        p = self.p
        return (p.w_ofi * ofi + p.w_cvd * cvd + p.w_momentum * mom) / (p.w_ofi + p.w_cvd + p.w_momentum)

    def test_positive_flow_buy(self):
        s = self.agent.analyze(make_state({"ofi": 0.5, "cvd_norm": 0.4, "inst_momentum": 0.6, "pv_corr": 0.2}))
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, abs(self.expected(0.5, 0.4, 0.6)))
        self.assertIn("buying pressure", s.reason)
        self.assertIn("OFI +0.500", s.reason)

    def test_negative_flow_sell(self):
        s = self.agent.analyze(make_state({"ofi": -0.5, "cvd_norm": -0.3, "inst_momentum": -0.4}))
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("selling pressure", s.reason)

    def test_neutral_hold_explained(self):
        s = self.agent.analyze(make_state({"ofi": 0.05, "cvd_norm": -0.02, "inst_momentum": 0.03}))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertTrue(s.available)
        self.assertIn("within", s.reason)

    def test_missing_input_unavailable(self):
        s = self.agent.analyze(make_state({"ofi": float("nan"), "cvd_norm": 0.1, "inst_momentum": 0.1}))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("ofi", s.reason)

    def test_feed_not_supplied(self):
        s = self.agent.analyze(make_state({"cvd_norm": 0.1, "inst_momentum": 0.1}, unavailable=("ofi",)))
        self.assertFalse(s.available)
        self.assertIn("feeds not supplied", s.reason)

    def test_deterministic(self):
        st = make_state({"ofi": 0.3, "cvd_norm": 0.2, "inst_momentum": 0.1})
        self.assertEqual(self.agent.analyze(st), self.agent.analyze(st))


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_order_flow.py`

```python
"""Hand-calculated tests for indicators/order_flow.py."""

import math
import unittest

import helpers  # noqa: F401  (sets sys.path)
import numpy as np
import pandas as pd

from indicators.order_flow import (
    cvd, cvd_change_normalized, institutional_momentum, order_flow_imbalance, price_volume_correlation,
)


class TestOFI(unittest.TestCase):
    def test_hand_calculated(self):
        # window 2: t1 = (1+2 - (1+0)) / (3+1) = 0.5 ; t2 = (2+3 - (0+1)) / 6 = 0.6667
        o = order_flow_imbalance(pd.Series([1.0, 2, 3]), pd.Series([1.0, 0, 1]), 2)
        self.assertTrue(math.isnan(o.iloc[0]))
        self.assertAlmostEqual(o.iloc[1], 0.5)
        self.assertAlmostEqual(o.iloc[2], 4 / 6)

    def test_sign_convention_and_bounds(self):
        self.assertEqual(order_flow_imbalance(pd.Series([5.0]), pd.Series([0.0]), 1).iloc[0], 1.0)
        self.assertEqual(order_flow_imbalance(pd.Series([0.0]), pd.Series([5.0]), 1).iloc[0], -1.0)
        self.assertEqual(order_flow_imbalance(pd.Series([3.0]), pd.Series([3.0]), 1).iloc[0], 0.0)

    def test_zero_volume_is_nan(self):
        self.assertTrue(math.isnan(order_flow_imbalance(pd.Series([0.0, 0]), pd.Series([0.0, 0]), 2).iloc[1]))

    def test_missing_value_propagates_nan(self):
        o = order_flow_imbalance(pd.Series([1.0, np.nan, 1.0]), pd.Series([1.0, 1.0, 1.0]), 2)
        self.assertTrue(math.isnan(o.iloc[1]) and math.isnan(o.iloc[2]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            order_flow_imbalance(pd.Series([-1.0]), pd.Series([1.0]), 1)
        with self.assertRaises(ValueError):
            order_flow_imbalance(pd.Series([1.0]), pd.Series([1.0]), 0)
        with self.assertRaises(ValueError):
            order_flow_imbalance(pd.Series([1.0, 2]), pd.Series([1.0]), 1)

    def test_causal(self):
        b = pd.Series(np.arange(1.0, 11))
        a = pd.Series(np.ones(10))
        full = order_flow_imbalance(b, a, 3)
        b2 = b.copy()
        b2.iloc[-1] = 1000.0
        part = order_flow_imbalance(b2, a, 3)
        pd.testing.assert_series_equal(full.iloc[:-1], part.iloc[:-1])


class TestCVD(unittest.TestCase):
    def test_with_side(self):
        c = cvd(pd.Series([1.0, 1, 1]), pd.Series([10.0, 20, 30]), pd.Series([1.0, -0.5, np.nan]))
        self.assertEqual(list(c), [10.0, 0.0, 0.0])  # NaN side contributes zero

    def test_tick_rule_proxy(self):
        # signs: [0, +1, +1 (carried through unchanged price), -1] -> cumsum of [0, 2, 3, -4]
        c = cvd(pd.Series([10.0, 11, 11, 10]), pd.Series([1.0, 2, 3, 4]), None)
        self.assertEqual(list(c), [0.0, 2.0, 5.0, 1.0])

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            cvd(pd.Series([1.0]), pd.Series([1.0]), pd.Series([1.5]))

    def test_normalized_change(self):
        cv = pd.Series([0.0, 10.0, 0.0])
        n = cvd_change_normalized(cv, pd.Series([10.0, 10.0, 10.0]), 1)
        self.assertAlmostEqual(n.iloc[1], 1.0)
        self.assertAlmostEqual(n.iloc[2], -1.0)
        z = cvd_change_normalized(cv, pd.Series([0.0, 0.0, 0.0]), 1)
        self.assertTrue(math.isnan(z.iloc[1]))


class TestInstitutionalMomentum(unittest.TestCase):
    def test_constant_price_is_zero(self):
        s = institutional_momentum(pd.Series([100.0] * 10), pd.Series([1.0] * 10), 3)
        self.assertEqual(s.iloc[-1], 0.0)

    def test_zero_variance_trend(self):
        # constant 1% steps -> sigma 0, R > 0 -> sign 1; equal volume -> factor 0.5
        p = pd.Series(100 * 1.01 ** np.arange(10))
        s = institutional_momentum(p, pd.Series([1.0] * 10), 3)
        self.assertAlmostEqual(s.iloc[-1], 0.5)

    def test_insufficient_history(self):
        s = institutional_momentum(pd.Series(100 * 1.01 ** np.arange(10)), pd.Series([1.0] * 10), 3)
        self.assertTrue(s.iloc[:5].isna().all())
        self.assertFalse(math.isnan(s.iloc[5]))

    def test_matches_formula(self):
        rng = np.random.default_rng(0)
        p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))))
        v = pd.Series(rng.uniform(1, 2, 30))
        L = 5
        s = institutional_momentum(p, v, L)
        t = 29
        lp = np.log(p.to_numpy())
        R = lp[t] - lp[t - L]
        sig = np.std(np.diff(lp)[t - L : t], ddof=1)
        vr = v.iloc[t - L + 1 : t + 1].mean() / v.iloc[t - 2 * L + 1 : t - L + 1].mean()
        expect = math.tanh(R / (sig * math.sqrt(L))) * min(vr, 2) / 2
        self.assertAlmostEqual(s.iloc[t], expect, places=10)

    def test_zero_prior_volume_nan(self):
        s = institutional_momentum(pd.Series(100 * 1.01 ** np.arange(6)), pd.Series([0.0, 0, 0, 1, 1, 1]), 3)
        self.assertTrue(math.isnan(s.iloc[5]))

    def test_reproducible(self):
        p = pd.Series(100 * 1.01 ** np.arange(12))
        v = pd.Series(np.arange(1.0, 13))
        pd.testing.assert_series_equal(institutional_momentum(p, v, 3), institutional_momentum(p, v, 3))


class TestPVCorr(unittest.TestCase):
    def test_constant_is_nan(self):
        c = price_volume_correlation(pd.Series([100.0] * 10), pd.Series([1.0] * 10), 3)
        self.assertTrue(c.isna().all())

    def test_perfect_positive(self):
        p = pd.Series(np.exp(np.cumsum([0, 0.01, 0.02, 0.03, 0.04, 0.05])))
        v = pd.Series(np.exp(np.cumsum([0, 0.01, 0.02, 0.03, 0.04, 0.05])))
        self.assertAlmostEqual(price_volume_correlation(p, v, 3).iloc[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_regime_detector.py`

```python
"""RegimeDetectorAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.regime_detector import RegimeDetectorAgent
from core.models import SignalType


def st(ef, es, adx, rv):
    return make_state({"ema_fast": ef, "ema_slow": es, "adx": adx, "realized_vol": rv})


class TestRegimeDetector(unittest.TestCase):
    def setUp(self):
        self.p = config().agents.regime_detector
        self.agent = RegimeDetectorAgent(1.0, self.p)

    def test_trend_up(self):
        s = self.agent.analyze(st(110, 100, 30, 0.5))
        self.assertEqual(s.features["regime"], "trend_up")
        self.assertEqual(s.signal, SignalType.BUY)
        self.assertAlmostEqual(s.confidence, (30 - self.p.adx_trend) / (self.p.adx_strong - self.p.adx_trend))
        self.assertIn("ADX 30.0", s.reason)

    def test_trend_down(self):
        s = self.agent.analyze(st(90, 100, 45, 0.5))
        self.assertEqual(s.features["regime"], "trend_down")
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertEqual(s.confidence, 1.0)

    def test_range(self):
        s = self.agent.analyze(st(101, 100, 12, 0.5))
        self.assertEqual(s.features["regime"], "range")
        self.assertEqual(s.signal, SignalType.HOLD)

    def test_volatile_overrides_trend(self):
        s = self.agent.analyze(st(110, 100, 35, 2.0))
        self.assertEqual(s.features["regime"], "volatile")
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("volatility", s.reason)

    def test_warmup_unknown(self):
        s = self.agent.analyze(st(110, float("nan"), float("nan"), 0.5))
        self.assertFalse(s.available)
        self.assertEqual(s.features["regime"], "unknown")


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_risk_engine.py`

```python
"""RiskEngineAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.risk_engine import RiskEngineAgent
from core.models import SignalType


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.agent = RiskEngineAgent(1.0, self.cfg.agents.risk_engine, 0.8)

    def test_low_risk(self):
        s = self.agent.analyze(make_state({"realized_vol": 0.2, "atr_pct": 0.004, "market_drawdown": 0.0}))
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertEqual(s.features["risk_score"], 0.0)
        self.assertFalse(s.features["risk_veto"])
        self.assertEqual(s.confidence, 1.0)

    def test_hand_calculated_components(self):
        p = self.cfg.agents.risk_engine
        # vol: (0.9-0.3)/1.2 = 0.5; atr: (0.0225-0.005)/0.035 = 0.5; dd: 0.175/0.35 = 0.5
        s = self.agent.analyze(make_state({"realized_vol": 0.9, "atr_pct": 0.0225, "market_drawdown": -0.175}))
        for k in ("vol_score", "atr_score", "drawdown_score"):
            self.assertAlmostEqual(s.features[k], 0.5)
        self.assertAlmostEqual(s.features["risk_score"], 0.5)
        self.assertAlmostEqual(p.w_vol + p.w_atr + p.w_drawdown, 1.0)

    def test_extreme_risk_veto(self):
        s = self.agent.analyze(make_state({"realized_vol": 3.0, "atr_pct": 0.1, "market_drawdown": -0.5}))
        self.assertAlmostEqual(s.features["risk_score"], 1.0)
        self.assertTrue(s.features["risk_veto"])
        self.assertIn("ABOVE", s.reason)

    def test_missing_fails_safe(self):
        s = self.agent.analyze(make_state({"realized_vol": 0.5, "atr_pct": float("nan"), "market_drawdown": 0.0}))
        self.assertFalse(s.available)
        self.assertIn("fail-safe", s.reason)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_sentiment.py`

```python
"""Tests for indicators/sentiment.py."""

import math
import unittest

import helpers  # noqa: F401

from indicators.sentiment import composite_sentiment, funding_contrarian, normalize_score


class TestNormalize(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(normalize_score(75, 0, 100), 0.5)
        self.assertAlmostEqual(normalize_score(50, 0, 100), 0.0)
        self.assertEqual(normalize_score(150, 0, 100), 1.0)
        self.assertEqual(normalize_score(-5, 0, 100), -1.0)

    def test_missing_and_invalid(self):
        self.assertTrue(math.isnan(normalize_score(None, 0, 1)))
        self.assertTrue(math.isnan(normalize_score(float("nan"), 0, 1)))
        with self.assertRaises(ValueError):
            normalize_score(1, 1, 1)


class TestFunding(unittest.TestCase):
    def test_inside_threshold_zero(self):
        self.assertEqual(funding_contrarian(0.0003, 0.0003, 0.001), 0.0)
        self.assertEqual(funding_contrarian(-0.0002, 0.0003, 0.001), 0.0)

    def test_positive_is_bearish(self):
        self.assertAlmostEqual(funding_contrarian(0.00065, 0.0003, 0.001), -0.5)

    def test_negative_extreme_is_bullish_and_saturates(self):
        self.assertEqual(funding_contrarian(-0.01, 0.0003, 0.001), 1.0)

    def test_missing(self):
        self.assertTrue(math.isnan(funding_contrarian(None, 0.0003, 0.001)))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            funding_contrarian(0.0, 0.001, 0.001)


class TestComposite(unittest.TestCase):
    def test_hand_calculated_with_missing(self):
        c = composite_sentiment(0.5, float("nan"), -1.0, 0.35, 0.35, 0.30)
        self.assertAlmostEqual(c.score, (0.175 - 0.30) / 0.65)
        self.assertEqual(c.available_components, 2)
        self.assertAlmostEqual(c.weight_coverage, 0.65)

    def test_component_directions(self):
        self.assertGreater(composite_sentiment(1.0, None, None, 1, 1, 1).score, 0)
        self.assertLess(composite_sentiment(None, -1.0, None, 1, 1, 1).score, 0)
        self.assertLess(composite_sentiment(None, None, funding_contrarian(0.002, 0.0003, 0.001), 1, 1, 1).score, 0)

    def test_all_missing_nan(self):
        c = composite_sentiment(None, None, None, 1, 1, 1)
        self.assertTrue(math.isnan(c.score))
        self.assertEqual(c.available_components, 0)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            composite_sentiment(0, 0, 0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_sentiment_engine.py`

```python
"""SentimentEngineAgent on hand-crafted states."""

import unittest

from helpers import config, make_state

from agents.sentiment_engine import SentimentEngineAgent
from core.models import SignalType


class TestSentimentEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.agent = SentimentEngineAgent(1.0, self.cfg.agents.sentiment_engine)

    def test_positive_news_social_buy(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 90, "news_sentiment": 0.8, "funding_rate": 0.0001}))
        self.assertEqual(s.signal, SignalType.BUY)
        # social 0.8, news 0.8, funding term 0 -> (0.35*0.8+0.35*0.8+0)/1.0 = 0.56
        self.assertAlmostEqual(s.features["composite"], 0.56)
        self.assertAlmostEqual(s.confidence, 0.56)

    def test_negative_news_social_sell(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 10, "news_sentiment": -0.8, "funding_rate": 0.0}))
        self.assertEqual(s.signal, SignalType.SELL)

    def test_extreme_positive_funding_contrarian_sell(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 50, "news_sentiment": 0.0, "funding_rate": 0.003}))
        self.assertEqual(s.features["funding_term"], -1.0)
        self.assertEqual(s.signal, SignalType.SELL)
        self.assertIn("contrarian bearish", s.reason)

    def test_extreme_negative_funding_contrarian_buy(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 50, "news_sentiment": 0.0, "funding_rate": -0.003}))
        self.assertEqual(s.signal, SignalType.BUY)

    def test_missing_feed_not_neutral(self):
        # news missing: composite uses social + funding only and coverage reduces confidence
        s = self.agent.analyze(make_state(raw={"social_sentiment": 90, "funding_rate": 0.0001}, unavailable=("news_sentiment",)))
        self.assertAlmostEqual(s.features["composite"], (0.35 * 0.8) / 0.65)
        self.assertAlmostEqual(s.features["weight_coverage"], 0.65)
        self.assertIn("missing: news_sentiment", s.reason)

    def test_too_few_components_unavailable(self):
        s = self.agent.analyze(make_state(raw={"social_sentiment": 90}, unavailable=("news_sentiment", "funding_rate")))
        self.assertFalse(s.available)
        self.assertEqual(s.signal, SignalType.HOLD)
        self.assertIn("not treated as neutral", s.reason)


if __name__ == "__main__":
    unittest.main()
```

### `tests/test_volatility.py`

```python
"""Hand-calculated tests for indicators/volatility.py and indicators/trend.py."""

import math
import unittest

import helpers  # noqa: F401
import numpy as np
import pandas as pd

from indicators.trend import adx, ema
from indicators.volatility import (
    atr, atr_percent, garman_klass_volatility, parkinson_volatility, realized_volatility, rolling_drawdown,
    true_range, validate_ohlc, wilder_smooth,
)


class TestRealizedVol(unittest.TestCase):
    def test_hand_calculated(self):
        rv = realized_volatility(pd.Series([100.0, 110.0, 99.0]), 2, 1)
        r1, r2 = math.log(1.1), math.log(0.9)
        self.assertAlmostEqual(rv.iloc[2], abs(r1 - r2) / math.sqrt(2))
        self.assertTrue(math.isnan(rv.iloc[1]))

    def test_annualization(self):
        rv1 = realized_volatility(pd.Series([100.0, 110.0, 99.0]), 2, 1).iloc[2]
        rv4 = realized_volatility(pd.Series([100.0, 110.0, 99.0]), 2, 4).iloc[2]
        self.assertAlmostEqual(rv4, 2 * rv1)

    def test_constant_zero_variance(self):
        self.assertEqual(realized_volatility(pd.Series([5.0] * 5), 3, 365).iloc[-1], 0.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            realized_volatility(pd.Series([1.0, 0.0, 1.0]), 2, 1)
        with self.assertRaises(ValueError):
            realized_volatility(pd.Series([1.0, 2.0]), 1, 1)


class TestATR(unittest.TestCase):
    def setUp(self):
        self.h = pd.Series([10.0, 11, 12, 13])
        self.l = pd.Series([9.0, 10, 11, 12])
        self.c = pd.Series([9.5, 10.5, 11.5, 12.5])

    def test_true_range(self):
        self.assertEqual(list(true_range(self.h, self.l, self.c)), [1.0, 1.5, 1.5, 1.5])

    def test_wilder_hand_calculated(self):
        a = atr(self.h, self.l, self.c, 2)
        self.assertTrue(math.isnan(a.iloc[0]))
        self.assertAlmostEqual(a.iloc[1], 1.25)
        self.assertAlmostEqual(a.iloc[2], 1.375)
        self.assertAlmostEqual(a.iloc[3], 1.4375)
        self.assertAlmostEqual(atr_percent(self.h, self.l, self.c, 2).iloc[3], 1.4375 / 12.5)

    def test_wilder_insufficient(self):
        self.assertTrue(wilder_smooth(pd.Series([1.0]), 3).isna().all())


class TestRangeEstimators(unittest.TestCase):
    def test_parkinson(self):
        low = pd.Series([10.0, 10.0])
        high = low * math.e
        v = parkinson_volatility(high, low, 2, 4).iloc[1]
        self.assertAlmostEqual(v, math.sqrt(1 / (4 * math.log(2))) * 2)

    def test_garman_klass(self):
        low = pd.Series([10.0, 10.0])
        high = low * math.e
        o = c = pd.Series([15.0, 15.0])
        self.assertAlmostEqual(garman_klass_volatility(o, high, low, c, 2, 1).iloc[1], math.sqrt(0.5))

    def test_zero_range_is_zero(self):
        p = pd.Series([10.0, 10.0, 10.0])
        self.assertEqual(parkinson_volatility(p, p, 2, 1).iloc[-1], 0.0)
        self.assertEqual(garman_klass_volatility(p, p, p, p, 2, 1).iloc[-1], 0.0)


class TestDrawdownAndValidation(unittest.TestCase):
    def test_drawdown(self):
        d = rolling_drawdown(pd.Series([100.0, 120.0, 90.0]), 3)
        self.assertAlmostEqual(d.iloc[2], -0.25)
        self.assertTrue(math.isnan(d.iloc[1]))

    def test_validate_ohlc(self):
        o = pd.Series([10.0])
        with self.assertRaises(ValueError):
            validate_ohlc(o, pd.Series([9.0]), pd.Series([8.0]), pd.Series([10.0]))
        with self.assertRaises(ValueError):
            validate_ohlc(o, pd.Series([11.0]), pd.Series([10.5]), pd.Series([10.0]))
        validate_ohlc(o, pd.Series([11.0]), pd.Series([9.0]), pd.Series([10.5]))


class TestTrend(unittest.TestCase):
    def test_ema_warmup_and_constant(self):
        e = ema(pd.Series([5.0] * 6), 3)
        self.assertTrue(e.iloc[:2].isna().all())
        self.assertAlmostEqual(e.iloc[-1], 5.0)

    def test_adx_strong_uptrend(self):
        n = 60
        c = pd.Series(100 + np.arange(n, dtype=float))
        out = adx(c + 0.5, c - 0.5, c, 14)
        self.assertTrue(out["adx"].iloc[: 2 * 14 - 1].isna().all())
        self.assertFalse(math.isnan(out["adx"].iloc[2 * 14 - 1]))
        self.assertGreater(out["adx"].iloc[-1], 90)
        self.assertGreater(out["plus_di"].iloc[-1], out["minus_di"].iloc[-1])

    def test_adx_flat_zero(self):
        c = pd.Series([100.0] * 40)
        out = adx(c + 1, c - 1, c, 5)
        self.assertEqual(out["adx"].iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
```


## Deliverable 7 — What to build next


This ranking is a research hypothesis, not a guarantee. The framework's biggest weakness is that every
signal is evaluated on synthetic data, so the first priorities are about getting real evidence, not adding
features.

1. **Real trade-level and order-book data (aggressor-flagged prints, L2 depth, funding history).** OFI, CVD,
   sweeps and absorption are only meaningful on genuine aggressor and depth data. Today they are synthetic.
2. **Statistical validation on real data.** This means many walk-forward folds across symbols, block-bootstrap
   confidence intervals for Sharpe, and multiple-testing control (deflated Sharpe) for the threshold grid.
   Synthetic out-of-sample results are not evidence of alpha.
3. **Transaction-cost and market-impact model.** This would replace fixed bps with spread-, depth- and
   participation-dependent costs. Turnover-heavy agents may only look good under the flat-cost assumption.
4. **Agent-level calibration and regime-conditional evaluation.** This would score each agent per regime on
   matured outcomes and learn regime weights from data instead of the illustrative table.
5. **Portfolio-level sizing and multi-asset risk.** This means volatility targeting across instruments,
   correlation-aware exposure caps, and stop/exit rules decoupled from the entry vote, which is currently the
   main driver of the long holding periods.


## Deliverable 8 — Assumptions and limitations


## Implemented

* All five core agents, the optional OnChain agent (feature flag), the boss with weighted vote,
  agreement/coverage/strength, the risk veto, the kill switch and a template reasoner.
* Regime-conditional weights, optional family normalization, Kalman confidence smoothing.
* Backtester with next-bar-open fills, fees and slippage on every fill, a long-only default with an optional
  short mode, SELL close/reduce policies, risk-veto exposure reduction, and an unfilled-order report at
  window end.
* Benchmarks: no-trade, buy-and-hold and EMA trend, all on the identical window and costs.
* Evaluator (matured-only hit rate, confidence-weighted return, standalone PnL, signal-level incremental
  contribution) and execution-level ablation. Sensitivity grid via `--sensitivity`.
* Feedback loop (training only), walk-forward with embargo, validation threshold selection, train-fitted
  logistic calibration scored on test.
* Adversarial shadow agent, paper executor that logs simulated orders, live-feed stub, config and data
  validation, CLI and 168 unittest tests.

## Simplified approximations

* **Synthetic data:** regime-switching random walk. Order flow, depth, news and on-chain series are linked
  to the hidden regime drift (`flow_signal_strength=0.25`). Social sentiment and funding react to past
  returns. This link is a design choice, so any edge seen on synthetic data partly reflects how the
  generator was written.
* **CVD:** with synthetic data, `trade_side` is a net aggressor fraction. Without it, CVD falls back to a
  tick-rule proxy, flagged `cvd_is_proxy`.
* **Absorption ratio:** relative volume / relative range. It cannot separate absorption from two-sided churn
  without real aggressor data.
* **Costs:** fixed bps slippage and fees. There is no spread model, no market impact, no funding payments on
  positions and no borrow cost for shorts.
* **Mark-to-market:** final equity is marked at the last close with no liquidation cost. Buy-and-hold uses
  the same convention.
* **Walk-forward:** each segment starts flat with fresh capital and fresh Kalman/kill-switch state.
  Out-of-sample results are chained test-window returns, so a position still open at the end of a test
  window is marked, not closed. That is why OOS "closed trades" can be 0.
* **Hit rate:** uses close-to-close forward returns from the decision bar, not the executed fill price.
* **Incremental t-stat:** uses n_eff = n/h as a simple overlap correction, not a HAC estimator.
* **Kill switch:** the rolling Sharpe is undefined while flat (zero variance), so it can only trigger
  through drawdown then.
* **Confidence:** reported as an uncalibrated ensemble strength. Calibration is a diagnostic in the
  walk-forward report and is not used for sizing.
* **Symbol:** in synthetic mode, `--symbol` is a label only.

## Optional features not implemented

* Options-flow agent (on-chain was chosen instead).
* Ensemble of regime-specific bosses with a router (future extension).
* External LLM reasoner. The `Reasoner` protocol exists, but no LLM implementation is included.

## Requires external market data

* Genuine OFI/CVD (aggressor-flagged trades), order-book depth, social/news sentiment, funding history, and
  on-chain netflow/stablecoin supply. `LiveFeedStub` raises until a provider is implemented.

## Requires an exchange or broker integration

* Live order routing. `BrokerAdapter` is defined and `NullBroker` refuses all orders; `PaperExecutor` cannot
  be constructed with live orders enabled.
* Real fills, partial fills, order rejections, exchange fees/tiers and latency.

## Verification status

* The test suite (168 tests) and every documented CLI command were run on Python 3.13.16 with numpy 2.5.3,
  pandas 3.0.5, pydantic 2.13.5. All tests pass.
* All files compile under Python 3.11. The test suite could not be run on 3.11 in the build sandbox because
  package downloads were blocked there.
