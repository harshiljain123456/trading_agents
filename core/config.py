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
OPTIONAL_AGENTS: tuple[str, ...] = ("on_chain", "market_mood")
REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range", "volatile")
TIMEFRAME_BARS_PER_YEAR: dict[str, int] = {
    # Crypto trades 24/7, so bars per year = 365 days * bars per day.
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}
# NSE cash equities: ~250 sessions/year; 09:15-15:30 IST gives 7 hourly bars
# (the last one is 15 minutes long). 4h bars are not meaningful for NSE.
NSE_BARS_PER_YEAR: dict[str, int] = {"1h": 250 * 7, "1d": 250}
MARKET_TIMEZONE: dict[str, str] = {"crypto": "UTC", "nse": "Asia/Kolkata"}
TIMEFRAME_PANDAS_FREQ: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1D"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppSection(_Strict):
    seed: int = 42
    timezone: Literal["UTC", "Asia/Kolkata"] = "UTC"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class DataSection(_Strict):
    market: Literal["crypto", "nse"] = "crypto"
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
    sentiment_slot: Literal["crypto_sentiment", "market_mood"] = "crypto_sentiment"
    enable_optional_agent: bool = False
    optional_agent_type: Literal["on_chain"] = "on_chain"
    enable_kalman: bool = True
    enable_regime_weights: bool = True
    enable_adversarial: bool = True
    enable_family_normalization: bool = False


class IndicatorSection(_Strict):
    vix_rank_window: int = Field(250, ge=20)
    vix_change_window: int = Field(5, ge=1)
    index_ema: int = Field(100, ge=10)
    index_trend_scale: float = Field(0.05, gt=0)
    rs_lookback: int = Field(63, ge=5)
    rs_scale: float = Field(0.10, gt=0)
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
    allow_proxy_flow: bool = False
    proxy_confidence_scale: float = Field(0.5, gt=0, le=1)


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


class MarketMoodParams(_Strict):
    w_index_trend: float = Field(0.4, ge=0)
    w_rel_strength: float = Field(0.4, ge=0)
    w_vix: float = Field(0.2, ge=0)
    vix_high_pct: float = Field(0.90, gt=0.5, lt=1)
    vix_low_pct: float = Field(0.10, gt=0, lt=0.5)
    complacency_penalty: float = Field(0.5, ge=0, le=1)
    min_components: int = Field(2, ge=1, le=3)
    signal_threshold: float = Field(0.20, gt=0, lt=1)

    @model_validator(mode="after")
    def _w(self) -> "MarketMoodParams":
        if self.w_index_trend + self.w_rel_strength + self.w_vix <= 0:
            raise ValueError("market_mood component weights must sum to > 0")
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
            "market_mood": 1.0,
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
            "market_mood": "sentiment",
        }
    )
    market_intel: MarketIntelParams = Field(default_factory=MarketIntelParams)
    risk_engine: RiskEngineParams = Field(default_factory=RiskEngineParams)
    liquidity_engine: LiquidityParams = Field(default_factory=LiquidityParams)
    sentiment_engine: SentimentParams = Field(default_factory=SentimentParams)
    regime_detector: RegimeParams = Field(default_factory=RegimeParams)
    on_chain: OnChainParams = Field(default_factory=OnChainParams)
    market_mood: MarketMoodParams = Field(default_factory=MarketMoodParams)


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


class DesignsSection(_Strict):
    """Role-based alternative designs compared in Step 3 (fixed before testing)."""

    target_volatility: float = Field(0.30, gt=0)   # risk role: size = min(1, target_vol / realized_vol)
    rebalance_band: float = Field(0.10, ge=0, lt=1)  # ignore exposure changes smaller than this
    mood_cap: float = Field(0.5, ge=0, le=1)         # defensive: max exposure when market mood is bearish
    veto_cap: float = Field(0.5, ge=0, le=1)         # max exposure while the risk veto is active
    crash_windows: dict[str, tuple[str, str]] = Field(
        default_factory=lambda: {
            "2008 crash": ("2008-01-08", "2008-10-27"),
            "2020 covid": ("2020-01-17", "2020-03-23"),
        }
    )
    # Step 5 robustness test (all fixed before running it)
    cash_yield: float = Field(0.065, ge=0, lt=0.5)        # idle cash earns a liquid-fund rate (annual)
    stcg_rate: float = Field(0.20, ge=0, lt=1)            # equity short-term capital gains (held < 1 year)
    ltcg_rate: float = Field(0.125, ge=0, lt=1)           # equity long-term capital gains
    ltcg_exemption: float = Field(125_000.0, ge=0)        # LTCG exempt per financial year (rupees)
    tax_cess: float = Field(0.04, ge=0, lt=1)             # health & education cess on the tax
    ltcg_holding_days: int = Field(365, ge=1)
    robustness_capital: float = Field(1_000_000.0, gt=0)  # rupees; matters because the exemption is in rupees
    ema_grid: list[tuple[int, int]] = Field(default_factory=lambda: [(20, 100), (10, 50), (30, 150), (50, 200)])
    adx_grid: list[float] = Field(default_factory=lambda: [15.0, 20.0, 25.0])
    split_date: str = "2016-01-01"
    param_pass_share: float = Field(0.75, gt=0, le=1)     # share of settings C must win on


class CostsSection(_Strict):
    """Trading-cost model for the portfolio simulator (Steps 3c/5/7).

    ``flat``: fee_bps + slippage_bps on every rupee traded (earlier steps).
    ``india_delivery``: NSE equity-delivery charges as published by Zerodha (checked Oct 2026):
    STT 0.1% buy+sell, NSE transaction 0.00307%, SEBI Rs 10/crore, GST 18% on
    brokerage+transaction+SEBI, stamp duty 0.015% on buys, DP charge Rs 15.34 (incl. GST)
    per stock sold per day, brokerage 0 for delivery; plus slippage_bps.
    """

    model: Literal["flat", "india_delivery"] = "flat"
    stt: float = Field(0.001, ge=0)
    exchange_txn: float = Field(0.0000307, ge=0)
    sebi_fee: float = Field(0.000001, ge=0)
    stamp_duty_buy: float = Field(0.00015, ge=0)
    gst: float = Field(0.18, ge=0)
    brokerage_bps: float = Field(0.0, ge=0)
    brokerage_cap: float = Field(0.0, ge=0)        # rupees per order; 0 = no cap
    dp_charge_per_sell: float = Field(15.34, ge=0)  # rupees per stock sold per day, GST included


class FrameworkConfig(_Strict):
    """Root configuration object."""

    app: AppSection = Field(default_factory=AppSection)
    data: DataSection = Field(default_factory=DataSection)
    strategy: StrategySection = Field(default_factory=StrategySection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    costs: CostsSection = Field(default_factory=CostsSection)
    risk: RiskSection = Field(default_factory=RiskSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    walk_forward: WalkForwardSection = Field(default_factory=WalkForwardSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    indicators: IndicatorSection = Field(default_factory=IndicatorSection)
    agents: AgentsSection = Field(default_factory=AgentsSection)
    regime_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    kalman: KalmanSection = Field(default_factory=KalmanSection)
    synthetic: SyntheticSection = Field(default_factory=SyntheticSection)
    designs: DesignsSection = Field(default_factory=DesignsSection)

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
        if self.data.market == "nse" and self.data.timeframe == "4h":
            raise ValueError("data.timeframe '4h' is not supported for NSE; use '1d' or '1h'")
        if self.app.timezone != MARKET_TIMEZONE[self.data.market]:
            raise ValueError(
                f"app.timezone must be '{MARKET_TIMEZONE[self.data.market]}' for market '{self.data.market}'"
            )
        if ev.lookahead != self.strategy.prediction_horizon:
            raise ValueError("evaluation.lookahead must equal strategy.prediction_horizon")
        if not self.strategy.long_only and not self.strategy.allow_short:
            raise ValueError("set strategy.allow_short=true when strategy.long_only=false")
        return self

    @property
    def bars_per_year(self) -> int:
        """Annualization factor implied by ``data.market`` and ``data.timeframe``."""
        if self.data.market == "nse":
            return NSE_BARS_PER_YEAR[self.data.timeframe]
        return TIMEFRAME_BARS_PER_YEAR[self.data.timeframe]

    def enabled_agents(self) -> list[str]:
        """Names of agents that participate in the vote."""
        names = list(CORE_AGENTS)
        if self.features.sentiment_slot == "market_mood":
            names[names.index("sentiment_engine")] = "market_mood"
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
