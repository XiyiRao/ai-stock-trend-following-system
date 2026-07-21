from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemConfig:
    """Parameters for the guide-aligned V0/V1 market-regime layer."""

    ai_stocks: tuple[str, ...] = ("NVDA", "AVGO", "AMD", "ANET", "SMCI")
    growth_ticker: str = "QQQ"
    semiconductor_ticker: str = "^SOX"
    treasury_ticker: str = "^TNX"
    target_symbol: str = "300308"
    start_date: str = "2018-01-01"
    momentum_days: int = 20
    long_momentum_days: int = 60
    trend_days: int = 60
    long_trend_days: int = 120
    percentile_window: int = 756
    percentile_min_periods: int = 252
    minimum_ai_constituents: int = 3
    transaction_cost_bps: float = 10.0
    max_target_position: float = 0.80
    stock_minimum_score: float = 55.0
    breakout_days: int = 120
    atr_days: int = 20
    peak_days: int = 60
    drawdown_reduce_at: float = 0.15
    drawdown_exit_at: float = 0.22
    drawdown_reduced_multiplier: float = 0.50
    atr_stop_multiple: float = 3.0
    rebalance_threshold: float = 0.10
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ai_momentum": 0.40,
            "semiconductor": 0.25,
            "growth": 0.20,
            "rates": 0.15,
        }
    )

    @property
    def all_tickers(self) -> tuple[str, ...]:
        return self.ai_stocks + (
            self.growth_ticker,
            self.semiconductor_ticker,
            self.treasury_ticker,
        )
