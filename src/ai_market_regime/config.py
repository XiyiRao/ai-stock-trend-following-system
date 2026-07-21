from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemConfig:
    """Central parameters for the first market-regime layer."""

    ai_stocks: tuple[str, ...] = ("NVDA", "AVGO", "AMD", "SMCI", "ARM")
    nasdaq_ticker: str = "^NDX"
    semiconductor_ticker: str = "^SOX"
    treasury_ticker: str = "^TNX"
    target_ticker: str = "300308.SZ"
    start_date: str = "2018-01-01"
    momentum_days: int = 20
    trend_days: int = 60
    percentile_window: int = 1260
    percentile_min_periods: int = 252
    minimum_ai_constituents: int = 3
    transaction_cost_bps: float = 10.0
    score_weights: dict[str, float] = field(
        default_factory=lambda: {"ai_trend": 0.50, "semiconductor": 0.30, "liquidity": 0.20}
    )

    @property
    def all_tickers(self) -> tuple[str, ...]:
        return self.ai_stocks + (
            self.nasdaq_ticker,
            self.semiconductor_ticker,
            self.treasury_ticker,
            self.target_ticker,
        )
