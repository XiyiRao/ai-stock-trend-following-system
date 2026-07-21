"""AI market-regime, stock scoring, backtesting, and paper-trading package."""

from .backtest import performance_metrics, run_event_backtest
from .config import SystemConfig
from .scoring import build_market_scores
from .stock_scoring import build_stock_scores, combine_market_and_stock_scores

__all__ = [
    "SystemConfig",
    "build_market_scores",
    "build_stock_scores",
    "combine_market_and_stock_scores",
    "run_event_backtest",
    "performance_metrics",
]