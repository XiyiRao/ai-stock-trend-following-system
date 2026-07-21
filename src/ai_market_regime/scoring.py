from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SystemConfig


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rank the latest value in its trailing window on a 0-100 scale."""

    def rank_latest(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[~np.isnan(values)]
        if np.isnan(current) or len(valid) == 0:
            return np.nan
        less = np.sum(valid < current)
        equal = np.sum(valid == current)
        return float(100.0 * (less + 0.5 * equal) / len(valid))

    return series.rolling(window=window, min_periods=min_periods).apply(rank_latest, raw=True)


def position_from_score(score: pd.Series) -> pd.Series:
    values = np.select([score > 70, score >= 50, score >= 30], [1.00, 0.70, 0.30], default=0.00)
    result = pd.Series(values, index=score.index, dtype=float)
    return result.where(score.notna())


def regime_from_score(score: pd.Series) -> pd.Series:
    values = np.select(
        [score > 70, score >= 50, score >= 30],
        ["激进做多", "正常持仓", "降低仓位"],
        default="空仓",
    )
    result = pd.Series(values, index=score.index, dtype="object")
    return result.where(score.notna())


def build_market_scores(close: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    """Create component scores, composite AI_SCORE, regime, and target position."""

    missing = sorted(set(config.all_tickers).difference(close.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    close = close.sort_index().copy()
    momentum = close.pct_change(config.momentum_days, fill_method=None)
    ma60 = close.rolling(config.trend_days, min_periods=config.trend_days).mean()

    ai_columns = list(config.ai_stocks)
    ai_prices = close.loc[:, ai_columns]
    ai_returns = momentum.loc[:, ai_columns]
    available_count = ai_returns.notna().sum(axis=1)
    ai_return20 = ai_returns.mean(axis=1, skipna=True).where(
        available_count >= config.minimum_ai_constituents
    )
    ai_momentum_pct = rolling_percentile(
        ai_return20, config.percentile_window, config.percentile_min_periods
    )
    ai_ma60 = ma60.loc[:, ai_columns]
    ai_breadth = (ai_prices > ai_ma60).where(ai_prices.notna() & ai_ma60.notna()).mean(axis=1) * 100.0
    ai_trend_score = 0.70 * ai_momentum_pct + 0.30 * ai_breadth

    sox = close[config.semiconductor_ticker]
    sox_momentum_pct = rolling_percentile(
        momentum[config.semiconductor_ticker], config.percentile_window, config.percentile_min_periods
    )
    sox_above_ma60 = (sox > ma60[config.semiconductor_ticker]).astype(float) * 100.0
    sox_ma60_rising = (
        ma60[config.semiconductor_ticker] > ma60[config.semiconductor_ticker].shift(20)
    ).astype(float) * 100.0
    semiconductor_score = 0.50 * sox_momentum_pct + 0.30 * sox_above_ma60 + 0.20 * sox_ma60_rising

    yield20_change = close[config.treasury_ticker].diff(config.momentum_days)
    yield_change_pct = rolling_percentile(
        yield20_change, config.percentile_window, config.percentile_min_periods
    )
    falling_yield_score = 100.0 - yield_change_pct
    ndx = close[config.nasdaq_ticker]
    ndx_above_ma60 = (ndx > ma60[config.nasdaq_ticker]).astype(float) * 100.0
    ndx_momentum_pct = rolling_percentile(
        momentum[config.nasdaq_ticker], config.percentile_window, config.percentile_min_periods
    )
    liquidity_score = 0.50 * falling_yield_score + 0.30 * ndx_above_ma60 + 0.20 * ndx_momentum_pct

    weights = config.score_weights
    ai_score = (
        weights["ai_trend"] * ai_trend_score
        + weights["semiconductor"] * semiconductor_score
        + weights["liquidity"] * liquidity_score
    ).clip(0, 100)

    result = pd.DataFrame(
        {
            "AI_Return20": ai_return20,
            "AI_Breadth": ai_breadth,
            "AI_Trend_Score": ai_trend_score,
            "SOX_Close": sox,
            "SOX_MA60": ma60[config.semiconductor_ticker],
            "Semiconductor_Score": semiconductor_score,
            "TNX_Close": close[config.treasury_ticker],
            "TNX_Change20": yield20_change,
            "NDX_Close": ndx,
            "Liquidity_Score": liquidity_score,
            "AI_SCORE": ai_score,
        },
        index=close.index,
    )
    result["Market_Regime"] = regime_from_score(result["AI_SCORE"])
    result["Target_Position"] = position_from_score(result["AI_SCORE"])
    result.index.name = "Date"
    return result


def build_backtest(scores: pd.DataFrame, target_close: pd.Series, transaction_cost_bps: float = 10.0) -> pd.DataFrame:
    """Backtest next-session execution, with simple one-way transaction costs."""

    aligned = scores[["AI_SCORE", "Target_Position"]].join(target_close.rename("Target_Close"), how="left")
    aligned["Target_Return"] = aligned["Target_Close"].pct_change(fill_method=None)
    aligned["Executed_Position"] = aligned["Target_Position"].shift(1).fillna(0.0)
    turnover = aligned["Executed_Position"].diff().abs().fillna(aligned["Executed_Position"].abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    aligned["Strategy_Return"] = aligned["Executed_Position"] * aligned["Target_Return"].fillna(0.0) - cost
    aligned["BuyHold_Equity"] = (1.0 + aligned["Target_Return"].fillna(0.0)).cumprod()
    aligned["Strategy_Equity"] = (1.0 + aligned["Strategy_Return"]).cumprod()
    return aligned
