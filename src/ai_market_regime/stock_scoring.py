from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SystemConfig
from .scoring import rolling_percentile


REQUIRED_CHINA_COLUMNS = {"open", "high", "low", "close", "volume", "amount"}


def true_range(bars: pd.DataFrame) -> pd.Series:
    """Calculate daily true range without using future observations."""

    previous_close = bars["close"].shift(1)
    components = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1)


def _binary_points(condition: pd.Series, points: float) -> pd.Series:
    return condition.fillna(False).astype(float) * points


def build_stock_scores(bars: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    """Build the guide-aligned five-part STOCK_SCORE for 300308."""

    missing = sorted(REQUIRED_CHINA_COLUMNS.difference(bars.columns))
    if missing:
        raise ValueError(f"China bars missing columns: {', '.join(missing)}")

    frame = bars.sort_index().copy()
    close = frame["close"]
    frame["Return20"] = close.pct_change(config.momentum_days, fill_method=None)
    frame["Return60"] = close.pct_change(config.long_momentum_days, fill_method=None)
    frame["MA20"] = close.rolling(config.momentum_days).mean()
    frame["MA60"] = close.rolling(config.trend_days).mean()
    frame["MA120"] = close.rolling(config.long_trend_days).mean()
    frame["Volume_MA20"] = frame["volume"].rolling(config.momentum_days).mean()
    frame["Volume_MA60"] = frame["volume"].rolling(config.long_momentum_days).mean()
    frame["Volume_Ratio"] = frame["Volume_MA20"] / frame["Volume_MA60"]
    frame["Breakout_Ratio"] = close / close.rolling(config.breakout_days).max()
    frame["True_Range"] = true_range(frame)
    frame["ATR20"] = frame["True_Range"].rolling(config.atr_days).mean()
    frame["ATR_Pct"] = frame["ATR20"] / close
    frame["Peak60"] = close.rolling(config.peak_days).max()
    frame["Drawdown60"] = close / frame["Peak60"] - 1.0

    frame["Return20_Percentile"] = rolling_percentile(
        frame["Return20"], config.percentile_window, config.percentile_min_periods
    )
    frame["Return60_Percentile"] = rolling_percentile(
        frame["Return60"], config.percentile_window, config.percentile_min_periods
    )
    frame["Volume_Percentile"] = rolling_percentile(
        frame["Volume_Ratio"], config.percentile_window, config.percentile_min_periods
    )
    frame["ATR_Percentile"] = rolling_percentile(
        frame["ATR_Pct"], config.percentile_window, config.percentile_min_periods
    )

    frame["Trend_Score"] = (
        _binary_points(close > frame["MA20"], 7.5)
        + _binary_points(frame["MA20"] > frame["MA60"], 7.5)
        + _binary_points(frame["MA60"] > frame["MA120"], 7.5)
        + _binary_points(close > frame["MA120"], 7.5)
    )
    frame["Momentum_Score"] = (
        0.15 * frame["Return20_Percentile"]
        + 0.10 * frame["Return60_Percentile"]
    )
    frame["Volume_Price_Score"] = (
        (0.10 * frame["Volume_Percentile"]).clip(0, 10)
        + _binary_points((frame["Return20"] > 0) & (frame["Volume_Ratio"] > 1), 5)
    )
    frame["Breakout_Score"] = (
        (frame["Breakout_Ratio"].clip(0.85, 1.0) - 0.85) / 0.15 * 15
    ).clip(0, 15)
    volatility_quality = (100 - frame["ATR_Percentile"]).clip(0, 100) * 0.07
    drawdown_quality = ((frame["Drawdown60"] + 0.25) / 0.25).clip(0, 1) * 8
    frame["Risk_Quality_Score"] = volatility_quality + drawdown_quality
    frame["STOCK_SCORE"] = (
        frame["Trend_Score"]
        + frame["Momentum_Score"]
        + frame["Volume_Price_Score"]
        + frame["Breakout_Score"]
        + frame["Risk_Quality_Score"]
    ).clip(0, 100)

    frame["Trend_Eligible"] = (
        (frame["STOCK_SCORE"] >= config.stock_minimum_score)
        & (close > frame["MA60"])
        & (frame["MA20"] > frame["MA60"])
    )
    frame["ATR_Stop_Reference"] = frame["Peak60"] - config.atr_stop_multiple * frame["ATR20"]
    frame.index.name = "Date"
    return frame


def _filter_rebalances(targets: pd.Series, threshold: float) -> pd.Series:
    filtered: list[float] = []
    previous = 0.0
    for target in targets.fillna(0.0):
        target_value = float(target)
        if abs(target_value - previous) >= threshold:
            previous = target_value
        filtered.append(previous)
    return pd.Series(filtered, index=targets.index, dtype=float)


def _position_conclusion(position: float, raw_position: float) -> str:
    if position <= 0:
        return "空仓/观望"
    if position < raw_position:
        return "风险减仓"
    if position <= 0.30:
        return "小仓持有"
    if position <= 0.70:
        return "正常持有"
    return "积极持有"


def combine_market_and_stock_scores(
    market_scores: pd.DataFrame,
    stock_scores: pd.DataFrame,
    config: SystemConfig,
) -> pd.DataFrame:
    """Apply stock confirmation and drawdown controls below the market cap."""

    frame = market_scores.join(stock_scores, how="left")
    frame["Market_Position_Cap"] = frame["Target_Position"]
    frame["Raw_Target_Position"] = frame["Market_Position_Cap"].where(
        frame["Trend_Eligible"].fillna(False), 0.0
    )
    frame["Risk_Adjusted_Position"] = frame["Raw_Target_Position"].fillna(0.0)

    reduce_mask = frame["Drawdown60"] <= -config.drawdown_reduce_at
    exit_mask = frame["Drawdown60"] <= -config.drawdown_exit_at
    long_trend_exit = frame["close"] < frame["MA120"]
    frame.loc[reduce_mask, "Risk_Adjusted_Position"] *= config.drawdown_reduced_multiplier
    frame.loc[exit_mask | long_trend_exit, "Risk_Adjusted_Position"] = 0.0
    frame["Risk_Adjusted_Position"] = frame["Risk_Adjusted_Position"].clip(
        0.0, config.max_target_position
    )
    frame["Final_Target_Position"] = _filter_rebalances(
        frame["Risk_Adjusted_Position"], config.rebalance_threshold
    )

    risk_rule = pd.Series("正常", index=frame.index, dtype="object")
    risk_rule.loc[reduce_mask] = "60日回撤减仓"
    risk_rule.loc[long_trend_exit] = "跌破MA120清仓"
    risk_rule.loc[exit_mask] = "60日回撤清仓"
    frame["Risk_Rule"] = risk_rule
    frame["Position_Conclusion"] = [
        _position_conclusion(float(position), float(raw))
        for position, raw in zip(
            frame["Final_Target_Position"], frame["Raw_Target_Position"].fillna(0.0)
        )
    ]
    return frame