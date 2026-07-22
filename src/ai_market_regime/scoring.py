from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SystemConfig


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rank the latest value within its trailing history on a 0-100 scale."""

    def rank_latest(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[~np.isnan(values)]
        if np.isnan(current) or len(valid) == 0:
            return np.nan
        less = np.sum(valid < current)
        equal = np.sum(valid == current)
        return float(100.0 * (less + 0.5 * equal) / len(valid))

    return series.rolling(window=window, min_periods=min_periods).apply(rank_latest, raw=True)


def regime_cap_from_score(score: pd.Series) -> pd.Series:
    values = np.select(
        [score >= 75, score >= 60, score >= 45, score >= 30],
        [0.65, 0.50, 0.25, 0.10],
        default=0.05,
    )
    return pd.Series(values, index=score.index, dtype=float).where(score.notna())


def position_from_score(score: pd.Series, maximum: float = 0.65) -> pd.Series:
    return regime_cap_from_score(score).clip(upper=maximum)


def regime_from_score(score: pd.Series) -> pd.Series:
    values = np.select(
        [score >= 75, score >= 60, score >= 45, score >= 30],
        ["积极持有", "正常持有", "防御持有", "谨慎观察"],
        default="最低观察仓",
    )
    return pd.Series(values, index=score.index, dtype="object").where(score.notna())


def _binary_score(condition: pd.Series, valid: pd.Series) -> pd.Series:
    return (condition.astype(float) * 100.0).where(valid)


def _trend_score(
    price: pd.Series,
    momentum20: pd.Series,
    ma60: pd.Series,
    ma120: pd.Series,
    config: SystemConfig,
) -> pd.Series:
    momentum_pct = rolling_percentile(
        momentum20, config.percentile_window, config.percentile_min_periods
    )
    above_ma60 = _binary_score(price > ma60, price.notna() & ma60.notna())
    ma60_above_ma120 = _binary_score(ma60 > ma120, ma60.notna() & ma120.notna())
    return 0.40 * momentum_pct + 0.30 * above_ma60 + 0.30 * ma60_above_ma120


def build_market_scores(close: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    """Calculate the guide-aligned four-factor US market-regime score."""

    missing = sorted(set(config.all_tickers).difference(close.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    close = close.sort_index().copy()
    return20 = close.pct_change(config.momentum_days, fill_method=None)
    return60 = close.pct_change(config.long_momentum_days, fill_method=None)
    ma60 = close.rolling(config.trend_days, min_periods=config.trend_days).mean()
    ma120 = close.rolling(config.long_trend_days, min_periods=config.long_trend_days).mean()

    ai_columns = list(config.ai_stocks)
    available20 = return20[ai_columns].notna().sum(axis=1)
    available60 = return60[ai_columns].notna().sum(axis=1)
    ai_return20 = return20[ai_columns].mean(axis=1, skipna=True).where(
        available20 >= config.minimum_ai_constituents
    )
    ai_return60 = return60[ai_columns].mean(axis=1, skipna=True).where(
        available60 >= config.minimum_ai_constituents
    )
    ai_pct20 = rolling_percentile(
        ai_return20, config.percentile_window, config.percentile_min_periods
    )
    ai_pct60 = rolling_percentile(
        ai_return60, config.percentile_window, config.percentile_min_periods
    )
    ai_momentum_score = 0.60 * ai_pct20 + 0.40 * ai_pct60

    sox = close[config.semiconductor_ticker]
    semiconductor_score = _trend_score(
        sox,
        return20[config.semiconductor_ticker],
        ma60[config.semiconductor_ticker],
        ma120[config.semiconductor_ticker],
        config,
    )

    growth = close[config.growth_ticker]
    growth_score = _trend_score(
        growth,
        return20[config.growth_ticker],
        ma60[config.growth_ticker],
        ma120[config.growth_ticker],
        config,
    )

    yield_change20 = close[config.treasury_ticker].diff(config.momentum_days)
    rates_score = 100.0 - rolling_percentile(
        yield_change20, config.percentile_window, config.percentile_min_periods
    )

    weights = config.score_weights
    ai_score = (
        weights["ai_momentum"] * ai_momentum_score
        + weights["semiconductor"] * semiconductor_score
        + weights["growth"] * growth_score
        + weights["rates"] * rates_score
    ).clip(0, 100)

    result = pd.DataFrame(
        {
            "AI_Return20": ai_return20,
            "AI_Return60": ai_return60,
            "AI_Momentum_Score": ai_momentum_score,
            "SOX_Close": sox,
            "SOX_MA60": ma60[config.semiconductor_ticker],
            "SOX_MA120": ma120[config.semiconductor_ticker],
            "Semiconductor_Score": semiconductor_score,
            "QQQ_Close": growth,
            "QQQ_MA60": ma60[config.growth_ticker],
            "QQQ_MA120": ma120[config.growth_ticker],
            "Growth_Score": growth_score,
            "TNX_Close": close[config.treasury_ticker],
            "TNX_Change20": yield_change20,
            "Rates_Score": rates_score,
            "AI_SCORE": ai_score,
        },
        index=close.index,
    )
    result["Market_Regime"] = regime_from_score(result["AI_SCORE"])
    result["Regime_Cap"] = regime_cap_from_score(result["AI_SCORE"])
    result["Target_Position"] = position_from_score(
        result["AI_SCORE"], maximum=config.max_target_position
    )
    result.index.name = "US_Date"
    return result


def build_backtest(
    scores: pd.DataFrame,
    target_close: pd.Series,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Research baseline: execute the prior China-session target on the next row."""

    position_column = (
        "Final_Target_Position"
        if "Final_Target_Position" in scores.columns
        else "Target_Position"
    )
    aligned = scores[["AI_SCORE", position_column]].rename(
        columns={position_column: "Target_Position"}
    ).join(target_close.rename("Target_Close"), how="left")
    aligned["Target_Return"] = aligned["Target_Close"].pct_change(fill_method=None)
    aligned["Executed_Position"] = aligned["Target_Position"].shift(1).fillna(0.0)
    turnover = aligned["Executed_Position"].diff().abs().fillna(aligned["Executed_Position"].abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    aligned["Strategy_Return"] = (
        aligned["Executed_Position"] * aligned["Target_Return"].fillna(0.0) - cost
    )
    aligned["BuyHold_Equity"] = (1.0 + aligned["Target_Return"].fillna(0.0)).cumprod()
    aligned["Strategy_Equity"] = (1.0 + aligned["Strategy_Return"]).cumprod()
    return aligned
