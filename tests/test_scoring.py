import numpy as np
import pandas as pd
import pytest

from ai_market_regime import data as data_module
from ai_market_regime.alignment import align_us_scores_to_china_dates, build_alignment_audit
from ai_market_regime.china_data import DataQualityError, standardize_china_ohlcv, validate_china_ohlcv
from ai_market_regime.config import SystemConfig
from ai_market_regime.scoring import build_backtest, build_market_scores, position_from_score, regime_cap_from_score, rolling_percentile


def test_position_boundaries_and_single_stock_cap():
    scores = pd.Series([75.0, 74.9, 60.0, 59.9, 45.0, 44.9, np.nan])
    pd.testing.assert_series_equal(regime_cap_from_score(scores), pd.Series([1.0, 0.7, 0.7, 0.3, 0.3, 0.0, np.nan]))
    pd.testing.assert_series_equal(position_from_score(scores, maximum=0.8), pd.Series([0.8, 0.7, 0.7, 0.3, 0.3, 0.0, np.nan]))


def test_rolling_percentile_uses_trailing_values():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = rolling_percentile(series, window=3, min_periods=3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] > 80
    assert result.iloc[3] > 80


def test_scores_are_bounded_and_recent_listing_is_supported():
    config = SystemConfig(percentile_window=80, percentile_min_periods=20, trend_days=10, long_trend_days=20, long_momentum_days=30)
    index = pd.bdate_range("2023-01-02", periods=160)
    rng = np.random.default_rng(7)
    values = {}
    for number, ticker in enumerate(config.all_tickers):
        returns = rng.normal(0.0008 + number * 0.00005, 0.01, len(index))
        values[ticker] = 100 * np.cumprod(1 + returns)
    close = pd.DataFrame(values, index=index)
    close.loc[index[:80], "ANET"] = np.nan
    scores = build_market_scores(close, config)
    valid = scores["AI_SCORE"].dropna()
    assert not valid.empty
    assert valid.between(0, 100).all()
    assert scores["Target_Position"].dropna().between(0, 0.8).all()
    assert pd.notna(scores.loc[index[60], "AI_Return20"])


def test_backtest_executes_signal_next_china_session():
    index = pd.bdate_range("2024-01-01", periods=4)
    scores = pd.DataFrame({"AI_SCORE": [80, 80, 40, 40], "Target_Position": [0.8, 0.8, 0.3, 0.3]}, index=index)
    prices = pd.Series([100, 110, 121, 121], index=index)
    result = build_backtest(scores, prices, transaction_cost_bps=0)
    assert result["Executed_Position"].tolist() == [0.0, 0.8, 0.8, 0.3]
    assert np.isclose(result.loc[index[1], "Strategy_Return"], 0.08)


def test_alignment_excludes_same_day_us_observation():
    us_dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    us_scores = pd.DataFrame({"AI_SCORE": [55.0, 70.0]}, index=us_dates)
    china_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    aligned = align_us_scores_to_china_dates(us_scores, china_dates)
    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-02"), "US_Source_Date"])
    assert aligned.loc[pd.Timestamp("2024-01-03"), "US_Source_Date"] == pd.Timestamp("2024-01-02")
    assert aligned.loc[pd.Timestamp("2024-01-04"), "US_Source_Date"] == pd.Timestamp("2024-01-02")
    assert aligned.loc[pd.Timestamp("2024-01-05"), "US_Source_Date"] == pd.Timestamp("2024-01-04")
    assert build_alignment_audit(aligned, sample_size=10)["Strictly_Earlier"].all()


def test_china_ohlcv_standardization_and_quality_gate():
    raw = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "开盘": [100, 102], "最高": [105, 106], "最低": [99, 101], "收盘": [104, 103], "成交量": [1000, 1200], "成交额": [102000, 124000]})
    bars = standardize_china_ohlcv(raw)
    assert validate_china_ohlcv(bars)["passed"] is True
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "amount"]
    broken = bars.copy()
    broken.loc[broken.index[0], "high"] = 98
    with pytest.raises(DataQualityError, match="high below"):
        validate_china_ohlcv(broken)


def test_download_falls_back_to_cache_and_keeps_qqq_dates(tmp_path, monkeypatch):
    config = SystemConfig()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    cached = pd.DataFrame(100.0, index=dates, columns=config.all_tickers)
    cached.loc[dates[1], config.growth_ticker] = np.nan
    cache_path = tmp_path / "close.csv"
    cached.to_csv(cache_path)
    monkeypatch.setattr(data_module.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(data_module, "_download_chart_close", lambda *_: pd.DataFrame())
    with pytest.warns(RuntimeWarning, match="local cache"):
        result = data_module.download_close_prices(config, cache_path)
    assert result.index.tolist() == [dates[0], dates[2]]
    assert result.loc[dates[2], config.ai_stocks[0]] == 100.0


def test_partial_batch_download_is_completed_by_chart_fallback(tmp_path, monkeypatch):
    config = SystemConfig()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    batch_columns = pd.MultiIndex.from_tuples([("Close", config.growth_ticker)])
    batch = pd.DataFrame([[100.0], [101.0]], index=dates, columns=batch_columns)
    fallback = pd.DataFrame(90.0, index=dates, columns=config.all_tickers)
    monkeypatch.setattr(data_module.yf, "download", lambda *args, **kwargs: batch)
    monkeypatch.setattr(data_module, "_download_chart_close", lambda *_: fallback)
    result = data_module.download_close_prices(config, tmp_path / "close.csv")
    assert result.loc[dates[1], config.growth_ticker] == 101.0
    assert all(result[ticker].notna().all() for ticker in config.all_tickers)
