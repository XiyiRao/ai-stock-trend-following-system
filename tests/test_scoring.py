import numpy as np
import pandas as pd
import pytest

from ai_market_regime.config import SystemConfig
from ai_market_regime import data as data_module
from ai_market_regime.scoring import build_backtest, build_market_scores, position_from_score, rolling_percentile


def test_position_boundaries():
    scores = pd.Series([71.0, 70.0, 50.0, 49.9, 30.0, 29.9, np.nan])
    actual = position_from_score(scores)
    expected = pd.Series([1.0, 0.7, 0.7, 0.3, 0.3, 0.0, np.nan])
    pd.testing.assert_series_equal(actual, expected)


def test_rolling_percentile_uses_trailing_values():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = rolling_percentile(series, window=3, min_periods=3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] > 80
    assert result.iloc[3] > 80


def test_scores_are_bounded_and_arm_missing_is_supported():
    config = SystemConfig(percentile_window=80, percentile_min_periods=20, trend_days=10)
    index = pd.bdate_range("2023-01-02", periods=140)
    rng = np.random.default_rng(7)
    data = {}
    for number, ticker in enumerate(config.all_tickers):
        returns = rng.normal(0.0008 + number * 0.00005, 0.01, len(index))
        data[ticker] = 100 * np.cumprod(1 + returns)
    close = pd.DataFrame(data, index=index)
    close.loc[index[:80], "ARM"] = np.nan
    scores = build_market_scores(close, config)
    valid = scores["AI_SCORE"].dropna()
    assert not valid.empty
    assert valid.between(0, 100).all()
    assert pd.notna(scores.loc[index[60], "AI_Return20"])


def test_backtest_executes_signal_next_session():
    index = pd.bdate_range("2024-01-01", periods=4)
    scores = pd.DataFrame({"AI_SCORE": [80, 80, 40, 40], "Target_Position": [1, 1, 0.3, 0.3]}, index=index)
    prices = pd.Series([100, 110, 121, 121], index=index)
    result = build_backtest(scores, prices, transaction_cost_bps=0)
    assert result["Executed_Position"].tolist() == [0.0, 1.0, 1.0, 0.3]
    assert np.isclose(result.loc[index[1], "Strategy_Return"], 0.10)


def test_download_falls_back_to_cache_and_keeps_nasdaq_dates(tmp_path, monkeypatch):
    config = SystemConfig()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    cached = pd.DataFrame(100.0, index=dates, columns=config.all_tickers)
    cached.loc[dates[1], config.nasdaq_ticker] = np.nan
    cached.loc[dates[2], config.target_ticker] = np.nan
    cached.loc[dates[1], config.target_ticker] = 101.0
    cache_path = tmp_path / "close.csv"
    cached.to_csv(cache_path)

    monkeypatch.setattr(data_module.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(data_module, "_download_chart_close", lambda *_: pd.DataFrame())

    with pytest.warns(RuntimeWarning, match="local cache"):
        result = data_module.download_close_prices(config, cache_path)

    assert result.index.tolist() == [dates[0], dates[2]]
    assert result.loc[dates[2], config.target_ticker] == 101.0


def test_partial_batch_download_is_completed_by_chart_fallback(tmp_path, monkeypatch):
    config = SystemConfig()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    batch_columns = pd.MultiIndex.from_tuples([("Close", config.nasdaq_ticker)])
    batch = pd.DataFrame([[100.0], [101.0]], index=dates, columns=batch_columns)
    fallback = pd.DataFrame(90.0, index=dates, columns=config.all_tickers)

    monkeypatch.setattr(data_module.yf, "download", lambda *args, **kwargs: batch)
    monkeypatch.setattr(data_module, "_download_chart_close", lambda *_: fallback)

    result = data_module.download_close_prices(config, tmp_path / "close.csv")

    assert result.loc[dates[1], config.nasdaq_ticker] == 101.0
    assert all(result[ticker].notna().all() for ticker in config.all_tickers)
