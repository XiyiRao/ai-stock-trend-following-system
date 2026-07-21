import numpy as np
import pandas as pd
import pytest

from ai_market_regime import data as data_module
from ai_market_regime.alignment import align_us_scores_to_china_dates, build_alignment_audit
from ai_market_regime.china_data import DataQualityError, standardize_china_ohlcv, validate_china_ohlcv
from ai_market_regime.choice_data import standardize_choice_frame
from ai_market_regime.config import SystemConfig
from ai_market_regime.scoring import build_backtest, build_market_scores, position_from_score, regime_cap_from_score, rolling_percentile
from ai_market_regime.stock_scoring import build_stock_scores, combine_market_and_stock_scores


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


def test_choice_export_discards_footer_and_validates_symbol():
    raw = pd.DataFrame(
        {
            "证券代码": [300308.0, 300308.0, "数据来源：妙想Choice"],
            "交易时间": ["2026-07-20", "2026-07-21", None],
            "开盘价": [100.0, 102.0, np.nan],
            "最高价": [105.0, 110.0, np.nan],
            "最低价": [99.0, 101.0, np.nan],
            "收盘价": [104.0, 109.0, np.nan],
            "成交量": [1000.0, 1200.0, np.nan],
            "成交额": [102000.0, 128000.0, np.nan],
        }
    )
    bars = standardize_choice_frame(raw, "300308")
    assert bars.index.tolist() == [pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21")]
    assert bars.loc[pd.Timestamp("2026-07-21"), "close"] == 109.0

    raw.loc[0, "证券代码"] = "000001"
    with pytest.raises(DataQualityError, match="unexpected symbols"):
        standardize_choice_frame(raw, "300308")


def _rising_china_bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(np.linspace(50.0, 150.0, len(index)), index=index)
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(index)),
            "amount": close * np.linspace(1_000_000, 2_000_000, len(index)),
        },
        index=index,
    )


def test_stock_score_has_five_bounded_components_and_trend_confirmation():
    index = pd.bdate_range("2022-01-03", periods=360)
    config = SystemConfig(percentile_window=120, percentile_min_periods=60)
    result = build_stock_scores(_rising_china_bars(index), config)
    latest = result.iloc[-1]
    components = [
        latest["Trend_Score"],
        latest["Momentum_Score"],
        latest["Volume_Price_Score"],
        latest["Breakout_Score"],
        latest["Risk_Quality_Score"],
    ]
    assert np.isclose(latest["STOCK_SCORE"], sum(components))
    assert 0 <= latest["STOCK_SCORE"] <= 100
    assert 0 <= latest["Trend_Score"] <= 30
    assert 0 <= latest["Momentum_Score"] <= 25
    assert 0 <= latest["Volume_Price_Score"] <= 15
    assert 0 <= latest["Breakout_Score"] <= 15
    assert 0 <= latest["Risk_Quality_Score"] <= 15
    assert bool(latest["Trend_Eligible"])


def test_stock_confirmation_and_drawdown_never_exceed_market_cap():
    index = pd.bdate_range("2024-01-02", periods=3)
    config = SystemConfig(stock_minimum_score=55, rebalance_threshold=0.10)
    market = pd.DataFrame({"Target_Position": [0.8, 0.8, 0.8]}, index=index)
    stock = pd.DataFrame(
        {
            "STOCK_SCORE": [80.0, 80.0, 80.0],
            "Trend_Eligible": [True, True, True],
            "Drawdown60": [-0.05, -0.16, -0.23],
            "close": [100.0, 95.0, 90.0],
            "MA120": [80.0, 80.0, 80.0],
        },
        index=index,
    )
    result = combine_market_and_stock_scores(market, stock, config)
    assert result["Risk_Adjusted_Position"].tolist() == [0.8, 0.4, 0.0]
    assert (result["Final_Target_Position"] <= result["Market_Position_Cap"]).all()
    assert result.iloc[-1]["Risk_Rule"] == "60日回撤清仓"


def test_failed_stock_trend_forces_zero_position_even_in_strong_market():
    index = pd.bdate_range("2024-01-02", periods=2)
    config = SystemConfig()
    market = pd.DataFrame({"Target_Position": [0.8, 0.8]}, index=index)
    stock = pd.DataFrame(
        {
            "STOCK_SCORE": [40.0, 50.0],
            "Trend_Eligible": [False, False],
            "Drawdown60": [-0.01, -0.02],
            "close": [100.0, 101.0],
            "MA120": [90.0, 91.0],
        },
        index=index,
    )
    result = combine_market_and_stock_scores(market, stock, config)
    assert result["Final_Target_Position"].eq(0.0).all()
    assert result["Position_Conclusion"].eq("空仓/观望").all()
