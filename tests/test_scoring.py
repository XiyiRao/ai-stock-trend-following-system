import numpy as np
import pandas as pd
import pytest

from ai_market_regime import data as data_module
from ai_market_regime.backtest import performance_metrics, run_event_backtest, trade_fees
from ai_market_regime.paper import run_paper_daily
from ai_market_regime.research import chronological_splits, positions_for_config
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


def _backtest_config(**overrides):
    values = {
        "initial_cash": 100_000.0,
        "commission_rate": 0.0,
        "minimum_commission": 0.0,
        "sell_stamp_duty_rate": 0.0,
        "slippage_bps": 0.0,
        "max_order_value_cny": 1_000_000.0,
        "max_daily_turnover_ratio": 1.0,
    }
    values.update(overrides)
    return SystemConfig(**values)


def test_event_backtest_executes_previous_signal_at_next_open():
    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [98.0, 99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 103.0, 104.0],
            "volume": [1_000_000] * 4,
            "Final_Target_Position": [0.8, 0.8, 0.0, 0.0],
            "ATR20": [5.0] * 4,
        },
        index=index,
    )
    curve, trades = run_event_backtest(frame, _backtest_config())
    assert trades.iloc[0]["side"] == "BUY"
    assert pd.Timestamp(trades.iloc[0]["signal_date"]) == index[0]
    assert pd.Timestamp(trades.iloc[0]["execution_date"]) == index[1]
    assert curve.loc[index[0], "shares"] == 0


def test_event_backtest_blocks_one_price_limit_up():
    index = pd.bdate_range("2019-01-02", periods=2)
    frame = pd.DataFrame(
        {
            "open": [100.0, 110.0],
            "high": [102.0, 110.0],
            "low": [98.0, 110.0],
            "close": [100.0, 110.0],
            "volume": [1_000_000, 1_000_000],
            "Final_Target_Position": [0.8, 0.8],
            "ATR20": [5.0, 5.0],
        },
        index=index,
    )
    curve, trades = run_event_backtest(frame, _backtest_config())
    assert trades.empty
    assert "一字涨停" in curve.loc[index[1], "blocked_reason"]


def test_event_backtest_atr_stop_uses_prior_information():
    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 80.0, 82.0],
            "high": [102.0, 122.0, 85.0, 84.0],
            "low": [98.0, 99.0, 75.0, 80.0],
            "close": [100.0, 120.0, 82.0, 83.0],
            "volume": [1_000_000] * 4,
            "Final_Target_Position": [0.8, 0.8, 0.8, 0.8],
            "ATR20": [10.0, 10.0, 10.0, 10.0],
        },
        index=index,
    )
    curve, trades = run_event_backtest(frame, _backtest_config(atr_stop_multiple=3.0))
    assert trades["side"].tolist()[:2] == ["BUY", "SELL"]
    assert trades.iloc[1]["reason"] == "atr_stop"
    assert bool(curve.loc[index[2], "atr_stop_triggered"])


def test_fee_model_applies_minimum_commission_and_sell_stamp_duty():
    config = SystemConfig(commission_rate=0.00015, minimum_commission=5.0, sell_stamp_duty_rate=0.0005)
    assert trade_fees(1_000.0, False, config) == 5.0
    assert trade_fees(10_000.0, True, config, pd.Timestamp("2024-01-02")) == 10.0
    assert trade_fees(10_000.0, True, config, pd.Timestamp("2023-01-02")) == 15.0


def test_research_splits_are_ordered_and_positions_respect_cap():
    index = pd.bdate_range("2020-01-02", periods=100)
    splits = chronological_splits(index)
    assert splits["train"].max() < splits["validation"].min()
    assert splits["validation"].max() < splits["out_of_sample"].min()
    frame = pd.DataFrame(
        {
            "STOCK_SCORE": 80.0,
            "close": 120.0,
            "MA20": 110.0,
            "MA60": 100.0,
            "MA120": 90.0,
            "Drawdown60": -0.05,
            "Market_Position_Cap": 0.8,
        },
        index=index,
    )
    result = positions_for_config(frame, SystemConfig(max_target_position=0.5))
    assert result["Final_Target_Position"].max() == 0.5


def test_paper_daily_is_idempotent_and_executes_prior_plan(tmp_path):
    config = _backtest_config(max_target_position=0.8)
    payload = {
        "china_signal_date": "2026-07-20",
        "target_position": 0.3,
    }
    first_bar = pd.Series({"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1_000_000})
    first = run_paper_daily(tmp_path, payload, first_bar, 99.0, config, "abc123")
    assert first["account"]["shares"] == 0
    assert first["reconciliation"]["simulation_days"] == 1

    payload["china_signal_date"] = "2026-07-21"
    second_bar = pd.Series({"open": 103.0, "high": 106.0, "low": 101.0, "close": 105.0, "volume": 1_000_000})
    second = run_paper_daily(tmp_path, payload, second_bar, 102.0, config, "abc123")
    assert second["execution"]["status"] == "filled"
    assert second["execution"]["side"] == "BUY"
    assert second["account"]["shares"] > 0
    assert second["reconciliation"]["simulation_days"] == 2

    repeat = run_paper_daily(tmp_path, payload, second_bar, 102.0, config, "abc123")
    assert repeat["reconciliation"]["simulation_days"] == 2
    reconciliations = pd.read_csv(tmp_path / "outputs" / "daily_reconciliation.csv")
    assert len(reconciliations) == 2

def test_paper_kill_switch_disables_new_plan(tmp_path):
    (tmp_path / "STOP_TRADING").write_text("", encoding="utf-8")
    payload = {"china_signal_date": "2026-07-21", "target_position": 0.8}
    bar = pd.Series(
        {"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1_000_000}
    )
    summary = run_paper_daily(
        tmp_path,
        payload,
        bar,
        99.0,
        _backtest_config(),
        "abc123",
    )
    assert summary["next_order_plan"]["status"] == "disabled"
    assert summary["reconciliation"]["kill_switch"] is True
    assert summary["account"]["shares"] == 0
