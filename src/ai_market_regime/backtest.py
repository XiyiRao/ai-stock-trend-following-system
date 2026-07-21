from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import SystemConfig


@dataclass
class PortfolioState:
    cash: float
    shares: int = 0
    last_buy_date: pd.Timestamp | None = None
    highest_close_since_entry: float | None = None
    average_cost: float = 0.0


def limit_pct_for_date(trade_date: pd.Timestamp) -> float:
    """Approximate 300308 daily limit: 10% before 2020-08-24, then 20%."""

    return 0.20 if trade_date >= pd.Timestamp("2020-08-24") else 0.10


def _is_one_price_bar(row: pd.Series) -> bool:
    if not np.isfinite(row.get("open", np.nan)):
        return False
    return abs(float(row["high"]) - float(row["low"])) / max(float(row["open"]), 1e-12) < 0.001


def _is_buy_blocked(row: pd.Series, previous_close: float, tolerance: float) -> bool:
    theoretical_limit = previous_close * (1 + limit_pct_for_date(pd.Timestamp(row.name)))
    return bool(float(row["open"]) >= theoretical_limit * (1 - tolerance) and _is_one_price_bar(row))


def _is_sell_blocked(row: pd.Series, previous_close: float, tolerance: float) -> bool:
    theoretical_limit = previous_close * (1 - limit_pct_for_date(pd.Timestamp(row.name)))
    return bool(float(row["open"]) <= theoretical_limit * (1 + tolerance) and _is_one_price_bar(row))


def trade_fees(
    trade_value: float,
    is_sell: bool,
    config: SystemConfig,
    trade_date: pd.Timestamp | None = None,
) -> float:
    commission = max(abs(trade_value) * config.commission_rate, config.minimum_commission)
    stamp_rate = config.sell_stamp_duty_rate
    if trade_date is not None and pd.Timestamp(trade_date) < pd.Timestamp("2023-08-28"):
        stamp_rate = 0.001
    stamp = abs(trade_value) * stamp_rate if is_sell else 0.0
    return commission + stamp


def _bar_is_tradable(row: pd.Series) -> bool:
    prices = pd.to_numeric(row[["open", "high", "low", "close"]], errors="coerce")
    return bool(prices.notna().all() and (prices > 0).all() and float(row.get("volume", 0)) > 0)


def _order_share_cap(equity: float, price: float, config: SystemConfig) -> int:
    allowed_value = min(config.max_order_value_cny, equity * config.max_daily_turnover_ratio)
    return int(allowed_value / max(price, 1e-12) // config.lot_size * config.lot_size)


def run_event_backtest(
    frame: pd.DataFrame,
    config: SystemConfig,
    position_column: str = "Final_Target_Position",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute T close signals at T+1 open with A-share trading constraints."""

    required = {"open", "high", "low", "close", "volume", position_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Backtest frame missing columns: {', '.join(missing)}")

    data = frame.sort_index().copy()
    state = PortfolioState(cash=float(config.initial_cash))
    records: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    previous_close: float | None = None

    for row_number, (trade_date, row) in enumerate(data.iterrows()):
        trade_date = pd.Timestamp(trade_date).normalize()
        signal_date = None if row_number == 0 else pd.Timestamp(data.index[row_number - 1]).normalize()
        signal_position = 0.0 if row_number == 0 else float(
            np.nan_to_num(data.iloc[row_number - 1][position_column], nan=0.0)
        )
        signal_position = float(np.clip(signal_position, 0.0, 1.0))
        blocked_reason = ""
        stop_triggered = False
        stop_price = np.nan
        traded_value_today = 0.0
        fees_today = 0.0

        if not _bar_is_tradable(row):
            blocked_reason = "停牌或行情无效"
            mark_price = previous_close if previous_close is not None else 0.0
            equity = state.cash + state.shares * mark_price
            records.append(
                {
                    "date": trade_date,
                    "signal_date": signal_date,
                    "cash": state.cash,
                    "shares": state.shares,
                    "close": mark_price,
                    "equity": equity,
                    "position": 0.0 if equity <= 0 else state.shares * mark_price / equity,
                    "signal_position": signal_position,
                    "blocked_reason": blocked_reason,
                    "atr_stop_triggered": False,
                    "atr_stop_price": np.nan,
                    "traded_value": 0.0,
                    "fees": 0.0,
                    "ai_score": row.get("AI_SCORE", np.nan),
                    "stock_score": row.get("STOCK_SCORE", np.nan),
                }
            )
            continue

        open_price = float(row["open"])
        marked_equity = state.cash + state.shares * open_price

        if state.shares > 0 and row_number > 0:
            previous_row = data.iloc[row_number - 1]
            previous_atr = float(previous_row.get("ATR20", np.nan))
            if state.highest_close_since_entry is not None and np.isfinite(previous_atr):
                stop_price = state.highest_close_since_entry - config.atr_stop_multiple * previous_atr
                stop_triggered = open_price <= stop_price
                if stop_triggered:
                    signal_position = 0.0

        target_value = marked_equity * signal_position
        desired_shares = int(target_value / max(open_price, 1e-12) // config.lot_size * config.lot_size)
        delta_shares = desired_shares - state.shares
        share_cap = _order_share_cap(marked_equity, open_price, config)

        if previous_close is not None and delta_shares > 0:
            if _is_buy_blocked(row, previous_close, config.limit_tolerance):
                blocked_reason = "一字涨停，买入受阻"
            else:
                execution_price = open_price * (1 + config.slippage_bps / 10_000.0)
                buy_shares = min(delta_shares, share_cap)
                while buy_shares >= config.lot_size:
                    trade_value = buy_shares * execution_price
                    fee = trade_fees(trade_value, False, config, trade_date)
                    if trade_value + fee <= state.cash:
                        break
                    buy_shares -= config.lot_size
                if buy_shares >= config.lot_size:
                    trade_value = buy_shares * execution_price
                    fee = trade_fees(trade_value, False, config, trade_date)
                    prior_cost = state.average_cost * state.shares
                    state.cash -= trade_value + fee
                    state.shares += buy_shares
                    state.average_cost = (prior_cost + trade_value + fee) / state.shares
                    state.last_buy_date = trade_date
                    state.highest_close_since_entry = max(
                        state.highest_close_since_entry or float(row["close"]),
                        float(row["close"]),
                    )
                    traded_value_today = trade_value
                    fees_today = fee
                    trades.append(
                        {
                            "signal_date": signal_date,
                            "execution_date": trade_date,
                            "side": "BUY",
                            "shares": buy_shares,
                            "price": execution_price,
                            "value": trade_value,
                            "fee": fee,
                            "realized_pnl": np.nan,
                            "reason": "target_rebalance",
                        }
                    )
                elif delta_shares >= config.lot_size:
                    blocked_reason = "现金或单笔限额不足"

        elif previous_close is not None and delta_shares < 0:
            t_plus_one_ok = state.last_buy_date is None or trade_date > state.last_buy_date
            if not t_plus_one_ok:
                blocked_reason = "T+1限制，今日买入不可卖出"
            elif _is_sell_blocked(row, previous_close, config.limit_tolerance):
                blocked_reason = "一字跌停，卖出受阻"
            else:
                execution_price = open_price * (1 - config.slippage_bps / 10_000.0)
                requested = min(-delta_shares, state.shares)
                sell_shares = min(requested, share_cap)
                if share_cap < config.lot_size:
                    blocked_reason = "单笔限额不足一手"
                if sell_shares > 0:
                    trade_value = sell_shares * execution_price
                    fee = trade_fees(trade_value, True, config, trade_date)
                    realized_pnl = trade_value - fee - sell_shares * state.average_cost
                    state.cash += trade_value - fee
                    state.shares -= sell_shares
                    traded_value_today = trade_value
                    fees_today = fee
                    trades.append(
                        {
                            "signal_date": signal_date,
                            "execution_date": trade_date,
                            "side": "SELL",
                            "shares": sell_shares,
                            "price": execution_price,
                            "value": trade_value,
                            "fee": fee,
                            "realized_pnl": realized_pnl,
                            "reason": "atr_stop" if stop_triggered else "target_rebalance",
                        }
                    )
                    if state.shares == 0:
                        state.highest_close_since_entry = None
                        state.last_buy_date = None
                        state.average_cost = 0.0

        close_price = float(row["close"])
        if state.shares > 0:
            state.highest_close_since_entry = max(
                state.highest_close_since_entry or close_price,
                close_price,
            )
        equity = state.cash + state.shares * close_price
        position = 0.0 if equity <= 0 else state.shares * close_price / equity
        records.append(
            {
                "date": trade_date,
                "signal_date": signal_date,
                "cash": state.cash,
                "shares": state.shares,
                "close": close_price,
                "equity": equity,
                "position": position,
                "signal_position": signal_position,
                "blocked_reason": blocked_reason,
                "atr_stop_triggered": stop_triggered,
                "atr_stop_price": stop_price,
                "traded_value": traded_value_today,
                "fees": fees_today,
                "ai_score": row.get("AI_SCORE", np.nan),
                "stock_score": row.get("STOCK_SCORE", np.nan),
            }
        )
        previous_close = close_price

    equity_curve = pd.DataFrame(records).set_index("date")
    trade_log = pd.DataFrame(trades)
    return equity_curve, trade_log


def performance_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame | None = None,
) -> dict[str, float]:
    equity = equity_curve["equity"].dropna()
    if len(equity) < 2:
        return {}
    daily_return = equity.pct_change(fill_method=None).fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    annual_volatility = daily_return.std(ddof=0) * np.sqrt(252)
    sharpe = 0.0 if annual_volatility == 0 else daily_return.mean() / daily_return.std(ddof=0) * np.sqrt(252)
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = 0.0 if max_drawdown == 0 else annual_return / abs(max_drawdown)
    average_equity = float(equity.mean())
    turnover = 0.0 if average_equity <= 0 else float(equity_curve["traded_value"].sum() / average_equity)
    metrics: dict[str, float] = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "turnover_ratio": turnover,
        "holding_ratio": float((equity_curve["shares"] > 0).mean()),
        "total_fees": float(equity_curve["fees"].sum()),
        "trade_count": float(0 if trade_log is None else len(trade_log)),
    }
    if trade_log is not None and not trade_log.empty and "realized_pnl" in trade_log:
        sells = trade_log.loc[trade_log["side"] == "SELL", "realized_pnl"].dropna()
        if not sells.empty:
            gains = float(sells[sells > 0].sum())
            losses = float(-sells[sells < 0].sum())
            metrics["win_rate"] = float((sells > 0).mean())
            metrics["profit_factor"] = gains / losses if losses > 0 else float("inf")
    return metrics


def cost_stressed_config(config: SystemConfig, multiplier: float) -> SystemConfig:
    return replace(
        config,
        commission_rate=config.commission_rate * multiplier,
        minimum_commission=config.minimum_commission * multiplier,
        sell_stamp_duty_rate=config.sell_stamp_duty_rate * multiplier,
        slippage_bps=config.slippage_bps * multiplier,
    )