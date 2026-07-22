from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_event_backtest
from .config import SystemConfig


def run_historical_paper_replay(
    frame: pd.DataFrame,
    config: SystemConfig,
    years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replay the most recent calendar-year window with next-open execution."""

    if years < 1:
        raise ValueError("years must be at least 1")
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Final_Target_Position",
        "US_Source_Date",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Replay frame missing columns: {', '.join(missing)}")

    data = frame.sort_index().copy()
    data.index = pd.to_datetime(data.index).normalize()
    data["US_Source_Date"] = pd.to_datetime(data["US_Source_Date"], errors="coerce")
    valid = data.dropna(subset=list(required))
    if len(valid) < 2:
        raise ValueError("Replay needs at least two valid rows")

    execution_end = valid.index[-1]
    boundary = execution_end - pd.DateOffset(years=years)
    execution_rows = valid.loc[valid.index >= boundary]
    if execution_rows.empty:
        raise ValueError("Replay window contains no execution sessions")

    first_execution_position = valid.index.get_loc(execution_rows.index[0])
    if first_execution_position == 0:
        raise ValueError("Replay needs one signal row before the execution window")
    window = valid.iloc[first_execution_position - 1 :]

    if not (window["US_Source_Date"] < window.index.to_series()).all():
        raise ValueError("Replay rejected because US source dates are not strictly earlier")

    full_curve, trades = run_event_backtest(window, config)
    daily = full_curve.iloc[1:].copy()
    if len(daily) != len(execution_rows):
        raise RuntimeError("Replay execution-session count is inconsistent")

    initial_equity = float(config.initial_cash)
    equity_with_anchor = pd.concat(
        [pd.Series([initial_equity]), daily["equity"].reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity_with_anchor / equity_with_anchor.cummax() - 1.0
    final_equity = float(daily["equity"].iloc[-1])
    filled = trades.loc[trades["execution_date"].isin(daily.index)] if not trades.empty else trades
    empty_side = pd.Series(dtype=str)
    report: dict[str, Any] = {
        "mode": "historical_two_year_paper_replay",
        "research_only": True,
        "historical_replay_is_not_forward_paper_trading": True,
        "requested_calendar_years": years,
        "calendar_boundary_date": boundary.strftime("%Y-%m-%d"),
        "completed_execution_sessions": int(len(daily)),
        "signal_start_date": window.index[0].strftime("%Y-%m-%d"),
        "execution_start_date": daily.index[0].strftime("%Y-%m-%d"),
        "execution_end_date": daily.index[-1].strftime("%Y-%m-%d"),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": final_equity / initial_equity - 1.0,
        "max_drawdown": float(drawdown.min()),
        "trade_count": int(len(filled)),
        "buy_count": int((filled.get("side", empty_side) == "BUY").sum()),
        "sell_count": int((filled.get("side", empty_side) == "SELL").sum()),
        "total_fees": float(daily["fees"].sum()),
        "ending_cash": float(daily["cash"].iloc[-1]),
        "ending_shares": int(daily["shares"].iloc[-1]),
        "ending_position": float(daily["position"].iloc[-1]),
        "next_session_target_position": float(window.iloc[-1]["Final_Target_Position"]),
        "next_session_rebalance_pending": bool(
            abs(
                float(window.iloc[-1]["Final_Target_Position"])
                - float(daily["position"].iloc[-1])
            )
            > 1e-6
        ),
        "strict_us_cn_date_alignment": True,
        "execution_rule": "T close signal -> next China session open",
    }
    return daily, filled.reset_index(drop=True), report


def save_historical_paper_replay(
    frame: pd.DataFrame,
    output_dir: Path,
    config: SystemConfig,
    years: int = 2,
) -> dict[str, Any]:
    daily, trades, report = run_historical_paper_replay(frame, config, years)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        output_dir / "paper_replay_2y_daily.csv",
        index_label="execution_date",
        encoding="utf-8-sig",
    )
    trades.to_csv(
        output_dir / "paper_replay_2y_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "paper_replay_2y.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
