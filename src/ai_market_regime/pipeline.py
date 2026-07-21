from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .alignment import align_us_scores_to_china_dates, build_alignment_audit
from .china_data import (
    DataQualityError,
    download_china_calendar,
    download_china_ohlcv,
    file_sha256,
)
from .choice_data import load_choice_excel
from .config import SystemConfig
from .data import download_close_prices
from .backtest import run_event_backtest
from .reporting import latest_signal_text, save_backtest_outputs
from .research import run_research_suite
from .scoring import build_market_scores
from .stock_scoring import build_stock_scores, combine_market_and_stock_scores


def _latest_payload(scores: pd.DataFrame) -> dict[str, object]:
    required = ["AI_SCORE", "STOCK_SCORE", "Final_Target_Position", "US_Source_Date"]
    valid = scores.dropna(subset=required)
    if valid.empty:
        raise RuntimeError("Not enough history to calculate a combined market and stock score.")
    date_value = valid.index[-1]
    row = valid.iloc[-1]
    return {
        "china_signal_date": date_value.strftime("%Y-%m-%d"),
        "us_source_date": pd.Timestamp(row["US_Source_Date"]).strftime("%Y-%m-%d"),
        "close": round(float(row["close"]), 2),
        "ai_momentum_score": round(float(row["AI_Momentum_Score"]), 2),
        "semiconductor_score": round(float(row["Semiconductor_Score"]), 2),
        "growth_score": round(float(row["Growth_Score"]), 2),
        "rates_score": round(float(row["Rates_Score"]), 2),
        "ai_score": round(float(row["AI_SCORE"]), 2),
        "market_regime": str(row["Market_Regime"]),
        "regime_cap": round(float(row["Regime_Cap"]), 2),
        "market_position_cap": round(float(row["Market_Position_Cap"]), 2),
        "trend_score": round(float(row["Trend_Score"]), 2),
        "momentum_score": round(float(row["Momentum_Score"]), 2),
        "volume_price_score": round(float(row["Volume_Price_Score"]), 2),
        "breakout_score": round(float(row["Breakout_Score"]), 2),
        "risk_quality_score": round(float(row["Risk_Quality_Score"]), 2),
        "stock_score": round(float(row["STOCK_SCORE"]), 2),
        "trend_eligible": bool(row["Trend_Eligible"]),
        "ma20": round(float(row["MA20"]), 2),
        "ma60": round(float(row["MA60"]), 2),
        "ma120": round(float(row["MA120"]), 2),
        "drawdown60": round(float(row["Drawdown60"]), 4),
        "atr20": round(float(row["ATR20"]), 2),
        "atr_stop_reference": round(float(row["ATR_Stop_Reference"]), 2),
        "raw_target_position": round(float(row["Raw_Target_Position"]), 2),
        "risk_rule": str(row["Risk_Rule"]),
        "target_position": round(float(row["Final_Target_Position"]), 2),
        "position_conclusion": str(row["Position_Conclusion"]),
    }


def _plot_dashboard(scores: pd.DataFrame, backtest: pd.DataFrame, output_path: Path) -> None:
    valid = scores.dropna(subset=["AI_SCORE", "STOCK_SCORE"])
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    axes[0].plot(valid.index, valid["close"], color="#2563eb", linewidth=1.2, label="Close")
    axes[0].plot(valid.index, valid["MA60"], color="#f59e0b", linewidth=1.0, label="MA60")
    axes[0].plot(valid.index, valid["MA120"], color="#64748b", linewidth=1.0, label="MA120")
    axes[0].set_ylabel("300308")
    axes[0].set_title("AI Market Regime + 300308 Stock Confirmation")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(valid.index, valid["AI_SCORE"], color="#7c3aed", label="AI_SCORE")
    axes[1].axhline(45, color="#dc2626", alpha=0.45, linewidth=0.8)
    axes[1].axhline(60, color="#f59e0b", alpha=0.45, linewidth=0.8)
    axes[1].axhline(75, color="#16a34a", alpha=0.45, linewidth=0.8)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("AI score")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.2)

    axes[2].plot(valid.index, valid["STOCK_SCORE"], color="#0891b2", label="STOCK_SCORE")
    axes[2].axhline(55, color="#dc2626", alpha=0.5, linewidth=0.8, label="Entry threshold")
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Stock score")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.2)

    axes[3].step(
        valid.index,
        valid["Market_Position_Cap"] * 100,
        where="post",
        color="#94a3b8",
        label="Market cap",
    )
    axes[3].step(
        valid.index,
        valid["Final_Target_Position"] * 100,
        where="post",
        color="#ea580c",
        label="Final target",
    )
    axes[3].set_ylim(-5, 85)
    axes[3].set_ylabel("Position %")
    axes[3].legend(loc="upper left")
    axes[3].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _quality_report(
    us_close: pd.DataFrame,
    china_bars: pd.DataFrame,
    china_report: dict[str, object],
    calendar: pd.DatetimeIndex,
    aligned: pd.DataFrame,
    audit: pd.DataFrame,
    standardized_path: Path,
    us_path: Path,
    config: SystemConfig,
) -> dict[str, object]:
    today = pd.Timestamp.today().normalize()
    expected_sessions = calendar[(calendar >= china_bars.index.min()) & (calendar <= today)]
    expected_latest = expected_sessions.max() if len(expected_sessions) else china_bars.index.max()
    latest_is_current = china_bars.index.max() >= expected_latest
    missing_recent_sessions = expected_sessions[expected_sessions > china_bars.index.max()]
    session_lag = int(len(missing_recent_sessions))
    strict_alignment = bool(audit["Strictly_Earlier"].all()) if not audit.empty else False
    valid_ai_members_latest = int(
        us_close.loc[us_close.index.max(), list(config.ai_stocks)].notna().sum()
    )
    issues: list[str] = []
    report_warnings: list[str] = []
    if session_lag > 0:
        issues.append(f"China bars trail the calendar by {session_lag} open sessions")
    if not strict_alignment:
        issues.append("US/CN strict-date audit failed")
    if valid_ai_members_latest < 3:
        issues.append("fewer than three AI basket members are available")
    report = {
        "passed": not issues,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "china": china_report,
        "china_expected_latest_session": expected_latest.strftime("%Y-%m-%d"),
        "china_latest_is_current": latest_is_current,
        "china_open_session_lag": session_lag,
        "us_rows": int(len(us_close)),
        "us_start_date": us_close.index.min().strftime("%Y-%m-%d"),
        "us_end_date": us_close.index.max().strftime("%Y-%m-%d"),
        "valid_ai_members_latest": valid_ai_members_latest,
        "aligned_rows": int(len(aligned)),
        "strict_alignment_audit_passed": strict_alignment,
        "standardized_china_sha256": file_sha256(standardized_path),
        "us_adjusted_close_sha256": file_sha256(us_path),
        "warnings": report_warnings,
        "issues": issues,
    }
    if issues:
        raise DataQualityError("; ".join(issues))
    return report


def run_pipeline(
    root: Path,
    config: SystemConfig | None = None,
    include_research: bool = True,
) -> dict[str, object]:
    config = config or SystemConfig()
    output_dir = root / "outputs"
    raw_dir = root / "data" / "raw"
    standardized_dir = root / "data" / "standardized"
    output_dir.mkdir(parents=True, exist_ok=True)

    us_path = raw_dir / "us_adjusted_close.csv"
    us_close = download_close_prices(config, us_path)
    china_path = standardized_dir / "china_300308_qfq.csv"
    choice_path = root / "data" / "input" / "choice_300308.xlsx"
    if choice_path.exists():
        china_bars, china_report = load_choice_excel(
            choice_path,
            expected_symbol=config.target_symbol,
            standardized_path=china_path,
        )
    else:
        china_bars, china_report = download_china_ohlcv(
            config,
            raw_path=raw_dir / "china_300308_latest_raw.csv",
            standardized_path=china_path,
        )
    calendar = download_china_calendar(raw_dir / "china_trade_calendar.csv")

    us_scores = build_market_scores(us_close, config)
    aligned = align_us_scores_to_china_dates(us_scores, china_bars.index)
    stock_scores = build_stock_scores(china_bars, config)
    combined = combine_market_and_stock_scores(aligned, stock_scores, config)
    audit = build_alignment_audit(combined, sample_size=10)
    report = _quality_report(
        us_close,
        china_bars,
        china_report,
        calendar,
        combined,
        audit,
        china_path,
        us_path,
        config,
    )
    equity_curve, trade_log = run_event_backtest(combined, config)
    backtest_metrics = save_backtest_outputs(equity_curve, trade_log, output_dir)
    research_summary: dict[str, object] | None = None
    if include_research:
        research = run_research_suite(combined, config, output_dir)
        research_summary = research["summary"]

    us_scores.to_csv(output_dir / "us_market_scores.csv", index_label="US_Date", encoding="utf-8-sig")
    stock_scores.to_csv(output_dir / "stock_scores.csv", index_label="Date", encoding="utf-8-sig")
    combined.to_csv(output_dir / "market_state.csv", index_label="Date", encoding="utf-8-sig")
    audit.to_csv(output_dir / "time_alignment_audit.csv", index=False, encoding="utf-8-sig")
    equity_curve.to_csv(output_dir / "backtest.csv", index_label="Date", encoding="utf-8-sig")
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = _latest_payload(combined)
    payload["backtest_metrics"] = {
        key: round(float(value), 6) if isinstance(value, (int, float)) else value
        for key, value in backtest_metrics.items()
    }
    if research_summary is not None:
        payload["research_mechanical_gates_passed"] = bool(research_summary["all_gates_passed"])
    (output_dir / "latest_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "latest_signal.txt").write_text(latest_signal_text(payload), encoding="utf-8")
    _plot_dashboard(combined, equity_curve, output_dir / "market_dashboard.png")
    return payload
