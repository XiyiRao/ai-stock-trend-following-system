from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import cost_stressed_config, performance_metrics, run_event_backtest
from .config import SystemConfig


def _filter_targets(targets: pd.Series, threshold: float) -> pd.Series:
    result: list[float] = []
    previous = 0.0
    for value in targets.fillna(0.0):
        target = float(value)
        if (
            (previous <= 0.0 < target)
            or (target <= 0.0 < previous)
            or abs(target - previous) >= threshold
        ):
            previous = target
        result.append(previous)
    return pd.Series(result, index=targets.index, dtype=float)


def positions_for_config(frame: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    """Rebuild final positions from already-computed, past-only features."""

    result = frame.copy()
    eligible = (
        (result["STOCK_SCORE"] >= config.stock_minimum_score)
        & (result["close"] > result["MA60"])
        & (result["MA20"] > result["MA60"])
    )
    observation_floor = result["Market_Position_Cap"].clip(
        upper=config.minimum_observation_position
    ).fillna(0.0)
    target = result["Market_Position_Cap"].where(
        eligible, observation_floor
    ).fillna(0.0)
    target.loc[result["Drawdown60"] <= -config.drawdown_reduce_at] *= config.drawdown_reduced_multiplier
    target = target.clip(lower=observation_floor)
    target.loc[
        (result["Drawdown60"] <= -config.drawdown_exit_at)
        | (result["close"] < result["MA120"])
    ] = 0.0
    target = target.clip(0.0, config.max_target_position)
    result["Final_Target_Position"] = _filter_targets(target, config.rebalance_threshold)
    return result


def _metrics_row(name: str, curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    return {"name": name, **performance_metrics(curve, trades)}


def benchmark_frames(frame: pd.DataFrame, config: SystemConfig) -> dict[str, pd.DataFrame]:
    full = positions_for_config(frame, config)
    buy_hold = frame.copy()
    buy_hold["Final_Target_Position"] = 1.0
    ma60 = frame.copy()
    ma60["Final_Target_Position"] = np.where(ma60["close"] > ma60["MA60"], config.max_target_position, 0.0)
    ai_only = frame.copy()
    ai_only["Final_Target_Position"] = ai_only["Market_Position_Cap"].fillna(0.0)
    return {
        "full_strategy": full,
        "buy_and_hold": buy_hold,
        "ma60_only": ma60,
        "ai_score_only": ai_only,
    }


def chronological_splits(index: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    count = len(index)
    train_end = int(count * 0.60)
    validate_end = int(count * 0.80)
    return {
        "train": index[:train_end],
        "validation": index[train_end:validate_end],
        "out_of_sample": index[validate_end:],
    }


def run_research_suite(
    frame: pd.DataFrame,
    config: SystemConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_rows: list[dict[str, object]] = []
    curves: dict[str, pd.DataFrame] = {}
    trades_by_name: dict[str, pd.DataFrame] = {}
    for name, benchmark in benchmark_frames(frame, config).items():
        curve, trades = run_event_backtest(benchmark, config)
        curves[name] = curve
        trades_by_name[name] = trades
        benchmark_rows.append(_metrics_row(name, curve, trades))
    benchmark_metrics = pd.DataFrame(benchmark_rows).set_index("name")

    split_rows: list[dict[str, object]] = []
    full = positions_for_config(frame, config)
    for name, dates in chronological_splits(full.index).items():
        subset = full.loc[dates]
        curve, trades = run_event_backtest(subset, config)
        split_rows.append(_metrics_row(name, curve, trades))
    split_metrics = pd.DataFrame(split_rows).set_index("name")

    cost_rows: list[dict[str, object]] = []
    for multiplier in (1.0, 1.5, 2.0):
        stressed = cost_stressed_config(config, multiplier)
        curve, trades = run_event_backtest(full, stressed)
        cost_rows.append(_metrics_row(f"cost_{multiplier:.1f}x", curve, trades))
    cost_stress = pd.DataFrame(cost_rows).set_index("name")

    parameter_cases: list[tuple[str, SystemConfig]] = []
    for value in (50.0, 55.0, 60.0):
        parameter_cases.append((f"stock_minimum={value:g}", replace(config, stock_minimum_score=value)))
    for value in (0.12, 0.15, 0.18):
        parameter_cases.append((f"drawdown_reduce={value:.2f}", replace(config, drawdown_reduce_at=value)))
    for value in (0.20, 0.22, 0.26):
        parameter_cases.append((f"drawdown_exit={value:.2f}", replace(config, drawdown_exit_at=value)))
    for value in (2.5, 3.0, 3.5):
        parameter_cases.append((f"atr_multiple={value:.1f}", replace(config, atr_stop_multiple=value)))
    for value in (0.50, 0.65, 0.80):
        parameter_cases.append((f"max_position={value:.2f}", replace(config, max_target_position=value)))

    parameter_rows: list[dict[str, object]] = []
    for name, variant in parameter_cases:
        variant_frame = positions_for_config(frame, variant)
        curve, trades = run_event_backtest(variant_frame, variant)
        parameter_rows.append(_metrics_row(name, curve, trades))
    parameter_stability = pd.DataFrame(parameter_rows).set_index("name")

    walk_rows: list[dict[str, object]] = []
    first_date = full.index.min()
    last_date = full.index.max()
    train_end = first_date + pd.DateOffset(years=3)
    while train_end < last_date:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(train_end + pd.DateOffset(years=1), last_date)
        subset = full.loc[(full.index >= test_start) & (full.index <= test_end)]
        if len(subset) >= 60:
            curve, trades = run_event_backtest(subset, config)
            walk_rows.append(
                {
                    "train_start": first_date.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": subset.index.min().strftime("%Y-%m-%d"),
                    "test_end": subset.index.max().strftime("%Y-%m-%d"),
                    **performance_metrics(curve, trades),
                }
            )
        train_end += pd.DateOffset(months=6)
    walk_forward = pd.DataFrame(walk_rows)

    oos = split_metrics.loc["out_of_sample"].to_dict()
    double_cost = cost_stress.loc["cost_2.0x"].to_dict()
    gates = {
        "out_of_sample_annual_return_positive": bool(oos.get("annual_return", -1) > 0),
        "out_of_sample_max_drawdown_within_30pct": bool(oos.get("max_drawdown", -1) >= -0.30),
        "out_of_sample_sharpe_near_one": bool(oos.get("sharpe", 0) >= 0.80),
        "out_of_sample_calmar_above_0_8": bool(oos.get("calmar", 0) >= 0.80),
        "double_cost_total_return_positive": bool(double_cost.get("total_return", -1) > 0),
    }
    positive_parameter_share = float((parameter_stability["total_return"] > 0).mean())
    positive_walk_forward_share = float(
        (walk_forward["total_return"] > 0).mean() if not walk_forward.empty else 0.0
    )
    gates["parameter_neighborhood_majority_positive"] = positive_parameter_share >= 0.70
    gates["walk_forward_majority_positive"] = positive_walk_forward_share >= 0.70
    gates["drawdown_lower_than_buy_and_hold"] = bool(
        benchmark_metrics.loc["full_strategy", "max_drawdown"]
        > benchmark_metrics.loc["buy_and_hold", "max_drawdown"]
    )
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "research_only": True,
        "gate_results": gates,
        "all_gates_passed": bool(all(gates.values())),
        "positive_parameter_share": positive_parameter_share,
        "positive_walk_forward_share": positive_walk_forward_share,
        "note": "Passing these mechanical gates is necessary but not sufficient for live trading.",
    }

    benchmark_metrics.to_csv(output_dir / "benchmark_metrics.csv", encoding="utf-8-sig")
    split_metrics.to_csv(output_dir / "sample_split_metrics.csv", encoding="utf-8-sig")
    cost_stress.to_csv(output_dir / "cost_stress.csv", encoding="utf-8-sig")
    parameter_stability.to_csv(output_dir / "parameter_stability.csv", encoding="utf-8-sig")
    walk_forward.to_csv(output_dir / "walk_forward.csv", index=False, encoding="utf-8-sig")
    (output_dir / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "summary": summary,
        "benchmark_metrics": benchmark_metrics,
        "split_metrics": split_metrics,
        "cost_stress": cost_stress,
        "parameter_stability": parameter_stability,
        "walk_forward": walk_forward,
        "full_curve": curves["full_strategy"],
        "full_trades": trades_by_name["full_strategy"],
    }