from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import limit_pct_for_date, trade_fees
from .config import SystemConfig


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default.copy()
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def _append_log(path: Path, level: str, event: str, details: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "level": level,
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _kill_switch_active(root: Path) -> bool:
    env_value = os.getenv("KILL_SWITCH", "").strip().lower()
    return (root / "STOP_TRADING").exists() or env_value in {"1", "true", "yes", "on"}


def _default_state(config: SystemConfig) -> dict[str, Any]:
    return {
        "cash": config.initial_cash,
        "shares": 0,
        "average_cost": 0.0,
        "last_price": 0.0,
        "last_buy_date": None,
        "last_executed_plan_id": None,
        "processed_signal_dates": [],
        "consecutive_failures": 0,
    }


def _paper_execution(
    state: dict[str, Any],
    plan: dict[str, Any],
    trade_date: pd.Timestamp,
    bar: pd.Series,
    previous_close: float | None,
    config: SystemConfig,
) -> dict[str, Any]:
    open_price = float(bar["open"])
    target = float(np.clip(plan["target_position"], 0.0, config.max_target_position))
    equity = float(state["cash"]) + int(state["shares"]) * open_price
    target_shares = int(equity * target / open_price // config.lot_size * config.lot_size)
    delta = target_shares - int(state["shares"])
    allowed_value = min(config.max_order_value_cny, equity * config.max_daily_turnover_ratio)
    share_cap = int(allowed_value / open_price // config.lot_size * config.lot_size)
    result: dict[str, Any] = {
        "plan_id": plan["plan_id"],
        "signal_date": plan["signal_date"],
        "execution_date": trade_date.strftime("%Y-%m-%d"),
        "side": "NONE",
        "shares": 0,
        "price": open_price,
        "value": 0.0,
        "fee": 0.0,
        "status": "no_trade",
        "message": "目标仓位无需调整",
    }
    if delta == 0:
        state["last_executed_plan_id"] = plan["plan_id"]
        return result
    if float(bar.get("volume", 0)) <= 0:
        result.update(status="blocked", message="停牌或成交量无效")
        return result

    one_price = abs(float(bar["high"]) - float(bar["low"])) / max(open_price, 1e-12) < 0.001
    limit_pct = limit_pct_for_date(trade_date)
    if delta > 0:
        if previous_close and open_price >= previous_close * (1 + limit_pct) * (1 - config.limit_tolerance) and one_price:
            result.update(status="blocked", message="一字涨停，模拟买入受阻")
            return result
        execution_price = open_price * (1 + config.slippage_bps / 10_000.0)
        shares = min(delta, share_cap)
        while shares >= config.lot_size:
            value = shares * execution_price
            fee = trade_fees(value, False, config, trade_date)
            if value + fee <= float(state["cash"]):
                break
            shares -= config.lot_size
        if shares < config.lot_size:
            result.update(status="blocked", message="现金或单笔限额不足")
            return result
        value = shares * execution_price
        fee = trade_fees(value, False, config, trade_date)
        prior_cost = float(state["average_cost"]) * int(state["shares"])
        state["cash"] = float(state["cash"]) - value - fee
        state["shares"] = int(state["shares"]) + shares
        state["average_cost"] = (prior_cost + value + fee) / int(state["shares"])
        state["last_buy_date"] = trade_date.strftime("%Y-%m-%d")
        result.update(
            side="BUY",
            shares=shares,
            price=execution_price,
            value=value,
            fee=fee,
            status="filled",
            message="模拟买入完成",
        )
    else:
        last_buy = pd.Timestamp(state["last_buy_date"]) if state.get("last_buy_date") else None
        if last_buy is not None and trade_date <= last_buy:
            result.update(status="blocked", message="T+1限制，今日买入不可卖出")
            return result
        if previous_close and open_price <= previous_close * (1 - limit_pct) * (1 + config.limit_tolerance) and one_price:
            result.update(status="blocked", message="一字跌停，模拟卖出受阻")
            return result
        execution_price = open_price * (1 - config.slippage_bps / 10_000.0)
        if share_cap < config.lot_size:
            result.update(status="blocked", message="单笔限额不足一手")
            return result
        shares = min(-delta, int(state["shares"]), share_cap)
        if shares <= 0:
            result.update(status="blocked", message="无可卖股数")
            return result
        value = shares * execution_price
        fee = trade_fees(value, True, config, trade_date)
        state["cash"] = float(state["cash"]) + value - fee
        state["shares"] = int(state["shares"]) - shares
        if int(state["shares"]) == 0:
            state["average_cost"] = 0.0
            state["last_buy_date"] = None
        result.update(
            side="SELL",
            shares=shares,
            price=execution_price,
            value=value,
            fee=fee,
            status="filled",
            message="模拟卖出完成",
        )
    state["last_executed_plan_id"] = plan["plan_id"]
    return result


def run_paper_daily(
    root: Path,
    payload: dict[str, Any],
    latest_bar: pd.Series,
    previous_close: float | None,
    config: SystemConfig,
    git_hash: str,
) -> dict[str, Any]:
    """Run one idempotent after-close paper cycle and prepare the next order."""

    output_dir = root / "outputs"
    log_path = root / "logs" / "daily.log"
    lock_path = root / "work" / "paper_daily.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.open("x").close()
    except FileExistsError as exc:
        raise RuntimeError("Paper daily process is already running") from exc

    try:
        state_path = output_dir / "paper_account.json"
        plan_path = output_dir / "next_order_plan.json"
        state = _read_json(state_path, _default_state(config))
        signal_date = str(payload["china_signal_date"])
        trade_date = pd.Timestamp(signal_date).normalize()
        if latest_bar.name is not None:
            bar_date = pd.Timestamp(latest_bar.name).normalize()
            if bar_date != trade_date:
                raise RuntimeError(
                    f"Signal/bar date mismatch: signal={trade_date.date()}, bar={bar_date.date()}"
                )
        kill_switch = _kill_switch_active(root)
        prior_plan = _read_json(plan_path, {})
        execution: dict[str, Any] | None = None

        if (
            prior_plan
            and prior_plan.get("status") == "planned"
            and prior_plan.get("plan_id") != state.get("last_executed_plan_id")
            and pd.Timestamp(prior_plan["signal_date"]) < trade_date
        ):
            if kill_switch:
                execution = {
                    "plan_id": prior_plan["plan_id"],
                    "status": "disabled",
                    "message": "KILL_SWITCH/STOP_TRADING已启用，禁止模拟执行",
                }
            else:
                execution = _paper_execution(
                    state, prior_plan, trade_date, latest_bar, previous_close, config
                )
                _append_csv(output_dir / "paper_trades.csv", execution)
                _append_log(log_path, "INFO", "paper_execution", execution)

        processed = list(state.get("processed_signal_dates", []))
        if signal_date not in processed:
            processed.append(signal_date)
        state["processed_signal_dates"] = processed
        state["last_price"] = float(latest_bar["close"])
        equity = float(state["cash"]) + int(state["shares"]) * float(latest_bar["close"])
        target = float(np.clip(payload["target_position"], 0.0, config.max_target_position))
        target_shares = int(
            equity * target / float(latest_bar["close"]) // config.lot_size * config.lot_size
        )
        plan_id = hashlib.sha256(
            f"{signal_date}|{target:.4f}|{git_hash}".encode("utf-8")
        ).hexdigest()[:20]
        new_plan = {
            "plan_id": plan_id,
            "signal_date": signal_date,
            "execute_on": "next_china_session_open_after_09_35",
            "symbol": config.target_symbol,
            "target_position": target,
            "reference_close": float(latest_bar["close"]),
            "current_shares": int(state["shares"]),
            "estimated_target_shares": target_shares,
            "estimated_delta_shares": target_shares - int(state["shares"]),
            "max_order_value_cny": config.max_order_value_cny,
            "status": "disabled" if kill_switch else "planned",
            "git_hash": git_hash,
            "research_only": True,
        }
        _write_json(plan_path, new_plan)
        _write_json(state_path, state)

        reconciliation = {
            "date": signal_date,
            "target_position": target,
            "target_shares": target_shares,
            "actual_shares": int(state["shares"]),
            "share_difference": target_shares - int(state["shares"]),
            "cash": float(state["cash"]),
            "market_value": int(state["shares"]) * float(latest_bar["close"]),
            "equity": equity,
            "pending_plan_id": plan_id,
            "kill_switch": kill_switch,
            "simulation_days": len(processed),
            "git_hash": git_hash,
        }
        existing_recon = output_dir / "daily_reconciliation.csv"
        if not existing_recon.exists() or signal_date not in pd.read_csv(existing_recon)["date"].astype(str).tolist():
            _append_csv(existing_recon, reconciliation)
        summary = {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "signal_date": signal_date,
            "execution": execution,
            "next_order_plan": new_plan,
            "account": state,
            "reconciliation": reconciliation,
            "twenty_day_validation_complete": len(processed) >= 20,
        }
        _write_json(output_dir / "paper_daily_summary.json", summary)
        _append_log(log_path, "INFO", "daily_cycle_complete", reconciliation)
        return summary
    except Exception as exc:
        _append_log(log_path, "ERROR", "daily_cycle_failed", {"error": str(exc)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)