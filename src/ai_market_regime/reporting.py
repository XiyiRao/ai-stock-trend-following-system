from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import performance_metrics


def save_backtest_outputs(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = performance_metrics(equity_curve, trade_log)
    equity_curve.to_csv(output_dir / "equity_curve.csv", encoding="utf-8-sig")
    trade_log.to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    pd.Series(metrics, name="value").to_csv(output_dir / "metrics.csv", encoding="utf-8-sig")
    (output_dir / "backtest_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(equity_curve.index, equity_curve["equity"], color="#2563eb", linewidth=1.3)
    axis.set_title("300308 Full Strategy Equity Curve (Research Only)")
    axis.set_ylabel("Equity (CNY)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "equity_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    drawdown = equity_curve["equity"] / equity_curve["equity"].cummax() - 1
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.fill_between(drawdown.index, drawdown, 0, color="#dc2626", alpha=0.30)
    axis.plot(drawdown.index, drawdown, color="#dc2626", linewidth=1.0)
    axis.set_title("300308 Full Strategy Drawdown (Research Only)")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return metrics


def latest_signal_text(payload: dict[str, object]) -> str:
    return (
        f"日期: {payload['china_signal_date']}\n"
        f"AI产业状态分: {float(payload['ai_score']):.2f}/100\n"
        f"中际旭创个股分: {float(payload['stock_score']):.2f}/100\n"
        f"市场仓位上限: {float(payload['market_position_cap']):.0%}\n"
        f"最终目标仓位: {float(payload['target_position']):.0%}\n"
        f"操作结论: {payload['position_conclusion']}\n"
        f"风控状态: {payload['risk_rule']}\n"
        f"收盘价: {float(payload['close']):.2f}\n"
        f"MA20/MA60/MA120: {float(payload['ma20']):.2f} / "
        f"{float(payload['ma60']):.2f} / {float(payload['ma120']):.2f}\n"
        f"60日回撤: {float(payload['drawdown60']):.2%}\n"
        "说明: 研究信号，不构成投资建议，不可直接连接真实账户。\n"
    )