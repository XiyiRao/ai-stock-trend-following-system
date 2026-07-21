from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import SystemConfig
from .data import download_close_prices
from .scoring import build_backtest, build_market_scores


def _latest_payload(scores: pd.DataFrame) -> dict[str, object]:
    valid = scores.dropna(subset=["AI_SCORE", "Target_Position"])
    if valid.empty:
        raise RuntimeError("Not enough history to calculate a market score.")
    date_value = valid.index[-1]
    row = valid.iloc[-1]
    return {
        "date": date_value.strftime("%Y-%m-%d"),
        "ai_trend_score": round(float(row["AI_Trend_Score"]), 2),
        "semiconductor_score": round(float(row["Semiconductor_Score"]), 2),
        "liquidity_score": round(float(row["Liquidity_Score"]), 2),
        "ai_score": round(float(row["AI_SCORE"]), 2),
        "market_regime": str(row["Market_Regime"]),
        "target_position": round(float(row["Target_Position"]), 2),
    }


def _plot_dashboard(scores: pd.DataFrame, backtest: pd.DataFrame, output_path: Path) -> None:
    valid = scores.dropna(subset=["AI_SCORE"])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(backtest.index, backtest["Target_Close"], color="#2563eb", linewidth=1.2)
    axes[0].set_ylabel("300308.SZ")
    axes[0].set_title("AI Market Regime and Target Position")
    axes[0].grid(alpha=0.2)
    axes[1].plot(valid.index, valid["AI_SCORE"], color="#7c3aed", label="AI_SCORE")
    axes[1].axhspan(70, 100, color="#16a34a", alpha=0.12)
    axes[1].axhspan(50, 70, color="#84cc16", alpha=0.10)
    axes[1].axhspan(30, 50, color="#f59e0b", alpha=0.10)
    axes[1].axhspan(0, 30, color="#dc2626", alpha=0.10)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.2)
    axes[2].step(valid.index, valid["Target_Position"] * 100, where="post", color="#ea580c")
    axes[2].set_ylim(-5, 105)
    axes[2].set_ylabel("Position %")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(root: Path, config: SystemConfig | None = None) -> dict[str, object]:
    config = config or SystemConfig()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    close = download_close_prices(config, root / "data" / "raw" / "close_prices.csv")
    scores = build_market_scores(close, config)
    backtest = build_backtest(scores, close[config.target_ticker], config.transaction_cost_bps)
    scores.to_csv(output_dir / "market_state.csv", index_label="Date", encoding="utf-8-sig")
    backtest.to_csv(output_dir / "backtest.csv", index_label="Date", encoding="utf-8-sig")
    payload = _latest_payload(scores)
    (output_dir / "latest_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_dashboard(scores, backtest, output_dir / "market_dashboard.png")
    return payload
