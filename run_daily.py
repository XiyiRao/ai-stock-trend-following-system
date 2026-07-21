from pathlib import Path
import json
import subprocess

import pandas as pd

from ai_market_regime.config import SystemConfig
from ai_market_regime.paper import run_paper_daily
from ai_market_regime.pipeline import run_pipeline


def _git_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = SystemConfig()
    payload = run_pipeline(root, config, include_research=False)
    bars = pd.read_csv(
        root / "data" / "standardized" / "china_300308_qfq.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()
    latest_bar = bars.iloc[-1]
    previous_close = float(bars.iloc[-2]["close"]) if len(bars) >= 2 else None
    summary = run_paper_daily(
        root,
        payload,
        latest_bar,
        previous_close,
        config,
        _git_hash(root),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))