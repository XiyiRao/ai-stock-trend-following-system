import json
from pathlib import Path

import pandas as pd

from ai_market_regime.config import SystemConfig
from ai_market_regime.replay import save_historical_paper_replay


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    market_state_path = output_dir / "market_state.csv"
    if not market_state_path.exists():
        raise SystemExit("outputs/market_state.csv is missing; run run_daily.py first")

    frame = pd.read_csv(market_state_path, index_col="Date", parse_dates=True)
    report = save_historical_paper_replay(
        frame,
        output_dir,
        SystemConfig(),
        years=2,
    )

    quality_path = output_dir / "data_quality_report.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        china_source = quality.get("china", {})
        report["source"] = "Choice Excel derived market_state"
        report["source_file"] = china_source.get("source_file", "choice_300308.xlsx")
        report["source_sha256"] = china_source.get("source_sha256")
        report["source_start_date"] = china_source.get("start_date")
        report["source_end_date"] = china_source.get("end_date")
        (output_dir / "paper_replay_2y.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
