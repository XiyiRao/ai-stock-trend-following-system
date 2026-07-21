from pathlib import Path
import json

from ai_market_regime.pipeline import run_pipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    payload = run_pipeline(root, include_research=True)
    print(json.dumps(payload.get("backtest_metrics", {}), ensure_ascii=False, indent=2))
    print(f"research_mechanical_gates_passed: {payload.get('research_mechanical_gates_passed', False)}")