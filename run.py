from pathlib import Path

from ai_market_regime.pipeline import run_pipeline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    result = run_pipeline(project_root)
    print("Latest AI market signal")
    for key, value in result.items():
        print(f"{key}: {value}")
