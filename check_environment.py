from pathlib import Path
import json

from ai_market_regime.environment import check_environment


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(check_environment(root), ensure_ascii=False, indent=2))