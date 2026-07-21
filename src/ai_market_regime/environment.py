from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
import sys


REQUIRED_PACKAGES = (
    "pandas",
    "numpy",
    "matplotlib",
    "openpyxl",
    "requests",
    "yfinance",
    "akshare",
    "pytest",
)


def check_environment(root: Path) -> dict[str, object]:
    packages: dict[str, dict[str, object]] = {}
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            packages[name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:
            packages[name] = {"available": False, "error": str(exc)}
    report = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "all_packages_available": all(item["available"] for item in packages.values()),
        "choice_excel_present": (root / "data" / "input" / "choice_300308.xlsx").exists(),
        "stop_trading_present": (root / "STOP_TRADING").exists(),
        "live_account_configured": False,
        "note": "Live account credentials are intentionally not stored in this public repository.",
    }
    output = root / "outputs" / "environment_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report