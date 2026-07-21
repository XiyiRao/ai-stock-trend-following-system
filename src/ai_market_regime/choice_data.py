from __future__ import annotations

from pathlib import Path

import pandas as pd

from .china_data import DataQualityError, file_sha256, validate_china_ohlcv


CHOICE_COLUMN_MAP = {
    "交易时间": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def standardize_choice_frame(raw: pd.DataFrame, expected_symbol: str) -> pd.DataFrame:
    """Normalize a Choice K-line export and discard footer/blank rows."""

    frame = raw.rename(columns=lambda value: str(value).strip()).copy()
    required = {"证券代码", *CHOICE_COLUMN_MAP}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataQualityError(f"Choice export missing fields: {', '.join(missing)}")

    frame = frame.rename(columns=CHOICE_COLUMN_MAP)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["date"].notna()].copy()
    symbols = (
        frame["证券代码"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    unexpected = sorted(set(symbols[symbols != expected_symbol]))
    if unexpected:
        raise DataQualityError(
            f"Choice export contains unexpected symbols: {', '.join(unexpected)}"
        )

    result = frame.loc[:, ["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.drop_duplicates("date", keep="last").set_index("date").sort_index()
    result.index.name = "Date"
    validate_china_ohlcv(result)
    return result


def load_choice_excel(
    path: Path,
    expected_symbol: str,
    standardized_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a local Choice Excel export as the preferred A-share source."""

    raw = pd.read_excel(path)
    frame = standardize_choice_frame(raw, expected_symbol)
    report = validate_china_ohlcv(frame)
    report.update(
        {
            "source": "choice_excel",
            "source_file": path.name,
            "source_sha256": file_sha256(path),
            "fallback_errors": [],
        }
    )
    if standardized_path is not None:
        standardized_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(standardized_path, index_label="Date", encoding="utf-8-sig")
    return frame, report
