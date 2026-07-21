from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import warnings

import pandas as pd

from .config import SystemConfig


CHINA_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


class DataQualityError(RuntimeError):
    """Raised when market data fails a hard quality gate."""


def standardize_china_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize AKShare A-share daily bars to the project schema."""

    renamed = raw.rename(columns=AKSHARE_COLUMN_MAP).copy()
    required = {"date", *CHINA_COLUMNS}
    missing = sorted(required.difference(renamed.columns))
    if missing:
        raise DataQualityError(f"China data missing fields: {', '.join(missing)}")
    result = renamed.loc[:, ["date", *CHINA_COLUMNS]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for column in CHINA_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    result = result.set_index("date").sort_index()
    result.index.name = "Date"
    return result


def validate_china_ohlcv(frame: pd.DataFrame) -> dict[str, object]:
    """Apply the guide's hard OHLCV quality gates."""

    issues: list[str] = []
    if frame.empty:
        issues.append("empty dataset")
    if frame.index.has_duplicates:
        issues.append("duplicate dates")
    if not frame.index.is_monotonic_increasing:
        issues.append("dates are not sorted")
    missing_columns = sorted(set(CHINA_COLUMNS).difference(frame.columns))
    if missing_columns:
        issues.append("missing columns: " + ", ".join(missing_columns))
    if not missing_columns and not frame.empty:
        prices = frame[["open", "high", "low", "close"]]
        if prices.isna().any().any():
            issues.append("missing OHLC values")
        if (prices <= 0).any().any():
            issues.append("non-positive prices")
        if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
            issues.append("high below open/close")
        if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
            issues.append("low above open/close")
        if frame[["volume", "amount"]].isna().any().any():
            issues.append("missing volume/amount")
        if (frame[["volume", "amount"]] < 0).any().any():
            issues.append("negative volume/amount")
    report = {
        "passed": not issues,
        "rows": int(len(frame)),
        "start_date": frame.index.min().strftime("%Y-%m-%d") if not frame.empty else None,
        "end_date": frame.index.max().strftime("%Y-%m-%d") if not frame.empty else None,
        "issues": issues,
    }
    if issues:
        raise DataQualityError("; ".join(issues))
    return report


def _load_standardized_cache(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=list(CHINA_COLUMNS), dtype=float)
    cached = pd.read_csv(path, index_col=0, parse_dates=True)
    cached.index = pd.to_datetime(cached.index).tz_localize(None).normalize()
    cached.index.name = "Date"
    return cached.reindex(columns=list(CHINA_COLUMNS)).sort_index()


def _sina_symbol(symbol: str) -> str:
    return ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol


def download_china_ohlcv(
    config: SystemConfig,
    raw_path: Path | None = None,
    standardized_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Download qfq-adjusted A-share bars with Eastmoney/Sina/cache fallback."""

    cached = _load_standardized_cache(standardized_path)
    fresh = pd.DataFrame()
    source = "cache"
    errors: list[str] = []

    try:
        import akshare as ak
    except Exception as exc:
        if cached.empty:
            raise RuntimeError(f"AKShare import failed: {exc}") from exc
        ak = None
        errors.append(f"akshare import: {exc}")

    if ak is not None:
        downloaders = (
            (
                "eastmoney",
                lambda: ak.stock_zh_a_hist(
                    symbol=config.target_symbol,
                    period="daily",
                    start_date=pd.Timestamp(config.start_date).strftime("%Y%m%d"),
                    end_date=date.today().strftime("%Y%m%d"),
                    adjust="qfq",
                ),
            ),
            (
                "sina",
                lambda: ak.stock_zh_a_daily(
                    symbol=_sina_symbol(config.target_symbol),
                    start_date=pd.Timestamp(config.start_date).strftime("%Y%m%d"),
                    end_date=date.today().strftime("%Y%m%d"),
                    adjust="qfq",
                ),
            ),
        )
        for name, downloader in downloaders:
            try:
                raw = downloader()
                if raw is not None and not raw.empty:
                    fresh = standardize_china_ohlcv(raw)
                    source = name
                    if raw_path is not None:
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
                    break
                errors.append(f"{name}: empty response")
            except Exception as exc:
                errors.append(f"{name}: {exc}")

    if fresh.empty:
        if cached.empty:
            raise RuntimeError("All China data sources failed: " + "; ".join(errors))
        warnings.warn(
            "China data sources failed; continuing with local cache: " + "; ".join(errors),
            RuntimeWarning,
            stacklevel=2,
        )
        frame = cached
    else:
        # A full refresh avoids mixing qfq adjustment bases across corporate actions.
        frame = fresh

    frame = frame.sort_index()
    report = validate_china_ohlcv(frame)
    report["source"] = source
    report["fallback_errors"] = errors
    if standardized_path is not None:
        standardized_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(standardized_path, index_label="Date", encoding="utf-8-sig")
    return frame, report


def download_china_calendar(cache_path: Path | None = None) -> pd.DatetimeIndex:
    """Download the China open-session calendar, with a cached fallback."""

    cached = pd.DatetimeIndex([])
    if cache_path is not None and cache_path.exists():
        cached_frame = pd.read_csv(cache_path, parse_dates=["trade_date"])
        cached = pd.DatetimeIndex(cached_frame["trade_date"]).normalize()
    try:
        import akshare as ak

        raw = ak.tool_trade_date_hist_sina()
        column = "trade_date" if "trade_date" in raw.columns else raw.columns[0]
        calendar = pd.DatetimeIndex(pd.to_datetime(raw[column], errors="coerce").dropna()).normalize()
        calendar = calendar.sort_values().unique()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"trade_date": calendar}).to_csv(cache_path, index=False)
        return pd.DatetimeIndex(calendar)
    except Exception as exc:
        if len(cached) == 0:
            raise RuntimeError(f"China trade-calendar download failed: {exc}") from exc
        warnings.warn(f"Calendar download failed; using cache: {exc}", RuntimeWarning, stacklevel=2)
        return cached


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
