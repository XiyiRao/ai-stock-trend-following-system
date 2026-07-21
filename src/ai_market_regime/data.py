from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

from .config import SystemConfig


def _extract_close(downloaded: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise RuntimeError("Downloaded data does not contain a Close field.")
        close = downloaded["Close"].copy()
    else:
        if "Close" not in downloaded.columns:
            raise RuntimeError("Downloaded data does not contain a Close field.")
        close = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().reindex(columns=list(tickers))
    return close.apply(pd.to_numeric, errors="coerce")


def download_close_prices(config: SystemConfig, cache_path: Path | None = None) -> pd.DataFrame:
    """Download adjusted close prices and optionally update a local CSV cache."""

    end = (date.today() + timedelta(days=1)).isoformat()
    downloaded = pd.DataFrame()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            downloaded = yf.download(
                list(config.all_tickers),
                start=config.start_date,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
                timeout=30,
            )
            if not downloaded.empty:
                break
        except Exception as exc:  # yfinance exposes several transport exceptions
            last_error = exc
        if attempt < 2:
            time.sleep(2**attempt)
    if downloaded.empty:
        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(
            "Yahoo Finance returned no data after 3 attempts; it may be rate limited."
            + detail
        )
    close = _extract_close(downloaded, config.all_tickers).ffill(limit=5)
    missing_entirely = [ticker for ticker in config.all_tickers if close[ticker].notna().sum() == 0]
    if missing_entirely:
        raise RuntimeError(f"No observations returned for: {', '.join(missing_entirely)}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            close = pd.concat([cached, close]).sort_index()
            close = close[~close.index.duplicated(keep="last")]
            close = close.reindex(columns=list(config.all_tickers))
        close.to_csv(cache_path, index_label="Date")
    return close
