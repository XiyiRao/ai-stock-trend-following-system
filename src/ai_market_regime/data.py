from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote
import warnings

import pandas as pd
import requests

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

    class _MissingYFinance:
        @staticmethod
        def download(*args, **kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    yf = _MissingYFinance()

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


def _download_chart_close(config: SystemConfig, start_date: str) -> pd.DataFrame:
    """Use Yahoo's public chart endpoint when yfinance's batch route is limited."""

    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    period2 = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).timestamp())
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 ai-market-regime/0.1"})
    series_by_ticker: dict[str, pd.Series] = {}
    errors: list[str] = []

    for ticker in config.all_tickers:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"
        try:
            response = session.get(
                url,
                params={
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "div,splits",
                    "includeAdjustedClose": "true",
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json().get("chart", {}).get("result")
            if not result:
                raise RuntimeError("empty chart result")
            chart = result[0]
            timestamps = chart.get("timestamp") or []
            indicators = chart.get("indicators", {})
            adjusted = indicators.get("adjclose") or []
            values = adjusted[0].get("adjclose") if adjusted else None
            if values is None:
                quotes = indicators.get("quote") or []
                values = quotes[0].get("close") if quotes else None
            if not timestamps or values is None or len(timestamps) != len(values):
                raise RuntimeError("incomplete chart data")
            index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
            series_by_ticker[ticker] = pd.Series(values, index=index, name=ticker, dtype=float)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    if errors:
        warnings.warn("Yahoo chart fallback errors: " + "; ".join(errors), RuntimeWarning, stacklevel=2)
    if not series_by_ticker:
        return pd.DataFrame(columns=list(config.all_tickers), dtype=float)
    return pd.concat(series_by_ticker.values(), axis=1).reindex(columns=list(config.all_tickers))


def download_close_prices(config: SystemConfig, cache_path: Path | None = None) -> pd.DataFrame:
    """Download adjusted closes, falling back to chart API and local cache."""

    cached = pd.DataFrame(columns=list(config.all_tickers), dtype=float)
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        cached = cached.reindex(columns=list(config.all_tickers)).sort_index()

    download_start = config.start_date
    cache_is_complete = (not cached.empty) and all(
        cached[ticker].notna().sum() > 0 for ticker in config.all_tickers
    )
    if cache_is_complete:
        download_start = (cached.index.max() - pd.Timedelta(days=90)).date().isoformat()
    if cache_is_complete and not YFINANCE_AVAILABLE:
        warnings.warn(
            "yfinance is not installed; continuing with the local cache.",
            RuntimeWarning,
            stacklevel=2,
        )
        growth_observed = cached[config.growth_ticker].notna()
        return cached.loc[growth_observed]

    end = (date.today() + timedelta(days=1)).isoformat()
    downloaded = pd.DataFrame()
    last_error: Exception | None = None
    try:
        downloaded = yf.download(
            list(config.all_tickers),
            start=download_start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
            timeout=30,
        )
    except Exception as exc:  # yfinance exposes several transport exceptions
        last_error = exc

    if downloaded.empty:
        fresh = _download_chart_close(config, download_start)
    else:
        fresh = _extract_close(downloaded, config.all_tickers)
        if any(fresh[ticker].notna().sum() == 0 for ticker in config.all_tickers):
            fresh = fresh.combine_first(_download_chart_close(config, download_start))

    if fresh.empty and cached.empty:
        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(
            "Yahoo Finance returned no data and no cache is available; it may be rate limited."
            + detail
        )

    if fresh.empty:
        warnings.warn(
            "Yahoo Finance returned no data; continuing with the local cache.",
            RuntimeWarning,
            stacklevel=2,
        )
        close = cached.copy()
    else:
        close = fresh.combine_first(cached)

    close = close.sort_index()
    # Retain actual QQQ sessions; do not fill missing US observations.
    growth_observed = close[config.growth_ticker].notna()
    close = close.loc[growth_observed]
    missing_entirely = [ticker for ticker in config.all_tickers if close[ticker].notna().sum() == 0]
    if missing_entirely:
        raise RuntimeError(f"No observations returned for: {', '.join(missing_entirely)}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(cache_path, index_label="Date")
    return close
