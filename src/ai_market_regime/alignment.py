from __future__ import annotations

import numpy as np
import pandas as pd


def align_us_scores_to_china_dates(
    us_scores: pd.DataFrame,
    china_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each China session to the latest strictly earlier US observation."""

    left = pd.DataFrame({"Date": pd.DatetimeIndex(china_dates).tz_localize(None).normalize()})
    left = left.drop_duplicates().sort_values("Date")
    right = us_scores.copy().sort_index()
    right.index = pd.to_datetime(right.index).tz_localize(None).normalize()
    right.index.name = "US_Source_Date"
    right = right.reset_index().drop_duplicates("US_Source_Date", keep="last")
    aligned = pd.merge_asof(
        left,
        right.sort_values("US_Source_Date"),
        left_on="Date",
        right_on="US_Source_Date",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("Date")
    aligned.index.name = "Date"

    valid = aligned["US_Source_Date"].notna()
    if not (aligned.loc[valid, "US_Source_Date"] < aligned.index[valid]).all():
        raise RuntimeError("Cross-market alignment failed: a US date is not strictly earlier.")
    return aligned


def build_alignment_audit(aligned: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
    """Create deterministic samples for manual future-leakage review."""

    valid = aligned.dropna(subset=["US_Source_Date"])
    if valid.empty:
        return pd.DataFrame(columns=["CN_Date", "US_Source_Date", "Strictly_Earlier", "Lag_Days"])
    count = min(sample_size, len(valid))
    positions = np.linspace(0, len(valid) - 1, count, dtype=int)
    sampled = valid.iloc[positions].reset_index()[["Date", "US_Source_Date"]]
    sampled = sampled.rename(columns={"Date": "CN_Date"})
    sampled["Strictly_Earlier"] = sampled["US_Source_Date"] < sampled["CN_Date"]
    sampled["Lag_Days"] = (sampled["CN_Date"] - sampled["US_Source_Date"]).dt.days
    return sampled
