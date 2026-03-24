"""Calendar alignment for US-JP trading day pairs.

Key concept:
- US market closes on day t
- JP market trades on day t+1 (next JP business day)
- Friday US -> Monday JP (typical case)
- We pair US close on t with JP open-to-close on the next JP trading day
"""

from __future__ import annotations

import pandas as pd


def align_us_jp_dates(
    us_dates: pd.DatetimeIndex,
    jp_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a mapping from US trading dates to JP next-day trading dates.

    For each US trading date t, find the next JP trading date >= t+1 calendar day.

    Returns:
        DataFrame with columns ['us_date', 'jp_date']
    """
    us_sorted = us_dates.sort_values()
    jp_sorted = jp_dates.sort_values()

    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    jp_idx = 0

    for us_date in us_sorted:
        # Find the first JP date strictly after the US date
        next_calendar = us_date + pd.Timedelta(days=1)
        while jp_idx < len(jp_sorted) and jp_sorted[jp_idx] < next_calendar:
            jp_idx += 1
        if jp_idx >= len(jp_sorted):
            break
        pairs.append((us_date, jp_sorted[jp_idx]))
        # Do NOT advance jp_idx here — multiple US dates could map to the same
        # JP date in rare cases (shouldn't happen normally, but be safe)

    df = pd.DataFrame(pairs, columns=["us_date", "jp_date"])
    # Remove duplicate JP dates (keep first US date mapping)
    df = df.drop_duplicates(subset="jp_date", keep="first")
    return df


def build_aligned_dataset(
    us_close: pd.DataFrame,
    jp_close: pd.DataFrame,
    jp_open: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned datasets where each row corresponds to a US-JP day pair.

    Returns:
        date_map: DataFrame with us_date, jp_date columns
        us_close_aligned: US close prices indexed by pair index
        jp_close_aligned: JP close prices indexed by pair index
        jp_open_aligned:  JP open prices indexed by pair index
    """
    us_dates = us_close.dropna(how="all").index
    jp_dates = jp_close.dropna(how="all").index

    date_map = align_us_jp_dates(us_dates, jp_dates)

    # Filter to dates that exist in both
    mask_us = date_map["us_date"].isin(us_close.index)
    mask_jp = date_map["jp_date"].isin(jp_close.index) & date_map["jp_date"].isin(
        jp_open.index
    )
    date_map = date_map[mask_us & mask_jp].reset_index(drop=True)

    us_close_aligned = us_close.loc[date_map["us_date"].values].reset_index(drop=True)
    jp_close_aligned = jp_close.loc[date_map["jp_date"].values].reset_index(drop=True)
    jp_open_aligned = jp_open.loc[date_map["jp_date"].values].reset_index(drop=True)

    return date_map, us_close_aligned, jp_close_aligned, jp_open_aligned
