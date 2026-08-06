"""
Google-Sheet loading, filtering, and contiguous range-splitting for the
AB automation.

Only this module talks to the sheet. The rest of the automation deals in
plain ``UserRange`` dataclasses so unit-testing does not require network.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

# [Reason] Reuse the existing helper without any auth changes; follows the same
# convention as other scripts in the project (see yt_download.py line 4).
_UTIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "util")
if _UTIL_DIR not in sys.path:
    sys.path.append(_UTIL_DIR)

from google_utils import read_spreadsheet  # noqa: E402  (path-append above)


# --- Column names in the AB sheet (fixed by the sheet schema) --------------
SR_COL: str = "Sr. no"
UPLOADED_COL: str = "Uploaded Pecha Tool"
# [Reason] Actual header in the AB sheet includes the unit suffix.
DURATION_COL: str = "Standard Duration (hh:mm:ss)"
GROUP_COL: str = "Group"

# [Reason] Only Group A rows are in scope; other groups belong to other pipelines.
GROUP_TARGET_LABEL: str = "Group A"
_GROUP_A_ACCEPTED: frozenset[str] = frozenset(
    {"a", "group a", "groupa", "group_a", "group-a"}
)


# --- Public dataclasses ----------------------------------------------------
@dataclass(frozen=True)
class SheetStats:
    """Row counts derived from the loaded sheet."""

    total: int
    already_uploaded: int
    remaining: int


@dataclass(frozen=True)
class UserRange:
    """A single contiguous ``Sr. no`` range that will be run through the pipeline.

    ``sub_index`` is > 1 only when a user's chunk had to be split because
    already-uploaded rows sat between two remaining rows (gap handling).
    """

    user_index: int
    sub_index: int
    from_id: int
    to_id: int
    rows: int
    duration: pd.Timedelta


class SheetError(Exception):
    """Raised when the sheet cannot be used (missing columns, empty, etc.)."""


# --- Loading & filtering ---------------------------------------------------
def load_ab_sheet(sheet_id: str) -> pd.DataFrame:
    """Fetch the default worksheet, validate schema, coerce ``Sr. no`` to int."""
    df = read_spreadsheet(sheet_id)
    if df is None or df.empty:
        raise SheetError("Google Sheet returned no rows.")

    missing = [
        c for c in (SR_COL, UPLOADED_COL, DURATION_COL, GROUP_COL) if c not in df.columns
    ]
    if missing:
        raise SheetError(
            "Sheet is missing required column(s): " + ", ".join(missing)
        )

    df = df.copy()
    # [Reason] gviz returns numbers as floats/strings; coerce and drop invalid rows.
    df[SR_COL] = pd.to_numeric(df[SR_COL], errors="coerce")
    df = df.dropna(subset=[SR_COL])
    df[SR_COL] = df[SR_COL].astype(int)
    df = df.sort_values(SR_COL).reset_index(drop=True)
    return df


def filter_group_a(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Keep only rows belonging to Group A. Returns ``(clean_df, dropped_others)``.

    Matching is case-insensitive and tolerates the common variants
    (``A``, ``group A``, ``groupA``, ``group_A``, ``group-A``).
    """
    normalized = (
        df[GROUP_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    is_group_a = normalized.isin(_GROUP_A_ACCEPTED)
    dropped = int((~is_group_a).sum())
    return df.loc[is_group_a].reset_index(drop=True), dropped


def drop_rows_without_duration(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows where ``Standard Duration (hh:mm:ss)`` is empty.

    Such rows are treated as "not real work items" and MUST NOT influence the
    Total / Already Uploaded / Remaining counts. Returns ``(clean_df, dropped)``.
    """
    duration = df[DURATION_COL]
    # [Reason] Treat NaN and whitespace-only strings as "no duration".
    is_missing = duration.isna() | (
        duration.astype(str).str.strip().isin(["", "nan", "NaN", "None"])
    )
    dropped = int(is_missing.sum())
    return df.loc[~is_missing].reset_index(drop=True), dropped


def filter_remaining(df: pd.DataFrame) -> Tuple[pd.DataFrame, SheetStats]:
    """Return (remaining_df, stats) where remaining has empty 'Uploaded Pecha Tool'."""
    uploaded = df[UPLOADED_COL]
    # [Reason] Treat NaN and whitespace-only strings as "not uploaded yet".
    is_empty = uploaded.isna() | (
        uploaded.astype(str).str.strip().isin(["", "nan", "NaN", "None"])
    )
    remaining = df[is_empty].reset_index(drop=True)
    stats = SheetStats(
        total=len(df),
        already_uploaded=int((~is_empty).sum()),
        remaining=len(remaining),
    )
    return remaining, stats


def list_empty_ranges(remaining_df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Return contiguous ``Sr. no`` ranges (inclusive) present in ``remaining_df``.

    Example: if remaining_df has Sr. no [1734, 1735, 1738, 1739, 1742],
    returns [(1734, 1735), (1738, 1739), (1742, 1742)].
    """
    if remaining_df.empty:
        return []
    srs = sorted(int(x) for x in remaining_df[SR_COL].tolist())
    ranges: List[Tuple[int, int]] = []
    start = prev = srs[0]
    for s in srs[1:]:
        if s == prev + 1:
            prev = s
            continue
        ranges.append((start, prev))
        start = prev = s
    ranges.append((start, prev))
    return ranges


def slice_by_sr_range(
    remaining_df: pd.DataFrame,
    start_sr: int,
    end_sr: int,
) -> pd.DataFrame:
    """Return rows whose ``Sr. no`` is within the inclusive ``[start_sr, end_sr]`` window."""
    mask = (remaining_df[SR_COL] >= start_sr) & (remaining_df[SR_COL] <= end_sr)
    return remaining_df.loc[mask].reset_index(drop=True)


# --- Duration parsing ------------------------------------------------------
def parse_durations(series: pd.Series) -> Tuple[pd.Series, int]:
    """Parse HH:MM:SS text to Timedelta. Returns (series, num_invalid).

    Invalid entries become ``Timedelta(0)`` so a bad row can never break a run,
    and the count is surfaced to the UI so the user is informed.
    """
    parsed = pd.to_timedelta(series.astype(str).str.strip(), errors="coerce")
    invalid = int(parsed.isna().sum())
    return parsed.fillna(pd.Timedelta(0)), invalid


def format_hhmmss(td: pd.Timedelta) -> str:
    """Format a Timedelta as zero-padded HH:MM:SS (hours can exceed 24)."""
    total_seconds = int(td.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# --- Splitting -------------------------------------------------------------
def _split_row_counts(total: int, users: int) -> List[int]:
    """Split ``total`` rows across ``users`` chunks; extra rows go to the first chunks."""
    base, extra = divmod(total, users)
    return [base + (1 if i < extra else 0) for i in range(users)]


def _contiguous_subranges(chunk: pd.DataFrame) -> List[Tuple[int, int, pd.DataFrame]]:
    """Break a chunk into contiguous ``Sr. no`` sub-ranges (jump > 1 => gap)."""
    if chunk.empty:
        return []
    sr = chunk[SR_COL].tolist()
    subranges: List[Tuple[int, int, pd.DataFrame]] = []
    start = 0
    for i in range(1, len(sr)):
        if sr[i] != sr[i - 1] + 1:
            sub = chunk.iloc[start:i]
            subranges.append((int(sub[SR_COL].iloc[0]), int(sub[SR_COL].iloc[-1]), sub))
            start = i
    sub = chunk.iloc[start:]
    subranges.append((int(sub[SR_COL].iloc[0]), int(sub[SR_COL].iloc[-1]), sub))
    return subranges


def build_user_ranges(
    remaining: pd.DataFrame,
    total_rows: int,
    users: int,
) -> Tuple[List[UserRange], bool, int]:
    """Take the first ``total_rows`` of ``remaining``, split among ``users``.

    Returns:
        ranges: flat list of ``UserRange`` (one per contiguous sub-range).
        had_gaps: True if any user's chunk was split because of an upload gap.
        invalid_durations: count of rows whose 'Standard Duration' was unparseable.
    """
    if total_rows <= 0 or users <= 0 or remaining.empty:
        return [], False, 0

    selected = remaining.head(total_rows).copy()
    durations, invalid = parse_durations(selected[DURATION_COL])
    # [Reason] Attach parsed durations for per-chunk sums without re-parsing.
    selected = selected.assign(_duration_td=durations.values)

    per_user_counts = _split_row_counts(len(selected), users)

    ranges: List[UserRange] = []
    had_gaps = False
    offset = 0
    for user_i, count in enumerate(per_user_counts, start=1):
        if count == 0:
            continue
        chunk = selected.iloc[offset : offset + count]
        offset += count

        subranges = _contiguous_subranges(chunk)
        if len(subranges) > 1:
            had_gaps = True

        for sub_i, (from_id, to_id, sub_df) in enumerate(subranges, start=1):
            ranges.append(
                UserRange(
                    user_index=user_i,
                    sub_index=sub_i,
                    from_id=from_id,
                    to_id=to_id,
                    rows=len(sub_df),
                    duration=sub_df["_duration_td"].sum(),
                )
            )

    return ranges, had_gaps, invalid
