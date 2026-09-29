#!/usr/bin/env python3
"""suvi_sdo_query.py — Query and download matched SUVI/SDO (AIA + HMI) observations.

Downloads SUVI and SDO (AIA + HMI) data for a given time interval, in one of three
modes:
  - quiet-sun:    find a window (contiguous or not) free of flares above a threshold.
  - flare-event:  download around HEK-matched flares of the requested class(es).
  - indistinct:   download everything in the range, no filtering.

By default the script only reports what is available (dry run, printed to the
terminal or written to CSV); pass --download to actually fetch files.

Usage:
  python scripts/suvi_sdo_query.py --start 2023-03-15T00:00:00 --end 2023-03-16T00:00:00 \\
      --mode quiet-sun --duration 1h
  python scripts/suvi_sdo_query.py --start 2023-03-15T00:00:00 --end 2023-03-16T00:00:00 \\
      --mode quiet-sun --duration 1h --csv --download
  python scripts/suvi_sdo_query.py --start 2023-08-05T00:00:00 --end 2023-08-06T00:00:00 \\
      --mode flare-event --flare-classes M X --download
  python scripts/suvi_sdo_query.py --start 2023-03-15T00:00:00 --end 2023-03-15T06:00:00 \\
      --mode indistinct --csv
"""

import argparse
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import astropy.units as u
import pandas as pd
from sunpy.net import Fido, attrs as a

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from suvi_sdo_common import (  # noqa: E402
    AIA_EXTRA_CHANNELS,
    GOES_SATELLITE_NUMBER_DEFAULT,
    HMI_PHYSOBS_MAP,
    HMI_PRODUCTS_DEFAULT,
    HMI_VECTOR_SEGMENTS,
    HMI_VECTOR_SERIES,
    SUVI_CHANNELS_DEFAULT,
    SUVI_TO_AIA_CHANNEL_MAP,
    channel_dir_name,
    csv_paths,
    download_root,
    get_timestamp,
    goes_to_numeric,
    hmi_products_from_manifest,
    load_manifest,
    nearest_indices,
    parse_class_threshold,
    timestamp_to_seconds,
    write_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class QuietSunUnavailableError(Exception):
    """Raised when the requested quiet-sun window cannot be satisfied."""


class NoFlareEventsError(Exception):
    """Raised when no flare events match the requested class filter (or exist at all)."""


# ---------------------------------------------------------------------------
# HEK flare search and GOES-class-threshold interval math
# ---------------------------------------------------------------------------


def query_hek_flares(start: str, end: str) -> pd.DataFrame:
    """Query HEK (via SWPC) for flare events in a time range.

    Args:
        start: Range start, ISO-8601 string.
        end: Range end, ISO-8601 string.

    Returns:
        A DataFrame with (at least) columns event_peaktime, event_starttime,
        event_endtime, fl_goescls, ar_noaanum. Empty (but with those columns)
        if no flares are found in the range.
    """
    result = Fido.search(a.Time(start, end), a.hek.FL, a.hek.FRM.Name == "SWPC")
    columns = ["event_peaktime", "event_starttime", "event_endtime", "fl_goescls", "ar_noaanum"]
    if "hek" not in result.keys() or len(result["hek"]) == 0:
        return pd.DataFrame(columns=columns)
    table = result["hek"]
    one_dim_columns = [name for name in table.colnames if len(table[name].shape) <= 1]
    df = table[one_dim_columns].to_pandas()
    for column in columns:
        if column not in df.columns:
            df[column] = None
    df["event_peaktime"] = pd.to_datetime(df["event_peaktime"])
    df["event_starttime"] = pd.to_datetime(df["event_starttime"])
    df["event_endtime"] = pd.to_datetime(df["event_endtime"])
    return df


def flare_intervals(df_hek: pd.DataFrame, min_class_numeric: float) -> list[tuple[datetime, datetime]]:
    """Extract (start, end) intervals for flares at or above a class threshold.

    Args:
        df_hek: A flare DataFrame from query_hek_flares().
        min_class_numeric: Minimum GOES class (as returned by goes_to_numeric())
            for a flare to count as "interrupting".

    Returns:
        A list of (start, end) datetime tuples, sorted by start time.
    """
    if df_hek.empty:
        return []
    numeric_class = df_hek["fl_goescls"].apply(goes_to_numeric)
    matched = df_hek[numeric_class >= min_class_numeric]
    intervals = list(zip(matched["event_starttime"], matched["event_endtime"]))
    return sorted(intervals, key=lambda pair: pair[0])


def merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping or touching (start, end) intervals.

    Args:
        intervals: A list of (start, end) tuples. Need not be pre-sorted.

    Returns:
        A list of non-overlapping (start, end) tuples, sorted by start time.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def quiet_gaps(
    range_start: datetime, range_end: datetime, interrupting_intervals: list[tuple[datetime, datetime]]
) -> list[tuple[datetime, datetime]]:
    """Compute the flare-free gaps within a range, given merged interrupting intervals.

    Args:
        range_start: Start of the range to search within.
        range_end: End of the range to search within.
        interrupting_intervals: Merged (non-overlapping) intervals that break
            quiet-sun contiguity, from merge_intervals(flare_intervals(...)).

    Returns:
        A list of (start, end) gap tuples covering everything in
        [range_start, range_end] not covered by interrupting_intervals.
    """
    gaps = []
    cursor = range_start
    for interval_start, interval_end in interrupting_intervals:
        clipped_start = max(interval_start, range_start)
        clipped_end = min(interval_end, range_end)
        if clipped_start >= clipped_end:
            continue
        if cursor < clipped_start:
            gaps.append((cursor, clipped_start))
        cursor = max(cursor, clipped_end)
    if cursor < range_end:
        gaps.append((cursor, range_end))
    return gaps


def select_contiguous_window(
    gaps: list[tuple[datetime, datetime]], duration
) -> tuple[datetime, datetime]:
    """Pick the earliest flare-free gap long enough to hold a contiguous window.

    Args:
        gaps: Flare-free gaps, chronologically ordered, from quiet_gaps().
        duration: A pandas.Timedelta, the requested quiet-sun duration.

    Returns:
        A (window_start, window_end) tuple: the start of the first qualifying
        gap, and that same start plus duration.

    Raises:
        QuietSunUnavailableError: If no gap is at least duration long. The
        error message names the longest gap actually found.
    """
    longest = None
    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start
        if longest is None or gap_length > longest[1] - longest[0]:
            longest = (gap_start, gap_end)
        if gap_length >= duration:
            return gap_start, gap_start + duration
    longest_duration = (longest[1] - longest[0]) if longest else pd.Timedelta(0)
    longest_start = longest[0] if longest else None
    raise QuietSunUnavailableError(
        f"No contiguous quiet-sun window of at least {duration} is available "
        f"(longest available contiguous quiet-sun window found: {longest_duration} "
        f"starting at {longest_start})."
    )


def select_noncontiguous_windows(
    gaps: list[tuple[datetime, datetime]], duration
) -> list[tuple[datetime, datetime]]:
    """Accumulate flare-free gaps chronologically until duration is covered.

    Args:
        gaps: Flare-free gaps, chronologically ordered, from quiet_gaps().
        duration: A pandas.Timedelta, the total quiet-sun duration requested.

    Returns:
        A list of (start, end) windows (a subset/truncation of gaps) whose
        total duration equals the requested duration.

    Raises:
        QuietSunUnavailableError: If the sum of all gaps is less than
        duration. The error message reports the total quiet time available.
    """
    windows = []
    remaining = duration
    for gap_start, gap_end in gaps:
        if remaining <= pd.Timedelta(0):
            break
        gap_length = gap_end - gap_start
        if gap_length >= remaining:
            windows.append((gap_start, gap_start + remaining))
            remaining = pd.Timedelta(0)
        else:
            windows.append((gap_start, gap_end))
            remaining -= gap_length
    if remaining > pd.Timedelta(0):
        available_total = duration - remaining
        raise QuietSunUnavailableError(
            f"Only {available_total} of quiet-sun time is available, less than the "
            f"requested {duration}, even allowing interrupted periods to be skipped."
        )
    return windows


def classify_timestamp(ts: datetime, df_hek: pd.DataFrame, min_class_numeric: float) -> str:
    """Classify a single timestamp as quiet-sun, a flare class, or neither.

    Args:
        ts: The timestamp to classify.
        df_hek: A flare DataFrame from query_hek_flares().
        min_class_numeric: Minimum GOES class for a flare to count as
            "interrupting" quiet-sun (see goes_to_numeric()).

    Returns:
        "quiet-sun" if no flare overlaps ts; "flare-class-<X>" if the
        strongest overlapping flare is >= min_class_numeric; "neither" if a
        flare overlaps but stays below the threshold.
    """
    if df_hek.empty:
        return "quiet-sun"
    overlapping = df_hek[(df_hek["event_starttime"] <= ts) & (ts <= df_hek["event_endtime"])]
    if overlapping.empty:
        return "quiet-sun"
    numeric_class = overlapping["fl_goescls"].apply(goes_to_numeric)
    strongest = overlapping.loc[numeric_class.idxmax()]
    if goes_to_numeric(strongest["fl_goescls"]) >= min_class_numeric:
        return f"flare-class-{strongest['fl_goescls']}"
    return "neither"


def match_flare_classes(df_hek: pd.DataFrame, flare_classes: list[str] | None) -> pd.DataFrame:
    """Filter HEK flares by a list of requested GOES classes.

    Args:
        df_hek: A flare DataFrame from query_hek_flares().
        flare_classes: Class prefixes to match (e.g. ["M", "X"] or ["M5.0"]).
            None means no filtering -- every event in df_hek is returned.

    Returns:
        The filtered DataFrame (a view/copy of df_hek).
    """
    if flare_classes is None:
        return df_hek
    prefixes = tuple(c.upper() for c in flare_classes)
    return df_hek[df_hek["fl_goescls"].str.upper().str.startswith(prefixes)]


def resolve_flare_windows(df_matched: pd.DataFrame, padding) -> list[dict]:
    """Build padded download windows and per-flare metadata from matched flares.

    Args:
        df_matched: Flares already filtered by match_flare_classes().
        padding: A pandas.Timedelta added before/after each flare's HEK
            start/end time.

    Returns:
        A list of dicts, one per flare, each with keys start, end, goes_class,
        peak_time, ar_noaanum. The caller merges the (start, end) pairs
        separately when it needs deduplicated download windows; this list
        keeps one entry per flare for the manifest.
    """
    events = []
    for _, row in df_matched.iterrows():
        events.append(
            {
                "start": row["event_starttime"] - padding,
                "end": row["event_endtime"] + padding,
                "goes_class": row["fl_goescls"],
                "peak_time": row["event_peaktime"],
                "ar_noaanum": row["ar_noaanum"],
            }
        )
    return events


# ---------------------------------------------------------------------------
# Instrument search wrappers
# ---------------------------------------------------------------------------


def _safe_fido_search(*args, **kwargs):
    """Run Fido.search, catching network/service errors instead of propagating them.

    A multi-hour batch download loops over many (channel, window) searches;
    one transient VSO/JSOC outage should not crash the whole job and discard
    everything downloaded so far.

    Args:
        *args: Positional attrs passed to Fido.search.
        **kwargs: Keyword attrs passed to Fido.search.

    Returns:
        The Fido.search result, or None if the search itself raised (e.g. a
        "Network is unreachable" VSO error). Callers must treat None the same
        as "nothing found for this request".
    """
    try:
        return Fido.search(*args, **kwargs)
    except Exception:
        SEARCH_FAILURES[0] += 1
        logger.warning("Fido.search failed (network/service error) -- treating as no results for this request.", exc_info=True)
        return None


def search_aia(channel_angstrom: int, start, end):
    """Search VSO for AIA data at one wavelength over a time range.

    Args:
        channel_angstrom: AIA wavelength in Angstrom.
        start: Range start.
        end: Range end.

    Returns:
        A sunpy search-result table, or [] if nothing matched or the search failed.
    """
    result = _safe_fido_search(
        a.Time(start, end), a.Instrument("AIA"), a.Level.one, a.Wavelength(channel_angstrom * u.Angstrom)
    )
    if result is None or len(result) == 0:
        return []
    return result["vso"] if "vso" in result.keys() else []


def search_suvi(channel_angstrom: int, start, end, goes_satellite_number: int):
    """Search for SUVI L2 data at one wavelength over a time range.

    Args:
        channel_angstrom: SUVI wavelength in Angstrom.
        start: Range start.
        end: Range end.
        goes_satellite_number: GOES-R series satellite number (e.g. 18).

    Returns:
        A sunpy search-result table, or [] if nothing matched or the search failed.
    """
    result = _safe_fido_search(
        a.Time(start, end),
        a.Instrument("SUVI"),
        a.Level.two,
        a.Wavelength(channel_angstrom * u.Angstrom),
        a.goes.SatelliteNumber(goes_satellite_number),
    )
    if result is None or len(result) == 0:
        return []
    return result["suvi"] if "suvi" in result.keys() else []


def search_hmi_product(product: str, start, end):
    """Search VSO for an HMI product (magnetogram or dopplergram) over a time range.

    Args:
        product: One of "magnetogram", "dopplergram" (keys of HMI_PHYSOBS_MAP).
        start: Range start.
        end: Range end.

    Returns:
        A sunpy search-result table, or [] if nothing matched or the search failed.

    Raises:
        KeyError: If product is not a recognized HMI_PHYSOBS_MAP key.
    """
    physobs_name = HMI_PHYSOBS_MAP[product]
    result = _safe_fido_search(a.Time(start, end), a.Instrument.hmi, getattr(a.Physobs, physobs_name))
    if result is None or len(result) == 0:
        return []
    return result["vso"] if "vso" in result.keys() else []


def search_hmi_vector_segment(segment: str, start, end, jsoc_email: str):
    """Search JSOC for one segment of the full-disk HMI vector field product.

    Args:
        segment: One of HMI_VECTOR_SEGMENTS ("field", "inclination", "azimuth",
            "disambig").
        start: Range start.
        end: Range end.
        jsoc_email: An email address pre-registered with JSOC
            (http://jsoc.stanford.edu/ajax/register_email.html); required by
            JSOC's export system.

    Returns:
        A sunpy JSOCResponse table, or [] if nothing matched or the search
        failed. Fido wraps it in a UnifiedResponse keyed "jsoc" (as it does
        "vso"/"suvi" for the other clients), so it is unwrapped here --
        extract_row_timestamps() cannot read the wrapper itself.
    """
    result = _safe_fido_search(
        a.Time(start, end),
        a.jsoc.Series(HMI_VECTOR_SERIES),
        a.jsoc.Segment(segment),
        a.jsoc.Notify(jsoc_email),
    )
    if result is None or len(result) == 0:
        return []
    return result["jsoc"] if "jsoc" in result.keys() else []


def extract_row_timestamps(search_result) -> list[datetime]:
    """Extract per-row timestamps from a Fido search-result table.

    Different clients (VSO, SUVI, JSOC) may name their timestamp column
    differently, so this tries a short list of known candidates.

    Args:
        search_result: A sunpy search-result table (or a client-specific
            table/QueryResponse) with zero or more rows.

    Returns:
        A list of datetimes, one per row, in table order. Empty if the table
        has no rows or no recognized timestamp column.
    """
    if len(search_result) == 0:
        return []
    try:
        if hasattr(search_result, "colnames"):
            # Drop multidimensional columns (e.g. a Wavelength range column like
            # "304.0 .. 304.0") before to_pandas(), which can't represent them --
            # same fix the notebook already applies to HEK's result table.
            one_dim_names = [name for name in search_result.colnames if len(search_result[name].shape) <= 1]
            df = search_result[one_dim_names].to_pandas()
        elif hasattr(search_result, "to_pandas"):
            df = search_result.to_pandas()
        else:
            df = pd.DataFrame(search_result)
    except Exception:
        logger.warning("Could not convert search result to a DataFrame to extract timestamps.", exc_info=True)
        return []
    candidate_columns = ["Start Time", "Time", "time", "date_obs", "DATE-OBS", "T_REC"]
    for column in candidate_columns:
        if column in df.columns:
            return list(pd.to_datetime(df[column]))
    logger.warning(
        "Could not find a timestamp column among %s in search result columns %s -- "
        "verify the correct column name for this client empirically.",
        candidate_columns,
        list(df.columns),
    )
    return []


# Per-run tallies, read by main() to decide the manifest's final status honestly.
DOWNLOAD_TALLY: dict[str, dict[str, int]] = {}
SEARCH_FAILURES: list[int] = [0]

# Fetch settings, overridden from the CLI in main(). The AIA/HMI export servers answered ~40% of
# 5-way-parallel requests with errors/timeouts in real runs, so the default is deliberately low.
FETCH_MAX_CONN: list[int] = [2]
FETCH_RETRIES: list[int] = [3]
RETRY_WAIT_SECONDS: list[float] = [60.0]

# --match-suvi bookkeeping: largest |AIA/HMI time - SUVI reference time| per folder, in seconds.
PAIRING_MAX_DELTA: dict[str, float] = {}
# Search margin around a window so the nearest frame can be chosen on both sides of its edges.
MATCH_MARGIN = pd.Timedelta(minutes=6)


def _record_expected(dest_dir: Path, expected: int, missing: int) -> None:
    """Accumulate how many files a folder should have and how many are still absent.

    Args:
        dest_dir: The folder the download targeted.
        expected: Number of remote rows the search reported for this call.
        missing: How many of those are still not present locally.
    """
    entry = DOWNLOAD_TALLY.setdefault(str(dest_dir), {"expected": 0, "missing": 0})
    entry["expected"] += expected
    entry["missing"] += missing


def _missing_indices(remote_timestamps: list, dest_dir: Path, tolerance_seconds: float) -> list[int]:
    """Find which remote rows have no valid local file in dest_dir.

    A local file counts as present only if its DATE-OBS can be read and lies within
    tolerance_seconds of the remote row's timestamp -- a truncated or corrupt file
    that fails to parse is treated as absent.

    Args:
        remote_timestamps: Per-row timestamps of the search result, in table order.
        dest_dir: Directory holding the local FITS files.
        tolerance_seconds: Maximum timestamp difference for a match.

    Returns:
        Indices (into remote_timestamps) of rows without a matching local file.
    """
    existing_seconds = []
    for f in sorted(Path(dest_dir).glob("*.fits")):
        try:
            # SUVI rows are keyed by product start, but its DATE-OBS falls 1-3 min later
            # (per channel), so the filename's start stamp is the comparable value.
            suvi_start = re.search(r"_s(\d{8}T\d{6})Z", f.name)
            existing_seconds.append(
                timestamp_to_seconds(suvi_start.group(1)) if suvi_start else timestamp_to_seconds(get_timestamp(f))
            )
        except Exception:
            logger.warning("Could not read timestamp from existing file %s, ignoring it for dedup.", f)
    return [
        i for i, ts in enumerate(remote_timestamps)
        if not any(abs(ts.timestamp() - existing) <= tolerance_seconds for existing in existing_seconds)
    ]


def download_missing(search_result, dest_dir: Path, tolerance_seconds: float = 5.0) -> list[Path]:
    """Download only the rows of a search result not already present locally.

    Compares each remote row's timestamp against existing local FITS files'
    DATE-OBS (within a tolerance), so re-running the same query is idempotent
    instead of the all-or-nothing "skip if any .fits exists" behaviour of the
    original notebook.

    Args:
        search_result: A sunpy search-result table to (partially) fetch.
        dest_dir: Directory to download into (created if missing).
        tolerance_seconds: Maximum timestamp difference, in seconds, for a
            remote row to be considered "already downloaded".

    Returns:
        A sorted list of all local FITS file paths in dest_dir after the
        download (both pre-existing and newly fetched).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    remote_timestamps = extract_row_timestamps(search_result)
    if not remote_timestamps:
        return sorted(dest_dir.glob("*.fits"))

    indices = _missing_indices(remote_timestamps, dest_dir, tolerance_seconds)
    if not indices:
        logger.info("All %d file(s) already present in %s, skipping download.", len(remote_timestamps), dest_dir)
    for attempt in range(1 + FETCH_RETRIES[0]):
        if not indices:
            break
        if attempt:
            wait = min(RETRY_WAIT_SECONDS[0] * 2 ** (attempt - 1), 900.0)
            logger.info(
                "Retrying %d missing file(s) in %s (attempt %d) after %.0f s.", len(indices), dest_dir, attempt + 1, wait,
            )
            time.sleep(wait)
        # overwrite=True: our own timestamp-based check already decided these specific
        # files are missing. Fido.fetch's own default (overwrite=False) matches existing
        # files by name only, so a truncated file from an interrupted download -- which
        # fails the get_timestamp() parse and so doesn't count as present -- would
        # otherwise be silently skipped forever instead of being replaced.
        Fido.fetch(search_result[indices], path=str(dest_dir), overwrite=True, max_conn=FETCH_MAX_CONN[0])
        indices = _missing_indices(remote_timestamps, dest_dir, tolerance_seconds)

    # Re-check after fetching: parfive reports per-file HTTP failures only in its own log
    # and Fido.fetch does not raise, so without this the run would look successful even when
    # the server answered most requests with errors.
    still_missing = len(_missing_indices(remote_timestamps, dest_dir, tolerance_seconds))
    _record_expected(dest_dir, expected=len(remote_timestamps), missing=still_missing)
    if still_missing:
        logger.warning(
            "%d of %d file(s) still missing in %s after the download attempt (server errors?).",
            still_missing, len(remote_timestamps), dest_dir,
        )
    return sorted(dest_dir.glob("*.fits"))


# ---------------------------------------------------------------------------
# Dry-run listing
# ---------------------------------------------------------------------------


def build_listing_df(
    instrument: str,
    channels: list[int],
    windows: list[tuple[datetime, datetime]],
    mode: str,
    df_hek: pd.DataFrame,
    min_class_numeric: float,
    goes_satellite_number: int,
    window_flare_class: str | None = None,
) -> pd.DataFrame:
    """Build the dry-run availability listing for one instrument.

    Args:
        instrument: "aia" or "suvi".
        channels: Wavelengths (in the instrument's own numbering) to search.
        windows: Time windows to search within (quiet-sun: the selected
            window(s); flare-event: the per-flare padded windows; indistinct:
            a single [start, end] window).
        mode: One of "quiet-sun", "flare-event", "indistinct" -- controls
            which label column is added.
        df_hek: Flare DataFrame, used only in indistinct mode to classify
            each timestamp.
        min_class_numeric: Threshold used for indistinct-mode classification.
        goes_satellite_number: GOES-R satellite number, for SUVI searches.
        window_flare_class: In flare-event mode, the GOES class label to
            apply to every row from this call (callers loop per flare window
            and pass that flare's class here).

    Returns:
        A DataFrame with columns dates, times, and a mode-specific label
        column (quiet-sun / flare-class / type_of_observation).
    """
    rows = []
    for window_start, window_end in windows:
        for channel in channels:
            if instrument == "aia":
                result = search_aia(channel, window_start, window_end)
            else:
                result = search_suvi(channel, window_start, window_end, goes_satellite_number)
            for ts in extract_row_timestamps(result):
                row = {"dates": ts.date().isoformat(), "times": ts.strftime("%H:%M:%S")}
                if mode == "quiet-sun":
                    row["quiet-sun"] = "yes"
                elif mode == "flare-event":
                    row["flare-class"] = window_flare_class
                else:
                    row["type_of_observation"] = classify_timestamp(ts, df_hek, min_class_numeric)
                rows.append(row)
    columns = ["dates", "times"] + (
        ["quiet-sun"] if mode == "quiet-sun" else ["flare-class"] if mode == "flare-event" else ["type_of_observation"]
    )
    return pd.DataFrame(rows, columns=columns)


def print_or_write(df: pd.DataFrame, csv_path: Path | None, label: str) -> None:
    """Print a listing DataFrame to the terminal, or write it to CSV.

    Args:
        df: The listing DataFrame to output.
        csv_path: If given, write df here (creating parent directories) and
            log the path instead of printing. If None, print df to stdout.
        label: A short instrument label ("SDO"/"SUVI") used in the printed
            header, for readability when both listings go to the terminal.
    """
    if csv_path is None:
        print(f"\n=== {label} availability listing ===")
        print(df.to_string(index=False) if not df.empty else "(no rows)")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s listing to %s (%d rows).", label, csv_path, len(df))


# ---------------------------------------------------------------------------
# HMI download orchestration
# ---------------------------------------------------------------------------


def subset_nearest(search_result, target_seconds: list[float], label: str):
    """Keep only the rows of a search result nearest in time to each target.

    Args:
        search_result: A sunpy search-result table (or [] if empty/failed).
        target_seconds: Reference times (Unix seconds) to match, e.g. SUVI observation times.
        label: Folder label used to log and record the worst time mismatch.

    Returns:
        The sub-table of chosen rows, or [] if there is nothing to match against.
        The largest |row time - target| is folded into PAIRING_MAX_DELTA[label].
    """
    timestamps = extract_row_timestamps(search_result)
    if not timestamps or not target_seconds:
        return []
    indices, max_delta = nearest_indices([t.timestamp() for t in timestamps], target_seconds)
    PAIRING_MAX_DELTA[label] = max(PAIRING_MAX_DELTA.get(label, 0.0), max_delta)
    logger.info(
        "%s: %d of %d available frame(s) are nearest to a SUVI time (worst mismatch %.0f s).",
        label, len(indices), len(timestamps), max_delta,
    )
    return search_result[indices]


def suvi_observation_times(root: Path, suvi_channels: list[int]) -> tuple[dict[int, list[float]], list[float]]:
    """Read the real observation times of the SUVI files already on disk.

    SUVI cycles through its channels inside each 4-minute product, so channels of
    one product are observed up to ~3 minutes apart (measured on real data): there
    is no single "SUVI instant". Each channel therefore keeps its own DATE-OBS
    list, and a per-product median is provided as the reference for anything that
    has no SUVI counterpart (AIA 335/1600, HMI).

    Args:
        root: The download-root directory holding suvi/<channel>/ folders.
        suvi_channels: SUVI wavelengths whose folders to read.

    Returns:
        A (per_channel, product_medians) tuple: per_channel maps each SUVI
        wavelength to its sorted DATE-OBS times in Unix seconds; product_medians
        is the sorted median time of each 4-minute product across channels.
    """
    per_channel: dict[int, list[float]] = {}
    products: dict[str, list[float]] = defaultdict(list)
    for channel in suvi_channels:
        seconds = []
        for f in sorted((root / "suvi" / channel_dir_name(channel)).glob("*.fits")):
            try:
                t = timestamp_to_seconds(get_timestamp(f))
            except Exception:
                logger.warning("Could not read the observation time of %s; skipping it as a reference.", f)
                continue
            seconds.append(t)
            match = re.search(r"_s(\d{8}T\d{6})Z", f.name)
            products[match.group(1) if match else str(int(t // 240))].append(t)
        per_channel[channel] = sorted(seconds)
    return per_channel, sorted(statistics.median(v) for v in products.values())


def _targets_in_window(target_seconds: list[float], window_start, window_end) -> list[float]:
    """Select the reference times that belong to a download window.

    A SUVI product that overlaps the window can be observed up to one product
    length (4 min) outside it, so the window is widened by that much.

    Args:
        target_seconds: Reference times in Unix seconds.
        window_start: Window start.
        window_end: Window end.

    Returns:
        The subset of target_seconds inside the widened window.
    """
    slack = 240.0
    lo, hi = pd.Timestamp(window_start).timestamp() - slack, pd.Timestamp(window_end).timestamp() + slack
    return [t for t in target_seconds if lo <= t <= hi]


def download_matched(
    root: Path,
    windows: list[tuple[datetime, datetime]],
    suvi_channels: list[int],
    aia_channels: list[int],
    aia_extra_channels: list[int],
    hmi_products: list[str],
    jsoc_email: str | None,
    goes_satellite_number: int,
) -> dict:
    """Download SUVI in full, and only the AIA/HMI frames nearest in time to it.

    Translating between instruments needs the Sun's state as simultaneous as
    possible, so instead of every native-cadence frame this fetches SUVI (the
    reference, fast and reliable) and then, per SUVI observation, the closest
    frame of the paired AIA channel (matched against that SUVI channel's own
    times) and of each unpaired product (matched against per-product medians).

    Args:
        root: The download-root directory.
        windows: Time windows to download within.
        suvi_channels: SUVI wavelengths.
        aia_channels: AIA wavelengths paired 1:1 with suvi_channels (same order).
        aia_extra_channels: AIA wavelengths with no SUVI pair (335/1600).
        hmi_products: Subset of ["magnetogram", "dopplergram", "vector"].
        jsoc_email: Required if "vector" is requested.
        goes_satellite_number: GOES-R satellite number for the SUVI search.

    Returns:
        The manifest "hmi_products" entry from download_hmi_products().

    Raises:
        ValueError: If "vector" is requested without a jsoc_email.
    """
    for channel in suvi_channels:
        for window_start, window_end in windows:
            result = search_suvi(channel, window_start, window_end, goes_satellite_number)
            download_missing(result, root / "suvi" / channel_dir_name(channel))

    per_channel, product_medians = suvi_observation_times(root, suvi_channels)
    logger.info(
        "Reference: %d SUVI product(s); matching AIA/HMI frames to them (%s).",
        len(product_medians), {c: len(v) for c, v in per_channel.items()},
    )

    for suvi_channel, aia_channel in zip(suvi_channels, aia_channels):
        folder = channel_dir_name(aia_channel)
        for window_start, window_end in windows:
            targets = _targets_in_window(per_channel[suvi_channel], window_start, window_end)
            result = search_aia(aia_channel, window_start - MATCH_MARGIN, window_end + MATCH_MARGIN)
            download_missing(subset_nearest(result, targets, f"aia/{folder}"), root / "aia" / folder)

    for aia_channel in aia_extra_channels:
        folder = channel_dir_name(aia_channel)
        for window_start, window_end in windows:
            targets = _targets_in_window(product_medians, window_start, window_end)
            result = search_aia(aia_channel, window_start - MATCH_MARGIN, window_end + MATCH_MARGIN)
            download_missing(subset_nearest(result, targets, f"aia/{folder}"), root / "aia" / folder)

    return download_hmi_products(hmi_products, windows, root, jsoc_email, ref_seconds=product_medians)


def build_requested_hmi_entry(hmi_products: list[str]) -> dict:
    """Build the manifest hmi_products entry for what was requested, before downloading.

    Written into the early (pre-download) manifest.json so a crashed run still
    records intent. download_hmi_products()'s return value later overwrites this
    with the post-download outcome, in the same shape but marking "downloaded"
    instead of "requested".

    Args:
        hmi_products: Subset of ["magnetogram", "dopplergram", "vector"].

    Returns:
        A dict in the same shape download_hmi_products() returns.
    """
    return {
        "magnetogram": "magnetogram" in hmi_products,
        "dopplergram": "dopplergram" in hmi_products,
        "vector": (
            {"requested": True, "segments": list(HMI_VECTOR_SEGMENTS)}
            if "vector" in hmi_products else None
        ),
    }


def download_hmi_products(
    hmi_products: list[str],
    windows: list[tuple[datetime, datetime]],
    root: Path,
    jsoc_email: str | None,
    ref_seconds: list[float] | None = None,
) -> dict:
    """Download the requested HMI products over a set of windows.

    Args:
        hmi_products: Subset of ["magnetogram", "dopplergram", "vector"].
        windows: Time windows to download within.
        root: The download-root directory; HMI files go under
            root/hmi/<product>/ (or root/hmi/vector/<segment>/ for vector).
        jsoc_email: Required if "vector" is in hmi_products; ignored otherwise.
        ref_seconds: If given (--match-suvi), download only the frame nearest to
            each of these reference times instead of every frame in the window.

    Returns:
        A dict describing what was attempted, suitable for manifest.json's
        "hmi_products" key.
    """
    manifest_entry: dict = {"magnetogram": False, "dopplergram": False, "vector": None}
    for product in ("magnetogram", "dopplergram"):
        if product not in hmi_products:
            continue
        dest_dir = root / "hmi" / product
        for window_start, window_end in windows:
            if ref_seconds is None:
                result = search_hmi_product(product, window_start, window_end)
            else:
                result = subset_nearest(
                    search_hmi_product(product, window_start - MATCH_MARGIN, window_end + MATCH_MARGIN),
                    _targets_in_window(ref_seconds, window_start, window_end), f"hmi/{product}",
                )
            download_missing(result, dest_dir)
        manifest_entry[product] = True

    if "vector" in hmi_products:
        if not jsoc_email:
            raise ValueError(
                "--hmi-products vector requires --jsoc-email (register a free email at "
                "http://jsoc.stanford.edu/ajax/register_email.html before using this flag)."
            )
        for segment in HMI_VECTOR_SEGMENTS:
            dest_dir = root / "hmi" / "vector" / segment
            for window_start, window_end in windows:
                if ref_seconds is None:
                    result = search_hmi_vector_segment(segment, window_start, window_end, jsoc_email)
                else:
                    result = subset_nearest(
                        search_hmi_vector_segment(
                            segment, window_start - MATCH_MARGIN, window_end + MATCH_MARGIN, jsoc_email,
                        ),
                        _targets_in_window(ref_seconds, window_start, window_end), f"hmi/vector/{segment}",
                    )
                download_missing(result, dest_dir)
        manifest_entry["vector"] = {
            "attempted": True,
            "segments": HMI_VECTOR_SEGMENTS,
            "note": (
                "RAW, NOT Bx/By/Bz -- disambiguation (using the disambig bit over azimuth) and a "
                "heliographic geometric conversion are not applied here; that is a separate task."
            ),
        }
    return manifest_entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse (defaults to sys.argv[1:] via argparse).

    Returns:
        The parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Query and download matched SUVI/SDO (AIA + HMI) observations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resume", default=None, metavar="FOLDER",
        help=(
            "Resume an interrupted download from an existing folder's manifest.json instead of "
            "defining a new query -- rebuilds mode/channels/windows/HMI products from it, skips the "
            "dry-run listing, and goes straight to downloading (already-present files are skipped, "
            "truncated ones are replaced). Mutually exclusive with --start/--end/--mode and every "
            "other query-defining flag. --jsoc-email must still be passed if the manifest requested "
            "'vector' -- it is never persisted to disk."
        ),
    )
    parser.add_argument("--start", default=None, help="Range start, ISO 8601 (e.g. 2023-03-15T00:00:00). Required unless --resume is used.")
    parser.add_argument("--end", default=None, help="Range end, ISO 8601. Required unless --resume is used.")
    parser.add_argument(
        "--mode", default=None, choices=["quiet-sun", "flare-event", "indistinct"],
        help="Download mode: quiet-sun, flare-event, or indistinct (everything, unfiltered). Required unless --resume is used.",
    )
    parser.add_argument(
        "--channels", nargs="+", type=int, default=None,
        help=(
            "SUVI wavelengths in Angstrom to download (e.g. --channels 304). Each SUVI wavelength's "
            "paired AIA wavelength is derived automatically. Default: all 6 shared channels."
        ),
    )
    parser.add_argument(
        "--aia-extra-channels", nargs="+", type=int, choices=AIA_EXTRA_CHANNELS, default=None,
        help=(
            "AIA-only wavelengths that Surya uses but have no SUVI equivalent at all "
            "(335, 1600 -- see docs/suvi_channel_mapping_preprocessing.md). Downloaded into "
            "aia/<channel>/ with no matching suvi/ folder. Default: none."
        ),
    )
    parser.add_argument(
        "--all-sdo-channels", action="store_true",
        help=(
            "Convenience flag for the full SDO side of Surya's 13 channels: the 6 SUVI-paired AIA "
            "channels (--channels default) + AIA 335/1600 (--aia-extra-channels default) + all 3 "
            "HMI products (--hmi-products default). Overrides --channels/--aia-extra-channels/"
            "--hmi-products if given alongside this flag."
        ),
    )
    parser.add_argument(
        "--min-interrupt-class", default="C",
        help="Minimum GOES flare class (bare letter or full class) that breaks quiet-sun contiguity.",
    )
    parser.add_argument(
        "--goes-satellite-number", type=int, default=GOES_SATELLITE_NUMBER_DEFAULT,
        help="GOES satellite number for the SUVI query.",
    )
    parser.add_argument(
        "--duration", default=None,
        help="Amount of quiet-sun time requested, as a pandas.Timedelta string (e.g. 1h). Required in quiet-sun mode.",
    )
    parser.add_argument(
        "--contiguous", dest="contiguous", action=argparse.BooleanOptionalAction, default=True,
        help="Require the quiet-sun window to be uninterrupted (default), or allow skipping interrupting periods with --no-contiguous.",
    )
    parser.add_argument(
        "--flare-classes", nargs="+", default=None,
        help="GOES flare classes to include (e.g. --flare-classes M X). Omit for every flare in range.",
    )
    parser.add_argument(
        "--flare-padding", default="0min",
        help="Extra time padded before/after each matched flare, as a pandas.Timedelta string.",
    )
    parser.add_argument(
        "--match-suvi", action="store_true",
        help=(
            "Download SUVI in full but only the AIA/HMI frame NEAREST IN TIME to each SUVI observation "
            "(instead of every native-cadence frame). Matches what a SUVI<->SDO translation needs -- the "
            "most simultaneous Sun states -- and cuts ~300 AIA frames/channel/hour to ~one per SUVI image."
        ),
    )
    parser.add_argument(
        "--max-conn", type=int, default=FETCH_MAX_CONN[0],
        help="Parallel connections per fetch. AIA/HMI servers failed ~40%% of 5-way parallel requests; default 2.",
    )
    parser.add_argument(
        "--retries", type=int, default=FETCH_RETRIES[0],
        help="Extra attempts per folder after a failed fetch, with exponentially growing waits (default 3).",
    )
    parser.add_argument(
        "--retry-wait", type=float, default=RETRY_WAIT_SECONDS[0],
        help="Seconds before the first retry; doubles each attempt, capped at 900 s (default 60).",
    )
    parser.add_argument(
        "--sweeps", type=int, default=3,
        help="After the full pass, repeat it up to this many times while files are still missing (default 3).",
    )
    parser.add_argument(
        "--sweep-wait", type=float, default=600.0,
        help="Seconds before the first sweep; doubles each sweep, capped at 3600 s (default 600).",
    )
    parser.add_argument("--csv", action="store_true", help="Write the dry-run listing to CSV instead of printing it.")
    parser.add_argument("--output-dir", default="data", help="Root directory for the download folder.")
    parser.add_argument(
        "--download", action="store_true",
        help="Actually fetch the FITS files (and write manifest.json). Without this, only the dry-run listing runs.",
    )
    parser.add_argument(
        "--hmi-products", nargs="+", choices=["magnetogram", "dopplergram", "vector"],
        default=list(HMI_PRODUCTS_DEFAULT), help="HMI products to include (SUVI has no HMI equivalent).",
    )
    parser.add_argument(
        "--jsoc-email", default=None,
        help="Email pre-registered with JSOC, required if 'vector' is in --hmi-products.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point: parse args, resolve download windows, list and/or download.

    Two entry paths: a fresh query (--start/--end/--mode, listing + optional
    --download) or --resume <folder>, which rebuilds everything from that
    folder's manifest.json and skips straight to downloading.
    """
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)
    FETCH_MAX_CONN[0] = args.max_conn
    FETCH_RETRIES[0] = args.retries
    RETRY_WAIT_SECONDS[0] = args.retry_wait

    if args.resume and (args.start or args.end or args.mode):
        logger.error("--resume cannot be combined with --start/--end/--mode.")
        sys.exit(2)
    if not args.resume and not (args.start and args.end and args.mode):
        logger.error("Either --resume FOLDER, or --start/--end/--mode together, are required.")
        sys.exit(2)

    if args.resume:
        root = Path(args.resume)
        manifest = load_manifest(root)
        mode = manifest["mode"]
        suvi_channels = manifest["channels"]["suvi_angstrom"]
        aia_channels = manifest["channels"]["aia_angstrom"]
        aia_extra_channels = manifest["channels"].get("aia_extra_angstrom", [])
        hmi_products = hmi_products_from_manifest(manifest)
        match_suvi = bool(manifest.get("match_suvi", False))
        windows = [(pd.Timestamp(w["start"]), pd.Timestamp(w["end"])) for w in manifest["download_windows"]]
        logger.info(
            "Resuming %s download in %s (%d window(s), status was %r).",
            mode, root, len(windows), manifest.get("status"),
        )
    else:
        if args.mode == "quiet-sun" and args.duration is None:
            logger.error("--mode quiet-sun requires --duration.")
            sys.exit(2)
        if args.mode != "quiet-sun" and (args.duration is not None):
            logger.warning("--duration is only used in --mode quiet-sun; ignoring it.")
        if args.mode != "flare-event" and (args.flare_classes is not None or args.flare_padding != "0min"):
            logger.warning("--flare-classes/--flare-padding are only used in --mode flare-event; ignoring them.")

        min_class_numeric = parse_class_threshold(args.min_interrupt_class)

        if args.all_sdo_channels:
            if args.channels or args.aia_extra_channels or args.hmi_products != list(HMI_PRODUCTS_DEFAULT):
                logger.warning("--all-sdo-channels overrides --channels/--aia-extra-channels/--hmi-products.")
            suvi_channels = list(SUVI_CHANNELS_DEFAULT)
            aia_extra_channels = list(AIA_EXTRA_CHANNELS)
            args.hmi_products = list(HMI_PRODUCTS_DEFAULT)
        else:
            suvi_channels = args.channels or SUVI_CHANNELS_DEFAULT
            aia_extra_channels = args.aia_extra_channels or []
        hmi_products = args.hmi_products
        match_suvi = args.match_suvi

        try:
            aia_channels = [SUVI_TO_AIA_CHANNEL_MAP[c] for c in suvi_channels]
        except KeyError as exc:
            logger.error("No AIA equivalent for SUVI channel %sA.", exc.args[0])
            sys.exit(2)

        range_start = pd.Timestamp(args.start)
        range_end = pd.Timestamp(args.end)
        df_hek = query_hek_flares(args.start, args.end)

        flare_events: list[dict] | None = None
        try:
            if args.mode == "quiet-sun":
                duration = pd.Timedelta(args.duration)
                merged = merge_intervals(flare_intervals(df_hek, min_class_numeric))
                gaps = quiet_gaps(range_start, range_end, merged)
                if args.contiguous:
                    window = select_contiguous_window(gaps, duration)
                    windows = [window]
                else:
                    windows = select_noncontiguous_windows(gaps, duration)
            elif args.mode == "flare-event":
                df_matched = match_flare_classes(df_hek, args.flare_classes)
                if df_matched.empty:
                    if args.flare_classes:
                        raise NoFlareEventsError(
                            f"No flare events of class(es) {args.flare_classes} found between {args.start} and {args.end}."
                        )
                    raise NoFlareEventsError(f"No flare events found between {args.start} and {args.end}.")
                flare_events = resolve_flare_windows(df_matched, pd.Timedelta(args.flare_padding))
                windows = merge_intervals([(e["start"], e["end"]) for e in flare_events])
            else:  # indistinct
                windows = [(range_start, range_end)]
        except QuietSunUnavailableError as exc:
            logger.error(
                "%s (requested between %s and %s, min-interrupt-class=%s).",
                exc, args.start, args.end, args.min_interrupt_class,
            )
            sys.exit(1)
        except NoFlareEventsError as exc:
            logger.error(str(exc))
            sys.exit(1)

        mode = args.mode
        root = download_root(args.output_dir, mode, args.start, args.end)

        for instrument, channels in (("aia", aia_channels + aia_extra_channels), ("suvi", suvi_channels)):
            window_flare_class = None
            if mode == "flare-event" and flare_events:
                # One listing call per flare window so each gets its own class label.
                frames = []
                for event in flare_events:
                    frame = build_listing_df(
                        instrument, channels, [(event["start"], event["end"])], mode, df_hek,
                        min_class_numeric, args.goes_satellite_number, window_flare_class=event["goes_class"],
                    )
                    frames.append(frame)
                listing = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                listing = build_listing_df(
                    instrument, channels, windows, mode, df_hek, min_class_numeric, args.goes_satellite_number,
                )
            sdo_csv, suvi_csv = csv_paths(root, mode, args.start, args.end)
            csv_path = (sdo_csv if instrument == "aia" else suvi_csv) if args.csv else None
            print_or_write(listing, csv_path, label="SDO" if instrument == "aia" else "SUVI")

        if not args.download:
            logger.info("Dry run only -- pass --download to fetch these files.")
            return

        # Written now, before any download starts, so a crash mid-download still leaves a
        # record of what was requested -- that's what --resume reads back later.
        manifest = {
            "mode": mode,
            "requested_start": args.start,
            "requested_end": args.end,
            "min_interrupt_class": args.min_interrupt_class,
            "goes_satellite_number": args.goes_satellite_number,
            "channels": {
                "suvi_angstrom": suvi_channels,
                "aia_angstrom": aia_channels,
                "aia_extra_angstrom": aia_extra_channels,
            },
            "hmi_products": build_requested_hmi_entry(hmi_products),
            "match_suvi": match_suvi,
            "download_windows": [{"start": str(s), "end": str(e)} for s, e in windows],
            "quiet_sun": (
                {"requested_duration": args.duration, "contiguous_requested": args.contiguous}
                if mode == "quiet-sun" else None
            ),
            "flare_events": (
                [
                    {
                        "goes_class": e["goes_class"], "peak_time": str(e["peak_time"]),
                        "start_time": str(e["start"]), "end_time": str(e["end"]), "ar_noaanum": e["ar_noaanum"],
                    }
                    for e in flare_events
                ]
                if flare_events else None
            ),
            "created_at": datetime.utcnow().isoformat(),
            "script_version": "1.0",
            "status": "in_progress",
        }
        write_manifest(root, manifest)
        logger.info("Wrote manifest.json (status=in_progress) to %s before starting the download.", root)

    for sweep in range(args.sweeps + 1):
        DOWNLOAD_TALLY.clear()
        SEARCH_FAILURES[0] = 0
        PAIRING_MAX_DELTA.clear()
        try:
            if match_suvi:
                hmi_manifest_entry = download_matched(
                    root, windows, suvi_channels, aia_channels, aia_extra_channels,
                    hmi_products, args.jsoc_email, args.goes_satellite_number,
                )
            else:
                for suvi_channel, aia_channel in zip(suvi_channels, aia_channels):
                    for window_start, window_end in windows:
                        aia_result = search_aia(aia_channel, window_start, window_end)
                        download_missing(aia_result, root / "aia" / channel_dir_name(aia_channel))
                        suvi_result = search_suvi(suvi_channel, window_start, window_end, args.goes_satellite_number)
                        download_missing(suvi_result, root / "suvi" / channel_dir_name(suvi_channel))

                for aia_channel in aia_extra_channels:
                    for window_start, window_end in windows:
                        aia_result = search_aia(aia_channel, window_start, window_end)
                        download_missing(aia_result, root / "aia" / channel_dir_name(aia_channel))

                hmi_manifest_entry = download_hmi_products(hmi_products, windows, root, args.jsoc_email)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)
        if not SEARCH_FAILURES[0] and not any(c["missing"] for c in DOWNLOAD_TALLY.values()):
            break
        if sweep < args.sweeps:
            wait = min(args.sweep_wait * 2 ** sweep, 3600.0)
            logger.warning(
                "Pass %d left files missing or searches failed (server errors?); sweeping again in %.0f s (%d sweep(s) left).",
                sweep + 1, wait, args.sweeps - sweep,
            )
            time.sleep(wait)

    manifest["hmi_products"] = hmi_manifest_entry
    missing_by_folder = {
        Path(folder).relative_to(root).as_posix() if Path(folder).is_relative_to(root) else folder: counts
        for folder, counts in DOWNLOAD_TALLY.items() if counts["missing"] > 0
    }
    total_missing = sum(c["missing"] for c in missing_by_folder.values())
    total_expected = sum(c["expected"] for c in DOWNLOAD_TALLY.values())
    incomplete = bool(total_missing or SEARCH_FAILURES[0])
    manifest["status"] = "incomplete" if incomplete else "complete"
    manifest["missing_files"] = missing_by_folder
    manifest["search_failures"] = SEARCH_FAILURES[0]
    if match_suvi:
        manifest["pairing_max_delta_seconds"] = {k: round(v, 1) for k, v in sorted(PAIRING_MAX_DELTA.items())}
    write_manifest(root, manifest)
    if incomplete:
        logger.error(
            "Download INCOMPLETE: %d of %d expected file(s) missing across %d folder(s), %d search(es) failed. "
            "Re-run with --resume %s to retry only what is missing. Details in manifest.json.",
            total_missing, total_expected, len(missing_by_folder), SEARCH_FAILURES[0], root,
        )
        sys.exit(3)
    logger.info("Download complete (%d file(s) verified). manifest.json (status=complete) written to %s.", total_expected, root)


if __name__ == "__main__":
    main()
