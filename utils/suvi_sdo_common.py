"""suvi_sdo_common.py — Shared helpers for suvi_sdo_query.py and visualization.py.

Not a CLI entry point. Holds the channel-mapping table, GOES-class arithmetic, FITS
timestamp helpers, the folder/file naming scheme, manifest.json read/write, and the
two label-box overlay implementations (OpenCV for video frames, Matplotlib for static
plots) shared by both scripts.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

# --- Channel mapping (see docs/suvi_channel_mapping_preprocessing.md, section 5) ---

SUVI_CHANNELS_DEFAULT = [94, 131, 171, 195, 284, 304]
SUVI_TO_AIA_CHANNEL_MAP = {94: 94, 131: 131, 171: 171, 304: 304, 195: 193, 284: 211}
AIA_TO_SUVI_CHANNEL_MAP = {aia: suvi for suvi, aia in SUVI_TO_AIA_CHANNEL_MAP.items()}
GOES_SATELLITE_NUMBER_DEFAULT = 18

# AIA channels Surya needs that have no SUVI equivalent at all (see
# docs/suvi_channel_mapping_preprocessing.md, section 2) -- not derivable from
# SUVI_TO_AIA_CHANNEL_MAP, must be requested explicitly.
AIA_EXTRA_CHANNELS = [335, 1600]

# --- HMI products (no SUVI equivalent -- HMI has no magnetograph counterpart on SUVI) ---

HMI_PHYSOBS_MAP = {"magnetogram": "los_magnetic_field", "dopplergram": "los_velocity"}
HMI_VECTOR_SERIES = "hmi.B_720s"
HMI_VECTOR_SEGMENTS = ["field", "inclination", "azimuth", "disambig"]
HMI_PRODUCTS_DEFAULT = ["magnetogram", "dopplergram", "vector"]

# --- GOES flare class arithmetic ---

GOES_LETTER_VALUES = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}

MANIFEST_FILENAME = "manifest.json"


def goes_to_numeric(goes_class: str) -> float:
    """Convert a GOES flare class string into a sortable numeric value.

    The mapping is letter_value + number/100, so classes compare correctly both
    across and within letters (e.g. "C9.0" < "M1.0", "M1.0" < "M2.0").

    Args:
        goes_class: A GOES class string, e.g. "M1.2", "X5.0", or a bare letter "C".

    Returns:
        A float suitable for sorting/thresholding. Returns 0.0 (with a logged
        warning) for an empty or unrecognized string instead of raising, since
        this is used inside pandas.apply over HEK rows that may contain malformed
        or missing classes.
    """
    if not goes_class:
        logger.warning("Empty GOES class string, treating as 0.0.")
        return 0.0
    letter = goes_class[0].upper()
    if letter not in GOES_LETTER_VALUES:
        logger.warning("Unrecognized GOES class letter %r in %r, treating as 0.0.", letter, goes_class)
        return 0.0
    try:
        number = float(goes_class[1:]) if len(goes_class) > 1 else 0.0
    except ValueError:
        logger.warning("Unparseable GOES class magnitude in %r, treating magnitude as 0.0.", goes_class)
        number = 0.0
    return GOES_LETTER_VALUES[letter] + number / 100.0


def parse_class_threshold(class_str: str) -> float:
    """Parse a user-supplied GOES class threshold into a numeric value.

    Accepts either a bare letter ("C") or a full class ("M1.0"). A bare letter
    resolves to that letter's integer value with no fractional component, i.e.
    the floor of that letter's range (so "C" as a threshold means "C0.0 or
    stronger").

    Args:
        class_str: A GOES class threshold string, e.g. "C" or "M1.0".

    Returns:
        A float comparable with goes_to_numeric()'s output.
    """
    normalized = class_str.strip().upper()
    if normalized in GOES_LETTER_VALUES:
        return float(GOES_LETTER_VALUES[normalized])
    return goes_to_numeric(normalized)


def get_timestamp(fits_file) -> str:
    """Read the DATE-OBS header value from a FITS file.

    Handles both compressed (AIA-style, DATE-OBS in HDU 1) and uncompressed
    (SUVI/HMI-style, DATE-OBS in HDU 0) FITS layouts by checking for a
    COMPRESSED_IMAGE extension name rather than assuming a fixed HDU index.

    Args:
        fits_file: Path to a FITS file (str or Path).

    Returns:
        The DATE-OBS header value as a string.
    """
    with fits.open(fits_file) as hdul:
        if "COMPRESSED_IMAGE" in [hdu.name for hdu in hdul]:
            return hdul[1].header["DATE-OBS"]
        return hdul[0].header["DATE-OBS"]


def timestamp_to_seconds(timestamp_str: str) -> float:
    """Convert an ISO-8601 timestamp string into Unix seconds (UTC).

    FITS DATE-OBS values are always UTC even when they carry no explicit
    'Z'/offset suffix. Uses pandas.Timestamp rather than stdlib
    datetime.fromisoformat(...).timestamp(), because the stdlib version
    treats a timezone-naive datetime as *local* system time -- on a non-UTC
    machine that silently shifts every comparison by the local UTC offset,
    which previously made every already-downloaded file compare as "missing"
    (masked before because Fido.fetch's own overwrite=False happened to skip
    same-named files anyway; became an active data-loss risk once this
    function's result was used to justify overwrite=True downloads).

    Args:
        timestamp_str: Timestamp string, e.g. a FITS DATE-OBS value
            ("2023-08-05T21:45:05.13" or "...Z").

    Returns:
        Seconds since the Unix epoch (UTC), as a float.
    """
    return pd.Timestamp(timestamp_str).timestamp()


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Log-scale and percentile-normalize an image array to uint8 for display.

    Matches the notebook's normalizar_debug: clip non-positive values, take
    log10, then linearly rescale the 2nd-98th percentile range to [0, 255].

    Args:
        img: 2D array of raw pixel values (any numeric dtype).

    Returns:
        A uint8 array of the same shape, in [0, 255].
    """
    img = np.asarray(img, dtype=np.float32).copy()
    img[img <= 0] = 1
    img_log = np.log10(img)
    vmin = np.nanpercentile(img_log, 2)
    vmax = np.nanpercentile(img_log, 98)
    img_norm = (img_log - vmin) / (vmax - vmin + 1e-10)
    img_norm = np.clip(img_norm, 0, 1) * 255
    return img_norm.astype(np.uint8)


def channel_dir_name(angstrom: int) -> str:
    """Format a wavelength in Angstrom as a zero-padded 3-digit directory name.

    Args:
        angstrom: Wavelength in Angstrom, e.g. 304.

    Returns:
        A 3-digit zero-padded string, e.g. "304", "094".
    """
    return f"{angstrom:03d}"


def format_ts_for_name(ts) -> str:
    """Format a timestamp for use in file/folder names.

    Uses a full timestamp (not just a date) since quiet-sun windows can be as
    short as one minute -- date-only names would collide across same-day runs.

    Args:
        ts: A datetime, pandas.Timestamp, or ISO-8601 string.

    Returns:
        A string of the form "YYYYMMDDTHHMMSS".
    """
    return pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")


def mode_folder_name(mode: str, start, end) -> str:
    """Build the download folder name for a given mode and time range.

    Args:
        mode: One of "quiet-sun", "flare-event", "indistinct".
        start: Range start (datetime, Timestamp, or ISO string).
        end: Range end (datetime, Timestamp, or ISO string).

    Returns:
        A folder name like "quiet-sun_20230315T100000_20230315T110000".
    """
    return f"{mode}_{format_ts_for_name(start)}_{format_ts_for_name(end)}"


def download_root(output_dir, mode: str, start, end) -> Path:
    """Build the full download-root path for a query.

    Args:
        output_dir: Root output directory (e.g. "data").
        mode: One of "quiet-sun", "flare-event", "indistinct".
        start: Range start.
        end: Range end.

    Returns:
        A Path such as data/quiet-sun_20230315T100000_20230315T110000.
    """
    return Path(output_dir) / mode_folder_name(mode, start, end)


def csv_paths(root: Path, mode: str, start, end) -> tuple[Path, Path]:
    """Build the (sdo_csv, suvi_csv) output paths for a query's dry-run listing.

    Args:
        root: The download-root directory (from download_root()).
        mode: One of "quiet-sun", "flare-event", "indistinct".
        start: Range start.
        end: Range end.

    Returns:
        A (sdo_csv_path, suvi_csv_path) tuple. Filenames match the spec exactly
        per mode (quiet-sun/flare-event keep the mode in the name; indistinct
        drops it).

    Raises:
        ValueError: If mode is not one of the three recognized values.
    """
    s, e = format_ts_for_name(start), format_ts_for_name(end)
    if mode == "quiet-sun":
        sdo_name, suvi_name = f"sdo_quiet-sun_{s}_{e}.csv", f"suvi_quiet-sun_{s}_{e}.csv"
    elif mode == "flare-event":
        sdo_name, suvi_name = f"sdo_flare-event_{s}_{e}.csv", f"suvi_flare-event_{s}_{e}.csv"
    elif mode == "indistinct":
        sdo_name, suvi_name = f"sdo_{s}_{e}.csv", f"suvi_{s}_{e}.csv"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return root / sdo_name, root / suvi_name


def video_path(videos_root: Path, mode: str, start, end) -> Path:
    """Build the output path for a generated video.

    Args:
        videos_root: The repo's videos/ directory.
        mode: One of "quiet-sun", "flare-event", "indistinct" (read from the
            download folder's manifest.json).
        start: The rendered window's start.
        end: The rendered window's end.

    Returns:
        A Path like videos/sdo-suvi-quiet-sun-20230315T100000_20230315T110000.mp4.
    """
    s, e = format_ts_for_name(start), format_ts_for_name(end)
    return Path(videos_root) / f"sdo-suvi-{mode}-{s}_{e}.mp4"


def write_manifest(root: Path, manifest: dict) -> None:
    """Write a manifest.json file into a download-root directory.

    Args:
        root: The download-root directory (created if missing).
        manifest: The manifest contents, JSON-serializable (datetimes are
            coerced to strings via json.dumps(default=str)).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_manifest(root: Path) -> dict:
    """Read a manifest.json file from a download-root directory.

    Args:
        root: The download-root directory.

    Returns:
        The parsed manifest contents.

    Raises:
        FileNotFoundError: If root does not contain a manifest.json.
    """
    with open(Path(root) / MANIFEST_FILENAME) as f:
        return json.load(f)


def hmi_products_from_manifest(manifest: dict) -> list[str]:
    """Reconstruct a --hmi-products list from a manifest's hmi_products entry.

    Used by --resume to rebuild the original CLI request from manifest.json
    instead of requiring the user to retype it.

    Args:
        manifest: A manifest.json dict, as loaded by load_manifest().

    Returns:
        A list of the HMI product names ("magnetogram", "dopplergram",
        "vector") that were requested, in that fixed order. Empty if none
        were requested.
    """
    hmi_products = manifest.get("hmi_products") or {}
    products = [name for name in ("magnetogram", "dopplergram") if hmi_products.get(name)]
    if hmi_products.get("vector"):
        products.append("vector")
    return products


def cv2_label_box(frame_bgr, text: str, origin: tuple = (10, 10), alpha: float = 0.6,
                   font_scale: float = 0.8, thickness: int = 2, padding: int = 8):
    """Draw a semi-transparent white label box with black text onto a BGR frame.

    Modifies frame_bgr in place (and returns it) so a single frame can receive
    multiple calls (e.g. one per panel of an hstacked video frame).

    Args:
        frame_bgr: A cv2/numpy BGR image array.
        text: The label text to draw.
        origin: (x, y) pixel coordinates of the box's top-left corner, local to
            this frame -- callers should apply this to each panel's own
            sub-array before hstacking, not to the combined frame, so the box
            always sits in that panel's corner.
        alpha: Opacity of the white box background (0-1).
        font_scale: cv2 font scale for the label text.
        thickness: Line thickness for both the box border-less fill and text.
        padding: Padding in pixels between the text and the box edges.

    Returns:
        The modified frame_bgr array (same object, mutated in place).
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    box_w = text_w + 2 * padding
    box_h = text_h + baseline + 2 * padding
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (255, 255, 255), thickness=-1)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
    text_origin = (x + padding, y + padding + text_h)
    cv2.putText(frame_bgr, text, text_origin, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return frame_bgr


def mpl_label_box(ax, text: str, loc: str = "upper left") -> None:
    """Draw a semi-transparent white label box with text onto a Matplotlib axis.

    Args:
        ax: A Matplotlib Axes instance.
        text: The label text to draw.
        loc: One of "upper left", "upper right", "lower left", "lower right".

    Raises:
        KeyError: If loc is not one of the four recognized corners.
    """
    positions = {
        "upper left": {"x": 0.02, "y": 0.98, "va": "top", "ha": "left"},
        "upper right": {"x": 0.98, "y": 0.98, "va": "top", "ha": "right"},
        "lower left": {"x": 0.02, "y": 0.02, "va": "bottom", "ha": "left"},
        "lower right": {"x": 0.98, "y": 0.02, "va": "bottom", "ha": "right"},
    }
    pos = positions[loc]
    ax.text(
        pos["x"], pos["y"], text, transform=ax.transAxes, va=pos["va"], ha=pos["ha"],
        fontsize=10, color="black",
        bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
    )


def nearest_indices(candidate_seconds: list[float], target_seconds: list[float]) -> tuple[list[int], float]:
    """Pick, for each target time, the candidate closest to it in time.

    Used to download only the AIA/HMI observation nearest to each SUVI image
    instead of every frame in a window.

    Args:
        candidate_seconds: Timestamps (Unix seconds) of the available rows, in
            table order. Need not be sorted.
        target_seconds: Timestamps (Unix seconds) to match against.

    Returns:
        A (indices, max_delta) tuple: the sorted, de-duplicated indices into
        candidate_seconds (several targets may share one candidate), and the
        largest absolute time difference in seconds between any target and its
        chosen candidate. Returns ([], 0.0) if either list is empty.
    """
    if not candidate_seconds or not target_seconds:
        return [], 0.0
    chosen = set()
    max_delta = 0.0
    for target in target_seconds:
        best = min(range(len(candidate_seconds)), key=lambda i: abs(candidate_seconds[i] - target))
        chosen.add(best)
        max_delta = max(max_delta, abs(candidate_seconds[best] - target))
    return sorted(chosen), max_delta
