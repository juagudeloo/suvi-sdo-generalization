#!/usr/bin/env python3
"""visualization.py — Plot and animate matched SUVI/AIA observations.

Reads a download folder produced by suvi_sdo_query.py (channel-per-subfolder
layout, with manifest.json) and either:
  - plot:  render one SUVI/AIA pair side by side, optionally marking a flare peak.
  - video: render an interleaved SUVI/AIA video over the whole folder, a date
           subrange, or a duration before/after/around a given timestamp.

Metadata about flare peaks always comes from the download folder's manifest.json
-- this script never re-queries HEK itself.

Usage:
  python scripts/visualization.py --folder data/flare-event_..._... \\
      plot --file data/flare-event_..._.../suvi/304/some_file.fits --mark-flare-peak
  python scripts/visualization.py --folder data/quiet-sun_..._... video --channel 304
  python scripts/visualization.py --folder data/quiet-sun_..._... video --channel 304 \\
      --around 2023-03-15T10:22:00 --window 30min --direction both
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from suvi_sdo_common import (  # noqa: E402
    AIA_TO_SUVI_CHANNEL_MAP,
    SUVI_TO_AIA_CHANNEL_MAP,
    channel_dir_name,
    cv2_label_box,
    get_timestamp,
    load_manifest,
    mpl_label_box,
    normalize_for_display,
    timestamp_to_seconds,
    video_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------


def available_channels(folder: Path, instrument: str) -> list[int]:
    """List the channels (in that instrument's own numbering) present in a folder.

    Args:
        folder: A download-root folder (as produced by suvi_sdo_query.py).
        instrument: "aia" or "suvi".

    Returns:
        A sorted list of wavelengths in Angstrom.
    """
    instrument_dir = Path(folder) / instrument
    if not instrument_dir.is_dir():
        return []
    return sorted(int(p.name) for p in instrument_dir.iterdir() if p.is_dir())


def list_channel_files(folder: Path, instrument: str, channel_angstrom: int) -> list[Path]:
    """List the FITS files for one instrument/channel in a download folder.

    Args:
        folder: A download-root folder.
        instrument: "aia" or "suvi".
        channel_angstrom: Wavelength in Angstrom, in that instrument's numbering.

    Returns:
        A sorted list of FITS file paths (empty if the channel dir is missing).
    """
    channel_dir = Path(folder) / instrument / channel_dir_name(channel_angstrom)
    if not channel_dir.is_dir():
        return []
    return sorted(channel_dir.glob("*.fits"))


def indexed_timeline(folder: Path, instrument: str, channel_angstrom: int) -> list[dict]:
    """Build a chronologically sorted timeline of files for one instrument/channel.

    Args:
        folder: A download-root folder.
        instrument: "aia" or "suvi".
        channel_angstrom: Wavelength in Angstrom, in that instrument's numbering.

    Returns:
        A list of {"file": Path, "timestamp": str, "seconds": float} dicts,
        sorted by seconds.
    """
    entries = []
    for f in list_channel_files(folder, instrument, channel_angstrom):
        ts = get_timestamp(f)
        entries.append({"file": f, "timestamp": ts, "seconds": timestamp_to_seconds(ts)})
    return sorted(entries, key=lambda e: e["seconds"])


def closest_file(target_seconds: float, timeline: list[dict]) -> dict:
    """Find the timeline entry closest in time to a target.

    Args:
        target_seconds: Reference time, in Unix seconds.
        timeline: A non-empty list of entries from indexed_timeline().

    Returns:
        The entry whose "seconds" value is nearest to target_seconds.

    Raises:
        ValueError: If timeline is empty.
    """
    if not timeline:
        raise ValueError("Cannot find the closest file in an empty timeline.")
    return min(timeline, key=lambda e: abs(e["seconds"] - target_seconds))


def other_instrument_and_channel(instrument: str, channel_angstrom: int) -> tuple[str, int]:
    """Map an (instrument, channel) pair to its cross-instrument counterpart.

    Args:
        instrument: "aia" or "suvi".
        channel_angstrom: Wavelength in Angstrom, in that instrument's numbering.

    Returns:
        A (other_instrument, other_channel_angstrom) tuple.

    Raises:
        ValueError: If channel_angstrom has no mapped counterpart (see
            docs/suvi_channel_mapping_preprocessing.md -- AIA 335/1600 and every
            HMI channel have no SUVI equivalent at all, so they can never be
            paired for a side-by-side plot or video).
    """
    if instrument == "aia":
        if channel_angstrom not in AIA_TO_SUVI_CHANNEL_MAP:
            raise ValueError(
                f"AIA {channel_angstrom}A has no SUVI equivalent (see "
                "docs/suvi_channel_mapping_preprocessing.md) -- it cannot be paired for plot/video."
            )
        return "suvi", AIA_TO_SUVI_CHANNEL_MAP[channel_angstrom]
    if channel_angstrom not in SUVI_TO_AIA_CHANNEL_MAP:
        raise ValueError(f"SUVI {channel_angstrom}A has no AIA equivalent.")
    return "aia", SUVI_TO_AIA_CHANNEL_MAP[channel_angstrom]


def find_pair(folder: Path, file_path: Path) -> tuple[dict, dict, str, str]:
    """Find the closest cross-instrument file to a given reference file.

    Args:
        folder: The download-root folder containing file_path.
        file_path: A FITS file under folder/<instrument>/<channel>/.

    Returns:
        A (this_entry, other_entry, this_instrument, other_instrument) tuple,
        where each entry is a dict like indexed_timeline()'s output.

    Raises:
        ValueError: If file_path is not located under a recognized
            folder/<instrument>/<channel>/ layout, or the other instrument has
            no files for the mapped channel.
    """
    file_path = Path(file_path)
    channel_dir = file_path.parent
    instrument_dir = channel_dir.parent
    instrument = instrument_dir.name
    if instrument not in ("aia", "suvi"):
        raise ValueError(f"Could not determine instrument from path {file_path} (expected .../aia/<ch>/... or .../suvi/<ch>/...).")
    channel_angstrom = int(channel_dir.name)
    other_instrument, other_channel = other_instrument_and_channel(instrument, channel_angstrom)

    this_timestamp = get_timestamp(file_path)
    this_entry = {"file": file_path, "timestamp": this_timestamp, "seconds": timestamp_to_seconds(this_timestamp)}

    other_timeline = indexed_timeline(folder, other_instrument, other_channel)
    if not other_timeline:
        raise ValueError(f"No {other_instrument} files found for channel {other_channel}A in {folder}.")
    other_entry = closest_file(this_entry["seconds"], other_timeline)
    return this_entry, other_entry, instrument, other_instrument


# ---------------------------------------------------------------------------
# Flare-peak metadata (manifest.json only, never HEK)
# ---------------------------------------------------------------------------


def nearest_flare_event(manifest: dict, target_seconds: float) -> dict | None:
    """Find the flare event in a manifest closest to (or containing) a timestamp.

    Args:
        manifest: A manifest.json dict, as loaded by load_manifest().
        target_seconds: Reference time, in Unix seconds.

    Returns:
        The matching flare_events entry, or None if the manifest has no
        flare_events (i.e. it is not a flare-event download).
    """
    events = manifest.get("flare_events")
    if not events:
        return None
    for event in events:
        start = pd.Timestamp(event["start_time"]).timestamp()
        end = pd.Timestamp(event["end_time"]).timestamp()
        if start <= target_seconds <= end:
            return event
    return min(events, key=lambda e: abs(pd.Timestamp(e["peak_time"]).timestamp() - target_seconds))


# ---------------------------------------------------------------------------
# plot subcommand
# ---------------------------------------------------------------------------


def render_pair_plot(
    this_entry: dict, other_entry: dict, this_instrument: str, other_instrument: str,
    flare_event: dict | None = None, peak_tolerance_seconds: float = 360.0,
):
    """Render a side-by-side SUVI/AIA plot with no gap between panels.

    Args:
        this_entry: The reference file's timeline entry.
        other_entry: The matched cross-instrument file's timeline entry.
        this_instrument: "aia" or "suvi" (which instrument this_entry belongs to).
        other_instrument: The other of "aia"/"suvi".
        flare_event: A manifest flare_events entry to annotate, or None.
        peak_tolerance_seconds: If the SUVI frame's timestamp is within this
            many seconds of the flare's peak_time, highlight that panel's
            border to mark "this is (near) the peak frame".

    Returns:
        The Matplotlib Figure.
    """
    entries = {this_instrument: this_entry, other_instrument: other_entry}
    suvi_entry, aia_entry = entries["suvi"], entries["aia"]

    with fits.open(suvi_entry["file"]) as hdul:
        suvi_img = hdul[1].data
    with fits.open(aia_entry["file"]) as hdul:
        aia_img = hdul[1].data if "COMPRESSED_IMAGE" in [h.name for h in hdul] else hdul[0].data

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0}, sharey=True)
    for ax, img, label in ((axes[0], suvi_img, f"SUVI  {suvi_entry['timestamp']}"), (axes[1], aia_img, f"AIA  {aia_entry['timestamp']}")):
        ax.imshow(np.log10(np.clip(img.astype(np.float32), 1, None)), cmap="turbo")
        ax.set_xticks([])
        ax.set_yticks([])
        mpl_label_box(ax, label, loc="upper left")

    if flare_event is not None:
        summary = f"Flare peak: {flare_event['goes_class']} @ {flare_event['peak_time']}"
        mpl_label_box(axes[0], summary, loc="lower left")
        peak_seconds = pd.Timestamp(flare_event["peak_time"]).timestamp()
        if abs(suvi_entry["seconds"] - peak_seconds) <= peak_tolerance_seconds:
            for spine in axes[0].spines.values():
                spine.set_color("red")
                spine.set_linewidth(3)

    return fig


def run_plot(args: argparse.Namespace) -> None:
    """Handle the `plot` subcommand.

    Args:
        args: Parsed CLI arguments (folder, file, mark_flare_peak, output, show).
    """
    folder = Path(args.folder)
    this_entry, other_entry, this_instrument, other_instrument = find_pair(folder, args.file)

    flare_event = None
    if args.mark_flare_peak:
        manifest = load_manifest(folder)
        if manifest.get("mode") != "flare-event":
            logger.warning("--mark-flare-peak was given but %s is not a flare-event download; ignoring.", folder)
        else:
            flare_event = nearest_flare_event(manifest, this_entry["seconds"])

    fig = render_pair_plot(this_entry, other_entry, this_instrument, other_instrument, flare_event)

    output = Path(args.output) if args.output else folder / "plots" / f"{this_instrument}_{Path(args.file).stem}_pair.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, bbox_inches="tight")
    logger.info("Saved plot to %s.", output)
    if args.show:
        plt.show()


# ---------------------------------------------------------------------------
# video subcommand
# ---------------------------------------------------------------------------


def resolve_window(folder: Path, manifest: dict, args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve the (start, end) window to render a video for.

    Args:
        folder: The download-root folder (used for its manifest as a fallback).
        manifest: The folder's manifest.json contents.
        args: Parsed CLI arguments (start, end, around, window, direction).

    Returns:
        A (start, end) Timestamp tuple.
    """
    if args.start or args.end:
        return pd.Timestamp(args.start), pd.Timestamp(args.end)
    if args.around:
        anchor = pd.Timestamp(args.around)
        span = pd.Timedelta(args.window)
        if args.direction == "before":
            return anchor - span, anchor
        if args.direction == "after":
            return anchor, anchor + span
        return anchor - span, anchor + span
    return pd.Timestamp(manifest["requested_start"]), pd.Timestamp(manifest["requested_end"])


def build_video(
    folder: Path, suvi_channel: int, window_start: pd.Timestamp, window_end: pd.Timestamp,
    fps: int, codec: str, output: Path,
) -> None:
    """Render an interleaved SUVI/AIA video for one channel pair over a window.

    Reuses the notebook's sync strategy: AIA drives the frame loop; the SUVI
    frame is held until a later SUVI frame becomes closer in time to the
    current AIA frame.

    Args:
        folder: The download-root folder.
        suvi_channel: SUVI wavelength in Angstrom; the paired AIA channel is
            derived via SUVI_TO_AIA_CHANNEL_MAP.
        window_start: Start of the window to render.
        window_end: End of the window to render.
        fps: Output video frame rate.
        codec: cv2.VideoWriter fourcc codec (e.g. "mp4v").
        output: Output video file path (parent directories created as needed).

    Raises:
        ValueError: If either instrument has no frames within the window.
    """
    aia_channel = SUVI_TO_AIA_CHANNEL_MAP[suvi_channel]
    start_s, end_s = window_start.timestamp(), window_end.timestamp()
    suvi_data = [e for e in indexed_timeline(folder, "suvi", suvi_channel) if start_s <= e["seconds"] <= end_s]
    aia_data = [e for e in indexed_timeline(folder, "aia", aia_channel) if start_s <= e["seconds"] <= end_s]
    if not suvi_data or not aia_data:
        raise ValueError(f"No frames in the requested window for {folder} (suvi={len(suvi_data)}, aia={len(aia_data)}).")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(suvi_data[0]["file"]) as hdul:
        suvi_shape = hdul[1].data.shape
    frame_size = (suvi_shape[1] * 2, suvi_shape[0])
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, frame_size)

    suvi_idx = 0
    for i, aia_frame in enumerate(aia_data):
        aia_seconds = aia_frame["seconds"]
        while suvi_idx < len(suvi_data) - 1:
            current_gap = abs(aia_seconds - suvi_data[suvi_idx]["seconds"])
            next_gap = abs(aia_seconds - suvi_data[suvi_idx + 1]["seconds"])
            if next_gap < current_gap:
                suvi_idx += 1
            else:
                break
        suvi_frame = suvi_data[suvi_idx]

        with fits.open(suvi_frame["file"]) as hdul:
            suvi_img = hdul[1].data
        with fits.open(aia_frame["file"]) as hdul:
            aia_img = hdul[1].data if "COMPRESSED_IMAGE" in [h.name for h in hdul] else hdul[0].data

        suvi_gray = normalize_for_display(suvi_img)
        aia_gray = cv2.resize(normalize_for_display(aia_img), (suvi_shape[1], suvi_shape[0]))
        suvi_bgr = cv2.cvtColor(suvi_gray, cv2.COLOR_GRAY2BGR)
        aia_bgr = cv2.cvtColor(aia_gray, cv2.COLOR_GRAY2BGR)

        cv2_label_box(suvi_bgr, f"SUVI  {suvi_frame['timestamp']}")
        cv2_label_box(aia_bgr, f"AIA  {aia_frame['timestamp']}")

        writer.write(np.hstack([suvi_bgr, aia_bgr]))
        if (i + 1) % 25 == 0 or i == len(aia_data) - 1:
            logger.info("Rendered frame %d/%d.", i + 1, len(aia_data))

    writer.release()
    logger.info("Saved video to %s (%d frames).", output, len(aia_data))


def run_video(args: argparse.Namespace) -> None:
    """Handle the `video` subcommand.

    Args:
        args: Parsed CLI arguments (channel, start, end, around, window,
            direction, fps, codec, output).
    """
    folder = Path(args.folder)
    manifest = load_manifest(folder)

    channel = args.channel
    if channel is None:
        channels = available_channels(folder, "suvi")
        if len(channels) != 1:
            logger.error("Multiple channels available in this folder: %s. Specify --channel.", channels)
            sys.exit(2)
        channel = channels[0]

    window_start, window_end = resolve_window(folder, manifest, args)
    output = Path(args.output) if args.output else video_path(REPO_ROOT / "videos", manifest["mode"], window_start, window_end)
    build_video(folder, channel, window_start, window_end, args.fps, args.codec, output)


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
        description="Plot and animate matched SUVI/AIA observations from a suvi_sdo_query.py download folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--folder", required=True, help="A download folder produced by suvi_sdo_query.py (contains manifest.json).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_parser = subparsers.add_parser("plot", help="Plot one SUVI/AIA pair side by side.")
    plot_parser.add_argument("--file", required=True, help="A .fits file under --folder/aia/<ch>/ or --folder/suvi/<ch>/.")
    plot_parser.add_argument("--mark-flare-peak", action="store_true", help="Annotate with the nearest flare peak from manifest.json.")
    plot_parser.add_argument("--output", default=None, help="Output PNG path.")
    plot_parser.add_argument("--show", dest="show", action=argparse.BooleanOptionalAction, default=False, help="Also call plt.show().")

    video_parser = subparsers.add_parser("video", help="Render an interleaved SUVI/AIA video.")
    video_parser.add_argument("--channel", type=int, default=None, help="SUVI wavelength (Angstrom) to render; required if the folder has more than one.")
    window_group = video_parser.add_mutually_exclusive_group()
    window_group.add_argument("--start", default=None, help="Window start, ISO 8601. Use with --end.")
    window_group.add_argument("--around", default=None, help="Anchor timestamp for a duration-based window. Use with --window.")
    video_parser.add_argument("--end", default=None, help="Window end, ISO 8601. Use with --start.")
    video_parser.add_argument("--window", default=None, help="pandas.Timedelta string (e.g. 1h), required with --around.")
    video_parser.add_argument("--direction", choices=["before", "after", "both"], default="both", help="With --around: window before, after, or centered on the anchor.")
    video_parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    video_parser.add_argument("--codec", default="mp4v", help="cv2.VideoWriter fourcc codec.")
    video_parser.add_argument("--output", default=None, help="Override the output video path.")

    args = parser.parse_args(argv)
    if args.command == "video" and args.around and not args.window:
        parser.error("--around requires --window.")
    return args


def main() -> None:
    """Entry point: parse args and dispatch to the plot or video subcommand."""
    args = parse_args()
    if args.command == "plot":
        run_plot(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()
