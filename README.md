# suvi-sdo-generalization

Does NASA/IBM's [Surya](https://arxiv.org/abs/2508.14112) foundation model (frozen encoder,
pretrained on SDO/AIA+HMI) generalize to GOES-R SUVI observations? Only 6 of Surya's 13 input
channels have any SUVI equivalent (4 direct + 2 approximate) — see
[`docs/suvi_channel_mapping_preprocessing.md`](docs/suvi_channel_mapping_preprocessing.md) for the
full channel-by-channel breakdown.

## Quickstart

```bash
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter
cd /scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization

# Dry run: list what's available for a quiet-sun window, no download
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
    --mode quiet-sun --duration 1h

# Real download: SUVI in full + only the AIA/HMI frame nearest in time to each SUVI
# observation (recommended -- AIA's native cadence is ~300 frames/hour/channel, JSOC
# export is ~2 min/file serial, and the servers are flaky under load)
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
    --mode quiet-sun --duration 1h --all-sdo-channels --match-suvi --download \
    --jsoc-email your_registered_email@example.com
```

Long downloads should go through SLURM instead of an interactive session (AIA exports can take
hours): `sbatch tools/download.sh <same flags>`. Full flag reference, retry/resume behavior, and
worked examples: **[`docs/suvi_sdo_query_usage.md`](docs/suvi_sdo_query_usage.md)**.

**Visualizing an existing download** (SUVI/AIA pair side by side, or an interleaved video):

```bash
FOLDER=data/matched/quiet-sun_20230315T090000_20230315T150000

python scripts/visualization.py --folder $FOLDER \
    plot --file $FOLDER/suvi/304/dr_suvi-l2-ci304_g18_s20230315T092000Z_e20230315T092400Z_v1-0-2.fits

python scripts/visualization.py --folder $FOLDER video --channel 304
```

Actually run against the data already in this repo. The AIA panel comes out sharp; the SUVI panel
comes out grainy even after fixing the normalization (see commit `2cf9082`) — confirmed across all
6 downloaded SUVI channels, not a one-off corrupt file or a reading bug (checked independently with
`astropy` and `sunpy.map.Map`, same values both ways). **Confirmed abnormal by the SUVI
collaborator (30 Sep 2026)** — this is not expected SUVI behavior. Frames this noisy can't be
processed by Surya; a quality filter to detect and exclude them is needed before the generalization
test can run for real (see `docs/suvi_channel_mapping_preprocessing.md` and the Notion task tracking
this). Full reference: **[`docs/visualization_usage.md`](docs/visualization_usage.md)**.

## Contents

| Path | What it is |
|---|---|
| [`scripts/suvi_sdo_query.py`](scripts/suvi_sdo_query.py) | Main tool. Searches and downloads matched SUVI + SDO (AIA/HMI) observations for a time window, in `quiet-sun`, `flare-event`, or `indistinct` mode. Without `--download` it only lists availability. Writes a `manifest.json` per download (what was requested, what was verified on disk, `status: complete/incomplete`) that allows resuming with `--resume` without remembering the original command. |
| [`scripts/visualization.py`](scripts/visualization.py) | Plots or animates SUVI/AIA pairs from an already-downloaded folder, reading `manifest.json` (never re-queries HEK). See [`docs/visualization_usage.md`](docs/visualization_usage.md). |
| [`utils/suvi_sdo_common.py`](utils/suvi_sdo_common.py) | Helpers shared by the two scripts above: channel mapping, FITS timestamps, folder/file naming, manifest read/write. |
| [`tools/download.sh`](tools/download.sh) | Launches `suvi_sdo_query.py --download` as a SLURM job (node `maxwell`, partition `gpu.cecc`, no GPU). Takes the same flags as the script, as-is. |
| [`docs/suvi_channel_mapping_preprocessing.md`](docs/suvi_channel_mapping_preprocessing.md) | Which Surya channels have a SUVI equivalent, a resolution comparison, and what preprocessing would be needed before feeding SUVI data through the encoder. |
| [`notebook/01-download_sdo_and_suvi.ipynb`](notebook/01-download_sdo_and_suvi.ipynb) | Original exploratory notebook (single channel, single event) — superseded in practice by `suvi_sdo_query.py`, kept for reference. |

`data/` and `logs/` are in `.gitignore` — downloaded data and SLURM logs don't live in the repo.

## Current status

Download tool functional and verified (retries, resume, honest incomplete-status reporting).
**Not done yet:** running Surya's encoder on mapped SUVI data — everything so far is download and
feasibility analysis, not the generalization experiment itself.
