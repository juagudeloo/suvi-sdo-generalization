# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this repo is

**SUVI/SDO generalization experiment**: does NASA/IBM's Surya foundation model (frozen encoder,
pretrained on SDO/AIA+HMI) generalize to GOES-R SUVI observations? Developed with an external
collaborator working on SUVI, in the same framework as `../vision-language-model-surya` but
**not one of the PhD's core objectives** — it's a complementary side project.

**Priority note**: `vision-language-model-surya` (the VLM milestone) always takes priority over
this repo when both have pending work at the same time. Same deadlines, different order — see
`../CLAUDE.md` for the standing rule.

Split out from `vision-language-model-surya` on 21 Sep 2026 — that work started inside the VLM
repo's `notebook/` folder before it was clear this deserved its own repo. History before the split
lives only in the old location's git-untracked working tree (this repo starts with no prior git
history of its own).

## Contents

- `notebook/01-download_sdo_and_suvi.ipynb` — downloads AIA 304 Å + SUVI 304 Å around a real X1.6
  flare (2023-08-05), compares shapes/stats, builds a side-by-side video. Covers **one channel
  only** — not the 13-channel input Surya actually needs.
- `notebook/data/aia/`, `notebook/data/suvi/` — raw FITS from that notebook.
- `notebook/data/suvi_quietsun/` — separate download: all 6 SUVI channels (94/131/171/195/284/304
  Å), 1h window, quiet-sun (2023-03-15 10:00–11:04 UTC, verified flare-free via HEK/SWPC). This
  data exists but **the code that produced it was never captured in a notebook** — it ran directly
  in a terminal session. Worth writing up properly before relying on it further.
- `docs/suvi_channel_mapping_preprocessing.md` — channel-mapping table (AIA↔SUVI), resolution
  comparison (verified from real FITS headers, not estimated), and what preprocessing Surya's
  `StandardScaler`-based normalization would need before SUVI data could go through the encoder.
  **Key finding**: only 6 of Surya's 13 input channels have any SUVI equivalent (4 direct + 2
  approximate) — the other 7 (2 AIA + all 5 HMI) have no SUVI substitute at all, since GOES/SUVI
  carries no magnetograph. The generalization experiment can only ever test a 6-channel subset, not
  the full input.

## What's NOT done yet

The actual generalization test — running Surya's encoder on SUVI-mapped channels and seeing what
comes out — has never been attempted. Everything so far is data collection and scoping (what's
possible, what isn't). Do not assume this is closer to done than that.

## Constraints

Same as `vision-language-model-surya`: open-source models only in the runtime stack; Surya's own
constraints (frozen encoder, 4096×4096 native resolution, `nasa-ibm-ai4science/Surya-1.0` on
Hugging Face, code at `NASA-IMPACT/Surya`) apply here too since it's the same model.

## Commands

None yet — no environment file, no test suite. Reuses the `pytorch_jupyter` conda env
(`/homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter`) documented in
`../vision-language-model-surya/CLAUDE.md` and `../Surya`.
