# T0 — Scene Ingress & Creation

> Phase 1 of [the M11 process](README.md). Turn a real capture (video or photos) into a
> **content-addressed image pool** in the store, under a **declared camera profile**.
> Everything downstream keys off what you establish here.

## Inputs → Outputs

| In | Out |
|---|---|
| source video(s) or photo set + a one-time **capture declaration** | `scenes/<scene>/` with a content-addressed `images/<hash>/` pool + per-image metadata, and a `capture.json` that drives the solver/undistort choice |

## Step 1 — Declare the capture (`capture.json`)

The **only hand-authored file**. It lives at `scenes/<scene>/capture.json`:

```json
{ "make": "DJI", "model": "DJI Action 3", "mode": "fisheye", "modality": "hyperlapse" }
```

| Field | Values | Why it matters |
|---|---|---|
| `make` / `model` | EXIF strings | corroboration of the lens; **not** load-bearing on its own |
| `mode` | `fisheye` \| `dewarped` | **the distinguishing input that is NOT in EXIF.** `fisheye` → COLMAP/FastMap camera model `SIMPLE_RADIAL_FISHEYE`, undistorted to pinhole before solving. `dewarped` (in-camera) → **does not reconstruct in COLMAP under any model** → route to SLAM / feed-forward (HUG-SCN-004). |
| `modality` | `hyperlapse` \| `video` \| `photos` | picks the solver path (sequential/video vs sparse-photo) |

**Why declared, not inferred (T-010):** inferring distortion from scene edges was tried and
is unreliable on natural/foliage scenes — the verdict flipped between runs (STO-SCN-091/096
conclusion #3). The camera is a property of the *device + mode*, so declare it once.

## Step 2 — Ingest

```bash
v4exec ingest <scene> --host tbeeprz [--raw <dir>] [--capture-mode <mode>]
```

- Extracts frames (video) / imports photos → **content-addressed pool** `images/<hash>/`
  (each `image.*` + a per-image `metadata.json` with `original_name`, EXIF, etc.).
- Content-addressing means re-ingesting the same source is a NOOP — identity is the image hash.
- Frames are **never fps-subsampled at ingest** — thinning is a later, sharpness/coverage-aware
  decision (T1 pre-cull). Ingest preserves the full pool.

## Scene-folder structure (what you'll see in `scenes/<scene>/`)

| Path | What |
|---|---|
| `capture.json` | the declaration (above) |
| `scene.toml` | scene-level config |
| `images/<hash>/` | content-addressed source image pool (`image.*` + `metadata.json`) |
| `images/subsets/<sub>/` | curated/selected **subsets**; under each: `cameras/<solve>/` (T1 solve + covis + select + scout live here) |
| `represent/<model>/<id>/` | reconstructions (matcha / da3 / colmap / vggt / mast3r), each with `meshify/…/renders/…` |
| `spine/` | spine segmentation artifacts (T1, long pools) |
| `views/<slot>/` | the virtual **render cameras** (T2) |
| `jobs/` | per-run job records (host/GPU/duration/digest) |
| `scores.jsonl` | view rankings (T4) — **the data; commit it** |
| `videos/` | source video(s) |
| `_migrated` / `_unsorted` / `_migration-orphans` | legacy pre-v4 migration residue — ignore for new scenes |

## Gotchas

- **COLMAP version must match across hosts** (a 3.10 mapper can't read a 3.11.1 DB).
- DJI Action 3 in hyperlapse → `fisheye` + `SIMPLE_RADIAL_FISHEYE` (the proven 001-patio profile).
- The store is **v4 content-addressed; `v4exec` is the sole writer** — never hand-create files
  under `images/`/`represent/` (only `capture.json`/`scene.toml` are authored).

## Automation status

`v4exec ingest` is one command. The only manual act is writing `capture.json`. ✅ automated.

## Next

→ [T1 — Scouting & Spine](T1-scouting-spine.md)
