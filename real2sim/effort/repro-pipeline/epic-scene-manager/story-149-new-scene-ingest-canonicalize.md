---
xid: STO-SCN-149
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-16
depends-on: [STO-SCN-146]
bd-id: krabby-n4bs
assignee: scout
---

# New Scene — ingest (video/images/folder, MOVE-or-UPLOAD) + canonicalize to content-hash

## Summary

The **New Scene** flow up to a canonicalized image set: create the scene (auto 3-digit code +
kebab name), pick **1 video / N images / a folder**, **MOVE or UPLOAD** the source, land it in the
right store path, then **canonicalize** to `images/<hash>/{image.jpg,metadata.json}` with progress.

## Context

Covers creation steps **1–8** of the EPI-SCN-SCENE-MANAGER flow. Canonicalize + capture-metadata
extraction already exist (capture-profile ingest lineage, `tests/test_capture_profile_ingest.py` +
`v4exec`); this story wires them behind the UI. Full spec: **EPI-SCN-SCENE-MANAGER § Creation flow**.

## Design / scope
1. **New Scene** → next free **3-digit code** `XXX`; type name → **kebab-case**.
2. **File picker** (local filesystem): 1 video OR N images OR a folder of homogeneous images
   (validate homogeneity; N videos = later).
3. **MOVE vs UPLOAD** toggle — local default **MOVE** (back-end move, no copy).
4. Land source: video → `scenes/XXX-<name>/videos/capture/video.<ext>`; images → `…/images/ingress/*`.
5. **Canonicalize** → `images/<hash>/image.jpg` + `images/<hash>/metadata.json`
   (images: move + extract metadata; video: extract frames + extract metadata). **Show progress.**
- New endpoints: create-scene, ingest (move/upload), canonicalize (progress via `/api/jobs/` style).

## Definition of Done
- [x] New Scene creates `scenes/XXX-<name>/` with the next free code + kebab name.
- [~] Picker handles 1 video / N images / a folder; MOVE (default local) or UPLOAD lands it correctly.
      — **server-side source path** (video / image / folder) + MOVE **or** COPY done; **browser UPLOAD deferred** (multipart) — see note.
- [x] Canonicalize produces `images/<hash>/{image.<ext>,metadata.json}` (from images AND from video frames) with live progress.
- [x] Reuses the existing canonicalize/metadata extraction (no re-implementation).
- [ ] **Operator-verified (T-020):** Scenes → + New Scene → name + a server path (folder or video) → Create+Ingest; watch progress; confirm the new scene appears + Metadata shows the canonical count.

## Build notes (2026-06-16)
- **Core module** `scene_ingest.py` (stdlib + ffmpeg, **no numpy**): `kebab`,
  `next_code`, `create_scene`, `canonicalize_file/_files` (reuse
  `v4core.file_hash` → `images/<hash>/{image.<ext>,metadata.json}` with
  `original_name`; **content-hash dedup**, idempotent), `extract_frames`
  (ffmpeg kernel of `extract_frames.sh` — not the legacy `/data`-hardcoded
  script), `ingest_images` (folder/N-images via `images/ingress/`),
  `ingest_video` (→ `videos/capture/video.<ext>`, frames @ fps, canonicalize),
  `ingest_path` (dispatch).
- **Endpoints** (`rate_renders/server.py`): `POST /api/scene-new` {name};
  `POST /api/scene/<scene>/ingest` {source, mode, fps} runs in a **background
  thread** writing `ingest_status.json`; `GET …/ingest-status` for polling;
  `GET /api/all-scenes` lists **every** scene dir (the Rank `/api/scenes` only
  lists render-bearing scenes, so a fresh scene was invisible — this was the
  one gap the e2e test caught).
- **Frontend** (`scenes.js`): New-Scene form (name → live create, source path,
  copy/move, fps) with a live progress bar polling `ingest-status`, then
  refreshes the selector (now `/api/all-scenes`) + lands on Metadata.
- **Verified:** `tests/test_scene_ingest.py` + driver + HTTP e2e on a temp
  store — image folder (3→2 dedup), **real ffmpeg video** (4 frames extracted,
  canonicalized, video moved to `videos/capture/`), both surfaced in
  `/api/all-scenes` with thumbs.
- **Deferred:** browser file **UPLOAD** (multipart) — v1 takes a server-side
  path, which fits "we operate locally → MOVE/COPY on the host".

## Out of scope
- The reconstruction pipeline (STO-SCN-150) — this story stops at a canonicalized image set.
- N-video capture.
