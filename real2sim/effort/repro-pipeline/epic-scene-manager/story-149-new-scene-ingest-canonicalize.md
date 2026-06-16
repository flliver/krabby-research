---
xid: STO-SCN-149
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
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
- [ ] New Scene creates `scenes/XXX-<name>/` with the next free code + kebab name.
- [ ] Picker handles 1 video / N images / a folder; MOVE (default local) or UPLOAD lands it correctly.
- [ ] Canonicalize produces `images/<hash>/{image.jpg,metadata.json}` (from images AND from video frames) with live progress.
- [ ] Reuses the existing canonicalize/metadata extraction (no re-implementation).

## Out of scope
- The reconstruction pipeline (STO-SCN-150) — this story stops at a canonicalized image set.
- N-video capture.
