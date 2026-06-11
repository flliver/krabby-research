---
xid: STO-SCN-038
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-09
depends-on: []
bd-id: krabby-zsw
title: Source-pin all transform tool code (no un-versioned snapshots)
priority: 1
assignee: krabby
shipped: 2026-06-11
tasks: 5
complete: 4
---

# Source-pin all transform tool code (no un-versioned snapshots)

## Summary
Every tool a transform runs (MAtCha, MASt3R, SLAM3R, VGGT, conditioning scripts) builds from **source-controlled, pinned code** — no bind-mounted host snapshots, no in-place patches outside version control.

## Context
2026-06-09 finding: the production MAtCha "source of truth" is `tbeeprz:~/scratch/MAtCha` — 653 code files, NOT a git repo, patches applied in place, bind-mounted over the image's empty `/opt/MAtCha`. Content fingerprint `106c21e725807af4103b094979fc7df4` (code-file tree hash). If that directory is lost, the production pipeline is unrecoverable. The images (`images/matcha/` etc.) already encode patches as scripts — the gap is the drifted live copy + the bind-mount practice.

## Definition of Done
- [x] Archive the tbeeprz `~/scratch/MAtCha` snapshot verbatim (tar + hash) to durable storage before anything else touches it
- [x] Diff snapshot vs `images/matcha/` Dockerfile+patches; fold any drift back into the patch scripts (version-controlled). 2026-06-09: drift = ONE file; root cause was a paren-matching bug in `patch_matcha_torch_load.py` (kwarg inserted inside nested `os.path.join`). Fixed scanner reproduces production exactly — except `2d-gaussian-splatting/extract_mesh.py`, where PRODUCTION carries the same latent bug un-fixed (line never executed); new patch output is the corrected form, deviation intentional.
- [x] Image build produces a complete, runnable `/opt/MAtCha` (0.2.x selfcontained; labels carry refs)
- [ ] Same audit for mast3r/slam3r/vggt images (sbeeprz/dbeeprz may hold similar snapshots — check)
- [x] `specification.json` git_sha populated from image label (runner does this; verified in 013/006 runs)

## Status notes

- 2026-06-09: **Snapshot archived.** `j:/games/krabby/archives/matcha-snapshot-20260609.tar.zst`
  (6.8 GB, sha256 e8b8ddaea30140f7274e2d580f2a4090b3f123f10d3c07a16d6250281e733db5,
  verified on j; SHA256SUMS alongside). Source tree code-md5 106c21e7… The remaining
  DoD items (diff vs image patches, image rebuild, sibling-host audit) stay open.

- 2026-06-09: krabby picked up per operator directive (build approved).
  Upstream base identified: b119fd96 (2025-04-07, repo dormant) — 849/850
  code files byte-identical to snapshot after fixed patches. Dockerfile
  pinned to that SHA + OCI labels (selfcontained/upstream_sha/patchset).
  `run_transform.py`: skips host-snapshot bind mount for self-contained
  images (label-driven), errors loudly when a legacy image has no
  snapshot, populates spec git_sha from the image label (DoD item 5
  mechanism). Image build `krabby-matcha:0.2-selfcontained` running on
  dbeeprz (clean-room — d never had the image or snapshot). Test run on
  d next; sibling-host audit (mast3r/slam3r/vggt snapshots) after.

- 2026-06-11: SHIPPED. Self-contained image landed (0.2.x line, labels
  carry upstream_sha+patchset), registry distribution live
  (j.pski.org:5000), git_sha populated from labels in production runs
  (013, 006-da3), krabby pipeline tools BAKED per the 2026-06-10
  tooling-provenance policy. Remaining tail — the mast3r/slam3r/vggt
  image audit — transfers to EPI-SCN-PIPELINE-STUDIO story 2
  (transform catalog declares image per task).
- 2026-06-11: Closed with --force; 1/5 DoD boxes unchecked. Reason: see story status notes 2026-06-11 closeout
