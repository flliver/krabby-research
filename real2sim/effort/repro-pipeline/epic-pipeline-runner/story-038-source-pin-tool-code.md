---
xid: STO-SCN-038
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-09
depends-on: []
bd-id: krabby-zsw
title: Source-pin all transform tool code (no un-versioned snapshots)
priority: 1
assignee: devex
---

# Source-pin all transform tool code (no un-versioned snapshots)

## Summary
Every tool a transform runs (MAtCha, MASt3R, SLAM3R, VGGT, conditioning scripts) builds from **source-controlled, pinned code** — no bind-mounted host snapshots, no in-place patches outside version control.

## Context
2026-06-09 finding: the production MAtCha "source of truth" is `tbeeprz:~/scratch/MAtCha` — 653 code files, NOT a git repo, patches applied in place, bind-mounted over the image's empty `/opt/MAtCha`. Content fingerprint `106c21e725807af4103b094979fc7df4` (code-file tree hash). If that directory is lost, the production pipeline is unrecoverable. The images (`images/matcha/` etc.) already encode patches as scripts — the gap is the drifted live copy + the bind-mount practice.

## Definition of Done
- [ ] Archive the tbeeprz `~/scratch/MAtCha` snapshot verbatim (tar + hash) to durable storage before anything else touches it
- [ ] Diff snapshot vs `images/matcha/` Dockerfile+patches; fold any drift back into the patch scripts (version-controlled)
- [ ] Image build produces a complete, runnable `/opt/MAtCha` (no runtime source mount needed); record source refs (upstream SHA + patch set) into image labels
- [ ] Same audit for mast3r/slam3r/vggt images (sbeeprz/dbeeprz may hold similar snapshots — check)
- [ ] `specification.json` `parameters.git_sha`/source-ref populated by the build, never null for new runs
