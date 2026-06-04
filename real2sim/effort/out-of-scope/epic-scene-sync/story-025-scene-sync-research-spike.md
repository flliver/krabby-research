---
xid: STO-SCN-025
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-04
depends-on: []
bd-id: krabby-0e1
title: Define what 'scene synchronization' means & scope the effort (AID working session)
assignee: principal
shipped: 2026-06-04
tasks: 6
complete: 6
---

# Define what "scene synchronization" means & scope the effort

## Summary

**A definition spike, done *with* the AID — and completed.** "Scene
synchronization" was an undefined term. The task was to sit with the AID, decide
what it actually means for this project, and turn it into a concrete, scoped
effort. **Outcome: it means the scene-data lifecycle** — organize, distribute,
consume-in-Docker, locally inspect, and S3-sync our M11 reconstruction data —
and that effort is now authored as `EPI-SCN-SCENE-SYNC` with a sized backlog.

## Context

The term arrived ambiguous. It could plausibly have meant either:
- **(geometry)** registering/fusing multiple submaps into one scale-consistent
  walkable environment (camera-spine registration, TSDF re-fusion, watertight
  merge), or
- **(data)** managing the *lifecycle* of scene datasets across S3, the fleet,
  containers, and laptops.

This story's job was to resolve that ambiguity with the AID rather than guess.
(An earlier draft of this story pre-committed to the *geometry* reading; that was
premature — the AID session below settled it the other way.)

## What we decided — "synchronization" = the scene-data lifecycle

In a working session with the AID (2026-06-04), the meaning was fixed:

> Take the ~50 GB of M11 scene-reconstruction data — today scattered and
> inconsistently laid out — and make it **organized, distributable to the fleet,
> consumable inside containers, locally inspectable, and synchronized with S3**,
> with the goals: minimize S3 push/pull, maximize LAN sync, keep S3 secrets out
> of code, and keep access simple for other engineers.

That definition is now the premise of `EPI-SCN-SCENE-SYNC`.

## What we accomplished

- **Defined the term** (above) and made it the epic's premise — disambiguated
  from the geometry reading.
- **Audited the real data:** ~50 GB, ~21 scenes, inconsistent per-scene layout;
  the existing `s3://krabby-real2sim-scenes` bucket (profile `krabby`).
- **Gathered live fleet facts** (baeprz via ops): no shared storage, abundant
  disk (`/games` 1.8 T on always-on `j`), 1 GbE flat LAN, only `rsync` installed,
  no S3 client/creds yet → `j` is the gateway/cache anchor; LAN-first wins on
  egress-dedup, not speed.
- **Cross-checked grants/contracts:** established the **prototype vs canon vs
  did-not-build** framing — all M11-to-date is prototype (redesignable); other
  delivered milestones are canon; third-party tool layouts (COLMAP/MAtCha/MASt3R/
  VGGT) are fixed and wrapped, not reorganized. Confirmed **data is not code**,
  the canon container contract (`code @ /workspace`, `data -v <host>:/data`), and
  that `krabby scenes` stays out of the lean `krabby-launcher`.
- **Authored the epic + schema:** `EPI-SCN-SCENE-SYNC` with the
  pipeline-of-transformations scene schema (`input/ → pipeline-<slug>/
  transform-NN/{specification.json, results.json, data/} → output/`), tiering
  (research/collab/public), dual provenance, maturity (prototype→promoted).
- **Produced a sized backlog:** `STO-SCN-026..033` (schema, migration+journal
  provenance, tiering, S3/secrets, sync CLI, fleet distribution, Docker consume,
  local inspection).
- **Made the in/out-of-scope call:** the *geometry* reading (submap fusion) is a
  **separate** concern, out of M11 scope, **deferred to M12+** — see below.

## Definition of Done

- [x] Worked with the AID to disambiguate "scene synchronization"; settled on the
      **data-lifecycle** meaning (not geometric submap-fusion).
- [x] Audited the real M11 data + existing S3 bucket; gathered live fleet facts.
- [x] Cross-checked grants/contracts (prototype/canon/did-not-build; data-not-code;
      container contract; CLI boundary).
- [x] Authored `EPI-SCN-SCENE-SYNC` with the pipeline-of-transformations schema +
      key decisions.
- [x] Proposed a sized backlog of follow-on stories (`STO-SCN-026..033`).
- [x] Recorded the explicit in/out-of-scope call for geometry vs data.

## Out of scope — the geometry reading (deferred to M12+)

Geometric submap-fusion (aligning/fusing multiple sub-scenes into one
scale-consistent walkable environment) is **not** what this effort is, and **not**
needed in M11 (every M11 capture is a single room fitting one MAtCha run). It is
deferred to M12+ / multi-submap captures. Prior art is preserved so it isn't lost
when the journal tree is archived:

- Journal: `threads/matcha-quality/notes/2026-05-01T174650-submap-based-mesh-fusion…`
  and `2026-05-04T120000-submap-fusion-strategy-detailed…` (submap-fusion /
  camera-"spine" strategy).
- `HUG-SCN-001` (TSDF ≫ tetra) and `STO-SCN-016` (scale-calibration ★) — the
  scale-drift + watertightness caveats any fusion scheme would inherit.
- Tooling in hand: MAtCha `extract_tsdf_mesh.py` (multires TSDF fusion), Open3D
  `ScalableTSDFVolume`, MASt3R-SfM as the cross-run pose source.

When geometric fusion is taken up, mint fresh stories under the owning milestone.

## Status notes

- 2026-06-04: Created as an open-ended "what must we build?" spike (pre-committed
  to the geometry reading).
- 2026-06-04: **Rewritten** after the AID working session — the task was to
  *define* "synchronization", and we did: it means the scene-data lifecycle. Premise,
  accomplishments, and DoD updated to reflect what we actually did; geometry reading
  recorded as out-of-scope/deferred. (Earlier mistake: I'd left the geometry framing
  stale and closed it as "superseded" — wrong; this is a *completed definition task*.)
