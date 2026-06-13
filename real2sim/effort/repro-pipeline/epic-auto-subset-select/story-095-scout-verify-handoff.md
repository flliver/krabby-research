---
xid: STO-SCN-095
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-13
depends-on: [STO-SCN-094]
bd-id: krabby-9qo
assignee: krabby
---

# Scout-gaussian verification surface + handoff to reconstruct graphs

## Summary

Render a scout gaussian of the scene, show the auto-proposed N cameras (and coverage
gaps) inside it for a human to accept / drop / add, then emit `FINAL N` for the existing
reconstruct graphs.

## Context

The single human gate (design story, conclusion #6) and the clean handoff. Consumes the
proposed-N from STO-SCN-094. Scout gaussian via DA3 feed-forward is proven feasible (32
views, 12.7 GB, ~32 s, native 3DGS); two-pass compositing of splat + camera frustums is
also proven (prototype in `/tmp/gsviewer`).

## Problem

Automated selection is good but not infallible (coverage gaps, an odd angle). The human
needs to *see* the proposal in the actual scene and override cheaply — then the result
must hand off unchanged to the reconstruct graphs (a frame-index list + poses).

## Design

### Approach

Build a scout gaussian (DA3) in the solve frame; render it with the proposed-N camera
frustums overlaid (two-pass splat + overlay, proven) and the coverage map. Human accepts /
drops / adds views. Output `FINAL N` = frame indices + poses, in the form the reconstruct
graphs already consume. The splat is the QA lens, not the selector.

### Changes

| File | Change |
|------|--------|
| scout-gaussian stage | DA3 feed-forward gaussian in solve frame |
| verify UI | splat + proposed-N frustums + coverage; accept/drop/add |
| handoff | emit `FINAL N` (frame-index list + poses) for reconstruct |
| seam handle | when part of a spine, also emit retained anchor frames + local poses (OUT side of the segment boundary contract, STO-SCN-096) for global registration |

## Definition of Done

- [ ] Scout gaussian renders with the proposed-N cameras + coverage gaps visible.
- [ ] Human can accept / drop / add views; result persists as `FINAL N`.
- [ ] `FINAL N` consumed unchanged by an existing reconstruct graph end-to-end.

## Implementation Notes

**Scout gaussian.** DA3 feed-forward (`infer_gs`) on the proposed-N in the solve frame —
proven feasible this session (32 views, 12.7 GB peak, ~32 s forward, ~3.6 M gaussians,
native 3DGS). If proposed-N exceeds the DA3 view ceiling, build the scout on a
coverage-representative 32-view subset (it's a QA lens, not the final reconstruction).

**Viewer (proven two-pass composite).** GaussianSplats3D (@mkkellogg). The working recipe
from the `/tmp/gsviewer` prototype: render **splats** via `viewer.render()` (autoclear-on)
and **overlays** (proposed-N frustums, coverage heat, gap markers) via
`renderer.render(overlayScene, camera)` (autoclear-off). The DropInViewer / embedded
non-self-driven paths did **not** composite, and self-driven Viewer + overlays-in-threeScene
dropped the splats — the two-pass split is the one that works. Don't re-derive this.

**Controls.** accept / drop / add. "Add" picks from the **posed pool** (not an arbitrary
point) so the added view already has a pose. Output `FINAL N` = frame-index list + poses in
the **exact schema the reconstruct graphs already consume** — the v4 `--sfm posed`
`posed.json` / scene `cameras.json` shape (so the handoff is unchanged; the splat changes
nothing downstream).

**Seam handle (spine OUT).** When part of a spine, also emit the retained anchor frames +
their local-gauge poses (the OUT side of the boundary contract) for global registration
(STO-SCN-098).

**Known tension.** A fisheye + sparse scout may need undistortion for clean splats — flagged
below as out of scope, but the risk surfaces here.

## Out of scope

- The reconstruct graphs themselves.
- Strong-fisheye undistortion (note tension if scout is fisheye + sparse).
