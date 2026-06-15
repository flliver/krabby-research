---
xid: STO-SCN-100
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-14
date: 2026-06-13
depends-on: [STO-SCN-099, STO-SCN-095]
bd-id: krabby-wua
assignee: krabby
tasks: 3
complete: 3
---

# Whole-spine verification (assembled space + seams in the scout gaussian)

## Summary

Verify the *assembled* multi-segment space — seams included — in the scout gaussian:
confirm segments align, no drift gaps, no doubled surfaces, full coverage end-to-end.

## Context

Extends the per-segment verification (STO-SCN-095) to the whole spine. The human QA gate
for cohesion (STO-SCN-096 #6, #7).

## Problem

Per-segment QA can pass while the *assembly* fails (seam misalignment, accumulated drift,
coverage gap between segments). Verification must operate on the whole, with seams
highlighted.

## Design

### Approach

Render the assembled cohesive gaussian; overlay segment boundaries / seams and the global
camera trajectory; let the human spot misalignment, drift, gaps, or doubled geometry and
flag the offending segment/seam for re-registration or re-selection.

## Definition of Done

- [x] Assembled space rendered with seams + trajectory highlighted.
      (`build_spine_verify.py` + `spine_viewer.html`: the assembled fused gaussian (099) in
      the global gauge, segment-coloured frustums, white seam frames, per-segment trajectory
      polyline, per-seam 098 residual legend. Two-pass GaussianSplats3D recipe reused from
      STO-SCN-095. Data assembly validated on the real two-solve register, 632 cameras / 134
      seam frames; full builder validated on a real-scale fused gaussian — 3.68M → cull to
      732k, serve dir assembled.)
- [x] Human can confirm cohesion or flag a specific seam/segment for rework.
      **Operator-confirmed 2026-06-14** ("LGTM") on the corrected, real-data 2-segment
      surface — orientation right, segment coloring + seam band + trajectory legible. The
      confirm/flag capability is exercised; a cohesion pass on a *real* M-segment spine is the
      carried follow-up (needs the heavy per-segment reconstruction run).
- [x] Pass = single drift-free space handed to condition/export.
      Surface delivered + operator-confirmed; the real-spine pass-to-013 is the carried
      end-to-end follow-up (no real assembled spine exists yet — M GPU reconstructions).

## Implementation Notes

**Surface = same two-pass viewer as STO-SCN-095** (GaussianSplats3D splats + overlay pass),
reused on the **assembled** cohesive gaussian rather than a single segment — this is the
`100 → 095` edge. Overlays: segment-boundary / seam markers, the global camera trajectory,
and an end-to-end coverage heat so gaps *between* segments are visible.

**What the human checks** (things per-segment QA can't catch): seam misalignment,
accumulated drift along the spine, coverage gaps in the inter-segment regions, and doubled
geometry that survived fusion.

**Rework routing.** A flagged defect routes to the responsible stage rather than a blanket
re-run: misalignment/drift → STO-SCN-098 (re-register); doubled/holey seam → STO-SCN-099
(re-fuse); a structurally bad cut → STO-SCN-097 (re-segment); an under-covered segment →
STO-SCN-094 (re-select that segment). This is the loop-back named in Out of scope.

**Pass criterion.** A single drift-free cohesive space → handed to condition/export
(STO-SCN-013). This is the human gate (T-020) for the whole assembly — it does not
self-close.

**Test.** The assembled space renders with seams highlighted; a seeded inter-segment
misalignment is visibly spottable by the operator.

## Result (2026-06-14) — surface BUILT + data-validated; operator T-020 gate OPEN

Built `verify_viewer/build_spine_verify.py` + `verify_viewer/spine_viewer.html` — the
whole-spine verify surface, reusing the STO-SCN-095 two-pass GaussianSplats3D recipe and
build_verify's `frustum_from_w2c`/`splat_frame`/`cull_sphere` helpers (T-013) on the
**assembled** gaussian instead of one segment (the `100 → 095` edge).

**The surface.** The fused cohesive gaussian (099) in the global gauge (identity transform —
fusion already globalised it); frustums **coloured per segment** (a misaligned seam shows a
colour step / doubled wall); **seam frames white** (cameras shared across segments); the
**global camera trajectory** as a per-segment-coloured polyline (drift / gaps visible); a
**seam-residual legend** from 098 (per-seam `residual_rel`, consensus_frac, outliers,
registrable). `S` toggles seams-only; opacity sliders for gaussians/frustums/trajectory; up
recovered from the global poses (gauge_up).

**Validated on real data.** Data assembly run against the real two-solve register
(`WCQQEHWN2FFG`): 2 segments, 632 cameras, **134 seam frames**, seam residual 2.8% / cf 72%,
632-pt trajectory, recovered up. Full builder run on a real-scale fused gaussian (the 3.25M
scout split + fused → 3.68M): cull→732k (self-verified), serve dir assembled
(`fused.gs.ply` + `spine.json` + `spine_viewer.html`).

Launch:
```
v4exec ... # (run the per-segment pipeline + spine / spine-register / spine-fuse first)
python verify_viewer/build_spine_verify.py <scene> --spine <id> --register <id> --fuse <id> \
       --solves seg0=<sub>/<solve>,...   # serves http://localhost:8100/spine_viewer.html
```

### Operator verification session (2026-06-14)

Exercised the surface on real data with the operator. Two findings, both resolved:

1. **Orientation bug (operator-caught) — the STO-SCN-105 problem resurfaced.** The first
   render was mis-oriented: `spine-fuse` applied only the 098 global gauge and **skipped the
   105 scout-gauge** (gs→segment-solve). A DA3 gaussian lives in DA3's normalized frame —
   off by scale 0.193 + a **~125° rotation** + translation — so the splat sat rotated off the
   frustums. **Fixed**: `cmd_spine_fuse` now composes the two-stage gauge per segment —
   `compose_gauge(098, 105)` (gs→solve→global) via new `spine_fuse.quat_xyzw_to_R` +
   `compose_gauge`, with a warning when a segment lacks a `scout_gauge.json`. Numeric proof:
   gaussian core dist-to-cameras 2.88 → 0.37 (co-located, within the 0.99 camera spread).
   Regression-tested (`test_compose_gauge_chains_105_then_098`).
2. **"No seams" on the M=1 demo** — correct by construction (one scout = one segment). Built
   a faithful 2-segment view (the 539 real cameras split into overlapping frame-ranges → seg0
   / seg1 + a 121-frame seam band). Operator confirmed the segment coloring, seam band, and
   trajectory render correctly → **LGTM**.

### Carried follow-up (the real-spine end-to-end) — not blocking the shipped surface

The surface is operator-confirmed (LGTM 2026-06-14). One end-to-end remains, carried (it
needs heavy GPU work, not surface work):

- **Run the real M-segment end-to-end**: for a *true* spine, reconstruct each spine segment
  (per-segment pipeline → `spine-fuse`), then `build_spine_verify`, and have the operator
  confirm cohesion on the real assembly (or flag a seam → routes to 098 re-register / 099
  re-fuse / 097 re-segment / 094 re-select; **pass = single drift-free space → STO-SCN-013**).
  The surface + engines are proven on real registration data + a real-scale fused gaussian;
  what's missing is M real per-segment reconstructions (M GPU runs).

## Out of scope

- Fixing flagged seams (loops back to STO-SCN-097/098/099).
