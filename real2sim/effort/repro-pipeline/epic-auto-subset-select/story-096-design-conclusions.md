---
xid: STO-SCN-096
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-15
date: 2026-06-13
depends-on: []
bd-id: krabby-1tr
assignee: krabby
tasks: 4
complete: 4
---

# Design: automated frame-subset selection — approach & conclusions

## Summary

The design record for EPI-SCN-AUTO-SUBSET-SELECT: the conclusions reached in the
2026-06-13 session on how to go, fully automatically, from a large image pool to the
best-N subset for high-quality reconstruction — with a single human verification step in
gaussian space.

## Context

The work began as "render a gaussian splat so we can position cameras and pick them by
hand." It reframed, correctly, to: **given a massive pool of images, what are the best N
to produce a high-quality mesh?** That is a *view-selection* problem, and the right place
to solve it is on the data the pose solve already produces. This story captures the
conclusions so they aren't lost in chat.

## The pipeline (conclusion)

```
video|images → EXTRACT (EXIF→camera model) → PRE-CULL (sharp+dedup, pose-free)
   → POSE pool (camera-model-correct solver) → co-visibility graph
   → AUTO-SELECT N (greedy coverage+connectivity)
   → SCOUT GAUSSIAN → ★ HUMAN VERIFY (accept/drop/add in splat)
   → FINAL N ═══► downstream RECONSTRUCT graphs
```

The entire left column is automated; the ★ is the only human touch; `FINAL N` is the
clean handoff the existing reconstruct graphs consume unchanged.

## Conclusions

1. **Small pool → use all; large pool → select.** The selection machinery only engages
   above the downstream model's view ceiling. Small curated sets (the ~12-photo scenes)
   already work by feeding all of them.

2. **"Best N" is a coverage-vs-redundancy optimization on the co-visibility graph.**
   Once views are posed, SfM hands you (for free) which images see which 3D points and
   how much each pair overlaps. Selection balances four things:
   - **Coverage** — every surface seen by enough views (≥3 for triangulation).
   - **Triangulation quality** — intersection angles ~10–30° (baseline wide enough for
     depth, not so wide matching fails).
   - **Connectivity** — the selected view-graph stays connected with sufficient pairwise
     overlap, or registration/fusion breaks (this is exactly what the 300-frame drift
     violated).
   - **Quality minus redundancy** — sharp, well-exposed; drop near-duplicates.
   The method is **greedy submodular coverage maximization**: repeatedly add the view
   with the largest marginal coverage+triangulation gain, subject to connectivity, until
   N (the model's sweet spot) or coverage saturates.

3. **Camera model comes from metadata / capture profile, NOT from pixel inference.**
   A DJI in fisheye mode is fisheye regardless of scene. Inferring distortion from
   scene edges was tried and is unreliable on natural/foliage scenes (the verdict flipped
   between runs). EXIF + known capture mode → camera model: **fisheye → `SIMPLE_RADIAL_FISHEYE`**;
   **dewarped → COLMAP-incompatible under *any* model** (reconciled 2026-06-13 to HUG-SCN-004's
   verdict; in-camera-dewarped footage does not reconstruct in COLMAP — route it to SLAM /
   feed-forward, not OPENCV). The distinguishing input (fisheye vs dewarped) is **not in EXIF**
   and must be declared per scene; EXIF (make/model) is corroboration only. Implemented in
   STO-SCN-091 (`capture_profiles.json` + `capture_profile.py`).

4. **Solver must match the modality.** Sparse-view retrieval methods (MASt3R-SfM/DUSt3R)
   drift on dense video — they have no temporal prior (the 001 nebula: a near-spherical
   camera blob where a handheld walk must be ~coplanar). Dense/video wants sequential or
   feed-forward solving; sparse photo sets want the sparse-view methods. (Landscape captured
   in the research corpus: COLMAP-sequential, GLOMAP, FastMap (GPU), VGGT/DA3 feed-forward,
   video SLAM.)

5. **Physical capture priors are cheap validity gates.** A handheld ground walk ⇒
   camera centers near-coplanar. The ratio of out-of-plane to in-plane spread flags a
   failed solve before any downstream work (the drift solve scored ~60%; a good one ~a few %).

6. **The gaussian splat is the verification surface, not the selector.** You don't need
   the splat to choose — the co-visibility graph quantifies coverage/overlap, so selection
   is automated. The splat is where a human *sees* the proposed cameras and coverage gaps
   and overrides ("nothing covers the far corner — add a view"). Scout gaussian via DA3
   feed-forward is proven feasible (32 views, 12.7 GB, ~32 s, native 3DGS).

7. **(Longer-term) "Best-N" generalizes to "best-N along a spine of M segments."** A
   single video is too big to pose/reconstruct at once, and the ultimate goal is one
   *cohesive* space — possibly many segments along the video's trajectory ("spine"). That
   reframes this from "pick N from a bag" into a **submap / pose-graph (SLAM-shaped)**
   problem. Consequences:
   - **Selection is not purely local** — each segment's edge views must stay co-visible
     with its neighbors (an *boundary-overlap budget*) so the M sub-reconstructions
     register. The co-visibility graph must remain connected *across* segment seams.
   - **Posing splits into local + global** — per-segment local solves, then global
     registration (pose-graph optimization, loop closure, global BA). M locally-good
     segments that aren't globally consistent are still M disjoint reconstructions; drift
     accumulates along the spine and must be corrected globally.
   - **A new stage exists: seam management + cohesive fusion** of submaps into one gauge.
   - **Solver tilts to sequential/SLAM-with-submaps** (purpose-built); FastMap/feed-forward
     become per-segment workers under a global pose graph.
   - **Fits DAG-of-dags**: each segment is a sub-graph; a *spine graph* composes the M with
     registration edges + loop closures. The per-segment unit (this epic) stays valid; the
     spine is the composing layer — scaffolded as a sibling epic.

## Validation update (2026-06-13) — solver path proven end-to-end

Conclusions #3/#4 are now empirically settled (data + detail in STO-SCN-093):
- **Solver = FastMap** (GPU SfM, deployed STO-SCN-101), **fisheye undistorted to pinhole
  first** (FastMap takes only PINHOLE/SIMPLE_RADIAL) using the per-camera calibration
  (STO-SCN-102, RMS 0.86 px).
- **Refinement to conclusion #1 for hyperlapse:** "small pool → use all" generalizes — the
  undistort FOV-crop costs overlap at *sparse* baselines (300→227 registered), but the
  **full blur/dup-culled pool registers completely** (539/539). So for hyperlapse, **keep
  the full culled pool, don't thin** — FastMap is GPU-scalable; the 300 ceiling was a
  mast3r-sfm artifact. The pre-cull (092) must also **order by capture time**, not store
  hash (403 vs 4 near-dups found).

## Segment boundary contract (the ×M interface)

The per-segment epic stays **spine-agnostic**; its only coupling to the spine is a thin
contract at the segment boundary. This is the semantic content of the cross-epic ×M
dependency edges — a segment never "knows" it's in a spine, it just honors an optional
boundary spec and emits a seam handle.

**IN (optional; injected by the spine layer; empty for M=1):**
- `boundary_spec` — the anchor frames this segment shares with each neighbor that **must
  be retained**, plus the overlap region(s) to keep covered. Consumed by the selector
  (STO-SCN-094) as the **boundary-overlap budget**. Empty ⇒ plain standalone best-N.
- `camera_model` — the one global camera model (STO-SCN-091), computed once, identical
  for every segment (it's a property of the camera, not the segment).

**OUT (the seam handle the spine consumes):**
- the normal segment output (local-gauge poses + reconstruction), **plus**
- the retained anchor frames with their local-gauge poses — the handle the global
  registration (STO-SCN-098) uses to compute the relative transform between neighbors
  (rotation, translation, **and scale** — resolving each segment's arbitrary SfM gauge)
  and stitch the global pose graph.

**Edge ↔ contract mapping:** `097→091,092` (segmentation needs ingest) · `098→093`
(×M poses in) · `099→095` (×M reconstructions in) · `100→095` (scout surface). At **M=1**
the boundary_spec is empty and there is no registration — the segment *is* the whole scene.

## Downstream boundary: where the spine hands off to mesh-conditioning (reconciled 2026-06-13)

These two repro-pipeline epics produce **one cohesive geometry**; they do **not** own the
terminal preparation for physics/USD. That arc already exists as the legacy
`effort/condition-usd/` epics (re-imported 2026-06-03):

```
… 098 → 099 (fuse) → 013 (condition) → 014 (verify watertight) → 015 (smooth)
                          └────────────► EPI-SCN-MESH-CONDITION ─► EPI-SCN-USD-EXPORT
```

The boundary is a single producer→consumer edge, **STO-SCN-099 → STO-SCN-013**:

- **STO-SCN-099 (spine, fuse) owns inter-segment seam fusion** — dedup of doubled walls /
  overlap blending *after* global registration. Its DoD already promises geometry
  "consumable by downstream condition/export."
- **STO-SCN-013 (mesh-condition) owns post-fusion conditioning** — make the single fused
  mesh manifold + watertight + gap-filled prior to physics. It does **not** re-do seam
  fusion.
- **M=1 degenerate:** STO-SCN-099 is a pass-through (nothing to register/fuse), and 013
  conditions the lone reconstruction directly — so the same edge holds whether the scene
  is one space or a spine.

This corrects a pre-existing overlap: STO-SCN-013's original journal (Jun-3) *anticipated*
the spine seam-fusion before the spine epic existed; that responsibility now lives in
STO-SCN-099, and 013 was retargeted + given `depends-on: STO-SCN-099`.

## Definition of Done

- [x] Approach, the pipeline graph, and the six conclusions recorded here.
- [x] Segment boundary contract (IN/OUT) recorded — the meaning of the ×M edges.
- [x] Downstream boundary to mesh-conditioning/USD-export recorded (099 → 013 edge).
- [x] Operator concurrence on the approach (this story is the record of it).
      **Concurred 2026-06-15**: the operator drove the full v4 build of this design
      (091–095 + 103 selector + 097–100 spine + 105 scout-gauge), exercised the verify
      surfaces, and directed the v4-pipeline documentation into `RECIPES.md` § "v4 pipeline".
      The approach is validated and shipped end-to-end.

## Out of scope

- Implementation of any stage (those are STO-SCN-091..095).
- Reconstruct-graph internals.

## Implementation Notes

Distilled from the 2026-06-13 design session. The lens-from-edges tool built mid-session
was discarded as unreliable (conclusion #3); the broadly-useful generic landscape was
contributed to the OLAI research corpus (`3d-reconstruction/pose-solver-landscape`,
proposal `kp-20260613-88fb`).
