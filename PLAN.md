# M11 Plan

**Status:** active working plan
**Last updated:** 2026-05-06
**Goal:** ship M11 (Patina Foundation grant — Krabby-Uno Milestone 11) ASAP, with reliable watertight reconstruction, then locomotion-model integration

---

## Tracking

| Layer | Where | Notes |
|---|---|---|
| **Authoritative scope** | [Patina M11 OVERVIEW.md](https://github.com/flliver/patina-foundation-grants/blob/main/grants/Krabby-Uno/Milestone11-Scene-Reconstruction/OVERVIEW.md) | Tasks T0–T4 + acceptance criteria. The ICA defers technical scope here. |
| **Commercial terms (ICA)** | `krabby-contracts/milestones/M11/M11.md` | Acceptance is judged against the grant overview, not this plan. |
| **Issue/task tracking** | `milestones/011-scene-reconstruction/beads/` (this dir's sibling) | Beads DB; prefix `m11-*`. Run `bd ready` to see unblocked work. Run `bd list --status open` to see all open beads. JSONL export auto-syncs at `beads/.beads/issues.jsonl`. |
| **Working memory** | `milestones/011-scene-reconstruction/journal/` | Working notes; not authoritative. |
| **This plan** | this file | High-level phase structure; bridge between grant and beads. Acceptance is *not* judged against PLAN.md. |

### Grant-task ↔ PLAN-phase ↔ Beads mapping

| Grant Task | PLAN Phases | Open beads | Closed beads |
|---|---|---|---|
| T0 — Sparse reconstruction | A (eval), B5 (frame curation) | — (rollup `m11-di0` ✓) | `m11-nk8`, `m11-2bg`, `m11-di0` |
| T1 — Dense + watertight mesh | A (initial tetra), B1–B4 (post-processing), C (validation, optional) | C2 `m11-2cd`, C3 `m11-8u6` | A `m11-23n`, B1–B4 (`5wp`, `1oc`, `8wy`, `pbo`), C `ep4`+`s0h`+`5ef`, T1 rollup `dua` |
| T2 — Conditioning + USD + IsaacSim | D (conditioning), E (USD + IsaacSim) | D1 `m11-21m`, D2 `m11-400`, D3 `m11-87v`, **E1 `m11-u3l` (P0)**, **E2 `m11-uy5` (P0)**, **E3 `m11-dt8` (P0)** | — |
| T3 — EP + Holosoma in Docker | F1, F2 | F1 `m11-bz6`, F2 `m11-x6i` | — |
| T4 — Hexapod + stable demo | F3, F4 | F3 `m11-1ij`, F4 `m11-4tl` (the deliverable) | — |
| (cross-cutting risks) | — | **R1 `m11-d06` (P0 schedule)**, R2 `m11-kr9` (T0/T1 disclosure) | — |

**Phase G (post-M11 stretch — submap fusion)** is intentionally not in beads; revisit on M12+ scoping.

### Quick commands

```bash
cd milestones/011-scene-reconstruction/beads
bd ready             # unblocked work (sorted by priority)
bd list              # all open beads
bd list --status closed
bd show m11-u3l      # show a specific bead
bd close m11-XXX --reason "..."   # close with provenance
```

---

## M11 scope (per `grants/grants/Krabby-Uno/Milestone11-Scene-Reconstruction/OVERVIEW.md`)

The full M11 is **all of T0–T4**, not just mesh production:

| Task | Deliverable | Status |
|---|---|---|
| T0 | Sparse reconstruction (camera poses + sparse cloud), 2–3 scenes | ✅ Done — Robust MASt3R-SfM pipeline scales to 350+ frames; poses generated for scene 004 and bicycle. COLMAP conversion supported. |
| T1 | Dense reconstruction + **watertight triangle mesh**, each scene | ✅ Done — Pivoted to TSDF-based meshes, which are vastly superior. Multiple high-quality variants produced for scene 004 and bicycle. |
| T2 | Mesh conditioning + USD export + load in IsaacSim, robot spawns + depth-sensor returns plausible readings | 🟡 In Progress — Full post-processing pipeline (orient, cull, color, cameras) is complete and automated. USD export + IsaacSim load is the remaining step. |
| T3 | Extreme Parkour + Holosoma in Docker, both consuming the same USD envs | ❌ Not Started |
| T4 | Both models adapted to hexapod, stable locomotion demo on at least one scene | ❌ Not Started |

**The grant assumes COLMAP MVS + Poisson** as the canonical T0/T1 pipeline (with SLAM3R/MASt3R-SLAM/Spann3R explicitly listed in Appendix A as alternatives). **MAtCha is not in the grant text.** We chose it because it produces watertight meshes natively, satisfying the T1 acceptance criterion, just via a different tool. This is a defensible substitution but worth flagging when reporting back to the grant.

---

## Status snapshot (today)

**What we have:**
- An end-to-end, reproducible pipeline for turning raw video into high-quality, post-processed 3D scenes.
- A major quality breakthrough by switching from tetrahedral to **TSDF meshes**, which are visually superior.
- Multiple high-quality TSDF meshes for "scene 004" and the new "bicycle" benchmark scene.
- A full **post-processing pipeline (Phase B)** that automatically handles gravity alignment, background culling, vertex coloring, and camera visualization.
- A powerful **interactive 3D camera selection tool** (`viewer.py`) for curating the best viewpoints from a large pool of candidate frames.
- A scalable MASt3R-SfM pipeline that can generate poses for 350+ frames.

**What's missing for M11 acceptance:**
- The final step of T2: USD export and IsaacSim integration.
- All of T3 (Extreme Parkour + Holosoma Docker integration).
- All of T4 (hexapod adaptation + locomotion demo).

**Risks:**
- IsaacSim integration is the largest unknown — we haven't validated that our TSDF meshes work as collision geometry yet. May require V-HACD.
- Scale calibration is unsolved across all our captures (no reference objects in the videos). This must be addressed in Phase C.
- T3 and T4 are substantial Docker/RL work, not just mesh-pipeline tuning.

---

## Work plan (priority order)

### Phase A — Pipeline Evaluation & Initial Mesh Generation
**Status: ✅ Done.** This foundational phase involved a comprehensive evaluation of multiple 3D reconstruction pipelines (including COLMAP, MASt3R-SLAM, and MAtCha). MAtCha was selected as the primary pipeline as it was the only candidate that reliably produced watertight meshes end-to-end, a key grant requirement. We successfully generated initial tetrahedral meshes for three distinct scenes (001-patio, 003-firepit, and 004-sky-house). The most critical outcome of this phase was the discovery that all generated meshes, regardless of the scene, suffered from a consistent set of post-processing issues (e.g., no ground plane, incorrect orientation, background noise), which directly motivated the creation of the Phase B post-processing pipeline.

### Phase B — Post-processing pipeline
**Status: ✅ Done.** The tooling for B1-B5 is complete and has been used to process the new TSDF meshes. The individual components included:
- [x] **B1 — Auto-deduce ground plane:** RANSAC-based plane detection to orient the mesh to a Z-up coordinate system with the floor at z=0.
- [x] **B2 — Auto-cull "out-of-bounds" geometry:** Removing distant, irrelevant geometry captured by the camera.
- [x] **B3 — Auto-include camera locations in mesh:** Adding debug markers for interpretability in Blender.
- [x] **B4 — Auto-project color:** Projecting vertex colors from source frames onto the mesh.
- [x] **B5 — Frame-selection tooling:** An interactive 3D viewer for manually curating the best frames for reconstruction.

### Phase C — MAtCha Output Validation ★ Current Focus

**Goal:** Before proceeding, we must validate that our pipeline can reproduce the reference quality for both TSDF and Adaptive Tetrahedralization meshes as shown in the official MAtCha paper. This ensures we are using the best possible output from the core algorithm.

| # | Task | Notes |
|---|---|---|
| C1 | - [ ] **Reproduce Reference Perspectives** | In our "bicycle" scene Blender file, create camera views that precisely match the two reference images from the MAtCha project page (saved in `reference_images/`). This will provide a direct, apples-to-apples comparison. |
| C2 | - [ ] **Validate TSDF Mesh Quality** | Render our best TSDF mesh from the reference perspective. The goal is to confirm we have already matched the quality of reference (a). We believe this is complete, but this step will formally verify it. |
| C3 | - [ ] **Match Adaptive Tetrahedralization Quality** | This is a critical gap. We must experiment with MAtCha's parameters (e.g., alignment configs, regularization) to produce a tetrahedral mesh that matches the quality of reference (b). This may involve revisiting the "tetra-era" experiments. |

**Phase C exit criterion:** We have generated tetrahedral and TSDF meshes for the bicycle scene that visually match the quality and detail of the official MAtCha reference images when viewed from the same perspective.

### Phase D — Final Mesh Conditioning for Simulation

**Goal:** Transform the validated, high-quality meshes into simulation-ready assets with clean, manifold, and watertight collision geometry. This is a critical prerequisite for IsaacSim.

| # | Task | Notes |
|---|---|---|
| D1 | - [ ] **Merge & Gap-Fill Surfaces** | Use surface reconstruction techniques (e.g., continuing with TSDF fusion, or Poisson reconstruction on the output) to resolve conflicts, merge nearby surfaces, and fill any remaining holes or gaps to ensure the mesh is manifold. |
| D2 | - [ ] **Ensure Watertightness** | Verify that the output from D1 is fully watertight, as required by physics simulators. |
| D3 | - [ ] **Final Surface Smoothing** | Apply a final smoothing pass to the geometry, with a preference for **Taubin smoothing** to minimize surface shrinkage. |

**Phase D exit criterion:** All processed meshes are confirmed to be watertight and are visually clean, ready for USD conversion.

### Phase E — USD Export & IsaacSim Integration

**Goal:** At least one scene loads in IsaacSim with correct scale, Z-up orientation, and physics properties. Robot spawns on the floor; depth sensor returns plausible readings.

| # | Task | Notes |
|---|---|---|
| E1 | - [ ] **Scale calibration strategy** | Must measure a known real-world distance in existing scenes and apply uniform scale post-hoc. Future captures must include a reference object. |
| E2 | - [ ] **Mesh-to-USD pipeline** | Use Isaac Lab's `MeshConverter`. This may involve creating two separate meshes: the high-quality visual mesh and a simplified, watertight collision proxy (e.g., from V-HACD convex decomposition). |
| E3 | - [ ] **IsaacSim load + spawn test** | Spawn a robot on each scene's mesh floor; verify depth sensor returns readings consistent with the scene geometry. |

**Phase E exit criterion:** All three scenes loadable in IsaacSim, robot spawns, depth sensor works.

### Phase F — Locomotion Model Integration & Demo

**Goal:** The actual M11 deliverable — Extreme Parkour and Holosoma running on the reconstructed environments with hexapod embodiment.

| # | Task | Notes |
|---|---|---|
| F1 | - [ ] **Extreme Parkour Dockerfile** | Per grant — runs in its own container, launches IsaacSim, consumes USD envs, outputs trajectory + metrics. |
| F2 | - [ ] **Holosoma Dockerfile** | Per grant — same shape, proprioception-only (no vision). |
| F3 | - [ ] **Hexapod adaptation** | URDF/embodiment configs, action/observation space updates, reward shaping for tripod-bias gait. |
| F4 | - [ ] **Stable locomotion demo** | The deliverable — both models demonstrate stable locomotion on at least one reconstructed scene. |

**Phase F exit criterion:** Acceptance criteria for T3 and T4 from the grant are satisfied.

### Phase G — Future (post-M11): Large-Scale Reconstruction via Submap Fusion

**Concept**: To scale our pipeline from single rooms to entire properties, we will adopt a submap-based fusion strategy. This approach avoids the limitations of a single large reconstruction by creating multiple, smaller, overlapping 3D scenes and then stitching them together. This is a stretch goal and is not required for M11, but the techniques developed in Phase C are foundational for this work.

The core of this strategy is to use the continuous camera path—the "spine"—as the ground truth for aligning the sub-scenes.

**Reference**: The full 11-step detailed workflow is captured in the journal note `milestones/011-scene-reconstruction/journal/journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-04T120000-submap-fusion-strategy-detailed.md`.

---

## Today's next concrete steps

Based on the 2026-05-02 handoff, the immediate priorities are:

1.  - [ ] **Finalize Bicycle Scene Comparison Views:** Place 3 named cameras in the bicycle scene's `.blend` file and render the comparison matrix. This will make the new scene rankable alongside scene 004.
2.  - [ ] **Commit Recent Code Changes:** A significant number of new scripts and modifications to the workspace (e.g., `colmap_to_cameras_json.py`, multi-scene support in the rating server) are ready to be committed.
3.  - [ ] **Address Stale Tetra-era Rankings:** Decide whether to drop or tag the old rankings for scene 004, now that the superior TSDF meshes have replaced them.
4.  - [ ] **Capture Journal Notes:** Draft and capture notes on the "TSDF >> tetra" discovery and other findings from the last session.
5.  - [ ] **Begin Phase C:** Start working on the final mesh conditioning to ensure our scenes are watertight and simulation-ready.
