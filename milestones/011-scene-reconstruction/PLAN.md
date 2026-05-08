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
| **This plan** | this file | Task hierarchy (T0–T4) with phase letters as sub-organization; bridge between grant and beads. Acceptance is *not* judged against PLAN.md. |

### Naming convention

Beads follow `T<N>.<sub-id>` (where `<sub-id>` is the historical phase
letter+number, e.g. `B1`, `C2`, `E1`). Cross-cutting risks use `R<n>`.
Grant Task `T<N>` is primary; phase letters (A–F) are sub-organization
inside a Task. Phase G is post-M11 stretch (no Task assignment).

### Grant-task ↔ Phase ↔ Beads mapping

| Grant Task | Phases (sub-organization) | Open beads | Closed beads |
|---|---|---|---|
| **T0** — Sparse reconstruction | A subset (eval), B5 (frame curation) | — (rollup `m11-di0 T0` ✓) | `m11-nk8 T1.A`, `m11-2bg T0.B5`, `m11-di0 T0` |
| **T1** — Dense + watertight mesh | A (initial tetra), B1–B4 (post-processing), C (validation, optional) | `m11-2cd T1.C2`, `m11-8u6 T1.C3` | `m11-23n T1.A2`, B1–B4 (`5wp T1.B1`, `1oc T1.B2`, `8wy T1.B3`, `pbo T1.B4`), `ep4 T1.C-Schema` + `s0h T1.C-Manual` + `5ef T1.C-AutoLocalize`, rollup `dua T1` |
| **T2** — Conditioning + USD + IsaacSim | D (conditioning), E (USD + IsaacSim) | `m11-21m T2.D1`, `m11-400 T2.D2`, `m11-87v T2.D3`, **`m11-u3l T2.E1 (P0)`**, **`m11-uy5 T2.E2 (P0)`**, **`m11-dt8 T2.E3 (P0)`** | — |
| **T3** — EP + Holosoma in Docker | F1, F2 | `m11-bz6 T3.F1`, `m11-x6i T3.F2` | — |
| **T4** — Hexapod + stable demo | F3, F4 | `m11-1ij T4.F3`, `m11-4tl T4.F4` (the deliverable) | — |
| (cross-cutting risks) | — | **`m11-d06 R1 (P0 schedule)`**, `m11-kr9 R2 (T0/T1 disclosure)` | — |

**Phase G (post-M11 stretch — submap fusion)** is intentionally not in
beads or a Task; revisit on M12+ scoping.

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

**What we have (T0 + T1 closed):**
- An end-to-end, reproducible pipeline for turning raw video into high-quality, post-processed 3D scenes.
- A major quality breakthrough by switching from tetrahedral to **TSDF meshes**, which are visually superior.
- Multiple high-quality TSDF meshes for "scene 004" and the new "bicycle" benchmark scene.
- A full **post-processing pipeline (T1 Phase B)** that automatically handles gravity alignment, background culling, vertex coloring, and camera visualization.
- A powerful **interactive 3D camera selection tool** (`viewer.py`) for curating the best viewpoints from a large pool of candidate frames.
- A scalable MASt3R-SfM pipeline that can generate poses for 350+ frames.

**What's missing for M11 acceptance:**
- All of T2 (mesh conditioning, USD export, IsaacSim integration — three open P0 beads).
- All of T3 (Extreme Parkour + Holosoma Docker integration).
- All of T4 (hexapod adaptation + locomotion demo — F4 is the deliverable).

**Risks:**
- T2 IsaacSim integration is the largest unknown — TSDF meshes not yet validated as collision geometry; may require V-HACD.
- Scale calibration (T2.E1) is unsolved across all captures (no reference objects in the videos). This is a P0 BLOCKER with no fallback yet.
- T3 and T4 are substantial Docker/RL work, not just mesh-pipeline tuning.

---

## Work plan (Task hierarchy)

Reorganized 2026-05-06 from Phase A–G to Task 0–4 to align with grant
overview, ICA acceptance, and beads naming. Phase letters preserved as
sub-organization within each Task.

### T0 — Sparse Reconstruction

**Status: ✅ Done.** Acceptance per grant: COLMAP `sparse/` (or
equivalent) with poses + intrinsics + sparse cloud, verified visually.
Delivered via MASt3R-SfM + COLMAP-conversion path (substitution per
grant Appendix A; disclosure tracked as **R2**).

- **Phase A (eval) / [`m11-nk8`]:** Pipeline evaluation chose MAtCha
  end-to-end. (Cross-cutting; also feeds T1.)
- **Phase B5 / [`m11-2bg`]:** Frame-Selection Tooling (Camera Selection
  Viewer) for SfM-scaling.

### T1 — Dense Reconstruction & Watertight Mesh

**Status: ✅ Done (TSDF route).** Acceptance per grant: dense cloud +
watertight OBJ/PLY per scene; walkable surfaces accurate. Delivered via
MAtCha TSDF; multiple high-quality variants for scene 004 + bicycle.
Tetra-match (C2/C3) is decoupled per Manager memo 2026-05-06.

#### Phase A — Pipeline Evaluation & Initial Mesh Generation
**Status: ✅ Done.** Comprehensive evaluation of 3D reconstruction
pipelines (COLMAP, MASt3R-SLAM, MAtCha). MAtCha selected as the only
candidate reliably producing watertight meshes end-to-end. Initial
tetrahedral meshes generated for three scenes (001-patio, 003-firepit,
004-sky-house). The post-processing gap inventory drove Phase B.

- [x] [`m11-nk8`] **A** — Pipeline evaluation & MAtCha selection
- [x] [`m11-23n`] **A2** — Initial tetrahedral meshes for 3 scenes

#### Phase B — Post-Processing Pipeline
**Status: ✅ Done.** Tooling for B1-B4 + B5 viewer (B5 listed under T0).

- [x] [`m11-5wp`] **B1** — Auto-deduce ground plane (RANSAC) → Z-up, floor at z=0
- [x] [`m11-1oc`] **B2** — Auto-cull out-of-bounds geometry
- [x] [`m11-8wy`] **B3** — Auto-include camera locations as Blender markers
- [x] [`m11-pbo`] **B4** — Auto-project vertex colors

#### Phase C — MAtCha Output Validation (decoupled from T1 acceptance)
**Goal:** Validate our pipeline reproduces the reference quality from
the official MAtCha paper for both TSDF and adaptive-tetra mesh
extraction. **Per Manager memo 2026-05-06: this is decoupled from T1
acceptance** — TSDF satisfies the watertight grant criterion; tetra-match
is a self-imposed quality bar.

- [x] [`m11-ep4`] **C-Schema** — `comparison_views.json` schema v4 + bidirectional injection
- [x] [`m11-s0h`] **C-Manual** — Manual `cam_ref` placement (bicycle TSDF)
- [x] [`m11-5ef`] **C-AutoLocalize** — Auto-localize reference cameras via SfM-extend
- [ ] [`m11-2cd`] **C2** — Validate TSDF mesh quality vs paper reference (P3, nice-to-have)
- [ ] [`m11-8u6`] **C3** — Match adaptive tetrahedralization quality (P3, nice-to-have)

### T2 — Mesh Conditioning + USD + IsaacSim ★ Critical Path

**Status: 🟡 In Progress.** Acceptance per grant: floater removal, hole
fill, smoothing, decimation; collision mesh (convex decomp); USD with
correct scale, Z-up, physics; loads in IsaacSim; robot spawns; depth
sensor returns plausible readings. Three open P0 beads (E1 / E2 / E3)
form the longest unknown-tail in the milestone.

#### Phase D — Final Mesh Conditioning for Simulation
**Goal:** Transform validated TSDF meshes into simulation-ready assets
with clean, manifold, watertight collision geometry.

- [ ] [`m11-21m`] **D1** — Merge & gap-fill surfaces (Poisson on TSDF output)
- [ ] [`m11-400`] **D2** — Verify watertightness (genus / manifold report)
- [ ] [`m11-87v`] **D3** — Final Taubin smoothing pass (P3)

#### Phase E — USD Export & IsaacSim Integration
**Goal:** ≥1 scene loads in IsaacSim with correct scale, Z-up, physics;
robot spawns; depth sensor returns plausible readings.

- [ ] [`m11-u3l`] **E1** — Scale calibration strategy (★ BLOCKER, no fallback yet) **P0**
- [ ] [`m11-uy5`] **E2** — Mesh-to-USD via Isaac Lab `MeshConverter` (visual + V-HACD collision) **P0**
- [ ] [`m11-dt8`] **E3** — IsaacSim load + robot spawn + depth sensor returns **P0**

**T2 exit criterion:** ≥1 scene loadable in IsaacSim end-to-end with
working depth sensor on the reconstructed mesh.

### T3 — Locomotion Models in Docker

**Status: ❌ Not started.** Acceptance per grant: two Dockerfiles (EP +
Holosoma); both launch IsaacSim independently; both consume same USD
envs; depth-based observations (EP) and proprioception-only (Holosoma)
both work; consistent trajectory/metric output schema.

#### Phase F1 / F2 — Locomotion Containers
- [ ] [`m11-bz6`] **F1** — Extreme Parkour Dockerfile (depth-based) **P1**
- [ ] [`m11-x6i`] **F2** — Holosoma Dockerfile (proprio-only) **P1**

### T4 — Hexapod Adaptation & Stable Demo

**Status: ❌ Not started.** Acceptance per grant: URDF/embodiment for
hexapod; reward shaping (penalize simultaneous-leg, encourage tripod
alternation); both models demonstrate stable hexapod locomotion in ≥1
environment without falls over a defined window.

#### Phase F3 / F4 — Adaptation + the Deliverable
- [ ] [`m11-1ij`] **F3** — Hexapod URDF + reward shaping (in both stacks) **P1**
- [ ] [`m11-4tl`] **F4** — Stable locomotion demo on ≥1 reconstructed scene **P1** *(the deliverable)*

**T4 exit criterion = milestone acceptance:** F4 demonstrates both EP
and Holosoma running stable hexapod gaits on at least one reconstructed
scene; trajectory + metric outputs collected; demo evidence committed.

### Cross-cutting risks (R-prefix; not Task-bound)

- [ ] [`m11-d06`] **R1** — Schedule trajectory re-baseline conversation with Fletcher **P0**
- [ ] [`m11-kr9`] **R2** — T0/T1 tool-substitution disclosure in final M11 README

### Phase G — Post-M11 Stretch: Large-Scale Reconstruction via Submap Fusion

**Not a Task. Not in beads.** Stretch goal: scale the pipeline from
single rooms to entire properties via submap-based fusion (using the
continuous camera path as the "spine" for sub-scene alignment). The
techniques developed in Phase C are foundational for this work; revisit
on M12+ scoping.

**Reference:** The full 11-step detailed workflow is captured in the
journal note `journal/journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-04T120000-submap-fusion-strategy-detailed.md`.

---

## What's ready to work on now

Beads `bd ready` (run from `beads/`) gives the live unblocked queue.
Current top of queue (2026-05-06):

| Bead | Title | Priority |
|---|---|---|
| `m11-d06` | **R1** — Schedule trajectory re-baseline with Fletcher | **P0** |
| `m11-u3l` | **T2.E1** — Scale Calibration Strategy ★ BLOCKER | **P0** |
| `m11-21m` | **T2.D1** — Merge & Gap-Fill Mesh Surfaces | P2 |
| `m11-kr9` | **R2** — T0/T1 Tool-Substitution Disclosure | P2 |
| `m11-2cd` | **T1.C2** — Validate TSDF Mesh Quality (nice-to-have) | P3 |
| `m11-8u6` | **T1.C3** — Match Adaptive Tetra Quality (nice-to-have) | P3 |

Per Manager memo 2026-05-06, the recommended sequence:
1. **R1** schedule re-baseline with Fletcher (this week — see Manager memo)
2. **T2.E1** scale calibration (largest single technical unknown; gates E2/E3)
3. **T2.D1** merge/gap-fill in parallel (gates D2 → E2)
4. Defer **T1.C2 / T1.C3** (decoupled from T1 acceptance per memo)
5. **R2** disclosure paragraph drafted before final M11 README ships
