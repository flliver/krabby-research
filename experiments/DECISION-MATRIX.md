# Pipeline Decision Matrix — M11 Scene Reconstruction

A structured comparison of every reconstruction pipeline we evaluated,
scored against the M11 milestone's actual deliverable requirements.

**Snapshot date:** 2026-04-30 (post-Phase-A retrospective)
**Leading candidate:** **MAtCha** (now validated on 3 scenes: 001, 003, 004)

---

## M11 requirements (the rubric)

| Req | Description | Hard constraint? |
|-----|-------------|------------------|
| R1 | Produces a **watertight surface mesh** (no holes, hexapod-collision-safe) | **Hard** — no watertight mesh means no IsaacSim collision deliverable |
| R2 | Recovers **metric-accurate scale** (or scale via a known reference object in capture) | **Hard** — sim scale must match real |
| R3 | Recovers **camera intrinsics + extrinsics** (for downstream NeRF / texturing / re-render) | Soft — useful but not blocking the locomotion deliverable |
| R4 | Multi-arch container — runs on RTX 4080 (sm_89) and RTX 5080 (sm_120) | Soft — speeds fleet-wide use |
| R5 | Realistic **wall-clock per scene** for iterative work (≤ 1 hr ideal, ≤ 4 hr acceptable) | Soft — affects how many scenes we can deliver |
| R6 | **Build complexity** is bounded (≤ 1 day to stand up; reproducible from a Dockerfile) | Soft — but compounds across scenes |
| R7 | **Validated end-to-end** on real M11 video, not just the paper's test set | **Hard** — paper claims don't ship |

---

## The matrix

Scoring legend:

- **✅** = meets requirement, validated
- **🟡** = partial / requires extra work / has caveats
- **❌** = does not meet
- **❓** = untested in this milestone

| | COLMAP | MASt3R-SLAM | SLAM3R | VGGT | **MAtCha** | AnyRecon |
|---|---|---|---|---|---|---|
| **R1 Watertight mesh** | 🟡 (post-Poisson + density crop) | 🟡 (post Open3D ball pivot — *not* watertight) | 🟡 (post-Poisson, untested for M11) | ❌ (point cloud only, untested) | **✅ native via TSDF + tetra** | ❓ (NVS, not surface) |
| **R2 Metric scale** | 🟡 needs reference object | 🟡 needs reference object | 🟡 needs reference object | 🟡 unknown | 🟡 needs reference object | 🟡 unknown |
| **R3 Camera poses** | ✅ (COLMAP-format) | ✅ (per-frame poses + COLMAP export) | ❌ (point cloud only, no poses) | ✅ (all-in-one forward pass) | ✅ (via MASt3R-SfM stage; COLMAP-format) | ✅ |
| **R4 Multi-arch RTX 5080** | 🟡 needs build-from-source with multi-arch flags | ✅ (NGC PyTorch 25.10 base) | 🟡 cu124 wheel, CUDA 12.8 — sm_120 untested | ❌ unworkable (40+ GB VRAM for >50 frames) | ✅ (PyTorch 2.7+cu128, our build) | ❓ |
| **R5 Wall-clock per scene** | 60–90 min (mapper + dense MVS) | 15–20 min/min of 2.7K input → 40 min for 4-min clip | similar to MASt3R-SLAM | OOM | **~11 min** end-to-end on 12 frames | ~90–105 s reported |
| **R6 Build complexity** | low (apt install, build CUDA from source) | **high** — 12 patches; the M11 stress-test of build engineering | medium — fewer patches than MASt3R-SLAM | low (paper code mostly works) | **high** — 8 patches (curope, cstdint, cfloat, ABI, CUDA include, faiss-cpu, weights_only, xformers trap) | unknown — not built |
| **R7 Validated on M11 video** | ✅ scene 001 sparse only | ✅ scenes 001, 003, 004 | ✅ scene 003 only | ❌ never produced output | ✅ scene 004 (this session) | ❌ not yet attempted |

---

## Per-pipeline summary

### COLMAP — the classical baseline

- **Strengths**: gold standard for camera-pose recovery; outputs in formats every downstream tool understands; runs everywhere.
- **Weaknesses for M11**: the *whole pipeline* (sparse → dense MVS → meshing) is slow (60–90 min/scene); needs the right camera model (`SIMPLE_RADIAL_FISHEYE` for DJI 155° FOV); fails completely on dewarped video.
- **Why it's not the leader**: dense MVS was never run for M11. Even if it had been, the resulting mesh would be unconditioned — we'd still need a Poisson + Taubin pass for watertightness, which the learned methods can match or beat at a fraction of the runtime.
- **When still useful**: as a **reference / metric-baseline** to compare learned methods against. Not as the deliverable.

### MASt3R-SLAM — the production workhorse

- **Strengths**: real-world tested on 3 of our 4 scenes, multi-arch image already on bbeeprz, recovers camera poses, fast enough.
- **Weaknesses for M11**: output is a **point cloud, not a mesh**. The Open3D ball-pivoting conditioning step we ran produced a 200K-tri OBJ but it is **not watertight** — there are real holes wherever the SLAM didn't densely sample. Switching the conditioning to Poisson+density-crop would help but adds another step and doesn't always converge well on SLAM-derived clouds.
- **Why it's not the leader**: the watertight requirement (R1) is unmet without a meaningful conditioning pass we haven't validated.
- **When still useful**: when you have a long video and want pose-accurate dense reconstruction. Best base layer to feed downstream specialised meshers.

### SLAM3R — simpler-build alternative

- **Strengths**: simpler Docker recipe than MASt3R-SLAM; single-process feed-forward (no `--shm-size=8g` deadlock); runs on cu124.
- **Weaknesses for M11**: **lacks camera-pose export**. For M11 specifically that's not blocking, but it loses points on R3. Otherwise the comparison versus MASt3R-SLAM is "MASt3R-SLAM is more complete; SLAM3R is easier to set up."
- **Why it's not the leader**: MASt3R-SLAM dominates it on outputs at the cost of more build complexity. Once MASt3R-SLAM is built, SLAM3R has no advantage.
- **When still useful**: a backup if MASt3R-SLAM build issues block iteration.

### VGGT — ruled out

- **Strengths**: one-shot global-attention transformer is conceptually clean.
- **Weaknesses for M11**: **structural mismatch with our hardware**. Global attention's VRAM scales with frame count; 16 GB caps at ~50 frames (our 4-min capture is 6804 frames). Untested on RTX 5080 because never reached runtime.
- **Why it's not the leader**: hardware-incompatible.
- **When still useful**: not for this milestone. Worth re-evaluating when we have a 40+ GB GPU available.

### MAtCha — the leader (as of 2026-04-30)

- **Strengths**: **produces a watertight mesh natively** via TSDF + tetrahedralization (R1 ✅). Sparse-view pipeline (12 frames sufficed); ~11 min wall-clock end-to-end (R5 ✅). Camera-pose recovery via MASt3R-SfM stage (R3 ✅). Multi-arch image runs on RTX 5080 (R4 ✅). Validated on scene 004 (R7 ✅).
- **Weaknesses**:
  - **High build complexity** (R6 🟡) — 8 patches vs MASt3R-SLAM's 12, but more architectural variety (pytorch3d-from-source, gcc-13 transitive includes, faiss-cpu fallback, etc.).
  - **VRAM ceiling at 24 frames** on 16 GB during chart alignment; we ran at 12 frames. May need investigation for larger scenes.
  - **Output mesh is geometrically dense** (21M tris from 12 frames) — needs decimation for downstream use; we produced a 200K-tri OBJ via Open3D quadric collapse.
- **Why it's the leader**: only pipeline that produces a watertight mesh end-to-end with no separate conditioning step. Wall-clock is competitive.
- **What we still need**: scale calibration against a known reference, second/third scene capture, and IsaacSim USD import validation.

### AnyRecon — speculative

- **Strengths (per the OLAI corpus entry)**: ~90–105s reported wall-clock; geometry-aware retrieval addresses the floater/ghosting class of error.
- **Weaknesses**: not yet tested on M11 hardware. Targeted at unordered photo sets, not video; the corpus entry itself notes MASt3R-SLAM/SLAM3R are typically better fits when video is available.
- **Why it's not the leader**: untested. Worth a session if MAtCha hits a wall on a future scene that AnyRecon's diffusion-fill might handle better.

---

## Phase A retrospective — what 3 MAtCha runs taught us (2026-04-30)

After running MAtCha on scenes 001 / 003 / 004, the qualitative pattern
is consistent:

| Scene | Source | Quality verdict |
|-------|--------|-----------------|
| 001 patio (outdoor, hyperlapse) | 4K @ 30fps, 31s | "Chaotic, but obviously the filmed scene. Includes too much background noise (far things) that would ideally be culled." |
| 003 firepit (outdoor, regular) | 4K @ 60fps, 5:31 | "Chaotic, but obviously the filmed scene. Also includes too much background noise." |
| 004 sky-house (semi-indoor) | 2.7K @ 30fps, 3:47 | "Dense in many areas, but obvious gaps in places — probably not covered." |

**Cross-cutting issues found in all three** (regardless of capture profile or scene):

- 🔴 No clear ground plane
- 🔴 Output mesh always "on a tilt" (no consistent up direction)
- 🟠 Background-noise pollution (especially 155° fisheye outdoor)
- 🟡 No camera locations visible in mesh
- 🟡 No vertex color from source frames

These are **mesh-conditioning / post-processing problems**, not MAtCha
problems. MAtCha satisfies its T1 acceptance criterion (watertight
mesh) on every scene we ran. The issues block T2 (USD export +
IsaacSim load), which needs gravity alignment and a usable ground plane.

## Recommendation (revised post-Phase-A)

**For the M11 deliverable, the right next investment is post-processing,
not more captures or pipeline hopping.** Specifically:

1. **Build a post-processing pipeline** (PLAN.md Phase B1-B4) that takes
   raw MAtCha tetra mesh → gravity-aligned, ground-plane-deduced,
   background-culled, vertex-colored, IsaacSim-ready USD.
2. **Defer Phase B5 (frame-selection tooling) and B6 (MAtCha-internal
   tuning)** until we know whether the post-processed output is good
   enough. The mesh density issues might be an artifact of the
   post-processing gap, not the pipeline.
3. **Recapture future scenes with the validated 2.7K @ 30fps profile +
   reference object + (if QR-cull ships) corner fiducials.** Don't
   recapture 001/003/004 yet — fix post-processing first; recapture
   only if the post-processing reveals genuine capture-side gaps.
4. **Defer COLMAP, SLAM3R, VGGT, AnyRecon** unless MAtCha output post
   conditioning fails to meet T2 acceptance criteria.

Shortest path to M11: **fix the post-processing**, not more pipelines or more captures.

---

## Open questions to resolve

These are the things we don't know yet that could change the recommendation:

1. **Does the MAtCha mesh, after decimation, actually produce good IsaacSim collision behaviour?** Need to convert one scene to USD and walk a hexapod through it.
2. ~~**Is the 16-frame VRAM ceiling per-resolution?**~~ — **answered 2026-05-01**: yes, 768×432 lifts the ceiling to ~15-17 frames, but the resulting mesh quality is **visibly worse** than the 12-frame 1024×576 baseline. Resolution-loss dominates view-count-gain at this scale. See `experiments/004-matcha-lowres-sky-house/README.md` (negative result).
3. **What's the actual quality difference** between MAtCha's mesh and MASt3R-SLAM's ball-pivoting mesh on the same scene? You have both in Blender now.
4. **Can we get scale recovery without a reference object?** Some VINS or IMU-fused methods support this; would simplify capture protocol.
5. **Does AnyRecon's generative novel-view synthesis** add anything when we have video (vs unordered photos)? Probably not, but worth a 1-day spike if MAtCha hits a wall.
6. **Is the bottleneck per-pixel detail or view diversity?** Negative result on (more frames, lower res) suggests detail dominates. The opposite-direction test (same 12 frames at higher resolution, e.g. 1280×720, if it fits in VRAM) would confirm. Untested.
7. **Does manual frame curation at the same 12-frame budget improve quality?** (B5 — pick the 12 most viewpoint-diverse frames from a wider candidate pool). Untested. Higher likelihood of producing real quality gains than any further VRAM gymnastics.
