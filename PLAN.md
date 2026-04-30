# M11 Plan

**Status:** active working plan
**Last updated:** 2026-04-30
**Goal:** ship M11 (Patina Foundation grant — Krabby-Uno Milestone 11) ASAP, with reliable watertight reconstruction, then locomotion-model integration

---

## M11 scope (per `grants/grants/Krabby-Uno/Milestone11-Scene-Reconstruction/OVERVIEW.md`)

The full M11 is **all of T0–T4**, not just mesh production:

| Task | Deliverable | Status |
|------|-------------|--------|
| T0 | Sparse reconstruction (camera poses + sparse cloud), 2–3 scenes | 🟡 partial — COLMAP sparse on scene 001; MASt3R-SfM poses on scene 004 |
| T1 | Dense reconstruction + **watertight triangle mesh**, each scene | 🟡 partial — **scene 004 watertight via MAtCha (sole win so far)** |
| T2 | Mesh conditioning + USD export + load in IsaacSim, robot spawns + depth-sensor returns plausible readings | ❌ not started |
| T3 | Extreme Parkour + Holosoma in Docker, both consuming the same USD envs | ❌ not started |
| T4 | Both models adapted to hexapod, stable locomotion demo on at least one scene | ❌ not started |

**The grant assumes COLMAP MVS + Poisson** as the canonical T0/T1 pipeline (with SLAM3R/MASt3R-SLAM/Spann3R explicitly listed in Appendix A as alternatives). **MAtCha is not in the grant text.** We chose it because it produces watertight meshes natively, satisfying the T1 acceptance criterion, just via a different tool. This is a defensible substitution but worth flagging when reporting back to the grant.

---

## Status snapshot (today)

**What we have:**
- One end-to-end pipeline (`Dockerfile.matcha` on `krabby-matcha:latest`) reproducible on RTX 5080
- One watertight mesh (scene 004, 422 MB tetra → 200K-tri OBJ)
- 9 documented experiments under `experiments/` with a decision matrix
- All build patches captured in `docker/MATCHA-NOTES.md` + 4 patch scripts
- Workspace remote at `jeremyprz/krabby-workspace` (private)

**What's missing for M11 acceptance:**
- 2 more watertight meshes (need 2–3 total; have 1)
- All of T2 (USD export, IsaacSim integration, scale calibration)
- All of T3 (Extreme Parkour + Holosoma Docker integration)
- All of T4 (hexapod adaptation + locomotion demo)

**Risks:**
- Per-scene MAtCha quality may vary (only validated on indoor scene 004; outdoor 003 untested)
- Scale calibration is unsolved across all our captures (no reference objects in the videos)
- IsaacSim integration is the largest unknown — we haven't validated that ANY of our meshes work as collision geometry yet
- T3 and T4 are substantial Docker/RL work, not just mesh-pipeline tuning

---

## Work plan (priority order)

### Phase A — Three watertight meshes ★ current focus

**Goal:** produce watertight reconstructed meshes for scenes 001 and 003 to add to the existing scene 004, satisfying T1's "2–3 scenes" requirement.

| # | Task | Notes |
|---|------|-------|
| A1 | **MAtCha on scene 001 (patio, hyperlapse)** | Video already on bbeeprz fleet (or recoverable). Frame selection: 12–24 max. Be careful — hyperlapse means temporally-evenly-sampled frames are also viewpoint-evenly-sampled. May want to favor coverage over count |
| A2 | **MAtCha on scene 003 (firepit, 4K@60fps)** | Source video has more frames than we need. **Frame choice is critical.** Pick 12–18 viewpoint-diverse keyframes; don't temporally-uniformly-sample 60fps |
| A3 | **Per-experiment CAPTURE-LESSONS.md** | New convention. Each experiment folder (`experiments/<num>-<pipeline>-<scene>/`) gets a `CAPTURE-LESSONS.md` documenting what we learned about the *capture* — useful for the next time we reshoot a similar scene |

**Phase A exit criterion:** three scenes (001, 003, 004) each have a watertight mesh in `data/scenes/<scene>/matcha_output/mesh/<name>_matcha_200k.obj` (or equivalent for COLMAP/MASt3R-SLAM if we end up needing fallback).

### Phase B — Frame selection & MAtCha tuning

**Goal:** improve mesh quality past the MAtCha-12-frame baseline. **Only work on this if Phase A's outputs aren't good enough**, or in parallel during Phase A waiting.

| # | Task | Notes |
|---|------|-------|
| B1 | **Human-in-the-loop frame selector tool** | Since MAtCha takes 10–24 inputs, manual selection is feasible. Build a tool: extract candidate frames at higher density, present a contact sheet, let the human click 12–18 to keep, write the curated set. Probably a small Python+Streamlit/Gradio thing or a simple browser-based picker |
| B2 | **Lower-resolution keyframe test** | Cut input to 768×432 instead of 1024×576 — does the 16-GB chart-alignment ceiling lift to >12 frames? |
| B3 | **Binary-search the frame ceiling** | Run MAtCha at 14, 16, 18, 20 frames at native resolution to find the actual cliff |
| B4 | **Mesh-conditioning improvements** | Currently using Open3D quadric decimation only. Could try `pymeshlab` floater removal, hole closing, edge cleanup before/after decimation |

**Phase B exit criterion:** we know the actual VRAM ceiling, can curate frames manually, and have a documented mesh-quality improvement path.

### Phase C — T2 (USD export + IsaacSim load)

**Goal:** at least one scene loads in IsaacSim with correct scale, Z-up orientation, and physics properties. Robot spawns on the floor; depth sensor returns plausible readings.

| # | Task | Notes |
|---|------|-------|
| C1 | **Scale calibration strategy** | Either: (a) recapture each scene with a known-size reference object (preferred — fixes the data, not the pipeline); or (b) measure a known real-world distance post-hoc and apply uniform scale. **Recommend (a) for any future capture; do (b) for the existing scenes since we can't recapture them without re-flying to the location** |
| C2 | **Mesh-to-USD pipeline** | Use Isaac Lab's `MeshConverter` per the grant's recommended path. Verify Z-up orientation, physics properties (rigid body + mesh collider), instanceable format |
| C3 | **IsaacSim load + spawn test** | Spawn a robot on each scene's mesh floor; verify depth sensor returns readings consistent with the scene geometry. Flag any meshes that fail (e.g., self-intersecting, wrong-side normals) |

**Phase C exit criterion:** all three scenes loadable in IsaacSim, robot spawns, depth sensor works.

### Phase D — T3 + T4 (locomotion models + hexapod)

**Goal:** the actual M11 deliverable — Extreme Parkour and Holosoma running on the reconstructed environments with hexapod embodiment.

| # | Task | Notes |
|---|------|-------|
| D1 | **Extreme Parkour Dockerfile** | Per grant — runs in its own container, launches IsaacSim, consumes USD envs, outputs trajectory + metrics |
| D2 | **Holosoma Dockerfile** | Per grant — same shape, proprioception-only (no vision). Per Milestone 5, has a clean extension pattern |
| D3 | **Hexapod adaptation** | URDF/embodiment configs, action/observation space updates, reward shaping for tripod-bias gait |
| D4 | **Stable locomotion demo on at least one scene** | The deliverable — both models, hexapod embodiment, completes a run on at least one reconstructed scene |

**Phase D exit criterion:** acceptance criteria for T3 and T4 satisfied.

### Phase E — Future (post-M11): Lyra2-augmented re-capture

**Concept**: once a base reconstruction exists, use Lyra2 (or similar generative video model) to **virtually fly a camera through the reconstructed scene** at angles the original capture missed (under tables, behind chairs, etc.). Use those generated frames as input to a *re-reconstruction* pass that fills in the original capture's coverage gaps.

**Why this is post-M11:**

- We need a working base reconstruction first (T1 done) before we have anything to virtually fly through
- We need T2 done so we have IsaacSim to drive the virtual camera
- The output's geometric trustworthiness is unproven — generative video frames could hallucinate, and we'd be using them to inform the *physical* mesh that the robot trains on. That's a significant correctness risk that's worth a careful experiment, not a rushed addition before M11 ships
- See the deferred AnyRecon analysis (`experiments/004-anyrecon-sky-house/README.md`) for the same caveats applied to a different generative-pipeline question

**To preserve here:** the idea is good and worth a serious post-M11 experiment. Specifically, the case where it could legitimately add value is **scenes where the capture has known gaps the operator couldn't reach physically** (under furniture, behind walls in the same room, etc.). For typical walk-through captures with reasonable coverage it's redundant.

---

## Conventions established here

### `CAPTURE-LESSONS.md` per experiment folder

Going forward, each experiment folder may include a `CAPTURE-LESSONS.md` documenting capture-side findings specific to that scene. Examples of things that go in capture lessons:

- Camera path that worked / didn't work
- Lighting issues encountered
- Frame-rate / resolution choice rationale
- Things to do differently if recapturing the same scene
- Reference-object placement (when we start using them)

These are local-scope (one scene), distinct from the cross-cutting `OLAI corpus 3d-reconstruction/capture-profiles` topic which captures generalizable knowledge.

### Plan + execution discipline

- This file (`PLAN.md`) is the master schedule. Keep it terse.
- Per-experiment work goes in `experiments/<num>-<pipeline>-<scene>/README.md`
- Per-experiment capture lessons go in `experiments/<num>-<pipeline>-<scene>/CAPTURE-LESSONS.md`
- Decision matrix (`experiments/DECISION-MATRIX.md`) gets updated as new evidence comes in
- Cross-cutting build / patch knowledge goes in `docker/MAST3R-NOTES.md` / `docker/MATCHA-NOTES.md` + the OLAI corpus

---

## Open questions (resolve before next phase boundary)

1. **Scale calibration approach** — recapture with reference objects (option a) or measure-and-scale post-hoc (option b)? Recommend (a) for future, (b) for existing scenes.
2. **MAtCha quality bar** — what's the minimum mesh quality that passes IsaacSim's "robot spawns on floor + depth sensor returns plausible readings"? We won't know until we try, but worth eyeballing the existing scene 004 mesh in IsaacSim early to set the bar.
3. **Outdoor vs indoor MAtCha behavior** — scene 004 is indoor; scenes 001 and 003 are outdoor. Real possibility outdoor MAtCha quality differs (sky regions, distant objects, variable light). Phase A1 + A2 will tell us.
4. **Should we keep MASt3R-SLAM as a fallback for scenes where MAtCha fails?** The image is already built (`krabby-mast3r:latest`). For little extra effort we could run both pipelines on scenes 001 and 003 and pick the winner per scene.

---

## Today's next concrete step

**Start Phase A1: run MAtCha on scene 001.** The capture exists (942 frames of patio hyperlapse). The pipeline is reproducible (`runner.sh` from `experiments/004-matcha-sky-house/`). The unknowns are: how does MAtCha behave on a hyperlapse (vs the smooth 30fps motion of scene 004), and how do we sample frames from a hyperlapse to maximize viewpoint diversity.

This single experiment teaches us most of what we need to know about phase A.
