---
kind: note
captured: 2026-05-01T20:39:49-07:00
consolidated: true
tags: []
---
# Accomplishments and Next Steps

End-of-session checkpoint capturing what got done in 2026-05-01 and where the work picks back up.

## Accomplishments

### Journal infrastructure (morning)

- Stood up the manual filesystem journal at `milestones/011-scene-reconstruction/journal/` using OLAI's locked 4-resource layout (journal/thread/entry/note).
- Bootstrapped 3 threads (`inbox`, `matcha-quality`, `post-processing`), 4 entries, multiple notes capturing the Phase A retrospective and the planning around B5/Option C.
- Migrated all slugs from date-precision to timestamp-precision after the OLAI D5 amendment landed; `jlib.py audit` clean.
- Pulled in the research-side commit's lessons (cu128 wheel index for RTX 5080; `--shm-size=8g` mandatory for PyTorch containers — both verified on `matcha-build`).

### Code-read of MAtCha source (afternoon)

Cloned `Anttwo/MAtCha` to JDP-Mac for local inspection. Settled three open questions:

1. **`r` is not a single knob — it's a 4-level pyramid** `[0.05, 0.1, 0.2, 0.4]` with 8 features per level. The default config sets `use_multi_res_charts_encoding: True`. Option C's lever is truncating the resolutions list, not tuning a single value.
2. **Photometric supervision uses input-resolution images** (capped at 1600 long-edge); chart geometry is pinned at 512 long-edge. Option A would change photometric refinement but not chart geometry.
3. **`train.py --sfm_only` and `--image_idx` already exist** — no wrapper needed for B5. The viewer's output `selected_frames.json` plugs straight into MAtCha.

### SfM-scaling experiment (afternoon)

Bracketed the MASt3R-SfM ceiling on RTX 5080 / 16 GB.

- Sweep on bbeeprz: N=24, 60, 120, 200, 300 (all clean, all cameras returned).
- Parallel sweep on tbeeprz: N=350 (success, 15.5 GB peak — at the edge), N=500 (OOM at 15.45 GB).
- **Verdict: ~300 frames is the comfortable operating point; 350 is the upper bound; 400-500 OOMs.** Far above any human-curatable pool, so the SfM ceiling never binds for B5 candidate-pool sizing.
- Operational discovery: `nvidia-smi --query-compute-apps` revealed S&box (Source 2 sandbox owned by a different user, `benny`) holding 4.3 GB on bbeeprz baseline, contaminating early peak measurements. Sudo-killed mid-experiment to clean up. **Lesson captured**: always inspect per-process GPU before measuring.
- Watchdog pattern saved ~20 min of futile compute by killing the chain after N=300 succeeded, before it tried N=500.

Detailed measurements + logs in `experiments/004-sfm-scaling-sky-house/`.

### Camera Selection Viewer (afternoon → evening)

Built the Route B viser-based 3D viewer end-to-end (~900 lines across data.py, filters.py, ui.py, viewer.py, slots.py, clustering.py).

Capabilities now live:
- **3D camera frustums + textured image planes** rendered from `cameras.json` + frame thumbnails.
- **Camera path polyline** in temporal order.
- **Seven filters** composing as boolean AND: time range, temporal stride, spatial-cluster checkboxes (with invert), distance-from-selection, look-at gizmo, pHash dedupe, picked-status dropdown.
- **Selection state**: click-to-toggle frustum + bright marker sphere for visibility at any zoom; bulk Select Visible / Deselect Visible; Lock picks; Coverage colorize.
- **Counters**: Visible X/Y, Selected X/Y, both live-updated.
- **Named slot save/load**: filter+selection state serialized to `cameras.slots.json` parallel to the data.

Two real bugs caught + fixed during build:
- `forward_axes` had wrong sign: MASt3R-SfM uses OpenCV (+Z forward), not OpenGL (-Z). The look-at filter was inverted before the fix.
- `Image.LANCZOS` (deprecated) → `Image.Resampling.LANCZOS`; missing `scipy` in requirements.

### First curated MAtCha run

Picked 12 frames from the n350 viewer scene. Ran full pipeline on tbeeprz:
- Wall-clock: **648 sec (10:48)**, peak VRAM **7.7 GB** (well under tbeeprz's clean 14.8 GB free).
- Concurrent with Theo's Discord (sharing the same GPU at 263 MiB) — no contention issues.
- Output: watertight tetra mesh + 2DGS gaussians + per-frame SfM data at `~/outposts/krabby/data/011-scene-reconstruction/scenes/004-sky-house-curated-12/`.

This is the **first B5-validated mesh** the pipeline has produced. Frames were spatially curated rather than evenly-time-spaced.

### Post-processing (in flight at session end)

B1 orient → decimate → B4 project_color → B2 cull running on tbeeprz against the curated mesh. Will finalize as `scene_culled.blend` after a Blender pass on JDP-Mac (B3).

## Next steps (pick up here)

In priority order:

### 1. Finish the curated post-processing + validate B5 (tomorrow)

- Wait for the in-flight B1-B4 to finish on tbeeprz (`bfxefknnh` background task).
- Pull culled mesh + cameras + oriented_cameras + images to JDP-Mac.
- Run B3 (Blender headless) locally → produces `scene_culled.blend` for the curated mesh.
- **Compare side-by-side in Blender**:
  - `data/scenes/004-sky-house-dining/matcha_output/oriented/scene_culled.blend` (Phase A baseline, 12 evenly-time-spaced)
  - `data/scenes/004-sky-house-curated-12/oriented/scene_culled.blend` (this session's curated)
- Eyeball verdict on coverage / hole density / distant-noise / sharpness.
- **Decision point**: was curation worth it?

### 2. Iterate on curation if needed (or roll out if not)

- ✅ **If curated wins clearly**: roll out the workflow to scenes 001 and 003. Update DECISION-MATRIX. The slot-save system makes iteration cheap — try multiple curations per scene without losing context.
- 🟡 **Mixed result**: the slot system is built for this. Save the current pick as a baseline, try alternative curations, compare. Also worth trying the unpursued tracks: Option C (`r` truncation) and bracketing the MAtCha N ceiling (14, 16, 18, 20 picks at 1024×576).
- ❌ **Curated worse**: meaningful negative result. Capture in a journal entry, look elsewhere — capture protocol or MAtCha-internal knobs.

### 3. Phase C: USD export + IsaacSim load (the actual M11 deliverable)

This is real new engineering, ~2-3 days of focused work:
- Blender → USD export (multiple exporters; pick the one IsaacSim's USD parser handles well)
- Mesh decimation to ≤200K tris for PhysX collision performance
- Collision proxy generation (V-HACD convex decomposition)
- Coordinate-system sanity check (Blender +Z up vs IsaacSim convention)
- Material bake (vertex color → texture)
- IsaacSim load test: open scene, drop hexapod, verify gravity + collision.

### 4. Phase D: hexapod parkour validation in IsaacSim

The grant scope's M11 success criterion. Run the existing Extreme Parkour locomotion policy on the captured scene; verify the hexapod can walk through without falling through floors or getting stuck on hallucinated geometry. **This is where mesh quality really matters** — bad meshes produce IsaacSim physics issues that don't appear in Blender.

### 5. Beyond M11

Phase E (real-robot deployment) is out of scope for this milestone. Lessons from D inform M12+.

## Open research tracks (not blocking)

These are deferrable but worth knowing about:

- **Option C — chart-encoding-resolution `r` truncation**: change the active resolutions list from `[0.05, 0.1, 0.2, 0.4]` to `[0.05, 0.1, 0.2]` (drop the finest level). Tests the over-fitting-to-noisy-SfM hypothesis. Cheap experiment, untested.
- **Bracket the MAtCha N ceiling**: try N=14, 16, 18, 20 picks at 1024×576 in `train.py --image_idx`. Could lift the per-mesh budget from 12 to 16-20.
- **Submap-fusion architecture for multi-room captures (M12+)**: covered in journal note `2026-05-01T174650-submap-based-mesh-fusion-...`. Nothing actionable for M11.
- **Sliding-window vs PnP localization (M12+)**: covered in journal note `2026-05-01T174651-sliding-window-sfm-...`. Becomes relevant if global SfM ceiling ever binds (it doesn't for current curation).

## State at end of session

```
viewer running:  http://localhost:8080  (PID detached)
                 n350 dataset loaded; named-slot system live
                 selected_frames.json template at /tmp/n350-selection.json
bbeeprz:         idle (chain stopped earlier)
tbeeprz:         B1-B4 post-processing running on curated mesh
                 (background task `bfxefknnh`)
                 GPU shared with Theo's Discord at 263 MiB; no contention

Local data (gitignored): 
  data/sfm-scaling-out/   ~1 GB    SfM outputs for N=24..500
  data/scenes/             ~7.5 GB  Phase A meshes (older)

Local commits today:  4
                      bfe7e59  SfM-scaling: document local mirror layout + viewer invocation
                      465836c  Close SfM-scaling experiment + ship Camera Selection Viewer v0
                      a8cde37  Start SFM-scaling experiment + capture supporting research
                      7a4af0e  Migrate M11 journal slugs from date to timestamp precision
                      bcb5daf  Stand up M11 journal; archive journal-setup task

Outstanding journal work:
  - viewer/post-processing changes (filters, slots, etc.) since 465836c are uncommitted
  - this note + the post-processing finalize will need a final commit
```
