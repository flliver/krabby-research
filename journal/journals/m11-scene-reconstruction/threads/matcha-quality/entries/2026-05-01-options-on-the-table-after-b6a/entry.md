---
kind: entry
date: 2026-05-01
title: Options on the table after B6a — what to try next
mood: decided
consolidates_notes:
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01-chart-encoding-resolution-r-knob
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01-spatial-frame-curation-via-mast3r-sfm
tags: [matcha, planning, options, sparse-view, decided]
---

# Options on the table after B6a — what to try next

## Framing

After the lowres negative result (sibling entry, same date), the remaining levers fell into three categories. This entry was the standing list while the decision was deferred. **Decision (2026-05-01, end of session): pursue C, then a refined B.** Option A is on hold pending a code-read to verify whether it tests a meaningful variable (see updated A below).

## Option A — higher resolution at the same 12-frame budget

**Move:** test 1280×720 (or whatever fits) at 12 frames. Direct test of the opposite direction from B6a — if the bottleneck was per-pixel detail, this should improve the mesh.

**Pros:**
- Cheapest experiment to run (~30 min if it fits, no new tooling).
- Directly disconfirms or confirms the "detail was the bottleneck" hypothesis from the B6a debrief.

**Cons:**
- 1280×720 is 56% more pixels per frame than 1024×576. Chart-alignment may OOM at 12 frames; we have no data yet.
- If it OOMs, we drop to 11 or 10 frames at the higher resolution. Not catastrophic — the per-frame detail gain may compensate for one fewer view, but it's a confound on the experiment.
- Caveat: MAtCha's appendix says charts are downscaled internally to long-side ≤ 512. At 1280×720 the chart would be 512×288 (vs 512×288 already at 1024×576) — meaning **the chart resolution may not actually change**, only the photometric refinement input. Worth verifying in the code before running, or the experiment is testing a different variable than intended.

**My recommendation:** worth running, *but verify the internal-downscale behavior first* (T-013: read what's already there before assuming the knob does what you think it does).

**Decided (2026-05-01):** **on hold** until the code-read settles whether MAtCha's internal pipeline runs at the input resolution or at a downscaled ≤512 long-edge version. If the latter, A is testing nothing useful and should be dropped entirely.

## Option B — manual frame curation (B5), refined

**Original move:** build a contact-sheet picker (visual grid + click-to-pick UI), extract 60–100 candidate frames, hand-pick 12. Estimated ~2 hours of tool-building.

**Refined move (2026-05-01):** use **MASt3R-SfM standalone** to compute camera poses for the candidate pool, drop the cameras into Blender as Camera objects + textured image planes (extending the existing B3 tooling), hand-pick 12 by *spatial intuition* in 3D. **No mesh required.** No contact-sheet UI required. Full sketch in note `2026-05-01-spatial-frame-curation-via-mast3r-sfm`.

**Pros:**
- Higher leverage if it works — applies to all current and future scenes, not just one resolution regime.
- Tests a different hypothesis from A: that even-time-spacing is a bad sampling policy when the camera dwells in some places and moves fast in others.
- Aligns with how the MAtCha paper describes ideal sparse-view setups: views chosen for *coverage*, not *temporal uniformity*.
- The MASt3R-SfM-only step is fast (~1–2 min for 60 frames) and re-uses code already inside the krabby-matcha container.
- B3's existing `build_blender_scene.py` is most of the picker UI already.

**Cons:**
- Still requires a small amount of plumbing: a standalone MASt3R-SfM wrapper that stops after writing `cameras.json`, plus a B3 extension that accepts N cameras and skips mesh import. Total scope: a few hundred lines of Python.
- Per-scene curation time (~10 min/scene by hand in Blender). Probably tolerable.

**My recommendation:** **second priority after C.** Cost has dropped substantially with the SfM-cameras-in-Blender approach; this is now a meaningfully cheaper path than the original contact-sheet picker plan.

## Option C — chart-encoding resolution `r`

See sibling note `2026-05-01-chart-encoding-resolution-r-knob` for the full mechanism. In short: MAtCha has a per-chart 2D feature grid whose resolution `r` controls how locally vs globally the chart can deform. Paper uses `r=0.4` for unbounded 5–10 view scenes (our regime) and `r=0.1` for sparser DTU 3-view bounded scenes. Their rule: sparser SfM → lower r.

**Pros:**
- Free knob — no recapture, no VRAM impact, no extra runtime.
- Directly addresses the "noise from far things" complaint, because lower `r` makes MAtCha trust the per-image monodepth structure more and the SfM-driven deformation less. SfM is least reliable on distant background; reducing its per-pixel control there should reduce its capacity to inject distant-noise geometry.
- If the bottleneck is over-fitting to noisy SfM points (which is a plausible reading of "chaotic but recognizable"), this is the most direct intervention available.

**Cons:**
- Still untested; my prior is informed by the paper, not by our code's actual default.
- Need to first verify what `r` we're currently running with. May already be at 0.4; may be at something else; the train.py default is unknown to me right now.

**My recommendation:** verify default first, then run a small sweep `r ∈ {0.1, 0.2, 0.4}` on scene 004. Cheaper than A, smaller scope than B, addresses a different failure mode than either.

**Decided (2026-05-01):** **first priority.** Pursue C before B.

## What's NOT recommended ahead of A/B/C

- **B6 MAtCha-internal tuning** (gaussian-splat iters, TSDF thresholds) — too many knobs, low information per run.
- **Phase C** (USD export + IsaacSim) — accept current 12-frame baseline as good enough and move to milestone closeout. Only worth doing if Jeremy decides quality is acceptable and the milestone needs to ship.
- **AnyRecon** — already deferred.

## Decision log (2026-05-01)

Order of operations:

1. **C — `r` knob.** Verify the current default in `train.py` and the chart-deformation module. Then sweep `r ∈ {0.1, 0.2, 0.4}` on scene 004. Cheapest experiment we can run; addresses the over-fitting-to-noisy-SfM failure mode.
2. **B — refined manual curation.** Build the standalone-MASt3R-SfM wrapper, extend B3 to render cameras-without-mesh in Blender, run on 60 candidates from scene 004, hand-pick 12, run full MAtCha on those 12. Compare against the existing 12-frame baseline.
3. **A — higher resolution.** Only if a code-read confirms MAtCha's photometric stage uses the original-resolution input rather than a 512-downscaled version. Otherwise A is dropped.

Rationale for the order: C is the cheapest and most direct intervention against the most plausible failure mode (deformation grid over-fitting noisy SfM points). B is a known-good direction that requires a modest amount of plumbing. A may be testing nothing useful; resolve that question by reading code, not by running an experiment.
