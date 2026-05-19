# Capture lessons — 001 patio fisheye

## What we shot

- DJI Action 3, 4K hyperlapse, 30 fps, ~31 sec, 942 frames
- Native 155° fisheye
- Outdoor patio at a cabin
- Wide field of view captured both the patio (foreground) and **a lot of distant landscape** (background)

## What worked

- Hyperlapse capture profile produced enough viewpoint diversity for MAtCha to converge with 12 evenly-sampled frames
- Native fisheye (no dewarp) — reconstruction handled the distortion directly, consistent with the dewarp-dead-end finding from scene 002
- Camera path covered the patio area in a recognizable arc

## What didn't work

- **The 155° fisheye + outdoor scene captured too much "non-scene" geometry.** Trees in the distance, sky regions, and far-off ground all got reconstructed. They're not part of the M11 deliverable (the patio is — the trees aren't), and they pollute the mesh.
- **No reference object for scale calibration.** Scale of the output mesh is arbitrary.
- **Camera positions are not visible in the mesh** — there's no marker indicating where the operator walked.

## Lessons for the next time we capture this scene

1. **Reduce field-of-view bias toward the foreground.** Either:
   - Frame the camera more deliberately so the patio fills more of each frame
   - Crop the fisheye output before reconstruction to drop the outer (often distant) ring
   - Or just accept that we'll need a post-processing **bounding box / cull pass** (see PLAN.md "Post-processing requirements")
2. **Include a known-size reference object in the scene** — a meter stick, a printed checkerboard, a piece of paper with known dimensions. Critical for the M11 scale calibration in T2.
3. **Consider QR-code fiducials** at the boundary of "in-scope" geometry. If we mark the four corners of the area we care about, post-processing can auto-cull anything outside that boundary. (This is a project-wide capture protocol idea, not specific to scene 001 — see PLAN.md.)
4. **Filming pattern**: hyperlapse worked, but we got a single arc through the space. A figure-eight or "walking the perimeter then crossing diagonally" pattern would give better viewpoint coverage and likely improve mesh density in the interior of the captured region.

## Suggestions for re-shoot (if we're going back)

- Same DJI Action 3, native fisheye
- 2.7K @ 30fps locked exposure/WB (the validated profile from scene 004), NOT hyperlapse — gives MAtCha richer per-frame data and lets us cherry-pick the best 12-18 keyframes manually (per the planned frame-selection tool)
- Walk pattern: perimeter once (eye level) → perimeter once (low height) → diagonal cross
- Include reference object near center of patio
- 4 QR codes at the corners of the "scene boundary" if the QR-code-cull idea ships
- ~3-4 min of footage so we have ~18 viewpoint-diverse keyframes to choose from
