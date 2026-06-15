# T2 — View Selection

> Phase 3 of [the M11 process](README.md). Author the **1…N virtual cameras** that will be
> rendered for the comparison/runoff. These are *render viewpoints* — distinct from the
> "best-N frames for reconstruction" auto-selection (that's [T3a](T3a-reconstruction-preprocessing.md)).

## Inputs → Outputs

| In | Out |
|---|---|
| a scene `.blend` (or the scout surface) the operator can navigate | N named virtual cameras → `views/<slot>/view.json` + a canonical unified `cameras.json` the renderer (T3c) consumes |

## What "view selection" means here

Two different "selections" exist in M11 — keep them straight:

- **T2 view selection (this doc):** the operator picks **camera viewpoints** to render — the
  angles from which every reconstruction variant will be compared. A *creative/operator* call.
- **T3a frame/best-N selection:** an *automated* choice of which **source frames** feed the
  reconstruction (voxel coverage over the spine). Not this phase.

The same render views are applied to **every** variant, so the runoff (T4) compares like-for-like.

## Step — author the cameras, then register them

1. **Frame each view in Blender.** Open the scene `.blend` (MCP bridge), navigate
   (Shift+\` to walk) to the viewpoint you want.
2. **Capture it** with the `/camera-save <view-name> [purpose]` skill → `viewport_capture.py`
   captures the live viewport, deriving the **TRUE lens from the projection matrix** (not
   `space.lens`), writes the camera into the blend, and saves.
   - `purpose` defaults to `ab-comparison`; use `reference-match` for whitepaper-matching views.
   - Re-running with the same name **updates** the camera (re-frame + re-save is the loop).
3. **Register into the store (v4 path):**
   ```bash
   v4exec views-from-blend <scene> <blend>
   ```
   → emits `views/<slot>/view.json` per captured camera + the canonical unified cameras file
   the renderer reads. (This is the v4 successor to the legacy `sync_comparison_views.py`.)

## Settings / choices

- **Lens** is derived from the actual viewport projection — trust the capture, but **verify the
  screenshot** the skill returns matches what you framed (T-012); a stale Blender instance can
  hold the MCP port and land a capture in the wrong scene (seen 2026-06-10).
- **How many views (N):** enough to expose the geometry you care about (floor flatness, wall
  corners, far coverage). Each view multiplies the render+rank cost (variant × view).

## Gotchas

- **Verify the capture names the scene you meant** — confirm the returned screenshot + the
  `result.scene` field before moving on.
- Views are content in the store like everything else — author via the skill/`views-from-blend`,
  don't hand-edit `views/*/view.json` (T-025, `v4exec` is the sole writer).

## Automation status

Capture is operator-driven by design (it's a creative choice); registration is one command.
🟡 operator-in-the-loop (intentionally).

## Next

→ [T3a — Reconstruction Pre-Processing](T3a-reconstruction-preprocessing.md)
