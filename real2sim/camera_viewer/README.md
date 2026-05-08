# Camera Selection Viewer (Route B)

A web-based 3D viewer for hand-curating which video frames to feed into a MAtCha mesh reconstruction. Built on [viser](https://github.com/nerfstudio-project/viser) (the Nerfstudio team's Python WebGL viewer).

**Status:** in design / v0 skeleton

## Why this exists

The B5 plan says: extract a candidate pool of N frames from a source video, run MASt3R-SfM to recover camera poses, then hand-pick 12 for the actual MAtCha run. With pools of 60–400 candidate cameras (see the SfM-scaling experiment), a flat list is unworkable. The viewer turns the candidate set into a 3D scene with composable filters and click-to-pick selection.

Full feasibility analysis: `milestones/011-scene-reconstruction/journal/journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T153502-camera-selection-ui-feasibility/note.md`. This implementation is **Route B** from that note.

## Design goals

1. **Loadable from a `cameras.json` produced by MASt3R-SfM** — no preprocessing, no separate pose file. The `--sfm_only` output is the input.
2. **Visualize the camera path in 3D** — frustums + textured image planes + temporal polyline. The viewer's job is to make the spatial structure of the candidate set obvious.
3. **Compose filters that are independent booleans** — a camera is visible iff *all* active filters say it should be. Filters never fight each other.
4. **Click-to-pick** — single click toggles a camera between unpicked / picked. Selection persists across filter changes.
5. **Output a `selected_frames.json`** — the file MAtCha consumes via `--image_idx`. End-to-end repro path: pool → SfM → viewer → `--image_idx` → MAtCha.

## Non-goals

- Replacing Blender. Blender's already great for one-off inspection; this is for *curation* specifically. Use whichever you prefer for the corresponding job.
- Mesh viewing. The whole point is no mesh required (we don't have one yet — that's what we're curating to produce).
- Viewing point clouds. The `points.ply` from SfM is sparse, noisy, and not useful for picking. We *might* add it as an optional layer later if it helps localize cameras spatially; v0 doesn't.

## Inputs

```
cameras.json    # MASt3R-SfM output
{
  "filepaths":  [str × N],     # absolute paths to source frames
  "focals":     [float × N],   # one per frame
  "cams2world": [4×4 × N]      # cam-to-world transform per frame
}

frames/         # directory containing the source JPEGs
                # filepaths in cameras.json must resolve under here
                # (or be absolute and reachable as-is)
```

The schema is bare-minimum on purpose — no principal points, no distortion, no per-camera image sizes. Per the unposed.yaml MASt3R config, principal points are pinned to image center; image sizes are read from the JPEG headers.

## Outputs

```
selected_frames.json
{
  "source_pool":   "/path/to/frames",          # which pool these indices reference
  "n_pool":        N,                           # total candidate count
  "selected_idx":  [int, ...],                  # zero-based indices into filepaths
  "selected_at":   "2026-05-01T17:50:00-07:00" # capture timestamp
}
```

This file plugs directly into MAtCha:

```bash
python train.py -s frames/ --image_idx $(jq -r '.selected_idx | @sh' selected_frames.json)
```

## Architecture

```
viewer.py            # main entry, argparse, viser server lifecycle
├── data.py          # cameras.json loader, image-plane texture loader
├── filters.py       # filter composition + selection state
├── clustering.py    # k-means on positions (v0); co-visibility from SfM graph (later)
└── ui.py            # GUI panel composition (sliders, checkboxes, counter, save button)
```

Each module has one job, one piece of state. The `viewer.py` entry point composes them; modules don't know about each other.

### Filter composition

A camera is visible iff `all(f.passes(cam_idx) for f in active_filters)`. Filters are independent classes implementing:

```python
class Filter(Protocol):
    name: str                 # for the GUI label
    def passes(self, cam_idx: int) -> bool: ...
    def reset(self) -> None: ...
```

Filter implementations (in priority order for v0):

| Filter | UI control | v0 status |
|--------|------------|-----------|
| `TimeRangeFilter` | dual-handle slider over frame index | v0 ✓ |
| `PickedStatusFilter` | tri-state: show all / show picked only / show unpicked only | v0 ✓ |
| `SpatialClusterFilter` | checkboxes (k-means clusters on 3D position) | v0 ✓ |
| `ViewDirectionFilter` | 6-way checkboxes (front/back/up/down/left/right of scene centroid) | v1 |
| `ImageSimilarityFilter` | ASMK-cluster checkboxes | v2 (requires running ASMK first) |
| `CovisibilityFilter` | scene-graph cluster checkboxes | v2 (requires SfM graph data) |

### Selection state

A flat dict-of-bool mapped by camera index. Click toggles. Filters don't mutate selection — they just hide/show. Selection persists across filter changes so users can build a selection iteratively.

## Click semantics (v0)

| Event | Action |
|-------|--------|
| Single-click frustum | toggle picked status (writes to selection dict) |
| Hover over frustum | tooltip with frame index + small thumbnail (deferred to v1) |
| Double-click frustum | fly viewer camera to that frustum's view |
| "Save selection" GUI button | write `selected_frames.json`, print path to console |

## Performance considerations

For ~60 cameras: trivial.

For ~200–400 cameras: image-plane texture memory is the concern. Each 1024×576 JPEG decodes to ~2 MB raw RGB in browser memory. 400 × 2 MB = 800 MB. WebGL handles it on modern GPUs but it's noticeable. **Mitigation built into v0:** image-plane textures are downscaled to 512 long-edge before sending to viser. Originals stay on disk for the actual MAtCha run. Cuts client-side texture memory by 4×.

## Dependencies

- Python ≥ 3.10 (viser requirement)
- viser
- numpy
- pillow (for image loading + downscale)
- scikit-learn (for k-means clustering)

`requirements.txt` pins versions known to work.

## How to run

```bash
# From the camera_viewer directory
python -m viewer \
  --cameras /path/to/cameras.json \
  --frames /path/to/frames/ \
  --output selected_frames.json \
  --port 8080

# Open http://localhost:8080 (or the host's IP if running remotely)
# Pick frames, click "Save selection", quit.
```

If the cameras.json was produced on bbeeprz/tbeeprz, rsync it + the frames dir to wherever you're running the viewer (likely JDP-Mac for browser access). Or run viser on bbeeprz and tunnel: `ssh -L 8080:localhost:8080 bbeeprz`.

## Open questions

1. **Is viser the right choice?** Alternatives: Nerfstudio's built-in viewer (more complex to repurpose), a Three.js-from-scratch app (more work), gradio + plotly 3D (less interactive). Decision: viser, per the feasibility note. Will revisit if v0 reveals API limitations.

2. **Should frustum textures be the source images or downscaled previews?** v0 uses 512-long-edge downscales for texture memory. Eyeball test once running: are the previews readable enough to inform selection decisions, or do we need full-res?

3. **Where should the viewer run?** v0 supports either local (JDP-Mac) or remote (bbeeprz/tbeeprz). The `--frames` arg accepts any path the host can reach. Cross-host browser access via SSH tunnel works.

4. **Persistent vs ephemeral selection?** v0 saves on button click only — not autosave. If the user closes the browser before saving, selection is lost. Worth adding autosave-to-temp-file in v1 if this bites in practice.

## Roadmap

- **v0 (this commit):** skeleton + working frustums + time-range filter + picked-status filter + spatial-cluster filter + click-to-pick + save button.
- **v1:** view-direction filter, hover thumbnails, autosave, double-click fly-to.
- **v2:** ASMK + co-visibility filters (require precomputed graph data — interface decision pending).
- **v3 (if ever):** mesh + point-cloud overlays, persistence across re-launches.

## Status (right now)

v0 design landed. Skeleton code follows in `viewer.py`, `data.py`, `filters.py`, `ui.py`, `clustering.py`. Tested against the cameras.json from the SfM-scaling experiment's N=60 run.
