#!/bin/bash
# Render the Cartesian product of (variant × view) → one PNG per cell.
# Output structure (STO-SCN-058 — the render belongs to the RUN that
# produced it; what the runoff compares is the pipeline configuration):
#   <scene>/pipeline-<p>/run-<r>/renders/<view>.png   the render
#   <scene>/pipeline-<p>/run-<r>/renders/<view>.json  settings sidecar
# Cross-variant comparison (one view, all variants) is an AGGREGATION
# (rate_renders does it at read time) — not a storage layout.
#
# STO-SCN-045: operates on the scene store (scenes/<scene>/), where a
# "variant" is a pipeline run: pipeline-<p>/run-<r> → variant label
# "<p>--<r>" (e.g. matcha--12-dense-strong). Views come from the unified
# scene-level cameras.json (schema 5) — required; legacy layouts are not
# supported (migrate via sync_comparison_views.py).
#
# Usage:
#   render_comparison_matrix.sh [--scene dtu-bicycle] \
#                                [--scenes-root /var/krabby/scenes] \
#                                [--render-engine BLENDER_WORKBENCH] \
#                                [--width 1920 --height 1080] \
#                                [--variants "matcha--12-strong matcha--16-strong"] \
#                                [--views "view1 view2"] \
#                                [--mesh-source oriented|tsdf] \
#                                [--purpose ab-comparison]
#
# Defaults: variants=auto-discover all pipeline-*/run-* with the requested
# mesh present; views=all ab-comparison views from cameras.json
# (reference-match views are excluded by default — pass --purpose any to
# include all, or --purpose reference-match to render only those).

set -euo pipefail

# --- defaults ---
RENDER_ENGINE=BLENDER_WORKBENCH
WIDTH=1920
HEIGHT=1080
SCENE="dtu-bicycle"
SCENES_ROOT="${KRABBY_SCENES_ROOT:-/var/krabby/scenes}"
VARIANTS=""
VIEWS=""
MESH_SOURCE=oriented # 'oriented' (tetra→cull→color) or 'tsdf' (multi-res TSDF)
PURPOSE_FILTER="ab-comparison"  # filter views by `purpose` field; "any" disables

while [[ $# -gt 0 ]]; do
    case "$1" in
        --render-engine)   RENDER_ENGINE="$2"; shift 2 ;;
        --width)           WIDTH="$2"; shift 2 ;;
        --height)          HEIGHT="$2"; shift 2 ;;
        --scene)           SCENE="$2"; shift 2 ;;
        --scenes-root)     SCENES_ROOT="$2"; shift 2 ;;
        --variants)        VARIANTS="$2"; shift 2 ;;
        --views)           VIEWS="$2"; shift 2 ;;
        --mesh-source)     MESH_SOURCE="$2"; shift 2 ;;
        --purpose)         PURPOSE_FILTER="$2"; shift 2 ;;
        -h|--help)
            head -29 "$0" | tail -28 | sed 's/^# //; s/^#$//'
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve mesh-source-specific relpaths (inside transform-NN-*/data/)
case "$MESH_SOURCE" in
    oriented)
        MESH_RELPATH="oriented/oriented_500k_colored_culled.ply"
        ;;
    tsdf)
        MESH_RELPATH="tsdf_meshes/multires_tsdf_post_oriented.ply"
        ;;
    *)
        echo "ERROR: --mesh-source must be 'oriented' or 'tsdf' (got '$MESH_SOURCE')"
        exit 1
        ;;
esac

WORKSPACE=$(dirname "$(realpath "$0")")
SCENE_DIR=$SCENES_ROOT/$SCENE
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender

# --- resolve views file: unified cameras.json (schema 5) only ---
VIEWS_JSON=$SCENE_DIR/cameras.json
if [ ! -f "$VIEWS_JSON" ]; then
    echo "ERROR: no $VIEWS_JSON — migrate the scene first:"
    echo "  Blender --background --python sync_comparison_views.py -- \\"
    echo "      <run>/scene.blend <run>/.../mast3r_sfm/cameras.json $VIEWS_JSON"
    exit 1
fi
echo "Views from: $VIEWS_JSON"

# --- resolve variants: every pipeline-*/run-* with the requested data ---
# Variant label: "<pipeline>--<run>" minus prefixes, e.g.
# pipeline-matcha/run-12-strong → matcha--12-strong.
variant_to_rundir() {  # matcha--12-strong → pipeline-matcha/run-12-strong
    local v="$1"
    echo "pipeline-${v%%--*}/run-${v#*--}"
}

if [ -z "$VARIANTS" ]; then
    VARIANTS=""
    for RD in "$SCENE_DIR"/pipeline-*/run-*/; do
        [ -d "$RD" ] || continue
        # transform dir glob: data lives at transform-NN-*/data/
        TD=$(ls -d "$RD"transform-*/data 2>/dev/null | head -1) || true
        [ -n "${TD:-}" ] || continue
        [ -f "$TD/$MESH_RELPATH" ] || continue
        P=$(basename "$(dirname "${RD%/}")"); P=${P#pipeline-}
        R=$(basename "${RD%/}"); R=${R#run-}
        VARIANTS="$VARIANTS $P--$R"
    done
    VARIANTS=$(echo $VARIANTS)  # trim
fi
if [ -z "$VARIANTS" ]; then
    echo "ERROR: no variants found with $MESH_RELPATH under $SCENE_DIR/pipeline-*/run-*/"
    exit 1
fi
echo "Variants: $VARIANTS"

# --- resolve views (with optional purpose filter) ---
# Default: only render views with purpose='ab-comparison' (skip reference-match,
# capture-spine, etc.). Pass --purpose any to render every view in the JSON, or
# --purpose <other> to filter to a different purpose.
# Backward compat: views without an explicit `purpose` field are treated as
# 'ab-comparison' (matches schema v3 → v4 migration semantics).
if [ -z "$VIEWS" ]; then
    VIEWS=$(python3 -c "
import json
d = json.load(open('$VIEWS_JSON'))
filt = '$PURPOSE_FILTER'
def kept(v):
    if filt == 'any':
        return True
    return v.get('purpose', 'ab-comparison') == filt
print(' '.join(v['name'] for v in d['views'] if kept(v)))
")
fi
if [ -z "$VIEWS" ]; then
    echo "ERROR: no views match purpose filter '$PURPOSE_FILTER' in $VIEWS_JSON"
    exit 1
fi
echo "Views:    $VIEWS  (purpose filter: $PURPOSE_FILTER)"
echo "Render:   ${WIDTH}×${HEIGHT}, engine=$RENDER_ENGINE"
echo "Output:   $SCENE_DIR/pipeline-<p>/run-<r>/renders/<view>.png (+ .json sidecar)"
echo

# --- the matrix ---
TOTAL=0
DONE=0
for V in $VARIANTS; do
    for VIEW in $VIEWS; do
        TOTAL=$((TOTAL + 1))
    done
done

for V in $VARIANTS; do
    RUNDIR=$SCENE_DIR/$(variant_to_rundir "$V")
    TD=$(ls -d "$RUNDIR"/transform-*/data 2>/dev/null | head -1) || true
    if [ -z "${TD:-}" ] || [ ! -f "$TD/$MESH_RELPATH" ]; then
        echo "  SKIP variant $V (no transform data / $MESH_RELPATH)"
        for VIEW in $VIEWS; do DONE=$((DONE + 1)); done
        continue
    fi
    # Frames dir: prefer the run's own SfM images; fall back to scene input
    FRAMES_DIR="$TD/mast3r_sfm/images"
    if [ ! -d "$FRAMES_DIR" ]; then
        FRAMES_DIR=$(ls -d "$SCENE_DIR"/input/preproc-*/data 2>/dev/null | head -1) || true
    fi
    # Scratch .blend per variant — render artifact, not the canonical
    # run-dir scene.blend (which STO-SCN-047 owns).
    BLEND_PATH=$RUNDIR/matrix_render.blend
    for VIEW in $VIEWS; do
        DONE=$((DONE + 1))
        OUT_DIR=$RUNDIR/renders
        mkdir -p "$OUT_DIR"
        OUT_PNG=$OUT_DIR/$VIEW.png
        echo "[$DONE/$TOTAL] $V × $VIEW → $OUT_PNG"
        $BLENDER --background --python $WORKSPACE/build_blender_scene.py -- \
            --mesh "$TD/$MESH_RELPATH" \
            --cameras-original "$TD/mast3r_sfm/cameras.json" \
            --cameras-oriented "$TD/oriented/oriented_cameras.json" \
            ${FRAMES_DIR:+--frames-dir "$FRAMES_DIR"} \
            --output "$BLEND_PATH" \
            --view-camera-pose "$VIEWS_JSON" \
            --view-name "$VIEW" \
            --render-output "$OUT_PNG" \
            --render-width "$WIDTH" \
            --render-height "$HEIGHT" \
            --render-engine "$RENDER_ENGINE" 2>&1 | grep -E '^Saved|residuals|rendered|ERROR' | head -3
        # settings sidecar — the render is an artifact OF this run's
        # configuration; record exactly what produced it (STO-SCN-058)
        python3 - "$RUNDIR" "$VIEW" "$VIEWS_JSON" <<SIDECAR
import json, sys, datetime
from pathlib import Path
rundir, view, views_json = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
views = json.load(open(views_json))
vcam = next((v for v in views.get("views", []) if v["name"] == view), None)
params = {}
for tdir in sorted(rundir.glob("transform-*")):
    sp = tdir / "specification.json"
    if sp.is_file():
        try: params[tdir.name] = json.load(open(sp)).get("parameters", {})
        except ValueError: params[tdir.name] = {"error": "spec unreadable"}
side = {
  "schema_version": "1",
  "view": view,
  "view_camera": vcam,
  "render": {"engine": "$RENDER_ENGINE", "width": $WIDTH, "height": $HEIGHT,
             "mesh_source": "$MESH_SOURCE", "mesh_relpath": "$MESH_RELPATH",
             "provenance": "measured",
             "rendered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")},
  "produced_by": {"pipeline": "${V%%--*}", "run": "${V#*--}",
                  "transform_parameters": params},
}
(rundir / "renders" / f"{view}.json").write_text(json.dumps(side, indent=2) + "\n")
SIDECAR
    done
    rm -f "$BLEND_PATH"
done

echo
echo "Done. Renders live in each run: pipeline-<p>/run-<r>/renders/<view>.png"
echo "Cross-variant comparison: rate_renders aggregates at read time (:8090)."
