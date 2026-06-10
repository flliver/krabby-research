#!/bin/bash
# Render the Cartesian product of (variant × view) → one PNG per cell.
# Output structure:
#   <render_root>/<view_name>/<variant>.png
# This makes per-variant comparison easy (open one view dir → all variants
# from that angle) AND per-view comparison easy (look at one variant across
# all angles).
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
RENDER_ROOT=$SCENE_DIR/comparison_renders
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
echo "Output:   $RENDER_ROOT/<view>/<variant>.png"
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
        OUT_DIR=$RENDER_ROOT/$VIEW
        mkdir -p "$OUT_DIR"
        OUT_PNG=$OUT_DIR/$V.png
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
    done
    rm -f "$BLEND_PATH"
done

echo
echo "Done. Matrix at $RENDER_ROOT/"
echo
echo "Per-view directories (variants side-by-side from same angle):"
for VIEW in $VIEWS; do
    echo "  $RENDER_ROOT/$VIEW/  ($(ls $RENDER_ROOT/$VIEW 2>/dev/null | wc -l | tr -d ' ') variants)"
done
