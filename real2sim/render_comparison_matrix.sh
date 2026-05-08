#!/bin/bash
# Render the Cartesian product of (variant × view) → one PNG per cell.
# Output structure:
#   <render_root>/<view_name>/<variant>.png
# This makes per-variant comparison easy (open one view dir → all variants
# from that angle) AND per-view comparison easy (look at one variant across
# all angles).
#
# Usage:
#   render_comparison_matrix.sh [--render-engine BLENDER_WORKBENCH] \
#                                [--width 1920 --height 1080] \
#                                [--scene 004-sky-house-dining] \
#                                [--variants "12 12-strong 16-strong"] \
#                                [--views "view1 view2"] \
#                                [--purpose ab-comparison]
#
# Defaults: scene=004-sky-house-dining, variants=auto-discover all
# 004-sky-house-curated-* dirs, views=all ab-comparison views from
# comparison_views.json (reference-match views are excluded by default —
# pass --purpose any to include all, or --purpose reference-match to render
# only those).

set -euo pipefail

# --- defaults ---
RENDER_ENGINE=BLENDER_WORKBENCH
WIDTH=1920
HEIGHT=1080
SCENE="004-sky-house-dining"
VARIANTS=""
VIEWS=""
VARIANT_PREFIX=""    # auto-resolved from comparison_views.json if absent
MESH_SOURCE=oriented # 'oriented' (tetra→cull→color) or 'tsdf' (multi-res TSDF)
PURPOSE_FILTER="ab-comparison"  # filter views by `purpose` field; "any" disables

while [[ $# -gt 0 ]]; do
    case "$1" in
        --render-engine)   RENDER_ENGINE="$2"; shift 2 ;;
        --width)           WIDTH="$2"; shift 2 ;;
        --height)          HEIGHT="$2"; shift 2 ;;
        --scene)           SCENE="$2"; shift 2 ;;
        --variants)        VARIANTS="$2"; shift 2 ;;
        --views)           VIEWS="$2"; shift 2 ;;
        --variant-prefix)  VARIANT_PREFIX="$2"; shift 2 ;;
        --mesh-source)     MESH_SOURCE="$2"; shift 2 ;;
        --purpose)         PURPOSE_FILTER="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -24 | sed 's/^# //; s/^#$//'
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve mesh-source-specific paths
case "$MESH_SOURCE" in
    oriented)
        MESH_RELPATH="oriented/oriented_500k_colored_culled.ply"
        BLEND_RELPATH="oriented/scene_culled.blend"
        ;;
    tsdf)
        MESH_RELPATH="tsdf_meshes/multires_tsdf_post_oriented.ply"
        BLEND_RELPATH="tsdf_meshes/scene_tsdf.blend"
        ;;
    *)
        echo "ERROR: --mesh-source must be 'oriented' or 'tsdf' (got '$MESH_SOURCE')"
        exit 1
        ;;
esac

WORKSPACE=$(dirname "$(realpath "$0")")
LB=/private/var/krabby/workspace/milestones/011-scene-reconstruction/data/scenes
VIEWS_JSON=$LB/$SCENE/comparison_views.json
RENDER_ROOT=$LB/$SCENE/comparison_renders
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender

if [ ! -f "$VIEWS_JSON" ]; then
    echo "ERROR: $VIEWS_JSON not found"; exit 1
fi

# --- resolve variant prefix from JSON (allows --variant-prefix override) ---
if [ -z "$VARIANT_PREFIX" ]; then
    VARIANT_PREFIX=$(python3 -c "
import json, sys
d = json.load(open('$VIEWS_JSON'))
print(d.get('variant_prefix', ''))" 2>/dev/null)
fi
if [ -z "$VARIANT_PREFIX" ]; then
    echo "ERROR: no variant_prefix in $VIEWS_JSON and --variant-prefix not given"
    echo "       (add a 'variant_prefix' field to the JSON, e.g. 'dtu-bicycle')"
    exit 1
fi
echo "Variant prefix: $VARIANT_PREFIX"

# --- resolve variants ---
if [ -z "$VARIANTS" ]; then
    VARIANTS=$(ls -d $LB/$VARIANT_PREFIX-curated-* 2>/dev/null | xargs -n1 basename | sed "s/^${VARIANT_PREFIX}-curated-//" | tr '\n' ' ')
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
    SD=$LB/$VARIANT_PREFIX-curated-$V
    MESH_PATH=$SD/$MESH_RELPATH
    BLEND_PATH=$SD/$BLEND_RELPATH
    if [ ! -f "$MESH_PATH" ]; then
        echo "  SKIP variant $V (no $MESH_RELPATH)"
        # Still increment DONE for the skipped views
        for VIEW in $VIEWS; do DONE=$((DONE + 1)); done
        continue
    fi
    for VIEW in $VIEWS; do
        DONE=$((DONE + 1))
        OUT_DIR=$RENDER_ROOT/$VIEW
        mkdir -p "$OUT_DIR"
        OUT_PNG=$OUT_DIR/$V.png
        echo "[$DONE/$TOTAL] $V × $VIEW → $OUT_PNG"
        $BLENDER --background --python $WORKSPACE/build_blender_scene.py -- \
            --mesh "$MESH_PATH" \
            --cameras-original "$SD/mast3r_sfm/cameras.json" \
            --cameras-oriented "$SD/oriented/oriented_cameras.json" \
            --frames-dir "$SD/mast3r_sfm/images" \
            --output "$BLEND_PATH" \
            --view-camera-pose "$VIEWS_JSON" \
            --view-name "$VIEW" \
            --render-output "$OUT_PNG" \
            --render-width "$WIDTH" \
            --render-height "$HEIGHT" \
            --render-engine "$RENDER_ENGINE" 2>&1 | grep -E '^Saved|residuals|using view' | head -3
    done
done

echo
echo "Done. Matrix at $RENDER_ROOT/"
echo
echo "Per-view directories (variants side-by-side from same angle):"
for VIEW in $VIEWS; do
    echo "  $RENDER_ROOT/$VIEW/  ($(ls $RENDER_ROOT/$VIEW 2>/dev/null | wc -l | tr -d ' ') variants)"
done
