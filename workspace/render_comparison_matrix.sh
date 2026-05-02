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
#                                [--views "view1 view2"]
#
# Defaults: scene=004-sky-house-dining, variants=auto-discover all
# 004-sky-house-curated-* dirs, views=all from comparison_views.json.

set -euo pipefail

# --- defaults ---
RENDER_ENGINE=BLENDER_WORKBENCH
WIDTH=1920
HEIGHT=1080
SCENE="004-sky-house-dining"
VARIANTS=""
VIEWS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --render-engine) RENDER_ENGINE="$2"; shift 2 ;;
        --width)         WIDTH="$2"; shift 2 ;;
        --height)        HEIGHT="$2"; shift 2 ;;
        --scene)         SCENE="$2"; shift 2 ;;
        --variants)      VARIANTS="$2"; shift 2 ;;
        --views)         VIEWS="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -24 | sed 's/^# //; s/^#$//'
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

WORKSPACE=$(dirname "$(realpath "$0")")
LB=/private/var/krabby/workspace/milestones/011-scene-reconstruction/data/scenes
VIEWS_JSON=$LB/$SCENE/comparison_views.json
RENDER_ROOT=$LB/$SCENE/comparison_renders
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender

if [ ! -f "$VIEWS_JSON" ]; then
    echo "ERROR: $VIEWS_JSON not found"; exit 1
fi

# --- resolve variants ---
if [ -z "$VARIANTS" ]; then
    VARIANTS=$(ls -d $LB/${SCENE%-*}-curated-* 2>/dev/null | xargs -n1 basename | sed "s/^004-sky-house-curated-//" | tr '\n' ' ')
fi
echo "Variants: $VARIANTS"

# --- resolve views ---
if [ -z "$VIEWS" ]; then
    VIEWS=$(python3 -c "import json; d=json.load(open('$VIEWS_JSON')); print(' '.join(v['name'] for v in d['views']))")
fi
echo "Views:    $VIEWS"
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
    SD=$LB/${SCENE%-*}-curated-$V
    if [ ! -d "$SD/oriented" ]; then
        echo "  SKIP variant $V (no oriented/ dir)"
        continue
    fi
    for VIEW in $VIEWS; do
        DONE=$((DONE + 1))
        OUT_DIR=$RENDER_ROOT/$VIEW
        mkdir -p "$OUT_DIR"
        OUT_PNG=$OUT_DIR/$V.png
        echo "[$DONE/$TOTAL] $V × $VIEW → $OUT_PNG"
        $BLENDER --background --python $WORKSPACE/build_blender_scene.py -- \
            --mesh "$SD/oriented/oriented_500k_colored_culled.ply" \
            --cameras-original "$SD/mast3r_sfm/cameras.json" \
            --cameras-oriented "$SD/oriented/oriented_cameras.json" \
            --frames-dir "$SD/mast3r_sfm/images" \
            --output "$SD/oriented/scene_culled.blend" \
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
