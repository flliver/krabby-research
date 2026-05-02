#!/bin/bash
# Run B1-B4 post-processing on a variant inside the matcha-build container.
# Mirrors the existing variants' oriented/ layout (500K decimation target).
#
# Usage (on tbeeprz):
#   bash scripts/post_process.sh 12-dense-strong-r3
#
# Expects this script to live in <experiment-root>/scripts/ alongside
# ../../../workspace/{orient_mesh,project_color,cull_mesh}.py
# (i.e. the milestone's `workspace/` dir).
set -euo pipefail

VARIANT="${1:?need variant suffix, e.g. 12-dense-strong-r3}"
CONTAINER=matcha-build
SCENE_DIR=/data/scenes/004-sky-house-curated-$VARIANT
ORIENTED_DIR=$SCENE_DIR/oriented
TETRA_PLY=$SCENE_DIR/tetra_meshes/tetra_mesh_binary_search_7.ply
CAMERAS=$SCENE_DIR/mast3r_sfm/cameras.json
IMAGES_DIR=$SCENE_DIR/mast3r_sfm/images
DECIMATE_TARGET=500000   # match existing variants' oriented_500k naming

# Resolve workspace scripts. Uses $WORKSPACE_ROOT if set; otherwise tries
# the standard milestone layout (../../workspace), then ~/work/workspace
# (the rsync mirror layout used on tbeeprz).
HERE=$(dirname "$(realpath "$0")")
EXP_ROOT=$(realpath "$HERE/..")
if [ -n "${WORKSPACE_ROOT:-}" ]; then
    :   # honour env override
elif [ -d "$EXP_ROOT/../../workspace" ]; then
    WORKSPACE_ROOT=$(realpath "$EXP_ROOT/../../workspace")
elif [ -d "$HOME/work/workspace" ]; then
    WORKSPACE_ROOT="$HOME/work/workspace"
else
    echo "ERROR: cannot find workspace dir with orient_mesh.py + friends"
    echo "       (set WORKSPACE_ROOT, or place at ../../workspace, or ~/work/workspace)"
    exit 1
fi

[ -f "$WORKSPACE_ROOT/orient_mesh.py" ] || {
    echo "ERROR: workspace/orient_mesh.py not found at $WORKSPACE_ROOT"
    exit 1
}

# ---- 1. push scripts into container ---------------------------------------
echo "=== push post-processing scripts to $CONTAINER ==="
docker exec "$CONTAINER" mkdir -p /tmp/post-proc
for SCRIPT in orient_mesh.py project_color.py cull_mesh.py; do
    docker cp "$WORKSPACE_ROOT/$SCRIPT" "$CONTAINER:/tmp/post-proc/$SCRIPT"
done
echo "  ok"

# ---- 2. run B1 (orient) ----------------------------------------------------
echo
echo "=== B1: orient_mesh ==="
# $ORIENTED_DIR is a container-internal path (/data/...) — only mkdir inside.
docker exec "$CONTAINER" mkdir -p "$ORIENTED_DIR"

START=$(date +%s)
docker exec "$CONTAINER" bash -c "
    source /opt/matcha/bin/activate
    python /tmp/post-proc/orient_mesh.py \
        --tetra '$TETRA_PLY' \
        --cameras '$CAMERAS' \
        --output '$ORIENTED_DIR'
"
echo "  ($(( $(date +%s) - START ))s)"

# ---- 3. inline decimate (B1 + decimate were separate stages) ---------------
echo
echo "=== decimate to ${DECIMATE_TARGET} triangles ==="
START=$(date +%s)
docker exec "$CONTAINER" bash -c "
    source /opt/matcha/bin/activate
    python -c \"
import open3d as o3d, os, time
src = '$ORIENTED_DIR/oriented_tetra.ply'
target = $DECIMATE_TARGET
mesh = o3d.io.read_triangle_mesh(src)
print(f'  in: {len(mesh.vertices):,}v / {len(mesh.triangles):,}t')
t0 = time.time()
dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
dec.remove_degenerate_triangles()
dec.remove_unreferenced_vertices()
dec.remove_duplicated_triangles()
dec.compute_vertex_normals()
out_obj = '$ORIENTED_DIR/oriented_{0}k.obj'.format(target // 1000)
out_ply = '$ORIENTED_DIR/oriented_{0}k.ply'.format(target // 1000)
o3d.io.write_triangle_mesh(out_obj, dec)
o3d.io.write_triangle_mesh(out_ply, dec)
print(f'  out: {len(dec.vertices):,}v / {len(dec.triangles):,}t in {time.time()-t0:.0f}s')
\"
"
echo "  ($(( $(date +%s) - START ))s)"

# ---- 4. B4 project color ---------------------------------------------------
echo
echo "=== B4: project_color ==="
DEC_PLY=$ORIENTED_DIR/oriented_$((DECIMATE_TARGET / 1000))k.ply
ORIENTED_CAMS=$ORIENTED_DIR/oriented_cameras.json
COLORED_PLY=$ORIENTED_DIR/oriented_$((DECIMATE_TARGET / 1000))k_colored.ply

START=$(date +%s)
docker exec "$CONTAINER" bash -c "
    source /opt/matcha/bin/activate
    python /tmp/post-proc/project_color.py \
        --mesh '$DEC_PLY' \
        --cameras '$CAMERAS' \
        --oriented-cameras '$ORIENTED_CAMS' \
        --images '$IMAGES_DIR' \
        --output '$COLORED_PLY'
"
echo "  ($(( $(date +%s) - START ))s)"

# ---- 5. B2 cull -----------------------------------------------------------
echo
echo "=== B2: cull_mesh ==="
CULLED_PLY=$ORIENTED_DIR/oriented_$((DECIMATE_TARGET / 1000))k_colored_culled.ply

START=$(date +%s)
docker exec "$CONTAINER" bash -c "
    source /opt/matcha/bin/activate
    python /tmp/post-proc/cull_mesh.py \
        --mesh '$COLORED_PLY' \
        --cameras '$CAMERAS' \
        --oriented-cameras '$ORIENTED_CAMS' \
        --output '$CULLED_PLY'
"
echo "  ($(( $(date +%s) - START ))s)"

# ---- 6. summary -----------------------------------------------------------
echo
echo "=== final layout ==="
docker exec "$CONTAINER" ls -la "$ORIENTED_DIR" 2>&1
echo
echo "  ✓ done."
