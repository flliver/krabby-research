#!/bin/bash
# Canonical wrapper template for MAtCha + post-processing runs.
# Demonstrates the script-driven progress reporting pattern.
#
# Pattern: the script sources ~/lib_progress.sh and calls progress_set
# at each phase boundary. No outside actor needs to poll. progress_clear
# fires automatically via the EXIT trap installed by progress_init —
# regardless of how the script ends (success, error, Ctrl-C, OOM).
#
# Why this pattern matters:
#   - Phase boundaries are KNOWN by the script. Polling guesses them.
#   - The EXIT trap guarantees the dashboard never holds stale state.
#   - Out-of-band `nanny-progress set …` pushes are fragile (race with
#     real phase transitions, get overwritten, or get left behind on crash).
#
# Customize: edit the docker-exec block + the phase labels for the
# specific pipeline you're running. The skeleton below is a curated MAtCha
# end-to-end run (full pipeline + B1-B4 post-processing).

set -uo pipefail
source ~/lib_progress.sh

# ---- arguments / paths ----
VARIANT="${1:?usage: run_template.sh <variant-tag> <frames-dir>}"
FRAMES_DIR="${2:?usage: run_template.sh <variant-tag> <frames-dir>}"
ALIGNMENT_CONFIG="${3:-strong}"

SCENE_DIR="/data/scenes/004-sky-house-curated-$VARIANT"
HOST_OUT="$HOME/outposts/krabby/data/011-scene-reconstruction/scenes/004-sky-house-curated-$VARIANT"
LOG_MATCHA="/tmp/run-curated-$VARIANT.log"
LOG_POST="/tmp/postprocess-curated-$VARIANT.log"

# 5 phases at the wrapper level: train.py is one big phase (its internals
# are not addressable from here), then 4 post-processing phases.
progress_init 5

# ---- Phase 1: full MAtCha pipeline (SfM → align → refine → tetra) ----
progress_set 1 0 "MAtCha full pipeline"

T0=$(date +%s)
docker exec matcha-build bash -c "
  source /opt/matcha/bin/activate
  export PYTHONPATH=/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd /opt/MAtCha
  python train.py \
    -s $FRAMES_DIR \
    -o $SCENE_DIR \
    --alignment_config $ALIGNMENT_CONFIG \
    --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
    --depthanything_encoder vitl 2>&1
" > "$LOG_MATCHA" 2>&1
RC=$?
T1=$(date +%s)
echo "[matcha] RC=$RC wall=$((T1-T0))s"

if [ $RC -ne 0 ] || [ ! -f "$HOST_OUT/tetra_meshes/tetra_mesh_binary_search_7.ply" ]; then
  echo "MAtCha failed; bailing on post-processing."
  tail -10 "$LOG_MATCHA"
  exit 1
fi

# ---- Phase 2: B1 orient + decimate ----
OUT="$SCENE_DIR/oriented"
TETRA="$SCENE_DIR/tetra_meshes/tetra_mesh_binary_search_7.ply"
CAMS="$SCENE_DIR/mast3r_sfm/cameras.json"
IMAGES="$SCENE_DIR/mast3r_sfm/images"

progress_set 2 0 "B1 orient + decimate to 500K"
docker exec matcha-build bash -c "
  source /opt/matcha/bin/activate
  set -uo pipefail
  mkdir -p $OUT
  python /scripts/orient_mesh.py --tetra $TETRA --cameras $CAMS --output $OUT 2>&1 | tail -10
  python - <<PYEOF
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('$OUT/oriented_tetra.ply')
print(f'in: {len(mesh.vertices):,}v / {len(mesh.triangles):,}t')
dec = mesh.simplify_quadric_decimation(target_number_of_triangles=500_000)
dec.remove_degenerate_triangles(); dec.remove_unreferenced_vertices()
dec.remove_duplicated_triangles(); dec.compute_vertex_normals()
o3d.io.write_triangle_mesh('$OUT/oriented_500k.ply', dec)
o3d.io.write_triangle_mesh('$OUT/oriented_500k.obj', dec)
print(f'out: {len(dec.vertices):,}v / {len(dec.triangles):,}t')
PYEOF
" >> "$LOG_POST" 2>&1

# ---- Phase 3: B4 project_color ----
progress_set 3 0 "B4 project vertex colors"
docker exec matcha-build bash -c "
  source /opt/matcha/bin/activate
  python /scripts/project_color.py \
    --mesh $OUT/oriented_500k.ply --cameras $CAMS \
    --oriented-cameras $OUT/oriented_cameras.json \
    --images $IMAGES \
    --output $OUT/oriented_500k_colored.ply 2>&1 | tail -3
" >> "$LOG_POST" 2>&1

# ---- Phase 4: B2 cull ----
progress_set 4 0 "B2 cull"
docker exec matcha-build bash -c "
  source /opt/matcha/bin/activate
  python /scripts/cull_mesh.py \
    --mesh $OUT/oriented_500k_colored.ply --cameras $CAMS \
    --oriented-cameras $OUT/oriented_cameras.json \
    --output $OUT/oriented_500k_colored_culled.ply 2>&1 | tail -3
" >> "$LOG_POST" 2>&1

# ---- Phase 5: report ----
progress_set 5 0 "summarize"
echo "DONE: $HOST_OUT/oriented/oriented_500k_colored_culled.ply"
ls -la "$HOST_OUT/oriented/" | head

# progress_clear fires automatically here via EXIT trap.
