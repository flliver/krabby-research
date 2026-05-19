#!/bin/bash
# MAtCha runner — sky-house-dining scene
# Validated 2026-04-30 on bbeeprz (RTX 5080, 16 GB).
#
# Prerequisites:
#   - krabby-matcha:latest image present (build via docker/Dockerfile.matcha,
#     or distribute via `docker save | ssh dst docker load` from a host that
#     already has it)
#   - matcha-build container is the long-running interactive container we
#     used during the porting session. For new runs, start a fresh one:
#       docker run -d -it --gpus all --shm-size=8g --name matcha-build \
#           -v /home/jeremy/outposts/krabby/data/011-scene-reconstruction:/data \
#           krabby-matcha:latest bash
#
# Inputs (default for this scene):
#   - source video: /data/videos/004-sky-house-dining.mp4
#   - sample 24 evenly-spaced frames at 1024px wide (only first 12 used; see notes)
#
# Output:
#   - /data/matcha_output/004-sky-house/
#       tetra_meshes/tetra_mesh_binary_search_7.ply  (watertight, ~422 MB, ~21M tris)

set -e

SCENE=${1:-004-sky-house-dining}
N_IMAGES=${2:-12}            # 12 fits 16 GB VRAM; 24 OOMs at chart alignment
ENCODER=${3:-vitl}           # vitl/vitb/vits/vitg — NOT large/base/small/giant
SOURCE_VIDEO="/data/videos/${SCENE}.mp4"
FRAMES_DIR="/data/frames/${SCENE}-matcha-24"
OUTPUT_DIR="/data/matcha_output/${SCENE}"

# Step 1: extract 24 sparse keyframes if missing.
# We use ffmpeg from the krabby-mast3r image because the matcha image
# doesn't include ffmpeg.
if [ ! -d "$FRAMES_DIR" ] || [ "$(ls "$FRAMES_DIR" 2>/dev/null | wc -l)" -lt 24 ]; then
    echo "=== Extracting frames to $FRAMES_DIR ==="
    docker exec matcha-build bash -c "mkdir -p '$FRAMES_DIR'"
    docker run --rm \
        -v /home/jeremy/outposts/krabby/data/011-scene-reconstruction:/data \
        krabby-mast3r:latest \
        ffmpeg -y -i "$SOURCE_VIDEO" \
               -vf "fps=24/227,scale=1024:-2" -q:v 2 \
               "$FRAMES_DIR/frame_%04d.jpg"
fi
echo "Frames present: $(docker exec matcha-build bash -c "ls $FRAMES_DIR | wc -l")"

# Step 2: run MAtCha
echo "=== Running MAtCha (n_images=$N_IMAGES, encoder=$ENCODER) ==="
docker exec matcha-build bash -c "
    source /opt/matcha/bin/activate
    export PYTHONPATH='/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn:\$PYTHONPATH'
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd /opt/MAtCha
    rm -rf '$OUTPUT_DIR'
    mkdir -p '$OUTPUT_DIR'
    python train.py \
        -s '$FRAMES_DIR' \
        -o '$OUTPUT_DIR' \
        --sfm_config unposed \
        --n_images $N_IMAGES \
        --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
        --depthanything_encoder $ENCODER \
        2>&1 | tee '$OUTPUT_DIR/run.log'
"

# Step 3: report outputs
echo
echo "=== Outputs ==="
docker exec matcha-build bash -c "
    find '$OUTPUT_DIR' -type f -size +1M -exec ls -lh {} \;
"

echo
echo "=== Done ==="
echo "Tetra mesh: $OUTPUT_DIR/tetra_meshes/tetra_mesh_binary_search_7.ply"
echo "Decimate to 200K tris with: docker exec matcha-build python /tmp/decimate.py"
