#!/bin/bash
# Run the r-knob-sweep r3 experiment on a host with the matcha-build container.
#
# Pushes the source patch + new YAML into the container, applies the patch
# (idempotent), stages frames from 12-dense-strong, and runs MAtCha with
# --alignment_config strong-r3 --dense_regul strong against the same 12 frames.
#
# Expects this script to live in <experiment-root>/scripts/ alongside
# ../patches/apply_chart_resolutions_patch.py and ../configs/strong-r3.yaml.
#
# Run on the host (tbeeprz), not inside the container:
#     bash scripts/run_r3_remote.sh
set -euo pipefail

# ---- config ----------------------------------------------------------------
CONTAINER=matcha-build
DATA_BASE=/home/jeremy/outposts/krabby/data/011-scene-reconstruction
HOST_SCENES=$DATA_BASE/scenes
SRC_VARIANT=004-sky-house-curated-12-dense-strong
NEW_VARIANT=004-sky-house-curated-12-dense-strong-r3

# Resolve the experiment root from this script's location
HERE=$(dirname "$(realpath "$0")")
EXP_ROOT=$(realpath "$HERE/..")
PATCH_FILE=$EXP_ROOT/patches/apply_chart_resolutions_patch.py
YAML_FILE=$EXP_ROOT/configs/strong-r3.yaml

# ---- pre-flight ------------------------------------------------------------
echo "=== pre-flight ==="
[ -f "$PATCH_FILE" ] || { echo "ERROR: patch file missing: $PATCH_FILE"; exit 1; }
[ -f "$YAML_FILE" ]  || { echo "ERROR: yaml file missing: $YAML_FILE"; exit 1; }

if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER\$"; then
    echo "ERROR: container '$CONTAINER' is not running"
    docker ps -a --filter "name=matcha" --format 'table {{.Names}}\t{{.Status}}' || true
    exit 1
fi

if ! docker exec "$CONTAINER" nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "ERROR: GPU not visible from inside $CONTAINER (CUDA missing?)"
    echo "       (operational lesson — container needs restart, see journal note 2026-05-01T222605)"
    exit 1
fi
echo "  container ok, GPU visible"

# ---- 1. push artifacts into container --------------------------------------
echo
echo "=== push patch + yaml into $CONTAINER ==="
docker cp "$PATCH_FILE" "$CONTAINER:/tmp/apply_chart_resolutions_patch.py"
docker cp "$YAML_FILE"  "$CONTAINER:/opt/MAtCha/configs/charts_alignment/strong-r3.yaml"
echo "  ok"

# ---- 2. apply the source patch (idempotent) --------------------------------
echo
echo "=== apply patch ==="
docker exec "$CONTAINER" python3 /tmp/apply_chart_resolutions_patch.py

# ---- 3. stage source frames -----------------------------------------------
echo
echo "=== stage source frames ==="
SRC_IMAGES=$HOST_SCENES/$SRC_VARIANT/mast3r_sfm/images
NEW_OUTPUT=$HOST_SCENES/$NEW_VARIANT
NEW_IMAGES=$NEW_OUTPUT/mast3r_sfm/images
[ -d "$SRC_IMAGES" ] || { echo "ERROR: source images not at $SRC_IMAGES"; exit 1; }

if [ -d "$NEW_IMAGES" ] && [ "$(ls -1 "$NEW_IMAGES" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  $NEW_IMAGES already populated ($(ls -1 "$NEW_IMAGES" | wc -l) files); skipping copy"
else
    mkdir -p "$NEW_IMAGES"
    cp -r "$SRC_IMAGES/." "$NEW_IMAGES/"
    echo "  copied $(ls -1 "$NEW_IMAGES" | wc -l) frames → $NEW_IMAGES"
fi

# ---- 4. background VRAM sampler -------------------------------------------
LOG_DIR=$NEW_OUTPUT/run_logs
mkdir -p "$LOG_DIR"
VRAM_LOG=$LOG_DIR/nvidia-smi.csv
RUN_LOG=$LOG_DIR/train.log

# Header
echo "ts,used_mib,total_mib" > "$VRAM_LOG"
(
    while true; do
        line=$(nvidia-smi --query-gpu=memory.used,memory.total \
                          --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        echo "$(date -Iseconds),$line" >> "$VRAM_LOG"
        sleep 5
    done
) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# ---- 5. run MAtCha --------------------------------------------------------
echo
echo "=== run train.py (--alignment_config strong-r3 --dense_regul strong) ==="
echo "  output → $NEW_OUTPUT"
echo "  log    → $RUN_LOG"
echo "  vram   → $VRAM_LOG  (5-sec sampler)"
echo

CONT_SOURCE=/data/scenes/$NEW_VARIANT/mast3r_sfm/images
CONT_OUTPUT=/data/scenes/$NEW_VARIANT

START=$(date +%s)
docker exec "$CONTAINER" bash -c "
    source /opt/matcha/bin/activate
    export PYTHONPATH='/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn:\$PYTHONPATH'
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd /opt/MAtCha
    python train.py \
        -s $CONT_SOURCE \
        -o $CONT_OUTPUT \
        --sfm_config unposed \
        --alignment_config strong-r3 \
        --dense_regul strong \
        --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
        --depthanything_encoder vitl
" 2>&1 | tee "$RUN_LOG"
EXIT_CODE=${PIPESTATUS[0]}
DURATION=$(( $(date +%s) - START ))

# ---- 6. summary -----------------------------------------------------------
PEAK_VRAM=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$VRAM_LOG")
echo
echo "=== summary ==="
echo "  exit:        $EXIT_CODE"
echo "  duration:    ${DURATION}s"
echo "  peak VRAM:   ${PEAK_VRAM} MiB"
echo "  output dir:  $NEW_OUTPUT"
echo "  run log:     $RUN_LOG"
echo "  vram log:    $VRAM_LOG"

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  ✗ run failed — see $RUN_LOG"
    exit "$EXIT_CODE"
fi

# Confirm key artifacts produced
for ARTIFACT in mast3r_sfm/cameras.json mast3r_sfm/charts_data.npz tetra_meshes; do
    if [ -e "$NEW_OUTPUT/$ARTIFACT" ]; then
        echo "  ✓ $ARTIFACT"
    else
        echo "  ✗ $ARTIFACT MISSING"
    fi
done
echo "  ✓ done."
