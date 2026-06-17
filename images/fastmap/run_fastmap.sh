#!/usr/bin/env bash
# run_fastmap.sh — GPU SfM solve via the krabby-fastmap container, with
# phase/percent progress emitted through lib_progress.sh (-> MQTT on the fleet,
# best-effort). STO-SCN-101/093; the host orchestrator cmd_solve invokes.
#
# Pipeline (all GPU): [undistort fisheye->pinhole] -> colmap feature_extractor
# -> colmap <matcher> -> fastmap. Each stage is a separate `docker run` against a
# shared /data volume (so nanny-progress on the host emits per phase). Database
# is REUSED if present (extract+match skipped) unless REUSE_DB=0.
#
# Usage:
#   run_fastmap.sh <image_dir> <output_dir> [camera_model] [matcher] [timeout_s]
# Optional undistort (env): UNDISTORT_MODE=fisheye UNDISTORT_MAKE=DJI
#   UNDISTORT_MODEL="DJI Action 3" [UNDISTORT_BALANCE=0.0]
#   -> phase 1 undistorts <image_dir> to <data>/undist (baked 102 calibration);
#      the solve then runs on the undistorted pinhole frames.
set -uo pipefail

IMG_DIR="${1:?usage: run_fastmap.sh <image_dir> <output_dir> [camera_model] [matcher] [timeout_s]}"
OUT_DIR="${2:?need output_dir}"
CAMERA_MODEL="${3:-SIMPLE_RADIAL}"
MATCHER="${4:-exhaustive_matcher}"
FASTMAP_TIMEOUT="${5:-1200}"
IMAGE="${KRABBY_FASTMAP_IMAGE:-krabby-fastmap:0.1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib_progress.sh"

DATA="$(cd "$IMG_DIR/.." && pwd)"     # mount parent so db + out sit beside images
IMG_BASE="$(basename "$IMG_DIR")"
OUT_BASE="$(basename "$OUT_DIR")"
DB_BASE="database.db"
# Undistorted frames (if any) land in <data>/undist; the solve runs on them.
IMG_SRC="$IMG_BASE"
[ -n "${UNDISTORT_MODE:-}" ] && IMG_SRC="undist"
# Self-healing registry: prefer the copy STAGED with the job (current) over the
# baked container copy (goes stale when a camera is added — STO-SCN-091).
PROFILES="/opt/krabby-tools/capture_profiles.json"
[ -f "$DATA/capture_profiles.json" ] && PROFILES="/data/capture_profiles.json"
# FastMap's run.py REFUSES a pre-existing output dir -> remove, let it create.
rm -rf "$OUT_DIR"

drun() { docker run --rm --gpus all --shm-size=8g -v "$DATA":/data "$IMAGE" "$@"; }
_emit_pct() { [ "${1:-}" != "${_LAST_PCT:-x}" ] && { progress_percent "$1"; _LAST_PCT="$1"; }; }

progress_init 5

if [ -s "$DATA/$DB_BASE" ] && [ "${REUSE_DB:-1}" = "1" ]; then
    _progress_log "reusing existing $DB_BASE (undistort+extract+match skipped; REUSE_DB=0 to force)"
    progress_set 1 100 "undistort (cached)"
    progress_set 2 100 "feature extraction (cached)"
    progress_set 3 100 "matching (cached)"
else
    # ── phase 1: undistort fisheye -> pinhole (optional) ────────────────
    if [ -n "${UNDISTORT_MODE:-}" ]; then
        progress_set 1 0 "undistort fisheye -> pinhole"
        drun python /opt/krabby-tools/undistort_fisheye.py \
            --images "/data/$IMG_BASE" --out "/data/undist" \
            --make "${UNDISTORT_MAKE}" --model "${UNDISTORT_MODEL}" --mode "${UNDISTORT_MODE}" \
            --balance "${UNDISTORT_BALANCE:-0.0}" \
            --profiles "$PROFILES" \
          || { _progress_log "undistort FAILED"; exit 1; }
        progress_percent 100
    else
        progress_set 1 100 "no undistort (pinhole input)"
    fi

    # ── phase 2: GPU feature extraction ─────────────────────────────────
    progress_set 2 0 "colmap feature extraction ($CAMERA_MODEL)"
    rm -f "$DATA/$DB_BASE"
    _LAST_PCT=x
    drun colmap feature_extractor \
            --database_path "/data/$DB_BASE" --image_path "/data/$IMG_SRC" \
            --FeatureExtraction.use_gpu 1 --ImageReader.single_camera 1 \
            --ImageReader.camera_model "$CAMERA_MODEL" 2>&1 \
      | while IFS= read -r line; do
            printf '%s\n' "$line"
            case "$line" in *"Processed file ["*)
                nums=$(printf '%s' "$line" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\1 \2/p')
                set -- $nums; [ -n "${2:-}" ] && [ "$2" -gt 0 ] && _emit_pct $(( 100 * $1 / $2 )) ;;
            esac
        done
    [ "${PIPESTATUS[0]}" -eq 0 ] || { _progress_log "feature extraction FAILED"; exit 1; }

    # ── phase 3: GPU matching (block-level percent) ─────────────────────
    progress_set 3 0 "colmap $MATCHER"
    _LAST_PCT=x
    drun colmap "$MATCHER" --database_path "/data/$DB_BASE" --FeatureMatching.use_gpu 1 2>&1 \
      | while IFS= read -r line; do
            printf '%s\n' "$line"
            case "$line" in *"Processing block ["*)
                nums=$(printf '%s' "$line" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\), \([0-9]*\)\/\([0-9]*\)\].*/\1 \2 \3 \4/p')
                set -- $nums
                [ -n "${4:-}" ] && [ "$(( $2 * $4 ))" -gt 0 ] && _emit_pct $(( 100 * ( ($1-1)*$4 + $3 ) / ($2*$4) )) ;;
            esac
        done
    [ "${PIPESTATUS[0]}" -eq 0 ] || { _progress_log "matching FAILED"; exit 1; }
    progress_percent 100
fi

# ── phase 4: FastMap GPU pose + sparse triangulation ────────────────────
progress_set 4 0 "fastmap pose estimation"
_LAST_PCT=x
timeout "$FASTMAP_TIMEOUT" \
  docker run --rm --gpus all --shm-size=8g -v "$DATA":/data "$IMAGE" \
    python /opt/fastmap/run.py \
        --database "/data/$DB_BASE" --image_dir "/data/$IMG_SRC" \
        --output_dir "/data/$OUT_BASE" --headless 2>&1 \
  | while IFS= read -r line; do
        printf '%s\n' "$line"
        case "$line" in
          *fastmap.rotation*)                  _emit_pct 30 ;;
          *fastmap.translation*)               _emit_pct 55 ;;
          *fastmap.*triangulat*)               _emit_pct 75 ;;
          *Writing*|*Saving*|*fastmap.*track*) _emit_pct 90 ;;
        esac
    done
fm_rc=${PIPESTATUS[0]}
if [ "$fm_rc" -eq 124 ]; then _progress_log "fastmap TIMEOUT after ${FASTMAP_TIMEOUT}s"; exit 124; fi
[ "$fm_rc" -eq 0 ] || { _progress_log "fastmap FAILED rc=$fm_rc"; exit 1; }
# carry the undistort intrinsics next to the model for provenance
[ -n "${UNDISTORT_MODE:-}" ] && cp "$DATA/undist/intrinsics.json" "$DATA/$OUT_BASE/" 2>/dev/null || true

# ── phase 5: done ──────────────────────────────────────────────────────
progress_set 5 100 "done"
echo "[run_fastmap] sparse output under $OUT_DIR:"
find "$OUT_DIR" -maxdepth 2 -type f | sed 's/^/  /'
# progress_clear fires automatically on EXIT (lib_progress trap)
