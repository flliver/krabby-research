#!/usr/bin/env bash
# run_fastmap.sh — GPU SfM solve via the krabby-fastmap container, with
# phase/percent progress emitted through lib_progress.sh (-> MQTT on the fleet,
# best-effort). STO-SCN-101; the hardened run path consumed by STO-SCN-093.
#
# Pipeline (all GPU): colmap feature_extractor -> colmap <matcher> -> fastmap.
# Each stage is a separate `docker run` against a shared /data volume so the
# database persists between them. If a database already exists it is REUSED
# (extract+match skipped) unless REUSE_DB=0.
#
# Usage:
#   run_fastmap.sh <image_dir> <output_dir> [camera_model] [matcher] [timeout_s]
#     camera_model: SIMPLE_RADIAL (default) | SIMPLE_PINHOLE
#                   (FastMap supports ONLY these; fisheye must be undistorted
#                    to pinhole first — STO-SCN-093 concern)
#     matcher:      exhaustive_matcher (default) | vocab_tree_matcher
#                   (NOT sequential for hyperlapse — HUG-SCN-004)
#     timeout_s:    hard cap on the FastMap stage (default 1200)
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
# FastMap's run.py REFUSES a pre-existing output dir (FileExistsError) — so we
# remove it and let FastMap create it fresh (do NOT mkdir it).
rm -rf "$OUT_DIR"

drun() { docker run --rm --gpus all --shm-size=8g -v "$DATA":/data "$IMAGE" "$@"; }

# Throttle: only emit when the integer percent actually changes (the matcher and
# FastMap rotation loop print thousands of lines — without this we'd flood MQTT).
_emit_pct() { [ "${1:-}" != "${_LAST_PCT:-x}" ] && { progress_percent "$1"; _LAST_PCT="$1"; }; }

progress_init 4

if [ -s "$DATA/$DB_BASE" ] && [ "${REUSE_DB:-1}" = "1" ]; then
    _progress_log "reusing existing $DB_BASE (extract+match skipped; REUSE_DB=0 to force fresh)"
    progress_set 1 100 "feature extraction (cached)"
    progress_set 2 100 "matching (cached)"
else
    # ── phase 1: GPU feature extraction ─────────────────────────────────
    progress_set 1 0 "colmap feature extraction ($CAMERA_MODEL)"
    rm -f "$DATA/$DB_BASE"
    _LAST_PCT=x
    drun colmap feature_extractor \
            --database_path "/data/$DB_BASE" \
            --image_path "/data/$IMG_BASE" \
            --FeatureExtraction.use_gpu 1 \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model "$CAMERA_MODEL" 2>&1 \
      | while IFS= read -r line; do
            printf '%s\n' "$line"
            case "$line" in *"Processed file ["*)
                nums=$(printf '%s' "$line" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\1 \2/p')
                set -- $nums; [ -n "${2:-}" ] && [ "$2" -gt 0 ] && _emit_pct $(( 100 * $1 / $2 )) ;;
            esac
        done
    [ "${PIPESTATUS[0]}" -eq 0 ] || { _progress_log "feature extraction FAILED"; exit 1; }

    # ── phase 2: GPU matching (block-level percent) ─────────────────────
    progress_set 2 0 "colmap $MATCHER"
    _LAST_PCT=x
    drun colmap "$MATCHER" --database_path "/data/$DB_BASE" --FeatureMatching.use_gpu 1 2>&1 \
      | while IFS= read -r line; do
            printf '%s\n' "$line"
            case "$line" in *"Processing block ["*)
                # "Processing block [i/N, j/M]" -> linear over N*M blocks
                nums=$(printf '%s' "$line" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\), \([0-9]*\)\/\([0-9]*\)\].*/\1 \2 \3 \4/p')
                set -- $nums
                [ -n "${4:-}" ] && [ "$(( $2 * $4 ))" -gt 0 ] && _emit_pct $(( 100 * ( ($1-1)*$4 + $3 ) / ($2*$4) )) ;;
            esac
        done
    [ "${PIPESTATUS[0]}" -eq 0 ] || { _progress_log "matching FAILED"; exit 1; }
    progress_percent 100
fi

# ── phase 3: FastMap GPU pose + sparse triangulation ────────────────────
progress_set 3 0 "fastmap pose estimation"
_LAST_PCT=x
timeout "$FASTMAP_TIMEOUT" \
  docker run --rm --gpus all --shm-size=8g -v "$DATA":/data "$IMAGE" \
    python /opt/fastmap/run.py \
        --database "/data/$DB_BASE" --image_dir "/data/$IMG_BASE" \
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

# ── phase 4: done ────────────────────────────────────────────────────────
progress_set 4 100 "done"
echo "[run_fastmap] sparse output under $OUT_DIR:"
find "$OUT_DIR" -maxdepth 2 -type f | sed 's/^/  /'
# progress_clear fires automatically on EXIT (lib_progress trap)
