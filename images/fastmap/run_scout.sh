#!/usr/bin/env bash
# run_scout.sh — DA3 da3@1 scout gaussian in a SUPPLIED pose gauge, with
# phase progress emitted via lib_progress.sh (-> MQTT on the host). STO-SCN-095.
#
# Host orchestrator (docker-runs the baked containers):
#   phase 1: undistort fisheye -> pinhole  (krabby-fastmap; matches posed K)
#   phase 2: da3@1 posed scout gaussian    (krabby-da3; reads cameras/posed.json)
#
# Layout (under <data> = parent of <image_dir>):
#   <data>/images            fisheye scout views (frame_*.jpg)
#   <data>/cameras/posed.json  [{name,w2c,K}] from posed_from_sparse (solve gauge)
#   <data>/da3_infer_posed.py  staged da3@1 driver
#   -> writes <data>/scout_out/  (gs_ply + colmap + glb)
#
# Usage: run_scout.sh <image_dir> <out_dir> <make> <model> <mode> [res]
set -uo pipefail

IMG_DIR="${1:?usage: run_scout.sh <image_dir> <out_dir> <make> <model> <mode> [res]}"
OUT_DIR="${2:?need out_dir}"
MAKE="${3:?make}"; MODEL="${4:?model}"; MODE="${5:?mode}"; RES="${6:-504}"
FASTMAP_IMAGE="${KRABBY_FASTMAP_IMAGE:-j.pski.org:5000/krabby-fastmap:0.2}"
DA3_IMAGE="${KRABBY_DA3_IMAGE:-j.pski.org:5000/krabby-da3:0.4}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib_progress.sh"
DATA="$(cd "$IMG_DIR/.." && pwd)"
IMGB="$(basename "$IMG_DIR")"
rm -rf "$DATA/scout_out" "$DATA/undist"

progress_init 2

# Self-healing registry: prefer the copy STAGED with the job (always current)
# over the baked container copy, which goes stale the moment a camera is added
# (STO-SCN-091). Falls back to the baked one if the job didn't stage it.
PROFILES="/opt/krabby-tools/capture_profiles.json"
[ -f "$DATA/capture_profiles.json" ] && PROFILES="/data/capture_profiles.json"

# ── phase 1: undistort fisheye -> pinhole — ONLY for fisheye captures. ────────
# Rectilinear footage (e.g. iPhone) is ALREADY pinhole; fisheye-undistorting it
# is wrong and needs a fisheye calibration it doesn't have. Route from the
# capture FACT (mode), mirroring the solve's undistort=False for non-fisheye
# (solve_plan). DA3 then reads the original images, which match posed.json K
# (the solve ran on those same images). Anything other than 'fisheye' skips
# undistort — the safe default (feeding originals is recoverable; wrongly
# undistorting is not).
if [ "$MODE" = "fisheye" ]; then
  progress_set 1 0 "undistort scout views"
  docker run --rm --gpus all --shm-size=8g -v "$DATA":/data "$FASTMAP_IMAGE" \
      python /opt/krabby-tools/undistort_fisheye.py \
          --images "/data/$IMGB" --out /data/undist \
          --make "$MAKE" --model "$MODEL" --mode "$MODE" \
          --profiles "$PROFILES" \
    || { _progress_log "undistort FAILED"; exit 1; }
  progress_percent 100
  DA3_IN="/data/undist"
else
  _progress_log "mode=$MODE (non-fisheye) -> skip undistort; DA3 reads images directly"
  progress_set 1 100 "skip undistort (rectilinear)"
  DA3_IN="/data/$IMGB"
fi

# ── phase 2: da3@1 posed scout gaussian (in the supplied gauge) ──────────────
progress_set 2 0 "DA3 scout gaussian (posed)"
docker run --rm --gpus all --shm-size=8g -v "$DATA":/data "$DA3_IMAGE" \
    python /data/da3_infer_posed.py "$DA3_IN" /data/scout_out "$RES" \
  || { _progress_log "da3 scout FAILED"; exit 1; }
# chown the root-written outputs back to the caller
docker run --rm -v "$DATA":/data alpine chown -R "$(id -u):$(id -g)" /data || true

# Fault-tolerant gate: a silent no-output run must FAIL here (with the log on
# the host), not downstream in the gather rsync with an opaque "no such dir".
if [ -z "$(find "$DATA/scout_out" -name '*.ply' 2>/dev/null | head -1)" ]; then
  _progress_log "scout produced NO .ply in scout_out — failing (see above)"
  exit 1
fi
progress_set 2 100 "done"

echo "[run_scout] scout outputs:"
find "$DATA/scout_out" -maxdepth 2 -type f \( -name "*.ply" -o -name "*.json" \) | sed 's/^/  /'
# progress_clear fires on EXIT (lib_progress trap)
