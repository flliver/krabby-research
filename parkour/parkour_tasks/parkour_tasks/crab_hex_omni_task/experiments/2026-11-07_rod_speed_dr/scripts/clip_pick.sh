#!/bin/bash
# REPORT section 11 decision 4 (the user's yes, 2026-11-09): ONE headless 30-s clip of the pick c31000 (sha 155b5981) for
# the user's veto, never a gate. Task 4's clip recipe (../../2026-10-16_task4_pipeline/step11/clip_F19.sh, which made
# F19's c28000 clip) with this worktree's code on PYTHONPATH (as gate_row.sh):
#   - play_stream_video.py streams frames (NEVER gym RecordVideo / play.py --video);
#   - Student-Play, 1 env, 1500 steps (30 s at 50 Hz), on the wall field env F19/envs/p3b.field_hard.env:
#     0 kg (no KRABBY_PAYLOAD_* key), DR and stress knobs off (no KRABBY_DR_* / KRABBY_STRESS_* key);
#   - every parent KRABBY_* unset; systemd-run MemoryMax 40G cap + timeout (SIGTERM).
# Side effect, as play.py: play_stream_video.py writes exported_deploy/ beside the checkpoint.
# One Isaac process machine-wide: waits until none runs. Prints "CLIP END rc=N" last.
set -u
CD=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=$(cd "$CD/../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
FF=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
ENV_FILE=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/F19/envs/p3b.field_hard.env
CK=$W/parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19/model_31000.pt
SHA=155b5981
STEPS=1500
LOG=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rod_speed_dr/clip_pick.log
SRC=$(dirname "$CK")/videos/play/stream-${STEPS}steps.mp4
DST=$CD/clip/student_c31000_field_hard.mp4

end() { echo "CLIP END rc=$1 ${2:-}"; exit "$1"; }
[ "$(sha256sum "$CK" | cut -c1-8)" = "$SHA" ] || end 2 "(checkpoint sha is not $SHA)"
[ -e "$SRC" ] && end 2 "($SRC exists; move it aside first)"
grep -qE '^KRABBY_(DR|STRESS|PAYLOAD)_' "$ENV_FILE" && end 2 "($ENV_FILE sets a DR / stress / payload key)"
avail=$(df --output=avail -B1G "$W" | tail -1 | tr -d ' ')
[ "$avail" -ge 30 ] || end 3 "(only $avail GB free; CHARTER 3.5 floor 30 GB)"
envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$ENV_FILE" | tr '\n' ' ')
unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')

while ps -ww -eo args | grep -qE '^[^ ]*[i]saac_venv/bin/python'; do sleep 20; done
echo "CLIP START $(date '+%F %T') ck=$CK steps=$STEPS log=$LOG"
t0=$(date +%s)
( cd "$W/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 \
    PYTHONPATH="$W/parkour/parkour_tasks:$W/parkour" $envs \
    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 timeout --signal=TERM 2400 \
    "$PY" scripts/rsl_rl/play_stream_video.py --task Isaac-Crab-Hex-Omni-Student-Play-v0 --headless --video \
    --video_length $STEPS --num_envs 1 --enable_cameras --checkpoint "$CK" ) > "$LOG" 2>&1
rc=$?
echo "CLIP RUN rc=$rc $(( $(date +%s) - t0 )) s"
[ $rc = 0 ] && [ -f "$SRC" ] || end 1 "(play_stream_video failed; log $LOG)"
mkdir -p "$CD/clip" && cp "$SRC" "$DST" || end 4 "(copy to $DST failed)"
"$FF" -y -loglevel error -i "$DST" -vf scale=960:-2 -c:v libx264 -crf 30 -preset slow -an "${DST%.mp4}_small.mp4" \
  || end 4 "(ffmpeg downscale failed)"
ls -la "$CD/clip"
end 0
