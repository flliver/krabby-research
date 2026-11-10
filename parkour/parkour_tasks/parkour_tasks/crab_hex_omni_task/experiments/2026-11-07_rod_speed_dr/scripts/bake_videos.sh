#!/bin/bash
# The bake's sample videos (the user, 2026-11-09: "once its done baking record sample videos with the baked model").
# VIDEO ONLY, read by no gate. Three Isaac runs, one at a time, on this worktree's code (PYTHONPATH as rsd_stand.py),
# from the BAKED files: policy/rsd_9_p3b_rodspeed_dr/model_28991.pt (TDR, 03c3e472) and
# policy/rsd_stand_net/model_5000.pt (SH_RSD_TDR, 89f4428b); the student from its cited checkpoint (c31000, 155b5981).
#   1. walker + stand net: sh_record_video.sh --plan mix --seed 1514 --terrain field_medium, the plan of F19's
#      sh_mix_F19_s1514_rough video (reference stamp: SH_RSD_TDR's hraise__omni es1 row);
#   2. the same mix on --terrain field_max (F19's sh_mix_F19_s1514_roughest);
#   3. the depth student alone: play_stream_video.py, Student-Play, 1 env, 3000 steps (60 s at 50 Hz) on the play env of
#      record F19/envs/p3b.env (0 kg; no DR / stress / payload key), from a scratch copy of the checkpoint so the export
#      side effect (exported_deploy/ beside it) never touches the run dir the bake's ONNX files come from.
# Each mp4 also gets a _phone.mp4 (960 px wide). Outputs, gitignored, in the main checkout:
#   parkour/logs/videos/sh_stand_height/sh_mix_RSD_s1514_{rough,roughest}{.log,_sequence.mp4,_sequence_phone.mp4,
#   _sequence_sanity.json} and parkour/outputs/omni_plan/rod_speed_dr/videos/student_c31000_p3b{,_phone}.mp4.
# Memory caps: the recorder's own MemoryMax 24G scope; the student run MemoryMax 40G (clip_pick.sh's). SIGTERM only.
# Sentinels on stdout (the queue log): "VQ START", "VQ STEP <k> rc=N", "VQ END rc=N".
set -u
CD=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=$(cd "$CD/../../../../../.." && pwd)
M=/home/nickmagus/krabby/krabby-research
PKG=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
S=$PKG/experiments/2026-10-13_stand_height/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
FF=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
TDR=$PKG/policy/rsd_9_p3b_rodspeed_dr/model_28991.pt
SH=$PKG/policy/rsd_stand_net/model_5000.pt
REF=$M/parkour/outputs/omni_plan/phase2/lr/SH_RSD_TDR_sh_s19/hraise__omni/seed001/2026-11-09_21-04-04
ST_SRC=$W/parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19/model_31000.pt
ST_ENV=$PKG/experiments/2026-10-16_task4_pipeline/F19/envs/p3b.env
VOUT=$M/parkour/logs/videos/sh_stand_height
SOUT=$M/parkour/outputs/omni_plan/rod_speed_dr/videos
STEPS=3000
# sh_record_video.sh's busy patterns (Kit launches only: a pytest or CPU helper run is not an Isaac process)
ISAAC_PAT=$(sed -n "s/^ISAAC_PAT='\\(.*\\)'$/\\1/p" "$S/sh_record_video.sh")
NONKIT_PAT=$(sed -n "s/^NONKIT_PAT='\\(.*\\)'$/\\1/p" "$S/sh_record_video.sh")

end() { echo "VQ END rc=$1 ${2:-} $(date '+%F %T')"; exit "$1"; }
sha8() { sha256sum "$1" | cut -c1-8; }
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
idle() { while busy; do sleep 20; done; }
phone() { "$FF" -y -loglevel error -i "$1" -vf scale=960:-2 -c:v libx264 -crf 30 -preset slow -an "${1%.mp4}_phone.mp4"; }

echo "VQ START $(date '+%F %T')"
[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || end 2 "(could not read the busy patterns from sh_record_video.sh)"
[ "$(sha8 "$TDR")" = 03c3e472 ] || end 2 "(TDR baked file sha is not 03c3e472)"
[ "$(sha8 "$SH")" = 89f4428b ] || end 2 "(SH baked file sha is not 89f4428b)"
[ "$(sha8 "$ST_SRC")" = 155b5981 ] || end 2 "(student checkpoint sha is not 155b5981)"
[ -f "$REF/run_meta.json" ] || end 2 "(no reference stamp at $REF)"
grep -qE '^KRABBY_(DR|STRESS|PAYLOAD)_' "$ST_ENV" && end 2 "($ST_ENV sets a DR / stress / payload key)"
avail=$(df --output=avail -B1G "$M" | tail -1 | tr -d ' ')
[ "$avail" -ge 30 ] || end 3 "(only $avail GB free; floor 30 GB)"

# every parent KRABBY_* / walker / stand / phase-2 variable unset; the worktree on PYTHONPATH
unset_args=$(env | grep -oE '^(KRABBY|SN_WALKER|SN_STAND|P2|P2R|SH|TP)_[A-Za-z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
RUN=(env $unset_args PYTHONPATH="$W/parkour/parkour_tasks:$W/parkour")

k=0
for spec in "field_medium sh_mix_RSD_s1514_rough" "field_max sh_mix_RSD_s1514_roughest"; do
  set -- $spec; terrain=$1; prefix=$2; k=$((k + 1))
  idle
  echo "VQ STEP $k START $prefix terrain=$terrain $(date '+%F %T')"
  for try in $(seq 1 20); do                # exit 3 = busy at the launcher's own check (a race with another launch)
    "${RUN[@]}" SH_VIDEO_OUT="$VOUT" bash "$S/sh_record_video.sh" --plan mix --seed 1514 --terrain "$terrain" \
      --prefix "$prefix" -- --checkpoint "$TDR" --stand-ckpt "$SH" --stand-mask 1 --deadband 0.25 --eval-run "$REF"
    rc=$?
    [ $rc = 3 ] || break
    sleep 30; idle
  done
  [ $rc = 0 ] || { echo "VQ STEP $k rc=$rc (launcher refused; see $VOUT/$prefix.log)"; end 1; }
  until grep -qE "SHVIDEO END $prefix unit=.* rc=" "$VOUT/$prefix.log" 2>/dev/null; do sleep 20; done
  rc=$(grep -E "SHVIDEO END $prefix unit=.* rc=" "$VOUT/$prefix.log" | tail -1 | sed 's/.* rc=//')
  mp4=$VOUT/${prefix}_sequence.mp4
  [ "$rc" = 0 ] && [ -f "$mp4" ] || { echo "VQ STEP $k rc=${rc:-?} (recorder failed; see $VOUT/$prefix.log)"; end 1; }
  phone "$mp4" || { echo "VQ STEP $k rc=4 (phone copy failed)"; end 4; }
  echo "VQ STEP $k rc=0 $mp4 $(date '+%F %T')"
done

k=$((k + 1))
RD=$SOUT/student_run
[ -e "$RD/videos/play/stream-${STEPS}steps.mp4" ] && end 2 "($RD/videos/play/stream-${STEPS}steps.mp4 exists)"
mkdir -p "$RD" && { [ -e "$RD/model_31000.pt" ] || ln "$ST_SRC" "$RD/model_31000.pt"; } || end 2 "(no scratch copy)"
[ "$(sha8 "$RD/model_31000.pt")" = 155b5981 ] || end 2 "(scratch copy sha is not 155b5981)"
envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$ST_ENV" | tr '\n' ' ')
idle
echo "VQ STEP $k START student c31000 env=$(basename "$ST_ENV") steps=$STEPS $(date '+%F %T')"
( cd "$W/parkour" && "${RUN[@]}" OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 $envs \
    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 timeout --signal=TERM 2400 \
    "$PY" scripts/rsl_rl/play_stream_video.py --task Isaac-Crab-Hex-Omni-Student-Play-v0 --headless --video \
    --video_length $STEPS --num_envs 1 --enable_cameras --checkpoint "$RD/model_31000.pt" ) > "$SOUT/student_c31000_p3b.log" 2>&1
rc=$?
src=$RD/videos/play/stream-${STEPS}steps.mp4
[ $rc = 0 ] && [ -f "$src" ] || { echo "VQ STEP $k rc=$rc (play_stream_video failed; see $SOUT/student_c31000_p3b.log)"; end 1; }
cp "$src" "$SOUT/student_c31000_p3b.mp4" && phone "$SOUT/student_c31000_p3b.mp4" || { echo "VQ STEP $k rc=4"; end 4; }
echo "VQ STEP $k rc=0 $SOUT/student_c31000_p3b.mp4 $(date '+%F %T')"
end 0
