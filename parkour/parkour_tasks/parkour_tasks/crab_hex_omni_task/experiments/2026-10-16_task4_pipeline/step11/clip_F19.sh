#!/bin/bash
# PLAN step 12 STOP-END prep: ONE headless 30-s student clip of F19's selected student c28000 for the user's veto
# (never a gate). Modelled exactly on R18's render (scratchpad/clip_r18.cmd, CHARTER 2026-10-21): play_stream_video.py
# (streams frames; NEVER gym RecordVideo / play.py --video), Student-Play, the wall field env (F19/envs/p3b.field_hard.env;
# no STUDENT_RUNNER_KNOBS_REFUSED key and no KRABBY_PAYLOAD_* in it -> 0 kg), every parent KRABBY_* unset, under a
# systemd-run MemoryMax cap + timeout, detached. One Isaac process machine-wide: wait until none runs.
set -u
R=/home/nickmagus/krabby/krabby-research
PL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline
CK=$R/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
LOG=/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/b8e96414-701e-4216-9940-452abfe18030/scratchpad/step11/clip_F19.log
while ps -ww -eo args | grep -qE '^[^ ]*[i]saac_venv/bin/python'; do sleep 20; done
cd $R/parkour && env $(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ') OMNI_KIT_ACCEPT_EULA=yes TERM=xterm \
  KRABBY_ACTION_LPF_HZ=0 KRABBY_ACTION_LPF_OBS=0 KRABBY_APEX_W=1.0 KRABBY_CLOCK_W=1.0 KRABBY_FIELD_BLOCKS=14:28 KRABBY_FIELD_EDGES=wall KRABBY_FIELD_KINDS=0.4:0.4:0.2 KRABBY_FIELD_PAD_R=1.7 KRABBY_FLAT_TERRAIN_DIFF=0.70:1.0 KRABBY_FLAT_TERRAIN_FLAT_FRAC=0.0 KRABBY_FLAT_TERRAIN_GEOM=field KRABBY_FLAT_TERRAIN_MODE=light KRABBY_INIT_STD=0.8 KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_OMNI_ANG_VEL_CLIP=0.2 KRABBY_OMNI_ANG_VEL_MAX=0.3 KRABBY_OMNI_CLOCK_HIDE=1 KRABBY_OMNI_CLOCK_TURN_R=1.2 KRABBY_OMNI_CMD_OBS=0 KRABBY_OMNI_COMMAND_FRAME=world KRABBY_OMNI_FACING_OBS_CLIP=0.6 KRABBY_OMNI_FACING_OBS_ENCODING=raw KRABBY_OMNI_RELAY_HYST=0 KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 KRABBY_OMNI_STAND_HOLD=0 KRABBY_OMNI_STAND_OBS=0 KRABBY_OMNI_STAND_PARK=0 KRABBY_OMNI_YAW_RANGE=-3.14159:3.14159 KRABBY_PLANT=A15+B KRABBY_PLAY_DEBUG_VIS=0 KRABBY_RSI_FRAC=0 KRABBY_STD_PARAM=soft KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.1  \
  systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 timeout --signal=TERM 2400 \
  /home/nickmagus/krabby/isaac_venv/bin/python scripts/rsl_rl/play_stream_video.py \
  --task Isaac-Crab-Hex-Omni-Student-Play-v0 --headless --video --video_length 1500 --num_envs 1 --enable_cameras \
  --checkpoint $CK > $LOG 2>&1
rc=$?
SRC=$R/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/videos/play/stream-1500steps.mp4
[ $rc = 0 ] && [ -f $SRC ] || { echo "clip FAILED rc=$rc (log $LOG)"; exit 1; }
mkdir -p $PL/F19/clip && cp $SRC $PL/F19/clip/student_c28000_field_hard.mp4
FF=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
"$FF" -y -loglevel error -i $PL/F19/clip/student_c28000_field_hard.mp4 -vf scale=960:-2 -c:v libx264 -crf 30 -preset slow -an \
  $PL/F19/clip/student_c28000_field_hard_small.mp4
ls -la $PL/F19/clip
