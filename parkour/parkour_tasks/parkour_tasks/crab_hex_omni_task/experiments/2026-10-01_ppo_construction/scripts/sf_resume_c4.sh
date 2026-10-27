#!/usr/bin/env bash
# User pause 2026-10-05 (a GUI view of B's turning): STOP_QUEUE stopped sf_queue.sh after SF__B_c4_s16 (between arms;
# no run was cut). Resume exactly where it stopped: the remaining side-first c4 arms (same run_arm.sh settings as
# sf_queue.sh's c4 step), the c4 gate (CHARTER 13.4), then FROM_STEP=rest sf_queue.sh (rest, c6 reads, C0 trigger).
#
# Usage (detached): setsid nohup bash scripts/sf_resume_c4.sh ["SF__B_c4_s17 SF__B_c4_s18"] > $OUT/sf_resume_c4.log 2>&1 < /dev/null &
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction
LOG=$OUT/queue.log
A=$OUT/arms_p1
FB=$OUT/arms_fb6k
REMAIN=${1:-"SF__B_c4_s17 SF__B_c4_s18"}
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
[ -f "$OUT/STOP_QUEUE" ] && { say "STOP sf_resume_c4: STOP_QUEUE present at start"; exit 4; }
say "sf_resume_c4 start: c4 [$REMAIN] -> c4 gate -> FROM_STEP=rest sf_queue.sh"
STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$FB RESUME_READ=1 ACCEPT_WHY="" bash "$S/run_arm.sh" $REMAIN
rc=$?
if [ "$rc" != 0 ]; then say "STOP sf_resume_c4: rc $rc in the side-first c4 step -- NOTIFY"; exit 4; fi
[ -f "$OUT/STOP_QUEUE" ] && { say "STOP sf_resume_c4: STOP_QUEUE after the c4 step -- NOTIFY"; exit 4; }
g=$("$PY" "$S/sf_read.py" --c4-gate 2>&1); rc=$?
say "C4GATE rc $rc :: $(echo "$g" | tail -1 | cut -c1-400)"
if [ "$rc" != 0 ]; then say "STOP sf_resume_c4: the c4 gate (CHARTER 13.4) -- NOTIFY stop and ask"; exit 4; fi
FROM_STEP=rest exec bash "$S/sf_queue.sh"
