#!/usr/bin/env bash
# User decision 2026-10-06: stop the C0 side-first tails after SF__C0_rest_s16 (s17/s18 not run; the read becomes
# descriptive, n=1). sf_queue.sh stops by STOP_QUEUE before SF__C0_rest_s17, so its c6_reads never run for s16. This
# waits for that stop, then runs the same reads sf_queue.sh's c6_reads would: c6 pre-cut lr + turn band at c6 and c6x.
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction
LOG=$OUT/queue.log
A=$OUT/arms_p1
BAND_MAN=$C/scenarios_turn_band.yaml
REF=SF__C0_rest_s16
QPID=${1:?sf_queue pid}
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
recf() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$A/$1.record.json" "$2" 2>/dev/null; }
say "sf_c0s16_post start: waiting for sf_queue pid $QPID"
while kill -0 "$QPID" 2>/dev/null; do sleep 30; done
last=$(grep -E "STOP sf_queue|sf_queue end" "$LOG" | tail -1)
case "$last" in *"side-first C0 rest"*) ;; *) say "STOP sf_c0s16_post: unexpected queue end :: $(echo "$last" | cut -c1-200) -- NOTIFY"; exit 4;; esac
[ -f "$A/$REF.record.json" ] && [ -n "$(recf $REF checkpoint)" ] || { say "STOP sf_c0s16_post: no record/head for $REF -- NOTIFY"; exit 4; }
out=$("$PY" "$S/p1_read.py" --c6 "$REF" 2>&1) || { say "STOP sf_c0s16_post: no c6 head/stamp :: $(echo "$out" | tail -1 | cut -c1-200) -- NOTIFY"; exit 4; }
read -r ck st <<< "$(echo "$out" | tail -1)"
ARMS_DIR=$A bash "$S/run_arm.sh" --lr-eval "${REF}_c6" "$ck" "$st" || { say "STOP sf_c0s16_post: c6 lr failed -- NOTIFY"; exit 4; }
ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval "${REF}_c6_tb" "$ck" "$st" "$BAND_MAN" turn_band__omni || { say "STOP sf_c0s16_post: c6 band failed -- NOTIFY"; exit 4; }
ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval "${REF}_tb" "$(recf $REF checkpoint)" "$(recf $REF driver_stamp)" "$BAND_MAN" turn_band__omni || { say "STOP sf_c0s16_post: c6x band failed -- NOTIFY"; exit 4; }
rm -f "$OUT/STOP_QUEUE"
say "sf_c0s16_post end: C0 s16 c6 + band reads done; STOP_QUEUE cleared; GPU idle (campaign queue finished by user decision)"
