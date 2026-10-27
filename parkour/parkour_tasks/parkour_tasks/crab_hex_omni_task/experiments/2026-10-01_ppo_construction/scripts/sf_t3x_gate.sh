#!/usr/bin/env bash
# User amendment 2026-10-05 (CHARTER 13.6): hold before side-first B when the t3x turning fails, and analyse why first.
# sf_queue.sh was stopped by hand (STOP_QUEUE) during the last t3x arm, so it exits after the t3x training step and
# before the t3x turn-band reads. This script waits for that exit, runs the t3x turn-band reads and the t3x rule exactly as
# sf_queue.sh would, then:
#   - in-band turn settles (B t3x in-band heading_final mean <= C0 t3 mean + 0.05 rad) -> side-first resumes as
#     pre-registered: FROM_STEP=c4 sf_queue.sh (a TAILS CANDIDATE reading is still notified as the user's call);
#   - otherwise (turn not in-band, INCOMPLETE, or any read error) -> HOLD: nothing launches; NOTIFY; analysis.
#
# Usage (detached): setsid nohup bash scripts/sf_t3x_gate.sh <sf_queue pid> > $OUT/sf_t3x_gate.log 2>&1 < /dev/null &
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction
LOG=$OUT/queue.log
A=$OUT/arms_p1
BAND_MAN=$C/scenarios_turn_band.yaml
T3X="P1B6__B_t3x_s16 P1B6__B_t3x_s17 P1B6__B_t3x_s18 P1B6__C0_t3x_s16"
QPID=${1:?sf_queue pid}
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
hold() { say "HOLD sf_t3x_gate: $* -- NOTIFY (side-first not launched; user 2026-10-05: analyse the turning first)"; exit 4; }
recf() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$A/$1.record.json" "$2" 2>/dev/null; }

say "sf_t3x_gate start: waiting for sf_queue pid $QPID to stop after the t3x step"
while kill -0 "$QPID" 2>/dev/null; do sleep 30; done
last=$(grep -E "STOP sf_queue|sf_queue end" "$LOG" | tail -1)
case "$last" in
  *"STOP_QUEUE after the t3x step"*) ;;
  *) hold "sf_queue did not stop where expected :: $(echo "$last" | cut -c1-200)";;
esac
for ref in $T3X; do
  [ -f "$A/$ref.record.json" ] && [ -n "$(recf "$ref" checkpoint)" ] || hold "no record/head for $ref"
done
rm -f "$OUT/STOP_QUEUE"
say "sf_t3x_gate: t3x step done; STOP_QUEUE cleared; turn-band reads on [$T3X]"
for ref in $T3X; do
  ck=$(recf "$ref" checkpoint); st=$(recf "$ref" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || hold "no checkpoint/stamp for $ref"
  ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval "${ref}_tb" "$ck" "$st" "$BAND_MAN" turn_band__omni || hold "turn band $ref failed"
  [ -f "$OUT/STOP_QUEUE" ] && hold "STOP_QUEUE set during the t3x turn-band reads"
done
g=$("$PY" "$S/sf_read.py" --t3x-rule 2>&1)
say "T3XRULE :: $(echo "$g" | tail -1 | cut -c1-500)"
settles=$(echo "$g" | tail -1 | python3 -c "import json,sys
try: d=json.loads(sys.stdin.read()); print('yes' if d.get('in_band_settles') is True and d.get('reading') not in ('INCOMPLETE','RESUME ARTIFACT') else 'no')
except Exception: print('no')")
case "$g" in *'"PLATEAU"'*) ;; *) say "NOTIFY t3x rule: $(echo "$g" | tail -1 | cut -c1-200) -- the user's call (CHARTER 13.3)";; esac
if [ "$settles" != yes ]; then hold "the t3x in-band turn does not settle (or the read is incomplete)"; fi
say "sf_t3x_gate: in-band turn settles -> side-first resumes (FROM_STEP=c4)"
FROM_STEP=c4 setsid nohup bash "$S/sf_queue.sh" > "$OUT/sf_queue_c4.log" 2>&1 < /dev/null &
say "sf_t3x_gate end: sf_queue relaunched from c4 (pid $!)"
