#!/usr/bin/env bash
# PPO-construction arm runner (CHARTER 7; plan step 2 item 1 / step 3 item 8). Adapted from the side bisect's
# hardened scripts/queue_eplen2.sh with its review fixes kept:
#   F1 unit death detected while polling, with the journal tail logged;
#   F2 the head is taken from state.json after the unit ends (an infra retry is followed);
#   F3 the anchored idle check before every launch and every direct eval (never `pgrep -f`);
#   F5 the run dir is found from the orchestrator's run name;
#   evals under `systemd-run --scope` MemoryMax=48G + `timeout 3600`; --accept-candidates-change on the first launch.
# Dropped: F4's FALLBACK_ST (an arm with no driver stamp is a STOP, never another run's env).
#
# Usage: run_arm.sh <reference> [<reference> ...]       references: names of candidates.yaml (<chain>_s<seed>)
#        run_arm.sh --post <reference> <checkpoint>       steps 3-6 only, for a finished run (no launch)
# x6k forks (CHARTER 10, 2026-10-03): RESUME_READ=1 replaces the from-scratch reads (200 / 1000 / live, whose windows a
# run that starts at it 2999 never logs) with ONE resume-continuity read (scripts/resume_read.py: iteration numbering,
# sigma, priv_reg and bound ramps continue the source head; FAIL = a wiring kill + STOP_QUEUE); the source head is the
# fork reference's name minus "_x6k" in $FORK_SRC_ARMS (default $OUT/arms). EXTRA_LR_EVALS="4000 5000" adds the
# both-hands eval of those checkpoints. ARMS_DIR keeps fork records out of the step-3 arms dir verdict.py reads.
# The driver's one experiment-level retry trains as <tag>_e2 in a new run dir and files its evals under <tag>:
# the run-dir search takes the LATEST training line since the launch, the watcher follows a retry (its reads
# restart on the new run dir), and the stamp search strips the _e<k> suffix (2026-10-02 fix: C_lr_rsi__B_s12's
# retry ran with no wiring/dose reads and reached no stamp).
# Each reference runs serially:
#   1. launch_candidates.sh --campaign-dir $C --reference-chain <ref> (first launch: --accept-candidates-change);
#   2. while it trains: progress notes (early_read.py) and, for construction arms, the pre-registered kills
#      (CHARTER 4.2/4.3/6: wiring at it 200, DEAD + dose at it 1000, the 4a divergence kills throughout;
#      `scripts/verdict.py --check-run`), which stop the unit with `systemctl --user stop` (never SIGKILL)
#      unless AUTO_KILL=0. verdict.py's exit code says which kill: 3 WIRING and 4 DOSE are true stops -- the
#      runner touches $OUT/STOP_QUEUE (no further launch) and logs a NOTIFY line; 5 DEAD and 6 DIVERGED kill
#      the run and the queue continues. 2 = not enough telemetry yet: the check is retried at the next poll
#      (a read is latched only once it decided). Every killed run still gets $ARMS/<ref>.record.json with
#      "killed", "killed_kind", "killed_at", so verdict.py counts it (DEAD / DIVERGED = not FAST) or excludes
#      it (WIRING: rerun under a fresh reference; DOSE: the CHARTER 4.3 stop) -- it never silently vanishes;
#   3. the driver-eval stamp -> side22_classify.py + along_cmd.py (a side22_25 driver row);
#   4. two-direction chains (pool carries side_r): the both-hands eval of the head (side22_25lr__omni, SB
#      scenarios_side_lr.yaml, with the arm's own driver-stamp env) -> classify + along; screen chains also get
#      the same eval of model_2500 (CHARTER 4.5 late rise; C0 included);
#   5. odd RMS (sym/odd_rms.py) on the classifying stamp; a record JSON $ARMS/<ref>.record.json for verdict.py;
#   6. the terminal line "DONE <ref>" or "STOP <ref> <reason>" in $LOG. A hand counts FAST only if its feet_ss
#      class AND its along-command class are FAST (CHARTER 4.1), applied here, not by hand; a missing read
#      (no feet class, or a feet-FAST hand without an along class) is NA, never FAST (fail closed).
# The watcher (Monitor on $LOG) push-notifies every line containing " NOTIFY " or " STOP ".
# One-direction chains (C bed) skip step 4: their class is the driver stamp's lat_l.
# A file $OUT/STOP_QUEUE stops the queue before the next launch (the running arm finishes normally); the runner
# writes it itself after a WIRING or DOSE kill. Exit codes: 0 queue done, 4 stopped by STOP_QUEUE, 6 launch refused.
#
# Environment overrides:
#   CODE_ROOT        checkout whose launcher, driver and eval script run (default: the main checkout, where the code
#                    is landed before any run). The driver's own code is the launcher's checkout.
#   PYTHONPATH_EVAL  PYTHONPATH of the direct evals (default $CODE_ROOT/parkour/parkour_tasks:$CODE_ROOT/parkour)
#   CAMPAIGN_DIR     the campaign dir passed to the launcher (default: this script's parent dir)
#   ACCEPT_WHY       the --accept-candidates-change reason on the first launch ('' = do not pass the flag)
#   AUTO_KILL        1 (default) = apply the pre-registered kills; 0 = log them as KILL-ADVISED only
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
CODE_ROOT=${CODE_ROOT:-$R}
C=${CAMPAIGN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
S=$C/scripts
L=$CODE_ROOT/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
EVAL=$CODE_ROOT/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
PYP=${PYTHONPATH_EVAL:-$CODE_ROOT/parkour/parkour_tasks:$CODE_ROOT/parkour}
LRMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/scenarios_side_lr.yaml
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/$(basename "$C")
RUNS=$R/parkour/logs/rsl_rl/crab_hex_omni_flat_walk
OUT=$R/parkour/outputs/omni_plan/ppo_construction
ARMS=${ARMS_DIR:-$OUT/arms}
FORK_SRC_ARMS=${FORK_SRC_ARMS:-$OUT/arms}
RESUME_READ=${RESUME_READ:-0}
EXTRA_LR_EVALS=${EXTRA_LR_EVALS:-}
LROUT=$OUT/lr
SYMCACHE=$OUT/sym_cache
LOG=$OUT/queue.log
ORCH=$C/orchestrator.log
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
AUTO_KILL=${AUTO_KILL:-1}
mkdir -p "$ARMS" "$LROUT" "$SYMCACHE"
[ -d "$ARMS" ] && [ -w "$ARMS" ] || { echo "run_arm.sh: records dir $ARMS missing or not writable (ARMS_DIR=${ARMS_DIR:-})" >&2; exit 2; }
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
wait_unit() { until ! systemctl --user is-active -q "$1"; do sleep 30; done; }
drivers_idle() { while ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"; do sleep 30; done; }
jget() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));v=d.get(sys.argv[2],{}).get(sys.argv[3]);print('' if v is None else v)" "$1" "$2" "$3" 2>/dev/null; }
hand_fast() {  # FEET_CLASS ALONG_CLASS -> the hand's class (FAST only if both FAST; else the stricter read)
  # fail closed: no feet class, or a feet-FAST hand with no along class (along_cmd.py crashed / no json) -> NA
  if [ -z "$1" ]; then echo NA
  elif [ "$1" = FAST ] && [ "$2" = FAST ]; then echo FAST
  elif [ "$1" = FAST ] && [ -n "$2" ]; then echo "$2"
  elif [ "$1" = FAST ]; then echo NA
  else echo "$1"; fi
}
chain_info() {  # REF -> "CHAIN TWO_DIR(0|1) SCREEN(0|1) SEED" from candidates.yaml
  "$PY" - "$C/candidates.yaml" "$1" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
ref = d["references"][sys.argv[2]]
chain = ref.get("chain") or sys.argv[2]
seed = int(ref["seed"])            # the reference's own seed (a rerun's fresh name need not end in _s<seed>)
stages = d["chains"][chain]["stages"]
pool = str((stages[-1].get("overlay") or {}).get("KRABBY_OMNI_CMD_POOL", ""))
two = int("side_r" in pool)
screen = int(len(stages) == 1 and int(stages[0]["iters"]) == 3000 and not chain.startswith(("FB_", "P1_"))
             and str(stages[0].get("resume_from", "scratch")) == "scratch")   # a fork's run dir has no model_2500
print(chain, two, screen, seed)
PY
}
any_stamp_of() {  # RUN_DIR -> newest complete driver stamp of that run (any row)
  local rd=$1 tag; tag=$(basename "$rd" | cut -d_ -f3- | sed -E 's/_e[0-9]+$//')  # a retry's evals file under <tag>
  ls -d "$EV/$tag"/*/seed001/*/ 2>/dev/null | while read -r d; do
    grep -q "$(basename "$rd")" "$d/run_meta.json" 2>/dev/null && [ -f "$d/complete.json" ] && echo "$d"; done | tail -1
}
orch_run_name_after() {  # "YYYY-MM-DD HH:MM:SS" [first|last] -> run name of the first (default) or latest training line at/after it
  python3 - "$1" "$ORCH" "${2:-first}" <<'PY'
import re, sys
t0, path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    lines = open(path, errors="replace")
except OSError:
    sys.exit(0)
hit = None
for line in lines:
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (seed\d+_X\d+_\S+?): Isaac-", line)
    if m and m.group(1) >= t0:
        hit = m.group(2)
        if mode == "first":
            break
if hit:
    print(hit)
PY
}
follow_retry() {  # CURRENT_RUN_DIR -> the newest run dir of the latest training line since the launch ($T0S) when it
  # differs from CURRENT_RUN_DIR: the driver's experiment-level retry trains as <tag>_e2 in a new run dir; an infra
  # retry logs <tag>_r<k> but keeps the run name (2026-10-03 review), so the _r<k> suffix is stripped before the glob
  local rn rd
  rn=$(orch_run_name_after "$T0S" last)
  [ -n "$rn" ] || return 0
  rn=$(echo "$rn" | sed -E 's/_r[0-9]+$//')
  rd=$(ls -dt "$RUNS"/*_"$rn" 2>/dev/null | head -1)
  [ -n "$rd" ] && [ "$rd" != "${1:-}" ] && echo "$rd"
  return 0
}
head_of() {  # REF T0_EPOCH -> ckpt recorded in state.json for REF at/after T0 (follows an infra retry)
  python3 - "$1" "$2" "$C/state.json" <<'PY'
import json, sys, datetime, os
ref, t0, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    recs = json.load(open(path)).get("ladder", {}).get("references", {}).get(ref, [])
except (OSError, ValueError):
    recs = []
for r in reversed(recs):
    at = datetime.datetime.fromisoformat(r["at"]).timestamp()
    if at >= t0 - 60 and os.path.exists(r["ckpt"]):
        print(r["ckpt"]); break
PY
}
FIRST_ACC=1
ACCEPT_WHY=${ACCEPT_WHY-"ppo_construction run_arm.sh: candidates.yaml as of $(date -I)"}
launch() {  # REF -> UNIT, T0, T0S
  local ref=$1 lout args=()
  [ "$FIRST_ACC" = 1 ] && [ -n "$ACCEPT_WHY" ] && args=(--accept-candidates-change "$ACCEPT_WHY")
  UNIT=""
  for try in $(seq 40); do
    drivers_idle
    T0S=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
    lout=$(cd "$R" && "$L" --campaign-dir "$C" --reference-chain "$ref" "${args[@]}" 2>&1)
    UNIT=$(echo "$lout" | head -1)
    case "$UNIT" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    UNIT=""; sleep 60
  done
  [ -z "$UNIT" ] && return 6
  FIRST_ACC=0
  say "launched $ref unit $UNIT (code root $CODE_ROOT)"
}
kill_kind() {  # verdict.py --check-run exit code -> kill kind ('' = not a kill)
  case "$1" in 3) echo wiring ;; 4) echo dose ;; 5) echo dead ;; 6) echo diverged ;; *) echo "" ;; esac
}
kill_arm() {  # REF UNIT RC AT REASON
  local kind; kind=$(kill_kind "$3")
  if [ "$AUTO_KILL" = 1 ]; then
    printf '%s\t%s\t%s\n' "$kind" "$4" "$5" > "$ARMS/$1.killed"
    say "KILL $1 ($kind at it $4: $5): systemctl --user stop $2"
    systemctl --user stop "$2"
    case "$kind" in
      wiring|dose)
        touch "$OUT/STOP_QUEUE"
        say "NOTIFY $1 killed for $kind at it $4: queue STOPPED ($OUT/STOP_QUEUE). CHARTER 4.2/4.3:" \
            "wiring -> fix and rerun under a fresh reference; dose -> check wiring, wiring right = stop and ask" ;;
      *) say "NOTIFY $1 killed $kind at it $4 (counts as not FAST); the queue continues" ;;
    esac
  else
    say "KILL-ADVISED $1 ($kind at it $4: $5) -- AUTO_KILL=0, the run continues"
  fi
}
watch_arm() {  # REF RUN_DIR UNIT CHAIN TRAIN_LOG (background): progress notes + the pre-registered kills
  # A read is LATCHED only once it decided (rc 0 or a kill code); rc 2 (not enough telemetry) retries at the next
  # poll, logged once. verdict.py itself turns telemetry still missing 200 iterations past the window into rc 3.
  local ref=$1 rd=$2 unit=$3 chain=$4 tl=$5 last out rc w200=0 w1000=0 n200=0 n1000=0 kk nrd wres=0 nres=0 nerr=0
  while systemctl --user is-active -q "$unit"; do
    sleep 120
    if [ -n "${T0S:-}" ]; then
      nrd=$(follow_retry "$rd")
      if [ -n "$nrd" ]; then
        say "note $ref: following the driver's retry -> run dir $nrd (the reads restart there)"
        rd=$nrd; tl=$C/logs/$(basename "$nrd" | cut -d_ -f3-)_train.log; w200=0; w1000=0; n200=0; n1000=0
      fi
    fi
    [ -n "$rd" ] || continue
    if [ "${RESUME_READ:-0}" = 1 ]; then     # x6k fork: one resume-continuity read, no from-scratch reads
      [ "$wres" = 1 ] && continue
      out=$("$PY" "$S/resume_read.py" --fork "$rd" --source "$(fork_source_rd "$ref")" --json "$ARMS/${ref}_resume.json" 2>&1); rc=$?
      if [ "$rc" = 2 ]; then
        [ "$nres" = 0 ] && say "RESUME $ref rc 2 (not enough fork telemetry; retrying each poll) :: $(echo "$out" | tail -1 | cut -c1-300)"
        nres=1
      elif [ "$rc" = 0 ] || [ "$rc" = 3 ]; then
        wres=1
        say "RESUME $ref rc $rc :: $(echo "$out" | tail -1 | cut -c1-500)"
        if [ "$rc" = 3 ]; then kill_arm "$ref" "$unit" 3 resume "$(echo "$out" | tail -1 | cut -c1-300)"; [ "$AUTO_KILL" = 1 ] && return; fi
      else                                    # a reader crash is not a verdict: logged once, retried, never a kill
        [ "$nerr" = 0 ] && say "NOTIFY RESUME $ref rc $rc: reader error, not a verdict (retrying each poll) :: $(echo "$out" | tail -1 | cut -c1-300)"
        nerr=1
      fi
      continue
    fi
    last=$("$PY" "$S/verdict.py" --last-iter "$rd" 2>/dev/null); last=${last:-0}
    if [ "$w200" = 0 ] && [ "$last" -ge 210 ]; then
      out=$("$PY" "$S/verdict.py" --check-run "$rd" --chain "$chain" --at 200 --train-log "$tl" 2>&1); rc=$?
      if [ "$rc" = 2 ]; then
        [ "$n200" = 0 ] && say "CHECK200 $ref rc 2 (not enough telemetry; retrying each poll) :: $(echo "$out" | tail -1 | cut -c1-300)"
        n200=1
      else
        w200=1
        say "CHECK200 $ref rc $rc :: $(echo "$out" | tail -1 | cut -c1-400)"
        kk=$(kill_kind "$rc")
        if [ -n "$kk" ]; then kill_arm "$ref" "$unit" "$rc" 200 "$(echo "$out" | tail -1 | cut -c1-300)"; [ "$AUTO_KILL" = 1 ] && return; fi
      fi
    fi
    if [ "$w1000" = 0 ] && [ "$last" -ge 1060 ]; then
      out=$("$PY" "$S/verdict.py" --check-run "$rd" --chain "$chain" --at 1000 --train-log "$tl" 2>&1); rc=$?
      if [ "$rc" = 2 ]; then
        [ "$n1000" = 0 ] && say "CHECK1000 $ref rc 2 (not enough telemetry; retrying each poll) :: $(echo "$out" | tail -1 | cut -c1-300)"
        n1000=1
      else
        w1000=1
        say "CHECK1000 $ref rc $rc :: $(echo "$out" | tail -1 | cut -c1-400)"
        say "EARLY $ref $("$PY" "$S/early_read.py" "$rd" 1000 2>/dev/null | grep EARLY | cut -c1-120) (progress note only)"
        kk=$(kill_kind "$rc")
        if [ "$kk" = dead ] && [ "${ADVISE_DEAD:-0}" = 1 ]; then   # CHARTER 11.3 (FB6k): DEAD-at-1000 is uncalibrated
          say "KILL-ADVISED $ref dead at it 1000 (ADVISE_DEAD=1: not stopped) :: $(echo "$out" | tail -1 | cut -c1-300)"
        elif [ -n "$kk" ]; then kill_arm "$ref" "$unit" "$rc" 1000 "$(echo "$out" | tail -1 | cut -c1-300)"; [ "$AUTO_KILL" = 1 ] && return; fi
      fi
    fi
    # 4a divergence kills, every poll (an immediate rc 0 "n/a" for every other arm, without reading tfevents)
    out=$("$PY" "$S/verdict.py" --check-run "$rd" --chain "$chain" --at live --train-log "$tl" 2>&1); rc=$?
    if [ "$rc" = 6 ]; then
      say "CHECKLIVE $ref rc 6 :: $(echo "$out" | tail -1 | cut -c1-400)"
      kill_arm "$ref" "$unit" 6 live "$(echo "$out" | tail -1 | cut -c1-300)"; [ "$AUTO_KILL" = 1 ] && return
    fi
  done
}
fork_source_rd() {  # FORK_REF -> the run dir of the head its chain forks: candidates.yaml's resume_from.reference, then
  # that reference's record in $FORK_SRC_ARMS (a fresh-name rerun of a fork names the same reference; 2026-10-03 review)
  python3 - "$C/candidates.yaml" "$1" "$FORK_SRC_ARMS" <<'PY' 2>/dev/null
import json, sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
chain = d["references"][sys.argv[2]].get("chain") or sys.argv[2]
src = d["chains"][chain]["stages"][0]["resume_from"]["reference"]
print(json.load(open(f"{sys.argv[3]}/{src}.record.json"))["run_dir"])
PY
}
lr_eval() {  # NAME CHECKPOINT ENV_STAMP -> sets LR_DIR LRL LRR (hand classes, feet AND along)
  local name=$1 ck=$2 st=$3 envs d fl fr al ar
  LR_DIR=""; LRL=""; LRR=""
  envs=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))" "$st/run_meta.json")
  drivers_idle
  ( cd "$R/parkour" && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH="$PYP" $envs \
      $CAP timeout 3600 "$PY" "$EVAL" --headless --manifest "$LRMAN" --scenario side22_25lr__omni \
      --checkpoint "$ck" --save-raw --no-plot --output-root "$LROUT/$name" ) > "$LROUT/$name.log" 2>&1
  d=$(ls -d "$LROUT/$name"/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f "$d/complete.json" ]; then return 1; fi
  LR_DIR=${d%/}
  "$PY" "$S/side22_classify.py" --json "$ARMS/${name}_lr.json" "$name=$LR_DIR" 2>&1 | grep -E "lat_|CLASS" >> "$LOG"
  "$PY" "$S/along_cmd.py" --json "$ARMS/${name}_lr_along.json" "$name=$LR_DIR" 2>&1 | grep -E "lat_|CLASS" >> "$LOG"
  fl=$(jget "$ARMS/${name}_lr.json" "$name" class); fr=$(jget "$ARMS/${name}_lr.json" "$name" class_r)
  al=$(jget "$ARMS/${name}_lr_along.json" "$name" class_along); ar=$(jget "$ARMS/${name}_lr_along.json" "$name" class_along_r)
  LRL=$(hand_fast "$fl" "$al"); LRR=$(hand_fast "$fr" "$ar")
  say "EVENT ${name}_lr feet L $fl R $fr | along L $al R $ar | hand L $LRL R $LRR"
}
write_record() {  # REF [KILLED_FILE] -> $ARMS/REF.record.json (every path verdict.py needs; a killed run too)
  REF=$1 KILLED_FILE=${2:-} CHAIN=$CHAIN TWO=$TWO SCREEN=$SCREEN SEED=$SEED UNIT=$UNIT RD=$RD CK=$CK ST=$ST \
  LRD=${LRD_HEAD:-} LRD2500=${LRD_2500:-} \
  HANDL=${HANDL:-} HANDR=${HANDR:-} RUNCLS=${RUNCLS:-} ARMS=$ARMS TRAINLOG=$TRAINLOG CODE_ROOT=$CODE_ROOT \
  EXTRA_DONE=${EXTRA_DONE:-} \
  python3 - <<'PY'
import datetime, json, os
e = os.environ
ref, arms = e["REF"], e["ARMS"]
chain = e["CHAIN"]
bed, _, arm = chain.rpartition("__")
p = lambda name: os.path.join(arms, name) if os.path.isfile(os.path.join(arms, name)) else None
killed = {}
if e.get("KILLED_FILE") and os.path.isfile(e["KILLED_FILE"]):
    kind, at, reason = (open(e["KILLED_FILE"]).read().rstrip("\n").split("\t", 2) + ["", "", ""])[:3]
    killed = {"killed": reason or kind or "killed", "killed_kind": kind or None,
              "killed_at": int(at) if at.isdigit() else (at or None)}
rec = {
    "ref": ref, "chain": chain, "bed": bed or chain, "arm": arm if bed else "", "seed": int(e["SEED"]),
    "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"), **killed,
    "two_direction": e["TWO"] == "1", "screen_chain": e["SCREEN"] == "1", "unit": e["UNIT"],
    "run_dir": e["RD"], "checkpoint": e["CK"], "train_log": e["TRAINLOG"], "code_root": e["CODE_ROOT"],
    "driver_stamp": e["ST"] or None,
    "driver_class_json": p(f"{ref}.json"), "driver_along_json": p(f"{ref}_along.json"),
    "lr_stamp": e["LRD"] or None, "lr_class_json": p(f"{ref}_lr.json"), "lr_along_json": p(f"{ref}_lr_along.json"),
    "lr2500_stamp": e["LRD2500"] or None, "lr2500_class_json": p(f"{ref}_m2500_lr.json"),
    "lr2500_along_json": p(f"{ref}_m2500_lr_along.json"),
    "odd_rms_json": p(f"{ref}_odd.json"),
    "hand_l": e["HANDL"] or None, "hand_r": e["HANDR"] or None, "run_class_runner": e["RUNCLS"] or None,
    "note": "run_class_runner is the runner's FAST rule only (feet AND along); verdict.py adds the tripwire.",
}
extra = {}
for item in (e.get("EXTRA_DONE") or "").split():
    n, _, d = item.partition("=")
    extra[n] = {"stamp": d, "class_json": p(f"{ref}_m{n}_lr.json"), "along_json": p(f"{ref}_m{n}_lr_along.json")}
if extra:
    rec["extra_lr"] = extra
if p(f"{ref}_resume.json"):
    rec["resume_read_json"] = p(f"{ref}_resume.json")
json.dump(rec, open(os.path.join(arms, f"{ref}.record.json"), "w"), indent=1)
PY
}
run_ref() {  # REF -> 0 DONE, 1 STOP (logged), 2 killed (recorded), 6 launch refused
  local ref=$1 rn=""
  RD=""; CK=""; ST=""; LRD_HEAD=""; LRD_2500=""; HANDL=""; HANDR=""; RUNCLS=""; TRAINLOG=""; SEED=""; EXTRA_DONE=""
  read -r CHAIN TWO SCREEN SEED < <(chain_info "$ref") || { say "STOP $ref not a reference of $C/candidates.yaml"; return 1; }
  [ -n "${CHAIN:-}" ] && [ -n "${SEED:-}" ] || { say "STOP $ref not a reference of $C/candidates.yaml"; return 1; }
  rm -f "$ARMS/$ref.killed"
  launch "$ref" || { say "STOP $ref launch refused 40 times"; return 6; }
  for _ in $(seq 90); do
    rn=$(orch_run_name_after "$T0S" last)
    if [ -n "$rn" ]; then RD=$(ls -dt "$RUNS"/*_"$(echo "$rn" | sed -E 's/_r[0-9]+$//')" 2>/dev/null | head -1); [ -n "$RD" ] && break; fi
    systemctl --user is-active -q "$UNIT" || break
    sleep 10
  done
  if [ -n "$RD" ]; then
    say "$ref run dir $RD"
    watch_arm "$ref" "$RD" "$UNIT" "$CHAIN" "$C/logs/${rn}_train.log" &
  elif systemctl --user is-active -q "$UNIT"; then
    say "note $ref: run dir not found yet (unit alive); the watcher keeps looking each poll"
    watch_arm "$ref" "" "$UNIT" "$CHAIN" "" &
  fi
  wait_unit "$UNIT"
  wait
  tail -2 "$ORCH" 2>/dev/null | cut -c1-300 >> "$LOG"
  if [ -f "$ARMS/$ref.killed" ]; then
    [ -n "$rn" ] && TRAINLOG=$C/logs/${rn}_train.log
    write_record "$ref" "$ARMS/$ref.killed"
    say "STOP $ref killed: $(tr '\t' ' ' < "$ARMS/$ref.killed" | cut -c1-300) -- recorded in $ARMS/$ref.record.json" \
        "(relaunching this reference would RESUME the stopped run: rerun under a fresh reference name)"
    return 2
  fi
  post_ref "$ref"
}
post_ref() {  # REF -> steps 3-6 for a finished run: CK preset (--post) or the head recorded in state.json
  local ref=$1 tag fl al
  [ -n "$CK" ] || CK=$(head_of "$ref" "$T0")
  if [ -z "$CK" ]; then
    say "STOP $ref no head recorded in state.json after launch; unit journal: $(journalctl --user -u "$UNIT" -n 3 --no-pager 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    return 1
  fi
  RD=$(dirname "$CK"); tag=$(basename "$RD" | cut -d_ -f3-)
  TRAINLOG=$C/logs/${tag}_train.log
  ST=$(any_stamp_of "$RD"); ST=${ST%/}
  if [ -z "$ST" ]; then say "STOP $ref no complete driver eval stamp for $RD"; write_record "$ref"; return 1; fi
  if grep -q '"scenario_id": "side22_25__omni"' "$ST/run_meta.json"; then
    "$PY" "$S/side22_classify.py" --json "$ARMS/$ref.json" "$ref=$ST" 2>&1 | grep -E "lat_|CLASS" >> "$LOG"
    "$PY" "$S/along_cmd.py" --json "$ARMS/${ref}_along.json" "$ref=$ST" 2>&1 | grep -E "lat_|CLASS" >> "$LOG"
    fl=$(jget "$ARMS/$ref.json" "$ref" class); al=$(jget "$ARMS/${ref}_along.json" "$ref" class_along)
    say "EVENT $ref driver feet L $fl | along L $al"
    [ "$TWO" = 0 ] && HANDL=$(hand_fast "$fl" "$al")
  fi
  if [ "$TWO" = 1 ]; then
    if ! lr_eval "$ref" "$CK" "$ST"; then say "STOP $ref both-hands eval produced no complete stamp (log $LROUT/$ref.log)"; write_record "$ref"; return 1; fi
    LRD_HEAD=$LR_DIR; HANDL=$LRL; HANDR=$LRR
    if [ "$SCREEN" = 1 ]; then
      if [ -f "$RD/model_2500.pt" ] && lr_eval "${ref}_m2500" "$RD/model_2500.pt" "$ST"; then
        LRD_2500=$LR_DIR
      else
        say "note $ref: the model_2500 both-hands eval failed or has no checkpoint (the late-rise read is incomplete)"
      fi
    fi
    EXTRA_DONE=""
    for n in $EXTRA_LR_EVALS; do
      if [ -f "$RD/model_$n.pt" ] && lr_eval "${ref}_m$n" "$RD/model_$n.pt" "$ST"; then EXTRA_DONE="$EXTRA_DONE $n=$LR_DIR"
      else say "note $ref: the model_$n both-hands eval failed or has no checkpoint"; fi
    done
    if [ "$HANDL" = FAST ] && [ "$HANDR" = FAST ]; then RUNCLS=FAST; else RUNCLS=NOT-FAST; fi
    "$PY" "$S/sym/odd_rms.py" --cache "$SYMCACHE" --json "$ARMS/${ref}_odd.json" "$ref=$LRD_HEAD" 2>&1 | grep HAND | sed "s/^/  [odd] $ref /" >> "$LOG"
  else
    if [ "$HANDL" = FAST ]; then RUNCLS=FAST; else RUNCLS=NOT-FAST; fi
    "$PY" "$S/sym/odd_rms.py" --cache "$SYMCACHE" --json "$ARMS/${ref}_odd.json" "$ref=$ST" 2>&1 | grep HAND | sed "s/^/  [odd] $ref /" >> "$LOG"
  fi
  write_record "$ref"
  say "EVENT $ref CLASS hand L ${HANDL:-NA} R ${HANDR:-NA} -> run $RUNCLS (runner rule: feet AND along; tripwire in verdict.py)"
  say "DONE $ref"
  return 0
}

if [ "${1:-}" = "--post" ]; then
  [ $# -eq 3 ] || { echo "usage: $0 --post <reference> <checkpoint>" >&2; exit 2; }
  RD=""; CK=$3; ST=""; LRD_HEAD=""; LRD_2500=""; HANDL=""; HANDR=""; RUNCLS=""; TRAINLOG=""; UNIT="(post-only)"; T0=0
  read -r CHAIN TWO SCREEN SEED < <(chain_info "$2") || { echo "not a reference of $C/candidates.yaml: $2" >&2; exit 2; }
  [ -n "${CHAIN:-}" ] && [ -f "$CK" ] || { echo "no reference $2 or no checkpoint $CK" >&2; exit 2; }
  say "run_arm post-only: $2 $CK"
  post_ref "$2"; exit $?
fi
[ $# -ge 1 ] || { echo "usage: $0 <reference> [<reference> ...]" >&2; exit 2; }
say "run_arm start: $* (campaign $C, code root $CODE_ROOT, AUTO_KILL $AUTO_KILL, ARMS $ARMS, RESUME_READ $RESUME_READ, EXTRA_LR_EVALS '$EXTRA_LR_EVALS')"
for ref in "$@"; do
  if [ -f "$OUT/STOP_QUEUE" ]; then say "queue stopped by $OUT/STOP_QUEUE before $ref"; exit 4; fi
  run_ref "$ref"; rc=$?
  [ "$rc" = 6 ] && exit 6
done
say "run_arm end: $*"
