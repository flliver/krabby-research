#!/usr/bin/env bash
# Omni phase-2 arm runner (CHARTER 5 / 8). Adapted from ../2026-10-01_ppo_construction/scripts/run_arm.sh, with its
# hardened review fixes kept (unit death detected while polling; the head taken from state.json after the unit ends,
# following an infra retry; the anchored idle check before every launch and every direct eval -- never `pgrep -f`; the
# run dir found from the orchestrator's run name; evals under `systemd-run --scope` MemoryMax=48G + `timeout 3600`).
# Every path is repointed to this campaign: OUT=parkour/outputs/omni_plan/phase2, RUNS=crab_hex_omni_teacher (base 2a
# trains Omni-Teacher), EV=gait_eval/omni_phases/<this dir>.
#
# Usage: run_arm.sh <reference> [<reference> ...]          references of candidates.yaml (P2__<arm>_s<seed>)
#        run_arm.sh --post <reference> <checkpoint>          the post steps only, for a finished run (no launch)
#        run_arm.sh --lr-eval <name> <checkpoint> <env_stamp>
#                   side22_25lr__omni (both hands) with that stamp's env -> $ARMS/<name>_lr{,_along}.json + sidecar
#        run_arm.sh --row-eval <name> <checkpoint> <env_stamp> <manifest> <scenario> <env_seed>
#                   one eval row at env_seed <k> (the harness's --env-seed; never hard-coded) with that stamp's env
#                   -> $LROUT/<name>/<scenario>/seed00<k>/<ts> + its code-state sidecar
#
# Each reference runs serially:
#   0. refusals before anything is launched: the chain carries a placeholder (an overlay value starting "TBD" or
#      containing "PENDING": CHARTER 4/5 values not fixed yet) -> "STOP <ref> placeholder", exit 7; the fork source's
#      sha256 differs from anchors.yaml (CHARTER 2) -> exit 7; the live reads' preflight fails (p2_live.py --preflight
#      against $CODE_ROOT: every arm and CAL2 need the obstacle-tile level / promotion metrics emitted, E1 / F1 a
#      replay_gate.json magnitude, F1 its stand-only construction) -> exit 7, so no run is launched only to be killed.
#   1. launch_candidates.sh --campaign-dir $C --reference-chain <ref> [--candidates <launch view>]. The driver validates
#      the WHOLE candidates file, so while ANY chain still carries a placeholder the launch goes through
#      $C/launch_view.yaml = candidates.yaml minus every placeholder-carrying chain and its references (nothing else
#      changed); with none left, candidates.yaml itself.
#   2. while it trains, the CHARTER 8 live reads, iterations counted FROM THE FORK START (the fork's first logged
#      iteration, 14996 for a model_14995 fork):
#        resume  (once) the precedent's resume_read.py vs the source head's run dir (fork_source_rd resolves the chain's
#                `resume_from: {checkpoint}`): rc 3 = kill;
#        wiring  at +200 (p2_live.py --at 200: incl. the overlay audit and term liveness);
#        terrain at +1000 (p2_live.py --at 1000);
#        collapse every poll after +300 (p2_live.py --at collapse).
#      rc 2 = not enough telemetry (retried next poll, logged once); rc 3 = KILL: `systemctl --user stop` (never
#      SIGKILL), touch $OUT/STOP_QUEUE, a NOTIFY line -- EVERY phase-2 kill stops the queue and asks. A reader crash
#      (any other rc) is logged once and retried, never a kill. AUTO_KILL=0 only advises.
#   3. post: the head (state.json); for an arm, its driver suite stamp gets its code-state sidecar; for a calibration
#      smoke (P2__CAL*), the dump audit (p2_live.py --smoke-audit; CAL2 also needs moving terrain levels) -- a FAIL is
#      a STOP. A record $ARMS/<ref>.record.json; the terminal line "DONE <ref>" or "STOP <ref> <reason>".
# The code-state sidecar (CHARTER 5) is <stamp>/code_state.json: run_candidates.code_sha(), eval_sha() and a dirty flag
# on the code + eval pathspecs (run_candidates.dirty_paths()). Record commits and tool-file edits move none of them.
# The watcher (Monitor on $LOG) push-notifies every line containing " NOTIFY " or " STOP ".
# A file $OUT/STOP_QUEUE stops the queue before the next launch. Exit codes: 0 done, 4 stopped by STOP_QUEUE,
# 5 a reference stopped (STOP_ON_REF_STOP=1), 6 launch refused, 7 refused before launch (placeholder / sha).
#
# Environment overrides: CODE_ROOT, PYTHONPATH_EVAL, CAMPAIGN_DIR, ACCEPT_WHY, AUTO_KILL (precedent meanings);
# ARMS_DIR (records dir, default $OUT/arms).
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
CODE_ROOT=${CODE_ROOT:-$R}
S=$(cd "$(dirname "$0")" && pwd)
C=${CAMPAIGN_DIR:-$(cd "$S/.." && pwd)}
PREC=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-01_ppo_construction/scripts
L=$CODE_ROOT/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
EVAL=$CODE_ROOT/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
PYP=${PYTHONPATH_EVAL:-$CODE_ROOT/parkour/parkour_tasks:$CODE_ROOT/parkour}
LRMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/scenarios_side_lr.yaml
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/$(basename "$C")
RUNS=$R/parkour/logs/rsl_rl/crab_hex_omni_teacher
OUT=${P2_OUT:-$R/parkour/outputs/omni_plan/phase2}
ARMS=${ARMS_DIR:-$OUT/arms}
LROUT=$OUT/lr
LOG=$OUT/queue.log
STOPQ=$OUT/STOP_QUEUE
ORCH=$C/orchestrator.log
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
AUTO_KILL=${AUTO_KILL:-1}
mkdir -p "$ARMS" "$LROUT"
[ -d "$ARMS" ] && [ -w "$ARMS" ] || { echo "run_arm.sh: records dir $ARMS missing or not writable" >&2; exit 2; }
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
wait_unit() { until ! systemctl --user is-active -q "$1"; do sleep 30; done; }
drivers_idle() { while ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"; do sleep 30; done; }
jget() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));v=d.get(sys.argv[2],{}).get(sys.argv[3]);print('' if v is None else v)" "$1" "$2" "$3" 2>/dev/null; }
hand_fast() {  # FEET_CLASS ALONG_CLASS -> the hand's class (FAST only if both FAST; fail closed: NA)
  if [ -z "$1" ]; then echo NA
  elif [ "$1" = FAST ] && [ "$2" = FAST ]; then echo FAST
  elif [ "$1" = FAST ] && [ -n "$2" ]; then echo "$2"
  elif [ "$1" = FAST ]; then echo NA
  else echo "$1"; fi
}
chain_info() {  # REF -> "CHAIN SEED SMOKE(0|1)" from candidates.yaml
  "$PY" - "$C/candidates.yaml" "$1" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
ref = d["references"][sys.argv[2]]
chain = ref.get("chain") or sys.argv[2]
d["chains"][chain]
print(chain, int(ref["seed"]), int(chain.startswith("P2__CAL")))
PY
}
chain_placeholders() {  # CHAIN -> prints "KEY=VALUE" per placeholder overlay value; exit 1 if any, 0 if none, 2 on error
  "$PY" - "$C/candidates.yaml" "$1" <<'PY'
import re, sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
ch = (d.get("chains") or {}).get(sys.argv[2])
if ch is None:
    sys.exit(2)
bad = [f"{k}={v}" for st in ch.get("stages") or [] for k, v in (st.get("overlay") or {}).items()
       if re.match(r"(?i)^\s*TBD", str(v)) or "PENDING" in str(v).upper()]
print(" ".join(bad))
sys.exit(1 if bad else 0)
PY
}
launch_view() {  # -> prints the --candidates file to launch with ('' = candidates.yaml itself, no placeholder anywhere)
  "$PY" - "$C/candidates.yaml" "$C/launch_view.yaml" <<'PY'
import re, sys, yaml
src, dst = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(src))
ph = lambda v: bool(re.match(r"(?i)^\s*TBD", str(v))) or "PENDING" in str(v).upper()  # noqa: E731
bad = {n for n, ch in (d.get("chains") or {}).items()
       for st in ch.get("stages") or [] for v in (st.get("overlay") or {}).values() if ph(v)}
if not bad:
    print("")
    sys.exit(0)
d["chains"] = {n: ch for n, ch in d["chains"].items() if n not in bad}
d["references"] = {n: r for n, r in (d.get("references") or {}).items() if (r.get("chain") or n) not in bad}
head = ("# GENERATED by scripts/run_arm.sh launch_view -- DO NOT EDIT. candidates.yaml minus the chains that still carry\n"
        "# a placeholder (" + ", ".join(sorted(bad)) + ") and their references; nothing else changed.\n")
open(dst, "w").write(head + yaml.safe_dump(d, sort_keys=False, width=200))
print(dst)
PY
}
verify_head_sha() {  # CHAIN -> 0 when the chain's fork source matches anchors.yaml's sha256 for its seed
  "$PY" - "$C/candidates.yaml" "$C/anchors.yaml" "$1" "$R" <<'PY'
import hashlib, sys, yaml
from pathlib import Path
cands, anchors, chain, repo = yaml.safe_load(open(sys.argv[1])), yaml.safe_load(open(sys.argv[2])), sys.argv[3], Path(sys.argv[4])
rf = cands["chains"][chain]["stages"][0].get("resume_from")
ck = rf.get("checkpoint") if isinstance(rf, dict) else None
want = {a["checkpoint"]: a["sha256"] for a in anchors["anchors"].values()}
if ck not in want:
    print(f"fork source {ck} is not an anchors.yaml head"); sys.exit(1)
h = hashlib.sha256((repo / ck).read_bytes()).hexdigest() if (repo / ck).is_file() else None
if h != want[ck]:
    print(f"sha256 of {ck} is {h}, anchors.yaml records {want[ck]}"); sys.exit(1)
print(f"sha256 ok {h[:12]}")
PY
}
write_sidecar() {  # STAMP_DIR -> STAMP_DIR/code_state.json (CHARTER 5 code-state sidecar)
  ( cd "$CODE_ROOT" && PYTHONPATH="$PYP" CUDA_VISIBLE_DEVICES="" "$PY" - "$1" "$CODE_ROOT" <<'PY'
import datetime, importlib.util, json, sys
from pathlib import Path
stamp, root = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "_p2_rc", root / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py")
rc = importlib.util.module_from_spec(spec); sys.modules["_p2_rc"] = rc; spec.loader.exec_module(rc)
dirty = rc.dirty_paths(root)
out = {"code_sha": rc.code_sha(root), "eval_sha": rc.eval_sha(root), "dirty": bool(dirty), "dirty_paths": dirty,
       "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "writer": "phase2 run_arm.sh"}
json.dump(out, open(stamp / "code_state.json", "w"), indent=1)
print(f"sidecar code {out['code_sha'][:12]} eval {out['eval_sha'][:12]} dirty {out['dirty']}")
PY
  ) >> "$LOG" 2>&1
  [ -f "$1/code_state.json" ]
}
any_stamp_of() {  # RUN_DIR -> newest complete driver stamp of that run (any row)
  local rd=$1 tag; tag=$(basename "$rd" | cut -d_ -f3- | sed -E 's/_(e[0-9]+|c|w)$//')
  ls -d "$EV/$tag"/*/seed001/*/ 2>/dev/null | while read -r d; do
    grep -q "$(basename "$rd")" "$d/run_meta.json" 2>/dev/null && [ -f "$d/complete.json" ] && echo "$d"; done | tail -1
}
orch_run_name_after() {  # "YYYY-MM-DD HH:MM:SS" [first|last] -> run name of the first (default) or latest training line
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
follow_retry() {  # CURRENT_RUN_DIR -> the newest run dir of the latest training line since $T0S when it differs
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
ACCEPT_WHY=${ACCEPT_WHY-"omni phase2 run_arm.sh: candidates as of $(date -I)"}
launch() {  # REF -> UNIT, T0, T0S
  local ref=$1 lout args=() view
  [ "$FIRST_ACC" = 1 ] && [ -n "$ACCEPT_WHY" ] && args=(--accept-candidates-change "$ACCEPT_WHY")
  view=$(launch_view) || { say "STOP $ref: launch view could not be written"; LAUNCH_WHY="launch view failed"; return 6; }
  [ -n "$view" ] && args+=(--candidates "$view") && say "note $ref: placeholders remain in candidates.yaml -- launching through $view"
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
  [ -z "$UNIT" ] && { LAUNCH_WHY="launch refused 40 times"; return 6; }
  FIRST_ACC=0
  say "launched $ref unit $UNIT (code root $CODE_ROOT)"
}
kill_arm() {  # REF UNIT KIND AT REASON -- every phase-2 kill stops the queue (CHARTER 8: "each stops the queue and asks")
  if [ "$AUTO_KILL" = 1 ]; then
    printf '%s\t%s\t%s\n' "$3" "$4" "$5" > "$ARMS/$1.killed"
    say "KILL $1 ($3 at $4: $5): systemctl --user stop $2"
    systemctl --user stop "$2"
    touch "$STOPQ"
    say "NOTIFY $1 killed for $3 at $4: queue STOPPED ($STOPQ). CHARTER 8: stop and ask"
  else
    say "KILL-ADVISED $1 ($3 at $4: $5) -- AUTO_KILL=0, the run continues"
  fi
}
live_read() {  # REF UNIT KIND AT RC OUT -> 0 continue, 1 killed (the watcher returns)
  case "$5" in
    3) kill_arm "$1" "$2" "$3" "$4" "$(echo "$6" | tail -1 | cut -c1-300)"; [ "$AUTO_KILL" = 1 ] && return 1; return 0 ;;
    *) return 0 ;;
  esac
}
fork_source_rd() {  # REF -> the run dir of the head its chain forks (resume_from: {checkpoint: <repo-relative .pt>})
  python3 - "$C/candidates.yaml" "$1" "$R" <<'PY' 2>/dev/null
import sys, yaml
from pathlib import Path
d = yaml.safe_load(open(sys.argv[1]))
chain = d["references"][sys.argv[2]].get("chain") or sys.argv[2]
rf = d["chains"][chain]["stages"][0]["resume_from"]
print((Path(sys.argv[3]) / rf["checkpoint"]).parent)
PY
}
watch_arm() {  # REF RUN_DIR UNIT CHAIN TRAIN_LOG (background): the CHARTER 8 live reads, counted from the fork start
  local ref=$1 rd=$2 unit=$3 chain=$4 tl=$5 out rc it first last nrd src
  local wres=0 w200=0 w1000=0 nres=0 n200=0 n1000=0 ncol=0 nerr=0
  src=$(fork_source_rd "$ref")
  while systemctl --user is-active -q "$unit"; do
    sleep 120
    if [ -n "${T0S:-}" ]; then
      nrd=$(follow_retry "$rd")
      if [ -n "$nrd" ]; then
        say "note $ref: following the driver's retry -> run dir $nrd (the reads restart there)"
        rd=$nrd; tl=$C/logs/$(basename "$nrd" | cut -d_ -f3-)_train.log; wres=0; w200=0; w1000=0
      fi
    fi
    [ -n "$rd" ] || continue
    it=$("$PY" "$S/p2_live.py" --iters "$rd" 2>/dev/null); read -r first last <<< "${it:-0 0}"
    [ "${first:-0}" -gt 0 ] || continue
    if [ "$wres" = 0 ]; then
      out=$("$PY" "$PREC/resume_read.py" --fork "$rd" --source "$src" --json "$ARMS/${ref}_resume.json" 2>&1); rc=$?
      if [ "$rc" = 0 ] || [ "$rc" = 3 ]; then
        wres=1; say "RESUME $ref rc $rc :: $(echo "$out" | tail -1 | cut -c1-400)"
        live_read "$ref" "$unit" resume "it $last" "$rc" "$out" || return 0
      elif [ "$rc" = 2 ]; then
        [ "$nres" = 0 ] && say "RESUME $ref rc 2 (not enough fork telemetry; retrying each poll)"; nres=1
      else
        [ "$nerr" = 0 ] && say "NOTIFY RESUME $ref rc $rc: reader error, not a verdict (retrying) :: $(echo "$out" | tail -1 | cut -c1-300)"; nerr=1
      fi
    fi
    if [ "$w200" = 0 ] && [ "$last" -ge $((first + 210)) ]; then
      out=$(cd "$CODE_ROOT" && PYTHONPATH="$PYP" "$PY" "$S/p2_live.py" --check "$rd" --chain "$chain" --at 200 --train-log "$tl" --candidates "$C/candidates.yaml" --code-root "$CODE_ROOT" 2>&1); rc=$?
      if [ "$rc" = 2 ]; then [ "$n200" = 0 ] && say "CHECK200 $ref rc 2 (retrying each poll) :: $(echo "$out" | tail -1 | cut -c1-300)"; n200=1
      elif [ "$rc" = 0 ] || [ "$rc" = 3 ]; then
        w200=1; say "CHECK200 $ref rc $rc :: $(echo "$out" | tail -1 | cut -c1-500)"
        live_read "$ref" "$unit" wiring "+200" "$rc" "$out" || return 0
      else [ "$nerr" = 0 ] && say "NOTIFY CHECK200 $ref rc $rc: reader error (retrying) :: $(echo "$out" | tail -1 | cut -c1-300)"; nerr=1; fi
    fi
    if [ "$w1000" = 0 ] && [ "$last" -ge $((first + 1060)) ]; then
      out=$("$PY" "$S/p2_live.py" --check "$rd" --chain "$chain" --at 1000 --candidates "$C/candidates.yaml" 2>&1); rc=$?
      if [ "$rc" = 2 ]; then [ "$n1000" = 0 ] && say "CHECK1000 $ref rc 2 (retrying each poll) :: $(echo "$out" | tail -1 | cut -c1-300)"; n1000=1
      elif [ "$rc" = 0 ] || [ "$rc" = 3 ]; then
        w1000=1; say "CHECK1000 $ref rc $rc :: $(echo "$out" | tail -1 | cut -c1-500)"
        live_read "$ref" "$unit" terrain "+1000" "$rc" "$out" || return 0
      else [ "$nerr" = 0 ] && say "NOTIFY CHECK1000 $ref rc $rc: reader error (retrying)"; nerr=1; fi
    fi
    if [ "$last" -ge $((first + 300)) ]; then
      out=$("$PY" "$S/p2_live.py" --check "$rd" --chain "$chain" --at collapse --source "$src" --candidates "$C/candidates.yaml" 2>&1); rc=$?
      if [ "$rc" = 3 ]; then
        say "CHECKCOLLAPSE $ref rc 3 :: $(echo "$out" | tail -1 | cut -c1-400)"
        live_read "$ref" "$unit" collapse "+$((last - first))" "$rc" "$out" || return 0
      elif [ "$rc" = 2 ] && [ "$ncol" = 0 ]; then say "CHECKCOLLAPSE $ref rc 2 (retrying each poll)"; ncol=1; fi
    fi
  done
}
row_eval() {  # NAME CHECKPOINT ENV_STAMP MANIFEST SCENARIO ENV_SEED -> sets ROW_DIR (the complete stamp) or returns 1
  local name=$1 ck=$2 st=$3 man=$4 sc=$5 es=$6 envs d sd
  ROW_DIR=""
  case "$es" in ''|*[!0-9]*) say "STOP row-eval $name $sc: env_seed '$es' is not a positive integer"; return 1 ;; esac
  sd=$(printf 'seed%03d' "$es")
  envs=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))" "$st/run_meta.json")
  drivers_idle
  ( cd "$R/parkour" && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH="$PYP" $envs \
      $CAP timeout 3600 "$PY" "$EVAL" --headless --manifest "$man" --scenario "$sc" --env-seed "$es" \
      --checkpoint "$ck" --save-raw --no-plot --output-root "$LROUT/$name" ${EVAL_EXTRA_ARGS:-} ) > "$LROUT/$name.$sc.$sd.log" 2>&1
  d=$(ls -d "$LROUT/$name"/"$sc"/"$sd"/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f "$d/complete.json" ]; then return 1; fi
  ROW_DIR=${d%/}
  write_sidecar "$ROW_DIR" || { say "STOP row-eval $name $sc: no code-state sidecar written for $ROW_DIR"; return 1; }
}
lr_eval() {  # NAME CHECKPOINT ENV_STAMP -> sets LR_DIR LRL LRR (hand classes, feet AND along)
  local name=$1 ck=$2 st=$3 envs d fl fr al ar
  LR_DIR=""; LRL=""; LRR=""
  envs=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))" "$st/run_meta.json")
  drivers_idle
  ( cd "$R/parkour" && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH="$PYP" $envs \
      $CAP timeout 3600 "$PY" "$EVAL" --headless --manifest "$LRMAN" --scenario side22_25lr__omni --env-seed 1 \
      --checkpoint "$ck" --save-raw --no-plot --output-root "$LROUT/$name" ) > "$LROUT/$name.lr.log" 2>&1
  d=$(ls -d "$LROUT/$name"/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f "$d/complete.json" ]; then return 1; fi
  LR_DIR=${d%/}
  write_sidecar "$LR_DIR" || { say "STOP lr-eval $name: no code-state sidecar written for $LR_DIR"; return 1; }
  "$PY" "$PREC/side22_classify.py" --json "$ARMS/${name}_lr.json" "$name=$LR_DIR" 2>&1 | grep -E "lat_|CLASS" >> "$LOG"
  "$PY" "$PREC/along_cmd.py" --json "$ARMS/${name}_lr_along.json" "$name=$LR_DIR" 2>&1 | grep -E "lat_|CLASS" >> "$LOG"
  fl=$(jget "$ARMS/${name}_lr.json" "$name" class); fr=$(jget "$ARMS/${name}_lr.json" "$name" class_r)
  al=$(jget "$ARMS/${name}_lr_along.json" "$name" class_along); ar=$(jget "$ARMS/${name}_lr_along.json" "$name" class_along_r)
  LRL=$(hand_fast "$fl" "$al"); LRR=$(hand_fast "$fr" "$ar")
  say "EVENT ${name}_lr feet L $fl R $fr | along L $al R $ar | hand L $LRL R $LRR"
}
write_record() {  # REF [KILLED_FILE] -> $ARMS/REF.record.json
  REF=$1 KILLED_FILE=${2:-} CHAIN=$CHAIN SEED=$SEED SMOKE=$SMOKE UNIT=$UNIT RD=$RD CK=$CK ST=$ST ARMS=$ARMS \
  TRAINLOG=$TRAINLOG CODE_ROOT=$CODE_ROOT SMOKE_AUDIT=${SMOKE_AUDIT:-} \
  python3 - <<'PY'
import datetime, json, os
e = os.environ
ref, arms = e["REF"], e["ARMS"]
p = lambda name: os.path.join(arms, name) if os.path.isfile(os.path.join(arms, name)) else None  # noqa: E731
killed = {}
if e.get("KILLED_FILE") and os.path.isfile(e["KILLED_FILE"]):
    kind, at, reason = (open(e["KILLED_FILE"]).read().rstrip("\n").split("\t", 2) + ["", "", ""])[:3]
    killed = {"killed": reason or kind or "killed", "killed_kind": kind or None, "killed_at": at or None}
chain = e["CHAIN"]
rec = {"ref": ref, "chain": chain, "arm": chain.split("__", 1)[-1].rsplit("_s", 1)[0], "seed": int(e["SEED"]),
       "smoke": e["SMOKE"] == "1", "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"), **killed,
       "unit": e["UNIT"], "run_dir": e["RD"] or None, "checkpoint": e["CK"] or None, "train_log": e["TRAINLOG"] or None,
       "code_root": e["CODE_ROOT"], "driver_stamp": e["ST"] or None,
       "resume_read_json": p(f"{ref}_resume.json"), "smoke_audit": e.get("SMOKE_AUDIT") or None}
json.dump(rec, open(os.path.join(arms, f"{ref}.record.json"), "w"), indent=1)
PY
}
run_ref() {  # REF -> 0 DONE, 1 STOP (logged), 2 killed (recorded), 6 launch refused, 7 refused before launch
  local ref=$1 rn="" ph msg
  RD=""; CK=""; ST=""; TRAINLOG=""; SEED=""; SMOKE=0; SMOKE_AUDIT=""; UNIT=""
  read -r CHAIN SEED SMOKE < <(chain_info "$ref" 2>/dev/null) || true
  [ -n "${CHAIN:-}" ] && [ -n "${SEED:-}" ] || { say "STOP $ref not a reference of $C/candidates.yaml"; return 1; }
  ph=$(chain_placeholders "$CHAIN"); case $? in
    0) ;;
    1) say "STOP $ref refused: placeholder value(s) not fixed yet: $ph (CHARTER 4/5) -- NOTIFY"; return 7 ;;
    *) say "STOP $ref refused: placeholder check failed for chain $CHAIN -- NOTIFY"; return 7 ;;
  esac
  msg=$(verify_head_sha "$CHAIN" 2>&1) || { say "STOP $ref refused: $msg (CHARTER 2) -- NOTIFY"; return 7; }
  say "$ref fork source $msg"
  msg=$(cd "$CODE_ROOT" && "$PY" "$S/p2_live.py" --preflight --chain "$CHAIN" --code-root "$CODE_ROOT" \
        --candidates "$C/candidates.yaml" 2>&1) || { say "STOP $ref refused: live-read preflight :: $(echo "$msg" | tail -1 | cut -c1-600) -- NOTIFY"; return 7; }
  say "$ref preflight $(echo "$msg" | tail -1 | cut -c1-200)"
  rm -f "$ARMS/$ref.killed"
  LAUNCH_WHY=""
  launch "$ref" || { say "STOP $ref ${LAUNCH_WHY:-launch failed}"; return 6; }
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
post_ref() {  # REF -> the post steps for a finished run: CK preset (--post) or the head recorded in state.json
  local ref=$1 tag out rc req=""
  [ -n "$CK" ] || CK=$(head_of "$ref" "$T0")
  if [ -z "$CK" ]; then
    say "STOP $ref no head recorded in state.json after launch; unit journal: $(journalctl --user -u "$UNIT" -n 3 --no-pager 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    return 1
  fi
  RD=$(dirname "$CK"); tag=$(basename "$RD" | cut -d_ -f3-)
  TRAINLOG=$C/logs/${tag}_train.log
  if [ "$SMOKE" = 1 ]; then
    [ "$CHAIN" = P2__CAL2_s18 ] && req=--require-terrain-move
    out=$(cd "$CODE_ROOT" && PYTHONPATH="$PYP" "$PY" "$S/p2_live.py" --smoke-audit "$RD" --chain "$CHAIN" --train-log "$TRAINLOG" \
          --candidates "$C/candidates.yaml" $req 2>&1); rc=$?
    SMOKE_AUDIT=$(echo "$out" | tail -1 | cut -c1-2000)
    write_record "$ref"
    say "SMOKE $ref audit rc $rc :: $(echo "$out" | tail -1 | cut -c1-500)"
    if [ "$rc" != 0 ]; then say "STOP $ref calibration-smoke dump audit FAILED (CHARTER 5) -- NOTIFY"; return 1; fi
    say "NOTIFY $ref smoke audit PASS; calibrate with scripts/p2_read.py --calibrate $RD (CHARTER 5: mean track_ratio_obst_p50, last 100 its)"
    say "DONE $ref"
    return 0
  fi
  ST=$(any_stamp_of "$RD"); ST=${ST%/}
  if [ -z "$ST" ]; then say "STOP $ref no complete driver eval stamp for $RD"; write_record "$ref"; return 1; fi
  write_sidecar "$ST" || { say "STOP $ref no code-state sidecar written for the driver stamp $ST"; write_record "$ref"; return 1; }
  write_record "$ref"
  say "EVENT $ref head $CK driver suite stamp $ST (head reads: scripts/p2_queue.sh)"
  say "DONE $ref"
  return 0
}

if [ "${1:-}" = "--lr-eval" ]; then
  [ $# -eq 4 ] || { echo "usage: $0 --lr-eval <name> <checkpoint> <env_stamp>" >&2; exit 2; }
  [ -f "$3" ] && [ -f "$4/run_meta.json" ] || { echo "no checkpoint $3 or no stamp $4" >&2; exit 2; }
  say "run_arm lr-eval: $2 $3 (env $4)"
  if lr_eval "$2" "$3" "$4"; then exit 0; fi
  say "STOP lr-eval $2: no complete stamp (log $LROUT/$2.lr.log)"; exit 1
fi
if [ "${1:-}" = "--row-eval" ]; then
  [ $# -eq 7 ] || { echo "usage: $0 --row-eval <name> <checkpoint> <env_stamp> <manifest> <scenario> <env_seed>" >&2; exit 2; }
  [ -f "$3" ] && [ -f "$4/run_meta.json" ] && [ -f "$5" ] || { echo "missing checkpoint/stamp/manifest" >&2; exit 2; }
  case "$7" in ''|*[!0-9]*|0) echo "env_seed must be a positive integer, got '$7'" >&2; exit 2 ;; esac
  say "run_arm row-eval: $2 $6 env_seed $7 $3 (env $4)"
  if row_eval "$2" "$3" "$4" "$5" "$6" "$7"; then say "EVENT $2 $6 seed$7 stamp $ROW_DIR"; exit 0; fi
  say "STOP row-eval $2 $6 env_seed $7: no complete stamp (log $LROUT/$2.$6.$(printf 'seed%03d' "$7").log)"; exit 1
fi
if [ "${1:-}" = "--post" ]; then
  [ $# -eq 3 ] || { echo "usage: $0 --post <reference> <checkpoint>" >&2; exit 2; }
  RD=""; CK=$3; ST=""; TRAINLOG=""; UNIT="(post-only)"; T0=0; SMOKE_AUDIT=""
  read -r CHAIN SEED SMOKE < <(chain_info "$2" 2>/dev/null) || true
  [ -n "${CHAIN:-}" ] && [ -f "$CK" ] || { echo "no reference $2 or no checkpoint $CK" >&2; exit 2; }
  say "run_arm post-only: $2 $CK"
  post_ref "$2"; exit $?
fi
[ $# -ge 1 ] || { echo "usage: $0 <reference> [<reference> ...]" >&2; exit 2; }
say "run_arm start: $* (campaign $C, code root $CODE_ROOT, AUTO_KILL $AUTO_KILL, ARMS $ARMS)"
for ref in "$@"; do
  if [ -f "$STOPQ" ]; then say "queue stopped by $STOPQ before $ref"; exit 4; fi
  run_ref "$ref"; rc=$?
  [ "$rc" = 6 ] && exit 6
  [ "$rc" = 7 ] && exit 7
  if [ "$rc" != 0 ] && [ "${STOP_ON_REF_STOP:-0}" = 1 ]; then say "STOP run_arm: $ref ended rc $rc (STOP_ON_REF_STOP=1) -- NOTIFY"; exit 5; fi
done
say "run_arm end: $*"
