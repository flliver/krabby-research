#!/bin/bash
# P2 add-on (user request 2026-11-14, "a video with the various new terrain types"): one clip per rough-terrain
# kind for the walker c1 and for the depth student C1S, then one tour mp4 per policy. VIDEO ONLY, read by no gate.
# The recorder is experiments/tools/record_categories.py in SEQUENCE MODE on the read's own kind rows
# (manifests/scenarios_rt.yaml), its KRABBY_* environment replaced, before Kit, by the env_vars of that policy's own
# eval of the row (c1: confirm-1 RTC es5 stamps; C1S: the P2 read's P2R es9 stamps), with the row runner's
# --task Isaac-Crab-Hex-Omni-Student-Play-v0 and --policy-role teacher | student (the recorder's 2026-11-14 options).
# Bands = the P2 read's: slopes / ledge_up / ledge_down / pits b2, hills / gravel / rubble b3. Holds kept: fwd,
# lat_l, diag, back (the row's own 8 s each, after its 1.5 s prologue; ~4.6 m out at the farthest, past every
# kind's crossing radius); stand and the turns are left out.
# Sources are SHA-CHECKED (each reference's checkpoint_sha256 / scenario_id / role / task; the checkpoints' files).
# VIDEO MEMORY HAZARD rules: the recorder streams its mp4 (imageio, constant memory), never gym RecordVideo; each run
# under systemd-run --user --scope MemoryMax=40G MemorySwapMax=0, SIGTERM-only timeout; DerivedDataCache <= 78 GB and
# >= 30 GB free before each run. One Isaac process at a time: the campaign's .isaac.lock is held for the whole run and
# sh_record_video.sh's busy patterns are waited out before each launch.
#
#   bash p2_terrain_videos.sh --dry      the source checks + the plan; nothing launched or written
#   bash p2_terrain_videos.sh --launch   start --run DETACHED (setsid nohup), log <VOUT>/p2_terrain_videos.log
#   bash p2_terrain_videos.sh --run      the foreground run
#   P2TV_ONLY="c1:slopes_b2"             (optional) run just these policy:row pairs, space-separated (the smoke)
#   P2TV_CAM_EYE / P2TV_CAM_LOOKAT / P2TV_HOLDS / P2TV_TAG   (optional) camera offsets, kept holds, prefix suffix
# Sentinels (the log): "P2TV START", "P2TV STEP <k> START|rc=N", "P2TV PHONE <mp4>", "P2TV END rc=N".
set -u
CD=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)            # 2026-11-10_rough_terrain
W=$(cd "$CD/../../../../../.." && pwd)                           # the worktree root
M=/home/nickmagus/krabby/krabby-research
PKG=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
REC=$PKG/experiments/tools/record_categories.py
SHREC=$PKG/experiments/2026-10-13_stand_height/scripts/sh_record_video.sh
PY=/home/nickmagus/krabby/isaac_venv/bin/python
FF=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
MANIFEST=$CD/manifests/scenarios_rt.yaml
TASK=Isaac-Crab-Hex-Omni-Student-Play-v0
C1=$W/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt
C1_SHA=6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1
C1S=$W/parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-12_15-08-00_RT_ST_s19/model_34499.pt
C1S_SHA=6b48ef33d26b9523c764f48c664897d756838bb1ca70424492f8f7bdf54236a1
RTLR=$M/parkour/outputs/omni_plan/rough_terrain/lr
P2LR=$M/parkour/outputs/omni_plan/rough_terrain_p2/lr
VOUT=$M/parkour/outputs/omni_plan/rough_terrain_p2/videos/terrain
LOG=$VOUT/p2_terrain_videos.log
ISAAC_LOCK=$CD/.isaac.lock
STOPS=("$VOUT/STOP_QUEUE" "$CD/results/p2/STOP_QUEUE" "$CD/results/STOP_QUEUE")
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=78
FREE_MIN_GB=30
HOLDS=${P2TV_HOLDS:-fwd,lat_l,diag,back}
CAM_EYE=${P2TV_CAM_EYE:-1.0,-6.5,0.5}        # low side follow-cam (probe B 2026-11-14: shows ledge rims + far mounds; the 3/4 views showed no relief)
CAM_LOOKAT=${P2TV_CAM_LOOKAT:-1.0,0.0,-0.5}   # offsets from the robot root, world-aligned
TAG=${P2TV_TAG:-}                            # a probe's prefix suffix (tours are built only on a full plan)
ROWS=(slopes_b2 hills_b3 ledge_up_b2 ledge_down_b2 pits_b2 gravel_b3 rubble_b3)
declare -A WHAT=(
  [slopes_b2]="SLOPES 7-13 deg"
  [hills_b3]="HILLS 13-20 deg"
  [ledge_up_b2]="LEDGE UP 0.10-0.19 m"
  [ledge_down_b2]="LEDGE DOWN 0.12-0.22 m"
  [pits_b2]="POTHOLES 0.13-0.22 m deep"
  [gravel_b3]="GRAVEL 4-5 cm, mu 0.2-0.5"
  [rubble_b3]="RUBBLE 0.10-0.15 m"
)
ISAAC_PAT=$(sed -n "s/^ISAAC_PAT='\\(.*\\)'$/\\1/p" "$SHREC")
NONKIT_PAT=$(sed -n "s/^NONKIT_PAT='\\(.*\\)'$/\\1/p" "$SHREC")

MODE=
case "${1:-}" in
  --dry) MODE=dry ;; --launch) MODE=launch ;; --run) MODE=run ;;
  *) awk 'NR > 1 && /^set -u/ {exit} NR > 1 {print}' "$0"; exit 2 ;;
esac

end() { echo "P2TV END rc=$1 ${2:-} $(date '+%F %T')"; exit "$1"; }
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
stop_hit() { local f; for f in "${STOPS[@]}"; do [ -e "$f" ] && { echo "$f"; return 0; }; done; return 1; }
idle() { while busy; do sf=$(stop_hit) && end 9 "($sf exists)"; sleep 20; done; }
ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }
free_gb() { df --output=avail -B1G "$M" | tail -1 | tr -d ' '; }
disk_ok() {
  local d f; d=$(ddc_gb); f=$(free_gb)
  [ "$d" -le "$DDC_MAX_GB" ] || { echo "DerivedDataCache ${d} GB > ${DDC_MAX_GB} GB (move it aside first)"; return 1; }
  [ "$f" -ge "$FREE_MIN_GB" ] || { echo "only ${f} GB free (floor ${FREE_MIN_GB} GB)"; return 1; }
  return 0
}
phone() { "$FF" -y -loglevel error -i "$1" -vf scale=960:-2 -c:v libx264 -crf 30 -preset slow -an "${1%.mp4}_phone.mp4"; }

# ------------------------------------------------------------------------------- the sources (sha-checked, CPU)
# stdout: "REF_<pol>_<row>=<run dir>" assignments (eval'd); stderr: "P2TV SRC OK|BAD ..."; exit 0 all OK, 1 BAD
resolve() {
  python3 -I -B - "$C1" "$C1_SHA" "$C1S" "$C1S_SHA" "$RTLR" "$P2LR" "$MANIFEST" "$TASK" "${ROWS[@]}" <<'PY'
import hashlib, json, shlex, sys
from pathlib import Path
c1, c1_sha, c1s, c1s_sha, rtlr, p2lr, manifest, task = sys.argv[1:9]
rows = sys.argv[9:]
bad, out = [], {}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def ok(m): print(f"P2TV SRC OK  {m}", file=sys.stderr)
for name, ck, want in (("c1", c1, c1_sha), ("C1S", c1s, c1s_sha)):
    if not Path(ck).is_file(): bad.append(f"{name} {ck} missing")
    elif sha(ck) != want: bad.append(f"{name} {ck} sha != {want[:8]}")
    else: ok(f"{name} {ck} sha {want[:8]}")
if not (Path(c1s).parent / "params" / "agent.yaml").is_file(): bad.append("C1S has no params/agent.yaml beside it")
man = Path(manifest).read_text()
pols = {"c1": (Path(rtlr), "RTC", "seed005", c1_sha, "teacher"), "C1S": (Path(p2lr), "P2R", "seed009", c1s_sha, "student")}
for pol, (root, tag, seed, want, role) in pols.items():
    for r in rows:
        sid = f"rt_{r}__omni"
        if f"- id: {sid}\n" not in man: bad.append(f"{sid} not in {manifest}"); continue
        slot = root / f"{tag}_{want[:8]}_rt_{r}" / sid / seed
        stamps = sorted(d for d in slot.glob("*") if (d / "run_meta.json").is_file()) if slot.is_dir() else []
        if not stamps: bad.append(f"{pol} {r}: no run_meta under {slot}"); continue
        d = stamps[-1]
        m = json.loads((d / "run_meta.json").read_text())
        why = [f"{k} {m.get(k)!r}" for k, v in (("checkpoint_sha256", want), ("scenario_id", sid), ("role", role),
                                                 ("task", task)) if m.get(k) != v]
        if why: bad.append(f"{pol} {r} reference {d}: {why}"); continue
        ok(f"{pol} {r} reference {d} ({len(m.get('env_vars') or {})} env vars)")
        out[f"REF_{pol}_{r}"] = str(d)
for b in bad: print(f"P2TV SRC BAD {b}", file=sys.stderr)
for k, v in out.items(): print(f"{k}={shlex.quote(v)}")
sys.exit(1 if bad else 0)
PY
}

[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || { echo "P2TV BAD could not read the busy patterns from $SHREC"; exit 2; }
SRC=$(resolve); src_rc=$?
eval "$SRC"

# the plan: "pol row" pairs (P2TV_ONLY narrows it)
PLAN=()
for pol in c1 C1S; do for r in "${ROWS[@]}"; do PLAN+=("$pol:$r"); done; done
if [ -n "${P2TV_ONLY:-}" ]; then PLAN=($P2TV_ONLY); fi

if [ "$MODE" = dry ]; then
  echo "P2TV DRY sources rc=$src_rc (0 all OK, 1 BAD)"
  for pr in "${PLAN[@]}"; do
    pol=${pr%%:*}; r=${pr#*:}; ref=REF_${pol}_${r}
    echo "P2TV DRY plan $pol rt_${r}__omni holds $HOLDS ref ${!ref:-MISSING} -> $VOUT/${pol}_${r}_sequence.mp4"
  done
  echo "P2TV DRY tours: $VOUT/tour_c1_6d1f29be.mp4, $VOUT/tour_C1S_6b48ef33.mp4 (+ _phone)"
  echo "P2TV DRY DerivedDataCache $(ddc_gb) GB (<= $DDC_MAX_GB), free $(free_gb) GB (>= $FREE_MIN_GB); Isaac busy now: $(busy && echo yes || echo no)"
  [ -x "$FF" ] && echo "P2TV DRY ffmpeg OK" || echo "P2TV DRY BAD no ffmpeg at $FF"
  bash -n "$0" && echo "P2TV DRY bash -n OK"
  [ "$src_rc" = 0 ] || exit 1
  exit 0
fi

if [ "$MODE" = launch ]; then
  [ "$src_rc" = 0 ] || { echo "P2TV REFUSED sources not all OK (rc=$src_rc); run --dry" >&2; exit 2; }
  r=$(disk_ok) || { echo "P2TV REFUSED $r" >&2; exit 3; }
  mkdir -p "$VOUT"
  P2TV_ONLY="${P2TV_ONLY:-}" P2TV_CAM_EYE="$CAM_EYE" P2TV_CAM_LOOKAT="$CAM_LOOKAT" P2TV_HOLDS="$HOLDS" P2TV_TAG="$TAG" \
    setsid nohup bash "$0" --run >> "$LOG" 2>&1 < /dev/null &
  echo "P2TV_LAUNCH pid=$! log=$LOG (watch: ^P2TV END)"
  exit 0
fi

# ------------------------------------------------------------------------------------------------- the run
echo "P2TV START $(date '+%F %T') plan ${PLAN[*]} holds $HOLDS cam eye $CAM_EYE lookat $CAM_LOOKAT tag '${TAG}'"
[ "$src_rc" = 0 ] || end 2 "(sources not all OK: rc=$src_rc)"
mkdir -p "$VOUT"
exec 9>>"$ISAAC_LOCK"
waited=0
until flock -n 9; do
  [ "$waited" = 0 ] && echo "P2TV WAIT $ISAAC_LOCK is held $(date '+%F %T')"; waited=1
  sf=$(stop_hit) && end 9 "($sf exists)"; sleep 30
done
echo "P2TV LOCK $ISAAC_LOCK held"
unset_args=$(env | grep -oE '^(KRABBY|SN_WALKER|SN_STAND|P2|P2R|P2TV|SH|TP|RSD|RT|RTQ)_[A-Za-z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
RUN=(env $unset_args -u PYTHONPATH PYTHONPATH="$W/parkour/parkour_tasks:$W/parkour")
k=0
for pr in "${PLAN[@]}"; do
  pol=${pr%%:*}; r=${pr#*:}; k=$((k + 1)); ref=REF_${pol}_${r}
  [ -n "${!ref:-}" ] || end 2 "(no reference for $pr)"
  if [ "$pol" = c1 ]; then ck=$C1; role=teacher; who="c1 teacher"
  else ck=$C1S; role=student; who="C1S student"; fi
  prefix=${pol}_${r}${TAG}
  [ -e "$VOUT/${prefix}_sequence.mp4" ] && end 2 "($VOUT/${prefix}_sequence.mp4 exists: move it first)"
  sf=$(stop_hit) && end 9 "($sf exists)"
  r2=$(disk_ok) || end 3 "($r2)"
  idle
  echo "P2TV STEP $k START $prefix role=$role ref=${!ref} $(date '+%F %T')"
  ( cd "$W/parkour" && "${RUN[@]}" OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 \
      systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 timeout --signal=TERM 1800 \
      "$PY" "$REC" --headless --enable_cameras --checkpoint "$ck" --eval-run "${!ref}" --manifest "$MANIFEST" \
      --task "$TASK" --policy-role "$role" --sequence "rt_${r}__omni" --sequence-holds "$HOLDS" \
      --cam-eye="$CAM_EYE" --cam-lookat="$CAM_LOOKAT" --out-dir "$VOUT" --prefix "$prefix" --label "${WHAT[$r]} | $who" ) > "$VOUT/${prefix}.log" 2>&1
  rc=$?
  mp4=$VOUT/${prefix}_sequence.mp4
  [ $rc = 0 ] && [ -f "$mp4" ] || { echo "P2TV STEP $k rc=$rc (recorder failed; see $VOUT/${prefix}.log)"; end 1; }
  echo "P2TV STEP $k rc=0 $mp4 $(date '+%F %T')"
done

# the tours: each policy's clips in ROWS order, concatenated (same writer, same codec: stream copy); full plan only
[ -z "${P2TV_ONLY:-}" ] && [ -z "$TAG" ] || end 0 "(probe or partial plan: no tours)"
for pol in c1 C1S; do
  lst=$VOUT/tour_${pol}.txt; : > "$lst"; n=0
  for r in "${ROWS[@]}"; do
    f=$VOUT/${pol}_${r}_sequence.mp4
    [ -f "$f" ] && { echo "file '$f'" >> "$lst"; n=$((n + 1)); }
  done
  [ "$n" = "${#ROWS[@]}" ] || { echo "P2TV TOUR $pol skipped ($n of ${#ROWS[@]} clips)"; continue; }
  sha8=$([ "$pol" = c1 ] && echo "${C1_SHA:0:8}" || echo "${C1S_SHA:0:8}")
  tour=$VOUT/tour_${pol}_${sha8}.mp4
  "$FF" -y -loglevel error -f concat -safe 0 -i "$lst" -c copy "$tour" || end 4 "(concat $pol failed)"
  phone "$tour" || end 4 "(phone copy $pol failed)"
  echo "P2TV PHONE ${tour%.mp4}_phone.mp4"
done
end 0
