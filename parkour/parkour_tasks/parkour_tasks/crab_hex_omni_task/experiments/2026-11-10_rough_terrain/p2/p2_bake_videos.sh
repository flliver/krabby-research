#!/bin/bash
# P2 step 10 (p2/P2.md): the bake proposal's sample videos -- RSD's scripts/bake_videos.sh re-pointed. VIDEO ONLY,
# read by no gate. Three Isaac runs, one at a time, on this worktree's code (PYTHONPATH = the worktree), from
# SHA-CHECKED P2 sources (every one re-checked at the run; a missing one refuses):
#   * WALKER  c1 = RT_s19 model_29500 (the worktree run dir; sha256 6d1f29be...);
#   * SH      results/p2/stand/sh_head.json (ckpt + sha256; the file's sha must equal it), its stage ACCEPTED:
#             gates/sh.json PASS / WATCH, or gates/sh_reanchor.json reanchored (incl. a K4 accept-and-disclose,
#             gates/sh_decision.json); SN = results/p2/stand/sn_head.json, its file sha == the record == SH's
#             --warm-sha (provenance), its stage accepted (gates/sn.json PASS / WATCH or sn_reanchor.json);
#   * REF     SH's newest complete hraise__omni es1 stamp (main phase2/lr/<SH>/hraise__omni/seed001): walker sha =
#             c1, stand specialist sha = SH, recorded at results/p2/pin.json's build, clean;
#   * PICK    results/p2/sel.json "pick" (ckpt, sha256, iter; verdict SELECTED) -- or P2V_PICK_CKPT + P2V_PICK_SHA
#             (+ P2V_PICK_ITER) when the selection record lives elsewhere; the file's sha must equal it;
#   * ONNX    results/p2/onnx_PICK.json (P2V_ONNX_JSON): checkpoint_sha256 = the pick's, complete, parity_ok, every
#             exported file present with its recorded sha256 (provenance; the clip plays the .pt, as RSD's did).
# The runs (RSD's plans, same seed and terrains, the outputs under rough_terrain_p2/videos):
#   1. walker + stand net: sh_record_video.sh --plan mix --seed 1514 --terrain field_medium (F19's / RSD's
#      sh_mix_*_s1514_rough plan), --stand-mask 1 --deadband 0.25 --eval-run REF;
#   2. the same mix on --terrain field_max (..._roughest);
#   3. the depth student alone: play_stream_video.py, Student-Play, 1 env, 3000 steps (60 s at 50 Hz) on the play
#      env of record F19/envs/p3b.env (0 kg; no DR / stress / payload key), from a scratch HARDLINK of the pick in
#      the videos dir, so the export side effect (exported_deploy/ beside it) never touches the student run dir.
# Each mp4 also gets a _phone.mp4 (960 px wide, libx264 crf 30): "P2VQ PHONE <path>" lines -- sending them is the
# parent's job (SendUserFile).
# VIDEO MEMORY HAZARD rules (memory: video-recording-memory-hazard): never gym RecordVideo -- the recorder streams its
# mp4 (sh_record_video.py's writer) and the student clip uses play_stream_video.py (constant-memory imageio stream);
# caps: the recorder's own systemd scope MemoryMax 24G (SH_VIDEO_MEMORY_MAX), the student run systemd-run --scope
# MemoryMax=40G MemorySwapMax=0; SIGTERM-only timeouts; the DerivedDataCache <= 78 GB (SH_DDC_MAX_GB=78) and >= 30 GB
# free before each run.
# One Isaac process at a time (P2.md Left 10): the STAND QUEUE must be idle (its lock <P2_OUT>/stand/
# p2_stand_queue.lock free -- held by this script for its run, so no stand step starts mid-video), then the
# campaign's .isaac.lock is waited for and held, then sh_record_video.sh's own busy patterns are waited out.
#
#   bash p2_bake_videos.sh --dry      every source check + the plan + the recorder's own --dry-run (the seed-1514 mix
#                                     schedules, CPU, nothing written); no lock probed, nothing launched. Exit 0 when
#                                     nothing is BAD (PENDING sources -- the stand or the pick not there yet -- are
#                                     listed), 1 when something present is wrong
#   bash p2_bake_videos.sh --launch   start --run DETACHED (setsid nohup), log <VOUT>/p2_bake_videos.log; prints
#                                     P2VQ_LAUNCH
#   bash p2_bake_videos.sh --run      the foreground run
# Sentinels (the log): "P2VQ START", "P2VQ STEP <k> START|rc=N", "P2VQ PHONE <mp4>", "P2VQ END rc=N".
set -u
CD=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)            # 2026-11-10_rough_terrain
W=$(cd "$CD/../../../../../.." && pwd)                           # the worktree root
M=/home/nickmagus/krabby/krabby-research
PKG=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
S=$PKG/experiments/2026-10-13_stand_height/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
FF=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
C1=$W/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt
C1_SHA=6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1
RES=$CD/results/p2
STAND_REC=${P2V_STAND_REC:-$RES/stand}      # override for the dry-mode tests only (e.g. RSD's results/stand)
SEL=${P2V_SEL_JSON:-$RES/sel.json}
ONNX_REC=${P2V_ONNX_JSON:-$RES/onnx_PICK.json}
P2PIN=$RES/pin.json
ST_ENV=$PKG/experiments/2026-10-16_task4_pipeline/F19/envs/p3b.env
P2OUT=$M/parkour/outputs/omni_plan/rough_terrain_p2
LROUT=$M/parkour/outputs/omni_plan/phase2/lr
VOUT=$P2OUT/videos
LOG=$VOUT/p2_bake_videos.log
STAND_QLOCK=$P2OUT/stand/p2_stand_queue.lock
STAND_QLOG=$P2OUT/stand/queue.log
ISAAC_LOCK=$CD/.isaac.lock
STOPS=("$VOUT/STOP_QUEUE" "$RES/STOP_QUEUE" "$CD/results/STOP_QUEUE")
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=78
FREE_MIN_GB=30
STEPS=3000
SEED=1514
# sh_record_video.sh's busy patterns (Kit launches only: a pytest or CPU helper run is not an Isaac process)
ISAAC_PAT=$(sed -n "s/^ISAAC_PAT='\\(.*\\)'$/\\1/p" "$S/sh_record_video.sh")
NONKIT_PAT=$(sed -n "s/^NONKIT_PAT='\\(.*\\)'$/\\1/p" "$S/sh_record_video.sh")

MODE=
case "${1:-}" in
  --dry) MODE=dry ;; --launch) MODE=launch ;; --run) MODE=run ;;
  *) awk 'NR > 1 && /^set -u/ {exit} NR > 1 {print}' "$0"; exit 2 ;;
esac

end() { echo "P2VQ END rc=$1 ${2:-} $(date '+%F %T')"; exit "$1"; }
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
stop_hit() { local f; for f in "${STOPS[@]}"; do [ -e "$f" ] && { echo "$f"; return 0; }; done; return 1; }
idle() { while busy; do sf=$(stop_hit) && end 9 "($sf exists)"; sleep 20; done; }
ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }
free_gb() { df --output=avail -B1G "$M" | tail -1 | tr -d ' '; }
disk_ok() {
  local d f; d=$(ddc_gb); f=$(free_gb)
  [ "$d" -le "$DDC_MAX_GB" ] || { echo "DerivedDataCache ${d} GB > ${DDC_MAX_GB} GB (move it aside first: the parent's job)"; return 1; }
  [ "$f" -ge "$FREE_MIN_GB" ] || { echo "only ${f} GB free (floor ${FREE_MIN_GB} GB)"; return 1; }
  return 0
}
phone() { "$FF" -y -loglevel error -i "$1" -vf scale=960:-2 -c:v libx264 -crf 30 -preset slow -an "${1%.mp4}_phone.mp4"; }

# ------------------------------------------------------------------------------- the sources (sha-checked, CPU)
# prints plain "K=V" assignments on stdout (eval'd below; NOT exported: P2V_PICK_SHA / P2V_PICK_ITER are also the
# input override names, and an exported copy would reach the --launch child as a half override without
# P2V_PICK_CKPT; every P2V_* value is expanded by this script's own shell only) and "P2VQ SRC OK|PENDING|BAD ..."
# lines on stderr;
# exit 0 = all present and right, 10 = only PENDING, 1 = something present is wrong
resolve() {
  python3 -I -B - "$C1" "$C1_SHA" "$STAND_REC" "$SEL" "$ONNX_REC" "$P2PIN" "$LROUT" "$ST_ENV" \
    "${P2V_PICK_CKPT:-}" "${P2V_PICK_SHA:-}" "${P2V_PICK_ITER:-}" <<'PY'
import hashlib, json, re, shlex, sys
from pathlib import Path
(c1, c1_sha, stand, sel, onnx, pinp, lrout, st_env, o_ck, o_sha, o_it) = sys.argv[1:12]
stand, lrout = Path(stand), Path(lrout)
bad, pend, out = [], [], {}
def rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def ok(m): print(f"P2VQ SRC OK      {m}", file=sys.stderr)
# walker
if not Path(c1).is_file(): bad.append(f"walker c1 {c1} missing")
elif sha(c1) != c1_sha: bad.append(f"walker c1 {c1} sha != {c1_sha[:8]}")
else: ok(f"walker c1 {c1} sha {c1_sha[:8]}"); out["P2V_WALKER"] = c1
pin = rj(pinp) or {}
if not pin.get("code_sha"): bad.append(f"no P2 pin {pinp}")
# SN / SH
sn, sh = rj(stand / "sn_head.json"), rj(stand / "sh_head.json")
def stage_ok(sid):
    g = rj(stand / "gates" / f"{sid}.json") or {}
    r = rj(stand / "gates" / f"{sid}_reanchor.json") or {}
    d = rj(stand / "gates" / f"{sid}_decision.json") or {}
    acc = g.get("verdict") in ("PASS", "WATCH") or bool(r.get("reanchored")) or d.get("decision") == "accept, disclose"
    return acc, g.get("verdict"), r.get("reanchored") or (d.get("decision") or None)
for sid, h in (("sn", sn), ("sh", sh)):
    if not h:
        pend.append(f"{sid.upper()} head ({stand / (sid + '_head.json')}: the stand re-pair has not produced it)")
        continue
    ck = Path(h.get("ckpt") or "")
    if not ck.is_file(): bad.append(f"{sid.upper()} head {ck} missing"); continue
    if sha(ck) != h.get("sha256"): bad.append(f"{sid.upper()} head {ck} sha != its record {str(h.get('sha256'))[:8]}"); continue
    acc, v, ra = stage_ok(sid)
    if not acc:
        (pend if v is None else bad).append(f"{sid.upper()} stage not accepted (gate {v}, reanchored {ra})"); continue
    ok(f"{sid.upper()} {h.get('name')} {ck} sha {h['sha256'][:8]} (gate {v}, reanchored {ra})")
    out[f"P2V_{sid.upper()}"] = str(ck); out[f"P2V_{sid.upper()}_SHA"] = h["sha256"]; out[f"P2V_{sid.upper()}_NAME"] = h.get("name") or ""
if sn and sh and "P2V_SN" in out and "P2V_SH" in out:
    a = [str(x) for x in sh.get("args") or []]
    ws = a[a.index("--warm-sha") + 1] if "--warm-sha" in a else None
    if ws != sn.get("sha256"): bad.append(f"SH --warm-sha {str(ws)[:8]} != SN sha {sn['sha256'][:8]}")
    else: ok(f"SH warm-started from SN {ws[:8]}")
    dec = rj(stand / "gates" / "sh_decision.json")
    if dec: ok(f"SH stage carries a K4 accept-and-disclose record: {dec.get('reason')}")
# REF: SH's newest complete hraise__omni es1 stamp
if "P2V_SH" in out:
    slot = lrout / out["P2V_SH_NAME"] / "hraise__omni" / "seed001"
    ref = None
    for d in sorted(slot.glob("*"), reverse=True) if slot.is_dir() else []:
        if d.is_dir() and ((d / "p2_complete.json").is_file() or (d / "complete.json").is_file()):
            ref = d; break
    if ref is None: bad.append(f"no complete reference stamp under {slot}")
    else:
        m = rj(ref / "run_meta.json") or {}
        cs = rj(ref / "code_state.json") or {}
        spec = ((m.get("p2r_mode") or {}).get("specialist") or {}).get("sha256")
        why = []
        if m.get("checkpoint_sha256") != c1_sha: why.append(f"walker sha {str(m.get('checkpoint_sha256'))[:8]}")
        if spec != out["P2V_SH_SHA"]: why.append(f"stand specialist sha {str(spec)[:8]}")
        if (cs.get("code_sha"), cs.get("eval_sha")) != (pin.get("code_sha"), pin.get("eval_sha")) or str(cs.get("dirty")) != "False":
            why.append(f"build {str(cs.get('code_sha'))[:8]}/{str(cs.get('eval_sha'))[:8]} dirty {cs.get('dirty')}")
        if why: bad.append(f"reference stamp {ref}: {why}")
        else: ok(f"reference stamp {ref} (walker c1, stand SH, P2 pin)"); out["P2V_REF"] = str(ref)
# PICK
if o_ck or o_sha:
    if not (o_ck and o_sha): bad.append("P2V_PICK_CKPT and P2V_PICK_SHA go together")
    pick = {"ckpt": o_ck, "sha256": o_sha, "iter": int(o_it) if o_it else None, "src": "env P2V_PICK_*"}
else:
    s = rj(sel)
    if s is None: pend.append(f"the pick ({sel}: selection has not run)"); pick = None
    elif s.get("verdict") != "SELECTED" or not s.get("pick"):
        pend.append(f"the pick ({sel}: verdict {s.get('verdict')!r})"); pick = None
    else: pick = dict(s["pick"], src=str(sel))
if pick:
    ck = Path(pick.get("ckpt") or "")
    it = pick.get("iter")
    if it is None:
        mm = re.match(r"^model_(\d+)\.pt$", ck.name); it = int(mm.group(1)) if mm else None
    if not ck.is_file(): bad.append(f"pick {ck} missing")
    elif sha(ck) != pick.get("sha256"): bad.append(f"pick {ck} sha != {str(pick.get('sha256'))[:8]}")
    elif it is None: bad.append(f"pick {ck}: no iteration")
    else:
        ok(f"pick {ck} sha {pick['sha256'][:8]} iter {it} ({pick['src']})")
        out.update(P2V_PICK=str(ck), P2V_PICK_SHA=pick["sha256"], P2V_PICK_ITER=str(it))
    o = rj(onnx)
    if o is None: pend.append(f"the pick's ONNX record ({onnx})")
    else:
        why = []
        if o.get("checkpoint_sha256") != pick.get("sha256"): why.append(f"checkpoint_sha256 {str(o.get('checkpoint_sha256'))[:8]} != the pick's")
        if o.get("complete") is not True or o.get("parity_ok") is not True: why.append(f"complete {o.get('complete')} parity_ok {o.get('parity_ok')}")
        if "ok" in o and o.get("ok") is not True: why.append(f"ok {o.get('ok')} (p2_onnx.py: parity + I/O json + sha)")
        for k, f in (o.get("files") or {}).items():
            p = Path(f.get("path") or "")
            if not p.is_file() or sha(p) != f.get("sha256"): why.append(f"{k} {p} missing or sha differs")
        if not o.get("files"): why.append("no files")
        if why: bad.append(f"ONNX record {onnx}: {why}")
        else: ok(f"ONNX {o.get('out_dir')} ({len(o['files'])} files, shas match; parity ok)")
# student env
try:
    keys = [ln.split("=", 1)[0] for ln in Path(st_env).read_text().splitlines() if re.match(r"^KRABBY_[A-Z0-9_]+=", ln)]
    odd = [k for k in keys if re.match(r"^KRABBY_(DR|STRESS|PAYLOAD)_", k)]
    if odd: bad.append(f"{st_env} sets {odd}")
    else: ok(f"student play env {st_env} ({len(keys)} KRABBY_* keys; no DR / stress / payload)")
except OSError: bad.append(f"no {st_env}")
for p in pend: print(f"P2VQ SRC PENDING {p}", file=sys.stderr)
for b in bad: print(f"P2VQ SRC BAD     {b}", file=sys.stderr)
for k, v in out.items(): print(f"{k}={shlex.quote(v)}")   # plain assignments, never export (see above)
sys.exit(1 if bad else 10 if pend else 0)
PY
}

[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || { echo "P2VQ BAD could not read the busy patterns from $S/sh_record_video.sh"; exit 2; }
SRC=$(resolve); src_rc=$?
eval "$SRC"

if [ "$MODE" = dry ]; then
  echo "P2VQ DRY sources rc=$src_rc (0 all OK, 10 PENDING only, 1 BAD)"
  echo "P2VQ DRY outputs $VOUT: sh_mix_RTC1_<SH sha8>_s${SEED}_{rough,roughest}_sequence{,_phone}.mp4, student_<pick sha8>_p3b{,_phone}.mp4"
  echo "P2VQ DRY locks (NOT probed): the stand queue $STAND_QLOCK must be free (held for the run), then wait for + hold $ISAAC_LOCK"
  echo "P2VQ DRY stand queue log last sentinel: $(grep -E '^P2STQ ' "$STAND_QLOG" 2>/dev/null | tail -1 || true)"
  echo "P2VQ DRY DerivedDataCache $(ddc_gb) GB (<= $DDC_MAX_GB), free $(free_gb) GB (>= $FREE_MIN_GB); Isaac busy now: $(busy && echo yes || echo no)"
  [ -x "$FF" ] && echo "P2VQ DRY ffmpeg $FF OK" || echo "P2VQ DRY BAD no ffmpeg at $FF"
  # the recorder's own dry run: the seed-1514 mix schedule per terrain (CPU; nothing written)
  sh8=${P2V_SH_SHA:-PENDING}; sh8=${sh8:0:8}
  for t in field_medium field_max; do
    pre="sh_mix_RTC1_${sh8}_s${SEED}_$([ "$t" = field_medium ] && echo rough || echo roughest)"
    echo "P2VQ DRY recorder --dry-run terrain=$t prefix=$pre:"
    env -u PYTHONPATH SH_VIDEO_OUT="$VOUT" SH_DDC_MAX_GB=$DDC_MAX_GB bash "$S/sh_record_video.sh" --dry-run --plan mix --seed $SEED \
      --terrain "$t" --prefix "$pre" -- --checkpoint "${P2V_WALKER:-$C1}" --stand-ckpt "${P2V_SH:-<SH model_5000.pt>}" \
      --stand-mask 1 --deadband 0.25 --eval-run "${P2V_REF:-<SH hraise__omni es1 stamp>}" 2>&1 | sed 's/^/    /' | grep -E "plan mix|mix:|stands at|busy|DerivedDataCache|systemd-run|would write|sh_record_video.sh:" | head -8
  done
  echo "P2VQ DRY student: systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 timeout --signal=TERM 2400 $PY scripts/rsl_rl/play_stream_video.py --task Isaac-Crab-Hex-Omni-Student-Play-v0 --headless --video --video_length $STEPS --num_envs 1 --enable_cameras --checkpoint $VOUT/student_run_<sha8>/model_<iter>.pt (env $(basename "$ST_ENV"))"
  bash -n "$0" && echo "P2VQ DRY bash -n OK"
  [ "$src_rc" = 1 ] && exit 1
  exit 0
fi

if [ "$MODE" = launch ]; then
  [ "$src_rc" = 0 ] || { echo "P2VQ REFUSED sources not all OK (rc=$src_rc); run --dry" >&2; exit 2; }
  r=$(disk_ok) || { echo "P2VQ REFUSED $r" >&2; exit 3; }
  mkdir -p "$VOUT"
  setsid nohup bash "$0" --run >> "$LOG" 2>&1 < /dev/null &
  echo "P2VQ_LAUNCH pid=$! log=$LOG (watch: ^P2VQ END, ^P2VQ PHONE)"
  exit 0
fi

# ------------------------------------------------------------------------------------------------- the run
echo "P2VQ START $(date '+%F %T')"
[ "$src_rc" = 0 ] || end 2 "(sources not all OK: rc=$src_rc)"
r=$(disk_ok) || end 3 "($r)"
mkdir -p "$VOUT" "$P2OUT/stand"
exec 7>>"$STAND_QLOCK"
waited=0
until flock -n 7; do
  [ "$waited" = 0 ] && echo "P2VQ WAIT the stand queue holds $STAND_QLOCK $(date '+%F %T')"; waited=1
  sf=$(stop_hit) && end 9 "($sf exists)"; sleep 30
done
last=$(grep -E '^P2STQ ' "$STAND_QLOG" 2>/dev/null | tail -1 || true)
echo "P2VQ stand queue idle (lock free; last sentinel: ${last:-none})"
exec 9>>"$ISAAC_LOCK"
waited=0
until flock -n 9; do
  [ "$waited" = 0 ] && echo "P2VQ WAIT $ISAAC_LOCK is held $(date '+%F %T')"; waited=1
  sf=$(stop_hit) && end 9 "($sf exists)"; sleep 30
done
echo "P2VQ LOCK $ISAAC_LOCK held"

# every parent KRABBY_* / walker / stand / phase-2 variable unset; the worktree on PYTHONPATH
unset_args=$(env | grep -oE '^(KRABBY|SN_WALKER|SN_STAND|P2|P2R|SH|TP|RSD|RT|RTQ)_[A-Za-z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
RUN=(env $unset_args -u PYTHONPATH PYTHONPATH="$W/parkour/parkour_tasks:$W/parkour")
SH8=${P2V_SH_SHA:0:8}

k=0
for spec in "field_medium sh_mix_RTC1_${SH8}_s${SEED}_rough" "field_max sh_mix_RTC1_${SH8}_s${SEED}_roughest"; do
  set -- $spec; terrain=$1; prefix=$2; k=$((k + 1))
  sf=$(stop_hit) && end 9 "($sf exists)"
  r=$(disk_ok) || end 3 "($r)"
  [ -e "$VOUT/${prefix}_sequence.mp4" ] && end 2 "($VOUT/${prefix}_sequence.mp4 exists: a new prefix or move it)"
  idle
  echo "P2VQ STEP $k START $prefix terrain=$terrain walker=${C1_SHA:0:8} stand=$SH8 ref=$P2V_REF $(date '+%F %T')"
  for try in $(seq 1 20); do                # exit 3 = busy at the launcher's own check (a race with another launch)
    "${RUN[@]}" SH_VIDEO_OUT="$VOUT" SH_DDC_MAX_GB=$DDC_MAX_GB bash "$S/sh_record_video.sh" --plan mix --seed $SEED \
      --terrain "$terrain" --prefix "$prefix" -- --checkpoint "$P2V_WALKER" --stand-ckpt "$P2V_SH" --stand-mask 1 \
      --deadband 0.25 --eval-run "$P2V_REF"
    rc=$?
    [ $rc = 3 ] || break
    sleep 30; idle
  done
  [ $rc = 0 ] || { echo "P2VQ STEP $k rc=$rc (launcher refused; see $VOUT/$prefix.log)"; end 1; }
  until grep -qE "SHVIDEO END $prefix unit=.* rc=" "$VOUT/$prefix.log" 2>/dev/null; do sleep 20; done
  rc=$(grep -E "SHVIDEO END $prefix unit=.* rc=" "$VOUT/$prefix.log" | tail -1 | sed 's/.* rc=//')
  mp4=$VOUT/${prefix}_sequence.mp4
  [ "$rc" = 0 ] && [ -f "$mp4" ] || { echo "P2VQ STEP $k rc=${rc:-?} (recorder failed; see $VOUT/$prefix.log)"; end 1; }
  phone "$mp4" || { echo "P2VQ STEP $k rc=4 (phone copy failed)"; end 4; }
  echo "P2VQ PHONE ${mp4%.mp4}_phone.mp4"
  echo "P2VQ STEP $k rc=0 $mp4 $(date '+%F %T')"
done

k=$((k + 1))
P8=${P2V_PICK_SHA:0:8}
RD=$VOUT/student_run_$P8
SCK=$RD/model_${P2V_PICK_ITER}.pt
OUTMP4=$VOUT/student_${P8}_p3b.mp4
[ -e "$OUTMP4" ] && end 2 "($OUTMP4 exists)"
[ -e "$RD/videos/play/stream-${STEPS}steps.mp4" ] && end 2 "($RD/videos/play/stream-${STEPS}steps.mp4 exists)"
mkdir -p "$RD" && { [ -e "$SCK" ] || ln "$P2V_PICK" "$SCK"; } || end 2 "(no scratch hardlink of the pick)"
[ "$(sha256sum "$SCK" | cut -d' ' -f1)" = "$P2V_PICK_SHA" ] || end 2 "(scratch copy sha is not the pick's ${P8})"
envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$ST_ENV" | tr '\n' ' ')
sf=$(stop_hit) && end 9 "($sf exists)"
r=$(disk_ok) || end 3 "($r)"
idle
echo "P2VQ STEP $k START student $P8 iter $P2V_PICK_ITER env=$(basename "$ST_ENV") steps=$STEPS $(date '+%F %T')"
( cd "$W/parkour" && "${RUN[@]}" OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 $envs \
    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 timeout --signal=TERM 2400 \
    "$PY" scripts/rsl_rl/play_stream_video.py --task Isaac-Crab-Hex-Omni-Student-Play-v0 --headless --video \
    --video_length $STEPS --num_envs 1 --enable_cameras --checkpoint "$SCK" ) > "$VOUT/student_${P8}_p3b.log" 2>&1
rc=$?
src=$RD/videos/play/stream-${STEPS}steps.mp4
[ $rc = 0 ] && [ -f "$src" ] || { echo "P2VQ STEP $k rc=$rc (play_stream_video failed; see $VOUT/student_${P8}_p3b.log)"; end 1; }
cp "$src" "$OUTMP4" && phone "$OUTMP4" || { echo "P2VQ STEP $k rc=4"; end 4; }
echo "P2VQ PHONE ${OUTMP4%.mp4}_phone.mp4"
echo "P2VQ STEP $k rc=0 $OUTMP4 $(date '+%F %T')"
end 0
