#!/usr/bin/env python3
"""PLAN step 8 (Phase C preflight): the JOB TABLE the GPU queue (``t4_stepC_queue.sh``) runs, one job at a time.

Order = cheapest disqualifier first: C1 inertness -> C2 eval bridge -> C3 payload -> C4 camera / student -> C5 sweep
anchors -> C6 stand -> C7 full-graph dry pass. Every job has: a name, a group, a kind, a GPU-h estimate, a failure
class (``stop``: a STOP-C trigger -- the queue STOPS; ``continue``: report-only -- the queue goes on and the jobs that
need it are blocked), a completion test read from its OUTPUTS (so the queue is resumable: a complete job is skipped)
and a bash script (written to ``SC/jobs/<name>.sh``; the queue runs it). Jobs whose inputs only exist after an
earlier job (the payload pilot's checkpoints, N_p, the S1 curve's first unstable level, the distill pilots' heads) are
generated from the files on disk at every ``next``; until then ``list`` shows them as placeholders with their estimate.

The pass rules of every ``check`` job are PL/preflight/STEPC_RULES.md (implemented in ``t4_stepC_read.py --check``).

    t4_stepC_jobs.py list [--only C1,C3] [--json]       the job list, status and GPU-h (nothing is launched)
    t4_stepC_jobs.py next [--only ...]                  -> "JOB <name> <isaac 0|1> <gpu_h>" | "STOP <name> <why>" | "END <why>"
    t4_stepC_jobs.py mark <name> <rc>                   -> "DONE|FAIL|STOP <name> <detail>" (state in SC/stepC_state.json)
    t4_stepC_jobs.py retry <name>|failed                clear a failed / stopped job (and its group's checks)
    t4_stepC_jobs.py accept <name> --why TEXT           accept a STOP (recorded; the queue continues past it)
Kit-free: it reads records and composes commands; it never launches anything itself.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t4_stepC_lib as C  # noqa: E402

GROUPS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
TOOLS = C.EXP / "tools"
GATE_ROW = C.TP / "gate_row.sh"
DISTILL = C.TP / "distill.sh"
T4ROW = C.PL / "scripts/t4_eval_row.sh"
RUN_ARM = C.EXP / "2026-10-07_omni_phase2/scripts/run_arm.sh"
SN_DIR = C.EXP / "2026-10-11_stand_network/scripts"
SH_DIR = C.EXP / "2026-10-13_stand_height/scripts"
READ = HERE / "t4_stepC_read.py"
PYP = f"{C.R}/parkour/parkour_tasks:{C.R}/parkour"
CAP = "systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
WALK_PLAY = "Isaac-Crab-Hex-Omni-Walk-Play-v0"
STUDENT_PLAY = "Isaac-Crab-Hex-Omni-Student-Play-v0"
STUDENT_TRAIN = "Isaac-Crab-Hex-Omni-Student-v0"
TEACHER_TRAIN = "Isaac-Crab-Hex-Omni-Teacher-v0"
LEVEL_ENV = {"KRABBY_PAYLOAD_ZERO_FRAC": "0", "KRABBY_PAYLOAD_COM_XY": "0", "KRABBY_PAYLOAD_COM_Z": "0.309"}
SN_OPTS = ["--mode", "dagger", "--arm", "mask", "--seed", "17", "--iterations", "4000", "--smoke", "--smoke-iters", "100"]
SH_OPTS = ["--mode", "dagger", "--arm", "mask", "--seed", "17", "--iterations", "5000", "--deadband", "0.25",
           "--trim-branch", "--trim-min", "0.0999", "--smoke", "--smoke-iters", "100"]
#: distill seed (the pipeline's walker seed for R18)
DSEED = 18


@dataclass
class Job:
    name: str
    group: str
    kind: str                  # row | adhoc | driver | distill | kit | p2row | shrows | stand | cpu | check | dry | placeholder
    gpu_h: float
    on_fail: str               # stop | continue
    script: str = ""
    done: object = None        # callable -> bool (None: never done by files; check jobs use the state)
    requires: tuple = ()
    note: str = ""
    isaac: bool = True
    meta: dict = field(default_factory=dict)


def q(x) -> str:
    return shlex.quote(str(x))


def envfile_args(path: Path, strip=()) -> str:
    """``KEY=VALUE`` words of an env file for ``env`` (values carry no whitespace: tp_lib.env_file_text refuses it)."""
    env = C.TL().read_env_file(Path(path))
    return " ".join(f"{k}={q(v)}" for k, v in sorted(env.items()) if k not in strip)


# ================================================================================================ job builders
def row_job(name: str, group: str, rj: dict, gpu_h: float, on_fail="continue", requires=(), note="",
            manifest=None, scenario=None, env_file=None) -> Job:
    """A gate_row.sh row from a pipeline row_job dict (TP/pipeline.py row_job: env file, manifest, scenario, args,
    trained categories, level file, stripped keys) -- the pipeline's own row composition."""
    man = manifest or rj["manifest"]
    sc = scenario or rj["scenario"]
    ef = env_file or rj["env_file"]
    pre = [f"TP_OUT={q(C.OUT)}", f"TP_ROW_TIMEOUT={int(rj.get('timeout') or 10800)}"]
    if rj.get("level_file"):
        pre.append(f"TP_LEVEL_FILE={q(rj['level_file'])}")
    if rj.get("strip"):
        pre.append(f"TP_STRIP_KEYS={q(' '.join(rj['strip']))}")
    cmd = " ".join(pre + ["bash", q(GATE_ROW), q(rj["name"]), q(rj["ckpt"]), q(ef), q(man), q(sc), str(rj["es"])]
                   + [q(a) for a in rj["args"]])
    ck, es, nm = rj["ckpt"], rj["es"], rj["name"]
    return Job(name=name, group=group, kind="row", gpu_h=gpu_h, on_fail=on_fail, script=cmd + "\n",
               done=lambda: C.row_stamp(nm, sc, es, ck) is not None, requires=tuple(requires), note=note,
               meta={"row_name": nm, "scenario": sc, "es": es, "ckpt": str(ck)})


def prow(stage: str, key: str, ckpt, name: str, *, level=None, env_stage=None, extra=(), strip=()) -> dict:
    """TP/pipeline.py row_job on the R18 plan (its generated env files; stage ``env_stage`` or ``stage``)."""
    rj = C.P().row_job(C.ctx(), stage, key, ckpt=str(ckpt), name=name, env_stage=env_stage, level=level,
                       extra_args=tuple(extra), strip=tuple(strip))
    fam = rj["family"]
    rj["timeout"] = (C.ctx().spec.d["rows"].get(fam) or {}).get("timeout_s") or C.ctx().spec.d.get("row_timeout_s")
    return rj


def raw_row(name: str, ckpt, env_file: Path, manifest: Path, scenario: str, es: int, args=(), level_file=None) -> dict:
    return {"name": name, "ckpt": str(ckpt), "env_file": str(env_file), "manifest": str(manifest),
            "scenario": scenario, "es": es, "args": list(args), "level_file": level_file, "strip": [],
            "timeout": 5400}


def driver_job(name, group, chain, gpu_h, max_h, on_fail="stop", requires=(), note="") -> Job:
    s = f"""set -u
out=$(bash {q(TOOLS / 'launch_candidates.sh')} --campaign-dir {q(C.SC)} --reference-chain {chain} --seed 18 ${{T4C_DRIVER_ARGS:-}} 2>&1)
unit=$(printf '%s\\n' "$out" | grep -E '^cands-' | head -n 1)
[ -n "$unit" ] || {{ echo "launcher printed no unit: $(printf '%s' "$out" | tail -n 3 | tr '\\n' ' ')"; exit 4; }}
echo "unit=$unit"
sleep 20
deadline=$(( $(date +%s) + {int(max_h * 3600)} ))
while systemctl --user is-active -q "$unit"; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "wall bound {max_h} h: systemctl --user stop $unit (the driver's SIGTERM path; never SIGKILL)"
    systemctl --user stop "$unit"; exit 124
  fi
  sleep 30
done
exit 0
"""
    return Job(name=name, group=group, kind="driver", gpu_h=gpu_h, on_fail=on_fail, script=s,
               done=lambda: C.driver_head(chain) is not None, requires=tuple(requires), note=note,
               meta={"chain": chain})


def distill_job(name, group, run_name, ckpt, start_iter, iters, env_file, gpu_h, on_fail="stop", hydra=(),
                requires=(), note="") -> Job:
    h = (" -- " + " ".join(q(x) for x in hydra)) if hydra else ""
    s = (f"set -u\nTP_DISTILL_OUT={q(C.OUT / 'distill')} TP_DISTILL_TIMEOUT={int(max(3600, iters * 40 + 1800))} "
         f"bash {q(DISTILL)} --name {run_name} --checkpoint {q(ckpt)} --env-file {q(env_file)} --seed {DSEED} "
         f"--iterations {iters} --num-envs 192 --task {STUDENT_TRAIN} --experiment {C.STUDENT_EXP} --wait-idle{h}\n")
    return Job(name=name, group=group, kind="distill", gpu_h=gpu_h, on_fail=on_fail, script=s,
               done=lambda: C.student_final(run_name, start_iter, iters) is not None, requires=tuple(requires),
               note=note, meta={"run_name": run_name, "start": start_iter, "iters": iters, "ckpt": str(ckpt)})


def kit_job(name, group, script_py: Path, args: list, env_file: Path, out_json: Path, gpu_h, on_fail="stop",
            strip=(), extra_env: dict | None = None, timeout_s=1800, requires=(), note="") -> Job:
    ex = " ".join(f"{k}={q(v)}" for k, v in sorted((extra_env or {}).items()))
    s = (f"set -u\nmkdir -p {q(Path(out_json).parent)}\ncd {q(C.R / 'parkour')} && env OMNI_KIT_ACCEPT_EULA=yes "
         f"TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH={q(PYP)} {envfile_args(env_file, strip)} {ex} {CAP} "
         f"timeout --signal=TERM {timeout_s} {C.PY} {q(script_py)} --headless "
         + " ".join(q(a) for a in args) + f" --out {q(out_json)}\n")
    return Job(name=name, group=group, kind="kit", gpu_h=gpu_h, on_fail=on_fail, script=s,
               done=lambda: (C.read_json(out_json, {}) or {}).get("complete") is True, requires=tuple(requires),
               note=note, meta={"out": str(out_json)})


def check_job(name, group, on_fail, requires=(), note="") -> Job:
    s = f"set -u\n{C.PY} -I {q(READ)} --check {name}\n"
    return Job(name=name, group=group, kind="check", gpu_h=0.0, on_fail=on_fail, script=s, done=None,
               requires=tuple(requires), note=note, isaac=False)


def placeholder(name, group, gpu_h, why) -> Job:
    return Job(name=name, group=group, kind="placeholder", gpu_h=gpu_h, on_fail="continue", note=why, isaac=False)


# ================================================================================================ C1 inertness
def c1() -> list:
    J = []
    dump_out = C.OUT / "dumps"
    J.append(Job(name="C1_dumps", group="C1", kind="kit", gpu_h=0.25, on_fail="stop", script=(
        f"set -u\n{C.PY} {q(C.PL / 'scripts/t4_dump_cfgs.py')} --out {q(dump_out)} && touch {q(dump_out / '.complete')}\n"),
        done=lambda: (dump_out / ".complete").is_file(), isaac=True,
        note="13 knob-off cfg dumps at the build tree (compared with the M1a reference, STEPC_RULES C1.1)"))
    for st in C.STAGES6:
        J.append(driver_job(f"C1_PC_{st}", "C1", f"PC_{st}_s18", 0.05, 1.0,
                            note=f"20-it {st} stage smoke (P-config; s1: P-early + bitwise vs A2)"))
    J.append(driver_job("C1_PC_s1pay00", "C1", "PC_s1pay00_s18", 0.05, 1.0,
                        note="20-it s1 smoke, payload ARMED 0:0 / ZERO_FRAC 1 (shared tags vs C1_PC_s1)"))
    E = C.SC / "envs"
    sm = C.ctx().spec

    def man(fam):
        return C.OMNI / sm.d["rows"][fam]["manifest"]

    def meta_args(ref):
        m = C.read_json(Path(ref) / "run_meta.json", {})
        a = []
        if m.get("command_adapter") == "body_hold":
            a += ["--command-adapter", "body_hold"]
        if m.get("trained_categories"):
            a += ["--trained-categories", ",".join(m["trained_categories"])]
        return a

    def ref_ck(ref):
        return C.read_json(Path(ref) / "run_meta.json", {}).get("checkpoint")

    rows = [("C1_S1_suite", "T4C1_S1", "REFS_suite_es1", "suite", None),
            ("C1_S1_field_hard", "T4C1_S1", "REF_field_hard_es1", "field_hard", None),
            ("C1_S1_suite_body", "T4C1_S1body", "ANC_suite_body_es1", "suite_body", None),
            ("C1_X43_suite", "T4C1_X43", "X43_suite_es1", "suite", None),
            ("C1_X60_suite", "T4C1_X60", "X60_suite_es1", "suite", None),
            ("C1_X10_suite", "T4C1_X10", "X10_suite_es1", "suite", None),
            ("C1_S1_suite_L000", "T4C1_S1L000", "REFS_suite_es1", "suite", C.ctx().plan["levels"]["0"])]
    for jn, rn, refk, fam, lvl in rows:
        ref = C.REF[refk]
        rj = raw_row(rn, ref_ck(ref), E / f"{rn}.{fam}.env", man(fam), sm.d["rows"][fam]["scenario"], 1,
                     meta_args(ref), level_file=(str(C.ctx().env_file(lvl)) if lvl else None))
        J.append(row_job(jn, "C1", rj, 0.12 if fam != "field_hard" else 0.06, on_fail="stop",
                         note=f"bitwise vs {refk} (the recorded env, exactly)"))
    fwd_ck = C.FWD_3A_CK
    J.append(row_job("C1_fwd3a_old", "C1", raw_row("T4C1_fwd3a_old", fwd_ck, E / "fwd3a.env",
                                                    C.SC / "rows/scenarios_fwd3a_oldshape.yaml", "slow__A15pB", 1,
                                                    ["--task", "Isaac-Crab-Hex-Student-v0"]),
                     0.08, on_fail="stop", note="forward 3a slow row in the RECORDED manifest shape, bitwise"))
    J.append(row_job("C1_fwd3a_cur", "C1", raw_row("T4C1_fwd3a_cur", fwd_ck, E / "fwd3a.env",
                                                    C.FWD / "experiments/eval/scenarios_morph.yaml", "slow__A15pB", 1,
                                                    ["--task", "Isaac-Crab-Hex-Student-v0"]),
                     0.08, on_fail="continue", note="forward 3a slow row, today's shape (report: 0.75 / 25 expected)"))
    J.append(check_job("C1", "C1", "stop", note="STEPC_RULES C1"))
    return J


# ================================================================================================ C2 eval bridge
C2_FAMILIES = ("suite@1", "side22@1", "field_hard@1", "field_light@1", "ring_obst@1", "field_turn@1",
               "field_suite@1", "suite_body@1", "turnlevels_body@1", "grid_body@1")
C2_GPU = {"suite": 0.12, "side22": 0.06, "field_hard": 0.06, "field_light": 0.06, "ring_obst": 0.06,
          "field_turn": 0.06, "field_suite": 0.12, "suite_body": 0.12, "turnlevels_body": 0.12, "grid_body": 0.24,
          "dirsweep_body": 0.2, "walkstop": 0.05}
S1_PAIR_ES2 = ("suite@2", "field_hard@2", "field_light@2", "ring_obst@2", "field_turn@2")


def c2() -> list:
    J = []
    for key in C2_FAMILIES:
        fam = key.split("@")[0]
        J.append(row_job(f"C2_S1_{key}", "C2", prow("p2a", key, C.S1_CK, "T4C2_S1"), C2_GPU[fam],
                         note="pipeline row path (gate_row.sh, R18 composed env) vs the recorded stamp"))
    J.append(row_job("C2_X60_turn_band@1", "C2", prow("c6x", "turn_band@1", C.X60_C6X_CK, "T4C2_X60"), 0.08,
                     note="turn_band has no S1 stamp: bridged on the X60 c6x head (its recorded tb stamp)"))
    sm = C.ctx().spec
    for key in ("suite@1", "walkstop@1"):
        fam, es = key.split("@")
        man, sc = C.OMNI / sm.d["rows"][fam]["manifest"], sm.d["rows"][fam]["scenario"]
        s = (f"set -u\nP2_REANCHOR=1 bash {q(RUN_ARM)} --row-eval T4C2_REFS_S1 {q(C.S1_CK)} {q(C.REF['REFS_suite_es1'])} "
             f"{q(man)} {sc} {es}\n")
        J.append(Job(name=f"C2_REFS_{key}", group="C2", kind="p2row", gpu_h=C2_GPU[fam], on_fail="continue", script=s,
                     done=(lambda sc=sc, es=es: C.p2_stamp("T4C2_REFS_S1", sc, int(es), C.S1_SHA) is not None),
                     note="the REFS(T*) launcher path (run_arm --row-eval, P2_REANCHOR=1) vs REFS_S1_s18"))
    for jn, nm, cmd, rows in (("C2_SH_T17_height", "T4C2_SH_T17", "manifest", ("hwalkstop__omni@1", "hraise__omni@1")),
                              ("C2_SH_T17_zero", "T4C2_SH_T17z", "zero", ("walkstop__omni@1",))):
        s = (f"set -u\nP2_REANCHOR=1 bash {q(SH_DIR / 'sh_eval_rows.sh')} --deadband 0.25 {nm} stand_net {cmd} armed "
             f"{q(C.SH_T17_CK)} 1 " + " ".join(rows) + "\n")
        J.append(Job(name=jn, group="C2", kind="shrows", gpu_h=0.1 * len(rows), on_fail="continue", script=s,
                     done=(lambda nm=nm, rows=rows: all(C.p2_stamp(nm, r.split("@")[0], int(r.split("@")[1]))
                                                         for r in rows)),
                     note="SH_T17 through the SH launcher (stand_net, deadband 0.25); aggregates compared"))
    for key in S1_PAIR_ES2:
        fam = key.split("@")[0]
        J.append(row_job(f"C2_S1_{key}", "C2", prow("p2a", key, C.S1_CK, "T4C2_S1"), C2_GPU[fam],
                         note="S1's es2 0-kg rows: the pairing side of C3 (paired 0-kg check) and C4 (parity)"))
    J.append(check_job("C2", "C2", "continue", note="per-family bitwise; a differing family is re-anchored (no STOP)"))
    return J


# ================================================================================================ C3 payload
def curve_jobs(group: str, prefix: str, ckpt, stage: str, first_note: str, est_levels: int = 4) -> list:
    """The 0..270 kg curve: levels ascending, each level's 4 rows, stopping at the first unstable level (payload_read
    stable-at-L on the a-priori lines). Higher levels appear only once the lower ones are stable."""
    J = []
    for kg in C.LEVELS:
        nm = C.curve_names(prefix, kg)
        rjs = {k: prow(stage, k, ckpt, nm, level=kg) for k in C.LEVEL_ROWS}
        stamps = {k: C.row_stamp(nm, rj["scenario"], rj["es"], ckpt) for k, rj in rjs.items()}
        for k, rj in rjs.items():
            J.append(row_job(f"{nm}.{k}", group, rj, 0.12 if k.startswith("suite") else 0.06,
                             note=f"{first_note} level {kg} kg"))
        if any(v is None for v in stamps.values()):
            rest = [x for x in C.LEVELS if x > kg]
            if rest:
                J.append(placeholder(f"{prefix}_L>{kg:03d}", group, 0.36 * max(0, est_levels - 1 - C.LEVELS.index(kg)),
                                     f"levels {rest} kg, each only if the previous one is stable"))
            break
        if not C.level_read(stamps)["stable"]:
            break
    return J


def c3() -> list:
    J = []
    E = C.SC / "envs"
    R18 = C.ctx().dir / "envs"
    J.append(kit_job("C3_readback_train", "C3", HERE / "t4_stepC_readback_play.py",
                     ["--task", TEACHER_TRAIN, "--num-envs", "16", "--mode", "train", "--rounds", "24"],
                     R18 / "p3a.train.env", C.OUT / "readback/train.json", 0.05,
                     extra_env={"KRABBY_PLANT": C.ctx().spec.d["plant"]},
                     note="training-values payload (0:90, zf .25, xy .10, z .309) on the Teacher env, partial resets"))
    J.append(kit_job("C3_readback_level", "C3", HERE / "t4_stepC_readback_play.py",
                     ["--task", WALK_PLAY, "--num-envs", "16", "--mode", "level", "--level-kg", "90", "--rounds", "8"],
                     E / "p2a_L090.env", C.OUT / "readback/level90.json", 0.05,
                     note="a level row's env (Walk-Play + L090): every env exactly 90 kg at (0, 0, 0.309)"))
    J.append(check_job("C3_readback", "C3", "stop"))
    lv = C.ctx().spec.d["payload"]["level_env"]
    for kg in C.LEVELS:
        nm = f"T4C3_settle_L{kg:03d}"
        extra = " ".join([f"KRABBY_PAYLOAD_KG={kg}:{kg}"] + [f"{k}={v}" for k, v in sorted(lv.items())])
        s = (f"set -u\nT4_OUT={q(C.OUT)} T4_EXTRA_ENV={q(extra)} bash {q(T4ROW)} {nm} {q(C.S1_CK)} "
             f"{q(R18 / 'p2a.env')} - settle_L{kg:03d} 1 --task {WALK_PLAY} --holds 0:10:stand --episodes 32 "
             "--zero-actions\n")
        J.append(Job(name=f"C3_settle_L{kg:03d}", group="C3", kind="adhoc", gpu_h=0.012, on_fail="stop", script=s,
                     done=(lambda nm=nm: C.adhoc_stamp(nm) is not None),
                     note="deck-CoM zero-action settle (A3 protocol, payload event)"))
    J.append(check_job("C3_settle", "C3", "stop", note="tips at <= 90 kg / sum foot force / realised mass"))
    J.append(driver_job("C3_timing", "C3", "PT_p3a_s18", 0.08, 1.5, on_fail="continue",
                        note="100-it payload training smoke from S1 (s/it vs p2a's 1.49)"))
    J += curve_jobs("C3", "T4C3_S1", C.S1_CK, "p2a", "S1 0-270 curve")
    J.append(check_job("C3_S1_curve", "C3", "stop", note="S1 at 0 kg must pass every stable-at-L line; curve reported"))
    J.append(driver_job("C3_pilot", "C3", "PP_p3a_s18", 1.3, 6.0, on_fail="stop",
                        note="3000-it payload pilot from S1 at the D5 training values (save cadence 100: +500k reads)"))
    J.append(check_job("C3_pilot_live", "C3", "stop", requires=("C3_pilot",),
                       note="live rules replayed on the pilot's tfevents (collapse under load)"))
    if C.pilot_run() is None:
        J.append(placeholder("C3_np_reads", "C3", 1.4, "suite es1-2 at 90 kg on the pilot at +500 ... +3000 (12 rows)"))
        J.append(placeholder("C3_gate_rows", "C3", 1.8, "the full p3a gate on the pilot at N_p (11 0-kg rows + curve)"))
        J.append(check_job("C3_np", "C3", "stop", requires=("C3_pilot",)))
        J.append(check_job("C3_gate", "C3", "stop", requires=("C3_pilot",)))
        return J
    for off in C.PILOT_OFFSETS:
        c = C.pilot_ckpt_at(off)
        nm = f"T4C3_pilot_c{c['iter']}_L090"
        for k in ("suite@1", "suite@2"):
            J.append(row_job(f"{nm}.{k}", "C3", prow("p3a", k, c["ckpt"], nm, level=90), 0.12,
                             requires=("C3_pilot",), note=f"N_p read at +{off} (model_{c['iter']})"))
    J.append(check_job("C3_np", "C3", "stop", requires=("C3_pilot",), note="DESIGN 2.2 N_p rule"))
    npr = (C.read_json(C.SC / "checks/C3_np.json", {}) or {}).get("N_p")
    if not isinstance(npr, int):
        J.append(placeholder("C3_gate_rows", "C3", 1.8, "the full p3a gate on the pilot at N_p (after C3_np)"))
        J.append(check_job("C3_gate", "C3", "stop", requires=("C3_np",)))
        return J
    c = C.pilot_ckpt_at(npr)
    nm = f"T4C3_pilot_c{c['iter']}"
    for k in C.ZERO_KG_ROWS:
        fam = k.split("@")[0]
        J.append(row_job(f"{nm}.{k}", "C3", prow("p3a", k, c["ckpt"], nm), C2_GPU.get(fam, 0.06),
                         requires=("C3_np",), note=f"p3a 0-kg row list on the pilot at N_p={npr}"))
    J += curve_jobs("C3", nm, c["ckpt"], "p3a", f"pilot@N_p={npr} curve")
    J.append(check_job("C3_gate", "C3", "stop", requires=("C3_np",),
                       note="floors + stable at 90 + the paired 0-kg check vs S1 (its_past_p2a = N_p)"))
    return J


# ================================================================================================ C4 camera + student
def c4() -> list:
    J = []
    E = C.SC / "envs"
    R18 = C.ctx().dir / "envs"
    refused = C.P().student_refused_keys()
    sel = C.ctx().spec.stage("sel")
    for mode, ef in (("flat", E / "p2a.stripped.env"), ("field", E / "p2a.field_hard.stripped.env")):
        J.append(kit_job(f"C4_camera_{mode}", "C4", HERE / "t4_stepC_camera_play.py",
                         ["--task", STUDENT_PLAY, "--num-envs", "64", "--mode", mode, "--resets", "4"],
                         ef, C.OUT / f"camera/{mode}.json", 0.06,
                         note="per camera: axis error, ground band (flat) / field-block hits (field)"))
    J.append(check_job("C4_camera", "C4", "stop"))
    J.append(kit_job("C4_readback_student", "C4", HERE / "t4_stepC_readback_play.py",
                     ["--task", STUDENT_TRAIN, "--num-envs", "8", "--mode", "train", "--rounds", "12"],
                     E / "p3a.distill.stripped.env", C.OUT / "readback/student.json", 0.06,
                     note="the distill env (Student-v0 + T*'s overlay incl. payload): realised per-env mass / CoM"))
    J.append(check_job("C4_readback", "C4", "stop"))
    for key in ("suite@1", "side22@1", "field_hard@1", "field_hard@2", "field_light@1"):
        fam = key.split("@")[0]
        J.append(row_job(f"C4_parity_{key}", "C4", prow("sel", key, C.S1_CK, "T4C4_S1_SP", env_stage="p2a",
                                                        extra=sel["teacher_args"], strip=refused),
                         C2_GPU[fam] * 1.3, note="S1 --policy-role teacher on Student-Play (vs its Walk-Play C2 row)"))
    J.append(check_job("C4_parity", "C4", "stop"))
    J.append(distill_job("C4_timing", "C4", "T4C4_timing", C.S1_CK, C.S1_ITER, 200, R18 / "p2a.distill.env", 0.7,
                         note="200-it 4-camera distill from S1 (STOP if > 4 h per 1k its)"))
    J.append(check_job("C4_timing_check", "C4", "stop", requires=("C4_timing",)))
    for x in ("a", "b"):
        J.append(distill_job(f"C4_det_{x}", "C4", f"T4C4_det_{x}", C.S1_CK, C.S1_ITER, 50, R18 / "p2a.distill.env",
                             0.2, on_fail="continue", note="2 x 50-it same-seed determinism (report)"))
    J.append(check_job("C4_det", "C4", "continue", requires=("C4_det_a", "C4_det_b")))
    J.append(distill_job("C4_pilot_depth", "C4", "T4C4_pilot_depth", C.S1_CK, C.S1_ITER, 1000,
                         R18 / "p2a.distill.env", 2.8, note="depth pilot, 1000 its from S1"))
    J.append(distill_job("C4_pilot_zero", "C4", "T4C4_pilot_zero", C.S1_CK, C.S1_ITER, 1000, R18 / "p2a.distill.env",
                         2.8, hydra=("agent.depth_encoder.zero_depth=true",),
                         note="zeroed-depth pilot (agent.depth_encoder.zero_depth=true), 1000 its from S1"))
    for pname, rn, keys in (("depth", "T4C4_pilot_depth", ("suite@1", "field_hard@1", "field_hard@2")),
                            ("zero", "T4C4_pilot_zero", ("field_hard@1", "field_hard@2"))):
        ck = C.student_final(rn, C.S1_ITER, 1000)
        if ck is None:
            J.append(placeholder(f"C4_{pname}_rows", "C4", 0.08 * len(keys), f"{keys} on the {pname} pilot's head"))
            continue
        for key in keys:
            fam = key.split("@")[0]
            J.append(row_job(f"C4_{pname}_{key}", "C4", prow("sel", key, ck, f"T4C4_p{pname}", env_stage="p2a",
                                                            extra=sel["student_args"], strip=refused),
                             C2_GPU[fam] * 1.3, requires=(f"C4_pilot_{pname}",), note="Student-Play, student role"))
        if pname == "depth":
            for key, shp, sc in (("suite@1", "scenarios_shape_suite.yaml", "suite__omni_blk"),
                                 ("field_hard@1", "scenarios_shape_field.yaml", "field_hard__omni_noblk")):
                rj = prow("sel", key, ck, "T4C4_pdepth_shape", env_stage="p2a", extra=sel["student_args"],
                          strip=refused)
                J.append(row_job(f"C4_shape_{key}", "C4", rj, 0.12, manifest=C.SC / "rows" / shp, scenario=sc,
                                 requires=("C4_pilot_depth",), note="the other manifest shape (the split, STEPC C4.5)"))
    J.append(check_job("C4_depth", "C4", "stop", requires=("C4_pilot_depth", "C4_pilot_zero")))
    J.append(check_job("C4_live", "C4", "stop", requires=("C4_pilot_depth",)))
    npr = (C.read_json(C.SC / "checks/C3_np.json", {}) or {}).get("N_p")
    c = C.pilot_ckpt_at(npr) if isinstance(npr, int) else None
    if c is None:
        J.append(placeholder("C4_distill_payload", "C4", 0.85, "200-it distill from the payload pilot at N_p + a 90-kg row"))
        J.append(check_job("C4_distill_payload_check", "C4", "stop", requires=("C3_np",)))
        return J
    J.append(distill_job("C4_distill_payload", "C4", "T4C4_dpay", c["ckpt"], c["iter"], 200,
                         R18 / "p3a.distill.env", 0.7, requires=("C3_np",),
                         note=f"distill from the payload pilot model_{c['iter']} under the p3a overlay"))
    ck = C.student_final("T4C4_dpay", c["iter"], 200)
    if ck is None:
        J.append(placeholder("C4_dpay_row", "C4", 0.15, "suite es1 at 90 kg, Student-Play, student role"))
    else:
        J.append(row_job("C4_dpay_suite@1_L090", "C4", prow("sel", "suite@1", ck, "T4C4_dpay_L090", env_stage="p3a",
                                                            level=90, extra=sel["student_args"], strip=refused),
                         0.15, requires=("C4_distill_payload",), note="one 90-kg row (realised payload asserted)"))
    J.append(check_job("C4_distill_payload_check", "C4", "stop", requires=("C4_distill_payload",)))
    return J


# ================================================================================================ C5 sweep anchors
def c5() -> list:
    J = []
    for seed, ck in ((16, C.X10_CK), (17, C.X11_CK), (18, C.S1_CK)):
        for es in (1, 2):
            J.append(row_job(f"C5_s{seed}_dirsweep@{es}", "C5", prow("p2a", f"dirsweep_body@{es}", ck, f"T4C5_s{seed}"),
                             0.2, note="report-only sweep anchor"))
    J.append(check_job("C5", "C5", "continue"))
    return J


# ================================================================================================ C6 stand
def stand_name(base: str, job: str) -> str:
    """The DAgger run name of stand job ``job`` (its launcher refuses a non-empty run dir: a retried job trains
    afresh under ``<base>_r<attempt>``)."""
    att = int(((load_state()["jobs"].get(job) or {}).get("attempt")) or 1)
    return base if att <= 1 else f"{base}_r{att}"


def walker_spec() -> dict | None:
    """SN_WALKER_SPEC for the payload pilot at N_p (TP pipeline.walker_spec_env's shape: S1_OVERLAY = the p2a overlay,
    OVERLAY_CHAIN = a one-stage chain carrying it verbatim, REFS_ENV_STAMP = a 0-kg suite es1 stamp of the walker)."""
    npr = (C.read_json(C.SC / "checks/C3_np.json", {}) or {}).get("N_p")
    if not isinstance(npr, int):
        return None
    c = C.pilot_ckpt_at(npr)
    st = C.row_stamp(f"T4C3_pilot_c{c['iter']}", "suite__omni", 1, c["ckpt"])
    if st is None:
        return None
    spec = {"S1_CKPT": str(c["ckpt"]), "S1_SHA256": C.sha256(c["ckpt"]),
            "S1_AGENT_YAML": str(Path(c["ckpt"]).parent / "params/agent.yaml"), "REFS_ENV_STAMP": str(st),
            "CANDIDATES": str(C.SC / "candidates.yaml"), "OVERLAY_CHAIN": "PC_p2a_s18",
            "S1_OVERLAY": C.ctx().spec.overlay("p2a"), "REFS_NAME": "REFS_T4C6"}
    return {"SN_WALKER_SPEC": json.dumps(spec, sort_keys=True), "SN_WALKER_CK": str(c["ckpt"]),
            "SN_WALKER_ST": str(st), "P2_REANCHOR": "1"}


def stand_job(name, group, which, base_name, args, env: dict, gpu_h, on_fail, requires=(), note="") -> Job:
    rn = stand_name(base_name, name)
    d = SN_DIR if which == "sn" else SH_DIR
    outv = "SN_OUT" if which == "sn" else "SH_OUT"
    root = C.OUT / ("stand_network" if which == "sn" else "stand_height")
    wf = C.SC / "jobs" / f"{name}.walker.json"
    pre = [f"{outv}={q(root)}"]
    lines = ["set -u"]
    if env:
        C.write_json(wf, env)
        lines.append(f"W={q(wf)}")
        for k in env:
            lines.append(f"export {k}=\"$({C.PY} -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "
                         f"\"$W\" {k})\"")
    lines.append(" ".join(pre + ["bash", q(d / f"{which}_dagger.sh"), "--wait-idle", rn, "--"] + [q(a) for a in args]))
    return Job(name=name, group=group, kind="stand", gpu_h=gpu_h, on_fail=on_fail, script="\n".join(lines) + "\n",
               done=lambda: (root / "runs" / rn / "smoke.json").is_file(), requires=tuple(requires), note=note,
               meta={"run_name": rn, "root": str(root)})


def c6() -> list:
    J = []
    w = walker_spec()
    if w is None:
        J.append(placeholder("C6_new_walker", "C6", 0.25, "SN / SH 100-it smokes beside the payload pilot (needs N_p + "
                                                           "its 0-kg suite es1 stamp)"))
        J.append(check_job("C6_new", "C6", "stop", requires=("C3_np",)))
    else:
        emit = C.OUT / "stand/emit"
        wf = C.SC / "jobs" / "C6_emit.walker.json"
        C.write_json(wf, w)
        exp = "".join(f"export {k}=\"$({C.PY} -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "
                      f"{q(wf)} {k})\"\n" for k in w)
        s = (f"set -u\nmkdir -p {q(emit)}\n{exp}"
             f"SN_OUT={q(C.OUT / 'stand_network')} bash {q(SN_DIR / 'sn_dagger.sh')} --dry-run T4C6_emit_sn -- "
             + " ".join(q(a) for a in SN_OPTS) + f" > {q(emit / 'sn.txt')} 2>&1; echo \"sn rc=$?\"\n"
             f"SH_OUT={q(C.OUT / 'stand_height')} bash {q(SH_DIR / 'sh_dagger.sh')} --dry-run T4C6_emit_sh -- "
             + " ".join(q(a) for a in SH_OPTS) + f" > {q(emit / 'sh.txt')} 2>&1; echo \"sh rc=$?\"\n"
             f"touch {q(emit / '.complete')}\n")
        J.append(Job(name="C6_emit", group="C6", kind="cpu", gpu_h=0.0, on_fail="stop", script=s,
                     done=lambda: (emit / ".complete").is_file(), isaac=False,
                     note="Kit-free --dry-run env composition beside the new walker (no KRABBY_PAYLOAD_*)"))
        J.append(stand_job("C6_sn_new", "C6", "sn", "T4C6_SN_new", SN_OPTS, w, 0.1, "stop",
                           note="SN 100-it smoke beside the payload pilot (SN_WALKER_SPEC)"))
        sn_root = C.OUT / "stand_network/runs" / stand_name("T4C6_SN_new", "C6_sn_new")
        sm = C.read_json(sn_root / "smoke.json", {}) or {}
        fin = ((sm.get("run") or {}).get("final_checkpoint"))
        if not fin:
            cks = C.saved_ckpts(sn_root) if sn_root.is_dir() else []
            fin = str(cks[-1][1]) if cks else None
        if fin and Path(fin).is_file():
            J.append(stand_job("C6_sh_new", "C6", "sh", "T4C6_SH_new",
                               SH_OPTS + ["--warm-start", str(fin), "--warm-sha", C.sha256(fin)], w, 0.1, "stop",
                               requires=("C6_sn_new",), note="SH 100-it smoke, warm start = the SN smoke head"))
        else:
            J.append(placeholder("C6_sh_new", "C6", 0.1, "SH smoke beside the new walker (warm start = C6_sn_new)"))
        J.append(check_job("C6_new", "C6", "stop", requires=("C6_emit", "C6_sn_new")))
    for x in ("a", "b"):
        J.append(stand_job(f"C6_det_sn_{x}", "C6", "sn", f"T4C6_SN_det_{x}", SN_OPTS, {}, 0.1, "continue",
                           note="SN same-seed determinism (walker unset = the bundled S1, DS 17)"))
    for x in ("a", "b"):
        J.append(stand_job(f"C6_det_sh_{x}", "C6", "sh", f"T4C6_SH_det_{x}", SH_OPTS, {}, 0.1, "continue",
                           note="SH same-seed determinism (default warm start = the pinned A-mask s17)"))
    J.append(check_job("C6_det", "C6", "continue", note="decides x_rule (bitwise | report_only)"))
    return J


# ================================================================================================ C7 dry pass
def c7() -> list:
    v = C.SC / "c7" / "dry_verdict.json"
    s = f"set -u\n{C.PY} -I {q(HERE / 't4_stepC_dry.py')} --mode \"${{T4C_DRY:-reduced}}\"\n"
    return [Job(name="C7_dry", group="C7", kind="dry", gpu_h=6.0, on_fail="stop", script=s,
                done=lambda: (C.read_json(v, {}) or {}).get("verdict") in ("PASS", "FAIL"),
                note="pipeline.py run --run dry (reduced rows by default; T4C_DRY=official for the official spec, "
                     "~12 GPU-h) with INJECT, auto-continue, SIGTERM-resume"),
            check_job("C7", "C7", "stop", requires=("C7_dry",))]


# ================================================================================================ state + driver
def all_jobs(only=None) -> list:
    out = []
    for g, fn in zip(GROUPS, (c1, c2, c3, c4, c5, c6, c7)):
        if only and g not in only:
            continue
        out += fn()
    return out


def load_state() -> dict:
    return C.read_json(C.STATE, None) or {"schema": 1, "jobs": {}, "accepted": [], "log": []}


def save_state(st: dict) -> None:
    C.write_json(C.STATE, st)


def status_of(job: Job, st: dict) -> str:
    r = st["jobs"].get(job.name) or {}
    if r.get("status") == "accepted":
        return "accepted"
    if job.kind == "placeholder":
        return "placeholder"
    if job.kind == "check":
        return r.get("status") or "pending"
    if job.done is not None and job.done():
        return "done"
    if r.get("status") in ("failed", "stopped"):
        return r["status"]
    return "pending"


def satisfied(name: str, jobs_by: dict, st: dict) -> str:
    """'ok' | 'wait' | 'blocked' for one requirement."""
    j = jobs_by.get(name)
    r = st["jobs"].get(name) or {}
    if r.get("status") == "accepted":
        return "ok"
    if j is None:                    # outside the --only groups (or not generated yet): the state decides
        if r.get("status") == "done":
            return "ok"
        return "blocked" if r.get("status") in ("failed", "stopped") else "wait"
    s = status_of(j, st)
    if s in ("done", "accepted"):
        return "ok"
    if s in ("failed", "stopped"):
        return "blocked"
    return "wait"


def cmd_next(only) -> int:
    st = load_state()
    jobs = all_jobs(only)
    by = {j.name: j for j in jobs}
    for j in jobs:
        s = status_of(j, st)
        if s in ("done", "accepted", "placeholder", "failed"):
            continue
        if s == "stopped":
            print(f"STOP {j.name} unresolved STOP (t4_stepC_queue.sh --retry {j.name} | --accept {j.name} --why ...)")
            return 0
        req = [satisfied(r, by, st) for r in j.requires]
        if "blocked" in req or "wait" in req:
            continue
        JD = C.SC / "jobs"
        JD.mkdir(parents=True, exist_ok=True)
        (JD / f"{j.name}.sh").write_text(f"# {j.name} [{j.group} {j.kind}] {j.note}\n" + j.script)
        print(f"JOB {j.name} {1 if j.isaac else 0} {j.gpu_h:.2f}")
        return 0
    pend = [j.name for j in jobs if status_of(j, st) in ("pending", "placeholder")]
    print(f"END pending/blocked={len(pend)} {' '.join(pend[:8])}")
    return 0


def cmd_mark(name: str, rc: int) -> int:
    st = load_state()
    jobs = {j.name: j for j in all_jobs()}
    j = jobs.get(name)
    if j is None:
        print(f"FAIL {name} unknown job")
        return 0
    rec = st["jobs"].setdefault(name, {})
    rec.update({"rc": int(rc), "at": datetime.datetime.now().isoformat(timespec="seconds")})
    if j.kind == "check":
        if rc in (0, 2):
            rec["status"] = "done"
            rec["verdict"] = "PASS" if rc == 0 else "REPORT"
            out = f"DONE {name} {rec['verdict']}"
        elif j.on_fail == "stop":
            rec["status"] = "stopped"
            out = f"STOP {name} check rc={rc} (STEPC_RULES; see SC/checks/{name}.json)"
        else:
            rec["status"] = "done"
            rec["verdict"] = f"rc={rc}"
            out = f"DONE {name} rc={rc} (report-only)"
    else:
        ok = j.done() if j.done is not None else rc == 0
        if ok:
            rec["status"] = "done"
            out = f"DONE {name}"
        elif j.on_fail == "stop":
            rec["status"] = "stopped"
            out = f"STOP {name} rc={rc}: outputs incomplete (a STOP-C job)"
        else:
            rec["status"] = "failed"
            out = f"FAIL {name} rc={rc}: outputs incomplete (report-only; dependents blocked)"
    st["log"].append({"job": name, "rc": int(rc), "status": rec["status"], "at": rec["at"]})
    save_state(st)
    print(out)
    return 0


def cmd_retry(name: str) -> int:
    st = load_state()
    jobs = all_jobs()
    names = [j.name for j in jobs if (st["jobs"].get(j.name) or {}).get("status") in ("failed", "stopped")] \
        if name == "failed" else [name]
    groups = set()
    for n in names:
        rec = st["jobs"].setdefault(n, {})
        rec["status"] = "retry"
        rec["attempt"] = int(rec.get("attempt") or 1) + (1 if any(j.name == n and j.kind == "stand" for j in jobs) else 0)
        groups |= {j.group for j in jobs if j.name == n}
    for j in jobs:
        if j.kind == "check" and j.group in groups:
            (st["jobs"].get(j.name) or {}).pop("status", None)
    save_state(st)
    print(f"retry: {names} (checks of {sorted(groups)} cleared)")
    return 0


def cmd_accept(name: str, why: str) -> int:
    if not why:
        print("accept needs --why", file=sys.stderr)
        return 2
    st = load_state()
    rec = st["jobs"].setdefault(name, {})
    rec["status"] = "accepted"
    st["accepted"].append({"job": name, "why": why, "at": datetime.datetime.now().isoformat(timespec="seconds")})
    save_state(st)
    print(f"accepted {name}: {why}")
    return 0


def cmd_list(only, as_json: bool) -> int:
    st = load_state()
    jobs = all_jobs(only)
    by = {j.name: j for j in jobs}
    rows, tot = [], {}
    for j in jobs:
        s = status_of(j, st)
        if s == "pending" and any(satisfied(r, by, st) == "blocked" for r in j.requires):
            s = "blocked"
        rows.append({"name": j.name, "group": j.group, "kind": j.kind, "gpu_h": j.gpu_h, "on_fail": j.on_fail,
                     "status": s, "requires": list(j.requires), "note": j.note})
        t = tot.setdefault(j.group, {"gpu_h": 0.0, "remaining_h": 0.0, "n": 0})
        t["gpu_h"] += j.gpu_h
        t["n"] += 1
        if s not in ("done", "accepted"):
            t["remaining_h"] += j.gpu_h
    if as_json:
        print(json.dumps({"jobs": rows, "totals": tot}, indent=1))
        return 0
    for r in rows:
        print(f"{r['group']}  {r['name']:<34s} {r['kind']:<11s} {r['gpu_h']:5.2f} h  {r['on_fail']:<8s} "
              f"{r['status']:<11s} {r['note'][:70]}")
    print("-" * 100)
    g_all = sum(t["gpu_h"] for t in tot.values())
    g_rem = sum(t["remaining_h"] for t in tot.values())
    for g, t in tot.items():
        print(f"{g}: {t['n']:3d} jobs  {t['gpu_h']:6.2f} GPU-h (remaining {t['remaining_h']:.2f})")
    print(f"TOTAL {g_all:.1f} GPU-h (remaining {g_rem:.1f}); C7 at the reduced dry spec (official: +~6 GPU-h)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("list", "next"):
        p = sub.add_parser(n)
        p.add_argument("--only", default=None, help="comma list of groups (C1..C7)")
        if n == "list":
            p.add_argument("--json", action="store_true")
    m = sub.add_parser("mark")
    m.add_argument("name")
    m.add_argument("rc", type=int)
    r = sub.add_parser("retry")
    r.add_argument("name")
    a = sub.add_parser("accept")
    a.add_argument("name")
    a.add_argument("--why", default="")
    o = ap.parse_args(argv)
    only = set(o.only.split(",")) if getattr(o, "only", None) else None
    if o.cmd == "list":
        return cmd_list(only, o.json)
    if o.cmd == "next":
        return cmd_next(only)
    if o.cmd == "mark":
        return cmd_mark(o.name, o.rc)
    if o.cmd == "retry":
        return cmd_retry(o.name)
    return cmd_accept(o.name, o.why)


if __name__ == "__main__":
    sys.exit(main())
