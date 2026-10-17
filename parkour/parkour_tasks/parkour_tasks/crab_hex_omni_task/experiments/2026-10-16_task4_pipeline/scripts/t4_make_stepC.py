#!/usr/bin/env python3
"""PLAN step 8: write the STATIC inputs of the Phase-C queue into ``PL/preflight/stepC/`` (CPU; byte-stable).

* ``candidates.yaml`` -- the step-8 driver chains (a campaign dir of its own, so the Phase-A records stay untouched):
    PC_<stage>_s18     the six Phase-A smokes again, VERBATIM (``preflight/candidates.yaml`` PF_* stage dicts): P-config
                       repeated after C1, s1 P-early + bitwise against A2;
    PC_s1pay00_s18     PC_s1 + the payload ARMED at 0 kg (KG 0:0, ZERO_FRAC 1, COM_XY 0, COM_Z 0.309): shared tags
                       against PC_s1 (C1);
    PT_p3a_s18         100 its from S1 (bundled 6_p2a_field/model_19994.pt) at the D5 TRAINING values = the spec's p3a
                       overlay (``Spec.overlay('p3a')``: p2a VERBATIM + KG 0:90, ZERO_FRAC 0.25, COM_XY 0.10, COM_Z 0.309)
                       -- the payload timing (C3);
    PP_p3a_s18         the same, 3000 its -- the payload pilot (C3: N_p, the S1-pilot p3a gate, the live calibration).
  Every chain is one stage, no stage eval, seed 18. PC_p2a_s18 also serves C6 as the one-stage chain that carries the
  p2a overlay verbatim (the stand tools' OVERLAY_CHAIN check).
* ``envs/`` -- the C1 bridge rows' env files (each = a recorded stamp's ``run_meta.env_vars`` EXACTLY), the forward 3a
  canary env, and the stripped / combined variants of the R18 env files the Kit checks use.
* ``rows/`` -- the manifest-shape twins: forward 3a ``slow__A15pB`` in its RECORDED shape (an env block with the
  recorded KRABBY_HEX_USD_PATH); ``suite__omni_blk`` (suite__omni + a no-op block KRABBY_RSI_FRAC '0', the value the
  pre-Kit env already holds) and ``field_hard__omni_noblk`` (field_hard__omni without its block; the block keys are
  in the pre-Kit env file already).
* ``dump_ref/`` -- the M1a knob-off reference dumps (copied from the step-3 scratchpad when present).
* ``INJECT.json`` -- the C7 dry-pass injection template; ``dryspec/`` -- the REDUCED dry spec (C7 default).

    t4_make_stepC.py [--check]          (--check: compare with what is on disk, write nothing; exit 1 on drift)
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t4_stepC_lib as C  # noqa: E402

SEED = 18
M1A_DUMPS = Path("/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/b8e96414-701e-4216-9940-452abfe18030/"
                 "scratchpad/m1a_dumps/after")
PAY00 = {"KRABBY_PAYLOAD_KG": "0:0", "KRABBY_PAYLOAD_ZERO_FRAC": "1", "KRABBY_PAYLOAD_COM_XY": "0",
         "KRABBY_PAYLOAD_COM_Z": "0.309"}
#: C7 injections (pipeline.py Queue.inject kinds, once per (stage, kind)): a walker FAIL, an INVALID (rows rerun once),
#: a GUARD trip, a live kill on the distill and on a DAgger stage. The student-branch HALT is NATURAL (a 50-it student
#: has one candidate and fails its block); the walker gates HALT naturally too (20-it heads).
INJECT = {"c4": ["FAIL"], "c5": ["INVALID"], "c6": ["GUARD"], "st": ["LIVE"], "sn": ["LIVE"]}
#: the reduced dry spec's row lists (C7 default; STEPC_RULES C7): every stage KIND and every row KIND (suite, lr,
#: field, body, level, Student-Play, stand) is exercised once; the stand rows are kept whole (their readers need them)
DRY_ROWS = {
    "s1": ["suite@1:driver", "side22@1"], "c4": ["suite@1:driver"], "c5": ["suite@1:driver"],
    "c6": ["suite@1:driver"], "c6x": ["suite@1:driver"],
    "p2a": ["suite@1:driver", "side22@1", "field_hard@1", "suite_body@1"],
}


def pf_chains() -> dict:
    return yaml.safe_load((C.PL / "preflight/candidates.yaml").read_text())


def candidates() -> str:
    pf = pf_chains()
    ch, refs = {}, {}

    def add(cid, src, stage, note):
        top = {k: v for k, v in pf["chains"][src].items() if k not in ("stages", "notes")}
        ch[cid] = {"notes": note, **copy.deepcopy(top), "stages": [stage]}
        refs[cid] = {"chain": cid, "seed": SEED}

    for st in C.STAGES6:
        s = copy.deepcopy(pf["chains"][f"PF_{st}_s18"]["stages"][0])
        add(f"PC_{st}_s18", f"PF_{st}_s18", s, f"Task 4 step-8 C1 smoke: PF_{st}_s18 verbatim (20 its).")
    s = copy.deepcopy(pf["chains"]["PF_s1_s18"]["stages"][0])
    s["overlay"] = {**s["overlay"], **PAY00}
    add("PC_s1pay00_s18", "PF_s1_s18", s, "Task 4 step-8 C1: PF_s1 + the payload armed at 0 kg (KG 0:0, ZERO_FRAC 1).")
    spec = C.ctx().spec
    for cid, iters, note in (("PT_p3a_s18", 100, "C3 payload timing (100 its)"),
                             ("PP_p3a_s18", C.PILOT_ITERS, "C3 payload pilot (3000 its; checkpoints every 100)")):
        s = copy.deepcopy(pf["chains"]["PF_p2a_s18"]["stages"][0])
        s["stage"] = "p3a"
        s["iters"] = iters
        s["overlay"] = spec.overlay("p3a")
        s["categories"] = list(spec.stage("p3a")["categories"])
        s["from_scratch"] = False
        s["resume_from"] = {"checkpoint": str(C.S1_BUNDLED.relative_to(C.R))}
        s["eval"] = []
        add(cid, "PF_p2a_s18", s, f"Task 4 step-8 {note}: S1 (bundled 6_p2a_field, sha {C.S1_SHA[:12]}) at the D5 "
                                  "training values (the spec's p3a overlay).")
    doc = {**{k: copy.deepcopy(pf[k]) for k in ("schema", "plant", "promotion", "decision_rule")},
           "seeds": {"screen": SEED, "confirm": 16},
           "screen": {"chain": "PC_s1_s18", "eval": [], "iters": 20},
           "baseline": {"id": "PC_s1_s18", "chain": "PC_s1_s18"},
           "chains": ch, "references": refs, "candidates": [], "milestones": []}
    doc["decision_rule"]["text"] = ("TASK 4 STEP 8 (2026-10-16_task4_pipeline/preflight/stepC): RECORD ONLY. Phase-C "
                                    "smokes and the payload pilot; every read is scripts/t4_stepC_read.py.")
    return "# GENERATED by scripts/t4_make_stepC.py -- do not edit by hand.\n" + yaml.safe_dump(doc, sort_keys=False,
                                                                                               width=120)


def env_files() -> dict:
    out = {}
    stamp_rows = [("T4C1_S1", "suite", "REFS_suite_es1"), ("T4C1_S1", "field_hard", "REF_field_hard_es1"),
                  ("T4C1_S1body", "suite_body", "ANC_suite_body_es1"), ("T4C1_X43", "suite", "X43_suite_es1"),
                  ("T4C1_X60", "suite", "X60_suite_es1"), ("T4C1_X10", "suite", "X10_suite_es1"),
                  ("T4C1_S1L000", "suite", "REFS_suite_es1")]
    for rn, fam, refk in stamp_rows:
        env = C.stamp_env(C.REF[refk])
        if not env:
            raise SystemExit(f"{refk}: {C.REF[refk]} has no env_vars")
        out[f"envs/{rn}.{fam}.env"] = C.env_text(env)
    out["envs/fwd3a.env"] = C.env_text(C.FWD_3A_ENV)
    r18 = C.ctx().dir / "envs"
    refused = set(C.P().student_refused_keys())
    TL = C.TL()
    for src, dst in (("p2a.env", "p2a.stripped.env"), ("p2a.field_hard.env", "p2a.field_hard.stripped.env"),
                     ("p3a.distill.env", "p3a.distill.stripped.env")):
        env = {k: v for k, v in TL.read_env_file(r18 / src).items() if k not in refused}
        out[f"envs/{dst}"] = C.env_text(env)
    lvl = TL.read_env_file(C.ctx().env_file(C.ctx().plan["levels"]["90"]))
    out["envs/p2a_L090.env"] = C.env_text({**TL.read_env_file(r18 / "p2a.env"), **lvl})
    return out


def shape_manifests() -> dict:
    out = {}
    morph = yaml.safe_load((C.FWD / "experiments/eval/scenarios_morph.yaml").read_text())
    row = copy.deepcopy(next(s for s in morph["scenarios"] if s["id"] == "slow__A15pB"))
    rec = C.read_json(C.REF["FWD3A_slow_es1"] / "run_meta.json", {})
    row["env"] = {"KRABBY_HEX_USD_PATH": rec["env_vars"]["KRABBY_HEX_USD_PATH"]}
    row["notes"] = (str(row.get("notes", "")) + " | Task 4 step 8 C1: the RECORDED (pre-2026-09-09) manifest shape -- "
                    "an env block with the recorded KRABBY_HEX_USD_PATH (run_meta of the 2026-09-08 3a stamp).")
    top = {k: v for k, v in morph.items() if k != "scenarios"}
    out["rows/scenarios_fwd3a_oldshape.yaml"] = ("# GENERATED by scripts/t4_make_stepC.py (step 8 C1)\n"
                                                 + yaml.safe_dump({**top, "scenarios": [row]}, sort_keys=False))
    for src, sid, new_id, fn, kind in (("eval/scenarios_omni.yaml", "suite__omni", "suite__omni_blk",
                                        "rows/scenarios_shape_suite.yaml", "blk"),
                                       ("2026-10-07_omni_phase2/scenarios_p2.yaml", "field_hard__omni",
                                        "field_hard__omni_noblk", "rows/scenarios_shape_field.yaml", "noblk")):
        m = yaml.safe_load((C.EXP / src).read_text())
        row = copy.deepcopy(next(s for s in m["scenarios"] if s["id"] == sid))
        row["id"] = new_id
        if kind == "blk":
            row["env"] = {"KRABBY_RSI_FRAC": "0"}
            row["notes"] = "Task 4 step 8 C4: suite__omni + a NO-OP env block (the pre-Kit value) -- the other shape."
        else:
            row.pop("env", None)
            row["notes"] = ("Task 4 step 8 C4: field_hard__omni WITHOUT its env block (its keys come pre-Kit from the "
                            "row's env file) -- the other shape.")
        top = {k: v for k, v in m.items() if k != "scenarios"}
        out[fn] = "# GENERATED by scripts/t4_make_stepC.py (step 8 C4)\n" + yaml.safe_dump({**top, "scenarios": [row]},
                                                                                          sort_keys=False)
    return out


def dryspec() -> dict:
    d = yaml.safe_load(C.SPEC.read_text())
    d["notes"] = ("REDUCED dry spec (GENERATED by scripts/t4_make_stepC.py, PLAN step 8 C7): the official spec with the "
                  "dry run only and shorter gate row lists -- the stage graph, the overlays, the stand rows and every "
                  "row KIND are the official ones. Never an official run.")
    d["runs"] = {"dry": copy.deepcopy(d["runs"]["dry"])}
    d["thresholds"] = dict(d["thresholds"])
    d["lock"] = "pipeline.lock.json"
    for st in d["stages"]:
        if st["id"] in DRY_ROWS:
            st["gate"]["rows"] = list(DRY_ROWS[st["id"]])
        if st["id"] == "sel":
            st["prescreen_rows"] = ["suite@1", "field_hard@1"]
            st["prescreen_level"] = {"kg": 90, "rows": ["suite@1"]}
            st["full_block_rows"] = ["suite@1", "side22@1", "field_hard@1"]
            st["full_block_level_rows"] = ["suite@1"]
            st["completion_margin"] = {**st["completion_margin"], "manifest_shape_split": 0.05}
    d["row_lists"]["zero_kg"] = ["suite@1:driver", "side22@1", "field_hard@1"]
    d["payload"]["level_rows"] = ["suite@1"]
    d["live"] = {**d["live"], "poll_s": 30}
    d["row_out"] = "parkour/outputs/omni_plan/task4/stepC/dryspec_rows"
    out = {"dryspec/pipeline.yaml": "# GENERATED by scripts/t4_make_stepC.py -- the REDUCED C7 dry spec.\n"
                                    + yaml.safe_dump(d, sort_keys=False, width=120)}
    for n in ("thresholds_v0.json", "threshold_inputs.yaml", "cameras.yaml"):
        out[f"dryspec/{n}"] = (C.PL / n).read_text()
    return out


def files() -> dict:
    out = {"candidates.yaml": candidates(), "INJECT.json": json.dumps(INJECT, indent=1, sort_keys=True) + "\n"}
    out.update(env_files())
    out.update(shape_manifests())
    out.update(dryspec())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fs = files()
    drift = []
    for rel, text in sorted(fs.items()):
        p = C.SC / rel
        if a.check:
            if not p.is_file() or p.read_text() != text:
                drift.append(rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.is_file() or p.read_text() != text:
            p.write_text(text)
    ref = C.SC / "dump_ref"
    if not a.check and M1A_DUMPS.is_dir() and not ref.is_dir():
        ref.mkdir(parents=True)
        for f in sorted(M1A_DUMPS.glob("*.json")):
            shutil.copy2(f, ref / f.name)
        (ref / "SOURCE.txt").write_text(f"copied from {M1A_DUMPS} (M1a knob-off 'after' dumps, byte-identical to "
                                        "'before' at 5f0b45b8 per m1a/dump_check.json); dump root "
                                        f"{M1A_DUMPS.parents[1] / 'wt_m1a'}\n")
    if a.check:
        print("\n".join(drift) if drift else "t4_make_stepC --check: byte-identical")
        return 1 if drift else 0
    print(f"{C.SC}: {len(fs)} files; dump_ref {'present' if ref.is_dir() else 'MISSING (C1_dumps compare is report-only)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
