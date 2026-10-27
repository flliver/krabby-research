#!/usr/bin/env python3
"""After one body reference chain (one stage) finished: dump audit + the step-6 stage-end read.

1. Finds the reference's recorded head (``body_cmd_train/state.json`` -> ``ladder.references[chain][-1]``), its run
   dir and the driver's own ``suite__omni_body`` es1 stamp (``gait_eval/omni_phases/body_cmd_train/<tag>/``).
2. DUMP AUDIT on the run's ``params/env.yaml`` (PLAN step 6: "before each stage trains, its realised dump is
   asserted"; the driver audits the overlay itself, this re-checks the body essentials): body command class + the
   frozen envelope, obs encoding none, the three swapped reward funcs at weights 0.2 / 2.0 / 1.0, facing-turn weight
   0, ellipse 0.35:0.35:0.25, no RSI event, turn floor 0.21, pool / slots = the stage overlay.
3. ``--smoke``: stops here (the smoke checks are bc_smoke_check.py). Otherwise the stage-end read
   (bc_stage_read.py) vs the same-stage anchors; after c6 also a ``turnlevels__omni_body`` eval (the low-end check).
Exit: the stage read's code (0 OK / 2 NOTIFY / 3 STOP), 4 = audit failure or missing record.

    bc_after_stage.py --chain BC_<stage>_s<seed> [--smoke]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
C = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
TRAIN = C / "body_cmd_train"
EVAL_ROOT = R / "parkour/logs/rsl_rl/gait_eval/omni_phases" / TRAIN.name
LR = R / "parkour/outputs/omni_plan/body_cmd/lr"
MB = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml"
ENV_FREEZE = C / "envelope_freeze.json"
STAGE_ANCHOR = {"fb6k": "STG_fb6k", "c4": "STG_c4", "c5": "STG_c5", "c6": "STG_c6", "c6x": "STG_c6x",
                "p2a": "ANC_S1"}


def newest(d: Path) -> Path | None:
    hits = sorted(p for p in d.glob("*/") if (p / "run_meta.json").is_file()) if d.is_dir() else []
    return hits[-1] if hits else None


def audit(run_dir: Path, overlay: dict, env: dict) -> list[str]:
    bad = []
    d = yaml.unsafe_load((run_dir / "params/env.yaml").read_text())
    c = d["commands"]["base_velocity"]
    r = d["rewards"]
    if "CrabHexOmniBodyCommand" not in str(c.get("class_type")):
        bad.append(f"command class {c.get('class_type')}")
    want = (env["w_min"], env["ceiling"], env["w_max"])
    got = (c.get("body_w_min"), c.get("body_ceiling"), c.get("body_w_max"))
    if tuple(float(x) for x in got) != tuple(float(x) for x in want):
        bad.append(f"envelope {got} != frozen {want}")
    if [list(map(float, p)) for p in (c.get("body_cap") or [])] != [list(map(float, p)) for p in env["cap"]]:
        bad.append(f"cap {c.get('body_cap')} != frozen {env['cap']}")
    if tuple(float(x) for x in c.get("speed_ellipse")) != (0.35, 0.35, 0.25):
        bad.append(f"ellipse {c.get('speed_ellipse')}")
    pool = overlay.get("KRABBY_OMNI_CMD_POOL")
    if pool and c.get("cmd_pool") != pool:
        bad.append(f"pool {c.get('cmd_pool')} != overlay {pool}")
    for term, fn, w in (("reward_tracking_yaw", "RewardYawWindowExp", 0.2), ("reward_yaw_progress",
                                                                             "RewardYawProgressTent", 2.0),
                        ("track_ang_vel_z_exp", "track_ang_vel_z_exp_straight_gated", 1.0)):
        t = r.get(term) or {}
        if fn not in str(t.get("func")) or abs(float(t.get("weight", -9)) - w) > 1e-12:
            bad.append(f"{term}: func {t.get('func')} weight {t.get('weight')} (want {fn} @ {w})")
    want_rate = "instant" if str(overlay.get("KRABBY_OMNI_BODY_TENT_RATE", "")).strip().lower() == "instant" else None
    got_rate = ((r.get("reward_yaw_progress") or {}).get("params") or {}).get("rate")
    if got_rate != want_rate:
        bad.append(f"reward_yaw_progress rate {got_rate!r} != overlay {want_rate!r}")
    if float((r.get("reward_facing_turn") or {}).get("weight", 0.0)) != 0.0:
        bad.append("reward_facing_turn weight != 0")
    if (d.get("events") or {}).get("rsi_reference_reset") is not None:
        bad.append("rsi_reference_reset event present")
    if abs(float(d.get("omni_clock_turn_floor") or 0.0) - 0.21) > 1e-9:
        bad.append(f"omni_clock_turn_floor {d.get('omni_clock_turn_floor')}")
    enc = [v.get("params", {}).get("facing_obs_encoding") for v in d["observations"]["policy"].values()
           if isinstance(v, dict) and isinstance(v.get("params"), dict) and "facing_obs_encoding" in v["params"]]
    if enc != ["none"]:
        bad.append(f"obs facing encoding {enc}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cands = yaml.safe_load((TRAIN / "candidates.yaml").read_text())
    stage = cands["chains"][a.chain]["stages"][0]
    seed = int(cands["references"][a.chain]["seed"])
    state = json.loads((TRAIN / "state.json").read_text())
    refs = ((state.get("ladder") or {}).get("references") or {}).get(a.chain) or []
    out = {"chain": a.chain, "stage": stage["stage"], "seed": seed}
    if not refs:
        out["error"] = "no recorded reference run"
        print(json.dumps(out))
        return 4
    ref = refs[-1]
    ck = Path(ref["ckpt"]) if Path(ref["ckpt"]).is_absolute() else R / ref["ckpt"]
    run_dir = ck.parent
    tag = run_dir.name.split("_", 2)[-1]   # <date>_<time>_<tag>
    out.update({"xid": ref.get("xid"), "ckpt": str(ck), "run_dir": str(run_dir), "tag": tag})
    env = json.loads(ENV_FREEZE.read_text())["envelope"]
    bad = audit(run_dir, stage["overlay"], env)
    out["audit"] = bad or "PASS"
    reads = C / "stage_reads"
    reads.mkdir(exist_ok=True)
    if bad or a.smoke:
        (reads / f"{a.chain}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=1))
        return 4 if bad else 0
    stamp = newest(EVAL_ROOT / tag / "suite__omni_body" / "seed001")
    if stamp is None:
        out["error"] = f"no driver suite__omni_body stamp under {EVAL_ROOT / tag}"
        (reads / f"{a.chain}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out))
        return 4
    out["head_stamp"] = str(stamp)
    anchors = [newest(LR / f"{STAGE_ANCHOR[stage['stage']]}_s{s}" / "suite__omni_body" / "seed001")
               for s in (16, 17, 18)]
    if any(x is None for x in anchors):
        out["error"] = f"missing stage anchors {anchors}"
        print(json.dumps(out))
        return 4
    cmd = [PY, "-I", str(HERE / "bc_stage_read.py"), "--stage", f"{stage['stage']}_b", "--head", str(stamp),
           "--categories", ",".join(stage["categories"]), "--json", str(reads / f"{a.chain}.read.json")]
    for s in anchors:
        cmd += ["--anchor", str(s)]
    if stage["stage"] == "c6":
        name = f"LOW_{a.chain}"
        rc = subprocess.run([str(HERE / "bc_eval_row.sh"), name, str(ck), str(stamp), str(MB),
                             "turnlevels__omni_body", "1"]).returncode
        low = newest(LR / name / "turnlevels__omni_body" / "seed001")
        if rc != 0 or low is None:
            out["error"] = f"low-end turnlevels eval failed rc={rc}"
            print(json.dumps(out))
            return 4
        cmd += ["--low-end", str(low)]
        out["low_end_stamp"] = str(low)
    rc = subprocess.run(cmd).returncode
    out["stage_read_rc"] = rc
    (reads / f"{a.chain}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
