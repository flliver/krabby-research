#!/usr/bin/env python3
"""Window 0 (ii): the 300-iteration DOSE SMOKE of arm C, by hand through ``run_phases.train``, OUTSIDE
the ladder (stride/posture plan step 12(ii); CHARTER.md "Window 0").

WHAT IT TRAINS. W's baked copy (``policy/5_c6x_clockfree/model_11995.pt``, sha256 3a527683eccd...)
forked for ``--iters`` (300) iterations of arm C's COMPOSED stage env -- composed here by the driver's
own ``compose_recipe`` (the baseline recipe's ``c6y`` env from ``candidates.yaml``: C1d's c6 overlay, K,
both knobs at eps, the split flag) with BOTH arm doses laid over it (S's ``KRABBY_OMNI_STROKE_W`` and
P's ``KRABBY_OMNI_CENTRING_W``, read from the yaml, so a dose edit is picked up) -- plus:

* the README PLAY block's policy-input keys ``KRABBY_OMNI_CLOCK_HIDE=1``, ``KRABBY_FLAT_TERRAIN_GEOM=ring``,
  ``KRABBY_OMNI_FACING_OBS_CLIP=0.6`` (W trained with 0.6; the obs term's class default is no clip).
  NOT the README EVAL block: it sets ``KRABBY_CLOCK_W``/``KRABBY_APEX_W`` 1.0 and ``KRABBY_RSI_FRAC`` 0,
  which would train the clock back in -- refused below by name;
* ``KRABBY_OMNI_STROKE_PARITY_{DUMP,ENVS,STEPS}`` (``mdp/crab_hex_omni_stroke_parity.py``) IN
  ``extra_env``: ``run_phases._subenv`` strips every ``KRABBY_*`` of the parent shell, so exporting
  them would silently record nothing.

The ε-armed B never exercises the doses nor the partial clip absorption the arms run, which is why the
smoke runs C. ``--print-env`` prints the composed env and exits (CPU only; no Isaac, no GPU).

AFTER IT: ``r`` per category (CHARTER.md "Window 0" (ii)) with ``replay_stroke_posture.py income ...
--train-log-dir <the run dir this prints>``, and the parity check with ``replay_stroke_posture.py
parity <dump>``.

Run it from the checkout whose code should train: after SA2 that is the main checkout (the venv's
editable finder resolves ``parkour_tasks`` there). From a git worktree it refuses unless PYTHONPATH
puts that worktree's ``parkour/parkour_tasks`` and ``parkour`` first (``tests/unit/conftest.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
TOOLS = HERE.parent / "tools"
CANDS = HERE / "candidates.yaml"
FORK_REL = "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt"
FORK_SHA256_PREFIX = "3a527683eccd"
STAGE = "c6y"
OUT = REPO / "parkour/outputs/omni_plan/stroke_posture/window0"
#: README play block, policy-input keys only (never the eval block's reward frame / RSI 0).
PLAY_KEYS = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_FLAT_TERRAIN_GEOM": "ring",
             "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
#: What the composed env must carry (K + split) -- the yaml's require_env, re-asserted.
MUST = {"KRABBY_CLOCK_W": "0.001", "KRABBY_APEX_W": "0.001", "KRABBY_OMNI_CLOCK_HIDE": "1",
        "KRABBY_OMNI_REWARD_SPLIT": "1"}
#: The README eval block's keys that must NOT reach a training env.
EVAL_BLOCK_ONLY = {"KRABBY_RSI_FRAC": "0", "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_TRACK_L1_W": "-1.0",
                   "KRABBY_PLAY_DEBUG_VIS": "0"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _worktree_guard() -> None:
    parts = REPO.parts
    in_worktree = any(parts[i] == ".claude" and parts[i + 1] == "worktrees" for i in range(len(parts) - 1))
    if not in_worktree:
        return
    pp = [Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    need = [REPO / "parkour/parkour_tasks", REPO / "parkour"]
    if pp[:2] != [n.resolve() for n in need]:
        raise SystemExit(f"run from a worktree ({REPO}) but PYTHONPATH does not start with {need}: the "
                         "training child would import the MAIN checkout's parkour_tasks/parkour_isaaclab")


def compose(rc, iters: int, parity_dump: Path, parity_envs: int, parity_steps: int) -> dict:
    cands = rc.load_candidates(CANDS)
    rc.validate_candidates(cands)                       # strict: refuses a spec the ladder would refuse
    base = rc.compose_recipe({"entries": []}, cands, "window0", upto=rc.candidate_ids(cands)[0])
    env = rc.stage_env(base, STAGE)
    doses = {}
    for cid, key in (("S", "KRABBY_OMNI_STROKE_W"), ("P", "KRABBY_OMNI_CENTRING_W")):
        doses[key] = str(rc.candidate_by_id(cands, cid)["overlay"][STAGE][key])
    env.update(doses)                                   # arm C = B + both doses
    for k, v in PLAY_KEYS.items():
        if k in env and str(env[k]) != v:
            raise SystemExit(f"{k}: the composed env has {env[k]!r}, the play block {v!r}")
        env[k] = v
    for k, v in MUST.items():
        if str(env.get(k)) != v:
            raise SystemExit(f"{k} must be {v!r} in the smoke env (K / split), got {env.get(k)!r}")
    leaked = sorted(k for k in EVAL_BLOCK_ONLY if k in env)
    if leaked:
        raise SystemExit(f"README eval-block keys in a training env: {leaked}")
    env.update({"KRABBY_OMNI_STROKE_PARITY_DUMP": str(parity_dump),
                "KRABBY_OMNI_STROKE_PARITY_ENVS": str(parity_envs),
                "KRABBY_OMNI_STROKE_PARITY_STEPS": str(parity_steps)})
    for k in rc.rp.TRAIN_ENV_RESERVED:
        assert k not in env, k
    return {"env": env, "doses": doses, "recipe_id": base["recipe_id"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", default="3")
    ap.add_argument("--tag", default="window0_doseC")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--parity-envs", type=int, default=16)
    ap.add_argument("--parity-steps", type=int, default=300)
    ap.add_argument("--print-env", action="store_true", help="print the composed env and exit (CPU only)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    rc = _load(TOOLS / "run_candidates.py", "run_candidates_window0")
    fork = REPO / FORK_REL
    comp = compose(rc, args.iters, out / f"parity_{args.tag}.npz", args.parity_envs, args.parity_steps)
    print(json.dumps({"fork": str(fork), "iters": args.iters, "seed": args.seed, "tag": args.tag,
                      "base_recipe_id": comp["recipe_id"], "doses": comp["doses"], "extra_env": comp["env"]},
                     indent=1))
    if args.print_env:
        return 0
    _worktree_guard()
    digest = hashlib.sha256(fork.read_bytes()).hexdigest()
    if not digest.startswith(FORK_SHA256_PREFIX):
        raise SystemExit(f"{fork}: sha256 {digest[:12]} is not W's baked copy {FORK_SHA256_PREFIX}")
    rp = rc.rp
    c = rp.Campaign(out / "campaign_stub" / "2026-09-26_stroke_posture", "A15+B", str(args.seed))
    rp._wait_isaac_clear()
    ckpt, status, log = rp.train(c, rp.ph.PHASES["1a"], args.tag, args.iters, str(fork), False,
                                 extra_env=comp["env"], run_name=args.tag)
    run_dir = str(Path(ckpt).parent) if ckpt else None
    res = {"ckpt": ckpt, "status": status, "log": str(log), "run_dir": run_dir,
           "parity_dump": comp["env"]["KRABBY_OMNI_STROKE_PARITY_DUMP"], "doses": comp["doses"]}
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("WINDOW0 DOSE SMOKE DONE", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
