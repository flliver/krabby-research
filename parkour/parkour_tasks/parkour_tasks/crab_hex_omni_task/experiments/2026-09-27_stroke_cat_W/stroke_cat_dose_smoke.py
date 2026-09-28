#!/usr/bin/env python3
"""The W leg's DOSE SMOKE: a 300-iteration, by-hand ``run_phases.train`` fork of the per-category
reference chain's composed ``c6y_scat``/``c6y`` env (per-category stroke plan, Phase B step 6;
CHARTER.md beside this file). Generalised from ``../2026-09-26_stroke_posture/window0_dose_smoke.py``
and ``../2026-09-26_stroke_formation/formation_dose_smoke.py``, but composing the env from the
CAMPAIGN'S OWN ``references:`` entry rather than a candidate's overlay (plan "Left for
implementation" item 3): this campaign runs no candidate at all, so the doses to smoke live directly
on chain ``c6y_scat``'s one stage, reached through ``run_candidates.reference_recipe(cands, "S_cat",
code)`` + ``run_candidates.stage_env(recipe, "c6y")`` -- never ``compose_recipe``.

WHAT IT TRAINS. W's baked ``policy/5_c6x_clockfree/model_11995.pt`` (the SAME checkpoint ``c6y``
forks; its sha256 is CHECKED AGAINST ``policy/manifest.yaml`` at run time, never hardcoded) forked for
``--iters`` (300) iterations of the reference chain's composed env -- chain ``c6y_scat``, stage
``c6y``: C1d's c6 overlay verbatim, K (the clock-free removal set) held in the chain, the split flag,
the posture knob at eps, and the per-category stroke knobs at their PROVISIONAL smoke-scale (replay-
scale) doses already written into ``candidates.yaml`` (``KRABBY_OMNI_STROKE_W`` /
``KRABBY_OMNI_STROKE_CAT_W``, read at run time so a dose edit before the real launch is picked up) --
plus:

* the README PLAY block's remaining policy-input keys, ``KRABBY_FLAT_TERRAIN_GEOM=ring`` and
  ``KRABBY_OMNI_FACING_OBS_CLIP=0.6`` (W trained with 0.6; the obs term's class default is no clip).
  ``KRABBY_OMNI_CLOCK_HIDE=1`` is NOT added here: it is already in the composed stage env (K, pinned
  by ``require_env``) -- :func:`compose` VERIFIES this rather than setting it, so a require_env
  regression on the chain is caught here instead of silently overwritten.
  NOT the README EVAL block: it sets ``KRABBY_CLOCK_W``/``KRABBY_APEX_W`` 1.0 and ``KRABBY_RSI_FRAC``
  0, which would train the clock back in -- refused below by name;
* ``KRABBY_OMNI_STROKE_PARITY_{DUMP,ENVS,STEPS}`` (``mdp/crab_hex_omni_stroke_parity.py``) IN
  ``extra_env``: ``run_phases._subenv`` strips every ``KRABBY_*`` of the parent shell, so exporting
  them would silently record nothing.

``--print-env`` prints the composed env and exits (CPU only; no Isaac, no GPU) -- and works even
before the checkpoint's sha256 is checked or the worktree guard runs. ``candidates.yaml`` is read AT
RUN TIME (a concurrent agent may still be writing it): a missing file, or a missing ``S_cat``
reference, is refused with a clear message, not a traceback.

AFTER IT: r per category (CHARTER.md's dose-smoke section) with ``../tools/replay_stroke_posture.py
income ... --train-log-dir <the run dir this prints>``, and the parity check with
``../tools/replay_stroke_posture.py parity <dump>``.

Run it from the checkout whose code should train: after landing that is the main checkout (the venv's
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
OMNI = HERE.parents[1]                          # .../crab_hex_omni_task
TOOLS = HERE.parent / "tools"
CANDS = HERE / "candidates.yaml"
MANIFEST = OMNI / "policy" / "manifest.yaml"
POLICY_DIR = OMNI / "policy"
FORK_STAGE_DIR = "5_c6x_clockfree"              # W's baked copy -- the SAME checkpoint c6y forks
REFERENCE = "S_cat"
CHAIN = "c6y_scat"
STAGE = "c6y"
OUT = REPO / "parkour/outputs/omni_plan/stroke_cat/W_dose_smoke"
#: The README play block's remaining policy-input keys; NEVER KRABBY_OMNI_CLOCK_HIDE here (module
#: docstring) -- that one must already be in the composed env via require_env, verified in compose().
PLAY_KEYS = {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
#: What the composed env must carry (candidates.yaml's require_env: K + the split flag), re-asserted.
MUST = {"KRABBY_CLOCK_W": "0.001", "KRABBY_APEX_W": "0.001", "KRABBY_OMNI_CLOCK_HIDE": "1",
        "KRABBY_OMNI_REWARD_SPLIT": "1"}
#: The README eval block's keys that must NOT reach a training env (W's clock-free frame, as in
#: window0_dose_smoke.py -- this campaign's economy is W's, not t3's clock-held one).
EVAL_BLOCK_ONLY = {"KRABBY_RSI_FRAC": "0", "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_TRACK_L1_W": "-1.0",
                   "KRABBY_PLAY_DEBUG_VIS": "0"}
#: The two per-category stroke knobs the smoke doses -- both must already be on the chain's stage
#: overlay (candidates.yaml's c6y_scat, PROVISIONAL replay-scale values; CHARTER.md's dose rule).
DOSE_KEYS = ("KRABBY_OMNI_STROKE_W", "KRABBY_OMNI_STROKE_CAT_W")


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


def _stage_checkpoint(stage_dir: str) -> tuple[Path, str]:
    """``(checkpoint path, sha256)`` of ``stage_dir`` in ``policy/manifest.yaml`` (module docstring:
    the check is against the manifest, never a hardcoded prefix). Refuses a manifest with no such
    stage."""
    import yaml  # the harness's own dependency (run_candidates.load_candidates uses it the same way)

    if not MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST}: not on disk -- cannot find {stage_dir}'s checkpoint / sha256")
    doc = yaml.safe_load(MANIFEST.read_text()) or {}
    for st in doc.get("stages") or []:
        if st.get("dir") == stage_dir:
            file = st.get("file")
            sha = st.get("sha256")
            if not file or not sha:
                raise SystemExit(f"{MANIFEST}: stage {stage_dir!r} is missing file/sha256")
            return POLICY_DIR / stage_dir / str(file), str(sha)
    raise SystemExit(f"{MANIFEST}: no stage with dir {stage_dir!r} (known: "
                     f"{[st.get('dir') for st in doc.get('stages') or []]})")


def compose(rc, parity_dump: Path, parity_envs: int, parity_steps: int) -> dict:
    """The reference chain's composed ``c6y_scat``/``c6y`` env, read from ``candidates.yaml`` AT RUN
    TIME (module docstring): ``reference_recipe`` + ``stage_env``, never ``compose_recipe`` -- this
    campaign runs no candidate for the doses to live on."""
    if not CANDS.is_file():
        raise SystemExit(f"{CANDS}: not written yet -- cannot compose {CHAIN}'s {STAGE} env (this "
                         "campaign's candidates.yaml; it may still be in progress)")
    cands = rc.load_candidates(CANDS)
    rc.validate_candidates(cands)                       # strict: refuses a spec the ladder would refuse
    refs = cands.get("references") or {}
    if REFERENCE not in refs:
        raise SystemExit(f"{CANDS}: reference {REFERENCE!r} not found (known: {sorted(refs)})")
    recipe = rc.reference_recipe(cands, REFERENCE, "stroke_cat_dose_smoke")
    if recipe["chain"] != CHAIN:
        raise SystemExit(f"references.{REFERENCE}.chain is {recipe['chain']!r}, expected {CHAIN!r}")
    env = rc.stage_env(recipe, STAGE)
    # CLOCK_HIDE must already be here (require_env on the chain): verified, never set (module docstring).
    if str(env.get("KRABBY_OMNI_CLOCK_HIDE")) != "1":
        raise SystemExit(f"KRABBY_OMNI_CLOCK_HIDE must already be '1' in {CHAIN}'s {STAGE} stage env "
                         f"(require_env carries it), got {env.get('KRABBY_OMNI_CLOCK_HIDE')!r}")
    for key in DOSE_KEYS:
        if key not in env:
            raise SystemExit(f"{CANDS}: {CHAIN}'s {STAGE} stage overlay has no {key} (the smoke-scale "
                             "per-category dose)")
    for k, v in PLAY_KEYS.items():
        if k in env and str(env[k]) != v:
            raise SystemExit(f"{k}: the composed env has {env[k]!r}, the play block wants {v!r}")
        env[k] = v
    for k, v in MUST.items():
        if str(env.get(k)) != v:
            raise SystemExit(f"{k} must be {v!r} in the smoke env (require_env), got {env.get(k)!r}")
    leaked = sorted(k for k in EVAL_BLOCK_ONLY if k in env)
    if leaked:
        raise SystemExit(f"README eval-block keys in a training env: {leaked}")
    env.update({"KRABBY_OMNI_STROKE_PARITY_DUMP": str(parity_dump),
                "KRABBY_OMNI_STROKE_PARITY_ENVS": str(parity_envs),
                "KRABBY_OMNI_STROKE_PARITY_STEPS": str(parity_steps)})
    for k in rc.rp.TRAIN_ENV_RESERVED:
        assert k not in env, k
    doses = {k: env[k] for k in DOSE_KEYS}
    return {"env": env, "doses": doses, "recipe_id": recipe["recipe_id"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", default="3")
    ap.add_argument("--tag", default="W_dose_smokeS_cat")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--parity-envs", type=int, default=16)
    ap.add_argument("--parity-steps", type=int, default=300)
    ap.add_argument("--print-env", action="store_true", help="print the composed env and exit (CPU only)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    rc = _load(TOOLS / "run_candidates.py", "run_candidates_stroke_cat_dose_smoke")
    comp = compose(rc, out / f"parity_{args.tag}.npz", args.parity_envs, args.parity_steps)
    fork, expected_sha = _stage_checkpoint(FORK_STAGE_DIR)
    print(json.dumps({"fork": str(fork), "fork_sha256_expected": expected_sha, "iters": args.iters,
                      "seed": args.seed, "tag": args.tag, "base_recipe_id": comp["recipe_id"],
                      "doses": comp["doses"], "extra_env": comp["env"]}, indent=1))
    if args.print_env:
        return 0
    _worktree_guard()
    if not fork.is_file():
        raise SystemExit(f"{fork}: not on disk (policy/manifest.yaml names it as {FORK_STAGE_DIR}'s checkpoint)")
    digest = hashlib.sha256(fork.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise SystemExit(f"{fork}: sha256 {digest} != policy/manifest.yaml's {expected_sha} for "
                         f"{FORK_STAGE_DIR}")
    rp = rc.rp
    c = rp.Campaign(out / "campaign_stub" / "2026-09-27_stroke_cat_W", "A15+B", str(args.seed))
    rp._wait_isaac_clear()
    ckpt, status, log = rp.train(c, rp.ph.PHASES["1a"], args.tag, args.iters, str(fork), False,
                                 extra_env=comp["env"], run_name=args.tag)
    run_dir = str(Path(ckpt).parent) if ckpt else None
    res = {"ckpt": ckpt, "status": status, "log": str(log), "run_dir": run_dir,
           "parity_dump": comp["env"]["KRABBY_OMNI_STROKE_PARITY_DUMP"], "doses": comp["doses"]}
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("W DOSE SMOKE DONE", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
