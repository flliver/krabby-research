#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``), PLAN H4 / H5 / step 5 -- the height DAgger trainer's entry point.
Copied from the closed predecessor's ``sn_dagger.py`` (sha256 f33bb051...; never edited) and adapted. Campaign-local:
outside code_sha / eval_sha.

LAUNCH THROUGH ``sh_dagger.sh`` (process env composed before Python starts, a ``systemd-run --scope`` memory cap, a
``timeout --signal=TERM`` wall-clock bound; stop with ``systemctl --user stop <unit>``, never SIGKILL; one Isaac process
at a time -- it refuses while any runs).

    sh_dagger.py --mode dagger --arm mask --seed S --iterations N --out RUN_DIR [--num-envs 1024]
                 [--smoke [--smoke-iters 100]] [--buffer-gb 4] [--deadband D] [--trim-branch [--trim-min X]]
                 [--warm-start PATH --warm-sha SHA]
                 [--headless --device cuda:0 ...AppLauncher args]
    sh_dagger.py --emit-env                    # no Kit: the `export` / `unset` lines sh_dagger.sh evals
    sh_dagger.py --read RUN_DIR [--premise LO:HI]   # no Kit: the step-6 premise read (JSON)

WHAT A RUN DOES (``sh_dagger_core.py``; constants in ``sh_dagger_spec.py``; the env composition in ``sh_starts.py``;
every height semantic in ``sh_core.py``):

* the process env is the A-mask s17 trainer's, EXACTLY (``Isaac-Crab-Hex-Omni-Teacher-v0``, preset 2a + the full
  ``P2__S1_s18`` overlay, pool stand share 0.5 with matching SLOTS, ``KRABBY_RESAMPLE_S=3:12``, STAND_OBS 0); every cfg
  value the plan rests on is asserted before ``gym.make`` (the predecessor's list + the raw +-4.8 rod clip, the 0.02-s
  step, the wrapped terms present);
* WALKER OVERRIDE (M18 Task 4, 2026-10-16; unset = the above, exactly): ``SN_WALKER_SPEC`` in the process env
  replaces S1 s18 / the ``P2__S1_s18`` overlay by another walker (the predecessor's ``sn_dagger_spec.py`` reads it;
  ``sh_dagger_spec.py`` NOTE). The Task 4 pipeline sets it for the stand stages beside T*
  (``experiments/tools/pipeline/pipeline.py`` ``walker_spec_env``, together with the eval rows' ``SN_WALKER_CK`` /
  ``SN_WALKER_ST`` and ``P2_REANCHOR=1``): that is how the stand network of record SH_T4F19 (``policy/manifest.yaml``
  ``stand_network`` = f19_stand_net) was trained, with ``--warm-start`` / ``--warm-sha`` = SN_T4F19's model_4000;
* between ``parse_env_cfg`` and ``gym.make`` the campaign-local composition (``sh_starts.compose_env_cfg``): one tall
  kinematic table cuboid per env, the start event (reset, LAST), the push / hip-contact termination wrappers, the
  command subclass (h_cmd per stand draw, forced first stands); its private generators are this run's ``starts`` /
  ``hcmd`` streams (created before ``gym.make``, so the construction reset already draws from them);
* the wrapper is built with ``clip_actions = sh_core.HARNESS_CLIP`` (4.8, runtime; the agent config keeps its 1.0,
  asserted) -- the router clamps S1's rows to +-1 explicitly;
* S1 frozen on non-stand rows; the student (warm start: A-mask s17 unless ``--warm-start``, W[:, 15] zeroed) or R_h (beta, the 10 % control
  slice) on the effective stand rows (slot | restoring), executing q(h_lim) + clamp(residual); R_h's residual labels;
  checkpoints every 500 iterations and at the end (the FINAL one is judged); ``train_log.jsonl`` / ``stands.jsonl`` /
  ``episodes.jsonl``;
* ``--smoke``: ``--smoke-iters`` iterations of the EXACT config; ``smoke.json`` with a verdict for every check of PLAN
  step 5 (``sh_dagger_spec.SMOKE_DAGGER``); exit code 6 when any fails.
* ``--deadband D`` (default 0 = off: the run is bitwise the one without the option; CHARTER 2026-10-14 "train it in"):
  the residual deadband (``sh_core.apply_deadband``) on the STUDENT-executed stand action only -- the labels stay R_h's
  residuals and the R_h control / beta-R executors are untouched; recorded in run.json (``opts.deadband``,
  ``sh_dagger.residual_deadband``, ``sh_dagger.clip``), status.json and the checkpoint infos; evaluate such a head with
  the launcher's ``--sh-deadband D`` (forward parity at D is a smoke check).
* ``--trim-branch`` (default off: the run is bitwise the one without the option; CHARTER 2026-10-14 "retrain the pilot
  with a trim weight + the deadband in the loop"): a stand-row leg whose R_h branch is HOLD and whose R_h cam label is
  non-zero (``|label| > 1e-6``, ``sh_dagger_spec.TRIM_LABEL_MIN``; ``--trim-min X`` overrides it, only together with
  ``--trim-branch``) is a 6th category ("trim") of the per-leg inverse-branch-frequency weights (cap W_MAX unchanged);
  nothing else changes. Recorded in run.json (``opts.trim_branch`` / ``opts.trim_min``, ``sh_dagger.trim_branch`` /
  ``sh_dagger.trim_min``), status.json and the checkpoint infos; smoke check ``trim_branch_weights``.
* ``--read`` also reports the run's trim branch and the trim DIAGNOSTIC (every run with the diagnostic, on or off; inside
  ``premise.trim_diag``): on student-executed sigma-0 stand rows, legs R_h HOLDs with the cam in TRIM_BAND and the foot
  loaded, the median executed cam command vs R_h's label (toward home; ratio), pooled over the premise window and per
  500-iteration block.

Exit codes: 0 done, 1 refused after Kit started or crashed, 2 refused before Kit, 4 interrupted (SIGTERM), 6 smoke FAIL.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
SCRIPTS = _THIS.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: torch-free at import (rule L6): the spec, sh_core and the env composition module
SPEC = _load(SCRIPTS / "sh_dagger_spec.py", "_sh_dagger_spec")
SH = SPEC.SH
SN = SH.SN
STARTS_PATH = SCRIPTS / "sh_starts.py"
CORE_PATH = SCRIPTS / "sh_dagger_core.py"


# ===================================================================================== argv
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--mode", choices=SPEC.MODES, default=None)
    ap.add_argument("--arm", choices=tuple(SPEC.ARMS), default="mask")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--num-envs", dest="num_envs", type=int, default=SPEC.NUM_ENVS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--smoke-iters", dest="smoke_iters", type=int, default=100)
    ap.add_argument("--buffer-gb", dest="buffer_gb", type=float, default=SPEC.BUFFER_GB)
    ap.add_argument("--deadband", type=float, default=0.0)
    ap.add_argument("--trim-branch", dest="trim_branch", action="store_true", default=False)
    ap.add_argument("--trim-min", dest="trim_min", type=float, default=None)
    # NOTE(M18 Task 4 pipeline step 6, 2026-10-16; DESIGN B6): the warm start, default the pinned A-mask s17
    # (sh_core.STAND_NET_CKPT / STAND_NET_SHA256: unset = today's run exactly); both flags together or neither.
    # That default is HISTORICAL: A-mask s17 was the stand network of record 2026-10-13 .. 10-15 only and is now
    # policy/manifest.yaml `superseded:` SN_Amask_s17 (sha 769cf253, file kept); the net of record is the manifest's
    # `stand_network` (f19_stand_net = SH_T4F19, itself warm-started here from SN_T4F19 by these two flags).
    ap.add_argument("--warm-start", dest="warm_start", default=None)
    ap.add_argument("--warm-sha", dest="warm_sha", default=None)
    ap.add_argument("--emit-env", dest="emit_env", action="store_true", default=False)
    ap.add_argument("--read", default=None)
    ap.add_argument("--premise", default=None)
    return ap


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """``argv`` WITHOUT the program name -> (own options, the AppLauncher argv). Every refusal is pre-Kit."""
    ap = build_parser()
    o, rest = ap.parse_known_args(list(argv))
    stray = [a for a in rest if a.startswith(("--mode", "--arm", "--seed", "--iter", "--num", "--out", "--smoke",
                                              "--buffer", "--emit", "--read", "--premise", "--deadband",
                                              "--trim", "--warm"))]
    if stray:
        raise SystemExit(f"sh_dagger: malformed argument(s) {stray}")
    if o.emit_env or o.read:
        return o, rest
    if o.mode is None or o.seed is None or o.iterations is None or o.out is None:
        raise SystemExit("sh_dagger: --mode, --seed, --iterations and --out are required")
    if o.iterations <= 0 or o.num_envs <= 0 or o.buffer_gb <= 0:
        raise SystemExit("sh_dagger: --iterations / --num-envs / --buffer-gb must be > 0")
    if o.smoke and not 0 < o.smoke_iters <= o.iterations:
        raise SystemExit(f"sh_dagger: --smoke-iters {o.smoke_iters} must be in (0, --iterations]")
    try:
        o.deadband = SH.check_deadband(o.deadband)
    except ValueError as exc:
        raise SystemExit(f"sh_dagger: --deadband: {exc}") from exc
    if o.trim_min is not None:
        if not o.trim_branch:
            raise SystemExit("sh_dagger: --trim-min needs --trim-branch (it would not be applied)")
        try:
            o.trim_min = SPEC.check_trim_min(o.trim_min)
        except ValueError as exc:
            raise SystemExit(f"sh_dagger: --trim-min: {exc}") from exc
    bad = SPEC.path_reasons(o.out)
    if bad:
        raise SystemExit("sh_dagger: --out refused (the checkpoints go to an unquoted launcher argument): "
                         + "; ".join(bad))
    o.warm_start, o.warm_sha = warm_start_args(o.warm_start, o.warm_sha)
    return o, rest


def warm_start_args(path, sha) -> tuple:
    """``--warm-start`` / ``--warm-sha`` -> (Path, sha) or (None, None) for the pinned default. Both or neither; the
    path absolute and shell-safe (it is recorded and re-read); the sha 64 lower-case hex. Pre-Kit refusals."""
    if path is None and sha is None:
        return None, None
    if path is None or sha is None:
        raise SystemExit("sh_dagger: --warm-start and --warm-sha go together (the sha pins the file the run warms "
                         "from)")
    bad = SPEC.path_reasons(path)
    if bad:
        raise SystemExit("sh_dagger: --warm-start refused: " + "; ".join(bad))
    s = str(sha).strip().lower()
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise SystemExit(f"sh_dagger: --warm-sha must be 64 hex characters, got {sha!r}")
    return Path(path), s


def warm_start_of(o) -> tuple:
    """(path, sha) the run warms from: the CLI pair, else the pinned A-mask s17 (today's default)."""
    if getattr(o, "warm_start", None) is not None:
        return Path(o.warm_start), o.warm_sha
    return SH.STAND_NET_CKPT, SH.STAND_NET_SHA256


def out_dir_reasons(out: Path) -> list[str]:
    """One job, no resume, no overwrite: the run dir must be new or empty."""
    if out.exists() and not out.is_dir():
        return [f"{out} exists and is not a directory"]
    if out.is_dir() and any(out.iterdir()):
        return [f"{out} already holds files ({sorted(p.name for p in out.iterdir())[:5]}); a run never resumes "
                "or overwrites"]
    return []


def pre_kit_reasons(o: argparse.Namespace, environ) -> list[str]:
    out = []
    out += [f"env: {r}" for r in SPEC.env_reasons(environ)]
    out += [f"env (D5): {r}" for r in SN.env_reasons(environ, env4=False)]
    out += out_dir_reasons(Path(o.out))
    if not SPEC.S1_CKPT.is_file():
        out.append(f"S1 checkpoint missing: {SPEC.S1_CKPT}")
    elif SN.sha256_file(SPEC.S1_CKPT) != SPEC.S1_SHA256:
        out.append(f"S1 checkpoint sha256 is not the pinned {SPEC.S1_SHA256[:12]}")
    warm, warm_sha = warm_start_of(o)
    if not warm.is_file():
        out.append(f"warm start missing: {warm}")
    elif SN.sha256_file(warm) != warm_sha:
        out.append(f"warm start sha256 is not the pinned A-mask s17 {warm_sha[:12]}"
                   if getattr(o, "warm_start", None) is None else
                   f"warm start sha256 is not the --warm-sha {warm_sha[:12]} ({warm})")
    if not (SPEC.REFS_ENV_STAMP / "run_meta.json").is_file():
        out.append(f"REFS env stamp missing: {SPEC.REFS_ENV_STAMP}")
    if not SPEC.S1_AGENT_YAML.is_file():
        out.append(f"S1 agent.yaml missing: {SPEC.S1_AGENT_YAML}")
    return out


# ===================================================================================== no-Kit modes
def overlay_from_candidates(path=SPEC.CANDIDATES, chain=SPEC.OVERLAY_CHAIN) -> dict:
    import yaml  # noqa: PLC0415  (a separate, Kit-free process)

    d = yaml.safe_load(Path(path).read_text())
    stages = d["chains"][chain]["stages"]
    if len(stages) != 1:
        raise SystemExit(f"sh_dagger: {chain} has {len(stages)} stages, expected 1")
    return {str(k): ("" if v is None else str(v)) for k, v in (stages[0].get("overlay") or {}).items()}


def emit_env(environ) -> str:
    """The shell lines ``sh_dagger.sh`` evals: ``unset`` every KRABBY_* / P2R_* / SN_STAND_* variable, then ``export``
    the composed env (the predecessor's, unchanged). Refuses when ``S1_OVERLAY`` drifted from candidates.yaml."""
    got = overlay_from_candidates()
    if got != SPEC.S1_OVERLAY:
        diff = sorted(set(got.items()) ^ set(SPEC.S1_OVERLAY.items()))
        raise SystemExit(f"sh_dagger: the P2__S1_s18 overlay in candidates.yaml differs from S1_OVERLAY: {diff[:6]}")
    env = SPEC.train_env(environ)
    lines = [f"unset {k}" for k in sorted(environ) if str(k).startswith(("KRABBY_",) + SPEC.FORBIDDEN_PREFIXES)]
    keys = list(SPEC.PROCESS_ENV) + list(SPEC.expected_krabby_env())
    lines += [f"export {k}={shlex.quote(env[k])}" for k in keys]
    return "\n".join(lines) + "\n"


def run_deadband(run_dir) -> float:
    """The residual deadband a run recorded (run.json ``sh_dagger`` block + its clip record, ``opts.deadband``); a
    run.json without it (every run before the option: SH_SMOKE, SH_P18) or no run.json -> 0.0; disagreement ->
    ValueError."""
    p = Path(run_dir) / "run.json"
    if not p.is_file():
        return 0.0
    run = json.loads(p.read_text())
    d = SH.recorded_deadband(run.get("sh_dagger"))
    opt = (run.get("opts") or {}).get("deadband")
    if opt is not None and SH.check_deadband(opt) != d:
        raise ValueError(f"{p}: opts.deadband {opt!r} != sh_dagger.{SH.RESIDUAL_DEADBAND_KEY} {d!r}")
    return d


def run_trim_branch(run_dir) -> dict:
    """The trim branch a run recorded (run.json ``opts`` + ``sh_dagger``); a run.json without it (every run before the
    option: SH_SMOKE, SH_P18 ...) or no run.json -> off; disagreement -> ValueError."""
    p = Path(run_dir) / "run.json"
    if not p.is_file():
        return {"trim_branch": False, "trim_min": None}
    run = json.loads(p.read_text())
    opt, sh = run.get("opts") or {}, run.get("sh_dagger") or {}
    got = [(bool(d.get("trim_branch", False)), d.get("trim_min")) for d in (opt, sh)]
    if got[0] != got[1]:
        raise ValueError(f"{p}: opts trim_branch / trim_min {got[0]} != sh_dagger's {got[1]}")
    return {"trim_branch": got[0][0], "trim_min": got[0][1]}


def read_run(run_dir: str, premise: str | None) -> dict:
    d = Path(run_dir)
    recs = SPEC.read_stands(d / "stands.jsonl") if (d / "stands.jsonl").is_file() else []
    log = SPEC.read_jsonl(d / "train_log.jsonl")
    eps = SPEC.read_jsonl(d / "episodes.jsonl")
    lo, hi = 0, None
    if premise is not None:
        a, _, b = premise.partition(":")
        lo, hi = int(a or 0), (int(b) if b else None)
    sel = [e for e in eps if int(e["it"]) >= lo and (hi is None or int(e["it"]) < hi)]
    by_kind = {}
    for k in SPEC.START_KINDS:
        g = [e for e in sel if e["kind"] == k and not e["run_start"]]
        by_kind[k] = {"episodes": len(g), "falls": sum(1 for e in g if e["end"] == "fall"),
                      "hip_ends": sum(1 for e in g if e["cause"] == "hip")}
    return {"run": str(run_dir), "n_records": len(recs), "n_episodes": len(eps), SH.RESIDUAL_DEADBAND_KEY:
            run_deadband(run_dir), **run_trim_branch(run_dir), "premise": SPEC.premise_read(recs, log, lo, hi),
            "episodes_by_kind": by_kind}


# ===================================================================================== the Kit run
def provenance(o, env_dump_sha: str) -> dict:
    from datetime import datetime, timezone  # noqa: PLC0415

    db = SH.check_deadband(getattr(o, "deadband", 0.0))
    trim = bool(getattr(o, "trim_branch", False))
    trim_min = (SPEC.check_trim_min(SPEC.TRIM_LABEL_MIN if getattr(o, "trim_min", None) is None else o.trim_min)
                if trim else None)
    files = {"sh_dagger": _THIS, "sh_dagger_core": CORE_PATH, "sh_dagger_spec": SCRIPTS / "sh_dagger_spec.py",
             "sh_starts": STARTS_PATH, "sh_core": SCRIPTS / "sh_core.py", "sh_kin": SCRIPTS / "sh_kin.py",
             "sn_common": SH.SN_COMMON_PATH, "sn_dagger_spec_pred": SPEC.PRED_SCRIPTS / "sn_dagger_spec.py",
             "sn_table_core": SPEC.PRED_SCRIPTS / "sn_table_core.py", "restance_module": SN.RESTANCE_MODULE,
             "facts_json": SH.FACTS_JSON}
    return {"sn_dagger": {"stand_mask": int(SPEC.STAND_MASK), "s1_sha256": SPEC.S1_SHA256, "arm": o.arm,
                          "seed": o.seed, "iterations": o.iterations, "r_knob": SN.RESTANCE_D3},
            "sh_dagger": {
                "task": SPEC.TASK_ID, "mode": o.mode, "arm": o.arm, "stand_mask": int(SPEC.STAND_MASK),
                "seed": o.seed, "iterations": o.iterations, "smoke": bool(o.smoke), "num_envs": o.num_envs,
                SH.RESIDUAL_DEADBAND_KEY: db, "trim_branch": trim, "trim_min": trim_min,
                "s1": str(SPEC.S1_CKPT), "s1_sha256": SPEC.S1_SHA256,
                **{f"{k}_sha256": SN.sha256_file(p) for k, p in files.items()},
                "r_knob": SN.RESTANCE_D3, "forward": "act_inference(sh_core.stand_net_input(obs, est, mask=1, "
                "h_cmd_limited=h_lim, restoring), hist_encoding=True)", "trainable": "actor.actor_backbone",
                "frozen": ["actor.scan_encoder", "actor.history_encoder", "actor.priv_encoder", "critic", "std_raw",
                           "estimator"],
                "beta": "1 -> 0 linearly over the first third of --iterations, per episode",
                "control_share": SPEC.CONTROL_SHARE, "sigma_table": [list(x) for x in SPEC.SIGMA_TABLE],
                "lr": SPEC.LR, "minibatches": SPEC.MINIBATCHES, "minibatch": SPEC.MINIBATCH,
                "onset_share": SPEC.ONSET_SHARE, "w_max": SPEC.W_MAX, "buffer_gb": o.buffer_gb,
                "steps_per_iter": SPEC.STEPS_PER_ITER, "krabby_env_sha256": env_dump_sha,
                "start_shares": dict(zip(SPEC.START_KINDS, SPEC.START_SHARES)),
                "pose_shares": dict(zip(SPEC.POSES, SPEC.POSE_SHARES)),
                "tops_m": [SPEC.TOP_MIN_M, SPEC.TOP_MAX_M], "table_cmd_margin_m": SPEC.TABLE_CMD_MARGIN_M,
                "h_cmd_atom": [SPEC.H_CMD_ATOM, SPEC.H_CMD_ATOM_P],
                "first_stand_s": {"settle": SPEC.FIRST_STAND_SETTLE_S, "hold": SPEC.FIRST_STAND_HOLD_S,
                                  "margin": SPEC.FIRST_STAND_MARGIN_S},
                "up_rise_m": SPEC.UP_RISE_M, "table_park_clear_m": SPEC.TABLE_PARK_CLEAR_M,
                "core": SH.provenance(db),
                # the CLI warm start (Task 4) is recorded only when given: an unset run's run.json is today's
                **({"warm_start_cli": {"path": str(o.warm_start), "sha256": o.warm_sha}}
                   if getattr(o, "warm_start", None) is not None else {}),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}}


def run_kit(o: argparse.Namespace, rest: list[str]) -> int:
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    ap = argparse.ArgumentParser(allow_abbrev=False)
    AppLauncher.add_app_launcher_args(ap)
    app_args = ap.parse_args(rest)
    simulation_app = AppLauncher(app_args).app
    try:
        return _run(o, app_args)
    finally:
        simulation_app.close()


def _run(o: argparse.Namespace, app_args) -> int:
    import hashlib  # noqa: PLC0415

    import gymnasium as gym  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    parkour = SPEC.REPO / "parkour"
    for p in (str(parkour / "scripts"), str(parkour / "scripts" / "rsl_rl"), str(parkour / "parkour_tasks")):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(SPEC.REPO))
    sys.path.insert(0, str(parkour))
    sys.path.insert(0, str(parkour / "parkour_tasks"))
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    import parkour_tasks  # noqa: F401,PLC0415  (the omni preset expands KRABBY_OMNI_PHASE here, setdefault)
    from isaaclab.assets import RigidObjectCfg  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg  # noqa: PLC0415
    from isaaclab.sensors.ray_caster import RayCaster  # noqa: PLC0415  (the harness's import, H:169)
    from isaaclab.utils.io import dump_yaml  # noqa: PLC0415
    from isaaclab.utils.warp import raycast_mesh  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: PLC0415
    from scripts.rsl_rl.modules.crab_actor_critic_with_encoder import CrabHexActorCriticRMA  # noqa: PLC0415
    from scripts.rsl_rl.modules.feature_extractors.estimator import DefaultEstimator  # noqa: PLC0415
    from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict  # noqa: PLC0415
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: PLC0415

    core = _load(CORE_PATH, "_sh_dagger_core")
    starts = core.STARTS
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(exist_ok=True)

    refs_env = json.loads((SPEC.REFS_ENV_STAMP / "run_meta.json").read_text())["env_vars"]
    post = [f"D5: {r}" for r in SN.env_reasons(os.environ, env4=True)]
    parity = SPEC.obs_parity_reasons(os.environ, refs_env)
    if post or parity:
        raise SystemExit("sh_dagger: environment after the preset expansion refused:\n  " + "\n  ".join(post + parity))
    krabby = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    dump = json.dumps(krabby, indent=1, sort_keys=True)
    (out / "params" / "krabby_env.json").write_text(dump)

    device = app_args.device
    env_cfg = parse_env_cfg(SPEC.TASK_ID, device=device, num_envs=o.num_envs)
    env_cfg.seed = o.seed
    agent_cfg = load_cfg_from_registry(SPEC.TASK_ID, "rsl_rl_cfg_entry_point")
    train_cfg = agent_cfg_to_train_dict(agent_cfg)
    bad = SPEC.cfg_reasons(env_cfg, agent_cfg)
    s1_agent = yaml.safe_load(SPEC.S1_AGENT_YAML.read_text())
    bad += SPEC.network_cfg_reasons(train_cfg, s1_agent)
    if bad:
        raise SystemExit("sh_dagger: config refused before gym.make:\n  " + "\n  ".join(bad))
    # the campaign-local composition (PLAN H5), its private streams created BEFORE gym.make (the construction reset
    # already runs the start event and the command subclass)
    gens = core.make_generators(o.seed, device)
    mesh_key = env_cfg.scene.height_scanner.mesh_prim_paths[0]
    state = starts.StartState(o.num_envs, device, gen_start=gens["starts"], gen_hcmd=gens["hcmd"],
                              mesh_fn=lambda: RayCaster.meshes.get(mesh_key), raycast_fn=raycast_mesh)
    changes = starts.compose_env_cfg(env_cfg, state, sim_utils=sim_utils, RigidObjectCfg=RigidObjectCfg,
                                     EventTermCfg=EventTermCfg)
    bad = starts.composition_reasons(env_cfg)
    if bad:
        raise SystemExit("sh_dagger: env composition refused before gym.make:\n  " + "\n  ".join(bad))
    (out / "params" / "composition.json").write_text(json.dumps(changes, indent=1, default=str))
    for c in changes:
        print(f"[SH] change {c['what']}: {c['old']} -> {c['new']} ({c['why']})", flush=True)
    env = ParkourRslRlVecEnvWrapper(gym.make(SPEC.TASK_ID, cfg=env_cfg, render_mode=None),
                                    clip_actions=SH.HARNESS_CLIP)
    try:
        if int(env.num_obs) != core.NUM_OBS or int(env.num_actions) != core.NUM_ACT:
            raise SystemExit(f"sh_dagger: env obs / action dims {env.num_obs} / {env.num_actions}, want 1149 / 18")
        # the copied clip assert, updated (PLAN H1): the stand-net harness clip is 4.8 at runtime, the config's 1.0
        if float(env.clip_actions) != SH.HARNESS_CLIP or float(agent_cfg.clip_actions) != 1.0:
            raise SystemExit(f"sh_dagger: wrapper clip {env.clip_actions} / config clip {agent_cfg.clip_actions}, "
                             f"want {SH.HARNESS_CLIP} / 1.0")
        dump_yaml(str(out / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(out / "params" / "agent.yaml"), agent_cfg)
        mesh = RayCaster.meshes.get(mesh_key)
        probe = starts.KitProbe(env, mesh=mesh, raycast_fn=raycast_mesh)
        warm, warm_sha = warm_start_of(o)
        nets = core.build_networks(CrabHexActorCriticRMA, DefaultEstimator, train_cfg, SPEC.S1_CKPT, env.device,
                                   warm_path=warm, warm_sha=warm_sha)
        if nets.s1_sha != SPEC.S1_SHA256:
            raise SystemExit("sh_dagger: S1's sha changed under the run")
        opts = core.RunOpts(mode=o.mode, arm=o.arm, seed=o.seed, iterations=o.iterations,
                            stop_after=(o.smoke_iters if o.smoke else o.iterations), num_envs=o.num_envs,
                            buffer_gb=o.buffer_gb, smoke=o.smoke, deadband=o.deadband,
                            trim_branch=o.trim_branch, trim_min=o.trim_min)
        infos = provenance(o, hashlib.sha256(dump.encode()).hexdigest())
        infos["sh_dagger"]["clip"] = {**SH.clip_record(o.deadband), "env_clip_actions": float(env.clip_actions),
                                      "config_clip_actions": float(agent_cfg.clip_actions)}
        (out / "run.json").write_text(json.dumps({"opts": opts.to_dict(), **infos}, indent=1, sort_keys=True,
                                                 default=str))

        def terrain():
            return env.unwrapped.parkour_manager.get_term("base_parkour").terrain.terrain_levels

        tr = core.Trainer(env, nets, opts, out, state=state, probe=probe, gens=gens, train_cfg=train_cfg,
                          policy_cls=CrabHexActorCriticRMA, est_cls=DefaultEstimator, infos=infos, terrain_fn=terrain,
                          config_clip=float(agent_cfg.clip_actions), log=lambda s: print(s, flush=True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(tr, "stop_requested", True))
        res = tr.run()
        print(f"[SH] run {res['status']}: {json.dumps(res)}", flush=True)
        rc = 0 if res["status"] == "done" else 4
        (out / "env_counts.json").write_text(json.dumps(state.counts(), indent=1, sort_keys=True))
        if o.smoke and res["status"] == "done":
            env_par = SN.env_reasons(os.environ, env4=True) + SPEC.obs_parity_reasons(os.environ, refs_env)
            checks = core.smoke_checks(tr, env_parity=env_par, ckpt=res["final_checkpoint"],
                                       env_clip=float(env.clip_actions))
            verdict = SPEC.smoke_verdict(checks)
            (out / "smoke.json").write_text(json.dumps({"verdict": verdict, "checks": checks, "run": res,
                                                         "s_per_iter": res["s_per_iter"]}, indent=1,
                                                        sort_keys=True, default=str))
            print(f"[SH] SMOKE {verdict['verdict']} failed={verdict['failed']} problems={verdict['problems']}",
                  flush=True)
            if verdict["verdict"] != "PASS":
                rc = 6
        return rc
    finally:
        env.close()


#: the exit code of every refusal before Kit (2 = nothing started)
REFUSED_RC = 2


def main(argv: list[str] | None = None, environ=None) -> int:
    """Returns the exit code; every pre-Kit refusal prints its message to stderr and returns 2."""
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ
    try:
        o, rest = parse(argv[1:])
        if o.emit_env:
            sys.stdout.write(emit_env(environ))
            return 0
        if o.read:
            print(json.dumps(read_run(o.read, o.premise), indent=1, sort_keys=True))
            return 0
        bad = pre_kit_reasons(o, environ)
        if bad:
            raise SystemExit("sh_dagger: refused before Kit:\n  " + "\n  ".join(bad))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr, flush=True)
            return REFUSED_RC
        raise
    return run_kit(o, rest)


if __name__ == "__main__":
    sys.exit(main())
