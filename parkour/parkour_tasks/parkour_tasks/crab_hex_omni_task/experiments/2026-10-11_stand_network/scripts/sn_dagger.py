#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 4 -- the DAgger trainer (``--mode dagger``) and the A1 recoverability probe
(``--mode recover_probe``, PLAN Amendment 1 step 6b). Campaign-local: excluded from code_sha / eval_sha (D4).

LAUNCH THROUGH ``sn_dagger.sh`` (it composes the process environment before Python starts, runs this file under a
``systemd-run --scope`` memory cap with setsid-safe logging; stop it with ``systemctl --user stop <unit>``, never
SIGKILL). The orchestrator serializes it against evals (one Isaac process at a time; run_arm.sh's ``drivers_idle``
does NOT see this file name -- map risk T14; sn_dagger.sh refuses while any Isaac process runs).

    sn_dagger.py --mode dagger --arm mask|full --seed S --iterations N --out RUN_DIR [--num-envs 1024]
                 [--smoke [--smoke-iters 100]] [--buffer-gb 4] [--headless --device cuda:0 ...AppLauncher args]
    sn_dagger.py --mode recover_probe --seed S --iterations N --out RUN_DIR [--smoke [--smoke-iters 100]] ...
    sn_dagger.py --emit-env                    # no Kit: the `export` / `unset` lines sn_dagger.sh evals
    sn_dagger.py --read RUN_DIR [--premise LO:HI] [--a1]     # no Kit: the step-7 notify read / the A1 read (JSON)

WHAT A RUN DOES (details in ``sn_dagger_core.py``; constants and readers in ``sn_dagger_spec.py``):
* env per PLAN D2 (``Isaac-Crab-Hex-Omni-Teacher-v0``, preset 2a + the FULL ``P2__S1_s18`` overlay, pool stand share
  0.5 with matching SLOTS, ``KRABBY_RESAMPLE_S=3:12``, ``KRABBY_OMNI_STAND_OBS=0``), wrapped with ``clip_actions`` 1.0
  (the executed action is clipped exactly as in deployment); every cfg value the plan depends on is asserted before
  ``gym.make`` (fail closed); ``params/{env,agent}.yaml`` + the ``KRABBY_*`` environment dumped;
* frozen S1 (``policy/6_p2a_field/model_19994.pt``'s twin) on non-stand rows, the student (DAgger) or R (beta / the
  10 % control slice / A1) on the latched stand rows, routed by ``cmd.stand_mask()`` read right after ``env.step``;
* ``dagger``: R labels on visited states, beta 1 -> 0 over the first third, DART sigma per episode, the capped
  reservoir, branch-balanced + LIFT-onset-oversampled minibatches, ``model_<it>.pt`` every 500 iterations and at the
  end (the FINAL checkpoint is the judged one), ``train_log.jsonl`` per iteration, ``stands.jsonl`` per stand;
* ``recover_probe``: R on every stand after a per-stand spoiler (control 1/2, statue / chatter / stuck lift 1/6 each),
  no student, no optimizer, sigma 0; ``a1.json`` = ``sn_dagger_spec.a1_read`` (the PASS rule fixed before the read);
* ``--smoke``: runs ``--smoke-iters`` iterations of the EXACT config (``--iterations`` is still the schedule horizon,
  so beta is asserted as a function, not reached) and writes ``smoke.json`` with a verdict for every check of PLAN
  step 6's list (DAgger) or the A1-mode list; exit code 6 when any check fails.

Pre-Kit refusals (message on stderr, exit code 2, nothing imported from Isaac): bad arguments, an out dir that is
relative / has whitespace or glob characters / already holds a run, the process ``KRABBY_*`` set not EXACTLY the
composed env, any ``P2R_*`` / ``SN_STAND_*`` variable, S1's sha256 not the pinned one, the REFS env stamp missing.
Exit codes: 0 done, 1 refused after Kit started (cfg / post-preset D5) or crashed, 2 refused before Kit,
4 interrupted (SIGTERM: ``systemctl --user stop`` or sn_dagger.sh's ``timeout``), 6 smoke FAIL.
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


#: torch-free at import (map risk L6): the spec and sn_common only
SPEC = _load(SCRIPTS / "sn_dagger_spec.py", "_sn_dagger_spec")
SN = _load(SCRIPTS / "sn_common.py", "_sn_dagger_entry_common")
CORE_PATH = SCRIPTS / "sn_dagger_core.py"


# ===================================================================================== argv
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--mode", choices=SPEC.MODES, default=None)
    ap.add_argument("--arm", choices=tuple(SPEC.ARMS), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--num-envs", dest="num_envs", type=int, default=SPEC.NUM_ENVS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--smoke-iters", dest="smoke_iters", type=int, default=100)
    ap.add_argument("--buffer-gb", dest="buffer_gb", type=float, default=SPEC.BUFFER_GB)
    ap.add_argument("--emit-env", dest="emit_env", action="store_true", default=False)
    ap.add_argument("--read", default=None)
    ap.add_argument("--premise", default=None)
    ap.add_argument("--a1", action="store_true", default=False)
    return ap


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """``argv`` WITHOUT the program name -> (own options, the AppLauncher argv). Every refusal is pre-Kit."""
    ap = build_parser()
    o, rest = ap.parse_known_args(list(argv))
    stray = [a for a in rest if a.startswith(("--mode", "--arm", "--seed", "--iter", "--num", "--out", "--smoke",
                                              "--buffer", "--emit", "--read", "--premise", "--a1"))]
    if stray:
        raise SystemExit(f"sn_dagger: malformed argument(s) {stray}")
    if o.emit_env or o.read:
        return o, rest
    if o.mode is None or o.seed is None or o.iterations is None or o.out is None:
        raise SystemExit("sn_dagger: --mode, --seed, --iterations and --out are required")
    if o.mode == "dagger" and o.arm is None:
        raise SystemExit("sn_dagger: --mode dagger needs --arm mask|full")
    if o.mode == "recover_probe" and o.arm is not None:
        raise SystemExit("sn_dagger: --mode recover_probe runs no student (no --arm)")
    if o.iterations <= 0 or o.num_envs <= 0 or o.buffer_gb <= 0:
        raise SystemExit("sn_dagger: --iterations / --num-envs / --buffer-gb must be > 0")
    if o.smoke and not 0 < o.smoke_iters <= o.iterations:
        raise SystemExit(f"sn_dagger: --smoke-iters {o.smoke_iters} must be in (0, --iterations]")
    bad = SPEC.path_reasons(o.out)
    if bad:
        raise SystemExit("sn_dagger: --out refused (the checkpoints go to the stand_net launcher's unquoted "
                         "EVAL_EXTRA_ARGS): " + "; ".join(bad))
    return o, rest


def out_dir_reasons(out: Path) -> list[str]:
    """One job, no resume, no overwrite (PLAN step 7): the run dir must be new or empty."""
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
        raise SystemExit(f"sn_dagger: {chain} has {len(stages)} stages, expected 1")
    return {str(k): ("" if v is None else str(v)) for k, v in (stages[0].get("overlay") or {}).items()}


def emit_env(environ) -> str:
    """The shell lines ``sn_dagger.sh`` evals: ``unset`` every KRABBY_* / P2R_* / SN_STAND_* variable, then ``export``
    the composed env (map section 3.1). Refuses when ``S1_OVERLAY`` drifted from candidates.yaml."""
    got = overlay_from_candidates()
    if got != SPEC.S1_OVERLAY:
        diff = sorted(set(got.items()) ^ set(SPEC.S1_OVERLAY.items()))
        raise SystemExit(f"sn_dagger: the P2__S1_s18 overlay in candidates.yaml differs from S1_OVERLAY: {diff[:6]}")
    env = SPEC.train_env(environ)
    lines = [f"unset {k}" for k in sorted(environ) if str(k).startswith(("KRABBY_",) + SPEC.FORBIDDEN_PREFIXES)]
    keys = list(SPEC.PROCESS_ENV) + list(SPEC.expected_krabby_env())
    lines += [f"export {k}={shlex.quote(env[k])}" for k in keys]
    return "\n".join(lines) + "\n"


def read_run(run_dir: str, premise: str | None, a1: bool) -> dict:
    recs = SPEC.read_stands(Path(run_dir) / "stands.jsonl")
    out = {"run": str(run_dir), "n_records": len(recs)}
    if premise is not None:
        lo, _, hi = premise.partition(":")
        out["premise"] = SPEC.premise_read(recs, int(lo or 0), int(hi) if hi else None)
    if a1:
        out["a1"] = SPEC.a1_read(recs)
    if premise is None and not a1:
        out["premise"] = SPEC.premise_read(recs)
    return out


# ===================================================================================== the Kit run
def provenance(o, env_dump_sha: str) -> dict:
    from datetime import datetime, timezone  # noqa: PLC0415

    files = {"sn_dagger": _THIS, "sn_dagger_core": CORE_PATH, "sn_dagger_spec": SCRIPTS / "sn_dagger_spec.py",
             "sn_common": SCRIPTS / "sn_common.py", "restance_module": SN.RESTANCE_MODULE}
    return {"sn_dagger": {
        "task": SPEC.TASK_ID, "mode": o.mode, "arm": o.arm, "stand_mask": SPEC.ARMS.get(o.arm, 0), "seed": o.seed,
        "iterations": o.iterations, "smoke": bool(o.smoke), "num_envs": o.num_envs,
        "s1": str(SPEC.S1_CKPT), "s1_sha256": SPEC.S1_SHA256,
        **{f"{k}_sha256": SN.sha256_file(p) for k, p in files.items()},
        "r_knob": SN.RESTANCE_D3, "forward": SN.FORWARD, "trainable": "actor.actor_backbone",
        "frozen": ["actor.scan_encoder", "actor.history_encoder", "actor.priv_encoder", "critic", "std_raw",
                   "estimator"],
        "beta": "1 -> 0 linearly over the first third of --iterations, per episode", "control_share":
            SPEC.CONTROL_SHARE, "sigma_table": [list(x) for x in SPEC.SIGMA_TABLE], "lr": SPEC.LR,
        "minibatches": SPEC.MINIBATCHES, "minibatch": SPEC.MINIBATCH, "onset_share": SPEC.ONSET_SHARE,
        "w_max": SPEC.W_MAX, "buffer_gb": o.buffer_gb, "steps_per_iter": SPEC.STEPS_PER_ITER,
        "krabby_env_sha256": env_dump_sha, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}}


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

    # the harness's sys.path surgery (H:152-161): parkour/scripts shadows pip rsl_rl
    parkour = SPEC.REPO / "parkour"
    for p in (str(parkour / "scripts"), str(parkour / "scripts" / "rsl_rl"), str(parkour / "parkour_tasks")):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(SPEC.REPO))
    sys.path.insert(0, str(parkour))
    sys.path.insert(0, str(parkour / "parkour_tasks"))
    import parkour_tasks  # noqa: F401,PLC0415  (the omni preset expands KRABBY_OMNI_PHASE here, setdefault)
    from isaaclab.utils.io import dump_yaml  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: PLC0415
    from scripts.rsl_rl.modules.crab_actor_critic_with_encoder import CrabHexActorCriticRMA  # noqa: PLC0415
    from scripts.rsl_rl.modules.feature_extractors.estimator import DefaultEstimator  # noqa: PLC0415
    from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict  # noqa: PLC0415
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: PLC0415

    core = _load(CORE_PATH, "_sn_dagger_core")
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(exist_ok=True)

    # D5 after the preset's expansion: STAND_OBS / mask / proxy / RESTANCE / hold / park + ENV4, and obs parity with
    # the REFS eval stamp on every obs-affecting key
    refs_env = json.loads((SPEC.REFS_ENV_STAMP / "run_meta.json").read_text())["env_vars"]
    post = [f"D5: {r}" for r in SN.env_reasons(os.environ, env4=True)]
    parity = SPEC.obs_parity_reasons(os.environ, refs_env)
    if post or parity:
        raise SystemExit("sn_dagger: environment after the preset expansion refused:\n  " + "\n  ".join(post + parity))
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
        raise SystemExit("sn_dagger: config refused before gym.make:\n  " + "\n  ".join(bad))
    env = ParkourRslRlVecEnvWrapper(gym.make(SPEC.TASK_ID, cfg=env_cfg, render_mode=None),
                                    clip_actions=agent_cfg.clip_actions)
    try:
        if int(env.num_obs) != core.NUM_OBS or int(env.num_actions) != core.NUM_ACT:
            raise SystemExit(f"sn_dagger: env obs / action dims {env.num_obs} / {env.num_actions}, want 1149 / 18")
        if float(env.clip_actions) != 1.0:
            raise SystemExit(f"sn_dagger: the wrapper clips at {env.clip_actions}, want 1.0 (deployment)")
        dump_yaml(str(out / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(out / "params" / "agent.yaml"), agent_cfg)
        nets = core.build_networks(CrabHexActorCriticRMA, DefaultEstimator, train_cfg, SPEC.S1_CKPT, env.device)
        if nets.s1_sha != SPEC.S1_SHA256:
            raise SystemExit("sn_dagger: S1's sha changed under the run")
        opts = core.RunOpts(mode=o.mode, arm=o.arm, seed=o.seed, iterations=o.iterations,
                            stop_after=(o.smoke_iters if o.smoke else o.iterations), num_envs=o.num_envs,
                            buffer_gb=o.buffer_gb, smoke=o.smoke)
        infos = provenance(o, hashlib.sha256(dump.encode()).hexdigest())
        (out / "run.json").write_text(json.dumps({"opts": opts.to_dict(), **infos}, indent=1, sort_keys=True))

        def terrain():
            return env.unwrapped.parkour_manager.get_term("base_parkour").terrain.terrain_levels

        tr = core.Trainer(env, nets, opts, out, train_cfg=train_cfg, policy_cls=CrabHexActorCriticRMA,
                          est_cls=DefaultEstimator, infos=infos, terrain_fn=terrain,
                          log=lambda s: print(s, flush=True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(tr, "stop_requested", True))
        res = tr.run()
        print(f"[SN] run {res['status']}: {json.dumps(res)}", flush=True)
        rc = 0 if res["status"] == "done" else 4
        if o.mode == "recover_probe":
            a1 = SPEC.a1_read(list(tr.tracker.done))
            (out / "a1.json").write_text(json.dumps(a1, indent=1, sort_keys=True))
            print(f"[SN] A1 verdict {a1['verdict']} (checks {[(c['check'], c['pass']) for c in a1['checks']]})",
                  flush=True)
        if o.smoke and res["status"] == "done":
            if o.mode == "dagger":
                env_par = SN.env_reasons(os.environ, env4=True) + SPEC.obs_parity_reasons(os.environ, refs_env)
                checks = core.smoke_checks_dagger(tr, env_parity=env_par, ckpt=res["final_checkpoint"])
                kind = "dagger"
            else:
                checks = core.smoke_checks_a1(tr)
                kind = "a1"
            verdict = SPEC.smoke_verdict(kind, checks)
            (out / "smoke.json").write_text(json.dumps({"verdict": verdict, "checks": checks, "run": res,
                                                         "s_per_iter": res["s_per_iter"]}, indent=1,
                                                        sort_keys=True, default=str))
            print(f"[SN] SMOKE {kind} {verdict['verdict']} failed={verdict['failed']} problems={verdict['problems']}",
                  flush=True)
            if verdict["verdict"] != "PASS":
                rc = 6
        return rc
    finally:
        env.close()


#: the exit code of every refusal before Kit (sn_dagger.sh / the orchestrator key on it: 2 = nothing started)
REFUSED_RC = 2


def main(argv: list[str] | None = None, environ=None) -> int:
    """Returns the exit code. Every pre-Kit refusal (``parse`` / ``pre_kit_reasons`` / ``--emit-env``'s overlay
    check raise ``SystemExit(<message>)``) prints its message to stderr and returns :data:`REFUSED_RC` (2), never the
    interpreter's 1 for a string SystemExit; argparse's own usage errors already exit 2."""
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ
    try:
        o, rest = parse(argv[1:])
        if o.emit_env:
            sys.stdout.write(emit_env(environ))
            return 0
        if o.read:
            print(json.dumps(read_run(o.read, o.premise, o.a1), indent=1, sort_keys=True))
            return 0
        bad = pre_kit_reasons(o, environ)
        if bad:
            raise SystemExit("sn_dagger: refused before Kit:\n  " + "\n  ".join(bad))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr, flush=True)
            return REFUSED_RC
        raise
    return run_kit(o, rest)


if __name__ == "__main__":
    sys.exit(main())
