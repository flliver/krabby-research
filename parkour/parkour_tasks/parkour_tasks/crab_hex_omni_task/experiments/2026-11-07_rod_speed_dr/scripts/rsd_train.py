#!/usr/bin/env python3
"""Rod-speed DR campaign: launch and supervise ONE teacher fine-tune (CHARTER 2, 3.2, 3.5; plan step 6).

    /home/nickmagus/krabby/isaac_venv/bin/python -I -B rsd_train.py --arm dr|ctl --name N --checkpoint ABS \\
        --env-file ABS [--dr lo:hi[:ramp_steps]] [--iterations 3000] [--num-envs 256] [--seed 19] \\
        [--timeout-s 14400] [--expect-counter C] [--wait-idle] [--dry-run | --preflight-only]

Arms (CHARTER 3.2): ``dr`` = TDR (``--dr`` defaults to the fine-tune setting ``0.6:1.35:12000``; the extension passes
``0.6:1.35``), ``ctl`` = TCTL (no knob; ``--dr`` refused). Resume T* ``policy/f19_8_p3b_payload180/model_25992.pt``
(after EXTEND: TDR's ``model_28991``), ``--env-file`` = ``F19/envs/p3b.distill.env``.

The environment: the user-service environment (``systemctl --user show-environment``) minus every ``KRABBY_*``,
``TP_*``, ``DRS_*``, ``RSD_*`` and ``PYTHONPATH``; plus ``OMNI_KIT_ACCEPT_EULA=yes``, ``TERM=xterm``,
``PYTHONUNBUFFERED=1`` and ``PYTHONPATH=<W>/parkour/parkour_tasks:<W>/parkour`` (without it ``scripts/`` and
``parkour_isaaclab`` resolve to the MAIN checkout through the editable .pth); plus every ``KRABBY_`` line of the env
file VERBATIM (empty values kept) and, on the dr arm, ``KRABBY_DR_ROD_SPEED``. Refused in the env file:
``KRABBY_STRESS_*`` (CHARTER 2), ``KRABBY_PHASE`` / ``KRABBY_SIDE_PHASE`` (run_phases ``_subenv``), and
``KRABBY_DR_ROD_SPEED`` (it comes from ``--dr`` on the dr arm only).

The command is ``run_phases.train``'s (``run_phases.py:1511-1516`` order), with cwd ``<W>/parkour`` and the trainer
spawned WITHOUT ``-I`` (it would drop PYTHONPATH):
    PY train.py --task Isaac-Crab-Hex-Omni-Teacher-v0 --headless --num_envs 256 --seed 19 --max_iterations 3000
       --resume --checkpoint CK --run_name N
wrapped in ``systemd-run --user --scope -q --unit=rsd-train-N-<epoch> -p MemoryMax=48G -p MemorySwapMax=0
timeout --signal=TERM <timeout-s>``. Before the launch: distill.sh's busy pattern (one Isaac process at a time;
``--wait-idle`` waits, else exit 3) and the CHARTER 3.5 guard (< 30 GB free or DerivedDataCache > 78 GB: exit 4).

Stops (watchdogs, a failed in-run pre-flight, SIGTERM / SIGINT / SIGHUP to this launcher, ``touch
results/runs/N.stop``): SIGTERM to the trainer python's own pid, 120 s, SIGTERM again, 120 s, SIGKILL only as the
last resort (recorded as ``sigkill``). Watchdogs: ``Mean reward|value_function|surrogate: nan`` or ``Traceback`` in
the log tail, no log progress for 900 s, the wall bound overrun. A crashed run is NEVER resumed (CHARTER 2: the env
step counter, and so the ramp, would restart) -- relaunch under a new name.

Pre-flight (pre-launch ones refuse; in-run ones stop the run with status ``preflight_fail``):
  PF1  code + eval paths clean; code_sha = results/launch.json; eval_sha = 9fe964d7... (and the pin's)
  PF2  module origins (python -P with the composed env and cwd, train.py's sys.path surgery) all under the worktree
  PF3  checkpoint sha, CPU torch.load iter / alg_state counter + lr (taken from the file and recorded; T* cross-check);
       ctl resumes T* only; dr on T* ramps; dr on any other checkpoint (EXTEND) needs --expect-counter and no ramp
  PF4  env-file sha; composed KRABBY_* = F19 orchestrator.log:30 p3b dict minus PYTHONUNBUFFERED (+ the DR key)
  PF5  argv = the run_phases template, absolute checkpoint, cwd
  PF6  the knob parse (dr_knobs.py by path): lo, hi, ramp_steps = ramp_steps/24 iterations (ctl: no knob armed)
  PF7  the predicted first draw (rod_speed_dr.py by path; seed rod_speed_dr_seed(seed), envs 0..N-1, step 0)
  PF8  no run of that name in the worktree's or main's parkour/logs (nor its log / record); guard; not busy
  PF9  in-run: 'Loading model checkpoint from: CK' + 'Restored algorithm state: counter <PF3>, learning_rate <PF3>'
  PF10 in-run: the [KNOB] / [omni-knobs] multiset = p3b's train log (+ exactly the one DR [KNOB] line on dr)
  PF11 in-run: the [DR] line = PF7's (dr; it names the worktree's module), none on ctl
  PF12 in-run: params/env.yaml + agent.yaml = p3b's after root normalisation, except the rod_speed_dr block (dr) and
       run_name / load_checkpoint
  PF13 in-run: the first block is 'Learning iteration <iter>/<iter+iterations>'; DR/ lines in range (dr), none (ctl)
  PF14 after the run: model_<iter+iterations-1>.pt exists (iter on CPU); the DR/rod_speed_lo trace follows the ramp,
       every DR/ min inside [lo_eff(block end), hi], max <= hi, late DR/rod_speed_mean ~ (lo+hi)/2 (dr); no DR/ line
       (ctl)

Writes (not in --dry-run / --preflight-only): the log ``<MAIN>/parkour/outputs/omni_plan/rod_speed_dr/train/N.log``
(the trainer's stdout+stderr plus sentinel lines ``RSD TRAIN START|PF|END|FAIL N ...``, mirrored to
``train/rsd_train_events.log``), the record ``results/runs/N.json`` (pre-flight, argv, composed env, code_sha,
eval_sha, HEAD, file hashes, the outcome), the composed env ``envs/N.train.env``. The run dir stays in the worktree's
``parkour/logs/rsl_rl/crab_hex_omni_teacher/<ts>_N`` (no copy).

Exit codes: 0 ok (or a dry run); 2 usage / refusal before any check; 3 busy (no --wait-idle) or the launcher lock;
4 guard; 5 a pre-launch pre-flight failed; 6 an in-run pre-flight failed (stopped); 7 the trainer failed (nan,
traceback, stall, timeout, crash, no final checkpoint) or was interrupted; 8 PF14 failed.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rsd_launch as RL  # noqa: E402

TASK = "Isaac-Crab-Hex-Omni-Teacher-v0"
EXPERIMENT = "crab_hex_omni_teacher"
DEFAULT_DR = "0.6:1.35:12000"
MEMORY_MAX = "48G"
PYTHONPATH = f"{RL.PARKOUR / 'parkour_tasks'}:{RL.PARKOUR}"
ADDED_ENV = {"OMNI_KIT_ACCEPT_EULA": "yes", "TERM": "xterm", "PYTHONUNBUFFERED": "1", "PYTHONPATH": PYTHONPATH}
BASE_DROP_PREFIXES = ("KRABBY_", "TP_", "DRS_", "RSD_")
BASE_DROP_KEYS = ("PYTHONPATH",)
REFUSED_PREFIXES = ("KRABBY_STRESS_",)
REFUSED_KEYS = ("KRABBY_PHASE", "KRABBY_SIDE_PHASE")
STALL_S = 900
POLL_S = 15.0
IDLE_POLL_S = 30.0
TAIL_CHARS = 20000
HEAD_MAX_BYTES = 8 << 20
NAN_RE = re.compile(r"Mean (reward|value_function|surrogate):\s*nan", re.I)
#: PF13 / PF14 tolerances: the runner prints 4 decimals
PRINT_TOL = 2e-4
#: PF14: the late window (iterations from the end) and the |late mean - (lo+hi)/2| bound (256 envs: the draw's SE is
#: 0.75/sqrt(12)/16 = 0.0135 per snapshot; the late window averages >= 1000 autocorrelated snapshots)
LATE_ITERS = 1000
LATE_MEAN_TOL = 0.03
#: PF14: the late DR/rod_speed_min (max) must come within this of lo (hi): 256 uniforms
EDGE_TOL = 0.05
#: PF13 / PF14: how many iterations a logged DR/rod_speed_lo may lag the ramp. The env rewrites extras["log"] only in
#: _reset_idx, so a step without a reset re-reports the last reset's values (an OLDER, higher lo_eff); with ~0.15
#: resets per step over 256 envs a lag of 50 iterations (1200 steps) never happens by chance, while a knob that does
#: not ramp, or ramps the wrong way, still fails (the low edge is never relaxed).
LAG_ITERS = 50
DR_KEYS = ("DR/rod_speed_lo", "DR/rod_speed_mean", "DR/rod_speed_min", "DR/rod_speed_max")
#: the module origins PF2 checks (train.py's imports and the knob path)
PROBE_MODULES = (
    "parkour_tasks",
    "parkour_tasks.crab_hex_forward_task.mdp.parkour_actions",
    "parkour_tasks.crab_hex_forward_task.mdp.rod_speed_dr",
    "parkour_tasks.crab_hex_omni_task.config.crab_hex.omni_knobs",
    "parkour_tasks.crab_hex_omni_task.config.crab_hex.dr_knobs",
    "scripts.rsl_rl.runner_factory",
    "scripts.rsl_rl.modules.on_policy_runner_with_extractor",
    "parkour_isaaclab",
    "parkour_isaaclab.envs",
)
#: run_phases.train's command, as text (PF5 checks the order is still the one this launcher reproduces)
RUN_PHASES_CMD_TEXT = (
    'cmd = [PY, str(TRAIN), "--task", spec.task, "--headless", "--num_envs", num_envs,',
    '"--seed", c.seed, "--max_iterations", str(iters)]',
    'cmd += ["--resume", "--checkpoint", resume_ckpt]',
    'cmd += ["--run_name", str(run_name)]',
)
EXIT_OK, EXIT_USAGE, EXIT_BUSY, EXIT_GUARD, EXIT_PF, EXIT_PF_INRUN, EXIT_TRAIN, EXIT_PF14 = 0, 2, 3, 4, 5, 6, 7, 8

STOP_SIGNALS: list[str] = []


class Usage(Exception):
    pass


# ============================================================================== CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=("dr", "ctl"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--dr", default=None, help=f"lo:hi[:ramp_steps] (dr arm; default {DEFAULT_DR})")
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--timeout-s", type=int, default=14400)
    ap.add_argument("--expect-counter", type=int, default=None,
                    help="also require this alg_state counter (e.g. 30450 for TDR's model_28991 after EXTEND)")
    ap.add_argument("--wait-idle", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)
    why = RL.valid_name(a.name)
    if why:
        raise Usage(why)
    for v in ("iterations", "num_envs", "timeout_s"):
        if getattr(a, v) <= 0:
            raise Usage(f"--{v.replace('_', '-')} must be a positive integer")
    if a.seed < 0:
        raise Usage("--seed must be >= 0")
    for flag, p in (("--checkpoint", a.checkpoint), ("--env-file", a.env_file)):
        if not os.path.isabs(p):
            raise Usage(f"{flag} must be absolute: {p!r}")
        if re.search(r"[\s*?\[\]]", p):
            raise Usage(f"{flag} has whitespace / glob characters: {p!r}")
        if not Path(p).is_file():
            raise Usage(f"{flag}: no such file {p}")
    if a.arm == "ctl" and a.dr is not None:
        raise Usage("--dr is refused on the ctl arm (TCTL is the no-DR control)")
    if a.arm == "dr" and a.dr is None:
        a.dr = DEFAULT_DR
    if a.dr is not None and (not a.dr.strip() or re.search(r"\s", a.dr)):
        raise Usage(f"--dr must be lo:hi[:ramp_steps] without whitespace, got {a.dr!r}")
    return a


# ============================================================================== pure builders
def dr_var() -> str:
    return RL.dr_knobs().ROD_SPEED_VAR


def compose_env(args, base: dict, env_lines) -> tuple[dict, dict, list[str]]:
    """``(child env, its KRABBY_* dict, refusals)``."""
    problems = []
    env = {k: v for k, v in base.items() if not k.startswith(BASE_DROP_PREFIXES) and k not in BASE_DROP_KEYS}
    env.update(ADDED_ENV)
    krabby = {}
    dvar = dr_var()
    for key, value in env_lines:
        if key.startswith(REFUSED_PREFIXES):
            problems.append(f"{key} in the env file: a training launch never carries a KRABBY_STRESS_* key")
        if key in REFUSED_KEYS:
            problems.append(f"{key} in the env file: the omni launch never exports it (run_phases _subenv)")
        if key == dvar:
            problems.append(f"{key} in the env file: the DR key comes from --dr on the dr arm only")
        if key.startswith("KRABBY_DR_ROD") and key != dvar:
            problems.append(f"{key} in the env file: a look-alike of {dvar} (dr_knobs refuses it)")
        krabby[key] = value
    if args.arm == "dr":
        krabby[dvar] = args.dr
    env.update(krabby)
    return env, krabby, problems


def trainer_argv(args) -> list[str]:
    """run_phases.train's command (``run_phases.py:1511-1516``) for this launch."""
    return [RL.PY, str(RL.TRAIN), "--task", TASK, "--headless", "--num_envs", str(args.num_envs),
            "--seed", str(args.seed), "--max_iterations", str(args.iterations),
            "--resume", "--checkpoint", args.checkpoint, "--run_name", args.name]


def template_argv(args) -> list[str]:
    """PF5's independent spelling of the same template."""
    return ["/home/nickmagus/krabby/isaac_venv/bin/python",
            str(RL.W / "parkour" / "scripts" / "rsl_rl" / "train.py"),
            "--task", "Isaac-Crab-Hex-Omni-Teacher-v0", "--headless", "--num_envs", str(args.num_envs),
            "--seed", str(args.seed), "--max_iterations", str(args.iterations), "--resume", "--checkpoint",
            str(Path(args.checkpoint)), "--run_name", args.name]


def wrap(argv: list[str], unit: str, timeout_s: int) -> list[str]:
    return ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={MEMORY_MAX}",
            "-p", "MemorySwapMax=0", "timeout", "--signal=TERM", str(int(timeout_s)), *argv]


def env_file_text(env: dict, krabby: dict) -> str:
    lines = ["# composed by rsd_train.py: the user-service environment minus KRABBY_* / TP_* / DRS_* / RSD_* /",
             "# PYTHONPATH, plus the keys below (the base keys are listed in results/runs/<name>.json)"]
    lines += [f"{k}={env[k]}" for k in ADDED_ENV]
    lines += [f"{k}={v}" for k, v in sorted(krabby.items())]
    return "\n".join(lines) + "\n"


def p3b_dict() -> dict:
    """The env dict of ``F19/orchestrator.log`` line 30 (the p3b launch), ``ast.literal_eval``'d."""
    lines = RL.P3B_ORCH_LOG.read_text().split("\n")
    line = lines[RL.P3B_ORCH_LINE - 1]
    if f"{RL.P3B_TAG}:" not in line or " | env " not in line:
        raise RuntimeError(f"{RL.P3B_ORCH_LOG}:{RL.P3B_ORCH_LINE} is not the {RL.P3B_TAG} launch line")
    return ast.literal_eval(line.split(" | env ", 1)[1])


def steps_per_iteration() -> int:
    m = re.search(r"^num_steps_per_env:\s*(\d+)\s*$", (RL.P3B_PARAMS / "agent.yaml").read_text(), re.M)
    if not m:
        raise RuntimeError(f"no num_steps_per_env in {RL.P3B_PARAMS / 'agent.yaml'}")
    return int(m.group(1))


def head_before_first_iteration(text: str) -> str:
    m = RL.ITER_RE.search(text)
    return text[:m.start()] if m else text


def p3b_knob_lines() -> list[str]:
    return RL.knob_lines(head_before_first_iteration(_read_head(RL.P3B_TRAIN_LOG)))


def _read_head(path: Path, n: int = HEAD_MAX_BYTES) -> str:
    with open(path, "rb") as fh:
        return fh.read(n).decode(errors="replace")


# ============================================================================== yaml comparison (PF12)
def strip_block(text: str, key: str) -> tuple[str, list[str]]:
    """Remove the ``key:`` mapping entry at any indent together with its more-indented children."""
    out, removed, cut_indent = [], [], None
    for ln in text.split("\n"):
        ind = len(ln) - len(ln.lstrip(" "))
        if cut_indent is not None:
            if ln.strip() and ind <= cut_indent:
                cut_indent = None
            else:
                removed.append(ln)
                continue
        if re.match(r"^\s*%s:(\s|$)" % re.escape(key), ln):
            cut_indent = ind
            removed.append(ln)
            continue
        out.append(ln)
    return "\n".join(out), removed


def strip_top_keys(text: str, keys) -> tuple[str, list[str]]:
    out, removed = [], []
    for ln in text.split("\n"):
        if any(re.match(r"^%s:(\s|$)" % re.escape(k), ln) for k in keys):
            removed.append(ln)
        else:
            out.append(ln)
    return "\n".join(out), removed


def compare_params(new_dir: Path, ref_dir: Path, arm: str, spec: dict | None) -> dict:
    """PF12: the new run's params/*.yaml against p3b's, after root normalisation."""
    import difflib  # noqa: PLC0415

    out = {"ok": True, "files": {}}
    for fname in ("env.yaml", "agent.yaml"):
        p_new, p_ref = new_dir / fname, ref_dir / fname
        if not p_new.is_file():
            out["ok"] = False
            out["files"][fname] = {"error": f"missing {p_new}"}
            continue
        new, ref = RL.normalise_roots(p_new.read_text()), RL.normalise_roots(p_ref.read_text())
        rec = {}
        if fname == "env.yaml":
            new, removed_new = strip_block(new, "rod_speed_dr")
            ref, removed_ref = strip_block(ref, "rod_speed_dr")
            rec["rod_speed_dr_block"] = removed_new
            if removed_ref:
                rec["error_ref"] = f"p3b's env.yaml carries a rod_speed_dr block: {removed_ref[:3]}"
                out["ok"] = False
            if arm == "ctl" and removed_new:
                rec["error_block"] = "the ctl arm's env.yaml carries a rod_speed_dr block"
                out["ok"] = False
            if arm == "dr":
                blob = "\n".join(removed_new)
                want = [f"lo: {spec['lo']}", f"hi: {spec['hi']}", f"ramp_steps: {spec['ramp_steps']}"]
                miss = [w for w in want if w not in blob]
                if miss:
                    rec["error_block"] = f"rod_speed_dr block missing {miss} (got {removed_new})"
                    out["ok"] = False
        else:
            new, rm_new = strip_top_keys(new, ("run_name", "load_checkpoint"))
            ref, rm_ref = strip_top_keys(ref, ("run_name", "load_checkpoint"))
            rec["excepted"] = {"new": rm_new, "p3b": rm_ref}
        diff = list(difflib.unified_diff(ref.split("\n"), new.split("\n"), "p3b/" + fname, "new/" + fname,
                                         lineterm="", n=1))
        rec["diff"] = diff[:80]
        rec["equal"] = not diff
        if diff:
            out["ok"] = False
        out["files"][fname] = rec
    return out


# ============================================================================== pre-launch pre-flight
def pf(pid: str, ok, detail: str, **data) -> dict:
    return {"id": pid, "ok": ok, "detail": detail, "data": data}


PROBE_SNIPPET = r"""
import importlib.util, json, sys
sys.path.insert(0, {script_dir!r})
sys.path.insert(0, {src!r})
def spec_of(name):
    parts = name.split('.')
    spec = importlib.util.find_spec(parts[0])
    for i in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            return None
        full, path, spec = '.'.join(parts[:i + 1]), list(spec.submodule_search_locations), None
        for finder in sys.meta_path:
            fs = getattr(finder, 'find_spec', None)
            if fs is not None:
                spec = fs(full, path)
                if spec is not None:
                    break
    return spec
mods = {{}}
for n in {names!r}:
    s = spec_of(n)
    mods[n] = None if s is None else {{'origin': s.origin, 'locations': list(s.submodule_search_locations or [])}}
print(json.dumps({{'sys_path': sys.path[:6], 'modules': mods}}))
"""


def module_probe(env: dict) -> dict:
    code = PROBE_SNIPPET.format(script_dir=str(RL.TRAIN.parent),
                                src=str(Path(os.path.abspath(RL.TRAIN.parent / ".." / ".." / "parkour_tasks"))),
                                names=list(PROBE_MODULES))
    r = subprocess.run([RL.PY, "-P", "-B", "-c", code], cwd=str(RL.PARKOUR), env=env, capture_output=True,
                       text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"module probe rc={r.returncode}: {r.stderr.strip()[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def under_w(path) -> bool:
    return bool(path) and str(path).startswith(str(RL.W) + os.sep)


def pf1() -> dict:
    try:
        cs = RL.code_state()
    except Exception as exc:  # noqa: BLE001
        return pf("PF1", False, f"code_state failed: {exc}")
    pin = RL.launch_pin()
    why = []
    if cs.get("dirty"):
        why.append(f"dirty code/eval paths {cs['dirty'][:6]}{' ...' if len(cs['dirty']) > 6 else ''}")
    if Path(cs.get("repo", "")) != RL.W:
        why.append(f"run_candidates.REPO {cs.get('repo')} is not this worktree")
    if pin is None:
        why.append(f"no launch pin {RL.LAUNCH_PIN} (written when the reference arm starts)")
    else:
        if pin.get("code_sha") != cs["code_sha"]:
            why.append(f"code_sha {cs['code_sha'][:12]} != pin {str(pin.get('code_sha'))[:12]}")
        if pin.get("eval_sha") != RL.EVAL_SHA_PIN:
            why.append(f"the pin's eval_sha {str(pin.get('eval_sha'))[:12]} != {RL.EVAL_SHA_PIN[:12]}")
    if cs["eval_sha"] != RL.EVAL_SHA_PIN:
        why.append(f"eval_sha {cs['eval_sha'][:12]} != {RL.EVAL_SHA_PIN[:12]}")
    return pf("PF1", not why, "; ".join(why) or f"clean; code_sha {cs['code_sha'][:12]} = pin; eval_sha "
              f"{cs['eval_sha'][:8]}", code_state=cs, pin=pin)


def pf2(env: dict) -> dict:
    try:
        probe = module_probe(env)
    except Exception as exc:  # noqa: BLE001
        return pf("PF2", False, f"module probe failed: {exc}")
    bad = []
    for n, rec in probe["modules"].items():
        if rec is None:
            bad.append(f"{n}: not found")
            continue
        where = rec["origin"] if rec["origin"] not in (None, "namespace") else (rec["locations"] or [None])[0]
        if not under_w(where):
            bad.append(f"{n}: {where}")
    return pf("PF2", not bad, "; ".join(bad) or "every module resolves under the worktree", **probe)


def pf3(args) -> dict:
    try:
        f = RL.checkpoint_facts(Path(args.checkpoint))
    except Exception as exc:  # noqa: BLE001
        return pf("PF3", False, f"torch.load failed: {exc}")
    why = []
    if f["file_iter"] is None:
        why.append("the checkpoint is not named model_<iter>.pt")
    elif f["iter"] != f["file_iter"]:
        why.append(f"iter {f['iter']} != the filename's {f['file_iter']}")
    if not isinstance(f["counter"], int) or f["counter"] <= 0:
        why.append(f"alg_state counter {f['counter']!r} (a resume would rebuild it, not restore it)")
    lr = f["learning_rate"]
    if not isinstance(lr, float) or not math.isfinite(lr) or lr <= 0:
        why.append(f"alg_state learning_rate {lr!r}")
    known = RL.KNOWN_CHECKPOINTS.get(f["sha256"])
    if known:
        for k in ("iter", "counter"):
            if f[k] != known[k]:
                why.append(f"{known['label']}: {k} {f[k]} != {known[k]}")
        if isinstance(lr, float) and not math.isclose(lr, known["learning_rate"], rel_tol=1e-9):
            why.append(f"{known['label']}: learning_rate {lr} != {known['learning_rate']}")
    if args.expect_counter is not None and f["counter"] != args.expect_counter:
        why.append(f"counter {f['counter']} != --expect-counter {args.expect_counter}")
    # CHARTER 3.1 / 3.2: TDR and TCTL resume T*, the fine-tune ramps; the EXTEND launch (TDR's model_28991) runs
    # without a ramp and is named explicitly by --expect-counter
    tstar = bool(known and known.get("tstar"))
    try:
        ramp = RL.dr_knob_line(args.dr)[0]["ramp_steps"] if args.arm == "dr" else None
    except Exception:  # noqa: BLE001 -- PF6 reports the parse
        ramp = None
    if args.arm == "ctl" and not tstar:
        why.append("the ctl arm (TCTL) resumes T* only (CHARTER 3.2)")
    elif args.arm == "dr" and tstar and ramp == 0:
        why.append("a T* fine-tune ramps (CHARTER 3.1 0.6:1.35:12000); a ramp-free --dr is the extension's")
    elif args.arm == "dr" and not tstar:
        if args.expect_counter is None:
            why.append("a resume of a checkpoint other than T* (the EXTEND launch) needs --expect-counter")
        if ramp:
            why.append(f"the extension runs without a ramp (CHARTER 3.1: --dr 0.6:1.35), got ramp_steps {ramp}")
    expected = {
        "loading": f"[INFO]: Loading model checkpoint from: {args.checkpoint}",
        "restored": (f"[INFO]: Restored algorithm state: counter {f['counter']}, learning_rate {lr:.3g}"
                     if isinstance(lr, float) else None),
        "first_title": f"Learning iteration {f['iter']}/{(f['iter'] or 0) + args.iterations}",
        "final_ckpt": f"model_{(f['iter'] or 0) + args.iterations - 1}.pt",
    }
    f.pop("keys", None)
    detail = "; ".join(why) or (f"sha {f['sha256'][:8]} iter {f['iter']} counter {f['counter']} lr {lr:.3g}"
                                + (f" (= {known['label']})" if known else " (expected values taken from the file)"))
    return pf("PF3", not why, detail, facts=f, known=known, expected=expected)


def pf4(args, krabby: dict, refusals: list[str], env_problems: list[str]) -> dict:
    why = list(env_problems) + list(refusals)
    try:
        want = p3b_dict()
    except Exception as exc:  # noqa: BLE001
        return pf("PF4", False, f"cannot read the p3b launch dict: {exc}")
    want.pop("PYTHONUNBUFFERED", None)
    if args.arm == "dr":
        want[dr_var()] = args.dr
    missing = sorted(set(want) - set(krabby))
    extra = sorted(set(krabby) - set(want))
    differ = sorted(k for k in set(want) & set(krabby) if want[k] != krabby[k])
    if missing or extra or differ:
        why.append(f"composed KRABBY_* != p3b's: missing {missing} extra {extra} differ {differ}")
    sha = RL.sha256_file(args.env_file)
    return pf("PF4", not why, "; ".join(why) or f"env file sha {sha[:12]}; {len(krabby)} KRABBY_* keys = p3b's "
              f"{len(want) - (1 if args.arm == 'dr' else 0)}" + (" + the DR key" if args.arm == "dr" else ""),
              env_file=args.env_file, env_file_sha256=sha, n_keys=len(krabby))


def pf5(args, argv: list[str]) -> dict:
    why = []
    tmpl = template_argv(args)
    if argv != tmpl:
        why.append(f"argv != template: {argv} vs {tmpl}")
    if argv[:2] != [RL.PY, str(RL.TRAIN)]:
        why.append("the trainer must be spawned as `PY train.py` (no -I / -E: they drop PYTHONPATH)")
    if not os.path.isabs(args.checkpoint):
        why.append("the checkpoint is not absolute")
    if not RL.TRAIN.is_file():
        why.append(f"no {RL.TRAIN}")
    rp = RL.RUN_PHASES_PY.read_text()
    gone = [t for t in RUN_PHASES_CMD_TEXT if t not in rp]
    if gone:
        why.append(f"run_phases.train's command no longer reads {gone}")
    return pf("PF5", not why, "; ".join(why) or f"argv = run_phases template; cwd {RL.PARKOUR}", argv=argv,
              cwd=str(RL.PARKOUR))


def pf6(args, krabby: dict) -> dict:
    try:
        spi = steps_per_iteration()
    except Exception as exc:  # noqa: BLE001
        return pf("PF6", False, f"cannot read num_steps_per_env: {exc}")
    mod = RL.dr_knobs()
    try:
        armed = mod.parse_dr_knobs(krabby)
    except ValueError as exc:
        return pf("PF6", False, f"dr_knobs refuses the composed env: {exc}")
    if args.arm == "ctl":
        return pf("PF6", not armed, f"armed {armed}" if armed else "no DR knob armed (ctl)", armed=armed,
                  steps_per_iteration=spi)
    try:
        spec, line = RL.dr_knob_line(args.dr)
    except Exception as exc:  # noqa: BLE001
        return pf("PF6", False, f"{dr_var()}={args.dr!r} refused: {exc}")
    ramp_iters = spec["ramp_steps"] / spi
    return pf("PF6", armed.get(dr_var()) == spec,
              f"lo {spec['lo']:g} hi {spec['hi']:g} ramp_steps {spec['ramp_steps']} = {ramp_iters:g} iterations "
              f"at {spi} steps/iteration", spec=spec, knob_line=line, ramp_iterations=ramp_iters,
              steps_per_iteration=spi)


def pf7(args, spec: dict | None) -> dict:
    if args.arm == "ctl":
        return pf("PF7", None, "no draw (ctl arm)")
    if not spec:
        return pf("PF7", False, "no prediction: PF6 did not parse the knob")
    try:
        p = RL.predicted_dr(spec, args.num_envs, args.seed, 0)
    except Exception as exc:  # noqa: BLE001
        return pf("PF7", False, f"prediction failed: {exc}")
    ok = p["line"] is not None and under_w(p["module"]) and spec["lo"] - 1e-9 <= p["lo_eff"]
    return pf("PF7", ok, f"seed {p['seed']} n {p['num_envs']} bounds [{p['lo_eff']:g}, {p['hi']:g}] min/mean/max "
              f"{p['min']:.4f}/{p['mean']:.4f}/{p['max']:.4f} q10/q50/q90 {p['q10']:.4f}/{p['q50']:.4f}/"
              f"{p['q90']:.4f} sha16 {p['sha16']}", **p)


def pf8(args, *, check_busy: bool = True) -> dict:
    why = []
    runs = RL.existing_runs(args.name)
    if runs:
        why.append(f"run(s) named {args.name} exist: {runs} (a crashed run is never resumed: pick a new name)")
    log = RL.TRAIN_OUT / f"{args.name}.log"
    if log.exists():
        why.append(f"{log} exists")
    rec = RL.RUNS / f"{args.name}.json"
    if rec.exists():
        why.append(f"{rec} exists")
    g = RL.guard()
    why += g["problems"]
    b = RL.busy(exclude=(os.getpid(),)) if check_busy else []
    if b:
        why.append(f"busy: {[f'{pid} {a[:120]}' for pid, a in b]}")
    return pf("PF8", not why, "; ".join(why) or f"name free; guard ok (free {min(v for v in g['free_gb'].values() if isinstance(v, float)):.0f} GB, "
              f"DDC {g['ddc_gb']:.0f} GB); not busy", existing=runs, guard=g, busy=[list(x) for x in b],
              guard_problems=g["problems"])


# ============================================================================== in-run pre-flight (PF9-13)
def inrun_checks(args, text: str, pre: dict, run_dir: Path | None) -> list[dict]:
    head = head_before_first_iteration(text)
    head_lines = [ln.rstrip("\r") for ln in head.split("\n")]
    exp = pre["PF3"]["data"].get("expected") or {}
    out = []
    # PF9
    miss = [k for k in ("loading", "restored") if not exp.get(k) or exp[k] not in head_lines]
    got = [ln for ln in head_lines if "Loading model checkpoint from" in ln or "Restored algorithm state" in ln]
    out.append(pf("PF9", not miss, f"missing {[exp.get(k) for k in miss]}; got {got}" if miss
                  else "checkpoint path and alg_state restored as recorded", got=got))
    # PF10
    want = list(pre["_p3b_knob_lines"])
    if args.arm == "dr":
        want.append(pre["PF6"]["data"]["knob_line"])
    d = RL.multiset_diff(RL.knob_lines(head), want)
    out.append(pf("PF10", not d["missing"] and not d["extra"],
                  f"missing {d['missing']} extra {d['extra']}" if (d["missing"] or d["extra"])
                  else f"{len(want)} [KNOB]/[omni-knobs] lines = p3b's" + (" + the DR line" if args.arm == "dr"
                                                                             else ""), **d))
    # PF11
    drl = RL.dr_lines(head)
    if args.arm == "dr":
        pred = pre["PF7"]["data"].get("line")
        ok = drl == [pred] and f"module={RL.RSD_PY}" in (drl[0] if drl else "")
        out.append(pf("PF11", ok, "the [DR] line = PF7's (worktree module)" if ok else f"got {drl}, want [{pred}]",
                      got=drl))
    else:
        out.append(pf("PF11", not drl, "no [DR] line (ctl)" if not drl else f"unexpected {drl}", got=drl))
    # PF12
    if run_dir is None:
        out.append(pf("PF12", False, f"no run dir *_{args.name} under {RL.W}/parkour/logs/rsl_rl/{EXPERIMENT}"))
    else:
        cmp = compare_params(run_dir / "params", RL.P3B_PARAMS, args.arm, pre["PF6"]["data"].get("spec"))
        bad = {f: r for f, r in cmp["files"].items() if not r.get("equal") or any(k.startswith("error") for k in r)}
        out.append(pf("PF12", cmp["ok"], f"differs: {json.dumps(bad)[:1500]}" if not cmp["ok"]
                      else "params/env.yaml + agent.yaml = p3b's (roots normalised; DR block / run_name / "
                           "load_checkpoint excepted)", run_dir=str(run_dir), compare=cmp))
    # PF13
    blocks = RL.iteration_blocks(text)
    if len(blocks) < 2:
        out.append(pf("PF13", False, f"{len(blocks)} iteration title(s) in the log (need a complete first block)"))
    else:
        it, tot, blk = blocks[0]
        it0 = pre["PF3"]["data"]["facts"]["iter"]
        why = []
        if (it, tot) != (it0, it0 + args.iterations):
            why.append(f"first block {it}/{tot}, want {it0}/{it0 + args.iterations}")
        vals = RL.block_values(blk, "DR/")
        if args.arm == "dr":
            why += dr_block_problems(vals, pre["PF6"]["data"]["spec"], 0, pre["PF6"]["data"]["steps_per_iteration"])
        elif vals:
            why.append(f"DR/ lines on the ctl arm: {vals}")
        out.append(pf("PF13", not why, "; ".join(why) or f"Learning iteration {it}/{tot}" +
                      (f"; DR/ {vals}" if vals else ""), title=[it, tot], dr=vals))
    return out


def dr_block_problems(vals: dict, spec: dict, k: int, spi: int) -> list[str]:
    """One iteration block's DR/ values against the knob (``k`` = iterations since the launch)."""
    why = [f"missing {key}" for key in DR_KEYS if key not in vals]
    if why:
        return why
    if any(not math.isfinite(v) for v in vals.values()):
        return [f"non-finite DR/ values {vals}"]
    mod = RL.rod_speed_dr()
    lo_hi_step = mod.rod_speed_bounds(spi * max(0, k - LAG_ITERS), spec["lo"], spec["hi"], spec["ramp_steps"])[0]
    lo_lo_step = mod.rod_speed_bounds(spi * (k + 1) + 1, spec["lo"], spec["hi"], spec["ramp_steps"])[0]
    lo, mean, mn, mx = (vals[key] for key in DR_KEYS)
    t = PRINT_TOL
    if not lo_lo_step - t <= lo <= lo_hi_step + t:
        why.append(f"DR/rod_speed_lo {lo} outside the ramp's [{lo_lo_step:.4f}, {lo_hi_step:.4f}] at +{k} "
                   f"(lag <= {LAG_ITERS} iterations)")
    # every env's scale was drawn at its last reset, when lo_eff >= lo_eff(block end): min >= lo_lo_step
    if mn < lo_lo_step - t or mx > spec["hi"] + t:
        why.append(f"DR/rod_speed_min/max {mn}/{mx} outside [{lo_lo_step:.4f}, {spec['hi']}] (lo_eff at +{k})")
    if not mn - t <= mean <= mx + t:
        why.append(f"DR/rod_speed_mean {mean} outside min/max {mn}/{mx}")
    return why


# ============================================================================== PF14
def pf14(args, text: str, run_dir: Path | None, pre: dict) -> dict:
    it0 = pre["PF3"]["data"]["facts"]["iter"]
    final_iter = it0 + args.iterations - 1
    why, data = [], {}
    final = run_dir / f"model_{final_iter}.pt" if run_dir else None
    if final is None or not final.is_file():
        why.append(f"no {final or 'run dir'}")
    else:
        try:
            f = RL.checkpoint_facts(final)
            data["final"] = {k: f[k] for k in ("path", "sha256", "iter", "counter", "learning_rate")}
            if f["iter"] != final_iter:
                why.append(f"{final.name} carries iter {f['iter']}")
        except Exception as exc:  # noqa: BLE001
            why.append(f"torch.load {final}: {exc}")
    blocks = RL.iteration_blocks(text)
    trace = [(it - it0, RL.block_values(b, "DR/")) for it, _, b in blocks]
    with_dr = [(k, v) for k, v in trace if v]
    data["blocks"] = len(blocks)
    data["blocks_with_dr"] = len(with_dr)
    if args.arm == "ctl":
        if with_dr:
            why.append(f"{len(with_dr)} block(s) carry DR/ lines on the ctl arm")
    else:
        spec = pre["PF6"]["data"]["spec"]
        spi = pre["PF6"]["data"]["steps_per_iteration"]
        if len(with_dr) < 0.9 * max(len(blocks), 1):
            why.append(f"only {len(with_dr)} of {len(blocks)} blocks carry DR/ lines")
        bad = [(k, p) for k, v in with_dr for p in dr_block_problems(v, spec, k, spi)]
        if bad:
            why.append(f"{len(bad)} DR/ block problem(s), first {bad[:3]}")
        late = [v for k, v in with_dr if k >= args.iterations - LATE_ITERS]
        if late:
            mid = (spec["lo"] + spec["hi"]) / 2
            lm = sum(v["DR/rod_speed_mean"] for v in late) / len(late)
            lmin = sum(v["DR/rod_speed_min"] for v in late) / len(late)
            lmax = sum(v["DR/rod_speed_max"] for v in late) / len(late)
            data["late"] = {"n": len(late), "mean": lm, "min_mean": lmin, "max_mean": lmax, "mid": mid}
            if abs(lm - mid) > LATE_MEAN_TOL:
                why.append(f"late DR/rod_speed_mean {lm:.4f} not within {LATE_MEAN_TOL} of (lo+hi)/2 = {mid:.4f}")
            if lmin > spec["lo"] + EDGE_TOL or lmax < spec["hi"] - EDGE_TOL:
                why.append(f"late DR/ min/max means {lmin:.4f}/{lmax:.4f} do not reach the band "
                           f"[{spec['lo']}, {spec['hi']}] within {EDGE_TOL}")
        else:
            why.append("no late DR/ blocks")
        ramp_iters = spec["ramp_steps"] / spi
        post = [v["DR/rod_speed_lo"] for k, v in with_dr if k >= ramp_iters + LAG_ITERS]
        if post and max(abs(x - spec["lo"]) for x in post) > PRINT_TOL:
            why.append(f"DR/rod_speed_lo after the ramp is not lo={spec['lo']} (max dev "
                       f"{max(abs(x - spec['lo']) for x in post):.4f})")
        data["lo_trace"] = [(k, v["DR/rod_speed_lo"]) for k, v in with_dr[:: max(1, len(with_dr) // 40)]]
    return pf("PF14", not why, "; ".join(why) or f"{final.name} (iter {final_iter}); " +
              (f"DR/ trace follows the ramp, late mean {data['late']['mean']:.4f}" if args.arm == "dr"
               else "no DR/ lines (ctl)"), **data)


# ============================================================================== provenance
def file_hashes(args) -> dict:
    paths = [RL.TRAIN, RL.RSD_PY, RL.ACTIONS_PY, RL.DR_KNOBS_PY, RL.OMNI_KNOBS_PY, RL.STRESS_KNOBS_PY,
             RL.RUN_PHASES_PY, RL.DISTILL_SH, Path(__file__).resolve(), Path(RL.__file__).resolve(),
             Path(args.env_file), Path(args.checkpoint), RL.P3B_ORCH_LOG, RL.P3B_TRAIN_LOG,
             RL.P3B_PARAMS / "env.yaml", RL.P3B_PARAMS / "agent.yaml"]
    out = {}
    for p in paths:
        try:
            out[str(p)] = RL.sha256_file(p)
        except OSError as exc:
            out[str(p)] = f"error: {exc}"
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


class Sentinel:
    """``RSD TRAIN ...`` lines: the run log (each on its own line: the child may be mid-line) + the events log."""

    def __init__(self, name: str, log: Path | None):
        self.name, self.log = name, log

    def __call__(self, kind: str, msg: str) -> None:
        line = f"RSD TRAIN {kind} {self.name} {msg} at={RL.now_iso()}"
        print(line, flush=True)
        for p, lead in ((self.log, "\n"), (RL.EVENTS_LOG, "")):
            if p is None:
                continue
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as fh:
                    fh.write(f"{lead}{line}\n")
            except OSError:
                pass


# ============================================================================== main flow
def preflight(args, env: dict, krabby: dict, refusals: list[str], env_problems: list[str],
              *, check_busy: bool = True) -> dict:
    res = {}
    res["PF1"] = pf1()
    res["PF2"] = pf2(env)
    res["PF3"] = pf3(args)
    res["PF4"] = pf4(args, krabby, refusals, env_problems)
    res["PF5"] = pf5(args, trainer_argv(args))
    res["PF6"] = pf6(args, krabby)
    res["PF7"] = pf7(args, res["PF6"]["data"].get("spec"))
    res["PF8"] = pf8(args, check_busy=check_busy)
    try:
        res["_p3b_knob_lines"] = p3b_knob_lines()
    except OSError as exc:
        res["_p3b_knob_lines"] = []
        res["PF4"]["ok"] = False
        res["PF4"]["detail"] += f"; cannot read p3b's train log: {exc}"
    return res


def pf_table(res: dict) -> list[str]:
    out = []
    for k in sorted((k for k in res if k.startswith("PF")), key=lambda x: int(x[2:])):
        r = res[k]
        tag = "PASS" if r["ok"] else ("n/a" if r["ok"] is None else "FAIL")
        out.append(f"{k} {tag} {r['detail']}")
    return out


def expectations(args, res: dict) -> dict:
    exp = dict(res["PF3"]["data"].get("expected") or {})
    exp["knob_lines"] = list(res["_p3b_knob_lines"]) + (
        [res["PF6"]["data"]["knob_line"]] if args.arm == "dr" and res["PF6"]["data"].get("knob_line") else [])
    exp["dr_line"] = res["PF7"]["data"].get("line") if args.arm == "dr" else None
    exp["run_dir_glob"] = str(RL.W / "parkour" / "logs" / "rsl_rl" / EXPERIMENT / f"<ts>_{args.name}")
    return exp


def _stop_requested(stop_file: Path) -> str | None:
    if STOP_SIGNALS:
        return f"signal {STOP_SIGNALS[0]}"
    if stop_file.exists():
        return f"stop file {stop_file}"
    return None


def _tail(path: Path, n: int = TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def supervise(args, proc, log: Path, unit: str, say, pre: dict, record: dict, rec_path: Path) -> int:
    t0 = time.time()
    stop_file = RL.RUNS / f"{args.name}.stop"
    inrun_done = False
    status, why = None, None

    def stop(reason: str):
        say("PF" if reason.startswith("PF") else "STOP", f"stopping the trainer: {reason}")
        rec = RL.stop_trainer(lambda: RL.trainer_pids(unit, args.name, exclude=(proc.pid,)),
                              say=lambda m: say("STOP", m))
        record.setdefault("stops", []).append({"reason": reason, **rec})
        if rec.get("sigkill"):
            say("FAIL", "sigkill=1: the trainer needed SIGKILL -- open the DerivedDataCache watch before the next Kit "
                        "start")
        try:
            proc.wait(timeout=RL.KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            pass
        return rec

    while proc.poll() is None:
        time.sleep(POLL_S)
        reason = _stop_requested(stop_file)
        if reason:
            stop(reason)
            status, why = "interrupted", reason
            break
        tail = _tail(log)
        if NAN_RE.search(tail):
            stop("NaN in the training log")
            status, why = "nan", "NaN in the training log"
            break
        if "Traceback" in tail:
            stop("Traceback in the training log")
            status, why = "traceback", "Traceback in the training log"
            break
        try:
            idle = time.time() - log.stat().st_mtime
        except OSError:
            idle = 0.0
        if idle > STALL_S:
            stop(f"no log progress for {STALL_S // 60} min")
            status, why = "stall", f"no log progress for {STALL_S} s"
            break
        if time.time() - t0 > args.timeout_s + 2 * RL.TERM_GRACE_S + 300:
            stop("the wall bound passed and the trainer is still alive")
            status, why = "timeout_hang", "alive past the timeout bound"
            break
        if not inrun_done:
            text = _read_head(log)
            if len(RL.ITER_RE.findall(text)) >= 2:
                results = inrun_checks(args, text, pre, RL.find_run_dir(args.name, EXPERIMENT))
                inrun_done = True
                record["preflight_inrun"] = results
                write_json(rec_path, record)
                for r in results:
                    say("PF", f"{r['id']} {'PASS' if r['ok'] else 'FAIL'} {r['detail'][:600]}")
                failed = [r["id"] for r in results if not r["ok"]]
                if failed:
                    stop(f"{','.join(failed)} failed in-run")
                    status, why = "preflight_fail", f"{','.join(failed)} failed"
                    break
    rc = proc.wait()
    record["rc"] = rc
    record["ended"] = RL.now_iso()
    record["wall_s"] = round(time.time() - t0, 1)
    run_dir = RL.find_run_dir(args.name, EXPERIMENT)
    record["run_dir"] = str(run_dir) if run_dir else None
    if status is None and not inrun_done:
        # the trainer exited before a complete first block: read what there is (it will fail)
        results = inrun_checks(args, _read_head(log), pre, run_dir)
        record["preflight_inrun"] = results
        if any(not r["ok"] for r in results):
            status, why = "preflight_fail", "the trainer exited before the in-run checks passed"
    if status is None:
        if _stop_requested(stop_file):
            status, why = "interrupted", _stop_requested(stop_file)
        elif rc == 124:
            status, why = "timeout", f"timeout --signal=TERM {args.timeout_s}"
        elif rc != 0:
            status, why = "crashed", f"rc {rc}"
    if status is None:
        text = Path(log).read_text(errors="replace")
        r14 = pf14(args, text, run_dir, pre)
        record["pf14"] = r14
        say("PF", f"PF14 {'PASS' if r14['ok'] else 'FAIL'} {r14['detail'][:600]}")
        if not r14["ok"]:
            status, why = "pf14_fail", r14["detail"][:300]
        else:
            status = "ok"
    record["status"], record["why"] = status, why
    write_json(rec_path, record)
    if status == "ok":
        say("END", f"status=ok rc={rc} ckpt={record['pf14']['data']['final']['path']}")
        return EXIT_OK
    say("FAIL", f"status={status} rc={rc} why={why} -- never auto-resumed (CHARTER 2); relaunch under a new name")
    return {"preflight_fail": EXIT_PF_INRUN, "pf14_fail": EXIT_PF14}.get(status, EXIT_TRAIN)


def _on_signal(signum, _frame) -> None:
    STOP_SIGNALS.append(signal.Signals(signum).name)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except Usage as exc:
        print(f"rsd_train: {exc}", file=sys.stderr)
        return EXIT_USAGE
    dry = args.dry_run or args.preflight_only
    env_lines, env_problems = RL.read_env_lines(Path(args.env_file))
    base = RL.systemd_user_env()
    env, krabby, refusals = compose_env(args, base, env_lines)
    if refusals and not dry:
        for r in refusals:
            print(f"rsd_train: refused: {r}", file=sys.stderr)
        return EXIT_USAGE
    unit = f"rsd-train-{args.name}-{int(time.time())}"
    argv_t = trainer_argv(args)
    cmd = wrap(argv_t, unit, args.timeout_s)
    log = RL.TRAIN_OUT / f"{args.name}.log"
    rec_path = RL.RUNS / f"{args.name}.json"
    env_path = RL.ENVS / f"{args.name}.train.env"

    old_handlers = {}
    try:
        return _launch(args, dry, env, krabby, refusals, env_problems, base, unit, argv_t, cmd, log, rec_path,
                       env_path, old_handlers)
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)


def _launch(args, dry, env, krabby, refusals, env_problems, base, unit, argv_t, cmd, log, rec_path, env_path,
            old_handlers) -> int:
    lock_fh = None
    if not dry:
        RL.TRAIN_OUT.mkdir(parents=True, exist_ok=True)
        lock_fh = open(RL.TRAIN_OUT / "rsd_train.lock", "a")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("rsd_train: another rsd_train launcher holds the lock", file=sys.stderr)
            return EXIT_BUSY
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[s] = signal.signal(s, _on_signal)
        if args.wait_idle:
            while RL.busy(exclude=(os.getpid(),)):
                if STOP_SIGNALS:
                    return EXIT_TRAIN
                time.sleep(IDLE_POLL_S)
    res = preflight(args, env, krabby, refusals, env_problems)
    table = pf_table(res)
    record = {
        "name": args.name, "arm": args.arm, "created": RL.now_iso(), "launcher_argv": sys.argv,
        "cli": vars(args), "task": TASK, "experiment": EXPERIMENT, "trainer_argv": argv_t, "wrapper_argv": cmd,
        "cwd": str(RL.PARKOUR), "unit": unit, "log": str(log), "env_path": str(env_path),
        "env": {"krabby": krabby, "added": dict(ADDED_ENV), "base_keys": sorted(base),
                "dropped_from_base": sorted(k for k in base if k.startswith(BASE_DROP_PREFIXES)
                                            or k in BASE_DROP_KEYS),
                "full_env_sha256": RL.sha256_text(json.dumps(env, sort_keys=True))},
        "code_sha": (res["PF1"]["data"].get("code_state") or {}).get("code_sha"),
        "eval_sha": (res["PF1"]["data"].get("code_state") or {}).get("eval_sha"),
        "head": RL.git_head(), "file_hashes": file_hashes(args),
        "preflight": {k: v for k, v in res.items() if k.startswith("PF")},
        "expected_inrun": expectations(args, res),
    }
    failed = [k for k in res if k.startswith("PF") and res[k]["ok"] is False]
    if dry:
        print("\n".join(table))
        if args.dry_run:
            print("\n# composed env file (envs/%s.train.env, not written in a dry run)" % args.name)
            print(env_file_text(env, krabby), end="")
            print("# cwd", RL.PARKOUR)
            print("# command:", " ".join(cmd))
            print("# log", log, "| record", rec_path)
            print("# expected in-run lines:")
            for k, v in record["expected_inrun"].items():
                print(f"#   {k}: {v}")
            if refusals:
                print("# REFUSALS (a real launch exits 2):", refusals)
            return EXIT_OK
        return EXIT_OK if not failed else EXIT_PF
    if failed:
        print("\n".join(table), file=sys.stderr)
        g = res["PF8"]["data"]
        say = Sentinel(args.name, None)
        say("FAIL", f"status=refused failed={','.join(failed)} (nothing launched)")
        if failed == ["PF8"] and g.get("busy") and not g.get("guard_problems") and not g.get("existing"):
            return EXIT_BUSY
        if failed == ["PF8"] and g.get("guard_problems"):
            return EXIT_GUARD
        return EXIT_PF
    # ------------------------------------------------------------------ launch
    RL.ENVS.mkdir(parents=True, exist_ok=True)
    env_path.write_text(env_file_text(env, krabby))
    record["status"] = "launched"
    record["launched"] = RL.now_iso()
    write_json(rec_path, record)
    say = Sentinel(args.name, log)
    say("START", f"arm={args.arm} unit={unit}.scope ck={args.checkpoint} iters={args.iterations} "
                 f"dr={args.dr or '-'} log={log}")
    for line in table:
        say("PF", line)
    pre = dict(res)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(cmd, cwd=str(RL.PARKOUR), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    record["launcher_child_pid"] = proc.pid
    write_json(rec_path, record)
    try:
        return supervise(args, proc, log, unit, say, pre, record, rec_path)
    finally:
        if lock_fh is not None:
            lock_fh.close()


if __name__ == "__main__":
    sys.exit(main())
