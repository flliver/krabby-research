#!/usr/bin/env python3
"""PLAN step 8: launch and supervise THE rough-terrain teacher fine-tune (and the step-5 50-iteration launcher smoke).

    /home/nickmagus/krabby/isaac_venv/bin/python -I -B rt_train.py --name RT_s19 [--checkpoint TDR]
        [--env-file envs/RT_train.env] [--iterations 6000] [--num-envs 256] [--seed 19] [--timeout-s 18000]
        [--wait-idle] [--smoke] [--dry-run | --preflight-only]

The run (PLAN Design "Training"): resume TDR (``policy/rsd_9_p3b_rodspeed_dr/model_28991.pt``, sha 03c3e472; its
``alg_state`` -- counter 30450, learning rate -- is restored by the runner from the checkpoint, as rsd_train.py's
EXTEND launch did) on Isaac-Crab-Hex-Omni-Teacher-v0, ``--num_envs 256 --seed 19 --max_iterations 6000``: the
runner's iterations run ``it0 .. it0 + 6000 - 1`` and it saves ``model_<it0 + 5999>.pt`` last (model_34990.pt from
it0 28991: the absolute target rsd_train.py computes). ``--smoke`` (step 5) allows a name RT5_* and fewer
iterations; otherwise the name must be RT_s19-like and the iterations 6000 (pre-registered).

The child's environment: the user-service environment (``systemctl --user show-environment``) minus every KRABBY_*,
TP_*, RSD_*, RTQ_*, KIT_*, DRS_* and PYTHONPATH, plus OMNI_KIT_ACCEPT_EULA=yes, TERM=xterm, PYTHONUNBUFFERED=1,
PYTHONPATH=<this worktree>/parkour/parkour_tasks:<this worktree>/parkour, plus the env file's KRABBY_* lines verbatim
(rt_machinery.compose_child_env). The trainer is ``PY train.py ...`` (never -I: it would drop PYTHONPATH), cwd
<this worktree>/parkour, inside ``systemd-run --user --scope -q --unit=rt-train-<name>-<epoch> -p MemoryMax=48G -p
MemorySwapMax=0 timeout --signal=TERM <timeout>``. Stops (watchdogs, a failed in-run check, a signal to this
launcher, ``touch results/runs/<name>.stop``) send SIGTERM to the trainer python's own pid, wait 120 s, SIGTERM
again, wait 120 s -- never SIGKILL (a Kit killed mid-write corrupts the DerivedDataCache); a trainer still alive is
recorded and the launcher fails. A crashed run is never resumed: relaunch under a new name.

Pre-flight (pre-launch, refusing): PF1 the live build == results/pin.json and the code / eval tree clean; PF2 the env
file == TDR's training env + exactly the planned changes (rt_machinery.train_env_problems) and no smoke-only /
stress / phase key; PF3 the checkpoint: sha TDR, CPU torch.load iter = the filename's, alg_state counter / lr present
(TDR: 28991 / 30450); PF4 argv = the run_phases template; PF5 module origins (python -P with the composed env and
cwd) all under this worktree; PF6 the name is free (no run dir, log or record); PF7 the guard (DerivedDataCache, free
disk) and not busy (``--wait-idle`` waits). Post-launch: PF8 the trainer python's ``/proc/<pid>/environ``:
PYTHONPATH = this worktree, its KRABBY_* = the composed set exactly. In-run (after two iteration titles): PF9
'Loading model checkpoint from: CK' + 'Restored algorithm state: counter <PF3>, learning_rate <PF3>'; PF10 the
first title 'Learning iteration it0/it0+iterations'; PF11 the rt knob lines ([KNOB] KRABBY_SITE_MIX / SITE_LADDER /
RT_FRICTION) and the [rt_friction] draw line; PF12 the per-kind telemetry site_cnt_end/<kind> of every mix kind in
the first block. PF14 params/env.yaml + agent.yaml of the run dir against TDR's run params
(``rt_machinery.TDR_PARAMS``): every difference outside the allow-list the planned changes explain (terrain generator /
site keys, friction contract, DR ramp, run name / checkpoint / iterations / seed) is a refusal, and every allowed and
disallowed difference is logged (``rt_machinery.py params-diff`` is the same check on its own; the smoke block runs it
as S.train50.params). After: PF13 model_<final>.pt with that iter.

The iteration-1500 read (PLAN step 3 / settled 6): when the first block at it0 + 1500 is logged, ONE line
``RTQ NOTIFY early_read it=1500 fall_share[300-500]->[1300-1500] <kind>=a->b ...`` goes to stdout, the run log and
the events log (report only: no stop, no remedy).

Writes: the run log ``<MAIN>/parkour/outputs/omni_plan/rough_terrain/train/<name>.log`` (trainer stdout+stderr and
``RT TRAIN START|PF|NOTIFY|END|FAIL`` lines, mirrored to train/rt_train_events.log), the record
``results/runs/<name>.json``. Exit: 0 ok / dry (a dry pass with a failed pre-flight returns 5, except the expected pre-pin PF1: no pin yet / dirty
tree); 2 usage; 3 busy / lock; 4 guard; 5 pre-launch pre-flight; 6 in-run
pre-flight; 7 the trainer failed or was stopped; 8 PF13.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rt_machinery as M  # noqa: E402

MEMORY_MAX = "48G"
RUNS = M.RESULTS / "runs"
EVENTS = M.TRAIN_OUT / "rt_train_events.log"
STALL_S = 900
POLL_S = 15.0
TERM_GRACE_S = 120.0
PF8_WAIT_S = 180.0
REFUSED_KEYS = ("KRABBY_EPISODE_S", "KRABBY_PHASE", "KRABBY_SIDE_PHASE")
REFUSED_PREFIXES = ("KRABBY_STRESS_",)
NAN_RE = re.compile(r"Mean (reward|value_function|surrogate):\s*nan", re.I)
RT_KNOB_LINES = ("[KNOB] KRABBY_SITE_MIX=", "[KNOB] KRABBY_SITE_LADDER=rt consumed", "[KNOB] KRABBY_RT_FRICTION=")
EXIT_OK, EXIT_USAGE, EXIT_BUSY, EXIT_GUARD, EXIT_PF, EXIT_PF_INRUN, EXIT_TRAIN, EXIT_FINAL = 0, 2, 3, 4, 5, 6, 7, 8
STOP_SIGNALS: list[str] = []
PROBE_MODULES = ("parkour_tasks", "parkour_tasks.crab_hex_omni_task.mdp.rt_friction",
                 "parkour_tasks.crab_hex_omni_task.mdp.omni_terrain_knobs",
                 "parkour_tasks.crab_hex_omni_task.config.crab_hex.omni_knobs",
                 "parkour_tasks.crab_hex_forward_task.mdp.rod_speed_dr", "scripts.rsl_rl.runner_factory",
                 "scripts.rsl_rl.modules.on_policy_runner_with_extractor", "parkour_isaaclab",
                 "parkour_isaaclab.terrains.omni.site_terrains")
#: find a module's file WITHOUT importing its parents (a package __init__ may start Kit): rsd_train.py's probe
PROBE_SNIPPET = r"""
import importlib.util, json, sys
sys.path.insert(0, {script_dir!r})
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
    mods[n] = None if s is None else (s.origin if s.origin not in (None, 'namespace') else
                                       (list(s.submodule_search_locations or []) or [None])[0])
print(json.dumps(mods))
"""


class Usage(Exception):
    pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", default=str(M.TDR_CK))
    ap.add_argument("--env-file", default=str(M.RT_TRAIN_ENV))
    ap.add_argument("--iterations", type=int, default=M.TRAIN_ITERS)
    ap.add_argument("--num-envs", type=int, default=M.TRAIN_ENVS)
    ap.add_argument("--seed", type=int, default=M.TRAIN_SEED)
    ap.add_argument("--timeout-s", type=int, default=M.TRAIN_TIMEOUT_S)
    ap.add_argument("--wait-idle", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="the step-5 launcher smoke (name RT5_*, any iterations)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9_]+$", a.name):
        raise Usage(f"--name {a.name!r} must be [A-Za-z0-9_]+")
    if a.smoke and not a.name.startswith("RT5_"):
        raise Usage("--smoke runs are named RT5_*")
    if not a.smoke and (a.name.startswith("RT5_") or a.iterations != M.TRAIN_ITERS or a.num_envs != M.TRAIN_ENVS
                        or a.seed != M.TRAIN_SEED):
        raise Usage(f"the step-8 run is pre-registered: --iterations {M.TRAIN_ITERS} --num-envs {M.TRAIN_ENVS} "
                    f"--seed {M.TRAIN_SEED}, a non-RT5_ name (use --smoke for a launcher smoke)")
    for v in ("iterations", "num_envs", "timeout_s"):
        if getattr(a, v) <= 0:
            raise Usage(f"--{v.replace('_', '-')} must be positive")
    for flag, p in (("--checkpoint", a.checkpoint), ("--env-file", a.env_file)):
        if not os.path.isabs(p) or re.search(r"[\s*?\[\]]", p) or not Path(p).is_file():
            raise Usage(f"{flag} must be an existing absolute path without whitespace / globs: {p!r}")
    return a


# ============================================================================== helpers
def systemd_user_env() -> dict:
    r = subprocess.run(["systemctl", "--user", "show-environment", "-o", "json"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip().startswith("{"):
        return {str(k): str(v) for k, v in json.loads(r.stdout).items()}
    r = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, text=True, check=True)
    out = {}
    for ln in r.stdout.splitlines():
        k, sep, v = ln.partition("=")
        if sep:
            out[k] = v[2:-1].encode().decode("unicode_escape") if v.startswith("$'") and v.endswith("'") else v
    return out


def base_env() -> dict:
    env = systemd_user_env()
    if env.get("CUDA_VISIBLE_DEVICES", None) == "":
        env.pop("CUDA_VISIBLE_DEVICES")
    return env


def checkpoint_facts(path: Path) -> dict:
    import torch  # noqa: PLC0415

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    st = ck.get("alg_state") if isinstance(ck, dict) else None
    m = re.match(r"^model_(\d+)\.pt$", Path(path).name)
    return {"path": str(path), "sha256": M.sha256_file(path), "iter": ck.get("iter") if isinstance(ck, dict) else None,
            "file_iter": int(m.group(1)) if m else None,
            "counter": (st or {}).get("counter") if isinstance(st, dict) else None,
            "learning_rate": (st or {}).get("learning_rate") if isinstance(st, dict) else None}


def pf(pid, ok, detail, **data) -> dict:
    return {"id": pid, "ok": ok, "detail": detail, "data": data}


def existing_runs(name: str) -> list[str]:
    pat = re.compile(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_%s$" % re.escape(name))
    out = []
    for root in (M.W, M.MAIN):
        base = Path(root) / "parkour" / "logs" / "rsl_rl"
        if base.is_dir():
            for exp in base.iterdir():
                if exp.is_dir():
                    out += [str(d) for d in exp.iterdir() if d.is_dir() and pat.match(d.name)]
    return sorted(out)


def find_run_dir(name: str) -> Path | None:
    return M.find_run_dir(name)


def trainer_pids(name: str, exclude=()) -> list[int]:
    out = []
    skip = set(exclude) | {os.getpid()}
    for pid, a in M.ps_pid_args():
        t = a.split()
        if pid in skip or not t or not t[0].endswith("isaac_venv/bin/python") or len(t) < 2:
            continue
        if t[1].endswith("/scripts/rsl_rl/train.py") and any(t[i:i + 2] == ["--run_name", name]
                                                              for i in range(len(t) - 1)):
            out.append(pid)
    return sorted(out)


def proc_environ(pid: int) -> dict:
    raw = Path(f"/proc/{int(pid)}/environ").read_bytes()
    out = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.decode(errors="replace").partition("=")
        if sep:
            out[k] = v
    return out


def environ_problems(env: dict, krabby: dict) -> list[str]:
    """PF8: the trainer's real environment against the composed one."""
    why = []
    if env.get("PYTHONPATH") != M.PYTHONPATH_W:
        why.append(f"PYTHONPATH {env.get('PYTHONPATH')!r} != {M.PYTHONPATH_W!r}")
    got = {k: v for k, v in env.items() if k.startswith("KRABBY_")}
    if got != krabby:
        why.append(f"KRABBY_* differ: {sorted(k for k in set(got) | set(krabby) if got.get(k) != krabby.get(k))}")
    if env.get("OMNI_KIT_ACCEPT_EULA") != "yes":
        why.append("OMNI_KIT_ACCEPT_EULA != yes")
    return why


def pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[:1] not in (["Z"], ["X"])


def stop_trainer(pids, say) -> dict:
    """SIGTERM, 120 s, SIGTERM, 120 s. Never SIGKILL (the DerivedDataCache hazard): a survivor is recorded."""
    rec = {"pids": list(pids), "terms": 0, "alive_after": []}
    for n in (1, 2):
        for p in pids:
            if pid_alive(p):
                try:
                    os.kill(p, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        rec["terms"] = n
        say("STOP", f"SIGTERM #{n} to trainer pid(s) {list(pids)}")
        t0 = time.time()
        while time.time() - t0 < TERM_GRACE_S and any(pid_alive(p) for p in pids):
            time.sleep(2.0)
        if not any(pid_alive(p) for p in pids):
            return rec
    rec["alive_after"] = [p for p in pids if pid_alive(p)]
    say("FAIL", f"trainer pid(s) {rec['alive_after']} alive after two SIGTERMs: NOT killed (stop it by hand; "
                "check the DerivedDataCache before the next Kit start)")
    return rec


# ============================================================================== pre-flight
def preflight(a, env: dict, krabby: dict, compose_problems: list, check_busy: bool = True) -> dict:
    res = {}
    try:
        live = M.live_build()
        bad, warn = M.pin_problems(M.read_pin(), live, M.campaign_files(M.FROZEN_GLOBS),
                                   M.campaign_files(M.REPORTED_GLOBS))
        res["PF1"] = pf("PF1", not bad, "; ".join(bad) or f"build == pin (code {live['code_sha'][:12]} eval "
                        f"{live['eval_sha'][:12]}), clean", live=live, warnings=warn, problems=bad)
    except Exception as exc:  # noqa: BLE001
        res["PF1"] = pf("PF1", False, f"build check failed: {exc}")
    why = list(compose_problems) + M.train_env_problems(Path(a.env_file))
    why += [f"{k} in the env file (smoke-only / phase key)" for k in krabby if k in REFUSED_KEYS]
    why += [f"{k} in the env file (stress key)" for k in krabby if k.startswith(REFUSED_PREFIXES)]
    res["PF2"] = pf("PF2", not why, "; ".join(why) or f"env file = the planned set ({len(krabby)} KRABBY_* keys)",
                    env_file=a.env_file, env_file_sha256=M.sha256_file(a.env_file), krabby=krabby)
    try:
        f = checkpoint_facts(Path(a.checkpoint))
        why = []
        if f["sha256"] != M.TDR_SHA:
            why.append(f"checkpoint sha {f['sha256'][:8]} is not TDR's {M.TDR_SHA[:8]}")
        if f["iter"] is None or f["iter"] != f["file_iter"]:
            why.append(f"iter {f['iter']} != the filename's {f['file_iter']}")
        if f["iter"] != M.TDR_ITER or f["counter"] != M.TDR_COUNTER:
            why.append(f"iter/counter {f['iter']}/{f['counter']} != TDR's {M.TDR_ITER}/{M.TDR_COUNTER}")
        lr = f["learning_rate"]
        if not isinstance(lr, float) or not math.isfinite(lr) or lr <= 0:
            why.append(f"alg_state learning_rate {lr!r}")
        it0 = int(f["iter"] or 0)
        exp = {"loading": f"[INFO]: Loading model checkpoint from: {a.checkpoint}",
               "restored": (f"[INFO]: Restored algorithm state: counter {f['counter']}, learning_rate {lr:.3g}"
                            if isinstance(lr, float) else None),
               "first_title": f"Learning iteration {it0}/{it0 + a.iterations}",
               "final_iter": M.final_iteration(it0, a.iterations)}
        res["PF3"] = pf("PF3", not why, "; ".join(why) or f"TDR sha {f['sha256'][:8]} iter {f['iter']} counter "
                        f"{f['counter']} lr {lr:.3g}; final model_{exp['final_iter']}.pt", facts=f, expected=exp)
    except Exception as exc:  # noqa: BLE001
        res["PF3"] = pf("PF3", False, f"torch.load failed: {exc}", expected={})
    argv = M.trainer_argv(a.name, a.checkpoint, a.iterations, a.num_envs, a.seed)
    tmpl = ["/home/nickmagus/krabby/isaac_venv/bin/python", str(M.W / "parkour/scripts/rsl_rl/train.py"), "--task",
            "Isaac-Crab-Hex-Omni-Teacher-v0", "--headless", "--num_envs", str(a.num_envs), "--seed", str(a.seed),
            "--max_iterations", str(a.iterations), "--resume", "--checkpoint", a.checkpoint, "--run_name", a.name]
    why = [] if argv == tmpl else [f"argv {argv} != template {tmpl}"]
    if not Path(argv[1]).is_file():
        why.append(f"no {argv[1]}")
    res["PF4"] = pf("PF4", not why, "; ".join(why) or "argv = the run_phases template", argv=argv)
    try:
        code = PROBE_SNIPPET.format(script_dir=str(M.PARKOUR / "scripts" / "rsl_rl"), names=list(PROBE_MODULES))
        r = subprocess.run([M.PY, "-P", "-B", "-c", code], cwd=str(M.PARKOUR), env=env, capture_output=True,
                           text=True, timeout=180)
        mods = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        bad = [f"{n}: {p}" for n, p in mods.items() if not p or not str(p).startswith(str(M.W) + os.sep)]
        if r.returncode != 0:
            bad.append(f"probe rc {r.returncode}: {r.stderr.strip()[-300:]}")
        res["PF5"] = pf("PF5", not bad, "; ".join(bad) or "every module resolves under this worktree", modules=mods)
    except Exception as exc:  # noqa: BLE001
        res["PF5"] = pf("PF5", False, f"module probe failed: {exc}")
    why = [f"run dir(s) {existing_runs(a.name)}"] if existing_runs(a.name) else []
    for p in (M.TRAIN_OUT / f"{a.name}.log", RUNS / f"{a.name}.json"):
        if p.exists():
            why.append(f"{p} exists (a crashed run is never resumed: pick a new name)")
    res["PF6"] = pf("PF6", not why, "; ".join(why) or "name free")
    ddc = M.ddc_gb()
    free = {str(p): M.free_gb(p) for p in (M.MAIN, M.W) if Path(p).exists()}
    why = M.guard_problems(ddc, free)
    b = M.busy(exclude=(os.getpid(),)) if check_busy else []
    if b:
        why.append(f"busy: {[f'{pid} {x[:100]}' for pid, x in b]}")
    res["PF7"] = pf("PF7", not why, "; ".join(why) or f"guard ok (DDC {ddc:.0f} GB, free "
                    f"{min(free.values()):.0f} GB); not busy", ddc_gb=ddc, free_gb=free, busy=b)
    return res


#: the PF1 problems a dry pass tolerates (the pin is written once at the end of step 5; a parallel session may dirty
#: the tree): anything else failing PF1 (a moved sha, a changed frozen file) is a real failure
PF1_PRE_PIN = ("no pin ", "dirty code/eval paths")


def dry_failures(res: dict) -> list[str]:
    """The pre-flight ids a DRY pass counts as failures: every failed id except a PF1 whose only problems are the
    expected pre-pin ones (no pin yet / dirty tree: what S.check reports before ``build_check.py pin``)."""
    out = []
    for k in sorted(res, key=lambda x: int(x[2:])):
        r = res[k]
        if r["ok"]:
            continue
        if k == "PF1" and (r.get("data") or {}).get("problems") and all(
                str(p).startswith(PF1_PRE_PIN) for p in r["data"]["problems"]):
            continue
        out.append(k)
    return out


def params_pf(a, krabby: dict) -> dict:
    """PF14 (in-run, once params/ exists): the run's params/env.yaml + agent.yaml against TDR's, every difference
    outside the planned-change allow-list (rt_machinery.params_allow_rules) refused; every difference is listed."""
    run_dir = M.find_run_dir(a.name)
    if run_dir is None or not (run_dir / "params").is_dir():
        return pf("PF14", False, f"no run dir / params/ for {a.name} yet")
    expect = {"name": a.name, "iterations": a.iterations, "seed": a.seed, "checkpoint": a.checkpoint}
    res = M.params_check(run_dir / "params", M.TDR_PARAMS, expect, krabby, smoke=a.smoke)
    bad = [f"{f} {d['path']}: {M._short(d['ref'], 60)} -> {M._short(d['new'], 60)} [{d['why']}]"
           for f, r in res["files"].items() for d in r["disallowed"]] + [
        f"{f}: {r['error']}" for f, r in res["files"].items() if r.get("error")]
    n_ok = sum(len(r["allowed"]) for r in res["files"].values())
    detail = ("; ".join(bad[:8]) + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else "")) if bad else (
        f"params/env.yaml + agent.yaml = TDR's except {n_ok} planned difference(s)")
    return pf("PF14", res["ok"], detail, result=res)


def inrun_checks(text: str, pre: dict, kinds) -> list[dict]:
    head = text[:M.ITER_RE.search(text).start()] if M.ITER_RE.search(text) else text
    lines = [ln.rstrip("\r") for ln in head.split("\n")]
    exp = pre["PF3"]["data"].get("expected") or {}
    out = []
    miss = [k for k in ("loading", "restored") if not exp.get(k) or exp[k] not in lines]
    out.append(pf("PF9", not miss, f"missing {[exp.get(k) for k in miss]}" if miss else "checkpoint loaded, alg_state "
                  "restored as the file records"))
    blocks = M.iteration_blocks(text)
    title = f"Learning iteration {blocks[0][0]}/{blocks[0][1]}" if blocks else None
    out.append(pf("PF10", title == exp.get("first_title"), f"first title {title!r}, want {exp.get('first_title')!r}"))
    knob = [k for k in RT_KNOB_LINES if not any(ln.startswith(k) for ln in lines)]
    fr = any(ln.startswith("[rt_friction] ") for ln in lines)
    out.append(pf("PF11", not knob and fr, f"missing knob lines {knob}; [rt_friction] line {fr}" if knob or not fr
                  else "rt knob lines and the [rt_friction] draw line present"))
    vals = {}
    for _it, _tot, blk in blocks[:2]:                  # the first two blocks (the zero-filled count tags)
        vals.update(M.block_values(blk, M.CNT + "end/"))
    miss = [k for k in kinds if f"{M.CNT}end/{k}" not in vals]
    out.append(pf("PF12", not miss, f"site_cnt_end/<kind> missing for {miss}" if miss else
                  f"per-kind telemetry for all {len(kinds)} kinds"))
    return out


# ============================================================================== main
def _on_signal(signum, _frame):
    STOP_SIGNALS.append(signal.Signals(signum).name)


class Say:
    def __init__(self, name: str, log: Path | None):
        self.name, self.log = name, log

    def __call__(self, kind: str, msg: str, raw: bool = False) -> None:
        line = msg if raw else f"RT TRAIN {kind} {self.name} {msg} at={M.now_iso()}"
        print(line, flush=True)
        for p, lead in ((self.log, "\n"), (EVENTS, "")):
            if p is None:
                continue
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as fh:
                    fh.write(f"{lead}{line}\n")
            except OSError:
                pass


def main(argv=None) -> int:
    try:
        a = parse_args(argv)
    except Usage as exc:
        print(f"rt_train: {exc}", file=sys.stderr)
        return EXIT_USAGE
    dry = a.dry_run or a.preflight_only
    env, krabby, cprobs = M.compose_child_env(base_env(), a.env_file)
    unit = f"rt-train-{a.name}-{int(time.time())}"
    argv_t = M.trainer_argv(a.name, a.checkpoint, a.iterations, a.num_envs, a.seed)
    cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={MEMORY_MAX}",
           "-p", "MemorySwapMax=0", "timeout", "--signal=TERM", str(a.timeout_s), *argv_t]
    log = M.TRAIN_OUT / f"{a.name}.log"
    rec_path = RUNS / f"{a.name}.json"
    kinds = sorted(M.parse_mix(krabby.get(M.PLANNED_MIX_VAR, "")))
    if dry:
        res = preflight(a, env, krabby, cprobs, check_busy=False)
        for k in sorted(res, key=lambda x: int(x[2:])):
            r = res[k]
            print(f"{k} {'PASS' if r['ok'] else 'FAIL'} {r['detail']}")
        if a.dry_run:
            print("# composed child env (KRABBY_* + the added keys; the base is the user-service env minus "
                  "KRABBY_/TP_/RSD_/RTQ_/KIT_/DRS_/PYTHONPATH):")
            for k in M.ADDED_ENV:
                print(f"{k}={env[k]}")
            for k, v in sorted(krabby.items()):
                print(f"{k}={v}")
            print(f"# cwd {M.PARKOUR}")
            print("# command:", " ".join(cmd))
            print(f"# log {log} | record {rec_path} | expected final "
                  f"model_{(res['PF3']['data'].get('expected') or {}).get('final_iter')}.pt")
            print(f"# iteration-{M.EARLY_READ_AT} read: fall share per kind {kinds} vs iterations "
                  f"{M.EARLY_READ_REF[0]}-{M.EARLY_READ_REF[1]} (one RTQ NOTIFY line, report only)")
        unexpected = dry_failures(res)
        print(f"RT TRAIN DRY {a.name} preflight={'PASS' if all(r['ok'] for r in res.values()) else 'FAIL'}"
              + (f" unexpected={unexpected}" if unexpected else ""))
        return EXIT_PF if unexpected else EXIT_OK
    M.TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    lock = open(M.TRAIN_OUT / "rt_train.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("rt_train: another rt_train launcher holds the lock", file=sys.stderr)
        return EXIT_BUSY
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _on_signal)
    if a.wait_idle:
        while M.busy(exclude=(os.getpid(),)):
            if STOP_SIGNALS:
                return EXIT_TRAIN
            time.sleep(30)
    res = preflight(a, env, krabby, cprobs)
    say = Say(a.name, None)
    record = {"name": a.name, "created": M.now_iso(), "cli": vars(a), "trainer_argv": argv_t, "wrapper_argv": cmd,
              "unit": unit, "cwd": str(M.PARKOUR), "log": str(log), "krabby": krabby, "added_env": M.ADDED_ENV,
              "base_keys": sorted(env), "preflight": res}
    failed = [k for k, r in res.items() if not r["ok"]]
    for k in sorted(res, key=lambda x: int(x[2:])):
        say("PF", f"{k} {'PASS' if res[k]['ok'] else 'FAIL'} {res[k]['detail'][:400]}")
    if failed:
        record["status"] = "preflight_refused"
        M.write_json(rec_path, record)
        say("FAIL", f"pre-flight {failed} failed: nothing launched")
        return EXIT_GUARD if failed == ["PF7"] else EXIT_PF
    say = Say(a.name, log)
    M.write_json(rec_path, record)
    say("START", f"unit={unit} log={log}")
    with open(log, "ab") as fh:
        proc = subprocess.Popen(cmd, cwd=str(M.PARKOUR), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    return supervise(a, proc, log, say, res, record, rec_path, krabby, kinds)


def supervise(a, proc, log: Path, say, pre: dict, record: dict, rec_path: Path, krabby: dict, kinds) -> int:
    t0 = time.time()
    stop_file = RUNS / f"{a.name}.stop"
    it0 = int(pre["PF3"]["data"]["facts"]["iter"])
    pf8_done = inrun_done = early_done = False
    status = why = None

    def stop(reason):
        say("STOP", f"stopping the trainer: {reason}")
        record.setdefault("stops", []).append({"reason": reason, **stop_trainer(trainer_pids(a.name), say)})

    while proc.poll() is None:
        time.sleep(POLL_S)
        if STOP_SIGNALS or stop_file.exists():
            reason = f"signal {STOP_SIGNALS[0]}" if STOP_SIGNALS else f"stop file {stop_file}"
            stop(reason)
            status, why = "interrupted", reason
            break
        if not pf8_done:
            pids = trainer_pids(a.name)
            if pids:
                try:
                    probs = environ_problems(proc_environ(pids[0]), krabby)
                except OSError as exc:
                    probs = [f"cannot read /proc/{pids[0]}/environ: {exc}"]
                pf8_done = True
                record["preflight"]["PF8"] = pf("PF8", not probs, "; ".join(probs) or
                                                f"trainer pid {pids[0]} environ: PYTHONPATH + KRABBY_* as composed")
                say("PF", f"PF8 {'PASS' if not probs else 'FAIL'} {record['preflight']['PF8']['detail'][:400]}")
                if probs:
                    stop("PF8 failed")
                    status, why = "preflight_fail", "PF8"
                    break
            elif time.time() - t0 > PF8_WAIT_S:
                pf8_done = True
                record["preflight"]["PF8"] = pf("PF8", False, f"no trainer python found within {PF8_WAIT_S:.0f} s")
                say("PF", "PF8 FAIL no trainer pid")
                stop("PF8: no trainer pid")
                status, why = "preflight_fail", "PF8"
                break
        try:
            text = log.read_text(errors="replace")
        except OSError:
            text = ""
        tail = text[-20000:]
        if NAN_RE.search(tail) or "Traceback" in tail:
            stop("NaN or Traceback in the training log")
            status, why = "crashed", "NaN or Traceback in the log"
            break
        try:
            idle = time.time() - log.stat().st_mtime
        except OSError:
            idle = 0.0
        if idle > STALL_S:
            stop(f"no log progress for {STALL_S} s")
            status, why = "stall", "no log progress"
            break
        if time.time() - t0 > a.timeout_s + 2 * TERM_GRACE_S + 300:
            stop("alive past the wall bound")
            status, why = "timeout_hang", "alive past the wall bound"
            break
        if not inrun_done and len(M.ITER_RE.findall(text)) >= 2:
            inrun_done = True
            res = inrun_checks(text, pre, kinds)
            res.append(params_pf(a, krabby))
            record["preflight_inrun"] = res
            M.write_json(rec_path, record)
            for r in res:
                say("PF", f"{r['id']} {'PASS' if r['ok'] else 'FAIL'} {r['detail'][:400]}")
            bad = [r["id"] for r in res if not r["ok"]]
            if bad:
                stop(f"{','.join(bad)} failed in-run")
                status, why = "preflight_fail", ",".join(bad)
                break
        if not early_done:
            line = M.early_line(text, it0)
            if line:
                early_done = True
                record["early_read"] = line
                M.write_json(rec_path, record)
                say("NOTIFY", line, raw=True)
    rc = proc.wait()
    record.update(rc=rc, ended=M.now_iso(), wall_s=round(time.time() - t0, 1))
    run_dir = find_run_dir(a.name)
    record["run_dir"] = str(run_dir) if run_dir else None
    if status is None and rc != 0:
        status, why = ("timeout", f"timeout {a.timeout_s} s") if rc == 124 else ("crashed", f"rc {rc}")
    if status is None and not inrun_done:
        status, why = "preflight_fail", "the trainer ended before the in-run checks"
    if status is None:
        final = M.final_iteration(it0, a.iterations)
        ck = run_dir / f"model_{final}.pt" if run_dir else None
        ok = False
        if ck is not None and ck.is_file():
            try:
                f = checkpoint_facts(ck)
                ok = f["iter"] == final
                record["final"] = f
            except Exception as exc:  # noqa: BLE001
                record["final_error"] = str(exc)
        record["PF13"] = pf("PF13", ok, f"{ck} iter {final}" if ok else f"no model_{final}.pt with iter {final}")
        say("PF", f"PF13 {'PASS' if ok else 'FAIL'} {record['PF13']['detail']}")
        status = "ok" if ok else "final_fail"
    record["status"], record["why"] = status, why
    M.write_json(rec_path, record)
    if status == "ok":
        say("END", f"status=ok rc={rc} ckpt={record['final']['path']} sha={record['final']['sha256'][:8]}")
        return EXIT_OK
    say("FAIL", f"status={status} rc={rc} why={why} -- never auto-resumed; relaunch under a new name")
    return {"preflight_fail": EXIT_PF_INRUN, "final_fail": EXIT_FINAL}.get(status, EXIT_TRAIN)


if __name__ == "__main__":
    sys.exit(main())
