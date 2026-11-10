#!/usr/bin/env python3
"""Shared launcher helpers of the rod-speed DR campaign (CHARTER sections 2, 3.2, 3.4, 3.5) -- CPU only, never Kit.

Used by ``rsd_train.py`` (the teacher fine-tune launcher) and ``rsd_distill_check.py`` (the distill +10-iteration
checks). Nothing here launches Isaac: the helpers read ``ps`` / ``/proc``, the disk, the Kit DerivedDataCache size,
the git tree (through ``experiments/tools/run_candidates.py`` in a ``-I -B`` subprocess, the sweep's ``code_state``
pattern), checkpoints on CPU and the campaign's knob modules by path (``dr_knobs.py``, ``rod_speed_dr.py``: pure).

* :func:`isaac_patterns` -- the busy-check pattern pair read from W's ``tools/pipeline/distill.sh`` (one source).
* :func:`busy` -- every Isaac process but this one (the patterns match ``rsd_train.py`` itself: ``[^ ]*train.py``).
* :func:`guard` -- CHARTER 3.5: stop below :data:`FREE_MIN_GB` free, or with the DerivedDataCache above
  :data:`DDC_MAX_GB`.
* :func:`stop_trainer` -- SIGTERM the trainer python's own pid (never the scope: ``timeout`` would re-forward it),
  :data:`TERM_GRACE_S`, SIGTERM again, :data:`TERM_GRACE_S`, SIGKILL only as the last resort (recorded).
* :func:`predicted_dr` / :func:`dr_knob_line` -- the ``[DR]`` / ``[KNOB]`` lines an armed run must print, computed by the
  very modules the run imports (path-loaded from this worktree).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CD = HERE.parent
#: CD -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> the worktree root
W = CD.parents[5]
#: the main checkout (read-only data: Task 4 stamps / logs, p3b's run dir; the campaign's output tree lives here)
MAIN = Path("/home/nickmagus/krabby/krabby-research")
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"

PARKOUR = W / "parkour"
TRAIN = PARKOUR / "scripts" / "rsl_rl" / "train.py"
PKG = PARKOUR / "parkour_tasks" / "parkour_tasks"
OMNI = PKG / "crab_hex_omni_task"
TOOLS = OMNI / "experiments" / "tools"
DISTILL_SH = TOOLS / "pipeline" / "distill.sh"
RUN_PHASES_PY = TOOLS / "run_phases.py"
LIVE_WATCH_PY = TOOLS / "pipeline" / "live_watch.py"
RSD_PY = PKG / "crab_hex_forward_task" / "mdp" / "rod_speed_dr.py"
ACTIONS_PY = PKG / "crab_hex_forward_task" / "mdp" / "parkour_actions.py"
DR_KNOBS_PY = OMNI / "config" / "crab_hex" / "dr_knobs.py"
OMNI_KNOBS_PY = OMNI / "config" / "crab_hex" / "omni_knobs.py"
STRESS_KNOBS_PY = OMNI / "config" / "crab_hex" / "stress_knobs.py"

OUT_ROOT = MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr"
TRAIN_OUT = OUT_ROOT / "train"
DISTILL_OUT = OUT_ROOT / "distill"
EVENTS_LOG = TRAIN_OUT / "rsd_train_events.log"
RESULTS = CD / "results"
RUNS = RESULTS / "runs"
ENVS = CD / "envs"
LAUNCH_PIN = RESULTS / "launch.json"

#: p3b, the run T* came out of (main checkout, read-only)
F19_MAIN = MAIN / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "experiments" / \
    "2026-10-16_task4_pipeline" / "F19"
P3B_TAG = "seed19_X08_p3b_A15pB"
P3B_ORCH_LOG = F19_MAIN / "orchestrator.log"
P3B_ORCH_LINE = 30
P3B_TRAIN_LOG = F19_MAIN / "logs" / f"{P3B_TAG}_train.log"
P3B_RUN_DIR = MAIN / "parkour" / "logs" / "rsl_rl" / "crab_hex_omni_teacher" / f"2026-10-22_11-14-00_{P3B_TAG}"
P3B_PARAMS = P3B_RUN_DIR / "params"
#: Task 4's first-attempt-passed distill (the +10 reference run)
ST_A2_NAME = "T4F19_st_a2"
ST_A2_LOG = MAIN / "parkour" / "outputs" / "omni_plan" / "task4" / "distill" / f"{ST_A2_NAME}.log"
ST_A2_RUN_DIR = MAIN / "parkour" / "logs" / "rsl_rl" / "crab_hex_omni_student" / f"2026-10-22_23-41-17_{ST_A2_NAME}"

#: eval_sha must stay this (CHARTER section 2)
EVAL_SHA_PIN = "9fe964d7f46d1116d1d2983b0ece43f2430695cc35c9f33fd4cdc28f169ed5b8"
#: T* (CHARTER 3.2): sha -> the values its checkpoint must carry (a cross-check; PF3 always reads the file itself)
KNOWN_CHECKPOINTS = {
    "03b6009a6e952133f6c21c36c8d315b76f945d431615ef6d6798b7f4a5f59fc7":
        {"label": "T* f19_8_p3b_payload180/model_25992", "iter": 25992, "counter": 27300, "learning_rate": 1.5e-5,
         "tstar": True},
}

DDC = Path("/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache")
#: CHARTER 3.5: stop and notify below 30 GB free, or with the DerivedDataCache above 78 GB
FREE_MIN_GB = 30.0
DDC_MAX_GB = 78.0
#: run_phases.TERM_GRACE_S / pipeline.yaml live.term_grace_s
TERM_GRACE_S = 120.0
KILL_WAIT_S = 60.0

NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
#: the sweep's gate names (CHARTER section 2: no row of this campaign uses them)
FORBIDDEN_NAME_PREFIXES = ("DRS_", "DRST_")
RUN_DIR_RE = r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_%s$"
KNOB_LINE_RE = re.compile(r"^\[(KNOB|omni-knobs)\] ")
DR_LINE_RE = re.compile(r"^\[DR\] ")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")


# ============================================================================== small pure helpers
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_by_path(path: Path, name: str):
    """A module by path under a private name, no bytecode (the campaign-driver precedent)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = old
    return mod


def normalise_roots(text: str, w: Path | None = None, main: Path | None = None) -> str:
    """The worktree root, THEN the main root, replaced by ``<ROOT>`` (the worktree lives inside main)."""
    return text.replace(str(w or W), "<ROOT>").replace(str(main or MAIN), "<ROOT>")


def valid_name(name: str) -> str | None:
    """None when ``name`` is a usable run name, else why not."""
    if not NAME_RE.match(name or ""):
        return f"name {name!r} must be [A-Za-z0-9_]+ (distill.sh's rule)"
    if name.startswith(FORBIDDEN_NAME_PREFIXES):
        return f"name {name!r} uses a sweep prefix {FORBIDDEN_NAME_PREFIXES} (CHARTER section 2)"
    return None


def existing_runs(name: str, roots=None) -> list[str]:
    """Every ``<parkour>/logs/rsl_rl/<experiment>/<timestamp>_<name>`` under the worktree's and main's logs."""
    pat = re.compile(RUN_DIR_RE % re.escape(name))
    out = []
    for root in (roots or (W, MAIN)):
        base = Path(root) / "parkour" / "logs" / "rsl_rl"
        if not base.is_dir():
            continue
        for exp in base.iterdir():
            if exp.is_dir():
                out += [str(d) for d in exp.iterdir() if d.is_dir() and pat.match(d.name)]
    return sorted(out)


def find_run_dir(name: str, experiment: str, root: Path | None = None) -> Path | None:
    """The newest ``<root>/parkour/logs/rsl_rl/<experiment>/<timestamp>_<name>`` (None before the trainer made it)."""
    base = Path(root or W) / "parkour" / "logs" / "rsl_rl" / experiment
    pat = re.compile(RUN_DIR_RE % re.escape(name))
    found = sorted(d for d in base.glob(f"*_{name}") if d.is_dir() and pat.match(d.name)) if base.is_dir() else []
    return found[-1] if found else None


def read_env_lines(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """``([(KEY, VALUE)], problems)`` of an env file: every ``KRABBY_*`` line VERBATIM (an empty value is kept; nothing
    is stripped or unquoted), blank lines and ``#`` comments skipped; anything else, a repeated key or a ``$REPO`` value
    (no launcher resolves it) is a problem."""
    out, problems, seen = [], [], set()
    for i, raw in enumerate(Path(path).read_text().split("\n"), 1):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^(KRABBY_[A-Z0-9_]+)=(.*)$", line)
        if not m:
            problems.append(f"{path}:{i}: not a KRABBY_*=VALUE line: {line[:80]!r}")
            continue
        key, value = m.group(1), m.group(2)
        if key in seen:
            problems.append(f"{path}:{i}: {key} repeated")
        if "$REPO" in value:
            problems.append(f"{path}:{i}: {key} carries a $REPO token (no launcher resolves it)")
        seen.add(key)
        out.append((key, value))
    return out, problems


def knob_lines(text: str) -> list[str]:
    """The ``[KNOB]`` / ``[omni-knobs]`` lines of a log (roots normalised), in order."""
    return [normalise_roots(ln.rstrip("\r")) for ln in text.split("\n") if KNOB_LINE_RE.match(ln)]


def dr_lines(text: str) -> list[str]:
    return [ln.rstrip("\r") for ln in text.split("\n") if DR_LINE_RE.match(ln)]


def multiset_diff(got: list[str], want: list[str]) -> dict:
    """``{"missing": [...], "extra": [...]}`` of two line multisets (both empty = equal)."""
    from collections import Counter
    g, w = Counter(got), Counter(want)
    return {"missing": sorted((w - g).elements()), "extra": sorted((g - w).elements())}


def iteration_blocks(text: str) -> list[tuple[int, int, str]]:
    """``[(it, tot, block_text)]``: each ``Learning iteration it/tot`` title with the text up to the next title (the
    last block runs to the end of ``text``, so it may be incomplete)."""
    marks = [(m.start(), int(m.group(1)), int(m.group(2))) for m in ITER_RE.finditer(text)]
    out = []
    for j, (pos, it, tot) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(text)
        out.append((it, tot, text[pos:end]))
    return out


def block_values(block: str, prefix: str) -> dict:
    """``{key: float}`` of ``<key>: <number>`` lines in one iteration block whose key starts with ``prefix``."""
    out = {}
    for m in re.finditer(r"^\s*(%s[\w/]*):\s*(\S+)\s*$" % re.escape(prefix), block, re.M):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            out[m.group(1)] = float("nan")
    return out


# ============================================================================== knob modules (pure, by path)
def dr_knobs():
    return load_by_path(DR_KNOBS_PY, "rsd_launch_dr_knobs")


def rod_speed_dr():
    return load_by_path(RSD_PY, "rsd_launch_rod_speed_dr")


def stress_rod_speed_var() -> str:
    return load_by_path(STRESS_KNOBS_PY, "rsd_launch_stress_knobs").ROD_SPEED_VAR


def dr_knob_line(value: str) -> tuple[dict, str]:
    """``(spec, [KNOB] line)`` that ``dr_knobs.apply_dr_knobs`` writes / prints for ``KRABBY_DR_ROD_SPEED=value``."""
    mod = dr_knobs()
    cfg = SimpleNamespace(actions=SimpleNamespace(joint_pos=SimpleNamespace()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.apply_dr_knobs(cfg, {mod.ROD_SPEED_VAR: value}, exclusive_with=(stress_rod_speed_var(),))
    lines = [ln for ln in buf.getvalue().split("\n") if ln]
    if len(lines) != 1:
        raise RuntimeError(f"dr_knobs printed {len(lines)} lines for {value!r}: {lines}")
    return getattr(cfg.actions.joint_pos, mod.ROD_SPEED_DR_ATTR), lines[0]


def predicted_dr(spec: dict, num_envs: int, seed: int, step: int = 0) -> dict:
    """The first-reset draw a run must make: ``RodSpeedDR(spec, num_envs, cpu, seed).reset(all envs, step)``, i.e.
    seed ``rod_speed_dr_seed(seed)``, envs 0..N-1, ``s = lo_eff + (hi - lo_eff) u`` in float32 -- with the ``[DR]``
    line the module prints (its ``module=`` is this worktree's file, as in the run) and the draw's quantiles."""
    mod = rod_speed_dr()
    obj = mod.RodSpeedDR(dict(spec), int(num_envs), "cpu", int(seed))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        obj.reset(None, int(step))
    lines = [ln for ln in buf.getvalue().split("\n") if ln]
    s = obj.scale[:, 0].clone()
    import torch  # noqa: PLC0415 -- the Isaac venv's (rsd_train runs under it)

    q = torch.quantile(s.double(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    lo_eff, hi = obj.bounds(int(step))
    return {"line": lines[0] if lines else None, "lines": lines, "seed": obj.seed, "env_seed": int(seed),
            "num_envs": int(num_envs), "step": int(step), "lo_eff": lo_eff, "hi": hi,
            "min": float(s.min()), "mean": float(s.mean()), "max": float(s.max()),
            "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]),
            "sha16": mod._sha16(s), "module": str(RSD_PY)}


# ============================================================================== checkpoints
def checkpoint_facts(path: Path) -> dict:
    """CPU ``torch.load``: ``iter``, ``alg_state`` counter / learning_rate, the filename iteration and the sha."""
    import torch  # noqa: PLC0415

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    st = ck.get("alg_state") if isinstance(ck, dict) else None
    m = re.match(r"^model_(\d+)\.pt$", Path(path).name)
    return {"path": str(path), "sha256": sha256_file(path), "iter": ck.get("iter") if isinstance(ck, dict) else None,
            "file_iter": int(m.group(1)) if m else None,
            "counter": (st or {}).get("counter") if isinstance(st, dict) else None,
            "learning_rate": (st or {}).get("learning_rate") if isinstance(st, dict) else None,
            "keys": sorted(ck) if isinstance(ck, dict) else None}


# ============================================================================== the tree
CODE_STATE_SNIPPET = (
    "import json, subprocess, sys\n"
    "sys.dont_write_bytecode = True\n"
    "sys.path.insert(0, '.')\n"
    "import run_candidates as r\n"
    "st = subprocess.run(['git', '-C', str(r.REPO), 'status', '--porcelain', '--', *r.CODE_PATHSPECS, "
    "*r.EVAL_PATHSPECS], capture_output=True, text=True).stdout.split('\\n')\n"
    "dirty = [l[3:] for l in st if l.strip() and not r.code_drop(l[3:])]\n"
    "print(json.dumps({'code_sha': r.code_sha(), 'eval_sha': r.eval_sha(), 'dirty': dirty, 'repo': str(r.REPO)}))\n"
)


def code_state() -> dict:
    """``run_candidates`` code_sha / eval_sha / dirty code+eval paths of THIS worktree (a ``-I -B`` subprocess)."""
    out = subprocess.run([PY, "-I", "-B", "-c", CODE_STATE_SNIPPET], cwd=str(TOOLS), capture_output=True, text=True,
                         timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"code_state failed rc={out.returncode}: {out.stderr.strip()[-400:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def git_head(repo: Path | None = None) -> str | None:
    r = subprocess.run(["git", "-C", str(repo or W), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def launch_pin() -> dict | None:
    try:
        return json.loads(LAUNCH_PIN.read_text())
    except (OSError, ValueError):
        return None


# ============================================================================== busy / guard
def isaac_patterns(distill_sh: Path | None = None) -> tuple[str, str]:
    """``(ISAAC_PAT, NONKIT_PAT)`` exactly as W's ``distill.sh`` spells them (single-quoted ERE, Python-compatible)."""
    text = Path(distill_sh or DISTILL_SH).read_text()
    found = {}
    for var in ("ISAAC_PAT", "NONKIT_PAT"):
        m = re.search(r"^%s='([^']*)'\s*$" % var, text, re.M)
        if not m:
            raise RuntimeError(f"{distill_sh or DISTILL_SH}: no {var}='...' line")
        found[var] = m.group(1)
    return found["ISAAC_PAT"], found["NONKIT_PAT"]


def ps_pid_args() -> list[tuple[int, str]]:
    p = subprocess.run(["ps", "-ww", "-eo", "pid=,args="], capture_output=True, text=True)
    out = []
    for ln in p.stdout.splitlines():
        f = ln.strip().split(None, 1)
        if len(f) == 2 and f[0].isdigit():
            out.append((int(f[0]), f[1]))
    return out


def isaac_match(args: str, pats: tuple[str, str] | None = None) -> bool:
    isaac, nonkit = pats or isaac_patterns()
    return bool(re.search(isaac, args)) and not re.search(nonkit, args)


def busy(exclude=(), procs=None, pats=None) -> list[tuple[int, str]]:
    """distill.sh's ``busy``: every process line matching ISAAC_PAT and not NONKIT_PAT -- minus ``exclude`` (this
    launcher's own pid: the pattern's ``[^ ]*train.py`` matches ``rsd_train.py``)."""
    pats = pats or isaac_patterns()
    skip = set(int(x) for x in exclude)
    return [(pid, a) for pid, a in (procs if procs is not None else ps_pid_args())
            if pid not in skip and isaac_match(a, pats)]


def ddc_gb(path: Path | None = None) -> float:
    d = Path(path or DDC)
    if not d.is_dir():
        return 0.0
    r = subprocess.run(["du", "-s", "--block-size=1G", str(d)], capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0


def guard(free_paths=None, ddc_path: Path | None = None) -> dict:
    """CHARTER 3.5 free-space / DerivedDataCache guard. ``problems`` empty = go."""
    free = {}
    for p in (free_paths or (MAIN, W)):
        try:
            free[str(p)] = round(shutil.disk_usage(str(p)).free / 2 ** 30, 1)
        except OSError as exc:
            free[str(p)] = None
            free[f"{p}:error"] = str(exc)
    gb = ddc_gb(ddc_path)
    problems = [f"free disk {v:.0f} GB < {FREE_MIN_GB:g} GB at {k}" for k, v in free.items()
                if isinstance(v, float) and v < FREE_MIN_GB]
    problems += [f"cannot stat free space at {k}" for k, v in free.items() if v is None]
    if gb > DDC_MAX_GB:
        problems.append(f"Kit DerivedDataCache {gb:.0f} GB > {DDC_MAX_GB:g} GB (a prune is the user's call)")
    return {"free_gb": free, "ddc_gb": gb, "free_min_gb": FREE_MIN_GB, "ddc_max_gb": DDC_MAX_GB,
            "problems": problems}


# ============================================================================== the trainer's pid and its stop
def scope_pids(unit: str | None) -> set[int]:
    if not unit:
        return set()
    name = unit if unit.endswith(".scope") else unit + ".scope"
    r = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", name],
                       capture_output=True, text=True)
    cg = r.stdout.strip()
    if not cg:
        return set()
    try:
        return {int(x) for x in Path("/sys/fs/cgroup" + cg, "cgroup.procs").read_text().split()}
    except (OSError, ValueError):
        return set()


def run_name_match(args: str, name: str) -> bool:
    a = args.split()
    return any(a[i:i + 2] == ["--run_name", name] for i in range(len(a) - 1)) and any(
        x.endswith("/train.py") for x in a)


def trainer_pids(unit: str | None, name: str, procs=None, pats=None, exclude=()) -> list[int]:
    """The Isaac python(s) running ``train.py --run_name <name>``: in the scope's cgroup or by command line; never
    ``timeout``, ``systemd-run`` or a launcher (they do not match ISAAC_PAT's python prefix)."""
    pats = pats or isaac_patterns()
    scope = scope_pids(unit)
    skip = set(int(x) for x in exclude) | {os.getpid()}
    return sorted(pid for pid, a in (procs if procs is not None else ps_pid_args())
                  if pid not in skip and isaac_match(a, pats) and "train.py" in a
                  and (pid in scope or run_name_match(a, name)))


def pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[:1] not in (["Z"], ["X"])


def _wait(cond, seconds: float, step: float = 2.0, sleep=time.sleep, clock=time.time) -> bool:
    t0 = clock()
    while True:
        if cond():
            return True
        if clock() - t0 >= seconds:
            return False
        sleep(step)


def stop_trainer(pids_fn, *, grace: float = TERM_GRACE_S, kill_wait: float = KILL_WAIT_S, say=print,
                 alive=pid_alive, send=os.kill, sleep=time.sleep, clock=time.time) -> dict:
    """SIGTERM the trainer pid(s), ``grace`` s, SIGTERM again, ``grace`` s, SIGKILL only as the last resort.

    ``pids_fn()`` returns the live trainer pids. ``sigkill`` in the result means a Kit was killed mid-run: the caller
    records it and notifies (the DerivedDataCache it may have been writing is the 2026-08-13 / 09-18 OOM hazard)."""
    out = {"at": now_iso(), "pids": [], "terms": 0, "sigkill": False, "grace_s": grace}
    pids = list(pids_fn())
    if not pids:
        out["note"] = "no trainer python found (it had exited, or never started)"
        return out
    out["pids"] = pids
    t0 = clock()

    def gone():
        return not any(alive(p) for p in pids)

    for n in (1, 2):
        for p in pids:
            if alive(p):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    send(p, signal.SIGTERM)
        out["terms"] = n
        say(f"SIGTERM #{n} to trainer pid(s) {pids}")
        if _wait(gone, grace, sleep=sleep, clock=clock):
            out["exited_after_s"] = round(clock() - t0, 1)
            return out
    for p in pids:
        if alive(p):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                send(p, signal.SIGKILL)
    out["sigkill"] = True
    say(f"SIGKILL (last resort) to trainer pid(s) {pids}: open the DerivedDataCache watch before the next Kit start")
    out["exited"] = _wait(gone, kill_wait, sleep=sleep, clock=clock)
    out["exited_after_s"] = round(clock() - t0, 1)
    return out


# ============================================================================== the user-service environment
def _decode_c_escapes(s: str) -> str:
    """systemd's ``$'...'`` body (C escapes over ASCII) -> the value."""
    return s.encode("utf-8").decode("unicode_escape")


def systemd_user_env() -> dict:
    """``systemctl --user show-environment`` as a dict (JSON output; the text form's ``$'...'`` decoded as a
    fallback)."""
    r = subprocess.run(["systemctl", "--user", "show-environment", "-o", "json"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip().startswith("{"):
        return {str(k): str(v) for k, v in json.loads(r.stdout).items()}
    r = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, text=True, check=True)
    out = {}
    for ln in r.stdout.splitlines():
        k, sep, v = ln.partition("=")
        if not sep:
            continue
        if v.startswith("$'") and v.endswith("'"):
            v = _decode_c_escapes(v[2:-1])
        out[k] = v
    return out
