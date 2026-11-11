#!/usr/bin/env python3
"""Rough-terrain campaign RUN MACHINERY (PLAN steps 5, 6, 8, 9): the pure CPU half -- paths, env composition, the
build pin, the row plans the queue runs, the guards, the sentinels and the training-log telemetry parse. Nothing here
starts Kit; the queue (``rt_queue.py``) executes the plans, the launchers (``tools/pipeline/gate_row.sh``,
``rt_kit_run.sh``, ``rt_train.py``) compose and cap every Isaac process.

    rt_machinery.py plan --block smoke|baseline|train|gate [--new-ckpt ABS] [--thresholds JSON]   the step list
    rt_machinery.py make-envs [--check]          the smoke env files (envs/RT_smoke_*.env), derived, byte-checked
    rt_machinery.py train-env-check [--env-file ABS]        RT_train.env == TDR's train env + the planned changes
    rt_machinery.py audit --block baseline|gate [--new-sha SHA] [--thresholds JSON]   every planned row read OK?
    rt_machinery.py early-line LOG [--it0 N]     the iteration-1500 fall-share line from a training log (report)
    rt_machinery.py params-diff --name N [--smoke ...]   a run's params/{env,agent}.yaml vs TDR's: every allowed and
                                                 every disallowed difference (PF14; S.train50.params)
    rt_machinery.py gate-ident --new-ckpt ABS [--record JSON]   the gate's new checkpoint == the step-8 run's final model
    rt_machinery.py neutral-refs                 TDR's saved nominal L090 field_light es1 stamps (keyed by sha)

Block contents (``plan``):
* smoke (step 5): S.check (build == pin), S.envs (smoke env files), S.neutral (TDR on its saved nominal L090
  field_light es1 row through gate_row.sh, every new knob off) + S.neutral.cmp (bitwise, else the class rule);
  S.train_ref / S.train_rt (rt_kit_train_smoke.py on the knobs-off reference and on RT_train.env, both with the
  smoke-only KRABBY_EPISODE_S) + S.train.cmp; S.train50 (rt_train.py, 50 real iterations of the launcher path);
  S.train50.params (CPU: its run params vs TDR's, planned differences only);
  per kind x band S.eval.<row> (gate_row.sh, --episodes 4, es1) + S.probe.<row> (rt_kit_eval_probe.py: the
  realised eval tiles and mu_eff), then S.eval.cmp.
* baseline (step 6): B.check; TDR at es3 on the 21 kind rows; TDR's retention rows (rows_rt.json env seeds);
  B.audit (every row found OK by the readout's own keying) and B.read (readout.py baseline).
* train (step 8): T.check, T.envcheck, T.train (rt_train.py: resume TDR, 6000 iterations, RT_s19).
* gate (step 9): G.check (+ thresholds decision proceed), both teachers at the frozen gate env seeds on every gated
  kind's gate band and every solved kind's R band, the new teacher's retention rows; G.audit, G.read (readout.py
  gates).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/scripts
CAMP = HERE.parent
EXPS = CAMP.parent
OMNI = EXPS.parent                                        # crab_hex_omni_task
W = Path(__file__).resolve().parents[7]                   # this worktree's root (parkour/ + .git)
PARKOUR = W / "parkour"
TOOLS = EXPS / "tools"
PIPE = TOOLS / "pipeline"
GATE_ROW = PIPE / "gate_row.sh"
DISTILL_SH = PIPE / "distill.sh"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
MAIN = Path("/home/nickmagus/krabby/krabby-research")
OUT = Path(os.environ.get("RT_TP_OUT") or MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain")
SMOKE_OUT = OUT / "smoke"
LOGS = OUT / "logs"
TRAIN_OUT = OUT / "train"
RSD_LR = MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr" / "lr"
RESULTS = CAMP / "results"
RUNS = RESULTS / "runs"
PIN_PATH = RESULTS / "pin.json"
STOP_FILE = RESULTS / "STOP_QUEUE"
LOCK_PATH = CAMP / ".isaac.lock"
ENVS = CAMP / "envs"
RT_TRAIN_ENV = ENVS / "RT_train.env"
SMOKE_TRAIN_ENV = ENVS / "RT_smoke_train.env"
SMOKE_REF_ENV = ENVS / "RT_smoke_ref.env"
RSD_CAMP = EXPS / "2026-11-07_rod_speed_dr"
TDR_TRAIN_ENV = RSD_CAMP / "envs" / "RSD_TDR_s19.train.env"
CLASSFLIP = RSD_CAMP / "scripts" / "classflip_check.py"
THRESHOLDS = CAMP / "thresholds_rough_v1.json"
MANIFESTS = CAMP / "manifests"
RT_KIT_RUN = HERE / "rt_kit_run.sh"
KIT_TRAIN_SMOKE = HERE / "rt_kit_train_smoke.py"
KIT_EVAL_PROBE = HERE / "rt_kit_eval_probe.py"
SMOKE_CHECK = HERE / "rt_smoke_check.py"
RT_TRAIN = HERE / "rt_train.py"
BUILD_CHECK = HERE / "build_check.py"
READOUT = HERE / "readout.py"
MAKE_THRESHOLDS = HERE / "make_thresholds.py"

TDR_CK = OMNI / "policy" / "rsd_9_p3b_rodspeed_dr" / "model_28991.pt"
TDR_SHA = "03c3e4728ac2175b4fbff66bc3944134c28fcbec0f6d8cc4aee4e5f3640c8567"
TDR_ITER, TDR_COUNTER = 28991, 30450
TEACHER_TASK_TRAIN = "Isaac-Crab-Hex-Omni-Teacher-v0"
TRAIN_NAME = "RT_s19"
TRAIN_ITERS, TRAIN_ENVS, TRAIN_SEED = 6000, 256, 19
TRAIN_TIMEOUT_S = 18000                     # ~5 h, SIGTERM only (the run's planned 2.5 h, doubled)
SMOKE_TRAIN_ITERS = 50
SMOKE_EPISODE_S = "6"                       # KRABBY_EPISODE_S of the two Kit training smokes ONLY (never training)
SMOKE_NUM_ENVS = 400                        # 20 columns x 20 envs
EVAL_SMOKE_EPISODES = 4
ROW_TIMEOUT_S = 2700                        # gate_row.sh TP_ROW_TIMEOUT (a suite row takes ~450 s)
KIT_TIMEOUT_S = 2400
SPAWN_Z_M = 1.085                           # crab_hex_scene_cfg.py:167
SPAWN_TOL_M = 0.05
EARLY_S = 2.0
EARLY_READ_AT, EARLY_READ_REF, EARLY_READ_WIN = 1500, (300, 500), 200
#: guards before every Isaac step (distill.sh's DerivedDataCache cap; the disk floor is low because the stamps of
#: steps 6 + 9 alone are ~30 GB and the disk had 46 GB free on 2026-11-10)
DDC = Path("/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache")
DDC_MAX_GB = float(os.environ.get("RTQ_DDC_MAX_GB") or 80.0)     # 74 GB on 2026-11-10; rt_kit_run.sh caps 80 too
DDC_GROWTH_MAX_GB = 3.0                     # one step growing the cache more than this = the corruption signature
FREE_MIN_GB = float(os.environ.get("RTQ_FREE_MIN_GB") or 20.0)
#: the training run's added (non-KRABBY) env, exactly RT_train.env's non-KRABBY lines
PYTHONPATH_W = f"{W / 'parkour' / 'parkour_tasks'}:{W / 'parkour'}"
ADDED_ENV = {"OMNI_KIT_ACCEPT_EULA": "yes", "TERM": "xterm", "PYTHONUNBUFFERED": "1", "PYTHONPATH": PYTHONPATH_W}
BASE_DROP_PREFIXES = ("KRABBY_", "TP_", "RSD_", "RTQ_", "KIT_", "DRS_")
BASE_DROP_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
#: RT_train.env = TDR's training env with exactly these changes (PLAN Design / step 2; CHARTER 2026-11-10)
PLANNED_SET = {"KRABBY_FLAT_TERRAIN_GEOM": "site", "KRABBY_SITE_LADDER": "rt",
               "KRABBY_RT_FRICTION": "gravel:0.2:0.5", "KRABBY_DR_ROD_SPEED": "0.6:1.35"}
PLANNED_MIX_VAR = "KRABBY_SITE_MIX"
SENTINEL_RE = re.compile(r"^RTQ (?P<label>\S+) rc=(?P<rc>-?\d+)(?: (?P<note>.*))?$")
DONE_RE = re.compile(r"^RTQ_DONE (?P<block>\S+) fail=(?P<fail>\d+)(?: (?P<note>.*))?$")
NOTIFY_RE = re.compile(r"^RTQ NOTIFY ")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
BLOCKS = ("smoke", "baseline", "train", "gate")


# ============================================================================== small helpers
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def load_by_path(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def rt_spec():
    return load_by_path(HERE / "rt_spec.py", "rt_spec")


def readout():
    """readout.py under the campaign-unique module name make_thresholds.py uses."""
    return load_by_path(READOUT, "rt_readout")


def short_sha(sha: str) -> str:
    return str(sha)[:8]


# ============================================================================== env files and composition
def read_env_file(path) -> tuple[list[tuple[str, str]], list[str]]:
    """``([(KEY, VALUE)], problems)``: every ``KEY=VALUE`` line verbatim (empty values kept); blank / ``#`` lines
    skipped; a malformed line, a repeated key or a ``$`` in a value (no launcher expands it) is a problem."""
    out, problems, seen = [], [], set()
    for i, raw in enumerate(Path(path).read_text().split("\n"), 1):
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if not m:
            problems.append(f"{path}:{i}: not a KEY=VALUE line: {line[:80]!r}")
            continue
        key, value = m.group(1), m.group(2)
        if key in seen:
            problems.append(f"{path}:{i}: {key} repeated")
        if "$" in value:
            problems.append(f"{path}:{i}: {key} value carries a '$' (no launcher expands it)")
        seen.add(key)
        out.append((key, value))
    return out, problems


def krabby_of(path) -> dict:
    pairs, _ = read_env_file(path)
    return {k: v for k, v in pairs if k.startswith("KRABBY_")}


def compose_child_env(base: dict, env_file, level_file=None) -> tuple[dict, dict, list[str]]:
    """``(child env, its KRABBY_* dict, problems)``: ``base`` minus every KRABBY_* / TP_* / RSD_* / RTQ_* / KIT_* /
    DRS_* and PYTHONPATH, plus :data:`ADDED_ENV`, plus the env file's ``KRABBY_*`` lines verbatim and (optional) a
    level file's ``KRABBY_PAYLOAD_*`` lines (refused when the env file already sets one) -- the gate_row.sh /
    kit_run.sh contract in Python (rt_train.py's launch)."""
    problems = []
    env = {k: v for k, v in base.items() if not k.startswith(BASE_DROP_PREFIXES) and k not in BASE_DROP_KEYS}
    env.update(ADDED_ENV)
    pairs, probs = read_env_file(env_file)
    problems += probs
    krabby = {k: v for k, v in pairs if k.startswith("KRABBY_")}
    for k, v in pairs:
        if not k.startswith("KRABBY_") and k in ADDED_ENV and v != ADDED_ENV[k]:
            problems.append(f"{env_file}: {k}={v!r} differs from the launcher's {ADDED_ENV[k]!r}")
    if level_file:
        lv_pairs, lp = read_env_file(level_file)
        problems += lp
        for k, v in lv_pairs:
            if not k.startswith("KRABBY_PAYLOAD_"):
                continue
            if k in krabby:
                problems.append(f"level key {k} already set by {env_file}")
            krabby[k] = v
    env.update(krabby)
    return env, krabby, problems


def parse_mix(text: str) -> dict:
    out = {}
    for part in (p for p in str(text).split(",") if p.strip()):
        k, _, v = part.partition("=")
        out[k.strip()] = int(v)
    return out


def planned_train_env() -> dict:
    """The KRABBY_* set RT_train.env must hold: TDR's training env (RSD_TDR_s19.train.env) with exactly the planned
    changes (geometry site, ladder rt, the 20-column mix, the friction contract, DR without its ramp)."""
    want = krabby_of(TDR_TRAIN_ENV)
    want.update(PLANNED_SET)
    spec = rt_spec()
    want[PLANNED_MIX_VAR] = ",".join(f"{k}={v}" for k, v in spec.TRAIN_MIX.items())
    return dict(sorted(want.items()))


def train_env_problems(env_file: Path = RT_TRAIN_ENV) -> list[str]:
    """RT_train.env against :func:`planned_train_env` (keys, values; the mix compared as a mapping) and its
    non-KRABBY lines against :data:`ADDED_ENV` (PYTHONPATH = this worktree)."""
    pairs, problems = read_env_file(env_file)
    have = {k: v for k, v in pairs if k.startswith("KRABBY_")}
    other = {k: v for k, v in pairs if not k.startswith("KRABBY_")}
    want = planned_train_env()
    missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    if missing or extra:
        problems.append(f"key set != the planned set: missing {missing} extra {extra}")
    for k in sorted(set(want) & set(have)):
        if k == PLANNED_MIX_VAR:
            if parse_mix(have[k]) != parse_mix(want[k]) or sum(parse_mix(have[k]).values()) != 20:
                problems.append(f"{k}={have[k]} != the planned mix {want[k]}")
        elif have[k] != want[k]:
            problems.append(f"{k}={have[k]!r} != planned {want[k]!r}")
    if other != ADDED_ENV:
        problems.append(f"non-KRABBY lines {other} != {ADDED_ENV}")
    return problems


# ============================================================================== run params vs TDR's (PF14)
#: TDR's own run params (the p3b EXTEND run the fine-tune resumes; its recorded env.yaml / agent.yaml)
TDR_PARAMS = (MAIN / "parkour" / "logs" / "rsl_rl" / "crab_hex_omni_teacher" / "2026-11-07_19-49-46_RSD_TDR_s19"
              / "params")
ABSENT = "<absent>"
_ROOT_RE = re.compile(re.escape(str(MAIN)) + r"(?:/\.claude/worktrees/[^/\s'\"]+)?")


def normalise_roots(text: str) -> str:
    """Every ``<main>`` / ``<main>/.claude/worktrees/<name>`` prefix -> ``<ROOT>`` (a worktree lives inside main)."""
    return _ROOT_RE.sub("<ROOT>", text)


def find_run_dir(name: str) -> Path | None:
    """This worktree's ``logs/rsl_rl/crab_hex_omni_teacher/<timestamp>_<name>`` (the newest), or None."""
    base = PARKOUR / "logs" / "rsl_rl" / "crab_hex_omni_teacher"
    pat = re.compile(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_%s$" % re.escape(name))
    hits = sorted(d for d in base.glob(f"*_{name}") if d.is_dir() and pat.match(d.name)) if base.is_dir() else []
    return hits[-1] if hits else None


def load_params_yaml(path) -> dict:
    """A run's params/*.yaml as plain data: ``!!python/tuple`` -> tuple, every other ``!!python/...`` node (object,
    object/apply, name) -> a ``{"__py__": tag, ...}`` mapping (nothing is imported or executed)."""
    import yaml  # noqa: PLC0415

    class _Loader(yaml.SafeLoader):
        pass

    def generic(loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return {"__py__": suffix, "value": loader.construct_scalar(node)}
        if isinstance(node, yaml.SequenceNode):
            return {"__py__": suffix, "seq": loader.construct_sequence(node, deep=True)}
        return {"__py__": suffix, **loader.construct_mapping(node, deep=True)}

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple",
                            lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", generic)
    return yaml.load(Path(path).read_text(), Loader=_Loader) or {}


def flatten_params(obj, prefix: str = "") -> dict:
    """``{dotted.path: leaf}``: mappings recurse by key, lists of mappings by ``[i]``, scalar lists / tuples are one
    leaf (a list); strings have their roots normalised."""
    out = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out[prefix] = {}
        for k, v in obj.items():
            out.update(flatten_params(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)) and any(isinstance(x, (dict, list, tuple)) for x in obj):
        out[f"{prefix}.__len__"] = len(obj)
        for i, v in enumerate(obj):
            out.update(flatten_params(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = [normalise_roots(x) if isinstance(x, str) else x for x in obj]
    else:
        out[prefix] = normalise_roots(obj) if isinstance(obj, str) else obj
    return out


def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eq(x, want) -> bool:
    return (abs(float(x) - float(want)) <= 1e-9) if _isnum(x) and _isnum(want) else x == want


def _must(want, what):
    """A rule check: the new value must equal ``want``."""
    def check(_ref, new, _ctx):
        return None if _eq(new, want) else f"{what}: new {new!r} != expected {want!r}"
    return check


def params_allow_rules(krabby: dict, expect: dict, smoke: bool) -> dict:
    """``{"env.yaml": [rule...], "agent.yaml": [rule...]}``, each rule ``(regex, why, check)``: the differences the
    PLANNED changes explain, derived from the run's own env file (``krabby`` = its KRABBY_* dict) and the launcher's
    expectations (``expect`` = name, iterations, seed, checkpoint). First matching rule wins; a rule's ``check``
    (new value against the plan) returns None or a refusal. Anything unmatched is a DISALLOWED difference."""
    env, agent = [], []
    site = krabby.get("KRABBY_FLAT_TERRAIN_GEOM") == "site"
    if site:
        env.append((r"(scene\.terrain|sim)\.physics_material\.friction_combine_mode",
                    "friction contract: ground friction combine mode", _must("multiply", "combine mode"))
                   if krabby.get("KRABBY_RT_FRICTION") else None)
        env.append((r"(scene\.terrain|sim)\.physics_material\.(static_friction|dynamic_friction)",
                    "friction contract: ground friction 1.0 (multiply)", _must(1.0, "ground friction"))
                   if krabby.get("KRABBY_RT_FRICTION") else None)
        env.append((r"scene\.terrain\.(?!physics_material(\.|$)).*",
                    "terrain generator / site sub-terrain keys (KRABBY_FLAT_TERRAIN_GEOM=site, SITE_LADDER, SITE_MIX)",
                    None))
        env.append((r"parkours\.base_parkour\.(clear_radius_m|ring_outer_radius_m)",
                    "parkour term radii synced to the site terrain (omni_terrain_knobs)", None))
    if krabby.get("KRABBY_RT_FRICTION"):
        lo, hi = (float(x) for x in krabby["KRABBY_RT_FRICTION"].split(":")[1:3])
        env.append((r"events\.physics_material\.func", "friction contract: the startup material event replaced",
                    lambda _r, new, _c: None if isinstance(new, str) and new.endswith("rt_friction_contract")
                    else f"material event func {new!r} is not rt_friction_contract"))
        env.append((r"events\.physics_material\.params\.gravel_range", "friction contract: gravel mu range",
                    _must([lo, hi], "gravel_range")))
        env.append((r"events\.physics_material\.params\.gravel_sub_terrains(\[\d+\]|\.__len__)?",
                    "friction contract: gravel sub-terrain keys", None))
        env.append((r"rt_friction(\..*)?", "friction contract: cfg.rt_friction (the privileged friction input)", None))
    dr = krabby.get("KRABBY_DR_ROD_SPEED")
    if dr:
        parts = dr.split(":")
        env.append((r"actions\.joint_pos\.rod_speed_dr\.ramp_steps", "rod-speed DR ramp (TDR 12000 steps -> the knob's)",
                    _must(int(parts[2]) if len(parts) > 2 else 0, "ramp_steps")))
    env.append((r"seed", "run seed", _must(expect["seed"], "seed")))
    if smoke and krabby.get("KRABBY_EPISODE_S"):
        env.append((r"episode_length_s", "smoke-only episode length", _must(float(krabby["KRABBY_EPISODE_S"]),
                                                                           "episode_length_s")))
    agent += [(r"run_name", "run name", _must(expect["name"], "run_name")),
              (r"load_checkpoint", "resume checkpoint path",
               _must(normalise_roots(str(expect["checkpoint"])), "load_checkpoint")),
              (r"max_iterations", "iteration count", _must(expect["iterations"], "max_iterations")),
              (r"seed", "run seed", _must(expect["seed"], "seed"))]
    return {"env.yaml": [r for r in env if r], "agent.yaml": agent}


def params_diff(new: dict, ref: dict, rules: list, expect=None) -> dict:
    """The new run's flattened params against the reference's: ``{"allowed": [...], "disallowed": [...],
    "n_compared": n}``; a difference is ``{"path", "ref", "new", "why"}``."""
    fn, fr = flatten_params(new), flatten_params(ref)
    allowed, bad = [], []
    for path in sorted(set(fn) | set(fr)):
        a, b = fr.get(path, ABSENT), fn.get(path, ABSENT)
        if a == b or (_isnum(a) and _isnum(b) and _eq(a, b)):
            continue
        rec = {"path": path, "ref": a, "new": b}
        rule = next((r for r in rules if re.fullmatch(r[0], path)), None)
        if rule is None:
            rec["why"] = "not in the allow-list (an unplanned difference)"
            bad.append(rec)
            continue
        refuse = rule[2](a, b, expect) if rule[2] else None
        rec["why"] = refuse or rule[1]
        (bad if refuse else allowed).append(rec)
    return {"allowed": allowed, "disallowed": bad, "n_compared": len(set(fn) | set(fr))}


def params_check(new_dir: Path, ref_dir: Path, expect: dict, krabby: dict, smoke: bool = False) -> dict:
    """PF14: a run's ``params/env.yaml`` + ``agent.yaml`` against TDR's (roots normalised), every difference outside
    :func:`params_allow_rules` refused. ``{"ok", "files": {name: params_diff result | {"error"}}}``."""
    rules = params_allow_rules(krabby, expect, smoke)
    out = {"ok": True, "files": {}}
    for fname in ("env.yaml", "agent.yaml"):
        pn, pr = Path(new_dir) / fname, Path(ref_dir) / fname
        missing = [str(p) for p in (pn, pr) if not p.is_file()]
        if missing:
            out["files"][fname] = {"error": f"missing {missing}", "allowed": [], "disallowed": []}
            out["ok"] = False
            continue
        try:
            res = params_diff(load_params_yaml(pn), load_params_yaml(pr), rules[fname], expect)
        except Exception as exc:  # noqa: BLE001
            out["files"][fname] = {"error": f"{type(exc).__name__}: {exc}", "allowed": [], "disallowed": []}
            out["ok"] = False
            continue
        out["files"][fname] = res
        out["ok"] = out["ok"] and not res["disallowed"]
    return out


def _short(v, n: int = 140) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def smoke_env_texts() -> dict:
    """``{path: text}`` of the two Kit training-smoke env files: RT_train.env's KRABBY_* (resp. the knobs-off
    reference: RT_train.env with geometry field and the three rt keys removed, which IS TDR's training env without
    the DR ramp -- asserted) plus the smoke-only ``KRABBY_EPISODE_S``. Never a training env."""
    rt = krabby_of(RT_TRAIN_ENV)
    ref = dict(rt)
    ref["KRABBY_FLAT_TERRAIN_GEOM"] = "field"
    for k in ("KRABBY_SITE_LADDER", PLANNED_MIX_VAR, "KRABBY_RT_FRICTION"):
        ref.pop(k, None)
    tdr = krabby_of(TDR_TRAIN_ENV)
    tdr["KRABBY_DR_ROD_SPEED"] = PLANNED_SET["KRABBY_DR_ROD_SPEED"]
    if ref != tdr:
        diff = sorted(k for k in set(ref) | set(tdr) if ref.get(k) != tdr.get(k))
        raise RuntimeError(f"the knobs-off reference is not TDR's training env (no ramp): differ {diff}")
    out = {}
    for path, d, what in ((SMOKE_TRAIN_ENV, rt, "RT_train.env"),
                          (SMOKE_REF_ENV, ref, "the knobs-off reference (TDR's training env, DR 0.6:1.35 no ramp)")):
        d = dict(d)
        if "KRABBY_EPISODE_S" in d:
            raise RuntimeError("KRABBY_EPISODE_S already in the training env")
        d["KRABBY_EPISODE_S"] = SMOKE_EPISODE_S
        head = (f"# GENERATED by rt_machinery.py make-envs -- do not edit. PLAN step 5 Kit training smoke env:\n"
                f"# {what} + KRABBY_EPISODE_S={SMOKE_EPISODE_S} (smoke-only short episodes, so every column resets;\n"
                f"# never used for training). Launchers read ^KRABBY_ lines only.\n")
        out[path] = head + "".join(f"{k}={v}\n" for k, v in sorted(d.items()))
    return out


def make_envs(check: bool) -> list[str]:
    bad = []
    for path, text in smoke_env_texts().items():
        if check:
            if not path.is_file() or path.read_text() != text:
                bad.append(f"{path.name} differs from its derivation (run make-envs)")
        else:
            path.write_text(text)
    return bad


# ============================================================================== build and pin
def _run_candidates():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import run_candidates as r  # noqa: PLC0415

    return r


def live_build() -> dict:
    """code_sha / eval_sha exactly as gate_row.sh stamps them (run_candidates' own functions), the dirty code + eval
    paths (gate_row.sh's dirty rule), HEAD."""
    r = _run_candidates()
    st = subprocess.run(["git", "-C", str(r.REPO), "status", "--porcelain", "--", *r.CODE_PATHSPECS,
                         *r.EVAL_PATHSPECS], capture_output=True, text=True).stdout.split("\n")
    dirty = sorted(ln[3:] for ln in st if ln.strip() and not r.code_drop(ln[3:]))
    head = subprocess.run(["git", "-C", str(r.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"code_sha": r.code_sha(), "eval_sha": r.eval_sha(), "dirty": dirty, "head": head, "repo": str(r.REPO)}


#: campaign files whose bytes define the rows (frozen with the pin) and the run machinery (reported when changed)
FROZEN_GLOBS = ("manifests/*.json", "manifests/*.yaml", "manifests/envs/*.env", "envs/*.env")
REPORTED_GLOBS = ("scripts/*.py", "scripts/*.sh")


def campaign_files(globs) -> dict:
    out = {}
    for g in globs:
        for p in sorted(CAMP.glob(g)):
            if p.is_file():
                out[str(p.relative_to(CAMP))] = sha256_file(p)
    return out


def pin_doc(live: dict) -> dict:
    return {"schema": "rt_pin_v1", "code_sha": live["code_sha"], "eval_sha": live["eval_sha"], "head": live["head"],
            "repo": live["repo"], "pinned_at": now_iso(), "frozen_files": campaign_files(FROZEN_GLOBS),
            "reported_files": campaign_files(REPORTED_GLOBS),
            "rule": "code_sha / eval_sha frozen from step 5 through step 9 (PLAN step 5); a stage refuses a live "
                    "build that differs, a dirty code/eval tree, or a changed frozen campaign file"}


def pin_problems(pin: dict | None, live: dict, frozen_now: dict, reported_now: dict) -> tuple[list, list]:
    """``(refusals, warnings)`` of the live build against the pin."""
    if pin is None:
        bad = [f"no pin {PIN_PATH} (write it once: build_check.py pin)"]
        if live.get("dirty"):
            bad.append(f"dirty code/eval paths {live['dirty'][:6]}{' ...' if len(live['dirty']) > 6 else ''}")
        return bad, []
    bad, warn = [], []
    if pin.get("schema") != "rt_pin_v1":
        bad.append(f"pin schema {pin.get('schema')!r}")
    for k in ("code_sha", "eval_sha"):
        if live.get(k) != pin.get(k):
            bad.append(f"{k} {str(live.get(k))[:12]} != pin {str(pin.get(k))[:12]}")
    if live.get("dirty"):
        bad.append(f"dirty code/eval paths {live['dirty'][:6]}{' ...' if len(live['dirty']) > 6 else ''}")
    if live.get("repo") and pin.get("repo") and live["repo"] != pin["repo"]:
        bad.append(f"repo {live['repo']} != pin {pin['repo']}")
    fz = pin.get("frozen_files") or {}
    moved = sorted(k for k in set(fz) | set(frozen_now) if fz.get(k) != frozen_now.get(k))
    if moved:
        bad.append(f"frozen campaign files changed since the pin: {moved}")
    rp = pin.get("reported_files") or {}
    moved_r = sorted(k for k in set(rp) | set(reported_now) if rp.get(k) != reported_now.get(k))
    if moved_r:
        warn.append(f"run-machinery / read-out scripts changed since the pin (report): {moved_r}")
    if live.get("head") != pin.get("head"):
        warn.append(f"HEAD {str(live.get('head'))[:10]} != pin {str(pin.get('head'))[:10]} (records may move HEAD)")
    return bad, warn


def read_pin(path: Path = PIN_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ============================================================================== stamps
def slot_dir(tp_out, name: str, scenario: str, es: int) -> Path:
    return Path(tp_out) / "lr" / name / scenario / f"seed{int(es):03d}"


def stamp_build(stamp) -> dict | None:
    """A stamp's ``code_state.json`` (gate_row.sh: code_sha, eval_sha, dirty as the string "True"/"False")."""
    try:
        return json.loads((Path(stamp) / "code_state.json").read_text())
    except (OSError, ValueError):
        return None


def build_mismatch(cs: dict | None, pin: dict | None) -> list[str]:
    """Why a recorded build (a stamp's code_state.json, or a probe JSON's ``build``) is NOT the pinned one; [] when
    it is: code_sha and eval_sha equal to the pin's and ``str(dirty) == "False"``."""
    if pin is None:
        return [f"no pin {PIN_PATH}"]
    if not cs:
        return ["no recorded build (code_state.json)"]
    bad = [f"{k} {str(cs.get(k))[:12]} != pin {str(pin.get(k))[:12]}" for k in ("code_sha", "eval_sha")
           if cs.get(k) != pin.get(k)]
    if str(cs.get("dirty")) != "False":
        bad.append(f"dirty {cs.get('dirty')!r}")
    return bad


def complete_stamps(slot: Path, pin: dict | None = None) -> list[Path]:
    """Stamps of a slot that gate_row.sh completed (``tp_row.json`` written last, rc 0), oldest first. With ``pin``
    (results/pin.json's dict) only the stamps recorded at the pinned build (:func:`build_mismatch` empty)."""
    out = []
    if Path(slot).is_dir():
        for d in sorted(p for p in Path(slot).iterdir() if p.is_dir()):
            try:
                tp = json.loads((d / "tp_row.json").read_text())
            except (OSError, ValueError):
                continue
            if int(tp.get("rc", -1)) != 0:
                continue
            if pin is not None and build_mismatch(stamp_build(d), pin):
                continue
            out.append(d)
    return out


def neutral_refs(lr: Path = RSD_LR) -> list[Path]:
    """TDR's saved nominal L090 field_light es1 stamps (keyed by checkpoint sha256 and the composed env, never by
    name): the stamps of rod_speed_dr/lr whose run_meta sha is TDR's, scenario field_light__omni, env seed 1, and
    whose KRABBY env equals this worktree's T00_nominal_L090_A env file + L090 level file."""
    want = dict(krabby_of(RSD_CAMP / "envs" / "T00_nominal_L090_A__field_light.env"))
    want.update({k: v for k, v in krabby_of(EXPS / "2026-10-16_task4_pipeline" / "F19" / "envs" / "levels" /
                                            "L090.env").items() if k.startswith("KRABBY_PAYLOAD_")})
    out = []
    if not Path(lr).is_dir():
        return out
    for meta in sorted(Path(lr).glob("*/field_light__omni/seed001/*/run_meta.json")):
        st = meta.parent
        if not (st / "tp_row.json").is_file():
            continue
        try:
            m = json.loads(meta.read_text())
        except ValueError:
            continue
        if m.get("checkpoint_sha256") != TDR_SHA or m.get("scenario_id") != "field_light__omni":
            continue
        if dict(m.get("env_vars") or {}) == want:
            out.append(st)
    return out


# ============================================================================== the plans
@dataclasses.dataclass
class Step:
    """One queue step. ``how``: cpu | gate | kit | train. ``argv`` may hold ``@stamp:<label>`` placeholders (the
    runner substitutes the newest complete stamp of that gate step's slot)."""

    label: str
    how: str
    argv: list
    env: dict = dataclasses.field(default_factory=dict)
    slot: tuple | None = None                 # gate rows: (tp_out, name, scenario, es)
    stop_on_fail: bool = False
    dry_exec: bool = False                    # cpu steps that also run in a dry pass (read-only checks)
    note: str = ""

    @property
    def isaac(self) -> bool:
        return self.how in ("gate", "kit", "train")


def _cpu(label, script, *args, stop=False, dry=False, note="") -> Step:
    return Step(label, "cpu", [PY, "-I", "-B", str(script), *map(str, args)], stop_on_fail=stop, dry_exec=dry,
                note=note)


def row_name(prefix: str, sha: str, rid: str) -> str:
    name = f"{prefix}_{short_sha(sha)}_{rid.replace('__omni', '')}"
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise ValueError(f"bad row name {name!r}")
    return name


def gate_step(label, *, name, ck, row: dict, es: int, tp_out, extra=()) -> Step:
    """A gate_row.sh launch of a rows_rt.json row (or a dict of the same shape with absolute paths)."""
    spec = rt_spec()

    def ab(p):
        return str(p if os.path.isabs(str(p)) else spec.OMNI / p)

    env = {"TP_OUT": str(tp_out), "TP_ROW_TIMEOUT": str(ROW_TIMEOUT_S)}
    if row.get("level_file"):
        env["TP_LEVEL_FILE"] = ab(row["level_file"])
    argv = ["bash", str(GATE_ROW), name, str(ck), ab(row["env_file"]), ab(row["manifest"]), row["scenario"],
            str(int(es)), *row["args"], *map(str, extra)]
    return Step(label, "gate", argv, env=env, slot=(str(tp_out), name, row["scenario"], int(es)))


def kit_step(label, env_file, script, *args, level_file=None) -> Step:
    env = {"KIT_TIMEOUT": str(KIT_TIMEOUT_S)}
    if level_file:
        env["KIT_LEVEL_FILE"] = str(level_file)
    return Step(label, "kit", ["bash", str(RT_KIT_RUN), str(env_file), str(script), *map(str, args)], env=env)


def check_steps(prefix: str) -> list:
    return [_cpu(f"{prefix}.check", BUILD_CHECK, "check", stop=True, dry=True,
                 note="the live build == results/pin.json, clean code/eval tree")]


def neutral_row() -> dict:
    """TDR's nominal L090 field_light es1 row (the rod-speed G1 cell T00_nominal_L090_A), this worktree's files."""
    spec = rt_spec()
    return {"id": "neutral_L090_field_light", "env_file": "experiments/2026-11-07_rod_speed_dr/envs/"
            "T00_nominal_L090_A__field_light.env", "level_file": "experiments/2026-10-16_task4_pipeline/F19/envs/levels/"
            "L090.env", "manifest": "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml",
            "scenario": "field_light__omni", "args": list(spec.TEACHER_ARGS)}


def plan_smoke(train50_name: str = "RT5_smoke50") -> list:
    spec = rt_spec()
    rows = spec.load_rows()
    res5 = RESULTS / "step5"
    steps = check_steps("S")
    steps.append(_cpu("S.envs", HERE / "rt_machinery.py", "make-envs", "--check", stop=True, dry=True,
                      note="the smoke env files match their derivation from RT_train.env / TDR's train env"))
    steps.append(_cpu("S.trainenv", HERE / "rt_machinery.py", "train-env-check", stop=True, dry=True))
    nr = neutral_row()
    steps.append(gate_step("S.neutral", name=row_name("RT5", TDR_SHA, nr["id"]), ck=TDR_CK, row=nr, es=1,
                           tp_out=SMOKE_OUT))
    refs = []
    for r in neutral_refs():
        refs += ["--ref", str(r)]
    steps.append(_cpu("S.neutral.cmp", SMOKE_CHECK, "neutrality", "--new", "@stamp:S.neutral", *refs,
                      "--out", res5 / "neutrality.json"))
    steps.append(kit_step("S.train_ref", SMOKE_REF_ENV, KIT_TRAIN_SMOKE, "--headless", "--mode", "ref",
                          "--num-envs", SMOKE_NUM_ENVS, "--seed", TRAIN_SEED, "--checkpoint", TDR_CK,
                          "--out", res5 / "train_smoke_ref.json"))
    steps.append(kit_step("S.train_rt", SMOKE_TRAIN_ENV, KIT_TRAIN_SMOKE, "--headless", "--mode", "rt",
                          "--num-envs", SMOKE_NUM_ENVS, "--seed", TRAIN_SEED, "--checkpoint", TDR_CK,
                          "--out", res5 / "train_smoke_rt.json"))
    steps.append(_cpu("S.train.cmp", SMOKE_CHECK, "train", "--rt", res5 / "train_smoke_rt.json",
                      "--ref", res5 / "train_smoke_ref.json", "--out", res5 / "train_smoke.json"))
    steps.append(Step("S.train50", "train", [PY, "-I", "-B", str(RT_TRAIN), "--name", train50_name,
                                             "--iterations", str(SMOKE_TRAIN_ITERS), "--timeout-s", "3600",
                                             "--wait-idle", "--smoke"],
                      note="50 real iterations through the step-8 launcher (pre-flight, environ, in-run checks)"))
    steps.append(_cpu("S.train50.params", HERE / "rt_machinery.py", "params-diff", "--name", train50_name, "--smoke",
                      "--iterations", SMOKE_TRAIN_ITERS, "--seed", TRAIN_SEED, "--checkpoint", TDR_CK,
                      note="the smoke run's params/env.yaml + agent.yaml == TDR's except the planned changes (PF14)"))
    for r in rows["kind_rows"]:
        rid = r["id"]
        steps.append(gate_step(f"S.eval.{rid}", name=row_name("RT5", TDR_SHA, rid), ck=TDR_CK, row=r, es=1,
                               tp_out=SMOKE_OUT, extra=("--episodes", EVAL_SMOKE_EPISODES)))
        steps.append(kit_step(f"S.probe.{rid}", spec.OMNI / r["env_file"], KIT_EVAL_PROBE, "--headless",
                              "--manifest", spec.OMNI / r["manifest"], "--scenario", r["scenario"],
                              "--env-seed", 1, "--episodes", EVAL_SMOKE_EPISODES, "--checkpoint", TDR_CK,
                              "--out", res5 / "probe" / f"{rid}.json"))
    steps.append(_cpu("S.eval.cmp", SMOKE_CHECK, "eval", "--rows-root", SMOKE_OUT / "lr",
                      "--probe-dir", res5 / "probe", "--out", res5 / "eval_smoke.json"))
    return steps


def baseline_rows() -> list:
    """``[(label, rid, es)]`` of step 6: TDR at es3 on the 21 kind rows, then TDR's retention rows."""
    rows = rt_spec().load_rows()
    out = [(f"B.kind.{r['id']}.es3", r["id"], 3) for r in rows["kind_rows"]]
    for r in rows["retention_rows"]:
        for es in r["env_seeds"]:
            out.append((f"B.ret.{r['id']}.es{es}", r["id"], int(es)))
    return out


def plan_baseline() -> list:
    rows = rt_spec().load_rows()
    by_id = {r["id"]: r for r in rows["kind_rows"] + rows["retention_rows"]}
    steps = check_steps("B")
    for label, rid, es in baseline_rows():
        steps.append(gate_step(label, name=row_name("RT6", TDR_SHA, rid), ck=TDR_CK, row=by_id[rid], es=es, tp_out=OUT))
    steps.append(_cpu("B.audit", HERE / "rt_machinery.py", "audit", "--block", "baseline", dry=True,
                      note="every step-6 row found OK by readout.find (sha256 + composed env + args)"))
    steps.append(_cpu("B.read", READOUT, "baseline", "--tdr", TDR_SHA, "--out", RESULTS / "step6" / "baseline.json"))
    return steps


def plan_train(name: str = TRAIN_NAME) -> list:
    steps = check_steps("T")
    steps.append(_cpu("T.envcheck", HERE / "rt_machinery.py", "train-env-check", stop=True, dry=True))
    steps.append(Step("T.train", "train", [PY, "-I", "-B", str(RT_TRAIN), "--name", name, "--wait-idle"],
                      stop_on_fail=True, note="resume TDR, 6000 iterations, RT_train.env, MemoryMax 48G, SIGTERM-only"))
    return steps


def load_thresholds(path: Path = THRESHOLDS) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "rt_thresholds_v1":
        raise ValueError(f"{path}: schema {doc.get('schema')!r}")
    return doc


def gate_rows(thr: dict) -> list:
    """``[(rid, ess)]``: every gated kind's gate band and every solved kind's R band, at the frozen gate seeds."""
    spec = rt_spec()
    ess = tuple(int(e) for e in (thr.get("rule") or {}).get("gate_env_seeds") or (1, 2))
    out = []
    for kind in spec.KINDS:
        kd = (thr.get("kinds") or {}).get(kind) or {}
        band = kd.get("gate_band") if kd.get("status") == "gated" else (
            kd.get("r_band") if kd.get("status") == "solved" else None)
        if band:
            out.append((spec.row_id(kind, band), ess))
    return out


def plan_gate(new_ckpt, thresholds: Path = THRESHOLDS, train_name: str = TRAIN_NAME) -> list:
    rows = rt_spec().load_rows()
    by_id = {r["id"]: r for r in rows["kind_rows"] + rows["retention_rows"]}
    new_ckpt = Path(new_ckpt)
    new_sha = sha256_file(new_ckpt) if new_ckpt.is_file() else "0" * 64
    steps = check_steps("G")
    steps.append(_cpu("G.thresholds", MAKE_THRESHOLDS, "--check", "--out", thresholds, stop=True, dry=True,
                      note="thresholds_rough_v1.json regenerates byte-equal (frozen at STOP 0)"))
    steps.append(_cpu("G.ident", HERE / "rt_machinery.py", "gate-ident", "--new-ckpt", new_ckpt, "--thresholds",
                      thresholds, "--record", RUNS / f"{train_name}.json", stop=True, dry=True))
    thr = load_thresholds(thresholds) if Path(thresholds).is_file() else {"kinds": {}, "rule": {}}
    for rid, ess in gate_rows(thr):
        for es in ess:
            steps.append(gate_step(f"G.new.{rid}.es{es}", name=row_name("RT9", new_sha, rid), ck=new_ckpt,
                                   row=by_id[rid], es=es, tp_out=OUT))
            steps.append(gate_step(f"G.tdr.{rid}.es{es}", name=row_name("RT9", TDR_SHA, rid), ck=TDR_CK,
                                   row=by_id[rid], es=es, tp_out=OUT))
    for r in rows["retention_rows"]:
        for es in r["env_seeds"]:
            steps.append(gate_step(f"G.new.{r['id']}.es{es}", name=row_name("RT9", new_sha, r["id"]), ck=new_ckpt,
                                   row=r, es=es, tp_out=OUT))
    steps.append(_cpu("G.audit", HERE / "rt_machinery.py", "audit", "--block", "gate", "--new-sha", new_sha,
                      "--thresholds", thresholds, dry=True))
    steps.append(_cpu("G.read", READOUT, "gates", "--new", new_sha, "--tdr", TDR_SHA, "--thresholds", thresholds,
                      "--out", RESULTS / "step9" / "gates.json"))
    return steps


def plan(block: str, new_ckpt=None, thresholds: Path = THRESHOLDS, train_name: str = TRAIN_NAME) -> list:
    if block == "smoke":
        steps = plan_smoke(train_name if train_name.startswith("RT5_") else "RT5_smoke50")
    elif block == "baseline":
        steps = plan_baseline()
    elif block == "train":
        steps = plan_train(train_name)
    elif block == "gate":
        if not new_ckpt:
            raise ValueError("the gate block needs --new-ckpt (the trained teacher's final checkpoint)")
        steps = plan_gate(new_ckpt, thresholds, train_name)
    else:
        raise ValueError(f"unknown block {block!r}; one of {BLOCKS}")
    labels = [s.label for s in steps]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate step labels")
    slots = [s.slot for s in steps if s.slot]
    if len(slots) != len(set(slots)):
        raise RuntimeError("two gate rows share an output slot")
    return steps


def audit(block: str, new_sha: str | None = None, thresholds: Path = THRESHOLDS, pin=None) -> dict:
    """Every row a block planned, found through readout's own keying (sha256 + scenario + seed + composed env +
    args): ``{"rows": [...], "bad": n, "builds": [...], "pin_problems": [...]}``. ``pin_problems`` is non-empty when
    there is no pin or an OK row was recorded at a build other than the pin's (a row skipped onto a pre-re-pin
    build never feeds make_thresholds / the gates). ``pin`` defaults to results/pin.json."""
    ro = readout()
    want = []
    if block == "baseline":
        want = [(rid, es, TDR_SHA) for _l, rid, es in baseline_rows()]
    elif block == "gate":
        thr = load_thresholds(thresholds)
        for rid, ess in gate_rows(thr):
            want += [(rid, es, s) for es in ess for s in (new_sha, TDR_SHA)]
        for r in rt_spec().load_rows()["retention_rows"]:
            want += [(r["id"], int(es), new_sha) for es in r["env_seeds"]]
    else:
        raise ValueError("audit --block baseline|gate")
    R = ro.Rows([OUT / "lr"], shas={s for _r, _e, s in want})
    rows = []
    for rid, es, sha in want:
        f = R.found(rid, es, sha)
        rows.append({"row": rid, "es": es, "sha": short_sha(sha), "status": f["status"], "stamp": f["stamp"],
                     "problems": f["problems"][:3], "code_sha": str(f.get("code_sha"))[:12],
                     "eval_sha": str(f.get("eval_sha"))[:12],
                     "_build": (f.get("code_sha"), f.get("eval_sha"))})
    pin = read_pin() if pin is None else pin
    pin_probs = []
    if pin is None:
        pin_probs.append(f"no pin {PIN_PATH}")
    else:
        off = sorted({r["_build"] for r in rows if r["status"] == "OK"} - {(pin.get("code_sha"), pin.get("eval_sha"))},
                     key=str)
        if off:
            pin_probs.append(f"OK rows at a build other than the pin's {short_sha(str(pin.get('code_sha')))}/"
                             f"{short_sha(str(pin.get('eval_sha')))}: "
                             + "; ".join(f"{short_sha(str(c))}/{short_sha(str(e))}: "
                                         f"{[r['row'] + ' es' + str(r['es']) for r in rows if r['_build'] == (c, e)][:4]}"
                                         for c, e in off))
    for r in rows:
        r.pop("_build")
    builds = sorted({(r["code_sha"], r["eval_sha"]) for r in rows if r["status"] == "OK"})
    return {"block": block, "rows": rows, "bad": sum(r["status"] != "OK" for r in rows), "builds": builds,
            "pin_problems": pin_probs}


# ============================================================================== guards
def ddc_gb(path: Path = DDC) -> float:
    if not Path(path).is_dir():
        return 0.0
    r = subprocess.run(["du", "-s", "--block-size=1G", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0


def free_gb(path) -> float:
    return shutil.disk_usage(str(path)).free / 2 ** 30


def guard_problems(ddc: float, free: dict) -> list:
    out = []
    if ddc > DDC_MAX_GB:
        out.append(f"Kit DerivedDataCache {ddc:.0f} GB > {DDC_MAX_GB:g} GB (quarantine it first; the user's call)")
    for k, v in free.items():
        if v < FREE_MIN_GB:
            out.append(f"free disk {v:.0f} GB < {FREE_MIN_GB:g} GB at {k}")
    return out


def isaac_patterns() -> tuple[str, str]:
    """distill.sh's (ISAAC_PAT, NONKIT_PAT) with this campaign's Kit entry points (``rt_kit_*.py``) added."""
    text = DISTILL_SH.read_text()
    pats = {}
    for var in ("ISAAC_PAT", "NONKIT_PAT"):
        m = re.search(r"^%s='([^']*)'\s*$" % var, text, re.M)
        if not m:
            raise RuntimeError(f"{DISTILL_SH}: no {var}")
        pats[var] = m.group(1)
    isaac = pats["ISAAC_PAT"].replace("(run_phases|", "(rt_kit_[A-Za-z0-9_]+|run_phases|", 1)
    if isaac == pats["ISAAC_PAT"]:
        raise RuntimeError("distill.sh ISAAC_PAT no longer starts its alternation with (run_phases|")
    return isaac, pats["NONKIT_PAT"]


def ps_pid_args() -> list[tuple[int, str]]:
    p = subprocess.run(["ps", "-ww", "-eo", "pid=,args="], capture_output=True, text=True)
    out = []
    for ln in p.stdout.splitlines():
        f = ln.strip().split(None, 1)
        if len(f) == 2 and f[0].isdigit():
            out.append((int(f[0]), f[1]))
    return out


def busy(exclude=(), procs=None, pats=None) -> list[tuple[int, str]]:
    isaac, nonkit = pats or isaac_patterns()
    skip = {int(x) for x in exclude}
    return [(pid, a) for pid, a in (procs if procs is not None else ps_pid_args())
            if pid not in skip and re.search(isaac, a) and not re.search(nonkit, a)]


# ============================================================================== sentinels
def sentinel(label: str, rc: int, note: str = "") -> str:
    return f"RTQ {label} rc={int(rc)}" + (f" {note}" if note else "")


def done_line(block: str, fail: int, note: str = "") -> str:
    return f"RTQ_DONE {block} fail={int(fail)}" + (f" {note}" if note else "")


def parse_events(lines) -> list[dict]:
    """The decision-relevant queue events: a step with rc != 0, a NOTIFY line, the RTQ_DONE line."""
    out = []
    for i, raw in enumerate(lines, 1):
        ln = raw.rstrip("\n")
        m = SENTINEL_RE.match(ln)
        if m and m.group("label") != "NOTIFY" and int(m.group("rc")) != 0:
            out.append({"line": i, "kind": "fail", "label": m.group("label"), "rc": int(m.group("rc")), "text": ln})
        elif NOTIFY_RE.match(ln):
            out.append({"line": i, "kind": "notify", "text": ln})
        elif DONE_RE.match(ln):
            d = DONE_RE.match(ln)
            out.append({"line": i, "kind": "done", "block": d.group("block"), "fail": int(d.group("fail")),
                        "text": ln})
    return out


# ============================================================================== training log telemetry
def iteration_blocks(text: str) -> list[tuple[int, int, str]]:
    marks = [(m.start(), int(m.group(1)), int(m.group(2))) for m in ITER_RE.finditer(text)]
    out = []
    for j, (pos, it, tot) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(text)
        out.append((it, tot, text[pos:end]))
    return out


def block_values(block: str, prefix: str) -> dict:
    out = {}
    for m in re.finditer(r"^\s*(%s[\w/+.-]*):\s*(\S+)\s*$" % re.escape(prefix), block, re.M):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            out[m.group(1)] = float("nan")
    return out


CNT = "Metrics/base_parkour/site_cnt_"


def fall_shares(blocks, it0: int, lo: int, hi: int) -> dict:
    """``{kind: {"failed", "ended", "share"}}`` over the blocks with ``lo <= it - it0 < hi``: the zero-filled
    per-reset-call counts ``site_cnt_failed/<kind>`` / ``site_cnt_end/<kind>`` summed (both carry the same
    1/num_steps_per_env scale, so their ratio is the share of the window's ended episodes that ended on
    crab_failure)."""
    acc: dict = {}
    for it, _tot, blk in blocks:
        if not lo <= it - it0 < hi:
            continue
        for key, val in block_values(blk, CNT).items():
            stem, _, kind = key[len(CNT):].partition("/")
            if stem in ("failed", "end") and kind:
                a = acc.setdefault(kind, {"failed": 0.0, "ended": 0.0, "blocks": 0})
                a["failed" if stem == "failed" else "ended"] += val
                if stem == "end":
                    a["blocks"] += 1
    for a in acc.values():
        a["share"] = (a["failed"] / a["ended"]) if a["ended"] > 0 else None
    return acc


def early_line(text: str, it0: int, at: int = EARLY_READ_AT, ref=EARLY_READ_REF, win: int = EARLY_READ_WIN):
    """The iteration-``at`` notification line (PLAN step 3 / settled 6: report only, no stop, no remedy): each
    kind's fall share over iterations ``[at - win, at)`` against ``[ref)``. None until a block at ``at`` exists."""
    blocks = iteration_blocks(text)
    if not any(it - it0 >= at for it, _t, _b in blocks):
        return None
    now = fall_shares(blocks, it0, at - win, at)
    base = fall_shares(blocks, it0, ref[0], ref[1])

    def f(x):
        return "na" if x is None else f"{x:.3f}"

    parts = []
    for kind in sorted(set(now) | set(base)):
        b, n = (base.get(kind) or {}).get("share"), (now.get(kind) or {}).get("share")
        parts.append(f"{kind}={f(b)}->{f(n)}")
    return (f"RTQ NOTIFY early_read it={at} fall_share[{ref[0]}-{ref[1]}]->[{at - win}-{at}] " + " ".join(parts)
            + " (report only: no stop, no remedy)")


# ============================================================================== training argv
def trainer_argv(name: str, checkpoint, iterations: int = TRAIN_ITERS, num_envs: int = TRAIN_ENVS,
                 seed: int = TRAIN_SEED) -> list[str]:
    """run_phases.train's command shape (rsd_train.py's template): no -I (it would drop PYTHONPATH)."""
    return [PY, str(PARKOUR / "scripts" / "rsl_rl" / "train.py"), "--task", TEACHER_TASK_TRAIN, "--headless",
            "--num_envs", str(num_envs), "--seed", str(seed), "--max_iterations", str(iterations), "--resume",
            "--checkpoint", str(checkpoint), "--run_name", name]


def final_iteration(it0: int, iterations: int) -> int:
    """The runner saves ``model_<it0 + iterations - 1>.pt`` last (iterations it0 .. it0 + iterations - 1)."""
    return int(it0) + int(iterations) - 1


def gate_ident_problems(new_ckpt: Path, record_path: Path) -> list[str]:
    """The checkpoint handed to the gate block must BE the step-8 run's final model: ``record_path`` (results/runs/
    RT_s19.json) status ``ok``, a real run (not the RT5_* launcher smoke), ``final.sha256`` == sha256(new_ckpt) and
    ``final.iter`` == the planned last iteration (``final_iteration(TDR_ITER, TRAIN_ITERS)`` = 34990)."""
    try:
        rec = json.loads(Path(record_path).read_text())
    except (OSError, ValueError) as exc:
        return [f"no readable step-8 run record {record_path}: {exc}"]
    why = []
    name = str(rec.get("name"))
    if name.startswith("RT5_") or (rec.get("cli") or {}).get("smoke"):
        why.append(f"run record {record_path} is a launcher smoke ({name}), not the step-8 run")
    if name != TRAIN_NAME:
        why.append(f"run record name {name!r} != {TRAIN_NAME!r}")
    if rec.get("status") != "ok":
        why.append(f"run record status {rec.get('status')!r} != 'ok' (why: {rec.get('why')})")
    fin = rec.get("final") or {}
    want_iter = final_iteration(TDR_ITER, TRAIN_ITERS)
    if fin.get("iter") != want_iter:
        why.append(f"run record final iter {fin.get('iter')!r} != {want_iter}")
    if Path(new_ckpt).is_file() and fin.get("sha256") != sha256_file(new_ckpt):
        why.append(f"sha256({Path(new_ckpt).name}) {sha256_file(new_ckpt)[:8]} != the run record's final "
                   f"{str(fin.get('sha256'))[:8]}")
    return why


def _params_diff_cli(a) -> int:
    """``params-diff``: print every allowed and every disallowed difference, then one ``RT PARAMS OK|BAD`` line."""
    run_dir = a.run_dir or find_run_dir(a.name)
    if run_dir is None:
        print(f"RT PARAMS BAD no run dir *_{a.name} under {PARKOUR / 'logs' / 'rsl_rl' / 'crab_hex_omni_teacher'}")
        return 1
    params = Path(run_dir) / "params" if (Path(run_dir) / "params").is_dir() else Path(run_dir)
    expect = {"name": a.name, "iterations": a.iterations, "seed": a.seed, "checkpoint": a.checkpoint}
    res = params_check(params, a.ref_dir, expect, krabby_of(a.env_file), smoke=a.smoke)
    n_ok = n_bad = 0
    for fname, r in res["files"].items():
        if r.get("error"):
            print(f"RT PARAMS DISALLOWED {fname}: {r['error']}")
            n_bad += 1
        for d in r["allowed"]:
            n_ok += 1
            print(f"RT PARAMS allowed {fname} {d['path']}: {_short(d['ref'])} -> {_short(d['new'])}  [{d['why']}]")
        for d in r["disallowed"]:
            n_bad += 1
            print(f"RT PARAMS DISALLOWED {fname} {d['path']}: {_short(d['ref'])} -> {_short(d['new'])}  [{d['why']}]")
    print(f"RT PARAMS {'OK' if res['ok'] else 'BAD'} run={params} ref={a.ref_dir} allowed={n_ok} disallowed={n_bad}")
    return 0 if res["ok"] else 1


# ============================================================================== CLI
def _print_plan(steps) -> None:
    for s in steps:
        env = " ".join(f"{k}={v}" for k, v in sorted(s.env.items()))
        print(f"{s.label}|{s.how}|{'stop' if s.stop_on_fail else ''}|{env}|{' '.join(s.argv)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--block", required=True, choices=BLOCKS)
    p.add_argument("--new-ckpt", default=None)
    p.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    p = sub.add_parser("make-envs")
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("train-env-check")
    p.add_argument("--env-file", type=Path, default=RT_TRAIN_ENV)
    p = sub.add_parser("audit")
    p.add_argument("--block", required=True, choices=("baseline", "gate"))
    p.add_argument("--new-sha", default=None)
    p.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    p = sub.add_parser("gate-ident")
    p.add_argument("--new-ckpt", required=True, type=Path)
    p.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    p.add_argument("--record", type=Path, default=RUNS / f"{TRAIN_NAME}.json",
                   help="the step-8 run record (results/runs/RT_s19.json)")
    p = sub.add_parser("params-diff")
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", type=Path, default=None, help="default: this worktree's newest <ts>_<name> run dir")
    p.add_argument("--ref-dir", type=Path, default=TDR_PARAMS)
    p.add_argument("--env-file", type=Path, default=RT_TRAIN_ENV)
    p.add_argument("--iterations", type=int, default=TRAIN_ITERS)
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--checkpoint", default=str(TDR_CK))
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("early-line")
    p.add_argument("log", type=Path)
    p.add_argument("--it0", type=int, default=TDR_ITER)
    sub.add_parser("neutral-refs")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        _print_plan(plan(a.block, a.new_ckpt, a.thresholds))
        return 0
    if a.cmd == "make-envs":
        bad = make_envs(a.check)
        print(f"RT ENVS {'OK' if not bad else 'BAD'} {'checked' if a.check else 'written'}"
              + (f" -- {'; '.join(bad)}" if bad else ""))
        return 1 if bad else 0
    if a.cmd == "train-env-check":
        bad = train_env_problems(a.env_file)
        print(f"RT TRAINENV {'OK' if not bad else 'BAD'} {a.env_file}" + (f" -- {'; '.join(bad)}" if bad else ""))
        return 1 if bad else 0
    if a.cmd == "audit":
        try:
            res = audit(a.block, a.new_sha, a.thresholds)
        except (OSError, ValueError) as exc:
            print(f"RT AUDIT ERROR {type(exc).__name__}: {exc}")
            return 1
        for r in res["rows"]:
            if r["status"] != "OK":
                print(f"  {r['status']} {r['row']} es{r['es']} {r['sha']} {r['problems']}")
        for p in res["pin_problems"]:
            print(f"  PIN {p}")
        print(f"RT AUDIT {a.block} rows={len(res['rows'])} bad={res['bad']} builds={res['builds']} "
              f"pin_ok={not res['pin_problems']}")
        return 1 if res["bad"] or len(res["builds"]) > 1 or res["pin_problems"] else 0
    if a.cmd == "gate-ident":
        why = []
        if not a.new_ckpt.is_file():
            why.append(f"no new checkpoint {a.new_ckpt}")
        elif sha256_file(a.new_ckpt) == TDR_SHA:
            why.append("the new checkpoint IS TDR")
        why += gate_ident_problems(a.new_ckpt, a.record)
        try:
            thr = load_thresholds(a.thresholds)
            if thr.get("decision") != "proceed":
                why.append(f"thresholds decision {thr.get('decision')!r} (PLAN: steps 8-9 run only on 'proceed')")
            if thr.get("tdr_sha256") not in (None, TDR_SHA):
                why.append(f"thresholds frozen on TDR {str(thr.get('tdr_sha256'))[:8]}")
            n = len(gate_rows(thr))
            if not n:
                why.append("no gate / R band in the thresholds file")
        except (OSError, ValueError) as exc:
            why.append(f"thresholds: {exc}")
        print(f"RT GATEIDENT {'OK' if not why else 'BAD'} new={a.new_ckpt}" + (f" -- {'; '.join(why)}" if why else ""))
        return 1 if why else 0
    if a.cmd == "params-diff":
        return _params_diff_cli(a)
    if a.cmd == "early-line":
        line = early_line(a.log.read_text(errors="replace"), a.it0)
        print(line or f"RT EARLY no block at +{EARLY_READ_AT} yet")
        return 0
    if a.cmd == "neutral-refs":
        for r in neutral_refs():
            print(r)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
