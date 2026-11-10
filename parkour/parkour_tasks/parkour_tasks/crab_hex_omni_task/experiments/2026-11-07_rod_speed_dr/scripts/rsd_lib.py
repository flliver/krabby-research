#!/usr/bin/env python3
"""Rod-speed DR campaign (CHARTER 2026-11-07): the gate library -- policies, the pin, rows, stamp checks, reads and
verdicts. CPU only: nothing here boots Kit, trains or evaluates (``rsd_gate.py`` runs the rows).

Every rule is the CHARTER's (section 2 ground rules, section 3.3 G1, 3.4 the student stage and G2, 3.5 the disk guard,
3.6 the stop rules); where the CHARTER leaves a detail open, the choice is written next to the code and listed in
``IMPLEMENTATION_NOTES`` (the gate json carries it).

Reused read-only, by import (never copied): the sweep's ``cells.py`` (env_text, level_file_for, TERM, the cell ids) and
``stress_read.py`` (read_stamps -- the 18 per-seed lines --, deltas, level_verdict) with every ``DRS_*`` variable
cleared while they load; Task 4's ``tools/pipeline`` gate_read (stamp_problems, floor_reads, completion, cat_summary,
walker_decisive, zero_kg_check, verdict_of, Thresholds), payload_read (level_reads), select_student
(candidate_checkpoints, rank, next_action, completion_pairs, tracking_watch), thresholds.py and
``tools/run_candidates.py`` (code_sha, eval_sha, code_drop, CODE_PATHSPECS, EVAL_PATHSPECS). None of them is edited.

Names: gate rows ``RSDT_<sha8>_<cell>`` (teacher role) / ``RSDS_<sha8>_<cell>`` (student role), cells in the sweep's
id convention (``T00_nominal_<base>_<launch>``, ``T01_rod_speed_0p7_L090`` ...); selection rows
``RSDSEL_<sha8>_L<kg>`` (student candidate) / ``RSDSELT_<sha8>_L<kg>`` (teacher), Task 4's ``sel`` row specs. A
relaunch (the bridge's one relaunch, G2's class-ambiguity relaunch) appends ``_RL``; the first launch stays the
reference. Stamps: ``$RSD_TP_OUT/lr/<NAME>/<scenario>/seed00K/<time>/`` (default the main checkout's
``parkour/outputs/omni_plan/rod_speed_dr``); results ``$RSD_RESULTS`` (default ``<campaign>/results``).
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
W = Path(__file__).resolve().parents[7]                     # this worktree's root
MAIN = Path("/home/nickmagus/krabby/krabby-research")       # the main checkout (stamps, logs: data only)
OMNI_REL = Path("parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task")
OMNI = W / OMNI_REL
EXP = OMNI / "experiments"
TOOLS = EXP / "tools"
PIPE = TOOLS / "pipeline"
T4_REL = OMNI_REL / "experiments/2026-10-16_task4_pipeline"
T4 = W / T4_REL
T4_MAIN = MAIN / T4_REL
SWEEP = EXP / "2026-11-06_dr_stress_sweep"
GATE_ROW = PIPE / "gate_row.sh"
SPEC_PATH = T4 / "pipeline.yaml"
PLAN_PATH = T4 / "F19/plan.json"
TH_PATH = T4 / "thresholds_v1_2.json"
TH_INPUTS = T4 / "threshold_inputs.yaml"
POLICY_MANIFEST = OMNI / "policy/manifest.yaml"
TASK = "Isaac-Crab-Hex-Omni-Student-Play-v0"
TP_OUT = Path(os.environ.get("RSD_TP_OUT") or MAIN / "parkour/outputs/omni_plan/rod_speed_dr")
RESULTS = Path(os.environ.get("RSD_RESULTS") or CAMPAIGN / "results")
ENVS = CAMPAIGN / "envs"
SWEEP_TP_OUT = MAIN / "parkour/outputs/omni_plan/dr_stress"
T4_TP_OUT = MAIN / "parkour/outputs/omni_plan/task4/F19"
DDC = Path("/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache")
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"

EVAL_SHA = "9fe964d7f46d1116d1d2983b0ece43f2430695cc35c9f33fd4cdc28f169ed5b8"
FREE_MIN_GB = 30.0          # CHARTER 3.5: stop below 30 GB free
DDC_MAX_GB = 78.0           # CHARTER 3.5: stop above 78 GB (distill.sh's own limit is 80, same du --block-size=1G unit)
ROW_TIMEOUT_S = 5400
ROD_RATED = (0.028, 0.033)  # m/s: hip, knee (the harness records [hip x6, knee x6])
DOSE_TOL = 1e-6
ES3_CAP = 4                 # CHARTER 3.3: at most 4 es3 cell-seeds per gate
G2_AMBIG = 0.04             # CHARTER 3.4: a pooled shortfall in (m, m + 0.04] is class-ambiguous
COMPLETION_LINE = 0.85
ITS_PAST_P2A = {"g1": 2999, "g1_extended": 5998, "g2": 0}
RELAUNCH = "_RL"

TSTAR_CKPT = OMNI / "policy/f19_8_p3b_payload180/model_25992.pt"
TSTAR_SHA_PREFIX = "03b6009a"
C28000_CKPT = MAIN / "parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt"

#: the cells of the campaign (sweep cell ids)
NOM = {b: f"T00_nominal_{b}_A" for b in ("L000", "L090", "L135")}
X07 = "T01_rod_speed_0p7_L090"
X08 = "T01_rod_speed_0p8_L090"
X07_L000 = "T01_rod_speed_0p7_L000"
X135_L000 = "T01_rod_speed_1p35_L000"
X135_L090 = "T01_rod_speed_1p35_L090"
S_CELLS = [X07, X08, X07_L000, X135_L000, X135_L090]       # S: x0.7 / x0.8 L090 + the three band edges
CARVE_CELLS = (X07, X08)                                   # the CAPABILITY-LIMITED carve-out applies here only
N_CELLS = [NOM["L000"], NOM["L090"], NOM["L135"]]
REF_CELLS = N_CELLS + [X07, X08]                           # the in-pin reference arm (old T* at G1, c28000 at G2)
TCTL_CELLS = [NOM["L090"], X07, X08]
BRIDGE_TEACHER = [(NOM["L090"], 1), (NOM["L090"], 2), (X07, 1), (X07, 2), (X08, 1), (X08, 2)]
BRIDGE_STUDENT = [(NOM["L090"], 1), (X07, 1)]
CLOSING_CELL = "T00_nominal_L090_Z"
G2_WATCH = ["T02_rod_effort_0p85_L090", "T13_payload_kg_150_payload"]
SEVEN = ["side22@1", "field_hard@1", "field_hard@2", "ring_obst@1", "ring_obst@2", "field_turn@1", "field_turn@2"]
LEVEL_FILES = {"L000": None, "L090": T4 / "F19/envs/levels/L090.env", "L135": T4 / "F19/envs/levels/L135.env"}
BASE_KG = {"L000": None, "L090": 90, "L135": 135}

IMPLEMENTATION_NOTES = [
    "es3 stamps are read with the es1 line ids (payload_read.level_reads reads env seeds 1-2 only; the es1 and es2 "
    "line values are asserted identical before the relabel).",
    "The tracking carve-out at an es3 seed uses max(|drift es1|, |drift es2|) (thresholds.json has no es3 drift); "
    "it needs the in-pin reference's seed-K stamp, and no reference runs es3, so an es3 tracking line is read "
    "absolute. Likewise the L135 shared-capability-limit rule needs the reference's same seed and is absent at es3.",
    "Near-line (es3) needs exactly ONE failing line key over es1 + es2 (the CHARTER's 'its only failing or "
    "knife-edge line'), every failing value inside its knife-edge band, and a roll / pitch miss on one seed only.",
    "EXTEND's 'only stress cells fail, on completion or survival' = every failing S line is field_light_completion "
    "or survival.<c> (an h_meas / turn_fc / tracking / roll / pitch miss makes it FAIL-STRESS).",
    "DR null (TDR x0.7 - old < m, so TCTL never runs): FAIL-REGRESSION if N or R fails, else UNATTRIBUTED if S "
    "passes (no attribution possible), else FAIL-STRESS noted 'DR null' (the dose audit precedes recording it).",
    "A pending es3 / relaunch is resolved by enumeration: when every outcome of the pending cells gives the same "
    "verdict, that verdict stands; otherwise PENDING-ES3 / RELAUNCH-DUE, or UNRESOLVED once the 4 cell-seed "
    "allowance is exceeded. The allowance is per GATE: the post-EXTEND G1 read also counts the pre-EXTEND TDR's "
    "es3 cell-seeds (read g1 --extended --prior-tdr), and a G2 class-ambiguity relaunch is charged as one 4 "
    "cell-seed block per ambiguous cell from the moment it is due (a half-finished relaunch is not double-counted).",
    "The bridge and G2 relaunches use the gate name + '_RL'; the first launch stays the in-pin reference.",
    "A row whose stamp fails the checks twice (two launches) stops the queue (decide-record-notify).",
    "The shared L000 rows (TDR's / old T*'s L000 nominal = the suite@1/2 + field_light@1/2 selection / 0-kg rows) "
    "run with the cell composition (no TP_STRIP_KEYS); every refused runner key is absent from p3b.env / "
    "p3b.field_light.env (checked), so the strip is a no-op and the rows are identical.",
    "G2's cross-pin R WATCH track_err line = select_student.tracking_watch (|ratio - 1|, +0.05) of the pick vs "
    "c28000's Task 4 full block; completion rows flag below c28000 - m (completion_pairs, m = max(0.05, 2 SE)).",
]

_MODS: dict = {}


class Stop(Exception):
    """A decide-record-notify stop (the driver writes ``RSD STOP <why>``)."""

    def __init__(self, why: str, code: int = 4):
        super().__init__(why)
        self.why, self.code = why, code


class PolicyError(Exception):
    pass


# ============================================================================== module loading (read-only reuse)
def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
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


def tp():
    """Task 4's tp_lib (registered under its own name 'tp_lib', as gate_read / payload_read share it)."""
    if "tp" not in _MODS:
        mod = _load("tp_lib", PIPE / "tp_lib.py")
        mod.configure_from(SPEC_PATH)                       # side22 classifier / sh_kin paths
        _MODS["tp"] = mod
    return _MODS["tp"]


def gr():
    tp()
    return _load("rsd_tp_gate_read", PIPE / "gate_read.py")


def pr():
    tp()
    return _load("rsd_tp_payload_read", PIPE / "payload_read.py")


def ss():
    tp()
    return _load("rsd_tp_select_student", PIPE / "select_student.py")


def thm():
    return tp().thresholds_mod()


def rc():
    return tp().rc()


def sweep():
    """The sweep's cells + stress_read, loaded under private names with every ``DRS_*`` variable cleared (their
    output root and results dir are then the sweep's own). ``stress_read`` imports ``cells`` by plain name: the
    alias exists only while it loads; ``sys.path`` and the environment are restored afterwards."""
    if "sweep" in _MODS:
        return _MODS["sweep"]
    saved = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("DRS_")]}
    path0 = list(sys.path)
    prev = sys.modules.get("cells")
    try:
        cells = _load("rsd_sweep_cells", SWEEP / "scripts/cells.py")
        sys.modules["cells"] = cells
        sr = _load("rsd_sweep_stress_read", SWEEP / "scripts/stress_read.py")
    finally:
        if prev is None:
            sys.modules.pop("cells", None)
        else:
            sys.modules["cells"] = prev
        sys.path[:] = path0
        os.environ.update(saved)
    _MODS["sweep"] = SimpleNamespace(C=cells, SR=sr)
    return _MODS["sweep"]


def spec() -> dict:
    if "spec" not in _MODS:
        _MODS["spec"] = tp().load_yaml(SPEC_PATH)
    return _MODS["spec"]


def plan() -> dict:
    if "plan" not in _MODS:
        _MODS["plan"] = json.loads(PLAN_PATH.read_text())
    return _MODS["plan"]


def sel_spec() -> dict:
    return next(s for s in spec()["stages"] if s["id"] == "sel")


def p3b_plan() -> dict:
    return plan()["stages"]["p3b"]


def trained() -> list:
    """The campaign's trained list = Task 4 p3b's (T*'s stage)."""
    return list(p3b_plan()["trained"])


def trained_translate() -> list:
    return [c for c in trained() if c in gr().TRANSLATE]


def th_doc() -> dict:
    if "thdoc" not in _MODS:
        _MODS["thdoc"] = json.loads(TH_PATH.read_text())
    return _MODS["thdoc"]


def th_view():
    return gr().Thresholds(th_doc(), TH_PATH)


def resolution_files() -> list:
    """Task 4's resolution files (``threshold_inputs.yaml``, what ``pipeline.resolution_files`` reads)."""
    return list(tp().load_yaml(TH_INPUTS).get("resolution_files") or [])


def resolution():
    if "res" not in _MODS:
        files = [tp().absp(p) for p in resolution_files()]
        _MODS["res"] = (thm().resolution_R(files), thm().resolution_R_guard(files))
    return _MODS["res"]


def drift_table() -> dict:
    z = next(x for x in th_doc()["lines"] if x["id"] == "p3a.capability.zero_kg_paired")
    return dict(z["drift_by_statistic"])


def refused_keys() -> list:
    mod = _load("rsd_omni_student_spec", OMNI / "config/crab_hex/omni_student_spec.py")
    return list(mod.STUDENT_RUNNER_KNOBS_REFUSED)


def completion_cfg() -> dict:
    return dict(sel_spec()["completion_margin"])


# ============================================================================== small io
def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def read_json(p: Path, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else default


def write_json(p: Path, obj) -> None:
    tp().write_json_atomic(Path(p), obj)


_SHA: dict = {}


def sha256_file(p: Path) -> str:
    p = Path(p)
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _SHA:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def its_of(ckpt) -> int | None:
    m = re.search(r"model_(\d+)\.pt$", str(ckpt))
    return int(m.group(1)) if m else None


# ============================================================================== policies
POLICY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")


def policies_path() -> Path:
    return RESULTS / "policies.json"


def manifest_student_sha() -> str:
    """c28000's sha256 as policy/manifest.yaml records it: the live stage ``f19_9_student`` (the F19 era) or, since the
    2026-11-09 rod-speed DR bake moved that stage to ``superseded:``, the verbatim stage block of the superseded entry
    ``F19_student``; either way its ``source`` must be c28000's run file."""
    d = tp().load_yaml(POLICY_MANIFEST)
    st = next((s for s in d["stages"] if s.get("dir") == "f19_9_student"), None)
    where = "f19_9_student"
    if st is None:
        sup = next((e for e in d.get("superseded") or [] if e.get("id") == "F19_student"), None)
        st = (sup or {}).get("block")
        where = "superseded F19_student block"
        if not isinstance(st, dict) or st.get("dir") != "f19_9_student":
            raise PolicyError("policy/manifest.yaml records c28000 neither as stage f19_9_student nor as the "
                              "superseded F19_student block")
    want_src = C28000_CKPT.relative_to(MAIN).as_posix()
    if st.get("source") != want_src:
        raise PolicyError(f"policy/manifest.yaml {where} source {st.get('source')!r} != {want_src!r}")
    return st["sha256"]


def _check_ckpt(path: Path, role: str) -> None:
    if not path.is_absolute():
        raise PolicyError(f"checkpoint {path} is not absolute")
    if not path.is_file():
        raise PolicyError(f"no checkpoint {path}")
    if its_of(path) is None:
        raise PolicyError(f"{path.name} is not model_<iter>.pt")
    if role == "student" and not (path.parent / "params" / "agent.yaml").is_file():
        raise PolicyError(f"student checkpoint {path} has no params/agent.yaml beside it (its run dir's)")
    if role not in ("teacher", "student"):
        raise PolicyError(f"role {role!r} (want teacher | student)")


def _builtin(name: str) -> dict | None:
    if name == "TSTAR":
        return {"checkpoint": str(TSTAR_CKPT), "role": "teacher",
                "note": "T* = F19 p3b model_25992 (a live stage of policy/manifest.yaml, TDR's resume point)"}
    if name == "C28000":
        return {"checkpoint": str(C28000_CKPT), "role": "student",
                "note": "F19 depth student c28000 (policy/manifest.yaml superseded F19_student)"}
    return None


def _verify_builtin(name: str, sha: str) -> None:
    if name == "TSTAR" and not sha.startswith(TSTAR_SHA_PREFIX):
        raise PolicyError(f"TSTAR sha256 {sha[:12]} does not start {TSTAR_SHA_PREFIX}")
    if name == "C28000" and sha != manifest_student_sha():
        raise PolicyError(f"C28000 sha256 {sha[:12]} != c28000's policy/manifest.yaml record (stage f19_9_student or "
                          f"superseded F19_student) {manifest_student_sha()[:12]}")


def load_registry() -> dict:
    return read_json(policies_path(), {"policies": {}}) or {"policies": {}}


def policy_add(name: str, checkpoint, role: str, note: str = "") -> dict:
    """Bind NAME to a checkpoint's sha256 (idempotent for the same sha; refuses another sha)."""
    if not POLICY_RE.match(name):
        raise PolicyError(f"policy name {name!r} (want [A-Z][A-Z0-9_]*, <= 32 chars)")
    path = Path(checkpoint)
    _check_ckpt(path, role)
    sha = sha256_file(path)
    b = _builtin(name)
    if b is not None:
        if str(path) != b["checkpoint"] or role != b["role"]:
            raise PolicyError(f"{name} is pre-bound to {b['checkpoint']} ({b['role']})")
        _verify_builtin(name, sha)
    reg = load_registry()
    have = reg["policies"].get(name)
    if have is not None:
        if have["sha256"] != sha:
            raise PolicyError(f"policy {name} is bound to sha256 {have['sha256'][:12]} ({have['checkpoint']}); "
                              f"refusing to rebind it to {sha[:12]} ({path})")
        if have["role"] != role:
            raise PolicyError(f"policy {name} has role {have['role']}, not {role}")
        return policy(name)
    reg["policies"][name] = {"checkpoint": str(path), "sha256": sha, "role": role, "added_at": now(),
                             **({"note": note} if note else {})}
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(policies_path(), reg)
    return policy(name)


def policy(name: str) -> dict:
    """The bound policy, its file re-hashed (a changed file is refused); the pre-bound TSTAR / C28000 bind on first
    use after their sha checks."""
    reg = load_registry()
    have = reg["policies"].get(name)
    if have is None:
        b = _builtin(name)
        if b is None:
            raise PolicyError(f"no policy {name!r} in {policies_path()} (rsd_gate.py policy add)")
        return policy_add(name, b["checkpoint"], b["role"], b["note"])
    path = Path(have["checkpoint"])
    _check_ckpt(path, have["role"])
    sha = sha256_file(path)
    if sha != have["sha256"]:
        raise PolicyError(f"policy {name}: {path} now hashes {sha[:12]}, bound to {have['sha256'][:12]}")
    if _builtin(name) is not None:
        _verify_builtin(name, sha)
    return {"name": name, "checkpoint": str(path), "sha256": sha, "sha8": sha[:8], "role": have["role"],
            "iteration": its_of(path)}


def placeholder_policy(name: str, role: str) -> dict:
    """A dry-run stand-in for a policy not registered yet (TDR / TCTL / PICK before they exist): its rows print
    with ``sha8`` DRY<name> and an unbound checkpoint; nothing can run with it."""
    tag = re.sub(r"[^A-Za-z0-9]", "", name)[:5].upper()
    return {"name": name, "checkpoint": f"/UNBOUND/{name}/model_0.pt", "sha256": f"DRY{tag}".ljust(64, "0"),
            "sha8": f"DRY{tag}"[:8], "role": role, "iteration": None, "placeholder": True}


def policy_for(name: str, role: str, dry: bool) -> dict:
    """``policy(name)``; in a dry run an unregistered name gets a placeholder."""
    if dry and name not in load_registry()["policies"] and _builtin(name) is None:
        return placeholder_policy(name, role)
    return policy(name)


def ephemeral_policy(ckpt, role: str, name: str) -> dict:
    """A selection candidate (not registered): the same file checks, its sha computed."""
    path = Path(ckpt)
    _check_ckpt(path, role)
    sha = sha256_file(path)
    return {"name": name, "checkpoint": str(path), "sha256": sha, "sha8": sha[:8], "role": role,
            "iteration": its_of(path)}


# ============================================================================== the pin and the guards
def launch_path() -> Path:
    return RESULTS / "launch.json"


def launch_pin() -> dict | None:
    return read_json(launch_path())


def code_state() -> dict:
    """code_sha / eval_sha of this checkout (HEAD trees) + the dirty code / eval paths (gate_row's own test)."""
    r = rc()
    st = subprocess.run(["git", "-C", str(r.REPO), "status", "--porcelain", "--", *r.CODE_PATHSPECS,
                         *r.EVAL_PATHSPECS], capture_output=True, text=True, check=True).stdout.split("\n")
    dirty = [ln[3:] for ln in st if ln.strip() and not r.code_drop(ln[3:])]
    head = subprocess.run(["git", "-C", str(r.REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    return {"code_sha": r.code_sha(), "eval_sha": r.eval_sha(), "head": head, "dirty": dirty,
            "repo": str(r.REPO)}


def write_pin() -> dict:
    """``pin``: results/launch.json from a CLEAN tree; refuses a dirty tree, an eval_sha other than 9fe964d7 and
    re-pinning to other hashes (one code pin from the pre-flight to G2)."""
    cs = code_state()
    if cs["dirty"]:
        raise Stop(f"dirty_tree {cs['dirty'][:8]}", 5)
    if cs["eval_sha"] != EVAL_SHA:
        raise Stop(f"eval_sha {cs['eval_sha'][:12]} != {EVAL_SHA[:12]} (CHARTER section 2)", 5)
    have = launch_pin()
    if have is not None:
        if (have["code_sha"], have["eval_sha"]) != (cs["code_sha"], cs["eval_sha"]):
            raise Stop(f"a different pin exists (code {have['code_sha'][:12]} / eval {have['eval_sha'][:12]}); "
                       "one code pin from the pre-flight to G2: re-pinning is the user's call", 5)
        return have
    pin = {"code_sha": cs["code_sha"], "eval_sha": cs["eval_sha"], "head": cs["head"], "pinned_at": now()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(launch_path(), pin)
    return pin


def check_tree() -> dict:
    """Before every row: a clean tree whose hashes equal the pin."""
    cs = code_state()
    if cs["dirty"]:
        raise Stop(f"dirty_tree {cs['dirty'][:8]}", 5)
    pin = launch_pin()
    if pin is None:
        raise Stop("no pin (results/launch.json): run `rsd_gate.py pin` on the clean knob commit", 5)
    if (pin["code_sha"], pin["eval_sha"]) != (cs["code_sha"], cs["eval_sha"]):
        raise Stop(f"code_changed_since_pin pin={pin['code_sha'][:12]}/{pin['eval_sha'][:12]} "
                   f"now={cs['code_sha'][:12]}/{cs['eval_sha'][:12]}", 5)
    if pin["eval_sha"] != EVAL_SHA:
        raise Stop(f"pinned eval_sha {pin['eval_sha'][:12]} != {EVAL_SHA[:12]}", 5)
    return pin


def ddc_gb(path: Path = DDC) -> float:
    """``du -s --block-size=1G`` (distill.sh's / pipeline.ddc_gb's unit)."""
    if not Path(path).is_dir():
        return 0.0
    r = subprocess.run(["du", "-s", "--block-size=1G", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0


def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / (1 << 30)


def disk_problems(out: Path | None = None, ddc: Path = DDC) -> list[str]:
    """CHARTER 3.5: stop below 30 GB free or with the Kit DerivedDataCache above 78 GB."""
    bad = []
    f = free_gb(out or TP_OUT)
    if f < FREE_MIN_GB:
        bad.append(f"free space {f:.1f} GB < {FREE_MIN_GB:g} GB")
    d = ddc_gb(ddc)
    if d > DDC_MAX_GB:
        bad.append(f"Kit DerivedDataCache {d:.0f} GB > {DDC_MAX_GB:g} GB (a prune is the user's call)")
    return bad


# ============================================================================== cells and env files
NOMINAL_RE = re.compile(r"^T00_nominal_(L\d{3})_([A-Z][A-Z0-9]*)$")


def cell(cid: str) -> dict:
    """A campaign cell: ``T00_nominal_<base>_<launch>`` (L000 / L090 / L135), or a sweep cell (T01 rod speed, the
    G2 watch cells T02 x0.85 L090 and T13 150 kg)."""
    m = NOMINAL_RE.match(cid)
    if m:
        base = m.group(1)
        if base not in LEVEL_FILES:
            raise KeyError(f"nominal base {base} (want L000 / L090 / L135)")
        lf = LEVEL_FILES[base]
        return {"id": cid, "term": None, "level": None, "base": base, "kind": "nominal", "rod_s": 1.0,
                "level_file": None if lf is None else str(lf), "level_kg": BASE_KG[base]}
    C = sweep().C
    by = {c["id"]: c for c in C.cells()}
    if cid not in by:
        raise KeyError(f"unknown cell {cid}")
    c = by[cid]
    t = C.TERM[c["term"]]
    if c["term"] not in ("T01", "T02", "T13"):
        raise KeyError(f"cell {cid}: term {c['term']} is not part of this campaign")
    if t["kind"] == "payload_kg":
        lf, kg = ENVS / "levels" / f"L{int(c['level']):03d}.env", int(c["level"])
    else:
        lf, kg = C.level_file_for(t, c["level"], c["base"]), BASE_KG[c["base"]]
    return {"id": cid, "term": c["term"], "level": c["level"], "base": c["base"], "kind": t["kind"],
            "rod_s": float(c["level"]) if c["term"] == "T01" else 1.0,
            "level_file": None if lf is None else str(lf), "level_kg": kg}


def env_file(cid: str, fam: str) -> Path:
    return ENVS / f"{cid}__{fam}.env"


def cell_env_text(cid: str, fam: str) -> str:
    C = sweep().C
    c = cell(cid)
    term = None if c["term"] is None else C.TERM[c["term"]]
    return C.env_text(term, c["level"], fam)


def prepare_cell(cid: str) -> None:
    """Compose the cell's env files (cells.env_text) and, for T13, its level file (cells.level_text, which must
    equal the sweep's own file byte for byte)."""
    ENVS.mkdir(parents=True, exist_ok=True)
    C = sweep().C
    for fam in C.ROWS:
        p = env_file(cid, fam)
        txt = cell_env_text(cid, fam)
        if txt.startswith(("KRABBY_DR_",)) or "\nKRABBY_DR_" in txt:
            raise Stop(f"{p.name}: an eval env carries KRABBY_DR_* (CHARTER section 2)", 5)
        if not p.is_file() or p.read_text() != txt:
            p.write_text(txt)
    c = cell(cid)
    if c["kind"] == "payload_kg":
        t = C.TERM[c["term"]]
        txt = C.level_text(t, c["level"])
        sw = C.CAMPAIGN / "levels" / Path(c["level_file"]).name
        if not sw.is_file() or sw.read_text() != txt:
            raise Stop(f"T13 level file {sw} differs from cells.level_text", 5)
        lf = Path(c["level_file"])
        lf.parent.mkdir(parents=True, exist_ok=True)
        if not lf.is_file() or lf.read_text() != txt:
            lf.write_text(txt)


def strip_noop_problems() -> list[str]:
    """The shared L000 rows run without TP_STRIP_KEYS: that is identical to Task 4's stripped row only while no
    refused runner key is in the row's env file."""
    bad = []
    keys = set(refused_keys())
    for rel in {p3b_plan()["env"], *p3b_plan()["row_env"].values()}:
        env = tp().read_env_file(T4 / "F19" / rel)
        hit = sorted(keys & set(env))
        if hit:
            bad.append(f"F19/{rel} carries refused keys {hit}")
    return bad


# ============================================================================== rows
def parse_row(key: str) -> dict:
    m = re.match(r"^([a-z_0-9]+)@(\d+)$", key)
    if not m:
        raise ValueError(f"row key {key!r} is not family@es")
    return {"family": m.group(1), "es": int(m.group(2)), "key": key}


def cell_name(pol: dict, cid: str, suffix: str = "") -> str:
    return f"{'RSDT' if pol['role'] == 'teacher' else 'RSDS'}_{pol['sha8']}_{cid}{suffix}"


def sel_name(pol: dict, kg: int | None) -> str:
    return f"{'RSDSELT' if pol['role'] == 'teacher' else 'RSDSEL'}_{pol['sha8']}_L{int(kg or 0):03d}"


def cell_rows(pol: dict, cid: str, es: int, suffix: str = "") -> list[dict]:
    """The two gate rows (suite, field_light) of a cell-seed, composed exactly as the sweep's ``rows_for`` (the
    sweep's ROWS: env file = cells.env_text, manifest, scenario, ``--task`` + suite ``--trained-categories``, then
    ``--policy-role``)."""
    C = sweep().C
    c = cell(cid)
    out = []
    for fam, sp in C.ROWS.items():
        out.append({
            "kind": "cell", "cell": cid, "suffix": suffix, "family": fam, "es": int(es), "policy": pol["name"],
            "role": pol["role"], "name": cell_name(pol, cid, suffix), "ckpt": pol["checkpoint"],
            "ckpt_sha": pol["sha256"], "env_file": str(env_file(cid, fam)), "level_file": c["level_file"],
            "level_kg": c["level_kg"], "manifest": str(sp["manifest"]), "scenario": sp["scenario"],
            "args": [*sp["args"], "--policy-role", pol["role"]], "strip": [],
            "trained": trained() if fam == "suite" else None, "rod_s": c["rod_s"], "zero_depth": False,
            "key": f"{fam}@{int(es)}"})
    return out


def sel_row(pol: dict, key: str, level: int | None = None, *, t4: Path = T4, omni: Path = OMNI,
            name: str | None = None) -> dict:
    """One selection row as Task 4's ``row_job`` composes it (pipeline.py 1299-1314 with env_stage p3b): the p3b row
    env (``plan.json`` row_env, else env), the row family's args + the sel stage's student / teacher args + the
    suite's ``--trained-categories``, TP_STRIP_KEYS = the refused runner knobs, a level row at ``level`` kg (90,
    set explicitly). ``t4`` / ``omni``: the checkout whose paths the row names (the replay passes the main
    checkout, which Task 4's recorded rows name)."""
    if level is not None and int(level) <= 0:
        # sel_name(pol, 0) == sel_name(pol, None): a 0-kg "level" row would share the no-level row's stamp dir
        raise PolicyError(f"sel_row {key}: level {level!r} kg -- a level row needs a positive kg (CHARTER 3.4: 90)")
    r = parse_row(key)
    fam = r["family"]
    spec_r = spec()["rows"][fam]
    pe = p3b_plan()
    env_rel = pe["row_env"].get(fam, pe["env"])
    st = sel_spec()
    args = list(spec_r.get("args") or []) + list(st["student_args" if pol["role"] == "student" else "teacher_args"])
    if spec_r.get("trained_categories") and pe.get("trained_csv"):
        args += ["--trained-categories", pe["trained_csv"]]
    lf = None if level is None else str(t4 / "F19" / plan()["levels"][str(int(level))])
    return {"kind": "sel", "cell": None, "suffix": "", "family": fam, "es": r["es"], "policy": pol["name"],
            "role": pol["role"], "name": name or sel_name(pol, level), "ckpt": pol["checkpoint"],
            "ckpt_sha": pol["sha256"], "env_file": str(t4 / "F19" / env_rel), "level_file": lf,
            "level_kg": None if level is None else int(level), "manifest": str(omni / spec_r["manifest"]),
            "scenario": spec_r["scenario"], "args": args, "strip": refused_keys(),
            "trained": (list(pe["trained"]) if spec_r.get("trained_categories") else None), "rod_s": 1.0,
            "zero_depth": False, "key": key + (f"@L{int(level):03d}" if level is not None else "")}


def row_id(row: dict) -> str:
    return f"{row['name']} {row['scenario']} es{row['es']}"


def stamp_dir(name: str, scenario: str, es: int, root: Path | None = None) -> Path | None:
    """The newest COMPLETE stamp (``tp_row.json`` is written last) of a row, or None."""
    base = Path(root or TP_OUT) / "lr" / name / scenario / f"seed{int(es):03d}"
    if not base.is_dir():
        return None
    done = sorted(p for p in base.iterdir() if (p / "tp_row.json").is_file())
    return done[-1] if done else None


def row_stamp(row: dict, root: Path | None = None) -> Path | None:
    return stamp_dir(row["name"], row["scenario"], row["es"], root)


# ============================================================================== stamp checks
def expect_env(row: dict) -> dict:
    """The composed env the stamp's ``run_meta.env_vars`` must equal key for key: the env file (gate_row's
    ``^KRABBY_`` lines) minus the stripped keys, plus the level file."""
    env = {k: v for k, v in tp().read_env_file(Path(row["env_file"])).items() if k.startswith("KRABBY_")}
    for k in row.get("strip") or []:
        env.pop(k, None)
    if row.get("level_file"):
        env.update({k: v for k, v in tp().read_env_file(Path(row["level_file"])).items()
                    if k.startswith("KRABBY_PAYLOAD_")})
    return env


def expected_stripped(row: dict) -> list:
    keys = set(tp().read_env_file(Path(row["env_file"])))
    return [k for k in (row.get("strip") or []) if k in keys]


def dose_read(stamp: Path) -> dict:
    """max |rod_speed| over every raw npz of a stamp: (hip cols 0-5, knee cols 6-11)."""
    import numpy as np  # noqa: PLC0415

    files = sorted(glob.glob(str(Path(stamp) / "raw" / "episode_*.npz")))
    if not files:
        return {"n": 0, "why": "no raw/episode_*.npz"}
    h = k = 0.0
    for f in files:
        with np.load(f) as z:
            if "rod_speed" not in z.files:
                return {"n": len(files), "why": f"{Path(f).name} has no rod_speed"}
            r = np.abs(z["rod_speed"].astype(np.float64))
            if r.ndim != 2 or r.shape[1] != 12:
                return {"n": len(files), "why": f"{Path(f).name} rod_speed shape {r.shape}"}
            h = max(h, float(r[:, :6].max())) if r.size else h
            k = max(k, float(r[:, 6:].max())) if r.size else k
    return {"n": len(files), "hip": h, "knee": k}


def dose_problems(stamp: Path, s: float, lab: str) -> list[str]:
    d = dose_read(stamp)
    if "why" in d:
        return [f"{lab}: dose unreadable ({d['why']})"]
    want = (s * ROD_RATED[0], s * ROD_RATED[1])
    bad = []
    for j, (got, w) in enumerate(((d["hip"], want[0]), (d["knee"], want[1]))):
        if abs(got - w) > DOSE_TOL:
            bad.append(f"{lab}: dose {'hip' if j == 0 else 'knee'} max |rod_speed| {got:.7f} != {s:g} x "
                       f"{ROD_RATED[j]} = {w:.7f} m/s (tol {DOSE_TOL:g})")
    return bad


def own_pin(stamp: Path) -> dict:
    cs = read_json(Path(stamp) / "code_state.json", {}) or {}
    return {"code_sha": cs.get("code_sha"), "eval_sha": cs.get("eval_sha")}


def row_problems(stamp: Path, row: dict, pin: dict | None, *, dose: bool = True, relax_pin: bool = False) -> list:
    """Every launch check of one stamp (empty = valid). ``relax_pin`` (the replay only): the stamp is compared to its
    own code_state hashes, so only the pin equality is relaxed (dirty is still checked)."""
    G = gr()
    stamp = Path(stamp)
    lab = row_id(row)
    if relax_pin:
        pin = own_pin(stamp)
    exp = {"label": lab, "stage": "p3b", "pin": pin or {}, "source": "row", "env": expect_env(row),
           "carry": spec()["carry_env"], "ckpt_sha": row["ckpt_sha"], "trained": row["trained"], "task": TASK}
    if row.get("check_payload", True):
        exp["level_kg"] = row["level_kg"]
    out = list(G.stamp_problems(stamp, exp))
    if pin is None and not relax_pin:
        out.append(f"{lab}: no pin to check against")
    if not (stamp / "run_meta.json").is_file():
        return out
    meta = json.loads((stamp / "run_meta.json").read_text())
    if meta.get("role") != row["role"]:
        out.append(f"{lab}: run_meta.role {meta.get('role')!r} != {row['role']!r}")
    stu = meta.get("student")
    if row["role"] == "student":
        want = bool(row.get("zero_depth", False))
        if not isinstance(stu, dict) or stu.get("zero_depth") is not want:
            out.append(f"{lab}: run_meta.student.zero_depth {(stu or {}).get('zero_depth')!r} (want {want})")
    elif stu not in (None, {}):
        out.append(f"{lab}: run_meta.student present on a teacher row")
    seed = (meta.get("env_overrides") or {}).get("seed")
    if seed != row["es"]:
        out.append(f"{lab}: env seed {seed!r} != {row['es']}")
    if meta.get("scenario_id") != row["scenario"]:
        out.append(f"{lab}: scenario_id {meta.get('scenario_id')!r} != {row['scenario']!r}")
    dr = sorted(k for k in (meta.get("env_vars") or {}) if str(k).startswith("KRABBY_DR_"))
    if dr:
        out.append(f"{lab}: eval env carries {dr} (no KRABBY_DR_* on an eval row)")
    tpr = read_json(stamp / "tp_row.json")
    if tpr is None:
        out.append(f"{lab}: no tp_row.json")
    else:
        want_tp = {"by": "TP gate_row.sh", "name": row["name"], "checkpoint": row["ckpt"],
                   "env_file": row["env_file"], "level_file": row["level_file"] or None, "manifest": row["manifest"],
                   "scenario": row["scenario"], "env_seed": int(row["es"]), "args": list(row["args"]),
                   "stripped_keys": expected_stripped(row), "rc": 0}
        for k, v in want_tp.items():
            if tpr.get(k) != v:
                out.append(f"{lab}: tp_row.{k} {str(tpr.get(k))[:90]!r} != {str(v)[:90]!r}")
    if dose:
        out += dose_problems(stamp, float(row.get("rod_s", 1.0)), lab)
    return out


# ============================================================================== cell-seed reads
_LINE_EQ: dict = {}


def _es_lines_equal() -> None:
    """es3 is read with the es1 line ids: assert every p3a.stable90 es1 line equals its es2 twin."""
    if _LINE_EQ:
        return
    by = {x["id"]: x for x in pr().stable_lines(th_doc())}
    for lid, x in by.items():
        if lid.endswith(".es1"):
            y = by.get(lid[:-1] + "2")
            if y is None or (y["line"], y["direction"]) != (x["line"], x["direction"]):
                raise RuntimeError(f"stable90 line {lid} differs from its es2 twin; an es3 read cannot reuse it")
    _LINE_EQ["ok"] = True


def read_lines(sd: Path, fd: Path, es: int) -> dict:
    """The sweep's per-seed record (stress_read.read_stamps: 17 p3a.stable90 lines + field_light completion)."""
    SR = sweep().SR
    if int(es) in (1, 2):
        return SR.read_stamps(Path(sd), Path(fd), int(es))
    _es_lines_equal()
    rec = SR.read_stamps(Path(sd), Path(fd), 1)
    rec["es"] = int(es)
    rec["read_as"] = "es1 line ids (es1 == es2 line values)"
    return rec


def suite_extras(sd: Path) -> dict:
    """WATCH reads of a suite stamp: per-category n (the near-line survival count), stand survival, side ratios per
    hand (``by_category.side.by_hand.{l,r}.ratio.median``)."""
    cats = thm().suite_categories(Path(sd))
    a = json.loads((Path(sd) / "scenario_metrics.json").read_text())["aggregate"]
    bh = (((a.get("suite") or {}).get("by_category") or {}).get("side") or {}).get("by_hand") or {}
    return {"n": {c: v.get("n") for c, v in cats.items() if not c.startswith("_")},
            "stand_survival": (cats.get("stand") or {}).get("survival"),
            "side_ratio_by_hand": {h: ((bh.get(h) or {}).get("ratio") or {}).get("median") for h in ("l", "r")}}


def read_cellseed(rows: list, *, pin: dict | None, relax_pin: bool = False, dose: bool = True,
                  root: Path | None = None) -> dict | None:
    """One cell-seed (suite + field_light rows): None while either stamp is missing; ``pass`` None when a launch
    check fails (``problems``), never a reading."""
    by = {r["family"]: r for r in rows}
    sr, fr = by["suite"], by["field_light"]
    sd, fd = row_stamp(sr, root), row_stamp(fr, root)
    if sd is None or fd is None:
        return None
    rec = read_lines(sd, fd, sr["es"])
    probs = (row_problems(sd, sr, pin, dose=dose, relax_pin=relax_pin)
             + row_problems(fd, fr, pin, dose=dose, relax_pin=relax_pin))
    rec["problems"] = probs
    rec["line_pass"] = rec["pass"]
    if probs:
        rec["pass"] = None
    rec["extras"] = suite_extras(sd)
    rec["name"] = sr["name"]
    return rec


def values(rec: dict | None) -> dict | None:
    return None if rec is None else {k: v["value"] for k, v in sorted(rec["lines"].items())}


def seeds_of(pol: dict, cid: str, *, pin, ess=(1, 2, 3), suffix: str = "", root=None, rows_fn=None) -> dict:
    out = {}
    for es in ess:
        rows = (rows_fn or (lambda e: cell_rows(pol, cid, e, suffix)))(es)
        r = read_cellseed(rows, pin=pin, root=root)
        if r is not None:
            out[es] = r
    return out


# ============================================================================== the tracking carve-out
def carve_out(rec: dict, ref: dict, cat: str, es: int, its: float) -> dict:
    """CAPABILITY-LIMITED (CHARTER 3.3 S): a failing tracking line passes when the head's track_err (|r - 1|) is no
    worse than the in-pin reference's by more than max(R, 3 SE_diff, |drift| its / 5000) -- thresholds v1.2's
    per-category, per-seed 0-kg margin: R = the 99 % guard R of cat.<c>.ratio, SE_diff the env-clustered paired
    bootstrap (thresholds.paired_suite_se on |r - 1|), drift = Task 4's 0-kg Walk-Play calibration."""
    TH = thm()
    R, Rg = resolution()
    A = TH.suite_episodes(Path(rec["stamps"]["suite"]))
    B = TH.suite_episodes(Path(ref["stamps"]["suite"]))
    rows = TH.zero_kg_suite_rows({es: A}, {es: B}, [cat], R, Rg, translate="track_err")
    row = next(r for r in rows if r["statistic"] == f"track_err {cat}" and not r["watch"])
    dt = drift_table()
    if int(es) in (1, 2):
        d = dt.get(f"track_err {cat}|suite es{es}")
        dsrc = f"track_err {cat}|suite es{es}"
    else:
        ds = [dt.get(f"track_err {cat}|suite es{k}") for k in (1, 2)]
        d = max(abs(x) for x in ds if x is not None) if any(x is not None for x in ds) else None
        dsrc = f"max |drift| es1 / es2 (no es{es} drift)"
    out = {"line": f"tracking.{cat}", "es": int(es), "head": row.get("head"), "ref": row.get("p2a"),
           "worse_by": row.get("worse_by"), "se_diff": row.get("se_diff"), "R": row.get("R"),
           "R_source": row.get("R_source"), "drift": d, "drift_source": dsrc, "its_past_p2a": its,
           "label": "CAPABILITY-LIMITED", "calibration": "R and drift reused from Task 4's 0-kg Walk-Play calibration"}
    if row.get("missing") or row.get("R") is None or d is None:
        out.update({"pass": False, "why": "missing statistic, R or drift"})
        return out
    m = TH.zero_kg_margin(row["R"], row.get("se_diff") or 0.0, d, its)
    out["margin"] = m
    out["pass"] = bool(row["worse_by"] <= m["margin"] + 1e-12)
    return out


# ============================================================================== per-cell evaluation and es3
def seed_eval(rec: dict, ref: dict | None, *, carve: bool, l135: bool, its: float) -> dict:
    failing, carved, shared, excluded = [], [], [], []
    ref_ok = ref is not None and ref.get("pass") is not None
    for k, ln in sorted(rec["lines"].items()):
        if l135 and k == "field_light_completion":
            excluded.append(k)
            continue
        if ln["pass"]:
            continue
        if l135 and ref_ok and not (ref["lines"].get(k) or {}).get("pass", True):
            shared.append(k)                         # a shared capability limit: recorded, not a failure
            continue
        if carve and k.startswith("tracking.") and ref_ok:
            co = carve_out(rec, ref, k.split(".", 1)[1], rec["es"], its)
            carved.append(co)
            if co["pass"]:
                continue
        failing.append(k)
    if rec.get("n_lines") != 18:
        failing.append(f"n_lines={rec.get('n_lines')}")
    return {"pass": not failing, "failing": failing, "carved": carved, "shared_limit": shared, "excluded": excluded}


def near_line(fails: list, recs: dict, l135: bool) -> dict:
    """CHARTER 3.3 es3: the cell's only failing line is completion at 84-86/100, a survival count within one episode
    of its line, or a roll / pitch RMS within 0.003 of its line on one seed; every other line passes on both seeds."""
    keys = sorted({k for _, k in fails})
    if len(keys) != 1:
        return {"near": False, "why": f"{len(keys)} failing line keys {keys}"}
    k = keys[0]
    vals = []
    for es, _ in fails:
        ln = recs[es]["lines"].get(k)
        if ln is None:
            return {"near": False, "why": f"{k} unread"}
        v, line = ln["value"], ln["line"]
        vals.append((es, v, line))
        if v is None:
            return {"near": False, "why": f"{k} es{es} has no value"}
        if k == "field_light_completion":
            if l135 or not (0.84 - 1e-9 <= v <= 0.86 + 1e-9):
                return {"near": False, "why": f"completion {v} outside 0.84-0.86"}
        elif k.startswith("survival."):
            n = (recs[es]["extras"]["n"] or {}).get(k.split(".", 1)[1])
            if not n or abs(v * n - line * n) > 1.0 + 1e-9:
                return {"near": False, "why": f"{k} es{es} {v} x n={n} more than one episode from {line}"}
        elif k.startswith(("roll_rms.", "pitch_rms.")):
            if len(fails) != 1 or abs(v - line) > 0.003 + 1e-12:
                return {"near": False, "why": f"{k}: {len(fails)} seed(s), |{v} - {line}| (want one seed, <= 0.003)"}
        else:
            return {"near": False, "why": f"{k} is not an es3-eligible line"}
    return {"near": True, "key": k, "values": vals}


def cell_eval(cid: str, recs: dict, ref: dict | None, *, carve: bool, l135: bool, its: float) -> dict:
    """A cell's status for the gate: pass | fail | es3_due | missing, from es1 + es2 (+ es3 once)."""
    ref = ref or {}
    out = {"cell": cid, "carve": carve, "l135": l135, "seeds": {}, "fail_keys": [], "missing": []}
    for es in (1, 2):
        r = recs.get(es)
        if r is None:
            out["missing"].append(f"{cid} es{es}: stamps missing")
        elif r["pass"] is None:
            out["missing"].append(f"{cid} es{es}: launch checks fail {r['problems'][:3]}")
    if out["missing"]:
        out["status"] = "missing"
        return out
    ev = {es: seed_eval(recs[es], ref.get(es), carve=carve, l135=l135, its=its) for es in (1, 2)}
    out["seeds"] = {es: {**e, "values": values(recs[es])} for es, e in ev.items()}
    fails = [(es, k) for es in (1, 2) for k in ev[es]["failing"]]
    out["fail_keys"] = sorted({k for _, k in fails})
    if not fails:
        out["status"] = "pass"
        return out
    nl = near_line(fails, recs, l135)
    out["near_line"] = nl
    if not nl["near"]:
        out["status"] = "fail"
        return out
    r3 = recs.get(3)
    if r3 is None or r3["pass"] is None:
        out["status"] = "es3_due"
        if r3 is not None:
            out["missing"].append(f"{cid} es3: launch checks fail {r3['problems'][:3]}")
        return out
    e3 = seed_eval(r3, None, carve=carve, l135=l135, its=its)
    out["seeds"][3] = {**e3, "values": values(r3)}
    passing = sum(1 for e in (ev[1], ev[2], e3) if e["pass"])
    fl = [Path(recs[es]["stamps"]["field_light"]) for es in (1, 2, 3)]
    pooled = thm().field_completion(fl)["rate"]
    ok = passing >= 2 and (l135 or pooled >= COMPLETION_LINE - 1e-12)
    out["es3"] = {"seeds_passing": passing, "pooled_completion_3": pooled,
                  "completion_clause": "dropped (L135)" if l135 else f">= {COMPLETION_LINE}", "pass": ok}
    out["status"] = "pass" if ok else "fail"
    return out


# ============================================================================== pooled paired completion
def paired_completion(a: list, b: list) -> dict:
    """Pooled paired field_light completion of head ``a`` vs reference ``b``: lists of records in launch order
    (es1, es2[, the relaunch's es1, es2]); select_student.completion_pairs semantics (paired on (launch index,
    env_index); m = max(0.05, 2 SE_diff) -- the sel stage's completion_margin, split 0)."""
    if any(r is None or r.get("pass") is None for r in list(a) + list(b)) or len(a) != len(b) or not a:
        return {"missing": True, "pass": False, "why": "a launch is unread or invalid on either side"}
    TH = thm()
    A, B = {}, {}
    for i, (ra, rb) in enumerate(zip(a, b)):
        A.update({(i, e): v for e, v in TH.field_episodes(Path(ra["stamps"]["field_light"])).items()})
        B.update({(i, e): v for e, v in TH.field_episodes(Path(rb["stamps"]["field_light"])).items()})
    res = ss().completion_pairs({"field_light": A}, {"field_light": B}, completion_cfg())
    fl = dict(next(x for x in res if x["row"] == "field_light"))
    if fl.get("missing"):
        return {**fl, "pass": False}
    fl["diff"] = fl["student"] - fl["tstar"]
    fl["head"], fl["ref"] = fl.pop("student"), fl.pop("tstar")
    fl["n"] = len(A)
    fl["noninferior"] = fl.pop("pass")
    fl["beats_by_m"] = bool(fl["diff"] >= fl["margin"] - 1e-12)
    fl["pass"] = fl["noninferior"]
    return fl


def recs_list(recs: dict, ess=(1, 2)) -> list:
    return [recs.get(es) for es in ess]


# ============================================================================== R: the paired 0-kg statistic
def r_check(head: dict, ref: dict, its_past_p2a: float, trained_list: list | None = None) -> dict:
    """The campaign's R wrapper: gate_read.zero_kg_check (thresholds v1.2, track_err) of ``head`` against ``ref`` on
    the zero_kg row list, with the campaign's trained list (p3b's), ``its_past_p2a`` and Task 4's resolution files.
    ``head`` / ``ref``: {suite: {es: stamp}, side22: stamp, field_hard / field_light / ring_obst / field_turn:
    [stamps in env-seed order]}."""
    tl = trained_list if trained_list is not None else trained()
    tt = [c for c in tl if c in gr().TRANSLATE]
    res = gr().zero_kg_check(head, ref, tt, th_view(), its_past_p2a, resolution_files())
    res["inputs"] = {"its_past_p2a": its_past_p2a, "trained_translate": tt, "resolution_files": resolution_files(),
                     "thresholds": TH_PATH.name,
                     "calibration": "R and drift reused from Task 4's 0-kg Walk-Play calibration (thresholds v1.2)"}
    return res


def zero_kg_rows(pol: dict) -> list[dict]:
    """The zero_kg row list of a teacher (L000): suite@1/2 + field_light@1/2 = its L000 nominal cell rows, the other
    seven = its RSDSELT selection rows."""
    rows = []
    for es in (1, 2):
        rows += cell_rows(pol, NOM["L000"], es)
    rows += [sel_row(pol, k) for k in SEVEN]
    return rows


def zero_kg_stamps(rows: list, *, pin, root=None, dose=True, relax_pin=False) -> tuple:
    """{suite, side22, field rows} from rows (keys family@es) + every launch problem / missing stamp."""
    out = {"suite": {}, "side22": None, "field_hard": [], "field_light": [], "ring_obst": [], "field_turn": []}
    probs = []
    by = {}
    for r in rows:
        st = row_stamp(r, root)
        if st is None:
            probs.append(f"{row_id(r)}: stamp missing")
            continue
        p = row_problems(st, r, pin, dose=dose, relax_pin=relax_pin)
        probs += p
        by[(r["family"], r["es"])] = st
    for (fam, es), st in sorted(by.items()):
        if fam == "suite":
            out["suite"][es] = str(st)
        elif fam == "side22":
            out["side22"] = str(st)
        elif fam in out:
            out[fam].append(str(st))
    return out, probs


# ============================================================================== bridge
def bridge_read(pol_name: str = "TSTAR", *, pin, root=None) -> dict:
    """CHARTER 3.3 bridge (a code-path check only): each cell-seed's 18 line values (first launch, then the one
    relaunch ``_RL``) must equal the sweep's DRST (teacher) / DRS (student) stamp of that cell-seed, read through
    stress_read.read_stamps and validated against the sweep's own pin."""
    C, SR = sweep().C, sweep().SR
    pol = policy(pol_name)
    role = pol["role"]
    plan_ = BRIDGE_TEACHER if role == "teacher" else BRIDGE_STUDENT
    out = {"policy": pol_name, "sha256": pol["sha256"], "role": role, "cell_seeds": {}, "sweep_pin": SR.launch_pin()}
    for cid, es in plan_:
        c = cell(cid)
        term = None if c["term"] is None else C.TERM[c["term"]]
        sname = C.gate_name(cid, role)
        swp = SR.cellseed(sname, es, term, c["level"], c["base"], policy=role)
        first = read_cellseed(cell_rows(pol, cid, es), pin=pin, root=root)
        rel = read_cellseed(cell_rows(pol, cid, es, RELAUNCH), pin=pin, root=root)
        sv = values(swp) if swp is not None and swp.get("launch_error") is None else None
        ent = {"sweep_name": sname, "sweep_stamps": None if swp is None else swp["stamps"],
               "sweep_launch_error": None if swp is None else swp.get("launch_error"),
               "first": None if first is None else {"stamps": first["stamps"], "problems": first["problems"],
                                                    "values": values(first)},
               "relaunch": None if rel is None else {"stamps": rel["stamps"], "problems": rel["problems"],
                                                     "values": values(rel)}}
        if sv is None:
            status = "sweep_unread"
        elif first is None:
            status = "missing"
        elif first["problems"]:
            status = "invalid_first"
        elif values(first) == sv and len(sv) == 18:
            status = "bitwise_first"
        elif rel is None:
            status = "relaunch_due"
        elif rel["problems"]:
            status = "invalid_relaunch"
        elif values(rel) == sv:
            status = "bitwise_relaunch"
        else:
            status = "STOP_diagnose"
        if first is not None and sv is not None:
            ent["diff_first"] = {k: [values(first).get(k), sv.get(k)] for k in sorted(set(sv) | set(values(first)))
                                 if values(first).get(k) != sv.get(k)}
        if rel is not None and sv is not None:
            ent["diff_relaunch"] = {k: [values(rel).get(k), sv.get(k)] for k in sorted(set(sv) | set(values(rel)))
                                    if values(rel).get(k) != sv.get(k)}
        ent["status"] = status
        out["cell_seeds"][f"{cid}@es{es}"] = ent
    sts = [e["status"] for e in out["cell_seeds"].values()]
    out["relaunch_due"] = [k for k, e in out["cell_seeds"].items() if e["status"] == "relaunch_due"]
    if all(s in ("bitwise_first", "bitwise_relaunch") for s in sts):
        out["verdict"] = "PASS"
    elif any(s == "STOP_diagnose" for s in sts):
        out["verdict"] = "STOP"
        out["why"] = ("a cell-seed differs from the sweep on the first launch and on its relaunch: diagnose on CPU "
                      "(cfg dumps, per-stamp dose, first divergent step); a code-path difference stops the campaign, "
                      "a diagnosed class flip is recorded and G1 proceeds on the in-pin references")
    elif out["relaunch_due"]:
        out["verdict"] = "RELAUNCH_DUE"
    else:
        out["verdict"] = "INCOMPLETE"
    return out


# ============================================================================== closing
def closing_read(pol: dict, ref_rows_fn, *, pin, root=None) -> dict:
    """The closing re-run (L090 nominal es1, launch Z) against the first L090 nominal es1 (its class)."""
    z = read_cellseed(cell_rows(pol, CLOSING_CELL, 1), pin=pin, root=root)
    a = read_cellseed(ref_rows_fn(1), pin=pin, root=root)
    if z is None or a is None:
        st = "unread"
    elif z["pass"] is None or a["pass"] is None:
        st = "launch_error"
    else:
        st = "match" if values(z) == values(a) else "mismatch"
    return {"status": st, "flag": st != "match", "closing": values(z), "reference": values(a),
            "diff": (None if z is None or a is None else
                     {k: [values(z).get(k), values(a).get(k)] for k in values(a) if values(z).get(k) != values(a).get(k)})}


# ============================================================================== decisions
FAIL_REGRESSION, FAIL_STRESS, UNATTRIBUTED, EXTEND, PASS = (
    "FAIL-REGRESSION", "FAIL-STRESS", "UNATTRIBUTED", "EXTEND", "PASS")


def g1_decide(st: dict) -> tuple:
    """The CHARTER 3.3 verdict table on resolved statuses. ``st``: s_cells / n_cells {cid: (pass, fail_keys)},
    n_pooled_ok, r_ok, trig (TDR - old >= m at x0.7), a_ok (None = TCTL not read), tctl_moved (TCTL - old >= m at
    x0.7), half_gap (x0.7 pooled >= old + (0.85 - old) / 2), extended. Returns (verdict, notes)."""
    notes = []
    n_ok = all(p for p, _ in st["n_cells"].values()) and st["n_pooled_ok"]
    if not n_ok or not st["r_ok"]:
        return FAIL_REGRESSION, [x for x in ("N fails" if not n_ok else "", "R fails" if not st["r_ok"] else "") if x]
    s_ok = all(p for p, _ in st["s_cells"].values())
    if not st["trig"]:
        if s_ok:
            return UNATTRIBUTED, ["DR null at x0.7 (TDR - old < m): the lazy control never ran, A unmeasurable"]
        return FAIL_STRESS, ["DR null: TDR x0.7 not >= old + m (read TDR's DR/rod_speed_* past it 500 first: the "
                             "dose audit)"]
    if st["a_ok"] is None:
        return "TCTL-DUE", ["the lazy control is due (TDR x0.7 beats old T* by >= m; no N / R line has failed)"]
    if s_ok:
        if st["a_ok"]:
            if st["extended"] and st["tctl_moved"]:
                return FAIL_STRESS, ["after EXTEND, PASS also needs TCTL - old < m at x0.7"]
            return PASS, (["A compares TDR(+6000) with TCTL(+3000): length-unmatched"] if st["extended"] else [])
        return UNATTRIBUTED, ["S, N, R pass; A fails (stop + dose audit)"]
    keys = sorted({k for p, ks in st["s_cells"].values() if not p for k in ks})
    only_cs = all(k == "field_light_completion" or k.startswith("survival.") for k in keys)
    if st["extended"]:
        return FAIL_STRESS, [f"final read after EXTEND: S fails on {keys}"]
    if only_cs and st["a_ok"] and st["half_gap"] and st["trig"]:
        if not st["tctl_moved"]:
            return EXTEND, [f"only stress cells fail, on completion / survival ({keys})"]
        return FAIL_STRESS, ["no-DR continuation moved x0.7 by >= m"]
    why = []
    if not only_cs:
        why.append(f"S fails on lines other than completion / survival ({keys})")
    if not st["a_ok"]:
        why.append("A < m")
    if not st["half_gap"]:
        why.append("x0.7 pooled < old + (0.85 - old) / 2")
    return FAIL_STRESS, why


def g2_decide(st: dict) -> tuple:
    """CHARTER 3.4 G2: PASS iff S and N pass; a class-ambiguous pooled N cell is RELAUNCH-DUE until relaunched."""
    s_ok = all(p for p, _ in st["s_cells"].values())
    n_ok = all(p for p, _ in st["n_cells"].values())
    pooled = st["n_pooled"]
    if not s_ok or not n_ok or any(v == "fail" for v in pooled.values()):
        return "FAIL", [x for x in ("S fails" if not s_ok else "", "N fails" if not n_ok else "",
                                    "N pooled fails" if any(v == "fail" for v in pooled.values()) else "") if x]
    if any(v == "ambiguous" for v in pooled.values()):
        return "RELAUNCH-DUE", [f"class-ambiguous pooled shortfall at {[k for k, v in pooled.items() if v == 'ambiguous']}"]
    return PASS, []


def resolve(cells: dict, decide, base: dict, *, pending_status=("es3_due",), cap_left: int) -> dict:
    """Apply ``decide`` over every outcome of the pending cells (es3 due): one verdict for all outcomes stands;
    otherwise PENDING-ES3, or UNRESOLVED when the pending count exceeds the es3 allowance left."""
    pend = [k for k, v in cells.items() if v["status"] in pending_status]
    outs = set()
    allnotes = {}
    for combo in itertools.product((True, False), repeat=len(pend)):
        assign = dict(zip(pend, combo))
        st = dict(base)
        for group in ("s_cells", "n_cells"):
            st[group] = {k: ((assign[k], cells[k]["fail_keys"]) if k in assign else
                             (cells[k]["status"] == "pass", cells[k]["fail_keys"]))
                         for k in base[group]}
        v, n = decide(st)
        outs.add(v)
        allnotes[v] = n
    if len(outs) == 1:
        v = outs.pop()
        return {"verdict": v, "notes": allnotes[v], "pending": pend}
    if len(pend) > cap_left:
        return {"verdict": "UNRESOLVED", "notes": [f"{len(pend)} near-line cells need es3, {cap_left} of the "
                                                   f"{ES3_CAP} cell-seed allowance left"], "pending": pend,
                "possible": sorted(outs)}
    return {"verdict": "PENDING-ES3", "notes": [f"es3 due for {pend}"], "pending": pend, "possible": sorted(outs)}


def es3_used(pols: list, cids: list, *, pin, root=None) -> int:
    n = 0
    for pol in pols:
        for cid in cids:
            if row_stamp(cell_rows(pol, cid, 3)[0], root) is not None:
                n += 1
    return n


# ============================================================================== G1
def watch_of(recs: dict) -> dict:
    return {f"es{es}": {"stand_survival": r["extras"]["stand_survival"],
                        "side_ratio_by_hand": r["extras"]["side_ratio_by_hand"],
                        "completion": (r["lines"].get("field_light_completion") or {}).get("value")}
            for es, r in sorted(recs.items()) if r is not None}


def g1_read(tdr_name: str, old_name: str = "TSTAR", tctl_name: str | None = None, *, extended: bool = False,
            prior_tdr_name: str | None = None, pin=None, root=None) -> dict:
    """CHARTER 3.3: S, N, R, A, WATCH, es3, the lazy-control trigger, the verdict. ``prior_tdr_name`` (the
    post-EXTEND read): the pre-EXTEND TDR, whose es3 cell-seeds count against the same per-gate allowance."""
    pin = pin if pin is not None else launch_pin()
    its = ITS_PAST_P2A["g1_extended" if extended else "g1"]
    T, O = policy(tdr_name), policy(old_name)
    C = policy(tctl_name) if tctl_name else None
    es3_pols = [T] + ([policy(prior_tdr_name)] if prior_tdr_name else [])
    tdr = {cid: seeds_of(T, cid, pin=pin, root=root) for cid in S_CELLS + N_CELLS}
    old = {cid: seeds_of(O, cid, pin=pin, ess=(1, 2), root=root) for cid in REF_CELLS}
    tc = {cid: seeds_of(C, cid, pin=pin, ess=(1, 2), root=root) for cid in TCTL_CELLS} if C else {}
    missing = []
    for cid in REF_CELLS:
        for es in (1, 2):
            r = old[cid].get(es)
            if r is None or r["pass"] is None:
                missing.append(f"old {cid} es{es}" + ("" if r is None else f": {r['problems'][:2]}"))
    S = {cid: cell_eval(cid, tdr[cid], old.get(cid) if cid in CARVE_CELLS else None, carve=cid in CARVE_CELLS,
                        l135=False, its=its) for cid in S_CELLS}
    N = {cid: cell_eval(cid, tdr[cid], old[cid], carve=False, l135=cid == NOM["L135"], its=its) for cid in N_CELLS}
    for e in list(S.values()) + list(N.values()):
        missing += [f"TDR {m}" for m in e["missing"] if e["status"] == "missing"]
    pooled = {cid: paired_completion(recs_list(tdr[cid]), recs_list(old[cid])) for cid in N_CELLS}
    n_pooled_ok = all(p.get("pass") for p in pooled.values())
    zh, ph = zero_kg_stamps(zero_kg_rows(T), pin=pin, root=root)
    zo, po = zero_kg_stamps(zero_kg_rows(O), pin=pin, root=root)
    R = None
    if ph or po:
        missing += [f"R: {p}" for p in (ph + po)[:12]]
    else:
        R = r_check(zh, zo, its)
    x07 = paired_completion(recs_list(tdr[X07]), recs_list(old[X07]))
    trig = bool(x07.get("beats_by_m"))
    a = tctl_old = None
    if C:
        a = paired_completion(recs_list(tdr[X07]), recs_list(tc[X07]))
        tctl_old = paired_completion(recs_list(tc[X07]), recs_list(old[X07]))
        if a.get("missing") or tctl_old.get("missing"):
            missing.append("TCTL x0.7 rows unread / invalid")
    half_gap = (None if x07.get("missing") else
                bool(x07["head"] >= x07["ref"] + 0.5 * (COMPLETION_LINE - x07["ref"]) - 1e-12))
    base = {"s_cells": list(S), "n_cells": list(N), "n_pooled_ok": n_pooled_ok,
            "r_ok": (R is not None and R["verdict"] == "PASS"), "trig": trig,
            "a_ok": (None if a is None or a.get("missing") else a["beats_by_m"]),
            "tctl_moved": (None if tctl_old is None or tctl_old.get("missing") else tctl_old["beats_by_m"]),
            "half_gap": half_gap, "extended": extended}
    used = es3_used(es3_pols, S_CELLS + N_CELLS, pin=pin, root=root)
    cells_ = {**S, **N}
    if missing:
        definite_reg = (R is not None and R["verdict"] != "PASS") or any(
            e["status"] == "fail" for e in N.values()) or any(
            not p.get("missing") and not p.get("pass") for p in pooled.values())
        dec = ({"verdict": FAIL_REGRESSION, "notes": ["N or R fails on the rows read"], "pending": []}
               if definite_reg else {"verdict": "INCOMPLETE", "notes": missing[:20], "pending": []})
    else:
        dec = resolve(cells_, g1_decide, base, cap_left=ES3_CAP - used)
    tctl_due = bool(trig and not C and dec["verdict"] not in (FAIL_REGRESSION,)
                    and R is not None and R["verdict"] == "PASS"
                    and not any(e["status"] == "fail" for e in N.values()) and n_pooled_ok)
    closing = closing_read(T, lambda es: cell_rows(T, NOM["L090"], es), pin=pin, root=root)
    out = {
        "gate": "G1", "read_at": now(), "pin": pin, "extended": extended, "its_past_p2a": its,
        "policies": {"tdr": T, "old": O, "tctl": C},
        "S": S, "N": N, "N_pooled": pooled, "R": R, "R_problems": ph + po,
        "x07_tdr_vs_old": x07, "lazy_control_trigger": trig, "tctl_due": tctl_due,
        "A": a, "A_note": ("TDR(+6000) vs TCTL(+3000): length-unmatched" if extended and a else None),
        "tctl_vs_old_x07": tctl_old, "half_gap": half_gap, "es3_used": used, "es3_cap": ES3_CAP,
        "es3_policies": [p["name"] for p in es3_pols],
        "es3_due": [k for k, v in cells_.items() if v["status"] == "es3_due"],
        "verdict": dec["verdict"], "notes": dec["notes"], "possible": dec.get("possible"),
        "missing": missing,
        "watch": {"L135_completion": {"tdr": {f"es{es}": (r["lines"].get("field_light_completion") or {}).get("value")
                                              for es, r in tdr[NOM["L135"]].items()},
                                      "old": {f"es{es}": (r["lines"].get("field_light_completion") or {}).get("value")
                                              for es, r in old[NOM["L135"]].items()}},
                  "tdr": {cid: watch_of(tdr[cid]) for cid in S_CELLS + N_CELLS},
                  "tctl_L090_nominal": ({f"es{es}": {"pass": r["pass"], "failed": r["failed"]}
                                         for es, r in tc.get(NOM["L090"], {}).items()} if C else None)},
        "closing": closing, "implementation_notes": IMPLEMENTATION_NOTES,
    }
    return out


# ============================================================================== G2
def pick_nominal_rows(pick: dict, base: str):
    """The pick's L000 / L090 nominal = its selection full-block rows (RSDSEL_<sha8>_L000 / _L090)."""
    lvl = None if base == "L000" else 90

    def rows(es):
        return [sel_row(pick, f"suite@{es}", lvl), sel_row(pick, f"field_light@{es}", lvl)]
    return rows


def g2_read(pick_name: str, ref_name: str = "C28000", tdr_name: str | None = None, *, pin=None, root=None) -> dict:
    pin = pin if pin is not None else launch_pin()
    P, Rf = policy(pick_name), policy(ref_name)
    Tt = policy(tdr_name) if tdr_name else None
    its = ITS_PAST_P2A["g2"]
    pick = {cid: seeds_of(P, cid, pin=pin, root=root) for cid in S_CELLS + [NOM["L135"]]}
    for b in ("L000", "L090"):
        pick[NOM[b]] = seeds_of(P, NOM[b], pin=pin, root=root, ess=(1, 2), rows_fn=pick_nominal_rows(P, b))
        pick[NOM[b]].update(seeds_of(P, NOM[b], pin=pin, root=root, ess=(3,)))
    ref = {cid: seeds_of(Rf, cid, pin=pin, ess=(1, 2), root=root) for cid in REF_CELLS}
    missing = [f"ref {cid} es{es}" for cid in REF_CELLS for es in (1, 2)
               if ref[cid].get(es) is None or ref[cid][es]["pass"] is None]
    S = {cid: cell_eval(cid, pick[cid], ref.get(cid) if cid in CARVE_CELLS else None, carve=cid in CARVE_CELLS,
                        l135=False, its=its) for cid in S_CELLS}
    N = {cid: cell_eval(cid, pick[cid], ref[cid], carve=False, l135=cid == NOM["L135"], its=its) for cid in N_CELLS}
    for e in list(S.values()) + list(N.values()):
        missing += [f"pick {m}" for m in e["missing"] if e["status"] == "missing"]
    pooled, pooled_state = {}, {}
    relaunch_used = 0     # charged as one 4 cell-seed block per class-ambiguous cell, however many _RL stamps exist
    for cid in N_CELLS:
        p2 = paired_completion(recs_list(pick[cid]), recs_list(ref[cid]))
        ent = {"two_launch": p2}
        if p2.get("missing"):
            pooled_state[cid] = "missing"
        else:
            short = -p2["diff"]
            ent["shortfall"] = short
            if short <= p2["margin"] + 1e-12:
                pooled_state[cid] = "pass"
            elif short <= p2["margin"] + G2_AMBIG + 1e-12:
                pr_ = seeds_of(P, cid, pin=pin, ess=(1, 2), suffix=RELAUNCH, root=root)
                rr_ = seeds_of(Rf, cid, pin=pin, ess=(1, 2), suffix=RELAUNCH, root=root)
                relaunch_used += 4
                if len(pr_) == 2 and len(rr_) == 2:
                    p4 = paired_completion(recs_list(pick[cid]) + recs_list(pr_), recs_list(ref[cid]) + recs_list(rr_))
                    ent["four_launch"] = p4
                    pooled_state[cid] = ("missing" if p4.get("missing") else
                                         "pass" if -p4["diff"] <= p4["margin"] + 1e-12 else "fail")
                else:
                    pooled_state[cid] = "ambiguous"
            else:
                pooled_state[cid] = "fail"
        pooled[cid] = ent
    es3_only = es3_used([P], S_CELLS + N_CELLS, pin=pin, root=root)
    used = es3_only + relaunch_used
    cells_ = {**S, **N}
    base = {"s_cells": list(S), "n_cells": list(N), "n_pooled": pooled_state}
    if missing or any(v == "missing" for v in pooled_state.values()):
        definite = any(e["status"] == "fail" for e in cells_.values()) or any(v == "fail" for v in pooled_state.values())
        dec = ({"verdict": "FAIL", "notes": ["a line or pooled read fails on the rows read"], "pending": []}
               if definite else {"verdict": "INCOMPLETE", "notes": missing[:20], "pending": []})
    else:
        dec = resolve(cells_, g2_decide, base, cap_left=ES3_CAP - used)
        if dec["verdict"] == "RELAUNCH-DUE" and used > ES3_CAP:
            dec = {"verdict": "UNRESOLVED", "notes": [f"a class-ambiguous relaunch needs 4 cell-seeds, "
                                                      f"{ES3_CAP - es3_only} left"],
                   "pending": dec["pending"]}
    # WATCH cells against the sweep's DRS stamps (cross-pin)
    Cm, SR = sweep().C, sweep().SR
    watch_cells = {}
    for cid in G2_WATCH:
        c = cell(cid)
        mine = seeds_of(P, cid, pin=pin, ess=(1, 2), root=root)
        sw = {es: SR.cellseed(Cm.gate_name(cid), es, Cm.TERM[c["term"]], c["level"], c["base"]) for es in (1, 2)}
        watch_cells[cid] = {f"es{es}": {"pick": None if es not in mine else {"pass": mine[es]["pass"],
                                                                             "failed": mine[es]["failed"],
                                                                             "values": values(mine[es])},
                                        "sweep_c28000": None if sw[es] is None else {"pass": sw[es]["pass"],
                                                                                    "failed": sw[es]["failed"],
                                                                                    "values": values(sw[es])}}
                            for es in (1, 2)}
    rw = g2_r_watch(P, pin=pin, root=root)
    diag = None
    if Tt is not None:
        tx = seeds_of(Tt, X07, pin=pin, ess=(1, 2), root=root)
        diag = {"pick_vs_tdr_x07": paired_completion(recs_list(pick[X07]), recs_list(tx)),
                "pick": {es: values(r) for es, r in pick[X07].items()},
                "tdr": {es: values(r) for es, r in tx.items()}}
    closing = closing_read(P, pick_nominal_rows(P, "L090"), pin=pin, root=root)
    return {"gate": "G2", "read_at": now(), "pin": pin, "its_past_p2a": its,
            "policies": {"pick": P, "ref": Rf, "tdr": Tt}, "S": S, "N": N, "N_pooled": pooled,
            "N_pooled_state": pooled_state, "es3_used": used, "es3_cap": ES3_CAP, "es3_only": es3_only,
            "relaunch_charged": relaunch_used,
            "es3_due": [k for k, v in cells_.items() if v["status"] == "es3_due"],
            "relaunch_due": [k for k, v in pooled_state.items() if v == "ambiguous"],
            "verdict": dec["verdict"], "notes": dec["notes"], "possible": dec.get("possible"), "missing": missing,
            "watch": {"cells_vs_sweep": watch_cells,
                      "L135_completion": {f"es{es}": (r["lines"].get("field_light_completion") or {}).get("value")
                                          for es, r in pick[NOM["L135"]].items()},
                      "pick": {cid: watch_of(pick[cid]) for cid in S_CELLS + N_CELLS}},
            "R_watch": rw, "diagnostic": diag, "closing": closing, "implementation_notes": IMPLEMENTATION_NOTES}


def t4_c28000_block() -> dict:
    """c28000's Task 4 full block stamps (T4F19_sel_c28000, cross-pin)."""
    rows = {}
    for key in sel_spec()["full_block_rows"]:
        r = parse_row(key)
        sc = spec()["rows"][r["family"]]["scenario"]
        rows[key] = stamp_dir("T4F19_sel_c28000", sc, r["es"], T4_TP_OUT)
    return rows


def g2_r_watch(P: dict, *, pin, root=None) -> dict:
    """CHARTER 3.4 R (WATCH, cross-pin): the pick's full-block field_hard / ring_obst / field_turn completion and its
    0-kg track_err against c28000's Task 4 full block; a row below c28000 - m is flagged."""
    TH = thm()
    mine = {k: row_stamp(sel_row(P, k), root) for k in sel_spec()["full_block_rows"]}
    t4 = t4_c28000_block()
    if any(v is None for v in list(mine.values()) + list(t4.values())):
        return {"missing": sorted(k for k, v in list(mine.items()) + [("T4:" + a, b) for a, b in t4.items()]
                                  if v is None)}
    a, b = {}, {}
    for row in ("field_hard", "ring_obst", "field_turn"):
        ea, eb = {}, {}
        ks = sorted(k for k in mine if k.startswith(row + "@"))
        for i, k in enumerate(ks):
            ea.update({(i, e): v for e, v in TH.field_episodes(mine[k]).items()})
            eb.update({(i, e): v for e, v in TH.field_episodes(t4[k]).items()})
        a[row], b[row] = ea, eb
    pairs = [p for p in ss().completion_pairs(a, b, completion_cfg()) if p["row"] in a]
    sa = {int(k.split("@")[1]): TH.suite_episodes(Path(v)) for k, v in mine.items() if k.startswith("suite@")}
    sb = {int(k.split("@")[1]): TH.suite_episodes(Path(v)) for k, v in t4.items() if k.startswith("suite@")}
    tw = ss().tracking_watch(sa, sb, trained())
    flags = [f"{p['row']} completion {p.get('student')} < c28000 {p.get('tstar')} - m {p.get('margin')}"
             for p in pairs if not p.get("pass")] + [f["id"] for f in tw["flags"]]
    return {"completion": pairs, "tracking": tw, "flags": flags, "cross_pin": True}


# ============================================================================== selection (pipeline.py 2244-2410)
def prescreen_read(cand: dict, stamps: dict, lstamps: dict) -> dict:
    """pipeline.py ex_select's prescreen record (2273-2297), line for line."""
    G, th = gr(), th_view()
    st = sel_spec()
    tl = trained()
    tt = [c for c in tl if c in G.TRANSLATE]
    fl = G.floor_reads("sel", "student", {1: stamps["suite@1"]}, tl, th, env_seeds=(1,))
    lr = pr().level_reads(lstamps, th_doc(), trained_translate=tt)
    fh = G.completion([stamps[k] for k in st["prescreen_rows"] if k.startswith("field_hard")])
    cs = G.cat_summary(stamps["suite@1"])
    devs = [abs((cs.get(x) or {}).get("ratio") - 1.0) for x in tt if (cs.get(x) or {}).get("ratio") is not None]
    return {"iter": cand["iter"], "ckpt": cand["ckpt"],
            "misses": sum(1 for d in fl if not d.get("pass")) + len([x for x in lr["lines"]
                                                                     if x["id"].endswith(".es1") and not x["pass"]]),
            "field_hard": fh["rate"], "mean_abs_dev": (sum(devs) / len(devs)) if devs else None}


def full_block_read(cid: str, stamps: dict, lstamps: dict, tst_stamps: dict, g0: list, lvl: int = 90) -> dict:
    """pipeline.py full_block (2352-2410) with the level explicit (``lvl`` = 90; 2362 would read
    min(90, max_stable or 0))."""
    G, TH, S = gr(), thm(), ss()
    tl = trained()
    th = th_view()
    st = sel_spec()
    dec = G.walker_decisive("sel", stamps, tl, th, th_stage="student") if not g0 else []
    if not g0:
        stu_e, t_e = {}, {}
        for row in S.FIELD_ROWS:
            a, b = {}, {}
            ks = sorted(k for k in stamps if k.startswith(row + "@"))
            for i, k in enumerate(ks):
                a.update({(i, e): v for e, v in TH.field_episodes(stamps[k]).items()})
                b.update({(i, e): v for e, v in TH.field_episodes(tst_stamps[k]).items()})
            stu_e[row], t_e[row] = a, b
        pairs = S.completion_pairs(stu_e, t_e, st["completion_margin"])
        for p in pairs:
            dec.append({"id": f"student.capability.completion.{p['row']}", "statistic": f"{p['row']} completion "
                        "vs T* on Student-Play", "value": p.get("student"), "line": p.get("line"),
                        "direction": ">=", "pass": p["pass"], "rate": None, "unvalidated": True,
                        "rate_note": "paired; a lower bound (no distillation-seed variance)", **p})
    watch, flags = {}, []
    if not g0:
        watch = {"stable_at": lvl, "level": pr().level_reads(lstamps, th_doc(), trained_translate=[
            c for c in tl if c in G.TRANSLATE])}
        sa = {int(k.split("@")[1]): TH.suite_episodes(Path(v)) for k, v in stamps.items() if k.startswith("suite@")}
        sb = {int(k.split("@")[1]): TH.suite_episodes(Path(v)) for k, v in tst_stamps.items()
              if k.startswith("suite@")}
        watch["tracking"] = S.tracking_watch(sa, sb, tl)
        flags = watch["tracking"]["flags"]
    ok = not g0 and all(d.get("pass") for d in dec)
    return {"cand": cid, "pass": ok, "gate0": g0, "decisive": dec, "watch": watch, "watch_flags": flags,
            "verdict": G.verdict_of(dec, g0, watch_flags=flags)}


def sel_gate_doc(blocks: list) -> dict:
    """The gate record pipeline.full_block accumulates (2402-2406), over the blocks in run order."""
    G = gr()
    g = {"blocks": {}}
    for blk in blocks:
        g["blocks"][blk["cand"]] = blk
        ok = blk["pass"]
        flags = blk["watch_flags"]
        dec, g0 = blk["decisive"], blk["gate0"]
        g["verdict"] = (G.WATCH if flags else G.PASS) if ok else g.get("verdict", G.HALT)
        g["failed_lines"] = [d for d in dec if not d.get("pass")] or g0
        g["watch_flags"] = flags if ok else g.get("watch_flags", [])
    return g


def teacher_sel_rows(T: dict) -> dict:
    """The teacher's 11 full-block rows: suite@1/2 + field_light@1/2 = its L000 nominal cell rows (G1), the other
    seven its RSDSELT rows."""
    out = {}
    for es in (1, 2):
        for r in cell_rows(T, NOM["L000"], es):
            out[r["key"]] = r
    for k in SEVEN:
        out[k] = sel_row(T, k)
    return {k: out[k] for k in sel_spec()["full_block_rows"]}


def student_sel_rows(P: dict, which: str) -> dict:
    """{key: row} of a candidate: 'prescreen' (suite@1, field_hard@1/2), 'prescreen_level' (suite@1, field_light@1 at
    90 kg), 'full' (the 11), 'full_level' (the 4 at 90 kg)."""
    st = sel_spec()
    kg = int(st["prescreen_level"]["kg"])
    if which == "prescreen":
        return {k: sel_row(P, k) for k in st["prescreen_rows"]}
    if which == "prescreen_level":
        return {k: sel_row(P, k, kg) for k in st["prescreen_level"]["rows"]}
    if which == "full":
        return {k: sel_row(P, k) for k in st["full_block_rows"]}
    if which == "full_level":
        return {k: sel_row(P, k, 90) for k in st["full_block_level_rows"]}
    raise KeyError(which)


# ============================================================================== markdown
def _f(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def gate_md(res: dict) -> str:
    L = [f"# {res['gate']} read ({res['read_at']})", "",
         f"**Verdict: {res['verdict']}**" + (f" -- {'; '.join(res['notes'])}" if res.get("notes") else ""), ""]
    if res.get("missing"):
        L += ["Missing / invalid: " + "; ".join(res["missing"][:10]), ""]
    L += ["| cell | group | status | failing lines | es3 |", "|---|---|---|---|---|"]
    for grp in ("S", "N"):
        for cid, e in res[grp].items():
            L.append(f"| {cid} | {grp} | {e['status']} | {', '.join(e.get('fail_keys') or []) or '-'} | "
                     f"{_f((e.get('es3') or {}).get('pass'))} |")
    L += ["", "| pooled completion | head | ref | diff | m | binds | pass |", "|---|---|---|---|---|---|---|"]
    pool = res.get("N_pooled") or {}
    for cid, p in pool.items():
        p = p.get("two_launch", p)
        L.append(f"| N {cid} | {_f(p.get('head'))} | {_f(p.get('ref'))} | {_f(p.get('diff'))} | {_f(p.get('margin'))}"
                 f" | {p.get('binds', '-')} | {p.get('pass')} |")
    for lab in ("x07_tdr_vs_old", "A", "tctl_vs_old_x07"):
        p = res.get(lab)
        if p:
            L.append(f"| {lab} | {_f(p.get('head'))} | {_f(p.get('ref'))} | {_f(p.get('diff'))} | "
                     f"{_f(p.get('margin'))} | {p.get('binds', '-')} | beats by m: {p.get('beats_by_m')} |")
    if res["gate"] == "G1":
        r = res.get("R") or {}
        L += ["", f"R (paired 0-kg, its_past_p2a {res['its_past_p2a']}): {r.get('verdict', 'unread')}, max margin "
                  f"used {_f(r.get('max_margin_used'), 4)}" + (f", fails {r.get('fails')}" if r.get("fails") else ""),
              f"Lazy-control trigger (TDR x0.7 - old >= m): {res['lazy_control_trigger']}; TCTL due: "
              f"{res['tctl_due']}; half-gap clause: {res['half_gap']}"]
    else:
        L += ["", f"R WATCH (cross-pin) flags: {(res.get('R_watch') or {}).get('flags')}"]
    L += [f"es3 used {res['es3_used']} / {res['es3_cap']}; due {res['es3_due']}",
          f"Closing: {res['closing']['status']}", ""]
    return "\n".join(L) + "\n"


# ============================================================================== the CPU replay (step-8 precondition)
def _norm(x):
    """JSON-normalised (int keys -> str, tuples -> lists): what a recorded json would hold."""
    return json.loads(json.dumps(x, default=str))


def _diff(a, b, path="", out=None, limit=40):
    """Exact structural differences (float equality is exact)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing in mine' if k not in a else 'missing in record'}")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: mine {str(a)[:80]!r} != record {str(b)[:80]!r}")
    return out


def sweep_rows(cid: str, term, level, base: str, es: int, role: str) -> list[dict]:
    """The sweep's two rows of a cell-seed (cells.rows_for) in this library's row shape, for the launch checks."""
    C = sweep().C
    out = []
    for r in C.rows_for(cid, term, level, base, es, policy=role):
        lf = r["level_file"]
        kg = None
        if lf:
            kgt = tp().read_env_file(Path(lf)).get("KRABBY_PAYLOAD_KG", "0:0")
            kg = int(round(float(kgt.split(":")[0])))
        out.append({"kind": "sweep", "cell": cid, "suffix": "", "family": r["row"], "es": int(es), "policy": role,
                    "role": role, "name": r["name"], "ckpt": r["checkpoint"],
                    "ckpt_sha": sha256_file(Path(r["checkpoint"])), "env_file": r["env_file"], "level_file": lf,
                    "level_kg": kg, "manifest": r["manifest"], "scenario": r["scenario"], "args": r["args"],
                    "strip": [], "trained": trained() if r["row"] == "suite" else None,
                    "rod_s": float(level) if term is not None and term["term"] == "T01" else 1.0,
                    "zero_depth": bool(term is not None and term["kind"] == "zero_depth" and role == "student"),
                    "check_payload": term is None or term["kind"] != "payload_com", "key": f"{r['row']}@{es}"})
    return out


def _rec_diff(mine: dict | None, rec: dict | None) -> list[str]:
    if mine is None or rec is None:
        return [] if mine is None and rec is None else [f"mine {'None' if mine is None else 'read'} vs record "
                                                        f"{'None' if rec is None else 'read'}"]
    m = _norm(mine)
    d = []
    for f in ("lines", "failed", "n_lines", "survival_all", "lenient_feasible", "stamps", "es"):
        d += _diff(m.get(f), rec.get(f), f)
    if rec.get("launch_error") is None:
        if m["line_pass"] != rec.get("pass"):
            d.append(f"pass: mine {m['line_pass']} != record {rec.get('pass')}")
        if m["problems"]:
            d.append(f"my launch checks flag a stamp the sweep accepted: {m['problems'][:3]}")
    elif not m["problems"]:
        d.append(f"the sweep recorded a launch error {rec['launch_error'][:2]}; my checks pass")
    return d


def _sweep_read(cid, term, level, base, es, role, pin, dose):
    return read_cellseed(sweep_rows(cid, term, level, base, es, role), pin=pin, dose=dose, root=SWEEP_TP_OUT)


def replay_sweep(dose: bool = True) -> dict:
    """(a) The sweep's results/table.json and teacher.json line values and flags, from the DRS / DRST stamps, through
    this library's reader (stress_read.read_stamps + the campaign's launch checks against the sweep's own pin) and
    the sweep's own deltas / level_verdict on those reads."""
    C, SR = sweep().C, sweep().SR
    pin = SR.launch_pin()
    table = json.loads((SWEEP / "results/table.json").read_text())
    teacher = json.loads((SWEEP / "results/teacher.json").read_text())
    out = {"pin": pin, "cellseeds": 0, "cellseed_diffs": {}, "row_diffs": {}, "teacher_diffs": {}, "check_flags": {}}
    reads = {}
    for cid, e in table["reads"].items():
        c = e["cell"]
        t = C.TERM[c["term"]]
        seeds = {}
        for es_s, rec in e["seeds"].items():
            es = int(es_s)
            mine = _sweep_read(cid, t, c["level"], c["base"], es, "student", pin, dose)
            out["cellseeds"] += 1
            d = _rec_diff(mine, rec)
            if d:
                out["cellseed_diffs"][f"{cid}@es{es}"] = d
            if mine is not None:
                seeds[es] = mine
                if mine["problems"]:
                    out["check_flags"][f"{cid}@es{es}"] = mine["problems"][:4]
        reads[cid] = {"cell": c, "seeds": seeds}
    refs = {}
    for b, las in (("L000", ("A", "B")), ("L090", ("A",))):
        for es in (1, 2):
            refs[(b, es)] = [r for r in (_sweep_read(C.nominal_id(b, la), None, None, b, es, "student", pin, dose)
                                         for la in las) if r is not None and r["pass"] is not None]
    for row in table["rows"]:
        t = C.TERM[row["term"]]
        base = row["base"]
        ents = {lv: reads[C.cell_id(t, lv, base)] for lv in t["levels"]}
        per_level = {}
        if t["band"] is not None:
            for lv in t["levels"]:
                if C.inside_band(t, lv) and abs(lv - t["nominal"]) > SR.EPS:
                    per_level[C.fmt(lv)] = SR.level_verdict(t, lv, ents[lv])
        verdict = SR.worst(per_level.values()) if t["band"] is not None else None
        ref = "L090" if (base == "L090" or t["kind"] in ("payload_com", "payload_kg")) else "L000"
        flags, dl = [], {}
        for lv, e in ents.items():
            for es, r in e["seeds"].items():
                if r["pass"] is None:
                    continue
                d = SR.deltas(r, refs[(ref, es)], flag=t["kind"] != "payload_kg")
                dl.setdefault(C.fmt(lv), {})[es] = d
                flags += [f"{C.fmt(lv)}@es{es}:{k}" for k, x in d.items() if x["flag"]]
        d = (_diff(_norm(per_level), row["level_verdicts"], "level_verdicts") + _diff(verdict, row["verdict"], "verdict")
             + _diff(flags, row["drop_flags"], "drop_flags") + _diff(_norm(dl), row["deltas"], "deltas")
             + _diff({C.fmt(k): SR.level_status(v) for k, v in ents.items()}, row["status"], "status"))
        if d:
            out["row_diffs"][f"{row['term']}_{base}"] = d
    # the closing record (launch Z.. vs A / B at 0 kg es1)
    cl = json.loads((SWEEP / "results/closing.json").read_text())
    z = _sweep_read(C.nominal_id("L000", cl["launch"]), None, None, "L000", 1, "student", pin, dose)
    ab = {la: _sweep_read(C.nominal_id("L000", la), None, None, "L000", 1, "student", pin, dose) for la in ("A", "B")}
    matches = [la for la, r in ab.items() if r is not None and z is not None and values(r) == values(z)]
    out["closing"] = {"recorded": {"status": cl["status"], "matches": cl["matches"], "launch": cl["launch"]},
                      "mine": {"status": "match" if matches else "mismatch", "matches": matches},
                      "exact": (("match" if matches else "mismatch") == cl["status"] and matches == cl["matches"])}
    # teacher.json
    for k, rec in teacher["nominal"].items():
        b, es = k.split("@")
        mine = _sweep_read(C.nominal_id(b, "A"), None, None, b, int(es), "teacher", pin, dose)
        d = _rec_diff(mine, rec)
        if d:
            out["teacher_diffs"][f"nominal {k}"] = d
        out["cellseeds"] += 1
    tnom = {}
    for b in ("L000", "L090"):
        for es in (1, 2):
            tnom[f"{b}@{es}"] = read_cellseed(sweep_rows(C.nominal_id(b, "A"), None, None, b, es, "teacher"),
                                              pin=pin, dose=dose, root=SWEEP_TP_OUT)
    for cid, e in teacher["cells"].items():
        t = C.TERM[e["term"]]
        ref_base = "L090" if e["base"] == "L090" or t["kind"] in ("payload_com", "payload_kg") else "L000"
        for es_s, s in e["seeds"].items():
            es = int(es_s)
            tm = read_cellseed(sweep_rows(cid, t, e["level"], e["base"], es, "teacher"), pin=pin, dose=dose,
                               root=SWEEP_TP_OUT)
            sm = _sweep_read(cid, t, e["level"], e["base"], es, "student", pin, dose)
            out["cellseeds"] += 2
            d = _rec_diff(tm, s["teacher"]) + [f"student {x}" for x in _rec_diff(sm, s["student"])]
            ref = tnom.get(f"{ref_base}@{es}")
            td = (None if tm is None or tm["pass"] is None or ref is None or ref["pass"] is None
                  else SR.deltas(tm, [ref], flag=e["term"] != "T13"))
            d += _diff(_norm(td), s["teacher_deltas"], "teacher_deltas")
            if tm is not None and tm["problems"]:
                out["check_flags"][f"teacher {cid}@es{es}"] = tm["problems"][:4]
            if d:
                out["teacher_diffs"][f"{cid}@es{es}"] = d
    out["exact"] = not (out["cellseed_diffs"] or out["row_diffs"] or out["teacher_diffs"]) and out["closing"]["exact"]
    out["checks_clean"] = not out["check_flags"]
    return out


def _t4_sel_rows() -> dict:
    return json.loads((T4 / "F19/stage_reads/sel.rows.json").read_text())


def replay_sel(dose: bool = True) -> dict:
    """(b) Task 4's F19 selection (pipeline.py ex_select 2244-2318 + full_block 2352-2410) from the T4F19_sel_*
    stamps: the prescreen records, the rank, the teacher (T*) rows, the full block(s), the gate record and the pick.
    Pin checks relaxed (each stamp against its own code_state); every other launch check applies."""
    st = sel_spec()
    rec_rows = _t4_sel_rows()
    state = json.loads((T4 / "F19/pipeline_state.json").read_text())["stages"]["sel"]
    gate = json.loads((T4 / "F19/gates/sel.json").read_text())
    selj = json.loads((T4 / "F19/stage_reads/sel.json").read_text())
    ts = json.loads((T4 / "F19/stage_reads/tstar.json").read_text())
    omni_main = MAIN / OMNI_REL
    out = {"stamp_paths": [], "problems": {}, "diffs": {}}
    run_dir = C28000_CKPT.parent
    cands = ss().candidate_checkpoints(run_dir, ts["iteration"], st["offsets"])
    out["candidates"] = [c.get("ckpt") for c in cands]
    kg = int(st["prescreen_level"]["kg"])

    def locate(pol, key, level, name, rec_key):
        row = sel_row(pol, key, level, t4=T4_MAIN, omni=omni_main, name=name)
        s = row_stamp(row, T4_TP_OUT)
        want = (rec_rows.get(rec_key) or {}).get("stamp")
        if s is None or str(s) != want:
            out["stamp_paths"].append(f"{rec_key}: located {s} != recorded {want}")
        if s is not None:
            p = row_problems(s, row, None, dose=dose, relax_pin=True)
            if p:
                out["problems"][rec_key] = p[:4]
        return s, row

    pres, pols = {}, {}
    for c in cands:
        if not c.get("ckpt"):
            continue
        cid = f"c{c['iter']}"
        pol = ephemeral_policy(c["ckpt"], "student", cid)
        pols[cid] = pol
        stamps = {k: locate(pol, k, None, f"T4F19_sel_{cid}", f"{cid}:{k}")[0] for k in st["prescreen_rows"]}
        lst = {k: locate(pol, k, kg, f"T4F19_sel_{cid}_L{kg:03d}", f"{cid}:{k}@L{kg:03d}")[0]
               for k in st["prescreen_level"]["rows"]}
        pres[cid] = prescreen_read({"iter": c["iter"], "ckpt": c["ckpt"]}, stamps, lst)
    out["diffs"]["prescreen"] = _diff(_norm(pres), state["prescreen"], "prescreen")
    ranked = [f"c{p['iter']}" for p in ss().rank(list(pres.values()))]
    out["ranked"] = ranked
    out["diffs"]["ranked"] = _diff(ranked, state["ranked"], "ranked")
    T = ephemeral_policy(ts["ckpt"], "teacher", "tstar")
    tst = {k: locate(T, k, None, "T4F19_sel_tstar", f"tstar:{k}")[0] for k in st["full_block_rows"]}
    blocks, bstate = [], {}
    while True:
        act = ss().next_action({"ranked": ranked, "blocks": bstate}, int(st.get("max_failed_blocks", 2)))
        if act["do"] != "block":
            break
        cid = act["cand"]
        pol = pols[cid]
        stamps = {k: locate(pol, k, None, f"T4F19_sel_{cid}", f"{cid}:{k}")[0] for k in st["full_block_rows"]}
        lst = {k: locate(pol, k, 90, f"T4F19_sel_{cid}_L090", f"{cid}:{k}@L090")[0]
               for k in st["full_block_level_rows"]}
        g0 = [p for k, ps in out["problems"].items() if k.startswith(f"{cid}:") for p in ps]
        blk = full_block_read(cid, stamps, lst, tst, g0, lvl=90)
        blocks.append(blk)
        bstate[cid] = "PASS" if blk["pass"] else "FAIL"
        out["diffs"][f"block {cid}"] = _diff(_norm(blk), gate["blocks"].get(cid), f"blocks/{cid}")
    out["action"] = act
    g = sel_gate_doc(blocks)
    for k in ("verdict", "failed_lines", "watch_flags"):
        out["diffs"][f"gate {k}"] = _diff(_norm(g.get(k)), gate.get(k), k)
    out["diffs"]["blocks state"] = _diff(bstate, state["blocks"], "state.blocks")
    out["failed_blocks"] = [c for c, v in bstate.items() if v == "FAIL"]
    if act["do"] == "selected":
        c = pres[act["cand"]]
        pick = {"cand": act["cand"], "ckpt": c["ckpt"], "sha256": sha256_file(Path(c["ckpt"])), "iter": c["iter"],
                "tstar": ts["stage"]}
        out["pick"] = pick
        out["diffs"]["pick"] = _diff(pick, selj, "sel.json")
    else:
        out["diffs"]["pick"] = [f"no pick ({act})"]
    blk = next((b for b in blocks if b["cand"] == act.get("cand")), None)
    if blk:
        out["margins"] = {d["row"]: {k: d.get(k) for k in ("student", "tstar", "se_diff", "margin", "binds", "line",
                                                            "pass")}
                          for d in blk["decisive"] if d["id"].startswith("student.capability.completion.")}
    out["exact"] = not any(out["diffs"].values()) and not out["stamp_paths"]
    out["checks_clean"] = not out["problems"]
    return out


def zk_from_record(sid: str, *, dose: bool, out: dict) -> dict:
    """A Task 4 stage's zero-kg stamps (pipeline.head_zero_kg_stamps / p2a_zero_kg_stamps): the driver stamp at es1,
    the T4F19_<sid> rows located by name (newest complete), cross-checked with the recorded rows.json."""
    head = json.loads((T4 / f"F19/stage_reads/{sid}.json").read_text())
    rows = json.loads((T4 / f"F19/stage_reads/{sid}.rows.json").read_text())
    res = {"suite": {1: head["driver_stamp"]}}

    def loc(key):
        r = parse_row(key)
        sc = spec()["rows"][r["family"]]["scenario"]
        s = stamp_dir(f"T4F19_{sid}", sc, r["es"], T4_TP_OUT)
        want = (rows.get(key) or {}).get("stamp")
        if s is None or str(s) != want:
            out.setdefault("stamp_paths", []).append(f"{sid} {key}: located {s} != recorded {want}")
        if s is not None:
            gp = gr().stamp_complete_problems(s)
            if gp:
                out.setdefault("problems", {})[f"{sid} {key}"] = gp
            if dose:
                dp = dose_problems(s, 1.0, f"{sid} {key}")
                if dp:
                    out.setdefault("problems", {})[f"{sid} {key} dose"] = dp
        return None if s is None else str(s)
    if "suite@2" in rows:
        res["suite"][2] = loc("suite@2")
    res["side22"] = loc("side22@1") if "side22@1" in rows else None
    for row in ("field_hard", "field_light", "ring_obst", "field_turn"):
        res[row] = [loc(f"{row}@{es}") for es in (1, 2) if f"{row}@{es}" in rows]
    return res


def replay_zero_kg(dose: bool = True) -> dict:
    """(c) F19's recorded zero-kg verdicts through the campaign's OWN R wrapper (:func:`r_check`): p3a FAIL (max
    margin used 2.7099, four fails) and p3b PASS (0.6620), Task 4's own its_past_p2a (2999 / 5998) and p3b's
    trained list."""
    out = {"stages": {}}
    ref = zk_from_record("p2a", dose=dose, out=out)
    p2a_it = its_of(json.loads((T4 / "F19/stage_reads/p2a.json").read_text())["ckpt"])
    tl = trained()
    for sid, its in (("p3a", 2999), ("p3b", 5998)):
        head = zk_from_record(sid, dose=dose, out=out)
        h_it = its_of(json.loads((T4 / f"F19/stage_reads/{sid}.json").read_text())["ckpt"])
        res = r_check(head, ref, its, tl)
        rec = json.loads((T4 / f"F19/gates/{sid}.json").read_text())["zero_kg"]
        mine = {"verdict": res["verdict"], "max_margin_used": res["max_margin_used"], "fails": res["fails"],
                "translate_statistic": res.get("translate_statistic"),
                "thresholds_version": res.get("thresholds_version")}
        want = {k: rec.get(k) for k in mine}
        out["stages"][sid] = {"its_past_p2a": its, "its_from_heads": h_it - p2a_it, "mine": mine, "record": want,
                              "diffs": _diff(_norm(mine), want, sid)
                              + ([] if h_it - p2a_it == its else [f"its {h_it - p2a_it} != {its}"])}
    out["exact"] = all(not s["diffs"] for s in out["stages"].values()) and not out.get("stamp_paths")
    out["checks_clean"] = not out.get("problems")
    return out
