"""Config-identity checks for the three-phase training paradigm (``KRABBY_PHASE`` presets).

Each check builds configs INSIDE Isaac Sim (``crab_hex_phase_cfg_dump.py``; one headless boot per
fixture, no simulation stepped) and compares ``class_to_dict`` outputs:

* phase-2 ``Isaac-Crab-Hex-Teacher-v0`` in mode ``2a|2b|2c`` == ``Isaac-Crab-Hex-Flat-Walk-v0``
  under the same preset (the lineage of record trained those windows in the flat-walk task);
* a ``KRABBY_PHASE`` preset reproduces the recorded ``params/env.yaml`` / ``agent.yaml`` of the
  A15+B lineage window it encodes (``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-06_2130_a15b_lineage``);
* the preset equals the raw legacy ``KRABBY_*`` stack of that window;
* phase-3 ``Isaac-Crab-Hex-Student-v0`` carries the phase-2c MDP (terrain, commands, events,
  horizon, plant) into distillation;
* the Play variants keep the train MDP.

Opt in (≈ 6 Kit boots, ~5 min)::

    RUN_CRAB_HEX_CFG_IDENTITY=1 pytest tests/integration/test_crab_hex_phase_configs.py -v

Set ``KRABBY_CFG_DUMP_DIR`` to a directory holding ``cfg_<fixture>.json`` files to reuse dumps
(the fixture names are the keys of ``FIXTURES``). ``KRABBY_ISAAC_PYTHON`` overrides the venv python.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.isaacsim

REPO = Path(__file__).resolve().parents[2]
PARKOUR = REPO / "parkour"
DUMP = Path(__file__).resolve().parent / "crab_hex_phase_cfg_dump.py"
PY = os.environ.get("KRABBY_ISAAC_PYTHON", "/home/nickmagus/krabby/isaac_venv/bin/python")
LINEAGE_STATE = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-06_2130_a15b_lineage/state.json"
PLANT = "A15+B"

FLAT = "Isaac-Crab-Hex-Flat-Walk-v0"
FLAT_PLAY = "Isaac-Crab-Hex-Flat-Walk-Play-v0"
TEACHER = "Isaac-Crab-Hex-Teacher-v0"
TEACHER_PLAY = "Isaac-Crab-Hex-Teacher-Play-v0"
STUDENT = "Isaac-Crab-Hex-Student-v0"
STUDENT_PLAY = "Isaac-Crab-Hex-Student-Play-v0"

# fixture name -> (env for the dump process, tasks to build)
FIXTURES: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "1a": ({"KRABBY_PHASE": "1a", "KRABBY_PLANT": PLANT}, (FLAT, FLAT_PLAY)),
    "2a": ({"KRABBY_PHASE": "2a", "KRABBY_PLANT": PLANT}, (TEACHER, FLAT)),
    "2b": ({"KRABBY_PHASE": "2b", "KRABBY_PLANT": PLANT}, (TEACHER, FLAT)),
    "2c": ({"KRABBY_PHASE": "2c", "KRABBY_PLANT": PLANT}, (TEACHER, FLAT, TEACHER_PLAY, FLAT_PLAY)),
    "3a": ({"KRABBY_PHASE": "3a", "KRABBY_PLANT": PLANT}, (STUDENT, STUDENT_PLAY)),
    # no plant variable at all: the main asset (assets/crab.usda, A15+B) must be what the preset gets
    "2c_noplant": ({"KRABBY_PHASE": "2c"}, (FLAT,)),
}
# phase -> lineage window whose recorded params it must reproduce (A15+B lineage, seed 3)
RECORDED_WINDOW = {"1a": "0", "2a": "1", "2b": "2", "2c": "3"}

# fields that legitimately differ between a fresh cfg and a recorded run
ENV_VOLATILE = {("seed",), ("scene", "num_envs"), ("sim", "device")}
AGENT_VOLATILE = {("seed",), ("device",), ("max_iterations",), ("resume",), ("load_run",),
                  ("load_checkpoint",), ("run_name",), ("experiment_name",)}
# student subtrees that must equal the 2c teacher's (the student keeps its own observations,
# rewards, depth camera and camera-placement event; it has no reward curriculum)
STUDENT_SHARED = (("commands",), ("events",), ("terminations",), ("actions",), ("episode_length_s",),
                  ("decimation",), ("sim", "dt"), ("sim", "physx", "enable_external_forces_every_iteration"),
                  ("scene", "terrain"), ("scene", "robot", "spawn", "usd_path"))
STUDENT_ONLY = {("events", "random_camera_position")}
# fields the recorded env.yaml carries in runtime-resolved form (dumped after gym.make): prim
# paths formatted, terrain num_envs / env_spacing filled in, generator-level size / scales
# propagated into every sub-terrain by TerrainGenerator.__init__
RUNTIME_DROP = {("scene", "terrain", "num_envs"), ("scene", "terrain", "env_spacing")}
TG_PROPAGATED = ("size", "horizontal_scale", "vertical_scale", "slope_threshold")


# ------------------------------------------------------------------ helpers (pure)
def normalize(obj):
    """Tuples -> lists and non-primitive objects -> ``str`` (the dump serialises with ``default=str``;
    the recorded YAML carries e.g. ``slice`` objects)."""
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def diff_paths(a, b, path=(), out=None, limit=60):
    """Leaf-level differences between two nested dicts as (path, a, b)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((path + (k,), a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                diff_paths(a[k], b[k], path + (k,), out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, path + (i,), out, limit)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and abs(a - b) <= 1e-9 * max(1.0, abs(a))):
        out.append((path, a, b))
    return out


def strip(d: dict, paths) -> dict:
    d = copy.deepcopy(normalize(d))
    for p in paths:
        cur = d
        for k in p[:-1]:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict):
            cur.pop(p[-1], None)
    return d


OLD_CAMPAIGN_PREFIX = str(REPO / "sim_fine_tuning") + "/"
NEW_CAMPAIGN_PREFIX = str(REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments") + "/"


def _by_content(value):
    """Asset paths (``.usda``, ``.npz``) become ``sha256:<digest>`` of the file: the recorded runs name the
    A15+B variant file while the preset now yields the byte-identical main asset ``assets/crab.usda``
    (2026-09-09), and the RSI banks moved with the campaigns from ``sim_fine_tuning/`` to the
    package's ``experiments/`` (same bytes). Unknown/missing files are left as they are."""
    if not isinstance(value, str) or not value.endswith((".usda", ".usd", ".npz")):
        return value
    p = Path(value)
    if not p.is_file() and value.startswith(OLD_CAMPAIGN_PREFIX):
        p = Path(NEW_CAMPAIGN_PREFIX + value[len(OLD_CAMPAIGN_PREFIX):])
    if p.is_file():
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    return value


def runtime_resolve(d: dict) -> dict:
    """Bring a fresh cfg dict and a recorded (post-``gym.make``) one to the same form."""
    d = strip(d, RUNTIME_DROP)

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return _by_content(x.replace("{ENV_REGEX_NS}", "/World/envs/env_.*"))
        return x

    d = walk(d)
    tg = ((d.get("scene") or {}).get("terrain") or {}).get("terrain_generator") or {}
    for sub in (tg.get("sub_terrains") or {}).values():
        for k in TG_PROPAGATED:
            if k in tg and isinstance(sub, dict):
                sub[k] = tg[k]
    return d


def subtree(d: dict, path):
    cur = d
    for k in path:
        cur = cur[k]
    return cur


def fmt_diffs(diffs) -> str:
    return "\n".join(f"  {'/'.join(map(str, p))}: {a!r} != {b!r}" for p, a, b in diffs)


def load_recorded(run_dir: Path) -> tuple[dict, dict]:
    env = yaml.load((run_dir / "params" / "env.yaml").read_text(), Loader=yaml.UnsafeLoader)
    agent = yaml.load((run_dir / "params" / "agent.yaml").read_text(), Loader=yaml.UnsafeLoader)
    return normalize(env), normalize(agent)


def lineage_window(w: str) -> tuple[Path, dict] | None:
    if not LINEAGE_STATE.exists():
        return None
    rec = json.loads(LINEAGE_STATE.read_text()).get("windows", {}).get(w)
    if not rec or not rec.get("ckpt") or not Path(rec["ckpt"]).exists():
        return None
    return Path(rec["ckpt"]).parent, dict(rec["env"])


# ------------------------------------------------------------------ dump production
def _require() -> None:
    if not os.environ.get("RUN_CRAB_HEX_CFG_IDENTITY"):
        pytest.skip("set RUN_CRAB_HEX_CFG_IDENTITY=1 (boots Isaac Sim headless ~6x)")
    if not Path(PY).exists():
        pytest.skip(f"Isaac venv python not found: {PY}")


def dump(env_vars: dict[str, str], tasks: tuple[str, ...], out: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KRABBY_")}
    env.update({"OMNI_KIT_ACCEPT_EULA": "yes", "TERM": "xterm"})
    env.update(env_vars)
    cmd = [PY, str(DUMP), "--headless", "--out", str(out)]
    for t in tasks:
        cmd += ["--task", t]
    log = out.with_suffix(".log")
    with log.open("w") as fh:
        subprocess.run(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=900, check=False)
    if not out.exists():
        pytest.fail(f"dump produced no output ({out}); see {log}")
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def dumps(tmp_path_factory) -> dict[str, dict]:
    _require()
    reuse = os.environ.get("KRABBY_CFG_DUMP_DIR")
    d = Path(reuse) if reuse else tmp_path_factory.mktemp("cfgdump")
    out: dict[str, dict] = {}
    fixtures = dict(FIXTURES)
    win = lineage_window(RECORDED_WINDOW["2c"])
    if win is not None:
        fixtures["2c_legacy"] = (win[1], (FLAT,))
    for name, (env_vars, tasks) in fixtures.items():
        f = d / f"cfg_{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else dump(env_vars, tasks, f)
    return out


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("phase", ["2a", "2b", "2c"])
def test_phase2_teacher_mode_matches_flat_walk(dumps, phase):
    t, f = dumps[phase][TEACHER], dumps[phase][FLAT]
    env_diffs = diff_paths(normalize(t["env"]), normalize(f["env"]))
    assert not env_diffs, f"{phase}: Teacher-v0 env cfg != Flat-Walk-v0 env cfg\n{fmt_diffs(env_diffs)}"
    agent_diffs = diff_paths(strip(t["agent"], AGENT_VOLATILE), strip(f["agent"], AGENT_VOLATILE))
    assert not agent_diffs, f"{phase}: teacher runner != flat-walk runner\n{fmt_diffs(agent_diffs)}"
    assert t["agent"]["experiment_name"] == "crab_hex_teacher"
    assert dumps[phase]["_environ"]["KRABBY_HEX_TEACHER_MODE"] == phase


@pytest.mark.parametrize("phase", ["1a", "2a", "2b", "2c"])
def test_phase_preset_reproduces_recorded_window(dumps, phase):
    win = lineage_window(RECORDED_WINDOW[phase])
    if win is None:
        pytest.skip("A15+B lineage run directories not on this machine")
    run_dir, _ = win
    rec_env, rec_agent = load_recorded(run_dir)
    got = dumps[phase][FLAT]
    env_diffs = diff_paths(runtime_resolve(strip(rec_env, ENV_VOLATILE)),
                           runtime_resolve(strip(normalize(got["env"]), ENV_VOLATILE)))
    assert not env_diffs, f"{phase}: preset env cfg != recorded {run_dir.name}/params/env.yaml\n{fmt_diffs(env_diffs)}"
    agent_diffs = diff_paths(strip(rec_agent, AGENT_VOLATILE), strip(normalize(got["agent"]), AGENT_VOLATILE))
    assert not agent_diffs, f"{phase}: preset runner != recorded agent.yaml\n{fmt_diffs(agent_diffs)}"


def test_phase_preset_equals_legacy_env_stack(dumps):
    if "2c_legacy" not in dumps:
        pytest.skip("A15+B lineage state not on this machine")
    a, b = dumps["2c"][FLAT], dumps["2c_legacy"][FLAT]
    diffs = diff_paths(runtime_resolve(a["env"]), runtime_resolve(b["env"])) + diff_paths(normalize(a["agent"]), normalize(b["agent"]))
    assert not diffs, f"KRABBY_PHASE=2c != raw window-3 KRABBY_* stack\n{fmt_diffs(diffs)}"


def test_student_phase_mirrors_2c_mdp(dumps):
    s = runtime_resolve(strip(dumps["3a"][STUDENT]["env"], STUDENT_ONLY))
    t = runtime_resolve(strip(dumps["2c"][TEACHER]["env"], STUDENT_ONLY))
    diffs = []
    for p in STUDENT_SHARED:
        try:
            diffs += diff_paths(subtree(s, p), subtree(t, p), p)
        except KeyError:
            diffs.append((p, "<missing in one cfg>", ""))
    assert not diffs, f"3a student MDP != 2c teacher MDP on the shared managers\n{fmt_diffs(diffs)}"
    assert dumps["3a"][STUDENT]["agent"]["algorithm"]["class_name"] == "DistillationWithExtractor"
    # The vec-env wrapper clips RAW policy actions to +-clip_actions before the action term scales
    # them; an unset student clip on the teacher's full action space drove the joints at ~5x the
    # trained authority (2026-09-08 phase-3a failure). Student and teacher runners must agree.
    assert dumps["3a"][STUDENT]["agent"]["clip_actions"] == dumps["2c"][TEACHER]["agent"]["clip_actions"], \
        "student runner clip_actions differs from the 2c teacher runner"
    assert "KRABBY_PHASEOUT" not in dumps["3a"]["_environ"]


def test_no_plant_variable_means_the_main_asset(dumps):
    """KRABBY_PHASE alone (no KRABBY_PLANT / KRABBY_HEX_USD_PATH) builds the A15+B main asset and the
    same env as the preset with KRABBY_PLANT=A15+B."""
    a, b = dumps["2c_noplant"][FLAT], dumps["2c"][FLAT]
    assert a["env"]["scene"]["robot"]["spawn"]["usd_path"].endswith("assets/crab.usda")
    assert "KRABBY_HEX_USD_PATH" not in dumps["2c_noplant"]["_environ"]
    diffs = diff_paths(runtime_resolve(a["env"]), runtime_resolve(b["env"])) + diff_paths(normalize(a["agent"]), normalize(b["agent"]))
    assert not diffs, f"no-plant 2c != KRABBY_PLANT=A15+B 2c\n{fmt_diffs(diffs)}"


def test_play_variants_keep_the_train_mdp(dumps):
    a, b = normalize(dumps["2c"][TEACHER_PLAY]["env"]), normalize(dumps["2c"][FLAT_PLAY]["env"])
    diffs = diff_paths(a, b)
    assert not diffs, f"2c: Teacher-Play != Flat-Walk-Play\n{fmt_diffs(diffs)}"
    train = normalize(dumps["2c"][TEACHER]["env"])
    assert subtree(a, ("scene", "terrain", "terrain_generator", "difficulty_range")) == \
        subtree(train, ("scene", "terrain", "terrain_generator", "difficulty_range"))


if __name__ == "__main__":  # ad-hoc: python test_crab_hex_phase_configs.py <dump dir>
    sys.exit(pytest.main(["-v", __file__] + sys.argv[1:]))


def test_mdp_pins_match_dumps(dumps):
    """policy/mdp_pins.yaml (the summary's Isaac-only numbers) equals a fresh extraction from the dumps."""
    import importlib.util

    tool = PARKOUR / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "experiments" / "tools" / "policy_summary.py"
    spec = importlib.util.spec_from_file_location("policy_summary", tool)
    ps = importlib.util.module_from_spec(spec)
    sys.modules["policy_summary"] = ps
    spec.loader.exec_module(ps)
    manifest = ps.load_manifest()
    expect = ps.pins_from_dumps({k: v for k, v in dumps.items() if k in {s["phase"] for s in manifest["stages"]}}, manifest)
    diffs = diff_paths(normalize(ps.load_pins()), normalize(expect))
    assert not diffs, "mdp_pins.yaml is stale (bundle_policy.py --pin-mdp <dump dir>):\n" + fmt_diffs(diffs)

