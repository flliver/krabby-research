"""Guards on the forward-walk task's load-bearing literals, for the travel-frame seams.

The sideways sibling task reaches all of its new behaviour through attributes the forward cfg
classes do NOT have (``env.cfg.travel_yaw``, ``cmd_cfg.heading_offset``), read with a
``getattr(..., 0.0)`` default. Three things must therefore stay true, and none of them is
covered by ``policy/mdp_pins.yaml``:

1. the forward scene literals the sideways scene overrides (spawn rot, scanner offset, scan
   grid, depth-camera mount) are unchanged -- a changed scan ``size``/``resolution`` would
   silently invalidate the mirror sizing and every warm start;
2. the forward symmetry ``data_augmentation_func`` string still points at the forward mirror,
   and the forward tracking term still uses the stock ``track_lin_vel_xy_exp``;
3. no forward config file mentions ``travel_yaw``/``heading_offset`` at all, and every
   ``getattr`` seam in the shared library and the forward mdp/scripts defaults to ``0.0``.

Plus the inertness pin for ``penalty_base_pitch_forward_linear``: it reads chassis pitch (which
is body ROLL under sideways travel) and is only harmless while its weight stays 0 with no knob
mapped to it. Stdlib only.
"""

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "parkour" / "parkour_isaaclab"
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task"
SCENE_PY = PKG / "config" / "crab_hex" / "crab_hex_scene_cfg.py"
AGENT_PY = PKG / "config" / "crab_hex" / "agents" / "rsl_rl_ppo_cfg.py"
SUMMARY_TOOL = PKG / "experiments" / "tools" / "policy_summary.py"

FORWARD_SYMMETRY_FUNC = (
    "parkour_tasks.crab_hex_forward_task.mdp.crab_hex_mirror:crab_hex_symmetry_augmentation"
)

# (literal, what it is) -- the five scene numbers the sideways scene cfg re-declares.
SCENE_LITERALS = [
    ("rot=(1.0, 0.0, 0.0, 0.0)", "forward spawn rotation (identity yaw)"),
    ("pos=(0.6, 0.0, 20.0)", "height-scanner offset, 0.6 m ahead along body +x"),
    ("pattern_cfg.size = [2.4, 3.4]", "scan grid size (with resolution 0.3 -> 108 rays)"),
    ("resolution = 0.3", "scan grid resolution"),
    ("pos=(0.37, 0.0, 0.10)", "depth-camera mount on the +x face"),
]

GETATTR_SEAM = re.compile(r'getattr\([^,]+,\s*["\'](travel_yaw|heading_offset)["\']\s*,\s*([^)]+)\)')
SEAM_ROOTS = [LIB, PKG / "mdp", PKG / "scripts"]


def _summary():
    spec = importlib.util.spec_from_file_location("policy_summary", SUMMARY_TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["policy_summary"] = mod
    spec.loader.exec_module(mod)
    return mod


def _seam_sources():
    for root in SEAM_ROOTS:
        for path in sorted(root.rglob("*.py")):
            parts = path.parts
            if "__pycache__" in parts or "crab_hex_sideways_task" in parts:
                continue
            yield path


def test_forward_scene_literals_are_unchanged():
    src = SCENE_PY.read_text()
    missing = [(lit, what) for lit, what in SCENE_LITERALS if lit not in src]
    assert not missing, f"{SCENE_PY}: missing {missing}"


def test_forward_symmetry_func_still_points_at_the_forward_mirror():
    src = AGENT_PY.read_text()
    assert FORWARD_SYMMETRY_FUNC in src
    # ... and inside the flat-walk/phase-2 symmetry helper, not some other block.
    body = src.split("def _apply_flat_walk_symmetry", 1)[1].split("\n@configclass", 1)[0]
    assert FORWARD_SYMMETRY_FUNC in body


def test_forward_tracking_term_is_the_stock_isaaclab_one():
    ps = _summary()
    funcs = {t["name"]: t["func"] for t in ps.reward_terms_from_cfg_source()}
    assert funcs["track_lin_vel_xy_exp"] == "track_lin_vel_xy_exp"


def test_no_forward_config_file_knows_about_the_travel_frame():
    offenders = []
    paths = [p for p in (PKG / "config").rglob("*.py") if "__pycache__" not in p.parts]
    paths.append(LIB / "envs" / "mdp" / "parkour_commands" / "parkour_command_cfg.py")
    for path in paths:
        src = path.read_text()
        for token in ("travel_yaw", "heading_offset"):
            if token in src:
                offenders.append((str(path.relative_to(REPO)), token))
    assert not offenders, f"travel-frame tokens leaked into forward config: {offenders}"


def test_every_travel_frame_getattr_defaults_to_zero():
    bad = []
    seen = 0
    for path in _seam_sources():
        for attr, default in GETATTR_SEAM.findall(path.read_text()):
            seen += 1
            if default.strip() != "0.0":
                bad.append((str(path.relative_to(REPO)), attr, default.strip()))
    assert not bad, f"non-zero travel-frame default (forward path would change): {bad}"
    assert seen >= 4, "the travel-frame seams vanished from the shared library"


def test_forward_pitch_penalty_stays_inert():
    ps = _summary()
    terms = ps.reward_terms_from_cfg_source()
    weights = {t["name"]: t["default_weight"] for t in terms}
    assert weights["penalty_base_pitch_forward_linear"] == 0.0
    knobs = ps.knob_overrides_from_cfg_source()
    assert [k for k, term in knobs.items() if term == "penalty_base_pitch_forward_linear"] == []
