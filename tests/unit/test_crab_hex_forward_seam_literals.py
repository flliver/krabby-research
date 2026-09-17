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
3. no forward config file mentions ``travel_yaw``/``heading_offset`` (nor the omni task's
   ``omni_command``/``omni_clock_turn_radius``/``CrabHexOmni``) at all, and every ``getattr``
   seam in the shared library and the forward mdp/scripts defaults to the IDENTITY value for
   its attribute (``0.0`` / ``False``);
4. the three observation hooks the omni task overrides (``_steering_channels``,
   ``_command_channels``, ``ZERO_STEERING_HISTORY``) still contain exactly the expressions
   that used to sit inline in ``__call__`` -- they are a refactor, not a change.

Plus the inertness pin for ``penalty_base_pitch_forward_linear``: it reads chassis pitch (which
is body ROLL under sideways travel) and is only harmless while its weight stays 0 with no knob
mapped to it. Stdlib only.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "parkour" / "parkour_isaaclab"
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task"
SCENE_PY = PKG / "config" / "crab_hex" / "crab_hex_scene_cfg.py"
AGENT_PY = PKG / "config" / "crab_hex" / "agents" / "rsl_rl_ppo_cfg.py"
OBSERVATIONS_PY = PKG / "mdp" / "observations.py"
SUMMARY_TOOL = PKG / "experiments" / "tools" / "policy_summary.py"

OBS_CLASS = "CrabHexParkourObservations"

# The three observation hooks (omni-task plan D2). The omni sibling overrides them; the
# forward values must stay bit-identical, which means the hook bodies must stay EXACTLY the
# expressions that used to sit inline in ``__call__`` -- same ops, same order, same dtype.
# Pinned as normalized source (``ast.unparse``: formatting and comments do not matter, code
# does). A changed expression here silently changes every forward observation.
EXPECTED_HOOK_BODIES = {
    "_steering_channels": (
        "if os.environ.get('KRABBY_HEADING') is not None:\n"
        "    cmd_term = env.command_manager.get_term('base_velocity')\n"
        "    heading_err = wrap_to_pi(cmd_term.heading_target - self.asset.data.heading_w)\n"
        "    delta_yaw = torch.where(on_flat, heading_err, self.delta_yaw)\n"
        "    delta_next_yaw = torch.where(on_flat, heading_err, self.delta_next_yaw)\n"
        "else:\n"
        "    delta_yaw = torch.where(on_flat, torch.zeros_like(self.delta_yaw), self.delta_yaw)\n"
        "    delta_next_yaw = torch.where(on_flat, torch.zeros_like(self.delta_next_yaw), self.delta_next_yaw)\n"
        "return (delta_yaw, delta_next_yaw)"
    ),
    "_command_channels": "return torch.cat([0 * commands[:, 0:2], commands[:, 0:1]], dim=-1)",
}

# ... and the third hook: a class attribute, read at the history-blanking line.
EXPECTED_ZERO_STEERING_HISTORY = True
ZERO_STEERING_GUARD = (
    "if self.ZERO_STEERING_HISTORY:\n    obs_buf[:, 6:8] = 0"
)

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

GETATTR_SEAM = re.compile(
    r'getattr\([^,]+,\s*["\']'
    r"(travel_yaw|heading_offset|omni_command|omni_clock_turn_radius)"
    r'["\']\s*,\s*([^)]+)\)'
)
# The IDENTITY default per seam attribute: the value that makes the shared code take the
# forward path. ``travel_yaw``/``heading_offset`` -> no rotation; ``omni_command`` -> False
# (the ``cmd[:, 0]`` gates and ``cmd[:, 0].abs()`` cadence stay literal); ``omni_clock_turn_radius``
# -> 0.0 (no turn contribution to the clock even if the planar branch were taken).
SEAM_DEFAULTS = {
    "travel_yaw": "0.0",
    "heading_offset": "0.0",
    "omni_command": "False",
    "omni_clock_turn_radius": "0.0",
}
SEAM_ROOTS = [LIB, PKG / "mdp", PKG / "scripts"]

# Tokens that must never appear in a forward CONFIG file: the travel-frame seam attributes
# the sideways task adds, and the omni task's cfg flags and class-name prefix.
FORBIDDEN_SIBLING_TOKENS = (
    "travel_yaw",
    "heading_offset",
    "omni_command",
    "omni_clock_turn_radius",
    "CrabHexOmni",
)


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
            if (
                "__pycache__" in parts
                or "crab_hex_sideways_task" in parts
                or "crab_hex_omni_task" in parts
            ):
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


def test_no_forward_config_file_knows_about_the_sibling_tasks():
    offenders = []
    paths = [p for p in (PKG / "config").rglob("*.py") if "__pycache__" not in p.parts]
    paths.append(LIB / "envs" / "mdp" / "parkour_commands" / "parkour_command_cfg.py")
    for path in paths:
        src = path.read_text()
        for token in FORBIDDEN_SIBLING_TOKENS:
            if token in src:
                offenders.append((str(path.relative_to(REPO)), token))
    assert not offenders, f"sibling-task tokens leaked into forward config: {offenders}"


def test_every_sibling_seam_getattr_defaults_to_the_identity():
    bad = []
    seen = 0
    for path in _seam_sources():
        for attr, default in GETATTR_SEAM.findall(path.read_text()):
            seen += 1
            if default.strip() != SEAM_DEFAULTS[attr]:
                bad.append((str(path.relative_to(REPO)), attr, default.strip()))
    assert not bad, f"non-identity sibling-task default (forward path would change): {bad}"
    assert seen >= 4, "the travel-frame seams vanished from the shared library"
    # and both of the omni seams are actually wired, in the reward file and the action term
    found = {
        attr
        for path in _seam_sources()
        for attr, _ in GETATTR_SEAM.findall(path.read_text())
    }
    assert {"omni_command", "omni_clock_turn_radius"} <= found


def _obs_class() -> ast.ClassDef:
    tree = ast.parse(OBSERVATIONS_PY.read_text(), str(OBSERVATIONS_PY))
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == OBS_CLASS]
    assert len(found) == 1, f"{OBS_CLASS} not found (once) in {OBSERVATIONS_PY}"
    return found[0]


def _method_body_source(cls: ast.ClassDef, name: str) -> str:
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(methods) == 1, f"{OBS_CLASS}.{name} not found (once)"
    body = methods[0].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring: prose, not behaviour
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_observation_hook_bodies_are_the_pre_refactor_expressions():
    cls = _obs_class()
    for name, expected in EXPECTED_HOOK_BODIES.items():
        assert _method_body_source(cls, name) == expected, f"{OBS_CLASS}.{name} changed"


def test_forward_zero_steering_history_default_is_true():
    cls = _obs_class()
    assigns = [
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ZERO_STEERING_HISTORY" for t in n.targets)
    ]
    assert len(assigns) == 1, f"{OBS_CLASS}.ZERO_STEERING_HISTORY not declared (once)"
    assert ast.literal_eval(assigns[0].value) is EXPECTED_ZERO_STEERING_HISTORY


def test_history_blanking_is_guarded_by_the_class_attribute():
    """``obs_buf[:, 6:8] = 0`` happens if and only if ``self.ZERO_STEERING_HISTORY``."""
    cls = _obs_class()
    call = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    assert len(call) == 1
    guards = [
        ast.unparse(n)
        for n in ast.walk(call[0])
        if isinstance(n, ast.If) and "obs_buf[:, 6:8] = 0" in ast.unparse(n)
    ]
    assert guards == [ZERO_STEERING_GUARD], guards
    # and nowhere else in the file does the blanking happen unguarded
    tree = ast.parse(OBSERVATIONS_PY.read_text(), str(OBSERVATIONS_PY))
    blanks = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and ast.unparse(n) == "obs_buf[:, 6:8] = 0"
    ]
    assert len(blanks) == 1


def test_call_uses_the_hooks():
    cls = _obs_class()
    call = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"][0]
    src = ast.unparse(call)
    assert "delta_yaw, delta_next_yaw = self._steering_channels(env, on_flat)" in src
    assert "self._command_channels(commands)" in src
    # the inline forms must be gone -- one behaviour, one place
    assert "0 * commands[:, 0:2]" not in src
    assert "os.environ.get('KRABBY_HEADING')" not in src


def test_forward_pitch_penalty_stays_inert():
    ps = _summary()
    terms = ps.reward_terms_from_cfg_source()
    weights = {t["name"]: t["default_weight"] for t in terms}
    assert weights["penalty_base_pitch_forward_linear"] == 0.0
    knobs = ps.knob_overrides_from_cfg_source()
    assert [k for k, term in knobs.items() if term == "penalty_base_pitch_forward_linear"] == []
