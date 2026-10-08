"""Omni phase-2 stage p2b (``P2B_PLAN.md`` step 1, 2026-10-08): the tilt-guard and terrain-relative-lift knobs.

Both are DEFAULT OFF and set by no preset. Pinned here, without Isaac:

* the knob module (``config/crab_hex/motion_knobs.py``): ``KRABBY_ORIENT_ALL_TILES`` (TG) and
  ``KRABBY_FOOT_CLEAR_TERRAIN`` (CL) -- flag parsers, registration in ``MOTION_KNOB_PARSERS`` (and so
  the ladder's parser table), env-side (not runner-side, not eval-carried), and the applier on a
  stand-in cfg: unset / blank / ``0`` writes NOTHING (the rewards block is identical), armed swaps
  exactly the named funcs and keeps every weight and param, a missing term is refused before
  anything is swapped;
* the pure kernels (``mdp/omni_clearance.py``): the terrain sample under a foot over a pit, over a
  block, across a tile seam (the neighbour's pixel) and off the terrain (0), and that the root never
  enters the height;
* the env-facing twins on a stand-in env, against the LIBRARY terms lifted out of
  ``parkour_isaaclab/envs/mdp/rewards.py`` with ``ast`` and exec'd against stubs: equal to the library
  on bare ground at the nominal root height, unchanged by a lowered root at fixed feet (where the
  library pays the crouch), pit / block measured from the pit floor / block top, the same command
  gate and parkour_flat mask, the height-field cache.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task/mdp"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
OMNI_PHASES_PY = OMNI / "config/crab_hex/crab_hex_omni_phases.py"
CLEARANCE_PY = OMNI / "mdp/omni_clearance.py"
FWD_MDP_CFG_PY = TASKS / "crab_hex_forward_task/config/crab_hex/agents/parkour_mdp_cfg.py"
LIB_REWARDS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/rewards.py"
PARKOURS_DIR = REPO / "parkour/parkour_isaaclab/envs/mdp/parkours"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"

ORIENT, CLEAR = "KRABBY_ORIENT_ALL_TILES", "KRABBY_FOOT_CLEAR_TERRAIN"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Get-or-create loader of ``test_crab_hex_omni_field_exposure.py``: the real source dirs as the
    parent packages' ``__path__``, their ``__init__`` never executed (so relative imports resolve)."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    parts = dotted.split(".")
    directory = path.parent
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(directory)]
            sys.modules[name] = pkg
        elif str(directory) not in getattr(pkg, "__path__", []):
            pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
        directory = directory.parent
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


def _source_of(path: Path, name: str) -> str:
    src = path.read_text()
    node = [n for n in ast.parse(src, str(path)).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


mk = _load(MOTION_KNOBS_PY, "motion_knobs__p2b_test")
xs = _load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
oc = _load_real_module("parkour_tasks.crab_hex_omni_task.mdp.omni_clearance", CLEARANCE_PY)
if str(FWD_MDP) not in sys.path:
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_clock_reward as clock  # noqa: E402


# ======================================================================= parsers / registration
def test_parsers_and_registration():
    assert mk.ORIENT_ALL_TILES_VAR == ORIENT and mk.FOOT_CLEAR_TERRAIN_VAR == CLEAR
    for var in (ORIENT, CLEAR):
        parse = mk.MOTION_KNOB_PARSERS[var]
        assert parse("1") is True
        for off in (None, "", " ", "0"):
            assert parse(off) is False, (var, off)
        for bad in ("2", "true", "on", "yes"):
            with pytest.raises(ValueError, match=var):
                parse(bad)
        # env side (params/env.yaml), a reward func swap is not an eval semantic: never carried
        assert var not in mk.AGENT_KNOBS and var not in mk.EVAL_SEMANTIC_KNOBS
        assert var not in mk.EVAL_STRUCTURE_KNOBS and var not in mk.KNOB_OFF_VALUES
    assert {"ORIENT_ALL_TILES_VAR", "FOOT_CLEAR_TERRAIN_VAR", "FUNC_SWAPS"} <= set(mk.__all__)
    assert mk.FUNC_SWAPS == {
        ORIENT: {"reward_orientation": "reward_orientation_upright"},
        CLEAR: {"reward_foot_clearance": "reward_foot_clearance_terrain",
                "penalty_swing_min_clearance": "penalty_swing_min_clearance_terrain"},
    }


def test_the_ladder_parses_them():
    """The driver's parser table takes ``MOTION_KNOB_PARSERS`` whole (``**k.MOTION_KNOB_PARSERS``)."""
    rc = _load(RUN_CANDIDATES_PY, "run_candidates__p2b_test")
    parsers = rc.knob_parsers()
    for var in (ORIENT, CLEAR):
        assert parsers[var]("1") is True and parsers[var]("0") is False, var


def test_no_preset_arms_them():
    src = OMNI_PHASES_PY.read_text() + (TASKS / "crab_hex_forward_task/config/crab_hex/crab_hex_phases.py").read_text()
    for var in (ORIENT, CLEAR):
        assert var not in src, var


def test_knob_module_stays_pure():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('mk', {str(MOTION_KNOBS_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['mk'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"assert mod.MOTION_KNOB_PARSERS[{CLEAR!r}]('1') is True\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'parkour_isaaclab', 'isaaclab', 'torch'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_swap_targets_exist_by_source():
    """The real (lazy) swap targets: ``reward_orientation_upright`` in the library, the two twins in
    ``omni_clearance`` -- with the library's exact signatures, so the swap can pass params through."""
    assert mk._LIB_REWARDS_MOD == "parkour_isaaclab.envs.mdp.rewards"
    assert mk._CLEARANCE_MOD == "parkour_tasks.crab_hex_omni_task.mdp.omni_clearance"
    lib_src = LIB_REWARDS_PY.read_text()
    assert "def reward_orientation_upright(" in lib_src
    ns = _lib_ns()
    for twin, stock in (("reward_foot_clearance_terrain", "reward_foot_clearance"),
                        ("penalty_swing_min_clearance_terrain", "penalty_swing_min_clearance")):
        got, want = inspect.signature(getattr(oc, twin)), inspect.signature(ns[stock])
        assert [(p.name, p.default) for p in got.parameters.values()] == \
               [(p.name, p.default) for p in want.parameters.values()], twin
    # the stock upright form takes the same params as the obstacle-zeroed one
    assert [p.name for p in inspect.signature(ns["reward_orientation_upright"]).parameters.values()] == \
           [p.name for p in inspect.signature(ns["reward_orientation"]).parameters.values()]


def test_shipped_cfgs_are_untouched():
    """Default identity at the SOURCE: the forward rewards cfg still declares the stock funcs, and no
    shipped cfg names the twins or the knobs (they exist only when a knob swaps them in)."""
    fwd = FWD_MDP_CFG_PY.read_text()
    assert "func=mdp_rewards.reward_orientation,\n" in fwd
    assert "func=mdp_rewards.reward_foot_clearance,\n" in fwd
    assert "func=mdp_rewards.penalty_swing_min_clearance,\n" in fwd
    for path in (FWD_MDP_CFG_PY, OMNI / "mdp/crab_hex_omni_rewards.py", OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py"):
        text = path.read_text()
        assert "omni_clearance" not in text and ORIENT not in text and CLEAR not in text, path


# ======================================================================= the applier (stand-ins)
class _Bag:
    pass


class _Term:
    def __init__(self, func=None, weight=0.0, params=None):
        self.func, self.weight, self.params = func, weight, params


def _clear_params(min_m):
    return {"asset_cfg": ("robot", ".*_Footpad"), "sensor_cfg": ("contact_forces", ".*_Footpad"),
            "command_name": "base_velocity", "contact_force_threshold": 0.1, "min_clearance_m": min_m,
            "min_forward_speed_cmd": 0.12, "ground_offset_from_root_m": -1.05, "parkour_name": None}


def _cfg():
    """The rewards block a CL / TG dry run composes (KRABBY_FOOT_CLEAR_W 1.0 / _MIN 0.03 / _FLAT 1 /
    SWING_MIN_CLEAR_W -0.4 / ORIENT_W -1.5 already applied by the forward __post_init__)."""
    cfg = _Bag()
    cfg.commands = _Bag()
    cfg.commands.base_velocity = _Bag()
    cfg.actions = _Bag()
    cfg.actions.joint_pos = _Bag()
    cfg.rewards = _Bag()
    cfg.rewards.reward_action_rate = _Term(func="rate", weight=-0.05, params={"asset_cfg": "robot"})
    cfg.rewards.reward_orientation = _Term(func="reward_orientation", weight=-1.5,
                                           params={"asset_cfg": ("robot",), "parkour_name": "base_parkour"})
    cfg.rewards.reward_foot_clearance = _Term(func="reward_foot_clearance", weight=1.0,
                                              params=dict(_clear_params(0.03), max_clearance_m=0.26))
    cfg.rewards.penalty_swing_min_clearance = _Term(func="penalty_swing_min_clearance", weight=-0.4,
                                                    params=_clear_params(0.04))
    cfg.observations = _Bag()
    cfg.observations.policy = _Bag()
    return cfg


def _snap(obj):
    """``class_to_dict`` of the stand-in: every field, recursively, funcs by identity."""
    if isinstance(obj, (_Bag, _Term)):
        return {k: _snap(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    return obj


SWAPS = {"reward_orientation": "func:upright", "reward_foot_clearance": "func:clear_terrain",
         "penalty_swing_min_clearance": "func:swing_terrain"}


def _apply(cfg, env):
    mk.apply_motion_knobs(cfg, env, swap_funcs=SWAPS)


@pytest.mark.parametrize("env", [{}, {ORIENT: "", CLEAR: ""}, {ORIENT: "0", CLEAR: "0"}, {ORIENT: " ", CLEAR: "0"}])
def test_unset_blank_or_off_writes_nothing(env, capsys):
    pristine = _snap(_cfg())
    cfg = _cfg()
    _apply(cfg, env)
    assert _snap(cfg) == pristine
    assert "[KNOB]" not in capsys.readouterr().out
    # ... and off never even looks for its terms (a cfg without them is fine)
    bare = _cfg()
    for name in SWAPS:
        delattr(bare.rewards, name)
    _apply(bare, env)


def test_orient_all_tiles_swaps_the_func_only(capsys):
    cfg = _cfg()
    before = _snap(cfg)
    _apply(cfg, {ORIENT: "1"})
    term = cfg.rewards.reward_orientation
    assert term.func == "func:upright"
    assert term.weight == -1.5 and term.params == {"asset_cfg": ("robot",), "parkour_name": "base_parkour"}
    after = _snap(cfg)
    after["rewards"]["reward_orientation"]["func"] = "reward_orientation"
    assert after == before                                    # nothing else written
    assert f"[KNOB] {ORIENT}=1 consumed" in capsys.readouterr().out
    missing = _cfg()
    del missing.rewards.reward_orientation
    with pytest.raises(RuntimeError, match="reward_orientation"):
        _apply(missing, {ORIENT: "1"})
    with pytest.raises(ValueError, match=ORIENT):
        _apply(_cfg(), {ORIENT: "yes"})


def test_foot_clear_terrain_swaps_both_funcs_only(capsys):
    cfg = _cfg()
    before = _snap(cfg)
    _apply(cfg, {CLEAR: "1"})
    r = cfg.rewards
    assert r.reward_foot_clearance.func == "func:clear_terrain"
    assert r.penalty_swing_min_clearance.func == "func:swing_terrain"
    assert r.reward_orientation.func == "reward_orientation"  # the other knob's term untouched
    after = _snap(cfg)
    after["rewards"]["reward_foot_clearance"]["func"] = "reward_foot_clearance"
    after["rewards"]["penalty_swing_min_clearance"]["func"] = "penalty_swing_min_clearance"
    assert after == before                                    # weights, floors, FLAT mask, offset kept
    assert f"[KNOB] {CLEAR}=1 consumed" in capsys.readouterr().out
    # a missing term is refused BEFORE anything is swapped (no half-swapped CL cfg)
    half = _cfg()
    del half.rewards.penalty_swing_min_clearance
    with pytest.raises(RuntimeError, match="penalty_swing_min_clearance"):
        _apply(half, {CLEAR: "1"})
    assert half.rewards.reward_foot_clearance.func == "reward_foot_clearance"


def test_a_swap_at_zero_weight_is_refused():
    """Review of the step-1 diff (2026-10-08): arming a swap whose every term weighs 0 trains nothing -- refused at cfg
    build (like KRABBY_POWER_JOINTS), not left to the +200 liveness kill. One non-zero term is enough."""
    cfg = _cfg()
    cfg.rewards.reward_foot_clearance.weight = 0.0
    cfg.rewards.penalty_swing_min_clearance.weight = 0.0
    with pytest.raises(RuntimeError, match="weight 0"):
        _apply(cfg, {CLEAR: "1"})
    assert cfg.rewards.reward_foot_clearance.func == "reward_foot_clearance"      # nothing swapped
    one = _cfg()
    one.rewards.penalty_swing_min_clearance.weight = 0.0
    _apply(one, {CLEAR: "1"})
    assert one.rewards.reward_foot_clearance.func == "func:clear_terrain"
    zero_tg = _cfg()
    zero_tg.rewards.reward_orientation.weight = 0.0
    with pytest.raises(RuntimeError, match="weight 0"):
        _apply(zero_tg, {ORIENT: "1"})


def test_both_knobs_compose():
    cfg = _cfg()
    _apply(cfg, {ORIENT: "1", CLEAR: "1", "KRABBY_STAND_POSE_W": "0"})
    assert [getattr(cfg.rewards, n).func for n in SWAPS] == list(SWAPS.values())


def test_real_swap_funcs_resolve_by_name(monkeypatch):
    """``_default_swap_funcs`` imports the right module per knob and takes the FUNC_SWAPS names."""
    seen = []

    def fake_import(name):
        seen.append(name)
        return types.SimpleNamespace(**{v: f"{name}:{v}" for d in mk.FUNC_SWAPS.values() for v in d.values()})

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert mk._default_swap_funcs(ORIENT) == {
        "reward_orientation": "parkour_isaaclab.envs.mdp.rewards:reward_orientation_upright"}
    got = mk._default_swap_funcs(CLEAR)
    assert got == {"reward_foot_clearance": f"{mk._CLEARANCE_MOD}:reward_foot_clearance_terrain",
                   "penalty_swing_min_clearance": f"{mk._CLEARANCE_MOD}:penalty_swing_min_clearance_terrain"}
    assert seen == [mk._LIB_REWARDS_MOD, mk._CLEARANCE_MOD]


# ======================================================================= the pure kernels
# The shipped geometry: 10 m tiles, 0.08 m pixels -> 126 px stored, placed 125 px apart; 5 mm units.
SIZE, HS, TILE_PX, GRID, VS = (10.0, 10.0), 0.08, (126, 126), (3, 4), 0.005
PIT_Z, BLOCK_Z, SEAM_Z, CORNER_Z = -0.30, 0.20, 0.15, 0.25


def _centre(row: int, col: int) -> torch.Tensor:
    """World xy of tile (row, col)'s centre, with the generator's whole-terrain centring."""
    return torch.tensor([(row + 0.5) * SIZE[0] - GRID[0] * SIZE[0] / 2,
                         (col + 0.5) * SIZE[1] - GRID[1] * SIZE[1] / 2])


def _height_fields() -> np.ndarray:
    hf = np.zeros((GRID[0], GRID[1], *TILE_PX), dtype=np.int16)
    hf[1, 1, 30:41, 30:41] = round(PIT_Z / VS)      # a pit on tile (1, 1), centre pixel 35
    hf[1, 1, 80:91, 80:91] = round(BLOCK_Z / VS)    # a block on tile (1, 1), centre pixel 85
    hf[2, 1, 0:9, 58:68] = round(SEAM_Z / VS)       # a block just past tile (1, 1)'s +x seam
    hf[0, 0] = round(CORNER_Z / VS)                 # the corner tile raised everywhere
    return hf


def _xy(row: int, col: int, px: float, py: float) -> tuple[float, float]:
    """Tile-local pixel (may lie past the tile) -> world xy, on tile (row, col)'s frame."""
    c = _centre(row, col)
    return float(c[0] - 0.5 * SIZE[0] + px * HS), float(c[1] - 0.5 * SIZE[1] + py * HS)


HF_M = oc.global_height_field_m(_height_fields(), VS)


def _h(points, row=1, col=1):
    """foot_height_above_terrain for one env with feet ``[(x, y, z), ...]`` on tile (row, col)."""
    feet = torch.tensor([points], dtype=torch.float32)
    return oc.foot_height_above_terrain(feet, _centre(row, col).unsqueeze(0), torch.tensor([row]),
                                        torch.tensor([col]), HF_M, SIZE, HS, TILE_PX, GRID)[0]


def test_global_field_is_metres_assembled_like_the_exposure_field():
    assert HF_M.dtype == torch.float32 and HF_M.shape == (GRID[0] * 126, GRID[1] * 126)
    raw = xs.global_height_field(torch.from_numpy(_height_fields()))
    assert torch.allclose(HF_M, raw.float() * VS)
    assert torch.isclose(HF_M[1 * 126 + 35, 1 * 126 + 35], torch.tensor(PIT_Z))


def test_pit_block_and_bare_ground():
    """A foot over the pit is measured from the pit FLOOR, one over the block from its TOP."""
    pit, block, bare = _xy(1, 1, 35, 35), _xy(1, 1, 85, 85), _xy(1, 1, 62, 62)
    h = _h([(*pit, 0.02), (*block, 0.25), (*bare, 0.05), (*pit, -0.25)])
    assert torch.allclose(h, torch.tensor([0.02 - PIT_Z, 0.25 - BLOCK_Z, 0.05, -0.25 - PIT_Z]), atol=1e-6)
    assert h[0] > 0.02                                     # more than a root-relative form at that z


def test_seam_reads_the_neighbour_tile():
    """0.32 m past the +x seam (neighbour pixel 4): the neighbour's block, not the own tile's border
    (the own-tile clamp would read pixel 125 of tile (1, 1), bare ground)."""
    x, y = _xy(1, 1, 125 + 4, 63)
    h = _h([(x, y, 0.30)])
    assert torch.isclose(h[0], torch.tensor(0.30 - SEAM_Z), atol=1e-6)
    # the same point read from the neighbour's own frame agrees
    assert torch.isclose(_h([(x, y, 0.30)], row=2, col=1)[0], h[0])


def test_off_terrain_reads_zero():
    """Past the outermost tiles terrain_z = 0, although the clamped index lands on the raised corner."""
    on = _xy(0, 0, 62, 62)
    off_x, off_y = _xy(0, 0, -4, 62), _xy(0, 0, 62, -4)
    h = _h([(*on, 0.4), (*off_x, 0.4), (*off_y, 0.4)], row=0, col=0)
    assert torch.allclose(h, torch.tensor([0.4 - CORNER_Z, 0.4, 0.4]), atol=1e-6)
    tz = oc.terrain_height_under(torch.tensor([[off_x[0]]]), torch.tensor([[off_x[1]]]), _centre(0, 0).unsqueeze(0),
                                 torch.tensor([0]), torch.tensor([0]), HF_M, SIZE, HS, TILE_PX, GRID)
    assert float(tz) == 0.0


def test_per_env_tiles_and_shapes():
    """Rows on different own tiles in one batch; (N, F) in, (N, F) out."""
    feet = torch.tensor([[[*_xy(1, 1, 35, 35), 0.0], [*_xy(1, 1, 62, 62), 0.1]],
                         [[*_xy(0, 0, 62, 62), 0.3], [*_xy(0, 0, -4, 62), 0.3]]])
    centres = torch.stack([_centre(1, 1), _centre(0, 0)])
    h = oc.foot_height_above_terrain(feet, centres, torch.tensor([1, 0]), torch.tensor([1, 0]), HF_M,
                                     SIZE, HS, TILE_PX, GRID)
    assert h.shape == (2, 2)
    assert torch.allclose(h, torch.tensor([[-PIT_Z, 0.1], [0.3 - CORNER_Z, 0.3]]), atol=1e-6)


# ======================================================================= the env-facing twins
def _lib_ns():
    """The library terms and helpers, lifted out of rewards.py and exec'd against stubs."""
    ns = {"torch": torch, "SceneEntityCfg": lambda *a, **k: None, "Articulation": object, "ParkourEvent": object,
          "ParkourManagerBasedRLEnv": object, "clock_command_speed": clock.clock_command_speed}
    for name in ("_clock_cmd_speed", "_cmd_gate", "_parkour_flat_mask", "_foot_swing_height_above_ground",
                 "reward_foot_clearance", "penalty_swing_min_clearance", "reward_orientation",
                 "reward_orientation_upright"):
        exec(compile("from __future__ import annotations\n" + _source_of(LIB_REWARDS_PY, name),
                     str(LIB_REWARDS_PY), "exec"), ns)
    return ns


LIB = _lib_ns()
FEET = [1, 2, 3, 4, 5, 6]          # body ids of the six feet (robot and sensor share them, as in the lib)


class _Scene(dict):
    def __init__(self, robot, sensors):
        super().__init__(robot=robot)
        self.sensors = sensors


def _env(feet_xyz, *, root_z=1.0, row=1, col=1, swing=None, cmd=(0.3, 0.0, 0.0), on_flat=False,
         hf=None, omni=True):
    """A one-tile stand-in env: feet (N, 6, 3) world positions, the omni term's tile frame."""
    feet = torch.as_tensor(feet_xyz, dtype=torch.float32)
    n = feet.shape[0]
    body_pos = torch.zeros(n, 7, 3)
    body_pos[:, FEET] = feet
    root = torch.zeros(n, 3)
    root[:, :2] = _centre(row, col)
    root[:, 2] = root_z
    swing = torch.ones(n, 6, dtype=torch.bool) if swing is None else torch.as_tensor(swing)
    forces = torch.zeros(n, 3, 7, 3)
    forces[:, 0, FEET, 2] = torch.where(swing, torch.tensor(0.0), torch.tensor(300.0))
    robot = types.SimpleNamespace(data=types.SimpleNamespace(body_pos_w=body_pos, root_pos_w=root))
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w_history=forces))
    terrain = types.SimpleNamespace(
        terrain_levels=torch.full((n,), row), terrain_types=torch.full((n,), col),
        terrain_generator_class=types.SimpleNamespace(height_fields=_height_fields() if hf is None else hf),
        cfg=types.SimpleNamespace(terrain_generator=types.SimpleNamespace(vertical_scale=VS)))
    term = types.SimpleNamespace(
        env_origins=torch.cat([_centre(row, col).expand(n, 2), torch.zeros(n, 1)], dim=1), terrain=terrain,
        _tile_size=SIZE, _hscale=HS, _tile_px=TILE_PX, _grid_rc=GRID,
        env_per_terrain_name=np.array([["parkour_flat" if on_flat else "omni_field"]] * n))
    command = torch.tensor([cmd] * n, dtype=torch.float32)
    return types.SimpleNamespace(
        num_envs=n, device="cpu", scene=_Scene(robot, {"contact_forces": sensor}),
        cfg=types.SimpleNamespace(omni_command=omni, omni_clock_turn_radius=0.3),
        command_manager=types.SimpleNamespace(get_command=lambda name: command),
        parkour_manager=types.SimpleNamespace(get_term=lambda name: term))


@pytest.fixture(autouse=True)
def _lib_helpers(monkeypatch):
    """The twins' lazy library import -> the lifted helpers (the real module needs Isaac)."""
    monkeypatch.setattr(oc, "_lib_rewards", lambda: types.SimpleNamespace(**LIB))


ASSET = types.SimpleNamespace(name="robot", body_ids=FEET)
SENSOR = types.SimpleNamespace(name="contact_forces", body_ids=FEET)
CLEAR_KW = {"min_clearance_m": 0.03, "max_clearance_m": 0.26, "ground_offset_from_root_m": -1.0}
SWING_KW = {"min_clearance_m": 0.04, "ground_offset_from_root_m": -1.0}


def _bare_feet(z, n=1):
    """Six feet over bare ground of tile (1, 1), at heights ``z`` (len 6)."""
    xy = [_xy(1, 1, 50 + 5 * i, 62) for i in range(6)]
    return [[(x, y, zi) for (x, y), zi in zip(xy, z)]] * n


def _both(env, lib_name, kw, parkour_name=None):
    twin = getattr(oc, lib_name + "_terrain")(env, ASSET, SENSOR, parkour_name=parkour_name, **kw)
    stock = LIB[lib_name](env, ASSET, SENSOR, parkour_name=parkour_name, **kw)
    return twin, stock


@pytest.mark.parametrize("lib_name,kw", [("reward_foot_clearance", CLEAR_KW),
                                         ("penalty_swing_min_clearance", SWING_KW)])
def test_equal_to_the_library_on_bare_ground_at_the_nominal_root(lib_name, kw):
    """Root at 1.0 m with offset -1.0: the library's nominal ground IS the bare terrain, so the twin
    is the library term (swing mask, floor, clamp, mean, gate) to float precision."""
    z = [0.0, 0.02, 0.035, 0.05, 0.12, 0.40]
    swing = [[True, True, True, False, True, True]]
    env = _env(_bare_feet(z), swing=swing)
    twin, stock = _both(env, lib_name, kw)
    assert torch.allclose(twin, stock, atol=1e-7) and float(stock) > 0.0


@pytest.mark.parametrize("lib_name,kw", [("reward_foot_clearance", CLEAR_KW),
                                         ("penalty_swing_min_clearance", SWING_KW)])
def test_a_lowered_root_at_fixed_feet_changes_nothing(lib_name, kw):
    feet = _bare_feet([0.0, 0.02, 0.035, 0.05, 0.08, 0.12])
    hi, lo = _env(feet, root_z=1.0), _env(feet, root_z=0.97)
    twin_hi, stock_hi = _both(hi, lib_name, kw)
    twin_lo, stock_lo = _both(lo, lib_name, kw)
    assert torch.equal(twin_hi, twin_lo)                   # terrain-relative: the crouch earns nothing
    assert not torch.equal(stock_hi, stock_lo)             # root-relative: the crouch moved the term


def test_pit_and_block_on_the_income():
    """One foot in swing over the pit 2 cm below the bare ground, one on the block 5 cm above its top."""
    pit, block = _xy(1, 1, 35, 35), _xy(1, 1, 85, 85)
    others = _bare_feet([0.0] * 6)[0][2:]
    swing = [[True, True, False, False, False, False]]
    env = _env([[(*pit, -0.02), (*block, BLOCK_Z + 0.05), *others]], swing=swing)
    twin, stock = _both(env, "reward_foot_clearance", CLEAR_KW)
    pit_h, block_h = -0.02 - PIT_Z, 0.05
    want = (min(pit_h - 0.03, 0.26) + (block_h - 0.03)) / 6.0
    assert torch.isclose(twin, torch.tensor([want]), atol=1e-6)
    # root-relative, the pit foot is BELOW ground (no income) and the block foot is 0.25 m up
    assert torch.isclose(stock, torch.tensor([(BLOCK_Z + 0.05 - 0.03) / 6.0]), atol=1e-6)


def test_pit_and_block_on_the_micro_swing_penalty():
    """A swing foot 2 cm above the BLOCK TOP is a micro-swing; one hanging into the pit is not."""
    pit, block = _xy(1, 1, 35, 35), _xy(1, 1, 85, 85)
    others = _bare_feet([0.0] * 6)[0][2:]
    swing = [[True, True, False, False, False, False]]
    env = _env([[(*pit, 0.0), (*block, BLOCK_Z + 0.02), *others]], swing=swing)
    twin, stock = _both(env, "penalty_swing_min_clearance", SWING_KW)
    assert torch.isclose(twin, torch.tensor([1.0 / 6.0]))  # the block foot only
    assert torch.isclose(stock, torch.tensor([1.0 / 6.0]))  # ... the PIT foot only, root-relative
    pit_only = _env([[(*pit, 0.0), *_bare_feet([0.1] * 6)[0][1:]]], swing=[[True] + [False] * 5])
    assert float(oc.penalty_swing_min_clearance_terrain(pit_only, ASSET, SENSOR, parkour_name=None, **SWING_KW)) == 0.0


@pytest.mark.parametrize("lib_name,kw", [("reward_foot_clearance", CLEAR_KW),
                                         ("penalty_swing_min_clearance", SWING_KW)])
def test_command_gate_and_flat_mask_are_the_librarys(lib_name, kw):
    z = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    twin = getattr(oc, lib_name + "_terrain")
    # a stand (zero command) gates both off; a pure turn gates them on under omni (clock speed)
    assert float(twin(_env(_bare_feet(z), cmd=(0.0, 0.0, 0.0)), ASSET, SENSOR, parkour_name=None, **kw)) == 0.0
    assert float(twin(_env(_bare_feet(z), cmd=(0.0, 0.0, 1.0)), ASSET, SENSOR, parkour_name=None, **kw)) > 0.0
    # parkour_flat: masked with a parkour name, paid with None (KRABBY_FOOT_CLEAR_FLAT=1)
    flat = _env(_bare_feet(z), on_flat=True)
    assert float(twin(flat, ASSET, SENSOR, parkour_name="base_parkour", **kw)) == 0.0
    assert float(twin(flat, ASSET, SENSOR, parkour_name=None, **kw)) > 0.0
    for env in (_env(_bare_feet(z), cmd=(-0.3, 0.0, 0.0)), _env(_bare_feet(z), cmd=(0.0, 0.0, 1.0), omni=False)):
        t, s = _both(env, lib_name, kw, parkour_name="base_parkour")
        assert torch.allclose(t, s, atol=1e-7)


def test_ground_offset_is_ignored_and_no_sensor_is_zero():
    env = _env(_bare_feet([0.0, 0.05, 0.1, 0.1, 0.2, 0.3]))
    a = oc.reward_foot_clearance_terrain(env, ASSET, SENSOR, parkour_name=None, **CLEAR_KW)
    b = oc.reward_foot_clearance_terrain(env, ASSET, SENSOR, parkour_name=None,
                                         **dict(CLEAR_KW, ground_offset_from_root_m=-0.5))
    assert torch.equal(a, b)
    nosensor = types.SimpleNamespace(name="missing", body_ids=FEET)
    assert torch.equal(oc.reward_foot_clearance_terrain(env, ASSET, nosensor), torch.zeros(1))
    assert torch.equal(oc.penalty_swing_min_clearance_terrain(env, ASSET, nosensor), torch.zeros(1))


def test_height_field_is_built_once_and_cached_on_the_env():
    hf = _height_fields()
    env = _env(_bare_feet([0.1] * 6), hf=hf)
    first = oc.reward_foot_clearance_terrain(env, ASSET, SENSOR, parkour_name=None, **CLEAR_KW)
    cached = getattr(env, oc.HEIGHT_FIELD_CACHE_ATTR)
    assert torch.allclose(cached, HF_M)
    hf[1, 1] = 1000                                        # a later change to the generator array ...
    again = oc.reward_foot_clearance_terrain(env, ASSET, SENSOR, parkour_name=None, **CLEAR_KW)
    assert torch.equal(first, again) and getattr(env, oc.HEIGHT_FIELD_CACHE_ATTR) is cached  # ... is not re-read


def test_refuses_a_parkour_term_without_the_omni_tile_frame():
    env = _env(_bare_feet([0.1] * 6))
    term = env.parkour_manager.get_term("base_parkour")
    del term._grid_rc
    with pytest.raises(RuntimeError, match="_grid_rc"):
        oc.reward_foot_clearance_terrain(env, ASSET, SENSOR, parkour_name=None, **CLEAR_KW)
    assert oc.TERRAIN_TERM_NAME == "base_parkour"
