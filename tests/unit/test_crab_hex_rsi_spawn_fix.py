"""PLAN H B4: RSI resets are placed where reset_root_state places (mock env, no Isaac).

Contract: with ``fix_spawn=True`` the RSI root x equals origin_x - (size_y + offset) --
the reset_root_state formula -- and y equals origin_y; unarmed keeps the historical
tile-centre placement (origin xy) bit-for-bit. Every RSI reset is reported to the parkour
term with kind SPAWN_RSI.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
MDP_DIR = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

# --- stubs: isaaclab.utils.math.quat_apply (other suites may have stubbed ``isaaclab``
# already without the math submodule, so patch whatever is present) ---
def _quat_apply(q, v):  # identity rotation is all the test needs
    return v


_isaaclab = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
_utils = sys.modules.setdefault("isaaclab.utils", getattr(_isaaclab, "utils", None) or types.ModuleType("isaaclab.utils"))
_math_mod = sys.modules.setdefault("isaaclab.utils.math", getattr(_utils, "math", None) or types.ModuleType("isaaclab.utils.math"))
if not hasattr(_math_mod, "quat_apply"):
    _math_mod.quat_apply = _quat_apply
_utils.math = _math_mod
_isaaclab.utils = _utils

NOTES = []
_events = types.ModuleType("parkour_isaaclab.envs.mdp.events")
_events.note_spawn_to_parkour = lambda env, ids, x_w, kind: NOTES.append((ids.clone(), x_w.clone(), kind))
_stats = types.ModuleType("parkour_isaaclab.envs.mdp.parkours.exposure_stats")
_stats.SPAWN_RSI = 2
for name, mod in {
    "parkour_isaaclab": types.ModuleType("parkour_isaaclab"),
    "parkour_isaaclab.envs": types.ModuleType("parkour_isaaclab.envs"),
    "parkour_isaaclab.envs.mdp": types.ModuleType("parkour_isaaclab.envs.mdp"),
    "parkour_isaaclab.envs.mdp.events": _events,
    "parkour_isaaclab.envs.mdp.parkours": types.ModuleType("parkour_isaaclab.envs.mdp.parkours"),
    "parkour_isaaclab.envs.mdp.parkours.exposure_stats": _stats,
}.items():
    sys.modules.setdefault(name, mod)

import crab_hex_rsi  # noqa: E402

SIZE_Y, OFFSET = 4.0, 3.0


class _Robot:
    def __init__(self, n):
        self.data = types.SimpleNamespace(default_root_state=torch.zeros(n, 13))
        self.data.default_root_state[:, 2] = 0.3
        self.data.default_root_state[:, 3] = 1.0
        self.poses = {}

    def write_root_pose_to_sim(self, pose, env_ids):
        self.poses["pose"] = (env_ids.clone(), pose.clone())

    def write_root_velocity_to_sim(self, vel, env_ids):
        pass

    def write_joint_state_to_sim(self, jp, jv, env_ids):
        pass


class _Scene:
    def __init__(self, robot, origins):
        self._items = {"robot": robot}
        self.env_origins = origins
        self.terrain = types.SimpleNamespace(
            cfg=types.SimpleNamespace(terrain_generator=types.SimpleNamespace(size=(16.0, SIZE_Y)))
        )

    def __getitem__(self, key):
        return self._items[key]


def _env(n=8):
    robot = _Robot(n)
    origins = torch.zeros(n, 3)
    origins[:, 0] = torch.arange(n).float() * 16.0 + 100.0
    origins[:, 1] = 7.0
    env = types.SimpleNamespace(
        device="cpu",
        scene=_Scene(robot, origins),
        event_manager=types.SimpleNamespace(get_term_cfg=lambda name: types.SimpleNamespace(params={"offset": OFFSET})),
        action_manager=types.SimpleNamespace(get_term=lambda name: types.SimpleNamespace(rsi_clock_staged=torch.zeros(n))),
    )
    return env, robot, origins


@pytest.fixture()
def bank(tmp_path):
    m = 5
    path = tmp_path / "bank.npz"
    np.savez(path, joint_pos=np.zeros((m, 18), np.float32), joint_vel=np.zeros((m, 18), np.float32),
             root_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (m, 1)), root_z=np.full(m, 0.31, np.float32),
             root_lin_vel_b=np.zeros((m, 3), np.float32), clock_phase=np.zeros(m, np.float32))
    crab_hex_rsi._BANK_CACHE.clear()
    return str(path)


def test_fixed_spawn_matches_reset_root_state_formula(bank):
    NOTES.clear()
    env, robot, origins = _env()
    ids = torch.arange(8)
    crab_hex_rsi.reset_from_reference_states(env, ids, bank, fraction=1.0, fix_spawn=True)
    env_ids, pose = robot.poses["pose"]
    expected_x = origins[:, 0] - (SIZE_Y + OFFSET)  # == reset_root_state: origin - (size[1] + offset)
    assert torch.allclose(pose[:, 0], expected_x)
    assert torch.allclose(pose[:, 1], origins[:, 1])
    assert torch.allclose(pose[:, 2], torch.full((8,), 0.31))
    # reported to the parkour term as RSI spawns
    assert len(NOTES) == 1 and NOTES[0][2] == 2
    assert torch.allclose(NOTES[0][1], expected_x)


def test_unarmed_keeps_tile_centre_placement(bank):
    NOTES.clear()
    env, robot, origins = _env()
    ids = torch.arange(8)
    crab_hex_rsi.reset_from_reference_states(env, ids, bank, fraction=1.0)
    _, pose = robot.poses["pose"]
    assert torch.equal(pose[:, 0], origins[:, 0])
    assert torch.equal(pose[:, 1], origins[:, 1])
    assert len(NOTES) == 1 and NOTES[0][2] == 2


def test_fraction_zero_seeds_nothing(bank):
    NOTES.clear()
    env, robot, _ = _env()
    crab_hex_rsi.reset_from_reference_states(env, torch.arange(8), bank, fraction=0.0, fix_spawn=True)
    assert "pose" not in robot.poses and NOTES == []
