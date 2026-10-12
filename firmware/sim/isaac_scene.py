"""Load a single-env crab hex Isaac Lab scene for firmware-in-the-loop tests.

Call only after ``AppLauncher`` has started (same ordering as ``hal/server/isaac/main.py`` and
``verify_crab_joint_drive.py``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from firmware.sim.simulated_robot_config import SimulatedRobotConfig

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

DEFAULT_CRAB_HEX_PLAY_TASK = "Isaac-Crab-Hex-Flat-Walk-Play-v0"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parkour_tasks_src() -> Path:
    return repo_root() / "parkour" / "parkour_tasks"


def ensure_parkour_tasks_on_path() -> None:
    src = str(parkour_tasks_src())
    if src not in sys.path:
        sys.path.insert(0, src)


def default_crab_simple_usd_path() -> Path:
    return repo_root() / "assets" / "crab_simple.usda"


def configure_crab_simple_usd_env() -> None:
    """Use ``assets/crab_simple.usda`` when unset (joint names in ``SimulatedRobotConfig``)."""
    usd = default_crab_simple_usd_path()
    if usd.is_file():
        os.environ.setdefault("KRABBY_HEX_USD_PATH", str(usd))


def _import_parkour_env_deps() -> None:
    ensure_parkour_tasks_on_path()
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401


def build_crab_hex_env_cfg(
    task: str = DEFAULT_CRAB_HEX_PLAY_TASK,
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
    use_fabric: bool = True,
    with_gravity: bool = False,
    fix_base: bool = False,
) -> Any:
    """Parse parkour crab env cfg with ``num_envs=1`` defaults suited to firmware smoke tests."""
    _import_parkour_env_deps()
    configure_crab_simple_usd_env()
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=num_envs,
        use_fabric=use_fabric,
    )
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "parkours") and env_cfg.parkours is not None:
        env_cfg.parkours.base_parkour.debug_vis = False
    if hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        env_cfg.commands.base_velocity.debug_vis = False

    if not with_gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        if hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
            env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    if fix_base and hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    return env_cfg


def make_crab_hex_env(
    task: str = DEFAULT_CRAB_HEX_PLAY_TASK,
    **cfg_kwargs: Any,
) -> Any:
    """``gym.make`` wrapper after ``AppLauncher``."""
    import gymnasium as gym

    env_cfg = build_crab_hex_env_cfg(task, **cfg_kwargs)
    return gym.make(task, cfg=env_cfg)


def get_robot_articulation(env: Any) -> Articulation:
    return env.unwrapped.scene["robot"]


def _scalar_joint_index(joint_id: Any) -> int:
    if hasattr(joint_id, "item"):
        return int(joint_id.item())
    return int(joint_id)


def resolve_joint_indices_by_name(
    robot: Articulation,
    joint_names: Sequence[str],
) -> dict[str, int]:
    """Map exact USD joint names to articulation DOF indices via ``find_joints``."""
    names = list(joint_names)
    joint_ids, resolved_names = robot.find_joints(names, preserve_order=True)
    if len(joint_ids) != len(names):
        raise KeyError(
            f"find_joints expected {len(names)} joints, got {len(joint_ids)} "
            f"for {names!r} (resolved {list(resolved_names)!r})"
        )
    return {
        str(name): _scalar_joint_index(jid)
        for name, jid in zip(resolved_names, joint_ids, strict=True)
    }


def resolve_front_board_joint_indices(
    robot: Articulation,
    config: SimulatedRobotConfig | None = None,
) -> dict[str, int]:
    """Resolve the six FRONT-board joints from ``SimulatedRobotConfig``."""
    cfg = config or SimulatedRobotConfig.front_board_rev3()
    return resolve_joint_indices_by_name(robot, cfg.isaac_joint_names())


def step_env_once(env: Any) -> tuple[Any, ...]:
    """Single zero-action physics step (headless smoke)."""
    import torch

    device = env.unwrapped.device
    actions = torch.zeros(env.action_space.shape, device=device)
    with torch.inference_mode():
        return env.step(actions)
