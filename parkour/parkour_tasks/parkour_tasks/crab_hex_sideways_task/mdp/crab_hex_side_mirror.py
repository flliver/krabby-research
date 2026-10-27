# SPDX-License-Identifier: BSD-3-Clause
"""Fore-aft mirror maps for the sideways-walk task (symmetry-regularized PPO).

The sideways task travels along body **+y**, so the symmetry plane that preserves travel is the
**x = 0** plane: front legs swap with rear legs, left stays left. Under that reflection each
tripod set maps onto ITSELF ({FL, MR, RL} -> {RL, MR, FL}), so -- unlike the forward task's L/R
mirror, which swaps the two tripod sets -- this loss does not enforce A/B duty balance; duty
balance is watched in the evals instead. What it does remove is a permanent lead-ROW preference
(pushing with the rear row vs pulling with the front row), which is the sideways analogue of the
handedness the seed-basin search showed unconstrained training breaks randomly.

Everything is derived from the forward module (``crab_hex_forward_task/mdp/crab_hex_mirror.py``,
loaded BY FILE PATH as ``crab_hex_mirror__forward``; it imports only torch): the maps are built
at runtime from RESOLVED name lists, never hand-typed indices, and the only difference is the
mirror *plane* handed to ``CrabHexMirror``.

Derivation -- reflection through the plane x = 0, i.e. ``M = diag(-1, 1, 1)`` in body frame:

* **Pseudo-vectors** (angular velocity, roll/pitch attitude) keep the component ALONG the plane
  normal and flip the in-plane ones: ``wx`` and ``roll`` +1, ``wy`` / ``wz`` / ``pitch`` -1.
  (Quaternion check: reflecting a rotation gives ``q' = (w, x, -y, -z)``, whose
  ``atan2``/``asin`` roll/pitch are exactly ``(roll, -pitch)``; ``w_b' = -M w_b``.)
* **True vectors** flip x: body ``lin_vx`` -1 / ``lin_vy`` +1 (``v_b' = M v_b``), and likewise the
  priv_explicit 3-vectors and the CoM offset -> ``VEC3_SIGNS = (-1, +1, +1)``.
* **Joints.** Cam shaft and the passive hip yaw are yaw-type (rotation about body +z, in-plane)
  -> -1. The hip and knee PITCH axes lie along body x for legs on the +-y faces, i.e. ALONG the
  plane normal, and front and rear legs on the same side carry no relative USD flip -> both +1.
  That knee +1 is the one sign that differs from the L/R mirror's -1, where the 180-deg Z frame
  flip on right legs makes the knee odd.
* **Steering.** ``delta_yaw`` / ``delta_next_yaw`` are the heading error measured about body +z
  (in-plane pseudo-scalar) -> -1.
* **Command.** ``cmd`` is the travel speed along +y, which the reflection preserves -> +1.
* **Clock.** The reflection maps each tripod set to itself, so the mirrored gait is the SAME
  schedule at the SAME phase (not advanced by pi as in the L/R mirror) -> ``sin``/``cos`` +1.
* **Height scan.** The scanner look-ahead offset lies in the plane (its x is 0), so flipping the
  ray index ``ix`` is a symmetry of the grid -> ``scan_axis="x"``.

The FK reflection was verified numerically (2026-09-11) against ``crab_hex_foot_fk.leg_points``:
``leg_points(dst, -yaw, hip, knee) == diag(-1, 1, 1) @ leg_points(src, yaw, hip, knee)`` for the
pairs FL<->RL, FR<->RR, ML, MR at random poses (worst error 0.0); it is pinned by
``tests/unit/test_crab_hex_side_mirror.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_FWD_MODULE_NAME = "crab_hex_mirror__forward"
_FWD_MIRROR_PY = (
    Path(__file__).resolve().parents[2] / "crab_hex_forward_task" / "mdp" / "crab_hex_mirror.py"
)


def _load_by_path(path: Path, name: str):
    """Load a module by file path under ``name`` (never through ``parkour_tasks.*``)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for a real file
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_fwd = sys.modules.get(_FWD_MODULE_NAME)
if _fwd is None:
    _fwd = _load_by_path(_FWD_MIRROR_PY, _FWD_MODULE_NAME)

# ---------------------------------------------------------------------------- mirror plane x = 0
#: Front <-> rear, side letter preserved; the mid legs are fixed points.
LEG_SWAP = {"FL": "RL", "RL": "FL", "FR": "RR", "RR": "FR", "ML": "ML", "MR": "MR"}

#: joint-name substring -> mirror sign. Knee is +1 here (vs -1 for the L/R mirror): the pitch
#: axes lie along the plane normal and there is no relative USD flip between the F and R rows.
JOINT_SIGN_RULES = (
    ("_Body_CamShaft_", -1.0),
    ("_Body_Hip_", -1.0),
    ("_Hip_Femur_", 1.0),
    ("_Femur_Tibia_", 1.0),
)

# Per-step proprio head layout of CrabHexParkourObservations (same 17-dim term as the forward
# task -- only the signs change):
# [wx, wy, wz, roll, pitch, 0*dy, delta_yaw, delta_next_yaw, 0*cmd_vx, 0*cmd_vy, cmd_vx,
#  env_idx, invert_env_idx, lin_vx, lin_vy, clock_sin, clock_cos]
HEAD_SIGNS = [1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
              1.0, 1.0]

#: priv_explicit 3-vectors and the CoM offset under x -> -x.
VEC3_SIGNS = (-1.0, 1.0, 1.0)


class CrabHexSideMirror(_fwd.CrabHexMirror):
    """The forward mirror builder with the fore-aft (x = 0) plane's constants.

    Same positional signature as ``CrabHexMirror`` so ``_resolve_layout`` feeds either one.
    """

    def __init__(
        self,
        joint_names: list[str],
        action_joint_names: list[str],
        contact_body_names: list[str],
        scan_nx: int,
        scan_ny: int,
        scan_ordering: str,
        history_length: int,
        device: torch.device | str,
    ):
        super().__init__(
            joint_names,
            action_joint_names,
            contact_body_names,
            scan_nx,
            scan_ny,
            scan_ordering,
            history_length,
            device,
            head_signs=list(HEAD_SIGNS),
            joint_sign_rules=JOINT_SIGN_RULES,
            leg_swap=LEG_SWAP,
            scan_axis="x",
            vec3_signs=VEC3_SIGNS,
        )


#: Own cache (the forward module keeps its own): one mirror per env instance.
_CACHE: dict[int, CrabHexSideMirror] = {}


def _get_mirror(env, obs_dim: int) -> CrabHexSideMirror:
    """Build (once per env) from resolved names; history length inferred from ``obs_dim``."""
    key = id(env)
    if key not in _CACHE:
        _CACHE[key] = CrabHexSideMirror(
            **_fwd._resolve_layout(env, obs_dim, n_head=len(HEAD_SIGNS))
        )
    return _CACHE[key]


def crab_hex_side_symmetry_augmentation(obs=None, actions=None, env=None, obs_type="policy"):
    """rsl-rl symmetry entry point: returns (cat([obs, mirror(obs)]), cat([act, mirror(act)]))."""
    return _fwd._augment(_get_mirror, _CACHE, obs, actions, env, obs_type)
