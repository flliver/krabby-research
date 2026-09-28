"""Pin the measured-hardware joint limits in ``assets/crab.usda`` (the main asset; the limits are
identical in every generated variant).

Pure-text parse of the USD -- no Isaac Sim needed. Guards the 2026-08-20 hardware
measurement set (physical robot: yaw +-25 deg cam throw, hip 45-150 deg measured from
vertical-UP, knee 5-140 deg interior angle) against a silent revert by a future asset
regeneration. The right-leg (FR/MR/RR) knee limits are mirrored because those joints carry
a 180-degree frame flip (``localRot0 = (0, 0, 1, 0)``).

Conventions (see crab_hex_dimensions.py, the single source of truth):
  hip:   sim = 90 - (180 - from_up), positive = femur down -> [-45, +60]
  knee L: sim = 90 - interior                              -> [-50, +85]
  knee R: sign-flipped frame                               -> [-85, +50]
  Body_Hip: passive, cam-slaved; hard +-28 so soft (0.9x) = +-25.2 > 25.0 cam throw

The expected values are written out literally AND recomputed from crab_hex_dimensions --
a change to either the USD, the dimensions module, or the convention arithmetic fails here.
"""

import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
USDA_PATH = REPO_ROOT / "assets" / "crab.usda"  # main asset; joint limits are plant-invariant
_DIMS_PATH = (
    REPO_ROOT
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "mdp"
    / "crab_hex_dimensions.py"
)
_spec = importlib.util.spec_from_file_location("crab_hex_dimensions", _DIMS_PATH)
dims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dims)

LEFT_LEGS = ("FL", "ML", "RL")
RIGHT_LEGS = ("FR", "MR", "RR")
ALL_LEGS = LEFT_LEGS + RIGHT_LEGS

# joint-name template -> {leg-prefix: (lower_deg, upper_deg)} -- hardware table, literal.
EXPECTED_LIMITS = {
    "{leg}_Body_Hip_RevoluteJoint": {leg: (-28.0, 28.0) for leg in ALL_LEGS},
    "{leg}_Hip_Femur_RevoluteJoint": {leg: (-45.0, 60.0) for leg in ALL_LEGS},
    "{leg}_Femur_Tibia_RevoluteJoint": {
        **{leg: (-50.0, 85.0) for leg in LEFT_LEGS},
        **{leg: (-85.0, 50.0) for leg in RIGHT_LEGS},
    },
}

SOFT_LIMIT_FACTOR = 0.9  # crab_hex_scene_cfg.py soft_joint_pos_limit_factor
THETA_HIP_MAX = math.asin(dims.YAW_K)  # 25.0 deg exactly: K derived from measured throw


def test_literal_table_matches_dimensions_module():
    """The literal table above and the conversion arithmetic in crab_hex_dimensions agree."""
    assert dims.HIP_SIM_LIMITS_DEG == pytest.approx((-45.0, 60.0))
    assert dims.KNEE_SIM_LIMITS_LEFT_DEG == pytest.approx((-50.0, 85.0))
    assert dims.KNEE_SIM_LIMITS_RIGHT_DEG == pytest.approx((-85.0, 50.0))
    assert dims.YAW_HARD_LIMIT_DEG == pytest.approx(28.0)
    # And that the arithmetic still encodes the raw hardware measurements:
    assert dims.HIP_ROM_FROM_UP_DEG == (45.0, 150.0)
    assert dims.KNEE_INTERIOR_ROM_DEG == (5.0, 140.0)
    assert dims.YAW_THROW_DEG == 25.0


def _parse_joint_blocks(text: str) -> dict[str, dict[str, float]]:
    """Return {joint_name: {attr: value}} for every PhysicsRevoluteJoint block."""
    blocks: dict[str, dict[str, float]] = {}
    pattern = re.compile(
        r'def PhysicsRevoluteJoint "(?P<name>\w+)".*?\{(?P<body>.*?)\n\s*\}',
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        attrs: dict[str, float] = {}
        for lim in ("lowerLimit", "upperLimit"):
            lm = re.search(rf"physics:{lim} = (-?[\d.]+)", m.group("body"))
            if lm:
                attrs[lim] = float(lm.group(1))
        blocks[m.group("name")] = attrs
    return blocks


@pytest.fixture(scope="module")
def joint_blocks() -> dict[str, dict[str, float]]:
    return _parse_joint_blocks(USDA_PATH.read_text())


@pytest.mark.parametrize("template", sorted(EXPECTED_LIMITS))
def test_limits_match_hardware_table(joint_blocks, template):
    for leg, (lo, hi) in EXPECTED_LIMITS[template].items():
        name = template.format(leg=leg)
        assert name in joint_blocks, f"{name} missing from {USDA_PATH.name}"
        attrs = joint_blocks[name]
        assert attrs.get("lowerLimit") == pytest.approx(lo), f"{name} lowerLimit"
        assert attrs.get("upperLimit") == pytest.approx(hi), f"{name} upperLimit"


def test_cam_shaft_joints_are_continuous(joint_blocks):
    for leg in ALL_LEGS:
        name = f"{leg}_Body_CamShaft_RevoluteJoint"
        assert name in joint_blocks, f"{name} missing"
        attrs = joint_blocks[name]
        assert "lowerLimit" not in attrs and "upperLimit" not in attrs, (
            f"{name} must stay limit-free: the cam shaft spins continuously"
        )


def test_right_leg_knee_limits_mirror_left(joint_blocks):
    for left, right in zip(LEFT_LEGS, RIGHT_LEGS):
        l_attrs = joint_blocks[f"{left}_Femur_Tibia_RevoluteJoint"]
        r_attrs = joint_blocks[f"{right}_Femur_Tibia_RevoluteJoint"]
        assert r_attrs["lowerLimit"] == -l_attrs["upperLimit"]
        assert r_attrs["upperLimit"] == -l_attrs["lowerLimit"]


@pytest.mark.parametrize("template", sorted(EXPECTED_LIMITS))
def test_defaults_inside_soft_limits(joint_blocks, template):
    """soft_joint_pos_limit_factor scales the range about its midpoint. Defaults are the
    actuator mid-stroke poses from crab_hex_linkage (needs torch, like the cam check)."""
    mdp_dir = str(_DIMS_PATH.parent)
    if mdp_dir not in sys.path:
        sys.path.insert(0, mdp_dir)
    pytest.importorskip("torch")
    import crab_hex_linkage as lk

    defaults_rad = {
        "{leg}_Body_Hip_RevoluteJoint": {leg: 0.0 for leg in ALL_LEGS},
        "{leg}_Hip_Femur_RevoluteJoint": {leg: lk.hip_default_rad() for leg in ALL_LEGS},
        "{leg}_Femur_Tibia_RevoluteJoint": {
            **{leg: lk.knee_default_left_rad() for leg in LEFT_LEGS},
            **{leg: lk.knee_default_right_rad() for leg in RIGHT_LEGS},
        },
    }
    for leg, (lo, hi) in EXPECTED_LIMITS[template].items():
        lo_r, hi_r = math.radians(lo), math.radians(hi)
        mid, half = (lo_r + hi_r) / 2, (hi_r - lo_r) / 2
        soft_lo = mid - SOFT_LIMIT_FACTOR * half
        soft_hi = mid + SOFT_LIMIT_FACTOR * half
        default = defaults_rad[template][leg]
        assert soft_lo < default < soft_hi, (
            f"{template.format(leg=leg)} default {default} outside soft "
            f"[{soft_lo:.3f}, {soft_hi:.3f}]"
        )


def test_body_hip_soft_limit_clears_cam_sweep():
    """The passive Body_Hip soft limit must not clip the cam-slaved sweep."""
    soft = SOFT_LIMIT_FACTOR * math.radians(dims.YAW_HARD_LIMIT_DEG)
    assert THETA_HIP_MAX < soft, (
        f"cam sweep {THETA_HIP_MAX:.4f} rad would be clipped by soft limit {soft:.4f}"
    )


def test_cam_mapping_module_uses_measured_throw():
    """crab_hex_cam_mapping must derive K from the measured throw, not the stale SVG value."""
    mdp_dir = str(_DIMS_PATH.parent)
    if mdp_dir not in sys.path:
        sys.path.insert(0, mdp_dir)
    torch = pytest.importorskip("torch")  # noqa: F841 -- cam mapping imports torch
    import crab_hex_cam_mapping as cam

    assert cam.THETA_HIP_MAX == pytest.approx(math.radians(dims.YAW_THROW_DEG))
