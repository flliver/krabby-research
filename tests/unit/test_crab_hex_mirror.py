"""Unit tests for the crab-hex L/R mirror maps (mirror-symmetry training campaign,
parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/).

Pure torch. Pins:
1. All maps are involutions: mirror(mirror(x)) == x exactly (perm is pairwise, partner joints
   share a sign rule so sign^2 == 1).
2. Permutations are bijections.
3. Sign conventions: camshaft/hip-yaw/knee flip, hip-femur doesn't; lateral/yaw/roll proprio
   channels flip, forward/pitch don't.
4. The scan permutation is a pure lateral flip for both grid orderings.
5. A hand-built synthetic case: A-set-planted contact pattern mirrors to the B-set pattern.
"""

import sys
from pathlib import Path

import pytest
import torch

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from crab_hex_mirror import (  # noqa: E402
    _HEAD_SIGNS,
    CrabHexMirror,
    build_name_permutation,
    build_scan_permutation,
)

# Representative articulation joint order (per-leg blocks; real order is resolved at runtime —
# the map builder must work for ANY order, which these tests exercise).
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
JOINT_TYPES = [
    "_Body_CamShaft_RevoluteJoint",
    "_Body_Hip_RevoluteJoint",
    "_Hip_Femur_RevoluteJoint",
    "_Femur_Tibia_RevoluteJoint",
]
JOINT_NAMES = [leg + jt for jt in JOINT_TYPES for leg in LEGS]  # type-major, 24
ACTION_NAMES = [
    leg + jt
    for jt in ("_Body_CamShaft_RevoluteJoint", "_Hip_Femur_RevoluteJoint", "_Femur_Tibia_RevoluteJoint")
    for leg in LEGS
]  # 18
CONTACT_NAMES = [leg + "_Footpad" for leg in LEGS]
HISTORY = 10
NX, NY = 12, 11


@pytest.fixture()
def mirror() -> CrabHexMirror:
    return CrabHexMirror(
        joint_names=JOINT_NAMES,
        action_joint_names=ACTION_NAMES,
        contact_body_names=CONTACT_NAMES,
        scan_nx=NX,
        scan_ny=NY,
        scan_ordering="xy",
        history_length=HISTORY,
        device="cpu",
    )


def test_dims(mirror):
    # head grew 15 -> 17 with the gait-clock sin/cos (gait-formation-v2 Phase 1)
    assert mirror.n_prop == 17 + 24 + 24 + 18 + 6 == 89
    assert mirror.obs_dim == 89 + NX * NY + 9 + (5 + 48) + HISTORY * 89 == 1173


def test_obs_involution(mirror):
    x = torch.randn(7, mirror.obs_dim)
    assert torch.equal(mirror.mirror_obs(mirror.mirror_obs(x)), x)


def test_action_involution(mirror):
    a = torch.randn(5, 18)
    assert torch.equal(mirror.mirror_actions(mirror.mirror_actions(a)), a)


def test_obs_perm_is_bijection(mirror):
    assert sorted(mirror.obs_perm.tolist()) == list(range(mirror.obs_dim))
    assert sorted(mirror.act_perm.tolist()) == list(range(18))


def test_mirror_is_not_identity(mirror):
    x = torch.randn(3, mirror.obs_dim)
    assert not torch.equal(mirror.mirror_obs(x), x)


def test_joint_sign_rules():
    perm = build_name_permutation(JOINT_NAMES)
    for i, name in enumerate(JOINT_NAMES):
        partner = JOINT_NAMES[perm[i]]
        assert partner[:2] != name[:2] and partner[2:] == name[2:]


def test_action_sign_conventions(mirror):
    # action layout (type-major): camshaft 0-5 flip, hip_femur 6-11 no flip, knee 12-17 flip
    a = torch.zeros(1, 18)
    a[0, 0] = 1.0  # FL camshaft
    m = mirror.mirror_actions(a)
    fr_idx = ACTION_NAMES.index("FR_Body_CamShaft_RevoluteJoint")
    assert m[0, fr_idx] == -1.0
    a = torch.zeros(1, 18)
    a[0, ACTION_NAMES.index("ML_Hip_Femur_RevoluteJoint")] = 1.0
    m = mirror.mirror_actions(a)
    assert m[0, ACTION_NAMES.index("MR_Hip_Femur_RevoluteJoint")] == 1.0
    a = torch.zeros(1, 18)
    a[0, ACTION_NAMES.index("RL_Femur_Tibia_RevoluteJoint")] = 1.0
    m = mirror.mirror_actions(a)
    assert m[0, ACTION_NAMES.index("RR_Femur_Tibia_RevoluteJoint")] == -1.0


def test_head_signs_flip_lateral_channels(mirror):
    x = torch.zeros(1, mirror.obs_dim)
    for idx, expected in [(0, -1.0), (1, 1.0), (2, -1.0), (3, -1.0), (4, 1.0),
                          (6, -1.0), (10, 1.0), (13, 1.0), (14, -1.0)]:
        x.zero_()
        x[0, idx] = 1.0
        assert mirror.mirror_obs(x)[0, idx] == expected, f"head dim {idx}"


@pytest.mark.parametrize("ordering", ["xy", "yx"])
def test_scan_permutation_is_lateral_flip(ordering):
    perm = build_scan_permutation(NX, NY, ordering)
    assert sorted(perm) == list(range(NX * NY))
    # applying twice = identity
    assert [perm[perm[i]] for i in range(NX * NY)] == list(range(NX * NY))
    # a ray at (ix, iy) maps to (ix, ny-1-iy)
    if ordering == "xy":
        src = 3 * NX + 5          # iy=3, ix=5
        assert perm[src] == (NY - 1 - 3) * NX + 5
    else:
        src = 5 * NY + 3          # ix=5, iy=3
        assert perm[src] == 5 * NY + (NY - 1 - 3)


def test_contact_pattern_maps_A_set_to_B_set(mirror):
    # contact fill block sits at [17+48+18 : 89] = last 6 of the step; order FL,FR,ML,MR,RL,RR
    x = torch.zeros(1, mirror.obs_dim)
    base = 17 + 48 + 18
    for i, planted in enumerate([1, 0, 0, 1, 1, 0]):  # A set planted (FL,MR,RL)
        x[0, base + i] = float(planted)
    m = mirror.mirror_obs(x)
    assert m[0, base:base + 6].tolist() == [0, 1, 1, 0, 0, 1]  # B set planted


def test_history_slots_use_step_map(mirror):
    # a value in history slot k, head dim 14 (lin_vy) must land in the same slot, flipped
    hist_base = 89 + NX * NY + 9 + 53
    assert hist_base + HISTORY * 89 == mirror.obs_dim  # layout sanity
    x = torch.zeros(1, mirror.obs_dim)
    slot = 4
    x[0, hist_base + slot * 89 + 14] = 2.0
    m = mirror.mirror_obs(x)
    assert m[0, hist_base + slot * 89 + 14] == -2.0


def test_priv_latent_ratios_permute_without_flip(mirror):
    # stiffness-ratio block: after scan(132)+priv_e(9)+mass/com/friction(5)
    base = 89 + NX * NY + 9 + 5
    x = torch.zeros(1, mirror.obs_dim)
    fl_cam = JOINT_NAMES.index("FL_Body_CamShaft_RevoluteJoint")
    x[0, base + fl_cam] = 0.7
    m = mirror.mirror_obs(x)
    fr_cam = JOINT_NAMES.index("FR_Body_CamShaft_RevoluteJoint")
    assert m[0, base + fr_cam] == pytest.approx(0.7)  # magnitude ratio: no sign flip
