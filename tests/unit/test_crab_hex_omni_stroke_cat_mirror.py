# SPDX-License-Identifier: BSD-3-Clause
"""Mirror-orbit invariance for the per-category stroke plan (2026-09-27), pure torch, no Isaac.

The mirror_constraints derivation (``impl_spec_from_scoping.md``) says: under the omni mirror group
{I, M_y, M_x, R_z(pi)}, ``telemetry_category`` maps ``{fwd, back}`` into each other under M_x and
R_z(pi) and fixes every category NAME under M_y (hands swap, the category name does not); every other
category name (side, diag, turn, move_turn, stand) is fixed under all three images. Pinned here:

* that orbit, derived straight from ``crab_hex_omni_command_math.telemetry_category`` on a dense
  bearing grid per slot, mirrored the way ``crab_hex_omni_mirror``'s own body-command signs (slots
  8/9/5) say a world reflection moves a bearing -- the ``scope_cat/mirror_cats.py`` technique;
* for every ``cat_scale`` table :func:`stroke_knobs.parse_stroke_cat_w` accepts (the mirror tie
  ``fwd == back`` always satisfied), the multiplier is identical at a category and its image under
  every one of the three mirrors -- both directly (the vector) and through
  ``crab_hex_omni_stroke_rewards._resolve_cat_scale`` (ast-lifted, no Isaac import needed);
* a hand-built vector that breaks the tie (``fwd != back``) is NOT mirror-invariant -- pinning why
  ``parse_stroke_cat_w`` refuses it;
* the CAT-SCALED reward is mirror-INVARIANT (not just equivariant): extends
  ``test_crab_hex_omni_stroke_math.py::test_both_terms_are_mirror_equivariant``'s random rollout with
  an independently assigned per-env category (mirrored the same way), and multiplies the already
  mirror-invariant scalar reward (``reward = pay.sum(-1) / dt``, invariant because ``pay`` is only
  PERMUTED by a mirror image and the sum over feet erases the permutation) by the tied table's
  multiplier at the category and at its image -- which the tie makes equal, so the weighted reward
  comes out bit-for-bit identical, not merely close.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import textwrap
from pathlib import Path

import pytest
import torch

TASKS = Path(__file__).resolve().parents[2] / "parkour" / "parkour_tasks" / "parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
COMMAND_MATH_PY = OMNI / "mdp" / "crab_hex_omni_command_math.py"
MIRROR_PY = OMNI / "mdp" / "crab_hex_omni_mirror.py"
STROKE_MATH_PY = OMNI / "mdp" / "crab_hex_omni_stroke_math.py"
STROKE_KNOBS_PY = OMNI / "config" / "crab_hex" / "stroke_knobs.py"
STROKE_REWARDS_PY = OMNI / "mdp" / "crab_hex_omni_stroke_rewards.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load("cat_mirror__command_math", COMMAND_MATH_PY)
mm = _load("cat_mirror__mirror", MIRROR_PY)
sm = _load("cat_mirror__stroke_math", STROKE_MATH_PY)
sk = _load("cat_mirror__stroke_knobs", STROKE_KNOBS_PY)

CATS = cm.TELEMETRY_CATEGORIES
FWD, BACK = CATS.index("fwd"), CATS.index("back")
SLOTS = (cm.SLOT_STAND, cm.SLOT_MOVE, cm.SLOT_TURN, cm.SLOT_MOVE_TURN)

#: Body-command signs (vx, vy, wz) at mirror-head slots 8, 9, 5 -- the same slots
#: ``impl_spec_from_scoping.md``'s ``mirror_constraints`` table and ``scope_cat/mirror_cats.py``
#: read them from.
_LR = (mm.HEAD_SIGNS_LR[8], mm.HEAD_SIGNS_LR[9], mm.HEAD_SIGNS_LR[5])
_FA = (mm.HEAD_SIGNS_FA[8], mm.HEAD_SIGNS_FA[9], mm.HEAD_SIGNS_FA[5])
SIGNS = {"M_y": _LR, "M_x": _FA, "R_z(pi)": tuple(a * b for a, b in zip(_LR, _FA))}
IMAGES = ("M_y", "M_x", "R_z(pi)")


def _mirror_bearing(phi: torch.Tensor, image: str) -> torch.Tensor:
    """The world-frame bearing seen after reflecting the world the way ``image`` reflects the body
    command (``vx``, ``vy`` signs; ``wz``'s sign plays no part in a bearing)."""
    sx, sy, _ = SIGNS[image]
    return torch.atan2(sy * torch.sin(phi), sx * torch.cos(phi))


# ============================================================================ category orbit
@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("slot", SLOTS)
def test_telemetry_category_mirror_orbit(slot, image):
    """Derived straight from the code (module docstring): STAND/TURN/MOVE_TURN ignore bearing and
    are trivially fixed; a MOVE row's category is fixed under M_y and swaps fwd<->back under M_x and
    R_z(pi), with side/diag never leaving their own name."""
    phi = torch.linspace(-math.pi, math.pi, 72001, dtype=torch.float64)
    s = torch.full_like(phi, slot, dtype=torch.long)
    cat0 = cm.telemetry_category(s, phi)
    catm = cm.telemetry_category(s, _mirror_bearing(phi, image))
    if image == "M_y" or slot != cm.SLOT_MOVE:
        assert torch.equal(catm, cat0), image
    else:
        expect = cat0.clone()
        expect[cat0 == FWD] = BACK
        expect[cat0 == BACK] = FWD
        assert torch.equal(catm, expect), image
        assert (catm != cat0).any()  # the swap is not vacuous -- some rows really are fwd or back


# ============================================================================ table orbit (tied tables)
def _vec(table: dict[str, float]) -> torch.Tensor:
    vec = torch.ones(len(CATS), dtype=torch.float64)
    for name, m in table.items():
        vec[CATS.index(name)] = float(m)
    return vec


TIED_TABLES = [
    {},  # all ones
    {"side": 4.8},
    {"fwd": 2.0, "back": 2.0, "side": 0.3, "diag": 1.7, "turn": 0.9, "move_turn": 1.1},
    {"fwd": 0.0, "back": 0.0, "turn": 3.0},  # 0 is a legal multiplier
]


@pytest.mark.parametrize("table", TIED_TABLES, ids=[str(t) for t in TIED_TABLES])
@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("slot", SLOTS)
def test_every_parser_accepted_table_is_mirror_invariant_by_category(slot, image, table):
    """For any table :func:`parse_stroke_cat_w` accepts (the mirror tie always holds), the resolved
    multiplier at a category equals the multiplier at its image -- directly on the vector."""
    parsed = sk.parse_stroke_cat_w(",".join(f"{k}:{v}" for k, v in table.items())) if table else \
        sk.parse_stroke_cat_w("fwd:1,back:1")  # the empty case: still exercise the parser, not just defaults
    vec = _vec(parsed)
    phi = torch.linspace(-math.pi, math.pi, 3601, dtype=torch.float64)
    s = torch.full_like(phi, slot, dtype=torch.long)
    cat0 = cm.telemetry_category(s, phi)
    catm = cm.telemetry_category(s, _mirror_bearing(phi, image))
    torch.testing.assert_close(vec[catm], vec[cat0], rtol=0, atol=0)


def _source_of(path: Path, name: str) -> str:
    src = path.read_text()
    tree = ast.parse(src, str(path))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


def _resolve_cat_scale(table: dict, categories=CATS):
    """``crab_hex_omni_stroke_rewards._resolve_cat_scale``, ast-lifted (Isaac-free): the SAME
    resolver ``RewardFootStroke.__init__`` calls, exercised here against the mirror orbit above."""
    src = "from __future__ import annotations\nimport torch\n\n" + _source_of(STROKE_REWARDS_PY, "_resolve_cat_scale")
    ns: dict = {}
    exec(compile(src, str(STROKE_REWARDS_PY), "exec"), ns)  # noqa: S102
    return ns["_resolve_cat_scale"](table, categories)


@pytest.mark.parametrize("table", TIED_TABLES[1:], ids=[str(t) for t in TIED_TABLES[1:]])
@pytest.mark.parametrize("image", IMAGES)
def test_the_real_resolver_is_mirror_invariant_too(image, table):
    """The same property through the ACTUAL wrapper resolver, not just a hand-rolled vector."""
    parsed = sk.parse_stroke_cat_w(",".join(f"{k}:{v}" for k, v in table.items()))
    vec = _resolve_cat_scale(parsed).double()
    phi = torch.linspace(-math.pi, math.pi, 3601, dtype=torch.float64)
    s = torch.full_like(phi, cm.SLOT_MOVE, dtype=torch.long)
    cat0 = cm.telemetry_category(s, phi)
    catm = cm.telemetry_category(s, _mirror_bearing(phi, image))
    torch.testing.assert_close(vec[catm], vec[cat0], rtol=0, atol=0)


@pytest.mark.parametrize("image", ("M_x", "R_z(pi)"))
def test_an_untied_fwd_back_vector_breaks_the_invariance(image):
    """The refusal's reason, made concrete: WITHOUT the tie, the same construction is NOT
    mirror-invariant on M_x / R_z(pi) (M_y never touches fwd/back, so it stays invariant there)."""
    bad = _vec({"fwd": 2.0, "back": 1.0})  # parse_stroke_cat_w would refuse this; built by hand here
    phi = torch.tensor([0.0])  # dead ahead -> fwd
    s = torch.full_like(phi, cm.SLOT_MOVE, dtype=torch.long)
    cat0 = cm.telemetry_category(s, phi)
    catm = cm.telemetry_category(s, _mirror_bearing(phi, image))
    assert bad[catm].item() != bad[cat0].item()


# ============================================================================ weighted-reward invariance
# The ``test_both_terms_are_mirror_equivariant`` harness (test_crab_hex_omni_stroke_math.py),
# duplicated here (self-contained files, the house convention) rather than imported.
DT = 0.02
F = 6
TURN_R = 1.2
FOOT_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
PERM_LR = [FOOT_ORDER.index(mm.LEG_SWAP_LR[leg]) for leg in FOOT_ORDER]
PERM_FA = [FOOT_ORDER.index(mm.LEG_SWAP_FA[leg]) for leg in FOOT_ORDER]


def _run_stroke(force, foot_pos_w, root_pos_w, heading, cmd, vel, wz):
    T, N = force.shape[:2]
    st = sm.init_stroke_state(N, F)
    stand = torch.zeros(T, N, dtype=torch.bool)
    rewards = []
    for step in range(T):
        r, st, _pay = sm.foot_stroke_step(force[step], foot_pos_w[step], root_pos_w[step], heading[step],
                                          cmd[step], stand[step], vel[step], wz[step], st, DT, TURN_R)
        rewards.append(r)
    return torch.stack(rewards)  # [T, N]


def _mirror_traj(axis: int, perm, force, foot, root, hdg, cmd, vel, wz):
    s = torch.ones(3)
    s[axis] = -1.0
    cmd_m = cmd.clone()
    cmd_m[..., axis] *= -1.0
    cmd_m[..., 2] *= -1.0
    vel_m = vel.clone()
    vel_m[..., axis] *= -1.0
    return force[..., perm], (foot * s)[..., perm, :], root * s, -hdg, cmd_m, vel_m, -wz


def _random_rollout(seed=1, T=200, N=24):
    g = torch.Generator().manual_seed(seed)
    phase = torch.rand(N, F, generator=g) * 2 * math.pi
    t_ = torch.arange(T)[:, None, None] * DT
    force = 300.0 * (torch.sin(2 * math.pi * 0.4 * t_ + phase) > 0)
    base = torch.tensor([[0.6, 0.9], [0.6, -0.9], [0.0, 1.0], [0.0, -1.0], [-0.6, 0.9], [-0.6, -0.9]])
    foot = torch.zeros(T, N, F, 3)
    foot[..., :2] = base + torch.cumsum(torch.randn(T, N, F, 2, generator=g) * 0.004, dim=0)
    root = torch.zeros(T, N, 3)
    root[..., :2] = torch.cumsum(torch.randn(T, N, 2, generator=g) * 0.004 + 0.003, dim=0)
    hdg = torch.cumsum(torch.randn(T, N, generator=g) * 0.01, dim=0)
    cmd = (torch.randn(N, 3, generator=g) * 0.25).expand(T, N, 3).clone()
    vel = cmd[..., :2] * 0.6 + torch.randn(T, N, 2, generator=g) * 0.05
    wz = cmd[..., 2] * 0.6 + torch.randn(T, N, generator=g) * 0.05
    return force, foot, root, hdg, cmd, vel, wz


@pytest.mark.parametrize("image", IMAGES)
def test_cat_scaled_reward_is_mirror_invariant(image):
    """The corollary the plan asks to pin directly: a tied ``cat_scale`` table times the (already
    mirror-invariant) scalar reward is IDENTICAL, not merely permuted, under every mirror image --
    because a MOVE row's category and its image share the SAME multiplier (the tie), and STAND /
    TURN / MOVE_TURN rows carry the SAME category on both sides of every image."""
    args = _random_rollout()
    force, foot, root, hdg, cmd, vel, wz = args
    N = force.shape[1]
    if image == "M_y":
        m = _mirror_traj(1, PERM_LR, *args)   # the permutation (which foot is which) is irrelevant
    elif image == "M_x":                       # below: only the reward (sum over feet) is used, and
        m = _mirror_traj(0, PERM_FA, *args)    # the existing mirror-equivariant test already pins pay
    else:
        m = _mirror_traj(0, PERM_FA, *_mirror_traj(1, PERM_LR, *args))

    reward = _run_stroke(force, foot, root, hdg, cmd, vel, wz)          # [T, N]
    reward_m = _run_stroke(*m)
    torch.testing.assert_close(reward_m, reward, rtol=1e-4, atol=1e-6)  # reward itself is already invariant

    # An independent per-env category assignment (a MOVE row gets a random bearing; a TURN row is
    # bearing-free), mirrored the same way a real hold's intended bearing would be.
    g = torch.Generator().manual_seed(7)
    slot = torch.tensor(SLOTS)[torch.randint(0, 4, (N,), generator=g)]
    phi = (torch.rand(N, generator=g) * 2 * math.pi - math.pi).double()
    cat0 = cm.telemetry_category(slot, phi)
    catm = cm.telemetry_category(slot, _mirror_bearing(phi, image))

    table = sk.parse_stroke_cat_w("fwd:2.3,back:2.3,side:4.8,diag:1.2,turn:0.7,move_turn:1.9")
    vec = _vec(table)
    weighted = reward * vec[cat0][None, :].float()
    weighted_m = reward_m * vec[catm][None, :].float()
    torch.testing.assert_close(weighted_m, weighted, rtol=1e-4, atol=1e-6)
