"""Unit tests for the omni per-foot stance-stroke income and the body-centring bonus (pure torch).

These pin down the arithmetic that trains (stride/posture plan, step 2):

* a planted stance sweep pays ``(sweep / L)^p`` exactly once, at close;
* the pay does not depend on world yaw or position;
* a turn's tangential sweep pays, and the opposite spin pays nothing;
* sway that nets to zero pays nothing, and a world slide pays almost nothing (the per-stance slip gate);
* short stances pay 0, and a split stance pays less than one long stance;
* stale (repeated) force readings move the pay by at most one step's sweep;
* STAND pays 0, turn-only pays > 0, reset/RSI stances pay 0, and an open stance at reset is discarded unpaid;
* both terms are invariant under the omni mirror group {I, M_y, M_x, R_z(pi)}.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

TASKS = Path(__file__).resolve().parents[2] / "parkour" / "parkour_tasks" / "parkour_tasks"
STROKE_MATH_PY = TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_stroke_math.py"
OMNI_MIRROR_PY = TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_mirror.py"
CLOCK_REWARD_PY = TASKS / "crab_hex_forward_task" / "mdp" / "crab_hex_clock_reward.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm = _load("crab_hex_omni_stroke_math__test", STROKE_MATH_PY)

DT = 0.02
F = 6
L_TRANSLATION = 0.35  # the clock's planned stroke for any translation at or below 0.35 m/s


# ------------------------------------------------------------------------------------------ harness
def _run_stroke(force, foot_pos_w, root_pos_w, heading, cmd, stand=None, v_clk=None, resets=(), **kw):
    """Run ``foot_stroke_step`` over a ``[T, N, ...]`` trajectory; returns (reward [T, N], pay [T, N, F])."""
    T, N = force.shape[:2]
    if stand is None:
        stand = torch.zeros(T, N, dtype=torch.bool)
    if v_clk is None:
        v_clk = torch.maximum(torch.linalg.norm(cmd[..., :2], dim=-1), 1.2 * cmd[..., 2].abs())
    st = sm.init_stroke_state(N, F)
    rewards, pays = [], []
    for t in range(T):
        if t in resets:
            sm.reset_stroke_state(st)
        r, st, pay = sm.foot_stroke_step(force[t], foot_pos_w[t], root_pos_w[t], heading[t], cmd[t],
                                         stand[t], v_clk[t], st, DT, **kw)
        rewards.append(r)
        pays.append(pay)
    return torch.stack(rewards), torch.stack(pays)


def _translation_stance(v=0.25, stance_steps=20, lead=4, tail=4, heading=0.0, origin=(0.0, 0.0),
                        foot_body=(0.5, 0.7), direction=(1.0, 0.0), slide=0.0):
    """One env: the body translates at ``v`` along body ``direction``; foot 0 is loaded for
    ``stance_steps`` steps and is fixed in the world (or slides with the body at ``slide * v``)."""
    T = lead + stance_steps + tail
    dir_b = torch.tensor(direction) / torch.linalg.norm(torch.tensor(direction))
    c, s = math.cos(heading), math.sin(heading)
    rot = torch.tensor([[c, -s], [s, c]])
    dir_w = rot @ dir_b
    root = torch.zeros(T, 1, 3)
    foot = torch.zeros(T, 1, F, 3)
    force = torch.zeros(T, 1, F)
    fb = rot @ torch.tensor(foot_body)
    for t in range(T):
        root[t, 0, :2] = torch.tensor(origin) + dir_w * v * DT * t
        foot[t, 0, :, :2] = root[t, 0, :2] + fb  # unloaded feet ride with the body
        if lead <= t < lead + stance_steps:
            force[t, 0, 0] = 300.0
    # foot 0: planted where it touched down, optionally sliding with the body
    td = root[lead, 0, :2] + fb
    for t in range(lead, T):
        foot[t, 0, 0, :2] = td + dir_w * slide * v * DT * (t - lead)
    cmd = torch.zeros(T, 1, 3)
    cmd[:, 0, :2] = dir_b * v
    hdg = torch.full((T, 1), heading)
    return force, foot, root, hdg, cmd


# ------------------------------------------------------------------------------------------ stroke
def test_planted_sweep_pays_normalised_square_once_at_close():
    v, K = 0.25, 20
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    rew, pay = _run_stroke(force, foot, root, hdg, cmd)
    # K loaded samples give K-1 loaded intervals; the first loaded step needs the previous reading loaded too.
    expected = ((K - 1) * v * DT / L_TRANSLATION) ** 2
    assert pay[..., 0].sum().item() == pytest.approx(expected, rel=1e-5)
    nz = (pay[..., 0] > 0).nonzero()
    assert len(nz) == 1  # exactly once
    t_close = nz[0, 0].item()
    assert t_close == 4 + K + sm.N_OFF - 1  # closes on the N_OFF-th consecutive unloaded step
    assert rew[t_close, 0].item() == pytest.approx(expected / DT, rel=1e-5)
    assert pay[..., 1:].abs().sum().item() == 0.0


@pytest.mark.parametrize("heading", [0.0, 0.7, -2.1, math.pi])
@pytest.mark.parametrize("origin", [(0.0, 0.0), (13.0, -4.5)])
def test_pay_is_independent_of_world_yaw_and_position(heading, origin):
    ref = _run_stroke(*_translation_stance())[1].sum()
    got = _run_stroke(*_translation_stance(heading=heading, origin=origin))[1].sum()
    assert got.item() == pytest.approx(ref.item(), rel=1e-4)


@pytest.mark.parametrize("direction", [(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (0.7, 0.7)])
def test_every_translation_direction_pays_the_same(direction):
    ref = _run_stroke(*_translation_stance())[1].sum()
    got = _run_stroke(*_translation_stance(direction=direction))[1].sum()
    assert got.item() == pytest.approx(ref.item(), rel=1e-4)


def _turn_stance(w_cmd=0.3, w_act=0.3, r0=(0.3, 0.9), K=20, lead=4, tail=4):
    T = lead + K + tail
    root = torch.zeros(T, 1, 3)
    hdg = torch.tensor([[w_act * DT * t] for t in range(T)])
    foot = torch.zeros(T, 1, F, 3)
    force = torch.zeros(T, 1, F)
    r0 = torch.tensor(r0)
    for t in range(T):
        c, s = math.cos(hdg[t, 0]), math.sin(hdg[t, 0])
        foot[t, 0, :, :2] = torch.tensor([[c, -s], [s, c]]) @ r0
        if lead <= t < lead + K:
            force[t, 0, 0] = 300.0
    c, s = math.cos(hdg[lead, 0]), math.sin(hdg[lead, 0])
    planted = torch.tensor([[c, -s], [s, c]]) @ r0
    foot[lead:, 0, 0, :2] = planted
    cmd = torch.zeros(T, 1, 3)
    cmd[:, 0, 2] = w_cmd
    return force, foot, root, hdg, cmd


def test_turn_in_place_tangential_sweep_pays():
    w, K, r0 = 0.3, 20, (0.3, 0.9)
    rew, pay = _run_stroke(*_turn_stance(w_cmd=w, w_act=w, r0=r0, K=K))
    radius = math.hypot(*r0)
    f_hz = 0.5 * min(1.2 * w / 0.35, 1.0)
    L = (w * radius) / (2 * f_hz)
    # chord-vs-arc: the per-step body-frame displacement of a planted foot is a chord, projected on the tangent
    expected = ((K - 1) * radius * w * DT / L) ** 2
    assert pay.sum().item() == pytest.approx(expected, rel=1e-3)
    assert pay.sum().item() > 0


def test_turning_the_wrong_way_pays_nothing():
    _, pay = _run_stroke(*_turn_stance(w_cmd=0.3, w_act=-0.3))
    assert pay.sum().item() == 0.0


def test_sway_that_nets_to_zero_pays_nothing():
    v, K = 0.25, 21
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    # body goes forward for half the stance, then back to where it started by the last loaded step
    # (k = K - 1); foot planted
    lead = 4
    for t in range(lead, lead + K + 4):
        k = t - lead
        x = v * DT * max(min(k, K - 1 - k), 0)
        root[t, 0, 0] = root[lead, 0, 0] + x
    _, pay = _run_stroke(force, foot, root, hdg, cmd)
    assert pay.sum().item() == pytest.approx(0.0, abs=1e-6)


def test_world_slide_is_gated_out():
    planted = _run_stroke(*_translation_stance(slide=0.0))[1].sum().item()
    # foot slides forward at half the body speed: body-relative sweep halves, world path = sweep -> rho = 1
    slide = _run_stroke(*_translation_stance(slide=0.5))[1].sum().item()
    half_sweep_ungated = planted / 4.0
    gate = math.exp(-(((1.0 - sm.RHO0) / sm.DRHO) ** 2))
    assert slide == pytest.approx(half_sweep_ungated * gate, rel=1e-3)
    assert slide < 0.1 * half_sweep_ungated


@pytest.mark.parametrize("K,pays", [(2, False), (5, False), (6, True)])
def test_short_stances_pay_nothing(K, pays):
    _, pay = _run_stroke(*_translation_stance(stance_steps=K))
    assert (pay.sum().item() > 0) is pays


def test_split_stance_pays_less_than_one_long_stance():
    long_ = _run_stroke(*_translation_stance(stance_steps=24))[1].sum().item()
    force, foot, root, hdg, cmd = _translation_stance(stance_steps=24)
    force[4 + 11: 4 + 13, 0, 0] = 0.0  # two unloaded steps in the middle close it and reopen it
    split = _run_stroke(force, foot, root, hdg, cmd)[1].sum().item()
    assert 0 < split < long_


def test_one_step_dip_does_not_close_the_stance():
    force, foot, root, hdg, cmd = _translation_stance(stance_steps=24)
    force[4 + 12, 0, 0] = 5.0
    _, pay = _run_stroke(force, foot, root, hdg, cmd)
    assert (pay[..., 0] > 0).sum().item() == 1


def test_stale_force_readings_move_pay_by_at_most_one_step():
    v, K = 0.25, 40
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    stale = force.clone()
    for t in range(1, stale.shape[0]):
        if t % 5 == 0:
            stale[t] = stale[t - 1]  # the contact sensor skipped its refresh: last reading repeats
    p_ref = _run_stroke(force, foot, root, hdg, cmd)[1].sum().item()
    p_stale = _run_stroke(stale, foot, root, hdg, cmd)[1].sum().item()
    one_step = v * DT / L_TRANSLATION
    assert abs(math.sqrt(p_stale) - math.sqrt(p_ref)) <= one_step + 1e-6


def test_stand_pays_nothing_and_turn_only_pays():
    force, foot, root, hdg, cmd = _translation_stance()
    stand = torch.ones(force.shape[:2], dtype=torch.bool)
    assert _run_stroke(force, foot, root, hdg, cmd, stand=stand)[1].sum().item() == 0.0
    assert _run_stroke(*_turn_stance())[1].sum().item() > 0.0


def test_below_clock_stop_speed_pays_nothing():
    force, foot, root, hdg, cmd = _translation_stance(v=0.19)
    assert _run_stroke(force, foot, root, hdg, cmd)[1].sum().item() == 0.0


def test_stance_opened_right_after_reset_is_not_paid():
    # the stance opens on step 1 after the reset (an RSI frame that starts mid-stance)
    force, foot, root, hdg, cmd = _translation_stance(lead=1)
    _, pay = _run_stroke(force, foot, root, hdg, cmd, resets=(0,))
    assert pay.sum().item() == 0.0
    # the same stance opening 3 steps after the reset is paid
    force, foot, root, hdg, cmd = _translation_stance(lead=3)
    assert _run_stroke(force, foot, root, hdg, cmd, resets=(0,))[1].sum().item() > 0.0


def test_open_stance_at_reset_is_discarded_unpaid():
    force, foot, root, hdg, cmd = _translation_stance(stance_steps=20)
    _, pay = _run_stroke(force, foot, root, hdg, cmd, resets=(4 + 15,))  # reset (fall/timeout) mid-stance
    assert pay.sum().item() == 0.0


def test_output_is_nonnegative_and_finite_under_fuzzing():
    g = torch.Generator().manual_seed(0)
    T, N = 200, 64
    force = torch.rand(T, N, F, generator=g) * 200.0 * (torch.rand(T, N, F, generator=g) > 0.4)
    foot = torch.cumsum(torch.randn(T, N, F, 3, generator=g) * 0.01, dim=0)
    root = torch.cumsum(torch.randn(T, N, 3, generator=g) * 0.01, dim=0)
    hdg = torch.cumsum(torch.randn(T, N, generator=g) * 0.05, dim=0)
    cmd = torch.randn(T, N, 3, generator=g) * 0.3
    stand = torch.rand(T, N, generator=g) < 0.1
    rew, pay = _run_stroke(force, foot, root, hdg, cmd, stand=stand, resets=(50, 120))
    assert torch.isfinite(rew).all() and (rew >= 0).all() and (pay >= 0).all()


# ------------------------------------------------------------------------------------------ centring
def _run_centring(foot, root, hdg, cmd, vel_b, stand=None, resets=(), **kw):
    T, N = cmd.shape[:2]
    if stand is None:
        stand = torch.zeros(T, N, dtype=torch.bool)
    st = sm.init_centring_state(N)
    out, offs = [], []
    for t in range(T):
        if t in resets:
            sm.reset_centring_state(st)
        b, st, o = sm.body_centring_step(foot[t], root[t], hdg[t], cmd[t], stand[t], vel_b[t], st, DT, **kw)
        out.append(b)
        offs.append(o)
    return torch.stack(out), torch.stack(offs)


def _six_feet(offset_y=0.0, T=10):
    """Six feet symmetric about the body, shifted by ``offset_y`` in body y (a body lead of -offset_y on +y)."""
    base = torch.tensor([[0.6, 0.9], [0.6, -0.9], [0.0, 1.0], [0.0, -1.0], [-0.6, 0.9], [-0.6, -0.9]])
    base = base + torch.tensor([0.0, offset_y])
    foot = torch.zeros(T, 1, F, 3)
    foot[..., :2] = base
    root = torch.zeros(T, 1, 3)
    hdg = torch.zeros(T, 1)
    return foot, root, hdg


def test_centring_pays_one_on_a_centred_side_walk_and_less_when_the_body_leads():
    foot, root, hdg = _six_feet(offset_y=0.0)
    cmd = torch.zeros(10, 1, 3); cmd[..., 1] = 0.22
    vel = torch.zeros(10, 1, 2); vel[..., 1] = 0.10
    b, o = _run_centring(foot, root, hdg, cmd, vel)
    assert b[-1].item() == pytest.approx(1.0, abs=1e-6) and o[-1].item() == pytest.approx(0.0, abs=1e-6)
    foot, root, hdg = _six_feet(offset_y=-0.23)  # feet 0.23 m behind the body along +y: the omni side posture
    b, o = _run_centring(foot, root, hdg, cmd, vel)
    assert o[-1].item() == pytest.approx(0.23, abs=1e-6)
    assert b[-1].item() == pytest.approx(math.exp(-((0.23 / sm.CENTRING_SIGMA_M) ** 2)), rel=1e-5)


@pytest.mark.parametrize("cmd_vec", [(0.25, 0.0, 0.0), (-0.22, 0.0, 0.0), (0.0, 0.0, 0.3), (0.0, 0.0, 0.0)])
def test_centring_pays_nothing_on_fwd_back_turn_and_zero_command(cmd_vec):
    foot, root, hdg = _six_feet()
    cmd = torch.tensor(cmd_vec).expand(10, 1, 3).clone()
    vel = cmd[..., :2].clone()
    b, _ = _run_centring(foot, root, hdg, cmd, vel)
    assert b.abs().sum().item() == 0.0


def test_centring_pays_nothing_on_stand_or_to_a_statue():
    foot, root, hdg = _six_feet()
    cmd = torch.zeros(10, 1, 3); cmd[..., 1] = 0.22
    stand = torch.ones(10, 1, dtype=torch.bool)
    assert _run_centring(foot, root, hdg, cmd, torch.zeros(10, 1, 2), stand=stand)[0].abs().sum().item() == 0.0
    assert _run_centring(foot, root, hdg, cmd, torch.zeros(10, 1, 2))[0].abs().sum().item() == 0.0


def test_centroid_ema_restarts_at_reset():
    foot, root, hdg = _six_feet(T=20)
    foot[10:, ..., 1] -= 0.3  # the stance shifts; after a reset the EMA must start from the new centroid
    cmd = torch.zeros(20, 1, 3); cmd[..., 1] = 0.22
    vel = torch.zeros(20, 1, 2); vel[..., 1] = 0.1
    _, o = _run_centring(foot, root, hdg, cmd, vel, resets=(10,))
    assert o[10].item() == pytest.approx(0.3, abs=1e-6)


# ------------------------------------------------------------------------------------------ mirror
mirror = _load("crab_hex_omni_mirror__stroke_test", OMNI_MIRROR_PY)
FOOT_ORDER = _load("crab_hex_clock_reward__stroke_test", CLOCK_REWARD_PY).FOOT_ORDER
PERM_LR = [FOOT_ORDER.index(mirror.LEG_SWAP_LR[leg]) for leg in FOOT_ORDER]
PERM_FA = [FOOT_ORDER.index(mirror.LEG_SWAP_FA[leg]) for leg in FOOT_ORDER]


def _mirror(axis: int, perm, force, foot, root, hdg, cmd, vel):
    """Reflect the world across ``axis`` (0: x -> -x is M_x, 1: y -> -y is M_y) and relabel the feet.

    A world reflection with ``heading -> -heading`` is exactly the body-frame mirror of that axis; the
    command's ``axis`` component and ``wz`` flip sign, and so does the body velocity's ``axis`` component.
    """
    s = torch.ones(3); s[axis] = -1.0
    foot_m, root_m = foot * s, root * s
    cmd_m = cmd.clone(); cmd_m[..., axis] *= -1.0; cmd_m[..., 2] *= -1.0
    vel_m = vel.clone(); vel_m[..., axis] *= -1.0
    return force[..., perm], foot_m[..., perm, :], root_m, -hdg, cmd_m, vel_m


def _random_rollout(seed=1, T=300, N=16):
    g = torch.Generator().manual_seed(seed)
    phase = torch.rand(N, F, generator=g) * 2 * math.pi
    t = torch.arange(T)[:, None, None] * DT
    force = 300.0 * (torch.sin(2 * math.pi * 0.4 * t + phase) > 0)  # alternating stances
    base = torch.tensor([[0.6, 0.9], [0.6, -0.9], [0.0, 1.0], [0.0, -1.0], [-0.6, 0.9], [-0.6, -0.9]])
    foot = torch.zeros(T, N, F, 3)
    foot[..., :2] = base + torch.cumsum(torch.randn(T, N, F, 2, generator=g) * 0.004, dim=0)
    root = torch.zeros(T, N, 3)
    root[..., :2] = torch.cumsum(torch.randn(T, N, 2, generator=g) * 0.004 + 0.003, dim=0)
    hdg = torch.cumsum(torch.randn(T, N, generator=g) * 0.01, dim=0)
    cmd = (torch.randn(N, 3, generator=g) * 0.25).expand(T, N, 3).clone()
    vel = torch.randn(T, N, 2, generator=g) * 0.1
    return force, foot, root, hdg, cmd, vel


@pytest.mark.parametrize("image", ["M_y", "M_x", "R_z(pi)"])
def test_both_terms_are_mirror_equivariant(image):
    force, foot, root, hdg, cmd, vel = _random_rollout()
    args = (force, foot, root, hdg, cmd, vel)
    if image == "M_y":
        m, perm = _mirror(1, PERM_LR, *args), PERM_LR
    elif image == "M_x":
        m, perm = _mirror(0, PERM_FA, *args), PERM_FA
    else:
        m = _mirror(0, PERM_FA, *_mirror(1, PERM_LR, *args))
        perm = [PERM_LR[i] for i in PERM_FA]
    _, pay = _run_stroke(force, foot, root, hdg, cmd)
    _, pay_m = _run_stroke(m[0], m[1], m[2], m[3], m[4])
    assert pay.sum().item() > 0
    torch.testing.assert_close(pay_m, pay[..., perm], rtol=1e-4, atol=1e-6)
    b, o = _run_centring(foot, root, hdg, cmd, vel)
    b_m, o_m = _run_centring(m[1], m[2], m[3], m[4], m[5])
    torch.testing.assert_close(b_m, b, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(o_m, o, rtol=1e-4, atol=1e-6)
