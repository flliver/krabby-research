"""Unit tests for the omni per-foot stance-stroke income and the body-centring bonus (pure torch).

These pin down the arithmetic that trains (stride/posture plan, step 2, revised after its adversarial
review):

* a planted stance sweep pays ``(sweep / L)^p`` exactly once, at close;
* the pay does not depend on world yaw, position or travel direction;
* a turn's tangential sweep pays, and the opposite spin pays nothing;
* sway inside a stance nets out, and a world slide is gated out per stance;
* short stances pay 0, and a split stance pays less than one long stance;
* stale (repeated) force readings move the pay by at most one step's sweep;
* STAND pays 0, turn-only pays > 0, reset/RSI stances pay 0, and an open stance at reset is discarded unpaid;
* the review's exploits pay nothing:
  - a stick-slip shuffle whose repositions hide in one-step load dips;
  - backward paddling under load (no more than a planted foot);
  - a stance carried into STAND;
  - tripod-locked rocking in place;
* the facing relay's yaw rate is ignored on non-turning slots;
* the centring bonus is lateral-only, statue-proof (a rocking statue too), and re-seeds on resample;
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
TURN_R = 1.2  # the omni env's omni_clock_turn_radius
L_TRANSLATION = 0.35  # the clock's planned stroke for any translation at or below 0.35 m/s


# ------------------------------------------------------------------------------------------ harness
def _run_stroke(force, foot_pos_w, root_pos_w, heading, cmd, vel=None, wz=None, stand=None, turning=None,
                resets=(), **kw):
    """Run ``foot_stroke_step`` over a ``[T, N, ...]`` trajectory; returns (reward [T, N], pay [T, N, F]).

    ``vel`` / ``wz`` are the body's planar velocity and yaw rate (default: it tracks the command exactly).
    """
    T, N = force.shape[:2]
    kw.setdefault("power", 2.0)  # the arithmetic below is written for p = 2; the shipped default is pinned separately
    vel = cmd[..., :2] if vel is None else vel
    wz = cmd[..., 2] if wz is None else wz
    stand = torch.zeros(T, N, dtype=torch.bool) if stand is None else stand
    st = sm.init_stroke_state(N, F)
    rewards, pays = [], []
    for t in range(T):
        if t in resets:
            sm.reset_stroke_state(st)
        r, st, pay = sm.foot_stroke_step(
            force[t], foot_pos_w[t], root_pos_w[t], heading[t], cmd[t], stand[t], vel[t], wz[t], st, DT,
            TURN_R, None if turning is None else turning[t], **kw)
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
    td = root[lead, 0, :2] + fb
    for t in range(lead, T):
        foot[t, 0, 0, :2] = td + dir_w * slide * v * DT * (t - lead)
    cmd = torch.zeros(T, 1, 3)
    cmd[:, 0, :2] = dir_b * v
    hdg = torch.full((T, 1), heading)
    return force, foot, root, hdg, cmd


def _planted_pay(v=0.25, K=20):
    return ((K - 1) * v * DT / L_TRANSLATION) ** 2


# ------------------------------------------------------------------------------------------ stroke
def test_planted_sweep_pays_normalised_square_once_at_close():
    v, K = 0.25, 20
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    rew, pay = _run_stroke(force, foot, root, hdg, cmd)
    expected = _planted_pay(v, K)  # K loaded samples -> K-1 loaded intervals
    assert pay[..., 0].sum().item() == pytest.approx(expected, rel=1e-5)
    nz = (pay[..., 0] > 0).nonzero()
    assert len(nz) == 1
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
    foot[lead:, 0, 0, :2] = torch.tensor([[c, -s], [s, c]]) @ r0
    cmd = torch.zeros(T, 1, 3)
    cmd[:, 0, 2] = w_cmd
    vel = torch.zeros(T, 1, 2)
    wz = torch.full((T, 1), w_act)
    return (force, foot, root, hdg, cmd), dict(vel=vel, wz=wz)


def test_turn_in_place_tangential_sweep_pays():
    w, K, r0 = 0.3, 20, (0.3, 0.9)
    args, kw = _turn_stance(w_cmd=w, w_act=w, r0=r0, K=K)
    _, pay = _run_stroke(*args, **kw)
    radius = math.hypot(*r0)
    f_hz = 0.5 * min(TURN_R * w / 0.35, 1.0)
    L = max(w * radius, 0.5 * TURN_R * w) / (2 * f_hz)
    expected = ((K - 1) * radius * w * DT / L) ** 2  # chord vs arc: equal to 1e-3 at these angles
    assert pay.sum().item() == pytest.approx(expected, rel=1e-3)


def test_turning_the_wrong_way_pays_nothing():
    args, kw = _turn_stance(w_cmd=0.3, w_act=-0.3)
    assert _run_stroke(*args, **kw)[1].sum().item() == 0.0


def test_sway_that_nets_to_zero_pays_nothing():
    v, K = 0.25, 21
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    lead = 4
    for t in range(lead, lead + K + 4):
        k = t - lead
        root[t, 0, 0] = root[lead, 0, 0] + v * DT * max(min(k, K - 1 - k), 0)  # out and back within the stance
    _, pay = _run_stroke(force, foot, root, hdg, cmd)
    assert pay.sum().item() == pytest.approx(0.0, abs=1e-6)


def test_world_slide_is_gated_out():
    planted = _run_stroke(*_translation_stance(slide=0.0))[1].sum().item()
    slide = _run_stroke(*_translation_stance(slide=0.5))[1].sum().item()  # own sweep halves; rho = 1
    half_sweep_ungated = planted / 4.0
    gate = math.exp(-(((1.0 - sm.RHO0) / sm.DRHO) ** 2))
    assert slide == pytest.approx(half_sweep_ungated * gate, rel=1e-3)
    assert slide < 0.1 * half_sweep_ungated


@pytest.mark.parametrize("K,pays", [(2, False), (5, False), (6, True)])
def test_short_stances_pay_nothing(K, pays):
    assert (_run_stroke(*_translation_stance(stance_steps=K))[1].sum().item() > 0) is pays


def test_split_stance_pays_less_than_one_long_stance():
    long_ = _run_stroke(*_translation_stance(stance_steps=24))[1].sum().item()
    force, foot, root, hdg, cmd = _translation_stance(stance_steps=24)
    force[4 + 11: 4 + 13, 0, 0] = 0.0  # two unloaded steps close it; it reopens
    split = _run_stroke(force, foot, root, hdg, cmd)[1].sum().item()
    assert 0 < split < long_


def test_one_step_dip_neither_closes_the_stance_nor_loses_its_sweep():
    force, foot, root, hdg, cmd = _translation_stance(stance_steps=24)
    force[4 + 12, 0, 0] = 5.0
    _, pay = _run_stroke(force, foot, root, hdg, cmd)
    assert (pay[..., 0] > 0).sum().item() == 1
    assert pay.sum().item() == pytest.approx(_planted_pay(K=24), rel=1e-5)  # the dip's sweep counts in full


def test_stale_force_readings_move_pay_by_at_most_one_step():
    v, K = 0.25, 40
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    stale = force.clone()
    for t in range(1, stale.shape[0]):
        if t % 5 == 0:
            stale[t] = stale[t - 1]  # the contact sensor skipped its refresh: last reading repeats
    p_ref = _run_stroke(force, foot, root, hdg, cmd)[1].sum().item()
    p_stale = _run_stroke(stale, foot, root, hdg, cmd)[1].sum().item()
    assert abs(math.sqrt(p_stale) - math.sqrt(p_ref)) <= v * DT / L_TRANSLATION + 1e-6


def test_stand_pays_nothing_and_turn_only_pays():
    force, foot, root, hdg, cmd = _translation_stance()
    stand = torch.ones(force.shape[:2], dtype=torch.bool)
    assert _run_stroke(force, foot, root, hdg, cmd, stand=stand)[1].sum().item() == 0.0
    args, kw = _turn_stance()
    assert _run_stroke(*args, **kw)[1].sum().item() > 0.0


def test_below_clock_stop_speed_pays_nothing():
    assert _run_stroke(*_translation_stance(v=0.19))[1].sum().item() == 0.0


def test_stance_opened_right_after_reset_is_not_paid():
    _, pay = _run_stroke(*_translation_stance(lead=2), resets=(0,))  # third reward call after the reset
    assert pay.sum().item() == 0.0
    assert _run_stroke(*_translation_stance(lead=3), resets=(0,))[1].sum().item() > 0.0


def test_open_stance_at_reset_is_discarded_unpaid():
    _, pay = _run_stroke(*_translation_stance(stance_steps=20), resets=(4 + 15,))  # fall/timeout mid-stance
    assert pay.sum().item() == 0.0


# ------------------------------------------------------------------------------------------ the review's exploits
@pytest.mark.parametrize("K", [3, 7, 15])
def test_stick_slip_shuffle_hidden_in_one_step_dips_pays_almost_nothing(K):
    """The foot holds while the body advances, then re-plants AHEAD during a one-step load dip, every K steps.
    Its net sweep relative to the body is ~0: a shuffle, not a stroke."""
    v, S = 0.25, 60
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=S)
    lead = 4
    anchor = foot[lead, 0, 0, :2].clone()
    for t in range(lead, lead + S):
        k = t - lead
        if k > 0 and k % K == 0:
            force[t, 0, 0] = 5.0  # dip
            anchor = anchor + torch.tensor([v * DT * K, 0.0])  # re-plant a cycle's advance ahead
        foot[t, 0, 0, :2] = anchor
    shuffle = _run_stroke(force, foot, root, hdg, cmd)[1].sum().item()
    assert shuffle <= 0.05 * _planted_pay(v, S)


@pytest.mark.parametrize("s", [0.3, 0.6, 1.0])
def test_backward_paddling_earns_no_more_than_a_planted_foot(s):
    planted = _run_stroke(*_translation_stance(slide=0.0))[1].sum().item()
    paddle = _run_stroke(*_translation_stance(slide=-s))[1].sum().item()  # the loaded foot slides BACKWARD
    assert paddle <= planted * (1 + 1e-5)


def test_a_stance_carried_into_stand_pays_nothing():
    force, foot, root, hdg, cmd = _translation_stance(stance_steps=30)
    stand = torch.zeros(force.shape[:2], dtype=torch.bool)
    stand[4 + 20:, 0] = True  # the command resamples to STAND mid-stance
    assert _run_stroke(force, foot, root, hdg, cmd, stand=stand)[1].sum().item() == 0.0


def _alternating_gait(T, v, rock: bool):
    """Two feet (A = 0, B = 1) alternate 1-s stances at the clock's 0.5 Hz. The body either walks
    steadily forward (+v) or rocks: +v while A is planted, -v while B is planted (net progress 0)."""
    half = int(round(1.0 / DT))
    force = torch.zeros(T, 1, F)
    foot = torch.zeros(T, 1, F, 3)
    root = torch.zeros(T, 1, 3)
    vel = torch.zeros(T, 1, 2)
    x = 0.0
    planted = {0: None, 1: None}
    body = {0: torch.tensor([0.4, 0.8]), 1: torch.tensor([0.4, -0.8])}
    for t in range(T):
        a_phase = (t // half) % 2 == 0
        vx = v if (a_phase or not rock) else -v
        vel[t, 0, 0] = vx
        root[t, 0, 0] = x
        for i, down in ((0, a_phase), (1, not a_phase)):
            if down:
                if planted[i] is None:
                    planted[i] = root[t, 0, :2] + body[i]
                foot[t, 0, i, :2] = planted[i]
                force[t, 0, i] = 300.0
            else:
                planted[i] = None
                foot[t, 0, i, :2] = root[t, 0, :2] + body[i]
        x += vx * DT
    cmd = torch.zeros(T, 1, 3)
    cmd[..., 0] = v
    return (force, foot, root, torch.zeros(T, 1), cmd), dict(vel=vel)


def test_tripod_locked_rocking_in_place_pays_nothing_once_the_progress_guard_has_settled():
    T = int(30 / DT)
    args, kw = _alternating_gait(T, 0.25, rock=True)
    _, pay_rock = _run_stroke(*args, **kw)
    args, kw = _alternating_gait(T, 0.25, rock=False)
    _, pay_walk = _run_stroke(*args, **kw)
    late = int(16 / DT)
    assert pay_walk[late:].sum().item() > 0.0
    assert pay_rock[late:].sum().item() == 0.0


def test_facing_relay_yaw_rate_is_ignored_on_non_turning_slots():
    force, foot, root, hdg, cmd = _translation_stance()
    ref = _run_stroke(force, foot, root, hdg, cmd)[1].sum().item()
    relay = cmd.clone()
    relay[..., 2] = 0.3  # the facing P-control's relay on a MOVE slot
    no_turn = torch.zeros(force.shape[:2], dtype=torch.bool)
    got = _run_stroke(force, foot, root, hdg, relay, wz=torch.zeros(force.shape[:2]), turning=no_turn)[1].sum().item()
    assert got == pytest.approx(ref, rel=1e-6)
    turn = torch.ones(force.shape[:2], dtype=torch.bool)
    moved = _run_stroke(force, foot, root, hdg, relay, wz=torch.zeros(force.shape[:2]), turning=turn)[1].sum().item()
    assert moved != pytest.approx(ref, rel=1e-3)


def test_output_is_nonnegative_and_finite_under_fuzzing():
    g = torch.Generator().manual_seed(0)
    T, N = 200, 64
    force = torch.rand(T, N, F, generator=g) * 200.0 * (torch.rand(T, N, F, generator=g) > 0.4)
    foot = torch.cumsum(torch.randn(T, N, F, 3, generator=g) * 0.01, dim=0)
    root = torch.cumsum(torch.randn(T, N, 3, generator=g) * 0.01, dim=0)
    hdg = torch.cumsum(torch.randn(T, N, generator=g) * 0.05, dim=0)
    cmd = torch.randn(T, N, 3, generator=g) * 0.3
    vel = torch.randn(T, N, 2, generator=g) * 0.2
    wz = torch.randn(T, N, generator=g) * 0.2
    stand = torch.rand(T, N, generator=g) < 0.1
    turning = torch.rand(T, N, generator=g) < 0.5
    rew, pay = _run_stroke(force, foot, root, hdg, cmd, vel=vel, wz=wz, stand=stand, turning=turning, resets=(50, 120))
    assert torch.isfinite(rew).all() and (rew >= 0).all() and (pay >= 0).all()


def test_loaded_stance_edges_is_pure():
    force = torch.tensor([[60.0, 10.0]])
    open_ = torch.tensor([[False, True]])
    low = torch.tensor([[0, 1]])
    prev_ok = torch.tensor([[True, False]])
    has_prev = torch.tensor([True])
    snapshot = [t.clone() for t in (force, open_, low, prev_ok, has_prev)]
    e = sm.loaded_stance_edges(force, open_, low, prev_ok, has_prev)
    for a, b in zip(snapshot, (force, open_, low, prev_ok, has_prev)):
        assert torch.equal(a, b)
    assert e.opening.tolist() == [[True, False]] and e.closing.tolist() == [[False, True]]
    assert e.loaded.tolist() == [[True, False]]


# ------------------------------------------------------------------------------------------ centring
def _run_centring(foot, root, hdg, cmd, vel_b, stand=None, lat_share=None, resampled=None, resets=(), **kw):
    T, N = cmd.shape[:2]
    stand = torch.zeros(T, N, dtype=torch.bool) if stand is None else stand
    st = sm.init_centring_state(N)
    out, offs = [], []
    for t in range(T):
        if t in resets:
            sm.reset_centring_state(st)
        b, st, o = sm.body_centring_step(
            foot[t], root[t], hdg[t], cmd[t], stand[t], vel_b[t], st, DT,
            None if lat_share is None else lat_share[t], None if resampled is None else resampled[t], **kw)
        out.append(b)
        offs.append(o)
    return torch.stack(out), torch.stack(offs)


def _six_feet(offset_y=0.0, T=10):
    """Six feet symmetric about the body, shifted by ``offset_y`` in body y (a body lead of -offset_y on +y)."""
    base = torch.tensor([[0.6, 0.9], [0.6, -0.9], [0.0, 1.0], [0.0, -1.0], [-0.6, 0.9], [-0.6, -0.9]])
    foot = torch.zeros(T, 1, F, 3)
    foot[..., :2] = base + torch.tensor([0.0, offset_y])
    return foot, torch.zeros(T, 1, 3), torch.zeros(T, 1)


def _side(T=10, v_cmd=0.22, v_act=0.10):
    cmd = torch.zeros(T, 1, 3)
    cmd[..., 1] = v_cmd
    vel = torch.zeros(T, 1, 2)
    vel[..., 1] = v_act
    return cmd, vel


def test_centring_pays_one_on_a_centred_side_walk_and_less_when_the_body_leads():
    cmd, vel = _side()
    b, o = _run_centring(*_six_feet(0.0), cmd, vel)
    assert b[-1].item() == pytest.approx(1.0, abs=1e-6) and o[-1].item() == pytest.approx(0.0, abs=1e-6)
    b, o = _run_centring(*_six_feet(-0.23), cmd, vel)  # feet 0.23 m behind the body along +y: the omni side posture
    assert o[-1].item() == pytest.approx(0.23, abs=1e-6)
    assert b[-1].item() == pytest.approx(math.exp(-((0.23 / sm.CENTRING_SIGMA_M) ** 2)), rel=1e-5)


@pytest.mark.parametrize("cmd_vec", [(0.25, 0.0, 0.0), (-0.22, 0.0, 0.0), (0.0, 0.0, 0.3), (0.0, 0.0, 0.0)])
def test_centring_pays_nothing_on_fwd_back_turn_and_zero_command(cmd_vec):
    cmd = torch.tensor(cmd_vec).expand(10, 1, 3).clone()
    b, _ = _run_centring(*_six_feet(), cmd, cmd[..., :2].clone())
    assert b.abs().sum().item() == 0.0


def test_centring_pays_nothing_on_stand_or_to_a_statue():
    cmd, _ = _side()
    stand = torch.ones(10, 1, dtype=torch.bool)
    assert _run_centring(*_six_feet(), cmd, torch.zeros(10, 1, 2), stand=stand)[0].abs().sum().item() == 0.0
    assert _run_centring(*_six_feet(), cmd, torch.zeros(10, 1, 2))[0].abs().sum().item() == 0.0


def test_centring_pays_nothing_to_a_rocking_statue():
    T = int(10 / DT)
    cmd, _ = _side(T)
    vel = torch.zeros(T, 1, 2)
    vel[:, 0, 1] = 0.06 * torch.sin(2 * math.pi * 1.0 * torch.arange(T) * DT)
    b, _ = _run_centring(*_six_feet(T=T), cmd, vel)
    assert b[int(4 / DT):].max().item() == 0.0


def test_centroid_ema_restarts_at_reset_and_at_a_resample():
    foot, root, hdg = _six_feet(T=20)
    foot[10:, ..., 1] -= 0.3  # the stance shifts; a fresh env must start from the new centroid
    cmd, vel = _side(20)
    _, o = _run_centring(foot, root, hdg, cmd, vel, resets=(10,))
    assert o[10].item() == pytest.approx(0.3, abs=1e-6)
    resampled = torch.zeros(20, 1, dtype=torch.bool)
    resampled[10] = True
    _, o = _run_centring(foot, root, hdg, cmd, vel, resampled=resampled)
    assert o[10].item() == pytest.approx(0.3, abs=1e-6)
    _, o = _run_centring(foot, root, hdg, cmd, vel)
    assert 0 < o[10].item() < 0.3  # without either, the EMA lags


def test_lateral_share_can_come_from_the_intended_bearing():
    cmd, vel = _side()
    cmd[..., 0] = 0.05  # a heading error rotated the body-frame command a little
    lat = torch.ones(10, 1)  # the hold's intended bearing is pure lateral
    b_given, _ = _run_centring(*_six_feet(), cmd, vel, lat_share=lat)
    b_body, _ = _run_centring(*_six_feet(), cmd, vel)
    assert b_given[-1].item() == pytest.approx(1.0, abs=1e-6)
    assert b_body[-1].item() < 1.0


# ------------------------------------------------------------------------------------------ mirror
mirror = _load("crab_hex_omni_mirror__stroke_test", OMNI_MIRROR_PY)
FOOT_ORDER = _load("crab_hex_clock_reward__stroke_test", CLOCK_REWARD_PY).FOOT_ORDER
PERM_LR = [FOOT_ORDER.index(mirror.LEG_SWAP_LR[leg]) for leg in FOOT_ORDER]
PERM_FA = [FOOT_ORDER.index(mirror.LEG_SWAP_FA[leg]) for leg in FOOT_ORDER]


def _mirror(axis: int, perm, force, foot, root, hdg, cmd, vel, wz):
    """Reflect the world across ``axis`` (0: x -> -x is M_x, 1: y -> -y is M_y) and relabel the feet.

    A world reflection with ``heading -> -heading`` is exactly the body-frame mirror of that axis: the
    command's ``axis`` component, the body velocity's ``axis`` component and every yaw rate flip sign.
    """
    s = torch.ones(3)
    s[axis] = -1.0
    cmd_m = cmd.clone()
    cmd_m[..., axis] *= -1.0
    cmd_m[..., 2] *= -1.0
    vel_m = vel.clone()
    vel_m[..., axis] *= -1.0
    return force[..., perm], (foot * s)[..., perm, :], root * s, -hdg, cmd_m, vel_m, -wz


def _random_rollout(seed=1, T=400, N=16):
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
    vel = cmd[..., :2] * 0.6 + torch.randn(T, N, 2, generator=g) * 0.05
    wz = cmd[..., 2] * 0.6 + torch.randn(T, N, generator=g) * 0.05
    return force, foot, root, hdg, cmd, vel, wz


@pytest.mark.parametrize("image", ["M_y", "M_x", "R_z(pi)"])
def test_both_terms_are_mirror_equivariant(image):
    args = _random_rollout()
    if image == "M_y":
        m, perm = _mirror(1, PERM_LR, *args), PERM_LR
    elif image == "M_x":
        m, perm = _mirror(0, PERM_FA, *args), PERM_FA
    else:
        m = _mirror(0, PERM_FA, *_mirror(1, PERM_LR, *args))
        perm = [PERM_LR[i] for i in PERM_FA]
    force, foot, root, hdg, cmd, vel, wz = args
    _, pay = _run_stroke(force, foot, root, hdg, cmd, vel=vel, wz=wz)
    _, pay_m = _run_stroke(m[0], m[1], m[2], m[3], m[4], vel=m[5], wz=m[6])
    assert pay.sum().item() > 0
    torch.testing.assert_close(pay_m, pay[..., perm], rtol=1e-4, atol=1e-6)
    b, o = _run_centring(foot, root, hdg, cmd, vel)
    b_m, o_m = _run_centring(m[1], m[2], m[3], m[4], m[5])
    assert b.sum().item() > 0
    torch.testing.assert_close(b_m, b, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(o_m, o, rtol=1e-4, atol=1e-6)


def test_info_reports_the_closing_stance_before_it_is_cleared():
    v, K = 0.25, 20
    force, foot, root, hdg, cmd = _translation_stance(v=v, stance_steps=K)
    st = sm.init_stroke_state(1, F)
    seen = []
    for t in range(force.shape[0]):
        info = {}
        _, st, pay = sm.foot_stroke_step(force[t], foot[t], root[t], hdg[t], cmd[t], torch.zeros(1, dtype=torch.bool),
                                         cmd[t, :, :2], cmd[t, :, 2], st, DT, TURN_R, info=info)
        if info["closing"][0, 0]:
            seen.append((info["B"][0, 0].item(), info["n"][0, 0].item(), pay[0, 0].item()))
    assert len(seen) == 1
    B, n, pay = seen[0]
    assert B == pytest.approx((K - 1) * v * DT, rel=1e-5) and n == K - 1 and pay > 0


# ------------------------------------------------------------------------------------------ re-review fixes
def _walker_run(T, v=0.25, resample_at=(), rest_until=0):
    """The steady two-foot walker, optionally at rest (zero body velocity) until ``rest_until`` and with
    command resamples flagged at ``resample_at``. Returns pay [T, 1, F]."""
    args, kw = _alternating_gait(T, v, rock=False)
    force, foot, root, hdg, cmd = args
    vel = kw["vel"].clone()
    vel[:rest_until] = 0.0
    st = sm.init_stroke_state(1, F)
    pays = []
    for t in range(T):
        res = torch.tensor([t in resample_at])
        _, st, pay = sm.foot_stroke_step(force[t], foot[t], root[t], hdg[t], cmd[t], torch.zeros(1, dtype=torch.bool),
                                         vel[t], torch.zeros(1), st, DT, TURN_R, torch.zeros(1, dtype=torch.bool), res)
        pays.append(pay)
    return torch.stack(pays)


def test_progress_guard_starts_a_hold_at_its_pass_value_not_cold():
    T = int(12 / DT)
    pay = _walker_run(T)
    # the first stance opens inside the reset taint and is (correctly) unpaid; the next two close at ~2 s and ~3 s
    first, steady = pay[int(1.5 / DT): int(3.5 / DT)].sum().item() / 2.0, pay[int(6 / DT):].sum().item() / 6.0
    assert first >= 0.9 * steady  # an honest gait is paid in full from the hold's first paid stance


def test_progress_guard_reseeds_at_a_resample():
    T = int(20 / DT)
    # a long stretch of zero progress drives the guard to 0; a resample then gives the new hold a clean start
    pay_no = _walker_run(T, rest_until=int(10 / DT))
    pay_re = _walker_run(T, rest_until=int(10 / DT), resample_at=(int(10 / DT),))
    win = slice(int(10 / DT), int(12 / DT))
    assert pay_re[win].sum().item() > pay_no[win].sum().item()


def _yaw_wiggle_walk(T, v=0.25, amp=0.05):
    """The two-foot walker translating along world +x while its heading wiggles by +-``amp`` in phase with
    the stances (a linear ramp -amp -> +amp during A's stance and back during B's)."""
    half = int(round(1.0 / DT))
    force = torch.zeros(T, 1, F)
    foot = torch.zeros(T, 1, F, 3)
    root = torch.zeros(T, 1, 3)
    hdg = torch.zeros(T, 1)
    body = {0: torch.tensor([0.4, 0.8]), 1: torch.tensor([0.4, -0.8])}
    planted = {0: None, 1: None}
    for t in range(T):
        k = t % half
        a_phase = (t // half) % 2 == 0
        frac = k / half
        psi = (-amp + 2 * amp * frac) if a_phase else (amp - 2 * amp * frac)
        hdg[t, 0] = psi
        root[t, 0, 0] = v * DT * t
        c, s = math.cos(psi), math.sin(psi)
        rot = torch.tensor([[c, -s], [s, c]])
        for i, down in ((0, a_phase), (1, not a_phase)):
            here = root[t, 0, :2] + rot @ body[i]
            if down:
                if planted[i] is None:
                    planted[i] = here
                foot[t, 0, i, :2] = planted[i]
                force[t, 0, i] = 300.0
            else:
                planted[i] = None
                foot[t, 0, i, :2] = here
    cmd = torch.zeros(T, 1, 3)
    cmd[..., 0] = v
    vel = torch.zeros(T, 1, 2)
    vel[..., 0] = v
    return force, foot, root, hdg, cmd, vel


def test_heading_wiggle_is_not_paid_as_stroke_on_a_non_turning_slot():
    T = int(20 / DT)
    no_turn = torch.zeros(T, 1, dtype=torch.bool)
    f, fo, ro, h, c, vel = _yaw_wiggle_walk(T, amp=0.0)
    honest = _run_stroke(f, fo, ro, h, c, vel=vel, wz=torch.zeros(T, 1), turning=no_turn)[1].sum().item()
    assert honest > 0
    rot_incl = []
    for amp in (0.05, -0.05):  # both phases of the wiggle relative to the tripods
        f, fo, ro, h, c, vel = _yaw_wiggle_walk(T, amp=amp)
        wiggle = _run_stroke(f, fo, ro, h, c, vel=vel, wz=torch.zeros(T, 1), turning=no_turn)[1].sum().item()
        assert wiggle <= 1.01 * honest
        rot_incl.append(_run_stroke(f, fo, ro, h, c, vel=vel, wz=torch.zeros(T, 1))[1].sum().item())  # rotation counts
    assert max(rot_incl) > 1.05 * honest  # with rotation-inclusive differencing one phase WOULD pay; the change closes it


def test_shipped_power_is_the_sa0_choice():
    assert sm.POWER == 1.5  # SA0 decision, 2026-09-26 (REPLAY_GATE.md); change only with a new replay gate
