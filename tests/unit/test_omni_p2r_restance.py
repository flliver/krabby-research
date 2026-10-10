"""Omni phase-2 stage p2r Z2: the scripted re-stance controller R (``mdp/crab_hex_omni_restance.py``;
``P2R_Z2_PLAN.md`` step 1 + its "Amendment 2026-10-10": CAMS-ONLY, KINEMATIC re-step; CHARTER 12.1), on CPU.

Pinned here (pure torch, the module loaded BY PATH exactly as the CPU tools and the PPO load it):

* the observation contract: cams at columns 18 + 2 i, hip-pitch positions at 29 + i, against a real eval stamp's
  joint order (when one is on this machine); R reads NO load signal (the proxy columns are not read);
* the unit cases: spawn (cams home, hips at default -> the zero statue, every leg HOLD); the up / down thresholds
  (``p_down = down_frac * 0.25 * lift``, ``p_up = up_frac * 0.25 * lift``; a loaded leg sagging to p = -0.018
  still counts as DOWN); the tripod gate (a tripod steps only while the OTHER tripod is all DOWN); the larger
  out-of-band sum wins, an exact tie goes to A; the deadlock case (one tripod all homed, the other with ONE down
  out-of-band leg -> that tripod is selected); HOME needs the other tripod down, else SEAT; the cam laws (homing
  only while UP, the +-trim only while down AND in band, 0 otherwise); LIFT / HOME rods = lift with the knee
  ``-rho * side * lift``, SEAT / HOLD / WAIT rods 0;
* mirror equivariance under both planes (LR: legs [1,0,3,2,5,4], cam -1, hip +1, knee -1; FA: legs [4,5,2,3,0,1],
  cam -1, hip +1, knee +1) on random observations at several constant sets and seeds;
* ``parse_restance`` / ``format_restance`` round trips and refusals (incl. the removed load-sensing constants);
  every output inside [-1, 1]; only the documented columns are read.
NOTE(p2r Z2 cams-only, 2026-10-10): rewritten for the kinematic R (the load-sensing seat-law cases are gone).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
EVAL_STAMPS = REPO / "parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-07_omni_phase2"
LR_STAMPS = REPO / "parkour/outputs/omni_plan/phase2/lr"


def _load_by_path(path: Path, name: str):
    """No ``sys.modules`` registration: the module must load the way the CPU tools load it."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_by_path(RESTANCE_PY, "_t_restance_by_path")
P0 = R.RestanceParams()
W = 1149
LR_PERM = [1, 0, 3, 2, 5, 4]
FA_PERM = [4, 5, 2, 3, 0, 1]
#: (leg permutation, cam sign, hip sign, knee sign) per plane
PLANES = {"lr": (LR_PERM, -1.0, 1.0, -1.0), "fa": (FA_PERM, -1.0, 1.0, 1.0)}
CAM, HIP, KNEE = slice(0, 6), slice(6, 12), slice(12, 18)
HOLD, WAIT, LIFT, HOME, SEAT = range(5)
A, B = list(R.TRIPOD_A), list(R.TRIPOD_B)
#: hip-pitch observations: at default (down), a loaded leg's sag (down), mid-travel (neither), fully lifted (up)
P_DEF, P_SAG = 0.0, -0.018
ROD = R.ROD_SCALE


def _p_mid(p=P0) -> float:
    return 0.5 * (p.p_up + p.p_down)


def _p_air(p=P0) -> float:
    return ROD * p.lift                                            # the lift target itself (well past p_up)


def _obs(n: int = 1, cam=0.0, hip=0.0, gen: torch.Generator | None = None) -> torch.Tensor:
    obs = torch.randn(n, W, generator=gen) if gen is not None else torch.zeros(n, W)
    obs[:, list(R.CAM_OBS_COLS)] = torch.as_tensor(cam, dtype=torch.float32).expand(n, 6)
    obs[:, list(R.HIP_POS_OBS_COLS)] = torch.as_tensor(hip, dtype=torch.float32).expand(n, 6)
    return obs


def _side() -> torch.Tensor:
    return torch.tensor(R.SIDE)


def _legs(**kw) -> torch.Tensor:
    """A per-leg vector from ``FL=..`` style keywords, 0 elsewhere."""
    v = torch.zeros(6)
    for k, x in kw.items():
        v[R.LEGS.index(k)] = x
    return v


def _hips(default=P_DEF, **kw) -> torch.Tensor:
    v = torch.full((6,), float(default))
    for k, x in kw.items():
        v[R.LEGS.index(k)] = x
    return v


def _assert_rods(target: torch.Tensor, lifting: list[bool], p=P0) -> None:
    want_hip = torch.tensor([p.lift if f else 0.0 for f in lifting])
    assert torch.allclose(target[HIP], want_hip, atol=0), (target[HIP], want_hip)
    assert torch.allclose(target[KNEE], -p.rho * _side() * want_hip, atol=1e-7)


# ===================================================================== the contract
def test_constants_and_the_observation_contract():
    assert R.RESTANCE_VAR == "KRABBY_RESTANCE"
    assert R.CAM_OBS_COLS == (18, 20, 22, 24, 26, 28)
    assert R.HIP_POS_OBS_COLS == (29, 30, 31, 32, 33, 34)
    assert R.PROXY_OBS_COLS == (83, 84, 85, 86, 87, 88) and R.MIN_OBS_WIDTH == 89
    assert R.ROD_SCALE == 0.25
    assert R.LEGS == ("FL", "FR", "ML", "MR", "RL", "RR") and R.SIDE == (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    assert R.TRIPOD_A == (0, 3, 4) and R.TRIPOD_B == (1, 2, 5)
    assert R.BRANCH_NAMES == ("hold", "wait", "lift", "home", "seat")
    assert (R.BRANCH_HOLD, R.BRANCH_WAIT, R.BRANCH_LIFT, R.BRANCH_HOME, R.BRANCH_SEAT) == (0, 1, 2, 3, 4)
    assert (R.CAM_ACT, R.HIP_ACT, R.KNEE_ACT, R.ACTION_DIM) == (slice(0, 6), slice(6, 12), slice(12, 18), 18)
    # 17 head columns, then joint_pos - default over the 24 articulation joints (hip pitch = joints 12..17)
    assert R.HIP_POS_OBS_COLS[0] == 17 + 12


def _a_stamp_meta() -> Path | None:
    for root, pat in ((EVAL_STAMPS, "*/suite__omni/seed001/*/run_meta.json"),
                      (LR_STAMPS, "P2R_PROBE_s18/*/seed001/*/run_meta.json")):
        metas = sorted(root.glob(pat))
        if metas:
            return metas[0]
    return None


def test_obs_columns_match_a_real_stamp_joint_order():
    meta = _a_stamp_meta()
    if meta is None:
        pytest.skip("no omni phase-2 stamp on this machine")
    names = json.loads(meta.read_text())["joint_names"]
    assert len(names) == 24
    for i, leg in enumerate(R.LEGS):
        # joint_pos_delta opens at column 17 of the proprio head
        assert names[R.CAM_OBS_COLS[i] - 17] == f"{leg}_Body_CamShaft_RevoluteJoint"
        assert names[R.HIP_POS_OBS_COLS[i] - 17] == f"{leg}_Hip_Femur_RevoluteJoint"


def test_thresholds_follow_the_lift():
    assert P0.p_down == pytest.approx(P0.down_frac * 0.25 * P0.lift)
    assert P0.p_up == pytest.approx(P0.up_frac * 0.25 * P0.lift)
    assert P0.p_up < P0.p_down < 0.0                               # up is further along the (negative) lift
    p = R.RestanceParams(lift=-0.8, up_frac=0.7, down_frac=0.2)
    assert (p.p_up, p.p_down) == pytest.approx((0.7 * 0.25 * -0.8, 0.2 * 0.25 * -0.8))


# ===================================================================== unit cases
def test_spawn_is_the_zero_statue():
    """Spawn: cams at home and every hip at its default -> every leg HOLD, every output exactly 0."""
    target, branch = R.restance_target(_obs(3), P0)
    assert torch.equal(branch, torch.full((3, 6), HOLD))
    assert torch.equal(target, torch.zeros(3, 18))


def test_up_and_down_thresholds_incl_a_loaded_sag():
    """A leg is DOWN at p >= p_down (a loaded leg sagging to -0.018 is down), UP at p <= p_up, neither between."""
    assert P0.p_down < P_SAG < 0.0
    for p_hip, want in ((P_DEF, HOLD), (P_SAG, HOLD), (P0.p_down, HOLD), (0.05, HOLD),
                        (P0.p_down - 1e-4, SEAT), (_p_mid(), SEAT), (P0.p_up, SEAT), (_p_air(), SEAT)):
        # one leg (FL, homed cam) at p_hip, the rest at default: down -> HOLD; not down and homed -> SEAT
        _, branch = R.restance_target(_obs(1, hip=_hips(FL=p_hip)), P0)
        assert int(branch[0, 0]) == want, (p_hip, branch)
    # the cam homing law fires only from p_up on (UP), never in the gap between the two thresholds
    cam = _legs(FL=0.04)                                            # inside eps_air: SEAT, homing while UP
    for p_hip, homing in ((P0.p_up, True), (_p_air(), True), (_p_mid(), False), (P0.p_down - 1e-4, False)):
        target, branch = R.restance_target(_obs(1, cam=cam, hip=_hips(FL=p_hip)), P0)
        assert int(branch[0, 0]) == SEAT
        assert float(target[0, 0]) == pytest.approx(-P0.k_c * 0.04 if homing else 0.0, abs=1e-7), p_hip
    # a loaded sag inside the band gets the trim, as at default
    t_def, _ = R.restance_target(_obs(1, cam=cam, hip=_hips()), P0)
    t_sag, _ = R.restance_target(_obs(1, cam=cam, hip=_hips(FL=P_SAG)), P0)
    assert float(t_def[0, 0]) == float(t_sag[0, 0]) == pytest.approx(-P0.k_h * 0.04)


def test_tripod_gate_needs_the_other_tripod_all_down():
    cam = _legs(FL=0.3)                                            # FL (tripod A) out of band, everyone down
    target, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert branch[0].tolist() == [LIFT, HOLD, HOLD, HOLD, HOLD, HOLD]
    _assert_rods(target[0], [True, False, False, False, False, False])
    assert float(target[0, 0]) == 0.0                              # a lifting (still down) leg's cam waits
    # one B leg off the ground (ML mid-travel): tripod A may not step -> FL WAITs (rods 0, cam 0)
    for p_ml in (_p_mid(), _p_air()):
        target, branch = R.restance_target(_obs(1, cam=cam, hip=_hips(ML=p_ml)), P0)
        assert int(branch[0, 0]) == WAIT, p_ml
        assert float(target[0, 0]) == 0.0 and float(target[0, HIP][0]) == 0.0 and float(target[0, KNEE][0]) == 0.0
    # a B leg merely sagging under load is still down: FL steps
    _, branch = R.restance_target(_obs(1, cam=cam, hip=_hips(FR=P_SAG, ML=P_SAG, RR=P_SAG)), P0)
    assert int(branch[0, 0]) == LIFT
    # the same gate for tripod B (FR, ML, RR) against an A leg off the ground
    cam_b = _legs(RR=-0.4)
    _, branch = R.restance_target(_obs(1, cam=cam_b), P0)
    assert branch[0].tolist() == [HOLD, HOLD, HOLD, HOLD, HOLD, LIFT]
    _, branch = R.restance_target(_obs(1, cam=cam_b, hip=_hips(RL=_p_mid())), P0)
    assert int(branch[0, 5]) == WAIT


def test_tripod_gate_larger_sum_wins_and_a_tie_goes_to_a():
    cam = _legs(FL=0.2, FR=0.5, MR=0.2)                            # A: 0.2 + 0.2 = 0.4; B: 0.5
    _, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert branch[0].tolist() == [WAIT, LIFT, HOLD, WAIT, HOLD, HOLD]
    cam = _legs(FL=0.3, FR=0.5, MR=-0.3)                           # A 0.6 > B 0.5 (|c| sums)
    _, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert branch[0].tolist() == [LIFT, WAIT, HOLD, LIFT, HOLD, HOLD]
    cam = _legs(FL=0.25, FR=0.5, MR=0.25)                          # exact tie -> A
    _, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert branch[0].tolist() == [LIFT, WAIT, HOLD, LIFT, HOLD, HOLD]
    # in-band legs of either tripod never count toward the sum (only down, out-of-band legs want to step)
    cam = _legs(FL=0.2, FR=0.15, ML=0.1, RR=0.1)                    # B's in-band 0.1 + 0.1 do not lift 0.15 past 0.2
    _, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert branch[0].tolist() == [LIFT, WAIT, HOLD, HOLD, HOLD, HOLD]


def test_the_deadlock_case_selects_the_tripod_with_the_out_of_band_leg():
    """One tripod all homed, the other with ONE down out-of-band leg: that tripod steps (the homed one has nothing to
    step for, so it must not block the other), in both orientations, also with the homed tripod sagging."""
    for leg in ("RR", "MR", "FL", "ML"):
        cam = _legs(**{leg: -0.25})
        for hips in (_hips(), _hips(default=P_SAG)):
            target, branch = R.restance_target(_obs(1, cam=cam, hip=hips), P0)
            want = [HOLD] * 6
            want[R.LEGS.index(leg)] = LIFT
            assert branch[0].tolist() == want, (leg, hips)
            lifting = [i == R.LEGS.index(leg) for i in range(6)]
            _assert_rods(target[0], lifting)


def test_home_needs_the_other_tripod_down_else_seat():
    # tripod A in the air (lifted), out of the air band; tripod B all down: A's legs HOME (rods lift, cams home)
    cam = _legs(FL=0.8, MR=-2.0, RL=0.2)
    hips = _hips(FL=_p_air(), MR=_p_air(), RL=_p_air())
    target, branch = R.restance_target(_obs(1, cam=cam, hip=hips), P0)
    assert branch[0].tolist() == [HOME, HOLD, HOLD, HOME, HOME, HOLD]
    _assert_rods(target[0], [i in A for i in range(6)])
    assert torch.allclose(target[0, CAM][A], -(P0.k_c * cam[A]).clamp(-1, 1))
    assert float(target[0, 3]) == 1.0 and float(target[0, 0]) == -1.0   # saturated at the clip
    assert bool((target[0, HIP][A] < 0).all())                          # lift = negative hip pitch
    # mid-travel (not yet UP): still HOME (rods keep lifting) but the cam waits until UP
    target, branch = R.restance_target(_obs(1, cam=cam, hip=_hips(FL=_p_mid(), MR=_p_mid(), RL=_p_mid())), P0)
    assert branch[0].tolist() == [HOME, HOLD, HOLD, HOME, HOME, HOLD]
    assert torch.equal(target[0, CAM][A], torch.zeros(3))
    _assert_rods(target[0], [i in A for i in range(6)])
    # once inside the air band the leg SEATs (rods 0) and keeps homing while UP
    cam2 = cam.clone()
    cam2[0] = 0.049
    target, branch = R.restance_target(_obs(1, cam=cam2, hip=hips), P0)
    assert branch[0].tolist() == [SEAT, HOLD, HOLD, HOME, HOME, HOLD]
    assert float(target[0, HIP][0]) == 0.0 and float(target[0, 0]) == pytest.approx(-P0.k_c * 0.049)
    # the support rule: a B leg that left the ground while A is up -> nobody may lift; every non-down leg SEATs
    hips_b = hips.clone()
    hips_b[1] = _p_mid()                                                # FR (B) not down
    target, branch = R.restance_target(_obs(1, cam=_legs(FL=0.8, FR=0.6, MR=-2.0, RL=0.2), hip=hips_b), P0)
    assert branch[0].tolist() == [SEAT, SEAT, HOLD, SEAT, SEAT, HOLD]
    _assert_rods(target[0], [False] * 6)
    # an out-of-band leg off the ground with BOTH tripods partly up (e.g. caught mid-stride at the stand bit): SEAT
    _, branch = R.restance_target(_obs(1, cam=torch.full((6,), 0.8), hip=torch.full((6,), _p_mid())), P0)
    assert torch.equal(branch, torch.full((1, 6), SEAT))


def test_cam_laws():
    """Homing only while UP, the +-trim only while down AND inside eps_c, 0 otherwise -- branch by branch."""
    # HOLD: down, in band -> -clip(k_h c, +-trim); eps_c inclusive
    cam = _legs(FL=0.09, FR=-0.09, MR=0.02, RL=-0.1, RR=0.1)
    target, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert torch.equal(branch[0], torch.full((6,), HOLD))
    assert torch.allclose(target[0, CAM], -(P0.k_h * cam).clamp(-P0.trim, P0.trim))
    assert float(target[0, CAM].abs().max()) <= P0.trim + 1e-9
    p = R.RestanceParams(k_h=1.0, trim=0.03)                           # the authority clip binds
    t, _ = R.restance_target(_obs(1, cam=cam), p)
    assert float(t[0, 0]) == pytest.approx(-0.03) and float(t[0, 1]) == pytest.approx(0.03)
    # WAIT and LIFT (down, out of band): cam 0
    cam = _legs(FL=0.5, FR=0.6)                                        # B selected (0.6 > 0.5): FL WAITs, FR LIFTs
    target, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert branch[0].tolist()[:2] == [WAIT, LIFT]
    assert float(target[0, 0]) == 0.0 and float(target[0, 1]) == 0.0
    # HOME / SEAT while UP: -clip(k_c c, +-1); while not yet UP: 0
    for hip, want in ((_p_air(), lambda c: -(P0.k_c * c).clamp(-1, 1)), (_p_mid(), lambda c: torch.zeros_like(c))):
        cam = _legs(FL=0.3, MR=0.02)                                   # FL HOME, MR SEAT (inside eps_air)
        target, branch = R.restance_target(_obs(1, cam=cam, hip=_hips(FL=hip, MR=hip)), P0)
        assert branch[0].tolist()[0] == HOME and branch[0].tolist()[3] == SEAT
        assert torch.allclose(target[0, CAM][[0, 3]], want(cam[[0, 3]]))
    # the cam laws are odd in c
    cam = torch.tensor([0.3, -0.2, 0.05, -0.07, 0.6, -0.01])
    hips = _hips(FL=_p_air(), MR=_p_air(), RL=_p_air())
    t_pos, _ = R.restance_target(_obs(1, cam=cam, hip=hips), P0)
    t_neg, _ = R.restance_target(_obs(1, cam=-cam, hip=hips), P0)
    assert torch.allclose(t_neg[0, CAM], -t_pos[0, CAM]) and torch.equal(t_neg[0, 6:], t_pos[0, 6:])


def test_rods_per_branch():
    """LIFT / HOME: hip = lift, knee = -rho * side * lift; SEAT / HOLD / WAIT: both rods 0 -- at several constants."""
    for p in (P0, R.RestanceParams(lift=-0.4, rho=2.0), R.RestanceParams(lift=-0.9, rho=0.0)):
        cam = _legs(FL=0.8, FR=0.3, MR=0.05, RL=0.4)                    # tripod A up; FR (B) out of band, down
        hips = _hips(FL=_p_air(p), MR=_p_air(p), RL=_p_air(p))
        target, branch = R.restance_target(_obs(1, cam=cam, hip=hips), p)
        # FL / RL HOME (out of the air band), MR SEAT (homed in the air), FR WAIT (A is up), ML / RR HOLD
        assert branch[0].tolist() == [HOME, WAIT, HOLD, SEAT, HOME, HOLD]
        _assert_rods(target[0], [True, False, False, False, True, False], p)
        assert float(target[0, KNEE][0]) == pytest.approx(-p.rho * p.lift)          # left leg: knee = -rho * lift
        # LIFT
        target, branch = R.restance_target(_obs(1, cam=_legs(FR=0.3)), p)
        assert int(branch[0, 1]) == LIFT
        _assert_rods(target[0], [False, True, False, False, False, False], p)
        assert float(target[0, KNEE][1]) == pytest.approx(p.rho * p.lift)            # right leg: knee = +rho * lift


# ===================================================================== mirror equivariance
def _mirror_obs(obs: torch.Tensor, plane: str) -> torch.Tensor:
    perm, s_cam, s_hip, _s_knee = PLANES[plane]
    out = obs.clone()
    cam = obs[:, list(R.CAM_OBS_COLS)]
    hp = obs[:, list(R.HIP_POS_OBS_COLS)]
    out[:, list(R.CAM_OBS_COLS)] = s_cam * cam[:, perm]
    out[:, list(R.HIP_POS_OBS_COLS)] = s_hip * hp[:, perm]
    return out


def _mirror_action(a: torch.Tensor, plane: str) -> torch.Tensor:
    perm, s_cam, s_hip, s_knee = PLANES[plane]
    return torch.cat([s_cam * a[:, CAM][:, perm], s_hip * a[:, HIP][:, perm], s_knee * a[:, KNEE][:, perm]], -1)


def _random_obs(n: int, seed: int, p=P0) -> torch.Tensor:
    """Random observations straddling every band: cams around eps_air / eps_c, hips around p_up / p_down, with a
    bias toward whole tripods down (so the selection and the deadlock case are exercised, not only SEAT)."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, W, generator=g)
    obs[:, list(R.CAM_OBS_COLS)] = (torch.rand(n, 6, generator=g) - 0.5) * 0.8
    span = 1.6 * ROD * p.lift                                      # from +0.3 x |lift travel| above default to past it
    hp = torch.rand(n, 6, generator=g) * span - 0.3 * span
    down_row = torch.rand(n, 6, generator=g) < 0.6                 # most legs pinned near default (down)
    hp = torch.where(down_row, (torch.rand(n, 6, generator=g) - 0.5) * 0.02, hp)
    obs[:, list(R.HIP_POS_OBS_COLS)] = hp
    return obs


PARAM_SETS = [P0, R.RestanceParams(lift=-0.7, rho=0.8, up_frac=0.7, down_frac=0.3, k_c=4.0, k_h=0.5, trim=0.05,
                                   eps_c=0.15, eps_air=0.03),
              R.RestanceParams(lift=-0.4, rho=2.0)]


@pytest.mark.parametrize("plane", ["lr", "fa"])
@pytest.mark.parametrize("pi", range(len(PARAM_SETS)))
@pytest.mark.parametrize("seed", [17, 29])
def test_mirror_equivariance_on_random_inputs(plane, pi, seed):
    p = PARAM_SETS[pi]
    obs = _random_obs(4000, seed=seed + pi, p=p)
    target, branch = R.restance_target(obs, p)
    m_target, m_branch = R.restance_target(_mirror_obs(obs, plane), p)
    perm = PLANES[plane][0]
    assert torch.equal(m_branch, branch[:, perm])
    assert torch.allclose(m_target, _mirror_action(target, plane), atol=1e-6)
    # the draw really exercised every branch (a vacuous test would pass with HOLD everywhere)
    shares = R.branch_shares(branch)
    assert all(v > 0.01 for v in shares.values()), shares
    # the mirror is an involution on the inputs, so the check also holds back
    assert torch.equal(_mirror_obs(_mirror_obs(obs, plane), plane), obs)


@pytest.mark.parametrize("plane", ["lr", "fa"])
def test_mirror_equivariance_on_the_unit_cases(plane):
    """The named unit-case observations (spawn, the gate, the sums, the deadlock, homing, support) mirrored."""
    air, mid = _p_air(), _p_mid()
    cases = [
        _obs(1),
        _obs(1, cam=_legs(FL=0.3)),
        _obs(1, cam=_legs(FL=0.3, FR=0.6, MR=0.2)),                # not a tie (a tie breaks LR: -> A)
        _obs(1, cam=_legs(RR=-0.25), hip=_hips(default=P_SAG)),
        _obs(1, cam=_legs(FL=0.8, MR=-2.0, RL=0.049), hip=_hips(FL=air, MR=air, RL=air)),
        _obs(1, cam=_legs(FL=0.8, FR=0.6, MR=-2.0), hip=_hips(FL=air, FR=mid, MR=air, RL=air)),
        _obs(1, cam=_legs(FL=0.09, FR=-0.09, MR=0.02, RL=-0.1, RR=0.1)),
    ]
    perm = PLANES[plane][0]
    for obs in cases:
        target, branch = R.restance_target(obs, P0)
        m_target, m_branch = R.restance_target(_mirror_obs(obs, plane), P0)
        assert torch.equal(m_branch, branch[:, perm])
        assert torch.allclose(m_target, _mirror_action(target, plane), atol=1e-7)


def test_lr_maps_tripod_a_to_b_and_fa_keeps_them():
    assert sorted(LR_PERM[i] for i in R.TRIPOD_A) == sorted(R.TRIPOD_B)
    assert sorted(FA_PERM[i] for i in R.TRIPOD_A) == sorted(R.TRIPOD_A)
    side = torch.tensor(R.SIDE)
    assert torch.equal(side[LR_PERM], -side) and torch.equal(side[FA_PERM], side)


# ===================================================================== bounds, columns read
def test_outputs_stay_inside_the_clip():
    for p in PARAM_SETS + [R.RestanceParams(lift=-1.0, rho=1.0, k_c=50.0, k_h=50.0, trim=1.0)]:
        obs = _random_obs(2000, seed=3, p=p)
        obs[:, list(R.CAM_OBS_COLS)] *= 20.0                        # far outside +-pi as well
        target, branch = R.restance_target(obs, p)
        assert target.shape == (2000, 18) and branch.shape == (2000, 6) and branch.dtype == torch.long
        assert float(target.abs().max()) <= 1.0
        assert bool(torch.isfinite(target).all())


def test_only_the_documented_columns_are_read():
    obs = _random_obs(64, seed=5)
    target, branch = R.restance_target(obs, P0)
    other = obs.clone()
    keep = list(R.CAM_OBS_COLS) + list(R.HIP_POS_OBS_COLS)
    mask = torch.ones(W, dtype=torch.bool)
    mask[keep] = False
    other[:, mask] += 7.0                                          # incl. the proxy columns: R reads no load signal
    t2, b2 = R.restance_target(other, P0)
    assert torch.equal(t2, target) and torch.equal(b2, branch)
    # narrower than 89 columns is refused; exactly 89 is enough
    with pytest.raises(ValueError, match="89"):
        R.restance_target(obs[:, :88], P0)
    with pytest.raises(ValueError, match="89"):
        R.restance_target(obs[0], P0)
    t3, _ = R.restance_target(obs[:, :89], P0)
    assert torch.equal(t3, target)


def test_branch_shares():
    b = torch.tensor([[0, 0, 1, 2, 3, 4], [4, 4, 4, 4, 4, 4]])
    s = R.branch_shares(b)
    assert s == {"hold": 2 / 12, "wait": 1 / 12, "lift": 1 / 12, "home": 1 / 12, "seat": 7 / 12}
    assert R.branch_shares(torch.zeros(0, 6, dtype=torch.long)) == {n: 0.0 for n in R.BRANCH_NAMES}


# ===================================================================== parse / format
def test_parse_format_round_trip():
    for off in (None, "", "  ", "0"):
        assert R.parse_restance(off) is None
    assert R.parse_restance("default") == P0 and R.parse_restance(" 1 ") == P0
    p = R.parse_restance("lift=-0.6, rho=0.4,up_frac=0.7")
    assert (p.lift, p.rho, p.up_frac) == (-0.6, 0.4, 0.7) and p.down_frac == P0.down_frac
    for q in PARAM_SETS + [p]:
        text = R.format_restance(q)
        assert " " not in text
        assert R.parse_restance(text) == q
        assert R.RestanceParams.from_dict(q.to_dict()) == q
        assert all(isinstance(v, float) for v in q.to_dict().values())
    assert set(P0.to_dict()) == {"lift", "rho", "up_frac", "down_frac", "k_c", "k_h", "trim", "eps_c", "eps_air"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        P0.lift = -0.9                                              # frozen


@pytest.mark.parametrize("text,match", [
    ("lift=0.3", "lift"), ("lift=0", "lift"), ("lift=-1.5", "lift"),
    ("up_frac=0.3", "down_frac"), ("down_frac=0", "down_frac"), ("up_frac=1.0", "up_frac"),
    ("down_frac=0.6,up_frac=0.6", "down_frac"), ("k_c=0", "k_c"), ("k_h=-1", "k_h"), ("trim=0", "trim"),
    ("eps_c=0", "eps_c"), ("eps_air=0.2", "eps_air"), ("rho=-0.1", "rho"), ("lift=-0.8,rho=1.5", "clip"),
    ("trim=1.5", "trim"), ("bogus=1", "unknown"), ("lift=-0.5,lift=-0.6", "twice"), ("lift", "k=v"),
    ("lift=abc", "not a number"), ("lift=nan", "finite"), ("k_c=inf", "finite"),
    # the load-sensing constants are gone with the 2026-10-10 amendment
    ("theta_L=0.6", "unknown"), ("theta_S=0.9", "unknown"), ("kappa=0.25", "unknown"), ("dmax=0.6", "unknown"),
])
def test_parse_refusals(text, match):
    with pytest.raises(ValueError, match=match):
        R.parse_restance(text)


def test_from_dict_refuses_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        R.RestanceParams.from_dict({**P0.to_dict(), "gain": 1.0})
    assert math.isclose(R.RestanceParams.from_dict({"lift": "-0.4"}).lift, -0.4)
