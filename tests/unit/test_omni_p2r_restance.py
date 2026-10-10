"""Omni phase-2 stage p2r Z2: the scripted re-stance controller R (``mdp/crab_hex_omni_restance.py``;
``P2R_Z2_PLAN.md`` step 1, CHARTER 12), on CPU.

Pinned here (pure torch, the module loaded BY PATH exactly as the CPU tools and the PPO load it):

* the observation contract: cams at columns 18 + 2 i, the proxy at 83 + i, against the omni obs term's own
  ``STAND_PROXY_COL`` and a real eval stamp's joint order (when one is on this machine);
* the unit cases the plan names: spawn (every proxy -0.5 "unknown", cams home -> all six SEAT with the seat
  extension), mid-air homing, the tripod gate (a tripod steps only while the other one is fully loaded; the larger
  out-of-band sum wins; the deadlock case: one tripod all homed, the other with one loaded out-of-band leg -> that
  tripod is selected), the seat law's equilibrium below theta_S under a standing extension;
* mirror equivariance under both planes (LR: legs [1,0,3,2,5,4], cam -1, hip +1, knee -1; FA: legs
  [4,5,2,3,0,1], cam -1, hip +1, knee +1) on random observations, at the seed constants and at other constants;
* ``parse_restance`` / ``format_restance`` round trips and refusals; every output inside [-1, 1]; only the
  documented columns are read.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
OMNI_OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
EVAL_STAMPS = REPO / "parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-07_omni_phase2"


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


def _obs(n: int = 1, cam=0.0, h=-0.5, gen: torch.Generator | None = None) -> torch.Tensor:
    obs = torch.randn(n, W, generator=gen) if gen is not None else torch.zeros(n, W)
    obs[:, list(R.CAM_OBS_COLS)] = torch.as_tensor(cam, dtype=torch.float32).expand(n, 6)
    obs[:, list(R.PROXY_OBS_COLS)] = torch.as_tensor(h, dtype=torch.float32).expand(n, 6)
    return obs


def _side() -> torch.Tensor:
    return torch.tensor(R.SIDE)


def _seat(h: torch.Tensor, p=P0) -> torch.Tensor:
    return p.kappa * (p.theta_S - h).clamp(0.0, p.dmax)


# ===================================================================== the contract
def test_constants_and_the_observation_contract():
    assert R.RESTANCE_VAR == "KRABBY_RESTANCE"
    assert R.CAM_OBS_COLS == (18, 20, 22, 24, 26, 28)
    assert R.PROXY_OBS_COLS == (83, 84, 85, 86, 87, 88) and R.MIN_OBS_WIDTH == 89
    assert R.LEGS == ("FL", "FR", "ML", "MR", "RL", "RR") and R.SIDE == (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    assert R.TRIPOD_A == (0, 3, 4) and R.TRIPOD_B == (1, 2, 5)
    assert R.BRANCH_NAMES == ("hold", "wait", "lift", "home", "seat")
    # the omni obs term writes its held proxy at the same first column (read from source: no Isaac import)
    assert "    STAND_PROXY_COL = 83\n" in OMNI_OBS_PY.read_text()
    # 17 head columns + 24 joint positions + 24 joint velocities + 18 actions -> the proxy at 83
    assert 17 + 2 * 24 + 18 == R.PROXY_OBS_COLS[0]


def test_cam_columns_match_a_real_stamp_joint_order():
    metas = sorted(EVAL_STAMPS.glob("*/suite__omni/seed001/*/run_meta.json"))
    if not metas:
        pytest.skip("no omni phase-2 suite stamp on this machine")
    names = json.loads(metas[0].read_text())["joint_names"]
    assert len(names) == 24
    for i, leg in enumerate(R.LEGS):
        # joint_pos_delta opens at column 17 of the proprio head
        assert names[R.CAM_OBS_COLS[i] - 17] == f"{leg}_Body_CamShaft_RevoluteJoint"


# ===================================================================== unit cases
def test_spawn_seats_all_six_with_the_seat_extension():
    target, branch = R.restance_target(_obs(3), P0)
    assert torch.equal(branch, torch.full((3, 6), SEAT))
    want_hip = P0.kappa * min(P0.theta_S + 0.5, P0.dmax)        # h = -0.5 -> the clamped extension
    assert want_hip > 0.0
    assert torch.allclose(target[:, HIP], torch.full((3, 6), want_hip))
    assert torch.allclose(target[:, KNEE], -P0.rho * _side() * want_hip)
    assert torch.equal(target[:, CAM].abs(), torch.zeros(3, 6))   # cams at home: no cam command
    # spawn with cams inside the air band but off zero: SEAT, and the air homing law still trims them
    cam = torch.tensor([0.04, -0.04, 0.0, 0.03, -0.01, 0.02])
    target, branch = R.restance_target(_obs(1, cam=cam), P0)
    assert torch.equal(branch, torch.full((1, 6), SEAT))
    assert torch.allclose(target[0, CAM], -(P0.k_c * cam).clamp(-1, 1))


def test_mid_air_homing():
    # tripod A (FL, MR, RL) in the air, tripod B (FR, ML, RR) planted and loaded: A's legs HOME
    cam = torch.tensor([0.8, 0.0, 0.0, -2.0, 0.2, 0.0])
    h = torch.tensor([-0.5, 0.6, 0.6, 0.2, 0.34, 0.6])            # A below theta_L, B loaded
    target, branch = R.restance_target(_obs(1, cam=cam, h=h), P0)
    assert branch[0].tolist() == [HOME, HOLD, HOLD, HOME, HOME, HOLD]
    a = [0, 3, 4]
    assert torch.allclose(target[0, CAM][a], -(P0.k_c * cam[a]).clamp(-1, 1))
    assert target[0, CAM][3] == 1.0 and target[0, CAM][0] == -1.0  # saturated at the clip
    assert torch.allclose(target[0, HIP][a], torch.full((3,), P0.lift))
    assert torch.allclose(target[0, KNEE][a], -P0.rho * _side()[a] * P0.lift)
    assert bool((target[0, HIP][a] < 0).all())                     # lift = negative hip pitch
    # once inside the air band the leg seats (and keeps homing its cam)
    cam2 = cam.clone()
    cam2[0] = 0.049
    _, branch2 = R.restance_target(_obs(1, cam=cam2, h=h), P0)
    assert branch2[0].tolist() == [SEAT, HOLD, HOLD, HOME, HOME, HOLD]


def test_support_rule_an_unloaded_leg_without_a_loaded_other_tripod_seats():
    """NOTE(p2r Z2 probe, 2026-10-10): HOME needs the OTHER tripod fully loaded (the P2R_RSTt1_s18 pitch-overs:
    a planted-tripod leg that lost load lifted while the stepping tripod was in the air)."""
    cam = torch.tensor([0.8, 1.2, 0.0, -0.5, 0.3, 0.0])
    h = torch.tensor([-0.5, 0.0, 0.6, -0.5, -0.5, 0.6])            # A in the air; FR (B) lost its load
    target, branch = R.restance_target(_obs(1, cam=cam, h=h), P0)
    # FR's other tripod (A) is not loaded -> FR seats (presses down), never lifts; A's legs' other tripod (B) is not
    # fully loaded either (FR) -> they seat too, so every unloaded leg presses down until support is back
    assert branch[0].tolist() == [SEAT, SEAT, HOLD, SEAT, SEAT, HOLD]
    assert bool((target[0, HIP][[0, 1, 3, 4]] >= 0).all())
    # every unloaded leg everywhere (spawn-like but out of band): all seat
    _, branch = R.restance_target(_obs(1, cam=torch.full((6,), 0.8), h=-0.5), P0)
    assert torch.equal(branch, torch.full((1, 6), SEAT))


def test_tripod_gate_needs_the_other_tripod_fully_loaded():
    loaded = 0.6
    cam = torch.zeros(6)
    cam[0] = 0.3                                                   # FL (tripod A) out of band
    target, branch = R.restance_target(_obs(1, cam=cam, h=loaded), P0)
    assert branch[0].tolist() == [LIFT, HOLD, HOLD, HOLD, HOLD, HOLD]
    assert target[0, CAM][0] == 0.0                                # a lifting leg homes only once in the air
    assert target[0, HIP][0] == pytest.approx(P0.lift)
    # a B leg in the air: tripod A may not step -> its out-of-band leg WAITs, and the B leg homes
    h = torch.full((6,), loaded)
    h[2] = -0.5                                                    # ML (tripod B) unloaded
    cam[2] = 0.4
    target, branch = R.restance_target(_obs(1, cam=cam, h=h), P0)
    assert branch[0].tolist() == [WAIT, HOLD, HOME, HOLD, HOLD, HOLD]
    assert target[0, CAM][0] == 0.0 and target[0, HIP][0] == pytest.approx(float(_seat(torch.tensor(loaded))))


def test_tripod_gate_larger_sum_wins_and_a_tie_goes_to_a():
    loaded = torch.full((6,), 0.6)
    cam = torch.tensor([0.2, 0.5, 0.0, 0.2, 0.0, 0.0])            # A: 0.2 + 0.2 = 0.4; B: 0.5
    _, branch = R.restance_target(_obs(1, cam=cam, h=loaded), P0)
    assert branch[0].tolist() == [WAIT, LIFT, HOLD, WAIT, HOLD, HOLD]
    cam = torch.tensor([0.3, 0.5, 0.0, 0.3, 0.0, 0.0])            # A 0.6 > B 0.5
    _, branch = R.restance_target(_obs(1, cam=cam, h=loaded), P0)
    assert branch[0].tolist() == [LIFT, WAIT, HOLD, LIFT, HOLD, HOLD]
    cam = torch.tensor([0.25, 0.5, 0.0, 0.25, 0.0, 0.0])          # exact tie -> A
    _, branch = R.restance_target(_obs(1, cam=cam, h=loaded), P0)
    assert branch[0].tolist() == [LIFT, WAIT, HOLD, LIFT, HOLD, HOLD]


def test_the_deadlock_case_selects_the_tripod_with_the_out_of_band_leg():
    """One tripod all homed (and loaded), the other with ONE loaded out-of-band leg: that tripod steps (the homed
    one has nothing to step for, so it must not block the other), in both orientations."""
    h = torch.full((6,), 0.6)
    for leg, tripod in ((5, R.TRIPOD_B), (3, R.TRIPOD_A)):
        cam = torch.zeros(6)
        cam[leg] = -0.25
        target, branch = R.restance_target(_obs(1, cam=cam, h=h), P0)
        want = [HOLD] * 6
        want[leg] = LIFT
        assert branch[0].tolist() == want, (leg, tripod)
        assert target[0, HIP][leg] == pytest.approx(P0.lift)
    # ... and with one leg of the homed tripod a little light but still loaded (h >= theta_L), the same
    h2 = h.clone()
    h2[0] = P0.theta_L
    cam = torch.zeros(6)
    cam[5] = -0.25
    _, branch = R.restance_target(_obs(1, cam=cam, h=h2), P0)
    assert branch[0, 5] == LIFT


def test_loaded_cam_trim_inside_the_band():
    cam = torch.tensor([0.09, -0.09, 0.0, 0.02, -0.1, 0.1])        # all within eps_c (0.10 inclusive)
    target, branch = R.restance_target(_obs(1, cam=cam, h=0.6), P0)
    assert torch.equal(branch[0], torch.full((6,), HOLD))
    want = -(P0.k_h * cam).clamp(-P0.trim, P0.trim)
    assert torch.allclose(target[0, CAM], want)
    assert float(target[0, CAM].abs().max()) <= P0.trim + 1e-9


def test_the_seat_law_equilibrium_sits_below_theta_s_with_a_standing_extension():
    """The seat law is proportional: the hip command is kappa * (theta_S - h) for theta_L <= h < theta_S and 0
    at/above theta_S. A leg that needs a standing extension u_s > 0 to stay pressed (compliance, the gate) therefore
    settles where the command equals u_s -- at h* = theta_S - u_s / kappa, strictly BELOW theta_S."""
    h = torch.linspace(P0.theta_L, 1.0, 651)
    obs = _obs(h.numel(), cam=0.0, h=h[:, None].expand(-1, 6))
    target, branch = R.restance_target(obs, P0)
    hip = target[:, HIP][:, 0]
    assert bool((branch == HOLD).all())
    assert torch.equal(hip[h >= P0.theta_S], torch.zeros(int((h >= P0.theta_S).sum())))
    assert bool((hip[h < P0.theta_S] > 0).all())
    assert bool((hip[1:] <= hip[:-1] + 1e-7).all())               # non-increasing in the load
    for u_s in (0.02, 0.05, 0.1):
        h_star = P0.theta_S - u_s / P0.kappa
        assert P0.theta_L < h_star < P0.theta_S
        t, _ = R.restance_target(_obs(1, cam=0.0, h=h_star), P0)
        assert float(t[0, HIP][0]) == pytest.approx(u_s, abs=1e-6)
        # lighter than h* -> more extension than u_s (the leg presses on); heavier -> less (it settles back)
        lo, _ = R.restance_target(_obs(1, cam=0.0, h=h_star - 0.01), P0)
        hi, _ = R.restance_target(_obs(1, cam=0.0, h=h_star + 0.01), P0)
        assert float(lo[0, HIP][0]) > u_s > float(hi[0, HIP][0])
    # the knee moves with the hip in every press, with the per-side sign
    assert torch.allclose(target[:, KNEE], -P0.rho * _side()[None, :] * target[:, HIP])


# ===================================================================== mirror equivariance
def _mirror_obs(obs: torch.Tensor, plane: str) -> torch.Tensor:
    perm, s_cam, _s_hip, _s_knee = PLANES[plane]
    out = obs.clone()
    cam = obs[:, list(R.CAM_OBS_COLS)]
    h = obs[:, list(R.PROXY_OBS_COLS)]
    out[:, list(R.CAM_OBS_COLS)] = s_cam * cam[:, perm]
    out[:, list(R.PROXY_OBS_COLS)] = h[:, perm]
    return out


def _mirror_action(a: torch.Tensor, plane: str) -> torch.Tensor:
    perm, s_cam, s_hip, s_knee = PLANES[plane]
    return torch.cat([s_cam * a[:, CAM][:, perm], s_hip * a[:, HIP][:, perm], s_knee * a[:, KNEE][:, perm]], -1)


def _random_obs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, W, generator=g)
    # straddle every band: cams around eps_air / eps_c, proxies around theta_L / theta_S
    obs[:, list(R.CAM_OBS_COLS)] = (torch.rand(n, 6, generator=g) - 0.5) * 0.8
    obs[:, list(R.PROXY_OBS_COLS)] = torch.rand(n, 6, generator=g) * 1.5 - 0.5
    return obs


PARAM_SETS = [P0, R.RestanceParams(lift=-0.7, rho=0.8, theta_L=0.1, theta_S=0.6, kappa=1.2, dmax=0.5, k_c=4.0,
                                   k_h=0.5, trim=0.05, eps_c=0.15, eps_air=0.03)]


@pytest.mark.parametrize("plane", ["lr", "fa"])
@pytest.mark.parametrize("pi", range(len(PARAM_SETS)))
def test_mirror_equivariance_on_random_inputs(plane, pi):
    p = PARAM_SETS[pi]
    obs = _random_obs(4000, seed=17 + pi)
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
def test_mirror_equivariance_on_the_seed_constants_unit_cases(plane):
    """The named unit-case observations (spawn, homing, the gate, the deadlock) mirrored, at the seed constants."""
    cases = [
        _obs(1),
        _obs(1, cam=torch.tensor([0.8, -0.3, 0.06, -2.0, 0.2, -0.051]), h=-0.5),
        _obs(1, cam=torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]), h=0.6),
        _obs(1, cam=torch.tensor([0.2, 0.5, 0.0, 0.2, 0.0, 0.0]), h=0.6),
        _obs(1, cam=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.25]), h=0.6),
        _obs(1, cam=torch.tensor([0.09, -0.09, 0.0, 0.02, -0.1, 0.1]), h=torch.tensor([0.6, 0.4, 0.8, 0.36, 0.6, 1.0])),
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
    for p in PARAM_SETS + [R.RestanceParams(lift=-1.0, rho=1.0, kappa=1.0, dmax=1.0, k_c=50.0, trim=1.0)]:
        obs = _random_obs(2000, seed=3)
        obs[:, list(R.CAM_OBS_COLS)] *= 20.0                        # far outside +-pi as well
        target, branch = R.restance_target(obs, p)
        assert target.shape == (2000, 18) and branch.shape == (2000, 6) and branch.dtype == torch.long
        assert float(target.abs().max()) <= 1.0
        assert bool(torch.isfinite(target).all())


def test_only_the_documented_columns_are_read():
    obs = _random_obs(64, seed=5)
    target, branch = R.restance_target(obs, P0)
    other = obs.clone()
    keep = list(R.CAM_OBS_COLS) + list(R.PROXY_OBS_COLS)
    mask = torch.ones(W, dtype=torch.bool)
    mask[keep] = False
    other[:, mask] += 7.0
    t2, b2 = R.restance_target(other, P0)
    assert torch.equal(t2, target) and torch.equal(b2, branch)
    # narrower than 89 columns is refused (the proxy would be missing); exactly 89 is enough
    with pytest.raises(ValueError, match="89"):
        R.restance_target(obs[:, :88], P0)
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
    p = R.parse_restance("lift=-0.6, rho=0.4,theta_L=0.3")
    assert (p.lift, p.rho, p.theta_L) == (-0.6, 0.4, 0.3) and p.theta_S == P0.theta_S
    for q in PARAM_SETS + [p]:
        text = R.format_restance(q)
        assert " " not in text
        assert R.parse_restance(text) == q
        assert R.RestanceParams.from_dict(q.to_dict()) == q
        assert all(isinstance(v, float) for v in q.to_dict().values())
    assert set(P0.to_dict()) == {"lift", "rho", "theta_L", "theta_S", "kappa", "dmax", "k_c", "k_h", "trim",
                                 "eps_c", "eps_air"}
    with pytest.raises(dataclass_frozen()):
        P0.lift = -0.9                                              # frozen


def dataclass_frozen():
    import dataclasses
    return dataclasses.FrozenInstanceError


@pytest.mark.parametrize("text,match", [
    ("lift=0.3", "lift"), ("lift=-1.5", "lift"), ("theta_L=0.8", "theta_L"), ("theta_S=1.2", "theta_S"),
    ("theta_L=-0.6", "theta_L"), ("kappa=0", "kappa"), ("eps_air=0.2", "eps_air"), ("rho=-0.1", "rho"),
    ("kappa=2,dmax=0.6", "clip"), ("trim=1.5", "trim"), ("bogus=1", "unknown"), ("lift=-0.5,lift=-0.6", "twice"),
    ("lift", "k=v"), ("lift=abc", "not a number"), ("lift=nan", "finite"),
])
def test_parse_refusals(text, match):
    with pytest.raises(ValueError, match=match):
        R.parse_restance(text)


def test_from_dict_refuses_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        R.RestanceParams.from_dict({**P0.to_dict(), "gain": 1.0})
    assert math.isclose(R.RestanceParams.from_dict({"lift": "-0.4"}).lift, -0.4)
