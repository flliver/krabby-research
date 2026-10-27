"""The omni BODY-FRAME command mode (robot-frame velocity commands campaign, step 1), on CPU.

``experiments/2026-10-15_body_cmd/STEP1_SPEC.md`` §8. ``KRABBY_OMNI_COMMAND_FRAME=body`` commands the
omni walker with a held robot-frame ``(vx, vy, wz)`` instead of a world travel direction + world
facing. Everything here is DEFAULT OFF, so half of this file pins that the WORLD path did not move
(the sampler against the HEAD ``60d7f45d`` sources, bitwise, RNG state included; the presets; the
clock seam) and the other half pins the body semantics:

* the pure math (``crab_hex_omni_command_math``): the envelope grammar, the coupled cap, the ONE
  violation definition, the deployment shaper, the ``body_hold`` adapter;
* the command term (``CrabHexOmniBodyCommand``): same uniforms in the same order as the world
  sampler, every draw inside the envelope, stand exactly ``(0, 0, 0)``, slot = f(command), held
  ``vel_command_b``, legacy fields, the eval hooks, the identity methods, the refusals;
* the observation (``facing_obs_encoding="none"``: cols 6-7 exactly 0 in obs and history);
* the reward set (the windowed incomes and the class terms that call them; the straight gate);
* the knob layer (``omni_knobs._apply_command_frame``: every refusal, the swap, world writes nothing);
* the presets (``1a_b`` / ``1b_b`` / ``2a_b`` = base + ``BODY_DELTA``) and the driver's dump checks.

Nothing imports Isaac or launches a process (git is read for the HEAD sources; those tests skip
without it). The command term is loaded through ``test_crab_hex_omni_command.py``'s stub harness,
the observation through ``test_crab_hex_omni_observations.py``'s, the reward terms through
``test_crab_hex_omni_rewards.py``'s source-exec helper -- each loaded by path, like
``test_omni_p2r_z2_core.py`` loads its siblings.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
OMNI_MDP = OMNI / "mdp"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
BODY_KNOBS_PY = OMNI / "config/crab_hex/body_knobs.py"
OMNI_PHASES_PY = OMNI / "config/crab_hex/crab_hex_omni_phases.py"
YAW_WINDOW_PY = OMNI_MDP / "crab_hex_omni_yaw_window.py"
OMNI_REWARDS_PY = OMNI_MDP / "crab_hex_omni_rewards.py"
RUN_CANDIDATES = OMNI / "experiments/tools/run_candidates.py"
CLOCK_MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp"
LIB_REWARDS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/rewards.py"
S1_DUMP = (REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
           / "params/env.yaml")

HEAD_SHA = "60d7f45d"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the sibling harnesses (each sets up its own stubs, idempotently)
H = _load(UNIT / "test_crab_hex_omni_command.py", "omni_cmd_harness__body_cmd")
OH = _load(UNIT / "test_crab_hex_omni_observations.py", "omni_obs_harness__body_cmd")
RH = _load(UNIT / "test_crab_hex_omni_rewards.py", "omni_rew_harness__body_cmd")
om = H.omni
mm = H.omni_math
yw = _load(YAW_WINDOW_PY, "omni_yaw_window__body_cmd")
knobs = _load(OMNI_KNOBS_PY, "crab_hex_omni_knobs__body_cmd")
bk = knobs._bk
phases = _load(OMNI_PHASES_PY, "crab_hex_omni_phases")
if str(CLOCK_MDP) not in sys.path:
    sys.path.insert(0, str(CLOCK_MDP))
from crab_hex_clock_reward import clock_command_speed  # noqa: E402

# --------------------------------------------------------------------------------- fixtures
ENV_TXT = "0.05:0.167:0.21"
CAP_TXT = "0@0.21,0.22@0.15,0.30@0.04"          # crosses w_min at v = 0.29273 (> the 0.21 floor)
ENVELOPE = mm.parse_body_envelope(ENV_TXT, CAP_TXT)
ELLIPSE = (0.35, 0.35, 0.25)
FLOOR = 0.2 + 0.01                                # lin_vel_clip + WALK_MARGIN_M_S
N = 4096
DT = 0.02

#: The shipped stage pools (``experiments/2026-10-15_body_cmd/dump_targets.json``), with their slots.
POOL_C4 = ("stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,"
           "side_l_turn:0.025,side_r_turn:0.025", (0.15, 0.6, 0.15, 0.1))
POOL_C6 = ("stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,"
           "side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07", (0.15, 0.55, 0.1, 0.2))
POOL_FB = ("stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2", (0.2, 0.8, 0.0, 0.0))
#: (name, cfg overrides) for every sampler branch the world chain of record runs.
WORLD_CASES = [
    ("pool_less_1c", dict(slot_probs=(0.15, 0.55, 0.10, 0.20), facing_delta=(0.2, 0.6),
                          speed_ellipse=(0.35, 0.25, 0.25))),
    ("pool_less_1a_band", dict(slot_probs=(0.2, 0.8, 0.0, 0.0), dir_band=(0.0, 0.0),
                               speed_ellipse=(0.35, 0.25, 0.25))),
    ("pool_c4", dict(cmd_pool=POOL_C4[0], slot_probs=POOL_C4[1], speed_ellipse=ELLIPSE)),
    ("pool_c6", dict(cmd_pool=POOL_C6[0], slot_probs=POOL_C6[1], speed_ellipse=ELLIPSE)),
    ("pool_fb", dict(cmd_pool=POOL_FB[0], slot_probs=POOL_FB[1], speed_ellipse=ELLIPSE)),
]
SEEDS = (1, 7, 20261015)


def _env(n: int, yaw: torch.Tensor | None = None, cfg=None):
    robot = H._Robot(n, H._spread_yaw(n) if yaw is None else yaw)
    env = types.SimpleNamespace(num_envs=n, device="cpu", scene=H._Scene(robot), step_dt=DT,
                                max_episode_length=2000)
    if cfg is not None:
        env.cfg = cfg
    return env, robot


def _world_cfg(mod=om, **over):
    cfg = mod.CrabHexOmniCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_control_stiffness=1.5,
        ranges=H.ParkourCommandCfg.Ranges(lin_vel_x=(0.0, 0.35), heading=(0.0, 0.0)),
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def _body_cfg(envelope=ENVELOPE, **over):
    cfg = om.CrabHexOmniBodyCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_control_stiffness=1.5,
        ranges=H.ParkourCommandCfg.Ranges(lin_vel_x=(0.0, 0.35), heading=(0.0, 0.0)),
        body_w_min=envelope.w_min, body_ceiling=envelope.ceiling, body_w_max=envelope.w_max,
        body_cap=envelope.cap,
    )
    over.setdefault("speed_ellipse", ELLIPSE)
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def _body_term(cfg=None, n=N, yaw=None, env_cfg=None):
    env, robot = _env(n, yaw, env_cfg)
    return om.CrabHexOmniBodyCommand(_body_cfg() if cfg is None else cfg, env), robot


def _resample(term, seed, ids=None):
    ids = torch.arange(term.num_envs) if ids is None else ids
    torch.manual_seed(seed)
    term._resample_command(ids)
    return ids


def _cmd(term):
    c = term.vel_command_b
    return c[:, 0], c[:, 1], c[:, 2]


# ============================================================ the HEAD sampler, bitwise
@pytest.fixture(scope="module")
def head_mod(tmp_path_factory):
    """The HEAD ``60d7f45d`` command + math modules, loaded into a private package (git show)."""
    out = tmp_path_factory.mktemp("omni_mdp_head")
    for name in ("crab_hex_omni_command.py", "crab_hex_omni_command_math.py"):
        rel = (OMNI_MDP / name).relative_to(REPO).as_posix()
        try:
            src = subprocess.run(["git", "-C", str(REPO), "show", f"{HEAD_SHA}:{rel}"],
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"git show {HEAD_SHA}:{rel} unavailable ({exc})")
        (out / name).write_text(src)
    H._synthetic_package("_omni_mdp_head_60d7f45d", out)
    return importlib.import_module("_omni_mdp_head_60d7f45d.crab_hex_omni_command")


def _world_state(term):
    return (term.travel_dir_w, term.speed_cmd, term.heading_target, term.slot_id, term.type_id,
            term.vel_command_b, term.phi_b, term.recap_mask)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", [c[0] for c in WORLD_CASES])
def test_world_sampler_is_bitwise_head(head_mod, case, seed):
    """Same seed, same cfg, same yaw: identical tensors and an identical RNG state after."""
    over = dict(WORLD_CASES)[case]
    yaw = H._spread_yaw(N)
    env_h, robot_h = _env(N, yaw)
    env_n, robot_n = _env(N, yaw)
    t_head = head_mod.CrabHexOmniCommand(_world_cfg(head_mod, **over), env_h)
    t_new = om.CrabHexOmniCommand(_world_cfg(om, **over), env_n)

    _resample(t_head, seed)
    rng_head = torch.get_rng_state()
    _resample(t_new, seed)
    assert torch.equal(torch.get_rng_state(), rng_head), "the world sampler's RNG consumption moved"
    for a, b in zip(_world_state(t_head), _world_state(t_new)):
        assert torch.equal(a, b)
    # the per-step re-projection after the body turns
    for robot in (robot_h, robot_n):
        robot.data.heading_w = robot.data.heading_w + 0.37
    t_head._update_command()
    t_new._update_command()
    for a, b in zip(_world_state(t_head), _world_state(t_new)):
        assert torch.equal(a, b)
    # a partial resample (the training shape), then the metrics
    ids = torch.arange(0, N, 3)
    _resample(t_head, seed + 1, ids)
    _resample(t_new, seed + 1, ids)
    for a, b in zip(_world_state(t_head), _world_state(t_new)):
        assert torch.equal(a, b)
    t_head._update_metrics()
    t_new._update_metrics()
    assert set(t_head.metrics) == set(t_new.metrics)
    for key in t_head.metrics:
        assert torch.equal(t_head.metrics[key], t_new.metrics[key]), key
    # the identity methods are the HEAD expressions, bitwise
    sig_head = torch.stack((t_head.slot_id.float(), t_head.travel_dir_w, t_head.heading_target), dim=-1)
    assert torch.equal(t_new.command_signature(), sig_head)
    assert torch.equal(t_new.intended_bearing_b(), t_head.travel_dir_w - t_head.heading_target)
    assert torch.equal(t_new.telemetry_category(), t_head.telemetry_category())


def test_world_cfg_gains_no_field(head_mod):
    """The world cfg CLASS declares exactly HEAD's fields (a knob-off dump cannot grow one)."""
    def fields(cls):
        return tuple(getattr(cls, "__cfg_fields__"))
    assert fields(om.CrabHexOmniCommandCfg) == fields(head_mod.CrabHexOmniCommandCfg)
    assert vars(_world_cfg(om)).keys() == vars(_world_cfg(head_mod)).keys()
    body_only = set(fields(om.CrabHexOmniBodyCommandCfg)) - set(fields(om.CrabHexOmniCommandCfg))
    assert body_only == {"body_w_min", "body_ceiling", "body_w_max", "body_cap"}


# ============================================================ the pure math: the envelope
def test_parse_body_envelope_round_trip():
    assert ENVELOPE.w_min == 0.05 and ENVELOPE.ceiling == 0.167 and ENVELOPE.w_max == 0.21
    assert ENVELOPE.cap == ((0.0, 0.21), (0.22, 0.15), (0.30, 0.04))
    # whitespace tolerated
    assert mm.parse_body_envelope(" 0.05 : 0.167:0.21 ", " 0 @ 0.21 , 0.22@0.15,0.30@0.04 ") == ENVELOPE
    assert mm.BODY_ENVELOPE_KNOB == bk.BODY_ENVELOPE_VAR == "KRABBY_OMNI_BODY_ENVELOPE"
    assert mm.BODY_CAP_KNOB == bk.BODY_CAP_VAR == "KRABBY_OMNI_BODY_CAP"


@pytest.mark.parametrize("env_txt", ["", "0.05:0.2", "0.05:0.2:0.3:0.4", "0:0.1:0.2", "0.1:0.1:0.2",
                                     "0.1:0.3:0.2", "x:0.1:0.2", "0.05:nan:0.2", "-0.05:0.1:0.2"])
def test_envelope_triple_refusals(env_txt):
    with pytest.raises(ValueError, match="KRABBY_OMNI_BODY_ENVELOPE"):
        mm.parse_body_envelope(env_txt, CAP_TXT)


@pytest.mark.parametrize("cap_txt", ["", "0.1@0.2", "0@0.2,0.3@0.1,0.2@0.05", "0@0.2,0.2@0.2,0.2@0.1",
                                     "0@0.1,0.2@0.15", "0@0.2,0.2@0", "0@0.2,", "0@0.2,0.3", "0@0.2@0.1",
                                     "0@0.2,x@0.1", "0@0.2,0.3@inf"])
def test_cap_shape_refusals(cap_txt):
    with pytest.raises(ValueError, match="KRABBY_OMNI_BODY_CAP"):
        mm.parse_body_envelope(ENV_TXT, cap_txt)


def test_cap_above_w_max_is_refused():
    with pytest.raises(ValueError, match="w_max"):
        mm.parse_body_envelope("0.05:0.167:0.21", "0@0.25,0.3@0.1")
    assert mm.parse_body_envelope("0.05:0.167:0.21", "0@0.21").cap == ((0.0, 0.21),)


def test_body_cap_wz_is_piecewise_linear_and_constant_beyond_the_last_point():
    e = ENVELOPE
    assert mm.body_cap_wz(0.0, e) == 0.21
    assert mm.body_cap_wz(0.22, e) == pytest.approx(0.15)
    assert mm.body_cap_wz(0.11, e) == pytest.approx(0.18)
    assert mm.body_cap_wz(0.26, e) == pytest.approx(0.095)
    assert mm.body_cap_wz(0.30, e) == pytest.approx(0.04)
    for beyond in (0.31, 0.5, 3.0):
        assert mm.body_cap_wz(beyond, e) == pytest.approx(0.04)
    assert mm.body_cap_wz(-0.11, e) == mm.body_cap_wz(0.11, e), "|v|"
    v = torch.linspace(-0.6, 0.6, 1201, dtype=torch.float64)
    got = mm.body_cap_wz(v, e)
    assert got.shape == v.shape and got.dtype == v.dtype
    want = torch.tensor([mm.body_cap_wz(float(x), e) for x in v], dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-15, rtol=0.0)
    assert bool((got[1:601] - got[:600] >= -1e-15).all()), "non-increasing in |v| (mirror half)"
    one = mm.BodyEnvelope(0.05, 0.1, 0.2, ((0.0, 0.12),))
    assert torch.equal(mm.body_cap_wz(torch.tensor([0.0, 0.3]), one), torch.tensor([0.12, 0.12]))


def test_body_v_turn_max():
    vt = mm.body_v_turn_max(ENVELOPE)
    assert vt == pytest.approx(0.22 + (0.05 - 0.15) * 0.08 / (0.04 - 0.15))
    assert mm.body_cap_wz(vt, ENVELOPE) == pytest.approx(0.05, abs=1e-12)
    assert mm.body_cap_wz(vt + 1e-6, ENVELOPE) < 0.05
    assert mm.body_v_turn_max(mm.parse_body_envelope(ENV_TXT, "0@0.21,0.3@0.06")) == math.inf
    assert mm.body_v_turn_max(mm.BodyEnvelope(0.05, 0.1, 0.2, ((0.0, 0.04),))) == 0.0
    flat = mm.parse_body_envelope(ENV_TXT, "0@0.21,0.25@0.05,0.3@0.01")
    assert mm.body_v_turn_max(flat) == pytest.approx(0.25)


def test_sample_yaw_rate_uses_sample_facings_sign_convention():
    """Same two uniforms -> the same sign and magnitude as ``sample_facing``'s change."""
    yaw = torch.zeros(5000, dtype=torch.float64)
    g = torch.Generator().manual_seed(11)
    psi = mm.sample_facing(yaw, 0.2, 0.6, generator=g)
    g = torch.Generator().manual_seed(11)
    u_mag = torch.rand(5000, generator=g, dtype=torch.float64)
    u_sign = torch.rand(5000, generator=g, dtype=torch.float64)
    wz = mm.sample_yaw_rate(u_mag, u_sign, 0.2, 0.6)
    assert torch.equal(wz, psi), "lo + u_mag (hi - lo), signed by u_sign < 0.5 -> -1"
    lo = torch.full((5000,), 0.05, dtype=torch.float64)
    hi = torch.linspace(0.05, 0.21, 5000, dtype=torch.float64)
    w = mm.sample_yaw_rate(u_mag, u_sign, lo, hi)
    assert bool((w.abs() >= lo - 1e-15).all() and (w.abs() <= hi + 1e-15).all())
    assert torch.equal(torch.sign(w), torch.sign(psi))


def test_slot_from_body_cmd():
    vx = torch.tensor([0.0, 0.25, 0.0, -0.22, 0.0, 1e-9])
    vy = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.25, 0.0])
    wz = torch.tensor([0.0, 0.0, 0.1, -0.06, -0.2, 0.0])
    assert mm.slot_from_body_cmd(vx, vy, wz).tolist() == [0, 1, 2, 3, 3, 1]
    assert mm.slot_from_body_cmd(vx, vy, wz).dtype == torch.long


@pytest.mark.parametrize("cmd, kinds", [
    ((0.0, 0.0, 0.0), ()),
    ((0.25, 0.0, 0.0), ()),
    ((0.0, -0.35, 0.0), ()),
    ((-0.25, 0.0, 0.0), ()),
    ((0.0, 0.0, 0.05), ()),
    ((0.0, 0.0, -0.21), ()),
    ((0.22, 0.0, 0.15), ()),
    ((0.15, 0.0, 0.0), ("speed",)),
    ((0.36, 0.0, 0.0), ("speed",)),
    ((-0.26, 0.0, 0.0), ("speed",)),
    ((0.0, 0.0, 0.04), ("|wz|",)),
    ((0.0, 0.0, 0.22), ("|wz|",)),
    ((0.30, 0.0, 0.05), ("cap",)),
    ((0.25, 0.0, -0.2), ("cap",)),
    ((0.1, 0.0, 0.3), ("speed", "|wz|", "cap")),
    ((float("nan"), 0.0, 0.0), ("non-finite",)),
])
def test_body_cmd_violations(cmd, kinds):
    got = mm.body_cmd_violations(*cmd, ENVELOPE, ELLIPSE, FLOOR)
    assert len(got) == len(kinds), got
    for text, kind in zip(got, kinds):
        assert kind in text, (text, kind)
    # the batch helper is the same definition
    t = [torch.tensor([x], dtype=torch.float64) for x in cmd]
    assert bool(mm.body_cmd_violation_mask(*t, ENVELOPE, ELLIPSE, FLOOR)[0]) == bool(kinds)


def test_violation_tolerance_and_exact_zero():
    assert mm.body_cmd_violations(FLOOR - 5e-7, 0.0, 0.0, ENVELOPE, ELLIPSE, FLOOR) == ()
    assert mm.body_cmd_violations(FLOOR - 5e-6, 0.0, 0.0, ENVELOPE, ELLIPSE, FLOOR) != ()
    # a speed of 1e-9 is MOVING (the slot rule's exact != 0), so it is below the floor
    assert mm.body_cmd_violations(1e-9, 0.0, 0.0, ENVELOPE, ELLIPSE, FLOOR) != ()
    assert mm.body_cmd_violations(0.0, 0.0, 1e-9, ENVELOPE, ELLIPSE, FLOOR) != ()


def _random_cmds(n=20000, seed=5):
    g = torch.Generator().manual_seed(seed)
    vx = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 1.0
    vy = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 1.0
    wz = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 0.8
    # exact zeros and edge values in the mix
    vx[:500] = 0.0
    vy[250:750] = 0.0
    wz[::7] = 0.0
    wz[1::11] = 0.03
    return vx, vy, wz


@pytest.mark.parametrize("turns", [True, False])
def test_shape_body_command_output_is_always_valid(turns):
    vx, vy, wz = _random_cmds()
    nvx, nvy, nwz, clipped = mm.shape_body_command(vx, vy, wz, ENVELOPE, ELLIPSE, FLOOR, turns=turns)
    assert not bool(mm.body_cmd_violation_mask(nvx, nvy, nwz, ENVELOPE, ELLIPSE, FLOOR).any())
    if not turns:
        assert torch.equal(nwz, torch.zeros_like(nwz))
    # the dead zone, the bearing kept, and clipped exactly where something moved
    moved = (torch.abs(nvx - vx) > 1e-9) | (torch.abs(nvy - vy) > 1e-9) | (torch.abs(nwz - wz) > 1e-9)
    assert torch.equal(clipped, moved)
    moving = (vx != 0) | (vy != 0)
    assert torch.allclose(torch.atan2(nvy, nvx)[moving], torch.atan2(vy, vx)[moving], atol=1e-9)
    assert bool(((nwz == 0) | (torch.sign(nwz) == torch.sign(wz))).all()), "the sign is kept"
    if turns:
        dead = (wz != 0) & (wz.abs() < ENVELOPE.w_min)
        assert bool((nwz[dead] == 0).all())


def test_shape_body_command_leaves_a_valid_command_bitwise_alone():
    for cmd in ((0.25, 0.0, 0.0), (0.0, 0.0, 0.1), (0.22, 0.0, -0.12), (0.0, 0.0, 0.0), (-0.24, 0.0, 0.0)):
        out = mm.shape_body_command(*cmd, ENVELOPE, ELLIPSE, FLOOR)
        assert out == (*cmd, False), (cmd, out)


def test_shape_body_command_named_cases():
    sh = lambda *c, **k: mm.shape_body_command(*c, ENVELOPE, ELLIPSE, FLOOR, **k)  # noqa: E731
    assert sh(0.1, 0.0, 0.0)[:3] == pytest.approx((FLOOR, 0.0, 0.0))       # floor
    assert sh(0.5, 0.0, 0.0)[:3] == pytest.approx((0.35, 0.0, 0.0))        # ellipse cap, bearing kept
    assert sh(0.0, 0.0, 0.03)[:3] == (0.0, 0.0, 0.0)                       # dead zone
    assert sh(0.0, 0.0, -0.5)[:3] == pytest.approx((0.0, 0.0, -0.21))      # pure turn -> w_max
    vt = mm.body_v_turn_max(ENVELOPE)
    vx, vy, w, clipped = sh(0.34, 0.0, 0.15)                               # moving turn, cap < w_min
    assert (vx, vy) == pytest.approx((vt, 0.0)) and w == pytest.approx(0.05) and clipped
    vx, vy, w, _c = sh(0.22, 0.0, 0.2)                                     # clamp to the cap
    assert vx == 0.22 and w == pytest.approx(0.15)
    assert sh(0.25, 0.0, 0.1, turns=False)[:3] == (0.25, 0.0, 0.0)
    # tensor and float paths agree
    t = mm.shape_body_command(torch.tensor([0.34], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64),
                              torch.tensor([0.15], dtype=torch.float64), ENVELOPE, ELLIPSE, FLOOR)
    assert float(t[0][0]) == pytest.approx(vt) and float(t[2][0]) == pytest.approx(0.05)


def test_shape_body_command_refuses_an_envelope_with_no_moving_turn():
    bad = mm.parse_body_envelope(ENV_TXT, "0@0.21,0.1@0.04")
    with pytest.raises(ValueError, match="no moving turn"):
        mm.shape_body_command(0.25, 0.0, 0.1, bad, ELLIPSE, FLOOR)
    with pytest.raises(ValueError, match="walk floor"):
        mm.shape_body_command(0.25, 0.0, 0.1, ENVELOPE, (0.35, 0.2, 0.25), FLOOR)


def test_body_hold_step_integrates_with_an_anti_windup_bound():
    # small lead: the target advances by wz * dt from the face
    assert mm.body_hold_step(0.1, 0.0, 0.2, DT) == pytest.approx(0.1 + 0.2 * DT)
    # a head that does not turn: the target saturates at windup ahead of the yaw
    face = 0.0
    for _ in range(500):
        face = mm.body_hold_step(face, 0.0, 0.21, DT)
    assert face == pytest.approx(0.6)
    face = mm.body_hold_step(face, 0.0, -0.21, DT, windup=0.3)
    assert face == pytest.approx(0.3)
    # tensors: the bound holds on every row, across the wrap
    g = torch.Generator().manual_seed(2)
    f = (torch.rand(1000, generator=g, dtype=torch.float64) - 0.5) * 20
    y = (torch.rand(1000, generator=g, dtype=torch.float64) - 0.5) * 20
    w = (torch.rand(1000, generator=g, dtype=torch.float64) - 0.5) * 0.4
    out = mm.body_hold_step(f, y, w, DT)
    assert bool((mm.wrap_to_pi(out - y).abs() <= 0.6 + 1e-12).all())
    floats = torch.tensor([mm.body_hold_step(float(a), float(b), float(c), DT) for a, b, c in zip(f, y, w)],
                          dtype=torch.float64)
    assert torch.allclose(mm.wrap_to_pi(floats - out), torch.zeros_like(out), atol=1e-12)


def test_body_hold_world_row():
    row = mm.body_hold_world_row(0.0, 0.25, 0.1, 0.3, 1.0)
    assert row.speed == pytest.approx(0.25) and row.dir == pytest.approx(1.0 + math.pi / 2) and row.face == 0.3
    assert (row.speed, row.dir, row.face) == tuple(row)
    still = mm.body_hold_world_row(0.0, 0.0, 0.1, 0.3, 1.0)
    assert still.speed == 0.0 and still.dir == pytest.approx(1.0)
    t = mm.body_hold_world_row(torch.tensor([0.25]), torch.tensor([0.0]), torch.tensor([0.0]),
                               torch.tensor([0.2]), torch.tensor([0.5]))
    assert float(t.speed[0]) == pytest.approx(0.25) and float(t.dir[0]) == pytest.approx(0.5)


# ============================================================ the body command term
BODY_CASES = [
    ("pool_less", dict(slot_probs=(0.15, 0.55, 0.10, 0.20))),
    ("pool_c4", dict(cmd_pool=POOL_C4[0], slot_probs=POOL_C4[1])),
    ("pool_c6", dict(cmd_pool=POOL_C6[0], slot_probs=POOL_C6[1])),
    ("pool_fb", dict(cmd_pool=POOL_FB[0], slot_probs=POOL_FB[1])),
]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", [c[0] for c in BODY_CASES])
def test_body_sampler_draws_inside_the_envelope(case, seed):
    term, _robot = _body_term(_body_cfg(**dict(BODY_CASES)[case]))
    _resample(term, seed)
    vx, vy, wz = _cmd(term)
    assert not bool(mm.body_cmd_violation_mask(vx, vy, wz, ENVELOPE, ELLIPSE, om.omni_walk_floor(term.cfg)).any())
    # slot = f(command), on every draw
    assert torch.equal(term.slot_id, mm.slot_from_body_cmd(vx, vy, wz))
    # stand rows are EXACTLY (0, 0, 0) -- no -0.0 either
    stand = term.slot_id == mm.SLOT_STAND
    assert bool(stand.any())
    s = term.vel_command_b[stand]
    assert torch.equal(s, torch.zeros_like(s)) and not bool(torch.signbit(s).any())
    # every non-turning row commands wz == 0 exactly, every turning row |wz| >= w_min
    turning = mm.slot_is_turning(term.slot_id)
    assert bool((wz[~turning] == 0).all()) and bool((wz[turning].abs() >= ENVELOPE.w_min - 1e-7).all())
    # held state matches the command
    assert torch.equal(term.body_wz, wz)
    moving = mm.slot_is_moving(term.slot_id)
    assert torch.allclose(torch.hypot(vx, vy)[moving], term.body_speed[moving], atol=1e-6)
    assert bool((term.body_speed[~moving] == 0).all())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", [c[0] for c in BODY_CASES])
def test_body_sampler_consumes_the_world_rng_stream(case, seed):
    """Same seed: the RNG state after a body resample equals the world one's, and the shared
    draws agree (slot, body bearing, speed where not cap-reduced, the turn's sign)."""
    over = dict(BODY_CASES)[case]
    yaw = H._spread_yaw(N)
    env_w, _rw = _env(N, yaw)
    world = om.CrabHexOmniCommand(_world_cfg(speed_ellipse=ELLIPSE, **over), env_w)
    body, _rb = _body_term(_body_cfg(**over), yaw=yaw)
    _resample(world, seed)
    rng_world = torch.get_rng_state()
    _resample(body, seed)
    assert torch.equal(torch.get_rng_state(), rng_world)
    assert torch.equal(body.slot_id, world.slot_id) and torch.equal(body.type_id, world.type_id)
    assert torch.equal(body.body_phi, world.phi_b), "phi = wrap(theta - yaw), the world's own expression"
    reduced = body.body_speed == torch.tensor(body.body_v_turn_max, dtype=body.body_speed.dtype)
    assert torch.equal(body.body_speed[~reduced], world.speed_cmd[~reduced])
    turning = mm.slot_is_turning(world.slot_id)
    assert torch.equal(torch.sign(body.body_wz[turning]), torch.sign(world.heading_target - yaw)[turning])


def test_the_coupled_cap_and_the_v_turn_max_reduction():
    """A move+turn-only pool: every row turns inside the cap; rows drawn above v_turn_max walk at it."""
    term, _r = _body_term(_body_cfg(cmd_pool="fwd_turn:0.5,side_l_turn:0.5", slot_probs=(0, 0, 0, 1)))
    _resample(term, 3)
    vt = term.body_v_turn_max
    speed, wz = term.body_speed, term.body_wz
    assert bool((speed <= vt + 1e-6).all()) and bool((speed >= FLOOR - 1e-6).all())
    reduced = torch.isclose(speed, torch.tensor(vt, dtype=speed.dtype), atol=1e-7)
    assert 0 < int(reduced.sum()) < term.num_envs, "the fixture must exercise both branches"
    cap = mm.body_cap_wz(speed.double(), ENVELOPE)
    assert bool((wz.abs().double() <= cap + 1e-6).all()) and bool((wz.abs() >= ENVELOPE.w_min - 1e-7).all())
    # a pure-turn row is NOT capped by the coupled frontier: it draws up to w_max
    turn, _r = _body_term(_body_cfg(cmd_pool="turn:1.0", slot_probs=(0, 0, 1, 0)))
    _resample(turn, 3)
    assert float(turn.body_wz.abs().max()) > 0.2 and float(turn.body_wz.abs().max()) <= 0.21 + 1e-7
    assert torch.equal(turn.body_speed, torch.zeros_like(turn.body_speed))


def test_vel_command_b_is_held_while_the_yaw_moves():
    term, robot = _body_term()
    _resample(term, 4)
    held = term.vel_command_b.clone()
    for k in range(5):
        robot.data.heading_w = mm.wrap_to_pi(robot.data.heading_w + 0.4 * (k + 1))
        term._update_command()
        assert torch.equal(term.vel_command_b, held), "no re-projection, no relay"
        yaw = robot.data.heading_w
        assert torch.equal(term.heading_target, yaw)
        assert torch.equal(term.travel_dir_w, yaw + term.body_phi)
        assert torch.equal(term.speed_cmd, term.body_speed)
        assert torch.equal(term.phi_b, term.body_phi)
        assert not bool(term.recap_mask.any())


def test_body_metrics_keep_their_names_and_read_zero_facing_error():
    term, robot = _body_term()
    world, _rw = _env(N)
    names = set(om.CrabHexOmniCommand(_world_cfg(speed_ellipse=ELLIPSE), world).metrics)
    assert set(term.metrics) == names, "no new metric"
    _resample(term, 5)
    robot.data.heading_w = robot.data.heading_w + 0.25           # physics moved the yaw
    robot.data.root_ang_vel_b[:, 2] = torch.sign(term.body_wz) * 0.1
    term._update_metrics()
    assert torch.equal(term.metrics["facing_err_abs"], torch.zeros(N))
    assert torch.equal(term.metrics["recap_frac"], torch.zeros(N))
    # yaw_progress keeps its cmd_wz != 0 gate: +0.1 on every turning row, divided by the episode
    assert float(term.metrics["yaw_progress"][0]) == pytest.approx(0.1 / 2000)


def test_command_signature_changes_exactly_at_resample_and_hook():
    term, robot = _body_term(_body_cfg(slot_probs=(0.15, 0.55, 0.10, 0.20)))
    _resample(term, 6)
    s0 = term.command_signature().clone()
    assert torch.equal(s0, torch.stack((term.body_phi, term.body_speed, term.body_wz), dim=-1))
    robot.data.heading_w = robot.data.heading_w + 0.7
    term._update_command()
    term._update_metrics()
    assert torch.equal(term.command_signature(), s0), "a yaw change is not a new command"
    ids = torch.arange(0, N, 4)
    _resample(term, 7, ids)
    changed = (term.command_signature() != s0).any(dim=-1)
    want = torch.zeros(N, dtype=torch.bool)
    want[ids] = True
    assert torch.equal(changed, want)
    assert torch.equal(term.intended_bearing_b(), term.body_phi)
    s1 = term.command_signature().clone()
    term.schedule_hook_body(0.25, 0.0, 0.1)
    s2 = term.command_signature().clone()
    assert bool((s2 != s1).any(dim=-1).all())
    term.schedule_hook_body(0.25, 0.0, 0.1)
    assert torch.equal(term.command_signature(), s2), "the same row hooked again is not a new command"
    # telemetry reads the held bearing: a hooked lateral row is "side", whatever the yaw does
    term.schedule_hook_body({"vx": 0.0, "vy": 0.25, "wz": 0.0})
    robot.data.heading_w = robot.data.heading_w + 1.3
    term._update_command()
    side = mm.TELEMETRY_CATEGORIES.index("side")
    assert bool((term.telemetry_category() == side).all())


def test_world_command_signature_is_the_old_expression_and_changes_only_at_resample():
    env, robot = _env(N)
    world = om.CrabHexOmniCommand(_world_cfg(speed_ellipse=ELLIPSE, cmd_pool=POOL_C6[0], slot_probs=POOL_C6[1]),
                                  env)
    _resample(world, 8)
    sig = world.command_signature().clone()
    assert torch.equal(sig, torch.stack((world.slot_id.float(), world.travel_dir_w, world.heading_target), -1))
    robot.data.heading_w = robot.data.heading_w + 0.05
    world._update_command()
    assert torch.equal(world.command_signature(), sig)
    assert torch.equal(world.intended_bearing_b(), world.travel_dir_w - world.heading_target)


def test_schedule_hook_body_validates_pins_and_writes_verbatim():
    term, robot = _body_term()
    _resample(term, 9)
    term.schedule_hook_body(0.22, 0.0, -0.12)
    assert torch.equal(term.vel_command_b, torch.tensor([[0.22, 0.0, -0.12]] * N))
    assert bool((term.slot_id == mm.SLOT_MOVE_TURN).all()) and bool((term.type_id == -1).all())
    assert torch.equal(term.heading_target, robot.data.heading_w)
    assert torch.equal(term.speed_cmd, torch.full((N,), 0.22))
    term.schedule_hook_body(types.SimpleNamespace(vx=0.0, vy=0.0, wz=0.0))
    assert bool((term.slot_id == mm.SLOT_STAND).all())
    assert torch.equal(term.vel_command_b, torch.zeros(N, 3))
    term.schedule_hook_body({"vx": 0.0, "vy": 0.0, "wz": 0.2})
    assert bool((term.slot_id == mm.SLOT_TURN).all())
    for bad in ((0.15, 0.0, 0.0), (0.3, 0.0, 0.1), (0.0, 0.0, 0.3), (0.0, 0.0, 0.01)):
        with pytest.raises(ValueError, match="outside the body envelope"):
            term.schedule_hook_body(*bad)
    with pytest.raises(ValueError, match="finite"):
        term.schedule_hook_body(float("nan"), 0.0, 0.0)


def test_the_hooks_refuse_the_other_route():
    term, _r = _body_term()
    with pytest.raises(RuntimeError, match="route mismatch"):
        term.schedule_hook({"speed": 0.25, "dir": 0.0, "face": 0.0})
    env, _rw = _env(8)
    world = om.CrabHexOmniCommand(_world_cfg(speed_ellipse=ELLIPSE), env)
    with pytest.raises(RuntimeError, match="route mismatch"):
        world.schedule_hook_body(0.25, 0.0, 0.0)


def test_the_body_schedule_hook_override_only_refuses():
    """The world hook's source is pinned by ``test_crab_hex_omni_command.py``; the body class's override
    of it must be nothing but the refusal (no body semantics hiding behind the world name)."""
    import ast  # noqa: PLC0415

    src = (OMNI_MDP / "crab_hex_omni_command.py").read_text()
    body = [c for c in ast.parse(src).body if isinstance(c, ast.ClassDef) and c.name == "CrabHexOmniBodyCommand"][0]
    hook = [f for f in body.body if isinstance(f, ast.FunctionDef) and f.name == "schedule_hook"][0]
    stmts = [s for s in hook.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(stmts) == 1 and isinstance(stmts[0], ast.Raise)


@pytest.mark.parametrize("mutate, match", [
    (lambda c: setattr(c, "body_w_min", None), "body_w_min"),
    (lambda c: setattr(c, "body_cap", ((0.0, 0.21), (0.1, 0.04))), "no moving turn"),
    (lambda c: setattr(c, "body_cap", ((0.0, 0.3),)), "w_max"),
    (lambda c: setattr(c, "relay_hyst", (0.25, 0.05)), "relay"),
    (lambda c: (setattr(c, "cmd_pool", "stand:0.2,turn:0.8:0.5"), setattr(c, "slot_probs", (0.2, 0, 0.8, 0))),
     "ang_scale"),
])
def test_the_body_term_refuses_at_construction(mutate, match):
    cfg = _body_cfg()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        _body_term(cfg, n=8)


def test_the_body_term_refuses_an_installed_rsi_event_and_a_staged_row():
    env_cfg = types.SimpleNamespace(events=types.SimpleNamespace(rsi_reference_reset=object()))
    with pytest.raises(ValueError, match="rsi_reference_reset"):
        _body_term(n=8, env_cfg=env_cfg)
    term, _r = _body_term(n=8)
    term.rsi_cmd_staged[3] = torch.tensor([0.0, 0.25, 0.0])
    with pytest.raises(RuntimeError, match="RSI"):
        _resample(term, 1)


def _once_per_step():
    """``crab_hex_omni_stroke_rewards._OncePerStep`` exec'd from the shipped source (its resample test)."""
    import ast  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    path = OMNI_MDP / "crab_hex_omni_stroke_rewards.py"
    src = path.read_text()
    node = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "_OncePerStep"][0]
    ns = {"torch": torch, "ParkourManagerBasedRLEnv": object}
    exec(compile("from __future__ import annotations\n" + textwrap.dedent(ast.get_source_segment(src, node)),
                 str(path), "exec"), ns)

    class _Probe(ns["_OncePerStep"]):
        device = "cpu"

    return _Probe()


@pytest.mark.parametrize("frame", ["world", "body"])
def test_the_stroke_terms_resample_detector_reads_the_command_signature(frame):
    """(b)-class consumer: a yaw change is never a resample, a redraw always is -- in BOTH frames."""
    if frame == "world":
        env, robot = _env(N)
        term = om.CrabHexOmniCommand(_world_cfg(speed_ellipse=ELLIPSE, slot_probs=(0.15, 0.55, 0.1, 0.2)), env)
    else:
        term, robot = _body_term(_body_cfg(slot_probs=(0.15, 0.55, 0.1, 0.2)))
    probe = _once_per_step()
    probe._init_resample(N)
    _resample(term, 21)
    assert bool(probe._resampled(term).all()), "a fresh env counts as resampled"
    robot.data.heading_w = robot.data.heading_w + 0.2
    term._update_command()
    assert not bool(probe._resampled(term).any())
    ids = torch.arange(0, N, 5)
    _resample(term, 22, ids)
    want = torch.zeros(N, dtype=torch.bool)
    want[ids] = True
    assert torch.equal(probe._resampled(term), want)
    src = (OMNI_MDP / "crab_hex_omni_stroke_rewards.py").read_text()
    assert "cmd_term.command_signature()" in src and "cmd_term.intended_bearing_b()" in src


# ============================================================ the observation (D2)
def _obs_term(encoding, seed=7):
    g = torch.Generator().manual_seed(seed)
    cmd = torch.zeros(OH.N, 4)
    cmd[:, 0:2] = (torch.rand(OH.N, 2, generator=g) - 0.5) * 0.6
    cmd[:, 2] = (torch.rand(OH.N, generator=g) - 0.5) * 0.4
    heading_target = (torch.rand(OH.N, generator=g) - 0.5) * 6.0
    env = OH._Env(OH.N, cmd, heading_target)
    term = OH.OmniObs(OH._ParamCfg(facing_obs_encoding=encoding, facing_obs_clip=0.6), env)
    return term, env, cmd


def test_none_encoding_zeroes_cols_6_7_in_the_observation_and_every_history_row(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, cmd = _obs_term("none")
    assert term.facing_obs_encoding == "none"
    env.episode_length_buf = torch.tensor([1, 1, 50, 50, 0, 50])     # fresh rows replicate the row
    obs = None
    for step in range(3):
        env.common_step_counter = step
        obs = term(env, None, None, "base_parkour", OH.HISTORY)
    assert obs.shape == (OH.N, OH.OBS_DIM), "width unchanged (1149)"
    for cols in (obs[:, 6:8], term._obs_history_buffer[:, :, 6:8]):
        assert torch.equal(cols, torch.zeros_like(cols)) and not bool(torch.signbit(cols).any())
    hist_base = OH.N_PROP + OH.NSCAN + 9 + 5 + 2 * OH.NJ
    for k in range(OH.HISTORY):
        row = hist_base + k * OH.N_PROP
        assert torch.equal(obs[:, row + 6:row + 8], torch.zeros(OH.N, 2))
    # slot 5 and 8-10 keep their meanings (they read vel_command_b)
    assert torch.equal(obs[:, 5], cmd[:, 2])
    assert torch.equal(obs[:, 8:10], cmd[:, 0:2])
    assert torch.allclose(obs[:, 10], torch.linalg.norm(cmd[:, :2], dim=-1))
    assert torch.equal(term._obs_history_buffer[:, -1, 5], cmd[:, 2])


def test_none_encoding_changes_only_cols_6_7(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    none_t, none_env, _ = _obs_term("none")
    raw_t, raw_env, _ = _obs_term("raw")
    a = none_t(none_env, None, None, "base_parkour", OH.HISTORY)
    b = raw_t(raw_env, None, None, "base_parkour", OH.HISTORY)
    changed = {c for c in range(OH.N_PROP) if not torch.equal(a[:, c], b[:, c])}
    assert changed == {6, 7}
    assert torch.equal(a[:, OH.N_PROP:OH.N_PROP + OH.NSCAN], b[:, OH.N_PROP:OH.N_PROP + OH.NSCAN])


def test_the_mirror_keeps_its_tables_under_none():
    mirror = _load(OMNI_MDP / "crab_hex_omni_mirror.py", "omni_mirror__body_cmd")
    assert mirror.head_signs("none") == mirror.head_signs("raw")
    assert mirror.FACING_OBS_ENCODINGS == knobs.FACING_OBS_ENCODINGS == OH.FACING_OBS_ENCODINGS
    with pytest.raises(ValueError):
        mirror.head_signs(None)


# ============================================================ the rewards (D4 as amended)
def test_window_constants_are_the_charters():
    assert (yw.YAW_WINDOW_S, yw.YAW_WINDOW_SIGMA) == (2.0, 0.10)
    assert yw.window_steps(2.0, 0.02) == 100
    assert yw.window_steps(2.0, 0.005) == 400
    for bad in ((0.0, 0.02), (2.0, 0.0), (0.001, 0.02), (float("nan"), 0.02)):
        with pytest.raises(ValueError):
            yw.window_steps(*bad)


def _run_window(ach_rate, cmd_rate, steps=100, n_steps=100):
    w = yw.YawWindow(len(ach_rate), n_steps)
    s = None
    for _ in range(steps):
        s = w.push(ach_rate * DT, cmd_rate * DT)
    return w, s


def test_window_income_perfect_tracker_and_still_stander():
    c = torch.tensor([0.05, 0.08, 0.167, 0.21, -0.1, 0.0])
    _w, (sa, sc) = _run_window(c.clone(), c)
    assert torch.equal(yw.yaw_window_income(sa, sc, 0.10), torch.ones(6)), "a perfect tracker earns 1.0"
    _w, (sa, sc) = _run_window(torch.zeros(6), c)
    got = yw.yaw_window_income(sa, sc, 0.10)
    assert float(got[0]) == pytest.approx(math.exp(-1.0), rel=1e-5), "still stander on c = 0.05 -> exp(-1)"
    assert float(got[-1]) == 1.0, "a still robot on a straight row holds its heading perfectly"


def test_tent_values_sign_clamp_and_exact_zero_on_straight_rows():
    c = 0.1
    cases = {          # achieved rate -> tent value
        c: c, 0.0: 0.0, 2 * c: 0.0, 3 * c: -c, -c: -c, 0.5 * c: 0.5 * c, -0.5: -0.21,
    }
    for ach, want in cases.items():
        for sign in (1.0, -1.0):
            _w, (sa, sc) = _run_window(torch.tensor([sign * ach]), torch.tensor([sign * c]))
            got = float(yw.yaw_tent_income(sa, sc, 2.0, 0.21)[0])
            assert got == pytest.approx(want, abs=1e-6), (ach, sign, got)
    # exactly +0.0 where the commanded window is exactly zero, whatever the robot does
    wobble = torch.tensor([0.3, -0.3, 0.0, 1.0])
    _w, (sa, sc) = _run_window(wobble, torch.zeros(4))
    got = yw.yaw_tent_income(sa, sc, 2.0, 0.21)
    assert torch.equal(got, torch.zeros(4)) and not bool(torch.signbit(got).any())


def test_window_zero_fills_at_reset_and_not_at_a_command_change():
    c = torch.tensor([0.1, 0.1])
    w, _s = _run_window(c.clone(), c, steps=150)
    w.reset([0])
    sa, sc = w.push(torch.tensor([0.0, 0.0]), c * DT)
    assert float(sc[0]) == pytest.approx(0.1 * DT), "the reset row sums only the steps since the reset"
    assert float(sc[1]) == pytest.approx(0.1 * DT * 100, rel=1e-5), "the other row keeps its window"
    # a command that flips inside the window: a perfect tracker still earns 1.0 (both histories flip)
    w = yw.YawWindow(1, 100)
    for k in range(160):
        rate = torch.tensor([0.12 if k < 110 else -0.07])
        sa, sc = w.push(rate * DT, rate * DT)
    assert float(yw.yaw_window_income(sa, sc, 0.10)[0]) == 1.0
    w.reset()
    assert torch.equal(w.ach, torch.zeros(1, 100)) and torch.equal(w.cmd, torch.zeros(1, 100))


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return "cpu"


_REW_NS = {
    "torch": torch,
    "ManagerTermBase": _ManagerTermBase,
    "SceneEntityCfg": RH._SceneEntityCfg,
    "Articulation": object,
    "ParkourManagerBasedRLEnv": object,
    "RewTerm": object,
    "track_ang_vel_z_exp": RH.track_ang_vel_z_exp,
    "YAW_WINDOW_S": yw.YAW_WINDOW_S,
    "YAW_WINDOW_SIGMA": yw.YAW_WINDOW_SIGMA,
    "YawWindow": yw.YawWindow,
    "window_steps": yw.window_steps,
    "yaw_tent_income": yw.yaw_tent_income,
    "yaw_window_income": yw.yaw_window_income,
}
RH._exec_module_level(OMNI_REWARDS_PY, ("track_ang_vel_z_exp_straight_gated", "_body_cmd_cfg_field"), _REW_NS)


def _exec_class(name):
    import ast  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    src = OMNI_REWARDS_PY.read_text()
    node = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == name][0]
    code = textwrap.dedent(ast.get_source_segment(src, node))
    exec(compile("from __future__ import annotations\n" + code, str(OMNI_REWARDS_PY), "exec"), _REW_NS)
    return _REW_NS[name]


RewardYawWindowExp = _exec_class("RewardYawWindowExp")
RewardYawProgressTent = _exec_class("RewardYawProgressTent")


def _rew_env(n=4, w_max=0.21):
    env = RH._Env(n)
    env.step_dt = DT
    env.cfg = types.SimpleNamespace(commands=types.SimpleNamespace(
        base_velocity=types.SimpleNamespace(body_w_max=w_max)))
    return env


def test_the_class_terms_call_the_pure_incomes():
    assert (RewardYawWindowExp.WINDOW_S, RewardYawWindowExp.SIGMA, RewardYawProgressTent.WINDOW_S) == (2.0, 0.1, 2.0)
    env = _rew_env()
    win = RewardYawWindowExp(types.SimpleNamespace(params={"window_s": 2.0, "sigma": 0.1}), env)
    tent = RewardYawProgressTent(types.SimpleNamespace(params={"window_s": 2.0, "clip": None,
                                                               "command_name": "base_velocity"}), env)
    assert win.window.n_steps == tent.window.n_steps == 100 and tent.w_max == 0.21
    assert tent.window_t == pytest.approx(2.0)
    ref = yw.YawWindow(4, 100)
    g = torch.Generator().manual_seed(4)
    for _ in range(130):
        env.command_manager.cmd[:, 2] = torch.tensor([0.0, 0.05, -0.15, 0.21])
        env.asset.data.root_ang_vel_b[:, 2] = torch.randn(4, generator=g) * 0.2
        a = win(env, asset_cfg=RH.ROBOT, parkour_name="base_parkour", command_name="base_velocity",
                window_s=2.0, sigma=0.1)
        b = tent(env, command_name="base_velocity", asset_cfg=RH.ROBOT, clip=None, window_s=2.0)
        sa, sc = ref.push(env.asset.data.root_ang_vel_b[:, 2] * DT, env.command_manager.cmd[:, 2] * DT)
        assert torch.equal(a, yw.yaw_window_income(sa, sc, 0.1))
        assert torch.equal(b, yw.yaw_tent_income(sa, sc, 100 * DT, 0.21))
    assert float(b[0]) == 0.0, "the straight row's tent is exactly zero"
    win.reset([1])
    tent.reset([1])
    assert torch.equal(win.window.ach[1], torch.zeros(100)) and torch.equal(tent.window.cmd[1], torch.zeros(100))


def test_the_tent_reads_w_max_off_the_body_command_cfg_or_refuses():
    env = _rew_env()
    env.cfg.commands.base_velocity = types.SimpleNamespace(ang_vel_max=0.3)    # a WORLD command cfg
    with pytest.raises(RuntimeError, match="body_w_max"):
        RewardYawProgressTent(types.SimpleNamespace(params={"clip": None}), env)
    assert RewardYawProgressTent(types.SimpleNamespace(params={"clip": 0.4}), env).w_max == 0.4


@pytest.mark.skipif(RH.track_ang_vel_z_exp is None, reason="isaaclab's rewards.py not locatable")
def test_straight_gated_ang_vel_is_the_stock_term_on_straight_rows_and_zero_on_turns():
    gated = _REW_NS["track_ang_vel_z_exp_straight_gated"]
    env = RH._Env(6)
    env.command_manager.cmd[:, 2] = torch.tensor([0.0, 0.0, 0.05, -0.2, 0.0, 0.1])
    env.asset.data.root_ang_vel_b[:, 2] = torch.tensor([0.0, 0.3, 0.05, 0.1, -0.2, 0.0])
    got = gated(env, std=0.5, command_name="base_velocity", asset_cfg=RH.ROBOT)
    stock = RH.track_ang_vel_z_exp(env, std=0.5, command_name="base_velocity", asset_cfg=RH.ROBOT)
    straight = env.command_manager.cmd[:, 2] == 0.0
    assert torch.equal(got[straight], stock[straight])
    assert torch.equal(got[~straight], torch.zeros(int((~straight).sum())))


# ============================================================ the clock seam (spec §5)
def test_clock_turn_floor_default_is_bitwise_the_world_path():
    g = torch.Generator().manual_seed(1)
    for _ in range(4):
        cmd = (torch.rand(257, 3, generator=g) - 0.5) * 0.8
        cmd[::5, 2] = 0.0
        for r in (0.0, 1.2):
            old = torch.maximum(torch.linalg.norm(cmd[:, :2], dim=-1), r * cmd[:, 2].abs())
            assert torch.equal(clock_command_speed(cmd, True, r), old)
            assert torch.equal(clock_command_speed(cmd, True, r, 0.0), old)
        assert torch.equal(clock_command_speed(cmd, False, 1.2, 0.21), cmd[:, 0].abs()), "forward branch inert"


def test_clock_turn_floor_keeps_a_slow_turn_moving():
    from crab_hex_dimensions import CLOCK_CMD_STOP_M_S  # noqa: PLC0415

    cmd = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, -0.08], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
                        [0.22, 0.0, 0.05], [0.0, 0.0, 0.3]])
    world = clock_command_speed(cmd, True, 1.2)
    body = clock_command_speed(cmd, True, 1.2, 0.21)
    assert torch.allclose(world, torch.tensor([0.06, 0.096, 0.0, 0.3, 0.22, 0.36]))
    assert torch.allclose(body, torch.tensor([0.21, 0.21, 0.0, 0.3, 0.22, 0.36]))
    assert bool((body[[0, 1, 4]] > CLOCK_CMD_STOP_M_S).all()) and not bool(world[0] > CLOCK_CMD_STOP_M_S)
    assert bool((body[[0, 1]] > 0.12).all()), "the gait gates' 0.12 opens on a slow turn"


def test_the_clock_seam_callers_read_the_floor_with_a_zero_default():
    """``_clock_cmd_speed`` (every clock-speed gait gate) and ``_cmd_gate``, exec'd from the shipped
    source: world (no attribute) bitwise as before; body opens the 0.12 gate at 0.05 rad/s."""
    ns = {"torch": torch, "clock_command_speed": clock_command_speed}
    RH._exec_module_level(LIB_REWARDS_PY, ("_clock_cmd_speed", "_cmd_gate"), ns)
    cmd = torch.tensor([[0.0, 0.0, 0.05], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    world = types.SimpleNamespace(cfg=types.SimpleNamespace(omni_command=True, omni_clock_turn_radius=1.2))
    body = types.SimpleNamespace(cfg=types.SimpleNamespace(omni_command=True, omni_clock_turn_radius=1.2,
                                                           omni_clock_turn_floor=0.21))
    assert torch.equal(ns["_clock_cmd_speed"](world, cmd), clock_command_speed(cmd, True, 1.2))
    assert ns["_cmd_gate"](world, cmd, 0.12, signed=False).tolist() == [False, True, False, True]
    assert ns["_cmd_gate"](body, cmd, 0.12, signed=False).tolist() == [True, True, False, True]
    actions = (CLOCK_MDP / "parkour_actions.py").read_text()
    assert 'float(getattr(self._env.cfg, "omni_clock_turn_floor", 0.0))' in actions


# ============================================================ the knob layer (spec §1)
def test_body_knob_names_parsers_and_exports():
    assert (knobs.COMMAND_FRAME_VAR, knobs.BODY_ENVELOPE_VAR, knobs.BODY_CAP_VAR) == (
        "KRABBY_OMNI_COMMAND_FRAME", "KRABBY_OMNI_BODY_ENVELOPE", "KRABBY_OMNI_BODY_CAP")
    for name in ("COMMAND_FRAME_VAR", "BODY_ENVELOPE_VAR", "BODY_CAP_VAR", "BODY_KNOB_PARSERS",
                 "body_frame_env_refusals", "parse_command_frame"):
        assert name in knobs.__all__, name
    assert set(knobs.BODY_KNOB_PARSERS) == {knobs.COMMAND_FRAME_VAR, knobs.BODY_ENVELOPE_VAR, knobs.BODY_CAP_VAR,
                                            knobs.BODY_TENT_RATE_VAR}
    assert knobs.parse_command_frame(None) == knobs.parse_command_frame(" ") == "world"
    assert knobs.parse_command_frame(" Body ") == "body" and knobs.parse_command_frame("world") == "world"
    for bad in ("robot", "1", "b"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_COMMAND_FRAME"):
            knobs.parse_command_frame(bad)
    assert knobs.parse_body_envelope_knob("") is None and knobs.parse_body_cap_knob(" ") is None
    assert knobs.parse_body_envelope_knob(ENV_TXT) == (0.05, 0.167, 0.21)
    assert knobs.parse_body_cap_knob(CAP_TXT) == ENVELOPE.cap
    assert bk.KNOB_OFF_VALUES == {"KRABBY_OMNI_COMMAND_FRAME": "world"}


BODY_ENV = {"KRABBY_OMNI_COMMAND_FRAME": "body", "KRABBY_OMNI_BODY_ENVELOPE": ENV_TXT,
            "KRABBY_OMNI_BODY_CAP": CAP_TXT}


@pytest.mark.parametrize("var, value", [
    ("KRABBY_OMNI_FACING", "0.2:0.6"),
    ("KRABBY_OMNI_FACING_OBS_CLIP", "0.6"),
    ("KRABBY_OMNI_FACING_OBS_ENCODING", "raw"),
    ("KRABBY_OMNI_FACING_OBS_ENCODING", "sincos"),
    ("KRABBY_OMNI_FACING_TURN_W", "0.4"),
    ("KRABBY_OMNI_TRACK_FACING_SIGMA", "0.5"),
    ("KRABBY_OMNI_RELAY_HYST", "1"),
    ("KRABBY_OMNI_ANG_TURN_ONLY", "1"),
    ("KRABBY_RSI_FRAC", "0.2"),
    ("KRABBY_OMNI_CMD_POOL", "stand:0.2,turn:0.8:0.5"),
    ("KRABBY_OMNI_STROKE_W", "0.5"),
    ("KRABBY_OMNI_CENTRING_W", "0.5"),
    ("KRABBY_OMNI_BODY_ENVELOPE", ""),
    ("KRABBY_OMNI_BODY_CAP", ""),
])
def test_every_body_mode_refusal(var, value):
    env = dict(BODY_ENV, **{var: value})
    reasons = knobs.body_frame_env_refusals(env)
    assert len(reasons) == 1 and var in reasons[0], reasons


@pytest.mark.parametrize("var, value", [
    ("KRABBY_OMNI_FACING_OBS_ENCODING", "none"), ("KRABBY_OMNI_FACING_OBS_ENCODING", ""),
    ("KRABBY_OMNI_FACING_TURN_W", "0"), ("KRABBY_OMNI_RELAY_HYST", "0"), ("KRABBY_OMNI_ANG_TURN_ONLY", "0"),
    ("KRABBY_RSI_FRAC", "0"), ("KRABBY_OMNI_CMD_POOL", POOL_C6[0]), ("KRABBY_OMNI_CMD_POOL", ""),
    ("KRABBY_OMNI_FACING", ""), ("KRABBY_OMNI_ANG_VEL_MAX", "0.3"), ("KRABBY_YAW_PROGRESS_W", "2.0"),
    ("KRABBY_OMNI_STROKE_W", "0"), ("KRABBY_OMNI_CENTRING_W", ""),
])
def test_the_off_spellings_pass_in_body_mode(var, value):
    assert knobs.body_frame_env_refusals(dict(BODY_ENV, **{var: value})) == []


@pytest.mark.parametrize("env, var", [
    ({"KRABBY_OMNI_BODY_ENVELOPE": ENV_TXT}, "KRABBY_OMNI_BODY_ENVELOPE"),
    ({"KRABBY_OMNI_BODY_CAP": CAP_TXT}, "KRABBY_OMNI_BODY_CAP"),
    ({"KRABBY_OMNI_FACING_OBS_ENCODING": "none"}, "KRABBY_OMNI_FACING_OBS_ENCODING"),
    ({"KRABBY_OMNI_COMMAND_FRAME": "world", "KRABBY_OMNI_FACING_OBS_ENCODING": "NONE"},
     "KRABBY_OMNI_FACING_OBS_ENCODING"),
])
def test_every_world_mode_refusal(env, var):
    reasons = knobs.body_frame_env_refusals(env)
    assert len(reasons) == 1 and var in reasons[0], reasons
    assert knobs.body_frame_env_refusals({"KRABBY_OMNI_FACING_OBS_ENCODING": "sincos"}) == []


# -- the whole knob block, end to end, on a cfg shaped like the real omni env cfg
def _f_facing(*a, **k):
    """Stand-in for ``crab_hex_omni_rewards.reward_facing_exp``."""


def _f_progress(*a, **k):
    """Stand-in for ``crab_hex_omni_rewards.reward_yaw_progress``."""


def _f_stock_ang(*a, **k):
    """Stand-in for isaaclab's ``track_ang_vel_z_exp``."""


def _f_straight(*a, **k):
    """Stand-in for ``crab_hex_omni_rewards.track_ang_vel_z_exp_straight_gated``."""


def _f_gated(*a, **k):
    """Stand-in for ``crab_hex_omni_rewards.track_ang_vel_z_exp_turn_gated``."""


class _FakeWindowTerm:
    WINDOW_S = yw.YAW_WINDOW_S
    SIGMA = yw.YAW_WINDOW_SIGMA


class _FakeTentTerm:
    WINDOW_S = yw.YAW_WINDOW_S


REWARD_FUNCS = {"facing_func": _f_facing, "progress_func": _f_progress, "stock_ang": _f_stock_ang,
                "window_cls": _FakeWindowTerm, "tent_cls": _FakeTentTerm, "straight_func": _f_straight}


class _ObsFunc:
    OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}


def _term(func, weight, **params):
    return types.SimpleNamespace(func=func, weight=weight, params=dict(params))


def _full_cfg(**cmd_over):
    cmd = _world_cfg(speed_ellipse=ELLIPSE, **cmd_over)
    rewards = types.SimpleNamespace(
        reward_tracking_yaw=_term(_f_facing, 0.2, asset_cfg="robot", parkour_name="base_parkour",
                                  command_name="base_velocity"),
        reward_yaw_progress=_term(_f_progress, 1.0, command_name="base_velocity", asset_cfg="robot", clip=None),
        track_ang_vel_z_exp=_term(_f_stock_ang, 1.0, command_name="base_velocity", std=0.5),
        reward_facing_turn=_term(None, 0.0, command_name="base_velocity"),
        reward_forward_progress_along_command=_term(None, 0.6),
        reward_clock_schedule=_term(None, 1.0, command_name="base_velocity", combine="product"),
    )
    obs_term = _term(_ObsFunc, 0.0, asset_cfg="robot", history_length=10, facing_obs_encoding="raw")
    cfg = types.SimpleNamespace(
        omni_clock_turn_radius=1.2,
        commands=types.SimpleNamespace(base_velocity=cmd),
        rewards=rewards,
        observations=types.SimpleNamespace(policy=types.SimpleNamespace(extreme_parkour_observations=obs_term)),
    )
    return cfg


def _apply(monkeypatch, cfg, environ):
    real_cmd = knobs._apply_command_knobs
    monkeypatch.setattr(knobs, "_apply_command_knobs",
                        lambda c, e, **kw: real_cmd(c, e, cmd_cls=om.CrabHexOmniCommandCfg))
    real_turn_only = knobs._apply_ang_turn_only_knob
    monkeypatch.setattr(knobs, "_apply_ang_turn_only_knob",
                        lambda c, e, **kw: real_turn_only(c, e, gated_func=_f_gated, stock_func=_f_stock_ang))
    monkeypatch.setattr(knobs, "_body_command_api", lambda: (om.CrabHexOmniBodyCommandCfg, om.omni_walk_floor))
    monkeypatch.setattr(knobs, "_body_reward_funcs", lambda: dict(REWARD_FUNCS))
    ring = types.ModuleType("parkour_tasks.crab_hex_omni_task.mdp.omni_terrain_knobs")
    ring.apply_omni_ring_knobs = lambda c, env=None, classes=None: None
    monkeypatch.setitem(sys.modules, "parkour_tasks.crab_hex_omni_task.mdp.omni_terrain_knobs", ring)
    knobs.apply_omni_knobs(cfg, environ)
    return cfg


def _plain(obj):
    """A cfg object as nested plain data (the stub cfg classes compare by identity)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_plain(v) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type) and not callable(obj):
        return (type(obj).__name__, _plain(vars(obj)))
    return obj


def _snapshot(cfg):
    return copy.deepcopy({
        "cmd": _plain(dict(vars(cfg.commands.base_velocity))),
        "cmd_cls": type(cfg.commands.base_velocity),
        "rewards": {k: (v.func, v.weight, dict(v.params)) for k, v in vars(cfg.rewards).items()},
        "obs": dict(cfg.observations.policy.extreme_parkour_observations.params),
        "top": {k: v for k, v in vars(cfg).items() if k not in ("commands", "rewards", "observations")},
    })


def test_world_mode_writes_nothing(monkeypatch):
    environ = {"KRABBY_OMNI_ANG_VEL_MAX": "0.3", "KRABBY_OMNI_COMMAND_FRAME": "world"}
    a = _snapshot(_apply(monkeypatch, _full_cfg(), {"KRABBY_OMNI_ANG_VEL_MAX": "0.3"}))
    b = _snapshot(_apply(monkeypatch, _full_cfg(), environ))
    c = _snapshot(_full_cfg())
    assert a == b
    assert a["rewards"] == c["rewards"] and a["obs"] == c["obs"] and a["cmd_cls"] is c["cmd_cls"]
    assert "omni_clock_turn_floor" not in a["top"]
    with pytest.raises(RuntimeError, match="world"):
        _apply(monkeypatch, _full_cfg(), {"KRABBY_OMNI_ANG_VEL_MAX": "0.3", "KRABBY_OMNI_BODY_CAP": CAP_TXT})
    with pytest.raises(RuntimeError, match="none"):
        _apply(monkeypatch, _full_cfg(), {"KRABBY_OMNI_ANG_VEL_MAX": "0.3",
                                           "KRABBY_OMNI_FACING_OBS_ENCODING": "none"})


def test_body_mode_end_to_end(monkeypatch, capsys):
    cfg = _full_cfg(cmd_pool=POOL_C6[0], slot_probs=POOL_C6[1])
    world_cmd = cfg.commands.base_velocity
    world_cmd.stand_hold = True              # a motion knob's instance-only attribute
    world_cmd.eval_slot_latch = True
    before = dict(vars(world_cmd))
    environ = dict(BODY_ENV, **{"KRABBY_OMNI_ANG_VEL_MAX": "0.2",    # 1.2 x 0.2 < 0.25: fails the
                                "KRABBY_OMNI_ANG_VEL_CLIP": "0.1",   # world clock-radius rule
                                "KRABBY_OMNI_FACING_OBS_ENCODING": "none", "KRABBY_YAW_PROGRESS_W": "2.0",
                                "KRABBY_RSI_FRAC": "0"})
    _apply(monkeypatch, cfg, environ)
    out = capsys.readouterr().out
    assert "clock-radius rule" in out and "SKIPPED" in out and "KRABBY_OMNI_COMMAND_FRAME=body consumed" in out
    body = cfg.commands.base_velocity
    assert type(body) is om.CrabHexOmniBodyCommandCfg and body.class_type is om.CrabHexOmniBodyCommand
    for key, value in before.items():                # every field + extra carried over, unchanged
        if key in ("class_type", "ang_vel_max"):
            continue
        assert getattr(body, key) == value, key
    assert body.ang_vel_max == 0.2 and body.clips.ang_vel_clip == 0.1, "the command knobs ran first"
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_CLOCK_TURN_R"):        # ... which world mode refuses
        _apply(monkeypatch, _full_cfg(), {k: v for k, v in environ.items() if k not in BODY_ENV
                                          and k != "KRABBY_OMNI_FACING_OBS_ENCODING"})
    assert (body.body_w_min, body.body_ceiling, body.body_w_max, body.body_cap) == tuple(ENVELOPE)
    assert body.stand_hold is True and body.eval_slot_latch is True
    obs = cfg.observations.policy.extreme_parkour_observations.params
    assert obs["facing_obs_encoding"] == "none" and "facing_obs_clip" not in obs
    r = cfg.rewards
    assert r.reward_tracking_yaw.func is _FakeWindowTerm and r.reward_tracking_yaw.weight == 0.2
    assert r.reward_tracking_yaw.params == {"asset_cfg": "robot", "parkour_name": "base_parkour",
                                            "command_name": "base_velocity", "window_s": 2.0, "sigma": 0.1}
    assert r.reward_yaw_progress.func is _FakeTentTerm and r.reward_yaw_progress.weight == 2.0
    assert r.reward_yaw_progress.params == {"command_name": "base_velocity", "asset_cfg": "robot", "clip": None,
                                            "window_s": 2.0}
    assert r.track_ang_vel_z_exp.func is _f_straight and r.track_ang_vel_z_exp.params["std"] == 0.5
    assert r.reward_facing_turn.weight == 0.0
    assert cfg.omni_clock_turn_floor == pytest.approx(0.21)
    # ... and the realised cfg builds the real body term, which samples inside the envelope
    term, _r = _body_term(body, n=512)
    _resample(term, 12)
    vx, vy, wz = _cmd(term)
    assert not bool(mm.body_cmd_violation_mask(vx, vy, wz, ENVELOPE, ELLIPSE, FLOOR).any())
    with pytest.raises(RuntimeError, match="twice"):
        knobs._apply_command_frame(cfg, environ, body_api=(om.CrabHexOmniBodyCommandCfg, om.omni_walk_floor),
                                   reward_funcs=dict(REWARD_FUNCS))


@pytest.mark.parametrize("extra, match", [
    ({"KRABBY_OMNI_FACING": "0.2:0.6"}, "KRABBY_OMNI_FACING"),
    ({"KRABBY_OMNI_FACING_TURN_W": "0.3"}, "KRABBY_OMNI_FACING_TURN_W"),
    ({"KRABBY_OMNI_RELAY_HYST": "1"}, "KRABBY_OMNI_RELAY_HYST"),
    ({"KRABBY_OMNI_ANG_TURN_ONLY": "1"}, "KRABBY_OMNI_ANG_TURN_ONLY"),
    ({"KRABBY_OMNI_BODY_CAP": ""}, "KRABBY_OMNI_BODY_CAP"),
])
def test_body_mode_refuses_through_the_entry_point(monkeypatch, extra, match):
    environ = dict(BODY_ENV, **{"KRABBY_OMNI_ANG_VEL_MAX": "0.3"}, **extra)
    with pytest.raises((RuntimeError, ValueError), match=match):
        _apply(monkeypatch, _full_cfg(), environ)


def test_body_mode_refuses_the_realised_cfg_state():
    api = (om.CrabHexOmniBodyCommandCfg, om.omni_walk_floor)
    cfg = _full_cfg()
    cfg.events = types.SimpleNamespace(rsi_reference_reset=object())
    with pytest.raises(RuntimeError, match="rsi_reference_reset"):
        knobs._apply_command_frame(cfg, BODY_ENV, body_api=api, reward_funcs=dict(REWARD_FUNCS))
    cfg = _full_cfg()
    cfg.rewards.reward_facing_turn.weight = 0.4
    with pytest.raises(RuntimeError, match="reward_facing_turn"):
        knobs._apply_command_frame(cfg, BODY_ENV, body_api=api, reward_funcs=dict(REWARD_FUNCS))
    cfg = _full_cfg()
    cfg.observations.policy.extreme_parkour_observations.params["facing_obs_clip"] = 0.6
    with pytest.raises(RuntimeError, match="facing_obs_clip"):
        knobs._apply_command_frame(cfg, BODY_ENV, body_api=api, reward_funcs=dict(REWARD_FUNCS))
    cfg = _full_cfg()
    cfg.rewards.track_ang_vel_z_exp.func = _f_straight          # someone else owns the func
    with pytest.raises(RuntimeError, match="refusing to clobber"):
        knobs._apply_command_frame(cfg, BODY_ENV, body_api=api, reward_funcs=dict(REWARD_FUNCS))


# ============================================================ the presets (spec §6)
def test_body_delta_is_pinned_key_by_key():
    assert phases.BODY_DELTA == {
        "KRABBY_OMNI_COMMAND_FRAME": "body",
        "KRABBY_OMNI_FACING_OBS_ENCODING": "none",
        "KRABBY_YAW_PROGRESS_W": "2.0",
        "KRABBY_YAW_W": "0.2",
        "KRABBY_RSI_FRAC": "0",
        "KRABBY_OMNI_FACING": None,
        "KRABBY_OMNI_FACING_OBS_CLIP": None,
    }
    assert phases.BODY_PHASE_NAMES == ("1a_b", "1b_b", "2a_b")
    assert not set(phases.BODY_PHASE_NAMES) & set(phases.PHASE_NAMES)


@pytest.mark.parametrize("base", ["1a", "1b", "2a"])
def test_each_body_preset_is_its_base_plus_the_delta(base):
    spec, twin = phases.PHASES[base], phases.PHASES[f"{base}_b"]
    want = dict(spec.env)
    for key in ("KRABBY_OMNI_FACING", "KRABBY_OMNI_FACING_OBS_CLIP"):
        want.pop(key, None)
    want.update({"KRABBY_OMNI_COMMAND_FRAME": "body", "KRABBY_OMNI_FACING_OBS_ENCODING": "none",
                 "KRABBY_YAW_PROGRESS_W": "2.0", "KRABBY_YAW_W": "0.2", "KRABBY_RSI_FRAC": "0"})
    assert twin.env == want
    assert (twin.task, twin.experiment, twin.iterations, twin.teacher_mode, twin.kind, twin.window) == (
        spec.task, spec.experiment, spec.iterations, spec.teacher_mode, spec.kind, spec.window)
    assert twin.name == f"{base}_b"
    assert twin.resume_from == {"1a": None, "1b": "1a_b", "2a": None}[base], "never a WORLD head"
    # reachable, and refusing to build without the envelope (the campaign overlay supplies it)
    ev = phases.phase_env(f"{base}_b", "A15+B")
    assert ev["KRABBY_OMNI_PHASE"] == f"{base}_b" and ev["KRABBY_OMNI_COMMAND_FRAME"] == "body"
    reasons = knobs.body_frame_env_refusals(ev)
    assert len(reasons) == 2 and all("REQUIRED" in r for r in reasons), reasons
    ev.update({"KRABBY_OMNI_BODY_ENVELOPE": ENV_TXT, "KRABBY_OMNI_BODY_CAP": CAP_TXT})
    assert knobs.body_frame_env_refusals(ev) == []


def test_activate_phase_expands_a_body_preset():
    environ = {"KRABBY_OMNI_PHASE": "2a_b"}
    applied = phases.activate_phase(environ)
    assert environ["KRABBY_OMNI_COMMAND_FRAME"] == "body" and applied["KRABBY_HEX_TEACHER_MODE"] == "2a"
    assert "KRABBY_OMNI_FACING" not in environ and "KRABBY_OMNI_FACING_OBS_CLIP" not in environ


def test_world_presets_are_head(tmp_path):
    """Every world preset is the HEAD ``60d7f45d`` one (the RSI bank path normalised: the HEAD copy
    is loaded from a temp dir, and the bank path is derived from the module's own location)."""
    rel = OMNI_PHASES_PY.relative_to(REPO).as_posix()
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{HEAD_SHA}:{rel}"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git show unavailable ({exc})")
    path = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "crab_hex_omni_phases_head.py"   # parents[6] exists
    path.parent.mkdir(parents=True)
    path.write_text(src)
    head = _load(path, "crab_hex_omni_phases_head_60d7f45d")
    assert head.PHASE_NAMES == phases.PHASE_NAMES
    for name in phases.PHASE_NAMES:
        h, n = head.PHASES[name], phases.PHASES[name]
        h_env = dict(h.env, KRABBY_RSI_BANK=n.env["KRABBY_RSI_BANK"])
        assert Path(h.env["KRABBY_RSI_BANK"]).name == Path(n.env["KRABBY_RSI_BANK"]).name
        assert n.env == h_env, name
        assert (n.task, n.experiment, n.iterations, n.resume_from, n.teacher_mode, n.eval_scenarios) == (
            h.task, h.experiment, h.iterations, h.resume_from, h.teacher_mode, h.eval_scenarios), name


# ============================================================ the driver (spec §7)
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES, "run_candidates_omni_body_cmd_test")


def test_driver_whitelist_parsers_and_bases(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for key in ("KRABBY_OMNI_COMMAND_FRAME", "KRABBY_OMNI_BODY_ENVELOPE", "KRABBY_OMNI_BODY_CAP"):
        assert key in whitelist and key in parsers and key in rc.OVERLAY_DUMP_CHECKS, key
        assert key not in rc.AGENT_DUMP_KEYS, f"{key} is env side"
    for base in ("1a_b", "1b_b", "2a_b"):
        assert base in rc.PHASES_ALLOWED_BASES
    assert set(("1a", "1b", "1c", "2a", "2b")) <= set(rc.PHASES_ALLOWED_BASES)


@pytest.mark.parametrize("key, good, bad", [
    ("KRABBY_OMNI_COMMAND_FRAME", ["body", "world", ""], ["robot", "1"]),
    ("KRABBY_OMNI_BODY_ENVELOPE", [ENV_TXT, ""], ["0.05:0.2", "0.2:0.1:0.3", "x"]),
    ("KRABBY_OMNI_BODY_CAP", [CAP_TXT, "0@0.2", ""], ["0.1@0.2", "0@0.1,0.2@0.15", "0@x"]),
])
def test_driver_round_trips(rc, key, good, bad):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for value in good:
        reasons = []
        rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
        assert reasons == [], (value, reasons)
    for value in bad:
        reasons = []
        rc._check_knob(key, value, "x", whitelist, parsers, reasons, set())
        assert any("does not round-trip" in r for r in reasons), (value, reasons)


def test_driver_composition_check(rc):
    reasons = []
    rc._check_body_frame_composition(dict(phases.phase_env("1a_b")), "stage s", reasons)
    assert len(reasons) == 2 and all(r.startswith("stage s: ") and "REQUIRED" in r for r in reasons)
    env = dict(phases.phase_env("1a_b"), **BODY_ENV)
    reasons = []
    rc._check_body_frame_composition(env, "s", reasons)
    assert reasons == []
    reasons = []
    rc._check_body_frame_composition(dict(env, KRABBY_OMNI_FACING="0.2:0.6"), "s", reasons)
    assert len(reasons) == 1 and "KRABBY_OMNI_FACING" in reasons[0]
    reasons = []
    rc._check_body_frame_composition(dict(phases.phase_env("1c"), KRABBY_OMNI_BODY_CAP=CAP_TXT), "s", reasons)
    assert len(reasons) == 1 and "world" in reasons[0]
    reasons = []
    rc._check_body_frame_composition(dict(phases.phase_env("1c")), "s", reasons)
    assert reasons == [], "every world preset composes clean"


# -- synthetic params/env.yaml dumps, written like the trainer's (yaml.dump of class_to_dict)
_OMNI_CMD = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command"
_OMNI_REW = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_rewards"


def _dump(frame="world", **mut) -> str:
    import yaml  # noqa: PLC0415

    body = frame == "body"
    cmd = {"class_type": f"{_OMNI_CMD}:{'CrabHexOmniBodyCommand' if body else 'CrabHexOmniCommand'}",
           "resampling_time_range": (10.0, 10.0), "asset_name": "robot",
           "clips": {"lin_vel_clip": 0.2, "ang_vel_clip": 0.2}, "slot_probs": (0.15, 0.55, 0.1, 0.2),
           "speed_ellipse": ELLIPSE, "facing_delta": (0.2, 0.6), "ang_vel_max": 0.3, "dir_band": None,
           "cmd_pool": POOL_C6[0]}
    if body:
        cmd.update(body_w_min=0.05, body_ceiling=0.167, body_w_max=0.21, body_cap=ENVELOPE.cap)
    params = {"history_length": 10, "facing_obs_encoding": "none" if body else "raw"}
    if not body:
        params["facing_obs_clip"] = 0.6
    obs = {"policy": {"concatenate_terms": True, "extreme_parkour_observations": {
        "func": "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_observations:CrabHexOmniParkourObservations",
        "params": params, "clip": (-100, 100)}}}
    funcs = {"reward_tracking_yaw": "RewardYawWindowExp" if body else "reward_facing_exp",
             "reward_yaw_progress": "RewardYawProgressTent" if body else "reward_yaw_progress",
             "track_ang_vel_z_exp": "track_ang_vel_z_exp_straight_gated" if body else None}
    rewards = {
        "reward_tracking_yaw": {"func": f"{_OMNI_REW}:{funcs['reward_tracking_yaw']}",
                                "params": {"command_name": "base_velocity"}, "weight": 0.2},
        "reward_yaw_progress": {"func": f"{_OMNI_REW}:{funcs['reward_yaw_progress']}",
                                "params": {"command_name": "base_velocity", "clip": None}, "weight": 2.0},
        "track_ang_vel_z_exp": {"func": (f"{_OMNI_REW}:track_ang_vel_z_exp_straight_gated" if body
                                         else "isaaclab.envs.mdp.rewards:track_ang_vel_z_exp"),
                                "params": {"command_name": "base_velocity", "std": 0.5}, "weight": 1.0},
        "reward_facing_turn": {"func": f"{_OMNI_REW}:reward_facing_exp_turn_slot", "params": {}, "weight": 0.0},
    }
    events = {"reset_root_state": {"func": "x:reset_root_state_ring", "params": {"offset": 0.0}}}
    if not body:
        events["rsi_reference_reset"] = {"func": "x:reset_from_reference_states_omni",
                                         "params": {"fraction": 0.2}, "mode": "reset"}
    doc = {"seed": 18, "scene": {"num_envs": 256}, "commands": {"base_velocity": cmd}, "observations": obs,
           "rewards": rewards, "events": events, "omni_command": True, "omni_clock_turn_radius": 1.2}
    if body:
        doc["omni_clock_turn_floor"] = 0.21
    for path, value in mut.items():
        node = doc
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        if value is _DROP:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


_DROP = object()


def test_dump_checks_both_directions(rc):
    world, body = _dump("world"), _dump("body")
    chk = rc.OVERLAY_DUMP_CHECKS
    assert chk["KRABBY_OMNI_COMMAND_FRAME"](world, "world") and chk["KRABBY_OMNI_COMMAND_FRAME"](world, "")
    assert not chk["KRABBY_OMNI_COMMAND_FRAME"](world, "body")
    assert chk["KRABBY_OMNI_COMMAND_FRAME"](body, "body")
    assert not chk["KRABBY_OMNI_COMMAND_FRAME"](body, "world")
    assert chk["KRABBY_OMNI_BODY_ENVELOPE"](body, ENV_TXT) and not chk["KRABBY_OMNI_BODY_ENVELOPE"](body, "")
    assert not chk["KRABBY_OMNI_BODY_ENVELOPE"](body, "0.05:0.167:0.22")
    assert chk["KRABBY_OMNI_BODY_ENVELOPE"](world, "") and not chk["KRABBY_OMNI_BODY_ENVELOPE"](world, ENV_TXT)
    assert chk["KRABBY_OMNI_BODY_CAP"](body, CAP_TXT) and not chk["KRABBY_OMNI_BODY_CAP"](body, "")
    assert not chk["KRABBY_OMNI_BODY_CAP"](body, "0@0.21,0.22@0.15,0.30@0.05")
    assert not chk["KRABBY_OMNI_BODY_CAP"](body, "0@0.21,0.22@0.15")
    assert chk["KRABBY_OMNI_BODY_CAP"](world, "") and not chk["KRABBY_OMNI_BODY_CAP"](world, CAP_TXT)


@pytest.mark.parametrize("mut", [
    {"commands__base_velocity__class_type": f"{_OMNI_CMD}:CrabHexOmniCommand"},
    {"commands__base_velocity__body_w_max": _DROP},
    {"commands__base_velocity__body_cap": _DROP},
    {"observations__policy__extreme_parkour_observations__params__facing_obs_encoding": "raw"},
    {"observations__policy__extreme_parkour_observations__params__facing_obs_clip": 0.6},
    {"rewards__reward_tracking_yaw__func": f"{_OMNI_REW}:reward_facing_exp"},
    {"rewards__reward_yaw_progress__func": f"{_OMNI_REW}:reward_yaw_progress"},
    {"rewards__track_ang_vel_z_exp__func": "isaaclab.envs.mdp.rewards:track_ang_vel_z_exp"},
    {"rewards__reward_facing_turn__weight": 0.3},
    {"events__rsi_reference_reset": {"func": "x:r", "params": {"fraction": 0.2}}},
    {"omni_clock_turn_floor": _DROP},
])
def test_the_body_frame_check_fails_on_every_missing_change(rc, mut):
    assert not rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_COMMAND_FRAME"](_dump("body", **mut), "body")


@pytest.mark.parametrize("mut", [
    {"commands__base_velocity__class_type": f"{_OMNI_CMD}:CrabHexOmniBodyCommand"},
    {"commands__base_velocity__body_w_min": 0.05},
    {"omni_clock_turn_floor": 0.21},
    {"observations__policy__extreme_parkour_observations__params__facing_obs_encoding": "none"},
])
def test_the_world_frame_check_fails_on_any_body_trace(rc, mut):
    assert not rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_COMMAND_FRAME"](_dump("world", **mut), "world")


@pytest.mark.skipif(not S1_DUMP.exists(), reason="the S1 p2a run's params/env.yaml is not on this machine")
def test_the_real_s1_dump_is_world(rc):
    txt = S1_DUMP.read_text(errors="ignore")
    chk = rc.OVERLAY_DUMP_CHECKS
    assert chk["KRABBY_OMNI_COMMAND_FRAME"](txt, "world") and not chk["KRABBY_OMNI_COMMAND_FRAME"](txt, "body")
    assert chk["KRABBY_OMNI_BODY_ENVELOPE"](txt, "") and not chk["KRABBY_OMNI_BODY_ENVELOPE"](txt, ENV_TXT)
    assert chk["KRABBY_OMNI_BODY_CAP"](txt, "") and not chk["KRABBY_OMNI_BODY_CAP"](txt, CAP_TXT)


# ============================================================ KRABBY_OMNI_BODY_TENT_RATE (2026-10-16, c6_b stop)
def test_tent_rate_knob_parses_and_is_refused_in_world_mode():
    assert knobs.parse_body_tent_rate(None) == "window" and knobs.parse_body_tent_rate(" Instant ") == "instant"
    with pytest.raises(ValueError, match="KRABBY_OMNI_BODY_TENT_RATE"):
        knobs.parse_body_tent_rate("fast")
    assert knobs.body_frame_env_refusals(dict(BODY_ENV, KRABBY_OMNI_BODY_TENT_RATE="instant")) == []
    reasons = knobs.body_frame_env_refusals({"KRABBY_OMNI_BODY_TENT_RATE": "instant"})
    assert len(reasons) == 1 and "KRABBY_OMNI_BODY_TENT_RATE" in reasons[0]
    assert "KRABBY_OMNI_BODY_TENT_RATE" in knobs.BODY_KNOB_PARSERS


def test_instant_income_is_one_sided_at_the_command_and_zero_on_straight_rows():
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_yw_instant", OMNI / "mdp/crab_hex_omni_yaw_window.py")
    yw = _ilu.module_from_spec(spec)
    spec.loader.exec_module(yw)
    wz = torch.tensor([0.0, 0.05, 0.10, 0.30, -0.10, -0.40, 0.08])
    c = torch.tensor([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.0])
    out = yw.yaw_instant_income(wz, c, 0.21)
    assert torch.allclose(out, torch.tensor([0.0, 0.05, 0.10, 0.10, -0.10, -0.21, 0.0]))
    # the right hand is the mirror
    assert torch.allclose(yw.yaw_instant_income(-wz, -c, 0.21), out)
    assert out[-1].item() == 0.0 and not torch.signbit(out[-1])


def test_tent_rate_dump_check_both_directions():
    """The driver's OVERLAY_DUMP_CHECKS entry for KRABBY_OMNI_BODY_TENT_RATE fails on a dump without the change."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_rc_tent", OMNI / "experiments/tools/run_candidates.py")
    rc = _ilu.module_from_spec(spec)
    import sys as _sys
    _sys.modules["_rc_tent"] = rc
    spec.loader.exec_module(rc)
    chk = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_BODY_TENT_RATE"]
    on = "rewards:\n  reward_yaw_progress:\n    params:\n      rate: instant\n      window_s: 2.0\n"
    off = "rewards:\n  reward_yaw_progress:\n    params:\n      window_s: 2.0\n"
    assert chk(on, "instant") and not chk(off, "instant")
    assert chk(off, "") and chk(off, "window") and not chk(on, "window")
