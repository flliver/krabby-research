"""The clock-hide knob (clock-deprecation plan step 2, 2026-09-23), on CPU, without Isaac.

``KRABBY_OMNI_CLOCK_HIDE=1`` zeroes the gait clock's sin/cos (obs dims 15-16 and, through the
history buffer, every history copy -- 22 columns of the 1149-wide vector) and
``KRABBY_OMNI_CLOCK_HIDE_RAMP=t0:t1`` phases that in per episode. Every failure mode is silent:
a clock that is still visible in the history, a ramp that hides episodes before ``t0``, a latch
that re-draws every step, a knob-off run that is no longer bitwise today's. So the REAL omni
observation class runs here on top of the REAL forward ``__call__`` and the REAL base
``reset``, with the harness of ``test_crab_hex_omni_observations.py`` (ast-lifted methods, the
omni class exec'd from its own source as a class):

* knob off: bitwise today's observation, and ``reset`` does nothing but the base reset;
* HIDE: all 22 clock columns are exactly 0 from the first compute after a reset, width 1149;
* composition with ``stand_obs``; the ramp's endpoints, one Bernoulli draw per reset row, the
  forced-hidden tail, the ``Metrics/clock_hidden_frac`` log in both modes;
* a ramp without HIDE is refused (knob and term); the knob module writes only the new params
  (the pure stand-in for "knob-set dumps differ only in the new lines");
* ``build_mirrors`` is unchanged, and X08:c6's state dict loads into the real policy class at the
  HIDE observation width.

Nothing is registered under ``isaaclab.*`` / ``parkour_isaaclab.*``.
"""

import ast
import difflib
import importlib
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
OMNI_OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
BASE_OBS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/observations.py"
CURRICULUMS_PY = TASKS / "crab_hex_forward_task/mdp/curriculums.py"
CLOCK_KNOBS_PY = OMNI / "config/crab_hex/clock_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
OMNI_MIRROR_PY = OMNI / "mdp/crab_hex_omni_mirror.py"
OBS_TESTS_PY = REPO / "tests/unit/test_crab_hex_omni_observations.py"
MIRROR_TESTS_PY = REPO / "tests/unit/test_crab_hex_omni_mirror.py"
RSL_MODULES = REPO / "parkour/scripts/rsl_rl/modules"
X08_C6_DIR = REPO / "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB"
X08_C6 = X08_C6_DIR / "model_7996.pt"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The observation-term harness (stub forward base with the real forward ``__call__`` + hooks),
# reused from its own file under a private name rather than copied.
H = _load(OBS_TESTS_PY, "omni_obs_tests__clock_hide")
curriculums = _load(CURRICULUMS_PY, "curriculums__clock_hide_test")
ck = _load(CLOCK_KNOBS_PY, "clock_knobs__clock_hide_test")

N_PROP, NSCAN, NJ, HISTORY = H.N_PROP, H.NSCAN, H.NJ, H.HISTORY
HIST_BASE = N_PROP + NSCAN + 9 + 5 + 2 * NJ
CLOCK_DIMS = (15, 16)


def _hist_cols(step: int = N_PROP):
    """The clock's columns inside the returned history block (``HISTORY`` frames of ``step``)."""
    base = step + NSCAN + 9 + 5 + 2 * NJ
    return [base + k * step + d for k in range(HISTORY) for d in CLOCK_DIMS]


def _all_clock_cols(step: int = N_PROP):
    return list(CLOCK_DIMS) + _hist_cols(step)


# The real base reset (``ExtremeParkourObservations.reset``: zero the reset rows' history).
_BASE_RESET_NS = {"torch": torch}
exec(compile("from __future__ import annotations\nfrom typing import Sequence\n"
             + H._source_of(BASE_OBS_PY, cls="ExtremeParkourObservations", name="reset"),
             str(BASE_OBS_PY), "exec"), _BASE_RESET_NS)


class _StubBaseWithReset(H._StubForwardObservations):
    reset = _BASE_RESET_NS["reset"]


_NS = dict(H._OMNI_NS)
_NS.update(CrabHexParkourObservations=_StubBaseWithReset, ramp_value=curriculums.ramp_value)
exec(
    compile(
        "from __future__ import annotations\n"
        + H._module_assign(OMNI_OBS_PY, "FACING_OBS_ENCODINGS") + "\n"
        + H._module_assign(OMNI_OBS_PY, "STAND_OBS_DIM") + "\n"
        + H._module_assign(OMNI_OBS_PY, "LPF_OBS_DIM") + "\n\n"
        + H._source_of(OMNI_OBS_PY, cls=None, name="_check_facing_obs_encoding") + "\n\n"
        + H._source_of(OMNI_OBS_PY, cls=None, name="_check_omni") + "\n\n"
        + H._source_of(OMNI_OBS_PY, cls=None, name="CrabHexOmniParkourObservations"),
        str(OMNI_OBS_PY),
        "exec",
    ),
    _NS,
)
OmniObs = _NS["CrabHexOmniParkourObservations"]
LOG_KEY = "Metrics/clock_hidden_frac"


class _Cfg:
    def __init__(self, **params):
        self.params = dict(params)


def _env(n=H.N, phase=None):
    env = H._Env(n, torch.zeros(n, 4), torch.zeros(n))
    env.extras = {}
    g = torch.Generator().manual_seed(11)
    env.action_manager.get_term("joint_pos").clock_phase = (
        torch.rand(n, generator=g) * 6.2 + 0.05 if phase is None else phase
    )
    return env


def _term(n=H.N, **params):
    env = _env(n)
    term = OmniObs(_Cfg(**params) if params else None, env)
    return term, env


def _call(term, env, **params):
    return term(env, None, None, "base_parkour", HISTORY, **params)


@pytest.fixture(autouse=True)
def _no_heading(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)


# ============================================================== knob off: bitwise today
def test_the_call_signature_carries_both_params_with_off_defaults():
    sig = inspect.signature(OmniObs.__call__)
    assert sig.parameters["clock_hide"].default is False
    assert sig.parameters["clock_hide_ramp"].default is None
    assert OmniObs.CLOCK_HIDDEN_FRAC_KEY == LOG_KEY


def test_unset_is_bitwise_today_across_steps_and_resets():
    """No param == ``clock_hide=False``, and both leave the clock columns at sin/cos(phase)."""
    plain, env_a = _term()
    explicit, env_b = _term(clock_hide=False)
    assert plain.clock_hide is False and plain.clock_hide_ramp is None and plain._clock_hidden is None
    phase = env_a.action_manager.get_term("joint_pos").clock_phase
    rng = torch.get_rng_state()
    for step in range(4):
        env_a.common_step_counter = env_b.common_step_counter = step
        if step == 2:
            plain.reset(torch.tensor([0, 3]))
            explicit.reset(torch.tensor([0, 3]))
        a = _call(plain, env_a)
        b = _call(explicit, env_b, clock_hide=False)
        assert torch.equal(a, b), step
        assert torch.equal(a[:, 15], torch.sin(phase)) and torch.equal(a[:, 16], torch.cos(phase))
    # the knob-off reset is the base reset only: no log, no random draw
    assert env_a.extras == {} and env_b.extras == {}
    assert torch.equal(torch.get_rng_state(), rng)
    # and the knob-off hook is the forward hook, verbatim
    fwd = H._FWD_HOOK_NS["_clock_channels"](plain, env_a, phase)
    got = plain._clock_channels(env_a, phase)
    assert torch.equal(got[0], fwd[0]) and torch.equal(got[1], fwd[1])


def test_knob_off_matches_the_five_slot_contract_of_the_shipped_term():
    """The omni term still differs from the forward term only in slots 5-10 with the knob off."""
    omni, env = _term()
    fwd_env = _env()
    fwd = H._StubForwardObservations(None, fwd_env)
    o = _call(omni, env)
    f = fwd(fwd_env, None, None, "base_parkour", HISTORY)
    changed = [c for c in range(N_PROP) if not torch.equal(o[:, c], f[:, c])]
    assert set(changed) <= {5, 6, 7, 8, 9, 10}


# ================================================================= HIDE (no ramp)
def test_hide_zeroes_all_22_clock_columns_from_the_first_compute_after_a_reset():
    term, env = _term(clock_hide=True)
    ref, ref_env = _term()                      # the same term with the clock visible
    assert term.clock_hide is True and term.clock_hide_ramp is None
    assert len(_all_clock_cols()) == 22
    # a mid-episode history full of VISIBLE clock values, then a reset of every row
    term._obs_history_buffer[:, :, 15:17] = 0.7
    term.reset(None)
    env.episode_length_buf = torch.zeros(H.N, dtype=torch.long)
    ref.reset(None)
    ref_env.episode_length_buf = torch.zeros(H.N, dtype=torch.long)
    for step in range(3):
        env.common_step_counter = ref_env.common_step_counter = step
        obs = _call(term, env, clock_hide=True)
        vis = _call(ref, ref_env)
        assert obs.shape == (H.N, 1149)
        for c in _all_clock_cols():
            assert torch.equal(obs[:, c], torch.zeros(H.N)), (step, c)
            assert not torch.signbit(obs[:, c]).any(), "exact +0.0, never -0.0"
        # nothing else moves: every other column is the visible term's
        others = [c for c in range(obs.shape[1]) if c not in set(_all_clock_cols())]
        assert torch.equal(obs[:, others], vis[:, others]), step
        assert not torch.equal(vis[:, 15], torch.zeros(H.N))           # the reference is visible
        assert torch.equal(term._obs_history_buffer[:, :, 15:17], torch.zeros(H.N, HISTORY, 2))
        env.episode_length_buf += 1
        ref_env.episode_length_buf += 1


def test_hide_logs_a_hidden_fraction_of_one_at_every_reset():
    term, env = _term(clock_hide=True)
    rng = torch.get_rng_state()
    term.reset(torch.tensor([1, 2]))
    assert float(env.extras["log"][LOG_KEY]) == 1.0
    env.extras = {"log": {"other": 3.0}}                  # the env re-creates the dict per reset
    term.reset(None)
    assert float(env.extras["log"][LOG_KEY]) == 1.0 and env.extras["log"]["other"] == 3.0
    assert torch.equal(torch.get_rng_state(), rng), "no draw without a ramp"


def test_hide_composes_with_stand_obs():
    phase = torch.linspace(0.3, 5.9, H.N)
    for params in ({"stand_obs": True, "clock_hide": True},
                   {"stand_obs": True, "clock_hide": True, "clock_hide_ramp": (0, 240)}):
        env = _env(phase=phase)
        cmd_term = env.command_manager.get_term("base_velocity")
        cmd_term.slot_id = torch.tensor([0, 1, 2, 3, 0, 1])            # rows 0 and 4 stand
        stand = cmd_term.slot_id == 0
        term = OmniObs(_Cfg(**params), env)
        step = N_PROP + 1
        obs = _call(term, env, **params)
        assert obs.shape == (H.N, step + NSCAN + 9 + 5 + 2 * NJ + HISTORY * step)
        assert torch.equal(obs[:, 17], stand.float())                  # the stand bit survives
        if "clock_hide_ramp" in params:
            # t = 0 < t1: nothing latched hidden, so only the stand rows are zeroed ...
            assert torch.equal(obs[stand, 15], torch.zeros(2))
            assert torch.equal(obs[~stand, 15], torch.sin(phase[~stand]))
            # ... and from t1 on every row is (the returned history block is the PAST, which was
            # visible -- the newest pushed row is the one that must read zero)
            env.common_step_counter = 240
            obs = _call(term, env, **params)
            for c in CLOCK_DIMS:
                assert torch.equal(obs[:, c], torch.zeros(H.N)), (params, c)
            assert torch.equal(term._obs_history_buffer[:, -1, 15:17], torch.zeros(H.N, 2))
            continue
        for c in _all_clock_cols(step):
            assert torch.equal(obs[:, c], torch.zeros(H.N)), (params, c)


# =================================================================== the ramp
def test_ramp_value_endpoints_with_eps_zero():
    rv = curriculums.ramp_value
    t0, t1 = 24000, 48000
    for t in (0, 1, t0 - 1, t0):
        assert rv(t, 0.0, 1.0, t0, t1, eps=0.0) == 0.0, t              # p(t <= t0) == 0, exactly
    assert rv(t0, 0.0, 1.0, t0, t1) == pytest.approx(1e-3)             # (why eps=0 matters)
    for t in (t1, t1 + 1, 10 * t1):
        assert rv(t, 0.0, 1.0, t0, t1, eps=0.0) == 1.0, t
    assert rv((t0 + t1) // 2, 0.0, 1.0, t0, t1, eps=0.0) == pytest.approx(0.5)


class _CountingRand:
    """``torch.rand`` stand-in: records every call's size and returns a fixed uniform."""

    def __init__(self, value):
        self.value, self.calls = value, []

    def __call__(self, n, device=None, **_kw):
        self.calls.append(int(n))
        return torch.full((int(n),), float(self.value))


def test_ramp_hides_nothing_up_to_t0_even_for_a_zero_uniform(monkeypatch):
    term, env = _term(clock_hide=True, clock_hide_ramp=(240, 480))
    rand = _CountingRand(0.0)                          # the smallest uniform torch.rand can return
    monkeypatch.setattr(torch, "rand", rand)
    for t in (0, 100, 240):
        env.common_step_counter = t
        term.reset(None)
        assert not term._clock_hidden.any(), t
        assert float(env.extras["log"][LOG_KEY]) == 0.0
        obs = _call(term, env, clock_hide=True, clock_hide_ramp=(240, 480))
        assert torch.equal(obs[:, 15], torch.sin(env.action_manager.get_term("joint_pos").clock_phase))
    assert rand.calls == [H.N, H.N, H.N]               # one draw per reset row, per reset


def test_ramp_latches_one_draw_per_reset_row_and_holds_it_for_the_episode(monkeypatch):
    term, env = _term(clock_hide=True, clock_hide_ramp=(240, 480))
    rand = _CountingRand(0.4)
    monkeypatch.setattr(torch, "rand", rand)
    env.common_step_counter = 360                      # the midpoint: p = 0.5
    ids = torch.tensor([0, 2, 5])
    term.reset(ids)
    assert rand.calls == [3]
    want = torch.zeros(H.N, dtype=torch.bool)
    want[ids] = True                                   # 0.4 < 0.5
    assert torch.equal(term._clock_hidden, want)
    assert float(env.extras["log"][LOG_KEY]) == pytest.approx(3 / H.N)
    # stepping never re-draws, and the latch holds while p moves
    for t in range(361, 380):
        env.common_step_counter = t
        obs = _call(term, env, clock_hide=True, clock_hide_ramp=(240, 480))
        assert torch.equal(obs[ids, 15], torch.zeros(3))
        assert torch.equal(obs[~want, 15],
                           torch.sin(env.action_manager.get_term("joint_pos").clock_phase[~want]))
    assert rand.calls == [3]
    # a later reset re-draws ONLY its own rows, against the p of its own time
    env.common_step_counter = 250                      # p small: 0.4 > p -> visible again
    term.reset(torch.tensor([2]))
    assert rand.calls == [3, 1]
    want[2] = False
    assert torch.equal(term._clock_hidden, want)


def test_ramp_draws_are_bernoulli_p_of_t():
    n = 4000
    term, env = _term(n=n, clock_hide=True, clock_hide_ramp=(0, 1000))
    torch.manual_seed(0)
    for t, p in ((250, 0.5 * (1 - 0.7071068)), (500, 0.5), (750, 0.5 * (1 + 0.7071068))):
        env.common_step_counter = t
        term.reset(None)
        assert float(term._clock_hidden.float().mean()) == pytest.approx(p, abs=0.03), t
        assert float(env.extras["log"][LOG_KEY]) == pytest.approx(p, abs=0.03)


def test_every_row_is_forced_hidden_from_t1_on_even_without_a_reset(monkeypatch):
    term, env = _term(clock_hide=True, clock_hide_ramp=(240, 480))
    env.common_step_counter = 479
    obs = _call(term, env, clock_hide=True, clock_hide_ramp=(240, 480))
    assert not torch.equal(obs[:, 15], torch.zeros(H.N))          # nothing latched yet
    env.common_step_counter = 480
    obs = _call(term, env, clock_hide=True, clock_hide_ramp=(240, 480))
    assert torch.equal(obs[:, 15], torch.zeros(H.N)) and torch.equal(obs[:, 16], torch.zeros(H.N))
    # a reset at t >= t1 latches p = 1 even for the largest uniform, and logs 1.0
    monkeypatch.setattr(torch, "rand", _CountingRand(0.9999999))
    term.reset(torch.tensor([0]))
    assert bool(term._clock_hidden[0]) and float(env.extras["log"][LOG_KEY]) == 1.0
    # the log reads the EFFECTIVE mask: 1.0 after t1 even with rows that have not reset since
    assert int(term._clock_hidden.sum()) == 1


def test_a_hidden_episode_reads_zero_in_all_22_columns_from_its_first_compute(monkeypatch):
    """Ramp mode: the base reset zeroes the history, so a freshly hidden row is clean at once."""
    term, env = _term(clock_hide=True, clock_hide_ramp=(0, 240))
    for t in range(3):                                  # a visible stretch fills the history
        env.common_step_counter = t
        _call(term, env, clock_hide=True, clock_hide_ramp=(0, 240))
        env.episode_length_buf += 1
    assert term._obs_history_buffer[:, :, 15].abs().sum() > 0
    monkeypatch.setattr(torch, "rand", _CountingRand(0.0))
    env.common_step_counter = 120                       # p = 0.5 > 0.0 -> hidden
    term.reset(torch.tensor([1]))
    env.episode_length_buf[1] = 0
    obs = _call(term, env, clock_hide=True, clock_hide_ramp=(0, 240))
    for c in _all_clock_cols():
        assert float(obs[1, c]) == 0.0, c
    assert float(obs[0, 15]) != 0.0                     # a row that did not reset stays visible


# ======================================================== refusals and construction
@pytest.mark.parametrize("params", [{"clock_hide_ramp": (0, 240)},
                                    {"clock_hide": False, "clock_hide_ramp": (0, 240)}])
def test_a_ramp_without_hide_is_refused_by_the_term(params):
    with pytest.raises(ValueError, match="clock_hide"):
        _term(**params)


@pytest.mark.parametrize("ramp", [(240, 240), (480, 240), (-24, 240), (0.5, 240), ("0", "240"),
                                  (0, 240, 480), 240])
def test_a_malformed_ramp_is_refused_by_the_term(ramp):
    with pytest.raises(ValueError, match="clock_hide_ramp"):
        _term(clock_hide=True, clock_hide_ramp=ramp)


def test_a_list_ramp_normalises_to_the_tuple():
    term, _ = _term(clock_hide=True, clock_hide_ramp=[24000, 48000])
    assert term.clock_hide_ramp == (24000, 48000) and isinstance(term.clock_hide_ramp, tuple)


def test_the_clock_params_are_construction_time_only():
    term, env = _term()
    with pytest.raises(ValueError, match="construction-time"):
        _call(term, env, clock_hide=True)
    hid, henv = _term(clock_hide=True, clock_hide_ramp=(0, 240))
    _call(hid, henv, clock_hide=True, clock_hide_ramp=[0, 240])        # an equal ramp is fine
    with pytest.raises(ValueError, match="construction-time"):
        _call(hid, henv, clock_hide=True, clock_hide_ramp=(0, 480))
    with pytest.raises(ValueError, match="construction-time"):
        _call(hid, henv, clock_hide=True)


# ================================================================ the knob module
def test_parsers():
    assert ck.CLOCK_HIDE_VAR == "KRABBY_OMNI_CLOCK_HIDE"
    assert ck.CLOCK_HIDE_RAMP_VAR == "KRABBY_OMNI_CLOCK_HIDE_RAMP"
    for text, want in ((None, False), ("", False), (" ", False), ("0", False), ("1", True),
                       (" 1 ", True)):
        assert ck.parse_clock_hide(text) is want, text
    for bad in ("true", "yes", "2", "on", "off"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_CLOCK_HIDE"):
            ck.parse_clock_hide(bad)
    assert ck.parse_clock_hide_ramp(None) is None and ck.parse_clock_hide_ramp("") is None
    got = ck.parse_clock_hide_ramp("24000:48000")
    assert got == (24000, 48000) and isinstance(got, tuple)
    assert ck.parse_clock_hide_ramp("0:24") == (0, 24)
    for bad in ("24000", "1:2:3", "a:b", "1.5:24", "24:24", "48:24", "-24:24"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_CLOCK_HIDE_RAMP"):
            ck.parse_clock_hide_ramp(bad)
    assert set(ck.CLOCK_KNOB_PARSERS) == {ck.CLOCK_HIDE_VAR, ck.CLOCK_HIDE_RAMP_VAR}
    assert ck.EVAL_SEMANTIC_KNOBS == (ck.CLOCK_HIDE_VAR,)          # the ramp is never carried
    assert ck.KNOB_OFF_VALUES == {ck.CLOCK_HIDE_VAR: "0"}


class _Bag:
    pass


class _TermCfg:
    def __init__(self, func=None, params=None):
        self.func, self.params = func, params


def _cfg():
    cfg = _Bag()
    cfg.observations = _Bag()
    cfg.observations.policy = _Bag()
    cfg.observations.policy.extreme_parkour_observations = _TermCfg(
        func=OmniObs, params={"history_length": 10, "facing_obs_clip": 0.6,
                              "facing_obs_encoding": "raw"})
    cfg.observations.policy.other = _TermCfg(func=object, params={"x": 1})
    return cfg


def _dump(cfg) -> list[str]:
    """``params`` of every obs term, dumped the way ``isaaclab.utils.io.dump_yaml`` does."""
    data = {name: {"params": term.params} for name, term in vars(cfg.observations.policy).items()}
    return yaml.dump(data, default_flow_style=False, sort_keys=False).splitlines()


def test_unset_blank_or_off_writes_nothing():
    pristine = _dump(_cfg())
    for env in ({}, {ck.CLOCK_HIDE_VAR: ""}, {ck.CLOCK_HIDE_VAR: "0"},
                {ck.CLOCK_HIDE_VAR: "0", ck.CLOCK_HIDE_RAMP_VAR: ""}, ck.KNOB_OFF_VALUES):
        cfg = _cfg()
        ck.apply_clock_knobs(cfg, env)
        assert _dump(cfg) == pristine, env


def test_knob_set_dumps_differ_only_in_the_new_lines():
    pristine = _dump(_cfg())
    hide = _cfg()
    ck.apply_clock_knobs(hide, {ck.CLOCK_HIDE_VAR: "1"})
    added = [ln for ln in difflib.ndiff(pristine, _dump(hide)) if ln[:1] in "+-"]
    assert added == ["+     clock_hide: true"]
    ramp = _cfg()
    ck.apply_clock_knobs(ramp, {ck.CLOCK_HIDE_VAR: "1", ck.CLOCK_HIDE_RAMP_VAR: "24000:48000"})
    added = [ln for ln in difflib.ndiff(pristine, _dump(ramp)) if ln[:1] in "+-"]
    assert added == ["+     clock_hide: true", "+     clock_hide_ramp: !!python/tuple",
                     "+     - 24000", "+     - 48000"]
    params = ramp.observations.policy.extreme_parkour_observations.params
    assert params["clock_hide_ramp"] == (24000, 48000) and isinstance(params["clock_hide_ramp"], tuple)
    assert ramp.observations.policy.other.params == {"x": 1}       # only the omni term is armed
    # ... and the armed params build the term they describe
    term = OmniObs(_Cfg(**{k: v for k, v in params.items() if k.startswith("clock")}), _env())
    assert term.clock_hide is True and term.clock_hide_ramp == (24000, 48000)


@pytest.mark.parametrize("hide", [None, "", "0"])
def test_a_ramp_without_hide_is_refused_by_the_knob(hide):
    env = {ck.CLOCK_HIDE_RAMP_VAR: "0:240"}
    if hide is not None:
        env[ck.CLOCK_HIDE_VAR] = hide
    with pytest.raises(ValueError, match="KRABBY_OMNI_CLOCK_HIDE"):
        ck.apply_clock_knobs(_cfg(), env)


def test_hide_without_an_omni_obs_term_is_refused():
    cfg = _cfg()
    del cfg.observations.policy.extreme_parkour_observations
    with pytest.raises(RuntimeError, match="no omni observation term"):
        ck.apply_clock_knobs(cfg, {ck.CLOCK_HIDE_VAR: "1"})


def test_omni_knobs_reexports_and_applies_the_clock_knobs():
    knobs = _load(OMNI_KNOBS_PY, "omni_knobs__clock_hide_test")
    names = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}
    assert {"KRABBY_OMNI_CLOCK_HIDE", "KRABBY_OMNI_CLOCK_HIDE_RAMP"} <= names   # the whitelist
    assert knobs.CLOCK_KNOB_PARSERS is knobs._ck.CLOCK_KNOB_PARSERS
    assert knobs.parse_clock_hide_ramp("0:24") == (0, 24)
    fn = [n for n in ast.parse(OMNI_KNOBS_PY.read_text()).body
          if isinstance(n, ast.FunctionDef) and n.name == "apply_omni_knobs"][0]
    calls = [ast.unparse(s) for s in fn.body if isinstance(s, ast.Expr)]
    assert "_ck.apply_clock_knobs(cfg, env)" in calls
    # before the motion and stroke/posture knobs, which stay LAST in that order (pinned by
    # test_crab_hex_omni_motion_knobs.py)
    assert calls.index("_ck.apply_clock_knobs(cfg, env)") < calls.index("_mk.apply_motion_knobs(cfg, env)")
    # NOTE(body-frame command mode, 2026-10-15): the command frame is applied after them all (the one
    # application point of KRABBY_OMNI_COMMAND_FRAME; world writes nothing).
    assert calls[-1] == "_apply_command_frame(cfg, env)"
    assert calls[-2] == "_sk.apply_stroke_knobs(cfg, env)"


def test_no_preset_arms_a_clock_knob():
    src = ((OMNI / "config/crab_hex/crab_hex_omni_phases.py").read_text()
           + (TASKS / "crab_hex_forward_task/config/crab_hex/crab_hex_phases.py").read_text()
           + (OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py").read_text())
    for var in (ck.CLOCK_HIDE_VAR, ck.CLOCK_HIDE_RAMP_VAR, '"clock_hide"'):
        assert var not in src, var


# ================================================================== the mirror
def test_build_mirrors_is_unchanged_and_maps_the_hidden_clock_to_zero():
    mt = _load(MIRROR_TESTS_PY, "omni_mirror_tests__clock_hide")
    base_lr, base_fa = mt.omni.build_mirrors(mt._Env(), mt.OBS_DIM)
    env = mt._Env()
    params = env.cfg.observations.policy.extreme_parkour_observations.params
    params.update(clock_hide=True, clock_hide_ramp=(24000, 48000))
    lr, fa = mt.omni.build_mirrors(env, mt.OBS_DIM)
    for a, b in ((base_lr, lr), (base_fa, fa)):
        assert torch.equal(a.obs_perm, b.obs_perm) and torch.equal(a.obs_sign, b.obs_sign)
    assert mt.OBS_DIM == 1149
    # a HIDE observation's clock columns stay zero under both planes and their product
    term, tenv = _term(clock_hide=True)
    term.reset(None)
    obs = _call(term, tenv, clock_hide=True)
    for img in (lr.mirror_obs(obs), fa.mirror_obs(obs), fa.mirror_obs(lr.mirror_obs(obs))):
        for c in _all_clock_cols():
            assert torch.equal(img[:, c], torch.zeros(H.N)), c


# ======================================================== X08:c6 loads under HIDE
def _rsl_modules():
    if "_rslmods_clock_hide" not in sys.modules:
        pkg = types.ModuleType("_rslmods_clock_hide")
        pkg.__path__ = [str(RSL_MODULES)]
        sys.modules["_rslmods_clock_hide"] = pkg
    return importlib.import_module("_rslmods_clock_hide.crab_actor_critic_with_encoder")


@pytest.mark.skipif(not X08_C6.is_file(), reason="X08:c6 checkpoint is not on this machine")
def test_x08_c6_state_dict_loads_at_the_hide_width():
    """HIDE is width-preserving: the head of record loads strictly and acts on a hidden obs."""
    term, env = _term(clock_hide=True)
    term.reset(None)
    obs = _call(term, env, clock_hide=True)
    assert obs.shape[1] == 1149
    ckpt = torch.load(X08_C6, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    assert state["critic.0.weight"].shape[1] == obs.shape[1]      # the critic reads all 1149
    policy_cfg = dict(yaml.safe_load((X08_C6_DIR / "params/agent.yaml").read_text())["policy"])
    policy_cfg.pop("class_name")
    mod = _rsl_modules()
    policy = mod.CrabHexActorCriticRMA(obs.shape[1], state["std"].shape[0], **policy_cfg)
    assert policy.load_state_dict(state) is not None                 # strict: no key/shape mismatch
    with torch.no_grad():
        act = policy.act_inference(obs, hist_encoding=True)
    assert act.shape == (H.N, 18) and bool(torch.isfinite(act).all())
