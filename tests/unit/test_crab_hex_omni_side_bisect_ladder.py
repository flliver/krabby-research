# SPDX-License-Identifier: BSD-3-Clause
"""The driver wiring for the side-speed-bisection plan's step 1 (2026-09-28): the two new knobs'
registration in ``run_candidates.py`` (``knob_whitelist``, ``knob_parsers``, ``AGENT_DUMP_KEYS``,
``OVERLAY_DUMP_CHECKS``), the cross-knob composition rule
(``_check_cmd_obs_mirror_composition`` -- ``KRABBY_OMNI_CMD_OBS=scalar`` needs
``KRABBY_OMNI_MIRROR_IMAGES=fa`` in the SAME composed training stage), and the EVAL_CARRY split
that keeps the runner-only mirror knob out of every eval process while still carrying the
observation knob into all of them.

Twin of ``test_crab_hex_omni_stroke_ladder.py`` (the ``_check_stroke_cat_w_composition``
precedent this file's composition-rule tests follow) and
``test_crab_hex_omni_clock_ladder.py`` (the ``OVERLAY_DUMP_CHECKS`` dump-text precedent).

The module-level behaviour these driver checks are wired FOR -- the fore-aft-only mirror image
count, the observation slot zeroing, the knob-off identity of both appliers -- is pinned where
the sibling knobs of the same kind already live: ``test_crab_hex_omni_mirror.py`` (the two-image
entry point), ``test_crab_hex_omni_observations.py`` (``cmd_obs_scalar``) and
``test_crab_hex_omni_phases.py`` (``_apply_cmd_obs_knob``, the two parsers, and the runner-side
``_repoint_symmetry`` knob-off identity, lifted out of ``agents/rsl_rl_ppo_cfg.py`` with ``ast``
since that module imports isaaclab at load time).

Pure Python; ``run_candidates.py`` is loaded by path (the ``test_crab_hex_omni_stroke_ladder.py``
pattern) and exercised against the existing
``tests/unit/fixtures/candidates_v2_2026-09-22.yaml`` fixture -- no Isaac, no subprocess, no real
campaign directory.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TOOLS = OMNI / "experiments/tools"
CANDS_YAML = REPO / "tests/unit/fixtures/candidates_v2_2026-09-22.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(TOOLS / "run_candidates.py", "run_candidates_side_bisect_ladder_test")


@pytest.fixture(scope="module")
def k(rc):
    return rc.knobs()


@pytest.fixture
def cands(rc):
    return rc.load_candidates(CANDS_YAML)


def _refusals(rc, doc, subject=None) -> list[str]:
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    return [r for v in verdicts if subject is None or v["id"] == subject for r in v["reasons"]]


# ================================================================== knob registration
def test_the_fixture_still_validates_before_any_of_this(rc, cands):
    """Sanity anchor: every assertion below reads a DIFFERENCE against this clean baseline."""
    verdicts = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


def test_side_bisect_knob_names_are_visible_through_knobs_and_the_whitelist(rc, k):
    whitelist = rc.knob_whitelist()
    assert k.MIRROR_IMAGES_VAR in whitelist
    assert k.CMD_OBS_VAR in whitelist


def test_side_bisect_knob_parsers_are_registered(rc, k):
    parsers = rc.knob_parsers()
    assert parsers[k.MIRROR_IMAGES_VAR]("fa") == "fa"
    assert parsers[k.CMD_OBS_VAR]("scalar") == "scalar"
    assert parsers[k.CMD_OBS_VAR]("") is None
    with pytest.raises(ValueError):
        parsers[k.MIRROR_IMAGES_VAR]("lr")
    with pytest.raises(ValueError):
        parsers[k.CMD_OBS_VAR]("vector")


def test_mirror_images_is_an_agent_side_dump_key_and_cmd_obs_is_not(rc, k):
    """The symmetry loss lives on the RUNNER cfg (``params/agent.yaml``); the observation knob
    lives on the ENV cfg (``params/env.yaml``) -- ``overlay_checks`` reads the two dumps
    separately, keyed on :data:`AGENT_DUMP_KEYS`."""
    assert k.MIRROR_IMAGES_VAR in rc.AGENT_DUMP_KEYS
    assert k.CMD_OBS_VAR not in rc.AGENT_DUMP_KEYS


def test_side_bisect_knobs_have_an_overlay_dump_check(rc, k):
    assert k.MIRROR_IMAGES_VAR in rc.OVERLAY_DUMP_CHECKS
    assert k.CMD_OBS_VAR in rc.OVERLAY_DUMP_CHECKS


def test_a_candidate_overlay_setting_both_knobs_round_trips_and_dumps(rc, cands, k):
    """Both knobs pass ``_check_knob``'s whitelist/parser gate on an ordinary chain stage --
    composed TOGETHER, so the cross-knob rule below does not also fire and mask this check."""
    doc = copy.deepcopy(cands)
    stage = doc["chains"]["scratch_1c"]["stages"][0]
    stage["overlay"].update({k.MIRROR_IMAGES_VAR: "fa", k.CMD_OBS_VAR: "scalar"})
    reasons = _refusals(rc, doc, "chain:scratch_1c")
    assert not any("not a known knob" in r or "no entry in OVERLAY_DUMP_CHECKS" in r for r in reasons), reasons
    assert not any(k.CMD_OBS_VAR in r and k.MIRROR_IMAGES_VAR in r for r in reasons), reasons


# --------------------------------------------- the cross-knob composition rule (plan step 1b)
def test_cmd_obs_scalar_without_fa_mirror_is_refused_at_dry_run(rc, cands, k):
    doc = copy.deepcopy(cands)
    stage = doc["chains"]["scratch_1c"]["stages"][0]
    stage["overlay"][k.CMD_OBS_VAR] = "scalar"                # no KRABBY_OMNI_MIRROR_IMAGES set
    reasons = _refusals(rc, doc, "chain:scratch_1c")
    assert any(k.CMD_OBS_VAR in r and k.MIRROR_IMAGES_VAR in r for r in reasons), reasons


def test_cmd_obs_scalar_with_fa_mirror_composes_cleanly_at_dry_run(rc, cands, k):
    doc = copy.deepcopy(cands)
    stage = doc["chains"]["scratch_1c"]["stages"][0]
    stage["overlay"].update({k.CMD_OBS_VAR: "scalar", k.MIRROR_IMAGES_VAR: "fa"})
    reasons = _refusals(rc, doc, "chain:scratch_1c")
    assert not any(k.CMD_OBS_VAR in r and k.MIRROR_IMAGES_VAR in r for r in reasons), reasons


def test_check_cmd_obs_mirror_composition_directly(rc, k):
    reasons: list[str] = []
    rc._check_cmd_obs_mirror_composition({k.CMD_OBS_VAR: "scalar"}, "where", reasons)
    assert len(reasons) == 1 and "where" in reasons[0] and k.MIRROR_IMAGES_VAR in reasons[0]
    for env in ({k.CMD_OBS_VAR: "scalar", k.MIRROR_IMAGES_VAR: "lr"},   # not 'fa'
                {k.CMD_OBS_VAR: "scalar", k.MIRROR_IMAGES_VAR: ""}):    # blank -> not 'fa' either
        r2: list[str] = []
        rc._check_cmd_obs_mirror_composition(env, "where", r2)
        assert len(r2) == 1, env
    r3: list[str] = []
    rc._check_cmd_obs_mirror_composition({k.CMD_OBS_VAR: "scalar", k.MIRROR_IMAGES_VAR: "fa"},
                                         "where", r3)
    assert r3 == []
    # CMD_OBS absent or off: nothing to check, whatever MIRROR_IMAGES says
    for env in ({}, {k.CMD_OBS_VAR: ""}, {k.CMD_OBS_VAR: "off"}, {k.CMD_OBS_VAR: "0"}):
        r4: list[str] = []
        rc._check_cmd_obs_mirror_composition(env, "where", r4)
        assert r4 == [], env
    # a bad CMD_OBS value is left to _check_knob's own parser gate, not this rule
    r5: list[str] = []
    rc._check_cmd_obs_mirror_composition({k.CMD_OBS_VAR: "vector"}, "where", r5)
    assert r5 == []


# ================================================================== EVAL_CARRY (plan step 1b)
def test_cmd_obs_is_carried_into_eval_and_mirror_images_never_is(rc, k):
    """The runner-only mirror knob is structurally absent from every eval build: the symmetry
    loss does not run there, so there is nothing for the composition rule above to see or fire
    on -- an eval process is refused by neither knob, whatever the TRAINING overlay set."""
    rp = rc.rp
    assert k.MIRROR_IMAGES_VAR not in rp.EVAL_CARRY
    assert k.CMD_OBS_VAR in rp.EVAL_CARRY
    spec = rp.ph.PHASES["1c"]
    ev = rp.eval_env_for(spec, "A15+B", keep=rp.CANARY_KEYS, keep_source=rp.ph.FORMATION,
                         carry_overlay={k.CMD_OBS_VAR: "scalar", k.MIRROR_IMAGES_VAR: "fa"})
    assert ev[k.CMD_OBS_VAR] == "scalar"
    assert k.MIRROR_IMAGES_VAR not in ev


def test_cmd_obs_default_carries_the_off_spelling_when_no_preset_sets_it(rc, k):
    rp = rc.rp
    assert rp.EVAL_CARRY_DEFAULTS[k.CMD_OBS_VAR] in k.CMD_OBS_OFF_SPELLINGS
    spec = rp.ph.PHASES["1c"]
    pe = rp.ph.phase_env("1c")
    assert k.CMD_OBS_VAR not in pe, "no shipped preset arms this knob (it would make the default dead code)"
    ev = rp.eval_env_for(spec, "A15+B", keep=rp.CANARY_KEYS, keep_source=rp.ph.FORMATION)
    assert ev[k.CMD_OBS_VAR] == rp.EVAL_CARRY_DEFAULTS[k.CMD_OBS_VAR]


# ================================================================== the dump-audit checks
#: Real-shaped ``params/agent.yaml`` text (the ``symmetry_cfg`` block a runner dump carries).
AGENT_MIRROR_OFF = """
policy:
  class_name: CrabHexActorCriticRMA
algorithm:
  entropy_coef: 0.01
  symmetry_cfg:
    use_data_augmentation: true
    data_augmentation_func: parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:crab_hex_omni_symmetry_augmentation
"""

AGENT_MIRROR_ON = """
policy:
  class_name: CrabHexActorCriticRMA
algorithm:
  entropy_coef: 0.01
  symmetry_cfg:
    use_data_augmentation: true
    data_augmentation_func: parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:crab_hex_omni_symmetry_augmentation_fa
"""


def test_mirror_images_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_MIRROR_IMAGES"]
    assert check(AGENT_MIRROR_ON, "fa")                       # armed, dump shows the fa swap
    assert not check(AGENT_MIRROR_OFF, "fa")                  # armed but the dump shows the base path
    with pytest.raises(ValueError):
        check(AGENT_MIRROR_ON, "lr")                          # not a recognised mode at all


#: Real-shaped ``params/env.yaml`` text (the omni observation term's ``params`` block).
ENV_CMD_OBS_OFF = """
observations:
  policy:
    extreme_parkour_observations:
      params:
        history_length: 10
        facing_obs_clip: 0.6
        facing_obs_encoding: raw
"""

ENV_CMD_OBS_ON = """
observations:
  policy:
    extreme_parkour_observations:
      params:
        history_length: 10
        facing_obs_clip: 0.6
        facing_obs_encoding: raw
        cmd_obs_scalar: true
"""


def test_cmd_obs_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_CMD_OBS"]
    assert check(ENV_CMD_OBS_OFF, "")                         # off: no key at all
    assert check(ENV_CMD_OBS_OFF, "off")                      # every off spelling matches "absent"
    assert not check(ENV_CMD_OBS_OFF, "scalar")                # armed but the dump shows no key
    assert check(ENV_CMD_OBS_ON, "scalar")                    # armed and the dump shows true
    assert not check(ENV_CMD_OBS_ON, "")                       # the knob is off but the dump shows true


# ---- dump checks for the EXISTING knobs the plan's R arm overlays (plan step 1c, 2026-09-28):
# the appliers and parsers already existed (R's four knobs were live in the pre-Revision-4 1b
# preset); only these driver-side OVERLAY_DUMP_CHECKS entries were missing.
ENV_CLOCK_COMBINE_SUM = """
rewards:
  reward_clock_schedule:
    func: x:clock_schedule_income
    weight: 0.4
    params:
      combine: sum
"""

ENV_CLOCK_COMBINE_PRODUCT = """
rewards:
  reward_clock_schedule:
    func: x:clock_schedule_income
    weight: 0.4
    params:
      combine: product
"""


def test_clock_combine_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_CLOCK_COMBINE"]
    assert check(ENV_CLOCK_COMBINE_SUM, "sum")
    assert not check(ENV_CLOCK_COMBINE_SUM, "product")
    assert check(ENV_CLOCK_COMBINE_PRODUCT, "product")
    assert not check(ENV_CLOCK_COMBINE_PRODUCT, "sum")


#: ``KRABBY_TRACK_SIGMA2`` stores the VARIANCE; the dump's kernel carries ``std = sqrt(sigma2)``.
ENV_TRACK_STD_0_2 = """
rewards:
  track_lin_vel_xy_exp:
    func: x:track_lin_vel_xy_exp
    weight: 1.25
    params:
      command_name: base_velocity
      std: 0.2
"""


def test_track_sigma2_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_TRACK_SIGMA2"]
    assert check(ENV_TRACK_STD_0_2, "0.04")                   # sqrt(0.04) == 0.2
    assert not check(ENV_TRACK_STD_0_2, "0.09")               # sqrt(0.09) == 0.3 != 0.2


ENV_TRACK_L1_ON = """
rewards:
  penalty_tracking_error_l1:
    func: x:y
    weight: -2.0
"""


def test_track_l1_w_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_TRACK_L1_W"]
    assert check(ENV_TRACK_L1_ON, "-2.0")
    assert not check(ENV_TRACK_L1_ON, "-1.0")


#: ``KRABBY_RSI_FRAC``: 0 (or unset) means the term is ABSENT -- ``_repoint_rsi`` never builds it.
ENV_RSI_OFF = """
events:
  reset_root_state:
    func: x:y
"""

ENV_RSI_ON = """
events:
  rsi_reference_reset:
    func: x:reset_from_reference_states_omni
    params:
      bank_path: rsi_bank_omni.npz
      fraction: 0.2
      source: all
"""


def test_rsi_frac_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_RSI_FRAC"]
    assert check(ENV_RSI_OFF, "0.0")                          # frac 0: the term must be absent
    assert check(ENV_RSI_OFF, "0")
    assert not check(ENV_RSI_ON, "0.0")                       # armed but the dump HAS the term
    assert check(ENV_RSI_ON, "0.2")                           # armed at the matching fraction
    assert not check(ENV_RSI_ON, "0.5")                       # armed at a DIFFERENT fraction


# ================================================================== suite rows cover the trained set
CAMPAIGN = OMNI / "experiments/2026-09-28_side_bisect/candidates.yaml"


def test_the_side_bisect_campaign_validates_clean(rc):
    """The real campaign file, every reference chain included, passes the dry-run validator."""
    verdicts = rc.validate_candidates(rc.load_candidates(CAMPAIGN), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


def test_every_1b_stage_trained_set_is_scoreable_on_side22_25(rc):
    """What the eval really receives: ``--trained-categories`` from the 1b composed env, checked by
    the harness's own ``suite_cli_args`` against the gate row (review 2026-09-29: the first row had
    no ``stand`` category, so every arm's eval would have halted ``eval_infra``)."""
    doc = rc.load_candidates(CAMPAIGN)
    S = rc.rp._gait_schedule()
    scenarios, _ = S.load_manifest(rc.rp.OMNI_MANIFEST)
    row = next(x for x in scenarios if x.id == "side22_25__omni")
    for name, chain in doc["chains"].items():
        for stage in chain["stages"]:
            env = rc._effective_stage_cfg(doc, stage, stage.get("overlay") or {})
            csv = rc._trained_csv(rc.eval_categories().trained_categories(env).to_dict())
            assert csv, name
            assert S.suite_cli_args(row, trained_categories=csv) == csv.split(","), name


def test_a_suite_row_missing_a_trained_category_is_refused_at_dry_run(rc):
    class _TC:
        def to_dict(self):
            return {"present": ["stand", "fwd", "side"], "decisive": ["stand"]}

    reasons: list = []
    rc._check_suite_rows_cover_trained({"eval": ["side22_25"]}, _TC(), "chain:X/s0", reasons)
    assert len(reasons) == 1 and "['fwd'] are not categories" in reasons[0], reasons
    reasons = []
    rc._check_suite_rows_cover_trained({"eval": ["suite"]}, _TC(), "chain:X/s0", reasons)
    assert reasons == []


# ================================================================== KRABBY_RESAMPLE_S dump check (reversal test)
_RESAMPLE_DUMP = """commands:
  base_velocity:
    class_type: parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command:CrabHexOmniCommand
    resampling_time_range: !!python/tuple
    - 40.0
    - 40.0
    debug_vis: false
"""


def test_resample_s_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_RESAMPLE_S"]
    assert check(_RESAMPLE_DUMP, "40:40")
    assert check(_RESAMPLE_DUMP, "40.0:40")
    assert not check(_RESAMPLE_DUMP, "10:10")
    assert not check(_RESAMPLE_DUMP.replace("resampling_time_range", "other_range"), "40:40")
