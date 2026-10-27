#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 4 -- the DAgger trainer's TORCH-FREE specification (constants, schedules,
the process environment, the stand-record readers, the A1 PASS rule, the step-6 smoke lists).

Imported BEFORE Kit starts by ``sn_dagger.py`` (and by ``sn_dagger.sh`` through ``sn_dagger.py --emit-env``), so it
imports nothing heavier than the standard library (map risk L6: no torch, no numpy, no yaml at module level).
Everything a run's VERDICT depends on lives here as a pure function over plain records, so the CPU tests pin it and
the trainer, the smoke and the notify / A1 reads cannot drift:

* :data:`S1_OVERLAY` -- the ``P2__S1_s18`` overlay VERBATIM (``2026-10-07_omni_phase2/candidates.yaml``; empty
  strings included); :data:`OVERLAY_CHANGES` -- the four keys PLAN D2 / D5 change (pool stand share 0.5 with matching
  SLOTS, ``KRABBY_RESAMPLE_S=3:12``, ``KRABBY_OMNI_STAND_OBS=0``); :func:`train_env` the process environment
  (map section 3.1); :func:`env_reasons` the pre-Kit fail-closed check (map T2 / T3); :func:`obs_parity_reasons` the D5
  obs-parity rule against an eval stamp (every obs-affecting ``EVAL_CARRY`` key).
* :func:`beta` (PLAN D3: 1 -> 0 linearly over the first third, then 0), :data:`SIGMA_TABLE` (CHARTER OPEN-5),
  :data:`CONTROL_SHARE` (OPEN-4), the A1 assignment table (PLAN Amendment 1).
* stand records (one dict per stand; written by the trainer's tracker, ``stands.jsonl``) and their readers:
  :func:`scored` (OPEN-7 / A1 on the stand's FIRST draw: its scheduled length >= the floor, not cut by the episode
  timeout, not a run-start stand, and it either reached its scheduled end or fell before it),
  :func:`premise_read` (the step-7 notify statistic, student vs the concurrent R-executed control slice),
  :func:`a1_read` (per spoiler + control, the A1 PASS rule fixed before the read).
* :data:`SMOKE_DAGGER` / :data:`SMOKE_A1` -- PLAN step 6's dose list and the A1-mode smoke list; the smoke JSON must
  carry a verdict for every listed check (:func:`smoke_verdict`).

Decisions recorded here for CHARTER section 3 (made by the step-4 builder where PLAN / CHARTER leave the choice):
* the step-7 "stands longer than 4 s" is the SCHEDULED length (OPEN-7), with A1's scoring (not timeout-cut; a fall
  scores homed 0) -- the actual length would let falls shrink the denominator, the failure the A1 audit fixed;
* AMENDMENT TO OPEN-7 (review 2026-10-12; log in CHARTER section 3 before the A1 read): scoring uses the stand's
  FIRST draw only -- scheduled length = the entry draw's (never the merge-extended one), the fall = a fall before
  that draw's scheduled end, homed / |c| / trim read on that draw's last stand row. OPEN-7's merge extension happens
  only to stands that SURVIVED their first draw, so a floor on the merged length admitted a short-first-draw stand
  only if it did not fall (outcome-dependent inclusion: ~37 % of the scored set, nearly every early spoiler fall in it
  dropped -- fall gaps shrink, homed shares inflate, the A1 ratio drifts to PASS). The first draw is fixed at entry,
  so inclusion no longer depends on the outcome. The merged fields (``sched_len_s``, ``cut``, ``end``, ``homed``,
  ``abs_c``) stay in the records, reported only. Stands already in progress when the run starts (``entry_kind``
  ``run_start``, an ``init_at_random_ep_len`` artefact: spawned at reset, scheduled from a random episode clock) are
  excluded;
* "loaded legs" in A1's trim-band share = R's DOWN legs (hip-pitch obs >= p_down) on the last stand row;
* LIFT-onset oversampling: ONSET_SHARE = 0.10 of every minibatch drawn from the onset rows (when any exist);
  per-leg inverse-branch-frequency weights capped at W_MAX = 20 (OPEN-3);
* the A1 fall margin is the PLAN's +0.01 (binding); max(0.01, 2 SE(diff)) is reported beside it, never used.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
#: scripts -> 2026-10-11_stand_network -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour
REPO = HERE.parents[6]
TASK_ID = "Isaac-Crab-Hex-Omni-Teacher-v0"
PRESET = "2a"
PLANT = "A15+B"
S1_CKPT = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
S1_SHA256 = "968cc550989919cdc3fa36ae047379febabee5659cde0ed4d099195acb13c23c"
S1_AGENT_YAML = S1_CKPT.parent / "params" / "agent.yaml"
#: the REFS_S1_s18 suite es1 env stamp (the eval env every stand_net row runs under; PLAN D5)
REFS_ENV_STAMP = REPO / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24"
CANDIDATES = HERE.parents[1] / "2026-10-07_omni_phase2" / "candidates.yaml"
OVERLAY_CHAIN = "P2__S1_s18"
DEFAULT_RUNS = REPO / "parkour/outputs/omni_plan/stand_network/runs"

MODES = ("dagger", "recover_probe")
#: PLAN D1 / D3: the one screened arm pair -> the actor-side stand mask flag
ARMS = {"mask": 1, "full": 0}

# ===================================================================================== the process environment
#: ``P2__S1_s18`` (candidates.yaml, stage p2a overlay) VERBATIM, empty strings included (map T2). Pinned against the
#: file by the tests and re-read by ``sn_dagger.py --emit-env``.
S1_OVERLAY = {
    "KRABBY_OMNI_CMD_POOL": "stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,"
                            "side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07",
    "KRABBY_OMNI_SLOTS": "0.15:0.55:0.1:0.2",
    "KRABBY_OMNI_DIR_BAND": "",
    "KRABBY_OMNI_FACING": "0.2:1.2",
    "KRABBY_YAW_W": "0.2",
    "KRABBY_CLOCK_W": "0.001",
    "KRABBY_APEX_W": "0.001",
    "KRABBY_OMNI_CLOCK_HIDE": "1",
    "KRABBY_OMNI_MIRROR_IMAGES": "fa",
    "KRABBY_RSI_FRAC": "0",
    "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
    "KRABBY_ACTION_BOUND_LOSS": "0.001",
    "KRABBY_ACTION_BOUND_RAMP": "0.0001:0.001:500",
    "KRABBY_ACTION_BOUND": "1.1",
    "KRABBY_STD_PARAM": "soft",
    "KRABBY_INIT_STD": "0.8",
    "KRABBY_ENTROPY_COEF": "0.0025",
    "KRABBY_ACTION_RATE_W": "-0.05",
    "KRABBY_PPO_HAND_DIAG": "1",
    "KRABBY_PHASEOUT": "",
    "KRABBY_FLAT_TERRAIN_GEOM": "field",
    "KRABBY_FIELD_PAD_R": "1.7",
    "KRABBY_FIELD_BLOCKS": "14:28",
    "KRABBY_FIELD_KINDS": "0.4:0.4:0.2",
    "KRABBY_FIELD_EDGES": "wall",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0",
    "KRABBY_TERRAIN_PROMOTE": "0.7:0.4",
}
#: the REFS head name the readers (sn_calib NAMES["REFS"], sh_read's K5 refs sha) anchor on
REFS_NAME = "REFS_S1_s18"

# ===================================================================================== the walker override (Task 4)
#: NOTE(M18 Task 4 pipeline step 6, 2026-10-16; DESIGN B6 "stand on a new walker"): ONE override, the environment
#: variable ``SN_WALKER_SPEC`` = a JSON object. UNSET (or blank) -> every constant above stays EXACTLY what it is
#: (today's behaviour: S1 s18, REFS_S1_s18, P2__S1_s18). SET -> its keys replace :data:`WALKER_SPEC_KEYS` (all
#: required except ``S1_AGENT_YAML``, which defaults to ``<S1_CKPT dir>/params/agent.yaml``); paths absolute. The
#: override walker's overlay must carry the SAME command pool and slots as P2__S1_s18 (the stand-0.5 POOL / SLOTS
#: below are derived from them) and no ``KRABBY_PAYLOAD_*`` key (the stand nets train and are read at 0 kg).
#: A malformed override RAISES at import: the trainer, the launchers' --emit-env and the readers all stop before Kit.
WALKER_SPEC_VAR = "SN_WALKER_SPEC"
WALKER_SPEC_KEYS = ("S1_CKPT", "S1_SHA256", "S1_AGENT_YAML", "REFS_ENV_STAMP", "CANDIDATES", "OVERLAY_CHAIN",
                    "S1_OVERLAY", "REFS_NAME")
_WALKER_SPEC_OPTIONAL = ("S1_AGENT_YAML",)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def walker_spec_override(environ=None) -> dict | None:
    """``SN_WALKER_SPEC`` parsed and checked -> ``{key: value}`` (paths as :class:`Path`), or None when unset."""
    import os  # noqa: PLC0415  (stdlib)

    env = os.environ if environ is None else environ
    raw = env.get(WALKER_SPEC_VAR)
    if raw is None or not str(raw).strip():
        return None
    try:
        spec = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{WALKER_SPEC_VAR} is not JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError(f"{WALKER_SPEC_VAR} must be a JSON object, got {type(spec).__name__}")
    unknown = sorted(set(spec) - set(WALKER_SPEC_KEYS))
    if unknown:
        raise ValueError(f"{WALKER_SPEC_VAR}: unknown key(s) {unknown}; known: {list(WALKER_SPEC_KEYS)}")
    missing = [k for k in WALKER_SPEC_KEYS if k not in spec and k not in _WALKER_SPEC_OPTIONAL]
    if missing:
        raise ValueError(f"{WALKER_SPEC_VAR}: missing key(s) {missing} (an override states the whole walker)")
    out: dict = {}
    for k in ("S1_CKPT", "S1_AGENT_YAML", "REFS_ENV_STAMP", "CANDIDATES"):
        if k not in spec:
            continue
        p = Path(str(spec[k]))
        if not p.is_absolute():
            raise ValueError(f"{WALKER_SPEC_VAR}: {k} must be an absolute path, got {spec[k]!r}")
        out[k] = p
    out.setdefault("S1_AGENT_YAML", out["S1_CKPT"].parent / "params" / "agent.yaml")
    sha = str(spec["S1_SHA256"]).strip().lower()
    if not _HEX64.match(sha):
        raise ValueError(f"{WALKER_SPEC_VAR}: S1_SHA256 must be 64 hex characters, got {spec['S1_SHA256']!r}")
    out["S1_SHA256"] = sha
    for k in ("OVERLAY_CHAIN", "REFS_NAME"):
        v = str(spec[k]).strip()
        if not re.match(r"^[A-Za-z0-9_]+$", v):
            raise ValueError(f"{WALKER_SPEC_VAR}: {k} must be [A-Za-z0-9_]+, got {spec[k]!r}")
        out[k] = v
    ov = spec["S1_OVERLAY"]
    if not isinstance(ov, dict) or not all(isinstance(k, str) for k in ov):
        raise ValueError(f"{WALKER_SPEC_VAR}: S1_OVERLAY must be an object of KRABBY_* -> string")
    ov = {str(k): ("" if v is None else str(v)) for k, v in ov.items()}
    bad = sorted(k for k in ov if not k.startswith("KRABBY_"))
    if bad:
        raise ValueError(f"{WALKER_SPEC_VAR}: S1_OVERLAY keys must be KRABBY_*, got {bad}")
    pay = sorted(k for k in ov if k.startswith("KRABBY_PAYLOAD_"))
    if pay:
        raise ValueError(f"{WALKER_SPEC_VAR}: S1_OVERLAY carries {pay}: the stand nets train and are read at 0 kg "
                         "(pass the walker's p2a overlay, without the payload knobs)")
    for k in ("KRABBY_OMNI_CMD_POOL", "KRABBY_OMNI_SLOTS"):
        if ov.get(k) != _S1_OVERLAY_DEFAULT.get(k):
            raise ValueError(f"{WALKER_SPEC_VAR}: S1_OVERLAY {k}={ov.get(k)!r} differs from P2__S1_s18's "
                             f"{_S1_OVERLAY_DEFAULT.get(k)!r}: the stand-0.5 POOL / SLOTS are derived from it")
    out["S1_OVERLAY"] = ov
    return out


_S1_OVERLAY_DEFAULT = dict(S1_OVERLAY)
#: the parsed override (None = today's constants), recorded by whoever reports the walker
WALKER_OVERRIDE = walker_spec_override()
if WALKER_OVERRIDE is not None:
    S1_CKPT = WALKER_OVERRIDE["S1_CKPT"]
    S1_SHA256 = WALKER_OVERRIDE["S1_SHA256"]
    S1_AGENT_YAML = WALKER_OVERRIDE["S1_AGENT_YAML"]
    REFS_ENV_STAMP = WALKER_OVERRIDE["REFS_ENV_STAMP"]
    CANDIDATES = WALKER_OVERRIDE["CANDIDATES"]
    OVERLAY_CHAIN = WALKER_OVERRIDE["OVERLAY_CHAIN"]
    S1_OVERLAY = dict(WALKER_OVERRIDE["S1_OVERLAY"])
    REFS_NAME = WALKER_OVERRIDE["REFS_NAME"]

#: PLAN D2: stand share 0.5 (stand weight 0.85 against S1's other weights, within-slot proportions kept) with the
#: MATCHING slot projection (map T3: a pool without SLOTS falls back to the legacy sampler silently; a mismatch is loud).
POOL = ("stand:0.85,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,"
        "side_r_turn:0.035,omni_turn:0.07")
SLOTS = "0.85:0.55:0.1:0.2"
RESAMPLE_S = "3:12"
#: the four overlay changes (recorded as deviations from the P2__S1_s18 training env in CHARTER section 3)
OVERLAY_CHANGES = {"KRABBY_OMNI_CMD_POOL": POOL, "KRABBY_OMNI_SLOTS": SLOTS, "KRABBY_RESAMPLE_S": RESAMPLE_S,
                   "KRABBY_OMNI_STAND_OBS": "0"}
#: the driver's own keys (run_phases.train): the preset and the plant
DRIVER_ENV = {"KRABBY_OMNI_PHASE": PRESET, "KRABBY_PLANT": PLANT}
#: non-KRABBY process keys (run_phases._subenv + train)
PROCESS_ENV = {"OMNI_KIT_ACCEPT_EULA": "yes", "TERM": "xterm", "PYTHONUNBUFFERED": "1"}
#: never set in the trainer's process (map section 3.1)
FORBIDDEN_ENV = ("KRABBY_PHASE", "KRABBY_SIDE_PHASE", "KRABBY_HEX_USD_PATH", "KRABBY_OMNI_STAND_MASK",
                 "KRABBY_OMNI_STAND_PROXY", "KRABBY_RESTANCE")
#: variable prefixes that must be absent from the trainer's process (P2L / launcher fallbacks: not this process's)
FORBIDDEN_PREFIXES = ("P2R_", "SN_STAND_")


def expected_krabby_env() -> dict[str, str]:
    """The trainer process's COMPLETE ``KRABBY_*`` set at start (before the preset's import-time setdefault)."""
    env = dict(DRIVER_ENV)
    env.update(S1_OVERLAY)
    env.update(OVERLAY_CHANGES)
    return env


def train_env(base: dict) -> dict[str, str]:
    """Map section 3.1: ``base`` with every ``KRABBY_*`` removed, plus :data:`PROCESS_ENV` and
    :func:`expected_krabby_env` (the preset expands at import with ``setdefault``, so explicit keys win)."""
    env = {str(k): str(v) for k, v in base.items() if not str(k).startswith("KRABBY_")
           and not str(k).startswith(FORBIDDEN_PREFIXES)}
    env.update(PROCESS_ENV)
    env.update(expected_krabby_env())
    return env


def env_reasons(environ) -> list[str]:
    """Pre-Kit, fail closed (map T2 / T3 / section 3.1): the process's ``KRABBY_*`` set EXACTLY
    :func:`expected_krabby_env` (no extra key, none missing, same values incl. empty strings), the Kit EULA accepted,
    no forbidden key and no ``P2R_*`` / ``SN_STAND_*`` variable."""
    out = []
    want = expected_krabby_env()
    have = {k: str(v) for k, v in environ.items() if str(k).startswith("KRABBY_")}
    for k in sorted(set(want) | set(have)):
        if k not in have:
            out.append(f"{k} missing (want {want[k]!r})")
        elif k not in want:
            out.append(f"{k}={have[k]!r} is not part of the trainer env (start from a clean KRABBY_* base)")
        elif have[k] != want[k]:
            out.append(f"{k}={have[k]!r}, want {want[k]!r}")
    # FORBIDDEN_ENV keys are never in ``want``: a present one is reported as an extra key above
    bad = sorted(k for k in environ if str(k).startswith(FORBIDDEN_PREFIXES))
    if bad:
        out.append(f"forbidden variable(s) {bad} in the trainer's environment")
    if str(environ.get("OMNI_KIT_ACCEPT_EULA", "")).strip().lower() not in ("yes", "y", "true", "1"):
        out.append("OMNI_KIT_ACCEPT_EULA is not 'yes' (a headless Kit start blocks on the EULA prompt)")
    return out


#: ``run_phases.EVAL_CARRY`` / ``EVAL_CARRY_DEFAULTS`` (the keys that decide what the policy OBSERVES and what command it
#: is scored against; pinned equal to run_phases by the tests)
EVAL_CARRY = ("KRABBY_OMNI_FACING_OBS_CLIP", "KRABBY_OMNI_SPEED_ELLIPSE", "KRABBY_OMNI_ANG_VEL_MAX",
              "KRABBY_OMNI_CLOCK_TURN_R", "KRABBY_OMNI_YAW_RANGE", "KRABBY_OMNI_FACING_OBS_ENCODING",
              "KRABBY_OMNI_ANG_VEL_CLIP", "KRABBY_OMNI_RELAY_HYST", "KRABBY_OMNI_STAND_HOLD",
              "KRABBY_OMNI_STAND_PARK", "KRABBY_ACTION_LPF_HZ", "KRABBY_ACTION_LPF_OBS", "KRABBY_OMNI_STAND_OBS",
              "KRABBY_OMNI_CLOCK_HIDE", "KRABBY_OMNI_CMD_OBS", "KRABBY_STD_PARAM", "KRABBY_INIT_STD")
EVAL_CARRY_DEFAULTS = {"KRABBY_OMNI_FACING_OBS_ENCODING": "raw", "KRABBY_OMNI_ANG_VEL_CLIP": "0.2",
                       "KRABBY_OMNI_RELAY_HYST": "0", "KRABBY_OMNI_STAND_HOLD": "0",
                       "KRABBY_OMNI_STAND_PARK": "0", "KRABBY_ACTION_LPF_HZ": "0",
                       "KRABBY_ACTION_LPF_OBS": "0", "KRABBY_OMNI_STAND_OBS": "0",
                       "KRABBY_OMNI_CLOCK_HIDE": "0", "KRABBY_OMNI_CMD_OBS": "0",
                       "KRABBY_STD_PARAM": "0", "KRABBY_INIT_STD": "0"}
#: obs knobs outside EVAL_CARRY that must be OFF on both sides (D1: the stand net masks actor-side only)
OBS_OFF_EXTERNAL = ("KRABBY_OMNI_STAND_MASK", "KRABBY_OMNI_STAND_PROXY")


def obs_parity_reasons(train_environ, eval_env: dict) -> list[str]:
    """PLAN D5 obs parity + step 6 "the trainer env carries every obs-affecting var of the eval stamp": for every
    :data:`EVAL_CARRY` key the trainer's EFFECTIVE value (after the preset's import-time expansion) equals the eval
    stamp's, where an absent key means the class default the eval spells as :data:`EVAL_CARRY_DEFAULTS`."""
    out = []
    for k in EVAL_CARRY:
        t = train_environ.get(k)
        e = eval_env.get(k)
        t = None if t is None else str(t)
        e = None if e is None else str(e)
        d = EVAL_CARRY_DEFAULTS.get(k)
        same = (t == e) or (t is None and e == d) or (e is None and t == d)
        if not same:
            out.append(f"{k}: trainer {t!r} vs eval {e!r} (default {d!r})")
    for k in OBS_OFF_EXTERNAL:
        for side, env in (("trainer", train_environ), ("eval", eval_env)):
            v = env.get(k)
            if v is not None and str(v).strip() not in ("", "0"):
                out.append(f"{k}={v!r} on the {side} side: want off (D1: the stand net masks actor-side only)")
    return out


# ===================================================================================== cfg asserts (map section 3.2 item 4)
def _get(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    return cur


def _tuple(v):
    return None if v is None else tuple(float(x) for x in v)


def cfg_reasons(env_cfg, agent_cfg) -> list[str]:
    """Fail closed BEFORE ``gym.make`` (map section 3.2 item 4; duck-typed so the CPU tests can drive it): the command
    sampler (resample 3-12 s, the stand-0.5 pool and its SLOTS, no direction band, the S1 ellipse, no eval latch),
    40-s episodes, the terrain promotion, S1's crab_failure thresholds, the observation params the eval stamp's obs
    depends on (clock hidden, facing clip / encoding, no stand bit / mask / proxy, no corruption), the action delay,
    no RSI reset, and the runner's clip / std."""
    out = []

    def want(path, got, exp):
        if got != exp:
            out.append(f"{path} = {got!r}, want {exp!r}")

    def get(obj, path):
        try:
            return _get(obj, path)
        except (AttributeError, KeyError, TypeError):
            return "<missing>"

    bv = "commands.base_velocity"
    want(f"{bv}.resampling_time_range", _tuple(get(env_cfg, f"{bv}.resampling_time_range")), (3.0, 12.0))
    want(f"{bv}.cmd_pool", get(env_cfg, f"{bv}.cmd_pool"), POOL)
    want(f"{bv}.slot_probs", _tuple(get(env_cfg, f"{bv}.slot_probs")), tuple(float(x) for x in SLOTS.split(":")))
    want(f"{bv}.dir_band", get(env_cfg, f"{bv}.dir_band"), None)
    want(f"{bv}.speed_ellipse", _tuple(get(env_cfg, f"{bv}.speed_ellipse")), (0.35, 0.35, 0.25))
    latch = getattr(_get(env_cfg, bv), "eval_slot_latch", False)
    if latch:
        out.append(f"{bv}.eval_slot_latch = {latch!r}: training samples the slot (no eval latch)")
    want("episode_length_s", float(get(env_cfg, "episode_length_s")), 40.0)
    bp = "parkours.base_parkour"
    want(f"{bp}.move_up_frac", get(env_cfg, f"{bp}.move_up_frac"), 0.7)
    want(f"{bp}.move_down_frac", get(env_cfg, f"{bp}.move_down_frac"), 0.4)
    want(f"{bp}.freeze_terrain_levels", get(env_cfg, f"{bp}.freeze_terrain_levels"), False)
    cf = "terminations.crab_failure.params"
    want(f"{cf}.limit_angle", get(env_cfg, f"{cf}.limit_angle"), 0.5)
    want(f"{cf}.contact_force_threshold", get(env_cfg, f"{cf}.contact_force_threshold"), 500.0)
    want(f"{cf}.minimum_root_height_z", get(env_cfg, f"{cf}.minimum_root_height_z"), None)
    op = "observations.policy.extreme_parkour_observations.params"
    want(f"{op}.clock_hide", get(env_cfg, f"{op}.clock_hide"), True)
    want(f"{op}.facing_obs_clip", get(env_cfg, f"{op}.facing_obs_clip"), 0.6)
    want(f"{op}.facing_obs_encoding", get(env_cfg, f"{op}.facing_obs_encoding"), "raw")
    params = get(env_cfg, op)
    for k in ("stand_obs", "stand_bit_clock", "stand_proxy_hold", "stand_mask"):
        if isinstance(params, dict) and params.get(k):
            out.append(f"{op}.{k} = {params.get(k)!r}: want off (D1 / D5)")
    want("observations.policy.enable_corruption", get(env_cfg, "observations.policy.enable_corruption"), False)
    want("actions.joint_pos.action_delay_steps", list(get(env_cfg, "actions.joint_pos.action_delay_steps") or []),
         [1, 1])
    want("actions.joint_pos.use_delay", get(env_cfg, "actions.joint_pos.use_delay"), True)
    if getattr(getattr(env_cfg, "events", None), "rsi_reference_reset", None) is not None:
        out.append("events.rsi_reference_reset is set: want no RSI (KRABBY_RSI_FRAC=0)")
    want("agent.clip_actions", float(get(agent_cfg, "clip_actions")), 1.0)
    want("agent.policy.noise_std_type", get(agent_cfg, "policy.noise_std_type"), "soft")
    want("agent.policy.init_noise_std", float(get(agent_cfg, "policy.init_noise_std")), 0.8)
    return out


#: the policy / estimator cfg keys that decide the network's shape and forward (compared with S1's own agent.yaml)
NETWORK_POLICY_KEYS = ("class_name", "actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std",
                       "noise_std_type", "tanh_encoder_output", "scan_encoder_dims", "priv_encoder_dims", "actor")
NETWORK_ESTIMATOR_KEYS = ("class_name", "num_prop", "num_scan", "num_priv_explicit", "num_priv_latent", "num_hist",
                          "hidden_dims")


def network_cfg_reasons(train_cfg: dict, s1_agent: dict) -> list[str]:
    """The trainer's runner cfg (task registry, under the trainer env) builds S1's network: every shape / forward key
    equals S1's own ``params/agent.yaml`` (json-normalized: tuples == lists)."""
    norm = lambda v: json.loads(json.dumps(v))  # noqa: E731
    out = []
    for sec, keys in (("policy", NETWORK_POLICY_KEYS), ("estimator", NETWORK_ESTIMATOR_KEYS)):
        a, b = train_cfg.get(sec) or {}, s1_agent.get(sec) or {}
        for k in keys:
            if norm(a.get(k)) != norm(b.get(k)):
                out.append(f"{sec}.{k}: trainer {a.get(k)!r} vs S1 {b.get(k)!r}")
    return out


# ===================================================================================== schedules (PLAN D3, CHARTER OPEN-2..6)
EX_STUDENT, EX_BETA_R, EX_CONTROL = 0, 1, 2
EXECUTORS = ("student", "beta_r", "control")
#: OPEN-4: the R-executed control slice, a per-episode draw (envs stay exchangeable)
CONTROL_SHARE = 0.10
#: OPEN-5: per-episode DART sigma on executed stand rows (student and beta-R episodes): ((sigma, probability), ...)
SIGMA_TABLE = ((0.0, 0.5), (0.1, 0.25), (0.3, 0.25))
#: OPEN-2
LR = 3e-4
STEPS_PER_ITER = 24
MINIBATCHES = 16
MINIBATCH = 4096
BUFFER_GB = 4.0
NUM_ENVS = 1024
MAX_GRAD_NORM = 1.0
CKPT_EVERY = 500
#: OPEN-3 (builder's numbers, CHARTER section 3): onset rows' share of every minibatch; the per-leg weight cap
ONSET_SHARE = 0.10
W_MAX = 20.0
#: the stand-end read
HOMED_EPS = 0.1                 # |c| <= 0.1 rad on all six legs (R's eps_c; the step-7 / A1 statistic)
TRIM_BAND = (0.1, 0.3)          # R's trim-only band (|c| in (0.1, 0.3]) on the down legs at stand end
EXIT_FALL_WINDOW_S = 2.0        # p2_read.p2r_hold_reads: a fall within 2 s (100 steps) of an exit
PREMISE_MIN_SCHED_S = 4.0       # step 7: sigma = 0 stands longer than 4 s (scheduled; see the module docstring)
PREMISE_RATIO = 0.5             # step 7: student < 0.5 x R's at mid-run -> STOP and ask


def beta(it: int, iterations: int) -> float:
    """PLAN D3: R executes stand rows at the start, beta -> 0 linearly over the FIRST THIRD of the iterations, then
    0. Evaluated at the iteration an episode starts (the draw is per episode)."""
    it, iterations = int(it), int(iterations)
    if iterations <= 0:
        raise ValueError(f"beta: iterations must be > 0, got {iterations}")
    third = iterations / 3.0
    return max(0.0, 1.0 - it / third) if it < third else 0.0


def sigma_of(u: float) -> float:
    """``u`` ~ U[0, 1) -> the episode's DART sigma per :data:`SIGMA_TABLE`."""
    acc = 0.0
    for s, p in SIGMA_TABLE:
        acc += p
        if u < acc:
            return s
    return SIGMA_TABLE[-1][0]


# ===================================================================================== A1 (PLAN Amendment 1, CHARTER OPEN-7)
AS_CONTROL, AS_STATUE, AS_CHATTER, AS_STUCK = 0, 1, 2, 3
ASSIGNS = ("control", "statue", "chatter", "stuck_lift")
ASSIGN_PROBS = (0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
SPOIL_WINDOW_S = (1.0, 4.0)     # the spoiler acts for U[1, 4] s after entry, then R
CHATTER_SIGMA = 0.3
STUCK_CAM_TARGET = 0.4          # rad: the stuck leg's cam is driven to +-0.4 by R's saturated in-air P-law
A1_MIN_SCHED_S = 8.0
A1_CONTROL_FLOOR = 0.75         # user decision 2026-10-12
A1_SPOILER_RATIO = 0.8
A1_FALL_MARGIN = 0.01
R_TIME_BINS = (0.0, 2.0, 4.0, 6.0, 8.0, math.inf)


def assign_of(u: float) -> int:
    acc = 0.0
    for k, p in enumerate(ASSIGN_PROBS):
        acc += p
        if u < acc:
            return k
    return len(ASSIGN_PROBS) - 1


# ===================================================================================== stand records -> reads
#: the stand-record fields every reader relies on (the tracker writes them; ``stands.jsonl``)
RECORD_KEYS = ("it_entry", "it_end", "env", "entry_ep", "last_row_ep", "end_ep", "sched_end_ep", "max_ep",
               "sched_len_s", "length_s", "cut", "end", "kind", "sigma", "take_ep", "r_time_s", "homed",
               "max_abs_c", "abs_c", "down", "n_down", "trim_legs", "exit_fall", "exit_censored", "entry_kind",
               "prev_kind", "fell_before_takeover", "stuck_leg", "stuck_sign", "take_hip", "take_c",
               "terrain_level", "mode", "n_merges",
               # the FIRST draw (the scored stand; module docstring, amendment to OPEN-7)
               "sched_end0_ep", "sched_len0_s", "cut0", "end0", "fall0", "homed0", "abs_c0", "max_abs_c0",
               "n_down0", "trim_legs0", "r_time0_s")
#: ``end0`` values a scored stand can have: it reached its first draw's scheduled end, or fell before it
END0_SCORED = ("reached", "fall")


def scored(rec: dict, min_sched_s: float) -> bool:
    """OPEN-7 / A1, on the stand's FIRST draw (amendment, module docstring): a stand counts when the scheduled
    length of the draw it ENTERED on is >= ``min_sched_s``, that draw's scheduled end lies before the episode timeout,
    it is not a run-start stand, and it either reached that end or fell before it (fall = 1, homed = 0). Every clause
    but the last is fixed at entry, and the last always holds for a recorded stand that is not cut -- inclusion does
    not depend on the outcome."""
    return (float(rec["sched_len0_s"]) >= float(min_sched_s) - 1e-9 and not bool(rec["cut0"])
            and rec.get("entry_kind") != "run_start" and rec["end0"] in END0_SCORED)


def premise_scored(rec: dict) -> bool:
    """PLAN step 7: a sigma = 0 stand whose first draw is scheduled LONGER than 4 s (strict), scored as A1 scores
    (first draw; not timeout-cut; a fall counts, homed 0)."""
    return (float(rec["sigma"]) == 0.0 and float(rec["sched_len0_s"]) > PREMISE_MIN_SCHED_S
            and scored(rec, PREMISE_MIN_SCHED_S))


def homed_scored(rec: dict) -> bool:
    """A scored stand's homed outcome: it reached its first draw's end with |c| <= 0.1 on all legs there."""
    return rec["end0"] == "reached" and bool(rec["homed0"])


def fall_scored(rec: dict) -> bool:
    """A scored stand's fall outcome: it fell before its first draw's scheduled end."""
    return rec["end0"] == "fall"


def _share(k: int, n: int) -> dict:
    p = (k / n) if n else None
    se = math.sqrt(p * (1 - p) / n) if n else None
    return {"k": int(k), "n": int(n), "p": p, "se": se}


def group_stats(recs: list[dict]) -> dict:
    """Homed / fall shares over already-SCORED stands, on their FIRST draw (:func:`homed_scored` /
    :func:`fall_scored`; the trim band on the first draw's last row), + exit falls over the stands' actual finalized
    exits (the stand -> walk handover, whenever it happened)."""
    n = len(recs)
    homed = sum(1 for r in recs if homed_scored(r))
    falls = sum(1 for r in recs if fall_scored(r))
    exits = [r for r in recs if r["end"] == "exit" and r.get("exit_fall") is not None and not r.get("exit_censored")]
    xf = sum(1 for r in exits if bool(r["exit_fall"]))
    pre = sum(1 for r in recs if fall_scored(r) and bool(r.get("fell_before_takeover")))
    down = sum(int(r.get("n_down0") or 0) for r in recs if r["end0"] == "reached")
    trim = sum(int(r.get("trim_legs0") or 0) for r in recs if r["end0"] == "reached")
    return {"n": n, "homed": _share(homed, n), "fall": _share(falls, n), "exit_fall": _share(xf, len(exits)),
            "falls_before_takeover": int(pre), "trim_band_down_legs": _share(trim, down)}


def premise_read(records: list[dict], it_lo: int = 0, it_hi: int | None = None) -> dict:
    """PLAN step 7 notify read: train-time closed-loop homed share at stand end (|c| <= 0.1 all legs) on sigma = 0
    stands whose first draw is scheduled > 4 s (:func:`premise_scored`), and exit falls, for the STUDENT and for the
    concurrent R-executed control slice (same iterations, same envs / terrain levels; excluded from the student's
    value). Stands are
    attributed to the iteration they END in (``[it_lo, it_hi)``). ``stop``: student < 0.5 x control."""
    sel = [r for r in records if r.get("mode", "dagger") == "dagger" and int(r["it_end"]) >= int(it_lo)
           and (it_hi is None or int(r["it_end"]) < int(it_hi))]
    out = {"window": [int(it_lo), None if it_hi is None else int(it_hi)], "min_sched_s": PREMISE_MIN_SCHED_S}
    for name in EXECUTORS:
        g = [r for r in sel if r["kind"] == name and premise_scored(r)]
        out[name] = group_stats(g)
    s, c = out["student"]["homed"]["p"], out["control"]["homed"]["p"]
    out["ratio"] = (s / c) if (s is not None and c) else None
    out["stop"] = None if out["ratio"] is None else bool(out["ratio"] < PREMISE_RATIO)
    return out


def r_time_bin(t: float) -> str:
    for lo, hi in zip(R_TIME_BINS[:-1], R_TIME_BINS[1:]):
        if lo <= t < hi:
            return f"[{lo:g},{hi:g})"
    return f"[{R_TIME_BINS[-2]:g},inf)"


def a1_read(records: list[dict]) -> dict:
    """PLAN Amendment 1 (rule fixed before the read): scored stands (:func:`scored` on the FIRST draw: scheduled
    >= 8 s, not timeout-cut, not a run-start stand; a fall before the draw's end scores homed 0; |c| on the draw's last
    stand row) per assignment. PASS iff the control's homed share >= 0.75
    and, for every spoiler, homed >= 0.8 x the control's and stand-fall share <= the control's + 0.01. Any group with
    no scored stand -> INVALID (fail closed). Also reported: falls before takeover, homed by R-time bin, the trim-band
    share of the down legs, exit falls, the control split by the previous stand's assignment (carry-over), and the
    alternative fall margin max(0.01, 2 SE(diff)) (informational, never used)."""
    sel = [r for r in records if r.get("mode") == "recover_probe"]
    groups = {}
    for k, name in enumerate(ASSIGNS):
        g = [r for r in sel if r["kind"] == name and scored(r, A1_MIN_SCHED_S)]
        st = group_stats(g)
        bins: dict = {}
        for r in g:
            b = r_time_bin(float(r["r_time0_s"]))
            bins.setdefault(b, [0, 0])
            bins[b][1] += 1
            bins[b][0] += int(homed_scored(r))
        st["homed_by_r_time"] = {b: _share(*v) for b, v in sorted(bins.items(), key=lambda kv: float(
            kv[0][1:].split(",")[0]))}
        groups[name] = st
    ctrl = [r for r in sel if r["kind"] == "control" and scored(r, A1_MIN_SCHED_S)]
    carry = {}
    for prev in ("none",) + ASSIGNS:
        g = [r for r in ctrl if (r.get("prev_kind") or "none") == prev]
        carry[prev] = group_stats(g)["homed"]
    excluded = {"short": 0, "cut": 0, "run_start": 0, "timeout_or_other": 0}
    for r in sel:
        if float(r["sched_len0_s"]) < A1_MIN_SCHED_S - 1e-9:
            excluded["short"] += 1
        elif bool(r["cut0"]):
            excluded["cut"] += 1
        elif r.get("entry_kind") == "run_start":
            excluded["run_start"] += 1
        elif r["end0"] not in END0_SCORED:
            excluded["timeout_or_other"] += 1
    reasons = []
    c = groups["control"]
    if not c["n"]:
        reasons.append("control: no scored stand")
    for name in ASSIGNS[1:]:
        if not groups[name]["n"]:
            reasons.append(f"{name}: no scored stand")
    verdict = "INVALID" if reasons else None
    checks = []
    if verdict is None:
        ch, cf = c["homed"]["p"], c["fall"]["p"]
        checks.append({"check": "control homed >= 0.75", "value": ch, "line": A1_CONTROL_FLOOR,
                       "pass": ch >= A1_CONTROL_FLOOR})
        for name in ASSIGNS[1:]:
            g = groups[name]
            gh, gf = g["homed"]["p"], g["fall"]["p"]
            checks.append({"check": f"{name} homed >= 0.8 x control", "value": gh, "line": A1_SPOILER_RATIO * ch,
                           "pass": gh >= A1_SPOILER_RATIO * ch})
            se = math.sqrt(gf * (1 - gf) / g["n"] + cf * (1 - cf) / c["n"])
            checks.append({"check": f"{name} stand-fall share <= control + 0.01", "value": gf,
                           "line": cf + A1_FALL_MARGIN, "pass": gf <= cf + A1_FALL_MARGIN + 1e-12,
                           "informational_margin_max_0p01_2se": max(A1_FALL_MARGIN, 2 * se)})
        verdict = "PASS" if all(x["pass"] for x in checks) else "STOP"
    return {"rule": "PLAN Amendment 1: control homed >= 0.75; every spoiler homed >= 0.8 x control and stand-fall "
                    "share <= control + 0.01 (scored on the FIRST draw: scheduled >= 8 s, not timeout-cut, not a "
                    "run-start stand, fall before the draw's end = homed 0)",
            "verdict": verdict, "invalid_reasons": reasons, "checks": checks, "groups": groups,
            "control_by_previous_stand": carry, "excluded": excluded, "n_records": len(sel)}


def read_stands(path) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ===================================================================================== the step-6 smoke lists
#: PLAN step 6 (the DAgger arms): (check name, the PLAN text it asserts)
SMOKE_DAGGER = (
    ("stand_row_share", "stand-row share ~ 0.5"),
    ("handovers", "handover count"),
    ("branch_shares_student_rows", "branch shares on student-visited rows"),
    ("lift_onset_after_balancing", "LIFT-onset >= 1 % after balancing"),
    ("beta_schedule", "the beta schedule function (asserted, not reached)"),
    ("dart_sigma_realized", "DART sigma realized"),
    ("obs_parity_env", "the trainer env carries every obs-affecting var of the eval stamp"),
    ("obs_parity_cols", "cols 0-1148 of a stand row: layout / clock / stand-bit columns as the eval's (D5)"),
    ("forward_parity", "a trainer-captured obs batch fed to the stand_net forward returns the trainer's action "
                       "bitwise (in-process, same batch); capture saved for the offline StandNetMode replay"),
    ("s1_deterministic", "S1 runs deterministic act_inference"),
    ("buffer_projection", "buffer memory projected to the final iteration count fits the cap"),
    ("ckpt_loads", "saved ckpt loads strictly (rsl_rl load path), estimator == S1's (stand_net walkstop row: "
                   "sn_eval_rows.sh)"),
    ("terrain_level_recorded", "every stand record carries its terrain level (step 7: 'same terrain levels')"),
)
#: PLAN step 6 "A1-mode smoke (Amendment 1)"
SMOKE_A1 = (
    ("a1_assignment_shares", "assignment ~ 1/2 control and 1/6 per spoiler"),
    ("a1_executed_r_or_spoiler", "executed stand rows are R or spoiler only"),
    ("a1_no_optimizer", "no optimizer step runs"),
    ("a1_dart_zero", "DART sigma = 0"),
    ("a1_statue_zero", "statue executes exactly 0"),
    ("a1_chatter_sigma", "chatter's realised sigma ~ 0.3 per channel"),
    ("a1_stuck_lift_state", "on most stuck-lift stands the held leg is up (hip <= R's up threshold) with |c| > 0.3"),
    ("a1_last_row_read", "the stand-end value is read on the last stand row before the exit"),
    ("a1_scored_filter", "scored-stand filter per A1"),
    ("terrain_level_recorded", "every stand record carries its terrain level (A1: 'same terrain levels')"),
)


def smoke_verdict(kind: str, checks: list[dict]) -> dict:
    """Every check of the ``kind`` list must be present exactly once with a boolean ``pass`` (missing / extra /
    duplicated -> FAIL, fail closed)."""
    want = [n for n, _ in (SMOKE_DAGGER if kind == "dagger" else SMOKE_A1)]
    got = [c.get("name") for c in checks]
    problems = []
    missing = [n for n in want if n not in got]
    extra = [n for n in got if n not in want]
    dup = sorted({n for n in got if got.count(n) > 1})
    if missing:
        problems.append(f"missing checks {missing}")
    if extra:
        problems.append(f"unlisted checks {extra}")
    if dup:
        problems.append(f"duplicated checks {dup}")
    bad = [c.get("name") for c in checks if not isinstance(c.get("pass"), bool)]
    if bad:
        problems.append(f"checks without a boolean verdict {bad}")
    failed = [c["name"] for c in checks if c.get("pass") is False]
    ok = not problems and not failed
    return {"kind": kind, "verdict": "PASS" if ok else "FAIL", "problems": problems, "failed": failed,
            "plan_text": dict(SMOKE_DAGGER if kind == "dagger" else SMOKE_A1)}


# ===================================================================================== paths
_BAD_PATH = re.compile(r"[\s*?\[\]{}$`'\"\\;|&<>()!~#]")


def path_reasons(path) -> list[str]:
    """A run dir / checkpoint path the stand_net launcher can take later (EVAL_EXTRA_ARGS is unquoted, map L9)."""
    s = str(path)
    out = []
    if _BAD_PATH.search(s):
        out.append(f"{s!r} contains whitespace / glob / shell characters")
    if not Path(s).is_absolute():
        out.append(f"{s!r} is not absolute")
    return out
