"""Generated blocks of ``policy/POLICY_SUMMARY.md`` and the ``policy/mdp_pins.yaml`` extractor.

``POLICY_SUMMARY.md`` is hand-written prose (goals, rationale, reward / terrain / knob explanations)
with numeric tables that are GENERATED between marker comments::

    <!-- generated:<id> -->
    ...table...
    <!-- /generated:<id> -->

``bundle_policy.py --sync`` rewrites every marked block from three sources of truth:

* ``config/crab_hex/crab_hex_phases.py`` (pure Python): the per-phase ``KRABBY_*`` stacks, the
  gait-income schedule and ramps, the promotion rescale, the legacy presets;
* ``policy/manifest.yaml``: heads, shas, source runs, evals, notes;
* ``policy/mdp_pins.yaml``: the Isaac-only numbers (effective reward weights and params, terrain
  generator fields, commands, events, terminations, actions, runner / algorithm settings), extracted
  by ``bundle_policy.py --pin-mdp <dump dir>`` from the config dumps that
  ``tests/integration/test_crab_hex_phase_configs.py`` produces, and validated against fresh dumps by
  that test (``test_mdp_pins_match_dumps``).

Pure-python cross-checks (``tests/unit/test_policy_of_record.py``) prove that every pin derivable
from a preset knob agrees with ``phase_env()`` and that the committed document equals a fresh render.
Stdlib + PyYAML only; no Isaac import.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]                                   # crab_hex_forward_task/
REPO = PKG.parents[3]
POLICY = PKG / "policy"
SUMMARY = POLICY / "POLICY_SUMMARY.md"
PINS = POLICY / "mdp_pins.yaml"
PHASES_PY = PKG / "config" / "crab_hex" / "crab_hex_phases.py"
KNOBS_PY = PKG / "mdp" / "exposure_knobs.py"
CURRICULUMS_PY = PKG / "mdp" / "curriculums.py"
MDP_CFG_PY = PKG / "config" / "crab_hex" / "agents" / "parkour_mdp_cfg.py"
REWARDS_CLASS = "CrabHexFlatWalkRewardsCfg"

DOCUMENTED_PHASES = ("1a", "2a", "2b", "2c", "3a", "3b")
RL_PHASES = ("1a", "2a", "2b", "2c")
# envs per training kind, as the phase driver launches them (experiments/tools/run_phases.py NUM_ENVS)
NUM_ENVS = {"rl": 256, "distill": 192}
SEED_OF_RECORD = 3

# Reward-PARAMETER knobs read by CrabHexFlatWalkRewardsCfg.__post_init__ (the weight knobs are parsed
# from the source; these set params and are listed here so Appendix B is complete).
PARAM_KNOBS = {
    "KRABBY_TRACK_SIGMA2": ("track_lin_vel_xy_exp", "params.std = sqrt(value)"),
    "KRABBY_FOOT_CLEAR_MIN": ("reward_foot_clearance", "params.min_clearance_m"),
    "KRABBY_FOOT_CLEAR_FLAT": ("reward_foot_clearance / penalty_swing_min_clearance", "=1 lifts the parkour_flat mask (params.parkour_name = None)"),
    "KRABBY_APEX_M": ("reward_clock_swing_apex", "params.apex_m"),
    "KRABBY_AIRTIME_THRESH": ("reward_feet_air_time_positive", "params.threshold"),
    "KRABBY_MAX_SPEED_SCALE": ("reward_forward_progress_along_command", "params.max_speed_scale"),
    "KRABBY_CLOCK_COMBINE": ("reward_clock_schedule", "params.combine: sum or product"),
    "KRABBY_TRIPOD_MIN_PERIOD": ("reward_tripod_schedule", "params.min_period"),
    "KRABBY_TRIPOD_MAX_PERIOD": ("reward_tripod_schedule", "params.max_period"),
    "KRABBY_TRIPOD_CORR_TAU": ("reward_tripod_schedule", "params.corr_tau"),
    "KRABBY_TRIPOD_MIN_AMP": ("reward_tripod_schedule", "params.min_amp"),
}

# Non-reward knobs: the cfg field each one sets (config/crab_hex/crab_hex_env_cfg.py apply_flat_walk_knobs).
KNOB_FIELDS = {
    "KRABBY_LIN_VEL_X": "commands.base_velocity.ranges.lin_vel_x (lo:hi)",
    "KRABBY_HEADING": "commands.base_velocity.ranges.heading (lo:hi); presence also switches the flat-tile steering observation to the heading error",
    "KRABBY_HEADING_STIFFNESS": "commands.base_velocity.heading_control_stiffness",
    "KRABBY_STAND_FRAC": "commands.base_velocity.stand_frac (walking slots)",
    "KRABBY_RESAMPLE_S": "commands.base_velocity.resampling_time_range (lo:hi)",
    "KRABBY_EPISODE_S": "episode_length_s",
    "KRABBY_RSI_FRAC": "events.rsi_reference_reset (created when > 0; params.fraction)",
    "KRABBY_RSI_BANK": "events.rsi_reference_reset.params.bank_path",
    "KRABBY_RSI_SPAWN_FIX": "events.rsi_reference_reset.params.fix_spawn",
    "KRABBY_DR_PUSH": "events.push_by_setting_velocity.params.velocity_range x/y = (-v, v); unset removes the event",
    "KRABBY_DR_MASS": "events.randomize_rigid_body_mass.params.mass_distribution_params (lo:hi, add); unset removes the event",
    "KRABBY_DR_COM": "events.randomize_rigid_body_com.params.com_range x/y/z = (-v, v); unset removes the event",
    "KRABBY_FLAT_TERRAIN_MODE": "light: freeze levels, mix obstacles into the flat tiles (see FLAT_FRAC / DIFF / GEOM)",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "terrain_generator.sub_terrains.parkour_flat.proportion (the rest split evenly over gap / hurdle / step / stones)",
    "KRABBY_FLAT_TERRAIN_DIFF": "terrain_generator.difficulty_range (lo:hi)",
    "KRABBY_FLAT_TERRAIN_GEOM": "geometry preset: shallow (default) | recal2b2 | recal2b2w",
    "KRABBY_FLAT_TERRAIN_CURRICULUM": "1: terrain_generator.curriculum = True and levels unfrozen (rows become difficulty levels)",
    "KRABBY_TERRAIN_PROMOTE": "parkours.base_parkour.move_up_frac : move_down_frac",
    "KRABBY_CORRIDOR_HALF_WIDTH": "gap / hurdle / step half_valid_width override (lo:hi)",
    "KRABBY_STONE_WIDTH": "stepping-stone width override",
    "KRABBY_SPAWN_OFFSET": "events.reset_root_state.params.offset (platform spawn)",
    "KRABBY_SPAWN_SPREAD": "events.reset_root_state.params.spread / spread_frac (downrange spawns)",
    "KRABBY_TERM_ANGLE": "terminations.crab_failure.params.limit_angle",
    "KRABBY_TERM_CONTACT_N": "terminations.crab_failure.params.contact_force_threshold",
    "KRABBY_CAM_CLIP_LO": "actions.joint_pos.clip for the cam-shaft channels, lower bound",
    "KRABBY_ACTION_SCALE": "actions.joint_pos.scale for the hip / knee position channels",
    "KRABBY_SYM_LOSS_COEF": "runner symmetry_cfg.mirror_loss_coeff (0 disables the mirror loss)",
    "KRABBY_PHASEOUT": "cfg.curriculum: one cosine ramp term per entry (term:w0:w1:t0:t1)",
    "KRABBY_HEX_TEACHER_MODE": "Teacher-v0 mode: 2a | 2b | 2c rebuild the flat-walk MDP (legacy: bridge | 2b1 | 2b2 | full)",
    "KRABBY_PLANT": "plant by name (crab_hex_phases.PLANTS); nothing = the main asset",
    "KRABBY_STUDENT_MDP": "1 forces the phase-3 student MDP without a KRABBY_PHASE preset",
}

# Knobs a reader may expect to have been used and that the lineage left at their defaults.
KNOBS_AT_DEFAULT = (
    "KRABBY_SPAWN_OFFSET", "KRABBY_SPAWN_SPREAD", "KRABBY_RSI_SPAWN_FIX", "KRABBY_CORRIDOR_HALF_WIDTH",
    "KRABBY_STONE_WIDTH", "KRABBY_TERM_ANGLE", "KRABBY_TERM_CONTACT_N", "KRABBY_CAM_CLIP_LO",
    "KRABBY_ACTION_SCALE", "KRABBY_SYM_LOSS_COEF", "KRABBY_HEADING_STIFFNESS", "KRABBY_AIRTIME_THRESH",
    "KRABBY_APEX_M", "KRABBY_CLOCK_COMBINE", "KRABBY_MAX_SPEED_SCALE",
)

GEOMETRY_KEYS = ("gap_depth", "gap_size", "pit_depth", "incline_height", "last_incline_height", "step_height",
                 "hurdle_height_range", "stone_width", "stone_len", "last_stone_len", "half_valid_width",
                 "x_range", "y_range", "noise_range")
TERRAIN_KEYS = ("curriculum", "difficulty_range", "num_rows", "num_cols", "num_goals", "size", "border_width",
                "horizontal_scale", "vertical_scale", "slope_threshold")
ALGO_KEYS = ("class_name", "learning_rate", "schedule", "desired_kl", "clip_param", "entropy_coef", "value_loss_coef",
             "use_clipped_value_loss", "gamma", "lam", "num_learning_epochs", "num_mini_batches", "max_grad_norm",
             "priv_reg_coef_schedual", "dagger_update_freq")
POLICY_KEYS = ("class_name", "activation", "init_noise_std", "actor_hidden_dims", "critic_hidden_dims",
               "scan_encoder_dims", "priv_encoder_dims")
ACTOR_KEYS = ("num_prop", "num_scan", "num_hist", "num_priv_explicit", "num_priv_latent")
RUNNER_KEYS = ("class_name", "runner_class_name", "experiment_name", "clip_actions", "num_steps_per_env", "save_interval",
               "empirical_normalization")
ESTIMATOR_KEYS = ("class_name", "hidden_dims", "learning_rate", "train_with_estimated_states")
DEPTH_KEYS = ("backbone_class_name", "encoder_class_name", "depth_shape", "hidden_dims", "learning_rate", "num_steps_per_env")
EVENT_KEYS = ("push_by_setting_velocity", "randomize_rigid_body_mass", "randomize_rigid_body_com", "rsi_reference_reset",
              "reset_root_state", "physics_material", "reset_robot_joints", "base_external_force_torque", "random_camera_position")


# ---------------------------------------------------------------------------------------- loaders
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def load_pure() -> SimpleNamespace:
    """The pure-python modules the summary is derived from."""
    return SimpleNamespace(ph=_load(PHASES_PY, "crab_hex_phases"), xk=_load(KNOBS_PY, "exposure_knobs"),
                           cur=_load(CURRICULUMS_PY, "curriculums"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return yaml.safe_load((POLICY / "manifest.yaml").read_text())


def load_pins() -> dict:
    if not PINS.exists():
        raise SystemExit(f"{PINS} missing: run bundle_policy.py --pin-mdp <dump dir> (see the identity test)")
    return yaml.safe_load(PINS.read_text())


def write_pins(pins: dict) -> None:
    hdr = ("# GENERATED by experiments/tools/bundle_policy.py --pin-mdp <dump dir> from the Isaac config dumps that\n"
           "# tests/integration/crab_hex_phase_cfg_dump.py writes (one per KRABBY_PHASE preset). The only place the\n"
           "# Isaac-only numbers of the policy of record (reward weights / params, terrain generator, commands,\n"
           "# events, terminations, actions, runner) live in git. Validated against fresh dumps by\n"
           "# tests/integration/test_crab_hex_phase_configs.py::test_mdp_pins_match_dumps and against the presets by\n"
           "# tests/unit/test_policy_of_record.py. Do not hand-edit: regenerate after any MDP change.\n")
    PINS.write_text(hdr + yaml.safe_dump(pins, sort_keys=True, width=110, allow_unicode=True))


# ---------------------------------------------------------------------------------- reward source
def _literal_number(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_number(node.operand)
    raise ValueError(f"reward weight is not a literal: {ast.dump(node)}")


def _class_node(path: Path, cls: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node
    raise KeyError(f"{cls} not found in {path}")


def reward_terms_from_cfg_source(path: Path = MDP_CFG_PY, cls: str = REWARDS_CLASS) -> list[dict]:
    """Every ``RewTerm`` declared on the reward config class: name, func, literal default weight."""
    out = []
    for stmt in _class_node(path, cls).body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.value, ast.Call)):
            continue
        fn = stmt.value.func
        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) != "RewTerm":
            continue
        entry = {"name": stmt.targets[0].id, "func": None, "default_weight": None}
        for kw in stmt.value.keywords:
            if kw.arg == "func":
                entry["func"] = ast.unparse(kw.value).split(".")[-1]
            elif kw.arg == "weight":
                entry["default_weight"] = _literal_number(kw.value)
        if entry["default_weight"] is None:
            raise ValueError(f"{entry['name']}: no literal weight")
        out.append(entry)
    return out


def knob_overrides_from_cfg_source(path: Path = MDP_CFG_PY, cls: str = REWARDS_CLASS) -> dict[str, str]:
    """``{KRABBY_*_W: term}`` from the ``_overrides`` dict literal in the class's ``__post_init__``."""
    for stmt in _class_node(path, cls).body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__post_init__":
            for node in ast.walk(stmt):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and getattr(node.targets[0], "id", None) == "_overrides" and isinstance(node.value, ast.Dict)):
                    return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise KeyError("_overrides dict not found")


# ------------------------------------------------------------------------------------ pins
def _scalar(v):
    """Keep scalars, strings, None and (nested) lists of them; drop dicts (asset / sensor cfgs)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, list):
        return [_scalar(x) for x in v]
    return None


def _pick(d: dict, keys) -> dict:
    return {k: _scalar(d[k]) for k in keys if k in d and (d[k] is None or _scalar(d[k]) is not None)}


def _rel(path: str | None) -> str | None:
    if not path:
        return path
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def pins_from_dumps(dumps: dict[str, dict], manifest: dict) -> dict:
    """The summary-relevant subset of the Isaac config dumps, one entry per manifest stage phase."""
    pins: dict = {}
    for s in manifest["stages"]:
        phase, task = s["phase"], s["task"]
        if phase not in dumps:
            raise KeyError(f"no dump for phase {phase!r} (have {sorted(dumps)})")
        d = dumps[phase]
        if task not in d:
            raise KeyError(f"dump for {phase} has no task {task!r} (have {sorted(k for k in d if not k.startswith('_'))})")
        env, agent = d[task]["env"], d[task]["agent"]
        rewards = {}
        for name, t in sorted(env["rewards"].items()):
            params = {k: _scalar(v) for k, v in (t.get("params") or {}).items() if not isinstance(v, dict)}
            rewards[name] = {"func": t["func"].split(":")[-1], "weight": t["weight"], "params": params}
        tg = env["scene"]["terrain"]["terrain_generator"]
        terrain = _pick(tg, TERRAIN_KEYS)
        terrain["sub_terrains"] = {n: dict(proportion=st["proportion"], **_pick(st, GEOMETRY_KEYS))
                                   for n, st in sorted(tg["sub_terrains"].items())}
        pk = env["parkours"]["base_parkour"]
        parkour = _pick(pk, ("freeze_terrain_levels", "move_up_frac", "move_down_frac", "num_future_goal_obs",
                             "next_goal_threshold", "reach_goal_delay"))
        cm = env["commands"]["base_velocity"]
        commands = {"ranges": {k: _scalar(v) for k, v in cm["ranges"].items()}, "clips": {k: _scalar(v) for k, v in cm["clips"].items()},
                    **_pick(cm, ("resampling_time_range", "stand_frac", "heading_control_stiffness", "small_commands_to_zero"))}
        events = {}
        for name in EVENT_KEYS:
            ev = env["events"].get(name)
            if ev is None:
                events[name] = None
                continue
            params = {}
            for k, v in (ev.get("params") or {}).items():
                if k == "bank_path":
                    bank = Path(v)
                    params["bank"] = {"file": _rel(v), "sha256": sha256_of(bank) if bank.exists() else None}
                elif isinstance(v, dict) and k in ("velocity_range", "com_range"):
                    params[k] = {kk: _scalar(vv) for kk, vv in v.items()}
                elif not isinstance(v, dict):
                    params[k] = _scalar(v)
            events[name] = {"mode": ev.get("mode"), **_pick(ev, ("interval_range_s", "is_global_time")), "params": params}
        term = env["terminations"]["crab_failure"]
        terminations = {"crab_failure": {k: _scalar(v) for k, v in term["params"].items() if not isinstance(v, dict)},
                        "time_out_terms": sorted(k for k, v in env["terminations"].items() if v.get("time_out"))}
        act = env["actions"]["joint_pos"]
        actions = {**_pick(act, ("scale", "clip", "use_delay", "action_delay_steps", "delay_update_global_steps", "history_length")),
                   "scale": dict(sorted(act["scale"].items())), "clip": {k: _scalar(v) for k, v in sorted(act["clip"].items())},
                   "joint_names": list(act["joint_names"])}
        curriculum = None
        if env.get("curriculum"):
            curriculum = {n: {k: _scalar(v) for k, v in c["params"].items()} for n, c in sorted(env["curriculum"].items())}
        alg = agent["algorithm"]
        agent_pins = {
            "runner": _pick(agent, RUNNER_KEYS),
            "algorithm": _pick(alg, ALGO_KEYS),
            "symmetry": None if alg.get("symmetry_cfg") is None else _pick(alg["symmetry_cfg"], ("mirror_loss_coeff", "use_mirror_loss", "use_data_augmentation")),
            "policy": {**_pick(agent["policy"], POLICY_KEYS), "actor": _pick(agent["policy"]["actor"], ACTOR_KEYS)},
            "estimator": _pick(agent.get("estimator") or {}, ESTIMATOR_KEYS),
            "depth_encoder": _pick(agent.get("depth_encoder") or {}, DEPTH_KEYS) if agent.get("depth_encoder") else None,
        }
        pins[phase] = {
            "task": task, "usd_path": _rel(env["scene"]["robot"]["spawn"]["usd_path"]),
            "episode_length_s": env["episode_length_s"], "decimation": env["decimation"], "sim_dt": env["sim"]["dt"],
            "external_forces_every_iteration": env["sim"]["physx"].get("enable_external_forces_every_iteration"),
            "rewards": rewards, "terrain": terrain, "parkour": parkour, "commands": commands, "events": events,
            "terminations": terminations, "actions": actions, "curriculum": curriculum, "agent": agent_pins,
        }
    return pins


def read_dump_dir(dump_dir: Path, phases) -> dict[str, dict]:
    out = {}
    for p in phases:
        f = Path(dump_dir) / f"cfg_{p}.json"
        if f.exists():
            out[p] = json.loads(f.read_text())
    return out


# --------------------------------------------------------------------------------- derivations
def effective_weights(pure, phase: str, terms=None, knob_map=None) -> dict[str, float]:
    """Window-start reward weights of an RL phase: class defaults overridden through the knob map."""
    terms = terms or reward_terms_from_cfg_source()
    knob_map = knob_map or knob_overrides_from_cfg_source()
    env = pure.ph.phase_env(phase, pure.ph.MAIN_PLANT)
    w = {t["name"]: t["default_weight"] for t in terms}
    for knob, term in knob_map.items():
        if knob in env:
            w[term] = float(env[knob])
    return w


def knob_diff(prev: dict | None, cur: dict) -> dict[str, list]:
    skip = {"KRABBY_PHASE"}
    prev = prev or {}
    return {"added": sorted(k for k in cur if k not in prev and k not in skip),
            "changed": sorted(k for k in cur if k in prev and cur[k] != prev[k] and k not in skip),
            "removed": sorted(k for k in prev if k not in cur and k not in skip)}


def ramps_of(pure, phase: str) -> list[tuple[str, float, float, int, int]]:
    env = pure.ph.phase_env(phase, pure.ph.MAIN_PLANT)
    spec = env.get("KRABBY_PHASEOUT")
    if not spec:
        return []
    return [(r["term_name"], float(r["w0"]), float(r["w1"]), int(r["t0"]), int(r["t1"])) for r in pure.cur.parse_phaseout_spec(spec)]


def _fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3g}"
        s = f"{v:.4g}"
        return s
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    return str(v)


def _w(v: float) -> str:
    return ("+" if v > 0 else "") + _fmt(v)


def _ramp_str(ramps, term: str) -> str:
    for t, w0, w1, t0, t1 in ramps:
        if t == term:
            return f"{_fmt(w0)} -> {_fmt(w1)}"
    return ""


def _phase_ramp_text(pure, phase: str) -> str:
    r = ramps_of(pure, phase)
    return "; ".join(f"{t} {_fmt(w0)} -> {_fmt(w1)}" for t, w0, w1, _, _ in r) if r else "none"


def _stage_of(m: dict, phase: str) -> dict | None:
    return next((s for s in m["stages"] if s["phase"] == phase), None)


def _geometry_name(env: dict) -> str:
    return env.get("KRABBY_FLAT_TERRAIN_GEOM", "shallow (default)")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _sha12(s: str) -> str:
    return f"`{s[:12]}`"


# ---------------------------------------------------------------------------------- renderers
def render_blocks(m: dict, pins: dict, pure=None) -> dict[str, str]:
    pure = pure or load_pure()
    ph, cur, xk = pure.ph, pure.cur, pure.xk
    terms = reward_terms_from_cfg_source()
    knob_map = knob_overrides_from_cfg_source()
    term_knob = {t: k for k, t in knob_map.items()}
    order = [t["name"] for t in terms]
    envs = {p: ph.phase_env(p, ph.MAIN_PLANT) for p in DOCUMENTED_PHASES}
    stages = {s["phase"]: s for s in m["stages"]}
    rl_pinned = [p for p in RL_PHASES if p in pins]
    B: dict[str, str] = {}

    # --- lineage -------------------------------------------------------------------------------
    rows = []
    for s in m["stages"]:
        spec = ph.PHASES[s["phase"]]
        ev = s.get("evals", {})
        def e(k):
            x = ev.get(k)
            return f"{x['completion']:.2f} / {x['falls']} / {x['tripod']:.2f}" if x else "—"
        rows.append([f"[`{s['dir']}/`]({s['dir']}/)", f"`{s['phase']}`", f"`{spec.task}`" + (f" mode `{spec.teacher_mode}`" if spec.teacher_mode else ""),
                     s["iterations"], s.get("resume_from", "—"), f"`{s['file']}` {_sha12(s['sha256'])}", e("slow"), e("step"), e("obst")])
    B["lineage"] = _table(["Stage", "Preset", "Task", "Iterations", "Resumes", "Head (sha12)", "Flat canary", "Step onset", "Obstacles 0.20-0.70"], rows) + \
        "\n\nEvals: completion / falls out of 100 / tripod score (`policy/manifest.yaml`). `3b` (`" + ph.PHASES["3b"].task + \
        "`, +5k on the 0.70-0.90 band) was run and is **not baked**; it has no stage folder."

    # --- contract (values identical in every RL phase; from the 2c pins) -------------------------
    ref = pins.get("2c") or pins[rl_pinned[-1]]
    a = ref["actions"]
    rows = [
        ["Plant / USD", f"`{ref['usd_path']}` (`KRABBY_PLANT` = `{ph.MAIN_PLANT}`, nothing exported)"],
        ["Control rate", f"sim dt {_fmt(ref['sim_dt'])} s x decimation {ref['decimation']} = {1 / (ref['sim_dt'] * ref['decimation']):.0f} Hz"],
        ["Episode length", f"{_fmt(ref['episode_length_s'])} s (= {int(ref['episode_length_s'] / ref['commands']['resampling_time_range'][0])} command holds of {_fmt(ref['commands']['resampling_time_range'][0])} s)"],
        ["Actions (18)", f"scale cam pi = {_fmt(a['scale']['.*_Body_CamShaft_RevoluteJoint'])} rad/s (velocity), hip / knee {_fmt(a['scale']['.*_Hip_Femur_RevoluteJoint'])} rad; clip cam {_fmt(a['clip']['.*_Body_CamShaft_RevoluteJoint'])}, hip / knee {_fmt(a['clip']['.*_Hip_Femur_RevoluteJoint'])}; delay {_fmt(a['action_delay_steps'])} step, history {a['history_length']}"],
        ["Runner action clip", f"`clip_actions` = {_fmt(ref['agent']['runner']['clip_actions'])} (raw policy output)"],
        ["Terminations", f"`crab_failure`: tilt > {_fmt(ref['terminations']['crab_failure']['limit_angle'])} rad or hip contact > {_fmt(ref['terminations']['crab_failure']['contact_force_threshold'])} N; time-out terms {', '.join('`' + t + '`' for t in ref['terminations']['time_out_terms'])}"],
        ["Mirror-symmetry loss", f"coefficient {_fmt(ref['agent']['symmetry']['mirror_loss_coeff'])} in every RL phase; none in distillation" if ref["agent"]["symmetry"] else "off"],
        ["Terrain tiles", f"{_fmt(ref['terrain']['size'])} m, {ref['terrain']['num_rows']} rows x {ref['terrain']['num_cols']} cols, cell {_fmt(ref['terrain']['horizontal_scale'])} m, {ref['terrain']['num_goals']} goals per tile"],
        ["Spawn", f"platform spawn offset {_fmt(ref['events']['reset_root_state']['params'].get('offset'))} m (tile-local x = {_fmt(xk.TILE_WIDTH_M - ref['events']['reset_root_state']['params'].get('offset', 0))} m); friction randomised in {_fmt(ref['events']['physics_material']['params'].get('friction_range'))} ({ref['events']['physics_material']['params'].get('num_buckets')} buckets) in every phase"],
    ]
    B["contract"] = _table(["Fixed across phases", "Value (from the pins)"], rows)

    # --- knob matrix -----------------------------------------------------------------------------
    keys = sorted({k for p in DOCUMENTED_PHASES for k in envs[p]} - {"KRABBY_PHASE"})
    rows = []
    for k in keys:
        cells = []
        for p in DOCUMENTED_PHASES:
            v = envs[p].get(k)
            if v is None:
                cells.append("—")
            elif k == "KRABBY_PHASEOUT":
                cells.append("; ".join(f"{t.replace('reward_', '')} {_fmt(w0)}->{_fmt(w1)}" for t, w0, w1, _, _ in ramps_of(pure, p)))
            elif k == "KRABBY_RSI_BANK":
                cells.append("`" + Path(v).name + "`")
            else:
                cells.append(f"`{v}`")
        rows.append([f"`{k}`"] + cells)
    B["knob-matrix"] = _table(["Knob"] + [f"`{p}`" for p in DOCUMENTED_PHASES], rows) + \
        f"\n\n`KRABBY_PHASEOUT` ramps run over the first {ph.RAMP_ITERS} iterations of the window (= {ph.RAMP_ITERS * ph.STEPS_PER_ITER} env steps, `t0:t1` = 0:{ph.RAMP_ITERS * ph.STEPS_PER_ITER}); eps = {ph.EPS}."

    # --- per phase ---------------------------------------------------------------------------------
    prev_phase = None
    prev_stage = None
    for p in DOCUMENTED_PHASES:
        spec = ph.PHASES[p]
        s = stages.get(p)
        env = envs[p]
        # head
        if s:
            head = [f"- **File:** [`{s['dir']}/{s['file']}`]({s['dir']}/{s['file']}) -- sha256 {_sha12(s['sha256'])} (full in README)",
                    f"- **Source run (not tracked):** `{s['source']}`",
                    f"- **Trained in:** {s['trained_in']}",
                    f"- **Preset:** `KRABBY_PHASE={p}` -> `{spec.task}`" + (f" mode `{spec.teacher_mode}`" if spec.teacher_mode else "") +
                    f", kind `{spec.kind}`, {s['iterations']}, resumes {s.get('resume_from', '—')}; {NUM_ENVS[spec.kind]} envs, seed {SEED_OF_RECORD}, {spec.iterations} iterations (experiment dir `{spec.experiment}`)"]
        else:
            head = [f"- **No stage folder** (not baked). Preset `KRABBY_PHASE={p}` -> `{spec.task}`, kind `{spec.kind}`, {spec.iterations} iterations, resumes `{spec.resume_from}`; notes: {spec.notes}"]
        B[f"phase-head-{p}"] = "\n".join(head)
        # diff
        d = knob_diff(envs[prev_phase] if prev_phase else None, env)
        lines = []
        if prev_phase is None:
            lines.append(f"The formation stack ({len(env) - 1} knobs, `KRABBY_PHASE` excluded):")
            lines += [f"- `{k}={env[k]}`" if k != "KRABBY_RSI_BANK" else f"- `{k}=…/{Path(env[k]).name}`" for k in sorted(env) if k != "KRABBY_PHASE"]
        else:
            for k in d["added"]:
                lines.append(f"- added `{k}={env[k]}`")
            for k in d["changed"]:
                if k == "KRABBY_PHASEOUT":
                    lines.append(f"- ramps this window: {_phase_ramp_text(pure, p)}")
                else:
                    lines.append(f"- `{k}`: `{envs[prev_phase][k]}` -> `{env[k]}`")
            for k in d["removed"]:
                lines.append(f"- dropped `{k}` (was `{envs[prev_phase][k]}`)")
            if not (d["added"] or d["changed"] or d["removed"]):
                lines.append("- no knob changes")
            lines.append(f"- everything else carried over from `{prev_phase}` ({len(env) - 1 - len(d['added']) - len(d['changed'])} knobs unchanged)")
        B[f"phase-diff-{p}"] = "\n".join(lines)
        if p in pins:
            pin = pins[p]
            ramps = ramps_of(pure, p)
            if p in RL_PHASES:
                first_active = {}
                for q in rl_pinned:
                    for t, r in pins[q]["rewards"].items():
                        if r["weight"] != 0 and t not in first_active:
                            first_active[t] = q
                rows = []
                for t in order:
                    r = pin["rewards"].get(t)
                    if not r or r["weight"] == 0:
                        continue
                    notes = []
                    pr = r["params"]
                    for key in ("std", "threshold", "min_clearance_m", "max_speed_scale", "apex_m", "combine", "target_speed", "max_feet_on_ground", "max_idle_steps", "min_actual_speed"):
                        if key in pr:
                            notes.append(f"{key} {_fmt(pr[key])}")
                    if "parkour_name" in pr:
                        if t == "reward_orientation":
                            notes.append("flat tiles only")
                        elif t == "reward_lin_vel_z":
                            notes.append("halved on obstacle tiles")
                        elif t in ("reward_foot_clearance", "penalty_swing_min_clearance"):
                            notes.append("flat mask lifted" if pr["parkour_name"] is None else "obstacle tiles only")
                        elif t in ("reward_tracking_goal_vel", "reward_tracking_yaw", "reward_obstacle_clearance"):
                            notes.append("obstacle tiles only")
                    rows.append([f"[`{t}`](#{t})", _w(r["weight"]), _ramp_str(ramps, t) or "—", ", ".join(notes) or "—", f"`{first_active.get(t, p)}`"])
                B[f"phase-rewards-{p}"] = _table(["Term", "Weight (window start)", "Ramp in this window", "Params of note", "Active since"], rows) + \
                    f"\n\n{len(rows)} of {len(pin['rewards'])} registered terms are non-zero; the rest are inert (see [4.9](#49-registered-but-inert)). Weights are the values at the start of the window; a ramp is a cosine over the first {ph.RAMP_ITERS} iterations of the window."
            else:
                B[f"phase-rewards-{p}"] = ("No RL reward: the algorithm is `" + pin["agent"]["algorithm"]["class_name"] +
                                          "` (teacher-action imitation). The env still registers " +
                                          ", ".join(f"`{t}` (weight {_fmt(r['weight'])})" for t, r in pin["rewards"].items()) +
                                          " -- telemetry only, not a learning signal. No `KRABBY_PHASEOUT` curriculum.")
            # terrain
            tg = pin["terrain"]
            st = tg["sub_terrains"]
            flat = st["parkour_flat"]["proportion"]
            rows = [["Mode", f"`{env.get('KRABBY_FLAT_TERRAIN_MODE', 'off')}` (levels {'frozen' if pin['parkour']['freeze_terrain_levels'] else 'unfrozen'})"],
                    ["Geometry preset", f"`{_geometry_name(env)}`"],
                    ["Flat tiles", f"{_fmt(flat)} of columns; each of gap / hurdle / step / stones {_fmt(st['parkour_gap']['proportion'])} (demo {_fmt(st['parkour_demo']['proportion'])})"],
                    ["Difficulty", f"{_fmt(tg['difficulty_range'])}, curriculum {'on (rows = levels)' if tg['curriculum'] else 'off (uniform draw per tile)'}"],
                    ["Promotion", f"move up above {_fmt(pin['parkour']['move_up_frac'])} x cmd_vx x episode, down below {_fmt(pin['parkour']['move_down_frac'])} x" + (" (inert: levels frozen)" if pin["parkour"]["freeze_terrain_levels"] else "")],
                    ["Gap", f"depth {_fmt(st['parkour_gap'].get('gap_depth'))} m, size `{st['parkour_gap'].get('gap_size')}`, corridor half-width {_fmt(st['parkour_gap'].get('half_valid_width'))} m, y offset {_fmt(st['parkour_gap'].get('y_range'))}"],
                    ["Hurdle", f"height `{st['parkour_hurdle'].get('hurdle_height_range')}`, corridor half-width {_fmt(st['parkour_hurdle'].get('half_valid_width'))} m"],
                    ["Step", f"height `{st['parkour_step'].get('step_height')}`, corridor half-width {_fmt(st['parkour_step'].get('half_valid_width'))} m"],
                    ["Stones + pit", f"pit depth {_fmt(st['parkour'].get('pit_depth'))} m, incline `{st['parkour'].get('incline_height')}`, stone width {_fmt(st['parkour'].get('stone_width'))} m, stone y offset `{st['parkour'].get('y_range')}`"],
                    ["Roughness", f"noise {_fmt(st['parkour_flat'].get('noise_range'))} m on every tile"]]
            B[f"phase-terrain-{p}"] = _table(["Terrain", "Value"], rows)
            # other
            cm, ev, ag = pin["commands"], pin["events"], pin["agent"]
            push, mass, com, rsi = ev["push_by_setting_velocity"], ev["randomize_rigid_body_mass"], ev["randomize_rigid_body_com"], ev["rsi_reference_reset"]
            rows = [["Commands", f"vx {_fmt(cm['ranges']['lin_vel_x'])} m/s, heading {_fmt(cm['ranges']['heading'])} rad (stiffness {_fmt(cm['heading_control_stiffness'])}), walking slots stand_frac {_fmt(cm['stand_frac'])}, resample every {_fmt(cm['resampling_time_range'][0])} s, lin_vel_clip {_fmt(cm['clips']['lin_vel_clip'])}"],
                    ["Episode", f"{_fmt(pin['episode_length_s'])} s"],
                    ["RSI", (f"fraction {_fmt(rsi['params']['fraction'])} from `{Path(rsi['params']['bank']['file']).name}` (sha256 {_sha12(rsi['params']['bank']['sha256'])}), fix_spawn {rsi['params']['fix_spawn']}" if rsi else "off")],
                    ["DR push", (f"±{_fmt(push['params']['velocity_range']['x'][1])} m/s on x / y every {_fmt(push['interval_range_s'][0])} s" if push else "off (event removed)")],
                    ["DR mass", (f"{_fmt(mass['params']['mass_distribution_params'])} kg added to the chassis" if mass else "off (event removed)")],
                    ["DR CoM", (f"±{_fmt(com['params']['com_range']['x'][1])} m on x / y / z" if com else "off (event removed)")],
                    ["Actions", f"scale hip / knee {_fmt(pin['actions']['scale']['.*_Hip_Femur_RevoluteJoint'])}, clip {_fmt(pin['actions']['clip']['.*_Hip_Femur_RevoluteJoint'])}, delay {_fmt(pin['actions']['action_delay_steps'])}, history {pin['actions']['history_length']}; runner clip_actions {_fmt(ag['runner']['clip_actions'])}"],
                    ["Terminations", f"tilt {_fmt(pin['terminations']['crab_failure']['limit_angle'])} rad, hip contact {_fmt(pin['terminations']['crab_failure']['contact_force_threshold'])} N"],
                    ["Algorithm", f"`{ag['algorithm']['class_name']}` lr {_fmt(ag['algorithm']['learning_rate'])} ({ag['algorithm']['schedule']}, desired_kl {_fmt(ag['algorithm']['desired_kl'])}), entropy {_fmt(ag['algorithm']['entropy_coef'])}, clip {_fmt(ag['algorithm']['clip_param'])}, {ag['algorithm']['num_learning_epochs']} epochs x {ag['algorithm']['num_mini_batches']} minibatches, {ag['runner']['num_steps_per_env']} steps/env, gamma {_fmt(ag['algorithm']['gamma'])}, lambda {_fmt(ag['algorithm']['lam'])}"],
                    ["Policy", f"`{ag['policy']['class_name']}` actor / critic {_fmt(ag['policy']['actor_hidden_dims'])}, scan encoder {_fmt(ag['policy']['scan_encoder_dims'])}, priv encoder {_fmt(ag['policy']['priv_encoder_dims'])}, init noise std {_fmt(ag['policy']['init_noise_std'])}"],
                    ["Mirror loss", (f"coefficient {_fmt(ag['symmetry']['mirror_loss_coeff'])}" if ag["symmetry"] else "none")],
                    ["Envs / seed", f"{NUM_ENVS[spec.kind]} envs, seed {SEED_OF_RECORD}"]]
            if ag.get("depth_encoder"):
                de = ag["depth_encoder"]
                rows.append(["Depth encoder", f"`{de['backbone_class_name']}` + `{de['encoder_class_name']}`, image {_fmt(de['depth_shape'])}, hidden {de['hidden_dims']}, lr {_fmt(de['learning_rate'])}, {de['num_steps_per_env']} steps/env"])
            B[f"phase-other-{p}"] = _table(["Configuration", "Value"], rows)
            # evals
            if s and s.get("evals"):
                rows = []
                for key, label in (("slow", "flat canary"), ("step", "step onset"), ("obst", "obstacles 0.20-0.70")):
                    e0 = s["evals"].get(key)
                    if not e0:
                        continue
                    e1 = prev_stage["evals"].get(key) if prev_stage and prev_stage.get("evals") else None
                    delta = f"{e0['completion'] - e1['completion']:+.2f} / {e0['falls'] - e1['falls']:+d}" if e1 else "—"
                    rows.append([label, f"{e0['completion']:.2f}", str(e0["falls"]), f"{e0['tripod']:.3f}", str(e0["n"]), delta])
                B[f"phase-evals-{p}"] = _table(["Scenario", "Completion", "Falls", "Tripod", "n", "vs previous stage (completion / falls)"], rows)
        if s:
            prev_stage = s
        prev_phase = p

    # --- reward matrix + inert list ---------------------------------------------------------------
    active = [t for t in order if any(pins[q]["rewards"].get(t, {}).get("weight", 0) != 0 for q in rl_pinned)]
    rows = []
    for t in active:
        cells = []
        for q in rl_pinned:
            w = pins[q]["rewards"][t]["weight"]
            r = _ramp_str(ramps_of(pure, q), t)
            cells.append("0" if w == 0 else (r if r else _w(w)))
        rows.append([f"[`{t}`](#{t})", f"`{pins[rl_pinned[0]]['rewards'][t]['func']}`"] + cells + [f"`{term_knob[t]}`" if t in term_knob else "—"])
    B["reward-matrix"] = _table(["Term", "Function"] + [f"`{q}`" for q in rl_pinned] + ["Weight knob"], rows) + \
        f"\n\n{len(active)} active terms; cells are the window-start weight or `w0 -> w1` for a term ramped in that window (cosine over the first {ph.RAMP_ITERS} iterations, eps = {ph.EPS}). Phase `3a` has no RL reward."
    inert = [t for t in order if t not in active]
    rows = [[f"`{t}`", f"`{pins[rl_pinned[0]]['rewards'][t]['func']}`", f"`{term_knob[t]}`" if t in term_knob else "—"] for t in inert]
    B["reward-inert"] = _table(["Term", "Function", "Arming knob"], rows) + f"\n\n{len(inert)} terms registered at weight 0 in every phase of record (skipped by the reward manager; their `Episode_Reward/` curves stay at zero)."

    # --- terrain base + geometry + per-phase --------------------------------------------------------
    tg = ref["terrain"]
    rows = [["Tile size", f"{_fmt(tg['size'])} m"], ["Grid", f"{tg['num_rows']} rows x {tg['num_cols']} cols"],
            ["Height-field cell", f"{_fmt(tg['horizontal_scale'])} m horizontal, {_fmt(tg['vertical_scale'])} m vertical"],
            ["Border", f"{_fmt(tg['border_width'])} m"], ["Goals per tile", str(tg["num_goals"])],
            ["Slope threshold", _fmt(tg["slope_threshold"])],
            ["Sub-terrains", ", ".join(f"`{n}`" for n in tg["sub_terrains"])]]
    B["terrain-base"] = _table(["Generator", "Value"], rows)
    g1, g2 = pins[rl_pinned[0]]["terrain"]["sub_terrains"], pins["2a"]["terrain"]["sub_terrains"] if "2a" in pins else pins[rl_pinned[-1]]["terrain"]["sub_terrains"]
    def cell(st, sub, key):
        v = st[sub].get(key)
        return "—" if v is None else (f"`{v}`" if isinstance(v, str) else _fmt(v))
    spec_rows = [("Gap depth (m)", "parkour_gap", "gap_depth"), ("Gap size (m, expr in difficulty d)", "parkour_gap", "gap_size"),
                 ("Gap corridor half-width (m)", "parkour_gap", "half_valid_width"), ("Gap / hurdle / step lateral offset y (m)", "parkour_gap", "y_range"),
                 ("Hurdle height (m)", "parkour_hurdle", "hurdle_height_range"), ("Hurdle corridor half-width (m)", "parkour_hurdle", "half_valid_width"),
                 ("Step height (m)", "parkour_step", "step_height"), ("Step corridor half-width (m)", "parkour_step", "half_valid_width"),
                 ("Pit depth (m)", "parkour", "pit_depth"), ("Incline height (m)", "parkour", "incline_height"),
                 ("Last incline height (m)", "parkour", "last_incline_height"), ("Stone width (m)", "parkour", "stone_width"),
                 ("Stone lateral offset y (m)", "parkour", "y_range"), ("Roughness noise (m)", "parkour_flat", "noise_range")]
    rows = [[label, cell(g1, sub, key), cell(g2, sub, key)] for label, sub, key in spec_rows]
    B["terrain-geometry"] = _table(["Parameter", f"`{_geometry_name(envs['1a'])}` (phase 1a pins)", f"`{_geometry_name(envs['2a'])}` (phases 2a-3a pins)"], rows) + \
        f"\n\n`recal2b2w` = `recal2b2` + the corridor widening from `mdp/exposure_knobs.py`: half-width {_fmt(xk.RECAL2B2W_HALF_VALID_WIDTH)} m on gap / hurdle / step, stone width {_fmt(xk.RECAL2B2W_STONE_WIDTH)} m, lateral offset {_fmt(xk.RECAL2B2W_Y_RANGE)} m, stone offset `{xk.RECAL2B2W_STONE_Y_RANGE}` -- sized for the crab's {_fmt(xk.CRAB_HALF_STANCE_M)} m half-stance on {_fmt(xk.TILE_WIDTH_M)} m wide tiles (bound {_fmt(xk.CORRIDOR_HALF_WIDTH_MAX_M)} m). The `recal2b2` values are the same rows with the stock corridor widths."
    rows = []
    for p in DOCUMENTED_PHASES:
        env = envs[p]
        pin = pins.get(p)
        if pin:
            tg = pin["terrain"]
            rows.append([f"`{p}`", f"`{env.get('KRABBY_FLAT_TERRAIN_MODE', 'off')}`", _fmt(tg["sub_terrains"]["parkour_flat"]["proportion"]), f"`{_geometry_name(env)}`", _fmt(tg["difficulty_range"]),
                         "on" if tg["curriculum"] else "off", f"{_fmt(pin['parkour']['move_up_frac'])} : {_fmt(pin['parkour']['move_down_frac'])}", "yes" if pin["parkour"]["freeze_terrain_levels"] else "no"])
        else:
            rows.append([f"`{p}`", f"`{env.get('KRABBY_FLAT_TERRAIN_MODE', 'off')}`", env.get("KRABBY_FLAT_TERRAIN_FLAT_FRAC", "—"), f"`{_geometry_name(env)}`", env.get("KRABBY_FLAT_TERRAIN_DIFF", "—"),
                         "on" if env.get("KRABBY_FLAT_TERRAIN_CURRICULUM") else "off", env.get("KRABBY_TERRAIN_PROMOTE", "—").replace(":", " : "), "no" if env.get("KRABBY_FLAT_TERRAIN_CURRICULUM") else "yes"])
    B["terrain-phases"] = _table(["Phase", "Mode", "Flat fraction", "Geometry", "Difficulty", "Curriculum", "Promote up : down", "Levels frozen"], rows) + \
        "\n\n(`3b` from its preset: no pins, not baked.)"

    # --- other-config per phase ---------------------------------------------------------------------
    items = [("Commands vx (m/s)", lambda e, q: e.get("KRABBY_LIN_VEL_X", "—")), ("Walking slots stand_frac", lambda e, q: e.get("KRABBY_STAND_FRAC", "—")),
             ("Resample (s)", lambda e, q: e.get("KRABBY_RESAMPLE_S", "—")), ("Heading (rad)", lambda e, q: e.get("KRABBY_HEADING", "0:0")),
             ("Episode (s)", lambda e, q: e.get("KRABBY_EPISODE_S", "—")), ("RSI fraction", lambda e, q: e.get("KRABBY_RSI_FRAC", "off")),
             ("DR push (m/s)", lambda e, q: e.get("KRABBY_DR_PUSH", "off")), ("DR mass (kg)", lambda e, q: e.get("KRABBY_DR_MASS", "off")),
             ("DR CoM (m)", lambda e, q: e.get("KRABBY_DR_COM", "off")),
             ("Mirror loss", lambda e, q: (_fmt(q["agent"]["symmetry"]["mirror_loss_coeff"]) if q and q["agent"]["symmetry"] else ("none" if q else "—"))),
             ("Learning rate", lambda e, q: _fmt(q["agent"]["algorithm"]["learning_rate"]) if q else "—"),
             ("Minibatches", lambda e, q: str(q["agent"]["algorithm"]["num_mini_batches"]) if q else "—"),
             ("Envs", lambda e, q: "—")]
    rows = []
    for label, fn in items:
        cells = []
        for p_ in DOCUMENTED_PHASES:
            if label == "Envs":
                cells.append(str(NUM_ENVS[ph.PHASES[p_].kind]))
            else:
                cells.append(f"`{fn(envs[p_], pins.get(p_))}`")
        rows.append([label] + cells)
    B["other-phases"] = _table(["Setting"] + [f"`{p}`" for p in DOCUMENTED_PHASES], rows)

    # --- evals ---------------------------------------------------------------------------------------
    rows = []
    prev = None
    for s in m["stages"]:
        ev = s.get("evals", {})
        cells = []
        for key in ("slow", "step", "obst"):
            e0 = ev.get(key)
            e1 = prev["evals"].get(key) if prev and prev.get("evals") else None
            cells.append(f"{e0['completion']:.2f} / {e0['falls']} / {e0['tripod']:.2f}" + (f" ({e0['completion'] - e1['completion']:+.2f})" if e1 and e0 else "") if e0 else "—")
        rows.append([f"`{s['phase']}` `{s['dir']}`"] + cells)
        prev = s
    B["evals"] = _table(["Stage", "Flat canary", "Step onset", "Obstacles 0.20-0.70"], rows) + \
        "\n\nCompletion / falls (of 100 episodes) / tripod score; parentheses = completion delta vs the previous stage. Source: `policy/manifest.yaml` (the producing campaigns' `state.json`)."

    # --- legacy presets, env stacks, knob map ---------------------------------------------------------
    legacy = [n for n in ph.PHASES if n.startswith("legacy_golden")]
    B["legacy-presets"] = ", ".join(f"`{n}`" for n in legacy) + f" ({len(legacy)} presets; all on `{ph.PHASES[legacy[0]].task}`, 20 s episodes, no walking slots, `recal2b2` with promotion `{ph.PHASES['legacy_golden_2a'].env['KRABBY_TERRAIN_PROMOTE']}` from `legacy_golden_2a`, the P0-null RSI bank for 1a / 2a and the per-window `pg_r*` banks from 2b)."
    parts = []
    for p in DOCUMENTED_PHASES:
        env = envs[p]
        body = "\n".join(f"{k}={env[k]}" if k != "KRABBY_RSI_BANK" else f"{k}=<pkg>/{Path(env[k]).relative_to(PKG)}" for k in sorted(env))
        parts.append(f"`{p}` ({len(env)} keys):\n\n```\n{body}\n```")
    B["env-stacks"] = "\n\n".join(parts)
    rows = [[f"`{k}`", f"`{t}`", "weight"] for k, t in knob_map.items()]
    rows += [[f"`{k}`", f"`{t}`", what] for k, (t, what) in PARAM_KNOBS.items()]
    B["knob-map"] = _table(["Knob", "Reward term", "Sets"], rows) + "\n\nWeight knobs are presence-based: an explicitly exported value wins even at `0` (that is how a baked default is disabled)."
    return B


# ------------------------------------------------------------------------------------ markers
_MARK = re.compile(r"<!-- generated:([a-z0-9-]+) -->\n(.*?)<!-- /generated:\1 -->", re.DOTALL)


def block_ids(text: str) -> list[str]:
    return [m.group(1) for m in _MARK.finditer(text)]


def update_summary(text: str, blocks: dict[str, str]) -> str:
    ids = block_ids(text)
    missing = [i for i in ids if i not in blocks]
    unused = [i for i in blocks if i not in ids]
    if missing or unused:
        raise SystemExit(f"POLICY_SUMMARY.md markers out of sync with the renderer: no renderer for {missing}; no marker for {unused}")
    return _MARK.sub(lambda mm: f"<!-- generated:{mm.group(1)} -->\n{blocks[mm.group(1)]}\n<!-- /generated:{mm.group(1)} -->", text)


def render_summary(m: dict | None = None, pins: dict | None = None) -> str:
    m = m or load_manifest()
    pins = pins or load_pins()
    return update_summary(SUMMARY.read_text(), render_blocks(m, pins))


def check_summary(m: dict) -> list[str]:
    problems = []
    if not PINS.exists():
        return [f"{PINS.relative_to(REPO)}: missing (bundle_policy.py --pin-mdp <dump dir>)"]
    pins = load_pins()
    for s in m["stages"]:
        if s["phase"] not in pins:
            problems.append(f"mdp_pins.yaml: no entry for stage {s['dir']} (phase {s['phase']})")
    if not SUMMARY.exists():
        return problems + [f"{SUMMARY.relative_to(REPO)}: missing"]
    if problems:
        return problems
    text = SUMMARY.read_text()
    try:
        if update_summary(text, render_blocks(m, pins)) != text:
            problems.append(f"{SUMMARY.relative_to(REPO)}: generated blocks are stale (bundle_policy.py --sync)")
    except SystemExit as e:  # marker / renderer mismatch
        problems.append(str(e))
    return problems
