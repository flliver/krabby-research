# SPDX-License-Identifier: BSD-3-Clause
"""Training phases of record for the crab hexapod (paradigm restored 2026-09-07).

Phase 1 (pure student)        1a   Isaac-Crab-Hex-Flat-Walk-v0        0 -> 5k     gait formation
Phase 2 (teacher-student)     2a   Isaac-Crab-Hex-Teacher-v0 mode 2a  5k -> 10k   elements @5k, apex/airtime/stride -> half
                              2b   Isaac-Crab-Hex-Teacher-v0 mode 2b  10k -> 15k  elements @10k, satellites -> eps
                              2c   Isaac-Crab-Hex-Teacher-v0 mode 2c  15k -> 20k  clock 1.0 -> 0.5   (= policy of record)
Phase 3 (student distillation) 3a  Isaac-Crab-Hex-Student-v0          20k -> +5k  depth student on the 2c MDP
                              3b   Isaac-Crab-Hex-Student-v0          +5k -> +10k difficulty band 0.70-0.90

Each phase is a dict of the SAME ``KRABBY_*`` environment knobs the fine-tuning campaigns used;
``activate_phase()`` expands ``KRABBY_PHASE`` (+ ``KRABBY_PLANT``) into the process environment
with ``setdefault`` before any config module reads it, so the knob code paths -- and therefore
the trained MDPs -- are unchanged and bit-identical to the recorded campaigns
(``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-06_2130_a15b_lineage/run_lineage.py::window_stack`` for 1a-2c,
``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure/run_exposure.py::lineage_stack`` for the
legacy golden schedule). Explicitly set variables always win over the preset.

Pure Python (no Isaac imports): unit-tested in ``tests/unit/test_crab_hex_phases.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parents[6]
ASSETS = REPO_ROOT / "assets"
EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"  # the task's campaign records

EPS = "0.001"
STEPS_PER_ITER = 24
RAMP_ITERS = 1000
WINDOW_ITERS = 5000

# ---------------------------------------------------------------------------------- plants
# Plant name -> USDA path relative to assets/ (None = the MAIN asset ``assets/crab.usda``, which
# needs no variable). Since 2026-09-09 the main asset IS the A15+B geometry (15 deg outer-mount
# splay, outer yaw axes 2.5 in from the body ends). Records written before that date say
# "golden" / "base" for the 2026-08-20 measured-hardware build: that is ``legacy_golden`` =
# ``assets/variants/crab_simple__splay00_axis5p5in.usda``.
# The table is locked to ``assets/scripts/generate_crab.py``'s VARIANTS by a unit test.
MAIN_PLANT = "A15+B"
MAIN_ASSET = ASSETS / "crab.usda"
# file names that mean "the main plant" (the variant file is byte-identical to crab.usda and is
# what the a15b lineage / phase-pipeline runs recorded)
MAIN_ASSET_NAMES = ("crab.usda", "crab_simple__splay15_axis2p5in.usda")
PLANTS: dict[str, str | None] = {
    "A15+B": None,
    "main": None,
    # 2026-08-20 measured-hardware build (splay 0, axes 5.5 in): the "golden"/"base" plant of every
    # pre-a15b head. (``assets/crab_simple.usda`` is NOT a plant: since 2026-09-09 it is the hand-authored
    # 2026-08-09 campaign-baseline Cube model, kept for reference; no plant name maps to it.)
    "legacy_golden": "variants/crab_simple__splay00_axis5p5in.usda",
    "golden": "variants/crab_simple__splay00_axis5p5in.usda",   # alias used by pre-2026-09-09 records and commands
    "B": "variants/crab_simple__splay00_axis2p5in.usda",
    "A10": "variants/crab_simple__splay10_axis5p5in.usda",
    "A15": "variants/crab_simple__splay15_axis5p5in.usda",
    "A20": "variants/crab_simple__splay20_axis5p5in.usda",
    "A10+B": "variants/crab_simple__splay10_axis2p5in.usda",
    "A20+B": "variants/crab_simple__splay20_axis2p5in.usda",
}


def plant_usd_path(name: str) -> str | None:
    """Absolute USDA path for a named plant; ``None`` for the main plant (config default)."""
    if name not in PLANTS:
        raise KeyError(f"unknown plant {name!r}; known: {sorted(PLANTS)}")
    rel = PLANTS[name]
    return None if rel is None else str(ASSETS / rel)


def plant_name_for_path(path: str | None) -> str | None:
    """Reverse lookup by file name: which named plant a spawned USD is (``None`` if unknown)."""
    if not path:
        return None
    name = Path(str(path)).name
    if name in MAIN_ASSET_NAMES:
        return MAIN_PLANT
    for plant, rel in PLANTS.items():
        if rel is not None and Path(rel).name == name:
            return plant  # first match wins: "legacy_golden" precedes its "golden" alias
    return None


# ---------------------------------------------------------------------------------- knob blocks
# The 0-5k formation config of record (== run_formation_arms.BASELINE == round-search BASELINE).
FORMATION: dict[str, str] = {
    "KRABBY_CLOCK_W": "1.0",
    "KRABBY_APEX_W": "1.0",
    "KRABBY_RSI_FRAC": "0.2",
    "KRABBY_RSI_BANK": str(EXPERIMENTS / "2026-08-26_2200_gated_lineage/rsi_bank_P0_null.npz"),
    "KRABBY_FLAT_TERRAIN_MODE": "light",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    "KRABBY_TRACK_SIGMA2": "0.1",
    "KRABBY_TRACK_L1_W": "-1.0",
    "KRABBY_LIN_VEL_X": "0.0:0.35",
}
# Graduated curriculum elements of record: placed at the start of window 1 (@5k) and window 2 (@10k).
ELEMENTS_5K: dict[str, str] = {
    "KRABBY_YAW_W": "0.2",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5",
    "KRABBY_FLAT_TERRAIN_GEOM": "recal2b2",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70",
    "KRABBY_FLAT_TERRAIN_CURRICULUM": "1",
    "KRABBY_TERRAIN_PROMOTE": "0.45:0.25",
    "KRABBY_DR_PUSH": "0.5",
    "KRABBY_DR_MASS": "-0.5:1.5",
    "KRABBY_DR_COM": "0.01",
    "KRABBY_EDGE_W": "-0.3",
    "KRABBY_STUMBLE_W": "-1.0",
    "KRABBY_COLLISION_W": "-2.0",
}
ELEMENTS_10K: dict[str, str] = {
    "KRABBY_CLEARANCE_W": "0.9",
    "KRABBY_FOOT_CLEAR_FLAT": "1",
    "KRABBY_FOOT_CLEAR_W": "1.0",
    "KRABBY_FOOT_CLEAR_MIN": "0.03",
    "KRABBY_SWING_MIN_CLEAR_W": "-0.4",
    "KRABBY_HEADING": "-1.2:1.2",
    "KRABBY_GOAL_VEL_W": "0.75",
}
# Validated training-side levers (morphology x exposure campaign, 2026-09-05/06).
WALKING_SLOTS: dict[str, str] = {"KRABBY_STAND_FRAC": "0.2"}
LONG_EPISODES: dict[str, str] = {"KRABBY_EPISODE_S": "40", "KRABBY_RESAMPLE_S": "10:10"}
WIDENED_CORRIDORS: dict[str, str] = {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2w"}


def promote_fracs_for_horizon(up: float, down: float, horizon_s: float, ref_s: float = 20.0) -> str:
    """Promotion distance = frac x cmd x T: scale both fractions by ref/T to hold the equilibrium."""
    if horizon_s <= 0 or ref_s <= 0:
        raise ValueError("horizons must be positive")
    k = ref_s / horizon_s
    return f"{up * k:.4g}:{down * k:.4g}"


# Gait-income schedule of record (PLAN G bake): start weights per window and cosine ramps
# (term, w0, w1) over the first RAMP_ITERS of the window. Window index w = 5k*w -> 5k*(w+1).
GAIT_WEIGHTS: dict[int, dict[str, str]] = {
    0: {"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "1.0"},
    1: {"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "1.0", "KRABBY_AIRTIME_W": "0.8", "KRABBY_STRIDE_W": "0.5"},
    2: {"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "0.5", "KRABBY_AIRTIME_W": "0.4", "KRABBY_STRIDE_W": "0.25"},
    3: {"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": EPS, "KRABBY_AIRTIME_W": EPS, "KRABBY_STRIDE_W": EPS},
    4: {"KRABBY_CLOCK_W": "0.5", "KRABBY_APEX_W": EPS, "KRABBY_AIRTIME_W": EPS, "KRABBY_STRIDE_W": EPS},
    5: {"KRABBY_CLOCK_W": "0.2", "KRABBY_APEX_W": EPS, "KRABBY_AIRTIME_W": EPS, "KRABBY_STRIDE_W": EPS},
}
GAIT_RAMPS: dict[int, list[tuple[str, float, float]]] = {
    0: [],
    1: [("reward_clock_swing_apex", 1.0, 0.5), ("reward_feet_air_time_positive", 0.8, 0.4),
        ("reward_stride_length", 0.5, 0.25)],
    2: [("reward_clock_swing_apex", 0.5, 0.001), ("reward_feet_air_time_positive", 0.4, 0.001),
        ("reward_stride_length", 0.25, 0.001)],
    3: [("reward_clock_schedule", 1.0, 0.5)],
    4: [("reward_clock_schedule", 0.5, 0.2)],
    5: [("reward_clock_schedule", 0.2, 0.001)],
}


def phaseout_spec(ramps: list[tuple[str, float, float]]) -> str:
    return ",".join(f"{t}:{w0}:{w1}:0:{RAMP_ITERS * STEPS_PER_ITER}" for t, w0, w1 in ramps)


# Legacy golden lineage: the RSI bank re-harvested at each bake and active in the next window.
LEGACY_GOLDEN_BANKS: dict[int, str] = {
    0: FORMATION["KRABBY_RSI_BANK"],
    1: FORMATION["KRABBY_RSI_BANK"],
    2: str(EXPERIMENTS / "2026-08-31_1414_gait_income_phaseout/rsi_bank_pg_r1.npz"),
    3: str(EXPERIMENTS / "2026-08-31_1414_gait_income_phaseout/rsi_bank_pg_r2.npz"),
    4: str(EXPERIMENTS / "2026-08-31_1414_gait_income_phaseout/rsi_bank_pg_r3.npz"),
    5: str(EXPERIMENTS / "2026-08-31_1414_gait_income_phaseout/rsi_bank_pg_r4.npz"),
}


# ---------------------------------------------------------------------------------- phase specs
@dataclass(frozen=True)
class PhaseSpec:
    name: str
    task: str
    iterations: int
    env: dict[str, str]
    resume_from: str | None            # previous phase name (None = from scratch)
    teacher_mode: str | None = None    # KRABBY_HEX_TEACHER_MODE for Teacher-v0 phases
    experiment: str = "crab_hex_flat_walk"
    kind: str = "rl"                   # "rl" | "distill"
    window: int | None = None          # legacy window index (5k units)
    notes: str = ""
    eval_scenarios: tuple[str, ...] = field(default=("slow", "step", "obst"))


def _window_env(w: int, *, elements: bool, levers: bool, legacy: bool) -> dict[str, str]:
    """Env stack of lineage window ``w`` (schedule of record). ``levers`` = the validated
    training-side levers (walking slots, 40-s episodes, widened corridors, promotion x 20/40);
    ``legacy`` = the golden lineage's per-window banks."""
    ev = dict(FORMATION)
    if elements and w >= 1:
        ev.update(ELEMENTS_5K)
    if elements and w >= 2:
        ev.update(ELEMENTS_10K)
    if levers:
        ev.update(WALKING_SLOTS)
        ev.update(LONG_EPISODES)
        if w >= 1:
            ev.update(WIDENED_CORRIDORS)
            ev["KRABBY_TERRAIN_PROMOTE"] = promote_fracs_for_horizon(0.45, 0.25, 40.0)
    ev.update(GAIT_WEIGHTS[w])
    if GAIT_RAMPS[w]:
        ev["KRABBY_PHASEOUT"] = phaseout_spec(GAIT_RAMPS[w])
    ev["KRABBY_RSI_BANK"] = LEGACY_GOLDEN_BANKS[w] if legacy else FORMATION["KRABBY_RSI_BANK"]
    return ev


# Keys that define the MDP a distillation student must share with its teacher (no reward
# weights, no reward anneal, no gait-income terms: distillation has no RL reward).
_MDP_KEYS = (
    "KRABBY_LIN_VEL_X", "KRABBY_HEADING", "KRABBY_STAND_FRAC", "KRABBY_EPISODE_S", "KRABBY_RESAMPLE_S",
    "KRABBY_FLAT_TERRAIN_MODE", "KRABBY_FLAT_TERRAIN_FLAT_FRAC", "KRABBY_FLAT_TERRAIN_DIFF",
    "KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_FLAT_TERRAIN_CURRICULUM", "KRABBY_TERRAIN_PROMOTE",
    "KRABBY_DR_PUSH", "KRABBY_DR_MASS", "KRABBY_DR_COM", "KRABBY_RSI_FRAC", "KRABBY_RSI_BANK",
)


def _student_env(teacher_env: Mapping[str, str], diff: str | None = None) -> dict[str, str]:
    ev = {k: teacher_env[k] for k in _MDP_KEYS if k in teacher_env}
    if diff is not None:
        ev["KRABBY_FLAT_TERRAIN_DIFF"] = diff
    return ev


def build_phases() -> dict[str, PhaseSpec]:
    p: dict[str, PhaseSpec] = {}
    # --- schedule of record (A15+B lineage, validated levers) ---
    p["1a"] = PhaseSpec("1a", "Isaac-Crab-Hex-Flat-Walk-v0", WINDOW_ITERS, _window_env(0, elements=False, levers=True, legacy=False),
                        None, None, "crab_hex_flat_walk", "rl", 0, "phase 1 pure student: gait formation with walking slots and 40-s episodes")
    for name, w, note in (("2a", 1, "elements @5k, apex/airtime/stride -> half"),
                          ("2b", 2, "elements @10k, satellites -> eps"),
                          ("2c", 3, "clock 1.0 -> 0.5 (policy of record)")):
        prev = "1a" if w == 1 else f"2{chr(ord('a') + w - 2)}"
        p[name] = PhaseSpec(name, "Isaac-Crab-Hex-Teacher-v0", WINDOW_ITERS, _window_env(w, elements=True, levers=True, legacy=False),
                            prev, name, "crab_hex_teacher", "rl", w, f"phase 2 teacher-student: {note}")
    env_2c = p["2c"].env
    p["3a"] = PhaseSpec("3a", "Isaac-Crab-Hex-Student-v0", WINDOW_ITERS, _student_env(env_2c), "2c", None,
                        "crab_hex_student", "distill", None, "phase 3 student distillation on the 2c MDP")
    p["3b"] = PhaseSpec("3b", "Isaac-Crab-Hex-Student-v0", WINDOW_ITERS, _student_env(env_2c, diff="0.70:0.90"), "3a", None,
                        "crab_hex_student", "distill", None, "phase 3 student distillation, harder band 0.70-0.90")
    # --- legacy golden schedule (20-s episodes, recal2b2, pg banks; reproduction only) ---
    p["legacy_golden_1a"] = PhaseSpec("legacy_golden_1a", "Isaac-Crab-Hex-Flat-Walk-v0", WINDOW_ITERS,
                                      _window_env(0, elements=False, levers=False, legacy=True), None, None,
                                      "crab_hex_flat_walk", "rl", 0, "golden lineage window 0")
    for w, name in ((1, "2a"), (2, "2b"), (3, "2c"), (4, "2d"), (5, "2e")):
        prev = "legacy_golden_1a" if w == 1 else f"legacy_golden_2{chr(ord('a') + w - 2)}"
        p[f"legacy_golden_{name}"] = PhaseSpec(f"legacy_golden_{name}", "Isaac-Crab-Hex-Flat-Walk-v0", WINDOW_ITERS,
                                               _window_env(w, elements=True, levers=False, legacy=True), prev, None,
                                               "crab_hex_flat_walk", "rl", w, f"golden lineage window {w}")
    return p


PHASES: dict[str, PhaseSpec] = build_phases()

PHASE_MODES = {name: spec.teacher_mode for name, spec in PHASES.items() if spec.teacher_mode}
STUDENT_PHASES = tuple(name for name, spec in PHASES.items() if spec.kind == "distill")


def phase_env(name: str, plant: str | None = None) -> dict[str, str]:
    """Full environment a phase expands to (knobs + teacher mode + plant), without touching os.environ."""
    spec = PHASES[name]
    ev = dict(spec.env)
    ev["KRABBY_PHASE"] = name
    if spec.teacher_mode:
        ev["KRABBY_HEX_TEACHER_MODE"] = spec.teacher_mode
    if plant:
        ev["KRABBY_PLANT"] = plant
        usd = plant_usd_path(plant)
        if usd:
            ev["KRABBY_HEX_USD_PATH"] = usd
    return ev


def activate_phase(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Expand ``KRABBY_PHASE`` / ``KRABBY_PLANT`` into ``environ`` with setdefault.

    Returns the keys that were set (empty when no phase is requested). Explicit variables win.
    """
    env = os.environ if environ is None else environ
    name = env.get("KRABBY_PHASE", "").strip()
    plant = env.get("KRABBY_PLANT", "").strip() or None
    if not name and not plant:
        return {}
    applied: dict[str, str] = {}
    expanded = phase_env(name, plant) if name else {}
    if plant and not name:
        usd = plant_usd_path(plant)
        if usd:
            expanded["KRABBY_HEX_USD_PATH"] = usd
    for k, v in expanded.items():
        if k in ("KRABBY_PHASE", "KRABBY_PLANT"):
            continue
        if k not in env:
            env[k] = v
            applied[k] = v
    return applied


def is_student_phase(environ: Mapping[str, str] | None = None) -> bool:
    """True when the student cfg must mirror the phase-2c teacher MDP: a phase-3 preset is active,
    or ``KRABBY_STUDENT_MDP=1`` asks for that structure without the preset's terrain / RSI / DR
    knobs (evaluation of a phase-3 student on the same explicit knobs a teacher eval uses)."""
    env = os.environ if environ is None else environ
    if env.get("KRABBY_PHASE", "").strip() in STUDENT_PHASES:
        return True
    return env.get("KRABBY_STUDENT_MDP", "").strip().lower() in ("1", "true", "yes", "phase")
