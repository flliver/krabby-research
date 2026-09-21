# SPDX-License-Identifier: BSD-3-Clause
"""Command categories of the omni suite and "which of them a stage trained" (plan 2026-09-21, D4/D5).

Isaac-free and outside ``mdp/`` and ``config/`` (so outside ``code_sha``; it is an EVAL-side file,
hashed into ``eval_sha``). The command math and the knob parsers are loaded BY PATH
(``importlib.util.spec_from_file_location``, the ``run_candidates.pool_math()`` pattern): a dotted
``parkour_tasks...`` import would run the package ``__init__``'s Isaac ``import_packages``.

THE CONTRACTS (each pinned by ``tests/unit/test_crab_hex_omni_eval_categories.py``)

* **Categories and holds.** :data:`CATEGORIES` = ``stand, fwd, side, back, diag, turn, move_turn``;
  :data:`CATEGORY_HOLDS` maps each to its suite hold label(s) -- one hold, or a ``(l, r)`` pair for
  ``side`` / ``turn`` / ``move_turn`` (:data:`HAND_OF_HOLD`). The union of the holds is exactly the
  suite row's labels (``suite__omni`` in ``scenarios_omni.yaml``; the ``settle`` hold is never an
  episode's own hold and belongs to no category).

* **trained_categories(env) -> TrainedCategories.** ``env`` is the COMPOSED training env of a stage
  (``ph.phase_env(stage.base) | stage overlay``), never the overlay alone. A key the env does not
  carry takes the command cfg's CLASS DEFAULT (:data:`CMD_CFG_DEFAULTS`, pinned against the class
  body's source). Shares are NORMALISED (``weight / sum(weights)``), so an un-renormalised
  ``set_share`` edit gives the same sets. ``present`` = share > 0; ``decisive`` = share >=
  ``POOL_FLOOR_SHARE`` AT THE SUITE'S FULL DEMAND; ``watch`` = present - decisive, each with its
  note (``watch (reduced demand)``, ``watch (one hand)``, or ``watch (below floor share)``).
  Pool mode reads ``KRABBY_OMNI_CMD_POOL`` through :data:`TYPE_TO_CATEGORIES`; ``omni`` supports all
  four bearings; ``side`` needs BOTH hands (or ``omni``); a scale < 1 demotes only the categories its
  own type maps to (``{"*_turn": 0.3}`` touches ``move_turn`` alone, never ``turn``). Slots mode (no
  pool) maps slot 0 -> stand, 1 -> ``omni`` or the types whose band lies inside ``dir_band``,
  2 -> turn, 3 -> ``omni_turn`` or the band types' ``*_turn``. ``turn`` is decisive whatever the
  FACING band or the observation encoding (both only ANNOTATE: :func:`turn_annotation`). A stand
  share of 0 raises :class:`EvalMismatch` (the driver's ``halt:eval_mismatch``).

* **expand_rule(rule, decisive, present, *, absolute_units) -> dict.** Turns a ``rule_version: 2``
  decision rule (its ``per_category`` templates, :data:`RULE_V2_PER_CATEGORY`) into TODAY'S FLAT
  SHAPE -- ``blocks: {name: {level, metrics: [...]}}``, ``guards: {level, metrics: [...]}``,
  ``watch`` -- with metric ids ``cat.<c>.<key>`` (dotted paths into ``summarize()['cat']``) and the
  composite ``cat_tripod_min``. One unit system per call: ratio -> blocks ``F, F_mt, T``, guard
  ``cat.move_turn.ratio``; ``absolute_units`` -> ``F_abs, F_mt_abs, T_abs``, guard
  ``cat.move_turn.achieved_speed`` (``guards_abs``). Block members and non-survival guards are kept
  for DECISIVE categories only; the ``survival`` guard is kept for every PRESENT category; every other
  member/guard of a non-decisive category moves to ``watch`` with its note (``untrained`` for a
  category the stage does not train at all). A block left with no member stays in ``blocks`` with
  ``metrics: []`` and is listed in ``na_blocks`` -- N/A, never TIE. ``watch`` is a dict
  ``{id: note | None}`` (iterating it yields the ids, which is all today's readers do).
  :func:`expand_rule_union` is the UNION of both unit systems (six blocks, ten guards at B0), which
  resolution's D00s / D00bs use.

* **composite_guards(screen, decisive) -> dict.** ``{"cat_tripod_min": min(cat.<c>.tripod)}`` over
  the decisive TRANSLATION categories (``fwd, side, back, diag`` -- never ``move_turn``); ``None``
  when any of those tripods is missing (a MISSING value, never a silently narrower min); ``{}`` when
  no translation category is decisive. :func:`with_composites` is the shallow-copy injection every
  comparison applies to both screens.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
CMD_MATH_PY = HERE / "mdp" / "crab_hex_omni_command_math.py"
KNOBS_PY = HERE / "config" / "crab_hex" / "omni_knobs.py"
#: Where the class defaults below are pinned (by source text, never by import).
CMD_CFG_PY = HERE / "mdp" / "crab_hex_omni_command.py"
CMD_CFG_CLASS = "CrabHexOmniCommandCfg"
ENV_CFG_PY = HERE / "config" / "crab_hex" / "crab_hex_omni_env_cfg.py"

# ------------------------------------------------------------------ by-path loads (no Isaac)
_MATH = None
_KNOBS = None


def _load(path: Path, name: str):
    """Import a module by PATH -- no package, no Isaac import (the campaign-driver precedent)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def command_math():
    """``crab_hex_omni_command_math`` (``POOL_TYPES``, ``POOL_FLOOR_SHARE``, ``parse_cmd_pool``).

    Lazy: it imports torch, and a reader that only wants :data:`CATEGORIES` should not pay for it.
    """
    global _MATH
    if _MATH is None:
        _MATH = _load(CMD_MATH_PY, "crab_hex_omni_command_math__eval_categories")
    return _MATH


def knobs():
    """``omni_knobs`` (the parsers every env value goes through), loaded by path."""
    global _KNOBS
    if _KNOBS is None:
        _KNOBS = _load(KNOBS_PY, "omni_knobs__eval_categories")
    return _KNOBS


# ------------------------------------------------------------------ categories and holds
CATEGORIES = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
"""The seven categories, in the suite's wave order. No category is WATCH-only by construction: a
category is decisive when the recipe trains it at the suite's demand and watched otherwise."""

TRANSLATION_CATEGORIES = ("fwd", "side", "back", "diag")
"""The categories ``cat_tripod_min`` is taken over (``move_turn`` is not one: its tripod is WATCH)."""

PAIRED_CATEGORIES = ("side", "turn", "move_turn")
HANDS = ("l", "r")

CATEGORY_HOLDS: dict[str, tuple[str, ...]] = {
    "stand": ("stand",),
    "fwd": ("fwd",),
    "side": ("lat_l", "lat_r"),
    "back": ("back",),
    "diag": ("diag",),
    "turn": ("turn_l90", "turn_r90"),
    "move_turn": ("move_turn", "move_turn_r"),
}
"""Category -> its suite hold labels; a pair is ``(l, r)``. The legacy label vocabulary is kept
wherever the command is the same, so every per-hold reader consumes these unchanged."""

HAND_OF_HOLD: dict[str, str] = {
    "lat_l": "l", "lat_r": "r",
    "turn_l90": "l", "turn_r90": "r",
    "move_turn": "l", "move_turn_r": "r",
}
"""The hand of every PAIRED hold (singleton holds carry no hand)."""

SUITE_HOLDS = ("stand", "fwd", "lat_l", "lat_r", "back", "diag", "turn_l90", "turn_r90",
               "move_turn", "move_turn_r")
"""The suite's own holds in wave order (plan D1's table; the ``settle`` hold is not one)."""

SETTLE_LABEL = "settle"

#: The suite's in-place turn demand (turn_l90 / turn_r90: 90 degrees from face 0), for the
#: FACING-band annotation.
SUITE_TURN_DEMAND_RAD = math.pi / 2.0

CAT_KEYS = ("n", "n_env", "survival", "ratio", "achieved_speed", "tripod", "facing_closed",
            "yaw_progress", "net_yaw", "settle_s", "heading_final", "drift_speed", "direction_error",
            "cross_track_ratio")
"""The keys of ``summarize()['cat'][c]`` (D5); ``by_hand`` holds the same keys per hand."""

COMPOSITES = ("cat_tripod_min",)

TYPE_TO_CATEGORIES: dict[str, frozenset[str]] = {
    "stand": frozenset({"stand"}),
    "fwd": frozenset({"fwd"}),
    "side_l": frozenset({"side"}),
    "side_r": frozenset({"side"}),
    "back": frozenset({"back"}),
    "omni": frozenset({"fwd", "side", "back", "diag"}),
    "turn": frozenset({"turn"}),
    "fwd_turn": frozenset({"move_turn"}),
    "side_l_turn": frozenset({"move_turn"}),
    "side_r_turn": frozenset({"move_turn"}),
    "back_turn": frozenset({"move_turn"}),
    "omni_turn": frozenset({"move_turn"}),
}
"""Pool type -> the categories it trains (``omni`` = uniform bearing, hence every bearing)."""

#: The hand a pool type trains inside a paired category (``side`` only: a turn's sign is drawn
#: uniformly, so ``turn`` and ``move_turn`` always train both hands).
_SIDE_HAND_OF_TYPE = {"side_l": ("l",), "side_r": ("r",), "omni": ("l", "r")}
#: Turning types -- an ``ang_scale < 1`` on one of these reduces its categories' demand.
_TURNING_TYPES = frozenset({"turn", "fwd_turn", "side_l_turn", "side_r_turn", "back_turn", "omni_turn"})
#: Moving types -- a ``speed_scale < 1`` on one of these reduces its categories' demand.
_MOVING_TYPES = frozenset({"fwd", "side_l", "side_r", "back", "omni", "fwd_turn", "side_l_turn",
                           "side_r_turn", "back_turn", "omni_turn"})
#: A body band lies inside ``dir_band`` within this tolerance (``1.5708`` vs ``pi/2``).
BAND_TOL = 1e-4
#: Floating slack on the floor-share comparison (a composed pool's shares sum to 1 +- 1e-12).
_SHARE_EPS = 1e-9

# ------------------------------------------------------------------ command-cfg class defaults
#: ``CrabHexOmniCommandCfg`` class-body defaults for the keys a stage env may omit (the 1c and
#: 2a/2b presets carry no ``KRABBY_OMNI_DIR_BAND``, several omit ``KRABBY_OMNI_FACING``). Held here
#: as constants -- importing the cfg would import Isaac -- and pinned against the class body's
#: SOURCE by a unit test.
CMD_CFG_DEFAULTS = {
    "slot_probs": (0.15, 0.55, 0.10, 0.20),
    "dir_band": None,
    "facing_delta": (0.2, 0.6),
    "cmd_pool": None,
}
#: The omni observation term's default encoding (``params["facing_obs_encoding"]`` in the env cfg;
#: pinned the same way) and its facing clip when the env sets none.
FACING_OBS_ENCODING_DEFAULT = "raw"
FACING_OBS_CLIP_VAR = "KRABBY_OMNI_FACING_OBS_CLIP"


class EvalMismatch(ValueError):
    """A stage the suite cannot evaluate as specified (the driver's ``halt:eval_mismatch``)."""


# ------------------------------------------------------------------ trained categories
NOTE_REDUCED = "watch (reduced demand)"
NOTE_ONE_HAND = "watch (one hand)"
NOTE_BELOW_FLOOR = "watch (below floor share)"
NOTE_UNTRAINED = "untrained"


class TrainedCategories(NamedTuple):
    """What a stage's command distribution covers (D4).

    ``decisive``
        Categories trained at share >= ``POOL_FLOOR_SHARE`` at the suite's full demand.
    ``present``
        Categories with share > 0 (``decisive`` is a subset).
    ``watch``
        ``{category: note}`` for present - decisive; the note is :data:`NOTE_REDUCED`,
        :data:`NOTE_ONE_HAND` or :data:`NOTE_BELOW_FLOOR`. Iterating it yields the names.
    ``hands``
        ``{category: ("l", "r") | ("l",) | ("r",)}`` for every present paired category.
    ``demand``
        ``{category: fraction}`` for every present category -- 1.0 at full demand, else the largest
        scale among the types that train it.
    ``source``
        ``"pool"`` (``KRABBY_OMNI_CMD_POOL``) or ``"slots"`` (the legacy four-slot sampler).
    ``facing_band_hi``
        The FACING band's upper edge, radians (an annotation, never a demotion).
    ``encoding``
        The facing-observation encoding, ``"raw"`` or ``"sincos"`` (an annotation).
    ``notes``
        Free-text lines for the RESOLUTION / DECISION lines (the ``omni`` support note, the scale
        edits, the turn-magnitude / encoding annotation).
    """

    decisive: frozenset
    present: frozenset
    watch: dict
    hands: dict
    demand: dict
    source: str
    facing_band_hi: float
    encoding: str
    notes: tuple

    def to_dict(self) -> dict:
        """JSON-able form (the header's ``trained_categories_by_stage[stage]``, the screen's copy)."""
        return {
            "decisive": _ordered(self.decisive),
            "present": _ordered(self.present),
            "watch": {c: self.watch[c] for c in _ordered(self.watch)},
            "hands": {c: list(self.hands[c]) for c in _ordered(self.hands)},
            "demand": {c: self.demand[c] for c in _ordered(self.demand)},
            "source": self.source,
            "facing_band_hi": self.facing_band_hi,
            "encoding": self.encoding,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> TrainedCategories:
        """Inverse of :meth:`to_dict` (a recorded header or screen copy back to the tuple)."""
        return cls(
            decisive=frozenset(d.get("decisive") or ()),
            present=frozenset(d.get("present") or ()),
            watch=dict(d.get("watch") or {}),
            hands={c: tuple(h) for c, h in (d.get("hands") or {}).items()},
            demand=dict(d.get("demand") or {}),
            source=str(d.get("source") or ""),
            facing_band_hi=float(d.get("facing_band_hi") or 0.0),
            encoding=str(d.get("encoding") or FACING_OBS_ENCODING_DEFAULT),
            notes=tuple(d.get("notes") or ()),
        )


def _ordered(cats: Iterable[str]) -> list[str]:
    """Category names in :data:`CATEGORIES` order (unknown names refused by name)."""
    s = set(cats)
    unknown = sorted(s - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categories {unknown}; known: {CATEGORIES}")
    return [c for c in CATEGORIES if c in s]


def category_of_hold(label: str) -> str | None:
    """The category a suite hold label belongs to; ``None`` for the settle hold, KeyError otherwise."""
    if label == SETTLE_LABEL:
        return None
    for cat, holds in CATEGORY_HOLDS.items():
        if label in holds:
            return cat
    raise KeyError(f"hold {label!r} is not a suite hold; known: {SUITE_HOLDS + (SETTLE_LABEL,)}")


def decisive_set(tc) -> frozenset:
    """The decisive categories of a :class:`TrainedCategories` or of its :meth:`~TrainedCategories.to_dict` form."""
    if isinstance(tc, TrainedCategories):
        return frozenset(tc.decisive)
    if isinstance(tc, Mapping):
        return frozenset(tc.get("decisive") or ())
    raise TypeError(f"decisive_set takes a TrainedCategories or its dict form, got {type(tc).__name__}")


def previous_categories(earlier) -> frozenset:
    """D6's ``previous_categories = U decisive(stages[:i])``.

    ``earlier`` is the earlier stages' composed envs (Mappings of ``KRABBY_*`` strings), their
    :class:`TrainedCategories`, or their dict forms -- one per stage, stages with ``eval: []``
    included.
    """
    out: set[str] = set()
    for item in earlier:
        if isinstance(item, TrainedCategories) or (isinstance(item, Mapping) and "decisive" in item):
            out |= decisive_set(item)
        else:
            out |= trained_categories(item).decisive
    return frozenset(out)


def _set(env: Mapping, var: str) -> str | None:
    """The env value when it is SET and non-empty, else None (the knob block's "unset")."""
    raw = env.get(var)
    if raw is None or not str(raw).strip():
        return None
    return str(raw)


def _parse_pool(env: Mapping):
    """The pool the env trains, or None (absent / ``none`` / ``off`` / empty = the slot sampler)."""
    raw = env.get(knobs().CMD_POOL_VAR, CMD_CFG_DEFAULTS["cmd_pool"])
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in ("", "none", "off", "null"):
        return None
    return command_math().parse_cmd_pool(text)


def _slot_types(env: Mapping) -> tuple[list[tuple[str, float, float, float]], float]:
    """Slots mode as ``(type, weight, ang_scale, speed_scale)`` rows (every scale 1.0) + the total.

    Slot 1 is ``omni`` when ``dir_band`` is None, else every body type whose zero-width band lies
    inside the band (each carries the slot's full weight: support, not a split -- hence the total is
    the SLOT total, never the row sum); slot 3 likewise with the ``*_turn`` twins. A band no type
    lies inside supports nothing.
    """
    kn = knobs()
    raw_slots = _set(env, kn.SLOTS_VAR)
    slots = kn.parse_slots(raw_slots) if raw_slots is not None else CMD_CFG_DEFAULTS["slot_probs"]
    # dir_band is applied whenever PRESENT (an empty value is a meaningful "no band").
    raw_band = env.get(kn.DIR_BAND_VAR)
    band = kn.parse_dir_band(raw_band) if raw_band is not None else CMD_CFG_DEFAULTS["dir_band"]
    types = command_math().POOL_TYPES
    if band is None:
        move, move_turn = ["omni"], ["omni_turn"]
    else:
        lo, hi = float(band[0]), float(band[1])
        move = [n for n in ("fwd", "side_l", "side_r", "back")
                if lo - BAND_TOL <= types[n].band[0] and types[n].band[1] <= hi + BAND_TOL]
        move_turn = [f"{n}_turn" for n in move]
    rows = [("stand", float(slots[0]), 1.0, 1.0), ("turn", float(slots[2]), 1.0, 1.0)]
    rows += [(n, float(slots[1]), 1.0, 1.0) for n in move]
    rows += [(n, float(slots[3]), 1.0, 1.0) for n in move_turn]
    return rows, float(sum(float(p) for p in slots))


def _category_scale(name: str, ang: float, spd: float) -> float:
    """The demand fraction a type trains its categories at (1.0 = the suite's full demand)."""
    scale = 1.0
    if name in _TURNING_TYPES:
        scale = min(scale, float(ang))
    if name in _MOVING_TYPES:
        scale = min(scale, float(spd))
    return scale


def trained_categories(env: Mapping) -> TrainedCategories:
    """D4: the categories a stage's COMPOSED training env trains -- see the module docstring."""
    kn = knobs()
    floor = float(command_math().POOL_FLOOR_SHARE) - _SHARE_EPS
    pool = _parse_pool(env)
    if pool is not None:
        source = "pool"
        rows = list(zip(pool.names, (float(w) for w in pool.weights), pool.ang_scale, pool.speed_scale))
        total = sum(w for _n, w, _a, _s in rows)
    else:
        source = "slots"
        rows, total = _slot_types(env)
    if not total > 0.0:
        raise EvalMismatch("the stage's command distribution has no weight at all")

    # Per category (and per side hand): the total share, and the share at FULL demand.
    share = {c: 0.0 for c in CATEGORIES}
    full = {c: 0.0 for c in CATEGORIES}
    scales: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    side_share = {h: 0.0 for h in HANDS}
    side_full = {h: 0.0 for h in HANDS}
    notes: list[str] = []
    for name, weight, ang, spd in rows:
        s = weight / total
        if s <= 0.0:
            continue
        scale = _category_scale(name, ang, spd)
        if scale < 1.0:
            notes.append(f"{name}: demand x{scale:g} ({', '.join(_ordered(TYPE_TO_CATEGORIES[name]))} "
                         "trained below the suite's demand)")
        for c in TYPE_TO_CATEGORIES[name]:
            share[c] += s
            scales[c].append(scale)
            if scale >= 1.0:
                full[c] += s
        for h in _SIDE_HAND_OF_TYPE.get(name, ()):
            side_share[h] += s
            if scale >= 1.0:
                side_full[h] += s
        if name == "omni":
            notes.append(f"omni {s:.3f}: fwd, side, back, diag decisive by uniform-bearing support")

    if share["stand"] <= 0.0:
        raise EvalMismatch("the stage trains no stand share: the suite's settle hold is part of every "
                           "category's episode, so the stage cannot be scored on it")

    decisive, present, watch, demand = set(), set(), {}, {}
    for c in CATEGORIES:
        if share[c] <= 0.0:
            continue
        present.add(c)
        demand[c] = 1.0 if (full[c] >= floor) else max(scales[c])
        if c == "side":
            full_ok = side_full["l"] >= floor and side_full["r"] >= floor
            total_ok = side_share["l"] >= floor and side_share["r"] >= floor
            one_hand = (side_share["l"] >= floor) != (side_share["r"] >= floor)
        else:
            full_ok, total_ok, one_hand = full[c] >= floor, share[c] >= floor, False
        if full_ok:
            decisive.add(c)
        elif total_ok:
            watch[c] = NOTE_REDUCED
        elif one_hand:
            watch[c] = NOTE_ONE_HAND
        else:
            watch[c] = NOTE_BELOW_FLOOR

    hands = {}
    for c in PAIRED_CATEGORIES:
        if c not in present:
            continue
        hands[c] = tuple(h for h in HANDS if side_share[h] > 0.0) if c == "side" else HANDS

    raw_facing = _set(env, kn.FACING_VAR)
    facing = kn.parse_facing(raw_facing) if raw_facing is not None else CMD_CFG_DEFAULTS["facing_delta"]
    raw_enc = _set(env, kn.FACING_OBS_ENCODING_VAR)
    encoding = kn.parse_facing_obs_encoding(raw_enc) if raw_enc is not None else FACING_OBS_ENCODING_DEFAULT
    clip = _set(env, FACING_OBS_CLIP_VAR)
    tc = TrainedCategories(
        decisive=frozenset(decisive), present=frozenset(present), watch=watch, hands=hands,
        demand=demand, source=source, facing_band_hi=float(facing[1]), encoding=encoding,
        notes=tuple(dict.fromkeys(notes)),
    )
    return tc._replace(notes=tc.notes + (turn_annotation(tc, facing_obs_clip=clip),))


def turn_annotation(tc: TrainedCategories, *, facing_obs_clip: str | float | None = None) -> str:
    """``turn: demand 1.571 rad > FACING band 0.600; encoding raw clips at 0.6`` -- never a demotion.

    The encoding changes what the policy observes, not which command category it trained, so the
    line is printed on the RESOLUTION and DECISION lines and nothing reads it as a verdict.
    """
    rel = ">" if SUITE_TURN_DEMAND_RAD > tc.facing_band_hi else "<="
    head = f"turn: demand {SUITE_TURN_DEMAND_RAD:.3f} rad {rel} FACING band {tc.facing_band_hi:.3f}"
    if tc.encoding == "sincos":
        return f"{head}; encoding sincos sees the magnitude"
    if facing_obs_clip is None:
        return f"{head}; encoding raw (no obs clip set)"
    return f"{head}; encoding raw clips at {float(facing_obs_clip):g}"


# ------------------------------------------------------------------ decision rule v2
#: The ``per_category`` templates of ``decision_rule`` (``rule_version: 2``), in the form
#: ``candidates.yaml`` carries them (lists, not tuples, so the yaml and this constant compare equal).
#: ``blocks.<name>.members`` / ``guards.members`` / ``watch.members`` map a category to ``cat`` keys;
#: ``guards_abs.members`` REPLACES a category's non-survival guard keys under ``absolute_units``;
#: ``watch.legacy`` maps a legacy summary id to its print note.
RULE_V2_PER_CATEGORY = {
    "blocks": {
        "F": {"level": 0.95, "units": "ratio", "members": {"turn": ["facing_closed", "yaw_progress"]}},
        "F_mt": {"level": 0.95, "units": "ratio",
                 "members": {"move_turn": ["facing_closed", "yaw_progress"]}},
        "T": {"level": 0.95, "units": "ratio",
              "members": {"fwd": ["ratio"], "side": ["ratio"], "back": ["ratio"], "diag": ["ratio"]}},
        "F_abs": {"level": 0.95, "units": "absolute", "members": {"turn": ["facing_closed", "net_yaw"]}},
        "F_mt_abs": {"level": 0.95, "units": "absolute",
                     "members": {"move_turn": ["facing_closed", "net_yaw"]}},
        "T_abs": {"level": 0.95, "units": "absolute",
                  "members": {"fwd": ["achieved_speed"], "side": ["achieved_speed"],
                              "back": ["achieved_speed"], "diag": ["achieved_speed"]}},
    },
    "guards": {
        "level": 0.99,
        "members": {"stand": ["survival"], "fwd": ["survival"], "side": ["survival"],
                    "back": ["survival"], "diag": ["survival"], "turn": ["survival"],
                    "move_turn": ["survival", "ratio"]},
        "composite": ["cat_tripod_min"],
    },
    "guards_abs": {"members": {"move_turn": ["achieved_speed"]}},
    "watch": {
        "members": {
            "stand": ["drift_speed"],
            "fwd": ["direction_error", "cross_track_ratio", "heading_final"],
            "side": ["direction_error", "cross_track_ratio", "heading_final"],
            "back": ["direction_error", "cross_track_ratio", "heading_final"],
            "diag": ["direction_error", "cross_track_ratio", "heading_final"],
            "turn": ["settle_s", "heading_final"],
            "move_turn": ["tripod", "direction_error", "cross_track_ratio", "settle_s"],
        },
        "hands": True,
        "legacy": {"completion": "legacy", "tripod_walking_dir": "legacy",
                   "track_ratio_tail": "legacy", "track_ratio_move_tail": "legacy",
                   "turn_survival_l90": "N/A (suite)", "turn_survival_r90": "N/A (suite)"},
    },
}

BLOCKS_RATIO = ("F", "F_mt", "T")
BLOCKS_ABS = ("F_abs", "F_mt_abs", "T_abs")
#: Top-level keys of the raw rule that the expansion REPLACES (everything else is carried through).
_EXPANDED_KEYS = ("blocks", "guards", "watch", "per_category")

_CAT_ID = re.compile(r"^cat\.(?P<c>[a-z_]+)\.(?:by_hand\.(?P<h>[lr])\.)?(?P<key>[a-z_]+)$")


def cat_id(category: str, key: str, hand: str | None = None) -> str:
    """``cat.<c>.<key>`` / ``cat.<c>.by_hand.<h>.<key>`` -- a dotted path into ``summarize()``."""
    return f"cat.{category}.by_hand.{hand}.{key}" if hand else f"cat.{category}.{key}"


def parse_cat_id(mid: str) -> tuple[str, str | None, str] | None:
    """``(category, hand | None, key)`` for a ``cat.*`` id, else None."""
    m = _CAT_ID.match(str(mid))
    if not m or m.group("c") not in CATEGORIES or m.group("key") not in CAT_KEYS:
        return None
    return m.group("c"), m.group("h"), m.group("key")


def default_rule_v2(**extra) -> dict:
    """A ``rule_version: 2`` decision rule carrying :data:`RULE_V2_PER_CATEGORY` (tests, dry runs)."""
    return {"rule_version": 2, "per_category": copy.deepcopy(RULE_V2_PER_CATEGORY), **extra}


def _check_keys(where: str, members: Mapping) -> None:
    for c, keys in (members or {}).items():
        if c not in CATEGORIES:
            raise ValueError(f"decision_rule.per_category.{where} names category {c!r}; known: {CATEGORIES}")
        bad = [k for k in keys or () if k not in CAT_KEYS]
        if bad:
            raise ValueError(f"decision_rule.per_category.{where}.{c} names keys {bad}; known: {CAT_KEYS}")


def _per_category(rule: Mapping) -> dict:
    """The validated ``per_category`` templates of a v2 rule (a v1 rule is refused by name)."""
    if not isinstance(rule, Mapping) or int(rule.get("rule_version") or 1) != 2:
        raise ValueError("expand_rule takes a rule_version: 2 decision rule; a v1 rule is scored as it "
                         "stands (its blocks/guards are already flat)")
    pc = rule.get("per_category")
    if not isinstance(pc, Mapping) or not pc.get("blocks"):
        raise ValueError("decision_rule (rule_version 2) carries no per_category.blocks")
    for name, body in pc["blocks"].items():
        if (body or {}).get("units") not in ("ratio", "absolute"):
            raise ValueError(f"decision_rule.per_category.blocks.{name}.units must be ratio|absolute")
        _check_keys(f"blocks.{name}.members", body.get("members") or {})
    _check_keys("guards.members", (pc.get("guards") or {}).get("members") or {})
    _check_keys("guards_abs.members", (pc.get("guards_abs") or {}).get("members") or {})
    _check_keys("watch.members", (pc.get("watch") or {}).get("members") or {})
    bad = [c for c in (pc.get("guards") or {}).get("composite") or () if c not in COMPOSITES]
    if bad:
        raise ValueError(f"decision_rule.per_category.guards.composite names {bad}; known: {COMPOSITES}")
    return pc


def _cats(x) -> frozenset:
    if x is None:
        return frozenset()
    s = frozenset(x.keys() if isinstance(x, Mapping) else x)
    _ordered(s)  # refuses unknown names
    return s


def _guard_keys(pc: Mapping, c: str, absolute_units: bool) -> tuple[list[str], list[str]]:
    """``(guard keys, keys moved to WATCH by the unit swap)`` for one category."""
    keys = list(((pc.get("guards") or {}).get("members") or {}).get(c) or [])
    swap = ((pc.get("guards_abs") or {}).get("members") or {}).get(c)
    if swap is None:
        return keys, []
    survival = [k for k in keys if k == "survival"]
    ratio_side = [k for k in keys if k != "survival"]
    if absolute_units:
        return survival + [k for k in swap if k != "survival"], ratio_side
    return keys, [k for k in swap if k not in keys]


def expand_rule(rule: Mapping, decisive, present, *, absolute_units: bool,
                watch_notes: Mapping | None = None) -> dict:
    """D5: a v2 rule restricted to ``decisive`` (+ ``present``'s survival guards), in today's flat shape.

    ``decisive`` / ``present`` are category names (any iterable, or a mapping whose keys are names);
    ``decisive`` is folded into ``present``. ``watch_notes`` gives the note of a present, non-decisive
    category (``TrainedCategories.watch``); without one it reads ``watch``. See the module docstring
    for the contract; the returned dict carries every other top-level key of ``rule`` unchanged.
    """
    pc = _per_category(rule)
    dec = _cats(decisive)
    pres = _cats(present) | dec
    wn = dict(watch_notes or {})
    units = "absolute" if absolute_units else "ratio"

    def note(c: str) -> str | None:
        if c in dec:
            return None
        if c in pres:
            return wn.get(c) or "watch"
        return NOTE_UNTRAINED

    out = {k: copy.deepcopy(v) for k, v in rule.items() if k not in _EXPANDED_KEYS}
    watch: dict[str, str | None] = {}
    blocks: dict[str, dict] = {}
    for name, body in pc["blocks"].items():
        if body.get("units") != units:
            continue
        members = body.get("members") or {}
        metrics, cats = [], []
        for c in CATEGORIES:
            for key in members.get(c) or ():
                if c in dec:
                    metrics.append(cat_id(c, key))
                    cats.append(c)
                else:
                    watch.setdefault(cat_id(c, key), note(c))
        blocks[name] = {"level": float(body.get("level", 0.95)), "metrics": metrics,
                        "categories": list(dict.fromkeys(cats))}

    gbody = pc.get("guards") or {}
    guard_ids: list[str] = []
    for c in CATEGORIES:
        keys, swapped = _guard_keys(pc, c, absolute_units)
        for key in keys:
            keep = (c in pres) if key == "survival" else (c in dec)
            if keep:
                guard_ids.append(cat_id(c, key))
            else:
                watch.setdefault(cat_id(c, key), note(c))
        for key in swapped:
            if c in pres:
                watch.setdefault(cat_id(c, key), note(c))
    composite: dict[str, list[str]] = {}
    if "cat_tripod_min" in (gbody.get("composite") or ()):
        members = [c for c in TRANSLATION_CATEGORIES if c in dec]
        if members:
            guard_ids.append("cat_tripod_min")
            composite["cat_tripod_min"] = members
        for c in TRANSLATION_CATEGORIES:
            if c not in dec:
                watch.setdefault(cat_id(c, "tripod"), note(c))

    wbody = pc.get("watch") or {}
    for c in CATEGORIES:
        if c not in pres:
            continue
        for key in (wbody.get("members") or {}).get(c) or ():
            watch.setdefault(cat_id(c, key), note(c))
    if wbody.get("hands"):
        for c in PAIRED_CATEGORIES:
            keys: list[str] = []
            for body in pc["blocks"].values():
                if body.get("units") == units:
                    keys += list((body.get("members") or {}).get(c) or ())
            keys += _guard_keys(pc, c, absolute_units)[0]
            for h in HANDS:
                for key in dict.fromkeys(keys):
                    watch.setdefault(cat_id(c, key, h), note(c))
    for mid, n in (wbody.get("legacy") or {}).items():
        watch.setdefault(str(mid), n)

    scored = set(guard_ids).union(*(set(b["metrics"]) for b in blocks.values()))
    out.update({
        "rule_version": 2,
        "expanded": True,
        "units": units,
        "decisive": _ordered(dec),
        "present": _ordered(pres),
        "blocks": blocks,
        "na_blocks": [name for name, b in blocks.items() if not b["metrics"]],
        "guards": {"level": float(gbody.get("level", 0.99)), "metrics": guard_ids},
        "composite": composite,
        "watch": {mid: n for mid, n in watch.items() if mid not in scored},
    })
    return out


def expand_rule_union(rule: Mapping, decisive, present, *, watch_notes: Mapping | None = None) -> dict:
    """The UNION of the ratio and absolute-units expansions (resolution's D00s / D00bs).

    Blocks ``F, F_mt, T, F_abs, F_mt_abs, T_abs``; guards = the union of both systems' guard lists
    in order (at B0: seven ``survival``, ``cat.move_turn.ratio``, ``cat_tripod_min``,
    ``cat.move_turn.achieved_speed``); ``watch`` = both systems' watch ids minus anything scored in
    either. ``units: "union"``.
    """
    ratio = expand_rule(rule, decisive, present, absolute_units=False, watch_notes=watch_notes)
    absu = expand_rule(rule, decisive, present, absolute_units=True, watch_notes=watch_notes)
    out = dict(ratio)
    out["units"] = "union"
    out["blocks"] = {**ratio["blocks"], **absu["blocks"]}
    out["na_blocks"] = ratio["na_blocks"] + absu["na_blocks"]
    out["guards"] = {"level": ratio["guards"]["level"],
                     "metrics": list(dict.fromkeys(ratio["guards"]["metrics"] + absu["guards"]["metrics"]))}
    out["composite"] = {**ratio["composite"], **absu["composite"]}
    scored = set(out["guards"]["metrics"]).union(*(set(b["metrics"]) for b in out["blocks"].values()))
    watch = dict(ratio["watch"])
    for mid, n in absu["watch"].items():
        watch.setdefault(mid, n)
    out["watch"] = {mid: n for mid, n in watch.items() if mid not in scored}
    return out


def expand_for(rule: Mapping, tc, *, absolute_units: bool) -> dict:
    """:func:`expand_rule` at a :class:`TrainedCategories` (or its dict form), watch notes included."""
    if not isinstance(tc, TrainedCategories):
        tc = TrainedCategories.from_dict(tc)
    return expand_rule(rule, tc.decisive, tc.present, absolute_units=absolute_units,
                       watch_notes=tc.watch)


# ------------------------------------------------------------------ composite guards
def _dget(d, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _finite(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def composite_guards(screen: Mapping, decisive) -> dict:
    """``{"cat_tripod_min": min over the decisive translation categories' cat.<c>.tripod}``.

    ``move_turn`` never enters (its tripod is WATCH). ``None`` when any member's tripod is missing or
    non-finite -- a MISSING value, never a silently narrower min; ``{}`` when no translation category
    is decisive (the guard is then absent from the expanded rule too).
    """
    if isinstance(decisive, TrainedCategories) or (isinstance(decisive, Mapping) and "decisive" in decisive):
        decisive = decisive_set(decisive)
    members = [c for c in TRANSLATION_CATEGORIES if c in _cats(decisive)]
    if not members:
        return {}
    values = [_finite(_dget(screen, cat_id(c, "tripod"))) for c in members]
    return {"cat_tripod_min": None if any(v is None for v in values) else min(values)}


def with_composites(screen: Mapping | None, decisive) -> dict | None:
    """A SHALLOW copy of ``screen`` with :func:`composite_guards` injected (None stays None)."""
    if screen is None:
        return None
    return {**screen, **composite_guards(screen, decisive)}
