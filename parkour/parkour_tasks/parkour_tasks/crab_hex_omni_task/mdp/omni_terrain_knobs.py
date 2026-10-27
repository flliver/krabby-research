# SPDX-License-Identifier: BSD-3-Clause
"""Ring-terrain knobs and the "is the ring actually installed?" guard (plan D8, spec B7).

The omni task shares the forward task's whole knob block (``apply_flat_walk_knobs``), which
knows nothing about rings: its ``KRABBY_FLAT_TERRAIN_GEOM`` branch dispatches to the
corridor geometry helpers, all of which are keyed on the corridor sub-terrain names
(``parkour_gap``, ``parkour_hurdle``, ...) and are therefore inert on a ring generator.
``apply_omni_ring_knobs`` runs AFTER it and does three things:

1. **Asserts the ring generator is the one installed**, by the CLASS of the sub-terrain
   cfgs rather than by their names -- a forward generator that leaked into an omni env
   through a stale cfg, an env-var preset or a copy-pasted scene would otherwise train
   happily on corridors while every omni reward, the ring spawn and the radial exposure
   quietly measured a course that is not there.
2. **Refuses a leaked geometry preset**: the omni presets set
   ``KRABBY_FLAT_TERRAIN_GEOM=ring`` in every window, so any other value (or none) means a
   forward preset is in the environment, and the run is not the one that was asked for.
3. **Applies the two ring knobs** -- ``KRABBY_RING_CLEAR_R`` (metres) and
   ``KRABBY_RING_BLOCKS`` (``"sparse:dense"`` block counts) -- into the sub-terrain cfgs,
   and keeps the parkour term's copy of the ring geometry in step with the terrain's (they
   are two independent numbers describing one disc; a mismatch silently redefines
   ``reach_edge`` and ``field_frac``).

Everything except the last step of :func:`apply_omni_ring_knobs` is pure: the parsers and
the class-based classification take the cfg classes as arguments, so
``tests/unit/test_crab_hex_omni_terrain_knobs.py`` exercises them without Isaac Sim.

THE FIELD GEOMETRY (omni phase-2 plan, step 2, 2026-10-06)
----------------------------------------------------------
``KRABBY_FLAT_TERRAIN_GEOM=field`` selects a second omni course: a small clear spawn pad, then
bars, slabs and pits over the whole tile square (``OmniFieldTerrainCfg``). Three pieces here:

* :func:`install_omni_terrain_generator` swaps the scene's generator for a DEEP COPY of
  ``OMNI_FIELD_TERRAINS_CFG`` -- called FIRST in both omni ``__post_init__``s (before the forward
  knob block), so the forward block's flat fraction / difficulty band / curriculum switch and an
  eval row's ``env:`` block (the harness writes the env before ``parse_env_cfg``) reach the field
  exactly as they reach the ring. ``ring`` (and anything else) is a no-op: the ring path is the
  scene default, untouched.
* :func:`require_omni_geom` -- the guard :func:`apply_omni_ring_knobs` now dispatches on -- accepts
  ``ring`` or ``field`` and REFUSES MIXED KNOBS: a ``KRABBY_RING_*`` under ``field`` or a
  ``KRABBY_FIELD_*`` under ``ring`` is a recipe that would silently apply nothing (each knob family
  only knows its own sub-terrain class). ``require_ring_geom`` is unchanged.
* :func:`apply_omni_field_knobs` -- the field twin of the ring branch: class assertion (field +
  flat only), the ``KRABBY_FIELD_*`` knobs (:data:`FIELD_VARS`), and the same parkour-term radius
  sync (``clear_radius_m`` = the pad, so ``reach_edge`` means "left the pad" and ``field_frac`` /
  the sectors are measured beyond it).
"""
from __future__ import annotations

import math
import os

__all__ = [
    "GEOM_VAR",
    "RING_GEOM",
    "CLEAR_R_VAR",
    "BLOCKS_VAR",
    "parse_clear_radius",
    "parse_ring_blocks",
    "require_ring_geom",
    "classify_sub_terrains",
    "assert_ring_sub_terrains",
    "assert_ring_generator",
    "apply_ring_geometry",
    "apply_omni_ring_knobs",
    # field geometry (omni phase-2 plan step 2)
    "FIELD_GEOM",
    "OMNI_GEOMS",
    "RING_VARS",
    "FIELD_PAD_R_VAR",
    "FIELD_BLOCKS_VAR",
    "FIELD_KINDS_VAR",
    "FIELD_EDGES_VAR",
    "FIELD_MIN_GAP_VAR",
    "FIELD_SIZE_SCALE_VAR",
    "FIELD_VARS",
    "FIELD_PAD_R_MIN",
    "FIELD_BLOCKS_MAX",
    "parse_field_pad_r",
    "parse_field_blocks",
    "parse_field_kinds",
    "parse_field_edges",
    "parse_field_min_gap",
    "parse_field_size_scale",
    "FIELD_KNOB_PARSERS",
    "read_omni_geom",
    "require_omni_geom",
    "is_field_generator",
    "assert_field_sub_terrains",
    "apply_field_geometry",
    "apply_omni_field_knobs",
    "install_omni_terrain_generator",
]

GEOM_VAR = "KRABBY_FLAT_TERRAIN_GEOM"
RING_GEOM = "ring"
CLEAR_R_VAR = "KRABBY_RING_CLEAR_R"
BLOCKS_VAR = "KRABBY_RING_BLOCKS"
RING_VARS = (CLEAR_R_VAR, BLOCKS_VAR)

FIELD_GEOM = "field"
OMNI_GEOMS = (RING_GEOM, FIELD_GEOM)
FIELD_PAD_R_VAR = "KRABBY_FIELD_PAD_R"
FIELD_BLOCKS_VAR = "KRABBY_FIELD_BLOCKS"
FIELD_KINDS_VAR = "KRABBY_FIELD_KINDS"
FIELD_EDGES_VAR = "KRABBY_FIELD_EDGES"
FIELD_MIN_GAP_VAR = "KRABBY_FIELD_MIN_GAP"
FIELD_SIZE_SCALE_VAR = "KRABBY_FIELD_SIZE_SCALE"
FIELD_VARS = (
    FIELD_PAD_R_VAR, FIELD_BLOCKS_VAR, FIELD_KINDS_VAR, FIELD_EDGES_VAR,
    FIELD_MIN_GAP_VAR, FIELD_SIZE_SCALE_VAR,
)

#: Smallest spawn pad accepted (m). Toe radius from foot FK is <= 1.46 m (hardware max 1.55 m), so
#: below 1.6 m a foot can land inside a block at spawn: a depenetration kick and a fall the policy
#: did not cause, and every eval's opening settle corrupted with it.
FIELD_PAD_R_MIN = 1.6
#: The pad must leave at least this much of the half-width (m) for blocks.
FIELD_PAD_R_HEADROOM = 1.0
#: Field half-width the pad bound is checked against when no cfg is at hand (``0.5*9.92 - 0.16``).
FIELD_HALF_WIDTH = 4.80
#: Upper bound on blocks per tile (the dense default is 28; 60 is already ~35 % cover).
FIELD_BLOCKS_MAX = 60


# ------------------------------------------------------------------------------ parsers


def parse_clear_radius(text: str) -> float:
    """``KRABBY_RING_CLEAR_R`` -> metres. Finite and strictly positive."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{CLEAR_R_VAR} must be a number in metres, got {text!r}") from exc
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{CLEAR_R_VAR} must be finite and positive, got {text!r}")
    return value


def parse_ring_blocks(text: str) -> tuple[int, int]:
    """``KRABBY_RING_BLOCKS`` -> ``(sparse, dense)`` block counts.

    Format ``"sparse:dense"`` (the ``lo:hi`` shape every other two-valued Krabby knob
    uses). Both must be non-negative integers; ``"8:16"`` is the default pair. Floats are
    rejected rather than truncated -- ``"8.5:16"`` is a typo, and silently rounding it
    would leave a campaign reporting a block count nobody asked for.
    """
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f'{BLOCKS_VAR} must be "sparse:dense", got {text!r}')
    out = []
    for part in parts:
        token = part.strip()
        try:
            value = int(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{BLOCKS_VAR} entries must be integers, got {text!r}") from exc
        if value < 0:
            raise ValueError(f"{BLOCKS_VAR} entries must be non-negative, got {text!r}")
        out.append(value)
    return out[0], out[1]


def require_ring_geom(environ=None) -> str:
    """Require ``KRABBY_FLAT_TERRAIN_GEOM == "ring"``; raise naming the leaked value.

    Compared after ``strip().lower()``, exactly as ``apply_flat_walk_knobs`` reads it, so
    the two never disagree about what the variable says.
    """
    env = os.environ if environ is None else environ
    raw = env.get(GEOM_VAR)
    value = None if raw is None else str(raw).strip().lower()
    if value != RING_GEOM:
        shown = "<unset>" if raw is None else repr(raw)
        raise RuntimeError(
            f"the omni task requires {GEOM_VAR}={RING_GEOM!r}, found {shown}. "
            "A forward/sideways geometry preset is in the environment: the omni presets set "
            f"{GEOM_VAR}={RING_GEOM} in every window, and any other value means this process "
            "inherited a corridor preset (whose geometry helpers are silently inert on the "
            "ring generator)."
        )
    return value


# ------------------------------------------------------------------- generator identity


def classify_sub_terrains(sub_terrains, ring_cls, flat_cls) -> tuple[list[str], list[str], list[str]]:
    """Split a generator's ``sub_terrains`` into ``(ring_keys, flat_keys, foreign_keys)``.

    Classification is by CLASS, not by key name: the ``"parkour_flat"`` key is deliberately
    reused by the omni generator (nine shared sites key off that exact string), so names
    alone cannot tell a ring generator from a corridor one. Keys are returned in the
    mapping's own order, which is the order the block counts are applied in.
    """
    ring_keys, flat_keys, foreign = [], [], []
    for key, sub in sub_terrains.items():
        if isinstance(sub, ring_cls):
            ring_keys.append(key)
        elif isinstance(sub, flat_cls):
            flat_keys.append(key)
        else:
            foreign.append(key)
    return ring_keys, flat_keys, foreign


def assert_ring_sub_terrains(sub_terrains, ring_cls, flat_cls) -> tuple[list[str], list[str]]:
    """Raise unless every sub-terrain is an omni one and at least one is a ring.

    Accepts exactly: at least one ``OmniRingTerrainCfg`` sub-terrain, and NO sub-terrain of
    any other family (``OmniFlatTerrainCfg`` is the only other allowed class). A corridor
    generator therefore fails on its first sub-terrain instead of training a corridor under
    an omni command term.
    """
    if not sub_terrains:
        raise RuntimeError("terrain generator has no sub_terrains; expected the omni ring generator.")
    ring_keys, flat_keys, foreign = classify_sub_terrains(sub_terrains, ring_cls, flat_cls)
    if foreign:
        shown = ", ".join(f"{k}={type(sub_terrains[k]).__name__}" for k in foreign)
        raise RuntimeError(
            "the omni task requires the omni ring generator, but these sub-terrains are "
            f"neither {ring_cls.__name__} nor {flat_cls.__name__}: {shown}"
        )
    if not ring_keys:
        raise RuntimeError(
            f"the omni ring generator must carry at least one {ring_cls.__name__} "
            f"sub-terrain; found only flat ones ({', '.join(flat_keys) or 'none'})."
        )
    return ring_keys, flat_keys


def assert_ring_generator(terrain_generator, ring_cls, flat_cls) -> tuple[list[str], list[str]]:
    """:func:`assert_ring_sub_terrains` on a generator cfg. Returns ``(ring, flat)`` keys."""
    if terrain_generator is None:
        raise RuntimeError(
            "the omni task requires scene.terrain.terrain_generator to be "
            "OMNI_RING_TERRAINS_CFG; found None (a flat/plane terrain importer)."
        )
    return assert_ring_sub_terrains(
        getattr(terrain_generator, "sub_terrains", None), ring_cls, flat_cls
    )


# -------------------------------------------------------------------------- application


def apply_ring_geometry(
    sub_terrains,
    ring_cls,
    flat_cls,
    *,
    clear_radius: float | None = None,
    blocks: tuple[int, int] | None = None,
) -> None:
    """Write the two ring knobs into the sub-terrain cfgs (no-op where the knob is None).

    ``clear_radius`` goes to EVERY omni sub-terrain, ring and flat alike: the flat tiles
    place the same sector anchors at ``(clear + outer) / 2``, so leaving them behind would
    move the anchors between tile types and make ``goals_passed`` mean two different things
    within one population.

    ``blocks`` is ``(sparse, dense)`` and is applied to the ring sub-terrains in the
    generator's own order. Exactly two are required -- the omni generator ships
    ``omni_ring_sparse`` then ``omni_ring_dense`` -- because a positional pair has no
    meaning against any other count.
    """
    ring_keys, flat_keys = assert_ring_sub_terrains(sub_terrains, ring_cls, flat_cls)
    if clear_radius is not None:
        for key in ring_keys + flat_keys:
            sub = sub_terrains[key]
            outer = float(getattr(sub, "ring_outer_radius"))
            if not clear_radius < outer:
                raise ValueError(
                    f"{CLEAR_R_VAR}={clear_radius} must be below sub-terrain '{key}'"
                    f".ring_outer_radius={outer}"
                )
            sub.clear_radius = float(clear_radius)
    if blocks is not None:
        if len(ring_keys) != 2:
            raise RuntimeError(
                f'{BLOCKS_VAR} is a "sparse:dense" pair and needs exactly two ring '
                f"sub-terrains; found {len(ring_keys)} ({', '.join(ring_keys) or 'none'})"
            )
        for key, count in zip(ring_keys, blocks):
            sub_terrains[key].n_blocks = int(count)


def apply_omni_ring_knobs(cfg, environ=None, classes=None, field_classes=None) -> None:
    """Guard the ring geometry and apply ``KRABBY_RING_*`` (called from ``__post_init__``).

    Args:
        cfg: the omni env cfg (needs ``scene.terrain.terrain_generator``; the parkour term
            at ``parkours.base_parkour`` is synced when present).
        environ: mapping to read the knobs from (defaults to ``os.environ``).
        classes: ``(ring_cls, flat_cls)``; defaults to the real
            ``parkour_isaaclab.terrains.omni`` cfg classes, imported lazily so this module
            stays importable (and unit-testable) without Isaac.

    Order matters: the geometry preset is checked FIRST, so a process that inherited a
    forward preset fails with that message rather than with a confusing class assertion.

    Omni phase-2 plan step 2 (2026-10-06): this is now the DISPATCHER. :func:`require_omni_geom`
    accepts ``ring`` or ``field`` (and refuses mixed knob families); ``field`` goes to
    :func:`apply_omni_field_knobs` (``field_classes`` = ``(field_cls, flat_cls)``, defaulting to the
    real cfg classes), ``ring`` runs the body below exactly as before.
    """
    env = os.environ if environ is None else environ
    if require_omni_geom(env) == FIELD_GEOM:
        apply_omni_field_knobs(cfg, env, classes=field_classes)
        return

    if classes is None:
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import (  # noqa: PLC0415
            OmniFlatTerrainCfg,
            OmniRingTerrainCfg,
        )

        ring_cls, flat_cls = OmniRingTerrainCfg, OmniFlatTerrainCfg
    else:
        ring_cls, flat_cls = classes

    terrain = getattr(cfg.scene, "terrain", None)
    tg = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    assert_ring_generator(tg, ring_cls, flat_cls)

    raw_r = env.get(CLEAR_R_VAR)
    raw_blocks = env.get(BLOCKS_VAR)
    apply_ring_geometry(
        tg.sub_terrains,
        ring_cls,
        flat_cls,
        clear_radius=parse_clear_radius(raw_r) if raw_r else None,
        blocks=parse_ring_blocks(raw_blocks) if raw_blocks else None,
    )

    # The parkour term carries its OWN copy of the ring geometry (it is what defines
    # reach_edge / field_frac / the sector anchors it credits). Two numbers, one disc: keep
    # them equal, and say so when they were not.
    parkour = getattr(getattr(cfg, "parkours", None), "base_parkour", None)
    if parkour is not None and hasattr(parkour, "clear_radius_m"):
        ring_keys, flat_keys = classify_sub_terrains(tg.sub_terrains, ring_cls, flat_cls)[:2]
        sub = tg.sub_terrains[(ring_keys + flat_keys)[0]]
        for term_attr, sub_attr in (("clear_radius_m", "clear_radius"),
                                    ("ring_outer_radius_m", "ring_outer_radius")):
            want = float(getattr(sub, sub_attr))
            if float(getattr(parkour, term_attr)) != want:
                print(
                    f"[omni-knobs] parkours.base_parkour.{term_attr} "
                    f"({getattr(parkour, term_attr)}) did not match the terrain's "
                    f"{sub_attr} ({want}); syncing to the terrain.",
                    flush=True,
                )
                setattr(parkour, term_attr, want)


# =============================================================================== FIELD
# Omni phase-2 plan, step 2 (2026-10-06). Every knob below is default-off: unset (or blank) writes
# nothing, and with KRABBY_FLAT_TERRAIN_GEOM=ring none of this code runs at all.


def _is_blank(raw) -> bool:
    return raw is None or str(raw).strip() == ""


def _parse_float(var: str, text) -> float:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{var} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var} must be finite, got {text!r}")
    return value


def parse_field_pad_r(text, half_width: float = FIELD_HALF_WIDTH) -> float:
    """``KRABBY_FIELD_PAD_R`` -> the clear spawn pad radius, metres.

    Refused below :data:`FIELD_PAD_R_MIN` (1.6 m; a foot could start inside a block) and at or
    above ``half_width - 1.0`` (a pad that leaves under a metre of field is not a field).
    """
    value = _parse_float(FIELD_PAD_R_VAR, text)
    if value < FIELD_PAD_R_MIN:
        raise ValueError(
            f"{FIELD_PAD_R_VAR}={text!r} is below {FIELD_PAD_R_MIN} m: toe reach is up to 1.46 m "
            "(hardware 1.55 m), so a smaller pad lets a spawn put a foot inside a block"
        )
    limit = float(half_width) - FIELD_PAD_R_HEADROOM
    if not value < limit:
        raise ValueError(
            f"{FIELD_PAD_R_VAR}={text!r} must stay below the field half-width {half_width} m minus "
            f"{FIELD_PAD_R_HEADROOM} m (= {limit:.2f} m)"
        )
    return value


def parse_field_blocks(text) -> tuple[int, int]:
    """``KRABBY_FIELD_BLOCKS`` -> ``(sparse, dense)`` blocks per tile, integers in
    ``[0, FIELD_BLOCKS_MAX]`` (floats refused, as for ``KRABBY_RING_BLOCKS``)."""
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f'{FIELD_BLOCKS_VAR} must be "sparse:dense", got {text!r}')
    out = []
    for part in parts:
        try:
            value = int(part.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FIELD_BLOCKS_VAR} entries must be integers, got {text!r}") from exc
        if not 0 <= value <= FIELD_BLOCKS_MAX:
            raise ValueError(f"{FIELD_BLOCKS_VAR} entries must be in [0, {FIELD_BLOCKS_MAX}], got {text!r}")
        out.append(value)
    return out[0], out[1]


def parse_field_kinds(text) -> tuple[float, float, float]:
    """``KRABBY_FIELD_KINDS`` -> ``(bar, slab, pit)`` fractions (``"0.4:0.4:0.2"`` default mix).

    Three non-negative numbers that must SUM TO 1 (within 1e-6). They are not renormalised: a
    mix that does not add up is a typo, and quietly rescaling it would train a mix nobody wrote.
    """
    parts = str(text).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f'{FIELD_KINDS_VAR} must be "bar:slab:pit", got {text!r}')
    vals = tuple(_parse_float(FIELD_KINDS_VAR, p) for p in parts)
    if any(v < 0.0 for v in vals):
        raise ValueError(f"{FIELD_KINDS_VAR} fractions must be non-negative, got {text!r}")
    if abs(sum(vals) - 1.0) > 1e-6:
        raise ValueError(f"{FIELD_KINDS_VAR} fractions must sum to 1, got {text!r} (sum {sum(vals):.6g})")
    return vals


def parse_field_edges(text) -> bool:
    """``KRABBY_FIELD_EDGES`` -> ``vertical_edges``: ``ramp`` (False, the default rendering) or
    ``wall`` (True: forced vertical risers on every block edge)."""
    value = str(text).strip().lower()
    if value not in ("ramp", "wall"):
        raise ValueError(f'{FIELD_EDGES_VAR} must be "ramp" or "wall", got {text!r}')
    return value == "wall"


def parse_field_min_gap(text) -> float:
    """``KRABBY_FIELD_MIN_GAP`` -> metres of clear ground required between blocks, in [0, 1]."""
    value = _parse_float(FIELD_MIN_GAP_VAR, text)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{FIELD_MIN_GAP_VAR} must be in [0, 1] m, got {text!r}")
    return value


def parse_field_size_scale(text) -> float:
    """``KRABBY_FIELD_SIZE_SCALE`` -> multiplier on the bar/slab/pit SIZE ranges, in (0.25, 1.5]."""
    value = _parse_float(FIELD_SIZE_SCALE_VAR, text)
    if not 0.25 < value <= 1.5:
        raise ValueError(f"{FIELD_SIZE_SCALE_VAR} must be in (0.25, 1.5], got {text!r}")
    return value


FIELD_KNOB_PARSERS = {
    FIELD_PAD_R_VAR: parse_field_pad_r,
    FIELD_BLOCKS_VAR: parse_field_blocks,
    FIELD_KINDS_VAR: parse_field_kinds,
    FIELD_EDGES_VAR: parse_field_edges,
    FIELD_MIN_GAP_VAR: parse_field_min_gap,
    FIELD_SIZE_SCALE_VAR: parse_field_size_scale,
}
"""``{KRABBY_FIELD_*: parser}`` -- for the campaign driver's value round-trip table."""


def read_omni_geom(environ=None) -> str | None:
    """``KRABBY_FLAT_TERRAIN_GEOM`` as ``apply_flat_walk_knobs`` reads it (``strip().lower()``),
    or None when unset."""
    env = os.environ if environ is None else environ
    raw = env.get(GEOM_VAR)
    return None if raw is None else str(raw).strip().lower()


def require_omni_geom(environ=None) -> str:
    """Require ``KRABBY_FLAT_TERRAIN_GEOM`` in ``("ring", "field")`` and no MIXED knob family.

    Raises ``RuntimeError`` naming the leaked value for anything else (same message family as
    :func:`require_ring_geom`), and naming the offending variables when a ``KRABBY_RING_*`` knob is
    set under ``field`` or a ``KRABBY_FIELD_*`` knob under ``ring``. Blank values count as unset,
    as they do when the knobs are applied.
    """
    env = os.environ if environ is None else environ
    raw = env.get(GEOM_VAR)
    value = read_omni_geom(env)
    if value not in OMNI_GEOMS:
        shown = "<unset>" if raw is None else repr(raw)
        raise RuntimeError(
            f"the omni task requires {GEOM_VAR} in {OMNI_GEOMS}, found {shown}. "
            "A forward/sideways geometry preset is in the environment: the omni presets set "
            f"{GEOM_VAR}={RING_GEOM} in every window (or {FIELD_GEOM} for the phase-2 field), and "
            "any other value means this process inherited a corridor preset (whose geometry "
            "helpers are silently inert on the omni generators)."
        )
    foreign = FIELD_VARS if value == RING_GEOM else RING_VARS
    leaked = [k for k in foreign if not _is_blank(env.get(k))]
    if leaked:
        raise RuntimeError(
            f"{GEOM_VAR}={value!r} but {', '.join(leaked)} is set: those knobs belong to the "
            f"{'field' if value == RING_GEOM else 'ring'} geometry and would apply nothing here. "
            "Unset them, or switch the geometry."
        )
    return value


def is_field_generator(sub_terrains, field_cls) -> bool:
    """True when any sub-terrain is a ``field_cls`` -- how the parkour term tells field mode
    from ring mode (by class, like the ring guard; proportions may be 0 after the knob block)."""
    return any(isinstance(sub, field_cls) for sub in (sub_terrains or {}).values())


def assert_field_sub_terrains(sub_terrains, field_cls, flat_cls) -> tuple[list[str], list[str]]:
    """Raise unless every sub-terrain is field or flat and at least one is field.

    :func:`classify_sub_terrains` with the field class in the ring slot, so a ring sub-terrain
    under ``field`` is FOREIGN (the scene default was not swapped -- the install hook did not
    run) and fails here instead of training a ring while reporting a field.
    """
    if not sub_terrains:
        raise RuntimeError("terrain generator has no sub_terrains; expected the omni field generator.")
    field_keys, flat_keys, foreign = classify_sub_terrains(sub_terrains, field_cls, flat_cls)
    if foreign:
        shown = ", ".join(f"{k}={type(sub_terrains[k]).__name__}" for k in foreign)
        raise RuntimeError(
            f"{GEOM_VAR}={FIELD_GEOM} requires the omni field generator, but these sub-terrains are "
            f"neither {field_cls.__name__} nor {flat_cls.__name__}: {shown} "
            "(was install_omni_terrain_generator called before the knob block?)"
        )
    if not field_keys:
        raise RuntimeError(
            f"the omni field generator must carry at least one {field_cls.__name__} sub-terrain; "
            f"found only flat ones ({', '.join(flat_keys) or 'none'})."
        )
    return field_keys, flat_keys


def apply_field_geometry(
    sub_terrains,
    field_cls,
    flat_cls,
    *,
    pad_r: float | None = None,
    blocks: tuple[int, int] | None = None,
    kinds: tuple[float, float, float] | None = None,
    vertical_edges: bool | None = None,
    min_gap: float | None = None,
    size_scale: float | None = None,
) -> None:
    """Write the field knobs into the sub-terrain cfgs (each a no-op where it is None).

    ``pad_r`` goes to EVERY omni sub-terrain, field and flat alike (the flat tiles' anchors sit at
    ``(pad + half-width) / 2`` like the field's); ``blocks`` is ``(sparse, dense)`` over exactly two
    field sub-terrains in generator order; ``kinds`` sets ``bar_fraction`` / ``pit_fraction`` (slab
    = the rest); ``size_scale`` multiplies the three size RANGES of every field sub-terrain.
    """
    field_keys, flat_keys = assert_field_sub_terrains(sub_terrains, field_cls, flat_cls)
    if pad_r is not None:
        for key in field_keys + flat_keys:
            sub = sub_terrains[key]
            half = float(getattr(sub, "ring_outer_radius"))
            if not pad_r < half - FIELD_PAD_R_HEADROOM:
                raise ValueError(
                    f"{FIELD_PAD_R_VAR}={pad_r} must be below sub-terrain '{key}'.ring_outer_radius="
                    f"{half} minus {FIELD_PAD_R_HEADROOM} m"
                )
            sub.clear_radius = float(pad_r)
    if blocks is not None:
        if len(field_keys) != 2:
            raise RuntimeError(
                f'{FIELD_BLOCKS_VAR} is a "sparse:dense" pair and needs exactly two field '
                f"sub-terrains; found {len(field_keys)} ({', '.join(field_keys) or 'none'})"
            )
        for key, count in zip(field_keys, blocks):
            sub_terrains[key].n_blocks = int(count)
    for key in field_keys:
        sub = sub_terrains[key]
        if kinds is not None:
            sub.bar_fraction = float(kinds[0])
            sub.pit_fraction = float(kinds[2])
        if vertical_edges is not None:
            sub.vertical_edges = bool(vertical_edges)
        if min_gap is not None:
            sub.min_gap = float(min_gap)
        if size_scale is not None:
            for attr in ("bar_length_range", "bar_thickness_range", "slab_side_range"):
                lo, hi = getattr(sub, attr)
                setattr(sub, attr, (float(lo) * size_scale, float(hi) * size_scale))


def _sync_parkour_radii(cfg, sub, *, note: str | None = None) -> None:
    """Keep ``parkours.base_parkour``'s two radii equal to the terrain's (see the ring branch)."""
    parkour = getattr(getattr(cfg, "parkours", None), "base_parkour", None)
    if parkour is None or not hasattr(parkour, "clear_radius_m"):
        return
    for term_attr, sub_attr in (("clear_radius_m", "clear_radius"),
                                ("ring_outer_radius_m", "ring_outer_radius")):
        want = float(getattr(sub, sub_attr))
        if float(getattr(parkour, term_attr)) != want:
            print(
                f"[omni-knobs] parkours.base_parkour.{term_attr} "
                f"({getattr(parkour, term_attr)}) did not match the terrain's "
                f"{sub_attr} ({want}); syncing to the terrain"
                + (f" ({note})." if note else "."),
                flush=True,
            )
            setattr(parkour, term_attr, want)


def apply_omni_field_knobs(cfg, environ=None, classes=None) -> None:
    """The ``field`` branch of :func:`apply_omni_ring_knobs` (guarded there by
    :func:`require_omni_geom`): field-generator class assertion, ``KRABBY_FIELD_*``, radius sync.

    ``classes`` is ``(field_cls, flat_cls)``; defaults to the real cfg classes (lazy import).
    The pad-radius bound is checked against the installed sub-terrains' own half-width.
    """
    env = os.environ if environ is None else environ
    if classes is None:
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import (  # noqa: PLC0415
            OmniFieldTerrainCfg,
            OmniFlatTerrainCfg,
        )

        field_cls, flat_cls = OmniFieldTerrainCfg, OmniFlatTerrainCfg
    else:
        field_cls, flat_cls = classes

    terrain = getattr(cfg.scene, "terrain", None)
    tg = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    if tg is None:
        raise RuntimeError(
            "the omni field geometry requires scene.terrain.terrain_generator to be "
            "OMNI_FIELD_TERRAINS_CFG; found None (a flat/plane terrain importer)."
        )
    field_keys, flat_keys = assert_field_sub_terrains(tg.sub_terrains, field_cls, flat_cls)
    half = min(float(getattr(tg.sub_terrains[k], "ring_outer_radius")) for k in field_keys + flat_keys)

    def _get(var, parse, **kw):
        raw = env.get(var)
        return None if _is_blank(raw) else parse(raw, **kw)

    kinds = _get(FIELD_KINDS_VAR, parse_field_kinds)
    apply_field_geometry(
        tg.sub_terrains,
        field_cls,
        flat_cls,
        pad_r=_get(FIELD_PAD_R_VAR, parse_field_pad_r, half_width=half),
        blocks=_get(FIELD_BLOCKS_VAR, parse_field_blocks),
        kinds=kinds,
        vertical_edges=_get(FIELD_EDGES_VAR, parse_field_edges),
        min_gap=_get(FIELD_MIN_GAP_VAR, parse_field_min_gap),
        size_scale=_get(FIELD_SIZE_SCALE_VAR, parse_field_size_scale),
    )
    # The parkour term's class defaults describe the RING (2.0 / 4.5); under the field they always
    # differ, so the sync is expected here -- and it is what makes reach_edge mean "left the pad".
    _sync_parkour_radii(cfg, tg.sub_terrains[(field_keys + flat_keys)[0]], note="field geometry")


def install_omni_terrain_generator(cfg, environ=None, field_generator=None) -> str | None:
    """Install the omni FIELD generator when ``KRABBY_FLAT_TERRAIN_GEOM=field``; else a no-op.

    Called FIRST in both omni env ``__post_init__``s (after the phase-mode guard, before
    ``super().__post_init__()``), so everything downstream -- the forward knob block's flat
    fraction, difficulty band and curriculum switch, the PLAY head, :func:`apply_omni_ring_knobs`
    -- sees the field exactly as it sees the ring. The generator is a ``copy.deepcopy`` of
    ``OMNI_FIELD_TERRAINS_CFG`` (``field_generator`` overrides it for tests): the knob block
    mutates sub-terrain cfgs in place, and a shared module-level object would carry one env cfg's
    knobs into the next.

    Returns the geometry value read (None when unset). Does NOT validate it -- the guard in
    :func:`apply_omni_ring_knobs` does, after the knob block, with the full message.
    """
    import copy  # noqa: PLC0415

    env = os.environ if environ is None else environ
    geom = read_omni_geom(env)
    if geom != FIELD_GEOM:
        return geom
    if field_generator is None:
        from parkour_isaaclab.terrains.omni.config.omni_field import (  # noqa: PLC0415
            OMNI_FIELD_TERRAINS_CFG,
        )

        field_generator = OMNI_FIELD_TERRAINS_CFG
    terrain = getattr(cfg.scene, "terrain", None)
    if terrain is None:
        raise RuntimeError(
            f"{GEOM_VAR}={FIELD_GEOM} but the scene has no terrain importer to install the field into"
        )
    terrain.terrain_generator = copy.deepcopy(field_generator)
    return geom
