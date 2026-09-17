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
]

GEOM_VAR = "KRABBY_FLAT_TERRAIN_GEOM"
RING_GEOM = "ring"
CLEAR_R_VAR = "KRABBY_RING_CLEAR_R"
BLOCKS_VAR = "KRABBY_RING_BLOCKS"


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


def apply_omni_ring_knobs(cfg, environ=None, classes=None) -> None:
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
    """
    env = os.environ if environ is None else environ
    require_ring_geom(env)

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
