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

THE SITE GEOMETRY (site teacher, experiments/2026-10-28_site_teacher/DESIGN.md §3, 2026-10-28; ported from
``worktree-sen0581-cameras`` 2a3e3c51 by the rough-terrain campaign, experiments/2026-11-10_rough_terrain/PLAN.md
step 1, 2026-11-10)
-----------------------------------------------------------------------------------------------
``KRABBY_FLAT_TERRAIN_GEOM=site`` selects the construction-site course (``OMNI_SITE_TERRAINS_CFG``: one
column kind per sub-terrain, ``site_raster.SITE_SUBTERRAIN_KEYS``). :func:`install_omni_terrain_generator`
installs it like the field; :func:`require_omni_geom` accepts it, refusing ``KRABBY_RING_*`` and ALLOWING
``KRABBY_FIELD_*`` (they configure the two legacy-field columns); :func:`apply_omni_site_knobs` (the
``site`` branch of the dispatcher) asserts the generator, applies ``KRABBY_SITE_MIX`` (columns) and
``KRABBY_SITE_DIFF`` (the site-difficulty band), the pad / roughness / ramp caps, the radius sync and the
pad-level spawn. The site terrain knobs (:data:`SITE_VARS`) under ring / field are IGNORED with a
``[KNOB] ... ignored`` line, never refused (a row's geometry is its own, DESIGN §3.7).

``KRABBY_SITE_LADDER`` (PLAN_rev1 B4, 2026-10-29) is terrain geometry like ``KRABBY_SITE_DIFF`` (never stripped
from an eval row): blank = today's ladder, ``b4`` = the bottom rung (``site_terrains.SITE_LADDER_BOTTOMS``), which
the site applier arms by writing ``site_terrains.SITE_LADDER_BODIES["b4"]`` into every hazard sub-terrain's
``function``. Its parent is the site geometry itself; under ring / field it is ignored like every site knob.

Port scope (rough-terrain, 2026-11-10): only the site TERRAIN knobs above are ported. The site teacher's other
``KRABBY_SITE_*`` knobs (site physics DR, site curriculum / rewards / terminations, the critic group) and
``KRABBY_MAX_INIT_TERRAIN_LEVEL`` are NOT, so :func:`refuse_unported_site_knobs` REFUSES any of them
(:data:`SITE_UNPORTED_VARS`) in every geometry rather than letting a stale site env file apply nothing. The ramp cap
reads no site foot friction (:func:`site_friction_floor` is None: ramps stay at 20 deg).

THE ROUGH-TERRAIN LADDER (experiments/2026-11-10_rough_terrain/PLAN.md step 2, 2026-11-10)
------------------------------------------------------------------------------------------
``KRABBY_SITE_LADDER=rt`` (under ``KRABBY_FLAT_TERRAIN_GEOM=site`` only) selects the traversable-terrain columns:
:func:`install_omni_terrain_generator` installs ``OMNI_SITE_RT_TERRAINS_CFG`` (:data:`RT_SUBTERRAIN_KEYS`, column
order = key order: gravel next to the flat / field columns) instead of the site generator, and the site applier
takes its rt branch -- ``KRABBY_SITE_MIX`` is REQUIRED (blank refused) and may name only :data:`RT_COLUMN_KINDS` (blocker
and unused site kinds are refused), every rt hazard column gets ``site_terrains.RT_BODY``; everything else (FIELD_*
on the legacy columns, pad, d mapping, caps, radius sync, pad-level spawn) as on the site. Today's ladder and ``b4``
are untouched.

``KRABBY_RT_FRICTION=gravel:LO:HI`` (:func:`parse_rt_friction`, :func:`apply_rt_friction_knobs`; DEFAULT OFF, a
training AND gravel-eval knob, never stripped from an eval row) arms the friction contract: ground (terrain and sim
material) friction combine ``multiply`` at mu 1.0, and the robot's startup material event replaced by
``mdp/rt_friction.rt_friction_contract`` -- one effective mu per env on EVERY shape (static = dynamic): ``(b + 1) / 2``
of the event's own ``num_buckets`` buckets ``b ~ U(friction_range)`` (TDR's 64 x U(0.6, 2.0)) off the gravel columns,
``U(LO, HI)`` on them, from a generator keyed by (env seed, env index) -- and ``cfg.rt_friction`` (the privileged
friction input then reads ``2 mu_eff - 1``, TDR's own encoding). Only under ``site`` + ``rt``; an rt mix with gravel
columns needs it. Unset: nothing is written.
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
    # site geometry (site teacher DESIGN §1.2 / §3; skeleton, milestone M1)
    "SITE_GEOM",
    "SITE_MIX_VAR",
    "SITE_DIFF_VAR",
    "SITE_VARS",
    "SITE_MIX_MAX_COLS",
    "DEFAULT_SITE_MIX",
    "parse_site_mix",
    "parse_site_diff",
    "SITE_KNOB_PARSERS",
    "DEFAULT_SITE_DIFF",
    "SITE_PAD_R",
    "SITE_NOISE_MAX",
    "site_ramp_cap_deg",
    "assert_site_sub_terrains",
    "apply_site_mix",
    "apply_omni_site_knobs",
    # site bottom rung (PLAN_rev1 B4, 2026-10-29)
    "SITE_LADDER_VAR",
    "SITE_LADDERS",
    "parse_site_ladder",
    # rough-terrain port (2026-11-10): the site teacher's knobs this branch does NOT carry
    "SITE_UNPORTED_VARS",
    "refuse_unported_site_knobs",
    "site_friction_floor",
    # rough-terrain ladder (2026-11-10, rough-terrain PLAN step 2)
    "RT_LADDER",
    "RT_COLUMN_KINDS",
    "RT_SUBTERRAIN_KEYS",
    "RT_HAZARD_KINDS",
    "RT_REFUSED_KINDS",
    "RT_TRAIN_MIX",
    "assert_rt_sub_terrains",
    "RT_FRICTION_VAR",
    "RT_FRICTION_ATTR",
    "RT_FRICTION_KIND",
    "RT_FRICTION_MU_MIN",
    "RT_FRICTION_MU_MAX",
    "RT_GRAVEL_KEYS",
    "parse_rt_friction",
    "RT_KNOB_PARSERS",
    "apply_rt_friction_knobs",
]

GEOM_VAR = "KRABBY_FLAT_TERRAIN_GEOM"
RING_GEOM = "ring"
CLEAR_R_VAR = "KRABBY_RING_CLEAR_R"
BLOCKS_VAR = "KRABBY_RING_BLOCKS"
RING_VARS = (CLEAR_R_VAR, BLOCKS_VAR)

FIELD_GEOM = "field"
# ``SITE_GEOM`` (defined with the site knobs below) is the third value.
OMNI_GEOMS = (RING_GEOM, FIELD_GEOM, "site")
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

# Site geometry (site teacher, experiments/2026-10-28_site_teacher/DESIGN.md §3). ``site`` is a VALUE
# of KRABBY_FLAT_TERRAIN_GEOM (the third of OMNI_GEOMS).
SITE_GEOM = "site"
SITE_MIX_VAR = "KRABBY_SITE_MIX"
SITE_DIFF_VAR = "KRABBY_SITE_DIFF"
#: The bottom-rung ladder (PLAN_rev1 B4): blank = today's ladder.
SITE_LADDER_VAR = "KRABBY_SITE_LADDER"
SITE_VARS = (SITE_MIX_VAR, SITE_DIFF_VAR, SITE_LADDER_VAR)
#: ``site_terrains.SITE_LADDER_BOTTOMS``' names restated (standalone, torch-free module; a test pins the two equal).
SITE_LADDERS = ("b4",)
#: Upper bound on the total column count of a mix.
SITE_MIX_MAX_COLS = 40
#: The mix when KRABBY_SITE_MIX is unset under ``site``: 20 columns, flat 0.2, legacy field 0.2,
#: hazards 0.6 (pits get 2 as the holes-row kind).
DEFAULT_SITE_MIX = ("flat=4,field_sparse=2,field_dense=2,pits=2,trench=1,ledge_down=1,ledge_up=1,terraces=1,"
                    "snags=1,straddle=1,chests=1,walls=1,rubble=1,ramps=1")
# ``site_raster.SITE_KINDS`` restated (this module is path-loaded standalone and stays torch-free);
# a test pins the two equal.
_SITE_KINDS = ("flat", "field_sparse", "field_dense", "pits", "trench", "ledge_down", "ledge_up", "terraces",
               "snags", "straddle", "chests", "walls", "rubble", "ramps")
# ``site_raster.SITE_SUBTERRAIN_KEYS`` restated (same order, same reason, same pin).
_SITE_SUBTERRAIN_KEYS = ("parkour_flat", "site_field_sparse", "site_field_dense", "site_pits", "site_trench",
                         "site_ledge_down", "site_ledge_up", "site_terraces", "site_snags", "site_straddle",
                         "site_chests", "site_walls", "site_rubble", "site_ramps")
#: The default site-difficulty band (``KRABBY_SITE_DIFF`` unset).
DEFAULT_SITE_DIFF = (0.0, 1.0)
#: The site pad (m): the default, and the largest ``KRABBY_FIELD_PAD_R`` the site layouts accept (their
#: hazard radii start 0.3 m beyond it; ``site_terrains.PAD_R_M``).
SITE_PAD_R = 1.7
#: The half-width every site feature lies in (m); ``ring_outer_radius_m`` is synced to it.
SITE_HALF_WIDTH = 4.80
#: Largest roughness amplitude (m) a site sub-terrain may carry (``noise_range[1]``; the PLAN's 1-3 cm cap).
SITE_NOISE_MAX = 0.03
#: Ramp slope cap (deg) and its margin below the friction angle (``site_terrains.RAMP_MAX_DEG`` / §3.3).
SITE_RAMP_MAX_DEG = 20.0
SITE_RAMP_FRICTION_MARGIN_DEG = 3.0
#: Forward-block knobs the site applier supersedes / refuses.
FLAT_FRAC_VAR = "KRABBY_FLAT_TERRAIN_FLAT_FRAC"
RSI_FRAC_VAR = "KRABBY_RSI_FRAC"
#: The spawn event the site applier arms ``spawn_level_z`` on.
SPAWN_EVENT = "reset_root_state"
#: The site teacher's knobs that the rough-terrain port does NOT carry (2026-11-10): its physics DR
#: (``site_dr_knobs``), its curriculum / rewards / terminations and PLAN_rev1 training-only keys
#: (``site_task_knobs``, ``site_keys.EVAL_STRIP_KEYS`` on the site branch) and its critic-only obs group. Each is
#: REFUSED when set (:func:`refuse_unported_site_knobs`), in every geometry: on this branch it would apply nothing.
SITE_UNPORTED_VARS = (
    # site physics DR
    "KRABBY_SITE_GROUND_COMBINE", "KRABBY_SITE_FOOT_FRICTION", "KRABBY_SITE_FOOT_DYN_RATIO",
    "KRABBY_SITE_FOOT_RESTITUTION", "KRABBY_SITE_ACTION_DELAY", "KRABBY_SITE_PROP_NOISE", "KRABBY_SITE_ROD_SCALE",
    "KRABBY_SITE_CAM_SCALE",
    # site curriculum, rewards, terminations
    "KRABBY_SITE_PROMOTE", "KRABBY_SITE_CRASH_N", "KRABBY_SITE_VOID_W", "KRABBY_SITE_EDGE_W", "KRABBY_SITE_BLOCKER_W",
    "KRABBY_SITE_SLACK", "KRABBY_SITE_IMPACT",
    # PLAN_rev1 training-only keys
    "KRABBY_SITE_SLACK_GEOM", "KRABBY_SITE_SLACK_HALFW", "KRABBY_SITE_PROMOTE_RULE", "KRABBY_MAX_INIT_TERRAIN_LEVEL",
    # the critic-only observation group
    "KRABBY_SITE_CRITIC_OBS",
)
#: The rough-terrain ladder (``KRABBY_SITE_LADDER=rt``; ``site_terrains.RT_LADDER`` restated, a test pins the two).
RT_LADDER = "rt"
#: The rt generator's column kinds, in its sub-terrain (= column) order; ``site_raster.TILE_KINDS`` names each one.
RT_COLUMN_KINDS = ("flat", "gravel", "field_sparse", "field_dense", "slopes", "hills", "ledge_up", "ledge_down",
                  "pits", "rubble")
#: ``OMNI_SITE_RT_TERRAINS_CFG``'s sub-terrain keys, in :data:`RT_COLUMN_KINDS` order (``config/omni_site.py``; pinned).
RT_SUBTERRAIN_KEYS = ("parkour_flat", "rt_gravel", "site_field_sparse", "site_field_dense", "rt_slopes", "rt_hills",
                      "site_ledge_up", "site_ledge_down", "site_pits", "site_rubble")
#: The rt hazard columns (``site_terrains.RT_KINDS``): every one stamps on ``site_terrains.RT_BODY``.
RT_HAZARD_KINDS = ("gravel", "slopes", "hills", "ledge_up", "ledge_down", "pits", "rubble")
#: Site column kinds an rt mix may not name: the blockers (PLAN settled 1: nothing that forces a stop) and the site
#: kinds the rough-terrain set does not use.
RT_REFUSED_KINDS = ("chests", "walls", "trench", "terraces", "snags", "straddle", "ramps")
#: The rough-terrain training mix (PLAN Design, 20 columns).
RT_TRAIN_MIX = ("flat=3,field_sparse=2,field_dense=1,slopes=2,hills=2,ledge_up=2,ledge_down=2,pits=2,gravel=2,"
                "rubble=2")
#: The rt-only column kinds (not site kinds): a mix naming one is an rt mix.
_RT_ONLY_KINDS = tuple(k for k in RT_COLUMN_KINDS if k not in _SITE_KINDS)
#: The friction contract (``KRABBY_RT_FRICTION=gravel:LO:HI``).
RT_FRICTION_VAR = "KRABBY_RT_FRICTION"
#: The env-cfg attribute the contract writes (``mdp/rt_friction.RT_FRICTION_ATTR``; the observation reads it).
RT_FRICTION_ATTR = "rt_friction"
#: The one column kind the contract's low-mu band applies to, and its sub-terrain keys.
RT_FRICTION_KIND = "gravel"
RT_GRAVEL_KEYS = ("rt_gravel",)
#: Bounds on the gravel band (typo guard; PhysX needs mu >= 0, TDR's effective mu tops out at 1.5).
RT_FRICTION_MU_MIN, RT_FRICTION_MU_MAX = 0.05, 2.0
#: The robot material event the contract replaces (TDR's ``events.physics_material``) and the leaf of its function.
RT_MATERIAL_EVENT = "physics_material"
RT_MATERIAL_EVENT_FUNC = "randomize_rigid_body_material"


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


def apply_omni_ring_knobs(cfg, environ=None, classes=None, field_classes=None, site_classes=None) -> None:
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

    Site teacher (DESIGN §3.1, 2026-10-28): ``site`` goes to :func:`apply_omni_site_knobs`
    (``site_classes`` = ``(flat_cls, site_field_cls, site_cls)``); under ring / field a set
    ``KRABBY_SITE_*`` prints ``[KNOB] <var>=<value> ignored (geometry is <g>)`` and applies nothing.
    """
    env = os.environ if environ is None else environ
    geom = require_omni_geom(env)
    refuse_unported_site_knobs(env)
    if geom == SITE_GEOM:
        apply_omni_site_knobs(cfg, env, classes=site_classes)
        return
    # DESIGN §3.7: a site knob under ring / field is a hand-composed env -- ignored, never refused.
    for var in SITE_VARS:
        if not _is_blank(env.get(var)):
            print(f"[KNOB] {var}={env.get(var)} ignored (geometry is {geom})", flush=True)
    if geom == FIELD_GEOM:
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
    """Require ``KRABBY_FLAT_TERRAIN_GEOM`` in :data:`OMNI_GEOMS` and no MIXED knob family.

    Raises ``RuntimeError`` naming the leaked value for anything else (same message family as
    :func:`require_ring_geom`), and naming the offending variables when a ``KRABBY_RING_*`` knob is
    set under ``field`` or a ``KRABBY_FIELD_*`` knob under ``ring``. Blank values count as unset,
    as they do when the knobs are applied.

    Under ``site`` (DESIGN §3.1) ``KRABBY_RING_*`` is refused and ``KRABBY_FIELD_*`` is ALLOWED: it
    configures the site generator's two legacy-field columns.
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


def install_omni_terrain_generator(cfg, environ=None, field_generator=None, site_generator=None,
                                   rt_generator=None) -> str | None:
    """Install the omni FIELD generator when ``KRABBY_FLAT_TERRAIN_GEOM=field`` (the SITE generator,
    ``OMNI_SITE_TERRAINS_CFG``, when ``site``; ``site_generator`` overrides it for tests -- and the rough-terrain
    generator ``OMNI_SITE_RT_TERRAINS_CFG`` when ``site`` with ``KRABBY_SITE_LADDER=rt``, ``rt_generator`` its test
    override); else a no-op.

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
    if geom == SITE_GEOM and parse_site_ladder(env.get(SITE_LADDER_VAR)) == RT_LADDER:
        if rt_generator is None:
            from parkour_isaaclab.terrains.omni.config.omni_site import (  # noqa: PLC0415
                OMNI_SITE_RT_TERRAINS_CFG,
            )

            rt_generator = OMNI_SITE_RT_TERRAINS_CFG
        generator = rt_generator
    elif geom == SITE_GEOM:
        if site_generator is None:
            from parkour_isaaclab.terrains.omni.config.omni_site import (  # noqa: PLC0415
                OMNI_SITE_TERRAINS_CFG,
            )

            site_generator = OMNI_SITE_TERRAINS_CFG
        generator = site_generator
    elif geom == FIELD_GEOM:
        if field_generator is None:
            from parkour_isaaclab.terrains.omni.config.omni_field import (  # noqa: PLC0415
                OMNI_FIELD_TERRAINS_CFG,
            )

            field_generator = OMNI_FIELD_TERRAINS_CFG
        generator = field_generator
    else:
        return geom
    terrain = getattr(cfg.scene, "terrain", None)
    if terrain is None:
        if geom == FIELD_GEOM:
            raise RuntimeError(
                f"{GEOM_VAR}={FIELD_GEOM} but the scene has no terrain importer to install the field into"
            )
        raise RuntimeError(f"{GEOM_VAR}={geom} but the scene has no terrain importer to install the {geom} into")
    terrain.terrain_generator = copy.deepcopy(generator)
    return geom


# ================================================================================ SITE
# Site teacher (experiments/2026-10-28_site_teacher/DESIGN.md §1.2, §3). Default-off: the parsers map
# blank to None, and nothing below runs unless KRABBY_FLAT_TERRAIN_GEOM=site.


_AUTO = object()


def parse_site_mix(text, ladder=_AUTO) -> dict[str, int] | None:
    """``KRABBY_SITE_MIX`` -> ``{kind: count}`` over every site column kind, in ``SITE_KINDS`` order
    (kinds the text omits are 0); None when blank.

    Grammar ``name=count[,name=count]...``, whitespace around tokens allowed; ``name`` a site column
    kind, ``count`` a non-negative integer (floats refused); duplicates refused; total in
    ``[1, SITE_MIX_MAX_COLS]``.

    Rough-terrain ladder: with ``ladder="rt"`` -- or, ``ladder`` not given, when the text names an rt-only kind
    (slopes / hills / gravel) -- the names are :data:`RT_COLUMN_KINDS` (a :data:`RT_REFUSED_KINDS` name is refused) and the
    dict is over :data:`RT_COLUMN_KINDS`, in that order. ``ladder`` given as a site ladder (None / ``b4``) refuses the
    rt-only kinds.
    """
    if _is_blank(text):
        return None
    names_in = [tok.partition("=")[0].strip() for tok in str(text).split(",")]
    rt = any(name in _RT_ONLY_KINDS for name in names_in) if ladder is _AUTO else ladder == RT_LADDER
    kinds = RT_COLUMN_KINDS if rt else _SITE_KINDS
    counts: dict[str, int] = {}
    for token in str(text).split(","):
        name, sep, raw = token.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep or not name:
            raise ValueError(f'{SITE_MIX_VAR} entries must be "name=count", got {token.strip()!r} in {text!r}')
        if rt and name in RT_REFUSED_KINDS:
            raise ValueError(f"{SITE_MIX_VAR}: kind {name!r} is refused under {SITE_LADDER_VAR}={RT_LADDER} (no "
                             f"blockers or unused site kinds; valid: {', '.join(RT_COLUMN_KINDS)}) in {text!r}")
        if not rt and name in _RT_ONLY_KINDS:
            raise ValueError(f"{SITE_MIX_VAR}: kind {name!r} exists only under {SITE_LADDER_VAR}={RT_LADDER}, got "
                             f"{text!r}")
        if name not in kinds:
            raise ValueError(f"{SITE_MIX_VAR}: unknown kind {name!r} in {text!r}; valid: {', '.join(kinds)}")
        if name in counts:
            raise ValueError(f"{SITE_MIX_VAR}: kind {name!r} given twice in {text!r}")
        if not raw.isdigit() or not raw.isascii():
            raise ValueError(f"{SITE_MIX_VAR}: count for {name!r} must be a non-negative integer, got {raw!r}")
        counts[name] = int(raw)
    total = sum(counts.values())
    if not 1 <= total <= SITE_MIX_MAX_COLS:
        raise ValueError(f"{SITE_MIX_VAR} must total 1..{SITE_MIX_MAX_COLS} columns, got {total} ({text!r})")
    return {name: counts.get(name, 0) for name in kinds}


def parse_site_diff(text) -> tuple[float, float] | None:
    """``KRABBY_SITE_DIFF`` -> ``(lo, hi)`` site-difficulty band, ``0 <= lo <= hi <= 1``; None when
    blank (the applier's default band is ``0:1``)."""
    if _is_blank(text):
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f'{SITE_DIFF_VAR} must be "lo:hi", got {text!r}')
    lo, hi = (_parse_float(SITE_DIFF_VAR, part) for part in parts)
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"{SITE_DIFF_VAR} needs 0 <= lo <= hi <= 1, got {text!r}")
    return lo, hi


def parse_site_ladder(text) -> str | None:
    """``KRABBY_SITE_LADDER`` -> a :data:`SITE_LADDERS` name or :data:`RT_LADDER` (read as ``strip().lower()``); None
    when blank (today's ladder). Anything else is refused."""
    if _is_blank(text):
        return None
    value = str(text).strip().lower()
    if value not in SITE_LADDERS and value != RT_LADDER:
        raise ValueError(f"{SITE_LADDER_VAR} must be blank (today's ladder) or one of {SITE_LADDERS + (RT_LADDER,)}, "
                         f"got {text!r}")
    return value


SITE_KNOB_PARSERS = {
    SITE_MIX_VAR: parse_site_mix,
    SITE_DIFF_VAR: parse_site_diff,
    SITE_LADDER_VAR: parse_site_ladder,
}
"""``{KRABBY_SITE_*: parser}`` -- for the campaign driver's value round-trip table (DESIGN §1.2)."""


def refuse_unported_site_knobs(environ=None) -> None:
    """Raise ``RuntimeError`` naming every :data:`SITE_UNPORTED_VARS` key set (non-blank) in ``environ``.

    Called first in :func:`apply_omni_ring_knobs`, so it runs in every omni process; with none of them set (every
    main recipe and eval row) it reads the environment and writes nothing."""
    env = os.environ if environ is None else environ
    armed = [f"{var}={env.get(var)!r}" for var in SITE_UNPORTED_VARS if not _is_blank(env.get(var))]
    if armed:
        raise RuntimeError(
            f"{', '.join(armed)}: the site teacher's DR / curriculum / reward / termination / critic knobs are NOT "
            "ported to this branch (only KRABBY_SITE_MIX / KRABBY_SITE_DIFF / KRABBY_SITE_LADDER are); unset them"
        )


def site_friction_floor(environ=None) -> float | None:
    """The site foot-friction floor the ramp cap reads: always None on this branch (the site teacher's
    ``site_dr_knobs.parse_friction_floor`` read ``KRABBY_SITE_FOOT_FRICTION``, which is not ported and is refused by
    :func:`refuse_unported_site_knobs`), so :func:`site_ramp_cap_deg` is its 20 deg cap."""
    return None


def site_ramp_cap_deg(mu_floor: float | None) -> float:
    """``min(20, degrees(atan(mu_floor)) - 3)`` (DESIGN §3.3, "capped by friction"); 20 deg when the site
    foot friction is not armed. ``site_terrains.ramp_cap_deg`` restated (a test pins the two equal)."""
    if mu_floor is None:
        return SITE_RAMP_MAX_DEG
    return min(SITE_RAMP_MAX_DEG, math.degrees(math.atan(float(mu_floor))) - SITE_RAMP_FRICTION_MARGIN_DEG)


def assert_site_sub_terrains(sub_terrains, flat_cls, site_field_cls, site_cls) -> None:
    """Raise unless ``sub_terrains`` is the site generator's: keys exactly the site sub-terrain keys in
    order, ``parkour_flat`` a ``flat_cls``, the two legacy-field keys ``site_field_cls``, and every hazard
    key a ``site_cls`` whose ``kind`` is its column kind. A ring / field generator (the install hook did
    not run) or a hand-edited mix of keys fails here, by class, like the ring and field guards."""
    if not sub_terrains:
        raise RuntimeError("terrain generator has no sub_terrains; expected the omni site generator.")
    keys = tuple(sub_terrains)
    if keys != _SITE_SUBTERRAIN_KEYS:
        raise RuntimeError(
            f"{GEOM_VAR}={SITE_GEOM} requires the omni site generator's sub-terrains {_SITE_SUBTERRAIN_KEYS} "
            f"in that order, found {keys} (was install_omni_terrain_generator called before the knob block?)"
        )
    want = {"parkour_flat": flat_cls, "site_field_sparse": site_field_cls, "site_field_dense": site_field_cls}
    bad = []
    for name, key in zip(_SITE_KINDS, _SITE_SUBTERRAIN_KEYS):
        sub = sub_terrains[key]
        cls = want.get(key, site_cls)
        if not isinstance(sub, cls) or (cls is site_cls and getattr(sub, "kind", None) != name):
            bad.append(f"{key}={type(sub).__name__}(kind={getattr(sub, 'kind', None)!r})")
    if bad:
        raise RuntimeError(f"{GEOM_VAR}={SITE_GEOM}: wrong site sub-terrain classes / kinds: {', '.join(bad)}")


def assert_rt_sub_terrains(sub_terrains, flat_cls, site_field_cls, site_cls) -> None:
    """:func:`assert_site_sub_terrains` for the rough-terrain generator: keys exactly :data:`RT_SUBTERRAIN_KEYS` in
    order, ``parkour_flat`` a ``flat_cls``, the two legacy-field keys ``site_field_cls``, every rt hazard key a
    ``site_cls`` whose ``kind`` is its column kind."""
    if not sub_terrains:
        raise RuntimeError("terrain generator has no sub_terrains; expected the omni rough-terrain generator.")
    keys = tuple(sub_terrains)
    if keys != RT_SUBTERRAIN_KEYS:
        raise RuntimeError(
            f"{GEOM_VAR}={SITE_GEOM} with {SITE_LADDER_VAR}={RT_LADDER} requires the rough-terrain generator's "
            f"sub-terrains {RT_SUBTERRAIN_KEYS} in that order, found {keys} (was install_omni_terrain_generator called "
            "with the same environment before the knob block?)"
        )
    want = {"parkour_flat": flat_cls, "site_field_sparse": site_field_cls, "site_field_dense": site_field_cls}
    bad = []
    for name, key in zip(RT_COLUMN_KINDS, RT_SUBTERRAIN_KEYS):
        sub = sub_terrains[key]
        cls = want.get(key, site_cls)
        if not isinstance(sub, cls) or (cls is site_cls and getattr(sub, "kind", None) != name):
            bad.append(f"{key}={type(sub).__name__}(kind={getattr(sub, 'kind', None)!r})")
    if bad:
        raise RuntimeError(f"{GEOM_VAR}={SITE_GEOM} ({RT_LADDER}): wrong sub-terrain classes / kinds: {', '.join(bad)}")


def apply_site_mix(tg, mix: dict[str, int], kinds=None, keys=None) -> None:
    """``tg.num_cols = total`` and each sub-terrain's ``proportion = count / total`` in site-key order
    (zero-count keys stay declared with proportion 0). The generator's curriculum rule
    (``index / num_cols + 0.001 < cumsum(p)``) then gives each kind exactly ``count`` consecutive
    columns in canonical order (DESIGN §3.4). ``kinds`` / ``keys`` default to the site tables (the rt branch passes
    :data:`RT_COLUMN_KINDS` / :data:`RT_SUBTERRAIN_KEYS`)."""
    kinds = _SITE_KINDS if kinds is None else kinds
    keys = _SITE_SUBTERRAIN_KEYS if keys is None else keys
    total = int(sum(mix.values()))
    tg.num_cols = total
    for name, key in zip(kinds, keys):
        tg.sub_terrains[key].proportion = int(mix[name]) / total


def apply_omni_site_knobs(cfg, environ=None, classes=None, ladder_bodies=None) -> None:
    """The ``site`` branch of :func:`apply_omni_ring_knobs` (DESIGN §3.1), in this order:

    1. refuse ``KRABBY_RSI_FRAC > 0`` (RSI frames are flat-ground absolute z; every F19 stage sets 0);
    2. :func:`assert_site_sub_terrains` (``classes`` = ``(flat_cls, site_field_cls, site_cls)``, default the
       real cfg classes, imported lazily);
    3. the ``KRABBY_FIELD_*`` knobs on the legacy columns (``apply_field_geometry`` on the filtered
       ``{site_field_sparse, site_field_dense, parkour_flat}`` mapping; the pad separately, below);
    4. the pad radius (``KRABBY_FIELD_PAD_R`` or :data:`SITE_PAD_R`) on EVERY sub-terrain -- a pad above
       1.7 m is refused here (the hazard radii are laid out for it);
    5. the mix (``KRABBY_SITE_MIX`` or :data:`DEFAULT_SITE_MIX`, :func:`apply_site_mix`; a set
       ``KRABBY_FLAT_TERRAIN_FLAT_FRAC`` is superseded and reported ``ignored``);
    6. the d mapping: ``site_d_range`` (``KRABBY_SITE_DIFF`` or 0:1) and ``gen_d_range`` (the generator's
       ``difficulty_range`` as the forward block left it) into every site hazard sub-terrain; with
       ``KRABBY_SITE_LADDER`` set (PLAN_rev1 B4), also that ladder's body into every hazard sub-terrain's
       ``function`` (``ladder_bodies`` = ``{ladder: body}``, default ``site_terrains.SITE_LADDER_BODIES``,
       imported lazily); blank writes nothing (today's ``omni_site_terrain``, today's dump);
    7. the roughness cap (any sub-terrain with ``noise_range[1] > 0.03`` refused) and the ramp cap
       (:func:`site_ramp_cap_deg` of :func:`site_friction_floor`, None here: 20 deg) on every hazard sub-terrain;
    8. the parkour-term radius sync (``clear_radius_m`` = pad, ``ring_outer_radius_m`` = 4.80) and
       ``events.reset_root_state.params["spawn_level_z"] = True`` (DESIGN §3.8).

    Only reached under ``KRABBY_FLAT_TERRAIN_GEOM=site``, which is itself the armed knob: ring / field
    processes never call it.

    Rough-terrain ladder (``KRABBY_SITE_LADDER=rt``, rough-terrain PLAN step 2): the same steps on the rt generator
    (:func:`assert_rt_sub_terrains`, :data:`RT_COLUMN_KINDS` / :data:`RT_SUBTERRAIN_KEYS`), with ``KRABBY_SITE_MIX``
    REQUIRED (blank refused; :data:`RT_REFUSED_KINDS` refused by the parser), ``site_terrains.RT_BODY`` (or
    ``ladder_bodies["rt"]``) written into every rt hazard column, and an rt mix with gravel columns refused unless
    ``KRABBY_RT_FRICTION`` arms the friction contract (gravel's low friction comes only from it).
    """
    env = os.environ if environ is None else environ
    rt = parse_site_ladder(env.get(SITE_LADDER_VAR)) == RT_LADDER
    kinds, keys = (RT_COLUMN_KINDS, RT_SUBTERRAIN_KEYS) if rt else (_SITE_KINDS, _SITE_SUBTERRAIN_KEYS)
    raw_rsi = env.get(RSI_FRAC_VAR)
    if not _is_blank(raw_rsi) and _parse_float(RSI_FRAC_VAR, raw_rsi) > 0.0:
        raise RuntimeError(
            f"{GEOM_VAR}={SITE_GEOM} refuses {RSI_FRAC_VAR}={raw_rsi}: RSI frames carry flat-ground absolute z, "
            "which is wrong on raised / sunken site pads (every F19 stage sets 0)"
        )
    if classes is None:
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import (  # noqa: PLC0415
            OmniFlatTerrainCfg,
            OmniSiteFieldTerrainCfg,
            OmniSiteTerrainCfg,
        )

        flat_cls, site_field_cls, site_cls = OmniFlatTerrainCfg, OmniSiteFieldTerrainCfg, OmniSiteTerrainCfg
    else:
        flat_cls, site_field_cls, site_cls = classes

    terrain = getattr(cfg.scene, "terrain", None)
    tg = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    if tg is None:
        raise RuntimeError(
            "the omni site geometry requires scene.terrain.terrain_generator to be OMNI_SITE_TERRAINS_CFG; "
            "found None (a flat/plane terrain importer)."
        )
    subs = tg.sub_terrains
    if rt:
        assert_rt_sub_terrains(subs, flat_cls, site_field_cls, site_cls)
        hazard_keys = tuple(k for k in RT_SUBTERRAIN_KEYS
                            if k not in ("parkour_flat", "site_field_sparse", "site_field_dense"))
    else:
        assert_site_sub_terrains(subs, flat_cls, site_field_cls, site_cls)
        hazard_keys = _SITE_SUBTERRAIN_KEYS[3:]

    def _get(var, parse, **kw):
        raw = env.get(var)
        return None if _is_blank(raw) else parse(raw, **kw)

    # 3. the legacy columns' FIELD_* knobs (the pad is every column's, step 4)
    legacy = {k: subs[k] for k in ("site_field_sparse", "site_field_dense", "parkour_flat")}
    apply_field_geometry(
        legacy, site_field_cls, flat_cls,
        blocks=_get(FIELD_BLOCKS_VAR, parse_field_blocks),
        kinds=_get(FIELD_KINDS_VAR, parse_field_kinds),
        vertical_edges=_get(FIELD_EDGES_VAR, parse_field_edges),
        min_gap=_get(FIELD_MIN_GAP_VAR, parse_field_min_gap),
        size_scale=_get(FIELD_SIZE_SCALE_VAR, parse_field_size_scale),
    )
    # 4. the pad on every sub-terrain
    pad = _get(FIELD_PAD_R_VAR, parse_field_pad_r, half_width=SITE_HALF_WIDTH)
    if pad is not None and pad > SITE_PAD_R + 1e-9:
        raise ValueError(
            f"{FIELD_PAD_R_VAR}={env.get(FIELD_PAD_R_VAR)!r} is above the site pad {SITE_PAD_R} m: the site "
            "hazards start 0.3 m beyond a 1.7 m pad (DESIGN §3.3), so a larger pad would cut into them"
        )
    pad = SITE_PAD_R if pad is None else pad
    for key in keys:
        subs[key].clear_radius = float(pad)
    # 5. the mix
    raw_mix = env.get(SITE_MIX_VAR)
    mix = parse_site_mix(raw_mix, RT_LADDER if rt else None)
    if mix is None:
        if rt:
            raise ValueError(f"{SITE_MIX_VAR} is required under {SITE_LADDER_VAR}={RT_LADDER} (no default rough-terrain "
                             f"mix; the training mix is {RT_TRAIN_MIX!r})")
        mix = parse_site_mix(DEFAULT_SITE_MIX)
    else:
        print(f"[KNOB] {SITE_MIX_VAR}={raw_mix} consumed", flush=True)
    if rt and mix[RT_FRICTION_KIND] > 0 and _is_blank(env.get(RT_FRICTION_VAR)):
        raise RuntimeError(f"{SITE_MIX_VAR}={raw_mix!r} has {RT_FRICTION_KIND} columns but {RT_FRICTION_VAR} is unset: "
                           "the gravel tile's low friction comes only from the friction contract (set e.g. "
                           f"{RT_FRICTION_VAR}={RT_FRICTION_KIND}:0.2:0.5)")
    if not _is_blank(env.get(FLAT_FRAC_VAR)):
        print(f"[KNOB] {FLAT_FRAC_VAR}={env.get(FLAT_FRAC_VAR)} ignored (geometry is {SITE_GEOM}: "
              f"{SITE_MIX_VAR} sets the columns)", flush=True)
    apply_site_mix(tg, mix, kinds, keys)
    # 6. the d mapping
    raw_diff = env.get(SITE_DIFF_VAR)
    site_d = parse_site_diff(raw_diff)
    if site_d is None:
        site_d = DEFAULT_SITE_DIFF
    else:
        print(f"[KNOB] {SITE_DIFF_VAR}={raw_diff} consumed", flush=True)
    gen_d = tuple(float(v) for v in tg.difficulty_range)
    raw_ladder = env.get(SITE_LADDER_VAR)
    ladder = parse_site_ladder(raw_ladder)
    body = None
    if ladder is not None:
        if ladder_bodies is None:
            from parkour_isaaclab.terrains.omni.site_terrains import RT_BODY, SITE_LADDER_BODIES  # noqa: PLC0415

            ladder_bodies = {**SITE_LADDER_BODIES, RT_LADDER: RT_BODY}
        body = ladder_bodies[ladder]
        print(f"[KNOB] {SITE_LADDER_VAR}={raw_ladder} consumed", flush=True)
    # 7. roughness cap, ramp cap
    rough = [f"{k}={tuple(subs[k].noise_range)}" for k in keys
             if float(subs[k].noise_range[1]) > SITE_NOISE_MAX + 1e-12]
    if rough:
        raise RuntimeError(
            f"{GEOM_VAR}={SITE_GEOM}: roughness above the {SITE_NOISE_MAX} m site cap on {', '.join(rough)} "
            "(the forward MODE-light block writes (0.02, 0.02))"
        )
    refuse_unported_site_knobs(env)
    ramp_deg = site_ramp_cap_deg(site_friction_floor(env))
    for key in hazard_keys:
        sub = subs[key]
        sub.site_d_range = tuple(site_d)
        sub.gen_d_range = gen_d
        sub.ramp_max_deg = float(ramp_deg)
        if body is not None:
            sub.function = body
    # 8. the parkour-term radius sync and the pad-level spawn
    _sync_parkour_radii(cfg, subs["parkour_flat"], note="site geometry")
    events = getattr(cfg, "events", None)
    spawn = getattr(events, SPAWN_EVENT, None) if events is not None else None
    if spawn is None or getattr(spawn, "params", None) is None:
        raise RuntimeError(f"{GEOM_VAR}={SITE_GEOM} needs events.{SPAWN_EVENT} (the ring spawn) to arm spawn_level_z")
    spawn.params["spawn_level_z"] = True


# ================================================================================ ROUGH-TERRAIN FRICTION CONTRACT
# Rough-terrain PLAN step 2 (2026-11-10), the FIXED INTERFACE's friction contract. Default off: blank writes nothing.


def parse_rt_friction(text) -> dict | None:
    """``KRABBY_RT_FRICTION=gravel:LO:HI`` -> ``{"kind": "gravel", "lo": LO, "hi": HI}``; None when blank.

    ``kind`` must be :data:`RT_FRICTION_KIND` (the only low-mu column); ``RT_FRICTION_MU_MIN <= LO <= HI <=
    RT_FRICTION_MU_MAX``, both finite."""
    if _is_blank(text):
        return None
    parts = [p.strip() for p in str(text).strip().split(":")]
    if len(parts) != 3:
        raise ValueError(f'{RT_FRICTION_VAR} must be "{RT_FRICTION_KIND}:LO:HI", got {text!r}')
    if parts[0].lower() != RT_FRICTION_KIND:
        raise ValueError(f"{RT_FRICTION_VAR}: the contract's low-mu band applies to {RT_FRICTION_KIND!r} columns only, "
                         f"got {parts[0]!r} in {text!r}")
    lo, hi = (_parse_float(RT_FRICTION_VAR, part) for part in parts[1:])
    if not RT_FRICTION_MU_MIN <= lo <= hi <= RT_FRICTION_MU_MAX:
        raise ValueError(f"{RT_FRICTION_VAR} needs {RT_FRICTION_MU_MIN:g} <= LO <= HI <= {RT_FRICTION_MU_MAX:g}, "
                         f"got {text!r}")
    return {"kind": RT_FRICTION_KIND, "lo": lo, "hi": hi}


RT_KNOB_PARSERS = {RT_FRICTION_VAR: parse_rt_friction}
"""``{KRABBY_RT_*: parser}`` -- the campaign driver's round-trip table (``run_candidates.knob_parsers``)."""


def _func_leaf(func) -> str:
    name = getattr(func, "__name__", None)
    return str(name) if name else str(func).strip().rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def _set_ground_material(material, where: str) -> None:
    """Ground side of the contract: friction combine ``multiply`` at static = dynamic = 1.0 (restitution and its
    combine mode untouched)."""
    if material is None:
        raise RuntimeError(f"{RT_FRICTION_VAR} is set but the env cfg has no {where}")
    material.friction_combine_mode = "multiply"
    material.static_friction = 1.0
    material.dynamic_friction = 1.0


def apply_rt_friction_knobs(cfg, environ=None, *, exclusive_with=(), contract_func=None) -> dict | None:
    """Arm the friction contract when ``KRABBY_RT_FRICTION`` is set; unset (or blank) writes nothing, prints nothing
    and returns None.

    Called by ``omni_knobs.apply_omni_knobs`` after the payload knobs and BEFORE the stress knobs (``exclusive_with`` =
    the stress sweep's fixed-mu variable, which config modules may not spell; refusing it here means a stress mu never
    reaches the replaced event). Set, it requires ``KRABBY_FLAT_TERRAIN_GEOM=site``
    with ``KRABBY_SITE_LADDER=rt`` and TDR's material event (``events.physics_material``, function
    ``randomize_rigid_body_material``), then:

    * ``scene.terrain.physics_material`` and ``sim.physics_material``: friction combine ``multiply``, static =
      dynamic = 1.0 (PhysX takes the higher-priority combine mode of the pair, and multiply outranks the robot side's
      average), so the effective mu of a contact is the robot shape's own;
    * ``events.physics_material``: ``func`` -> ``mdp/rt_friction.rt_friction_contract`` (``contract_func`` overrides
      it for tests), params ``asset_cfg`` (kept), ``friction_range`` / ``num_buckets`` (kept: the bucket
      distribution TDR's event draws), ``gravel_range = (LO, HI)``, ``gravel_sub_terrains`` (:data:`RT_GRAVEL_KEYS`);
      mode stays ``startup``;
    * ``cfg.rt_friction`` (:data:`RT_FRICTION_ATTR`): the spec the privileged friction observation reads (it then
      reports ``2 mu_eff - 1``).
    """
    env = os.environ if environ is None else environ
    raw = env.get(RT_FRICTION_VAR)
    spec = parse_rt_friction(raw)
    if spec is None:
        return None
    geom = read_omni_geom(env)
    ladder = parse_site_ladder(env.get(SITE_LADDER_VAR))
    if geom != SITE_GEOM or ladder != RT_LADDER:
        raise RuntimeError(f"{RT_FRICTION_VAR}={raw!r} needs {GEOM_VAR}={SITE_GEOM} and {SITE_LADDER_VAR}={RT_LADDER} "
                           f"(found {geom!r} / {ladder!r}): the contract's gravel columns exist only there")
    clash = [v for v in exclusive_with if not _is_blank(env.get(v))]
    if clash:
        raise ValueError(f"{RT_FRICTION_VAR} and {clash} are exclusive: both set the robot's friction")
    events = getattr(cfg, "events", None)
    event = getattr(events, RT_MATERIAL_EVENT, None) if events is not None else None
    if event is None or getattr(event, "params", None) is None:
        raise RuntimeError(f"{RT_FRICTION_VAR} is set but events.{RT_MATERIAL_EVENT} is off (the contract replaces it)")
    if _func_leaf(event.func) != RT_MATERIAL_EVENT_FUNC:
        raise RuntimeError(f"{RT_FRICTION_VAR}: events.{RT_MATERIAL_EVENT}.func is {_func_leaf(event.func)!r}, not "
                           f"TDR's {RT_MATERIAL_EVENT_FUNC!r} (already replaced, or another material event)")
    params = event.params
    for key in ("asset_cfg", "friction_range", "num_buckets"):
        if key not in params:
            raise RuntimeError(f"{RT_FRICTION_VAR}: events.{RT_MATERIAL_EVENT}.params has no {key!r}")
    friction_range = tuple(float(v) for v in params["friction_range"])
    num_buckets = int(params["num_buckets"])
    if contract_func is None:
        from parkour_tasks.crab_hex_omni_task.mdp.rt_friction import rt_friction_contract  # noqa: PLC0415

        contract_func = rt_friction_contract
    scene = getattr(cfg, "scene", None)
    _set_ground_material(getattr(getattr(scene, "terrain", None), "physics_material", None),
                         "scene.terrain.physics_material")
    sim_material = getattr(getattr(cfg, "sim", None), "physics_material", None)
    if sim_material is not None:
        _set_ground_material(sim_material, "sim.physics_material")
    event.func = contract_func
    event.params = {
        "asset_cfg": params["asset_cfg"],
        "friction_range": friction_range,
        "num_buckets": num_buckets,
        "gravel_range": (float(spec["lo"]), float(spec["hi"])),
        "gravel_sub_terrains": tuple(RT_GRAVEL_KEYS),
    }
    setattr(cfg, RT_FRICTION_ATTR, {"gravel_range": [float(spec["lo"]), float(spec["hi"])],
                                    "friction_range": list(friction_range), "num_buckets": num_buckets,
                                    "obs": "2*mu_eff-1"})
    print(f"[KNOB] {RT_FRICTION_VAR}={str(raw).strip()} -> ground friction combine multiply at mu 1.0; "
          f"events.{RT_MATERIAL_EVENT} = rt_friction_contract (every shape one mu per env, static = dynamic: "
          f"(b + 1) / 2 of {num_buckets} buckets b ~ U{friction_range} off {RT_FRICTION_KIND}, "
          f"U({spec['lo']:g}, {spec['hi']:g}) on {RT_FRICTION_KIND}; keyed by (seed, env index)); "
          f"cfg.{RT_FRICTION_ATTR} armed (friction obs = 2 mu_eff - 1)", flush=True)
    return spec
