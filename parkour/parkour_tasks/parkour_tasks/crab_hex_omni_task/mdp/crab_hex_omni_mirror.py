# SPDX-License-Identifier: BSD-3-Clause
"""Four-image symmetry augmentation for the omni task: L/R, fore-aft, and their product.

The omni robot travels in any world direction and faces any world heading, so BOTH body
mirror planes preserve the task -- unlike the forward task (travel along +x, only the
``y = 0`` plane) and the sideways task (travel along +y, only ``x = 0``). Applying both
gives the Klein four-group ``{I, M_y, M_x, M_x M_y}``; the fourth element is the elementwise
product of the other two and is the PROPER rotation ``R_z(pi)``.

What each image buys:

* ``M_y`` (y -> -y) swaps the tripod sets ({FL, MR, RL} <-> {FR, ML, RR}) -- it is the
  forward task's handedness constraint, the one the seed-basin search showed unconstrained
  training breaks at random.
* ``M_x`` (x -> -x) maps each tripod set onto ITSELF (FL<->RL, FR<->RR, mid legs fixed) and
  removes a permanent lead-ROW preference: pushing with the rear row vs pulling with the
  front row. For an omni gait that also ties forward travel to backward travel, which has no
  RSI seed and would otherwise be learned from scratch.
* ``M_x M_y = R_z(pi)`` ties every commanded direction to its opposite.

CONSTANTS ARE BORROWED, NOT RETYPED. The joint sign rules, leg swaps, scan axis and 3-vector
signs come from the two sibling modules (``crab_hex_mirror.py``, ``crab_hex_side_mirror.py``),
path-loaded exactly as the sideways module path-loads the forward one -- so a correction
there propagates here instead of silently diverging. Only the 17-dim proprio HEAD tables are
stated in this file, because the omni observation term redefines four of those slots.

HEAD LAYOUT (``CrabHexOmniParkourObservations``; only slots 5-10 differ from the forward
term's meanings)::

    0,1,2  wx, wy, wz          pseudo-vector  (w' = -M w)
    3,4    roll, pitch         pseudo-vector components
    5      cmd wz              yaw pseudo-scalar: flips under EITHER reflection
    6,7    facing error        yaw pseudo-scalar (x2)
    8,9    cmd vx_b, cmd vy_b  true vector     (v' = M v)
    10     ||cmd_xy||          scalar: invariant under both planes
    11,12  env_idx flags       scalars
    13,14  lin vx, lin vy      true vector
    15,16  clock sin/cos       -1 under M_y (tripod sets swap => schedule advanced by pi),
                               +1 under M_x (each set maps to itself)

THE FACING SLOTS DEPEND ON THE OBSERVATION'S ENCODING (2026-09-18, plan Part A2). The table
above is the ``raw`` encoding: slots 6/7 both carry the signed angle ``err = wrap(psi - yaw)``,
a yaw pseudo-scalar, hence -1 under BOTH reflections. Under ``facing_obs_encoding="sincos"``
the same two slots carry ``sin(err)`` and ``cos(err)``, and the derivation runs:

1. The raw literals say ``err -> -err`` under EITHER plane (-1 at slots 6 and 7 in both
   tables) -- read that off the shipped tables, do not re-derive it from a mental picture.
2. It is true because a reflection maps a world heading ``theta`` to ``-theta`` (``M_y``) or
   to ``pi - theta`` (``M_x``), and a DIFFERENCE of two headings negates either way:
   ``(pi - psi) - (pi - yaw) = -(psi - yaw)``.
3. ``sin`` is ODD: ``sin(-err) = -sin(err)``, so slot 6 keeps **-1 in both tables** -- the
   same literal as ``raw``.
4. ``cos`` is EVEN: ``cos(-err) = +cos(err)``, so slot 7 becomes **+1 in both tables**. That
   single entry is the whole difference between the two encodings.
5. ``R_z(pi) = M_x M_y`` maps ``err -> err`` (both headings rotate together), so slots 6 and 7
   are both +1 there -- and indeed the elementwise product gives ``(-1)(-1) = +1`` at slot 6
   and ``(+1)(+1) = +1`` at slot 7, the same value ``raw``'s product already had.

:func:`head_signs` returns the triple for an encoding; :func:`build_mirrors` resolves which
one this env asked for from the observation term's own params (:func:`_facing_obs_encoding`)
and REFUSES to guess -- a mirror that silently assumed ``raw`` under a ``sincos`` head would
teach the policy that its cosine channel flips sign under a reflection it does not flip
under, which is invisible in every training curve.

Numerically ``HEAD_SIGNS_LR`` coincides with the forward table and ``HEAD_SIGNS_FA`` differs
from the sideways table at slot 8 alone -- see ``tests/unit/test_crab_hex_omni_mirror.py``,
which pins both difference sets. The coincidences are NOT accidents to rely on: the forward
head's slots 5, 8 and 9 are structurally zero (``0 * delta_yaw``, ``0 * commands[:, 0:2]``),
so their signs there were free; here all three carry live numbers and the values above are
derived from the frame rules, not inherited.

THE SCANNER MUST BE CENTRED. ``M_x`` flips the ray index ``ix`` and ``M_y`` flips ``iy``;
both are symmetries of the grid only if the scanner's body-frame offset lies IN the plane.
The forward scanner sits 0.6 m ahead (``pos=(0.6, 0.0, 20.0)``), which the fore-aft plane
does not contain -- hence the omni scene's ``pos=(0.0, 0.0, 20.0)``
(:data:`REQUIRED_SCANNER_OFFSET_XY`, plan D7).

ENTRY POINT. ``crab_hex_omni_symmetry_augmentation`` is the rsl-rl ``data_augmentation_func``
(``PPOWithExtractor``): ``f(obs, actions, env, obs_type) -> (obs_aug, act_aug)`` where each
output is ``cat([x, M_y x, M_x x, M_x M_y x])`` along the batch dim. The fork derives
``num_aug`` from ``obs.shape[0] / original_batch_size`` (4 here) and pairs obs and action
rows POSITIONALLY -- ``mse(mean_actions[B:], augmented_action_means[B:])`` -- so the two
outputs must list the images in the same order. Either input may be None; an actions-only
call requires a previous obs call on the same env (the maps are sized from the live obs dim).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_TASKS_DIR = Path(__file__).resolve().parents[2]
_FWD_MODULE_NAME = "crab_hex_mirror__forward"
_FWD_MIRROR_PY = _TASKS_DIR / "crab_hex_forward_task" / "mdp" / "crab_hex_mirror.py"
_SIDE_MODULE_NAME = "crab_hex_side_mirror__foreaft"
_SIDE_MIRROR_PY = _TASKS_DIR / "crab_hex_sideways_task" / "mdp" / "crab_hex_side_mirror.py"


def _load_by_path(path: Path, name: str):
    """Load a module by file path under ``name`` (never through ``parkour_tasks.*``)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for a real file
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_once(path: Path, name: str):
    mod = sys.modules.get(name)
    return mod if mod is not None else _load_by_path(path, name)


#: The forward (L/R) mirror module: ``CrabHexMirror``, ``_resolve_layout``, the L/R constants.
_fwd = _load_once(_FWD_MIRROR_PY, _FWD_MODULE_NAME)
#: The sideways (fore-aft) mirror module: the ``x = 0`` plane's joint/leg/vec3 constants.
_side = _load_once(_SIDE_MIRROR_PY, _SIDE_MODULE_NAME)


# ------------------------------------------------------------------------- the two planes
#: ``M_y``: y -> -y. Same table as the forward task's (the omni head's live slots 5/8/9/10
#: happen to want the signs the forward head's dead slots carried); derived, not inherited.
HEAD_SIGNS_LR = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
                 1.0, -1.0, -1.0, -1.0]

#: ``M_x``: x -> -x. The sideways table with slot 8 flipped: there it was a dead zero, here
#: it is the x component of the body-frame velocity command and must negate.
HEAD_SIGNS_FA = [1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
                 -1.0, 1.0, 1.0, 1.0]

N_HEAD = len(HEAD_SIGNS_LR)

#: The composed image ``M_x M_y = R_z(pi)``: elementwise product of the two tables.
#: (Documentary -- the augmentation applies the two maps in sequence rather than this.)
#: Like the two above, this is the ``raw`` encoding's table; :func:`head_signs` returns the
#: per-encoding triple. It happens to be encoding-INDEPENDENT (slot 7 is +1 either way,
#: because the two tables' entries flip together), which is the derivation's own check.
HEAD_SIGNS_ROT_Z_PI = [a * b for a, b in zip(HEAD_SIGNS_LR, HEAD_SIGNS_FA)]

#: L/R plane constants -- the forward module's, verbatim.
LEG_SWAP_LR = _fwd._LEG_SWAP
JOINT_SIGN_RULES_LR = _fwd._JOINT_SIGN_RULES
SCAN_AXIS_LR = "y"
VEC3_SIGNS_LR = (1.0, -1.0, 1.0)

#: Fore-aft plane constants -- the sideways module's, verbatim (knee +1: the pitch axes lie
#: along the plane normal and the F/R rows carry no relative USD flip).
LEG_SWAP_FA = _side.LEG_SWAP
JOINT_SIGN_RULES_FA = _side.JOINT_SIGN_RULES
SCAN_AXIS_FA = "x"
VEC3_SIGNS_FA = _side.VEC3_SIGNS

#: Body-frame xy the height scanner's offset MUST have for both flips to be symmetries.
REQUIRED_SCANNER_OFFSET_XY = (0.0, 0.0)

#: The four images, in the order both outputs list them.
IMAGE_NAMES = ("identity", "M_y", "M_x", "M_x M_y")

#: The two facing-error head slots, and the encodings :func:`head_signs` knows.
FACING_SIN_SLOT = 6
FACING_COS_SLOT = 7
FACING_OBS_ENCODINGS = ("raw", "sincos")

#: The observation term the encoding is read off, as the omni env cfg declares it.
OBS_GROUP_ATTR = "policy"
OBS_TERM_ATTR = "extreme_parkour_observations"
OBS_ENCODING_PARAM = "facing_obs_encoding"


def head_signs(encoding: str = "raw") -> tuple[list, list, list]:
    """``(lr, fa, rot_z_pi)`` head sign tables for a facing-error ``encoding``.

    ``"raw"`` returns copies of the module constants verbatim. ``"sincos"`` returns copies
    with slot 7 (``cos(err)``, an EVEN function of a quantity both reflections negate) flipped
    to +1 in BOTH plane tables; slot 6 (``sin(err)``, odd) keeps the raw -1. The composed
    table is recomputed as the elementwise product in both cases, so it is +1 at slot 7 either
    way -- see the module docstring's five-step derivation.

    Raises:
        ValueError: for any other spelling. The mirror must never fall back to a default: the
            wrong table is a silent, plausible-looking training corruption.
    """
    token = str(encoding).strip().lower()
    if token not in FACING_OBS_ENCODINGS:
        raise ValueError(
            f"crab_hex_omni_mirror.head_signs: unknown facing observation encoding "
            f"{encoding!r}; expected one of {FACING_OBS_ENCODINGS}."
        )
    lr = list(HEAD_SIGNS_LR)
    fa = list(HEAD_SIGNS_FA)
    if token == "sincos":
        lr[FACING_COS_SLOT] = 1.0
        fa[FACING_COS_SLOT] = 1.0
    return lr, fa, [a * b for a, b in zip(lr, fa)]


def _facing_obs_encoding(env) -> str:
    """The facing encoding this env's OBSERVATION TERM declares, or a named ``ValueError``.

    ``env.cfg.observations.policy.extreme_parkour_observations.params["facing_obs_encoding"]``,
    defaulting to ``"raw"`` only when the params dict is genuinely reachable and simply does
    not carry the key (an older cfg, or a hand-built term) -- never when the path itself
    cannot be walked. A mirror built against the wrong encoding flips the sign of a channel
    that does not flip, which no training curve shows; so an unresolvable cfg is a refusal,
    not a default.
    """
    uenv = getattr(env, "unwrapped", env)
    cfg = getattr(uenv, "cfg", None)
    group = getattr(getattr(cfg, "observations", None), OBS_GROUP_ATTR, None)
    term = getattr(group, OBS_TERM_ATTR, None)
    params = getattr(term, "params", None)
    if not isinstance(params, dict):
        raise ValueError(
            "crab_hex_omni_mirror: cannot resolve "
            f"env.cfg.observations.{OBS_GROUP_ATTR}.{OBS_TERM_ATTR}.params on "
            f"{type(uenv).__name__}, so the facing-error encoding the head was built with is "
            "unknown and the mirror sign tables cannot be chosen. (Refusing rather than "
            f"assuming {FACING_OBS_ENCODINGS[0]!r}: under 'sincos' slot "
            f"{FACING_COS_SLOT} does NOT flip under either reflection, and a mirror that got "
            "that wrong would train a symmetry the observation does not have.)"
        )
    return str(params.get(OBS_ENCODING_PARAM, FACING_OBS_ENCODINGS[0])).strip().lower()


def build_mirrors(env, obs_dim: int) -> tuple[object, object]:
    """``(lr, fa)`` -- the two ``CrabHexMirror`` maps for this env's resolved layout.

    One ``_resolve_layout`` call feeds both: the name lists, scan grid and history length
    are properties of the observation, not of the mirror plane. The head sign tables come from
    :func:`head_signs` at the encoding :func:`_facing_obs_encoding` reads off the env's own
    observation term -- the mirror and the observation are two halves of one contract.
    """
    lr_signs, fa_signs, _ = head_signs(_facing_obs_encoding(env))
    layout = _fwd._resolve_layout(env, obs_dim, n_head=N_HEAD)
    lr = _fwd.CrabHexMirror(
        **layout,
        head_signs=lr_signs,
        joint_sign_rules=JOINT_SIGN_RULES_LR,
        leg_swap=LEG_SWAP_LR,
        scan_axis=SCAN_AXIS_LR,
        vec3_signs=VEC3_SIGNS_LR,
    )
    fa = _fwd.CrabHexMirror(
        **layout,
        head_signs=fa_signs,
        joint_sign_rules=JOINT_SIGN_RULES_FA,
        leg_swap=LEG_SWAP_FA,
        scan_axis=SCAN_AXIS_FA,
        vec3_signs=VEC3_SIGNS_FA,
    )
    return lr, fa


#: Own cache (each sibling module keeps its own): one pair of mirrors per env instance.
_CACHE: dict[int, tuple] = {}


def _get_mirrors(env, obs_dim: int) -> tuple:
    key = id(env)
    if key not in _CACHE:
        _CACHE[key] = build_mirrors(env, obs_dim)
    return _CACHE[key]


def four_images_obs(obs: torch.Tensor, lr, fa) -> torch.Tensor:
    """``cat([x, M_y x, M_x x, M_x M_y x])`` along the batch dim."""
    m_y = lr.mirror_obs(obs)
    m_x = fa.mirror_obs(obs)
    m_xy = fa.mirror_obs(m_y)
    return torch.cat([obs, m_y, m_x, m_xy], dim=0)


def four_images_actions(actions: torch.Tensor, lr, fa) -> torch.Tensor:
    """``cat([a, M_y a, M_x a, M_x M_y a])`` -- the SAME image order as the obs."""
    m_y = lr.mirror_actions(actions)
    m_x = fa.mirror_actions(actions)
    m_xy = fa.mirror_actions(m_y)
    return torch.cat([actions, m_y, m_x, m_xy], dim=0)


def crab_hex_omni_symmetry_augmentation(obs=None, actions=None, env=None, obs_type="policy"):
    """rsl-rl symmetry entry point: four images of obs and/or actions, same order."""
    if obs is None and id(env) not in _CACHE:
        raise RuntimeError(
            "the symmetry augmentation entry point must see an obs batch before an "
            "actions-only call (the mirrors are sized from the live obs dim)"
        )
    lr, fa = _get_mirrors(env, obs.shape[-1] if obs is not None else 0)
    obs_out = None
    act_out = None
    if obs is not None:
        if obs.shape[-1] != lr.obs_dim:
            raise ValueError(
                f"symmetry augmentation: {obs_type} obs dim {obs.shape[-1]} != policy dim "
                f"{lr.obs_dim}; extend the map before using a distinct critic space"
            )
        obs_out = four_images_obs(obs, lr, fa)
    if actions is not None:
        act_out = four_images_actions(actions, lr, fa)
    return obs_out, act_out
