# SPDX-License-Identifier: BSD-3-Clause
"""The omnidirectional observation term (omni plan D2/A5). Width 1149, unchanged.

The forward term (``crab_hex_forward_task/mdp/observations.py::CrabHexParkourObservations``)
builds a 17-dim proprio head whose steering and command slots are single-axis by
construction: two steering channels that are ZEROED on flat tiles (the parkour goal heading
means nothing there), two dead command slots and one live command slot carrying
``cmd[:, 0]`` -- "speed along the one travel axis". None of that survives contact with a
task whose command is a 2-D body-frame velocity plus a yaw rate.

This subclass re-fills five of those slots and changes nothing else -- same term, same
width, same dtype, same history mechanics::

    slot 5      cmd[:, 2]                     the yaw-rate command   (was 0 * delta_yaw)
    slots 6, 7  wrap(heading_target - yaw)    the FACING error, on every tile, both channels
    slots 8, 9  cmd[:, 0], cmd[:, 1]          the body-frame xy velocity command
    slot 10     ||cmd[:, :2]||                its magnitude          (was cmd[:, 0])

THE FACING-ERROR ENCODING (``facing_obs_encoding``, knob ``KRABBY_OMNI_FACING_OBS_ENCODING``;
2026-09-18, plan Part A2). ``"raw"`` -- the SHIPPED default, bitwise what the table above says
-- writes the clipped signed angle into both channels. ``"sincos"`` writes ``sin(err)`` into
slot 6 and ``cos(err)`` into slot 7 instead: two bounded channels that are continuous across
+-pi and that stay informative past the clip, where a raw angle saturates. See
:data:`CrabHexOmniParkourObservations.facing_obs_encoding` for the interaction with
``facing_obs_clip`` (the clip applies to ``raw`` only) and ``crab_hex_omni_mirror.head_signs``
for the mirror sign tables the choice selects -- the two MUST agree, which is why the mirror
reads the encoding off this term's params rather than taking a default.

``ZERO_STEERING_HISTORY = False``: the forward task blanks dims 6-7 in the history copy so
the policy cannot read the goal heading out of its own past (it is supposed to steer from
what it sees). Here those dims ARE the command -- a facing error the policy must integrate
over time to turn smoothly -- so the history keeps them. This is a teacher-only task (no
depth student), so no distillation contract depends on the blanking.

WHY BOTH STEERING CHANNELS CARRY THE SAME NUMBER. The forward pair is (error to the current
goal, error to the NEXT goal) -- a look-ahead the ring course does not have: there is one
facing target and it does not advance. Writing it into both slots keeps the layout (and the
mirror's sign table) untouched and costs one redundant channel.

SLOT 5 AND WHY THIS CLASS OVERRIDES ``__call__``. Slots 6-10 are reached through the two
hooks the forward class exposes (``_steering_channels`` / ``_command_channels``). Slot 5 is
NOT: the forward ``__call__`` writes it inline as ``0 * delta_yaw[:, None]``, an expression
no hook yields and that cannot be re-pointed from a subclass -- ``0 * x`` is zero for every
finite ``x``, so no return value of ``_steering_channels`` can produce a yaw-rate command
there. Editing the forward file was out of scope (two lineages of record were trained
through it), so this class wraps ``__call__`` instead: it calls ``super().__call__``, then
writes slot 5 of the returned observation AND of the history row that call just pushed. The
write is a single constant column index (``YAW_RATE_OBS_SLOT``), pinned by
``tests/unit/test_crab_hex_omni_observations.py`` against the real forward ``__call__``.
The returned observation's history BLOCK needs no patching: it is the buffer as of the
previous step, which this same code patched then.
"""

from __future__ import annotations

import os

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

from parkour_tasks.crab_hex_forward_task.mdp.observations import CrabHexParkourObservations

#: The only two spellings :data:`CrabHexOmniParkourObservations.facing_obs_encoding` accepts.
#: Must stay equal to ``omni_knobs.FACING_OBS_ENCODINGS`` (the knob validates at config build,
#: this class at term construction; pinned by ``tests/unit/test_crab_hex_omni_observations.py``).
FACING_OBS_ENCODINGS = ("raw", "sincos")


def _check_facing_obs_encoding(value) -> str:
    """``value`` -> one of :data:`FACING_OBS_ENCODINGS`, or a named ``ValueError``.

    Spelled out here as well as in the knob parser because the two guard different doors: the
    knob validates what a LAUNCHER asked for, this validates what the built cfg actually
    carries (a hand-edited params dict, a recipe overlay, an eval harness). An unknown
    encoding that fell through to ``raw`` would change the observation the policy is trained
    on while every log line looked like the other one.
    """
    token = str(value).strip().lower()
    if token not in FACING_OBS_ENCODINGS:
        raise ValueError(
            f"facing_obs_encoding must be one of {FACING_OBS_ENCODINGS}, got {value!r}"
        )
    return token


def _check_omni(cfg=None) -> None:
    """Refuse to run with ``KRABBY_HEADING`` set (omni plan D2, caveat 7).

    ``KRABBY_HEADING`` is the forward/sideways turning-school knob: it widens
    ``commands.base_velocity.ranges.heading`` and switches the forward steering channels to
    the heading-command error on flat tiles. Under the omni command term BOTH of its effects
    are inert -- the omni sampler ignores ``ranges.heading`` (it samples a facing delta
    relative to the current yaw) and this class already puts the facing error on every tile.
    A run launched with it set would therefore look configured for turning and be nothing of
    the sort, which is exactly the kind of silence this package refuses.

    ``cfg`` is accepted and ignored so an env cfg's ``__post_init__`` can call
    ``_check_omni(self)`` and the observation term ``_check_omni()``.
    """
    if os.environ.get("KRABBY_HEADING") is not None:
        raise RuntimeError(
            "KRABBY_HEADING is set, but the omni task steers facing through the command "
            "term's own facing sampler (KRABBY_OMNI_FACING); ranges.heading is inert here "
            "and the steering channels already carry the facing error on every tile. Unset "
            "KRABBY_HEADING (the omni presets never export it)."
        )


class CrabHexOmniParkourObservations(CrabHexParkourObservations):
    """``CrabHexParkourObservations`` with the omnidirectional command and facing slots."""

    #: The facing error is the command: it must survive into the observation history.
    ZERO_STEERING_HISTORY = False

    #: ``obs slot -> index into ``vel_command_b``` for every command channel the policy sees.
    #: The eval harness reads this to check that an injected command really reached the
    #: policy (``command_override_max_deviation``) and to derive
    #: ``POLICY_SEES_COMMAND_CHANNEL``. Slot 10 is NOT listed: it is a norm, not a channel.
    OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}

    #: Column of the per-step proprio head that carries the yaw-rate command (see the module
    #: docstring for why it is written after the fact rather than through a hook).
    YAW_RATE_OBS_SLOT = 5

    #: Column that carries ``||cmd[:, :2]||`` (documentary: the mirror's sign table and the
    #: harness's deviation check both name it).
    CMD_NORM_OBS_SLOT = 10

    #: Optional symmetric clip (rad) on the facing-error channels 6/7, ``None`` = raw error.
    #: Set through the obs term's ``params["facing_obs_clip"]`` (knob
    #: ``KRABBY_OMNI_FACING_OBS_CLIP``). Why it exists (2026-09-17): a 1c-band head trains on
    #: |err| <= 0.6 rad, and the harness's 90-degree turn hands it 1.57 rad -- 2.6x anything it
    #: has seen -- which the third yaw smoke showed destabilises it (all 16 episodes fell inside
    #: the +pi/2-face holds while training telemetry showed +0.086 rad/s of yaw progress). The
    #: yaw-rate command already saturates beyond the dead band, so beyond the clip the policy
    #: needs only the sign; clipping keeps every out-of-band target inside the trained range.
    facing_obs_clip: float | None = None

    #: How the facing error reaches slots 6/7: ``"raw"`` (the shipped default -- one clipped
    #: signed angle in BOTH channels, bitwise what every run to date trained on) or
    #: ``"sincos"`` (slot 6 = ``sin(err)``, slot 7 = ``cos(err)``). Set through the obs term's
    #: ``params["facing_obs_encoding"]`` (knob ``KRABBY_OMNI_FACING_OBS_ENCODING``).
    #:
    #: WHY IT EXISTS (2026-09-18, plan Part A2, mechanism 3). The raw channels are clipped at
    #: 0.6 rad in every preset and the yaw-rate command saturates 0.2 rad past the dead band,
    #: so above 0.6 rad the policy is MAGNITUDE-BLIND: a half turn and a 35-degree turn look
    #: identical, and the 0.2-s observation history cannot integrate a 12-s turn. ``sincos``
    #: is bounded by construction (no out-of-distribution input, which is what the clip was
    #: introduced to prevent -- Revision 3) while staying informative over the whole circle,
    #: and it is continuous across +-pi, where the raw error jumps by 2 pi.
    #:
    #: THE CLIP APPLIES TO ``raw`` ONLY, and deliberately. ``facing_obs_clip`` exists because
    #: a raw angle 2.6x larger than anything a head trained on destabilised it; ``sin`` and
    #: ``cos`` cannot leave ``[-1, 1]``, so there is no out-of-band value to clamp, and
    #: clamping the ANGLE first would cap ``cos`` at ``cos(clip)`` = 0.825 and throw away
    #: exactly the magnitude information this encoding exists to supply -- i.e. it would make
    #: the knob silently inert on top of the presets, all of which set the clip. So under
    #: ``sincos`` the clip is IGNORED (not refused: every preset sets it, and an overlay that
    #: adds the encoding must not have to unset it). Pinned by the obs tests.
    #:
    #: THE MIRROR MUST AGREE: under ``sincos`` slot 7 stops being a yaw pseudo-scalar (a
    #: cosine is EVEN, so it does not flip under either reflection).
    #: ``crab_hex_omni_mirror.build_mirrors`` reads this very param off the env cfg and
    #: selects ``head_signs(encoding)`` accordingly -- never a default.
    facing_obs_encoding: str = "raw"

    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        _check_omni()
        clip = None
        encoding = "raw"
        params = getattr(cfg, "params", None)
        if isinstance(params, dict) and params.get("facing_obs_clip") is not None:
            clip = float(params["facing_obs_clip"])
            if not clip > 0.0:
                raise ValueError(f"facing_obs_clip must be > 0 rad, got {clip!r}")
        if isinstance(params, dict) and params.get("facing_obs_encoding") is not None:
            encoding = _check_facing_obs_encoding(params["facing_obs_encoding"])
        self.facing_obs_clip = clip
        self.facing_obs_encoding = encoding
        super().__init__(cfg, env)

    # ------------------------------------------------------------------ the two hooks
    def _steering_channels(
        self, env: ParkourManagerBasedRLEnv, on_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The FACING error ``wrap(heading_target - heading_w)``, on every tile, twice.

        ``on_flat`` is ignored on purpose (the forward term uses it to blank the parkour
        goal heading on flat tiles; there is no goal heading here). ``heading_target`` is a
        WORLD yaw written by ``CrabHexOmniCommand``; ``heading_w`` is the same quantity the
        command term's yaw-rate P-control measures against, so the observed error is exactly
        the error slot 5's command is driving to zero.

        Under ``facing_obs_encoding == "sincos"`` the pair is ``(sin(err), cos(err))``
        instead, on the UNCLIPPED error -- see the class attribute's docstring for why the
        clip does not apply there. The ``"raw"`` branch below is untouched and is what every
        shipped preset runs.
        """
        cmd_term = env.command_manager.get_term("base_velocity")
        facing_err = wrap_to_pi(cmd_term.heading_target - self.asset.data.heading_w)
        if self.facing_obs_encoding == "sincos":
            return torch.sin(facing_err), torch.cos(facing_err)
        clip = self.facing_obs_clip
        if clip is not None:
            facing_err = torch.clamp(facing_err, -clip, clip)
        return facing_err, facing_err

    def _command_channels(self, commands: torch.Tensor) -> torch.Tensor:
        """Obs dims 8-10: ``[cmd_vx_b, cmd_vy_b, ||cmd_xy||]``.

        The norm is redundant with the pair but cheap, and it is the one command channel
        that is invariant under BOTH mirror planes -- a scalar the policy can read without
        first undoing a reflection.
        """
        cmd_xy = commands[:, 0:2]
        speed = torch.linalg.norm(cmd_xy, dim=-1, keepdim=True)
        return torch.cat([cmd_xy, speed], dim=-1)

    # ------------------------------------------------------- slot 5 (no forward hook)
    def _yaw_rate_channel(self, env: ParkourManagerBasedRLEnv) -> torch.Tensor:
        """``(N,)`` yaw-rate command for obs slot 5."""
        commands = env.command_manager.get_command("base_velocity")
        return torch.nan_to_num(commands[:, 2], nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        history_length: int,
        facing_obs_clip: float | None = None,
        facing_obs_encoding: str | None = None,
    ) -> torch.Tensor:
        # The observation manager validates the term's ``params`` against THIS signature and
        # passes them as keyword arguments, so the clip must be an optional parameter here
        # (``__init__`` also reads ``cfg.params`` for the same key -- the two agree by
        # construction; a per-call value wins so a stale instance attribute cannot mask it).
        if facing_obs_clip is not None:
            clip = float(facing_obs_clip)
            if not clip > 0.0:
                raise ValueError(f"facing_obs_clip must be > 0 rad, got {clip!r}")
            self.facing_obs_clip = clip
        # Same contract for the encoding: it is a declared param (the omni obs term ships it
        # at "raw" in every preset, which is what KRABBY_OMNI_FACING_OBS_ENCODING overwrites),
        # so the manager passes it here on every step.
        if facing_obs_encoding is not None:
            self.facing_obs_encoding = _check_facing_obs_encoding(facing_obs_encoding)
        observations = super().__call__(
            env, asset_cfg, sensor_cfg, parkour_name, history_length
        )
        slot = self.YAW_RATE_OBS_SLOT
        wz = self._yaw_rate_channel(env)
        # The current step, as the policy reads it...
        observations[:, slot] = wz
        # ... and the history row ``super().__call__`` has just pushed. On the first step of
        # an episode that call REPLICATED the row across the whole history (the forward
        # ``torch.where(episode_length_buf <= 1, stack([obs_buf] * H), ...)`` branch), so
        # those rows are patched too -- otherwise a fresh episode would carry H copies of a
        # zeroed yaw-rate command.
        self._obs_history_buffer[:, -1, slot] = wz
        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            self._obs_history_buffer[fresh, :, slot] = wz[fresh].unsqueeze(-1)
        return observations
