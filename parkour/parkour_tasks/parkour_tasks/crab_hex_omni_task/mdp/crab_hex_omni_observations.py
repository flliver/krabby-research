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

TWO OPTIONAL HEAD EXTENSIONS (motion-quality campaign 2026-09-22, DEFAULT OFF; both change the
observation WIDTH, so a head trained without them cannot load with them -- from-scratch confirm
runs only). Each is armed by a param the shipped cfg does not carry, and is appended to the
per-step proprio head right after the clock (dims 17+), in this order:

    ``stand_obs``  (``KRABBY_OMNI_STAND_OBS=1``)  1 dim: the STAND-slot bit
                   (``command.stand_mask()``); while it is set the clock sin/cos (dims 15-16)
                   read 0 -- a standing robot has no gait phase to track.
    ``lpf_obs``    (``KRABBY_ACTION_LPF_OBS=1``) 12 dims: the action term's low-passed rod-length
                   command (``rod_cmd_filtered``: hip x6 | knee x6, leg order, [-1, 1] per
                   stroke window). Requires ``KRABBY_ACTION_LPF_HZ``.

``crab_hex_omni_mirror.build_mirrors`` reads the same two params off the env cfg and extends its
head tables to match (the stand bit is invariant under both planes; the rod channels are leg
scalars, permuted with the plane's leg swap and never negated).

THE SCALAR COMMAND OBSERVATION (``cmd_obs_scalar``, knob ``KRABBY_OMNI_CMD_OBS=scalar``;
side-speed-bisection plan step 1b, 2026-09-28, DEFAULT OFF, WIDTH-PRESERVING). Zeroes ONLY obs
slots 8-9 -- the two dead-reckoned components of :meth:`_command_channels`'s ``cmd_xy`` -- so the
command's travel DIRECTION never reaches the policy; slot 10 (``||cmd_xy||``) keeps the command's
MAGNITUDE, and slots 5-7 (the yaw-rate command, the facing error) are untouched (kept on purpose:
they carry no travel direction, and zeroing them would make the policy heading-blind under the
world-held relay, a confound the sideways head this plan compares against never had -- plan audit
pass 1). This is arm C's "direction-blind" observation: paired with
``KRABBY_OMNI_MIRROR_IMAGES=fa`` (a from-scratch, +y-only-sampled run), the policy has no channel
left from which to infer -y even indirectly through the mirror loss.

THE HISTORY BUFFER NEEDS NO SEPARATE PATCH (unlike slot 5, see below). ``_command_channels`` is
one of the hooks ``super().__call__`` folds into ``obs_buf`` BEFORE ``obs_buf`` is pushed onto
``_obs_history_buffer`` (the ``torch.stack`` / ``torch.cat`` near the end of that method) -- so a
hook's own output is what both the CURRENT step's ``observations`` and every future history row
carry, with no extra step. Zeroing slots 8-9 inside :meth:`_command_channels` therefore zeroes
them in the live observation and in the history it seeds, consistently, by construction. (Slot 5
needed its own post-hoc patch of both ``observations`` and ``_obs_history_buffer`` -- see "SLOT 5
AND WHY THIS CLASS OVERRIDES ``__call__``" below -- precisely because the forward ``__call__``
writes it inline as ``0 * delta_yaw``, an expression no hook's return value can reach; slots 8-9
have no such problem, since the whole 3-wide command block IS a hook's return value.)

THE CLOCK-HIDE KNOB (clock-deprecation plan step 2, 2026-09-23; DEFAULT OFF, WIDTH-PRESERVING).
``params["clock_hide"]`` (``KRABBY_OMNI_CLOCK_HIDE=1``) makes :meth:`_clock_channels` return
(0, 0) on every row -- the deployment value, mirror-invariant, so a formed head keeps loading
at 1149 wide and the history copies / critic follow through ``obs_buf``. With
``params["clock_hide_ramp"] = (t0, t1)`` (env steps; refused without ``clock_hide``) the mask is
per EPISODE instead: :meth:`reset` latches ``hidden ~ Bernoulli(p(t))`` once per reset, with
``p`` the cosine ``curriculums.ramp_value(t, 0, 1, t0, t1, eps=0)`` on
``env.common_step_counter``, and every row is hidden from ``t1`` on. Either way the term logs
``Metrics/clock_hidden_frac`` into ``env.extras["log"]`` at each reset (1.0 without a ramp),
which the campaign's removal verification reads. It composes with ``stand_obs`` (both zero).

THE STAND BIT IN THE HIDDEN CLOCK COLUMN (``params["stand_bit_clock"]``, ``KRABBY_OMNI_STAND_OBS=clock``;
omni phase-2 stage p2r, ``P2R_STATUE_PLAN.md`` step 1, 2026-10-09; DEFAULT OFF, WIDTH-PRESERVING). Under
the clock-hide knob dims 15-16 read (0, 0) on every row, so a head trained clock-free (S1) has an unused
input column. This param writes ``command.stand_mask()`` (the command term's LATCHED stand slot) into
column :attr:`CrabHexOmniParkourObservations.STAND_BIT_CLOCK_COL` (15) of the CURRENT step's
observation only -- after ``super().__call__`` has pushed this step's row onto the history buffer, so
history column 15 stays 0 exactly as in S1 (the slot-5 history re-patch is deliberately NOT copied).
Construction-time only, and refused unless ``clock_hide`` is set with no ramp (a visible or ramped
clock would share the column) and unless ``stand_obs`` is off (the two are alternative stand bits).
The single observation group feeds both actor and critic, so the critic sees the same current-step
bit. ``crab_hex_omni_mirror.build_mirrors`` reads the same param and makes column 15 mirror-invariant.

THE SAMPLE-AND-HOLD CONTACT PROXY ON STAND ROWS (``params["stand_proxy_hold"]``,
``KRABBY_OMNI_STAND_PROXY=hold``; omni phase-2 stage p2r Z2, ``P2R_Z2_PLAN.md`` step 1, 2026-10-10; DEFAULT OFF,
WIDTH-PRESERVING). The current-sense proxy (columns 83-88, ``_get_contact_fill``) is gated by rod speed, so a
parked leg reads -0.5 whatever it carries and a still statue is blind to its loads. This param keeps a per-env,
per-leg HELD buffer (initialised and reset to -0.5, "unknown"): EVERY step, on every row, a leg's held value
becomes its live proxy (columns 83-88 as ``super().__call__`` produced them) wherever that leg's KNEE rod gate is
full -- ``|knee_rod_speed| / (_CURRENT_SENSE_GATE_FRACTION * KNEE_SPEED_M_S) >= 1``, the knee-inclusive scale
(user decision 2026-10-10: a hip-only reading is never held). Then, on STAND rows (``command.stand_mask()``, the
same latched slot as the column-15 bit), the CURRENT step's columns 83-88 read the held values; the history rows
stay the raw live proxy (the push happens inside ``super().__call__`` before the overwrite, exactly as for column
15), and walking rows are untouched. At each reset the term logs ``Metrics/stand_proxy_live_frac`` into
``env.extras["log"]``: the share of (stand row, leg) entries since the last log whose held value equals the live
one bitwise. Construction-time only; refused without ``stand_bit_clock`` (the stand rows are the bit's) and with
``lpf_obs`` (the appended head would move the proxy off columns 83-88 (the first, load-sensing R read them; the cams-only R of 2026-10-10 reads none)).

THE STAND-ROW INPUT MASK (``params["stand_mask"]``, ``KRABBY_OMNI_STAND_MASK=1``; omni phase-2 stage p2r Z2d,
``P2R_Z2D_PLAN.md`` step 1, 2026-10-11; DEFAULT OFF, WIDTH-PRESERVING). NOTE(p2r Z2d, 2026-10-11): on STAND rows
(``command.stand_mask()``, the latched slot the column-15 bit carries) the RETURNED observation's last-action columns
65-82 read 0 and its proxy columns 83-88 read -0.5 ("parked"), in the current step and in each of the 10 real
history frames of the returned history block -- ``crab_hex_omni_restance.stand_mask_obs``, the one function the
cloning bank loader also applies. Applied after the stand-bit / held-proxy patches; ``_obs_history_buffer`` is
never written (the stored history keeps the raw rows), walking rows are untouched. At each reset the term logs
``Metrics/stand_mask_frac`` into ``env.extras["log"]``: the share of rows masked since the last log (nothing while
no row was seen). Construction-time only; refused without ``stand_bit_clock``, with ``stand_proxy_hold`` (both write
columns 83-88) and with ``lpf_obs`` (the appended head would shift the layout the mask writes).

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

from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage
from parkour_tasks.crab_hex_forward_task.mdp.curriculums import ramp_value
from parkour_tasks.crab_hex_forward_task.mdp.observations import (
    _CURRENT_SENSE_GATE_FRACTION,
    CrabHexParkourObservations,
)
from parkour_tasks.crab_hex_omni_task.mdp import crab_hex_omni_restance as restance

#: The only two spellings :data:`CrabHexOmniParkourObservations.facing_obs_encoding` accepts.
#: Must stay equal to ``omni_knobs.FACING_OBS_ENCODINGS`` (the knob validates at config build,
#: this class at term construction; pinned by ``tests/unit/test_crab_hex_omni_observations.py``).
FACING_OBS_ENCODINGS = ("raw", "sincos")

#: Widths of the two optional head extensions (see the module docstring).
STAND_OBS_DIM = 1
LPF_OBS_DIM = 12


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

    #: True zeroes obs slots 8-9 (the command's body-frame x/y) while keeping slot 10
    #: (``||cmd_xy||``); slots 5-7 are unaffected. Set through the obs term's
    #: ``params["cmd_obs_scalar"]`` (knob ``KRABBY_OMNI_CMD_OBS=scalar``). Does not change the
    #: observation WIDTH, so -- unlike ``stand_obs`` / ``lpf_obs`` -- it is not construction-locked
    #: and may vary call to call, like ``facing_obs_clip`` / ``facing_obs_encoding``.
    cmd_obs_scalar: bool = False

    #: NOTE(clock-deprecation step 2, 2026-09-23): the training-log key the clock-hide knob
    #: writes at every reset -- the fraction of rows whose clock channels read (0, 0).
    CLOCK_HIDDEN_FRAC_KEY = "Metrics/clock_hidden_frac"

    #: NOTE(p2r statue, 2026-10-09): the column ``stand_bit_clock`` writes the stand bit into -- the
    #: clock-sin channel, which reads 0 under ``clock_hide``. A class attribute (not a module
    #: constant) so the unit tests that exec this class from source need no extra name.
    STAND_BIT_CLOCK_COL = 15

    #: Off unless ``params["stand_bit_clock"]`` (``KRABBY_OMNI_STAND_OBS=clock``); see the module docstring.
    stand_bit_clock: bool = False

    #: NOTE(p2r Z2, 2026-10-10): the first of the six current-sense proxy columns of the CURRENT step
    #: (``17 + 2 * 24 joints + 18 actions``; ``crab_hex_omni_restance.PROXY_OBS_COLS``, pinned by the tests).
    STAND_PROXY_COL = 83
    #: The proxy's value for "unknown" (a parked rod's reading): the held buffer's init and reset value.
    STAND_PROXY_UNKNOWN = -0.5
    #: The training-log key the held proxy writes at every reset (see the module docstring).
    STAND_PROXY_LIVE_FRAC_KEY = "Metrics/stand_proxy_live_frac"

    #: Off unless ``params["stand_proxy_hold"]`` (``KRABBY_OMNI_STAND_PROXY=hold``); see the module docstring.
    stand_proxy_hold: bool = False

    #: NOTE(p2r Z2d, 2026-10-11): the training-log key the stand-row mask writes at every reset (the share of rows
    #: masked since the last log; see the module docstring).
    STAND_MASK_FRAC_KEY = "Metrics/stand_mask_frac"

    #: Off unless ``params["stand_mask"]`` (``KRABBY_OMNI_STAND_MASK=1``); see the module docstring.
    stand_mask: bool = False

    @staticmethod
    def _check_clock_hide_ramp(ramp, clock_hide: bool) -> tuple[int, int] | None:
        """``None`` or ``(t0, t1)`` env steps, ``0 <= t0 < t1``; a ramp without the hide flag raises.

        The knob parser refuses the same things at config build; this guards what the built cfg
        actually carries (a hand-edited params dict, an overlay), because a ramp the term then
        ignored would train the full clock while the recipe claimed a removal schedule.
        """
        if ramp is None:
            return None
        if not clock_hide:
            raise ValueError("clock_hide_ramp is set but clock_hide is not: the ramp only schedules "
                             "when the hidden clock applies")
        try:
            t0, t1 = ramp
        except (TypeError, ValueError) as exc:
            raise ValueError(f"clock_hide_ramp must be a (t0, t1) pair of env steps, got {ramp!r}") from exc
        if int(t0) != t0 or int(t1) != t1 or not 0 <= int(t0) < int(t1):
            raise ValueError(f"clock_hide_ramp needs integer env steps 0 <= t0 < t1, got {ramp!r}")
        return (int(t0), int(t1))

    @staticmethod
    def _check_stand_bit_clock(stand_bit_clock: bool, clock_hide: bool, ramp, stand_obs: bool) -> None:
        """``stand_bit_clock`` needs the clock hidden on EVERY row (``clock_hide``, no ramp) and no
        ``stand_obs``; the knob parser refuses the same at config build, this guards the built cfg."""
        if not stand_bit_clock:
            return
        if not clock_hide or ramp is not None:
            raise ValueError("stand_bit_clock writes the stand bit into the clock column, which is free only "
                             f"while the clock is hidden on every row (clock_hide=True, no ramp); got "
                             f"clock_hide={clock_hide}, clock_hide_ramp={ramp}")
        if stand_obs:
            raise ValueError("stand_bit_clock and stand_obs are two alternative stand bits; set one")

    @staticmethod
    def _check_stand_proxy_hold(stand_proxy_hold: bool, stand_bit_clock: bool, lpf_obs: bool) -> None:
        """NOTE(p2r Z2, 2026-10-10): ``stand_proxy_hold`` needs ``stand_bit_clock`` (its stand rows are the bit's)
        and no ``lpf_obs`` (the appended head would move the proxy off columns 83-88); the knob parser refuses
        the same at config build, this guards the built cfg."""
        if not stand_proxy_hold:
            return
        if not stand_bit_clock:
            raise ValueError("stand_proxy_hold holds the contact proxy on the stand-bit rows, which needs "
                             "stand_bit_clock (KRABBY_OMNI_STAND_OBS=clock)")
        if lpf_obs:
            raise ValueError("stand_proxy_hold and lpf_obs: the appended filtered-rod head would move the proxy "
                             "off columns 83-88, which the re-stance controller reads")

    @staticmethod
    def _check_stand_mask(stand_mask: bool, stand_bit_clock: bool, stand_proxy_hold: bool, lpf_obs: bool) -> None:
        """NOTE(p2r Z2d, 2026-10-11): ``stand_mask`` needs ``stand_bit_clock`` (its stand rows are the bit's), and
        refuses ``stand_proxy_hold`` (both write columns 83-88) and ``lpf_obs`` (the appended head would shift the
        layout the mask writes); the knob parser refuses the same at config build, this guards the built cfg."""
        if not stand_mask:
            return
        if not stand_bit_clock:
            raise ValueError("stand_mask masks the stand-bit rows, which needs stand_bit_clock "
                             "(KRABBY_OMNI_STAND_OBS=clock)")
        if stand_proxy_hold:
            raise ValueError("stand_mask and stand_proxy_hold both write the proxy columns 83-88 on stand rows; "
                             "set one")
        if lpf_obs:
            raise ValueError("stand_mask and lpf_obs: the appended filtered-rod head would shift the obs layout "
                             "the mask writes (columns 65-88 and the history frames)")

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
        self.cmd_obs_scalar = bool(params.get("cmd_obs_scalar", False)) if isinstance(params, dict) else False
        # Read BEFORE super().__init__: the forward __init__ sizes the history buffer from
        # ``_extra_head_dim()``, which reads these two.
        self.stand_obs = bool(params.get("stand_obs", False)) if isinstance(params, dict) else False
        self.lpf_obs = bool(params.get("lpf_obs", False)) if isinstance(params, dict) else False
        # NOTE(clock-deprecation step 2, 2026-09-23): the clock-hide knob (width unchanged).
        self.clock_hide = bool(params.get("clock_hide", False)) if isinstance(params, dict) else False
        ramp = params.get("clock_hide_ramp") if isinstance(params, dict) else None
        self.clock_hide_ramp = (
            None if ramp is None else self._check_clock_hide_ramp(ramp, self.clock_hide)
        )
        # NOTE(p2r statue, 2026-10-09): the stand bit in the hidden clock column (width unchanged).
        self.stand_bit_clock = bool(params.get("stand_bit_clock", False)) if isinstance(params, dict) else False
        self._check_stand_bit_clock(self.stand_bit_clock, self.clock_hide, self.clock_hide_ramp, self.stand_obs)
        # NOTE(p2r Z2, 2026-10-10): the sample-and-hold proxy on stand rows (width unchanged).
        self.stand_proxy_hold = (
            bool(params.get("stand_proxy_hold", False)) if isinstance(params, dict) else False
        )
        self._check_stand_proxy_hold(self.stand_proxy_hold, self.stand_bit_clock, self.lpf_obs)
        # NOTE(p2r Z2d, 2026-10-11): the stand-row input mask (width unchanged).
        self.stand_mask = bool(params.get("stand_mask", False)) if isinstance(params, dict) else False
        self._check_stand_mask(self.stand_mask, self.stand_bit_clock, self.stand_proxy_hold, self.lpf_obs)
        super().__init__(cfg, env)
        # The per-episode latch (ramp mode). All-False before the first reset is the draw at
        # t = 0, where p(0) = 0 for every legal t0 >= 0; rows are forced hidden from t1 on anyway.
        self._clock_hidden = (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if self.clock_hide else None
        )
        # The held proxy (N, 6) and the live-share counters since the last log (device tensors: no sync per
        # step). None while the knob is off, so nothing below touches a knob-off term.
        self._stand_proxy_held = None
        self._stand_proxy_n_live = self._stand_proxy_n_total = None
        if self.stand_proxy_hold:
            self._stand_proxy_held = torch.full((self.num_envs, 6), self.STAND_PROXY_UNKNOWN, device=self.device)
            self._stand_proxy_n_live = torch.zeros((), dtype=torch.long, device=self.device)
            self._stand_proxy_n_total = torch.zeros((), dtype=torch.long, device=self.device)
        # NOTE(p2r Z2d, 2026-10-11): the masked-row counters since the last log (device tensors: no sync per step);
        # None while the knob is off.
        self._stand_mask_n_masked = self._stand_mask_n_total = None
        if self.stand_mask:
            self._stand_mask_n_masked = torch.zeros((), dtype=torch.long, device=self.device)
            self._stand_mask_n_total = torch.zeros((), dtype=torch.long, device=self.device)

    # ------------------------------------------------ the optional head extensions (default off)
    def _extra_head_dim(self) -> int:
        return (STAND_OBS_DIM if self.stand_obs else 0) + (LPF_OBS_DIM if self.lpf_obs else 0)

    def _stand_bit(self, env: ParkourManagerBasedRLEnv) -> torch.Tensor:
        """``(N,)`` float: 1.0 on the STAND-slot rows (the command term's latched slot)."""
        return env.command_manager.get_term("base_velocity").stand_mask().float()

    def _extra_head_channels(self, env: ParkourManagerBasedRLEnv) -> torch.Tensor | None:
        parts = []
        if self.stand_obs:
            parts.append(self._stand_bit(env)[:, None])
        if self.lpf_obs:
            parts.append(env.action_manager.get_term("joint_pos").rod_cmd_filtered)
        return torch.cat(parts, dim=-1) if parts else None

    def _clock_channels(self, env: ParkourManagerBasedRLEnv, clock_phase: torch.Tensor):
        sin_c, cos_c = super()._clock_channels(env, clock_phase)
        if self.clock_hide:
            return self._hide_clock(env, *self._stand_clock(env, sin_c, cos_c))
        return self._stand_clock(env, sin_c, cos_c)

    def _stand_clock(self, env: ParkourManagerBasedRLEnv, sin_c, cos_c):
        if not self.stand_obs:
            return sin_c, cos_c
        moving = (1.0 - self._stand_bit(env))[:, None]
        return sin_c * moving, cos_c * moving

    # -------------------------------------------- the clock-hide knob (default off, 2026-09-23)
    def _clock_hidden_rows(self, env: ParkourManagerBasedRLEnv) -> torch.Tensor | None:
        """``(N,)`` bool of rows whose clock reads (0, 0); ``None`` = every row (no ramp)."""
        if self.clock_hide_ramp is None:
            return None
        if int(env.common_step_counter) >= self.clock_hide_ramp[1]:
            return torch.ones_like(self._clock_hidden)
        return self._clock_hidden

    def _hide_clock(self, env: ParkourManagerBasedRLEnv, sin_c, cos_c):
        """Exact zeros (never ``-0.0``) on the hidden rows; the visible ones pass through."""
        hidden = self._clock_hidden_rows(env)
        if hidden is None:
            return torch.zeros_like(sin_c), torch.zeros_like(cos_c)
        mask = hidden[:, None]
        return (torch.where(mask, torch.zeros_like(sin_c), sin_c),
                torch.where(mask, torch.zeros_like(cos_c), cos_c))

    def reset(self, env_ids=None) -> None:
        """The base reset, then (ramp mode) one Bernoulli(p(t)) latch per reset row, and the log.

        ``p(t) = ramp_value(t, 0, 1, t0, t1, eps=0)``: ``eps=0`` because the reward-weight default
        (1e-3) would hide 0.1 % of episodes BEFORE ``t0``. Knob off: the base reset, nothing else.
        """
        super().reset(env_ids)
        if self.stand_proxy_hold:
            self._reset_stand_proxy(env_ids)
        if self.stand_mask:
            self._log_stand_mask()
        if not self.clock_hide:
            return
        env = self._env
        if self.clock_hide_ramp is not None:
            t0, t1 = self.clock_hide_ramp
            p = ramp_value(int(env.common_step_counter), 0.0, 1.0, t0, t1, eps=0.0)
            if env_ids is None:
                ids = torch.arange(self.num_envs, device=self.device)
            else:
                ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
            self._clock_hidden[ids] = torch.rand(ids.numel(), device=self.device) < p
            frac = self._clock_hidden_rows(env).float().mean()
        else:
            frac = torch.ones((), device=self.device)
        extras = getattr(env, "extras", None)
        if isinstance(extras, dict):
            extras.setdefault("log", {})[self.CLOCK_HIDDEN_FRAC_KEY] = frac

    # ------------------------------- the sample-and-hold proxy on stand rows (p2r Z2, 2026-10-10)
    def _reset_stand_proxy(self, env_ids=None) -> None:
        """Reset rows read "unknown" (-0.5) again; log the live share since the last log, then restart it.

        Nothing is logged while no (stand row, leg) entry was seen since the last log (the runner skips a key
        an episode-info dict lacks): a share of nothing is not a 1.0.
        """
        if env_ids is None:
            self._stand_proxy_held[:] = self.STAND_PROXY_UNKNOWN
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
            self._stand_proxy_held[ids] = self.STAND_PROXY_UNKNOWN
        total = int(self._stand_proxy_n_total)
        if total == 0:
            return
        frac = self._stand_proxy_n_live.double() / total
        extras = getattr(self._env, "extras", None)
        if isinstance(extras, dict):
            extras.setdefault("log", {})[self.STAND_PROXY_LIVE_FRAC_KEY] = frac.float()
        self._stand_proxy_n_live.zero_()
        self._stand_proxy_n_total.zero_()

    # ------------------------------------------------- the stand-row input mask (p2r Z2d, 2026-10-11)
    def _log_stand_mask(self) -> None:
        """Log the share of rows masked since the last log, then restart it. Nothing is logged while no row was
        seen since the last log (as ``_reset_stand_proxy``: a share of nothing is not a 0.0)."""
        total = int(self._stand_mask_n_total)
        if total == 0:
            return
        frac = self._stand_mask_n_masked.double() / total
        extras = getattr(self._env, "extras", None)
        if isinstance(extras, dict):
            extras.setdefault("log", {})[self.STAND_MASK_FRAC_KEY] = frac.float()
        self._stand_mask_n_masked.zero_()
        self._stand_mask_n_total.zero_()

    def _mask_stand_rows(self, env: ParkourManagerBasedRLEnv, observations: torch.Tensor) -> None:
        """The shared mask on the stand rows of the RETURNED observation, in place (``_obs_history_buffer`` is never
        touched); counts the rows for the liveness log."""
        rows = env.command_manager.get_term("base_velocity").stand_mask().bool()
        restance.stand_mask_obs(observations, rows, inplace=True)
        self._stand_mask_n_masked += rows.sum()
        self._stand_mask_n_total += rows.numel()

    def _knee_gate_full(self, env: ParkourManagerBasedRLEnv) -> torch.Tensor:
        """``(N, 6)`` bool: the leg's KNEE rod drive gate is full -- the forward proxy's ``gate_k``
        (``_get_contact_fill``), same expression, at 1. The hip gate is deliberately not read."""
        action_term = env.action_manager.get_term("joint_pos")
        gate_k = (
            action_term.knee_rod_speed.abs()
            / (_CURRENT_SENSE_GATE_FRACTION * linkage.KNEE_SPEED_M_S)
        ).clamp(0.0, 1.0)
        return gate_k >= 1.0

    def _hold_stand_proxy(self, env: ParkourManagerBasedRLEnv, observations: torch.Tensor) -> None:
        """Update the held buffer from this step's live proxy, then write it into the CURRENT step's stand rows.

        In place on ``observations`` (columns 83-88 only); ``_obs_history_buffer`` is never touched.
        """
        c0 = self.STAND_PROXY_COL
        live = observations[:, c0:c0 + 6].clone()
        self._stand_proxy_held = torch.where(self._knee_gate_full(env), live, self._stand_proxy_held)
        stand = env.command_manager.get_term("base_velocity").stand_mask().bool()[:, None]
        observations[:, c0:c0 + 6] = torch.where(stand, self._stand_proxy_held, live)
        self._stand_proxy_n_total += stand.sum() * 6
        self._stand_proxy_n_live += ((self._stand_proxy_held == live) & stand).sum()

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
        """Obs dims 8-10: ``[cmd_vx_b, cmd_vy_b, ||cmd_xy||]`` -- or, under ``cmd_obs_scalar``,
        ``[0, 0, ||cmd_xy||]``.

        The norm is redundant with the pair but cheap, and it is the one command channel
        that is invariant under BOTH mirror planes -- a scalar the policy can read without
        first undoing a reflection. ``cmd_obs_scalar`` (side-speed-bisection plan step 1b) keeps
        exactly that invariant channel and drops the pair: the norm is computed from the REAL
        command either way, so a masked run is still driven by the same speed the un-masked one
        is, just without the direction.

        ``getattr(self, ...)``, not ``self.cmd_obs_scalar``: the existing unit test calls this
        hook unbound with ``self=None`` (it is otherwise a pure function of ``commands``, like the
        forward class's own version), and the off default must keep that working.
        """
        cmd_xy = commands[:, 0:2]
        speed = torch.linalg.norm(cmd_xy, dim=-1, keepdim=True)
        if getattr(self, "cmd_obs_scalar", False):
            cmd_xy = torch.zeros_like(cmd_xy)
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
        stand_obs: bool = False,
        lpf_obs: bool = False,
        clock_hide: bool = False,
        clock_hide_ramp: tuple[int, int] | None = None,
        cmd_obs_scalar: bool = False,
        stand_bit_clock: bool = False,
        stand_proxy_hold: bool = False,
        stand_mask: bool = False,
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
        # Same contract, and the same reason: the manager passes it here on every step, and
        # unlike stand_obs/lpf_obs it does not fix the observation WIDTH, so it may vary freely.
        self.cmd_obs_scalar = bool(cmd_obs_scalar)
        # The two head extensions fix the WIDTH, so they cannot change after construction.
        if bool(stand_obs) != self.stand_obs or bool(lpf_obs) != self.lpf_obs:
            raise ValueError(
                "stand_obs / lpf_obs changed after the observation term was built "
                f"(built {self.stand_obs}/{self.lpf_obs}, called {stand_obs}/{lpf_obs}); they set "
                "the observation width and are construction-time only"
            )
        # The clock-hide knob keeps the width, but its per-episode latch and log are set up at
        # construction, so it is construction-time only too.
        if bool(clock_hide) != self.clock_hide or (
            (clock_hide_ramp is not None or self.clock_hide_ramp is not None)
            and self._check_clock_hide_ramp(clock_hide_ramp, bool(clock_hide)) != self.clock_hide_ramp
        ):
            raise ValueError(
                "clock_hide / clock_hide_ramp changed after the observation term was built "
                f"(built {self.clock_hide}/{self.clock_hide_ramp}, called {clock_hide}/"
                f"{clock_hide_ramp}); they are construction-time only"
            )
        if bool(stand_bit_clock) != self.stand_bit_clock:
            raise ValueError(
                "stand_bit_clock changed after the observation term was built (built "
                f"{self.stand_bit_clock}, called {stand_bit_clock}); it is construction-time only"
            )
        if bool(stand_proxy_hold) != self.stand_proxy_hold:
            raise ValueError(
                "stand_proxy_hold changed after the observation term was built (built "
                f"{self.stand_proxy_hold}, called {stand_proxy_hold}); it is construction-time only"
            )
        if bool(stand_mask) != self.stand_mask:
            raise ValueError(
                "stand_mask changed after the observation term was built (built "
                f"{self.stand_mask}, called {stand_mask}); it is construction-time only"
            )
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
        if self.stand_bit_clock:
            # NOTE(p2r statue, 2026-10-09): the CURRENT step's column 15 only. The row
            # ``super().__call__`` just pushed onto the history carries the hidden clock's 0 there,
            # and nothing here touches ``_obs_history_buffer`` -- unlike slot 5 above, on purpose.
            observations[:, self.STAND_BIT_CLOCK_COL] = self._stand_bit(env)
        if self.stand_proxy_hold:
            # NOTE(p2r Z2, 2026-10-10): the held proxy on the CURRENT step's stand rows only; the row
            # ``super().__call__`` pushed onto the history carries the live proxy, and stays so.
            self._hold_stand_proxy(env, observations)
        if self.stand_mask:
            # NOTE(p2r Z2d, 2026-10-11): last action 0 / proxy -0.5 on the stand rows of the RETURNED observation
            # (current step + its real history frames); the stored history keeps the raw rows.
            self._mask_stand_rows(env, observations)
        return observations
