"""The ONE omni depth-student step (Task 4 pipeline, step 4 / M1b, DESIGN B4). DEFAULT OFF.

Every entry point that runs the omni depth student calls this module and nothing else: the trainer
(``OnPolicyRunnerWithExtractor._learn_vision_omni``), ``play.py``, ``play_stream_video.py`` and the eval
harness (``eval_crab_hex_gait.py``'s student branch, wired at pipeline step 5), plus the deploy exporter
(``omni_student_export``), which wraps the same :class:`OmniStudentObsPrep` and
``RecurrentMultiDepthBackbone.step``. None of the forward student paths reaches it: each caller routes here
only when :func:`is_omni_student` is true of the run's agent cfg (``depth_encoder.omni_student``, written
only by the omni student runner cfg and dumped into ``params/agent.yaml``).

What one step does (``OmniStudentStep.act``), on the env's 1149-wide policy observation:

1. **The student's copy** (:class:`OmniStudentObsPrep`, out of place -- the caller's ``obs`` is never written):
   * columns 11-12 (``env_idx`` / ``invert_env_idx``: "not flat" / "flat" tile flags, which the deployed
     robot cannot know) read the constant **field pair (1, 0)** -- in the current step AND in the ten real
     history frames (``259 + 89 j + {11, 12}``; DESIGN D3);
   * columns 207-215 (the actor's ``priv_explicit`` slice, ``num_prop + num_scan`` = 75 + 132) receive the
     frozen estimator's output, computed from the student copy's ``[:, :75]`` (the same write every teacher
     act path does). NOTE: with the actor's 75-wide slicing of the 1149 layout these nine columns hold, in
     the live observation, ``priv_latent[1:10]`` (CoM xyz, friction, five stiffness ratios) -- that is what
     the estimator was trained to predict; the body velocity x 2 sits at columns 197-199, inside the scan
     slice the depth latent replaces;
   * columns 6-7 (the facing command) and 13-14 (the true body velocity x 2, privileged by D2) are
     untouched, and so is everything else.
2. **Encode every** :data:`DEPTH_PERIOD` **env steps** (``common_step_counter % 5 == 0``, the image term's own
   refresh): the ``(N, K=4, 58, 87)`` images of the :data:`DEPTH_GROUPS` (camera order front, left, right,
   rear) and the student copy's ``[:, :89]`` (``num_prop`` 89, D4) -> ``RecurrentMultiDepthBackbone`` -> a
   32-d latent, held between encodes.
3. **Act**: ``depth_actor(student copy, hist_encoding=True, scandots_latent=latent)``.

``zero_depth`` (``depth_encoder.zero_depth``, default False; set only by a Hydra override such as
``agent.depth_encoder.zero_depth=true``) feeds all-zero images instead -- the step-8 "depth vs zeroed-depth"
pilot on this same code path. It is recorded in ``params/agent.yaml``.

``reset(dones)`` zeroes the GRU state of the done rows (and, with no argument, of every row and the held
latent). The held latent of a done row stays the old episode's until the next encode (<= 4 steps), exactly
as the image term's reset refill is up to one refresh stale (DESIGN B3: recorded, not fixed).

The teacher label (trainer only): ``teacher.act_inference`` on the TRUE observation (true tile flags) with
its own estimator write -- the observation the teacher acts on in every eval of record.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------------ the layout
#: The omni policy observation width (``crab_hex_omni_observations``; 89 + 108 + 9 + 53 + 10 x 89).
OBS_WIDTH = 1149
#: The real per-step proprio head (``obs_buf``) -- also the depth encoder's ``num_prop`` (D4).
OBS_BUF_WIDTH = 89
DEPTH_NUM_PROP = 89
#: The first real history frame: obs_buf 89 + heights 108 + priv_explicit 9 + priv_latent 53.
HISTORY_OFFSET = 259
HISTORY_FRAMES = 10
#: Never written by the student step (half-open column ranges).
FACING_COLS = (6, 8)          # the facing command (both channels)
VELOCITY_COLS = (13, 15)      # true planar body velocity x 2 -- privileged, kept (D2)
#: The tile-flag pair: col 11 = ``~on_flat`` ("env_idx"), col 12 = ``on_flat`` ("invert_env_idx").
TILE_FLAG_COLS = (11, 12)
#: A field tile's flags (``omni_field``: field sub-terrains are not named ``parkour_flat``) -- the constant
#: pair the student reads everywhere (training, eval, deploy; D3).
FIELD_TILE_FLAGS = (1.0, 0.0)
#: Encode cadence = the image term's refresh (``crab_hex_omni_observations.DEPTH_TERM_PERIOD``).
DEPTH_PERIOD = 5
#: The K image groups, in the encoder's camera order (``omni_student_spec.DEPTH_GROUPS``).
DEPTH_GROUPS = ("depth_front", "depth_left", "depth_right", "depth_rear")
#: The agent-cfg routing flag (``depth_encoder.omni_student``).
OMNI_STUDENT_FLAG = "omni_student"


def tile_flag_columns(width: int = OBS_WIDTH) -> tuple[int, ...]:
    """The tile-flag columns of a ``width``-wide observation: current step, plus the 10 history frames
    when the observation is full width."""
    starts = [0]
    if width >= HISTORY_OFFSET + HISTORY_FRAMES * OBS_BUF_WIDTH:
        starts += [HISTORY_OFFSET + OBS_BUF_WIDTH * j for j in range(HISTORY_FRAMES)]
    return tuple(s + c for s in starts for c in TILE_FLAG_COLS)


def is_omni_student(train_cfg) -> bool:
    """True iff the agent cfg (train dict, dumped ``agent.yaml`` dict, or cfg object) marks the omni student."""
    if train_cfg is None:
        return False
    de = train_cfg.get("depth_encoder") if isinstance(train_cfg, Mapping) else getattr(train_cfg, "depth_encoder", None)
    if de is None:
        return False
    flag = de.get(OMNI_STUDENT_FLAG, False) if isinstance(de, Mapping) else getattr(de, OMNI_STUDENT_FLAG, False)
    return bool(flag)


def stack_depth_images(observations: Mapping, groups=DEPTH_GROUPS, device=None) -> torch.Tensor:
    """``{group: (N, H, W)}`` -> ``(N, K, H, W)`` in ``groups`` order (a missing group raises)."""
    missing = [g for g in groups if g not in observations]
    if missing:
        raise KeyError(f"omni student: observation groups {missing} missing (have {sorted(observations)})")
    images = torch.stack([observations[g] for g in groups], dim=1)
    return images if device is None else images.to(device)


# ---------------------------------------------------------------------------- the student's observation
class OmniStudentObsPrep(nn.Module):
    """``obs`` -> the student's copy: the constant field pair on every tile-flag column, then the estimator write.

    Out of place (``torch.where`` / ``torch.cat``): the input is never modified, columns outside the two
    writes are bit-for-bit the input's. A module (buffers + the frozen estimator) so the deploy export wraps
    this very code (``omni_student_export``).
    """

    def __init__(self, estimator: nn.Module, *, num_prop: int, num_scan: int, num_priv_explicit: int,
                 width: int = OBS_WIDTH, field_flags=FIELD_TILE_FLAGS):
        super().__init__()
        self.estimator = estimator
        self.num_prop = int(num_prop)
        self.est_start = int(num_prop) + int(num_scan)
        self.est_stop = self.est_start + int(num_priv_explicit)
        self.width = int(width)
        if not (OBS_BUF_WIDTH <= self.est_start < self.est_stop <= HISTORY_OFFSET):
            raise ValueError(f"estimator columns [{self.est_start}, {self.est_stop}) leave the head/priv region")
        mask = torch.zeros(1, self.width, dtype=torch.bool)
        fill = torch.zeros(1, self.width)
        cols = tile_flag_columns(self.width)
        for i, col in enumerate(cols):  # (11, 12) pairs: current step, then each history frame
            mask[0, col] = True
            fill[0, col] = float(field_flags[i % len(TILE_FLAG_COLS)])
        self.register_buffer("flag_mask", mask, persistent=False)
        self.register_buffer("flag_fill", fill, persistent=False)

    def tile_flags(self, x: torch.Tensor) -> torch.Tensor:
        """The field pair on the tile-flag columns of ``x`` (any width up to :data:`OBS_WIDTH`)."""
        w = x.shape[1]
        return torch.where(self.flag_mask[:, :w], self.flag_fill[:, :w].to(x.dtype), x)

    def estimator_write(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` with columns ``[num_prop + num_scan, + num_priv_explicit)`` = the frozen estimator of ``x[:, :num_prop]``."""
        with torch.no_grad():
            est = self.estimator(x[:, : self.num_prop])
        return torch.cat([x[:, : self.est_start], est.to(x.dtype), x[:, self.est_stop:]], dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[1] != self.width:
            raise ValueError(f"omni student: policy observation is {obs.shape[1]} wide, expected {self.width}")
        return self.estimator_write(self.tile_flags(obs))


# --------------------------------------------------------------------------------------- the one step
class OmniStudentStep:
    """The omni depth student's per-env-step computation -- see the module docstring for the contract.

    Build it with :meth:`from_runner` (every entry point does). ``act`` works with or without autograd: the
    trainer calls it with gradients on (the latent and actions carry the graph to ``update_depth_actor``),
    play / harness under ``torch.inference_mode()``.
    """

    def __init__(self, depth_encoder: nn.Module, depth_actor: nn.Module, estimator: nn.Module, *,
                 num_prop: int, num_scan: int, num_priv_explicit: int, depth_num_prop: int = DEPTH_NUM_PROP,
                 teacher: nn.Module | None = None, groups=DEPTH_GROUPS, period: int = DEPTH_PERIOD,
                 device=None, zero_depth: bool = False):
        self.depth_encoder = depth_encoder
        self.depth_actor = depth_actor
        self.teacher = teacher
        self.groups = tuple(groups)
        self.period = int(period)
        self.depth_num_prop = int(depth_num_prop)
        self.zero_depth = bool(zero_depth)
        enc_prop = getattr(depth_encoder, "num_prop", self.depth_num_prop)
        enc_k = getattr(depth_encoder, "num_cameras", len(self.groups))
        if int(enc_prop) != self.depth_num_prop or int(enc_k) != len(self.groups):
            raise ValueError(f"omni student: encoder takes num_prop={enc_prop}, K={enc_k}; the step feeds "
                             f"num_prop={self.depth_num_prop}, K={len(self.groups)} ({self.groups})")
        self.prep = OmniStudentObsPrep(estimator, num_prop=num_prop, num_scan=num_scan,
                                       num_priv_explicit=num_priv_explicit)
        if device is not None:
            self.prep.to(device)
        self.depth_latent: torch.Tensor | None = None

    @classmethod
    def from_runner(cls, runner, device=None) -> "OmniStudentStep":
        """The step of a built ``OnPolicyRunnerWithExtractor`` (omni student cfg; networks as loaded)."""
        cfg = runner.cfg
        if not is_omni_student(cfg):
            raise ValueError("OmniStudentStep.from_runner: the runner cfg does not mark the omni student "
                             f"(depth_encoder.{OMNI_STUDENT_FLAG})")
        if cfg.get("empirical_normalization"):
            raise ValueError("OmniStudentStep: empirical_normalization is not supported (the omni student "
                             "runner never sets it)")
        est, de = cfg["estimator"], cfg["depth_encoder"]
        alg = runner.alg
        return cls(alg.depth_encoder, alg.depth_actor, alg.estimator,
                   num_prop=int(est["num_prop"]), num_scan=int(est["num_scan"]),
                   num_priv_explicit=int(est["num_priv_explicit"]), depth_num_prop=int(de["num_prop"]),
                   teacher=alg.policy, device=runner.device if device is None else device,
                   zero_depth=bool(de.get("zero_depth", False)))

    # ----------------------------------------------------------------------------------- the pieces
    def depth_images(self, observations: Mapping, device=None) -> torch.Tensor:
        images = stack_depth_images(observations, self.groups, device)
        return torch.zeros_like(images) if self.zero_depth else images

    def student_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.prep(obs)

    def encode(self, obs_student: torch.Tensor, depth_images: torch.Tensor, step_counter: int) -> torch.Tensor:
        """Refresh the held latent on the encode cadence (or when none is held); return it."""
        if self.depth_latent is None or int(step_counter) % self.period == 0:
            self.depth_latent = self.depth_encoder(depth_images, obs_student[:, : self.depth_num_prop])
        return self.depth_latent

    # ------------------------------------------------------------------------------------ the API
    def act(self, obs: torch.Tensor, depth, step_counter: int) -> torch.Tensor:
        """Student actions for the env's policy ``obs``; ``depth`` = the obs-group dict or ``(N, K, H, W)``."""
        if isinstance(depth, Mapping):
            depth = self.depth_images(depth, device=obs.device)
        elif self.zero_depth:
            depth = torch.zeros_like(depth)
        obs_student = self.student_obs(obs)
        latent = self.encode(obs_student, depth, step_counter)
        return self.depth_actor(obs_student, hist_encoding=True, scandots_latent=latent)

    def teacher_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """The distillation label: the teacher on the TRUE observation with its own estimator write."""
        if self.teacher is None:
            raise ValueError("OmniStudentStep.teacher_actions: built without a teacher policy")
        with torch.no_grad():
            return self.teacher.act_inference(self.prep.estimator_write(obs), hist_encoding=True)

    def reset(self, dones=None) -> None:
        """GRU reset on the done rows; no argument = every row, and the held latent is dropped."""
        self.depth_encoder.reset(dones)
        if dones is None:
            self.depth_latent = None

    def detach(self) -> None:
        """Cut the graph at an update boundary (GRU state and held latent)."""
        self.depth_encoder.detach_hidden_states()
        if self.depth_latent is not None:
            self.depth_latent = self.depth_latent.detach()
