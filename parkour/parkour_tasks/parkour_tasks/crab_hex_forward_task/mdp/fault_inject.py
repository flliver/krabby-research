# SPDX-License-Identifier: BSD-3-Clause
"""Mid-episode fault injection for the fault-injection sweep (omni, 2026-11-15). DEFAULT OFF.

Armed only by ``crab_hex_omni_task/config/crab_hex/fault_knobs.py``, which writes the env-cfg attribute
:data:`FAULT_ATTR` (a plain dict) for ONE cell (``KRABBY_FAULT=<id>:<level>``). Without that attribute
:func:`get_injector` returns ``None``, no term keeps a reference, and every hook below is skipped (``getattr(self,
"_fault", None) is None``): the knob-off path is bitwise the old one.

ONE injector per env (``env._fault_injector``), built by the crab action term (constructed before the observation
manager) and shared with the policy observation term and the four depth terms. Its hooks (CHARTER §3):

==================  ======================================================================================  =========
stage               where (called from)                                                                     faults
==================  ======================================================================================  =========
``sensor``          ``observations.CrabHexParkourObservations.__call__``, on the RAW channels right after    S1-S8,
                    the stress-obs hook: before the scales, the ``torch.cat`` and the history push           Z1, S12,
                                                                                                             D1
``depth_raw``       ``crab_hex_omni_observations.image_features_batched._processed`` at a refresh (never    S10, S11,
                    the reset refill), on the raw ``(N, H, W)`` metres BEFORE ``process_depth_batch``        Z1
``on_actions``      ``parkour_actions.process_actions`` after the clip + history push + delayed read, before  B1-B3,
                    the scale/offset (the applied action); also the per-step dose draws and the A2 write    B5, A2
``rod_dlam``        ``_apply_pitch_linkage`` per physics substep, on the clamped rod slew before it is       B4, B6,
                    integrated (hip, then knee)                                                              A1, A3
``after_linkage``   end of ``_apply_pitch_linkage``: the broken leg's joint targets (A2)                     A2
``cam_cmd`` /       ``apply_actions`` per substep: the cam velocity command before the acceleration slew /   B4, B6 /
``cam_applied``     the slewed cam target before it is written                                               A4
==================  ======================================================================================  =========

Step index. ``k = env.episode_length_buf`` at the hook. In ``process_actions`` and the physics substeps of a step
that is the policy step ``t`` (incremented only after physics); in the observation computed at the end of step
``t`` it is ``t + 1`` -- the index of the policy step that CONSUMES that observation (0 right after a reset). So a
fault active on steps ``[onset, onset + d)`` corrupts the observations of policy steps onset..onset+d-1 and the
actions of those same steps; a permanent fault ``[onset, end)``.

Onsets (CHARTER §3.2). A table ``(WAVES_MAX, N)`` of onset STEPS drawn ONCE, at construction, from a private CPU
``torch.Generator`` seeded ``(env_cfg.seed + 0x6661756C74) % 2**63`` (independent of the cell):
``lo_steps + floor(u * (hi_steps - lo_steps))`` with ``u ~ U[0, 1)`` (float64) and ``lo_steps = round(lo / dt)``,
``hi_steps = round(hi / dt)`` -- U(lo, hi) s floored to the step grid. Dose draws (S3/S6 headings, S4 walk, B2, B3)
come from a SECOND private CPU generator re-seeded ``(env_cfg.seed + 0x646F7365 + wave) % 2**63`` at each wave
start (headings first, then one fixed-shape draw per env step for all envs, whatever fell), so a wave's doses do not
depend on what earlier waves did. No other random stream is touched.

Wave index. The eval harness resets EVERY env explicitly (``env.reset()``) at the start of each suite wave, and the
RSL-RL wrapper once at construction (= wave 0); mid-wave auto-resets come from ``env.step`` for the terminated envs
only. The injector counts waves itself: a reset of all ``N`` envs that is NOT an in-step reset opens a new wave
(``wave`` = number of such resets - 1). An in-step reset is recognised by ``env.reset_buf[ids].all()`` AND
``env.episode_length_buf[ids] > 0`` (``step()`` recomputes ``reset_buf`` and has just incremented the length before
it resets; an explicit reset after a step in which every env was reset sees those lengths at 0), so even "every env
fell on the same step" is not a new wave. The harness asserts ``injector.wave == wave`` at every wave start
(:meth:`FaultInjector.check_wave`), so a miscount is a launch error, never a silent misalignment.

Resets. At every episode reset all fault state of the reset envs is cleared (holds, walk, frozen frames, delay
draws, previous action, rod / cam records, A2 gains restored in PhysX and in the actuator model). An in-step
(mid-wave) reset also DISARMS the env for the rest of the wave (a fallen env stays done; the table is never redrawn
or re-armed); the next wave re-arms every env with that wave's row of the table.

Probes. Per (wave, env) summaries (running max / counts on device, flushed to host at each wave start and at
:meth:`FaultInjector.sidecar`), written by the harness as ``fault_onsets.json`` next to the stamp's
``run_meta.json`` (schema :data:`SIDECAR_SCHEMA`, see :meth:`FaultInjector.sidecar`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

#: The env-cfg attribute ``fault_knobs`` writes (absent = off).
FAULT_ATTR = "fault_inject"
#: The env attribute holding the shared injector (the depth term reads it by this literal).
ENV_ATTR = "_fault_injector"
#: ``b"fault"`` / ``b"dose"`` -- added to the env seed (CHARTER §3.2 names the first).
ONSET_SEED_SALT = 0x6661756C74
DOSE_SEED_SALT = 0x646F7365
#: Rows of the onset table (a suite row runs 10 waves; the single-pass rows 1).
WAVES_MAX = 64
#: "never ends" (permanent faults).
NEVER = 1 << 40
SIDECAR_NAME = "fault_onsets.json"
#: The depth term's refresh period (steps) and lag (refreshes) -- ``crab_hex_omni_observations.DEPTH_TERM_PERIOD`` and
#: ``image_features_batched.lag``; the depth term overwrites them at registration (:meth:`FaultInjector.depth_timing`).
DEPTH_PERIOD_DEFAULT = 5
DEPTH_LAG_DEFAULT = 1
SIDECAR_SCHEMA = "fault_onsets/v1"

#: Rod-window limits (``crab_hex_linkage``); imported lazily so this module stays torch-only for the unit tests.
_LIMITS = None

SENSOR_KINDS = frozenset({"S1", "S2", "S3", "S4", "S5", "S6", "S8", "Z1", "S12", "D1"})
DEPTH_KINDS = frozenset({"S10", "S11", "Z1"})
ACTION_KINDS = frozenset({"B1", "B2", "B3", "B5"})
ROD_KINDS = frozenset({"B4", "B6", "A1", "A3"})
CAM_CMD_KINDS = frozenset({"B4", "B6"})
CAM_APPLIED_KINDS = frozenset({"A4"})
GAIN_KINDS = frozenset({"A2"})
LEG_KINDS = frozenset({"A1", "A2", "A3", "A4"})
KNOWN_KINDS = SENSOR_KINDS | DEPTH_KINDS | ACTION_KINDS | ROD_KINDS | CAM_APPLIED_KINDS | GAIN_KINDS
#: Faults on the student's INPUTS (obs / depth path): refused unless the policy role is the student -- the teacher
#: reads privileged state the sensor stage never touches, so they would be bypassed (``check_role``).
STUDENT_ONLY_KINDS = SENSOR_KINDS | DEPTH_KINDS
TOF_CAMERAS = ("left", "right", "rear")
FRONT_CAMERA = "front"
#: PhysX actuator groups of the rod joints (``crab_hex_scene_cfg``).
HIP_ACTUATOR = "hip_femur"
KNEE_ACTUATOR = "femur_tibia"
_SPEC_KEYS = {"id", "level", "knob", "family", "duration_s", "duration_steps", "permanent", "params", "onset_s",
              "from_start"}


def onset_seed(env_seed) -> int:
    base = 0 if env_seed is None else int(env_seed)
    return (base + ONSET_SEED_SALT) % (1 << 63)


#: The golden-ratio multiplier that spreads the env seed before the wave is added (no (seed, wave) collisions).
DOSE_SEED_MIX = 0x9E3779B97F4A7C15


def dose_seed(env_seed, wave: int) -> int:
    """``(env_seed * DOSE_SEED_MIX + DOSE_SEED_SALT + wave) % 2**63`` -- a pure function; the multiplier keeps
    ``dose_seed(s, w + 1)`` apart from ``dose_seed(s + 1, w)`` (a plain sum made them equal)."""
    base = 0 if env_seed is None else int(env_seed)
    return (base * DOSE_SEED_MIX + DOSE_SEED_SALT + int(wave)) % (1 << 63)


def onset_window_steps(lo_s: float, hi_s: float, step_dt: float) -> tuple[int, int]:
    lo, hi = int(round(float(lo_s) / float(step_dt))), int(round(float(hi_s) / float(step_dt)))
    if not 0 <= lo < hi:
        raise ValueError(f"onset window ({lo_s}, {hi_s}) s is empty on the {step_dt} s step grid")
    return lo, hi


def onset_table(lo_s: float, hi_s: float, step_dt: float, num_envs: int, env_seed, waves: int = WAVES_MAX):
    """``(waves, num_envs)`` long onset steps (CPU), a pure function of the window, dt, N and the env seed."""
    lo, hi = onset_window_steps(lo_s, hi_s, step_dt)
    g = torch.Generator(device="cpu")
    g.manual_seed(onset_seed(env_seed))
    u = torch.rand((int(waves), int(num_envs)), generator=g, dtype=torch.float64)
    return lo + torch.floor(u * (hi - lo)).long()


def get_injector(env):
    """The env's shared :class:`FaultInjector`, built on first call from ``env.cfg.fault_inject``; ``None`` (and
    nothing written onto the env) while the cfg carries no fault."""
    inj = getattr(env, ENV_ATTR, None)
    if inj is not None:
        return inj
    spec = getattr(getattr(env, "cfg", None), FAULT_ATTR, None)
    if not spec:
        return None
    inj = FaultInjector(dict(spec), num_envs=int(env.num_envs), step_dt=float(env.step_dt), device=env.device,
                        env_seed=getattr(env.cfg, "seed", None))
    setattr(env, ENV_ATTR, inj)
    return inj


#: Probes that only re-measure the injector's own output against its own state (mask / wiring self-checks).
_SELF_CHECK_SUFFIXES = ("_pre_dev", "_post_dev", "_change_max", "_held_dev", "_frozen_dev", "_slew_dev",
                        "_cmd_dev", "_dlam_absmax", "_slew_ratio_max", "_at_cap_n", "_len_drift")
_SELF_CHECK_NAMES = {"bias_dev", "vxy_absmax", "cf_absmax", "front_raw_absmax", "front_img_dev", "s12_hold_dev",
                     "d1_jpd_dev", "d1_jv_cf_absmax", "a4_cam_target_absmax", "b6_cam_cmd_absmax"}
#: Probes read from the simulator (asset data, PhysX, the real sensor channel before the fault).
_SIM_NAMES = {"a2_gain_pre", "a2_gain_post", "a2_gain_pre_onset_dev", "a2_gain_wave_start_dev", "a2_target_leave_max",
              "a4_cam_joint_vel_absmax", "a4_cam_angle_drift", "front_raw_in_max"}


def probe_kind(name: str) -> str:
    """The sidecar label of a probe: ``self_check`` (the injector's output vs its own state: proves the mask / wiring,
    not the plant), ``sim`` (read back from the simulator), ``model_estimate`` (IsaacLab's actuator-model estimate,
    not PhysX), ``dose`` (the realised dose draws / counts), ``activity`` (first / last active step, counts)."""
    if name == "a2_applied_torque_absmax":
        return "model_estimate"
    if name.startswith(("first_", "last_")) or name.endswith("_active_n") or name in ("a2_write_step",
                                                                                     "a2_gain_pre_onset_reads"):
        return "activity"
    if name in _SIM_NAMES or name.endswith("_raw_move"):
        return "sim"
    if name in _SELF_CHECK_NAMES or name.startswith("b6_") or name.endswith(_SELF_CHECK_SUFFIXES):
        return "self_check"
    return "dose"


def _round(x: float) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(f"{float(x):.6g}")


def _rowmax(t: torch.Tensor) -> torch.Tensor:
    """``(N, ...)`` -> ``(N,)`` max |t| over every non-env axis."""
    return t.abs().reshape(t.shape[0], -1).amax(dim=1) if t.numel() else torch.zeros(t.shape[0], device=t.device)


class FaultInjector:
    """Per-env fault state, onset table, dose draws and probes for ONE armed cell (see the module docstring)."""

    def __init__(self, spec: dict, *, num_envs: int, step_dt: float, device, env_seed=None):
        unknown = sorted(set(spec) - _SPEC_KEYS)
        if unknown:
            raise ValueError(f"{FAULT_ATTR}: unknown spec keys {unknown}")
        self.spec = dict(spec)
        self.fid = str(spec["id"])
        self.params = dict(spec.get("params") or {})
        self.kind = str(self.params.get("as", self.fid))          # A5 = A1 at 90 kg
        if self.kind not in KNOWN_KINDS:
            raise ValueError(f"{FAULT_ATTR}: unknown fault {self.fid!r}")
        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = device
        self.env_seed = env_seed
        self.from_start = bool(spec.get("from_start", False))
        if spec.get("duration_steps") is not None:
            self.duration = int(spec["duration_steps"])
        elif spec.get("duration_s") is not None:
            self.duration = int(round(float(spec["duration_s"]) / self.step_dt))
        else:
            self.duration = None
        if self.duration is not None and self.duration < 1:
            raise ValueError(f"{FAULT_ATTR}: duration {self.duration} steps")
        lo_s, hi_s = (float(v) for v in spec["onset_s"])
        self.window_steps = onset_window_steps(lo_s, hi_s, self.step_dt)
        self.table = onset_table(lo_s, hi_s, self.step_dt, self.num_envs, env_seed).to(device)
        self.dose_gen = torch.Generator(device="cpu")
        n = self.num_envs
        self.wave = -1
        self.armed = torch.zeros(n, dtype=torch.bool, device=device)
        self.start = torch.full((n,), NEVER, dtype=torch.long, device=device)
        self.end = torch.full((n,), NEVER, dtype=torch.long, device=device)
        self.records: list[dict] = []
        self._flushed: set[int] = set()
        self.bound: set[str] = set()
        self._checked = False
        self.term = None
        self.binding: dict = {}
        # state (allocated lazily where the shape is only known at the first call)
        self.heading = torch.zeros(n, device=device)
        self.holds: dict[str, torch.Tensor] = {}
        self.walk = torch.zeros(n, 2, device=device)
        self.s12_prev = None
        self.frozen: dict[str, torch.Tensor] = {}
        self.prev_applied = None
        self.last_dlam: dict[str, torch.Tensor] = {}
        self.len0: dict[str, torch.Tensor] = {}
        self.last_cam_cmd = None
        self.cam_theta0 = None
        self.gain_written = torch.zeros(n, dtype=torch.bool, device=device)
        self.saved_physx = torch.zeros(n, 3, 2, device=device)       # [kp, kd, fmax] x [hip, knee]
        self.saved_act = torch.zeros(n, 3, 2, device=device)
        self.frozen_tgt = torch.zeros(n, 2, device=device)
        self._last_obs_k = torch.full((n,), -1, dtype=torch.long, device=device)
        self._step_draw = None
        self.p: dict[str, torch.Tensor] = {}
        self.fell_at = torch.full((n,), -1, dtype=torch.long, device=device)
        self.depth_period, self.depth_lag = DEPTH_PERIOD_DEFAULT, DEPTH_LAG_DEFAULT
        self.gain_snapshot = None        # A2: the leg's PhysX [kp, kd, fmax] x [hip, knee] at construction (N, 3, 2)
        self.state_cleared: list[bool] = []
        self.p_wave_start: dict[str, torch.Tensor] = {}
        self.role = None

    # ------------------------------------------------------------------ binding
    def register(self, stage: str) -> None:
        """A consumer stage announces itself (``"obs"``, ``"depth:<camera>"``); checked at the first step."""
        self.bound.add(str(stage))

    def depth_timing(self, period: int, lag: int) -> None:
        """The depth term's refresh period (steps) and lag (refreshes): when a faulted frame leaves the policy."""
        self.depth_period, self.depth_lag = int(period), int(lag)

    def check_role(self, role: str) -> None:
        """The harness's call once the policy role is resolved: sensor / depth / anchor faults need the student."""
        self.role = str(role)
        if self.kind in STUDENT_ONLY_KINDS and self.role != "student":
            raise RuntimeError(f"fault {self.fid} acts on the student's observation / depth inputs; the policy role "
                               f"is {self.role!r}, which reads privileged state the fault never touches (refused)")

    @staticmethod
    def camera_name(sensor_name: str) -> str:
        s = str(sensor_name)
        return s[len("depth_camera_"):] if s.startswith("depth_camera_") else s

    def bind_action_term(self, term) -> None:
        """Called from the crab action term's ``__init__``: resolves the left leg / joints the cell acts on."""
        self.term = term
        self.register("action")
        names = list(term._joint_names)
        pitch_names = [names[c] for c in term._pitch_action_cols]
        knee_names = [names[c] for c in term._knee_action_cols]
        cam_names = [names[c] for c in term._cam_action_cols]
        joint_names = list(term._asset.joint_names)
        self.binding = {"action_joint_names": names}
        if self.kind in LEG_KINDS:
            leg = str(self.params["leg"])
            hits = [i for i, nm in enumerate(pitch_names) if nm.split("_")[0] == leg]
            if len(hits) != 1:
                raise RuntimeError(f"fault {self.fid}: leg {leg} not found once in {pitch_names}")
            li = hits[0]
            if knee_names[li].split("_")[0] != leg:
                raise RuntimeError(f"fault {self.fid}: knee order {knee_names} does not follow the hips {pitch_names}")
            jh, jk = int(term._pitch_joint_ids[li]), int(term._knee_joint_ids[li])
            if joint_names[jh] != pitch_names[li] or joint_names[jk] != knee_names[li]:
                raise RuntimeError(f"fault {self.fid}: joint ids {jh, jk} are {joint_names[jh], joint_names[jk]}")
            self.leg_col = li
            self.leg_joint_ids = [jh, jk]
            self.binding.update({"leg": leg, "leg_col": li, "joints": [joint_names[jh], joint_names[jk]],
                                 "joint_ids": [jh, jk]})
            cams = [i for i, nm in enumerate(cam_names) if nm.split("_")[0] == leg]
            if self.kind == "A4":
                if len(cams) != 1:
                    raise RuntimeError(f"fault A4: leg {leg} cam not found once in {cam_names}")
                self.cam_col = cams[0]
                self.cam_joint_id = int(term._cam_shaft_joint_ids[self.cam_col])
                self.binding.update({"cam_col": self.cam_col, "cam_joint": joint_names[self.cam_joint_id]})
            if self.kind == "A2":
                # the nominal gains every pre-onset read-back is compared with (PhysX is up: sim.reset() precedes the
                # managers)
                all_ids = torch.arange(self.num_envs, device=self.device)
                self.gain_snapshot = self._read_gains(term._asset, all_ids, self.leg_joint_ids).to(self.device)
                self.binding["a2_gain_snapshot_env0"] = [float(x) for x in self.gain_snapshot[0].reshape(-1)]
        if self.kind in CAM_CMD_KINDS | CAM_APPLIED_KINDS:
            shaft = [joint_names[int(j)] for j in term._cam_shaft_joint_ids]
            if shaft != cam_names:
                raise RuntimeError(f"fault {self.fid}: cam action columns {cam_names} != cam shaft joints {shaft}")
        if self.kind in ("B3", "B5"):
            hist = int(term._action_history_buf.shape[1])
            if not bool(getattr(term, "_use_delay", False)) or hist < 4:
                raise RuntimeError(f"fault {self.fid}: needs use_delay and a >= 4 deep action history (got "
                                   f"use_delay={getattr(term, '_use_delay', None)}, history {hist})")
        if self.kind == "D1":
            jids = term._joint_ids
            if isinstance(jids, slice):
                jids = list(range(len(joint_names)))[jids]
            cols = list(term._pitch_action_cols) + list(term._knee_action_cols)
            self.d1_cols = cols
            self.d1_joint_ids = [int(jids[c]) for c in cols]
            self.binding.update({"d1_target_joints": [joint_names[j] for j in self.d1_joint_ids]})

    def _check_bound(self) -> None:
        need = {"action"}
        if self.kind in SENSOR_KINDS:
            need.add("obs")
        if self.kind in ("S10", "Z1"):
            need.add(f"depth:{FRONT_CAMERA}")
        if self.kind == "S11":
            need.update(f"depth:{c}" for c in TOF_CAMERAS)
        missing = sorted(need - self.bound)
        if missing:
            raise RuntimeError(f"fault {self.fid}: the stage(s) {missing} never bound -- the fault would be inert "
                               f"(bound: {sorted(self.bound)})")
        self._checked = True

    # ------------------------------------------------------------------ masks / probes
    def _k(self, env) -> torch.Tensor:
        return env.episode_length_buf.to(self.device).long()

    def _masks(self, k: torch.Tensor):
        act = self.armed & (k >= self.start) & (k < self.end)
        pre = self.armed & (k < self.start)
        post = self.armed & (k >= self.end)
        return act, pre, post

    def _pmax(self, name: str, mask: torch.Tensor, val: torch.Tensor) -> None:
        cur = self.p.get(name)
        if cur is None:
            cur = torch.zeros(self.num_envs, device=self.device)
        val = val.to(cur.dtype)
        self.p[name] = torch.where(mask, torch.maximum(cur, val), cur)

    def _pcount(self, name: str, mask: torch.Tensor) -> None:
        cur = self.p.get(name)
        if cur is None:
            cur = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.p[name] = cur + mask.long()

    def _pset(self, name: str, mask: torch.Tensor, val: torch.Tensor, width: int | None = None) -> None:
        cur = self.p.get(name)
        if cur is None:
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            cur = torch.full(shape, float("nan"), device=self.device)
        m = mask if width is None else mask[:, None]
        self.p[name] = torch.where(m, val.to(cur.dtype), cur)

    def _activity(self, stage: str, act: torch.Tensor, k: torch.Tensor) -> None:
        first = self.p.get(f"first_{stage}")
        if first is None:
            first = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            last = first.clone()
        else:
            last = self.p[f"last_{stage}"]
        self.p[f"first_{stage}"] = torch.where(act & (first < 0), k, first)
        self.p[f"last_{stage}"] = torch.where(act, torch.maximum(last, k), last)

    def _devs(self, stage: str, pre, post, dev: torch.Tensor) -> None:
        self._pmax(f"{stage}_pre_dev", pre, dev)
        if self.duration is not None:
            self._pmax(f"{stage}_post_dev", post, dev)

    # ------------------------------------------------------------------ resets / waves
    def check_wave(self, wave: int) -> None:
        if int(wave) != self.wave:
            raise RuntimeError(f"fault injector counted wave {self.wave} but the harness is at wave {wave}: the "
                               "onset table would be misaligned (launch error)")

    def on_reset(self, term, env_ids=None) -> None:
        """The crab action term's reset hook (once per env reset, inside ``_reset_idx``)."""
        env = term._env
        n = self.num_envs
        if env_ids is None:
            ids = torch.arange(n, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        elb = env.episode_length_buf.to(self.device)
        rb = getattr(env, "reset_buf", None)
        in_step = (rb is not None and bool(rb.to(self.device)[ids].bool().all())
                   and bool((elb[ids] > 0).all()))
        full = int(torch.unique(ids).numel()) == n
        if full and not in_step:
            self._new_wave(term)
        else:
            newly = self.armed[ids] & (self.fell_at[ids] < 0)
            self.fell_at[ids] = torch.where(newly, elb[ids].long(), self.fell_at[ids])
            self.armed[ids] = False
            self._restore_gains(term, ids)
            self._clear(ids)

    def _new_wave(self, term) -> None:
        if self.wave >= 0:
            self._flush()
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._restore_gains(term, all_ids)
        self.wave += 1
        if self.wave >= WAVES_MAX:
            raise RuntimeError(f"fault injector: more than {WAVES_MAX} waves")
        onset = self.table[self.wave]
        self.onset = onset.clone()
        self.start = torch.zeros_like(onset) if self.from_start else onset.clone()
        if self.duration is None or self.from_start:
            self.end = torch.full_like(onset, NEVER)
        else:
            self.end = onset + int(self.duration)
        self.armed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.fell_at = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.p = {}
        self._clear(all_ids)
        self._check_cleared()
        if self.kind in GAIN_KINDS and self.gain_snapshot is not None:
            # after the restore above: every env's leg gains must be the construction snapshot again
            # kept apart from the running probes (``self.p`` restarts empty with every wave), merged at the flush
            self.p_wave_start = {"a2_gain_wave_start_dev": self._gain_dev(term, all_ids)}
        self.dose_gen.manual_seed(dose_seed(self.env_seed, self.wave))
        if self.kind in ("S3", "S6"):
            theta = torch.rand(self.num_envs, generator=self.dose_gen, dtype=torch.float64) * (2.0 * math.pi)
            self.heading = theta.to(device=self.device, dtype=torch.float32)

    def _clear(self, ids: torch.Tensor) -> None:
        for t in self.holds.values():
            t[ids] = 0.0
        self.walk[ids] = 0.0
        if self.s12_prev is not None:
            self.s12_prev[ids] = 0.0
        for t in self.frozen.values():
            t[ids] = 0.0
        if self.prev_applied is not None:
            self.prev_applied[ids] = 0.0
        for t in self.last_dlam.values():
            t[ids] = 0.0
        for t in self.len0.values():
            t[ids] = float("nan")
        if self.last_cam_cmd is not None:
            self.last_cam_cmd[ids] = 0.0
        if self.cam_theta0 is not None:
            self.cam_theta0[ids] = float("nan")
        self.frozen_tgt[ids] = 0.0
        self._last_obs_k[ids] = -1

    #: tensors that are NOT per-episode fault state (tables, masks, per-wave draws, saved restore values, probes)
    _NOT_STATE = frozenset({"table", "armed", "start", "end", "onset", "fell_at", "heading", "saved_physx", "saved_act",
                            "gain_snapshot", "p", "p_wave_start"})
    #: cleared values that are not 0
    _CLEARED = {"len0": float("nan"), "cam_theta0": float("nan"), "_last_obs_k": -1, "gain_written": False}

    def _check_cleared(self) -> None:
        """Every per-episode fault state tensor (any tensor attribute, or dict of tensors, not in ``_NOT_STATE``) is
        at its cleared value at a wave start; appends ``True`` to ``state_cleared`` or raises (a launch error)."""
        bad = []
        for name, val in vars(self).items():
            if name in self._NOT_STATE:
                continue
            items = ([(name, val)] if isinstance(val, torch.Tensor) else
                     [(f"{name}[{k}]", v) for k, v in val.items() if isinstance(v, torch.Tensor)]
                     if isinstance(val, dict) else [])
            want = self._CLEARED.get(name, 0)
            for label, t in items:
                if isinstance(want, float) and math.isnan(want):
                    ok = bool(torch.isnan(t).all())
                else:
                    ok = bool((t == want).all())
                if not ok:
                    bad.append(label)
        if bad:
            raise RuntimeError(f"fault injector: state not cleared at the start of wave {self.wave}: {bad}")
        self.state_cleared.append(True)

    def _gain_dev(self, term, ids: torch.Tensor) -> torch.Tensor:
        """``(N,)`` max |PhysX read-back - construction snapshot| over [kp, kd, fmax] x [hip, knee] (0 off ``ids``)."""
        out = torch.zeros(self.num_envs, device=self.device)
        rb = self._read_gains(term._asset, ids, self.leg_joint_ids).to(self.device)
        out[ids] = (rb - self.gain_snapshot[ids]).abs().reshape(ids.numel(), -1).amax(dim=1)
        return out

    # ------------------------------------------------------------------ sensor stage
    def sensor(self, env, ang_vel, roll_pitch, lin_vel_xy, joint_pos_delta, joint_vel, contact_fill):
        """The six raw channel blocks -> the same six, faulted (new tensors; the inputs are not modified)."""
        self.register("obs")
        if self.kind not in SENSOR_KINDS:
            return ang_vel, roll_pitch, lin_vel_xy, joint_pos_delta, joint_vel, contact_fill
        k = self._k(env)
        act, pre, post = self._masks(k)
        new = k != self._last_obs_k
        self._last_obs_k = k.clone()
        ins = (ang_vel, roll_pitch, lin_vel_xy, joint_pos_delta, joint_vel, contact_fill)
        av, rp, vxy, jpd, jv, cf = ins
        a1 = act[:, None]
        kind = self.kind

        def hold(name, raw):
            h = self.holds.get(name)
            if h is None:
                h = torch.zeros_like(raw)
            h = torch.where(pre[:, None], raw, h)
            self.holds[name] = h
            out = torch.where(a1, h, raw)
            self._pmax(f"{name}_held_dev", act, _rowmax(out - h))      # 0: the channel is constant while active
            self._pmax(f"{name}_raw_move", act, _rowmax(raw - h))      # > 0: the real value moved meanwhile
            return out

        if kind == "S1":
            vxy = torch.where(a1, torch.zeros_like(vxy), vxy)
            self._pmax("vxy_absmax", act, _rowmax(vxy))
        if kind in ("S2", "Z1"):
            vxy = hold("vxy", vxy)
        if kind in ("S5", "Z1"):
            av = hold("ang_vel", av)
            rp = hold("roll_pitch", rp)
        if kind in ("S3", "S6"):
            b = float(self.params["bias"])
            bias = torch.stack((torch.cos(self.heading), torch.sin(self.heading)), dim=-1) * b
            raw = vxy if kind == "S3" else rp
            out = raw + torch.where(a1, bias.to(raw.dtype), torch.zeros_like(raw))
            self._pmax("bias_dev", act, _rowmax(out - raw - bias))
            self._pset("heading_rad", act, self.heading)
            if kind == "S3":
                vxy = out
            else:
                rp = out
        if kind == "S4":
            vxy = vxy + torch.where(a1, self.walk.to(vxy.dtype), torch.zeros_like(vxy))
            ten_s = act & (k == self.start + int(round(10.0 / self.step_dt)) - 1)
            self._pset("walk_10s", ten_s, self.walk, width=2)
            self._pset("walk_last", act, self.walk, width=2)
        if kind == "S8":
            if self.params.get("mode") == "zero":
                cf = torch.where(a1, torch.zeros_like(cf), cf)
                self._pmax("cf_absmax", act, _rowmax(cf))
            else:
                cf = hold("contact_fill", cf)
        if kind == "S12":
            raw7 = torch.cat((av, rp, vxy), dim=-1)
            if self.s12_prev is None:
                self.s12_prev = torch.zeros_like(raw7)
            refresh = (k <= 0) | (torch.div(3 * k, 5, rounding_mode="floor")
                                  != torch.div(3 * (k - 1), 5, rounding_mode="floor"))
            keep = act & ~refresh
            out7 = torch.where(keep[:, None], self.s12_prev, raw7)
            self._pmax("s12_hold_dev", keep, _rowmax(out7 - self.s12_prev))
            self._pcount("s12_refresh_n", act & refresh & new)
            self._pcount("s12_hold_n", keep & new)
            self.s12_prev = torch.where(act[:, None], out7, self.s12_prev)
            av, rp, vxy = out7[:, 0:3], out7[:, 3:5], out7[:, 5:7]
        if kind == "D1":
            term = self.term
            clipped = term._action_history_buf[:, -1]
            scale = term._scale if isinstance(term._scale, torch.Tensor) else torch.full_like(clipped, float(term._scale))
            offset = term._offset if isinstance(term._offset, torch.Tensor) else torch.full_like(clipped, float(term._offset))
            cols = self.d1_cols
            tgt = clipped[:, cols] * scale[:, cols] + offset[:, cols]
            jids = self.d1_joint_ids
            want = tgt - term._asset.data.default_joint_pos[:, jids]
            jpd = jpd.clone()
            jpd[:, jids] = torch.where(a1, want.to(jpd.dtype), jpd[:, jids])
            jv = torch.where(a1, torch.zeros_like(jv), jv)
            cf = torch.where(a1, torch.zeros_like(cf), cf)
            self._pmax("d1_jpd_dev", act, _rowmax(jpd[:, jids] - want))
            self._pmax("d1_jv_cf_absmax", act, torch.maximum(_rowmax(jv), _rowmax(cf)))
        outs = (av, rp, vxy, jpd, jv, cf)
        dev = torch.stack([_rowmax(o - i) for o, i in zip(outs, ins)], dim=1).amax(dim=1)
        self._devs("obs", pre, post, dev)
        self._pmax("obs_change_max", act, dev)
        self._pcount("obs_active_n", act & new)
        self._activity("obs", act, k)
        return outs

    # ------------------------------------------------------------------ depth stage
    def depth_raw(self, env, camera: str, raw: torch.Tensor) -> torch.Tensor:
        """``(N, H, W)`` raw metres of one camera at an image refresh -> the same, faulted (a new tensor)."""
        if self.kind not in DEPTH_KINDS:
            return raw
        k = self._k(env)
        act, pre, post = self._masks(k)
        a3 = act[:, None, None]
        out = raw
        if camera == FRONT_CAMERA and self.kind in ("S10", "Z1"):
            out = torch.where(a3, torch.zeros_like(raw), raw)
            self._pmax("front_raw_absmax", act, _rowmax(out))
            self._pmax("front_raw_in_max", act, _rowmax(raw))
        if camera in TOF_CAMERAS and self.kind == "S11":
            fr = self.frozen.get(camera)
            if fr is None:
                fr = torch.zeros_like(raw)
            fr = torch.where(pre[:, None, None], raw, fr)
            self.frozen[camera] = fr
            out = torch.where(a3, fr, raw)
            self._pmax(f"{camera}_frozen_dev", act, _rowmax(out - fr))
            self._pmax(f"{camera}_raw_move", act, _rowmax(raw - fr))
        dev = _rowmax(out - raw)
        self._devs(f"depth_{camera}", pre, post, dev)
        self._pmax(f"depth_{camera}_change_max", act, dev)
        self._pcount(f"depth_{camera}_active_n", act)
        self._activity(f"depth_{camera}", act, k)
        return out

    def depth_image(self, env, camera: str, image: torch.Tensor) -> None:
        """Probe of the processed image (S10 / Z1 front: -0.5 everywhere while active)."""
        if camera == FRONT_CAMERA and self.kind in ("S10", "Z1"):
            act, _, _ = self._masks(self._k(env))
            self._pmax("front_img_dev", act, _rowmax(image + 0.5))

    # ------------------------------------------------------------------ action stage
    def on_actions(self, term) -> None:
        """After the delayed read in ``process_actions`` (``term._raw_actions`` = the applied action, pre-scale)."""
        if not self._checked:
            self._check_bound()
        env = term._env
        k = self._k(env)
        act, pre, post = self._masks(k)
        n = self.num_envs
        kind = self.kind
        if kind == "S4":                                            # one draw per env step, for every env
            inc = torch.randn((n, 2), generator=self.dose_gen) * float(self.params["sigma"])
            nxt = self.armed & (k + 1 >= self.start) & (k + 1 < self.end)
            self.walk = self.walk + torch.where(nxt[:, None], inc.to(self.device), torch.zeros_like(self.walk))
        if kind == "A2":
            last_pre = self.armed & (k == self.start - 1)
            if self.gain_snapshot is not None and bool(last_pre.any()):
                ids = last_pre.nonzero().flatten()
                self._pmax("a2_gain_pre_onset_dev", last_pre, self._gain_dev(term, ids))
                self._pcount("a2_gain_pre_onset_reads", last_pre)
            self._a2_write(term, k)
        if kind not in ACTION_KINDS:
            return
        raw_in = term._raw_actions.clone()
        if self.prev_applied is None:
            self.prev_applied = torch.zeros_like(raw_in)
        out = raw_in
        if kind == "B1":
            out = torch.where(act[:, None], self.prev_applied, raw_in)
            self._pcount("repeat_n", act)
        elif kind == "B2":
            u = torch.rand(n, generator=self.dose_gen).to(self.device)
            rep = act & (u < float(self.params["p"]))
            out = torch.where(rep[:, None], self.prev_applied, raw_in)
            self._pcount("repeat_n", rep)
        elif kind == "B3":
            dd = torch.randint(1, 4, (n,), generator=self.dose_gen).to(self.device)
            hist = term._action_history_buf
            idx = hist.shape[1] - 1 - dd
            gathered = hist[torch.arange(n, device=self.device), idx]
            out = torch.where(act[:, None], gathered, raw_in)
            for d in (1, 2, 3):
                self._pcount(f"delay{d}_n", act & (dd == d))
        elif kind == "B5":
            hist = term._action_history_buf
            out = torch.where(act[:, None], hist[:, hist.shape[1] - 1 - int(self.params["delay"])], raw_in)
        term._raw_actions[:] = out
        self.prev_applied = out.clone()
        dev = _rowmax(out - raw_in)
        self._devs("act", pre, post, dev)
        self._pmax("act_change_max", act, dev)
        self._pcount("act_active_n", act)
        self._activity("act", act, k)

    # ------------------------------------------------------------------ rods (physics substeps)
    def rod_dlam(self, term, which: str, lam_cmd, rod_len, dlam, max_step):
        """One rod family's clamped slew ``(N, 6)`` -> the slew actually integrated (``which``: ``hip`` / ``knee``)."""
        kind = self.kind
        if kind not in ROD_KINDS:
            return dlam
        global _LIMITS
        if _LIMITS is None:
            from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as _lk  # noqa: PLC0415

            _LIMITS = {"hip": (_lk.HIP_LEN_MIN_M, _lk.HIP_LEN_MAX_M), "knee": (_lk.KNEE_LEN_MIN_M, _lk.KNEE_LEN_MAX_M)}
        k = self._k(term._env)
        act, pre, post = self._masks(k)
        a1 = act[:, None]
        out = dlam
        if kind == "B4":
            last = self.last_dlam.get(which)
            if last is None:
                last = torch.zeros_like(dlam)
            lo, hi = _LIMITS[which]
            ext = (rod_len + last).clamp(lo, hi) - rod_len
            out = torch.where(a1, ext, dlam)
            clipped = (ext - last).abs() > 1e-6          # at a window limit (float32 rounding is ~1e-7 m)
            self._pmax(f"b4_{which}_slew_dev", act, _rowmax(torch.where(clipped, torch.zeros_like(ext), ext - last)))
            self._pcount(f"b4_{which}_clipped_n", act & clipped.any(dim=1))
            self._pmax(f"b4_{which}_last_slew_absmax", act, _rowmax(last))
            self.last_dlam[which] = torch.where(a1, last, out)
        elif kind == "B6":
            out = torch.where(a1, torch.zeros_like(dlam), dlam)
            self._pmax(f"b6_{which}_dlam_absmax", act, _rowmax(out))
        else:                                                      # A1 / A3: one leg
            c = self.leg_col
            col = dlam[:, c]
            if kind == "A1":
                new_col = torch.where(act, torch.zeros_like(col), col)
            else:
                ms = (torch.zeros_like(dlam) + max_step)[:, c]
                scale = float(self.params["slew_scale"])
                lim = ms * scale
                want = torch.maximum(torch.minimum(lam_cmd[:, c] - rod_len[:, c], lim), -lim)
                new_col = torch.where(act, want, col)
                ratio = new_col.abs() / ms.clamp_min(1e-12)
                self._pmax(f"a3_{which}_slew_ratio_max", act, ratio)
                self._pcount(f"a3_{which}_at_cap_n", act & (ratio >= scale * (1.0 - 1e-5)))
            out = dlam.clone()
            out[:, c] = new_col
            l0 = self.len0.get(which)
            if l0 is None:
                l0 = torch.full((self.num_envs,), float("nan"), device=self.device)
            l0 = torch.where(act & torch.isnan(l0), rod_len[:, c], l0)
            self.len0[which] = l0
            drift = (rod_len[:, c] + new_col - l0).abs()
            self._pmax(f"{kind.lower()}_{which}_dlam_absmax", act, new_col.abs())
            self._pmax(f"{kind.lower()}_{which}_len_drift", act, torch.nan_to_num(drift, nan=0.0))
        dev = _rowmax(out - dlam)
        self._devs(f"rod_{which}", pre, post, dev)
        self._pmax(f"rod_{which}_change_max", act, dev)
        self._pcount(f"rod_{which}_active_n", act)
        self._activity(f"rod_{which}", act, k)
        return out

    # ------------------------------------------------------------------ cams (physics substeps)
    def cam_cmd(self, term, cam_cmd):
        """The cam velocity command ``(N, 6)`` before the acceleration slew (B4 keeps the last one, B6 zeroes)."""
        if self.kind not in CAM_CMD_KINDS:
            return cam_cmd
        k = self._k(term._env)
        act, pre, post = self._masks(k)
        a1 = act[:, None]
        if self.kind == "B4":
            if self.last_cam_cmd is None:
                self.last_cam_cmd = torch.zeros_like(cam_cmd)
            out = torch.where(a1, self.last_cam_cmd, cam_cmd)
            self._pmax("b4_cam_cmd_dev", act, _rowmax(out - self.last_cam_cmd))
            self._pmax("b4_cam_last_absmax", act, _rowmax(self.last_cam_cmd))
            self.last_cam_cmd = torch.where(a1, self.last_cam_cmd, out)
        else:
            out = torch.where(a1, torch.zeros_like(cam_cmd), cam_cmd)
            self._pmax("b6_cam_cmd_absmax", act, _rowmax(out))
        dev = _rowmax(out - cam_cmd)
        self._devs("cam", pre, post, dev)
        self._activity("cam", act, k)
        return out

    def cam_applied(self, term) -> None:
        """A4: the stuck leg's slewed cam velocity target set to 0 (in place on ``term._cam_vel_applied``)."""
        if self.kind not in CAM_APPLIED_KINDS:
            return
        k = self._k(term._env)
        act, pre, post = self._masks(k)
        c = self.cam_col
        applied = term._cam_vel_applied
        before = applied[:, c].clone()
        applied[:, c] = torch.where(act, torch.zeros_like(before), before)
        data = term._asset.data
        theta = data.joint_pos[:, self.cam_joint_id]
        if self.cam_theta0 is None:
            self.cam_theta0 = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.cam_theta0 = torch.where(act & torch.isnan(self.cam_theta0), theta, self.cam_theta0)
        self._pmax("a4_cam_target_absmax", act, applied[:, c].abs())
        self._pmax("a4_cam_joint_vel_absmax", act, data.joint_vel[:, self.cam_joint_id].abs())
        self._pmax("a4_cam_angle_drift", act, torch.nan_to_num((theta - self.cam_theta0).abs(), nan=0.0))
        dev = (applied[:, c] - before).abs()
        self._devs("cam", pre, post, dev)
        self._activity("cam", act, k)

    # ------------------------------------------------------------------ A2: a broken linkage (PhysX gains)
    @staticmethod
    def _read_gains(asset, env_ids, joint_ids) -> torch.Tensor:
        """``(n, 3, 2)`` PhysX read-back [stiffness, damping, max force] x [hip, knee] (CPU -> float)."""
        view = asset.root_physx_view
        cpu_ids = env_ids.cpu()
        rows = [view.get_dof_stiffnesses(), view.get_dof_dampings(), view.get_dof_max_forces()]
        return torch.stack([r[cpu_ids][:, joint_ids].float() for r in rows], dim=1)

    def _actuators(self, asset):
        out = []
        for name, jid in ((HIP_ACTUATOR, self.leg_joint_ids[0]), (KNEE_ACTUATOR, self.leg_joint_ids[1])):
            act = asset.actuators[name]
            ji = act.joint_indices
            if isinstance(ji, slice):
                local = jid
            else:
                lst = ji.tolist() if isinstance(ji, torch.Tensor) else list(ji)
                local = lst.index(jid)
            out.append((act, local))
        return out

    def _a2_write(self, term, k) -> None:
        ids = (self.armed & (k == self.start) & ~self.gain_written).nonzero().flatten()
        if ids.numel() == 0:
            return
        asset = term._asset
        jids = list(self.leg_joint_ids)
        pre = self._read_gains(asset, ids, jids).to(self.device)
        self.saved_physx[ids] = pre
        for i, (act, local) in enumerate(self._actuators(asset)):
            self.saved_act[ids, 0, i] = act.stiffness[ids, local].float()
            self.saved_act[ids, 1, i] = act.damping[ids, local].float()
            self.saved_act[ids, 2, i] = act.effort_limit[ids, local].float()
            act.stiffness[ids, local] = 0.0
            act.damping[ids, local] = 0.0
            act.effort_limit[ids, local] = 0.0
        zeros = torch.zeros(ids.numel(), 2, device=self.device)
        asset.write_joint_stiffness_to_sim(zeros, joint_ids=jids, env_ids=ids)
        asset.write_joint_damping_to_sim(zeros, joint_ids=jids, env_ids=ids)
        asset.write_joint_effort_limit_to_sim(zeros, joint_ids=jids, env_ids=ids)
        post = self._read_gains(asset, ids, jids).to(self.device)
        self.frozen_tgt[ids] = asset.data.joint_pos_target[ids][:, jids]
        self.gain_written[ids] = True
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        full_pre = torch.zeros(self.num_envs, 6, device=self.device)
        full_post = torch.zeros(self.num_envs, 6, device=self.device)
        full_pre[ids] = pre.reshape(ids.numel(), 6)
        full_post[ids] = post.reshape(ids.numel(), 6)
        self._pset("a2_gain_pre", mask, full_pre, width=6)       # [kp_h, kp_k, kd_h, kd_k, fmax_h, fmax_k]
        self._pset("a2_gain_post", mask, full_post, width=6)
        self._pset("a2_write_step", mask, k.float())

    def _restore_gains(self, term, ids: torch.Tensor) -> None:
        if self.kind not in GAIN_KINDS:
            return
        sel = ids[self.gain_written[ids]]
        if sel.numel() == 0:
            return
        asset = term._asset
        jids = list(self.leg_joint_ids)
        saved = self.saved_physx[sel]
        asset.write_joint_stiffness_to_sim(saved[:, 0], joint_ids=jids, env_ids=sel)
        asset.write_joint_damping_to_sim(saved[:, 1], joint_ids=jids, env_ids=sel)
        asset.write_joint_effort_limit_to_sim(saved[:, 2], joint_ids=jids, env_ids=sel)
        for i, (act, local) in enumerate(self._actuators(asset)):
            act.stiffness[sel, local] = self.saved_act[sel, 0, i].to(act.stiffness.dtype)
            act.damping[sel, local] = self.saved_act[sel, 1, i].to(act.damping.dtype)
            act.effort_limit[sel, local] = self.saved_act[sel, 2, i].to(act.effort_limit.dtype)
        self.gain_written[sel] = False

    def after_linkage(self, term) -> None:
        """A2: the broken leg's hip / knee targets stay frozen at their onset values (velocity target 0)."""
        if self.kind not in GAIN_KINDS:
            return
        k = self._k(term._env)
        act, _, _ = self._masks(k)
        act = act & self.gain_written
        asset = term._asset
        data = asset.data
        leave = torch.zeros(self.num_envs, device=self.device)
        for i, jid in enumerate(self.leg_joint_ids):
            pos_t = torch.where(act, self.frozen_tgt[:, i], data.joint_pos_target[:, jid])
            vel_t = torch.where(act, torch.zeros_like(pos_t), data.joint_vel_target[:, jid])
            asset.set_joint_position_target(pos_t[:, None], joint_ids=[jid])
            asset.set_joint_velocity_target(vel_t[:, None], joint_ids=[jid])
            leave = torch.maximum(leave, (data.joint_pos[:, jid] - self.frozen_tgt[:, i]).abs())
        self._pmax("a2_target_leave_max", act, leave)
        self._pmax("a2_applied_torque_absmax", act, _rowmax(data.applied_torque[:, self.leg_joint_ids]))
        self._activity("gain", act, k)

    # ------------------------------------------------------------------ sidecar
    def _flush(self) -> None:
        if self.wave < 0 or self.wave in self._flushed:
            return
        self._flushed.add(self.wave)
        to = lambda t: t.detach().cpu().tolist()  # noqa: E731
        firsts = [v for kk, v in self.p.items() if kk.startswith("first_")]
        lasts = [v for kk, v in self.p.items() if kk.startswith("last_")]
        if firsts:
            f = torch.stack(firsts)
            f = torch.where(f < 0, torch.full_like(f, NEVER), f).amin(dim=0)
            first = torch.where(f >= NEVER, torch.full_like(f, -1), f)
            last = torch.stack(lasts).amax(dim=0)
        else:
            first = last = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        probes = {}
        for name, t in sorted({**self.p_wave_start, **self.p}.items()):
            vals = to(t)
            if t.dtype.is_floating_point:
                vals = ([[_round(x) for x in row] for row in vals] if t.dim() == 2 else [_round(x) for x in vals])
            probes[name] = vals
        rec = {
            "wave": self.wave,
            "onset": to(self.onset),
            "start": to(self.start),
            "planned_last": None if self.duration is None or self.from_start else to(self.end - 1),
            "first_active": to(first),
            "last_active": to(last),
            "fell_at": to(self.fell_at),
            "probes": probes,
        }
        if self.kind in ("S3", "S6"):
            rec["heading_rad"] = [_round(x) for x in to(self.heading)]
        rec["probe_kind"] = {name: probe_kind(name) for name in sorted(probes)}
        rec["state_cleared"] = bool(self.state_cleared and self.state_cleared[-1])
        if self.kind in DEPTH_KINDS:
            rec.update(self._policy_visible())
        self.records.append(rec)

    def _policy_visible(self) -> dict:
        """Depth cells: when the last faulted frame leaves what the policy sees. The term returns the image ``lag``
        refreshes old and the student re-encodes on the same refresh cadence (latent held between), so a frame
        captured at refresh step r is policy-visible through ``r + period * (lag + 1) - 1``."""
        cams = (FRONT_CAMERA,) if self.kind in ("S10", "Z1") else TOF_CAMERAS
        lasts = [self.p[f"last_depth_{c}"] for c in cams if f"last_depth_{c}" in self.p]
        span = self.depth_period * (self.depth_lag + 1)
        out = {"depth_visible_span_steps": span}
        if self.duration is None or self.from_start:
            out["policy_visible_last"] = None
            out["policy_visible_bound"] = None
            return out
        r = torch.stack(lasts).amax(dim=0) if lasts else torch.full((self.num_envs,), -1, device=self.device)
        vis = torch.where(r >= 0, r + span - 1, torch.full_like(r, -1))
        out["policy_visible_last"] = vis.long().cpu().tolist()       # from the realised last faulted refresh
        out["policy_visible_bound"] = (self.end - 1 + span).cpu().tolist()   # planned_last + span (upper bound)
        return out

    def sidecar(self, extra: dict | None = None) -> dict:
        """The ``fault_onsets.json`` document. Flushes the current wave (call once, after the last step).

        Schema ``fault_onsets/v1``: ``fault`` (the cfg spec), ``kind``, ``duration_steps`` (None = permanent),
        ``dt``, ``num_envs``, ``onset`` (window s / steps, seed, rule), ``dose_rng``, ``binding`` (leg / joints /
        bound stages), ``waves`` -- one record per wave, columnar over env index (episode = wave * num_envs + env):
        ``onset`` (table step), ``start`` (first step the fault may act: onset, or 0 for an anchor),
        ``planned_last`` (onset + d - 1, transients), ``first_active`` / ``last_active`` (steps the fault actually
        acted at, any stage; -1 none), ``fell_at`` (episode step of a mid-wave reset, -1 none; the env is disarmed
        after it), ``probes`` (per-stage ``<stage>_pre_dev`` / ``_post_dev`` = max |faulted - nominal| before the
        onset / after a transient cleared, both 0 by contract; ``first_/last_<stage>``; ``*_active_n`` counts; and
        the dose probes of the cell: ``*_held_dev`` 0 for a hold, ``*_raw_move`` the real channel's excursion,
        ``vxy_absmax`` / ``cf_absmax`` 0 for a zeroing, ``bias_dev``, ``walk_10s`` / ``walk_last`` (S4 walk vector),
        ``s12_*``, ``d1_*``, ``front_raw_absmax`` / ``front_img_dev`` 0 (S10/Z1), ``<cam>_frozen_dev`` 0 and
        ``<cam>_raw_move`` (S11), ``repeat_n`` / ``delay<d>_n`` (B), ``b4_*_slew_dev`` 0, ``b6_*`` 0,
        ``a1_*_dlam_absmax`` 0 / ``*_len_drift``, ``a3_*_slew_ratio_max`` <= 0.3, ``a4_cam_*``, ``a2_gain_pre`` /
        ``a2_gain_post`` (PhysX read-back [kp_h, kp_k, kd_h, kd_k, fmax_h, fmax_k] before / after the write) and
        ``a2_target_leave_max``; ``a2_gain_pre_onset_dev`` / ``a2_gain_wave_start_dev`` = PhysX read-back at
        k = onset - 1 / at each wave start (after the restore) vs the construction snapshot, 0 by contract). Per wave
        also: ``probe_kind`` (probe -> self_check / sim / model_estimate / dose / activity, :func:`probe_kind`),
        ``state_cleared`` (every fault state tensor at its cleared value at the wave start; a failure raises), and on
        depth cells ``depth_visible_span_steps``, ``policy_visible_last`` (realised last faulted refresh + span - 1,
        -1 none; None permanent) and ``policy_visible_bound`` (planned_last + span): read a depth transient's "off
        again" from these, not from planned_last. Floats at 6 significant digits.
        """
        self._flush()
        lo_s, hi_s = (float(v) for v in self.spec["onset_s"])
        doc = {
            "schema": SIDECAR_SCHEMA,
            "fault": self.spec,
            "kind": self.kind,
            "duration_steps": self.duration,
            "dt": self.step_dt,
            "num_envs": self.num_envs,
            "env_seed": None if self.env_seed is None else int(self.env_seed),
            "onset": {"window_s": [lo_s, hi_s], "window_steps": list(self.window_steps),
                      "seed": onset_seed(self.env_seed), "salt": hex(ONSET_SEED_SALT), "waves_drawn": WAVES_MAX,
                      "rule": "lo_steps + floor(u * (hi_steps - lo_steps)), u ~ U[0,1) float64, one (waves, N) "
                              "table drawn at construction; step k = env.episode_length_buf (the policy step that "
                              "consumes the observation / applies the action)"},
            "dose_rng": {"salt": hex(DOSE_SEED_SALT),
                         "mix": hex(DOSE_SEED_MIX),
                         "rule": "CPU generator re-seeded (env_seed * mix + salt + wave) % 2**63 at each wave start"},
            "wave_rule": "wave = number of explicit resets of all envs - 1 (in-step resets: env.reset_buf[ids] all and "
                         "episode_length_buf[ids] > 0); asserted against the harness at each wave start",
            "binding": dict(self.binding, stages=sorted(self.bound)),
            "waves": self.records,
        }
        if extra:
            doc.update(extra)
        return doc

    def write_sidecar(self, path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.sidecar(extra), separators=(",", ":")) + "\n")
        return path

    def write_sidecar_into(self, run_dir, harness: dict) -> dict:
        """The harness's call: ``<run_dir>/fault_onsets.json`` (with ``harness`` recorded), and the ``run_meta``
        summary that points at it."""
        path = self.write_sidecar(Path(run_dir) / SIDECAR_NAME, {"harness": dict(harness)})
        return {"sidecar": path.name, "schema": SIDECAR_SCHEMA, "knob": self.spec.get("knob"), "kind": self.kind,
                "duration_steps": self.duration, "onset_s": list(self.spec.get("onset_s") or []),
                "from_start": self.from_start, "waves_recorded": len(self.records)}
