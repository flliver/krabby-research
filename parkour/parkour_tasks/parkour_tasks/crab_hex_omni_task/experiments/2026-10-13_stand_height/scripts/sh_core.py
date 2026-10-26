#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``) -- the CORE: the ONE implementation of the height-commanded stand
semantics that the trainer, the eval launcher, the table / ground test and the readers all import (PLAN.md H1-H6,
step 2). Pure tensor code; nothing here touches Isaac, an env or a file other than the frozen facts and checkpoints.

LAZY IMPORTS ONLY. The eval launcher loads this module before Kit starts: at import it reads only the stdlib, the frozen
block of ``facts.json`` and the predecessor's torch-free ``sn_common`` (by path, no bytecode). torch, numpy, R's module
and the posture map ``sh_kin`` (numpy) load on the first call that needs them (post-Kit in the launcher).

WHAT IS HERE (each item is the plan's rule; the API list at the bottom of this docstring is what builders import):

1. **Posture map** ``q(h)`` (H1) as torch tensors: ``sh_kin.q`` tabulated on a 0.5 mm grid whose nodes include 1.064 m,
   linearly interpolated in float64 (interpolation error < 2e-6 action units), ``h`` clamped to the frozen
   [H_MIN, H_MAX] first (ground starts measure 0.31 / 0.52 m), ``q(1.064) == (0, 0)`` exactly (a node; also any ``h``
   within :data:`Q_SNAP_M` of 1.064, so a float32 1.064 maps to exactly 0 as well).
2. **R_h** (H3): ``restance_target`` with the D3 constants on the observation whose hip-pitch columns (29-34) are
   re-referenced to ``q(h)`` (``p - 0.25 * q_hip``; R reads no knee column -- pinned by the tests -- so the knee
   re-reference is a no-op and is not materialised); HOLD / WAIT / SEAT -> ``q``; LIFT / HOME -> ``q`` + R's D3 lift
   (hip ``lift``, knee ``-rho * side * lift`` relative to ``q``); cams = R's cam law unchanged; rods clamped to
   +-4.8 (cams +-1). With ``h = 1.064`` and ``clip = 1`` it is ``restance_target`` BITWISE (an exactly-zero offset is
   never added, so even the sign of R's zeros survives).
3. **Rate limiter** (H2): 0.08 m/s on a per-env float64 state that snaps exactly onto its target; init at the command
   for walk-handover starts (a 1.064 command stays bitwise inert) and at ``clamp(h_meas, H_MIN, H_MAX)`` for table /
   ground / random-height starts.
4. **h_meas** (H2): root z minus the mean terrain height under the six feet (= p2s body height = the harness's
   ``root_pos_w[:, 2] - mean(terrain_z)``); :func:`terrain_z_under_feet` is the harness's raycast with the raycast
   function injected (and it refuses a missing mesh: the harness's ``root_z - 1.05`` fallback would make h_meas a
   constant).
5. **Stand-net input** (H2, H6): ``sn_common.stand_net_input`` (estimate on the UNMASKED prop, then the NEW tensor, then
   the mask on it) and then, on that clone only, column 15 = ``(h_cmd_limited - 1.064) / 0.4`` on every row and, on
   restore rows, current-frame columns 5, 8, 9, 10 (yaw-rate command, xy command, its norm) = 0. The caller's obs is
   never written. (A stand row's column 5 carries the heading relay's yaw rate -- STAND_HOLD is off in every run of
   this lineage -- while a restore row shows 0: the plan's choice, noted here, not changed.)
6. **Router** (H1): ``a_s1 = clamp(S1, +-1)`` (the runtime wrapper clip is 4.8 in the stand-net harnesses, so S1's
   +-1 is explicit here); ``a_stand = q(h_cmd_limited) + clamp(a_net, +-1)``, rods clamped to the env's raw +-4.8,
   cams to +-1 (q has 0 cams); rows select by the EFFECTIVE stand vector ``stand_slot | restoring``.
   **Residual deadband** (CHARTER section 3, 2026-10-14 "USER DECISION at the step-6 stop: probe a residual
   deadband"; OFF by default): ``stand_action(..., deadband=d)`` -- after the +-1 clamp and before ``+ q``, on the 6 hip
   and 6 knee columns only (cams never), ``r <- r if |r| >= d else 0`` (hard, compared in the residual's dtype).
   ``d = 0`` skips the op: the action path is bitwise the one without the option. It acts on every row the stand
   action composes, so restore rows (stand rows, H6) get it too; S1's rows never do. Records carry the value under
   :data:`RESIDUAL_DEADBAND_KEY`; a record without it ran at 0 (:func:`recorded_deadband`). The step-6 probe value is
   :data:`RESIDUAL_DEADBAND_PROBE` = 0.25 (the midpoint of R_h's binary 0 / 0.5 hip / knee labels). An eval record also
   carries the deadband its stand checkpoint was TRAINED with (:data:`TRAINED_DEADBAND_KEY`); the readers compare the
   two (:func:`deadband_parity`) and NOTE a difference, never refuse it (the step-6 probe evaluates a head trained at 0
   with 0.25 on purpose; a head trained at d and evaluated without it is the config-parity break the note surfaces).
7. **H6 restore state machine** (:class:`HeightController`, terrain-independent): engage when the slot leaves stand
   while the limited command is > 2 cm from 1.064; drive the command to 1.064 through the same limiter; complete when
   the limiter has reached 1.064 AND every hip rod (from the MEASURED hip angle, joint space) is within
   :data:`RESTORE_ROD_TOL_M` of rod(q(1.064)) for 0.2 s; timeout ``(1.064 - H_MIN) / 0.08 + 1 s``; a stand arriving
   mid-restore is a stand->stand merge. Events per step: engaged / completed / timed_out / merged (+ reset cuts).
8. **Warm start** (H4): A-mask s17 (``policy/stand_net/model_4000.pt``, sha256 pinned) with the actor backbone's
   layer-0 column 15 zeroed (``W[:, 15]``, norm 1.74 before); everything else (critic, std, estimator) unchanged.
   That default is HISTORICAL: A-mask s17 is superseded (``policy/manifest.yaml`` ``superseded:`` SN_Amask_s17, its
   file kept in place) and is not the stand network of record; see :data:`STAND_NET_CKPT`.
9. **Residual label** (H4): ``R_h - q(h)``, refused (ValueError) if any element exceeds 1 (+ :data:`LABEL_TOL`) -- R_h
   lies within ``q +- 1`` on hip / knee (lift 0.5) and R's cams are within +-1, so the clamped residual can express
   the teacher exactly.

THE RESTORE ROD TOLERANCE (delta, H6; PLAN left it open). **Re-pinned at step 4 to ``RESTORE_ROD_TOL_M = 0.008``**
(CHARTER section 3, 2026-10-13): the smallest whole mm >= the p99 of R's own settled walking-height stands' max-leg
hip-rod deviation (best 0.2-s window per hold; P2R_RSTg_s18 walkstop es1 7.16 mm / es2 6.75 mm, max 8.02 mm). At 4 mm
R_h's launcher smoke (hwalkstop es1) timed out 119 / 600 restores, every one settled in HOLD with one loaded leg at
4.1-4.9 mm of PD sag; the 4-mm band sits inside R's normal sag (R > 4 mm on 16-20 % of settled walkstop holds), and
the rod reading cannot tell a lifting leg from a planted one anyway (R_h LIFT rows: p50 2.9 mm). Replayed on the
recorded smoke: 4 mm reproduces 481 / 119 exactly; 8 mm completes 600 / 600 (median 2.9 s). The original 4-mm note
follows. ``RESTORE_ROD_TOL_M = 0.004`` m of hip rod on
every leg, measured from the joint angle (``hip_len(default + obs[:, 29 + i])``) against ``hip_len(default)`` (=
rod(q(1.064)) = 0.550 m). Why 4 mm and not the 2 mm the build brief suggested -- measured on the four walkstop stamps
of the step-1 force facts (A-mask s17 es1, R = P2R_RSTg_s18 es1 / es2, 50 episodes each, stand rows held > 3 s at the
walking height itself, no height command involved):

* a loaded hip sags under its PD (16000 N*m/rad, tau_hip ~ 160-190 N*m): per-leg mean hip-rod offset -1.2 ... -1.9
  mm; the max over six legs has p50 2.4 mm (A-mask) / 2.9 mm (R), p90 3.3 / 4.7-5.9 mm (R's tail = its re-steps);
* at 2 mm only 7-24 % of those settled rows qualify and R reaches the 10-step (0.2 s) completion in 10-20 of 150 stands
  -- the restore would time out on robots standing still at walking height (a false K4 failure, not a policy one);
  at 4 mm: A-mask 150 / 150, R 125-134 / 150; at 5 mm: R 138-144 / 150;
* 4 mm of hip rod is ~2.0 cm of root height at 1.064 (1 mm rod ~ 5.1 mm height: dq_hip/dh 6.7 units/m x 0.25 rad
  x dL/dtheta 0.117 m/rad), i.e. the same 2 cm band whose violation ENGAGES the restore, and it is below R's DOWN band
  (|p_down| = 0.05 rad = 5.8 mm of rod), so a leg R would not call DOWN (lifted, mid re-step) can never complete.

API (what the builders import; tensor functions take / return device tensors and never sync the host, except
``residual_label`` / ``rh_label`` with ``check=True`` and ``HeightController.counts``):

constants  H_SETTLED, H_MIN, H_MAX, FROZEN, HEIGHT_OBS_SCALE, HEIGHT_RATE_M_S, HEIGHT_COL, RESTORE_ZERO_COLS,
           RAW_CLIP, CAM_CLIP, S1_CLIP, RESIDUAL_CLIP, HARNESS_CLIP, RESTORE_ENGAGE_M, RESTORE_SETTLE_S,
           RESTORE_ROD_TOL_M, RESTORE_TIMEOUT_S, START_KINDS, STAND_NET_CKPT, STAND_NET_SHA256, SN (sn_common),
           RESIDUAL_DEADBAND_PROBE, RESIDUAL_DEADBAND_KEY, DEADBAND_ACT
q-map      clamp_h(h) / q_map(h, *, dtype, device) -> (q_hip, q_knee_left) / q18(h, *, dtype, device) -> (N, 18) /
           q_table() -> (grid, q_hip, q_knee_left) float64 numpy / encode_height(h, *, dtype, device)
R_h        rh_target(obs, h_cmd_limited, params=None, *, clip=RAW_CLIP) -> (target (N, 18), branch (N, 6))
           residual_label(target, h_cmd_limited, *, check=True) -> (N, 18) / label_excess(label) -> device scalar /
           rh_label(obs, h_cmd_limited, params=None, *, check=True) -> RhLabel(label, target, branch, q18)
limiter    HeightLimiter(n, device, *, dt, ...)  .state  .reset(rows, value)  .step(target) -> state
h_meas     h_meas(root_z, terrain_z6) / terrain_z_under_feet(foot_pos_w, mesh, raycast_fn)
input      stand_net_input(obs, estimator, *, p0, p1, num_prop, mask, h_cmd_limited, restoring=None) -> x
router     s1_clamp(a) / stand_action(a_net, h_cmd_limited, *, deadband=0.0) -> (a_stand, q18) /
           route(stand_eff, a_stand, a_s1) / router_actions(stand_eff, a_net, a_s1, h_cmd_limited, *, deadband=0.0)
           -> RouterOut
deadband   check_deadband(d) -> float / apply_deadband(r, d) (torch) / apply_deadband_np(r, d) (numpy twin) /
           recorded_deadband(record) -> float (missing = 0.0) / recorded_trained_deadband(record) -> float | None /
           deadband_parity(record) -> {"evaluated", "trained", "note"}
restore    HeightController(n, device, *, dt, ...)  .reset(rows, start, *, h_cmd=None, h_meas=None) -> cut mask
           .force_restore(rows)  .step(stand_slot, h_cmd, obs) -> HeightStep  .counts() -> dict
           hip_rod_dev(obs) -> (N, 6) m / hip_rods_ok(obs, tol) -> (N,) bool
warm start zero_height_column(model_state_dict) -> (new_sd, norm_before) / warm_start_state(path, *, sha256) -> dict
           / load_warm_start(policy, path, *, sha256) -> record
records    clip_record(deadband=0.0) / provenance(deadband=0.0) -> dict (shas + constants for run_meta)
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: scripts -> 2026-10-13_stand_height
CAMPAIGN = HERE.parent
#: scripts -> 2026-10-13_stand_height -> experiments -> crab_hex_omni_task
TASK = HERE.parents[2]
REPO = HERE.parents[6]
PRED_SCRIPTS = TASK / "experiments" / "2026-10-11_stand_network" / "scripts"
SN_COMMON_PATH = PRED_SCRIPTS / "sn_common.py"
SH_KIN_PATH = HERE / "sh_kin.py"
FACTS_JSON = CAMPAIGN / "facts.json"
#: the DEFAULT warm start (PLAN H4) = A-mask s17, the stand network of record when this campaign ran (2026-10-13 ..
#: 10-15). HISTORICAL since the 2026-10-24 F19 bake: A-mask s17 is ``superseded:`` SN_Amask_s17 in policy/manifest.yaml
#: (this sha = its files[0].sha256; the file stays in stand_net/). The stand network of record is the manifest's
#: ``stand_network`` (heads.stand = f19_stand_net: SH_T4F19 model_5000, warm-started from SN_T4F19 =
#: f19_stand_net/model_4000.pt). Kept as the default on purpose -- an unset sh_dagger run is the campaign's run exactly,
#: sh_table_test's --stand-ckpt default and the amask_old checks read these two as "A-mask s17", and the tests pin
#: them. Warm-starting from another net is the explicit ``sh_dagger.py --warm-start PATH --warm-sha SHA`` (as the Task 4
#: pipeline's SH stage passed SN_T4F19), never a change here.
STAND_NET_CKPT = TASK / "policy" / "stand_net" / "model_4000.pt"
STAND_NET_SHA256 = "769cf25331402afdfb23756100ec7c215dacf984a321e90a7b39d2969e4941ff"


def _load_by_path(path: Path, name: str):
    """A private module instance loaded by path (not registered in ``sys.modules``; no ``__pycache__`` written)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_core: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the predecessor's ONE stand-net input module (torch-free at import; R loaded by path inside its functions)
SN = _load_by_path(SN_COMMON_PATH, "_sh_core_sn_common")

# ===================================================================================== constants
H_SETTLED = 1.064            #: walking height: the settled sim root height at the zero action; q(H_SETTLED) = 0
HEIGHT_OBS_SCALE = 0.4       #: stand-net input column 15 = (h_cmd_limited - H_SETTLED) / HEIGHT_OBS_SCALE
HEIGHT_RATE_M_S = 0.08       #: the h_cmd rate limiter
HEIGHT_COL = 15              #: the hidden clock column the height command is written into (stand-net input only)
#: current-frame command columns a restore row shows as a stand (5 yaw-rate cmd, 8-9 xy cmd, 10 its norm)
RESTORE_ZERO_COLS = (5, 8, 9, 10)
ROD_SCALE = 0.25             #: rad per action unit, hip and knee
NUM_LEGS = 6
ACTION_DIM = 18
SIDE = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)    #: right-leg knee action = -(left-convention value)
CAM_ACT = slice(0, 6)
HIP_ACT = slice(6, 12)
KNEE_ACT = slice(12, 18)
#: hip-pitch obs columns (joint_pos - default, rad); = crab_hex_omni_restance.HIP_POS_OBS_COLS (pinned by the tests)
HIP_POS_OBS_COLS = tuple(29 + i for i in range(NUM_LEGS))
#: R reads only the 89-wide current-step head (crab_hex_omni_restance.MIN_OBS_WIDTH)
R_OBS_WIDTH = 89
RAW_CLIP = 4.8               #: the env action term's raw rod clip under preset 2a (crab_hex_env_cfg.py:272)
CAM_CLIP = 1.0               #: the action term's cam clip
S1_CLIP = 1.0                #: S1's rows, clamped explicitly by the router (its +-1 is load-bearing)
RESIDUAL_CLIP = 1.0          #: the stand net's residual a_net
HARNESS_CLIP = 4.8           #: runtime ``env.clip_actions`` in the stand-net harnesses (no config edit)
#: the step-6 residual-deadband probe (CHARTER 2026-10-14): the midpoint of R_h's binary hip / knee labels (0 / 0.5)
RESIDUAL_DEADBAND_PROBE = 0.25
#: the record key every run / stamp / cell writes the deadband under (absent = 0: written before the option existed)
RESIDUAL_DEADBAND_KEY = "residual_deadband"
#: the key an eval record writes the deadband its stand checkpoint was TRAINED with under (the launcher's
#: run_meta stand_height block, from infos.sh_dagger; null when the checkpoint is not a campaign-trained one, absent on
#: records written before the option). Recorded and compared (:func:`deadband_parity`), never refused.
TRAINED_DEADBAND_KEY = "trained_residual_deadband"
#: the residual columns the deadband acts on: the 6 hips and the 6 knees (cams 0-5 never)
DEADBAND_ACT = slice(HIP_ACT.start, KNEE_ACT.stop)
RESTORE_ENGAGE_M = 0.02      #: H6: a restore engages when the limited command is farther than this from H_SETTLED
RESTORE_SETTLE_S = 0.2       #: H6: hip rods within RESTORE_ROD_TOL_M of rod(q(1.064)) this long (consecutive steps)
RESTORE_ROD_TOL_M = 0.008    #: H6 delta, m of hip rod on every leg (re-pinned at step 4; see the module docstring)
Q_GRID_STEP_M = 0.0005       #: q table node spacing; nodes at H_SETTLED + k * step
Q_SNAP_M = 1.0e-6            #: |h - H_SETTLED| <= this maps to q = 0 exactly (float32 1.064 is 1.05e-8 off)
LABEL_TOL = 1.0e-5           #: residual-label bound slack (float rounding of (q + r) - q)
WARM_ZERO_KEY = "actor.actor_backbone.0.weight"
START_KINDS = ("walk", "table", "ground", "random")
#: limiter init per start kind (PLAN H2): the command for walk-handover starts, the measured height otherwise
START_INIT = {"walk": "command", "table": "measured", "ground": "measured", "random": "measured"}


def _read_frozen() -> dict:
    d = json.loads(FACTS_JSON.read_text())
    fr = d.get("frozen")
    if not isinstance(fr, dict) or not str(fr.get("status", "")).startswith("FROZEN"):
        raise RuntimeError(f"sh_core: {FACTS_JSON} carries no FROZEN block (step-1 facts / CHARTER log)")
    return fr


#: the CHARTER-frozen step-1 facts (h_min / h_max, table tops); a re-freeze rewrites facts.json, never this file
FROZEN = _read_frozen()
H_MIN = float(FROZEN["h_min_m"])
H_MAX = float(FROZEN["h_max_m"])
if not H_MIN < H_SETTLED < H_MAX:
    raise RuntimeError(f"sh_core: frozen range [{H_MIN}, {H_MAX}] does not contain the walking height {H_SETTLED}")
#: H6 restore timeout (PLAN formula): the longest ramp to walking height from below + 1 s
RESTORE_TIMEOUT_S = (H_SETTLED - H_MIN) / HEIGHT_RATE_M_S + 1.0


def steps_for(seconds: float, dt: float) -> int:
    """Whole env steps covering ``seconds`` (ceil, robust to float noise): 0.2 s -> 10, 5.3125 s -> 266 at dt 0.02."""
    return int(math.ceil(float(seconds) / float(dt) - 1e-9))


# ===================================================================================== lazy loaders
_CACHE: dict = {}


def _kin():
    """``sh_kin`` (numpy; the posture map and the linkage), loaded by path on first use."""
    if "kin" not in _CACHE:
        _CACHE["kin"] = _load_by_path(SH_KIN_PATH, "_sh_core_kin")
    return _CACHE["kin"]


def _geom() -> dict:
    """Python floats of the hip linkage (sh_kin): default hip angle and the rod-length formula's constants."""
    if "geom" not in _CACHE:
        K = _kin()
        th0 = float(K.TH0)
        _CACHE["geom"] = {"th0": th0, "r": float(K.FEMUR_ATTACH_R_M), "au": float(K.HIP_ANCHOR_U_M),
                          "ad": float(K.HIP_ANCHOR_D_M), "rod_ref": float(K.hip_len(th0)),
                          "dL_dth_ref": float(K.dLh_dth(th0))}
    return _CACHE["geom"]


def _as64(x, device=None):
    """``x`` as a float64 tensor (a python float never passes through float32)."""
    import torch  # noqa: PLC0415

    if torch.is_tensor(x):
        return x.to(device=x.device if device is None else device, dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _vec(x, n: int, device, *, default=None, name="h"):
    """``x`` (None / python float / 0-d / (n,) tensor) -> an (n,) float64 tensor on ``device`` (never aliases a caller
    tensor of another dtype; a float64 (n,) tensor on the device is returned as is -- callers never write it)."""
    import torch  # noqa: PLC0415

    if x is None:
        if default is None:
            raise ValueError(f"sh_core: {name} is required")
        x = default
    t = torch.as_tensor(x, dtype=torch.float64, device=device)
    if t.dim() == 0:
        t = t.expand(n)
    if tuple(t.shape) != (n,):
        raise ValueError(f"sh_core: {name} must be a scalar or ({n},), got {tuple(t.shape)}")
    return t


def _bool_rows(rows, n: int, device):
    import torch  # noqa: PLC0415

    if rows is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    r = torch.as_tensor(rows, device=device)
    if r.dtype != torch.bool or tuple(r.shape) != (n,):
        raise ValueError(f"sh_core: rows must be an ({n},) bool tensor, got {r.dtype} {tuple(r.shape)}")
    return r


# ===================================================================================== 1. the posture map
def clamp_h(h):
    """``h`` clamped to the frozen [H_MIN, H_MAX] (torch tensor in, float64 tensor out)."""
    return _as64(h).clamp(H_MIN, H_MAX)


def q_table():
    """``(grid, q_hip, q_knee_left)`` float64 numpy arrays: ``sh_kin.q`` on nodes ``H_SETTLED + k * Q_GRID_STEP_M``
    covering [H_MIN - step, H_MAX + step]; the 1.064 node is exactly (0, 0) (asserted < 1e-12 before it is set)."""
    if "qtab" not in _CACHE:
        import numpy as np  # noqa: PLC0415

        K = _kin()
        k_lo = math.floor((H_MIN - H_SETTLED) / Q_GRID_STEP_M + 1e-9) - 1
        k_hi = math.ceil((H_MAX - H_SETTLED) / Q_GRID_STEP_M - 1e-9) + 1
        ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
        grid = H_SETTLED + Q_GRID_STEP_M * ks.astype(np.float64)
        i0 = int(np.nonzero(ks == 0)[0][0])
        if grid[i0] != H_SETTLED:  # pragma: no cover
            raise RuntimeError("sh_core: the q grid misses the 1.064 node")
        qh, qk = (np.array(a, dtype=np.float64) for a in K.q(grid))
        if abs(qh[i0]) > 1e-12 or abs(qk[i0]) > 1e-12:
            raise RuntimeError(f"sh_core: sh_kin.q(1.064) = ({qh[i0]}, {qk[i0]}) is not the anchor (0, 0)")
        qh[i0] = 0.0
        qk[i0] = 0.0
        if not (np.all(np.diff(grid) > 0) and np.all(np.diff(qh) > 0)):
            raise RuntimeError("sh_core: q grid / q_hip not strictly increasing")
        if np.abs(qh).max() > RAW_CLIP or np.abs(qk).max() > RAW_CLIP:
            raise RuntimeError("sh_core: q leaves the raw clip on the frozen range")
        for a in (grid, qh, qk):
            a.setflags(write=False)
        _CACHE["qtab"] = (grid, qh, qk)
    return _CACHE["qtab"]


def _q_torch(device):
    import torch  # noqa: PLC0415

    key = ("qt", str(device))
    if key not in _CACHE:
        grid, qh, qk = q_table()
        _CACHE[key] = tuple(torch.tensor(a.copy(), dtype=torch.float64, device=device) for a in (grid, qh, qk))
    return _CACHE[key]


def q_map(h, *, dtype=None, device=None):
    """``(q_hip, q_knee_left)`` (N,) in action units at the CLAMPED height (float64 linear interpolation in
    :func:`q_table`, exact at every node, 0 exactly at 1.064 +- Q_SNAP_M), cast to ``dtype`` (default float32)."""
    import torch  # noqa: PLC0415

    dtype = torch.float32 if dtype is None else dtype
    hc = _as64(h, device).clamp(H_MIN, H_MAX)
    device = hc.device
    grid, tqh, tqk = _q_torch(device)
    idx = (torch.searchsorted(grid, hc.contiguous(), right=True) - 1).clamp(0, grid.numel() - 2)
    g0 = grid[idx]
    w = (hc - g0) / (grid[idx + 1] - g0)
    qh = tqh[idx] + w * (tqh[idx + 1] - tqh[idx])
    qk = tqk[idx] + w * (tqk[idx + 1] - tqk[idx])
    snap = (hc - H_SETTLED).abs() <= Q_SNAP_M
    zero = torch.zeros_like(qh)
    qh = torch.where(snap, zero, qh)
    qk = torch.where(snap, zero, qk)
    return qh.to(dtype), qk.to(dtype)


def q18(h, *, dtype=None, device=None):
    """The posture as an (N, 18) action: cams 0, hips ``q_hip``, knees ``SIDE * q_knee_left`` (zeros are +0.0)."""
    import torch  # noqa: PLC0415

    qh, qk = q_map(h, dtype=dtype, device=device)
    qh = qh.reshape(-1)
    qk = qk.reshape(-1)
    side = torch.tensor(SIDE, dtype=qk.dtype, device=qk.device)
    cams = torch.zeros(qh.shape[0], NUM_LEGS, dtype=qh.dtype, device=qh.device)
    hips = qh[:, None].expand(-1, NUM_LEGS)
    knees = qk[:, None] * side[None, :] + 0.0          # + 0.0 turns the right legs' -0.0 into +0.0
    return torch.cat([cams, hips, knees], dim=1)


def encode_height(h, *, dtype=None, device=None):
    """Stand-net input column 15: ``(h - H_SETTLED) / HEIGHT_OBS_SCALE`` (float64 arithmetic; exactly 0 at 1.064)."""
    import torch  # noqa: PLC0415

    dtype = torch.float32 if dtype is None else dtype
    return ((_as64(h, device) - H_SETTLED) / HEIGHT_OBS_SCALE).to(dtype)


# ===================================================================================== 2. R_h and the residual label
def r_params():
    """R's validated D3 constants (``sn_common.r_params``: round-trip and not-the-defaults checked)."""
    return SN.r_params()


def rh_target(obs, h_cmd_limited, params=None, *, clip: float = RAW_CLIP):
    """``R_h`` (PLAN H3): ``(target (N, 18), branch (N, 6) int64)``.

    ``restance_target`` (D3 constants unless ``params``) on a copy of the 89-wide current head whose hip-pitch columns
    are re-referenced to the posture, ``p - 0.25 * q_hip(h)``; then ``target = q18(h) + R's output`` (an exactly-zero
    offset is not added, so ``h = 1.064`` returns R's tensor values bit for bit), rods clamped to ``+-clip`` (default
    the raw 4.8), cams to +-1. R reads only columns 3-4, cams and hips; the knee and every other column are R's
    business only through these. ``h_cmd_limited``: scalar or (N,), clamped to [H_MIN, H_MAX] by the q map."""
    tgt, branch, _ = _rh(obs, h_cmd_limited, params, clip)
    return tgt, branch


def _rh(obs, h_cmd_limited, params, clip):
    import torch  # noqa: PLC0415

    if obs.dim() != 2 or obs.shape[1] < R_OBS_WIDTH:
        raise ValueError(f"rh_target: obs must be (N, >= {R_OBS_WIDTH}), got {tuple(obs.shape)}")
    clip = float(clip)
    if not 0.0 < clip <= RAW_CLIP:
        raise ValueError(f"rh_target: clip {clip} must be in (0, {RAW_CLIP}]")
    p = r_params() if params is None else params
    n, dev, dt = int(obs.shape[0]), obs.device, obs.dtype
    h = _vec(h_cmd_limited, n, dev, name="h_cmd_limited")
    q = q18(h, dtype=dt, device=dev)
    head = obs[:, :R_OBS_WIDTH].clone()
    hips = list(HIP_POS_OBS_COLS)
    off = ROD_SCALE * q[:, HIP_ACT]
    head[:, hips] = torch.where(off != 0, head[:, hips] - off, head[:, hips])
    rel, branch = SN.load_rst().restance_target(head, p)
    tgt = torch.where(q != 0, rel + q, rel)
    tgt = torch.cat([tgt[:, CAM_ACT].clamp(-CAM_CLIP, CAM_CLIP), tgt[:, HIP_ACT.start:].clamp(-clip, clip)], dim=1)
    return tgt, branch, q


def label_excess(label):
    """Device scalar: ``max(|label|) - 1`` (> LABEL_TOL = a label the residual clip cannot express). For a trainer
    that accumulates the check without a host sync per step (``residual_label(..., check=False)``)."""
    return label.abs().amax() - RESIDUAL_CLIP


def residual_label(target, h_cmd_limited, *, check: bool = True, tol: float = LABEL_TOL):
    """The DAgger label ``target - q18(h)`` (PLAN H4). With ``check`` (one host sync) raises ValueError when any element
    exceeds ``1 + tol`` -- the student's ``clamp(a_net, +-1)`` could not express that teacher target (R_h is within
    ``q +- 1`` on hip / knee by construction: lift 0.5, knee 0.5; R's cams are within +-1). ``check=False``: no sync;
    accumulate :func:`label_excess` instead."""
    n = int(target.shape[0])
    h = _vec(h_cmd_limited, n, target.device, name="h_cmd_limited")
    lab = target - q18(h, dtype=target.dtype, device=target.device)
    if check and lab.numel():
        worst = float(lab.abs().max())
        if worst > RESIDUAL_CLIP + float(tol):
            raise ValueError(f"residual_label: |R_h - q| reaches {worst:.6g} > {RESIDUAL_CLIP} (the residual clip): "
                             "the teacher is outside what the stand net can express")
    return lab


RhLabel = collections.namedtuple("RhLabel", ("label", "target", "branch", "q18"))


def rh_label(obs, h_cmd_limited, params=None, *, check: bool = True) -> RhLabel:
    """``R_h`` (raw clip 4.8) at the limited command, its residual label ``target - q18`` (checked unless ``check`` is
    False) and the posture used."""
    tgt, branch, q = _rh(obs, h_cmd_limited, params, RAW_CLIP)
    lab = tgt - q
    if check and lab.numel():
        worst = float(lab.abs().max())
        if worst > RESIDUAL_CLIP + LABEL_TOL:
            raise ValueError(f"rh_label: |R_h - q| reaches {worst:.6g} > {RESIDUAL_CLIP} (the residual clip)")
    return RhLabel(lab, tgt, branch, q)


# ===================================================================================== 3. the rate limiter
class HeightLimiter:
    """The 0.08 m/s h_cmd limiter: per-env float64 ``state`` that moves at most ``rate * dt`` per step and SNAPS onto
    its target when within one step (so a constant target is reached exactly and a state already equal to its target
    never moves -- a 1.064 command stays bitwise 1.064). Targets and init values are clamped to [h_min, h_max]."""

    def __init__(self, n: int, device, *, dt: float, rate: float = HEIGHT_RATE_M_S, h_min: float = H_MIN,
                 h_max: float = H_MAX, init: float = H_SETTLED):
        import torch  # noqa: PLC0415

        if not float(dt) > 0 or not float(rate) > 0 or not float(h_min) < float(h_max):
            raise ValueError(f"HeightLimiter: dt {dt}, rate {rate}, range [{h_min}, {h_max}]")
        self.n, self.device, self.dt, self.rate = int(n), device, float(dt), float(rate)
        self.h_min, self.h_max = float(h_min), float(h_max)
        self.max_step = self.rate * self.dt
        self.state = torch.full((self.n,), min(max(float(init), self.h_min), self.h_max), dtype=torch.float64,
                                device=device)

    def clamp(self, h):
        return _vec(h, self.n, self.device).clamp(self.h_min, self.h_max)

    def reset(self, rows, value) -> None:
        """``state[rows] = clamp(value)`` (``value`` scalar or (n,))."""
        import torch  # noqa: PLC0415

        r = _bool_rows(rows, self.n, self.device)
        self.state = torch.where(r, self.clamp(value), self.state)

    def step(self, target):
        """One env step toward ``clamp(target)``; returns the new state (also kept)."""
        import torch  # noqa: PLC0415

        t = self.clamp(target)
        d = t - self.state
        # bounded between the state and the target, so a rounding of state + step can never overshoot by an ulp
        moved = torch.minimum(torch.maximum(self.state + torch.sign(d) * self.max_step, torch.minimum(self.state, t)),
                              torch.maximum(self.state, t))
        self.state = torch.where(d.abs() <= self.max_step, t, moved)
        return self.state


# ===================================================================================== 4. h_meas
def h_meas(root_z, terrain_z6):
    """The shared measured height (PLAN H2): ``root_z (N,) - mean(terrain_z6 (N, 6), -1)`` (= p2s body height = the
    harness's ``root_pos_w[:, 2] - terrain_z.mean(1)``). Returns float64."""
    import torch  # noqa: PLC0415

    rz = torch.as_tensor(root_z).to(torch.float64).reshape(-1)
    tz = torch.as_tensor(terrain_z6).to(torch.float64)
    if tz.dim() != 2 or tz.shape[1] != NUM_LEGS or tz.shape[0] != rz.shape[0]:
        raise ValueError(f"h_meas: terrain_z6 must be ({rz.shape[0]}, 6), got {tuple(tz.shape)}")
    return rz - tz.mean(dim=-1)


def terrain_z_under_feet(foot_pos_w, mesh, raycast_fn):
    """Ground height under each foot: the harness's ``_terrain_z_under_feet`` (a downward raycast from 5 m above each
    foot into the terrain mesh; a kinematic table is not in that mesh, so a table start reads the floor). ``raycast_fn``
    = ``isaaclab.utils.warp.raycast_mesh`` (injected: no Isaac import here). ``foot_pos_w`` (N, 6, 3) -> (N, 6).
    Refuses ``mesh is None`` (the harness's ``root_z - 1.05`` fallback would make h_meas a constant)."""
    if mesh is None:
        raise RuntimeError("terrain_z_under_feet: no terrain mesh (RayCaster.meshes lookup failed); h_meas needs it")
    starts = foot_pos_w.clone()
    starts[..., 2] += 5.0
    dirs = starts.new_zeros(starts.shape)
    dirs[..., 2] = -1.0
    hits = raycast_fn(starts.reshape(-1, 3).contiguous(), dirs.reshape(-1, 3).contiguous(), mesh=mesh)[0]
    return hits.reshape(foot_pos_w.shape)[..., 2]


# ===================================================================================== 5. the stand-net input
def stand_net_input(obs, estimator, *, p0: int, p1: int, num_prop: int, mask, h_cmd_limited, restoring=None):
    """The stand network's actor input (PLAN H2 / H6).

    ``sn_common.stand_net_input(obs, estimator, p0, p1, num_prop, mask)`` -- the estimate on the caller's UNMASKED
    prop ``obs[:, :num_prop]``, a NEW tensor, the mask (iff ``mask``) on it -- then on that new tensor only: column 15 =
    :func:`encode_height` (``h_cmd_limited``) on every row (history frames untouched) and, on ``restoring`` rows ((N,)
    bool), current-frame columns 5, 8, 9, 10 = 0. ``obs`` is never written."""
    import torch  # noqa: PLC0415

    x = SN.stand_net_input(obs, estimator, p0=p0, p1=p1, num_prop=num_prop, mask=mask)
    if x.data_ptr() == obs.data_ptr():  # pragma: no cover - sn_common always concatenates
        raise RuntimeError("stand_net_input: sn_common returned the caller's tensor")
    n = int(x.shape[0])
    if not int(num_prop) > max(HEIGHT_COL, *RESTORE_ZERO_COLS):
        raise ValueError(f"stand_net_input: num_prop {num_prop} does not cover columns {HEIGHT_COL} / "
                         f"{RESTORE_ZERO_COLS}")
    h = _vec(h_cmd_limited, n, x.device, name="h_cmd_limited")
    x[:, HEIGHT_COL] = encode_height(h, dtype=x.dtype, device=x.device)
    if restoring is not None:
        r = _bool_rows(restoring, n, x.device)
        cols = list(RESTORE_ZERO_COLS)
        x[:, cols] = torch.where(r[:, None], torch.zeros_like(x[:, cols]), x[:, cols])
    return x


# ===================================================================================== 6. the router
def s1_clamp(a_s1):
    """S1's executed action: ``clamp(S1, +-1)`` (bitwise the runner wrapper's ``torch.clamp`` at clip 1)."""
    import torch  # noqa: PLC0415

    return torch.clamp(a_s1, -S1_CLIP, S1_CLIP)


def check_deadband(deadband) -> float:
    """The residual deadband as a python float in [0, RESIDUAL_CLIP) (ValueError otherwise: a bool, None, a non-number,
    non-finite, negative, or >= 1 -- which would zero every residual short of the clamp)."""
    if isinstance(deadband, bool) or deadband is None:
        raise ValueError(f"residual deadband {deadband!r} is not a number")
    try:
        d = float(deadband)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"residual deadband {deadband!r} is not a number") from exc
    if not math.isfinite(d) or not 0.0 <= d < RESIDUAL_CLIP:
        raise ValueError(f"residual deadband {d!r} must be in [0, {RESIDUAL_CLIP})")
    return d


def apply_deadband(r, deadband=0.0):
    """The residual deadband on an (N, 18) CLAMPED residual: hip / knee entries (:data:`DEADBAND_ACT`) with
    ``|r| < deadband`` become +0.0, every other entry (cams, and hip / knee entries ``>= deadband``) is ``r``'s own
    value. ``deadband == 0`` returns ``r`` itself (no op at all). The comparison runs in ``r``'s dtype."""
    d = check_deadband(deadband)
    if d == 0.0:
        return r
    import torch  # noqa: PLC0415

    hk = r[:, DEADBAND_ACT]
    hk = torch.where(hk.abs() < d, torch.zeros_like(hk), hk)
    return torch.cat([r[:, :DEADBAND_ACT.start], hk], dim=1)


def apply_deadband_np(r, deadband=0.0):
    """:func:`apply_deadband` in numpy on ``(..., 18)`` (a new array; bitwise the torch op for the same float32 input:
    the threshold is cast to ``r``'s dtype). ``deadband == 0`` returns ``r`` unchanged."""
    d = check_deadband(deadband)
    if d == 0.0:
        return r
    import numpy as np  # noqa: PLC0415

    out = np.array(r, copy=True)
    hk = out[..., DEADBAND_ACT]
    out[..., DEADBAND_ACT] = np.where(np.abs(hk) < np.asarray(d, dtype=out.dtype), np.zeros((), dtype=out.dtype), hk)
    return out


def recorded_deadband(record) -> float:
    """The deadband a record ran with: ``record[RESIDUAL_DEADBAND_KEY]`` and / or ``record["clip"][...]`` (the run_meta
    stand_height block, a table cell's meta, a trainer run.json / infos block). Neither present (every record written
    before the option existed, or no record) -> 0.0. Malformed, or the two present and different -> ValueError."""
    if not isinstance(record, dict):
        return 0.0
    vals = []
    if RESIDUAL_DEADBAND_KEY in record:
        vals.append(check_deadband(record[RESIDUAL_DEADBAND_KEY]))
    clip = record.get("clip")
    if isinstance(clip, dict) and RESIDUAL_DEADBAND_KEY in clip:
        vals.append(check_deadband(clip[RESIDUAL_DEADBAND_KEY]))
    if len(set(vals)) > 1:
        raise ValueError(f"recorded residual deadband disagrees with its clip record: {vals}")
    return vals[0] if vals else 0.0


def recorded_trained_deadband(record) -> float | None:
    """The deadband an eval record says its stand checkpoint was TRAINED with (``record[TRAINED_DEADBAND_KEY]``,
    checked by :func:`check_deadband`); None when the record carries none (no record, key absent -- written before the
    option -- or null: not a campaign-trained checkpoint). Malformed -> ValueError."""
    if not isinstance(record, dict) or record.get(TRAINED_DEADBAND_KEY) is None:
        return None
    return check_deadband(record[TRAINED_DEADBAND_KEY])


def deadband_parity(record) -> dict:
    """``{"evaluated": recorded_deadband(record), "trained": recorded_trained_deadband(record), "note": str | None}``:
    ``note`` says the two differ when a trained value is recorded and is not the evaluated one (None otherwise). A
    NOTE, never a refusal (the step-6 probe's mismatch is intended). Malformed -> ValueError (either value)."""
    ev, tr = recorded_deadband(record), recorded_trained_deadband(record)
    note = (None if tr is None or tr == ev else
            f"evaluated at residual deadband {ev:g} but the stand checkpoint was trained at {tr:g} (config parity: "
            "intended only for a probe)")
    return {"evaluated": ev, "trained": tr, "note": note}


def stand_action(a_net, h_cmd_limited, *, deadband: float = 0.0):
    """``(a_stand, q18)``: ``q18(h) + clamp(a_net, +-1)`` (an exactly-zero posture is not added: at 1.064 the result is
    ``clamp(a_net, +-1)`` bit for bit), rods clamped to the raw +-4.8, cams to +-1. ``deadband`` > 0: the clamped
    residual's hip / knee entries below it are zeroed BEFORE ``+ q`` (:func:`apply_deadband`); 0 = no op."""
    import torch  # noqa: PLC0415

    n = int(a_net.shape[0])
    a = apply_deadband(torch.clamp(a_net, -RESIDUAL_CLIP, RESIDUAL_CLIP), deadband)
    q = q18(_vec(h_cmd_limited, n, a.device, name="h_cmd_limited"), dtype=a.dtype, device=a.device)
    s = torch.where(q != 0, a + q, a)
    s = torch.cat([s[:, CAM_ACT].clamp(-CAM_CLIP, CAM_CLIP), s[:, HIP_ACT.start:].clamp(-RAW_CLIP, RAW_CLIP)], dim=1)
    return s, q


def route(stand_eff, a_stand, a_s1):
    """``where(stand_eff, a_stand, clamp(a_s1, +-1))`` -- ``stand_eff`` = the EFFECTIVE stand vector (slot | restoring);
    ``a_stand`` an executed stand action (:func:`stand_action`, or R_h's target on an R-executed row)."""
    import torch  # noqa: PLC0415

    if stand_eff.dtype != torch.bool or stand_eff.dim() != 1 or int(stand_eff.shape[0]) != int(a_s1.shape[0]):
        raise ValueError(f"route: stand_eff must be an ({a_s1.shape[0]},) bool tensor, got {stand_eff.dtype} "
                         f"{tuple(stand_eff.shape)}")
    if tuple(a_stand.shape) != tuple(a_s1.shape):
        raise ValueError(f"route: shapes differ {tuple(a_stand.shape)} vs {tuple(a_s1.shape)}")
    return torch.where(stand_eff[:, None], a_stand.to(a_s1.dtype), s1_clamp(a_s1))


RouterOut = collections.namedtuple("RouterOut", ("actions", "a_stand", "q18", "a_s1"))


def router_actions(stand_eff, a_net, a_s1, h_cmd_limited, *, deadband: float = 0.0) -> RouterOut:
    """One call: ``a_stand, q = stand_action(a_net, h, deadband=deadband)``, ``actions = route(stand_eff, a_stand,
    a_s1)``; returns ``RouterOut(actions, a_stand, q18, clamp(a_s1, +-1))`` (raws record ``a_net`` and ``q18``
    separately). The deadband acts on the stand action only: S1's rows are ``clamp(S1, +-1)`` whatever it is."""
    a_stand, q = stand_action(a_net, h_cmd_limited, deadband=deadband)
    return RouterOut(route(stand_eff, a_stand, a_s1), a_stand, q, s1_clamp(a_s1))


def effective_stand(stand_slot, restoring):
    """``stand_slot | restoring`` (both (N,) bool)."""
    return stand_slot.bool() | restoring.bool()


# ===================================================================================== 7. the H6 restore
def hip_rod_dev(obs):
    """``(N, 6)`` float64: measured hip rod length minus rod(q(1.064)), m -- ``hip_len(default + obs[:, 29 + i])``
    (the joint-space reading; a loaded leg's PD sag shows here, see RESTORE_ROD_TOL_M)."""
    import torch  # noqa: PLC0415

    g = _geom()
    th = obs[:, list(HIP_POS_OBS_COLS)].to(torch.float64) + g["th0"]
    pu = g["r"] * torch.cos(th) - g["au"]
    pd = g["r"] * torch.sin(th) - g["ad"]
    return torch.sqrt(pu * pu + pd * pd) - g["rod_ref"]


def hip_rods_ok(obs, tol: float = RESTORE_ROD_TOL_M):
    """``(N,)`` bool: every hip rod within ``tol`` of rod(q(1.064))."""
    return (hip_rod_dev(obs).abs() <= float(tol)).all(dim=1)


HeightStep = collections.namedtuple(
    "HeightStep", ("h_lim", "stand_eff", "restoring", "engaged", "completed", "timed_out", "merged", "target"))


class HeightController:
    """Per-env height state: the limiter + the H6 restore machine, in the ONE per-step order the trainer router and
    the eval launcher both use. Call :meth:`step` once per env step BEFORE computing the actions for ``obs`` (with the
    latched stand slot that ``obs`` encodes), and :meth:`reset` on the rows the env reset after ``env.step``.

    ``step(stand_slot, h_cmd, obs)`` (all (N,)):
      1. edges: ``leaving = prev_slot & ~slot``; ENGAGE = leaving and |limited command (the one the last stand row
         used) - 1.064| > 2 cm; MERGE = a stand slot on a restoring row (stand -> stand: the restore ends, the new stand
         continues from the limiter's current state; the caller resamples its command);
      2. rows restoring since an earlier step: ``settled`` counts consecutive observations with the limited command
         already at exactly 1.064 and every hip rod within delta; COMPLETE at ``settle_steps`` (0.2 s);
         TIMEOUT once the restore has acted ``timeout_steps`` rows (hand back anyway). Either hands this row to S1;
      3. the limiter advances toward ``where(slot, h_cmd, 1.064)`` (walking and restoring rows target 1.064) -> the
         ``h_lim`` used for q and column 15 on THIS row;
      4. ``stand_eff = slot | restoring``; restore rows' age += 1.
    A stand commanded at 1.064 never engages (its limited command is exactly 1.064), on any terrain."""

    def __init__(self, n: int, device, *, dt: float, rate: float = HEIGHT_RATE_M_S, h_min: float = H_MIN,
                 h_max: float = H_MAX, engage_m: float = RESTORE_ENGAGE_M, rod_tol_m: float = RESTORE_ROD_TOL_M,
                 settle_s: float = RESTORE_SETTLE_S, timeout_s: float | None = None):
        import torch  # noqa: PLC0415

        self.n, self.device, self.dt = int(n), device, float(dt)
        self.limiter = HeightLimiter(n, device, dt=dt, rate=rate, h_min=h_min, h_max=h_max)
        self.engage_m, self.rod_tol_m = float(engage_m), float(rod_tol_m)
        self.timeout_s = (H_SETTLED - float(h_min)) / float(rate) + 1.0 if timeout_s is None else float(timeout_s)
        longest = max(H_SETTLED - float(h_min), float(h_max) - H_SETTLED) / float(rate)
        if self.timeout_s < longest:
            raise ValueError(f"HeightController: timeout {self.timeout_s:.3f} s is shorter than the longest ramp "
                             f"{longest:.3f} s to walking height in [{h_min}, {h_max}]")
        self.settle_steps = steps_for(settle_s, dt)
        self.timeout_steps = steps_for(self.timeout_s, dt)
        z = lambda dtype: torch.zeros(self.n, dtype=dtype, device=device)  # noqa: E731
        self.restoring = z(torch.bool)
        self.prev_slot = z(torch.bool)
        self.age = z(torch.int64)            # restore rows already acted in the current restore
        self.settled = z(torch.int64)        # consecutive completion-condition observations
        self._counts = {k: z(torch.int64) for k in ("engaged", "completed", "timed_out", "merged", "reset_cut")}

    @property
    def h_lim(self):
        """The limited command (float64 (N,)) the last step used."""
        return self.limiter.state

    def reset(self, rows, start, *, h_cmd=None, h_meas=None):
        """Episode reset of ``rows`` ((N,) bool). ``start``: one of :data:`START_KINDS` for all rows, or an (N,) int64
        tensor of indices into START_KINDS (0 walk, 1 table, 2 ground, 3 random). ``walk`` inits the limiter at the
        COMMAND (``h_cmd``; default 1.064 -- pass the first row's target: the stand's command when the episode starts in
        a stand slot); ``table`` / ``ground`` / ``random`` at ``clamp(h_meas)`` (``h_meas`` required unless ``start`` is
        the string ``"walk"``). Clears the restore. No host sync. Returns the (N,) bool mask of rows whose restore the
        reset cut (a missed hand-back for the caller's K4 count)."""
        import torch  # noqa: PLC0415

        r = _bool_rows(rows, self.n, self.device)
        cmd = _vec(h_cmd, self.n, self.device, default=H_SETTLED, name="h_cmd")
        if isinstance(start, str):
            if start not in START_KINDS:
                raise ValueError(f"HeightController.reset: start {start!r} not in {START_KINDS}")
            if START_INIT[start] == "command":
                init = cmd
            else:
                init = _vec(h_meas, self.n, self.device, name="h_meas (table / ground / random starts)")
        else:
            code = torch.as_tensor(start, device=self.device)
            if code.dtype not in (torch.int64, torch.int32) or tuple(code.shape) != (self.n,):
                raise ValueError(f"HeightController.reset: start must be a kind string or an ({self.n},) int tensor "
                                 f"of START_KINDS indices, got {code.dtype} {tuple(code.shape)}")
            meas = _vec(h_meas, self.n, self.device, name="h_meas (table / ground / random starts)")
            init = torch.where(code.long() == START_KINDS.index("walk"), cmd, meas)
        self.limiter.reset(r, init)
        cut = r & self.restoring
        self._counts["reset_cut"] += cut.to(torch.int64)
        self.restoring = self.restoring & ~r
        self.prev_slot = self.prev_slot & ~r
        self.age = torch.where(r, torch.zeros_like(self.age), self.age)
        self.settled = torch.where(r, torch.zeros_like(self.settled), self.settled)
        return cut

    def force_restore(self, rows) -> None:
        """Start a restore on ``rows`` now (e.g. at spawn on a random-height start whose first slot is a walk); counted
        as engaged. A stand slot on the next step merges it as usual."""
        import torch  # noqa: PLC0415

        r = _bool_rows(rows, self.n, self.device) & ~self.restoring
        self.restoring = self.restoring | r
        self.age = torch.where(r, torch.zeros_like(self.age), self.age)
        self.settled = torch.where(r, torch.zeros_like(self.settled), self.settled)
        self._counts["engaged"] += r.to(torch.int64)

    def step(self, stand_slot, h_cmd, obs) -> HeightStep:
        """See the class docstring. ``stand_slot`` (N,) bool, ``h_cmd`` the raw stand height command (scalar / (N,);
        None = 1.064; read on stand rows only), ``obs`` the policy observation of this step (hip columns 29-34)."""
        import torch  # noqa: PLC0415

        slot = _bool_rows(stand_slot, self.n, self.device)
        h_prev = self.limiter.state
        # 1. edges
        leaving = self.prev_slot & ~slot
        merged = self.restoring & slot
        engaged = leaving & ~self.restoring & ((h_prev - H_SETTLED).abs() > self.engage_m)
        self.restoring = (self.restoring & ~slot) | engaged
        fresh = engaged | merged
        self.age = torch.where(fresh, torch.zeros_like(self.age), self.age)
        self.settled = torch.where(fresh, torch.zeros_like(self.settled), self.settled)
        # 2. completion / timeout of restores already running
        cont = self.restoring & ~engaged
        ok = cont & (h_prev == H_SETTLED) & hip_rods_ok(obs, self.rod_tol_m)
        self.settled = torch.where(ok, self.settled + 1, torch.where(cont, torch.zeros_like(self.settled),
                                                                     self.settled))
        completed = cont & (self.settled >= self.settle_steps)
        timed_out = cont & ~completed & (self.age >= self.timeout_steps)
        self.restoring = self.restoring & ~(completed | timed_out)
        # 3. the limiter
        cmd = _vec(h_cmd, self.n, self.device, default=H_SETTLED, name="h_cmd")
        target = torch.where(slot, cmd, torch.full_like(cmd, H_SETTLED))
        h_lim = self.limiter.step(target)
        # 4. bookkeeping
        self.age = torch.where(self.restoring, self.age + 1, self.age)
        self.prev_slot = slot
        for k, v in (("engaged", engaged), ("completed", completed), ("timed_out", timed_out), ("merged", merged)):
            self._counts[k] += v.to(torch.int64)
        return HeightStep(h_lim, slot | self.restoring, self.restoring.clone(), engaged, completed, timed_out, merged,
                          self.limiter.clamp(target))

    def counts(self) -> dict:
        """Cumulative event totals over all envs (one host sync)."""
        return {k: int(v.sum()) for k, v in self._counts.items()}

    def record(self) -> dict:
        """The realised constants (run_meta)."""
        return {"dt": self.dt, "rate_m_s": self.limiter.rate, "h_min": self.limiter.h_min, "h_max": self.limiter.h_max,
                "engage_m": self.engage_m, "rod_tol_m": self.rod_tol_m, "settle_steps": self.settle_steps,
                "timeout_s": self.timeout_s, "timeout_steps": self.timeout_steps}


# ===================================================================================== 8. the warm start
def zero_height_column(model_state_dict: dict):
    """``(new_state_dict, norm_before)``: a copy (every tensor cloned) with ``actor.actor_backbone.0.weight[:, 15]`` =
    0. The backbone's input is ``cat(obs[:, :num_prop], scan latent, priv explicit, hist latent)``, so its column 15
    IS obs column 15 (the height command) and nothing else reads that column on the stand-net path (the estimator
    takes the unwritten caller obs; the history view starts at column 399)."""
    import torch  # noqa: PLC0415

    if WARM_ZERO_KEY not in model_state_dict:
        raise KeyError(f"zero_height_column: {WARM_ZERO_KEY} missing")
    out = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in model_state_dict.items()}
    w = out[WARM_ZERO_KEY]
    if w.dim() != 2 or w.shape[1] <= HEIGHT_COL:
        raise ValueError(f"zero_height_column: {WARM_ZERO_KEY} has shape {tuple(w.shape)}")
    norm = float(w[:, HEIGHT_COL].norm())
    w[:, HEIGHT_COL] = 0.0
    return out, norm


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def warm_start_state(path=STAND_NET_CKPT, *, sha256: str | None = STAND_NET_SHA256) -> dict:
    """The warm start (PLAN H4): the checkpoint at ``path`` (sha256 checked unless ``sha256`` is None) loaded on the CPU
    with :func:`zero_height_column` applied to its ``model_state_dict``; ``estimator_state_dict`` / ``iter`` /
    ``infos`` kept (no optimizer state: the trainer builds a fresh Adam). Adds ``warm_start`` (the record)."""
    import torch  # noqa: PLC0415

    path = Path(path)
    got = _sha256(path)
    if sha256 is not None and got != sha256:
        raise RuntimeError(f"warm_start_state: {path} sha256 {got[:12]} != pinned {sha256[:12]}")
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd, norm = zero_height_column(ck["model_state_dict"])
    rec = {"source": str(path), "sha256": got, "zeroed": f"{WARM_ZERO_KEY}[:, {HEIGHT_COL}]",
           "w15_norm_before": norm, "iter": ck.get("iter")}
    return {"model_state_dict": sd,
            "estimator_state_dict": {k: v.detach().clone() for k, v in ck["estimator_state_dict"].items()},
            "iter": ck.get("iter"), "infos": ck.get("infos"), "warm_start": rec}


def load_warm_start(policy, path=STAND_NET_CKPT, *, sha256: str | None = STAND_NET_SHA256) -> dict:
    """Load the warm start into ``policy`` (strict; tensors copied to the policy's device by ``load_state_dict``).
    Returns the record (plus the estimator state, for the caller's identity check against S1's)."""
    st = warm_start_state(path, sha256=sha256)
    policy.load_state_dict(st["model_state_dict"], strict=True)
    return {**st["warm_start"], "estimator_state_dict": st["estimator_state_dict"]}


# ===================================================================================== records
def clip_record(deadband: float = 0.0) -> dict:
    """The realised clip semantics (run_meta; step-5 asserts read it), with the run's residual deadband
    (:data:`RESIDUAL_DEADBAND_KEY`; 0 = off, the action path without the option)."""
    d = check_deadband(deadband)
    return {"harness_clip_actions": HARNESS_CLIP, "s1_clamp": S1_CLIP, "residual_clamp": RESIDUAL_CLIP,
            "stand_rod_clamp": RAW_CLIP, "stand_cam_clamp": CAM_CLIP,
            "stand_action": ("q18(h_lim) + clamp(a_net, +-1), rods +-4.8, cams +-1" if d == 0.0 else
                             f"q18(h_lim) + deadband(clamp(a_net, +-1), {d!r} on hip / knee), rods +-4.8, cams +-1"),
            "s1_action": "clamp(S1, +-1)", RESIDUAL_DEADBAND_KEY: d,
            "residual_deadband_rule": "after the +-1 clamp, before + q, hip / knee columns 6-17 only (cams never): "
                                      "r <- r if |r| >= deadband else 0 (hard); 0 = off"}


def provenance(deadband: float = 0.0) -> dict:
    """Shas + constants for run_meta: this module, sh_kin, sn_common, R's module, facts.json, the warm start; ``clip``
    = :func:`clip_record` at the run's deadband."""
    return {"sh_core_sha256": _sha256(Path(__file__).resolve()), "sh_kin_sha256": _sha256(SH_KIN_PATH),
            "sn_common_sha256": _sha256(SN_COMMON_PATH), "restance_sha256": _sha256(SN.RESTANCE_MODULE),
            "facts_json_sha256": _sha256(FACTS_JSON), "stand_net_ckpt": str(STAND_NET_CKPT),
            "stand_net_sha256": STAND_NET_SHA256, "restance_d3": SN.RESTANCE_D3,
            "h_settled": H_SETTLED, "h_min": H_MIN, "h_max": H_MAX, "height_obs_scale": HEIGHT_OBS_SCALE,
            "height_col": HEIGHT_COL, "rate_m_s": HEIGHT_RATE_M_S, "restore_zero_cols": list(RESTORE_ZERO_COLS),
            "restore_engage_m": RESTORE_ENGAGE_M, "restore_rod_tol_m": RESTORE_ROD_TOL_M,
            "restore_settle_s": RESTORE_SETTLE_S, "restore_timeout_s": RESTORE_TIMEOUT_S,
            "q_grid_step_m": Q_GRID_STEP_M, "q_snap_m": Q_SNAP_M, "clip": clip_record(deadband),
            "residual_deadband_probe": RESIDUAL_DEADBAND_PROBE}
