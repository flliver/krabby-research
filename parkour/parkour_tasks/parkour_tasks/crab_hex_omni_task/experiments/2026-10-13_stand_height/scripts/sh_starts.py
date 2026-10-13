#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``), PLAN H5 -- the campaign-local ENV COMPOSITION of the height DAgger
trainer: per-episode start modes, the table, and the three wrappers (command resample, push, hip-contact termination).
Applied between ``parse_env_cfg`` and ``gym.make`` by ``sh_dagger.py`` (:func:`compose_env_cfg`); no config file, no
``KRABBY_*`` variable and no shared module is edited.

TORCH- / NUMPY-FREE AT IMPORT (rule L6): the entry loads this file before Kit; torch, numpy, ``sh_kin`` and the
predecessor's ``sn_table_core`` (geometry, linkage) load inside the functions that need them. Isaac objects are never
imported here: the env, the robot, the table, the raycast and the cfg classes are passed in (so the CPU tests drive every
piece with stubs).

WHAT A RESET DOES (:meth:`StartState.on_reset`, the reset event ``sh_start`` appended LAST to the reset terms, so it
runs after the ring spawn and the joint reset; draws from the private ``starts`` generator, a FIXED count per call):

* kind ~ (walk 0.50, table 0.25, ground 0.15, random-height 0.10); table / ground poses ~ (knee_folded 0.60, rod_min
  0.20, a random fold 0.20); top ~ U[top_min, top_max]; the random-height h ~ U[h_min, h_max];
* the floor: one downward raycast at the root and at the six default toe positions (spawn xy / yaw of the ring spawn),
  plus one under each of the drawn pose's own probe points (:func:`pose_probe_points`: every leg's pivot / knee / toe
  and the body corners / centre -- ``sh_table_core.probe_points_body``, the K3 test's points);
* walk: untouched (the ring spawn + the scaled default joints), the table parked;
* table: the robot at its ring xy / yaw, root z = floor(root) + top + 0.15875 + 0.005 (belly 5 mm above the top),
  joints = the retracted pose (targets = joints, zero velocities); the tall kinematic cuboid centred under the root, its
  x along the body x, top at floor(root) + top (its z placement sets the top);
* ground: root z = the highest floor under the POSE's probe points (the K3 rule, ``sh_table_core.sh_start``: a folded
  pad / hip plate rests where the default toes are not) + (the pose's lowest leg-plane hull point below the root) +
  0.005;
* random-height: joints = the stance posture q(h) (the action term's target angles), root z = max floor + its rigid
  height + 0.005 (feet loaded at spawn);
* per env (exposed to the trainer): ``kind``, ``pose``, ``top``, the analytic spawn height ``h_spawn`` (root z minus the
  mean floor under the toes), ``spawn_lowest`` (the chassis' lowest point at spawn), the hip-contact disarm (ground
  only), ``up`` (walk / random from spawn; table / ground once risen), the pending forced first stand; the outgoing
  episode's fields are snapshotted into ``prev_*`` first (the trainer reads them for ``episodes.jsonl``).

THE COMMAND WRAPPER (:func:`make_command_class`: a subclass of the env's command class whose ``_resample_command``
calls the original, then :meth:`StartState.on_resample`): every resample draws the row's stand height command
``h_cmd`` (private ``hcmd`` generator, two uniforms per row -- so a stand -> stand merge redraws); a pending row (a
table / ground / random-height episode's first command) is forced to a stand of ``ramp + settle + 3 s + margin``
(``time_left`` is set after the base ``_resample`` drew it, the base increments the counter after). The row whose forced
command expires (its next resample, merge or not) is the "slot change" the table park keys on.

PER STEP (:meth:`StartState.post_physics`, called by the termination wrapper -- the only hook that runs once per env
step after physics with the env in hand): the chassis' rise above its spawn; ``up`` and the hip-contact re-arm latch at
a 0.10 m rise; the table is parked (20 m under the floor) at the first check after the slot change once the chassis'
lowest point is 0.05 m above the top. The termination wrapper returns ``where(disarmed, tilt-only, full)`` (both from the
ORIGINAL function: the hip clause is ``contact_force_threshold``) and records the cause. The push wrapper restores the
pre-push root velocity of every env that is not up (the original runs unchanged, so the env's RNG stream is too).
"""
from __future__ import annotations

import copy
import functools
import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2]
PRED_SCRIPTS = TASK / "experiments" / "2026-10-11_stand_network" / "scripts"
STC_PATH = PRED_SCRIPTS / "sn_table_core.py"
KIN_PATH = HERE / "sh_kin.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_starts: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


SPEC = _load(HERE / "sh_dagger_spec.py", "_sh_starts_spec")
SH = SPEC.SH

LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
LEFT = frozenset({"FL", "ML", "RL"})
TABLE_NAME = "sh_table"
TABLE_PRIM = "{ENV_REGEX_NS}/ShTable"
#: ``crab_hex_omni_command_math.SLOT_STAND`` (pinned by the tests)
SLOT_STAND = 0
BODY_HALF_Z_M = 0.15875
#: marker attributes on the wrapped callables / class (composition_reasons reads them)
MARK = "__sh_dagger_wrapper__"

_MODS: dict = {}


def stc():
    """The predecessor's ``sn_table_core`` (torch-free at import; its linkage loads torch on first use)."""
    if "stc" not in _MODS:
        _MODS["stc"] = _load(STC_PATH, "_sh_starts_stc")
    return _MODS["stc"]


def kin():
    if "kin" not in _MODS:
        _MODS["kin"] = _load(KIN_PATH, "_sh_starts_kin")
    return _MODS["kin"]


# ===================================================================================== the pose bank (CPU, once)
def lowest_parts(hip: float, knee: float) -> dict:
    """Lowest leg-plane z (body frame, root at 0) of each leg part (``sh_facts._lowest_parts``, the same hulls)."""
    S = stc()
    tib = S.tibia_hull(hip, knee)
    lp = S.leg_plane_points(hip, knee)
    plate_bottom = -(S.PLATE_CORE_HALF_Z_M + S.PLATE_RADIUS_M)
    femur_low = min(lp["pivot"][1], lp["knee"][1]) - S.FEMUR_RADIUS_M
    tib_low = min(p[1] for p in tib)
    parts = {"tibia_or_pad": tib_low, "femur": femur_low, "hip_plate": plate_bottom, "belly": -S.BODY_HALF_M[2]}
    low = min(parts.values())
    first = "+".join(sorted(k for k, v in parts.items() if v <= low + 1e-3))
    return {"parts_z": parts, "lowest_z": low, "first_contact": first}


def pose_row(name: str, hip: float, knee: float) -> dict:
    lp = lowest_parts(hip, knee)
    leg_low = min(v for k, v in lp["parts_z"].items() if k != "belly")
    return {"name": name, "hip": float(hip), "knee_left": float(knee), "lowest_z": lp["lowest_z"],
            "first_contact": lp["first_contact"], "ground_root_z": -lp["lowest_z"] + SPEC.SPAWN_CLEARANCE_M,
            "table_top_min": SPEC.TOE_FLOOR_CLEAR_M - leg_low - BODY_HALF_Z_M - SPEC.SPAWN_CLEARANCE_M}


def pose_bank_rows(n_grid: int = SPEC.RANDOM_FOLD_GRID) -> list[dict]:
    """Row 0 knee_folded, row 1 rod_min (``sn_table_core.retracted_pose``), rows 2.. the facts.json random-fold family
    (hip uniform in [-44.5 deg, -30 deg], knee in [max(knee at the knee-rod minimum, stop + 0.5 deg), the largest fold
    keeping 2 cm to the plate / body] at that hip) on an ``n_grid`` x ``n_grid`` grid -- facts.json's own grid."""
    import numpy as np  # noqa: PLC0415

    S, K = stc(), kin()
    rows = []
    for name in ("knee_folded", "rod_min"):
        p = S.retracted_pose(name)
        rows.append(pose_row(name, p["hip_rad"], p["knee_left_rad"]))
    lim = S.joint_limits_rad()
    m = math.radians(S.LIMIT_MARGIN_DEG)
    for hp in np.linspace(lim["hip"][0] + m, math.radians(SPEC.RANDOM_FOLD_HIP_MAX_DEG), int(n_grid)):
        k_lo = max(float(K.knee_angle(K.KNEE_LEN_MIN_M, hp)), lim["knee_left"][0] + m)
        k_hi = S.max_clear_knee_fold(float(hp), step_rad=2e-3)["knee_rad"]
        for kn in np.linspace(k_lo, k_hi, int(n_grid)):
            rows.append(pose_row("random_fold", float(hp), float(kn)))
    return rows


def stance_table(step: float = 0.001):
    """``(h, hip, knee_left, h_rigid)`` numpy arrays over [h_min, h_max]: the action term's TARGET angles of the
    posture q(h) (``sh_kin.angles_from_action(sh_kin.q(h))``) and their rigid (tilted-pad) height ``sh_kin.h_geom``."""
    import numpy as np  # noqa: PLC0415

    K = kin()
    n = int(round((SH.H_MAX - SH.H_MIN) / step))
    hs = np.linspace(SH.H_MIN, SH.H_MAX, n + 1)
    qh, qk = K.q(hs)
    th, tk = K.angles_from_action(qh, qk)
    return hs, np.asarray(th, dtype=np.float64), np.asarray(tk, dtype=np.float64), np.asarray(K.h_geom(th, tk))


def pose_probe_points(hip: float, knee: float) -> list:
    """``(23, 2)`` body-frame xy under which a GROUND spawn of the pose (hip, knee) reads its floor: every leg's pivot /
    knee / toe (foot FK, plant A15+B, Body_Hip 0) + the body box corners and centre -- ``sh_table_core
    .probe_points_body`` point for point (the K3 table / ground test's rule; pinned by a test)."""
    S = stc()
    pts = []
    for p in S.foot_points_body(hip, knee).values():
        for key in ("pivot", "knee", "toe"):
            pts.append([float(p[key][0]), float(p[key][1])])
    hx, hy = float(S.BODY_HALF_M[0]), float(S.BODY_HALF_M[1])
    pts += [[sx * hx, sy * hy] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)] + [[0.0, 0.0]]
    return pts


def floor_points_body():
    """``(7, 2)`` body-frame xy where the floor is sampled: the root and the six DEFAULT toes (``crab_hex_foot_fk``)."""
    S, K = stc(), kin()
    fk = S.foot_fk()
    pts = [(0.0, 0.0)]
    for leg in LEGS:
        t = fk.leg_points(leg, yaw=0.0, hip=float(K.TH0), knee=float(K.TK0))["toe"]
        pts.append((float(t[0]), float(t[1])))
    return pts


# ===================================================================================== helpers
def _ids(env_ids, n: int, device):
    import torch  # noqa: PLC0415

    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(n, device=device, dtype=torch.int64)[env_ids if isinstance(env_ids, slice) else slice(None)]
    return torch.as_tensor(env_ids, device=device).reshape(-1).long()


def _interp(x, xs, ys):
    """Linear interpolation on a monotone float64 grid (torch, no host sync)."""
    import torch  # noqa: PLC0415

    xc = x.clamp(float(xs[0]), float(xs[-1]))
    i = (torch.searchsorted(xs, xc.contiguous(), right=True) - 1).clamp(0, xs.numel() - 2)
    w = (xc - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + w * (ys[i + 1] - ys[i])


def yaw_of(quat):
    """World yaw of a (w, x, y, z) quaternion batch."""
    import torch  # noqa: PLC0415

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lowest_and_cos(pos, quat):
    """The chassis box's lowest corner z and cos(tilt) (``sn_table_core.lowest_z_and_cos_tilt``), float64."""
    p, q = pos.double(), quat.double()
    return stc().lowest_z_and_cos_tilt(p[:, 2], q[:, 0], q[:, 1], q[:, 2], q[:, 3])


COUNTER_KEYS = (
    "episodes_walk", "episodes_table", "episodes_ground", "episodes_random", "fold_fallbacks", "floor_fallbacks",
    "hcmd_free_n", "hcmd_free_atom", "hcmd_free_sum", "hcmd_free_sumsq", "hcmd_table_n", "hcmd_table_atom",
    "hcmd_table_violations", "forced_table", "forced_ground", "forced_random", "forced_walk",
    "push_events", "push_applied", "push_suppressed", "push_restore_mismatch",
    "hip_would_fire_disarmed", "hip_term_disarmed", "hip_term_armed", "tilt_terms", "rearms", "ups",
    "parks", "parks_before_change", "parks_not_clear", "clear_first_not_parked", "park_delay_sum", "first_checks",
    "first_checks_clear", "top_sum", "top_n", "hcmd_ground_first_n", "hcmd_ground_first_bounded",
    "hcmd_ground_first_violations", "ended_table", "ended_ground", "never_up_table", "never_up_ground")
MINMAX_KEYS = ("hcmd_free_min", "hcmd_free_max", "hcmd_table_margin_min", "top_min", "top_max", "park_delay_min",
               "park_delay_max")


# ===================================================================================== the per-env state
class StartState:
    """Per-env start-mode / command / table / latch state on the device, shared by the env composition (reset event,
    command subclass, push and termination wrappers) and the trainer (which reads it after every ``env.step``)."""

    def __init__(self, n: int, device, *, gen_start, gen_hcmd, mesh_fn=None, raycast_fn=None, bank=None,
                 stance=None, floor_pts=None, shares=SPEC.START_SHARES, pose_shares=SPEC.POSE_SHARES,
                 table_name: str = TABLE_NAME):
        import torch  # noqa: PLC0415

        self.n, self.device = int(n), device
        self.gen_start, self.gen_hcmd = gen_start, gen_hcmd
        self.mesh_fn, self.raycast_fn = mesh_fn, raycast_fn
        self.table_name = table_name
        f64, i64 = torch.float64, torch.int64
        if len(shares) != 4 or abs(sum(shares) - 1.0) > 1e-9 or len(pose_shares) != 3 \
                or abs(sum(pose_shares) - 1.0) > 1e-9:
            raise ValueError(f"StartState: shares {shares} / {pose_shares} must sum to 1")
        self.shares, self.pose_shares = tuple(float(x) for x in shares), tuple(float(x) for x in pose_shares)
        self._cum = torch.cumsum(torch.tensor(self.shares, dtype=f64, device=device), 0)
        self._cum_pose = torch.cumsum(torch.tensor(self.pose_shares, dtype=f64, device=device), 0)
        rows = pose_bank_rows() if bank is None else bank
        if len(rows) < 3 or rows[0]["name"] != "knee_folded" or rows[1]["name"] != "rod_min":
            raise ValueError("StartState: the pose bank must be [knee_folded, rod_min, random folds...]")
        self.bank_rows = rows
        t = lambda key: torch.tensor([float(r[key]) for r in rows], dtype=f64, device=device)  # noqa: E731
        self.b_hip, self.b_knee = t("hip"), t("knee_left")
        self.b_ground_root, self.b_tmin = t("ground_root_z"), t("table_top_min")
        self.b_probe = torch.tensor([r["probe_xy"] if "probe_xy" in r else pose_probe_points(r["hip"], r["knee_left"])
                                     for r in rows], dtype=f64, device=device)          # (n_bank, 23, 2)
        self.n_folds = len(rows) - 2
        hs, th, tk, hg = stance_table() if stance is None else stance
        self.s_h, self.s_hip, self.s_knee, self.s_hg = (torch.as_tensor(a, dtype=f64, device=device).clone()
                                                       for a in (hs, th, tk, hg))
        self.floor_pts = torch.tensor(floor_points_body() if floor_pts is None else floor_pts, dtype=f64,
                                      device=device)
        z = lambda dt, v=0: torch.full((self.n,), v, dtype=dt, device=device)  # noqa: E731
        nan = float("nan")
        self.kind = z(i64, 0)
        self.pose = z(i64, -1)
        self.fold = z(i64, -1)
        self.top = z(f64, nan)
        self.top_w = z(f64, nan)
        self.table_active = z(torch.bool, False)
        self.spawn_root_z = z(f64, nan)
        self.spawn_lowest = z(f64, nan)
        self.h_spawn = z(f64, nan)
        self.floor_c = z(f64, nan)
        self.hip_disarmed = z(torch.bool, False)
        self.up = z(torch.bool, True)
        self.first_pending = z(torch.bool, False)
        self.first_active = z(torch.bool, False)
        self.first_done = z(torch.bool, False)
        self.first_checked = z(torch.bool, False)
        self.first_len_s = z(f64, nan)
        self.change_ep, self.park_ep, self.rearm_ep, self.up_ep = z(i64, -1), z(i64, -1), z(i64, -1), z(i64, -1)
        self.clear_first = z(torch.int8, -1)
        self.term_cause = z(torch.int8, 0)          # this step's crab_failure cause: 0 none, 1 tilt, 2 hip
        self.h_cmd = z(f64, SH.H_SETTLED)
        self.rise = z(f64, nan)
        self.started = z(torch.bool, False)
        self.prev = {k: getattr(self, k).clone() for k in self.SNAPSHOT}
        self.prev_valid = z(torch.bool, False)
        self.c = {k: torch.zeros((), dtype=f64, device=device) for k in COUNTER_KEYS}
        for k in MINMAX_KEYS:
            self.c[k] = torch.full((), math.inf if k.endswith("_min") else -math.inf, dtype=f64, device=device)
        self.c["poses_table"] = torch.zeros(3, dtype=f64, device=device)
        self.c["poses_ground"] = torch.zeros(3, dtype=f64, device=device)
        self._jmap = None
        self._stand_type = None

    SNAPSHOT = ("kind", "pose", "fold", "top", "first_len_s", "change_ep", "park_ep", "rearm_ep", "up_ep",
                "clear_first", "h_spawn", "spawn_root_z")

    # ---------------------------------------------------------------- the floor
    def floor_z(self, xy, yaw, pts=None):
        """``(k, P)`` floor height under body-frame points (default: the root and the six default toes, ``(7, 2)``;
        or per row ``pts`` ``(k, P, 2)``) -- raycast into the terrain mesh; 0 where the mesh or a hit is missing
        (counted: ``floor_fallbacks``)."""
        import torch  # noqa: PLC0415

        k = int(xy.shape[0])
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        if pts is None:
            bx, by = self.floor_pts[None, :, 0], self.floor_pts[None, :, 1]
        else:
            bx, by = pts[..., 0], pts[..., 1]
        wx = xy[:, 0:1].double() + c * bx - s * by
        wy = xy[:, 1:2].double() + s * bx + c * by
        mesh = self.mesh_fn() if self.mesh_fn is not None else None
        if mesh is None or self.raycast_fn is None:
            self.c["floor_fallbacks"] += float(k)
            return torch.zeros(k, int(wx.shape[1]), dtype=torch.float64, device=self.device)
        starts = torch.stack([wx, wy, torch.full_like(wx, 50.0)], dim=-1).reshape(-1, 3).float().contiguous()
        dirs = torch.zeros_like(starts)
        dirs[:, 2] = -1.0
        hits = self.raycast_fn(starts, dirs, mesh=mesh)[0].reshape(k, -1, 3)[..., 2].double()
        ok = torch.isfinite(hits)
        self.c["floor_fallbacks"] += (~ok).sum().double()
        return torch.where(ok, hits, torch.zeros_like(hits))

    # ---------------------------------------------------------------- joints
    def _joint_map(self, robot):
        """hip / knee joint columns (articulation order) and the knee side signs (right legs mirror the knee)."""
        import torch  # noqa: PLC0415

        if self._jmap is None:
            names = list(robot.joint_names)
            hip, knee, side = [], [], []
            for leg in LEGS:
                hip.append(names.index(f"{leg}_Hip_Femur_RevoluteJoint"))
                knee.append(names.index(f"{leg}_Femur_Tibia_RevoluteJoint"))
                side.append(1.0 if leg in LEFT else -1.0)
            for leg in LEGS:
                for j in (f"{leg}_Body_Hip_RevoluteJoint", f"{leg}_Body_CamShaft_RevoluteJoint"):
                    names.index(j)                                   # present (fail loudly otherwise)
            self._jmap = {"n": len(names), "hip": torch.tensor(hip, device=self.device),
                          "knee": torch.tensor(knee, device=self.device),
                          "side": torch.tensor(side, dtype=torch.float64, device=self.device)}
        return self._jmap

    def pose_joints(self, robot, hip, knee_left):
        """``(k, n_joints)`` joint positions: hips ``hip``, knees ``side * knee_left``, Body_Hip / cams 0 (= their
        defaults)."""
        import torch  # noqa: PLC0415

        jm = self._joint_map(robot)
        k = int(hip.shape[0])
        jp = torch.zeros(k, jm["n"], dtype=torch.float64, device=self.device)
        jp[:, jm["hip"]] = hip.double()[:, None].expand(-1, 6)
        jp[:, jm["knee"]] = knee_left.double()[:, None] * jm["side"][None, :]
        return jp

    # ---------------------------------------------------------------- the table
    def write_table(self, env, ids, pos, yaw):
        """Kinematic table pose (``write_root_pose_to_sim`` + the same pose as its kinematic target, as
        ``sn_table_core.sn_table_start``)."""
        import torch  # noqa: PLC0415

        table = env.scene[self.table_name]
        n = int(ids.numel())
        if n == 0:
            return
        quat = torch.zeros(n, 4, dtype=torch.float64, device=self.device)
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)
        pose = torch.cat([pos.double(), quat], dim=-1).to(torch.float32)
        table.write_root_pose_to_sim(pose, env_ids=ids)
        try:
            full = table.data.root_link_pose_w.clone()
            full_xyzw = torch.cat([full[:, :3], full[:, 4:7], full[:, 3:4]], dim=-1)
            table.root_physx_view.set_kinematic_targets(full_xyzw, ids)
        except Exception:  # noqa: BLE001 -- a backend without kinematic targets: the teleport stands alone
            pass

    def _park_pos(self, env, ids):
        import torch  # noqa: PLC0415

        org = env.scene.env_origins[ids].double()
        return torch.stack([org[:, 0], org[:, 1], torch.full_like(org[:, 0], SPEC.TABLE_PARK_Z_M)], dim=-1)

    # ---------------------------------------------------------------- the reset event
    def on_reset(self, env, env_ids) -> None:
        import torch  # noqa: PLC0415

        ids = _ids(env_ids, self.n, self.device)
        k = int(ids.numel())
        if k == 0:
            return
        f64 = torch.float64
        # 0) snapshot the outgoing episode (the trainer reads prev_* for the rows done on this step)
        for key in self.SNAPSHOT:
            self.prev[key][ids] = getattr(self, key)[ids]
        self.prev_valid[ids] = self.started[ids]
        was = self.started[ids]
        for code, name in ((SPEC.SK_TABLE, "table"), (SPEC.SK_GROUND, "ground")):
            ended = was & (self.kind[ids] == code)
            self.c[f"ended_{name}"] += ended.sum().double()
            self.c[f"never_up_{name}"] += (ended & ~self.up[ids]).sum().double()
        self.started[ids] = True
        # 1) draws: a fixed (k, 6) block per call
        u = torch.rand(k, 6, generator=self.gen_start, device=self.device, dtype=f64)
        kind = torch.searchsorted(self._cum, u[:, 0].contiguous(), right=True).clamp(max=3)
        pose = torch.searchsorted(self._cum_pose, u[:, 1].contiguous(), right=True).clamp(max=2)
        top = SPEC.TOP_MIN_M + u[:, 3] * (SPEC.TOP_MAX_M - SPEC.TOP_MIN_M)
        h_rand = SH.H_MIN + u[:, 4] * (SH.H_MAX - SH.H_MIN)
        is_walk, is_table = kind == SPEC.SK_WALK, kind == SPEC.SK_TABLE
        is_ground, is_random = kind == SPEC.SK_GROUND, kind == SPEC.SK_RANDOM
        retracted = is_table | is_ground
        # random fold: uniform over the grid cells allowed at this start (on a table: legs >= 2 cm above the floor)
        valid = (~is_table)[:, None] | (self.b_tmin[None, 2:] <= top[:, None] + 1e-12)
        cnt = valid.sum(dim=1)
        j = torch.floor(u[:, 2] * cnt.to(f64)).to(torch.int64).clamp(min=0)
        j = torch.minimum(j, (cnt - 1).clamp(min=0))
        csum = torch.cumsum(valid.to(torch.int64), dim=1)
        pick = ((csum == (j + 1)[:, None]) & valid).to(torch.int64).argmax(dim=1)
        fallback = retracted & (pose == 2) & (cnt == 0)
        pose = torch.where(fallback, torch.zeros_like(pose), pose)
        bidx = torch.where(pose == 2, 2 + pick, pose)
        # 2) the ring-spawned root (xy / yaw kept) and the floor under it
        robot = env.scene["robot"]
        root = robot.data.root_link_pose_w[ids].clone()
        xy, quat = root[:, 0:2].double(), root[:, 3:7].double()
        yaw = yaw_of(quat)
        fl = self.floor_z(xy, yaw)
        floor_c, floor6, floor_max = fl[:, 0], fl[:, 1:].mean(dim=1), fl.max(dim=1).values
        floor_pose = self.floor_z(xy, yaw, self.b_probe[bidx]).max(dim=1).values      # ground: the pose's own points
        # 3) spawn heights and joints per kind
        hip_r = _interp(h_rand, self.s_h, self.s_hip)
        knee_r = _interp(h_rand, self.s_h, self.s_knee)
        hg_r = _interp(h_rand, self.s_h, self.s_hg)
        z_table = floor_c + top + BODY_HALF_Z_M + SPEC.SPAWN_CLEARANCE_M
        z_ground = floor_pose + self.b_ground_root[bidx]
        z_random = floor_max + hg_r + SPEC.SPAWN_CLEARANCE_M
        root_z = torch.where(is_table, z_table, torch.where(is_ground, z_ground, torch.where(
            is_random, z_random, root[:, 2].double())))
        hip = torch.where(retracted, self.b_hip[bidx], hip_r)
        knee = torch.where(retracted, self.b_knee[bidx], knee_r)
        nw = ~is_walk
        idx = ids[nw]
        if idx.numel():
            p = root[nw].clone()
            p[:, 2] = root_z[nw].to(p.dtype)
            robot.write_root_pose_to_sim(p, env_ids=idx)
            robot.write_root_velocity_to_sim(torch.zeros(int(idx.numel()), 6, dtype=p.dtype, device=self.device),
                                             env_ids=idx)
            jdt = robot.data.joint_pos.dtype
            jp = self.pose_joints(robot, hip[nw], knee[nw]).to(jdt)
            jv = torch.zeros_like(jp)
            robot.write_joint_state_to_sim(jp, jv, env_ids=idx)
            robot.set_joint_position_target(jp, env_ids=idx)
            robot.set_joint_velocity_target(jv, env_ids=idx)
        # 4) the table: under the robot (top by z placement) on table starts, parked otherwise
        tpos = torch.stack([xy[:, 0], xy[:, 1], floor_c + top - 0.5 * SPEC.TABLE_SIZE_M[2]], dim=-1)
        tpos = torch.where(is_table[:, None], tpos, self._park_pos(env, ids))
        self.write_table(env, ids, tpos, torch.where(is_table, yaw, torch.zeros_like(yaw)))
        # 5) the state
        nan = torch.full_like(top, float("nan"))
        self.kind[ids] = kind
        self.pose[ids] = torch.where(retracted, pose, torch.full_like(pose, -1))
        self.fold[ids] = torch.where(retracted & (pose == 2), pick, torch.full_like(pick, -1))
        self.top[ids] = torch.where(is_table, top, nan)
        self.top_w[ids] = torch.where(is_table, floor_c + top, nan)
        self.table_active[ids] = is_table
        self.spawn_root_z[ids] = root_z
        self.spawn_lowest[ids] = root_z - BODY_HALF_Z_M
        self.floor_c[ids] = floor_c
        self.h_spawn[ids] = root_z - floor6
        self.hip_disarmed[ids] = is_ground
        self.up[ids] = is_walk | is_random
        self.first_pending[ids] = nw
        for name in ("first_active", "first_done", "first_checked"):
            getattr(self, name)[ids] = False
        self.first_len_s[ids] = nan
        m1 = torch.full_like(kind, -1)
        self.change_ep[ids], self.park_ep[ids], self.rearm_ep[ids] = m1, m1, m1
        self.up_ep[ids] = torch.where(is_walk | is_random, torch.zeros_like(kind), m1)
        self.clear_first[ids] = torch.full((k,), -1, dtype=torch.int8, device=self.device)
        self.rise[ids] = torch.zeros_like(top)
        # 6) counters (device; no sync)
        c = self.c
        for code, name in enumerate(SPEC.START_KINDS):
            c[f"episodes_{name}"] += (kind == code).sum().double()
        oh = torch.nn.functional.one_hot(pose.clamp(min=0), 3).double()
        c["poses_table"] += (oh * is_table.double()[:, None]).sum(dim=0)
        c["poses_ground"] += (oh * is_ground.double()[:, None]).sum(dim=0)
        c["fold_fallbacks"] += fallback.sum().double()
        tt = torch.where(is_table, top, torch.full_like(top, math.nan))
        c["top_sum"] += torch.nan_to_num(tt, nan=0.0).sum()
        c["top_n"] += is_table.sum().double()
        c["top_min"] = torch.minimum(c["top_min"], torch.where(is_table, top, torch.full_like(top, math.inf)).min())
        c["top_max"] = torch.maximum(c["top_max"], torch.where(is_table, top, torch.full_like(top, -math.inf)).max())

    # ---------------------------------------------------------------- the command wrapper
    def stand_type(self, cmd) -> int:
        if self._stand_type is None:
            pool = getattr(cmd, "_pool", None)
            self._stand_type = int(pool.names.index("stand")) if pool is not None and "stand" in pool.names else -1
        return self._stand_type

    def draw_h_cmd(self, ids):
        """The stand height command for the rows ``ids`` (two uniforms per row from the private ``hcmd`` stream):
        1.064 w.p. 0.35 (where allowed), else U[lo, h_max]; lo = h_min, or max(h_min, top + 0.264) while the table
        is under the robot, or -- the FIRST (forced) command of a ground episode -- max(h_min, h_spawn + 0.14625)
        (the K3 ground rule, ``sh_table_core.cmd_range``: a fold that spawns high must still rise >= 0.10 m to latch
        "up" / re-arm the hip clause; knee_folded / rod_min spawns keep lo = h_min); the atom only when 1.064 >= lo."""
        import torch  # noqa: PLC0415

        k = int(ids.numel())
        u = torch.rand(k, 2, generator=self.gen_hcmd, device=self.device, dtype=torch.float64)
        active = self.table_active[ids]
        lo_t = (self.top[ids] + SPEC.TABLE_CMD_MARGIN_M).clamp(min=SH.H_MIN)
        lo = torch.where(active, lo_t, torch.full_like(lo_t, SH.H_MIN))
        lo = torch.nan_to_num(lo, nan=SH.H_MIN)
        g_first = self.first_pending[ids] & (self.kind[ids] == SPEC.SK_GROUND) & ~active
        lo_g = torch.nan_to_num(self.h_spawn[ids] + SPEC.GROUND_FIRST_CMD_RISE_M, nan=SH.H_MIN)
        lo_g = lo_g.clamp(SH.H_MIN, SH.H_MAX)
        bounded = g_first & (lo_g > SH.H_MIN)
        lo = torch.where(g_first, torch.maximum(lo, lo_g), lo)
        atom_ok = lo <= SPEC.H_CMD_ATOM + 1e-12
        h = lo + u[:, 1] * (SH.H_MAX - lo)
        atom = atom_ok & (u[:, 0] < SPEC.H_CMD_ATOM_P)
        h = torch.where(atom, torch.full_like(h, SPEC.H_CMD_ATOM), h)
        # stats (the bounded ground-first draws are counted apart: they are not U[h_min, h_max])
        c = self.c
        c["hcmd_ground_first_n"] += g_first.sum().double()
        c["hcmd_ground_first_bounded"] += bounded.sum().double()
        c["hcmd_ground_first_violations"] += (g_first & (h < lo_g - 1e-9)).sum().double()
        free = ~active & ~bounded
        fna = free & ~atom
        c["hcmd_free_n"] += free.sum().double()
        c["hcmd_free_atom"] += (free & atom).sum().double()
        c["hcmd_free_sum"] += torch.where(fna, h, torch.zeros_like(h)).sum()
        c["hcmd_free_sumsq"] += torch.where(fna, h * h, torch.zeros_like(h)).sum()
        c["hcmd_free_min"] = torch.minimum(c["hcmd_free_min"], torch.where(fna, h, torch.full_like(h, math.inf)).min())
        c["hcmd_free_max"] = torch.maximum(c["hcmd_free_max"], torch.where(fna, h, torch.full_like(h, -math.inf)).max())
        margin = h - (self.top[ids] + SPEC.TABLE_CMD_MARGIN_M)
        c["hcmd_table_n"] += active.sum().double()
        c["hcmd_table_atom"] += (active & atom).sum().double()
        c["hcmd_table_violations"] += (active & (margin < -1e-9)).sum().double()
        c["hcmd_table_margin_min"] = torch.minimum(
            c["hcmd_table_margin_min"], torch.where(active, margin, torch.full_like(margin, math.inf)).min())
        return h

    def on_resample(self, cmd, env_ids) -> None:
        """After the base ``_resample_command``: draw h_cmd for the rows; force the pending rows' first command to a
        stand of ``ramp + settle + 3 s + margin``; mark the expiry of an active forced command (the slot change)."""
        import torch  # noqa: PLC0415

        ids = _ids(env_ids, self.n, self.device)
        if int(ids.numel()) == 0:
            return
        h = self.draw_h_cmd(ids)
        self.h_cmd[ids] = h
        pend = self.first_pending[ids]
        active = self.first_active[ids]
        expired = active & ~pend
        ep = cmd._env.episode_length_buf[ids].to(torch.int64)
        self.first_done[ids] = self.first_done[ids] | expired
        self.change_ep[ids] = torch.where(expired, ep, self.change_ep[ids])
        self.first_active[ids] = (active & ~expired) | pend
        h_init = self.h_spawn[ids].clamp(SH.H_MIN, SH.H_MAX)
        flen = ((h - h_init).abs() / SH.HEIGHT_RATE_M_S + SPEC.FIRST_STAND_SETTLE_S + SPEC.FIRST_STAND_HOLD_S
                + SPEC.FIRST_STAND_MARGIN_S)
        flen = torch.nan_to_num(flen, nan=SPEC.FIRST_STAND_SETTLE_S + SPEC.FIRST_STAND_HOLD_S)
        self.first_len_s[ids] = torch.where(pend, flen, self.first_len_s[ids])
        cmd.slot_id[ids] = torch.where(pend, torch.full_like(cmd.slot_id[ids], SLOT_STAND), cmd.slot_id[ids])
        cmd.speed_cmd[ids] = torch.where(pend, torch.zeros_like(cmd.speed_cmd[ids]), cmd.speed_cmd[ids])
        yaw = cmd.robot.data.heading_w[ids]
        cmd.heading_target[ids] = torch.where(pend, yaw.to(cmd.heading_target.dtype), cmd.heading_target[ids])
        cmd.type_id[ids] = torch.where(pend, torch.full_like(cmd.type_id[ids], self.stand_type(cmd)), cmd.type_id[ids])
        cmd.time_left[ids] = torch.where(pend, flen.to(cmd.time_left.dtype), cmd.time_left[ids])
        kind = self.kind[ids]
        for code, name in enumerate(SPEC.START_KINDS):
            self.c[f"forced_{name}"] += (pend & (kind == code)).sum().double()
        self.first_pending[ids] = False
        cmd._write_command(ids)

    # ---------------------------------------------------------------- per step, after physics
    def post_physics(self, env) -> None:
        import torch  # noqa: PLC0415

        robot = env.scene["robot"]
        low, _ = lowest_and_cos(robot.data.root_pos_w, robot.data.root_quat_w)
        rise = low - self.spawn_lowest
        self.rise = rise
        ep = env.episode_length_buf.to(torch.int64)
        risen = rise >= SPEC.UP_RISE_M
        re = self.hip_disarmed & risen
        self.rearm_ep = torch.where(re, ep, self.rearm_ep)
        self.hip_disarmed = self.hip_disarmed & ~risen
        nu = ~self.up & risen
        self.up_ep = torch.where(nu, ep, self.up_ep)
        self.up = self.up | risen
        self.c["rearms"] += re.sum().double()
        self.c["ups"] += nu.sum().double()
        chk = self.table_active & self.first_done
        clear = low >= self.top_w + SPEC.TABLE_PARK_CLEAR_M
        first_chk = chk & ~self.first_checked
        self.clear_first = torch.where(first_chk, clear.to(torch.int8), self.clear_first)
        self.first_checked = self.first_checked | first_chk
        self.c["first_checks"] += first_chk.sum().double()
        self.c["first_checks_clear"] += (first_chk & clear).sum().double()
        park = chk & clear
        self.c["clear_first_not_parked"] += (first_chk & clear & ~park).sum().double()
        ids = torch.nonzero(park).flatten()
        if ids.numel():
            self.write_table(env, ids, self._park_pos(env, ids), torch.zeros(int(ids.numel()), dtype=torch.float64,
                                                                              device=self.device))
            self.table_active[ids] = False
            self.park_ep[ids] = ep[ids]
            d = (ep[ids] - self.change_ep[ids]).double()
            c = self.c
            c["parks"] += float(ids.numel())
            c["parks_before_change"] += ((self.change_ep[ids] < 0) | (d < 1)).sum().double()
            c["parks_not_clear"] += (~clear[ids]).sum().double()
            c["park_delay_sum"] += d.sum()
            c["park_delay_min"] = torch.minimum(c["park_delay_min"], d.min())
            c["park_delay_max"] = torch.maximum(c["park_delay_max"], d.max())

    def combine_termination(self, full, tilt):
        """``where(hip_disarmed, tilt-only, full)`` + the cause of this step's crab_failure (0 / 1 tilt / 2 hip)."""
        import torch  # noqa: PLC0415

        full, tilt = full.bool(), tilt.bool()
        hip_only = full & ~tilt
        out = torch.where(self.hip_disarmed, tilt, full)
        self.term_cause = torch.where(out & tilt, 1, torch.where(out & hip_only, 2, 0)).to(torch.int8)
        c = self.c
        c["hip_would_fire_disarmed"] += (hip_only & self.hip_disarmed).sum().double()
        c["hip_term_disarmed"] += (out & hip_only & self.hip_disarmed).sum().double()
        c["hip_term_armed"] += (out & hip_only & ~self.hip_disarmed).sum().double()
        c["tilt_terms"] += (out & tilt).sum().double()
        return out

    def push(self, orig, env, env_ids, *args, **kwargs):
        """Run the ORIGINAL push, then restore the pre-push root velocity of every targeted env that is not up."""
        import torch  # noqa: PLC0415

        robot = env.scene["robot"]
        target = _ids(env_ids, self.n, self.device)
        down = target[~self.up[target]]
        saved = robot.data.root_vel_w[down].clone()
        out = orig(env, env_ids, *args, **kwargs)
        c = self.c
        c["push_events"] += 1.0
        nd = int(down.numel())
        if nd:
            robot.write_root_velocity_to_sim(saved, env_ids=down)
            c["push_restore_mismatch"] += (robot.data.root_vel_w[down] != saved).any(dim=1).sum().double()
        c["push_applied"] += float(int(target.numel()) - nd)
        c["push_suppressed"] += float(nd)
        return out

    # ---------------------------------------------------------------- reads
    def counts(self) -> dict:
        """Every counter as a python number (one host sync)."""
        out = {}
        for k, v in self.c.items():
            vals = v.detach().cpu().tolist()
            out[k] = vals
        return out

    def snapshot_rows(self, rows) -> list[dict]:
        """The outgoing-episode snapshot (``prev_*``) of ``rows`` as host dicts (rows done on this step)."""
        import torch  # noqa: PLC0415

        rows = torch.as_tensor(rows, device=self.device).long()
        keys = self.SNAPSHOT + ("valid",)
        cols = [self.prev[k][rows].to(torch.float64) for k in self.SNAPSHOT] + [self.prev_valid[rows].to(torch.float64)]
        table = torch.stack(cols, dim=1).cpu().tolist()                 # one host transfer
        ints = {"kind", "pose", "fold", "change_ep", "park_ep", "rearm_ep", "up_ep", "clear_first"}
        out = []
        for row in table:
            d = {k: (int(v) if k in ints else v) for k, v in zip(keys, row)}
            d["valid"] = bool(d["valid"])
            out.append(d)
        return out


# ===================================================================================== the wrappers / cfg composition
def make_reset_event(state: StartState):
    def sh_start(env, env_ids):
        state.on_reset(env, env_ids)

    setattr(sh_start, MARK, "reset_event")
    return sh_start


def _rename(wrapper, orig) -> None:
    """After ``functools.wraps`` (which keeps the ORIGINAL signature visible to the manager's parameter check through
    ``__wrapped__``): a name / module that says "wrapped" in every cfg dump (``env.yaml``), so the dump never shows the
    plain original."""
    wrapper.__name__ = f"sh_wrapped_{getattr(orig, '__name__', 'func')}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__module__ = __name__


def wrap_termination(orig, state: StartState):
    """The ``crab_failure`` func: :meth:`StartState.post_physics`, then the ORIGINAL with and without its hip clause
    (``contact_force_threshold=None``) -> ``where(disarmed, tilt-only, full)``. ``functools.wraps`` keeps the original
    signature visible to the manager's parameter check."""
    if getattr(orig, MARK, None):
        raise RuntimeError("wrap_termination: already wrapped")

    @functools.wraps(orig)
    def sh_crab_failure(env, *args, **kwargs):
        state.post_physics(env)
        full = orig(env, *args, **kwargs)
        kw = dict(kwargs)
        kw["contact_force_threshold"] = None
        tilt = orig(env, *args, **kw)
        return state.combine_termination(full, tilt)

    _rename(sh_crab_failure, orig)
    setattr(sh_crab_failure, MARK, "termination")
    return sh_crab_failure


def wrap_push(orig, state: StartState):
    if getattr(orig, MARK, None):
        raise RuntimeError("wrap_push: already wrapped")

    @functools.wraps(orig)
    def sh_push(env, env_ids, *args, **kwargs):
        return state.push(orig, env, env_ids, *args, **kwargs)

    _rename(sh_push, orig)
    setattr(sh_push, MARK, "push")
    return sh_push


def make_command_class(base_cls, state: StartState):
    """A subclass of the env's command class whose ``_resample_command`` runs the original, then
    :meth:`StartState.on_resample`."""

    class ShHeightCommand(base_cls):  # noqa: D401
        """CrabHexOmniCommand + the stand-height campaign's per-resample h_cmd draw and forced first stand."""

        def _resample_command(self, env_ids):
            super()._resample_command(env_ids)
            state.on_resample(self, env_ids)

    setattr(ShHeightCommand, MARK, "command")
    ShHeightCommand.__module__ = __name__
    return ShHeightCommand


def compose_env_cfg(env_cfg, state: StartState, *, sim_utils, RigidObjectCfg, EventTermCfg) -> list[dict]:
    """The campaign-local composition (PLAN H5), each change recorded: the table (scene), the start event (reset,
    LAST), the push / termination wrappers, the command subclass. Refuses a cfg that is not the one the wrappers
    expect."""
    changes = []

    def note(what, old, new, why):
        changes.append({"what": what, "old": old, "new": new, "why": why})

    tmat = copy.deepcopy(env_cfg.scene.terrain.physics_material)
    sx, sy, sz = SPEC.TABLE_SIZE_M
    env_cfg.scene.sh_table = RigidObjectCfg(
        prim_path=TABLE_PRIM,
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=tmat,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SPEC.TABLE_PARK_Z_M)),
    )
    note("scene.sh_table", None, f"kinematic cuboid {sx} x {sy} x {sz} m per env, parked at z {SPEC.TABLE_PARK_Z_M}",
         "PLAN H5 table starts: top set by z placement under the robot's spawn xy / yaw")
    env_cfg.events.sh_start = EventTermCfg(func=make_reset_event(state), mode="reset", params={})
    note("events.sh_start", None, "sh_starts.StartState.on_reset (reset, LAST)", "PLAN H5 per-episode start modes")
    push = env_cfg.events.push_by_setting_velocity
    push.func = wrap_push(push.func, state)
    note("events.push_by_setting_velocity.func", "push_by_setting_velocity", "wrapped (velocity restored on envs "
         "not up)", "PLAN H5 pushes suppressed until the robot is up")
    cf = env_cfg.terminations.crab_failure
    cf.func = wrap_termination(cf.func, state)
    note("terminations.crab_failure.func", "terminate_crab_hex_failure", "wrapped (hip clause off per env until "
         "re-armed; post-physics state)", "PLAN H5 ground starts")
    bv = env_cfg.commands.base_velocity
    old = bv.class_type
    bv.class_type = make_command_class(old, state)
    note("commands.base_velocity.class_type", getattr(old, "__name__", str(old)), "ShHeightCommand (subclass)",
         "PLAN H5 h_cmd per stand draw + the forced first stand")
    return changes


def composition_reasons(env_cfg) -> list[str]:
    """Fail closed after :func:`compose_env_cfg`: every piece in place, the start event LAST among the reset terms."""
    out = []
    ev = env_cfg.events
    names = [k for k, v in vars(ev).items() if v is not None and getattr(v, "mode", None) == "reset"]
    if not names or names[-1] != "sh_start":
        out.append(f"reset event order {names}: sh_start must be last")
    if getattr(getattr(ev, "sh_start", None), "func", None) is None or \
            getattr(ev.sh_start.func, MARK, None) != "reset_event":
        out.append("events.sh_start is not the campaign's reset event")
    if getattr(ev.push_by_setting_velocity.func, MARK, None) != "push":
        out.append("events.push_by_setting_velocity.func is not wrapped")
    if getattr(env_cfg.terminations.crab_failure.func, MARK, None) != "termination":
        out.append("terminations.crab_failure.func is not wrapped")
    if getattr(env_cfg.commands.base_velocity.class_type, MARK, None) != "command":
        out.append("commands.base_velocity.class_type is not the campaign subclass")
    tb = getattr(env_cfg.scene, "sh_table", None)
    if tb is None or getattr(tb, "prim_path", None) != TABLE_PRIM:
        out.append("scene.sh_table missing")
    else:
        rp = getattr(getattr(tb, "spawn", None), "rigid_props", None)
        if not getattr(rp, "kinematic_enabled", False):
            out.append("scene.sh_table is not kinematic")
    return out


# ===================================================================================== the Kit-side probe
class KitProbe:
    """Per-step measurements the trainer reads after every ``env.step`` (and once at the start): root z, the
    chassis' lowest point, ``h_meas`` (``sh_core.h_meas`` with the harness's raycast into the terrain mesh), feet loaded
    (contact >= 20 N and footpad <= 0.15 m above the floor under it), the table's realised top, the action the env
    received (post wrapper clip)."""

    def __init__(self, env, *, mesh, raycast_fn, table_name: str = TABLE_NAME):
        u = env.unwrapped
        self.u = u
        self.robot = u.scene["robot"]
        names = [f"{leg}_Footpad" for leg in LEGS]
        self.foot_bid, fb = self.robot.find_bodies(names, preserve_order=True)
        self.contact = u.scene.sensors["contact_forces"]
        self.foot_sid, fs = self.contact.find_bodies(names, preserve_order=True)
        if list(fb) != names or list(fs) != names:
            raise RuntimeError(f"KitProbe: foot order {fb} / {fs}")
        if mesh is None:
            raise RuntimeError("KitProbe: no terrain mesh (h_meas needs the raycast)")
        self.mesh, self.raycast_fn = mesh, raycast_fn
        self.table = u.scene[table_name]

    def __call__(self) -> dict:
        import torch  # noqa: PLC0415

        d = self.robot.data
        pos, quat = d.root_pos_w, d.root_quat_w
        low, _ = lowest_and_cos(pos, quat)
        fpos = d.body_pos_w[:, self.foot_bid, :]
        tz = SH.terrain_z_under_feet(fpos, self.mesh, self.raycast_fn)
        hm = SH.h_meas(pos[:, 2], tz)
        f = self.contact.data.net_forces_w[:, self.foot_sid].norm(dim=-1)
        loaded = ((f >= SPEC.FOOT_LOADED_N) & ((fpos[..., 2] - tz) <= SPEC.FOOT_FLOOR_M)).sum(dim=1)
        tp = self.table.data.root_pos_w[:, 2].double() + 0.5 * SPEC.TABLE_SIZE_M[2]
        return {"root_z": pos[:, 2].double(), "lowest_z": low.double(), "h_meas": hm, "feet_loaded": loaded.long(),
                "table_top_w": tp, "applied_action": self.u.action_manager.action.detach().clone(),
                "terrain_z6": tz.double()}
