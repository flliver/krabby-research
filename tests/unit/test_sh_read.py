"""Stand-height campaign (``2026-10-13_stand_height``) step 2: ``scripts/sh_read.py`` -- the K1-K6 readers -- on
SYNTHETIC stamps (CPU; no Isaac / Kit / GPU). The synthetic files are read through the REAL ``sh_eval_common``
contract names, ``p2_read`` / ``p2s_stand`` / ``gait_eval`` estimators, ``sh_table_core`` (K3) and ``sh_facts``
(forces).

The synthetic world: the two height rows of ``scenarios_sh.yaml`` (their gates from the manifest itself:
``hwalkstop__omni`` K1 / K4 / K6 -- a walk-handover stand in every height bin, restore-required hand-backs and one
plain hand-back at 1.064; ``hraise__omni`` K2 / K4 -- raises and lowers inside one stand), simulated per role with the
REAL ``sh_core.HeightController`` (limiter + H6 restore) and a first-order height plant whose hip angles follow q(h);
run_meta written as the launcher's ``block_record`` writes it; a synthetic REFS walkstop stamp for K4's onset line;
K3 cell JSON built with ``sh_table_core.classify`` / ``cell_record``; K5 through a stub of ``sn_calib``'s IO plus the
REAL ``sn_calib.calibrate`` on its own synthetic reads.

Pinned: the control table lands (known-good PASS, known-bad FAIL on its named clause); a clean head passes all six;
a tapping head fails K6 stillness, a slow head K2, a falling head K1 / K2; missing raws fail closed (INVALID) only on
the gates whose row they break, missing stamps read PENDING; restore-arming / controller per role; the K3 contract,
lines and R_h premise; the zero-command K5 check; one checkpoint per head; the force table (foot-load statics vs the PD
reading, cross-checked against sh_kin) and the h_max re-check; the CLI writes only where told and refuses the
predecessor / phase-2 dirs; the module imports torch- / numpy- / yaml-free; nothing writes bytecode next to campaign /
predecessor / phase-2 files; one residual deadband per head (CHARTER 2026-10-14: absent = 0, a mix across height
stamps / gated cells / K5 stamps = every gate INVALID, a malformed record INVALIDates its stamp, the value recorded).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1].parent
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
READ_PATH = SCRIPTS / "sh_read.py"
PRED_SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
P2_SCRIPTS = TASK / "experiments/2026-10-07_omni_phase2/scripts"
DT = 0.02
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
JOINT_NAMES = ([n for leg in LEGS for n in (f"{leg}_Body_Hip_RevoluteJoint", f"{leg}_Body_CamShaft_RevoluteJoint")]
               + [f"{leg}_Hip_Femur_RevoluteJoint" for leg in LEGS]
               + [f"{leg}_Femur_Tibia_RevoluteJoint" for leg in LEGS])
SIDE = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
WALK_ROW, RAISE_ROW = "hwalkstop__omni", "hraise__omni"
STAMP_TS = "2026-10-13_00-00-00"
S1_SHA = "a" * 64
HEAD_SHA = {"HEAD": "b" * 64, "HEAD_TAP": "c" * 64, "HEAD_SLOW": "d" * 64, "HEAD_FALL": "e" * 64,
            "HEAD_BADKEY": "f" * 64}

#: (label, speed, dir, hold_s, height). WALK: a walk-handover stand in every bin (0.80 / 0.92 / 1.064 / 1.15 / 1.30),
#: four restore-required hand-backs and a plain one at 1.064. RAISE: 1.064 -> 0.80 -> 1.28 inside one stand, then a
#: restore into a walk, a walk-handover 0.92 stand and a restore into the last walk.
WALK = [("fwd", 0.25, 0.0, 1.0, None), ("stand", 0.0, 0.0, 6.0, 0.80), ("lat_l", 0.20, math.pi / 2, 6.0, None),
        ("stand", 0.0, 0.0, 5.0, 0.92), ("back", 0.20, math.pi, 5.0, None), ("stand", 0.0, 0.0, 4.0, 1.15),
        ("fwd", 0.25, 0.0, 4.0, None), ("stand", 0.0, 0.0, 4.0, 1.064), ("lat_l", 0.20, math.pi / 2, 3.0, None),
        ("stand", 0.0, 0.0, 6.0, 1.30), ("back", 0.20, math.pi, 5.0, None)]
RAISE = [("fwd", 0.25, 0.0, 1.0, None), ("stand", 0.0, 0.0, 2.0, 1.064), ("stand", 0.0, 0.0, 5.0, 0.80),
         ("stand", 0.0, 0.0, 8.0, 1.28), ("lat_l", 0.20, math.pi / 2, 5.0, None), ("stand", 0.0, 0.0, 4.0, 0.92),
         ("back", 0.20, math.pi, 4.0, None)]
ZSCHED = [("fwd", 0.25, 0.0, 1.0, None), ("stand", 0.0, 0.0, 2.0, 1.064), ("lat_l", 0.20, math.pi / 2, 1.0, None)]
WSCHED = [("fwd", 0.25, 0.0, 2.0, None), ("stand", 0.0, 0.0, 3.0, 1.064), ("lat_l", 0.20, math.pi / 2, 2.0, None),
          ("stand", 0.0, 0.0, 3.0, 1.064), ("back", 0.20, math.pi, 2.0, None), ("stand", 0.0, 0.0, 2.0, 1.064)]
DEFAULT = dict(restore=True, gain=0.3, lo=None, hi=None, cams_home=True, tap=False, fall_hold=None, load=376.0)
#: role -> (simulation params, launcher control, stand_ckpt_kind or None)
VARIANTS = {"R_h": ({}, "r_h", None), "R_old": (dict(lo=0.92, hi=1.22), "r_old", None),
            "Amask_old": (dict(lo=0.93, hi=1.21), "amask_old", "amask_s17_w15zeroed"),
            "posture": (dict(cams_home=False), "posture_only", None),
            "no_restore": (dict(restore=False), "r_h", None),
            "HEAD": ({}, "stand_net", "sh_dagger"), "HEAD_TAP": (dict(tap=True), "stand_net", "sh_dagger"),
            "HEAD_SLOW": (dict(gain=0.02), "stand_net", "sh_dagger"),
            "HEAD_FALL": (dict(fall_hold={WALK_ROW: 1, RAISE_ROW: 3}), "stand_net", "sh_dagger"),
            "HEAD_BADKEY": ({}, "stand_net", "sh_dagger")}
N_EPS = 2


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def _pyc_snapshot() -> dict:
    out = {}
    for d in (SCRIPTS, PRED_SCRIPTS, P2_SCRIPTS):
        pc = d / "__pycache__"
        out[str(pc)] = sorted((p.name, p.stat().st_mtime_ns) for p in pc.iterdir()) if pc.is_dir() else None
    return out


#: The caches as they stood when THIS module's first test started (filled by ``_pyc_before`` below). Not taken at
#: import: pytest imports every module at collection, and the suite's other modules (the phase-2 / stand-network
#: tests and the tools they run) may legitimately write bytecode into these dirs before this module runs whenever
#: PYTHONDONTWRITEBYTECODE is unset (CI) -- that is not this module's doing, and the check is about what THESE tests
#: load. Autouse + module scope makes it the first fixture this module sets up, ahead of the module fixtures that load
#: the tools.
_PYC_BEFORE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _pyc_before():
    _PYC_BEFORE.clear()
    _PYC_BEFORE.update(_pyc_snapshot())


@pytest.fixture(scope="module")
def R():
    return _load(READ_PATH, "_t_sh_read")


# ===================================================================================== the synthetic simulator
def simulate(R, sched, params: list[dict], row: str | None = None) -> list[dict]:
    """One batch (rows = episodes, each with its own params): the REAL sh_core.HeightController (restore armed / never
    engaging), a first-order height plant, hips at q(h), cams, foot loads, walking velocity. Raw keys = the reader's
    (the launcher contract in force)."""
    EC, C = R.EC, R.CORE
    K = C._kin()
    seg = np.concatenate([np.full(int(round(h[3] / DT)), i, dtype=np.int16) for i, h in enumerate(sched)])
    T, E = int(seg.size), len(params)
    P = [dict(DEFAULT, **p) for p in params]
    gain = np.array([p["gain"] for p in P])
    lo = np.array([-np.inf if p["lo"] is None else p["lo"] for p in P])
    hi = np.array([np.inf if p["hi"] is None else p["hi"] for p in P])
    restore = np.array([p["restore"] for p in P])
    home = np.array([p["cams_home"] for p in P])
    tap = np.array([p["tap"] for p in P])
    load = np.array([p["load"] for p in P])
    ctl = {True: C.HeightController(E, "cpu", dt=DT), False: C.HeightController(E, "cpu", dt=DT, engage_m=1e9)}
    th0 = float(K.TH0)
    h, cam, vel = np.full(E, C.H_SETTLED), np.zeros(E), np.zeros(E)
    age = np.zeros(E, dtype=np.int64)
    ev = {EC.RAW_ENGAGED: "engaged", EC.RAW_COMPLETED: "completed", EC.RAW_TIMED_OUT: "timed_out",
          EC.RAW_MERGED: "merged"}
    A = {"root_pos_w": np.zeros((T, E, 3), np.float32), "terrain_z": np.zeros((T, E, 6), np.float32),
         "foot_force_norm": np.zeros((T, E, 6), np.float32), "foot_pos_w": np.zeros((T, E, 6, 3), np.float32),
         "foot_lin_vel_w": np.zeros((T, E, 6, 3), np.float32), "joint_pos": np.zeros((T, E, 24), np.float32),
         "joint_vel": np.zeros((T, E, 24), np.float32), "applied_torque": np.zeros((T, E, 24), np.float32),
         "rod_speed": np.zeros((T, E, 12), np.float32), "crab_failure": np.zeros((T, E), bool),
         "done": np.zeros((T, E), bool), "segment_id": np.repeat(seg[:, None], E, 1),
         "steady_mask": np.zeros((T, E), bool), "cmd_applied": np.zeros((T, E, 3), np.float32),
         "root_lin_vel_b": np.zeros((T, E, 3), np.float32), "heading_err": np.zeros((T, E), np.float32),
         R.RAW_SLOT: np.zeros((T, E), bool), R.RAW_EFF: np.zeros((T, E), bool),
         R.RAW_RESTORING: np.zeros((T, E), bool), R.RAW_H_CMD: np.zeros((T, E), np.float64),
         R.RAW_H_LIM: np.zeros((T, E), np.float64), R.RAW_A_NET: np.zeros((T, E, 18), np.float32),
         R.RAW_Q: np.zeros((T, E, 18), np.float32), **{k: np.zeros((T, E), bool) for k in ev}}
    starts = np.concatenate([[0], np.cumsum([int(round(x[3] / DT)) for x in sched])])
    h_prev = h.copy()
    for t in range(T):
        i = int(seg[t])
        _lab, speed, dr, _hs, hh = sched[i]
        slot = hh is not None
        hc = float(hh) if slot else C.H_SETTLED
        qh, _qk = (x.numpy() for x in C.q_map(torch.tensor(h, dtype=torch.float64), dtype=torch.float64))
        # hip: default + 0.25 rad per action unit (exactly the q map); knee: the stance posture (a rod-length action)
        th, tk = th0 + 0.25 * qh, K.stance_angles(np.clip(h, C.H_MIN, C.H_MAX))[1]
        jp = np.zeros((E, 24))
        jp[:, 1:12:2] = cam[:, None]
        jp[:, 12:18] = th[:, None]
        jp[:, 18:24] = tk[:, None] * SIDE
        obs = torch.zeros(E, 89, dtype=torch.float64)
        obs[:, 29:35] = torch.as_tensor(np.repeat((th - th0)[:, None], 6, 1))
        st = {k: c.step(torch.full((E,), slot, dtype=torch.bool), hc, obs) for k, c in ctl.items()}

        def pick(f):
            return np.where(restore, f(st[True]).numpy(), f(st[False]).numpy())
        h_lim, eff = pick(lambda s: s.h_lim), pick(lambda s: s.stand_eff).astype(bool)
        tri = np.zeros(6)
        tri[[0, 3, 4] if (t // 15) % 2 == 0 else [1, 2, 5]] = 2.0
        ff = np.where(eff[:, None], 1.0, tri[None, :]) * load[:, None]
        ff[eff & tap & (age > 100) & (age % 15 < 2), 0] = 0.0
        loaded = ff > 50.0
        tau_h, tau_k = K.stance_joint_torques(np.repeat(th[:, None], 6, 1), np.repeat(tk[:, None], 6, 1), ff)
        at = np.zeros((E, 24))
        at[:, 12:18] = 1.14 * tau_h
        at[:, 18:24] = 1.14 * tau_k * SIDE
        moving = np.abs(h - h_prev) > 1e-7
        A["rod_speed"][t] = np.where(moving[:, None], 0.01, 0.0)
        A["joint_vel"][t] = np.where(moving[:, None], 0.1, 0.0)
        A["root_pos_w"][t, :, 2] = h
        A["foot_force_norm"][t] = ff
        A["foot_pos_w"][t, :, :, 2] = np.where(loaded, 0.0, 0.05)
        A["joint_pos"][t] = jp
        A["applied_torque"][t] = at
        A["steady_mask"][t] = (t - starts[i]) * DT >= 1.0
        if speed > 0:
            A["cmd_applied"][t, :, 0], A["cmd_applied"][t, :, 1] = speed * math.cos(dr), speed * math.sin(dr)
            A["root_lin_vel_b"][t, :, 0], A["root_lin_vel_b"][t, :, 1] = vel * math.cos(dr), vel * math.sin(dr)
        A[R.RAW_SLOT][t] = slot
        A[R.RAW_EFF][t] = eff
        A[R.RAW_RESTORING][t] = pick(lambda s: s.restoring)
        A[R.RAW_H_CMD][t] = hc
        A[R.RAW_H_LIM][t] = h_lim
        A[R.RAW_Q][t] = np.where(eff[:, None], C.q18(torch.as_tensor(h_lim), dtype=torch.float32).numpy(), 0.0)
        for k, f in ev.items():
            A[k][t] = pick(lambda s, f=f: getattr(s, f))
        # plant -> the next row
        h_prev = h.copy()
        target = np.where(eff, np.clip(h_lim, lo, hi), C.H_SETTLED)
        h = h + gain * (target - h)
        cam = np.where(eff, np.where(home, cam * 0.7, 0.3), 0.05 * math.sin(0.3 * t))
        vel = np.where(~eff & (speed > 0), vel + 0.3 * (speed - vel), vel * 0.3)
        age = np.where(eff, age + 1, 0)
    out = []
    for e, p in enumerate(P):
        end = T
        fh = p["fall_hold"]
        if fh is not None and row in fh:
            fr = int(starts[fh[row]] + 100)
            A["crab_failure"][fr, e] = True
            end = fr + 1
        A["done"][end - 1, e] = True
        out.append({k: np.ascontiguousarray(v[:end, e]) for k, v in A.items()})
    return out


def launcher_meta(R, sched, *, row: str | None, gates=(), control="r_h", restore=True, kind=None, sha=None,
                  command="manifest") -> dict:
    """run_meta as the stand_height launcher writes it (block_record's fields the reader uses)."""
    labels = [h[0] for h in sched]
    hh = [h[4] if h[0] == "stand" else None for h in sched]
    rec = dict(R.CORE.HeightController(1, "cpu", dt=DT, engage_m=(R.CORE.RESTORE_ENGAGE_M if restore else 1e9))
               .record())
    if not restore:
        rec["engage_m"] = None
    blk = {"control": control, "command": command, "restore": "armed" if restore else "off", "stand_ckpt_kind": kind,
           "hold_heights": hh if command == "manifest" else R.EC.zero_hold_heights(labels),
           "height_row": ({"manifest": "synthetic.yaml", "scenario": row, "labels": labels, "hold_heights": hh,
                           "gates": list(gates)} if command == "manifest" else None),
           "controller": rec, "h_meas": {"valid": True, "terrain_source": "raycast_mesh"}}
    mode = {"variant": R.EC.VARIANT, "s1_checkpoint_sha256": S1_SHA, R.EC.BLOCK_KEY: blk}
    if sha is not None:
        mode["specialist"] = {"path": "/synthetic.pt", "sha256": sha}
        mode["stand_mask"] = 1
    return {"hold_labels": labels, "hold_speeds": [h[1] for h in sched], "dt": DT, "joint_names": JOINT_NAMES,
            "env_vars": {"KRABBY_OMNI_STAND_OBS": "0"}, "env_vars_manifest": {}, "num_envs": N_EPS,
            "checkpoint": "synthetic", "checkpoint_sha256": S1_SHA, "terrain_source": "raycast_mesh", "p2r_mode": mode}


def write_stamp(lr: Path, name: str, row: str, seed: int, eps: list[dict], meta: dict, *, drop=None) -> Path:
    d = lr / name / row / f"seed{seed:03d}" / STAMP_TS
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "run_meta.json").write_text(json.dumps(meta))
    (d / "p2_complete.json").write_text("{}")
    for i, ep in enumerate(eps):
        ep = dict(ep)
        for k in (drop or {}).get(i, ()):
            ep.pop(k, None)
        np.savez_compressed(d / "raw" / f"episode_{i:02d}.npz", **ep)
    return d


# ----------------------------------------------------------------------------------- K3 synthetic cell JSON
def _classify(TC, kind: str, start: str, top, h_cmd: float, T: int = 400) -> dict:
    t = np.arange(T)
    spawn = (top + 0.005) if top is not None else 0.005
    rise = np.clip(t / 150.0, 0, 1) * 0.30
    feet = np.where(t > 150, 6, 0)
    hm = np.where(t > 200, h_cmd + 0.005, 0.6)
    cos = np.ones(T)
    if kind == "march":
        feet = np.where((t // 10) % 2 == 0, 3, 6)
    if kind == "fall":
        cos = np.where(t > 300, math.cos(0.8), 1.0)
    series = {"lowest_z": spawn + rise, "cos_tilt": cos, "feet_on_floor": feet, "h_meas": hm, "hip_force": np.zeros(T),
              "cam_abs": np.zeros(T), "root_z": spawn + rise + 0.16}
    return TC.classify(series, DT, start=start, spawn_lowest_z=spawn, spawn_root_z=spawn + 0.16, h_cmd=h_cmd,
                       table_top=top)


def write_cells(R, root: Path, role: str, seed: int, controller: str, mix: dict, *, sha=None, overrides=None,
                cells=None) -> Path:
    """The 12 K3 cells (or ``cells``) of one role / seed as the table test writes them (``cell_record``); ``mix`` =
    {kind: n} per cell, ``overrides`` = {cell: mix}."""
    TC = R._tc()
    out = root / "tables" / role / f"es{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for cell in cells or TC.K3_CELLS:
        surface, _pose = TC.parse_cell(cell)
        top = TC.SURFACE_TOP[surface]
        h_cmd = (top + 0.32) if top is not None else 1.064
        start = TC.start_kind(surface)
        m = (overrides or {}).get(cell, mix)
        protos = {k: _classify(TC, k, start, top, h_cmd) for k in m}
        per_env = [dict(protos[k]) for k, n in m.items() for _ in range(n)]
        meta = {"controller": controller, "label": role, "seed": seed, "smoke": False, "video": False,
                "stand_sha256": sha, "episode_s": 8.0, "k3_episode_s": 8.0, "dt": DT}
        (out / f"{role}_s{seed}__{cell}.json").write_text(json.dumps(TC.cell_record(meta, cell, [], per_env)))
    return out


# ----------------------------------------------------------------------------------- K5 stub
def fake_snc(R, stamps_by_head: dict, verdict_over: dict | None = None):
    snc = R._snc()
    ns = types.SimpleNamespace(NAMES=dict(snc.NAMES), KIND=dict(snc.KIND), ROLES=tuple(snc.ROLES))
    ns.read_reference = lambda role, name, kind: {"role": role, "name": name, "kind": kind}
    ns.merge_r = lambda base, sn: base
    ns.valid_only_point = lambda name: {"valid": {"suite@1": {"ckpt_sha": S1_SHA}}}

    def calibrate(reads, identity, heads=()):
        rows = {}
        for key, rd in reads.items():
            g = {k: {"verdict": (verdict_over or {}).get((key, k), "PASS"), "failed": [], "invalid": []}
                 for k in snc.GATES}
            rows[key] = {"gates": g, "stamps": stamps_by_head.get(key, {}), "kind": rd.get("kind")}
        return {"rows": rows, "mismatches": [], "summary": {"expected_cells": 15, "lands_yes": 15}}
    ns.calibrate = calibrate
    return ns


# ----------------------------------------------------------------------------------- the world
@pytest.fixture(scope="module")
def world(R, tmp_path_factory):
    root = tmp_path_factory.mktemp("sh_read_world")
    lr = root / "lr"
    rg = R.row_gates({})
    assert set(rg) == {WALK_ROW, RAISE_ROW}
    names = list(VARIANTS)
    eps = {}
    for row, sched in ((WALK_ROW, WALK), (RAISE_ROW, RAISE)):
        params = [VARIANTS[n][0] for n in names for _ in range(N_EPS)]
        allep = simulate(R, sched, params, row)
        for j, n in enumerate(names):
            p, control, kind = VARIANTS[n]
            mine = allep[j * N_EPS:(j + 1) * N_EPS]
            eps[(n, row)] = mine
            sha = HEAD_SHA.get(n) or (R.CORE.STAND_NET_SHA256 if kind == "amask_s17_w15zeroed" else None)
            meta = launcher_meta(R, sched, row=row, gates=rg[row], control=control,
                                 restore=dict(DEFAULT, **p)["restore"], kind=kind, sha=sha)
            drop = {1: [R.RAW_H_LIM]} if (n == "HEAD_BADKEY" and row == WALK_ROW) else None
            for s in (1, 2):
                write_stamp(lr, f"SYN_{n}", row, s, mine, meta, drop=drop)
    # REFS walkstop (the onset line) and the zero-command K5 stamps
    w = simulate(R, WSCHED, [{}] * N_EPS)
    z = simulate(R, ZSCHED, [{}] * N_EPS)
    zbad = copy.deepcopy(z)
    zbad[1][R.RAW_H_CMD][60:70] = 1.10
    zst = {}
    for s in (1, 2):
        write_stamp(lr, "REFS_SYN", "walkstop__omni", s, w, launcher_meta(R, WSCHED, row=None, command="zero"))
        for name, data, sha in (("SYN_HEAD_z0", z, HEAD_SHA["HEAD"]), ("SYN_TAP_z0", zbad, HEAD_SHA["HEAD_TAP"])):
            zm = launcher_meta(R, ZSCHED, row=None, control="stand_net", kind="sh_dagger", sha=sha, command="zero")
            zst.setdefault(name, {})[f"suite@{s}"] = str(write_stamp(lr, name, "suite__omni", s, data, zm))
    tables = {}
    for s in (1, 2):
        tables.setdefault("R_h", {})[str(s)] = [str(write_cells(R, root, "R_h", s, "r_h", {"good": 100}))]
        tables.setdefault("S1", {})[str(s)] = [str(write_cells(R, root, "S1", s, "s1", {"march": 100}))]
        tables.setdefault("HEAD", {})[str(s)] = [str(write_cells(R, root, "HEAD", s, "stand_net",
                                                                 {"good": 99, "fall": 1}, sha=HEAD_SHA["HEAD"]))]
    roles = {n: {"name": f"SYN_{n}"} for n in ("R_h", "R_old", "Amask_old", "posture", "no_restore")}
    roles["R_h"]["table"] = tables["R_h"]
    roles["S1"] = {"table": tables["S1"]}
    roles["HEAD"] = {"name": "SYN_HEAD", "head": True, "k5_name": "SYN_HEAD_z0", "table": tables["HEAD"]}
    roles["HEAD_TAP"] = {"name": "SYN_HEAD_TAP", "head": True, "k5_name": "SYN_TAP_z0"}
    for n in ("HEAD_SLOW", "HEAD_FALL", "HEAD_BADKEY"):
        roles[n] = {"name": f"SYN_{n}", "head": True}
    roles["HEAD_MISSING"] = {"name": "SYN_NOPE", "head": True}
    spec = {"lr_root": str(lr), "seeds": [1, 2], "refs": "REFS_SYN", "roles": roles, "k5": {"amask": "SN_Amask_s17"}}
    out = R.run(spec, k5=True, forces=True, snc=fake_snc(R, zst))
    return {"root": root, "lr": lr, "spec": spec, "out": out, "eps": eps, "zst": zst, "rg": rg}


# ===================================================================================== module / pure pieces
def test_import_is_torch_numpy_yaml_free_and_writes_no_bytecode():
    code = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(READ_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
        "assert 'numpy' not in sys.modules, 'numpy imported at module level'\n"
        "assert 'yaml' not in sys.modules, 'yaml imported at module level'\n"
        "print(m.H_MIN, m.H_MAX, m.K_BINS)\n"
    )
    before = {p for p in SCRIPTS.iterdir()}
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["0.719", "1.346", "5"]
    assert not (SCRIPTS / "__pycache__").exists()
    assert {p for p in SCRIPTS.iterdir()} == before


def test_bins_percentile_and_share_line(R):
    assert R.BIN_EDGES[0] == R.H_MIN and abs(R.BIN_EDGES[-1] - R.H_MAX) < 1e-12 and R.K_BINS >= 5
    assert [R.height_bin(h) for h in (0.719, 0.80, 0.92, 1.064, 1.15, 1.30, 1.346)] == [0, 0, 1, 2, 3, 4, 4]
    assert R.height_bin(0.70) is None and R.height_bin(1.40) is None
    v = [0.01, 0.02, 0.03, 0.04]
    assert R.pctl(v, 50) == pytest.approx(np.percentile(v, 50)) and R.pctl(v, 90) == pytest.approx(np.percentile(v, 90))
    assert R.pctl([0.0] * 9 + [math.inf], 50) == 0.0
    assert math.isinf(R.pctl([0.0] * 9 + [math.inf], 90))       # 9 * 0.9 = 8.1: touches the inf sample
    assert R.pctl([0.0] * 10 + [math.inf], 90) == 0.0              # 10 * 0.9 = 9.0: exactly a finite sample
    assert R.pctl([], 90) is None
    line, se = R.share_line(1.0, 100, 0.97, 100)
    assert se == pytest.approx(math.sqrt(0.97 * 0.03 / 100)) and line == pytest.approx(1.0 - 0.05)
    line, se = R.share_line(0.5, 50, 0.5, 50)
    assert line == pytest.approx(0.5 - 2 * math.sqrt(2 * 0.25 / 50))
    # the block's restore: the launcher's armed / off words or a bool; anything else is malformed
    assert [R.restore_armed({"restore": x}) for x in ("armed", "off", True, False, "maybe", None)] == [
        True, False, True, False, None, None]


def test_constants_follow_the_frozen_facts_the_core_and_the_contracts(R):
    fr = json.loads((CAMP / "facts.json").read_text())["frozen"]
    assert (R.H_MIN, R.H_MAX) == (fr["h_min_m"], fr["h_max_m"])
    assert R.CORE.RESTORE_ROD_TOL_M == 0.008 and R.CORE.RESTORE_ENGAGE_M == 0.02
    assert R._P().Z2_POST["post_settle_steps_per_s"] == 0.5 and R._P().Z2_POST["post_settle_taps_per_s"] == 1.0
    assert (R.K1_WINDOW_S, R.K1_ARRIVAL_DELAY_S, R.K4_ONSET_EXTRA_S) == (2.0, 1.0, 1.5)
    assert R.row_gates({}) == {WALK_ROW: ("K1", "K4", "K6"), RAISE_ROW: ("K2", "K4")}
    TC = R._tc()
    assert [t for t in TC.K3_TOPS_M] == fr["k3_eval_tops_m"] and TC.CMD_MARGIN_EVAL_M == 0.31
    assert len(TC.K3_CELLS) == 15 and TC.ENVS_PER_CELL == 100
    assert TC.K3_GATED_CELLS == tuple(f"{s}__{p}" for s in ("table_0p635", "table_0p6985", "table_0p762")
                                      for p in ("knee_folded", "random"))
    assert list(TC.K3_REPORTED_TOPS_M) == fr["k3_reported_tops_m"] and list(TC.K3_GATED_POSES) == fr["k3_gated_poses"]
    # the reader's slot / effective keys are the contract's, and the effective vector is re-derived
    assert R.RAW_SLOT != R.RAW_EFF and {R.RAW_SLOT, R.RAW_EFF} <= {getattr(R.EC, "RAW_STAND_SLOT", None),
                                                                   R.EC.RAW_STAND_MODE,
                                                                   getattr(R.EC, "RAW_STAND_EFF", None)}
    # the control table is the plan's: known-good R_h everywhere, the listed known-bads
    assert {r for (g, r), v in R.EXPECT.items() if v == "PASS"} == {"R_h", "Amask"}
    assert {(g, r) for (g, r), v in R.EXPECT.items() if v == "FAIL"} == {
        ("K1", "R_old"), ("K1", "Amask_old"), ("K2", "R_old"), ("K2", "Amask_old"), ("K3", "S1"),
        ("K4", "no_restore"), ("K6", "posture")}


# ===================================================================================== the synthetic world: controls
def test_control_table_lands(world):
    out = world["out"]
    assert out["mismatches"] == [], out["mismatches"]
    bad = [x for x in out["control_table"] if x["lands"] != "yes"]
    assert not bad, bad
    assert len(out["control_table"]) == 14


def test_known_bads_fail_where_the_plan_says(world):
    g = world["out"]["gates"]
    k1 = g["K1"]["R_old"]["clauses"]
    assert not k1["median bin0 es1 <= 0.02"]["ok"] and not k1["median bin4 es1 <= 0.02"]["ok"]
    assert k1["median bin2 es1 <= 0.02"]["ok"]                         # 1.064 is reachable on the old path
    assert math.isinf(g["K2"]["Amask_old"]["clauses"]["time-to-2cm excess p90 es2 <= 0 s"]["value"])
    nr = g["K4"]["no_restore"]
    assert nr["verdict"] == "FAIL" and nr["clauses"]["miss es1 <= max(1, R_h's)"]["value"] == 2 * (4 + 2)
    assert nr["reported"]["es1"]["status"]["miss"] == 12 and nr["reported"]["es1"]["status"]["plain"] == 2
    po = g["K6"]["posture"]
    assert all(not c["ok"] for c in po["clauses"].values() if c["short"] == "homed" and c["state"] == "eval")
    assert po["clauses"]["stillness taps/s pooled es1 <= 1"]["ok"] is False
    assert g["K3"]["S1"]["clauses"]["STOOD rate tables es1 >= R_h's - max(0.05, 2 SE)"]["value"] == 0.0


def test_rh_reads(world):
    g = world["out"]["gates"]
    for k in ("K1", "K2", "K3", "K4", "K5", "K6"):
        assert g[k]["R_h"]["verdict"] == "PASS", (k, g[k]["R_h"].get("failed"), g[k]["R_h"].get("pending"))
    rep = g["K4"]["R_h"]["reported"]["es1"]
    assert rep["status"] == {"plain": 2, "complete": 12, "miss": 0, "cut": 0, "fell": 0, "merged": 0}
    assert rep["timeouts_raw"] == 0
    # restore time = the ramp (dh / 0.08) + the 10-row settle (completion on the 10th row after arrival): the
    # 1.30 -> 1.064 restore; the plant lag (~5 mm) stays inside the rod tolerance
    assert 0.236 / 0.08 + 0.2 - 2 * DT <= rep["restore_s"]["max"] <= 0.236 / 0.08 + 0.6
    assert rep["handback_abs_h_minus_1064_m"]["max"] < 0.02
    k2 = g["K2"]["R_h"]["reported"]["es1"]
    assert k2["events"] == 2 * 3 and k2["short"] == 0                  # 1.064->0.80, 0.80->1.28, walk->0.92
    assert set(k2["by_dir"]) == {"raise", "lower"}
    k6 = g["K6"]["R_h"]["reported"]["es1"]
    assert all(b["homed_share"] == 1.0 and b["opened"] >= 2 and b["steps_per_s"] == 0.0 for b in k6.values())
    assert world["out"]["premise_rh_standup"]["misses"] == []
    assert world["out"]["reads"]["R_h"]["es1"][WALK_ROW]["dose"]["rows_hip_below_m1"] > 0   # q past the old -1


# ===================================================================================== heads
def test_clean_head_passes_every_gate(world):
    h = world["out"]["heads"]["HEAD"]
    assert h["verdicts"] == {k: "PASS" for k in ("K1", "K2", "K3", "K4", "K5", "K6")} and h["all_pass"]
    assert h["stand_sha256"] == [HEAD_SHA["HEAD"]]


def test_bad_heads_fail_their_gate(world):
    hv = {k: v["verdicts"] for k, v in world["out"]["heads"].items()}
    assert hv["HEAD_TAP"]["K6"] == "FAIL" and hv["HEAD_TAP"]["K1"] == "PASS" and hv["HEAD_TAP"]["K4"] == "PASS"
    tap = world["out"]["gates"]["K6"]["HEAD_TAP"]["clauses"]["stillness taps/s pooled es1 <= 1"]
    assert tap["value"] > 1.0 and not tap["ok"]
    assert hv["HEAD_SLOW"]["K2"] == "FAIL"
    assert hv["HEAD_FALL"]["K2"] == "FAIL" and hv["HEAD_FALL"]["K1"] == "FAIL"
    fc = world["out"]["gates"]["K2"]["HEAD_FALL"]["clauses"]["stand falls es1 <= max(1, R_h's)"]
    assert fc["value"] == 2 and fc["line"] == 1
    k1f = world["out"]["gates"]["K1"]["HEAD_FALL"]["per_bin"]["es1 bin0"]
    assert k1f["fail_closed_holds"] == 2 and "fall in the hold" in k1f["why_fail_closed"]


def test_missing_raws_fail_closed_on_their_row_and_missing_stamps_pend(world):
    hv = {k: v["verdicts"] for k, v in world["out"]["heads"].items()}
    # sh_h_lim missing in the WALK row: every gate that row feeds is INVALID; K2 (the RAISE row only) still reads
    assert {hv["HEAD_BADKEY"][k] for k in ("K1", "K4", "K6")} == {"INVALID"} and hv["HEAD_BADKEY"]["K2"] == "PASS"
    r = world["out"]["reads"]["HEAD_BADKEY"]["es1"][WALK_ROW]
    assert r["status"] == "invalid" and any("sh_h_lim" in x and "fail closed" in x for x in r["reasons"])
    assert {hv["HEAD_MISSING"][k] for k in ("K1", "K2", "K4", "K6")} == {"PENDING"}
    assert hv["HEAD_MISSING"]["K3"] == "PENDING" and hv["HEAD_MISSING"]["K5"] == "PENDING"


def test_k5_zero_command_check_and_sn_calib_rows(R, world):
    g = world["out"]["gates"]["K5"]
    assert g["R_h"]["sn_calib_row"] == "R" and g["Amask"]["sn_calib_row"] == "SN_Amask_s17"
    assert g["HEAD"]["verdict"] == "PASS"
    assert g["HEAD_TAP"]["verdict"] == "INVALID"                       # its zero-command stamp commands 1.10
    assert any("not a zero-command pass" in x for x in g["HEAD_TAP"]["invalid"])
    # a failing / pending sn_calib gate; a head whose stamps run another S1; mixed checkpoints across stamps
    snc = fake_snc(R, {}, {("SYN_HEAD_z0", "G2"): "FAIL", ("SN_Amask_s17", "G4"): "PENDING"})
    res = R.k5_read({"k5": {}}, {"HEAD": "SYN_HEAD_z0"}, snc=snc)["gates"]
    assert res["HEAD"]["verdict"] == "FAIL" and res["Amask"]["verdict"] == "PENDING"
    assert res["R_h"]["verdict"] == "PASS"
    snc = fake_snc(R, world["zst"])
    snc.valid_only_point = lambda name: {"valid": {"suite@1": {"ckpt_sha": "9" * 64}}}
    res = R.k5_read({"k5": {}}, {"HEAD": "SYN_HEAD_z0"}, snc=snc)["gates"]
    assert res["HEAD"]["verdict"] == "INVALID" and any("!= REFS" in x for x in res["HEAD"]["invalid"])
    mixed = {"SYN_HEAD_z0": {"suite@1": world["zst"]["SYN_HEAD_z0"]["suite@1"],
                             "suite@2": world["zst"]["SYN_TAP_z0"]["suite@2"]}}
    res = R.k5_read({"k5": {}}, {"HEAD": "SYN_HEAD_z0"}, snc=fake_snc(R, mixed))["gates"]
    assert res["HEAD"]["verdict"] == "INVALID" and any("mix stand checkpoints" in x for x in res["HEAD"]["invalid"])
    assert R.zero_command_reasons(None) == []


def test_k5_on_the_real_sn_calib_synthetic_reads(R):
    snc = R._snc()
    w = snc._syn_world()
    w["H_ok"] = snc._syn_reads("H_ok", kind="obs0")
    w["H_bad"] = snc._syn_reads("H_bad", kind="obs0",
                                still={1: snc._syn_still(steps=2.0), 2: snc._syn_still(steps=2.0)})
    out = snc.calibrate(w, heads=("H_ok", "H_bad"))
    res = R.k5_from_calib(out, {"R_h": "R", "A": "H_ok", "B": "H_bad"})
    assert res["R_h"]["verdict"] == "PASS" and res["A"]["verdict"] == "PASS"
    assert res["B"]["verdict"] == "FAIL" and "G1 STILL" in res["B"]["failed"]
    res = R.k5_from_calib(out, {"A": "H_ok"}, {"A": ["a campaign reason"]})
    assert res["A"]["verdict"] == "INVALID"


def test_one_stand_checkpoint_per_head(R, world):
    spec = copy.deepcopy(world["spec"])
    spec["roles"] = {k: v for k, v in spec["roles"].items() if k in ("R_h", "HEAD")}
    spec["roles"]["HEAD"]["k5_name"] = "SYN_TAP_z0"                     # another checkpoint's zero-command stamps
    out = R.run(spec, k5=True, forces=False, snc=fake_snc(R, world["zst"]))
    h = out["heads"]["HEAD"]
    assert set(h["verdicts"].values()) == {"INVALID"} and len(h["stand_sha256"]) == 2


# ===================================================================================== K3 contract + premise
def test_k3_contract_fails_closed(R, tmp_path):
    TC = R._tc()
    d = write_cells(R, tmp_path, "X", 1, "r_h", {"good": 100})
    t = R.read_table_cells([str(d)], 1, controller="r_h")
    assert t["reasons"] == [] and sorted(t["cells"]) == sorted(TC.K3_CELLS)
    assert R.read_table_cells([str(d)], 2, controller="r_h")["reasons"]          # the wrong env seed
    assert R.read_table_cells([str(d)], 1, controller="stand_net")["reasons"]    # another controller's cells
    assert R.read_table_cells([str(d)], 1, controller="r_h", amask=True)["reasons"]   # not the A-mask file
    f = sorted(d.glob("*table_0p635__knee_folded.json"))[0]
    rec = json.loads(f.read_text())
    rec["aggregate"]["n_stood"] -= 1                                    # disagrees with its per-env records
    rec["per_env"][0]["h_cmd_m"] = 0.635 + 0.30                          # below top + 0.31
    f.write_text(json.dumps(rec))
    r = R.read_table_cells([str(d)], 1, controller="r_h")["reasons"]
    assert any("disagrees" in x for x in r) and any("command <" in x for x in r)
    g = R.gate_k3("X", {"X": {1: R.read_table_cells([str(d)], 1)}, "R_h": {1: R.read_table_cells([str(d)], 1)}},
                  seeds=(1,))
    assert g["verdict"] == "INVALID"
    rec["meta"]["smoke"] = True
    f.write_text(json.dumps(rec))
    t = R.read_table_cells([str(f)], 1)
    assert not t["cells"] and t["notes"] and not t["reasons"]


def test_k3_lines_and_premise(R, tmp_path):
    TC = R._tc()
    rh = R.read_table_cells([str(write_cells(R, tmp_path, "R_h", 1, "r_h", {"good": 100}))], 1)
    lo = R.read_table_cells([str(write_cells(R, tmp_path, "LO", 1, "x", {"good": 90, "march": 10}))], 1)
    fa = R.read_table_cells([str(write_cells(R, tmp_path, "FA", 1, "x", {"good": 98, "fall": 2}))], 1)
    g = R.gate_k3("LO", {"LO": {1: lo}, "R_h": {1: rh}}, seeds=(1,))
    assert g["verdict"] == "FAIL"                                       # 0.90 < 1.0 - max(0.05, 2 SE)
    g = R.gate_k3("FA", {"FA": {1: fa}, "R_h": {1: rh}}, seeds=(1,))
    assert g["verdict"] == "FAIL" and "falls/100 tables es1 <= max(1, R_h's)" in g["failed"]
    pm = R.k3_premise({"R_h": {1: rh}}, seeds=(1,))
    assert pm["misses"] == []
    bad = R.read_table_cells([str(write_cells(R, tmp_path, "RB", 1, "r_h", {"good": 100},
                                              overrides={"ground__rod_min": {"good": 85, "march": 15}}))], 1)
    assert R.k3_premise({"R_h": {1: bad}}, seeds=(1,))["misses"] == []          # ground: reported, never a miss
    bad = R.read_table_cells([str(write_cells(R, tmp_path, "RG", 1, "r_h", {"good": 100},
                                              overrides={"table_0p6985__knee_folded": {"good": 70, "march": 30}}))], 1)
    pm = R.k3_premise({"R_h": {1: bad}}, seeds=(1,))                             # 170 / 200 on that gated top
    assert pm["misses"] == ["es1 top table_0p6985"] and pm["gated_cells"] == list(TC.K3_GATED_CELLS)
    assert pm["seeds"]["es1"]["top table_0p6985"]["n_envs"] == 200
    short = R.read_table_cells([str(write_cells(R, tmp_path, "SH", 1, "x", {"good": 60}))], 1)
    assert R.gate_k3("SH", {"SH": {1: short}, "R_h": {1: rh}}, seeds=(1,))["verdict"] == "PENDING"
    part = R.read_table_cells([str(write_cells(R, tmp_path, "PA", 1, "x", {"good": 100}, cells=TC.K3_CELLS[:6]))], 1)
    assert R.gate_k3("PA", {"PA": {1: part}, "R_h": {1: rh}}, seeds=(1,))["verdict"] == "PENDING"


def test_k3_reads_the_gated_cells_only(R, tmp_path):
    """Step-4 re-scope: the 6 gated cells decide K3 and the R_h premise; a reported cell (the 18-in table, every rod_min
    cell, every ground cell) is read when present, never required, and never flips a line, the premise or the verdict
    -- even a terrible one, or one whose file breaks the contract (a note, not INVALID)."""
    TC = R._tc()
    rh = R.read_table_cells([str(write_cells(R, tmp_path, "R_h", 1, "r_h", {"good": 100}))], 1)
    hd = R.read_table_cells([str(write_cells(R, tmp_path, "HD", 1, "x", {"good": 100}))], 1)
    base = R.gate_k3("HD", {"HD": {1: hd}, "R_h": {1: rh}}, seeds=(1,))
    assert base["verdict"] == "PASS" and set(base["clauses"]) == {
        "STOOD rate tables es1 >= R_h's - max(0.05, 2 SE)", "falls/100 tables es1 <= max(1, R_h's)"}
    assert base["gated_cells"] == list(TC.K3_GATED_CELLS) and "6 GATED cells" in base["statistic"]
    rep = base["reported"]["es1"]
    assert rep["gated_cells_present"] == list(TC.K3_GATED_CELLS) and len(rep["reported_cells_present"]) == 9
    assert rep["per_cell"]["ground__rod_min"]["gated"] is False and rep["per_cell"]["table_0p635__random"]["gated"]
    assert rep["reported_lines"]["ground"]["gated"] is False and rep["reported_lines"]["ground"]["pass"] is True
    terrible = {c: {"march": 100} for c in TC.K3_REPORTED_CELLS}
    for role, ov in (("HT", terrible), ("RT", terrible)):
        write_cells(R, tmp_path, role, 1, "x" if role == "HT" else "r_h", {"good": 100}, overrides=ov)
    ht = R.read_table_cells([str(tmp_path / "tables" / "HT" / "es1")], 1)
    rt = R.read_table_cells([str(tmp_path / "tables" / "RT" / "es1")], 1)
    for subj, ref in ((ht, rh), (hd, rt), (ht, rt)):
        g = R.gate_k3("X", {"X": {1: subj}, "R_h": {1: ref}}, seeds=(1,))
        assert g["verdict"] == base["verdict"] and g["clauses"] == base["clauses"]
    assert R.k3_premise({"R_h": {1: rt}}, seeds=(1,))["misses"] == []
    assert ht["per_cell"]["ground__knee_folded"]["stood_share"] == 0.0                 # still read and reported
    # reported cells absent: never PENDING; a gated cell absent: PENDING
    only = R.read_table_cells([str(write_cells(R, tmp_path, "OG", 1, "x", {"good": 100}, cells=TC.K3_GATED_CELLS))], 1)
    assert R.gate_k3("OG", {"OG": {1: only}, "R_h": {1: rh}}, seeds=(1,))["verdict"] == "PASS"
    assert R.k3_premise({"R_h": {1: only}}, seeds=(1,))["misses"] == []
    five = R.read_table_cells([str(write_cells(R, tmp_path, "G5", 1, "x", {"good": 100},
                                               cells=TC.K3_GATED_CELLS[:5]))], 1)
    assert R.gate_k3("G5", {"G5": {1: five}, "R_h": {1: rh}}, seeds=(1,))["verdict"] == "PENDING"
    assert R.k3_premise({"R_h": {1: five}}, seeds=(1,))["seeds"]["es1"]["state"] == "pending"
    # a reported cell's contract problem is a note; a gated cell's is INVALID
    d = tmp_path / "tables" / "HD" / "es1"
    for cell, want in (("ground__rod_min", "PASS"), ("table_0p762__knee_folded", "INVALID")):
        f = d / f"HD_s1__{cell}.json"
        keep = f.read_text()
        rec = json.loads(keep)
        rec["aggregate"]["n_stood"] -= 1
        f.write_text(json.dumps(rec))
        t = R.read_table_cells([str(d)], 1)
        assert R.gate_k3("HD", {"HD": {1: t}, "R_h": {1: rh}}, seeds=(1,))["verdict"] == want, cell
        if want == "PASS":
            assert not t["reasons"] and any("[reported cell, not gated]" in x and "disagrees" in x for x in t["notes"])
        f.write_text(keep)
    # the head's one-checkpoint rule reads the gated cells' shas; a reported cell from another file is a note
    write_cells(R, tmp_path, "SA", 1, "x", {"good": 100}, sha="a" * 64, cells=TC.K3_GATED_CELLS)
    write_cells(R, tmp_path, "SA", 1, "x", {"good": 100}, sha="b" * 64, cells=TC.K3_REPORTED_CELLS)
    t = R.read_table_cells([str(tmp_path / "tables" / "SA" / "es1")], 1)
    assert t["stand_sha256"] == ["a" * 64] and t["stand_sha256_reported"] == ["b" * 64] and not t["reasons"]
    assert any("[reported cells, not gated] stand checkpoints" in x for x in t["notes"])


def test_markdown_says_pending_premise_and_shows_reported_checkpoints(R, world, tmp_path):
    """Review fixes: a pending R_h premise (a gated cell missing / short) reads "pending (...)" in the markdown, not
    "no miss read"; the reported cells' stand checkpoints are recorded and printed beside the gated cells'."""
    TC = R._tc()
    meta = {"generated": "t", "tool_sha256": "0" * 64, "exit_code": 2}
    out = copy.deepcopy(world["out"])
    assert "no miss read" in R.markdown(dict(out, meta=meta))
    five = R.read_table_cells([str(write_cells(R, tmp_path, "R5", 1, "r_h", {"good": 100},
                                               cells=TC.K3_GATED_CELLS[:5]))], 1)
    pm = R.k3_premise({"R_h": {1: five}}, seeds=(1,))
    assert pm["misses"] == [] and pm["seeds"]["es1"]["state"] == "pending"
    md = R.markdown(dict(out, premise_rh_standup=pm, meta=meta))
    line = next(ln for ln in md.splitlines() if ln.startswith("**R_h stand-up premise"))
    assert "pending (es1: " in line and TC.K3_GATED_CELLS[5] in line and "no miss read" not in line
    # the stand checkpoints: the run record carries both lists; the markdown flags a reported-only checkpoint
    assert out["tables"]["HEAD"]["es1"]["stand_sha256"] == [HEAD_SHA["HEAD"]]
    assert out["tables"]["HEAD"]["es1"]["stand_sha256_reported"] == [HEAD_SHA["HEAD"]]
    assert out["gates"]["K3"]["HEAD"]["reported"]["es1"]["stand_sha256_reported"] == [HEAD_SHA["HEAD"]]
    md = R.markdown(dict(out, meta=meta))
    assert "**K3 stand checkpoints" in md and "another checkpoint" not in md
    out["tables"]["HEAD"]["es1"]["stand_sha256_reported"] = ["9" * 64]
    md = R.markdown(dict(out, meta=meta))
    row = next(ln for ln in md.splitlines() if ln.startswith("- HEAD es1: gated cells' stand checkpoint"))
    assert "9" * 12 in row and "reported cells ran another checkpoint" in row


# ===================================================================================== contract + K4 + forces
def test_stamp_contract(R, world):
    lr, rg = world["lr"], world["rg"]
    good = lr / "SYN_HEAD" / WALK_ROW / "seed001" / STAMP_TS
    rule = R.role_rule("HEAD", True)
    assert R.stamp_contract_reasons(good, row=WALK_ROW, gates=rg[WALK_ROW], rule=rule) == []
    assert any("not this role's" in x for x in R.stamp_contract_reasons(good, rule=R.role_rule("R_h", False)))
    assert any("needs False" in x for x in R.stamp_contract_reasons(good, rule=R.role_rule("no_restore", False)))
    assert any("gates" in x for x in R.stamp_contract_reasons(good, row=WALK_ROW, gates=rg[RAISE_ROW]))
    assert any("scenario" in x for x in R.stamp_contract_reasons(good, row=RAISE_ROW))
    m = json.loads((good / "run_meta.json").read_text())
    blk = lambda x: x["p2r_mode"][R.EC.BLOCK_KEY]  # noqa: E731
    for mutate, want in ((lambda x: blk(x)["hold_heights"].pop(), "one entry per hold"),
                         (lambda x: blk(x)["hold_heights"].__setitem__(1, 0.65), "outside"),
                         (lambda x: blk(x)["hold_heights"].__setitem__(0, 1.0), "not a stand"),
                         (lambda x: blk(x)["controller"].__setitem__("rod_tol_m", 0.002), "rod_tol_m"),
                         (lambda x: blk(x).__setitem__("restore", "maybe"), "neither a bool"),
                         (lambda x: blk(x).__setitem__("h_meas", {"valid": False}), "h_meas not valid"),
                         (lambda x: x["p2r_mode"].__setitem__("variant", "stand_net"), "variant"),
                         (lambda x: x["p2r_mode"].pop(R.EC.BLOCK_KEY), "missing"),
                         (lambda x: x["env_vars"].__setitem__("KRABBY_FLAT_TERRAIN_MODE", "light"), "flat")):
        mm = copy.deepcopy(m)
        mutate(mm)
        assert any(want in r for r in R.meta_reasons(mm)), want
    ep = dict(world["eps"][("HEAD", WALK_ROW)][0])
    ep2 = dict(ep, **{R.RAW_H_CMD: np.where(ep["segment_id"] == 3, 1.20, ep[R.RAW_H_CMD])})
    assert any("did not reach the controller" in r for r in R.episode_reasons(ep2, m, 0))
    eff = ep[R.RAW_EFF].copy()
    eff[5] = ~eff[5]
    assert any("stand_eff = slot | restoring" in r for r in R.episode_reasons(dict(ep, **{R.RAW_EFF: eff}), m, 0))


def test_k4_a_restore_that_runs_into_the_next_stand_is_a_merge_not_a_hand_back(R):
    """stand 1.30 -> a 0.5-s walk (shorter than the ~3.1-s restore) -> stand 0.90 -> back: the first exit merges into
    the next stand (no hand-back, no miss); the second hands back after its own restore."""
    sched = [("fwd", 0.25, 0.0, 1.0, None), ("stand", 0.0, 0.0, 5.0, 1.30), ("fwd", 0.25, 0.0, 0.5, None),
             ("stand", 0.0, 0.0, 5.0, 0.90), ("back", 0.20, math.pi, 6.0, None)]
    meta = launcher_meta(R, sched, row="x", gates=("K4",))
    ep = simulate(R, sched, [{}])[0]
    assert R.episode_reasons(ep, meta, 0) == []
    k4 = R.read_episode(ep, R.Ctx(meta), 0)["k4"]
    assert [h["status"] for h in k4] == ["merged", "complete"] and not any(h["miss"] for h in k4)
    assert k4[0]["handback_row"] is None and k4[1]["onset_pair"] and k4[1]["onset_s"] is not None
    assert ep[R.EC.RAW_MERGED].any()                                     # the controller saw the same merge
    ep = simulate(R, sched, [dict(restore=False)])[0]
    k4 = R.read_episode(ep, R.Ctx(meta), 0)["k4"]
    assert [h["status"] for h in k4] == ["miss", "miss"] and all(h["restore_s"] == 0.0 for h in k4)


def test_k4_completion_is_the_recorded_event_and_the_rederivation_cross_checks(R, tmp_path):
    """Review fix: K4's complete / miss comes from the launcher's recorded H6 events (sh_completed / sh_timed_out at
    the hand-back row); the reader's joint-space re-derivation is a cross-check whose disagreement makes the episode
    (and the stamp) INVALID -- never a silent "miss" for every controller. Without the recorded events the reader
    falls back to the re-derivation."""
    sched = [("fwd", 0.25, 0.0, 1.0, None), ("stand", 0.0, 0.0, 5.0, 1.30), ("back", 0.20, math.pi, 6.0, None)]
    meta = launcher_meta(R, sched, row="x", gates=("K4",))
    ep = simulate(R, sched, [{}])[0]
    r = R.read_episode(ep, R.Ctx(meta), 0)
    (h,) = r["k4"]
    assert h["status"] == "complete" and h["completion_source"] == "recorded" and h["complete_rederived"] is True
    assert r["invalid"] == [] and bool(ep[R.RAW_COMPLETED][h["handback_row"]])
    t = h["handback_row"]
    # (a) the reader reads the hips differently from the controller: recorded complete, re-derived not -> INVALID
    jp = ep["joint_pos"].copy()
    jp[t - 3, 12] += 0.10                                                # a hip-pitch joint (the simulator's col 12)
    bad = dict(ep, joint_pos=jp)
    r = R.read_episode(bad, R.Ctx(meta), 0)
    assert r["k4"][0]["status"] == "complete" and r["k4"][0]["complete_rederived"] is False
    assert len(r["invalid"]) == 1 and "K4 cross-check" in r["invalid"][0]
    st = write_stamp(tmp_path / "lr", "XCHK", "x", 1, [bad], meta)
    res = R.read_height_stamp(str(st), row="x", gates=("K4",))
    assert res["status"] == "invalid" and any("K4 cross-check" in x for x in res["reasons"])
    # (b) a recorded timeout where the joints say complete: still a disagreement (INVALID), status miss
    tmo, comp = ep[R.RAW_TIMED_OUT].copy(), ep[R.RAW_COMPLETED].copy()
    tmo[t], comp[t] = True, False
    r = R.read_episode(dict(ep, **{R.RAW_TIMED_OUT: tmo, R.RAW_COMPLETED: comp}), R.Ctx(meta), 0)
    assert r["k4"][0]["status"] == "miss" and r["k4"][0]["miss"] and "K4 cross-check" in r["invalid"][0]
    # (c) an armed restore that hands back with neither event recorded: INVALID
    comp[t], tmo[t] = False, False
    r = R.read_episode(dict(ep, **{R.RAW_TIMED_OUT: tmo, R.RAW_COMPLETED: comp}), R.Ctx(meta), 0)
    assert any("neither" in x for x in r["invalid"])
    # (d) no recorded events (an older stamp): the re-derivation decides
    r = R.read_episode({k: v for k, v in ep.items() if k != R.RAW_COMPLETED}, R.Ctx(meta), 0)
    assert r["k4"][0]["completion_source"] == "rederived" and r["k4"][0]["status"] == "complete" and not r["invalid"]
    # (e) restore off (the K4 known-bad): no completion recorded, no timeout -> miss, and NOT invalid
    off = launcher_meta(R, sched, row="x", gates=("K4",), restore=False)
    r = R.read_episode(simulate(R, sched, [dict(restore=False)])[0], R.Ctx(off), 0)
    assert r["k4"][0]["status"] == "miss" and r["k4"][0]["completion_source"] == "recorded" and not r["invalid"]


def test_force_table_is_the_foot_load_statics_beside_the_pd_reading(R, world):
    frc = world["out"]["forces"]["R_h"]["pooled"]
    top = R.bin_label(4)
    b = frc[top]
    assert b["n_rows"] > 0 and b["F_hip_six_leg_mean_pd_torque_N"] > b["F_hip_six_leg_mean_foot_load_N"]
    F = R._facts()
    K = F.K
    rows = [F._settled_rows(world["lr"] / "SYN_R_h" / row / f"seed00{s}" / STAMP_TS)
            for s in (1, 2) for row in (WALK_ROW, RAISE_ROW)]
    th, tk = np.concatenate([r["th"] for r in rows]), np.concatenate([r["tk"] for r in rows])
    ff, h = np.concatenate([r["ff"] for r in rows]), np.concatenate([r["h"] for r in rows])
    m = np.array([R.height_bin(x) == 4 for x in h])
    fh, _fk = K.rod_forces(th, tk, *K.stance_joint_torques(th, tk, ff, leg_weight=True))
    assert b["F_hip_six_leg_mean_foot_load_N"] == pytest.approx(float(np.abs(fh[m]).mean(1).mean()), rel=1e-9)
    assert b["n_rows"] == int(m.sum())
    assert 1700 < b["F_hip_six_leg_mean_foot_load_N"] < 2000            # ~ the static model at 1.28 / 1.30 m
    hm = world["out"]["h_max_recheck"]
    assert hm["state"] == "ok" and hm["value_N"] == pytest.approx(b["F_hip_six_leg_mean_foot_load_N"])


def test_h_max_recheck_flags_a_refreeze_and_unreadable_raws(R, tmp_path):
    def rows(h, f):
        n = 50
        return {"h": np.full(n, h), "fl_h": np.full((n, 6), f), "fl_k": np.full((n, 6), 900.0),
                "pd_h": np.full((n, 6), f * 1.14), "pd_k": np.full((n, 6), 1000.0), "nload": np.full(n, 6)}
    assert R.h_max_recheck([rows(1.30, 1950.0)])["state"] == "ok"
    r = R.h_max_recheck([rows(1.30, 2050.0), rows(1.30, 2010.0)])
    assert r["state"] == "REFREEZE" and r["value_N"] == pytest.approx(2030.0)
    assert R.h_max_recheck([rows(1.0, 2500.0)])["state"] == "pending"   # nothing in the top bin
    # review fix: the bin mean (1.25 / 1.30 / 1.346 holds) can pass while F at h_max is over the cap -> reported,
    # FLAGGED, the pre-registered bin read stays the decision
    r = R.h_max_recheck([rows(1.25, 1950.0), rows(1.30, 1980.0), rows(1.346, 2030.0), rows(1.33, 2010.0)])
    assert r["state"] == "ok" and r["value_N"] == pytest.approx(1992.5)
    near = r["at_h_max"]
    assert near["n_rows"] == 100 and near["value_N"] == pytest.approx(2020.0) and near["over_cap"] is True
    assert near["window_m"] == pytest.approx([R.H_MAX - 0.02, R.H_MAX + 0.02])
    assert len(r["flags"]) == 1 and r["flags"][0].startswith("AT_H_MAX_OVER_CAP")
    assert R.exit_code({"mismatches": [], "premise_rh_standup": {"misses": []}, "h_max_recheck": r, "heads": {},
                        "pending": []}) == 0                              # a flag, not the pre-registered STOP
    ok = R.h_max_recheck([rows(1.25, 1900.0), rows(1.346, 1990.0)])
    assert ok["flags"] == [] and ok["at_h_max"]["over_cap"] is False
    assert R.h_max_recheck([rows(1.25, 1900.0)])["flags"][0].startswith("AT_H_MAX_NO_ROWS")
    t = R.force_table([rows(0.8, 1100.0), rows(1.30, 2050.0)])
    assert t[R.bin_label(4)]["over_cap_foot_load"] and not t[R.bin_label(0)]["over_cap_foot_load"]
    # an R_h stamp without the force raws: the gates still read, the re-check is INVALID (a STOP)
    lr = tmp_path / "lr"
    rg = R.row_gates({})
    for row, sched in ((WALK_ROW, WALK), (RAISE_ROW, RAISE)):
        eps = simulate(R, sched, [{}] * N_EPS, row)
        for s in (1, 2):
            write_stamp(lr, "NOF", row, s, eps, launcher_meta(R, sched, row=row, gates=rg[row]),
                        drop={0: ["applied_torque"]})
    spec = {"lr_root": str(lr), "refs_onset_s": {"1": 0.1, "2": 0.1}, "roles": {"R_h": {"name": "NOF"}}}
    out = R.run(spec, k5=False, forces=True)
    assert out["reads"]["R_h"]["es1"][WALK_ROW]["status"] == "ok"
    assert out["h_max_recheck"]["state"] == "INVALID" and "applied_torque" in out["h_max_recheck"]["errors"][0]
    assert R.exit_code(dict(out, premise_rh_standup={"misses": []})) == 4


def test_restore_arming_and_controller_must_match_the_role(R, world):
    spec = copy.deepcopy(world["spec"])
    spec["roles"] = {"R_h": {"name": "SYN_no_restore"}, "no_restore": {"name": "SYN_R_h"},
                     "posture": {"name": "SYN_R_h"}}
    spec["refs_onset_s"] = {"1": 0.1, "2": 0.1}
    out = R.run(spec, k5=False, forces=False)
    for role, want in (("R_h", "needs True"), ("no_restore", "needs False"), ("posture", "not this role's")):
        r = out["reads"][role]["es1"][WALK_ROW]
        assert r["status"] == "invalid" and any(want in x for x in r["reasons"]), (role, r["reasons"])
    assert out["gates"]["K4"]["no_restore"]["lands"] == "NO"           # INVALID on a listed cell is a mismatch


def test_refs_onset_comes_from_p2_read_walkstop_read(world):
    ro = world["out"]["refs_onset"]
    assert ro[1]["source"].startswith("p2_read.walkstop_read(REFS_SYN") and ro[1]["status"] == "ok"
    assert 0.0 < ro[1]["onset_median_s"] < 0.3


def test_exit_code_and_named_clause_rule(R, world):
    assert R.exit_code(world["out"]) == 3                                # heads fail, no control mismatch
    res = {"verdict": "FAIL", "clauses": {
        "miss es1": {"short": "miss", "seed": 1, "state": "eval", "ok": False},
        "miss es2": {"short": "miss", "seed": 2, "state": "eval", "ok": True},
        "exit falls es2": {"short": "exit falls", "seed": 2, "state": "eval", "ok": False}}}
    ld, why = R.lands("K4", "no_restore", res)
    assert ld == "NO" and "named clause" in why
    assert R.lands("K4", "no_restore", {"verdict": "INVALID"})[0] == "NO"
    assert R.lands("K1", "R_h", {"verdict": "PENDING"})[0] == "pending"
    assert R.lands("K1", "posture", {"verdict": "FAIL"})[0] == "reported"


# ===================================================================================== CLI
def test_cli_writes_json_and_md_only_where_told(R, world, tmp_path):
    spec = copy.deepcopy(world["spec"])
    spec["roles"] = {k: v for k, v in spec["roles"].items() if k in ("R_h", "HEAD", "no_restore")}
    spec["refs_onset_s"] = {"1": 0.12, "2": 0.12}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    j, m = tmp_path / "o" / "read.json", tmp_path / "o" / "read.md"
    rc = R.main(["--spec", str(sp), "--json", str(j), "--md", str(m), "--no-k5"])
    assert rc == 2                                                       # K5 not read + roles absent: pending
    d = json.loads(j.read_text())
    assert d["refs_onset"]["1"]["source"] == "spec override"
    assert d["meta"]["tool_sha256"] and d["meta"]["constants"]["h_max"] == 1.346
    assert d["gates"]["K4"]["no_restore"]["lands"] == "yes" and d["heads"]["HEAD"]["verdicts"]["K1"] == "PASS"
    assert "# Stand height -- K1-K6 read" in m.read_text()
    for bad in (PRED_SCRIPTS.parent / "x.json", P2_SCRIPTS / "x.md"):
        with pytest.raises(SystemExit):
            R.main(["--spec", str(sp), "--json", str(bad), "--md", str(tmp_path / "ok.md"), "--no-k5"])
        assert not bad.exists()
    with pytest.raises(SystemExit):                                      # no default output paths
        R.main(["--spec", str(sp)])
    bad_spec = tmp_path / "bad.json"
    bad_spec.write_text(json.dumps({"roles": {"MYSTERY": {"name": "x"}}}))
    with pytest.raises(SystemExit):
        R.main(["--spec", str(bad_spec), "--json", str(j), "--md", str(m)])


def test_no_bytecode_next_to_campaign_predecessor_or_phase2_files(world):
    assert not (SCRIPTS / "__pycache__").exists()
    assert _pyc_snapshot() == _PYC_BEFORE


# ===================================================================================== the residual deadband
# CHARTER section 3, 2026-10-14: a head's stamps / table cells must share ONE residual deadband (absent = 0); a mix is
# INVALID for every gate of that head (the one-checkpoint rule's shape); the head's output records the value.
EC_BLOCK = "stand_height"


def _set_deadband(stamps_root: Path, value, *, clip: bool = True):
    for f in Path(stamps_root).rglob("run_meta.json"):
        m = json.loads(f.read_text())
        b = m["p2r_mode"][EC_BLOCK]
        b["residual_deadband"] = value
        if clip:
            b["clip"] = {"residual_deadband": value}
        f.write_text(json.dumps(m))


def _set_cell_deadband(cell_dir: Path, value):
    for f in Path(cell_dir).glob("*.json"):
        rec = json.loads(f.read_text())
        rec["meta"]["residual_deadband"] = value
        f.write_text(json.dumps(rec))


def test_old_stamps_and_cells_read_as_deadband_zero(R, world):
    """Every synthetic stamp / cell of the world was written without the key: the clean head reads 0 and passes."""
    h = world["out"]["heads"]["HEAD"]
    assert h["residual_deadband"] == 0.0 and h["all_pass"]
    assert world["out"]["reads"]["HEAD"]["es1"][WALK_ROW]["residual_deadband"] == 0.0
    assert world["out"]["tables"]["HEAD"]["es1"]["residual_deadband"] == [0.0]
    assert "one value per head" in R.constants()["residual_deadband"]


def test_one_residual_deadband_per_head(R, world, tmp_path):
    lr = tmp_path / "lr"
    shutil.copytree(world["lr"], lr)
    roles = {k: copy.deepcopy(v) for k, v in world["spec"]["roles"].items() if k in ("R_h", "HEAD")}
    # (a) the head's table cells ran at 0.25, its height stamps without the key (0) -> a mix: every gate INVALID
    cells = tmp_path / "cells"
    tbl = {}
    for s in (1, 2):
        d = write_cells(R, cells, "HEAD", s, "stand_net", {"good": 99, "fall": 1}, sha=HEAD_SHA["HEAD"])
        _set_cell_deadband(d, 0.25)
        tbl[str(s)] = [str(d)]
    roles["HEAD"]["table"] = tbl
    spec = dict(world["spec"], lr_root=str(lr), roles=roles)
    out = R.run(spec, k5=False, forces=False)
    h = out["heads"]["HEAD"]
    assert set(h["verdicts"].values()) == {"INVALID"} and h["residual_deadband"] == [0.0, 0.25]
    inv = out["gates"]["K1"]["HEAD"]["invalid"]
    assert any("mix residual deadbands" in x for x in inv), inv
    assert out["heads"]["HEAD"]["stand_sha256"] == [HEAD_SHA["HEAD"]]          # one checkpoint: not the reason
    assert out["gates"]["K1"]["R_h"]["verdict"] == "PASS"                      # controls are not held to it
    # (b) every height stamp at 0.25 too: one deadband -> the head reads as before (K5 not read here), value recorded
    _set_deadband(lr / "SYN_HEAD", 0.25)
    out = R.run(spec, k5=False, forces=False)
    h = out["heads"]["HEAD"]
    assert h["residual_deadband"] == 0.25
    assert {g: v for g, v in h["verdicts"].items() if g != "K5"} == {g: "PASS" for g in ("K1", "K2", "K3", "K4", "K6")}
    assert out["reads"]["HEAD"]["es2"][RAISE_ROW]["residual_deadband"] == 0.25
    md = R.markdown(dict(out, meta={"generated": "t", "tool_sha256": "0" * 64, "exit_code": 2}))
    assert "- HEAD: residual deadband 0.25" in md
    # (c) one height stamp at another value -> a mix again
    one = lr / "SYN_HEAD" / WALK_ROW / "seed002"
    _set_deadband(one, 0.3)
    out = R.run(spec, k5=False, forces=False)
    assert set(out["heads"]["HEAD"]["verdicts"].values()) == {"INVALID"}
    assert out["heads"]["HEAD"]["residual_deadband"] == [0.25, 0.3]
    # (d) a malformed record (block vs its clip, or not a number) makes that stamp INVALID on its own
    _set_deadband(one, 0.25)
    f = next(one.rglob("run_meta.json"))
    m = json.loads(f.read_text())
    m["p2r_mode"][EC_BLOCK]["clip"] = {"residual_deadband": 0.3}
    f.write_text(json.dumps(m))
    r = R.read_height_stamp(str(f.parent), row=WALK_ROW, gates=world["rg"][WALK_ROW], rule=R.role_rule("HEAD", True))
    assert r["status"] == "invalid" and any("residual_deadband" in x for x in r["reasons"])
    m["p2r_mode"][EC_BLOCK]["residual_deadband"] = True
    m["p2r_mode"][EC_BLOCK].pop("clip")
    f.write_text(json.dumps(m))
    r = R.read_height_stamp(str(f.parent), row=WALK_ROW, gates=world["rg"][WALK_ROW], rule=R.role_rule("HEAD", True))
    assert r["status"] == "invalid" and any("residual_deadband" in x for x in r["reasons"])


def test_k5_stamps_join_the_one_deadband_rule(R, world, tmp_path):
    lr = tmp_path / "lr"
    shutil.copytree(world["lr"], lr)
    zst = {name: {k: v.replace(str(world["lr"]), str(lr)) for k, v in by.items()} for name, by in world["zst"].items()}
    roles = {k: copy.deepcopy(v) for k, v in world["spec"]["roles"].items() if k in ("R_h", "HEAD")}
    spec = dict(world["spec"], lr_root=str(lr), roles=roles)
    _set_deadband(lr / "SYN_HEAD_z0", 0.25)                     # the zero-command stamps at 0.25, the rest at 0
    out = R.run(spec, k5=True, forces=False, snc=fake_snc(R, zst))
    assert out["k5"]["head_residual_deadband"]["HEAD"] == [0.25]
    assert set(out["heads"]["HEAD"]["verdicts"].values()) == {"INVALID"}
    assert out["heads"]["HEAD"]["residual_deadband"] == [0.0, 0.25]


# the deadband a head's checkpoint was TRAINED with vs the one it was evaluated at (review of the 2026-10-14 option):
# recorded per stamp by the launcher; a difference is a NOTE on the head (both values kept), never a refusal
def _set_trained_deadband(stamps_root: Path, value):
    for f in Path(stamps_root).rglob("run_meta.json"):
        m = json.loads(f.read_text())
        m["p2r_mode"][EC_BLOCK]["trained_residual_deadband"] = value
        f.write_text(json.dumps(m))


def test_trained_vs_evaluated_deadband_is_a_head_note_not_a_refusal(R, world, tmp_path):
    h0 = world["out"]["heads"]["HEAD"]
    assert h0["trained_residual_deadband"] is None and h0["deadband_notes"] == []          # old stamps: none recorded
    lr = tmp_path / "lr"
    shutil.copytree(world["lr"], lr)
    roles = {k: copy.deepcopy(v) for k, v in world["spec"]["roles"].items() if k in ("R_h", "HEAD")}
    spec = dict(world["spec"], lr_root=str(lr), roles=roles)
    base = R.run(spec, k5=False, forces=False)
    # (a) trained at 0.25, evaluated without the record (0): the parity break is NOTED, the verdicts do not move
    _set_trained_deadband(lr / "SYN_HEAD", 0.25)
    out = R.run(spec, k5=False, forces=False)
    h = out["heads"]["HEAD"]
    assert h["verdicts"] == base["heads"]["HEAD"]["verdicts"] and h["residual_deadband"] == 0.0
    assert h["trained_residual_deadband"] == 0.25 and len(h["deadband_notes"]) == 1, h["deadband_notes"]
    assert "(0.0, 0.25)" in h["deadband_notes"][0] and "not a refusal" in h["deadband_notes"][0]
    assert out["reads"]["HEAD"]["es1"][WALK_ROW]["trained_residual_deadband"] == 0.25
    assert set(out["heads"]) == {"HEAD"}                                                  # controls carry no note
    md = R.markdown(dict(out, meta={"generated": "t", "tool_sha256": "0" * 64, "exit_code": 2}))
    assert "trained at 0.25" in md and "**NOTE** [HEAD]" in md
    # (b) trained and evaluated equal on every stamp (both 0.25): no note (the table cells, still at 0, make the
    # one-deadband mix -- that rule, not this note, reads them)
    _set_deadband(lr / "SYN_HEAD", 0.25)
    out = R.run(spec, k5=False, forces=False)
    h = out["heads"]["HEAD"]
    assert h["trained_residual_deadband"] == 0.25 and h["deadband_notes"] == []
    assert out["reads"]["HEAD"]["es2"][RAISE_ROW]["residual_deadband"] == 0.25
    # (c) a null trained value (not a campaign-trained checkpoint) is "not recorded": no note
    _set_trained_deadband(lr / "SYN_HEAD", None)
    assert R.run(spec, k5=False, forces=False)["heads"]["HEAD"]["deadband_notes"] == []
    # (d) a malformed trained value makes that stamp INVALID on its own
    f = next((lr / "SYN_HEAD" / WALK_ROW / "seed002").rglob("run_meta.json"))
    m = json.loads(f.read_text())
    m["p2r_mode"][EC_BLOCK]["trained_residual_deadband"] = True
    f.write_text(json.dumps(m))
    r = R.read_height_stamp(str(f.parent), row=WALK_ROW, gates=world["rg"][WALK_ROW], rule=R.role_rule("HEAD", True))
    assert r["status"] == "invalid" and any("trained_residual_deadband" in x for x in r["reasons"]), r["reasons"]


def test_k5_stamps_join_the_trained_deadband_note(R, world, tmp_path):
    lr = tmp_path / "lr"
    shutil.copytree(world["lr"], lr)
    zst = {name: {k: v.replace(str(world["lr"]), str(lr)) for k, v in by.items()} for name, by in world["zst"].items()}
    roles = {k: copy.deepcopy(v) for k, v in world["spec"]["roles"].items() if k in ("R_h", "HEAD")}
    spec = dict(world["spec"], lr_root=str(lr), roles=roles)
    _set_trained_deadband(lr / "SYN_HEAD_z0", 0.25)             # only the zero-command stamps record a trained 0.25
    base = R.run(dict(world["spec"], lr_root=str(world["lr"]), roles=roles), k5=True, forces=False,
                 snc=fake_snc(R, world["zst"]))
    out = R.run(spec, k5=True, forces=False, snc=fake_snc(R, zst))
    par = out["k5"]["head_deadband_parity"]["HEAD"]
    assert par and {(x["evaluated"], x["trained"]) for x in par} == {(0.0, 0.25)}
    h = out["heads"]["HEAD"]
    assert h["verdicts"] == base["heads"]["HEAD"]["verdicts"] and h["trained_residual_deadband"] == 0.25
    assert len(h["deadband_notes"]) == 1 and f"{len(par)} of " in h["deadband_notes"][0]
    _set_trained_deadband(lr / "SYN_HEAD_z0", "x")              # malformed on a K5 stamp: a K5 reason (INVALID)
    out = R.run(spec, k5=True, forces=False, snc=fake_snc(R, zst))
    assert out["heads"]["HEAD"]["verdicts"]["K5"] == "INVALID"
    assert any("trained_residual_deadband" in x for x in out["k5"]["campaign_reasons"]["HEAD"])
