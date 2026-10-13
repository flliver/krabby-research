#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``), PLAN H7 "Reported: ... observe pass" / Verification "observe
pass per head" -- the closed-loop OBSERVE PASS of a stand_height head (CPU, read-only, gates nothing).

A THIN WRAPPER of the closed predecessor's ``sn_observe_read.py`` (loaded by path as a private instance, never edited):
its pure statistics are reused unchanged -- ``Acc`` (stand runs, LIFT onsets, the HOLD-cam / LIFT-hip self-checks),
``summarize`` (per-branch |applied - R|, mu_hip on LIFT rows, LIFT-onset launch, HOLD cam-trim slope, counts),
``Acc.pool``, ``markdown``. What this wrapper changes is only what a stand_height stamp means:

* ROWS: the head's zero-command rows (``walkstop@1/2``, ``suite@1/2`` -- the K5 passes) and its height rows
  (``hwalkstop@1/2``, ``hraise@1/2`` -- scenarios_sh.yaml), each the newest COMPLETE stamp under
  ``lr/<HEAD>/<row>__omni/seed00<k>/`` (``sn_observe_read.stamp_complete``'s rule); families ``zero`` / ``height`` /
  ``all`` pooled.
* STAND ROWS = ``p2r_stand_mode``, which the stand_height launcher records as WHO ACTED (slot | restoring: H6
  restore rows are stand-network rows, PLAN H6).
* THE RESIDUAL FRAME (H1 / H3): at a height h the stand net executes ``q(h) + clip(a_net, +-1)`` and R_h's target is
  ``q(h) + R's law`` (HOLD / WAIT / SEAT = q, LIFT / HOME = q + the D3 lift, cams = R's). Everything is read relative
  to the posture the launcher recorded (``sh_q18``): "applied" = ``clip(sh_a_net, +-1)`` (the predecessor's
  clip(stand action, +-1)), "R" = ``sh_rh_target - sh_q18``. At 1.064 (q = 0, every zero-command row) this is the
  predecessor's read bit for bit; its self-checks hold in this frame (R_h's cams are R's; its LIFT hip residual is
  ``lift``, float32 rounding <= 5e-7 < the 1e-6 check).
* VALIDITY (per stamp, fail closed -> INVALID, exit 3): ``p2r_mode.variant == stand_height``; ``stand_mask`` an int
  0 / 1; ``specialist.sha256`` 64 hex; the stand_height block's ``control`` a NETWORK controller (stand_net /
  amask_old: an observe pass reads a network), its ``r_knob`` == the D3 constants, its ``command`` the row's (zero on
  walkstop / suite, manifest on the height rows); ``raw_series`` carries every raw read here;
  ``sn_common.env_reasons`` on ``run_meta.env_vars``; six cam joints; ``dt`` > 0. Across the head's stamps: one stand
  checkpoint sha, one mask, one S1 sha. Per episode (this wrapper's own routing check, the predecessor's is
  satisfied by construction): the harness's recorded ``actions`` on stand rows == ``where(q != 0, a_net + q, a_net)``
  bitwise (the launcher's pre-clip record of a network controller) and ``sh_stand_action`` == ``q + clip(a_net)``
  (rods clamped to +-4.8, cams +-1; <= 1e-6).

CLI (CPU only; writes ONLY the two given paths)::

    CUDA_VISIBLE_DEVICES="" python sh_observe_read.py --head NAME [--rows walkstop@1 ... hraise@2]
                                                      --json OUT.json --md OUT.md [--lr DIR]

    rows: <walkstop|suite|hwalkstop|hraise>[__omni]@<env seed>. Exit: 0 every row READ; 2 a row PENDING (none
    INVALID); 3 a row INVALID (its reasons in the JSON; nothing from it is pooled); 1 usage.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # every module below is loaded by path: never write __pycache__ beside them

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PRED_SCRIPTS = HERE.parents[1] / "2026-10-11_stand_network" / "scripts"
OBS_PATH = PRED_SCRIPTS / "sn_observe_read.py"
EC_PATH = HERE / "sh_eval_common.py"
REPO = HERE.parents[6]
LR_DEFAULT = REPO / "parkour/outputs/omni_plan/phase2/lr"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the predecessor's observe read (a PRIVATE instance: its pure statistics; its module globals are never rebound)
OBS = _load(OBS_PATH, "_sh_observe_sn_observe_read")
SN = OBS.SN
#: this campaign's eval contract (run_meta / raw names; torch-free at import)
EC = _load(EC_PATH, "_sh_observe_eval_common")

ROW_DIRS = {"walkstop": "walkstop__omni", "suite": "suite__omni", "hwalkstop": "hwalkstop__omni",
            "hraise": "hraise__omni"}
ZERO_ROWS = ("walkstop", "suite")
HEIGHT_ROWS = ("hwalkstop", "hraise")
ROWS_DEFAULT = ("walkstop@1", "walkstop@2", "suite@1", "suite@2", "hwalkstop@1", "hwalkstop@2", "hraise@1",
                "hraise@2")
FAMILIES = {"zero": ZERO_ROWS, "height": HEIGHT_ROWS}
RAW_STAND = EC.RAW_STAND_MODE                    # p2r_stand_mode: who acted (slot | restoring)
RAW_READ = (RAW_STAND, "actions", "joint_pos", EC.RAW_A_NET, EC.RAW_Q, EC.RAW_RH_TARGET, EC.RAW_RH_BRANCH,
            EC.RAW_STAND_ACTION, EC.RAW_S1_ACTION)
EXEC_TOL = 1e-6
READ, PENDING, INVALID = OBS.READ, OBS.PENDING, OBS.INVALID
RC_OK, RC_USAGE, RC_PENDING, RC_INVALID = OBS.RC_OK, OBS.RC_USAGE, OBS.RC_PENDING, OBS.RC_INVALID
FRAME_NOTE = ("Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = "
              "p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the "
              "executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at "
              "1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), "
              "height = hwalkstop + hraise (scenarios_sh.yaml).")


# ===================================================================================== rows + stamps
def parse_row(spec: str) -> tuple[str, int]:
    """``hwalkstop@1`` / ``suite__omni@2`` -> (``hwalkstop``, 1). ValueError otherwise."""
    m = re.fullmatch(r"(walkstop|suite|hwalkstop|hraise)(?:__omni)?@([1-9][0-9]*)", str(spec).strip())
    if not m:
        raise ValueError(f"bad row {spec!r}: want <walkstop|suite|hwalkstop|hraise>[__omni]@<env seed>")
    return m.group(1), int(m.group(2))


def resolve(lr: Path, head: str, row: str, seed: int) -> tuple[Path | None, list[str]]:
    """``(newest complete stamp or None, newer incomplete stamps)`` under ``lr/<head>/<row>__omni/seed00<k>/``."""
    d = Path(lr) / head / ROW_DIRS[row] / f"seed{seed:03d}"
    dirs = sorted(p for p in d.glob("*/") if p.is_dir()) if d.is_dir() else []
    hits = [p for p in dirs if OBS.stamp_complete(p)]
    best = hits[-1] if hits else None
    newer = [str(p) for p in dirs if not OBS.stamp_complete(p) and (best is None or p.name > best.name)]
    return best, newer


def meta_reasons(meta: dict | None, row: str) -> list[str]:
    """Per-stamp validity from run_meta (module docstring, VALIDITY). [] = valid."""
    if not isinstance(meta, dict):
        return ["run_meta.json missing or unreadable"]
    pm = meta.get("p2r_mode")
    if not isinstance(pm, dict):
        return ["no run_meta.p2r_mode (not a stand_height stamp)"]
    out = []
    if pm.get("variant") != EC.VARIANT:
        out.append(f"p2r_mode.variant {pm.get('variant')!r} != {EC.VARIANT!r}")
    mk = pm.get(EC.MASK_KEY)
    if type(mk) is not int or mk not in (0, 1):
        out.append(f"p2r_mode.{EC.MASK_KEY} {mk!r} is not an int 0 / 1")
    sp = pm.get("specialist")
    sha = sp.get("sha256") if isinstance(sp, dict) else None
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        out.append("p2r_mode.specialist.sha256 missing or malformed")
    blk = pm.get(EC.BLOCK_KEY)
    if not isinstance(blk, dict):
        out.append(f"no p2r_mode.{EC.BLOCK_KEY} block")
    else:
        if blk.get("control") not in EC.NET_CONTROLLERS:
            out.append(f"p2r_mode.{EC.BLOCK_KEY}.control {blk.get('control')!r} is not a network controller "
                       f"{EC.NET_CONTROLLERS} (an observe pass reads a stand network)")
        if blk.get("r_knob") != SN.RESTANCE_D3:
            out.append(f"p2r_mode.{EC.BLOCK_KEY}.r_knob {blk.get('r_knob')!r} != the D3 constants")
        want = "zero" if row in ZERO_ROWS else "manifest"
        if blk.get("command") != want:
            out.append(f"p2r_mode.{EC.BLOCK_KEY}.command {blk.get('command')!r} != {want!r} on a {row} row")
    rs = pm.get("raw_series") or []
    miss = [k for k in RAW_READ if k not in ("actions", "joint_pos") and k not in rs]
    if miss:
        out.append(f"p2r_mode.raw_series lacks {miss}")
    env = meta.get("env_vars")
    if not isinstance(env, dict):
        out.append("run_meta.env_vars missing")
    else:
        out += [f"env: {r}" for r in SN.env_reasons(env)]
    if OBS.cam_columns(meta) is None:
        out.append(f"run_meta.joint_names does not name six *{OBS.CAM_JOINT_SUFFIX} joints")
    dt = meta.get("dt")
    if not isinstance(dt, (int, float)) or not dt > 0:
        out.append(f"run_meta.dt {dt!r} is not > 0")
    return out


def arm_of(meta: dict | None) -> dict:
    pm = (meta or {}).get("p2r_mode") or {}
    blk = pm.get(EC.BLOCK_KEY) if isinstance(pm.get(EC.BLOCK_KEY), dict) else {}
    sp = pm.get("specialist") if isinstance(pm.get("specialist"), dict) else {}
    return {"specialist_sha256": sp.get("sha256"), "specialist_path": sp.get("path"),
            "stand_mask": pm.get(EC.MASK_KEY), "s1_checkpoint_sha256": pm.get("s1_checkpoint_sha256"),
            "stand_ckpt_kind": blk.get("stand_ckpt_kind"), "trained_stand_mask": blk.get("trained_stand_mask"),
            "control": blk.get("control")}


# ============================================================================ episodes -> the predecessor's Acc
def residual_episode(d: dict) -> tuple[dict, list[str]]:
    """One stand_height raw episode -> ``(the predecessor's episode dict in the residual frame, reasons)`` (pure).
    reasons = this wrapper's routing / execution self-checks (module docstring)."""
    stand = np.asarray(d[RAW_STAND], dtype=bool)
    a_net = np.asarray(d[EC.RAW_A_NET], dtype=np.float32)
    q = np.asarray(d[EC.RAW_Q], dtype=np.float32)
    tgt = np.asarray(d[EC.RAW_RH_TARGET], dtype=np.float32)
    act = np.asarray(d["actions"], dtype=np.float32)
    ex = np.asarray(d[EC.RAW_STAND_ACTION], dtype=np.float32)
    reasons = []
    if stand.any():
        pre = np.where(q != 0, a_net + q, a_net)
        bad = int((act[stand] != pre[stand]).any(axis=1).sum())
        if bad:
            reasons.append(f"the recorded actions differ from where(q != 0, a_net + q, a_net) on {bad} stand rows "
                           "(not the stand network's pre-clip record)")
        want = q.astype(np.float64) + np.clip(a_net.astype(np.float64), -1.0, 1.0)
        want[:, 0:6] = np.clip(want[:, 0:6], -1.0, 1.0)
        want[:, 6:] = np.clip(want[:, 6:], -4.8, 4.8)
        err = float(np.abs(ex[stand].astype(np.float64) - want[stand]).max())
        if err > EXEC_TOL:
            reasons.append(f"{EC.RAW_STAND_ACTION} != q + clip(a_net, +-1) on stand rows (max |diff| {err:.3g})")
    z = np.zeros_like(a_net)
    out = {RAW_STAND: stand,
           SN.RAW_STAND_ACTION: np.where(stand[:, None], a_net, z),
           SN.RAW_R_TARGET: np.where(stand[:, None], tgt - q, z),
           SN.RAW_R_BRANCH: np.asarray(d[EC.RAW_RH_BRANCH]),
           SN.RAW_S1_ACTION: np.asarray(d[EC.RAW_S1_ACTION], dtype=np.float32),
           # the predecessor's routing check compares these on stand rows: satisfied by construction here (this
           # wrapper's own check above compares the harness record with the network's output)
           "actions": np.where(stand[:, None], a_net, act),
           "joint_pos": np.asarray(d["joint_pos"])}
    return out, reasons


def read_stamp(stamp: Path, meta: dict, params):
    """Every raw episode of ``stamp`` into the predecessor's ``Acc`` (residual frame); ``(acc, reasons)``."""
    rst = OBS.rst()
    cams = OBS.cam_columns(meta)
    dt = float(meta["dt"])
    window = int(round(OBS.LAUNCH_WINDOW_S / dt))
    files = sorted((Path(stamp) / "raw").glob("episode_*.npz"))
    if not files:
        return None, ["no raw/episode_*.npz"]
    acc = OBS.Acc()
    reasons = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            miss = [k for k in RAW_READ if k not in z.files]
            if miss:
                return None, [f"{f.name} lacks raw key(s) {miss}"]
            d = {k: z[k] for k in RAW_READ}
        dd, why = residual_episode(d)
        reasons += [f"{f.name}: {w}" for w in why]
        acc.add_episode(dd, cams, window, params, rst.BRANCH_LIFT, rst.BRANCH_HOLD)
    return acc, reasons + OBS.check_reasons(acc)


# ===================================================================================== the read
def observe(head: str, rows: list[str], lr: Path) -> dict:
    params = OBS.r_params()
    keys = []
    for spec in rows:
        row, seed = parse_row(spec)
        k = f"{row}@{seed}"
        if k not in keys:
            keys.append(k)
    out = {"head": head, "lr": str(lr), "rows_requested": keys, "r_knob": SN.RESTANCE_D3,
           "r_params": params.to_dict(), "frame": "residual: applied = clip(sh_a_net, +-1); R = sh_rh_target - sh_q18",
           "rows": {}, "pending": [], "invalid": {}, "pooled": {}}
    stamps, metas = {}, {}
    for k in keys:
        row, seed = parse_row(k)
        st, newer = resolve(lr, head, row, seed)
        if st is None:
            msg = f"{k}: no complete stamp under {Path(lr) / head / ROW_DIRS[row] / f'seed{seed:03d}'}"
            if newer:
                msg += f" (in progress: {', '.join(Path(n).name for n in newer)})"
            out["rows"][k] = {"status": PENDING, "stamp": None, "in_progress": newer}
            out["pending"].append(msg)
            continue
        stamps[k] = st
        metas[k] = OBS._json(st / "run_meta.json")
        out["rows"][k] = {"status": None, "stamp": str(st), "in_progress": newer}
    arms = {k: arm_of(metas[k]) for k in stamps}
    cons = OBS.consistency_reasons(arms)
    out["arm"] = next(iter(arms.values())) if arms and not cons else (arms or None)
    accs, dts = {}, {}
    for k, st in stamps.items():
        why = meta_reasons(metas[k], parse_row(k)[0]) + cons
        acc = None
        if not why:
            acc, why2 = read_stamp(st, metas[k], params)
            why += why2
        if why:
            out["rows"][k]["status"] = INVALID
            out["invalid"][k] = why
            continue
        dts[k] = float(metas[k]["dt"])
        accs[k] = acc
        out["rows"][k]["status"] = READ
        out["rows"][k].update(OBS.summarize(acc, dts[k], params))
    fams = {f: [k for k in accs if parse_row(k)[0] in rws] for f, rws in FAMILIES.items()}
    fams["all"] = list(accs)
    for fam, ks in fams.items():
        if not ks:
            continue
        dt_set = {dts[k] for k in ks}
        if len(dt_set) != 1:
            out["pooled"][fam] = {"rows_included": ks, "error": f"rows differ in dt {sorted(dt_set)}"}
            continue
        blk = OBS.summarize(OBS.Acc.pool([accs[k] for k in ks]), dt_set.pop(), params)
        blk["rows_included"] = ks
        out["pooled"][fam] = blk
    out["status"] = "refused" if out["invalid"] else "partial" if out["pending"] else "complete"
    return out


def markdown(out: dict) -> str:
    """The predecessor's markdown with this campaign's frame note under the title."""
    text = OBS.markdown(out).replace("# Observe pass —", "# Stand-height observe pass —", 1)
    head, _, rest = text.partition("\n")
    return f"{head}\n\n{FRAME_NOTE}\n{rest}"


# ===================================================================================== CLI
def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", required=True, help="the stand_height head name (lr/<NAME>)")
    ap.add_argument("--rows", nargs="+", default=list(ROWS_DEFAULT),
                    help="rows to read, <walkstop|suite|hwalkstop|hraise>[__omni]@<es> (default: all four, es1 + es2)")
    ap.add_argument("--json", type=Path, required=True, help="write the JSON here (no default)")
    ap.add_argument("--md", type=Path, required=True, help="write the markdown here")
    ap.add_argument("--lr", type=Path, default=LR_DEFAULT, help="the phase-2 lr dir (default %(default)s)")
    a = ap.parse_args(argv)
    try:
        for r in a.rows:
            parse_row(r)
    except ValueError as exc:
        print(f"sh_observe_read: {exc}", file=sys.stderr)
        return RC_USAGE
    for p in (a.json, a.md):
        if not p.parent.is_dir():
            print(f"sh_observe_read: output directory {p.parent} does not exist", file=sys.stderr)
            return RC_USAGE
    out = observe(a.head, a.rows, a.lr)
    out["generated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    out["tool"] = str(Path(__file__).resolve())
    out["tool_sha256"] = _sha(Path(__file__).resolve())
    out["sn_observe_read_sha256"] = _sha(OBS_PATH)
    out["sh_eval_common_sha256"] = _sha(EC_PATH)
    out["common_sha256"] = _sha(OBS.COMMON_PATH)
    out["restance_module_sha256"] = _sha(SN.RESTANCE_MODULE)
    out = OBS._jsonable(out)
    a.json.write_text(json.dumps(out, indent=1, default=str))
    a.md.write_text(markdown(out))
    rc = RC_INVALID if out["invalid"] else RC_PENDING if out["pending"] else RC_OK
    print(f"sh_observe_read: {a.head} status {out['status']} -- "
          + ", ".join(f"{k} {r['status']}" for k, r in out["rows"].items()) + f"; wrote {a.json} and {a.md}")
    for p in out["pending"]:
        print(f"PENDING: {p}")
    for k, why in out["invalid"].items():
        print(f"INVALID {k}: {'; '.join(why)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
