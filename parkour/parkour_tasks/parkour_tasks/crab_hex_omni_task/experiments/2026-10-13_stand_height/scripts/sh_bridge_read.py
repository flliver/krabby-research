#!/usr/bin/env python3
"""Stand-height campaign, PLAN.md step 4 -- the identity-bridge, swap-live and smoke reads of ``stand_height`` stamps
(CPU only, read-only; never edits a phase-2 file, writes no bytecode).

    sh_bridge_read.py --head NAME [--refs REFS_S1_s18] [--rows ROW@ES,...] (--identity | --live | --smoke) [--json P]
    sh_bridge_read.py --sh-stamp DIR [--refs-stamp DIR] (--identity | --live | --smoke) [--json P]

Default rows: ``--identity`` / ``--live`` the zero-command identity rows (suite / walkstop / field_hard @1, REFS =
REFS_S1_s18's newest complete stamp of the same row); ``--smoke`` the height rows (hwalkstop / hraise @1, no REFS).

Per row, from the stamps alone:

* **meta** (what the launcher writes; ``sh_eval_common`` names): ``p2r_mode.variant == "stand_height"``; the
  controller / command / restore / inner mode vocabulary; S1's sha (== the stamp's and REFS's); network controllers:
  ``stand_mask`` an int 0 / 1, the stand checkpoint sha, the estimator identity, the checkpoint kind and its pairing
  (S1 only as the stand_net zero-command bridge; A-mask s17 with mask 1 and ``W[:, 15]`` zeroed; a trainer checkpoint's
  trained mask == the stamp's); the D3 knob; the CLIP: wrapper 1.0 -> realised 4.8, S1 / residual / cam clamps 1, rods
  4.8, every policy call's action swapped at env.step, executed |S1 rows| <= 1, |stand rods| <= 4.8, |cams| <= 1; the
  restore record (``controller`` = HeightController.record() + restore, == sh_core's constants and the run's dt);
  ``hold_heights`` one entry per hold (a manifest row: its ``height_row`` record == the manifest file NOW, no drift, and
  its labels == the stamp's; a zero command: 1.064 on "stand" holds, null elsewhere, no height_row); the D5 env
  (``sn_common.env_reasons``); h_meas valid on manifest rows.
* **raws** (every episode; who acted = ``p2r_stand_mode``, the latched slot = ``sh_stand_slot``):
  ``p2r_stand_mode == sh_stand_slot | sh_restoring``; off-stand series clean; R_h branch on
  every stand row; the RECORDED ``actions`` on stand rows == the controller's pre-clamp value (network
  ``where(q != 0, a_net + q, a_net)``, scripted its executed target); the EXECUTED stand action == sh_core's composition
  ``q + clamp(a_net, +-1)`` with rods +-4.8 / cams +-1, re-derived from ``sh_a_net`` and ``sh_q18`` (network controllers);
  the limited command in [h_min, h_max] moving <= 0.08 m/s; the restore lifecycle (engage -> complete / time out /
  merge, never twice open; restore rows == open restores; armed: engaged == leave-off-height; off: none); the per-step
  raw command == the row's per-hold map (``height_table`` of ``hold_heights``) or 1.064 (zero command).
* ``--identity`` (S1 as the stand checkpoint, mask 0, zero command, restore armed): the meta and raws above plus no
  restore, h == 1.064, q == 0, the stand action == clamp(S1) and the recorded stand rows == the raw network output; then
  ``p2_read.p2r_bitwise_diff(REFS, SH)`` -- EVERY raw key REFS has (``actions`` included), every episode -- empty ->
  BITWISE. On a miss, the stand-network campaign's rule: MISS_SMALL when ``max |actions - REFS actions|`` <= 1e-5 (then
  every G1-G5 statistic must match: the orchestrator's comparison), else MISS (STOP and ask).
* ``--live``: some stand step's executed stand action differs from clamp(S1); with a REFS stamp, the actions before each
  episode's first stand entry == REFS's (non-vacuous on walkstop).
* ``--smoke`` (the step-4 launcher smoke, manifest rows): the per-hold command reaches the limiter (every scheduled stand
  height arrives exactly inside its hold), ``q`` (``sh_q18`` == ``sh_core.q18(h_lim)`` on stand rows; 0 on the old path),
  and R_h (HOLD / WAIT / SEAT legs' hip target == q, LIFT / HOME == q + the D3 lift, re-derived on CPU); hip targets
  beyond +-1 on stand rows (q-path controllers, when a scheduled height needs one); restores engage and complete
  (armed) or never engage with leave-off-height events (off); timeouts reported; median h_meas at settled 1.064 stands
  reported (the flat-ground ~1.062 check is read by eye).

Exit: 0 every row PASS (identity: BITWISE); 3 an identity MISS_SMALL (no FAIL); 4 a FAIL / MISS; 2 a stamp missing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # the phase-2 scripts are imported read-only: never write their __pycache__

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
P2_SCRIPTS = HERE.parents[1] / "2026-10-07_omni_phase2" / "scripts"
if str(P2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(P2_SCRIPTS))
import p2_read as P  # noqa: E402  (read-only)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EC = _load(HERE / "sh_eval_common.py", "_sh_eval_common_bridge_read")
CORE = EC.CORE
SN = CORE.SN

REFS_NAME = "REFS_S1_s18"
IDENTITY_ROWS = tuple(f"{r}@1" for r in EC.IDENTITY_ROW_IDS)
HEIGHT_ROWS = ("hwalkstop__omni@1", "hraise__omni@1")
MISS_ATOL = 1e-5
Q_ATOL = 1e-6
PASS, FAIL, BITWISE, MISS_SMALL, MISS, MISSING = "PASS", "FAIL", "BITWISE", "MISS_SMALL", "MISS", "MISSING"


def _meta(stamp) -> dict:
    return json.loads((Path(stamp) / "run_meta.json").read_text())


def _episodes(stamp) -> dict:
    return {f.name: f for f in sorted((Path(stamp) / "raw").glob("episode_*.npz"))}


def _close(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# ===================================================================================== meta
def meta_reasons(sh_stamp, refs_stamp=None, *, identity: bool = False) -> list[str]:
    """The stand_height validity fields of one stamp (fail closed)."""
    out = []
    sm = _meta(sh_stamp)
    blk = sm.get("p2r_mode") or {}
    if blk.get("variant") != EC.VARIANT:
        out.append(f"p2r_mode.variant {blk.get('variant')!r} != {EC.VARIANT!r}")
    sh = blk.get(EC.BLOCK_KEY)
    if not isinstance(sh, dict) or not sh:
        return out + [f"p2r_mode.{EC.BLOCK_KEY} block missing"]
    ctl, cmd, rst = sh.get("control"), sh.get("command"), sh.get("restore")
    if ctl not in EC.CONTROLLERS or cmd not in EC.COMMANDS or rst not in EC.RESTORES:
        return out + [f"control / command / restore {ctl!r} / {cmd!r} / {rst!r} not in the vocabulary"]
    if sh.get("inner_mode") != EC.INNER_MODE[ctl]:
        out.append(f"inner_mode {sh.get('inner_mode')!r} != {EC.INNER_MODE[ctl]!r} for {ctl}")
    s1 = blk.get("s1_checkpoint_sha256")
    if not s1 or s1 != sm.get("checkpoint_sha256"):
        out.append(f"p2r_mode.s1_checkpoint_sha256 {str(s1)[:12]} != the stamp's checkpoint_sha256")
    if refs_stamp is not None and s1 != _meta(refs_stamp).get("checkpoint_sha256"):
        out.append(f"S1 {str(s1)[:12]} is not REFS's checkpoint")
    net = ctl in EC.NET_CONTROLLERS
    mask = blk.get(EC.MASK_KEY)
    sp = (blk.get("specialist") or {}).get("sha256") if isinstance(blk.get("specialist"), dict) else None
    kind = sh.get("stand_ckpt_kind")
    if net:
        if isinstance(mask, bool) or mask not in (0, 1) or sh.get("stand_mask") != mask:
            out.append(f"p2r_mode.{EC.MASK_KEY} {mask!r} is not an int 0 / 1 equal to the block's "
                       f"{sh.get('stand_mask')!r}")
        if not (isinstance(sp, str) and re.fullmatch(r"[0-9a-f]{64}", sp)):
            out.append(f"p2r_mode.specialist.sha256 {sp!r} is not a sha256")
        if sh.get("estimator_equals_s1") is not True:
            out.append(f"estimator_equals_s1 is {sh.get('estimator_equals_s1')!r}")
        if kind not in EC.CKPT_KINDS:
            out.append(f"stand_ckpt_kind {kind!r} missing / unknown")
        if kind == EC.CKPT_KIND_S1 and sp != s1:
            out.append(f"stand_ckpt_kind s1_identity but the stand checkpoint {str(sp)[:12]} is not S1 {str(s1)[:12]}")
        if kind == EC.CKPT_KIND_AMASK:
            w = sh.get("w15") or {}
            if sp != CORE.STAND_NET_SHA256:
                out.append("stand_ckpt_kind amask but the stand checkpoint is not the pinned A-mask s17 file")
            if w.get("w15_max_abs_after") != 0.0 or not (w.get("w15_norm_before") or 0) > 0:
                out.append(f"A-mask s17 W[:, 15] not zeroed (w15 record {w!r})")
        if kind == EC.CKPT_KIND_TRAINED:
            tm = sh.get("trained_stand_mask")
            if type(tm) is not int or tm != mask:
                out.append(f"trained_stand_mask {tm!r} != stand_mask {mask!r} (an arm evaluated under the other mask)")
        expect = EC.ckpt_kind(sp, s1)
        if expect is not None and kind != expect:
            out.append(f"stand_ckpt_kind {kind!r} but the hashes say {expect!r}")
        out += [f"checkpoint pairing: {r}" for r in EC.ckpt_kind_reasons(
            kind if kind != EC.CKPT_KIND_TRAINED else None, controller=ctl, command=cmd,
            stand_mask=mask if not isinstance(mask, bool) else None)]
    else:
        if blk.get("specialist") is not None or blk.get(EC.MASK_KEY) is not None:
            out.append(f"a scripted controller ({ctl}) with a stand checkpoint / mask in run_meta")
    if sh.get("r_knob") != SN.RESTANCE_D3:
        out.append(f"r_knob {sh.get('r_knob')!r} != the D3 string")
    clip = sh.get("clip") or {}
    for k, want in (("wrapper_clip_before", EC.S1_TRAIN_CLIP), ("wrapper_clip_realised", CORE.HARNESS_CLIP),
                    ("s1_clamp", CORE.S1_CLIP), ("residual_clamp", CORE.RESIDUAL_CLIP),
                    ("stand_rod_clamp", CORE.RAW_CLIP), ("stand_cam_clamp", CORE.CAM_CLIP)):
        if not _close(clip.get(k), want):
            out.append(f"clip.{k} {clip.get(k)!r} != {want}")
    calls, swaps = sh.get("calls"), sh.get("exec_swaps")
    if not calls or swaps != calls:
        out.append(f"{swaps!r} executed-action swaps for {calls!r} policy calls")
    st = sh.get("stats") or {}
    for k, lim in (("s1_exec_abs_max", CORE.S1_CLIP), ("stand_rod_exec_abs_max", CORE.RAW_CLIP),
                   ("stand_cam_exec_abs_max", CORE.CAM_CLIP)):
        v = st.get(k)
        if v is None or not float(v) <= lim + 1e-6:
            out.append(f"stats.{k} {v!r} exceeds {lim}")
    cr = sh.get("controller") or {}
    if cr.get("restore") != rst:
        out.append(f"controller.restore {cr.get('restore')!r} != {rst!r}")
    for k, want in (("rate_m_s", CORE.HEIGHT_RATE_M_S), ("h_min", CORE.H_MIN), ("h_max", CORE.H_MAX),
                    ("rod_tol_m", CORE.RESTORE_ROD_TOL_M)):
        if not _close(cr.get(k), want):
            out.append(f"controller.{k} {cr.get(k)!r} != sh_core's {want}")
    if not _close(cr.get("dt"), sm.get("dt")) or cr.get("settle_steps") != CORE.steps_for(
            CORE.RESTORE_SETTLE_S, float(sm.get("dt") or 0.02)):
        out.append(f"controller dt / settle_steps {cr.get('dt')!r} / {cr.get('settle_steps')!r} do not match the "
                   f"run's dt {sm.get('dt')!r}")
    if rst == "armed" and not _close(cr.get("engage_m"), CORE.RESTORE_ENGAGE_M):
        out.append(f"restore armed but engage_m {cr.get('engage_m')!r} != {CORE.RESTORE_ENGAGE_M}")
    if rst == "off" and cr.get("engage_m") is not None:
        out.append(f"restore off but engage_m {cr.get('engage_m')!r}")
    hrow, hh = sh.get("height_row"), sh.get("hold_heights")
    labels = list(sm.get("hold_labels") or [])
    if not isinstance(hh, list) or len(hh) != len(labels):
        out.append(f"hold_heights {hh!r} is not one entry per hold ({len(labels)} holds)")
    if cmd == "manifest":
        if not isinstance(hrow, dict):
            out.append("a manifest row without its height_row record")
        else:
            try:
                now = EC.height_row(hrow.get("manifest"), hrow.get("scenario"))
                if now["hold_heights"] != hrow.get("hold_heights") or now["labels"] != hrow.get("labels"):
                    out.append("the manifest's height row changed since the run (drift)")
            except (OSError, ValueError) as exc:
                out.append(f"the manifest height row no longer validates: {exc}")
            if labels != list(hrow.get("labels") or []):
                out.append("run_meta.hold_labels != the height row's labels")
            if hh != hrow.get("hold_heights"):
                out.append("hold_heights != the height row's")
        if not (sh.get("h_meas") or {}).get("valid"):
            out.append(f"h_meas invalid on a manifest row ({(sh.get('h_meas') or {}).get('reason')})")
    else:
        if hrow is not None:
            out.append("a zero-command stamp carries a height_row record")
        if isinstance(hh, list) and hh != EC.zero_hold_heights(labels):
            out.append(f"zero command but hold_heights {hh!r} (want 1.064 on 'stand' holds, null elsewhere)")
    if sm.get("env_vars") is None:
        out.append("run_meta.env_vars missing")
    else:
        out += [f"env: {r}" for r in SN.env_reasons(sm["env_vars"])]
    if identity:
        if ctl != "stand_net" or kind != EC.CKPT_KIND_S1 or cmd != "zero" or rst != "armed" or mask != 0:
            out.append(f"identity: want stand_net / s1_identity / zero / armed / mask 0, got {ctl} / {kind} / {cmd} / "
                       f"{rst} / {mask!r}")
        if (sh.get("events_alive") or {}).get("engaged") != 0:
            out.append(f"identity: restores engaged {(sh.get('events_alive') or {}).get('engaged')!r}")
        if st.get("stand_rows_differ_from_s1") != 0:
            out.append(f"identity: {st.get('stand_rows_differ_from_s1')!r} stand rows differ from clamp(S1)")
    return out


# ===================================================================================== raws
def _restore_lifecycle(restoring, engaged, completed, timed_out, merged):
    """``(fails, open_at_end, counts)`` for one episode's restore events."""
    fails, is_open = [], False
    counts = {"engaged": 0, "completed": 0, "timed_out": 0, "merged": 0}
    for t in range(len(restoring)):
        if engaged[t]:
            if is_open:
                fails.append(f"step {t}: a restore engaged while one was open")
            is_open = True
            counts["engaged"] += 1
        ends = [k for k, v in (("completed", completed), ("timed_out", timed_out), ("merged", merged)) if v[t]]
        if len(ends) > 1 or (ends and engaged[t]):
            fails.append(f"step {t}: conflicting restore events {ends + (['engaged'] if engaged[t] else [])}")
        if ends:
            if not is_open:
                fails.append(f"step {t}: restore {ends[0]} with none open")
            is_open = False
            counts[ends[0]] += 1
        if bool(restoring[t]) != is_open:
            fails.append(f"step {t}: sh_restoring {bool(restoring[t])} but the event lifecycle says {is_open}")
            is_open = bool(restoring[t])
    return fails, int(is_open), counts


def _stand_exec(a_net, q):
    """sh_core.stand_action re-derived in numpy float32 (bitwise for the same float32 inputs)."""
    import numpy as np  # noqa: PLC0415

    a = np.clip(a_net, np.float32(-CORE.RESIDUAL_CLIP), np.float32(CORE.RESIDUAL_CLIP))
    s = np.where(q != 0, a + q, a).astype(np.float32)
    s[..., CORE.CAM_ACT] = np.clip(s[..., CORE.CAM_ACT], np.float32(-CORE.CAM_CLIP), np.float32(CORE.CAM_CLIP))
    s[..., CORE.HIP_ACT.start:] = np.clip(s[..., CORE.HIP_ACT.start:], np.float32(-CORE.RAW_CLIP),
                                          np.float32(CORE.RAW_CLIP))
    return s


def raw_read(sh_stamp, *, identity: bool = False, live: bool = False, smoke: bool = False, refs_stamp=None) -> dict:
    """Every raw check of the module docstring over one stamp."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    sm = _meta(sh_stamp)
    sh = (sm.get("p2r_mode") or {}).get(EC.BLOCK_KEY) or {}
    ctl, cmd, rst = sh.get("control"), sh.get("command"), sh.get("restore")
    net = ctl in EC.NET_CONTROLLERS
    old = ctl in EC.OLD_PATH_CONTROLLERS
    hh = list(sh.get("hold_heights") or []) if cmd == "manifest" else None
    labels = list(sm.get("hold_labels") or [])
    lift = float(CORE.r_params().lift)
    out = {"episodes": 0, "stand_steps": 0, "restore_steps": 0, "stand_steps_differ_from_s1": 0,
           "pre_entry_episodes_checked": 0, "restores": {"engaged": 0, "completed": 0, "timed_out": 0, "merged": 0,
                                                         "open_at_end": 0},
           "leave_off_height": 0, "hip_beyond_1_steps": 0, "q_max_abs_diff": 0.0, "rh_hip_max_abs_diff": 0.0,
           "stand_holds": 0, "stand_holds_arrived": 0, "h_meas_median_at_1064": None, "fails": []}
    fails = out["fails"]
    ref_files = _episodes(refs_stamp) if refs_stamp else {}
    need = {"actions", EC.RAW_STAND_MODE, "segment_id", "dt", *EC.RAW_KEYS}
    hm_1064 = []
    for name, f in _episodes(sh_stamp).items():
        d = np.load(f)
        out["episodes"] += 1
        miss = sorted(need - set(d.files))
        if miss:
            fails.append(f"{name}: raw series missing {miss}")
            continue
        mode = np.asarray(d[EC.RAW_STAND_MODE]).astype(bool)                 # who acted (slot | restoring)
        slot, restoring = np.asarray(d[EC.RAW_STAND_SLOT]), np.asarray(d[EC.RAW_RESTORING])
        ev = {k: np.asarray(d[k]) for k in (EC.RAW_ENGAGED, EC.RAW_COMPLETED, EC.RAW_TIMED_OUT, EC.RAW_MERGED,
                                            EC.RAW_LEAVE_OFF)}
        acts = np.asarray(d["actions"])
        q, a_net, sa, s1a = (np.asarray(d[k]) for k in (EC.RAW_Q, EC.RAW_A_NET, EC.RAW_STAND_ACTION, EC.RAW_S1_ACTION))
        rh, rb = np.asarray(d[EC.RAW_RH_TARGET]), np.asarray(d[EC.RAW_RH_BRANCH])
        h_cmd, h_lim, h_meas = (np.asarray(d[k]) for k in (EC.RAW_H_CMD, EC.RAW_H_LIM, EC.RAW_H_MEAS))
        seg, dt = np.asarray(d["segment_id"]), float(np.asarray(d["dt"]))
        out["stand_steps"] += int(mode.sum())
        out["restore_steps"] += int(restoring.sum())
        if not np.array_equal(mode, slot | restoring):
            fails.append(f"{name}: p2r_stand_mode != sh_stand_slot | sh_restoring")
        if (restoring & slot).any():
            fails.append(f"{name}: a restore row on a stand-slot step")
        off = ~mode
        if any(np.asarray(v)[off].any() for v in (q, a_net, sa, s1a, rh)) or (rb[off] != -1).any():
            fails.append(f"{name}: a stand_height series is set on an off-stand step")
        if mode.any():
            if (rb[mode] < 0).any():
                fails.append(f"{name}: a stand step carries no R_h branch")
            want = np.where(q != 0, a_net + q, a_net).astype(acts.dtype) if net else sa.astype(acts.dtype)
            if not np.array_equal(acts[mode], want[mode]):
                fails.append(f"{name}: recorded stand-row actions are not the controller's pre-clamp value")
            if net and not np.array_equal(sa[mode], _stand_exec(a_net[mode], q[mode])):
                fails.append(f"{name}: the executed stand action is not q + clamp(a_net, +-1) (rods 4.8, cams 1)")
            out["stand_steps_differ_from_s1"] += int((sa[mode] != s1a[mode]).any(axis=-1).sum())
            out["hip_beyond_1_steps"] += int((np.abs(sa[mode][:, CORE.HIP_ACT]) > 1.0).any(axis=-1).sum())
        if (h_lim < CORE.H_MIN - 1e-12).any() or (h_lim > CORE.H_MAX + 1e-12).any():
            fails.append(f"{name}: the limited command leaves [h_min, h_max]")
        if h_lim.size > 1 and (np.abs(np.diff(h_lim)) > CORE.HEIGHT_RATE_M_S * dt + 1e-12).any():
            fails.append(f"{name}: the limited command moves faster than {CORE.HEIGHT_RATE_M_S} m/s")
        eng, comp, tout, merg, leave = (ev[k] for k in (EC.RAW_ENGAGED, EC.RAW_COMPLETED, EC.RAW_TIMED_OUT,
                                                        EC.RAW_MERGED, EC.RAW_LEAVE_OFF))
        lf, open_end, counts = _restore_lifecycle(restoring, eng, comp, tout, merg)
        fails += [f"{name}: {x}" for x in lf[:3]]
        for k, v in counts.items():
            out["restores"][k] += v
        out["restores"]["open_at_end"] += open_end
        out["leave_off_height"] += int(leave.sum())
        if (comp | tout).any() and (slot[comp | tout]).any():
            fails.append(f"{name}: a restore handed back on a stand-slot step")
        if merg.any() and not slot[merg].all():
            fails.append(f"{name}: a merge off the stand slot")
        if rst == "armed" and not np.array_equal(eng, leave):
            fails.append(f"{name}: restore armed but engaged != leave-off-height")
        if rst == "off" and (eng.any() or restoring.any()):
            fails.append(f"{name}: restore off but a restore ran")
        if hh is not None:
            want_h = EC.height_table(seg[:, None], hh)[:, 0]
            if not np.array_equal(h_cmd, want_h):
                fails.append(f"{name}: sh_h_cmd is not the manifest's per-hold map")
        elif not (h_cmd == CORE.H_SETTLED).all():
            fails.append(f"{name}: a zero-command stamp with a raw command != 1.064")
        if identity:
            if eng.any() or restoring.any() or not (h_lim == CORE.H_SETTLED).all() or q.any():
                fails.append(f"{name}: identity: a restore / a height / a posture appeared")
            if mode.any() and (not np.array_equal(sa[mode], s1a[mode]) or not np.array_equal(acts[mode], a_net[mode])):
                fails.append(f"{name}: identity: the stand rows are not S1's own action")
        first_on = int(np.argmax(mode)) if mode.any() else 0
        if first_on and name in ref_files:
            ra = np.asarray(np.load(ref_files[name])["actions"])
            n = min(first_on, acts.shape[0], ra.shape[0])
            out["pre_entry_episodes_checked"] += 1
            if n and not np.array_equal(acts[:n], ra[:n]):
                fails.append(f"{name}: pre-entry actions (steps 0:{n}) differ from REFS")
        if smoke and mode.any():
            hx = np.full(int(mode.sum()), CORE.H_SETTLED) if old else h_lim[mode]
            qx = CORE.q18(torch.from_numpy(np.ascontiguousarray(hx)), dtype=torch.float32).numpy()
            out["q_max_abs_diff"] = max(out["q_max_abs_diff"], float(np.abs(q[mode] - qx).max()))
            ql = CORE.q18(torch.from_numpy(np.ascontiguousarray(h_lim[mode])), dtype=torch.float32).numpy()
            hips_rh, hips_q, br = rh[mode][:, CORE.HIP_ACT], ql[:, CORE.HIP_ACT], rb[mode]
            lifting = (br == 2) | (br == 3)                       # BRANCH_LIFT, BRANCH_HOME
            want_hip = np.clip(np.where(lifting, hips_q + np.float32(lift), hips_q), -CORE.RAW_CLIP, CORE.RAW_CLIP)
            out["rh_hip_max_abs_diff"] = max(out["rh_hip_max_abs_diff"], float(np.abs(hips_rh - want_hip).max()))
            settled = mode & slot & ~restoring & (h_cmd == CORE.H_SETTLED) & (h_lim == CORE.H_SETTLED)
            hm_1064 += [float(x) for x in h_meas[settled] if math.isfinite(float(x))]
        if smoke and hh is not None:
            for i, hi in enumerate(hh):
                idx = np.nonzero(seg == i)[0]
                # a stand hold the episode lived through (steps recorded after it: the rows end on a walk)
                if hi is None or idx.size == 0 or idx[-1] + 1 >= seg.size:
                    continue
                out["stand_holds"] += 1
                if (h_lim[idx] == float(hi)).any():
                    out["stand_holds_arrived"] += 1
    if hm_1064:
        out["h_meas_median_at_1064"] = float(np.median(hm_1064))
    if smoke:
        if out["q_max_abs_diff"] > Q_ATOL:
            fails.append(f"smoke: sh_q18 differs from sh_core.q18(h) by {out['q_max_abs_diff']:.3g}")
        if out["rh_hip_max_abs_diff"] > Q_ATOL:
            fails.append(f"smoke: R_h's hip targets miss q (+ lift) by {out['rh_hip_max_abs_diff']:.3g}")
        if hh is not None:
            heights = [h for h in hh if h is not None]
            if out["stand_holds"] and out["stand_holds_arrived"] != out["stand_holds"]:
                fails.append(f"smoke: {out['stand_holds'] - out['stand_holds_arrived']} of {out['stand_holds']} "
                             "stand holds never reached their commanded height in the limiter")
            q_path = not old and ctl in ("stand_net", "r_h", "posture_only")
            needs = [h for h in heights if abs(float(CORE.q18(torch.tensor([float(h)]))[0, CORE.HIP_ACT.start])) > 1.0]
            if q_path and needs and not out["hip_beyond_1_steps"]:
                fails.append("smoke: no stand-row hip target beyond +-1 although the schedule needs one")
            off_heights = [h for h in heights if abs(float(h) - CORE.H_SETTLED) > CORE.RESTORE_ENGAGE_M]
            if rst == "armed" and off_heights and (not out["restores"]["engaged"] or not out["restores"]["completed"]):
                fails.append(f"smoke: restores engaged {out['restores']['engaged']} / completed "
                             f"{out['restores']['completed']} (armed, off-1.064 stands scheduled)")
            if rst == "off" and off_heights and not out["leave_off_height"]:
                fails.append("smoke: restore off but no hand-back at height recorded")
    if live and not identity and out["stand_steps"] and not out["stand_steps_differ_from_s1"]:
        fails.append("live: the stand controller's executed action equals clamp(S1) on every stand step")
    if not out["episodes"]:
        fails.append("no raw episodes")
    if not out["stand_steps"]:
        fails.append("no stand step")
    return out


# ===================================================================================== identity bridge
def max_action_diff(refs_stamp, sh_stamp) -> float:
    import numpy as np  # noqa: PLC0415

    ra, rb = _episodes(refs_stamp), _episodes(sh_stamp)
    if set(ra) - set(rb):
        return float("inf")
    worst = 0.0
    for name in sorted(ra):
        a, b = np.asarray(np.load(ra[name])["actions"]), np.asarray(np.load(rb[name])["actions"])
        if a.shape != b.shape:
            return float("inf")
        if a.size:
            worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
    return worst


def identity_read(refs_stamp, sh_stamp) -> dict:
    diff = P.p2r_bitwise_diff(str(refs_stamp), str(sh_stamp))
    if not diff:
        return {"verdict": BITWISE, "first_mismatches": [], "max_action_diff": 0.0}
    mad = max_action_diff(refs_stamp, sh_stamp)
    return {"verdict": MISS_SMALL if mad <= MISS_ATOL else MISS, "first_mismatches": diff[:10],
            "n_mismatches": len(diff), "max_action_diff": mad,
            "note": ("PASS-with-note only if every G1-G5 statistic is also identical (the stand-network campaign's "
                     "miss rule; compare with its sn_calib.py on OWN output files); else STOP and ask")
            if mad <= MISS_ATOL else "STOP and ask"}


def read_row(refs_stamp, sh_stamp, *, how: str) -> dict:
    if how not in ("identity", "live", "smoke"):
        raise ValueError(how)
    if not sh_stamp or (how == "identity" and not refs_stamp):
        return {"verdict": MISSING, "refs_stamp": refs_stamp, "sh_stamp": sh_stamp}
    identity = how == "identity"
    res = {"refs_stamp": (str(refs_stamp) if refs_stamp else None), "sh_stamp": str(sh_stamp), "how": how,
           "meta_fails": meta_reasons(sh_stamp, refs_stamp, identity=identity),
           "raws": raw_read(sh_stamp, identity=identity, live=how == "live", smoke=how == "smoke",
                            refs_stamp=refs_stamp)}
    bad = bool(res["meta_fails"] or res["raws"]["fails"])
    if identity:
        res["bridge"] = identity_read(refs_stamp, sh_stamp)
        v = res["bridge"]["verdict"]
        res["verdict"] = FAIL if bad or v == MISS else (MISS_SMALL if v == MISS_SMALL else PASS)
    else:
        res["verdict"] = FAIL if bad else PASS
    return res


def exit_code(rows: dict) -> int:
    vs = [r["verdict"] for r in rows.values()]
    if any(v == FAIL for v in vs):
        return 4
    if any(v == MISSING for v in vs):
        return 2
    if any(v == MISS_SMALL for v in vs):
        return 3
    return 0


def _row_spec(spec: str) -> tuple[str, int]:
    row, sep, es = spec.rpartition("@")
    if not sep or not row or not es.isdigit() or int(es) < 1:
        raise SystemExit(f"sh_bridge_read: row spec {spec!r} must be ROW@ES (e.g. walkstop__omni@1)")
    return row, int(es)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", default=None, help="the stand_height head's stamp name under $P2_OUT/lr")
    ap.add_argument("--refs", default=REFS_NAME)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--refs-stamp", default=None)
    ap.add_argument("--sh-stamp", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--identity", action="store_true", help="S1 as the stand checkpoint, mask 0, zero command")
    g.add_argument("--live", action="store_true", help="the stand controller's action reaches the stand rows")
    g.add_argument("--smoke", action="store_true", help="the step-4 launcher smoke on manifest (height) rows")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    how = "identity" if a.identity else ("live" if a.live else "smoke")
    if (a.head is None) == (a.sh_stamp is None):
        raise SystemExit("sh_bridge_read: give exactly one of --head NAME or --sh-stamp DIR")
    rows = {}
    if a.sh_stamp:
        rows["stamp"] = read_row(a.refs_stamp, a.sh_stamp, how=how)
    else:
        specs = a.rows.split(",") if a.rows else list(HEIGHT_ROWS if how == "smoke" else IDENTITY_ROWS)
        for spec in [s.strip() for s in specs if s.strip()]:
            row, es = _row_spec(spec)
            refs = P.newest_stamp(a.refs, row, es) if row in EC.IDENTITY_ROW_IDS else None
            rows[f"{row}@{es}"] = read_row(refs, P.newest_stamp(a.head, row, es), how=how)
    for k, r in rows.items():
        rw = r.get("raws") or {}
        br = r.get("bridge") or {}
        print(f"{k}: {r['verdict']}  bridge={br.get('verdict')} max|da|={br.get('max_action_diff')} "
              f"stand_steps={rw.get('stand_steps')} differ={rw.get('stand_steps_differ_from_s1')} "
              f"restores={rw.get('restores')} hip>1={rw.get('hip_beyond_1_steps')} "
              f"arrived={rw.get('stand_holds_arrived')}/{rw.get('stand_holds')} "
              f"h_meas@1.064={rw.get('h_meas_median_at_1064')} pre_entry_checked={rw.get('pre_entry_episodes_checked')}")
        for why in (r.get("meta_fails") or []) + (rw.get("fails") or [])[:10] + (br.get("first_mismatches") or [])[:5]:
            print(f"    - {why}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, sort_keys=True, default=str))
    rc = exit_code(rows)
    print(f"sh_bridge_read: exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
