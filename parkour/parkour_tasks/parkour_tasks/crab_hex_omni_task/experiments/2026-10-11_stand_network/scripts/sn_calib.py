#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 2 -- the gate-calibration read (CPU only, read-only).

Reads the pre-registered closed-loop gates G1-G5 (PLAN.md D6) on the existing reference stamps, before any training,
and checks that every (gate, reference) cell the plan lists "lands as listed". Every statistic is the phase-2
readers' own (``2026-10-07_omni_phase2/scripts/p2_read.py`` and ``p2s_stand.py``, imported read-only by path; never
edited, no bytecode written). Where PLAN.md D5 needs a different validity rule than p2_read's, the rule lives here and
every bypass is written into the output.

REFERENCES (stamp names under $P2_OUT/lr/<NAME>/<row>/seed00<k>/<stamp>/):
  REFS   REFS_S1_s18     S1 alone (STAND_OBS=0)
  HOLD   P2R_HOLD_s18    zero-action hold at the latched stand slot (p2r_mode hold, STAND_OBS=0)
  R      P2R_RSTg_s18    the scripted re-stance controller R at the stand slot (p2r_mode restance, STAND_OBS=clock):
                         suite / walkstop / field_turn; field_hard from SN_RSTg_s18 es1-4 (R re-run at the current
                         code by scripts/sn_calib_rows.sh) when all four are complete, else P2R_RSTg_s18 es1 (n = 100)
                         and the field_hard clauses read PARTIAL
  Z1     P2__Z1_s18      in-network statue (STAND_OBS=clock); its lr suite es1 is missing -- p2_read.z2_suite_stamp
                         resolves the record's driver stamp, recorded in the output
  Z2d    P2__Z2dR1_s18   masked distillation of R (STAND_OBS=clock, STAND_MASK=1)
  ST2M   P2R_ST2M_s18    the ST2 specialist at the stand slot (p2r_mode st2, STAND_OBS=0)
  C0R    C0R_s18         continuation control (attribution floor; every gate reported, none expected)

GATES (PLAN.md D6; es1 tune / es2 held out; field_hard pooled es1-4 n = 400; field_turn es1 only):
  G1 STILL     p2s_stand.still_clauses on the suite stand, all six clauses, es1 AND es2.
  G2 HOMED     p2_read.z2_stand_gates (relative=True, probe = P2R_RSTg_s18): walkstop homed >= 0.90, settled >= 0.90,
               median settle <= probe's same-seed median + 1 s, post-settle steps / taps / KS -- its non-STILL,
               non-falls clauses, per seed.
  G3 FALLS     walkstop stand falls (raw count per env seed) <= max(1, Z1's count) [CHARTER 2026-10-12]; suite stand falls (p2s head_summary) == 0;
               walkstop exit-handover falls within 2 s (p2_read.p2r_hold_reads handover_falls_by_entry_phase, summed
               over phases) <= max(1, R's), a RAW COUNT per env seed (200 exits per seed). p2r_hold_reads returns
               n_episodes 0 (not None) when no episode carries p2r_stand_mode: read here as None -> INVALID for a
               stamp that has run_meta.p2r_mode, n/a for a plain single-network eval (REFS, Z1, Z2d, C0R).
               Denominator: the head's raw episode count must equal R's same-seed count and no episode may exit
               stand mode more often than the schedule has stand -> move boundaries (else INVALID); an exit the
               schedule has but the episode never made fails the clause closed unless that episode fell in stand
               mode (a stand fall is the stand-falls clause's, inside its pre-registered max(1, Z1's) floor).
  G4 WALLS     field_hard pooled completion >= REFS - max(0.05, 2 SE) (p2s_verdict's OBSTACLE LOSS line,
               p2_read.se_need); field_turn completion >= REFS - max(0.05, band), band = p2s_cost's
               max(binom_se(REFS, n), COMP_BAND_FLOOR); field stand-fall rate <= HOLD + 2 SE(pooled diff)
               (p2_read.z2_field_vs_hold) on field_turn es1 and on field_hard pooled es1-4.
  G5 HANDOVER  walkstop median onset <= REFS + max(0.3 s, band) (p2s_cost: walkstop_read / p2s_onset /
               p2s_stand_bands), per env seed AND pooled es1+es2; walkstop onset STALLS (a stand -> move hold the
               episode reached whose onset is censored, p2s_stand.episode_onset_latency None) per env seed <= REFS's
               same-seed count + 1 (the median drops censored onsets, so a stall is gated here; head and REFS must
               read the same episode count); stand body height within 0.03 m of REFS (p2_read.z2_height_vs_refs:
               suite + walkstop, per seed).
  Reported only: six-down, hip-yaw mean / max, settle median, KS.

VERDICTS per (gate, reference): INVALID (a stamp the gate uses -- or a line it reads -- breaks its validity rule, or a
line source's complete stamp has no readable statistic) > FAIL (an evaluated clause fails; decisive even when another
clause is PENDING; a head's complete stamp with no readable statistic fails closed) > PENDING (a stamp is missing) >
PARTIAL (R's field_hard from P2R_RSTg_s18 es1 only; the provisional result is shown) > PASS.
"Lands as listed": yes / NO (the opposite verdict, INVALID, or a known-bad that fails without its named clause) /
pending (PENDING or PARTIAL) / reported (no expectation).

    sn_calib.py [--json PATH] [--md PATH] [--jobs N] [--head NAME ...]   read, write calib.json + calib.md;
                                                                           exit 0 all land / 2 pending only / 4 a
                                                                           mismatch (step 2: STOP, notify, ask)
    sn_calib.py --selftest                                                 synthetic checks of the verdict logic

``--head NAME`` reads a stand-network head (none exists at step 2) under the D5 rule: every stamp STAND_OBS=0 and
run_meta.p2r_mode.variant == stand_net driving REFS's S1, and every stamp of the head carrying the SAME stand
checkpoint sha (``p2r_mode.specialist.sha256``) and the SAME mask flag (``p2r_mode.stand_mask``, 0 / 1) -- the field
names the step-3 launcher must write (constants STAND_NET_CKPT_KEY / STAND_NET_MASK_KEY) -- fail closed.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # the phase-2 scripts are imported read-only: never write their __pycache__

import argparse  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
P2_DIR = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
P2_SCRIPTS = P2_DIR / "scripts"
if str(P2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(P2_SCRIPTS))
import p2_read as P  # noqa: E402  (read-only import; the ONE reader set)

S = P.PS()                          # p2s_stand, the same module instance p2_read uses
SN_LOG = R / "parkour/outputs/omni_plan/stand_network/sn_calib.log"

WS, FH, FT, SUITE = P.WS, P.FH, P.FT, P.SUITE
SEEDS = P.Z2_SEEDS                  # (1, 2): es1 tune, es2 held out
FH_SEEDS = P.P2B_FH_SEEDS           # (1, 2, 3, 4)
FH_POOL = f"{FH}@1-4"               # the pooled field_hard pseudo-row fed to z2_field_vs_hold
FT1 = f"{FT}@1"
EPS = P.EPS

NAMES = {"REFS": "REFS_S1_s18", "HOLD": "P2R_HOLD_s18", "R": "P2R_RSTg_s18", "R_SN": "SN_RSTg_s18",
         "Z1": "P2__Z1_s18", "Z2d": "P2__Z2dR1_s18", "ST2M": "P2R_ST2M_s18", "C0R": "C0R_s18"}
ROLES = ("REFS", "HOLD", "R", "Z1", "Z2d", "ST2M", "C0R")
#: validity rule per role: obs0 (D5 obs parity, STAND_OBS=0), tier1 (obs0 + p2r_mode_fails, step-1 heads), rst
#: (p2_read.rst_invalid_reasons, clock kept), z (p2_read's Z2 head rule), stand_net (D5, campaign heads)
KIND = {"REFS": "obs0", "HOLD": "tier1", "R": "rst", "R_SN": "rst", "Z1": "z", "Z2d": "z", "ST2M": "tier1",
        "C0R": "obs0"}
TIER1_TAG = {"HOLD": "HOLD", "ST2M": "ST2M"}
GATES = ("G1", "G2", "G3", "G4", "G5")
GATE_TITLE = {"G1": "STILL", "G2": "HOMED", "G3": "FALLS", "G4": "WALLS", "G5": "HANDOVER"}
PASS, FAIL, PENDING, INVALID, PARTIAL = "PASS", "FAIL", "PENDING", "INVALID", "PARTIAL"
#: PLAN.md D6 "known-good must pass" / "known-bad must fail" (step 2: every one must land as listed)
EXPECT = {("G1", "R"): PASS, ("G1", "HOLD"): PASS, ("G1", "REFS"): FAIL, ("G1", "ST2M"): FAIL,
          ("G2", "R"): PASS, ("G2", "HOLD"): FAIL, ("G2", "Z1"): FAIL, ("G2", "Z2d"): FAIL, ("G2", "REFS"): FAIL,
          ("G3", "R"): PASS, ("G3", "HOLD"): PASS,
          ("G4", "HOLD"): PASS, ("G4", "R"): PASS,
          ("G5", "HOLD"): PASS, ("G5", "R"): PASS}
#: a known-bad's parenthetical in D6: the gate must fail ON this clause (at every seed read)
NAMED_CLAUSE = {("G1", "REFS"): "steps_per_s", ("G1", "ST2M"): "taps_per_s"}
STAND_OBS_VAR = "KRABBY_OMNI_STAND_OBS"
STAND_NET_VARIANT = "stand_net"
G3_EXIT_FLOOR = 1                   # exit-handover falls per env seed <= max(1, R's)
G3_STAND_FALLS_FLOOR = 1  # CHARTER 2026-10-12: walkstop stand falls, raw count per env seed <= max(1, Z1 count)
G5_STALL_FLOOR = 1                  # walkstop onset stalls per env seed <= REFS's same-seed count + 1
#: the step-3 stand_net launcher's run_meta contract (PLAN step 3: "run_meta block with ckpt sha + mask"): the stand
#: checkpoint's sha256 in the p2r launcher's second-network record (as st2's ``specialist``) and the --stand-mask flag
STAND_NET_CKPT_KEY = ("specialist", "sha256")
STAND_NET_MASK_KEY = "stand_mask"


# ================================================================================================ D5 validity (pure)
def _env_val(env: dict | None, k: str) -> str | None:
    v = (env or {}).get(k)
    return None if v is None else str(v).strip()


def obs0_reasons(name: str, k: str, env: dict | None) -> list[str]:
    """PLAN.md D5 obs parity for a stamp that must run col 15 = 0 (S1 / REFS / HOLD and every stand-net head):
    ``KRABBY_OMNI_STAND_OBS`` exactly "0" (absent fails closed), the held proxy off, the env-level stand mask off
    (D1: the stand net's mask is actor-side; S1 never sees masked frames)."""
    out = []
    v = _env_val(env, STAND_OBS_VAR)
    if v != "0":
        out.append(f"{name} {k}: {STAND_OBS_VAR}={v!r}, want '0' (D5 obs parity: col 15 = 0)")
    if (_env_val(env, P.STAND_PROXY_VAR) or "").lower() not in P.STAND_PROXY_OFF:
        out.append(f"{name} {k}: {P.STAND_PROXY_VAR}={_env_val(env, P.STAND_PROXY_VAR)!r}, want it off")
    if (_env_val(env, P.STAND_MASK_VAR) or "").lower() not in P.STAND_MASK_OFF:
        out.append(f"{name} {k}: {P.STAND_MASK_VAR}={_env_val(env, P.STAND_MASK_VAR)!r}, want it off (D1: the "
                   "stand net masks actor-side; the env-level mask is not used)")
    return out


def stand_net_arm(mode: dict | None) -> tuple[str | None, int | None]:
    """``(stand checkpoint sha256, mask flag)`` a stand_net stamp's ``run_meta.p2r_mode`` records under the step-3
    launcher contract (:data:`STAND_NET_CKPT_KEY`, :data:`STAND_NET_MASK_KEY`); None for a missing / malformed field
    (a sha that is not 64 lowercase hex, a mask that is not 0 / 1 -- a bool is malformed)."""
    m = mode or {}
    sp = m.get(STAND_NET_CKPT_KEY[0])
    sha = sp.get(STAND_NET_CKPT_KEY[1]) if isinstance(sp, dict) else None
    sha = sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None
    raw = m.get(STAND_NET_MASK_KEY)
    mask = (int(str(raw).strip()) if raw is not None and not isinstance(raw, bool) and str(raw).strip() in ("0", "1")
            else None)
    return sha, mask


def stand_net_reasons(name: str, k: str, env: dict | None, mode: dict | None, refs_sha: str | None) -> list[str]:
    """PLAN.md D5 for a campaign head (replaces p2_read's Z2 "STAND_OBS=clock or INVALID"): :func:`obs0_reasons` +
    ``run_meta.p2r_mode.variant == stand_net`` + the mode wrapper drives REFS's own S1 checkpoint + the stamp names
    its stand checkpoint sha and mask flag (:func:`stand_net_arm`). Fail closed. One stamp; the across-stamp rule is
    :func:`stand_net_consistency`."""
    out = obs0_reasons(name, k, env)
    if not mode:
        return out + [f"{name} {k}: no run_meta.p2r_mode (want variant {STAND_NET_VARIANT!r})"]
    if mode.get("variant") != STAND_NET_VARIANT:
        out.append(f"{name} {k}: p2r_mode.variant {mode.get('variant')!r} != {STAND_NET_VARIANT!r}")
    ck = mode.get("s1_checkpoint_sha256")
    if not refs_sha or ck != refs_sha:
        out.append(f"{name} {k}: p2r_mode.s1_checkpoint_sha256 {str(ck)[:12]} != REFS {str(refs_sha)[:12]}")
    sha, mask = stand_net_arm(mode)
    if sha is None:
        out.append(f"{name} {k}: p2r_mode.{'.'.join(STAND_NET_CKPT_KEY)} missing or malformed (the stand checkpoint "
                   "sha; fail closed)")
    if mask is None:
        out.append(f"{name} {k}: p2r_mode.{STAND_NET_MASK_KEY} {mode.get(STAND_NET_MASK_KEY)!r} missing or not 0 / 1 "
                   "(the --stand-mask flag; fail closed)")
    return out


def stand_net_consistency(name: str, modes: dict) -> dict:
    """PLAN.md D5 across ONE head's stamps (pure): every stamp must carry the same stand checkpoint sha and the same
    mask flag (A-mask and A-full share the variant and S1's sha, so a NAME mixing both arms -- or a re-trained
    checkpoint -- would otherwise read valid). ``modes`` = {stamp key: run_meta.p2r_mode}; returns {key: [reasons]},
    every stamp of a mixed head INVALID. A missing field is :func:`stand_net_reasons`' (per stamp)."""
    arms = {k: stand_net_arm(m) for k, m in modes.items()}
    shas = sorted({a[0] for a in arms.values() if a[0]})
    masks = sorted({a[1] for a in arms.values() if a[1] is not None})
    out = {k: [] for k in modes}
    if len(shas) > 1 or len(masks) > 1:
        why = (f"{name}: the head's stamps mix stand checkpoints {[s[:12] for s in shas]} / masks {masks} "
               "(one arm, one checkpoint per head)")
        for k in modes:
            out[k].append(f"{why} [{k}: {str(arms[k][0])[:12]}, mask {arms[k][1]}]")
    return out


# ================================================================================================ IO: one reference
def _sub_point(pt: dict, k: str) -> dict:
    """The one-stamp view of a light point the p2_read validity functions take (so every reason maps to a key)."""
    return {"name": pt["name"], "stamps": {k: pt["stamps"][k]}, "valid": {k: (pt.get("valid") or {}).get(k)},
            "ws_env": pt.get("ws_env") or {}}


def valid_only_point(name: str) -> dict:
    """A point with stamps + stamp_validity only (json reads): the REFS side of invalid_reasons / p2r_mode_fails."""
    st = {}
    for k in P.Z2_LIGHT_STAMPS:
        row, s = k.rsplit("@", 1)
        st[k] = P.z2_suite_stamp(name) if k == "suite@1" else P.newest_stamp(name, SUITE if row == "suite" else row,
                                                                            int(s))
    return {"name": name, "stamps": st, "valid": {k: P.stamp_validity(v) for k, v in st.items()}}


def stamp_reasons(pt: dict, kind: str, refs_v: dict, tag: str | None = None) -> tuple[dict, dict]:
    """``(reasons by key, bypassed p2_read reasons by key)``. rst / z keep p2_read's own rules; obs0 / tier1 /
    stand_net apply D5 obs parity in place of p2_read's Z2 env rule, whose reasons are recorded as bypassed."""
    refs_sha = ((refs_v.get("valid") or {}).get("suite@1") or {}).get("ckpt_sha")
    reasons, bypass = {}, {}
    for k, stamp in (pt.get("stamps") or {}).items():
        if not stamp:
            continue
        sub = _sub_point(pt, k)
        env = P._stamp_env(stamp)
        if kind == "rst":
            r = P.rst_invalid_reasons(sub, refs_v, raw_bit=True)
        elif kind == "z":
            r = (P.invalid_reasons(sub, refs_v, (k,)) + P.walkstop_flat_fails(sub) + P.z_clock_invalid_reasons(sub)
                 + P.z2_invalid_reasons(sub))
        elif kind in ("obs0", "tier1"):
            r = P.invalid_reasons(sub, refs_v, (k,)) + P.walkstop_flat_fails(sub) + obs0_reasons(pt["name"], k, env)
            if kind == "tier1":
                r += P.p2r_mode_fails(sub, tag, refs_sha, stamps=(k,))
            bypass[k] = P.z2_env_reasons(sub)
        elif kind == "stand_net":
            r = (P.invalid_reasons(sub, refs_v, (k,)) + P.walkstop_flat_fails(sub)
                 + stand_net_reasons(pt["name"], k, env, P.p2r_mode_of(stamp), refs_sha))
            bypass[k] = P.z2_env_reasons(sub)
        else:
            raise ValueError(f"unknown validity kind {kind!r}")
        reasons[k] = r
    if kind == "stand_net":
        cons = stand_net_consistency(pt["name"], {k: P.p2r_mode_of(v) for k, v in (pt.get("stamps") or {}).items()
                                                  if v})
        for k, r in cons.items():
            reasons[k] = reasons.get(k, []) + r
    return reasons, bypass


def design_exits_per_episode(labels) -> int:
    """Stand -> non-stand hold boundaries in a schedule (walkstop fwd, stand, lat_l, stand, back, stand: 2) -- the
    stand-mode exits one full episode makes under the D5 switch contract (exit on the step the slot changes)."""
    labels = list(labels or [])
    return sum(1 for i in range(len(labels) - 1) if labels[i] == "stand" and labels[i + 1] != "stand")


def exit_audit(stamp: str) -> dict | None:
    """The exit clause's denominator, per episode, from the raws: stand-mode exits (the p2r_hold_reads definition: a
    step whose p2r_stand_mode is 0 after a 1) against the schedule's :func:`design_exits_per_episode`. ``extra`` =
    exits beyond the design (a switch-contract breach); a missing exit is ``missing_excused`` when its episode fell
    inside a stand-mode segment (p2r_hold_reads' falls_in_stand_mode definition, crab_failure on a stand-mode step: the
    stand-falls clause's business) and ``missing_unexcused`` otherwise. None if a raw lacks p2r_stand_mode."""
    import numpy as np  # noqa: PLC0415
    labels = (P._json(Path(stamp) / "run_meta.json") or {}).get("hold_labels") or []
    per = design_exits_per_episode(labels)
    out = {"episodes": 0, "design_exits_per_episode": per, "exits": 0, "extra": 0, "missing_excused": 0,
           "missing_unexcused": 0, "episodes_with_stand_mode_fall": 0}
    for f in P._p2r_dir_episodes(stamp).values():
        with np.load(f) as d:
            if "p2r_stand_mode" not in d.files:
                return None
            m = np.asarray(d["p2r_stand_mode"]).astype(bool)
            cf = np.asarray(d["crab_failure"]).astype(bool) if "crab_failure" in d.files else None
        k = int(((~m[1:]) & m[:-1]).sum()) if m.size > 1 else 0
        fell = bool(cf is not None and m.any() and cf[m].any())
        out["episodes"] += 1
        out["exits"] += k
        out["episodes_with_stand_mode_fall"] += int(fell)
        if k > per:
            out["extra"] += k - per
        elif k < per:
            out["missing_excused" if fell else "missing_unexcused"] += per - k
    out["design_exits"] = out["episodes"] * per
    return out


def exit_read(stamp: str | None) -> dict:
    """G3's exit clause input from :func:`p2_read.p2r_hold_reads`, with its missing-raw case made explicit:
    ``status`` missing (no stamp) / no_mode (no run_meta.p2r_mode: a plain single-network eval -> n/a) / no_raw
    (p2r_mode but no episode carries p2r_stand_mode -> None -> INVALID) / partial_raw (some episodes lack it ->
    INVALID) / ok (``count`` = handover_falls_by_entry_phase summed over phases)."""
    if not stamp:
        return {"status": "missing", "count": None}
    mode = P.p2r_mode_of(stamp)
    n_raw = len(P._p2r_dir_episodes(stamp))
    hr = P.p2r_hold_reads(stamp)
    n_ep = int((hr or {}).get("n_episodes") or 0)
    base = {"stamp": stamp, "has_p2r_mode": bool(mode), "n_raw_episodes": n_raw, "n_mode_episodes": n_ep,
            "hold_reads": hr}
    if n_ep == 0:
        return dict(base, status="no_raw" if mode else "no_mode", count=None)
    if n_ep < n_raw:
        return dict(base, status="partial_raw", count=None)
    by = {str(k): int(v) for k, v in (hr.get("handover_falls_by_entry_phase") or {}).items()}
    au = exit_audit(stamp)
    if au is None:
        return dict(base, status="partial_raw", count=None)
    return dict(base, status="ok", count=sum(by.values()), by_phase=by, exits=au["exits"], audit=au)


@functools.lru_cache(maxsize=32)
def onset_split(stamp: str | None) -> dict | None:
    """The walkstop onsets split by whether the episode reached the move hold: ``stalled`` = a stand -> move pair
    (p2s_stand.stand_onset_pairs) whose move segment is in the raws but whose onset is censored
    (p2s_stand.episode_onset_latency None: S1 never got going inside the hold), ``unreached`` = the episode ended
    before that move hold (a fall: G3's business). walkstop_read's ``onset_censored`` = stalled + unreached; its
    median drops both. None if the stamp has no schedule or no raws."""
    import numpy as np  # noqa: PLC0415
    if not stamp:
        return None
    meta = P._json(Path(stamp) / "run_meta.json") or {}
    labels, speeds, dt = meta.get("hold_labels") or [], meta.get("hold_speeds") or [], float(meta.get("dt", 0.02))
    pairs = S.stand_onset_pairs(labels, speeds) if speeds else []
    out = {"stamp": stamp, "n_episodes": 0, "pairs_per_episode": len(pairs), "reached": 0, "stalled": 0,
           "unreached": 0, "stalled_by_pair": {lab: 0 for _s, _m, lab in pairs}}
    for _m, _l, _e, d in S._raw_episodes(stamp, lambda e: True):
        out["n_episodes"] += 1
        seg = np.asarray(d["segment_id"])
        for _s, mi, lab in pairs:
            if not (seg == mi).any():
                out["unreached"] += 1
                continue
            out["reached"] += 1
            if S.episode_onset_latency(d, dt, mi) is None:
                out["stalled"] += 1
                out["stalled_by_pair"][lab] += 1
    return out if out["n_episodes"] and pairs else None


def mode_brief(mode: dict | None) -> dict | None:
    if not mode:
        return None
    rst = mode.get("restance") or {}
    sha, mask = stand_net_arm(mode)
    return {"variant": mode.get("variant"), "s1_checkpoint_sha256": mode.get("s1_checkpoint_sha256"),
            "launcher_sha256": mode.get("launcher_sha256"), "harness_sha256": mode.get("harness_sha256"),
            "restance_module_sha256": rst.get("module_sha256"), "restance_knob": rst.get("knob"),
            "enter_rule": mode.get("enter_rule"), "specialist_sha256": sha, "stand_mask": mask,
            "stand_bank_rows": mode.get("stand_bank_rows"), "stand_bank_seen": mode.get("stand_bank_seen")}


def read_reference(role: str, name: str, kind: str) -> dict:
    """Every read one reference feeds (IO). Stamps resolve exactly as p2_read's Z2 reads resolve them
    (:func:`p2_read.z2_light_point`); field_hard es2-4 stand summaries, walkstop_read and the exit reads are added."""
    t0 = time.time()
    pt = P.z2_light_point(name)
    st = dict(pt["stamps"])
    notes = []
    lr_suite = P.newest_stamp(name, SUITE, 1)
    if st.get("suite@1") and st["suite@1"] != lr_suite:
        notes.append(f"suite es1: {'lr/' + name + '/' + SUITE + '/seed001 has no complete stamp' if not lr_suite else 'lr row ' + lr_suite + ' not used'}"
                     f"; p2_read.z2_suite_stamp resolved the arm record's driver_stamp {st['suite@1']}")
    field = {k: dict(v) for k, v in (pt.get("field") or {}).items()}
    for s in FH_SEEDS:
        k = f"{FH}@{s}"
        if st.get(k) and not (field.get(k) or {}).get("stand"):
            field.setdefault(k, {})["stand"] = P.p2s_summary(st[k])
    ws = {s: P.walkstop_read(st.get(f"{WS}@{s}")) for s in SEEDS}
    ex = {s: exit_read(st.get(f"{WS}@{s}")) for s in SEEDS}
    osp = {s: onset_split(st.get(f"{WS}@{s}")) for s in SEEDS}
    refs_v = valid_only_point(NAMES["REFS"])
    reasons, bypass = stamp_reasons(pt, kind, refs_v, TIER1_TAG.get(role))
    return {"role": role, "name": name, "kind": kind, "stamps": st, "notes": notes, "missing": not any(st.values()),
            "z2": pt["z2"], "still": pt["still"], "field": field, "walkstop": ws, "exit": ex, "onset_split": osp,
            "p2r_mode": {k: mode_brief(P.p2r_mode_of(v)) for k, v in st.items() if v},
            "code_state": {k: (pt.get("valid") or {}).get(k, {}).get("sidecar") if (pt.get("valid") or {}).get(k)
                           else None for k in st if st[k]},
            "validity": reasons, "bypassed": bypass, "fh_seeds": list(FH_SEEDS), "fh_source": None,
            "stand_net_arm": ({k: list(stand_net_arm(P.p2r_mode_of(v))) for k, v in st.items() if v}
                              if kind == "stand_net" else None),
            "read_s": round(time.time() - t0, 1)}


def identity_check() -> dict:
    """SN_RSTg_s18 field_hard es1 vs P2R_RSTg_s18 field_hard es1, every raw key bitwise (p2_read.p2r_bitwise_diff;
    P2R_RSTg is the reference side). PENDING when the SN stamp is not complete."""
    import numpy as np  # noqa: PLC0415
    a, b = P.newest_stamp(NAMES["R"], FH, 1), P.newest_stamp(NAMES["R_SN"], FH, 1)
    out = {"reference": a, "rerun": b, "modes": {"P2R_RSTg_s18": mode_brief(P.p2r_mode_of(a)),
                                                  "SN_RSTg_s18": mode_brief(P.p2r_mode_of(b))},
           "code_state": {"P2R_RSTg_s18": (P.stamp_validity(a) or {}).get("sidecar"),
                          "SN_RSTg_s18": (P.stamp_validity(b) or {}).get("sidecar")}}
    if not a or not b:
        return dict(out, verdict=PENDING, why=[f"stamp missing (P2R_RSTg_s18 {bool(a)}, SN_RSTg_s18 {bool(b)})"])
    ea, eb = P._p2r_dir_episodes(a), P._p2r_dir_episodes(b)
    diff = P.p2r_bitwise_diff(a, b)
    extra = []
    if ea and eb:
        k0 = sorted(set(ea) & set(eb))
        if k0:
            with np.load(ea[k0[0]]) as da, np.load(eb[k0[0]]) as db:
                extra = sorted(set(db.files) - set(da.files))
    return dict(out, **identity_verdict(diff, len(ea), len(eb), extra), bank=bank_compare(a, b))


BANK_KEYS = ("obs", "target", "branch", "proxy", "hold_label", "category")
LAUNCHER = P2_SCRIPTS / "p2r_mode_eval_crab_hex_gait.py"


def _logged_bank_max(stamp: str) -> list[int]:
    """The ``bank_max=`` the launcher printed for this stamp's row (lr/<NAME>.<row>.seed00k.log; every distinct
    value, the log may hold reruns)."""
    p = Path(stamp)
    log = p.parents[3] / f"{p.parents[2].name}.{p.parents[1].name}.{p.parent.name}.log"
    try:
        return sorted({int(x) for x in re.findall(r"\[P2R\] variant=\S+ .*?bank_max=(\d+)", log.read_text())})
    except OSError:
        return []


def bank_compare(a: str, b: str) -> dict:
    """The stand bank beside the identity check (``<stamp>/p2r_stand_bank.npz``, NOT a raw key, so not part of the
    bitwise verdict): rows kept / rows seen (run_meta.p2r_mode), the cap each run logged, and whether every row of the
    re-run's bank is a row of the reference's by (episode, t) with ``BANK_KEYS`` bitwise equal. Reported only."""
    import numpy as np  # noqa: PLC0415
    ma, mb = P.p2r_mode_of(a) or {}, P.p2r_mode_of(b) or {}
    m = re.search(r"^DEFAULT_BANK_MAX = (\d+)", LAUNCHER.read_text(), re.M) if LAUNCHER.is_file() else None
    out = {side: {"rows": md.get("stand_bank_rows"), "seen": md.get("stand_bank_seen"),
                  "bank_max_logged": _logged_bank_max(st), "file": md.get("stand_bank")}
           for side, md, st in (("reference", ma, a), ("rerun", mb, b))}
    out["launcher_default_bank_max"] = int(m.group(1)) if m else None
    fa, fb = (Path(a) / str(ma.get("stand_bank") or "")), (Path(b) / str(mb.get("stand_bank") or ""))
    if not (ma.get("stand_bank") and mb.get("stand_bank") and fa.is_file() and fb.is_file()):
        return dict(out, subset=None, keys_equal=None, note="a stand bank is missing")
    with np.load(fa) as da, np.load(fb) as db:
        ka = da["episode"].astype(np.int64) * 10 ** 7 + da["t"].astype(np.int64)
        kb = db["episode"].astype(np.int64) * 10 ** 7 + db["t"].astype(np.int64)
        idx = {int(k): i for i, k in enumerate(ka)}
        pos = [idx.get(int(k)) for k in kb]
        outside = sum(1 for x in pos if x is None)
        ib = np.array([j for j, x in enumerate(pos) if x is not None], dtype=np.int64)
        ia = np.array([x for x in pos if x is not None], dtype=np.int64)
        eq = {k: (bool(np.array_equal(da[k][ia], db[k][ib])) if ia.size and k in da.files and k in db.files
                  else None) for k in BANK_KEYS}
    return dict(out, rows_outside_reference=outside, subset=outside == 0, keys_equal=eq)


BANK_NOT_READ = ("No later step of this campaign reads a stand bank: D3 trains by online DAgger on the states the "
                 "student visits (its own capped buffer), not by offline BC on p2r banks; the gates read raws only.")


def bank_cause(bk: dict | None) -> str:
    """Why the two stand banks differ in size, from the logged caps (pure). The reservoir code and
    DEFAULT_BANK_MAX = 20000 are unchanged in the launcher since f018f357 (checked with git log -p for this
    calibration): the 7006dff1 -> 7a16b8d1 / b5ba0c63 -> 59539301 sha moves did not touch the stand bank."""
    if not bk or bk.get("subset") is None:
        return BANK_NOT_READ
    r, n = (bk.get("reference") or {}).get("bank_max_logged") or [], (bk.get("rerun") or {}).get("bank_max_logged") or []
    if r and n and r != n:
        return (f"The size difference is the bank CAP, not the launcher sha: the probe ran with bank_max {r} "
                f"(P2R_BANK_MAX set for it, P2R_Z2C_PLAN.md step 3), the re-run with {n} (sn_calib_rows.sh sets no "
                f"cap: the launcher default {bk.get('launcher_default_bank_max')}), so the re-run's bank is a "
                "reservoir sample of the same rows; the stand-bank code did not move with the launcher sha. "
                + BANK_NOT_READ)
    return BANK_NOT_READ


def bank_note(bk: dict | None) -> str:
    """One line for calib.md (pure)."""
    if not bk:
        return "stand bank: not read"
    r, n = bk.get("reference") or {}, bk.get("rerun") or {}
    if bk.get("subset") is None:
        return f"stand bank: {bk.get('note')}"
    same = r.get("rows") == n.get("rows") and r.get("seen") == n.get("seen")
    eq = bk.get("keys_equal") or {}
    s = (f"stand bank (`p2r_stand_bank.npz`, not a raw key): reference {r.get('rows')} rows kept of {r.get('seen')} "
         f"seen (logged bank_max {r.get('bank_max_logged') or '?'}), re-run {n.get('rows')} of {n.get('seen')} "
         f"(logged bank_max {n.get('bank_max_logged') or '?'}; launcher default {bk.get('launcher_default_bank_max')})")
    if same:
        return s + "; same size."
    s += (f" -- the banks DIFFER in size; the re-run's rows are {'a subset of' if bk.get('subset') else 'NOT a subset of'}"
          f" the reference's by (episode, t)"
          + (f", {', '.join(k for k, v in eq.items() if v)} bitwise equal on every shared row" if any(eq.values()) else "")
          + (f"; NOT equal: {', '.join(k for k, v in eq.items() if v is False)}" if any(v is False for v in eq.values())
             else "") + ".")
    return s


def identity_verdict(diff: list[str], n_ref: int, n_rerun: int, extra_keys=()) -> dict:
    """Pure: PASS iff both stamps hold raw episodes, the same number, and p2r_bitwise_diff found nothing."""
    keys = set()
    for line in diff:
        m = re.match(r"^episode_\d+ (\S+): not bitwise equal", line) or re.match(r"^episode_\d+: key '([^']+)'", line)
        if m:
            keys.add(m.group(1))
    why = []
    if not n_ref or not n_rerun:
        why.append(f"no raw episodes (reference {n_ref}, rerun {n_rerun})")
    if n_ref != n_rerun:
        why.append(f"episode count differs (reference {n_ref}, rerun {n_rerun})")
    if diff:
        why.append(f"{len(diff)} key-episode mismatches over keys {sorted(keys)}")
    return {"verdict": FAIL if why else PASS, "why": why or ["every raw key of every episode bitwise equal"],
            "differing_keys": sorted(keys), "n_mismatch_lines": len(diff), "first_mismatches": diff[:8],
            "episodes": [n_ref, n_rerun], "keys_only_in_rerun": list(extra_keys)}


# ================================================================================================ R composite (pure)
def merge_r(base: dict, sn: dict | None) -> dict:
    """R = P2R_RSTg_s18 (suite, walkstop, field_turn) + SN_RSTg_s18 field_hard es1-4 when all four are complete;
    else P2R_RSTg_s18's field_hard es1 alone (n = 100) and the field_hard clauses read PARTIAL."""
    r = copy.deepcopy(base)
    sn = sn or {}
    have = [s for s in FH_SEEDS if (sn.get("stamps") or {}).get(f"{FH}@{s}")]
    if len(have) == len(FH_SEEDS):
        for s in FH_SEEDS:
            k = f"{FH}@{s}"
            r["stamps"][k] = sn["stamps"][k]
            r["field"][k] = (sn.get("field") or {}).get(k) or {}
            r["validity"][k] = (sn.get("validity") or {}).get(k) or []
            r["p2r_mode"][k] = (sn.get("p2r_mode") or {}).get(k)
            r["code_state"][k] = (sn.get("code_state") or {}).get(k)
        r["fh_seeds"] = list(FH_SEEDS)
        r["fh_source"] = {"name": sn.get("name"), "seeds": list(FH_SEEDS), "partial": False}
    else:
        for s in FH_SEEDS[1:]:
            k = f"{FH}@{s}"
            for d in (r["stamps"], r["field"], r["validity"]):
                d.pop(k, None)
        r["fh_seeds"] = [1]
        r["fh_source"] = {"name": base.get("name"), "seeds": [1], "partial": True,
                          "sn_complete_seeds": have,
                          "note": f"SN_RSTg_s18 field_hard complete at es{have or '-'} only: R's field_hard is "
                                  "P2R_RSTg_s18 es1 (n = 100), PARTIAL"}
    r["name"] = f"{base.get('name')} + {sn.get('name') or NAMES['R_SN']} (field_hard)"
    return r


# ================================================================================================ verdicts (pure)
def _cl(value, line, ok, state="eval", seed=None, partial=False, short=None, note=None) -> dict:
    """One clause. ``state``: eval (read; ``ok`` decides) / pending (an input is missing) / na (does not apply)."""
    return {"value": value, "line": line, "ok": bool(ok) if state == "eval" else None, "state": state, "seed": seed,
            "partial": bool(partial), "short": short, "note": note}


def verdict_of(clauses: dict, invalid: list) -> dict:
    """INVALID > FAIL (an evaluated, non-partial clause fails; decisive over PENDING) > PENDING (a clause pending, or
    nothing evaluable) > PARTIAL (a partial clause; ``provisional`` = its own pass / fail) > PASS."""
    failed = [k for k, c in clauses.items() if c["state"] == "eval" and not c["ok"]]
    pend = [k for k, c in clauses.items() if c["state"] == "pending"]
    na = [k for k, c in clauses.items() if c["state"] == "na"]
    part = [k for k, c in clauses.items() if c["state"] == "eval" and c["partial"]]
    out = {"failed": failed, "pending": pend, "na": na, "partial": part, "provisional": None,
           "invalid": list(invalid)}
    if invalid:
        return dict(out, verdict=INVALID)
    if [k for k in failed if not clauses[k]["partial"]]:
        return dict(out, verdict=FAIL)
    if pend or not [c for c in clauses.values() if c["state"] == "eval"]:
        return dict(out, verdict=PENDING)
    if part:
        return dict(out, verdict=PARTIAL, provisional=FAIL if [k for k in part if k in failed] else PASS)
    return dict(out, verdict=PASS)


def _invalid(srcs) -> list[str]:
    """Validity reasons of every (reads, stamp key) a gate uses -- its own stamps and its line sources."""
    out = []
    for rd, keys in srcs:
        for k in keys:
            out += [f"[{rd.get('role')}] {r}" for r in ((rd.get("validity") or {}).get(k) or [])]
    return out


def _invalid_why(res: dict) -> list[str]:
    return list(res.get("invalid") or [])


def _sources(srcs) -> dict:
    return {f"{rd.get('role')} {k}": (rd.get("stamps") or {}).get(k) for rd, keys in srcs for k in keys}


def _seed_get(d, s):
    d = d or {}
    return d.get(s) if s in d else d.get(str(s))


def _has(rd: dict, key: str) -> bool:
    """The reference has a (complete: p2_read.newest_stamp only returns complete stamps) stamp at ``key``."""
    return bool((rd.get("stamps") or {}).get(key))


def _st(rd: dict, key: str, read) -> str:
    """A head clause's state: ``eval`` when its read exists, or when its stamp does (a complete stamp whose read is
    None FAILS closed -- the clause's own None value fails it); ``pending`` only when the stamp is missing."""
    return "eval" if read or _has(rd, key) else "pending"


def _unreadable(rd: dict, key: str, what: str) -> list[str]:
    """A LINE source's complete stamp with no readable statistic -> an INVALID reason (the line cannot be drawn); a
    missing line stamp is PENDING (no reason)."""
    return ([f"[{rd.get('role')}] {key}: complete stamp, {what} unreadable -- the line cannot be drawn"]
            if _has(rd, key) else [])


def _closed(read, what: str) -> str | None:
    return None if read else f"complete stamp, {what} None: fails closed"


SUITE_KEYS = tuple(f"suite@{s}" for s in SEEDS)
WS_KEYS = tuple(f"{WS}@{s}" for s in SEEDS)
STILL_SHORT = {"steps_per_s": "steps", "clearance_m": "clear", "off_ground_share": "offgnd",
               "hover_episodes_per_100": "hover", "taps_per_s": "taps", "ks_m_s": "KS"}


def gate_g1(rd: dict) -> dict:
    cl = {}
    for s in SEEDS:
        summ = _seed_get(rd.get("still"), s)
        state = _st(rd, f"suite@{s}", summ)
        for k, (v, t, ok) in S.still_clauses(summ).items():
            cl[f"{k} suite es{s}"] = _cl(v, t, ok, state, s, short=STILL_SHORT.get(k, k),
                                         note=None if state == "pending" else _closed(summ, "suite stand summary"))
    srcs = [(rd, SUITE_KEYS)]
    return dict(verdict_of(cl, _invalid(srcs)), clauses=cl, sources=_sources(srcs),
                statistic="p2s_stand.still_clauses(p2s_stand.head_summary(suite es<k>))")


def z2_parts(rd: dict, probe: dict, z1: dict) -> dict:
    """p2_read.z2_stand_gates per seed (relative to the probe), its clauses split by gate: STILL -> G1 (cross-check),
    the falls clause -> G3, the rest -> G2; each clause's pending state from its own inputs."""
    out = {}
    for s in SEEDS:
        g = P.z2_stand_gates(rd.get("z2") or {}, rd.get("still") or {}, z1.get("z2"), probe.get("z2"), seeds=(s,),
                             relative=True)
        ws = _seed_get((rd.get("z2") or {}).get("walkstop"), s)
        pw = _seed_get((probe.get("z2") or {}).get("walkstop"), s)
        zw = _seed_get((z1.get("z2") or {}).get("walkstop"), s)
        wk = f"{WS}@{s}"
        head_state = _st(rd, wk, ws)
        g1x, g2, g3, inv2, inv3 = {}, {}, {}, [], []
        for name, (v, t, ok) in g["clauses"].items():
            if name.startswith("STILL "):
                g1x[name] = (v, t, ok)
            elif "stand falls" in name:
                if not zw:
                    inv3 += _unreadable(z1, wk, f"Z1 walkstop es{s} Z2 read (the stand-falls line)")
                # CHARTER 2026-10-12 (user decision, before any trained head's gates were read): the clause is the
                # RAW COUNT of walkstop stand falls <= max(1, Z1's count) per env seed, not z2_stand_gates' rate per
                # reached stands (one fall that cuts an episode's later stands reads 300/298 = 1.0067 > 1).
                vc = None if not ws else ws.get("falls")
                tc = None if not zw or zw.get("falls") is None else max(G3_STAND_FALLS_FLOOR, zw["falls"])
                okc = None if vc is None or tc is None else vc <= tc
                g3[f"walkstop stand falls (count) es{s} <= max({G3_STAND_FALLS_FLOOR}, Z1's count)"] = _cl(
                    vc, tc, okc, head_state if zw else "pending", s, short="ws falls",
                    note=(None if head_state == "pending" else _closed(ws, f"walkstop es{s} Z2 read"))
                    or f"z2_stand_gates' own rate clause (reported): {name} = {v} vs {t}")
            else:
                line_missing = name.startswith("median settle") and (not pw or pw.get("settle_median_s") is None)
                if line_missing:
                    inv2 += _unreadable(probe, wk, f"probe walkstop es{s} median settle (the relative settle line)")
                short = ("settle med" if name.startswith("median settle") else
                         name.split(" walkstop")[0].replace("post_settle_", "post ").replace("_per_s", "").replace(
                             "_m_s", " KS").replace("ks KS", "KS"))
                g2[name] = _cl(v, t, ok, "pending" if line_missing else head_state, s, short=short,
                               note=None if head_state == "pending" else _closed(ws, f"walkstop es{s} Z2 read"))
        out[s] = {"g1_crosscheck": g1x, "g2": g2, "g3": g3, "inv_g2": inv2, "inv_g3": inv3,
                  "z2_stand_gates_verdict": g["verdict"], "z2_stand_gates_pending": g["pending"],
                  "z2_stand_gates_reported": g["reported"]}
    return out


def gate_g2(rd: dict, parts: dict, probe: dict) -> dict:
    cl = {k: c for s in SEEDS for k, c in parts[s]["g2"].items()}
    srcs = [(rd, WS_KEYS), (probe, WS_KEYS)]
    inv = _invalid(srcs) + [x for s in SEEDS for x in parts[s].get("inv_g2") or []]
    return dict(verdict_of(cl, inv), clauses=cl, sources=_sources(srcs),
                statistic="p2_read.z2_stand_gates(relative=True, probe=P2R_RSTg_s18): homed / settled / median "
                          "settle / post-settle clauses",
                z2_stand_gates_verdict={s: parts[s]["z2_stand_gates_verdict"] for s in SEEDS})


def gate_g3(rd: dict, parts: dict, z1: dict, r: dict, head_kind: str | None = None) -> dict:
    cl, inv_extra = {}, []
    role = rd.get("role")
    for s in SEEDS:
        cl.update(parts[s]["g3"])
        inv_extra += parts[s].get("inv_g3") or []
        summ = _seed_get(rd.get("still"), s)
        v = None if not summ else summ.get("falls")
        state = _st(rd, f"suite@{s}", summ)
        cl[f"suite stand falls es{s} == 0"] = _cl(v, 0, v is not None and v == 0, state, s, short="suite falls",
                                                  note=None if state == "pending" else _closed(summ, "suite stand "
                                                                                                    "summary"))
        e, re_ = _seed_get(rd.get("exit"), s) or {"status": "missing"}, _seed_get(r.get("exit"), s) or {
            "status": "missing"}
        name = f"walkstop exit-handover falls within 2 s es{s} <= max({G3_EXIT_FLOOR}, R's)"
        st = e.get("status")
        if st == "no_mode" and head_kind != "stand_net":
            cl[name] = _cl(None, None, None, "na", s, short="exit falls",
                           note="no run_meta.p2r_mode: a single-network eval has no stand-mode exit")
            continue
        if st == "missing":
            cl[name] = _cl(None, None, None, "pending", s, short="exit falls", note="walkstop stamp missing")
            continue
        if st != "ok":
            inv_extra.append(f"[{rd.get('role')}] walkstop es{s}: exit read {st} (p2r_hold_reads n_episodes "
                             f"{e.get('n_mode_episodes')} of {e.get('n_raw_episodes')} raw episodes) -> None")
            cl[name] = _cl(None, None, None, "pending", s, short="exit falls", note=f"exit read {st}")
            continue
        rs = re_.get("status")
        if rs == "missing":
            cl[name] = _cl(e["count"], None, None, "pending", s, short="exit falls", note="R walkstop stamp missing")
            continue
        if rs != "ok":
            inv_extra.append(f"[R] walkstop es{s}: exit read {rs} -> the line is unreadable")
            cl[name] = _cl(e["count"], None, None, "pending", s, short="exit falls", note=f"R exit read {rs}")
            continue
        inv_extra += exit_denominator_reasons(role, s, e, re_)
        au, ra = e.get("audit") or {}, re_.get("audit") or {}
        unexc = au.get("missing_unexcused")
        line = max(G3_EXIT_FLOOR, re_["count"])
        cl[name] = _cl(e["count"], line, e["count"] <= line and unexc == 0, "eval", s, short="exit falls",
                       note=f"{e['count']} falls in {e.get('exits')} exits (design {au.get('design_exits')}, R "
                            f"{ra.get('exits')}); exits not made: {au.get('missing_excused')} after a stand-mode fall "
                            f"(the stand-falls clause's), {unexc} otherwise"
                            + (" -> fails closed (the raw count is only comparable on R's full set)" if unexc else ""))
    srcs = [(rd, WS_KEYS + SUITE_KEYS), (z1, WS_KEYS), (r, WS_KEYS)]
    return dict(verdict_of(cl, _invalid(srcs) + inv_extra), clauses=cl, sources=_sources(srcs),
                statistic="z2_stand_gates falls clause (line max(1, Z1)); p2s head_summary suite falls; "
                          "p2r_hold_reads handover_falls_by_entry_phase summed (raw count per env seed, on R's "
                          "denominator: exit_audit)")


def exit_denominator_reasons(role: str | None, s: int, e: dict, re_: dict) -> list[str]:
    """INVALID reasons for G3's raw-count exit clause (pure): the head's raw episode count must equal R's same-seed
    count, both must run the same schedule (stand -> move boundaries per episode), R must have made every exit its
    schedule has (R's line is a count on that full set), and the head may not exit stand mode more often than the
    schedule allows (D5 switch contract). An exit the head's episode never made is not here: it fails the clause
    closed unless the episode fell in stand mode (:func:`exit_audit`)."""
    au, ra = e.get("audit"), re_.get("audit")
    if not au or not ra:
        return [f"[{role}] walkstop es{s}: exit audit missing (head {bool(au)}, R {bool(ra)}) -> no denominator"]
    out = []
    if e.get("n_raw_episodes") != re_.get("n_raw_episodes") or au["episodes"] != ra["episodes"]:
        out.append(f"[{role}] walkstop es{s}: {au['episodes']} episodes vs R's {ra['episodes']} -- the raw-count line "
                   "needs R's denominator")
    if au["design_exits_per_episode"] != ra["design_exits_per_episode"]:
        out.append(f"[{role}] walkstop es{s}: {au['design_exits_per_episode']} stand exits per episode by schedule "
                   f"vs R's {ra['design_exits_per_episode']} (a different schedule)")
    if ra["exits"] != ra["design_exits"] or ra["extra"]:
        out.append(f"[R] walkstop es{s}: R made {ra['exits']} exits of {ra['design_exits']} by schedule -- R's "
                   "raw-count line is not on the full set")
    if au["extra"]:
        out.append(f"[{role}] walkstop es{s}: {au['extra']} stand-mode exits beyond the schedule's "
                   f"{au['design_exits']} (D5 switch contract: one exit per stand -> move boundary)")
    return out


def fh_pool(rd: dict, seeds) -> tuple:
    """(p2_read.pooled_seeds completion over ``seeds`` -> (rate, n) or None, pooled stand-hold falls {falls, n} or
    None) of one reference's field_hard rows."""
    f = rd.get("field") or {}
    comps = {s: (f.get(f"{FH}@{s}") or {}).get("comp") for s in seeds}
    pc = P.pooled_seeds({"comp": {"field_hard": comps}}, "field_hard", tuple(seeds))
    stands = [(f.get(f"{FH}@{s}") or {}).get("stand") for s in seeds]
    ps = (None if any(x is None for x in stands) else
          {"falls": sum(int(x.get("falls") or 0) for x in stands), "n": sum(int(x.get("n") or 0) for x in stands)})
    return pc, ps


def _comp_ok(c) -> bool:
    """A completion read (p2_read.comp_read) with a rate and a nonzero n -- what pooled_seeds needs."""
    return bool(c) and c.get("rate") is not None and bool(c.get("n"))


def _fv_point(rd: dict, seeds) -> dict:
    """The arm / HOLD view z2_field_vs_hold reads, with the pooled field_hard pseudo-row added."""
    pc, ps = fh_pool(rd, seeds)
    have_all = all((rd.get("stamps") or {}).get(f"{FH}@{s}") for s in seeds)
    stamps = dict(rd.get("stamps") or {}, **{FH_POOL: "pooled" if have_all else None})
    field = dict(rd.get("field") or {}, **{FH_POOL: {"stand": ps, "comp": ({"rate": pc[0], "n": pc[1]} if pc
                                                                             else None)}})
    return {"name": rd.get("name"), "stamps": stamps, "field": field}


def gate_g4(rd: dict, refs: dict, hold: dict) -> dict:
    seeds = tuple(rd.get("fh_seeds") or FH_SEEDS)
    partial = bool((rd.get("fh_source") or {}).get("partial"))
    cl, inv = {}, []
    f, rf = rd.get("field") or {}, refs.get("field") or {}
    hk, rk = [f"{FH}@{s}" for s in seeds], [f"{FH}@{s}" for s in FH_SEEDS]
    h_miss = [k for k in hk if not _has(rd, k)]
    h_bad = [k for k in hk if _has(rd, k) and not _comp_ok((f.get(k) or {}).get("comp"))]
    r_miss = [k for k in rk if not _has(refs, k)]
    r_bad = [k for k in rk if _has(refs, k) and not _comp_ok((rf.get(k) or {}).get("comp"))]
    for k in r_bad:
        inv += _unreadable(refs, k, "schedule_completion_rate (the field_hard line)")
    name = f"field_hard es{'+'.join(map(str, seeds))} completion >= REFS - max({P.P2S_OBST_FH}, 2 SE)"
    if h_bad:
        cl[name] = _cl(None, None, False, "eval", None, partial, short="field_hard",
                       note=f"complete stamp(s) {h_bad} without schedule_completion_rate: fails closed")
    elif h_miss or r_miss or r_bad:
        cl[name] = _cl(None, None, None, "pending", None, partial, short="field_hard",
                       note=f"stamps missing: head {h_miss}, REFS {r_miss}" + (f"; REFS unreadable {r_bad}" if r_bad
                                                                                 else ""))
    else:
        (fh, nf), (rfh, nr) = fh_pool(rd, seeds)[0], fh_pool(refs, FH_SEEDS)[0]
        se = P.se_need(rfh, nr, fh, nf)["se"]
        line = rfh - max(P.P2S_OBST_FH, P.P2B_SE_K * se)
        cl[name] = _cl(fh, line, fh >= line - EPS, "eval", None, partial, short="field_hard",
                       note=f"n={nf} vs REFS {rfh:.4f} n={nr}, SE {se:.4f}")
    fa, fr = (f.get(FT1) or {}).get("comp"), (rf.get(FT1) or {}).get("comp")
    name = f"field_turn es1 completion >= REFS - max({P.P2S_FT_FLOOR}, band)"
    if not _comp_ok(fr):
        inv += _unreadable(refs, FT1, "schedule_completion_rate (the field_turn line)")
    if _has(rd, FT1) and not _comp_ok(fa):
        cl[name] = _cl((fa or {}).get("rate"), None, False, "eval", 1, short="field_turn",
                       note="complete field_turn stamp without schedule_completion_rate: fails closed")
    elif not _comp_ok(fa) or not _comp_ok(fr):
        cl[name] = _cl((fa or {}).get("rate"), None, None, "pending", 1, short="field_turn",
                       note=f"field_turn es1 stamp missing (head {_has(rd, FT1)}, REFS {_has(refs, FT1)})")
    else:
        band = max(P.binom_se(fr["rate"], int(fr.get("n") or 100)), P.COMP_BAND_FLOOR)
        line = fr["rate"] - max(P.P2S_FT_FLOOR, band)
        cl[name] = _cl(fa["rate"], line, fa["rate"] >= line - EPS, "eval", 1, short="field_turn",
                       note=f"band {band:.4f} = max(binom_se(REFS {fr['rate']:.3f}, n={fr.get('n')}), "
                            f"{P.COMP_BAND_FLOOR}); es1 only (REFS and R have no field_turn es2)")
    arm, hp = _fv_point(rd, seeds), _fv_point(hold, FH_SEEDS)
    fv = P.z2_field_vs_hold(arm, hp, rows=(FT1, FH_POOL))
    for k, short, part in ((FT1, "stand-fall FT", False), (FH_POOL, "stand-fall FH", partial)):
        key = f"{k} stand-fall rate <= HOLD + {P.Z2_FIELD_SE_K:g} SE(diff)"
        if key in fv["clauses"]:
            v, t, ok = fv["clauses"][key]
            cl[key] = _cl(v, t, ok, "eval", 1 if k == FT1 else None, part, short=short,
                          note=None if v is not None else "complete stamp(s), stand summary None: fails closed")
            continue
        hold_keys = (FT1,) if k == FT1 else tuple(rk)
        hold_pending = any(f" {k} " in p for p in fv["pending"])
        if hold_pending and all(_has(hold, x) for x in hold_keys):
            inv.append(f"[HOLD] {k}: complete stamp(s), stand summary / completion unreadable -- the stand-fall line "
                       "cannot be drawn")
        cl[key] = _cl(None, None, None, "pending", None, part, short=short,
                      note=f"not run: {fv['not_run']}; HOLD pending: {fv['pending']}")
    fv_default = P.z2_field_vs_hold(rd, hold)          # the RSTg precedent's rows (field_turn es1, field_hard es1)
    fh_keys = tuple(f"{FH}@{s}" for s in seeds)
    srcs = [(rd, (FT1,) + fh_keys), (refs, (FT1,) + tuple(rk)), (hold, (FT1,) + tuple(rk))]
    hfh = fh_pool(hold, FH_SEEDS)
    return dict(verdict_of(cl, _invalid(srcs) + inv), clauses=cl, sources=_sources(srcs),
                statistic="p2s_verdict OBSTACLE LOSS field_hard line (se_need); p2s_cost field_turn band; "
                          "z2_field_vs_hold stand-fall clause on field_turn es1 + field_hard pooled",
                fh_source=rd.get("fh_source"),
                reported={"z2_field_vs_hold default rows (FT es1, FH es1)": fv_default["clauses"],
                          "completion vs HOLD (pooled rows)": {k: v for k, v in fv["clauses"].items()
                                                               if "completion" in k},
                          "field_hard pooled stand falls": fh_pool(rd, seeds)[1],
                          "HOLD field_hard pooled": hfh[0], "HOLD field_hard stand falls": hfh[1]})


def _p2s_view(rd: dict) -> dict:
    """The p2s-shaped view p2s_onset / p2s_stand_bands read: suite head summaries + walkstop_read per seed."""
    return {"p2s": {"suite": rd.get("still") or {}, "walkstop": rd.get("walkstop") or {}}}


def gate_g5(rd: dict, refs: dict) -> dict:
    cl, inv = {}, []
    role = rd.get("role")
    band = P.p2s_stand_bands(_p2s_view(refs)).get("onset_s")
    for s in SEEDS:
        wr = _seed_get(refs.get("walkstop"), s)
        if not wr or wr.get("onset_median_s") is None:
            inv += _unreadable(refs, f"{WS}@{s}", f"walkstop es{s} onset median (the onset line / band)")
    for s in SEEDS:
        wk = f"{WS}@{s}"
        w, wr = _seed_get(rd.get("walkstop"), s), _seed_get(refs.get("walkstop"), s)
        name = f"walkstop onset median es{s} <= REFS + max({P.P2S_ONSET_FLOOR}, band)"
        if not wr or wr.get("onset_median_s") is None or band is None or _st(rd, wk, w) == "pending":
            cl[name] = _cl((w or {}).get("onset_median_s"), None, None, "pending", s, short="onset")
        else:
            line = wr["onset_median_s"] + max(P.P2S_ONSET_FLOOR, band)
            v = (w or {}).get("onset_median_s")
            cl[name] = _cl(v, line, v is not None and v <= line + EPS, "eval", s, short="onset",
                           note=None if v is not None else ("every onset censored (fail closed)" if w else
                                                            _closed(w, "walkstop_read")))
        # the stall clause: the median drops censored onsets, so a stall S1 never recovers from is counted here
        sp, spr = _seed_get(rd.get("onset_split"), s), _seed_get(refs.get("onset_split"), s)
        name = f"walkstop onset stalls es{s} <= REFS + {G5_STALL_FLOOR}"
        if not spr:
            inv += _unreadable(refs, wk, f"walkstop es{s} onset split (the stall line)")
            cl[name] = _cl((sp or {}).get("stalled"), None, None, "pending", s, short="stalls")
        elif _st(rd, wk, sp) == "pending":
            cl[name] = _cl(None, None, None, "pending", s, short="stalls")
        elif not sp:
            cl[name] = _cl(None, spr["stalled"] + G5_STALL_FLOOR, False, "eval", s, short="stalls",
                           note=_closed(sp, "walkstop onset split"))
        else:
            inv += stall_denominator_reasons(role, s, sp, spr)
            line = spr["stalled"] + G5_STALL_FLOOR
            cl[name] = _cl(sp["stalled"], line, sp["stalled"] <= line, "eval", s, short="stalls",
                           note=f"{sp['stalled']} stalled of {sp['reached']} move holds reached ({sp['unreached']} "
                                f"not reached: the episode ended first, G3's); REFS {spr['stalled']} of "
                                f"{spr['reached']}; walkstop_read onset_censored {(w or {}).get('onset_censored')}")
    oa, orf = P.p2s_onset(_p2s_view(rd)), P.p2s_onset(_p2s_view(refs))
    name = f"walkstop onset median pooled es1+es2 <= REFS + max({P.P2S_ONSET_FLOOR}, band) (p2s_cost)"
    have = all(_seed_get(rd.get("walkstop"), s) or _has(rd, f"{WS}@{s}") for s in SEEDS)
    if have and orf is not None and band is not None:
        line = orf + max(P.P2S_ONSET_FLOOR, band)
        cl[name] = _cl(oa, line, oa is not None and oa <= line + EPS, "eval", None, short="onset pooled",
                       note=None if oa is not None else "no uncensored onset / walkstop_read None: fails closed")
    else:
        cl[name] = _cl(oa, None, None, "pending", None, short="onset pooled")
    arm = {"still": rd.get("still") or {}, "z2": rd.get("z2") or {}}
    h = P.z2_height_vs_refs(arm, dict(refs, name=refs.get("name")), seeds=SEEDS)
    for k, (v, t, ok) in h["clauses"].items():
        lab, seed = k.split(" ")[0], int(k.split(" ")[1][2:])
        key = f"suite@{seed}" if lab == "suite" else f"{WS}@{seed}"
        a_summ = (_seed_get(rd.get("still"), seed) if lab == "suite"
                  else _seed_get((rd.get("z2") or {}).get("walkstop"), seed))
        rv = (h["heights"].get(f"{lab} es{seed}") or [None, None])[1]
        if rv is None:
            inv += _unreadable(refs, key, f"{lab} es{seed} body height (the height line)")
            cl[k] = _cl(v, t, None, "pending", seed, short=f"height {lab}")
            continue
        state = _st(rd, key, a_summ)
        cl[k] = _cl(v, t, ok, state, seed, short=f"height {lab}",
                    note=None if state == "pending" else _closed(a_summ, f"{lab} es{seed} summary"))
    for p in h["pending"]:
        m = re.search(r" (suite|walkstop) es(\d+) body height$", p)
        if m:
            inv += _unreadable(refs, f"suite@{m.group(2)}" if m.group(1) == "suite" else f"{WS}@{m.group(2)}",
                               f"{m.group(1)} es{m.group(2)} summary (the height line)")
        cl[f"pending: {p}"] = _cl(None, None, None, "pending", None, short="height")
    srcs = [(rd, WS_KEYS + SUITE_KEYS), (refs, WS_KEYS + SUITE_KEYS)]
    return dict(verdict_of(cl, _invalid(srcs) + inv), clauses=cl, sources=_sources(srcs),
                statistic="p2s_cost onset (walkstop_read median, p2s_stand_bands onset band) per seed + pooled; "
                          "onset stalls (p2s_stand.stand_onset_pairs / episode_onset_latency on reached move holds) "
                          "per seed vs REFS + 1; z2_height_vs_refs (suite + walkstop, per seed)",
                stalls={f"es{s}": {"head": _seed_get(rd.get("onset_split"), s),
                                   "refs": _seed_get(refs.get("onset_split"), s)} for s in SEEDS},
                onset_band=band, onset={"head_pooled": oa, "refs_pooled": orf,
                                        **{f"head_es{s}": (_seed_get(rd.get("walkstop"), s) or {}).get("onset_median_s")
                                           for s in SEEDS},
                                        **{f"refs_es{s}": (_seed_get(refs.get("walkstop"), s) or {}).get(
                                            "onset_median_s") for s in SEEDS}},
                heights=h["heights"])


def stall_denominator_reasons(role: str | None, s: int, sp: dict, spr: dict) -> list[str]:
    """INVALID for G5's stall count (pure): it is a raw count, so the head must read REFS's same-seed episode count
    and onset pairs per episode (the same schedule)."""
    if sp.get("n_episodes") == spr.get("n_episodes") and sp.get("pairs_per_episode") == spr.get("pairs_per_episode"):
        return []
    return [f"[{role}] walkstop es{s}: {sp.get('n_episodes')} episodes x {sp.get('pairs_per_episode')} onset pairs vs "
            f"REFS {spr.get('n_episodes')} x {spr.get('pairs_per_episode')} -- the stall count needs REFS's "
            "denominator"]


def reported_only(rd: dict) -> dict:
    out = {}
    for s in SEEDS:
        zs = _seed_get((rd.get("z2") or {}).get("suite"), s) or {}
        zw = _seed_get((rd.get("z2") or {}).get("walkstop"), s) or {}
        su = _seed_get(rd.get("still"), s) or {}
        w = _seed_get(rd.get("walkstop"), s) or {}
        out[f"es{s}"] = {"six_down suite (steady)": zs.get("six_down_stand_share"),
                         "six_down walkstop (last 5 s)": zw.get("six_down_stand_share"),
                         "hip_yaw_mean_deg walkstop": zw.get("hip_yaw_mean_deg"),
                         "hip_yaw_max_deg walkstop": zw.get("hip_yaw_max_deg"),
                         "hip_yaw_mean_deg suite": zs.get("hip_yaw_mean_deg"),
                         "settle_median_s walkstop": zw.get("settle_median_s"),
                         "settle_median_s suite": zs.get("settle_median_s"),
                         "KS suite stand": su.get("ks_m_s"),
                         "KS walkstop stands (pooled median)": (w.get("pooled") or {}).get("ks_m_s"),
                         "post-settle KS walkstop": zw.get("post_settle_ks_m_s")}
    return out


def lands(gate: str, role: str, res: dict) -> tuple[str, str | None]:
    """("yes" / "NO" / "pending" / "reported", why) for one cell against PLAN.md D6."""
    exp = EXPECT.get((gate, role))
    v = res.get("verdict")
    if exp is None:
        return "reported", None
    if v == INVALID:
        return "NO", "INVALID (a listed cell cannot be read)"
    if v == PENDING:
        return "pending", "PENDING"
    if v == PARTIAL:
        return "pending", f"PARTIAL (provisional {res.get('provisional')})"
    if v != exp:
        return "NO", f"expected {exp}, observed {v}"
    named = NAMED_CLAUSE.get((gate, role))
    if named and exp == FAIL:
        ev = [k for k, c in res["clauses"].items() if k.startswith(named) and c["state"] == "eval"]
        bad = [k for k in ev if not res["clauses"][k]["ok"]]
        if not ev or len(bad) != len(ev):
            return "NO", f"fails, but not on its named clause {named} at every seed read ({bad} of {ev})"
    return "yes", None


def calibrate(reads: dict, identity: dict | None = None, heads=()) -> dict:
    """The whole read (pure): every gate for every reference (+ ``heads``), expectations, mismatches, pending."""
    refs, hold, probe, z1 = reads["REFS"], reads["HOLD"], reads["R"], reads["Z1"]
    rows, mismatches, pending = {}, [], []
    for role in tuple(ROLES) + tuple(heads):
        rd = reads.get(role)
        if not rd:
            continue
        parts = z2_parts(rd, probe, z1)
        g = {"G1": gate_g1(rd), "G2": gate_g2(rd, parts, probe), "G3": gate_g3(rd, parts, z1, probe, rd.get("kind")),
             "G4": gate_g4(rd, refs, hold), "G5": gate_g5(rd, refs)}
        # G1 cross-check: z2_stand_gates' own STILL clauses == still_clauses
        xs = {k: c for s in SEEDS for k, c in parts[s]["g1_crosscheck"].items()}
        g["G1"]["z2_stand_gates_still_agrees"] = all(
            (xs.get(f"STILL {k}") or (None, None, None))[2] == (c["ok"] if c["state"] == "eval" else False)
            for k, c in g["G1"]["clauses"].items())
        for gate, res in g.items():
            res["expected"] = EXPECT.get((gate, role))
            res["lands"], res["lands_why"] = lands(gate, role, res)
            if res["lands"] == "NO":
                mismatches.append(f"{gate} {GATE_TITLE[gate]} x {role} ({rd.get('name')}): {res['lands_why']}")
            if res["verdict"] in (PENDING, PARTIAL):
                why = [k for k in res["pending"]] or res["partial"]
                pending.append(f"{gate} {GATE_TITLE[gate]} x {role}: {res['verdict']}"
                               + (f" (provisional {res['provisional']})" if res.get("provisional") else "")
                               + (f" -- {'; '.join(why)[:300]}" if why else ""))
            elif res["pending"]:
                pending.append(f"{gate} {GATE_TITLE[gate]} x {role}: verdict {res['verdict']} with pending clauses "
                               f"{res['pending']}")
        rows[role] = {"name": rd.get("name"), "kind": rd.get("kind"), "notes": rd.get("notes") or [],
                      "gates": g, "reported": reported_only(rd), "validity": rd.get("validity"),
                      "bypassed_p2_read_reasons": {k: v for k, v in (rd.get("bypassed") or {}).items() if v},
                      "stamps": rd.get("stamps"), "p2r_mode": rd.get("p2r_mode"), "code_state": rd.get("code_state"),
                      "fh_source": rd.get("fh_source"),
                      "z2_stand_gates": {s: {"verdict": parts[s]["z2_stand_gates_verdict"],
                                             "pending": parts[s]["z2_stand_gates_pending"]} for s in SEEDS},
                      "exit_reads": {s: _seed_get(rd.get("exit"), s) for s in SEEDS},
                      "onset_split": {s: _seed_get(rd.get("onset_split"), s) for s in SEEDS},
                      "stand_net_arm": rd.get("stand_net_arm")}
    if identity is not None and identity.get("verdict") == PENDING:
        pending.append(f"identity check: PENDING -- {'; '.join(identity.get('why') or [])}")
    r_ws = probe.get("walkstop") or {}
    r_onset = {"statistic": "p2s_stand.episode_onset_latency via p2_read.walkstop_read (stand -> lat_l / back, "
                            "5-step trailing mean reaches 0.5 x |cmd| and holds 0.5 s); pooled = p2_read.p2s_onset",
               "stamps": {f"es{s}": (_seed_get(r_ws, s) or {}).get("stamp") for s in SEEDS},
               **{f"es{s}": dict({k: (_seed_get(r_ws, s) or {}).get(k) for k in ("onset_median_s", "onset",
                                                                                   "onset_censored", "n_episodes")},
                                 split=_seed_get(probe.get("onset_split"), s))
                  for s in SEEDS},
               "pooled_median_s": P.p2s_onset(_p2s_view(probe)),
               "refs_pooled_median_s": P.p2s_onset(_p2s_view(refs)),
               "hold_pooled_median_s": P.p2s_onset(_p2s_view(hold))}
    n_exp = len(EXPECT)
    n_yes = sum(1 for (gate, role) in EXPECT if rows.get(role, {}).get("gates", {}).get(gate, {}).get("lands") == "yes")
    invalid_cells = [f"{gate} x {role}: {'; '.join(_invalid_why(row['gates'][gate]))[:400]}"
                     for role, row in rows.items() for gate in GATES if row["gates"][gate]["verdict"] == INVALID]
    return {"rows": rows, "mismatches": mismatches, "pending": pending, "identity": identity, "r_onset": r_onset,
            "invalid_cells": invalid_cells, "coverage": coverage(rows),
            "summary": {"expected_cells": n_exp, "lands_yes": n_yes, "mismatches": len(mismatches),
                        "pending_expected": sum(1 for (gate, role) in EXPECT
                                                if rows.get(role, {}).get("gates", {}).get(gate, {}).get("lands")
                                                == "pending")},
            "listed_numbers": listed_numbers(rows)}


def coverage(rows: dict) -> dict:
    """Negative-control coverage per gate (pure, reported): which references FAIL it, which D6 lists as known-bad,
    and -- per clause kind (the clause's short name) -- the range of values the references read and the tightest
    passing margin |line - value|. A gate no reference fails is unexercised as a negative control: its sensitivity
    rests on its line definitions, not on an observed fail."""
    out = {}
    for gate in GATES:
        fails = [role for role, row in rows.items() if row["gates"][gate]["verdict"] == FAIL]
        kinds: dict = {}
        for role, row in rows.items():
            for k, c in row["gates"][gate]["clauses"].items():
                v, t = c["value"], c["line"]
                if c["state"] != "eval" or isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                d = kinds.setdefault(c["short"] or k, {"min": v, "max": v, "n": 0, "tightest": None})
                d["min"], d["max"], d["n"] = min(d["min"], v), max(d["max"], v), d["n"] + 1
                if c["ok"] and isinstance(t, (int, float)) and not isinstance(t, bool):
                    m = abs(t - v)
                    if d["tightest"] is None or m < d["tightest"]["margin"]:
                        d["tightest"] = {"role": role, "clause": k, "value": v, "line": t, "margin": m}
        out[gate] = {"failing_references": fails,
                     "known_bad_listed": [r for (g, r), v in EXPECT.items() if g == gate and v == FAIL],
                     "exercised_as_negative_control": bool(fails), "by_clause": kinds}
    return out


def listed_numbers(rows: dict) -> list[dict]:
    """PLAN.md D6's parenthetical numbers beside what this read computes (reported, not gated)."""
    def clv(role, gate, key):
        c = (((rows.get(role) or {}).get("gates") or {}).get(gate) or {}).get("clauses", {}).get(key)
        return None if not c else c["value"]

    def g5(role, key):
        return ((((rows.get(role) or {}).get("gates") or {}).get("G5") or {}).get("onset") or {}).get(key)
    out = [{"what": "REFS suite stand steps/s es1 / es2", "listed": 2.83,
            "computed": [clv("REFS", "G1", f"steps_per_s suite es{s}") for s in SEEDS]},
           {"what": "ST2M suite stand taps/s es1 / es2", "listed": 16,
            "computed": [clv("ST2M", "G1", f"taps_per_s suite es{s}") for s in SEEDS]},
           {"what": "HOLD field_hard pooled n=400", "listed": 0.810,
            "computed": [clv("HOLD", "G4", f"field_hard es1+2+3+4 completion >= REFS - max({P.P2S_OBST_FH}, 2 SE)")]},
           {"what": "HOLD walkstop onset pooled (s)", "listed": 0.12, "computed": [g5("HOLD", "head_pooled")]},
           {"what": "REFS walkstop onset pooled (s)", "listed": 0.14, "computed": [g5("REFS", "head_pooled")]},
           {"what": "R median-settle line es1 / es2 (s)", "listed": [3.92, 3.96],
            "computed": [(((rows.get("R") or {}).get("gates") or {}).get("G2") or {}).get("clauses", {}).get(
                f"median settle walkstop es{s} <= probe + {P.Z2_SETTLE_PLUS_S} s", {}).get("line") for s in SEEDS]},
           {"what": "R walkstop homed es1 / es2", "listed": [0.970, 0.977],
            "computed": [clv("R", "G2", f"homed walkstop es{s}") for s in SEEDS]}]
    return out


# ================================================================================================ deviations
DEVIATIONS = [
    "VALIDITY (D5): REFS_S1_s18, P2R_HOLD_s18, P2R_ST2M_s18 and C0R_s18 run STAND_OBS=0 (col 15 = 0). p2_read's Z2 "
    "env rule (z2_env_reasons: 'STAND_OBS=clock or INVALID') would read every one of their stamps INVALID; sn_calib "
    "replaces it for them with the D5 obs-parity rule (STAND_OBS exactly '0', absent fails closed; proxy off; "
    "env-level STAND_MASK off) + p2_read.invalid_reasons vs REFS (ENV4, row env block, bridged code-state sidecar) + "
    "walkstop_flat_fails (+ p2r_mode_fails variant hold / st2 driving REFS's S1 for HOLD / ST2M). The p2_read "
    "reasons bypassed are recorded per stamp in calib.json rows.<role>.bypassed_p2_read_reasons.",
    "VALIDITY (D5, campaign heads; --head): STAND_OBS exactly '0' + run_meta.p2r_mode.variant == 'stand_net' + "
    "p2r_mode.s1_checkpoint_sha256 == REFS's + proxy / env mask off + invalid_reasons vs REFS, fail closed -- in "
    "place of p2_read's Z2 'STAND_OBS=clock or INVALID'. Plus (verifier fix): every stamp names its stand checkpoint "
    "sha (p2r_mode.specialist.sha256, the p2r launcher's second-network record) and its mask flag "
    "(p2r_mode.stand_mask, 0 / 1), and ALL of a head's stamps carry ONE sha and ONE mask -- A-mask and A-full share "
    "the variant and S1's sha, so a NAME mixing both arms or a re-trained checkpoint would otherwise read valid. "
    "These two field names are a contract the step-3 launcher must write (constants STAND_NET_CKPT_KEY / "
    "STAND_NET_MASK_KEY); a launcher that names them otherwise reads INVALID on every stamp until they are aligned. "
    "No stand_net stamp exists yet: exercised by --selftest only.",
    "VALIDITY kept from p2_read (not a deviation, recorded): R stamps (P2R_RSTg_s18, SN_RSTg_s18) use "
    "p2_read.rst_invalid_reasons unchanged (STAND_OBS=clock, proxy off, p2r_mode variant restance on REFS's S1, raw "
    "stand bit; PLAN 'Left for implementation': R keeps the clock rule); Z1 / Z2d use p2_read's Z2 head rule "
    "(z_clock_invalid_reasons + z2_invalid_reasons incl. the mask rule and raw stand bit) + invalid_reasons vs REFS.",
    "READ PATH: REFS / HOLD (and ST2M, C0R) are calibrated by calling z2_stand_gates / still_clauses / "
    "z2_field_vs_hold / z2_height_vs_refs directly on their summaries (p2_read.z2_light_point reads), not through "
    "p2_read's --p2r-z2 path, which reads them INVALID and reads one Z head only. Z1 / Z2d are read the same way: "
    "the --p2r-z2 ladder rungs that are not D6 gates (HAND LOSS, DOSE, FLAT FAIL, attribution) are not run.",
    "G2 / G3 are subsets of p2_read.z2_stand_gates' clauses (relative=True, probe P2R_RSTg_s18): its STILL clauses "
    "are G1's (cross-checked equal to still_clauses), its walkstop-falls clause is G3's, the rest is G2. Its own "
    "combined verdict (which also needs the suite Z2 read for the reported six-down) is recorded per seed.",
    "G3 exit clause: p2_read.p2r_hold_reads returns n_episodes 0 (not None, contrary to its docstring) when no "
    "episode carries p2r_stand_mode; sn_calib maps n_episodes == 0 or < the raw episode count to None -> INVALID "
    "for a stamp with run_meta.p2r_mode (and for a stand_net head), and to n/a for a plain single-network eval "
    "without p2r_mode (REFS, Z1, Z2d, C0R: no stand-mode switch, no exit). R's line source must read ok.",
    "G3 exit clause DENOMINATOR (verifier fix; the plan says only 'raw count per env seed (200 exits)'): a raw count "
    "is only comparable on R's set, so per env seed the head must read R's raw episode count and schedule (stand -> "
    "move boundaries per episode, from run_meta.hold_labels), R must have made every scheduled exit (R: 200 of 200), "
    "and the head may make no exit beyond the schedule (D5 switch contract) -- any of these broken -> INVALID. A "
    "scheduled exit the head's episode never made FAILS the clause closed, unless that episode fell inside a "
    "stand-mode segment (p2r_hold_reads' falls_in_stand_mode definition): a stand fall is the walkstop stand-falls "
    "clause's, which keeps its pre-registered max(1, Z1's) per 300 floor instead of being double-counted here. "
    "Exits, design exits, excused / unexcused missing exits are in calib.json rows.<role>.exit_reads.<seed>.audit.",
    "G4 field_hard line: p2s_verdict / p2r_tier1_verdict's inline OBSTACLE LOSS expression restated with "
    "p2_read.se_need, P2S_OBST_FH 0.05, P2B_SE_K 2 (fh >= REFS - max(0.05, 2 SE)); pooled with p2_read.pooled_seeds.",
    "G4 field_turn band: p2_read has one -- p2s_cost's inline band_ft = max(binom_se(REFS, n), COMP_BAND_FLOOR "
    "0.04) -- restated (line REFS - max(P2S_FT_FLOOR 0.05, band)); es1 only (REFS and R have no field_turn es2).",
    "G4 stand-fall clause: p2_read.z2_field_vs_hold on field_turn es1 and on a pooled field_hard es1-4 pseudo-row "
    "(n = 400 each side; R PARTIAL: es1, n = 100), not the RSTg precedent's field_hard es1 row (whose result is "
    "reported); its completion-vs-HOLD clauses are reported, not gated (D6 G4's completion lines are vs REFS).",
    "G4 for R: SN_RSTg_s18 field_hard es1-4 when all four are complete, else P2R_RSTg_s18 es1 (n = 100) and the "
    "field_hard clauses PARTIAL (provisional verdict shown); suite / walkstop / field_turn stay P2R_RSTg_s18's.",
    "G5 onset: read per env seed (D6 'each at es1 and es2': walkstop_read onset median vs REFS same seed) AND pooled "
    "es1+es2 (p2s_onset, p2s_cost's own statistic), line REFS + max(0.3, p2s_stand_bands onset band); all must "
    "hold. A present walkstop read whose every onset is censored (median None) fails closed.",
    "G5 onset STALLS (verifier fix; not in D6 or p2s_cost): walkstop_read's median and p2s_onset drop censored "
    "onsets, so a head after which S1 never gets going on most handovers would pass on the few that do -- the "
    "'Out: S1 sees the student's actions' failure D5 says onset measures. Added per env seed: stalled onsets (a "
    "stand -> move hold the episode reached whose p2s_stand.episode_onset_latency is None) <= REFS's same-seed count "
    "+ 1 (REFS / R / HOLD: 0 of 200 each), head and REFS on the same episode count and onset pairs (else INVALID). "
    "Onsets of move holds the episode never reached (it ended first) are not stalls: that is a fall, G3's. Counting "
    "censored onsets as +inf in the median was the alternative: not adopted, it only binds once half the onsets "
    "stall. The same hole remains in p2s_cost (phase-2 code, not edited).",
    "G5 body height: p2_read.z2_height_vs_refs (suite stand AND walkstop stands, per seed, 0.03 m) -- the Z2 premise "
    "version -- not p2s_cost's es1/es2-mean suite height.",
    "VERDICT combination: an evaluated failing clause is decisive (FAIL) even when another clause is PENDING; "
    "p2_read's z2_stand_gates / p2r_rst_gate return PENDING in that case. A MISSING stamp is PENDING, never a FAIL. "
    "A head's COMPLETE stamp (p2_read.newest_stamp returns complete stamps only) whose read or statistic is None "
    "fails its clause closed, and a LINE source's (REFS / HOLD / R / Z1) complete stamp with no readable statistic "
    "makes the gate INVALID (the line cannot be drawn) -- on every gate (verifier fix: G4 read such a stamp PENDING). "
    "p2_read keys PENDING on the summary, not the stamp: a stamp whose summary is None reads PENDING there; here it "
    "does not, because a complete stamp with no statistic is a broken eval, not a pending one.",
    "C0R_s18: every gate computed and reported (no expectation); D6 names it only as G4's attribution floor.",
]


# ================================================================================================ markdown
def _fmt(v, nd=3):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{nd}g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return str(v)


def _cell_values(res: dict) -> str:
    parts = []
    for k, c in res["clauses"].items():
        tag = c["short"] or k
        sd = f" es{c['seed']}" if c["seed"] is not None else ""
        if c["state"] == "na":
            parts.append(f"{tag}{sd} n/a")
        elif c["state"] == "pending":
            parts.append(f"{tag}{sd} ?")
        elif not c["ok"]:
            parts.append(f"**{tag}{sd} {_fmt(c['value'], 4)} (line {_fmt(c['line'], 4)})**")
        else:
            parts.append(f"{tag}{sd} {_fmt(c['value'], 4)}")
    return "; ".join(parts)


def markdown(out: dict) -> str:
    rows = out["rows"]
    sm = out["summary"]
    L = ["# Stand network -- gate calibration (PLAN.md step 2)", "",
         f"Generated {out['meta']['generated']} by `scripts/sn_calib.py` (CPU, read-only; readers: phase-2 "
         f"`p2_read.py` sha256 {out['meta']['p2_read_sha256'][:12]}, `p2s_stand.py` "
         f"{out['meta']['p2s_stand_sha256'][:12]}). Every number below is from an existing reader; the D5 validity "
         "rule and every other departure from p2_read's rules is listed under Deviations.", "",
         f"**Lands as listed: {sm['lands_yes']} / {sm['expected_cells']} expected cells yes, {sm['mismatches']} "
         f"mismatch(es), {sm['pending_expected']} pending.**", ""]
    if out["mismatches"]:
        L += ["**Mismatches (PLAN.md step 2: STOP, notify, ask):**", ""] + [f"- {m}" for m in out["mismatches"]] + [""]
    notes = [f"{role}: {n}" for role, row in rows.items() for n in row.get("notes") or []]
    if notes:
        L += ["**Stamp resolution notes:**", ""] + [f"- {n}" for n in notes] + [""]
    L += ["## D6 gates: expected vs observed", "",
          "Values: es1/es2 suffix = env seed; bold = a failing clause (its line in brackets); ? = pending; "
          "n/a = does not apply.", "",
          "| gate | reference | expected | observed | lands as listed | values |", "|---|---|---|---|---|---|"]
    for gate in GATES:
        for role, row in rows.items():
            res = row["gates"][gate]
            obs = res["verdict"] + (f" (prov. {res['provisional']})" if res.get("provisional") else "")
            exp = res.get("expected") or "--"
            ld = res["lands"] + (f": {res['lands_why']}" if res.get("lands_why") and res["lands"] != "yes" else "")
            L.append(f"| {gate} {GATE_TITLE[gate]} | {role} | {exp} | {obs} | {ld} | {_cell_values(res)} |")
    idn = out.get("identity") or {}
    L += ["", "## Identity check: SN_RSTg_s18 vs P2R_RSTg_s18, field_hard es1, raw keys bitwise", "",
          f"**{idn.get('verdict')}** -- {'; '.join(idn.get('why') or [])}", ""]
    if idn.get("differing_keys"):
        L.append(f"- differing keys: {', '.join(idn['differing_keys'])}")
    if idn.get("first_mismatches"):
        L += [f"- `{x}`" for x in idn["first_mismatches"][:5]]
    modes = idn.get("modes") or {}
    for nm, m in modes.items():
        if m:
            cs = (idn.get("code_state") or {}).get(nm) or (None, None, None)
            L.append(f"- {nm}: launcher {str(m.get('launcher_sha256'))[:8]}, restance module "
                     f"{str(m.get('restance_module_sha256'))[:8]}, code_sha {str(cs[0])[:8]}, eval_sha "
                     f"{str(cs[1])[:8]}, dirty {cs[2]}; R knob {m.get('restance_knob')}")
    L += [f"- stamps: reference `{idn.get('reference')}`, re-run `{idn.get('rerun')}`"]
    if idn.get("bank") is not None:
        L += [f"- {bank_note(idn.get('bank'))} {bank_cause(idn.get('bank'))}"]
    ro = out["r_onset"]

    def _sp(s, k):
        return ((ro.get(f"es{s}") or {}).get("split") or {}).get(k)
    L += ["", "## R's walkstop onset (no earlier read had it)", "",
          "| | es1 | es2 | pooled |", "|---|---|---|---|",
          f"| R median onset (s) | {_fmt((ro.get('es1') or {}).get('onset_median_s'))} | "
          f"{_fmt((ro.get('es2') or {}).get('onset_median_s'))} | {_fmt(ro.get('pooled_median_s'))} |",
          f"| censored | {(ro.get('es1') or {}).get('onset_censored')} | {(ro.get('es2') or {}).get('onset_censored')}"
          " | |",
          f"| stalled (reached, censored) / reached | {_sp(1, 'stalled')} / {_sp(1, 'reached')} | "
          f"{_sp(2, 'stalled')} / {_sp(2, 'reached')} | |",
          f"| REFS / HOLD pooled (s) | | | {_fmt(ro.get('refs_pooled_median_s'))} / "
          f"{_fmt(ro.get('hold_pooled_median_s'))} |", "",
          f"Per pair: es1 {json.dumps((ro.get('es1') or {}).get('onset'))}; es2 "
          f"{json.dumps((ro.get('es2') or {}).get('onset'))}. {ro['statistic']}."]
    L += ["", "## Negative-control coverage (reported, no verdict changes)", "",
          "A gate no reference fails is unexercised as a negative control: the calibration shows its known-good "
          "pass but never shows its line can bind, so its sensitivity rests on the line definitions, not on an "
          "observed fail. D6 lists no known-bad for G3-G5, so this is not a mismatch.", ""]
    for gate in GATES:
        cv = (out.get("coverage") or {}).get(gate) or {}
        head = (f"**{gate} {GATE_TITLE[gate]}**: "
                + (f"failed by {', '.join(cv['failing_references'])}" if cv.get("failing_references")
                   else "**no reference fails -- unexercised as a negative control**")
                + (f" (D6 known-bad: {', '.join(cv['known_bad_listed'])})" if cv.get("known_bad_listed") else
                   " (D6 lists no known-bad)"))
        L.append(f"- {head}")
        if not cv.get("failing_references"):
            for kind, d in (cv.get("by_clause") or {}).items():
                t = d.get("tightest")
                L.append(f"  - {kind}: references read {_fmt(d['min'], 4)} .. {_fmt(d['max'], 4)} over {d['n']} "
                         "clause reads" + (f"; tightest pass {t['role']} {_fmt(t['value'], 4)} vs line "
                                           f"{_fmt(t['line'], 4)} (margin {_fmt(t['margin'], 3)})" if t else ""))
    L += ["", "Margins are |line - value| on passing clauses. A margin of 0 at value 0 means different things: "
          "`suite falls` is `== 0` (one fall fails), but the G4 stand-fall lines are HOLD + 2 SE of the POOLED "
          "difference, which is 0 only while head and HOLD both read 0 falls and moves with the head's own count "
          "(one fall in 100 vs HOLD's 0 draws a line of about 0.02 and passes), so a 0 margin there is not a tight pass."]
    L += ["", "## Pending", ""] + ([f"- {p}" for p in out["pending"]] or ["- none"])
    L += ["", "## INVALID cells", ""] + ([f"- {x}" for x in out.get("invalid_cells") or []] or ["- none"])
    L += ["", "## Plan's listed numbers vs computed (reported)", "", "| what | listed | computed |", "|---|---|---|"]
    L += [f"| {x['what']} | {x['listed']} | {', '.join(_fmt(v, 4) for v in x['computed'])} |"
          for x in out["listed_numbers"]]
    L += ["", "## Reported only (not gated)", "",
          "| reference | seed | six-down suite / walkstop | hip yaw mean / max walkstop (deg) | settle median walkstop (s) "
          "| KS suite / walkstop / post-settle |", "|---|---|---|---|---|---|"]
    for role, row in rows.items():
        for s in SEEDS:
            r = row["reported"][f"es{s}"]
            L.append(f"| {role} | es{s} | {_fmt(r['six_down suite (steady)'])} / {_fmt(r['six_down walkstop (last 5 s)'])} "
                     f"| {_fmt(r['hip_yaw_mean_deg walkstop'])} / {_fmt(r['hip_yaw_max_deg walkstop'])} | "
                     f"{_fmt(r['settle_median_s walkstop'])} | {_fmt(r['KS suite stand'])} / "
                     f"{_fmt(r['KS walkstop stands (pooled median)'])} / {_fmt(r['post-settle KS walkstop'])} |")
    L += ["", "## Validity", ""]
    for role, row in rows.items():
        bad = {k: v for k, v in (row.get("validity") or {}).items() if v}
        byp = row.get("bypassed_p2_read_reasons") or {}
        L.append(f"- **{role}** ({row['name']}, rule `{row['kind']}`): "
                 + ("valid on every stamp read" if not bad else f"INVALID: {json.dumps(bad)[:400]}")
                 + (f"; p2_read Z2-env reasons bypassed on {len(byp)} stamp(s) (D5), e.g. "
                    f"`{next(iter(byp.values()))[0]}`" if byp else ""))
        for n in row.get("notes") or []:
            L.append(f"  - note: {n}")
        if row.get("fh_source"):
            L.append(f"  - field_hard source: {json.dumps(row['fh_source'])}")
    L += ["", "## Deviations from p2_read's rules", ""] + [f"{i}. {d}" for i, d in enumerate(out["deviations"], 1)]
    L += ["", "Full per-clause values, lines, stamp paths and code states: `calib.json`.", ""]
    return "\n".join(L)


# ================================================================================================ main
def _sha(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def run(json_path: Path, md_path: Path, jobs: int, heads=()) -> dict:
    tasks = {role: (role, NAMES[role], KIND[role]) for role in tuple(ROLES) + ("R_SN",)}
    for h in heads:
        tasks[h] = (h, h, "stand_net")
    t0 = time.time()
    reads, ident = {}, None
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {role: ex.submit(read_reference, *a) for role, a in tasks.items()}
            fid = ex.submit(identity_check)
            reads = {role: f.result() for role, f in futs.items()}
            ident = fid.result()
    else:
        reads = {role: read_reference(*a) for role, a in tasks.items()}
        ident = identity_check()
    reads["R"] = merge_r(reads["R"], reads.pop("R_SN"))
    out = calibrate(reads, ident, heads=tuple(heads))
    log_tail = SN_LOG.read_text().splitlines()[-6:] if SN_LOG.is_file() else []
    out["meta"] = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "tool": str(Path(__file__).resolve()), "tool_sha256": _sha(Path(__file__).resolve()),
                   "p2_read_sha256": _sha(P2_SCRIPTS / "p2_read.py"),
                   "p2s_stand_sha256": _sha(P2_SCRIPTS / "p2s_stand.py"),
                   "p2_out": str(P.OUT), "names": NAMES, "read_seconds": round(time.time() - t0, 1),
                   "per_reference_read_seconds": {r: reads[r].get("read_s") for r in reads if r in ROLES},
                   "sn_calib_log_tail": log_tail, "expectations": {f"{g} {r}": v for (g, r), v in EXPECT.items()}}
    out["deviations"] = DEVIATIONS
    json_path.write_text(json.dumps(_jsonable(out), indent=1, default=str))
    md_path.write_text(markdown(out))
    return out


# ================================================================================================ self-test
def _syn_still(steps=0.0, taps=0.0, clear=0.0, og=0.0, hover=0.0, ks=0.10, height=1.065, falls=0, n=100):
    return {"steps_per_s": steps, "taps_per_s": taps, "clearance_m": clear, "off_ground_share": og,
            "hover_episodes_per_100": hover, "ks_m_s": ks, "body_height_m": height, "falls": falls, "n": n}


def _syn_z2ws(homed=0.97, settled=0.97, settle=2.9, post=(0.11, 0.0, 0.15), falls300=0.0, height=1.062, falls=None):
    # raw count (G3, CHARTER 2026-10-12): default = the rate's count on a full 300-stand set
    falls = int(round(falls300)) if falls is None else falls
    return {"homed_share": homed, "settled_share": settled, "settle_median_s": settle,
            "post_settle_steps_per_s": post[0], "post_settle_taps_per_s": post[1], "post_settle_ks_m_s": post[2],
            "falls_per_300": falls300, "falls": falls, "body_height_m": height, "six_down_stand_share": 0.3,
            "hip_yaw_mean_deg": 0.3, "hip_yaw_max_deg": 2.0}


def _syn_audit(exits=200, eps=100, per=2, extra=0, exc=0, unexc=0):
    return {"episodes": eps, "design_exits_per_episode": per, "exits": exits, "extra": extra, "missing_excused": exc,
            "missing_unexcused": unexc, "episodes_with_stand_mode_fall": 1 if exc else 0, "design_exits": eps * per}


def _syn_reads(role, *, still=None, z2ws=None, onset=0.14, exit_=("ok", 0), fh=(0.81, 0.81, 0.81, 0.81),
               fh_stand_falls=(0, 0, 0, 0), ft=0.99, ft_stand_falls=0, height_ws=1.062, kind="obs0",
               fh_partial=False, validity=None, drop=(), audit=None, stalled=0, split_eps=100):
    still = still if still is not None else {1: _syn_still(), 2: _syn_still()}
    z2ws = z2ws if z2ws is not None else {1: _syn_z2ws(height=height_ws), 2: _syn_z2ws(height=height_ws)}
    seeds = (1,) if fh_partial else FH_SEEDS
    st = {f"suite@{s}": f"/syn/{role}/suite{s}" for s in SEEDS}
    st.update({f"{WS}@{s}": f"/syn/{role}/ws{s}" for s in SEEDS})
    st[FT1] = f"/syn/{role}/ft1"
    st.update({f"{FH}@{s}": f"/syn/{role}/fh{s}" for s in seeds})
    field = {FT1: {"stand": {"falls": ft_stand_falls, "n": 100}, "comp": {"rate": ft, "n": 100}}}
    for i, s in enumerate(seeds):
        field[f"{FH}@{s}"] = {"stand": {"falls": fh_stand_falls[i], "n": 100}, "comp": {"rate": fh[i], "n": 100}}
    ex = {}
    for s in SEEDS:
        stt, cnt = exit_
        au = audit or _syn_audit()
        ex[s] = ({"status": stt, "count": cnt, "exits": au["exits"], "n_raw_episodes": au["episodes"],
                  "audit": au} if stt == "ok" else {"status": stt, "count": None, "n_raw_episodes": 100})
    ws = {s: {"stamp": f"/syn/{role}/ws{s}", "onset_median_s": onset, "onset_values": [onset] * (200 - stalled),
              "onset_censored": stalled, "n_episodes": split_eps, "pooled": {"ks_m_s": 0.2}} for s in SEEDS}
    osp = {s: {"n_episodes": split_eps, "pairs_per_episode": 2, "reached": 2 * split_eps, "stalled": stalled,
               "unreached": 0} for s in SEEDS}
    rd = {"role": role, "name": f"SYN_{role}", "kind": kind, "stamps": st, "notes": [],
          "z2": {"suite": {1: {"six_down_stand_share": 0.3}, 2: {"six_down_stand_share": 0.3}}, "walkstop": z2ws},
          "still": still, "field": field, "walkstop": ws, "exit": ex, "onset_split": osp, "p2r_mode": {},
          "code_state": {}, "validity": validity or {k: [] for k in st}, "bypassed": {}, "fh_seeds": list(seeds),
          "fh_source": {"partial": True, "seeds": [1]} if fh_partial else None}
    for k in drop:                                   # a missing stamp: no stamp, no read
        rd["stamps"][k] = None
        rd["field"].pop(k, None)
        if k.startswith("suite@"):
            rd["still"] = {s: v for s, v in rd["still"].items() if f"suite@{s}" != k}
        if k.startswith(f"{WS}@"):
            s = int(k.rsplit("@", 1)[1])
            rd["z2"]["walkstop"] = {x: v for x, v in rd["z2"]["walkstop"].items() if x != s}
            rd["walkstop"] = {x: v for x, v in rd["walkstop"].items() if x != s}
            rd["onset_split"] = {x: v for x, v in rd["onset_split"].items() if x != s}
            rd["exit"][s] = {"status": "missing", "count": None}
    return rd


def _syn_world(**over):
    """A synthetic reference set that lands exactly as PLAN.md D6 lists (``over`` replaces a role's reads)."""
    march = {1: _syn_still(steps=2.83, taps=2.8, clear=0.135, og=0.34, ks=1.72, height=1.068),
             2: _syn_still(steps=2.83, taps=2.7, clear=0.135, og=0.34, ks=1.72, height=1.068)}
    unhomed = {s: _syn_z2ws(homed=0.0, settled=0.0, settle=None, post=(None, None, None), height=1.068)
               for s in SEEDS}
    w = {"REFS": _syn_reads("REFS", still=march, z2ws=unhomed, onset=0.14, exit_=("no_mode", None),
                            fh=(0.8125,) * 4, fh_stand_falls=(0, 0, 0, 0)),
         "HOLD": _syn_reads("HOLD", z2ws={s: _syn_z2ws(homed=0.0, settled=0.0, settle=None, post=(None, None, None))
                                          for s in SEEDS}, onset=0.12, fh=(0.81,) * 4, kind="tier1"),
         "R": _syn_reads("R", onset=0.12, fh=(0.81,) * 4, ft=1.0, kind="rst"),
         "Z1": _syn_reads("Z1", z2ws={s: _syn_z2ws(homed=0.0, settled=0.0, settle=None, post=(None, None, None))
                                      for s in SEEDS}, exit_=("no_mode", None), kind="z"),
         "Z2d": _syn_reads("Z2d", z2ws={s: _syn_z2ws(homed=0.02, settled=0.18, settle=3.5) for s in SEEDS},
                           exit_=("no_mode", None), kind="z"),
         "ST2M": _syn_reads("ST2M", still={1: _syn_still(steps=0.17, taps=16.1), 2: _syn_still(steps=0.17, taps=15.3)},
                            kind="tier1"),
         "C0R": _syn_reads("C0R", still=march, exit_=("no_mode", None), fh=(0.785,) * 4)}
    w.update(over)
    return w


def _self_test() -> int:
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)

    # 1. the combiner
    c = {"a": _cl(1, 2, True), "b": _cl(None, None, None, "pending"), "c": _cl(3, 2, False)}
    check(verdict_of(c, [])["verdict"] == FAIL, "combiner: a failing clause is decisive over a pending one")
    check(verdict_of({"a": _cl(1, 2, True), "b": _cl(None, None, None, "pending")}, [])["verdict"] == PENDING,
          "combiner: pending")
    check(verdict_of({"a": _cl(1, 2, True)}, ["x"])["verdict"] == INVALID, "combiner: invalid dominates")
    v = verdict_of({"a": _cl(1, 2, True), "p": _cl(0.7, 0.75, False, partial=True)}, [])
    check(v["verdict"] == PARTIAL and v["provisional"] == FAIL, "combiner: partial carries provisional FAIL")
    check(verdict_of({"n": _cl(None, None, None, "na")}, [])["verdict"] == PENDING, "combiner: nothing evaluable")
    check(verdict_of({"a": _cl(1, 2, True), "n": _cl(None, None, None, "na")}, [])["verdict"] == PASS,
          "combiner: n/a does not block a pass")

    # 2. the synthetic world lands exactly as listed
    out = calibrate(_syn_world())
    check(out["mismatches"] == [], f"synthetic world: no mismatches (got {out['mismatches']})")
    check(out["summary"]["lands_yes"] == len(EXPECT), f"synthetic world: every expected cell lands "
          f"({out['summary']['lands_yes']}/{len(EXPECT)})")
    rows = out["rows"]
    check(rows["REFS"]["gates"]["G3"]["clauses"]["walkstop exit-handover falls within 2 s es1 <= max(1, R's)"]["state"]
          == "na", "G3: REFS exit clause n/a (no p2r_mode)")
    check(abs(rows["R"]["gates"]["G2"]["clauses"]["median settle walkstop es1 <= probe + 1.0 s"]["line"] - 3.9)
          < 1e-9, "G2: R's settle line = its own median + 1 s")
    check(rows["REFS"]["gates"]["G1"]["failed"] == [f"{k} suite es{s}" for s in SEEDS for k in
                                                   ("steps_per_s", "clearance_m", "off_ground_share", "taps_per_s",
                                                    "ks_m_s")], "G1: REFS fails steps, clearance, off-ground, taps, KS")
    check(all(r["gates"]["G1"]["z2_stand_gates_still_agrees"] for r in rows.values()),
          "G1: z2_stand_gates' STILL clauses agree with still_clauses")

    # 3. mismatch detection
    bad = calibrate(_syn_world(HOLD=_syn_reads("HOLD", onset=0.12, kind="tier1")))   # HOLD homes -> G2 PASS
    check(any(m.startswith("G2 HOMED x HOLD") for m in bad["mismatches"]), "mismatch: HOLD passing G2 is flagged")
    st2 = _syn_reads("ST2M", still={1: _syn_still(steps=0.9, taps=0.5), 2: _syn_still(steps=0.9, taps=0.5)},
                     kind="tier1")
    bad = calibrate(_syn_world(ST2M=st2))
    check(any("named clause taps_per_s" in m for m in bad["mismatches"]),
          "mismatch: ST2M failing G1 on steps, not its named taps clause, is flagged")

    # 4. G1 pending / decisive
    g = gate_g1(_syn_reads("X", drop=("suite@2",)))
    check(g["verdict"] == PENDING, "G1: a missing es2 suite stamp is PENDING")
    g = gate_g1(_syn_reads("X", still={1: _syn_still(steps=2.0)}, drop=("suite@2",)))
    check(g["verdict"] == FAIL, "G1: es1 failing is decisive with es2 missing")
    g = gate_g1(_syn_reads("X", validity={"suite@1": ["bad env"]}))
    check(g["verdict"] == INVALID, "G1: an invalid suite stamp makes the gate INVALID")
    h = _syn_reads("X")
    h["still"].pop(2)
    check(gate_g1(h)["verdict"] == FAIL, "G1: a complete suite es2 stamp with no stand summary fails closed")

    # 5. G3 exit clause
    w = _syn_world()
    probe, z1 = w["R"], w["Z1"]
    hd = _syn_reads("H", exit_=("no_raw", None), kind="tier1")
    g = gate_g3(hd, z2_parts(hd, probe, z1), z1, probe, "tier1")
    check(g["verdict"] == INVALID, "G3: p2r_mode present but no p2r_stand_mode raw -> INVALID")
    hd = _syn_reads("H", exit_=("no_mode", None), kind="stand_net")
    g = gate_g3(hd, z2_parts(hd, probe, z1), z1, probe, "stand_net")
    check(g["verdict"] == INVALID, "G3: a stand_net head without p2r_mode is INVALID, never n/a")
    hd = _syn_reads("H", exit_=("ok", 2))
    g = gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)
    check(g["verdict"] == FAIL, "G3: 2 exit falls vs R's 0 (line max(1, 0) = 1) fails")
    hd = _syn_reads("H", exit_=("ok", 1))
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == PASS, "G3: 1 exit fall passes the floor")
    r3 = _syn_reads("R", exit_=("ok", 3), kind="rst")
    hd = _syn_reads("H", exit_=("ok", 3))
    check(gate_g3(hd, z2_parts(hd, r3, z1), z1, r3)["verdict"] == PASS, "G3: line = max(1, R's count)")
    hd = _syn_reads("H", z2ws={s: _syn_z2ws(falls300=2.0) for s in SEEDS})
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == FAIL,
          "G3: walkstop falls 2/300 > max(1, Z1's 0)")
    hd = _syn_reads("H", z2ws={s: _syn_z2ws(falls300=1.00671, falls=1) for s in SEEDS})
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == PASS,
          "G3 (CHARTER 2026-10-12): 1 walkstop fall that cut an episode (rate 300/298 = 1.0067) passes the count line 1")
    hd = _syn_reads("H", still={1: _syn_still(falls=1), 2: _syn_still()})
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == FAIL, "G3: one suite stand fall fails")

    au = _syn_audit(exits=20, eps=10)
    hd = _syn_reads("H", exit_=("ok", 1), audit=au)
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == INVALID,
          "G3: 1 exit fall in 20 exits (10 episodes) vs R's 200 is INVALID, not a lenient pass")
    hd = _syn_reads("H", exit_=("ok", 0), audit=_syn_audit(exits=198, unexc=2))
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == FAIL,
          "G3: 2 scheduled exits never made (no stand fall) fail closed")
    hd = _syn_reads("H", exit_=("ok", 0), audit=_syn_audit(exits=198, exc=2))
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == PASS,
          "G3: exits lost to a stand-mode fall are the stand-falls clause's (inside its 1/300 floor)")
    hd = _syn_reads("H", exit_=("ok", 0), audit=_syn_audit(exits=203, extra=3))
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == INVALID,
          "G3: exits beyond the schedule break the D5 switch contract -> INVALID")
    rbad = _syn_reads("R", exit_=("ok", 0), kind="rst", audit=_syn_audit(exits=190, unexc=10))
    hd = _syn_reads("H", exit_=("ok", 0))
    check(gate_g3(hd, z2_parts(hd, rbad, z1), z1, rbad)["verdict"] == INVALID,
          "G3: R short of its scheduled exits -> its raw-count line is not on the full set -> INVALID")
    hd = _syn_reads("H")
    hd["still"].pop(2)
    check(gate_g3(hd, z2_parts(hd, probe, z1), z1, probe)["verdict"] == FAIL,
          "G3: a complete suite es2 stamp with no stand summary fails closed")

    # 6. G4
    refs, hold = w["REFS"], w["HOLD"]
    check(gate_g4(_syn_reads("H", fh=(0.70,) * 4), refs, hold)["verdict"] == FAIL, "G4: field_hard 0.70 vs 0.8125")
    check(gate_g4(_syn_reads("H", fh=(0.79,) * 4), refs, hold)["verdict"] == PASS, "G4: field_hard 0.79 within 2 SE")
    g = gate_g4(_syn_reads("R", fh=(0.81,), fh_partial=True, ft=1.0), refs, hold)
    check(g["verdict"] == PARTIAL and g["provisional"] == PASS, "G4: R from es1 only reads PARTIAL (provisional PASS)")
    g = gate_g4(_syn_reads("R", fh=(0.81,), fh_partial=True, ft=0.85), refs, hold)
    check(g["verdict"] == FAIL, "G4: a field_turn failure is not hidden by PARTIAL")
    check(gate_g4(_syn_reads("H", ft=0.93), refs, hold)["verdict"] == FAIL, "G4: field_turn 0.93 < 0.99 - 0.05")
    check(gate_g4(_syn_reads("H", ft=0.95), refs, hold)["verdict"] == PASS, "G4: field_turn 0.95 >= 0.94")
    check(gate_g4(_syn_reads("H", ft_stand_falls=5), refs, hold)["verdict"] == FAIL,
          "G4: 5/100 field_turn stand falls vs HOLD 0/100 (pooled 2 SE(diff) 0.044)")
    check(gate_g4(_syn_reads("H", fh_stand_falls=(1, 0, 1, 0)), refs, hold)["verdict"] == PASS,
          "G4: 2/400 field_hard stand falls vs HOLD 0/400 within 2 SE(diff)")
    check(gate_g4(_syn_reads("H", drop=(f"{FH}@4",)), refs, hold)["verdict"] == PENDING,
          "G4: a missing field_hard es4 row is PENDING")
    check(gate_g4(_syn_reads("H"), refs, _syn_reads("HOLD", validity={f"{FH}@2": ["x"]}))["verdict"] == INVALID,
          "G4: an invalid HOLD line source makes the gate INVALID")

    h = _syn_reads("H")
    h["field"][FT1]["comp"] = None
    check(gate_g4(h, refs, hold)["verdict"] == FAIL, "G4: a complete field_turn stamp with no completion fails closed")
    h = _syn_reads("H")
    h["field"][FT1]["comp"] = {"rate": None, "n": 100}
    check(gate_g4(h, refs, hold)["verdict"] == FAIL, "G4: a present completion read with rate None fails closed")
    h = _syn_reads("H")
    h["field"][f"{FH}@3"]["comp"] = None
    check(gate_g4(h, refs, hold)["verdict"] == FAIL, "G4: a complete field_hard es3 stamp with no completion fails")
    r4 = _syn_reads("REFS", exit_=("no_mode", None), fh=(0.8125,) * 4)
    r4["field"][FT1]["comp"] = None
    check(gate_g4(_syn_reads("H"), r4, hold)["verdict"] == INVALID,
          "G4: REFS's complete field_turn stamp with no completion -> line unreadable -> INVALID")
    h4 = _syn_reads("HOLD", kind="tier1")
    h4["field"][FT1]["stand"] = None
    check(gate_g4(_syn_reads("H"), refs, h4)["verdict"] == INVALID,
          "G4: HOLD's complete field_turn stamp with no stand summary -> stand-fall line unreadable -> INVALID")
    h = _syn_reads("H")
    h["field"][FT1]["stand"] = None
    check(gate_g4(h, refs, hold)["verdict"] == FAIL, "G4: the head's field_turn stand summary None fails closed")

    # 7. G5
    check(gate_g5(_syn_reads("H", onset=0.40), refs)["verdict"] == PASS, "G5: onset 0.40 <= 0.14 + 0.3")
    check(gate_g5(_syn_reads("H", onset=0.50), refs)["verdict"] == FAIL, "G5: onset 0.50 > 0.44")
    h = _syn_reads("H")
    h["walkstop"][2]["onset_median_s"] = None
    check(gate_g5(h, refs)["verdict"] == FAIL, "G5: a present walkstop read with every onset censored fails closed")
    check(gate_g5(_syn_reads("H", height_ws=1.02), refs)["verdict"] == FAIL, "G5: walkstop body height 0.04 m off")
    check(gate_g5(_syn_reads("H", drop=(f"{WS}@2",)), refs)["verdict"] == PENDING, "G5: missing walkstop es2")
    g = gate_g5(_syn_reads("H", onset=0.12, stalled=180), refs)        # the verifier's probe
    check(g["verdict"] == FAIL and not g["clauses"]["walkstop onset stalls es1 <= REFS + 1"]["ok"]
          and g["clauses"][f"walkstop onset median es1 <= REFS + max({P.P2S_ONSET_FLOOR}, band)"]["ok"],
          "G5: 180 of 200 onsets stalled, 0.12 s on the rest -> FAIL on the stall clause (the median alone passed)")
    check(gate_g5(_syn_reads("H", stalled=1), refs)["verdict"] == PASS, "G5: 1 stall vs REFS 0 is inside the +1 floor")
    check(gate_g5(_syn_reads("H", stalled=2), refs)["verdict"] == FAIL, "G5: 2 stalls vs REFS 0 fail")
    check(gate_g5(_syn_reads("H", split_eps=50), refs)["verdict"] == INVALID,
          "G5: a stall count on 50 episodes vs REFS's 100 is INVALID (raw count, other denominator)")
    h = _syn_reads("H")
    h["walkstop"].pop(2)
    h["onset_split"].pop(2)
    check(gate_g5(h, refs)["verdict"] == FAIL, "G5: a complete walkstop stamp with no walkstop read fails closed")
    r5 = _syn_reads("REFS", onset=0.14, exit_=("no_mode", None))
    r5["walkstop"][1]["onset_median_s"] = None
    check(gate_g5(_syn_reads("H"), r5)["verdict"] == INVALID,
          "G5: REFS's complete walkstop stamp with no onset median makes the line unreadable -> INVALID")

    # 8. D5 validity (pure)
    sha = "a" * 64
    ok_env = {"KRABBY_OMNI_STAND_OBS": "0"}
    mode = {"variant": "stand_net", "s1_checkpoint_sha256": sha, "specialist": {"path": "/x.pt", "sha256": "c" * 64},
            "stand_mask": 1}
    check(stand_net_reasons("H", "k", ok_env, mode, sha) == [],
          "D5: STAND_OBS=0 + variant stand_net on REFS's S1 + stand ckpt sha + mask is valid")
    check(stand_net_reasons("H", "k", {"KRABBY_OMNI_STAND_OBS": "clock"}, mode, sha) != [], "D5: clock is INVALID")
    check(stand_net_reasons("H", "k", {}, mode, sha) != [], "D5: STAND_OBS absent fails closed")
    check(stand_net_reasons("H", "k", dict(ok_env, KRABBY_OMNI_STAND_MASK="1"), mode, sha) != [],
          "D5: the env-level stand mask is INVALID")
    check(stand_net_reasons("H", "k", ok_env, dict(mode, variant="restance"), sha) != [],
          "D5: another p2r_mode variant is INVALID")
    check(stand_net_reasons("H", "k", ok_env, dict(mode, s1_checkpoint_sha256="b" * 64), sha) != [],
          "D5: a different S1 is INVALID")
    check(stand_net_reasons("H", "k", ok_env, None, sha) != [], "D5: no p2r_mode is INVALID")
    check(stand_net_reasons("H", "k", ok_env, mode, None) != [], "D5: an unknown REFS sha fails closed")
    check(stand_net_reasons("H", "k", ok_env, {k: v for k, v in mode.items() if k != "specialist"}, sha) != [],
          "D5: no stand checkpoint sha recorded fails closed")
    check(stand_net_reasons("H", "k", ok_env, {k: v for k, v in mode.items() if k != "stand_mask"}, sha) != [],
          "D5: no mask flag recorded fails closed")
    check(stand_net_reasons("H", "k", ok_env, dict(mode, stand_mask=True), sha) != [], "D5: a bool mask is malformed")
    check(stand_net_arm(dict(mode, stand_mask="0")) == ("c" * 64, 0), "D5: mask '0' reads 0")
    cons = stand_net_consistency("H", {"suite@1": mode, "suite@2": mode, f"{WS}@1": mode})
    check(all(v == [] for v in cons.values()), "D5: one sha + one mask across a head's stamps is consistent")
    cons = stand_net_consistency("H", {"suite@1": mode, f"{WS}@1": dict(mode, stand_mask=0)})
    check(all(v for v in cons.values()), "D5: a head mixing A-mask and A-full stamps is INVALID on every stamp")
    cons = stand_net_consistency("H", {"suite@1": mode, f"{WS}@1": dict(mode, specialist={"sha256": "d" * 64})})
    check(all(v for v in cons.values()), "D5: a head mixing two stand checkpoints is INVALID on every stamp")
    check(obs0_reasons("REFS", "k", {"KRABBY_OMNI_STAND_OBS": "0"}) == [], "obs0: REFS-like env is valid")
    check(obs0_reasons("REFS", "k", {"KRABBY_OMNI_STAND_OBS": "clock"}) != [], "obs0: clock is not obs parity")
    check(obs0_reasons("REFS", "k", {"KRABBY_OMNI_STAND_OBS": "0", "KRABBY_OMNI_STAND_PROXY": "hold"}) != [],
          "obs0: the held proxy is INVALID")

    # 9. R composite + identity verdict
    base = _syn_reads("R", fh=(0.81,), fh_partial=True, kind="rst")
    base["name"] = "P2R_RSTg_s18"
    sn = _syn_reads("R_SN", fh=(0.80, 0.82, 0.79, 0.81), kind="rst")
    sn["name"] = "SN_RSTg_s18"
    m = merge_r(base, sn)
    check(m["fh_seeds"] == list(FH_SEEDS) and not m["fh_source"]["partial"]
          and m["stamps"][f"{FH}@3"] == "/syn/R_SN/fh3", "merge_r: SN es1-4 complete -> used")
    sn3 = _syn_reads("R_SN", drop=(f"{FH}@4",), kind="rst")
    m = merge_r(base, sn3)
    check(m["fh_seeds"] == [1] and m["fh_source"]["partial"] and f"{FH}@2" not in m["stamps"]
          and m["stamps"][f"{FH}@1"] == "/syn/R/fh1", "merge_r: SN incomplete -> P2R_RSTg es1, PARTIAL")
    iv = identity_verdict([], 100, 100)
    check(iv["verdict"] == PASS, "identity: an empty diff on 100/100 episodes passes")
    iv = identity_verdict(["episode_3 actions: not bitwise equal to REFS (shapes (5,) vs (5,))",
                           "episode_4: key 'p2r_branch' (in REFS) missing from the probe stamp"], 100, 100)
    check(iv["verdict"] == FAIL and iv["differing_keys"] == ["actions", "p2r_branch"], "identity: keys parsed")
    check(identity_verdict([], 0, 0)["verdict"] == FAIL, "identity: no episodes is not a pass")

    # 10. lands
    res = {"verdict": PASS, "clauses": {}}
    check(lands("G1", "R", res)[0] == "yes" and lands("G1", "REFS", res)[0] == "NO", "lands: yes / NO")
    check(lands("G1", "R", {"verdict": INVALID})[0] == "NO", "lands: INVALID on a listed cell is a mismatch")
    check(lands("G1", "R", {"verdict": PENDING})[0] == "pending", "lands: PENDING")
    check(lands("G4", "R", {"verdict": PARTIAL, "provisional": PASS})[0] == "pending", "lands: PARTIAL is pending")
    check(lands("G1", "C0R", res)[0] == "reported", "lands: no expectation -> reported")

    # 10b. negative-control coverage + the identity bank note
    cov = calibrate(_syn_world())["coverage"]
    check([g for g in GATES if not cov[g]["exercised_as_negative_control"]] == ["G3", "G4", "G5"],
          f"coverage: G3-G5 are unexercised in the synthetic world (got {[(g, cov[g]['failing_references']) for g in GATES]})")
    t = cov["G4"]["by_clause"]["field_hard"]["tightest"]
    check(t["role"] == "C0R" and abs(t["margin"] - (t["value"] - t["line"])) < 1e-12,
          "coverage: the tightest field_hard pass is C0R's")
    bk = {"reference": {"rows": 47500, "seen": 47500, "bank_max_logged": [100000]},
          "rerun": {"rows": 20000, "seen": 47500, "bank_max_logged": [20000]}, "launcher_default_bank_max": 20000,
          "subset": True, "rows_outside_reference": 0, "keys_equal": {k: True for k in BANK_KEYS}}
    check("DIFFER" in bank_note(bk) and "a subset of" in bank_note(bk) and "bank CAP" in bank_cause(bk)
          and "No later step" in bank_cause(bk), "identity: the bank note names the size difference and its cause")
    check("same size" in bank_note(dict(bk, rerun=bk["reference"])), "identity: same-size banks read same size")

    # 11. markdown renders
    out = calibrate(_syn_world(), identity={"verdict": PENDING, "why": ["stamp missing"]})
    out["meta"] = {"generated": "selftest", "p2_read_sha256": "x" * 12, "p2s_stand_sha256": "y" * 12}
    out["deviations"] = DEVIATIONS
    md = markdown(out)
    check("| G2 HOMED | HOLD | FAIL | FAIL | yes |" in md, "markdown: the D6 table renders")
    check(any(p.startswith("identity check: PENDING") for p in out["pending"]), "pending: identity PENDING listed")

    for f in fails:
        print(f"SELFTEST FAIL: {f}")
    print(f"sn_calib selftest: {'PASS' if not fails else 'FAIL'} ({len(fails)} failing check(s))")
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, default=CAMPAIGN / "calib.json")
    ap.add_argument("--md", type=Path, default=CAMPAIGN / "calib.md")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--head", action="append", default=[], metavar="NAME",
                    help="a stand-network head (D5 validity rule, no expectations)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _self_test()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    out = run(a.json, a.md, max(1, a.jobs), heads=a.head)
    sm = out["summary"]
    print(f"sn_calib: {sm['lands_yes']}/{sm['expected_cells']} expected cells land, {sm['mismatches']} mismatch(es), "
          f"{sm['pending_expected']} pending; identity {(out.get('identity') or {}).get('verdict')}; "
          f"wrote {a.json} and {a.md}")
    for m in out["mismatches"]:
        print(f"MISMATCH: {m}")
    for p in out["pending"]:
        print(f"PENDING: {p}")
    return 4 if out["mismatches"] else 2 if out["pending"] else 0


if __name__ == "__main__":
    sys.exit(main())
