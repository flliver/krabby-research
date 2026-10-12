#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 3 -- the identity-bridge and swap-live read of ``stand_net`` stamps (CPU only,
read-only; never edits a phase-2 file, writes no bytecode).

    sn_bridge_read.py --head NAME [--refs REFS_S1_s18] [--rows ROW@ES,...] (--identity | --live) [--json PATH]
    sn_bridge_read.py --refs-stamp DIR --sn-stamp DIR (--identity | --live) [--json PATH]

Default rows (the step-3 identity bridge): ``suite__omni@1,walkstop__omni@1,field_hard__omni@1``; a stamp is the
newest COMPLETE one under ``$P2_OUT/lr/<NAME>/<row>/seed00<es>/`` (``p2_read.newest_stamp``).

Per row, from the stamps alone:

* **meta** (the head-validity contract the launcher writes; ``sn_common`` names): ``run_meta.p2r_mode.variant ==
  "stand_net"``; ``stand_mask`` an int 0 / 1 (a bool is malformed); ``specialist.sha256`` 64 lowercase hex;
  ``s1_checkpoint_sha256`` == the REFS stamp's ``checkpoint_sha256``; the ``stand_net`` block with
  ``estimator_equals_s1`` true, ``r_knob`` == the D3 string and the trained-arm record (``stand_ckpt_kind``
  ``sn_dagger`` with ``trained_stand_mask`` == ``stand_mask``, or ``s1_identity`` with the specialist sha == S1's);
  ``env_vars`` pass ``sn_common.env_reasons`` (D5); identity: the stand checkpoint IS S1 (specialist sha == S1 sha)
  and the mask is 0.
* **swap-live** (the restance proof's twin, P2L ``restance_swap_live``): (a) on every ``p2r_stand_mode`` step the
  applied ``actions`` == ``sn_stand_action`` bitwise, ``sn_r_branch`` >= 0; off stand ``sn_stand_action`` /
  ``sn_s1_action`` / ``sn_r_target`` are 0 and ``sn_r_branch`` -1; (b) stand-mode share > 0; (c) ``actions`` before
  each episode's first stand entry == REFS's same episode, bitwise (no leak into walking rows; non-vacuous on walkstop,
  vacuous on the suite, which enters stand at its settle, step 0 -- the checked-episode count is reported);
  (d) ``--identity``: ``sn_stand_action == sn_s1_action`` on every stand step and the run_meta differ count is 0;
  ``--live``: some stand step's ``sn_stand_action`` differs from ``sn_s1_action``.
* **identity bridge** (``--identity``): ``p2_read.p2r_bitwise_diff(REFS, SN)`` -- every raw key REFS has, every
  episode, ``np.array_equal`` -- empty -> BITWISE. On a miss, the CHARTER section 3 rule (stated 2026-10-12, before
  the run): PASS-with-note only if actions differ <= 1e-5 AND every G1-G5 statistic is identical, else STOP and ask.
  This reader decides the first clause only (``max |actions - REFS actions|``): MISS_SMALL (<= 1e-5; the G1-G5
  comparison is then the orchestrator's) or MISS (STOP and ask). The G1-G5 comparison MUST name its own outputs --
  sn_calib's defaults are the committed step-2 record (``calib.json`` / ``calib.md``) and would be overwritten:

      sn_calib.py --head SN_ID_s18 --json <campaign>/bridge_calib.json --md <campaign>/bridge_calib.md

Exit: 0 every row PASS (identity: BITWISE); 3 an identity MISS_SMALL (no FAIL); 4 a FAIL / MISS; 2 a stamp missing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # the phase-2 scripts are imported read-only: never write their __pycache__

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
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


SN = _load(HERE / "sn_common.py", "_sn_common_bridge_read")

REFS_NAME = "REFS_S1_s18"
DEFAULT_ROWS = ("suite__omni@1", "walkstop__omni@1", "field_hard__omni@1")
MISS_ATOL = 1e-5
PASS, FAIL, BITWISE, MISS_SMALL, MISS, MISSING = "PASS", "FAIL", "BITWISE", "MISS_SMALL", "MISS", "MISSING"
#: the miss rule's G1-G5 comparison: OWN output files (sn_calib's defaults are the committed step-2 calib.json / .md)
CALIB_BRIDGE_CMD = (f"{HERE / 'sn_calib.py'} --head SN_ID_s18 --json {HERE.parent / 'bridge_calib.json'} "
                    f"--md {HERE.parent / 'bridge_calib.md'}")
#: stand_net.stand_ckpt_kind values the launcher writes (sn_stand_net_eval_crab_hex_gait.CKPT_KIND_*)
CKPT_KIND_TRAINED, CKPT_KIND_S1 = "sn_dagger", "s1_identity"


def _meta(stamp) -> dict:
    return json.loads((Path(stamp) / "run_meta.json").read_text())


def _episodes(stamp) -> dict:
    return {f.name: f for f in sorted((Path(stamp) / "raw").glob("episode_*.npz"))}


# ===================================================================================== meta
def meta_reasons(sn_stamp, refs_stamp, *, identity: bool) -> list[str]:
    """The stand_net head-validity fields of one stamp (fail closed)."""
    out = []
    sm, rm = _meta(sn_stamp), _meta(refs_stamp)
    blk = sm.get("p2r_mode") or {}
    if blk.get("variant") != SN.VARIANT:
        out.append(f"p2r_mode.variant {blk.get('variant')!r} != {SN.VARIANT!r}")
    mask = blk.get(SN.MASK_KEY)
    if isinstance(mask, bool) or mask not in (0, 1):
        out.append(f"p2r_mode.{SN.MASK_KEY} {mask!r} is not an int 0 / 1")
    sp = blk.get(SN.CKPT_KEY[0]) or {}
    sha = sp.get(SN.CKPT_KEY[1]) if isinstance(sp, dict) else None
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        out.append(f"p2r_mode.{'.'.join(SN.CKPT_KEY)} {sha!r} is not a sha256")
    s1 = blk.get("s1_checkpoint_sha256")
    if not s1 or s1 != rm.get("checkpoint_sha256") or s1 != sm.get("checkpoint_sha256"):
        out.append(f"p2r_mode.s1_checkpoint_sha256 {str(s1)[:12]} != REFS / stamp checkpoint_sha256 "
                   f"{str(rm.get('checkpoint_sha256'))[:12]} / {str(sm.get('checkpoint_sha256'))[:12]}")
    sn = blk.get(SN.BLOCK_KEY) or {}
    if sn.get("estimator_equals_s1") is not True:
        out.append(f"p2r_mode.{SN.BLOCK_KEY}.estimator_equals_s1 is {sn.get('estimator_equals_s1')!r}")
    if sn.get("r_knob") != SN.RESTANCE_D3:
        out.append(f"p2r_mode.{SN.BLOCK_KEY}.r_knob {sn.get('r_knob')!r} != the D3 string")
    kind, tm = sn.get("stand_ckpt_kind"), sn.get("trained_stand_mask")
    if kind == CKPT_KIND_TRAINED:
        if type(tm) is not int or tm != mask:
            out.append(f"p2r_mode.{SN.BLOCK_KEY}.trained_stand_mask {tm!r} != stand_mask {mask!r} (an arm evaluated "
                       "under the other mask)")
    elif kind == CKPT_KIND_S1:
        if tm is not None or sha != s1:
            out.append(f"p2r_mode.{SN.BLOCK_KEY}.stand_ckpt_kind s1_identity but the stand checkpoint "
                       f"{str(sha)[:12]} is not S1 {str(s1)[:12]} (trained_stand_mask {tm!r})")
    else:
        out.append(f"p2r_mode.{SN.BLOCK_KEY}.stand_ckpt_kind {kind!r} missing / unknown (the trained-arm check)")
    if sm.get("env_vars") is None:
        out.append("run_meta.env_vars missing")
    else:
        out += [f"env: {r}" for r in SN.env_reasons(sm["env_vars"])]
    if identity:
        if sha != s1:
            out.append(f"identity: the stand checkpoint {str(sha)[:12]} is not S1 {str(s1)[:12]}")
        if mask != 0:
            out.append(f"identity: stand_mask {mask!r} != 0")
        if sn.get("stand_rows_action_differs_from_s1") != 0:
            out.append(f"identity: run_meta differ count {sn.get('stand_rows_action_differs_from_s1')!r} != 0")
    return out


# ===================================================================================== swap-live
def swap_live(refs_stamp, sn_stamp, *, identity: bool) -> dict:
    """(a)-(d) of the module docstring, from the raws alone."""
    import numpy as np  # noqa: PLC0415

    out = {"stand_actions_equal_stand_net": True, "off_stand_series_clean": True, "stand_mode_share": None,
           "pre_entry_matches_refs": True, "pre_entry_episodes_checked": 0, "stand_steps": 0,
           "stand_steps_differ_from_s1": 0, "episodes": 0, "fails": []}
    ref_files = _episodes(refs_stamp) if refs_stamp else {}
    shares = []
    need = {"p2r_stand_mode", *SN.RAW_KEYS, "actions"}
    for name, f in _episodes(sn_stamp).items():
        d = np.load(f)
        out["episodes"] += 1
        miss = sorted(need - set(d.files))
        if miss:
            out["fails"].append(f"{name}: raw series missing {miss}")
            out["stand_actions_equal_stand_net"] = False
            continue
        mode = np.asarray(d["p2r_stand_mode"]).astype(bool)
        acts = np.asarray(d["actions"])
        st, s1 = np.asarray(d[SN.RAW_STAND_ACTION]), np.asarray(d[SN.RAW_S1_ACTION])
        tgt, br = np.asarray(d[SN.RAW_R_TARGET]), np.asarray(d[SN.RAW_R_BRANCH])
        shares.append(float(mode.mean()) if mode.size else 0.0)
        if mode.any():
            if not np.array_equal(acts[mode], st[mode].astype(acts.dtype)):
                out["stand_actions_equal_stand_net"] = False
                out["fails"].append(f"{name}: a stand-mode action differs from sn_stand_action")
            if (br[mode] < 0).any():
                out["stand_actions_equal_stand_net"] = False
                out["fails"].append(f"{name}: a stand-mode step carries no R branch code")
            diff = int((st[mode] != s1[mode]).any(axis=-1).sum())
            out["stand_steps"] += int(mode.sum())
            out["stand_steps_differ_from_s1"] += diff
        off = ~mode
        if (st[off] != 0).any() or (s1[off] != 0).any() or (tgt[off] != 0).any() or (br[off] != -1).any():
            out["off_stand_series_clean"] = False
            out["fails"].append(f"{name}: a stand_net series is set on an off-stand step")
        first_on = int(np.argmax(mode)) if mode.any() else 0
        if first_on and name in ref_files:
            ra = np.asarray(np.load(ref_files[name])["actions"])
            n = min(first_on, acts.shape[0], ra.shape[0])
            out["pre_entry_episodes_checked"] += 1
            if n and not np.array_equal(acts[:n], ra[:n]):
                out["pre_entry_matches_refs"] = False
                out["fails"].append(f"{name}: pre-entry actions (steps 0:{n}) differ from REFS")
    out["stand_mode_share"] = float(np.mean(shares)) if shares else None
    if not out["episodes"]:
        out["fails"].append("no raw episodes")
    if not out["stand_mode_share"]:
        out["fails"].append("stand-mode share is not > 0")
    # (c) is vacuous on the suite (it enters stand at its settle, step 0) and non-vacuous on walkstop: the count is
    # reported (pre_entry_episodes_checked), never a fail by itself
    if identity and out["stand_steps_differ_from_s1"]:
        out["fails"].append(f"identity: {out['stand_steps_differ_from_s1']} stand steps where the stand net's action "
                            "differs from S1's")
    if not identity and out["stand_steps"] and not out["stand_steps_differ_from_s1"]:
        out["fails"].append("live: the stand net's action equals S1's on every stand step (the swap is not live)")
    return out


# ===================================================================================== identity bridge
def max_action_diff(refs_stamp, sn_stamp) -> float:
    """``max |actions_SN - actions_REFS|`` over the shared episodes (inf on a shape mismatch or a missing episode)."""
    import numpy as np  # noqa: PLC0415

    ra, rb = _episodes(refs_stamp), _episodes(sn_stamp)
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


def identity_read(refs_stamp, sn_stamp) -> dict:
    diff = P.p2r_bitwise_diff(str(refs_stamp), str(sn_stamp))
    if not diff:
        return {"verdict": BITWISE, "first_mismatches": [], "max_action_diff": 0.0}
    mad = max_action_diff(refs_stamp, sn_stamp)
    return {"verdict": MISS_SMALL if mad <= MISS_ATOL else MISS, "first_mismatches": diff[:10],
            "n_mismatches": len(diff), "max_action_diff": mad,
            "note": ("CHARTER miss rule: PASS-with-note only if every G1-G5 statistic is also identical "
                     f"({CALIB_BRIDGE_CMD}); else STOP and ask") if mad <= MISS_ATOL else "STOP and ask"}


def read_row(refs_stamp, sn_stamp, *, identity: bool) -> dict:
    if not refs_stamp or not sn_stamp:
        return {"verdict": MISSING, "refs_stamp": refs_stamp, "sn_stamp": sn_stamp}
    res = {"refs_stamp": str(refs_stamp), "sn_stamp": str(sn_stamp), "identity": identity,
           "meta_fails": meta_reasons(sn_stamp, refs_stamp, identity=identity),
           "swap_live": swap_live(refs_stamp, sn_stamp, identity=identity)}
    bad = bool(res["meta_fails"] or res["swap_live"]["fails"])
    if identity:
        res["bridge"] = identity_read(refs_stamp, sn_stamp)
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
        raise SystemExit(f"sn_bridge_read: row spec {spec!r} must be ROW@ES (e.g. walkstop__omni@1)")
    return row, int(es)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", default=None, help="the stand_net head's stamp name under $P2_OUT/lr")
    ap.add_argument("--refs", default=REFS_NAME)
    ap.add_argument("--rows", default=",".join(DEFAULT_ROWS))
    ap.add_argument("--refs-stamp", default=None)
    ap.add_argument("--sn-stamp", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--identity", action="store_true", help="S1 as the stand checkpoint, mask 0: bitwise = REFS")
    g.add_argument("--live", action="store_true", help="a trained stand net: its action must reach stand rows")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if (a.head is None) == (a.sn_stamp is None):
        raise SystemExit("sn_bridge_read: give exactly one of --head NAME or --sn-stamp DIR")
    rows = {}
    if a.sn_stamp:
        if not a.refs_stamp:
            raise SystemExit("sn_bridge_read: --sn-stamp needs --refs-stamp")
        rows["stamp"] = read_row(a.refs_stamp, a.sn_stamp, identity=a.identity)
    else:
        for spec in [s for s in a.rows.split(",") if s.strip()]:
            row, es = _row_spec(spec.strip())
            rows[f"{row}@{es}"] = read_row(P.newest_stamp(a.refs, row, es), P.newest_stamp(a.head, row, es),
                                           identity=a.identity)
    for k, r in rows.items():
        sl = r.get("swap_live") or {}
        br = r.get("bridge") or {}
        print(f"{k}: {r['verdict']}  bridge={br.get('verdict')} max|da|={br.get('max_action_diff')} "
              f"stand_share={sl.get('stand_mode_share')} stand_steps={sl.get('stand_steps')} "
              f"differ={sl.get('stand_steps_differ_from_s1')} pre_entry_checked={sl.get('pre_entry_episodes_checked')}")
        for why in (r.get("meta_fails") or []) + (sl.get("fails") or [])[:10] + (br.get("first_mismatches") or [])[:5]:
            print(f"    - {why}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, sort_keys=True, default=str))
    rc = exit_code(rows)
    print(f"sn_bridge_read: exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
