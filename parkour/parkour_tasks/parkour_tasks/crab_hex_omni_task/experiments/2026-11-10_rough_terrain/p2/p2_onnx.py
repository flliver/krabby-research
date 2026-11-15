#!/usr/bin/env python3
"""P2 step 6 (p2/P2.md): the deploy ONNX pair for the selection's pick -- the rod_speed_dr ``rsd_onnx.py`` fork.
CPU only (torch on CPU, CUDA hidden; no Kit, no GPU): rsd_onnx.py's own ``main`` does the export (Task 4's
``omni_student_export.export_omni_student``, unchanged) and the parity (the DR sweep's ``onnx_parity.py``, unchanged,
as a subprocess, actions and latents within 1e-4 on the recorded G0 rollout: a network-equivalence check, so the
inputs need not come from the pick). This wrapper adds the P2 rules:

* the checkpoint is the pick of results/p2/sel.json (verdict SELECTED), named by sha (PLAN item 9) and re-hashed;
* the export goes into a P2 output dir (``<MAIN>/parkour/outputs/omni_plan/rough_terrain_p2/onnx/PICK_<sha8>``),
  never into ``policy/`` (refused), never into a non-empty dir (rsd_onnx refuses);
* the exported ``omni_student_io.json`` must EQUAL c31000's (the bundled ``policy/rsd_10_student`` copy, itself
  checked byte-equal to RSD's export ``rod_speed_dr/onnx/PICK_c31000``);
* ``results/p2/onnx_PICK.json`` records the sha256 of the .pt and of both ONNX graphs (+ the TorchScript pair and
  the io json), the parity record and the io comparison; ``complete: true`` last.

    PY=/home/nickmagus/krabby/isaac_venv/bin/python
    $PY -I -B p2_onnx.py --from-sel                    export the pick (exit 0 ok; 3 parity / io miss; 2 refused)
    $PY -I -B p2_onnx.py --dry                         the plan: pick, paths, commands; nothing written
    $PY -I -B p2_onnx.py --self-test --scratch DIR     c31000 (MAIN logs, sha 155b5981; PLAN item 16) exported into
                                                       DIR (outside policy/ and results/): parity + io equality +
                                                       the ONNX graphs vs the bundled rsd_10_student sha256 (report)
Prints ``P2_ONNX END ok=<bool> ...`` last.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/p2
CAMP = HERE.parent
EXPS = CAMP.parent
OMNI = EXPS.parent
W = Path(__file__).resolve().parents[7]
MAIN = Path("/home/nickmagus/krabby/krabby-research")
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
RSD_ONNX = EXPS / "2026-11-07_rod_speed_dr" / "scripts" / "rsd_onnx.py"
P2_OUT = MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain_p2"
ONNX_ROOT = P2_OUT / "onnx"
RESULTS_P2 = CAMP / "results" / "p2"
SEL_JSON = RESULTS_P2 / "sel.json"
PICK_JSON = RESULTS_P2 / "onnx_PICK.json"
POLICY_DIR = OMNI / "policy"
C31000_IO = POLICY_DIR / "rsd_10_student" / "omni_student_io.json"
C31000_BUNDLE = POLICY_DIR / "rsd_10_student"
RSD_PICK_DIR = MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr" / "onnx" / "PICK_c31000"
RSD_ONNX_PICK = EXPS / "2026-11-07_rod_speed_dr" / "results" / "onnx_PICK.json"
C31000_CK = MAIN / "parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19/model_31000.pt"
C31000_SHA = "155b5981a718ec2c41460490c38e2aadb61c60c7b48dd67e3a33a45b1d5e634e"
C1_SHA = "6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1"
RECORD = MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr" / "onnx" / "G0_record.npz"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def say(line: str) -> None:
    print(line, flush=True)


def load_rsd_onnx():
    spec = importlib.util.spec_from_file_location("_p2_rsd_onnx", str(RSD_ONNX))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert Path(mod.W).resolve() == W.resolve(), f"rsd_onnx resolves its worktree to {mod.W}, not {W}"
    return mod


def under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def pick_of_sel(sel_path: Path = SEL_JSON) -> tuple[dict | None, list]:
    doc = json.loads(sel_path.read_text()) if sel_path.is_file() else None
    if doc is None:
        return None, [f"no {sel_path} (p2_select.py --decide writes it)"]
    bad = []
    if doc.get("verdict") != "SELECTED":
        bad.append(f"{sel_path} verdict {doc.get('verdict')!r} (want SELECTED)")
    pk = doc.get("pick") or {}
    for k in ("cand", "ckpt", "sha256", "iter"):
        if pk.get(k) in (None, ""):
            bad.append(f"{sel_path} pick has no {k}")
    if pk.get("teacher_sha256") not in (None, C1_SHA):
        bad.append(f"pick teacher sha {str(pk.get('teacher_sha256'))[:8]} != c1 {C1_SHA[:8]}")
    return pk, bad


def io_reference() -> tuple[dict | None, list, dict]:
    """c31000's omni_student_io.json (the bundled policy/rsd_10_student copy), checked equal to RSD's export."""
    bad, info = [], {}
    if not C31000_IO.is_file():
        return None, [f"no {C31000_IO}"], info
    info["bundled"] = {"path": str(C31000_IO), "sha256": sha256(C31000_IO)}
    rsd_io = RSD_PICK_DIR / "omni_student_io.json"
    if rsd_io.is_file():
        info["rsd_export"] = {"path": str(rsd_io), "sha256": sha256(rsd_io)}
        if info["rsd_export"]["sha256"] != info["bundled"]["sha256"]:
            bad.append(f"bundled {C31000_IO} != RSD's export {rsd_io}")
    return json.loads(C31000_IO.read_text()), bad, info


def export_and_check(ck: Path, want_sha: str, out_dir: Path, record_out: Path | None, label: dict,
                     record: Path = RECORD) -> int:
    """rsd_onnx.main on (ck, out_dir), then the io equality; writes ``record_out`` (when given). Exit as rsd_onnx,
    or 3 on an io miss."""
    t0 = time.time()
    bad = []
    if not ck.is_file():
        bad.append(f"no checkpoint {ck}")
    elif sha256(ck) != want_sha:
        bad.append(f"{ck} sha256 {sha256(ck)[:8]} != {want_sha[:8]}")
    if under(out_dir, POLICY_DIR) or under(out_dir, CAMP):
        bad.append(f"out dir {out_dir} is under policy/ or the campaign dir (P2 exports go to {ONNX_ROOT})")
    if out_dir.exists() and any(out_dir.iterdir()):
        bad.append(f"{out_dir} exists and is not empty")
    ref_io, ref_bad, ref_info = io_reference()
    bad += ref_bad
    if bad:
        say(f"P2_ONNX refused: {'; '.join(bad)}")
        say("P2_ONNX END ok=False refused=1")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU only (never the GPU the distill / rows use)
    R = load_rsd_onnx()
    rsd_json = out_dir / "rsd_onnx.json"
    rc = R.main(["--checkpoint", str(ck), "--out-dir", str(out_dir), "--out", str(rsd_json), "--record", str(record)])
    rec = json.loads(rsd_json.read_text()) if rsd_json.is_file() else {}
    files = rec.get("files") or {}
    io_path = Path((files.get("io") or {}).get("path") or out_dir / "omni_student_io.json")
    io_new = json.loads(io_path.read_text()) if io_path.is_file() else None
    io_equal = io_new is not None and io_new == ref_io
    io_diff = sorted(k for k in set(io_new or {}) | set(ref_io or {}) if (io_new or {}).get(k) != (ref_io or {}).get(k))
    bundle_cmp = {}
    for k, fn in (("policy_onnx", "policy.onnx"), ("depth_onnx", "depth_latest.onnx")):
        b = C31000_BUNDLE / fn
        if b.is_file() and k in files:
            bundle_cmp[k] = {"new": files[k]["sha256"], "c31000_bundled": sha256(b),
                             "equal": files[k]["sha256"] == sha256(b)}
    ok = bool(rc == 0 and rec.get("parity_ok") and io_equal and rec.get("checkpoint_sha256") == want_sha)
    res = {"tool": "p2_onnx.py", "schema": "p2_onnx_pick_v1", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **label,
           "checkpoint": str(ck), "checkpoint_sha256": rec.get("checkpoint_sha256"), "checkpoint_iter":
           rec.get("checkpoint_iter"), "out_dir": str(out_dir),
           "sha256": {"pt": rec.get("checkpoint_sha256"), "policy_onnx": (files.get("policy_onnx") or {}).get("sha256"),
                      "depth_onnx": (files.get("depth_onnx") or {}).get("sha256"),
                      "policy_jit": (files.get("policy_jit") or {}).get("sha256"),
                      "depth_jit": (files.get("depth_jit") or {}).get("sha256"),
                      "io": (files.get("io") or {}).get("sha256")},
           "files": files, "parity": rec.get("parity"), "parity_ok": bool(rec.get("parity_ok")), "tol": rec.get("tol"),
           "rsd_onnx_rc": rc, "rsd_onnx_record": str(rsd_json), "record": str(record),
           "record_sha256": rec.get("record_sha256"),
           "io_equal_c31000": io_equal, "io_differs_on": io_diff, "io_reference": ref_info,
           "onnx_vs_c31000_bundle": bundle_cmp,
           "rule": "P2.md step 6: parity <= 1e-4 vs the .pt (rsd_onnx.py / onnx_parity.py, unchanged); I/O json == "
                   "c31000's omni_student_io.json; exported into a P2 output dir, never policy/",
           "ok": ok, "wall_s": time.time() - t0, "complete": True}
    if record_out is not None:
        record_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = record_out.with_suffix(record_out.suffix + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, record_out)
    p = rec.get("parity") or {}
    say(f"P2_ONNX END ok={ok} iter={rec.get('checkpoint_iter')} sha={str(rec.get('checkpoint_sha256'))[:8]} "
        f"parity_ok={rec.get('parity_ok')} act_err={p.get('max_abs_action_err')} lat_err={p.get('max_abs_latent_err')} "
        f"io_equal_c31000={io_equal}{' differs ' + str(io_diff) if io_diff else ''} "
        f"policy_onnx={str(res['sha256']['policy_onnx'])[:8]} depth_onnx={str(res['sha256']['depth_onnx'])[:8]} "
        f"out={record_out or '(not recorded)'}")
    if rc != 0:
        return rc
    return 0 if ok else 3


def plan_from_sel() -> tuple[dict | None, list, Path | None]:
    pk, bad = pick_of_sel()
    out_dir = ONNX_ROOT / f"PICK_{pk['sha256'][:8]}" if pk and pk.get("sha256") else None
    return pk, bad, out_dir


def cmd_from_sel() -> int:
    pk, bad, out_dir = plan_from_sel()
    if bad:
        say(f"P2_ONNX refused: {'; '.join(bad)}")
        say("P2_ONNX END ok=False refused=1")
        return 2
    if PICK_JSON.exists():
        say(f"P2_ONNX refused: {PICK_JSON} exists (one export per pick; move it aside is the user's call)")
        say("P2_ONNX END ok=False refused=1")
        return 2
    label = {"pick": {k: pk.get(k) for k in ("cand", "sha256", "iter", "ckpt", "teacher", "teacher_sha256",
                                              "policy")}, "sel_json": str(SEL_JSON), "sel_json_sha256": sha256(SEL_JSON)}
    return export_and_check(Path(pk["ckpt"]), pk["sha256"], out_dir, PICK_JSON, label)


def cmd_dry() -> int:
    pk, bad, out_dir = plan_from_sel()
    say("# P2 ONNX (dry): nothing is written")
    say(f"#   rsd_onnx.py   {RSD_ONNX} ({'present' if RSD_ONNX.is_file() else 'MISSING'})")
    try:
        R = load_rsd_onnx()
        say(f"#   parity tool   {R.SWEEP_PARITY} ({'present' if Path(R.SWEEP_PARITY).is_file() else 'MISSING'}), "
            f"tol {R.TOL}")
    except Exception as exc:  # noqa: BLE001
        say(f"#   rsd_onnx import FAILED: {type(exc).__name__}: {exc}")
    say(f"#   G0 record     {RECORD} ({'present' if RECORD.is_file() else 'MISSING'})")
    ref_io, ref_bad, ref_info = io_reference()
    say(f"#   io reference  {C31000_IO} sha {ref_info.get('bundled', {}).get('sha256', '?')[:8]} "
        f"(== RSD export: {not ref_bad})")
    if bad:
        say(f"#   pick          NOT AVAILABLE: {'; '.join(bad)}")
        ck, sha_, out_dir = Path("<pick ckpt from results/p2/sel.json>"), "<sha256>", ONNX_ROOT / "PICK_<sha8>"
    else:
        ck, sha_ = Path(pk["ckpt"]), pk["sha256"]
        say(f"#   pick          {pk['cand']} sha256 {sha_} iter {pk['iter']} ckpt {ck} "
            f"({'present' if ck.is_file() else 'MISSING'})")
    say(f"#   out dir       {out_dir} ({'exists' if out_dir.exists() else 'new'})")
    say(f"#   record        {PICK_JSON} ({'EXISTS: a real run refuses' if PICK_JSON.exists() else 'new'})")
    say(f"#   command       CUDA_VISIBLE_DEVICES= {PY} -I -B {RSD_ONNX} --checkpoint {ck} --out-dir {out_dir} "
        f"--out {out_dir}/rsd_onnx.json --record {RECORD}   (called in-process via rsd_onnx.main)")
    say(f"#   then          io json == {C31000_IO}; sha256 of .pt / policy.onnx / depth_latest.onnx -> {PICK_JSON}")
    say(f"P2_ONNX DRY pick_ready={not bad}")
    return 0


def cmd_self_test(scratch: Path) -> int:
    scratch = Path(scratch).resolve()
    if under(scratch, POLICY_DIR) or under(scratch, CAMP) or under(scratch, P2_OUT):
        say(f"P2_ONNX refused: --scratch {scratch} must be outside policy/, the campaign dir and the P2 outputs")
        return 2
    out_dir = scratch / "PICK_c31000_selftest"
    rec = scratch / "onnx_selftest.json"
    rc = export_and_check(C31000_CK, C31000_SHA, out_dir, rec,
                          {"self_test": "c31000 re-exported (MAIN logs by sha, PLAN item 16)"})
    doc = json.loads(rec.read_text()) if rec.is_file() else {}
    rsd = json.loads(RSD_ONNX_PICK.read_text()) if RSD_ONNX_PICK.is_file() else {}
    same = {k: (doc.get("files") or {}).get(k, {}).get("sha256") == (rsd.get("files") or {}).get(k, {}).get("sha256")
            for k in ("policy_onnx", "depth_onnx", "io", "policy_jit", "depth_jit")}
    say(f"P2_ONNX_SELFTEST {'PASS' if rc == 0 and doc.get('ok') else 'FAIL'} rc={rc} parity_ok={doc.get('parity_ok')} "
        f"io_equal_c31000={doc.get('io_equal_c31000')} sha_equal_to_RSD_export={same} "
        f"bundle={ {k: v['equal'] for k, v in (doc.get('onnx_vs_c31000_bundle') or {}).items()} }")
    return 0 if rc == 0 and doc.get("ok") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-sel", action="store_true")
    g.add_argument("--dry", action="store_true")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.dry:
        return cmd_dry()
    if a.self_test:
        if a.scratch is None:
            ap.error("--self-test needs --scratch DIR")
        return cmd_self_test(a.scratch)
    return cmd_from_sel()


if __name__ == "__main__":
    sys.exit(main())
