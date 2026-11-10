#!/usr/bin/env python3
"""Rod-speed DR campaign, step 12 (CPU, no Kit): the deploy ONNX pair for the selected student, and its parity with
the evaluated ``.pt`` (CHARTER 3.4: "ONNX export plus CPU parity for the pick").

1. Rebuild the student's three networks on CPU from the checkpoint and its run dir's ``params/agent.yaml``
   (``onnx_parity.build_step`` of the DR stress sweep, reused read-only).
2. Write ``policy.{pt,onnx}``, ``depth_latest.{pt,onnx}`` and ``omni_student_io.json`` with the Task 4 exporter
   (``parkour/scripts/rsl_rl/modules/omni_student_export.export_omni_student``, unchanged) into ``--out-dir``
   (refused when it exists and is not empty).
3. Run the sweep's ``onnx_parity.py`` (unchanged, as a subprocess) on the recorded G0 rollout against the new graphs:
   actions and latents within 1e-4, the sweep's rule. The rollout is c28000's (a Student-Play episode, 0 kg); parity
   is a network-equivalence check, so the inputs need not come from the pick.

    /home/nickmagus/krabby/isaac_venv/bin/python -B -I rsd_onnx.py --checkpoint PT --out-dir DIR --out JSON \\
        [--record NPZ]

Writes ``--out`` (the export's files with their sha256, the parity record) ending in ``"complete": true``. Exit 0 iff
the parity holds, 3 when it does not, 2 on a usage error or a refused output dir.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
W = HERE.parents[6]
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
SWEEP_PARITY = (W / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-11-06_dr_stress_sweep"
                / "scripts/onnx_parity.py")
RECORD = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rod_speed_dr/onnx/G0_record.npz")
TOL = 1e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_parity():
    spec = importlib.util.spec_from_file_location("_rsd_onnx_parity", str(SWEEP_PARITY))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(ck: Path, out_dir: Path) -> tuple[dict, int]:
    import yaml  # noqa: PLC0415

    op = load_parity()
    step, _step_mod, _depth_prop, it = op.build_step(ck)
    cfg = yaml.load((ck.parent / "params" / "agent.yaml").read_text(), Loader=yaml.FullLoader)
    est, de = cfg["estimator"], cfg["depth_encoder"]
    exp = importlib.import_module(f"{op.PKG}.omni_student_export")
    files = exp.export_omni_student(
        step.depth_actor, step.prep.estimator, step.depth_encoder, num_prop=int(est["num_prop"]),
        num_scan=int(est["num_scan"]), num_priv_explicit=int(est["num_priv_explicit"]), path=str(out_dir),
        image_shape=tuple(int(v) for v in de.get("depth_shape", (58, 87))), onnx=True,
        zero_depth=bool(de.get("zero_depth", False)))
    return files, it


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=str(RECORD))
    a = ap.parse_args(argv)
    t0 = time.time()
    ck, out_dir, rec = Path(a.checkpoint).resolve(), Path(a.out_dir).resolve(), Path(a.record).resolve()
    for p, what in ((ck, "checkpoint"), (ck.parent / "params" / "agent.yaml", "params/agent.yaml"), (rec, "record"),
                    (SWEEP_PARITY, "the sweep's onnx_parity.py")):
        if not p.is_file():
            print(f"RSD_ONNX refused: no {what} at {p}", flush=True)
            return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"RSD_ONNX refused: {out_dir} exists and is not empty", flush=True)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    files, it = export(ck, out_dir)
    parity_json = out_dir / "parity.json"
    cmd = [PY, "-B", "-I", str(SWEEP_PARITY), "--record", str(rec), "--out", str(parity_json),
           "--checkpoint", str(ck), "--onnx-dir", str(out_dir), "--tol", str(TOL)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    parity = json.loads(parity_json.read_text()) if parity_json.is_file() else {}
    ok = bool(parity.get("complete") and parity.get("parity_ok"))
    res = {"tool": "rsd_onnx.py", "checkpoint": str(ck), "checkpoint_sha256": sha256(ck), "checkpoint_iter": it,
           "out_dir": str(out_dir),
           "files": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in sorted(files.items())},
           "record": str(rec), "record_sha256": sha256(rec), "parity_cmd": cmd, "parity_rc": proc.returncode,
           "parity_tail": (proc.stdout + proc.stderr)[-2000:], "parity": parity, "tol": TOL, "parity_ok": ok,
           "wall_s": time.time() - t0, "complete": True}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"RSD_ONNX END ok={ok} iter={it} steps={parity.get('steps_compared')} encodes={parity.get('encodes')} "
          f"act_err={parity.get('max_abs_action_err')} lat_err={parity.get('max_abs_latent_err')} out={a.out}",
          flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
