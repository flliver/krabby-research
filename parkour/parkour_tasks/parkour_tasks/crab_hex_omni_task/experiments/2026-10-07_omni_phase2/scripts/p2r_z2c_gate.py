#!/usr/bin/env python3
"""Omni phase 2, stage p2r Z2c -- the FIDELITY GATE (P2R_Z2C_PLAN.md step 5; CHARTER 12). CPU only, read-only: no
Isaac, no CUDA, no training. NOTE(p2r Z2c, 2026-10-10).

WHAT IT JUDGES. ONE checkpoint (default ``<run dir>/model_20500.pt``, the Z2c run's +505 save) on ONE bank -- the
held-out WALKSTOP es2 stand bank (``p2r_stand_bank.npz`` of the drift-proof-R probe; the plan's audit pass 1: suite
es2 is a near-duplicate of the BC-trained suite es1 bank, so it is never pooled in). R is RECOMPUTED on the bank's obs
with the run's constants (``--restance`` or the run's ``params/agent.yaml`` ``algorithm.restance``; neither -> rc 1),
through :mod:`p2r_z2_fidelity` (the same replica, the same all-HOLD statistic the scorer reports).

THE STATISTIC (the EVAL path: own-estimator priv_explicit, ``hist_encoding=True``): on ALL-HOLD rows (every leg's
recomputed branch is HOLD), the per-leg mean signed ``mu - R`` of the hip (action cols 6..11) and the cam (0..5);
FAIL when either

    max over legs |hip| > 0.05      (Z2: 0.13-0.18, a partial LIFT blended into the clean stand)
    max over legs |cam| > 0.05      (half the drift-proof R's trim authority 0.1).

The training path (true priv, ``hist_encoding=False``) and the knee are REPORTED, never gated.

``--ckpt-state STATE.pt`` scores an OFFLINE actor through this exact path (the plan's step-4 positive control): the
state's ``model_state_dict`` (or a flat ``actor.*`` state dict) overrides ``--ckpt``'s weights key for key (every key
must exist in the base; ``estimator_state_dict``, when present, overrides the estimator).

HELD OUT. When the run's ``params/agent.yaml`` names its BC banks (``algorithm.bc_bank``), a gate bank among them is
refused (rc 1): the gate must read states BC never fit.

Exit codes (``scripts/run_arm.sh``'s Z2c branch): 0 PASS; 3 FAIL (kill: ``systemctl --user stop``, STOP_QUEUE,
NOTIFY); 2 WAIT (the checkpoint does not exist yet); 1 error (unreadable input, no R constants, no all-HOLD row, a
non-held-out bank). The LAST stdout line is one JSON object (``--json OUT`` also writes it).

CLI:
    p2r_z2c_gate.py --run-dir RUN [--ckpt MODEL.pt] [--bank BANK.npz] [--restance K=V,...] [--agent-yaml PATH]
                    [--ckpt-state STATE.pt] [--hip-line 0.05] [--cam-line 0.05] [--json OUT]
    (--bank defaults to $P2_Z2C_GATE_BANK)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import p2r_z2_fidelity as fid  # noqa: E402  (same directory, CPU only: the replica + the all-HOLD statistic)

GATE_CKPT = "model_20500.pt"       # the Z2c run's +505 save (GRAFT fork: first logged iteration 19995)
HIP_LINE = 0.05
CAM_LINE = 0.05
BANK_ENV = "P2_Z2C_GATE_BANK"
RC_PASS, RC_ERROR, RC_WAIT, RC_FAIL = 0, 1, 2, 3
VERDICT = {RC_PASS: "PASS", RC_ERROR: "ERROR", RC_WAIT: "WAIT", RC_FAIL: "FAIL"}


def judge(stat: dict, hip_line: float = HIP_LINE, cam_line: float = CAM_LINE) -> dict:
    """The gate's verdict on :func:`p2r_z2_fidelity.allhold_bias`'s result (pure)."""
    if not stat or not stat.get("allhold_rows"):
        return {"rc": RC_ERROR, "fails": [], "why": "the bank has no all-HOLD row: the gate cannot be read"}
    fails = []
    for name, line in (("hip", hip_line), ("cam", cam_line)):
        v = stat.get(f"{name}_max_abs")
        if v is None or not math.isfinite(v):
            return {"rc": RC_ERROR, "fails": [], "why": f"{name} statistic is {v!r}"}
        if v > line:
            fails.append(f"max over legs |mean all-HOLD mu_{name} - R_{name}| {v:.4g} > {line:g}")
    if fails:
        return {"rc": RC_FAIL, "fails": fails, "why": "; ".join(fails)}
    return {"rc": RC_PASS, "fails": [],
            "why": (f"hip {stat['hip_max_abs']:.4g} <= {hip_line:g} and cam {stat['cam_max_abs']:.4g} <= {cam_line:g} "
                    f"on {stat['allhold_rows']} all-HOLD rows")}


def apply_state(sd: dict, esd: dict, state_path: Path) -> tuple[dict, dict]:
    """``--ckpt-state``: the base weights overridden key for key (an unknown key is a ValueError)."""
    d = torch.load(str(state_path), map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        raise ValueError(f"{state_path}: not a state dict")
    if "model_state_dict" in d:
        msd = d["model_state_dict"]
    else:
        msd = {k: v for k, v in d.items() if k != "estimator_state_dict" and torch.is_tensor(v)}
    if not msd:
        raise ValueError(f"{state_path}: carries no actor weights")
    new_sd, new_esd = dict(sd), dict(esd)
    unknown = [k for k in msd if k not in sd]
    if unknown:
        raise ValueError(f"{state_path}: {len(unknown)} key(s) not in the base checkpoint, e.g. {unknown[:3]}")
    for k, v in msd.items():
        if tuple(v.shape) != tuple(sd[k].shape):
            raise ValueError(f"{state_path}: {k} has shape {tuple(v.shape)}, the base {tuple(sd[k].shape)}")
        new_sd[k] = v.detach().float()
    for k, v in (d.get("estimator_state_dict") or {}).items():
        new_esd[str(k)] = v.detach().float()
    return new_sd, new_esd


def bc_banks_of(agent_yaml: Path | None) -> list[str]:
    """``algorithm.bc_bank`` of a run's ``params/agent.yaml`` (a list or a comma string), resolved; [] if none."""
    if agent_yaml is None or not Path(agent_yaml).is_file():
        return []
    import yaml  # noqa: PLC0415

    class _L(yaml.SafeLoader):
        pass

    _L.add_multi_constructor("", lambda ld, suffix, node: None)
    cfg = yaml.load(open(agent_yaml), Loader=_L) or {}  # noqa: S506
    b = (cfg.get("algorithm") or {}).get("bc_bank") if isinstance(cfg, dict) else None
    if not b:
        return []
    items = b if isinstance(b, (list, tuple)) else str(b).split(",")
    return [resolved(str(x).strip()) for x in items if str(x).strip()]


def resolved(path) -> str:
    """A repo-relative path resolved against the repo root (as motion_knobs.resolve_anchor_ckpt), then made real."""
    p = Path(path)
    return str((p if p.is_absolute() else fid.R / p).resolve())


def gate(bank_path: Path, ckpt: Path, *, run_dir: Path | None = None, agent_yaml: Path | None = None,
         restance: str | None = None, ckpt_state: Path | None = None, hip_line: float = HIP_LINE,
         cam_line: float = CAM_LINE) -> dict:
    """The full gate read (IO): see the module docstring. Returns the JSON record, ``rc`` included."""
    out = {"ckpt": str(ckpt), "ckpt_state": (str(ckpt_state) if ckpt_state else None), "bank": str(bank_path),
           "run_dir": (str(run_dir) if run_dir else None), "lines": {"hip": hip_line, "cam": cam_line},
           "path": "eval (own-estimator priv_explicit, hist_encoding=True)"}
    if not Path(ckpt).is_file():
        return dict(out, rc=RC_WAIT, why=f"{ckpt} does not exist yet")
    if ckpt_state is not None and not Path(ckpt_state).is_file():
        return dict(out, rc=RC_ERROR, why=f"--ckpt-state {ckpt_state} is not a file")
    if not Path(bank_path).is_file():
        return dict(out, rc=RC_ERROR, why=f"bank {bank_path} is not a file")
    rst = fid.load_restance()
    ckpt = Path(ckpt)
    run_yaml = (Path(run_dir) / "params" / "agent.yaml") if run_dir else None
    ck_yaml = ckpt.parent / "params" / "agent.yaml"
    ays = [y for y in (agent_yaml, run_yaml, ck_yaml) if y is not None]
    aw = next((y for y in ays if Path(y).is_file()), fid.S1_AGENT_YAML)
    w = fid.widths(aw)
    params, src = fid.resolve_restance(rst, restance, ays)
    held = [b for y in (run_yaml, ck_yaml) for b in bc_banks_of(y)]
    out.update(agent_yaml=str(aw), restance_scored=rst.format_restance(params), restance_source=src,
               bc_banks_of_run=sorted(set(held)) or None)
    if resolved(bank_path) in set(held):
        return dict(out, rc=RC_ERROR, why=f"the gate bank {bank_path} is one of the run's BC banks: not held out")
    bank = fid.recompute_targets(fid.load_bank(Path(bank_path)), rst, params)
    obs = np.asarray(bank["obs"])
    out.update(rows=int(obs.shape[0]), obs_width=int(obs.shape[1]),
               stand_bit_share=(float((obs[:, fid.STAND_BIT_COL] == 1).mean()) if obs.shape[1] > fid.STAND_BIT_COL
                                else None),
               bank_restance=(str(bank["restance"]) if "restance" in bank else None),
               bank_has_residual=(bool(bank["has_residual"]) if "has_residual" in bank else None))
    if "stored_target" in bank:
        out["target_vs_stored_max_abs"] = float(np.abs(np.asarray(bank["target"], np.float32)
                                                       - np.asarray(bank["stored_target"], np.float32)).max())
    sd, esd = fid.load_ckpt(ckpt)
    if ckpt_state is not None:
        sd, esd = apply_state(sd, esd, Path(ckpt_state))
    paths = fid.score_paths(bank, sd, esd, w, rst)
    out["eval"], out["train_reported"] = paths["eval"], paths["train"]
    j = judge(paths["eval"], hip_line, cam_line)
    out.update(rc=j["rc"], fails=j["fails"], why=j["why"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=None, help=f"default <run dir>/{GATE_CKPT}")
    ap.add_argument("--bank", type=Path, default=None, help=f"default ${BANK_ENV}")
    ap.add_argument("--restance", default=None)
    ap.add_argument("--agent-yaml", type=Path, default=None)
    ap.add_argument("--ckpt-state", type=Path, default=None)
    ap.add_argument("--hip-line", type=float, default=HIP_LINE)
    ap.add_argument("--cam-line", type=float, default=CAM_LINE)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    bank = a.bank or (Path(os.environ[BANK_ENV]) if os.environ.get(BANK_ENV) else None)
    ckpt = a.ckpt or ((a.run_dir / GATE_CKPT) if a.run_dir else None)
    if bank is None or ckpt is None:
        out = {"rc": RC_ERROR, "why": f"need --bank (or ${BANK_ENV}) and --ckpt (or --run-dir)"}
    else:
        try:
            out = gate(bank, ckpt, run_dir=a.run_dir, agent_yaml=a.agent_yaml, restance=a.restance,
                       ckpt_state=a.ckpt_state, hip_line=a.hip_line, cam_line=a.cam_line)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            out = {"rc": RC_ERROR, "why": f"{exc.__class__.__name__}: {exc}", "ckpt": str(ckpt), "bank": str(bank)}
    out["verdict"] = VERDICT[int(out["rc"])]
    out["exit"] = int(out["rc"])
    text = json.dumps(out, default=str)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
    print(text)
    return int(out["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
