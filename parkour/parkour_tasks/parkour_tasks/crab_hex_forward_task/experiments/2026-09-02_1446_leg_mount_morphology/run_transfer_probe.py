#!/usr/bin/env python3
"""Rung (iv) driver: checkpoint transfer probe. Runs the 30k phase-out head and the 20k
graduation reference head on every configuration x {slow, fwd, step} scenario of
experiments/eval/scenarios_morph.yaml (one Isaac process per scenario, serial), saves raw traces, and
writes the transfer table (fall rate, fall classes, polygon margins) to RESULTS.md.
Stdlib only; resumable (skips scenario/head pairs whose metrics exist under the output root).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_morph.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
OUT_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/leg_mount_morphology"
RESULTS = HERE / "RESULTS.md"
HEADS = {
    "30k": PARKOUR / "logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt",
    "20k": PARKOUR / "logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt",
}
CONFIGS = ["base", "B", "A10", "A15", "A20", "A10pB", "A15pB", "A20pB"]
_VARIANTS = REPO / "assets/variants"
USD_BY_CFG = {"B": _VARIANTS / "crab_simple__splay00_axis2p5in.usda", "A10": _VARIANTS / "crab_simple__splay10_axis5p5in.usda",
              "A15": _VARIANTS / "crab_simple__splay15_axis5p5in.usda", "A20": _VARIANTS / "crab_simple__splay20_axis5p5in.usda",
              "A10pB": _VARIANTS / "crab_simple__splay10_axis2p5in.usda", "A15pB": _VARIANTS / "crab_simple__splay15_axis2p5in.usda",
              "A20pB": _VARIANTS / "crab_simple__splay20_axis2p5in.usda"}
SCENARIOS = ["slow", "fwd", "step"]


def log(msg: str) -> None:
    print(f"[transfer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def report(block: str, marker: str) -> None:
    with RESULTS.open("a") as fh:
        fh.write("\n" + block.rstrip() + f"\n>>> ENTRY {marker}\n")


def wait_isaac_clear() -> None:
    """Wait for every Isaac process to exit; the driver's own pid is excluded (pgrep -f would
    otherwise match 'run_<x>.py' against '<x>.py' and idle for the full timeout)."""
    pattern = r"isaac_venv/bin/python [^ ]*(statics_battery|scripted_gait_probe_v2|eval_crab_hex_gait|rsl_rl/train)\.py"
    for _ in range(60):
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if not any(int(p) != os.getpid() for p in out):
            return
        time.sleep(10)


def latest_metrics(head: str, sid: str) -> dict | None:
    d = OUT_ROOT / head / sid / "seed001"
    if not d.exists():
        return None
    runs = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)
    for r in reversed(runs):
        f = r / "scenario_metrics.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def run_eval(head: str, sid: str) -> dict | None:
    if latest_metrics(head, sid):
        log(f"{head}/{sid}: exists, skipping")
        return latest_metrics(head, sid)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    # KRABBY_HEX_USD_PATH is read when the scene config module is imported, i.e. before the
    # harness applies the manifest env block -> it must be in the process environment.
    env.pop("KRABBY_HEX_USD_PATH", None)
    usd = USD_BY_CFG.get(sid.split("__", 1)[1])
    if usd:
        env["KRABBY_HEX_USD_PATH"] = str(usd)
    lp = HERE / "transfer_logs" / f"{head}__{sid}.log"
    lp.parent.mkdir(exist_ok=True)
    log(f"{head}/{sid}: launching")
    t0 = time.time()
    with lp.open("w") as fh:
        try:
            subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST), "--scenario", sid,
                            "--checkpoint", str(HEADS[head]), "--no-plot", "--save-raw",
                            "--output-root", str(OUT_ROOT / head)],
                           cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
        except subprocess.TimeoutExpired:
            log(f"{head}/{sid}: timeout")
    log(f"{head}/{sid}: done in {(time.time() - t0) / 60:.1f} min")
    wait_isaac_clear()
    time.sleep(15)
    return latest_metrics(head, sid)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for head in HEADS:
        for cfg in CONFIGS:
            for sc in SCENARIOS:
                sid = f"{sc}__{cfg}"
                results[(head, cfg, sc)] = run_eval(head, sid)
    lines = ["## RUNG (iv) — checkpoint transfer probe (same policy, variant plant; one-sided)",
             "| head | config | scenario | completion | falls | pitch_fwd / back / roll | tripod | tracking | walk polygon tip p50 (deg) | prefall frac neg |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for head in HEADS:
        for cfg in CONFIGS:
            for sc in SCENARIOS:
                a = (results.get((head, cfg, sc)) or {}).get("aggregate")
                if not a:
                    lines.append(f"| {head} | {cfg} | {sc} | MISSING | | | | | | |")
                    continue
                fc = a.get("fall_classes", {})
                poly = a.get("support_polygon", {})
                tip = (poly.get("walking_tip_angle_fwd_deg") or {}).get("median")
                neg = (poly.get("prefall_frac_neg_margin") or {}).get("median")
                lines.append(
                    f"| {head} | {cfg} | {sc} | {a['schedule_completion_rate']:.2f} | "
                    f"{a['termination_reasons'].get('fall', 0)} | {fc.get('pitch_fwd', 0)} / {fc.get('pitch_back', 0)} / {fc.get('roll', 0)} | "
                    f"{(a['tripod_score'].get('median') or 0):.3f} | {(a['tracking_ratio'].get('median') or 0):.3f} | "
                    f"{tip if tip is None else round(tip, 1)} | {neg if neg is None else round(neg, 2)} |")
    lines.append("- fall-rate deltas vs the base row of the same head/scenario are the scored column (−30% = strong; unchanged = neutral: policy mismatch)")
    report("\n".join(lines), "rung iv transfer")
    log("transfer table written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
