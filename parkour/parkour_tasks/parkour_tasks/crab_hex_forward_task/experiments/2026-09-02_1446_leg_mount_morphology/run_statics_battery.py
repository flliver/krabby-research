#!/usr/bin/env python3
"""Rung (ii) driver: run statics_battery.py for every configuration, serially (one Isaac
process at a time), then write the statics table to RESULTS.md. Stdlib only; resumable
(skips configurations whose JSON already exists).

Configurations: the golden base + the 7 variant USDs in assets/variants. Each run is
~15-25 min (boot-dominated). Sign-off videos land in statics/video_<tag>/.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
VARIANTS = REPO / "assets" / "variants"
OUT = HERE / "statics"
RESULTS = HERE / "RESULTS.md"
CONFIGS = [
    ("base", None), ("B", "crab_simple__splay00_axis2p5in.usda"),
    ("A10", "crab_simple__splay10_axis5p5in.usda"), ("A15", "crab_simple__splay15_axis5p5in.usda"),
    ("A20", "crab_simple__splay20_axis5p5in.usda"),
    ("A10+B", "crab_simple__splay10_axis2p5in.usda"), ("A15+B", "crab_simple__splay15_axis2p5in.usda"),
    ("A20+B", "crab_simple__splay20_axis2p5in.usda"),
]


def log(msg: str) -> None:
    print(f"[statics {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def run_one(tag: str, usd: str | None) -> dict | None:
    js = OUT / f"statics_{tag}.json"
    if js.exists():
        log(f"{tag}: exists, skipping")
        return json.loads(js.read_text())
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    if usd:
        env["KRABBY_HEX_USD_PATH"] = str(VARIANTS / usd)
    else:
        env.pop("KRABBY_HEX_USD_PATH", None)
    cmd = [PY, str(HERE / "statics_battery.py"), "--headless", "--video", "--out_dir", str(OUT),
           "--tag", tag.replace("+", "_")]
    log(f"{tag}: launching ({usd or 'golden'})")
    t0 = time.time()
    with (OUT / f"statics_{tag.replace('+', '_')}.log").open("w") as fh:
        subprocess.run(cmd, cwd=REPO / "parkour", env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=5400)
    js2 = OUT / f"statics_{tag.replace('+', '_')}.json"
    log(f"{tag}: done in {(time.time() - t0) / 60:.1f} min, json {'OK' if js2.exists() else 'MISSING'}")
    wait_isaac_clear()
    time.sleep(20)
    return json.loads(js2.read_text()) if js2.exists() else None


def main() -> int:
    OUT.mkdir(exist_ok=True)
    results = {}
    for tag, usd in CONFIGS:
        results[tag] = run_one(tag, usd)
    base = results.get("base") or {}
    lines = ["## RUNG (ii) — statics battery (20 jittered settles + cam sweep per configuration)",
             "| config | sound? | upright/20 | penetr. | leg-link contact max N (settle / sweep) | A-share mean±sd | pitch_eq deg | root_z | com_z | yaw range (deg, RR) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, _ in CONFIGS:
        r = results.get(tag)
        if not r:
            lines.append(f"| {tag} | CRASH | — | — | — | — | — | — | — | — |")
            continue
        sound = (r["n_upright"] >= 19 and r["n_penetrating"] == 0 and r["leg_link_contact_max_N"] < 15.0)
        yr = r["sweep"]["yaw_range_rad"].get("RR_Body_Hip_RevoluteJoint", [float("nan"), float("nan")])
        lines.append(
            f"| {tag} | {'yes' if sound else 'NO'} | {r['n_upright']} | {r['n_penetrating']} | "
            f"{r['leg_link_contact_max_N']:.1f} / {r['sweep']['leg_link_contact_max_N']:.1f} | "
            f"{100 * r['A_share_mean']:.1f}±{100 * r['A_share_sd']:.1f}% | {r['pitch_eq_mean_deg']:+.2f} | "
            f"{r['root_z_mean']:.4f} | {r['com_z_mean']:.4f} | {math.degrees(yr[0]):+.1f}..{math.degrees(yr[1]):+.1f} |")
    if base:
        lines.append(f"- base root_z {base['root_z_mean']:.4f} m: variants within ±10 mm keep KRABBY_HEX_SPAWN_Z=1.085")
    lines.append("- sound = >=19/20 settles upright, no penetration, femur/hip contact < 15 N at neutral")
    report("\n".join(lines), "rung ii statics")
    log("statics table written to RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
