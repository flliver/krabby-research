#!/usr/bin/env python3
"""Rung (iii) driver: open-loop scripted tripod gait (scripted_gait_probe_v2 --best_combo) on
every configuration at w in {0.3, 0.5}, 8 jittered envs each, in-session baseline first.
Scores: median time-to-fall (ratio to base at the same w), survivors/8, pitch-forward share,
mean vx over the first 2 s (ratio), leg-link contact max. Run with the isaac venv python
(numpy + gait_eval.metrics are imported for the offline scoring). Serial, resumable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
PROBE = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/scripted_gait_probe_v2.py"
VARIANTS = REPO / "assets/variants"
OUT = HERE / "probe"
RESULTS = HERE / "RESULTS.md"
sys.path.insert(0, str(PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"))
from gait_eval import metrics as M  # noqa: E402

CONFIGS = [("base", None), ("B", "splay00_axis2p5in"), ("A10", "splay10_axis5p5in"),
           ("A15", "splay15_axis5p5in"), ("A20", "splay20_axis5p5in"), ("A10+B", "splay10_axis2p5in"),
           ("A15+B", "splay15_axis2p5in"), ("A20+B", "splay20_axis2p5in")]
WS = (0.3, 0.5)
NUM_ENVS = 8


def log(msg: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def npz_path(cfg: str, w: float) -> Path:
    return OUT / cfg.replace("+", "p") / f"probe2_w+{w:.2f}_ph0.00_ks+1_{cfg.replace('+', 'p')}_w{w:.1f}.npz"


def run_probe(cfg: str, tag: str | None, w: float) -> Path | None:
    out = npz_path(cfg, w)
    if out.exists():
        log(f"{cfg} w={w}: exists, skipping")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    if tag:
        env["KRABBY_HEX_USD_PATH"] = str(VARIANTS / f"crab_simple__{tag}.usda")
    else:
        env.pop("KRABBY_HEX_USD_PATH", None)
    cmd = [PY, str(PROBE), "--headless", "--best_combo", "--w", str(w), "--num_envs", str(NUM_ENVS),
           "--out_dir", str(out.parent), "--tag_suffix", f"_{cfg.replace('+', 'p')}_w{w:.1f}"]
    log(f"{cfg} w={w}: launching")
    t0 = time.time()
    with (out.parent / f"probe_w{w:.1f}.log").open("w") as fh:
        try:
            subprocess.run(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            log(f"{cfg} w={w}: timeout")
    log(f"{cfg} w={w}: done in {(time.time() - t0) / 60:.1f} min, npz {'OK' if out.exists() else 'MISSING'}")
    wait_isaac_clear()
    time.sleep(15)
    return out if out.exists() else None


def score(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    dt = float(z["dt"])
    done = z["done_all"]                      # (T, N)
    T, N = done.shape
    ttf, classes, vx0, tips, negs = [], [], [], [], []
    for i in range(N):
        idx = np.flatnonzero(done[:, i])
        end = int(idx[0]) if idx.size else T   # first episode only; frame idx[0] is already post-reset
        if end == 0:
            end = 1
        ttf.append((end + 1) * dt if idx.size else np.inf)
        q = z["root_quat_w_all"][:end, i]
        fail = np.zeros(end, dtype=bool)
        if idx.size:
            fail[-1] = True
        fd = M.fall_direction_metrics(q, fail, dt=dt)
        classes.append(fd["fall_class"] if idx.size else "none")
        vx0.append(float(np.mean(z["root_lin_vel_b_all"][: min(end, int(round(2.0 / dt))), i, 0])))
        sp = M.support_polygon_metrics(z["foot_pos_w_all"][:end, i], z["foot_force_N_all"][:end, i],
                                       z["root_pos_w_all"][:end, i], q, dt=dt, crab_failure=fail,
                                       walking_mask=np.ones(end, dtype=bool))
        tips.append(sp["walking"]["tip_angle_fwd_deg"]["p50"])
        negs.append(sp["prefall"]["frac_neg_margin"])
    finite = [t for t in ttf if np.isfinite(t)]
    # leg-link contact over PRE-FALL frames only (a toppled robot's tibia on the ground reads ~1 kN);
    # hip/femur = leg-leg / leg-body interference channel, tibia = tibia-ground or tibia-tibia.
    names = [str(n) for n in z["leg_link_names"]]
    hf = [i for i, n in enumerate(names) if not n.endswith("Tibia")]
    ti = [i for i, n in enumerate(names) if n.endswith("Tibia")]
    leg = z["leg_contact_N_all"]
    hf_max, ti_max = 0.0, 0.0
    for i in range(N):
        idx = np.flatnonzero(done[:, i])
        end = int(idx[0]) if idx.size else T
        if end > 0:
            hf_max = max(hf_max, float(leg[:end, i][:, hf].max()))
            ti_max = max(ti_max, float(leg[:end, i][:, ti].max()))
    return {
        "leg_contact_hipfemur_max_N": hf_max, "leg_contact_tibia_max_N": ti_max,
        "n": N, "survivors": int(sum(1 for t in ttf if not np.isfinite(t))),
        "ttf_median_s": float(np.median([t if np.isfinite(t) else T * dt for t in ttf])),
        "ttf_fallen_median_s": float(np.median(finite)) if finite else None,
        "pitch_fwd_share": float(np.mean([c == "pitch_fwd" for c in classes if c != "none"])) if finite else None,
        "classes": {c: classes.count(c) for c in set(classes)},
        "vx_first2s_mean": float(np.mean(vx0)),
        "walk_tip_p50_deg": float(np.nanmedian([t for t in tips if t is not None])) if any(t is not None for t in tips) else None,
        "prefall_frac_neg": float(np.nanmean([n for n in negs if n is not None])) if any(n is not None for n in negs) else None,
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    scores: dict[tuple[str, float], dict | None] = {}
    rescore = "--rescore" in sys.argv
    for cfg, tag in CONFIGS:
        for w in WS:
            p = npz_path(cfg, w) if rescore else run_probe(cfg, tag, w)
            scores[(cfg, w)] = score(p) if p and p.exists() else None
    (OUT / "probe_scores.json").write_text(json.dumps({f"{c}@{w}": s for (c, w), s in scores.items()}, indent=1))
    lines = ["## RUNG (iii) — open-loop scripted tripod gait (8 jittered envs per cell; in-session baseline)"
             + (" — RESCORED (pre-reset fall frame; pre-fall leg contact split hip/femur vs tibia)" if rescore else ""),
             "| config | w | survivors/8 | ttf median s (ratio to base) | pitch-fwd share | vx first 2 s (ratio) | walk tip p50 | prefall frac neg | hip/femur | tibia contact max N (pre-fall) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, _ in CONFIGS:
        for w in WS:
            s = scores.get((cfg, w))
            b = scores.get(("base", w))
            if not s:
                lines.append(f"| {cfg} | {w} | CRASH | | | | | | | |")
                continue
            r_ttf = s["ttf_median_s"] / b["ttf_median_s"] if b and b["ttf_median_s"] else float("nan")
            r_vx = s["vx_first2s_mean"] / b["vx_first2s_mean"] if b and abs(b["vx_first2s_mean"]) > 1e-3 else float("nan")
            pf = "—" if s["pitch_fwd_share"] is None else f"{s['pitch_fwd_share']:.2f}"
            tip = "—" if s["walk_tip_p50_deg"] is None else f"{s['walk_tip_p50_deg']:.1f}°"
            neg = "—" if s["prefall_frac_neg"] is None else f"{s['prefall_frac_neg']:.2f}"
            lines.append(f"| {cfg} | {w} | {s['survivors']} | {s['ttf_median_s']:.1f} ({r_ttf:.2f}×) | {pf} | "
                         f"{s['vx_first2s_mean']:.3f} ({r_vx:.2f}×) | {tip} | {neg} | {s['leg_contact_hipfemur_max_N']:.1f} | {s['leg_contact_tibia_max_N']:.1f} |")
    lines.append("- ttf median counts survivors at the full hold length; ratios are vs the in-session base row at the same w; no kills")
    report("\n".join(lines), "rung iii open-loop" + (" rescored" if rescore else ""))
    log("probe table written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
