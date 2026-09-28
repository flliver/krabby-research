#!/usr/bin/env python3
"""Rung (v) driver: 5k formation on seed 3 for every sound configuration (identical 0-5k config
to the lineage anchor 2026-08-31_03-42-16/model_4999.pt), first 2k doubling as the plant
soundness smoke, then slow/fwd/step evals via experiments/eval/scenarios_morph.yaml. The control row is
the anchor itself evaluated on the base plant in the same harness. Serial, resumable
(state.json). Run AFTER STOP 2. Stdlib only.

Usage: run_formation_arms.py [--configs base,B,A10,...] (default: all 8; base trains too so
the seed-3 anchor is reproduced in-campaign as a second control sample).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
LINEAGE = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-26_2200_gated_lineage"
MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_morph.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
FLAT_RUNS = PARKOUR / "logs/rsl_rl/crab_hex_flat_walk"
EVAL_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/leg_mount_morphology/formation"
VARIANTS = REPO / "assets/variants"
STATE = HERE / "formation_state.json"
RESULTS = HERE / "RESULTS.md"
ANCHOR = str(FLAT_RUNS / "2026-08-31_03-42-16/model_4999.pt")
SEED = "3"
ITERS = 5000
SMOKE_ITERS = 2000
# 0-5k formation config of record (== run_phaseout.BASELINE == round-search BASELINE).
BASELINE = {
    "KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "1.0",
    "KRABBY_RSI_FRAC": "0.2", "KRABBY_RSI_BANK": str(LINEAGE / "rsi_bank_P0_null.npz"),
    "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_LIN_VEL_X": "0.0:0.35",
}
CONFIGS = {"base": None, "B": "splay00_axis2p5in", "A10": "splay10_axis5p5in", "A15": "splay15_axis5p5in",
           "A20": "splay20_axis5p5in", "A10+B": "splay10_axis2p5in", "A15+B": "splay15_axis2p5in",
           "A20+B": "splay20_axis2p5in"}
SCENARIOS = ("slow", "fwd", "step")
LABEL_EP_LEN = "Mean episode length"
LABEL_FAIL = "Episode_Termination/crab_failure"
LABEL_COLL = "Episode_Reward/reward_collision"


def log(msg: str) -> None:
    print(f"[formation {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def report(block: str, marker: str) -> None:
    with RESULTS.open("a") as fh:
        fh.write("\n" + block.rstrip() + f"\n>>> ENTRY {marker}\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"arms": {}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=1))


def series(log_path: Path, label: str) -> list[float]:
    pat = re.compile(re.escape(label) + r":\s*(-?[0-9.]+)")
    out = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                pass
    return out


def tail_mean(vals: list[float], n: int = 20) -> float:
    t = vals[-n:]
    return sum(t) / len(t) if t else float("nan")


def wait_isaac_clear() -> None:
    pattern = r"isaac_venv/bin/python [^ ]*(statics_battery|scripted_gait_probe_v2|eval_crab_hex_gait|rsl_rl/train)\.py"
    for _ in range(60):
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if not any(int(p) != os.getpid() for p in out):
            return
        time.sleep(10)


def plant_env(cfg: str) -> dict:
    ev = dict(BASELINE)
    if CONFIGS[cfg]:
        ev["KRABBY_HEX_USD_PATH"] = str(VARIANTS / f"crab_simple__{CONFIGS[cfg]}.usda")
    else:
        # "base" = the 2026-08-20 golden geometry. Explicit since 2026-09-09: the config default
        # (assets/crab.usda) is now the A15+B plant of record.
        ev["KRABBY_HEX_USD_PATH"] = str(VARIANTS / "crab_simple__splay00_axis5p5in.usda")
    return ev


def latest_run_ckpt(t0: float) -> str | None:
    run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
    if run_dir.stat().st_mtime < t0:
        return None
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(models[-1]) if models else None


def train(cfg: str, tag: str) -> tuple[str | None, str, Path]:
    """Returns (ckpt, status, log). status in {ok, dead_plant, nan, infra}."""
    t0 = time.time()
    log_path = HERE / "formation_logs" / f"{tag}_train.log"
    log_path.parent.mkdir(exist_ok=True)
    cmd = [PY, str(TRAIN), "--task", "Isaac-Crab-Hex-Flat-Walk-v0", "--headless",
           "--num_envs", "256", "--seed", SEED, "--max_iterations", str(ITERS)]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update(plant_env(cfg))
    log(f"{tag}: training launched ({ITERS} iterations, seed {SEED}, plant {CONFIGS[cfg] or 'golden'})")
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT)
    status = "ok"
    try:
        while proc.poll() is None:
            time.sleep(120)
            try:
                if time.time() - log_path.stat().st_mtime > 900:
                    log(f"{tag}: no log progress for 15 min — infra death")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "infra", log_path
                txt = log_path.read_text(errors="ignore")
                if re.search(r"Mean (reward|value_function|surrogate):\s*nan", txt[-20000:], re.I) or "Traceback" in txt[-20000:]:
                    log(f"{tag}: NaN in the training log — unsound plant")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "nan", log_path
                ep = series(log_path, LABEL_EP_LEN)
                # Smoke at 2k: a dead plant shows no episode-length growth.
                if len(ep) >= SMOKE_ITERS and tail_mean(ep, 100) < 1.2 * tail_mean(ep[:100], 100):
                    log(f"{tag}: DEAD PLANT — episode length {tail_mean(ep[:100], 100):.0f} -> "
                        f"{tail_mean(ep, 100):.0f} by 2k")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "dead_plant", log_path
            except OSError:
                pass
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, "infra", log_path
    ckpt = latest_run_ckpt(t0)
    if ckpt is None or int(Path(ckpt).stem.split("_")[1]) < ITERS - 100:
        return None, "infra", log_path
    return ckpt, status, log_path


def evaluate(ckpt: str, cfg: str, scenario: str, tag: str) -> dict | None:
    sid = f"{scenario}__{cfg.replace('+', 'p')}"
    out_root = EVAL_ROOT / tag
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update({k: v for k, v in BASELINE.items() if k in ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2",
                                                          "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W", "KRABBY_APEX_W")})
    env.pop("KRABBY_HEX_USD_PATH", None)   # read at config import time: must be in the process env
    if CONFIGS[cfg]:
        env["KRABBY_HEX_USD_PATH"] = str(VARIANTS / f"crab_simple__{CONFIGS[cfg]}.usda")
    lp = HERE / "formation_logs" / f"{tag}_{sid}_eval.log"
    lp.parent.mkdir(exist_ok=True)
    for attempt in range(3):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST), "--scenario", sid,
                                "--checkpoint", ckpt, "--no-plot", "--save-raw", "--output-root", str(out_root)],
                               cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
            except subprocess.TimeoutExpired:
                pass
        d = out_root / sid / "seed001"
        if d.exists():
            latest = max(d.iterdir(), key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime >= t0 and (latest / "scenario_metrics.json").exists():
                return json.loads((latest / "scenario_metrics.json").read_text())["aggregate"]
        log(f"{tag}/{sid}: no fresh metrics — retry {attempt + 1}")
        wait_isaac_clear(); time.sleep(120)
    return None


def summarize(a: dict | None) -> dict:
    if not a:
        return {}
    fc = a.get("fall_classes", {})
    poly = a.get("support_polygon", {})
    return {"tripod": (a.get("tripod_score") or {}).get("median"),
            "completion": a.get("schedule_completion_rate"),
            "tracking": (a.get("tracking_ratio") or {}).get("median"),
            "falls": a.get("termination_reasons", {}).get("fall", 0),
            "n": a.get("n_episodes"),
            # share of FALLS that are pitch-forward (the running 2026-09-02 instance divided by all
            # episodes; assemble_stop3.py recomputes from scenario_metrics.json)
            "pitch_fwd_share": (fc.get("pitch_fwd", 0) / max(1, sum(v for k, v in fc.items() if k != "none")))
            if any(k != "none" for k in fc) else None,
            "pitch_max_abs_p50": ((a.get("orientation") or {}).get("pitch_max_abs") or {}).get("median"),
            "prefall_tip_p25": (poly.get("prefall_tip_angle_fwd_deg") or {}).get("p25"),
            "walk_tip_p50": (poly.get("walking_tip_angle_fwd_deg") or {}).get("median")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    cfgs = [c for c in args.configs.split(",") if c]
    st = load_state()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    # control row: the anchor on the base plant in this harness
    if "control" not in st["arms"]:
        st["arms"]["control"] = {"ckpt": ANCHOR, "status": "ok", "evals": {}}
    def run_evals(name: str, arm: dict) -> None:
        if not arm.get("ckpt"):
            return
        cfg_ = "base" if name == "control" else name
        for sc in SCENARIOS:
            if sc in arm["evals"]:
                continue
            log(f"{name}: eval {sc} launched")
            arm["evals"][sc] = summarize(evaluate(arm["ckpt"], cfg_, sc, name.replace("+", "p")))
            save_state(st)
            log(f"{name}: eval {sc} -> {arm['evals'][sc]}")

    run_evals("control", st["arms"]["control"])
    for cfg in cfgs:
        arm = st["arms"].setdefault(cfg, {"ckpt": None, "status": None, "evals": {}, "smoke": {}})
        if arm["ckpt"] is None and arm["status"] not in ("dead_plant", "nan"):
            ckpt, status, lp = train(cfg, f"form_{cfg.replace('+', 'p')}")
            if ckpt is None and status == "infra":
                log(f"{cfg}: infra death — one retry"); wait_isaac_clear(); time.sleep(120)
                ckpt, status, lp = train(cfg, f"form_{cfg.replace('+', 'p')}_r2")
            arm.update(ckpt=ckpt, status=status)
            if lp.exists():
                fail = series(lp, LABEL_FAIL); coll = series(lp, LABEL_COLL); ep = series(lp, LABEL_EP_LEN)
                arm["smoke"] = {"fail_2k": tail_mean(fail[:SMOKE_ITERS], 100) if fail else None,
                                "fail_5k": tail_mean(fail, 100) if fail else None,
                                "coll_2k": tail_mean(coll[:SMOKE_ITERS], 100) if coll else None,
                                "ep_len_2k": tail_mean(ep[:SMOKE_ITERS], 100) if ep else None}
            save_state(st)
            log(f"{cfg}: training finished -> {arm['status']} ({arm['ckpt']})")
            wait_isaac_clear(); time.sleep(30)
        run_evals(cfg, arm)
    ctrl = st["arms"]["control"]["evals"]
    lines = ["## RUNG (v) — 5k formation on seed 3 (variant plant) vs the lineage anchor (control, base plant)",
             "| arm | status | smoke fail@2k / coll@2k | slow tripod / compl / track (ratio to control) | fwd falls/n | fwd pitch_max p50 | step falls/n (ratio) | pitch-fwd share | prefall tip p10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    def r(x, y):
        return "—" if x is None or not y else f"{x / y:.2f}×"
    for name, arm in st["arms"].items():
        e = arm.get("evals", {}); s, f, p = e.get("slow", {}), e.get("fwd", {}), e.get("step", {})
        sm = arm.get("smoke", {})
        cs = ctrl.get("slow", {}); cp = ctrl.get("step", {})
        if not arm.get("ckpt"):
            lines.append(f"| {name} | {arm.get('status')} | {sm.get('fail_2k')} / {sm.get('coll_2k')} | UNSOUND — not evaluated | | | | | |")
            continue
        step_ratio = r(p.get("falls"), cp.get("falls")) if p else "—"
        lines.append(
            f"| {name} | {arm.get('status')} | {sm.get('fail_2k') if sm else '—'} / {sm.get('coll_2k') if sm else '—'} | "
            f"{s.get('tripod')} / {s.get('completion')} / {s.get('tracking')} ({r(s.get('tripod'), cs.get('tripod'))}, "
            f"{r(s.get('completion'), cs.get('completion'))}, {r(s.get('tracking'), cs.get('tracking'))}) | "
            f"{f.get('falls')}/{f.get('n')} | {f.get('pitch_max_abs_p50')} | {p.get('falls')}/{p.get('n')} ({step_ratio}) | "
            f"{f.get('pitch_fwd_share')} | {f.get('prefall_tip_p25')} |")
    lines.append("- control = anchor 2026-08-31_03-42-16/model_4999 on the base plant; 'base' arm = fresh seed-3 5k on the base plant (reproduction sample); 0.85× reference line on the canary ratios; 5k is a formation snapshot, not a lineage result")
    report("\n".join(lines), "rung v formation")
    log("formation table written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
