#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Obstacle-exposure campaign orchestrator (PLAN H, user-approved 2026-09-03).

Fork of the phase-out orchestrator (parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-31_1414_gait_income_phaseout/
run_phaseout.py): same train / canary / obstacle-eval / watchdog / systemd infrastructure,
new phases (plan: /home/nickmagus/.claude/plans/wiggly-gathering-sloth.md):

  a         A0 timeline probes (20k + 30k heads, in the lineage's ACTUAL training config:
            narrow recal2b2, window bank) -> E4 trench-only eval (30k, recal2b2 @ 0.00-0.05)
            -> E1 widened baselines B10/B20/B30 (recal2b2w @ 0.20-0.70)
            -> await_user_a  (push-notify; relaunch with --start-smoke)
  smoke     unarmed + armed 200-iter smokes (telemetry keys present, knobs bite)
  wave1     C0 control, C3, C1, C4, C2, C5 from the 20k head on recal2b2w (5k iters each,
            flat canary + fixed obstacle eval + exposure telemetry) -> await_user_c
  wave2     pre-registered pairings C1+C4, C1+C2 (+C5 when it passed its safety gates),
            strongest single + C5, triple if C1+C4 hits coverage but misses goals_passed,
            seed-2 of the winner -> await_user_d  (--start-d)
  d         replay 5k->30k with the winner armed on recal2b2w (seed 3), then seed 2 -> done

Stdlib only; resumable via state.json; every GPU job is a subprocess carrying the
KRABBY_* env-var stack of that run (unset = bit-identical lineage behaviour).
"""
from __future__ import annotations

import json
import math
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
PLANG = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-31_1414_gait_income_phaseout"
MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
PROBE = HERE / "training_timeline_probe.py"
FLAT_RUNS = PARKOUR / "logs/rsl_rl/crab_hex_flat_walk"
EVAL_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/obstacle_exposure"
STATE = HERE / "state.json"
REPORT = HERE / "REPORT.md"
CHANGELOG = HERE / "CHANGELOG.md"

# --- phase-out lineage heads (seed 3; PLAN G REPORT bake blocks) -----------------
ANCHOR = str(FLAT_RUNS / "2026-08-31_03-42-16/model_4999.pt")             # 5k, formation done
HEADS = {
    "10k": str(FLAT_RUNS / "2026-09-01_02-22-26/model_9998.pt"),          # r1 bake PASS head
    "20k": str(FLAT_RUNS / "2026-09-01_15-10-31/model_19996.pt"),         # r3 bake head (base of wave 1)
    "30k": str(FLAT_RUNS / "2026-09-02_00-42-53/model_29994.pt"),         # reference of record
}
# RSI bank ACTIVE in each 5k window (adopted at the previous bake): seg1 = P0-null,
# seg2 = pg_r1, seg3 = pg_r2 (the 20k head's window), seg4 = pg_r3 (wave-1 window),
# seg5 = pg_r4 (the 30k head's window).
BANKS = {
    1: str(LINEAGE / "rsi_bank_P0_null.npz"),
    2: str(PLANG / "rsi_bank_pg_r1.npz"),
    3: str(PLANG / "rsi_bank_pg_r2.npz"),
    4: str(PLANG / "rsi_bank_pg_r3.npz"),
    5: str(PLANG / "rsi_bank_pg_r4.npz"),
}
SEED = "3"
CONFIRM_SEED = "2"
STEPS_PER_ITER = 24
RAMP_ITERS = 1000
ARM_ITERS = 5000
SMOKE_ITERS = 200
BACKSTOP_PRINTS = 2500        # mid-segment backstop reads the log from here on (22.5k)
EPS = 0.001

BASELINE = {
    "KRABBY_CLOCK_W": "1.0",
    "KRABBY_APEX_W": "1.0",
    "KRABBY_RSI_FRAC": "0.2",
    "KRABBY_RSI_BANK": BANKS[1],
    "KRABBY_FLAT_TERRAIN_MODE": "light",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    "KRABBY_TRACK_SIGMA2": "0.1",
    "KRABBY_TRACK_L1_W": "-1.0",
    "KRABBY_LIN_VEL_X": "0.0:0.35",
}
SCHEDULE = {
    1: {
        "KRABBY_YAW_W": "0.2",
        "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5",
        "KRABBY_FLAT_TERRAIN_GEOM": "recal2b2",
        "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70",
        "KRABBY_FLAT_TERRAIN_CURRICULUM": "1",
        "KRABBY_TERRAIN_PROMOTE": "0.45:0.25",
        "KRABBY_DR_PUSH": "0.5",
        "KRABBY_DR_MASS": "-0.5:1.5",
        "KRABBY_DR_COM": "0.01",
        "KRABBY_EDGE_W": "-0.3",
        "KRABBY_STUMBLE_W": "-1.0",
        "KRABBY_COLLISION_W": "-2.0",
    },
    2: {
        "KRABBY_CLEARANCE_W": "0.9",
        "KRABBY_FOOT_CLEAR_FLAT": "1",
        "KRABBY_FOOT_CLEAR_W": "1.0",
        "KRABBY_FOOT_CLEAR_MIN": "0.03",
        "KRABBY_SWING_MIN_CLEAR_W": "-0.4",
        "KRABBY_HEADING": "-1.2:1.2",
        "KRABBY_GOAL_VEL_W": "0.75",
    },
}
CANARY_KEEP = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W",
               "KRABBY_APEX_W", "KRABBY_AIRTIME_W", "KRABBY_STRIDE_W",
               "KRABBY_HEADING", "KRABBY_HEADING_STIFFNESS", "KRABBY_ACTION_SCALE")
TERRAIN_PASS = ("KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_FLAT_TERRAIN_DIFF",
                "KRABBY_CORRIDOR_HALF_WIDTH", "KRABBY_STONE_WIDTH")
TERMS = {
    "reward_clock_schedule": "KRABBY_CLOCK_W",
    "reward_clock_swing_apex": "KRABBY_APEX_W",
    "reward_feet_air_time_positive": "KRABBY_AIRTIME_W",
    "reward_stride_length": "KRABBY_STRIDE_W",
}
W_20K = {"KRABBY_CLOCK_W": "0.5", "KRABBY_APEX_W": str(EPS), "KRABBY_AIRTIME_W": str(EPS), "KRABBY_STRIDE_W": str(EPS)}
W_30K = {k: str(EPS) for k in W_20K}
CAMPAIGN_GEOM = {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2w"}
OBST_EVAL = {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2w", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70"}
E4_EVAL = {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2", "KRABBY_FLAT_TERRAIN_DIFF": "0.00:0.05"}
LEVERS = {
    "C1": {"KRABBY_STAND_FRAC": "0.2"},
    "C3": {"KRABBY_SPAWN_OFFSET": "2.0"},
    "C4": {"KRABBY_SPAWN_SPREAD": "1.0:11.0:0.5"},
    "C5": {"KRABBY_RSI_SPAWN_FIX": "1"},
}
WAVE1_ORDER = ["C0", "C3", "C1", "C4", "C2", "C5"]
N_OBST = 6
TARGET = {"reach_obst_frac": 0.80, "goals_passed_mean": 2.0, "obst_coverage_3": 0.50,
          "obst_coverage_6": 0.20, "field_frac_mean": 0.20}
EXPO_BASE = ["reach_edge_frac", "reach_obst_frac", "field_frac_mean", "field_steps_mean",
             "goals_passed_mean"] + [f"obst_coverage_{k}" for k in range(1, N_OBST + 1)]
EXPO_KEYS = EXPO_BASE + [f"{k}_rsi" for k in EXPO_BASE] + [
    "crab_failure_flat", "crab_failure_obst", "crab_failure_obst_rsi", "crab_failure_obst_spread",
    "crab_failure_hazard_flat", "crab_failure_hazard_obst", "ep_steps_flat", "ep_steps_obst", "ep_steps_obst_spread",
    "spread_frac_actual", "rsi_frac_actual", "terrain_levels", "how_far_from_start_point",
    "current_goal_idx"]
REPLAY_SCHEDULE = {  # window s (5k*s -> 5k*(s+1)): start weights, ramps (term, w0, w1)
    1: ({"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "1.0", "KRABBY_AIRTIME_W": "0.8", "KRABBY_STRIDE_W": "0.5"},
        [("reward_clock_swing_apex", 1.0, 0.5), ("reward_feet_air_time_positive", 0.8, 0.4),
         ("reward_stride_length", 0.5, 0.25)]),
    2: ({"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "0.5", "KRABBY_AIRTIME_W": "0.4", "KRABBY_STRIDE_W": "0.25"},
        [("reward_clock_swing_apex", 0.5, EPS), ("reward_feet_air_time_positive", 0.4, EPS),
         ("reward_stride_length", 0.25, EPS)]),
    3: ({"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": str(EPS), "KRABBY_AIRTIME_W": str(EPS), "KRABBY_STRIDE_W": str(EPS)},
        [("reward_clock_schedule", 1.0, 0.5)]),
    4: ({"KRABBY_CLOCK_W": "0.5", "KRABBY_APEX_W": str(EPS), "KRABBY_AIRTIME_W": str(EPS), "KRABBY_STRIDE_W": str(EPS)},
        [("reward_clock_schedule", 0.5, 0.2)]),
    5: ({"KRABBY_CLOCK_W": "0.2", "KRABBY_APEX_W": str(EPS), "KRABBY_AIRTIME_W": str(EPS), "KRABBY_STRIDE_W": str(EPS)},
        [("reward_clock_schedule", 0.2, EPS)]),
}


# ------------------------------------------------------------------ infrastructure
def log(msg: str) -> None:
    print(f"[exposure {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def report(block: str, marker: str) -> None:
    with REPORT.open("a") as fh:
        fh.write(block.rstrip() + f"\n>>> ENTRY {marker}\n\n")


def changelog(row: str) -> None:
    with CHANGELOG.open("a") as fh:
        fh.write(row.rstrip() + "\n")


def notify(msg: str) -> None:
    changelog(f"> NOTIFY: {msg}")
    log(f"NOTIFY: {msg}")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"phase": "a", "run_no": 0, "a": {}, "smoke": {}, "arms": {}, "wave2": {},
            "t_star": None, "winner": None, "d": {}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def series(log_path: Path, label: str) -> list[float]:
    pat = re.compile(re.escape(label) + r":\s*(-?[0-9.]+(?:e-?\d+)?)")
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


def fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.3f}"
    return str(v)


def latest_run_ckpt() -> str:
    run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(models[-1])


def _wait_isaac_clear() -> None:
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", "rsl_rl/train.py|eval_crab_hex_gait|training_timeline_probe"],
                          capture_output=True).returncode != 0:
            return
        time.sleep(10)


def lineage_stack(window: int, weights: dict, bank: str | None = None, extra: dict | None = None) -> dict:
    """Env stack of the lineage in 5k-window ``window`` (elements placed <= window)."""
    ev = dict(BASELINE)
    for r in sorted(SCHEDULE):
        if r <= window:
            ev.update(SCHEDULE[r])
    ev["KRABBY_RSI_BANK"] = bank or BANKS[min(max(window, 1), 5)]
    ev.update(weights)
    if extra:
        ev.update(extra)
    return ev


def campaign_stack(extra: dict | None = None) -> dict:
    """Wave-1/2 stack: 20k head's window-4 lineage config + widened corridors + lever."""
    ev = lineage_stack(4, W_20K, BANKS[4], CAMPAIGN_GEOM)
    if extra:
        ev.update(extra)
    return ev


def _subenv(env_vars: dict) -> dict:
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    for k in list(env):
        if k.startswith("KRABBY_"):
            del env[k]
    env.update({k: str(v) for k, v in env_vars.items()})
    return env


def train(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None, seed: str = SEED,
          live_check=None) -> tuple[str | None, Path]:
    t0 = time.time()
    log_path = HERE / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", "Isaac-Crab-Hex-Flat-Walk-v0", "--headless",
           "--num_envs", "256", "--seed", seed, "--max_iterations", str(iters)]
    if resume_ckpt:
        cmd += ["--resume", "--checkpoint", resume_ckpt]
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=_subenv(env_vars), stdout=fh, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            time.sleep(120)
            try:
                if time.time() - log_path.stat().st_mtime > 900:
                    log(f"{tag}: no log progress for 15 min — treating as infra death")
                    proc.kill()
                    try:
                        proc.wait(timeout=60)
                    except Exception:
                        pass
                    return None, log_path
            except OSError:
                pass
            if live_check is not None:
                try:
                    why = live_check(log_path)
                except Exception:
                    why = None
                if why:
                    log(f"{tag}: LIVE ABORT — {why}")
                    proc.kill()
                    try:
                        proc.wait(timeout=60)
                    except Exception:
                        pass
                    return "ABORTED", log_path
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, log_path
    if proc.returncode == 0:
        return latest_run_ckpt(), log_path
    try:  # teardown-crash salvage: trust the on-disk final checkpoint over exit code
        base = int(Path(resume_ckpt).stem.split("_")[1]) if resume_ckpt else 0
        run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
        models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if models and run_dir.stat().st_mtime >= t0:
            idx = int(models[-1].stem.split("_")[1])
            if idx >= base + iters - 100:
                log(f"{tag}: rc != 0 but final checkpoint model_{idx} exists — salvaged")
                return str(models[-1]), log_path
    except Exception:
        pass
    return None, log_path


def train_retry(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None,
                seed: str = SEED, live_check=None) -> tuple[str | None, Path]:
    ckpt, tlog = train(env_vars, tag, iters, resume_ckpt, seed, live_check)
    for retry in (2, 3):
        if ckpt is not None:
            break
        txt = tlog.read_text(errors="ignore") if tlog.exists() else ""
        if "Traceback" in txt or "Error executing job" in txt:
            break
        log(f"{tag}: died without a Python error (boot balloon?) — infra retry {retry - 1}")
        _wait_isaac_clear()
        time.sleep(120)
        ckpt, tlog = train(env_vars, f"{tag}_r{retry}", iters, resume_ckpt, seed, live_check)
    return ckpt, tlog


def _eval_run(ckpt: str, ev: dict, lp: Path, out_root: Path, timeout: int) -> dict | None:
    t0 = time.time()
    with lp.open("a") as fh:
        try:
            subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST),
                            "--scenario", "flat_walk_slow_v2", "--checkpoint", ckpt, "--no-plot",
                            "--output-root", str(out_root)],
                           cwd=PARKOUR, env=_subenv(ev), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    seed_dir = out_root / "flat_walk_slow_v2" / "seed001"
    if not seed_dir.exists():
        return None
    d = max(seed_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if d.stat().st_mtime >= t0 and (d / "scenario_metrics.json").exists():
        return json.loads((d / "scenario_metrics.json").read_text())["aggregate"]
    return None


def _med(x):
    if isinstance(x, dict):
        return x.get("median") if x.get("median") is not None else x.get("mean")
    return x


def canary(ckpt: str, env_vars: dict, tag: str) -> dict:
    ev = {k: env_vars[k] for k in CANARY_KEEP if k in env_vars}
    lp = HERE / f"{tag}_canary.log"
    for _attempt in range(4):
        a = _eval_run(ckpt, ev, lp, EVAL_ROOT, 3600)
        if a is not None:
            creep = ((a.get("tracking_by_hold") or {}).get("creep") or {})
            return {"tripod": _med(a.get("tripod_score")) or 0.0,
                    "completion": a.get("schedule_completion_rate") or 0.0,
                    "tracking": _med(a.get("tracking_ratio")) or 0.0,
                    "slip": _med(a.get("slip_ratio")) or 0.0,
                    "tippy": _med(a.get("tippy_tap_fraction")) or 0.0,
                    "creep_vx": _med(creep.get("achieved_vx")) if creep else float("nan")}
        log(f"{tag}: canary produced no fresh metrics (boot balloon?) — cool-down, retry")
        _wait_isaac_clear()
        time.sleep(120)
    raise RuntimeError(f"{tag}: canary produced no fresh metrics after 4 attempts")


def obstacle_eval(ckpt: str, env_vars: dict, tag: str, terrain: dict) -> dict:
    """100-episode obstacle eval on an explicit terrain config (geometry + difficulty +
    corridor overrides pass through; all-obstacle tiles, frozen levels)."""
    ev = {k: env_vars[k] for k in CANARY_KEEP if k in env_vars}
    ev.update({k: env_vars[k] for k in TERRAIN_PASS if k in env_vars})
    ev.update(terrain)
    ev.update({"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0"})
    lp = HERE / f"{tag}_obst.log"
    for _attempt in range(3):
        a = _eval_run(ckpt, ev, lp, Path(str(EVAL_ROOT) + "_obst"), 5400)
        if a is not None:
            return {"completion": a.get("schedule_completion_rate") or 0.0,
                    "tripod": _med(a.get("tripod_score")) or 0.0,
                    "falls": (a.get("termination_reasons") or {}).get("fall", 0),
                    "terrain": dict(terrain)}
        log(f"{tag}: obstacle eval produced no fresh metrics — cool-down, retry")
        _wait_isaac_clear()
        time.sleep(120)
    raise RuntimeError(f"{tag}: obstacle eval produced no fresh metrics after 3 attempts")


def probe(ckpt: str, env_vars: dict, tag: str, steps: int = 4000, num_envs: int = 64) -> dict:
    out = HERE / f"probe_{tag}"
    lp = HERE / f"{tag}_probe.log"
    for _attempt in range(3):
        summary = out / "summary.json"
        if summary.exists():
            summary.unlink()
        with lp.open("a") as fh:
            proc = subprocess.Popen([PY, str(PROBE), "--headless", "--checkpoint", ckpt, "--num_envs", str(num_envs),
                                     "--steps", str(steps), "--out", str(out), "--label", tag],
                                    cwd=PARKOUR, env=_subenv(env_vars), stdout=fh, stderr=subprocess.STDOUT)
            t0 = time.time()
            while proc.poll() is None:
                time.sleep(20)
                # the probe's work is done once summary.json exists; Isaac teardown can hang
                if summary.exists() and time.time() - summary.stat().st_mtime > 60:
                    proc.kill()
                    break
                # a crashed probe also hangs in teardown: do not wait 90 min for it
                if "Traceback" in lp.read_text(errors="ignore"):
                    time.sleep(30)
                    proc.kill()
                    break
                if time.time() - t0 > 5400:
                    proc.kill()
                    break
            try:
                proc.wait(timeout=120)
            except Exception:
                pass
        if summary.exists():
            return json.loads(summary.read_text())
        txt = lp.read_text(errors="ignore")
        if "Traceback" in txt:
            raise RuntimeError(f"{tag}: probe crashed — see {lp}")
        log(f"{tag}: probe produced no summary — cool-down, retry")
        _wait_isaac_clear()
        time.sleep(120)
    raise RuntimeError(f"{tag}: probe produced no summary after 3 attempts")


def exposure_from_log(tlog: Path, n: int = 20) -> dict:
    out = {k: tail_mean(series(tlog, f"Metrics/base_parkour/{k}"), n) for k in EXPO_KEYS}
    out["stand_frac_actual"] = tail_mean(series(tlog, "Metrics/base_velocity/stand_frac_actual"), n)
    out["failure_all"] = tail_mean(series(tlog, "Episode_Termination/crab_failure"), n)
    out["eplen"] = tail_mean(series(tlog, "Mean episode length"), n)
    out["mean_reward"] = tail_mean(series(tlog, "Mean reward"), n)
    out["vloss"] = tail_mean(series(tlog, "Mean value_function"), 200)
    out["track"] = tail_mean(series(tlog, "Episode_Reward/track_lin_vel_xy_exp"), n)
    out["prints"] = len(series(tlog, "Mean reward"))
    return out


def _obst_ceiling(ctrl: dict | None, hazard_norm: bool) -> tuple[str, float]:
    """Obstacle-tile catastrophe ceiling: share max(0.80, C0 + 0.15); for horizon-changing arms
    the same ratio applied to the per-1000-step hazard (share x horizon is not comparable)."""
    c_obst = (ctrl or {}).get("crab_failure_obst", float("nan"))
    ceil_share = max(0.80, (c_obst + 0.15) if c_obst == c_obst else 0.0)
    if not hazard_norm:
        return "crab_failure_obst", ceil_share
    c_h = (ctrl or {}).get("crab_failure_hazard_obst", float("nan"))
    if not (c_h == c_h and c_obst == c_obst):
        return "crab_failure_hazard_obst", float("inf")
    return "crab_failure_hazard_obst", c_h * ceil_share / max(c_obst, 0.05)


def live_backstop(ctrl: dict | None, min_prints: int = BACKSTOP_PRINTS, hazard_norm: bool = False):
    """Mid-segment backstop: flat-tile failure > C0 + 0.10 or obstacle-tile failure above the
    catastrophe ceiling, sustained over the last 100 prints after ``min_prints``.
    ``hazard_norm`` (arms that change the episode length, e.g. C2 at 70 s): both checks run on
    the per-1000-step hazard with the +0.10-share allowance mapped into hazard space -- a raw
    share rises with the horizon at an unchanged hazard (bug found 2026-09-04 02:46: C2 was
    aborted at share 0.62 while its hazard was 0.34 vs the control's 0.59)."""
    c_flat = (ctrl or {}).get("crab_failure_flat", float("nan"))
    c_hf = (ctrl or {}).get("crab_failure_hazard_flat", float("nan"))
    obst_key, ceil_obst = _obst_ceiling(ctrl, hazard_norm)
    if hazard_norm:
        flat_key = "crab_failure_hazard_flat"
        allow_flat = c_hf * (1.0 + 0.10 / max(c_flat, 0.05)) if (c_hf == c_hf and c_flat == c_flat) else float("nan")
        unit = "/1k steps"
    else:
        flat_key = "crab_failure_flat"
        allow_flat = c_flat + 0.10 if c_flat == c_flat else float("nan")
        unit = ""

    def check(log_path: Path) -> str | None:
        fl = series(log_path, f"Metrics/base_parkour/{flat_key}")
        ob = series(log_path, f"Metrics/base_parkour/{obst_key}")
        if len(fl) < min_prints:
            return None
        f100, o100 = tail_mean(fl, 100), tail_mean(ob, 100)
        if allow_flat == allow_flat and f100 > allow_flat:
            return f"flat-tile failure {f100:.2f}{unit} > allowance {allow_flat:.2f}{unit} (control {c_flat:.2f} share / {c_hf:.2f}/1k)"
        if o100 > ceil_obst:
            return f"obstacle-tile failure {o100:.2f}{unit} > ceiling {ceil_obst:.2f}{unit}"
        return None

    return check


def phaseout_spec(ramps: list[tuple[str, float, float]]) -> str:
    return ",".join(f"{t}:{w0}:{w1}:0:{RAMP_ITERS * STEPS_PER_ITER}" for t, w0, w1 in ramps)


# ------------------------------------------------------------------ judging
def exposure_verdict(ex: dict, ctrl: dict | None) -> tuple[str, list[str]]:
    """PASS / PARTIAL / FAIL on the plan's exposure target (non-RSI obstacle-tile episodes)."""
    notes = []

    def ok(key: str) -> bool:
        v, thr = ex.get(key, float("nan")), TARGET[key]
        c = (ctrl or {}).get(key, float("nan"))
        hit = (v == v) and (v >= thr or (c == c and c >= 0.10 and v >= 3.0 * c))
        notes.append(f"{key} {fmt(v)} {'>=' if hit else '<'} {thr}" + (f" (C0 {fmt(c)})" if c == c else ""))
        return hit

    first = ok("reach_obst_frac")
    mid = [ok(k) for k in ("goals_passed_mean", "obst_coverage_3", "obst_coverage_6", "field_frac_mean")]
    if first and all(mid):
        return "PASS", notes
    if first:
        return "PARTIAL", notes
    return "FAIL", notes


def safety_verdict(rec: dict, ctrl: dict, hazard_norm: bool) -> list[str]:
    """Return the list of violated gait-safety gates (empty = safe)."""
    bad = []
    can, c_can = rec["canary"], ctrl["canary"]
    if can["completion"] < c_can["completion"] - 0.05:
        bad.append(f"canary completion {fmt(can['completion'])} < C0 {fmt(c_can['completion'])} - 0.05")
    if can["tripod"] < c_can["tripod"] - 0.05:
        bad.append(f"tripod {fmt(can['tripod'])} < C0 {fmt(c_can['tripod'])} - 0.05")
    if can["tracking"] < c_can["tracking"] - 0.05:
        bad.append(f"tracking {fmt(can['tracking'])} < C0 {fmt(c_can['tracking'])} - 0.05")
    if can["creep_vx"] == can["creep_vx"] and c_can["creep_vx"] == c_can["creep_vx"] \
            and can["creep_vx"] < c_can["creep_vx"] - 0.02:
        bad.append(f"creep speed {fmt(can['creep_vx'])} < C0 {fmt(c_can['creep_vx'])} - 0.02")
    ex, cex = rec["exposure"], ctrl["exposure"]
    if hazard_norm:
        c_h, c_share = cex["crab_failure_hazard_flat"], cex["crab_failure_flat"]
        allow = c_h * (1.0 + 0.05 / max(c_share, 0.05)) if c_h == c_h else float("nan")
        if allow == allow and ex["crab_failure_hazard_flat"] > allow:
            bad.append(f"flat-tile failure hazard {fmt(ex['crab_failure_hazard_flat'])}/1k steps > "
                       f"allowance {fmt(allow)} (C0 {fmt(c_h)}, +0.05-share equivalent)")
    else:
        if ex["crab_failure_flat"] > cex["crab_failure_flat"] + 0.05:
            bad.append(f"flat-tile failure {fmt(ex['crab_failure_flat'])} > C0 {fmt(cex['crab_failure_flat'])} + 0.05")
    # degenerate-mode watchdogs (phase-out orchestrator thresholds)
    if can["completion"] >= 0.90 and can["tracking"] < 0.40:
        bad.append(f"CREEP signature: completion {fmt(can['completion'])} with tracking {fmt(can['tracking'])}")
    if c_can.get("slip") and can.get("slip") and can["slip"] > 1.5 * c_can["slip"]:
        bad.append(f"slip {fmt(can['slip'])} > 1.5x C0 {fmt(c_can['slip'])}")
    if cex.get("vloss") == cex.get("vloss") and ex.get("vloss") == ex.get("vloss") and ex["vloss"] > 3.0 * cex["vloss"]:
        bad.append(f"value-loss spike {fmt(ex['vloss'])} > 3x C0 {fmt(cex['vloss'])}")
    if cex.get("mean_reward") and ex.get("mean_reward") == ex.get("mean_reward") \
            and ex["mean_reward"] < 0.05 * cex["mean_reward"]:
        bad.append(f"clip-floor: mean reward {fmt(ex['mean_reward'])} < 5% of C0 {fmt(cex['mean_reward'])}")
    return bad


def ceiling_verdict(rec: dict, ctrl: dict) -> str | None:
    hazard_norm = "KRABBY_EPISODE_S" in rec.get("extra", {})
    key, ceil = _obst_ceiling(ctrl["exposure"], hazard_norm)
    v = rec["exposure"].get(key, float("nan"))
    c_obst = ctrl["exposure"]["crab_failure_obst"]
    if v == v and v > ceil:
        unit = "/1k steps" if hazard_norm else ""
        return f"obstacle-tile failure {fmt(v)}{unit} > ceiling {fmt(ceil)}{unit} (share rule max(0.80, C0 {fmt(c_obst)} + 0.15)" + (", mapped to hazard)" if hazard_norm else ")")
    return None


def judge(name: str, rec: dict, ctrl: dict) -> dict:
    hazard_norm = "KRABBY_EPISODE_S" in rec["extra"]
    unsafe = safety_verdict(rec, ctrl, hazard_norm)
    ceiling = ceiling_verdict(rec, ctrl)
    if name.startswith("C5") or set(rec["extra"]) == {"KRABBY_RSI_SPAWN_FIX"}:
        rsi_ok = rec["exposure"]["crab_failure_obst_rsi"] <= ctrl["exposure"]["crab_failure_obst_rsi"] + 0.05
        verdict = "PASS" if (not unsafe and ceiling is None and rsi_ok) else "FAIL"
        return {"verdict": verdict, "kind": "hygiene", "exposure_status": "n/a", "exposure_notes": [],
                "unsafe": unsafe, "ceiling": ceiling,
                "rsi_failure_ok": rsi_ok}
    status, notes = exposure_verdict(rec["exposure"], ctrl["exposure"])
    if unsafe or ceiling:
        verdict = "FAIL"
    else:
        verdict = status
    return {"verdict": verdict, "kind": "exposure", "exposure_status": status, "exposure_notes": notes,
            "unsafe": unsafe, "ceiling": ceiling}


def wave2_eligible(rec: dict, ctrl: dict) -> tuple[bool, str]:
    """USER DECISION 2026-09-03 ("admit a lever to the pairings when it held every safety gate
    and raised reach_obst at least three-fold over C0"): PASS/PARTIAL always qualify; otherwise
    an exposure arm qualifies when no safety gate and no obstacle-tile ceiling was violated and
    reach_obst >= 3 x C0. The 0.80 first-obstacle floor stays the PASS/PARTIAL criterion."""
    v = rec.get("verdict")
    if v in ("PASS", "PARTIAL"):
        return True, f"verdict {v}"
    j = rec.get("judge") or {}
    if j.get("kind") != "exposure":
        return False, "hygiene arm (not an exposure lever)"
    if j.get("unsafe"):
        return False, "safety gate violated: " + "; ".join(j["unsafe"])
    if j.get("ceiling"):
        return False, "obstacle-tile ceiling: " + j["ceiling"]
    r = rec.get("exposure", {}).get("reach_obst_frac", float("nan"))
    c = ctrl.get("exposure", {}).get("reach_obst_frac", float("nan"))
    if r == r and c == c and r >= 3.0 * c:
        return True, f"safety held and reach_obst {r:.3f} >= 3x C0 {c:.3f}"
    return False, f"reach_obst {fmt(r)} < 3x C0 {fmt(c)}"


# ------------------------------------------------------------------ report blocks
def expo_lines(ex: dict) -> list[str]:
    cov = " ".join(f"{k}:{fmt(ex.get(f'obst_coverage_{k}', float('nan')))}" for k in range(1, N_OBST + 1))
    cov_rsi = " ".join(f"{k}:{fmt(ex.get(f'obst_coverage_{k}_rsi', float('nan')))}" for k in range(1, N_OBST + 1))
    return [
        f"- exposure (non-RSI obstacle tiles): reach_edge {fmt(ex.get('reach_edge_frac'))} | reach_obst "
        f"{fmt(ex.get('reach_obst_frac'))} | field_frac {fmt(ex.get('field_frac_mean'))} (steps "
        f"{fmt(ex.get('field_steps_mean'))}) | goals_passed {fmt(ex.get('goals_passed_mean'))}",
        f"- obst_coverage[1..6]: {cov}",
        f"- RSI episodes: reach_edge {fmt(ex.get('reach_edge_frac_rsi'))} | reach_obst {fmt(ex.get('reach_obst_frac_rsi'))} "
        f"| goals_passed {fmt(ex.get('goals_passed_mean_rsi'))} | coverage {cov_rsi}",
        f"- failure share flat {fmt(ex.get('crab_failure_flat'))} (hazard {fmt(ex.get('crab_failure_hazard_flat'))}/1k) | "
        f"obst {fmt(ex.get('crab_failure_obst'))} (hazard {fmt(ex.get('crab_failure_hazard_obst'))}/1k) | "
        f"obst-RSI {fmt(ex.get('crab_failure_obst_rsi'))} | obst-spread {fmt(ex.get('crab_failure_obst_spread'))} "
        f"(ep len {fmt(ex.get('ep_steps_obst_spread'))}) | campaign-wide {fmt(ex.get('failure_all'))}",
        f"- covariates: stand_frac_actual {fmt(ex.get('stand_frac_actual'))} | spread_frac_actual "
        f"{fmt(ex.get('spread_frac_actual'))} | rsi_frac_actual {fmt(ex.get('rsi_frac_actual'))} | terrain_levels "
        f"{fmt(ex.get('terrain_levels'))} | how_far {fmt(ex.get('how_far_from_start_point'))} | goal_idx "
        f"{fmt(ex.get('current_goal_idx'))} | ep len {fmt(ex.get('eplen'))} | prints {ex.get('prints')}",
    ]


def canary_line(can: dict) -> str:
    return (f"- flat canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | tracking "
            f"{fmt(can['tracking'])} | creep vx {fmt(can.get('creep_vx', float('nan')))} | slip {fmt(can['slip'])}")


def probe_block(tag: str, res: dict) -> list[str]:
    lines = [f"## A0 — training-timeline probe {tag}",
             f"- checkpoint: {Path(res['checkpoint']).parent.name}/{Path(res['checkpoint']).name} | episode "
             f"{res['episode_length_s']} s | resample {res['resampling_time_range']} | band {res['lin_vel_x']} | "
             f"policy std {fmt(res['policy_std_mean'])}"]
    for mode, s in res["modes"].items():
        ex, tl = s["exposure"], s["timeline"]
        lines.append(f"### {mode} ({s['episodes_pushed']} episodes, {s['steps']} steps x {s['num_envs']} envs, "
                     f"obstacle-tile share {fmt(s['obst_tile_frac'])}, terrain level {fmt(s['terrain_levels_mean'])})")
        lines += expo_lines(ex)
        prof = ", ".join(fmt(p) for p in tl["motion_profile_walking"])
        prof_c = ", ".join(fmt(p) for p in tl["motion_profile_cmd_on"])
        lines.append(f"- motion profile (walking fraction per {tl['motion_profile_bins_s']:.0f}-s bin, non-RSI): [{prof}]"
                     f" | command-on per bin: [{prof_c}] | late-motion ratio {fmt(tl.get('late_motion_ratio', float('nan')))}")
        prof_f = ", ".join(fmt(p) for p in tl.get("motion_profile_fast", []))
        lines.append(f"- fast (> {tl.get('fast_vx_threshold', 0.12)} m/s, above zero-command creep) per bin: [{prof_f}] | "
                     f"fast|cmd-on {fmt(tl.get('fast_given_cmd_on', float('nan')))} | fast|cmd-off "
                     f"{fmt(tl.get('fast_given_cmd_off', float('nan')))} | late-motion ratio (fast) "
                     f"{fmt(tl.get('late_motion_ratio_fast', float('nan')))}")
        lines.append(f"- walking fraction all {fmt(tl['walking_frac_all'])} / non-RSI {fmt(tl['walking_frac_non_rsi'])} | "
                     f"cmd-on fraction {fmt(tl['cmd_on_frac'])} | walking|cmd-on {fmt(tl['walking_given_cmd_on'])} | "
                     f"walking|cmd-off {fmt(tl['walking_given_cmd_off'])} | achieved vx (cmd-on) "
                     f"{fmt(tl['mean_root_vx_cmd_on'])} | vx while walking {fmt(tl['mean_root_vx_walking'])}")
    return lines


def derive_t_star(res20: dict) -> tuple[int, str]:
    """C2 horizon: (6.5 m + 0.5 m) / (0.8 x A0 achieved speed), rounded UP to 10 s, [40, 120]."""
    tl = (res20["modes"].get("stochastic") or next(iter(res20["modes"].values())))["timeline"]
    v = tl.get("mean_root_vx_cmd_on", float("nan"))
    if not (v == v) or v <= 0.03:
        return 60, f"achieved speed unusable ({fmt(v)}) -> plan default 60 s"
    t = math.ceil(7.0 / (0.8 * v) / 10.0) * 10
    t = int(min(120, max(40, t)))
    return t, f"7.0 m / (0.8 x {v:.3f} m/s) = {7.0 / (0.8 * v):.0f} s -> {t} s"


# ------------------------------------------------------------------ phases
def phase_a(st: dict) -> None:
    a = st["a"]
    # A0 probes: the lineage's ACTUAL training config (narrow recal2b2, window bank)
    for key, weights, window in (("20k", W_20K, 3), ("30k", W_30K, 5)):
        if f"probe_{key}" in a:
            continue
        st["run_no"] += 1
        tag = f"a{st['run_no']:03d}_probe_{key}"
        ev = lineage_stack(window, weights)
        log(f"{tag}: A0 timeline probe of the {key} head (window {window} config, bank {Path(ev['KRABBY_RSI_BANK']).name})")
        res = probe(HEADS[key], ev, tag)
        a[f"probe_{key}"] = res
        save_state(st)
        report("\n".join(probe_block(f"{key} head", res)), f"a0 probe {key}")
    # systematic-late-motion check (stochastic 20k): last bin > 2x the mean of the others
    tl = a["probe_20k"]["modes"]["stochastic"]["timeline"]
    ratio = tl.get("late_motion_ratio", float("nan"))
    systematic = ratio == ratio and ratio > 2.0
    if st["t_star"] is None:
        st["t_star"], why = derive_t_star(a["probe_20k"])
        a["t_star_why"] = why
        save_state(st)
        report("\n".join([
            "## A0 — verdicts",
            f"- motion timing: late-motion ratio {fmt(ratio)} -> "
            + ("**SYSTEMATIC late motion — scheduler/config defect suspected; arms NOT started**" if systematic
               else "motion is spread over random slots (no episode clock); exposure levers apply"),
            f"- C2 horizon T* = {st['t_star']} s ({why})",
        ]), "a0 verdicts")
    if systematic:
        st["phase"] = "await_user_a"
        save_state(st)
        notify("A0 found SYSTEMATIC late motion (last-bin walking > 2x the rest) — locate the scheduler defect "
               "before any arm; campaign PAUSED")
        return
    # E4 trench-only (30k head, narrow recal2b2 at difficulty ~0)
    if "e4" not in a:
        st["run_no"] += 1
        tag = f"a{st['run_no']:03d}_e4_trench"
        ev = lineage_stack(5, W_30K)
        a["e4"] = obstacle_eval(HEADS["30k"], ev, tag, E4_EVAL)
        save_state(st)
        report("\n".join([
            "## E4 — trench-only eval (30k head, recal2b2 @ difficulty 0.00-0.05: heights ~0, corridor unchanged)",
            f"- completion {fmt(a['e4']['completion'])} | tripod {fmt(a['e4']['tripod'])} | falls {a['e4']['falls']}/100",
            "- prediction was ~0.27 (trench alone reproduces the collapse)",
        ]), "e4")
    # E1 widened baselines
    for key, weights, window in (("10k", REPLAY_SCHEDULE[2][0], 2), ("20k", W_20K, 3), ("30k", W_30K, 5)):
        if f"e1_{key}" in a:
            continue
        st["run_no"] += 1
        tag = f"a{st['run_no']:03d}_e1_{key}"
        ev = lineage_stack(window, weights, extra=CAMPAIGN_GEOM)
        a[f"e1_{key}"] = obstacle_eval(HEADS[key], ev, tag, OBST_EVAL)
        save_state(st)
    report("\n".join([
        "## E1 — widened-geometry baselines (recal2b2w @ 0.20-0.70; the only valid comparators downstream)",
        f"- B10 {fmt(a['e1_10k']['completion'])} (narrow-geometry record 0.73) | B20 {fmt(a['e1_20k']['completion'])} "
        f"(0.60) | B30 {fmt(a['e1_30k']['completion'])} (0.27)",
        f"- falls/100: {a['e1_10k']['falls']} / {a['e1_20k']['falls']} / {a['e1_30k']['falls']}",
        "- prediction: 30k >= 0.55, 10k >= 0.85; if the heads are not lifted the widening is recorded as "
        "unsupported by the eval (stays in force per the user decision)",
    ]), "e1 baselines")
    st["phase"] = "await_user_a"
    save_state(st)
    notify(f"Phase A complete — A0 late-motion ratio {fmt(ratio)}, E4 {fmt(a['e4']['completion'])}, "
           f"E1 B10/B20/B30 {fmt(a['e1_10k']['completion'])}/{fmt(a['e1_20k']['completion'])}/"
           f"{fmt(a['e1_30k']['completion'])}; T*={st['t_star']} s; PAUSED before smoke + wave 1 (--start-smoke)")


def phase_smoke(st: dict) -> None:
    sm = st["smoke"]
    checks = []
    if "unarmed" not in sm:
        st["run_no"] += 1
        tag = f"s{st['run_no']:03d}_smoke_unarmed"
        ckpt, tlog = train_retry(campaign_stack(), tag, SMOKE_ITERS, HEADS["20k"])
        ex = exposure_from_log(tlog, 10)
        sm["unarmed"] = {"ckpt": ckpt, "exposure": ex}
        save_state(st)
    ex = sm["unarmed"]["exposure"]
    keys_ok = all(ex.get(k) == ex.get(k) for k in ("reach_edge_frac", "crab_failure_flat", "stand_frac_actual", "rsi_frac_actual"))
    checks.append(("unarmed: telemetry keys present", keys_ok))
    checks.append(("unarmed: stand_frac_actual ~ 0.57 (0.40-0.75)", 0.40 <= ex.get("stand_frac_actual", -1) <= 0.75))
    checks.append(("unarmed: spread_frac_actual == 0", ex.get("spread_frac_actual", 1.0) == 0.0))
    if "armed" not in sm:
        st["run_no"] += 1
        tag = f"s{st['run_no']:03d}_smoke_armed"
        extra = dict(LEVERS["C1"], **LEVERS["C4"], **LEVERS["C5"])
        ckpt, tlog = train_retry(campaign_stack(extra), tag, SMOKE_ITERS, HEADS["20k"])
        ex2 = exposure_from_log(tlog, 10)
        sm["armed"] = {"ckpt": ckpt, "exposure": ex2}
        save_state(st)
    ex2 = sm["armed"]["exposure"]
    checks.append(("armed: stand_frac_actual ~ 0.2 (0.05-0.40)", 0.05 <= ex2.get("stand_frac_actual", -1) <= 0.40))
    checks.append(("armed: spread_frac_actual > 0.25", ex2.get("spread_frac_actual", 0.0) > 0.25))
    checks.append(("armed: rsi_frac_actual 0.10-0.30", 0.10 <= ex2.get("rsi_frac_actual", -1) <= 0.30))
    checks.append(("armed: goals_passed_mean parses", ex2.get("goals_passed_mean") == ex2.get("goals_passed_mean")))
    ok = all(c for _, c in checks)
    report("\n".join(["## B smoke — unarmed vs armed (200 iters from the 20k head, recal2b2w)"]
                     + [f"- {'PASS' if c else 'FAIL'}: {n}" for n, c in checks]
                     + ["### unarmed"] + expo_lines(ex) + ["### armed (C1+C4+C5)"] + expo_lines(ex2)
                     + [f"- **{'SMOKE PASS' if ok else 'SMOKE FAIL — wave 1 not started'}**"]),
           f"smoke {'PASS' if ok else 'FAIL'}")
    if not ok:
        st["phase"] = "await_user_a"
        save_state(st)
        notify("B smoke FAILED — see REPORT; fix the code before --start-smoke again")
        sys.exit(2)
    st["phase"] = "wave1"
    save_state(st)


def wave1_arms(st: dict) -> list[tuple[str, dict]]:
    arms = []
    for name in WAVE1_ORDER:
        if name == "C0":
            arms.append((name, {}))
        elif name == "C2":
            arms.append((name, {"KRABBY_EPISODE_S": str(st["t_star"] or 60)}))
        else:
            arms.append((name, dict(LEVERS[name])))
    return arms


def run_arm(st: dict, name: str, extra: dict, seed: str = SEED, tag_prefix: str = "c") -> dict:
    ctrl = st["arms"].get("C0")
    st["run_no"] += 1
    tag = f"{tag_prefix}{st['run_no']:03d}_{name.replace('+', '_')}" + (f"_seed{seed}" if seed != SEED else "")
    ev = campaign_stack(extra)
    log(f"{tag}: arm {name} {extra} seed {seed} ({ARM_ITERS} iters from the 20k head, recal2b2w)")
    ckpt, tlog = train_retry(ev, tag, ARM_ITERS, HEADS["20k"], seed,
                             live_check=live_backstop(ctrl["exposure"] if ctrl else None,
                                                      hazard_norm="KRABBY_EPISODE_S" in extra))
    rec = {"name": name, "extra": extra, "seed": seed, "tag": tag, "ckpt": ckpt, "log": str(tlog)}
    if ckpt is None or ckpt == "ABORTED":
        rec["exposure"] = exposure_from_log(tlog)
        rec["verdict"] = "ABORTED" if ckpt == "ABORTED" else "CRASH"
        report("\n".join([f"## ARM {name} (seed {seed}) — {rec['verdict']}", f"- extra: {extra}"] + expo_lines(rec["exposure"])),
               f"arm {name} {rec['verdict']}")
        return rec
    rec["exposure"] = exposure_from_log(tlog)
    rec["canary"] = canary(ckpt, ev, tag)
    rec["obst"] = obstacle_eval(ckpt, ev, tag, OBST_EVAL)
    if ctrl is None:  # this IS the control
        rec["verdict"] = "CONTROL"
        rec["judge"] = {}
    else:
        rec["judge"] = judge(name, rec, ctrl)
        rec["verdict"] = rec["judge"]["verdict"]
    b20 = ((st["a"].get("e1_20k") or {}).get("completion"))
    lines = [f"## ARM {name} (seed {seed}) — **{rec['verdict']}**",
             f"- extra: {extra} | checkpoint {Path(ckpt).parent.name}/{Path(ckpt).name}"]
    lines += expo_lines(rec["exposure"])
    lines.append(canary_line(rec["canary"]))
    lines.append(f"- obstacle eval (recal2b2w 0.20-0.70): completion {fmt(rec['obst']['completion'])} | tripod "
                 f"{fmt(rec['obst']['tripod'])} | falls {rec['obst']['falls']}/100"
                 + (f" | vs B20 {fmt(b20)}" if b20 is not None else "")
                 + (f" | vs C0 {fmt(ctrl['obst']['completion'])}" if ctrl else ""))
    if ctrl:
        j = rec["judge"]
        lines.append(f"- exposure status: {j['exposure_status']} — " + "; ".join(j["exposure_notes"]))
        lines.append("- safety: " + ("all gates held" if not j["unsafe"] else "VIOLATED — " + "; ".join(j["unsafe"])))
        lines.append("- obstacle-tile ceiling: " + (j["ceiling"] or "under ceiling"))
        if j.get("kind") == "hygiene":
            lines.append(f"- RSI-episode failure share ok: {j['rsi_failure_ok']}")
    report("\n".join(lines), f"arm {name} {rec['verdict']}")
    changelog(f"- {time.strftime('%Y-%m-%d %H:%M')} arm {name} seed {seed}: {rec['verdict']} "
              f"(reach_obst {fmt(rec['exposure'].get('reach_obst_frac'))}, goals_passed "
              f"{fmt(rec['exposure'].get('goals_passed_mean'))}, cov3 {fmt(rec['exposure'].get('obst_coverage_3'))}, "
              f"cov6 {fmt(rec['exposure'].get('obst_coverage_6'))}, obst completion {fmt(rec['obst']['completion'])})")
    return rec


def phase_wave1(st: dict) -> None:
    for name, extra in wave1_arms(st):
        if name in st["arms"] and st["arms"][name].get("verdict") not in (None, "CRASH"):
            continue
        rec = run_arm(st, name, extra)
        st["arms"][name] = rec
        save_state(st)
        if name == "C0" and rec["verdict"] == "CRASH":
            notify("C0 control crashed — wave 1 cannot be judged; PAUSED")
            st["phase"] = "await_user_a"
            save_state(st)
            sys.exit(2)
    summary = ", ".join(f"{n}: {st['arms'][n]['verdict']}" for n in WAVE1_ORDER if n in st["arms"])
    st["phase"] = "await_user_c"
    save_state(st)
    notify(f"wave 1 complete — {summary}; PAUSED before wave 2 (--start-wave2)")


def _vars_of(names: list[str], st: dict) -> dict:
    ev = {}
    for n in names:
        ev.update(st["arms"][n]["extra"] if n in st["arms"] else LEVERS.get(n, {}))
    return ev


def wave2_plan(st: dict) -> tuple[list[tuple[str, dict]], dict, list[str], list[str]]:
    """Wave-2 arms as (label, env extra). A user-registered plan in state['wave2']['user_plan']
    (list of [label, extra]) replaces the computed pairings (USER DECISION 2026-09-04 08:05,
    option A: C1 + field-only spread 2.5:11.0 at 0.25, no C5); eligibility is still reported."""
    arms = st["arms"]
    ctrl = arms["C0"]
    elig = {n: wave2_eligible(arms[n], ctrl) for n in ("C1", "C2", "C3", "C4") if n in arms}
    passing = [n for n, (ok, _why) in elig.items() if ok]
    c5 = ["C5"] if arms.get("C5", {}).get("verdict") == "PASS" else []
    user_plan = st["wave2"].get("user_plan")
    if user_plan:
        plan = [(label, dict(extra)) for label, extra in user_plan]
        return plan, elig, passing, c5
    plan_names = []
    if "C1" in passing and "C4" in passing:
        plan_names.append(("C1+C4" + ("+C5" if c5 else ""), ["C1", "C4"] + c5))
    if "C1" in passing and "C2" in passing:
        plan_names.append(("C1+C2" + ("+C5" if c5 else ""), ["C1", "C2"] + c5))
    ranked = sorted(passing, key=lambda n: arms[n]["exposure"].get("goals_passed_mean", 0.0), reverse=True)
    if ranked and c5:
        top = ranked[0]
        if not any(top in names for _l, names in plan_names):
            plan_names.append((f"{top}+C5", [top] + c5))
    return [(label, _vars_of(names, st)) for label, names in plan_names], elig, passing, c5


def phase_wave2(st: dict) -> None:
    arms = st["arms"]
    w2 = st["wave2"]
    plan, elig, passing, c5 = wave2_plan(st)
    if "eligibility" not in w2:
        w2["eligibility"] = {n: {"eligible": ok, "why": why} for n, (ok, why) in elig.items()}
        save_state(st)
        report("\n".join(["## Wave 2 — pairing eligibility (user rule 2026-09-03: PASS/PARTIAL, or safety held and reach_obst >= 3x C0)"]
                         + [f"- {n}: {'ELIGIBLE' if ok else 'not eligible'} — {why}" for n, (ok, why) in elig.items()]
                         + [f"- C5 hygiene arm {'PASSED (joins every pairing)' if c5 else 'not passed (omitted from pairings)'}"]
                         + [f"- plan ({'user-registered' if w2.get('user_plan') else 'computed'}): "
                            + ("; ".join(f"{l} = {e}" for l, e in plan) if plan else "none")]),
               "wave2 eligibility")
    w2["plan"] = [l for l, _e in plan]
    save_state(st)
    if not plan:
        st["phase"] = "await_user_d"
        save_state(st)
        notify("wave 2: no pairing qualifies (no single lever passed or partially passed) — PAUSED; "
               "result = 'exposure and gait stability trade off' unless the user redirects")
        return
    for label, extra in plan:
        if label in w2 and isinstance(w2[label], dict) and w2[label].get("verdict") not in (None, "CRASH"):
            continue
        w2[label] = run_arm(st, label, extra, tag_prefix="w")
        save_state(st)
    # triple if a computed C1+C4 meets coverage but misses goals_passed
    lab14 = next((l for l in w2 if l.startswith("C1+C4") and isinstance(w2[l], dict) and "exposure" in w2[l]), None)
    if not w2.get("user_plan") and lab14 and "C2" in passing and w2[lab14].get("verdict") in ("PARTIAL",):
        ex = w2[lab14]["exposure"]
        if ex.get("obst_coverage_3", 0) >= TARGET["obst_coverage_3"] and ex.get("obst_coverage_6", 0) >= TARGET["obst_coverage_6"] \
                and ex.get("goals_passed_mean", 0) < TARGET["goals_passed_mean"]:
            label = "C1+C2+C4" + ("+C5" if c5 else "")
            if label not in w2:
                w2[label] = run_arm(st, label, _vars_of(["C1", "C2", "C4"] + c5, st), tag_prefix="w")
                save_state(st)
    # winner = best PASS (wave 2 first, then wave 1) by goals_passed; seed-2 replay of it
    cands = [(l, r) for l, r in w2.items() if isinstance(r, dict) and r.get("verdict") == "PASS"]
    cands += [(n, arms[n]) for n in passing if arms[n]["verdict"] == "PASS"]
    if cands:
        label, rec = max(cands, key=lambda lr: lr[1]["exposure"].get("goals_passed_mean", 0.0))
        st["winner"] = {"label": label, "extra": rec["extra"]}
        save_state(st)
        if "seed2" not in w2:
            w2["seed2"] = run_arm(st, label, rec["extra"], seed=CONFIRM_SEED, tag_prefix="w")
            save_state(st)
        s2 = w2["seed2"]["verdict"]
        st["phase"] = "await_user_d"
        save_state(st)
        notify(f"wave 2 complete — winner {label} (seed-2 {s2}); PAUSED before Phase D replay (--start-d)")
    else:
        best = max(((l, r) for l, r in w2.items() if isinstance(r, dict) and "exposure" in r),
                   key=lambda lr: lr[1]["exposure"].get("goals_passed_mean", 0.0), default=None)
        st["phase"] = "await_user_d"
        save_state(st)
        notify("wave 2 complete — no combination reached PASS"
               + (f"; best trade-off {best[0]} ({best[1]['verdict']})" if best else "") + "; PAUSED")


def phase_d(st: dict) -> None:
    """Replay 5k->30k with the winner armed on recal2b2w (seed 3), then seed 2."""
    if not st.get("winner"):
        log("phase d: no winner recorded — nothing to replay")
        st["phase"] = "done"
        save_state(st)
        return
    extra = dict(st["winner"]["extra"])
    d = st["d"]
    ctrl = st["arms"]["C0"]["exposure"]
    for seed in (SEED, CONFIRM_SEED):
        key = f"seed{seed}"
        rec = d.setdefault(key, {"head": ANCHOR, "segments": {}})
        for s in range(1, 6):
            if str(s) in rec["segments"]:
                continue
            st["run_no"] += 1
            tag = f"d{st['run_no']:03d}_replay_seed{seed}_seg{s}"
            weights, ramps = REPLAY_SCHEDULE[s]
            ev = lineage_stack(s, weights, BANKS[s], CAMPAIGN_GEOM)
            ev.update(extra)
            if ramps:
                ev["KRABBY_PHASEOUT"] = phaseout_spec(ramps)
            log(f"{tag}: replay window {s} ({5 * s}k -> {5 * (s + 1)}k) seed {seed} with {extra}")
            head, tlog = train_retry(ev, tag, ARM_ITERS, rec["head"], seed, live_check=live_backstop(ctrl))
            seg = {"ckpt": head, "log": str(tlog), "exposure": exposure_from_log(tlog)}
            if head is None or head == "ABORTED":
                seg["verdict"] = "ABORTED" if head == "ABORTED" else "CRASH"
                rec["segments"][str(s)] = seg
                save_state(st)
                report("\n".join([f"## D — replay seed {seed} window {s}: {seg['verdict']}"] + expo_lines(seg["exposure"])),
                       f"d seed{seed} seg{s} {seg['verdict']}")
                notify(f"Phase D replay seed {seed} window {s} {seg['verdict']} — PAUSED")
                st["phase"] = "await_user_d"
                save_state(st)
                return
            seg["canary"] = canary(head, ev, tag)
            if s in (1, 3, 5):
                seg["obst"] = obstacle_eval(head, ev, tag, OBST_EVAL)
            status, notes = exposure_verdict(seg["exposure"], None)
            seg["exposure_status"] = status
            rec["segments"][str(s)] = seg
            rec["head"] = head
            save_state(st)
            base_key = {1: "e1_10k", 3: "e1_20k", 5: "e1_30k"}.get(s)
            base = (st["a"].get(base_key) or {}).get("completion") if base_key else None
            lines = [f"## D — replay seed {seed} window {s} ({5 * s}k -> {5 * (s + 1)}k) — exposure {status}",
                     f"- checkpoint {Path(head).parent.name}/{Path(head).name} | ramps {ramps} | extra {extra}"]
            lines += expo_lines(seg["exposure"])
            lines.append(canary_line(seg["canary"]))
            if "obst" in seg:
                lines.append(f"- obstacle eval (recal2b2w): completion {fmt(seg['obst']['completion'])} vs widened baseline "
                             f"B{5 * (s + 1)}k {fmt(base)}")
            lines.append("- exposure notes: " + "; ".join(notes))
            report("\n".join(lines), f"d seed{seed} seg{s}")
    st["phase"] = "done"
    save_state(st)
    notify("Phase D complete (seed 3 + seed 2 replays 5k->30k) — re-bake decision is the user's")


def main() -> int:
    args = set(sys.argv[1:])
    st = load_state()
    if "--start-smoke" in args and st["phase"] == "await_user_a":
        st["phase"] = "smoke"
    if "--start-wave2" in args and st["phase"] == "await_user_c":
        st["phase"] = "wave2"
    if "--start-d" in args and st["phase"] == "await_user_d":
        st["phase"] = "d"
    save_state(st)
    if st["phase"] == "a":
        phase_a(st)
    if st["phase"] == "await_user_a":
        log("phase await_user_a: Phase A done — relaunch with --start-smoke to run the smokes + wave 1")
        return 0
    if st["phase"] == "smoke":
        phase_smoke(st)
    if st["phase"] == "wave1":
        phase_wave1(st)
    if st["phase"] == "await_user_c":
        log("phase await_user_c: wave 1 done — relaunch with --start-wave2")
        return 0
    if st["phase"] == "wave2":
        phase_wave2(st)
    if st["phase"] == "await_user_d":
        log("phase await_user_d: wave 2 done — relaunch with --start-d for the 5k->30k replay")
        return 0
    if st["phase"] == "d":
        phase_d(st)
    if st["phase"] == "done":
        log("phase done: campaign complete — re-bake decision is the user's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
