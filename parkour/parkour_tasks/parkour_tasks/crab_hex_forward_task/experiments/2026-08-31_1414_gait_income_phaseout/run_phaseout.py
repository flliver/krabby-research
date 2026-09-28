#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Gait-income phase-out reverse round search (PLAN G, approved 2026-08-31).

Fork of the graduated round-search orchestrator (parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-26_2200_
gated_lineage/run_round_search.py) with the phase-out modifications from the approved
plan (/home/nickmagus/.claude/plans/there-are-a-variety-federated-frog.md):

* elements = per-term anneal NOTCHES (cosine ramp over 1k iters via KRABBY_PHASEOUT,
  then hold; gates read the settled tail only, unit-weight normalized);
* runs on the SEED-3 lineage (reference-of-record seed), windows co-scheduled with
  the graduated curriculum (7 elements @5k, 3 @10k);
* r1 control = the seed-3 confirmation replay's own segment 1 (TB log on disk +
  retroactive canary); r2+ controls run fresh from the phase-out bake head;
* watchdogs: value-loss spike, mean-reward clip floor, creep signature, slip;
* RSI bank: P0-null throughout (as the confirmation replay trained), re-harvested
  only from bakes that pass the round backstop;
* phases: canaries -> scout -> AWAIT_USER (pre-approved GPU stops here; rounds
  start only with --start-rounds after the user decision).

Stdlib only; resumable via state.json.
"""
from __future__ import annotations

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
RESET_TOOL = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-17_1500_phased_flat/B1_critic_reset/make_critic_reset.py"
HARVEST_TOOL = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/harvest_rsi_bank.py"
MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
FLAT_RUNS = PARKOUR / "logs/rsl_rl/crab_hex_flat_walk"
EVAL_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/gait_income_phaseout"
STATE = HERE / "state.json"
REPORT = HERE / "REPORT.md"
CHANGELOG = HERE / "CHANGELOG.md"

# --- seed-3 lineage anchors (graduated schedule of record) -----------------------
ANCHOR = str(FLAT_RUNS / "2026-08-31_03-42-16/model_4999.pt")          # 5k, formation done
SEG1_LOG = LINEAGE / "confirm_050_seg1_train.log"                      # 5k-10k full-income control
SEG1_HEAD = str(FLAT_RUNS / "2026-08-31_06-04-16/model_9998.pt")       # 10k control head
REF_HEAD = str(FLAT_RUNS / "2026-08-31_10-46-19/model_19996.pt")       # reference of record (20k)
SEED = "3"                # the lineage seed (per user: search runs on seed 3)
CONFIRM_SEED = "2"        # primary seed — terminal replay before any bake proposal

STEPS_PER_ITER = 24       # CrabHexFlatWalkPPORunnerCfg num_steps_per_env
RAMP_ITERS = 1000
PROBE_ITERS = 2000
ESCALATE_ITERS = 3000     # probe continue: 2k -> 5k
ROUND_ITERS = 5000        # one window
MAX_ROUND = 3             # r1 (5k-10k) .. r3 (15k-20k); r4 (20k-25k) only by user opt-in

# Baseline core (identical to the round-search BASELINE; bank = P0-null as the
# confirmation replay trained the whole seed-3 lineage).
BASELINE = {
    "KRABBY_CLOCK_W": "1.0",
    "KRABBY_APEX_W": "1.0",
    "KRABBY_RSI_FRAC": "0.2",
    "KRABBY_RSI_BANK": str(LINEAGE / "rsi_bank_P0_null.npz"),
    "KRABBY_FLAT_TERRAIN_MODE": "light",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    "KRABBY_TRACK_SIGMA2": "0.1",
    "KRABBY_TRACK_L1_W": "-1.0",
    "KRABBY_LIN_VEL_X": "0.0:0.35",
}
# Graduated curriculum schedule of record: elements active from window r onward.
SCHEDULE = {
    1: {  # placed @5k -> active in the 5k-10k window and later
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
    2: {  # placed @10k
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

# --- notch catalogue -------------------------------------------------------------
# term -> (weight env var, baseline weight, notch targets in order). eps floor 1e-3:
# ParkourRewardManager skips weight==0.0 terms and their telemetry flatlines.
EPS = 0.001
TERMS = {
    "reward_clock_schedule": ("KRABBY_CLOCK_W", 1.0, [0.5, 0.2, EPS]),
    "reward_clock_swing_apex": ("KRABBY_APEX_W", 1.0, [0.5, EPS]),
    "reward_feet_air_time_positive": ("KRABBY_AIRTIME_W", 0.8, [0.4, EPS]),
    "reward_stride_length": ("KRABBY_STRIDE_W", 0.5, [0.25, EPS]),
}
# within-round test order: prescriptive-first
TERM_ORDER = ["reward_clock_schedule", "reward_clock_swing_apex",
              "reward_feet_air_time_positive", "reward_stride_length"]

SCOUT_VARS = {"KRABBY_CLOCK_W": str(EPS), "KRABBY_APEX_W": str(EPS),
              "KRABBY_AIRTIME_W": str(EPS), "KRABBY_STRIDE_W": str(EPS)}


# ------------------------------------------------------------------ infrastructure
def log(msg: str) -> None:
    print(f"[phaseout {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def report(block: str, marker: str) -> None:
    with REPORT.open("a") as fh:
        fh.write(block.rstrip() + f"\n>>> ENTRY {marker}\n\n")


def changelog(row: str) -> None:
    with CHANGELOG.open("a") as fh:
        fh.write(row.rstrip() + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "phase": "canaries", "round": 1, "run_no": 0,
        "weights": {t: TERMS[t][1] for t in TERM_ORDER},   # current accepted weights
        "notch_idx": {t: 0 for t in TERM_ORDER},           # next notch index per term
        "deferred": [],                                     # (term) notches failed this round
        "accepted_this_round": [],
        "bake_head": ANCHOR,
        "bank": BASELINE["KRABBY_RSI_BANK"],
        "controls": {},
        "history": {},                                      # term -> [(round, w_from, w_to, verdict)]
    }


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


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


def latest_run_ckpt() -> str:
    run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(models[-1])


def _wait_isaac_clear() -> None:
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", "rsl_rl/train.py|eval_crab_hex_gait"],
                          capture_output=True).returncode != 0:
            return
        time.sleep(10)


def train(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None,
          seed: str = SEED) -> tuple[str | None, Path]:
    t0 = time.time()
    log_path = HERE / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", "Isaac-Crab-Hex-Flat-Walk-v0", "--headless",
           "--num_envs", "256", "--seed", seed, "--max_iterations", str(iters)]
    if resume_ckpt:
        cmd += ["--resume", "--checkpoint", resume_ckpt]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update({k: str(v) for k, v in env_vars.items()})
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT)
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
            # Live catastrophic-failure watchdog (PLAN G): a failure tail far above the
            # window control after enough prints means the run is destroying the gait —
            # kill it early rather than paying the full segment (scout + notch probes).
            try:
                fail = series(log_path, "Episode_Termination/crab_failure")
                if len(fail) > 500 and tail_mean(fail, 100) > _live_fail_ceiling():
                    log(f"{tag}: LIVE ABORT — failure tail {tail_mean(fail, 100):.2f} > "
                        f"ceiling {_live_fail_ceiling():.2f}")
                    proc.kill()
                    try:
                        proc.wait(timeout=60)
                    except Exception:
                        pass
                    return "ABORTED", log_path
            except Exception:
                pass
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


_LIVE_FAIL_CEILING = [0.75]  # overwritten from the control record at phase entry


def _live_fail_ceiling() -> float:
    return _LIVE_FAIL_CEILING[0]


def train_retry(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None,
                seed: str = SEED) -> tuple[str | None, Path]:
    ckpt, tlog = train(env_vars, tag, iters, resume_ckpt, seed)
    for retry in (2, 3):
        if ckpt is not None:
            break
        txt = tlog.read_text(errors="ignore") if tlog.exists() else ""
        if "Traceback" in txt or "Error executing job" in txt:
            break
        log(f"{tag}: died without a Python error (boot balloon?) — infra retry {retry - 1}")
        _wait_isaac_clear()
        time.sleep(120)
        ckpt, tlog = train(env_vars, f"{tag}_r{retry}", iters, resume_ckpt, seed)
    return ckpt, tlog


def canary(ckpt: str, env_vars: dict, tag: str) -> dict:
    ev = {k: env_vars[k] for k in CANARY_KEEP if k in env_vars}
    lp = HERE / f"{tag}_canary.log"
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update(ev)
    for attempt in range(4):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST),
                                "--scenario", "flat_walk_slow_v2", "--checkpoint", ckpt, "--no-plot",
                                "--output-root", str(EVAL_ROOT)],
                               cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
            except subprocess.TimeoutExpired:
                pass
        d = max((EVAL_ROOT / "flat_walk_slow_v2" / "seed001").iterdir(), key=lambda p: p.stat().st_mtime)
        if d.stat().st_mtime >= t0 and (d / "scenario_metrics.json").exists():
            a = json.loads((d / "scenario_metrics.json").read_text())["aggregate"]
            slip = a.get("slip_ratio")
            if isinstance(slip, dict):
                slip = slip.get("median")
            return {"tripod": (a.get("tripod_score") or {}).get("median") or 0.0,
                    "completion": a.get("schedule_completion_rate") or 0.0,
                    "tracking": (a.get("tracking_ratio") or {}).get("median") or 0.0,
                    "slip": slip if slip is not None else 0.0,
                    "tippy": a.get("tippy_tap_fraction") or 0.0}
        log(f"{tag}: canary produced no fresh metrics (boot balloon?) — cool-down, retry")
        _wait_isaac_clear()
        time.sleep(120)
    raise RuntimeError(f"{tag}: canary produced no fresh metrics after 4 attempts")


def obstacle_eval(ckpt: str, env_vars: dict, tag: str) -> dict:
    ev = {k: env_vars[k] for k in CANARY_KEEP if k in env_vars}
    ev.update({"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0",
               "KRABBY_FLAT_TERRAIN_DIFF": env_vars.get("KRABBY_FLAT_TERRAIN_DIFF", "0.05:0.2"),
               "KRABBY_FLAT_TERRAIN_GEOM": env_vars.get("KRABBY_FLAT_TERRAIN_GEOM", "shallow")})
    lp = HERE / f"{tag}_obst.log"
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update(ev)
    with lp.open("w") as fh:
        try:
            subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST),
                            "--scenario", "flat_walk_slow_v2", "--checkpoint", ckpt, "--no-plot",
                            "--output-root", str(EVAL_ROOT) + "_obst"],
                           cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=5400)
        except subprocess.TimeoutExpired:
            pass
    d = max((Path(str(EVAL_ROOT) + "_obst") / "flat_walk_slow_v2" / "seed001").iterdir(),
            key=lambda p: p.stat().st_mtime)
    a = json.loads((d / "scenario_metrics.json").read_text())["aggregate"]
    return {"completion": a.get("schedule_completion_rate") or 0.0,
            "tripod": (a.get("tripod_score") or {}).get("median") or 0.0}


def probe_metrics(tlog: Path, clock_w: float, settle_from: int = 0,
                  at_prints: int | None = None) -> dict:
    """Settled-tail probe metrics, unit-weight normalized.

    ``clock_w`` is the run's SETTLED clock weight (normalization divisor: episodic
    sums are linear in weight — pinned by test_parkour_reward_manager_epsilon).
    ``settle_from`` drops mid-ramp prints (gates never read the ramp).
    """
    clock = series(tlog, "Episode_Reward/reward_clock_schedule")
    track = series(tlog, "Episode_Reward/track_lin_vel_xy_exp")
    fail = series(tlog, "Episode_Termination/crab_failure")
    eplen = series(tlog, "Mean episode length")
    vloss = series(tlog, "Mean value_function")
    mrew = series(tlog, "Mean reward")
    if at_prints:
        clock, track, fail = clock[:at_prints], track[:at_prints], fail[:at_prints]
        eplen, vloss, mrew = eplen[:at_prints], vloss[:at_prints], mrew[:at_prints]
    settled = slice(settle_from, None)
    return {
        "clock_unit": tail_mean(clock[settled]) / clock_w if clock_w else float("nan"),
        "clock_raw": tail_mean(clock[settled]),
        "track": tail_mean(track[settled]),
        "failure": tail_mean(fail[settled]),
        "eplen": tail_mean(eplen[settled]),
        "vloss": tail_mean(vloss[settled], 200),
        "mean_reward": tail_mean(mrew[settled]),
        "fail_trend": (tail_mean(fail, 100) - tail_mean(fail[:-100] or fail, 100)) if len(fail) > 200 else 0.0,
    }


def watchdogs(pm: dict, ctrl: dict, can: dict | None, tag: str) -> list[str]:
    """Return tripped-watchdog descriptions (empty = healthy). Thresholds provisional,
    calibrated from the r1 control record (plan: recalibrate on scout + control data)."""
    trips = []
    if ctrl.get("vloss") and pm.get("vloss") and pm["vloss"] > 3.0 * ctrl["vloss"]:
        trips.append(f"value-loss spike {pm['vloss']:.4f} > 3x control {ctrl['vloss']:.4f}")
    if ctrl.get("mean_reward") and pm.get("mean_reward") is not None \
            and pm["mean_reward"] < 0.05 * ctrl["mean_reward"]:
        trips.append(f"clip-floor: mean reward {pm['mean_reward']:.2f} < 5% of control {ctrl['mean_reward']:.2f}")
    if can is not None:
        if can["completion"] >= 0.90 and can["tracking"] < 0.40:
            trips.append(f"CREEP signature: completion {can['completion']:.2f} with tracking {can['tracking']:.2f}")
        cslip = (ctrl.get("canary") or {}).get("slip")
        if cslip and can.get("slip") and can["slip"] > 1.5 * cslip:
            trips.append(f"slip ratio {can['slip']:.3f} > 1.5x control {cslip:.3f}")
    for t in trips:
        log(f"{tag}: WATCHDOG {t}")
    return trips


def env_for(st: dict, rnd: int, extra: dict | None = None) -> dict:
    """Env stack for a run in window ``rnd``: baseline core + curriculum elements with
    placement <= rnd + current accepted gait weights + any candidate extras."""
    ev = dict(BASELINE)
    for r in sorted(SCHEDULE):
        if r <= rnd:
            ev.update(SCHEDULE[r])
    ev["KRABBY_RSI_BANK"] = st["bank"]
    for term, (var, _base, _notches) in TERMS.items():
        ev[var] = str(st["weights"][term])
    if extra:
        ev.update(extra)
    return ev


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# ------------------------------------------------------------------ phase: canaries
def phase_canaries(st: dict) -> None:
    """Retroactive canaries: 5k anchor (entry formation band) + seg1 10k head (r1
    control record). Eval-only GPU."""
    ev = env_for(st, rnd=1)
    log("retroactive canary: seed-3 5k anchor")
    can_anchor = canary(ANCHOR, ev, "retro_anchor")
    log("retroactive canary: seed-3 10k head (r1 control)")
    can_seg1 = canary(SEG1_HEAD, ev, "retro_seg1")
    # r1 control probe from the confirmation replay's seg1 log (full-window tail;
    # clock at full weight 1.0 -> unit == raw).
    cpm = probe_metrics(SEG1_LOG, clock_w=1.0)
    st["controls"]["1"] = {"probe": cpm, "canary": can_seg1}
    st["anchor_canary"] = can_anchor
    _LIVE_FAIL_CEILING[0] = min(0.9, cpm["failure"] + 0.35)
    band = "WITHIN band" if cpm["clock_unit"] >= 0.38 and cpm["failure"] <= 0.30 else "OUTSIDE band — review"
    report("\n".join([
        "## ENTRY GATE — retroactive canaries + r1 control record (seed-3 lineage)",
        f"- 5k anchor canary ({Path(ANCHOR).parent.name}/model_4999): tripod {fmt(can_anchor['tripod'])} | "
        f"completion {fmt(can_anchor['completion'])} | tracking {fmt(can_anchor['tracking'])} | "
        f"slip {fmt(can_anchor['slip'])}",
        f"- r1 control canary ({Path(SEG1_HEAD).parent.name}/model_9998): tripod {fmt(can_seg1['tripod'])} | "
        f"completion {fmt(can_seg1['completion'])} | tracking {fmt(can_seg1['tracking'])} | "
        f"slip {fmt(can_seg1['slip'])}",
        f"- r1 control probe (seg1 TB tail): unit-clock {fmt(cpm['clock_unit'])} | track income "
        f"{fmt(cpm['track'])} | failure {fmt(cpm['failure'])} | ep len {fmt(cpm['eplen'])} | "
        f"vloss {cpm['vloss']:.4f} | mean reward {fmt(cpm['mean_reward'])}",
        f"- formation band (unit-clock >= 0.38, failure <= 0.30): {band}",
    ]), "entry gate canaries")
    st["phase"] = "scout"
    save_state(st)


# ------------------------------------------------------------------ phase: scout
def phase_scout(st: dict) -> None:
    """The pre-approved falsification probe: full 5k-10k window from the anchor with
    ALL gait income at eps (hard set, env vars only). Then PAUSE for the user."""
    ctrl = st["controls"]["1"]
    st["run_no"] += 1
    tag = f"scout_{st['run_no']:03d}"
    ev = env_for(st, rnd=1, extra=SCOUT_VARS)
    log(f"{tag}: SCOUT — 5k-10k window, clock/apex/airtime/stride at eps={EPS}")
    ckpt, tlog = train_retry(ev, tag, ROUND_ITERS, ANCHOR)
    if ckpt is None:
        report(f"## HALT — scout crashed before producing a checkpoint (config/code error). "
               f"Fix and relaunch.", "HALT scout crash")
        changelog("> NOTIFY: scout crashed (config/code error) — needs a fix + relaunch")
        raise SystemExit(3)
    aborted = ckpt == "ABORTED"
    pm = probe_metrics(tlog, clock_w=EPS)
    can = None if aborted else canary(latest_run_ckpt() if aborted else ckpt, ev, tag)
    trips = watchdogs(pm, ctrl["probe"] | {"canary": ctrl["canary"]}, can, tag)
    cpm, ccan = ctrl["probe"], ctrl["canary"]
    lines = [
        "## SCOUT — all gait income at eps for the full 5k-10k window",
        "| metric | scout | seg1 control | margin |", "|---|---|---|---|",
        f"| unit-weight clock income | {fmt(pm['clock_unit'])} | {fmt(cpm['clock_unit'])} | "
        f"{(pm['clock_unit'] / cpm['clock_unit'] if cpm['clock_unit'] else float('nan')):.2f}x |",
        f"| tracking income | {fmt(pm['track'])} | {fmt(cpm['track'])} | "
        f"{(pm['track'] / cpm['track'] if cpm['track'] else float('nan')):.2f}x |",
        f"| failure tail | {fmt(pm['failure'])} | {fmt(cpm['failure'])} | {pm['failure'] - cpm['failure']:+.3f} |",
        f"| value loss | {pm['vloss']:.4f} | {cpm['vloss']:.4f} | "
        f"{(pm['vloss'] / cpm['vloss'] if cpm['vloss'] else float('nan')):.1f}x |",
        f"| mean reward | {fmt(pm['mean_reward'])} | {fmt(cpm['mean_reward'])} | "
        f"{(pm['mean_reward'] / cpm['mean_reward'] if cpm['mean_reward'] else float('nan')):.2f}x |",
    ]
    if can:
        lines.append(
            f"| canary t/c/tr/slip | {fmt(can['tripod'])}/{fmt(can['completion'])}/{fmt(can['tracking'])}/"
            f"{fmt(can['slip'])} | {fmt(ccan['tripod'])}/{fmt(ccan['completion'])}/{fmt(ccan['tracking'])}/"
            f"{fmt(ccan['slip'])} | tripod {(can['tripod'] / max(ccan['tripod'], 1e-9)):.2f}x |")
    if trips:
        lines.append(f"- watchdogs tripped: {'; '.join(trips)}")
    # world classification (plan Phase 0): collapse / middle / jackpot
    if aborted or (can is None):
        world = "COLLAPSE (live-abort: failure spike)"
    elif trips and any("CREEP" in t for t in trips):
        world = "COLLAPSE-BY-HACK (creep signature)"
    elif can["tripod"] >= 0.85 * ccan["tripod"] and pm["failure"] <= cpm["failure"] + 0.05 \
            and can["tracking"] >= 0.85 * ccan["tracking"]:
        world = "JACKPOT (gait holds, objectives match/beat control)"
    elif can["tripod"] < 0.5 * ccan["tripod"] or pm["failure"] > cpm["failure"] + 0.25:
        world = "COLLAPSE (gait income load-bearing at 5k)"
    else:
        world = "MIDDLE (partial degradation — notch search discovers per-window income needs)"
    lines.append(f"- **WORLD: {world}**")
    lines.append("- next: PAUSED for user decision on the round schedule (plan Phase 0.2)")
    report("\n".join(lines), "scout verdict")
    st["scout"] = {"probe": pm, "canary": can, "world": world, "ckpt": None if aborted else ckpt,
                   "watchdogs": trips}
    st["phase"] = "await_user"
    changelog(f"> NOTIFY: scout done — {world}; campaign PAUSED for user decision")
    save_state(st)


# ------------------------------------------------------------------ phase: rounds
def next_candidates(st: dict) -> list[str]:
    """Terms with a remaining notch, in prescriptive-first order; deferred notches
    re-qualify automatically (their notch_idx was not advanced)."""
    out = []
    for term in TERM_ORDER:
        if st["notch_idx"][term] < len(TERMS[term][2]):
            out.append(term)
    return out


def probe_candidate(st: dict, term: str) -> str:
    rnd = st["round"]
    var, _base, notches = TERMS[term]
    w_from = st["weights"][term]
    w_to = notches[st["notch_idx"][term]]
    ctrl = st["controls"][str(rnd)]
    cpm = st.get("ref_probe") or ctrl["probe"]
    ccan = st.get("ref_canary") or ctrl["canary"]
    st["run_no"] += 1
    tag = f"r{rnd}_{st['run_no']:03d}_{term.split('reward_')[-1]}_{w_to}"
    ramp_steps = RAMP_ITERS * STEPS_PER_ITER
    extra = {var: str(w_from),
             "KRABBY_PHASEOUT": f"{term}:{w_from}:{w_to}:0:{ramp_steps}"}
    ev = env_for(st, rnd, extra)
    log(f"{tag}: notch probe {w_from} -> {w_to} (ramp {RAMP_ITERS} iters, settle to {PROBE_ITERS})")
    ckpt, tlog = train_retry(ev, tag, PROBE_ITERS, st["bake_head"])
    if ckpt is None:
        report(f"## HALT — {tag} crashed (config/code error, not a verdict). {term} stays "
               f"queued; fix and relaunch.", f"HALT crash {term} round{rnd}")
        raise SystemExit(3)
    if ckpt == "ABORTED":
        report_decision(term, rnd, w_from, w_to, None, cpm, "FAIL",
                        "live watchdog abort (failure spike)", "", None, None)
        return "FAIL"
    clock_w_settled = w_to if term == "reward_clock_schedule" else st["weights"]["reward_clock_schedule"]
    pm = probe_metrics(tlog, clock_w=clock_w_settled, settle_from=RAMP_ITERS)
    trips = watchdogs(pm, cpm | {"canary": ccan}, None, tag)
    clock_ratio = pm["clock_unit"] / cpm["clock_unit"] if cpm["clock_unit"] else 0.0
    track_ratio = pm["track"] / cpm["track"] if cpm["track"] else 0.0
    fail_delta = pm["failure"] - cpm["failure"]
    esc = ""
    if trips:
        report_decision(term, rnd, w_from, w_to, pm, cpm, "FAIL",
                        f"watchdog trip: {'; '.join(trips)}", esc, None, None)
        return "FAIL"
    if fail_delta <= 0.05 and clock_ratio >= 0.85 and track_ratio >= 0.85:
        report_decision(term, rnd, w_from, w_to, pm, cpm, "PASS",
                        f"fail {fail_delta:+.3f} <= +0.05, unit-clock {clock_ratio:.2f}x >= 0.85, "
                        f"track {track_ratio:.2f}x >= 0.85 (settled tail)", esc, None, None)
        st["ref_probe"] = pm
        st.setdefault("costs", {})[f"{term}@{w_to}"] = {"clock_ratio": clock_ratio, "fail_delta": fail_delta}
        return "PASS"
    if fail_delta > 0.10 or clock_ratio < 0.70:
        report_decision(term, rnd, w_from, w_to, pm, cpm, "FAIL",
                        f"fail {fail_delta:+.3f} > +0.10 or unit-clock {clock_ratio:.2f}x < 0.70", esc, None, None)
        return "FAIL"
    esc = f"ambiguous at 2k (unit-clock {clock_ratio:.2f}x, track {track_ratio:.2f}x, fail {fail_delta:+.3f}); continued to 5k"
    ckpt2, tlog2 = train_retry(ev, tag + "_esc", ESCALATE_ITERS, ckpt)
    if ckpt2 is None or ckpt2 == "ABORTED":
        report_decision(term, rnd, w_from, w_to, pm, cpm, "FAIL",
                        "escalation crashed/aborted", esc, None, None)
        return "FAIL"
    can = canary(ckpt2, ev, tag + "_esc")
    trips = watchdogs(pm, cpm | {"canary": ccan}, can, tag)
    if trips:
        report_decision(term, rnd, w_from, w_to, pm, cpm, "FAIL",
                        f"escalation watchdog trip: {'; '.join(trips)}", esc, can, ccan)
        return "FAIL"
    ratio = min(can["tripod"] / max(ccan["tripod"], 1e-9),
                can["completion"] / max(ccan["completion"], 1e-9),
                can["tracking"] / max(ccan["tracking"], 1e-9))
    if can["tripod"] >= 0.85 * ccan["tripod"] and ratio >= 0.85:
        report_decision(term, rnd, w_from, w_to, pm, cpm, "PASS",
                        f"escalated canary: tripod {(can['tripod'] / max(ccan['tripod'], 1e-9)):.2f}x >= 0.85 "
                        f"and ratio {ratio:.2f} >= 0.85", esc, can, ccan)
        st["ref_probe"] = pm
        st["ref_canary"] = can
        st.setdefault("costs", {})[f"{term}@{w_to}"] = {"clock_ratio": clock_ratio, "fail_delta": fail_delta}
        return "PASS"
    report_decision(term, rnd, w_from, w_to, pm, cpm, "FAIL (AMBIGUOUS)",
                    f"canary ratio {ratio:.2f} < 0.85 after escalation — retries a later round",
                    esc, can, ccan)
    return "FAIL"


def report_decision(term: str, rnd: int, w_from, w_to, pm: dict | None, cpm: dict,
                    verdict: str, rule: str, esc: str, can: dict | None, ccan: dict | None) -> None:
    lines = [f"## NOTCH decision — {term} {w_from} -> {w_to} @ round {rnd}"]
    if pm:
        lines += ["| metric | candidate | reference | margin |", "|---|---|---|---|",
                  f"| unit-clock (settled) | {fmt(pm['clock_unit'])} | {fmt(cpm['clock_unit'])} | "
                  f"{(pm['clock_unit'] / cpm['clock_unit'] if cpm['clock_unit'] else float('nan')):.2f}x |",
                  f"| track income | {fmt(pm['track'])} | {fmt(cpm['track'])} | "
                  f"{(pm['track'] / cpm['track'] if cpm['track'] else float('nan')):.2f}x |",
                  f"| failure tail | {fmt(pm['failure'])} | {fmt(cpm['failure'])} | "
                  f"{pm['failure'] - cpm['failure']:+.3f} |"]
    if can and ccan:
        lines.append(f"| canary t/c/tr | {fmt(can['tripod'])}/{fmt(can['completion'])}/{fmt(can['tracking'])} | "
                     f"{fmt(ccan['tripod'])}/{fmt(ccan['completion'])}/{fmt(ccan['tracking'])} | |")
    if esc:
        lines.append(f"- escalation history: {esc}")
    lines.append(f"- **DECISION: {verdict}** — {rule}")
    report("\n".join(lines), f"decision {term}@{w_to} round{rnd} {verdict.split()[0]}")


def run_control(st: dict) -> None:
    """Fresh no-change control for r2+ (r1's control is the historical seg1 record)."""
    rnd = st["round"]
    st["run_no"] += 1
    tag = f"r{rnd}_{st['run_no']:03d}_control"
    ev = env_for(st, rnd)
    log(f"{tag}: round-{rnd} control (+{PROBE_ITERS} on bake head, no new notch)")
    ckpt, tlog = train_retry(ev, tag, PROBE_ITERS, st["bake_head"])
    if ckpt is None or ckpt == "ABORTED":
        report(f"## HALT — round-{rnd} control crashed/aborted", "halt control")
        sys.exit(2)
    pm = probe_metrics(tlog, clock_w=st["weights"]["reward_clock_schedule"])
    can = canary(ckpt, ev, tag)
    st["controls"][str(rnd)] = {"probe": pm, "canary": can}
    st["ref_probe"] = pm
    st["ref_canary"] = can
    _LIVE_FAIL_CEILING[0] = min(0.9, pm["failure"] + 0.35)
    report("\n".join([
        f"## BASELINE control — round {rnd} ({tag})",
        f"- probe: unit-clock {fmt(pm['clock_unit'])} | track {fmt(pm['track'])} | failure "
        f"{fmt(pm['failure'])} | vloss {pm['vloss']:.4f} | mean reward {fmt(pm['mean_reward'])}",
        f"- flat canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
        f"tracking {fmt(can['tracking'])} | slip {fmt(can['slip'])}",
    ]), f"baseline round{rnd}")
    save_state(st)


def bake_round(st: dict) -> None:
    """Round-end: train the full accepted set from the bake head to the round boundary
    (all accepted notches ramping simultaneously), backstop it, adopt as the new head,
    re-harvest RSI only on a passing backstop."""
    rnd = st["round"]
    ctrl = st["controls"][str(rnd)]
    accepted = list(st["accepted_this_round"])
    while True:
        specs = []
        extra = {}
        for term, w_from, w_to in st["accepted_this_round"]:
            var = TERMS[term][0]
            extra[var] = str(w_from)
            specs.append(f"{term}:{w_from}:{w_to}:0:{RAMP_ITERS * STEPS_PER_ITER}")
        if specs:
            extra["KRABBY_PHASEOUT"] = ",".join(specs)
        st["run_no"] += 1
        tag = f"r{rnd}_{st['run_no']:03d}_bake"
        ev = env_for(st, rnd, extra)
        log(f"{tag}: bake to round boundary (+{ROUND_ITERS}, {len(st['accepted_this_round'])} notches)")
        ckpt, tlog = train_retry(ev, tag, ROUND_ITERS, st["bake_head"])
        if ckpt is None or ckpt == "ABORTED":
            report(f"## HALT — bake round {rnd} crashed/aborted", "halt bake")
            sys.exit(2)
        clock_settled = st["weights"]["reward_clock_schedule"]
        for term, _f, w_to in st["accepted_this_round"]:
            if term == "reward_clock_schedule":
                clock_settled = w_to
        pm = probe_metrics(tlog, clock_w=clock_settled, settle_from=RAMP_ITERS)
        can = canary(ckpt, ev, tag)
        obst = obstacle_eval(ckpt, ev, tag)
        cpm, ccan = ctrl["probe"], ctrl["canary"]
        clock_ratio = pm["clock_unit"] / cpm["clock_unit"] if cpm["clock_unit"] else 0.0
        fail_delta = pm["failure"] - cpm["failure"]
        can_ratio = min(can["tripod"] / max(ccan["tripod"], 1e-9),
                        can["completion"] / max(ccan["completion"], 1e-9))
        trips = watchdogs(pm, cpm | {"canary": ccan}, can, tag)
        ok = clock_ratio >= 0.85 and fail_delta <= 0.10 and can_ratio >= 0.85 and not trips
        notches_txt = ", ".join(f"{t}:{f}->{w}" for t, f, w in st["accepted_this_round"]) or "none"
        lines = [f"## BAKE — round {rnd} boundary ({(rnd + 1) * 5}k)",
                 f"- notches: [{notches_txt}]",
                 f"- backstop vs round control: unit-clock {clock_ratio:.2f}x (>=0.85) | failure "
                 f"{fail_delta:+.3f} (<=+0.10) | canary ratio {can_ratio:.2f} (>=0.85)"
                 + (f" | watchdogs: {'; '.join(trips)}" if trips else ""),
                 f"- checkpoint: {Path(ckpt).parent.name}/{Path(ckpt).name}",
                 f"- flat canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
                 f"tracking {fmt(can['tracking'])} | slip {fmt(can['slip'])}",
                 f"- all-obstacle eval: completion {fmt(obst['completion'])} | tripod {fmt(obst['tripod'])}",
                 f"- **{'PASS' if ok else 'FAIL'}**"]
        if ok:
            report("\n".join(lines), f"bake round{rnd} PASS")
            for term, _f, w_to in st["accepted_this_round"]:
                st["weights"][term] = w_to
                st["notch_idx"][term] += 1
                st.setdefault("history", {}).setdefault(term, []).append([rnd, _f, w_to, "BAKED"])
            st["bake_head"] = ckpt
            st["last_bake_canary"] = can
            # RSI re-harvest — ONLY on a passing backstop (plan: a marginal bake keeps
            # the previous bank so degradation can't lock into the reset distribution).
            out = HERE / f"rsi_bank_pg_r{rnd}.npz"
            hev = {k: ev[k] for k in CANARY_KEEP if k in ev}
            hev["KRABBY_LIN_VEL_X"] = "0.25:0.35"
            henv = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
            henv.update(hev)
            with (HERE / f"{tag}_harvest.log").open("w") as fh:
                try:
                    subprocess.run([PY, str(HARVEST_TOOL), "--headless", "--num_envs", "16",
                                    "--steps", "800", "--checkpoint", ckpt, "--out", str(out)],
                                   cwd=PARKOUR, env=henv, stdout=fh, stderr=subprocess.STDOUT, timeout=900)
                except subprocess.TimeoutExpired:
                    log(f"{tag}: harvest timeout (teardown hang) — trusting the bank file check")
            if out.exists() and out.stat().st_size > 10000:
                st["bank"] = str(out)
                report(f"## RSI refresh — bank {out.name} adopted (backstop passed)", f"rsi r{rnd}")
            st["accepted_this_round"] = []
            save_state(st)
            return
        if not st["accepted_this_round"]:
            # No notches this round: the bake IS just the schedule continuation; a FAIL
            # here means the base schedule itself degraded — adopt with a warning.
            lines.append("- no notches to eject; adopting the boundary head with a WARNING")
            report("\n".join(lines), f"bake round{rnd} WARN")
            st["bake_head"] = ckpt
            st["last_bake_canary"] = can
            save_state(st)
            return
        costs = st.get("costs", {})
        costly = max(st["accepted_this_round"],
                     key=lambda tfw: costs.get(f"{tfw[0]}@{tfw[2]}", {}).get("fail_delta", 0)
                     + (1 - costs.get(f"{tfw[0]}@{tfw[2]}", {}).get("clock_ratio", 1)))
        lines.append(f"- ejecting most costly notch: **{costly[0]} -> {costly[2]}** — retries a later round")
        report("\n".join(lines), f"bake round{rnd} EJECT {costly[0]}")
        st["accepted_this_round"].remove(costly)
        st.setdefault("history", {}).setdefault(costly[0], []).append([rnd, costly[1], costly[2], "EJECTED"])
        save_state(st)
        accepted = list(st["accepted_this_round"])
        del accepted  # loop re-runs the reduced set


def phase_rounds(st: dict) -> None:
    while st["round"] <= st.get("max_round", MAX_ROUND):
        rnd = st["round"]
        if str(rnd) not in st["controls"]:
            run_control(st)
        if st.get("queue_round") != rnd:
            st["queue"] = next_candidates(st)
            st["queue_round"] = rnd
            save_state(st)
        while st["queue"]:
            term = st["queue"][0]
            w_from = st["weights"][term]
            w_to = TERMS[term][2][st["notch_idx"][term]]
            verdict = probe_candidate(st, term)
            if verdict == "PASS":
                st["accepted_this_round"].append([term, w_from, w_to])
            else:
                st.setdefault("history", {}).setdefault(term, []).append([rnd, w_from, w_to, "FAIL"])
            st["queue"].pop(0)
            save_state(st)
        bake_round(st)
        st["round"] += 1
        st["ref_probe"] = None
        st["ref_canary"] = None
        save_state(st)
    weights_txt = ", ".join(f"{t}={st['weights'][t]}" for t in TERM_ORDER)
    report(f"## ROUNDS COMPLETE — settled weights: {weights_txt} | head {st['bake_head']} — "
           f"terminal phase (RSI-off verification + primary-seed replay) awaits user go-ahead",
           "rounds complete")
    changelog(f"> NOTIFY: phase-out rounds complete — settled weights [{weights_txt}]; PAUSED before terminal phase")
    st["phase"] = "await_user_terminal"
    save_state(st)


# ------------------------------------------------------------------ phase: terminal
def _terminal_env_full_income(st: dict) -> dict:
    """Graduated schedule env (all elements) at FULL gait weights — the extended
    full-income control's configuration. Bank = P0-null per the confirmation-replay
    precedent (the reference lineage trained on it throughout)."""
    ev = dict(BASELINE)
    for r in sorted(SCHEDULE):
        ev.update(SCHEDULE[r])
    ev["KRABBY_RSI_BANK"] = str(LINEAGE / "rsi_bank_P0_null.npz")
    for term, (var, base, _n) in TERMS.items():
        ev[var] = str(base)
    return ev


# The discovered departure schedule (rounds result): per replay segment s (window
# [5k*s, 5k*(s+1)]), the gait-weight start values and the KRABBY_PHASEOUT ramps.
REPLAY_SCHEDULE = [
    # seg, start weights {term: w}, ramps [(term, w0, w1)]
    (0, {"reward_clock_schedule": 1.0, "reward_clock_swing_apex": 1.0,
         "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5}, []),
    (1, {"reward_clock_schedule": 1.0, "reward_clock_swing_apex": 1.0,
         "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5},
     [("reward_clock_swing_apex", 1.0, 0.5), ("reward_feet_air_time_positive", 0.8, 0.4),
      ("reward_stride_length", 0.5, 0.25)]),
    (2, {"reward_clock_schedule": 1.0, "reward_clock_swing_apex": 0.5,
         "reward_feet_air_time_positive": 0.4, "reward_stride_length": 0.25},
     [("reward_clock_swing_apex", 0.5, EPS), ("reward_feet_air_time_positive", 0.4, EPS),
      ("reward_stride_length", 0.25, EPS)]),
    (3, {"reward_clock_schedule": 1.0, "reward_clock_swing_apex": EPS,
         "reward_feet_air_time_positive": EPS, "reward_stride_length": EPS},
     [("reward_clock_schedule", 1.0, 0.5)]),
    (4, {"reward_clock_schedule": 0.5, "reward_clock_swing_apex": EPS,
         "reward_feet_air_time_positive": EPS, "reward_stride_length": EPS},
     [("reward_clock_schedule", 0.5, 0.2)]),
    (5, {"reward_clock_schedule": 0.2, "reward_clock_swing_apex": EPS,
         "reward_feet_air_time_positive": EPS, "reward_stride_length": EPS},
     [("reward_clock_schedule", 0.2, EPS)]),
]


def phase_terminal(st: dict) -> None:
    _LIVE_FAIL_CEILING[0] = 0.9
    ts = st.setdefault("terminal", {})
    final_head = st["bake_head"]
    final_can = st["last_bake_canary"]

    # --- step 1: extended full-income control (reference head 20k -> 30k) --------
    if "ext_head" not in ts:
        head = ts.get("ext_mid", REF_HEAD)
        start_seg = 2 if "ext_mid" in ts else 1
        for s in range(start_seg, 3):
            st["run_no"] += 1
            tag = f"t_{st['run_no']:03d}_extctrl_seg{s}"
            ev = _terminal_env_full_income(st)
            log(f"{tag}: extended full-income control segment {s}/2 ({15 + 5 * s}k -> {20 + 5 * s}k)")
            head2, tlog = train_retry(ev, tag, ROUND_ITERS, head)
            if head2 is None or head2 == "ABORTED":
                report(f"## HALT — extended control segment {s} crashed/aborted", "halt extctrl")
                sys.exit(2)
            head = head2
            if s == 1:
                ts["ext_mid"] = head
            else:
                ts["ext_head"] = head
                ts["ext_probe"] = probe_metrics(tlog, clock_w=1.0)
            save_state(st)
        can = canary(ts["ext_head"], _terminal_env_full_income(st), "t_extctrl_final")
        obst = obstacle_eval(ts["ext_head"], _terminal_env_full_income(st), "t_extctrl_final")
        ts["ext_canary"] = can
        ts["ext_obst"] = obst
        save_state(st)
        report("\n".join([
            "## TERMINAL — extended full-income control at 30k (matched iterations)",
            f"- checkpoint: {Path(ts['ext_head']).parent.name}/{Path(ts['ext_head']).name}",
            f"- probe (25k-30k tail): unit-clock {fmt(ts['ext_probe']['clock_unit'])} | track "
            f"{fmt(ts['ext_probe']['track'])} | failure {fmt(ts['ext_probe']['failure'])}",
            f"- flat canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
            f"tracking {fmt(can['tracking'])} | slip {fmt(can['slip'])}",
            f"- all-obstacle eval: completion {fmt(obst['completion'])} | tripod {fmt(obst['tripod'])}",
        ]), "terminal extctrl")

    # --- step 2: RSI-off verification of the phase-out final head ----------------
    if "rsi_off" not in ts:
        st["run_no"] += 1
        tag = f"t_{st['run_no']:03d}_rsioff"
        ev = env_for(st, rnd=5)
        ev["KRABBY_RSI_FRAC"] = "0"
        log(f"{tag}: RSI-off hold (+{PROBE_ITERS} from the final head, RSI_FRAC=0)")
        head, tlog = train_retry(ev, tag, PROBE_ITERS, final_head)
        if head is None or head == "ABORTED":
            report("## TERMINAL — RSI-off hold crashed/aborted: verification FAILED "
                   "(gait not shown self-sustaining without RSI)", "terminal rsioff FAIL")
            ts["rsi_off"] = {"verdict": "FAIL-crash"}
            save_state(st)
        else:
            pm = probe_metrics(tlog, clock_w=EPS)
            can = canary(head, ev, tag)
            ratio = min(can["tripod"] / max(final_can["tripod"], 1e-9),
                        can["completion"] / max(final_can["completion"], 1e-9),
                        can["tracking"] / max(final_can["tracking"], 1e-9))
            creep = can["completion"] >= 0.90 and can["tracking"] < 0.40
            ok = ratio >= 0.85 and not creep
            ts["rsi_off"] = {"verdict": "PASS" if ok else "FAIL", "canary": can, "ratio": ratio}
            save_state(st)
            report("\n".join([
                "## TERMINAL — RSI-off verification (+2k hold, KRABBY_RSI_FRAC=0)",
                f"- canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
                f"tracking {fmt(can['tracking'])} (vs final bake {fmt(final_can['tripod'])}/"
                f"{fmt(final_can['completion'])}/{fmt(final_can['tracking'])}) | ratio {ratio:.2f}",
                f"- probe: unit-clock {fmt(pm['clock_unit'])} | failure {fmt(pm['failure'])}",
                f"- **{'PASS — gait self-sustains without RSI resets' if ok else 'FAIL — head depends on RSI resets'}**",
            ]), f"terminal rsioff {ts['rsi_off']['verdict']}")

    # --- step 3: primary-seed replay of the full combined schedule ---------------
    seg_done = ts.get("replay_seg", -1)
    if "replay_head" not in ts:
        head = ts.get("replay_mid")
        for s, weights, ramps in REPLAY_SCHEDULE:
            if s <= seg_done:
                continue
            st["run_no"] += 1
            tag = f"t_{st['run_no']:03d}_replay_seg{s}"
            ev = dict(BASELINE)
            for r in sorted(SCHEDULE):
                if r <= s:
                    ev.update(SCHEDULE[r])
            ev["KRABBY_RSI_BANK"] = str(LINEAGE / "rsi_bank_P0_null.npz")
            for term, w in weights.items():
                ev[TERMS[term][0]] = str(w)
            if ramps:
                ev["KRABBY_PHASEOUT"] = ",".join(
                    f"{t}:{w0}:{w1}:0:{RAMP_ITERS * STEPS_PER_ITER}" for t, w0, w1 in ramps)
            log(f"{tag}: combined-schedule replay segment {s}/5 (seed {CONFIRM_SEED})")
            head2, tlog = train_retry(ev, tag, ROUND_ITERS, head, seed=CONFIRM_SEED)
            if head2 is None or head2 == "ABORTED":
                report(f"## TERMINAL — replay segment {s} crashed/aborted; schedule "
                       f"reproducibility NOT confirmed", "terminal replay FAIL")
                ts["replay_seg"] = s - 1
                save_state(st)
                sys.exit(2)
            head = head2
            ts["replay_mid"] = head
            ts["replay_seg"] = s
            save_state(st)
        ts["replay_head"] = head
        ev_final = env_for(st, rnd=5)
        can = canary(head, ev_final, "t_replay_final")
        obst = obstacle_eval(head, ev_final, "t_replay_final")
        ok = (can["tripod"] >= 0.85 * final_can["tripod"]
              and can["completion"] >= 0.85 * final_can["completion"])
        ts["replay_canary"] = can
        ts["replay_obst"] = obst
        ts["replay_verdict"] = "CONFIRMED" if ok else "DIVERGED"
        save_state(st)
        report("\n".join([
            f"## TERMINAL — combined-schedule replay (seed {CONFIRM_SEED}, 0 -> 30k)",
            f"- head: {Path(head).parent.name}/{Path(head).name}",
            f"- canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
            f"tracking {fmt(can['tracking'])} (vs primary phase-out {fmt(final_can['tripod'])}/"
            f"{fmt(final_can['completion'])}/{fmt(final_can['tracking'])})",
            f"- all-obstacle eval: completion {fmt(obst['completion'])} | tripod {fmt(obst['tripod'])}",
            f"- **{ts['replay_verdict']}**",
        ]), f"terminal replay {ts['replay_verdict']}")

    # --- verdict ------------------------------------------------------------------
    ec, eo = ts["ext_canary"], ts["ext_obst"]
    fo_log = HERE / "r5_022_bake_train.log"
    pm_final = probe_metrics(fo_log, clock_w=EPS, settle_from=RAMP_ITERS)
    final_obst = {"completion": 0.270, "tripod": 0.604}  # r5 bake block
    win_obstacle = final_obst["completion"] >= eo["completion"] + 0.05
    win_tracking = final_can["tracking"] >= ec["tracking"] + 0.10
    tripod_hold = final_can["tripod"] >= ec["tripod"] - 0.05
    fail_hold = pm_final["failure"] <= ts["ext_probe"]["failure"] + 0.05
    compl_hold = final_can["completion"] >= ec["completion"] - 0.05
    rsi_ok = ts["rsi_off"].get("verdict") == "PASS"
    replay_ok = ts.get("replay_verdict") == "CONFIRMED"
    win = (win_obstacle or win_tracking) and tripod_hold and fail_hold and compl_hold and rsi_ok and replay_ok
    lines = [
        "## TERMINAL VERDICT — phase-out 30k head vs extended full-income control (matched iterations)",
        "| criterion | phase-out | full-income ctrl | verdict |", "|---|---|---|---|",
        f"| obstacle completion (>= ctrl+0.05) | {fmt(final_obst['completion'])} | {fmt(eo['completion'])} | {'WIN' if win_obstacle else 'no'} |",
        f"| canary tracking (>= ctrl+0.10) | {fmt(final_can['tracking'])} | {fmt(ec['tracking'])} | {'WIN' if win_tracking else 'no'} |",
        f"| tripod (>= ctrl-0.05) | {fmt(final_can['tripod'])} | {fmt(ec['tripod'])} | {'HOLD' if tripod_hold else 'FAIL'} |",
        f"| failure tail (<= ctrl+0.05) | {fmt(pm_final['failure'])} | {fmt(ts['ext_probe']['failure'])} | {'HOLD' if fail_hold else 'FAIL'} |",
        f"| completion (within 0.05) | {fmt(final_can['completion'])} | {fmt(ec['completion'])} | {'HOLD' if compl_hold else 'FAIL'} |",
        f"| RSI-off verification | {ts['rsi_off'].get('verdict')} | — | {'PASS' if rsi_ok else 'FAIL'} |",
        f"| seed-{CONFIRM_SEED} replay | {ts.get('replay_verdict')} | — | {'PASS' if replay_ok else 'FAIL'} |",
        f"- **TERMINAL: {'WIN — bake proposal ready' if win else 'NOT A WIN under the plan criteria — results stand as measured'}**",
        "- Reminder: gait persisted at unit-clock 0.91x with ALL gait income at eps; the clock",
        "  OBSERVATION remains in the policy input (claims are about income, not the clock signal).",
    ]
    report("\n".join(lines), f"terminal verdict {'WIN' if win else 'RESULT'}")
    changelog(f"> NOTIFY: terminal phase complete — {'WIN' if win else 'result recorded'}; "
              f"bake decision is the user's")
    st["phase"] = "done"
    save_state(st)


def main() -> int:
    args = set(sys.argv[1:])
    st = load_state()
    if "--start-rounds" in args and st["phase"] == "await_user":
        st["phase"] = "round"
        save_state(st)
    if "--start-terminal" in args and st["phase"] == "await_user_terminal":
        st["phase"] = "terminal"
        save_state(st)
    if st["phase"] == "canaries":
        phase_canaries(st)
    if st["phase"] == "scout":
        phase_scout(st)
    if st["phase"] == "await_user":
        log("phase await_user: scout done — waiting for the user decision (relaunch with "
            "--start-rounds to begin the notch search)")
        return 0
    if st["phase"] == "round":
        phase_rounds(st)
    if st["phase"] == "await_user_terminal":
        log("phase await_user_terminal: rounds complete — relaunch with --start-terminal "
            "to run the terminal phase")
        return 0
    if st["phase"] == "terminal":
        phase_terminal(st)
    if st["phase"] == "done":
        log("phase done: campaign complete — bake decision is the user's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
