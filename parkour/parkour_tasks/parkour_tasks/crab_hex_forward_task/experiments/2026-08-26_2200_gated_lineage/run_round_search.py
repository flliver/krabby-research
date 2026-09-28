#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Round-based sequential-accumulation curriculum search (approved plan, 2026-08-27).

Places every curriculum element at the earliest 5k round it tolerates (0/5k/10k/15k,
cap 20k), one element at a time, with verdicts RELATIVE to each round's own no-change
baseline control. Detailed reporting per the plan's §5 contract: REPORT.md blocks for
every baseline, element decision, and bake — each terminated by a '>>> ENTRY' marker
line that the chat monitor relays.

Stdlib only; resumable via state_v2.json.
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
RESET_TOOL = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-17_1500_phased_flat/B1_critic_reset/make_critic_reset.py"
HARVEST_TOOL = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/harvest_rsi_bank.py"
MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
FLAT_RUNS = PARKOUR / "logs/rsl_rl/crab_hex_flat_walk"
EVAL_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/gated_lineage"
STATE = HERE / "state_v2.json"
REPORT = HERE / "REPORT.md"
CHANGELOG = HERE / "CHANGELOG.md"

PROBE_ITERS = 2000
ESCALATE_ITERS = 3000   # probe continue: 2k -> 5k
EXTEND_ITERS = 2000
BAKE_ITERS = 5000
MAX_ROUND = 3           # rounds at 0 / 5k / 10k / 15k; cap 20k
SEED = "2"
CONFIRM_SEED = "3"

BASELINE = {
    "KRABBY_CLOCK_W": "1.0",
    "KRABBY_APEX_W": "1.0",
    "KRABBY_RSI_FRAC": "0.2",
    "KRABBY_RSI_BANK": str(HERE / "rsi_bank_P0_null.npz"),
    "KRABBY_FLAT_TERRAIN_MODE": "light",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    "KRABBY_TRACK_SIGMA2": "0.1",
    "KRABBY_TRACK_L1_W": "-1.0",
    "KRABBY_LIN_VEL_X": "0.0:0.35",
}
CANARY_KEEP = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W",
               "KRABBY_APEX_W", "KRABBY_HEADING", "KRABBY_HEADING_STIFFNESS", "KRABBY_ACTION_SCALE")

ELEMENTS = [
    ("turning", {"KRABBY_HEADING": "-1.2:1.2"}, False, None),
    ("episode40", {"KRABBY_EPISODE_S": "40"}, False, None),
    ("yaw_income", {"KRABBY_YAW_W": "0.2"}, True, "reward_tracking_yaw"),
    ("goalvel_income", {"KRABBY_GOAL_VEL_W": "0.75"}, True, "reward_tracking_goal_vel"),
    ("terrain50", {"KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5"}, False, None),
    ("terrain_recal", {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70"}, False, None),
    ("terrain_curriculum", {"KRABBY_FLAT_TERRAIN_CURRICULUM": "1", "KRABBY_TERRAIN_PROMOTE": "0.45:0.25"}, False, None),
    ("safety_pack", {"KRABBY_EDGE_W": "-0.3", "KRABBY_STUMBLE_W": "-1.0", "KRABBY_COLLISION_W": "-2.0"}, True, None),
    ("clearance_pack", {"KRABBY_CLEARANCE_W": "0.9", "KRABBY_FOOT_CLEAR_FLAT": "1", "KRABBY_FOOT_CLEAR_W": "1.0",
                        "KRABBY_FOOT_CLEAR_MIN": "0.03", "KRABBY_SWING_MIN_CLEAR_W": "-0.4"}, True,
     "reward_obstacle_clearance"),
    ("speed_band", {"KRABBY_LIN_VEL_X": "0.0:0.55"}, False, None),
    ("dr_push", {"KRABBY_DR_PUSH": "0.5"}, False, None),
    ("dr_masscom", {"KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_COM": "0.01"}, False, None),
]
EL = {name: (vars_, reset, income) for name, vars_, reset, income in ELEMENTS}
ORDER = [name for name, *_ in ELEMENTS]


# ------------------------------------------------------------------ infrastructure
def log(msg: str) -> None:
    print(f"[rounds {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def report(block: str, marker: str) -> None:
    with REPORT.open("a") as fh:
        fh.write(block.rstrip() + f"\n>>> ENTRY {marker}\n\n")


def changelog(row: str) -> None:
    with CHANGELOG.open("a") as fh:
        fh.write(row.rstrip() + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"phase": "round", "round": 0, "accepted": {}, "pending": list(ORDER), "failed": [],
            "bake_head": None, "cand_idx": 0, "run_no": 0, "controls": {}, "anchor_done": False,
            "bank": BASELINE["KRABBY_RSI_BANK"]}


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


# Every Isaac subprocess runs in its OWN memory-capped scope (2026-08-28): the sporadic
# boot balloon then OOM-kills only that subprocess, never the orchestrator — whose unit
# is uncapped and tiny. --collect garbage-collects failed scopes.
# NOTE 2026-08-28: memory-capped scopes were tried and REVERTED — a MemoryMax scope
# SIGKILLs Isaac at spawn on this box (mechanism unidentified; small processes
# survive, Isaac dies instantly with rc 137). Isaac subprocesses run unscoped in
# the uncapped unit; OOMPolicy=continue + kernel OOM (kills the biggest consumer)
# + freshness gates + retries provide the balloon protection instead.
SCOPE = []


def _wait_isaac_clear() -> None:
    """Block (bounded) until no trainer/eval Isaac process remains — a retry boot that
    overlaps a dying Isaac's teardown gets OOM-killed by the combined footprint."""
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", "rsl_rl/train.py|eval_crab_hex_gait"],
                          capture_output=True).returncode != 0:
            return
        time.sleep(10)


def train(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None, seed: str = SEED) -> tuple[str | None, Path]:
    t0 = time.time()
    log_path = HERE / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", "Isaac-Crab-Hex-Flat-Walk-v0", "--headless",
           "--num_envs", "256", "--seed", seed, "--max_iterations", str(iters)]
    if resume_ckpt:
        cmd += ["--resume", "--checkpoint", resume_ckpt]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update({k: str(v) for k, v in env_vars.items()})
    with log_path.open("w") as fh:
        proc = subprocess.Popen(SCOPE + cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            time.sleep(120)
            # Wedge watchdog (2026-08-29): an OOM-killed trainer left a zombie child
            # the poll loop never reaped, freezing the campaign for 40+ min. A healthy
            # trainer logs every ~1.5 s — 15 min of log silence means it is dead/hung.
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
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, log_path
    if proc.returncode == 0:
        return latest_run_ckpt(), log_path
    # Teardown-crash salvage (2026-08-28): trainers sometimes complete every iteration,
    # save the final checkpoint, then die in Isaac teardown with rc != 0 — trust the
    # on-disk checkpoint over the exit code (the r0_016 cumcheck lost a finished 5k run
    # to this; same lesson as the RSI harvest). Chunks resume cumulatively, so the
    # expected final index is resume index + iters.
    try:
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
                seed: str = SEED) -> tuple[str | None, Path]:
    """train() with ONE retry when the death looks infrastructural (no Python
    traceback in the log — e.g. the sporadic Isaac boot memory balloon observed
    2026-08-27, which OOM-kills a run before any task code executes). A death
    WITH a traceback is a real config/code error and is never retried."""
    ckpt, tlog = train(env_vars, tag, iters, resume_ckpt, seed)
    for retry in (2, 3):
        if ckpt is not None:
            break
        txt = tlog.read_text(errors="ignore") if tlog.exists() else ""
        if "Traceback" in txt or "Error executing job" in txt:
            break  # real config/code error — never retried
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
    # Boot balloons arrive in storms (several consecutive boots OOM within minutes,
    # then clear — observed 2026-08-27/28): ride a storm out with generous attempts
    # and cool-downs rather than giving up after two.
    for attempt in range(4):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run(SCOPE + [PY, str(EVAL), "--headless", "--manifest", str(MANIFEST),
                                "--scenario", "flat_walk_slow_v2", "--checkpoint", ckpt, "--no-plot",
                                "--output-root", str(EVAL_ROOT)],
                               cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
            except subprocess.TimeoutExpired:
                pass  # freshness gate below decides; a hung eval may still have written metrics
        d = max((EVAL_ROOT / "flat_walk_slow_v2" / "seed001").iterdir(), key=lambda p: p.stat().st_mtime)
        # Freshness gate: a killed eval (boot balloon) must not verdict on the PREVIOUS
        # eval's stale metrics dir.
        if d.stat().st_mtime >= t0 and (d / "scenario_metrics.json").exists():
            a = json.loads((d / "scenario_metrics.json").read_text())["aggregate"]
            return {"tripod": (a.get("tripod_score") or {}).get("median") or 0.0,
                    "completion": a.get("schedule_completion_rate") or 0.0,
                    "tracking": (a.get("tracking_ratio") or {}).get("median") or 0.0}
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
            subprocess.run(SCOPE + [PY, str(EVAL), "--headless", "--manifest", str(MANIFEST),
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


def probe_metrics(tlog: Path, at_prints: int | None = None) -> dict:
    clock = series(tlog, "Episode_Reward/reward_clock_schedule")
    fail = series(tlog, "Episode_Termination/crab_failure")
    eplen = series(tlog, "Mean episode length")
    if at_prints:
        clock, fail, eplen = clock[:at_prints], fail[:at_prints], eplen[:at_prints]
    return {"clock": tail_mean(clock), "failure": tail_mean(fail), "eplen": tail_mean(eplen),
            "clock_trend": (tail_mean(clock, 100) - tail_mean(clock[:-100] or clock, 100)) if len(clock) > 200 else 0.0,
            "fail_trend": (tail_mean(fail, 100) - tail_mean(fail[:-100] or fail, 100)) if len(fail) > 200 else 0.0}


def capability_gap(tlog: Path, income_label: str | None) -> bool:
    if not income_label:
        return False
    inc = series(tlog, f"Episode_Reward/{income_label}")
    fail = series(tlog, "Episode_Termination/crab_failure")
    if len(inc) < 600 or len(fail) < 600:
        return False
    third = len(inc) // 3
    iu = [sum(abs(x) for x in inc[k:k + third]) / third for k in (0, third, 2 * third)]
    fu = [sum(fail[k:k + third]) / third for k in (0, third, 2 * third)]
    return iu[0] < iu[1] < iu[2] and fu[0] < fu[1] < fu[2] and fu[2] > 0.5


def env_for(st: dict, extra: dict | None = None) -> dict:
    ev = dict(BASELINE)
    ev["KRABBY_RSI_BANK"] = st["bank"]
    for name in st["accepted"]:
        ev.update(EL[name][0])
    if extra:
        ev.update(extra)
    return ev


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# ------------------------------------------------------------------ report blocks
def report_baseline(rnd: int, pm: dict, can: dict, tag: str, anchor: bool) -> None:
    lines = [f"## BASELINE control — round {rnd} ({tag})",
             f"- probe @2k: clock income {fmt(pm['clock'])} | failure tail {fmt(pm['failure'])} | "
             f"ep len {fmt(pm['eplen'])} | trends clock {pm['clock_trend']:+.3f} fail {pm['fail_trend']:+.3f}",
             f"- flat canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
             f"tracking {fmt(can['tracking'])}"]
    if anchor:
        lines.append("- sanity vs historical formation calibration (formed: clock 0.40–0.43 / fail 0.15–0.26): "
                     + ("WITHIN band" if pm["clock"] >= 0.38 and pm["failure"] <= 0.30 else "OUTSIDE band — review"))
    report("\n".join(lines), f"baseline round{rnd}")


def report_decision(name: str, rnd: int, pm: dict, cpm: dict, verdict: str, rule: str,
                    esc: str, can: dict | None, ccan: dict | None) -> None:
    lines = [f"## ELEMENT decision — {name} @ round {rnd}",
             f"| metric | candidate | control | margin |", "|---|---|---|---|",
             f"| clock income @2k | {fmt(pm['clock'])} | {fmt(cpm['clock'])} | "
             f"{(pm['clock'] / cpm['clock'] if cpm['clock'] else float('nan')):.2f}x |",
             f"| failure tail @2k | {fmt(pm['failure'])} | {fmt(cpm['failure'])} | "
             f"{pm['failure'] - cpm['failure']:+.3f} |",
             f"| ep len @2k | {fmt(pm['eplen'])} | {fmt(cpm['eplen'])} | "
             f"{pm['eplen'] - cpm['eplen']:+.0f} |"]
    if can and ccan:
        lines.append(f"| canary t/c/tr | {fmt(can['tripod'])}/{fmt(can['completion'])}/{fmt(can['tracking'])} | "
                     f"{fmt(ccan['tripod'])}/{fmt(ccan['completion'])}/{fmt(ccan['tracking'])} | ratio "
                     f"{min(can['tripod'] / max(ccan['tripod'], 1e-9), can['completion'] / max(ccan['completion'], 1e-9), can['tracking'] / max(ccan['tracking'], 1e-9)):.2f} |")
    if esc:
        lines.append(f"- escalation history: {esc}")
    lines.append(f"- **DECISION: {verdict}** — {rule}")
    report("\n".join(lines), f"decision {name} round{rnd} {verdict}")


def report_bake(rnd: int, st: dict, ckpt: str, can: dict, obst: dict, prev: dict | None) -> None:
    placements = ", ".join(f"{n}@r{r}" for n, r in sorted(st["accepted"].items(), key=lambda kv: kv[1]))
    lines = [f"## BAKE C{rnd + 1} — through iteration {(rnd + 1) * 5}k",
             f"- configuration: baseline core + [{placements or 'none'}]",
             f"- env stack: {json.dumps(env_for(st), sort_keys=True)}",
             f"- checkpoint: {Path(ckpt).parent.name}/{Path(ckpt).name}",
             f"- flat canary: tripod {fmt(can['tripod'])} | completion {fmt(can['completion'])} | "
             f"tracking {fmt(can['tracking'])}"
             + (f" (Δ vs prev bake {can['tripod'] - prev['tripod']:+.3f}/{can['completion'] - prev['completion']:+.2f})"
                if prev else ""),
             f"- all-obstacle eval: completion {fmt(obst['completion'])} | tripod {fmt(obst['tripod'])}"]
    report("\n".join(lines), f"bake C{rnd + 1}")


# ------------------------------------------------------------------ main phases
def probe_candidate(st: dict, name: str) -> str:
    """Run one candidate at the current round; return PASS|RETRY|
    (verdict recorded via report)."""
    rnd = st["round"]
    vars_, reset, income = EL[name]
    ctrl = st["controls"][str(rnd)]
    # Rolling reference (2026-08-27, user): the probe reference is the LAST PASSING run's
    # metrics — every verdict is a one-element delta vs its true counterfactual (the
    # accepted set without the candidate). The round control seeds it; each PASS rolls it
    # forward. The canary reference stays the round control's (escalations re-baseline it
    # lazily via ref_canary when a passing run produced one).
    cpm = st.get("ref_probe") or ctrl["probe"]
    ccan = st.get("ref_canary") or ctrl["canary"]
    st["run_no"] += 1
    tag = f"r{rnd}_{st['run_no']:03d}_{name}"
    resume = None if rnd == 0 else st["bake_head"]
    if reset and resume:
        rd = str(Path(resume).with_suffix("")) + "_critic_reset.pt"
        subprocess.run([PY, str(RESET_TOOL), "--src", resume, "--dst", rd], check=True, capture_output=True)
        resume = rd
    ev = env_for(st, vars_)
    log(f"{tag}: probe (round {rnd}, resume={'scratch' if resume is None else 'bake'})")
    ckpt, tlog = train_retry(ev, tag, PROBE_ITERS, resume)
    if ckpt is None:
        # A crash is a config/code error, never a formation verdict — retrying it each
        # round would launder a bug into a FAILED placement. Halt for a human/assistant
        # fix; the candidate stays at the queue head for the relaunch.
        report(f"## HALT — {tag} crashed before producing a checkpoint (config/code "
               f"error, not a formation verdict). {name} stays queued at round {rnd}; "
               f"fix and relaunch.", f"HALT crash {name} round{rnd}")
        changelog(f"| HALT | {tag} training crashed | — | — | — | — |")
        raise SystemExit(3)
    pm = probe_metrics(tlog)
    gap = capability_gap(tlog, income)
    clock_ratio = pm["clock"] / cpm["clock"] if cpm["clock"] else 0.0
    fail_delta = pm["failure"] - cpm["failure"]
    norm = ""
    # Hazard normalization (2026-08-29): when a candidate shifts episode length >25%
    # (episode40), per-EPISODE failure is denominator-biased — twice the episode is
    # twice the exposure at the same per-step hazard, and the bias re-fires every
    # round. Judge the per-step hazard scaled to the reference's episode length.
    if cpm.get("eplen") and pm.get("eplen") and abs(pm["eplen"] - cpm["eplen"]) > 0.25 * cpm["eplen"]:
        fail_delta = (pm["failure"] / pm["eplen"] - cpm["failure"] / cpm["eplen"]) * cpm["eplen"]
        norm = " [failure hazard-normalized for episode-length shift]"
    esc = ""
    if gap:
        report_decision(name, rnd, pm, cpm, "RETRY",
                        "capability-gap signature: element income rising while failure rises", esc, None, None)
        return "RETRY"
    if clock_ratio >= 0.95 and fail_delta <= 0.05:
        report_decision(name, rnd, pm, cpm, "PASS",
                        f"clock {clock_ratio:.2f}x >= 0.95 and failure {fail_delta:+.3f} <= +0.05{norm}", esc, None, None)
        st["ref_probe"] = pm
        st.setdefault("costs", {})[name] = {"clock_ratio": clock_ratio, "fail_delta": fail_delta}
        st["last_pass_ckpt"] = ckpt
        return "PASS"
    if clock_ratio < 0.85 or fail_delta > 0.10:
        report_decision(name, rnd, pm, cpm, "RETRY",
                        f"clock {clock_ratio:.2f}x < 0.85 or failure {fail_delta:+.3f} > +0.10 "
                        f"(one-element delta vs control){norm}", esc, None, None)
        return "RETRY"
    # ambiguous -> escalate: continue same run to 5k, compare canaries
    esc = f"ambiguous at 2k (clock {clock_ratio:.2f}x, fail {fail_delta:+.3f}{norm}); continued to 5k"
    ckpt2, tlog2 = train_retry(ev, tag + "_esc", ESCALATE_ITERS, ckpt)
    if ckpt2 is None:
        report(f"## HALT — {tag} escalation crashed (config/code error, not a formation "
               f"verdict). {name} stays queued at round {rnd}; fix and relaunch.",
               f"HALT crash {name} round{rnd}")
        changelog(f"| HALT | {tag} escalation crashed | — | — | — | — |")
        raise SystemExit(3)
    can = canary(ckpt2, ev, tag + "_esc")
    ratio = min(can["tripod"] / max(ccan["tripod"], 1e-9),
                can["completion"] / max(ccan["completion"], 1e-9),
                can["tracking"] / max(ccan["tracking"], 1e-9))
    if ratio >= 0.85:
        report_decision(name, rnd, pm, cpm, "PASS",
                        f"escalated canary ratio {ratio:.2f} >= 0.85 of control", esc, can, ccan)
        st["ref_probe"] = pm
        st["ref_canary"] = can
        st.setdefault("costs", {})[name] = {"clock_ratio": clock_ratio, "fail_delta": fail_delta}
        st["last_pass_ckpt"] = ckpt2
        return "PASS"
    # one trend extension
    trip_series = series(tlog2, "Episode_Reward/reward_clock_schedule")
    if len(trip_series) > 400 and tail_mean(trip_series, 100) > 1.1 * tail_mean(trip_series[:-100], 100):
        esc += "; trend extension +2k"
        ckpt3, _ = train_retry(ev, tag + "_ext", EXTEND_ITERS, ckpt2)
        if ckpt3:
            can = canary(ckpt3, ev, tag + "_ext")
            ratio = min(can["tripod"] / max(ccan["tripod"], 1e-9),
                        can["completion"] / max(ccan["completion"], 1e-9),
                        can["tracking"] / max(ccan["tracking"], 1e-9))
            if ratio >= 0.85:
                report_decision(name, rnd, pm, cpm, "PASS",
                                f"post-extension canary ratio {ratio:.2f} >= 0.85", esc, can, ccan)
                st["ref_probe"] = pm
                st["ref_canary"] = can
                st.setdefault("costs", {})[name] = {"clock_ratio": clock_ratio, "fail_delta": fail_delta}
                st["last_pass_ckpt"] = ckpt3
                return "PASS"
    report_decision(name, rnd, pm, cpm, "RETRY (AMBIGUOUS)",
                    f"canary ratio {ratio:.2f} < 0.85 after escalation — conservative retry next round",
                    esc, can, ccan)
    return "RETRY"


def run_control(st: dict) -> None:
    rnd = st["round"]
    st["run_no"] += 1
    tag = f"r{rnd}_{st['run_no']:03d}_control"
    ev = env_for(st)
    if rnd == 0:
        log(f"{tag}: ANCHOR run (baseline core, from scratch, 5k)")
        ckpt, tlog = train_retry(ev, tag, BAKE_ITERS, None)
        if ckpt is None:
            changelog("| HALT | anchor run crashed | — | — | — | — |")
            report("## HALT — anchor run crashed", "halt anchor")
            sys.exit(2)
        pm = probe_metrics(tlog, at_prints=PROBE_ITERS)
        can = canary(ckpt, ev, tag)
        st["anchor_ckpt"] = ckpt
    else:
        log(f"{tag}: round-{rnd} control (+2k on bake, no changes)")
        ckpt, tlog = train_retry(ev, tag, PROBE_ITERS, st["bake_head"])
        if ckpt is None:
            changelog(f"| HALT | round-{rnd} control crashed | — | — | — | — |")
            report(f"## HALT — round-{rnd} control crashed", "halt control")
            sys.exit(2)
        pm = probe_metrics(tlog)
        can = canary(ckpt, ev, tag)
    st["controls"][str(rnd)] = {"probe": pm, "canary": can}
    st["ref_probe"] = pm
    st["ref_canary"] = can
    report_baseline(rnd, pm, can, tag, anchor=(rnd == 0))
    save_state(st)


def cumulative_check(st: dict) -> None:
    """Round-end absolute backstop (user, 2026-08-27): the FULL accepted set must stay
    within GENEROUS bands of the round control (clock >= 0.85x, failure <= +0.10, canary
    >= 0.85x) — the per-element clear-fail lines become the cumulative pass lines. On
    failure, eject the most costly of THIS round's acceptances (by recorded marginal
    cost), move it to the next round, re-run the reduced set, repeat until passing."""
    rnd = st["round"]
    ctrl = st["controls"][str(rnd)]
    round_accepts = [n for n, r in st["accepted"].items() if r == rnd]
    if not round_accepts:
        return
    # FIX 2026-08-28: the check must train the full set to BAKE_ITERS so its canary is
    # duration-matched against the control's (a 2k-trained canary vs the 5k control
    # canary ejected healthy elements — safety_pack false ejection, round 0). The
    # passing checkpoint doubles as the bake head (bake() adopts it via cum_ckpt), so
    # the fair comparison costs nothing extra on the passing iteration. The probe-axis
    # comparison reads the candidate's tail (more-trained, mildly lenient) — this
    # backstop is generous by design.
    while True:
        ev = env_for(st)
        st["run_no"] += 1
        tag = f"r{rnd}_{st['run_no']:03d}_cumcheck"
        resume = None if rnd == 0 else st["bake_head"]
        salv = st.pop("salvage_ckpt", None)
        if salv:
            # One-shot: adopt a manually salvaged finished run instead of retraining.
            # No save_state here — the on-disk salvage keys must survive a crash during
            # the canary/verdict below (they leave disk at the next successful save).
            tlog = Path(st.pop("salvage_log"))
            ckpt = salv
            log(f"{tag}: adopting salvaged checkpoint {Path(salv).name}")
        else:
            log(f"{tag}: cumulative check ({len(st['accepted'])} accepted, +{BAKE_ITERS})")
            ckpt, tlog = train_retry(ev, tag, BAKE_ITERS, resume)
        if ckpt is None:
            report("## CUMULATIVE check crashed — proceeding to bake with current set",
                   f"cumcheck round{rnd} crash")
            return
        pm = probe_metrics(tlog)
        can = canary(ckpt, ev, tag + "_cum")
        cpm, ccan = ctrl["probe"], ctrl["canary"]
        clock_ratio = pm["clock"] / cpm["clock"] if cpm["clock"] else 0.0
        fail_delta = pm["failure"] - cpm["failure"]
        can_ratio = min(can["tripod"] / max(ccan["tripod"], 1e-9),
                        can["completion"] / max(ccan["completion"], 1e-9))
        ok = clock_ratio >= 0.85 and fail_delta <= 0.10 and can_ratio >= 0.85
        lines = [f"## CUMULATIVE check — round {rnd} ({len(round_accepts)} round-accepts)",
                 f"- full set vs round control: clock {clock_ratio:.2f}x (>=0.85) | "
                 f"failure {fail_delta:+.3f} (<=+0.10) | canary ratio {can_ratio:.2f} (>=0.85)",
                 f"- set: {', '.join(sorted(st['accepted']))}",
                 f"- **{'PASS' if ok else 'FAIL'}**"]
        if ok:
            st["cum_ckpt"] = ckpt
            save_state(st)
            report("\n".join(lines), f"cumcheck round{rnd} PASS")
            return
        costs = st.get("costs", {})
        pool = [n for n in round_accepts if n in st["accepted"]]
        if not pool:
            lines.append("- no ejectable elements remain — proceeding (HALT-worthy anomaly)")
            report("\n".join(lines), f"cumcheck round{rnd} exhausted")
            return
        costly = max(pool, key=lambda n: costs.get(n, {}).get("fail_delta", 0)
                     + (1 - costs.get(n, {}).get("clock_ratio", 1)))
        cinfo = costs.get(costly, {})
        lines.append(f"- ejecting most costly: **{costly}** (marginal fail {cinfo.get('fail_delta', 0):+.3f}, "
                     f"clock {cinfo.get('clock_ratio', 1):.2f}x) -> retry next round; re-testing reduced set")
        report("\n".join(lines), f"cumcheck round{rnd} EJECT {costly}")
        del st["accepted"][costly]
        st["pending"].append(costly)
        round_accepts.remove(costly)
        save_state(st)
        if not round_accepts:
            # Every acceptance ejected — nothing left to verify; the bake proceeds with
            # no new elements (the round's control/anchor already IS that configuration).
            report("## CUMULATIVE check — all round acceptances ejected; baking without "
                   "new elements", f"cumcheck round{rnd} emptied")
            return


def bake(st: dict) -> None:
    rnd = st["round"]
    st["run_no"] += 1
    tag = f"r{rnd}_{st['run_no']:03d}_bake"
    ev = env_for(st)
    resume = None if rnd == 0 else st["bake_head"]
    ckpt = st.pop("cum_ckpt", None)
    if ckpt:
        # The cumulative check already trained this exact set to BAKE_ITERS and passed
        # the backstop on it — that run IS the bake.
        log(f"{tag}: adopting the cumulative-check head as C{rnd + 1}")
    else:
        log(f"{tag}: baking C{rnd + 1} ({'scratch' if resume is None else 'resume'} +5k, "
            f"{len(st['accepted'])} elements)")
        ckpt, tlog = train_retry(ev, tag, BAKE_ITERS, resume)
    if ckpt is None:
        changelog(f"| HALT | bake C{rnd + 1} crashed | — | — | — | — |")
        report(f"## HALT — bake C{rnd + 1} crashed", "halt bake")
        sys.exit(2)
    can = canary(ckpt, ev, tag)
    obst = obstacle_eval(ckpt, ev, tag)
    prev = st.get("last_bake_canary")
    report_bake(rnd, st, ckpt, can, obst, prev)
    st["bake_head"] = ckpt
    st["last_bake_canary"] = can
    # RSI refresh from the bake
    out = HERE / f"rsi_bank_C{rnd + 1}.npz"
    hev = {k: ev[k] for k in CANARY_KEEP if k in ev}
    hev["KRABBY_LIN_VEL_X"] = "0.25:0.35"
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update(hev)
    # The bank file lands within the first minutes; the teardown hang afterwards is
    # routine. A timeout RAISES (it is not an exit code) — swallow it and let the
    # bank-file check below decide.
    with (HERE / f"{tag}_harvest.log").open("w") as fh:
        try:
            subprocess.run(SCOPE + [PY, str(HARVEST_TOOL), "--headless", "--num_envs", "16",
                            "--steps", "800", "--checkpoint", ckpt, "--out", str(out)],
                           cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=900)
        except subprocess.TimeoutExpired:
            log(f"{tag}: harvest timeout (teardown hang) — trusting the bank file check")
    if out.exists() and out.stat().st_size > 10000:
        st["bank"] = str(out)
        report(f"## RSI refresh — bank {out.name} adopted for round {rnd + 1}", f"rsi C{rnd + 1}")
    save_state(st)


def main() -> int:
    if not REPORT.exists():
        REPORT.write_text("# Round-search REPORT (approved plan 2026-08-27)\n\n")
    st = load_state()
    while st["phase"] == "round":
        rnd = st["round"]
        if str(rnd) not in st["controls"]:
            run_control(st)
        # Resume-safe iteration: this round's not-yet-probed candidates live on disk in
        # "queue" (a mid-round restart must not lose them or re-probe verdicted ones);
        # "pending" accumulates next-round retries. queue_round marks which round the
        # queue was built for, so a crash after the queue drains doesn't rebuild it
        # from the retries.
        if st.get("queue_round") != rnd:
            st["queue"] = list(st["pending"])
            st["pending"] = []
            st["queue_round"] = rnd
            save_state(st)
        while st["queue"]:
            name = st["queue"][0]
            if name not in st["accepted"]:
                verdict = probe_candidate(st, name)
                if verdict == "PASS":
                    st["accepted"][name] = rnd
                else:
                    st["pending"].append(name)
            st["queue"].pop(0)
            save_state(st)
        cumulative_check(st)
        bake(st)
        if not st["pending"]:
            st["phase"] = "confirm"
        elif rnd >= MAX_ROUND:
            st["failed"] = list(st["pending"])
            st["pending"] = []
            report("## SEARCH COMPLETE — elements FAILED (out of scope to fix): "
                   + ", ".join(st["failed"]), "search failed-elements")
            st["phase"] = "confirm"
        else:
            st["round"] += 1
        save_state(st)
    if st["phase"] == "confirm":
        # second-seed schedule replay: segments 0..round, elements activating at placements
        placements = st["accepted"]
        head = None
        for r in range(st["round"] + 1):
            ev = dict(BASELINE)
            ev["KRABBY_RSI_BANK"] = str(HERE / "rsi_bank_P0_null.npz")
            for n, pr in placements.items():
                if pr <= r:
                    ev.update(EL[n][0])
            st["run_no"] += 1
            tag = f"confirm_{st['run_no']:03d}_seg{r}"
            log(f"{tag}: schedule replay segment {r} (seed {CONFIRM_SEED})")
            head, _ = train_retry(ev, tag, BAKE_ITERS, head, seed=CONFIRM_SEED)
            if head is None:
                report(f"## CONFIRMATION segment {r} crashed — placement at round {r} reopened",
                       "confirm crash")
                changelog(f"| HALT | confirmation segment {r} crashed | — | — | — | — |")
                return 2
        ev = env_for(st)
        can = canary(head, ev, "confirm_final")
        obst = obstacle_eval(head, ev, "confirm_final")
        base_can = st.get("last_bake_canary") or {"tripod": 0, "completion": 0, "tracking": 0}
        ok = (can["tripod"] >= 0.85 * base_can["tripod"]
              and can["completion"] >= 0.85 * base_can["completion"])
        report(f"## SCHEDULE CONFIRMATION (seed {CONFIRM_SEED}) — canary "
               f"{fmt(can['tripod'])}/{fmt(can['completion'])}/{fmt(can['tracking'])} vs primary bake "
               f"{fmt(base_can['tripod'])}/{fmt(base_can['completion'])} — "
               f"{'CONFIRMED' if ok else 'DIVERGED — placements need review'} | obstacle "
               f"{fmt(obst['completion'])}", "confirmation")
        st["phase"] = "done"
        save_state(st)
    placements = ", ".join(f"{n}@{r * 5}k" for n, r in sorted(st["accepted"].items(), key=lambda kv: kv[1]))
    report(f"## GRADUATION CANDIDATE — schedule: [{placements}] | failed: "
           f"[{', '.join(st['failed']) or 'none'}] | head {st['bake_head']} — run graduation battery",
           "graduation")
    changelog(f"> NOTIFY: round-search complete — schedule [{placements}], failed [{', '.join(st['failed']) or 'none'}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
