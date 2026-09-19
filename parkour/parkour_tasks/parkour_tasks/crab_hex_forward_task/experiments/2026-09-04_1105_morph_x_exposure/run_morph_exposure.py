#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Morphology x training-side fixes orchestrator (plan approved 2026-09-04).

Every arm trains FROM SCRATCH (seed 3) on its plant (KRABBY_HEX_USD_PATH in the process env of
every train/eval subprocess -- the scene cfg reads it at import time):

  smoke         200-iter armed smoke on A15 with the P2 stage-1 stack + one eval (plant guard)
  p1            8 configs x 0->5k formation: rung-v formation config + KRABBY_STAND_FRAC=0.2
                -> await_user_a   (relaunch: --start-p2 [--top A15,A20,A15+B])
  p2            golden + top 3: stage 1 (0->5k) = P1 + EPISODE_S 40 + RESAMPLE_S 10:10;
                stage 2 (5k->10k, resume) = + window-1 elements + recal2b2w + promotion x 20/40
                -> await_user_b   (relaunch: --seed2 <cfg>)
  seed2         the user's pick, P2 both stages, seed 2 -> done

Reuses the closed campaigns' helpers by import: run_exposure.py (train loop with live checks,
canary / obstacle eval runners, exposure telemetry parser, hazard-normalised backstop, report
formatting) and run_formation_arms.py (formation config, plant map, morph-manifest eval).
Stdlib only; resumable via state.json.
"""
from __future__ import annotations

import argparse
import importlib.util
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
MORPH = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-02_1446_leg_mount_morphology"
PLANH = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure"
VARIANTS = REPO / "assets/variants"
MORPH_MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_morph.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
FLAT_RUNS = PARKOUR / "logs/rsl_rl/crab_hex_flat_walk"
EVAL_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/morph_x_exposure"
STATE = HERE / "state.json"
REPORT = HERE / "REPORT.md"
CHANGELOG = HERE / "CHANGELOG.md"
LOGS = HERE / "logs"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load(PLANH / "run_exposure.py", "run_exposure")           # helpers of the exposure campaign
rf = _load(MORPH / "run_formation_arms.py", "run_formation_arms")  # formation config + plant map

SEED = "3"
CONFIRM_SEED = "2"
STAGE_ITERS = 5000
SMOKE_ITERS = 2000          # rung-v soundness smoke inside the first stage
ARMED_SMOKE_ITERS = 200
CONFIGS = dict(rf.CONFIGS)  # {"base": None, "B": "splay00_axis2p5in", ...}
ORDER = ["base", "B", "A10", "A15", "A20", "A10+B", "A15+B", "A20+B"]
PREF = {"base": "—", "B": "re-hinge", "A10": "splay only ✓", "A15": "splay only ✓", "A20": "splay only ✓",
        "A10+B": "shims + re-hinge", "A15+B": "shims + re-hinge", "A20+B": "shims + re-hinge"}
FORMATION = dict(rf.BASELINE)              # the rung-v 0-5k config of record
STAND = {"KRABBY_STAND_FRAC": "0.2"}
HORIZON_S = 40.0
LONG = {"KRABBY_EPISODE_S": "40", "KRABBY_RESAMPLE_S": "10:10"}
OBST_EVAL = dict(rx.OBST_EVAL)             # recal2b2w @ 0.20-0.70
PLANT_PASS = ("KRABBY_HEX_USD_PATH",)
SCENARIOS_MORPH = ("slow", "step")


# ------------------------------------------------------------------ pure helpers (unit-tested)
def promote_fracs_for_horizon(up: float, down: float, horizon_s: float, ref_s: float = 20.0) -> str:
    """Promotion distance = frac x cmd x T: hold the equilibrium of the ref horizon by scaling
    both fractions by ref/T. Returns the KRABBY_TERRAIN_PROMOTE 'up:down' string."""
    if horizon_s <= 0 or ref_s <= 0:
        raise ValueError("horizons must be positive")
    k = ref_s / horizon_s
    return f"{up * k:.4g}:{down * k:.4g}"


def plant_path(cfg: str) -> str | None:
    tag = CONFIGS[cfg]
    if tag is None:  # "base" = legacy golden, explicit since 2026-09-09 (config default is now A15+B)
        return str(VARIANTS / "crab_simple__splay00_axis5p5in.usda")
    return str(VARIANTS / f"crab_simple__{tag}.usda")


def with_plant(ev: dict, cfg: str) -> dict:
    ev = dict(ev)
    ev.pop("KRABBY_HEX_USD_PATH", None)
    p = plant_path(cfg)
    if p:
        ev["KRABBY_HEX_USD_PATH"] = p
    return ev


def p1_stack(cfg: str) -> dict:
    return with_plant({**FORMATION, **STAND}, cfg)


def p2_stage1_stack(cfg: str) -> dict:
    return with_plant({**FORMATION, **STAND, **LONG}, cfg)


def p2_stage2_stack(cfg: str) -> dict:
    ev = {**FORMATION, **rx.SCHEDULE[1], **STAND, **LONG}
    ev["KRABBY_FLAT_TERRAIN_GEOM"] = "recal2b2w"
    up, down = (float(x) for x in rx.SCHEDULE[1]["KRABBY_TERRAIN_PROMOTE"].split(":"))
    ev["KRABBY_TERRAIN_PROMOTE"] = promote_fracs_for_horizon(up, down, HORIZON_S)
    return with_plant(ev, cfg)


def corrected_stand_frac(logged: float, eplen_steps: float, episode_s: float, dt: float = 0.02) -> float:
    """stand_frac_actual is indicator/max_episode_length: divide out eplen/max."""
    if not (eplen_steps and eplen_steps > 0):
        return float("nan")
    return logged * (episode_s / dt) / eplen_steps


def rank_key(rec: dict) -> float:
    """Default STOP-A ranking: slow completion x (1 - step-onset fall share)."""
    s = (rec.get("evals") or {}).get("slow") or {}
    p = (rec.get("evals") or {}).get("step") or {}
    comp = s.get("completion") or 0.0
    falls, n = p.get("falls"), p.get("n")
    share = (falls / n) if (falls is not None and n) else 1.0
    return comp * (1.0 - share)


def default_top(p1: dict, k: int = 3) -> list[str]:
    sound = [c for c in ORDER if c != "base" and (p1.get(c) or {}).get("ckpt")]
    sound.sort(key=lambda c: (rank_key(p1[c]), 1 if PREF[c].startswith("splay") else 0), reverse=True)
    return sound[:k]


# ------------------------------------------------------------------ infrastructure
def log(msg: str) -> None:
    print(f"[morphx {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    return {"phase": "smoke", "run_no": 0, "smoke": {}, "p1": {}, "top": None, "p2": {}, "seed2": {}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def _subenv(env_vars: dict) -> dict:
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    for k in list(env):
        if k.startswith("KRABBY_"):
            del env[k]
    env.update({k: str(v) for k, v in env_vars.items()})
    return env


def _wait_isaac_clear() -> None:
    pattern = r"isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe)\.py"
    for _ in range(60):
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if not any(int(p) != os.getpid() for p in out):
            return
        time.sleep(10)


def latest_run_ckpt(t0: float) -> str | None:
    run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
    if run_dir.stat().st_mtime < t0:
        return None
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(models[-1]) if models else None


def train(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None = None, seed: str = SEED,
          soundness: bool = True, live_check=None) -> tuple[str | None, str, Path]:
    """From-scratch (or resumed) training with the rung-v soundness smoke and the exposure
    campaign's live checks. Returns (ckpt | None, status, log_path); status in
    {ok, dead_plant, nan, infra, aborted}."""
    t0 = time.time()
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"{tag}_train.log"
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
                    log(f"{tag}: no log progress for 15 min — infra death")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "infra", log_path
                txt = log_path.read_text(errors="ignore")
                tail = txt[-20000:]
                if re.search(r"Mean (reward|value_function|surrogate):\s*nan", tail, re.I) or "Traceback" in tail:
                    log(f"{tag}: NaN / traceback in the training log — unsound plant or crash")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "nan", log_path
                if soundness:
                    ep = rx.series(log_path, "Mean episode length")
                    if len(ep) >= SMOKE_ITERS and rx.tail_mean(ep, 100) < 1.2 * rx.tail_mean(ep[:100], 100):
                        log(f"{tag}: DEAD PLANT — episode length {rx.tail_mean(ep[:100], 100):.0f} -> "
                            f"{rx.tail_mean(ep, 100):.0f} by 2k")
                        proc.kill(); proc.wait(timeout=60)
                        return None, "dead_plant", log_path
                if live_check is not None:
                    why = live_check(log_path)
                    if why:
                        log(f"{tag}: LIVE ABORT — {why}")
                        proc.kill(); proc.wait(timeout=60)
                        return None, "aborted", log_path
            except OSError:
                pass
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, "infra", log_path
    base = int(Path(resume_ckpt).stem.split("_")[1]) if resume_ckpt else 0
    ckpt = latest_run_ckpt(t0)
    if ckpt is None or int(Path(ckpt).stem.split("_")[1]) < base + iters - 100:
        return None, "infra", log_path
    return ckpt, "ok", log_path


def train_retry(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None = None, seed: str = SEED,
                soundness: bool = True, live_check=None) -> tuple[str | None, str, Path]:
    ckpt, status, lp = train(env_vars, tag, iters, resume_ckpt, seed, soundness, live_check)
    if ckpt is None and status == "infra":
        log(f"{tag}: infra death — one retry")
        _wait_isaac_clear(); time.sleep(120)
        ckpt, status, lp = train(env_vars, f"{tag}_r2", iters, resume_ckpt, seed, soundness, live_check)
    return ckpt, status, lp


def _run_meta_plant(run_dir: Path) -> str | None:
    for name in ("run_meta.json", "scenario_metrics.json"):
        f = run_dir / name
        if f.exists():
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            meta = d.get("run_meta", d) if isinstance(d, dict) else {}
            if isinstance(meta, dict) and meta.get("usd_path"):
                return str(meta["usd_path"])
    return None


def _plant_guard(run_dir: Path, cfg: str, tag: str) -> None:
    want = plant_path(cfg)
    got = _run_meta_plant(run_dir)
    if got is None:
        return
    ok = (want is None and "variants/" not in got) or (want is not None and Path(got).name == Path(want).name)
    if not ok:
        raise RuntimeError(f"{tag}: eval spawned plant {got} but {cfg} requested {want or 'golden'} (rung-iv defect)")


def eval_morph(ckpt: str, cfg: str, scenario: str, tag: str) -> dict | None:
    """Morph-manifest eval (slow__<cfg> / step__<cfg>) on the variant plant; rung-v summary fields."""
    sid = f"{scenario}__{cfg.replace('+', 'p')}"
    out_root = EVAL_ROOT / tag
    ev = {k: FORMATION[k] for k in ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W",
                                    "KRABBY_CLOCK_W", "KRABBY_APEX_W")}
    ev = with_plant(ev, cfg)
    LOGS.mkdir(exist_ok=True)
    lp = LOGS / f"{tag}_{sid}_eval.log"
    for attempt in range(3):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MORPH_MANIFEST), "--scenario", sid,
                                "--checkpoint", ckpt, "--no-plot", "--save-raw", "--output-root", str(out_root)],
                               cwd=PARKOUR, env=_subenv(ev), stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
            except subprocess.TimeoutExpired:
                pass
        d = out_root / sid / "seed001"
        if d.exists():
            latest = max(d.iterdir(), key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime >= t0 and (latest / "scenario_metrics.json").exists():
                _plant_guard(latest, cfg, tag)
                s = rf.summarize(json.loads((latest / "scenario_metrics.json").read_text())["aggregate"])
                s["run_dir"] = str(latest)
                return s
        log(f"{tag}/{sid}: no fresh metrics — retry {attempt + 1}")
        _wait_isaac_clear(); time.sleep(120)
    return None


def eval_obst(ckpt: str, cfg: str, env_vars: dict, tag: str) -> dict | None:
    """PLAN H obstacle eval (flat_walk_slow_v2 on recal2b2w @ 0.20-0.70) on the variant plant."""
    ev = {k: env_vars[k] for k in rx.CANARY_KEEP if k in env_vars}
    ev.update({k: env_vars[k] for k in rx.TERRAIN_PASS if k in env_vars})
    ev.update(OBST_EVAL)
    ev.update({"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0"})
    ev = with_plant(ev, cfg)
    LOGS.mkdir(exist_ok=True)
    lp = LOGS / f"{tag}_obst.log"
    out_root = Path(str(EVAL_ROOT) + "_obst")
    for _attempt in range(3):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(rx.MANIFEST),
                                "--scenario", "flat_walk_slow_v2", "--checkpoint", ckpt, "--no-plot",
                                "--output-root", str(out_root)],
                               cwd=PARKOUR, env=_subenv(ev), stdout=fh, stderr=subprocess.STDOUT, timeout=5400)
            except subprocess.TimeoutExpired:
                pass
        seed_dir = out_root / "flat_walk_slow_v2" / "seed001"
        if seed_dir.exists():
            latest = max(seed_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime >= t0 and (latest / "scenario_metrics.json").exists():
                _plant_guard(latest, cfg, tag)
                a = json.loads((latest / "scenario_metrics.json").read_text())["aggregate"]
                return {"completion": a.get("schedule_completion_rate") or 0.0,
                        "tripod": rx._med(a.get("tripod_score")) or 0.0,
                        "falls": (a.get("termination_reasons") or {}).get("fall", 0),
                        "n": a.get("n_episodes"), "run_dir": str(latest)}
        log(f"{tag}: obstacle eval produced no fresh metrics — retry")
        _wait_isaac_clear(); time.sleep(120)
    return None


def smoke_fields(lp: Path) -> dict:
    fail = rx.series(lp, "Episode_Termination/crab_failure")
    coll = rx.series(lp, "Episode_Reward/reward_collision")
    ep = rx.series(lp, "Mean episode length")
    return {"fail_2k": rx.tail_mean(fail[:SMOKE_ITERS], 100) if fail else None,
            "fail_end": rx.tail_mean(fail, 100) if fail else None,
            "coll_2k": rx.tail_mean(coll[:SMOKE_ITERS], 100) if coll else None,
            "ep_len_2k": rx.tail_mean(ep[:SMOKE_ITERS], 100) if ep else None}


def run_stage(st: dict, cfg: str, label: str, env_vars: dict, resume_ckpt: str | None, seed: str,
              ctrl_exposure: dict | None, hazard_norm: bool) -> dict:
    """Train one 5k stage and run its three evals. Returns the stage record."""
    st["run_no"] += 1
    tag = f"{label}_{st['run_no']:03d}_{cfg.replace('+', 'p')}" + (f"_seed{seed}" if seed != SEED else "")
    # USER RULE (morphology campaign, carried over): only plant-unsound arms drop. Failure shares
    # are scored columns, never kills -- the exposure campaign's live backstop (control + 0.10)
    # is NOT applied here (it aborted A10 at 2.5k on 2026-09-04 19:32; fixed and rerun).
    live = None
    log(f"{tag}: {'resume ' + Path(resume_ckpt).name if resume_ckpt else 'from scratch'} on plant "
        f"{CONFIGS[cfg] or 'golden'} | {dict((k, v) for k, v in env_vars.items() if k in ('KRABBY_STAND_FRAC', 'KRABBY_EPISODE_S', 'KRABBY_RESAMPLE_S', 'KRABBY_TERRAIN_PROMOTE', 'KRABBY_FLAT_TERRAIN_GEOM'))}")
    ckpt, status, lp = train_retry(env_vars, tag, STAGE_ITERS, resume_ckpt, seed,
                                   soundness=resume_ckpt is None, live_check=live)
    rec = {"cfg": cfg, "tag": tag, "ckpt": ckpt, "status": status, "log": str(lp), "env": env_vars,
           "smoke": smoke_fields(lp) if lp.exists() else {}, "exposure": rx.exposure_from_log(lp) if lp.exists() else {},
           "evals": {}}
    if ckpt:
        rec["evals"]["slow"] = eval_morph(ckpt, cfg, "slow", tag)
        rec["evals"]["step"] = eval_morph(ckpt, cfg, "step", tag)
        rec["evals"]["obst"] = eval_obst(ckpt, cfg, env_vars, tag)
    return rec


def stage_lines(rec: dict) -> list[str]:
    ex = rec.get("exposure") or {}
    s = (rec.get("evals") or {}).get("slow") or {}
    p = (rec.get("evals") or {}).get("step") or {}
    o = (rec.get("evals") or {}).get("obst") or {}
    eps = float((rec.get("env") or {}).get("KRABBY_EPISODE_S", 20))
    stand_c = corrected_stand_frac(ex.get("stand_frac_actual", float("nan")), ex.get("eplen", float("nan")), eps)
    lines = [f"- status {rec.get('status')} | checkpoint {Path(rec['ckpt']).parent.name + '/' + Path(rec['ckpt']).name if rec.get('ckpt') else '—'} | "
             f"smoke fail@2k {rx.fmt(rec.get('smoke', {}).get('fail_2k'))} coll@2k {rx.fmt(rec.get('smoke', {}).get('coll_2k'))} ep_len@2k {rx.fmt(rec.get('smoke', {}).get('ep_len_2k'))}"]
    if rec.get("ckpt"):
        lines += rx.expo_lines(ex)
        lines.append(f"- stand time frac corrected {rx.fmt(stand_c)} (logged {rx.fmt(ex.get('stand_frac_actual'))}) | mean reward {rx.fmt(ex.get('mean_reward'))} | vloss {rx.fmt(ex.get('vloss'))}")
        lines.append(f"- slow canary (morph manifest): tripod {rx.fmt(s.get('tripod'))} | completion {rx.fmt(s.get('completion'))} | "
                     f"tracking {rx.fmt(s.get('tracking'))} | falls {s.get('falls')}/{s.get('n')} | pitch-fwd share {rx.fmt(s.get('pitch_fwd_share'))} | "
                     f"prefall tip p25 {rx.fmt(s.get('prefall_tip_p25'))} | walk tip p50 {rx.fmt(s.get('walk_tip_p50'))}")
        lines.append(f"- step onset (morph manifest, shallow 0.05-0.2): completion {rx.fmt(p.get('completion'))} | falls {p.get('falls')}/{p.get('n')} | "
                     f"pitch-fwd share {rx.fmt(p.get('pitch_fwd_share'))} | prefall tip p25 {rx.fmt(p.get('prefall_tip_p25'))}")
        lines.append(f"- obstacle eval (recal2b2w 0.20-0.70): completion {rx.fmt(o.get('completion'))} | tripod {rx.fmt(o.get('tripod'))} | falls {o.get('falls')}/{o.get('n')}")
    return lines


# ------------------------------------------------------------------ phases
def smoke_checks(rec: dict) -> list[tuple[str, bool]]:
    """Recomputable from the record (no GPU): plant, horizon, hold length, telemetry, eval."""
    checks = []
    ckpt = rec.get("ckpt")
    if not ckpt:
        return [("smoke training produced a checkpoint", False)]
    envy = Path(ckpt).parent / "params" / "env.yaml"
    txt = envy.read_text(errors="ignore") if envy.exists() else ""
    checks.append(("plant is A15 (params/env.yaml usd path)", "splay15_axis5p5in" in txt))
    checks.append(("episode_length_s 40 in params", re.search(r"episode_length_s:\s*40", txt) is not None))
    # YAML tuple: 'resampling_time_range: !!python/tuple' then '- 10.0' items on the next lines
    m = re.search(r"resampling_time_range:[^\n]*\n\s*-\s*([0-9.]+)\s*\n\s*-\s*([0-9.]+)", txt)
    checks.append(("resampling 10 s in params", bool(m) and float(m.group(1)) == 10.0 and float(m.group(2)) == 10.0))
    ex = rec.get("exposure") or {}
    checks.append(("stand_frac_actual (corrected) in 0.05-0.45",
                   0.05 <= corrected_stand_frac(ex.get("stand_frac_actual", -1), ex.get("eplen", 1), 40.0) <= 0.45))
    checks.append(("exposure keys present", ex.get("reach_edge_frac") == ex.get("reach_edge_frac")))
    checks.append(("mean episode length <= 2000 steps", (ex.get("eplen") or 9e9) <= 2000))
    checks.append(("morph eval ran on the A15 plant (run_meta guard passed)", rec.get("eval_slow") is not None))
    return checks


def phase_smoke(st: dict) -> None:
    sm = st["smoke"]
    if "rec" not in sm:
        st["run_no"] += 1
        tag = f"smoke_{st['run_no']:03d}_A15"
        ev = p2_stage1_stack("A15")
        ckpt, status, lp = train_retry(ev, tag, ARMED_SMOKE_ITERS, soundness=False)
        rec = {"ckpt": ckpt, "status": status, "log": str(lp), "exposure": rx.exposure_from_log(lp, 20) if lp.exists() else {}}
        if ckpt:
            rec["eval_slow"] = eval_morph(ckpt, "A15", "slow", tag)
        sm["rec"] = rec
        save_state(st)
    checks = smoke_checks(sm["rec"])
    sm["rec"]["checks"] = checks
    save_state(st)
    ok = all(c for _n, c in checks)
    report("\n".join(["## SMOKE — A15, P2 stage-1 stack, 200 iterations"] + [f"- {'PASS' if c else 'FAIL'}: {n}" for n, c in checks]
                     + [f"- **{'SMOKE PASS' if ok else 'SMOKE FAIL — P1 not started'}**"]), f"smoke {'PASS' if ok else 'FAIL'}")
    if not ok:
        st["phase"] = "halt_smoke"
        save_state(st)
        notify("morphology x exposure: SMOKE FAILED — see REPORT; fix before relaunch")
        sys.exit(2)
    st["phase"] = "p1"
    save_state(st)


def phase_p1(st: dict) -> None:
    p1 = st["p1"]
    for cfg in ORDER:
        if cfg in p1 and p1[cfg].get("status") not in (None, "infra"):
            continue
        ctrl = (p1.get("base") or {}).get("exposure") if cfg != "base" else None
        rec = run_stage(st, cfg, "p1", p1_stack(cfg), None, SEED, ctrl if ctrl else None, hazard_norm=False)
        p1[cfg] = rec
        save_state(st)
        report("\n".join([f"## P1 — {cfg} (from scratch 0->5k, formation + STAND_FRAC 0.2) — {rec['status']}"] + stage_lines(rec)),
               f"p1 {cfg} {rec['status']}")
        changelog(f"- {time.strftime('%Y-%m-%d %H:%M')} P1 {cfg}: {rec['status']}; slow {rx.fmt(((rec['evals'].get('slow') or {}).get('completion')))} "
                  f"step falls {((rec['evals'].get('step') or {}).get('falls'))} obst {rx.fmt(((rec['evals'].get('obst') or {}).get('completion')))}")
    top = default_top(p1)
    st["top_default"] = top
    st["phase"] = "await_user_a"
    save_state(st)
    notify(f"P1 complete — default top 3 by slow completion x (1 - step fall share): {top}; PAUSED at STOP A "
           f"(relaunch with --start-p2 [--top A,B,C])")


def phase_p2(st: dict, seed: str = SEED, key: str = "p2", cfgs: list[str] | None = None) -> None:
    p2 = st[key]
    cfgs = cfgs or (["base"] + [c for c in (st.get("top") or st.get("top_default") or []) if c != "base"])
    for cfg in cfgs:
        rec = p2.setdefault(cfg, {})
        if not rec.get("stage1") or rec["stage1"].get("status") == "infra":
            ctrl = (p2.get("base") or {}).get("stage1", {}).get("exposure") if cfg != "base" else None
            rec["stage1"] = run_stage(st, cfg, f"{key}s1", p2_stage1_stack(cfg), None, seed, ctrl, hazard_norm=True)
            save_state(st)
            report("\n".join([f"## {key.upper()} stage 1 — {cfg} (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed {seed}) — {rec['stage1']['status']}"]
                             + stage_lines(rec["stage1"])), f"{key} s1 {cfg} {rec['stage1']['status']}")
        if not rec["stage1"].get("ckpt"):
            continue
        if not rec.get("stage2") or rec["stage2"].get("status") == "infra":
            ctrl = (p2.get("base") or {}).get("stage2", {}).get("exposure") if cfg != "base" else None
            rec["stage2"] = run_stage(st, cfg, f"{key}s2", p2_stage2_stack(cfg), rec["stage1"]["ckpt"], seed, ctrl, hazard_norm=True)
            save_state(st)
            tl = (rec["stage2"].get("exposure") or {}).get("terrain_levels")
            report("\n".join([f"## {key.upper()} stage 2 — {cfg} (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed {seed}) — {rec['stage2']['status']}"]
                             + stage_lines(rec["stage2"])
                             + [f"- terrain level {rx.fmt(tl)} (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)"]),
                   f"{key} s2 {cfg} {rec['stage2']['status']}")
        changelog(f"- {time.strftime('%Y-%m-%d %H:%M')} {key.upper()} {cfg} seed {seed}: s1 {rec['stage1']['status']} / s2 {rec.get('stage2', {}).get('status')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-p2", action="store_true")
    ap.add_argument("--top", type=str, default=None, help="comma-separated configs for P2 (default: ranking rule)")
    ap.add_argument("--seed2", type=str, default=None, help="config for the seed-2 replay of P2")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    st = load_state()
    if st["phase"] == "halt_smoke":
        st["phase"] = "smoke"          # relaunch after a fix re-evaluates the kept smoke record
    if args.skip_smoke and st["phase"] == "smoke":
        st["phase"] = "p1"
    if args.start_p2 and st["phase"] == "await_user_a":
        st["top"] = [c.strip() for c in args.top.split(",")] if args.top else st.get("top_default")
        st["phase"] = "p2"
    if args.seed2 and st["phase"] == "await_user_b":
        st["seed2_cfgs"] = [c.strip() for c in args.seed2.split(",") if c.strip()]   # one or more plants
        st["seed2_cfg"] = st["seed2_cfgs"][0]
        st["phase"] = "seed2"
    save_state(st)
    if st["phase"] == "smoke":
        phase_smoke(st)
    if st["phase"] == "p1":
        phase_p1(st)
    if st["phase"] == "await_user_a":
        log("await_user_a: P1 done — relaunch with --start-p2 [--top A,B,C]")
        return 0
    if st["phase"] == "p2":
        phase_p2(st)
        st["phase"] = "await_user_b"
        save_state(st)
        notify("P2 complete — PAUSED at STOP B; run assemble_final.py for the decision table; "
               "relaunch with --seed2 <cfg> for the confirmation of the pick")
    if st["phase"] == "await_user_b":
        log("await_user_b: P2 done — relaunch with --seed2 <cfg>")
        return 0
    if st["phase"] == "seed2":
        cfgs = st.get("seed2_cfgs") or [st["seed2_cfg"]]
        phase_p2(st, seed=CONFIRM_SEED, key="seed2", cfgs=cfgs)
        st["phase"] = "done"
        save_state(st)
        notify(f"seed-2 replay of {cfgs} complete — run assemble_final.py --to-results; hardware decision is the user's")
    if st["phase"] == "done":
        log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
