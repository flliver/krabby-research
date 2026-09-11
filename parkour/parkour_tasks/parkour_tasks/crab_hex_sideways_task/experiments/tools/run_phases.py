#!/usr/bin/env python3
"""Three-phase crab-hex training pipeline driver -- sideways-walk sibling (2026-09-11).

Runs the ``KRABBY_SIDE_PHASE`` presets of ``crab_hex_side_phases.py`` in order -- phase 1 pure
student (``1a``, ``Isaac-Crab-Hex-Side-Walk-v0``), phase 2 teacher-student (``2a/2b/2c``,
``Isaac-Crab-Hex-Side-Teacher-v0`` in the matching mode) and phase 3 student distillation
(``3a/3b``, ``Isaac-Crab-Hex-Side-Student-v0``) -- each resuming the previous phase's head, with
the campaign evals after every phase (flat canary + step onset on
``experiments/eval/scenarios_side.yaml`` and the recal2b2w obstacle eval), a REPORT / CHANGELOG /
state.json record, and stop-and-wait pauses.

The training subprocess environment carries ONLY ``KRABBY_SIDE_PHASE`` and ``KRABBY_PLANT``: every
knob comes from the preset (``activate_phase`` at config import). This driver never exports
``KRABBY_PHASE`` -- doing so would raise at config import (one task per process; see
``crab_hex_side_phases.py::activate_phase``). Evals never export the preset; they pass the same
explicit keys the forward campaigns used (plus ``KRABBY_STUDENT_MDP=1`` and the student task for
phase-3 heads).

    run_phases.py --campaign-dir <dir> --plant A15+B --phases 1a,2a,2b,2c,3a,3b [--seed 3]
                  [--from-checkpoint <pt>] [--iterations N] [--no-eval] [--pause-after 2c,3a,3b]

Resumable: rerunning skips phases already recorded ``ok`` for the seed (``--redo`` reruns them);
the first requested phase resumes from ``--from-checkpoint`` when given, else from the recorded
head of its ``resume_from`` phase. Launch through ``launch_phases.sh`` (systemd scope) and watch
with ``heartbeat_phases.sh``.
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
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_sideways_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
PKG = HERE.parents[1]
FWD_PKG = PKG.parent / "crab_hex_forward_task"
EVAL = FWD_PKG / "scripts/eval_crab_hex_gait.py"  # shared harness: one harness serves both tasks
SIDE_MANIFEST = PKG / "experiments/eval/scenarios_side.yaml"
PHASES_PY = PKG / "config/crab_hex/crab_hex_side_phases.py"
LOG_ROOT = PARKOUR / "logs/rsl_rl"
EVAL_ROOT = LOG_ROOT / "gait_eval/side_phases"
STUDENT_TASK = "Isaac-Crab-Hex-Side-Student-v0"
NUM_ENVS = {"rl": "256", "distill": "192"}      # distill: the student cfg default (depth ray-caster)
CANARY_KEYS = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W", "KRABBY_APEX_W")
SMOKE_ITERS = 2000       # from-scratch soundness window (rung-v)
STALL_S = 900
DEFAULT_PAUSE = "2c,3a,3b"
GATE_1A_TRIPOD = 0.40     # era-A pre-mirror-loss tripod level (measured on walking holds) -- a floor, not a bar
GATE_1A_COMPLETION = 0.70
WALKING_CMD_MIN = 0.2     # documentation: the harness's lin_vel_clip -- a hold commanded at |vx| <= this is a stand
                          # hold and gets tracking ratio n == 0, which is what walking_tripod() reads


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ph = _load(PHASES_PY, "crab_hex_side_phases")

# ------------------------------------------------------------------ inlined helpers
# The sideways driver has no campaign directories of its own to load these from by path (the
# forward driver loaded them from RX_PY / RF_PY, two of the forward package's own campaign
# scripts). Verbatim copies, 2026-09-11, from:
#   parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/
#     2026-09-03_1156_obstacle_exposure/run_exposure.py            (rx: everything but summarize)
#     2026-09-02_1446_leg_mount_morphology/run_formation_arms.py   (rf: summarize)
CANARY_KEEP = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W",
               "KRABBY_APEX_W", "KRABBY_AIRTIME_W", "KRABBY_STRIDE_W",
               "KRABBY_HEADING", "KRABBY_HEADING_STIFFNESS", "KRABBY_ACTION_SCALE")
TERRAIN_PASS = ("KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_FLAT_TERRAIN_DIFF",
                "KRABBY_CORRIDOR_HALF_WIDTH", "KRABBY_STONE_WIDTH")
OBST_EVAL = {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2w", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70"}
N_OBST = 6
EXPO_BASE = ["reach_edge_frac", "reach_obst_frac", "field_frac_mean", "field_steps_mean",
             "goals_passed_mean"] + [f"obst_coverage_{k}" for k in range(1, N_OBST + 1)]
EXPO_KEYS = EXPO_BASE + [f"{k}_rsi" for k in EXPO_BASE] + [
    "crab_failure_flat", "crab_failure_obst", "crab_failure_obst_rsi", "crab_failure_obst_spread",
    "crab_failure_hazard_flat", "crab_failure_hazard_obst", "ep_steps_flat", "ep_steps_obst", "ep_steps_obst_spread",
    "spread_frac_actual", "rsi_frac_actual", "terrain_levels", "how_far_from_start_point",
    "current_goal_idx"]


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


def _med(x):
    if isinstance(x, dict):
        return x.get("median") if x.get("median") is not None else x.get("mean")
    return x


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


def walking_tripod(a: dict) -> tuple[float | None, dict[str, float | None]]:
    """Per-hold tripod medians and their minimum over the WALKING holds of the schedule.

    The harness's episode ``tripod_score`` is a steady-step-weighted mean over EVERY hold, the
    cmd-0 stand hold included and with no command gate (gait_eval/report.py), so a head that truly
    stands at zero command scores ~0 on a third of the episode while a head that keeps stepping
    through its stop does not (the forward 1a reference: stand-hold tripod 0.453 at 0.066 m/s;
    the first sideways 1a head: 0.001 at 0.006 m/s, walking holds 0.537 / 0.533, episode 0.355).
    The gate therefore reads the holds the harness itself scores for tracking
    (``tracking_by_hold[h].ratio.n > 0``, i.e. |cmd| > lin_vel_clip) and takes the minimum, so
    every walking speed must clear the floor. 2026-09-11 finding; see the CHARTER's gate section.
    """
    by_hold = {h: (v or {}).get("median") for h, v in (a.get("tripod_score_by_hold") or {}).items()}
    track = a.get("tracking_by_hold") or {}
    # Walking holds are enumerated from the TRACKING block (the harness writes one entry per hold of
    # the schedule; ratio.n > 0 <=> |cmd| > lin_vel_clip = WALKING_CMD_MIN), NOT from the tripod
    # block, which only carries the holds where at least one episode produced a scorable gait
    # window: a walking hold with no tripod median is UNMEASURED (the head never cycled its feet
    # at that command), so the number is None and the gate halts instead of passing on the rest.
    walking_holds = [h for h, t in track.items() if ((((t or {}).get("ratio") or {}).get("n")) or 0) > 0]
    for h in walking_holds:
        by_hold.setdefault(h, None)          # named in the by-hold string as n/a
    meds = [by_hold[h] for h in walking_holds]
    if not meds or any(m is None for m in meds):
        return None, by_hold
    return min(meds), by_hold


def summarize(a: dict | None) -> dict:
    if not a:
        return {}
    fc = a.get("fall_classes", {})
    poly = a.get("support_polygon", {})
    tri_walk, tri_by_hold = walking_tripod(a)
    return {"tripod": (a.get("tripod_score") or {}).get("median"),
            "tripod_walking": tri_walk,       # the gated number (min over walking holds)
            "tripod_by_hold": tri_by_hold,
            "completion": a.get("schedule_completion_rate"),
            "tracking": (a.get("tracking_ratio") or {}).get("median"),
            "falls": a.get("termination_reasons", {}).get("fall", 0),
            "n": a.get("n_episodes"),
            # share of FALLS that are pitch-forward (in the travel frame: metrics.py/report.py
            # rotate fall_direction by travel_yaw, so this reads the same way as the forward number)
            "pitch_fwd_share": (fc.get("pitch_fwd", 0) / max(1, sum(v for k, v in fc.items() if k != "none")))
            if any(k != "none" for k in fc) else None,
            "pitch_max_abs_p50": ((a.get("orientation") or {}).get("pitch_max_abs") or {}).get("median"),
            "prefall_tip_p25": (poly.get("prefall_tip_angle_fwd_deg") or {}).get("p25"),
            "walk_tip_p50": (poly.get("walking_tip_angle_fwd_deg") or {}).get("median")}


# ------------------------------------------------------------------ campaign record
class Campaign:
    def __init__(self, cdir: Path, plant: str, seed: str):
        self.dir = cdir
        self.plant = plant
        self.seed = seed
        self.key = f"seed{seed}"
        self.state_path = cdir / "state.json"
        self.report = cdir / "REPORT.md"
        self.changelog = cdir / "CHANGELOG.md"
        self.logs = cdir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.st = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"campaign": self.dir.name, "plant": self.plant, "phase": "run", "run_no": 0, "keys": {}, "current": {}}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.st, indent=2))
        heads = {k: {p: r.get("ckpt") for p, r in recs.items() if r.get("ckpt")} for k, recs in self.st["keys"].items()}
        (self.dir / "heads.json").write_text(json.dumps(heads, indent=2))

    @property
    def recs(self) -> dict:
        return self.st["keys"].setdefault(self.key, {})

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with (self.dir / "orchestrator.log").open("a") as fh:
            fh.write(line + "\n")

    def changelog_row(self, row: str) -> None:
        with self.changelog.open("a") as fh:
            fh.write(f"\n### {datetime.now():%Y-%m-%d %H:%M} — {row}\n")

    def notify(self, msg: str) -> None:
        with self.changelog.open("a") as fh:
            fh.write(f"> NOTIFY: {msg}\n")
        with (self.dir / "notify.log").open("a") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M} {msg}\n")
        self.log(f"NOTIFY: {msg}")

    def report_block(self, block: str) -> None:
        with self.report.open("a") as fh:
            fh.write(block.rstrip("\n") + "\n\n")

    def ensure_header(self, phases: list[str], args) -> None:
        if self.report.exists():
            return
        rows = ["| phase | task / mode | iterations | resume from | notes |", "|---|---|---|---|---|"]
        for name in phases:
            s = ph.PHASES[name]
            mode = f"`{s.task}`" + (f" + mode `{s.teacher_mode}`" if s.teacher_mode else "")
            rows.append(f"| {name} | {mode} | {args.iterations or s.iterations} | {s.resume_from or 'scratch'} | {s.notes} |")
        head = (f"# Sideways-formation campaign `{self.dir.name}`\n\n"
                f"Plant **{self.plant}** (`{ph.plant_usd_path(self.plant) or 'assets/crab.usda (main)'}`), seed {self.seed}, "
                f"phases {', '.join(phases)}. Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/tools/run_phases.py`; presets: "
                f"`crab_hex_side_phases.py` (`KRABBY_SIDE_PHASE` / `KRABBY_PLANT`).\n\n" + "\n".join(rows))
        self.report.write_text(head + "\n\n")
        self.changelog.write_text(f"# CHANGELOG — {self.dir.name}\n")
        self.changelog_row(f"campaign start: plant {self.plant}, seed {self.seed}, phases {', '.join(phases)}"
                           + (f", iterations override {args.iterations}" if args.iterations else ""))


# ------------------------------------------------------------------ subprocess plumbing
def _subenv(env_vars: dict) -> dict:
    assert "KRABBY_PHASE" not in env_vars, "sideways driver must never export KRABBY_PHASE"
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    for k in list(env):
        if k.startswith("KRABBY_"):
            del env[k]
    env.update({k: str(v) for k, v in env_vars.items()})
    return env


def _wait_isaac_clear() -> None:
    pattern = r"isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe|crab_hex_phase_cfg_dump)\.py"
    for _ in range(60):
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if not any(int(p) != os.getpid() for p in out):
            return
        time.sleep(10)


def _ckpt_iter(ckpt: str | None) -> int:
    return int(Path(ckpt).stem.split("_")[1]) if ckpt else 0


def latest_run_ckpt(exp_dir: Path, t0: float) -> str | None:
    runs = [p for p in exp_dir.iterdir() if p.is_dir()] if exp_dir.exists() else []
    if not runs:
        return None
    run_dir = max(runs, key=lambda p: p.stat().st_mtime)
    if run_dir.stat().st_mtime < t0:
        return None
    models = sorted(run_dir.glob("model_*.pt"), key=_ckpt_iter)
    return str(models[-1]) if models else None


def with_plant(ev: dict, plant: str) -> dict:
    ev = dict(ev)
    usd = ph.plant_usd_path(plant)
    if usd:
        ev["KRABBY_HEX_USD_PATH"] = usd
    return ev


# ------------------------------------------------------------------ training
def train(c: Campaign, spec, tag: str, iters: int, resume_ckpt: str | None, soundness: bool) -> tuple[str | None, str, Path]:
    """One phase of training. Returns (ckpt | None, status, log); status in {ok, dead_plant, nan, infra}."""
    t0 = time.time()
    log_path = c.logs / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", spec.task, "--headless", "--num_envs", NUM_ENVS[spec.kind],
           "--seed", c.seed, "--max_iterations", str(iters)]
    if resume_ckpt:
        cmd += ["--resume", "--checkpoint", resume_ckpt]
    env = {"KRABBY_SIDE_PHASE": spec.name, "KRABBY_PLANT": c.plant}
    c.st["current"] = {"tag": tag, "phase": spec.name, "iters": iters, "kind": spec.kind, "log": str(log_path),
                       "resume": resume_ckpt, "target_final": _ckpt_iter(resume_ckpt) + iters - 1,
                       "started": datetime.now().isoformat(timespec="seconds")}
    c.save()
    c.log(f"{tag}: {spec.task}" + (f" mode {spec.teacher_mode}" if spec.teacher_mode else "") +
          f" | {iters} its | {'resume ' + Path(resume_ckpt).name if resume_ckpt else 'from scratch'} | env {env}")
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=_subenv(env), stdout=fh, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            time.sleep(120)
            try:
                if time.time() - log_path.stat().st_mtime > STALL_S:
                    c.log(f"{tag}: no log progress for {STALL_S // 60} min — infra death")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "infra", log_path
                tail = log_path.read_text(errors="ignore")[-20000:]
                if re.search(r"Mean (reward|value_function|surrogate):\s*nan", tail, re.I) or "Traceback" in tail:
                    c.log(f"{tag}: NaN / traceback in the training log")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "nan", log_path
                if spec.kind == "distill" and not getattr(train, "_warned", False):
                    ep = series(log_path, "Mean episode length")
                    if len(ep) >= 300 and tail_mean(ep, 50) < 400:
                        train._warned = True
                        c.log(f"{tag}: WARNING student rollouts still falling at {len(ep)} iterations "
                              f"(mean episode length {tail_mean(ep, 50):.0f} steps) — not killed (user rule), check the MDP")
                        c.notify(f"{tag}: student rollouts still falling at {len(ep)} its (ep len {tail_mean(ep, 50):.0f}) — inspect")
                if soundness:
                    ep = series(log_path, "Mean episode length")
                    if len(ep) >= SMOKE_ITERS and tail_mean(ep, 100) < 1.2 * tail_mean(ep[:100], 100):
                        c.log(f"{tag}: DEAD PLANT — episode length {tail_mean(ep[:100], 100):.0f} -> "
                              f"{tail_mean(ep, 100):.0f} by {SMOKE_ITERS}")
                        proc.kill(); proc.wait(timeout=60)
                        return None, "dead_plant", log_path
            except OSError:
                pass
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, "infra", log_path
    base = _ckpt_iter(resume_ckpt)
    ckpt = latest_run_ckpt(LOG_ROOT / spec.experiment, t0)
    if ckpt is None or _ckpt_iter(ckpt) < base + iters - 100:
        c.log(f"{tag}: no checkpoint at >= {base + iters - 100} under {spec.experiment} (got {ckpt})")
        return None, "infra", log_path
    return ckpt, "ok", log_path


def train_retry(c: Campaign, spec, tag: str, iters: int, resume_ckpt: str | None, soundness: bool):
    ckpt, status, lp = train(c, spec, tag, iters, resume_ckpt, soundness)
    if ckpt is None and status == "infra":
        c.log(f"{tag}: infra death — one retry")
        _wait_isaac_clear(); time.sleep(120)
        ckpt, status, lp = train(c, spec, f"{tag}_r2", iters, resume_ckpt, soundness)
    return ckpt, status, lp


# ------------------------------------------------------------------ evals
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


def _plant_guard(run_dir: Path, plant: str, tag: str) -> None:
    want = ph.plant_usd_path(plant)
    got = _run_meta_plant(run_dir)
    if got is None:
        return
    ok = (want is None and Path(got).name in ph.MAIN_ASSET_NAMES) or (want is not None and Path(got).name == Path(want).name)
    if not ok:
        raise RuntimeError(f"{tag}: eval spawned plant {got} but {plant} requested {want or 'the main asset'} (rung-iv defect)")


def _student_args(spec) -> list[str]:
    return ["--task", STUDENT_TASK] if spec.kind == "distill" else []


def _student_env(spec, ev: dict) -> dict:
    return dict(ev, KRABBY_STUDENT_MDP="1") if spec.kind == "distill" else ev


def _run_eval(c: Campaign, tag: str, sid: str, cmd_tail: list[str], env: dict, out_root: Path, plant: str, timeout: int):
    lp = c.logs / f"{tag}_{sid}_eval.log"
    for attempt in range(3):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run([PY, str(EVAL), "--headless", *cmd_tail, "--no-plot", "--output-root", str(out_root)],
                               cwd=PARKOUR, env=_subenv(env), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        d = out_root / sid / "seed001"
        if d.exists():
            latest = max(d.iterdir(), key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime >= t0 and (latest / "scenario_metrics.json").exists():
                _plant_guard(latest, plant, tag)
                return latest, json.loads((latest / "scenario_metrics.json").read_text())["aggregate"]
        c.log(f"{tag}/{sid}: no fresh metrics — retry {attempt + 1}")
        _wait_isaac_clear(); time.sleep(120)
    return None, None


def eval_morph(c: Campaign, spec, ckpt: str, scenario: str, tag: str) -> dict | None:
    """Side-manifest eval (slow__side / step__side): the campaign's flat canary / step onset."""
    sid = f"{scenario}__side"
    ev = _student_env(spec, with_plant({k: ph.FORMATION[k] for k in CANARY_KEYS if k in ph.FORMATION}, c.plant))
    latest, a = _run_eval(c, tag, sid, ["--manifest", str(SIDE_MANIFEST), "--scenario", sid, "--checkpoint", ckpt,
                                        "--save-raw", *_student_args(spec)], ev, EVAL_ROOT / c.dir.name / tag, c.plant, 3600)
    if a is None:
        return None
    s = summarize(a)
    s["run_dir"] = str(latest)
    return s


def eval_obst(c: Campaign, spec, ckpt: str, tag: str) -> dict | None:
    """PLAN H obstacle eval (flat_walk_slow_side on recal2b2w @ 0.20-0.70) on the plant."""
    pe = ph.phase_env(spec.name, c.plant)
    ev = {k: pe[k] for k in CANARY_KEEP if k in pe}
    ev.update({k: pe[k] for k in TERRAIN_PASS if k in pe})
    ev.update(OBST_EVAL)
    ev.update({"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0"})
    ev = _student_env(spec, with_plant(ev, c.plant))
    latest, a = _run_eval(c, tag, "flat_walk_slow_side", ["--manifest", str(SIDE_MANIFEST), "--scenario", "flat_walk_slow_side",
                                                          "--checkpoint", ckpt, *_student_args(spec)],
                          ev, Path(str(EVAL_ROOT / c.dir.name / tag) + "_obst"), c.plant, 5400)
    if a is None:
        return None
    return {"completion": a.get("schedule_completion_rate") or 0.0, "tripod": _med(a.get("tripod_score")) or 0.0,
            "falls": (a.get("termination_reasons") or {}).get("fall", 0), "n": a.get("n_episodes"), "run_dir": str(latest)}


# ------------------------------------------------------------------ records
def smoke_fields(lp: Path) -> dict:
    fail = series(lp, "Episode_Termination/crab_failure")
    coll = series(lp, "Episode_Reward/reward_collision")
    ep = series(lp, "Mean episode length")
    return {"fail_2k": tail_mean(fail[:SMOKE_ITERS], 100) if fail else None,
            "fail_end": tail_mean(fail, 100) if fail else None,
            "coll_2k": tail_mean(coll[:SMOKE_ITERS], 100) if coll else None,
            "ep_len_2k": tail_mean(ep[:SMOKE_ITERS], 100) if ep else None}


def student_fields(lp: Path) -> dict:
    """Distillation telemetry: the depth-actor / yaw losses printed by learn_vision."""
    txt = lp.read_text(errors="ignore") if lp.exists() else ""
    out = {}
    for key in ("depth_actor_loss", "yaw_loss", "depth_encoder_loss"):
        # learn_vision prints either "<key>: v" or "Mean <key> loss: v" depending on the branch
        vals = [float(m) for m in re.findall(rf"(?:Mean )?{key}(?: loss)?:\s*(-?[0-9.]+(?:e-?\d+)?)", txt)]
        if vals:
            out[key + "_first"] = sum(vals[:20]) / len(vals[:20])
            out[key + "_last"] = sum(vals[-20:]) / len(vals[-20:])
    out["iterations_logged"] = len(re.findall(r"Learning iteration", txt))
    # Survival of the student-driven rollouts (the 2026-09-08 phase-3a failure: episode length ~165
    # steps and crab_failure 1.0 for 5000 iterations while the depth losses kept falling).
    ep = series(lp, "Mean episode length") if lp.exists() else []
    fail = series(lp, "Episode_Termination/crab_failure") if lp.exists() else []
    out["ep_len_first"] = tail_mean(ep[:50], 50) if ep else None
    out["ep_len_last"] = tail_mean(ep, 50) if ep else None
    out["crab_failure_last"] = tail_mean(fail, 50) if fail else None
    return out


def _spawn_rot_ok(txt: str) -> bool:
    """The robot's scene.robot.init_state.rot in params/env.yaml: w ~ 0.7071, z ~ -0.7071
    (yaw -pi/2, Appendix A of the plan). The robot's is the first init_state block in the dump."""
    m = re.search(
        r"init_state:\s*\n\s*pos: !!python/tuple\s*\n(?:\s*-\s*[-\d.eE+]+\s*\n){3}"
        r"\s*rot: !!python/tuple\s*\n\s*-\s*([-\d.eE+]+)\s*\n\s*-\s*([-\d.eE+]+)\s*\n"
        r"\s*-\s*([-\d.eE+]+)\s*\n\s*-\s*([-\d.eE+]+)",
        txt,
    )
    if not m:
        return False
    w, z = float(m.group(1)), float(m.group(4))
    return abs(w - 0.7071067811865476) < 1e-3 and abs(z - (-0.7071067811865476)) < 1e-3


def phase_checks(rec: dict, spec, c: Campaign) -> list[tuple[str, bool]]:
    """Recomputable from the record (no GPU): plant, horizon, telemetry, task."""
    checks = []
    ckpt = rec.get("ckpt")
    if not ckpt:
        return [("training produced a checkpoint", False)]
    run_dir = Path(ckpt).parent
    envy = run_dir / "params" / "env.yaml"
    txt = envy.read_text(errors="ignore") if envy.exists() else ""
    want = ph.plant_usd_path(c.plant)
    checks.append((f"plant is {c.plant} (params/env.yaml usd path)", (Path(want).name in txt) if want else any(n in txt for n in ph.MAIN_ASSET_NAMES)))
    eps = ph.phase_env(spec.name, c.plant).get("KRABBY_EPISODE_S")
    if eps:
        checks.append((f"episode_length_s {eps} in params", re.search(rf"episode_length_s:\s*{float(eps)}", txt) is not None))
    agent_txt = (run_dir / "params" / "agent.yaml").read_text(errors="ignore") if (run_dir / "params" / "agent.yaml").exists() else ""
    checks.append((f"experiment {spec.experiment}", f"experiment_name: {spec.experiment}" in agent_txt))
    lp = Path(rec["log"])
    ltxt = lp.read_text(errors="ignore") if lp.exists() else ""
    if spec.kind == "rl":
        checks.append(("exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)", "reach_obst_frac" in ltxt))
        checks.append(("gait income telemetry logged (Episode_Reward/reward_clock_schedule)", "reward_clock_schedule" in ltxt))
        checks.append(("mirror-symmetry loss active (Loss/symmetry or mirror)", re.search(r"symmetry|mirror", ltxt, re.I) is not None))
        checks.append(("spawn yaw -pi/2 (params/env.yaml robot init_state rot)", _spawn_rot_ok(txt)))
    else:
        checks.append(("distillation algorithm (DistillationWithExtractor)", "DistillationWithExtractor" in agent_txt))
        checks.append(("depth losses logged", bool(rec.get("student", {}).get("depth_actor_loss_last") is not None
                                                  or re.search(r"depth", ltxt, re.I))))
        checks.append(("iteration counter continued from the teacher head",
                       _ckpt_iter(ckpt) >= _ckpt_iter(rec.get("resume")) + rec["iters"] - 100))
        st = rec.get("student") or {}
        checks.append(("student rollouts survive (mean episode length last 50 its >= 600 steps)",
                       (st.get("ep_len_last") or 0) >= 600))
        checks.append(("student failure share last 50 its < 0.9",
                       st.get("crab_failure_last") is not None and st["crab_failure_last"] < 0.9))
    return checks


def phase_lines(rec: dict, spec) -> list[str]:
    ex = rec.get("exposure") or {}
    s = (rec.get("evals") or {}).get("slow") or {}
    p = (rec.get("evals") or {}).get("step") or {}
    o = (rec.get("evals") or {}).get("obst") or {}
    ck = Path(rec["ckpt"]).parent.name + "/" + Path(rec["ckpt"]).name if rec.get("ckpt") else "—"
    lines = [f"- status **{rec.get('status')}** | checkpoint `{ck}` | resume `{Path(rec['resume']).name if rec.get('resume') else 'scratch'}` | "
             f"{rec.get('iters')} its | wall {rec.get('wall_h', float('nan')):.2f} h"]
    if spec.kind == "rl":
        sm = rec.get("smoke") or {}
        lines.append(f"- smoke fail@2k {fmt(sm.get('fail_2k'))} fail@end {fmt(sm.get('fail_end'))} coll@2k {fmt(sm.get('coll_2k'))} ep_len@2k {fmt(sm.get('ep_len_2k'))}")
        if rec.get("ckpt"):
            lines += expo_lines(ex)
            lines.append(f"- stand time frac (logged) {fmt(ex.get('stand_frac_actual'))} | mean reward {fmt(ex.get('mean_reward'))} | vloss {fmt(ex.get('vloss'))} | mean ep len {fmt(ex.get('eplen'))}")
    else:
        st = rec.get("student") or {}
        lines.append(f"- distillation: iterations logged {st.get('iterations_logged')} | depth_actor_loss first {fmt(st.get('depth_actor_loss_first'))} -> last {fmt(st.get('depth_actor_loss_last'))} | "
                     f"yaw_loss first {fmt(st.get('yaw_loss_first'))} -> last {fmt(st.get('yaw_loss_last'))}")
        lines.append(f"- student rollouts: mean episode length first50 {fmt(st.get('ep_len_first'))} -> last50 {fmt(st.get('ep_len_last'))} steps | "
                     f"crab_failure last50 {fmt(st.get('crab_failure_last'))}")
    if rec.get("ckpt") and rec.get("evals"):
        by_hold = " / ".join(f"{h} {fmt(v)}" for h, v in (s.get("tripod_by_hold") or {}).items()) or "n/a"
        lines.append(f"- slow canary (side manifest): walking-hold tripod {fmt(s.get('tripod_walking'))} (gated; by hold: {by_hold}; "
                     f"episode {fmt(s.get('tripod'))}) | completion {fmt(s.get('completion'))} | "
                     f"tracking {fmt(s.get('tracking'))} | falls {s.get('falls')}/{s.get('n')} | pitch-fwd share {fmt(s.get('pitch_fwd_share'))}")
        lines.append(f"- step onset (side manifest, shallow 0.05-0.2): completion {fmt(p.get('completion'))} | falls {p.get('falls')}/{p.get('n')} | "
                     f"pitch-fwd share {fmt(p.get('pitch_fwd_share'))}")
        lines.append(f"- obstacle eval (recal2b2w 0.20-0.70): completion {fmt(o.get('completion'))} | tripod {fmt(o.get('tripod'))} | falls {o.get('falls')}/{o.get('n')}")
    for name, ok in rec.get("checks") or []:
        lines.append(f"- check {'PASS' if ok else 'FAIL'}: {name}")
    return lines


def run_phase(c: Campaign, spec, resume_ckpt: str | None, iters: int, do_eval: bool) -> dict:
    c.st["run_no"] += 1
    tag = f"{c.key}_{spec.name}_{c.st['run_no']:03d}_{c.plant.replace('+', 'p')}" + (f"_it{iters}" if iters != spec.iterations else "")
    t0 = time.time()
    ckpt, status, lp = train_retry(c, spec, tag, iters, resume_ckpt, soundness=(resume_ckpt is None and iters >= SMOKE_ITERS))
    rec = {"phase": spec.name, "tag": tag, "task": spec.task, "mode": spec.teacher_mode, "ckpt": ckpt, "status": status,
           "resume": resume_ckpt, "iters": iters, "log": str(lp), "wall_h": (time.time() - t0) / 3600.0,
           "env": {"KRABBY_SIDE_PHASE": spec.name, "KRABBY_PLANT": c.plant}, "evals": {}}
    if spec.kind == "rl":
        rec["smoke"] = smoke_fields(lp) if lp.exists() else {}
        rec["exposure"] = exposure_from_log(lp) if lp.exists() else {}
    else:
        rec["student"] = student_fields(lp)
    if ckpt and do_eval:
        rec["evals"]["slow"] = eval_morph(c, spec, ckpt, "slow", tag)
        rec["evals"]["step"] = eval_morph(c, spec, ckpt, "step", tag)
        rec["evals"]["obst"] = eval_obst(c, spec, ckpt, tag)
        # written 1a gate (Part F): a failed formation must not burn the 2a-2c hours unattended.
        if spec.name == "1a" and rec["status"] == "ok":
            status, msg = gate_1a_verdict(rec["evals"].get("slow") or {})
            if status == "ok":
                c.log(f"{tag}: {msg}")
            else:
                rec["status"] = status
                c.notify(f"{tag}: {msg}")
    rec["checks"] = phase_checks(rec, spec, c)
    return rec


def gate_1a_verdict(s: dict) -> tuple[str, str]:
    """The written 1a gate on the slow-canary summary: ``("ok" | "gate_1a", message)``.

    Enforced pair: walking-hold tripod (``summarize()["tripod_walking"]``: the minimum over the holds
    with a walking command; the cmd-0 stand hold is excluded, see ``walking_tripod``) >= GATE_1A_TRIPOD
    AND completion >= GATE_1A_COMPLETION. No metrics (three eval attempts failed, or no walking hold
    in the schedule) is a halt too: an ungated head must not start 2a unattended -- proceeding is an
    explicit --continue decision.
    """
    tri, comp = s.get("tripod_walking"), s.get("completion")
    by_hold = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a"
                        for h, v in (s.get("tripod_by_hold") or {}).items()) or "none"
    if tri is None or comp is None:
        return "gate_1a", (f"1a formation gate could not be evaluated — slow__side produced no walking-hold tripod "
                           f"/ completion (walking tripod={tri!r}, completion={comp!r}; by hold: {by_hold}); campaign "
                           "HALTED (inspect the eval log, then relaunch with --continue to carry on ungated, or fix and --redo)")
    if tri < GATE_1A_TRIPOD or comp < GATE_1A_COMPLETION:
        return "gate_1a", (f"1a formation gate FAILED — walking-hold tripod {tri:.3f} (min over walking holds, need >= "
                           f"{GATE_1A_TRIPOD}; by hold: {by_hold}) / completion {comp:.3f} (need >= {GATE_1A_COMPLETION}) "
                           "on slow__side — campaign HALTED (relaunch with --continue to override and carry on, or fix and --redo)")
    return "ok", (f"1a formation gate passed — walking-hold tripod {tri:.3f} (by hold: {by_hold}) / "
                  f"completion {comp:.3f} on slow__side")


# ------------------------------------------------------------------ pause / continue
def interrupted_phase(c: Campaign) -> dict | None:
    """The phase recorded in state['current'] (a run cut short by a stop): its newest checkpoint and
    the iterations left to reach the phase's original final iteration."""
    cur = c.st.get("current") or {}
    if not cur.get("phase") or not cur.get("log"):
        return None
    spec = ph.PHASES[cur["phase"]]
    log = Path(cur["log"])
    txt = log.read_text(errors="ignore") if log.exists() else ""
    m = re.search(r"Exact experiment name requested from command line: (\S+)", txt)
    run_dir = LOG_ROOT / spec.experiment / m.group(1) if m else None
    if run_dir is None or not run_dir.exists():
        c.log(f"--continue: no run directory found for the interrupted {cur['phase']} run ({log})")
        return None
    models = sorted(run_dir.glob("model_*.pt"), key=_ckpt_iter)
    if not models:
        return None
    target = cur.get("target_final")
    if target is None:
        hdr = re.search(r"Learning iteration (\d+)/(\d+)", txt)
        target = int(hdr.group(2)) - 1 if hdr else None
    if target is None:
        return None
    last = models[-1]
    remaining = int(target) - _ckpt_iter(str(last)) + 1
    if remaining <= 0:
        return None
    started = cur.get("started")
    wall_h = 0.0
    try:
        wall_h = (log.stat().st_mtime - datetime.fromisoformat(started).timestamp()) / 3600.0 if started else 0.0
    except Exception:
        pass
    resume = cur.get("resume")
    if resume is None:          # older state records: take the loaded checkpoint from the train log
        mm = re.search(r"Loading model checkpoint from: (\S+)", txt)
        resume = mm.group(1) if mm else None
    return {"phase": cur["phase"], "run_dir": str(run_dir), "ckpt": str(last), "remaining": remaining,
            "iters": cur.get("iters", spec.iterations), "resume": resume, "log": str(log), "wall_h": wall_h}


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--plant", default=ph.MAIN_PLANT, choices=sorted(ph.PLANTS),
                    help="named plant (default: the main asset = A15+B; legacy_golden for pre-2026-09-09 reproductions)")
    ap.add_argument("--seed", default="3")
    ap.add_argument("--phases", default="1a,2a,2b,2c,3a,3b")
    ap.add_argument("--from-checkpoint", default=None, help="resume the first listed phase from this checkpoint")
    ap.add_argument("--iterations", type=int, default=None, help="override every phase's iteration count (smokes)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--pause-after", default=DEFAULT_PAUSE, help="phases after which to stop and wait ('' = none)")
    ap.add_argument("--redo", action="store_true", help="rerun phases already recorded ok for this seed")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="finish the interrupted phase recorded in state['current'] from its newest checkpoint "
                         "(remaining iterations only), then carry on with the listed phases; also overrides a "
                         "gate_1a halt (uses the recorded checkpoint without retraining)")
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in ph.PHASES]
    if unknown:
        ap.error(f"unknown phases {unknown}; known: {list(ph.PHASES)}")
    pause_after = {p.strip() for p in args.pause_after.split(",") if p.strip()}
    c = Campaign(Path(args.campaign_dir).resolve(), args.plant, str(args.seed))
    c.dir.mkdir(parents=True, exist_ok=True)
    c.ensure_header(phases, args)
    c.st["phase"] = "run"
    c.save()
    _wait_isaac_clear()

    prev_ckpt = args.from_checkpoint
    partial = interrupted_phase(c) if args.cont else None
    for i, name in enumerate(phases):
        spec = ph.PHASES[name]
        rec = c.recs.get(name)
        if partial and partial["phase"] == name:
            # Continue the interrupted run: resume its newest checkpoint for the remaining iterations,
            # then record the phase as a whole (original resume point, full iteration count).
            c.log(f"{name}: continuing interrupted run {Path(partial['run_dir']).name} from "
                  f"{Path(partial['ckpt']).name} ({partial['remaining']} of {partial['iters']} iterations left)")
            rec = run_phase(c, spec, partial["ckpt"], partial["remaining"], do_eval=not args.no_eval)
            rec.update({"resume": partial["resume"], "iters": partial["iters"], "continued_from": partial["ckpt"],
                        "interrupted_log": partial["log"], "wall_h": rec["wall_h"] + partial.get("wall_h", 0.0)})
            rec["checks"] = phase_checks(rec, spec, c)
            c.recs[name] = rec
            c.st["current"] = {}
            c.save()
            c.report_block(f"## Phase {name} — {spec.task} — {c.key} — {datetime.now():%Y-%m-%d %H:%M} (continued after a pause)\n\n"
                           + "\n".join(phase_lines(rec, spec)))
            c.changelog_row(f"phase {name} {rec['status']} (continued from {Path(partial['ckpt']).name}): "
                            f"{Path(rec['ckpt']).name if rec.get('ckpt') else 'no checkpoint'} ({rec['tag']})")
            if rec["status"] != "ok":
                c.st["phase"] = f"halt:{name} {rec['status']}"
                c.save(); c.notify(f"phase {name} {rec['status']} — campaign HALTED (see REPORT)"); return 1
            prev_ckpt = rec["ckpt"]
            partial = None
            if name in pause_after and i < len(phases) - 1:
                c.st["phase"] = f"await_user:{name}"
                c.save()
                c.notify(f"phase {name} done ({Path(rec['ckpt']).name}); PAUSED before {phases[i + 1]} — relaunch with the same "
                         f"--phases to continue")
                return 0
            continue
        if rec and rec.get("ckpt") and not args.redo and not (i == 0 and args.from_checkpoint) and (
                rec.get("status") == "ok" or (rec.get("status") == "gate_1a" and args.cont)):
            if rec.get("status") == "gate_1a":
                nxt = phases[i + 1] if i + 1 < len(phases) else "nothing"
                stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
                c.log(f"{name}: gate_1a recorded but --continue overrides — using {Path(rec['ckpt']).name} unretrained")
                # make the record self-describing: a later relaunch WITHOUT --continue must not retrain
                # this phase from scratch on top of a lineage that already resumed from its head
                rec["status"] = "ok"
                rec["gate_1a_override"] = f"{stamp} --continue (see the CHARTER Log)"
                c.recs[name] = rec
                c.save()
                c.changelog_row(f"phase {name}: gate_1a overridden by --continue — {Path(rec['ckpt']).name} carried into {nxt} "
                                "unretrained (gate revised to the walking-hold tripod, see CHARTER); record status -> ok")
                c.report_block(f"## Phase {name} — gate_1a overridden — {stamp}\n\n- `--continue`: {Path(rec['ckpt']).name} "
                               f"carried into {nxt} unretrained; the gate was revised to the walking-hold tripod (CHARTER: "
                               "gate section + Log); the 1a record now reads `status ok` with `gate_1a_override`")
            else:
                c.log(f"{name}: already recorded ok ({Path(rec['ckpt']).name}) — skipping")
            prev_ckpt = rec["ckpt"]
            continue
        resume = prev_ckpt if (i > 0 or args.from_checkpoint) else None
        if resume is None and spec.resume_from:
            resume = (c.recs.get(spec.resume_from) or {}).get("ckpt")
            if not resume:
                c.st["phase"] = f"halt:{name} needs the {spec.resume_from} head (pass --from-checkpoint)"
                c.save(); c.notify(c.st["phase"]); return 2
        iters = args.iterations or spec.iterations
        rec = run_phase(c, spec, resume, iters, do_eval=not args.no_eval)
        c.recs[name] = rec
        c.st["current"] = {}
        c.save()
        c.report_block(f"## Phase {name} — {spec.task}" + (f" (mode {spec.teacher_mode})" if spec.teacher_mode else "") +
                       f" — {c.key} — {datetime.now():%Y-%m-%d %H:%M}\n\n" + "\n".join(phase_lines(rec, spec)))
        c.changelog_row(f"phase {name} {rec['status']}: {Path(rec['ckpt']).name if rec.get('ckpt') else 'no checkpoint'} ({rec['tag']})")
        if rec["status"] != "ok":
            c.st["phase"] = f"halt:{name} {rec['status']}"
            c.save(); c.notify(f"phase {name} {rec['status']} — campaign HALTED (see REPORT)"); return 1
        prev_ckpt = rec["ckpt"]
        if name in pause_after and i < len(phases) - 1:
            c.st["phase"] = f"await_user:{name}"
            c.save()
            c.notify(f"phase {name} done ({Path(rec['ckpt']).name}); PAUSED before {phases[i + 1]} — relaunch with the same "
                     f"--phases to continue")
            return 0
    c.st["phase"] = "done"
    c.save()
    c.notify(f"phases {', '.join(phases)} complete for {c.key} — see REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
