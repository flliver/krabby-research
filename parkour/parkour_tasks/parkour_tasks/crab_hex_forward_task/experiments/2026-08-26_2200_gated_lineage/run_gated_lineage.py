#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""PLAN F gated-lineage orchestrator (2026-08-26).

Runs the element schedule from ~/.claude/plans/we-now-have-a-agile-allen.md as one
resumable campaign: 2k-iteration chunks, per-chunk gates (training failure tail + flat
canary + element metric + anti-correlation check), doubling dose ladders with
binary-search-on-failure (3 halvings then DEFER), critic resets on income-structure
changes, RSI bank refreshes at phase milestones, one CHANGELOG row per chunk, and a
persistent JSON state file so the campaign resumes exactly where it stopped.

Stdlib only. Launch under systemd-run (see launch_gated_lineage.sh).
"""
from __future__ import annotations

import json
import os
import re
import shlex
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
STATE = HERE / "state.json"
CHANGELOG = HERE / "CHANGELOG.md"

M1C = FLAT_RUNS / "2026-08-25_19-09-15/model_24900.pt"
CHUNK_ITERS = 2000
FAIL_GATE = 0.35
CANARY_GATE = {"tripod": 0.46, "completion": 0.80, "tracking": 0.37}
OBSTACLE_GATE = 0.48

BASE0 = {
    "KRABBY_CLOCK_W": "1.0",
    "KRABBY_APEX_W": "1.0",
    "KRABBY_RSI_FRAC": "0.2",
    "KRABBY_RSI_BANK": str(REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/rsi_bank_E1.npz"),
    "KRABBY_FLAT_TERRAIN_MODE": "light",
    "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8",
    "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    "KRABBY_TRACK_SIGMA2": "0.1",
    "KRABBY_TRACK_L1_W": "-1.0",
    "KRABBY_LIN_VEL_X": "0.0:0.35",
}
# Vars that alter obs/action/command semantics and must match at canary-eval time;
# terrain vars are deliberately excluded from the canary (it measures flat gait).
CANARY_VARS = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W",
               "KRABBY_CLOCK_W", "KRABBY_APEX_W", "KRABBY_HEADING",
               "KRABBY_HEADING_STIFFNESS", "KRABBY_ACTION_SCALE")
# KRABBY_EPISODE_S deliberately excluded: evals run their scenario-pinned lengths.

# ---------------------------------------------------------------------------------
# Element schedule. metric: canary | obstacle | turn | goal_idx | terrain_levels | speed
# ladder values are per-element semantics (see apply()).
SCHEDULE = [
    dict(id="P0_null", phase="P0", kind="null", ladder=[None], reset=False, metric="canary",
         milestone_harvest=True),
    dict(id="P1_turning", phase="P1", kind="var", var="KRABBY_HEADING",
         ladder=["-0.3:0.3", "-0.6:0.6", "-1.2:1.2"], reset=False, metric="turn",
         milestone_harvest=True),
    dict(id="P2_episode40", phase="P2", kind="var", var="KRABBY_EPISODE_S",
         ladder=["40"], reset=False, metric="canary"),
    dict(id="P2_yaw_income", phase="P2", kind="reward", var="KRABBY_YAW_W",
         ladder=["0.05", "0.1", "0.2"], reset=True, metric="goal_idx"),
    dict(id="P2_goalvel_income", phase="P2", kind="reward", var="KRABBY_GOAL_VEL_W",
         ladder=["0.19", "0.38", "0.75"], reset=True, metric="goal_idx", notify_after=True),
    dict(id="P3_frac35", phase="P3", kind="var", var="KRABBY_FLAT_TERRAIN_FLAT_FRAC",
         ladder=["0.65"], reset=False, metric="obstacle"),
    dict(id="P3_frac50", phase="P3", kind="var", var="KRABBY_FLAT_TERRAIN_FLAT_FRAC",
         ladder=["0.5"], reset=False, metric="obstacle"),
    dict(id="P3_diff_raise", phase="P3", kind="var", var="KRABBY_FLAT_TERRAIN_DIFF",
         ladder=["0.10:0.35"], reset=False, metric="obstacle"),
    dict(id="P3_geom_recal", phase="P3", kind="multivar",
         vars={"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70"},
         ladder=[None], reset=False, metric="obstacle", obstacle_gate=0.45),
    dict(id="P3_curriculum_on", phase="P3", kind="multivar",
         vars={"KRABBY_FLAT_TERRAIN_CURRICULUM": "1", "KRABBY_TERRAIN_PROMOTE": "0.45:0.25"},
         ladder=[None], reset=False, metric="terrain_levels"),
    dict(id="P3_safety_pack", phase="P3", kind="rewardpack",
         pack={"KRABBY_EDGE_W": -0.3, "KRABBY_STUMBLE_W": -1.0, "KRABBY_COLLISION_W": -2.0},
         ladder=[0.25, 0.5, 1.0], reset=True, metric="obstacle",
         milestone_harvest=True),
    dict(id="P4_obstacle_clear", phase="P4", kind="reward", var="KRABBY_CLEARANCE_W",
         ladder=["0.45", "0.9"], reset=True, metric="obstacle_improve"),
    dict(id="P4_foot_clear", phase="P4", kind="multivar",
         vars={"KRABBY_FOOT_CLEAR_FLAT": "1", "KRABBY_FOOT_CLEAR_W": "0.5",
               "KRABBY_FOOT_CLEAR_MIN": "0.03"},
         ladder=[None], reset=True, metric="obstacle_improve", skip_unless_prev_improved=True),
    dict(id="P4_swing_min", phase="P4", kind="reward", var="KRABBY_SWING_MIN_CLEAR_W",
         ladder=["-0.1", "-0.4"], reset=True, metric="obstacle_improve",
         skip_unless_prev_improved=True),
    dict(id="P5_speed45", phase="P5", kind="var", var="KRABBY_LIN_VEL_X",
         ladder=["0.0:0.45"], reset=False, metric="speed"),
    dict(id="P5_speed55", phase="P5", kind="var", var="KRABBY_LIN_VEL_X",
         ladder=["0.0:0.55"], reset=False, metric="speed", milestone_harvest=True,
         notify_after=True),
    dict(id="P6_push", phase="P6", kind="var", var="KRABBY_DR_PUSH",
         ladder=["0.25", "0.5"], reset=False, metric="canary"),
    dict(id="P6_mass_com", phase="P6", kind="multivar",
         vars={"KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_COM": "0.01"},
         ladder=[None], reset=False, metric="canary"),
    dict(id="P7_consolidate", phase="P7", kind="null", ladder=[None, None, None],
         reset=False, metric="graduation"),
]

# ---------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[lineage {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def changelog(row: str) -> None:
    with CHANGELOG.open("a") as fh:
        fh.write(row.rstrip() + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"base": dict(BASE0), "ckpt": str(M1C), "idx": 0, "ladder_pos": 0,
            "deferred": [], "chunk_no": 0, "prev_obstacle": None, "last_reset_iter": -10**9}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def run(cmd: list[str], env_extra: dict, log_path: Path, timeout_s: int) -> int:
    env = dict(os.environ)
    env.update({"OMNI_KIT_ACCEPT_EULA": "yes", "TERM": "xterm"})
    env.update({k: str(v) for k, v in env_extra.items()})
    with log_path.open("w") as fh:
        try:
            p = subprocess.run(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout_s)
            return p.returncode
        except subprocess.TimeoutExpired:
            return 124


def series(log_path: Path, label: str) -> list[float]:
    pat = re.compile(re.escape(label) + r":\s*(-?[0-9.]+)")
    vals = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    return vals


def latest_run_ckpt() -> str:
    run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(models[-1])


def train_chunk(env_vars: dict, ckpt: str, tag: str) -> tuple[str | None, Path]:
    log_path = HERE / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", "Isaac-Crab-Hex-Flat-Walk-v0", "--headless",
           "--num_envs", "256", "--seed", "2", "--max_iterations", str(CHUNK_ITERS),
           "--resume", "--checkpoint", ckpt]
    # Early-abort: run in a subprocess and poll the log for the destruction signature.
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update({k: str(v) for k, v in env_vars.items()})
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT)
    aborted = False
    try:
        while proc.poll() is None:
            time.sleep(120)
            fails = series(log_path, "Episode_Termination/crab_failure")
            if len(fails) >= 300:
                w = [sum(fails[i - 100:i]) / 100 for i in (len(fails) - 200, len(fails) - 100, len(fails))]
                if w[0] < w[1] < w[2] and w[2] > 0.6:
                    log(f"{tag}: destruction signature ({w}) — early abort")
                    proc.kill()
                    aborted = True
                    break
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
    if aborted:
        return None, log_path
    return latest_run_ckpt(), log_path


def eval_scenario(scenario: str, ckpt: str, env_vars: dict, tag: str) -> dict:
    log_path = HERE / f"{tag}_eval.log"
    rc = run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST), "--scenario", scenario,
              "--checkpoint", ckpt, "--no-plot", "--output-root", str(EVAL_ROOT)],
             env_vars, log_path, 3600)
    out = {"rc": rc}
    # newest eval dir under EVAL_ROOT/<scenario>
    droot = EVAL_ROOT / scenario / "seed001"
    if droot.exists():
        d = max(droot.iterdir(), key=lambda p: p.stat().st_mtime)
        sm = d / "scenario_metrics.json"
        if sm.exists():
            agg = json.loads(sm.read_text())["aggregate"]
            out["tripod"] = (agg.get("tripod_score") or {}).get("median") or 0.0
            out["completion"] = agg.get("schedule_completion_rate") or 0.0
            out["tracking"] = (agg.get("tracking_ratio") or {}).get("median") or 0.0
            out["dir"] = str(d)
    return out


def canary(ckpt: str, base: dict, tag: str) -> tuple[bool, dict]:
    ev = {k: base[k] for k in CANARY_VARS if k in base}
    r = eval_scenario("flat_walk_slow_v2", ckpt, ev, tag + "_canary")
    ok = (r.get("tripod", 0) >= CANARY_GATE["tripod"]
          and r.get("completion", 0) >= CANARY_GATE["completion"]
          and r.get("tracking", 0) >= CANARY_GATE["tracking"])
    return ok, r


def obstacle_eval(ckpt: str, base: dict, tag: str) -> dict:
    ev = {k: base[k] for k in CANARY_VARS if k in base}
    ev.update({"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0",
               "KRABBY_FLAT_TERRAIN_DIFF": base.get("KRABBY_FLAT_TERRAIN_DIFF", "0.05:0.2"),
               "KRABBY_FLAT_TERRAIN_GEOM": base.get("KRABBY_FLAT_TERRAIN_GEOM", "shallow")})
    return eval_scenario("flat_walk_slow_v2", ckpt, ev, tag + "_obst")


def turn_ratio(ckpt: str, base: dict, tag: str) -> float:
    ev = {k: base[k] for k in CANARY_VARS if k in base}
    r = eval_scenario("turn_walk_v1", ckpt, ev, tag + "_turn")
    d = r.get("dir")
    if not d:
        return 0.0
    ratios = []
    for f in Path(d).glob("metrics/episode_*.json"):
        ep = json.loads(f.read_text())
        for label, hold in (ep.get("holds") or {}).items():
            tr = (hold.get("tracking") or {}).get("wz") or {}
            cm, am = tr.get("cmd_mean"), tr.get("actual_mean")
            if cm and abs(cm) > 0.05 and am is not None:
                ratios.append(am / cm)
    ratios.sort()
    return ratios[len(ratios) // 2] if ratios else 0.0


def anti_corr_clean(train_log: Path, income_label: str | None) -> bool:
    if not income_label:
        return True
    inc = series(train_log, f"Episode_Reward/{income_label}")
    fail = series(train_log, "Episode_Termination/crab_failure")
    if len(inc) < 600 or len(fail) < 600:
        return True
    thirds = lambda v: [sum(v[i:i + len(v) // 3]) / (len(v) // 3)
                        for i in (0, len(v) // 3, 2 * len(v) // 3)]
    iu = thirds(inc); fu = thirds(fail)
    rising = lambda w: w[0] < w[1] < w[2]
    return not (rising([abs(x) for x in iu]) and rising(fu) and fu[2] > 0.5)


INCOME_LABEL = {"KRABBY_YAW_W": "reward_tracking_yaw",
                "KRABBY_GOAL_VEL_W": "reward_tracking_goal_vel",
                "KRABBY_CLEARANCE_W": "reward_obstacle_clearance",
                "KRABBY_FOOT_CLEAR_W": "reward_foot_clearance"}


def element_vars(el: dict, dose) -> dict:
    if el["kind"] in ("null",):
        return {}
    if el["kind"] == "var" or el["kind"] == "reward":
        return {el["var"]: dose}
    if el["kind"] == "multivar":
        return dict(el["vars"])
    if el["kind"] == "rewardpack":
        return {k: str(round(w * float(dose), 4)) for k, w in el["pack"].items()}
    raise ValueError(el["kind"])


def needs_reset(el: dict, first_rung: bool) -> bool:
    return bool(el.get("reset")) and first_rung


def gate(el: dict, st: dict, train_log: Path, ckpt: str, tag: str) -> tuple[bool, str]:
    fails = series(train_log, "Episode_Termination/crab_failure")
    skip = 10 if (st["chunk_no"] and st.get("just_reset")) else 0  # ~200 iters at print cadence
    tail = fails[-(20 + skip):-skip] if skip else fails[-20:]
    fail_tail = sum(tail) / len(tail) if tail else 1.0
    if fail_tail >= FAIL_GATE:
        if fail_tail < 0.5 and len(fails) > 40 and fails[-1] < fails[-40]:
            return False, f"borderline fail_tail {fail_tail:.2f} (falling)"
        return False, f"fail_tail {fail_tail:.2f}"
    ok, c = canary(ckpt, st["base"], tag)
    if not ok:
        return False, f"canary {c.get('tripod', 0):.2f}/{c.get('completion', 0):.2f}/{c.get('tracking', 0):.2f}"
    label = INCOME_LABEL.get(el.get("var", ""))
    if not anti_corr_clean(train_log, label):
        return False, "capability-gap signature (income up + failure up)"
    m = el["metric"]
    if m == "canary":
        return True, f"fail {fail_tail:.2f} canary ok"
    if m in ("obstacle", "obstacle_improve"):
        o = obstacle_eval(ckpt, st["base"], tag)
        comp = o.get("completion", 0)
        st["last_obstacle"] = comp
        bar = el.get("obstacle_gate", OBSTACLE_GATE)
        if m == "obstacle":
            return comp >= bar, f"obstacle completion {comp:.2f} (bar {bar})"
        prev = st.get("prev_obstacle") or 0
        improved = comp >= prev + 0.03
        st["element_improved"] = improved
        return improved, f"obstacle {comp:.2f} vs prev {prev:.2f} (+3pt rule)"
    if m == "turn":
        tr = turn_ratio(ckpt, st["base"], tag)
        return tr >= 0.5, f"turn ratio {tr:.2f}"
    if m == "goal_idx":
        g = series(train_log, "Metrics/base_parkour/current_goal_idx")
        gtail = sum(g[-20:]) / 20 if len(g) >= 20 else 0.0
        pre = st.get("goal_idx_pre", 0.0)
        st["goal_idx_last"] = gtail
        return gtail >= pre - 0.05, f"goal_idx {gtail:.2f} (pre {pre:.2f})"
    if m == "terrain_levels":
        tl = series(train_log, "Metrics/base_parkour/terrain_levels")
        if len(tl) < 40:
            return False, "terrain_levels not found"
        rising = sum(tl[-20:]) / 20 > sum(tl[:20]) / 20 + 0.05
        return rising, f"terrain_levels {sum(tl[:20])/20:.2f}->{sum(tl[-20:])/20:.2f}"
    if m == "speed":
        ev = {k: st["base"][k] for k in CANARY_VARS if k in st["base"]}
        r = eval_scenario("flat_walk_speed_v1", ckpt, ev, tag + "_speed")
        tr = r.get("tracking", 0)
        return tr >= 0.30, f"speed tracking {tr:.2f}"
    if m == "graduation":
        return True, "consolidation chunk"
    return False, f"unknown metric {m}"


def notify(msg: str) -> None:
    changelog(f"> NOTIFY: {msg}")
    log("NOTIFY: " + msg)


def harvest_bank(st: dict, tag: str) -> None:
    out = HERE / f"rsi_bank_{tag}.npz"
    ev = {k: st["base"][k] for k in CANARY_VARS if k in st["base"]}
    ev["KRABBY_LIN_VEL_X"] = "0.25:0.35"
    rc = run([PY, str(HARVEST_TOOL), "--headless", "--num_envs", "16", "--steps", "800",
              "--checkpoint", st["ckpt"], "--out", str(out)], ev, HERE / f"{tag}_harvest.log", 3600)
    # Isaac teardown can hang after the bank is written (observed P0: rc=124 with a valid
    # bank on disk) — the completion marker is the file, not the exit code.
    if out.exists() and out.stat().st_size > 10000:
        st["base"]["KRABBY_RSI_BANK"] = str(out)
        changelog(f"| {tag} | RSI refresh | bank {out.name} | — | — | — | refreshed | — |")
    else:
        changelog(f"| {tag} | RSI refresh FAILED rc={rc} — keeping previous bank |")


def main() -> int:
    HERE.mkdir(exist_ok=True)
    if not CHANGELOG.exists():
        CHANGELOG.write_text(
            "# PLAN F gated lineage — CHANGELOG\n\n"
            "| element | dose | halvings | reset | verdict detail | ckpt |\n|---|---|---|---|---|---|\n")
    st = load_state()
    while st["idx"] < len(SCHEDULE):
        el = SCHEDULE[st["idx"]]
        if el.get("skip_unless_prev_improved") and not st.get("element_improved", False):
            changelog(f"| {el['id']} | — | — | — | SKIPPED (skip rule: prior rung did not improve) | — |")
            st["idx"] += 1; st["ladder_pos"] = 0; save_state(st); continue
        if el["metric"] == "goal_idx" and st["ladder_pos"] == 0:
            st["goal_idx_pre"] = st.get("goal_idx_last", 0.0)
        if el["metric"] == "obstacle_improve" and st["ladder_pos"] == 0:
            st["prev_obstacle"] = st.get("last_obstacle")
        ladder = el["ladder"]
        while st["ladder_pos"] < len(ladder):
            dose = ladder[st["ladder_pos"]]
            attempt, halvings, passed = dose, 0, False
            while halvings <= 3:
                st["chunk_no"] += 1
                tag = f"c{st['chunk_no']:03d}_{el['id']}_r{st['ladder_pos']}h{halvings}"
                start = st["ckpt"]
                st["just_reset"] = False
                if needs_reset(el, st["ladder_pos"] == 0) and halvings == 0:
                    rd = str(Path(start).with_suffix("")) + "_critic_reset.pt"
                    subprocess.run([PY, str(RESET_TOOL), "--src", start, "--dst", rd],
                                   check=True, capture_output=True)
                    start = rd
                    st["just_reset"] = True
                ev = dict(st["base"]); ev.update(element_vars(el, attempt))
                log(f"{tag}: training (dose={attempt})")
                ckpt2, tlog = train_chunk(ev, start, tag)
                if ckpt2 is None:
                    verdict = "EARLY-ABORT (destruction)"
                    ok = False
                else:
                    ok, verdict = gate(el, st, tlog, ckpt2, tag)
                changelog(f"| {el['id']} | {attempt} | {halvings} | {st['just_reset']} | {verdict} | "
                          f"{Path(ckpt2).parent.name + '/' + Path(ckpt2).name if ckpt2 else '—'} |")
                if ok:
                    st["ckpt"] = ckpt2
                    st["base"].update(element_vars(el, attempt))
                    passed = True
                    break
                halvings += 1
                if el["kind"] in ("var", "multivar", "null"):
                    break  # non-scalar doses cannot be halved: single attempt per rung
                try:
                    attempt = str(float(attempt) / 2)
                except (TypeError, ValueError):
                    break
            if not passed:
                prior = [d for d in st["deferred"] if d["id"] == el["id"]]
                if prior:
                    notify(f"HALT: {el['id']} deferred twice — user decision needed")
                    save_state(st)
                    return 2
                st["deferred"].append({"id": el["id"], "after_idx": st["idx"] + 2})
                changelog(f"| {el['id']} | — | — | — | DEFERRED (re-attempt after idx {st['idx'] + 2}) | — |")
                break
            st["ladder_pos"] += 1
            save_state(st)
        if el.get("milestone_harvest") and passed:
            harvest_bank(st, el["id"])
        if el.get("notify_after"):
            notify(f"phase boundary reached after {el['id']}: ckpt {st['ckpt']}")
        st["idx"] += 1
        st["ladder_pos"] = 0
        # re-insert any deferred element whose window arrived
        for d in list(st["deferred"]):
            if d["after_idx"] == st["idx"] and not d.get("requeued"):
                orig = next(e for e in SCHEDULE if e["id"] == d["id"])
                SCHEDULE.insert(st["idx"], orig)
                d["requeued"] = True
        save_state(st)
    notify("GRADUATION: schedule complete — run graduation battery")
    return 0


if __name__ == "__main__":
    sys.exit(main())
