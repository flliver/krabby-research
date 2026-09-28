#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""PLAN F-r orchestrator: front-loaded curriculum search (2026-08-27).

Architecture (user-directed revision of PLAN F):
  - Phase S: from-scratch formation runs with ALL curriculum elements active from
    iteration 0. Formation gate at 5k (tripod >= 0.40, completion >= 0.60 on the flat
    canary; one +2k trend extension if tripod is clearly rising).
  - On failure: GROUP BISECTION over the active elements (halves trained from scratch,
    recurse into failures); each culprit is DEFERRED; after each removal the remaining
    full set re-runs (interactions are real). Ends when the front-loaded set passes.
  - Phase D: deferred elements activate at the formation head via resume, dose-laddered
    with the original PLAN F gates (canary, element metric, critic reset on income
    switches); an element that also breaks the formed gait moves out one milestone,
    then HALTs for the user.
  - Phase G: consolidation chunks + graduation battery.

Stdlib only; resumable via state_v2.json; one CHANGELOG row per run/chunk.
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
MANIFEST = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml"
EVAL = PARKOUR / "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
FLAT_RUNS = PARKOUR / "logs/rsl_rl/crab_hex_flat_walk"
EVAL_ROOT = PARKOUR / "logs/rsl_rl/gait_eval/gated_lineage"
STATE = HERE / "state_v2.json"
CHANGELOG = HERE / "CHANGELOG.md"

FORMATION_ITERS = 5000
EXTENSION_ITERS = 2000
FORMATION_GATE = {"tripod": 0.40, "completion": 0.60}
CANARY_GATE = {"tripod": 0.46, "completion": 0.80, "tracking": 0.37}
FAIL_GATE = 0.35
CHUNK_ITERS = 2000

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

# Curriculum elements: default placement = iteration 0, full dose. An element leaves the
# front-loaded set only via bisection-proven formation breakage. "ladder" is used ONLY
# when the element is deferred (post-formation dose-laddered introduction).
ELEMENTS = {
    "heading": {"vars": {"KRABBY_HEADING": "-1.2:1.2"},
                "ladder": ["-0.3:0.3", "-0.6:0.6", "-1.2:1.2"], "reset": False, "metric": "turn"},
    "episode40": {"vars": {"KRABBY_EPISODE_S": "40"}, "ladder": ["40"], "reset": False,
                  "metric": "canary"},
    "yaw_income": {"vars": {"KRABBY_YAW_W": "0.2"}, "ladder": ["0.05", "0.1", "0.2"],
                   "reset": True, "metric": "canary", "income": "reward_tracking_yaw"},
    "goalvel_income": {"vars": {"KRABBY_GOAL_VEL_W": "0.75"},
                       "ladder": ["0.19", "0.38", "0.75"], "reset": True, "metric": "canary",
                       "income": "reward_tracking_goal_vel"},
    "terrain50": {"vars": {"KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5"},
                  "ladder": ["0.65", "0.5"], "reset": False, "metric": "obstacle"},
    "terrain_recal": {"vars": {"KRABBY_FLAT_TERRAIN_GEOM": "recal2b2",
                               "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70"},
                      "ladder": [None], "reset": False, "metric": "obstacle"},
    "terrain_curriculum": {"vars": {"KRABBY_FLAT_TERRAIN_CURRICULUM": "1",
                                    "KRABBY_TERRAIN_PROMOTE": "0.45:0.25"},
                           "ladder": [None], "reset": False, "metric": "canary"},
    "safety_pack": {"vars": {"KRABBY_EDGE_W": "-0.3", "KRABBY_STUMBLE_W": "-1.0",
                             "KRABBY_COLLISION_W": "-2.0"},
                    "ladder": [0.25, 0.5, 1.0], "reset": True, "metric": "canary",
                    "pack": {"KRABBY_EDGE_W": -0.3, "KRABBY_STUMBLE_W": -1.0,
                             "KRABBY_COLLISION_W": -2.0}},
    "clearance_pack": {"vars": {"KRABBY_CLEARANCE_W": "0.9", "KRABBY_FOOT_CLEAR_FLAT": "1",
                                "KRABBY_FOOT_CLEAR_W": "1.0", "KRABBY_FOOT_CLEAR_MIN": "0.03",
                                "KRABBY_SWING_MIN_CLEAR_W": "-0.4"},
                       "ladder": [0.5, 1.0], "reset": True, "metric": "canary",
                       "pack": {"KRABBY_CLEARANCE_W": 0.9, "KRABBY_FOOT_CLEAR_W": 1.0,
                                "KRABBY_SWING_MIN_CLEAR_W": -0.4},
                       "pack_extra": {"KRABBY_FOOT_CLEAR_FLAT": "1", "KRABBY_FOOT_CLEAR_MIN": "0.03"}},
    "speed55": {"vars": {"KRABBY_LIN_VEL_X": "0.0:0.55"}, "ladder": ["0.0:0.45", "0.0:0.55"],
                "reset": False, "metric": "canary"},
    "dr_push": {"vars": {"KRABBY_DR_PUSH": "0.5"}, "ladder": ["0.25", "0.5"], "reset": False,
                "metric": "canary"},
    "dr_masscom": {"vars": {"KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_COM": "0.01"},
                   "ladder": [None], "reset": False, "metric": "canary"},
}
SEARCH_ORDER = list(ELEMENTS)


def log(msg: str) -> None:
    print(f"[flsearch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def changelog(row: str) -> None:
    with CHANGELOG.open("a") as fh:
        fh.write(row.rstrip() + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"mode": "search", "active": list(SEARCH_ORDER), "deferred": [], "run_no": 0,
            "pending_sets": [], "head": None, "defer_idx": 0, "ladder_pos": 0,
            "second_chance": {}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def merged_vars(active: list[str]) -> dict:
    ev = dict(BASELINE)
    for name in active:
        ev.update(ELEMENTS[name]["vars"])
    return ev


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


def latest_run_ckpt() -> str:
    run_dir = max(FLAT_RUNS.iterdir(), key=lambda p: p.stat().st_mtime)
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(models[-1])


def train(env_vars: dict, tag: str, iters: int, resume_ckpt: str | None) -> tuple[str | None, Path]:
    log_path = HERE / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", "Isaac-Crab-Hex-Flat-Walk-v0", "--headless",
           "--num_envs", "256", "--seed", "2", "--max_iterations", str(iters)]
    if resume_ckpt:
        cmd += ["--resume", "--checkpoint", resume_ckpt]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update({k: str(v) for k, v in env_vars.items()})
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            time.sleep(120)
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, log_path
    if proc.returncode != 0:
        return None, log_path
    return latest_run_ckpt(), log_path


def eval_canary(ckpt: str, env_vars: dict, tag: str) -> dict:
    keep = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W",
            "KRABBY_APEX_W", "KRABBY_HEADING", "KRABBY_HEADING_STIFFNESS", "KRABBY_ACTION_SCALE")
    ev = {k: env_vars[k] for k in keep if k in env_vars}
    log_path = HERE / f"{tag}_canary.log"
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    env.update(ev)
    with log_path.open("w") as fh:
        subprocess.run([PY, str(EVAL), "--headless", "--manifest", str(MANIFEST),
                        "--scenario", "flat_walk_slow_v2", "--checkpoint", ckpt, "--no-plot",
                        "--output-root", str(EVAL_ROOT)],
                       cwd=PARKOUR, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
    droot = EVAL_ROOT / "flat_walk_slow_v2" / "seed001"
    d = max(droot.iterdir(), key=lambda p: p.stat().st_mtime)
    a = json.loads((d / "scenario_metrics.json").read_text())["aggregate"]
    return {"tripod": (a.get("tripod_score") or {}).get("median") or 0.0,
            "completion": a.get("schedule_completion_rate") or 0.0,
            "tracking": (a.get("tracking_ratio") or {}).get("median") or 0.0}


def formation_attempt(active: list[str], st: dict, label: str) -> tuple[bool, str, str | None]:
    """One from-scratch formation run with `active` elements. Returns (formed, detail, ckpt)."""
    st["run_no"] += 1
    tag = f"s{st['run_no']:03d}_{label}"
    ev = merged_vars(active)
    log(f"{tag}: from-scratch formation with {len(active)} elements: {sorted(active)}")
    ckpt, tlog = train(ev, tag, FORMATION_ITERS, resume_ckpt=None)
    if ckpt is None:
        return False, "train crashed", None
    c = eval_canary(ckpt, ev, tag)
    formed = c["tripod"] >= FORMATION_GATE["tripod"] and c["completion"] >= FORMATION_GATE["completion"]
    detail = f"tripod {c['tripod']:.3f} completion {c['completion']:.2f} tracking {c['tracking']:.3f}"
    if not formed:
        trip = series(tlog, "Episode_Reward/reward_clock_schedule")
        rising = len(trip) > 600 and sum(trip[-200:]) / 200 > 1.2 * sum(trip[-600:-400]) / 200
        if c["tripod"] >= 0.25 and rising:
            log(f"{tag}: trend extension (+{EXTENSION_ITERS})")
            ckpt2, _ = train(ev, tag + "_ext", EXTENSION_ITERS, resume_ckpt=ckpt)
            if ckpt2:
                c = eval_canary(ckpt2, ev, tag + "_ext")
                formed = (c["tripod"] >= FORMATION_GATE["tripod"]
                          and c["completion"] >= FORMATION_GATE["completion"])
                detail += f" | ext: tripod {c['tripod']:.3f} completion {c['completion']:.2f}"
                ckpt = ckpt2
    changelog(f"| SEARCH {tag} | {len(active)} elems: {','.join(sorted(active))} | — | — | "
              f"{'FORMED' if formed else 'NOT FORMED'}: {detail} | {Path(ckpt).parent.name}/{Path(ckpt).name} |")
    return formed, detail, ckpt


def bisect_culprits(active: list[str], st: dict) -> list[str]:
    """Group-bisection: return the minimal-ish set of formation breakers within `active`."""
    if len(active) == 1:
        return active
    mid = len(active) // 2
    a, b = active[:mid], active[mid:]
    culprits = []
    formed_a, _, _ = formation_attempt(a, st, f"bisectA{len(a)}")
    if not formed_a:
        culprits += bisect_culprits(a, st)
    formed_b, _, _ = formation_attempt(b, st, f"bisectB{len(b)}")
    if not formed_b:
        culprits += bisect_culprits(b, st)
    if formed_a and formed_b:
        # both halves fine alone -> interaction failure; defer the highest-risk single
        # element by campaign priors (goal incomes > terrain load > speed > DR).
        priors = ["goalvel_income", "terrain_recal", "terrain50", "yaw_income", "speed55",
                  "dr_push", "clearance_pack", "safety_pack", "heading", "terrain_curriculum",
                  "dr_masscom", "episode40"]
        pick = next(p for p in priors if p in active)
        changelog(f"| SEARCH interaction | halves formed separately — deferring prior-ranked {pick} | — | — | — | — |")
        culprits.append(pick)
    return culprits


def element_vars_at(el: dict, dose) -> dict:
    if dose is None:
        return dict(el["vars"])
    if "pack" in el:
        ev = {k: str(round(w * float(dose), 4)) for k, w in el["pack"].items()}
        ev.update(el.get("pack_extra", {}))
        return ev
    key = next(iter(el["vars"]))
    if len(el["vars"]) == 1:
        return {key: str(dose)}
    return dict(el["vars"])


def gate_post_formation(el: dict, tlog: Path, ckpt: str, ev: dict, tag: str) -> tuple[bool, str]:
    fails = series(tlog, "Episode_Termination/crab_failure")
    tail = fails[-20:]
    fail_tail = sum(tail) / len(tail) if tail else 1.0
    if fail_tail >= FAIL_GATE:
        return False, f"fail_tail {fail_tail:.2f}"
    c = eval_canary(ckpt, ev, tag)
    ok = (c["tripod"] >= CANARY_GATE["tripod"] and c["completion"] >= CANARY_GATE["completion"]
          and c["tracking"] >= CANARY_GATE["tracking"])
    detail = f"fail {fail_tail:.2f} canary {c['tripod']:.2f}/{c['completion']:.2f}/{c['tracking']:.2f}"
    inc = el.get("income")
    if inc:
        i = series(tlog, f"Episode_Reward/{inc}")
        f = fails
        if len(i) > 600 and len(f) > 600:
            third = len(i) // 3
            iu = [sum(abs(x) for x in i[k:k + third]) / third for k in (0, third, 2 * third)]
            fu = [sum(f[k:k + third]) / third for k in (0, third, 2 * third)]
            if iu[0] < iu[1] < iu[2] and fu[0] < fu[1] < fu[2] and fu[2] > 0.5:
                return False, detail + " | capability-gap signature"
    return ok, detail


def main() -> int:
    st = load_state()
    # ---------------- Phase S: schedule search ----------------
    while st["mode"] == "search":
        formed, detail, ckpt = formation_attempt(st["active"], st, "full")
        if formed:
            st["head"] = ckpt
            st["mode"] = "defer"
            changelog(f"| SEARCH COMPLETE | front-loaded: {','.join(sorted(st['active']))} | "
                      f"deferred: {','.join(st['deferred']) or 'none'} | — | head {Path(ckpt).parent.name} | — |")
            save_state(st)
            break
        culprits = bisect_culprits(list(st["active"]), st)
        if not culprits:
            changelog("| SEARCH HALT | full set fails but bisection found no culprit — user decision | — | — | — | — |")
            return 2
        for cname in culprits:
            if cname in st["active"]:
                st["active"].remove(cname)
                st["deferred"].append(cname)
        changelog(f"| SEARCH deferral | {','.join(culprits)} proven formation-breaking | — | — | "
                  f"remaining front-load: {','.join(sorted(st['active']))} | — |")
        save_state(st)
    # ---------------- Phase D: deferred introductions ----------------
    if st["mode"] == "defer":
        ev_base = merged_vars(st["active"])
        while st["defer_idx"] < len(st["deferred"]):
            name = st["deferred"][st["defer_idx"]]
            el = ELEMENTS[name]
            ladder = el["ladder"]
            passed_all = True
            while st["ladder_pos"] < len(ladder):
                dose = ladder[st["ladder_pos"]]
                st["run_no"] += 1
                tag = f"d{st['run_no']:03d}_{name}_r{st['ladder_pos']}"
                start = st["head"]
                if el.get("reset") and st["ladder_pos"] == 0:
                    rd = str(Path(start).with_suffix("")) + "_critic_reset.pt"
                    subprocess.run([PY, str(RESET_TOOL), "--src", start, "--dst", rd],
                                   check=True, capture_output=True)
                    start = rd
                ev = dict(ev_base)
                ev.update(element_vars_at(el, dose))
                log(f"{tag}: deferred introduction (dose={dose})")
                ckpt2, tlog = train(ev, tag, CHUNK_ITERS, resume_ckpt=start)
                if ckpt2 is None:
                    ok, detail = False, "train crashed"
                else:
                    ok, detail = gate_post_formation(el, tlog, ckpt2, ev, tag)
                changelog(f"| DEFER {name} | {dose} | rung {st['ladder_pos']} | reset {el.get('reset', False) and st['ladder_pos'] == 0} | "
                          f"{'PASS' if ok else 'FAIL'}: {detail} | {Path(ckpt2).parent.name + '/' + Path(ckpt2).name if ckpt2 else '—'} |")
                if ok:
                    st["head"] = ckpt2
                    ev_base.update(element_vars_at(el, dose))
                    st["ladder_pos"] += 1
                    save_state(st)
                else:
                    if st["second_chance"].get(name):
                        changelog(f"| HALT | {name} failed post-formation twice — user decision | — | — | — | — |")
                        save_state(st)
                        return 2
                    st["second_chance"][name] = True
                    st["deferred"].append(name)  # re-queue at the end (next milestone)
                    passed_all = False
                    break
            st["defer_idx"] += 1
            st["ladder_pos"] = 0
            save_state(st)
        st["mode"] = "consolidate"
        save_state(st)
    # ---------------- Phase G: consolidation ----------------
    ev = merged_vars(st["active"])
    for name in st["deferred"]:
        if not st["second_chance"].get(name):
            ev.update(ELEMENTS[name]["vars"])
    for i in range(3):
        st["run_no"] += 1
        tag = f"g{st['run_no']:03d}_consolidate{i}"
        ckpt2, tlog = train(ev, tag, CHUNK_ITERS, resume_ckpt=st["head"])
        if ckpt2:
            st["head"] = ckpt2
            ok, detail = gate_post_formation({"metric": "canary"}, tlog, ckpt2, ev, tag)
            changelog(f"| CONSOLIDATE {i} | — | — | — | {detail} | {Path(ckpt2).parent.name}/{Path(ckpt2).name} |")
        save_state(st)
    changelog(f"> NOTIFY: GRADUATION CANDIDATE — head {st['head']} — run graduation battery")
    return 0


if __name__ == "__main__":
    sys.exit(main())
