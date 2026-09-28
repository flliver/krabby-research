#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A15+B lineage retrain (user decision 2026-09-06: "move forward with A15+B" -> full lineage retrain).

From scratch on the A15+B plant (15 deg splay shims + yaw axes at 2.5 in), seed 3, six 5k windows:
  w0  formation: rung-v formation config + STAND_FRAC 0.2 + 40-s episodes / 10-s holds
  w1  + window-1 elements (recal geometry -> recal2b2w, curriculum, promotion x 20/40, DR, yaw/edge/
      stumble/collision) + ramps apex 1.0->0.5, airtime 0.8->0.4, stride 0.5->0.25
  w2  + window-2 elements (clearance terms, heading band, goal-vel) + ramps apex/airtime/stride -> eps
  w3  clock 1.0->0.5 | w4 clock 0.5->0.2 | w5 clock 0.2->eps      (the phase-out schedule of record)
RSI bank P0-null throughout (confirmation-replay precedent; golden-harvested pg banks not used on the
new plant). After each window: morph-manifest slow/step evals + the recal2b2w obstacle eval on the
A15+B plant + exposure / per-tile hazard telemetry. Then reference evals of the golden 30k head of
record on the golden plant (same three scenarios) -> await_user (push-notify) -> --seed2 replay.
Reuses run_morph_exposure.py (stacks, train_retry, evals, report lines) and run_exposure.py
(REPLAY_SCHEDULE, phaseout_spec, SCHEDULE). Stdlib only; resumable via state.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
MORPHX = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-04_1105_morph_x_exposure"
STATE = HERE / "state.json"
REPORT = HERE / "REPORT.md"
CHANGELOG = HERE / "CHANGELOG.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mx = _load(MORPHX / "run_morph_exposure.py", "run_morph_exposure")
rx = mx.rx  # run_exposure helpers (REPLAY_SCHEDULE, phaseout_spec, SCHEDULE, series, fmt, ...)
mx.LOGS = HERE / "logs"          # keep this campaign's logs in its own dir
mx.EVAL_ROOT = REPO / "parkour/logs/rsl_rl/gait_eval/a15b_lineage"
mx.REPORT = REPORT
mx.CHANGELOG = CHANGELOG

CFG = "A15+B"
SEED = "3"
CONFIRM_SEED = "2"
WINDOWS = 6
GOLDEN_30K = str(REPO / "parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt")
GOLDEN_20K = str(REPO / "parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt")


def window_stack(w: int, cfg: str = CFG) -> dict:
    """Env stack of lineage window w (0 = formation) on the plant: validated levers + schedule."""
    if w == 0:
        return mx.p2_stage1_stack(cfg)
    ev = dict(mx.FORMATION)
    for r in sorted(rx.SCHEDULE):
        if r <= w:
            ev.update(rx.SCHEDULE[r])
    ev.update(mx.STAND)
    ev.update(mx.LONG)
    ev["KRABBY_FLAT_TERRAIN_GEOM"] = "recal2b2w"
    up, down = (float(x) for x in rx.SCHEDULE[1]["KRABBY_TERRAIN_PROMOTE"].split(":"))
    ev["KRABBY_TERRAIN_PROMOTE"] = mx.promote_fracs_for_horizon(up, down, mx.HORIZON_S)
    weights, ramps = rx.REPLAY_SCHEDULE[w]
    ev.update(weights)
    if ramps:
        ev["KRABBY_PHASEOUT"] = rx.phaseout_spec(ramps)
    ev["KRABBY_RSI_BANK"] = mx.FORMATION["KRABBY_RSI_BANK"]   # P0-null throughout
    return mx.with_plant(ev, cfg)


def fmt2(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.2f}"


def log(msg: str) -> None:
    print(f"[a15b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    return {"phase": "retrain", "run_no": 0, "windows": {}, "reference": {}, "seed2": {}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def run_window(st: dict, w: int, seed: str, key: str) -> dict:
    prev = st[key].get(str(w - 1), {}).get("ckpt") if w > 0 else None
    st["run_no"] += 1
    tag = f"{key}_w{w}_{st['run_no']:03d}_A15pB" + (f"_seed{seed}" if seed != SEED else "")
    ev = window_stack(w)
    log(f"{tag}: window {w} ({5 * w}k -> {5 * (w + 1)}k) seed {seed} | ramps {ev.get('KRABBY_PHASEOUT', 'none')}")
    ckpt, status, lp = mx.train_retry(ev, tag, mx.STAGE_ITERS, prev, seed, soundness=(w == 0), live_check=None)
    rec = {"window": w, "tag": tag, "ckpt": ckpt, "status": status, "log": str(lp), "env": ev,
           "smoke": mx.smoke_fields(lp) if lp.exists() else {},
           "exposure": rx.exposure_from_log(lp) if lp.exists() else {}, "evals": {}}
    if ckpt:
        rec["evals"]["slow"] = mx.eval_morph(ckpt, CFG, "slow", tag)
        rec["evals"]["step"] = mx.eval_morph(ckpt, CFG, "step", tag)
        rec["evals"]["obst"] = mx.eval_obst(ckpt, CFG, ev, tag)
    return rec


def reference_evals(st: dict) -> None:
    """Golden heads of record on the golden plant, same three scenarios (side-by-side reference)."""
    ref = st["reference"]
    for name, ckpt, weights in (("golden_30k", GOLDEN_30K, rx.W_30K), ("golden_20k", GOLDEN_20K, rx.W_20K)):
        if name in ref:
            continue
        ev = rx.lineage_stack(5 if name == "golden_30k" else 3, weights)
        ref[name] = {"ckpt": ckpt, "evals": {
            "slow": mx.eval_morph(ckpt, "base", "slow", name),
            "step": mx.eval_morph(ckpt, "base", "step", name),
            "obst": mx.eval_obst(ckpt, "base", ev, name)}}
        save_state(st)


def window_table(st: dict, key: str) -> list[str]:
    lines = ["| window | iters | canary tripod / compl / falls | step falls / compl | obst recal2b2w (falls) | hazard flat / obst (/1k) | reach_obst / cov3 / cov6 | level |",
             "|---|---|---|---|---|---|---|---|"]
    for w in range(WINDOWS):
        r = st[key].get(str(w))
        if not r:
            continue
        ev = r.get("evals") or {}; s, p, o = ev.get("slow") or {}, ev.get("step") or {}, ev.get("obst") or {}; ex = r.get("exposure") or {}
        lines.append(f"| w{w} | {5 * (w + 1)}k | {rx.fmt(s.get('tripod'))} / {rx.fmt(s.get('completion'))} / {s.get('falls', '—')} | "
                     f"{p.get('falls', '—')} / {rx.fmt(p.get('completion'))} | {rx.fmt(o.get('completion'))} ({o.get('falls', '—')}) | "
                     f"{rx.fmt(ex.get('crab_failure_hazard_flat'))} / {rx.fmt(ex.get('crab_failure_hazard_obst'))} | "
                     f"{rx.fmt(ex.get('reach_obst_frac'))} / {rx.fmt(ex.get('obst_coverage_3'))} / {rx.fmt(ex.get('obst_coverage_6'))} | {fmt2(ex.get('terrain_levels'))} |")
    for name, r in st.get("reference", {}).items():
        ev = r.get("evals") or {}; s, p, o = ev.get("slow") or {}, ev.get("step") or {}, ev.get("obst") or {}
        lines.append(f"| {name} (golden plant) | — | {rx.fmt(s.get('tripod'))} / {rx.fmt(s.get('completion'))} / {s.get('falls', '—')} | "
                     f"{p.get('falls', '—')} / {rx.fmt(p.get('completion'))} | {rx.fmt(o.get('completion'))} ({o.get('falls', '—')}) | — | — | — |")
    return lines


def report_window(st: dict, w: int, seed: str, key: str) -> None:
    rec = st[key][str(w)]
    weights, ramps = (None, []) if w == 0 else rx.REPLAY_SCHEDULE[w]
    report("\n".join([f"## {key.upper()} — window {w} ({5 * w}k -> {5 * (w + 1)}k, seed {seed}) — {rec['status']}",
                      f"- stack: {'formation (P2 stage-1)' if w == 0 else 'elements <= ' + str(min(w, 2)) + ' + recal2b2w + promotion x 20/40'} | ramps {ramps or 'none'}"]
                     + mx.stage_lines(rec)
                     + ([f"- terrain level {fmt2((rec.get('exposure') or {}).get('terrain_levels'))} (curriculum on; lineage band 3-7)"] if w > 0 else [])),
           f"{key} w{w} {rec['status']}")
    changelog(f"- {time.strftime('%Y-%m-%d %H:%M')} {key} w{w} seed {seed}: {rec['status']}; canary "
              f"{rx.fmt(((rec['evals'].get('slow') or {}).get('completion')))} step falls {((rec['evals'].get('step') or {}).get('falls'))} "
              f"obst {rx.fmt(((rec['evals'].get('obst') or {}).get('completion')))}")


def _reported(key: str, w: int) -> bool:
    return REPORT.exists() and f">>> ENTRY {key} w{w} " in REPORT.read_text()


def phase_retrain(st: dict, seed: str = SEED, key: str = "windows") -> None:
    for w in range(WINDOWS):
        rec = st[key].get(str(w))
        if rec and rec.get("ckpt"):
            if not _reported(key, w):          # resume after a crash between save and report
                report_window(st, w, seed, key)
            continue
        if w > 0 and not st[key].get(str(w - 1), {}).get("ckpt"):
            log(f"window {w - 1} has no checkpoint — cannot continue")
            sys.exit(2)
        rec = run_window(st, w, seed, key)
        st[key][str(w)] = rec
        save_state(st)
        report_window(st, w, seed, key)
        if not rec.get("ckpt"):
            notify(f"{key} window {w} {rec['status']} — lineage halted")
            st["phase"] = "halt"
            save_state(st)
            sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed2", action="store_true")
    args = ap.parse_args()
    st = load_state()
    if args.seed2 and st["phase"] == "await_user":
        st["phase"] = "seed2"
    save_state(st)
    if st["phase"] == "retrain":
        phase_retrain(st)
        reference_evals(st)
        report("\n".join(["## A15+B LINEAGE (seed 3) — per-window table with golden references"] + window_table(st, "windows")), "lineage table seed3")
        st["phase"] = "await_user"
        save_state(st)
        w5 = st["windows"]["5"]["evals"]
        notify(f"A15+B lineage seed 3 complete at 30k — canary {rx.fmt((w5.get('slow') or {}).get('completion'))}, obstacle eval "
               f"{rx.fmt((w5.get('obst') or {}).get('completion'))}; PAUSED (relaunch with --seed2 for the replay; bake decision is the user's)")
    if st["phase"] == "await_user":
        log("await_user: seed-3 lineage done — relaunch with --seed2")
        return 0
    if st["phase"] == "seed2":
        phase_retrain(st, seed=CONFIRM_SEED, key="seed2")
        report("\n".join(["## A15+B LINEAGE (seed 2 replay) — per-window table"] + window_table(st, "seed2")), "lineage table seed2")
        st["phase"] = "done"
        save_state(st)
        notify("A15+B lineage seed-2 replay complete — bake decision is the user's")
    if st["phase"] == "done":
        log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
