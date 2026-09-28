#!/usr/bin/env python3
"""STOP 2 assembly: one row per configuration combining rungs (i)-(iv) into the interim
decision table (markdown block in RESULTS.md + stop2_table.csv). Missing rungs print '—'.
Run with the isaac venv python. `--dry-run` prints without writing RESULTS.md."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
EV = REPO / "parkour/logs/rsl_rl/gait_eval/leg_mount_morphology"
CONFIGS = ["base", "B", "A10", "A15", "A20", "A10+B", "A15+B", "A20+B"]
KIN_EVAL = "gait_income_phaseout/2026-09-02_06-52-54"   # 30k head, flat trace set
HW = {"base": "none", "B": "re-hinge 3 in", "A10": "10° shims", "A15": "15° shims", "A20": "20° shims",
      "A10+B": "10° shims + re-hinge", "A15+B": "15° shims + re-hinge", "A20+B": "20° shims + re-hinge"}


def norm(v: str) -> str:
    return v.replace("p", "+") if v in ("A10pB", "A15pB", "A20pB") else v


def fmt(x, nd=1, suf=""):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{nd}f}{suf}" if isinstance(x, (int, float)) else str(x)


def rung_i() -> dict:
    out = {}
    with (HERE / "kinematic_screen.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["eval"] == KIN_EVAL:
                out[norm(r["variant"])] = {k: float(v) if k not in ("eval", "variant", "sideways_reachable") else v
                                          for k, v in r.items()}
    clr = {}
    with (HERE / "clearance_sweep.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["offset"] == "tripod_pi":
                v = norm(r["variant"]); clr[v] = min(clr.get(v, 1e9), float(r["min_clearance_mm"]))
    for v in out:
        out[v]["min_clearance_mm"] = clr.get(v)
    return out


def rung_ii() -> dict:
    p = HERE / "stop1_rows.json"
    if not p.exists():
        return {}
    rows = {r["tag"].replace("_", "+"): r for r in json.loads(p.read_text()) if not r.get("missing")}
    base = rows.get("base")
    for r in rows.values():
        r["lead_delta"] = r["lead_tip_deg"] - base["lead_tip_deg"] if base else None
        r["k_ratio"] = r["k_proxy"] / base["k_proxy"] if base and base["k_proxy"] else None
        r["sound"] = r["n_upright"] >= 19 and r["n_pen"] == 0 and r["leg_contact"] < 15.0
    return rows


def rung_iii() -> dict:
    p = HERE / "probe" / "probe_scores.json"
    return {k: v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def latest_agg(head: str, sid: str) -> dict | None:
    d = EV / head / sid / "seed001"
    if not d.exists():
        return None
    for r in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        f = r / "scenario_metrics.json"
        if f.exists():
            return json.loads(f.read_text()).get("aggregate")
    return None


def rung_iv() -> dict:
    out = {}
    for head in ("30k", "20k"):
        for cfg in CONFIGS:
            for sc in ("slow", "fwd", "step"):
                a = latest_agg(head, f"{sc}__{cfg.replace('+', 'p')}")
                if a:
                    fc = a.get("fall_classes", {}); poly = a.get("support_polygon", {})
                    out[(head, cfg, sc)] = {
                        "falls": a["termination_reasons"].get("fall", 0), "n": a["n_episodes"],
                        "completion": a["schedule_completion_rate"],
                        "pitch_fwd_share": (fc.get("pitch_fwd", 0) / max(1, sum(v for k, v in fc.items() if k != "none")))
                        if any(k != "none" for k in fc) else None,
                        "prefall_tip_p25": (poly.get("prefall_tip_angle_fwd_deg") or {}).get("p25"),
                        "walk_tip_p50": (poly.get("walking_tip_angle_fwd_deg") or {}).get("median"),
                    }
    return out


def main() -> int:
    ki, kii, kiii, kiv = rung_i(), rung_ii(), rung_iii(), rung_iv()
    hdr = ["config", "hardware", "splay", "axis in", "rung i walk tip p50/p10", "rung i lead tip p50", "rung i prefall frac neg",
           "rung i min clearance mm (tripod)", "rung i lateral reach loss mm", "rung ii sound", "rung ii standing lead tip Δ",
           "rung ii K proxy", "rung iii survivors/8 @0.3 / @0.5", "rung iii ttf ratio @0.5", "rung iii pitch-fwd share @0.5",
           "rung iii hip/femur N", "rung iv 30k falls fwd / step (Δ vs base)", "rung iv 20k falls fwd / step (Δ vs base)",
           "rung iv 30k step pitch-fwd share (of falls)", "rung iv 30k step prefall tip p25"]
    rows = []
    for cfg in CONFIGS:
        i, ii = ki.get(cfg, {}), kii.get(cfg, {})
        p3, p5 = kiii.get(f"{cfg}@0.3"), kiii.get(f"{cfg}@0.5")
        b5 = kiii.get("base@0.5")
        def iv(head, sc, key):
            e = kiv.get((head, cfg, sc)); return None if not e else e.get(key)
        def iv_delta(head, sc):
            v = iv(head, sc, "falls"); b = kiv.get((head, "base", sc), {}).get("falls")
            return "—" if v is None else (f"{v}" if b is None or cfg == "base" else f"{v} ({v - b:+d})")
        rows.append([
            cfg, HW[cfg], fmt(i.get("splay_deg"), 0, "°"), fmt(i.get("outer_axis_in"), 1),
            f"{fmt(i.get('walk_tip_fwd_p50'))} / {fmt(i.get('walk_tip_fwd_p10'))}", fmt(i.get("walk_lead_tip_p50")),
            fmt(i.get("prefall_frac_neg"), 2), fmt(i.get("min_clearance_mm"), 0), fmt(i.get("lateral_reach_loss_mm"), 0),
            ("yes" if ii.get("sound") else ("NO" if ii else "—")), fmt(ii.get("lead_delta"), 1, "°"), fmt(ii.get("k_ratio"), 2, "×"),
            f"{p3['survivors'] if p3 else '—'} / {p5['survivors'] if p5 else '—'}",
            fmt(p5["ttf_median_s"] / b5["ttf_median_s"] if p5 and b5 and b5["ttf_median_s"] else None, 2, "×"),
            fmt(p5.get("pitch_fwd_share") if p5 else None, 2), fmt(p5.get("leg_contact_hipfemur_max_N") if p5 else None, 0),
            f"{iv_delta('30k', 'fwd')} / {iv_delta('30k', 'step')}", f"{iv_delta('20k', 'fwd')} / {iv_delta('20k', 'step')}",
            fmt(iv("30k", "step", "pitch_fwd_share"), 2), fmt(iv("30k", "step", "prefall_tip_p25"), 1, "°"),
        ])
    lines = ["## STOP 2 — rungs i–iv combined (interim decision table; rung v pending the user's budget confirmation)",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["- rung i = same joint angles as the recorded 30k policy, mounts moved (lower bound); rung ii = 20 settles + sweep; "
              "rung iii = open-loop scripted tripod (8 envs, in-session base); rung iv = 30k/20k heads replayed on the variant plant (one-sided; unchanged falls = policy mismatch, not a null result)",
              "- the tibia contact channel reads tibia–ground load in this plant (colliders reach the floor), so hip/femur is the interference signal; tibia–tibia contact is covered kinematically by the rung i clearance sweep only"]
    block = "\n".join(lines)
    print(block)
    with (HERE / "stop2_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    if "--dry-run" not in sys.argv:
        with (HERE / "RESULTS.md").open("a") as fh:
            fh.write("\n" + block + "\n>>> ENTRY stop 2\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
