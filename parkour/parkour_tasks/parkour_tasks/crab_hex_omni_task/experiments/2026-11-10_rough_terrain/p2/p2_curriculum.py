#!/usr/bin/env python3
"""Rough-terrain P2 (p2/P2.md Design > Distill, "Curriculum read"): the distill's curriculum read -- CPU only, never
Kit, never a kill.

The +10-iteration checks cannot see the curriculum (levels are ~1 there), so at ~+300 and +1000 iterations (or any
iteration) this reads, from the distill's tfevents (default) or its train log:

    Metrics/base_parkour/site_level/slopes       (log: "Metrics/base_parkour/site_level/slopes: x")
    Metrics/base_parkour/site_level/ledge_down   (log: "Metrics/base_parkour/site_level/ledge_down: x")
    Metrics/base_parkour/terrain_levels          (log: "Metrics/base_parkour/terrain_levels: x")
    Loss_depth/depth_actor_loss                  (log: "depth_actor_loss: x"; a teacher run has none)

prints them, and appends one JSON line to results/p2/curriculum.jsonl (recorded either way). Iterations count from the
run's first logged iteration (the distill resumes c1's model_29500, so +300 = 29800, +1000 = 30500).

Statistic per tag (PLAN_P2.md as amended by audit pass 2: "21-iteration window means"): the mean of the 21 logged
iterations centred on the read, ``at - 10 .. at + 10`` (``win21``; None unless all 21 are logged), plus the logged
value at the read iteration (``point``, for the record only). The loss reference is the same 21-iteration window mean
centred on +50 (``loss_plus50`` = mean over +40..+60). This statistic reproduces the plan's quoted field-distill
ratios: RSD_ST_s19 0.986 -> 0.634 (0.64), T4F19_st_a2 0.68, T4R18_st 0.66.

Pre-registered trigger, evaluated at reads >= +1000 only:
  * either kind's level win21 below 3.0 (slopes, ledge_down; b2 = levels ~2.7-5.4; RT teacher 6.4 / 4.7), or
  * the depth actor loss win21 above 0.80 x its +50 window mean
-> one line "P2_CURRICULUM TRIGGER ..." -> notify and ask (extend, stop for the blended-mix fallback, or continue). The
run continues while the question is open: this script never signals anything. A missing loss (a teacher run) is
reported as n/a and only the level clause is evaluated; an incomplete window is noted, never evaluated. A read at
iteration N needs telemetry up to N + 10 (exit 2 before that); --latest reads at the last logged iteration - 10.

    PY=/home/nickmagus/krabby/isaac_venv/bin/python
    $PY -I -B p2_curriculum.py read [--name RT_ST_s19] (--rel N | --at N | --latest) [--source tb|log|both]
                                    [--run-dir DIR] [--log LOG] [--first N] [--jsonl PATH | --no-record] [--json]
    $PY -I -B p2_curriculum.py selftest     # RT_s19's own tfevents + log (no depth loss) and RSD_ST_s19's (no
                                            # site_level); writes nothing
Exit: 0 read done (trigger or not: grep the TRIGGER line), 2 the telemetry has not reached the read iteration yet,
5 usage / no telemetry.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import p2_distill_check as C  # noqa: E402  (rsd_launch / rsd_distill_check come with it)

RL = C.RL
T_SLOPES = "Metrics/base_parkour/site_level/slopes"
T_LEDGE = "Metrics/base_parkour/site_level/ledge_down"
T_LEVELS = "Metrics/base_parkour/terrain_levels"
T_LOSS = "Loss_depth/depth_actor_loss"
TAGS = {"slopes": T_SLOPES, "ledge_down": T_LEDGE, "terrain_levels": T_LEVELS, "depth_actor_loss": T_LOSS}
#: the train log's line for each tag: the metrics print their tag; the student runner prints the loss without its
#: "Loss_depth/" group, as "depth_actor_loss: x" in the first block and "Mean depth_actor_loss loss: x" after it
#: (RSD_ST_s19.log lines 334 / 477)
LOG_RES = {t: re.compile(r"^[ \t]*%s:[ \t]*(\S+)[ \t]*$" % re.escape(t), re.M) for t in (T_SLOPES, T_LEDGE, T_LEVELS)}
LOG_RES[T_LOSS] = re.compile(r"^[ \t]*(?:Mean )?depth_actor_loss(?: loss)?:[ \t]*(\S+)[ \t]*$", re.M)
KINDS = ("slopes", "ledge_down")
WIN_HALF = 10  # a 21-iteration window centred on the read: at-10 .. at+10
LOSS_REF_REL = 50
TRIGGER_FROM_REL = 1000
LEVEL_MIN = 3.0
LOSS_FRAC = 0.80
JSONL = C.RESULTS_P2 / "curriculum.jsonl"
RC_OK, RC_WAIT, RC_USAGE = 0, 2, 5
PFX = "P2 CURRICULUM"

RSD_ST_LOG = RL.MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr" / "distill" / "RSD_ST_s19.log"
RSD_ST_RUN = RL.MAIN / "parkour" / "logs" / "rsl_rl" / "crab_hex_omni_student" / "2026-11-08_02-57-24_RSD_ST_s19"


# ============================================================================== series
def series_tb(run_dir: Path) -> dict:
    """``{tag: [(step, value)]}`` for :data:`TAGS` from the run dir's tfevents (absent tags are left out)."""
    return C.live_watch().read_series(run_dir, tuple(TAGS.values()))


def series_log(log: Path) -> dict:
    """The same series from the train log's iteration blocks (rsl_rl's title lines only: ``C.title_blocks``)."""
    with open(log, "rb") as fh:
        text = fh.read().decode(errors="replace")
    out = {t: [] for t in TAGS.values()}
    blocks = C.title_blocks(text)
    for it, _tot, blk in blocks[:-1] if len(blocks) > 1 else []:  # the last block of a live log may be partial
        for tag, rx in LOG_RES.items():
            m = rx.search(blk)
            if m:
                try:
                    out[tag].append((it, float(m.group(1))))
                except ValueError:
                    out[tag].append((it, float("nan")))
    return {t: s for t, s in out.items() if s}


def first_step(ser: dict) -> int | None:
    steps = [s for v in ser.values() for s, _ in v]
    return min(steps) if steps else None


def last_step(ser: dict) -> int | None:
    steps = [s for v in ser.values() for s, _ in v]
    return max(steps) if steps else None


def at_read(pts, at: int, half: int = WIN_HALF) -> dict | None:
    """``win21`` = the mean over the logged iterations ``at - half .. at + half`` (None unless all ``2 half + 1`` are
    logged), ``point`` = the value logged at ``at`` (else the last one before it; record only)."""
    upto = [(s, v) for s, v in (pts or []) if s <= at]
    if not upto:
        return None
    win = {s: v for s, v in (pts or []) if at - half <= s <= at + half}
    full = len(win) == 2 * half + 1
    return {"point": upto[-1][1], "point_step": upto[-1][0],
            "win21": (sum(win.values()) / len(win)) if full else None, "n": len(win),
            "window": [at - half, at + half]}


# ============================================================================== the read
def read(ser: dict, *, at: int, first: int, source: str) -> dict:
    rel = at - first
    vals = {name: at_read(ser.get(tag), at) for name, tag in TAGS.items()}
    loss = ser.get(T_LOSS)
    ref = at_read(loss, first + LOSS_REF_REL) if loss else None
    loss_ref = ref["win21"] if ref else None
    lval = vals["depth_actor_loss"]
    rec = {"source": source, "first": first, "at": at, "rel": rel, "values": vals,
           "loss_first": loss[0][1] if loss else None, "loss_plus50": loss_ref,
           "loss_ratio": (lval["win21"] / loss_ref) if (loss_ref and lval and lval["win21"] is not None) else None,
           "tags": TAGS, "rule": {"from_rel": TRIGGER_FROM_REL, "statistic": "win21 = mean over at-10..at+10 (21 its)",
                                  "level_fires_if_win21_below": LEVEL_MIN,
                                  "loss_fires_if_win21_above_frac_of_plus50_win21": LOSS_FRAC,
                                  "loss_ref": "win21 centred on +50 (+40..+60)", "win_half": WIN_HALF}}
    trig = {"evaluated": rel >= TRIGGER_FROM_REL, "fired": False, "reasons": [], "notes": []}
    if trig["evaluated"]:
        for k in KINDS:
            v = vals[k]
            if v is None:
                trig["notes"].append(f"{k}: no {TAGS[k]} telemetry (not evaluated)")
            elif v["win21"] is None:
                trig["notes"].append(f"{k}: incomplete window {v['window']} n={v['n']} (not evaluated)")
            elif not v["win21"] >= LEVEL_MIN:  # NaN-safe: a NaN level fires
                trig["reasons"].append(f"{k} level win21 {v['win21']:.3f} below {LEVEL_MIN}")
        lv = vals["depth_actor_loss"]
        if lv is None or ref is None:
            trig["notes"].append("depth_actor_loss n/a (no Loss_depth/depth_actor_loss, e.g. a teacher run): loss "
                                 "clause not evaluated")
        elif lv["win21"] is None or loss_ref is None:
            trig["notes"].append(f"depth_actor_loss incomplete window (read n={lv['n']}, +50 ref n={ref['n']}): loss "
                                 "clause not evaluated")
        elif not lv["win21"] <= LOSS_FRAC * loss_ref:  # NaN-safe: a NaN loss fires
            trig["reasons"].append(f"depth_actor_loss win21 {lv['win21']:.4f} above {LOSS_FRAC} x +50 win21 "
                                   f"{loss_ref:.4f} = {LOSS_FRAC * loss_ref:.4f} (ratio {rec['loss_ratio']})")
        trig["fired"] = bool(trig["reasons"])
    rec["trigger"] = trig
    return rec


def fmt(v) -> str:
    if v is None:
        return "n/a"
    r = v["win21"]
    return f"win21 {r:.4f} (point {v['point']:.4f}@{v['point_step']})" if r is not None and math.isfinite(r) else \
        f"win21 n/a n={v['n']} (point {v['point']:.4f}@{v['point_step']})"


def emit(name: str, rec: dict) -> None:
    vals = rec["values"]
    print(f"{PFX} {name} at={rec['at']} (+{rec['rel']}) source={rec['source']} "
          + " ".join(f"{k}={fmt(vals[k])}" for k in TAGS)
          + f" loss(+{LOSS_REF_REL} win21)={'n/a' if rec['loss_plus50'] is None else round(rec['loss_plus50'], 4)}"
          + f" loss_ratio={'n/a' if rec['loss_ratio'] is None else round(rec['loss_ratio'], 4)}")
    t = rec["trigger"]
    if not t["evaluated"]:
        print(f"{PFX} {name} trigger: not evaluated (before +{TRIGGER_FROM_REL})")
    elif t["fired"]:
        print(f"P2_CURRICULUM TRIGGER {name} at={rec['at']} (+{rec['rel']}) reasons={t['reasons']} notes={t['notes']} "
              f"-> notify and ask (extend / stop for the blended-mix fallback / continue); the distill keeps running")
    else:
        print(f"{PFX} {name} trigger: none at +{rec['rel']} (levels win21 >= {LEVEL_MIN}, loss win21 <= {LOSS_FRAC} x +50 win21) "
              f"notes={t['notes']}")


def resolve(a) -> tuple[dict, str, dict]:
    run_dir = Path(a.run_dir) if a.run_dir else RL.find_run_dir(a.name, C.EXPERIMENT)
    log = Path(a.log) if a.log else C.DISTILL_OUT / f"{a.name}.log"
    src = a.source
    if src == "tb" and (run_dir is None or not Path(run_dir).is_dir()):
        src = "log"
    where = {"run_dir": str(run_dir) if run_dir else None, "log": str(log)}
    if src == "tb":
        return series_tb(Path(run_dir)), "tb", where
    if not log.is_file():
        return (series_tb(Path(run_dir)), "tb (no log to cross-check)", where) if src == "both" else ({}, src, where)
    if src == "log":
        return series_log(log), "log", where
    tb, lg = series_tb(Path(run_dir)), series_log(log)
    where["agree"] = agree(tb, lg)
    return tb, "tb (log cross-checked)", where


def agree(tb: dict, lg: dict, tol: float = 5.01e-5) -> dict:
    """tfevents vs the log's 4-decimal prints, tag by tag over the log's steps."""
    out = {}
    for tag in TAGS.values():
        a, b = dict(tb.get(tag) or []), dict(lg.get(tag) or [])
        common = sorted(set(a) & set(b))
        bad = [s for s in common if abs(a[s] - b[s]) > tol]
        out[tag] = {"tb": len(a), "log": len(b), "common": len(common), "max_abs": max(
            (abs(a[s] - b[s]) for s in common), default=None), "bad": bad[:5]}
    return out


def code_state() -> dict:
    try:
        cs = RL.code_state()
        return {"code_sha": cs["code_sha"], "eval_sha": cs["eval_sha"], "dirty": cs["dirty"]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def cmd_read(a) -> int:
    ser, source, where = resolve(a)
    if not ser:
        print(f"{PFX} {a.name} no telemetry ({where})", file=sys.stderr)
        return RC_USAGE
    first = a.first if a.first is not None else first_step(ser)
    last = last_step(ser)
    at = (last - WIN_HALF) if a.latest else (a.at if a.at is not None else first + a.rel)
    if last < at + WIN_HALF:
        print(f"{PFX} {a.name} WAIT telemetry at {last} (+{last - first}) < the read iteration {at} (+{at - first}) "
              f"+ {WIN_HALF} (the 21-iteration window)")
        return RC_WAIT
    rec = read(ser, at=at, first=first, source=source)
    rec.update({"name": a.name, "created": RL.now_iso(), **where, "build": code_state(), "schema": "p2_curriculum_v1"})
    emit(a.name, rec)
    if "agree" in where:
        print(f"{PFX} {a.name} tb vs log: " + "; ".join(f"{t.rsplit('/', 1)[-1]} common {v['common']} max_abs "
                                                         f"{v['max_abs']}" for t, v in where["agree"].items()))
    if not a.no_record:
        path = Path(a.jsonl) if a.jsonl else JSONL
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"{PFX} {a.name} recorded -> {path}")
    if a.json:
        print(json.dumps(rec, indent=1, default=str))
    return RC_OK


# ============================================================================== selftest
def cmd_selftest(a) -> int:
    """RT_s19 (teacher: site_level present, no depth loss) and RSD_ST_s19 (field student: depth loss, no site_level),
    each from tfevents and from the log, at +300 and +1000; nothing is written."""
    ok_all = True
    cases = (("RT_s19", C.C1_RUN_DIR, C.RT_S19_LOG, True, False), ("RSD_ST_s19", RSD_ST_RUN, RSD_ST_LOG, False, True))
    for name, run_dir, log, has_site, has_loss in cases:
        tb, lg = series_tb(run_dir), series_log(log)
        ag = agree(tb, lg)
        first = first_step(tb)
        print(f"{PFX} SELFTEST {name}: tb tags {sorted(tb)}; first {first} (log {first_step(lg)})")
        for rel in (300, 1000):
            for src, ser in (("tb", tb), ("log", lg)):
                rec = read(ser, at=first + rel, first=first, source=src)
                emit(f"SELFTEST:{name}", rec)
                site_ok = all((rec["values"][k] is not None) == has_site for k in KINDS)
                loss_ok = (rec["values"]["depth_actor_loss"] is not None) == has_loss and \
                    (rec["loss_plus50"] is not None) == has_loss
                trig_ok = rec["trigger"]["evaluated"] == (rel >= TRIGGER_FROM_REL)
                exp_ok = True
                if rel == 1000:  # the plan's quoted numbers; neither healthy run may trigger
                    exp_ok = not rec["trigger"]["fired"]
                    if name == "RSD_ST_s19":  # plan: 0.986 -> 0.634, ratio 0.64
                        r = rec["loss_ratio"]
                        exp_ok = exp_ok and r is not None and abs(r - 0.643) <= 0.005
                    if name == "RT_s19":  # plan: the RT teacher read 6.4 / 4.7 at +1000
                        sl, ld = rec["values"]["slopes"]["win21"], rec["values"]["ledge_down"]["win21"]
                        exp_ok = exp_ok and sl is not None and ld is not None and abs(sl - 6.4) <= 0.1 \
                            and abs(ld - 4.7) <= 0.1
                if not (site_ok and loss_ok and trig_ok and exp_ok):
                    ok_all = False
                    print(f"{PFX} SELFTEST {name} +{rel} {src} FAIL site_ok {site_ok} loss_ok {loss_ok} "
                          f"trig_ok {trig_ok} expected-values {exp_ok}")
        worst = max((v["max_abs"] or 0.0) for v in ag.values())
        agree_ok = all(not v["bad"] for v in ag.values()) and all(
            v["common"] == v["tb"] - (1 if v["tb"] > v["log"] else 0) or v["tb"] == 0 for v in ag.values())
        print(f"{PFX} SELFTEST {name} tb vs log: max |diff| {worst:.2e} over "
              + ", ".join(f"{t.rsplit('/', 1)[-1]} {v['common']}/{v['tb']}" for t, v in ag.items())
              + f" -> {'PASS' if agree_ok else 'FAIL'}")
        ok_all = ok_all and agree_ok
    print(f"{PFX} SELFTEST RESULT {'PASS' if ok_all else 'FAIL'} (nothing written)")
    return RC_OK if ok_all else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--name", default=C.NAME)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--rel", type=int, help="iterations after the run's first logged iteration (300, 1000, ...)")
    g.add_argument("--at", type=int, help="an absolute iteration")
    g.add_argument("--latest", action="store_true", help="the last logged iteration")
    r.add_argument("--source", choices=("tb", "log", "both"), default="tb")
    r.add_argument("--run-dir", default=None)
    r.add_argument("--log", default=None)
    r.add_argument("--first", type=int, default=None, help="override the first iteration (default: the series')")
    r.add_argument("--jsonl", default=None, help=f"default {JSONL}")
    r.add_argument("--no-record", action="store_true")
    r.add_argument("--json", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"read": cmd_read, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
