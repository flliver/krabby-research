#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 pipeline gate reader (CPU): Gate 0 INVALID checks, then the stage's DECISIVE lines from thresholds.json,
everything else WATCH. Writes ``<run dir>/gates/<stage>.json``; never boots Kit, trains or evaluates.

VERDICTS (exit codes, DESIGN 2.1): 0 PASS, 2 PASS + WATCH flags (notify), 3 HALT (a decisive line fails), 4 INVALID
(Gate 0: rerun the rows once, then halt), 5 GUARD (the run's pin is unreadable / the lock is missing).

GATE 0 (PLAN Gates -> Gate 0) -- a stamp is INVALID unless all of these hold:
* complete: ``scenario_metrics.json`` + ``run_meta.json`` + ``summary.md`` + ``raw/`` + a completion marker
  (``complete.json`` / ``p2_complete.json`` / ``tp_row.json``);
* the pin: a row stamp's ``code_state.json`` code_sha / eval_sha equal the lock's and ``dirty`` is ``False``; the
  driver's own stage-eval stamp is pinned through the driver's reference record (its code_sha / eval_sha);
* the carry env: ``run_meta.env_vars`` equals the composed stage env gen wrote (key for key) AND the named carry
  checks of the spec (``carry_env``: CLOCK_HIDE per stage -- 0 through c6, 1 from c6x --, STD_PARAM soft, INIT_STD
  0.8, ellipse 0.35:0.35:0.25, facing clip 0.6, RSI off, STAND_OBS 0, plant; COMMAND_FRAME world or absent);
* the checkpoint: ``run_meta.checkpoint_sha256`` equals the stage record's sha256;
* the payload: a level row's ``run_meta.payload`` reads EVERY env at exactly L kg, CoM (0, 0, 0.309), zero_frac 0;
  a 0-kg / REFS / stand row carries no payload (no ``run_meta.payload``, no ``KRABBY_PAYLOAD_*`` in its env);
* ``trained_categories``: a row whose family is marked ``trained_categories`` carries the stage's list EXACTLY (a
  stamp without it is INVALID, so null never silently drops or adds a floor);
* a body row records ``command_adapter: body_hold``.

DECISIVE LINES (thresholds.json, the ``recommended`` / ``both`` value sets minus the ``watch`` ids):
* floors: translate survival and ratio per TRAINED translate category (the stamp's list) per floor env seed, and
  the side22 line (FAST/FAST at p2a / p3a / student, not-SLOW at s1-c6x); p3b / p3c read p3a's floor values, and a
  miss there is a measured CEILING (``gate.ceiling_on_fail``), never a HALT;
* p2a: field_hard completion pooled es1-4 (n = 400) >= the three-seed line;
* p3a: stable at 90 (``payload_read.py``) and the paired 0-kg check against the run's own p2a head
  (``thresholds.zero_kg_*``, the same functions the threshold self-check runs);
* live collapse: the stage's live-watch record (a kill already HALTed the stage).
Every decisive read carries its chance-HALT rate (thresholds.json), so a HALT record can quote it.

    gate_read.py --spec PL/pipeline.yaml --run R18 --stage p2a [--thresholds FILE] [--out FILE]
    gate_read.py --spec PL/pipeline.yaml --lines-only --stage p2a --thresholds T.json --stamp suite@1=DIR --stamp field_hard@1=DIR ... [--trained fwd,side,...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _lib():
    import importlib.util  # noqa: PLC0415

    name = "tp_lib"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(HERE / "tp_lib.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = _lib()

PASS, WATCH, HALT, INVALID, GUARD = "PASS", "WATCH", "HALT", "INVALID", "GUARD"
CEILING = "CEILING"
EXIT = {PASS: 0, WATCH: 2, HALT: 3, CEILING: 2, INVALID: 4, GUARD: 5}
TRANSLATE = ("fwd", "side", "back", "diag")
COMPLETE_MARKERS = ("complete.json", "p2_complete.json", "tp_row.json")
PAYLOAD_TOL = 1e-6


# ============================================================================== thresholds view
class Thresholds:
    """The decisive lines of one thresholds.json, by stage (the recommended set; ``both`` lines included)."""

    def __init__(self, doc: dict, path: Path | None = None):
        self.doc, self.path = doc, path
        self.watch_ids = set((doc.get("recommended") or {}).get("watch") or [])
        self.lines = [x for x in doc["lines"] if x.get("value_set") in ("recommended", "both")]
        self.standard = [x for x in doc["lines"] if x.get("value_set") == "standard"]
        self.version = doc.get("version")

    @classmethod
    def load(cls, path: Path) -> "Thresholds":
        return cls(json.loads(Path(path).read_text()), Path(path))

    def by_id(self, lid: str, *, standard: bool = False) -> dict | None:
        for x in (self.standard if standard else self.lines):
            if x["id"] == lid:
                return x
        return None

    def floor_values(self, stage: str, *, standard: bool = False) -> dict:
        """{survival, ratio, side22 ('fastfast' | 'notslow'), env_seeds} of a stage's floor lines."""
        pool = self.standard if standard else self.lines
        out = {"survival": None, "ratio": None, "side22": None, "env_seeds": set(), "ids": []}
        for x in pool:
            if x.get("stage") != stage or x.get("kind") != "floor":
                continue
            out["ids"].append(x["id"])
            if ".floor.survival." in x["id"]:
                out["survival"] = x["line"]
                out["env_seeds"].add(int(x["env_seed"]))
            elif ".floor.ratio." in x["id"]:
                out["ratio"] = x["line"]
                out["env_seeds"].add(int(x["env_seed"]))
            elif ".floor.side22_" in x["id"]:
                out["side22"] = "fastfast" if x["id"].endswith("fastfast") else "notslow"
                out["side22_id"] = x["id"]
        out["env_seeds"] = sorted(out["env_seeds"])
        return out

    def rate(self, lid: str):
        x = self.by_id(lid)
        return None if x is None else x.get("rate")

    def unvalidated(self, lid: str) -> bool:
        x = self.by_id(lid)
        return bool(x and x.get("unvalidated"))


# ============================================================================== stamp readers
def _json(p: Path):
    return json.loads(Path(p).read_text())


def run_meta(stamp: Path) -> dict:
    return _json(Path(stamp) / "run_meta.json")


def aggregate(stamp: Path) -> dict:
    return _json(Path(stamp) / "scenario_metrics.json")["aggregate"]


def cat_summary(stamp: Path) -> dict:
    """run_phases.cat_summary on the stamp's aggregate (the driver's own reading of a suite row)."""
    return L.rc().rp.cat_summary(aggregate(stamp))


def completion(stamps) -> dict:
    return L.thresholds_mod().field_completion([Path(s) for s in stamps])


def side22(stamp: Path) -> dict:
    TH = L.thresholds_mod()
    return TH.side22_read(L.side22_mod(), Path(stamp))


# ============================================================================== Gate 0
def stamp_complete_problems(stamp: Path) -> list[str]:
    d = Path(stamp)
    out = []
    if not d.is_dir():
        return [f"{d}: no such stamp"]
    for f in ("scenario_metrics.json", "run_meta.json", "summary.md"):
        if not (d / f).is_file():
            out.append(f"{d.name}: incomplete stamp (no {f})")
    if not (d / "raw").is_dir():
        out.append(f"{d.name}: no raw/ (the raw npz must be kept)")
    if not any((d / m).is_file() for m in COMPLETE_MARKERS):
        out.append(f"{d.name}: no completion marker ({' / '.join(COMPLETE_MARKERS)})")
    return out


def carry_problems(env: dict, stage: str, carry: dict) -> list[str]:
    """The spec's NAMED carry checks on a stamp's env_vars (stage-dependent CLOCK_HIDE)."""
    out = []
    for k, v in (carry.get("common") or {}).items():
        if env.get(k) != v:
            out.append(f"carry env {k}={env.get(k)!r} (want {v!r})")
    ch = carry.get("clock_hide") or {}
    want = ch.get(stage, ch.get("default"))
    if want is not None and env.get("KRABBY_OMNI_CLOCK_HIDE") != want:
        out.append(f"carry env KRABBY_OMNI_CLOCK_HIDE={env.get('KRABBY_OMNI_CLOCK_HIDE')!r} (want {want!r} at {stage})")
    allowed = carry.get("command_frame_allowed", ["world", None])
    if env.get("KRABBY_OMNI_COMMAND_FRAME") not in allowed:
        out.append(f"carry env KRABBY_OMNI_COMMAND_FRAME={env.get('KRABBY_OMNI_COMMAND_FRAME')!r} (allowed {allowed})")
    for pre in carry.get("forbidden_prefixes") or []:
        bad = sorted(k for k in env if k.startswith(pre))
        if bad:
            out.append(f"carry env carries {bad} (forbidden on walker rows)")
    return out


def payload_problems(meta: dict, level_kg) -> list[str]:
    """``level_kg`` None = the row must carry NO payload; a number = every env at exactly that level (a level row:
    ZERO_FRAC 0, COM (0, 0, 0.309) -- the spec's level env)."""
    pl = meta.get("payload")
    if level_kg is None:
        # E1 (step 5): null / absent (a pre-E1 stamp) / an armed dict whose every env reads 0 kg all mean "no
        # payload" on a 0-kg / REFS / stand row; the key-for-key env compare above catches an armed env that the
        # composed stage env does not arm
        if pl in (None, {}):
            return []
        kg = pl.get("payload_kg") if isinstance(pl, dict) else None
        if isinstance(kg, list) and kg and all(abs(float(x)) <= PAYLOAD_TOL for x in kg):
            return []
        return [f"run_meta.payload carries a non-zero payload on a no-payload row ({str(pl)[:80]})"]
    if not isinstance(pl, dict):
        return [f"level row at {level_kg:g} kg has no run_meta.payload (the readback the gate needs)"]
    out = []
    kg = pl.get("payload_kg")
    if not isinstance(kg, list) or not kg:
        out.append("run_meta.payload.payload_kg missing / empty")
    elif any(abs(float(x) - float(level_kg)) > PAYLOAD_TOL for x in kg):
        bad = sorted({round(float(x), 6) for x in kg if abs(float(x) - float(level_kg)) > PAYLOAD_TOL})[:5]
        out.append(f"run_meta.payload.payload_kg != {level_kg:g} on some env ({bad})")
    rng = pl.get("kg_range")
    if not (isinstance(rng, list) and len(rng) == 2 and all(abs(float(x) - float(level_kg)) <= PAYLOAD_TOL
                                                           for x in rng)):
        out.append(f"run_meta.payload.kg_range {rng} != [{level_kg:g}, {level_kg:g}]")
    if float(pl.get("zero_frac", -1)) != 0.0 or float(pl.get("com_xy", -1)) != 0.0:
        out.append(f"level row zero_frac {pl.get('zero_frac')} / com_xy {pl.get('com_xy')} (want 0 / 0)")
    com = pl.get("payload_com_m")
    if isinstance(com, list) and com:
        # the CoM is checked only where a payload exists: a 0-kg env (the 0:0 level, L000) carries nothing to place,
        # and the event records its CoM as zeros (CHARTER step-8 note: C1 STOP-C on L000 was this reader, not the code)
        kgs = kg if isinstance(kg, list) and len(kg) == len(com) else [level_kg] * len(com)
        for c, k in zip(com, kgs):
            if abs(float(k)) <= PAYLOAD_TOL:
                continue
            if not (isinstance(c, list) and len(c) == 3 and abs(c[0]) <= PAYLOAD_TOL and abs(c[1]) <= PAYLOAD_TOL
                    and abs(c[2] - float(pl.get("com_z", 0.309))) <= PAYLOAD_TOL):
                out.append(f"level row payload CoM {c} != (0, 0, com_z)")
                break
    return out


def stamp_problems(stamp: Path, expect: dict) -> list[str]:
    """Every Gate-0 problem of ONE stamp against ``expect`` (pure on the stamp's files):

    ``expect`` keys: ``label``; ``stage``; ``pin`` {code_sha, eval_sha} (the lock's); ``source`` 'row' | 'driver';
    ``driver_pin`` {code_sha, eval_sha} (the reference record's, for a driver stamp); ``env`` (the composed env, or
    None to skip the key-for-key compare); ``carry`` (the spec's carry_env); ``ckpt_sha``; ``trained`` (a list, or
    None for a row without the flag); ``level_kg`` (None = no payload); ``adapter`` (e.g. 'body_hold' or None);
    ``task`` (the gym id the row must run, or None)."""
    lab = expect.get("label") or Path(stamp).name
    out = [f"{lab}: {p}" for p in stamp_complete_problems(stamp)]
    if out:
        return out
    meta = run_meta(stamp)
    env = {str(k): str(v) for k, v in (meta.get("env_vars") or {}).items()}
    pin = expect.get("pin") or {}
    if expect.get("source") == "driver":
        dp = expect.get("driver_pin") or {}
        for k in ("code_sha", "eval_sha"):
            if not dp.get(k) or dp.get(k) != pin.get(k):
                out.append(f"{lab}: driver record {k} {str(dp.get(k))[:12]} != pin {str(pin.get(k))[:12]}")
    else:
        cs = Path(stamp) / "code_state.json"
        if not cs.is_file():
            out.append(f"{lab}: no code_state.json (the row runner writes it after a clean exit)")
        else:
            c = _json(cs)
            for k in ("code_sha", "eval_sha"):
                if c.get(k) != pin.get(k):
                    out.append(f"{lab}: code_state {k} {str(c.get(k))[:12]} != pin {str(pin.get(k))[:12]}")
            if str(c.get("dirty")) != "False":
                out.append(f"{lab}: code_state dirty={c.get('dirty')} ({c.get('dirty_paths')})")
    if expect.get("env") is not None:
        want = {str(k): str(v) for k, v in expect["env"].items()}
        diff = sorted(set(want.items()) ^ set(env.items()))
        if diff:
            keys = sorted({k for k, _ in diff})
            out.append(f"{lab}: env_vars differ from the composed stage env on {keys[:8]}"
                       + (f" (+{len(keys) - 8})" if len(keys) > 8 else ""))
    out += [f"{lab}: {p}" for p in carry_problems(env, expect.get("stage") or "", expect.get("carry") or {})]
    if expect.get("ckpt_sha") and meta.get("checkpoint_sha256") != expect["ckpt_sha"]:
        out.append(f"{lab}: checkpoint_sha256 {str(meta.get('checkpoint_sha256'))[:12]} != the stage record's "
                   f"{expect['ckpt_sha'][:12]}")
    if "level_kg" in expect:
        out += [f"{lab}: {p}" for p in payload_problems(meta, expect.get("level_kg"))]
    if expect.get("trained") is not None:
        got = meta.get("trained_categories")
        if got is None:
            out.append(f"{lab}: run_meta.trained_categories missing (gate rows must pass --trained-categories)")
        elif list(got) != list(expect["trained"]):
            out.append(f"{lab}: trained_categories {got} != the stage's {expect['trained']}")
    if expect.get("adapter") and meta.get("command_adapter") != expect["adapter"]:
        out.append(f"{lab}: command_adapter {meta.get('command_adapter')!r} != {expect['adapter']!r}")
    if expect.get("task") and meta.get("task") != expect["task"]:
        out.append(f"{lab}: task {meta.get('task')!r} != {expect['task']!r}")
    return out


# ============================================================================== decisive lines
def _line(lid, statistic, value, line, direction, *, th: Thresholds | None, env_seed=None, extra=None,
          rate=None, unvalidated=None) -> dict:
    ok = None
    if value is not None and line is not None:
        ok = (value >= line - 1e-12) if direction == ">=" else (value <= line + 1e-12)
    rec = {"id": lid, "statistic": statistic, "value": value, "line": line, "direction": direction,
           "pass": bool(ok) if ok is not None else False, "missing": value is None,
           "rate": th.rate(lid) if (th is not None and rate is None) else rate,
           "unvalidated": th.unvalidated(lid) if (th is not None and unvalidated is None) else bool(unvalidated)}
    if env_seed is not None:
        rec["env_seed"] = env_seed
    if extra:
        rec.update(extra)
    return rec


def floor_reads(stage: str, th_stage: str, suites: dict, trained: list | None, th: Thresholds, *,
                standard: bool = False, env_seeds=None) -> list:
    """Floor lines on the suite stamps ``{env_seed: stamp}``: survival and ratio per TRAINED translate category.

    ``trained`` None -> the thresholds' own category list (a lines-only golden read of a recorded stamp);
    ``env_seeds`` restricts the floor env seeds (the student prescreen reads es1 only)."""
    fv = th.floor_values(th_stage, standard=standard)
    if fv["survival"] is None:
        return [{"id": f"{stage}.floor", "pass": False, "missing": True,
                 "why": f"thresholds.json has no floor lines for {th_stage}"}]
    out = []
    for es in (fv["env_seeds"] if env_seeds is None else [e for e in fv["env_seeds"] if e in env_seeds]):
        st = suites.get(es)
        if st is None:
            out.append({"id": f"{stage}.floor.es{es}", "pass": False, "missing": True,
                        "why": f"no suite stamp at env seed {es}"})
            continue
        cats = cat_summary(st)
        tc = [c for c in (trained if trained is not None else
                          [x.split(".")[3] for x in fv["ids"] if ".floor.survival." in x and x.endswith(f".es{es}")])
              if c in TRANSLATE]
        for c in tc:
            e = cats.get(c) or {}
            sid = f"{th_stage}.floor.survival.{c}.es{es}"
            rid = f"{th_stage}.floor.ratio.{c}.es{es}"
            out.append(_line(sid.replace(th_stage, stage, 1), f"translate survival, {c}", e.get("survival"),
                             fv["survival"], ">=", th=None, env_seed=es,
                             rate=th.rate(sid), unvalidated=th.unvalidated(sid),
                             extra={"thresholds_id": sid, **({"no_thresholds_line": True} if th.by_id(sid) is None
                                                            else {})}))
            out.append(_line(rid.replace(th_stage, stage, 1), f"translate ratio, {c}", e.get("ratio"),
                             fv["ratio"], ">=", th=None, env_seed=es,
                             rate=th.rate(rid), unvalidated=th.unvalidated(rid),
                             extra={"thresholds_id": rid, **({"no_thresholds_line": True} if th.by_id(rid) is None
                                                            else {})}))
    return out


SLOW_LIKE = {"SLOW", "DEAD", "DEAD?(n=0/NaN)"}


def side22_floor(stage: str, th_stage: str, stamp: Path | None, th: Thresholds, *, standard: bool = False) -> dict:
    fv = th.floor_values(th_stage, standard=standard)
    kind = fv.get("side22")
    lid = fv.get("side22_id") or f"{th_stage}.floor.side22"
    if kind is None:
        return {"id": f"{stage}.floor.side22", "pass": False, "missing": True,
                "why": f"thresholds.json has no side22 line for {th_stage}"}
    if stamp is None:
        return {"id": lid.replace(th_stage, stage, 1), "pass": False, "missing": True, "why": "no side22 stamp"}
    r = side22(stamp)
    cls = list(r["class"])
    ok = all(c == "FAST" for c in cls) if kind == "fastfast" else all(c not in SLOW_LIKE for c in cls)
    return {"id": lid.replace(th_stage, stage, 1), "thresholds_id": lid,
            "statistic": "side22_25lr feet_ss class, both hands" + (" FAST/FAST" if kind == "fastfast" else
                                                                   " not SLOW / DEAD"),
            "value": cls, "medians": r["median"], "line": kind, "pass": bool(ok), "missing": False,
            "rate": th.rate(lid), "unvalidated": th.unvalidated(lid)}


def field_hard_line(stage: str, stamps: list, th: Thresholds, *, require_env_seeds: bool = True) -> dict:
    """p2a's field_hard capability line. ``require_env_seeds`` False only for a golden known-bad read whose recorded
    rows are es1-2 (ANCW, n = 200, as thresholds.json reads it)."""
    x = th.by_id("p2a.capability.field_hard")
    if x is None:
        return {"id": "p2a.capability.field_hard", "pass": False, "missing": True, "why": "no line in thresholds"}
    want = list(x.get("env_seeds") or [1, 2, 3, 4])
    if require_env_seeds and len(stamps) != len(want):
        return {"id": f"{stage}.capability.field_hard", "pass": False, "missing": True,
                "why": f"field_hard needs env seeds {want}, got {len(stamps)} stamps"}
    c = completion(stamps)
    return _line(f"{stage}.capability.field_hard", "field_hard completion pooled es1-4 (n = 400)", c["rate"],
                 x["line"], ">=", th=None, rate=x.get("rate"), unvalidated=x.get("unvalidated"),
                 extra={"thresholds_id": "p2a.capability.field_hard", "n": c["n"], "per_env_seed": c["per_env_seed"]})


# ============================================================================== the paired 0-kg check
def zero_kg_check(head: dict, p2a: dict, trained_translate: list, th: Thresholds, its_past_p2a: float,
                  resolution_files: list) -> dict:
    """The paired 0-kg check (PLAN Payload 3f) of a payload head against its own p2a head.

    ``head`` / ``p2a``: ``{"suite": {es: stamp}, "side22": stamp, "<field row>": [stamps in env-seed order]}`` on
    the SAME rows and env seeds. Uses thresholds.zero_kg_suite_rows / zero_kg_completion_row / zero_kg_judge and the
    drift recorded in thresholds.json (``drift_by_statistic``)."""
    TH = L.thresholds_mod()
    z = th.by_id("p3a.capability.zero_kg_paired")
    if z is None or "drift_by_statistic" not in z:
        return {"verdict": "FAIL", "fails": ["thresholds.json has no zero_kg_paired line with drift_by_statistic "
                                             "(regenerate thresholds with the STOP-0 semantics)"], "rows": []}
    R = TH.resolution_R([L.absp(p) for p in resolution_files])
    Rg = TH.resolution_R_guard([L.absp(p) for p in resolution_files])
    A = {int(es): TH.suite_episodes(Path(s)) for es, s in head["suite"].items()}
    B = {int(es): TH.suite_episodes(Path(s)) for es, s in p2a["suite"].items()}
    if set(A) != set(B):
        return {"verdict": "FAIL", "fails": [f"suite env seeds differ: head {sorted(A)} vs p2a {sorted(B)}"],
                "rows": []}
    rows = TH.zero_kg_suite_rows(A, B, trained_translate, R, Rg)
    for row in TH.ZERO_KG_FIELD_ROWS:
        ha, pb = head.get(row) or [], p2a.get(row) or []
        ca, cb = {}, {}
        for i, (x, y) in enumerate(zip(ha, pb)):
            ca.update({(i, k): v for k, v in TH.field_episodes(Path(x)).items()})
            cb.update({(i, k): v for k, v in TH.field_episodes(Path(y)).items()})
        if len(ha) != len(pb) or not ha:
            rows.append({"statistic": f"{row} completion", "row": row, "watch": False, "missing": True,
                         "why": f"{row}: head {len(ha)} vs p2a {len(pb)} stamps"})
            continue
        rows.append(TH.zero_kg_completion_row(row, ca, cb, list(range(1, len(ha) + 1))))
    s22h = side22(head["side22"])["class"] if head.get("side22") else None
    s22p = side22(p2a["side22"])["class"] if p2a.get("side22") else None
    res = TH.zero_kg_judge(rows, z["drift_by_statistic"], its_past_p2a, side22_head=s22h, side22_p2a=s22p)
    if s22h is None or s22p is None:
        res["verdict"] = "FAIL"
        res["fails"].append("side22 stamp missing on the head or its p2a")
    used = [r.get("margin_used") for r in res["rows"] if not r.get("watch") and r.get("margin_used") is not None]
    res["max_margin_used"] = max(used) if used else None
    res["rate_note"] = z.get("rate_note")
    return res


# ============================================================================== stage read
def verdict_of(decisive: list, gate0: list, *, ceiling_on_fail: bool = False, watch_flags=None) -> str:
    if gate0:
        return INVALID
    if any(not d.get("pass") for d in decisive):
        return CEILING if ceiling_on_fail else HALT
    return WATCH if watch_flags else PASS


def watch_reads(stage: str, stamps: dict, rows_spec: dict) -> dict:
    """The reported (WATCH) numbers of a stage's rows: every category of every suite-kind row, every field row's
    completion, side22 medians. Pure re-readings; nothing here is decisive."""
    out = {}
    for key, st in sorted(stamps.items()):
        fam = key.split("@", 1)[0]
        kind = (rows_spec.get(fam) or {}).get("kind")
        try:
            if kind in ("suite", "body", "turn"):
                a = aggregate(st)
                if a.get("suite"):
                    cs = L.rc().rp.cat_summary(a)
                    out[key] = {c: {k: v.get(k) for k in ("n", "survival", "ratio", "facing_closed", "yaw_progress",
                                                         "tripod") if k in v} for c, v in cs.items()}
                else:
                    out[key] = {"completion": a.get("schedule_completion_rate")}
            elif kind == "field":
                a = aggregate(st)
                out[key] = {"completion": a.get("schedule_completion_rate"), "n": a.get("n_episodes")}
            elif kind == "lr":
                r = side22(st)
                out[key] = {"class": r["class"], "median": r["median"]}
        except Exception as exc:  # noqa: BLE001  (a WATCH read never decides; it reports why it is absent)
            out[key] = {"unreadable": repr(exc)[:200]}
    return out


def walker_decisive(stage: str, stamps: dict, trained: list | None, th: Thresholds, *, th_stage: str | None = None,
                    standard: bool = False) -> list:
    """The floors (+ field_hard at p2a) of a walker / payload / student stage on ``stamps`` {row@es: stamp}."""
    th_stage = th_stage or stage
    suites = {int(k.split("@")[1].split(":")[0]): v for k, v in stamps.items() if k.startswith("suite@")}
    out = floor_reads(stage, th_stage, suites, trained, th, standard=standard)
    out.append(side22_floor(stage, th_stage, stamps.get("side22@1"), th, standard=standard))
    if th_stage == "p2a" and not standard:
        fh = [stamps[f"field_hard@{es}"] for es in (1, 2, 3, 4) if f"field_hard@{es}" in stamps]
        out.append(field_hard_line(stage, fh, th))
    return out


def read_lines(stage: str, stamps: dict, th: Thresholds, *, trained: list | None = None,
               th_stage: str | None = None) -> dict:
    """Lines only, no Gate 0 (the golden reads of recorded stamps, PLAN step 7): the walker floors and p2a's
    field_hard line on ``stamps`` {"suite@1": dir, "side22@1": dir, "field_hard@1": dir, ...}."""
    dec = walker_decisive(stage, {k: Path(v) for k, v in stamps.items()}, trained, th, th_stage=th_stage)
    v = HALT if any(not d.get("pass") for d in dec) else PASS
    return {"stage": stage, "verdict": v, "decisive": dec, "halt_lines": [d["id"] for d in dec if not d.get("pass")]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lines-only", action="store_true")
    ap.add_argument("--stamp", action="append", default=[], metavar="ROW@ES=DIR")
    ap.add_argument("--trained", default=None, help="CSV (lines-only)")
    ap.add_argument("--th-stage", default=None)
    a = ap.parse_args(argv)
    if a.spec is not None:
        L.configure_from(a.spec)
    if a.lines_only:
        if a.thresholds is None or a.spec is None:
            ap.error("--lines-only needs --thresholds and --spec (its paths block)")
        th = Thresholds.load(a.thresholds)
        stamps = dict(s.split("=", 1) for s in a.stamp)
        res = read_lines(a.stage, stamps, th, trained=a.trained.split(",") if a.trained else None,
                         th_stage=a.th_stage)
        print(json.dumps(res, indent=1, default=str))
        return EXIT[res["verdict"]]
    if a.spec is None or a.run is None:
        ap.error("--spec and --run are required (or --lines-only)")
    P = L.load_by_path(HERE / "pipeline.py", "tp_pipeline")
    res = P.gate_stage(P.Ctx.open(a.spec, a.run), a.stage, thresholds=a.thresholds, out=a.out)
    print(json.dumps({k: res[k] for k in ("stage", "verdict", "halt_lines") if k in res}, indent=1, default=str))
    return EXIT.get(res["verdict"], 4)


if __name__ == "__main__":
    sys.exit(main())
