#!/usr/bin/env python3
"""Rough-terrain P2 (p2/P2.md Design > Distill, step 4): the c1 distill's +10-iteration checks, its record, its build
pin and its stop -- CPU only, never Kit.

A thin layer over the rod-speed DR campaign's ``rsd_distill_check`` / ``rsd_launch`` (imported from
``2026-11-07_rod_speed_dr/scripts``, byte-unchanged): its D2 / D3 / D4 are reused as they are; D1 and D5 are replaced
(the P2 differences), D6 runs only when the live rules apply, and D7 is new.

    PY=/home/nickmagus/krabby/isaac_venv/bin/python
    $PY -I -B p2_distill_check.py check [--name RT_ST_s19] [--teacher C1_ABS] [--log LOG] [--run-dir DIR] [--json]
    $PY -I -B p2_distill_check.py reference [--json]   # D1 / D5 / D7 parsing on RT_s19's own log + tfevents
    $PY -I -B p2_distill_check.py pin                  # build == the RT pin (code / eval sha), clean tree
    $PY -I -B p2_distill_check.py record --name N --teacher C1_ABS --env-file ENV [--launcher-argv ...]
    $PY -I -B p2_distill_check.py stop --name N [--unit UNIT] [--dry]

``check`` (once ``<P2 distill out>/NAME.log`` shows iteration first+10; ``p2_distill.sh`` prints the command):
  D1  the ``[KNOB]`` / ``[omni-knobs]`` multiset before the first iteration = ``RT_s19.log``'s minus the ``[KNOB]``
      lines of the runner-only keys ``distill.sh`` strips (``omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED`` that the
      env file sets; cross-checked against ``NAME.stripped``) -- this includes the two ``(site geometry)`` radius sync
      lines (asserted on their own too, with no ``(field geometry)`` line); the ``[rt_friction]`` line = the line
      predicted by the contract's own module (``rt_friction.contract_mu`` over the generator's column layout and
      IsaacLab's ``terrain_types`` rule, for 192 envs and seed 19 -- 20 gravel envs, 29-48; the predictor reproduces
      RT_s19's 256-env line verbatim, see ``reference``); every ``Metrics/base_parkour/site_level/<kind>`` of the mix
      and ``terrain_levels`` present in each complete iteration block up to +10;
  D2  (RSD) the ``[DR]`` line = the predicted first rod-speed draw (192 envs, seed 19, U(0.6, 1.35));
  D3  (RSD + full sha) the loaded checkpoint and ``params/agent.yaml`` ``load_checkpoint`` are the teacher, and its
      sha256 is c1's full sha;
  D4  (RSD) the first iteration = the teacher's iteration (29500) and the log reached +10;
  D5  a WINDOWED tail of RT_s19 (its 29301-29500, the iterations c1 was trained through; 200 points per tag) of
      ``Train/mean_episode_length`` and ``Episode_Termination/crab_failure`` -- the student live rules' reference.
      ``live_watch.tail_mean``'s default would read RT_s19's last 200 points (34791-34990), which is the wrong
      policy; that value is printed for contrast only;
  D6  the student live rules (``live_watch.check("student", ...)`` with the D5 tail and Task 4's ``pipeline.yaml``
      live config) ONLY once they apply (+250): before that the line reads N/A ("not yet applicable"), never PASS;
      once they apply the result is REPORT ONLY (plan step 4: as in RSD, the Task 4 student live rules are not
      applied after +10, because c1's own rolling-50 crab_failure crosses their c1-tail + 0.10 line);
  D7  pace: the mean ``Iteration time`` over the first 10 iterations, the implied 5000-iteration hours, and a flag
      above 10.5 s/it (the plan's budget re-notify line). A COST line only: it never changes the exit code.
Exit 3 = a real D1-D5 miss only -> ``stop`` (D6 and D7 never change the exit code); 2 = not yet at +10 (re-run later); 0 = pass.
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
CD = HERE.parent                                          # experiments/2026-11-10_rough_terrain
RSD_SCRIPTS = CD.parent / "2026-11-07_rod_speed_dr" / "scripts"
if str(RSD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RSD_SCRIPTS))
import rsd_launch as RL  # noqa: E402
import rsd_distill_check as RDC  # noqa: E402

MAIN = RL.MAIN
OMNI = RL.OMNI
EXPERIMENT = "crab_hex_omni_student"
NAME = "RT_ST_s19"
NUM_ENVS, SEED, ITERS = 192, 19, 5000
DR = "0.6:1.35"
#: c1 = RT_s19 model_29500 (CHARTER 2026-11-11 17:25, decision "A")
C1_SHA = "6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1"
C1_RUN_DIR = RL.W / "parkour" / "logs" / "rsl_rl" / "crab_hex_omni_teacher" / "2026-11-10_23-43-01_RT_s19"
C1_PATH = C1_RUN_DIR / "model_29500.pt"
RT_S19_LOG = MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain" / "train" / "RT_s19.log"
RT_S19_NUM_ENVS = 256
#: D5: the iterations c1 trained through (its last 200; saves model_29400 / model_29500 in the window)
TAIL_WINDOW = (29301, 29500)
TAIL_POINTS = 200
#: D7
PACE_ITERS = 10
PACE_FLAG_S = 10.5
P2_OUT = MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain_p2"
DISTILL_OUT = P2_OUT / "distill"
RESULTS_P2 = CD / "results" / "p2"
RUNS = RESULTS_P2 / "runs"
ENV_FILE = CD / "envs" / "P2_distill.env"
RT_ENV = CD / "envs" / "RT_train.env"
RT_PIN = CD / "results" / "pin.json"
P2_PIN = RESULTS_P2 / "pin.json"
CHARTER = CD / "CHARTER.md"
CHARTER_HEAD_RE = re.compile(r'^### 2026-11-11 17:25 \S+ User decision at the HALT_R: "A"', re.M)
SPEC_PY = OMNI / "config" / "crab_hex" / "omni_student_spec.py"
OTK_PY = OMNI / "mdp" / "omni_terrain_knobs.py"
RTF_PY = OMNI / "mdp" / "rt_friction.py"
P2_DISTILL_SH = HERE / "p2_distill.sh"
SITE_SYNC_RE = re.compile(r"^\[omni-knobs\] .*syncing to the terrain \((site|field) geometry\)\.$")
KNOB_KEY_RE = re.compile(r"^\[KNOB\] (KRABBY_[A-Z0-9_]+)=")
RTF_LINE_RE = re.compile(r"^\[rt_friction\] ")
ITER_TIME_RE = re.compile(r"^[ \t]*Iteration time:[ \t]*([0-9.]+)s[ \t]*$", re.M)
#: rsl_rl's title line only (leading blanks, the bold escape): a wrapper line QUOTING a title -- rt_train's
#: "RT TRAIN PF ... PF10 ... 'Learning iteration 28991/34991'" in RT_s19.log -- is not an iteration
TITLE_RE = re.compile(r"^[ \t]*(?:\x1b\[1m)?[ \t]*Learning iteration (\d+)/(\d+)", re.M)
METRIC_PREFIX = "Metrics/base_parkour/"
RC_PASS, RC_WAIT, RC_KILL = RDC.RC_PASS, RDC.RC_WAIT, RDC.RC_KILL
res = RDC.res
PFX = "P2 DISTILL CHECK"


def title_blocks(text: str) -> list[tuple[int, int, str]]:
    """``rsd_launch.iteration_blocks`` on rsl_rl's title lines only (see :data:`TITLE_RE`)."""
    marks = [(m.start(), int(m.group(1)), int(m.group(2))) for m in TITLE_RE.finditer(text)]
    return [(it, tot, text[pos:(marks[j + 1][0] if j + 1 < len(marks) else len(text))])
            for j, (pos, it, tot) in enumerate(marks)]


# ============================================================================== shared readers
def live_watch():
    """Task 4's ``live_watch`` under the name ``rsd_distill_check`` uses, with ``read_series`` memoised per
    (run dir, tags): RSD's D5 and this module's D5 then read a 133 MB tfevents once, not twice."""
    lw = RL.load_by_path(RL.LIVE_WATCH_PY, "rsd_live_watch")
    if not getattr(lw, "_p2_memo", False):
        orig, memo = lw.read_series, {}

        def read_series(run_dir, tags=None):
            key = (str(Path(run_dir).resolve()), None if tags is None else tuple(sorted(tags)))
            if key not in memo:
                memo[key] = orig(run_dir, tags)
            return memo[key]

        lw.read_series, lw._p2_memo = read_series, True
    return lw


def head_of(path: Path) -> str:
    return RDC._head(RDC._read(path))


def env_map(path: Path) -> dict:
    """``{KEY: VALUE}`` of an env file's ``KRABBY_*`` lines: RSD's strict reader for a KRABBY-only file (P2_distill.env),
    the plain ``KRABBY_*`` lines of a launcher env that also carries process keys (RT_train.env)."""
    pairs, problems = RL.read_env_lines(path) if _only_krabby(path) else (_krabby(path), [])
    if problems:
        raise ValueError("; ".join(problems))
    return dict(pairs)


def _only_krabby(path: Path) -> bool:
    return all(not ln.strip() or ln.lstrip().startswith("#") or ln.startswith("KRABBY_")
               for ln in path.read_text().split("\n"))


def _krabby(path: Path) -> list[tuple[str, str]]:
    """``(KEY, VALUE)`` of the ``KRABBY_*`` lines only (RT_train.env also carries process keys)."""
    return [(m.group(1), m.group(2)) for m in (re.match(r"^(KRABBY_[A-Z0-9_]+)=(.*)$", ln)
                                               for ln in path.read_text().split("\n")) if m]


def refused_keys() -> tuple[str, ...]:
    """``omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED`` by ast (distill.sh's own read)."""
    import ast  # noqa: PLC0415

    for node in ast.walk(ast.parse(SPEC_PY.read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED"
                                                for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"no STUDENT_RUNNER_KNOBS_REFUSED in {SPEC_PY}")


def stripped_keys(env_file: Path) -> list[str]:
    """The keys distill.sh strips from ``env_file`` (refused order, as it prints them)."""
    have = env_map(env_file)
    return [k for k in refused_keys() if k in have]


# ============================================================================== D1 pieces
def d1_want(reference_head: str, strip) -> tuple[list[str], list[str]]:
    """``(want, removed)``: the reference's knob lines minus the ``[KNOB] <stripped key>=`` lines."""
    want, removed = [], []
    for ln in RL.knob_lines(reference_head):
        m = KNOB_KEY_RE.match(ln)
        (removed if m and m.group(1) in strip else want).append(ln)
    return want, removed


def site_sync_lines(knob_lines: list[str]) -> dict:
    hits = [ln for ln in knob_lines if SITE_SYNC_RE.match(ln)]
    return {"site": [h for h in hits if "(site geometry)" in h], "field": [h for h in hits if "(field geometry)" in h]}


def rt_column_kinds(mix_text: str) -> list[str]:
    """The generator's columns, left to right: each kind ``count`` consecutive columns in ``RT_COLUMN_KINDS`` order
    (``omni_terrain_knobs.apply_site_mix``'s contract)."""
    otk = RL.load_by_path(OTK_PY, "p2_omni_terrain_knobs")
    mix = otk.parse_site_mix(mix_text, ladder=otk.RT_LADDER)
    return [k for k, c in mix.items() for _ in range(int(c))]


def predict_rt_friction(num_envs: int, seed: int, mix_text: str, friction_text: str, knob_line: str) -> dict:
    """The ``[rt_friction]`` line ``rt_friction.rt_friction_contract`` prints at startup (all envs), from the module
    itself: IsaacLab's curriculum ``terrain_types = floor(arange(N) / (N / num_cols))`` (float32, as the importer
    computes it), the gravel columns of the mix, the bucket range / count the ``[KNOB]`` line names, and
    ``contract_mu``. The f-string is the module's own."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    otk = RL.load_by_path(OTK_PY, "p2_omni_terrain_knobs")
    rtf = RL.load_by_path(RTF_PY, "p2_rt_friction")
    cols = rt_column_kinds(mix_text)
    spec = otk.parse_rt_friction(friction_text)
    m = re.search(r"of (\d+) buckets b ~ U\(([-0-9.e]+), ([-0-9.e]+)\)", knob_line or "")
    if not m:
        raise ValueError(f"no 'of N buckets b ~ U(lo, hi)' in the RT_FRICTION [KNOB] line: {knob_line!r}")
    num_buckets, frange = int(m.group(1)), (float(m.group(2)), float(m.group(3)))
    types = torch.div(torch.arange(num_envs), (num_envs / len(cols)), rounding_mode="floor").to(torch.long).numpy()
    is_g = np.array([c == otk.RT_FRICTION_KIND for c in cols], dtype=bool)
    g = is_g[types]
    mu = rtf.contract_mu(seed, np.arange(num_envs), g, frange, num_buckets, (spec["lo"], spec["hi"]))
    line = (f"[rt_friction] {len(mu)} envs: {int(g.sum())} gravel mu in [{mu[g].min() if g.any() else math.nan:.3f}, "
            f"{mu[g].max() if g.any() else math.nan:.3f}], {int((~g).sum())} others mu in "
            f"[{mu[~g].min() if (~g).any() else math.nan:.3f}, {mu[~g].max() if (~g).any() else math.nan:.3f}] "
            f"(seed {seed})")
    return {"line": line, "gravel_envs": int(g.sum()), "num_envs": num_envs, "seed": seed, "num_cols": len(cols),
            "gravel_cols": [i for i, c in enumerate(cols) if c == otk.RT_FRICTION_KIND],
            "gravel_env_index": [int(i) for i in np.nonzero(g)[0]], "num_buckets": num_buckets,
            "friction_range": frange, "gravel_range": (spec["lo"], spec["hi"]), "module": str(RTF_PY)}


def friction_knob_line(knob_lines: list[str]) -> str | None:
    hits = [ln for ln in knob_lines if ln.startswith("[KNOB] KRABBY_RT_FRICTION=")]
    return hits[0] if len(hits) == 1 else None


def site_kinds(mix_text: str) -> list[str]:
    out = []
    for k in rt_column_kinds(mix_text):
        if k not in out:
            out.append(k)
    return out


def site_level_presence(text: str, kinds, upto_rel: int) -> dict:
    """Each COMPLETE iteration block from the first through first+``upto_rel`` carries ``site_level/<kind>`` for every
    kind and ``terrain_levels`` (the last block of a live log may be partial and is skipped)."""
    blocks = title_blocks(text)
    if len(blocks) < 2:
        return {"ok": False, "blocks": 0, "missing": {"all": "fewer than 2 iteration blocks"}}
    first = blocks[0][0]
    done = [b for b in blocks[:-1] if b[0] <= first + upto_rel]
    want = [f"{METRIC_PREFIX}site_level/{k}" for k in kinds] + [f"{METRIC_PREFIX}terrain_levels"]
    missing = {}
    for it, _tot, blk in done:
        vals = RL.block_values(blk, METRIC_PREFIX)
        miss = [t for t in want if t not in vals]
        if miss:
            missing[it] = miss
    return {"ok": bool(done) and not missing, "blocks": len(done), "first": first,
            "last": done[-1][0] if done else None, "tags": want, "missing": missing}


def d1(head: str, text: str, *, reference_head: str, strip, mix_text: str, friction_text: str, num_envs: int,
       seed: int, stripped_file: Path | None, min_iters: int) -> dict:
    got = RL.knob_lines(head)
    want, removed = d1_want(reference_head, strip)
    why, data = [], {"removed_from_reference": removed, "strip": list(strip)}
    d = RL.multiset_diff(got, want)
    data.update(d)
    if d["missing"] or d["extra"]:
        why.append(f"[KNOB]/[omni-knobs] missing {d['missing']} extra {d['extra']}")
    sync = site_sync_lines(got)
    data["site_sync"] = sync
    if len(sync["site"]) != 2 or sync["field"]:
        why.append(f"site geometry sync lines {len(sync['site'])} (want 2), field geometry lines {len(sync['field'])}")
    if stripped_file is not None and stripped_file.is_file():
        on_disk = stripped_file.read_text().split()
        data["stripped_file"] = on_disk
        if sorted(on_disk) != sorted(strip):
            why.append(f"{stripped_file.name} {on_disk} != the computed strip {list(strip)}")
    kl = friction_knob_line(got) or friction_knob_line(RL.knob_lines(reference_head))
    try:
        p = predict_rt_friction(num_envs, seed, mix_text, friction_text, kl)
        have = [RL.normalise_roots(ln.rstrip("\r")) for ln in head.split("\n") if RTF_LINE_RE.match(ln)]
        data["rt_friction"] = {"got": have, "predicted": p["line"], "gravel_envs": p["gravel_envs"],
                               "gravel_cols": p["gravel_cols"]}
        if have != [p["line"]]:
            why.append(f"[rt_friction] got {have}, want [{p['line']}]")
    except Exception as exc:  # noqa: BLE001
        why.append(f"cannot predict the [rt_friction] line: {exc}")
    pres = site_level_presence(text, site_kinds(mix_text), min_iters)
    data["site_level"] = pres
    if not pres["ok"]:
        why.append(f"site_level / terrain_levels metrics missing in blocks {pres['missing']}")
    ok = not why
    rtf = data.get("rt_friction") or {}
    detail = ("; ".join(why) if why else
              f"{len(want)} knob lines = RT_s19's minus {len(removed)} stripped-key lines ({', '.join(strip)}); "
              f"2 site-geometry sync lines; [rt_friction] = predicted ({rtf.get('gravel_envs')} gravel of "
              f"{num_envs}); site_level x{len(pres['tags']) - 1} + terrain_levels in {pres['blocks']} blocks")
    return res("D1", ok, detail, **data)


# ============================================================================== D5 / D6 / D7
def windowed_tail(run_dir: Path, window=TAIL_WINDOW, points=TAIL_POINTS) -> dict:
    lw = live_watch()
    ser = lw.read_series(run_dir, (lw.T_EPLEN, lw.T_CRAB))
    out = {"run_dir": str(run_dir), "window": list(window)}
    for key, tag in (("ep_len", lw.T_EPLEN), ("crab_failure", lw.T_CRAB)):
        pts = [(s, v) for s, v in (ser.get(tag) or []) if window[0] <= s <= window[1]]
        out[key] = sum(v for _, v in pts) / len(pts) if pts else None
        out[f"n_{key}"] = len(pts)
        out[f"steps_{key}"] = [pts[0][0], pts[-1][0]] if pts else None
        allp = ser.get(tag) or []
        out[f"default_tail_{key}"] = lw.tail_mean(allp)
        out[f"default_tail_steps_{key}"] = [allp[-min(200, len(allp))][0], allp[-1][0]] if allp else None
    out["ok"] = all(out[k] is not None and out[f"n_{k}"] == points for k in ("ep_len", "crab_failure"))
    return out


def d5(teacher_run_dir: Path) -> tuple[dict, dict | None]:
    try:
        t = windowed_tail(teacher_run_dir)
    except Exception as exc:  # noqa: BLE001
        return res("D5", False, f"windowed tail from {teacher_run_dir} failed: {exc}"), None
    if not t["ok"]:
        return res("D5", False, f"windowed tail {TAIL_WINDOW} in {teacher_run_dir}: n ep_len {t['n_ep_len']}, n "
                                f"crab_failure {t['n_crab_failure']} (want {TAIL_POINTS} each)", tail=t), None
    return res("D5", True, f"c1 tail its {TAIL_WINDOW[0]}-{TAIL_WINDOW[1]} (n {TAIL_POINTS}): ep_len "
                           f"{t['ep_len']:.1f}, crab_failure {t['crab_failure']:.4f} (not used: the default last-200 "
                           f"{t['default_tail_steps_ep_len']} gives ep_len {t['default_tail_ep_len']:.1f}, "
                           f"crab_failure {t['default_tail_crab_failure']:.4f})", tail=t), \
        {"ep_len": t["ep_len"], "crab_failure": t["crab_failure"]}


def d6(run_dir: Path | None, first: int, last: int, tail: dict | None) -> dict:
    cfg = RDC.live_cfg()
    lw = live_watch()
    sc = {**lw.DEFAULTS["student"], **(cfg.get("student") or {})}
    starts = min(int(sc["loss_at"]), int(sc["from"]))
    if last < first + starts:
        return res("D6", "n/a", f"not yet applicable: the student live rules start at +{starts} (log at "
                                f"+{last - first})", starts_at=first + starts)
    if run_dir is None or tail is None:
        return res("D6", "report", "REPORT ONLY (plan step 4): live rules would apply but there is no student run "
                                   "dir / no D5 tail")
    r = lw.check("student", lw.read_series(run_dir), tstar_tail=tail, cfg=cfg)
    return res("D6", "report", f"REPORT ONLY (plan step 4: the student live rules are not applied after +10): live rc "
                               f"{r['rc']} rule {r.get('rule')} fails {r.get('fails')} watch {r.get('watch')}", live=r)


def pace(text: str, n: int = PACE_ITERS, iters: int = ITERS) -> dict:
    blocks = title_blocks(text)
    vals = []
    for it, _tot, blk in blocks[:n]:
        m = ITER_TIME_RE.search(blk)
        if m:
            vals.append((it, float(m.group(1))))
    if len(vals) < n:
        return {"readable": False, "n": len(vals), "why": f"{len(vals)} of {n} 'Iteration time' lines"}
    s = sum(v for _, v in vals) / len(vals)
    return {"readable": True, "n": len(vals), "its": [vals[0][0], vals[-1][0]], "s_per_it": s,
            "hours_total": s * iters / 3600.0, "flag": s > PACE_FLAG_S, "flag_line_s": PACE_FLAG_S}


def d7(text: str) -> dict:
    p = pace(text)
    if not p.get("readable"):
        return res("D7", "cost", f"pace not readable: {p.get('why')}", **p)
    msg = (f"pace {p['s_per_it']:.2f} s/it over its {p['its'][0]}-{p['its'][1]} -> {p['hours_total']:.1f} h for "
           f"{ITERS} its")
    msg += (f"; FLAG > {PACE_FLAG_S} s/it: re-notify the budget (never a kill)" if p["flag"]
            else f" (<= {PACE_FLAG_S} s/it)")
    return res("D7", "cost", msg, **p)


# ============================================================================== check
def checks(*, name: str, teacher: Path, log: Path, run_dir: Path | None, teacher_run_dir: Path, env_file: Path,
           reference_log: Path, stripped_file: Path | None, min_iters: int = 10) -> tuple[int, list[dict]]:
    live_watch()  # install the memo before RSD's D5 reads the teacher's tfevents
    rc, rsd = RDC.checks(name=name, teacher=teacher, log=log, run_dir=run_dir, dr=DR, num_envs=NUM_ENVS, seed=SEED,
                         teacher_run_dir=teacher_run_dir, min_iters=min_iters, reference_log=reference_log,
                         live=False)
    if rc == RC_WAIT:
        return rc, rsd
    by = {r["id"]: r for r in rsd}
    text = RDC._read(log)
    head = RDC._head(text)
    env = env_map(env_file)
    out = [d1(head, text, reference_head=head_of(reference_log), strip=stripped_keys(env_file),
              mix_text=env["KRABBY_SITE_MIX"], friction_text=env["KRABBY_RT_FRICTION"], num_envs=NUM_ENVS, seed=SEED,
              stripped_file=stripped_file, min_iters=min_iters)]
    out.append(by["D2"])
    d3 = dict(by["D3"])
    sha = (d3.get("data") or {}).get("teacher_sha256")
    if sha != C1_SHA:
        d3 = res("D3", False, f"teacher sha {str(sha)[:12]} != c1 {C1_SHA[:12]}; " + d3["detail"], **d3["data"])
    d3["detail"] = d3["detail"].replace("= TDR", "= c1").replace("!= TDR", "!= c1")
    out.append(d3)
    d4 = dict(by["D4"])
    d4["detail"] = d4["detail"].replace("(TDR iter", "(c1 iter")
    out.append(d4)
    r5, tail = d5(teacher_run_dir)
    out.append(r5)
    first, last = d4["data"]["first"], d4["data"]["last"]
    out.append(d6(run_dir, first, last, tail))
    out.append(d7(text))
    bad = [r for r in out if r["ok"] is False]
    return (RC_KILL if bad else RC_PASS), out


def _tag(ok) -> str:
    return {True: "PASS", False: "MISS", None: "WAIT", "n/a": "N/A", "cost": "COST", "report": "REPORT"}.get(ok, str(ok))


def emit(name: str, rc: int, out: list[dict], as_json: bool) -> None:
    for r in out:
        print(f"{PFX} {name} {r['id']} {_tag(r['ok'])} {r['detail'][:900]}")
    verdict = {RC_PASS: "PASS", RC_WAIT: "WAIT (re-run later)", RC_KILL: "MISS -> kill (p2_distill_check.py stop)"}
    print(f"{PFX} {name} RESULT {verdict[rc]}")
    if as_json:
        print(json.dumps({"rc": rc, "checks": out}, indent=1, default=str))


def cmd_check(a) -> int:
    teacher = Path(a.teacher)
    if not teacher.is_absolute() or not teacher.is_file():
        print(f"p2_distill_check: --teacher must be an existing absolute path: {a.teacher}", file=sys.stderr)
        return RC_KILL
    log = Path(a.log) if a.log else DISTILL_OUT / f"{a.name}.log"
    run_dir = Path(a.run_dir) if a.run_dir else RL.find_run_dir(a.name, EXPERIMENT)
    rc, out = checks(name=a.name, teacher=teacher, log=log, run_dir=run_dir,
                     teacher_run_dir=Path(a.teacher_run_dir) if a.teacher_run_dir else teacher.parent,
                     env_file=Path(a.env_file), reference_log=Path(a.reference_log),
                     stripped_file=DISTILL_OUT / f"{a.name}.stripped", min_iters=a.min_iters)
    emit(a.name, rc, out, a.json)
    return rc


# ============================================================================== reference (self-test)
def cmd_reference(a) -> int:
    """D1 / D5 / D7 parsing on RT_s19 itself (the teacher run c1 came from; 256 envs, seed 19)."""
    out, nm = [], "RT_s19(reference)"
    text = RDC._read(RT_S19_LOG)
    head = RDC._head(text)
    env, rt = env_map(ENV_FILE), dict(_krabby(RT_ENV))
    strip = stripped_keys(ENV_FILE)
    got = RL.knob_lines(head)
    want, removed = d1_want(head, strip)
    rkeys = [KNOB_KEY_RE.match(ln).group(1) for ln in removed]
    printed_strip = [k for k in strip if any(ln.startswith(f"[KNOB] {k}=") for ln in got)]
    ok = sorted(rkeys) == sorted(printed_strip) and len(want) + len(removed) == len(got)
    out.append(res("R1a", ok, f"RT_s19 {len(got)} knob lines -> D1 want {len(want)} after removing {len(removed)} "
                              f"([KNOB] lines of {rkeys}); strip {strip} (unprinted: "
                              f"{[k for k in strip if k not in printed_strip]})", want=want, removed=removed))
    sync = site_sync_lines(got)
    out.append(res("R1b", len(sync["site"]) == 2 and not sync["field"],
                   f"site geometry sync lines {len(sync['site'])}, field {len(sync['field'])}", **sync))
    try:
        kl = friction_knob_line(got)
        p256 = predict_rt_friction(RT_S19_NUM_ENVS, SEED, rt["KRABBY_SITE_MIX"], rt["KRABBY_RT_FRICTION"], kl)
        have = [ln.rstrip("\r") for ln in head.split("\n") if RTF_LINE_RE.match(ln)]
        out.append(res("R1c", have == [p256["line"]], f"predictor at 256 envs reproduces RT_s19's line: "
                                                      f"{p256['line']!r} vs got {have}", predicted=p256))
        p192 = predict_rt_friction(NUM_ENVS, SEED, env["KRABBY_SITE_MIX"], env["KRABBY_RT_FRICTION"], kl)
        out.append(res("R1d", 15 <= p192["gravel_envs"] <= 25, f"student prediction ({NUM_ENVS} envs, seed {SEED}, "
                                                               f"gravel cols {p192['gravel_cols']}): {p192['line']!r}",
                       predicted=p192))
    except Exception as exc:  # noqa: BLE001
        out.append(res("R1c", False, f"rt_friction prediction failed: {exc}"))
    pres = site_level_presence(text, site_kinds(env["KRABBY_SITE_MIX"]), 10)
    out.append(res("R1e", pres["ok"], f"site_level x{len(pres['tags']) - 1} + terrain_levels in {pres['blocks']} "
                                      f"blocks its {pres.get('first')}-{pres.get('last')}; missing {pres['missing']}",
                   site_level=pres))
    same = {k: (env.get(k), rt.get(k)) for k in sorted(set(env) | set(rt)) if env.get(k) != rt.get(k)}
    out.append(res("R1f", not same, f"P2_distill.env KRABBY_* == RT_train.env KRABBY_* ({len(env)} keys)" if not same
                   else f"differs: {same}", differs=same))
    r5, _tail = d5(C1_RUN_DIR)
    r5 = dict(r5, id="R5")
    out.append(r5)
    out.append(dict(d7(text), id="R7"))
    blocks = title_blocks(text)
    out.append(res("R4", bool(blocks) and blocks[0][0] == 28991, f"RT_s19 first iteration {blocks[0][0] if blocks else None}"
                   f" (TDR model_28991); c1 = 28991 + 509 = 29500"))
    bad = [r for r in out if r["ok"] is False]
    rc = RC_KILL if bad else RC_PASS
    emit(nm, rc, out, a.json)
    return rc


# ============================================================================== pin / record / stop
def pin_state() -> dict:
    """The live build against the RT pin (results/pin.json) and, when written, results/p2/pin.json."""
    pins = {}
    for label, p in (("rt", RT_PIN), ("p2", P2_PIN)):
        try:
            d = json.loads(p.read_text())
            pins[label] = {"path": str(p), "code_sha": d.get("code_sha"), "eval_sha": d.get("eval_sha")}
        except OSError:
            pins[label] = None
        except ValueError as exc:
            pins[label] = {"path": str(p), "error": str(exc)}
    why = []
    try:
        cs = RL.code_state()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "why": [f"code_state failed: {exc}"], "pins": pins}
    for label, pin in pins.items():
        if pin is None:
            if label == "rt":
                why.append(f"no {RT_PIN}")
            continue
        for k in ("code_sha", "eval_sha"):
            if pin.get(k) != cs[k]:
                why.append(f"live {k} {str(cs[k])[:8]} != {label} pin {str(pin.get(k))[:8]}")
    if cs["eval_sha"] != RL.EVAL_SHA_PIN:
        why.append(f"live eval_sha {cs['eval_sha'][:8]} != {RL.EVAL_SHA_PIN[:8]}")
    if cs["dirty"]:
        why.append(f"dirty code/eval paths: {cs['dirty']}")
    return {"ok": not why, "why": why, "live": cs, "pins": pins}


def cmd_pin(a) -> int:
    st = pin_state()
    live = st.get("live") or {}
    print(f"P2 PIN {'PASS' if st['ok'] else 'REFUSE'} code_sha {str(live.get('code_sha'))[:8]} eval_sha "
          f"{str(live.get('eval_sha'))[:8]} dirty {live.get('dirty')} p2 pin "
          f"{'present' if st['pins'].get('p2') else 'not written yet'}" + ("" if st["ok"] else f": {st['why']}"))
    return 0 if st["ok"] else 5


def charter_entry() -> str | None:
    text = CHARTER.read_text()
    m = CHARTER_HEAD_RE.search(text)
    if not m:
        return None
    end = text.find("\n### ", m.end())
    body = text[m.start(): end if end > 0 else len(text)]
    return body if ("6d1f29be" in body and "model_29500" in body) else None


def cmd_record(a) -> int:
    teacher, envf = Path(a.teacher), Path(a.env_file)
    ce = charter_entry()
    rec = {"name": a.name, "kind": "distill", "campaign": "rough_terrain P2", "created": RL.now_iso(),
           "launcher_argv": a.launcher_argv, "teacher": str(teacher), "teacher_sha256": RL.sha256_file(teacher),
           "teacher_label": "c1 = RT_s19 model_29500", "c1_sha256": C1_SHA,
           "teacher_facts": {k: v for k, v in RL.checkpoint_facts(teacher).items() if k != "keys"},
           "charter_decision": ce.split("\n", 1)[0] if ce else None,
           "env_file": str(envf), "env_file_sha256": RL.sha256_file(envf), "env_file_krabby": env_map(envf),
           "env_sources": {str(p): RL.sha256_file(p) for p in (RL.ENVS / "p3b_dr.distill.env", RT_ENV)},
           "stripped": stripped_keys(envf), "seed": SEED, "iterations": ITERS, "num_envs": NUM_ENVS,
           "head": RL.git_head(), "log": str(DISTILL_OUT / f"{a.name}.log"), "out": str(DISTILL_OUT),
           "file_hashes": {str(p): RL.sha256_file(p) for p in (RL.DISTILL_SH, RL.TRAIN, RL.RSD_PY, RL.DR_KNOBS_PY,
                                                               RL.ACTIONS_PY, RL.OMNI_KNOBS_PY, OTK_PY, RTF_PY,
                                                               Path(__file__).resolve(), P2_DISTILL_SH,
                                                               Path(RDC.__file__).resolve(),
                                                               Path(RL.__file__).resolve())},
           "checks_command": (f"{RL.PY} -I -B {Path(__file__).resolve()} check --name {a.name} --teacher {teacher}")}
    st = pin_state()
    rec["pin"] = st
    live = st.get("live") or {}
    rec["code_sha"], rec["eval_sha"], rec["dirty"] = live.get("code_sha"), live.get("eval_sha"), live.get("dirty")
    path = RUNS / f"{a.name}.distill.json"
    if path.exists():
        print(f"p2_distill_check: {path} exists (a distill is never relaunched under the same name)", file=sys.stderr)
        return 5
    if not st["ok"] or rec["teacher_sha256"] != C1_SHA or ce is None:
        print(f"p2_distill_check: refusing the record: pin {st['why']}, teacher sha {rec['teacher_sha256'][:8]}, "
              f"charter entry {'found' if ce else 'MISSING'}", file=sys.stderr)
        return 5
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"recorded {path}")
    return 0


def cmd_stop(a) -> int:
    unit = a.unit
    if unit is None:
        uf = DISTILL_OUT / f"{a.name}.unit"
        unit = uf.read_text().strip() if uf.is_file() else None
    pids = lambda: RL.trainer_pids(unit, a.name)  # noqa: E731
    if a.dry:
        print(f"P2 DISTILL STOP {a.name} dry: unit {unit}, trainer pid(s) {pids()} (nothing signalled)")
        return 0
    r = RL.stop_trainer(pids, say=lambda m: print(f"P2 DISTILL STOP {a.name} {m}"))
    print(json.dumps(r, default=str))
    if r.get("sigkill"):
        print(f"P2 DISTILL STOP {a.name} sigkill=1: open the DerivedDataCache watch before the next Kit start")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--name", default=NAME)
    c.add_argument("--teacher", default=str(C1_PATH))
    c.add_argument("--log", default=None)
    c.add_argument("--run-dir", default=None)
    c.add_argument("--teacher-run-dir", default=None)
    c.add_argument("--env-file", default=str(ENV_FILE))
    c.add_argument("--reference-log", default=str(RT_S19_LOG))
    c.add_argument("--min-iters", type=int, default=10)
    c.add_argument("--live", action="store_true", help="accepted for RSD compatibility; D6 runs whenever it applies, "
                                                            "report only (never changes rc)")
    c.add_argument("--json", action="store_true")
    r = sub.add_parser("reference")
    r.add_argument("--json", action="store_true")
    sub.add_parser("pin")
    rc_ = sub.add_parser("record")
    rc_.add_argument("--name", required=True)
    rc_.add_argument("--teacher", required=True)
    rc_.add_argument("--env-file", required=True)
    rc_.add_argument("--launcher-argv", nargs=argparse.REMAINDER, default=[])
    s = sub.add_parser("stop")
    s.add_argument("--name", required=True)
    s.add_argument("--unit", default=None)
    s.add_argument("--dry", action="store_true", help="list the trainer pid(s) only; signal nothing")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "reference": cmd_reference, "pin": cmd_pin, "record": cmd_record,
            "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
