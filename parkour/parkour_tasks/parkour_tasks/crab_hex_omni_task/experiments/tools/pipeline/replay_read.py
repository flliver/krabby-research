#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 replay reader (CPU): R18's "process reproduced" verdict and the X stand-replay rule.

PROCESS REPRODUCED (PLAN step 9; DESIGN 5 "R18 parity verdict"; the P-gates are NOT part of a gate pass):
* P-config  each stage's trainer-written ``params/{env,agent}.yaml`` against the lineage run dir of the same stage.
            Allowed differences (CHARTER 2026-10-16, step 2, pinned): keys present only in the new run whose value is
            the default-off value; filesystem paths, run names, timestamps; ``max_iterations``, ``resume``,
            ``load_run``, ``load_checkpoint``; the seed-derived run label. Anything else FAILS.
* P-early   s1's tfevents against X43's at steps 0-19, tag by tag. Timing tags (``Perf/*`` and the ``*/time`` tags,
            logged with wall-clock seconds as their step) are ignored; ``Loss/priv_reg`` is tolerated at |d| <= 2e-6;
            ``*hist_latent*`` at <= 4 ULP WHEN every other non-timing tag is bitwise or within its pinned tolerance
            (the CHARTER STOP-0 amendment, 2026-10-16: under it the Phase-A read -- priv_reg 1.4e-6, hist_latent
            2 ULP, everything else bitwise -- is reproduced). Everything else bitwise. The bitwise count out of the
            shared tags is reported against DESIGN's 112/116.
* P-live    the same rule on R18 s1's first 200 iterations (+200), read live (a fail is STOP-R).
* P-onset   reported: the first divergent iteration per tag over the whole s1 run (expected near 1905).
X (PLAN Stand): decisive ONLY on bitwise equality -- the stand head's file sha256 equals the recorded head's AND its
``train_log.jsonl`` loss curve is bitwise -- under the step-8-decided rule (``x_rule: bitwise``); otherwise X is
report-only against the K lines and never feeds STOP-R or holds F19.

    replay_read.py pconfig --new RUN_DIR --old RUN_DIR
    replay_read.py pearly --new RUN_DIR --old RUN_DIR [--steps 0:20]
    replay_read.py onset --new RUN_DIR --old RUN_DIR
    replay_read.py x --new RUN_DIR --ref-sha SHA [--ref-log DIR/train_log.jsonl] --rule bitwise|report_only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

ALLOWED_KEY = re.compile(r"(^|\.)(max_iterations|resume|load_run|load_checkpoint|run_name|seed|log_dir|"
                         r"experiment_name)$")
PATHLIKE = re.compile(r"(^/|\.pt$|\.usda?$|\.npz$|\.json$|\.ya?ml$|\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)")
PRIV_REG_TOL = 2e-6
HIST_LATENT_ULP = 4
DESIGN_BITWISE = (112, 116)


def _loader():
    import yaml  # noqa: PLC0415

    class AnyLoader(yaml.SafeLoader):
        pass

    def _any(loader, _suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    AnyLoader.add_multi_constructor("tag:yaml.org,2002:python/", _any)
    AnyLoader.add_multi_constructor("!", _any)
    return AnyLoader


def flat(d, pre="") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d):
            out[pre] = tuple(d)
        else:
            for i, v in enumerate(d):
                out.update(flat(v, f"{pre}[{i}]"))
    else:
        out[pre] = d
    return out


def _default_off(v) -> bool:
    return v in (None, False, 0, 0.0, "", "0", "off", "none", (), [])


def pconfig_texts(old_txt: dict, new_txt: dict) -> dict:
    """P-config on ``{"env.yaml": text, "agent.yaml": text}`` pairs (pure)."""
    import yaml  # noqa: PLC0415

    res = {"files": {}}
    ok = True
    for f in ("env.yaml", "agent.yaml"):
        a = flat(yaml.load(old_txt[f], Loader=_loader()))
        b = flat(yaml.load(new_txt[f], Loader=_loader()))
        bad, allowed = [], []
        for k in sorted(set(a) | set(b)):
            va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
            if va == vb:
                continue
            why = None
            if ALLOWED_KEY.search(k):
                why = "allowed key"
            elif va == "<absent>" and _default_off(vb):
                why = "new key, default-off"
            elif all(isinstance(x, str) and PATHLIKE.search(x) for x in (va, vb) if x != "<absent>"):
                why = "path / run name / timestamp"
            (allowed if why else bad).append({"key": k, "old": repr(va)[:160], "new": repr(vb)[:160],
                                              **({"why": why} if why else {})})
        res["files"][f] = {"n_keys_old": len(a), "n_keys_new": len(b), "allowed": allowed, "not_allowed": bad}
        ok = ok and not bad
    res["pass"] = ok
    return res


def pconfig(new: Path, old: Path) -> dict:
    txt = lambda d: {f: (Path(d) / "params" / f).read_text() for f in ("env.yaml", "agent.yaml")}  # noqa: E731
    return {"new": str(new), "old": str(old), **pconfig_texts(txt(old), txt(new))}


def scalars(run: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)} for t in ea.Tags()["scalars"]}


def _timing(tag: str) -> bool:
    return tag.startswith("Perf/") or tag.endswith("/time")


def pearly_series(old: dict, new: dict, steps) -> dict:
    """P-early on two ``{tag: {step: value}}`` dicts over ``steps`` (pure; the amended rule in the docstring)."""
    shared = sorted(set(old) & set(new))
    res = {"tags_old": len(old), "tags_new": len(new), "shared": len(shared), "only_old": sorted(set(old) - set(new)),
           "only_new": sorted(set(new) - set(old)), "bitwise": 0, "differ": [], "timing_ignored": []}
    hist = []
    for t in shared:
        st = [s for s in steps if s in old[t] and s in new[t]]
        if not st:
            continue
        if _timing(t):
            res["timing_ignored"].append(t)
            continue
        va = np.array([old[t][s] for s in st], dtype=np.float32)
        vb = np.array([new[t][s] for s in st], dtype=np.float32)
        if np.array_equal(va, vb, equal_nan=True):
            res["bitwise"] += 1
            continue
        d = float(np.nanmax(np.abs(va.astype(np.float64) - vb.astype(np.float64))))
        ulp = int(np.nanmax(np.abs(va.view(np.int32).astype(np.int64) - vb.view(np.int32).astype(np.int64))))
        rec = {"tag": t, "max_abs": d, "max_ulp": ulp, "first_step": int(st[int(np.argmax(va != vb))])}
        if t == "Loss/priv_reg":
            rec["tolerated"] = d <= PRIV_REG_TOL
            rec["rule"] = f"|d| <= {PRIV_REG_TOL:g}"
        elif "hist_latent" in t:
            rec["tolerated"] = None            # decided below: needs every other non-timing tag bitwise / tolerated
            rec["rule"] = f"<= {HIST_LATENT_ULP} ULP when every other non-timing tag is bitwise (STOP-0 amendment)"
            hist.append(rec)
        else:
            rec["tolerated"] = False
            rec["rule"] = "bitwise"
        res["differ"].append(rec)
    others_ok = all(r["tolerated"] for r in res["differ"] if r not in hist)
    for r in hist:
        r["tolerated"] = bool(others_ok and r["max_ulp"] <= HIST_LATENT_ULP)
    res["pass"] = all(r["tolerated"] for r in res["differ"])
    res["design_bitwise"] = f"{res['bitwise']}/{res['shared'] - len(res['timing_ignored'])} bitwise (DESIGN " \
                            f"{DESIGN_BITWISE[0]}/{DESIGN_BITWISE[1]})"
    return res


def pearly(new: Path, old: Path, steps=range(20)) -> dict:
    return {"new": str(new), "old": str(old), **pearly_series(scalars(old), scalars(new), list(steps))}


def onset(old: dict, new: dict) -> dict:
    """The first divergent iteration per shared non-timing tag (reported; expected near the priv_reg onset 1905)."""
    out = {}
    for t in sorted(set(old) & set(new)):
        if _timing(t):
            continue
        for s in sorted(set(old[t]) & set(new[t])):
            if np.float32(old[t][s]) != np.float32(new[t][s]):
                out[t] = int(s)
                break
    first = min(out.values()) if out else None
    return {"first_divergence": first, "by_tag": out}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_TIMING_KEY = re.compile(r"(time|fps|wall|_s_per|^t_[a-z_]+_s$)")


def loss_curve(train_log: Path) -> list:
    """The ``train_log.jsonl`` rows of a DAgger run, minus its wall-clock fields (``t_rollout_s``, ``t_train_s``, ...)."""
    rows = []
    for ln in Path(train_log).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows.append({k: v for k, v in r.items() if not _TIMING_KEY.search(k)})
    return rows


def x_read(new_ckpt: Path, ref_sha: str, *, rule: str, new_log: Path | None = None,
           ref_log: Path | None = None) -> dict:
    """PLAN Stand "X": decisive only on bitwise (file sha AND loss curve) under ``rule == 'bitwise'``. When either
    ``train_log.jsonl`` is genuinely unavailable the file sha decides alone, and the record says the curves were not
    compared (``loss_curve_equal`` None, ``curves_compared`` False)."""
    if rule not in ("bitwise", "report_only"):
        raise ValueError(f"x rule {rule!r} is not decided (PLAN step 8 determinism smoke: bitwise | report_only)")
    sha = sha256_file(new_ckpt)
    curve_eq = None
    compared = bool(new_log is not None and ref_log is not None and Path(new_log).is_file()
                    and Path(ref_log).is_file())
    if compared:
        curve_eq = loss_curve(new_log) == loss_curve(ref_log)
    bitwise = (sha == ref_sha) and (curve_eq if compared else True)
    return {"rule": rule, "sha256": sha, "ref_sha256": ref_sha, "sha_equal": sha == ref_sha, "loss_curve_equal":
            curve_eq, "curves_compared": compared,
            **({} if compared else {"curves_note": f"loss curves NOT compared (log unavailable: new {new_log}, "
                                                    f"ref {ref_log}); the file sha decided alone"}),
            "bitwise": bitwise,
            "verdict": ("REPRODUCED-BITWISE" if bitwise else ("NOT BITWISE" if rule == "bitwise" else "REPORT-ONLY")),
            "decisive": rule == "bitwise", "feeds_stop_r": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("pconfig", "pearly", "onset"):
        p = sub.add_parser(n)
        p.add_argument("--new", required=True, type=Path)
        p.add_argument("--old", required=True, type=Path)
        if n == "pearly":
            p.add_argument("--steps", default="0:20")
    x = sub.add_parser("x")
    x.add_argument("--new", required=True, type=Path)
    x.add_argument("--ref-sha", required=True)
    x.add_argument("--rule", required=True)
    x.add_argument("--new-log", type=Path, default=None)
    x.add_argument("--ref-log", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "pconfig":
        res = pconfig(a.new, a.old)
    elif a.cmd == "pearly":
        lo, hi = (int(v) for v in a.steps.split(":"))
        res = pearly(a.new, a.old, range(lo, hi))
    elif a.cmd == "onset":
        res = onset(scalars(a.old), scalars(a.new))
    else:
        res = x_read(a.new, a.ref_sha, rule=a.rule, new_log=a.new_log, ref_log=a.ref_log)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res.get("pass", True) else 3


if __name__ == "__main__":
    sys.exit(main())
