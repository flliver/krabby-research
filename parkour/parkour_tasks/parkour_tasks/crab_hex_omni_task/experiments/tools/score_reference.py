#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Score a campaign's REFERENCE runs against a NAMED OLD campaign's baseline screens. CPU ONLY.

Plan: the campaign record ``experiments/2026-09-27_stroke_cat_W/CHARTER.md`` (step 8's health line: "the
pooled F and F_mt blocks not WORSE ... scored offline by ``score_reference.py``"; its ``score_reference.py``
section); the user's plan draft is ``~/.claude/plans/per-category-stroke-weights.md`` (outside the repo). A ``--reference-chain`` run records its
head and parks: it writes no DECISION and has no baseline queue (``run_candidates.reference_loop``), so
its screens are scored here -- by the driver's OWN :func:`run_candidates.decide_pooled`, loaded BY PATH
from this checkout (nothing re-implemented), against the reused controls of an older campaign that ran
under the same ``eval_sha``. Never imports ``isaaclab``, never starts Kit, never touches a GPU, and
writes nothing but ``--json``.

WHAT IT FEEDS ``decide_pooled`` (each from the OLD campaign unless said otherwise)

* the candidate's s3 / s2 screens -- the NEW campaign's reference runs (``--reference``), or any two
  ordinary experiment headers (``--cand-xids``);
* the baseline's s3 / s2 screens -- the old campaign's headers ``--base-xids`` (also the anchor, as in
  a campaign where no adoption qualifies as an anchor yet);
* the resolution -- ``resolution.<eval_sha[:12]>.json`` (D00s), its ``eval_sha`` asserted equal to
  the old campaign's, its ``id`` set from the old ledger's D00s entry as ``load_resolution`` does;
* ``cand_spec`` -- ``--cand-spec <id>`` from the old ``candidates.yaml`` (only ``absolute_units`` and
  ``speculative_default`` are read), and the old ``decision_rule``;
* the seed terms -- :func:`run_candidates.seed_terms_for` on a minimal ctx stub (the old ledger, the
  old candidates, the HISTORICAL ``eval_sha``) over the candidate's and the baseline's chains: the old
  ``D00c-<chain>`` when it exists at that eval, else ``D00bs`` -- the element-wise MAX when the two
  chains differ (a reference chain such as ``c6y_scat`` has no D00c of its own, so it cites D00bs);
* ``floor_tripped`` -- the candidate's s2 header's ``floor_watch``, as the driver passes it.

Every header's ``eval_sha`` must equal the old campaign's (``ladder.eval_sha``): a screen under another
eval is not comparable (``--allow-eval-mismatch`` only for the extraction dry run, never a verdict).
``code_sha`` is REPORTED, never asserted: the reference runs train under the per-category knob's code,
so theirs differs from the controls' by construction (the plan's stated confound; its step-6 identity
checks are the control).

WHERE REFERENCE RUNS LIVE IN ``state.json`` (``run_candidates.record_reference`` / ``run_experiment``)

* ``ladder.references[<name>]`` -- a list, oldest first, one row per FINISHED run:
  ``{xid, seed, ckpt, stage_ckpts, recipe_id, chain, code_sha, eval_sha, floor_watch, at}``. The row
  carries NO screen.
* ``ladder.experiments[<xid>]`` (or ``ladder.squashed``) -- that run's header, ``kind: reference``,
  ``candidate: <name>``, ``seed``, ``status: ok``, ``chain``, ``floor_watch``, and ``screen`` (the
  driver's ``_screen_of``: the final stage's suite summary plus its ``trained_categories``), with the
  header's ``eval_sha`` naming the eval that produced the screen.

``--reference NAME`` picks, per seed (``--seeds``, default 3 and 2), the NEWEST row of
``ladder.references[NAME]`` (``--reference-s2 NAME2`` when the seed-2 run is its own reference name)
and reads its header's screen; the header must be ``kind: reference`` of that name and seed, ``status:
ok``, with a screen. ``--cand-xids`` reads two ordinary headers instead -- the path the self-check
below uses, since the formation campaign ran no reference.

SELF-CHECK. ``--check-entry D05`` compares the result with that ledger entry of the old campaign field
for field (pick, pooled confidence, the pooled ``why``, and every ``evidence`` key ``decide_pooled``
writes; the entry's own recorded anchor label is used and it is passed as ``prior``) and exits 1 on
any difference. Every non-float value (strings -- the ``why`` text included --, ints, bools, None, keys,
list lengths) must match exactly; a float matches when ``math.isclose(rel_tol=FLOAT_REL_TOL, abs_tol=0)``
(1e-12, i.e. a few ulp): ``decide_pooled``'s pooled statistics use the builtin ``sum()``, which Python
3.12 made compensated, so a 3.12 interpreter reproduces an entry recorded under 3.11 (the Isaac venv) to
within one ulp, not bit for bit (e.g. p_bar 0.9224999999999999 vs 0.9225). The tool still RECORDS what
the interpreter computes; only the comparison is tolerant. On the formation campaign:

    score_reference.py --old-campaign <exp>/2026-09-26_stroke_formation --base-xids X01 X02 \\
        --cand-xids X03 X04 --cand-spec S_f --check-entry D05

The per-category plan's W reading (reference S_cat vs the stroke_posture B pair):

    score_reference.py --old-campaign <exp>/2026-09-26_stroke_posture --base-xids X01 X02 \\
        --new-campaign <exp>/2026-09-27_stroke_cat_W --reference S_cat --cand-spec S [--json P]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
TOOLS = HERE.parent
RUN_CANDIDATES_PY = TOOLS / "run_candidates.py"

#: The ``decide_pooled`` evidence keys the self-check compares with a recorded entry.
EVIDENCE_KEYS = ("pooled", "per_seed", "anchored_drift", "resolution_id", "anchor", "missing", "levels",
                 "decisive", "present", "untrained", "na_blocks", "category_notes", "adopt_requires")

#: The self-check's float tolerance (module docstring, SELF-CHECK): relative, no absolute floor, so a 0.0 must
#: reproduce as exactly 0.0. 1e-12 admits the one-ulp drift of Python 3.12's compensated ``sum()`` against a
#: 3.11-recorded entry (relative ulp ~1.1e-16) with four orders of magnitude to spare, and still catches any
#: change a decision could turn on.
FLOAT_REL_TOL = 1e-12


def _load(path: Path, name: str):
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_RC = None


def rc():
    """``run_candidates.py`` of THIS checkout, loaded by path on first use."""
    global _RC
    if _RC is None:
        _RC = _load(RUN_CANDIDATES_PY, "score_reference__run_candidates")
    return _RC


# ============================================================================ campaign records
class Campaign:
    """One campaign dir's records: ``state.json``, ``ledger.json``, ``candidates.yaml``."""

    def __init__(self, path: Path | str):
        self.dir = Path(path).resolve()
        for name in ("state.json", "ledger.json", "candidates.yaml"):
            if not (self.dir / name).is_file():
                raise SystemExit(f"{self.dir}: no {name} -- not a campaign dir")
        self.state = json.loads((self.dir / "state.json").read_text())
        self.ledger = json.loads((self.dir / "ledger.json").read_text())
        self.cands = rc().load_candidates(self.dir / "candidates.yaml")

    @property
    def lad(self) -> dict:
        return self.state.get("ladder") or {}

    @property
    def eval_sha(self) -> str:
        return str(self.lad.get("eval_sha") or "")

    def header(self, xid: str) -> dict:
        h = rc().all_headers(self.state).get(xid)
        if h is None:
            raise SystemExit(f"{self.dir.name}: no experiment header {xid!r}")
        return h


def _ok_screen(camp: Campaign, h: dict, what: str) -> dict:
    if h.get("status") != "ok" or not h.get("screen"):
        raise SystemExit(f"{camp.dir.name} {h.get('xid')} ({what}): status {h.get('status')!r}, screen "
                         f"{'present' if h.get('screen') else 'absent'} -- not a finished screen")
    return h


def reference_header(camp: Campaign, name: str, seed: str) -> tuple[dict, dict]:
    """``(record row, header)`` of reference ``name``'s NEWEST finished run at ``seed`` (module
    docstring): ``ladder.references[name]`` -> its ``xid`` -> the header, which must be that
    reference's own ``kind: reference`` run with an ok screen."""
    rows = [r for r in rc().reference_records(camp.state, name) if str(r.get("seed")) == str(seed)]
    if not rows:
        have = sorted({str(r.get("seed")) for r in rc().reference_records(camp.state, name)})
        raise SystemExit(f"{camp.dir.name}: ladder.references[{name!r}] has no finished run at seed {seed} "
                         f"(seeds recorded: {have or 'none'})")
    row = rows[-1]
    h = camp.header(str(row.get("xid")))
    if h.get("kind") != "reference" or h.get("candidate") != name or str(h.get("seed")) != str(seed):
        raise SystemExit(f"{camp.dir.name} {row.get('xid')}: the header is kind {h.get('kind')!r} candidate "
                         f"{h.get('candidate')!r} seed {h.get('seed')!r}, not reference {name} s{seed}")
    return row, _ok_screen(camp, h, f"reference {name} s{seed}")


def _describe(camp: Campaign, h: dict, role: str, row: dict | None = None) -> dict:
    scr = h.get("screen") or {}
    return {"role": role, "campaign": str(camp.dir), "xid": h.get("xid"), "kind": h.get("kind"),
            "candidate": h.get("candidate"), "seed": str(h.get("seed")), "chain": h.get("chain"),
            "recipe_id": h.get("recipe_id"), "eval_sha": h.get("eval_sha"), "code_sha": h.get("code_sha"),
            "ckpt": h.get("ckpt"), "floor_watch": list(h.get("floor_watch") or []),
            "floor_park": list(h.get("floor_park") or []), "screen_stage": scr.get("stage"),
            "trained_decisive": sorted(((scr.get("trained_categories") or {}).get("decisive")) or []),
            **({"reference_row_at": row.get("at")} if row else {})}


# ============================================================================ scoring
def score(old: Campaign, base_hs: tuple[dict, dict], cand_hs: tuple[dict, dict], *, cand_spec_id: str,
          prior: dict | None = None, anchor_label: str | None = None) -> dict:
    """``decide_pooled`` on (cand s3, cand s2) vs (base s3, base s2) under the OLD campaign's rule,
    resolution and seed terms (module docstring)."""
    R = rc()
    cand_spec = R.candidate_by_id(old.cands, cand_spec_id)
    if cand_spec is None:
        raise SystemExit(f"--cand-spec {cand_spec_id!r} is not a candidate of {old.dir.name}/candidates.yaml "
                         f"({[c.get('id') for c in old.cands.get('candidates') or []]})")
    rule = old.cands.get("decision_rule") or {}
    eval_sha = old.eval_sha
    res_path = old.dir / f"resolution.{eval_sha[:12]}.json"
    if not res_path.is_file():
        raise SystemExit(f"{res_path}: the old campaign's suite resolution (D00s) is not on disk")
    resolution = json.loads(res_path.read_text())
    if str(resolution.get("eval_sha") or "") != eval_sha:
        raise SystemExit(f"{res_path.name} records eval_sha {resolution.get('eval_sha')}, not the campaign's "
                         f"{eval_sha}")
    ctx = SimpleNamespace(ledger=old.ledger, cands=old.cands, eval_sha=eval_sha)
    entry = R.current_resolution_entry(ctx, "D00s")
    if entry is not None:
        resolution["id"] = entry["id"]
    (b3, b2), (c3, c2) = base_hs, cand_hs
    terms, seed_ids = R.seed_terms_for(ctx, c3.get("chain"), b3.get("chain"))
    if anchor_label is None:
        anchor_label = (f"{old.dir.name} baseline {b3.get('candidate')} {b3.get('xid')}/{b2.get('xid')} "
                        "(reference scoring: no adoption, the reused controls are the anchor)")
    pooled = R.decide_pooled(
        c3["screen"], c2["screen"], b3["screen"], b2["screen"], resolution,
        cand_spec=cand_spec, rule=rule, prior=prior, seed_terms=terms,
        anchor_s3=b3["screen"], anchor_s2=b2["screen"], anchor_label=anchor_label,
        resolution_id=", ".join([R.resolution_label(resolution)] + seed_ids),
        floor_tripped=bool(c2.get("floor_watch")))
    cm = pooled["evidence"]["pooled"]
    worse = R._any_worse(cm)
    out = {"chosen": pooled["chosen"], "pooled_confidence": pooled["pooled_confidence"], "why": pooled["why"],
           "worse": worse, "resolution_file": str(res_path), "seed_term_ids": seed_ids,
           "cand_spec": cand_spec_id, "eval_sha": eval_sha, "evidence": pooled["evidence"]}
    if prior is not None:
        out.update(status=pooled["status"], self_overrule=pooled["self_overrule"], prior=pooled["prior"])
    return out


def _same(a, b, path: str = "") -> str | None:
    """``None`` when ``a`` and ``b`` (JSON-normalised) agree, else the first differing path. Floats compare with
    :data:`FLOAT_REL_TOL` (two NaNs agree); every other value, every key set and every list length exactly."""
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool) \
            and (isinstance(a, float) or isinstance(b, float)):
        if (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=FLOAT_REL_TOL, abs_tol=0.0):
            return None
        return path or "."
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path or '.'} (keys)"
        for k in a:
            d = _same(a[k], b[k], f"{path}.{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path or '.'} (length)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _same(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    return None if (type(a) is type(b) or {type(a), type(b)} <= {int, float}) and a == b else (path or ".")


def check_entry(result: dict, entry: dict) -> list[str]:
    """Field-for-field differences between :func:`score`'s result and a recorded ledger entry: exact on every
    non-float value, floats within :data:`FLOAT_REL_TOL` (module docstring, SELF-CHECK -- interpreter-portable)."""
    diffs = []
    norm = lambda x: json.loads(json.dumps(x, sort_keys=True, default=str))  # noqa: E731
    for mine, theirs in (("chosen", "chosen"), ("pooled_confidence", "pooled_confidence"), ("why", "why_pooled"),
                         ("status", "status")):
        if _same(norm(result.get(mine)), norm(entry.get(theirs))) is not None:
            diffs.append(f"{theirs}: recorded {entry.get(theirs)!r}, reproduced {result.get(mine)!r}")
    rec = entry.get("evidence") or {}
    ev = result["evidence"]
    for key in EVIDENCE_KEYS:
        if key not in ev and key not in rec:
            continue
        where = _same(norm(ev.get(key)), norm(rec.get(key)), key)
        if where is not None:
            diffs.append(f"evidence.{key} differs (at {where})")
    return diffs


# ============================================================================ printing
def _f(x, fmt="{:+.4f}") -> str:
    return "--" if x is None else fmt.format(x)


def render(result: dict, runs: list[dict]) -> str:
    ev = result["evidence"]
    cm = ev["pooled"]
    L = ["== score_reference: driver decide_pooled on reference screens vs reused controls", "",
         "runs:"]
    for r in runs:
        L.append(f"  {r['role']:8s} {Path(r['campaign']).name} {r['xid']} ({r['kind']} {r['candidate']} "
                 f"s{r['seed']}, chain {r['chain']}) eval_sha {str(r['eval_sha'])[:12]} code_sha "
                 f"{str(r['code_sha'])[:12]} floor_watch {r['floor_watch'] or 'none'}")
    L += ["", f"resolution {Path(result['resolution_file']).name} ({ev.get('resolution_id')}); cand_spec "
              f"{result['cand_spec']}; eval_sha {result['eval_sha'][:12]}", "",
          f"PICK {result['chosen']} ({result['pooled_confidence']}): {result['why']}",
          f"WORSE (pooled): {result['worse'] or 'none'}", "",
          "pooled blocks (z / R, class):"]
    for k, v in cm["blocks"].items():
        L.append(f"  {k:6s} z {_f(v.get('z'))}  R {_f(v.get('R'), '{:.4f}')}  {v.get('class')}"
                 + (f"  missing {v['missing']}" if v.get("missing") else ""))
        for row in v.get("rows") or []:
            L.append(f"      {row['metric']}: delta {_f(row.get('delta'))} (s3 {_f(row.get('delta_s3'))}, s2 "
                     f"{_f(row.get('delta_s2'))}) z {_f(row.get('z'), '{:+.3f}')}")
    g = cm["guards"]
    worse_g = set(g.get("worse") or [])
    L += ["", f"guards ({len(g.get('rows') or [])}, level {g.get('level')}; WORSE {sorted(worse_g) or 'none'}; "
              f"MISSING {g.get('missing') or 'none'}):"]
    for row in g.get("rows") or []:
        mark = "WORSE" if row["metric"] in worse_g else ("MISSING" if row.get("missing") else "ok")
        L.append(f"  {mark:7s} {row['metric']}: delta {_f(row.get('delta'))} z {_f(row.get('z'), '{:+.3f}')}")
    ps = ev["per_seed"]
    L += ["", "per seed (sign of sum z_b/R_b): " + ", ".join(
        f"{s} {ps[s]['score']:+.3f} (" + ", ".join(f"{k} {b['class']}" for k, b in ps[s]["blocks"].items()) + ")"
        for s in ("s3", "s2"))]
    L.append(f"anchored drift: {ev['anchored_drift'].get('verdict')} ({ev['anchored_drift'].get('why')})")
    return "\n".join(L)


# ============================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-campaign", required=True, help="the campaign whose baseline screens, rule, "
                    "resolution, seed terms and eval_sha are reused")
    ap.add_argument("--base-xids", nargs=2, required=True, metavar=("XS3", "XS2"),
                    help="the old campaign's baseline headers at seeds 3 and 2")
    ap.add_argument("--new-campaign", default=None,
                    help="the campaign holding the candidate runs (default: the old one)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--reference", default=None, metavar="NAME",
                     help="the new campaign's ladder.references[NAME] runs (module docstring)")
    who.add_argument("--cand-xids", nargs=2, default=None, metavar=("XS3", "XS2"),
                     help="two ordinary experiment headers instead (seeds 3 and 2)")
    ap.add_argument("--reference-s2", default=None, metavar="NAME",
                    help="with --reference: the seed-2 run's own reference name (default: NAME)")
    ap.add_argument("--seeds", nargs=2, default=("3", "2"), metavar=("S3", "S2"),
                    help="with --reference: the two seeds (default 3 2)")
    ap.add_argument("--cand-spec", required=False, default=None,
                    help="the old candidates.yaml candidate whose spec scores the change (required to score)")
    ap.add_argument("--check-entry", default=None, metavar="ID",
                    help="self-check: reproduce this old-ledger entry field for field (exit 1 on a difference)")
    ap.add_argument("--list-only", action="store_true",
                    help="print the four screens found (xid, seed, eval_sha, trained categories) and stop")
    ap.add_argument("--allow-eval-mismatch", action="store_true",
                    help="with --list-only: do not refuse a header under another eval_sha")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    old = Campaign(args.old_campaign)
    new = Campaign(args.new_campaign) if args.new_campaign else old
    b3 = _ok_screen(old, old.header(args.base_xids[0]), "baseline s3")
    b2 = _ok_screen(old, old.header(args.base_xids[1]), "baseline s2")
    runs = [_describe(old, b3, "base s3"), _describe(old, b2, "base s2")]
    if args.reference:
        names = (args.reference, args.reference_s2 or args.reference)
        (r3, c3), (r2, c2) = (reference_header(new, n, s) for n, s in zip(names, args.seeds))
        runs += [_describe(new, c3, "cand s3", r3), _describe(new, c2, "cand s2", r2)]
    else:
        c3 = _ok_screen(new, new.header(args.cand_xids[0]), "candidate s3")
        c2 = _ok_screen(new, new.header(args.cand_xids[1]), "candidate s2")
        runs += [_describe(new, c3, "cand s3"), _describe(new, c2, "cand s2")]

    bad = [r for r in runs if str(r["eval_sha"] or "") != old.eval_sha]
    if args.list_only:
        print(json.dumps({"old_eval_sha": old.eval_sha, "runs": runs}, indent=2))
        if bad:
            print(f"[score_reference] eval_sha differs from {old.dir.name}'s {old.eval_sha[:12]} on "
                  f"{[r['xid'] for r in bad]}: NOT comparable" + (" (allowed: --list-only only)"
                                                                     if args.allow_eval_mismatch else ""))
            return 0 if args.allow_eval_mismatch else 2
        return 0
    if bad:
        raise SystemExit(f"eval_sha differs from {old.dir.name}'s {old.eval_sha[:12]} on "
                         + ", ".join(f"{r['role']} {r['xid']} ({str(r['eval_sha'])[:12]})" for r in bad)
                         + ": the screens are not comparable; the old baseline would need re-evaluating")
    if not args.cand_spec:
        raise SystemExit("--cand-spec is required to score (the old candidates.yaml id whose spec applies)")
    prior = anchor = None
    if args.check_entry:
        prior = rc().entry_by_id(old.ledger, args.check_entry)
        if prior is None:
            raise SystemExit(f"--check-entry {args.check_entry}: no such entry in {old.dir.name}/ledger.json")
        anchor = (prior.get("evidence") or {}).get("anchor")
    result = score(old, (b3, b2), (c3, c2), cand_spec_id=args.cand_spec, prior=prior, anchor_label=anchor)
    result["runs"] = runs
    print(render(result, runs))
    rc_ = 0
    if args.check_entry:
        diffs = check_entry(result, prior)
        if diffs:
            print(f"[self-check] {args.check_entry}: {len(diffs)} DIFFERENCE(S)")
            for d in diffs:
                print(f"  - {d}")
            rc_ = 1
        else:
            print(f"[self-check] {args.check_entry}: reproduced field for field (chosen, pooled_confidence, "
                  f"why_pooled, status, evidence {[k for k in EVIDENCE_KEYS if k in result['evidence']]})")
        result["self_check"] = {"entry": args.check_entry, "diffs": diffs}
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    return rc_


if __name__ == "__main__":
    sys.exit(main())
