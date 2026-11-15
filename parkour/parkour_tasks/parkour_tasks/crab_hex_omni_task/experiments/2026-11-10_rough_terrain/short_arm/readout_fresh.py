#!/usr/bin/env python3
"""Amendment A (short arm) FRESH-SEED read-out: ARM step 2 part B, run at steps 3 / 8 / 10 (CPU; saved stamps only).

    readout_fresh.py --identity-check [--out JSON] [--scratch DIR]   step 3: identity map reproduces step 9's gates.json
    readout_fresh.py --g1   [--out JSON]                              step 8: G + the suite-only tracking lines, es5-6
    readout_fresh.py --full [--out JSON]                              step 10: readout.gates() unchanged, real es5-8

THE ONE CHANGE (ARM "Readout"): ``FreshRows`` subclasses ``readout.Rows`` and overrides ONLY ``found()``. A logical
env seed (what ``readout.gates()`` and its helpers ask for: the thresholds' gate_env_seeds, rows_rt env_seeds,
zero_kg_inputs' suite (1, 2) / side22 (1), rt_spec.ZERO_KG_FIELD) is mapped by ``EMAP = {1: 5, 2: 6, 3: 7, 4: 8}``
before ``readout.find`` keys the stamp index; every lookup goes through that one point (audit 1, probe P5: 104/104).
``readout`` / ``rt_spec`` are IMPORTED from ``../scripts`` (never copied: ``gates()`` hashes ``scripts/readout.py``).

FAIL-CLOSED (a slipped seed is silent, so every guard refuses loudly, non-zero exit, nothing written):
* ``found()`` raises on a logical es outside the map, on a non-injective map, on a mapped (real) es outside the mode's
  range (5-8 fresh; 1-4 identity), and on a served stamp whose run_meta ``env_overrides.seed`` or ``tp_row.env_seed``
  is not the real es (``readout.find`` accepts a stamp with no tp_row env_seed; this does not);
* --g1 / --full, BEFORE ``Rows`` is constructed (``Rows.__init__`` scans every stamp's metadata, RTF included):
  ``short_arm``'s ``results/short_arm/selection.json`` is committed and clean (its pick sha256 is the new policy, no
  ``--new``); thresholds_rough_v1.json sha256 == ce45e01b...; the frozen scripts' sha256s == the thresholds file's
  record (readout.gates()'s own comparison); the R-tracking inputs unchanged since c2054a8c, committed or in the
  working tree (``git diff --quiet c2054a8c -- P`` and ``git status --porcelain -- P`` empty; each P exists; the v1.2
  thresholds 969b2988 and the resolution file 77a2b062 by sha256 too);
* after the read: every served real es in 5-8, every served OK stamp's build (code_state code_sha / eval_sha) ==
  results/pin.json (readout already refuses a dirty stamp and two builds in one pair, not a build != pin), every TDR
  stamp served from an ``RTF_03c3e472_*`` row.
* --identity-check is the ONLY identity-map mode: Rows on a scratch symlink root of every lr name dir EXCEPT
  ``RTF_*`` / ``RTA_*`` / ``RTC_*`` and any name dir holding an env seed >= 5; PASS = every numeric leaf of step 9's
  gates.json within 1e-12 and every other leaf equal (the scratch root prefix mapped back to the lr dir), same
  verdict.

NAMING NOTE: the ARM text calls the confirm slots ``RTA_*``; rt_short.py (step 2 part A) names the SELECTION rows
``RTA_*`` (es1-2) and the confirm rows ``RTC_*`` (es5+). select.py's masked root therefore excludes ``RTF_*`` by name
(never listed) and the confirm slots by what they are (a name dir with a ``seedNNN`` dir at NNN >= 5, names only),
so it stays right under either naming.

OUTPUT LABELS: ``readout.gates()`` and ``zero_kg_check`` keep LOGICAL labels (``suite es1``: v1.2's drift dict is keyed
that way); the stamp records carry the REAL es. Every output adds ``real_seed`` beside each logical label, a
``seed_map``, the served-stamp list (rid, logical es, real es, sha256, stamp) and every guard result. The 6000-it drift
re-read is labelled "not applicable (report only)" (the arm is 510 / 1010 updates past TDR, not 6000).

Sentinels: ``RT_FRESH IDENTITY PASS|FAIL ...``, ``RT_FRESH G1 G=... SUITE=...``, ``RT_FRESH FULL verdict=...``,
``RT_FRESH REFUSED ...``.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/short_arm
CAMP = HERE.parent
SCRIPTS = CAMP / "scripts"
sys.path.insert(0, str(SCRIPTS))
import readout  # noqa: E402  (scripts/readout.py, imported: gates() self-checks scripts/readout.py's sha256)
import rt_spec as S  # noqa: E402

assert Path(readout.__file__).resolve().parent == SCRIPTS, f"readout imported from {readout.__file__}, not {SCRIPTS}"
assert Path(S.__file__).resolve().parent == SCRIPTS, f"rt_spec imported from {S.__file__}, not {SCRIPTS}"

# ============================================================================== pre-registered constants (ARM)
EMAP_FRESH = {1: 5, 2: 6, 3: 7, 4: 8}
REAL_FRESH = (5, 6, 7, 8)
EMAP_IDENTITY = {1: 1, 2: 2, 3: 3, 4: 4}
REAL_IDENTITY = (1, 2, 3, 4)
for _m in (EMAP_FRESH, EMAP_IDENTITY):
    assert len(set(_m.values())) == len(_m), f"seed map {_m} is not injective"
#: name dirs never linked into a masked root, NOT EVEN LISTED: TDR's fresh rows (RTF, es5-8, pre-registered unread).
#: The arm's own rows: rt_short.py names the selection rows RTA_* (es1-2) and the confirm rows RTC_* (es5+); the ARM
#: text's "RTA_* confirm slots" predates that naming. So the confirm slots are excluded by WHAT THEY ARE, not by name:
#: a name dir with any ``<scenario>/seedNNN`` dir at NNN >= FRESH_SEED_MIN (directory names only, nothing opened).
EXCLUDED_PREFIXES = ("RTF_",)
IDENTITY_EXCLUDED_PREFIXES = ("RTF_", "RTA_", "RTC_")     # --identity-check: step 9's rows only need RT6 / RT9
FRESH_SEED_MIN = 5
TDR_FRESH_PREFIX = "RTF_03c3e472_"
THRESHOLDS_SHA = "ce45e01b370d9530ba7fe8867e72a81b112052697a8ea97d5a86235d3b1a5254"
R_INPUTS_SINCE = "c2054a8c"                               # thresholds_rough_v1 frozen (step 7 STOP 0)
#: the R-tracking inputs (relative to crab_hex_omni_task/experiments): no change since c2054a8c (ARM fail-closed list)
R_INPUT_PATHS = ("tools/pipeline",
                 "2026-10-16_task4_pipeline/thresholds_v1_2.json",
                 "2026-10-16_task4_pipeline/threshold_inputs.yaml",
                 "2026-10-16_task4_pipeline/pipeline.yaml",
                 "2026-09-23_clock_deprecation/resolution.36e5a8af7000.json",
                 "2026-10-01_ppo_construction/scripts/side22_classify.py",
                 "2026-11-10_rough_terrain/manifests")
#: by sha256 as well (ARM: v1.2 969b2988, resolution 77a2b062)
R_INPUT_SHAS = {
    "2026-10-16_task4_pipeline/thresholds_v1_2.json":
        "969b2988ca288529b49fe4b49aa1437c5b8e5be1e91b0a239e70469b4ff781e2",
    "2026-09-23_clock_deprecation/resolution.36e5a8af7000.json":
        "77a2b062ef362adc22a444e3b256b32ad6c6ca2551ae18322dc87c512ca58e99",
}
#: the 11 decisive suite statistics per env seed (v1.2, track_err; step 9's gates.json R.tracking.suite_lines)
SUITE_STATS = ("track_err fwd", "track_err side", "track_err back", "track_err diag", "ratio move_turn",
               "survival fwd", "survival side", "survival back", "survival diag", "facing_closed turn",
               "yaw_progress turn")
SUITE_LOGICAL_ES = (1, 2)
RESULTS = CAMP / "results"
OUT_DIR = RESULTS / "short_arm"
SELECTION = OUT_DIR / "selection.json"
SELECTION_SCHEMA = "rt_short_selection_v1"
STEP9_GATES = RESULTS / "step9" / "gates.json"
PIN = RESULTS / "pin.json"
DRIFT_NA = "not applicable (report only)"
DRIFT_NA_WHY = ("readout.gates() re-reads a failing R line at v1.2's drift for 6000 its past TDR (step 9's run); the "
                "short arm is 510 / 1010 updates past TDR, so that re-read does not describe it (ARM Readout)")


class FreshRefused(Exception):
    """A fail-closed guard refused the read (non-zero exit, nothing written)."""


class FreshSeedError(Exception):
    """A seed lookup outside the pre-registered map, or a served stamp at the wrong env seed."""


# ============================================================================== the one override
class FreshRows(readout.Rows):
    """``readout.Rows`` whose ``found()`` maps the logical env seed through ``EMAP`` (and nothing else changes)."""

    EMAP = EMAP_FRESH
    REAL_OK = REAL_FRESH

    def found(self, rid: str, es: int, sha: str) -> dict:
        if isinstance(es, bool) or int(es) != es:
            raise FreshSeedError(f"{rid}: env seed {es!r} is not an integer")
        les = int(es)
        if len(set(self.EMAP.values())) != len(self.EMAP):
            raise FreshSeedError(f"seed map {self.EMAP} is not injective")
        if les not in self.EMAP:
            raise FreshSeedError(f"{rid} {str(sha)[:8]}: logical es{les} outside the map {sorted(self.EMAP)}")
        real = int(self.EMAP[les])
        if real not in self.REAL_OK:
            raise FreshSeedError(f"{rid} {str(sha)[:8]}: logical es{les} maps to es{real}, outside {self.REAL_OK}")
        f = super().found(rid, real, sha)
        if int(f["es"]) != real:
            raise FreshSeedError(f"{rid}: readout.find answered es{f['es']} for es{real}")
        rec = {"rid": rid, "logical_es": les, "real_es": real, "sha256": sha, "status": f["status"],
               "stamp": None, "stamp_as_read": f.get("stamp"), "name_dir": None, "code_sha": f.get("code_sha"),
               "eval_sha": f.get("eval_sha"), "run_meta_seed": None, "tp_row_env_seed": None}
        if f.get("stamp"):
            st = Path(f["stamp"])
            real_st = Path(os.path.realpath(st))
            meta = json.loads((st / "run_meta.json").read_text())
            tp = json.loads((st / "tp_row.json").read_text())
            rm_seed = (meta.get("env_overrides") or {}).get("seed")
            rec.update(stamp=str(real_st), name_dir=real_st.parents[2].name, run_meta_seed=rm_seed,
                       tp_row_env_seed=tp.get("env_seed"))
            if rm_seed is None or int(rm_seed) != real:
                raise FreshSeedError(f"{rid} {st}: run_meta env_overrides.seed {rm_seed!r} != es{real}")
            if tp.get("env_seed") is None or int(tp["env_seed"]) != real:
                raise FreshSeedError(f"{rid} {st}: tp_row env_seed {tp.get('env_seed')!r} != es{real} (a stamp "
                                     "with no tp_row env_seed is refused here)")
            if meta.get("checkpoint_sha256") != sha:
                raise FreshSeedError(f"{rid} {st}: checkpoint_sha256 {str(meta.get('checkpoint_sha256'))[:8]} != "
                                     f"{str(sha)[:8]}")
        if not hasattr(self, "served"):
            self.served = []
        self.served.append(rec)
        return f


class IdentityRows(FreshRows):
    """--identity-check only (step 3): the identity map on es1-4."""

    EMAP = EMAP_IDENTITY
    REAL_OK = REAL_IDENTITY


# ============================================================================== small helpers
def sha256(p: Path) -> str:
    return readout.sha256_file(Path(p))


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(S.REPO), capture_output=True, text=True,
                          env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, timeout=300)


def repo_rel(p: Path) -> str:
    return Path(p).resolve().relative_to(Path(S.REPO).resolve()).as_posix()


def fresh_seed_name_dirs(lr: Path | None = None, exclude=EXCLUDED_PREFIXES) -> list:
    """Name dirs (not ``exclude``d; those are never listed) holding any ``<scenario>/seedNNN`` dir with NNN >=
    FRESH_SEED_MIN: the confirm slots, whatever their prefix. Directory names only; no file is opened."""
    out = []
    for p in sorted(Path(lr or readout.RT_LR).iterdir()):
        if not p.is_dir() or p.name.startswith(exclude):
            continue
        for sd in p.iterdir():
            if sd.is_dir() and any(e.is_dir() and re.fullmatch(r"seed\d+", e.name) and int(e.name[4:]) >= FRESH_SEED_MIN
                                   for e in sd.iterdir()):
                out.append(p.name)
                break
    return out


def masked_root(base: Path | None, exclude=EXCLUDED_PREFIXES) -> tuple:
    """A scratch lr root: one symlink per lr NAME dir except ``exclude`` prefixes (never listed) and the name dirs
    holding env seeds >= FRESH_SEED_MIN (``fresh_seed_name_dirs``). Returns (TemporaryDirectory, root path, [linked
    names], [excluded names])."""
    fresh = set(fresh_seed_name_dirs(readout.RT_LR, exclude))
    td = tempfile.TemporaryDirectory(prefix="rt_short_lr_", dir=str(base) if base else None)
    root = Path(td.name) / "lr"
    root.mkdir()
    linked, skipped = [], []
    for p in sorted(readout.RT_LR.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith(exclude) or p.name in fresh:
            skipped.append(p.name)
            continue
        os.symlink(p, root / p.name)
        linked.append(p.name)
    return td, root, linked, skipped


def subst_prefix(x, old: str, new: str):
    """Every string leaf with the scratch root prefix ``old`` mapped back to ``new`` (the lr dir)."""
    if isinstance(x, dict):
        return {k: subst_prefix(v, old, new) for k, v in x.items()}
    if isinstance(x, list):
        return [subst_prefix(v, old, new) for v in x]
    if isinstance(x, str) and old in x:
        return x.replace(old, new)
    return x


# ============================================================================== guards (before Rows)
def committed_clean(path: Path) -> dict:
    """Tracked by git and no change in the index or the working tree (``git status --porcelain`` empty)."""
    rel = repo_rel(path)
    if not Path(path).is_file():
        return {"ok": False, "path": rel, "detail": "does not exist"}
    ls = git("ls-files", "--error-unmatch", "--", rel)
    st = git("status", "--porcelain", "--", rel)
    df = git("diff", "--quiet", "HEAD", "--", rel)
    ok = ls.returncode == 0 and st.returncode == 0 and not st.stdout.strip() and df.returncode == 0
    return {"ok": ok, "path": rel, "tracked": ls.returncode == 0, "status_porcelain": st.stdout.strip(),
            "diff_head_rc": df.returncode,
            "detail": "committed and clean" if ok else "NOT committed / not clean (git ls-files / status / diff HEAD)"}


def guard_thresholds(path: Path = readout.THRESHOLDS) -> dict:
    got = sha256(path)
    thr = readout.load_thresholds(path)
    frozen = (thr.get("sources") or {}).get("sha256") or {}
    now = {n: sha256(readout.HERE / n) for n in readout.FROZEN_SCRIPTS}
    probs = []
    if got != THRESHOLDS_SHA:
        probs.append(f"thresholds sha256 {got[:8]} != {THRESHOLDS_SHA[:8]}")
    if frozen != now:
        probs.append("frozen-script sha256s differ from the thresholds file's record: "
                     f"{sorted(n for n in set(frozen) | set(now) if frozen.get(n) != now.get(n))}")
    if thr.get("decision") != "proceed":
        probs.append(f"thresholds decision {thr.get('decision')!r} != 'proceed'")
    if thr.get("tdr_sha256") != readout.TDR_SHA:
        probs.append(f"thresholds tdr_sha256 {str(thr.get('tdr_sha256'))[:8]} != TDR {readout.TDR_SHA[:8]}")
    return {"ok": not probs, "problems": probs, "path": str(path), "sha256": got, "want_sha256": THRESHOLDS_SHA,
            "frozen_scripts": now, "frozen_scripts_recorded": frozen}


def guard_r_inputs() -> dict:
    """No change since c2054a8c under the R-tracking inputs, committed or in the working tree (+ the pinned shas, and
    Task 4's resolution files / side22 classifier, which must lie under the checked paths)."""
    exps = S.EXPS
    paths = [exps / p for p in R_INPUT_PATHS]
    probs = []
    missing = [repo_rel(p) if p.exists() else str(p) for p in paths if not p.exists()]
    if missing:
        return {"ok": False, "problems": [f"missing: {missing}"], "paths": [str(p) for p in paths]}
    rels = [repo_rel(p) for p in paths]
    anc = git("cat-file", "-e", f"{R_INPUTS_SINCE}^{{commit}}")
    if anc.returncode != 0:
        probs.append(f"{R_INPUTS_SINCE} is not a commit here: {anc.stderr.strip()[:200]}")
    df = git("diff", "--quiet", R_INPUTS_SINCE, "--", *rels)
    if df.returncode != 0:
        names = git("diff", "--name-only", R_INPUTS_SINCE, "--", *rels).stdout.split()
        probs.append(f"git diff {R_INPUTS_SINCE} -- <R inputs> rc {df.returncode}: {names[:12]} {df.stderr[:200]}")
    st = git("status", "--porcelain", "--", *rels)
    if st.returncode != 0 or st.stdout.strip():
        probs.append(f"git status --porcelain <R inputs> not empty: {st.stdout.strip().splitlines()[:12]} "
                     f"{st.stderr[:200]}")
    shas = {}
    for rel, want in R_INPUT_SHAS.items():
        shas[rel] = sha256(exps / rel)
        if shas[rel] != want:
            probs.append(f"{rel} sha256 {shas[rel][:8]} != {want[:8]}")
    # Task 4's threshold_inputs.yaml names the resolution files and the side22 classifier: each must be checked above
    covered = []
    try:
        import yaml  # noqa: PLC0415
        ti = yaml.safe_load((exps / "2026-10-16_task4_pipeline" / "threshold_inputs.yaml").read_text()) or {}
        named = list(ti.get("resolution_files") or []) + ([ti["side22_classifier"]] if ti.get("side22_classifier")
                                                          else [])
        for n in named:
            ok = any(n == r or n.startswith(r.rstrip("/") + "/") for r in rels)
            covered.append({"path": n, "under_checked_paths": ok})
            if not ok:
                probs.append(f"threshold_inputs.yaml names {n}, which is not under the checked R-input paths")
    except Exception as e:  # noqa: BLE001
        probs.append(f"threshold_inputs.yaml unreadable: {type(e).__name__}: {e}")
    return {"ok": not probs, "problems": probs, "since": R_INPUTS_SINCE, "paths": rels, "sha256": shas,
            "threshold_inputs_named": covered}


def load_pin() -> dict:
    pin = readout.read_json(PIN)
    if not pin or not pin.get("code_sha") or not pin.get("eval_sha"):
        raise FreshRefused(f"{PIN}: no code_sha / eval_sha")
    return pin


def load_selection() -> dict:
    """selection.json committed and clean; its pick (sha256 recomputed from the checkpoint) is the new policy."""
    cc = committed_clean(SELECTION)
    out = {"committed": cc, "ok": False}
    if not cc["ok"]:
        out["problems"] = [f"selection.json: {cc['detail']} (RTF stays unread until it is committed)"]
        return out
    sel = json.loads(SELECTION.read_text())
    pick = sel.get("pick") or {}
    probs = []
    if sel.get("schema") != SELECTION_SCHEMA:
        probs.append(f"selection schema {sel.get('schema')!r} != {SELECTION_SCHEMA!r}")
    sha, ck = pick.get("sha256"), pick.get("ckpt")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        probs.append(f"selection pick sha256 {sha!r} is not a sha256")
    elif not ck or not Path(ck).is_file():
        probs.append(f"selection pick checkpoint {ck!r} missing")
    elif sha256(Path(ck)) != sha:
        probs.append(f"selection pick checkpoint {ck} sha256 != recorded {sha[:8]}")
    if sha == readout.TDR_SHA:
        probs.append("selection pick is TDR")
    out.update(ok=not probs, problems=probs, pick=pick, sha256=sha, selection_sha256=sha256(SELECTION))
    return out


def guards_fresh() -> dict:
    """All pre-Rows guards of --g1 / --full (each evaluated; any failure refuses)."""
    g = {"selection": load_selection(), "thresholds": guard_thresholds(), "r_inputs": guard_r_inputs()}
    pin = load_pin()
    g["pin"] = {"ok": True, "code_sha": pin["code_sha"], "eval_sha": pin["eval_sha"], "path": str(PIN)}
    g["ok"] = all(v.get("ok") for v in g.values() if isinstance(v, dict))
    return g


# ============================================================================== post-read checks
def served_checks(R: FreshRows, pin: dict, tdr_from_rtf: bool) -> dict:
    served = list(getattr(R, "served", []))
    probs = []
    bad_es = [s for s in served if s["real_es"] not in R.REAL_OK]
    if bad_es:
        probs.append(f"{len(bad_es)} lookups at real es outside {R.REAL_OK}: {bad_es[:3]}")
    for s in served:
        if not s.get("stamp"):
            continue
        if (s.get("code_sha"), s.get("eval_sha")) != (pin["code_sha"], pin["eval_sha"]):
            probs.append(f"{s['rid']} es{s['real_es']} {s['sha256'][:8]}: build {str(s.get('code_sha'))[:8]}/"
                         f"{str(s.get('eval_sha'))[:8]} != pin {pin['code_sha'][:8]}/{pin['eval_sha'][:8]}")
        if s["name_dir"] and s["name_dir"].startswith(IDENTITY_EXCLUDED_PREFIXES) and R.EMAP is EMAP_IDENTITY:
            probs.append(f"identity read served an excluded row {s['name_dir']}")
        if tdr_from_rtf and s["sha256"] == readout.TDR_SHA and not str(s["name_dir"]).startswith(TDR_FRESH_PREFIX):
            probs.append(f"TDR {s['rid']} es{s['real_es']} served from {s['name_dir']}, not an {TDR_FRESH_PREFIX}* row")
    uniq = {}
    for s in served:
        uniq.setdefault((s["rid"], s["logical_es"], s["sha256"]), s)
    return {"ok": not probs, "problems": probs, "n_lookups": len(served), "n_unique": len(uniq),
            "real_es_served": sorted({s["real_es"] for s in served}),
            "served": [{k: s[k] for k in ("rid", "logical_es", "real_es", "sha256", "status", "stamp", "name_dir",
                                          "code_sha", "eval_sha", "run_meta_seed", "tp_row_env_seed")}
                       for s in uniq.values()]}


# ============================================================================== shared reads (also select.py)
def g_kinds(R: readout.Rows, new_sha: str, tdr_sha: str, thr: dict, kinds=None) -> dict:
    """G per gated kind: readout.gates()'s own G loop (pair_tables at the frozen gate_env_seeds, g_line with the
    thresholds file's frozen G rule, the row summaries, g_gate with its k), restricted to ``kinds`` if given."""
    g_rule = readout._frozen_rule(thr, "G", readout.G_RULE)
    ess = tuple(thr["rule"]["gate_env_seeds"])
    out = {"kinds": {}}
    gst = []
    for kind, kd in thr["kinds"].items():
        if kd["status"] != "gated" or (kinds is not None and kind not in kinds):
            continue
        rid = S.row_id(kind, kd["gate_band"])
        pt = readout.pair_tables(R, rid, ess, new_sha, tdr_sha, "success")
        rec = {"row": rid, "band": kd["gate_band"], "status": pt["status"], "why": pt.get("why")}
        if pt["status"] == "OK":
            rec.update(readout.g_line(pt["a"], pt["b"], rid, g_rule))
            rec["new_row"] = readout.summarize(pt["tables"]["new"], kind)
            rec["tdr_row"] = readout.summarize(pt["tables"]["ref"], kind)
            rec["aggregate"] = readout._aggregates(pt)
            rec["paired_wins_losses"] = {
                "new_only": int(sum(bool(pt["a"][k]) and not bool(pt["b"][k]) for k in pt["a"])),
                "tdr_only": int(sum(bool(pt["b"][k]) and not bool(pt["a"][k]) for k in pt["a"]))}
        out["kinds"][kind] = rec
        gst.append(pt["status"])
    lines = {k: v for k, v in out["kinds"].items() if v["status"] == "OK"}
    out.update(readout.g_gate(lines, int(thr["rule"]["k"])))
    if any(s != "OK" for s in gst):
        out["status"] = "INVALID" if "INVALID" in gst else "INCOMPLETE"
    out["env_seeds_logical"] = list(ess)
    out["g_rule"] = g_rule
    return out


def suite_only(R: readout.Rows, new_sha: str, tdr_sha: str, ess=SUITE_LOGICAL_ES) -> dict:
    """The decisive SUITE tracking lines only: gate_read.zero_kg_check on {suite: {logical es: stamp}} of both
    policies (v1.2, track_err, drift 0, Task 4's resolution files: readout.zero_kg's call), rows starting 'suite' and
    not watch. res['verdict'] / res['fails'] are never read (they carry the absent field / side22 rows)."""
    def inputs(sha):
        d = {"suite": {}, "side22": None, "field_hard": [], "field_light": [], "ring_obst": [], "field_turn": []}
        recs = []
        for es in ess:
            f = R.found("R_suite", es, sha)
            recs.append(f)
            if f["status"] == "OK":
                d["suite"][int(es)] = f["stamp"]
        return d, recs

    h, rh = inputs(new_sha)
    r, rr = inputs(tdr_sha)
    out = {"env_seeds_logical": list(ess), "its_past_p2a": readout.ITS_PAST_R,
           "stamps": {"new": rh, "tdr": rr}}
    bad = [f"{f['row']} es{f['es']} {str(f['sha256'])[:8]}: {f['status']} {f['problems'][:2]}" for f in rh + rr
           if f["status"] != "OK"]
    if bad:
        out.update(status="INCOMPLETE", why=bad, lines=[], n_lines=0, fails=[])
        return out
    bp = readout.build_problems(*rh, *rr)
    if bp:
        out.update(status="INVALID", why=bp, lines=[], n_lines=0, fails=[])
        return out
    zk = readout.gr().zero_kg_check(h, r, list(readout.ZERO_KG_TRANSLATE), readout.th_view(), readout.ITS_PAST_R,
                                    readout.resolution_files())
    out["translate_statistic"], out["thresholds_version"] = zk.get("translate_statistic"), zk.get("thresholds_version")
    lines = [x for x in zk.get("rows") or [] if not x.get("watch") and str(x.get("row", "")).startswith("suite")]
    want = {(f"suite es{es}", st) for es in ess for st in SUITE_STATS}
    got = [(x.get("row"), x.get("statistic")) for x in lines]
    probs = []
    if zk.get("translate_statistic") != readout.TRACK_STATISTIC or \
            zk.get("thresholds_version") != readout.TRACK_THRESHOLDS_VERSION:
        probs.append(f"zero_kg_check read {zk.get('translate_statistic')!r} under {zk.get('thresholds_version')!r}, "
                     f"not {readout.TRACK_STATISTIC!r} / {readout.TRACK_THRESHOLDS_VERSION!r}")
    if len(got) != len(set(got)) or set(got) != want:
        probs.append(f"decisive suite lines {sorted(set(got) ^ want)[:8]} differ from the expected "
                     f"{len(want)} (duplicates: {len(got) - len(set(got))})")
    fails = [f"{x['statistic']} {x['row']}" for x in lines if x.get("fails")]
    out.update(lines=lines, n_lines=len(lines), fails=fails, problems=probs,
               status="INVALID" if probs else ("FAIL" if fails else "PASS"))
    return out


# ============================================================================== labels
def relabel(x, emap: dict):
    """A copy with a ``real_seed`` label beside every logical seed label (dict keys ``esN``, rows ``suite esN``,
    ``fails`` text, ``env_seeds_logical``); readout.find's stamp records (row / es / sha256 / status) carry the REAL
    es, so they get ``logical_es`` instead."""
    inv = {v: k for k, v in emap.items()}

    def sub(s: str) -> str:
        return re.sub(r"suite es(\d+)", lambda m: f"suite es{m.group(1)} (real es{emap.get(int(m.group(1)), '?')})",
                      s)

    if isinstance(x, list):
        return [relabel(v, emap) for v in x]
    if not isinstance(x, dict):
        return x
    out = {k: relabel(v, emap) for k, v in x.items()}
    esk = [k for k in x if isinstance(k, str) and re.fullmatch(r"es\d+", k)]
    if esk:
        out["real_seed"] = {k: f"es{emap.get(int(k[2:]), '?')}" for k in esk}
    if isinstance(x.get("row"), str) and re.fullmatch(r"suite es\d+", x["row"]):
        out["real_seed"] = f"es{emap.get(int(x['row'][8:]), '?')}"
    if isinstance(x.get("fails"), list) and any(isinstance(s, str) for s in x["fails"]):
        out["fails_real_seed"] = [sub(s) if isinstance(s, str) else s for s in x["fails"]]
    if isinstance(x.get("env_seeds_logical"), list):
        out["env_seeds_real"] = [emap.get(int(e), None) for e in x["env_seeds_logical"]]
    if {"row", "es", "sha256", "status"} <= set(x) and isinstance(x.get("es"), int):
        out["es_is"] = "real"
        out["logical_es"] = inv.get(int(x["es"]))
    return out


def sources() -> dict:
    return {"readout_fresh.py": sha256(Path(__file__)), "readout.py": sha256(readout.HERE / "readout.py"),
            "rt_spec.py": sha256(readout.HERE / "rt_spec.py"), "thresholds_rough_v1.json": sha256(readout.THRESHOLDS)}


# ============================================================================== modes
def identity_check(out_path: Path | None, scratch: Path | None) -> int:
    rec = json.loads(STEP9_GATES.read_text())
    new_sha, tdr_sha = rec["policies"]["new"], rec["policies"]["tdr"]
    if tdr_sha != readout.TDR_SHA:
        raise FreshRefused(f"gates.json tdr {tdr_sha[:8]} != TDR")
    td, root, linked, skipped = masked_root(scratch, IDENTITY_EXCLUDED_PREFIXES)
    try:
        R = IdentityRows([root], dup="earliest", shas={new_sha, tdr_sha})
        thr = readout.load_thresholds(readout.THRESHOLDS)
        got = readout.gates(R, new_sha, tdr_sha, thr)
        got["thresholds"] = {"path": str(readout.THRESHOLDS), "sha256": sha256(readout.THRESHOLDS)}
        got = subst_prefix(readout._norm(got), str(root), str(readout.RT_LR))
        diffs = readout.deep_diff(rec, got, tol=readout.EPS, only_recorded=False)
        sc = served_checks(R, load_pin(), tdr_from_rtf=False)
    finally:
        td.cleanup()
    same_verdict = rec["verdict"] == got["verdict"]
    ok = not diffs and same_verdict and sc["ok"]
    res = {"schema": "rt_short_fresh_identity_v1", "mode": "identity-check", "pass": ok,
           "seed_map": {str(k): v for k, v in EMAP_IDENTITY.items()}, "reference": str(STEP9_GATES),
           "reference_sha256": sha256(STEP9_GATES), "policies": rec["policies"], "tolerance": readout.EPS,
           "numbers_compared": readout._count_numbers(rec), "differences": diffs, "verdict": got["verdict"],
           "verdict_recorded": rec["verdict"], "same_verdict": same_verdict,
           "root": {"excluded_prefixes": list(IDENTITY_EXCLUDED_PREFIXES), "excluded_es_min": FRESH_SEED_MIN,
                    "linked": len(linked), "excluded": len(skipped)},
           "served_checks": sc, "sources": sources()}
    print(f"RT_FRESH IDENTITY {'PASS' if ok else 'FAIL'} numbers={res['numbers_compared']} diffs={len(diffs)} "
          f"verdict={got['verdict']['verdict']} (recorded {rec['verdict']['verdict']}) lookups={sc['n_lookups']} "
          f"real_es={sc['real_es_served']} excluded_name_dirs={len(skipped)} served_problems={len(sc['problems'])}")
    for d in diffs[:20]:
        print(f"  diff {d}")
    for p in sc["problems"][:10]:
        print(f"  served {p}")
    if out_path:
        readout.write_json(out_path, res)
        print(f"RT_FRESH wrote {out_path}")
    return 0 if ok else 1


def fresh(mode: str, out_path: Path) -> int:
    g = guards_fresh()                                     # BEFORE Rows: nothing of RTF is scanned on a refusal
    if not g["ok"]:
        why = [p for k in ("selection", "thresholds", "r_inputs") for p in (g[k].get("problems") or [])]
        raise FreshRefused("; ".join(why) or "a guard failed")
    new_sha = g["selection"]["sha256"]
    tdr_sha = readout.TDR_SHA
    pin = load_pin()
    thr = readout.load_thresholds(readout.THRESHOLDS)
    R = FreshRows([readout.RT_LR], dup="earliest", shas={new_sha, tdr_sha})
    head = {"mode": mode, "policies": {"new": new_sha, "tdr": tdr_sha}, "pick": g["selection"]["pick"],
            "seed_map": {str(k): v for k, v in EMAP_FRESH.items()},
            "seed_note": "labels inside G / R / zero_kg lines are LOGICAL (es1 = real es5, es2 = es6, es3 = es7, "
                         "es4 = es8); readout.find's stamp records carry the REAL es (es_is: real)",
            "guards": g, "drift_reading_6000": {"status": DRIFT_NA, "why": DRIFT_NA_WHY}}
    if mode == "g1":
        G = g_kinds(R, new_sha, tdr_sha, thr)
        trk = suite_only(R, new_sha, tdr_sha)
        res = {"schema": "rt_short_fresh_g1_v1", **head, "G": G, "suite_tracking": trk,
               "step8_rule": "ARM step 8: G FAIL -> NOT ACCEPTED, confirm-2 does not run (step 10 on the confirm-1 "
                             "rows, then CPU diagnosis, step 11, HALT); G PASS -> confirm-2 at once and notify the "
                             "suite lines (failing suite lines do not stop confirm-2)"}
        sentinel = (f"RT_FRESH G1 G={G['status']} gains={G['gain_kinds']} regressed={G['regressed']} "
                    f"SUITE={trk['status']} fails={trk.get('fails')}")
    else:
        res = readout.gates(R, new_sha, tdr_sha, thr)
        res["thresholds"] = {"path": str(readout.THRESHOLDS), "sha256": sha256(readout.THRESHOLDS)}
        again = readout.decide(res["G"]["status"], res["R"]["status"])
        if res.get("verdict") != again:
            raise FreshRefused(f"gates() verdict {res.get('verdict')} != decide() {again}")
        trk = res["R"].get("tracking") or {}
        if isinstance(trk.get("drift_reading"), dict):
            trk["drift_reading"]["applicability"] = DRIFT_NA
            trk["drift_reading"]["why_not_applicable"] = DRIFT_NA_WHY
        res = {"schema": "rt_short_fresh_gates_v1", **head, **res}
        res["verdict_real_seeds"] = (f"{res['verdict']['verdict']} on REAL env seeds 5-8 (G at es5-6; field_hard "
                                     "es5-8; other R rows es5-6; side22 es5); TDR from its RTF rows")
        sentinel = (f"RT_FRESH FULL verdict={res['verdict']['verdict']} G={res['G'].get('status')} "
                    f"R={res['R'].get('status')} (real es5-8)")
    sc = served_checks(R, pin, tdr_from_rtf=True)
    if not sc["ok"]:
        raise FreshRefused("post-read: " + "; ".join(sc["problems"][:6]))
    res["served_checks"] = sc
    res["sources"] = sources()
    res = relabel(readout._norm(res), EMAP_FRESH)
    readout.write_json(out_path, res)
    print(sentinel + f" lookups={sc['n_lookups']} real_es={sc['real_es_served']}")
    print(f"RT_FRESH wrote {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--identity-check", action="store_true")
    m.add_argument("--g1", action="store_true")
    m.add_argument("--full", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None, help="--identity-check: where the symlink root is made")
    a = ap.parse_args(argv)
    try:
        if a.identity_check:
            return identity_check(a.out, a.scratch)
        if a.g1:
            return fresh("g1", a.out or OUT_DIR / "g1.json")
        return fresh("full", a.out or OUT_DIR / "gates_fresh.json")
    except (FreshRefused, FreshSeedError, readout.KeySetError) as e:
        print(f"RT_FRESH REFUSED {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
