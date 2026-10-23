#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 pipeline: the ONE entry point (PLAN Shape) -- gen | check | lock | run | continue | status | gate | report.

Everything a run is comes from the spec (``--spec <campaign>/pipeline.yaml``; no dated path lives in this file):

  gen      writes ``<run>/candidates.yaml`` (one reference chain per driver stage, each resuming the previous stage by
           ``{reference:}`` in the SAME file -- no cross-campaign reference, no literal checkpoint path), the composed
           env files (``<run>/envs/<stage>.env``, ``<stage>.<row family>.env`` for rows with an env block, the payload
           level files ``envs/levels/L###.env``: KG=L:L, ZERO_FRAC=0, COM_XY=0, COM_Z=0.309; 0 kg = 0:0) and
           ``<run>/plan.json`` (the resolved stage graph). Byte-stable: ``gen --check`` regenerates in memory and
           compares bytes.
  check    the spec (overlays verbatim against the shipped lineage files by sha256, key counts, stage names, the
           dependency graph, row manifests), the PLACEHOLDERs an official run refuses (N_p, x_rule, the student
           manifest-shape split), the driver's own validator on the generated candidates, gen byte equality, and the
           pin guard (``pipeline.lock.json``).
  lock     writes ``pipeline.lock.json``: every out-of-hash file a run depends on by sha256, the code / eval / tools
           shas, git HEAD (informational) and the external stack.
  run      the detached queue (``setsid nohup``; one Isaac process at a time; resumable): per stage the training
           (driver stages through ``launch_candidates.sh --reference-chain``, the student through ``distill.sh``, the
           stand DAgger through the stand launchers), the live watch, the gate rows (``gate_row.sh`` with
           ``--trained-categories`` from the stage's composed env), the CPU gate read (``gate_read.py``) into
           ``<run>/gates/<stage>.json``. A HALT halts the dependent subtree, writes ``<run>/halts/<stage>.json`` (the
           line, the read, its chance rate), appends the sentinel ``T4PIPE HALT ...`` to ``<run>/pipeline_queue.log``
           (the orchestrator's hook) and to ``pipeline_notify.jsonl``, and the queue keeps running whatever does not
           depend on it, then PARKS.
  continue --accept-chance STAGE | --amend STAGE --charter TEXT | --retry STAGE | --abort  (PLAN On FAIL: a HALT
           is a prompt; an accepted one lets the run continue from the halted stage and stays eligible, disclosed)
  continue --amend STAGE --retarget-tstar --charter TEXT [--why TEXT] [--no-relaunch]  (CHARTER 2026-10-22, F19 p3b
           -> "amend + switch now"; :func:`do_retarget_tstar`): after a re-pin whose thresholds change the paired 0-kg
           check, ONE command re-gates every trained payload head's 0-kg check under the current thresholds (stage
           statuses unchanged, previous records kept under ``history/``), re-picks T* by the existing rule, resets
           everything built on T* to fresh runs (the open HALT at STAGE closes as amended) and moves the pin like
           --amend; refused, with nothing written, when T* would not change. It never deletes STOP_PIPELINE: remove it
           before the command (which relaunches the queue), or pass --no-relaunch and remove it before ``run``. F19:
           ``continue --run F19 --amend st --retarget-tstar --charter 'CHARTER 2026-10-22 F19 p3b: amend + switch now
           (thresholds v1.2)' --why '...'``.
  status / gate / report.
Stopping (CHARTER 2026-10-18 rule amendment; PLAN's "never SIGKILL" for non-driver stages now follows
``run_phases._stop_child``): a driver stage is stopped with ``systemctl --user stop`` (the driver's own SIGTERM path,
``_stop_child`` on its trainer). A non-driver stage (distill, DAgger) is stopped by :func:`stop_trainer`: SIGTERM to
the trainer python's OWN pid -- never a scope-wide signal, which also reaches ``timeout`` and is re-forwarded (the C7
``st`` hang: 28 h at 100 % CPU) -- ``TERM_GRACE_S`` (120 s), SIGTERM again, 120 s, and only then SIGKILL as the last
resort. A SIGKILL is recorded in the stage record, notified, and opens the DerivedDataCache watch: before every Kit
launch (and every poll of a running stage) the cache size and the Kit logs started since the kill are checked for the
migration symptom ("deleted or corrupted blocks"), until one Kit has started clean. A progress watchdog
(``live.progress_watchdog_min``, default 30) stops a non-driver stage whose train log / tfevents / checkpoints stop
moving and HALTs it INFRA, so a silent hang never parks the queue.
ONE ISAAC AT A TIME: the queue is serial, every stage returns only after its Isaac processes have exited, and
:func:`pre_launch` waits -- with no give-up -- until no Isaac python of ANY kind remains (``tp_lib.ISAAC_PAT``: drivers,
train / eval / play and every sn_ / sh_ tool) immediately before EVERY Isaac launch (driver, row, distill, DAgger,
stand rows). No ``pgrep -f``: the busy check matches an anchored pattern against ``ps -ww -eo args`` and the queue's
liveness is its pidfile + /proc/<pid>/cmdline.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _lib():
    import importlib.util  # noqa: PLC0415

    if "tp_lib" in sys.modules:
        return sys.modules["tp_lib"]
    spec = importlib.util.spec_from_file_location("tp_lib", str(HERE / "tp_lib.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tp_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _lib()
GR = L.load_by_path(HERE / "gate_read.py", "tp_gate_read")


def PR():
    return L.load_by_path(HERE / "payload_read.py", "tp_payload_read")


def LW():
    return L.load_by_path(HERE / "live_watch.py", "tp_live_watch")


def RR():
    return L.load_by_path(HERE / "replay_read.py", "tp_replay_read")


def SS():
    return L.load_by_path(HERE / "select_student.py", "tp_select_student")


PLACEHOLDER = "PLACEHOLDER"
SUCCESS = ("pass", "accepted")
FREEZE_NOTICE = ("FREEZE (D14): from the pin until F19 ends, other sessions stay off the code / eval / TP / PL paths "
                 "and every file in the lock; the driver refuses a dirty code- or eval-bearing tree and the pin guard "
                 "refuses any drift. Commit or stash uncommitted edits on those paths before the pin.")
SETTLED = ("pass", "accepted", "ceiling", "skipped")
TERMINAL = SETTLED + ("halt", "blocked", "aborted")
GEN_HEADER = "# GENERATED by TP/pipeline.py gen from {spec} -- do not edit by hand.\n"
DRIVER_KINDS = ("driver",)
DDC = Path("/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache")
DDC_MAX_GB = 80
#: after a SIGKILL: growth of the cache beyond its size at the kill that HALTs (GUARD) the next launch
DDC_GROWTH_GB = 5
#: a Kit log started after the SIGKILL that is this old and shows no symptom closes the DerivedDataCache watch
DDC_LOG_SETTLE_S = 180
KIT_LOG_ROOT = DDC.parent.parent / "logs" / "Kit"
#: the DerivedDataCache corruption symptoms in a Kit log (memory: kit-derived-data-cache-corruption): the index
#: rewrite after a mid-write kill, and the in-memory migration loop (a new bucket file every second)
DDC_SYMPTOM = re.compile(r"deleted or corrupted blocks|rewriting index", re.I)
DDC_LOOP = re.compile(r"Started new bucket file")
DDC_LOOP_MIN = 3
#: run_phases.TERM_GRACE_S: the grace after each SIGTERM (``live.term_grace_s`` overrides)
TERM_GRACE_S = 120
KILL_WAIT_S = 60
TRAINER_FIND_S = 60
LAUNCHER_EXIT_S = 300
#: ``live.progress_watchdog_min`` default
WATCHDOG_MIN = 30
FREE_MIN_GB = 150
EXTERNAL_PKGS = ("isaacsim", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl-rl-lib", "torch", "numpy", "scipy",
                 "tensorboard", "gymnasium", "PyYAML")


def is_placeholder(v) -> bool:
    return isinstance(v, str) and v.strip().upper().startswith(PLACEHOLDER)


# ============================================================================== the spec
class Spec:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.dir = self.path.parent
        self.d = L.load_yaml(self.path)
        if int(self.d.get("schema", 0)) != 1:
            raise ValueError(f"{self.path}: schema {self.d.get('schema')} (want 1)")
        L.configure(self.d)
        self._by_id = {s["id"]: s for s in self.d["stages"]}

    # -- runs / stages
    def run(self, name: str) -> dict:
        if name not in self.d["runs"]:
            raise KeyError(f"no run {name!r} in {self.path} (known: {sorted(self.d['runs'])})")
        return self.d["runs"][name]

    def official(self, name: str) -> bool:
        return not bool(self.run(name).get("dry"))

    def run_dir(self, name: str) -> Path:
        return self.dir / self.run(name)["dir"]

    def stage(self, sid: str) -> dict:
        return self._by_id[sid]

    def stages(self, run: str) -> list[dict]:
        r = self.run(run)
        return [s for s in self.d["stages"] if not (s.get("x_replay") and not r.get("x_replay"))]

    def chain(self, run: str, sid: str) -> str:
        return self.d["chain_name"].format(run=run, stage=sid)

    def seed(self, run: str, sid: str) -> int:
        st = self.stage(sid)
        r = self.run(run)
        return int(r["dagger_seed"] if st["kind"] in ("dagger_sn", "dagger_sh") else r["walker_seed"])

    def n_p(self):
        return self.d.get("N_p")

    def iters(self, run: str, sid: str):
        r, st = self.run(run), self.stage(sid)
        ov = (r.get("stage_iters_override") or {}).get(sid)
        if ov is not None:
            return int(ov)
        if st["kind"] in DRIVER_KINDS and r.get("iters_override") is not None:
            return int(r["iters_override"])
        it = st.get("iters")
        if it == "N_p":
            return self.n_p()
        return it

    def thresholds_path(self) -> Path:
        return self.dir / self.d["thresholds"]["file"]

    def lock_path(self) -> Path:
        return self.dir / self.d["lock"]

    def row(self, fam: str) -> dict:
        return self.d["rows"][fam]

    def manifest(self, fam: str) -> Path:
        return L.OMNI / self.row(fam)["manifest"]

    def overlay(self, sid: str) -> dict:
        """The stage's training overlay: a lineage stage's verbatim; a payload stage = its source's + the payload."""
        st = self.stage(sid)
        if "overlay" in st:
            return dict(st["overlay"])
        if "overlay_from" in st:
            ov = dict(self.stage(st["overlay_from"])["overlay"])
            ov["KRABBY_PAYLOAD_KG"] = str(st["payload_kg"])
            ov.update({k: str(v) for k, v in self.d["payload"]["training"].items()})
            return ov
        raise KeyError(f"stage {sid} has no overlay")

    def driver_stages(self, run: str) -> list[dict]:
        return [s for s in self.stages(run) if s["kind"] in DRIVER_KINDS]


def parse_row(key: str) -> dict:
    """``family@es[:driver]`` -> {family, es, driver}."""
    m = re.match(r"^([a-z_0-9]+)@(\d+)(:driver)?$", key)
    if not m:
        raise ValueError(f"row key {key!r} is not family@es[:driver]")
    return {"family": m.group(1), "es": int(m.group(2)), "driver": bool(m.group(3)), "key": f"{m.group(1)}@{m.group(2)}"}


def manifest_rows(path: Path) -> dict:
    return {s["id"]: s for s in (L.load_yaml(path).get("scenarios") or [])}


# ============================================================================== check: lineage, graph, placeholders
def lineage_problems(spec: Spec) -> list[str]:
    out = []
    for sid, lin in (spec.d.get("lineage") or {}).items():
        src = L.EXP / lin["source"]
        if not src.is_file():
            out.append(f"lineage {sid}: source {src} missing")
            continue
        ch = (L.load_yaml(src).get("chains") or {}).get(lin["chain"])
        if ch is None:
            out.append(f"lineage {sid}: chain {lin['chain']} not in {lin['source']}")
            continue
        blk = next((s for s in ch.get("stages") or [] if s.get("stage") == lin["stage"]), None)
        if blk is None:
            out.append(f"lineage {sid}: stage {lin['stage']} not in chain {lin['chain']}")
            continue
        st = spec.stage(sid)
        ov = st.get("overlay") or {}
        if list(ov.items()) != list((blk.get("overlay") or {}).items()):
            out.append(f"lineage {sid}: the spec overlay is not the shipped one VERBATIM (keys / values / order)")
        if L.sha256_obj(blk.get("overlay") or {}) != lin["overlay_sha256"]:
            out.append(f"lineage {sid}: the shipped overlay's sha256 is no longer {lin['overlay_sha256'][:12]}")
        if L.sha256_obj(ov) != lin["overlay_sha256"]:
            out.append(f"lineage {sid}: the spec overlay's sha256 != {lin['overlay_sha256'][:12]}")
        if len(ov) != int(lin["keys"]):
            out.append(f"lineage {sid}: {len(ov)} overlay keys (want {lin['keys']})")
        if st.get("base") != blk.get("base"):
            out.append(f"lineage {sid}: base {st.get('base')} != shipped {blk.get('base')}")
        if st.get("iters") != blk.get("iters"):
            out.append(f"lineage {sid}: iters {st.get('iters')} != shipped {blk.get('iters')}")
        top = {k: v for k, v in ch.items() if k not in ("stages", "notes")}
        if (st.get("chain_keys") or {}) != top:
            out.append(f"lineage {sid}: chain keys {st.get('chain_keys')} != shipped {top}")
        if sorted(st.get("categories") or []) != sorted(blk.get("categories") or []):
            out.append(f"lineage {sid}: categories differ from the shipped stage")
    return out


#: live.student.loss_rule values (live_watch.judge_student): kill = the loss rule kills; watch = reported only
LOSS_RULES = ("kill", "watch")
#: the row lists (and read inputs) a stand stage runs on; an X stage takes the ones it does not name from its D stage
STAND_ROW_KEYS = {"dagger_sn": ("stand_rows",), "dagger_sh": ("height_rows", "k5_rows", "controls_spec")}


def stand_stage_cfg(spec: "Spec", sid: str) -> dict:
    """A stand stage's spec. An X stage (``x_replay``: "as D1 / D2 but beside the bundled S1", DESIGN X row) takes
    the row lists it does not name itself -- the stand / height / K5 rows, the controls spec, the deadband: the K reads
    X is reported against -- from its D stage (``rows_from``, else the same-kind non-X stage: sn / sh)."""
    st = spec.stage(sid)
    if not st.get("x_replay") or st["kind"] not in STAND_ROW_KEYS:
        return st
    src = st.get("rows_from") or next((s["id"] for s in spec.d["stages"]
                                       if s["kind"] == st["kind"] and not s.get("x_replay")), None)
    base = spec.stage(src) if src else {}
    return {**{k: base[k] for k in STAND_ROW_KEYS[st["kind"]] + ("deadband",) if k in base}, **st}


def graph_problems(spec: Spec) -> list[str]:
    out, seen = [], []
    for st in spec.d["stages"]:
        sid = st["id"]
        if sid in ("2a", "2b", "2c"):
            out.append(f"stage {sid}: run_candidates.compose_recipe blanks a stage named like a preset")
        if sid in seen:
            out.append(f"stage {sid} given twice")
        for k in ("needs", "settled", "after"):
            for dep in st.get(k) or []:
                if dep not in seen:
                    out.append(f"stage {sid}: {k} {dep} is not an EARLIER stage")
        seen.append(sid)
    for fam, r in spec.d["rows"].items():
        mp = L.OMNI / r["manifest"]
        if not mp.is_file():
            out.append(f"row family {fam}: manifest {mp} missing")
            continue
        if r["scenario"] not in manifest_rows(mp):
            out.append(f"row family {fam}: scenario {r['scenario']} not in {r['manifest']}")
    for st in spec.d["stages"]:
        rows = (st.get("gate") or {}).get("rows")
        keys = spec.d["row_lists"][rows] if isinstance(rows, str) else (rows or [])
        for key in keys:
            fam = parse_row(key)["family"]
            if fam not in spec.d["rows"]:
                out.append(f"stage {st['id']}: row family {fam} unknown")
    lr = ((spec.d.get("live") or {}).get("student") or {}).get("loss_rule", "kill")
    if lr not in LOSS_RULES:
        out.append(f"live.student.loss_rule {lr!r}: must be one of {'|'.join(LOSS_RULES)}")
    for k, dflt in (("progress_watchdog_min", WATCHDOG_MIN), ("term_grace_s", TERM_GRACE_S)):
        v = (spec.d.get("live") or {}).get(k, dflt)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
            out.append(f"live.{k} {v!r}: must be a positive number")
    for st in spec.d["stages"]:
        if st["kind"] in STAND_ROW_KEYS:
            cfg = stand_stage_cfg(spec, st["id"])
            need = list(STAND_ROW_KEYS[st["kind"]]) + (["refs_rows"] if st["kind"] == "dagger_sn"
                                                       and st.get("walker") == "tstar" else [])
            miss = [k for k in need if k not in cfg]
            if miss:
                out.append(f"stage {st['id']}: no {miss} (an X stage takes them from its D stage)")
        if st["kind"] in ("dagger_sn", "dagger_sh") and st.get("judged") is not None \
                and st["judged"] != f"model_{st.get('iters')}.pt":
            out.append(f"stage {st['id']}: judged {st['judged']} is not the final checkpoint model_{st.get('iters')}.pt "
                       "(the queue expects model_<launched iterations>.pt)")
    return out


def placeholder_problems(spec: Spec, run: str) -> list[str]:
    out = []
    if not spec.official(run):
        return out
    n_p = spec.n_p()
    if is_placeholder(n_p) or n_p is None:
        out.append(f"N_p is a PLACEHOLDER ({str(n_p)[:60]}...): set at PLAN step 8 before {run}")
    elif not (isinstance(n_p, int) and n_p % 500 == 0 and 1500 <= n_p <= 3000):
        out.append(f"N_p {n_p!r} must be an int, a multiple of 500, in [1500, 3000] (DESIGN 2.2)")
    if spec.run(run).get("x_replay") and is_placeholder(spec.d.get("x_rule")):
        out.append("x_rule is a PLACEHOLDER: the step-8 determinism smoke decides bitwise | report_only")
    if spec.d.get("x_rule") not in ("bitwise", "report_only") and not is_placeholder(spec.d.get("x_rule")):
        out.append(f"x_rule {spec.d.get('x_rule')!r} must be bitwise | report_only")
    sel = next((s for s in spec.stages(run) if s["kind"] == "select"), None)
    if sel and is_placeholder((sel.get("completion_margin") or {}).get("manifest_shape_split")):
        out.append("sel.completion_margin.manifest_shape_split is a PLACEHOLDER: measured at PLAN step 8")
    return out


# ============================================================================== gen
def decision_rule(spec: Spec) -> dict:
    src = spec.d["decision_rule_source"]
    rule = copy.deepcopy(L.load_yaml(L.EXP / src["file"])[src["key"]])
    rule["text"] = (f"TASK 4 PIPELINE ({spec.dir.name}): RECORD ONLY. Every gate of this run is TP/gate_read.py's, "
                    "against the committed thresholds; this driver only trains the reference chains and runs their "
                    "stage evals.")
    return rule


def candidates_doc(spec: Spec, run: str) -> dict:
    r = spec.run(run)
    chains, refs, prev = {}, {}, None
    for st in spec.driver_stages(run):
        sid = st["id"]
        cid = spec.chain(run, sid)
        if st["resume"] == "scratch":
            resume, scratch = "scratch", True
        else:
            if st["resume"] != prev:
                raise ValueError(f"stage {sid} resumes {st['resume']} but the previous driver stage is {prev}")
            resume, scratch = {"reference": spec.chain(run, st["resume"])}, False
        it = spec.iters(run, sid)
        stage = {"stage": sid, "base": st["base"], "iters": it if not is_placeholder(it) else str(it),
                 "from_scratch": scratch, "resume_from": resume, "overlay": spec.overlay(sid),
                 "eval": list(st.get("eval") or []), "categories": list(st["categories"]),
                 "floor": dict(st.get("floor") or {"kind": "none"})}
        chains[cid] = {"notes": f"Task 4 pipeline {run} stage {sid} ({st.get('lineage')} lineage"
                                f"{', payload ' + st['payload_kg'] + ' kg' if st.get('payload_kg') else ''}).",
                       **copy.deepcopy(st.get("chain_keys") or {}), "stages": [stage]}
        refs[cid] = {"chain": cid, "seed": int(r["walker_seed"])}
        prev = sid
    first = spec.chain(run, spec.driver_stages(run)[0]["id"])
    return {
        "schema": 1, "plant": spec.d["plant"],
        "seeds": {"screen": int(r["walker_seed"]), "confirm": 16},
        "screen": {"chain": first, "eval": [], "iters": spec.iters(run, spec.driver_stages(run)[0]["id"])},
        "promotion": {"enabled": False},
        "baseline": {"id": first, "chain": first},
        "decision_rule": decision_rule(spec),
        "chains": chains, "references": refs, "candidates": [], "milestones": [],
    }


def composed_envs(spec: Spec, run: str, sid: str) -> dict:
    """{train, eval, trained, csv, rows: {family: env}} of one driver stage, composed EXACTLY as the driver does
    (``run_phases.eval_env_for`` + the plant; ``run_candidates._effective_stage_cfg`` for the training env)."""
    rc = L.rc()
    rp = rc.rp
    st = spec.stage(sid)
    ov = spec.overlay(sid)
    train = rc._effective_stage_cfg({}, {"base": st["base"]}, ov)
    tc = rc.eval_categories().trained_categories(train).to_dict()
    csv = rc._trained_csv(tc)
    ev = dict(rp.eval_env_for(rp.ph.PHASES[st["base"]], spec.d["plant"], keep=rp.CANARY_KEYS,
                              keep_source=rp.ph.FORMATION, carry_overlay=ov))
    ev["KRABBY_PLANT"] = spec.d["plant"]
    rows = {}
    for fam, r in spec.d["rows"].items():
        blk = manifest_rows(L.OMNI / r["manifest"]).get(r["scenario"]) or {}
        if blk.get("env"):
            rows[fam] = {**ev, **{k: str(v) for k, v in blk["env"].items()}}
    return {"train": train, "eval": ev, "trained": tc, "csv": csv, "rows": rows}


def gate_rows(spec: Spec, sid: str) -> list[str]:
    g = spec.stage(sid).get("gate") or {}
    rows = g.get("rows")
    return list(spec.d["row_lists"][rows]) if isinstance(rows, str) else list(rows or [])


def gen_files(spec: Spec, run: str) -> dict:
    """{path relative to the run dir: text} -- every file gen writes (byte-stable)."""
    files = {}
    import yaml  # noqa: PLC0415

    files["candidates.yaml"] = (GEN_HEADER.format(spec=spec.path.name)
                                + yaml.safe_dump(candidates_doc(spec, run), sort_keys=False, width=120))
    plan = {"run": run, "spec": spec.path.name, "spec_sha256": L.sha256_file(spec.path),
            "thresholds": {"file": spec.d["thresholds"]["file"], "version": spec.d["thresholds"]["version"]},
            "walker_seed": spec.run(run)["walker_seed"], "dagger_seed": spec.run(run)["dagger_seed"],
            "order": [s["id"] for s in spec.stages(run)], "stages": {}, "levels": {}}
    for st in spec.stages(run):
        sid = st["id"]
        e = {"kind": st["kind"], "needs": st.get("needs") or [], "settled": st.get("settled") or [],
             "after": st.get("after") or [], "when": st.get("when"), "seed": spec.seed(run, sid),
             "iters": spec.iters(run, sid)}
        if st["kind"] in DRIVER_KINDS:
            ce = composed_envs(spec, run, sid)
            e.update({"chain": spec.chain(run, sid), "base": st["base"], "overlay_sha256": L.sha256_obj(spec.overlay(sid)),
                      "trained_csv": ce["csv"], "trained": ce["csv"].split(",") if ce["csv"] else [],
                      "env": f"envs/{sid}.env", "train_env": f"envs/{sid}.train.env",
                      "distill_env": f"envs/{sid}.distill.env",
                      "row_env": {fam: f"envs/{sid}.{fam}.env" for fam in sorted(ce["rows"])},
                      "gate_rows": gate_rows(spec, sid)})
            files[f"envs/{sid}.env"] = L.env_file_text(ce["eval"])
            files[f"envs/{sid}.train.env"] = L.env_file_text(ce["train"])
            # the driver's own training-process shape (run_phases.train: the preset + the plant + the overlay; the
            # preset expands at import) -- what distill.sh starts the student from (it strips the runner-only keys)
            files[f"envs/{sid}.distill.env"] = L.env_file_text({"KRABBY_OMNI_PHASE": st["base"],
                                                                "KRABBY_PLANT": spec.d["plant"], **spec.overlay(sid)})
            for fam, env in sorted(ce["rows"].items()):
                files[f"envs/{sid}.{fam}.env"] = L.env_file_text(env)
        plan["stages"][sid] = e
    lv = spec.d["payload"]
    for kg in lv["levels"]:
        env = {"KRABBY_PAYLOAD_KG": f"{kg}:{kg}", **{k: str(v) for k, v in lv["level_env"].items()}}
        files[f"envs/levels/L{int(kg):03d}.env"] = L.env_file_text(env)
        plan["levels"][str(kg)] = f"envs/levels/L{int(kg):03d}.env"
    files["plan.json"] = json.dumps(plan, indent=1, sort_keys=True) + "\n"
    return files


def gen(spec: Spec, run: str, *, check: bool = False) -> list[str]:
    """Write (or, with ``check``, compare) the generated files. Returns the differing / written paths."""
    files = gen_files(spec, run)
    rd = spec.run_dir(run)
    out = []
    for rel, text in sorted(files.items()):
        p = rd / rel
        if check:
            if not p.is_file() or p.read_text() != text:
                out.append(str(p))
        else:
            if not p.is_file() or p.read_text() != text:
                # atomic (a running queue's jobs read these files): temp file in the same dir + os.replace
                mode = (p.stat().st_mode & 0o777) if p.is_file() else 0o644
                L.write_text_atomic(p, text)
                os.chmod(p, mode)
            out.append(str(p))
    if check:
        envd = rd / "envs"
        if envd.is_dir():
            for p in envd.rglob("*.env"):
                if p.relative_to(rd).as_posix() not in files:
                    out.append(f"{p} (not generated by this spec)")
    return out


def validate_generated(spec: Spec, run: str) -> list[str]:
    """run_candidates.validate_candidates (strict) on the generated candidates (the driver's own refusals)."""
    cands = candidates_doc(spec, run)
    out = []
    try:
        verdicts = L.rc().validate_candidates(cands, strict=True, check_replay=False)
    except SystemExit as exc:                  # strict: the driver refuses by SystemExit, listing every reason
        return [f"driver validator: {ln.strip()}" for ln in str(exc.code).splitlines() if ln.strip()]
    for v in verdicts:
        if not v.get("ok", True):
            out += [f"driver validator ({v.get('id')}): {r}" for r in v.get("reasons") or []]
    return out


# ============================================================================== the lock / pin guard
def lock_files(spec: Spec) -> list[Path]:
    exp, omni = L.EXP, L.OMNI
    files = set()
    files.update(p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".sh"))
    for n in ("launch_candidates.sh", "run_candidates.py", "ledger_cli.py", "resolution.py"):
        files.add(L.TOOLS / n)
    files.add(spec.path)
    files.add(spec.thresholds_path())
    for n in ("threshold_inputs.yaml", "threshold_inputs_v1.yaml", "threshold_inputs_v1_1.yaml",
              "threshold_inputs_v1_2.yaml", "cameras.yaml"):
        if (spec.dir / n).is_file():
            files.add(spec.dir / n)
    for sub in ("scripts", "rows"):
        if (spec.dir / sub).is_dir():
            files.update(p for p in (spec.dir / sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for lin in (spec.d.get("lineage") or {}).values():
        files.add(exp / lin["source"])
    files.add(exp / spec.d["decision_rule_source"]["file"])
    for r in spec.d["rows"].values():
        files.add(omni / r["manifest"])
    files.add(omni / spec.d["paths"]["side22_classify"])
    files.add(omni / spec.d["paths"]["sh_kin"])
    for n in ("run_arm.sh", "p2_read.py", "p2_live.py", "p2s_stand.py"):
        files.add(omni / spec.d["paths"]["p2_scripts"] / n)
    st = spec.d.get("stand") or {}
    for key in ("sn_scripts", "sh_scripts"):
        d = omni / st[key]
        files.update(p for p in d.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".json", ".yaml"))
    sh_camp = omni / st["sh_scripts"] / ".."
    for n in ("facts.json", "scenarios_sh.yaml"):
        if (sh_camp / n).is_file():
            files.add((sh_camp / n).resolve())
    for s in spec.d["stages"]:
        if s.get("controls_spec"):
            files.add(omni / s["controls_spec"])
    ti = spec.dir / "threshold_inputs.yaml"
    if ti.is_file():
        for rf in (L.load_yaml(ti).get("resolution_files") or []):
            files.add(L.REPO / rf)
    files.add(L.REPO / "assets/crab.usda")
    files.add(L.REPO / spec.d["bundled_s1"]["ckpt"])
    files.update((omni / "policy/stand_net").glob("*.pt"))
    files.add(omni / "config/crab_hex/omni_student_spec.py")
    for run in spec.d["runs"]:
        rd = spec.run_dir(run)
        for rel in ("candidates.yaml", "plan.json"):
            if (rd / rel).is_file():
                files.add(rd / rel)
        if (rd / "envs").is_dir():
            files.update((rd / "envs").rglob("*.env"))
    return sorted(Path(p).resolve() for p in files)


def external_stack() -> dict:
    code = ("import importlib.metadata as m, json, sys\nout={}\n"
            f"for p in {list(EXTERNAL_PKGS)!r}:\n"
            "    try: out[p]=m.version(p)\n    except Exception: out[p]=None\n"
            "out['python']=sys.version.split()[0]\nprint(json.dumps(out, sort_keys=True))")
    try:
        r = subprocess.run([L.PY, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def lock_doc(spec: Spec) -> dict:
    rc = L.rc()
    head = subprocess.run(["git", "-C", str(L.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = {L.rel(p): (L.sha256_file(p) if p.is_file() else None) for p in lock_files(spec)}
    return {"schema": 1, "spec": spec.path.name, "written_at": L.now_iso(),
            "git_head_informational": head, "code_sha": rc.code_sha(), "eval_sha": rc.eval_sha(),
            "tools_sha": rc.tools_sha(), "dirty_code_eval": rc.dirty_paths(),
            "files": files, "external": external_stack(),
            "notes": "PLAN step 6 lock. The pin guard refuses a run whose live code_sha / eval_sha / any file here / "
                     "the external stack drift from it, a dirty code / eval tree, or a lock whose own sha moved since "
                     "the run started. Re-written (python pipeline.py lock) at the step-8 pin."}


def guard_problems(spec: Spec, *, pin_lock_sha: str | None = None, check_disk: bool = True) -> list[str]:
    lp = spec.lock_path()
    if not lp.is_file():
        return [f"no lock {lp} (python pipeline.py lock)"]
    lock = json.loads(lp.read_text())
    out = []
    if pin_lock_sha is not None and L.sha256_file(lp) != pin_lock_sha:
        out.append(f"{lp.name} changed since the run was pinned ({pin_lock_sha[:12]})")
    if lock.get("dirty_code_eval"):
        out.append(f"the lock was written on a dirty code / eval tree ({lock['dirty_code_eval'][:3]})")
    rc = L.rc()
    for k, fn in (("code_sha", rc.code_sha), ("eval_sha", rc.eval_sha)):
        live = fn()
        if live != lock.get(k):
            out.append(f"{k} drifted: live {live[:12]} vs lock {str(lock.get(k))[:12]}")
    dirty = rc.dirty_paths()
    if dirty:
        out.append(f"code / eval tree dirty: {dirty[:4]}{' ...' if len(dirty) > 4 else ''}")
    for rel_, sha in sorted((lock.get("files") or {}).items()):
        p = L.REPO / rel_
        live = L.sha256_file(p) if p.is_file() else None
        if live != sha:
            out.append(f"locked file drifted: {rel_} ({str(sha)[:12]} -> {str(live)[:12]})")
    missing = sorted(set(L.rel(p) for p in lock_files(spec)) - set(lock.get("files") or {}))
    if missing:
        out.append(f"files the run depends on are not in the lock: {missing[:4]}")
    ext = external_stack()
    if ext != lock.get("external"):
        diff = sorted(k for k in set(ext) | set(lock.get("external") or {})
                      if ext.get(k) != (lock.get("external") or {}).get(k))
        out.append(f"external stack drifted: {diff}")
    if check_disk:
        free = shutil.disk_usage(str(L.REPO)).free / 2 ** 30
        if free < FREE_MIN_GB:
            out.append(f"free disk {free:.0f} GB < {FREE_MIN_GB} GB")
    return out


def ddc_gb() -> float:
    if not DDC.is_dir():
        return 0.0
    r = subprocess.run(["du", "-s", "--block-size=1G", str(DDC)], capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0


# ============================================================================== the runtime (all side effects)
class Runtime:
    """Every external effect of the queue (the tests substitute a fake)."""

    poll_s = 120

    def __init__(self, log: Path):
        self.log = log

    def time(self) -> float:
        return time.time()

    def sleep(self, s: float) -> None:
        time.sleep(s)

    def isaac_busy(self) -> bool:
        return L.isaac_busy()

    def wait_idle(self, stop=None) -> None:
        while self.isaac_busy():
            if stop and stop():
                return
            self.sleep(30)

    def run(self, cmd: list, *, env: dict | None = None, log: Path | None = None, timeout: float | None = None) -> tuple:
        """Foreground; returns (rc, last stdout line)."""
        full = {**clean_env(), **(env or {})}
        Path(log or self.log).parent.mkdir(parents=True, exist_ok=True)
        with open(log or self.log, "a") as fh:
            fh.write(f"# {L.now_iso()} $ {' '.join(map(str, cmd))}\n")
            fh.flush()
            p = subprocess.run([str(c) for c in cmd], env=full, stdout=subprocess.PIPE, stderr=fh, text=True,
                               timeout=timeout)
            fh.write(p.stdout)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return p.returncode, (lines[-1] if lines else "")

    def spawn(self, cmd: list, *, env: dict | None = None, log: Path) -> subprocess.Popen:
        full = {**clean_env(), **(env or {})}
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a")
        fh.write(f"# {L.now_iso()} $ {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.Popen([str(c) for c in cmd], env=full, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=False)

    def launch_driver(self, cdir: Path, chain: str, seed: int) -> tuple:
        cmd = [str(L.TOOLS / "launch_candidates.sh"), "--campaign-dir", str(cdir), "--reference-chain", chain,
               "--seed", str(seed)]
        p = subprocess.run(cmd, env=clean_env(), capture_output=True, text=True)
        unit = next((ln.strip() for ln in p.stdout.splitlines() if ln.startswith("cands-")), None)
        return unit, (p.stdout + p.stderr)[-2000:]

    def unit_active(self, unit: str) -> bool:
        return subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode == 0

    def stop_unit(self, unit: str) -> None:
        """A driver: ``systemctl --user stop`` (KillMode=mixed: SIGTERM to the driver, which stops its child)."""
        subprocess.run(["systemctl", "--user", "stop", "--no-block", unit])

    def term_scope(self, unit: str) -> None:
        """SIGTERM to every process of a scope. Only for a scope whose trainer python has ALREADY exited (the
        ``timeout`` / launcher left behind): a scope-wide signal on a live trainer also reaches ``timeout``, which
        re-forwards it -- the C7 hang. A live trainer is stopped by :func:`stop_trainer`."""
        name = unit if unit.endswith(".scope") else unit + ".scope"
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", name])

    def ps_pid_args(self) -> list:
        """[(pid, args)] of every process (``ps -ww -eo pid=,args=``)."""
        p = subprocess.run(["ps", "-ww", "-eo", "pid=,args="], capture_output=True, text=True)
        out = []
        for ln in p.stdout.splitlines():
            f = ln.strip().split(None, 1)
            if len(f) == 2 and f[0].isdigit():
                out.append((int(f[0]), f[1]))
        return out

    def scope_pids(self, unit: str) -> set:
        """The pids of a scope's own cgroup (``systemctl show -p ControlGroup`` + cgroup.procs); empty if gone."""
        name = unit if unit.endswith(".scope") else unit + ".scope"
        r = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", name],
                           capture_output=True, text=True)
        cg = r.stdout.strip()
        if not cg:
            return set()
        try:
            return {int(x) for x in Path("/sys/fs/cgroup" + cg, "cgroup.procs").read_text().split()}
        except (OSError, ValueError):
            return set()

    def pid_alive(self, pid: int) -> bool:
        """Alive and not a zombie."""
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
        except OSError:
            return False
        return stat.rsplit(")", 1)[-1].split()[:1] not in (["Z"], ["X"])

    def signal(self, pid: int, sig: int) -> None:
        try:
            os.kill(int(pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def kit_logs(self) -> list:
        return sorted(KIT_LOG_ROOT.glob("*/*/kit_*.log")) if KIT_LOG_ROOT.is_dir() else []

    def read_head(self, p: Path, n: int = 8 << 20) -> str:
        try:
            with open(p, "rb") as fh:
                return fh.read(n).decode(errors="replace")
        except OSError:
            return ""

    def capture(self, cmd: list, *, env: dict | None = None, timeout: float = 600) -> tuple:
        """A Kit-free CPU read: (rc, stdout)."""
        p = subprocess.run([str(c) for c in cmd], env={**clean_env(), **(env or {})}, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout

    def notify_hook(self, text: str) -> None:
        cmd = os.environ.get("TP_NOTIFY_CMD")
        if cmd:
            subprocess.run([*cmd.split(), text], env=clean_env())


#: TP_* variables a child keeps (the orchestrator's notify hook must reach the detached queue)
KEEP_TP = ("TP_NOTIFY_CMD",)


def clean_env() -> dict:
    """The parent environment with every KRABBY_* / SN_WALKER_* / P2_* / TP_* variable removed (a child gets only what
    the queue passes explicitly), except :data:`KEEP_TP`."""
    return {k: v for k, v in os.environ.items()
            if k in KEEP_TP or not k.startswith(("KRABBY_", "SN_WALKER_", "P2_", "P2R_", "SH_", "SN_STAND_", "TP_"))}


class StopRequested(Exception):
    """STOP_PIPELINE appeared while the queue was about to launch an Isaac process: stop, launch nothing."""


class Halted(Exception):
    """A HALT record was written below the stage executor (e.g. :func:`pre_launch` inside a row): the stage ends."""


# ============================================================================== one Isaac at a time / stopping
def wait_isaac_idle(ctx: "Ctx", sid: str, what: str) -> None:
    """Immediately before EVERY Isaac launch: wait while ANY Isaac python runs (``tp_lib.ISAAC_PAT`` minus the Kit-free
    stand reads: drivers, train / eval / play, every sn_ / sh_ DAgger / table / eval tool), 30-s polls, NO give-up
    (``launch_candidates.sh``'s own wait omits the sn_ / sh_ pythons and gives up after 10 min). STOP_PIPELINE raises
    :class:`StopRequested`: the queue stops before launching (the stage resumes on the next ``run``)."""
    rt, said = ctx.rt, False
    while True:
        if (ctx.dir / "STOP_PIPELINE").exists():
            raise StopRequested(f"STOP_PIPELINE before {what}")
        if not rt.isaac_busy():
            return
        if not said:
            ctx.say("WAIT", f"stage={sid} {what}: another Isaac process is running -- waiting (no give-up)")
            said = True
        rt.sleep(30)


def kit_log_start(p: Path) -> float:
    """A Kit log's start time (``kit_YYYYMMDD_HHMMSS.log``, local time); else its mtime."""
    m = re.search(r"kit_(\d{8}_\d{6})\.log$", Path(p).name)
    if m:
        try:
            return time.mktime(time.strptime(m.group(1), "%Y%m%d_%H%M%S"))
        except ValueError:
            pass
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return 0.0


def ddc_watch_problem(ctx: "Ctx") -> str | None:
    """The DerivedDataCache watch a SIGKILL opened (``state.ddc_watch``): the cache size (> DDC_MAX_GB, or grown
    > DDC_GROWTH_GB since the kill) and every Kit log started since the kill (the migration symptom). A Kit log
    DDC_LOG_SETTLE_S old without the symptom closes the watch. None = nothing wrong (or no watch open)."""
    w = ctx.state.get("ddc_watch")
    if not w or w.get("closed_at"):
        return None
    gb = ddc_gb()
    w["ddc_gb_last"] = gb
    why = None
    if gb > DDC_MAX_GB:
        why = f"Kit DerivedDataCache {gb:.0f} GB > {DDC_MAX_GB}"
    elif gb > float(w.get("ddc_gb_at_kill") or 0.0) + DDC_GROWTH_GB:
        why = f"Kit DerivedDataCache grew to {gb:.0f} GB from {float(w.get('ddc_gb_at_kill') or 0):.0f} GB at the kill"
    logs = [p for p in ctx.rt.kit_logs() if kit_log_start(p) >= float(w["at_epoch"]) - 1.0]
    if why is None:
        for p in logs:
            t = ctx.rt.read_head(p)
            if DDC_SYMPTOM.search(t) or len(DDC_LOOP.findall(t)) >= DDC_LOOP_MIN:
                why = f"Kit log {p} shows the DerivedDataCache migration symptom"
                break
    if why:
        w["symptom"] = why
        ctx.save()
        return (f"after the SIGKILL of {w.get('stage')} ({w.get('at')}): {why} -- quarantine {DDC} "
                "(mv DerivedDataCache DerivedDataCache.broken_<date>), then --retry")
    settled = [p for p in logs if ctx.rt.time() - kit_log_start(p) >= DDC_LOG_SETTLE_S]
    if settled:
        w["closed_at"] = L.now_iso()
        w["clean_kit_log"] = str(settled[0])
        ctx.notify("NOTIFY", f"DerivedDataCache watch closed: {settled[0].name} started clean ({gb:.0f} GB)")
    ctx.save()
    return None


def pre_launch(ctx: "Ctx", sid: str, what: str) -> None:
    """Every Isaac launch goes through here: :func:`wait_isaac_idle`, then the post-SIGKILL DerivedDataCache watch
    (a problem HALTs the stage GUARD and raises :class:`Halted`)."""
    wait_isaac_idle(ctx, sid, what)
    bad = ddc_watch_problem(ctx)
    if bad:
        halt(ctx, sid, "GUARD", bad, resumable=("retry",))
        raise Halted(bad)


def trainer_pids(rt, unit: str | None, match=None) -> list:
    """The Isaac python(s) of a non-driver stage: in the scope's own cgroup, or (no scope yet / gone) any Isaac python
    whose command line satisfies ``match`` (a predicate on the ``ps`` args). Never ``timeout`` / the launcher."""
    scope = rt.scope_pids(unit) if unit else set()
    return sorted(pid for pid, args in rt.ps_pid_args()
                  if L.isaac_busy(args) and (pid in scope or (match is not None and match(args))))


def _wait_until(rt, cond, seconds: float, step: float = 5.0) -> bool:
    t0 = rt.time()
    while True:
        if cond():
            return True
        if rt.time() - t0 >= seconds:
            return False
        rt.sleep(step)


def stop_trainer(rt, *, unit: str | None, match, proc, grace: float = TERM_GRACE_S) -> dict:
    """``run_phases._stop_child`` for a non-driver stage, on the trainer python's OWN pid (never the scope: a
    scope-wide SIGTERM also reaches ``timeout``, which re-forwards it -- the C7 hang): SIGTERM, ``grace`` s, SIGTERM
    again, ``grace`` s, and only then SIGKILL (the last resort; ``sigkill`` in the result -- the caller records it,
    notifies and opens the DerivedDataCache watch). A launcher still before its trainer (none found within
    TRAINER_FIND_S) is SIGTERMed itself (bash, no Kit); a trainer forked in that race is stopped like any other."""
    t0 = rt.time()
    out = {"at": L.now_iso(), "unit": unit, "pids": [], "terms": 0, "sigkill": False, "grace_s": grace}
    pids = trainer_pids(rt, unit, match)
    while not pids and proc.poll() is None and rt.time() - t0 < TRAINER_FIND_S:
        rt.sleep(5)
        pids = trainer_pids(rt, unit, match)
    if not pids:
        if proc.poll() is None:
            out["launcher_terminated"] = True
            try:
                proc.terminate()
            except (OSError, AttributeError):
                pass
            _wait_until(rt, lambda: proc.poll() is not None, 60)
            pids = trainer_pids(rt, unit, match)
        if not pids:
            out["note"] = "no trainer python found (it had exited, or never started)"
            return out
    out["pids"] = pids

    def gone():
        return not any(rt.pid_alive(p) for p in pids)

    for n in (1, 2):
        for p in pids:
            if rt.pid_alive(p):
                rt.signal(p, signal.SIGTERM)
        out["terms"] = n
        if _wait_until(rt, gone, grace):
            out["exited_after_s"] = round(rt.time() - t0, 1)
            return out
    for p in pids:
        if rt.pid_alive(p):
            rt.signal(p, signal.SIGKILL)
    out["sigkill"] = True
    out["exited"] = _wait_until(rt, gone, KILL_WAIT_S)
    out["exited_after_s"] = round(rt.time() - t0, 1)
    return out


def progress_mtime(files) -> float | None:
    """The newest mtime among a stage's progress files (train log, tfevents, checkpoints); None before any exists."""
    ts = []
    for f in files:
        try:
            ts.append(Path(f).stat().st_mtime)
        except OSError:
            continue
    return max(ts) if ts else None


def student_progress_files(rd: Path | None) -> list:
    return [*rd.glob("events.out.tfevents.*"), *rd.glob("model_*.pt")] if rd is not None and rd.is_dir() else []


def dagger_progress_files(run_dir: Path) -> list:
    if not run_dir.is_dir():
        return []
    return [run_dir / "train_log.jsonl", *run_dir.glob("model_*.pt"), *run_dir.glob("events.out.tfevents.*")]


# ============================================================================== the run context
class Ctx:
    def __init__(self, spec: Spec, run: str, rt: Runtime | None = None):
        self.spec, self.run = spec, run
        self.dir = spec.run_dir(run)
        self.state_path = self.dir / "pipeline_state.json"
        self.qlog = self.dir / "pipeline_queue.log"
        self.notify_path = self.dir / "pipeline_notify.jsonl"
        self.rt = rt or Runtime(self.dir / "pipeline_children.log")
        self.plan = L.read_json(self.dir / "plan.json")
        self.state = L.read_json(self.state_path) or self._new_state()

    @classmethod
    def open(cls, spec_path, run, rt=None) -> "Ctx":
        return cls(Spec(spec_path), run, rt)

    def _new_state(self) -> dict:
        return {"schema": 1, "run": self.run, "created": L.now_iso(), "status": "new", "pin": None,
                "stages": {}, "halts": {}, "accepted": [], "amendments": []}

    def save(self) -> None:
        L.write_json_atomic(self.state_path, self.state)

    def srec(self, sid: str) -> dict:
        return self.state["stages"].setdefault(sid, {"status": "pending", "history": []})

    def set_status(self, sid: str, status: str, why: str = "") -> None:
        r = self.srec(sid)
        r["history"].append({"at": L.now_iso(), "from": r.get("status"), "to": status, "why": why})
        r["status"] = status
        self.save()

    def say(self, kind: str, text: str) -> str:
        line = L.sentinel(self.qlog, kind, f"run={self.run} {text}")
        return line

    def notify(self, kind: str, text: str, **data) -> None:
        line = self.say(kind, text)
        L.append_jsonl(self.notify_path, {"at": L.now_iso(), "kind": kind, "text": text, **data})
        try:
            self.rt.notify_hook(line)
        except Exception:  # noqa: BLE001  (a hook failure never stops the queue; the sentinel is the record)
            pass

    # -- records
    def stage_read_path(self, sid: str) -> Path:
        return self.dir / "stage_reads" / f"{sid}.json"

    def head(self, sid: str) -> dict | None:
        return L.read_json(self.stage_read_path(sid))

    def rows_path(self, sid: str) -> Path:
        return self.dir / "stage_reads" / f"{sid}.rows.json"

    def rows(self, sid: str) -> dict:
        return L.read_json(self.rows_path(sid), {}) or {}

    def env_file(self, rel: str) -> Path:
        return self.dir / rel

    def pin(self) -> dict:
        lock = L.read_json(self.spec.lock_path()) or {}
        return {"code_sha": lock.get("code_sha"), "eval_sha": lock.get("eval_sha")}

    def thresholds(self):
        return GR.Thresholds.load(self.spec.thresholds_path())


# ============================================================================== driver records
def driver_state(ctx: Ctx) -> dict:
    return L.read_json(ctx.dir / "state.json", {}) or {}


def reference_head(ctx: Ctx, chain: str, seed: int) -> dict | None:
    """The driver's newest finished run of reference ``chain`` at ``seed`` (``state.json`` ladder.references)."""
    recs = ((driver_state(ctx).get("ladder") or {}).get("references") or {}).get(chain) or []
    recs = [r for r in recs if str(r.get("seed")) == str(seed)]
    return recs[-1] if recs else None


#: ``parkour/scripts/rsl_rl/train.py`` prints the experiment root, then ONLY the timestamp half of the run dir: the dir
#: itself is ``<timestamp>_<run_name>`` whenever ``--run_name`` is given, and the driver always gives it
#: (``run_candidates``: ``run_name=train_tag``). ``run_phases.train_retry`` keeps that run_name on its infra retry,
#: whose log / current tag is ``<train_tag>_r2``.
RE_TRAIN_EXP_ROOT = re.compile(r"\[INFO\] Logging experiment in directory: (\S+)")
RE_TRAIN_EXACT_NAME = re.compile(r"Exact experiment name requested from command line: (\S+)")
TRAIN_LOG_SUFFIX = "_train.log"
DRIVER_RETRY_TAG_SUFFIX = "_r2"
#: the two train.py lines sit in the first ~60 lines of a train log; past this many bytes the whole log is read
TRAIN_LOG_HEAD_BYTES = 4 << 20


def _train_log_text(log: Path) -> str:
    with open(log, "rb") as fh:
        head = fh.read(TRAIN_LOG_HEAD_BYTES).decode(errors="ignore")
    if RE_TRAIN_EXACT_NAME.search(head) or log.stat().st_size <= TRAIN_LOG_HEAD_BYTES:
        return head
    return log.read_text(errors="ignore")


def _driver_stage_recs(ds: dict, xid: str) -> list[dict]:
    """The driver's per-stage records (``state.json`` keys[<seed key>]["<xid>:<stage>"]) of experiment ``xid``."""
    return [r for recs in (ds.get("keys") or {}).values() for k, r in (recs or {}).items()
            if isinstance(r, dict) and str(k).split(":", 1)[0] == xid]


def driver_window(ctx: Ctx, sid: str, prev: dict | None = None) -> dict:
    """The driver's running (or just-finished) training window of driver stage ``sid``, resolved to its run dir.

    ``{"state": ok | pending | ambiguous | mismatch, "run_dir": str | None, "why": str, "window": {...}}``; only
    ``ok`` carries a run dir. The dir is resolved from the driver's own records, never guessed:

    * the window must be this stage's: ``state.json`` current names a ``<tag>_train.log`` whose tag carries an
      experiment id (``_<xid>_``) the driver registered for this stage's chain and seed (a stale current from the
      previous stage's driver, a bootstrap, an eval or a replay is ``pending``);
    * that log's own train.py lines give the experiment root (cross-checked against the stage's phase experiment)
      and the run's timestamp; the dir is the ONE ``<root>/<timestamp>_<run_name>`` whose run_name is the window's
      tag (or the tag minus the driver's ``_r2`` infra-retry suffix). A timestamp match carrying another name is a
      ``mismatch``, two matches are ``ambiguous`` -- neither is read;
    * once the driver has recorded the window (same log), its checkpoint must lie in that dir (else ``mismatch``);
    * while the driver EVALUATES the window it just trained (current kind ``eval`` keeps the train record's started /
      resume / target_final / phase), the dir ``prev`` resolved for that same window stays valid.

    ``pending`` = the window or its dir does not exist yet (train.py prints the timestamp a few seconds before the
    runner creates the dir)."""
    ds = driver_state(ctx)
    cur = ds.get("current") or {}
    chain, seed = ctx.spec.chain(ctx.run, sid), ctx.spec.seed(ctx.run, sid)
    win = {k: cur.get(k) for k in ("started", "resume", "target_final", "phase")}

    def out(state: str, why: str, rd: Path | None = None, **extra) -> dict:
        return {"state": state, "run_dir": str(rd) if rd is not None else None, "why": why, "window": win, **extra}

    exps = (ds.get("ladder") or {}).get("experiments") or {}
    xids = sorted(x for x, e in exps.items()
                  if isinstance(e, dict) and e.get("chain") == chain and str(e.get("seed")) == str(seed))
    if not xids:
        return out("pending", f"the driver has not registered chain {chain} s{seed} yet")
    log = Path(str(cur.get("log") or ""))
    if not log.name.endswith(TRAIN_LOG_SUFFIX):
        if (cur.get("kind") == "eval" and prev and prev.get("state") == "ok" and prev.get("window") == win
                and prev.get("run_dir") and Path(prev["run_dir"]).is_dir()):
            return out("ok", f"the driver is evaluating the window it just trained ({cur.get('tag')}): "
                             f"{Path(prev['run_dir']).name} as resolved during training", Path(prev["run_dir"]),
                       tag=prev.get("tag"), log=prev.get("log"))
        return out("pending", f"no training window: current is {cur.get('kind')} {cur.get('tag')}")
    tag = str(cur.get("tag") or "")
    if log.name != f"{tag}{TRAIN_LOG_SUFFIX}":
        return out("mismatch", f"state.json current tag {tag!r} does not name its log {log.name}")
    mine = [x for x in xids if f"_{x}_" in f"_{tag}_"]
    if not mine:
        return out("pending", f"the current window {tag} is not chain {chain}'s ({', '.join(xids)})")
    if len(mine) > 1:
        return out("ambiguous", f"the window tag {tag} carries several experiments of chain {chain}: {mine}")
    if not log.is_file():
        return out("pending", f"{log.name} is not written yet")
    txt = _train_log_text(log)
    stamps = sorted(set(RE_TRAIN_EXACT_NAME.findall(txt)))
    if not stamps:
        return out("pending", f"train.py has not printed its run name in {log.name} yet")
    if len(stamps) > 1:
        return out("ambiguous", f"{log.name} names several runs {stamps}")
    ts = stamps[0]
    exp = L.rc().ph.PHASES[ctx.spec.stage(sid)["base"]].experiment
    want_root = L.LOG_ROOT / exp
    roots = sorted(set(RE_TRAIN_EXP_ROOT.findall(txt)))
    if len(roots) > 1:
        return out("ambiguous", f"{log.name} names several experiment roots {roots}")
    root = Path(roots[0]) if roots else want_root
    if root.resolve() != want_root.resolve():
        return out("mismatch", f"{log.name} logs to {root}, stage {sid}'s phase experiment is {want_root}")
    try:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and (p.name == ts or p.name.startswith(f"{ts}_")))
    except OSError:
        names = []
    if not names:
        return out("pending", f"no run dir {root.name}/{ts}_* yet")
    expect = {tag}
    if tag.endswith(DRIVER_RETRY_TAG_SUFFIX):
        expect.add(tag[:-len(DRIVER_RETRY_TAG_SUFFIX)])
    hits = [n for n in names if n[len(ts) + 1:] in expect]
    if len(hits) > 1:
        return out("ambiguous", f"{root.name} holds several dirs of window {tag}: {hits}")
    if not hits:
        return out("mismatch", f"{root.name}/{ts}* holds {names}, none carries the run_name {sorted(expect)}")
    rd = root / hits[0]
    for r in _driver_stage_recs(ds, mine[0]):
        if str(r.get("log") or "") == str(log) and r.get("ckpt") and Path(r["ckpt"]).parent != rd:
            return out("mismatch", f"the driver recorded {r['ckpt']} for {log.name}, not a checkpoint of {rd}")
    return out("ok", f"{rd.name} ({log.name})", rd, tag=tag, log=str(log))


def driver_run_dir(ctx: Ctx, sid: str) -> Path | None:
    """The run dir of the driver's running (or just-finished) training window of ``sid`` (:func:`driver_window`), or
    None. The resolution is kept in the stage record (``live_resolve``); an ``ambiguous`` / ``mismatch`` read is said
    and notified once per distinct reason (the live rules are then NOT read -- never on a guessed run)."""
    rec = ctx.srec(sid)
    prev = rec.get("live_resolve")
    res = driver_window(ctx, sid, prev)
    if (res["state"], res["why"], res["run_dir"]) != tuple((prev or {}).get(k) for k in ("state", "why", "run_dir")):
        rec["live_resolve"] = {**res, "at": L.now_iso()}
        ctx.save()
        if res["state"] in ("ambiguous", "mismatch"):
            ctx.notify("NOTIFY", f"stage={sid} live watch: the driver's run dir is {res['state']} ({res['why']}) -- "
                                 "the live rules are not read until it resolves")
        elif res["state"] == "ok" and (prev or {}).get("run_dir") != res["run_dir"]:
            ctx.say("LIVE", f"stage={sid} run_dir={res['run_dir']}")
    return Path(res["run_dir"]) if res["state"] == "ok" else None


def driver_stamp(ctx: Ctx, head: dict, row: str = "suite__omni", es: int = 1) -> Path | None:
    """The driver's own stage-eval stamp of a head (``gait_eval/omni_phases/<run dir>/<tag>/<row>/seed00k/<ts>``).

    The driver's record first: the stage record's ``evals[*].run_dir`` (keys["<xid>:<stage>"]), the stamp the driver
    scored. Its eval tag is the window's ORIGINAL tag, which differs from the run dir's run_name on a continued
    (``_c``), re-run (``_w``) or experiment-retried (``_e2``) window, so the name-derived lookup below is only the
    fallback for a head the driver recorded without evals."""
    want = (row, f"seed{es:03d}")
    ds = driver_state(ctx)
    recorded = []
    for st, ck in (head.get("stage_ckpts") or {}).items():
        if str(ck) != str(head.get("ckpt")):
            continue
        for r in _driver_stage_recs(ds, str(head.get("xid"))):
            if r.get("phase") == st and str(r.get("ckpt")) == str(ck):
                recorded += [Path(e["run_dir"]) for e in (r.get("evals") or {}).values()
                             if isinstance(e, dict) and e.get("run_dir")]
    recorded = sorted({p for p in recorded if tuple(p.parts[-3:-1]) == want and (p / "run_meta.json").is_file()})
    if len(recorded) == 1:
        return recorded[0]
    if len(recorded) > 1:
        ctx.notify("NOTIFY", f"driver_stamp: {len(recorded)} recorded {row} es{es} stamps for {head.get('ckpt')} "
                             f"({[p.name for p in recorded]}): none taken")
        return None
    tag = Path(head["ckpt"]).parent.name.split("_", 2)[-1]
    d = L.EVAL_ROOT / ctx.dir.name / tag / row / f"seed{es:03d}"
    hits = sorted(p for p in d.glob("*/") if (p / "run_meta.json").is_file()) if d.is_dir() else []
    return hits[-1] if hits else None


# ============================================================================== HALT records
def dependents(ctx: Ctx, sid: str) -> list[str]:
    """Every stage that (transitively) needs / settles on ``sid``."""
    out, frontier = [], {sid}
    for st in ctx.spec.stages(ctx.run):
        deps = set(st.get("needs") or []) | set(st.get("settled") or [])
        if deps & frontier:
            out.append(st["id"])
            frontier.add(st["id"])
    return out


def _modelled_rate(x: dict) -> bool:
    """A HALT line with a modelled chance rate: a number AND not a degenerate line (decision C: a degenerate line's
    rate is None in v1.1; a v0 / v1 line's plain 0.0 is caught by its ``rate_status`` / ``degenerate`` mark)."""
    return isinstance(x.get("rate"), (int, float)) and not isinstance(x.get("rate"), bool) and not GR.is_degenerate(x)


def watch_reads_text(flags) -> str:
    """``id=value (line L)`` per WATCH flag (a notification carries the reads, not only the ids)."""
    def num(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "; ".join(f"{f.get('id')}={num(f.get('value'))} (line {num(f.get('line'))})" for f in (flags or [])) or "-"


def chance_text(rec: dict) -> str:
    """The ``chance=`` field of a HALT notification: never a bare 0.0 / sum for a line without a modelled rate."""
    um = rec.get("chance_rate_unmodelled") or []
    s = rec.get("chance_rate_sum")
    if not um:
        return f"chance={s}"
    kinds = sorted({u.get("rate_status") or "no rate" for u in um})
    tail = f"{len(um)} unmodelled: {', '.join(kinds)}"
    return f"chance={s} ({tail})" if s is None else f"chance={s} (+{tail})"


def halt(ctx: Ctx, sid: str, klass: str, why: str, *, lines: list | None = None, gate: str | None = None,
         resumable: tuple = ("accept-chance", "amend", "abort")) -> dict:
    """A HALT record + sentinel + park. ``chance_rate_sum`` sums the MODELLED rates of the failed lines only; every line
    without one -- a degenerate fixed-t line (rate None + ``rate_status`` 'degenerate/unvalidated', CHARTER 2026-10-21
    decision C), a live kill, a drift line -- is listed in ``chance_rate_unmodelled`` and named in the notification, so
    a HALT never quotes 'chance=0.0' for an unmodelled line."""
    lines = [dict(x) for x in (lines or [])]
    for x in lines:
        if GR.is_degenerate(x):                   # a v0 / v1 gate record's plain 0.0: no modelled rate
            x.update(rate=None, rate_status=GR.DEGENERATE)
    rates = [x.get("rate") for x in lines if _modelled_rate(x)]
    unmod = [{"id": x.get("id"), "rate_status": x.get("rate_status")} for x in lines if not _modelled_rate(x)]
    rec = {"run": ctx.run, "stage": sid, "class": klass, "why": why, "at": L.now_iso(), "status": "open",
           "lines": [{k: x.get(k) for k in ("id", "statistic", "value", "line", "direction", "rate", "rate_status",
                                             "unvalidated", "env_seed", "why") if k in x} for x in lines],
           "chance_rate_sum": sum(rates) if rates else None,
           "chance_rate_unmodelled": unmod,
           "chance_rate_note": ("each failed decisive line's chance-HALT rate from thresholds.json (PLAN Rule; "
                                "reported, not capped), summed over the lines WITH a modelled rate; "
                                "chance_rate_unmodelled lists the rest -- degenerate fixed-t lines (sigma_hat 0: rate "
                                "None, 'degenerate/unvalidated', CHARTER 2026-10-21 decision C), live-kill and drift "
                                "lines (unvalidated, no rate)"),
           "gate": gate, "subtree": dependents(ctx, sid), "options": list(resumable)}
    n = len(list((ctx.dir / "halts").glob(f"{sid}.*.json"))) + 1 if (ctx.dir / "halts").is_dir() else 1
    p = ctx.dir / "halts" / f"{sid}.{n}.json"
    L.write_json_atomic(p, rec)
    ctx.state["halts"][sid] = {"class": klass, "status": "open", "record": L.rel(p)}
    ctx.set_status(sid, "halt", f"{klass}: {why}")
    first = rec["lines"][0] if rec["lines"] else {}
    ctx.notify("HALT", f"stage={sid} class={klass} record={L.rel(p)} line={first.get('id')} read={first.get('value')} "
                       f"{chance_text(rec)} why={why[:200]}", stage=sid, record=L.rel(p))
    return rec


# ============================================================================== rows
class RowFail(Exception):
    pass


_SHA_CACHE: dict = {}


def ckpt_sha(path) -> str:
    """sha256 of a checkpoint (cached on path + mtime + size: the rows of one head hash it once)."""
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = L.sha256_file(p)
    return _SHA_CACHE[key]


def stamp_ckpt_sha(stamp: Path):
    try:
        return json.loads((Path(stamp) / "run_meta.json").read_text()).get("checkpoint_sha256")
    except (OSError, ValueError):
        return None


def row_job(ctx: Ctx, sid: str, key: str, *, ckpt: str, name: str, env_stage: str | None = None,
            level: int | None = None, extra_args: tuple = (), strip: tuple = ()) -> dict:
    r = parse_row(key)
    fam = r["family"]
    spec_r = ctx.spec.row(fam)
    es_stage = env_stage or sid
    pe = ctx.plan["stages"][es_stage]
    env_rel = pe["row_env"].get(fam, pe["env"])
    args = list(spec_r.get("args") or []) + list(extra_args)
    if spec_r.get("trained_categories") and pe.get("trained_csv"):
        args += ["--trained-categories", pe["trained_csv"]]
    return {"key": key + (f"@L{level:03d}" if level is not None else ""), "row": r["key"], "family": fam,
            "es": r["es"], "name": name, "ckpt": ckpt, "env_file": str(ctx.env_file(env_rel)),
            "level_file": (str(ctx.env_file(ctx.plan["levels"][str(level)])) if level is not None else None),
            "manifest": str(ctx.spec.manifest(fam)), "scenario": spec_r["scenario"], "args": args,
            "strip": list(strip), "level": level}


def run_row(ctx: Ctx, sid: str, job: dict) -> Path:
    """One gate row through gate_row.sh (idempotent: a recorded complete stamp is reused). Retried once."""
    rows = ctx.rows(sid)
    have = rows.get(job["key"])
    if have and Path(have["stamp"]).is_dir() and (Path(have["stamp"]) / "tp_row.json").is_file():
        if stamp_ckpt_sha(Path(have["stamp"])) == ckpt_sha(job["ckpt"]):
            return Path(have["stamp"])
        ctx.say("ROW", f"stage={sid} row={job['key']}: the recorded stamp is of another checkpoint -- rerun")
    out_root = L.REPO / ctx.spec.d["row_out"] / ctx.run
    tmo = (ctx.spec.d["rows"].get(job["family"]) or {}).get("timeout_s") or ctx.spec.d.get("row_timeout_s") or 5400
    env = {"TP_OUT": str(out_root), "TP_ROW_TIMEOUT": str(int(tmo))}
    if job.get("level_file"):
        env["TP_LEVEL_FILE"] = job["level_file"]
    if job.get("strip"):
        env["TP_STRIP_KEYS"] = " ".join(job["strip"])
    cmd = [str(HERE / "gate_row.sh"), job["name"], job["ckpt"], job["env_file"], job["manifest"], job["scenario"],
           str(job["es"]), *job["args"]]
    last = None
    for attempt in (1, 2):
        pre_launch(ctx, sid, f"row {job['key']}")
        ctx.say("ROW", f"stage={sid} row={job['key']} name={job['name']} attempt={attempt}")
        rc, last = ctx.rt.run(cmd, env=env, log=ctx.dir / "tp_logs" / f"rows_{sid}.log")
        m = re.search(r"TPROW END name=\S+ rc=(\d+) stamp=(\S*)", last or "")
        if rc == 0 and m and m.group(1) == "0" and m.group(2):
            rows = ctx.rows(sid)
            rows[job["key"]] = {"stamp": m.group(2), "job": job, "at": L.now_iso()}
            L.write_json_atomic(ctx.rows_path(sid), rows)
            return Path(m.group(2))
    raise RowFail(f"row {job['key']} ({job['scenario']} es{job['es']}) failed twice: {last}")


# ============================================================================== gate read of a walker / payload stage
def expect_for(ctx: Ctx, sid: str, job_or_row: dict, *, source: str, head: dict, env_stage: str | None = None,
               level=None, driver_pin=None, task=None) -> dict:
    pe = ctx.plan["stages"][env_stage or sid]
    fam = job_or_row["family"]
    spec_r = ctx.spec.row(fam)
    env_rel = pe["row_env"].get(fam, pe["env"])
    env = L.read_env_file(ctx.env_file(env_rel))
    for k in job_or_row.get("strip") or []:
        env.pop(k, None)
    if level is not None:
        env.update(L.read_env_file(ctx.env_file(ctx.plan["levels"][str(level)])))
    return {"label": job_or_row.get("key") or fam, "stage": env_stage or sid, "pin": ctx.pin(), "source": source,
            "driver_pin": driver_pin, "env": env, "carry": ctx.spec.d["carry_env"], "ckpt_sha": head["sha256"],
            "trained": (pe["trained"] if spec_r.get("trained_categories") else None),
            "level_kg": level, "adapter": ("body_hold" if "body_hold" in (spec_r.get("args") or []) else None),
            "task": task}


def gate0(ctx: Ctx, sid: str, stamps: dict, expects: dict) -> list[str]:
    out = []
    for key, st in sorted(stamps.items()):
        out += GR.stamp_problems(Path(st), expects[key])
    return out


def write_gate(ctx: Ctx, sid: str, res: dict) -> Path:
    p = ctx.dir / "gates" / f"{sid}.json"
    th = ctx.spec.thresholds_path()
    res = {"run": ctx.run, "stage": sid, "read_at": L.now_iso(),
           "thresholds": {"file": th.name, "version": ctx.spec.d["thresholds"]["version"],
                          "sha256": L.sha256_file(th)}, **res}
    L.write_json_atomic(p, res)
    return p


def stage_stamps(ctx: Ctx, sid: str, head: dict, keys: list) -> tuple:
    """{row key: stamp} and {row key: Gate-0 expectation} for a walker stage's recorded rows + its driver stamp."""
    rows = ctx.rows(sid)
    stamps, expects = {}, {}
    for key in keys:
        r = parse_row(key)
        if r["driver"]:
            st = head.get("driver_stamp")
            if not st:
                stamps[r["key"]] = Path("/nonexistent/driver_stamp")
            else:
                stamps[r["key"]] = Path(st)
            expects[r["key"]] = expect_for(ctx, sid, r, source="driver", head=head,
                                           driver_pin={"code_sha": head.get("code_sha"),
                                                       "eval_sha": head.get("eval_sha")}, level=None)
        else:
            rec = rows.get(r["key"])
            stamps[r["key"]] = Path(rec["stamp"]) if rec else Path(f"/nonexistent/{r['key']}")
            expects[r["key"]] = expect_for(ctx, sid, {**r, **((rec or {}).get("job") or {})}, source="row",
                                           head=head, level=None)
    return stamps, expects


def gate_walker(ctx: Ctx, sid: str) -> dict:
    head = ctx.head(sid)
    th = ctx.thresholds()
    keys = gate_rows(ctx.spec, sid)
    stamps, expects = stage_stamps(ctx, sid, head, keys)
    g0 = gate0(ctx, sid, stamps, expects)
    trained = ctx.plan["stages"][sid]["trained"]
    dec = GR.walker_decisive(sid, stamps, trained, th) if not g0 else []
    live = ctx.srec(sid).get("live") or {}
    dec.append({"id": f"{sid}.live_collapse", "statistic": "live rules (unvalidated)", "value": live.get("last_rc"),
                "pass": True, "unvalidated": True, "rate": None})
    flags = []
    if sid in ("s1", "c4", "c5", "c6", "c6x") and not g0:
        std = GR.walker_decisive(sid, stamps, trained, th, standard=True)
        flags = [{"id": d["id"], "why": "misses the STANDARD (0.90 / 0.70 / FAST-FAST) line (WATCH)",
                  "value": d.get("value"), "line": d.get("line")} for d in std if not d.get("pass")]
    # CHARTER 2026-10-19 (R18 live-watch defect): a stage whose live watch never ran -- no rec["live"] -- or whose last
    # read was not RC_PASS is flagged WATCH instead of passing the live line silently
    if live.get("last_rc") != LW().RC_PASS:
        flags.append({"id": f"{sid}.live_collapse", "why": "the live watch has no PASS read for this stage "
                      "(never ran or last read not RC_PASS) -- the live line is vacuous (WATCH)",
                      "value": live.get("last_rc"), "line": None})
    verdict = GR.verdict_of(dec, g0, watch_flags=flags)
    res = {"verdict": verdict, "gate0": g0, "decisive": dec, "halt_lines": [d["id"] for d in dec if not d.get("pass")],
           "watch_flags": flags, "watch": GR.watch_reads(sid, {k: v for k, v in stamps.items() if Path(v).is_dir()},
                                                         ctx.spec.d["rows"]),
           "trained_categories": trained, "head": head}
    res["path"] = str(write_gate(ctx, sid, res))
    return res


# ============================================================================== payload stages
def level_keys(ctx: Ctx) -> list[str]:
    return list(ctx.spec.d["payload"]["level_rows"])


def payload_rows_and_curve(ctx: Ctx, sid: str) -> dict:
    """The 0-kg check rows, then the curve: levels ascending, each level's rows then its stability read, stopping at
    the first unstable level. Returns {curve: {L: stable}, reads: {L: read}}."""
    head = ctx.head(sid)
    name = f"T4{ctx.run}_{sid}"
    for key in gate_rows(ctx.spec, sid):
        r = parse_row(key)
        if r["driver"]:
            continue
        run_row(ctx, sid, row_job(ctx, sid, key, ckpt=head["ckpt"], name=name))
    curve, reads = {}, {}
    th_doc = json.loads(ctx.spec.thresholds_path().read_text())
    trained = [c for c in ctx.plan["stages"][sid]["trained"] if c in GR.TRANSLATE]
    for kg in ctx.spec.d["payload"]["levels"]:
        stamps = {}
        for key in level_keys(ctx):
            job = row_job(ctx, sid, key, ckpt=head["ckpt"], name=f"{name}_L{int(kg):03d}", level=int(kg))
            stamps[key] = run_row(ctx, sid, job)
        rd = PR().level_reads(stamps, th_doc, trained_translate=trained)
        curve[int(kg)], reads[int(kg)] = rd["stable"], rd
        rec = ctx.srec(sid)
        rec.setdefault("curve", {})[str(kg)] = rd["stable"]
        ctx.save()
        ctx.say("LEVEL", f"stage={sid} level={kg} stable={rd['stable']} failed={rd['failed'][:4]}")
        if not rd["stable"]:
            break
    return {"curve": curve, "reads": reads}


def p2a_zero_kg_stamps(ctx: Ctx) -> dict:
    rows = ctx.rows("p2a")
    head = ctx.head("p2a")
    out = {"suite": {1: head["driver_stamp"]}}
    if "suite@2" in rows:
        out["suite"][2] = rows["suite@2"]["stamp"]
    out["side22"] = (rows.get("side22@1") or {}).get("stamp")
    for row in ("field_hard", "field_light", "ring_obst", "field_turn"):
        out[row] = [rows[f"{row}@{es}"]["stamp"] for es in (1, 2) if f"{row}@{es}" in rows]
    return out


def head_zero_kg_stamps(ctx: Ctx, sid: str) -> dict:
    rows = ctx.rows(sid)
    head = ctx.head(sid)
    out = {"suite": {1: head["driver_stamp"]}}
    if "suite@2" in rows:
        out["suite"][2] = rows["suite@2"]["stamp"]
    out["side22"] = (rows.get("side22@1") or {}).get("stamp")
    for row in ("field_hard", "field_light", "ring_obst", "field_turn"):
        out[row] = [rows[f"{row}@{es}"]["stamp"] for es in (1, 2) if f"{row}@{es}" in rows]
    return out


def zero_kg_of(ctx: Ctx, sid: str, th) -> dict:
    """The paired 0-kg check (PLAN Payload 3f) of payload head ``sid`` against the run's own p2a head under the
    thresholds view ``th`` (gate_read.zero_kg_check: the line's translate mode, its drift). One call site for the gate
    and for the T* retarget re-gate (:func:`do_retarget_tstar`), so the two cannot read the check differently."""
    trained = ctx.plan["stages"][sid]["trained"]
    return GR.zero_kg_check(head_zero_kg_stamps(ctx, sid), p2a_zero_kg_stamps(ctx),
                            [c for c in trained if c in GR.TRANSLATE], th, its_past_p2a(ctx, sid),
                            resolution_files(ctx))


def zero_kg_record(zk: dict) -> dict:
    """The stage record's ``zero_kg`` (what T* reads): the verdict, the max margin use, the failing statistics and the
    translate mode / thresholds version that produced them (records stay unambiguous across a re-gate), plus the
    WATCH reads (from v1.2 the raw translate ratio drop: CHARTER 2026-10-22 keeps it reported, never decisive, so a
    record still shows a speed drop) and the WATCH rows that exceed their margin (``watch_exceeds``)."""
    reads, exceeds = [], []
    for r in zk.get("rows") or []:
        if not r.get("watch") or r.get("worse_by") is None:
            continue
        m = (r.get("margin") or {}).get("margin") if isinstance(r.get("margin"), dict) else None
        reads.append({"statistic": r["statistic"], "row": r["row"], "head": r.get("head"), "p2a": r.get("p2a"),
                      "worse_by": r["worse_by"], "margin": m, "exceeds": bool(r.get("exceeds"))})
        if r.get("exceeds"):
            exceeds.append(f"{r['statistic']} {r['row']}: worse by {r['worse_by']:.4f} > margin {m:.4f} (WATCH)")
    return {"verdict": zk["verdict"], "max_margin_used": zk.get("max_margin_used"), "fails": zk["fails"],
            "translate_statistic": zk.get("translate_statistic"), "thresholds_version": zk.get("thresholds_version"),
            "watch_exceeds": exceeds, "watch_reads": reads}


def zero_kg_line(sid: str, zk: dict) -> dict:
    """The payload gate's decisive 0-kg line (p3a: ``required_level`` set)."""
    return {"id": f"{sid}.capability.zero_kg_paired", "statistic": "paired 0-kg check vs the run's own p2a",
            "value": zk.get("max_margin_used"), "line": 1.0, "direction": "<=", "pass": zk["verdict"] == "PASS",
            "fails": zk["fails"], "rate": None, "unvalidated": True,
            "rate_note": "paired: <= Phi(-3) per statistic (thresholds.json); drift margin unvalidated"}


def payload_verdict(ctx: Ctx, sid: str, dec: list, g0: list, ms) -> tuple:
    """A payload stage's verdict and flags from its decisive lines (PLAN Payload: a p3b / p3c floor miss -- or a
    max_stable below the stage's own top -- is a measured CEILING, never a HALT). Returns (verdict, watch flags)."""
    g = ctx.spec.stage(sid).get("gate") or {}
    top = g.get("top_level")
    ceiling = bool(g.get("ceiling_on_fail"))
    reached_top = ms is not None and top is not None and ms >= top
    flags = []
    if ceiling and not reached_top:
        flags.append({"id": f"{sid}.ceiling", "why": f"max_stable {ms} < its own top {top}: a measured ceiling"})
    verdict = GR.verdict_of(dec, g0, ceiling_on_fail=ceiling, watch_flags=flags)
    if verdict in (GR.PASS, GR.WATCH) and ceiling and not reached_top:
        verdict = GR.CEILING
    return verdict, flags


def gate_payload(ctx: Ctx, sid: str, curve: dict | None = None) -> dict:
    st = ctx.spec.stage(sid)
    g = st.get("gate") or {}
    head = ctx.head(sid)
    th = ctx.thresholds()
    keys = gate_rows(ctx.spec, sid)
    stamps, expects = stage_stamps(ctx, sid, head, keys)
    rows = ctx.rows(sid)
    for k, rec in rows.items():                      # the level rows (key row@es@L###)
        if "@L" in k:
            job = rec["job"]
            stamps[k] = Path(rec["stamp"])
            expects[k] = expect_for(ctx, sid, job, source="row", head=head, level=job["level"])
    g0 = gate0(ctx, sid, stamps, expects)
    curve = curve if curve is not None else {int(k): v for k, v in (ctx.srec(sid).get("curve") or {}).items()}
    ms = PR().max_stable(curve)
    trained = ctx.plan["stages"][sid]["trained"]
    dec = []
    if not g0:
        walker_stamps = {k: v for k, v in stamps.items() if "@L" not in k}
        dec = GR.walker_decisive(sid, walker_stamps, trained, th, th_stage="p3a")
        req = g.get("required_level")
        if req is not None:
            dec.append({"id": f"{sid}.capability.stable_at_{req}", "statistic": f"stable at {req} kg (every level "
                        f"<= {req})", "value": ms, "line": req, "direction": ">=",
                        "pass": ms is not None and ms >= req, "rate": None, "unvalidated": True,
                        "rate_note": "pending step 8 (payload a-priori: the pilot value as the mean)"})
        zk = zero_kg_of(ctx, sid, th)
        ctx.srec(sid)["zero_kg"] = zero_kg_record(zk)
        if req is not None:
            dec.append(zero_kg_line(sid, zk))
    verdict, flags = payload_verdict(ctx, sid, dec, g0, ms)
    res = {"verdict": verdict, "gate0": g0, "decisive": dec, "halt_lines": [d["id"] for d in dec if not d.get("pass")],
           "watch_flags": flags, "max_stable": ms, "curve": curve, "label": PR().LABEL,
           "zero_kg": ctx.srec(sid).get("zero_kg"),
           "watch": GR.watch_reads(sid, {k: v for k, v in stamps.items() if Path(v).is_dir() and "@L" not in k},
                                   ctx.spec.d["rows"]), "head": head}
    if not g0 and verdict == GR.CEILING:
        res["limiting_factor"] = limiting(ctx, sid, curve, ms)
    res["path"] = str(write_gate(ctx, sid, res))
    ctx.srec(sid)["max_stable"] = ms
    ctx.save()
    return res


def limiting(ctx: Ctx, sid: str, curve: dict, ms) -> dict:
    """The limiting factor at the first unstable level (payload_read.limiting_factor)."""
    head = ctx.head(sid)
    try:
        ser = LW().read_series(Path(head["run_dir"]), ("Episode_Termination/crab_failure",
                                                       "Metrics/base_parkour/track_ratio_mean"))
        bb = PR().budget_bound(ser)
    except Exception as exc:  # noqa: BLE001
        bb = {"budget_bound": None, "why": repr(exc)[:200]}
    fail_l = next((lv for lv in sorted(curve) if not curve[lv]), None)
    rows = ctx.rows(sid)
    lr = {}
    if fail_l is not None:
        stamps = {k.rsplit("@L", 1)[0]: Path(r["stamp"]) for k, r in rows.items() if k.endswith(f"@L{fail_l:03d}")}
        th_doc = json.loads(ctx.spec.thresholds_path().read_text())
        lr = PR().level_reads(stamps, th_doc, trained_translate=[c for c in ctx.plan["stages"][sid]["trained"]
                                                                if c in GR.TRANSLATE])
    return {"first_unstable_level": fail_l, "budget": bb,
            **PR().limiting_factor(lr, live_killed=bool(ctx.srec(sid).get("live_killed")), budget=bb)}


def its_of(ckpt: str) -> int:
    m = re.search(r"model_(\d+)\.pt$", str(ckpt))
    return int(m.group(1)) if m else 0


def its_past_p2a(ctx: Ctx, sid: str) -> int:
    return its_of(ctx.head(sid)["ckpt"]) - its_of(ctx.head("p2a")["ckpt"])


def resolution_files(ctx: Ctx) -> list:
    ti = ctx.spec.dir / "threshold_inputs.yaml"
    return list((L.load_yaml(ti).get("resolution_files") or [])) if ti.is_file() else []


# ============================================================================== gate entry (CPU re-read too)
def gate_stage(ctx: Ctx, sid: str, *, thresholds: Path | None = None, out: Path | None = None) -> dict:
    if thresholds is not None:
        ctx.spec.d["thresholds"] = {"file": str(Path(thresholds).resolve()), "version": "re-read"}
    kind = ctx.spec.stage(sid)["kind"]
    cls = (ctx.spec.stage(sid).get("gate") or {}).get("class")
    if kind == "driver" and cls == "payload":
        res = gate_payload(ctx, sid)
    elif kind == "driver":
        res = gate_walker(ctx, sid)
    else:
        res = L.read_json(ctx.dir / "gates" / f"{sid}.json") or {"verdict": GR.INVALID,
                                                                 "halt_lines": ["no gate record"]}
    if out is not None:
        L.write_json_atomic(out, res)
    return res


# ============================================================================== the queue
class Queue:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.spec = ctx.spec

    # -- graph
    def status(self, sid: str) -> str:
        return (self.ctx.state["stages"].get(sid) or {}).get("status", "pending")

    def deps(self, sid: str, key: str) -> list[str]:
        """``needs`` / ``settled`` / ``after`` restricted to the run's own stages (X stages exist on R18 only)."""
        ids = {s["id"] for s in self.spec.stages(self.ctx.run)}
        return [d for d in (self.spec.stage(sid).get(key) or []) if d in ids]

    def blocked_by(self, sid: str) -> list[str]:
        """Dependencies that HALTed (or are blocked / aborted): the stage waits for the user's decision."""
        return [d for k in ("needs", "settled") for d in self.deps(sid, k)
                if self.status(d) in ("halt", "blocked", "aborted")]

    def skipped_by(self, sid: str) -> list[str]:
        """A ``needs`` dependency that ended as a ceiling or was skipped: nothing to build on -> skipped."""
        return [d for d in self.deps(sid, "needs") if self.status(d) in ("ceiling", "skipped")]

    def ready(self, sid: str) -> bool:
        if any(self.status(d) not in SUCCESS for d in self.deps(sid, "needs")):
            return False
        if any(self.status(d) not in SETTLED for d in self.deps(sid, "settled")):
            return False
        if any(self.status(d) in ("pending", "training", "rows", "gate") for d in self.deps(sid, "after")):
            return False
        return True

    def when_ok(self, sid: str) -> bool:
        cond = self.spec.stage(sid).get("when")
        if not cond:
            return True
        m = re.match(r"^max_stable\((\w+)\)\s*>=\s*(\d+)$", cond.strip())
        if not m:
            raise ValueError(f"stage {sid}: unknown condition {cond!r}")
        src = self.ctx.srec(m.group(1))
        ms = src.get("max_stable")
        return src.get("status") in SUCCESS and ms is not None and ms >= int(m.group(2))

    def next_stage(self) -> str | None:
        changed = True
        while changed:
            changed = False
            for st in self.spec.stages(self.ctx.run):
                sid = st["id"]
                s = self.status(sid)
                if s in TERMINAL:
                    continue
                if s == "pending" and self.blocked_by(sid):
                    self.ctx.set_status(sid, "blocked", f"needs {self.blocked_by(sid)}")
                    changed = True
                    continue
                if s == "pending" and self.skipped_by(sid):
                    self.ctx.set_status(sid, "skipped", f"{self.skipped_by(sid)} ended without a head to build on")
                    self.ctx.notify("NOTIFY", f"stage={sid} skipped ({self.skipped_by(sid)} ceiling / skipped)")
                    changed = True
                    continue
                if s == "pending" and self.ready(sid) and not self.when_ok(sid):
                    self.ctx.set_status(sid, "skipped", f"condition {st.get('when')} false")
                    self.ctx.notify("NOTIFY", f"stage={sid} skipped ({st.get('when')})")
                    changed = True
        for st in self.spec.stages(self.ctx.run):
            sid = st["id"]
            if self.status(sid) in ("training", "rows", "gate"):
                return sid                       # an interrupted stage resumes first
        for st in self.spec.stages(self.ctx.run):
            sid = st["id"]
            if self.status(sid) == "pending" and self.ready(sid):
                return sid
        return None

    def stop_requested(self) -> bool:
        return (self.ctx.dir / "STOP_PIPELINE").exists()

    def inject(self, sid: str, kind: str) -> bool:
        """PLAN step 8 full-graph dry pass ONLY (a run with ``dry: true``): ``<run dir>/INJECT.json`` = {stage: [kinds]}
        forces a FAIL / INVALID / GUARD / LIVE once per (stage, kind); never read on an official run."""
        if not self.spec.run(self.ctx.run).get("dry"):
            return False
        inj = L.read_json(self.ctx.dir / "INJECT.json", {}) or {}
        done = self.ctx.state.setdefault("injected", [])
        key = f"{sid}:{kind}"
        if kind in (inj.get(sid) or []) and key not in done:
            done.append(key)
            self.ctx.save()
            self.ctx.say("INJECT", f"stage={sid} kind={kind} (dry pass)")
            return True
        return False

    def guard(self, sid: str) -> bool:
        pin = (self.ctx.state.get("pin") or {}).get("lock_sha256")
        g = guard_problems(self.spec, pin_lock_sha=pin)
        if self.inject(sid, "GUARD"):
            g = g + ["dry-pass injected GUARD"]
        if g:
            halt(self.ctx, sid, "GUARD", "; ".join(g[:6]), resumable=("retry",))
            return False
        gb = ddc_gb()
        if gb > DDC_MAX_GB:
            halt(self.ctx, sid, "GUARD", f"Kit DerivedDataCache {gb:.0f} GB > {DDC_MAX_GB}: quarantine it first",
                 resumable=("retry",))
            return False
        return True

    def loop(self) -> str:
        ctx = self.ctx
        if ctx.state.get("status") == "aborted":
            ctx.say("END", "aborted")
            return "aborted"
        if ctx.state.get("pin") is None:
            lp = self.spec.lock_path()
            ctx.state["pin"] = {"lock_sha256": L.sha256_file(lp) if lp.is_file() else None, "at": L.now_iso(),
                                **ctx.pin()}
        ctx.state["status"] = "running"
        ctx.save()
        L.write_text_atomic(ctx.dir / "queue.pid", f"{os.getpid()}\n")     # the REAL pid (launch_queue's is provisional)
        ctx.say("START", f"pid={os.getpid()}")
        try:
            return self._loop()
        finally:
            qp = ctx.dir / "queue.pid"
            try:
                if qp.is_file() and qp.read_text().split()[0] == str(os.getpid()):
                    qp.unlink()
            except (OSError, IndexError):
                pass

    def _loop(self) -> str:
        ctx = self.ctx
        while True:
            if self.stop_requested():
                ctx.state["status"] = "stopped"
                ctx.save()
                ctx.say("STOPPED", "STOP_PIPELINE present")
                return "stopped"
            sid = self.next_stage()
            if sid is None:
                opened = sorted(s for s, h in ctx.state["halts"].items() if h.get("status") == "open")
                if opened:
                    ctx.state["status"] = "parked"
                    ctx.save()
                    ctx.notify("PARKED", f"open_halts={','.join(opened)} -- pipeline.py continue decides")
                    return "parked"
                ctx.state["status"] = "done"
                ctx.save()
                ctx.notify("END", "every stage terminal")
                return "done"
            try:
                self.execute(sid)
            except StopRequested as exc:
                ctx.state["status"] = "stopped"
                ctx.save()
                ctx.say("STOPPED", f"{exc} (stage {sid} resumes on the next run)")
                return "stopped"
            except Halted:
                pass                                # the HALT record is written; the loop goes on
            except RowFail as exc:
                halt(ctx, sid, "INFRA", str(exc), resumable=("retry", "abort"))
            except Exception as exc:  # noqa: BLE001  (a crash halts loudly with its traceback in the log)
                import traceback  # noqa: PLC0415

                with open(ctx.qlog, "a") as fh:
                    traceback.print_exc(file=fh)
                halt(ctx, sid, "CRASH", f"{type(exc).__name__}: {exc}"[:400], resumable=("retry", "abort"))

    def execute(self, sid: str) -> None:
        kind = self.spec.stage(sid)["kind"]
        fn = {"driver": self.ex_driver, "tstar": self.ex_tstar, "distill": self.ex_distill,
              "select": self.ex_select, "dagger_sn": self.ex_dagger_sn, "dagger_sh": self.ex_dagger_sh,
              "bundle": self.ex_bundle}[kind]
        self.ctx.say("STAGE", f"stage={sid} kind={kind} status={self.status(sid)}")
        fn(sid)

    # -- verdict application
    def apply(self, sid: str, res: dict) -> None:
        ctx = self.ctx
        if self.inject(sid, "INVALID"):
            res = {**res, "verdict": GR.INVALID, "gate0": ["dry-pass injected INVALID"]}
        elif self.inject(sid, "FAIL"):
            res = {**res, "verdict": GR.HALT, "decisive": [{"id": f"{sid}.injected", "pass": False, "value": None,
                                                            "rate": None}]}
        v = res["verdict"]
        gate_rel = L.rel(res["path"]) if res.get("path") else None
        ctx.srec(sid)["gate"] = {"verdict": v, "path": gate_rel}
        ctx.say("GATE", f"stage={sid} verdict={v} halt_lines={res.get('halt_lines')}")
        if v == GR.PASS:
            ctx.set_status(sid, "pass", "gate PASS")
        elif v == GR.WATCH:
            ctx.set_status(sid, "pass", "gate PASS + WATCH")
            ctx.notify("NOTIFY", f"stage={sid} PASS with WATCH flags {[f['id'] for f in res.get('watch_flags', [])]} "
                                 f"reads {watch_reads_text(res.get('watch_flags'))}")
        elif v == GR.CEILING:
            ctx.set_status(sid, "ceiling", f"measured ceiling (max_stable {res.get('max_stable')})")
            ctx.notify("NOTIFY", f"stage={sid} measured ceiling max_stable={res.get('max_stable')} "
                                 f"limiting={(res.get('limiting_factor') or {}).get('name')} ({PR().LABEL})")
        elif v == GR.INVALID:
            r = ctx.srec(sid)
            n = r.setdefault("attempts", {}).get("invalid", 0) + 1
            r["attempts"]["invalid"] = n
            if n == 1:
                rp = ctx.rows_path(sid)
                if rp.is_file():
                    rp.rename(rp.with_suffix(f".invalid{n}.json"))
                r.pop("curve", None)
                ctx.set_status(sid, "rows", f"Gate 0 INVALID once ({res['gate0'][:2]}): the rows rerun")
                ctx.notify("NOTIFY", f"stage={sid} INVALID once -- rows rerun: {res['gate0'][:3]}")
            else:
                halt(ctx, sid, "INVALID", "; ".join(res["gate0"][:6]), gate=gate_rel, resumable=("retry", "abort"))
        else:
            fails = [d for d in res["decisive"] if not d.get("pass")]
            halt(ctx, sid, "FAIL", f"decisive line(s) failed: {[d['id'] for d in fails]}", lines=fails, gate=gate_rel)

    # -- driver stages
    def ex_driver(self, sid: str) -> None:
        ctx, spec = self.ctx, self.spec
        rec = ctx.srec(sid)
        chain, seed = spec.chain(ctx.run, sid), spec.seed(ctx.run, sid)
        if rec["status"] == "pending":
            ctx.set_status(sid, "training", "launch")
        if self.status(sid) == "training":
            head = reference_head(ctx, chain, seed)
            if head is None:
                unit = rec.get("unit")
                if unit and ctx.rt.unit_active(unit):
                    ctx.say("ATTACH", f"stage={sid} unit={unit} (the driver is still running: watched, not relaunched)")
                else:
                    if not self.guard(sid):
                        return
                    pre_launch(ctx, sid, f"driver {chain}")
                    unit, out = ctx.rt.launch_driver(ctx.dir, chain, seed)
                    if not unit:
                        halt(ctx, sid, "DRIVER", f"launch_candidates printed no unit: {out[-400:]}",
                             resumable=("retry", "abort"))
                        return
                    rec["unit"] = unit
                    ctx.save()
                    ctx.say("LAUNCH", f"stage={sid} chain={chain} seed={seed} unit={unit}")
                killed = self.watch_driver(sid, unit)
                if killed:
                    return
                head = reference_head(ctx, chain, seed)
                if head is None:
                    halt(ctx, sid, "DRIVER", f"the driver recorded no head for {chain} (see {L.rel(ctx.dir)}/"
                                             "notify.log / orchestrator.log)", resumable=("retry", "abort"))
                    return
            if not self.record_head(sid, head):
                return
            ctx.set_status(sid, "rows", "head recorded")
        if self.status(sid) == "rows":
            if (spec.stage(sid).get("gate") or {}).get("class") == "payload":
                payload_rows_and_curve(ctx, sid)
            else:
                head = ctx.head(sid)
                for key in gate_rows(spec, sid):
                    if not parse_row(key)["driver"]:
                        run_row(ctx, sid, row_job(ctx, sid, key, ckpt=head["ckpt"], name=f"T4{ctx.run}_{sid}"))
            ctx.set_status(sid, "gate", "rows done")
        if self.status(sid) == "gate":
            if not self.guard(sid):
                return
            res = gate_stage(ctx, sid)
            self.apply(sid, res)

    def record_head(self, sid: str, head: dict) -> bool:
        ctx = self.ctx
        pin = ctx.pin()
        bad = [f"{k} {str(head.get(k))[:12]} != pin {str(pin.get(k))[:12]}" for k in ("code_sha", "eval_sha")
               if head.get(k) != pin.get(k)]
        ck = Path(head["ckpt"]) if Path(head["ckpt"]).is_absolute() else L.REPO / head["ckpt"]
        if not ck.is_file():
            bad.append(f"head checkpoint {ck} missing")
        if bad:
            halt(ctx, sid, "GUARD", f"the driver's head of {sid} is not at the pin: {bad}", resumable=("retry",))
            return False
        want = self.planned_last_iteration(sid)
        if want is not None and its_of(ck) != want:
            halt(ctx, sid, "DRIVER", f"the head of {sid} is model_{its_of(ck)}.pt, the plan says model_{want}.pt "
                                     "(previous head + iterations - 1)", resumable=("retry", "abort"))
            return False
        stamp = driver_stamp(ctx, head)
        rec = {"stage": sid, "chain": head.get("chain"), "xid": head.get("xid"), "seed": head.get("seed"),
               "ckpt": str(ck), "sha256": L.sha256_file(ck), "run_dir": str(ck.parent), "iteration": its_of(ck),
               "code_sha": head.get("code_sha"), "eval_sha": head.get("eval_sha"),
               "driver_stamp": str(stamp) if stamp else None, "recorded_at": L.now_iso()}
        L.write_json_atomic(ctx.stage_read_path(sid), rec)
        return self.process_reproduced(sid, ck.parent)

    def planned_last_iteration(self, sid: str):
        """The head iteration a driver stage must end on: (the previous stage's head iteration, or 0 from scratch)
        + its iterations - 1 (the lineage: 5999 / 7498 / 8997 / 10996 / 14995 / 19994)."""
        st = self.spec.stage(sid)
        it = self.spec.iters(self.ctx.run, sid)
        if not isinstance(it, int):
            return None
        if st["resume"] == "scratch":
            return it - 1
        prev = self.ctx.head(st["resume"])
        return None if prev is None else int(prev["iteration"]) + it - 1

    def process_reproduced(self, sid: str, run_dir: Path) -> bool:
        """R18 only (``replay_of``): P-config of this stage, and at s1 P-early (steps 0-19) + the onset report.
        A failure is STOP-R (PLAN step 9: the "process reproduced" verdict; the gates are read separately)."""
        ctx = self.ctx
        ref = (self.spec.run(ctx.run).get("replay_of") or {}).get(sid)
        if not ref or ctx.srec(sid).get("stop_r_accepted"):
            return True
        old = L.REPO / ref
        out = {"stage": sid, "reference": ref, "pconfig": RR().pconfig(run_dir, old)}
        bad = [] if out["pconfig"]["pass"] else [f"P-config: {[d['key'] for f in out['pconfig']['files'].values() for d in f['not_allowed']][:6]}"]
        if sid == "s1":
            pe = RR().pearly(run_dir, old, range(20))
            out["pearly"] = {k: pe[k] for k in ("pass", "design_bitwise", "differ", "shared", "bitwise")}
            out["onset"] = RR().onset(RR().scalars(old), RR().scalars(run_dir))
            if not pe["pass"]:
                bad.append(f"P-early: {[d['tag'] for d in pe['differ'] if not d['tolerated']]}")
        L.write_json_atomic(ctx.dir / "replay" / f"{sid}.json", out)
        ctx.srec(sid)["process_reproduced"] = not bad
        ctx.save()
        if bad:
            halt(ctx, sid, "STOP-R", "; ".join(bad), lines=[{"id": f"{sid}.process_reproduced", "value": bad,
                                                              "unvalidated": False, "rate": None}])
            return False
        return True

    def source_series(self, sid: str):
        st = self.spec.stage(sid)
        src = st.get("resume")
        if not src or src == "scratch":
            return None
        h = self.ctx.head(src)
        return LW().read_series(Path(h["run_dir"])) if h else None

    def watch_driver(self, sid: str, unit: str) -> bool:
        """Poll the unit; read the live rules every poll. True when the stage was KILLED (halted / ceiling)."""
        ctx, rt = self.ctx, self.ctx.rt
        rec = ctx.srec(sid)
        cfg = self.spec.d.get("live") or {}
        src = self.source_series(sid)
        plive_done = rec.get("p_live_done", False)
        while rt.unit_active(unit):
            rt.sleep(cfg.get("poll_s", rt.poll_s))
            bad = ddc_watch_problem(ctx)
            if bad:                                  # the Kit launched after a SIGKILL shows the cache symptom
                rt.stop_unit(unit)
                while rt.unit_active(unit):
                    rt.sleep(10)
                halt(ctx, sid, "GUARD", bad, resumable=("retry",))
                return True
            rd = driver_run_dir(ctx, sid)
            if rd is None:
                continue
            ser = LW().read_series(rd)
            r = LW().check("driver", ser, stage=sid, source=src, cfg=cfg,
                           accepted=tuple(rec.get("accepted_live") or ()))
            rec["live"] = {"last_rc": r["rc"], "last": r.get("last"), "run_dir": str(rd)}
            kill = r["rc"] == LW().RC_KILL
            why = r.get("fails")
            rule = r.get("rule")
            if not kill and self.inject(sid, "LIVE"):
                kill, rule, why = True, "injected", ["dry-pass injected live kill"]
            if not kill and sid == "s1" and self.spec.run(ctx.run).get("p_live") and not plive_done:
                at = int((cfg.get("p_live") or {}).get("at", 200))
                first, last = r.get("first"), r.get("last")
                if first is not None and last is not None and last >= first + at:
                    ref = L.REPO / cfg["p_live"]["reference_run"]
                    pl = RR().pearly(rd, ref, steps=range(first, first + at + 1))
                    rec["p_live"] = {k: pl[k] for k in ("pass", "design_bitwise", "differ")}
                    plive_done = rec["p_live_done"] = True
                    if not pl["pass"] and "p_live" not in (rec.get("accepted_live") or []):
                        kill, rule, why = True, "p_live", [f"P-live at +{at}: {[d['tag'] for d in pl['differ']]}"]
            ctx.save()
            if kill:
                rt.stop_unit(unit)
                while rt.unit_active(unit):
                    rt.sleep(10)
                rec["live_killed"] = {"rule": rule, "fails": why, "at": r.get("last")}
                ctx.save()
                if (self.spec.stage(sid).get("gate") or {}).get("ceiling_on_fail"):
                    ctx.set_status(sid, "ceiling", f"live kill ({rule}): a measured ceiling (never a HALT)")
                    ctx.notify("NOTIFY", f"stage={sid} live kill {rule} -> measured ceiling ({PR().LABEL})")
                else:
                    klass = "STOP-R" if rule == "p_live" else "LIVE"
                    halt(ctx, sid, klass, f"live rule {rule}: {why}", lines=[{"id": f"{sid}.live.{rule}",
                         "statistic": f"live rule {rule}", "value": why, "unvalidated": True, "rate": None}])
                return True
        return False

    # -- T*
    TSTAR_RULE = ("PLAN Payload -> T*: the trained payload head with the highest max_stable that passes its paired "
                  "0-kg check -- or whose 0-kg HALT the user accepted as chance -- tried in descending max_stable "
                  "(ties to the earlier stage); HALT only if none passes")

    def tstar_considered(self, sid: str) -> tuple:
        """Every payload head of the T* stage's candidate list, judged (CHARTER 2026-10-21 decision D: the T* record
        and its notification list EVERY head considered, not only the eligible ones -- p3b's 0-kg FAIL was dropped
        silently in R18). Returns (considered: one entry per TRAINED head with max_stable, its 0-kg verdict, the
        failing statistics, the max margin use, accepted or not, eligible, and why; not_trained: the candidates with
        no head). The selection rule itself is unchanged."""
        ctx = self.ctx
        considered, not_trained = [], []
        for c in self.spec.stage(sid)["candidates"]:
            r = ctx.state["stages"].get(c) or {}
            st = r.get("status", "pending")
            if st not in ("pass", "accepted", "ceiling") or not ctx.head(c):
                not_trained.append({"stage": c, "status": st,
                                    "why": f"not considered: no trained head (status {st}"
                                           + (f": {r['history'][-1].get('why')}" if r.get("history") else "") + ")"})
                continue
            zk = r.get("zero_kg") or {}
            verdict = zk.get("verdict")
            accepted = st == "accepted"
            mmu = zk.get("max_margin_used")
            considered.append({"stage": c, "status": st, "max_stable": r.get("max_stable"), "zero_kg": verdict,
                               "zero_kg_fails": list(zk.get("fails") or []), "max_margin_used": mmu,
                               "zero_kg_watch": list(zk.get("watch_exceeds") or []),
                               "accepted": accepted, "eligible": verdict == "PASS" or accepted,
                               "gate": (r.get("gate") or {}).get("path"),
                               "halt": (ctx.state.get("halts") or {}).get(c)})
        order = self.spec.stage(sid)["candidates"]
        elig = sorted((x for x in considered if x["eligible"]),
                      key=lambda x: (-(x["max_stable"] if x["max_stable"] is not None else -1),
                                     order.index(x["stage"])))
        t = elig[0] if elig else None
        for x in considered:
            use = f"max margin use {x['max_margin_used']:.2f}" if isinstance(x["max_margin_used"], (int, float)) \
                else "no margin use recorded"
            zk = f"paired 0-kg check {x['zero_kg']} ({use}" + (f"; fails: {'; '.join(x['zero_kg_fails'])}"
                                                               if x["zero_kg_fails"] else "") + ")"
            x["chosen"] = t is not None and x["stage"] == t["stage"]
            if x["chosen"]:
                x["why"] = (f"CHOSEN: the highest max_stable ({x['max_stable']}) among the eligible heads; {zk}"
                            + ("; its 0-kg HALT was accepted as chance by the user (disclosed)"
                               if x["accepted"] and x["zero_kg"] != "PASS" else ""))
            elif x["eligible"]:
                tie = x["max_stable"] == t["max_stable"]
                x["why"] = (f"eligible, not chosen: max_stable {x['max_stable']} "
                            + (f"ties T* ({t['stage']}) and the earlier stage wins" if tie
                               else f"< T*'s {t['max_stable']} ({t['stage']})") + f"; {zk}")
            else:
                x["why"] = (f"EXCLUDED: {zk}, and no HALT of it was accepted as chance (status {x['status']}"
                            + ("; a payload ceiling stage never HALTs, so this 0-kg FAIL reached no HALT"
                               if x["status"] == "ceiling" else "") + ")")
        return considered, not_trained, t

    @staticmethod
    def tstar_summary(considered: list, not_trained: list) -> str:
        """One clause per head for the T* notification."""
        out = []
        for x in considered:
            use = f"{x['max_margin_used']:.2f}" if isinstance(x["max_margin_used"], (int, float)) else "-"
            tag = "CHOSEN" if x.get("chosen") else ("eligible" if x["eligible"] else "EXCLUDED")
            fails = f" fails [{'; '.join(x['zero_kg_fails'])}]" if x["zero_kg_fails"] else ""
            watch = (f" WATCH [{'; '.join(x['zero_kg_watch'])}]" if x.get("zero_kg_watch") else "")
            out.append(f"{x['stage']}: max_stable={x['max_stable']} 0-kg={x['zero_kg']} margin_use={use}"
                       f"{' accepted-as-chance' if x['accepted'] else ''}{fails}{watch} -> {tag}")
        out += [f"{x['stage']}: not trained ({x['status']})" for x in not_trained]
        return " | ".join(out)

    def ex_tstar(self, sid: str) -> None:
        ctx = self.ctx
        considered, not_trained, t = self.tstar_considered(sid)
        summary = self.tstar_summary(considered, not_trained)
        if t is None:
            halt(ctx, sid, "FAIL", f"no trained payload head passes its paired 0-kg check (T*): {summary}",
                 lines=[{"id": "tstar.none", "statistic": "T* candidates", "value": None}]
                 + [{"id": f"tstar.{x['stage']}.zero_kg", "statistic": f"{x['stage']} paired 0-kg check",
                     "value": x["max_margin_used"], "line": 1.0, "direction": "<=", "rate": None,
                     "unvalidated": True, "why": x["why"]} for x in considered])
            return
        cands = [{"stage": x["stage"], "max_stable": x["max_stable"], "zero_kg": x["zero_kg"],
                  "accepted": x["accepted"], "max_margin_used": x["max_margin_used"]} for x in considered
                 if x["eligible"]]
        cands.sort(key=lambda x: (-(x["max_stable"] if x["max_stable"] is not None else -1),
                                  self.spec.stage(sid)["candidates"].index(x["stage"])))
        h = ctx.head(t["stage"])
        rec = {"stage": t["stage"], "ckpt": h["ckpt"], "sha256": h["sha256"], "run_dir": h["run_dir"],
               "iteration": h["iteration"], "max_stable": t["max_stable"], "candidates": cands,
               "considered": considered, "not_trained": not_trained, "rule": self.TSTAR_RULE,
               "driver_stamp": h.get("driver_stamp"), "label": PR().LABEL}
        L.write_json_atomic(ctx.stage_read_path(sid), rec)
        ctx.state["tstar"] = rec
        ctx.set_status(sid, "pass", f"T* = {t['stage']} (max_stable {t['max_stable']})")
        ctx.notify("NOTIFY", f"T*={t['stage']} max_stable={t['max_stable']} 0-kg margin used "
                             f"{t.get('max_margin_used')} ({PR().LABEL}); heads considered: {summary}")

    def tstar(self) -> dict:
        return self.ctx.head("tstar")

    # -- student
    def ex_distill(self, sid: str) -> None:
        ctx, spec = self.ctx, self.spec
        st = spec.stage(sid)
        ts = self.tstar()
        rec = ctx.srec(sid)
        if self.status(sid) == "pending":
            ctx.set_status(sid, "training", "distill")
        n_att = int(rec.get("distill_attempt", 1))
        name = f"T4{ctx.run}_{sid}" + (f"_a{n_att}" if n_att > 1 else "")
        iters = int(spec.iters(ctx.run, sid))
        if not rec.get("trained"):
            u = read_unit(name)
            if u and ctx.rt.unit_active(f"{u}.scope"):
                proc = UnitProc(ctx.rt, u)
                ctx.say("ATTACH", f"stage={sid} distill scope {u} still running: watched, not relaunched")
            else:
                if not self.guard(sid):
                    return
                pre_launch(ctx, sid, f"distill {name}")
                env_file = ctx.env_file(ctx.plan["stages"][ts["stage"]]["distill_env"])
                cmd = ["bash", str(HERE / "distill.sh"), "--name", name, "--checkpoint", ts["ckpt"],
                       "--env-file", str(env_file), "--seed", str(spec.seed(ctx.run, sid)),
                       "--iterations", str(iters), "--num-envs", str(st["num_envs"]),
                       "--task", st["task"], "--experiment", st["experiment"]]
                proc = ctx.rt.spawn(cmd, log=ctx.dir / "tp_logs" / f"{sid}.launcher.log")
                ctx.say("LAUNCH", f"stage={sid} distill name={name} from={ts['stage']}")
            killed = self.watch_student(sid, proc, ts)
            if killed:
                return
            rc = proc.wait()
            rd = find_student_run(st["experiment"], name)
            last = max((its_of(p) for p in rd.glob("model_*.pt")), default=None) if rd else None
            want = int(ts["iteration"]) + iters - 1
            if rc != 0 or rd is None or last is None or last < want:
                halt(ctx, sid, "DRIVER", f"distill.sh rc={rc}, run dir {rd}, last checkpoint {last} (want >= {want})",
                     resumable=("retry", "abort"))
                return
            rec["run_dir"] = str(rd)
            rec["trained"] = {"rc": rc, "at": L.now_iso()}
            ctx.save()
        rd = Path(rec["run_dir"])
        cands = SS().candidate_checkpoints(rd, ts["iteration"],
                                           spec.stage("sel")["offsets"] if "sel" in spec._by_id else SS().OFFSETS)
        L.write_json_atomic(ctx.stage_read_path(sid), {"stage": sid, "run_dir": str(rd), "tstar": ts["stage"],
                                                       "candidates": cands, "recorded_at": L.now_iso()})
        ctx.set_status(sid, "pass", f"student trained ({len(cands)} candidates)")

    def watch_student(self, sid: str, proc, ts: dict) -> bool:
        """Poll the distill until it exits: the DerivedDataCache watch, the progress watchdog, the student live rules.
        A stop goes through :meth:`stop_nondriver` (the trainer pid, never the scope). True when the stage HALTed."""
        ctx, rt = self.ctx, self.ctx.rt
        rec = ctx.srec(sid)
        cfg = self.spec.d.get("live") or {}
        tser = LW().read_series(Path(ts["run_dir"]), (LW().T_EPLEN, LW().T_CRAB))
        tail = {"ep_len": LW().tail_mean(tser.get(LW().T_EPLEN)), "crab_failure": LW().tail_mean(tser.get(LW().T_CRAB))}
        n_att = int(rec.get("distill_attempt", 1))
        name = f"T4{ctx.run}_{sid}" + (f"_a{n_att}" if n_att > 1 else "")
        started = rt.time()
        tok = ["--run_name", name]

        def match(args: str) -> bool:
            a = args.split()
            return any(a[i:i + 2] == tok for i in range(len(a) - 1))

        while proc.poll() is None:
            rt.sleep(cfg.get("poll_s", rt.poll_s))
            rd = find_student_run(self.spec.stage(sid)["experiment"], name)
            bad = ddc_watch_problem(ctx)
            if bad:
                self.stop_nondriver(sid, unit=read_unit(name), match=match, proc=proc, reason="DerivedDataCache watch")
                halt(ctx, sid, "GUARD", bad, resumable=("retry",))
                return True
            stall = self.stall_s(started, student_progress_files(rd))
            if stall is not None:
                why = (f"no training progress for {stall / 60:.0f} min (watchdog {self.watchdog_min():g} min; "
                       f"tfevents / checkpoints of {name}): stopped, HALT INFRA")
                self.stop_nondriver(sid, unit=read_unit(name), match=match, proc=proc, reason="progress watchdog")
                halt(ctx, sid, "INFRA", why, resumable=("retry", "abort"))
                return True
            if rd is None:
                continue
            ser = LW().read_series(rd)
            r = LW().check("student", ser, tstar_tail=tail, cfg=cfg, accepted=tuple(rec.get("accepted_live") or ()))
            rec["live"] = {"last_rc": r["rc"], "last": r.get("last"), "run_dir": str(rd)}
            if r.get("watch"):              # live.student.loss_rule: watch -- reported, never kills
                rec["live"]["watch"] = r["watch"]
                if not rec.get("live_watch"):
                    rec["live_watch"] = {"at": r.get("last"), "reads": r["watch"]}
                    ctx.notify("NOTIFY", f"stage={sid} student live WATCH (reported, not a kill): {r['watch'][:2]}")
            ctx.save()
            if r["rc"] != LW().RC_KILL and self.inject(sid, "LIVE"):
                r = {**r, "rc": LW().RC_KILL, "rule": "injected", "fails": ["dry-pass injected live kill"]}
            if r["rc"] == LW().RC_KILL:
                self.stop_nondriver(sid, unit=read_unit(name), match=match, proc=proc,
                                    reason=f"student live rule {r['rule']}")
                rec["live_killed"] = {"rule": r["rule"], "fails": r["fails"], "at": r.get("last")}
                ctx.save()
                halt(ctx, sid, "LIVE", f"student live rule {r['rule']}: {r['fails']}",
                     lines=[{"id": f"{sid}.live.{r['rule']}", "statistic": "student live rule", "value": r["fails"],
                             "unvalidated": True, "rate": None}])
                return True
        return False

    def ex_select(self, sid: str) -> None:
        """Prescreen every candidate, rank, read T* on Student-Play once, then full blocks in rank order."""
        ctx, spec = self.ctx, self.spec
        st = spec.stage(sid)
        ts = self.tstar()
        stu = ctx.head("st")
        th = ctx.thresholds()
        rec = ctx.srec(sid)
        if self.status(sid) == "pending":
            ctx.set_status(sid, "rows", "prescreen")
        if not self.guard(sid):
            return
        refused = student_refused_keys()
        tst = ts["stage"]
        trained = ctx.plan["stages"][tst]["trained"]
        tt = [c for c in trained if c in GR.TRANSLATE]
        pres = rec.setdefault("prescreen", {})
        th_doc = json.loads(spec.thresholds_path().read_text())
        for c in stu["candidates"]:
            if not c.get("ckpt"):
                continue
            cid = f"c{c['iter']}"
            if cid in pres:
                continue
            stamps = {}
            for key in st["prescreen_rows"]:
                job = row_job(ctx, sid, key, ckpt=c["ckpt"], name=f"T4{ctx.run}_sel_{cid}", env_stage=tst,
                              extra_args=tuple(st["student_args"]), strip=refused)
                stamps[key] = run_row(ctx, sid, {**job, "key": f"{cid}:{job['key']}"})
            lv = st["prescreen_level"]
            lstamps = {}
            for key in lv["rows"]:
                job = row_job(ctx, sid, key, ckpt=c["ckpt"], name=f"T4{ctx.run}_sel_{cid}_L{lv['kg']:03d}",
                              env_stage=tst, level=int(lv["kg"]), extra_args=tuple(st["student_args"]), strip=refused)
                lstamps[key] = run_row(ctx, sid, {**job, "key": f"{cid}:{job['key']}"})
            g0 = self.student_gate0(sid, c["ckpt"], tst, {**stamps, **{f"L:{k}": v for k, v in lstamps.items()}},
                                    level=int(lv["kg"]))
            if g0:
                halt(ctx, sid, "INVALID", f"prescreen {cid}: " + "; ".join(g0[:6]), resumable=("retry", "abort"))
                return
            fl = GR.floor_reads(sid, "student", {1: stamps["suite@1"]}, trained, th, env_seeds=(1,))
            lr = PR().level_reads(lstamps, th_doc, trained_translate=tt)
            fh = GR.completion([stamps[k] for k in st["prescreen_rows"] if k.startswith("field_hard")])
            cs = GR.cat_summary(stamps["suite@1"])
            devs = [abs((cs.get(x) or {}).get("ratio") - 1.0) for x in tt if (cs.get(x) or {}).get("ratio") is not None]
            pres[cid] = {"iter": c["iter"], "ckpt": c["ckpt"],
                         "misses": sum(1 for d in fl if not d.get("pass")) + len([x for x in lr["lines"]
                                                                                    if x["id"].endswith(".es1")
                                                                                    and not x["pass"]]),
                         "field_hard": fh["rate"], "mean_abs_dev": (sum(devs) / len(devs)) if devs else None}
            ctx.save()
        ranked = [p for p in SS().rank(list(pres.values()))]
        rec["ranked"] = [f"c{p['iter']}" for p in ranked]
        ctx.save()
        tst_stamps = self.tstar_on_student_play(sid, ts, refused)
        g0 = self.student_gate0(sid, ts["ckpt"], tst, tst_stamps, level=None)
        if g0:
            halt(ctx, sid, "INVALID", "T* on Student-Play: " + "; ".join(g0[:6]), resumable=("retry", "abort"))
            return
        while True:
            act = SS().next_action({"ranked": rec["ranked"], "blocks": rec.get("blocks") or {}},
                                   int(st.get("max_failed_blocks", 2)))
            if act["do"] == "selected":
                c = pres[act["cand"]]
                sel = {"cand": act["cand"], "ckpt": c["ckpt"], "sha256": L.sha256_file(Path(c["ckpt"])),
                       "iter": c["iter"], "tstar": tst}
                L.write_json_atomic(ctx.stage_read_path(sid), sel)
                ctx.state["selected_student"] = sel
                ctx.set_status(sid, "pass", f"selected {act['cand']}")
                g = L.read_json(ctx.dir / "gates" / f"{sid}.json") or {}
                wf = ((g.get("blocks") or {}).get(act["cand"]) or {}).get("watch_flags") or []
                ctx.notify("NOTIFY", f"student selected {act['cand']} (iter {c['iter']}); one headless clip goes "
                                     "to the user for a veto (never a gate); WATCH flags "
                                     + (f"{[f['id'] for f in wf]} reads {watch_reads_text(wf)}" if wf else "none"))
                return
            if act["do"] == "halt":
                g = L.read_json(ctx.dir / "gates" / f"{sid}.json") or {}
                halt(ctx, sid, "STUDENT", act["why"], lines=g.get("failed_lines") or [],
                     gate=L.rel(ctx.dir / "gates" / f"{sid}.json"))
                return
            cid = act["cand"]
            res = self.full_block(sid, cid, pres[cid], tst_stamps, ts, refused, th)
            rec.setdefault("blocks", {})[cid] = "PASS" if res["pass"] else "FAIL"
            ctx.save()

    def student_gate0(self, sid: str, ckpt: str, tstage: str, stamps: dict, *, level) -> list:
        """Gate 0 on Student-Play rows (the student's or T*'s): the pin, the composed T* env minus the refused runner
        knobs, the checkpoint's sha, the task, trained_categories; ``L:<row>`` keys are the level rows at ``level``."""
        refused = student_refused_keys()
        head = {"sha256": ckpt_sha(ckpt)}
        play = self.spec.stage(sid)["play_task"]
        expects = {}
        for key in stamps:
            lv = key.startswith("L:")
            r = parse_row(key[2:] if lv else key)
            expects[key] = expect_for(self.ctx, sid, {**r, "key": key, "strip": refused}, source="row", head=head,
                                      env_stage=tstage, level=(level if lv else None), task=play)
        return gate0(self.ctx, sid, stamps, expects)

    def tstar_on_student_play(self, sid: str, ts: dict, refused) -> dict:
        ctx, st = self.ctx, self.spec.stage(sid)
        out = {}
        for key in st["full_block_rows"]:
            job = row_job(ctx, sid, key, ckpt=ts["ckpt"], name=f"T4{ctx.run}_sel_tstar", env_stage=ts["stage"],
                          extra_args=tuple(st["teacher_args"]), strip=refused)
            out[key] = run_row(ctx, sid, {**job, "key": f"tstar:{job['key']}"})
        return out

    def full_block(self, sid: str, cid: str, cand: dict, tst_stamps: dict, ts: dict, refused, th) -> dict:
        ctx, st = self.ctx, self.spec.stage(sid)
        TH = L.thresholds_mod()
        tstage = ts["stage"]
        trained = ctx.plan["stages"][tstage]["trained"]
        stamps = {}
        for key in st["full_block_rows"]:
            job = row_job(ctx, sid, key, ckpt=cand["ckpt"], name=f"T4{ctx.run}_sel_{cid}", env_stage=tstage,
                          extra_args=tuple(st["student_args"]), strip=refused)
            stamps[key] = run_row(ctx, sid, {**job, "key": f"{cid}:{job['key']}"})
        lvl = min(90, int(ts.get("max_stable") or 0))
        lstamps = {}
        for key in st["full_block_level_rows"]:
            job = row_job(ctx, sid, key, ckpt=cand["ckpt"], name=f"T4{ctx.run}_sel_{cid}_L{lvl:03d}", env_stage=tstage,
                          level=lvl, extra_args=tuple(st["student_args"]), strip=refused)
            lstamps[key] = run_row(ctx, sid, {**job, "key": f"{cid}:{job['key']}"})
        g0 = self.student_gate0(sid, cand["ckpt"], tstage, {**stamps, **{f"L:{k}": v for k, v in lstamps.items()}},
                                level=lvl)
        dec = GR.walker_decisive(sid, stamps, trained, th, th_stage="student") if not g0 else []
        pairs = []
        if not g0:
            stu_e, t_e = {}, {}
            for row in SS().FIELD_ROWS:
                a, b = {}, {}
                ks = sorted(k for k in stamps if k.startswith(row + "@"))
                for i, k in enumerate(ks):
                    a.update({(i, e): v for e, v in TH.field_episodes(stamps[k]).items()})
                    b.update({(i, e): v for e, v in TH.field_episodes(tst_stamps[k]).items()})
                stu_e[row], t_e[row] = a, b
            pairs = SS().completion_pairs(stu_e, t_e, st["completion_margin"])
            for p in pairs:
                dec.append({"id": f"student.capability.completion.{p['row']}", "statistic": f"{p['row']} completion "
                            "vs T* on Student-Play", "value": p.get("student"), "line": p.get("line"),
                            "direction": ">=", "pass": p["pass"], "rate": None, "unvalidated": True,
                            "rate_note": "paired; a lower bound (no distillation-seed variance)", **p})
        th_doc = json.loads(self.spec.thresholds_path().read_text())
        watch, flags = {}, []
        if not g0:
            watch = {"stable_at": lvl, "level": PR().level_reads(lstamps, th_doc, trained_translate=[
                c for c in trained if c in GR.TRANSLATE])}
            # PLAN Gates -> Student "WATCH: tracking lines + 0.05" (CHARTER 2026-10-21 decision E): the student vs T*
            # on the paired tracking statistics of the suite rows (both read on Student-Play) -- WATCH only
            sa = {int(k.split("@")[1]): TH.suite_episodes(Path(v)) for k, v in stamps.items() if k.startswith("suite@")}
            sb = {int(k.split("@")[1]): TH.suite_episodes(Path(v)) for k, v in tst_stamps.items()
                  if k.startswith("suite@")}
            watch["tracking"] = SS().tracking_watch(sa, sb, trained)
            flags = watch["tracking"]["flags"]
        ok = not g0 and all(d.get("pass") for d in dec)
        blk = {"cand": cid, "pass": ok, "gate0": g0, "decisive": dec, "watch": watch, "watch_flags": flags,
               "verdict": GR.verdict_of(dec, g0, watch_flags=flags)}
        g = L.read_json(ctx.dir / "gates" / f"{sid}.json") or {"blocks": {}}
        g.setdefault("blocks", {})[cid] = blk
        g["verdict"] = (GR.WATCH if flags else GR.PASS) if ok else g.get("verdict", GR.HALT)
        g["failed_lines"] = [d for d in dec if not d.get("pass")] or g0
        g["watch_flags"] = flags if ok else g.get("watch_flags", [])
        write_gate(ctx, sid, g)
        ctx.say("BLOCK", f"stage={sid} cand={cid} pass={ok} failed={[d['id'] for d in dec if not d.get('pass')]} "
                         f"watch_flags={[f['id'] for f in flags]}")
        return blk

    # -- stand
    def walker_spec_env(self, sid: str) -> dict:
        """SN_WALKER_SPEC (+ CK / ST, P2_REANCHOR) for a stand stage beside T*. Beside the bundled S1 (the X stages)
        the walker stays unset (= today) and only ``P2_REANCHOR=1`` is set: REFH_S1_s18's sidecar predates M1, so
        run_arm.sh's code_guard would refuse the X stand rows at the pin code_sha (rc 7) -- the pin guard holds the
        code state instead (STEPC_RULES "TP changes" 1)."""
        st = self.spec.stage(sid)
        if st.get("walker") != "tstar":
            return {"P2_REANCHOR": "1"}
        ts = self.tstar()
        h = self.ctx.head(ts["stage"])
        spec = {"S1_CKPT": h["ckpt"], "S1_SHA256": h["sha256"],
                "S1_AGENT_YAML": str(Path(h["ckpt"]).parent / "params" / "agent.yaml"),
                "REFS_ENV_STAMP": h["driver_stamp"], "CANDIDATES": str(self.ctx.dir / "candidates.yaml"),
                "OVERLAY_CHAIN": self.spec.chain(self.ctx.run, "p2a"),
                "S1_OVERLAY": self.spec.overlay("p2a"), "REFS_NAME": self.refs_name()}
        return {"SN_WALKER_SPEC": json.dumps(spec, sort_keys=True), "SN_WALKER_CK": h["ckpt"],
                "SN_WALKER_ST": h["driver_stamp"], "P2_REANCHOR": "1"}

    def refs_name(self) -> str:
        return f"REFS_T4{self.ctx.run}"

    def stand_paths(self) -> dict:
        s = self.spec.d["stand"]
        return {k: (L.OMNI / v if not v.startswith("parkour/") else L.REPO / v) for k, v in s.items()}

    def ex_dagger_sn(self, sid: str) -> None:
        ctx, spec = self.ctx, self.spec
        st = stand_stage_cfg(spec, sid)
        P = self.stand_paths()
        rec = ctx.srec(sid)
        env = self.walker_spec_env(sid)
        ds = spec.seed(ctx.run, sid)
        name = self.dagger_name(f"SN_T4{ctx.run}_{sid}_s{ds}", sid)
        if self.status(sid) == "pending":
            ctx.set_status(sid, "training", "stand SN")
        steps = rec.setdefault("steps", {})
        if not self.guard(sid):
            return
        if st.get("walker") == "tstar" and not steps.get("refs"):
            h = ctx.head(self.tstar()["stage"])
            for key in st["refs_rows"]:
                r = parse_row(key)
                pre_launch(ctx, sid, f"REFS(T*) row {key}")
                rc, last = ctx.rt.run(["bash", str(P["p2_run_arm"]), "--row-eval", self.refs_name(), h["ckpt"],
                                       h["driver_stamp"], str(spec.manifest(r["family"])),
                                       spec.row(r["family"])["scenario"], str(r["es"])],
                                      env={"P2_REANCHOR": "1"}, log=ctx.dir / "tp_logs" / f"{sid}.refs.log")
                if rc != 0:
                    halt(ctx, sid, "INFRA", f"REFS(T*) row {key} rc={rc}: {last}", resumable=("retry", "abort"))
                    return
            steps["refs"] = True
            ctx.save()
        ck = P["sn_runs"] / name / self.judged_name(sid)
        if not ck.is_file():
            rc = self.train_dagger(sid, P["sn_scripts"] / "sn_dagger.sh", P["sn_scripts"] / "sn_dagger.py",
                                   P["sn_runs"], name, ["--mode", "dagger", "--arm", st["arm"], "--seed", str(ds),
                                                        "--iterations", str(spec.iters(ctx.run, sid))], env)
            if rc is None:
                return
            if rc != 0 or not ck.is_file():
                halt(ctx, sid, "DRIVER", f"sn_dagger.sh rc={rc}; {ck} {'present' if ck.is_file() else 'missing'}",
                     resumable=("retry", "abort"))
                return
        ctx.set_status(sid, "rows", "SN trained")
        if not steps.get("rows"):
            rows = [f"{spec.row(parse_row(k)['family'])['scenario']}@{parse_row(k)['es']}" for k in st["stand_rows"]]
            pre_launch(ctx, sid, "sn_eval_rows.sh")
            rc, last = ctx.rt.run(["bash", str(P["sn_scripts"] / "sn_eval_rows.sh"), name, str(ck), "1", *rows],
                                  env=env, log=ctx.dir / "tp_logs" / f"{sid}.rows.log")
            if rc != 0:
                halt(ctx, sid, "INFRA", f"sn_eval_rows.sh rc={rc}", resumable=("retry", "abort"))
                return
            steps["rows"] = True
            ctx.save()
        out = ctx.dir / "stand"
        out.mkdir(parents=True, exist_ok=True)
        refs = self.refs_name() if st.get("walker") == "tstar" else "REFS_S1_s18"
        ctx.rt.run([L.PY, "-I", str(P["sn_scripts"] / "sn_bridge_read.py"), "--head", name, "--refs", refs, "--live",
                    "--json", str(out / f"{sid}_bridge.json")], env=env, log=ctx.dir / "tp_logs" / f"{sid}.read.log")
        ctx.rt.run([L.PY, "-I", str(P["sn_scripts"] / "sn_calib.py"), "--head", name, "--json",
                    str(out / f"{sid}_calib.json"), "--md", str(out / f"{sid}_calib.md")], env=env,
                   log=ctx.dir / "tp_logs" / f"{sid}.read.log")
        rec_h = {"stage": sid, "name": name, "ckpt": str(ck), "sha256": L.sha256_file(ck), "refs": refs,
                 "calib": str(out / f"{sid}_calib.json"), "bridge": str(out / f"{sid}_bridge.json"),
                 "train_log": str(P["sn_runs"] / name / "train_log.jsonl")}
        L.write_json_atomic(ctx.stage_read_path(sid), rec_h)
        ctx.set_status(sid, "gate", "SN read")
        res = stand_gate(ctx, sid, "D1", rec_h, x=bool(st.get("x_replay")))
        self.apply_stand(sid, res)

    def ex_dagger_sh(self, sid: str) -> None:
        ctx, spec = self.ctx, self.spec
        st = stand_stage_cfg(spec, sid)
        P = self.stand_paths()
        rec = ctx.srec(sid)
        env = self.walker_spec_env(sid)
        ds = spec.seed(ctx.run, sid)
        name = self.dagger_name(f"SH_T4{ctx.run}_{sid}_s{ds}", sid)
        if self.status(sid) == "pending":
            ctx.set_status(sid, "training", "stand SH")
        steps = rec.setdefault("steps", {})
        if not self.guard(sid):
            return
        ck = P["sh_runs"] / name / self.judged_name(sid)
        if not ck.is_file():
            warm = []
            if st.get("warm_start_from"):
                sn = ctx.head(st["warm_start_from"])
                warm = ["--warm-start", sn["ckpt"], "--warm-sha", sn["sha256"]]
            rc = self.train_dagger(sid, P["sh_scripts"] / "sh_dagger.sh", P["sh_scripts"] / "sh_dagger.py",
                                   P["sh_runs"], name, ["--mode", "dagger", "--arm", "mask", "--seed", str(ds),
                                                        "--iterations", str(spec.iters(ctx.run, sid)), *st["opts"],
                                                        *warm], env)
            if rc is None:
                return
            if rc != 0 or not ck.is_file():
                halt(ctx, sid, "DRIVER", f"sh_dagger.sh rc={rc}; {ck} {'present' if ck.is_file() else 'missing'}",
                     resumable=("retry", "abort"))
                return
        ctx.set_status(sid, "rows", "SH trained")
        db = st.get("deadband", "0.25")
        if not steps.get("table"):
            pre_launch(ctx, sid, "sh_table_test.sh")
            rc, last = ctx.rt.run(["bash", str(P["sh_scripts"] / "sh_table_test.sh"), "--wait-idle", "--foreground",
                                   "--controllers", "stand_net", "--videos", "stand_net", "--stand-ckpt", str(ck),
                                   "--label", name, "--deadband", db], env=env,
                                  log=ctx.dir / "tp_logs" / f"{sid}.table.log")
            latest = P["sh_table_out"] / "LATEST"
            tdir = (P["sh_table_out"] / latest.read_text().strip()) if latest.is_file() else None
            if rc != 0 or tdir is None:
                halt(ctx, sid, "INFRA", f"sh_table_test.sh rc={rc}", resumable=("retry", "abort"))
                return
            steps["table"] = str(tdir)
            ctx.save()
        for which, rows_key, cmd_kind, nm in (("hrows", "height_rows", "manifest", name),
                                              ("k5rows", "k5_rows", "zero", f"{name}z")):
            if steps.get(which):
                continue
            rows = [f"{spec.row(parse_row(k)['family'])['scenario']}@{parse_row(k)['es']}" for k in st[rows_key]]
            pre_launch(ctx, sid, f"sh_eval_rows.sh ({which})")
            rc, last = ctx.rt.run(["bash", str(P["sh_scripts"] / "sh_eval_rows.sh"), "--deadband", db, nm,
                                   "stand_net", cmd_kind, "armed", str(ck), "1", *rows], env=env,
                                  log=ctx.dir / "tp_logs" / f"{sid}.{which}.log")
            if rc != 0:
                halt(ctx, sid, "INFRA", f"sh_eval_rows.sh ({which}) rc={rc}", resumable=("retry", "abort"))
                return
            steps[which] = True
            ctx.save()
        out = ctx.dir / "stand"
        out.mkdir(parents=True, exist_ok=True)
        refs = self.refs_name() if st.get("walker") == "tstar" else "REFS_S1_s18"
        rspec = sh_read_spec(self.spec, st, name, Path(steps["table"]), refs)
        sp = out / f"{sid}_read_spec.json"
        L.write_json_atomic(sp, rspec)
        ctx.rt.run([L.PY, "-I", str(P["sh_scripts"] / "sh_read.py"), "--spec", str(sp), "--json",
                    str(out / f"{sid}_read.json"), "--md", str(out / f"{sid}_read.md")], env=env,
                   log=ctx.dir / "tp_logs" / f"{sid}.read.log")
        # K5's per-clause values (thresholds.json reads them: the relaxed G4 applies there too) -- sn_calib on the
        # head's zero-command stamps, the way the thresholds' own K5 inputs were read
        ctx.rt.run([L.PY, "-I", str(P["sn_scripts"] / "sn_calib.py"), "--head", f"{name}z", "--json",
                    str(out / f"{sid}_k5_calib.json"), "--md", str(out / f"{sid}_k5_calib.md")], env=env,
                   log=ctx.dir / "tp_logs" / f"{sid}.read.log")
        rec_h = {"stage": sid, "name": name, "ckpt": str(ck), "sha256": L.sha256_file(ck), "refs": refs,
                 "read": str(out / f"{sid}_read.json"), "k5_calib": str(out / f"{sid}_k5_calib.json"),
                 "table": steps["table"], "train_log": str(P["sh_runs"] / name / "train_log.jsonl")}
        L.write_json_atomic(ctx.stage_read_path(sid), rec_h)
        ctx.set_status(sid, "gate", "SH read")
        res = stand_gate(ctx, sid, "D2", rec_h, x=bool(st.get("x_replay")))
        self.apply_stand(sid, res)

    def judged_name(self, sid: str) -> str:
        """A stand stage's judged (final) checkpoint: ``model_<the iterations it is launched with>.pt`` -- a run's
        ``stage_iters_override`` included (the dry pass trains 20 its: model_20.pt). The spec's ``judged`` names the
        full-length file and must agree with the stage's own ``iters`` (:func:`graph_problems`)."""
        return f"model_{int(self.spec.iters(self.ctx.run, sid))}.pt"

    def watchdog_min(self) -> float:
        return float((self.spec.d.get("live") or {}).get("progress_watchdog_min", WATCHDOG_MIN))

    def stall_s(self, started: float, files) -> float | None:
        """Seconds without training progress (the newest of ``files``, else the launch) when past the watchdog."""
        m = progress_mtime(files)
        s = self.ctx.rt.time() - max(float(started), m or 0.0)
        return s if s > self.watchdog_min() * 60.0 else None

    def stop_nondriver(self, sid: str, *, unit: str | None, match, proc, reason: str) -> dict:
        """Stop a distill / DAgger stage (:func:`stop_trainer`), record it in the stage record, and on a SIGKILL notify
        and open the DerivedDataCache watch; then wait (bounded) for the launcher, whose trainer is gone. The retry of
        a stopped stage trains afresh (``fresh_run_on_retry``: a DAgger run dir never resumes)."""
        ctx, rt = self.ctx, self.ctx.rt
        rec = ctx.srec(sid)
        grace = float((self.spec.d.get("live") or {}).get("term_grace_s", TERM_GRACE_S))
        ctx.say("STOP", f"stage={sid} {reason}: SIGTERM the trainer pid, {grace:.0f} s, SIGTERM, {grace:.0f} s, "
                        "SIGKILL only as the last resort")
        res = stop_trainer(rt, unit=unit, match=match, proc=proc, grace=grace)
        res["reason"] = reason
        rec.setdefault("stops", []).append(res)
        rec["fresh_run_on_retry"] = True
        if res.get("sigkill"):
            rec["sigkill"] = {"at": res["at"], "pids": res["pids"], "reason": reason, "exited": res.get("exited")}
            ctx.state["ddc_watch"] = {"stage": sid, "at": L.now_iso(), "at_epoch": rt.time(),
                                      "ddc_gb_at_kill": ddc_gb()}
            ctx.save()
            ctx.notify("NOTIFY", f"stage={sid} trainer pid(s) {res['pids']} SIGKILLed after 2 x SIGTERM + "
                                 f"{2 * grace:.0f} s ({reason}); the Kit DerivedDataCache (size + the next Kit log) "
                                 "is checked before the next Kit launch")
        _wait_until(rt, lambda: proc.poll() is not None, LAUNCHER_EXIT_S)
        if proc.poll() is None and unit and not trainer_pids(rt, unit, match):
            rt.term_scope(unit)                     # only `timeout` / the launcher is left: no Kit in it
            _wait_until(rt, lambda: proc.poll() is not None, 60)
        res["launcher_exited"] = proc.poll() is not None
        ctx.save()
        return res

    def dagger_name(self, base: str, sid: str) -> str:
        """A DAgger run never resumes (its launcher refuses a non-empty run dir): a stage retried after an accepted
        live kill trains afresh under ``<base>_a<n>``."""
        n = int(self.ctx.srec(sid).get("dagger_attempt", 1))
        return base if n <= 1 else f"{base}_a{n}"

    def train_dagger(self, sid: str, launcher: Path, trainer: Path, runs: Path, name: str, args: list,
                     env: dict):
        """One DAgger run through its launcher (foreground: the launcher's rc is the trainer's), watched every poll:
        the DerivedDataCache watch, the progress watchdog (train_log.jsonl / checkpoints; INFRA), and the trainer's
        own mid-run premise read (``<trainer> --read RUN --premise LO:HI`` at [N/3, N/2): its STOP lines, the recipe
        of record's) as the stage's live rule (unvalidated; LIVE). A stop goes through :meth:`stop_nondriver`: the
        trainer pid SIGTERMed twice with the grace, SIGKILL only as the last resort. Returns the rc, or None when
        stopped (the stage HALTed)."""
        ctx, rt = self.ctx, self.ctx.rt
        rec = ctx.srec(sid)
        cfg = self.spec.d.get("live") or {}
        iters = int(args[args.index("--iterations") + 1])
        lo, hi = iters // 3, iters // 2
        pre_launch(ctx, sid, f"DAgger {name}")
        proc = rt.spawn(["bash", str(launcher), "--wait-idle", name, "--", *args], env=env,
                        log=ctx.dir / "tp_logs" / f"{sid}.train.log")
        ctx.say("LAUNCH", f"stage={sid} dagger {name}")
        started = rt.time()
        out_tok = str(runs / name)

        def match(a: str) -> bool:
            t = a.split()
            return any(t[i] == "--out" and t[i + 1] == out_tok for i in range(len(t) - 1))

        read_done = rec.get("premise_read_done", False)
        while proc.poll() is None:
            rt.sleep(cfg.get("poll_s", rt.poll_s))
            kill, why, klass = False, None, "LIVE"
            run_dir = runs / name
            bad = ddc_watch_problem(ctx)
            if bad:
                kill, why, klass = True, [bad], "GUARD"
            if not kill:
                stall = self.stall_s(started, dagger_progress_files(run_dir))
                if stall is not None:
                    kill, klass = True, "INFRA"
                    why = [f"no training progress for {stall / 60:.0f} min (watchdog {self.watchdog_min():g} min; "
                           f"train_log.jsonl / checkpoints of {name}): stopped, HALT INFRA"]
            if not kill and self.inject(sid, "LIVE"):
                kill, why = True, ["dry-pass injected live kill"]
            if not kill and not read_done and (run_dir / "train_log.jsonl").is_file():
                last_it = dagger_last_it(run_dir / "train_log.jsonl")
                if last_it is not None and last_it >= hi:
                    rc_, out = rt.capture([L.PY, "-I", str(trainer), "--read", str(run_dir), "--premise", f"{lo}:{hi}"],
                                          env=env)
                    read_done = rec["premise_read_done"] = True
                    try:
                        prem = (json.loads(out) or {}).get("premise") or {}
                    except ValueError:
                        prem = {"stop": None, "unreadable": out[-300:]}
                    rec["premise"] = {"window": [lo, hi], "stop": prem.get("stop"),
                                      "reasons": prem.get("stop_reasons")}
                    ctx.save()
                    if prem.get("stop") and "premise" not in (rec.get("accepted_live") or []):
                        kill, why = True, [f"premise STOP at its {lo}-{hi}: {prem.get('stop_reasons')}"]
            if kill:
                up = runs / f"{name}.unit"
                unit = up.read_text().strip() if up.is_file() else None
                self.stop_nondriver(sid, unit=unit, match=match, proc=proc,
                                    reason={"LIVE": "DAgger live rule", "GUARD": "DerivedDataCache watch",
                                            "INFRA": "progress watchdog"}[klass])
                if klass != "LIVE":
                    halt(ctx, sid, klass, why[0], resumable=("retry",) if klass == "GUARD" else ("retry", "abort"))
                    return None
                rec["live_killed"] = {"rule": "premise" if why and "premise" in why[0] else "injected", "fails": why}
                ctx.save()
                halt(ctx, sid, "LIVE", f"DAgger live rule: {why}", lines=[{"id": f"{sid}.live.premise",
                     "statistic": "DAgger mid-run premise (unvalidated)", "value": why, "unvalidated": True,
                     "rate": None}])
                return None
        return proc.wait()

    def apply_stand(self, sid: str, res: dict) -> None:
        st = self.spec.stage(sid)
        if st.get("x_replay"):
            self.ctx.set_status(sid, "pass", f"X {res.get('x', {}).get('verdict')}")
            self.ctx.notify("NOTIFY", f"stage={sid} X replay {res.get('x', {}).get('verdict')} (decisive only if "
                                      "bitwise; never feeds STOP-R, never holds F19)")
            return
        self.apply(sid, res)

    # -- bundle
    def ex_bundle(self, sid: str) -> None:
        ctx = self.ctx
        heads = {}
        for st in self.spec.stages(ctx.run):
            h = ctx.head(st["id"])
            if h and h.get("ckpt") and st["id"] != sid:
                heads[st["id"]] = {"ckpt": h["ckpt"], "sha256": h.get("sha256")}
        rec = {"run": ctx.run, "tstar": ctx.state.get("tstar"), "student": ctx.state.get("selected_student"),
               "stand": {k: heads.get(k) for k in ("sn", "sh")}, "stage_heads_by_sha256": heads,
               "status": {s: v.get("status") for s, v in ctx.state["stages"].items()},
               "accepted_halts": ctx.state.get("accepted"),
               "notes": "keep-set record (DESIGN 2.4); bundle_experiment.py keep-set + local tag are PLAN step 11"}
        L.write_json_atomic(ctx.dir / "deliverable.json", rec)
        L.write_json_atomic(ctx.stage_read_path(sid), rec)
        ctx.set_status(sid, "pass", "deliverable.json written")


class UnitProc:
    """A distill scope the queue ATTACHES to after a restart (no Popen handle): polled by its unit; its rc is unknown
    (0), so the caller judges completion by the run dir's final checkpoint."""

    def __init__(self, rt, unit: str):
        self.rt, self.unit = rt, unit

    def poll(self):
        return None if self.rt.unit_active(f"{self.unit}.scope") else 0

    def terminate(self):
        """No launcher process to signal on an attach: the trainer is stopped by pid (:func:`stop_trainer`)."""
        return None

    def wait(self):
        while self.poll() is None:
            self.rt.sleep(30)
        return 0


def dagger_last_it(train_log: Path):
    last = None
    for ln in Path(train_log).read_text(errors="ignore").splitlines():
        try:
            it = json.loads(ln).get("it")
        except ValueError:
            continue
        if isinstance(it, int):
            last = it
    return last


def student_refused_keys() -> tuple:
    mod = L.load_by_path(L.OMNI / "config/crab_hex/omni_student_spec.py", "tp_omni_student_spec")
    return tuple(mod.STUDENT_RUNNER_KNOBS_REFUSED)


def find_student_run(experiment: str, name: str) -> Path | None:
    d = L.LOG_ROOT / experiment
    hits = sorted(p for p in d.glob(f"*_{name}") if p.is_dir()) if d.is_dir() else []
    return hits[-1] if hits else None


def read_unit(name: str) -> str | None:
    p = L.REPO / "parkour/outputs/omni_plan/task4/distill" / f"{name}.unit"
    return p.read_text().strip() if p.is_file() else None


def sh_read_spec(spec: Spec, st: dict, name: str, table_dir: Path, refs: str) -> dict:
    """The sh_read spec for a new head: the S1-era controls of record (lazy: never re-run unless a line fails) + the
    head's own role (its table stats, its K5 zero-command name) + REFS(T*)."""
    base = L.load_yaml(L.OMNI / st["controls_spec"]) if str(st["controls_spec"]).endswith((".yaml", ".yml")) else \
        json.loads((L.OMNI / st["controls_spec"]).read_text())
    roles = {k: v for k, v in base["roles"].items() if not v.get("head")}
    roles[name] = {"name": name, "head": True, "k5_name": f"{name}z",
                   "table": {str(s): [str(table_dir / f"stats_stand_net_s{s}__*.json")] for s in base.get("seeds", [1, 2])}}
    out = {k: v for k, v in base.items() if k not in ("roles", "refs")}
    out.update({"refs": refs, "roles": roles})
    return out


def stand_gate(ctx: Ctx, sid: str, which: str, rec: dict, *, x: bool = False) -> dict:
    """The stand lines of thresholds.json (D1 = SN G1-G5 on sn_calib's head row, D2 = SH K1-K6 on sh_read's head
    gates), K4 at <= 3 / 800 per env seed (D9) and the relaxed G4 recomputed on REFS(T*) (CHARTER STOP-0).

    A thresholds ``watch`` id (the STOP-0 G2 post-settle stillness lines; the SN G1 taps / steps lines from v1.1,
    CHARTER 2026-10-21 decision 1) is never decisive, but it is READ and RECORDED like a decisive clause
    (``watch_lines``: value, line, pass) and a miss raises a WATCH flag (``watch_flags``, gate_walker's shape), so the
    verdict is WATCH, not PASS (CHARTER 2026-10-21 decision B; before it the watch lines were dropped silently).
    A degenerate line's rate is None + ``rate_status`` (decision C), never its plain 0.0."""
    th = ctx.thresholds()
    watch = th.watch_ids
    stand = [x_ for x_ in th.lines if x_.get("stage") == ("D1 (SN)" if which == "D1" else "D2 (SH)")
             and x_.get("kind") == "stand"]
    dec, wrd, notes = [], [], []
    # the STOP-0 relaxed G4 (REFS(T*) - max(0.10, 2 SE)) applies to its clauses wherever they are read: D1's G4 and
    # K5's G4 (thresholds.json carries it as a recommended stand override on the SN.G4.<clause> key)
    relaxed = {k.split(".", 2)[2] for k in ((th.doc.get("recommended") or {}).get("stand_overrides") or {})
               if k.startswith("SN.G4.")}
    if which == "D1":
        doc = L.read_json(Path(rec["calib"])) or {}
        row = ((doc.get("rows") or {}).get(rec["name"]) or {}).get("gates") or {}

        def read(ln):
            g, cl = ln["gate"], ln["clause"]
            c = ((row.get(g) or {}).get("clauses") or {}).get(cl)
            return stand_clause(ln, c, rec, relaxed_g4=(g == "G4" and cl in relaxed))
    else:
        doc = L.read_json(Path(rec["read"])) or {}
        gates = doc.get("gates") or {}
        k5 = L.read_json(Path(rec["k5_calib"])) if rec.get("k5_calib") else None
        k5row = (((k5 or {}).get("rows") or {}).get(f"{rec['name']}z") or {}).get("gates") or {}

        def read(ln):
            if ln["gate"].startswith("K5/"):
                g = ln["gate"].split("/", 1)[1]
                c = ((k5row.get(g) or {}).get("clauses") or {}).get(ln["clause"])
                return stand_clause({**ln, "gate": g}, c, rec, relaxed_g4=(g == "G4" and ln["clause"] in relaxed))
            c = ((gates.get(ln["gate"]) or {}).get(rec["name"]) or {}).get("clauses", {}).get(ln["clause"])
            return stand_clause(ln, c, rec)
        notes.append({"sh_read_K5_verdicts": {g: v.get("value") for g, v in
                                              (((gates.get("K5") or {}).get(rec["name"]) or {}).get("clauses")
                                               or {}).items()}})
    for ln in stand:
        if ln["id"] in watch:
            wrd.append({**read(ln), "watch": True})
        else:
            dec.append(read(ln))
    flags = GR.watch_flags_of(wrd, "misses its WATCH line (thresholds 'watch' list: recorded, never decisive)")
    res = {"verdict": GR.verdict_of(dec, [], watch_flags=flags), "decisive": dec,
           "halt_lines": [d["id"] for d in dec if not d.get("pass")], "watch_lines": wrd, "watch_flags": flags,
           "head": rec, "notes": notes}
    if x:
        cmp_ = ctx.spec.stage(sid).get("compare") or {}
        rule = ctx.spec.d.get("x_rule")
        rule = rule if rule in ("bitwise", "report_only") else "report_only"
        ref_log = (L.REPO / cmp_["train_log"]) if cmp_.get("train_log") else None
        res["x"] = RR().x_read(Path(rec["ckpt"]), cmp_.get("sha256", ""), rule=rule,
                               new_log=Path(rec["train_log"]) if rec.get("train_log") else None, ref_log=ref_log)
        res["verdict"] = GR.PASS
    res["path"] = str(write_gate(ctx, sid, res))
    return res


def stand_clause(ln: dict, c: dict | None, rec: dict, *, relaxed_g4: bool = False) -> dict:
    """One stand clause: the relaxed G4 (recomputed on REFS(T*) from the clause's own REFS read), the override line
    when thresholds.json carries one (K4 <= 3 / 800), else the clause's own verdict at its S1-era line. The rate is
    the line's (None + ``rate_status`` for a degenerate line: GR.rate_fields, decision C)."""
    rf = GR.rate_fields(ln)
    base = {"id": ln["id"], "statistic": ln.get("clause"), **rf,
            "unvalidated": True if rf.get("rate_status") else ln.get("unvalidated")}
    if c is None:
        return {**base, "pass": False, "missing": True, "why": "clause not in the reader's output"}
    v = c.get("value")
    ov = ln.get("override_line")
    if relaxed_g4:
        line = relaxed_g4_line(c, ln)
        ok = v is not None and line is not None and v >= line - 1e-12
        return {**base, "value": v, "line": line, "direction": ">=", "pass": bool(ok),
                "line_rule": "REFS(T*) - max(0.10, 2 SE) (CHARTER STOP-0 relaxed G4)"}
    if ov is not None:
        d = ln.get("direction", "<=")
        ok = v is not None and ((v <= ov + 1e-12) if d == "<=" else (v >= ov - 1e-12))
        return {**base, "value": v, "line": ov, "direction": d, "pass": bool(ok)}
    return {**base, "value": v, "line": c.get("line"), "pass": bool(c.get("ok")), "state": c.get("state")}


def relaxed_g4_line(c: dict, ln: dict):
    """REFS - max(0.10, 2 SE) from the clause's own line note ("vs REFS <p> n=<n>, SE <se>" / "band <b>")."""
    note = str(c.get("note") or "")
    m = re.search(r"REFS ([0-9.]+) n=(\d+), SE ([0-9.]+)", note)
    if m:
        return float(m.group(1)) - max(0.10, 2.0 * float(m.group(3)))
    m = re.search(r"band ([0-9.]+) = max\(binom_se\(REFS ([0-9.]+)", note)
    if m:
        return float(m.group(2)) - max(0.10, float(m.group(1)))
    return None


# ============================================================================== continue / status
def queue_cmdline_ok(cmd: str, ctx: Ctx) -> bool:
    """Is ``cmd`` (a /proc cmdline, NUL -> space) THIS run's queue: ``pipeline.py _queue ... --run <run>``?"""
    parts = cmd.split()
    return ("_queue" in parts and any(x.endswith("pipeline.py") for x in parts)
            and any(a == "--run" and b == ctx.run for a, b in zip(parts, parts[1:])))


def queue_pid(ctx: Ctx) -> int | None:
    """The live queue of this run, or None: the pid in queue.pid (the queue writes its own at START) AND a /proc
    cmdline that is this run's ``_queue`` (never ``pgrep -f``)."""
    p = ctx.dir / "queue.pid"
    if not p.is_file():
        return None
    try:
        pid = int(p.read_text().split()[0])
    except (ValueError, IndexError):
        return None
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return None
    return pid if queue_cmdline_ok(cmd, ctx) else None


def launch_queue(spec: Spec, run: str) -> int:
    ctx = Ctx(spec, run)
    if queue_pid(ctx):
        print(f"a queue for {run} is already running (pid {queue_pid(ctx)})", file=sys.stderr)
        return 3
    ctx.dir.mkdir(parents=True, exist_ok=True)
    log = ctx.qlog
    cmd = [L.PY, "-I", str(HERE / "pipeline.py"), "_queue", "--spec", str(spec.path), "--run", run]
    # start_new_session detaches (setsid(2) in the child, before exec); NOT `setsid` the program -- util-linux setsid
    # FORKS when its caller is already a session leader, so the Popen pid would die at once and the real queue would
    # run under another pid. The queue writes its OWN pid into queue.pid at START (Queue.loop); this one is provisional.
    with open(log, "a") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=clean_env(),
                             start_new_session=True)
    (ctx.dir / "queue.pid").write_text(f"{p.pid}\n")
    print(f"queue started: pid {p.pid}; sentinels in {log} ({L.SENTINEL} START|STAGE|ROW|GATE|NOTIFY|HALT|PARKED|END)")
    return 0


def preflight(spec: Spec, run: str) -> list[str]:
    out = placeholder_problems(spec, run)
    out += lineage_problems(spec) + graph_problems(spec)
    out += [f"gen drift: {p}" for p in gen(spec, run, check=True)]
    out += guard_problems(spec)
    return out


def do_continue(spec: Spec, run: str, *, accept=None, amend=None, charter=None, retry=None, abort=False,
                why: str = "", relaunch: bool = True, retarget_tstar: bool = False) -> int:
    ctx = Ctx(spec, run)
    if queue_pid(ctx):
        print("the queue is running: continue acts on a parked run only", file=sys.stderr)
        return 3
    if retarget_tstar:
        if not amend:
            print("--retarget-tstar goes with --amend STAGE", file=sys.stderr)
            return 2
        return do_retarget_tstar(ctx, amend, charter=charter, why=why, relaunch=relaunch)
    if abort:
        ctx.state["status"] = "aborted"
        for sid, h in ctx.state["halts"].items():
            if h.get("status") == "open":
                h["status"] = "aborted"
        ctx.save()
        ctx.notify("NOTIFY", f"run aborted by the user ({why})")
        return 0
    sid = accept or amend or retry
    h = (ctx.state["halts"] or {}).get(sid)
    if h is None or h.get("status") != "open":
        print(f"no open HALT at {sid}", file=sys.stderr)
        return 2
    rec_p = L.REPO / h["record"]
    rec = L.read_json(rec_p)
    srec = ctx.srec(sid)
    if accept:
        if h["class"] in ("INVALID", "GUARD", "INFRA", "DRIVER", "CRASH"):
            print(f"a {h['class']} HALT is not a chance event: fix the cause and use --retry {sid}", file=sys.stderr)
            return 2
        if h["class"] == "STUDENT" or spec.stage(sid)["kind"] in ("tstar", "select"):
            print(f"the {sid} HALT ({h['class']}) leaves no head to continue from (no selected student / no T*): "
                  "accepting it as chance would build on nothing -- use --amend (a CHARTER entry) or --abort",
                  file=sys.stderr)
            return 2
        rec["status"] = "accepted"
        rec["resolution"] = {"by": "user", "at": L.now_iso(), "kind": "accept-chance", "why": why,
                             "disclosure": "accepted as chance: the run continues and stays eligible as the "
                                           "deliverable, disclosed in the CHARTER and the report (PLAN On FAIL)"}
        h["status"] = "accepted"
        ctx.state["accepted"].append({"stage": sid, "class": h["class"], "at": L.now_iso(), "why": why,
                                      "chance_rate_sum": rec.get("chance_rate_sum")})
        if h["class"] in ("LIVE", "STOP-R") and srec.get("live_killed"):
            rule = (srec.get("live_killed") or {}).get("rule")
            srec.setdefault("accepted_live", []).append(rule)
            srec.pop("live_killed", None)
            srec.pop("fresh_run_on_retry", None)
            if spec.stage(sid)["kind"] in ("dagger_sn", "dagger_sh"):
                srec["dagger_attempt"] = int(srec.get("dagger_attempt", 1)) + 1
                srec.pop("premise_read_done", None)
                ctx.set_status(sid, "pending", f"DAgger live {rule} accepted as chance: a fresh run (no resume)")
            elif spec.stage(sid)["kind"] == "distill":
                srec["distill_attempt"] = int(srec.get("distill_attempt", 1)) + 1
                srec.pop("trained", None)
                srec.pop("run_dir", None)
                ctx.set_status(sid, "pending", f"distill live {rule} accepted as chance: a fresh run _a"
                                               f"{srec['distill_attempt']} (records stay unambiguous)")
            else:
                ctx.set_status(sid, "training", f"live {rule} accepted as chance: the window resumes")
        elif h["class"] == "STOP-R":
            srec["stop_r_accepted"] = True
            ctx.set_status(sid, "training", "STOP-R accepted (statistical-only parity, disclosed)")
        else:
            ctx.set_status(sid, "accepted", f"{h['class']} accepted as chance")
    elif amend:
        if not charter:
            print("--amend needs --charter (the CHARTER entry that discloses the number)", file=sys.stderr)
            return 2
        rec["status"] = "amended"
        rec["resolution"] = {"by": "user", "at": L.now_iso(), "kind": "amend", "charter": charter, "why": why}
        h["status"] = "amended"
        lp = spec.lock_path()
        ctx.state["amendments"].append({"stage": sid, "at": L.now_iso(), "charter": charter,
                                        "old_lock_sha256": (ctx.state.get("pin") or {}).get("lock_sha256"),
                                        "new_lock_sha256": L.sha256_file(lp) if lp.is_file() else None})
        if ctx.state.get("pin"):
            ctx.state["pin"]["lock_sha256"] = L.sha256_file(lp) if lp.is_file() else None
        ctx.set_status(sid, "gate", f"amended ({charter}): re-gated under the amended spec / thresholds")
    else:
        rec["status"] = "retried"
        rec["resolution"] = {"by": "user", "at": L.now_iso(), "kind": "retry", "why": why}
        h["status"] = "retried"
        back = {"INVALID": "rows", "INFRA": None, "GUARD": None, "DRIVER": "training", "CRASH": None}.get(h["class"])
        prev = [x for x in srec.get("history", []) if x.get("to") != "halt"]
        target = back or (prev[-1]["to"] if prev else "pending")
        if h["class"] == "INVALID":
            srec.setdefault("attempts", {})["invalid"] = 0
        w = ctx.state.get("ddc_watch")
        if h["class"] == "GUARD" and w and not w.get("closed_at") and w.get("symptom"):
            # the user quarantined the cache: the NEXT Kit start after this retry must be clean
            w.update({"rearmed_at": L.now_iso(), "at_epoch": time.time(), "ddc_gb_at_kill": ddc_gb(),
                      "previous_symptom": w.pop("symptom")})
        kind = spec.stage(sid)["kind"]
        if srec.pop("fresh_run_on_retry", None) and kind in ("distill", "dagger_sn", "dagger_sh"):
            # the queue stopped this run (live rule / watchdog / cache watch): a DAgger run dir never resumes and a
            # distill attempt keeps its records unambiguous -- the retry trains afresh under _a<n>
            key = "distill_attempt" if kind == "distill" else "dagger_attempt"
            srec[key] = int(srec.get(key, 1)) + 1
            for k in ("trained", "run_dir", "premise_read_done", "live_killed"):
                srec.pop(k, None)
            target = "pending"
        ctx.set_status(sid, target if target not in TERMINAL else "pending", f"retry after {h['class']}")
    L.write_json_atomic(rec_p, rec)
    for d in dependents(ctx, sid):
        if ctx.srec(d).get("status") == "blocked":
            ctx.set_status(d, "pending", f"unblocked: {sid} {h['status']}")
    ctx.save()
    ctx.notify("NOTIFY", f"continue: {sid} {h['status']} ({why or charter or ''})")
    return launch_queue(spec, run) if relaunch else 0


# ============================================================================== continue --amend STAGE --retarget-tstar
def history_copy(p: Path) -> str | None:
    """Copy record ``p`` VERBATIM to ``<its dir>/history/<stem>.<k>.json`` (the next free k; never overwritten) and
    return its repo path -- a superseded record is kept, never lost (CHARTER 2026-10-22)."""
    p = Path(p)
    if not p.is_file():
        return None
    d = p.parent / "history"
    k = 1
    while (d / f"{p.stem}.{k}.json").exists():
        k += 1
    q = d / f"{p.stem}.{k}.json"
    L.write_text_atomic(q, p.read_text())
    return L.rel(q)


def distill_run_name(run: str, sid: str, attempt: int) -> str:
    """ex_distill's run name: ``T4<run>_<sid>``, ``_a<n>`` from the second attempt on."""
    return f"T4{run}_{sid}" + (f"_a{attempt}" if attempt > 1 else "")


def retarget_problems(ctx: Ctx, sid: str, tsid: str | None, *, charter: str | None, relaunch: bool) -> list[str]:
    """Why ``continue --amend <sid> --retarget-tstar`` refuses (nothing is written when any holds)."""
    spec = ctx.spec
    out = []
    if not charter:
        out.append("--amend needs --charter (the CHARTER entry that records the decision)")
    if tsid is None:
        out.append(f"run {ctx.run} has no T* stage (kind tstar)")
        return out
    h = (ctx.state.get("halts") or {}).get(sid)
    if h is None or h.get("status") != "open":
        out.append(f"no open HALT at {sid}")
    scope = [tsid] + dependents(ctx, tsid)
    if sid not in scope:
        out.append(f"--retarget-tstar re-picks T*: the amended HALT must be at {tsid} or a stage built on it "
                   f"({', '.join(scope)}), not {sid}")
    other = sorted(s for s, x in (ctx.state.get("halts") or {}).items() if x.get("status") == "open" and s != sid)
    if other:
        out.append(f"other open HALT(s) {other}: one decision per command -- resolve them first")
    if (ctx.dir / "STOP_PIPELINE").exists() and relaunch:
        out.append("STOP_PIPELINE is present, so the relaunched queue would stop at once. This command never deletes "
                   "it: remove it first (the main thread / the user), or pass --no-relaunch and remove it before "
                   "`pipeline.py run`")
    th = spec.thresholds_path()
    if not th.is_file():
        out.append(f"thresholds file {th} missing")
    elif json.loads(th.read_text()).get("version") != spec.d["thresholds"]["version"]:
        out.append(f"thresholds file version {json.loads(th.read_text()).get('version')} != spec "
                   f"{spec.d['thresholds']['version']} (which thresholds re-gate is ambiguous)")
    for d in scope:
        if spec.stage(d)["kind"] == "distill":
            name = distill_run_name(ctx.run, d, int(ctx.srec(d).get("distill_attempt", 1)))
            u = read_unit(name)
            if u and ctx.rt.unit_active(f"{u}.scope"):
                out.append(f"{d}: the distill scope {u} ({name}) is still active -- stop its trainer first")
    return out


def do_retarget_tstar(ctx: Ctx, sid: str, *, charter: str | None, why: str = "", relaunch: bool = True) -> int:
    """``continue --run <run> --amend <sid> --retarget-tstar --charter '<CHARTER ref>' --why '...'`` (CHARTER
    2026-10-22, F19 p3b -> USER DECISION "amend + switch now"): a thresholds amendment that changes the paired 0-kg
    check re-picks T* in ONE recorded command, after the re-pin (``pipeline.py lock``; the run's pin moves exactly as
    ``--amend`` moves it). ``sid`` is the stage holding the run's open HALT (F19: ``st.1``, DRIVER -- the p3a-based
    distill stopped by SIGTERM on its trainer pid), which must be T* itself or a stage built on it.

    a. RE-GATE: every TRAINED head of T*'s candidate list (p3a, p3b; p3c only if trained) re-reads its paired 0-kg
       check under the CURRENT thresholds file (:func:`zero_kg_of`, the gate's own call). Its stage record's
       ``zero_kg`` is replaced (the previous one appended to ``zero_kg_history``); its gate record is copied verbatim
       to ``gates/history/<stage>.<k>.json`` first, then updated in place (``zero_kg``, the decisive 0-kg line where
       the stage has one, the verdict recomputed by :func:`payload_verdict`, a ``regates`` entry naming the copy and
       both verdicts). The STAGE STATUS NEVER CHANGES: an accepted head stays ``accepted`` (its accepted HALT record
       is annotated with the re-gate, never reopened), a ceiling stays ``ceiling`` (a measured ceiling, unrelated to
       the 0-kg check).
    b. T*: :meth:`Queue.ex_tstar` -- the existing rule and code path (the highest max_stable that passes its 0-kg
       check, accepted-as-chance heads eligible) -- writes the T* record and its NOTIFY listing every head; the
       previous record is kept under ``stage_reads/history/`` and summarised in ``state.tstar_history``.
    c. RESET: every stage built on T* (F19: st, sel, sn, sh, bundle) goes back to ``pending`` with no stale step
       record: its record keys move to ``superseded`` (with the run name it trained under), its stage read / rows /
       gate records move to ``history/`` (copied verbatim, then removed), ``state.selected_student`` moves to
       ``superseded_selected_student``; a distill / DAgger stage that had started trains AFRESH under the next
       attempt (``distill_attempt`` + 1 -> ``T4F19_st_a2``) -- the stopped run dir is never resumed and stays on disk
       as a record. The HALT closes as ``amended`` with the CHARTER reference.
    d. PIN: ``state.amendments`` gains the entry (old / new lock sha, T* from / to, the thresholds file) and the pin's
       lock sha moves to the current lock, exactly like ``--amend``.
    e. REFUSES (exit 2, nothing written): no --charter; no open HALT at ``sid`` or ``sid`` not T* / built on it;
       another open HALT; STOP_PIPELINE present with a relaunch (the command never deletes it: the main thread removes
       it, or passes --no-relaunch and removes it before ``pipeline.py run``); the thresholds file's version is not
       the spec's; a distill scope still active; no trained candidate; a re-gate error; the re-gated T* is NONE or
       UNCHANGED (then nothing is reset). The queue running is refused by :func:`do_continue` (exit 3).
    It touches only ``ctx``'s run dir (never another run's state)."""
    spec = ctx.spec
    tsid = next((s["id"] for s in spec.stages(ctx.run) if s["kind"] == "tstar"), None)
    probs = retarget_problems(ctx, sid, tsid, charter=charter, relaunch=relaunch)
    if probs:
        print("retarget refused (nothing written):\n  " + "\n  ".join(probs), file=sys.stderr)
        return 2
    th_path = spec.thresholds_path()
    th = GR.Thresholds.load(th_path)
    th_rec = {"file": th_path.name, "version": spec.d["thresholds"]["version"], "sha256": L.sha256_file(th_path)}
    q = Queue(ctx)
    # -- a. re-gate in memory first: nothing is written until T* is known to change
    regated = {}
    for c in spec.stage(tsid)["candidates"]:
        r = ctx.state["stages"].get(c) or {}
        if r.get("status") not in ("pass", "accepted", "ceiling") or not ctx.head(c):
            continue                                  # not trained: ex_tstar lists it as before
        try:
            regated[c] = zero_kg_of(ctx, c, th)
        except Exception as exc:  # noqa: BLE001  (a refusal, never a half-written record)
            print(f"retarget refused (nothing written): the {c} 0-kg re-gate raised {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 2
    if not regated:
        print("retarget refused (nothing written): no trained payload head to re-gate", file=sys.stderr)
        return 2
    old_t = (ctx.state.get("tstar") or {}).get("stage")
    live = ctx.state
    sim = copy.deepcopy(live)
    for c, zk in regated.items():
        sim["stages"][c]["zero_kg"] = zero_kg_record(zk)
    ctx.state = sim
    try:
        _considered, _nt, t = q.tstar_considered(tsid)
    finally:
        ctx.state = live
    if t is None or t["stage"] == old_t:
        print(f"retarget refused (nothing written): under thresholds {th_rec['version']} the re-gated T* would be "
              + ("NONE (no trained head passes and none is accepted)" if t is None else f"UNCHANGED ({old_t})")
              + "; verdicts: " + "; ".join(f"{c} {zk['verdict']} (max margin use {zk.get('max_margin_used')})"
                                           for c, zk in regated.items()), file=sys.stderr)
        return 2
    PR()                                       # ex_tstar's label module: loaded before the first write
    at = L.now_iso()
    tag = {"at": at, "kind": "retarget_tstar", "charter": charter, "why": why, "thresholds": th_rec}
    say = []
    # -- a. the payload heads' records (status unchanged)
    for c, zk in regated.items():
        r = ctx.srec(c)
        before = r.get("zero_kg")
        after = {**zero_kg_record(zk), "regated": tag}
        r.setdefault("zero_kg_history", []).append({**(before or {}), "superseded_at": at,
                                                    "superseded_by_thresholds": th_rec["version"],
                                                    "charter": charter})
        r["zero_kg"] = after
        gp = ctx.dir / "gates" / f"{c}.json"
        g = L.read_json(gp)
        if g is not None:
            prev = history_copy(gp)
            v_before = g.get("verdict")
            for d in g.get("decisive") or []:
                if d.get("id") == f"{c}.capability.zero_kg_paired":
                    d.update({**zero_kg_line(c, zk), "regated_under": th_rec["version"]})
            if not g.get("gate0"):
                g["verdict"], g["watch_flags"] = payload_verdict(ctx, c, g.get("decisive") or [], [],
                                                                 g.get("max_stable"))
                g["halt_lines"] = [d["id"] for d in g.get("decisive") or [] if not d.get("pass")]
            g["zero_kg"] = after
            g["zero_kg_thresholds"] = th_rec
            g.setdefault("regates", []).append({**tag, "previous_record": prev, "verdict_before": v_before,
                                                "verdict_after": g["verdict"], "zero_kg_before": before,
                                                "zero_kg_after": zero_kg_record(zk),
                                                "note": "only the paired 0-kg check was re-read; every other line "
                                                        "is the original gate read (thresholds above)"})
            L.write_json_atomic(gp, g)
            if r.get("gate"):
                r["gate"] = {**r["gate"], "verdict": g["verdict"],
                             "regated": {"at": at, "verdict_before": v_before, "thresholds": th_rec["version"]}}
        hh = (ctx.state.get("halts") or {}).get(c)
        if hh and hh.get("status") == "accepted":
            hp = L.REPO / hh["record"]
            hr = L.read_json(hp) or {}
            hr.setdefault("regates", []).append({
                **tag, "zero_kg": zero_kg_record(zk),
                "note": (f"the accepted HALT stands (status 'accepted' unchanged): under thresholds "
                         f"{th_rec['version']} the paired 0-kg check reads {zk['verdict']}"
                         + (f" on {zk['fails']}" if zk["fails"] else ""))})
            L.write_json_atomic(hp, hr)
        r["history"].append({"at": at, "from": r.get("status"), "to": r.get("status"),
                             "why": f"0-kg re-gated under thresholds {th_rec['version']} ({charter}): "
                                    f"{(before or {}).get('verdict')} -> {zk['verdict']}; status unchanged"})
        say.append(f"{c} 0-kg {(before or {}).get('verdict')} -> {zk['verdict']} "
                   f"(max margin use {zk.get('max_margin_used')})")
    # -- b. the previous T* record is kept
    prev_ts = history_copy(ctx.stage_read_path(tsid))
    old_rec = ctx.state.get("tstar") or {}
    ctx.state.setdefault("tstar_history", []).append(
        {**{k: old_rec.get(k) for k in ("stage", "ckpt", "sha256", "iteration", "max_stable")},
         "superseded_at": at, "previous_record": prev_ts, "charter": charter,
         "superseded_by_thresholds": th_rec["version"]})
    # -- c. reset everything built on T*
    fresh, reset = {}, []
    for d in dependents(ctx, tsid):
        r = ctx.srec(d)
        kind = spec.stage(d)["kind"]
        stale = {k: v for k, v in r.items() if k not in ("status", "history", "superseded")}
        started = bool(stale) or r.get("status") not in ("pending", "blocked", "skipped")
        new = {"status": "pending", "history": r["history"], "superseded": list(r.get("superseded") or [])}
        if started:
            entry = {"at": at, "charter": charter, "status": r.get("status"), "records": stale}
            if kind == "distill":
                n = int(r.get("distill_attempt", 1))
                entry["run_name"] = distill_run_name(ctx.run, d, n)
                new["distill_attempt"] = fresh[d] = n + 1
            elif kind in ("dagger_sn", "dagger_sh"):
                n = int(r.get("dagger_attempt", 1))
                new["dagger_attempt"] = fresh[d] = n + 1
            moved = []
            for p in (ctx.stage_read_path(d), ctx.rows_path(d), ctx.dir / "gates" / f"{d}.json"):
                cp = history_copy(p)
                if cp:
                    p.unlink()
                    moved.append(cp)
            entry["records_moved_to"] = moved
            new["superseded"].append(entry)
        if not new["superseded"]:
            new.pop("superseded")
        new["history"].append({"at": at, "from": r.get("status"), "to": "pending",
                               "why": f"T* retargeted {old_t} -> {t['stage']} ({charter}): "
                                      + ("a fresh run" + (f" (attempt {fresh[d]})" if d in fresh else "")
                                         if started else "reset, nothing had run")})
        ctx.state["stages"][d] = new
        reset.append(d)
    if "selected_student" in ctx.state:
        ctx.state.setdefault("superseded_selected_student", []).append(
            {"at": at, "charter": charter, "record": ctx.state.pop("selected_student")})
    # -- c. the HALT closes as amended
    h = ctx.state["halts"][sid]
    rec_p = L.REPO / h["record"]
    rec = L.read_json(rec_p) or {}
    rec["status"] = "amended"
    rec["resolution"] = {"by": "user", "at": at, "kind": "amend", "charter": charter, "why": why,
                         "retarget_tstar": {"from": old_t, "to": t["stage"], "thresholds": th_rec,
                                            "regated": {c: zk["verdict"] for c, zk in regated.items()},
                                            "reset": reset, "fresh_attempts": fresh}}
    h["status"] = "amended"
    L.write_json_atomic(rec_p, rec)
    # -- d. the pin moves exactly like --amend
    lp = spec.lock_path()
    new_lock = L.sha256_file(lp) if lp.is_file() else None
    ctx.state["amendments"].append({"stage": sid, "at": at, "charter": charter,
                                    "old_lock_sha256": (ctx.state.get("pin") or {}).get("lock_sha256"),
                                    "new_lock_sha256": new_lock, "kind": "retarget_tstar",
                                    "tstar_from": old_t, "tstar_to": t["stage"], "thresholds": th_rec})
    if ctx.state.get("pin"):
        ctx.state["pin"]["lock_sha256"] = new_lock
    # -- b. T* through the existing rule and code path (writes the record + the NOTIFY listing every head)
    ctx.srec(tsid)["history"].append({"at": at, "from": ctx.srec(tsid).get("status"), "to": "gate",
                                      "why": f"re-pick: payload heads' 0-kg re-gated under thresholds "
                                             f"{th_rec['version']} ({charter})"})
    ctx.srec(tsid)["status"] = "gate"
    q.ex_tstar(tsid)
    trec = ctx.head(tsid) or {}
    trec["retarget"] = {**tag, "from": old_t, "previous_record": prev_ts}
    L.write_json_atomic(ctx.stage_read_path(tsid), trec)
    ctx.state["tstar"] = trec
    ctx.save()
    stop = (ctx.dir / "STOP_PIPELINE").exists()
    runs = [distill_run_name(ctx.run, d, n) if spec.stage(d)["kind"] == "distill" else f"{d} attempt {n}"
            for d, n in fresh.items()]
    msg = (f"continue: {sid} amended ({charter}) -- T* retargeted {old_t} -> {t['stage']} (max_stable "
           f"{t['max_stable']}) under thresholds {th_rec['version']}; " + "; ".join(say) + f"; reset {reset}"
           + (f"; fresh runs {runs}" if runs else ""))
    ctx.notify("NOTIFY", msg)
    print(msg)
    if not relaunch:
        print("not relaunched (--no-relaunch)" + ("; STOP_PIPELINE is still present: remove it, then "
                                                  f"`pipeline.py run --run {ctx.run}`" if stop else ""))
        return 0
    return launch_queue(spec, ctx.run)


def status_text(spec: Spec, run: str) -> str:
    ctx = Ctx(spec, run)
    s = ctx.state
    lines = [f"run {run}: {s.get('status')} (queue pid {queue_pid(ctx)}), pin {s.get('pin')}"]
    for st in spec.stages(run):
        r = s["stages"].get(st["id"]) or {}
        g = (r.get("gate") or {}).get("verdict")
        lines.append(f"  {st['id']:7s} {r.get('status', 'pending'):9s} gate={g} "
                     f"{('max_stable=' + str(r.get('max_stable'))) if 'max_stable' in r else ''}")
    for sid, h in (s.get("halts") or {}).items():
        lines.append(f"  HALT {sid}: {h['class']} {h['status']} ({h['record']})")
    if ctx.qlog.is_file():
        tail = [ln for ln in ctx.qlog.read_text().splitlines() if ln.startswith(L.SENTINEL)][-3:]
        lines += ["  last: " + t for t in tail]
    return "\n".join(lines)


# ============================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("gen", "check", "lock", "run", "continue", "status", "gate", "report", "_queue"))
    ap.add_argument("--spec", type=Path, default=os.environ.get("TP_SPEC"), required=os.environ.get("TP_SPEC") is None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--check", action="store_true", help="gen: compare instead of writing")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--accept-chance", default=None, metavar="STAGE")
    ap.add_argument("--amend", default=None, metavar="STAGE")
    ap.add_argument("--charter", default=None)
    ap.add_argument("--retry", default=None, metavar="STAGE")
    ap.add_argument("--abort", action="store_true")
    ap.add_argument("--why", default="")
    ap.add_argument("--no-relaunch", action="store_true")
    ap.add_argument("--retarget-tstar", action="store_true",
                    help="with --amend STAGE (T* or a stage built on it): re-gate the payload heads' paired 0-kg check "
                         "under the current thresholds, re-pick T*, reset T*'s subtree to fresh runs, move the pin; "
                         "refused when T* would not change (CHARTER 2026-10-22; do_retarget_tstar)")
    a = ap.parse_args(argv)
    spec = Spec(a.spec)
    if a.cmd == "gen":
        runs = [a.run] if a.run else list(spec.d["runs"])
        bad = []
        for r in runs:
            out = gen(spec, r, check=a.check)
            if a.check:
                bad += out
            else:
                print(f"{r}: {len(out)} files")
            if is_placeholder(spec.n_p()) and spec.official(r):
                print(f"  WARNING {r}: N_p is a PLACEHOLDER -- the payload stages carry it; check / run refuse")
        if a.check:
            print("\n".join(bad) if bad else "gen --check: byte-identical")
            return 1 if bad else 0
        return 0
    if a.cmd == "lock":
        doc = lock_doc(spec)
        L.write_json_atomic(spec.lock_path(), doc)
        print(f"wrote {spec.lock_path()}: {len(doc['files'])} files; dirty {len(doc['dirty_code_eval'])}")
        print(FREEZE_NOTICE)
        if doc["dirty_code_eval"]:
            print(f"  NOT A VALID PIN: the code / eval tree is dirty ({doc['dirty_code_eval'][:3]}): commit or stash "
                  "before the pin, then re-run lock.")
        return 0
    if a.cmd == "check":
        runs = [a.run] if a.run else [r for r in spec.d["runs"] if spec.official(r)]
        probs = lineage_problems(spec) + graph_problems(spec)
        for r in runs:
            probs += [f"{r}: {p}" for p in placeholder_problems(spec, r)]
            probs += [f"{r}: gen drift {p}" for p in gen(spec, r, check=True)]
            vspec, note = spec, ""
            if is_placeholder(spec.n_p()):
                # the structure is still validated, with a STAND-IN N_p (the placeholder itself is reported above)
                vspec = copy.deepcopy(spec)
                vspec.d["N_p"] = 1500
                note = " [validated with a stand-in N_p = 1500; N_p itself is a PLACEHOLDER]"
            try:
                probs += [f"{r}: {p}{note}" for p in validate_generated(vspec, r)]
            except Exception as exc:  # noqa: BLE001
                probs.append(f"{r}: driver validator raised {exc!r}")
        th = spec.thresholds_path()
        if not th.is_file():
            probs.append(f"thresholds file {th} missing")
        elif json.loads(th.read_text()).get("version") != spec.d["thresholds"]["version"]:
            probs.append(f"thresholds version {json.loads(th.read_text()).get('version')} != spec "
                         f"{spec.d['thresholds']['version']}")
        probs += [f"guard: {p}" for p in guard_problems(spec, check_disk=False)]
        print("\n".join(probs) if probs else "check: clean")
        return 1 if probs else 0
    if a.run is None:
        ap.error("--run is required")
    if a.cmd == "run":
        probs = preflight(spec, a.run)
        if probs:
            print("refused:\n  " + "\n  ".join(probs), file=sys.stderr)
            return 2
        return launch_queue(spec, a.run)
    if a.cmd == "_queue":
        for k in list(os.environ):
            if k.startswith(("KRABBY_", "SN_WALKER_", "P2R_", "SN_STAND_")):
                del os.environ[k]
        ctx = Ctx(spec, a.run)
        res = Queue(ctx).loop()
        return 0 if res in ("done", "parked", "stopped", "aborted") else 1
    if a.cmd == "continue":
        if sum(bool(x) for x in (a.accept_chance, a.amend, a.retry, a.abort)) != 1:
            ap.error("continue takes exactly one of --accept-chance / --amend / --retry / --abort")
        if a.retarget_tstar and not a.amend:
            ap.error("--retarget-tstar goes with --amend STAGE")
        return do_continue(spec, a.run, accept=a.accept_chance, amend=a.amend, charter=a.charter, retry=a.retry,
                           abort=a.abort, why=a.why, relaunch=not a.no_relaunch, retarget_tstar=a.retarget_tstar)
    if a.cmd == "status":
        print(status_text(spec, a.run))
        return 0
    if a.cmd == "gate":
        if not a.stage:
            ap.error("gate needs --stage")
        res = gate_stage(Ctx(spec, a.run), a.stage, thresholds=a.thresholds,
                         out=(spec.run_dir(a.run) / "gates" / f"{a.stage}.reread.json"))
        print(json.dumps({k: res.get(k) for k in ("verdict", "halt_lines", "gate0")}, indent=1, default=str))
        return GR.EXIT.get(res.get("verdict"), 4)
    if a.cmd == "report":
        rep = L.load_by_path(HERE / "report.py", "tp_report")
        return rep.main(["--spec", str(spec.path), "--run", a.run])
    return 2


if __name__ == "__main__":
    sys.exit(main())
