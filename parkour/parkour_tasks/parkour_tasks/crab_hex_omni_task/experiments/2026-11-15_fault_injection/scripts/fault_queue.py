#!/usr/bin/env python3
"""Fault-injection sweep: the ONE serial GPU queue runner (CHARTER §4 rows, §6 checks, §7.3 pin, §8 stops, §9 order).

    fault_queue.py pin [--repin]                        write results/pin.json (clean tree only; CHARTER §7.3)
    fault_queue.py make-smoke-envs [--check]            write / verify manifests/envs_smoke/*.env (smoke onset override)
    fault_queue.py list     --phase P                   print the phase's rows (label, slot, estimate)
    fault_queue.py estimate                             GPU-h per phase + the campaign projection
    fault_queue.py run      --phase P [--dry-run]       run the phase's rows serially (launch it detached: run_queue.sh)

Phases (CHARTER §9 order): smoke, nominals, es1, es2, kind, c1, closing. A row = one ``gate_row.sh`` launch
(``experiments/tools/pipeline/gate_row.sh NAME CKPT ENV_FILE MANIFEST SCENARIO ES [args]``) with ``TP_OUT`` = MAIN
``parkour/outputs/fault_sweep`` (stamps under ``$TP_OUT/lr/NAME/SCENARIO/seed00k/<time>/``) and ``TP_LEVEL_FILE`` =
cells.yaml ``levels.L090`` on an L090 row. Row names: ``FS_<ck>_<cell>_<payload>`` for a cell,
``FS_<ck>_NOM_<payload>_<launch>`` for a nominal launch (A, B, extra C/D, closing Z), ``FSS_...`` for the short
smoke-manifest rows. The smoke phase runs some rows of later phases under THEIR names (nominal A, the identity rows);
those phases then skip them (resumable: a slot with a complete stamp at the pin is skipped).

Before EVERY row (a failure writes results/STOP.json, prints ``QUEUE END status=stop phase=<p>`` and exits 1):
  * results/STOP.json absent (a previous stop is cleared by hand, never by the queue);
  * the live build == results/pin.json: code_sha / eval_sha (run_candidates' own functions, gate_row's dirty rule:
    the code + eval pathspecs clean) and the sha256 of CHARTER.md, cells.yaml, envs/*.env, manifests/**  (HEAD moving
    with the code unchanged is reported, not a stop);
  * free disk >= 60 GB at TP_OUT and at this worktree;
  * Kit DerivedDataCache growth <= 15 GB vs results/ddc_start.json (recorded by ``pin``, or on the first run);
  * the GPU-h projection (ledger spend + every not-yet-complete planned row of every phase, es2 at the CHARTER's
    35 cells until its selection exists) <= 27.
After every launched row: gate_row rc 0 and a NEW complete stamp (else a launch error, stop), then
``fault_read.py --root $TP_OUT/lr --check-only <stamp>``: rc 0 ok; rc 3 (class unmatched, CHARTER §5.1) relaunches
the row ONCE, and a second unmatched stamp of the row stops; any other rc (2 launch error, 4 usage) stops. A missing
fault_read.py stops before any launch. Ledger: results/queue_ledger.jsonl (row, phase, start, end, gpu_s, status
ok | class_unmatched | check_fail | launch_error, stamp, gate_rc, check_rc, relaunch). A row whose newest
complete stamp has no ledger check (the queue died mid-row) is checked before it is skipped.

es2 selection (CHARTER §4: es2 for every cell whose es1 worst-row F < 0.5) reads fault_read.py's full read,
results/read.json (``cell_seeds[].read.F_worst``, C1S es1); it must be newer than every es1 stamp and cover all 50
cells, else the es2 phase refuses loudly (rc 2, no STOP.json). ``--es2-cells a,b,...`` overrides it explicitly.

The last line of every ``run`` is ``QUEUE END status=<ok|stop> phase=<p> ...`` (also after a crash).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent                                           # experiments/2026-11-15_fault_injection
EXPS = CAMP.parent
OMNI = EXPS.parent                                           # crab_hex_omni_task
W = OMNI.parents[3]                                          # this worktree's root
TOOLS = EXPS / "tools"
GATE_ROW = TOOLS / "pipeline" / "gate_row.sh"
FAULT_READ = HERE / "fault_read.py"
CELLS_YAML = CAMP / "cells.yaml"
CHARTER = CAMP / "CHARTER.md"
ENVS = CAMP / "envs"
MANIFESTS = CAMP / "manifests"
SMOKE_MANIFEST = MANIFESTS / "scenarios_smoke.yaml"
SMOKE_ENVS = MANIFESTS / "envs_smoke"
RESULTS = CAMP / "results"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
MAIN = Path("/home/nickmagus/krabby/krabby-research")
TP_OUT_DEFAULT = MAIN / "parkour" / "outputs" / "fault_sweep"
DDC = Path("/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache")

BUDGET_GPU_H = 27.0                  # CHARTER §8 / §9
FREE_MIN_GB = 60.0                   # CHARTER §8
DDC_GROWTH_MAX_GB = 15.0             # CHARTER §8
ROW_TIMEOUT_S = 3600                 # gate_row TP_ROW_TIMEOUT (a full student suite row ~430 s)
BUSY_POLL_S = 30
ES2_CELLS_ESTIMATE = 35              # CHARTER §9 (until results/read.json selects es2)
ES2_F_MAX = 0.5                      # CHARTER §4
SUITE_CATS = "stand,fwd,side,back,diag,turn,move_turn"
SMOKE_SUITE_CATS = "stand,fwd"
SMOKE_FL_ONSET = "10.0:14.0"         # manifests/scenarios_smoke.yaml header
PHASES = ("smoke", "nominals", "es1", "es2", "kind", "c1", "closing")
CK_TAG = {"C1S": "C1S", "C1S_zero_depth": "C1S", "c1": "c1"}

# Per-row GPU-seconds model (CHARTER §9: ~10.7 min per full suite + field_light cell-seed; kind rows 2.4 GPU-h / 36).
# row_s = STARTUP_S + K_S_PER_EP_S * (episode seconds summed over waves), fitted so suite (10 x 18 s) = 428 s and
# field_light (80 s) = 214 s; the short smoke rows (16 envs) are charged per episode-second like 100-env rows
# (conservative: a 256-env run is latency-bound). Measured means from the ledger replace the model per scenario.
STARTUP_S = 43.0
K_S_PER_EP_S = 2.14
EP_SECONDS = {"suite__omni": 180.0, "field_light__omni": 80.0, "smk_suite__omni": 36.0, "smk_field_light__omni": 26.0}
FIXED_EST_S = {"rt_ledge_down_b2__omni": 240.0, "rt_pits_b2__omni": 240.0}

# The short smoke rows (scenarios_smoke.yaml) beside cells.yaml's four real rows.
SMOKE_ROWS = {
    "smk_suite": {"manifest": str(SMOKE_MANIFEST), "scenario": "smk_suite__omni", "env_row": "suite",
                  "args": ["--task", "Isaac-Crab-Hex-Omni-Student-Play-v0", "--trained-categories", SMOKE_SUITE_CATS]},
    "smk_field_light": {"manifest": str(SMOKE_MANIFEST), "scenario": "smk_field_light__omni", "env_row": "field_light",
                        "args": ["--task", "Isaac-Crab-Hex-Omni-Student-Play-v0"]},
}
# Dose smoke per fault id (CHARTER §6.1.3): the cell run for each id (a transient level where the id has one, so
# the off-again probe is exercised), and the short row it runs on (depth ids on the field, where the image has
# structure). S8 runs both levels (zero and held are separate mechanisms).
DOSE_SMOKES = [
    ("S1_2", "smk_suite"), ("S2_2", "smk_suite"), ("S3_0p2", "smk_suite"), ("S4_perm", "smk_suite"),
    ("S5_2", "smk_suite"), ("S6_0p1", "smk_suite"), ("S8_zero", "smk_suite"), ("S8_held", "smk_suite"),
    ("S10_2", "smk_field_light"), ("S11_2", "smk_field_light"), ("Z1_2", "smk_field_light"),
    ("B1_25", "smk_suite"), ("B2_0p2", "smk_suite"), ("B3_perm", "smk_suite"), ("B4_0p5", "smk_suite"),
    ("B5_perm", "smk_suite"), ("B6_0p5", "smk_suite"),
    ("A1_mid", "smk_suite"), ("A2_mid", "smk_suite"), ("A3_mid", "smk_suite"), ("A4_mid", "smk_suite"),
    ("A5_mid", "smk_suite"),
    ("S12_30", "smk_suite"), ("D1_1", "smk_suite"), ("T10b", "smk_suite"), ("T12", "smk_suite"),
]
# Identity smokes on full-size rows (CHARTER §6.1.4): (checkpoint key, cell, row, payload). These are grid rows run
# early under their grid names (es1 / kind / c1 then skip them).
IDENTITY_SMOKES = [
    ("C1S", "A2_mid", "suite", "L000"),          # strong fault, full suite: wave 0 gated, waves 1-9 counted
    ("C1S", "S1_perm", "field_light", "L000"),
    ("C1S", "S1_perm", "ledge_down", "L000"),
    ("C1S", "A5_mid", "field_light", "L090"),
    ("c1", "A1_mid", "field_light", "L000"),
]


class PlanError(RuntimeError):
    """A phase that cannot be planned (bad arguments, a missing input): rc 2, no STOP.json."""


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def say(line: str = "") -> None:
    print(line, flush=True)


# ============================================================================== cells.yaml and the rows
def load_cells() -> dict:
    doc = yaml.safe_load(CELLS_YAML.read_text())
    if doc.get("schema") != "fault_cells/v1":
        raise PlanError(f"{CELLS_YAML}: schema {doc.get('schema')!r} != fault_cells/v1")
    return doc


def ab(p) -> str:
    """cells.yaml paths are relative to the omni task dir unless absolute."""
    return str(p if os.path.isabs(str(p)) else OMNI / p)


@dataclasses.dataclass
class Row:
    phase: str                       # the phase that owns the row (where it is budgeted)
    name: str
    ck_key: str
    ck: str
    role: str
    cell: str                        # cell id, NOM or SMK_never
    rowkey: str                      # suite | field_light | ledge_down | pits | smk_suite | smk_field_light
    payload: str                     # L000 | L090
    es: int
    env_file: str
    manifest: str
    scenario: str
    args: list
    level_file: str | None
    kind: str = "grid"               # grid | nominal | smoke (smoke-only, never a grid read)

    @property
    def label(self) -> str:
        return f"{self.name}.{self.scenario.replace('__omni', '')}.es{self.es}"

    @property
    def slot(self) -> tuple:
        return (self.name, self.scenario, self.es)

    def slot_dir(self, tp_out: Path) -> Path:
        return Path(tp_out) / "lr" / self.name / self.scenario / f"seed{self.es:03d}"

    def est_s(self, measured: dict) -> float:
        if self.scenario in measured:
            return measured[self.scenario]
        if self.scenario in FIXED_EST_S:
            return FIXED_EST_S[self.scenario]
        return STARTUP_S + K_S_PER_EP_S * EP_SECONDS[self.scenario]

    def env(self, tp_out: Path) -> dict:
        e = {"TP_OUT": str(tp_out), "TP_ROW_TIMEOUT": str(ROW_TIMEOUT_S)}
        if self.level_file:
            e["TP_LEVEL_FILE"] = self.level_file
        return e

    def argv(self, gate_row: Path) -> list:
        return ["bash", str(gate_row), self.name, self.ck, self.env_file, self.manifest, self.scenario, str(self.es),
                *self.args]


class Planner:
    def __init__(self, doc: dict | None = None):
        self.doc = doc or load_cells()
        self.cells = {c["id"]: c for c in self.doc["cells"]}
        self.smokes = {s["id"]: s for s in self.doc.get("smokes") or []}

    # ---------------------------------------------------------------- one row
    def cell_spec(self, cell_id: str) -> dict:
        if cell_id == "NOM":
            return {"id": "NOM"}
        if cell_id in self.smokes:
            return self.smokes[cell_id]
        if cell_id not in self.cells:
            raise PlanError(f"unknown cell {cell_id!r}")
        return self.cells[cell_id]

    def row(self, phase: str, ck_key: str, cell_id: str, rowkey: str, payload: str, es: int, launch: str | None = None,
            kind: str = "grid") -> Row:
        spec = self.cell_spec(cell_id)
        ckd = self.doc["checkpoints"][ck_key]
        if rowkey in SMOKE_ROWS:
            sr = SMOKE_ROWS[rowkey]
            manifest, scenario, args, env_row = sr["manifest"], sr["scenario"], list(sr["args"]), sr["env_row"]
        else:
            rr = self.doc["rows"][rowkey]
            manifest, scenario, args, env_row = ab(rr["manifest"]), rr["scenario"], list(rr["args"]), rowkey
        args += ["--policy-role", ckd["role"]]
        suffix = "" if payload == "L000" else f"_{payload}"
        env_file = ENVS / f"{cell_id}__{env_row}{suffix}.env"
        if rowkey == "smk_field_light" and (spec.get("fault") or spec.get("env")) and cell_id != "NOM":
            env_file = SMOKE_ENVS / f"{cell_id}__smk_field_light{suffix}.env"
        level = self.doc["levels"].get(payload)
        tag = CK_TAG[ck_key]
        if cell_id == "NOM":
            if not launch:
                raise PlanError("a nominal row needs a launch letter")
            name = f"FS_{tag}_NOM_{payload}_{launch}"
            kind = "nominal" if kind == "grid" else kind
        else:
            name = f"FS_{tag}_{cell_id}_{payload}"
        if rowkey in SMOKE_ROWS:
            name = "FSS" + name[2:]
            kind = "smoke"
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise PlanError(f"bad row name {name!r}")
        return Row(phase, name, ck_key, ab(ckd["path"]), ckd["role"], cell_id, rowkey, payload, int(es), str(env_file),
                   manifest, scenario, args, ab(level) if level else None, kind)

    def cell_payload(self, cell_id: str) -> str:
        return self.cells[cell_id].get("payload", "L000")

    def cell_ck(self, cell_id: str) -> str:
        return self.cells[cell_id].get("checkpoint", "C1S")

    def pair(self, phase, ck_key, cell_id, payload, es, launch=None) -> list:
        return [self.row(phase, ck_key, cell_id, r, payload, es, launch) for r in ("suite", "field_light")]

    # ---------------------------------------------------------------- the phases
    def plan(self, phase: str, *, es2_cells=None, extra=(), results: Path = RESULTS, strict: bool = True,
             tp_out=None) -> list:
        if phase not in PHASES:
            raise PlanError(f"unknown phase {phase!r} (one of {PHASES})")
        return getattr(self, f"plan_{phase}")(es2_cells=es2_cells, extra=extra, results=results, strict=strict,
                                              tp_out=tp_out)

    def plan_smoke(self, **_):
        rows = []
        # 1. nominal A at es1 on every full row the smokes read against (nominals / kind / c1 then skip them)
        for r in ("suite", "field_light"):
            rows.append(self.row("nominals", "C1S", "NOM", r, "L000", 1, "A"))
        for r in ("ledge_down", "pits"):
            rows.append(self.row("kind", "C1S", "NOM", r, "L000", 1, "A"))
        rows.append(self.row("nominals", "C1S", "NOM", "field_light", "L090", 1, "A"))
        rows.append(self.row("c1", "c1", "NOM", "field_light", "L000", 1, "A"))
        # 2. armed-but-never (§6.1.2), full rows, bitwise vs the nominal of its launch class
        for smk in self.smokes.values():
            for r in smk["rows"]:
                rows.append(self.row("smoke", "C1S", smk["id"], r, "L000", 1, kind="smoke"))
        # 3. the short smoke rows' own nominal A (the reference for their pre-onset identity)
        rows.append(self.row("smoke", "C1S", "NOM", "smk_suite", "L000", 1, "A"))
        rows.append(self.row("smoke", "C1S", "NOM", "smk_suite", "L090", 1, "A"))
        rows.append(self.row("smoke", "C1S", "NOM", "smk_field_light", "L000", 1, "A"))
        # 4. a dose smoke per fault id (§6.1.3)
        ids = {c["fault"].split(":")[0] if c.get("fault") else c["id"] for c in self.cells.values()}
        covered = set()
        for cid, rk in DOSE_SMOKES:
            c = self.cells[cid]
            covered.add(c["fault"].split(":")[0] if c.get("fault") else cid)
            rows.append(self.row("smoke", self.cell_ck(cid), cid, rk, self.cell_payload(cid), 1))
        if ids - covered:
            raise PlanError(f"fault ids with no dose smoke: {sorted(ids - covered)}")
        # 5. identity smokes on full rows (§6.1.4), under their grid names
        for ck, cid, r, payload in IDENTITY_SMOKES:
            if self.cell_payload(cid) != payload:
                raise PlanError(f"identity smoke {cid} at {payload} but the cell runs at {self.cell_payload(cid)}")
            owner = "kind" if r in ("ledge_down", "pits") else ("c1" if ck == "c1" else "es1")
            rows.append(self.row(owner, ck, cid, r, payload, 1))
        return rows

    def plan_nominals(self, extra=(), **_):
        rows = []
        for es in (1, 2):
            for launch in ("A", "B"):
                rows += self.pair("nominals", "C1S", "NOM", "L000", es, launch)
        for es in (1, 2):
            rows += self.pair("nominals", "C1S", "NOM", "L090", es, "A")
        for spec in extra:                                   # "<L000|L090>:<launch>[:<es>]" (CHARTER §4: <= 2 pairs)
            parts = spec.split(":")
            if len(parts) not in (2, 3) or parts[0] not in ("L000", "L090") or not re.match(r"^[B-Y]$", parts[1]) \
                    or (len(parts) == 3 and parts[2] not in ("1", "2")):
                raise PlanError(f"--extra {spec!r}: want <L000|L090>:<launch B..Y>[:<1|2>]")
            ess = [int(parts[2])] if len(parts) == 3 else [1, 2]
            for es in ess:
                rows += self.pair("nominals", "C1S", "NOM", parts[0], es, parts[1])
        slots = [r.slot for r in rows]
        dup = sorted({s for s in slots if slots.count(s) > 1})
        if dup:
            raise PlanError(f"--extra repeats a planned nominal launch: {dup}")
        return rows

    def grid_rows(self, phase, cells, es) -> list:
        rows = []
        for cid in cells:
            rows += self.pair(phase, self.cell_ck(cid), cid, self.cell_payload(cid), es)
        return rows

    def plan_es1(self, **_):
        return self.grid_rows("es1", list(self.cells), 1)

    def es2_selection(self, results: Path, tp_out: Path | None = None) -> dict:
        """``{cell: es1 worst-row F}`` from fault_read.py's full read, ``results/read.json`` (schema fault_read/v1:
        ``cell_seeds[]`` with cell, policy, seed, payload and ``read.F_worst``), C1S seed 1 rows. The read must be
        newer than every es1 stamp's tp_row.json (run ``fault_read.py --out-dir results`` after the es1 phase -- a
        CPU step, the parent's), and must give every cell a worst F; otherwise the es2 phase refuses (rc 2)."""
        p = Path(results) / "read.json"
        if not p.is_file():
            raise PlanError(f"{p} does not exist: run `fault_read.py --out-dir {results}` after the es1 phase (or "
                            "pass --es2-cells)")
        doc = json.loads(p.read_text())
        if doc.get("schema") != "fault_read/v1":
            raise PlanError(f"{p}: schema {doc.get('schema')!r} != fault_read/v1")
        if tp_out is not None:
            es1 = self.plan_es1()
            newest = 0.0
            for r in es1:
                for st in complete_stamps(r.slot_dir(tp_out), None):
                    newest = max(newest, (st / "tp_row.json").stat().st_mtime)
            if newest and p.stat().st_mtime < newest:
                raise PlanError(f"{p} is older than the newest es1 stamp: re-run fault_read.py --out-dir {results}")
        f1 = {}
        for s in doc.get("cell_seeds") or []:
            if s.get("policy") == "C1S" and int(s.get("seed", 0)) == 1:
                f1[s["cell"]] = (s.get("read") or {}).get("F_worst")
        missing = [c for c in self.cells if f1.get(c) is None]
        if missing:
            raise PlanError(f"{p}: no es1 worst F for cells {missing} (an es1 row incomplete, failed or unread)")
        return {c: float(f1[c]) for c in self.cells}

    def plan_es2(self, es2_cells=None, results: Path = RESULTS, strict=True, tp_out=None, **_):
        if es2_cells is not None:
            cells = list(es2_cells)
            bad = [c for c in cells if c not in self.cells]
            if bad:
                raise PlanError(f"--es2-cells: unknown cells {bad}")
        else:
            try:
                sel = self.es2_selection(results, tp_out)
            except PlanError:
                if strict:
                    raise
                return None                                   # estimate-only callers fall back to the CHARTER count
            cells = [c for c, f in sel.items() if f < ES2_F_MAX]
        return self.grid_rows("es2", cells, 2)

    def plan_kind(self, **_):
        rows = []
        for es in (1, 2):
            for launch in ("A", "B"):
                for r in ("ledge_down", "pits"):
                    rows.append(self.row("kind", "C1S", "NOM", r, "L000", es, launch))
        for cid, c in self.cells.items():
            if c.get("kind_rows"):
                for es in (1, 2):
                    for r in ("ledge_down", "pits"):
                        rows.append(self.row("kind", self.cell_ck(cid), cid, r, self.cell_payload(cid), es))
        return rows

    def plan_c1(self, **_):
        rows = []
        for nom in self.doc.get("nominals") or []:
            if nom.get("c1"):
                rows += self.pair("c1", "c1", "NOM", nom["payload"], 1, "A")
        for cid, c in self.cells.items():
            if c.get("c1"):
                rows += self.pair("c1", "c1", cid, self.cell_payload(cid), 1)
        return rows

    def plan_closing(self, **_):
        return self.pair("closing", "C1S", "NOM", "L000", 1, "Z")


# ============================================================================== smoke env files
def smoke_env_texts(planner: Planner) -> dict:
    """``{file name: text}`` for manifests/envs_smoke/: each armed smk_field_light dose cell's envs/<cell>__field_light
    file with its KRABBY_FAULT_ONSET line replaced by SMOKE_FL_ONSET (lo unchanged -> the same class horizon)."""
    out = {}
    for cid, rk in DOSE_SMOKES:
        if rk != "smk_field_light":
            continue
        suffix = "" if planner.cell_payload(cid) == "L000" else "_" + planner.cell_payload(cid)
        src = ENVS / f"{cid}__field_light{suffix}.env"
        lines = src.read_text().splitlines()
        n = sum(1 for ln in lines if ln.startswith("KRABBY_FAULT_ONSET="))
        if n != 1:
            raise PlanError(f"{src}: {n} KRABBY_FAULT_ONSET lines (want 1)")
        body = [f"KRABBY_FAULT_ONSET={SMOKE_FL_ONSET}" if ln.startswith("KRABBY_FAULT_ONSET=") else ln
                for ln in lines if ln.startswith("KRABBY_")]
        head = (f"# GENERATED by scripts/fault_queue.py make-smoke-envs from envs/{src.name}: KRABBY_FAULT_ONSET -> "
                f"{SMOKE_FL_ONSET} (smk_field_light__omni is 26 s long). Smoke only; do not edit by hand.")
        out[f"{cid}__smk_field_light{suffix}.env"] = head + "\n" + "".join(f"{ln}\n" for ln in body)
    return out


def cmd_make_smoke_envs(a) -> int:
    files = smoke_env_texts(Planner())
    on_disk = {p.name for p in SMOKE_ENVS.glob("*.env")} if SMOKE_ENVS.is_dir() else set()
    if a.check:
        bad = sorted(set(files) ^ on_disk) + sorted(n for n in set(files) & on_disk
                                                   if (SMOKE_ENVS / n).read_text() != files[n])
        say(f"envs_smoke {'matches' if not bad else 'DIFFERS: ' + str(bad)} ({len(files)} files)")
        return 0 if not bad else 1
    SMOKE_ENVS.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        p = SMOKE_ENVS / n
        if not p.is_file() or p.read_text() != t:
            p.write_text(t)
    say(f"wrote {len(files)} smoke env files under {SMOKE_ENVS}")
    return 0


# ============================================================================== the pin and the guards
def frozen_files() -> dict:
    out = {}
    paths = [CHARTER, CELLS_YAML, *sorted(ENVS.glob("*.env")), *sorted(MANIFESTS.rglob("*.yaml")),
             *sorted(MANIFESTS.rglob("*.env"))]
    for p in paths:
        if p.is_file():
            out[str(p.relative_to(CAMP))] = sha256_file(p)
    return out


def live_build() -> dict:
    """code_sha / eval_sha exactly as gate_row.sh stamps them (run_candidates' own functions), gate_row's dirty rule
    (the code + eval pathspecs, minus run_candidates.code_drop), HEAD."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import run_candidates as r  # noqa: PLC0415

    st = subprocess.run(["git", "-C", str(r.REPO), "status", "--porcelain", "--", *r.CODE_PATHSPECS,
                         *r.EVAL_PATHSPECS], capture_output=True, text=True, check=True).stdout.split("\n")
    dirty = sorted(ln[3:] for ln in st if ln.strip() and not r.code_drop(ln[3:]))
    head = subprocess.run(["git", "-C", str(r.REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    return {"code_sha": r.code_sha(), "eval_sha": r.eval_sha(), "dirty": dirty, "head": head, "repo": str(r.REPO)}


def ddc_gb(path: Path = DDC) -> float:
    if not path.is_dir():
        return 0.0
    r = subprocess.run(["du", "-s", "--block-size=1M", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0]) / 1024.0
    except (IndexError, ValueError):
        return 0.0


def free_gb(path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free / 2 ** 30


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def checkpoint_problems(doc: dict) -> list:
    out = []
    for key, c in doc["checkpoints"].items():
        p = Path(ab(c["path"]))
        if not p.is_file():
            out.append(f"checkpoint {key}: no file {p}")
        elif not sha256_file(p).startswith(c["sha256_prefix"]):
            out.append(f"checkpoint {key}: sha256 of {p} does not start with {c['sha256_prefix']}")
    return out


def cmd_pin(a) -> int:
    pin_path = Path(a.pin_path)
    if pin_path.exists() and not a.repin:
        say(f"PIN refused: {pin_path} exists (a re-pin is a CHARTER §10 amendment: pass --repin)")
        return 2
    live = live_build()
    if live["dirty"] and not a.allow_dirty:
        say(f"PIN refused: the code / eval tree is dirty: {live['dirty']}")
        return 1
    if a.allow_dirty and pin_path.resolve() == (RESULTS / "pin.json").resolve():
        say("PIN refused: --allow-dirty is for a scratch --pin-path only")
        return 2
    doc = load_cells()
    probs = checkpoint_problems(doc)
    if probs and not a.allow_missing_ckpt:
        say("PIN refused: " + "; ".join(probs))
        return 1
    pin = {"schema": "fault_pin/v1", "code_sha": live["code_sha"], "eval_sha": live["eval_sha"], "head": live["head"],
           "repo": live["repo"], "dirty_at_pin": live["dirty"], "pinned_at": now_iso(), "files": frozen_files(),
           "checkpoints": {k: {"path": ab(c["path"]), "sha256_prefix": c["sha256_prefix"]}
                           for k, c in doc["checkpoints"].items()},
           "writer": "scripts/fault_queue.py pin"}
    write_json(pin_path, pin)
    ddc_path = pin_path.parent / "ddc_start.json"
    if not ddc_path.exists() or a.repin:
        write_json(ddc_path, {"ddc_gb": round(ddc_gb(), 3), "path": str(DDC), "at": now_iso()})
    say(f"PIN wrote {pin_path}: code_sha {live['code_sha'][:12]} eval_sha {live['eval_sha'][:12]} head "
        f"{live['head'][:12]} files {len(pin['files'])}")
    return 0


def pin_problems(pin: dict | None, live: dict, files_now: dict) -> tuple[list, list]:
    """(stops, notes) for the live build vs the pin (CHARTER §8: the tree dirty, or the pin moved)."""
    if pin is None:
        return ["no results/pin.json (write it with `fault_queue.py pin` after the code + CHARTER commit)"], []
    stops, notes = [], []
    for k in ("code_sha", "eval_sha"):
        if live[k] != pin.get(k):
            stops.append(f"{k} {live[k][:12]} != pin {str(pin.get(k))[:12]}")
    if live["dirty"]:
        stops.append(f"code / eval tree dirty: {live['dirty'][:8]}{' ...' if len(live['dirty']) > 8 else ''}")
    pf = pin.get("files") or {}
    for f in sorted(set(pf) | set(files_now)):
        if pf.get(f) != files_now.get(f):
            stops.append(f"frozen file {f} {'added' if f not in pf else 'removed' if f not in files_now else 'changed'}"
                         " since the pin")
    if live["head"] != pin.get("head"):
        notes.append(f"HEAD {live['head'][:12]} != pin head {str(pin.get('head'))[:12]} (code_sha / eval_sha equal)")
    return stops, notes


# ============================================================================== the ledger and the projection
class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list:
        if not self.path.is_file():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out

    def append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def spent_s(self) -> float:
        return sum(float(e.get("gpu_s") or 0.0) for e in self.entries())

    def checked_ok(self, stamp: str) -> bool:
        return any(e.get("stamp") == stamp and e.get("status") == "ok" for e in self.entries())

    def measured(self) -> dict:
        """mean gpu_s per scenario over the rows that completed (status ok)."""
        acc: dict = {}
        for e in self.entries():
            if e.get("status") == "ok" and e.get("gpu_s"):
                acc.setdefault(e["scenario"], []).append(float(e["gpu_s"]))
        return {k: sum(v) / len(v) for k, v in acc.items()}


def stamp_complete_at_pin(stamp: Path, pin: dict | None) -> bool:
    tp = read_json(stamp / "tp_row.json")
    if not tp or int(tp.get("rc", -1)) != 0:
        return False
    if pin is None:
        return True
    cs = read_json(stamp / "code_state.json") or {}
    return (cs.get("code_sha") == pin.get("code_sha") and cs.get("eval_sha") == pin.get("eval_sha")
            and str(cs.get("dirty")) == "False")


def complete_stamps(slot: Path, pin: dict | None) -> list:
    if not slot.is_dir():
        return []
    return [d for d in sorted(p for p in slot.iterdir() if p.is_dir()) if stamp_complete_at_pin(d, pin)]


def all_planned(planner: Planner, results: Path, extra=(), es2_cells=None) -> dict:
    """``{phase: rows or None}`` (None = es2 not plannable yet) for the projection; a slot appears once, under the
    first phase that runs it (the smoke phase runs some later-phase rows)."""
    out, seen = {}, set()
    for ph in PHASES:
        rows = planner.plan(ph, extra=extra if ph == "nominals" else (), es2_cells=es2_cells, results=results,
                            strict=False)
        if rows is None:
            out[ph] = None
            continue
        keep = []
        for r in rows:
            if r.slot not in seen:
                seen.add(r.slot)
                keep.append(r)
        out[ph] = keep
    return out


def projection(planner: Planner, ledger: Ledger, tp_out: Path, pin, results: Path, extra=(), es2_cells=None) -> dict:
    measured = ledger.measured()
    plan = all_planned(planner, results, extra, es2_cells)
    pair_s = (Row.est_s(planner.row("es2", "C1S", "NOM", "suite", "L000", 2, "A"), measured)
              + Row.est_s(planner.row("es2", "C1S", "NOM", "field_light", "L000", 2, "A"), measured))
    per = {}
    remaining = 0.0
    for ph, rows in plan.items():
        if rows is None:
            est = ES2_CELLS_ESTIMATE * pair_s
            per[ph] = {"rows": 2 * ES2_CELLS_ESTIMATE, "est_gpu_h": est / 3600, "remaining_gpu_h": est / 3600,
                       "note": f"es2 not selected yet: CHARTER's {ES2_CELLS_ESTIMATE} cells"}
            remaining += est
            continue
        tot = sum(r.est_s(measured) for r in rows)
        rem = sum(r.est_s(measured) for r in rows if not complete_stamps(r.slot_dir(tp_out), pin))
        per[ph] = {"rows": len(rows), "est_gpu_h": tot / 3600, "remaining_gpu_h": rem / 3600}
        remaining += rem
    spent = ledger.spent_s()
    return {"spent_gpu_h": spent / 3600, "remaining_gpu_h": remaining / 3600,
            "projected_gpu_h": (spent + remaining) / 3600, "per_phase": per, "measured_s": measured}


# ============================================================================== the runner
def isaac_patterns(gate_row: Path = GATE_ROW) -> tuple[str, str]:
    """gate_row.sh's own (ISAAC_PAT, NONKIT_PAT): the queue waits on exactly what gate_row waits on, BEFORE its
    timer starts (so a wait is never charged to the ledger). Read from the real gate_row even with a fake launcher."""
    text = gate_row.read_text()
    pats = []
    for var in ("ISAAC_PAT", "NONKIT_PAT"):
        m = re.search(r"^%s='([^']*)'\s*$" % var, text, re.M)
        if not m:
            raise PlanError(f"{gate_row}: no {var}")
        pats.append(m.group(1))
    return pats[0], pats[1]


def busy(pats) -> list:
    isaac, nonkit = pats
    p = subprocess.run(["ps", "-ww", "-eo", "pid=,args="], capture_output=True, text=True)
    me = os.getpid()
    out = []
    for ln in p.stdout.splitlines():
        f = ln.strip().split(None, 1)
        if len(f) == 2 and f[0].isdigit() and int(f[0]) != me and re.search(isaac, f[1]) and not re.search(nonkit, f[1]):
            out.append(f[1][:160])
    return out


def child_env(extra: dict, cpu: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_", "FS_"))}
    env.pop("PYTHONPATH", None)
    env.update(extra)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif env.get("CUDA_VISIBLE_DEVICES") == "":
        del env["CUDA_VISIBLE_DEVICES"]
    return env


def run_child(argv: list, env: dict, log: Path | None = None, echo=re.compile(r"^(TPROW END|FAULT_READ|FAULTCHECK)")) -> int:
    fh = open(log, "a") if log else None
    try:
        if fh:
            fh.write(f"### {now_iso()} {' '.join(map(str, argv))}\n")
            fh.flush()
        p = subprocess.Popen([str(x) for x in argv], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, text=True, bufsize=1, cwd=str(W))
        for line in p.stdout:
            if fh:
                fh.write(line)
            if fh is None or echo.match(line):
                sys.stdout.write("  " + line)
                sys.stdout.flush()
        return p.wait()
    finally:
        if fh:
            fh.close()


class Stop(Exception):
    def __init__(self, reasons, row=None):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)
        self.row = row


def shell_line(row: Row, gate_row: Path, tp_out: Path) -> str:
    env = " ".join(f"{k}={v}" for k, v in row.env(tp_out).items())
    return f"{env} {' '.join(row.argv(gate_row))}"


def cmd_list(a) -> int:
    planner = Planner()
    rows = planner.plan(a.phase, extra=a.extra, es2_cells=a.es2_cells, results=Path(a.results))
    measured = Ledger(Path(a.results) / "queue_ledger.jsonl").measured()
    tot = 0.0
    for r in rows:
        tot += r.est_s(measured)
        say(f"{r.label:58s} {r.kind:8s} owner={r.phase:9s} {r.ck_key:15s} {r.payload} est={r.est_s(measured):5.0f}s")
    say(f"# {len(rows)} rows, est {tot / 3600:.2f} GPU-h (rows owned by a later phase included)")
    return 0


def cmd_estimate(a) -> int:
    planner = Planner()
    results = Path(a.results)
    ledger = Ledger(results / "queue_ledger.jsonl")
    pin = read_json(Path(a.pin_path))
    pr = projection(planner, ledger, Path(a.tp_out), pin, results, a.extra, a.es2_cells)
    for ph, v in pr["per_phase"].items():
        say(f"{ph:9s} rows={v['rows']:4d} est={v['est_gpu_h']:5.2f} GPU-h remaining={v['remaining_gpu_h']:5.2f}"
            + (f"  ({v['note']})" if v.get("note") else ""))
    say(f"spent={pr['spent_gpu_h']:.2f} remaining={pr['remaining_gpu_h']:.2f} projected={pr['projected_gpu_h']:.2f} "
        f"cap={BUDGET_GPU_H}")
    return 0


def cmd_run(a) -> int:
    phase = a.phase
    st = {"stopped": False}
    try:
        return run_phase(a, st)
    except PlanError as exc:
        say(f"QUEUE PLAN REFUSED: {exc}")
        say(f"QUEUE END status=stop phase={phase} (plan refused; nothing launched)")
        return 2
    except (Exception, KeyboardInterrupt) as exc:   # any crash still ends the log with the sentinel
        why = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
        try:
            write_stop(Path(a.results), phase, None, [f"queue crashed: {why}"], dry=a.dry_run)
        finally:
            say(f"QUEUE END status=stop phase={phase} (queue crashed: {why})")
        return 1


def write_stop(results: Path, phase: str, row: Row | None, reasons: list, dry: bool = False, extra=None) -> None:
    if dry:
        return
    doc = {"schema": "fault_queue_stop/v1", "phase": phase, "row": row.label if row else None,
           "stamp_slot": list(row.slot) if row else None, "reasons": reasons, "at": now_iso(),
           "cleared_by": "hand only (the parent / user): delete this file to let the queue run again"}
    if extra:
        doc.update(extra)
    write_json(results / "STOP.json", doc)


def run_phase(a, st) -> int:
    phase = a.phase
    results = Path(a.results)
    tp_out = Path(a.tp_out)
    gate_row = Path(a.launcher or GATE_ROW)
    fault_read = Path(a.fault_read or FAULT_READ)
    test_mode = os.environ.get("FS_QUEUE_TEST") == "1"
    if test_mode and gate_row == GATE_ROW:
        raise PlanError("FS_QUEUE_TEST=1 is for a fake --launcher only")
    planner = Planner()
    rows = planner.plan(phase, extra=a.extra, es2_cells=a.es2_cells, results=results, tp_out=tp_out)
    if a.only:
        rows = [r for r in rows if r.label in set(a.only) or r.name in set(a.only)]
    ledger = Ledger(results / "queue_ledger.jsonl")
    pin_path = Path(a.pin_path)
    pin = read_json(pin_path)
    say(f"QUEUE START phase={phase} rows={len(rows)} dry={int(a.dry_run)} tp_out={tp_out} launcher={gate_row} "
        f"pin={pin_path if pin else 'NONE'} at={now_iso()} pid={os.getpid()}")

    if a.dry_run:
        live = live_build()
        stops, notes = pin_problems(pin, live, frozen_files())
        for s in stops + notes:
            say(f"# (dry) pin check: {s}")
        for p in checkpoint_problems(planner.doc):
            say(f"# (dry) {p}")
        pr = projection(planner, ledger, tp_out, pin, results, a.extra, a.es2_cells)
        say(f"# (dry) projection {pr['projected_gpu_h']:.2f} GPU-h (spent {pr['spent_gpu_h']:.2f}) cap {BUDGET_GPU_H}")
        missing = [r.env_file for r in rows if not Path(r.env_file).is_file()]
        for m in sorted(set(missing)):
            say(f"# (dry) MISSING env file {m}")
        measured = ledger.measured()
        for i, r in enumerate(rows, 1):
            done = complete_stamps(r.slot_dir(tp_out), pin)
            tag = f"SKIP (complete: {done[-1].name})" if done else f"est {r.est_s(measured):.0f}s"
            say(f"[{i:3d}] {r.label}  <{r.kind}, owner {r.phase}>  {tag}")
            say(f"      {shell_line(r, gate_row, tp_out)}")
            say(f"      then: {PY} -B {fault_read} --root {tp_out / 'lr'} --check-only <stamp>")
            if a.dry_gate:
                rc = run_child(r.argv(gate_row), child_env({**r.env(tp_out), "TP_ROW_DRY": "1"}, cpu=True))
                if rc != 0:
                    say(f"      gate_row TP_ROW_DRY rc={rc}")
        say(f"QUEUE END status=ok phase={phase} (dry: nothing launched)")
        return 0

    stop_json = results / "STOP.json"
    if stop_json.exists():
        say(f"QUEUE REFUSED: {stop_json} exists ({(read_json(stop_json) or {}).get('reasons')}); clear it by hand")
        say(f"QUEUE END status=stop phase={phase} (STOP.json present; nothing launched)")
        return 1
    lock_path = results / ".queue.lock"
    results.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        say(f"QUEUE REFUSED: another queue holds {lock_path}")
        say(f"QUEUE END status=stop phase={phase} (another queue runs; nothing launched)")
        return 3
    ddc_path = results / "ddc_start.json"
    if not ddc_path.exists():
        write_json(ddc_path, {"ddc_gb": round(ddc_gb(), 3), "path": str(DDC), "at": now_iso(),
                              "note": "recorded by the first queue run (no pin-time record)"})
    ddc0 = float((read_json(ddc_path) or {}).get("ddc_gb", 0.0))
    probs = checkpoint_problems(planner.doc)
    used = {r.ck_key for r in rows}
    probs = [p for p in probs if p.split(":")[0].split()[-1] in used]
    if not fault_read.is_file():
        probs.append(f"no {fault_read}: no row can be verified after it runs (CHARTER §6.2), so none is launched")
    for r in rows:
        if not Path(r.env_file).is_file():
            probs.append(f"no env file {r.env_file} (row {r.label})")
    if probs:
        return do_stop(results, phase, None, probs)
    pats = isaac_patterns()
    logd = tp_out / "logs" / "queue_rows"
    logd.mkdir(parents=True, exist_ok=True)

    n_run = n_skip = n_relaunch = 0
    fr_root = tp_out / "lr"

    def unmatched_stamps(label: str) -> set:
        return {e["stamp"] for e in ledger.entries() if e.get("row") == label and e.get("status") == "class_unmatched"}

    def after_check(r: Row, stamp: str, rc_c: int):
        """-> None (row ok), "relaunch", or a stop rc. fault_read --check-only: 0 ok, 2 launch error, 3 class
        unmatched (CHARTER §5.1: relaunch once; still unmatched -> stop), anything else (4 usage, 127 missing) stop."""
        clog = logd / f"{r.label}.check.log"
        if rc_c == 0:
            return None
        if rc_c == 3:
            seen = unmatched_stamps(r.label)
            if len(seen) < 2:
                say(f"# [{r.label}] class unmatched on {stamp}: relaunching once (CHARTER §5.1)")
                return "relaunch"
            return do_stop(results, phase, r, [f"class unmatched after one relaunch (CHARTER §5.1): stamps "
                                               f"{sorted(seen)} (log {clog})"])
        what = {2: "launch error", 127: "fault_read.py missing"}.get(rc_c, "fault_read failure")
        return do_stop(results, phase, r, [f"{what}: fault_read --check-only rc={rc_c} on {stamp} (log {clog})"])

    def status_of(rc_c: int) -> str:
        return {0: "ok", 3: "class_unmatched"}.get(rc_c, "check_fail")

    for i, r in enumerate(rows, 1):
        if (results / "STOP_QUEUE").exists():
            return do_stop(results, phase, r, [f"{results / 'STOP_QUEUE'} present (a requested stop)"])
        done = complete_stamps(r.slot_dir(tp_out), pin)
        relaunch = False
        if done:
            stamp = str(done[-1])
            if ledger.checked_ok(stamp):
                say(f"QROW {r.label} SKIP (complete at the pin: {stamp})")
                n_skip += 1
                continue
            prior = [e for e in ledger.entries() if e.get("stamp") == stamp and e.get("status") == "class_unmatched"]
            if prior:
                rc_c = 3
                say(f"# [{r.label}] newest stamp {stamp} was recorded class-unmatched")
            else:
                say(f"# [{r.label}] complete stamp {stamp} has no check in the ledger: checking it now")
                rc_c = run_check(fault_read, stamp, logd / f"{r.label}.check.log", fr_root)
                ledger.append({"row": r.label, "phase": phase, "name": r.name, "scenario": r.scenario, "es": r.es,
                               "stamp": stamp, "start": None, "end": now_iso(), "gpu_s": 0.0,
                               "status": status_of(rc_c), "check_rc": rc_c, "note": "pre-existing stamp"})
            res = after_check(r, stamp, rc_c)
            if res is None:
                say(f"QROW {r.label} SKIP (complete at the pin, checked now: {stamp})")
                n_skip += 1
                continue
            if res != "relaunch":
                return res
            relaunch = True
        while True:
            # ---- the stop rules checkable before a row (CHARTER §8)
            reasons = []
            live = live_build()
            stops, notes = pin_problems(pin, live, frozen_files())
            if test_mode:
                notes += [f"(FS_QUEUE_TEST) not a stop: {s}" for s in stops]
                stops = []
            reasons += stops
            for n in notes:
                say(f"# [{r.label}] {n}")
            for p in {tp_out, W}:
                f = free_gb(p)
                if f < FREE_MIN_GB:
                    reasons.append(f"free disk {f:.0f} GB < {FREE_MIN_GB:g} GB at {p}")
            d = ddc_gb()
            if d - ddc0 > DDC_GROWTH_MAX_GB:
                reasons.append(f"Kit DerivedDataCache grew {d - ddc0:.1f} GB > {DDC_GROWTH_MAX_GB:g} GB since "
                               f"{ddc_path} ({ddc0:.1f} -> {d:.1f} GB)")
            pr = projection(planner, ledger, tp_out, pin, results, a.extra, a.es2_cells)
            proj = pr["projected_gpu_h"] + (r.est_s(ledger.measured()) / 3600 if relaunch else 0.0)
            if proj > BUDGET_GPU_H:
                reasons.append(f"GPU-h projection {proj:.2f} > {BUDGET_GPU_H:g} (spent {pr['spent_gpu_h']:.2f})")
            if reasons:
                return do_stop(results, phase, r, reasons, {"projection": pr})
            # ---- wait for the GPU (gate_row would wait too; waiting here keeps the wait off the ledger)
            waited = 0
            while True:
                b = busy(pats)
                if not b:
                    break
                if waited % 600 == 0:
                    say(f"# [{r.label}] waiting: an Isaac python runs: {b[0]}")
                time.sleep(BUSY_POLL_S)
                waited += BUSY_POLL_S
            say(f"QROW {r.label} START{' RELAUNCH' if relaunch else ''} ({i}/{len(rows)}, {r.kind}, owner {r.phase}) "
                f"projected {proj:.2f} GPU-h at={now_iso()}")
            before = {str(s) for s in complete_stamps(r.slot_dir(tp_out), pin)}
            t0 = time.time()
            start = now_iso()
            rc = run_child(r.argv(gate_row), child_env(r.env(tp_out), cpu=False), logd / f"{r.label}.log")
            gpu_s = time.time() - t0
            new = [str(s) for s in complete_stamps(r.slot_dir(tp_out), pin) if str(s) not in before]
            stamp = new[-1] if new else None
            entry = {"row": r.label, "phase": phase, "name": r.name, "scenario": r.scenario, "es": r.es,
                     "stamp": stamp, "start": start, "end": now_iso(), "gpu_s": round(gpu_s, 1), "gate_rc": rc,
                     "relaunch": relaunch}
            n_run += 1
            n_relaunch += int(relaunch)
            if rc != 0 or not stamp:
                entry["status"] = "launch_error"
                ledger.append(entry)
                why = (f"gate_row rc={rc}" if rc != 0 else "gate_row rc 0 but no new complete stamp at the pin") + \
                      f" (log {logd / (r.label + '.log')})"
                return do_stop(results, phase, r, [f"launch error: {why}"])
            rc_c = run_check(fault_read, stamp, logd / f"{r.label}.check.log", fr_root)
            entry["check_rc"] = rc_c
            entry["status"] = status_of(rc_c)
            ledger.append(entry)
            res = after_check(r, stamp, rc_c)
            if res is None:
                say(f"QROW {r.label} OK {gpu_s:.0f}s stamp={stamp}")
                break
            if res != "relaunch":
                return res
            relaunch = True
    pr = projection(planner, ledger, tp_out, pin, results, a.extra, a.es2_cells)
    say(f"QUEUE END status=ok phase={phase} ran={n_run} relaunched={n_relaunch} skipped={n_skip} "
        f"spent={pr['spent_gpu_h']:.2f} projected={pr['projected_gpu_h']:.2f} GPU-h")
    return 0


def run_check(fault_read: Path, stamp: str, log: Path, root: Path) -> int:
    if not fault_read.is_file():
        with open(log, "a") as fh:
            fh.write(f"### {now_iso()} no {fault_read}\n")
        say(f"  FAULT_READ missing: {fault_read} (a row cannot be verified -> a launch error)")
        return 127
    return run_child([PY, "-B", str(fault_read), "--root", str(root), "--check-only", stamp],
                     child_env({}, cpu=True), log)


def do_stop(results: Path, phase: str, row: Row | None, reasons: list, extra=None) -> int:
    write_stop(results, phase, row, reasons, extra=extra)
    for rs in reasons:
        say(f"QUEUE STOP {row.label if row else '-'}: {rs}")
    say(f"QUEUE END status=stop phase={phase} row={row.label if row else '-'} (see {results / 'STOP.json'})")
    return 1


# ============================================================================== entry
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--results", default=str(RESULTS), help="results dir (ledger, STOP.json, read.json)")
        p.add_argument("--pin-path", default=None)
        p.add_argument("--tp-out", default=str(TP_OUT_DEFAULT))
        p.add_argument("--extra", action="append", default=[], help="nominals: extra launch <L000|L090>:<C..Y>[:<es>]")
        p.add_argument("--es2-cells", default=None, type=lambda s: [x for x in s.split(",") if x],
                       help="es2: explicit cell list (bypasses results/read.json)")

    p = sub.add_parser("pin")
    p.add_argument("--pin-path", default=str(RESULTS / "pin.json"))
    p.add_argument("--repin", action="store_true")
    p.add_argument("--allow-dirty", action="store_true", help="test only, with a scratch --pin-path")
    p.add_argument("--allow-missing-ckpt", action="store_true", help="test only")
    p = sub.add_parser("make-smoke-envs")
    p.add_argument("--check", action="store_true")
    for name in ("list", "run"):
        p = sub.add_parser(name)
        p.add_argument("--phase", required=True, choices=PHASES)
        common(p)
        if name == "run":
            p.add_argument("--dry-run", action="store_true", help="print the gate_row commands; launch nothing")
            p.add_argument("--dry-gate", action="store_true", help="with --dry-run: also run gate_row TP_ROW_DRY=1")
            p.add_argument("--only", action="append", default=None, help="run only these labels / names")
            p.add_argument("--launcher", default=None, help="a fake gate_row (tests; with FS_QUEUE_TEST=1)")
            p.add_argument("--fault-read", default=None, help="a fake fault_read.py (tests)")
    p = sub.add_parser("estimate")
    common(p)
    a = ap.parse_args(argv)
    if getattr(a, "pin_path", None) is None and a.cmd != "pin" and a.cmd != "make-smoke-envs":
        a.pin_path = str(Path(a.results) / "pin.json")
    try:
        if a.cmd == "pin":
            return cmd_pin(a)
        if a.cmd == "make-smoke-envs":
            return cmd_make_smoke_envs(a)
        if a.cmd == "list":
            return cmd_list(a)
        if a.cmd == "estimate":
            return cmd_estimate(a)
    except PlanError as exc:
        say(f"QUEUE PLAN REFUSED: {exc}")
        return 2
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
