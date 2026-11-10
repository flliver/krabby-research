"""DR stress sweep: the campaign reader's and driver's decision logic, on SYNTHETIC records (no stamps, no Isaac).

``experiments/2026-11-06_dr_stress_sweep/scripts/{cells,stress_read,run_sweep}.py`` turn ~90 cell-seeds of GPU time into
the verdict table, and CHARTER sections 3.3 (verdicts, deltas, launch errors) and 3.5 (stop rules) are their spec. A wrong
decision here is silent -- a launch error read as a failure, a split read as a pass, a deviation read as a stop, a stale
closing read as a match -- and it is only found after the queue has run, so each rule is pinned against hand-built records:

* cells: the CHARTER table as data (levels, bands, bases, nominals), 55 es1 cells / 21 band edges, gate names, the env
  and level files a cell gets, the keys its stamps MUST carry (the launch-error check), the stress values parse under the
  knob parsers;
* launch errors: ``stamp_problems`` (missing / different knob key, stray ``KRABBY_STRESS_*``, ``zero_depth`` polarity,
  dirty tree, hashes off the pin, NO pin at all) on tiny fake stamp dirs, and ``cellseed`` turning one into ``pass = None``;
* ``read_stamps`` against a fake line reader: es-K lines only, 18 lines to pass, the lenient (>= 0.5) flag;
* verdicts: ``level_status`` / ``level_verdict`` (a band edge needs BOTH seeds; a non-edge in-band level needs es2 only
  once its es1 fails; a launch error is never a failure), ``term_table`` (worst over in-band levels, per side, out-of-band
  levels never decide, bases combined worst-first, report-only terms get no verdict, tolerated and lenient edges, the
  references used, the terms 13-15 feasibility bound, the closing flag and its staleness);
* ``es2_plan`` (edges, in-band es1 failures, the terms 13-15 transition search, never a launch-error cell) and ``deltas``
  (strict ``> 0.05`` drop against EVERY nominal launch, survival / completion only, the 135 kg term never flags);
* the stop rules: ``baseline`` (failed line or launch error stops; completion stops only when EVERY launch of a base
  deviates; a split between the two 0 kg launches is a record, not a stop), ``closing_check`` (status, launch, staleness;
  a mismatch is a flag, never a stop) with its dynamic Z..Z9 launches, the driver's tree check before every row, valid
  stamps skipped / invalid ones re-run, env scrubbing, STOP_QUEUE and signals, the baseline gate for ``--from``, the
  zero-depth copy verified by sha256, and the CLI exit codes.

Nothing real is read: the three scripts are path-loaded under private names (they import each other by plain name, which
is aliased only while they load), the stamp directory is ``tmp_path``, and the heavy readers (``payload_read`` /
``thresholds``, numpy) are replaced by fakes.
"""

from __future__ import annotations

import collections
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour" / "parkour_tasks" / "parkour_tasks"
CAMPAIGN_DIR = TASKS / "crab_hex_omni_task" / "experiments" / "2026-11-06_dr_stress_sweep"
SCRIPTS = CAMPAIGN_DIR / "scripts"
CHARTER = CAMPAIGN_DIR / "CHARTER.md"
STRESS_KNOBS_PY = TASKS / "crab_hex_omni_task" / "config" / "crab_hex" / "stress_knobs.py"

if not (SCRIPTS / "stress_read.py").exists():
    pytest.skip("the 2026-11-06 DR stress sweep campaign is not in this checkout", allow_module_level=True)


def _exec(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_campaign():
    """``cells`` -> ``stress_read`` -> ``run_sweep`` under private names.

    They import each other by plain name and put their own directory on ``sys.path``, so the plain names are aliased
    only while they load; ``sys.path``, ``sys.modules`` and ``sys.dont_write_bytecode`` are restored afterwards."""
    saved_path, saved_dwb = list(sys.path), sys.dont_write_bytecode
    saved_mods = {n: sys.modules.get(n) for n in ("cells", "stress_read", "run_sweep")}
    try:
        cells = _exec(SCRIPTS / "cells.py", "drs_cells__reader_test")
        sys.modules["cells"] = cells
        stress_read = _exec(SCRIPTS / "stress_read.py", "drs_stress_read__reader_test")
        sys.modules["stress_read"] = stress_read
        run_sweep = _exec(SCRIPTS / "run_sweep.py", "drs_run_sweep__reader_test")
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_dwb
        for name, mod in saved_mods.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return cells, stress_read, run_sweep


C, SR, RS = _load_campaign()
SK = _exec(STRESS_KNOBS_PY, "stress_knobs__reader_test")
T = C.TERM


# =============================================================================== synthetic records
NO = object()                                    # "this seed was not read"
LINE_KEYS = (
    "survival.fwd", "survival.side", "survival.back", "survival.diag", "survival.turn", "survival.move_turn",
    "tracking.fwd", "tracking.side", "tracking.back", "tracking.diag", "turn_fc.turn",
    "h_meas.suite", "h_meas.field_light", "roll_rms.suite", "roll_rms.field_light",
    "pitch_rms.suite", "pitch_rms.field_light", "field_light_completion",
)
NOMINAL_VALUES = {"survival": 0.95, "tracking": 0.10, "turn_fc": 0.90, "h_meas": 0.90, "roll_rms": 0.05,
                  "pitch_rms": 0.05, "field_light_completion": 0.99}


def _lines(over=None) -> dict:
    out = {k: NOMINAL_VALUES[k.split(".")[0]] for k in LINE_KEYS}
    out.update(over or {})
    return out


def rec(passed=True, over=None, *, lenient=True, stamps=None) -> dict:
    """A cell-seed record shaped like ``cellseed()`` output; ``passed`` None = a launch error."""
    out = {"pass": passed, "lines": {k: {"value": v} for k, v in _lines(over).items()}, "lenient_feasible": lenient,
           "launch_error": ["synthetic launch error"] if passed is None else None,
           "failed": [] if passed else ["survival.fwd"]}
    if stamps is not None:
        out["stamps"] = stamps
    return out


def S(e1=NO, e2=NO) -> dict:
    """``{es: rec}`` of the seeds that were read; each arg is True / False / None (launch error) or a ready record."""
    return {es: (v if isinstance(v, dict) else rec(v)) for es, v in ((1, e1), (2, e2)) if v is not NO}


def cid(term: str, level, base: str = "L000") -> str:
    return C.cell_id(T[term], level, base)


def make_reads(fills=None) -> dict:
    """Every grid cell unread, then ``{cell_id: seeds}`` filled in -- the shape of ``grid_reads()``."""
    out = {c["id"]: {"cell": c, "seeds": {}} for c in C.cells()}
    for cell_id, seeds in (fills or {}).items():
        out[cell_id]["seeds"] = seeds
    return out


def row(table: dict, term: str, base: str | None = None) -> dict:
    hit = [r for r in table["rows"] if r["term"] == term and (base is None or r["base"] == base)]
    assert len(hit) == 1, (term, base, [(r["term"], r["base"]) for r in hit])
    return hit[0]


@pytest.fixture
def quiet(monkeypatch):
    """No nominal references and no closing record: ``term_table`` on the records alone."""
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [])
    monkeypatch.setattr(SR, "closing_record", lambda: None)


# =============================================================================== cells
#: CHARTER 3.2 as data: ``term -> (knob variable, levels, band, bases)``.
CHARTER_TABLE = {
    "T01": ("KRABBY_STRESS_ROD_SPEED", [0.5, 0.7, 0.8, 1.15, 1.35], (0.7, 1.35), ("L000", "L090")),
    "T02": ("KRABBY_STRESS_ROD_EFFORT", [0.5, 0.7, 0.85], (0.85, 1.0), ("L000", "L090")),
    "T03": ("KRABBY_STRESS_CAM", [0.5, 0.8, 1.2], (0.8, 1.2), ("L000",)),
    "T04": ("KRABBY_STRESS_KPKD", [0.5, 0.75, 1.25], (0.75, 1.25), ("L000",)),
    "T05": ("KRABBY_STRESS_LEG_MASS", [0.9, 1.15, 1.5], (0.9, 1.15), ("L000",)),
    "T06": ("KRABBY_STRESS_MU", [0.3, 0.4, 0.8, 1.2], (0.4, 1.2), ("L000",)),
    "T07": ("KRABBY_STRESS_ACT_DELAY", [2, 3, 4], (1, 2), ("L000",)),
    "T08": ("KRABBY_STRESS_OBS_NOISE", [0.5, 1.0, 2.0, 4.0], (0.0, 1.0), ("L000",)),
    "T09a": ("KRABBY_STRESS_CUR_SCALE", [0.85, 1.15], (0.85, 1.15), ("L000",)),
    "T09b": ("KRABBY_STRESS_CUR_NOISE", [0.1, 0.25], (0.0, 0.1), ("L000",)),
    "T10a": ("KRABBY_STRESS_VXY_NOISE", [0.05, 0.1, 0.2], (0.0, 0.1), ("L000",)),
    "T10b": ("KRABBY_STRESS_VXY_ZERO", [1], None, ("L000",)),
    "T11": ("KRABBY_STRESS_DEPTH_LAG", [2, 3], (1, 2), ("L000",)),
    "T12": (None, [1], None, ("L000",)),
    "T13": (None, [150.0, 165.0], None, ("payload",)),
    "T14": (None, [0.2, 0.3], None, ("L090",)),
    "T15": ("KRABBY_STRESS_PUSH_DV", [0.1, 0.2, 0.35, 0.5], None, ("L000",)),
}
NOMINALS = {"T01": 1.0, "T02": 1.0, "T03": 1.0, "T04": 1.0, "T05": 1.0, "T06": 1.0, "T07": 1, "T08": 0.0, "T09a": 1.0,
            "T09b": 0.0, "T10a": 0.0, "T10b": 0, "T11": 1, "T12": 0, "T13": 135.0, "T14": 0.0, "T15": 0.0}
KINDS = {"T12": "zero_depth", "T13": "payload_kg", "T14": "payload_com"}


def test_the_term_table_is_the_charter_table():
    assert [t["term"] for t in C.TERMS] == list(CHARTER_TABLE)
    for t in C.TERMS:
        var, levels, band, bases = CHARTER_TABLE[t["term"]]
        assert (t["var"], t["levels"], t["band"], t["bases"]) == (var, levels, band, bases), t["term"]
        assert t["nominal"] == NOMINALS[t["term"]] and t["kind"] == KINDS.get(t["term"], "stress"), t["term"]
    assert T["T01"]["band"] == (0.7, 1.35) and 1.35 in T["T01"]["levels"]       # the 2026-11-06 amendment


def test_the_charter_text_names_the_same_knobs_and_counts():
    text = CHARTER.read_text()
    named = set(re.findall(r"`STRESS_([A-Z_]+)`", text))
    assert named == {v.removeprefix("KRABBY_STRESS_") for v in SK.STRESS_VARS}
    cells = C.cells()
    by_base = collections.Counter(c["base"] for c in cells)
    # a reworded sentence FAILS here (it used to skip, which hid the cross-check for a whole round): update the pattern
    m = re.search(r"es1 grid:\*\* (\d+) cells \((\d+) at 0 kg, the (\d+) term-13 cells at ([\d. /]+) kg, (\d+) at L090", text)
    assert m is not None, "the CHARTER no longer words the es1 grid split this way"
    total, zero_kg, payload, payload_kg, l090 = m.groups()
    assert (int(total), int(zero_kg), int(payload), int(l090)) == (
        len(cells), by_base["L000"], by_base["payload"], by_base["L090"])
    assert [float(x) for x in payload_kg.split("/")] == T["T13"]["levels"]
    m = re.search(r"every band-edge level \((\d+)\)", text)
    assert m is not None and int(m.group(1)) == sum(1 for c in cells if c["level"] in C.band_edges(T[c["term"]]))
    m = re.search(r"Terms 1 and 2 also run at L090 \(90 kg payload\): (\d+) cells", text)
    assert m is not None and int(m.group(1)) == sum(
        1 for c in cells if c["base"] == "L090" and c["term"] in ("T01", "T02"))


def test_the_grid_is_55_es1_cells_with_21_band_edges():
    cells = C.cells()
    assert len(cells) == 55 and len({c["id"] for c in cells}) == 55
    assert collections.Counter(c["base"] for c in cells) == {"L000": 43, "L090": 10, "payload": 2}
    edges = [c for c in cells if c["level"] in C.band_edges(T[c["term"]])]
    assert len(edges) == 21
    assert collections.Counter(c["term"] for c in edges) == {
        "T01": 4, "T02": 2, "T03": 2, "T04": 2, "T05": 2, "T06": 2, "T07": 1, "T08": 1, "T09a": 2, "T09b": 1, "T10a": 1,
        "T11": 1}
    assert [c["id"] for c in cells][:3] == ["T01_rod_speed_0p5_L000", "T01_rod_speed_0p7_L000", "T01_rod_speed_0p8_L000"]
    assert C.band_edges(T["T08"]) == [1.0] and C.band_edges(T["T01"]) == [0.7, 1.35] and C.band_edges(T["T12"]) == []
    assert C.band_edges(T["T07"]) == [2] and C.band_edges(T["T15"]) == []     # a band end equal to the nominal is no level


def test_band_edges_exclude_a_level_equal_to_the_nominal():
    # no term of the table lists its own nominal as a level, so the docstring's "excluding the nominal" is pinned on a
    # synthetic term whose levels do include it
    term = {"band": (0.0, 1.0), "levels": [0.0, 0.5, 1.0], "nominal": 0.0}
    assert C.band_edges(term) == [1.0]
    assert C.band_edges({**term, "nominal": 1.0}) == [0.0]
    assert C.band_edges({**term, "nominal": 0.5}) == [0.0, 1.0]
    assert C.band_edges({**term, "band": None}) == []


def test_inside_band_and_side():
    t = T["T08"]
    assert [C.inside_band(t, lv) for lv in (0.5, 1.0, 2.0, 4.0)] == [True, True, False, False]
    assert C.inside_band(t, 1.0 + 1e-13) and not C.inside_band(t, 1.0 + 1e-9)    # the band ends are inclusive to 1e-12
    assert not any(C.inside_band(T[x], lv) for x in ("T10b", "T12", "T13", "T14", "T15") for lv in T[x]["levels"])
    assert [C.side(T["T01"], lv) for lv in T["T01"]["levels"]] == ["low", "low", "low", "high", "high"]
    assert {C.side(T[x], lv) for x in ("T07", "T08", "T13", "T15") for lv in T[x]["levels"]} == {"high"}


def test_gate_names_are_unique_and_legal_for_gate_row():
    cells = C.cells()
    names = [C.gate_name(c["id"]) for c in cells]
    names += [C.gate_name(C.nominal_id(b, la)) for b in ("L000", "L090") for la in ("A", "B", *RS.CLOSING_LAUNCHES)]
    names += [C.gate_name(c["id"], "teacher") for c in cells]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names)                 # gate_row.sh: NAME must be [A-Za-z0-9_]+
    assert C.gate_name("T01_x") == "DRS_T01_x" and C.gate_name("T01_x", "teacher") == "DRST_T01_x"
    assert C.nominal_id("L090", "B") == "T00_nominal_L090_B"
    assert [C.fmt(v) for v in (0.7, 2, 150.0, 1.15, 0.05, -0.5, 1e-3)] == ["0p7", "2", "150", "1p15", "0p05", "m0p5", "0p001"]


def test_every_stress_cell_value_parses_under_its_knob_parser():
    seen = []
    for t in C.TERMS:
        if t["var"] is None:
            continue
        seen.append(t["var"])
        for lv in t["levels"]:
            text = C.stress_value(t, lv)
            parsed = SK.parse_stress_knobs({t["var"]: text})
            assert list(parsed) == [t["var"]], (t["term"], lv, text)
            assert parsed[t["var"]] == (True if t["var"] == "KRABBY_STRESS_VXY_ZERO" else float(lv)), (t["term"], lv)
    assert sorted(seen) == sorted(SK.STRESS_VARS)                                # every knob has exactly one term
    t = T
    assert C.stress_value(t["T10b"], 1) == "1"
    assert C.stress_value(t["T07"], 2) == "2" and C.stress_value(t["T11"], 3) == "3"
    assert [C.stress_value(t["T01"], v) for v in (0.7, 1.15, 1.35)] == ["0.7", "1.15", "1.35"]
    assert C.stress_value(t["T02"], 1.0) == "1" and C.stress_value(t["T15"], 0.35) == "0.35"


@pytest.fixture
def fake_rows(tmp_path, monkeypatch):
    """The two row env files (one without a trailing newline), a base level file, and a campaign dir in tmp_path."""
    for name, text in (("suite", "KRABBY_A=1\nKRABBY_B=2\n"), ("field_light", "KRABBY_A=1\nKRABBY_FL=3")):
        path = tmp_path / f"{name}.env"
        path.write_text(text)
        monkeypatch.setitem(C.ROWS[name], "env", path)
    base90 = tmp_path / "L090_base.env"
    base90.write_text("KRABBY_PAYLOAD_COM_XY=0\nKRABBY_PAYLOAD_COM_Z=0.309\nKRABBY_PAYLOAD_KG=90:90\n"
                      "KRABBY_PAYLOAD_ZERO_FRAC=0\n")
    monkeypatch.setitem(C.BASE_LEVEL, "L090", base90)
    monkeypatch.setattr(C, "CAMPAIGN", tmp_path / "campaign")
    return tmp_path


def test_env_text_adds_exactly_one_stress_line_for_stress_kinds_and_none_otherwise(fake_rows):
    assert C.env_text(T["T08"], 1.0, "suite") == "KRABBY_A=1\nKRABBY_B=2\nKRABBY_STRESS_OBS_NOISE=1\n"
    assert C.env_text(T["T06"], 0.4, "field_light") == "KRABBY_A=1\nKRABBY_FL=3\nKRABBY_STRESS_MU=0.4\n"
    assert C.env_text(T["T10b"], 1, "suite").endswith("KRABBY_STRESS_VXY_ZERO=1\n")
    for term in C.TERMS:
        for lv in term["levels"]:
            for rw in ("suite", "field_light"):
                text = C.env_text(term, lv, rw)
                stress = [ln for ln in text.splitlines() if ln.startswith("KRABBY_STRESS_")]
                assert len(stress) == (1 if term["kind"] == "stress" else 0), (term["term"], lv, rw)
                assert text.endswith("\n") and text.startswith("KRABBY_A=1\n")
    for term in ("T12", "T13", "T14"):                                           # no stress line: zero-depth / payload cells
        assert C.env_text(T[term], T[term]["levels"][0], "field_light") == "KRABBY_A=1\nKRABBY_FL=3\n"
    assert C.env_text(None, None, "suite") == "KRABBY_A=1\nKRABBY_B=2\n"          # the nominal launches


def test_level_text_and_level_files(fake_rows):
    assert C.level_text(T["T13"], 150.0) == ("KRABBY_PAYLOAD_COM_XY=0\nKRABBY_PAYLOAD_COM_Z=0.309\n"
                                            "KRABBY_PAYLOAD_KG=150:150\nKRABBY_PAYLOAD_ZERO_FRAC=0\n")
    assert C.level_text(T["T14"], 0.2) == ("KRABBY_PAYLOAD_COM_XY=0.2\nKRABBY_PAYLOAD_COM_Z=0.309\n"
                                          "KRABBY_PAYLOAD_KG=90:90\nKRABBY_PAYLOAD_ZERO_FRAC=0\n")
    assert C.level_file_for(T["T13"], 165.0, "payload") == C.CAMPAIGN / "levels" / "L165.env"
    assert C.level_file_for(T["T13"], 90.0, "payload").name == "L090.env"        # three digits, like the base level L090
    assert C.level_file_for(T["T14"], 0.3, "L090") == C.CAMPAIGN / "levels" / "L090_comxy0p3.env"
    assert C.level_file_for(T["T06"], 0.4, "L000") is None and C.level_file_for(T["T12"], 1, "L000") is None
    assert C.level_file_for(T["T01"], 0.7, "L090") == C.BASE_LEVEL["L090"] and C.level_file_for(None, None, "L000") is None
    assert C.env_file("T08_x", "suite") == C.CAMPAIGN / "envs" / "T08_x__suite.env"


def test_expected_env_is_what_a_stamp_must_carry(fake_rows):
    payload = {"KRABBY_PAYLOAD_COM_XY": "0", "KRABBY_PAYLOAD_COM_Z": "0.309", "KRABBY_PAYLOAD_KG": "90:90",
               "KRABBY_PAYLOAD_ZERO_FRAC": "0"}
    assert C.expected_env(T["T06"], 0.4, "L000") == {"KRABBY_STRESS_MU": "0.4"}          # an unarmed base adds nothing
    assert C.expected_env(T["T01"], 0.7, "L090") == {"KRABBY_STRESS_ROD_SPEED": "0.7", **payload}
    assert C.expected_env(None, None, "L090") == payload and C.expected_env(None, None, "L000") == {}
    assert C.expected_env(T["T12"], 1, "L000") == {}
    t13 = C.expected_env(T["T13"], 150.0, "payload")                                      # the four payload keys, no stress key
    assert t13 == {**payload, "KRABBY_PAYLOAD_KG": "150:150"}
    t14 = C.expected_env(T["T14"], 0.2, "L090")
    assert t14 == {**payload, "KRABBY_PAYLOAD_COM_XY": "0.2"} and len(t14) == 4
    # a level file on disk wins over level_text (it is what gate_row applied); non-payload lines are not keys
    (C.CAMPAIGN / "levels").mkdir(parents=True)
    (C.CAMPAIGN / "levels" / "L150.env").write_text("KRABBY_PAYLOAD_KG=151:151\nKRABBY_OTHER=1\n# KRABBY_PAYLOAD_X=1\n")
    assert C.expected_env(T["T13"], 150.0, "payload") == {"KRABBY_PAYLOAD_KG": "151:151"}


def test_rows_for_composes_the_two_gate_rows_of_a_cell():
    rows = C.rows_for("T06_mu_0p4_L000", T["T06"], 0.4, "L000", 2)
    assert [r["row"] for r in rows] == ["suite", "field_light"] and {r["es"] for r in rows} == {2}
    for r in rows:
        assert r["name"] == "DRS_T06_mu_0p4_L000" and r["cell"] == "T06_mu_0p4_L000" and r["policy"] == "student"
        assert r["checkpoint"] == str(C.CKPT) and r["level_file"] is None
        assert r["args"][-2:] == ["--policy-role", "student"] and r["args"][:2] == ["--task", C.TASK]
        assert r["env_file"] == str(C.env_file("T06_mu_0p4_L000", r["row"]))
    assert rows[0]["scenario"] == "suite__omni" and "--trained-categories" in rows[0]["args"]
    assert rows[1]["scenario"] == "field_light__omni" and "--trained-categories" not in rows[1]["args"]
    zero = C.rows_for("T12_zero_depth_1_L000", T["T12"], 1, "L000", 1)
    assert {r["checkpoint"] for r in zero} == {str(C.ZERO_DEPTH_DIR / C.CKPT.name)}
    teacher = C.rows_for("T06_mu_0p4_L000", T["T06"], 0.4, "L000", 1, "teacher")
    assert {r["checkpoint"] for r in teacher} == {str(C.TSTAR)} and {r["name"] for r in teacher} == {"DRST_T06_mu_0p4_L000"}
    assert teacher[0]["args"][-2:] == ["--policy-role", "teacher"]
    l090 = C.rows_for("T01_rod_speed_0p7_L090", T["T01"], 0.7, "L090", 1)
    assert {r["level_file"] for r in l090} == {str(C.BASE_LEVEL["L090"])}
    assert C.checkpoint_for(None) == C.CKPT and C.checkpoint_for(T["T12"], "teacher") == C.TSTAR


def test_stamp_dir_is_the_newest_complete_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "TP_OUT", tmp_path)
    assert C.stamp_dir("DRS_X", "suite__omni", 1) is None                         # no row dir at all
    base = tmp_path / "lr" / "DRS_X" / "suite__omni" / "seed001"
    (base / "2026-11-06_10-00-00").mkdir(parents=True)
    assert C.stamp_dir("DRS_X", "suite__omni", 1) is None                         # a stamp without tp_row.json is incomplete
    (base / "2026-11-06_10-00-00" / "tp_row.json").write_text("{}")
    (base / "2026-11-06_09-00-00").mkdir()
    (base / "2026-11-06_09-00-00" / "tp_row.json").write_text("{}")
    (base / "2026-11-06_11-00-00").mkdir()                                       # newer but incomplete: ignored
    assert C.stamp_dir("DRS_X", "suite__omni", 1) == base / "2026-11-06_10-00-00"
    (base / "2026-11-06_12-00-00").mkdir()
    (base / "2026-11-06_12-00-00" / "tp_row.json").write_text("{}")
    assert C.stamp_dir("DRS_X", "suite__omni", 1) == base / "2026-11-06_12-00-00"
    assert C.stamp_dir("DRS_X", "suite__omni", 2) is None                         # the seed is part of the path


# =============================================================================== launch errors
def _stamp(tmp_path, env=None, *, zero_depth=False, code_state=NO, run_meta=None) -> Path:
    d = tmp_path / f"stamp{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    meta = {"env_vars": env or {}, "student": {"zero_depth": zero_depth}} if run_meta is None else run_meta
    (d / "run_meta.json").write_text(json.dumps(meta))
    if code_state is NO:
        code_state = {"dirty": "False", "code_sha": "c0de", "eval_sha": "e0a1"}
    if code_state is not None:
        (d / "code_state.json").write_text(json.dumps(code_state))
    return d


PIN = {"code_sha": "c0de", "eval_sha": "e0a1", "pinned_at": "2026-11-06T12:00:00"}


def test_a_clean_stamp_has_no_problems(tmp_path):
    st = _stamp(tmp_path, {"KRABBY_STRESS_MU": "0.4", "KRABBY_OTHER": "x"})
    assert SR.stamp_problems(st, {"KRABBY_STRESS_MU": "0.4"}, False, PIN) == []
    assert SR.stamp_problems(_stamp(tmp_path), {}, False, PIN) == []                # a nominal: no keys wanted, none present
    payload = {"KRABBY_PAYLOAD_KG": "90:90"}
    assert SR.stamp_problems(_stamp(tmp_path, payload), payload, None, PIN) == []   # zero_depth None: not checked


def test_a_missing_launch_pin_is_itself_a_problem_and_skips_the_hash_comparison(tmp_path):
    out = SR.stamp_problems(_stamp(tmp_path), {}, False, None)
    assert len(out) == 1 and "no launch pin" in out[0]                              # the driver writes it before any row
    off = SR.stamp_problems(_stamp(tmp_path, code_state={"dirty": "False", "code_sha": "x", "eval_sha": "y"}), {}, False, None)
    assert len(off) == 1 and "no launch pin" in off[0]                              # nothing to compare the hashes with
    assert len(SR.stamp_problems(_stamp(tmp_path, code_state=None), {}, False, None)) == 2          # dirty + no pin
    assert len(SR.stamp_problems(_stamp(tmp_path, {"KRABBY_STRESS_X": "1"}), {}, False, None)) == 2   # every check still runs


def test_a_missing_or_different_knob_key_is_a_problem(tmp_path):
    want = {"KRABBY_STRESS_MU": "0.4", "KRABBY_PAYLOAD_KG": "90:90"}
    missing = SR.stamp_problems(_stamp(tmp_path, {"KRABBY_PAYLOAD_KG": "90:90"}), want, False, PIN)
    assert len(missing) == 1 and "KRABBY_STRESS_MU" in missing[0] and "None" in missing[0]
    different = SR.stamp_problems(_stamp(tmp_path, {"KRABBY_STRESS_MU": "0.8", "KRABBY_PAYLOAD_KG": "90:90"}), want, False, PIN)
    assert len(different) == 1 and "'0.4'" in different[0] and "'0.8'" in different[0]
    assert len(SR.stamp_problems(_stamp(tmp_path, {}), want, False, PIN)) == 2          # both keys
    assert SR.stamp_problems(_stamp(tmp_path, {"KRABBY_STRESS_MU": 0.4}), {"KRABBY_STRESS_MU": "0.4"}, False, PIN) == []


def test_a_stray_stress_key_is_a_problem_and_other_keys_are_not(tmp_path):
    stray = SR.stamp_problems(_stamp(tmp_path, {"KRABBY_STRESS_CAM": "0.8", "KRABBY_STRESS_MU": "0.4"}),
                              {"KRABBY_STRESS_MU": "0.4"}, False, PIN)
    assert len(stray) == 1 and "unexpected KRABBY_STRESS_CAM" in stray[0]
    assert len(SR.stamp_problems(_stamp(tmp_path, {"KRABBY_STRESS_MU": "0.4"}), {}, False, PIN)) == 1     # a nominal with a knob
    other = {"KRABBY_OMNI_X": "1", "KRABBY_PAYLOADX": "1", "KRABBY_STRESSX": "1", "TP_OUT": "/x"}   # not the two prefixes
    assert SR.stamp_problems(_stamp(tmp_path, other), {}, False, PIN) == []


def test_a_stray_payload_key_is_a_problem_exactly_like_a_stray_stress_key(stamps_root):
    payload = {"KRABBY_PAYLOAD_COM_XY": "0", "KRABBY_PAYLOAD_COM_Z": "0.309", "KRABBY_PAYLOAD_KG": "90:90",
               "KRABBY_PAYLOAD_ZERO_FRAC": "0"}
    # the unit: a 0 kg cell wants no payload key, a 90 kg cell wants its four and nothing else
    one = SR.stamp_problems(_stamp(stamps_root, {"KRABBY_PAYLOAD_KG": "90:90"}), {}, False, PIN)
    assert len(one) == 1 and "unexpected KRABBY_PAYLOAD_KG='90:90'" in one[0]
    assert SR.stamp_problems(_stamp(stamps_root, payload), payload, None, PIN) == []
    extra = SR.stamp_problems(_stamp(stamps_root, {**payload, "KRABBY_PAYLOAD_NEW": "1"}), payload, None, PIN)
    assert len(extra) == 1 and "unexpected KRABBY_PAYLOAD_NEW='1'" in extra[0]
    both = SR.stamp_problems(_stamp(stamps_root, {"KRABBY_PAYLOAD_KG": "9:9", "KRABBY_STRESS_MU": "0.4"}), {}, False, PIN)
    assert len(both) == 2 and any("KRABBY_PAYLOAD_KG" in b for b in both) and any("KRABBY_STRESS_MU" in b for b in both)
    short = SR.stamp_problems(_stamp(stamps_root, {k: v for k, v in payload.items() if k != "KRABBY_PAYLOAD_KG"}),
                              payload, None, PIN)
    assert len(short) == 1 and short[0].startswith("KRABBY_PAYLOAD_KG=")               # a wanted key is still required
    # the reader: a launch error, never a reading (once per stamp) ...
    name = C.gate_name("T06_mu_0p4_L000")
    _cell_stamps(stamps_root, name, 1, {"KRABBY_STRESS_MU": "0.4", "KRABBY_PAYLOAD_KG": "90:90"})
    bad = SR.cellseed(name, 1, T["T06"], 0.4, "L000")
    assert bad["pass"] is None and len(bad["launch_error"]) == 2 and "unexpected KRABBY_PAYLOAD_KG" in bad["launch_error"][0]
    l090 = C.gate_name(C.nominal_id("L090", "A"))
    _cell_stamps(stamps_root, l090, 1, {**payload, "KRABBY_PAYLOAD_NEW": "1"})
    assert SR.cellseed(l090, 1, None, None, "L090")["pass"] is None
    # ... and the driver's: the same stamp fails row_problems, so it would be re-run, not skipped
    assert "unexpected KRABBY_PAYLOAD_KG" in RS.row_problems(_mu_row(), T["T06"], 0.4, "L000")[0]


@pytest.mark.parametrize("meta_zero, want, bad", [
    (False, False, False), (True, True, False), (True, False, True), (False, True, True), (None, False, True),
])
def test_zero_depth_must_match_the_cell_and_is_unchecked_for_none(tmp_path, meta_zero, want, bad):
    out = SR.stamp_problems(_stamp(tmp_path, zero_depth=meta_zero), {}, want, PIN)
    assert bool(out) is bad and (not bad or "zero_depth" in out[0])
    assert SR.stamp_problems(_stamp(tmp_path, zero_depth=meta_zero), {}, None, PIN) == []        # the teacher: not checked
    nostudent = _stamp(tmp_path, run_meta={"env_vars": {}})                                      # no student block at all
    assert SR.stamp_problems(nostudent, {}, False, PIN) != [] and SR.stamp_problems(nostudent, {}, None, PIN) == []


@pytest.mark.parametrize("state, bad", [
    ({"dirty": "False", "code_sha": "c0de", "eval_sha": "e0a1"}, False),
    ({"dirty": False, "code_sha": "c0de", "eval_sha": "e0a1"}, False),                  # a JSON bool is fine too
    ({"dirty": "True", "code_sha": "c0de", "eval_sha": "e0a1"}, True),
    ({"dirty": True, "code_sha": "c0de", "eval_sha": "e0a1"}, True),
    ({"code_sha": "c0de", "eval_sha": "e0a1"}, True),                                    # no dirty key
])
def test_a_dirty_or_unrecorded_tree_is_a_problem(tmp_path, state, bad):
    out = SR.stamp_problems(_stamp(tmp_path, code_state=state), {}, False, PIN)
    assert bool(out) is bad and (not bad or "dirty" in out[0])


def test_hashes_off_the_launch_pin_are_problems(tmp_path):
    ok = {"dirty": "False", "code_sha": "c0de", "eval_sha": "e0a1"}
    code = SR.stamp_problems(_stamp(tmp_path, code_state={**ok, "code_sha": "beef"}), {}, False, PIN)
    assert len(code) == 1 and "code_sha" in code[0] and "beef" in code[0] and "c0de" in code[0]
    ev = SR.stamp_problems(_stamp(tmp_path, code_state={**ok, "eval_sha": "beef"}), {}, False, PIN)
    assert len(ev) == 1 and "eval_sha" in ev[0]
    both = SR.stamp_problems(_stamp(tmp_path, code_state={**ok, "code_sha": "a", "eval_sha": "b"}), {}, False, PIN)
    assert len(both) == 2
    nostate = SR.stamp_problems(_stamp(tmp_path, code_state=None), {}, False, PIN)          # unrecorded: dirty + both hashes
    assert len(nostate) == 3


def _cell_stamps(tmp_path, name, es, env, *, zero_depth=False, code_state=NO, rows=("suite__omni", "field_light__omni"),
                 mtime=None, ts="2026-11-06_10-00-00"):
    """Both rows' stamps of a cell-seed; ``mtime`` sets the tp_row.json time, ``ts`` the stamp dir (the newest name wins)."""
    for scenario in rows:
        d = tmp_path / "lr" / name / scenario / f"seed{es:03d}" / ts
        d.mkdir(parents=True)
        meta = {"env_vars": env, "student": {"zero_depth": zero_depth}}
        (d / "run_meta.json").write_text(json.dumps(meta))
        cs = {"dirty": "False", "code_sha": "c0de", "eval_sha": "e0a1"} if code_state is NO else code_state
        (d / "code_state.json").write_text(json.dumps(cs))
        (d / "tp_row.json").write_text("{}")
        if mtime is not None:
            os.utime(d / "tp_row.json", (mtime, mtime))


@pytest.fixture
def stamps_root(tmp_path, monkeypatch):
    """A stamp tree and launch pin in tmp_path, and a ``read_stamps`` that returns a passing record."""
    monkeypatch.setattr(C, "TP_OUT", tmp_path)
    monkeypatch.setattr(SR, "LAUNCH", tmp_path / "launch.json")
    SR.LAUNCH.write_text(json.dumps(PIN))
    base90 = tmp_path / "L090_base.env"                                           # the 90 kg base's level file, as shipped
    base90.write_text("KRABBY_PAYLOAD_COM_XY=0\nKRABBY_PAYLOAD_COM_Z=0.309\nKRABBY_PAYLOAD_KG=90:90\n"
                      "KRABBY_PAYLOAD_ZERO_FRAC=0\n")
    monkeypatch.setitem(C.BASE_LEVEL, "L090", base90)
    monkeypatch.setattr(SR, "read_stamps", lambda sd, fd, es: {"es": es, "pass": True, "lines": {}, "failed": [],
                                                              "lenient_feasible": True, "n_lines": 18})
    return tmp_path


def test_cellseed_turns_a_launch_error_into_pass_none(stamps_root):
    name = C.gate_name("T06_mu_0p4_L000")
    _cell_stamps(stamps_root, name, 1, {"KRABBY_STRESS_MU": "0.4"})
    good = SR.cellseed(name, 1, T["T06"], 0.4, "L000")
    assert good["pass"] is True and good["launch_error"] is None
    assert SR.cellseed(name, 2, T["T06"], 0.4, "L000") is None                    # nothing read for es2
    _cell_stamps(stamps_root, name, 2, {"KRABBY_STRESS_MU": "0.4"}, rows=("suite__omni",))
    assert SR.cellseed(name, 2, T["T06"], 0.4, "L000") is None                    # one of the two stamps missing
    wrong = C.gate_name("T06_mu_0p8_L000")
    _cell_stamps(stamps_root, wrong, 1, {"KRABBY_STRESS_MU": "0.4"})              # it carries the OTHER level's value
    bad = SR.cellseed(wrong, 1, T["T06"], 0.8, "L000")
    assert bad["pass"] is None and bad["launch_error"] and len(bad["launch_error"]) == 2      # once per stamp
    assert "KRABBY_STRESS_MU" in bad["launch_error"][0]


def test_cellseed_checks_zero_depth_by_cell_kind_and_not_for_the_teacher(stamps_root):
    zd, nom = C.gate_name("T12_zero_depth_1_L000"), C.gate_name(C.nominal_id("L000", "A"))
    _cell_stamps(stamps_root, zd, 1, {}, zero_depth=True)
    _cell_stamps(stamps_root, nom, 1, {}, zero_depth=False)
    assert SR.cellseed(zd, 1, T["T12"], 1, "L000")["launch_error"] is None            # term 12 wants zero_depth true ...
    assert SR.cellseed(nom, 1, None, None, "L000")["launch_error"] is None            # ... every other student cell false
    _cell_stamps(stamps_root, "DRS_wrong12", 1, {}, zero_depth=False)
    _cell_stamps(stamps_root, "DRS_wrongnom", 1, {}, zero_depth=True)
    assert SR.cellseed("DRS_wrong12", 1, T["T12"], 1, "L000")["pass"] is None
    assert SR.cellseed("DRS_wrongnom", 1, None, None, "L000")["pass"] is None
    assert SR.cellseed("DRS_wrongnom", 1, None, None, "L000", "teacher")["pass"] is True    # the teacher is unchecked


def test_cellseed_requires_the_payload_keys_of_the_base_and_the_pin(stamps_root):
    payload = {"KRABBY_PAYLOAD_COM_XY": "0", "KRABBY_PAYLOAD_COM_Z": "0.309", "KRABBY_PAYLOAD_KG": "90:90",
               "KRABBY_PAYLOAD_ZERO_FRAC": "0"}
    name = C.gate_name(C.nominal_id("L090", "A"))
    _cell_stamps(stamps_root, name, 1, payload)
    assert SR.cellseed(name, 1, None, None, "L090")["launch_error"] is None
    _cell_stamps(stamps_root, "DRS_nopayload", 1, {})
    assert SR.cellseed("DRS_nopayload", 1, None, None, "L090")["pass"] is None             # the 90 kg base wants its 4 keys
    _cell_stamps(stamps_root, "DRS_offpin", 1, {}, code_state={"dirty": "False", "code_sha": "zzz", "eval_sha": "e0a1"})
    off = SR.cellseed("DRS_offpin", 1, None, None, "L000")
    assert off["pass"] is None and "code_sha" in off["launch_error"][0]
    SR.LAUNCH.unlink()                                                                    # without the pin NOTHING is a reading
    assert SR.launch_pin() is None
    nopin = SR.cellseed(name, 1, None, None, "L090")                                      # a perfect L090 nominal stamp
    assert nopin["pass"] is None and "no launch pin" in nopin["launch_error"][0]
    SR.LAUNCH.write_text(json.dumps(PIN))
    assert SR.launch_pin() == PIN and SR.cellseed(name, 1, None, None, "L090")["launch_error"] is None


# =============================================================================== read_stamps
def _level_line(key, es, value, ok=True):
    stat, _, scope = key.partition(".")
    return {"id": f"p3a.stable90.{key}.es{es}", "stat": stat, "scope": scope or None,
            "es": None if key == "field_light_completion" else es, "line": 0.9, "direction": ">=", "value": value, "pass": ok}


class _FakeReader:
    def __init__(self, lines):
        self.lines, self.calls = lines, []

    def level_reads(self, stamps, th):
        self.calls.append((dict(stamps), th))
        return {"lines": self.lines}


class _FakeThresholds:
    def __init__(self, cats):
        self.cats, self.paths = cats, []

    def suite_categories(self, path):
        self.paths.append(path)
        return self.cats


def _install_fakes(monkeypatch, lines, cats):
    reader, thr = _FakeReader(lines), _FakeThresholds(cats)
    monkeypatch.setattr(SR, "_payload_read", lambda: reader)
    monkeypatch.setattr(SR, "_thresholds_mod", lambda: thr)
    monkeypatch.setattr(SR, "_th", lambda: {"fake": "thresholds"})
    return reader, thr


ALL_CATS = {c: {"survival": 0.95} for c in SR.LENIENT_CATS}


def test_read_stamps_keeps_the_seeds_lines_and_the_completion(monkeypatch, tmp_path):
    both = [_level_line(k, es, 0.9 + 0.01 * es) for es in (1, 2) for k in LINE_KEYS if k != "field_light_completion"]
    both.append(_level_line("field_light_completion", None, 0.97))
    reader, thr = _install_fakes(monkeypatch, both, ALL_CATS)
    out = SR.read_stamps(tmp_path / "s", tmp_path / "f", 2)
    assert sorted(out["lines"]) == sorted(LINE_KEYS) and out["n_lines"] == 18 and out["es"] == 2
    assert out["lines"]["survival.fwd"]["value"] == pytest.approx(0.92)           # the es2 line, not es1's 0.91
    assert out["lines"]["field_light_completion"]["value"] == 0.97
    assert out["pass"] is True and out["failed"] == []
    assert out["lines"]["survival.fwd"] == {"value": pytest.approx(0.92), "line": 0.9, "direction": ">=", "pass": True}
    assert reader.calls == [({"suite@2": str(tmp_path / "s"), "field_light@2": str(tmp_path / "f")}, {"fake": "thresholds"})]
    assert thr.paths == [tmp_path / "s"]
    assert out["stamps"] == {"suite": str(tmp_path / "s"), "field_light": str(tmp_path / "f")}
    first = SR.read_stamps(tmp_path / "s", tmp_path / "f", 1)                    # and seed 1 gets ITS lines, not the later ones
    assert first["lines"]["survival.fwd"]["value"] == pytest.approx(0.91) and first["n_lines"] == 18 and first["es"] == 1
    assert reader.calls[1][0] == {"suite@1": str(tmp_path / "s"), "field_light@1": str(tmp_path / "f")}
    assert SR._key({"stat": "field_light_completion", "scope": None}) == "field_light_completion"
    assert SR._key({"stat": "h_meas", "scope": "suite"}) == "h_meas.suite"


def test_read_stamps_passes_only_with_18_passing_lines(monkeypatch, tmp_path):
    lines = [_level_line(k, 1, 0.95, ok=k not in ("h_meas.suite", "survival.turn")) for k in LINE_KEYS
             if k != "field_light_completion"] + [_level_line("field_light_completion", None, 0.99)]
    _install_fakes(monkeypatch, lines, ALL_CATS)
    out = SR.read_stamps(tmp_path, tmp_path, 1)
    assert out["pass"] is False and out["failed"] == ["h_meas.suite", "survival.turn"]            # sorted line keys
    short = [_level_line(k, 1, 0.95) for k in LINE_KEYS if k not in ("tracking.diag",)]
    short = [ln for ln in short if ln["stat"] != "field_light_completion"] + [_level_line("field_light_completion", None, 0.99)]
    _install_fakes(monkeypatch, short, ALL_CATS)
    out = SR.read_stamps(tmp_path, tmp_path, 1)
    assert out["n_lines"] == 17 and out["failed"] == [] and out["pass"] is False                    # a missing line is no pass


@pytest.mark.parametrize("surv, lenient", [
    ({}, True), ({"stand": 0.5}, True), ({"stand": 0.4999}, False), ({"diag": 0.3}, False), ({"move_turn": None}, False),
    ({"stand": 0.92, "fwd": 0.5, "side": 0.5}, True),
])
def test_the_lenient_flag_needs_every_suite_category_including_stand_at_half(monkeypatch, tmp_path, surv, lenient):
    cats = {c: {"survival": 0.95} for c in SR.LENIENT_CATS}
    for c, v in surv.items():
        cats[c] = {"survival": v}
    lines = [_level_line(k, 1, 0.95) for k in LINE_KEYS if k != "field_light_completion"]
    lines.append(_level_line("field_light_completion", None, 0.99))
    _install_fakes(monkeypatch, lines, cats)
    out = SR.read_stamps(tmp_path, tmp_path, 1)
    assert out["lenient_feasible"] is lenient and set(out["survival_all"]) == set(SR.LENIENT_CATS)
    assert SR.LENIENT_CATS == ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
    missing = {c: v for c, v in cats.items() if c != "back"}                                       # a category not read at all
    _install_fakes(monkeypatch, lines, missing)
    assert SR.read_stamps(tmp_path, tmp_path, 1)["lenient_feasible"] is False


# =============================================================================== verdicts
def test_worst_orders_fragile_marginal_incomplete_robust_and_ignores_none():
    assert SR.ORDER == ("FRAGILE", "MARGINAL", "INCOMPLETE", "ROBUST")
    assert SR.worst([]) is None and SR.worst([None, None]) is None and SR.worst(iter([None, "ROBUST"])) == "ROBUST"
    assert SR.worst(["ROBUST", "INCOMPLETE"]) == "INCOMPLETE"
    assert SR.worst(["ROBUST", "INCOMPLETE", "MARGINAL"]) == "MARGINAL"
    assert SR.worst(["MARGINAL", "FRAGILE", "ROBUST", None]) == "FRAGILE"
    assert SR.worst(v for v in ("ROBUST", "ROBUST")) == "ROBUST"


@pytest.mark.parametrize("seeds, status", [
    (S(), None), (S(True), "pass"), (S(False), "fail"), (S(None), "error"), (S(True, True), "pass"),
    (S(False, False), "fail"), (S(True, False), "split"), (S(False, True), "split"), (S(True, None), "error"),
    (S(False, None), "error"), (S(NO, True), "pass"),
])
def test_level_status(seeds, status):
    assert SR.level_status({"seeds": seeds}) == status


EDGE = (T["T08"], 1.0)          # T08: band (0, 1], levels 0.5 / 1.0 / 2.0 / 4.0 -> 1.0 is the only band edge
INBAND = (T["T08"], 0.5)        # in band, not an edge


@pytest.mark.parametrize("seeds, want", [
    (S(), "INCOMPLETE"), (S(None), "INCOMPLETE"), (S(True), "INCOMPLETE"), (S(False), "INCOMPLETE"),
    (S(NO, True), "INCOMPLETE"), (S(NO, False), "INCOMPLETE"),
    (S(True, True), "ROBUST"), (S(False, False), "FRAGILE"), (S(True, False), "MARGINAL"), (S(False, True), "MARGINAL"),
    (S(True, None), "INCOMPLETE"), (S(False, None), "INCOMPLETE"), (S(None, True), "INCOMPLETE"),
    (S(None, None), "INCOMPLETE"),
], ids=repr)
def test_a_band_edge_needs_both_seeds(seeds, want):
    assert SR.level_verdict(*EDGE, {"seeds": seeds}) == want


@pytest.mark.parametrize("seeds, want", [
    (S(), "INCOMPLETE"), (S(None), "INCOMPLETE"),                                # unread / launch error
    (S(True), None), (S(True, True), None),                                      # an es1 pass contributes nothing
    (S(False), "INCOMPLETE"),                                                    # an es1 failure needs es2
    (S(False, False), "FRAGILE"), (S(False, True), "MARGINAL"), (S(True, False), "MARGINAL"),
    (S(False, None), "INCOMPLETE"), (S(True, None), "INCOMPLETE"),               # a launch error is never a failure
], ids=repr)
def test_a_non_edge_in_band_level_needs_es2_only_after_an_es1_failure(seeds, want):
    assert SR.level_verdict(*INBAND, {"seeds": seeds}) == want


def _t08(f05, f10, **out_of_band):
    """T08 with 0.5 (in band), 1.0 (the edge) and optionally 2.0 / 4.0 (beyond the band) filled."""
    fills = {cid("T08", 0.5): f05, cid("T08", 1.0): f10}
    fills.update({cid("T08", float(k.removeprefix("lv"))): v for k, v in out_of_band.items()})
    return row(SR.term_table(make_reads(fills)), "T08")


@pytest.mark.parametrize("f05, f10, want", [
    (S(True), S(True, True), "ROBUST"),
    (S(), S(True, True), "INCOMPLETE"),                                          # an in-band level nobody read
    (S(False), S(True, True), "INCOMPLETE"),                                     # an in-band es1 failure not yet on es2
    (S(True, False), S(True, True), "MARGINAL"),
    (S(False, False), S(True, True), "FRAGILE"),
    (S(True), S(True, False), "MARGINAL"),
    (S(True), S(False, False), "FRAGILE"),
    (S(True), S(True), "INCOMPLETE"),                                            # the edge on one seed only
    (S(True), S(), "INCOMPLETE"),
    (S(True, True), S(True, True), "ROBUST"),
    (S(False, False), S(True, False), "FRAGILE"),                                # FRAGILE > MARGINAL
    (S(True, False), S(True), "MARGINAL"),                                       # MARGINAL > INCOMPLETE
    (S(), S(True), "INCOMPLETE"),
    (S(None), S(True, True), "INCOMPLETE"),                                      # a launch error is INCOMPLETE, never a failure
    (S(True), S(None, None), "INCOMPLETE"),
], ids=repr)
def test_the_term_verdict_is_the_worst_over_the_in_band_levels(quiet, f05, f10, want):
    r = _t08(f05, f10)
    assert r["verdict"] == want
    assert set(r["level_verdicts"]) == {"0p5", "1p0"[:1] + "p0" if False else "1"}  # keyed by C.fmt of the in-band levels


def test_out_of_band_levels_never_decide(quiet):
    ok = _t08(S(True, True), S(True, True), lv2=S(False, False), lv4=S(False, False))
    assert ok["verdict"] == "ROBUST" and set(ok["level_verdicts"]) == {"0p5", "1"}       # 2.0 and 4.0 have no verdict
    assert ok["status"] == {"0p5": "pass", "1": "pass", "2": "fail", "4": "fail"}         # but they are in the status map
    assert ok["side_verdicts"] == {"low": None, "high": "ROBUST"}
    # and the same for a term with both sides: 0.5 is outside T03's (0.8, 1.2) band
    r = row(SR.term_table(make_reads({cid("T03", 0.8): S(True, True), cid("T03", 1.2): S(True, True),
                                      cid("T03", 0.5): S(False, False)})), "T03")
    assert r["verdict"] == "ROBUST" and set(r["level_verdicts"]) == {"0p8", "1p2"}


def test_side_verdicts_are_the_worst_per_side(quiet):
    r = row(SR.term_table(make_reads({cid("T03", 0.8): S(True, True), cid("T03", 1.2): S(False, False),
                                      cid("T03", 0.5): S(True)})), "T03")
    assert r["level_verdicts"] == {"0p8": "ROBUST", "1p2": "FRAGILE"}
    assert r["side_verdicts"] == {"low": "ROBUST", "high": "FRAGILE"} and r["verdict"] == "FRAGILE"
    r = row(SR.term_table(make_reads({cid("T03", 1.2): S(True, True)})), "T03")           # 0.8 unread
    assert r["side_verdicts"] == {"low": "INCOMPLETE", "high": "ROBUST"} and r["verdict"] == "INCOMPLETE"
    r = row(SR.term_table(make_reads({cid("T06", lv): S(True, True) for lv in (0.4, 1.2)}
                                     | {cid("T06", 0.8): S(True, False)})), "T06")
    assert r["level_verdicts"] == {"0p4": "ROBUST", "0p8": "MARGINAL", "1p2": "ROBUST"}
    assert r["side_verdicts"] == {"low": "MARGINAL", "high": "ROBUST"}                  # 0.4 / 0.8 are below the nominal 1.0
    assert r["verdict"] == "MARGINAL"
    assert row(SR.term_table(make_reads()), "T07")["side_verdicts"] == {"low": None, "high": "INCOMPLETE"}


def test_report_only_terms_get_no_verdict(quiet):
    fails = {cid(t, lv, b): S(False, False) for t, b in (("T10b", "L000"), ("T12", "L000"), ("T13", "payload"),
                                                           ("T14", "L090"), ("T15", "L000")) for lv in T[t]["levels"]}
    table = SR.term_table(make_reads(fails))
    for term in ("T10b", "T12", "T13", "T14", "T15"):
        r = row(table, term)
        assert r["verdict"] is None and r["side_verdicts"] == {} and r["level_verdicts"] == {} and r["band"] is None, term
        assert set(r["status"].values()) == {"fail"} and table["combined_verdicts"][term] is None
    assert row(table, "T15")["tolerance"]["high"]["tolerated_edge"] is None             # the status and edges are still reported
    assert set(row(table, "T15")["status"]) == {"0p1", "0p2", "0p35", "0p5"}


def test_terms_one_and_two_combine_their_bases_worst_first(quiet):
    def robust(term, base):
        t = T[term]
        return {cid(term, lv, base): S(True, True) for lv in t["levels"] if C.inside_band(t, lv)}
    fills = robust("T01", "L000") | robust("T02", "L000") | robust("T01", "L090") | robust("T02", "L090")
    table = SR.term_table(make_reads(fills))
    assert [r["verdict"] for r in table["rows"] if r["term"] in ("T01", "T02")] == ["ROBUST"] * 4
    assert table["combined_verdicts"]["T01"] == "ROBUST" and table["combined_verdicts"]["T02"] == "ROBUST"
    fills[cid("T01", 0.7, "L090")] = S(False, False)                                    # FRAGILE on either base is FRAGILE
    table = SR.term_table(make_reads(fills))
    assert row(table, "T01", "L000")["verdict"] == "ROBUST" and row(table, "T01", "L090")["verdict"] == "FRAGILE"
    assert table["combined_verdicts"]["T01"] == "FRAGILE" and table["combined_verdicts"]["T02"] == "ROBUST"
    del fills[cid("T02", 0.85, "L000")]                                                  # an unread edge: INCOMPLETE beats ROBUST
    table = SR.term_table(make_reads(fills))
    assert row(table, "T02", "L000")["verdict"] == "INCOMPLETE" and table["combined_verdicts"]["T02"] == "INCOMPLETE"
    fills[cid("T02", 0.85, "L090")] = S(True, False)
    table = SR.term_table(make_reads(fills))
    assert table["combined_verdicts"]["T02"] == "MARGINAL"                               # MARGINAL beats INCOMPLETE
    assert [r["base"] for r in table["rows"] if r["term"] == "T01"] == ["L000", "L090"]
    assert len(table["rows"]) == 19 and set(table["combined_verdicts"]) == set(T)


def test_tolerated_edges_stop_at_the_first_level_that_is_not_a_pass(quiet):
    fills = {cid("T03", 0.8): S(True), cid("T03", 0.5): S(True), cid("T03", 1.2): S(True, True)}
    r = row(SR.term_table(make_reads(fills)), "T03")
    assert r["tolerance"]["low"] == {"tolerated_edge": 0.5, "lenient_edge": 0.5, "levels": [0.8, 0.5]}   # nearest first
    assert r["tolerance"]["high"]["tolerated_edge"] == 1.2 and r["tolerance"]["high"]["levels"] == [1.2]
    for near, far, edge in [(S(False), S(True), None), (S(True, False), S(True), None), (S(None), S(True), None),
                            (S(), S(True), None), (S(True, True), S(False), 0.8), (S(True, True), S(None), 0.8),
                            (S(True, True), S(), 0.8), (S(True, True), S(True, True), 0.5)]:
        r = row(SR.term_table(make_reads({cid("T03", 0.8): near, cid("T03", 0.5): far})), "T03")
        assert r["tolerance"]["low"]["tolerated_edge"] == edge, (near, far)
    # one side only (T08: every level is above the nominal), counted outward through 2.0 / 4.0 too
    r = row(SR.term_table(make_reads({cid("T08", 0.5): S(True), cid("T08", 1.0): S(True), cid("T08", 2.0): S(False),
                                      cid("T08", 4.0): S(True)})), "T08")
    assert r["tolerance"]["high"]["tolerated_edge"] == 1.0 and r["tolerance"]["high"]["levels"] == [0.5, 1.0, 2.0, 4.0]
    assert r["tolerance"]["low"] == {"tolerated_edge": None, "lenient_edge": None, "levels": []}


def test_the_lenient_edge_needs_a_read_and_lenient_feasibility_on_every_seed(quiet):
    def edge(fills):
        return row(SR.term_table(make_reads({cid("T08", lv): v for lv, v in fills.items()})), "T08")["tolerance"]["high"]
    full = {0.5: S(rec(True)), 1.0: S(rec(False, lenient=True)), 2.0: S(rec(True, lenient=False)), 4.0: S(rec(True))}
    got = edge(full)
    assert got["lenient_edge"] == 1.0 and got["tolerated_edge"] == 0.5            # a failing level can still be lenient-feasible
    assert edge({**full, 1.0: S(rec(None, lenient=True))})["lenient_edge"] == 0.5     # a launch error stops the lenient walk
    assert edge({**full, 0.5: S(rec(True, lenient=False))})["lenient_edge"] is None
    assert edge({**full, 1.0: S(rec(True), rec(True, lenient=False))})["lenient_edge"] == 0.5   # EVERY seed present
    assert edge({**full, 1.0: S(rec(True), rec(None, lenient=True))})["lenient_edge"] == 0.5
    assert edge({**full, 1.0: S()})["lenient_edge"] == 0.5                         # unread: stops
    assert edge({**full, 1.0: S(rec(True), rec(False))})["lenient_edge"] == 4.0 if False else True
    assert edge({0.5: S(rec(True)), 1.0: S(rec(True)), 2.0: S(rec(True)), 4.0: S(rec(True))})["lenient_edge"] == 4.0
    assert edge({})["lenient_edge"] is None


# =============================================================================== deltas
def _d(value, *refs, key="survival.fwd", flag=True):
    mk = lambda v: {"lines": {key: {"value": v}}}  # noqa: E731
    return SR.deltas(mk(value), [mk(r) for r in refs], flag=flag)[key]


def test_a_drop_of_exactly_0p05_does_not_flag_and_0p06_does():
    assert SR.DROP_FLAG == 0.05 and SR.EPS == 1e-9
    for value, ref in ((0.85, 0.90), (0.90, 0.95), (0.55, 0.60), (0.80, 0.85), (0.0, 0.05), (0.9499999995, 1.0)):
        assert _d(value, ref)["flag"] is False, (value, ref)                    # strict, with an epsilon for the float error
    for value, ref in ((0.84, 0.90), (0.89, 0.95), (0.0, 0.06), (0.5, 1.0), (0.9499999985, 1.0)):
        assert _d(value, ref)["flag"] is True, (value, ref)
    d = _d(0.84, 0.90)
    assert d == {"value": 0.84, "delta": [pytest.approx(-0.06)], "flag": True}


def test_the_flag_needs_a_drop_beyond_0p05_against_every_nominal_launch():
    assert _d(0.84, 0.90, 0.90)["flag"] is True
    assert _d(0.84, 0.90, 0.86)["flag"] is False and _d(0.84, 0.86, 0.90)["flag"] is False       # one launch within 0.05
    assert _d(0.84, 0.90, 0.80)["flag"] is False                                                 # the other launch is below us
    assert _d(0.84, 0.90, 0.90)["delta"] == [pytest.approx(-0.06), pytest.approx(-0.06)]
    assert _d(0.99, 0.90)["flag"] is False and _d(0.99, 0.90)["delta"] == [pytest.approx(0.09)]   # a rise is never a flag


@pytest.mark.parametrize("key, flags", [
    ("survival.fwd", True), ("survival.turn", True), ("survival.move_turn", True), ("field_light_completion", True),
    ("tracking.fwd", False), ("turn_fc.turn", False), ("h_meas.suite", False), ("h_meas.field_light", False),
    ("roll_rms.suite", False), ("pitch_rms.field_light", False),
])
def test_only_survival_and_completion_lines_can_flag(key, flags):
    assert _d(0.2, 0.9, key=key)["flag"] is flags and _d(0.2, 0.9, 0.9, key=key)["delta"] == [pytest.approx(-0.7)] * 2


def test_flag_false_suppresses_and_missing_values_never_flag():
    assert _d(0.5, 0.9, 0.9, flag=False) == {"value": 0.5, "delta": [pytest.approx(-0.4)] * 2, "flag": False}
    assert _d(None, 0.9) == {"value": None, "delta": [None], "flag": False}
    assert _d(0.5, None) == {"value": 0.5, "delta": [None], "flag": False}
    assert _d(0.5, 0.9, None)["flag"] is False and _d(0.5, 0.9, None)["delta"] == [pytest.approx(-0.4), None]
    assert _d(0.5) == {"value": 0.5, "delta": [], "flag": False}                 # no reference at all: reported, never flagged
    rec_ = {"lines": {"survival.fwd": {"value": 0.5}}}
    assert SR.deltas(rec_, [{"lines": {}}])["survival.fwd"] == {"value": 0.5, "delta": [None], "flag": False}


# =============================================================================== term_table: references and flags
def test_term_table_takes_deltas_against_the_right_nominal_and_never_flags_the_payload_term(monkeypatch):
    drop = rec(True, {"survival.fwd": 0.80})                                    # 0.15 below both nominals
    refs = {("L000", 1): [rec(True), rec(True, {"survival.fwd": 0.94})], ("L090", 1): [rec(True)]}
    calls = []

    def nominal_refs(base, es):
        calls.append((base, es))
        return refs.get((base, es), [])
    monkeypatch.setattr(SR, "nominal_refs", nominal_refs)
    monkeypatch.setattr(SR, "closing_record", lambda: None)
    fills = {cid("T15", 0.1): S(drop), cid("T12", 1): S(drop), cid("T01", 0.5, "L090"): S(drop),
             cid("T13", 150.0, "payload"): S(drop), cid("T14", 0.2, "L090"): S(drop),
             cid("T03", 0.5): S(rec(None, {"survival.fwd": 0.1}))}              # a launch error: no deltas, no lookups
    table = SR.term_table(make_reads(fills))
    assert {(r["term"], r["base"]): r["delta_reference"] for r in table["rows"]} == {
        **{(t["term"], b): "L000" for t in C.TERMS for b in t["bases"] if b == "L000"},
        ("T01", "L090"): "L090", ("T02", "L090"): "L090", ("T13", "payload"): "L090", ("T14", "L090"): "L090"}
    assert sorted(set(calls)) == [("L000", 1), ("L090", 1)] and ("L000", 2) not in calls      # only the seeds that were read
    assert calls.count(("L090", 1)) == 3                                                       # T01@L090, T13, T14
    t15 = row(table, "T15")["deltas"]["0p1"][1]["survival.fwd"]
    assert t15 == {"value": 0.80, "delta": [pytest.approx(-0.15), pytest.approx(-0.14)], "flag": True}
    assert row(table, "T15")["drop_flags"] == ["0p1@es1:survival.fwd"]
    assert row(table, "T12")["drop_flags"] == ["1@es1:survival.fwd"]
    assert row(table, "T01", "L090")["drop_flags"] == ["0p5@es1:survival.fwd"]
    t13 = row(table, "T13")
    assert t13["deltas"]["150"][1]["survival.fwd"]["delta"] == [pytest.approx(-0.15)]       # against the 90 kg nominal
    assert t13["deltas"]["150"][1]["survival.fwd"]["flag"] is False and t13["drop_flags"] == []   # ... and never flagged
    assert row(table, "T14")["drop_flags"] == ["0p2@es1:survival.fwd"]                         # the CoM term flags normally
    assert row(table, "T03")["deltas"] == {} and row(table, "T03")["drop_flags"] == []
    assert row(table, "T15")["nominal_refs_used"] == {"0p1@es1": 2} and row(table, "T15")["no_reference"] == []
    assert row(table, "T13")["nominal_refs_used"] == {"150@es1": 1} and row(table, "T03")["nominal_refs_used"] == {}
    assert SR._ref_base(C.cells()[0]) == "L000"


def test_a_nominal_that_is_a_launch_error_shows_as_fewer_references_and_none_as_no_reference(monkeypatch):
    refs = {("L000", 1): [rec(True)], ("L000", 2): []}                          # launch B dropped at es1, both dropped at es2
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: refs.get((base, es), []))
    monkeypatch.setattr(SR, "closing_record", lambda: None)
    drop = rec(True, {"survival.fwd": 0.80})
    table = SR.term_table(make_reads({cid("T15", 0.1): S(drop, drop), cid("T15", 0.2): S(drop), cid("T15", 0.35): S(None),
                                      cid("T01", 0.5, "L090"): S(drop)}))
    r = row(table, "T15")
    assert r["nominal_refs_used"] == {"0p1@es1": 1, "0p1@es2": 0, "0p2@es1": 1}      # the launch-error seed is not looked up
    assert r["no_reference"] == ["0p1@es2"]
    assert r["deltas"]["0p1"][2]["survival.fwd"] == {"value": 0.80, "delta": [], "flag": False}   # no reference: no flag
    assert r["deltas"]["0p1"][1]["survival.fwd"]["flag"] is True
    assert row(table, "T01", "L090")["no_reference"] == ["0p5@es1"]                  # the (empty) L090 nominal
    assert row(table, "T03")["no_reference"] == []


def test_nominal_refs_are_launches_a_and_b_at_0_kg_and_a_at_l090_and_never_a_launch_error(monkeypatch):
    nom = lambda b, la: C.gate_name(C.nominal_id(b, la))  # noqa: E731
    a, b, a90 = rec(True, {"survival.fwd": 0.9}), rec(True, {"survival.fwd": 0.8}), rec(True, {"survival.fwd": 0.7})
    recs = {nom("L000", "A"): a, nom("L000", "B"): b, nom("L090", "A"): a90, nom("L000", "Z"): rec(True),
            nom("L090", "B"): rec(True)}
    asked = []

    def cellseed(name, es, term=None, level=None, base="L000", policy="student"):
        asked.append((name, es, term, level, base))
        return recs.get(name)
    monkeypatch.setattr(SR, "cellseed", cellseed)
    assert SR.nominal_refs("L000", 1) == [a, b]
    assert asked == [(nom("L000", "A"), 1, None, None, "L000"), (nom("L000", "B"), 1, None, None, "L000")]   # never the closing Z
    asked.clear()
    assert SR.nominal_refs("L090", 2) == [a90]
    assert asked == [(nom("L090", "A"), 2, None, None, "L090")]                           # one launch at 90 kg
    recs[nom("L000", "B")] = rec(None)                                                    # a launch error is dropped ...
    assert SR.nominal_refs("L000", 1) == [a]
    recs[nom("L000", "A")] = None                                                         # ... and so is an unread launch
    assert SR.nominal_refs("L000", 1) == []


def test_ref_base_agrees_with_the_reference_term_table_uses(quiet):
    table = SR.term_table(make_reads())
    for cell in C.cells():
        assert SR._ref_base(cell) == row(table, cell["term"], cell["base"])["delta_reference"], cell["id"]


def test_term_table_flags_a_drop_only_against_both_nominal_classes(monkeypatch):
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [rec(True, {"field_light_completion": 1.00}),
                                                              rec(True, {"field_light_completion": 0.93})])
    monkeypatch.setattr(SR, "closing_record", lambda: None)
    fills = {cid("T15", 0.1): S(rec(True, {"field_light_completion": 0.91})),     # -0.09 / -0.02: not against both
             cid("T15", 0.2): S(rec(True, {"field_light_completion": 0.86}))}     # -0.14 / -0.07: against both
    r = row(SR.term_table(make_reads(fills)), "T15")
    assert r["drop_flags"] == ["0p2@es1:field_light_completion"]
    assert r["deltas"]["0p1"][1]["field_light_completion"]["flag"] is False


# ---- the terms 13-15 feasibility bound (report only)
def _fz(term, base, outcomes, *, levels=None):
    """``feasibility`` of ``term`` with the level outcomes given in table order (NO = unread)."""
    t = T[term]
    fills = {cid(term, lv, base): v for lv, v in zip(levels or t["levels"], outcomes) if v is not NO}
    return SR.feasibility(t, base, make_reads(fills))


def test_feasibility_is_last_pass_and_first_fail_from_the_nominal_outward():
    got = _fz("T15", "L000", [S(True, True), S(True, True), S(False, False), S(True, True)])
    assert got == {"last_pass": 0.2, "first_fail": 0.35, "split_levels": [], "single_seed_levels": [],
                   "search_stopped": None, "interval": [0.2, 0.35]}                  # the search ends at the first fail
    assert _fz("T15", "L000", [S(True, True)] * 4)["interval"] == [0.5, None]       # nothing failed: only a lower bound
    first = _fz("T15", "L000", [S(False, False)] + [S(True, True)] * 3)
    assert first["last_pass"] is None and first["first_fail"] == 0.1 and first["interval"] == [None, 0.1]
    shuffled = dict(T["T15"], levels=[0.5, 0.1, 0.35, 0.2])                           # table order does not matter
    fills = {cid("T15", 0.1): S(True, True), cid("T15", 0.2): S(True, True), cid("T15", 0.35): S(False, False),
             cid("T15", 0.5): S(False, False)}
    assert SR.feasibility(shuffled, "L000", make_reads(fills))["interval"] == [0.2, 0.35]


def test_a_split_level_is_listed_and_does_not_end_the_search():
    got = _fz("T15", "L000", [S(True, True), S(True, False), S(False, False), S(True, True)])
    assert got["split_levels"] == ["0p2"] and got["last_pass"] == 0.1 and got["first_fail"] == 0.35
    assert got["interval"] == [0.1, 0.35] and got["search_stopped"] is None          # the bound is an interval over the split
    both = _fz("T15", "L000", [S(False, True), S(True, True), S(True, True), S(True, True)])
    assert both["split_levels"] == ["0p1"] and both["first_fail"] is None
    assert both["last_pass"] == 0.5                                                  # a level after a split can still pass


def test_a_launch_error_or_an_unread_level_ends_the_search_and_is_not_a_failure():
    err = _fz("T15", "L000", [S(True, True), S(None), S(False, False), S(False, False)])
    assert err["last_pass"] == 0.1 and err["first_fail"] is None and err["search_stopped"] == "0p2: error"
    assert err["interval"] == [0.1, None]                                            # the fails after the error are not read
    unread = _fz("T15", "L000", [S(True, True), NO, S(False, False), S(False, False)])
    assert unread["last_pass"] == 0.1 and unread["first_fail"] is None and unread["search_stopped"] == "0p2: unread"
    nothing = _fz("T15", "L000", [NO] * 4)
    assert nothing == {"last_pass": None, "first_fail": None, "split_levels": [], "single_seed_levels": [],
                       "search_stopped": "0p1: unread", "interval": [None, None]}
    one_err = _fz("T15", "L000", [S(True, None), NO, NO, NO])                        # a launch error on ONE seed of a level
    assert one_err["search_stopped"] == "0p1: error" and one_err["last_pass"] is None


def test_levels_read_on_one_seed_only_are_listed():
    got = _fz("T15", "L000", [S(True), S(True, True), S(False), S(True)])
    assert got["single_seed_levels"] == ["0p1", "0p35"] and got["last_pass"] == 0.2 and got["first_fail"] == 0.35
    assert _fz("T15", "L000", [S(True, True)] * 4)["single_seed_levels"] == []
    assert _fz("T15", "L000", [S(NO, True), NO, NO, NO])["single_seed_levels"] == ["0p1"]    # an es2-only read is one seed too


@pytest.mark.parametrize("term, base", [("T13", "payload"), ("T14", "L090"), ("T15", "L000")])
def test_term_table_carries_the_feasibility_of_terms_13_to_15_only(quiet, term, base):
    t = T[term]
    fills = {cid(term, lv, base): S(True, True) for lv in t["levels"][:1]}
    fills.update({cid(term, lv, base): S(False, False) for lv in t["levels"][1:2]})
    table = SR.term_table(make_reads(fills))
    f = row(table, term)["feasibility"]
    assert f["last_pass"] == t["levels"][0] and f["first_fail"] == t["levels"][1] and f["interval"] == t["levels"][:2]
    for r in table["rows"]:
        if r["term"] not in ("T13", "T14", "T15"):
            assert r["feasibility"] is None, r["term"]


# ---- the closing flag
def _stamp_dir(tmp_path, name, mtime):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "tp_row.json"
    f.write_text("{}")
    os.utime(f, (mtime, mtime))
    return d


def test_newest_read_time_is_the_newest_tp_row_among_the_stamps_read(tmp_path):
    assert SR.newest_read_time(make_reads()) == 0.0
    a, b, c = (_stamp_dir(tmp_path, n, m) for n, m in (("a", 100.0), ("b", 300.0), ("c", 200.0)))
    nostamp = tmp_path / "incomplete"
    nostamp.mkdir()                                                                   # no tp_row.json: ignored
    reads = make_reads({cid("T15", 0.1): S(rec(True, stamps={"suite": str(a), "field_light": str(nostamp)})),
                        cid("T15", 0.2): S(rec(True, stamps={"suite": str(c), "field_light": str(c)}),
                                           rec(True, stamps={"suite": str(b), "field_light": str(a)})),
                        cid("T03", 0.8): S(rec(True))})                               # no "stamps" key at all
    assert SR.newest_read_time(reads) == 300.0
    reads[cid("T15", 0.2)]["seeds"].pop(2)
    assert SR.newest_read_time(reads) == 200.0


CLOSING = {"status": "match", "stale": False, "stamp_time": 200.0}


@pytest.mark.parametrize("closing, newest, flag", [
    (None, None, True),                                                               # no closing record at all
    (CLOSING, 100.0, False), (CLOSING, 200.0, False), (CLOSING, None, False),         # a fresh match: not flagged
    (CLOSING, 250.0, True),                                                           # a grid stamp newer than the closing
    ({**CLOSING, "stamp_time": None}, None, True), ({**CLOSING, "stale": True}, 100.0, True),
    ({"status": "match", "stamp_time": 200.0}, 100.0, True),                          # "stale" absent counts as stale
    ({**CLOSING, "status": "mismatch"}, 100.0, True), ({**CLOSING, "status": "unread"}, 100.0, True),
    ({**CLOSING, "status": "launch_error"}, 100.0, True),
    ({"match": True, "stale": False, "stamp_time": 200.0}, 100.0, True),              # the old record shape: no status
    ({}, None, True),
], ids=lambda v: repr(v)[:40])
def test_the_closing_flag_is_true_unless_a_fresh_match(monkeypatch, tmp_path, closing, newest, flag):
    fills = {}
    if newest is not None:
        d = _stamp_dir(tmp_path, "grid", newest)
        fills[cid("T15", 0.1)] = S(rec(True, stamps={"suite": str(d), "field_light": str(d)}))
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [])
    monkeypatch.setattr(SR, "closing_record", lambda: closing)
    table = SR.term_table(make_reads(fills))
    assert table["closing_flag"] is flag and table["closing"] == closing


@pytest.mark.parametrize("closing, newest, flag", [
    (None, None, True), (CLOSING, 100.0, False), ({**CLOSING, "status": "mismatch"}, 100.0, True),
    ({**CLOSING, "stale": True}, 100.0, True), (CLOSING, 250.0, True),
], ids=["no closing", "fresh match", "mismatch", "stale as recorded", "stale against the grid"])
def test_every_row_of_the_table_carries_the_closing_flag(monkeypatch, tmp_path, closing, newest, flag):
    """CHARTER 3.5: a closing flag marks EVERY cell -- unread rows, report-only rows and ROBUST rows included."""
    fills = {}
    if newest is not None:
        d = _stamp_dir(tmp_path, "grid", newest)
        fills[cid("T15", 0.1)] = S(rec(True, stamps={"suite": str(d), "field_light": str(d)}))
    fills[cid("T08", 0.5)] = S(True, True)                                            # a ROBUST-looking row among unread ones
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [])
    monkeypatch.setattr(SR, "closing_record", lambda: closing)
    table = SR.term_table(make_reads(fills))
    assert table["closing_flag"] is flag and len(table["rows"]) == 19
    assert [r["closing_flag"] for r in table["rows"]] == [flag] * 19                    # the same bool on every row
    assert {r["term"] for r in table["rows"]} == set(T)                                 # ... report-only terms included
    json.loads(json.dumps(table["rows"], default=str))                                  # and it serialises with the row


def test_the_closing_record_and_the_launch_pin_are_read_from_their_files(monkeypatch, tmp_path):
    monkeypatch.setattr(SR, "CLOSING", tmp_path / "closing.json")
    monkeypatch.setattr(SR, "LAUNCH", tmp_path / "launch.json")
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [])
    assert SR.closing_record() is None and SR.launch_pin() is None
    assert SR.term_table(make_reads())["closing_flag"] is True                          # no closing.json: flagged
    SR.CLOSING.write_text(json.dumps(CLOSING))
    SR.LAUNCH.write_text(json.dumps(PIN))
    assert SR.closing_record() == CLOSING and SR.launch_pin() == PIN
    table = SR.term_table(make_reads())
    assert table["closing"] == CLOSING and table["closing_flag"] is False               # a fresh match (nothing newer read)
    SR.CLOSING.write_text(json.dumps({**CLOSING, "status": "mismatch"}))
    assert SR.term_table(make_reads())["closing_flag"] is True


def test_the_table_serialises_and_renders(quiet, tmp_path, monkeypatch):
    fills = {cid("T08", 0.5): S(True), cid("T08", 1.0): S(True, True),
             cid("T15", 0.1): S(rec(True, {"survival.fwd": 0.7})), cid("T15", 0.2): S(rec(True, {"survival.fwd": 0.7})),
             cid("T13", 150.0, "payload"): S(True), cid("T13", 165.0, "payload"): S(False, False),
             cid("T14", 0.2, "L090"): S(None)}
    reads = make_reads(fills)
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [rec(True)] if base == "L000" else [])
    table = SR.term_table(reads)
    json.dumps({**table, "reads": reads}, default=str)                          # the JSON the driver writes cannot raise
    md = tmp_path / "table.md"
    SR.write_md(table, md)
    text = md.read_text()
    lines = text.splitlines()
    assert lines[0].startswith("| term | base | band | verdict |") and lines[1].startswith("|---|")
    assert sum(1 for ln in lines[2:2 + len(table["rows"])] if ln.startswith("| T")) == len(table["rows"]) == 19
    assert "**Combined verdicts (worst over bases):**" in text and "| T08 obs_noise | L000 | (0.0, 1.0) | ROBUST |" in text
    assert "**CLOSING NOMINAL FLAG** (status absent, stale True)" in text           # no closing record: flagged
    assert "| report-only term | base | last pass | first fail | split | one-seed levels | search stopped |" in text
    assert "| T13 payload_kg | payload | 150.0 | 165.0 | - | 150 | - |" in text
    assert "| T14 payload_com | L090 | None | None | - | - | 0p2: error |" in text
    assert "| T15 push_dv | L000 | 0.2 | None | - | 0p1, 0p2 | 0p35: unread |" in text
    assert "| T15 push_dv 0p1 L000 es1 | L000 (1) |" in text                        # the references used are counted
    assert "| T13 payload_kg 150 payload es1 | L090 (NO REF) |" in text             # ... and a missing one is said
    table["closing"], table["closing_flag"] = {"status": "mismatch", "stale": False}, True
    SR.write_md(table, md)
    assert "**CLOSING NOMINAL FLAG** (status mismatch, stale False)" in md.read_text()
    table["closing_flag"] = False
    SR.write_md(table, md)
    assert "CLOSING NOMINAL FLAG" not in md.read_text()


# =============================================================================== es2_plan
def test_grid_reads_has_one_entry_per_cell_with_the_seeds_that_were_read(monkeypatch):
    read = {(C.gate_name(cid("T08", 1.0)), 1), (C.gate_name(cid("T08", 1.0)), 2), (C.gate_name(cid("T01", 0.7, "L090")), 1)}
    asked = []

    def cellseed(name, es, term=None, level=None, base="L000", policy="student"):
        asked.append((name, es, term["term"], level, base))
        return rec(True) if (name, es) in read else None
    monkeypatch.setattr(SR, "cellseed", cellseed)
    reads = SR.grid_reads()
    assert list(reads) == [c["id"] for c in C.cells()] and len(asked) == 2 * 55
    assert all(reads[c["id"]]["cell"] == c for c in C.cells())
    assert sorted(reads[cid("T08", 1.0)]["seeds"]) == [1, 2] and sorted(reads[cid("T01", 0.7, "L090")]["seeds"]) == [1]
    assert reads[cid("T08", 0.5)]["seeds"] == {} and reads[cid("T01", 0.7, "L000")]["seeds"] == {}
    # each read is asked for with the cell's own term, level and base: the launch-error checks depend on them
    assert (C.gate_name(cid("T01", 0.7, "L090")), 2, "T01", 0.7, "L090") in asked
    assert (C.gate_name(cid("T13", 165.0, "payload")), 1, "T13", 165.0, "payload") in asked


def test_es2_plan_names_band_edges_with_an_es1_read_and_never_a_launch_error():
    fills = {cid("T08", 1.0): S(True),                 # band edge, es1 read -> yes
             cid("T07", 2): S(False),                  # band edge failing -> yes (the failure itself needs es2)
             cid("T03", 0.8): S(None),                 # band edge, launch error -> never
             cid("T03", 1.2): S(),                     # band edge unread -> nothing
             cid("T04", 0.75): S(NO, True)}            # only an es2 read: no es1 -> nothing
    assert SR.es2_plan(make_reads(fills)) == sorted([cid("T08", 1.0), cid("T07", 2)])
    assert SR.es2_plan(make_reads()) == []


def test_es2_plan_adds_in_band_es1_failures_but_not_passes_or_out_of_band_failures():
    fills = {cid("T08", 0.5): S(False),                # in band, not an edge, es1 failed -> yes
             cid("T06", 0.8): S(True),                 # in band, not an edge, es1 passed -> no
             cid("T08", 2.0): S(False), cid("T08", 4.0): S(False),   # beyond the band: es1 only
             cid("T01", 0.5): S(False),                # T01 0.5 is below the (0.7, 1.35) band
             cid("T01", 0.8, "L090"): S(False),        # in band at the other base -> yes
             cid("T05", 0.9): S(None)}                 # a launch error: never
    assert SR.es2_plan(make_reads(fills)) == sorted([cid("T08", 0.5), cid("T01", 0.8, "L090")])


def test_es2_plan_dedups_and_does_not_consult_es2():
    fills = {cid("T08", 1.0): S(False, True), cid("T08", 0.5): S(False, False)}
    plan = SR.es2_plan(make_reads(fills))
    assert plan == sorted([cid("T08", 0.5), cid("T08", 1.0)]) and len(plan) == len(set(plan))


@pytest.mark.parametrize("term, base, outcomes, want", [
    ("T15", "L000", [True, True, False, True], [0.2, 0.35]),          # last pass + first fail, then the search stops
    ("T15", "L000", [True, True, True, True], [0.5]),                 # all pass: only the last passing level
    ("T15", "L000", [False, True, True, True], [0.1]),                # the first level fails: only the first failing one
    ("T15", "L000", [True, None, True, True], [0.1]),                 # a launch error stops the search (not a failure)
    ("T15", "L000", [None, True, True, True], []),
    ("T15", "L000", [True, True, NO, True], [0.2]),                   # unread: the search cannot go past it
    ("T15", "L000", [NO, NO, NO, NO], []),
    ("T15", "L000", [True, False, False, False], [0.1, 0.2]),
    ("T13", "payload", [True, True], [165.0]),
    ("T13", "payload", [True, False], [150.0, 165.0]),
    ("T13", "payload", [False, True], [150.0]),
    ("T14", "L090", [False, False], [0.2]),
    ("T14", "L090", [True, NO], [0.2]),
])
def test_es2_plan_transitions_for_terms_13_to_15(term, base, outcomes, want):
    t = T[term]
    fills = {cid(term, lv, base): S(o) for lv, o in zip(t["levels"], outcomes) if o is not NO}
    assert SR.es2_plan(make_reads(fills)) == sorted(cid(term, lv, base) for lv in want)


def test_es2_plan_searches_from_the_nominal_outward_whatever_the_table_order(monkeypatch):
    t = dict(T["T15"], levels=[0.5, 0.1, 0.35, 0.2])                              # shuffled: sorted by distance from nominal 0
    monkeypatch.setitem(C.TERM, "T15", t)
    fills = {cid("T15", 0.1): S(True), cid("T15", 0.2): S(True), cid("T15", 0.35): S(False), cid("T15", 0.5): S(False)}
    reads = {c["id"]: {"cell": c, "seeds": {}} for c in C.cells()}
    for k, v in fills.items():
        reads[k]["seeds"] = v
    assert SR.es2_plan(reads) == sorted([cid("T15", 0.2), cid("T15", 0.35)])


# =============================================================================== baseline (CHARTER 3.5)
def _fake_baseline(monkeypatch, record, launches):
    """``record``: ``{(base, es): completion}``; ``launches``: ``{(base, launch, es): rec | None}``."""
    monkeypatch.setattr(SR, "record_seed", lambda base, es: rec(True, {"field_light_completion": record[(base, es)]}))
    by_name = {C.gate_name(C.nominal_id(b, la)): (b, la) for b in ("L000", "L090") for la in ("A", "B")}

    def cellseed(name, es, term=None, level=None, base="L000", policy="student"):
        assert term is None and level is None and policy == "student"
        b, la = by_name[name]
        assert b == base
        return launches.get((b, la, es))
    monkeypatch.setattr(SR, "cellseed", cellseed)


RECORD = {("L000", 1): 1.00, ("L000", 2): 0.99, ("L090", 1): 0.95, ("L090", 2): 0.95}      # CHARTER 3.5's record values


def _launches(a=(1.00, 0.99), b=(1.00, 0.99), l090=(0.95, 0.95)):
    out = {}
    for es in (1, 2):
        out[("L000", "A", es)] = rec(True, {"field_light_completion": a[es - 1]})
        out[("L000", "B", es)] = rec(True, {"field_light_completion": b[es - 1]})
        out[("L090", "A", es)] = rec(True, {"field_light_completion": l090[es - 1]})
    return out


def test_a_clean_baseline_does_not_stop(monkeypatch):
    _fake_baseline(monkeypatch, RECORD, _launches())
    res = SR.baseline()
    assert res["stop"] is False and res["split"] is False and res["reasons"] == []
    assert res["record"]["L000@1"] == {"completion": 1.0, "pass": True} and set(res["record"]) == {
        "L000@1", "L000@2", "L090@1", "L090@2"}
    assert set(res["launches"]) == {f"L000_A@{e}" for e in (1, 2)} | {f"L000_B@{e}" for e in (1, 2)} | {
        f"L090_A@{e}" for e in (1, 2)}
    assert res["launches"]["L090_A@2"] == {"pass": True, "failed": [], "completion": 0.95, "launch_error": None}


def test_a_failed_line_or_a_launch_error_on_any_nominal_launch_stops(monkeypatch):
    for key in (("L000", "A", 1), ("L000", "B", 2), ("L090", "A", 1)):
        launches = _launches()
        launches[key] = rec(False)
        _fake_baseline(monkeypatch, RECORD, launches)
        res = SR.baseline()
        assert res["stop"] is True and any("stable90 lines failed" in r and f"{key[0]}_{key[1]}@{key[2]}" in r
                                           for r in res["reasons"]), key
        launches[key] = rec(None)
        _fake_baseline(monkeypatch, RECORD, launches)
        res = SR.baseline()
        assert res["stop"] is True and any("launch error" in r for r in res["reasons"]), key
        assert res["launches"][f"{key[0]}_{key[1]}@{key[2]}"]["launch_error"] == ["synthetic launch error"]
    launches = _launches()
    del launches[("L000", "B", 1)]                                               # a launch that never produced stamps
    _fake_baseline(monkeypatch, RECORD, launches)
    res = SR.baseline()
    assert res["stop"] is True and any("L000_B@1: stamps missing" in r for r in res["reasons"])


def test_the_completion_rule_stops_only_when_every_launch_of_a_base_deviates(monkeypatch):
    # the two 0 kg launches are the documented two classes (1.00 / 0.94 against a record 1.00): one deviates, not a stop
    _fake_baseline(monkeypatch, RECORD, _launches(b=(0.94, 0.99)))
    res = SR.baseline()
    assert res["stop"] is False and res["split"] is True and res["reasons"] == []
    # both deviate by more than 0.04 -> stop, and the reason names the base, the seed and the record
    _fake_baseline(monkeypatch, RECORD, _launches(a=(0.90, 0.99), b=(0.94, 0.99)))
    res = SR.baseline()
    assert res["stop"] is True and len(res["reasons"]) == 1 and "L000@1" in res["reasons"][0] and "deviates" in res["reasons"][0]
    # the deviation is per seed: es2 deviating on both launches stops, es1 fine
    _fake_baseline(monkeypatch, RECORD, _launches(a=(1.0, 0.90), b=(1.0, 0.94)))
    res = SR.baseline()
    assert res["stop"] is True and "L000@2" in res["reasons"][0]
    # a single-launch base (L090): its one launch is "every launch"
    _fake_baseline(monkeypatch, RECORD, _launches(l090=(0.89, 0.95)))
    res = SR.baseline()
    assert res["stop"] is True and "L090@1" in res["reasons"][0] and res["split"] is False
    _fake_baseline(monkeypatch, RECORD, _launches(l090=(1.00, 0.95)))                 # +0.05 above the record: also a stop
    assert SR.baseline()["stop"] is True


@pytest.mark.parametrize("completion, stops", [
    (0.96, False),                  # |0.96 - 1.00| is 0.04000000000000004 in floats: exactly 0.04 is not a deviation
    (1.04, False), (0.9601, False), (0.9599, True), (1.0401, True), (0.90, True), (1.0, False), (0.99, False),
])
def test_a_deviation_of_exactly_0p04_is_not_a_stop(monkeypatch, completion, stops):
    assert SR.COMPLETION_TOL == 0.04
    _fake_baseline(monkeypatch, RECORD, _launches(a=(completion, 0.99), b=(completion, 0.99)))
    assert SR.baseline()["stop"] is stops


def test_the_split_flag_is_a_difference_between_the_two_0_kg_launches_and_never_a_stop(monkeypatch):
    _fake_baseline(monkeypatch, RECORD, _launches())
    assert SR.baseline()["split"] is False
    _fake_baseline(monkeypatch, RECORD, _launches(a=(1.00, 0.99), b=(0.99, 0.99)))
    res = SR.baseline()
    assert res["split"] is True and res["stop"] is False                         # es1 completions differ by 0.01
    _fake_baseline(monkeypatch, RECORD, _launches(a=(1.00, 0.99), b=(1.00, 0.98)))
    assert SR.baseline()["split"] is True                                        # any seed
    launches = _launches()
    del launches[("L000", "B", 1)]
    del launches[("L000", "B", 2)]
    _fake_baseline(monkeypatch, RECORD, launches)
    assert SR.baseline()["split"] is False                                       # nothing to compare


def test_stress_read_main_exit_codes(monkeypatch, tmp_path, capsys):
    out = tmp_path / "b.json"
    monkeypatch.setattr(sys, "argv", ["stress_read.py", "baseline", "--out", str(out)])
    monkeypatch.setattr(SR, "baseline", lambda: {"stop": True, "split": False, "reasons": ["x"]})
    assert SR.main() == 3 and json.loads(out.read_text())["stop"] is True        # exit 3 = STOP
    assert "DRS BASELINE stop=True" in capsys.readouterr().out
    monkeypatch.setattr(SR, "baseline", lambda: {"stop": False, "split": True, "reasons": []})
    assert SR.main() == 0
    monkeypatch.setattr(SR, "grid_reads", lambda: make_reads({cid("T08", 1.0): S(True)}))
    plan = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", ["stress_read.py", "es2plan", "--out", str(plan)])
    assert SR.main() == 0 and json.loads(plan.read_text()) == [cid("T08", 1.0)]
    monkeypatch.setattr(SR, "nominal_refs", lambda base, es: [])
    monkeypatch.setattr(SR, "closing_record", lambda: None)
    table, md = tmp_path / "t.json", tmp_path / "t.md"
    monkeypatch.setattr(sys, "argv", ["stress_read.py", "table", "--out", str(table), "--md", str(md)])
    assert SR.main() == 0
    data = json.loads(table.read_text())
    assert len(data["rows"]) == 19 and "reads" in data and md.read_text().startswith("| term |")


def test_record_seed_reads_the_newest_complete_task4_stamp_of_each_row(monkeypatch, tmp_path):
    assert SR.RECORD_NAMES == {"L000": "T4F19_sel_c28000", "L090": "T4F19_sel_c28000_L090"}
    monkeypatch.setattr(SR, "RECORD", tmp_path)
    for name in SR.RECORD_NAMES.values():
        for sc in ("suite__omni", "field_light__omni"):
            for ts, done in (("2026-10-23_10-00-00", True), ("2026-10-24_10-00-00", True), ("2026-10-25_10-00-00", False)):
                d = tmp_path / name / sc / "seed002" / ts
                d.mkdir(parents=True)
                if done:
                    (d / "tp_row.json").write_text("{}")
    seen = []
    monkeypatch.setattr(SR, "read_stamps", lambda sd, fd, es: seen.append((sd, fd, es)) or {"es": es})
    assert SR.record_seed("L090", 2) == {"es": 2}
    root = tmp_path / "T4F19_sel_c28000_L090"
    assert seen == [(root / "suite__omni" / "seed002" / "2026-10-24_10-00-00",
                     root / "field_light__omni" / "seed002" / "2026-10-24_10-00-00", 2)]   # the incomplete newer stamp is skipped
    SR.record_seed("L000", 2)
    assert seen[1][0].parts[-4] == "T4F19_sel_c28000"


def test_stress_read_main_cellseed_prints_one_record(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(SR, "cellseed", lambda *a: seen.append(a) or {"es": a[1], "pass": True})
    monkeypatch.setattr(sys, "argv", ["stress_read.py", "cellseed", "DRS_X", "2", "--term", "T06", "--level", "0.4",
                                      "--base", "L090", "--policy", "teacher"])
    assert SR.main() == 0
    assert seen == [("DRS_X", 2, T["T06"], 0.4, "L090", "teacher")]
    assert json.loads(capsys.readouterr().out) == {"es": 2, "pass": True}
    monkeypatch.setattr(sys, "argv", ["stress_read.py", "cellseed", "DRS_Y", "1"])
    SR.main()
    assert seen[1] == ("DRS_Y", 1, None, None, "L000", "student")                          # the defaults


# =============================================================================== run_sweep: the closing nominal
Z_NAME, A_NAME, B_NAME = (C.gate_name(C.nominal_id("L000", la)) for la in ("Z", "A", "B"))


def _put_stamp(root, name, scenario, es, mtime, ts="2026-11-06_10-00-00"):
    d = root / "lr" / name / scenario / f"seed{es:03d}" / ts
    d.mkdir(parents=True)
    f = d / "tp_row.json"
    f.write_text("{}")
    os.utime(f, (mtime, mtime))
    return d


def test_the_closing_launches_are_z_to_z9():
    assert RS.CLOSING_LAUNCHES == ("Z", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9")
    assert len({C.gate_name(C.nominal_id("L000", la)) for la in RS.CLOSING_LAUNCHES}) == 9


def test_stamp_time_is_the_newest_tp_row_of_a_cell_seeds_two_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "TP_OUT", tmp_path)
    assert RS._stamp_time("DRS_X", 1) is None
    _put_stamp(tmp_path, "DRS_X", "suite__omni", 1, 100.0)
    assert RS._stamp_time("DRS_X", 1) is None                                    # the field_light row is missing
    _put_stamp(tmp_path, "DRS_X", "field_light__omni", 1, 250.0)
    assert RS._stamp_time("DRS_X", 1) == 250.0 and RS._stamp_time("DRS_X", 2) is None
    incomplete = tmp_path / "lr" / "DRS_Y" / "suite__omni" / "seed001" / "2026-11-06_10-00-00"
    incomplete.mkdir(parents=True)                                               # no tp_row.json: not a stamp
    _put_stamp(tmp_path, "DRS_Y", "field_light__omni", 1, 100.0)
    assert RS._stamp_time("DRS_Y", 1) is None


def _times(monkeypatch, mapping):
    """``_stamp_time`` from a ``{(gate name, es): time}`` table."""
    monkeypatch.setattr(RS, "_stamp_time", lambda name, es: mapping.get((name, es)))


def _closing_rows_valid(monkeypatch, invalid=()):
    """``row_problems`` for the closing rows: valid (``[]``) except the launches in ``invalid``. Returns the calls seen.

    ``latest_closing`` counts a closing only when BOTH its rows pass ``row_problems(...) == []``, so a test that fakes
    the stamp TIMES with ``_times`` has to say what the stamps are worth too."""
    bad = {C.gate_name(C.nominal_id("L000", la)) for la in invalid}
    seen = []

    def row_problems(row_, term, level, base):
        seen.append((row_["name"], row_["scenario"], row_["es"], term, level, base))
        return ["unexpected KRABBY_STRESS_MU='0.4'"] if row_["name"] in bad else []
    monkeypatch.setattr(RS, "row_problems", row_problems)
    return seen


def test_newest_sweep_time_covers_the_nominals_and_the_grid_but_not_the_closing(monkeypatch):
    nom = lambda b, la: C.gate_name(C.nominal_id(b, la))  # noqa: E731
    _times(monkeypatch, {})
    assert RS.newest_sweep_time() == 0.0
    _times(monkeypatch, {(nom("L000", "A"), 1): 100.0, (nom("L000", "B"), 2): 120.0, (nom("L090", "A"), 2): 140.0,
                         (C.gate_name(cid("T06", 0.4)), 2): 300.0, (C.gate_name(cid("T13", 150.0, "payload")), 1): 280.0,
                         (nom("L000", "Z"), 1): 999.0, (nom("L000", "Z2"), 1): 998.0,       # the closing launches are no sweep
                         (nom("L090", "B"), 1): 888.0, (nom("L090", "Z"), 1): 777.0,        # and L090 has only launch A
                         (C.gate_name("T99_not_a_cell"), 1): 666.0})
    assert RS.newest_sweep_time() == 300.0


def test_latest_closing_is_the_newest_valid_closing_stamp_at_es1(monkeypatch):
    z = lambda la, es=1: (C.gate_name(C.nominal_id("L000", la)), es)  # noqa: E731
    seen = _closing_rows_valid(monkeypatch)
    _times(monkeypatch, {})
    assert RS.latest_closing() == (None, None) and seen == []                    # nothing stamped: nothing to judge
    _times(monkeypatch, {z("Z"): 100.0, z("Z2"): 300.0, z("Z3"): 200.0, z("Z4", 2): 900.0})   # Z4 only at es2: not a closing read
    assert RS.latest_closing() == ("Z2", 300.0)
    _times(monkeypatch, {z("Z"): 300.0, z("Z2"): 300.0})
    assert RS.latest_closing() == ("Z", 300.0)                                                 # a tie: the first launch
    _times(monkeypatch, {z("Z"): 100.0})
    seen.clear()
    assert RS.latest_closing() == ("Z", 100.0)                                   # validity is judged on BOTH rows, es1, no cell
    assert seen == [(C.gate_name(C.nominal_id("L000", "Z")), row_, 1, None, None, "L000") for row_ in
                    ("suite__omni", "field_light__omni")]
    # a closing whose stamps fail the launch checks is skipped, however new it is
    _times(monkeypatch, {z("Z"): 100.0, z("Z2"): 300.0, z("Z3"): 200.0})
    _closing_rows_valid(monkeypatch, invalid=("Z2",))
    assert RS.latest_closing() == ("Z3", 200.0)
    _closing_rows_valid(monkeypatch, invalid=("Z", "Z2", "Z3"))
    assert RS.latest_closing() == (None, None)


@pytest.mark.parametrize("closing, sweep, invalid, want", [
    ({}, {}, (), "Z"),                                              # nothing yet: launch Z
    ({"Z": 500.0}, {"newest": 300.0}, (), None),                    # a valid closing newer than every sweep stamp: done
    ({"Z": 300.0}, {"newest": 300.0}, (), None),                    # ... also when equal
    ({"Z": 200.0}, {"newest": 300.0}, (), "Z2"),                    # rows ran after Z: a fresh closing
    ({"Z": 200.0, "Z2": 250.0}, {"newest": 300.0}, (), "Z3"),
    ({"Z": 200.0, "Z2": 500.0}, {"newest": 300.0}, (), None),      # a later closing is the one that counts
    ({"Z2": 200.0}, {"newest": 300.0}, (), "Z"),                    # the first launch NOT yet used
    ({}, {"newest": 300.0}, (), "Z"),                               # sweep rows but no closing at all
    # a closing that fails the launch checks is no closing, and its id stays used up: the next free id runs
    ({"Z": 500.0}, {"newest": 300.0}, ("Z",), "Z2"),
    ({"Z": 500.0}, {}, ("Z",), "Z2"),                               # ... also with no sweep stamp at all
    ({"Z": 500.0, "Z2": 600.0}, {"newest": 300.0}, ("Z2",), None),  # the older VALID Z is still newer than the sweep
    ({"Z": 200.0, "Z2": 600.0}, {"newest": 300.0}, ("Z2",), "Z3"),  # ... and when it is not, a fresh one after both ids
    ({"Z": 500.0, "Z2": 600.0}, {"newest": 300.0}, ("Z", "Z2"), "Z3"),
], ids=lambda v: repr(v)[:28])
def test_next_closing_launch(monkeypatch, closing, sweep, invalid, want):
    _closing_rows_valid(monkeypatch, invalid)
    mapping = {(C.gate_name(C.nominal_id("L000", la)), 1): t for la, t in closing.items()}
    if "newest" in sweep:
        mapping[(C.gate_name(cid("T06", 0.4)), 1)] = sweep["newest"]
    _times(monkeypatch, mapping)
    assert RS.next_closing_launch() == want


@pytest.mark.parametrize("invalid", [(), RS.CLOSING_LAUNCHES], ids=["all valid", "all invalid"])
def test_the_nine_closing_launches_run_out_with_a_stop(monkeypatch, invalid):
    _closing_rows_valid(monkeypatch, invalid)                                    # an invalid closing still uses its id up
    mapping = {(C.gate_name(C.nominal_id("L000", la)), 1): 100.0 + i for i, la in enumerate(RS.CLOSING_LAUNCHES)}
    mapping[(C.gate_name(cid("T06", 0.4)), 1)] = 900.0
    _times(monkeypatch, mapping)
    with pytest.raises(RS.Stop) as exc:
        RS.next_closing_launch()
    assert exc.value.code == 8 and "exhausted" in exc.value.why


def _closing(monkeypatch, tmp_path, *, z=NO, a=NO, b=NO, latest=("Z", 100.0), newest=50.0):
    """``closing_check`` with fake references: ``z`` / ``a`` / ``b`` are ready records (None = never read)."""
    recs = {A_NAME: None if a is NO else a, B_NAME: None if b is NO else b}
    if latest[0] is not None:
        recs[C.gate_name(C.nominal_id("L000", latest[0]))] = None if z is NO else z
    asked = []

    def cellseed(name, es, term=None, level=None, base="L000", policy="student"):
        asked.append((name, es, term, level, base, policy))
        return recs[name]
    monkeypatch.setattr(RS.SR, "cellseed", cellseed)
    monkeypatch.setattr(RS, "latest_closing", lambda: latest)
    monkeypatch.setattr(RS, "newest_sweep_time", lambda: newest)
    monkeypatch.setattr(RS, "RESULTS", tmp_path)
    return asked


def test_the_closing_nominal_matches_when_its_18_lines_equal_launch_a_or_b(monkeypatch, tmp_path):
    za, zb = rec(True, {"survival.fwd": 0.93}), rec(True, {"survival.fwd": 0.95})
    a, b = rec(True, {"survival.fwd": 0.93}), rec(True, {"survival.fwd": 0.95})
    asked = _closing(monkeypatch, tmp_path, z=za, a=a, b=b)
    res = RS.closing_check()
    assert res["status"] == "match" and res["match"] is True and res["matches"] == ["A"] and res["launch"] == "Z"
    assert res["stamp_time"] == 100.0 and res["newest_sweep_time"] == 50.0 and res["stale"] is False
    assert res["z_pass"] is True and res["z_launch_error"] is None
    assert asked[0][:2] == (Z_NAME, 1) and all(c[1] == 1 and c[2:5] == (None, None, "L000") for c in asked)
    assert {c[0] for c in asked} == {Z_NAME, A_NAME, B_NAME}
    assert set(res["z_lines"]) == set(LINE_KEYS) and len(res["z_lines"]) == 18
    assert set(res["ref_lines"]) == {"A", "B"} and res["ref_lines"]["A"] == res["z_lines"]
    assert json.loads((tmp_path / "closing.json").read_text())["status"] == "match"       # and it is recorded
    _closing(monkeypatch, tmp_path, z=zb, a=a, b=b)
    res = RS.closing_check()
    assert res["matches"] == ["B"] and res["status"] == "match"                           # the other outcome class
    _closing(monkeypatch, tmp_path, z=rec(True), a=rec(True), b=rec(True))
    assert RS.closing_check()["matches"] == ["A", "B"]                                    # both classes identical


def test_a_closing_that_differs_even_slightly_is_a_mismatch(monkeypatch, tmp_path):
    a, b = rec(True, {"survival.fwd": 0.93}), rec(True, {"survival.fwd": 0.95})
    _closing(monkeypatch, tmp_path, z=rec(True, {"survival.fwd": 0.93 + 1e-9}), a=a, b=b)
    res = RS.closing_check()
    assert res["status"] == "mismatch" and res["match"] is False and res["matches"] == [] and res["z_pass"] is True
    extra = rec(True)
    extra["lines"]["extra.line"] = {"value": 1.0}
    _closing(monkeypatch, tmp_path, z=extra, a=rec(True), b=rec(True))
    assert RS.closing_check()["status"] == "mismatch"                                     # a different line set
    failing = rec(False, {"survival.fwd": 0.91})                                         # Z fails a line: it is not A / B
    _closing(monkeypatch, tmp_path, z=failing, a=a, b=b)
    res = RS.closing_check()
    assert res["status"] == "mismatch" and res["z_pass"] is False


def test_a_launch_error_reference_is_not_a_match_and_a_launch_error_z_is_its_own_status(monkeypatch, tmp_path):
    same = {"survival.fwd": 0.93}
    _closing(monkeypatch, tmp_path, z=rec(True, same), a=rec(None, same), b=rec(True, same))
    res = RS.closing_check()
    assert res["status"] == "match" and res["matches"] == ["B"]                           # A is a launch error: dropped
    _closing(monkeypatch, tmp_path, z=rec(True, same), a=rec(None, same), b=rec(None, same))
    assert RS.closing_check()["status"] == "mismatch"                                     # no usable reference
    _closing(monkeypatch, tmp_path, z=rec(True, same), a=None, b=rec(True, same))
    res = RS.closing_check()
    assert res["status"] == "match" and res["matches"] == ["B"] and res["ref_lines"]["A"] is None
    _closing(monkeypatch, tmp_path, z=rec(None, same), a=rec(True, same), b=rec(True, same))
    res = RS.closing_check()
    assert res["status"] == "launch_error" and res["match"] is False and res["matches"] == ["A", "B"]
    assert res["z_pass"] is None and res["z_launch_error"] == ["synthetic launch error"]


def test_an_unread_closing_is_a_status_and_stale(monkeypatch, tmp_path):
    asked = _closing(monkeypatch, tmp_path, latest=(None, None), a=rec(True), b=rec(True))
    res = RS.closing_check()
    assert res["status"] == "unread" and res["match"] is False and res["launch"] is None and res["stamp_time"] is None
    assert res["z_pass"] is None and res["z_lines"] is None and res["z_launch_error"] is None and res["stale"] is True
    assert {c[0] for c in asked} == {A_NAME, B_NAME}                                      # no Z was asked for
    _closing(monkeypatch, tmp_path, z=None, a=rec(True), b=rec(True))                      # the launch exists, the read is None
    assert RS.closing_check()["status"] == "unread"


def test_a_closing_older_than_a_sweep_stamp_is_stale_even_when_it_matches(monkeypatch, tmp_path):
    same = rec(True, {"survival.fwd": 0.93})
    for stamp_time, newest, stale in ((100.0, 50.0, False), (100.0, 100.0, False), (100.0, 100.5, True), (100.0, 0.0, False)):
        _closing(monkeypatch, tmp_path, z=same, a=same, b=rec(True), latest=("Z", stamp_time), newest=newest)
        res = RS.closing_check()
        assert res["status"] == "match" and res["stale"] is stale and res["newest_sweep_time"] == newest
    # the launch it read is the LATEST one, whatever its number
    asked = _closing(monkeypatch, tmp_path, z=same, a=same, b=rec(True), latest=("Z3", 100.0))
    res = RS.closing_check()
    assert res["launch"] == "Z3" and asked[0][0] == C.gate_name(C.nominal_id("L000", "Z3"))
    saved = json.loads((tmp_path / "closing.json").read_text())
    assert saved["launch"] == "Z3" and saved["stale"] is False and saved["stamp_time"] == 100.0
    assert set(saved) == {"status", "match", "matches", "launch", "stamp_time", "newest_sweep_time", "stale", "z_pass",
                          "z_launch_error", "z_lines", "ref_lines"}


# =============================================================================== run_sweep: the tree, the rows
def test_the_nominal_phase_is_a_and_b_at_0_kg_and_a_at_l090_each_at_es1_and_es2(monkeypatch):
    calls = []
    monkeypatch.setattr(RS, "run_cell", lambda cell, term, level, base, es, dry: calls.append((cell, term, level, base, es, dry)))
    RS.nominal(False)
    assert calls == [(C.nominal_id(b, la), None, None, b, es, False)
                     for b, la in (("L000", "A"), ("L000", "B"), ("L090", "A")) for es in (1, 2)]
    calls.clear()
    RS.nominal(True)
    assert len(calls) == 6 and all(c[-1] is True for c in calls)


def test_code_state_runs_the_run_candidates_snippet_from_the_tools_dir(monkeypatch):
    calls = []

    def run(cmd, cwd=None, capture_output=None, text=None, check=None):
        calls.append((cmd, cwd, check))
        return SimpleNamespace(stdout='some import noise\n{"code_sha": "c", "eval_sha": "e", "dirty": []}\n')
    monkeypatch.setattr(RS.subprocess, "run", run)
    assert RS.code_state() == {"code_sha": "c", "eval_sha": "e", "dirty": []}                  # the LAST output line is the JSON
    ((cmd, cwd, check),) = calls
    assert cmd[:3] == [RS.PY, "-I", "-c"] and cwd == str(RS.TOOLS) == str(C.EXPERIMENTS / "tools") and check is True
    for needle in ("run_candidates", "CODE_PATHSPECS", "EVAL_PATHSPECS", "code_drop", "code_sha()", "eval_sha()"):
        assert needle in cmd[3], needle                                                         # the gate_row code_state test


def _tree(monkeypatch, tmp_path, state):
    monkeypatch.setattr(RS, "code_state", lambda: state)
    monkeypatch.setattr(SR, "LAUNCH", tmp_path / "launch.json")
    monkeypatch.setattr(RS, "RESULTS", tmp_path)
    monkeypatch.setattr(RS, "LOG", tmp_path / "run_sweep.log")


def test_check_tree_pins_the_hashes_at_the_first_launch_and_refuses_drift(monkeypatch, tmp_path):
    clean = {"code_sha": "c0de", "eval_sha": "e0a1", "dirty": []}
    _tree(monkeypatch, tmp_path, {**clean, "dirty": ["parkour/x.py"]})
    with pytest.raises(RS.Stop) as exc:
        RS.check_tree()
    assert exc.value.code == 5 and "dirty_tree" in exc.value.why and not SR.LAUNCH.exists()    # a dirty tree pins nothing
    _tree(monkeypatch, tmp_path, clean)
    pin = RS.check_tree()                                                       # the first launch writes the pin
    assert pin["code_sha"] == "c0de" and pin["eval_sha"] == "e0a1" and "pinned_at" in pin
    assert json.loads(SR.LAUNCH.read_text()) == pin and "DRS PIN code_sha=c0de" in RS.LOG.read_text()
    assert RS.check_tree() == pin                                               # a resume on the same code is fine
    for drift in ({"code_sha": "beef"}, {"eval_sha": "beef"}):
        _tree(monkeypatch, tmp_path, {**clean, **drift})
        with pytest.raises(RS.Stop) as exc:
            RS.check_tree()
        assert exc.value.code == 5 and "code_changed_since_launch" in exc.value.why
    assert json.loads(SR.LAUNCH.read_text()) == pin                              # and the pin is never rewritten


def _driver(monkeypatch, tmp_path, returncodes=(0,)):
    """Point the driver at tmp_path, pin the tree, and replace ``subprocess.run`` by a recorder."""
    monkeypatch.setattr(C, "TP_OUT", tmp_path / "out")
    monkeypatch.setattr(SR, "LAUNCH", tmp_path / "launch.json")
    SR.LAUNCH.write_text(json.dumps(PIN))
    monkeypatch.setattr(RS, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(RS, "LOG", tmp_path / "results" / "run_sweep.log")
    monkeypatch.setattr(RS, "ROWLOGS", tmp_path / "rowlogs")
    monkeypatch.setattr(RS, "STOP_FILE", tmp_path / "STOP_QUEUE")
    monkeypatch.setattr(RS, "STOP_REQUESTED", [])
    monkeypatch.setattr(RS, "INVOCATION", "20261106T120000")
    trees = []
    monkeypatch.setattr(RS, "check_tree", lambda: trees.append(1) or PIN)
    runs = []
    codes = list(returncodes)

    def run(cmd, env=None, stdout=None, stderr=None):
        runs.append({"cmd": cmd, "env": env})
        return SimpleNamespace(returncode=codes.pop(0) if codes else 0)
    monkeypatch.setattr(RS.subprocess, "run", run)
    log = tmp_path / "results" / "run_sweep.log"
    return SimpleNamespace(runs=runs, trees=trees, out=tmp_path / "out", log=lambda: log.read_text())


def _mu_row(es=1, which=0):
    return C.rows_for("T06_mu_0p4_L000", T["T06"], 0.4, "L000", es)[which]


def test_row_problems_is_none_without_a_stamp_empty_for_a_valid_one_and_a_list_otherwise(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    row_ = _mu_row()
    assert RS.row_problems(row_, T["T06"], 0.4, "L000") is None
    _cell_stamps(drv.out, row_["name"], 1, {"KRABBY_STRESS_MU": "0.4"})
    assert RS.row_problems(row_, T["T06"], 0.4, "L000") == []
    assert RS.row_problems(row_, T["T06"], 0.8, "L000")                              # the wrong level's expectation
    assert RS.row_problems(_mu_row(es=2), T["T06"], 0.4, "L000") is None            # another seed has no stamp
    zd = C.rows_for("T12_zero_depth_1_L000", T["T12"], 1, "L000", 1)[0]
    _cell_stamps(drv.out, zd["name"], 1, {}, zero_depth=False)
    assert "zero_depth" in RS.row_problems(zd, T["T12"], 1, "L000")[0]              # term 12 wants a zero-depth stamp
    teacher = C.rows_for("T06_mu_0p4_L000", T["T06"], 0.4, "L000", 1, "teacher")[0]
    _cell_stamps(drv.out, teacher["name"], 1, {"KRABBY_STRESS_MU": "0.4"}, zero_depth=True)
    assert RS.row_problems(teacher, T["T06"], 0.4, "L000") == []                     # the teacher's zero_depth is not checked
    SR.LAUNCH.unlink()
    assert "no launch pin" in RS.row_problems(row_, T["T06"], 0.4, "L000")[0]        # without a pin no stamp is valid
    _cell_stamps(drv.out, row_["name"], 2, {"KRABBY_STRESS_MU": "0.4"}, rows=("suite__omni",))   # a row is judged on its own
    SR.LAUNCH.write_text(json.dumps(PIN))
    assert RS.row_problems(_mu_row(es=2, which=0), T["T06"], 0.4, "L000") == []
    assert RS.row_problems(_mu_row(es=2, which=1), T["T06"], 0.4, "L000") is None


def test_run_row_scrubs_the_parent_env_and_calls_gate_row_with_its_positional_contract(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    monkeypatch.setenv("KRABBY_LEAK", "1")
    monkeypatch.setenv("KRABBY_STRESS_MU", "0.1")
    monkeypatch.setenv("TP_STRIP_KEYS", "KRABBY_X")
    monkeypatch.setenv("TP_LEVEL_FILE", "/leak.env")
    monkeypatch.setenv("KEEP_ME", "yes")
    row_ = C.rows_for("T01_rod_speed_0p7_L090", T["T01"], 0.7, "L090", 2)[0]
    assert RS.run_row(row_, False, T["T01"], 0.7, "L090") == 0
    (call,) = drv.runs
    assert call["cmd"][:2] == ["bash", str(C.GATE_ROW)]
    assert call["cmd"][2:8] == [row_["name"], row_["checkpoint"], row_["env_file"], row_["manifest"], row_["scenario"], "2"]
    assert call["cmd"][8:] == row_["args"]                                    # gate_row.sh NAME CKPT ENV MANIFEST SCENARIO ES ...
    env = call["env"]
    assert not any(k.startswith(("KRABBY_", "TP_")) and k not in ("TP_OUT", "TP_ROW_TIMEOUT", "TP_LEVEL_FILE") for k in env)
    assert env["TP_OUT"] == str(C.TP_OUT) and env["TP_ROW_TIMEOUT"] == str(RS.ROW_TIMEOUT_S) == "5400"
    assert env["TP_LEVEL_FILE"] == row_["level_file"] != "/leak.env" and "TP_ROW_DRY" not in env and env["KEEP_ME"] == "yes"
    assert drv.trees == [1]                                                     # the tree is checked before the row ...
    assert "DRS ROW" in drv.log() and "rc=0" in drv.log() and "DRS SKIP" not in drv.log() and "DRS RERUN" not in drv.log()
    plain = _mu_row(es=1, which=1)
    RS.run_row(plain, True, T["T06"], 0.4, "L000", attempt=2)
    env = drv.runs[1]["env"]
    assert "TP_LEVEL_FILE" not in env and env["TP_ROW_DRY"] == "1"               # an unarmed base sets no level file
    assert drv.trees == [1]                                                     # ... but not for a dry run


def test_row_logs_carry_the_invocation_and_the_attempt(monkeypatch, tmp_path):
    import datetime
    assert re.fullmatch(r"\d{8}T\d{6}", RS.INVOCATION) and datetime.datetime.strptime(RS.INVOCATION, "%Y%m%dT%H%M%S")
    drv = _driver(monkeypatch, tmp_path)                                         # (which pins INVOCATION to a fixed value)
    row_ = _mu_row()
    RS.run_row(row_, False, T["T06"], 0.4, "L000")
    RS.run_row(row_, False, T["T06"], 0.4, "L000", attempt=2)
    monkeypatch.setattr(RS, "INVOCATION", "20261106T130000")                      # a resume must not overwrite the first logs
    RS.run_row(row_, False, T["T06"], 0.4, "L000")
    stem = f"{row_['name']}.suite__omni.es1"
    assert sorted(f.name for f in (tmp_path / "rowlogs").iterdir()) == [
        f"{stem}.20261106T120000.try1.log", f"{stem}.20261106T120000.try2.log", f"{stem}.20261106T130000.try1.log"]
    assert drv.log().count("DRS ROW") == 3


def test_run_row_skips_only_a_valid_stamp(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    row_ = _mu_row()
    _cell_stamps(drv.out, row_["name"], 1, {"KRABBY_STRESS_MU": "0.4"})
    assert RS.run_row(row_, False, T["T06"], 0.4, "L000") == 0
    assert drv.runs == [] and drv.trees == []                                    # a valid stamp: no launch, no tree check
    assert "DRS SKIP" in drv.log() and "valid stamp exists" in drv.log()
    (tmp_path / "STOP_QUEUE").write_text("")
    RS.STOP_REQUESTED.append("SIGTERM")
    assert RS.run_row(row_, False, T["T06"], 0.4, "L000") == 0 and drv.runs == []     # a skip comes before every stop


@pytest.mark.parametrize("what, kwargs, env, needle", [
    ("a wrong knob value", {}, {"KRABBY_STRESS_MU": "0.8"}, "KRABBY_STRESS_MU"),
    ("a missing knob key", {}, {}, "KRABBY_STRESS_MU"),
    ("a stray stress key", {}, {"KRABBY_STRESS_MU": "0.4", "KRABBY_STRESS_CAM": "0.8"}, "unexpected KRABBY_STRESS_CAM"),
    ("a stray payload key", {}, {"KRABBY_STRESS_MU": "0.4", "KRABBY_PAYLOAD_KG": "90:90"}, "unexpected KRABBY_PAYLOAD_KG"),
    ("a dirty tree", {"code_state": {"dirty": "True", "code_sha": "c0de", "eval_sha": "e0a1"}},
     {"KRABBY_STRESS_MU": "0.4"}, "dirty"),
    ("hashes off the pin", {"code_state": {"dirty": "False", "code_sha": "zzz", "eval_sha": "e0a1"}},
     {"KRABBY_STRESS_MU": "0.4"}, "code_sha"),
    ("a zero_depth mismatch", {"zero_depth": True}, {"KRABBY_STRESS_MU": "0.4"}, "zero_depth"),
])
def test_run_row_reruns_a_stamp_that_fails_the_launch_checks(monkeypatch, tmp_path, what, kwargs, env, needle):
    drv = _driver(monkeypatch, tmp_path)
    row_ = _mu_row()
    _cell_stamps(drv.out, row_["name"], 1, env, **kwargs)
    assert RS.run_row(row_, False, T["T06"], 0.4, "L000") == 0
    assert len(drv.runs) == 1 and drv.trees == [1], what                          # re-run, with the tree check first
    log = drv.log()
    assert "DRS RERUN" in log and needle in log and "DRS SKIP" not in log, what
    assert "stamp fails the launch checks" in log


def test_run_row_reruns_every_stamp_when_there_is_no_launch_pin(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    row_ = _mu_row()
    _cell_stamps(drv.out, row_["name"], 1, {"KRABBY_STRESS_MU": "0.4"})
    SR.LAUNCH.unlink()
    RS.run_row(row_, False, T["T06"], 0.4, "L000")
    assert len(drv.runs) == 1 and "no launch pin" in drv.log()


#: What makes a closing stamp a launch error: ``(what, _cell_stamps kwargs, env_vars it carries)``. A nominal carries no knob.
CLOSING_FAILURES = [
    ("a stray stress key", {}, {"KRABBY_STRESS_MU": "0.4"}),
    ("a stray payload key", {}, {"KRABBY_PAYLOAD_KG": "90:90"}),
    ("a dirty tree", {"code_state": {"dirty": "True", "code_sha": "c0de", "eval_sha": "e0a1"}}, {}),
    ("hashes off the pin", {"code_state": {"dirty": "False", "code_sha": "zzz", "eval_sha": "e0a1"}}, {}),
    ("a zero_depth stamp", {"zero_depth": True}, {}),
]
Z2_NAME = C.gate_name(C.nominal_id("L000", "Z2"))


@pytest.mark.parametrize("what, kwargs, env", CLOSING_FAILURES, ids=[c[0] for c in CLOSING_FAILURES])
def test_a_closing_stamp_that_fails_the_launch_checks_is_skipped_and_its_id_is_used_up(monkeypatch, tmp_path, what, kwargs, env):
    """CHARTER 3.5 ("a row is skipped only when its newest stamp passes the launch checks, otherwise it is re-run"), for
    the closing: an invalid Z is no closing at all, so ``latest_closing`` skips it and the next free id (Z2) is due. It
    used to count as done: the closing phase launched nothing and ``closing_check`` kept reporting launch_error."""
    drv = _driver(monkeypatch, tmp_path)
    zrow = C.rows_for(C.nominal_id("L000", "Z"), None, None, "L000", 1)[0]
    _cell_stamps(drv.out, Z_NAME, 1, env, mtime=500.0, **kwargs)
    assert RS.row_problems(zrow, None, None, "L000"), what                        # the stamp fails the launch checks ...
    assert RS.latest_closing() == (None, None), what                                # ... so it is no closing,
    assert RS.next_closing_launch() == "Z2", what                                   # and the next free id is due
    _cell_stamps(drv.out, Z2_NAME, 1, {}, mtime=600.0)                              # a valid Z2 is the closing from then on
    assert RS.latest_closing() == ("Z2", 600.0) and RS.next_closing_launch() is None, what


def test_a_closing_needs_both_of_its_rows_to_pass_the_launch_checks(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    _cell_stamps(drv.out, Z_NAME, 1, {}, rows=("suite__omni",), mtime=500.0)
    _cell_stamps(drv.out, Z_NAME, 1, {"KRABBY_STRESS_MU": "0.4"}, rows=("field_light__omni",), mtime=500.0)
    assert RS.latest_closing() == (None, None) and RS.next_closing_launch() == "Z2"       # one bad row spoils the closing
    _cell_stamps(drv.out, Z_NAME, 1, {}, rows=("field_light__omni",), mtime=700.0, ts="2026-11-06_11-00-00")
    assert RS.latest_closing() == ("Z", 700.0) and RS.next_closing_launch() is None       # the re-run (newest stamp) is valid
    _cell_stamps(drv.out, A_NAME, 1, {}, mtime=900.0)                                     # sweep rows ran after Z ...
    _cell_stamps(drv.out, Z2_NAME, 1, {}, rows=("suite__omni",), mtime=950.0)             # ... and Z2 is half stamped
    assert RS.latest_closing() == ("Z", 700.0) and RS.next_closing_launch() == "Z2"       # an unfinished Z2 is resumed


def test_closing_check_reads_the_newest_valid_closing_and_never_an_invalid_one(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    (tmp_path / "results").mkdir()
    records = {A_NAME: rec(True, {"survival.fwd": 0.93}), B_NAME: rec(True, {"survival.fwd": 0.95}),
               Z2_NAME: rec(True, {"survival.fwd": 0.93})}
    asked = []

    def cellseed(name, es, term=None, level=None, base="L000", policy="student"):
        asked.append(name)
        return records.get(name)
    monkeypatch.setattr(RS.SR, "cellseed", cellseed)
    _cell_stamps(drv.out, Z_NAME, 1, {"KRABBY_STRESS_MU": "0.4"}, mtime=700.0)           # the newest stamp, a launch error
    res = RS.closing_check()
    assert res["status"] == "unread" and res["launch"] is None and res["stale"] is True and Z_NAME not in asked
    _cell_stamps(drv.out, Z2_NAME, 1, {}, mtime=600.0)
    asked.clear()
    res = RS.closing_check()
    assert res["status"] == "match" and res["launch"] == "Z2" and res["matches"] == ["A"] and res["stamp_time"] == 600.0
    assert res["stale"] is False and Z_NAME not in asked and Z2_NAME in asked
    assert json.loads((tmp_path / "results" / "closing.json").read_text())["launch"] == "Z2"


def test_run_row_stops_for_stop_queue_and_signals_before_the_tree_check_and_the_launch(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    row_ = _mu_row()
    (tmp_path / "STOP_QUEUE").write_text("")
    with pytest.raises(RS.Stop) as exc:
        RS.run_row(row_, False, T["T06"], 0.4, "L000")
    assert exc.value.code == 4 and "STOP_QUEUE" in exc.value.why and drv.runs == [] and drv.trees == []
    (tmp_path / "STOP_QUEUE").unlink()
    RS._on_signal(signal.SIGTERM, None)
    RS._on_signal(signal.SIGINT, None)
    assert RS.STOP_REQUESTED == ["SIGTERM", "SIGINT"]
    with pytest.raises(RS.Stop) as exc:
        RS.run_row(row_, False, T["T06"], 0.4, "L000")
    assert exc.value.code == 4 and exc.value.why == "signal SIGTERM" and drv.runs == [] and drv.trees == []     # the first one
    with pytest.raises(RS.Stop):
        RS.run_row(row_, True, T["T06"], 0.4, "L000")                             # a dry run stops too
    RS.STOP_REQUESTED.clear()
    RS._on_signal(signal.SIGHUP, None)
    assert RS.STOP_REQUESTED == ["SIGHUP"]


def test_a_tree_that_changed_mid_queue_stops_the_next_row_before_it_launches(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    monkeypatch.setattr(RS, "check_tree", lambda: (_ for _ in ()).throw(RS.Stop("dirty_tree ['x.py']", 5)))
    with pytest.raises(RS.Stop) as exc:
        RS.run_row(_mu_row(), False, T["T06"], 0.4, "L000")
    assert exc.value.code == 5 and "dirty_tree" in exc.value.why and drv.runs == []


def test_a_dry_row_never_reads_stamps_or_checks_the_tree(monkeypatch, tmp_path):
    drv = _driver(monkeypatch, tmp_path)
    row_ = _mu_row()
    _cell_stamps(drv.out, row_["name"], 1, {"KRABBY_STRESS_MU": "0.4"})           # a valid stamp exists, a dry run still launches
    assert RS.run_row(row_, True, T["T06"], 0.4, "L000") == 0
    assert len(drv.runs) == 1 and drv.runs[0]["env"]["TP_ROW_DRY"] == "1" and drv.trees == []
    assert "DRS SKIP" not in drv.log()


def test_on_signal_records_the_signal_name(monkeypatch):
    monkeypatch.setattr(RS, "STOP_REQUESTED", [])
    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT"), (signal.SIGHUP, "SIGHUP")):
        RS._on_signal(sig, None)
    assert RS.STOP_REQUESTED == ["SIGTERM", "SIGINT", "SIGHUP"]


def test_run_cell_runs_suite_then_field_light_and_retries_a_failed_row_once(monkeypatch, tmp_path):
    calls = []
    codes = {("suite__omni", 1): 0, ("field_light__omni", 1): 8, ("field_light__omni", 2): 0}

    def run_row(row_, dry, term=None, level=None, base="L000", attempt=1):
        calls.append((row_["scenario"], attempt, dry, term, level, base))
        return codes[(row_["scenario"], attempt)]
    monkeypatch.setattr(RS, "run_row", run_row)
    RS.run_cell("T06_mu_0p4_L000", T["T06"], 0.4, "L000", 1, dry=False)
    assert [c[:3] for c in calls] == [("suite__omni", 1, False), ("field_light__omni", 1, False), ("field_light__omni", 2, False)]
    assert all(c[3:] == (T["T06"], 0.4, "L000") for c in calls)                  # the cell is passed on: stamps are judged by it
    calls.clear()
    codes.update({("suite__omni", 1): 8, ("suite__omni", 2): 8})                  # a second failure leaves the row unread
    RS.run_cell("T06_mu_0p4_L000", T["T06"], 0.4, "L000", 1, dry=False)
    assert [c[:2] for c in calls] == [("suite__omni", 1), ("suite__omni", 2), ("field_light__omni", 1), ("field_light__omni", 2)]
    calls.clear()
    RS.run_cell("T06_mu_0p4_L000", T["T06"], 0.4, "L000", 1, dry=True)           # a dry run never retries
    assert [c[:2] for c in calls] == [("suite__omni", 1), ("field_light__omni", 1)] and all(c[2] for c in calls)


def test_run_cell_writes_the_env_files_of_a_new_nominal_launch_on_demand(monkeypatch, fake_rows):
    monkeypatch.setattr(RS, "run_row", lambda *a, **k: 0)
    (C.CAMPAIGN / "envs").mkdir(parents=True)                                     # prepare() made it before any sweep
    cid_ = C.nominal_id("L000", "Z2")
    assert not C.env_file(cid_, "suite").exists()
    RS.run_cell(cid_, None, None, "L000", 1, dry=True)
    assert C.env_file(cid_, "suite").read_text() == "KRABBY_A=1\nKRABBY_B=2\n"
    assert C.env_file(cid_, "field_light").read_text() == "KRABBY_A=1\nKRABBY_FL=3\n"
    C.env_file(cid_, "suite").write_text("KEEP=1\n")
    RS.run_cell(cid_, None, None, "L000", 1, dry=True)
    assert C.env_file(cid_, "suite").read_text() == "KEEP=1\n"                    # an existing file is never rewritten
    RS.run_cell(C.cell_id(T["T06"], 0.4, "L000"), T["T06"], 0.4, "L000", 1, dry=True)
    assert not C.env_file(C.cell_id(T["T06"], 0.4, "L000"), "suite").exists()    # grid cells' files come from prepare()


# =============================================================================== run_sweep: the phases
def _sweep_harness(monkeypatch, tmp_path, *, baseline=None, closing=None, launch="Z2", table_flag=False):
    monkeypatch.setattr(C, "TP_OUT", tmp_path / "out")
    monkeypatch.setattr(RS, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(RS, "LOG", tmp_path / "results" / "run_sweep.log")
    ran = []
    monkeypatch.setattr(RS, "check_tree", lambda: ran.append(("tree",)) or PIN)
    monkeypatch.setattr(RS, "nominal", lambda dry: ran.append(("nominal", dry)))
    monkeypatch.setattr(RS, "run_cell", lambda cell, term, level, base, es, dry: ran.append((cell, es, dry)))
    monkeypatch.setattr(SR, "baseline", lambda: baseline or {"stop": False, "split": False, "reasons": []})
    monkeypatch.setattr(SR, "grid_reads", lambda: {})
    monkeypatch.setattr(SR, "es2_plan", lambda reads: [cid("T08", 1.0), cid("T07", 2)])
    monkeypatch.setattr(RS, "next_closing_launch", lambda: launch)
    monkeypatch.setattr(RS, "closing_check", lambda: closing or {"status": "match", "launch": launch, "matches": ["A"],
                                                                 "stale": False})
    monkeypatch.setattr(SR, "term_table", lambda reads: {"rows": [], "closing_flag": table_flag})
    monkeypatch.setattr(SR, "write_md", lambda table, path: Path(path).write_text("md\n"))
    return ran


def _sentinels(tmp_path):
    out = []
    for ln in (tmp_path / "results" / "run_sweep.log").read_text().splitlines():
        m = re.match(r"(DRS [A-Z0-9]+)", ln)
        out.append(m.group(1) if m else ln)
    return out


def test_a_full_sweep_writes_the_documented_sentinels_in_order(monkeypatch, tmp_path):
    ran = _sweep_harness(monkeypatch, tmp_path)
    assert RS.sweep(SimpleNamespace(start="nominal", dry=False)) == 0
    assert _sentinels(tmp_path) == ["DRS START", "DRS TREE", "DRS PHASE", "DRS BASELINE", "DRS PHASE", "DRS PHASE", "DRS ES2PLAN",
                                    "DRS PHASE", "DRS CLOSING", "DRS TABLE", "DRS END"]
    log = (tmp_path / "results" / "run_sweep.log").read_text()
    assert "DRS START phases=nominal,grid,es2,closing,table dry=False" in log
    assert "DRS TREE clean code_sha=c0de eval_sha=e0a1" in log and "DRS ES2PLAN n=2" in log
    assert "DRS CLOSING status=match launch=Z2 matches=['A'] stale=False" in log and "DRS FLAG" not in log
    assert "DRS TABLE rows=0 closing_flag=False" in log
    assert ran[0] == ("tree",) and ran[1] == ("nominal", False)
    grid = [r for r in ran if len(r) == 3 and r[1] == 1 and r[0] != C.nominal_id("L000", "Z2")]
    assert [r[0] for r in grid] == [c["id"] for c in C.cells()] and all(r[2] is False for r in grid)       # table order, es1
    assert [r[:2] for r in ran if len(r) == 3 and r[1] == 2] == [(cid("T08", 1.0), 2), (cid("T07", 2), 2)]  # the plan, at es2
    assert ran[-1] == (C.nominal_id("L000", "Z2"), 1, False)                      # the closing launch, then the check
    res = tmp_path / "results"
    assert json.loads((res / "baseline.json").read_text())["stop"] is False
    assert json.loads((res / "es2_plan.json").read_text()) == [cid("T08", 1.0), cid("T07", 2)]
    assert (res / "table.json").exists() and (res / "table.md").read_text() == "md\n"


def test_a_failed_baseline_stops_with_code_3_before_any_grid_row(monkeypatch, tmp_path):
    ran = _sweep_harness(monkeypatch, tmp_path, baseline={"stop": True, "split": False, "reasons": ["L000_A@1: launch error"]})
    with pytest.raises(RS.Stop) as exc:
        RS.sweep(SimpleNamespace(start="nominal", dry=False))
    assert exc.value.code == 3 and exc.value.why == "baseline" and ("nominal", False) in ran
    assert not any(len(r) == 3 for r in ran) and json.loads((tmp_path / "results" / "baseline.json").read_text())["stop"] is True
    assert "DRS BASELINE stop=True" in (tmp_path / "results" / "run_sweep.log").read_text()


@pytest.mark.parametrize("baseline_json, ok", [
    (None, False), ({"stop": True}, False), ({}, False), ({"stop": False}, True), ({"stop": False, "split": True}, True)])
def test_starting_from_a_later_phase_needs_a_passed_baseline(monkeypatch, tmp_path, baseline_json, ok):
    ran = _sweep_harness(monkeypatch, tmp_path)
    (tmp_path / "results").mkdir()
    if baseline_json is not None:
        (tmp_path / "results" / "baseline.json").write_text(json.dumps(baseline_json))
    a = SimpleNamespace(start="es2", dry=False)
    if ok:
        assert RS.sweep(a) == 0 and ("nominal", False) not in ran
    else:
        with pytest.raises(RS.Stop) as exc:
            RS.sweep(a)
        assert exc.value.code == 3 and "baseline_not_passed" in exc.value.why
        assert not any(len(r) == 3 for r in ran)                                     # and no row ran
    a_nominal = SimpleNamespace(start="nominal", dry=False)                          # a start from nominal never needs it
    (tmp_path / "results" / "baseline.json").unlink(missing_ok=True)
    assert RS.sweep(a_nominal) == 0


def test_a_dry_sweep_checks_nothing_and_plans_the_band_edges(monkeypatch, tmp_path):
    ran = _sweep_harness(monkeypatch, tmp_path)
    assert RS.sweep(SimpleNamespace(start="grid", dry=True)) == 0
    assert ("tree",) not in ran
    cells = [r for r in ran if len(r) == 3]
    assert len(cells) == 55 + 21 + 1 and all(r[2] is True for r in cells)
    assert [r[0] for r in cells[:55]] == [c["id"] for c in C.cells()] and {r[1] for r in cells[:55]} == {1}
    edges = [c["id"] for c in C.cells() if c["level"] in C.band_edges(T[c["term"]])]
    assert [r[0] for r in cells[55:76]] == edges and {r[1] for r in cells[55:76]} == {2}
    assert cells[76] == (C.nominal_id("L000", "Z"), 1, True)                         # the dry closing is always launch Z
    log = (tmp_path / "results" / "run_sweep.log").read_text()
    assert "DRS CLOSING" not in log and "DRS TABLE" not in log and log.rstrip().splitlines()[-1].startswith("DRS END")


def test_the_closing_phase_flags_a_mismatch_or_a_stale_closing_and_skips_the_launch_when_none_is_due(monkeypatch, tmp_path):
    ran = _sweep_harness(monkeypatch, tmp_path, closing={"status": "mismatch", "launch": "Z", "matches": [], "stale": False})
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "baseline.json").write_text(json.dumps({"stop": False}))
    RS.sweep(SimpleNamespace(start="closing", dry=False))
    log = (tmp_path / "results" / "run_sweep.log").read_text()
    assert "DRS FLAG closing_mismatch (every cell" in log and "closing_mismatch_stale" not in log
    assert ran[-1][0] == C.nominal_id("L000", "Z2")
    (tmp_path / "results" / "run_sweep.log").unlink()
    ran = _sweep_harness(monkeypatch, tmp_path, launch=None, closing={"status": "match", "launch": "Z", "matches": ["B"],
                                                                      "stale": True})
    RS.sweep(SimpleNamespace(start="closing", dry=False))
    log = (tmp_path / "results" / "run_sweep.log").read_text()
    assert "DRS FLAG closing_match_stale" in log                                      # a stale match is still flagged
    assert not any(len(r) == 3 for r in ran)                                          # next_closing_launch() None: nothing to run
    assert "DRS CLOSING status=match launch=Z matches=['B'] stale=True" in log


# =============================================================================== run_sweep: prepare, main
@pytest.fixture
def ckpt_tree(tmp_path, monkeypatch, fake_rows):
    src = tmp_path / "ckpt"
    (src / "params").mkdir(parents=True)
    (src / "model_28000.pt").write_bytes(bytes(range(256)) * 40)
    (src / "params" / "agent.yaml").write_text("a: 1\nzero_depth: false\nb: 2\n")
    (src / "params" / "env.yaml").write_bytes(b"env: yaml\n")
    monkeypatch.setattr(C, "CKPT", src / "model_28000.pt")
    monkeypatch.setattr(C, "ZERO_DEPTH_DIR", tmp_path / "zd")
    return src


def test_prepare_writes_every_env_and_level_file(ckpt_tree):
    RS.prepare()
    envs = sorted(p.name for p in (C.CAMPAIGN / "envs").iterdir())
    assert len(envs) == 2 * (6 + 55)                                              # A, B, Z at both bases + the 55 cells, two rows
    for cell in C.cells():
        for rw in C.ROWS:
            assert C.env_file(cell["id"], rw).read_text() == C.env_text(T[cell["term"]], cell["level"], rw), cell["id"]
    for base in ("L000", "L090"):
        for la in ("A", "B", "Z"):
            assert C.env_file(C.nominal_id(base, la), "suite").read_text() == "KRABBY_A=1\nKRABBY_B=2\n"
    levels = sorted(p.name for p in (C.CAMPAIGN / "levels").iterdir())
    assert levels == sorted(["L150.env", "L165.env", "L090_comxy0p2.env", "L090_comxy0p3.env"])
    assert (C.CAMPAIGN / "levels" / "L165.env").read_text() == C.level_text(T["T13"], 165.0)
    assert (C.CAMPAIGN / "levels" / "L090_comxy0p3.env").read_text() == C.level_text(T["T14"], 0.3)


def test_prepare_builds_the_zero_depth_copy_and_verifies_it_by_sha256(ckpt_tree, monkeypatch):
    zd = C.ZERO_DEPTH_DIR
    RS.prepare()
    assert (zd / "model_28000.pt").read_bytes() == (ckpt_tree / "model_28000.pt").read_bytes()
    assert (zd / "params" / "env.yaml").read_bytes() == b"env: yaml\n"
    assert (zd / "params" / "agent.yaml").read_text() == "a: 1\nzero_depth: true\nb: 2\n"        # the one flag flipped
    assert not (zd / "model_28000.pt").is_symlink()
    assert RS._sha256(zd / "model_28000.pt") == hashlib.sha256((ckpt_tree / "model_28000.pt").read_bytes()).hexdigest()
    copies = []
    real = RS.shutil.copy2
    monkeypatch.setattr(RS.shutil, "copy2", lambda *a, **k: copies.append(a) or real(*a, **k))
    RS.prepare()
    assert copies == []                                                           # a verified copy is left alone
    ck = zd / "model_28000.pt"
    data = bytearray(ck.read_bytes())
    data[100] ^= 0xFF                                                             # same size, one byte different
    ck.write_bytes(bytes(data))
    RS.prepare()
    assert len(copies) == 2                                                       # (the size alone would not catch it)
    assert ck.read_bytes() == (ckpt_tree / "model_28000.pt").read_bytes()
    copies.clear()
    env = zd / "params" / "env.yaml"
    env.write_bytes(b"env: yaMl\n")                                               # the env.yaml copy is compared by content too
    RS.prepare()
    assert env.read_bytes() == b"env: yaml\n" and len(copies) == 2
    copies.clear()
    (zd / "params" / "agent.yaml").write_text("a: 1\nzero_depth: false\nb: 2\n")  # and agent.yaml must carry the flipped flag
    RS.prepare()
    assert "zero_depth: true" in (zd / "params" / "agent.yaml").read_text()
    ck.unlink()
    ck.symlink_to(ckpt_tree / "model_28000.pt")                                    # a symlink is not a copy
    RS.prepare()
    assert ck.is_file() and not ck.is_symlink() and ck.read_bytes() == (ckpt_tree / "model_28000.pt").read_bytes()


@pytest.mark.parametrize("agent", ["a: 1\n", "zero_depth: false\nzero_depth: false\n"])
def test_prepare_refuses_an_agent_yaml_without_exactly_one_zero_depth_flag(ckpt_tree, agent):
    (ckpt_tree / "params" / "agent.yaml").write_text(agent)
    with pytest.raises(RuntimeError, match="exactly one 'zero_depth: false'"):
        RS.prepare()


def _main(monkeypatch, tmp_path, argv, sweep=lambda a: 0, prepare=lambda: None):
    pytest.importorskip("numpy")                                                  # main() fails fast without the reader's numpy
    shim = SimpleNamespace(signal=lambda sig, handler: registered.append((sig, handler)), SIGTERM=signal.SIGTERM,
                           SIGINT=signal.SIGINT, SIGHUP=signal.SIGHUP, Signals=signal.Signals)
    registered = []
    monkeypatch.setattr(RS, "signal", shim)                                       # (not the real signal module: pytest's own)
    monkeypatch.setattr(RS, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(RS, "LOG", tmp_path / "results" / "run_sweep.log")
    monkeypatch.setattr(RS, "prepare", prepare)
    monkeypatch.setattr(RS, "sweep", sweep)
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", *argv])
    return registered


def test_main_registers_the_stop_signals_and_returns_the_sweeps_code(monkeypatch, tmp_path):
    seen = []
    registered = _main(monkeypatch, tmp_path, ["--from", "grid", "--dry"], sweep=lambda a: seen.append(a) or 0)
    assert RS.main() == 0
    assert {sig for sig, _ in registered} == {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    assert all(handler is RS._on_signal for _, handler in registered)
    assert seen[0].start == "grid" and seen[0].dry is True


def test_main_turns_a_stop_into_its_code_and_a_crash_into_7(monkeypatch, tmp_path):
    def stop(a):
        raise RS.Stop("baseline", 3)
    _main(monkeypatch, tmp_path, [], sweep=stop)
    assert RS.main() == 3 and "DRS STOP baseline" in RS.LOG.read_text()

    def crash(a):
        raise RuntimeError("boom")
    _main(monkeypatch, tmp_path, [], sweep=crash)
    assert RS.main() == 7 and "DRS STOP crash RuntimeError: boom" in RS.LOG.read_text()
    assert "RuntimeError: boom" in (tmp_path / "results" / "crash.txt").read_text()
    _main(monkeypatch, tmp_path, ["--prepare-only"], sweep=crash)                       # --prepare-only never sweeps
    assert RS.main() == 0 and "DRS PREPARED" in RS.LOG.read_text()


def test_main_refuses_to_start_while_another_driver_holds_the_lock(monkeypatch, tmp_path, capsys):
    _main(monkeypatch, tmp_path, [])
    (tmp_path / "results").mkdir()
    with open(tmp_path / "results" / "run_sweep.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert RS.main() == 6
    assert "holds results/run_sweep.lock" in capsys.readouterr().out
    assert RS.main() == 0                                                              # released: it starts
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", "--from", "bogus"])
    with pytest.raises(SystemExit):
        RS.main()


def test_the_driver_and_the_reader_share_one_cells_module():
    assert RS.C is C and RS.SR is SR and SR.C is C
    assert RS.PHASES == ("nominal", "grid", "es2", "closing", "table")
    assert RS.RESULTS == SR.RESULTS and SR.LAUNCH == SR.RESULTS / "launch.json" and SR.CLOSING == SR.RESULTS / "closing.json"
