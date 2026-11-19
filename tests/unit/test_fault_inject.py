"""Fault-injection sweep (2026-11-15), on CPU: the knobs, the injector, the action-term hooks, the cell table.

``experiments/2026-11-15_fault_injection/CHARTER.md`` (sections 3, 3.6, 5.1, 6) is the spec. Every failure mode here is silent in a
real run (a fault that is armed but inert reads as "the policy tolerates it"; an onset table that moves with the cell breaks
the same-onset pairing every verdict rests on; a fault state that survives a reset leaks into the next episode), so each is
pinned without Isaac:

* A. ``fault_knobs.py``: every ``cells.yaml`` cell parses (id, level, duration, window), the refusals, knob-off writes nothing,
  armed writes exactly ``cfg.fault_inject`` + one ``[KNOB]`` line, the cfg checks, the module is pure.
* B. the onset table: pure in (window, dt, N, seed), independent of the cell, the earliest onsets of the four rows, the
  private generator (the global RNG is never touched), the dose seed.
* C. per fault family on fake tensors: pre-onset deviation 0, active exactly on ``[onset, onset + d)``, off after a transient,
  the dose of each id (S1-S12, D1, Z1, B1-B6, A1-A5). The action-term methods ``process_actions`` and ``_apply_pitch_linkage`` are
  exec'd from ``parkour_actions.py`` itself on a stand-in term (that file imports Isaac), as ``test_rod_speed_dr.py`` does.
* D. reset semantics: all state cleared, a mid-wave reset disarms (never redraws), the wave counter, ``check_wave``.
* E. the sidecar (``fault_onsets/v1``): keys, strict JSON, one flush.
* F. the wiring (AST / text pins of the hooks in the files that import Isaac) and the depth hook exec'd on a stand-in.
* G. ``cells.yaml`` / ``scripts/make_envs.py --check`` / every ``envs/*.env``.

The depth image is a stand-in (``raw / 2 - 0.5``: 0 m -> -0.5, the CHARTER's S10 image); ``process_depth_batch`` is not run.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour" / "parkour_tasks" / "parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task" / "mdp"
CAMPAIGN = OMNI / "experiments" / "2026-11-15_fault_injection"
FAULT_KNOBS_PY = OMNI / "config" / "crab_hex" / "fault_knobs.py"
OMNI_KNOBS_PY = OMNI / "config" / "crab_hex" / "omni_knobs.py"
OMNI_OBS_PY = OMNI / "mdp" / "crab_hex_omni_observations.py"
FAULT_INJECT_PY = FWD_MDP / "fault_inject.py"
ACTIONS_PY = FWD_MDP / "parkour_actions.py"
OBSERVATIONS_PY = FWD_MDP / "observations.py"
EVAL_PY = TASKS / "crab_hex_forward_task" / "scripts" / "eval_crab_hex_gait.py"
CELLS_YAML = CAMPAIGN / "cells.yaml"
ENVS_DIR = CAMPAIGN / "envs"
MAKE_ENVS_PY = CAMPAIGN / "scripts" / "make_envs.py"


def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


fk = _load(FAULT_KNOBS_PY, "fault_knobs__fault_test")
fi = _load(FAULT_INJECT_PY, "fault_inject__fault_test")
if str(FWD_MDP) not in sys.path:                      # the flat-import convention of the linkage / clock unit tests
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_clock_reward as clock  # noqa: E402
import crab_hex_linkage as linkage  # noqa: E402
# the window limits ``rod_dlam`` imports lazily from the package (which needs Isaac): preset on THIS private copy
fi._LIMITS = {"hip": (linkage.HIP_LEN_MIN_M, linkage.HIP_LEN_MAX_M),
              "knee": (linkage.KNEE_LEN_MIN_M, linkage.KNEE_LEN_MAX_M)}

DT = 0.02
SUB = 4                      # physics substeps per policy step (0.005 s each)
PHYS_DT = 0.005
ONS = "0.2:0.4"              # onset steps 10..19
ROW_WINDOWS = {"suite": ("7.5:11.5", 375), "field_light": ("10.0:60.0", 500),
               "ledge_down": ("9.6:11.0", 480), "pits": ("9.6:13.4", 480)}
CAMERAS = ("front", "left", "right", "rear")


def _yaml():
    return pytest.importorskip("yaml")


def _cells_doc() -> dict:
    return _yaml().safe_load(CELLS_YAML.read_text())


def _raises(fn, exc, *a, **k) -> str:
    with pytest.raises(exc) as ei:
        fn(*a, **k)
    return str(ei.value)


def _env(cell: str | None = None, onset: str | None = ONS, **extra) -> dict:
    env = {}
    if cell is not None:
        env["KRABBY_FAULT"] = cell
        if cell.split(":")[0] in ("S12", "D1"):
            env["KRABBY_FAULT_FROM_START"] = "1"
    if onset is not None:
        env["KRABBY_FAULT_ONSET"] = onset
    env.update(extra)
    return env


# =============================================================================== A. the knob module
def test_the_knob_module_and_the_injector_are_pure():
    def top_level(path):
        mods = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        return mods

    assert top_level(FAULT_KNOBS_PY) <= {"__future__", "math", "os"}               # omni_knobs path-loads it Isaac-free
    assert top_level(FAULT_INJECT_PY) <= {"__future__", "json", "math", "pathlib", "torch"}
    # config modules may not spell a stress variable (test_dr_stress_knobs scans them): the caller passes them in
    assert "KRABBY_STRESS_" not in FAULT_KNOBS_PY.read_text()


def test_names_agree_between_knobs_injector_and_observation_term():
    assert (fk.FAULT_VAR, fk.ONSET_VAR, fk.FROM_START_VAR) == ("KRABBY_FAULT", "KRABBY_FAULT_ONSET",
                                                               "KRABBY_FAULT_FROM_START")
    assert fk.FAULT_VARS == (fk.FAULT_VAR, fk.ONSET_VAR, fk.FROM_START_VAR)
    assert set(fk.FAULT_KNOB_PARSERS) == set(fk.FAULT_VARS)
    assert fk.FAULT_ATTR == fi.FAULT_ATTR == "fault_inject"
    assert fi.ENV_ATTR == "_fault_injector"
    assert '"_fault_injector"' in OMNI_OBS_PY.read_text()                         # the depth term reads the env by literal
    assert math.isclose(fk.S4_SIGMA, 0.2 / math.sqrt(1000.0)) and abs(fk.S4_SIGMA - 0.00632) < 1e-5   # CHARTER 3.3 S4
    assert fk.A5_PAYLOAD_KG == 90.0 and fk.LEGS == {"front": "FL", "mid": "ML", "rear": "RL"}    # left side only
    assert fk.DEPTH_FAULTS == ("S10", "S11", "Z1") and fk.DELAY_FAULTS == ("B3", "B5")


def test_every_catalog_fault_has_an_injector_kind():
    for fid, entry in fk.FAULTS.items():
        for level, (_dur, params) in entry["levels"].items():
            kind = params.get("as", fid)
            assert kind in fi.KNOWN_KINDS, (fid, level)
    assert fk.FAULTS["A5"]["levels"]["mid"][1]["as"] == "A1"                       # A5 = A1 mid at 90 kg


def test_the_catalog_is_the_charter_48_plus_2_anchors():
    n = sum(len(e["levels"]) for e in fk.FAULTS.values())
    assert n == 54 and len(fk.FAULTS) == 23   # 48 + the 2026-11-18 follow-up 6
    sizes = {"S1": 4, "S2": 5, "S3": 2, "S4": 1, "S5": 3, "S6": 2, "S8": 2, "S10": 3, "S11": 2, "Z1": 5, "B1": 4, "B2": 2,
             "B3": 1, "B4": 2, "B5": 1, "B6": 4, "A1": 3, "A2": 3, "A3": 1, "A4": 1, "A5": 1, "S12": 1, "D1": 1}
    assert {k: len(v["levels"]) for k, v in fk.FAULTS.items()} == sizes
    assert {k for k, v in fk.FAULTS.items() if v.get("from_start")} == {"S12", "D1"}


def test_cells_yaml_is_the_charter_50_and_every_fault_cell_parses_on_every_row():
    doc = _cells_doc()
    cells = doc["cells"]
    assert len(cells) == 56 and len({c["id"] for c in cells}) == 56   # + 6 follow-up
    seen = set()
    for cell in cells:
        if not cell.get("fault"):
            continue                                                                # T10b / T12: anchors without a fault knob
        for row, spec_row in doc["rows"].items():
            env = {"KRABBY_FAULT": cell["fault"], "KRABBY_FAULT_ONSET": spec_row["onset"]}
            if cell.get("from_start"):
                env["KRABBY_FAULT_FROM_START"] = "1"
            spec = fk.parse_fault_knobs(env)
            fid, level = cell["fault"].split(":")
            assert spec["id"] == fid and spec["knob"] == cell["fault"], (cell["id"], row)
            assert spec["onset_s"] == [float(x) for x in spec_row["onset"].split(":")]
            assert spec["from_start"] == bool(cell.get("from_start"))
            # the cell table's own duration columns agree with the catalog
            if "transient_s" in cell:
                assert spec["duration_s"] == cell["transient_s"] and not spec["permanent"], cell["id"]
            elif "transient_steps" in cell:
                assert spec["duration_steps"] == cell["transient_steps"] and not spec["permanent"], cell["id"]
            else:
                assert spec["permanent"] and spec["duration_s"] is None and spec["duration_steps"] is None, cell["id"]
        seen.add(cell["fault"])
    assert len(seen) == 54
    catalog = {f"{fid}:{lvl}" for fid, e in fk.FAULTS.items() for lvl in e["levels"]}
    assert seen == catalog                                                          # nothing in the catalog is ungridded
    # the anchors without a knob arm nothing in this module
    assert fk.parse_fault_knobs({"KRABBY_STRESS_VXY_ZERO": "1"}) is None
    for smk in doc.get("smokes") or []:
        assert fk.parse_fault_knobs({"KRABBY_FAULT": smk["fault"], "KRABBY_FAULT_ONSET": smk["onset"]})["id"] == "S1"


def test_parse_forms_and_types():
    spec = fk.parse_fault_knobs(_env("S1:0.5", "7.5:11.5"))
    assert spec["id"] == "S1" and spec["level"] == "0.5" and spec["family"] == "sensor"
    assert spec["duration_s"] == 0.5 and spec["duration_steps"] is None and not spec["permanent"]
    assert spec["onset_s"] == [7.5, 11.5] and spec["from_start"] is False and spec["params"] == {}
    assert fk.parse_fault_knobs(_env("s1:PERM"))["permanent"] is True              # case-insensitive
    assert fk.parse_fault_knobs(_env("S10:2.0"))["level"] == "2"                   # numeric levels compare as numbers
    assert fk.parse_fault_knobs(_env("S1:.5"))["level"] == "0.5"
    assert fk.parse_fault_knobs(_env("B1:25"))["duration_steps"] == 25
    assert fk.parse_fault_knobs(_env(" S1:2 "))["level"] == "2"      # outer blanks only
    assert fk.parse_fault_knobs(_env("S3:0.2"))["params"] == {"bias": 0.2}
    assert fk.parse_fault_knobs(_env("B2:0.05"))["params"] == {"p": 0.05}
    assert fk.parse_fault_knobs(_env("B3:perm"))["params"] == {"delays": [1, 2, 3]}
    assert fk.parse_fault_knobs(_env("B5:perm"))["params"] == {"delay": 2}
    assert fk.parse_fault_knobs(_env("A3:mid"))["params"] == {"leg": "ML", "slew_scale": 0.3}
    assert fk.parse_fault_knobs(_env("A1:rear"))["params"] == {"leg": "RL"}
    assert fk.parse_fault_knobs(_env("S12:30"))["from_start"] is True
    spec = fk.parse_fault_knobs(_env("S1:2"))
    spec["params"]["x"] = 1                                                         # the spec is a copy of the catalog
    assert "x" not in fk.FAULTS["S1"]["levels"]["2"][1]
    json.dumps(spec)                                                                # the spec is a plain, JSON dict
    assert fk.parse_fault_knobs({}) is None and fk.parse_fault_knobs({"KRABBY_OTHER": "1"}) is None
    assert fk.parse_fault_knobs({"KRABBY_FAULT": " ", "KRABBY_FAULT_ONSET": "", "KRABBY_FAULT_FROM_START": "\t"}) is None


def test_the_duration_conversion_is_charters_steps():
    """CHARTER 3.2: durations in steps at dt 0.02: 0.1 s = 5, 0.5 s = 25, 2 s = 100."""
    for knob, steps in (("S1:0.1", 5), ("S1:0.5", 25), ("S1:2", 100), ("B4:0.1", 5), ("B6:0.5", 25), ("B1:2", 2),
                        ("B1:25", 25)):
        inj = fi.FaultInjector(fk.parse_fault_knobs(_env(knob)), num_envs=2, step_dt=DT, device="cpu", env_seed=1)
        assert inj.duration == steps, knob
    inj = fi.FaultInjector(fk.parse_fault_knobs(_env("S1:perm")), num_envs=2, step_dt=DT, device="cpu", env_seed=1)
    assert inj.duration is None


@pytest.mark.parametrize("environ, why", [
    ({"KRABBY_FAULT": "S9:1"}, "unknown id"),
    ({"KRABBY_FAULT": "S1:0.3"}, "unknown level"),
    ({"KRABBY_FAULT": "S1:"}, "empty level"),
    ({"KRABBY_FAULT": "S1"}, "no level"),
    ({"KRABBY_FAULT": ":perm"}, "no id"),
    ({"KRABBY_FAULT": "S1:0.5:2"}, "two colons"),
    ({"KRABBY_FAULT": "S1:nan"}, "nan level"),
    ({"KRABBY_FAULT": "S1:0.5,S2:2"}, "two cells (comma)"),
    ({"KRABBY_FAULT": "S1:0.5;S2:2"}, "two cells (semicolon)"),
    ({"KRABBY_FAULT": "S1:0.5 S2:2"}, "two cells (space)"),
    ({"KRABBY_FAULT": "S1:0.5+S2:2"}, "two cells (plus)"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT2": "S2:2"}, "second KRABBY_FAULT* variable"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_LEVEL": "2"}, "unknown KRABBY_FAULT* variable"),
    ({"KRABBY_FAULTS": "S1:0.5"}, "misspelled variable alone"),
    ({"KRABBY_FAULT_ONSETS": "1:2"}, "misspelled onset variable alone"),
    ({"KRABBY_FAULT": "S1:0.5"}, "armed with no onset"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": ""}, "blank onset"),
    ({"KRABBY_FAULT_ONSET": "7.5:11.5"}, "onset without a fault"),
    ({"KRABBY_FAULT_FROM_START": "1"}, "from-start without a fault"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "7.5:11.5", "KRABBY_FAULT_FROM_START": "1"}, "FROM_START on a non-anchor"),
    ({"KRABBY_FAULT": "B1:5", "KRABBY_FAULT_ONSET": "7.5:11.5", "KRABBY_FAULT_FROM_START": "1"}, "FROM_START on B1"),
    ({"KRABBY_FAULT": "S12:30", "KRABBY_FAULT_ONSET": "7.5:11.5"}, "anchor S12 without FROM_START"),
    ({"KRABBY_FAULT": "D1:1", "KRABBY_FAULT_ONSET": "7.5:11.5"}, "anchor D1 without FROM_START"),
    ({"KRABBY_FAULT": "S12:30", "KRABBY_FAULT_ONSET": "7.5:11.5", "KRABBY_FAULT_FROM_START": "0"}, "anchor with FROM_START=0"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "7.5:11.5", "KRABBY_FAULT_FROM_START": "maybe"}, "garbage flag"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "7.5"}, "onset without a colon"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "a:b"}, "onset not a number"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "5:5"}, "empty window"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "9:5"}, "reversed window"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "-1:5"}, "negative lo"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "5:inf"}, "infinite hi"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "nan:5"}, "nan lo"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "1:3601"}, "beyond the typo guard"),
    ({"KRABBY_FAULT": "S1:0.5", "KRABBY_FAULT_ONSET": "1:2:3"}, "three-part window"),
])
def test_refusals(environ, why):
    msg = _raises(fk.parse_fault_knobs, ValueError, environ)
    assert "KRABBY_FAULT" in msg or "fault knob" in msg, (why, msg)                 # names the variable


def test_stress_plus_fault_is_refused_and_a_blank_stress_variable_is_not():
    stress = ("KRABBY_STRESS_MU", "KRABBY_STRESS_VXY_ZERO")
    env = _env("S1:0.5", "7.5:11.5", KRABBY_STRESS_MU="0.5")
    msg = _raises(fk.parse_fault_knobs, ValueError, env, exclusive_with=stress)
    assert "KRABBY_STRESS_MU" in msg and "exclusive" in msg
    assert fk.parse_fault_knobs(_env("S1:0.5", "7.5:11.5", KRABBY_STRESS_MU=""), exclusive_with=stress)["id"] == "S1"
    assert fk.parse_fault_knobs(_env("S1:0.5", "7.5:11.5"), exclusive_with=stress)["id"] == "S1"
    # a stress knob alone is not this module's business
    assert fk.parse_fault_knobs({"KRABBY_STRESS_MU": "0.5"}, exclusive_with=stress) is None
    # the anchor path (T10b) = the stress knob WITHOUT a fault; with a fault it is refused
    assert _raises(fk.apply_fault_knobs, ValueError, NS(), _env("S12:30", "7.5:11.5", KRABBY_STRESS_VXY_ZERO="1"),
                   exclusive_with=stress)


def test_the_refusal_precedes_any_write():
    class Strict:
        def __setattr__(self, k, v):
            raise AssertionError(f"wrote {k}")

    for environ in ({"KRABBY_FAULT": "S1:0.5"}, {"KRABBY_FAULT": "S9:1", "KRABBY_FAULT_ONSET": "1:2"},
                    _env("S1:0.5", "1:2", KRABBY_STRESS_MU="1")):
        with pytest.raises(ValueError):
            fk.apply_fault_knobs(Strict(), environ, exclusive_with=("KRABBY_STRESS_MU",))


def test_knob_off_writes_nothing_and_prints_nothing(capsys):
    class Strict:
        def __setattr__(self, k, v):
            raise AssertionError(f"knob-off wrote {k}")

    for environ in ({}, {"KRABBY_OTHER": "1"}, {"KRABBY_STRESS_MU": "0.5"}, {"KRABBY_FAULT": ""}):
        assert fk.apply_fault_knobs(Strict(), environ, exclusive_with=("KRABBY_STRESS_MU",)) is None
    cfg = NS(actions=NS(joint_pos=NS(use_delay=True, history_length=8)))
    before = copy.deepcopy(vars(cfg))
    assert fk.apply_fault_knobs(cfg, {}) is None
    assert vars(cfg).keys() == before.keys() and not hasattr(cfg, "fault_inject")
    assert capsys.readouterr().out == ""


def test_armed_writes_exactly_one_attribute_and_one_knob_line(capsys):
    writes = []

    class Rec(NS):
        def __setattr__(self, k, v):
            writes.append(k)
            super().__setattr__(k, v)

    cfg = Rec(actions=NS(joint_pos=NS(use_delay=True, history_length=8)))
    writes.clear()
    spec = fk.apply_fault_knobs(cfg, _env("B3:perm", "7.5:11.5"))
    assert writes == ["fault_inject"] and cfg.fault_inject is spec and spec["id"] == "B3"
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("[KNOB] KRABBY_FAULT=B3:perm") and "cfg.fault_inject" in lines[0]
    assert "permanent" in lines[0] and "U(7.5, 11.5)" in lines[0]


def test_cfg_checks():
    def cfg(use_delay=True, hist=8, depth=True, kg=None):
        obs = NS(**{g: NS(depth_image=object()) for g in fk.DEPTH_GROUPS}) if depth else NS()
        ev = NS(payload_reset=NS(params={"kg_range": kg})) if kg is not None else NS()
        return NS(actions=NS(joint_pos=NS(use_delay=use_delay, history_length=hist)), observations=obs, events=ev)

    on = "7.5:11.5"
    # no action term
    assert _raises(fk.apply_fault_knobs, RuntimeError, NS(), _env("S1:perm", on))
    # delay faults need use_delay and a >= 4 deep history
    for knob in ("B3:perm", "B5:perm"):
        assert "use_delay" in _raises(fk.apply_fault_knobs, RuntimeError, cfg(use_delay=False), _env(knob, on))
        assert "history_length" in _raises(fk.apply_fault_knobs, RuntimeError, cfg(hist=3), _env(knob, on))
        assert fk.apply_fault_knobs(cfg(hist=4), _env(knob, on))["id"] == knob[:2]
    # depth faults need the four depth groups
    for knob in ("S10:perm", "S11:perm", "Z1:perm"):
        assert "depth" in _raises(fk.apply_fault_knobs, RuntimeError, cfg(depth=False), _env(knob, on))
        assert fk.apply_fault_knobs(cfg(), _env(knob, on))["id"] == knob[:-5]
    partial = cfg()
    partial.observations.depth_rear = NS()
    assert "depth_rear" in _raises(fk.apply_fault_knobs, RuntimeError, partial, _env("S11:perm", on))
    # A5 needs the 90 kg payload event
    assert "90" in _raises(fk.apply_fault_knobs, RuntimeError, cfg(), _env("A5:mid", on))
    assert "90" in _raises(fk.apply_fault_knobs, RuntimeError, cfg(kg=(0.0, 90.0)), _env("A5:mid", on))
    assert fk.apply_fault_knobs(cfg(kg=(90.0, 90.0)), _env("A5:mid", on))["params"]["as"] == "A1"
    assert fk.apply_fault_knobs(cfg(kg=[90, 90]), _env("A5:mid", on))["id"] == "A5"
    # the plain families need nothing else (a teacher env without depth groups, no delay)
    for knob in ("S1:perm", "A1:mid", "B1:5", "B2:0.2", "B4:0.1", "S12:30"):
        assert fk.apply_fault_knobs(cfg(use_delay=False, hist=1, depth=False), _env(knob, on))["knob"] == knob


def test_omni_knobs_applies_the_fault_knobs_after_the_payload_and_with_the_stress_variables_passed_in():
    src = OMNI_KNOBS_PY.read_text()
    i_payload = src.index("_pk.apply_payload_knobs(cfg, env)")
    i_fault = src.index("fault_knobs.apply_fault_knobs(cfg, env, exclusive_with=_srk.STRESS_VARS)")
    assert i_payload < i_fault                                                      # A5 checks the payload event
    assert "crab_hex_omni_fault_knobs" in src and "fault_knobs.py" in src
    # path-loaded inside apply_omni_knobs, never at module scope: dir(omni_knobs) (the ladder whitelist) is unchanged
    tree = ast.parse(src)
    top_names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert "fault_knobs" not in top_names


# =============================================================================== B. the onset table
@pytest.mark.parametrize("row, window, lo_steps", [(r, w, s) for r, (w, s) in ROW_WINDOWS.items()])
def test_earliest_onsets_of_the_four_rows(row, window, lo_steps):
    """CHARTER 3.2: 375 / 500 / 480 / 480 steps; ``pre_onset_last_step`` of cells.yaml = lo - 1 (the class-check horizon)."""
    lo_s, hi_s = (float(x) for x in window.split(":"))
    lo, hi = fi.onset_window_steps(lo_s, hi_s, DT)
    assert lo == lo_steps
    table = fi.onset_table(lo_s, hi_s, DT, 1000, 1)
    assert int(table.min()) == lo and int(table.max()) < hi and tuple(table.shape) == (fi.WAVES_MAX, 1000)
    assert int(table.min()) >= lo
    doc = _cells_doc()
    assert doc["rows"][row]["onset"] == window and doc["rows"][row]["pre_onset_last_step"] == lo_steps - 1


def test_onset_table_is_deterministic_per_seed_and_the_documented_rule():
    t1 = fi.onset_table(7.5, 11.5, DT, 100, 1)
    assert torch.equal(t1, fi.onset_table(7.5, 11.5, DT, 100, 1))
    assert not torch.equal(t1, fi.onset_table(7.5, 11.5, DT, 100, 2))
    assert t1.dtype == torch.long and tuple(t1.shape) == (fi.WAVES_MAX, 100)
    g = torch.Generator(device="cpu").manual_seed((1 + 0x6661756C74) % 2 ** 63)
    u = torch.rand((fi.WAVES_MAX, 100), generator=g, dtype=torch.float64)
    assert torch.equal(t1, 375 + torch.floor(u * (575 - 375)).long())              # lo + floor(u * (hi - lo))
    assert fi.onset_table(7.5, 11.5, DT, 100, 1, waves=3).shape == (3, 100)
    assert torch.equal(fi.onset_table(7.5, 11.5, DT, 100, 1, waves=3), t1[:3])     # the first rows do not depend on the count
    assert fi.onset_seed(None) == fi.onset_seed(0) == 0x6661756C74
    assert fi.onset_seed(2 ** 63) == 0x6661756C74 and fi.onset_seed(-1) == (0x6661756C74 - 1) % 2 ** 63
    assert fi.dose_seed(1, 0) != fi.onset_seed(1) and fi.dose_seed(1, 0) != fi.dose_seed(1, 1)
    assert fi.dose_seed(None, 3) == fi.dose_seed(0, 3) == (0x646F7365 + 3) % 2 ** 63


def test_onset_window_refusals():
    for lo, hi in ((5.0, 5.0), (5.0, 5.01), (9.0, 1.0)):
        _raises(fi.onset_window_steps, ValueError, lo, hi, DT)
    assert fi.onset_window_steps(0.0, 0.02, DT) == (0, 1)


def test_onsets_are_independent_of_the_cell_id_and_the_level():
    base = None
    for fid, entry in fk.FAULTS.items():
        for level in entry["levels"]:
            spec = fk.parse_fault_knobs(_env(f"{fid}:{level}", "7.5:11.5"))
            inj = fi.FaultInjector(spec, num_envs=16, step_dt=DT, device="cpu", env_seed=7)
            base = inj.table if base is None else base
            assert torch.equal(inj.table, base), (fid, level)
    # from_start (anchors) and the same window: same table
    inj = fi.FaultInjector(fk.parse_fault_knobs(_env("S12:30", "7.5:11.5")), num_envs=16, step_dt=DT, device="cpu", env_seed=7)
    assert torch.equal(inj.table, base)
    # the seed is the env seed (the eval's --seed), the window is the row's
    assert not torch.equal(base, fi.onset_table(7.5, 11.5, DT, 16, 8))
    assert not torch.equal(base, fi.onset_table(10.0, 60.0, DT, 16, 7))


def test_the_global_rng_is_never_touched_by_construction_or_a_full_run():
    torch.manual_seed(123)
    state = torch.get_rng_state()
    fi.onset_table(10, 60, DT, 100, 1)
    fi.FaultInjector(fk.parse_fault_knobs(_env("S3:0.2")), num_envs=8, step_dt=DT, device="cpu", env_seed=1)
    assert torch.equal(state, torch.get_rng_state())
    for knob in ("S3:0.2", "S4:perm", "B2:0.2", "B3:perm", "S6:0.1"):                # the faults that draw doses
        rig = _make(_env(knob))
        _reset(rig)
        acts = _acts(8, 30)
        torch.manual_seed(123)
        state = torch.get_rng_state()
        for s in range(30):
            rig.inj.sensor(rig.env, *_raw(8, s))
            _step(rig, acts[s])
        assert torch.equal(state, torch.get_rng_state()), knob


# =============================================================================== the stand-in action term
def _source_of(path: Path, cls: str, name: str) -> str:
    src = path.read_text()
    klass = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


_METHODS: dict = {}


def _methods() -> dict:
    """``process_actions`` / ``_apply_pitch_linkage`` exec'd from ``parkour_actions.py`` (they import Isaac as a module)."""
    if not _METHODS:
        ns = {"torch": torch, "math": math, "linkage": linkage, "park_clock_phase": clock.park_clock_phase,
              "clock_command_speed": clock.clock_command_speed}
        for name in ("process_actions", "_apply_pitch_linkage"):
            code = "from __future__ import annotations\n" + _source_of(ACTIONS_PY, "CrabHexDelayedJointPositionAction", name)
            exec(compile(code, str(ACTIONS_PY), "exec"), ns)
            _METHODS[name] = ns[name]
    return _METHODS


_NAMES = ("FL", "FR", "ML", "MR", "RL", "RR")
JOINT_NAMES = ([f"{l}_Body_CamShaft_RevoluteJoint" for l in _NAMES] + [f"{l}_Hip_Femur_RevoluteJoint" for l in _NAMES]
               + [f"{l}_Femur_Tibia_RevoluteJoint" for l in _NAMES])
KP, KD, FMAX = 16000.0, 600.0, 1000.0


class _Asset:
    """A PhysX-like asset: gains are readable back, writes land per (env, joint), targets are stored."""

    def __init__(self, n):
        self.joint_names = list(JOINT_NAMES)
        self.data = NS(joint_pos=torch.zeros(n, 18), default_joint_pos=torch.zeros(n, 18), joint_vel=torch.zeros(n, 18),
                       joint_pos_target=torch.zeros(n, 18), joint_vel_target=torch.zeros(n, 18),
                       applied_torque=torch.zeros(n, 18))
        self.kp, self.kd, self.fm = (torch.full((n, 18), v) for v in (KP, KD, FMAX))
        self.root_physx_view = NS(get_dof_stiffnesses=lambda: self.kp.clone(), get_dof_dampings=lambda: self.kd.clone(),
                                  get_dof_max_forces=lambda: self.fm.clone())
        self.actuators = {
            "hip_femur": NS(joint_indices=list(range(6, 12)), stiffness=torch.full((n, 6), KP),
                            damping=torch.full((n, 6), KD), effort_limit=torch.full((n, 6), FMAX)),
            "femur_tibia": NS(joint_indices=list(range(12, 18)), stiffness=torch.full((n, 6), 6000.0),
                              damping=torch.full((n, 6), 200.0), effort_limit=torch.full((n, 6), 400.0))}

    def set_joint_position_target(self, t, joint_ids):
        self.data.joint_pos_target[:, joint_ids] = t

    def set_joint_velocity_target(self, t, joint_ids):
        self.data.joint_vel_target[:, joint_ids] = t

    @staticmethod
    def _w(buf, v, joint_ids, env_ids):
        buf[env_ids[:, None], torch.as_tensor(joint_ids)] = v

    def write_joint_stiffness_to_sim(self, v, joint_ids, env_ids):
        self._w(self.kp, v, joint_ids, env_ids)

    def write_joint_damping_to_sim(self, v, joint_ids, env_ids):
        self._w(self.kd, v, joint_ids, env_ids)

    def write_joint_effort_limit_to_sim(self, v, joint_ids, env_ids):
        self._w(self.fm, v, joint_ids, env_ids)


def _make(environ: dict, n: int = 8, seed: int = 1, use_delay: bool = True, register: bool = True) -> NS:
    """A stand-in crab action term + env + (when armed) the injector built and bound exactly as ``__init__`` does."""
    spec = fk.parse_fault_knobs(environ)
    cmd = torch.zeros(n, 3)
    env = NS(num_envs=n, step_dt=DT, physics_dt=PHYS_DT, device="cpu", common_step_counter=0,
             episode_length_buf=torch.zeros(n, dtype=torch.long),
             cfg=NS(seed=seed, omni_command=True, omni_clock_turn_radius=1.2),
             command_manager=NS(get_command=lambda name: cmd, get_term=lambda name: None))
    if spec is not None:
        env.cfg.fault_inject = spec
    t = NS()
    t.process_actions = types.MethodType(_methods()["process_actions"], t)
    t._apply_pitch_linkage = types.MethodType(_methods()["_apply_pitch_linkage"], t)
    t._env = t.env = env
    t.clock_phase = torch.zeros(n)
    t._stand_park = False
    t._delay_update_global_steps = 10 ** 9
    t._action_delay_steps = []
    t._clip_raw_actions = lambda a: a.clamp(-4.8, 4.8)
    t._action_history_buf = torch.zeros(n, 8, 18)
    t._use_delay = use_delay
    t.delay = torch.tensor(1.0)
    t._raw_actions = torch.zeros(n, 18)
    t._scale, t._offset = 0.25, torch.zeros(n, 18)
    t._lpf_alpha = None
    t._joint_names = list(JOINT_NAMES)
    t._joint_ids = list(range(18))
    t._cam_action_cols = list(range(6))
    t._cam_shaft_joint_ids = list(range(6))
    t._pitch_joint_ids, t._knee_joint_ids = list(range(6, 12)), list(range(12, 18))
    t._pitch_action_cols, t._knee_action_cols = list(range(6, 12)), list(range(12, 18))
    t._knee_sign = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    t._asset = _Asset(n)
    t._hip_rod_len = torch.full((n, 6), (linkage.HIP_LEN_MIN_M + linkage.HIP_LEN_MAX_M) / 2)
    t._knee_rod_len = torch.full((n, 6), (linkage.KNEE_LEN_MIN_M + linkage.KNEE_LEN_MAX_M) / 2)
    t._rod_sync_needed = torch.ones(n, dtype=torch.bool)
    t._cam_vel_applied = torch.zeros(n, 6)
    t._fault = fi.get_injector(env)
    if t._fault is not None:
        t._fault.bind_action_term(t)
        if register:
            for stage in ("obs", *(f"depth:{c}" for c in CAMERAS)):
                t._fault.register(stage)
    return NS(t=t, env=env, inj=t._fault, n=n)


def _reset(rig, keep_reset_buf: bool = False) -> None:
    """An explicit all-env reset as the wrapper / harness does it: the term's reset hook, the length buffer zeroed after."""
    if not keep_reset_buf:
        rig.env.reset_buf = torch.zeros(rig.n, dtype=torch.bool)
    rig.inj.on_reset(rig.t, None)
    rig.env.episode_length_buf[:] = 0


def _instep_reset(rig, ids) -> None:
    """``env.step()``'s auto-reset of the terminated envs: reset_buf set, lengths already incremented, hook, lengths zeroed."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    rig.env.reset_buf = torch.zeros(rig.n, dtype=torch.bool)
    rig.env.reset_buf[ids] = True
    rig.inj.on_reset(rig.t, ids)
    rig.env.episode_length_buf[ids] = 0


def _step(rig, actions, substeps: int = SUB) -> None:
    rig.t.process_actions(actions)
    for _ in range(substeps):
        rig.t._apply_pitch_linkage()
    rig.env.episode_length_buf += 1
    rig.env.common_step_counter += 1


def _acts(n: int, steps: int, seed: int = 5, scale: float = 2.0) -> torch.Tensor:
    return torch.randn(steps, n, 18, generator=torch.Generator().manual_seed(seed)) * scale


_RAW_SHAPES = (3, 2, 2, 18, 18, 6)


def _raw(n: int, s: int, seed: int = 3):
    """The six raw channel blocks of step ``s`` (ang_vel, roll_pitch, vxy, joint_pos_delta, joint_vel, contact_fill)."""
    g = torch.Generator().manual_seed(seed * 100003 + s)
    return tuple(torch.randn(n, w, generator=g) + 0.5 for w in _RAW_SHAPES)


def _onsets(rig) -> list[int]:
    return [int(x) for x in rig.inj.onset]


def _probes(rig) -> dict:
    return rig.inj.sidecar()["waves"][0]["probes"]


def _is_active(o: int, s: int, dur) -> bool:
    return s >= o and (dur is None or s < o + dur)


def _sensor_run(knob: str, steps: int, n: int = 8, seed: int = 1, onset: str = ONS, step_actions: bool = True):
    rig = _make(_env(knob, onset), n=n, seed=seed)
    _reset(rig)
    acts = _acts(n, steps)
    ins, outs = [], []
    for s in range(steps):
        raw = _raw(n, s)
        keep = tuple(x.clone() for x in raw)
        o1 = rig.inj.sensor(rig.env, *raw)
        o2 = rig.inj.sensor(rig.env, *raw)                                          # a repeated compute at the same step
        assert all(torch.equal(a, b) for a, b in zip(o1, o2)), "sensor() is not idempotent within a step"
        assert all(torch.equal(a, b) for a, b in zip(raw, keep)), "sensor() modified its inputs"
        ins.append(keep)
        outs.append(o1)
        if step_actions:
            _step(rig, acts[s])
        else:
            rig.env.episode_length_buf += 1
    return rig, ins, outs


AV, RP, VXY, JPD, JV, CF = range(6)


def _same(ins, outs, s, e, chans):
    return all(torch.equal(outs[s][c][e], ins[s][c][e]) for c in chans)


# =============================================================================== C. per fault family
SENSOR_STEPS = 140


@pytest.mark.parametrize("knob, dur", [("S1:0.1", 5), ("S1:0.5", 25), ("S1:2", 100), ("S1:perm", None)])
def test_S1_zeroes_vxy_on_the_window_only(knob, dur):
    rig, ins, outs = _sensor_run(knob, SENSOR_STEPS)
    for e, o in enumerate(_onsets(rig)):
        for s in range(SENSOR_STEPS):
            if _is_active(o, s, dur):
                assert float(outs[s][VXY][e].abs().max()) == 0.0, (knob, e, s)
            else:
                assert torch.equal(outs[s][VXY][e], ins[s][VXY][e]), (knob, e, s)   # pre-onset deviation 0; off again after
            assert _same(ins, outs, s, e, (AV, RP, JPD, JV, CF)), (knob, e, s)
    p = _probes(rig)
    assert max(p["vxy_absmax"]) == 0.0 and max(p["obs_pre_dev"]) == 0.0
    assert ("obs_post_dev" in p) == (dur is not None)
    if dur is not None:
        assert max(p["obs_post_dev"]) == 0.0 and p["obs_active_n"] == [dur] * 8
    w = rig.inj.sidecar()["waves"][0]
    assert w["first_active"] == w["onset"]


@pytest.mark.parametrize("knob, dur", [("S2:0.5", 25), ("S2:2", 100), ("S2:perm", None)])
def test_S2_holds_vxy_at_the_value_of_the_step_before_onset(knob, dur):
    rig, ins, outs = _sensor_run(knob, SENSOR_STEPS)
    for e, o in enumerate(_onsets(rig)):
        for s in range(SENSOR_STEPS):
            if _is_active(o, s, dur):
                assert torch.equal(outs[s][VXY][e], ins[o - 1][VXY][e]), (knob, e, s)
            else:
                assert torch.equal(outs[s][VXY][e], ins[s][VXY][e]), (knob, e, s)
            assert _same(ins, outs, s, e, (AV, RP, JPD, JV, CF))
    p = _probes(rig)
    assert max(p["vxy_held_dev"]) == 0.0 and min(p["vxy_raw_move"]) > 0.0            # the real channel kept moving
    assert max(p["obs_pre_dev"]) == 0.0


@pytest.mark.parametrize("knob, dur", [("S5:0.1", 5), ("S5:0.5", 25), ("S5:2", 100)])
def test_S5_holds_ang_vel_and_roll_pitch_at_the_pre_onset_values(knob, dur):
    rig, ins, outs = _sensor_run(knob, SENSOR_STEPS)
    for e, o in enumerate(_onsets(rig)):
        for s in range(SENSOR_STEPS):
            for c in (AV, RP):
                want = ins[o - 1][c][e] if _is_active(o, s, dur) else ins[s][c][e]
                assert torch.equal(outs[s][c][e], want), (knob, e, s, c)
            assert _same(ins, outs, s, e, (VXY, JPD, JV, CF))
    p = _probes(rig)
    assert max(p["ang_vel_held_dev"]) == 0.0 and max(p["roll_pitch_held_dev"]) == 0.0
    assert min(p["ang_vel_raw_move"]) > 0.0 and max(p["obs_post_dev"]) == 0.0


@pytest.mark.parametrize("knob, bias", [("S3:0.1", 0.1), ("S3:0.2", 0.2)])
def test_S3_adds_a_constant_bias_of_per_episode_heading(knob, bias):
    rig, ins, outs = _sensor_run(knob, 40)
    g = torch.Generator(device="cpu").manual_seed(fi.dose_seed(1, 0))
    theta = (torch.rand(8, generator=g, dtype=torch.float64) * 2.0 * math.pi).float()
    want = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1) * bias
    for e, o in enumerate(_onsets(rig)):
        for s in range(40):
            d = outs[s][VXY][e] - ins[s][VXY][e]
            if s >= o:                                                              # permanent
                assert torch.allclose(d, want[e], atol=1e-6), (knob, e, s)
                assert abs(float(d.norm()) - bias) < 1e-6
            else:
                assert torch.equal(outs[s][VXY][e], ins[s][VXY][e])
            assert _same(ins, outs, s, e, (AV, RP, JPD, JV, CF))
    assert len({round(float(x), 4) for x in theta}) == 8                            # one heading per env, not one for all
    p = _probes(rig)
    assert max(p["bias_dev"]) < 1e-6 and max(p["obs_pre_dev"]) == 0.0


@pytest.mark.parametrize("knob, bias", [("S6:0.05", 0.05), ("S6:0.1", 0.1)])
def test_S6_biases_roll_pitch_not_vxy(knob, bias):
    rig, ins, outs = _sensor_run(knob, 40)
    g = torch.Generator(device="cpu").manual_seed(fi.dose_seed(1, 0))
    phi = (torch.rand(8, generator=g, dtype=torch.float64) * 2.0 * math.pi).float()
    want = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1) * bias
    for e, o in enumerate(_onsets(rig)):
        for s in range(40):
            if s >= o:
                assert torch.allclose(outs[s][RP][e] - ins[s][RP][e], want[e], atol=1e-6)
            else:
                assert torch.equal(outs[s][RP][e], ins[s][RP][e])
            assert _same(ins, outs, s, e, (AV, VXY, JPD, JV, CF))


def test_S3_headings_are_a_per_wave_draw_of_the_dose_generator():
    rig = _make(_env("S3:0.2"))
    _reset(rig)
    h0 = rig.inj.heading.clone()
    rig.env.episode_length_buf += 5
    _reset(rig)
    h1 = rig.inj.heading.clone()
    assert not torch.equal(h0, h1)                                                  # re-seeded with the wave
    g = torch.Generator(device="cpu").manual_seed(fi.dose_seed(1, 1))
    assert torch.equal(h1, (torch.rand(8, generator=g, dtype=torch.float64) * 2.0 * math.pi).float())


def test_S4_walks_vxy_from_the_dose_generator_and_the_10s_rms_is_0p2():
    n, steps = 8, 60
    rig, ins, outs = _sensor_run("S4:perm", steps, n=n)
    g = torch.Generator(device="cpu").manual_seed(fi.dose_seed(1, 0))
    walk = torch.zeros(n, 2)
    walks = []
    for s in range(steps):                                                          # the walk as seen by step s's observation
        walks.append(walk.clone())
        inc = torch.randn((n, 2), generator=g) * fk.S4_SIGMA                        # drawn at step s's process_actions
        on = torch.tensor([s + 1 >= o for o in _onsets(rig)])
        walk = walk + torch.where(on[:, None], inc, torch.zeros_like(inc))
    for e, o in enumerate(_onsets(rig)):
        for s in range(steps):
            if s >= o:
                assert torch.allclose(outs[s][VXY][e], ins[s][VXY][e] + walks[s][e], atol=1e-6), (e, s)
            else:
                assert torch.equal(outs[s][VXY][e], ins[s][VXY][e])
            assert _same(ins, outs, s, e, (AV, RP, JPD, JV, CF))
    assert float(walks[-1].abs().max()) > 0.0
    # realized RMS at 10 s (CHARTER 3.3: ~0.2 m/s), from the sidecar probe, many envs, every env onset at step 0
    n = 256
    rig = _make(_env("S4:perm", "0:0.02"), n=n)
    _reset(rig)
    acts = _acts(n, 500)
    for s in range(500):
        rig.inj.sensor(rig.env, *_raw(n, s))
        _step(rig, acts[s], substeps=1)
    w10 = torch.tensor(_probes(rig)["walk_10s"])
    assert tuple(w10.shape) == (n, 2) and not torch.isnan(w10).any()
    rms = float((w10 ** 2).sum(dim=1).mean().sqrt())
    assert 0.17 < rms < 0.23, rms


@pytest.mark.parametrize("knob", ["S8:zero", "S8:held"])
def test_S8_zeroes_or_holds_contact_fill_only(knob):
    rig, ins, outs = _sensor_run(knob, 40)
    for e, o in enumerate(_onsets(rig)):
        for s in range(40):
            if s >= o:
                want = torch.zeros(6) if knob == "S8:zero" else ins[o - 1][CF][e]
                assert torch.equal(outs[s][CF][e], want), (knob, e, s)
            else:
                assert torch.equal(outs[s][CF][e], ins[s][CF][e])
            assert _same(ins, outs, s, e, (AV, RP, VXY, JPD, JV))
    p = _probes(rig)
    assert max(p["obs_pre_dev"]) == 0.0
    assert (max(p["cf_absmax"]) == 0.0) if knob == "S8:zero" else (max(p["contact_fill_held_dev"]) == 0.0)


def test_S12_refreshes_ang_vel_roll_pitch_vxy_on_the_3k_over_5_pattern_from_step_0():
    steps = 50
    rig, ins, outs = _sensor_run("S12:30", steps)
    for e in range(8):
        refresh = [True] + [(3 * k) // 5 != (3 * (k - 1)) // 5 for k in range(1, steps)]
        assert sum(refresh) == 30                                                   # 30 Hz at 50 Hz
        last = None
        for s in range(steps):
            raw7 = torch.cat((ins[s][AV][e], ins[s][RP][e], ins[s][VXY][e]))
            got7 = torch.cat((outs[s][AV][e], outs[s][RP][e], outs[s][VXY][e]))
            if refresh[s]:
                last = raw7
                assert torch.equal(got7, raw7), (e, s)                              # value age 0
            else:
                assert torch.equal(got7, last) and not torch.equal(got7, raw7), (e, s)   # value age 20 ms (one step)
            assert _same(ins, outs, s, e, (JPD, JV, CF))
    pat = [torch.equal(outs[s][AV][0], ins[s][AV][0]) for s in range(10)]
    assert pat == [True, False, True, False, True, True, False, True, False, True]
    p = _probes(rig)
    assert p["s12_refresh_n"] == [30] * 8 and p["s12_hold_n"] == [20] * 8 and max(p["s12_hold_dev"]) == 0.0
    w = rig.inj.sidecar()["waves"][0]
    assert w["start"] == [0] * 8 and w["first_active"] == [0] * 8 and w["planned_last"] is None
    assert w["onset"] == _onsets(rig) and min(w["onset"]) >= 10                     # the table is still drawn and recorded


def test_D1_substitutes_the_commanded_target_for_the_joint_state():
    n, steps = 8, 30
    rig = _make(_env("D1:1"))
    _reset(rig)
    acts = _acts(n, steps)
    jids = rig.inj.d1_joint_ids
    assert jids == list(range(6, 18)) and rig.inj.d1_cols == list(range(6, 18))   # hip + knee rod joints
    for s in range(steps):
        raw = _raw(n, s)
        last = rig.t._action_history_buf[:, -1].clone()                             # the last commanded (clipped) action
        out = rig.inj.sensor(rig.env, *raw)
        want = last[:, jids] * 0.25 + 0.0 - rig.t._asset.data.default_joint_pos[:, jids]
        assert torch.allclose(out[JPD][:, jids], want, atol=1e-7), s
        keep = [c for c in range(18) if c not in jids]
        assert torch.equal(out[JPD][:, keep], raw[JPD][:, keep])                    # the cam joints keep the measured delta
        assert float(out[JV].abs().max()) == 0.0 and float(out[CF].abs().max()) == 0.0
        assert all(torch.equal(out[c], raw[c]) for c in (AV, RP, VXY))
        _step(rig, acts[s])
    w = rig.inj.sidecar()["waves"][0]
    assert w["start"] == [0] * n and max(w["probes"]["d1_jpd_dev"]) == 0.0 and max(w["probes"]["d1_jv_cf_absmax"]) == 0.0


def test_the_pre_onset_probe_would_catch_a_leak():
    """The per-(wave, env) inactivity probe (CHARTER 6.2) is the only guard against a fault that acts before its onset: prove
    it is not trivially zero by leaking on purpose (the activity mask opens 2 steps early while ``pre`` stays honest)."""
    rig = _make(_env("S1:perm"))
    _reset(rig)
    honest = rig.inj._masks

    def leaky(k):
        act, pre, post = honest(k)
        return act | (rig.inj.armed & (k >= rig.inj.start - 2)), pre, post

    rig.inj._masks = leaky
    for s in range(30):
        rig.inj.sensor(rig.env, *_raw(8, s))
        rig.env.episode_length_buf += 1
    assert min(_probes(rig)["obs_pre_dev"]) > 0.0


def test_a_sensor_fault_acts_only_in_its_own_kinds():
    for knob in ("B1:5", "A1:mid", "S10:perm"):
        rig = _make(_env(knob))
        _reset(rig)
        rig.env.episode_length_buf += 25
        raw = _raw(8, 0)
        out = rig.inj.sensor(rig.env, *raw)
        assert all(a is b for a, b in zip(out, raw)), knob                          # the very same tensors, no work done


# ---------------------------------------------------------------- depth
def _frame(n: int, s: int, cam: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(7919 * s + 17 * CAMERAS.index(cam) + 1)
    return torch.rand(n, 6, 9, generator=g) * 2.0 + 0.1                            # metres, never 0


def _depth_run(knob: str, steps: int, every: int = 1, n: int = 8):
    rig = _make(_env(knob))
    _reset(rig)
    acts = _acts(n, steps)
    calls = []                                                                      # (s, cam, raw, out)
    for s in range(steps):
        if s % every == 0:
            for cam in CAMERAS:
                raw = _frame(n, s, cam)
                out = rig.inj.depth_raw(rig.env, cam, raw)
                rig.inj.depth_image(rig.env, cam, out / 2.0 - 0.5)                  # the stand-in normaliser
                calls.append((s, cam, raw, out))
        _step(rig, acts[s])
    return rig, calls


@pytest.mark.parametrize("knob, dur", [("S10:0.5", 25), ("S10:2", 100), ("S10:perm", None)])
def test_S10_zeroes_the_front_raw_frame_only(knob, dur):
    rig, calls = _depth_run(knob, SENSOR_STEPS)
    onsets = _onsets(rig)
    for s, cam, raw, out in calls:
        for e, o in enumerate(onsets):
            if cam == "front" and _is_active(o, s, dur):
                assert float(out[e].abs().max()) == 0.0 and float(raw[e].abs().min()) > 0.0, (knob, e, s)
            else:
                assert torch.equal(out[e], raw[e]), (knob, cam, e, s)               # ToF cameras never; front outside the window
    p = _probes(rig)
    assert max(p["front_raw_absmax"]) == 0.0 and min(p["front_raw_in_max"]) > 0.0
    assert max(p["front_img_dev"]) == 0.0                                           # image = -0.5 everywhere while active
    assert max(p["depth_front_pre_dev"]) == 0.0
    assert "depth_left_change_max" in p and max(p["depth_left_change_max"]) == 0.0
    assert max(p["depth_rear_change_max"]) == 0.0 and max(p["depth_right_change_max"]) == 0.0
    if dur is not None:
        assert max(p["depth_front_post_dev"]) == 0.0 and p["depth_front_active_n"] == [dur] * 8


@pytest.mark.parametrize("knob, dur", [("S11:2", 100), ("S11:perm", None)])
@pytest.mark.parametrize("every", [1, 5])
def test_S11_freezes_the_tof_frames_at_the_last_pre_onset_frames_and_leaves_the_front_alone(knob, dur, every):
    rig, calls = _depth_run(knob, SENSOR_STEPS, every=every)
    onsets = _onsets(rig)
    for cam in ("left", "right", "rear"):
        mine = [c for c in calls if c[1] == cam]
        for e, o in enumerate(onsets):
            frozen = [raw[e] for (s, _c, raw, _out) in mine if s < o][-1]            # the last frame captured before onset
            for s, _c, raw, out in mine:
                want = frozen if _is_active(o, s, dur) else raw[e]
                assert torch.equal(out[e], want), (knob, cam, e, s)
    for s, cam, raw, out in calls:
        if cam == "front":
            assert torch.equal(out, raw)                                            # S11 never touches the ZED
    p = _probes(rig)
    for cam in ("left", "right", "rear"):
        assert max(p[f"{cam}_frozen_dev"]) == 0.0 and min(p[f"{cam}_raw_move"]) > 0.0
        assert max(p[f"depth_{cam}_pre_dev"]) == 0.0
    assert "front_raw_absmax" not in p


@pytest.mark.parametrize("knob, dur", [("Z1:0.5", 25), ("Z1:2", 100), ("Z1:perm", None)])
def test_Z1_is_S10_and_S5_and_S2_at_the_same_onset_and_duration(knob, dur):
    steps = SENSOR_STEPS
    rig = _make(_env(knob))
    _reset(rig)
    acts = _acts(8, steps)
    ins, outs, fronts = [], [], []
    for s in range(steps):
        raw = _raw(8, s)
        ins.append(raw)
        outs.append(rig.inj.sensor(rig.env, *raw))
        fr, lf = _frame(8, s, "front"), _frame(8, s, "left")
        fronts.append((fr, rig.inj.depth_raw(rig.env, "front", fr)))
        assert torch.equal(rig.inj.depth_raw(rig.env, "left", lf), lf)               # the ToF cameras are not part of Z1
        _step(rig, acts[s])
    for e, o in enumerate(_onsets(rig)):
        for s in range(steps):
            act = _is_active(o, s, dur)
            for c in (AV, RP, VXY):                                                 # all three sensor channels hold together
                want = ins[o - 1][c][e] if act else ins[s][c][e]
                assert torch.equal(outs[s][c][e], want), (knob, e, s, c)
            assert _same(ins, outs, s, e, (JPD, JV, CF))
            fr, out = fronts[s]
            assert (float(out[e].abs().max()) == 0.0) if act else torch.equal(out[e], fr[e]), (knob, e, s)
    w = rig.inj.sidecar()["waves"][0]
    assert w["first_active"] == w["onset"]                                          # every channel starts at the table onset
    if dur is not None:
        assert w["last_active"] == [o + dur - 1 for o in w["onset"]]


# ---------------------------------------------------------------- action path
def _run_actions(knob: str, steps: int, n: int = 8, use_delay: bool = True, scale: float = 3.0):
    rig = _make(_env(knob), n=n, use_delay=use_delay)
    _reset(rig)
    acts = _acts(n, steps, scale=scale)
    applied, hist = [], []
    for s in range(steps):
        _step(rig, acts[s])
        applied.append(rig.t._raw_actions.clone())
        hist.append(rig.t._action_history_buf.clone())
    return rig, acts, applied, hist


def _nominal_applied(acts, s):
    """The delay-1 read: the previous step's clipped action (zeros at step 0)."""
    return acts[s - 1].clamp(-4.8, 4.8) if s >= 1 else torch.zeros_like(acts[0])


@pytest.mark.parametrize("steps_held", [2, 5, 10, 25])
def test_B1_holds_the_last_pre_onset_applied_action_for_n_steps(steps_held):
    steps = 70
    rig, acts, applied, _ = _run_actions(f"B1:{steps_held}", steps)
    for e, o in enumerate(_onsets(rig)):
        for s in range(steps):
            if o <= s < o + steps_held:
                assert torch.equal(applied[s][e], applied[o - 1][e]), (steps_held, e, s)
            else:
                assert torch.equal(applied[s][e], _nominal_applied(acts, s)[e]), (steps_held, e, s)
    p = _probes(rig)
    assert p["repeat_n"] == [steps_held] * 8 and max(p["act_pre_dev"]) == 0.0 and max(p["act_post_dev"]) == 0.0
    w = rig.inj.sidecar()["waves"][0]
    assert w["first_active"] == w["onset"] and w["last_active"] == [o + steps_held - 1 for o in w["onset"]]
    assert w["planned_last"] == w["last_active"]


def test_B1_applied_action_is_what_the_actuator_sees_the_processed_action():
    rig, acts, applied, _ = _run_actions("B1:5", 40)
    assert torch.equal(rig.t._processed_actions, rig.t._raw_actions * 0.25)         # scale applied AFTER the fault


@pytest.mark.parametrize("p_drop", [0.05, 0.2])
def test_B2_drops_actions_at_rate_p_from_the_dose_generator(p_drop):
    n, steps = 64, 140
    rig, acts, applied, _ = _run_actions(f"B2:{p_drop}", steps, n=n)
    g = torch.Generator(device="cpu").manual_seed(fi.dose_seed(1, 0))
    onsets = _onsets(rig)
    reps = 0
    active = 0
    for s in range(steps):
        u = torch.rand(n, generator=g)                                              # one draw per env step, for every env
        for e, o in enumerate(onsets):
            if s >= o:
                active += 1
                rep = bool(u[e] < p_drop)
                reps += rep
                want = applied[s - 1][e] if rep else _nominal_applied(acts, s)[e]
                assert torch.equal(applied[s][e], want), (e, s)
            else:
                assert torch.equal(applied[s][e], _nominal_applied(acts, s)[e])
    pr = _probes(rig)
    assert sum(pr["repeat_n"]) == reps and sum(pr["act_active_n"]) == active
    assert abs(reps / active - p_drop) < 0.025                                       # rate ~ p
    assert max(pr["act_pre_dev"]) == 0.0


def test_B3_draws_the_delay_per_env_per_step_from_1_2_3():
    n, steps = 64, 140
    rig, acts, applied, hist = _run_actions("B3:perm", steps, n=n)
    g = torch.Generator(device="cpu").manual_seed(fi.dose_seed(1, 0))
    onsets = _onsets(rig)
    for s in range(steps):
        dd = torch.randint(1, 4, (n,), generator=g)
        assert set(dd.tolist()) <= {1, 2, 3}
        for e, o in enumerate(onsets):
            if s >= o:
                assert torch.equal(applied[s][e], hist[s][e, 7 - int(dd[e])]), (e, s)   # delay dd: the dd-th last pushed
            else:
                assert torch.equal(applied[s][e], hist[s][e, 6])                         # the nominal delay 1
    p = _probes(rig)
    total = [p["delay1_n"][e] + p["delay2_n"][e] + p["delay3_n"][e] for e in range(n)]
    assert total == p["act_active_n"]                                                # every active step drew one of {1, 2, 3}
    for d in (1, 2, 3):
        assert 0.25 < sum(p[f"delay{d}_n"]) / sum(total) < 0.42                      # ~ uniform: mean delay 2 steps = 40 ms
    assert "delay4_n" not in p and "delay0_n" not in p


def test_B5_is_a_constant_delay_of_two_steps_from_onset():
    steps = 50
    rig, acts, applied, hist = _run_actions("B5:perm", steps)
    for e, o in enumerate(_onsets(rig)):
        for s in range(steps):
            want = hist[s][e, 5] if s >= o else hist[s][e, 6]
            assert torch.equal(applied[s][e], want), (e, s)
            if s >= o:
                assert torch.equal(applied[s][e], _nominal_applied(acts, s - 1)[e])    # the action of two steps ago
    p = _probes(rig)
    assert max(p["act_pre_dev"]) == 0.0 and p["act_active_n"] == [steps - o for o in _onsets(rig)]
    assert not any(k.startswith("delay") for k in p)                                # the latency-only control draws nothing


def test_B3_and_B5_refuse_an_action_term_without_a_delayed_history():
    for knob in ("B3:perm", "B5:perm"):
        assert "use_delay" in _raises(_make, RuntimeError, _env(knob), use_delay=False)


# ---------------------------------------------------------------- rods and cams
def _drive_rods(rig, acts_fn, steps: int):
    """Steps with the real ``_apply_pitch_linkage``; records (k, hip_len, knee_len, hip_speed, knee_speed) per substep."""
    rec = []
    for s in range(steps):
        rig.t.process_actions(acts_fn(s))
        for _ in range(SUB):
            rig.t._apply_pitch_linkage()
            rec.append((int(rig.env.episode_length_buf[0]), rig.t._hip_rod_len.clone(), rig.t._knee_rod_len.clone(),
                        rig.t.hip_rod_speed.clone(), rig.t.knee_rod_speed.clone()))
        rig.env.episode_length_buf += 1
        rig.env.common_step_counter += 1
    return rec


def _square(s, n=8, amp=0.6, period=7):
    return torch.full((n, 18), amp if (s // period) % 2 == 0 else -amp)


@pytest.mark.parametrize("knob, d", [("B4:0.1", 5), ("B4:0.5", 25)])
def test_B4_keeps_each_rod_slewing_at_its_last_pre_onset_slew(knob, d):
    rig = _make(_env(knob))
    _reset(rig)
    steps = 60
    rec = _drive_rods(rig, _square, steps)
    for e, o in enumerate(_onsets(rig)):
        for col, which in ((3, "hip"), (4, "knee")):
            pre = [r for r in rec if r[0] == o - 1]
            last_speed = pre[-1][col][e]                                            # the speed of the last pre-onset substep
            assert float(last_speed.abs().max()) > 0.0                              # a real, non-zero slew was in progress
            act = [r for r in rec if o <= r[0] < o + d]
            for r in act:
                assert torch.allclose(r[col][e], last_speed, atol=5e-5, rtol=0), (knob, e, which, r[0])
            nominal_changes = sum(not torch.allclose(r[col][e], last_speed, atol=5e-5, rtol=0)
                                  for r in rec if o + d <= r[0] < o + d + 8)
            assert nominal_changes > 0                                               # the runaway ends with the window
    p = _probes(rig)
    assert max(p["b4_hip_slew_dev"]) < 1e-6 and max(p["b4_knee_slew_dev"]) < 1e-6
    assert max(p["rod_hip_pre_dev"]) == 0.0 and max(p["rod_knee_pre_dev"]) == 0.0
    assert max(p["rod_hip_post_dev"]) == 0.0 and max(p["rod_knee_post_dev"]) == 0.0


def test_B4_clips_the_runaway_at_the_rod_window_limit():
    rig = _make(_env("B4:0.1", "0.2:0.22"))                                         # every env's onset = step 10
    _reset(rig)
    assert _onsets(rig) == [10] * 8
    ms = linkage.HIP_SPEED_M_S * PHYS_DT
    rig.t._hip_rod_len[:] = linkage.HIP_LEN_MIN_M + 46 * ms
    rig.t._rod_sync_needed[:] = False
    # drive the hip rod DOWN (a short rod is the lower limit), reverse the command during the fault
    rec = _drive_rods(rig, lambda s: torch.full((8, 18), -4.0 if s < 10 else 4.0), 16)
    before = [r for r in rec if r[0] == 9][-1]
    assert float(before[3][:, 2].max()) < 0.0                                       # still shortening at the last pre-onset substep
    lens = torch.stack([r[1][:, 2] for r in rec if 10 <= r[0] < 15])                # the 20 faulted substeps
    assert float(lens.min()) >= linkage.HIP_LEN_MIN_M - 1e-6                        # never below the window
    assert float(lens.min()) <= linkage.HIP_LEN_MIN_M + 1e-6                        # it ran into it ...
    assert torch.all(lens[1:] <= lens[:-1] + 1e-7)                                  # ... and kept going the old way, not back up
    assert float(lens[-1].max()) - linkage.HIP_LEN_MIN_M < 1e-6                     # and stays pinned at the limit
    p = _probes(rig)
    assert min(p["b4_hip_clipped_n"]) > 0
    clipped_speed = torch.stack([r[3][:, 2] for r in rec if r[0] == 14])[-1]
    assert float(clipped_speed.abs().max()) < 1e-3                                  # the clipped slew is ~0, not the old slew


@pytest.mark.parametrize("d", [5, 25])
def test_B6_freezes_every_rod_at_its_current_length(d):
    rig = _make(_env(f"B6:{0.1 if d == 5 else 0.5}"))
    _reset(rig)
    rec = _drive_rods(rig, _square, 60)
    for e, o in enumerate(_onsets(rig)):
        act = [r for r in rec if o <= r[0] < o + d]
        first = act[0]
        for r in act:
            assert torch.equal(r[1][e], first[1][e]) and torch.equal(r[2][e], first[2][e]), (d, e, r[0])
            assert float(r[3][e].abs().max()) == 0.0 and float(r[4][e].abs().max()) == 0.0   # zero slew -> zero rod speed
        after = [r for r in rec if r[0] == o + d + 6]
        assert not torch.equal(after[-1][1][e], first[1][e])                        # the rods move again when it clears
        pre = [r for r in rec if r[0] == o - 1][-1]
        assert float(pre[3][e].abs().max()) > 0.0                                   # and they were moving before
    p = _probes(rig)
    assert max(p["b6_hip_dlam_absmax"]) == 0.0 and max(p["b6_knee_dlam_absmax"]) == 0.0
    assert max(p["rod_hip_pre_dev"]) == 0.0 and max(p["rod_hip_post_dev"]) == 0.0


def test_cam_hooks_B4_keeps_the_last_cam_command_and_B6_commands_zero():
    n = 8
    for knob, d in (("B4:0.1", 5), ("B6:0.1", 5)):
        rig = _make(_env(knob))
        _reset(rig)
        ons = _onsets(rig)
        g = torch.Generator().manual_seed(11)
        cmds, outs = [], []
        for s in range(40):
            cmd = torch.randn(n, 6, generator=g) + 0.3
            for _ in range(SUB):
                outs.append((s, cmd, rig.inj.cam_cmd(rig.t, cmd)))
            cmds.append(cmd)
            rig.env.episode_length_buf += 1
        for s, cmd, out in outs:
            for e, o in enumerate(ons):
                if o <= s < o + d:
                    want = cmds[o - 1][e] if knob == "B4:0.1" else torch.zeros(6)
                    assert torch.equal(out[e], want), (knob, e, s)
                else:
                    assert torch.equal(out[e], cmd[e]), (knob, e, s)
        p = _probes(rig)
        assert max(p["cam_pre_dev"]) == 0.0 and max(p["cam_post_dev"]) == 0.0
        assert (max(p["b4_cam_cmd_dev"]) == 0.0) if knob == "B4:0.1" else (max(p["b6_cam_cmd_absmax"]) == 0.0)
    # a rod fault never touches the cam command, and vice versa
    rig = _make(_env("A1:mid"))
    _reset(rig)
    rig.env.episode_length_buf += 30
    cmd = torch.randn(n, 6)
    assert rig.inj.cam_cmd(rig.t, cmd) is cmd


def _record_rods(rig, acts_fn, steps):
    return _drive_rods(rig, acts_fn, steps)


@pytest.mark.parametrize("leg, col", [("front", 0), ("mid", 2), ("rear", 4)])
def test_A1_locks_that_legs_rod_lengths_and_no_other_leg(leg, col):
    steps = 60
    rig = _make(_env(f"A1:{leg}"))
    nom = _make(_env(None, None))
    assert nom.inj is None                                                          # knob off: no injector at all
    assert rig.inj.leg_col == col and rig.inj.binding["leg"] == {"front": "FL", "mid": "ML", "rear": "RL"}[leg]
    _reset(rig)
    big = lambda s: _square(s, amp=1.5, period=5)  # noqa: E731
    rec, rec_n = _record_rods(rig, big, steps), _record_rods(nom, big, steps)
    others = [c for c in range(6) if c != col]
    for e, o in enumerate(_onsets(rig)):
        lock = [r for r in rec if r[0] >= o]
        for r in lock:
            assert torch.equal(r[1][e, col], lock[0][1][e, col]) and torch.equal(r[2][e, col], lock[0][2][e, col])
            assert float(r[3][e, col]) == 0.0 and float(r[4][e, col]) == 0.0                 # slew 0, to the end (permanent)
        # pre-onset: identical to the unfaulted run, every leg; post-onset: the other legs are still identical
        for r, rn in zip(rec, rec_n):
            if r[0] < o:
                assert torch.equal(r[1][e], rn[1][e]) and torch.equal(r[2][e], rn[2][e])
            assert torch.equal(r[1][e, others], rn[1][e, others]) and torch.equal(r[2][e, others], rn[2][e, others])
        moved = [rn for rn in rec_n if rn[0] >= o]
        assert not torch.equal(moved[-1][1][e, col], moved[0][1][e, col])           # the unfaulted rod would have moved
    p = _probes(rig)
    for k in ("a1_hip_dlam_absmax", "a1_knee_dlam_absmax", "a1_hip_len_drift", "a1_knee_len_drift"):
        assert max(p[k]) == 0.0, k
    assert max(p["rod_hip_pre_dev"]) == 0.0


def test_A5_is_A1_mid_at_the_same_dose():
    a5, a1 = _make(_env("A5:mid")), _make(_env("A1:mid"))
    assert a5.inj.kind == "A1" and a5.inj.leg_col == a1.inj.leg_col == 2 and a5.inj.spec["id"] == "A5"
    for rig in (a5, a1):
        _reset(rig)
    r5, r1 = _record_rods(a5, _square, 40), _record_rods(a1, _square, 40)
    for x, y in zip(r5, r1):
        assert torch.equal(x[1], y[1]) and torch.equal(x[2], y[2])


def test_A3_caps_that_legs_rod_slew_at_0p3_of_the_rated_speed():
    steps = 60
    rig = _make(_env("A3:mid"))
    nom = _make(_env(None, None))
    _reset(rig)
    big = lambda s: _square(s, amp=2.0, period=6)  # noqa: E731
    rec, rec_n = _record_rods(rig, big, steps), _record_rods(nom, big, steps)
    col = rig.inj.leg_col
    assert col == 2
    cap = {"hip": 0.3 * linkage.HIP_SPEED_M_S, "knee": 0.3 * linkage.KNEE_SPEED_M_S}
    others = [c for c in range(6) if c != col]
    for e, o in enumerate(_onsets(rig)):
        for idx, which in ((3, "hip"), (4, "knee")):
            seen_cap = False
            for r, rn in zip(rec, rec_n):
                sp = abs(float(r[idx][e, col]))
                if r[0] < o:
                    assert torch.equal(r[idx][e], rn[idx][e])                       # pre-onset: bitwise nominal
                else:
                    assert sp <= cap[which] * (1 + 1e-4) + 1e-6, (e, which, r[0], sp)      # never above 0.3 x rated
                    seen_cap |= sp >= cap[which] * (1 - 1e-3)
                assert torch.equal(r[idx][e, others], rn[idx][e, others])               # the other legs run at the full rate
            assert seen_cap, (e, which)                                              # the cap actually binds
    # the other legs' lengths are the nominal ones for the whole run (they are independent rods)
    for r, rn in zip(rec, rec_n):
        assert torch.equal(r[1][:, others], rn[1][:, others])
    p = _probes(rig)
    assert max(p["a3_hip_slew_ratio_max"]) <= 0.3 + 1e-6 and max(p["a3_knee_slew_ratio_max"]) <= 0.3 + 1e-6
    assert min(p["a3_hip_at_cap_n"]) > 0 and max(p["rod_hip_pre_dev"]) == 0.0
    nominal_peak = max(abs(float(rn[3][0, col])) for rn in rec_n)
    assert nominal_peak > 0.9 * linkage.HIP_SPEED_M_S                                # the unfaulted leg does run at the full rate


def test_A4_holds_that_legs_cam_velocity_target_at_zero():
    n = 8
    rig = _make(_env("A4:mid"))
    _reset(rig)
    assert rig.inj.cam_col == 2 and rig.inj.cam_joint_id == 2 and rig.inj.binding["cam_joint"].startswith("ML_")
    ons = _onsets(rig)
    g = torch.Generator().manual_seed(2)
    rig.t._asset.data.joint_vel[:, 2] = 0.7
    for s in range(40):
        rig.t._cam_vel_applied = torch.randn(n, 6, generator=g) + 2.0
        before = rig.t._cam_vel_applied.clone()
        rig.t._asset.data.joint_pos[:, 2] = 0.01 * s
        rig.inj.cam_applied(rig.t)
        for e, o in enumerate(ons):
            if s >= o:
                assert float(rig.t._cam_vel_applied[e, 2]) == 0.0
                assert torch.equal(rig.t._cam_vel_applied[e, [0, 1, 3, 4, 5]], before[e, [0, 1, 3, 4, 5]])   # the other cams
            else:
                assert torch.equal(rig.t._cam_vel_applied[e], before[e])
        rig.env.episode_length_buf += 1
    p = _probes(rig)
    assert max(p["a4_cam_target_absmax"]) == 0.0 and max(p["cam_pre_dev"]) == 0.0
    assert all(v > 0 for v in p["a4_cam_joint_vel_absmax"])                        # the shaft really was spinning meanwhile
    # a non-A4 kind does nothing
    other = _make(_env("A1:mid"))
    _reset(other)
    other.env.episode_length_buf += 30
    other.t._cam_vel_applied = torch.ones(n, 6)
    other.inj.cam_applied(other.t)
    assert torch.equal(other.t._cam_vel_applied, torch.ones(n, 6))


def test_A2_writes_zero_gains_at_onset_keeps_the_target_and_restores_at_reset():
    n, steps = 8, 30
    rig = _make(_env("A2:rear"))
    _reset(rig)
    asset = rig.t._asset
    jh, jk = rig.inj.leg_joint_ids
    assert asset.joint_names[jh] == "RL_Hip_Femur_RevoluteJoint" and asset.joint_names[jk] == "RL_Femur_Tibia_RevoluteJoint"
    assert rig.inj.leg_col == 4
    acts = _acts(n, steps)
    ons = _onsets(rig)
    tgt_hist = []
    for s in range(steps):
        _step(rig, acts[s])
        tgt_hist.append(asset.data.joint_pos_target.clone())
        for e, o in enumerate(ons):
            kp = asset.kp[e, [jh, jk]]
            if s < o:
                assert torch.equal(kp, torch.full((2,), KP)) and float(asset.kd[e, jh]) == KD and float(asset.fm[e, jh]) == FMAX
            else:                                                                   # stiffness 0, damping 0, max effort 0
                assert float(asset.kp[e, [jh, jk]].abs().max()) == 0.0 and float(asset.kd[e, [jh, jk]].abs().max()) == 0.0
                assert float(asset.fm[e, [jh, jk]].abs().max()) == 0.0
                for name, ji in (("hip_femur", jh), ("femur_tibia", jk)):          # the actuator model too
                    act = asset.actuators[name]
                    assert float(act.stiffness[e, ji - act.joint_indices[0]]) == 0.0
                    assert float(act.damping[e, ji - act.joint_indices[0]]) == 0.0
                    assert float(act.effort_limit[e, ji - act.joint_indices[0]]) == 0.0
        others = [j for j in range(18) if j not in (jh, jk)]
        assert torch.equal(asset.kp[:, others], torch.full((n, 16), KP))            # no other joint's gains are written
    # the broken leg's targets stay at their onset values (velocity target 0); the linkage still drives the others
    for e, o in enumerate(ons):
        for s in range(o, steps):
            assert torch.equal(tgt_hist[s][e, [jh, jk]], tgt_hist[o][e, [jh, jk]]), (e, s)
        assert not torch.equal(tgt_hist[steps - 1][e, 6], tgt_hist[o][e, 6])         # the linkage still drives the other legs
    assert float(asset.data.joint_vel_target[:, [jh, jk]].abs().max()) == 0.0
    p = _probes(rig)
    assert p["a2_gain_pre"][0] == [KP, KP, KD, KD, FMAX, FMAX]
    assert p["a2_gain_post"][0] == [0.0] * 6
    assert p["a2_write_step"] == [float(o) for o in ons]
    # reset: gains come back, in PhysX and in the actuator model; a second wave writes them again
    _reset(rig)
    assert torch.equal(asset.kp, torch.full((n, 18), KP)) and torch.equal(asset.kd, torch.full((n, 18), KD))
    assert torch.equal(asset.fm, torch.full((n, 18), FMAX))
    assert float(asset.actuators["femur_tibia"].stiffness[:, 4].min()) == 6000.0
    assert float(asset.actuators["femur_tibia"].stiffness[:, 4].max()) == 6000.0
    assert float(asset.actuators["hip_femur"].damping[:, 4].min()) == KD
    assert float(asset.actuators["femur_tibia"].effort_limit[:, 4].min()) == 400.0
    assert not bool(rig.inj.gain_written.any())


def test_A2_a_mid_wave_reset_restores_only_the_reset_env():
    n = 8
    rig = _make(_env("A2:mid"))
    _reset(rig)
    jh, jk = rig.inj.leg_joint_ids
    acts = _acts(n, 40)
    for s in range(30):
        _step(rig, acts[s])
    assert float(rig.t._asset.kp[:, jh].max()) == 0.0
    _instep_reset(rig, [3])
    assert float(rig.t._asset.kp[3, jh]) == KP and float(rig.t._asset.kp[3, jk]) == KP     # restored on the fallen env
    assert float(rig.t._asset.kd[3, jh]) == KD and float(rig.t._asset.fm[3, jh]) == FMAX
    assert float(rig.t._asset.kp[[0, 1, 2, 4, 5, 6, 7], jh].max()) == 0.0                  # the rest stay broken
    for s in range(30, 40):                                                                 # a disarmed env is never rewritten
        _step(rig, acts[s])
    assert float(rig.t._asset.kp[3, jh]) == KP


# =============================================================================== D. reset semantics
def _exercise(rig, steps: int, seed: int = 9):
    """Every hook, every step, as the real terms call them (the hooks of a kind other than the cell's are no-ops)."""
    n, inj, t, env = rig.n, rig.inj, rig.t, rig.env
    acts = _acts(n, steps, seed=seed)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        inj.sensor(env, *_raw(n, s))
        if s % 5 == 0:
            for cam in CAMERAS:
                fr = _frame(n, s, cam)
                out = inj.depth_raw(env, cam, fr)
                inj.depth_image(env, cam, out / 2.0 - 0.5)
        t.process_actions(acts[s])
        for _ in range(SUB):
            t._apply_pitch_linkage()
            cmd = inj.cam_cmd(t, torch.randn(n, 6, generator=g) + 0.5)
            t._cam_vel_applied += (cmd - t._cam_vel_applied).clamp(-0.1, 0.1)
            inj.cam_applied(t)
        env.episode_length_buf += 1
        env.common_step_counter += 1


ALL_CELLS = [f"{fid}:{lvl}" for fid, e in fk.FAULTS.items() for lvl in e["levels"]]


def _state_tensors(inj) -> dict:
    out = {"holds." + k: v for k, v in inj.holds.items()}
    out.update({"frozen." + k: v for k, v in inj.frozen.items()})
    out.update({"last_dlam." + k: v for k, v in inj.last_dlam.items()})
    out.update({"len0." + k: v for k, v in inj.len0.items()})
    out["walk"] = inj.walk
    out["frozen_tgt"] = inj.frozen_tgt
    for name in ("s12_prev", "prev_applied", "last_cam_cmd", "cam_theta0"):
        if getattr(inj, name) is not None:
            out[name] = getattr(inj, name)
    return out


@pytest.mark.parametrize("knob, must_be_live", [
    ("S2:perm", ["holds.vxy"]), ("S5:2", ["holds.ang_vel"]), ("S8:held", ["holds.contact_fill"]),
    ("S4:perm", ["walk"]), ("S12:30", ["s12_prev"]), ("S11:perm", ["frozen.left", "frozen.rear"]),
    ("B1:25", ["prev_applied"]), ("B2:0.2", ["prev_applied"]), ("B4:0.5", ["last_dlam.hip", "last_cam_cmd"]),
    ("A1:mid", ["len0.hip"]), ("A3:mid", ["len0.knee"]), ("A4:mid", ["cam_theta0"]), ("A2:mid", ["frozen_tgt"]),
])
def test_all_fault_state_is_cleared_at_an_episode_reset(knob, must_be_live):
    rig = _make(_env(knob))
    _reset(rig)
    _exercise(rig, 60)
    state = _state_tensors(rig.inj)
    for name in must_be_live:                                                       # the test exercised the state it clears
        t = state[name]
        assert bool((t.nan_to_num(0.0) != 0).any()) or name == "cam_theta0", name
    if knob == "A4:mid":
        assert not bool(torch.isnan(rig.inj.cam_theta0).all())
    _reset(rig)
    for name, t in _state_tensors(rig.inj).items():
        if name.startswith("len0.") or name == "cam_theta0":
            assert bool(torch.isnan(t).all()), name                                 # "not captured yet"
        else:
            assert float(t.abs().max()) == 0.0, name
    inj = rig.inj
    assert bool(inj.armed.all()) and bool((inj._last_obs_k == -1).all()) and bool((inj.fell_at == -1).all())
    assert inj.p == {}                                                              # the probes restart with the wave
    assert not bool(inj.gain_written.any())


@pytest.mark.parametrize("knob", [k for k in ALL_CELLS if k not in ("S12:30", "D1:1")])
def test_the_fault_is_inactive_at_step_0_and_until_the_onset(knob):
    rig = _make(_env(knob))
    _reset(rig)
    k = rig.env.episode_length_buf
    act, pre, post = rig.inj._masks(k)
    assert not bool(act.any()) and bool(pre.all()) and not bool(post.any())         # step 0 of a fresh wave
    assert int(rig.inj.onset.min()) >= 10
    assert bool((rig.inj.start == rig.inj.onset).all())
    for knob_anchor in ("S12:30", "D1:1"):
        a = _make(_env(knob_anchor))
        _reset(a)
        act, _pre, _post = a.inj._masks(a.env.episode_length_buf)
        assert bool(act.all())                                                      # anchors: active from step 0 ...
        assert torch.equal(a.inj.onset, rig.inj.onset)                              # ... on the same recorded table


def test_a_mid_wave_reset_disarms_the_env_and_does_not_redraw_its_onset():
    n, steps = 8, 40
    rig = _make(_env("S1:perm"))
    _reset(rig)
    onset0, table0 = rig.inj.onset.clone(), rig.inj.table.clone()
    assert torch.equal(onset0, fi.onset_table(0.2, 0.4, DT, n, 1)[0])
    ins, outs = [], []
    for s in range(steps):
        if s == 5:                                                                 # env 3 falls, auto-reset in the step
            _instep_reset(rig, [3])
        raw = _raw(n, s)
        ins.append(raw)
        outs.append(rig.inj.sensor(rig.env, *raw))
        _step(rig, _acts(n, 1, seed=s)[0])
    assert rig.inj.wave == 0                                                       # an in-step reset is not a new wave
    assert not bool(rig.inj.armed[3]) and int(rig.inj.fell_at[3]) == 5
    assert bool(rig.inj.armed[[0, 1, 2, 4, 5, 6, 7]].all())
    assert torch.equal(rig.inj.onset, onset0) and torch.equal(rig.inj.table, table0)   # never redrawn
    for s in range(steps):
        for e in range(n):
            active = float(outs[s][VXY][e].abs().max()) == 0.0
            if e == 3:
                assert not active, s                                                # a fallen env stays done: never re-armed
            else:
                assert active == (s >= int(onset0[e])), (e, s)
    w = rig.inj.sidecar()["waves"][0]
    assert w["fell_at"][3] == 5 and w["fell_at"][0] == -1 and w["first_active"][3] == -1


def test_an_env_that_fell_while_the_fault_was_active_is_disarmed_and_clean():
    rig = _make(_env("S2:perm"))
    _reset(rig)
    onset = int(rig.inj.onset[2])
    for s in range(onset + 4):
        rig.inj.sensor(rig.env, *_raw(8, s))
        _step(rig, _acts(8, 1, seed=s)[0])
    assert float(rig.inj.holds["vxy"][2].abs().max()) > 0.0
    _instep_reset(rig, [2])
    assert float(rig.inj.holds["vxy"][2].abs().max()) == 0.0 and float(rig.inj.holds["vxy"][1].abs().max()) > 0.0
    assert int(rig.inj.fell_at[2]) == onset + 4 and not bool(rig.inj.armed[2])


def test_wave_counter_explicit_resets_open_waves_and_in_step_resets_do_not():
    n = 8
    rig = _make(_env("B1:5"))
    assert rig.inj.wave == -1
    _reset(rig)                                                                    # the wrapper's construction reset = wave 0
    assert rig.inj.wave == 0 and bool(rig.inj.armed.all())
    rig.inj.check_wave(0)
    acts = _acts(n, 20)
    for s in range(12):
        _step(rig, acts[s])
    _instep_reset(rig, [1, 4])
    assert rig.inj.wave == 0
    for s in range(12, 14):
        _step(rig, acts[s])
    # every env falls on the same step: still not a new wave (lengths > 0 at the hook)
    _instep_reset(rig, list(range(n)))
    assert rig.inj.wave == 0 and not bool(rig.inj.armed.any())
    # the harness's explicit reset right after, with reset_buf still all True and the lengths already 0: a new wave
    _reset(rig, keep_reset_buf=True)
    assert rig.inj.wave == 1 and bool(rig.inj.armed.all()) and bool((rig.inj.fell_at == -1).all())
    assert torch.equal(rig.inj.onset, fi.onset_table(0.2, 0.4, DT, n, 1)[1])         # that wave's row of the table
    assert not torch.equal(rig.inj.onset, fi.onset_table(0.2, 0.4, DT, n, 1)[0])
    rig.inj.check_wave(1)
    # an explicit reset with episode lengths > 0 and no env flagged (the normal wave boundary)
    for s in range(5):
        _step(rig, acts[s])
    _reset(rig)
    assert rig.inj.wave == 2
    # env_ids=None is every env
    rig.env.reset_buf = torch.zeros(n, dtype=torch.bool)
    rig.env.episode_length_buf += 3
    rig.inj.on_reset(rig.t, None)
    assert rig.inj.wave == 3
    # a reset of every env given as ids is the same
    rig.env.reset_buf = torch.zeros(n, dtype=torch.bool)
    rig.inj.on_reset(rig.t, torch.arange(n))
    assert rig.inj.wave == 4
    # an empty id list is a no-op
    rig.inj.on_reset(rig.t, torch.tensor([], dtype=torch.long))
    assert rig.inj.wave == 4
    assert len(rig.inj.records) == 4                                                # waves 0..3 flushed at the next wave's start


def test_check_wave_raises_on_a_mismatch_and_a_missing_reset():
    rig = _make(_env("S1:perm"))
    msg = _raises(rig.inj.check_wave, RuntimeError, 0)                              # before any reset: counted -1
    assert "wave -1" in msg and "wave 0" in msg and "misaligned" in msg
    _reset(rig)
    rig.inj.check_wave(0)
    assert "counted wave 0" in _raises(rig.inj.check_wave, RuntimeError, 1)         # the harness reset twice, the injector once
    rig.env.episode_length_buf += 4
    _reset(rig)
    _raises(rig.inj.check_wave, RuntimeError, 0)
    rig.inj.check_wave(1)


def test_too_many_waves_is_a_launch_error():
    rig = _make(_env("S1:perm"))
    _reset(rig)
    rig.inj.wave = fi.WAVES_MAX - 1
    rig.env.episode_length_buf += 2
    assert "waves" in _raises(_reset, RuntimeError, rig)


def test_a_new_wave_rearms_with_its_own_table_row_and_the_same_fault_state_machinery():
    n = 8
    rig = _make(_env("S5:0.1"))
    _reset(rig)
    for s in range(30):
        rig.inj.sensor(rig.env, *_raw(n, s))
        _step(rig, _acts(n, 1, seed=s)[0])
    rig.env.reset_buf = torch.zeros(n, dtype=torch.bool)
    rig.inj.on_reset(rig.t, None)
    rig.env.episode_length_buf[:] = 0
    assert rig.inj.wave == 1
    ins, outs = [], []
    for s in range(30):
        raw = _raw(n, s, seed=8)
        ins.append(raw)
        outs.append(rig.inj.sensor(rig.env, *raw))
        _step(rig, _acts(n, 1, seed=s)[0])
    for e, o in enumerate(_onsets(rig)):
        assert o == int(fi.onset_table(0.2, 0.4, DT, n, 1)[1][e])
        for s in range(30):
            want = ins[o - 1][AV][e] if _is_active(o, s, 5) else ins[s][AV][e]
            assert torch.equal(outs[s][AV][e], want), (e, s)                         # wave 1 holds ITS pre-onset values
    doc = rig.inj.sidecar()
    assert [w["wave"] for w in doc["waves"]] == [0, 1]
    assert doc["waves"][0]["onset"] != doc["waves"][1]["onset"]


def test_the_injector_refuses_unbound_stages_loudly():
    for knob, stages, missing in (("S1:perm", (), "obs"), ("S10:perm", ("obs",), "depth:front"),
                                  ("Z1:2", ("obs",), "depth:front"), ("S11:2", ("obs", "depth:left", "depth:right"), "depth:rear")):
        rig = _make(_env(knob), register=False)
        for s in stages:
            rig.inj.register(s)
        _reset(rig)
        msg = _raises(rig.t.process_actions, RuntimeError, _acts(8, 1)[0])
        assert missing in msg and "inert" in msg, (knob, msg)
    # a teacher-side cell needs only the action term: nothing to register
    rig = _make(_env("B1:5"), register=False)
    _reset(rig)
    rig.t.process_actions(_acts(8, 1)[0])
    # camera names map from the sensor entity names
    assert fi.FaultInjector.camera_name("depth_camera_front") == "front" and fi.FaultInjector.camera_name("left") == "left"


def test_knob_off_has_no_injector_and_leaves_the_env_untouched():
    rig = _make({})
    assert rig.inj is None and rig.t._fault is None and not hasattr(rig.env, fi.ENV_ATTR)
    assert fi.get_injector(NS(cfg=NS())) is None and fi.get_injector(NS()) is None
    env = NS(cfg=NS(fault_inject=None))
    assert fi.get_injector(env) is None and not hasattr(env, fi.ENV_ATTR)
    # the same injector is shared by every consumer of one env
    rig = _make(_env("S1:perm"))
    assert fi.get_injector(rig.env) is rig.inj and getattr(rig.env, fi.ENV_ATTR) is rig.inj
    assert _raises(fi.FaultInjector, ValueError, {"id": "S1", "bogus": 1}, num_envs=1, step_dt=DT, device="cpu")
    assert _raises(fi.FaultInjector, ValueError, {"id": "X9", "onset_s": [1, 2]}, num_envs=1, step_dt=DT, device="cpu")


# =============================================================================== E. the sidecar
_DOC_KEYS = {"schema", "fault", "kind", "duration_steps", "dt", "num_envs", "env_seed", "onset", "dose_rng", "wave_rule",
             "binding", "waves"}
_WAVE_KEYS = {"wave", "onset", "start", "planned_last", "first_active", "last_active", "fell_at", "probes", "probe_kind",
              "state_cleared"}
_PROBE_KINDS = {"self_check", "sim", "model_estimate", "dose", "activity"}


@pytest.mark.parametrize("knob", ALL_CELLS)
def test_every_cell_runs_all_hooks_and_writes_a_strict_json_sidecar(knob):
    n = 6
    rig = _make(_env(knob), n=n)
    _reset(rig)
    _exercise(rig, 45)
    rig.env.episode_length_buf += 0
    doc = rig.inj.sidecar()
    assert set(doc) == _DOC_KEYS
    assert doc["schema"] == fi.SIDECAR_SCHEMA == "fault_onsets/v1"
    assert doc["fault"]["knob"] == knob and doc["num_envs"] == n and doc["dt"] == DT and doc["env_seed"] == 1
    assert doc["kind"] == ("A1" if knob == "A5:mid" else knob.split(":")[0])
    assert set(doc["onset"]) == {"window_s", "window_steps", "seed", "salt", "waves_drawn", "rule"}
    assert doc["onset"]["window_steps"] == [10, 20] and doc["onset"]["waves_drawn"] == fi.WAVES_MAX
    assert doc["onset"]["seed"] == fi.onset_seed(1) and doc["onset"]["salt"] == hex(0x6661756C74)
    assert doc["dose_rng"]["salt"] == hex(0x646F7365) and "wave" in doc["wave_rule"]
    assert "stages" in doc["binding"] and "action" in doc["binding"]["stages"] and "action_joint_names" in doc["binding"]
    assert len(doc["waves"]) == 1
    w = doc["waves"][0]
    assert _WAVE_KEYS <= set(w)
    assert set(w["probe_kind"]) == set(w["probes"]) and set(w["probe_kind"].values()) <= _PROBE_KINDS
    assert w["state_cleared"] is True                                                # checked (and raised) at each wave start
    assert ("depth_visible_span_steps" in w) == (doc["kind"] in ("S10", "S11", "Z1"))
    for key in ("onset", "start", "first_active", "last_active", "fell_at"):
        assert len(w[key]) == n and all(isinstance(x, int) for x in w[key]), key
    assert w["wave"] == 0 and w["onset"] == _onsets(rig) and w["fell_at"] == [-1] * n
    assert w["start"] == ([0] * n if knob in ("S12:30", "D1:1") else w["onset"])
    assert (w["planned_last"] is None) == (doc["duration_steps"] is None or knob in ("S12:30", "D1:1"))
    assert ("heading_rad" in w) == (knob.split(":")[0] in ("S3", "S6"))
    json.dumps(doc, allow_nan=False)                                                 # strict JSON: no NaN / Infinity
    assert json.loads(json.dumps(doc)) == doc
    assert doc["fault"]["id"] == knob.split(":")[0]
    # every probe is a per-env column
    for name, vals in w["probes"].items():
        assert len(vals) == n, name
    # the cell acted (every env's onset is <= step 19 < 45) unless it only acts on stages this exercise does not feed
    assert all(a >= 0 for a in w["first_active"]), (knob, w["first_active"])
    assert all(a >= o for a, o in zip(w["first_active"], w["start"]))
    assert min(w["last_active"]) >= min(w["first_active"])


def test_sidecar_probe_names_of_the_charters_dose_checks_exist():
    """CHARTER 6.1.3: the realized-dose probes each smoke asserts."""
    want = {"S1:perm": ["vxy_absmax"], "S2:perm": ["vxy_held_dev", "vxy_raw_move"], "S5:2": ["ang_vel_held_dev"],
            "S3:0.2": ["bias_dev", "heading_rad"], "S8:zero": ["cf_absmax"], "S10:perm": ["front_raw_absmax", "front_img_dev"],
            "S11:perm": ["left_frozen_dev", "right_raw_move"], "S12:30": ["s12_refresh_n", "s12_hold_n", "s12_hold_dev"],
            "D1:1": ["d1_jpd_dev", "d1_jv_cf_absmax"], "B1:5": ["repeat_n"], "B2:0.2": ["repeat_n"],
            "B3:perm": ["delay1_n", "delay2_n", "delay3_n"], "B4:0.1": ["b4_hip_slew_dev", "b4_cam_cmd_dev"],
            "B6:0.1": ["b6_hip_dlam_absmax", "b6_cam_cmd_absmax"], "A1:mid": ["a1_hip_dlam_absmax", "a1_knee_len_drift"],
            "A3:mid": ["a3_hip_slew_ratio_max", "a3_hip_at_cap_n"],
            "A4:mid": ["a4_cam_target_absmax", "a4_cam_joint_vel_absmax", "a4_cam_angle_drift"],
            "A2:mid": ["a2_gain_pre", "a2_gain_post", "a2_target_leave_max", "a2_write_step"]}
    for knob, names in want.items():
        rig = _make(_env(knob))
        _reset(rig)
        _exercise(rig, 45)
        d = rig.inj.sidecar()["waves"][0]
        have = set(d["probes"]) | ({"heading_rad"} if "heading_rad" in d else set())
        assert set(names) <= have, (knob, set(names) - have)


def test_the_sidecar_flushes_once_and_does_not_double_count():
    rig = _make(_env("S1:perm"))
    _reset(rig)
    _exercise(rig, 30)
    d1 = rig.inj.sidecar()
    d2 = rig.inj.sidecar()
    assert len(d1["waves"]) == len(d2["waves"]) == 1 and d1["waves"] == d2["waves"]
    assert fi.FaultInjector(fk.parse_fault_knobs(_env("S1:perm")), num_envs=2, step_dt=DT, device="cpu")\
        .sidecar()["waves"] == []                                                    # nothing recorded before the first reset


def test_write_sidecar_into_writes_the_file_and_the_run_meta_summary(tmp_path):
    rig = _make(_env("S1:0.5"))
    _reset(rig)
    _exercise(rig, 30)
    rig.env.episode_length_buf += 1
    _reset(rig)
    _exercise(rig, 10)
    harness = {"scenario_id": "suite__omni", "n_waves": 2, "n_steps_planned": 40, "suite": True,
               "episode_axis": "wave * num_envs + env"}
    meta = rig.inj.write_sidecar_into(tmp_path, harness)
    assert (tmp_path / "fault_onsets.json").is_file() and fi.SIDECAR_NAME == "fault_onsets.json"
    assert meta == {"sidecar": "fault_onsets.json", "schema": "fault_onsets/v1", "knob": "S1:0.5", "kind": "S1",
                    "duration_steps": 25, "onset_s": [0.2, 0.4], "from_start": False, "waves_recorded": 2}
    doc = json.loads((tmp_path / "fault_onsets.json").read_text())
    assert doc["schema"] == "fault_onsets/v1" and doc["harness"] == harness and len(doc["waves"]) == 2
    assert [w["wave"] for w in doc["waves"]] == [0, 1]
    json.dumps(meta)
    # the single-line compact form the harness writes
    text = (tmp_path / "fault_onsets.json").read_text()
    assert text.endswith("\n") and text.count("\n") == 1


# ---------------------------------------------------------------- the later additions of the injector
def test_dose_seeds_do_not_collide_across_seeds_and_waves():
    """A plain ``seed + salt + wave`` made ``(s, w + 1)`` and ``(s + 1, w)`` the same stream (two seeds sharing doses)."""
    seen = {}
    for sd in range(40):
        for w in range(fi.WAVES_MAX):
            v = fi.dose_seed(sd, w)
            assert 0 <= v < 2 ** 63 and v not in seen, (sd, w, seen.get(v))
            seen[v] = (sd, w)
    assert fi.dose_seed(1, 0) != fi.dose_seed(0, 1)
    assert fi.dose_seed(1, 2) == (1 * fi.DOSE_SEED_MIX + fi.DOSE_SEED_SALT + 2) % 2 ** 63
    assert fi.dose_seed(None, 5) == fi.dose_seed(0, 5)
    assert not {fi.onset_seed(sd) for sd in range(40)} & set(seen)                   # and apart from the onset streams


@pytest.mark.parametrize("knob", ALL_CELLS)
def test_student_input_faults_are_refused_on_any_other_policy_role(knob):
    rig = _make(_env(knob))
    student_only = rig.inj.kind in (fi.SENSOR_KINDS | fi.DEPTH_KINDS)
    assert fi.STUDENT_ONLY_KINDS == fi.SENSOR_KINDS | fi.DEPTH_KINDS
    rig.inj.check_role("student")
    assert rig.inj.role == "student"
    for role in ("teacher", "none"):
        if student_only:
            msg = _raises(rig.inj.check_role, RuntimeError, role)
            assert "student" in msg and "privileged" in msg and role in msg, (knob, role)
        else:
            rig.inj.check_role(role)                                                 # B / A cells act on the action path
            assert rig.inj.role == role


def test_the_probe_kind_labels():
    assert fi.probe_kind("a2_applied_torque_absmax") == "model_estimate"
    for name in ("first_obs", "last_act", "obs_active_n", "a2_write_step", "a2_gain_pre_onset_reads"):
        assert fi.probe_kind(name) == "activity", name
    for name in ("vxy_raw_move", "a2_gain_pre", "a2_gain_post", "a2_target_leave_max", "front_raw_in_max",
                 "a4_cam_joint_vel_absmax", "a2_gain_pre_onset_dev", "a2_gain_wave_start_dev"):
        assert fi.probe_kind(name) == "sim", name
    for name in ("obs_pre_dev", "obs_post_dev", "vxy_absmax", "vxy_held_dev", "front_img_dev", "b6_hip_dlam_absmax",
                 "a1_hip_len_drift", "a3_hip_slew_ratio_max", "left_frozen_dev", "b4_hip_slew_dev", "s12_hold_dev"):
        assert fi.probe_kind(name) == "self_check", name
    for name in ("repeat_n", "delay2_n", "walk_10s", "s12_refresh_n"):
        assert fi.probe_kind(name) == "dose", name
    assert fi.probe_kind("bias_dev") == "self_check"


@pytest.mark.parametrize("knob, dur", [("S10:0.5", 25), ("S11:2", 100), ("Z1:0.5", 25), ("S10:perm", None)])
def test_depth_cells_record_when_the_last_faulted_frame_leaves_the_policy(knob, dur):
    """The depth term refreshes every 5 steps and returns the frame one refresh old (CHARTER 3.3): a frame captured at
    refresh step r is visible through ``r + 5 * (1 + 1) - 1``."""
    steps = 150
    rig = _make(_env(knob))
    rig.inj.depth_timing(5, 1)
    _reset(rig)
    acts = _acts(8, steps)
    for s in range(steps):
        if s % 5 == 0:
            for cam in CAMERAS:
                rig.inj.depth_raw(rig.env, cam, _frame(8, s, cam))
        _step(rig, acts[s])
    w = rig.inj.sidecar()["waves"][0]
    assert w["depth_visible_span_steps"] == 10
    if dur is None:
        assert w["policy_visible_last"] is None and w["policy_visible_bound"] is None
        return
    for e, o in enumerate(w["onset"]):
        last_refresh = max(r for r in range(0, steps, 5) if o <= r < o + dur)
        assert w["policy_visible_last"][e] == last_refresh + 9, (knob, e)
        assert w["policy_visible_bound"][e] == o + dur - 1 + 10                        # planned_last + span
        assert w["policy_visible_last"][e] <= w["policy_visible_bound"][e]
        assert w["policy_visible_last"][e] > w["planned_last"][e]                      # it outlives the planned window
    other = _make(_env("S1:0.5"))
    _reset(other)
    _exercise(other, 30)
    assert "depth_visible_span_steps" not in other.inj.sidecar()["waves"][0]
    rig.inj.depth_timing(4, 2)
    assert rig.inj.sidecar()["waves"][0]["depth_visible_span_steps"] == 10             # a flushed wave is not re-derived
    rig2 = _make(_env("S10:0.5"))
    rig2.inj.depth_timing(4, 2)
    _reset(rig2)
    _exercise(rig2, 60)
    assert rig2.inj.sidecar()["waves"][0]["depth_visible_span_steps"] == 12


def test_a_stale_fault_state_tensor_is_a_launch_error_at_the_next_wave_start():
    rig = _make(_env("S2:perm"))
    _reset(rig)
    _exercise(rig, 30)
    assert rig.inj.state_cleared == [True]
    rig.env.episode_length_buf += 1
    _reset(rig)                                                                      # clean: the scan passes at a wave start
    assert rig.inj.state_cleared == [True, True]
    rig.inj.holds["vxy"] = torch.ones(8, 2)                                         # state the clear forgot
    assert "holds[vxy]" in _raises(rig.inj._check_cleared, RuntimeError)
    rig.inj.holds["vxy"] = torch.zeros(8, 2)
    rig.inj._check_cleared()
    # a NEW per-episode tensor that ``_clear`` does not know about is caught by the generic scan
    rig.inj.some_new_state = torch.ones(8)
    rig.env.episode_length_buf += 1
    assert "some_new_state" in _raises(_reset, RuntimeError, rig)
    assert rig.inj.state_cleared == [True, True, True]                               # the failed start appended nothing
    # tables / masks / saved restore values are not state
    rig2 = _make(_env("A2:mid"))
    _reset(rig2)
    _exercise(rig2, 30)
    _reset(rig2)
    assert rig2.inj.state_cleared == [True, True]


def test_A2_gain_probes_read_physx_against_the_construction_snapshot():
    n = 8
    rig = _make(_env("A2:mid"))
    assert rig.inj.binding["a2_gain_snapshot_env0"] == [KP, KP, KD, KD, FMAX, FMAX]
    assert tuple(rig.inj.gain_snapshot.shape) == (n, 3, 2)
    _reset(rig)
    _exercise(rig, 45)
    _reset(rig)
    _exercise(rig, 12)
    doc = rig.inj.sidecar()
    assert len(doc["waves"]) == 2
    for w in doc["waves"]:
        p = w["probes"]
        assert p["a2_gain_wave_start_dev"] == [0.0] * n                              # restored to the snapshot at the wave start
        assert w["probe_kind"]["a2_gain_wave_start_dev"] == "sim" and w["probe_kind"]["a2_gain_post"] == "sim"
    p0 = doc["waves"][0]["probes"]
    assert p0["a2_gain_pre_onset_dev"] == [0.0] * n and p0["a2_gain_pre_onset_reads"] == [1] * n   # read at onset - 1
    assert doc["waves"][0]["probe_kind"]["a2_applied_torque_absmax"] == "model_estimate"
    # a restore that fails is a launch error (the stale gain flag), not a quiet second wave at broken gains
    rig2 = _make(_env("A2:mid"))
    _reset(rig2)
    _exercise(rig2, 45)
    rig2.inj._restore_gains = lambda term, ids: None
    assert "gain_written" in _raises(_reset, RuntimeError, rig2)
    # and a PhysX read-back that drifted from the snapshot shows in the probe
    rig3 = _make(_env("A2:mid"))
    _reset(rig3)
    jh = rig3.inj.leg_joint_ids[0]
    rig3.t._asset.kp[2, jh] = 123.0
    assert float(rig3.inj._gain_dev(rig3.t, torch.arange(8))[2]) == pytest.approx(KP - 123.0)


# =============================================================================== F. the wiring
def _class_method_src(path: Path, cls: str, name: str) -> str:
    return _source_of(path, cls, name)


def test_action_term_hooks_are_in_the_documented_places():
    cls = "CrabHexDelayedJointPositionAction"
    pa = _class_method_src(ACTIONS_PY, cls, "process_actions")
    assert pa.index("self._raw_actions[:] = clipped_actions") < pa.index("fault.on_actions(self)") \
        < pa.index("self._processed_actions = self._raw_actions * self._scale + self._offset")
    al = _class_method_src(ACTIONS_PY, cls, "_apply_pitch_linkage")
    for which, clamp, integrate in (("hip", "dlam_h = (lam_h_cmd", "self._hip_rod_len += dlam_h"),
                                    ("knee", "dlam_k = (lam_k_cmd", "self._knee_rod_len += dlam_k")):
        call = f'fault.rod_dlam(self, "{which}"'
        assert al.index(clamp) < al.index(call) < al.index(integrate), which        # on the clamped slew, before integration
    assert al.rstrip().endswith("fault.after_linkage(self)")                        # after the joint targets are set
    aa = _class_method_src(ACTIONS_PY, cls, "apply_actions")
    assert aa.index("fault.cam_cmd(self, cam_cmd)") < aa.index("self._cam_vel_applied +=") < aa.index("fault.cam_applied(self)") \
        < aa.index("self._cam_vel_applied, joint_ids=self._cam_shaft_joint_ids")
    rs = _class_method_src(ACTIONS_PY, cls, "reset")
    assert "fault.on_reset(self, env_ids)" in rs
    init = _class_method_src(ACTIONS_PY, cls, "__init__")
    assert "get_injector(env)" in init and "bind_action_term(self)" in init
    for src in (pa, al, aa, rs):
        assert 'getattr(self, "_fault", None)' in src                              # every hook is skipped when None
    # the hook is guarded everywhere it is called
    text = ACTIONS_PY.read_text()
    assert text.count("fault is not None") >= 6


def test_observation_hooks_are_in_the_documented_places():
    cls = "CrabHexParkourObservations"
    call = _class_method_src(OBSERVATIONS_PY, cls, "__call__")
    assert call.index("stress_obs(") < call.index("fault.sensor(") < call.index("obs_buf = torch.cat(")
    init = _class_method_src(OBSERVATIONS_PY, cls, "__init__")
    assert 'self._fault.register("obs")' in init and "get_injector(env)" in init
    dcls = "image_features_batched"
    proc = _class_method_src(OMNI_OBS_PY, dcls, "_processed")
    assert proc.index("fault.depth_raw(") < proc.index("process_depth_batch(raw, clip") < proc.index("fault.depth_image(")
    assert "refresh" in proc.split("\n")[0]
    dcall = _class_method_src(OMNI_OBS_PY, dcls, "__call__")
    assert "self._processed(refresh=True)" in dcall
    reset = _class_method_src(OMNI_OBS_PY, dcls, "reset")
    assert "refresh" not in reset                                                    # the reset refill is never faulted
    dinit = _class_method_src(OMNI_OBS_PY, dcls, "__init__")
    assert '"_fault_injector"' in dinit and 'self._fault.register(f"depth:{self._fault_camera}")' in dinit
    assert "self._fault.depth_timing(DEPTH_TERM_PERIOD, self.lag)" in dinit


def test_the_depth_hook_faults_a_refresh_and_never_the_reset_refill():
    ns = {"torch": torch, "calls": []}

    def process_depth_batch(raw, clip, resize, crop):
        ns["calls"].append(raw.clone())
        return raw / 2.0 - 0.5

    ns["process_depth_batch"] = process_depth_batch
    exec(compile("from __future__ import annotations\n" + _class_method_src(OMNI_OBS_PY, "image_features_batched", "_processed"),
                 str(OMNI_OBS_PY), "exec"), ns)
    rig = _make(_env("S10:perm", "0:0.02"))
    _reset(rig)
    rig.env.episode_length_buf += 3                                                  # past the onset (0)
    raw = torch.rand(8, 6, 9, 1) + 0.5
    term = NS(camera_sensor=NS(data=NS(output={"distance_to_camera": raw})), device="cpu", clipping_range=(0.0, 2.0),
              resize=(6, 9), crop=None, _env=rig.env, _fault=rig.inj, _fault_camera="front")
    proc = types.MethodType(ns["_processed"], term)
    img = proc(refresh=True)
    assert torch.equal(ns["calls"][-1], torch.zeros(8, 6, 9)) and torch.equal(img, torch.full((8, 6, 9), -0.5))
    img = proc()                                                                     # not a refresh (the reset refill): no fault
    assert torch.equal(ns["calls"][-1], raw.squeeze(-1)) and not torch.equal(img, torch.full((8, 6, 9), -0.5))
    img = proc(torch.tensor([1, 2]))
    assert torch.equal(ns["calls"][-1], raw[[1, 2]].squeeze(-1))
    term._fault = None                                                               # knob off: the old path
    assert torch.equal(proc(refresh=True), raw.squeeze(-1) / 2.0 - 0.5)
    assert _probes(rig)["front_raw_absmax"][0] == 0.0


def test_eval_harness_asserts_the_wave_and_writes_the_sidecar_before_the_run():
    src = EVAL_PY.read_text()
    assert 'fault_inj = getattr(unwrapped, "_fault_injector", None)' in src
    assert src.index("fault_inj.check_wave(wave)") < src.index("fault_inj.write_sidecar_into(")
    assert src.index("fault_inj.write_sidecar_into(") < src.index("R.write_run(", src.index("fault_inj.write_sidecar_into("))
    assert 'run_meta["fault_inject"]' in src
    assert src.index("fault_inj.check_role(") < src.index("fault_inj.check_wave(wave)")   # the role is refused before any wave
    assert '"student" if is_student else ("none" if args_cli.zero_actions else "teacher")' in src


# =============================================================================== G. cells.yaml, envs/, make_envs.py
def _base_lines(doc: dict, row: str) -> list[str]:
    src = OMNI / doc["rows"][row]["base_env"]
    return [ln for ln in src.read_text().splitlines() if ln.startswith("KRABBY_")]


def _expected_files() -> dict[str, list[str]]:
    """``{file name: expected KRABBY_ lines}``, derived from cells.yaml independently of make_envs.py."""
    doc = _cells_doc()
    out = {}

    def add(cid, row, payload, extra):
        name = f"{cid}__{row}{'' if payload == 'L000' else '_' + payload}.env"
        assert name not in out, name
        out[name] = _base_lines(doc, row) + extra

    for nom in doc["nominals"]:
        for row in ("suite", "field_light") + (("ledge_down", "pits") if nom.get("kind_rows") else ()):
            add(nom["id"], row, nom["payload"], [])
    for cell in doc["cells"]:
        for row in ("suite", "field_light") + (("ledge_down", "pits") if cell.get("kind_rows") else ()):
            extra = []
            if cell.get("fault"):
                extra = [f"KRABBY_FAULT={cell['fault']}", f"KRABBY_FAULT_ONSET={doc['rows'][row]['onset']}"]
                if cell.get("from_start"):
                    extra.append("KRABBY_FAULT_FROM_START=1")
            extra += [f"{k}={v}" for k, v in (cell.get("env") or {}).items()]
            add(cell["id"], row, cell.get("payload", "L000"), extra)
    for smk in doc["smokes"]:
        for row in smk["rows"]:
            add(smk["id"], row, "L000", [f"KRABBY_FAULT={smk['fault']}", f"KRABBY_FAULT_ONSET={smk['onset']}"])
    return out


def test_make_envs_check_passes_on_the_checked_in_envs(capsys):
    _yaml()
    me = _load(MAKE_ENVS_PY, "make_envs__fault_test")
    assert me.main(["--check"]) == 0
    assert "matches" in capsys.readouterr().out


def test_make_envs_check_detects_a_missing_stale_and_extra_file(tmp_path, monkeypatch, capsys):
    _yaml()
    me = _load(MAKE_ENVS_PY, "make_envs__fault_test")
    monkeypatch.setattr(me, "ENVS", tmp_path)
    assert me.main([]) == 0                                                         # regenerate into a scratch dir ...
    n_written = len(list(tmp_path.glob("*.env")))
    assert n_written == len(list(ENVS_DIR.glob("*.env"))) and me.main(["--check"]) == 0
    for p in tmp_path.glob("*.env"):                                                # ... byte-identical to the checked-in set
        assert p.read_text() == (ENVS_DIR / p.name).read_text()
    victim = tmp_path / "S1_perm__suite.env"
    original = victim.read_text()
    victim.write_text(original + "KRABBY_FAULT_EXTRA=1\n")
    assert me.main(["--check"]) == 1 and "stale" in capsys.readouterr().out
    victim.write_text(original)
    victim.unlink()
    assert me.main(["--check"]) == 1 and "missing" in capsys.readouterr().out
    victim.write_text(original)
    (tmp_path / "ZZ_extra__suite.env").write_text("KRABBY_X=1\n")
    assert me.main(["--check"]) == 1 and "extra" in capsys.readouterr().out
    with pytest.raises(SystemExit):                                                 # the writer refuses to leave strays
        me.main([])


def test_the_env_set_is_exactly_what_cells_yaml_implies():
    expected = _expected_files()
    on_disk = {p.name for p in ENVS_DIR.glob("*.env")}
    assert on_disk == set(expected), (sorted(on_disk ^ set(expected))[:8])
    doc = _cells_doc()
    # counts: 2 nominals (L000 with kind rows, L090 without), 50 cells x (2 rows + 2 per kind-row cell), 1 smoke x 4 rows
    kind_cells = sum(1 for c in doc["cells"] if c.get("kind_rows"))
    assert kind_cells == 7                                                          # S1/S2 perm, S3 0.2, S10/S11/Z1 perm, T12
    assert len(expected) == (4 + 2) + ((50 + 6) * 2 + kind_cells * 2) + 4 == 6 + 126 + 4 + 0   # + 6 follow-up cells


def test_every_env_files_krabby_lines_are_base_plus_fault_lines():
    expected = _expected_files()
    doc = _cells_doc()
    base_keys = {r: [ln.split("=", 1)[0] for ln in _base_lines(doc, r)] for r in ("suite", "field_light", "ledge_down", "pits")}
    for name, want in expected.items():
        text = (ENVS_DIR / name).read_text()
        got = [ln for ln in text.splitlines() if ln.startswith("KRABBY_")]
        assert got == want, name
        keys = [ln.split("=", 1)[0] for ln in got]
        assert len(keys) == len(set(keys)), name                                    # no duplicate key
        assert text.startswith("# GENERATED by experiments/2026-11-15_fault_injection/scripts/make_envs.py"), name
        # nothing but comments and KRABBY_ lines (the launchers read ^KRABBY_ only)
        assert all(ln.startswith(("KRABBY_", "#")) or not ln.strip() for ln in text.splitlines()), name
        row = name.split("__")[1].removesuffix(".env").removesuffix("_L090")
        assert keys[:len(base_keys[row])] == base_keys[row]                          # the base block first, in order
        assert not [k for k in keys if k.startswith("KRABBY_PAYLOAD_")], name        # the payload comes from TP_LEVEL_FILE


def test_every_env_file_parses_at_launch_and_arms_exactly_its_cell():
    doc = _cells_doc()
    by_id = {c["id"]: c for c in doc["cells"]}
    stress_names = ("KRABBY_STRESS_VXY_ZERO", "KRABBY_STRESS_MU")
    n_armed = 0
    for p in sorted(ENVS_DIR.glob("*.env")):
        cid, rest = p.name.removesuffix(".env").split("__")
        row = rest.removesuffix("_L090")
        env = dict(ln.split("=", 1) for ln in p.read_text().splitlines() if ln.startswith("KRABBY_"))
        spec = fk.parse_fault_knobs(env, exclusive_with=stress_names)
        cell = by_id.get(cid)
        if cid == "NOM" or (cell is not None and not cell.get("fault")):
            assert spec is None, p.name                                              # nominals, T10b, T12: nothing armed
            continue
        want = cell["fault"] if cell else "S1:perm"
        assert spec["knob"] == want, p.name
        assert spec["onset_s"] == ([3000.0, 3001.0] if cid == "SMK_never" else
                                   [float(x) for x in doc["rows"][row]["onset"].split(":")]), p.name
        assert spec["from_start"] == (want.split(":")[0] in ("S12", "D1")), p.name
        n_armed += 1
    assert n_armed == 4 + (50 + 6 - 2) * 2 + sum(1 for c in doc["cells"] if c.get("kind_rows") and c.get("fault")) * 2


def test_the_t10b_anchor_and_t12_and_l090_conventions():
    doc = _cells_doc()
    t10 = (ENVS_DIR / "T10b__suite.env").read_text()
    assert "KRABBY_STRESS_VXY_ZERO=1" in t10 and "KRABBY_FAULT" not in t10             # a stress knob, never with a fault
    t12 = (ENVS_DIR / "T12__suite.env").read_text()
    assert "KRABBY_FAULT" not in t12 and "KRABBY_STRESS" not in t12                    # a checkpoint change, base lines alone
    assert next(c for c in doc["cells"] if c["id"] == "T12")["checkpoint"] == "C1S_zero_depth"
    # an _L090 file carries the same KRABBY lines as its L000 twin (the payload is gate_row's level file)
    nom0 = [ln for ln in (ENVS_DIR / "NOM__suite.env").read_text().splitlines() if ln.startswith("KRABBY_")]
    nom9 = [ln for ln in (ENVS_DIR / "NOM__suite_L090.env").read_text().splitlines() if ln.startswith("KRABBY_")]
    assert nom0 == nom9
    a5 = next(c for c in doc["cells"] if c["id"] == "A5_mid")
    assert a5["payload"] == "L090" and (ENVS_DIR / "A5_mid__suite_L090.env").is_file() \
        and not (ENVS_DIR / "A5_mid__suite.env").exists()
    assert doc["levels"]["L000"] is None and doc["levels"]["L090"].endswith("L090.env")
    # the anchors' onset is the row's standard table
    for cid in ("S12_30", "D1_1"):
        for row in ("suite", "field_light"):
            lines = (ENVS_DIR / f"{cid}__{row}.env").read_text().splitlines()
            assert f"KRABBY_FAULT_ONSET={doc['rows'][row]['onset']}" in lines and "KRABBY_FAULT_FROM_START=1" in lines
    # the smoke's window is beyond every row's episode: armed but never active
    assert doc["smokes"][0]["onset"] == "3000.0:3001.0"
