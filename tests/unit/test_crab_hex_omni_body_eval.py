"""The eval harness v3 -- body rows routed through the term, the two adapters, the body scenario file (CPU).

Robot-frame velocity commands campaign, step 2 (``crab_hex_omni_task/experiments/2026-10-15_body_cmd/
STEP2_SPEC.md``; PLAN D5 / D6). Pinned here:

* manifest v3 (``gait_eval.schedule``): the shipped ``scenarios_omni_body.yaml`` parses; envelope symbols
  resolve; every refusal (route, envelope, unknown keys, out-of-envelope holds, the grid exemption and
  its speed check, the hold-duration rule, kind / hand / grid-label consistency); compile / waves;
  v1 / v2 manifests load with no v3 field set;
* the harness's route check (every rows x head x adapter combination) and its pure helpers, lifted out of
  ``eval_crab_hex_gait.py`` with ``ast`` (it imports Isaac at module level): the per-env hook twins
  against the REAL hooks (the world twin bitwise), the ``term`` / ``body_hold`` / ``world_rate``
  installers on real command terms (the ``test_crab_hex_omni_command.py`` stub harness);
* the body-yaw metrics on synthetic traces, the report's ``body`` / ``adapter`` blocks and aggregates;
* ``eval_categories`` (CAT_KEYS cross-pins, rule v3, a body env's trained categories);
* the driver (``run_phases``): EVAL_CARRY frame keys, the encoding carry, the adapter pairing, the body
  rows, ``EVAL_PATHSPECS`` and ``resolution``'s required carry keys.

Nothing imports Isaac or launches a process; every module is loaded BY PATH under a private name.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
SCRIPTS_DIR = TASKS / "crab_hex_forward_task/scripts"
EVAL_SCRIPT = SCRIPTS_DIR / "eval_crab_hex_gait.py"
BODY_MANIFEST = OMNI / "experiments/eval/scenarios_omni_body.yaml"
OMNI_MANIFEST = OMNI / "experiments/eval/scenarios_omni.yaml"
RUN_PHASES = OMNI / "experiments/tools/run_phases.py"
RUN_CANDIDATES = OMNI / "experiments/tools/run_candidates.py"
RESOLUTION = OMNI / "experiments/tools/resolution.py"
EVAL_CATEGORIES = OMNI / "eval_categories.py"
COMMAND_SAMPLING = REPO / "parkour/parkour_isaaclab/envs/mdp/parkour_commands/command_sampling.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import metrics as M  # noqa: E402
from gait_eval import report as R  # noqa: E402
from gait_eval import schedule as S  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the command-term stub harness (idempotent stubs; the real omni command module + math under test)
H = _load(UNIT / "test_crab_hex_omni_command.py", "omni_cmd_harness__body_eval")
om = H.omni
mm = H.omni_math

DT = 0.02
ENV_TXT = "0.05:0.167:0.21"
CAP_TXT = "0@0.21,0.22@0.13,0.30@0.10"       # the provisional manifest envelope
ENVELOPE = mm.parse_body_envelope(ENV_TXT, CAP_TXT)
ELLIPSE = (0.35, 0.35, 0.25)
FLOOR = 0.2 + 0.01


# ============================================================ helpers: manifests
V3_HEAD = {
    "manifest_version": 3,
    "command_route": "term",
    "envelope": {"w_min": 0.05, "ceiling": 0.167, "w_max": 0.21, "cap": [[0.0, 0.21], [0.22, 0.13], [0.30, 0.10]]},
    "speed_ellipse": {"fwd": 0.35, "lat": 0.35, "back": 0.25},
    "defaults": {"prologue_s": 1.5, "settle_s": 1.0, "lin_vel_clip": 0.2},
}


def _timeline(holds, **row):
    return {"id": row.pop("id", "tl"), "task": "Isaac-Crab-Hex-Omni-Walk-Play-v0", "checkpoint": "x",
            "episodes": 4, "episode_length_s": 80.0, "schedule": holds, **row}


def _suite(categories, kind="category_suite", **row):
    n = sum(len(c["holds"]) for c in categories)
    return {"id": row.pop("id", "su"), "kind": kind, "task": "Isaac-Crab-Hex-Omni-Walk-Play-v0", "checkpoint": "x",
            "episodes": 4, "waves": row.pop("waves", n), "episode_length_s": 18.0,
            "settle": {"hold_s": 2.0, "label": "settle"}, "hold_s": row.pop("hold_s", 12.0),
            "categories": categories, **row}


def _write(tmp_path, scenarios, **top):
    doc = {**json.loads(json.dumps(V3_HEAD)), **top, "scenarios": scenarios}
    doc = {k: v for k, v in doc.items() if v is not None}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    return path


GOOD_HOLDS = [{"vx": 0.0, "vy": 0.0, "wz": 0.0, "hold_s": 8.0, "label": "stand"},
              {"vx": 0.25, "vy": 0.0, "wz": 0.0, "hold_s": 8.0, "label": "fwd"},
              {"vx": 0.0, "vy": 0.0, "wz": "ceiling", "hold_s": 12.0, "label": "turn_l_ceiling"},
              {"vx": 0.22, "vy": 0.0, "wz": "-cap80", "hold_s": 8.0, "label": "move_turn_r"}]


# ============================================================ manifest v3: the shipped file
def test_the_shipped_body_manifest_parses_with_the_spec_rows():
    scs, _ = S.load_manifest(BODY_MANIFEST)
    by_id = {s.id: s for s in scs}
    assert set(by_id) == {"suite__omni_body", "turnlevels__omni_body", "grid__omni_body",
                          "field_hard__omni_body", "field_light__omni_body", "ring_obst__omni_body"}
    for sc in scs:
        assert sc.command_route == "term" and sc.manifest_version == 3
        assert sc.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0" and sc.episodes == 100 and sc.env_seed == 1
        assert sc.spawn_yaw == "first_face" and sc.spawn_yaw_rad() == 0.0
        assert sc.get_speed_ellipse() == (0.35, 0.35, 0.25)
        # FROZEN at step 4 (CHARTER 2026-10-15, envelope_freeze.md); was provisional 0.167 / (0.22, 0.13) (0.30, 0.10)
        assert tuple(sc.envelope) == (0.05, 0.168, 0.21, ((0.0, 0.21), (0.22, 0.105), (0.30, 0.065)))
    suite = by_id["suite__omni_body"]
    assert suite.kind == "category_suite" and suite.waves == 10
    assert suite.suite.settle_hold_s == 2.0 and suite.suite.hold_s == 12.0
    cmds = {h.label: (h.vx, h.vy, round(h.wz, 12), h.wz_symbol) for h in suite.suite.holds()}
    cap80 = 0.8 * mm.body_cap_wz(0.22, suite.envelope)
    assert cmds == {"stand": (0.0, 0.0, 0.0, None), "fwd": (0.25, 0.0, 0.0, None),
                    "lat_l": (0.0, 0.22, 0.0, None), "lat_r": (0.0, -0.22, 0.0, None),
                    "back": (-0.22, 0.0, 0.0, None), "diag": (0.1767767, 0.1767767, 0.0, None),
                    "turn_l_ceiling": (0.0, 0.0, 0.168, "ceiling"), "turn_r_ceiling": (0.0, 0.0, -0.168, "-ceiling"),
                    "move_turn_l": (0.22, 0.0, round(cap80, 12), "cap80"),
                    "move_turn_r": (0.22, 0.0, round(-cap80, 12), "-cap80")}
    assert cap80 == pytest.approx(0.084)
    assert [c.name for c in suite.suite.categories] == ["stand", "fwd", "side", "back", "diag", "turn", "move_turn"]
    levels = by_id["turnlevels__omni_body"]
    assert levels.waves == 10 and [c.kind for c in levels.suite.categories] == ["turn"] * 5
    assert sorted(abs(h.wz) for h in levels.suite.holds())[::2] == [0.05, 0.08, 0.15, 0.168, 0.21]
    grid = by_id["grid__omni_body"]
    assert grid.kind == "envelope_grid" and grid.waves == 20 and S.uses_waves(grid)
    cells = {(round(math.hypot(h.vx, h.vy), 6), "lat" if h.vy else "fwd", h.wz) for h in grid.suite.holds()}
    want = {(0.0, "fwd", w) for w in (0.0, 0.08, 0.15, 0.30)} | {
        (v, d, w) for v in (0.22, 0.30) for d in ("fwd", "lat") for w in (0.0, 0.08, 0.15, 0.30)}
    assert cells == want and len(cells) == 20
    p2 = yaml.safe_load((OMNI / "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml").read_text())
    p2rows = {r["id"]: r for r in p2["scenarios"]}
    for body_id, world_id in (("field_hard__omni_body", "field_hard__omni"),
                              ("field_light__omni_body", "field_light__omni"),
                              ("ring_obst__omni_body", "ring_obst__omni")):
        sc = by_id[body_id]
        assert sc.env_vars == {k: str(v) for k, v in p2rows[world_id]["env"].items()}, body_id
        assert sc.episode_length_s == float(p2rows[world_id]["episode_length_s"])
        assert [(h.label, h.hold_s) for h in sc.schedule] == [
            ("stand", 8.0), ("fwd", 8.0), ("lat_l", 8.0), ("diag", 8.0), ("back", 8.0),
            ("turn_l_ceiling", 12.0), ("turn_r_ceiling", 12.0), ("move_turn_l", 8.0)]
        assert sc.schedule[-1].wz_symbol == "cap80" and sc.schedule[5].wz == 0.168


def test_v1_v2_manifests_load_with_no_v3_field():
    for path in (OMNI_MANIFEST, SCRIPTS_DIR.parent / "experiments/eval/scenarios_v1.yaml"):
        for sc in S.load_manifest(path)[0]:
            assert sc.command_route is None and sc.envelope is None and sc.manifest_version is None
            assert all(h.wz_symbol is None for h in sc.schedule)


def test_the_body_walk_margin_restates_the_command_sampling_constant():
    tree = ast.parse(COMMAND_SAMPLING.read_text())
    values = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "WALK_MARGIN_M_S" for t in n.targets)]
    assert values == [S.BODY_WALK_MARGIN_M_S]


# ============================================================ manifest v3: parse / validate
def test_symbols_resolve_from_the_envelope(tmp_path):
    scs, _ = S.load_manifest(_write(tmp_path, [_timeline(GOOD_HOLDS)]))
    sc = scs[0]
    assert [(h.wz, h.wz_symbol) for h in sc.schedule] == [
        (0.0, None), (0.0, None), (0.167, "ceiling"), (-0.8 * mm.body_cap_wz(0.22, ENVELOPE), "-cap80")]
    for sym, want in (("w_min", 0.05), ("-w_min", -0.05), ("w_max", 0.21), ("-w_max", -0.21), ("-ceiling", -0.167)):
        assert S.resolve_wz(sym, 0.0, 0.0, ENVELOPE, "t") == (want, sym)
    assert S.resolve_wz("cap80", 0.0, 0.0, ENVELOPE, "t") == (pytest.approx(0.8 * 0.21), "cap80")
    assert S.resolve_wz(0.1, 0.0, 0.0, ENVELOPE, "t") == (0.1, None)
    for bad in ("cap90", "0.1", True, None, float("nan")):
        with pytest.raises(S.ManifestError, match="wz"):
            S.resolve_wz(bad, 0.0, 0.0, ENVELOPE, "t")


def test_the_envelope_block_is_parsed_through_the_step1_grammar():
    env = S.parse_envelope_block(V3_HEAD["envelope"])
    assert env == mm.parse_body_envelope(ENV_TXT, CAP_TXT)
    assert S.envelope_meta(env) == {"w_min": 0.05, "ceiling": 0.167, "w_max": 0.21,
                                    "cap": [[0.0, 0.21], [0.22, 0.13], [0.3, 0.1]]}
    for mutate, match in ((lambda b: b.update(w_min=0.2), "w_min"), (lambda b: b.pop("cap"), "missing"),
                          (lambda b: b.update(extra=1), "unknown"), (lambda b: b.update(cap=[[0.1, 0.2]]), "v = 0"),
                          (lambda b: b.update(cap=[[0.0, 0.3]]), "w_max"), (lambda b: b.update(cap=[0.0]), "pairs")):
        block = json.loads(json.dumps(V3_HEAD["envelope"]))
        mutate(block)
        with pytest.raises(S.ManifestError, match=match):
            S.parse_envelope_block(block)


@pytest.mark.parametrize("top, rows, match", [
    ({"command_route": None}, [_timeline(GOOD_HOLDS)], "manifest_version 3 requires"),
    ({"command_route": "hook"}, [_timeline(GOOD_HOLDS)], "command_route"),
    ({"ang_vel_max": 0.3}, [_timeline(GOOD_HOLDS)], "ang_vel_max"),
    ({"speed_ellipse": None}, [_timeline(GOOD_HOLDS)], "speed_ellipse"),
    ({"envelope": None}, [_timeline(GOOD_HOLDS)], "envelope"),
    ({}, [_timeline(GOOD_HOLDS, rotate_holds_by_env=True)], "rotate_holds_by_env"),
    ({}, [_timeline(GOOD_HOLDS, speed_ellipse={"fwd": 0.35, "lat": 0.35, "back": 0.25})], "speed_ellipse"),
    ({}, [_timeline([{"vx": 0.25, "vy": 0.0, "hold_s": 8.0, "label": "fwd"}])], "missing \\['wz'\\]"),
    ({}, [_timeline([{"speed": 0.25, "dir": 0.0, "face": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0, "hold_s": 8.0,
                      "label": "w"}])], "unknown"),
    ({}, [_timeline([{"vx": 0.0, "vy": 0.0, "wz": 0.3, "hold_s": 8.0, "label": "t"}])], "outside the body envelope"),
    ({}, [_timeline([{"vx": 0.0, "vy": 0.0, "wz": 0.02, "hold_s": 8.0, "label": "t"}])], "outside the body envelope"),
    ({}, [_timeline([{"vx": 0.15, "vy": 0.0, "wz": 0.0, "hold_s": 8.0, "label": "f"}])], "outside the body envelope"),
    ({}, [_timeline([{"vx": 0.22, "vy": 0.0, "wz": 0.2, "hold_s": 8.0, "label": "mt"}])], "outside the body envelope"),
    ({}, [_timeline([{"vx": 0.0, "vy": 0.3, "wz": 0.0, "hold_s": 8.0, "label": "l"}])], None),     # lat 0.35: OK
    ({}, [_timeline([{"vx": -0.3, "vy": 0.0, "wz": 0.0, "hold_s": 8.0, "label": "b"}])], "outside the body envelope"),
    ({}, [_timeline([{"vx": 0.25, "vy": 0.0, "wz": 0.0, "hold_s": 5.9, "label": "f"}])], "shorter than"),
    ({}, [_timeline([{"vx": 0.25, "vy": 0.0, "wz": 0.0, "hold_s": 6.0, "label": "f"}])], None),    # the rule's edge
    ({}, [_timeline(GOOD_HOLDS, episode_length_s=30.0)], "episode_length_s"),
    ({}, [_timeline(GOOD_HOLDS, probe="yaw")], "probe"),
])
def test_v3_refusals(tmp_path, top, rows, match):
    path = _write(tmp_path, rows, **top)
    if match is None:
        assert S.load_manifest(path)[0]
        return
    with pytest.raises(S.ManifestError, match=match):
        S.load_manifest(path)


def test_the_hold_duration_rule_reads_the_larger_of_settle_s_and_the_body_window(tmp_path):
    rows = [_timeline([{"vx": 0.25, "vy": 0.0, "wz": 0.0, "hold_s": 9.0, "label": "f"}],
                      defaults={"settle_s": 7.5})]
    with pytest.raises(S.ManifestError, match="9.5"):
        S.load_manifest(_write(tmp_path, rows))
    cats = [{"name": "fwd", "kind": "translate", "holds": [{"label": "fwd", "vx": 0.25, "vy": 0.0, "wz": 0.0}]}]
    with pytest.raises(S.ManifestError, match="shorter than"):
        S.load_manifest(_write(tmp_path, [_suite(cats, hold_s=5.0)]))


def _cat(name, kind, *holds):
    return {"name": name, "kind": kind, "holds": list(holds)}


SUITE_CATS = [
    _cat("stand", "stand", {"label": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0}),
    _cat("side", "translate", {"label": "lat_l", "vx": 0.0, "vy": 0.22, "wz": 0.0, "hand": "l"},
         {"label": "lat_r", "vx": 0.0, "vy": -0.22, "wz": 0.0, "hand": "r"}),
    _cat("turn", "turn", {"label": "turn_l", "vx": 0.0, "vy": 0.0, "wz": "w_min", "hand": "l"},
         {"label": "turn_r", "vx": 0.0, "vy": 0.0, "wz": "-w_min", "hand": "r"}),
]


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c[1]["holds"][0].update(hand="r"), "unbalanced|hand"),
    (lambda c: (c[2]["holds"][0].update(hand="r"), c[2]["holds"][1].update(hand="l")), "disagrees"),
    (lambda c: c[1].update(kind="turn"), "is a translate command"),
    (lambda c: c[0]["holds"][0].update(wz=0.1), "is a turn command"),
    (lambda c: c[2]["holds"][0].update(wz=0.3), "outside the body envelope"),
    (lambda c: c[0].update(name="Stand.x"), "must be"),
    (lambda c: c[2]["holds"][0].update(speed=0.0), "unknown"),
])
def test_v3_suite_refusals(tmp_path, mutate, match):
    cats = json.loads(json.dumps(SUITE_CATS))
    mutate(cats)
    with pytest.raises(S.ManifestError, match=match):
        S.load_manifest(_write(tmp_path, [_suite(cats)]))


def test_the_grid_is_exempt_from_the_yaw_rate_checks_but_not_from_speed_or_its_labels(tmp_path):
    cell = lambda v, d, w: _cat(S.grid_label(v if d == "fwd" else 0.0, v if d == "lat" else 0.0, w),  # noqa: E731
                                ("move_turn" if w else "translate") if v else ("turn" if w else "stand"),
                                {"label": S.grid_label(v if d == "fwd" else 0.0, v if d == "lat" else 0.0, w),
                                 "vx": v if d == "fwd" else 0.0, "vy": v if d == "lat" else 0.0, "wz": w})
    ok = [cell(0.0, "fwd", 0.30), cell(0.30, "lat", 0.30), cell(0.22, "fwd", 0.15)]
    assert [c["name"] for c in ok] == ["g_v0_w0p3", "g_v0p3_lat_w0p3", "g_v0p22_fwd_w0p15"]
    sc = S.load_manifest(_write(tmp_path, [_suite(ok, kind="envelope_grid")]))[0][0]
    assert sc.kind == "envelope_grid" and sc.waves == 3
    # the same 0.30 rad/s cell on a category suite is refused (the envelope applies)
    with pytest.raises(S.ManifestError, match="outside the body envelope"):
        S.load_manifest(_write(tmp_path, [_suite(ok[:1])]))
    # speed is still checked on the grid
    fast = [cell(0.36, "fwd", 0.0)]
    with pytest.raises(S.ManifestError, match="speed"):
        S.load_manifest(_write(tmp_path, [_suite(fast, kind="envelope_grid")]))
    # a cell must carry its derived label/name, one hold, no hand
    bad = json.loads(json.dumps(ok))
    bad[0]["holds"][0]["label"] = "spin"
    with pytest.raises(S.ManifestError, match="labelled"):
        S.load_manifest(_write(tmp_path, [_suite(bad, kind="envelope_grid")]))
    with pytest.raises(S.ManifestError, match="not a fwd / lat"):
        S.grid_label(0.2, 0.2, 0.0)


# ============================================================ manifest v3: compile
def test_a_v3_timeline_compiles_to_body_arrays_and_no_cmd(tmp_path):
    sc = S.load_manifest(_write(tmp_path, [_timeline(GOOD_HOLDS)]))[0][0]
    comp = S.compile_schedule(sc, dt=DT, num_envs=3)
    assert comp.body_route and comp.world_dir is None and comp.command_frame == "body"
    assert comp.n_steps == 75 + 400 + 400 + 600 + 400
    assert not comp.cmd.any() and comp.cmd.dtype == np.float32
    assert comp.body_vx.dtype == np.float64 and comp.body_vx.shape == (comp.n_steps, 3)
    assert comp.body_row(0) == S.BodyScheduleRow(vx=0.0, vy=0.0, wz=0.0, label="prologue", step=0)
    r = comp.body_row(75 + 400)
    assert (r.vx, r.vy, r.wz, r.label) == (0.25, 0.0, 0.0, "fwd") and r["vx"] == 0.25 and dict(r)["wz"] == 0.0
    assert comp.body_row(comp.n_steps + 10).label == "move_turn_r"
    assert comp.body_wz[75 + 800, 0] == 0.167                       # float64: the manifest's number, exactly
    assert comp.steady_mask[75 + 50, 0] and not comp.steady_mask[75 + 49, 0]  # settle_s 1.0 after each hold start
    assert (comp.segment_id[:75] == -1).all() and (comp.segment_id[75:475] == 0).all()
    assert (comp.body_vx == comp.body_vx[:, :1]).all()               # uniform across envs


def test_a_v3_suite_compiles_per_wave_and_stacks(tmp_path):
    sc = S.load_manifest(_write(tmp_path, [_suite(SUITE_CATS)]))[0][0]
    with pytest.raises(ValueError, match="compile_wave"):
        S.compile_schedule(sc, dt=DT)
    parts = [S.compile_wave(sc, k, dt=DT, num_envs=2) for k in range(sc.waves)]
    assert [p.body_row(500).label for p in parts] == ["stand", "lat_l", "lat_r", "turn_l", "turn_r"]
    assert parts[3].body_row(80).label == "settle" and parts[3].body_row(80).wz == 0.0
    st = S.stack_waves(parts)
    assert st.body_vx.shape == (775, 10) and st.world_dir is None and list(st.hold_of_env) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    meta = S.suite_run_meta(sc, dt=DT, num_envs=2, wave_seeds=[1] * 5, trained_categories=None)
    assert meta["suite"]["categories"][2]["holds"][0] == {"label": "turn_l", "vx": 0.0, "vy": 0.0, "wz": 0.05,
                                                          "wz_symbol": "w_min", "hand": "l"}
    # a legacy wave cannot be stacked with a body one
    legacy = S.load_manifest(OMNI_MANIFEST)[0]
    w = S.compile_wave([s for s in legacy if s.id == "suite__omni"][0], 0, dt=DT, num_envs=2)
    with pytest.raises(ValueError, match="does not share|compile_wave schedule"):
        S.stack_waves([parts[0], w])
    import dataclasses  # noqa: PLC0415 -- same shape, the other frame
    mixed = dataclasses.replace(w, hold_labels=list(parts[0].hold_labels), n_steps=parts[0].n_steps)
    with pytest.raises(ValueError, match="compile_wave schedule"):
        S.stack_waves([parts[0], mixed])


# ============================================================ the harness, lifted with ast
TREE = ast.parse(EVAL_SCRIPT.read_text())
LIFT_FUNCS = ("_env_command_frame", "_route_check", "_world_hook_rows", "_body_hook_rows", "_row_label",
              "_angle_dev", "_install_schedule_override_body", "_install_adapter_body_hold",
              "_install_adapter_world_rate", "_body_cmd_series", "_body_run_meta", "_world_rate_cap")
LIFT_CONSTS = ("COMMAND_ADAPTERS", "ADAPTER_TURNS", "COMMAND_ROUTES_RUN", "BODY_HOLD_WINDUP_RAD", "OBS_CMD_COLS")


def _lift():
    nodes = []
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in LIFT_FUNCS:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in LIFT_CONSTS for t in node.targets):
            nodes.append(node)
    found = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    assert found == set(LIFT_FUNCS)
    ns: dict = {"torch": torch, "math": math, "types": types, "np": np, "S": S}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVAL_SCRIPT), "exec"), ns)  # noqa: S102
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


HG = _lift()


def test_the_harness_declares_the_two_flags_and_the_route_dispatch():
    src = EVAL_SCRIPT.read_text()
    assert '"--command-adapter",\n    choices=["none", "body_hold", "world_rate"],\n    default="none",' in src
    assert '"--adapter-turns",\n    choices=["on", "off"],\n    default=None,' in src
    assert HG.COMMAND_ADAPTERS == ("none", "body_hold", "world_rate") and HG.BODY_HOLD_WINDUP_RAD == 0.6
    assert HG.OBS_CMD_COLS == (5, 11)
    main_src = src.split("def main() -> None:")[1]
    # the legacy install is reached only through the unchanged `world_frame` / body-row branches
    assert "    elif world_frame:\n        override_state = _install_schedule_override_world(" in main_src
    assert "    else:\n        override_state = _install_schedule_override(cmd_term, cmd_tensor)" in main_src
    assert "hook_driven = world_frame or route in (\"term\", \"body_hold\")" in main_src


# route-check table: (rows, head frame, adapter, grid) -> route or None (= SystemExit)
ROUTES = {
    ("v3", "body", "none", False): "term",
    ("v3", "body", "none", True): None,
    ("v3", "world", "body_hold", False): "body_hold",
    ("v3", "world", "body_hold", True): "body_hold",
    ("world", "world", "none", False): "legacy",
    ("world", None, "none", False): "legacy",
    ("world", "body", "world_rate", False): "world_rate",
    ("body", "world", "none", False): "legacy",
    ("body", None, "none", False): "legacy",
}


@pytest.mark.parametrize("rows", ["v3", "world", "body"])
@pytest.mark.parametrize("frame", ["body", "world", None])
@pytest.mark.parametrize("adapter", ["none", "body_hold", "world_rate"])
@pytest.mark.parametrize("grid", [False, True])
def test_route_check_table(rows, frame, adapter, grid):
    if grid and rows != "v3":
        return
    want = ROUTES.get((rows, frame, adapter, grid))
    if want is None:
        with pytest.raises(SystemExit, match="INVALID EVAL"):
            HG._route_check(rows, frame, adapter, grid=grid)
    else:
        assert HG._route_check(rows, frame, adapter, grid=grid) == want


def test_route_check_refuses_bad_flags():
    with pytest.raises(SystemExit, match="--command-adapter"):
        HG._route_check("world", "world", "rate")
    with pytest.raises(SystemExit, match="world_rate only"):
        HG._route_check("world", "world", "none", adapter_turns="off")
    with pytest.raises(SystemExit, match="world_rate only"):
        HG._route_check("v3", "world", "body_hold", adapter_turns="off")
    assert HG._route_check("world", "body", "world_rate", adapter_turns="off") == "world_rate"
    assert HG._route_check("world", "world", "none", adapter_turns="on") == "legacy"


# ============================================================ real command terms
def _fake_env(n, yaw=None):
    robot = H._Robot(n, yaw)
    return types.SimpleNamespace(num_envs=n, device="cpu", scene=H._Scene(robot), step_dt=DT, max_episode_length=4000,
                                 episode_length_buf=torch.ones(n, dtype=torch.long)), robot


def _world_term(n=6, yaw=None, **over):
    cfg = H._cfg(speed_ellipse=ELLIPSE, **over)
    env, robot = _fake_env(n, yaw)
    term = om.CrabHexOmniCommand(cfg, env)
    term.time_left = torch.zeros(n)
    return term, robot, env


def _body_term(n=6, yaw=None):
    cfg = om.CrabHexOmniBodyCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), heading_control_stiffness=1.5,
        ranges=H.ParkourCommandCfg.Ranges(lin_vel_x=(0.0, 0.35), heading=(0.0, 0.0)),
        body_w_min=ENVELOPE.w_min, body_ceiling=ENVELOPE.ceiling, body_w_max=ENVELOPE.w_max, body_cap=ENVELOPE.cap)
    cfg.speed_ellipse = ELLIPSE
    env, robot = _fake_env(n, yaw)
    term = om.CrabHexOmniBodyCommand(cfg, env)
    term.time_left = torch.zeros(n)
    return term, robot, env


def _spread(n):
    return H._spread_yaw(n)


def test_env_command_frame():
    assert HG._env_command_frame(_world_term()[0]) == "world"
    assert HG._env_command_frame(_body_term()[0]) == "body"
    assert HG._env_command_frame(types.SimpleNamespace()) is None


@pytest.mark.parametrize("row", [(0.25, 0.0, 0.0), (0.22, 1.5707963, 0.3), (0.0, 0.0, 1.2), (0.25, 3.0, -2.0)])
def test_the_world_hook_twin_is_the_real_hook_bitwise_on_uniform_rows(row):
    n = 64
    real, _r1, _e1 = _world_term(n, _spread(n))
    twin, _r2, _e2 = _world_term(n, _spread(n))
    speed, direction, face = row
    real.schedule_hook({"speed": speed, "dir": direction, "face": face})
    full = lambda v: torch.full((n,), float(v))  # noqa: E731
    HG._world_hook_rows(twin, full(speed), full(direction), full(face))
    for name in ("travel_dir_w", "speed_cmd", "heading_target", "type_id", "slot_id", "vel_command_b", "phi_b",
                 "recap_mask"):
        assert torch.equal(getattr(real, name), getattr(twin, name)), name
    with pytest.raises(ValueError, match="non-finite|negative"):
        HG._world_hook_rows(twin, full(-0.1), full(0.0), full(0.0))


@pytest.mark.parametrize("row", [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, -0.22, 0.0), (0.0, 0.0, 0.167),
                                 (0.22, 0.0, -0.104), (0.1767767, 0.1767767, 0.0)])
def test_the_body_hook_twin_matches_the_real_hook_on_uniform_rows(row):
    n = 16
    real, robot, _e = _body_term(n, _spread(n))
    twin, _r, _e2 = _body_term(n, _spread(n))
    real.schedule_hook_body(*row)
    full = lambda v: torch.full((n,), float(v))  # noqa: E731
    HG._body_hook_rows(twin, *(full(v) for v in row), mm, ELLIPSE, FLOOR)
    for name in ("vel_command_b", "slot_id", "type_id", "body_wz", "heading_target", "recap_mask"):
        assert torch.equal(getattr(real, name), getattr(twin, name)), name
    for name in ("body_speed", "body_phi", "travel_dir_w", "speed_cmd", "phi_b"):
        assert torch.allclose(getattr(real, name), getattr(twin, name), atol=1e-7), name
    bad = full(0.0)
    bad[3] = 0.3
    with pytest.raises(ValueError, match="env 3"):
        HG._body_hook_rows(twin, full(0.0), full(0.0), bad, mm, ELLIPSE, FLOOR)


def _v3_compiled(tmp_path, holds, num_envs):
    sc = S.load_manifest(_write(tmp_path, [_timeline(holds)]))[0][0]
    return sc, S.compile_schedule(sc, dt=DT, num_envs=num_envs)


def test_the_term_route_hooks_every_step_and_checks_its_state(tmp_path):
    n = 4
    term, _robot, _env = _body_term(n, _spread(n))
    _sc, comp = _v3_compiled(tmp_path, GOOD_HOLDS, n)
    st = HG._install_schedule_override_body(term, comp, "cpu", comp.hold_labels)
    assert torch.equal(term.vel_command_b, torch.zeros(n, 3))
    for t in (0, 100, 75 + 400, 75 + 800 + 10, comp.n_steps - 1):
        st["step"] = t
        term.compute(DT)
        r = comp.body_row(t)
        assert torch.equal(term.vel_command_b, torch.tensor([[r.vx, r.vy, r.wz]] * n, dtype=torch.float32))
        assert st["state_dev"](t) < 1e-6
    assert int(term.slot_id[0]) == mm.slot_from_body_cmd(torch.tensor([r.vx]), torch.tensor([r.vy]),
                                                         torch.tensor([r.wz]))[0]
    term.body_wz[:] = 0.0                      # a leaked resample shows up in the state check
    assert st["state_dev"](comp.n_steps - 1) > 0.05
    world, _rw, _ew = _world_term(n)
    with pytest.raises(SystemExit, match="BODY command term"):
        HG._install_schedule_override_body(world, comp, "cpu", comp.hold_labels)


def test_body_hold_integrates_with_anti_windup_carries_the_face_and_restarts_it_on_reset(tmp_path):
    n = 3
    world, robot, env = _world_term(n, torch.zeros(n))
    holds = [{"vx": 0.0, "vy": 0.0, "wz": 0.1, "hold_s": 8.0, "label": "turn"},
             {"vx": 0.25, "vy": 0.0, "wz": 0.0, "hold_s": 8.0, "label": "fwd"}]
    sc, comp = _v3_compiled(tmp_path, holds, n)
    st = HG._install_adapter_body_hold(world, comp, "cpu", comp.hold_labels, mm=mm, spawn_yaw=sc.spawn_yaw_rad())
    assert torch.equal(st["face"], torch.zeros(n)) and torch.equal(world.heading_target, torch.zeros(n))
    faces = []
    for t in range(75, 75 + 400):            # the yaw never moves: the target runs into the windup bound
        st["step"] = t
        world.compute(DT)
        faces.append(float(st["face"][0]))
        assert st["state_dev"](t) == 0.0
    assert faces[0] == pytest.approx(0.1 * DT, abs=1e-6)                 # float32 state
    assert max(faces) == pytest.approx(0.6, abs=1e-6) and faces[-1] == pytest.approx(0.6, abs=1e-6)
    assert faces[49] == pytest.approx(50 * 0.1 * DT, abs=1e-4)             # float32 wrap per step
    assert torch.equal(world.heading_target, st["face"]) and torch.equal(st["record"]["adapter_face"](), st["face"])
    # the robot follows half way: the face stays at most windup ahead of the yaw
    robot.data.heading_w[:] = 0.3
    st["step"] = 75 + 399                     # still the turn hold
    world.compute(DT)
    assert float(st["face"][0]) == pytest.approx(min(0.6 + 0.1 * DT, 0.3 + 0.6), abs=1e-5)
    robot.data.heading_w[:] = -0.2            # the robot turned away: the lead is clamped to +windup
    world.compute(DT)
    assert float(st["face"][0]) == pytest.approx(-0.2 + 0.6, abs=1e-6)
    # carried across the hold change (wz 0: the target holds), and the travel dir rides on the yaw
    st["step"] = 75 + 400 + 10
    before = st["face"].clone()
    env.episode_length_buf[1] = 0             # env 1 was just reset: its face restarts at the spawn yaw
    world.compute(DT)
    assert float(st["face"][0]) == pytest.approx(float(before[0]), abs=1e-6)
    assert float(st["face"][1]) == 0.0
    assert torch.allclose(world.travel_dir_w, robot.data.heading_w) and torch.allclose(world.speed_cmd, torch.full((n,), 0.25))
    assert bool((world.slot_id == mm.SLOT_MOVE_TURN).any() | (world.slot_id == mm.SLOT_MOVE).any())
    # the pre-loop apply re-seeds at the spawn yaw without integrating
    st["apply"](0)
    assert torch.equal(st["face"], torch.zeros(n))


def test_body_hold_refuses_a_body_term_and_an_armed_hook_knob(tmp_path):
    sc, comp = _v3_compiled(tmp_path, GOOD_HOLDS, 2)
    body, _r, _e = _body_term(2)
    with pytest.raises(SystemExit, match="WORLD omni command term"):
        HG._install_adapter_body_hold(body, comp, "cpu", comp.hold_labels, mm=mm, spawn_yaw=0.0)
    world, _rw, _ew = _world_term(2)
    world._slot_latch = True
    with pytest.raises(SystemExit, match="eval_slot_latch"):
        HG._install_adapter_body_hold(world, comp, "cpu", comp.hold_labels, mm=mm, spawn_yaw=0.0)


def _world_compiled(tmp_path, holds, num_envs):
    doc = {"manifest_version": 2, "speed_ellipse": {"fwd": 0.35, "lat": 0.25, "back": 0.25}, "ang_vel_max": 0.3,
           "scenarios": [{"id": "w", "task": "t", "checkpoint": "x", "episodes": num_envs, "episode_length_s": 80.0,
                          "schedule": holds}]}
    path = tmp_path / "w.json"
    path.write_text(json.dumps(doc))
    sc = S.load_manifest(path)[0][0]
    return sc, S.compile_schedule(sc, dt=DT, num_envs=num_envs)


WORLD_HOLDS = [{"speed": 0.25, "dir": 0.0, "face": 0.0, "hold_s": 8.0, "label": "fwd"},
               {"speed": 0.0, "dir": 0.0, "face": 1.5707963, "hold_s": 12.0, "label": "turn_l90"},
               {"speed": 0.22, "dir": 0.0, "face": 0.0, "hold_s": 12.0, "label": "move_turn"}]


def test_world_rate_is_the_world_terms_command_shaped_into_the_heads_envelope(tmp_path):
    n = 32
    yaw = _spread(n) * 0.5
    body, _rb, _eb = _body_term(n, yaw)
    world, _rw, _ew = _world_term(n, yaw)
    sc, comp = _world_compiled(tmp_path, WORLD_HOLDS, n)
    st = HG._install_adapter_world_rate(body, comp, "cpu", comp.hold_labels, mm=mm, ang_vel_max=sc.ang_vel_max,
                                        turns=True, walk_floor=FLOOR)
    for t in (100, 75 + 400 + 5, 75 + 400 + 600 + 5):
        st["step"] = t
        body.compute(DT)
        world.schedule_hook(comp.row(t))
        # before shaping: the world term's own body command (bitwise: the same math on the same inputs)
        assert torch.equal(st["record"]["adapter_wz_raw"](), world.vel_command_b[:, 2]), t
        unclamped = ~st["record"]["adapter_clipped"]()
        assert torch.equal(body.vel_command_b[unclamped], world.vel_command_b[unclamped]), t
        # shaped: always inside the head's envelope, xy untouched where only wz was clamped
        c = body.vel_command_b
        assert not bool(mm.body_cmd_violation_mask(c[:, 0], c[:, 1], c[:, 2], ENVELOPE, ELLIPSE, FLOOR).any())
        assert torch.equal(c[:, :2], world.vel_command_b[:, :2]), t
        assert st["state_dev"](t) < 1e-6
        # the facing is scored against the WORLD row's face
        assert torch.equal(st["heading_target"](), torch.full((n,), float(comp.row(t).face)))
    clipped = st["record"]["adapter_clipped"]()
    assert bool(clipped.any())                                  # the relay's 0.3 is above w_max 0.21
    wz_raw = st["record"]["adapter_wz_raw"]()
    assert torch.equal(clipped, (body.vel_command_b[:, 2] - wz_raw).abs() > 1e-9)


def test_world_rate_with_turns_off_commands_no_yaw_rate_and_counts_it_clipped(tmp_path):
    n = 8
    body, _rb, _eb = _body_term(n, _spread(n) * 0.5)
    sc, comp = _world_compiled(tmp_path, WORLD_HOLDS, n)
    st = HG._install_adapter_world_rate(body, comp, "cpu", comp.hold_labels, mm=mm, ang_vel_max=sc.ang_vel_max,
                                        turns=False, walk_floor=FLOOR)
    st["step"] = 75 + 400 + 5
    body.compute(DT)
    assert torch.equal(body.vel_command_b[:, 2], torch.zeros(n))
    raw = st["record"]["adapter_wz_raw"]()
    assert bool((raw != 0).any())
    assert torch.equal(st["record"]["adapter_clipped"](), raw != 0)
    world, _rw, _ew = _world_term(n)
    with pytest.raises(SystemExit, match="BODY command term"):
        HG._install_adapter_world_rate(world, comp, "cpu", comp.hold_labels, mm=mm, ang_vel_max=0.3, turns=True,
                                       walk_floor=FLOOR)


def test_body_cmd_series_and_run_meta(tmp_path):
    sc, comp = _v3_compiled(tmp_path, GOOD_HOLDS, 2)
    series = HG._body_cmd_series("term", comp, {})
    assert series.shape == (comp.n_steps, 2, 3) and series.dtype == np.float64
    assert np.array_equal(series[..., 2], comp.body_wz)
    host = {"cmd_applied": np.ones((5, 2, 3), dtype=np.float32)}
    assert HG._body_cmd_series("world_rate", comp, host).dtype == np.float64
    assert HG._body_cmd_series("legacy", comp, host) is None
    body, _r, _e = _body_term(2)
    meta = HG._body_run_meta(sc, route="term", adapter="none", adapter_turns="on", env_frame="body",
                             env_envelope=body.body_envelope, cmd_cfg=body.cfg, applied_overrides={"spawn_yaw_pin": 0.0},
                             max_sched_state_dev=0.0, walk_floor=FLOOR)
    assert meta["command_route"] == "term" and meta["command_adapter"] == "none" and meta["adapter_turns"] is None
    assert meta["env_command_frame"] == "body" and meta["envelope_manifest"] == V3_HEAD["envelope"] | {
        "cap": [[0.0, 0.21], [0.22, 0.13], [0.3, 0.1]]}
    assert meta["envelope_env"]["w_max"] == 0.21 and meta["obs_cmd_cols"] == [5, 6, 7, 8, 9, 10]
    assert meta["hold_body_cmds"][2] == {"label": "turn_l_ceiling", "vx": 0.0, "vy": 0.0, "wz": 0.167,
                                         "wz_symbol": "ceiling", "hold_s": 12.0}
    wr = HG._body_run_meta(sc, route="world_rate", adapter="world_rate", adapter_turns="off", env_frame="body",
                           env_envelope=body.body_envelope, cmd_cfg=body.cfg, applied_overrides={},
                           max_sched_state_dev=0.0, walk_floor=FLOOR)
    assert wr["adapter"]["turns"] is False and wr["adapter_turns"] == "off"


# ============================================================ metrics: body yaw
def _yaw_quat(yaw):
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def _trace(n, wz, v=(0.0, 0.0), wobble=0.0):
    t = np.arange(n) * DT
    w = np.full(n, float(wz)) + wobble * np.sin(2 * np.pi * 0.5 * t)   # 4 whole periods in 8 s
    yaw = np.concatenate([[0.0], np.cumsum(w[:-1]) * DT])
    ang = np.zeros((n, 3))
    ang[:, 2] = w
    vel = np.zeros((n, 3))
    vel[:, 0], vel[:, 1] = v
    pos = np.zeros((n, 3))
    return _yaw_quat(yaw), ang, vel, pos


def test_body_yaw_metrics_on_a_constant_rate_turn():
    q, ang, vel, pos = _trace(400, 0.15)
    for sign in (1.0, -1.0):
        cmd = np.tile([0.0, 0.0, sign * 0.15], (400, 1))
        b = M.body_yaw_metrics(q if sign > 0 else _yaw_quat(-np.unwrap(M.yaw_from_quat_wxyz(q))),
                               ang * sign, vel, pos, cmd, DT)
        assert b["net_yaw_rate"] == pytest.approx(sign * 0.15) and b["net_yaw_ratio"] == pytest.approx(1.0)
        assert b["wz_ratio"] == pytest.approx(1.0) and b["yaw_ripple"] == pytest.approx(0.0, abs=1e-12)
        assert b["heading_drift"] is None and b["planar_drift"] == 0.0 and b["body_ratio"] is None
        assert b["window_s"] == pytest.approx(399 * DT) and b["n_steps"] == 400
    # turning the WRONG way is a negative ratio per hand
    cmd = np.tile([0.0, 0.0, -0.15], (400, 1))
    assert M.body_yaw_metrics(q, ang, vel, pos, cmd, DT)["net_yaw_ratio"] == pytest.approx(-1.0)


def test_body_yaw_metrics_ripple_drift_and_body_ratio():
    q, ang, vel, pos = _trace(400, 0.1, wobble=0.2)
    b = M.body_yaw_metrics(q, ang, vel, pos, np.tile([0.0, 0.0, 0.1], (400, 1)), DT)
    assert b["yaw_ripple"] > 0.01 and b["net_yaw_ratio"] == pytest.approx(1.0, abs=0.02)
    means = ang[:, 2].reshape(8, 50).mean(axis=1)
    assert b["yaw_ripple"] == pytest.approx(float(np.std(means)))
    # straight hold: heading drift = |mean wz|; body ratio / cross on the commanded bearing
    q, ang, vel, pos = _trace(400, -0.01, v=(0.18, 0.02))
    b = M.body_yaw_metrics(q, ang, vel, pos, np.tile([0.2, 0.0, 0.0], (400, 1)), DT)
    assert b["heading_drift"] == pytest.approx(0.01) and b["net_yaw_ratio"] is None and b["wz_ratio"] is None
    assert b["body_ratio"] == pytest.approx(0.9) and b["cross_body"] == pytest.approx(0.02)
    # lateral bearing: the projection follows atan2(vy, vx)
    b = M.body_yaw_metrics(q, ang, vel, pos, np.tile([0.0, 0.2, 0.0], (400, 1)), DT)
    assert b["body_ratio"] == pytest.approx(0.1) and b["cross_body"] == pytest.approx(0.18)
    # pure turn: planar drift = |xy displacement| / T
    q, ang, vel, pos = _trace(101, 0.1)
    pos[:, 0] = np.linspace(0.0, 0.2, 101)
    b = M.body_yaw_metrics(q, ang, vel, pos, np.tile([0.0, 0.0, 0.1], (101, 1)), DT)
    assert b["planar_drift"] == pytest.approx(0.2 / (100 * DT))
    assert M.body_yaw_metrics(q[:0], ang[:0], vel[:0], pos[:0], np.zeros((0, 3)), DT) is None
    assert M.body_yaw_metrics(q[:60], ang[:60], vel[:60], pos[:60], np.tile([0, 0, 0.1], (60, 1)), DT)["yaw_ripple"] is None


# ============================================================ report: the body / adapter blocks
def _v3_raw(comp, n_envs, wz_scale=1.0):
    T = comp.n_steps
    wz = comp.body_wz * wz_scale
    yaw = np.concatenate([np.zeros((1, n_envs)), np.cumsum(wz[:-1], axis=0) * DT])
    ang = np.zeros((T, n_envs, 3))
    ang[..., 2] = wz
    vel = np.zeros((T, n_envs, 3))
    vel[..., 0], vel[..., 1] = comp.body_vx, comp.body_vy
    cmd = np.stack([comp.body_vx, comp.body_vy, comp.body_wz], axis=-1)
    contact = (np.arange(T)[:, None, None] // 10 + np.arange(6)[None, None, :]) % 2 == 0
    return {
        "foot_force_norm": np.broadcast_to(np.where(contact, 400.0, 0.0), (T, n_envs, 6)).copy(),
        "cmd_applied": cmd.copy(), "root_lin_vel_b": vel, "root_ang_vel_b": ang, "root_quat_w": _yaw_quat(yaw),
        "root_pos_w": np.zeros((T, n_envs, 3)), "foot_pos_w": np.zeros((T, n_envs, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, n_envs, 6, 3)), "foot_ang_vel_w": np.zeros((T, n_envs, 6, 3)),
        "terrain_z": np.zeros((T, n_envs, 6)), "actions": np.zeros((T, n_envs, 4)),
        "joint_vel": np.zeros((T, n_envs, 4)), "crab_failure": np.zeros((T, n_envs), dtype=bool),
        "travel_yaw": np.arctan2(cmd[..., 1], cmd[..., 0]),
    }


def test_score_run_adds_the_body_block_over_the_hold_relative_window(tmp_path):
    sc, comp = _v3_compiled(tmp_path, GOOD_HOLDS, 2)
    raw = _v3_raw(comp, 2, wz_scale=0.5)
    eps = R.score_run(raw, compiled=comp, scenario=sc, dt=DT, term_reason=["schedule_complete"] * 2,
                      n_steps_env=[comp.n_steps] * 2, travel_yaw=raw["travel_yaw"],
                      body_cmd=HG._body_cmd_series("term", comp, raw))
    holds = eps[0]["holds"]
    assert holds["turn_l_ceiling"]["body"]["n_steps"] == 600 - 200          # [4, 12) s
    assert holds["fwd"]["body"]["n_steps"] == 400 - 200                     # [4, 8) s
    assert holds["turn_l_ceiling"]["body"]["net_yaw_ratio"] == pytest.approx(0.5)
    assert holds["move_turn_r"]["body"]["net_yaw_ratio"] == pytest.approx(0.5)
    assert holds["move_turn_r"]["body"]["body_ratio"] == pytest.approx(1.0)
    assert holds["stand"]["body"]["heading_drift"] == 0.0
    assert "direction" in holds["fwd"] and "direction" not in holds["stand"]  # keyed on the body speed
    assert "facing" not in holds["fwd"] and "adapter" not in holds["fwd"]    # no heading_err / no adapter series
    agg = R.aggregate(eps)
    assert agg["body_by_hold"]["turn_l_ceiling"]["net_yaw_ratio"]["median"] == pytest.approx(0.5)
    assert "adapter_by_hold" not in agg
    txt = R.summary_text({"scenario_id": sc.id, "task": sc.task, "command_override_max_deviation": 0.0}, eps)
    assert "body yaw by hold" in txt and "turn_l_ceiling" in txt
    # a fall before 4 s into a hold leaves an empty window -> None, never a number
    early = R.score_run(raw, compiled=comp, scenario=sc, dt=DT, term_reason=["fall", "fall"],
                        n_steps_env=[75 + 400 + 150] * 2, travel_yaw=raw["travel_yaw"],
                        body_cmd=HG._body_cmd_series("term", comp, raw))
    assert early[0]["holds"]["fwd"]["body"] is None


def test_the_adapter_block_and_the_suite_body_keys(tmp_path):
    sc = S.load_manifest(_write(tmp_path, [_suite(SUITE_CATS)]))[0][0]
    parts = [S.compile_wave(sc, k, dt=DT, num_envs=2) for k in range(sc.waves)]
    comp = S.stack_waves(parts)
    raw = _v3_raw(comp, 10)
    raw["adapter_clipped"] = np.zeros((comp.n_steps, 10), dtype=bool)
    raw["adapter_clipped"][: comp.n_steps // 4] = True
    eps = R.score_run(raw, compiled=comp, scenario=sc, dt=DT, term_reason=["schedule_complete"] * 10,
                      n_steps_env=[comp.n_steps] * 10, travel_yaw=raw["travel_yaw"],
                      body_cmd=HG._body_cmd_series("term", comp, raw))
    meta = S.suite_run_meta(sc, dt=DT, num_envs=2, wave_seeds=[0] * 5, trained_categories=None)
    agg = R.aggregate(eps, suite=R.suite_aggregate_arg(meta))
    by_cat = agg["suite"]["by_category"]
    for c in by_cat.values():
        assert set(R.BODY_CATEGORY_METRIC_SOURCES) <= set(c)
    assert by_cat["turn"]["net_yaw_ratio"]["median"] == pytest.approx(1.0)
    assert by_cat["turn"]["by_hand"]["r"]["net_yaw_rate"]["median"] == pytest.approx(-0.05)
    assert 0.0 < by_cat["stand"]["adapter_clipped_frac"]["median"] < 1.0
    assert set(agg["adapter_by_hold"]) == set(comp.hold_labels)
    rp = _rp()
    cat = rp.summarize(agg)["cat"]
    ec = _ec()
    assert set(cat["turn"]) == set(ec.CAT_KEYS) | {"by_hand"}
    assert cat["turn"]["by_hand"]["l"]["net_yaw_ratio"] == pytest.approx(1.0)


# ============================================================ eval_categories
def _ec():
    return _load(EVAL_CATEGORIES, "eval_categories__body_eval")


def _rp():
    return _load(RUN_PHASES, "run_phases__body_eval")


def test_cat_keys_cross_pins():
    ec, rp = _ec(), _rp()
    assert ec.BODY_CAT_KEYS == M.BODY_YAW_KEYS + ("adapter_clipped_frac",)
    assert tuple(R.BODY_CATEGORY_METRIC_SOURCES) == ec.BODY_CAT_KEYS == rp.BODY_CAT_METRIC_KEYS
    assert set(ec.CAT_KEYS) == set(ec.CAT_KEYS_V2) | set(ec.BODY_CAT_KEYS)
    assert not set(ec.CAT_KEYS_V2) & set(ec.BODY_CAT_KEYS)
    tail = ec.MOTION_QUALITY_KEYS + ec.RIGIDITY_KEYS
    assert ec.CAT_KEYS[-len(tail):] == tail and ec.CAT_KEYS_V2[-len(tail):] == tail
    assert set(rp.CAT_METRIC_KEYS) | {"n", "n_env", "survival"} == set(ec.CAT_KEYS_V2)
    assert ec.parse_cat_id("cat.turn.by_hand.l.net_yaw_ratio") == ("turn", "l", "net_yaw_ratio")
    assert ec.metric_sign("cat.turn.yaw_ripple") == -1 and ec.metric_sign("cat.fwd.heading_drift@guard") == -1
    assert ec.metric_sign("cat.turn.net_yaw_ratio") == 1


def test_rule_v3_allows_the_body_keys_and_v2_expansion_is_unchanged():
    ec = _ec()
    v2 = ec.expand_rule(ec.default_rule_v2(), ec.CATEGORIES, ec.CATEGORIES, absolute_units=False)
    v3 = ec.expand_rule(ec.default_rule_v3(), ec.CATEGORIES, ec.CATEGORIES, absolute_units=False)
    assert v2["rule_version"] == 2 and v3["rule_version"] == 3
    assert {k: v for k, v in v2.items() if k != "rule_version"} == {k: v for k, v in v3.items() if k != "rule_version"}
    body = ec.default_rule_v3()
    body["per_category"]["blocks"]["T"]["members"]["turn"] = ["net_yaw_ratio"]
    r = ec.expand_rule(body, ec.CATEGORIES, ec.CATEGORIES, absolute_units=False)
    assert "cat.turn.net_yaw_ratio" in r["blocks"]["T"]["metrics"]
    v2bad = ec.default_rule_v2()
    v2bad["per_category"]["blocks"]["T"]["members"]["turn"] = ["net_yaw_ratio"]
    with pytest.raises(ValueError, match="keys"):
        ec.expand_rule(v2bad, ec.CATEGORIES, ec.CATEGORIES, absolute_units=False)
    with pytest.raises(ValueError, match="rule_version"):
        ec.expand_rule({"rule_version": 4, "per_category": {"blocks": {}}}, ec.CATEGORIES, ec.CATEGORIES,
                       absolute_units=False)


def test_trained_categories_of_a_body_env_are_its_world_twins():
    ec, rp = _ec(), _rp()
    for base in ("1a", "1b", "2a"):
        world = ec.trained_categories(rp.ph.phase_env(base))
        body_env = {**rp.ph.phase_env(f"{base}_b"), "KRABBY_OMNI_BODY_ENVELOPE": ENV_TXT, "KRABBY_OMNI_BODY_CAP": CAP_TXT}
        body = ec.trained_categories(body_env)
        assert (body.decisive, body.present, body.watch, body.hands, body.demand, body.source) == (
            world.decisive, world.present, world.watch, world.hands, world.demand, world.source), base
        assert body.encoding == "none" and body.facing_band_hi == 0.0
        assert body.notes[-1].startswith("turn: body frame") and ENV_TXT in body.notes[-1]
        assert "FACING band" in world.notes[-1]
        assert ec.TrainedCategories.from_dict(body.to_dict()) == body


# ============================================================ the driver (run_phases / run_candidates)
BODY_OVERLAY = {"KRABBY_OMNI_BODY_ENVELOPE": ENV_TXT, "KRABBY_OMNI_BODY_CAP": CAP_TXT}


def test_eval_carry_frame_keys_and_defaults():
    rp = _rp()
    bk = _load(OMNI / "config/crab_hex/body_knobs.py", "body_knobs__body_eval")
    assert set(bk.EVAL_SEMANTIC_KNOBS) <= set(rp.EVAL_CARRY)
    assert {bk.COMMAND_FRAME_VAR, bk.BODY_ENVELOPE_VAR, bk.BODY_CAP_VAR} <= set(rp.EVAL_CARRY)
    assert rp.EVAL_CARRY_DEFAULTS[bk.COMMAND_FRAME_VAR] == bk.KNOB_OFF_VALUES[bk.COMMAND_FRAME_VAR] == "world"
    assert rp.EVAL_CARRY_BODY_ONLY == (bk.BODY_ENVELOPE_VAR, bk.BODY_CAP_VAR)
    assert not set(rp.EVAL_CARRY_BODY_ONLY) & set(rp.EVAL_CARRY_DEFAULTS)
    res = _load(RESOLUTION, "resolution__body_eval")
    assert not set(res.required_carry_keys()) & set(rp.EVAL_CARRY_BODY_ONLY)


def test_a_world_eval_records_the_world_frame_and_carries_no_envelope():
    rp = _rp()
    for name in ("1c", "2a"):
        ev = rp.eval_env_for(rp.ph.PHASES[name], "A15+B", keep=rp.CANARY_KEYS, keep_source=rp.ph.FORMATION)
        assert ev["KRABBY_OMNI_COMMAND_FRAME"] == "world" and ev["KRABBY_OMNI_FACING_OBS_ENCODING"] == "raw"
        assert not set(rp.EVAL_CARRY_BODY_ONLY) & set(ev)
    with pytest.raises(ValueError, match="REFUSED in world mode"):
        rp.eval_env_for(rp.ph.PHASES["2a"], "A15+B", carry_overlay=BODY_OVERLAY)


def test_a_body_heads_eval_carries_none_the_frame_and_its_envelope():
    rp = _rp()
    for name in ("1a_b", "1b_b", "2a_b"):
        ev = rp.eval_env_for(rp.ph.PHASES[name], "A15+B", keep=rp.CANARY_KEYS, keep_source=rp.ph.FORMATION,
                             carry_overlay=BODY_OVERLAY)
        assert ev["KRABBY_OMNI_FACING_OBS_ENCODING"] == "none", name    # the preset's, never the raw default
        assert ev["KRABBY_OMNI_COMMAND_FRAME"] == "body"
        assert (ev["KRABBY_OMNI_BODY_ENVELOPE"], ev["KRABBY_OMNI_BODY_CAP"]) == (ENV_TXT, CAP_TXT)
        assert "KRABBY_OMNI_FACING_OBS_CLIP" not in ev
    # a body eval without its envelope cannot build: refused before a GPU is booked
    with pytest.raises(ValueError, match="REQUIRED"):
        rp.eval_env_for(rp.ph.PHASES["1a_b"], "A15+B")
    # an overlay that asks for a raw encoding in body mode is refused, not silently overridden
    with pytest.raises(ValueError, match="REFUSED in body mode"):
        rp.eval_env_for(rp.ph.PHASES["1a_b"], "A15+B",
                        carry_overlay={**BODY_OVERLAY, "KRABBY_OMNI_FACING_OBS_ENCODING": "raw"})


def test_a_world_base_switched_to_body_by_its_overlay_gets_none_not_the_raw_default(monkeypatch):
    rp = _rp()
    base = dict(rp.ph.phase_env("1a"))
    base.pop("KRABBY_OMNI_FACING_OBS_CLIP", None)
    monkeypatch.setattr(rp.ph, "phase_env", lambda name, plant=None: dict(base))
    ev = rp.eval_env_for(rp.ph.PHASES["1a"], "A15+B", carry_overlay={**BODY_OVERLAY, "KRABBY_OMNI_COMMAND_FRAME": "body"})
    assert ev["KRABBY_OMNI_FACING_OBS_ENCODING"] == "none"


def test_the_adapter_pairing_table():
    rp = _rp()
    assert rp.manifest_version(rp.OMNI_MANIFEST) == 2 and rp.manifest_version(rp.BODY_MANIFEST) == 3
    assert rp.command_adapter_for("world", 2) == "none" and rp.command_adapter_for("body", 2) == "world_rate"
    assert rp.command_adapter_for("world", 3) == "body_hold" and rp.command_adapter_for("body", 3) == "none"
    called = []
    turns = lambda: called.append(1) or True  # noqa: E731
    assert rp.eval_adapter_args({"KRABBY_OMNI_COMMAND_FRAME": "world"}, rp.OMNI_MANIFEST, turns) == []
    assert rp.eval_adapter_args({}, rp.BODY_MANIFEST, turns) == ["--command-adapter", "body_hold"]
    assert rp.eval_adapter_args({"KRABBY_OMNI_COMMAND_FRAME": "body"}, rp.BODY_MANIFEST, turns) == []
    assert not called                                               # turns are read only for world_rate
    assert rp.eval_adapter_args({"KRABBY_OMNI_COMMAND_FRAME": "body"}, rp.OMNI_MANIFEST, turns) == [
        "--command-adapter", "world_rate", "--adapter-turns", "on"]
    assert rp.eval_adapter_args({"KRABBY_OMNI_COMMAND_FRAME": "body"}, rp.OMNI_MANIFEST, lambda: False)[-1] == "off"
    # every pairing the driver can produce is one the harness accepts
    for frame in ("world", "body"):
        for version, rows in ((2, "world"), (3, "v3")):
            adapter = rp.command_adapter_for(frame, version)
            assert HG._route_check(rows, frame, adapter) in ("legacy", "term", "body_hold", "world_rate")


def test_stage_trains_turns():
    rp = _rp()
    assert rp.stage_trains_turns(rp.ph.PHASES["1a_b"]) is False
    assert rp.stage_trains_turns(rp.ph.PHASES["2a_b"], BODY_OVERLAY) is True
    assert rp.stage_trains_turns(rp.ph.PHASES["1a"], trained_categories="stand,fwd,move_turn") is True
    assert rp.stage_trains_turns(rp.ph.PHASES["1a"], trained_categories=["stand", "fwd"]) is False


def test_eval_rows_resolve_to_their_manifest():
    rp = _rp()
    assert rp.eval_row_target("slow") == (rp.OMNI_MANIFEST, "slow__omni")
    assert rp.eval_row_target("suite") == (rp.OMNI_MANIFEST, "suite__omni")
    assert rp.eval_row_target("suite_body") == (rp.BODY_MANIFEST, "suite__omni_body")
    assert rp.eval_row_target("field_hard__omni_body") == (rp.BODY_MANIFEST, "field_hard__omni_body")
    row = rp._manifest_row("turnlevels__omni_body")
    assert row["kind"] == "category_suite" and row["hold_s"]["turn_l_w_min"] == 12.0 and row["settle_hold_s"] == 2.0
    assert rp._is_suite_row("grid__omni_body") and rp._is_suite_row("suite__omni_body")
    assert not rp._is_suite_row("field_hard__omni_body") and not rp._is_suite_row("slow__omni")
    ends = rp._survival_end_steps("grid__omni_body", DT)
    assert set(ends.values()) == {775}
    assert rp._survival_end_steps("field_hard__omni_body", DT)["move_turn_l"] == round((1.5 + 72.0) / DT)
    for name in ("1a_b", "1b_b", "2a_b"):
        rows = rp.ph.PHASES[name].eval_scenarios
        assert rows == (("suite_body",) if name != "2a_b" else ("suite_body", "ring_obst_body")), name
        for r in rows:
            assert rp._manifest_row(rp.eval_row_target(r)[1])


def _capture_eval(rp, monkeypatch):
    seen = []

    def fake(c_, tag, sid, cmd_tail, env, out_root_, plant, timeout, **kw):
        seen.append((sid, list(cmd_tail), dict(env)))
        return None, None, {"status": "none", "rc": 1, "attempts": 1}

    monkeypatch.setattr(rp, "_run_eval", fake)
    c = types.SimpleNamespace(plant="A15+B", dir=Path("/tmp/campaign_body_eval_test"))
    return seen, c


def test_eval_morph_and_eval_obst_pass_the_pairing(monkeypatch):
    rp = _rp()
    seen, c = _capture_eval(rp, monkeypatch)
    # a body head on its own rows: no adapter flag
    rp.eval_morph(c, rp.ph.PHASES["2a_b"], "/a/m.pt", "suite_body", "tag", carry_overlay=BODY_OVERLAY)
    sid, tail, env = seen[-1]
    assert sid == "suite__omni_body" and tail[:4] == ["--manifest", str(rp.BODY_MANIFEST), "--scenario", sid]
    assert "--command-adapter" not in tail and env["KRABBY_OMNI_COMMAND_FRAME"] == "body"
    # a body head on the legacy world rows: world_rate, turns from what the stage trains
    rp.eval_morph(c, rp.ph.PHASES["2a_b"], "/a/m.pt", "slow", "tag", carry_overlay=BODY_OVERLAY)
    assert seen[-1][0] == "slow__omni" and seen[-1][1][-4:] == ["--command-adapter", "world_rate", "--adapter-turns", "on"]
    rp.eval_morph(c, rp.ph.PHASES["1a_b"], "/a/m.pt", "slow", "tag", carry_overlay=BODY_OVERLAY)
    assert seen[-1][1][-2:] == ["--adapter-turns", "off"]
    rp.eval_obst(c, rp.ph.PHASES["2a_b"], "/a/m.pt", "tag", carry_overlay=BODY_OVERLAY)
    assert seen[-1][0] == "ring__omni" and seen[-1][1][-4:] == ["--command-adapter", "world_rate", "--adapter-turns", "on"]
    # a world head on the body rows: body_hold; on the legacy rows: nothing (the unchanged command line)
    rp.eval_morph(c, rp.ph.PHASES["2a"], "/a/m.pt", "grid_body", "tag")
    assert seen[-1][0] == "grid__omni_body" and seen[-1][1][-2:] == ["--command-adapter", "body_hold"]
    rp.eval_morph(c, rp.ph.PHASES["2a"], "/a/m.pt", "slow", "tag")
    assert seen[-1][1] == ["--manifest", str(rp.OMNI_MANIFEST), "--scenario", "slow__omni", "--checkpoint", "/a/m.pt",
                           "--save-raw"]
    rp.eval_obst(c, rp.ph.PHASES["2a"], "/a/m.pt", "tag")
    assert seen[-1][1] == ["--manifest", str(rp.OMNI_MANIFEST), "--scenario", "ring__omni", "--checkpoint", "/a/m.pt"]
    # a body head on the grid has no valid pairing: refused before launch
    n = len(seen)
    with pytest.raises(ValueError, match="envelope grid"):
        rp.eval_morph(c, rp.ph.PHASES["2a_b"], "/a/m.pt", "grid_body", "tag", carry_overlay=BODY_OVERLAY)
    assert len(seen) == n


def test_run_candidates_hashes_the_body_manifest_and_resolves_body_rows():
    rc = _load(RUN_CANDIDATES, "run_candidates__body_eval")
    path = "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml"
    assert path in rc.EVAL_PATHSPECS and rc.pathspec_covers(path, rc.EVAL_PATHSPECS)
    assert not rc.pathspec_covers(path, rc.CODE_PATHSPECS)
    assert rc._sid_for("slow") == "slow__omni" and rc._sid_for("suite__omni") == "suite__omni"
    assert rc._sid_for("suite_body") == "suite__omni_body" and rc._sid_for("grid__omni_body") == "grid__omni_body"
    assert rc._eval_row_kind("suite_body") == "category_suite" and rc._eval_row_kind("grid_body") == "envelope_grid"
    # the suite-row coverage check reads the body manifest for a body row
    ec = _ec()
    rp = _rp()
    tc = ec.trained_categories({**rp.ph.phase_env("2a_b"), **BODY_OVERLAY})
    for row, refused in (("suite_body", False), ("turnlevels_body", True)):
        reasons: list = []
        rc._check_suite_rows_cover_trained({"eval": [row]}, tc, "stage s", reasons)
        assert bool(reasons) is refused, (row, reasons)


def test_world_rate_cap_is_the_heads_own_and_refuses_a_manifest_mismatch():
    """Step-2 review (2026-10-15): world_rate saturates at the HEAD's cfg.ang_vel_max, and a manifest that says
    otherwise is an INVALID eval (not a silent re-cap)."""
    import types as _t
    term = _t.SimpleNamespace(cfg=_t.SimpleNamespace(ang_vel_max=0.3))
    assert HG._world_rate_cap(term, _t.SimpleNamespace(ang_vel_max=0.3)) == 0.3
    assert HG._world_rate_cap(term, _t.SimpleNamespace(ang_vel_max=None)) == 0.3
    with pytest.raises(SystemExit, match="relay cap mismatch"):
        HG._world_rate_cap(term, _t.SimpleNamespace(ang_vel_max=0.25))
