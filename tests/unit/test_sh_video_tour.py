"""Stand-height campaign (``2026-10-13_stand_height``): the step-8 video's TOUR extension on CPU (no Isaac / Kit / GPU;
VIDEO ONLY, read by no gate).

Pinned here:

* the ``tour`` / ``tour_smoke`` plans (``sh_video_schedule.py``): holds, labels, heights, facings, the suite's speeds,
  the stand lengths (``|dh| / 0.08 + 3 + 3`` s), the restore allowances of the command holds after a stand
  (``base + |h - 1.064| / 0.08 + 1`` s), the table first stand (12 s >= the rods' rise + ramp + 3 + 3 s on the
  0.635 m knee_folded table), the total length; both pass ``sh_video_core.tour_row_reasons`` (the launcher's rules
  adapted for turns and a table first stand -- re-derived to equal ``sh_eval_common.height_row_reasons`` on every
  one-facing walk-first row) and the harness validator, and are written under the ``stand_height_video`` block that the
  launcher's loader cannot read;
* ``--stand-rule``: the schedule rule keys exactly the ``"stand"`` holds; a stand after a facing change is routed to
  the stand net under ``schedule`` and to S1 under ``slot`` (the eval's latched slot); ``height_step`` stays BITWISE the
  launcher's ``__call__`` under both rules (the controller fed the routed slot), and a restore engages on a
  schedule-keyed stand exit;
* ``--terrain``: ``flat`` = the reference env unchanged; ``field_medium`` = + the env block READ from
  ``scenarios_p2.yaml`` (field_turn__omni == field_suite__omni), written before Kit (source-level); where on the field
  (the review fix): the tile map, the tour's commanded path needing 2 field rows ahead of the spawn so the spawn row is
  pinned to 0..7 (from rows 8 / 9 the path reaches the flat border), the placement refusals, and the recorder's
  wiring (pin on the field only, the realised spawn re-checked, the tile captioned and recorded per hold);
* ``--table-start``: parsing / refusals (first hold not a stand, height < top + 0.31, unknown top / pose), the table
  controller reset (limiter at clamp(h_meas), no walk reset), the fall read (falls recorded, not terminated: tilt any
  time, hip contact only once risen >= 0.10 m) and its live tracker;
* ``sh_record_video.sh``: the tour / terrain / table options reach the recorder (dry run), the refusals, the timeouts.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
VIDEOS = CAMP / "videos"
GEN_PATH = SCRIPTS / "sh_video_schedule.py"
REC_PATH = SCRIPTS / "sh_record_video.py"
SH_PATH = SCRIPTS / "sh_record_video.sh"
LAUNCHER = SCRIPTS / "sh_eval_crab_hex_gait.py"
P2_MANIFEST = TASK / "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml"
T17_HEAD = REPO / "parkour/outputs/omni_plan/stand_height/runs/SH_T17/model_5000.pt"
T17_STAMPS = REPO / "parkour/outputs/omni_plan/phase2/lr/SH_T17/hraise__omni/seed001"
#: SH_T17's recipe replayed beside the same S1 s18 at the Task 4 pin (R18's X stand replay, CHARTER 2026-10-19/20).
#: SH_T17's stamps predate the 2026-10-16 eval commit 9072aca1 (harness sha), and since the 2026-10-26 doc note in
#: sh_core.py (user decision: accept the hash changes) NO recorded stand stamp carries the current sh_core sha either --
#: the recorder refuses them all (correctly) until a stand head is re-stamped by the current code.
X_HEAD = REPO / "parkour/outputs/omni_plan/stand_height/runs/SH_T4R18_sh_ref_s17/model_5000.pt"
X_STAMPS = REPO / "parkour/outputs/omni_plan/phase2/lr/SH_T4R18_sh_ref_s17/hraise__omni/seed001"
HARNESS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
#: the other files whose recorded sha the recorder's REFERENCE_CHECKS compare with the file on disk
#: (sh_record_video.py: SH_LAUNCHER = LAUNCHER below, EC_PATH, P2R_LAUNCHER, sh_core.provenance's five shas) and its
#: default walker (DEFAULT_CKPT = S1 s18: the stub test passes no --checkpoint)
SH_EVAL_COMMON = SCRIPTS / "sh_eval_common.py"
P2R_LAUNCHER = TASK / "experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py"
S1_DEFAULT = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
CORE_SHA_KEYS = ("sh_core_sha256", "sh_kin_sha256", "sn_common_sha256", "restance_sha256", "facts_json_sha256")


def _sha_or_none(path: Path):
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _reference_file_sha_mismatches(meta: dict, stand_head: Path, CORE, EC) -> list:
    """The names of the recorder's file-sha REFERENCE_CHECKS (sh_record_video.py) this stamp FAILS today: the harness,
    the stand-height launcher, sh_eval_common, the P2R launcher, sh_core.provenance's shas (sh_core / sh_kin / sn_common
    / the R module / facts.json), the stand checkpoint (--stand-ckpt = the head) and the default S1 walker."""
    pm = meta.get("p2r_mode") or {}
    shb = pm.get(EC.BLOCK_KEY) or {}
    prov = CORE.provenance(0.25)
    checks = {
        "harness_sha256": (pm.get("harness_sha256"), _sha_or_none(HARNESS)),
        "sh_launcher_sha256": (shb.get("wrapper_sha256"), _sha_or_none(LAUNCHER)),
        "sh_eval_common_sha256": (shb.get("common_sha256"), _sha_or_none(SH_EVAL_COMMON)),
        "p2r_launcher_sha256": (shb.get("p2r_launcher_sha256"), _sha_or_none(P2R_LAUNCHER)),
        **{k: ((shb.get("core") or {}).get(k), prov[k]) for k in CORE_SHA_KEYS},
        "stand_ckpt_sha256": ((pm.get("specialist") or {}).get("sha256"), _sha_or_none(stand_head)),
        "s1_ckpt_sha256": (meta.get("checkpoint_sha256"), _sha_or_none(S1_DEFAULT)),
    }
    return [k for k, (recorded, live) in checks.items() if recorded is None or recorded != live]


def _current_reference(CORE, EC):
    """((stand head, newest complete hraise es1 stamp), notes) for the first S1-paired SH head whose stamp passes EVERY
    file-sha reference check the recorder makes today (it refuses a stamp written by other code), X replay first,
    SH_T17 second; (None, notes) if none does -- the notes name each candidate's failing checks."""
    import json  # noqa: PLC0415

    notes = []
    for head, root in ((X_HEAD, X_STAMPS), (T17_HEAD, T17_STAMPS)):
        name = root.parents[1].name
        stamps = sorted(d for d in root.iterdir() if (d / "p2_complete.json").is_file()) if root.is_dir() else []
        if not stamps or not head.is_file():
            notes.append(f"{name}: head or stamp not on this machine")
            continue
        meta = json.loads((stamps[-1] / "run_meta.json").read_text())
        bad = _reference_file_sha_mismatches(meta, head, CORE, EC)
        if not bad:
            return (head, stamps[-1]), notes
        notes.append(f"{name} {stamps[-1].name}: {', '.join(bad)} differ")
    return None, notes
HALF_PI = 1.5707963
DS = 10

TOUR_LABELS = ["stand", "fwd", "stand", "stand", "fwd", "lat_l", "back", "stand", "lat_r", "diag", "stand",
               "turn_l90", "turn_r90", "stand", "move_turn_l", "move_turn_r", "stand", "fwd"]
TOUR_HEIGHTS = [1.064, None, 0.719, 1.346, None, None, None, 0.719, None, None, 1.346, None, None, 0.719, None, None,
                1.346, None]
#: (speed, dir, face) per command label (the suite's speeds; the tour's faces)
TOUR_CMD = {"fwd": (0.25, 0.0, 0.0), "lat_l": (0.22, HALF_PI, 0.0), "back": (0.22, 3.1415927, 0.0),
            "lat_r": (0.22, -HALF_PI, 0.0), "diag": (0.25, 0.7853982, 0.0), "turn_l90": (0.0, 0.0, HALF_PI),
            "turn_r90": (0.0, 0.0, 0.0), "move_turn_l": (0.22, 0.0, HALF_PI), "move_turn_r": (0.22, 0.0, 0.0)}
TOUR_BASE = [None, 20.0, None, None, 10.0, 10.0, 10.0, None, 10.0, 10.0, None, 10.0, 10.0, None, 10.0, 10.0, None, 8.0]
TOUR_LENGTHS = [12.0, 21.0, 10.4, 13.9, 14.6, 10.0, 10.0, 10.4, 15.4, 10.0, 9.6, 14.6, 10.0, 10.4, 15.4, 10.0, 9.6,
                12.6]


def _pyc_snapshot() -> dict:
    out = {}
    for d in (SCRIPTS, TASK / "experiments/2026-10-11_stand_network/scripts",
              TASK / "experiments/2026-10-07_omni_phase2/scripts"):
        for p in sorted(d.glob("__pycache__/*.pyc")):
            out[str(p)] = p.stat().st_mtime_ns
    return out


_PYC_BEFORE = _pyc_snapshot()


def _load(path: Path, name: str):
    """By path, never writing ``__pycache__`` next to the campaign / phase-2 / predecessor scripts."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def GEN():
    return _load(GEN_PATH, "_t_shvt_gen")


@pytest.fixture(scope="module")
def VC(GEN):
    return GEN.VC


@pytest.fixture(scope="module")
def CORE(VC):
    return VC.CORE


@pytest.fixture(scope="module")
def EC(VC):
    return VC.EC


@pytest.fixture(scope="module")
def S(GEN):
    return GEN.schedule_module()


@pytest.fixture(scope="module")
def HDR(GEN):
    return GEN.reference_header()


@pytest.fixture(scope="module")
def TV():
    """The base video test module (its launcher-mode double, network / estimator doubles, fake bin), reused."""
    return _load(Path(__file__).resolve().parent / "test_sh_video.py", "_t_shvt_base")


@pytest.fixture(scope="module")
def tour(GEN, HDR):
    return GEN.generate("tour", hdr=HDR)


@pytest.fixture(scope="module")
def tour_smoke(GEN, HDR):
    return GEN.generate("tour_smoke", hdr=HDR)


def _ceil_ds(s: float) -> float:
    return math.ceil(s * DS - 1e-9) / DS


# ===================================================================================== the tour plans
def test_tour_plan_holds_heights_facings_and_restore_allowances(tour, CORE, VC):
    holds, heights = tour["holds"], tour["heights"]
    assert [h.label for h in holds] == TOUR_LABELS and heights == TOUR_HEIGHTS
    assert tour["row_id"] == "sh_tour__omni"
    kinds = VC.hold_kinds(holds)
    assert [i for i, k in enumerate(kinds) if k == "turn"] == [11, 12]
    assert [i for i, k in enumerate(kinds) if k == "stand"] == [i for i, h in enumerate(heights) if h is not None]
    face = 0.0
    level, prev = None, None
    for i, (h, hh) in enumerate(zip(holds, heights)):
        sp, di, fa = h.as_world()
        if hh is not None:                                                       # a stand: speed 0 at the face held
            assert (sp, di, fa) == (0.0, 0.0, face) and h.label == "stand"
            want = 12.0 if i == 0 else _ceil_ds(abs(hh - level) / 0.08 + 3.0 + 3.0)
            assert h.hold_s == pytest.approx(want, abs=1e-9), (i, h.hold_s, want)
            level, prev = hh, hh
        else:
            assert (sp, di, fa) == TOUR_CMD[h.label], h.label
            extra = 0.0 if prev is None else abs(prev - 1.064) / 0.08 + 1.0           # the restore allowance
            assert h.hold_s == pytest.approx(_ceil_ds(TOUR_BASE[i] + extra), abs=1e-9), (i, h.label)
            level, prev, face = 1.064, None, fa
    assert [h.hold_s for h in holds] == pytest.approx(TOUR_LENGTHS, abs=1e-9)
    # (b) the first walk: 20 s of motion after the 1.064 get-up; (c) two consecutive stands h_min -> h_max, one facing
    assert holds[1].hold_s == 21.0 and heights[2:4] == [CORE.H_MIN, CORE.H_MAX] and holds[2].face == holds[3].face
    # every 2-3 commands a stand at h_min or h_max; the tour ends on a walk
    stand_idx = [i for i, h in enumerate(heights) if h is not None]
    gaps = [b - a - 1 for a, b in zip(stand_idx, stand_idx[1:])]
    assert gaps == [1, 0, 3, 2, 2, 2] and all(heights[i] in (CORE.H_MIN, CORE.H_MAX) for i in stand_idx[1:])
    sm = tour["summary"]
    assert sm["video_s"] == pytest.approx(221.4) and sm["prologue_s"] == 1.5 and sm["n_turns"] == 2
    assert sm["restores"] == 5 and sm["commands"] == [lab for lab, hh in zip(TOUR_LABELS, TOUR_HEIGHTS) if hh is None]
    assert sm["first_stand"]["table"]["top"] == 0.635 and sm["first_stand"]["table"]["pose"] == "knee_folded"


def test_tour_smoke_plan(tour_smoke, VC):
    holds, heights = tour_smoke["holds"], tour_smoke["heights"]
    assert [h.label for h in holds] == ["stand", "fwd", "stand", "turn_l90", "stand", "fwd"]
    assert heights == [1.064, None, 0.719, None, 1.346, None]
    assert [h.hold_s for h in holds] == pytest.approx([12.0, 7.0, 10.4, 11.4, 9.6, 12.6], abs=1e-9)
    assert holds[4].as_world() == (0.0, 0.0, HALF_PI)                              # a stand after the facing change
    assert holds[5].as_world() == (0.25, HALF_PI, HALF_PI)                         # body-forward at face +90
    assert VC.hold_kinds(holds) == ["stand", "walk", "stand", "turn", "stand", "walk"]
    assert tour_smoke["summary"]["video_s"] == pytest.approx(64.5)


def test_tour_rows_pass_the_adapted_rules_the_validator_and_the_table_need(GEN, VC, EC, S, HDR, tour, tour_smoke):
    for g in (tour, tour_smoke):
        hd = GEN.hold_dicts(g["holds"])
        fs = g["summary"]["first_stand"]
        assert VC.tour_row_reasons(hd, g["heights"], first_level=fs["level"], first_need_s=fs["need_s"]) == []
        assert GEN.tour_length_reasons(g["holds"], g["heights"]) == []
        S.validate_scenario(GEN._scenario(S, HDR, g["holds"], g["row_id"]))
        # the launcher's own rules refuse a tour row (a stand first, turns): it is never a launcher row
        why = EC.height_row_reasons(hd, g["heights"])
        assert any("must be a walk" in w for w in why)
        assert fs["level"] == pytest.approx(0.79875) and fs["sizing"]["reach_s"] > fs["sizing"]["ramp_s"]
        assert fs["tour_need_s"] == pytest.approx(fs["sizing"]["reach_s"] + 6.0) and fs["tour_need_s"] <= 12.0
        assert fs["need_s"] == pytest.approx(fs["sizing"]["reach_s"] + 3.0)
    # the same tour against other tables: rod_min's slower rise does not fit the 12-s get-up; 0.762 + 0.31 > 1.064
    with pytest.raises(ValueError, match="table stand"):
        GEN.generate("tour", hdr=HDR, table_start="0.635:rod_min")
    with pytest.raises(ValueError, match="0.31"):
        GEN.generate("tour", hdr=HDR, table_start="0.762")
    with pytest.raises(ValueError, match="must be a stand"):
        GEN.generate("raise", hdr=HDR, table_start="0.635")
    assert GEN.generate("tour", hdr=HDR, table_start="0.6985:knee_folded")["table_start"]["top"] == 0.6985


def test_tour_speeds_are_the_suites(GEN):
    assert GEN.suite_speed_reasons() == []
    suite = {}
    for r in yaml.safe_load(P2_MANIFEST.read_text())["scenarios"]:
        if r["id"] == "field_suite__omni":
            for c in r["categories"]:
                for h in c["holds"]:
                    suite[h["label"]] = (h["speed"], h["dir"], h["face"])
    for lab in ("fwd", "lat_l", "lat_r", "back", "diag"):
        assert GEN.TOUR_CMDS[lab] == tuple(float(x) for x in suite[lab]), lab
    assert GEN.TOUR_CMDS["diag"][1] == pytest.approx(math.radians(45.0), abs=1e-7)
    assert GEN.TOUR_CMDS["move_turn_l"][0] == suite["move_turn"][0] == 0.22


# ===================================================================================== the adapted rules
def _hd(label, speed, dirn, face, hold_s):
    return {"label": label, "speed": speed, "dir": dirn, "face": face, "hold_s": hold_s}


def test_tour_rules_reduce_to_the_launcher_rules_on_launcher_rows(GEN, VC, EC, HDR):
    rows = [GEN.generate("raise", hdr=HDR), GEN.generate("smoke", hdr=HDR)]
    rows += [GEN.generate("random", s, 150.0, hdr=HDR) for s in (0, 7, 42)]
    for g in rows:
        hd, hh = GEN.hold_dicts(g["holds"]), list(g["heights"])
        assert VC.tour_row_reasons(hd, hh) == [] and EC.height_row_reasons(hd, hh) == []
        mutants = []
        i_st = next(i for i, h in enumerate(hh) if h is not None)
        m = [dict(h) for h in hd]
        m[i_st]["hold_s"] = 1.0
        mutants.append((m, hh))                                                      # a short stand
        i_wk = next(i for i in range(1, len(hh)) if hh[i] is None and hh[i - 1] is not None)
        m = [dict(h) for h in hd]
        m[i_wk]["hold_s"] = 0.5
        mutants.append((m, hh))                                                      # a short walk after a stand
        mutants.append((hd, [None if h is None else 1.40 for h in hh]))              # out of range
        mutants.append((hd[:-1], hh[:-1]))                                           # ends on a stand
        m = [dict(h) for h in hd]
        m[0]["label"] = "stand"
        mutants.append((m, hh))                                                      # a walk labelled stand
        mutants.append((hd, [1.0] + hh[1:]))                                         # a walk with a height
        for mh, mhh in mutants:
            assert VC.tour_row_reasons(mh, mhh) and EC.height_row_reasons(mh, mhh)


def test_tour_rules_turn_and_label_keying(VC):
    ok = [_hd("fwd", 0.25, 0.0, 0.0, 8.0), _hd("stand", 0.0, 0.0, 0.0, 8.0), _hd("turn_l90", 0.0, 0.0, HALF_PI, 12.0),
          _hd("stand", 0.0, 0.0, HALF_PI, 8.0), _hd("fwd", 0.25, HALF_PI, HALF_PI, 8.0)]
    hh = [None, 1.2, None, 1.0, None]
    assert VC.tour_row_reasons(ok, hh) == []
    assert VC.hold_kinds(ok) == ["walk", "stand", "turn", "stand", "walk"]
    bad = [dict(h) for h in ok]
    bad[2]["label"] = "stand"                                                        # a turn labelled stand
    assert any("labelled 'stand'" in w for w in VC.tour_row_reasons(bad, hh))
    bad = [dict(h) for h in ok]
    bad[3]["label"] = "pause"                                                        # a same-face speed-0 hold
    assert any("must be labelled 'stand'" in w for w in VC.tour_row_reasons(bad, hh))
    assert any("carries a height" in w for w in VC.tour_row_reasons(ok, [None, 1.2, 1.1, 1.0, None]))
    short = [dict(h) for h in ok]
    short[2]["hold_s"] = 4.0                                                         # a turn right after a 1.2 stand
    assert any("restore ramp" in w for w in VC.tour_row_reasons(short, hh))
    # a stand first: allowed, entered at the start's level, and held to the caller's need
    first = [_hd("stand", 0.0, 0.0, 0.0, 12.0)] + ok[1:]
    assert VC.tour_row_reasons(first, [1.064, 1.2, None, 1.0, None], first_level=0.79875, first_need_s=11.0) == []
    assert any("start's need" in w for w in VC.tour_row_reasons(first, [1.064, 1.2, None, 1.0, None],
                                                                   first_level=0.79875, first_need_s=12.5))
    first[0]["hold_s"] = 5.0
    assert any("hold_s 5.0" in w for w in VC.tour_row_reasons(first, [1.064, 1.2, None, 1.0, None],
                                                                first_level=0.79875))


def test_tour_cli_writes_the_video_block_the_launcher_cannot_read(GEN, VC, EC, S, CORE, tmp_path):
    assert GEN.main(["--plan", "tour", "--table-start", "0.635:knee_folded", "--out-dir", str(tmp_path)]) == 0
    yml = tmp_path / "scenarios_sh_video_tour.yaml"
    js = yaml.safe_load((tmp_path / "sh_video_schedule_tour.json").read_text())
    assert js["table_start"]["cell"] == "table_0p635__knee_folded" and js["summary"]["video_s"] == pytest.approx(221.4)
    data = yaml.safe_load(yml.read_text())
    assert VC.VIDEO_BLOCK in data and EC.MANIFEST_BLOCK not in data
    assert data[VC.VIDEO_BLOCK]["rows"]["sh_tour__omni"]["rules"] == "tour"
    assert "--table-start 0.635:knee_folded" in yml.read_text().splitlines()[1]
    with pytest.raises(ValueError):
        EC.height_row(yml, "sh_tour__omni")                                          # the launcher's loader refuses it
    row = VC.load_height_row(yml, "sh_tour__omni", first_level=0.79875)
    assert row["hold_heights"] == TOUR_HEIGHTS and row["labels"] == TOUR_LABELS and row["rules"] == "tour"
    assert VC.video_row_names(yml) == ["sh_tour__omni"]
    sc = {s.id: s for s in S.load_manifest(yml)[0]}["sh_tour__omni"]
    assert sc.spawn_yaw_rad() == 0.0 and not sc.env_vars and sc.terrain_label == "flat"
    comp = S.compile_schedule(sc, dt=0.02, num_envs=1)
    tab = EC.height_table(comp.segment_id, row["hold_heights"])
    seg = np.asarray(comp.segment_id)[:, 0]
    for i, hh in enumerate(row["hold_heights"]):
        assert np.all(tab[seg == i, 0] == (CORE.H_SETTLED if hh is None else hh)), i
    assert np.all(tab[seg < 0, 0] == 1.064) and int(comp.n_steps) == round(221.4 / 0.02)
    flags = VC.schedule_stand_flags(sc.schedule, sc.spawn_yaw_rad())
    assert flags == [h is not None for h in row["hold_heights"]]
    mask = VC.step_stand_mask(seg, flags)
    assert mask[seg < 0].all()                                                       # the prologue holds the table stand
    # a launcher row still goes through sh_eval_common.height_row (unchanged)
    assert GEN.main(["--plan", "smoke", "--out-dir", str(tmp_path)]) == 0
    sm = tmp_path / "scenarios_sh_video_smoke.yaml"
    assert VC.load_height_row(sm, "sh_smoke__omni") == EC.height_row(sm, "sh_smoke__omni")
    with pytest.raises(ValueError, match="neither"):
        VC.load_height_row(sm, "sh_tour__omni")
    # a tour row with an env: block is refused (the recorder's --terrain sets the terrain)
    data["scenarios"][0]["env"] = {"KRABBY_FLAT_TERRAIN_GEOM": "field"}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="env"):
        VC.video_height_row(bad, "sh_tour__omni")
    assert GEN.main(["--plan", "tour", "--table-start", "0.762", "--out-dir", str(tmp_path / "x"), "--dry-run"]) == 2


# ===================================================================================== the stand rule
def test_stand_rule_keying_on_the_tour(VC, tour):
    holds = tour["holds"]
    flags = VC.schedule_stand_flags(holds)
    assert flags == [h.label == "stand" for h in holds]                              # the schedule rule = the labels
    ev = VC.eval_slot_stand_flags(holds)
    stands = [i for i, f in enumerate(flags) if f]
    assert [i for i in stands if not ev[i]] == [13, 16]                              # after the turns: latched TURN
    assert all(ev[i] for i in stands if i < 11)
    assert set(VC.STAND_RULES) == {"schedule", "slot"} and VC.DEFAULT_STAND_RULE == "schedule"
    es = torch.tensor([False])
    assert VC.routed_slot("schedule", schedule_stand=True, eval_slot=es).tolist() == [True]
    assert VC.routed_slot("slot", schedule_stand=True, eval_slot=es) is es           # the pre-tour path, untouched
    assert VC.routed_slot("schedule", schedule_stand=np.bool_(False), eval_slot=torch.tensor([True])).tolist() == [False]
    with pytest.raises(ValueError):
        VC.routed_slot("hover", schedule_stand=True, eval_slot=es)


def _turn_schedule():
    """walk 20 / stand 1.2 (face 0) 150 / turn_l90 200 / stand 0.9 at face +90 (a stand after a facing change) 200 /
    walk 160 (steps; one env)."""
    holds = [_hd("fwd", 0.25, 0.0, 0.0, 0.4), _hd("stand", 0.0, 0.0, 0.0, 3.0), _hd("turn_l90", 0.0, 0.0, HALF_PI, 4.0),
             _hd("stand", 0.0, 0.0, HALF_PI, 4.0), _hd("fwd", 0.25, HALF_PI, HALF_PI, 3.2)]
    heights = [None, 1.2, None, 0.9, None]
    seg = np.array([0] * 20 + [1] * 150 + [2] * 200 + [3] * 200 + [4] * 160)
    return holds, heights, seg


@pytest.mark.parametrize("deadband", [0.0, 0.25])
def test_height_step_is_bitwise_the_launcher_under_both_rules(VC, EC, TV, deadband):
    holds, heights, seg = _turn_schedule()
    flags = VC.schedule_stand_flags(holds)
    ev = VC.eval_slot_stand_flags(holds)
    assert flags == [False, True, False, True, False] and ev == [False, True, False, False, False]
    sched = VC.step_stand_mask(seg, flags)
    latched = VC.step_stand_mask(seg, ev)
    T, n = seg.size, 1
    h_np = EC.height_table(seg[:, None], heights)
    h_dev = torch.as_tensor(h_np, dtype=torch.float64)
    obs_all = TV._obs_seq(T, n, seed=3)
    est = TV._Est(TV.P1 - TV.P0, 0.01)
    out = {}
    for rule in ("schedule", "slot"):
        SHL = _load(LAUNCHER, f"_t_shvt_launcher_{rule}_{int(deadband * 100)}")
        slot_box = [None]
        ep_box = torch.zeros(n, dtype=torch.long)
        m = TV._launcher_mode(SHL, deadband, h_np, n, est, slot_box, ep_box)
        ctl = VC.new_controller(n, "cpu", dt=0.02, h_cmd0=h_dev[0])
        eff = np.zeros(T, dtype=bool)
        engaged = np.zeros(T, dtype=bool)
        with torch.inference_mode():
            for t in range(T):
                ep_box.fill_(t)
                slot = VC.routed_slot(rule, schedule_stand=bool(sched[t]), eval_slot=torch.tensor([bool(latched[t])]))
                slot_box[0] = slot                                                   # the launcher fed the same slot
                obs = obs_all[t].clone()
                m(obs.clone())
                want = m._exec
                m._ret = m._exec = None
                st = VC.height_step(ctl, obs, slot=slot, reset_rows=torch.tensor([t == 0]), h_cmd=h_dev[t],
                                    s1_policy=TV._s1, stand_policy=TV._stand, stand_estimator=est, p0=TV.P0, p1=TV.P1,
                                    num_prop=TV.NP, mask=1, deadband=deadband)
                assert torch.equal(st.actions, want), (rule, t)
                assert torch.equal(st.hs.h_lim, m.ctl.h_lim)
                eff[t] = bool(st.hs.stand_eff[0])
                engaged[t] = bool(st.hs.engaged[0])
        assert ctl.counts() == m.ctl.counts()
        out[rule] = (eff, engaged, ctl.counts())
    post_turn = seg == 3
    # schedule: the stand net acts on the whole stand after the turn, and its exit into the walk engages a restore
    assert out["schedule"][0][post_turn].all()
    assert out["schedule"][1][seg == 4].any() and out["schedule"][2]["engaged"] == 2
    # slot: the latched TURN keeps S1 on it (the restore from 1.2 completed inside the turn hold)
    assert not out["slot"][0][post_turn].any() and out["slot"][2]["engaged"] == 1
    # both: the first stand is the stand net's, its exit restores during the turn
    for rule in ("schedule", "slot"):
        assert out[rule][0][seg == 1].all() and out[rule][1][seg == 2].any()


# ===================================================================================== the terrain
def test_terrain_env_flat_is_the_reference_and_field_medium_is_read_from_the_p2_rows(VC):
    ref = {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_OMNI_STAND_OBS": "0", "KRABBY_PLANT": "A15+B"}
    env, rec = VC.terrain_env(ref, "flat")
    assert env == ref and env is not ref and rec["deviation"] == {} and rec["env_block"] is None
    env, rec = VC.terrain_env(ref, "field_medium")
    want = {"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0",
            "KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70", "KRABBY_FIELD_PAD_R": "1.7",
            "KRABBY_FIELD_BLOCKS": "14:28", "KRABBY_FIELD_KINDS": "0.4:0.4:0.2", "KRABBY_FIELD_EDGES": "wall"}
    assert rec["env_block"] == want and env == {**ref, **want}
    assert rec["deviation"]["KRABBY_FLAT_TERRAIN_GEOM"] == {"reference": "ring", "video": "field"}
    assert "KRABBY_PLANT" not in rec["deviation"] and len(rec["deviation"]) == 8
    raw = {r["id"]: r for r in yaml.safe_load(P2_MANIFEST.read_text())["scenarios"]}
    for rid in VC.FIELD_MEDIUM_ROWS:                                                  # read, not retyped
        assert {k: str(v) for k, v in raw[rid]["env"].items()} == want
    assert rec["source"]["rows"] == ["field_turn__omni", "field_suite__omni"]
    assert VC.setup_caption_text(rec, "schedule").startswith("TERRAIN field_medium (field DIFF 0.20:0.70, wall edges)")
    with pytest.raises(ValueError):
        VC.terrain_env(ref, "field_hard")


def test_field_env_on_the_t17_stamp_passes_the_launcher_refusals(VC):
    stamps = sorted(d for d in T17_STAMPS.iterdir() if (d / "run_meta.json").is_file()) if T17_STAMPS.is_dir() else []
    if not stamps:
        pytest.skip("the SH_T17 hraise es1 stamp is not on this machine")
    import json  # noqa: PLC0415

    rm = json.loads((stamps[-1] / "run_meta.json").read_text())
    ref = {k: str(v) for k, v in rm["env_vars"].items()}
    env, rec = VC.terrain_env(ref, "field_medium")
    assert VC.SN.env_reasons(env) == [] and VC.SN.env_reasons(ref) == []
    assert set(env) - set(ref) == {"KRABBY_FLAT_TERRAIN_MODE", "KRABBY_FLAT_TERRAIN_FLAT_FRAC",
                                   "KRABBY_FLAT_TERRAIN_DIFF", "KRABBY_FIELD_PAD_R", "KRABBY_FIELD_BLOCKS",
                                   "KRABBY_FIELD_KINDS", "KRABBY_FIELD_EDGES"}
    assert rm["p2r_mode"]["specialist"]["sha256"] == VC.EC.sha256_file(T17_HEAD)      # the head the CLI names
    assert VC.CORE.recorded_deadband(rm["p2r_mode"]["stand_height"]) == 0.25


def _line_of(tree, pred) -> list[int]:
    return sorted(n.lineno for n in ast.walk(tree) if pred(n))


def test_recorder_writes_the_terrain_env_before_kit_and_keeps_every_other_reference_check(tmp_path):
    import py_compile  # noqa: PLC0415

    py_compile.compile(str(REC_PATH), cfile=str(tmp_path / "rec.pyc"), doraise=True)
    src = REC_PATH.read_text()
    tree = ast.parse(src)
    src_of = lambda n: ast.get_source_segment(src, n) or ""  # noqa: E731
    kit = _line_of(tree, lambda n: isinstance(n, ast.Call) and src_of(n.func) == "AppLauncher")
    upd = _line_of(tree, lambda n: isinstance(n, ast.Call) and src_of(n) == "os.environ.update(RUN_ENV)")
    terr = _line_of(tree, lambda n: isinstance(n, ast.Assign)
                    and src_of(n) == "RUN_ENV, TERRAIN = VC.terrain_env(WANT_ENV, args_cli.terrain)")
    assert len(kit) == 1 and len(upd) == 1 and len(terr) == 1 and terr[0] < upd[0] < kit[0]
    # nothing writes the environment after Kit starts
    writes = _line_of(tree, lambda n: (isinstance(n, ast.Call) and re.match(r"os\.(environ\.(update|setdefault|pop)"
                                                                            r"|putenv|unsetenv)", src_of(n.func)))
                      or (isinstance(n, (ast.Assign, ast.Delete)) and "os.environ[" in src_of(n)))
    assert writes and max(writes) < kit[0], writes
    assert "if krabby_now != RUN_ENV:" in src                                           # re-checked after the import
    # every other reference check is still in the table the refusal reads
    ref = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and src_of(n.targets[0]) == "REFERENCE_CHECKS")
    keys = {k.value for k in ref.value.keys if isinstance(k, ast.Constant)}
    assert {"reference_variant", "control", "restore", "stand_mask", "residual_deadband", "stand_ckpt_sha256",
            "s1_ckpt_sha256", "harness_sha256", "sh_launcher_sha256", "sh_eval_common_sha256"} <= keys
    assert not any("terrain" in k or "env" in k for k in keys)                       # the deviation is recorded, not
    for needle in ('"terrain": dict(TERRAIN', '"reference_env_vars": WANT_ENV', '"env_vars": RUN_ENV',  # waived
                   '"table_start": table_summary', '"stand_rule": {"rule": RULE',
                   'parser.add_argument("--stand-rule", choices=VC.STAND_RULES, default=VC.DEFAULT_STAND_RULE',
                   'parser.add_argument("--terrain", choices=VC.TERRAINS, default=VC.DEFAULT_TERRAIN',
                   'parser.add_argument("--table-start", type=str, default=None',
                   'parser.add_argument("--table-park", action="store_true"',
                   "slot = VC.routed_slot(RULE, schedule_stand=stand_sched, eval_slot=eval_slot)",
                   "eval_slot = ctx.slot_rule.rederive_on_reset()",
                   "terrain_mesh = _mesh_record(mesh, mesh_key)", "FIELD_MIN_Z_RANGE_M",
                   "CORE.terrain_z_under_feet(fpos, mesh, raycast_mesh)"):
        assert needle in src, needle
    # the scenario's row env: block is still refused (the terrain comes from --terrain only)
    assert "carries an env: block" in src


_STUB_APP = '''
import json, os, sys


class AppLauncher:
    """A stub: the recorder's pre-Kit path runs for real; the constructor (Kit's boot) prints the env and exits 42."""

    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--enable_cameras", action="store_true")
        parser.add_argument("--device", default="cpu")

    def __init__(self, args):
        env = {k: v for k, v in os.environ.items() if k.startswith("KRABBY_")}
        print("STUB_KIT_BOOT " + json.dumps(env, sort_keys=True), flush=True)
        sys.exit(42)
'''


def _run_recorder_pre_kit(tmp_path, args) -> subprocess.CompletedProcess:
    stub = tmp_path / "stub"
    (stub / "isaaclab" / "app").mkdir(parents=True, exist_ok=True)
    (stub / "isaaclab" / "__init__.py").write_text("")
    (stub / "isaaclab" / "app" / "__init__.py").write_text(_STUB_APP)
    import os  # noqa: PLC0415

    env = {k: v for k, v in os.environ.items() if not k.startswith("KRABBY_")}
    env.update(PYTHONPATH=str(stub), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", KRABBY_STALE_VAR="x")
    return subprocess.run([sys.executable, "-B", str(REC_PATH), "--headless", "--enable_cameras",
                           "--out-dir", str(tmp_path / "out"), *args], env=env, capture_output=True, text=True,
                          timeout=600, cwd=str(tmp_path))


def test_recorder_pre_kit_path_with_a_stub_kit(GEN, VC, CORE, EC, tmp_path):
    """The recorder's real pre-Kit code (argument parsing, the reference checks, the row loaders, the table refusals,
    the env written before Kit) under a stub ``isaaclab.app`` whose AppLauncher exits before anything boots."""
    ref, notes = _current_reference(CORE, EC)
    if ref is None:
        pytest.skip("no stand stamp matches the current code -- re-stamp to re-enable (" + "; ".join(notes) + ")")
    stand_head, stamp = ref
    import json  # noqa: PLC0415

    vids = tmp_path / "v"
    assert GEN.main(["--plan", "tour_smoke", "--out-dir", str(vids)]) == 0
    assert GEN.main(["--plan", "smoke", "--out-dir", str(vids)]) == 0
    head = ["--stand-ckpt", str(stand_head), "--stand-mask", "1", "--deadband", "0.25", "--eval-run", str(stamp)]
    tour = ["--manifest", str(vids / "scenarios_sh_video_tour_smoke.yaml"), "--sequence", "sh_tour_smoke__omni"]
    ref_env = {k: str(v) for k, v in json.loads((stamp / "run_meta.json").read_text())["env_vars"].items()}

    def booted_env(r):
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("STUB_KIT_BOOT ")]
        assert r.returncode == 42 and len(line) == 1, (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
        return json.loads(line[0][len("STUB_KIT_BOOT "):])

    r = _run_recorder_pre_kit(tmp_path, tour + head + ["--terrain", "field_medium", "--table-start",
                                                         "0.635:knee_folded"])
    env = booted_env(r)
    assert env == VC.terrain_env(ref_env, "field_medium")[0] and "KRABBY_STALE_VAR" not in env
    assert "stand rule schedule" in r.stdout and "'cell': 'table_0p635__knee_folded'" in r.stdout
    r = _run_recorder_pre_kit(tmp_path, tour + head + ["--stand-rule", "slot"])
    assert booted_env(r) == ref_env and "stand rule slot" in r.stdout                  # flat: the stamp's env
    for extra, needle in ((["--table-park"], "--table-park needs --table-start"),
                          (["--table-start", "0.762"], "0.31"),
                          (["--table-start", "0.7"], "K3 table surface"),
                          (["--terrain", "field_hard"], "invalid choice")):
        r = _run_recorder_pre_kit(tmp_path, tour + head + extra)
        assert r.returncode not in (0, 42) and needle in (r.stdout + r.stderr), (extra, r.stdout[-2000:], r.stderr)
    smoke = ["--manifest", str(vids / "scenarios_sh_video_smoke.yaml"), "--sequence", "sh_smoke__omni"]
    r = _run_recorder_pre_kit(tmp_path, smoke + head + ["--table-start", "0.635"])
    assert r.returncode not in (0, 42) and "must be a stand" in (r.stdout + r.stderr)
    assert booted_env(_run_recorder_pre_kit(tmp_path, smoke + head)) == ref_env       # a launcher row, unchanged


def test_recorder_flat_path_is_the_pre_tour_path(VC):
    """``--terrain flat`` writes the reference env itself; ``--stand-rule slot`` routes on the eval slot object itself;
    every table piece sits behind ``TABLE`` / ``ctx.table`` (None without --table-start)."""
    src = REC_PATH.read_text()
    lines = src.splitlines()
    env, rec = VC.terrain_env({"KRABBY_X": "1"}, "flat")
    assert env == {"KRABBY_X": "1"} and not rec["deviation"]
    for needle in ("table_env = _compose_table(env_cfg, table_plan)", 'table_env["probe"] = _register_table_probe(',
                   "table_ctx = _table_start_state("):
        (i,) = [k for k, ln in enumerate(lines) if needle in ln]
        assert any(ln.strip() == "if TABLE is not None:" or ln.strip().startswith("if TABLE is not None:")
                   for ln in lines[max(0, i - 2): i]), needle
    assert "if ctx.table is not None and bool(reset_rows.any()):" in src
    assert "    if tbl is None:\n        return" in src
    assert 'raw_env = gym.make(scenario.task, cfg=env_cfg, render_mode="rgb_array")' in src
    assert "env = ParkourRslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)" in src


# ===================================================================================== where on the field
OMNI_FIELD_CFG = REPO / "parkour/parkour_isaaclab/terrains/omni/config/omni_field.py"


def _field_grid_from_source() -> dict:
    """The omni field generator's grid, READ from ``omni_field.py`` (its ``ParkourTerrainGeneratorCfg`` literals)."""
    tree = ast.parse(OMNI_FIELD_CFG.read_text())
    (call,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "")
               == "ParkourTerrainGeneratorCfg"]
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg in ("size", "num_rows", "num_cols",
                                                                               "border_width")}
    return kw


def _tile_centre(grid: dict, r: int, c: int) -> tuple[float, float]:
    sx, sy = grid["size"]
    return (r + 0.5) * sx - grid["num_rows"] * sx / 2.0, (c + 0.5) * sy - grid["num_cols"] * sy / 2.0


def test_field_grid_tile_map(VC):
    grid = VC.field_grid_of(_field_grid_from_source())
    assert grid == {"size": [10.0, 10.0], "num_rows": 10, "num_cols": 20, "border_width": 20.0}
    for r in range(grid["num_rows"]):
        for c in range(grid["num_cols"]):
            x, y = _tile_centre(grid, r, c)
            assert VC.tile_of(x, y, grid) == (r, c)
            assert VC.tile_of(x + 4.99, y - 4.99, grid) == (r, c)
    assert VC.tile_of(-50.01, -95.0, grid) is None and VC.tile_of(50.0, -95.0, grid) is None      # x: rows 0..9
    assert VC.tile_of(0.0, -100.01, grid) is None and VC.tile_of(0.0, 100.0, grid) is None        # y: cols 0..19
    xs, ys = [44.0, 46.0, 51.0, 52.0], [-95.0, -95.0, -95.0, -95.0]
    assert VC.tile_visits(xs, ys, grid) == {"tiles": {"9,0": 2}, "border_steps": 2, "steps": 4}
    assert VC.tile_caption_text(None, {}) == "on the BORDER (off the field grid)"
    assert VC.tile_caption_text((3, 0), {"3,0": "omni_field_dense"}) == "tile 3,0 omni_field_dense"
    rec = {"terrain": "field_medium",
           "env_block": {"KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70", "KRABBY_FIELD_EDGES": "wall"}}
    assert (VC.setup_caption_text(rec, "schedule", "tile 3,0 omni_field_dense")
            == "TERRAIN field_medium (field DIFF 0.20:0.70, wall edges) tile 3,0 omni_field_dense   |   stand rule: "
               "schedule")
    assert VC.setup_caption_text(rec, "schedule") == VC.setup_caption_text(rec, "schedule", None)


def test_tour_stays_on_the_field_only_with_the_pinned_spawn_row(VC, tour, tour_smoke):
    """The review's defect: unpinned, env 0 spawns on a uniformly random row 0..9 of column 0, and the tour's commanded
    path (+x, upper bound ~17.2 m) reaches the flat border from rows 8 and 9; the recorder pins
    ``max_init_terrain_level`` to the highest row that keeps the path (+ 2 m) on field tiles."""
    grid = VC.field_grid_of(_field_grid_from_source())
    ex = VC.commanded_extent(tour["holds"], prologue_s=1.5)                 # the prologue holds hold 0 (a stand)
    want_x = 21 * .25 + 14.6 * .25 - 10 * .22 + 10 * .25 * math.cos(0.7853982) + 15.4 * .22 + 10 * .22 + 12.6 * .25
    assert ex["x_max"] == pytest.approx(want_x, abs=1e-6) and ex["x_min"] == 0.0
    assert ex["y_max"] == pytest.approx(2.2) and ex["y_min"] == pytest.approx(2.2 - 15.4 * .22)
    assert len(ex["ends"]) == len(tour["holds"]) and ex["ends"][-1][0] == pytest.approx(want_x, abs=1e-5)
    need = VC.field_tiles_needed(ex, grid)
    assert VC.FIELD_PATH_MARGIN_M == 2.0
    assert need == {"ahead": 2, "behind": 0, "left": 0, "right": 0, "margin_m": 2.0}
    cap = VC.field_spawn_row_cap(need, grid)
    assert cap == 7
    for r in range(grid["num_rows"]):
        x0, y0 = _tile_centre(grid, r, VC.FIELD_SPAWN_COL)
        why = VC.field_path_reasons(x0, y0, ex, grid)
        assert (why == []) == (r <= cap), (r, why)                           # rows 8, 9 (unpinned 20 %) -> border
        tiles = VC.field_path_tiles(x0, y0, ex, grid)
        assert (r, 0) in tiles and all(c == 0 for _r, c in tiles)
    x0, y0 = _tile_centre(grid, cap, 0)
    assert VC.field_path_tiles(x0, y0, ex, grid) == [(cap, 0), (cap + 1, 0), (cap + 2, 0)]
    ex_s = VC.commanded_extent(tour_smoke["holds"], prologue_s=1.5)            # the smoke: +y 3.15 m at face +90
    need_s = VC.field_tiles_needed(ex_s, grid)
    assert (need_s["ahead"], need_s["left"], need_s["right"]) == (0, 1, 0) and VC.field_spawn_row_cap(need_s, grid) == 9
    assert all(VC.field_path_reasons(*_tile_centre(grid, r, 0), ex_s, grid) == [] for r in range(10))


def test_field_placement_refusals_and_the_prologue_leg(VC):
    grid = {"size": [10.0, 10.0], "num_rows": 10, "num_cols": 20, "border_width": 20.0}
    hold = lambda label, sp, di, s: {"label": label, "speed": sp, "dir": di, "face": 0.0, "hold_s": s}  # noqa: E731
    ex = VC.commanded_extent([hold("fwd", 0.25, 0.0, 10.0)], prologue_s=1.5)
    assert ex["x_max"] == pytest.approx(0.25 * 11.5) and ex["ends"] == [[pytest.approx(2.875), 0.0]]
    for holds, needle in (([hold("back", 0.25, math.pi, 40.0)], "behind"),
                          ([hold("lat_r", 0.25, -math.pi / 2, 20.0)], "-y of spawn column 0"),
                          ([hold("fwd", 0.25, 0.0, 400.0)], "ahead (+x)")):
        need = VC.field_tiles_needed(VC.commanded_extent(holds), grid)
        with pytest.raises(ValueError, match=re.escape(needle)):
            VC.field_spawn_row_cap(need, grid)


def test_recorder_pins_the_spawn_row_on_the_field_only_and_records_the_tiles():
    """Source-level: ``max_init_terrain_level`` is written only in ``_field_placement``, called only off the flat path,
    after the harness's overrides and before the env is built; the realised spawn is re-checked (refusing) before the
    first step; the tile is captioned and recorded per hold and in the sanity JSON's terrain block."""
    src = REC_PATH.read_text()
    tree = ast.parse(src)
    src_of = lambda n: ast.get_source_segment(src, n) or ""  # noqa: E731
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any("max_init_terrain_level" in src_of(t) for t in n.targets)]
    assert len(writes) == 1
    (fp,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_field_placement"]
    assert fp.lineno <= writes[0].lineno <= fp.end_lineno
    lines = src.splitlines()
    (call,) = [k for k, ln in enumerate(lines) if "field_place = _field_placement(env_cfg, scenario)" in ln]
    assert lines[call - 1].strip().startswith('if args_cli.terrain != "flat":')
    (ovr,) = [k for k, ln in enumerate(lines)
              if 'applied_overrides = H["_apply_env_overrides"](env_cfg, scenario)' in ln]
    (make,) = [k for k, ln in enumerate(lines) if 'raw_env = gym.make(scenario.task, cfg=env_cfg' in ln]
    (real,) = [k for k, ln in enumerate(lines)
               if "field_real, field_gen = _field_realised(unwrapped, robot, field_place)" in ln]
    (first,) = [k for k, ln in enumerate(lines) if "_run_sequence(ctx, scenario, seq_info, kinds)" in ln
                and not ln.lstrip().startswith("def ")]
    assert ovr < call < make < real < first and lines[real - 1].strip().startswith("if field_place is not None:")
    for needle in ('"terrain": dict(TERRAIN, terrain_mesh=ctx.terrain_mesh, field=field_summary,',
                   's["terrain_tiles"] = VC.tile_visits(rec["x_post"][idx], rec["y_post"][idx], grid)',
                   "extra = [(VC.setup_caption_text(TERRAIN, RULE, tile_text), VC.WHITE)]",
                   "why = VC.field_path_reasons(root[0], root[1], extent, grid)",
                   'raise SystemExit(f"--terrain {args_cli.terrain}: the video would leave the field: "'):
        assert needle in src, needle


# ===================================================================================== the table start
def test_parse_table_start_and_refusals(VC, tour):
    tb = VC.parse_table_start("0.635:knee_folded")
    assert tb["top"] == 0.635 and tb["pose"] == "knee_folded" and tb["cell"] == "table_0p635__knee_folded"
    assert tb["cmd_min"] == pytest.approx(0.945) and tb["spawn_h_plan"] == pytest.approx(0.79875)
    assert VC.parse_table_start("0.635") == tb and VC.parse_table_start(None) is None
    assert VC.parse_table_start("") is None
    for bad, why in (("0.7", "K3 table surface"), ("0.635:random", "random fold"), ("abc", "not a number"),
                     ("0.635:knee_folded:x", "TOP"), ("0.635:sitting", "pose")):
        with pytest.raises(ValueError, match=why):
            VC.parse_table_start(bad)
    kinds = VC.hold_kinds(tour["holds"])
    assert VC.table_start_reasons(tb, kinds, tour["heights"]) == []
    assert VC.table_start_reasons(VC.parse_table_start("0.6985"), kinds, tour["heights"]) == []   # 1.0085 <= 1.064
    why = VC.table_start_reasons(VC.parse_table_start("0.762"), kinds, tour["heights"])
    assert why and "top 0.762 + 0.31" in why[0]
    assert "must be a stand" in VC.table_start_reasons(tb, ["walk"] + kinds[1:], [None] + tour["heights"][1:])[0]
    assert VC.table_start_reasons(tb, kinds, [0.94] + tour["heights"][1:])                        # 0.94 < 0.945
    n = VC.table_need_s(tb, 1.064)
    assert n["h_init"] == pytest.approx(0.79875) and n["rule_s"] == pytest.approx(n["reach_s"] + 3.0)
    assert 4.0 < n["reach_s"] < 5.0                                                                # rise 1.18 + ramp 3.32


def test_recorder_refuses_table_start_misuse_before_kit():
    src = REC_PATH.read_text()
    pre_kit = src[: src.index("app_launcher = AppLauncher(args_cli)")]
    for needle in ("TABLE = VC.parse_table_start(args_cli.table_start)",
                   'raise SystemExit("--table-park needs --table-start")',
                   "_why = VC.table_start_reasons(TABLE,", "HEIGHT_ROW = VC.video_height_row("):
        assert needle in pre_kit, needle
    for needle in ("_why = VC.table_start_reasons(TABLE, kinds, HEIGHT_ROW[\"hold_heights\"])",
                   "if scenario.spawn_yaw_rad() != 0.0:", "TC.plan_envs([TABLE[\"cell\"]], 1,",
                   "if hold0 + 1e-9 < need[\"rule_s\"]:"):
        assert needle in src, needle


def test_table_start_reset_starts_the_limiter_at_the_measured_height(VC, TV):
    n = 1
    h_cmd = torch.full((n,), 1.064, dtype=torch.float64)
    est = TV._Est(TV.P1 - TV.P0, 0.01)
    obs = TV._obs_seq(1, n, seed=5)[0]
    kw = dict(s1_policy=TV._s1, stand_policy=TV._stand, stand_estimator=est, p0=TV.P0, p1=TV.P1, num_prop=TV.NP,
              mask=1, deadband=0.25)
    ctl = VC.new_controller(n, "cpu", dt=0.02, h_cmd0=h_cmd)
    rows, cut = VC.table_start_reset(ctl, torch.tensor([True]), h_cmd=h_cmd,
                                     h_meas=torch.tensor([0.79875], dtype=torch.float64))
    assert rows.tolist() == [False] and cut.tolist() == [False]
    assert float(ctl.h_lim[0]) == pytest.approx(0.79875)
    st = VC.height_step(ctl, obs.clone(), slot=torch.tensor([True]), reset_rows=rows, h_cmd=h_cmd, **kw)
    assert float(st.hs.h_lim[0]) == pytest.approx(0.79875 + 0.08 * 0.02)              # ramps up at 0.08 m/s
    assert bool(st.hs.stand_eff[0]) and not bool(st.hs.restoring[0])
    # the walk-handover path (no table) starts AT the command
    ctl2 = VC.new_controller(n, "cpu", dt=0.02, h_cmd0=h_cmd)
    st2 = VC.height_step(ctl2, obs.clone(), slot=torch.tensor([True]), reset_rows=torch.tensor([True]), h_cmd=h_cmd,
                         **kw)
    assert float(st2.hs.h_lim[0]) == pytest.approx(1.064)
    # the table measurement is clamped into the frozen range
    ctl3 = VC.new_controller(n, "cpu", dt=0.02)
    VC.table_start_reset(ctl3, torch.tensor([True]), h_cmd=h_cmd, h_meas=torch.tensor([0.5], dtype=torch.float64))
    assert float(ctl3.h_lim[0]) == pytest.approx(VC.CORE.H_MIN)
    tc = VC.table_core()
    assert VC.table_stood_now(0.11, 4, 1.05, 1.064) and not VC.table_stood_now(0.09, 6, 1.064, 1.064)
    assert not VC.table_stood_now(0.2, 3, 1.064, 1.064) and not VC.table_stood_now(0.2, 6, 1.0, 1.064)
    assert tc.STOOD_RISE_M == 0.10 and tc.STOOD_MIN_FEET == 4 and tc.STOOD_H_TOL_M == 0.04


def test_falls_are_recorded_not_terminated(VC):
    T, dt = 420, 0.02
    rise = np.minimum(np.arange(T) * 0.001, 0.2)                                       # >= 0.10 from step 100
    hip = np.zeros(T)
    hip[40:50] = 800.0                                                                   # before arming: not a fall
    hip[300:305] = 800.0                                                                 # armed: a hip-contact fall
    hip[306:310] = 800.0                                                                 # a second onset
    cos_t = np.ones(T)
    cos_t[350:360] = math.cos(0.6)                                                       # tilt > 0.5 rad
    starts = np.zeros(T, dtype=bool)
    starts[380] = True                                                                   # an env reset: re-arm
    rise2 = rise.copy()
    rise2[380:] = 0.0
    hip[385:390] = 900.0                                                                 # after the reset, not re-armed
    r = VC.table_fall_read(cos_t, hip, rise2, dt, tilt_max=0.5, contact_n=500.0, episode_start=starts)
    assert [(e["step"], e["cause"]) for e in r["events"]] == [(300, "hip_contact"), (306, "hip_contact"), (350, "tilt")]
    assert r["falls"] == 3 and r["armed_first_s"] == pytest.approx(101 * dt)
    assert r["hip_contact_pre_arm_steps"] == 10 + 5 and r["hip_force_pre_arm_max_n"] == 900.0
    assert r["max_tilt_rad"] == pytest.approx(0.6) and r["tilt_fall_steps"] == 10 and r["hip_fall_steps"] == 9
    tr = VC.FallTracker(dt, tilt_max=0.5, contact_n=500.0)
    for t in range(T):
        tr.update(t, cos_t[t], hip[t], rise2[t], episode_start=bool(starts[t]))
    assert tr.events == r["events"] and tr.armed_first_s == r["armed_first_s"]
    assert "never reset" in VC.FALL_RULE and "0.10 m" in VC.FALL_RULE
    # the recorder: crab_failure disabled on table starts only, no env reset of its own, the read + tracker recorded
    src = REC_PATH.read_text()
    body = src.split("def _compose_table", 1)[1].split("\ndef ", 1)[0]
    assert "env_cfg.terminations.crab_failure = None" in body and "STT.crab_failure_params(env_cfg.terminations)" in body
    assert src.count("crab_failure = None") == 1 and "env.reset(" not in src and ".reset(env_ids" not in src
    for needle in ("VC.FallTracker(dt, tilt_max=tbl[\"fail\"][\"limit_angle\"]", "VC.table_fall_read(rec[\"cos_tilt\"]",
                   '"tracker_matches_read": list(tbl["tracker"].events) == read["events"]',
                   "TC.classify(", "VC.table_caption_lines(TABLE,"):
        assert needle in src, needle
    lines = VC.table_caption_lines(VC.parse_table_start("0.635"), arrival_s=3.4, stood_s=4.12, falls=1,
                                   last_fall={"t_s": 6.0, "cause": "tilt"}, parked_s=None, park=False)
    assert "TABLE 0.635 m knee_folded: arrived 3.4 s, stood 4.1 s, table stays; falls 1" in lines[0][0]
    assert lines[1][1] == VC.RED and "no reset" in lines[1][0]


def test_table_composition_reuses_the_k3_pieces():
    src = REC_PATH.read_text()
    body = src.split("def _compose_table", 1)[1].split("\ndef ", 1)[0]
    for needle in ("TC.event_params(plan, joint_names)", "EventTermCfg(func=TC.sh_start, mode=\"reset\"",
                   "size=(TC.TABLE_X_M, TC.TABLE_Y_M, TC.TABLE_HEIGHT_M)", "mass=TC.TABLE_MASS_KG",
                   "kinematic_enabled=True", "pos=(0.0, 0.0, TC.PARK_CENTER_Z_M)", "prim_path=ST.TABLE_PRIM",
                   "resets[-1] != \"sh_start\""):
        assert needle in body, needle
    assert not re.search(r"\b0\.9\b|\b0\.7\b|\b0\.70\b|\b0\.90\b", body)               # no copied geometry
    park = src.split("def _table_step", 1)[1].split("\ndef ", 1)[0]
    for needle in ("ST.StartState.write_table(holder", "ST.StartState._park_pos(holder", "ST.SPEC.TABLE_PARK_CLEAR_M",
                   "s[\"prev_routed\"] is True and not st.routed"):
        assert needle in park, needle
    assert "TC.RUNTIME[\"terrain_probe\"] = TC.make_terrain_probe(mesh, raycast_mesh)" in src


# ===================================================================================== the .sh launcher
def test_launcher_passes_the_tour_options_through(TV, tmp_path):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    env, marker = TV._fake_bin(tmp_path, "/usr/bin/bash idle")
    before = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    head = str(T17_HEAD)
    for plan, tmo in (("tour", "5400"), ("tour_smoke", "1800")):
        r = subprocess.run(["bash", str(SH_PATH), "--dry-run", "--plan", plan, "--terrain", "field_medium",
                            "--table-start", "0.635:knee_folded", "--stand-rule", "schedule", "--prefix",
                            f"sh_{plan}_T17", "--", "--stand-ckpt", head, "--stand-mask", "1", "--deadband", "0.25"],
                           env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr
        cmd = [ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]
        assert len(cmd) == 1, r.stdout
        c = cmd[0]
        for needle in (f"timeout --signal=TERM {tmo} ", f"--sequence sh_{plan}__omni", f"scenarios_sh_video_{plan}.yaml",
                       "--terrain field_medium", "--table-start 0.635:knee_folded", "--stand-rule schedule",
                       f"--prefix sh_{plan}_T17", "--deadband 0.25", "MemoryMax=24G", "MemorySwapMax=0"):
            assert needle in c, (needle, c)
        assert " -k " not in c and "--table-park" not in c
        assert "first stand on the 0.635 m table" in r.stdout                        # the generator checked the table
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run", "--plan", "tour", "--table-start", "0.635", "--table-park"],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "--table-park" in r.stdout and "sh_tour__omni" in r.stdout, r.stderr
    assert not marker.exists() and not (tmp_path / "out").exists()
    after = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    assert after == before                                                              # a dry run writes nothing


def test_launcher_tour_refusals(TV, tmp_path):
    env, marker = TV._fake_bin(tmp_path, "/usr/bin/bash idle")
    for args, needle in ((["--dry-run", "--table-park"], "--table-park needs --table-start"),
                         (["--dry-run", "--plan", "walkabout"], "--plan must be"),
                         (["--dry-run", "--plan", "tour", "--smoke"], "exclusive"),
                         (["--dry-run", "--plan", "tour", "--seed", "3"], "exclusive"),
                         (["--dry-run", "--terrain", "field_hard"], "--terrain must be"),
                         (["--dry-run", "--stand-rule", "latched"], "--stand-rule must be"),
                         (["--dry-run", "--table-start", "0.635;rm"], "TOP[:POSE]"),
                         (["--dry-run", "--plan", "tour", "--", "--terrain", "field_medium"], "before --"),
                         (["--dry-run", "--plan", "tour", "--", "--table-start=0.635"], "before --"),
                         (["--dry-run", "--plan", "tour", "--table-start", "0.762"], "schedule generation failed"),
                         (["--dry-run", "--table-start", "0.635"], "schedule generation failed")):
        r = subprocess.run(["bash", str(SH_PATH), *args], env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 5 and needle in r.stderr, (args, r.returncode, r.stderr)
    assert not marker.exists()


def test_launcher_refuses_a_tour_while_busy_before_generating(TV, tmp_path):
    env, marker = TV._fake_bin(tmp_path, TV.ISAAC_LINE)
    gen_marker = tmp_path / "GEN_RAN"
    fake_py = tmp_path / "bin" / "python"
    fake_py.write_text(f"#!/bin/sh\ntouch {gen_marker}\nexit 1\n")
    fake_py.chmod(0o755)
    env["SH_PY"] = str(fake_py)
    r = subprocess.run(["bash", str(SH_PATH), "--plan", "tour", "--terrain", "field_medium", "--table-start", "0.635"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "another Isaac process is running" in r.stderr
    assert not marker.exists() and not gen_marker.exists()


def test_no_bytecode_from_the_tour_tools():
    """Nothing these tests load by path writes bytecode (the snapshot is taken when this module is imported; the
    campaign's / predecessor's pre-existing caches are someone else's and left alone)."""
    assert _pyc_snapshot() == _PYC_BEFORE
    for pat in ("sh_video*", "sh_record*", "sh_table_core*", "sh_starts*", "sh_table_test*"):
        assert not list(SCRIPTS.glob(f"__pycache__/{pat}")), pat
