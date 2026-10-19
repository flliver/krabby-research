"""M18 Task 4 pipeline tooling (TP: ``crab_hex_omni_task/experiments/tools/pipeline/``), CPU only.

Pinned (PLAN step 6 / 7; DESIGN 9):
* ``gen``: byte equality (gen twice -> identical files; ``gen --check`` clean), every driver stage resumes the previous
  stage by ``{reference:}`` in the SAME candidates file (no checkpoint / cross-campaign resume), the driver's own
  validator accepts the generated file, overlay sha / verbatim checks against the shipped lineage (and refuse a drift),
  the composed eval env equals the recorded lineage stamps' (modulo the inert COMMAND_FRAME=world), payload level files;
* the placeholders an official run refuses (N_p, x_rule, the student manifest-shape split);
* Gate 0 INVALID: flipped CLOCK_HIDE, missing trained_categories, payload readback mismatch, a non-zero payload on a
  0-kg row (all-zero = none), wrong pin, wrong checkpoint, incomplete stamp;
* golden gate reads on recorded stamps: PASS on S1 s16/17/18 at p2a, HALT on C0 at s1, FAIL on ANCW at p2a field_hard;
* the queue: HALT record (line, read, chance rate, subtree) + sentinel + park, ``continue --accept-chance`` resumes the
  halted subtree, INVALID reruns the rows once then halts, GUARD is not acceptable as chance, a student-branch HALT
  never holds the stand branch, a p3b floor miss is a measured ceiling (p3c skipped, T* from the heads so far);
* the pin guard refuses a drifted locked file / a moved lock;
* the stand overrides unset = today; the driver-side payload knobs (parsers, dump checks both directions, KG needs its
  companions); gate_row.sh refusals and key stripping; the Isaac pattern never matches a TP tool;
* live_watch replays on recorded tfevents; payload_read on S1's 0-kg stamps; replay_read's amended P-early rule;
  select_student's candidate mapping and its 2-block HALT.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TP = OMNI / "experiments/tools/pipeline"
PL = OMNI / "experiments/2026-10-16_task4_pipeline"
SPEC = PL / "pipeline.yaml"
REFS18 = REPO / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24"
X43_SUITE = (REPO / "parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-01_ppo_construction/seed18_X43_s1_A15pB/"
             "suite__omni/seed001/2026-10-04_16-15-43")
HAVE_STAMPS = REFS18.is_dir() and X43_SUITE.is_dir()
needs_stamps = pytest.mark.skipif(not HAVE_STAMPS, reason="recorded eval stamps are not on this machine")


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def P():
    return _load(TP / "pipeline.py", "tp_pipeline")


@pytest.fixture(scope="module")
def G(P):
    return P.GR


@pytest.fixture(scope="module")
def L(P):
    return P.L


@pytest.fixture(scope="module")
def S(P):
    return P.Spec(SPEC)


@pytest.fixture()
def tspec(tmp_path, P):
    """A copy of the spec + thresholds in a temp campaign dir (run dirs land there), N_p set."""
    d = tmp_path / "camp"
    d.mkdir()
    for n in ("thresholds_v0.json", "thresholds_v1.json", "threshold_inputs.yaml", "threshold_inputs_v1.yaml"):
        if (PL / n).is_file():
            shutil.copy(PL / n, d / n)
    doc = yaml.safe_load(SPEC.read_text())
    doc["N_p"] = 2500
    doc["x_rule"] = "report_only"
    for st in doc["stages"]:
        if st["id"] == "sel":
            st["completion_margin"]["manifest_shape_split"] = 0.03
    (d / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return P.Spec(d / "pipeline.yaml")


# ============================================================================== layout
def test_no_init_and_no_dated_path_in_tp_source():
    import re
    assert not (TP / "__init__.py").exists()
    for p in TP.iterdir():
        if p.suffix not in (".py", ".sh"):
            continue
        for i, ln in enumerate(p.read_text().splitlines(), 1):
            code = ln.split("#", 1)[0]
            if re.search(r"[\"'/]20\d\d-\d\d-\d\d_[a-z]", code) and '"""' not in ln:
                pytest.fail(f"{p.name}:{i} carries a dated path: {ln.strip()[:100]}")


def test_isaac_pattern_never_matches_a_tp_tool(L):
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    mine = [f"{py} -I {TP}/{n}.py _queue --spec x --run R18" for n in
            ("pipeline", "gate_read", "live_watch", "payload_read", "replay_read", "select_student", "report",
             "thresholds", "tp_lib")]
    assert not L.isaac_busy("\n".join(mine + [f"bash {TP}/gate_row.sh a b c", f"bash {TP}/distill.sh --name x"]))
    for ln in (f"{py} -u /x/eval_crab_hex_gait.py --headless", f"{py} /x/scripts/rsl_rl/train.py --task T",
               f"{py} /x/sh_dagger.py --headless --out /y", f"{py} -B /x/run_candidates.py --unit u"):
        assert L.isaac_busy(ln), ln
    assert not L.isaac_busy(f"{py} -B /x/sh_dagger.py --emit-env")


# ============================================================================== gen
def test_gen_is_byte_stable_and_check_is_clean(P, tspec):
    a = P.gen_files(tspec, "R18")
    b = P.gen_files(tspec, "R18")
    assert a == b
    P.gen(tspec, "R18")
    assert P.gen(tspec, "R18", check=True) == []
    (tspec.run_dir("R18") / "envs/s1.env").write_text("KRABBY_X=1\n")
    assert any("s1.env" in x for x in P.gen(tspec, "R18", check=True))


def test_gen_rewrites_atomically_and_keeps_the_mode(P, tspec):
    """A running queue's jobs read the generated files: gen replaces a changed file through a same-dir temp file and
    os.replace (no partial file, no temp left behind) and keeps the file's mode; an unchanged file is not touched."""
    P.gen(tspec, "R18")
    rd = tspec.run_dir("R18")
    f, g = rd / "envs/s1.env", rd / "envs/p2a.env"
    want = f.read_text()
    f.write_text("KRABBY_X=1\n")
    os.chmod(f, 0o640)
    ino_g, mt_g = g.stat().st_ino, g.stat().st_mtime_ns
    P.gen(tspec, "R18")
    assert f.read_text() == want and (f.stat().st_mode & 0o777) == 0o640
    assert (g.stat().st_ino, g.stat().st_mtime_ns) == (ino_g, mt_g)
    assert not [p for p in rd.rglob(".*") if p.is_file()]
    assert P.gen(tspec, "R18", check=True) == []


def test_committed_run_dirs_are_what_gen_writes_and_gen_is_idempotent(P, S):
    """PLAN step 7: ``gen --check`` on the COMMITTED R18 / F19 / dry run dirs (the real spec, nothing copied), and gen
    twice gives identical files. The placeholders (N_p, x_rule, the manifest-shape split) are the spec's own."""
    for run in ("R18", "F19", "dry"):
        assert P.gen_files(S, run) == P.gen_files(S, run), run
        assert P.gen(S, run, check=True) == [], run
    cli = subprocess.run([sys.executable, "-B", str(TP / "pipeline.py"), "gen", "--spec", str(SPEC), "--check"],
                         capture_output=True, text=True, timeout=300)
    assert cli.returncode == 0 and "byte-identical" in cli.stdout, cli.stdout + cli.stderr


def test_every_driver_stage_resumes_by_reference_in_the_same_file(P, tspec):
    for run in ("R18", "F19", "dry"):
        doc = P.candidates_doc(tspec, run)
        chains = doc["chains"]
        names = list(chains)
        assert names == [f"{run}_{s}" for s in ("s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b", "p3c")]
        for i, cid in enumerate(names):
            st = chains[cid]["stages"]
            assert len(st) == 1
            r = st[0]["resume_from"]
            if i == 0:
                assert r == "scratch" and st[0]["from_scratch"] is True
            else:
                assert r == {"reference": names[i - 1]}, (cid, r)
                assert "checkpoint" not in json.dumps(r)
            assert doc["references"][cid] == {"chain": cid, "seed": tspec.run(run)["walker_seed"]}
        text = yaml.safe_dump(doc)
        assert "parkour/logs/rsl_rl" not in text and "P2__S1_s18" not in text


def test_generated_candidates_pass_the_driver_validator(P, tspec):
    for run in ("R18", "F19", "dry"):
        assert P.validate_generated(tspec, run) == []


def test_seeds_iters_and_payload_overlays(P, tspec):
    assert [tspec.seed("R18", s) for s in ("s1", "p3a", "st", "sn", "sh")] == [18, 18, 18, 17, 17]
    assert {tspec.seed("F19", s) for s in ("s1", "p3a", "st", "sn", "sh")} == {19}
    assert [tspec.iters("R18", s) for s in ("s1", "c4", "c5", "c6", "c6x", "p2a", "p3a")] == \
        [6000, 1500, 1500, 2000, 4000, 5000, 2500]
    assert tspec.iters("dry", "p2a") == 20 and tspec.iters("dry", "st") == 50
    ov = tspec.overlay("p3b")
    p2a = tspec.overlay("p2a")
    assert list(ov)[:len(p2a)] == list(p2a)
    assert {k: ov[k] for k in ov if k.startswith("KRABBY_PAYLOAD_")} == {
        "KRABBY_PAYLOAD_KG": "0:180", "KRABBY_PAYLOAD_ZERO_FRAC": "0.25", "KRABBY_PAYLOAD_COM_XY": "0.10",
        "KRABBY_PAYLOAD_COM_Z": "0.309"}
    assert {s["id"] for s in tspec.stages("F19")} & {"sn_ref", "sh_ref"} == set()
    assert {"sn_ref", "sh_ref"} <= {s["id"] for s in tspec.stages("R18")}


def test_level_files_and_composed_envs(P, tspec, L):
    files = P.gen_files(tspec, "R18")
    for kg in (0, 45, 90, 135, 180, 225, 270):
        env = dict(ln.split("=", 1) for ln in files[f"envs/levels/L{kg:03d}.env"].splitlines())
        assert env == {"KRABBY_PAYLOAD_KG": f"{kg}:{kg}", "KRABBY_PAYLOAD_ZERO_FRAC": "0",
                       "KRABBY_PAYLOAD_COM_XY": "0", "KRABBY_PAYLOAD_COM_Z": "0.309"}
    plan = json.loads(files["plan.json"])
    assert plan["stages"]["s1"]["trained_csv"] == "stand,fwd,side,back"
    assert plan["stages"]["c4"]["trained_csv"] == "stand,fwd,side,back,turn,move_turn"
    assert plan["stages"]["p3a"]["trained_csv"] == "stand,fwd,side,back,diag,turn,move_turn"
    e = dict(ln.split("=", 1) for ln in files["envs/p3a.env"].splitlines())
    assert not any(k.startswith("KRABBY_PAYLOAD_") for k in e)              # the 0-kg rows are unarmed
    assert e["KRABBY_OMNI_CLOCK_HIDE"] == "1"
    assert dict(ln.split("=", 1) for ln in files["envs/s1.env"].splitlines())["KRABBY_OMNI_CLOCK_HIDE"] == "0"
    d = dict(ln.split("=", 1) for ln in files["envs/p3a.distill.env"].splitlines())
    assert d["KRABBY_OMNI_PHASE"] == "2a" and d["KRABBY_PAYLOAD_KG"] == "0:90"


@needs_stamps
def test_composed_eval_env_equals_the_recorded_lineage_stamps(P, tspec):
    """gen's composition is the driver's: X43 s1 and REFS_S1_s18 env_vars, modulo the inert COMMAND_FRAME=world the
    body-cmd campaign added to the carry (G0w)."""
    files = P.gen_files(tspec, "R18")
    for envf, stamp in (("envs/s1.env", X43_SUITE), ("envs/p2a.env", REFS18)):
        e = dict(ln.split("=", 1) for ln in files[envf].splitlines())
        m = json.loads((stamp / "run_meta.json").read_text())["env_vars"]
        assert set(e.items()) ^ set(m.items()) == {("KRABBY_OMNI_COMMAND_FRAME", "world")}


# ============================================================================== check
def test_lineage_overlays_are_verbatim_and_a_drift_is_refused(P, S, tspec):
    assert P.lineage_problems(S) == []
    assert P.graph_problems(S) == []
    assert [len(S.stage(s)["overlay"]) for s in ("s1", "c4", "c5", "c6", "c6x", "p2a")] == [14, 16, 16, 16, 19, 27]
    bad = copy.deepcopy(tspec)
    bad.stage("c6x")["overlay"]["KRABBY_OMNI_CLOCK_HIDE"] = "0"
    assert any("c6x" in x and "VERBATIM" in x for x in P.lineage_problems(bad))
    bad = copy.deepcopy(tspec)
    ov = bad.stage("s1")["overlay"]
    bad.stage("s1")["overlay"] = dict(reversed(list(ov.items())))      # same keys, another order
    assert any("s1" in x for x in P.lineage_problems(bad))


def _with_placeholders(P, tspec):
    """The spec as it was before the PLAN step-8 freeze: N_p, x_rule and the manifest-shape split are PLACEHOLDERs."""
    t = copy.deepcopy(tspec)
    t.d["N_p"] = "PLACEHOLDER: PLAN step 8 payload pilot sets it"
    t.d["x_rule"] = "PLACEHOLDER: PLAN step 8 determinism smoke"
    next(x for x in t.d["stages"] if x["id"] == "sel")["completion_margin"]["manifest_shape_split"] = \
        "PLACEHOLDER: measured at PLAN step 8"
    return t


def test_placeholders_refused_for_official_runs_only(P, S, tspec):
    ph = _with_placeholders(P, tspec)
    probs = P.placeholder_problems(ph, "R18")
    assert any("N_p" in x for x in probs) and any("x_rule" in x for x in probs)
    assert any("manifest_shape_split" in x for x in probs)
    assert not any("x_rule" in x for x in P.placeholder_problems(ph, "F19"))  # F19 has no X replay
    assert P.placeholder_problems(ph, "dry") == []
    assert P.placeholder_problems(tspec, "R18") == []
    for run in ("R18", "F19", "dry"):                       # the committed spec is FROZEN (PLAN step 8 values)
        assert P.placeholder_problems(S, run) == [], run
    t = copy.deepcopy(tspec)
    t.d["N_p"] = 1700
    assert any("multiple of 500" in x for x in P.placeholder_problems(t, "R18"))


# ============================================================================== Gate 0
def _fake_stamp(tmp_path, src: Path, name: str, *, meta_patch=None, code_state=None, drop=()):
    d = tmp_path / name
    d.mkdir()
    for f in ("scenario_metrics.json", "summary.md", "complete.json"):
        if (src / f).is_file() and f not in drop:
            shutil.copy(src / f, d / f)
    meta = json.loads((src / "run_meta.json").read_text())
    if meta_patch:
        meta_patch(meta)
    (d / "run_meta.json").write_text(json.dumps(meta))
    if "raw" not in drop:
        (d / "raw").mkdir()
    if code_state is not None:
        (d / "code_state.json").write_text(json.dumps(code_state))
    return d


def _expect(env, *, trained=None, level="absent", ckpt_sha=None, pin=None, stage="p2a", S=None):
    e = {"label": "row", "stage": stage, "pin": pin or {"code_sha": "c" * 64, "eval_sha": "e" * 64},
         "source": "row", "env": env, "carry": S.d["carry_env"], "ckpt_sha": ckpt_sha, "trained": trained}
    if level != "absent":
        e["level_kg"] = level
    return e


@needs_stamps
def test_gate0_invalid_cases(tmp_path, G, S):
    meta = json.loads((REFS18 / "run_meta.json").read_text())
    env = dict(meta["env_vars"])
    sha = meta["checkpoint_sha256"]
    cs = {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": "False"}
    trained = ["stand", "fwd", "side", "back", "diag", "turn", "move_turn"]

    def set_tc(m):
        m["trained_categories"] = list(trained)

    ok = _fake_stamp(tmp_path, REFS18, "ok", meta_patch=set_tc, code_state=cs)
    assert G.stamp_problems(ok, _expect(env, trained=trained, ckpt_sha=sha, level=None, S=S)) == []
    # flipped CLOCK_HIDE: the named carry check AND the key-for-key env compare
    def flip(m):
        set_tc(m)
        m["env_vars"]["KRABBY_OMNI_CLOCK_HIDE"] = "0"
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "flip", meta_patch=flip, code_state=cs),
                         _expect(env, trained=trained, ckpt_sha=sha, S=S))
    assert any("CLOCK_HIDE" in x and "carry" in x for x in p) and any("env_vars differ" in x for x in p)
    # missing trained_categories (the recorded REFS stamp itself carries null)
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "notc", code_state=cs), _expect(env, trained=trained, S=S))
    assert any("trained_categories missing" in x for x in p)
    # payload: a non-zero payload on a 0-kg row; all-zero / absent = none
    def armed(kg):
        def f(m):
            set_tc(m)
            m["payload"] = {"kg_range": [kg, kg], "zero_frac": 0.0, "com_xy": 0.0, "com_z": 0.309,
                            "payload_kg": [kg] * 100, "payload_com_m": [[0.0, 0.0, 0.309]] * 100}
        return f
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "pay90", meta_patch=armed(90.0), code_state=cs),
                         _expect(env, trained=trained, level=None, S=S))
    assert any("non-zero payload" in x for x in p)
    assert G.stamp_problems(_fake_stamp(tmp_path, REFS18, "pay0", meta_patch=armed(0.0), code_state=cs),
                            _expect(env, trained=trained, level=None, S=S)) == []
    # a level row: payload must read L on every env
    lv = {**env, "KRABBY_PAYLOAD_KG": "90:90"}

    def armed_env(kg, bad_env=False):
        def f(m):
            armed(kg)(m)
            m["env_vars"]["KRABBY_PAYLOAD_KG"] = "90:90"
            if bad_env:
                m["payload"]["payload_kg"][3] = 45.0
        return f
    assert G.stamp_problems(_fake_stamp(tmp_path, REFS18, "lv_ok", meta_patch=armed_env(90.0), code_state=cs),
                            _expect(lv, trained=trained, level=90, S=S)) == []
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "lv_bad", meta_patch=armed_env(90.0, True), code_state=cs),
                         _expect(lv, trained=trained, level=90, S=S))
    assert any("payload_kg != 90" in x for x in p)
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "lv_none", meta_patch=set_tc, code_state=cs),
                         _expect(env, trained=trained, level=90, S=S))
    assert any("no run_meta.payload" in x for x in p)
    # wrong pin / dirty tree / wrong checkpoint / incomplete
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "pin", meta_patch=set_tc,
                                     code_state={**cs, "eval_sha": "f" * 64, "dirty": "True"}),
                         _expect(env, trained=trained, S=S))
    assert any("eval_sha" in x for x in p) and any("dirty" in x for x in p)
    p = G.stamp_problems(ok, _expect(env, trained=trained, ckpt_sha="0" * 64, S=S))
    assert any("checkpoint_sha256" in x for x in p)
    p = G.stamp_problems(_fake_stamp(tmp_path, REFS18, "inc", code_state=cs, drop=("summary.md", "raw")),
                         _expect(env, S=S))
    assert any("summary.md" in x for x in p) and any("raw/" in x for x in p)


@needs_stamps
def test_c0_stamps_fail_gate0_on_their_carry_env(tmp_path, G, S):
    inp = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    st = REPO / inp["stages"]["s1"]["known_bad"]["C0_X39_s16"]["side22"]["stamp"]
    env = json.loads((st / "run_meta.json").read_text())["env_vars"]
    p = G.carry_problems(env, "s1", S.d["carry_env"])
    assert any("STD_PARAM" in x for x in p) and any("INIT_STD" in x for x in p)


def _stamps_in(doc, stage, path=""):
    if isinstance(doc, dict):
        if isinstance(doc.get("stamp"), str):
            yield stage, doc["stamp"], path
        for k, v in doc.items():
            yield from _stamps_in(v, stage, f"{path}/{k}")
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from _stamps_in(v, stage, f"{path}[{i}]")


@needs_stamps
def test_named_carry_check_on_every_recorded_stamp(G, S):
    """PLAN left-for-implementation item 2: the named carry env (stage-dependent CLOCK_HIDE; FACING_OBS_CLIP 0.6 on
    every stamp, P2-era included) PASSES on every recorded known-good stamp of every stage and FAILS on every C0
    known-bad (STD_PARAM / INIT_STD) -- so Gate 0 can never INVALID a good recorded row on its carry env."""
    inp = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    n_good = n_bad = 0
    for stage, d in inp["stages"].items():
        for st, stamp, path in _stamps_in(d, stage):
            meta = REPO / stamp / "run_meta.json"
            if not meta.is_file():
                continue
            env = {str(k): str(v) for k, v in json.loads(meta.read_text())["env_vars"].items()}
            p = G.carry_problems(env, st, S.d["carry_env"])
            if path.startswith("/known_bad/"):                     # C0 arms; ANCW (known_bad_field) carries a good env
                n_bad += 1
                assert p and (any("STD_PARAM" in x for x in p) or any("CLOCK_HIDE" in x for x in p)), (stage, path)
            else:
                n_good += 1
                assert p == [], (stage, path, p)
    assert n_good >= 80 and n_bad >= 16, (n_good, n_bad)


# ============================================================================== golden gate reads
@needs_stamps
def test_golden_reads(G, P):
    P.L.configure(yaml.safe_load(SPEC.read_text()))
    th = G.Thresholds.load(PL / "thresholds_v0.json")
    inp = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    p = inp["stages"]["p2a"]
    for s in (16, 17, 18):
        st = {f"suite@{es}": REPO / e["stamp"] for es, e in p["suite"][s].items()}
        st["side22@1"] = REPO / p["side22"][s]["stamp"]
        for i, e in enumerate(p["field"]["field_hard"][s]):
            st[f"field_hard@{i + 1}"] = REPO / e["stamp"]
        r = G.read_lines("p2a", st, th)
        assert r["verdict"] == G.PASS, (s, r["halt_lines"])
        assert any(d["id"] == "p2a.capability.field_hard" and d["pass"] for d in r["decisive"])
    s1 = inp["stages"]["s1"]
    for s in (16, 17, 18):
        st = {"suite@1": REPO / s1["suite"][s][1]["stamp"], "side22@1": REPO / s1["side22"][s]["stamp"]}
        assert G.read_lines("s1", st, th)["verdict"] == G.PASS
    for k, v in s1["known_bad"].items():
        r = G.read_lines("s1", {"suite@1": REPO / v["suite"]["stamp"], "side22@1": REPO / v["side22"]["stamp"]}, th)
        assert r["verdict"] == G.HALT and "s1.floor.side22_notslow" in r["halt_lines"], k
    for k, v in p["known_bad_field"].items():
        fh = G.field_hard_line("p2a", [REPO / e["stamp"] for e in v["field_hard"]], th, require_env_seeds=False)
        assert fh["pass"] is False and fh["value"] < 0.635 and fh["rate"] is not None, k
    with_two = G.field_hard_line("p2a", [REPO / e["stamp"] for e in p["known_bad_field"]["ANCW_s16"]["field_hard"]], th)
    assert with_two["missing"] and not with_two["pass"]                  # p2a's line needs es1-4


# ============================================================================== the queue (fake runtime)
class FakeRT:
    poll_s = 0

    def __init__(self, ctx_dir: Path, pin: dict, tmp: Path):
        self.dir, self.pin, self.tmp = ctx_dir, pin, tmp
        self.launched, self.rows, self.killed = [], [], []

    def time(self):
        return 0.0

    def sleep(self, s):
        return None

    def isaac_busy(self):
        return False

    def wait_idle(self, stop=None):
        return None

    def notify_hook(self, text):
        return None

    planned = None

    def launch_driver(self, cdir, chain, seed):
        st = json.loads((cdir / "state.json").read_text()) if (cdir / "state.json").is_file() else {}
        it = self.planned(chain.split("_", 1)[1]) if self.planned else 100
        ck = self.tmp / "runs" / f"{chain}" / f"model_{it}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(chain.encode())
        st.setdefault("ladder", {}).setdefault("references", {}).setdefault(chain, []).append(
            {"xid": "X01", "seed": str(seed), "ckpt": str(ck), "chain": chain, **self.pin})
        (cdir / "state.json").write_text(json.dumps(st))
        self.launched.append(chain)
        return f"cands-{chain}", ""

    def unit_active(self, unit):
        return False

    def stop_unit(self, unit):
        self.killed.append(unit)

    def term_scope(self, unit):
        self.killed.append(unit)

    def run(self, cmd, *, env=None, log=None, timeout=None):
        d = self.tmp / "stamps" / f"s{len(self.rows)}"
        d.mkdir(parents=True)
        (d / "tp_row.json").write_text("{}")
        self.rows.append(cmd)
        return 0, f"TPROW END name={cmd[1]} rc=0 stamp={d}"


@pytest.fixture(params=["F19"])
def qrun(request):
    return request.param


@pytest.fixture()
def qctx(tspec, P, tmp_path, monkeypatch, qrun):
    pin = {"code_sha": "c" * 64, "eval_sha": "e" * 64}
    tspec.lock_path().write_text(json.dumps(pin))
    P.gen(tspec, qrun)
    monkeypatch.setattr(P, "guard_problems", lambda *a, **k: [])
    monkeypatch.setattr(P, "ddc_gb", lambda: 0.0)
    monkeypatch.setattr(P, "driver_stamp", lambda ctx, head, row="suite__omni", es=1: tmp_path / "drv")
    verdicts = {}

    def fake_gate(ctx, sid, **kw):
        v = verdicts.get(sid, P.GR.PASS)
        if callable(v):
            v = v(ctx, sid)
        r = {"verdict": v, "gate0": ["bad env"] if v == P.GR.INVALID else [],
             "decisive": [{"id": f"{sid}.floor.ratio.side.es1", "pass": v not in (P.GR.HALT,), "value": 0.41,
                           "line": 0.5, "rate": 0.0056}], "halt_lines": []}
        if (ctx.spec.stage(sid).get("gate") or {}).get("class") == "payload":
            ms = {"p3a": 135, "p3b": 135, "p3c": 270}[sid]
            ctx.srec(sid)["max_stable"] = ms
            ctx.srec(sid)["zero_kg"] = {"verdict": "PASS", "max_margin_used": 0.4}
            r["max_stable"] = ms
        return r

    monkeypatch.setattr(P, "gate_stage", fake_gate)
    monkeypatch.setattr(P, "payload_rows_and_curve", lambda ctx, sid: {"curve": {}})
    ctx = P.Ctx(tspec, qrun)
    rt = FakeRT(ctx.dir, pin, tmp_path)
    rt.planned = lambda sid: P.Queue(ctx).planned_last_iteration(sid)
    ctx.rt = rt
    done = []

    def fake_kind(name):
        def ex(self, sid):
            done.append(sid)
            self.ctx.set_status(sid, "pass", f"fake {name}")
        return ex

    for k in ("ex_distill", "ex_select", "ex_dagger_sn", "ex_dagger_sh", "ex_bundle"):
        monkeypatch.setattr(P.Queue, k, fake_kind(k))
    return ctx, rt, verdicts, done


def _sentinels(ctx, kind):
    return [ln for ln in ctx.qlog.read_text().splitlines() if ln.startswith(f"T4PIPE {kind} ")]


def test_queue_runs_the_graph_and_reads_every_driver_stage(P, qctx):
    ctx, rt, verdicts, done = qctx
    assert P.Queue(ctx).loop() == "done"
    assert rt.launched == [f"F19_{s}" for s in ("s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b")]
    st = ctx.state["stages"]
    assert st["p3c"]["status"] == "skipped"                        # max_stable(p3b) 135 < 180
    assert st["tstar"]["status"] == "pass" and ctx.state["tstar"]["stage"] == "p3a"   # tie 135 -> the earlier
    assert done == ["st", "sel", "sn", "sh", "bundle"]
    assert _sentinels(ctx, "END")
    rows = [c for c in rt.rows if "side22_25lr__omni" in c]
    assert rows                                                     # gate rows through gate_row.sh
    suites = [c for c in rt.rows if c[5] == "suite__omni"]
    assert all("--trained-categories" in c for c in suites)
    assert all("--trained-categories" not in c for c in rt.rows if c[5] == "side22_25lr__omni")


def test_halt_records_parks_and_accept_chance_resumes(P, qctx):
    ctx, rt, verdicts, done = qctx
    verdicts["c4"] = P.GR.HALT
    assert P.Queue(ctx).loop() == "parked"
    st = ctx.state["stages"]
    assert st["c4"]["status"] == "halt" and st["c5"]["status"] == "blocked" and st["p2a"]["status"] == "blocked"
    rec = json.loads((ctx.dir / "halts/c4.1.json").read_text())
    assert rec["class"] == "FAIL" and rec["lines"][0]["id"] == "c4.floor.ratio.side.es1"
    assert rec["lines"][0]["value"] == 0.41 and rec["chance_rate_sum"] == pytest.approx(0.0056)
    assert "c5" in rec["subtree"] and "sh" in rec["subtree"]
    assert any("stage=c4 class=FAIL" in ln for ln in _sentinels(ctx, "HALT")) and _sentinels(ctx, "PARKED")
    assert done == []
    # the user accepts it as chance: the run continues from the halted stage, disclosed
    assert P.do_continue(ctx.spec, "F19", accept="c4", why="chance", relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "F19")
    ctx2.rt = rt
    assert ctx2.state["stages"]["c4"]["status"] == "accepted"
    assert ctx2.state["accepted"][0]["stage"] == "c4"
    assert json.loads((ctx.dir / "halts/c4.1.json").read_text())["status"] == "accepted"
    assert P.Queue(ctx2).loop() == "done"
    assert ctx2.state["stages"]["c5"]["status"] == "pass" and "bundle" in done


def test_invalid_reruns_the_rows_once_then_halts(P, qctx):
    ctx, rt, verdicts, done = qctx
    verdicts["s1"] = P.GR.INVALID
    assert P.Queue(ctx).loop() == "parked"
    assert ctx.state["stages"]["s1"]["attempts"]["invalid"] == 2
    assert ctx.state["halts"]["s1"]["class"] == "INVALID"
    assert len([c for c in rt.rows if c[5] == "side22_25lr__omni"]) == 2      # the side22 row ran twice
    assert P.do_continue(ctx.spec, "F19", accept="s1", relaunch=False) == 2     # INVALID is not chance
    calls = []
    verdicts["s1"] = lambda c, s: calls.append(s) or P.GR.PASS
    assert P.do_continue(ctx.spec, "F19", retry="s1", relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "F19")
    ctx2.rt = rt
    assert ctx2.state["stages"]["s1"]["status"] == "rows"
    assert P.Queue(ctx2).loop() == "done" and calls == ["s1"]


def test_guard_halts_and_is_never_acceptable_as_chance(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx
    monkeypatch.setattr(P, "guard_problems", lambda *a, **k: ["code_sha drifted: live aaa vs lock bbb"])
    assert P.Queue(ctx).loop() == "parked"
    assert ctx.state["halts"]["s1"]["class"] == "GUARD" and rt.launched == []
    assert P.do_continue(ctx.spec, "F19", accept="s1", relaunch=False) == 2


def test_student_halt_never_holds_the_stand_branch(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx

    def sel_halt(self, sid):
        P.halt(self.ctx, sid, "STUDENT", "2 failed full blocks")
    monkeypatch.setattr(P.Queue, "ex_select", sel_halt)
    assert P.Queue(ctx).loop() == "parked"
    st = ctx.state["stages"]
    assert st["sel"]["status"] == "halt"
    assert st["sn"]["status"] == "pass" and st["sh"]["status"] == "pass"
    assert st["bundle"]["status"] == "blocked"                      # the deliverable waits for the user's decision


def test_p3b_floor_miss_is_a_ceiling_and_tstar_picks_from_the_heads_so_far(P, qctx):
    ctx, rt, verdicts, done = qctx
    verdicts["p3b"] = P.GR.CEILING
    assert P.Queue(ctx).loop() == "done"
    st = ctx.state["stages"]
    assert st["p3b"]["status"] == "ceiling" and st["p3c"]["status"] == "skipped"
    assert ctx.state["tstar"]["stage"] == "p3a" and not ctx.state["halts"]


def test_live_kill_on_p3c_is_a_ceiling_and_on_c4_a_halt(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx
    kills = {"c4": True}
    monkeypatch.setattr(P.Queue, "watch_driver", lambda self, sid, unit: (
        (self.ctx.srec(sid).update(live_killed={"rule": "collapse", "fails": ["x"]}),
         P.halt(self.ctx, sid, "LIVE", "live rule collapse", lines=[{"id": f"{sid}.live.collapse"}]))
        and True) if kills.pop(sid, False) else False)
    assert P.Queue(ctx).loop() == "parked"
    assert ctx.state["halts"]["c4"]["class"] == "LIVE"
    assert P.do_continue(ctx.spec, "F19", accept="c4", relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "F19")
    assert ctx2.state["stages"]["c4"]["status"] == "training"          # the window resumes, rule accepted
    assert ctx2.state["stages"]["c4"]["accepted_live"] == ["collapse"]


def test_abort(P, qctx):
    ctx, rt, verdicts, done = qctx
    verdicts["s1"] = P.GR.HALT
    P.Queue(ctx).loop()
    assert P.do_continue(ctx.spec, "F19", abort=True, relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "F19")
    assert ctx2.state["status"] == "aborted" and P.Queue(ctx2).loop() == "aborted"


# ============================================================================== the pin guard
def test_pin_guard_refuses_drift(P, tspec, tmp_path, monkeypatch):
    f = tmp_path / "locked.txt"
    f.write_text("a")
    rc = P.L.rc()
    lock = {"code_sha": rc.code_sha(), "eval_sha": rc.eval_sha(), "dirty_code_eval": [],
            "files": {P.L.rel(f): P.L.sha256_file(f)}, "external": {"x": "1"}}
    tspec.lock_path().write_text(json.dumps(lock))
    monkeypatch.setattr(P, "external_stack", lambda: {"x": "1"})
    monkeypatch.setattr(P, "lock_files", lambda spec: [f])
    monkeypatch.setattr(rc, "dirty_paths", lambda: [])
    assert P.guard_problems(tspec, check_disk=False) == []
    f.write_text("b")
    assert any("locked file drifted" in x for x in P.guard_problems(tspec, check_disk=False))
    f.write_text("a")
    pin = P.L.sha256_file(tspec.lock_path())
    lock["files"]["extra"] = None
    tspec.lock_path().write_text(json.dumps(lock))
    assert any("changed since the run was pinned" in x
               for x in P.guard_problems(tspec, pin_lock_sha=pin, check_disk=False))
    monkeypatch.setattr(P, "external_stack", lambda: {"x": "2"})
    assert any("external stack" in x for x in P.guard_problems(tspec, check_disk=False))
    monkeypatch.setattr(rc, "dirty_paths", lambda: ["parkour/scripts/x.py"])
    assert any("dirty" in x for x in P.guard_problems(tspec, check_disk=False))
    tspec.lock_path().unlink()
    assert any("no lock" in x for x in P.guard_problems(tspec))


def test_the_lock_pins_every_out_of_hash_dependency(P, S):
    files = {P.L.rel(p) for p in P.lock_files(S)}
    for want in ("experiments/tools/pipeline/pipeline.py", "experiments/tools/pipeline/gate_row.sh",
                 "experiments/2026-10-16_task4_pipeline/pipeline.yaml",
                 "experiments/2026-10-16_task4_pipeline/thresholds_v1.json",
                 "experiments/2026-10-16_task4_pipeline/rows/scenarios_pl_body.yaml",
                 "experiments/2026-09-28_side_bisect/scenarios_side_lr.yaml",
                 "experiments/2026-10-01_ppo_construction/scenarios_turn_band.yaml",
                 "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml",
                 "experiments/2026-10-01_ppo_construction/scripts/side22_classify.py",
                 "experiments/2026-10-11_stand_network/scripts/sn_dagger_spec.py",
                 "experiments/2026-10-13_stand_height/scripts/sh_kin.py",
                 "experiments/tools/launch_candidates.sh"):
        assert any(f.endswith(want) for f in files), want
    assert "assets/crab.usda" in files
    committed = json.loads((PL / "pipeline.lock.json").read_text())
    assert set(committed["files"]) == files and committed["external"]


# ============================================================================== stand overrides unset = today
def test_stand_overrides_unset_are_today(monkeypatch):
    monkeypatch.delenv("SN_WALKER_SPEC", raising=False)
    sn = _load(OMNI / "experiments/2026-10-11_stand_network/scripts/sn_dagger_spec.py", "_t4p_sn_spec")
    assert sn.WALKER_OVERRIDE is None and sn.REFS_NAME == "REFS_S1_s18"
    assert sn.S1_SHA256.startswith("968cc550")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SN_WALKER_", "KRABBY_", "P2"))}
    for script in ("2026-10-11_stand_network/scripts/sn_eval_rows.sh",):
        r = subprocess.run(["bash", str(OMNI / "experiments" / script), "--dry-run"], capture_output=True, text=True,
                           env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        assert "2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt" in r.stdout
        assert "REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24" in r.stdout
    r = subprocess.run(["bash", str(OMNI / "experiments/2026-10-11_stand_network/scripts/sn_eval_rows.sh"),
                        "--dry-run"], capture_output=True, text=True, timeout=60,
                       env={**env, "SN_WALKER_CK": "/abs/w.pt"})
    assert r.returncode == 2 and "go together" in r.stderr


# ============================================================================== driver side (run_candidates)
@pytest.fixture(scope="module")
def RC(L):
    return L.rc()


def _payload_dump(kg=(0.0, 90.0), zf=0.25, xy=0.1, z=0.309):
    d = {"events": {"payload_reset": {"func": "parkour_tasks.crab_hex_omni_task.mdp.omni_events:payload_reset",
                                      "mode": "reset", "params": {
                                          "asset_cfg": {"name": "robot", "body_names": "body", "body_ids": slice(None)},
                                          "kg_range": tuple(kg), "zero_frac": zf, "com_xy": xy, "com_z": z,
                                          "box_size": (0.7112, 1.2192, 0.3)}}}}
    return yaml.dump(d)


def test_payload_knob_parsers_and_whitelist(RC):
    kp = RC.knob_parsers()
    for k in ("KRABBY_PAYLOAD_KG", "KRABBY_PAYLOAD_ZERO_FRAC", "KRABBY_PAYLOAD_COM_XY", "KRABBY_PAYLOAD_COM_Z"):
        assert k in kp and k in RC.knob_whitelist() and k in RC.OVERLAY_DUMP_CHECKS
    assert kp["KRABBY_PAYLOAD_KG"]("0:90") == (0.0, 90.0)
    with pytest.raises(ValueError):
        kp["KRABBY_PAYLOAD_KG"]("90")


def test_payload_dump_checks_both_directions(RC):
    c = RC.OVERLAY_DUMP_CHECKS
    on = _payload_dump()
    assert "!!python/tuple" in on
    off = yaml.dump({"events": {"randomize_rigid_body_mass": {"func": "x"}}})
    assert c["KRABBY_PAYLOAD_KG"](on, "0:90") and not c["KRABBY_PAYLOAD_KG"](on, "0:180")
    assert c["KRABBY_PAYLOAD_KG"](off, "") and not c["KRABBY_PAYLOAD_KG"](on, "")
    assert not c["KRABBY_PAYLOAD_KG"](off, "0:90")
    assert not c["KRABBY_PAYLOAD_KG"](_payload_dump().replace("0.7112", "0.5"), "0:90")     # the deck box
    assert c["KRABBY_PAYLOAD_ZERO_FRAC"](on, "0.25") and not c["KRABBY_PAYLOAD_ZERO_FRAC"](on, "0")
    assert c["KRABBY_PAYLOAD_COM_XY"](on, "0.10") and not c["KRABBY_PAYLOAD_COM_XY"](on, "0")
    assert c["KRABBY_PAYLOAD_COM_Z"](on, "0.309") and not c["KRABBY_PAYLOAD_COM_Z"](off, "0.309")
    lvl = _payload_dump(kg=(90.0, 90.0), zf=0.0, xy=0.0)
    assert c["KRABBY_PAYLOAD_KG"](lvl, "90:90") and c["KRABBY_PAYLOAD_ZERO_FRAC"](lvl, "0")


def test_kg_needs_its_companions_at_compose_time(RC):
    r = []
    RC._check_payload_composition({"KRABBY_PAYLOAD_KG": "0:90"}, "w", r)
    assert r and "needs" in r[0]
    r = []
    RC._check_payload_composition({"KRABBY_PAYLOAD_COM_Z": "0.309"}, "w", r)
    assert r and "set but" in r[0]
    r = []
    RC._check_payload_composition({"KRABBY_PAYLOAD_KG": "0:90", "KRABBY_PAYLOAD_ZERO_FRAC": "0.25",
                                   "KRABBY_PAYLOAD_COM_XY": "0.10", "KRABBY_PAYLOAD_COM_Z": "0.309"}, "w", r)
    assert r == []
    cands = yaml.safe_load((PL / "preflight/candidates.yaml").read_text())
    st = cands["chains"]["PF_p2a_s18"]["stages"][0]
    st["overlay"]["KRABBY_PAYLOAD_KG"] = "0:90"
    with pytest.raises(SystemExit) as ei:
        RC.validate_candidates(cands, strict=True, check_replay=False)
    assert "KRABBY_PAYLOAD_KG='0:90' needs" in str(ei.value.code)


# ============================================================================== gate_row.sh / distill.sh
def test_gate_row_refuses_relative_paths_and_a_missing_out(tmp_path):
    sh = str(TP / "gate_row.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_"))}
    r = subprocess.run(["bash", sh, "N", "rel/ck.pt", "/a", "/b", "suite__omni", "1"], capture_output=True, text=True,
                       env={**env, "TP_OUT": str(tmp_path)}, timeout=60)
    assert r.returncode == 2 and "relative path refused" in r.stdout and "TPROW END" in r.stdout.splitlines()[-1]
    r = subprocess.run(["bash", sh, "N", "/ck.pt", "/a", "/b", "suite__omni", "1"], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 2 and "TP_OUT" in r.stdout


def test_distill_strips_every_refused_runner_knob(tmp_path):
    spec = _load(OMNI / "config/crab_hex/omni_student_spec.py", "_t4p_student_spec")
    envf = tmp_path / "t.env"
    keys = {"KRABBY_OMNI_PHASE": "2a", "KRABBY_PLANT": "A15+B", "KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "0.8",
            **{k: "1" for k in spec.STUDENT_RUNNER_KNOBS_REFUSED}}
    envf.write_text("".join(f"{k}={v}\n" for k, v in sorted(keys.items())))
    ck = tmp_path / "model_1.pt"
    ck.write_bytes(b"x")
    r = subprocess.run(["bash", str(TP / "distill.sh"), "--dry-run", "--name", "T", "--checkpoint", str(ck),
                        "--env-file", str(envf), "--seed", "19", "--iterations", "50"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    envs = [ln for ln in r.stdout.splitlines() if ln.startswith("KRABBY_")]
    assert not any(ln.split("=")[0] in spec.STUDENT_RUNNER_KNOBS_REFUSED for ln in envs)
    assert "KRABBY_STD_PARAM=soft" in envs and "KRABBY_INIT_STD=0.8" in envs
    assert "--resume --checkpoint" in r.stdout and "--max_iterations 50" in r.stdout


def test_student_rows_strip_the_refused_keys(P, tspec):
    keys = P.student_refused_keys()
    assert "KRABBY_ENTROPY_COEF" in keys and "KRABBY_STD_PARAM" not in keys
    sel = tspec.stage("sel")
    assert sel["teacher_args"] == ["--task", "Isaac-Crab-Hex-Omni-Student-Play-v0", "--policy-role", "teacher"]
    assert "--policy-role" in sel["student_args"] and "student" in sel["student_args"]


# ============================================================================== live / payload / replay / select
def _ser(n=600, first=0, ep=1500.0, cf=0.1, std=0.8):
    return {"Train/mean_episode_length": [(first + i, ep) for i in range(n)],
            "Episode_Termination/crab_failure": [(first + i, cf) for i in range(n)],
            "PPO/effective_std": [(first + i, std) for i in range(n)]}


def test_live_rules_on_synthetic_series():
    LW = _load(TP / "live_watch.py", "tp_live_watch")
    src = _ser(n=300, first=0)
    ok = LW.check("driver", _ser(first=299), source=src)
    assert ok["rc"] == LW.RC_PASS
    col = LW.check("driver", _ser(first=299, ep=500.0), source=src)
    assert col["rc"] == LW.RC_KILL and col["rule"] == "collapse"
    assert LW.check("driver", _ser(first=299, ep=500.0), source=src, accepted=("collapse",))["rc"] != LW.RC_KILL
    rs = LW.check("driver", _ser(first=299, std=0.5), source=src)
    assert rs["rule"] == "resume"
    nan = _ser(first=299)
    nan["Loss/surrogate"] = [(300, float("nan"))]
    assert LW.check("driver", nan, source=src)["rule"] == "nan"
    scratch = _ser(first=0, cf=0.7)                       # a from-scratch s1 falls a lot early: no collapse rule
    scratch["Metrics/base_parkour/track_ratio_mean"] = [(i, 0.6) for i in range(1200)]
    assert LW.check("driver", scratch, stage="s1")["rc"] != LW.RC_KILL
    scratch["Metrics/base_parkour/track_ratio_mean"] = [(i, 0.2) for i in range(1200)]
    assert LW.check("driver", scratch, stage="s1")["rule"] == "dead_walker"
    stu = {"Loss_depth/depth_actor_loss": [(i, 10.0) for i in range(300)], **_ser(n=300)}
    assert LW.check("student", stu)["rule"] == "student"
    stu = {"Loss_depth/depth_actor_loss": [(i, 10.0 if i < 5 else 2.0) for i in range(300)], **_ser(n=300, cf=0.3)}
    k = LW.check("student", stu, tstar_tail={"ep_len": 1500.0, "crab_failure": 0.05})
    assert k["rc"] == LW.RC_KILL and "crab_failure" in k["fails"][0]                 # 0.30 > 0.05 + 0.10
    assert LW.check("student", stu, tstar_tail={"ep_len": 1500.0, "crab_failure": 0.25})["rc"] == LW.RC_PASS
    assert LW.check("student", stu, tstar_tail={"ep_len": 2000.0, "crab_failure": 0.25})["rc"] == LW.RC_KILL


def test_student_loss_rule_watch_reports_and_never_kills():
    """STEPC_RULES C4.8 / V.3: ``live.student.loss_rule`` kill (default) | watch. With watch the loss rule is computed
    and reported (``watch``) but never kills; the episode-length / crab_failure / NaN rules still kill."""
    LW = _load(TP / "live_watch.py", "tp_live_watch")
    flat = {"Loss_depth/depth_actor_loss": [(i, 10.0) for i in range(300)], **_ser(n=300)}     # never halves
    tail = {"ep_len": 1500.0, "crab_failure": 0.1}
    assert LW.DEFAULTS["student"]["loss_rule"] == "kill"
    k = LW.check("student", flat, tstar_tail=tail)
    assert k["rc"] == LW.RC_KILL and "depth_actor_loss" in k["fails"][0] and k["watch"] == []
    w = LW.check("student", flat, tstar_tail=tail, cfg={"student": {"loss_rule": "watch"}})
    assert w["rc"] == LW.RC_PASS and w["fails"] == [] and "depth_actor_loss" in w["watch"][0]
    assert w["reads"][0]["reads"]["loss"]["fails"] is True and w["reads"][0]["reads"]["loss"]["loss_rule"] == "watch"
    wk = LW.check("student", flat, tstar_tail={**tail, "ep_len": 2000.0}, cfg={"student": {"loss_rule": "watch"}})
    assert wk["rc"] == LW.RC_KILL and "episode length" in wk["fails"][0] and wk["watch"]
    nan = {**flat, "Loss_depth/depth_actor_loss": [(i, float("nan") if i == 7 else 10.0) for i in range(300)]}
    assert LW.check("student", nan, cfg={"student": {"loss_rule": "watch"}})["rc"] == LW.RC_KILL
    r = LW.replay("student", flat, every=50, tstar_tail=tail, cfg={"student": {"loss_rule": "watch"}})
    assert r["killed_at"] is None and r["watch_first"]["rel"] == 250, r
    assert LW.replay("student", flat, every=50, tstar_tail=tail)["killed_at"] == 250
    with pytest.raises(ValueError):
        LW.check("student", flat, cfg={"student": {"loss_rule": "maybe"}})


def test_spec_carries_the_loss_rule_and_check_refuses_a_bad_one(P, S, tspec):
    assert S.d["live"]["student"]["loss_rule"] == "kill"
    assert P.graph_problems(tspec) == []
    tspec.d["live"]["student"]["loss_rule"] = "maybe"
    assert any("loss_rule" in x for x in P.graph_problems(tspec))
    del tspec.d["live"]["student"]["loss_rule"]                       # absent = kill (today)
    assert P.graph_problems(tspec) == []


def test_watch_student_reports_a_watch_loss_rule_and_kills_by_default(P, tspec, tmp_path, monkeypatch):
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "R18")
    ctx = P.Ctx(tspec, "R18")
    q = P.Queue(ctx)
    LWm = P.LW()
    flat = {"Loss_depth/depth_actor_loss": [(i, 10.0) for i in range(300)], **_ser(n=300)}
    monkeypatch.setattr(LWm, "read_series", lambda rd, tags=None: flat)
    monkeypatch.setattr(P, "find_student_run", lambda exp, name: tmp_path)
    monkeypatch.setattr(P, "read_unit", lambda name: None)
    monkeypatch.setattr(ctx.rt, "sleep", lambda s: None)

    class Proc:
        def __init__(self):
            self.n = 0

        def poll(self):
            self.n += 1
            return None if self.n < 3 else 0

    tspec.d["live"]["student"]["loss_rule"] = "watch"
    assert q.watch_student("st", Proc(), {"run_dir": str(tmp_path)}) is False
    rec = ctx.srec("st")
    assert rec["live"]["last_rc"] == LWm.RC_PASS and "depth_actor_loss" in rec["live_watch"]["reads"][0]
    assert not rec.get("live_killed")
    tspec.d["live"]["student"]["loss_rule"] = "kill"
    assert q.watch_student("st", Proc(), {"run_dir": str(tmp_path)}) is True
    assert "depth_actor_loss" in ctx.srec("st")["live_killed"]["fails"][0]


X08_RUN = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
X60_RUN = REPO / "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB"
FWD3A = REPO / "parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01"


@pytest.mark.skipif(not (X08_RUN.is_dir() and X60_RUN.is_dir() and FWD3A.is_dir()), reason="run dirs absent")
def test_live_rules_replay_on_recorded_runs():
    pytest.importorskip("tensorboard")
    LW = _load(TP / "live_watch.py", "tp_live_watch")
    r = LW.replay("driver", LW.read_series(X08_RUN), every=250, stage="p2a", source=LW.read_series(X60_RUN))
    assert r["killed_at"] is None, r
    r = LW.replay("student", LW.read_series(FWD3A), every=250)        # the loss rule holds on the forward 3a logs
    assert r["killed_at"] is None, r
    # PLAN left-for-implementation item 9: EVERY student rule on the forward 3a logs. The episode-length and
    # crab_failure rules need a T* tail; the forward student's T* is its teacher run (the 3a agent.yaml resumes
    # 2026-09-07_04-38-50/model_19996.pt, a 20k-head run whose last-200 means are the tail).
    teacher = REPO / "parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50"
    if not teacher.is_dir():
        pytest.skip("the forward teacher run dir is absent")
    ser_t = LW.read_series(teacher)
    tail = {"ep_len": LW.tail_mean(ser_t[LW.T_EPLEN]), "crab_failure": LW.tail_mean(ser_t[LW.T_CRAB])}
    assert tail["ep_len"] > 1000 and 0.2 < tail["crab_failure"] < 0.7, tail
    for every in (50, 250):
        r = LW.replay("student", LW.read_series(FWD3A), every=every, tstar_tail=tail)
        assert r["killed_at"] is None and r["polls"] > 10, (every, r)
    # the rules are live, not vacuous: a 2x tighter T* tail does kill the healthy forward student
    r = LW.replay("student", LW.read_series(FWD3A), every=250,
                  tstar_tail={"ep_len": 2.5 * tail["ep_len"], "crab_failure": tail["crab_failure"]})
    assert r["killed_at"] is not None


@needs_stamps
def test_payload_level_reads_on_s1_at_0kg(P):
    P.L.configure(yaml.safe_load(SPEC.read_text()))
    PR = P.PR()
    th = json.loads((PL / "thresholds_v0.json").read_text())
    inp = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    rows = inp["p3a_apriori_s1"][18]
    st = {f"suite@{es}": REPO / e["stamp"] for es, e in rows["suite"].items()}
    st.update({f"field_light@{es}": REPO / e["stamp"] for es, e in rows["field_light"].items()})
    r = PR.level_reads(st, th, trained_translate=["fwd", "side", "back", "diag"])
    assert r["stable"] and r["failed"] == [] and len(r["lines"]) == 35
    assert r["label"] == "sim plant, force side not enforced"
    assert PR.max_stable({0: True, 45: True, 90: False, 135: True}) == 45
    assert PR.max_stable({0: False}) is None
    rise = PR.tail_rising(list(range(100)), [i * 0.01 for i in range(100)])
    flat = PR.tail_rising(list(range(100)), [0.5 + (0.01 if i % 2 else -0.01) for i in range(100)])
    assert rise["rising"] is True and flat["rising"] is False
    lf = PR.limiting_factor({"failed": ["p3a.stable90.survival.fwd.es1"]}, budget={"budget_bound": True})
    assert lf["class"] == 0
    assert PR.limiting_factor({"failed": ["p3a.stable90.survival.fwd.es1"]})["class"] == 3
    assert PR.limiting_factor({"failed": ["p3a.stable90.tracking.fwd.es1"]})["class"] == 4


def test_replay_read_amended_pearly_rule():
    RR = _load(TP / "replay_read.py", "tp_replay_read")
    import numpy as np
    base = {"Loss/surrogate": {0: 1.0, 1: 2.0}, "Loss/hist_latent": {0: 1.0, 1: 1.0}, "Loss/priv_reg": {0: 0.1, 1: 0.1},
            "Perf/total_fps": {0: 100.0, 1: 90.0}, "Train/mean_reward/time": {3: 1.0}}

    def ulp(x, k):
        return float(np.frombuffer((np.array([x], np.float32).view(np.int32) + k).tobytes(), np.float32)[0])
    new = copy.deepcopy(base)
    new["Loss/hist_latent"] = {0: ulp(1.0, 3), 1: 1.0}
    new["Loss/priv_reg"] = {0: 0.1 + 1.4e-6, 1: 0.1}
    new["Perf/total_fps"] = {0: 1.0, 1: 2.0}
    r = RR.pearly_series(base, new, range(2))
    assert r["pass"] is True                                       # the Phase-A shape: PASSES under the amendment
    new["Loss/hist_latent"] = {0: ulp(1.0, 5), 1: 1.0}
    assert RR.pearly_series(base, new, range(2))["pass"] is False  # > 4 ULP
    new["Loss/hist_latent"] = {0: ulp(1.0, 2), 1: 1.0}
    new["Loss/surrogate"] = {0: 1.0, 1: 2.0000002}
    r = RR.pearly_series(base, new, range(2))
    assert r["pass"] is False and all(not d["tolerated"] for d in r["differ"] if "hist" in d["tag"])
    x = RR.x_read(Path(__file__), "0" * 64, rule="report_only")
    assert x["verdict"] == "REPORT-ONLY" and x["feeds_stop_r"] is False
    with pytest.raises(ValueError):
        RR.x_read(Path(__file__), "0" * 64, rule="PLACEHOLDER: step 8")


def test_select_student_candidates_and_two_block_halt(tmp_path):
    SS = _load(TP / "select_student.py", "tp_select_student")
    for i in list(range(22500, 25001, 100)) + list(range(25200, 27501, 200)):
        (tmp_path / f"model_{i}.pt").write_bytes(b"")
    c = SS.candidate_checkpoints(tmp_path, 22494, SS.OFFSETS)
    assert [x["iter"] for x in c] == [23500, 24500, 25400, 26400, 27400]
    st = {"ranked": ["a", "b", "c"], "blocks": {}}
    assert SS.next_action(st) == {"do": "block", "cand": "a"}
    st["blocks"]["a"] = "FAIL"
    assert SS.next_action(st) == {"do": "block", "cand": "b"}
    st["blocks"]["b"] = "FAIL"
    assert SS.next_action(st)["do"] == "halt"
    assert SS.next_action({"ranked": ["a", "b"], "blocks": {"a": "FAIL", "b": "PASS"}}) == {"do": "selected", "cand": "b"}
    r = SS.rank([{"iter": 2, "misses": 0, "field_hard": 0.7, "mean_abs_dev": 0.1},
                 {"iter": 1, "misses": 1, "field_hard": 0.9, "mean_abs_dev": 0.0},
                 {"iter": 3, "misses": 0, "field_hard": 0.8, "mean_abs_dev": 0.2}])
    assert [x["iter"] for x in r] == [3, 2, 1]
    assert SS.completion_margin(0.02, {"floor": 0.05, "se_mult": 2.0, "manifest_shape_split": "PLACEHOLDER"})[
        "margin"] is None
    m = SS.completion_margin(0.04, {"floor": 0.05, "se_mult": 2.0, "manifest_shape_split": 0.03})
    assert m["margin"] == pytest.approx(0.08) and m["binds"] == "se"


# ============================================================================== CPU rehearsal on recorded rows
@needs_stamps
def test_gate_reads_rehearsed_on_recorded_s1_rows(P, tspec, monkeypatch):
    """gate_walker (p2a) and gate_payload (p3a: floors at p3a's values, stable at 90, the paired 0-kg check -- here S1
    against itself, so every paired difference is 0) on the RECORDED S1 s18 rows. Gate 0 is bypassed: the recorded
    stamps predate the pin (their own Gate 0 is pinned by test_gate0_invalid_cases)."""
    pytest.importorskip("scipy")
    inp = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    p = inp["stages"]["p2a"]
    tspec.lock_path().write_text(json.dumps({"code_sha": "c" * 64, "eval_sha": "e" * 64}))
    P.gen(tspec, "F19")
    ctx = P.Ctx(tspec, "F19")
    ck = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
    if not ck.is_file():
        pytest.skip("X08 checkpoint absent")
    rows = {"suite@2": REPO / p["suite"][18][2]["stamp"], "side22@1": REPO / p["side22"][18]["stamp"]}
    for row in ("field_hard", "field_light", "ring_obst", "field_turn"):
        for i, e in enumerate(p["field"][row][18]):
            rows[f"{row}@{i + 1}"] = REPO / e["stamp"]
    head = {"stage": "p2a", "ckpt": str(ck), "sha256": "x" * 64, "run_dir": str(ck.parent), "iteration": 19994,
            "driver_stamp": str(REPO / p["suite"][18][1]["stamp"])}
    for sid in ("p2a", "p3a"):
        P.L.write_json_atomic(ctx.stage_read_path(sid), {**head, "stage": sid})
        recs = {k: {"stamp": str(v), "job": {"family": k.split("@")[0], "key": k}} for k, v in rows.items()}
        if sid == "p3a":
            s1 = inp["p3a_apriori_s1"][18]
            for kg in (0, 45, 90):
                for es in (1, 2):
                    for fam, src in (("suite", s1["suite"][es]), ("field_light", s1["field_light"][es])):
                        recs[f"{fam}@{es}@L{kg:03d}"] = {"stamp": str(REPO / src["stamp"]),
                                                          "job": {"family": fam, "key": f"{fam}@{es}", "level": kg}}
            ctx.srec("p3a")["curve"] = {"0": True, "45": True, "90": True, "135": False}
        P.L.write_json_atomic(ctx.rows_path(sid), recs)
    monkeypatch.setattr(P, "gate0", lambda ctx, sid, stamps, expects: [])
    # no live record: the live line is vacuous, so the gate reads WATCH with exactly that flag (CHARTER 2026-10-19)
    r = P.gate_walker(ctx, "p2a")
    assert r["verdict"] == P.GR.WATCH, r["halt_lines"]
    assert [f["id"] for f in r["watch_flags"]] == ["p2a.live_collapse"]
    # a PASS live read clears it
    ctx.srec("p2a")["live"] = {"last_rc": P.LW().RC_PASS, "last": 19994}
    r = P.gate_walker(ctx, "p2a")
    assert r["verdict"] == P.GR.PASS, (r["halt_lines"], r["watch_flags"])
    ids = {d["id"] for d in r["decisive"]}
    assert "p2a.capability.field_hard" in ids and "p2a.floor.side22_fastfast" in ids
    assert {f"p2a.floor.survival.{c}.es{e}" for c in ("fwd", "side", "back", "diag") for e in (1, 2)} <= ids
    assert json.loads(Path(r["path"]).read_text())["thresholds"]["version"] == "v1"
    r = P.gate_payload(ctx, "p3a")
    assert r["verdict"] == P.GR.PASS, (r["halt_lines"], r.get("zero_kg"))
    assert r["max_stable"] == 90 and ctx.srec("p3a")["zero_kg"]["verdict"] == "PASS"
    assert {"p3a.capability.stable_at_90", "p3a.capability.zero_kg_paired"} <= {d["id"] for d in r["decisive"]}
    ctx.srec("p3a")["curve"] = {"0": True, "45": False}
    r = P.gate_payload(ctx, "p3a")
    assert r["verdict"] == P.GR.HALT and "p3a.capability.stable_at_90" in r["halt_lines"]


def test_stand_gate_rehearsed_on_the_recorded_heads(P, tspec, tmp_path):
    """stand_gate on SN_Amask_s17 (sn_calib format) and SH_T17 (sh_read format, + a K5 sn_calib doc built from the
    thresholds' own K5 inputs): every decisive stand line PASSES, K4 reads <= 3, the relaxed G4 is recomputed from
    the clause's REFS read (REFS - max(0.10, 2 SE)), the G2 stillness lines are WATCH (not read)."""
    sn_head = OMNI / "experiments/2026-10-11_stand_network/head_SN_Amask_s17.json"
    sh_head = OMNI / "experiments/2026-10-13_stand_height/head_SH_T17.json"
    if not (sn_head.is_file() and sh_head.is_file()):
        pytest.skip("stand head records absent")
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "F19")
    ctx = P.Ctx(tspec, "F19")
    r = P.stand_gate(ctx, "sn", "D1", {"name": "SN_Amask_s17", "calib": str(sn_head), "ckpt": "x"})
    assert r["verdict"] == P.GR.PASS, r["halt_lines"]
    g4 = next(d for d in r["decisive"] if d["id"].startswith("stand.D1.G4.field_hard es1+2+3+4 completion"))
    assert g4["line"] == pytest.approx(0.8125 - 0.10) and g4["value"] == pytest.approx(0.805)
    assert not any("post_settle_steps_per_s walkstop" in d["id"] for d in r["decisive"] if d["id"].startswith(
        "stand.D1.G2"))
    inp = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    k5 = {"rows": {"SH_T17z": {"gates": {}}}}
    for g, cls in inp["stand"]["k5_sn_calib"]["gates"].items():
        for cl, c in cls.items():
            note = None
            if cl.startswith("field_hard es1+2+3+4 completion"):
                note = "n=400 vs REFS 0.8125 n=400, SE 0.0278"
            elif cl.startswith("field_turn es1 completion"):
                note = "band 0.0400 = max(binom_se(REFS 0.990, n=100), 0.04); es1 only"
            v, line = c["values"][17], c["lines"][17]
            k5["rows"]["SH_T17z"]["gates"].setdefault(g, {"clauses": {}})["clauses"][cl] = {
                "value": v, "line": line, "ok": True, "note": note}
    kp = tmp_path / "k5.json"
    kp.write_text(json.dumps(k5))
    r = P.stand_gate(ctx, "sh", "D2", {"name": "SH_T17", "read": str(sh_head), "k5_calib": str(kp), "ckpt": "x"})
    assert r["verdict"] == P.GR.PASS, r["halt_lines"]
    k4 = [d for d in r["decisive"] if d["id"].startswith("stand.D2.K4.miss")]
    assert len(k4) == 2 and all(d["line"] == 3 for d in k4)
    assert any(d["id"].startswith("stand.D2.K5.G4.field_hard") and d["line"] == pytest.approx(0.7125)
               for d in r["decisive"])
    bad = json.loads(sh_head.read_text())
    bad["gates"]["K4"]["SH_T17"]["clauses"]["miss es2 <= max(1, R_h's)"]["value"] = 4
    bp = tmp_path / "sh_bad.json"
    bp.write_text(json.dumps(bad))
    r = P.stand_gate(ctx, "sh", "D2", {"name": "SH_T17", "read": str(bp), "k5_calib": str(kp), "ckpt": "x"})
    assert r["verdict"] == P.GR.HALT and any("K4.miss es2" in x for x in r["halt_lines"])


@pytest.mark.parametrize("qrun", ["dry"])
def test_dry_pass_injections_fire_on_the_dry_run_only(P, qctx, qrun):
    ctx, rt, verdicts, done = qctx
    (ctx.dir / "INJECT.json").write_text(json.dumps({"s1": ["INVALID"], "c4": ["FAIL"]}))
    assert P.Queue(ctx).loop() == "parked"
    assert ctx.state["stages"]["s1"]["attempts"]["invalid"] == 1 and ctx.state["stages"]["s1"]["status"] == "pass"
    assert ctx.state["halts"]["c4"]["class"] == "FAIL" and "s1:INVALID" in ctx.state["injected"]
    assert P.do_continue(ctx.spec, "dry", accept="c4", relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "dry")
    ctx2.rt = rt
    assert P.Queue(ctx2).loop() == "done"                 # each injection fires once


def test_injections_are_ignored_on_an_official_run(P, qctx):
    ctx, rt, verdicts, done = qctx
    (ctx.dir / "INJECT.json").write_text(json.dumps({"s1": ["FAIL", "GUARD"]}))
    assert P.Queue(ctx).loop() == "done" and not ctx.state["halts"]


def test_tp_tools_write_no_bytecode(tmp_path):
    code = (f"import importlib.util, sys\nsys.dont_write_bytecode = True\n"
            f"s = importlib.util.spec_from_file_location('tp_pipeline', {str(TP / 'pipeline.py')!r})\n"
            "m = importlib.util.module_from_spec(s); sys.modules['tp_pipeline'] = m; s.loader.exec_module(m)\n"
            "m.LW(); m.PR(); m.RR(); m.SS(); m.L.thresholds_mod()\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr
    assert not (TP / "__pycache__").exists()


def test_status_and_report_cli_on_a_parked_run(P, qctx, tmp_path):
    ctx, rt, verdicts, done = qctx
    verdicts["c4"] = P.GR.HALT
    P.Queue(ctx).loop()
    txt = P.status_text(ctx.spec, "F19")
    assert "c4      halt" in txt and "HALT c4: FAIL open" in txt and "PARKED" in txt
    rep = _load(TP / "report.py", "tp_report")
    out = tmp_path / "REPORT"
    assert rep.main(["--spec", str(ctx.spec.path), "--run", "F19", "--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text()
    assert "pending step 11" in md and "230.06" in md and "sim plant" in md.lower()


def test_walker_spec_env_is_accepted_by_the_stand_spec(P, tspec, tmp_path):
    """The SN_WALKER_SPEC the queue composes for a stand stage beside T* parses through sn_dagger_spec's own override
    check (0 kg: the p2a overlay, the run's own candidates file and p2a chain, REFS(T*)). X stages (beside the bundled
    S1) keep the walker unset (= today) and get only P2_REANCHOR=1: REFH_S1_s18's sidecar predates M1, so without it
    run_arm.sh's code_guard refuses their rows at the pin code_sha (STEPC_RULES "TP changes" 1)."""
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "R18")
    ctx = P.Ctx(tspec, "R18")
    ck = tmp_path / "run" / "model_22494.pt"
    ck.parent.mkdir()
    ck.write_bytes(b"t")
    h = {"stage": "p3a", "ckpt": str(ck), "sha256": "ab" * 32, "driver_stamp": str(tmp_path / "stamp"),
         "iteration": 22494, "run_dir": str(ck.parent)}
    P.L.write_json_atomic(ctx.stage_read_path("p3a"), h)
    P.L.write_json_atomic(ctx.stage_read_path("tstar"), {**h, "stage": "p3a"})
    q = P.Queue(ctx)
    env = q.walker_spec_env("sn")
    sn = _load(OMNI / "experiments/2026-10-11_stand_network/scripts/sn_dagger_spec.py", "_t4p_sn_spec2")
    ov = sn.walker_spec_override({"SN_WALKER_SPEC": env["SN_WALKER_SPEC"]})
    assert ov["S1_CKPT"] == ck and ov["S1_SHA256"] == "ab" * 32 and ov["REFS_NAME"] == "REFS_T4R18"
    assert ov["OVERLAY_CHAIN"] == "R18_p2a" and ov["CANDIDATES"] == ctx.dir / "candidates.yaml"
    assert not any(k.startswith("KRABBY_PAYLOAD_") for k in ov["S1_OVERLAY"])
    cands = yaml.safe_load((ctx.dir / "candidates.yaml").read_text())
    assert cands["chains"]["R18_p2a"]["stages"][0]["overlay"] == ov["S1_OVERLAY"]
    assert env["SN_WALKER_CK"] == str(ck) and env["P2_REANCHOR"] == "1"
    for sid in ("sn_ref", "sh_ref"):
        assert tspec.stage(sid)["walker"] == "bundled_s1"
        assert q.walker_spec_env(sid) == {"P2_REANCHOR": "1"}, sid


def test_gate_row_strips_student_runner_knobs_and_merges_a_level_file(tmp_path, P):
    keys = P.student_refused_keys()
    envf = tmp_path / "row.env"
    envf.write_text("KRABBY_OMNI_CLOCK_HIDE=1\nKRABBY_STD_PARAM=soft\n" + "".join(f"{k}=1\n" for k in keys[:3]))
    lvl = tmp_path / "L090.env"
    lvl.write_text("KRABBY_PAYLOAD_KG=90:90\nKRABBY_PAYLOAD_ZERO_FRAC=0\n")
    ck, man = tmp_path / "m.pt", tmp_path / "man.yaml"
    ck.write_bytes(b"")
    man.write_text("x: 1\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_"))}
    env.update(TP_OUT=str(tmp_path / "out"), TP_ROW_DRY="1", TP_STRIP_KEYS=" ".join(keys), TP_LEVEL_FILE=str(lvl),
               KRABBY_LEAK="1")
    r = subprocess.run(["bash", str(TP / "gate_row.sh"), "N", str(ck), str(envf), str(man), "suite__omni", "1",
                        "--trained-categories", "stand,fwd"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    kv = [ln for ln in lines if ln.startswith("KRABBY_")]
    assert kv == ["KRABBY_OMNI_CLOCK_HIDE=1", "KRABBY_STD_PARAM=soft", "KRABBY_PAYLOAD_KG=90:90",
                  "KRABBY_PAYLOAD_ZERO_FRAC=0"]
    assert all(k in lines[0] for k in keys[:3])
    assert "--trained-categories stand,fwd" in lines[-2]
    envf.write_text("KRABBY_PAYLOAD_KG=0:0\n")
    r = subprocess.run(["bash", str(TP / "gate_row.sh"), "N", str(ck), str(envf), str(man), "suite__omni", "1"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2 and "already set" in r.stdout


# ============================================================================== fable-check fixes (step 6 review)
def _fake_queue_proc(ctx_dir: Path, run: str):
    """A harmless long-lived process whose cmdline looks like this run's queue (not Kit; stopped with SIGTERM)."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "pipeline.py", "_queue", "--spec",
                             "x", "--run", run])


def test_a_live_queue_refuses_a_second_run_and_continue(P, qctx):
    ctx, rt, verdicts, done = qctx
    verdicts["c4"] = P.GR.HALT
    P.Queue(ctx).loop()
    assert not (ctx.dir / "queue.pid").exists()          # the loop removes its own pid on exit
    proc = _fake_queue_proc(ctx.dir, "F19")
    try:
        (ctx.dir / "queue.pid").write_text(f"{proc.pid}\n")
        assert P.queue_pid(ctx) == proc.pid
        assert P.launch_queue(ctx.spec, "F19") == 3
        assert P.do_continue(ctx.spec, "F19", accept="c4", relaunch=False) == 3
        assert P.queue_cmdline_ok(f"python -I /x/pipeline.py _queue --spec s --run F19", ctx)
        assert not P.queue_cmdline_ok(f"python -I /x/pipeline.py _queue --spec s --run F190", ctx)
        assert not P.queue_cmdline_ok(f"python -I /x/pipeline.py status --run F19", ctx)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    assert P.queue_pid(ctx) is None                       # a dead pid is no queue
    assert P.do_continue(ctx.spec, "F19", accept="c4", relaunch=False) == 0


def test_the_loop_writes_its_own_pid(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx
    seen = []
    orig = P.Queue._loop

    def spy(self):
        seen.append((ctx.dir / "queue.pid").read_text().strip())
        return orig(self)
    monkeypatch.setattr(P.Queue, "_loop", spy)
    P.Queue(ctx).loop()
    assert seen == [str(os.getpid())]
    src = (TP / "pipeline.py").read_text()
    assert '"setsid", "nohup"' not in src and "start_new_session=True" in src


def test_notify_hook_survives_clean_env(P, monkeypatch):
    monkeypatch.setenv("TP_NOTIFY_CMD", "echo hi")
    monkeypatch.setenv("TP_OUT", "/x")
    monkeypatch.setenv("KRABBY_X", "1")
    e = P.clean_env()
    assert e["TP_NOTIFY_CMD"] == "echo hi" and "TP_OUT" not in e and "KRABBY_X" not in e


def test_student_and_tstar_halts_refuse_accept_chance(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx

    def sel_halt(self, sid):
        P.halt(self.ctx, sid, "STUDENT", "2 failed full blocks")
    monkeypatch.setattr(P.Queue, "ex_select", sel_halt)
    P.Queue(ctx).loop()
    assert P.do_continue(ctx.spec, "F19", accept="sel", relaunch=False) == 2
    assert P.do_continue(ctx.spec, "F19", abort=True, relaunch=False) == 0


def test_tstar_none_refuses_accept_chance(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx
    orig = P.Queue.ex_tstar

    def no_tstar(self, sid):
        for c in ("p3a", "p3b"):
            self.ctx.srec(c)["zero_kg"] = {"verdict": "FAIL"}
        return orig(self, sid)
    monkeypatch.setattr(P.Queue, "ex_tstar", no_tstar)
    assert P.Queue(ctx).loop() == "parked"
    assert ctx.state["halts"]["tstar"]["class"] == "FAIL"
    assert P.do_continue(ctx.spec, "F19", accept="tstar", relaunch=False) == 2


def test_head_iteration_must_match_the_plan(P, qctx):
    ctx, rt, verdicts, done = qctx
    rt.planned = lambda sid: 7 if sid == "c4" else P.Queue(ctx).planned_last_iteration(sid)
    assert P.Queue(ctx).loop() == "parked"
    assert ctx.state["halts"]["c4"]["class"] == "DRIVER"
    assert ctx.state["stages"]["s1"]["status"] == "pass"
    h = P.Ctx(ctx.spec, "F19").head("s1")
    assert h["iteration"] == 5999


def test_distill_live_accept_retrains_under_a_new_name(P, qctx):
    ctx, rt, verdicts, done = qctx
    P.halt(ctx, "st", "LIVE", "student live rule loss", lines=[{"id": "st.live.student"}])
    ctx.srec("st")["live_killed"] = {"rule": "student"}
    ctx.srec("st")["trained"] = {"rc": 0}
    ctx.save()
    assert P.do_continue(ctx.spec, "F19", accept="st", relaunch=False) == 0
    r = P.Ctx(ctx.spec, "F19").srec("st")
    assert r["status"] == "pending" and r["distill_attempt"] == 2 and "trained" not in r
    assert r["accepted_live"] == ["student"]


def test_row_reuse_requires_the_intended_checkpoint_and_per_family_timeouts(P, qctx, tmp_path):
    ctx, rt, verdicts, done = qctx
    P.gen(ctx.spec, "F19")
    ck_a, ck_b = tmp_path / "a" / "model_1.pt", tmp_path / "b" / "model_1.pt"
    for c, b in ((ck_a, b"a"), (ck_b, b"b")):
        c.parent.mkdir()
        c.write_bytes(b)
    stamp = tmp_path / "stamp"
    stamp.mkdir()
    (stamp / "tp_row.json").write_text("{}")
    (stamp / "run_meta.json").write_text(json.dumps({"checkpoint_sha256": P.L.sha256_file(ck_a)}))
    job = P.row_job(ctx, "p2a", "dirsweep_body@1", ckpt=str(ck_a), name="N")
    P.L.write_json_atomic(ctx.rows_path("p2a"), {job["key"]: {"stamp": str(stamp), "job": job}})
    assert P.run_row(ctx, "p2a", job) == stamp and not rt.rows             # the same head: reused
    job_b = P.row_job(ctx, "p2a", "dirsweep_body@1", ckpt=str(ck_b), name="N")
    calls = []
    orig = rt.run
    rt.run = lambda cmd, env=None, log=None, timeout=None: (calls.append(env), orig(cmd, env=env, log=log))[1]
    assert P.run_row(ctx, "p2a", job_b) != stamp                            # another head: rerun
    assert calls[0]["TP_ROW_TIMEOUT"] == "10800"
    P.run_row(ctx, "p2a", P.row_job(ctx, "p2a", "side22@1", ckpt=str(ck_b), name="N"))
    assert calls[-1]["TP_ROW_TIMEOUT"] == "5400"
    assert "--trained-categories" not in job_b["args"] and job_b["args"][:2] == ["--command-adapter", "body_hold"]


def test_x_read_bitwise_with_and_without_logs(tmp_path):
    RR = _load(TP / "replay_read.py", "tp_replay_read")
    ck = tmp_path / "model_4000.pt"
    ck.write_bytes(b"head")
    sha = RR.sha256_file(ck)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"it": 1, "loss": 0.5, "t_rollout_s": 3.1}\n')
    b.write_text('{"it": 1, "loss": 0.5, "t_rollout_s": 9.9}\n')            # timing differs only
    x = RR.x_read(ck, sha, rule="bitwise", new_log=a, ref_log=b)
    assert x["bitwise"] and x["curves_compared"] and x["verdict"] == "REPRODUCED-BITWISE"
    b.write_text('{"it": 1, "loss": 0.50001, "t_rollout_s": 9.9}\n')
    x = RR.x_read(ck, sha, rule="bitwise", new_log=a, ref_log=b)
    assert not x["bitwise"] and x["verdict"] == "NOT BITWISE"
    x = RR.x_read(ck, sha, rule="bitwise", new_log=a, ref_log=tmp_path / "missing.jsonl")
    assert x["bitwise"] and x["curves_compared"] is False and "NOT compared" in x["curves_note"]
    assert not RR.x_read(ck, "0" * 64, rule="bitwise")["bitwise"]


def test_spec_compare_blocks_name_the_recorded_train_logs(S):
    for sid, name in (("sn_ref", "SN_Amask_s17"), ("sh_ref", "SH_T17")):
        c = S.stage(sid)["compare"]
        assert c["name"] == name and c["train_log"].endswith(f"/{name}/train_log.jsonl")


def test_shell_patterns_equal_tp_lib(L):
    import re
    for n in ("gate_row.sh", "distill.sh"):
        src = (TP / n).read_text()
        assert re.search(r"^ISAAC_PAT='([^\n]*)'$", src, re.M).group(1) == L.ISAAC_PAT, n
        assert re.search(r"^NONKIT_PAT='([^\n]*)'$", src, re.M).group(1) == L.NONKIT_PAT, n
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    assert not L.isaac_busy(f"{py} -I /x/2026/scripts/sn_calib.py --head SN_x --json /y")


def test_check_reports_the_placeholder_cleanly(P, S, tspec):
    ph = _with_placeholders(P, tspec)
    ph.path.write_text(yaml.safe_dump(ph.d, sort_keys=False))
    P.gen(P.Spec(ph.path), "F19")
    r = subprocess.run([sys.executable, "-I", str(TP / "pipeline.py"), "check", "--spec", str(ph.path), "--run", "F19"],
                       capture_output=True, text=True, timeout=600)
    assert "invalid literal for int()" not in r.stdout + r.stderr
    assert "N_p is a PLACEHOLDER" in r.stdout


def test_the_committed_spec_is_the_step8_freeze(P, S):
    """PLAN step 8 -> v1 (stepC_read.json patch): N_p 3000, x_rule report_only, the student manifest-shape split 0.0,
    live.student.loss_rule kill, thresholds_v1.json (its version = the spec's; self-check clean)."""
    assert S.d["N_p"] == 3000 and S.d["x_rule"] == "report_only"
    assert S.stage("sel")["completion_margin"]["manifest_shape_split"] == 0.0
    assert S.d["live"]["student"]["loss_rule"] == "kill"
    assert S.d["thresholds"] == {"file": "thresholds_v1.json", "version": "v1"}
    th = json.loads(S.thresholds_path().read_text())
    assert th["version"] == "v1" and th["self_check_failures"] == []
    patch = json.loads((PL / "preflight/stepC_read.json").read_text())["patch"]["pipeline.yaml"]
    assert patch["N_p"] == S.d["N_p"] and patch["x_rule"] == S.d["x_rule"]
    assert patch["stages[sel].completion_margin.manifest_shape_split"] == 0.0
    for sid in ("p3a", "p3b", "p3c"):
        assert S.iters("R18", sid) == 3000 and S.iters("F19", sid) == 3000


# ============================================================================== bridges / freeze notice (PLAN step 7)
def _stamp_with(tmp_path, name, *, metrics, episodes=None):
    import numpy as np  # noqa: PLC0415

    d = tmp_path / name
    (d / "raw").mkdir(parents=True)
    (d / "scenario_metrics.json").write_text(json.dumps(metrics))
    for i, arrs in enumerate(episodes or []):
        np.savez(d / "raw" / f"episode_{i:04d}.npz", **arrs)
    return d


def test_bridge_read_ignores_only_the_trained_categories_key(tmp_path):
    """PLAN left-for-implementation items 3 and 10: a gate row passes --trained-categories, so its aggregate's suite block
    carries the list where the recorded stamp carries null; that ONE key is excluded from a bridge, nothing else."""
    import numpy as np  # noqa: PLC0415

    BR = _load(TP / "bridge_read.py", "tp_bridge_read")
    ep = [{"x": np.arange(5, dtype=np.float32), "y": np.array([np.nan, 1.0])}]
    m_ref = {"aggregate": {"suite": {"trained_categories": None, "n": 3}, "rate": 0.5}, "episodes": [1, 2],
             "scenario_id": "s", "run_meta": {"t": 1}}
    m_new = json.loads(json.dumps(m_ref))
    m_new["aggregate"]["suite"]["trained_categories"] = ["fwd", "side"]
    m_new["run_meta"] = {"t": 2}                                        # run_meta is never compared
    ref = _stamp_with(tmp_path, "ref", metrics=m_ref, episodes=ep)
    new = _stamp_with(tmp_path, "new", metrics=m_new, episodes=ep)
    r = BR.compare(ref, new)
    assert r["bitwise"] and r["ignored_present"] == ["aggregate.suite.trained_categories"]
    assert not BR.compare(ref, new, ignore=())["bitwise"]
    assert BR.main(["--ref", str(ref), "--new", str(new)]) == 0
    assert BR.main(["--ref", str(ref), "--new", str(new), "--no-default-ignore"]) == 1
    # any other aggregate difference, a raw value, a raw key, an episode count: not equal
    for patch in (lambda m: m["aggregate"].__setitem__("rate", 0.51),
                  lambda m: m["aggregate"]["suite"].__setitem__("n", 4),
                  lambda m: m.__setitem__("episodes", [1, 3])):
        bad = json.loads(json.dumps(m_new))
        patch(bad)
        assert not BR.compare(ref, _stamp_with(tmp_path, f"b{id(patch)}", metrics=bad, episodes=ep))["bitwise"]
    assert not BR.compare(ref, _stamp_with(tmp_path, "rawval", metrics=m_new,
                                           episodes=[{**ep[0], "x": ep[0]["x"] + 1}]))["bitwise"]
    extra = _stamp_with(tmp_path, "rawkey", metrics=m_new, episodes=[{**ep[0], "obs_cmd": np.zeros(2)}])
    assert not BR.compare(ref, extra)["bitwise"]
    assert BR.compare(ref, extra, allow_new=("obs_cmd",))["bitwise"]
    assert not BR.compare(ref, _stamp_with(tmp_path, "fewer", metrics=m_new, episodes=ep * 2))["bitwise"]
    # the stand rows (item 10): the SH launcher's stamps carry no raw/ set that matches -- aggregates only
    stand_new = _stamp_with(tmp_path, "stand_new", metrics=m_new, episodes=[])
    assert BR.compare(ref, stand_new, mode="aggregate")["bitwise"]
    bad = json.loads(json.dumps(m_new))
    bad["aggregate"]["rate"] = 0.9
    assert not BR.compare(ref, _stamp_with(tmp_path, "stand_bad", metrics=bad), mode="aggregate")["bitwise"]
    with pytest.raises(ValueError):
        BR.compare(ref, new, mode="bogus")


@needs_stamps
def test_bridge_read_aggregate_mode_on_a_recorded_stand_and_suite_stamp(tmp_path):
    BR = _load(TP / "bridge_read.py", "tp_bridge_read")
    sm = json.loads((REFS18 / "scenario_metrics.json").read_text())
    assert sm["aggregate"]["suite"]["trained_categories"] is None          # the recorded stamps predate the flag
    twin = tmp_path / "twin"
    twin.mkdir()
    sm["aggregate"]["suite"]["trained_categories"] = ["stand", "fwd"]
    (twin / "scenario_metrics.json").write_text(json.dumps(sm))
    assert BR.compare(REFS18, twin, mode="aggregate")["bitwise"]
    assert not BR.compare(REFS18, twin, mode="aggregate", ignore=())["bitwise"]


def test_lock_prints_the_freeze_notice_and_flags_a_dirty_pin(P, tspec, monkeypatch, capsys):
    """PLAN left-for-implementation item 13: the notice says "commit or stash before the pin"."""
    monkeypatch.setattr(P, "lock_doc", lambda spec: {"files": {}, "dirty_code_eval": []})
    assert P.main(["lock", "--spec", str(tspec.path)]) == 0
    out = capsys.readouterr().out
    assert "FREEZE (D14)" in out and "Commit or stash" in out and "NOT A VALID PIN" not in out
    monkeypatch.setattr(P, "lock_doc", lambda spec: {"files": {}, "dirty_code_eval": ["parkour/x/mdp/a.py"]})
    assert P.main(["lock", "--spec", str(tspec.path)]) == 0
    out = capsys.readouterr().out
    assert "NOT A VALID PIN" in out and "commit or stash" in out


def test_level_payload_com_is_checked_only_where_a_payload_exists():
    """CHARTER step 8 (C1 STOP-C, 2026-10-16): the 0:0 level row (L000) records a zero CoM for its zero-mass payload;
    the level CoM check must skip 0-kg envs, and still refuse a wrong CoM on a loaded level."""
    import importlib.util
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2] / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/"
         "tools/pipeline/gate_read.py")
    spec = importlib.util.spec_from_file_location("_t4_gate_read_l000", p)
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    base = {"kg_range": [0.0, 0.0], "zero_frac": 0.0, "com_xy": 0.0, "com_z": 0.309}
    l000 = {"payload": {**base, "payload_kg": [0.0, 0.0], "payload_com_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}}
    assert gr.payload_problems(l000, 0) == []
    l090 = {"payload": {**base, "kg_range": [90.0, 90.0], "payload_kg": [90.0, 90.0],
                        "payload_com_m": [[0.0, 0.0, 0.309], [0.0, 0.0, 0.0]]}}
    assert any("CoM" in s for s in gr.payload_problems(l090, 90))
    l090["payload"]["payload_com_m"] = [[0.0, 0.0, 0.309], [0.0, 0.0, 0.309]]
    assert gr.payload_problems(l090, 90) == []


# ============================================================================== C7 defects (CHARTER 2026-10-18)
PYI = "/home/nickmagus/krabby/isaac_venv/bin/python"


class StopRT:
    """A fake runtime for the stop procedure: the clock advances with sleep; ``procs`` = {pid: {args, alive,
    terms_to_exit}}; ``scopes`` = {unit: {pids}}; every signal is recorded."""
    poll_s = 60

    def __init__(self, t0=None):
        import time as _time
        self.t = float(t0 if t0 is not None else _time.time())
        self.procs, self.scopes, self.signals, self.scope_terms, self.logs = {}, {}, [], [], []
        self.busy, self.spawned, self.ran = [], [], []
        self.events = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)

    def isaac_busy(self):
        self.events.append("busy?")
        return self.busy.pop(0) if self.busy else False

    def ps_pid_args(self):
        return [(p, d["args"]) for p, d in self.procs.items() if d["alive"]]

    def scope_pids(self, unit):
        return set(self.scopes.get(unit, ()))

    def pid_alive(self, pid):
        return bool(self.procs.get(pid, {}).get("alive"))

    def signal(self, pid, sig):
        import signal as _sig
        self.signals.append((pid, sig, self.t))
        d = self.procs.get(pid)
        if not d or not d["alive"]:
            return
        if sig == _sig.SIGKILL:
            d["alive"] = False
        elif sig == _sig.SIGTERM:
            d["terms"] = d.get("terms", 0) + 1
            if d.get("terms_to_exit") and d["terms"] >= d["terms_to_exit"]:
                d["alive"] = False

    def term_scope(self, unit):
        self.scope_terms.append(unit)

    def unit_active(self, unit):
        return False

    def kit_logs(self):
        return list(self.logs)

    def read_head(self, p, n=8 << 20):
        return Path(p).read_text()

    def notify_hook(self, text):
        return None

    def spawn(self, cmd, *, env=None, log=None):
        self.events.append("spawn")
        self.spawned.append(cmd)
        return self.launcher

    def run(self, cmd, *, env=None, log=None, timeout=None):
        self.events.append("run")
        self.ran.append(cmd)
        return 0, ""


class Launcher:
    """The launcher bash (distill.sh / s[nh]_dagger.sh): alive while its trainer pid is."""

    def __init__(self, rt, pid):
        self.rt, self.pid, self.terminated = rt, pid, False

    def poll(self):
        return None if (not self.terminated and (self.pid is None or self.rt.pid_alive(self.pid))) else 0

    def wait(self):
        return 0

    def terminate(self):
        self.terminated = True


def _stop_rt(terms_to_exit, unit="tp-distill-T4R18_st-1", name="T4R18_st"):
    rt = StopRT()
    rt.procs[4241] = {"args": f"timeout --signal=TERM 172800 {PYI} /r/parkour/scripts/rsl_rl/train.py --run_name {name}",
                      "alive": True}
    rt.procs[4242] = {"args": f"{PYI} /r/parkour/scripts/rsl_rl/train.py --task S --headless --run_name {name}",
                      "alive": True, "terms_to_exit": terms_to_exit}
    rt.scopes[unit] = {4241, 4242}
    rt.launcher = Launcher(rt, 4242)
    return rt


def test_stop_trainer_signals_the_trainer_pid_twice_then_sigkills_as_the_last_resort(P):
    """CHARTER 2026-10-18 (C7 defect 1): SIGTERM the trainer python's OWN pid (never the scope: ``timeout`` would
    re-forward it), wait TERM_GRACE_S, SIGTERM again, wait, and only then SIGKILL."""
    import signal as _sig
    rt = _stop_rt(terms_to_exit=1)
    r = P.stop_trainer(rt, unit="tp-distill-T4R18_st-1", match=None, proc=rt.launcher)
    assert [s[:2] for s in rt.signals] == [(4242, _sig.SIGTERM)] and not r["sigkill"] and r["terms"] == 1
    assert rt.procs[4241]["alive"] and rt.scope_terms == []                  # timeout is never signalled
    rt = _stop_rt(terms_to_exit=2)
    r = P.stop_trainer(rt, unit="tp-distill-T4R18_st-1", match=None, proc=rt.launcher)
    assert [s[:2] for s in rt.signals] == [(4242, _sig.SIGTERM)] * 2 and not r["sigkill"]
    assert rt.signals[1][2] - rt.signals[0][2] >= P.TERM_GRACE_S
    rt = _stop_rt(terms_to_exit=None)                                         # the C7 hang: SIGTERM never ends it
    r = P.stop_trainer(rt, unit="tp-distill-T4R18_st-1", match=None, proc=rt.launcher)
    assert [s[:2] for s in rt.signals] == [(4242, _sig.SIGTERM), (4242, _sig.SIGTERM), (4242, _sig.SIGKILL)]
    assert rt.signals[2][2] - rt.signals[0][2] >= 2 * P.TERM_GRACE_S and r["sigkill"] and r["exited"]
    # no scope yet: the trainer is found by its command line; a launcher still before its trainer is SIGTERMed itself
    rt = _stop_rt(terms_to_exit=1)
    rt.scopes.clear()
    r = P.stop_trainer(rt, unit=None, match=lambda a: "--run_name T4R18_st" in a, proc=rt.launcher)
    assert r["pids"] == [4242] and not rt.procs[4242]["alive"]
    rt = StopRT()
    rt.launcher = Launcher(rt, None)
    r = P.stop_trainer(rt, unit=None, match=lambda a: False, proc=rt.launcher)
    assert r["pids"] == [] and r["launcher_terminated"] and rt.signals == []


def _student_ctx(P, tspec, tmp_path, monkeypatch, rt, *, ser=None):
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "R18")
    ctx = P.Ctx(tspec, "R18")
    ctx.rt = rt
    LWm = P.LW()
    healthy = {"Loss_depth/depth_actor_loss": [(i, 10.0 if i < 5 else 1.0) for i in range(300)], **_ser(n=300)}
    monkeypatch.setattr(LWm, "read_series", lambda rd, tags=None: ser if ser is not None else healthy)
    rd = tmp_path / "2026-10-18_00-00-00_T4R18_st"
    rd.mkdir()
    ev = rd / "events.out.tfevents.1"
    ev.write_text("x")
    os.utime(ev, (rt.t, rt.t))
    monkeypatch.setattr(P, "find_student_run", lambda exp, name: rd)
    monkeypatch.setattr(P, "read_unit", lambda name: "tp-distill-T4R18_st-1")
    monkeypatch.setattr(P, "ddc_gb", lambda: 39.0)
    return ctx, rd


def test_a_student_live_kill_stops_the_trainer_pid_and_records_a_sigkill(P, tspec, tmp_path, monkeypatch):
    import signal as _sig
    rt = _stop_rt(terms_to_exit=None)
    flat = {"Loss_depth/depth_actor_loss": [(i, 10.0) for i in range(300)], **_ser(n=300)}     # the loss rule kills
    ctx, rd = _student_ctx(P, tspec, tmp_path, monkeypatch, rt, ser=flat)
    q = P.Queue(ctx)
    assert q.watch_student("st", rt.launcher, {"run_dir": str(tmp_path)}) is True
    rec = ctx.srec("st")
    assert [s[:2] for s in rt.signals] == [(4242, _sig.SIGTERM), (4242, _sig.SIGTERM), (4242, _sig.SIGKILL)]
    assert rt.scope_terms == [] and rt.procs[4241]["alive"]                     # never the scope / timeout
    assert rec["stops"][0]["sigkill"] and rec["sigkill"]["pids"] == [4242] and rec["fresh_run_on_retry"]
    assert ctx.state["ddc_watch"]["stage"] == "st" and ctx.state["ddc_watch"]["ddc_gb_at_kill"] == 39.0
    assert any("SIGKILLed" in ln for ln in _sentinels(ctx, "NOTIFY"))
    assert ctx.state["halts"]["st"]["class"] == "LIVE"


def test_the_progress_watchdog_stops_a_silent_distill_and_halts_infra(P, tspec, tmp_path, monkeypatch):
    rt = _stop_rt(terms_to_exit=1)
    ctx, rd = _student_ctx(P, tspec, tmp_path, monkeypatch, rt)          # healthy series, tfevents never move
    assert P.Queue(ctx).watch_student("st", rt.launcher, {"run_dir": str(tmp_path)}) is True
    h = json.loads((P.L.REPO / ctx.state["halts"]["st"]["record"]).read_text())
    assert h["class"] == "INFRA" and "no training progress" in h["why"] and "watchdog 30 min" in h["why"]
    assert 30 * 60 < rt.t - os.stat(rd / "events.out.tfevents.1").st_mtime <= 35 * 60
    assert not ctx.srec("st").get("sigkill") and "ddc_watch" not in ctx.state
    # the retry trains afresh (_a2): the stopped run's records stay unambiguous
    assert P.do_continue(ctx.spec, "R18", retry="st", relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "R18")
    assert ctx2.srec("st")["distill_attempt"] == 2 and ctx2.srec("st")["status"] == "pending"
    assert "fresh_run_on_retry" not in ctx2.srec("st")


def test_progress_keeps_the_watchdog_quiet(P, tspec, tmp_path, monkeypatch):
    rt = _stop_rt(terms_to_exit=1)
    ctx, rd = _student_ctx(P, tspec, tmp_path, monkeypatch, rt)
    q = P.Queue(ctx)
    for _ in range(60):                                                 # an hour of polls, a new tfevents write each
        rt.sleep(60)
        os.utime(rd / "events.out.tfevents.1", (rt.t, rt.t))
        assert q.stall_s(rt.t - 3600, P.student_progress_files(rd)) is None
    assert q.stall_s(rt.t - 3600, []) == pytest.approx(3600)              # nothing ever written: from the launch


def test_dagger_watchdog_and_injected_kill_stop_by_pid(P, tspec, tmp_path, monkeypatch):
    import signal as _sig
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "R18")
    ctx = P.Ctx(tspec, "R18")
    runs = tmp_path / "runs"
    name = "SN_T4R18_sn_s17"
    rt = StopRT()
    out = runs / name
    rt.procs[7001] = {"args": f"timeout --signal=TERM 14400 {PYI} /s/sn_dagger.py --headless --out {out}", "alive": True}
    rt.procs[7002] = {"args": f"{PYI} /s/sn_dagger.py --headless --mode dagger --out {out}", "alive": True,
                      "terms_to_exit": 1}
    rt.scopes["sn-dagger-x-1"] = {7001, 7002}
    rt.launcher = Launcher(rt, 7002)
    ctx.rt = rt
    monkeypatch.setattr(P, "ddc_gb", lambda: 39.0)
    out.mkdir(parents=True)
    (out / "train_log.jsonl").write_text('{"it": 1}\n')
    os.utime(out / "train_log.jsonl", (rt.t, rt.t))
    (runs / f"{name}.unit").write_text("sn-dagger-x-1\n")
    q = P.Queue(ctx)
    assert q.train_dagger("sn", tmp_path / "sn_dagger.sh", tmp_path / "sn_dagger.py", runs, name,
                          ["--mode", "dagger", "--iterations", "4000"], {}) is None
    assert rt.events[:2] == ["busy?", "spawn"]                               # the idle check right before the launch
    assert [s[:2] for s in rt.signals] == [(7002, _sig.SIGTERM)] and rt.procs[7001]["alive"]
    h = ctx.state["halts"]["sn"]
    assert h["class"] == "INFRA" and ctx.srec("sn")["fresh_run_on_retry"]
    assert P.do_continue(ctx.spec, "R18", retry="sn", relaunch=False) == 0
    ctx2 = P.Ctx(ctx.spec, "R18")
    assert ctx2.srec("sn")["dagger_attempt"] == 2 and P.Queue(ctx2).dagger_name(name, "sn") == name + "_a2"


def _kit_log(d: Path, start: float, text: str) -> Path:
    import time as _time
    p = d / f"kit_{_time.strftime('%Y%m%d_%H%M%S', _time.localtime(start))}.log"
    p.write_text(text)
    return p


def test_after_a_sigkill_the_cache_is_checked_before_the_next_kit_launch(P, tspec, tmp_path, monkeypatch):
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "R18")
    ctx = P.Ctx(tspec, "R18")
    rt = StopRT()
    ctx.rt = rt
    gb = {"v": 39.0}
    monkeypatch.setattr(P, "ddc_gb", lambda: gb["v"])
    P.pre_launch(ctx, "c4", "driver")                                         # no watch: nothing to check
    ctx.state["ddc_watch"] = {"stage": "st", "at": "t", "at_epoch": rt.t, "ddc_gb_at_kill": 39.0}
    _kit_log(tmp_path, rt.t - 600, "Found 9 deleted or corrupted blocks, rewriting index\n")   # BEFORE the kill
    rt.logs = sorted(tmp_path.glob("kit_*.log"))
    P.pre_launch(ctx, "c4", "driver")                                         # an older log is not "the next"
    bad = _kit_log(tmp_path, rt.t + 5, "[omni.datastore] Found 9 deleted or corrupted blocks, rewriting index\n")
    rt.logs = [bad]
    with pytest.raises(P.Halted):
        P.pre_launch(ctx, "c4", "driver")
    h = json.loads((P.L.REPO / ctx.state["halts"]["c4"]["record"]).read_text())
    assert h["class"] == "GUARD" and "quarantine" in h["why"] and h["options"] == ["retry"]
    # the user quarantines the cache and retries: the watch re-arms on the NEXT Kit start
    assert P.do_continue(ctx.spec, "R18", retry="c4", relaunch=False) == 0
    ctx = P.Ctx(ctx.spec, "R18")
    ctx.rt = rt
    w = ctx.state["ddc_watch"]
    assert "symptom" not in w and w["previous_symptom"] and w["at_epoch"] > rt.t - 1e6
    w["at_epoch"] = rt.t
    rt.t += 1
    rt.logs = [_kit_log(tmp_path, rt.t + 1, "app ready\n")]
    P.pre_launch(ctx, "c4", "driver")                                         # clean but young: the watch stays open
    assert not ctx.state["ddc_watch"].get("closed_at")
    rt.t += P.DDC_LOG_SETTLE_S + 5
    P.pre_launch(ctx, "c4", "driver")
    assert ctx.state["ddc_watch"]["closed_at"]                                # one clean Kit start closes it
    ctx.state["ddc_watch"] = {"stage": "st", "at": "t", "at_epoch": rt.t, "ddc_gb_at_kill": 39.0}
    gb["v"] = 39.0 + P.DDC_GROWTH_GB + 1
    with pytest.raises(P.Halted):
        P.pre_launch(ctx, "c4", "driver")                                     # the cache grew since the kill


def test_one_isaac_at_a_time_waits_with_no_give_up_before_every_launch(P, qctx, monkeypatch):
    """C7 defect 3: the queue's own full-pattern wait (every sn_ / sh_ tool included) runs immediately before EVERY
    Isaac launch, never gives up, and a STOP request stops the queue without launching."""
    ctx, rt, verdicts, done = qctx
    ev, busy = [], [True] * 2000                                               # an hour+ of another branch's Kit

    def isaac_busy():
        ev.append("check")
        return busy.pop(0) if busy else False

    real_launch, real_run = rt.launch_driver, rt.run
    monkeypatch.setattr(rt, "isaac_busy", isaac_busy)
    monkeypatch.setattr(rt, "launch_driver", lambda *a: (ev.append("launch"), real_launch(*a))[1])
    monkeypatch.setattr(rt, "run", lambda *a, **k: (ev.append("launch"), real_run(*a, **k))[1])
    assert P.Queue(ctx).loop() == "done"
    assert ev.count("launch") > 10 and ev[0] == "check" and ev.count("check") >= 2000
    for i, e in enumerate(ev):
        if e == "launch":
            assert ev[i - 1] == "check", i                                     # immediately before every launch
    assert len(_sentinels(ctx, "WAIT")) == 1


def test_a_stop_request_while_waiting_launches_nothing(P, qctx, monkeypatch):
    ctx, rt, verdicts, done = qctx
    n = {"k": 0}

    def isaac_busy():
        n["k"] += 1
        if n["k"] == 3:
            (ctx.dir / "STOP_PIPELINE").write_text("")
        return True

    monkeypatch.setattr(rt, "isaac_busy", isaac_busy)
    assert P.Queue(ctx).loop() == "stopped"
    assert rt.launched == [] and ctx.state["status"] == "stopped" and ctx.srec("s1")["status"] == "training"


def test_kit_lines_count_a_driver_and_its_trainer_as_one(L):
    """The C7 sampler's '2 Isaac processes' at 16:27:09 was the s1 driver (a CPU supervisor) + its own train.py (Kit
    log kit_20261017_162700): isaac_lines (the busy check) counts both, conservatively; kit_lines counts Kits."""
    ps = "\n".join([f"{PYI} /x/tools/run_candidates.py --unit cands-dry-1 --campaign-dir /d",
                    f"{PYI} /x/parkour/scripts/rsl_rl/train.py --task T --headless",
                    f"timeout --signal=TERM 172800 {PYI} /x/parkour/scripts/rsl_rl/train.py --task S",
                    f"{PYI} -I /x/tools/pipeline/pipeline.py _queue --spec s --run R18"])
    assert len(L.isaac_lines(ps)) == 2 and len(L.kit_lines(ps)) == 1
    assert len(L.kit_lines(ps + f"\n{PYI} /s/sh_dagger.py --headless --out /r/SH_x")) == 2


def test_stand_stages_expect_the_checkpoint_of_their_launched_iterations(P, tspec, tmp_path, monkeypatch):
    """C7 defect 2: sn_ref / sh_ref (and sn / sh) judge model_<launched iterations>.pt -- the dry pass's 20-it
    stand-in writes model_20.pt, not model_4000.pt."""
    assert P.Queue(P.Ctx(tspec, "R18")).judged_name("sn_ref") == "model_4000.pt"
    assert P.Queue(P.Ctx(tspec, "R18")).judged_name("sh") == "model_5000.pt"
    assert P.Queue(P.Ctx(tspec, "dry")).judged_name("sn_ref") == "model_20.pt"
    assert P.Queue(P.Ctx(tspec, "dry")).judged_name("sh_ref") == "model_20.pt"
    assert P.graph_problems(tspec) == []
    tspec.stage("sn")["judged"] = "model_3500.pt"
    assert any("judged model_3500.pt" in x for x in P.graph_problems(tspec))
    # X runs "as D1 / D2 but beside the bundled S1": it takes the row lists it does not name from its D stage
    cfg = P.stand_stage_cfg(tspec, "sh_ref")
    assert "height_rows" not in tspec.stage("sh_ref") and cfg["height_rows"] == tspec.stage("sh")["height_rows"]
    assert cfg["k5_rows"] == tspec.stage("sh")["k5_rows"] and cfg["controls_spec"] == tspec.stage("sh")["controls_spec"]
    assert P.stand_stage_cfg(tspec, "sn_ref")["stand_rows"] == tspec.stage("sn")["stand_rows"]
    rows = tspec.stage("sn").pop("stand_rows")
    assert {x.split(":")[0] for x in P.graph_problems(tspec) if "stand_rows" in x} >= {"stage sn", "stage sn_ref"}
    tspec.stage("sn")["stand_rows"] = rows
    # the X stage on the dry run: a 20-it DAgger run that wrote model_20.pt proceeds (no DRIVER HALT)
    tspec.stage("sn")["judged"] = "model_4000.pt"
    tspec.lock_path().write_text(json.dumps({"code_sha": "c", "eval_sha": "e"}))
    P.gen(tspec, "dry")
    ctx = P.Ctx(tspec, "dry")
    ctx.rt = StopRT()
    monkeypatch.setattr(P, "guard_problems", lambda *a, **k: [])
    monkeypatch.setattr(P, "ddc_gb", lambda: 0.0)
    runs = tmp_path / "sn_runs"
    paths = {"sn_runs": runs, "sn_scripts": tmp_path / "sn_scripts", "p2_run_arm": tmp_path / "run_arm.sh"}
    monkeypatch.setattr(P.Queue, "stand_paths", lambda self: paths)
    trained = []

    def fake_train(self, sid, launcher, trainer, runs_, name, args, env):
        it = int(args[args.index("--iterations") + 1])
        (runs_ / name).mkdir(parents=True, exist_ok=True)
        (runs_ / name / f"model_{it}.pt").write_bytes(b"x")
        trained.append((name, it))
        return 0

    monkeypatch.setattr(P.Queue, "train_dagger", fake_train)
    monkeypatch.setattr(P, "stand_gate", lambda ctx, sid, which, rec, x=False: {"verdict": "PASS", "x": {"verdict": "report_only"}})
    q = P.Queue(ctx)
    q.ex_dagger_sn("sn_ref")
    assert trained == [("SN_T4dry_sn_ref_s17", 20)]
    assert ctx.srec("sn_ref")["status"] == "pass" and "sn_ref" not in ctx.state["halts"]
    assert ctx.head("sn_ref")["ckpt"].endswith("SN_T4dry_sn_ref_s17/model_20.pt")
    assert ctx.rt.events.count("busy?") >= 1 and ctx.rt.events.index("busy?") < ctx.rt.events.index("run")
    # and sh_ref: trained, table, height + K5 rows (its D stage's lists), read -- no KeyError, no DRIVER HALT
    tab = tmp_path / "sh_table"
    (tab / "T1").mkdir(parents=True)
    (tab / "LATEST").write_text("T1\n")
    paths.update({"sh_runs": tmp_path / "sh_runs", "sh_scripts": tmp_path / "sh_scripts", "sh_table_out": tab})
    ctx.rt.events.clear()
    q.ex_dagger_sh("sh_ref")
    assert trained[-1] == ("SH_T4dry_sh_ref_s17", 20) and ctx.srec("sh_ref")["status"] == "pass"
    assert ctx.head("sh_ref")["ckpt"].endswith("SH_T4dry_sh_ref_s17/model_20.pt")
    ran = [" ".join(map(str, c)) for c in ctx.rt.ran]
    assert any("sh_eval_rows.sh" in c and "hwalkstop__omni@1" in c for c in ran), ran
    ev = ctx.rt.events                       # table, height rows, K5 rows (Isaac) each right after the idle check;
    runs = [i for i, e in enumerate(ev) if e == "run"]          # then sh_read + sn_calib (CPU reads)
    assert len(runs) == 5 and all(ev[i - 1] == "busy?" for i in runs[:3]) and ev[runs[3] - 1] == "run"


def test_the_spec_carries_the_watchdog_and_the_grace(P, S, tspec):
    assert S.d["live"]["progress_watchdog_min"] == 30 and S.d["live"]["term_grace_s"] == 120
    tspec.d["live"]["progress_watchdog_min"] = 0
    assert any("progress_watchdog_min" in x for x in P.graph_problems(tspec))


# ============================================================================== the driver's run dir (R18 live-watch fix)
#: R18's REAL records (2026-10-19): the driver's ``state.json`` current while p2a TRAINED (the eval that followed keeps
#: started / resume / target_final / phase and swaps kind / tag / iters / log -- R18_EVAL_CURRENT is that record
#: verbatim but for extra_env), and the two train.py lines of ``R18/logs/seed18_X06_p2a_A15pB_train.log`` (lines
#: 56-57; the root line's path is re-pointed at the fake LOG_ROOT). The run dir train.py made from them is
#: ``crab_hex_omni_teacher/2026-10-19_12-25-59_seed18_X06_p2a_A15pB``: the "Exact experiment name" line carries the
#: timestamp ONLY, so the old ``driver_run_dir`` built ``.../2026-10-19_12-25-59`` and returned None on every poll.
R18_TS = "2026-10-19_12-25-59"
R18_TAG = "seed18_X06_p2a_A15pB"
R18_RESUME = ("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
              "2026-10-19_10-17-50_seed18_X05_c6x_A15pB/model_14995.pt")
R18_EVAL_CURRENT = {"tag": "seed18_X06_p2a_A15pB_suite__omni", "phase": "2a", "iters": 1000, "kind": "eval",
                    "log": "{logs}/seed18_X06_p2a_A15pB_suite__omni_eval.log", "resume": R18_RESUME,
                    "target_final": 19994, "started": "2026-10-19T12:25:48"}
R18_TRAIN_LOG_LINES = ("[INFO] Logging experiment in directory: {root}",
                       "Exact experiment name requested from command line: 2026-10-19_12-25-59")


class LiveRT(FakeRT):
    """The driver unit stays active for ``polls`` polls (``on_poll(n)`` runs at each); a stopped unit is inactive."""

    def __init__(self, ctx_dir, pin, tmp, polls=0, on_poll=None):
        super().__init__(ctx_dir, pin, tmp)
        self.polls, self.on_poll, self.n = polls, on_poll, 0

    def unit_active(self, unit):
        if unit in self.killed or self.n >= self.polls:
            return False
        self.n += 1
        if self.on_poll:
            self.on_poll(self.n)
        return True


def _drv_ctx(P, tspec, tmp_path, monkeypatch, run, polls=0, on_poll=None):
    pin = {"code_sha": "c" * 64, "eval_sha": "e" * 64}
    tspec.lock_path().write_text(json.dumps(pin))
    P.gen(tspec, run)
    ctx = P.Ctx(tspec, run)
    ctx.rt = LiveRT(ctx.dir, pin, tmp_path, polls, on_poll)
    monkeypatch.setattr(P.L, "LOG_ROOT", tmp_path / "rsl_rl")
    monkeypatch.setattr(P, "ddc_watch_problem", lambda ctx: None)
    return ctx


def _drv_window(P, ctx, sid, *, xid, tag, ts, root_line=True, current=None):
    """The driver's records of one training window of stage ``sid`` (state.json ladder + current, the train log)."""
    chain, seed = ctx.spec.chain(ctx.run, sid), ctx.spec.seed(ctx.run, sid)
    exp = P.L.rc().ph.PHASES[ctx.spec.stage(sid)["base"]].experiment
    root = P.L.LOG_ROOT / exp
    logs = ctx.dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{tag}_train.log"
    lines = [ln.format(root=root) for ln in R18_TRAIN_LOG_LINES[0 if root_line else 1:]]
    log.write_text("[KNOB] KRABBY_ENTROPY_COEF=0.0025 consumed\n" + "\n".join(lines).replace(R18_TS, ts) + "\n")
    cur = current or {"tag": tag, "phase": ctx.spec.stage(sid)["base"], "iters": 5000, "kind": "rl", "log": str(log),
                      "resume": R18_RESUME, "target_final": 19994, "started": "2026-10-19T12:25:48"}
    st = {"current": cur, "ladder": {"experiments": {
        xid: {"xid": xid, "kind": "reference", "chain": chain, "seed": str(seed), "stages": [sid]},
        "X99": {"xid": "X99", "kind": "reference", "chain": "other_chain", "seed": str(seed), "stages": ["c6x"]}}},
        "keys": {f"seed{seed}": {}}}
    (ctx.dir / "state.json").write_text(json.dumps(st))
    return root, log, st


def test_driver_run_dir_resolves_the_r18_suffixed_run_dir(P, tspec, tmp_path, monkeypatch):
    ctx = _drv_ctx(P, tspec, tmp_path, monkeypatch, "R18")
    root, log, st = _drv_window(P, ctx, "p2a", xid="X06", tag=R18_TAG, ts=R18_TS)
    assert root.name == "crab_hex_omni_teacher"                       # the experiment R18's train log names
    assert P.driver_run_dir(ctx, "p2a") is None                       # train.py printed, the runner has no dir yet
    assert ctx.srec("p2a")["live_resolve"]["state"] == "pending"
    real = root / f"{R18_TS}_{R18_TAG}"
    for d in (real, root / f"2026-10-19_01-35-18_{R18_TAG}",           # the dry pass's run of the same name
              root.parent / "crab_hex_omni_flat_walk" / f"{R18_TS}_{R18_TAG}"):
        d.mkdir(parents=True)
    assert P.driver_run_dir(ctx, "p2a") == real
    assert not (root / R18_TS).exists()                               # what the old lookup built
    assert ctx.srec("p2a")["live_resolve"]["run_dir"] == str(real)
    # without the root line the stage's phase experiment is the root: same answer
    _drv_window(P, ctx, "p2a", xid="X06", tag=R18_TAG, ts=R18_TS, root_line=False)
    assert P.driver_run_dir(ctx, "p2a") == real
    # the driver's EVAL of the window it just trained (R18's real current now): the dir resolved in training holds
    ok = ctx.srec("p2a")["live_resolve"]
    ev = dict(R18_EVAL_CURRENT, log=R18_EVAL_CURRENT["log"].format(logs=ctx.dir / "logs"))
    st["current"] = ev
    (ctx.dir / "state.json").write_text(json.dumps(st))
    assert P.driver_run_dir(ctx, "p2a") == real
    assert P.driver_window(ctx, "p2a", None)["state"] == "pending"    # no training-time resolution: never guessed
    assert P.driver_window(ctx, "p2a", {**ok, "window": {**ok["window"], "started": "x"}})["state"] == "pending"
    # once the driver records the window, its checkpoint must lie in the dir
    rec = {"phase": "p2a", "tag": R18_TAG, "log": str(log), "ckpt": str(real / "model_19994.pt")}
    st["current"] = {"tag": R18_TAG, "phase": "2a", "iters": 5000, "kind": "rl", "log": str(log), "resume": R18_RESUME,
                     "target_final": 19994, "started": "2026-10-19T12:25:48"}
    st["keys"]["seed18"] = {"X06:p2a": rec}
    (ctx.dir / "state.json").write_text(json.dumps(st))
    assert P.driver_run_dir(ctx, "p2a") == real
    rec["ckpt"] = str(root / f"2026-10-19_01-35-18_{R18_TAG}" / "model_19994.pt")
    (ctx.dir / "state.json").write_text(json.dumps(st))
    assert P.driver_window(ctx, "p2a")["state"] == "mismatch" and P.driver_run_dir(ctx, "p2a") is None


def test_driver_run_dir_never_guesses(P, tspec, tmp_path, monkeypatch):
    ctx = _drv_ctx(P, tspec, tmp_path, monkeypatch, "R18")
    root, log, st = _drv_window(P, ctx, "p2a", xid="X06", tag=R18_TAG, ts=R18_TS)
    # a stale current of ANOTHER stage's window (e.g. the previous driver's) is not this stage's
    assert P.driver_window(ctx, "c6x")["state"] == "pending"
    # same second, another run name only: a mismatch, said and notified once (not per poll), never read
    (root / f"{R18_TS}_seed18_X07_p3a_A15pB").mkdir(parents=True)
    assert P.driver_run_dir(ctx, "p2a") is None and P.driver_run_dir(ctx, "p2a") is None
    notes = [json.loads(x) for x in ctx.notify_path.read_text().splitlines()]
    assert len([n for n in notes if "mismatch" in n["text"]]) == 1
    # the infra retry (log / tag <tag>_r2) keeps the run_name: both dirs present = ambiguous
    tag2 = f"{R18_TAG}_r2"
    _drv_window(P, ctx, "p2a", xid="X06", tag=tag2, ts="2026-10-19_12-40-00")
    (root / f"2026-10-19_12-40-00_{R18_TAG}").mkdir()
    assert P.driver_run_dir(ctx, "p2a") == root / f"2026-10-19_12-40-00_{R18_TAG}"
    (root / f"2026-10-19_12-40-00_{tag2}").mkdir()
    w = P.driver_window(ctx, "p2a")
    assert w["state"] == "ambiguous" and w["run_dir"] is None and P.driver_run_dir(ctx, "p2a") is None
    # a log whose train.py lines name another experiment root
    _drv_window(P, ctx, "p2a", xid="X06", tag=R18_TAG, ts=R18_TS)
    log.write_text(log.read_text().replace("crab_hex_omni_teacher", "crab_hex_omni_flat_walk"))
    assert P.driver_window(ctx, "p2a")["state"] == "mismatch"


def _fake_series(upto: int, crab: float = 0.2) -> dict:
    steps = range(0, upto + 1)
    return {"Loss/surrogate": [(s, 0.01) for s in steps], "Train/mean_episode_length": [(s, 500.0) for s in steps],
            "Episode_Termination/crab_failure": [(s, crab) for s in steps],
            "Metrics/base_parkour/track_ratio_mean": [(s, 0.8) for s in steps]}


@pytest.mark.parametrize("p_live_pass", [False, True])
def test_watch_driver_reads_the_live_rules_and_p_live_at_plus_at(P, tspec, tmp_path, monkeypatch, p_live_pass):
    upto = {"n": 0}
    reads, pearly_calls, checks = [], [], []

    def on_poll(n):
        upto["n"] = 100 * n

    ctx = _drv_ctx(P, tspec, tmp_path, monkeypatch, "R18", polls=4, on_poll=on_poll)
    root, log, st = _drv_window(P, ctx, "s1", xid="X01", tag="seed18_X01_s1_A15pB", ts="2026-10-19_05-12-46")
    rd = root / "2026-10-19_05-12-46_seed18_X01_s1_A15pB"
    rd.mkdir(parents=True)
    lw, rr = P.LW(), P.RR()
    real_check = lw.check
    monkeypatch.setattr(lw, "read_series", lambda run_dir, tags=None: (reads.append(Path(run_dir)),
                                                                        _fake_series(upto["n"]))[1])
    monkeypatch.setattr(lw, "check", lambda *a, **k: (checks.append(k.get("stage")), real_check(*a, **k))[1])

    def pearly(new, old, steps=range(20)):
        pearly_calls.append((Path(new), Path(old), list(steps)))
        return {"pass": p_live_pass, "design_bitwise": "55/144 bitwise", "differ": [] if p_live_pass else
                [{"tag": "Loss/entropy", "first_step": 20, "tolerated": False}]}
    monkeypatch.setattr(rr, "pearly", pearly)
    at = int(tspec.d["live"]["p_live"]["at"])
    assert at == 200 and tspec.run("R18")["p_live"]
    killed = P.Queue(ctx).watch_driver("s1", "cands-R18-1")
    rec = ctx.srec("s1")
    assert reads and set(reads) == {rd} and checks and set(checks) == {"s1"}      # the live check ran on the run
    assert rec["live"]["run_dir"] == str(rd) and rec["live"]["last_rc"] in (P.LW().RC_PASS, P.LW().RC_WAIT)
    ref = P.L.REPO / tspec.d["live"]["p_live"]["reference_run"]
    assert pearly_calls == [(rd, ref, list(range(0, at + 1)))]                 # once, at +at, never before
    assert rec["p_live_done"] is True and rec["p_live"]["pass"] is p_live_pass
    if p_live_pass:
        assert killed is False and rec["live"]["last"] == 400 and "s1" not in ctx.state["halts"]
        assert ctx.rt.killed == []
    else:
        assert killed is True and rec["live"]["last"] == at
        assert rec["live_killed"]["rule"] == "p_live" and ctx.rt.killed == ["cands-R18-1"]
        assert ctx.state["halts"]["s1"]["class"] == "STOP-R"


def test_dry_pass_driver_live_injection_is_reachable(P, tspec, tmp_path, monkeypatch):
    qrun = "dry"
    made = {}

    def on_poll(n):
        if n == 2:                                     # the runner creates the dir after the first poll
            made["rd"].mkdir(parents=True)

    ctx = _drv_ctx(P, tspec, tmp_path, monkeypatch, qrun, polls=5, on_poll=on_poll)
    seed = ctx.spec.seed(qrun, "c4")
    root, log, st = _drv_window(P, ctx, "c4", xid="X02", tag=f"seed{seed}_X02_c4_A15pB", ts="2026-10-19_00-55-24")
    made["rd"] = root / f"2026-10-19_00-55-24_seed{seed}_X02_c4_A15pB"
    monkeypatch.setattr(P.LW(), "read_series", lambda run_dir, tags=None: _fake_series(300))
    (ctx.dir / "INJECT.json").write_text(json.dumps({"c4": ["LIVE"]}))
    assert P.Queue(ctx).watch_driver("c4", "cands-dry-1") is True
    rec = ctx.srec("c4")
    assert ctx.rt.n == 2                                # poll 1: no dir yet (no injection); poll 2: resolved -> kill
    assert rec["live"]["run_dir"] == str(made["rd"]) and rec["live_killed"]["rule"] == "injected"
    assert ctx.state["halts"]["c4"]["class"] == "LIVE" and "c4:LIVE" in ctx.state["injected"]
    assert ctx.rt.killed == ["cands-dry-1"]


def test_driver_stamp_takes_the_drivers_recorded_eval(P, tspec, tmp_path, monkeypatch):
    """A continued window (``<tag>_c``) is evaluated under its ORIGINAL tag: the run-dir-derived name misses it."""
    ctx = _drv_ctx(P, tspec, tmp_path, monkeypatch, "R18")
    monkeypatch.setattr(P.L, "EVAL_ROOT", tmp_path / "gait_eval")
    ck = tmp_path / "rsl_rl/crab_hex_omni_teacher" / f"{R18_TS}_{R18_TAG}_c" / "model_19994.pt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")
    stamp = tmp_path / "gait_eval" / ctx.dir.name / R18_TAG / "suite__omni" / "seed001" / "2026-10-19_18-30-02"
    stamp.mkdir(parents=True)
    (stamp / "run_meta.json").write_text("{}")
    head = {"xid": "X06", "seed": "18", "ckpt": str(ck), "stage_ckpts": {"p2a": str(ck)}, "chain": "R18_p2a"}
    st = {"current": {}, "ladder": {"experiments": {"X06": {"chain": "R18_p2a", "seed": "18", "stages": ["p2a"]}}},
          "keys": {"seed18": {"X06:p2a": {"phase": "p2a", "tag": f"{R18_TAG}_c", "ckpt": str(ck),
                                          "evals": {"suite": {"run_dir": str(stamp)}}}}}}
    (ctx.dir / "state.json").write_text(json.dumps(st))
    assert P.driver_stamp(ctx, head) == stamp
    assert P.driver_stamp(ctx, head, row="side22_25lr__omni") is None
    st["keys"]["seed18"]["X06:p2a"]["evals"] = {}                    # no record: the name-derived fallback
    (ctx.dir / "state.json").write_text(json.dumps(st))
    assert P.driver_stamp(ctx, head) is None
