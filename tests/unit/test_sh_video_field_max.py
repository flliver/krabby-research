"""Stand-height campaign (``2026-10-13_stand_height``): the video's ``--terrain field_max`` on CPU (no Isaac / Kit /
GPU; VIDEO ONLY, read by no gate). Added 2026-10-24 for the user's "record a similar video on the roughest terrain
trained on" (the F19 deliverable: walker T* = p3b + the stand-height net SH_T4F19_sh_s19, after
``sh_mix_F19_s1514_rough``).

Pinned here:

* ``sh_video_core.terrain_env(ref, "field_max")`` = the reference env + ``field_hard__omni``'s env block READ from
  ``scenarios_p2.yaml`` with ``KRABBY_FLAT_TERRAIN_DIFF`` pinned ``0.70:1.0`` -> ``1.0:1.0`` (the pin narrows the
  row's band to its top, refused if the row's top is not the trained 1.0); the D5 / L10 refusals pass on it; it is the
  F19 T* training field (``F19/envs/p3b.train.env``) at its top, differing only in the band, the flat share and the
  curriculum switch;
* ``flat`` / ``field_medium`` are BYTE-IDENTICAL to the pre-field_max code: ``terrain_env`` / ``setup_caption_text``
  outputs equal the commit-36c80bc2 core's, every other pre-existing core function's source is unchanged, the
  launcher's dry-run output equals the commit-055e9f51 launcher's, and (on this machine's F19 files) the recorder's
  real pre-Kit path (a stub ``isaaclab.app``) prints and boots exactly what the commit-36c80bc2 recorder source does
  (run beside today's core, whose flat / field_medium outputs the first check pins to the old core's);
* the generator accepts ``lo == hi``: the REAL ``ParkourTerrainGenerator`` generation loops (extracted with ``ast``
  from ``parkour_terrain_generator.py`` and run on a stand-in generator) build every tile at exactly 1.0 under the
  random path and both curriculum paths; the trained band 0.20:1.0 under the curriculum builds 1.0 on the top row
  only (so field_max's tiles are the trained top row's difficulty); ``0.70:1.0`` unpinned builds none at 1.0;
* ``sh_video_core.tile_difficulty_probe`` records the difficulty each tile was built at, passes the generator's
  arguments / results through untouched and restores the class; ``field_difficulty_reasons`` refuses a missing,
  unprobed, off-grid or off-1.0 tile and a band other than [1.0, 1.0]; ``visited_difficulty`` / ``difficulty_summary``;
* the recorder's field_max wiring (source level): the band checked on the cfg before the build, the env built inside
  the probe (exactly one generator), the realised generator refused unless every tile is 1.0, the difficulty recorded
  per path / visited tile and per hold -- every piece behind ``FIELD_PIN`` (None on flat / field_medium);
* the launch: plan mix seed 1514 fits the field grid (2 x 2 tiles from spawn rows 0..8, every one at 1.0 under
  field_max): the rough video's seed, no reseed; ``sh_record_video.sh --terrain field_max`` reaches the recorder.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
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
CORE_PATH = SCRIPTS / "sh_video_core.py"
REC_PATH = SCRIPTS / "sh_record_video.py"
SH_PATH = SCRIPTS / "sh_record_video.sh"
SCRIPTS_REL = "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-13_stand_height/scripts"
#: the last commits that touched the core / recorder (36c80bc2) and the launcher (055e9f51) before field_max
PRE_CORE_COMMIT = "36c80bc2"
PRE_SH_COMMIT = "055e9f51"
P2_MANIFEST = TASK / "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml"
OMNI_FIELD_CFG = REPO / "parkour/parkour_isaaclab/terrains/omni/config/omni_field.py"
GEN_SRC = REPO / "parkour/parkour_isaaclab/terrains/parkour_terrain_generator.py"
ENV_CFG_SRC = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config/crab_hex/crab_hex_env_cfg.py"
P3B_TRAIN_ENV = TASK / "experiments/2026-10-16_task4_pipeline/F19/envs/p3b.train.env"
T_STAR = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt"
F19_SH = REPO / "parkour/outputs/omni_plan/stand_height/runs/SH_T4F19_sh_s19/model_5000.pt"
F19_STAMPS = REPO / "parkour/outputs/omni_plan/phase2/lr/SH_T4F19_sh_s19/hraise__omni/seed001"
LAUNCH_SEED = 1514
DIFF = "KRABBY_FLAT_TERRAIN_DIFF"


def _pyc_snapshot() -> dict:
    out = {}
    for d in (SCRIPTS, TASK / "experiments/2026-10-11_stand_network/scripts",
              TASK / "experiments/2026-10-07_omni_phase2/scripts"):
        for p in sorted(d.glob("__pycache__/*.pyc")):
            out[str(p)] = p.stat().st_mtime_ns
    return out


_PYC_BEFORE = _pyc_snapshot()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def _git_show(commit: str, rel: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git show {commit}:{rel} unavailable: {r.stderr.strip()}")
    return r.stdout


@pytest.fixture(scope="module")
def GEN():
    return _load(GEN_PATH, "_t_shvfm_gen")


@pytest.fixture(scope="module")
def VC(GEN):
    return GEN.VC


@pytest.fixture(scope="module")
def S(GEN):
    return GEN.schedule_module()


@pytest.fixture(scope="module")
def HDR(GEN):
    return GEN.reference_header()


@pytest.fixture(scope="module")
def OLD_VC():
    """The core as of :data:`PRE_CORE_COMMIT` (before field_max), executed with the real file's ``__file__`` (its
    by-path loads resolve to today's unchanged sh_eval_common / sh_dagger_spec / sn_video_core)."""
    src = _git_show(PRE_CORE_COMMIT, f"{SCRIPTS_REL}/sh_video_core.py")
    mod = types.ModuleType("_t_shvfm_core_pre")
    mod.__file__ = str(CORE_PATH)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        exec(compile(src, str(CORE_PATH), "exec"), mod.__dict__)  # noqa: S102 - the repo's own source
    finally:
        sys.dont_write_bytecode = old
    mod._source = src
    return mod


@pytest.fixture(scope="module")
def TV():
    """The base video test module (its fake bin for the launcher), reused."""
    return _load(Path(__file__).resolve().parent / "test_sh_video.py", "_t_shvfm_base")


def _field_grid(VC) -> dict:
    tree = ast.parse(OMNI_FIELD_CFG.read_text())
    (call,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "")
               == "ParkourTerrainGeneratorCfg"]
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg in ("size", "num_rows", "num_cols",
                                                                               "border_width")}
    return VC.field_grid_of(kw)


def _f19_stamp():
    if not (T_STAR.is_file() and F19_SH.is_file() and F19_STAMPS.is_dir()):
        pytest.skip("the F19 walker / stand net / SH hraise stamp is not on this machine")
    stamps = sorted(d for d in F19_STAMPS.iterdir() if (d / "p2_complete.json").is_file())
    if not stamps:
        pytest.skip("no complete F19 SH hraise stamp")
    return stamps[-1]


def _ref_envs() -> list[dict]:
    refs = [{"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_OMNI_STAND_OBS": "0", "KRABBY_PLANT": "A15+B"},
            {"KRABBY_X": "1", "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2"}]
    if F19_STAMPS.is_dir():
        for d in sorted(F19_STAMPS.iterdir()):
            if (d / "run_meta.json").is_file():
                refs.append({k: str(v) for k, v in json.loads((d / "run_meta.json").read_text())["env_vars"].items()})
    return refs


# ===================================================================================== the env block
def test_field_max_is_field_hard_with_the_band_pinned_to_its_top(VC):
    ref = {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_OMNI_STAND_OBS": "0", "KRABBY_PLANT": "A15+B"}
    assert VC.TERRAINS == ("flat", "field_medium", "field_max") and VC.DEFAULT_TERRAIN == "flat"
    env, rec = VC.terrain_env(ref, "field_max")
    raw = {r["id"]: r for r in yaml.safe_load(P2_MANIFEST.read_text())["scenarios"]}
    row = {k: str(v) for k, v in raw["field_hard__omni"]["env"].items()}               # read, not retyped
    assert row[DIFF] == "0.70:1.0" and row["KRABBY_FLAT_TERRAIN_FLAT_FRAC"] == "0.0"
    want = dict(row, **{DIFF: "1.0:1.0"})
    assert rec["env_block"] == want and env == {**ref, **want} and rec["row_env_block"] == row
    assert rec["terrain"] == "field_max" and rec["source"]["rows"] == ["field_hard__omni"]
    assert rec["source"]["manifest_sha256"] == VC.EC.sha256_file(P2_MANIFEST)
    assert rec["difficulty_pin"] == {"var": DIFF, "row": "0.70:1.0", "video": "1.0:1.0", "row_band": [0.7, 1.0],
                                     "want": 1.0}
    assert rec["deviation"][DIFF] == {"reference": None, "video": "1.0:1.0"}
    assert rec["deviation"]["KRABBY_FLAT_TERRAIN_GEOM"] == {"reference": "ring", "video": "field"}
    assert "KRABBY_PLANT" not in rec["deviation"] and len(rec["deviation"]) == 8
    # the medium field's block differs from field_max's in the band only
    assert {k: v for k, v in VC.field_env_block()["env"].items() if k != DIFF} == \
        {k: v for k, v in want.items() if k != DIFF}
    assert VC.setup_caption_text(rec, "schedule", "tile 7,0 omni_field_dense") == \
        "TERRAIN field_max (field DIFF 1.0:1.0, wall edges) tile 7,0 omni_field_dense   |   stand rule: schedule"
    for ref_env in _ref_envs()[2:]:                                                     # the F19 stamp's env
        e, _ = VC.terrain_env(ref_env, "field_max")
        assert VC.SN.env_reasons(e) == [] and VC.SN.env_reasons(ref_env) == []
    with pytest.raises(ValueError):
        VC.terrain_env(ref, "field_hard")


def test_parse_difficulty_band_matches_the_cfg_parse_and_refuses_junk(VC):
    assert VC.parse_difficulty_band("1.0:1.0") == (1.0, 1.0) and VC.parse_difficulty_band("0.70:1.0") == (0.7, 1.0)
    assert VC.parse_difficulty_band(VC.FIELD_MAX_DIFF) == (VC.FIELD_MAX_DIFFICULTY, VC.FIELD_MAX_DIFFICULTY)
    for bad in ("1.0", "a:b", "0.8:0.7", "1.0:1.1", "nan:1", "-0.1:0.5", "0:1:2"):
        with pytest.raises(ValueError):
            VC.parse_difficulty_band(bad)
    # the env cfg's own parse (no validation): float of each ':' part, straight into the generator's difficulty_range
    src = ENV_CFG_SRC.read_text()
    assert 'float(x) for x in os.environ.get("KRABBY_FLAT_TERRAIN_DIFF", "0.05:0.2").split(":")' in src
    assert "_apply_crab_hex_easy_mixed_terrain(tg, flat_proportion=_frac, difficulty_range=(_lo, _hi))" in src
    assert "    tg.difficulty_range = difficulty_range\n" in src


def test_field_max_refuses_a_row_whose_band_does_not_top_out_at_the_trained_top(VC, tmp_path):
    data = yaml.safe_load(P2_MANIFEST.read_text())
    for r in data["scenarios"]:
        if r.get("id") == "field_hard__omni":
            r["env"][DIFF] = "0.70:0.90"
    bad = tmp_path / "p2.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="does not top out"):
        VC.field_max_env_block(bad)
    for r in data["scenarios"]:
        if r.get("id") == "field_hard__omni":
            del r["env"][DIFF]
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="carries no"):
        VC.field_max_env_block(bad)


def test_field_max_is_the_trained_field_at_its_top(VC):
    """F19 T* p3b trained on the field (curriculum on, DIFF 0.20:1.0, half flat tiles); field_max = that field's knobs
    with every tile at the band's top: only the band, the flat share and the curriculum switch differ."""
    if not P3B_TRAIN_ENV.is_file():
        pytest.skip("the F19 p3b training env is not on this machine")
    train = {}
    for ln in P3B_TRAIN_ENV.read_text().splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            train[k.strip()] = v.strip()
    terrain = {k: v for k, v in train.items() if re.match(r"KRABBY_(FIELD_|FLAT_TERRAIN_|TERRAIN_)", k)}
    blk = VC.terrain_env({}, "field_max")[1]["env_block"]
    assert VC.parse_difficulty_band(terrain[DIFF]) == (0.2, VC.FIELD_MAX_DIFFICULTY)          # the trained top
    same = {k for k in terrain if blk.get(k) == terrain[k]}
    assert same == {"KRABBY_FIELD_BLOCKS", "KRABBY_FIELD_EDGES", "KRABBY_FIELD_KINDS", "KRABBY_FIELD_PAD_R",
                    "KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_FLAT_TERRAIN_MODE"}
    assert set(terrain) - same == {DIFF, "KRABBY_FLAT_TERRAIN_FLAT_FRAC", "KRABBY_FLAT_TERRAIN_CURRICULUM",
                                   "KRABBY_TERRAIN_PROMOTE"}
    assert set(blk) - set(terrain) == set()                                          # nothing the training lacked
    assert blk["KRABBY_FLAT_TERRAIN_FLAT_FRAC"] == "0.0" and "KRABBY_FLAT_TERRAIN_CURRICULUM" not in blk


# ===================================================================================== flat / field_medium unchanged
def test_flat_and_field_medium_outputs_equal_the_pre_field_max_core(VC, OLD_VC):
    assert OLD_VC.TERRAINS == VC.TERRAINS[:2] and OLD_VC.DEFAULT_TERRAIN == VC.DEFAULT_TERRAIN
    for ref in _ref_envs():
        for t in ("flat", "field_medium"):
            a, b = OLD_VC.terrain_env(ref, t), VC.terrain_env(ref, t)
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (t, ref)
            assert "difficulty_pin" not in b[1] and "row_env_block" not in b[1]          # FIELD_PIN is None there
            for tile in (None, "tile 3,0 omni_field_dense"):
                for rule in ("schedule", "slot"):
                    assert OLD_VC.setup_caption_text(b[1], rule, tile) == VC.setup_caption_text(b[1], rule, tile)


def test_every_other_pre_existing_core_function_is_unchanged(VC, OLD_VC):
    old_src, new_src = OLD_VC._source, CORE_PATH.read_text()
    old_t, new_t = ast.parse(old_src), ast.parse(new_src)
    defs = lambda t, s: {n.name: ast.get_source_segment(s, n) for n in t.body  # noqa: E731
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old_d, new_d = defs(old_t, old_src), defs(new_t, new_src)
    assert set(old_d) <= set(new_d)
    changed = sorted(k for k in old_d if old_d[k] != new_d[k])
    assert changed == ["terrain_env"], changed
    added = sorted(set(new_d) - set(old_d))
    assert added == sorted(["parse_difficulty_band", "field_max_env_block", "tile_difficulty_probe",
                            "difficulty_band_reasons", "field_difficulty_reasons", "difficulty_summary",
                            "visited_difficulty"]), added


def _dry(bash_args, env, script_text=None):
    if script_text is None:
        cmd = ["bash", str(SH_PATH), *bash_args]
    else:            # the old launcher's text with $0 = the real path (its HERE resolves to the real scripts dir)
        cmd = ["bash", "-c", script_text, str(SH_PATH), *bash_args]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    norm = re.sub(r"(sh-video-[A-Za-z0-9_-]+?)-\d+(\.scope)?", r"\1-<ts>\2", r.stdout)
    return r.returncode, norm, r.stderr


def test_launcher_dry_run_is_byte_identical_off_field_max(TV, tmp_path):
    old_sh = _git_show(PRE_SH_COMMIT, f"{SCRIPTS_REL}/sh_record_video.sh")
    env, marker = TV._fake_bin(tmp_path, "/usr/bin/bash idle")
    head = ["--", "--checkpoint", str(T_STAR), "--stand-ckpt", "x.pt", "--stand-mask", "1", "--deadband", "0.25"]
    runs = [["--dry-run", "--plan", "mix", "--seed", str(LAUNCH_SEED), "--total-s", "180", "--stand-rule", "schedule",
             "--terrain", t, "--prefix", "sh_mix_F19_s1514_rough"] + head for t in ("field_medium", "flat")]
    runs += [["--dry-run", "--plan", "mix", "--seed", str(LAUNCH_SEED)] + head, ["--dry-run", "--smoke"] + head,
             ["--dry-run", "--plan", "tour", "--terrain", "field_medium", "--table-start", "0.635"] + head]
    for args in runs:
        a, b = _dry(args, env, old_sh), _dry(args, env)
        assert a == b, (args, a, b)
    assert not marker.exists() and not (tmp_path / "out").exists()


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

#: runs a recorder SOURCE (today's or a commit's) as ``__main__`` with the real recorder's ``__file__``
_RUN_SRC = '''
import sys
src_path, real = sys.argv[1], sys.argv[2]
sys.argv = [real] + sys.argv[3:]
g = {"__name__": "__main__", "__file__": real}
exec(compile(open(src_path).read(), real, "exec"), g)
'''


def _pre_kit(tmp_path, src_path: Path, args) -> tuple[int, str, str]:
    stub = tmp_path / "stub"
    (stub / "isaaclab" / "app").mkdir(parents=True, exist_ok=True)
    (stub / "isaaclab" / "__init__.py").write_text("")
    (stub / "isaaclab" / "app" / "__init__.py").write_text(_STUB_APP)
    env = {k: v for k, v in os.environ.items() if not k.startswith("KRABBY_")}
    env.update(PYTHONPATH=str(stub), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", _RUN_SRC, str(src_path), str(REC_PATH), "--headless",
                        "--enable_cameras", "--out-dir", str(tmp_path / "out"), *args], env=env, capture_output=True,
                       text=True, timeout=600, cwd=str(tmp_path))
    return r.returncode, r.stdout, r.stderr


def _boot_env(stdout: str) -> dict:
    (line,) = [ln for ln in stdout.splitlines() if ln.startswith("STUB_KIT_BOOT ")]
    return json.loads(line[len("STUB_KIT_BOOT "):])


def test_recorder_pre_kit_flat_and_field_medium_equal_the_pre_field_max_recorder_and_field_max_boots(GEN, VC,
                                                                                                      tmp_path):
    """The REAL pre-Kit path (stub Kit) on the mix seed-1514 row with the F19 head: flat / field_medium print and boot
    exactly what the commit-36c80bc2 recorder source does (it loads today's core by its ``__file__``; the core's own
    flat / field_medium outputs are pinned to the old core's above); field_max boots the field_max env (reference
    checks all equal)."""
    stamp = _f19_stamp()
    old_rec = tmp_path / "old_rec.py"
    old_rec.write_text(_git_show(PRE_CORE_COMMIT, f"{SCRIPTS_REL}/sh_record_video.py"))
    vids = tmp_path / "v"
    assert GEN.main(["--plan", "mix", "--seed", str(LAUNCH_SEED), "--out-dir", str(vids)]) == 0
    base = ["--manifest", str(vids / "scenarios_sh_video_mix.yaml"), "--sequence", "sh_mix__omni",
            "--checkpoint", str(T_STAR), "--stand-ckpt", str(F19_SH), "--stand-mask", "1", "--deadband", "0.25",
            "--eval-run", str(F19_STAMPS), "--stand-rule", "schedule", "--prefix", "x"]
    ref_env = {k: str(v) for k, v in json.loads((stamp / "run_meta.json").read_text())["env_vars"].items()}
    for t in ("flat", "field_medium"):
        a = _pre_kit(tmp_path, old_rec, base + ["--terrain", t])
        b = _pre_kit(tmp_path, REC_PATH, base + ["--terrain", t])
        assert a[0] == b[0] == 42, (t, a[0], b[0], b[2][-2000:])
        assert a[1] == b[1], t                                                      # every pre-Kit line, verbatim
        assert _boot_env(b[1]) == VC.terrain_env(ref_env, t)[0]
    rc, out, err = _pre_kit(tmp_path, REC_PATH, base + ["--terrain", "field_max"])
    assert rc == 42, (rc, err[-3000:])
    assert _boot_env(out) == VC.terrain_env(ref_env, "field_max")[0]
    assert _boot_env(out)[DIFF] == "1.0:1.0" and "[SHV] reference checks: all equal" in out
    assert "pinned to 1.0:1.0: every tile at difficulty 1.0" in out


# ===================================================================================== the generator accepts lo == hi
def _real_generation_loops():
    """``ParkourTerrainGenerator._generate_random_terrains`` / ``_generate_curriculum_terrains``, executed from the
    generator's OWN source (``ast``-extracted; the module imports Kit at the top, so it cannot be imported here)."""
    tree = ast.parse(GEN_SRC.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParkourTerrainGenerator"]
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef)
           and n.name in ("_generate_random_terrains", "_generate_curriculum_terrains")]
    assert len(fns) == 2
    src = GEN_SRC.read_text()
    for f in fns:                   # the probe's premise: mesh(difficulty, cfg), then add(mesh, origin, row, col, ..)
        body = ast.get_source_segment(src, f)
        i = body.index("self._get_terrain_mesh(difficulty, sub_terrains_cfg)")
        j = body.index("self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal)")
        assert i < j and body.count("self._get_terrain_mesh(") == 1 and body.count("self._add_sub_terrain(") == 1
    ns = {"np": np}
    code = compile(ast.fix_missing_locations(ast.Module(body=fns, type_ignores=[])), str(GEN_SRC), "exec")
    exec(code, ns)  # noqa: S102 - the repo's own generator source
    return ns["_generate_random_terrains"], ns["_generate_curriculum_terrains"]


class _Sub:
    def __init__(self, p: float):
        self.proportion = p


def _stand_in_generator_cls():
    rand, curr = _real_generation_loops()

    class StandInGenerator:
        """The generator's state the two loops touch; the mesh build replaced by a sentinel (the probe's subject is
        the difficulty, which the loops compute before the build)."""

        _generate_random_terrains = rand
        _generate_curriculum_terrains = curr

        def __init__(self, lo, hi, *, curriculum, random_difficulty=False, rows=10, cols=20, seed=0,
                     props=(0.0, 0.5, 0.5)):
            self.cfg = types.SimpleNamespace(
                num_rows=rows, num_cols=cols, difficulty_range=(lo, hi), curriculum=curriculum,
                random_difficulty=random_difficulty,
                sub_terrains={"parkour_flat": _Sub(props[0]), "omni_field_sparse": _Sub(props[1]),
                              "omni_field_dense": _Sub(props[2])})
            self.np_rng = np.random.default_rng(seed)
            self.terrain_type = np.zeros((rows, cols))
            self.terrain_names = np.zeros((rows, cols, 1)).astype(str)
            self.goal_heights = np.zeros((rows, cols, 2))
            self.x_edge_maskes = np.zeros((rows, cols, 2, 2))
            self.height_fields = np.zeros((rows, cols, 2, 2))
            self.seen, self.added = [], []
            (self._generate_curriculum_terrains if curriculum else self._generate_random_terrains)()

        def _get_terrain_mesh(self, difficulty, cfg):
            self.seen.append(float(difficulty))
            return (("mesh", len(self.seen)), np.zeros(3), np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)),
                    np.zeros((2, 2)), None)

        def _add_sub_terrain(self, mesh, origin, row, col, goal):
            self.added.append((mesh, int(row), int(col)))

        def _store_block_mask(self, row, col, mask):
            pass

    return StandInGenerator


def test_the_real_generator_loops_build_every_tile_at_1_under_the_pinned_band(VC):
    G = _stand_in_generator_cls()
    grid = {"size": [10.0, 10.0], "num_rows": 10, "num_cols": 20, "border_width": 20.0}
    orig = {n: G.__dict__[n] for n in VC.PROBED_METHODS}
    for kw in ({"curriculum": False, "seed": 0}, {"curriculum": False, "seed": 7}, {"curriculum": True},
               {"curriculum": True, "random_difficulty": True, "seed": 3}):
        with VC.tile_difficulty_probe(G) as built:
            g = G(1.0, 1.0, **kw)
        assert built == [g] and {n: G.__dict__[n] for n in VC.PROBED_METHODS} == orig           # restored
        td = getattr(g, VC.TILE_DIFFICULTY_ATTR)
        assert len(td) == 200 and set(td.values()) == {1.0} and all(v == 1.0 for v in g.seen), kw
        assert VC.field_difficulty_reasons(td, grid, band=g.cfg.difficulty_range) == [], kw
        # paired in call order, and the original's result reached the original add untouched
        assert [m for m, _r, _c in g.added] == [("mesh", i + 1) for i in range(200)]
        assert [td[VC.tile_key((r, c))] for _m, r, c in g.added] == g.seen
        assert VC.difficulty_summary(td) == {"want": 1.0, "tiles": 200, "min": 1.0, "max": 1.0, "tiles_at_want": 200,
                                             "all_at_want": True}
    # the same loops unprobed: no attribute, the same tiles (the probe changes nothing it records)
    a, b = G(1.0, 1.0, curriculum=False, seed=5), None
    with VC.tile_difficulty_probe(G):
        b = G(1.0, 1.0, curriculum=False, seed=5)
    assert not hasattr(a, VC.TILE_DIFFICULTY_ATTR) and a.seen == b.seen
    assert (a.terrain_names == b.terrain_names).all() and (a.terrain_type == b.terrain_type).all()
    assert VC.field_difficulty_reasons(getattr(a, VC.TILE_DIFFICULTY_ATTR, None), grid) == [
        "the difficulty probe recorded no tile (the generator was not built inside it)"]


def test_the_trained_band_tops_out_at_1_on_the_top_row_only_and_field_hard_never_reaches_it(VC):
    """F19 T* p3b's field: DIFF 0.20:1.0 with the curriculum on (random_difficulty off: the cfg default) -> row r at
    0.2 + 0.8 r / 9, exactly 1.0 on row 9 only; field_max's every tile = that top row's difficulty. field_hard__omni
    unpinned (0.70:1.0, random) builds no tile at 1.0 -- why the pin."""
    G = _stand_in_generator_cls()
    grid = {"size": [10.0, 10.0], "num_rows": 10, "num_cols": 20, "border_width": 20.0}
    assert "random_difficulty: bool = False" in (REPO / "parkour/parkour_isaaclab/terrains/"
                                                 "parkour_terrain_generator_cfg.py").read_text()
    with VC.tile_difficulty_probe(G):
        g = G(0.2, 1.0, curriculum=True, props=(0.5, 0.25, 0.25))
    td = getattr(g, VC.TILE_DIFFICULTY_ATTR)
    for r in range(10):
        for c in range(20):
            assert td[VC.tile_key((r, c))] == pytest.approx(0.2 + 0.8 * r / 9, abs=1e-12)
            assert (td[VC.tile_key((r, c))] == VC.FIELD_MAX_DIFFICULTY) == (r == 9), (r, c)
    why = VC.field_difficulty_reasons(td, grid, band=g.cfg.difficulty_range)
    assert len(why) == 2 and "[0.2, 1.0]" in why[0] and "180 tile(s)" in why[1]
    with VC.tile_difficulty_probe(G):
        h = G(0.7, 1.0, curriculum=False, seed=1)
    hd = getattr(h, VC.TILE_DIFFICULTY_ATTR)
    assert all(0.7 <= v < 1.0 for v in hd.values()) and VC.field_difficulty_reasons(hd, grid)


def test_probe_restores_an_inherited_method_and_marks_an_unpaired_tile(VC):
    G = _stand_in_generator_cls()

    class Sub(G):
        pass

    with VC.tile_difficulty_probe(Sub) as built:
        assert "_get_terrain_mesh" in Sub.__dict__
        g = Sub(1.0, 1.0, curriculum=False)
        g._add_sub_terrain(("m",), np.zeros(3), 0, 0, None)               # an add with no difficulty pending
    assert built == [g] and all(n not in Sub.__dict__ for n in VC.PROBED_METHODS)
    assert getattr(g, VC.TILE_DIFFICULTY_ATTR)["0,0"] is None
    grid = {"size": [10.0, 10.0], "num_rows": 10, "num_cols": 20, "border_width": 20.0}
    assert "1 tile(s) were built below / off difficulty 1.0: 0,0=None" in \
        VC.field_difficulty_reasons(getattr(g, VC.TILE_DIFFICULTY_ATTR), grid)


def test_difficulty_reasons_and_summaries(VC):
    grid = {"size": [10.0, 10.0], "num_rows": 2, "num_cols": 3, "border_width": 20.0}
    full = {VC.tile_key((r, c)): 1.0 for r in range(2) for c in range(3)}
    assert VC.field_difficulty_reasons(full, grid, band=(1.0, 1.0)) == []
    assert VC.field_difficulty_reasons(full, grid, band=[1, 1]) == []
    assert "difficulty_range is [0.7, 1.0]" in VC.field_difficulty_reasons(full, grid, band=(0.7, 1.0))[0]
    assert "unreadable" in VC.field_difficulty_reasons(full, grid, band="x")[0]
    miss = dict(full)
    del miss["1,2"]
    assert VC.field_difficulty_reasons(miss, grid) == ["1 grid tile(s) carry no recorded difficulty: ['1,2']"]
    assert "outside the grid" in VC.field_difficulty_reasons(dict(full, **{"2,0": 1.0}), grid)[0]
    low = dict(full, **{"0,1": 0.9999999})
    assert VC.field_difficulty_reasons(low, grid) == ["1 tile(s) were built below / off difficulty 1.0: 0,1=0.9999999"]
    assert VC.difficulty_summary(low) == {"want": 1.0, "tiles": 6, "min": 0.9999999, "max": 1.0, "tiles_at_want": 5,
                                          "all_at_want": False}
    assert VC.difficulty_summary({})["all_at_want"] is False
    visits = VC.tile_visits([-5.0, -5.0, 5.0, 50.0], [-10.0, -10.0, -10.0, 0.0], grid)
    assert visits == {"tiles": {"0,0": 2, "1,0": 1}, "border_steps": 1, "steps": 4}
    vd = VC.visited_difficulty(visits, full)
    assert vd == {"want": 1.0, "per_tile": {"0,0": 1.0, "1,0": 1.0}, "steps": 4, "steps_at_want": 3,
                  "steps_off_want": 1, "border_steps": 1, "all_at_want": False}
    on = VC.tile_visits([-5.0, 5.0], [-10.0, -10.0], grid)
    assert VC.visited_difficulty(on, full)["all_at_want"] is True
    assert VC.visited_difficulty(on, low)["all_at_want"] is True                    # 0,1 not visited
    assert VC.visited_difficulty(VC.tile_visits([-5.0], [0.0], grid), low)["all_at_want"] is False


# ===================================================================================== the recorder's wiring
def test_recorder_field_max_wiring_is_behind_the_pin():
    src = REC_PATH.read_text()
    lines = src.splitlines()
    pre_kit = src[: src.index("app_launcher = AppLauncher(args_cli)")]
    assert 'FIELD_PIN = TERRAIN.get("difficulty_pin")' in pre_kit
    assert pre_kit.index("RUN_ENV, TERRAIN = VC.terrain_env(WANT_ENV, args_cli.terrain)") < \
        pre_kit.index('FIELD_PIN = TERRAIN.get("difficulty_pin")')
    # the env is built inside the probe on field_max only (a no-op context elsewhere), and one generator recorded
    (make,) = [k for k, ln in enumerate(lines) if 'raw_env = gym.make(scenario.task, cfg=env_cfg' in ln]
    assert lines[make - 1].strip() == "with probe as probed_gens:"
    assert lines[make - 3].strip() == "probe = (VC.tile_difficulty_probe(_terrain_generator_cls()) if FIELD_PIN is " \
                                      "not None"
    assert lines[make - 2].strip() == "else contextlib.nullcontext([]))"
    assert lines[make + 1].strip() == "if FIELD_PIN is not None and len(probed_gens) != 1:"
    assert src.count("VC.tile_difficulty_probe(") == 1
    assert "from parkour_isaaclab.terrains.parkour_terrain_generator import ParkourTerrainGenerator" in src
    # the cfg band before the build, the realised generator after it (refused), the read-outs -- each behind the pin
    for guard, needle, prefix in (
            ("if FIELD_PIN is not None:", "_why = VC.difficulty_band_reasons(getattr(tg, \"difficulty_range\", None), "
                                          "FIELD_PIN[\"want\"],", True),
            ("if FIELD_PIN is not None:", "tdiff = getattr(gen, VC.TILE_DIFFICULTY_ATTR, None)", False),
            ("if tdiff is not None:", 'rec["difficulty"] = tdiff.get(VC.tile_key((int(r), int(c))))', False),
            ('if ctx.field["realised"].get("difficulty") is not None:',
             "tdiff = getattr(ctx.field_gen, VC.TILE_DIFFICULTY_ATTR, None) or {}", False),
            ("if diff is not None:", 'out["difficulty"] = diff', False)):
        (i,) = [k for k, ln in enumerate(lines) if (ln.strip().startswith(needle) if prefix else ln.strip() == needle)]
        assert lines[i - 1].strip().startswith(guard), (needle, lines[i - 1])
    for needle in ('dwhy = VC.field_difficulty_reasons(tdiff, grid, band=band, want=FIELD_PIN["want"])',
                   "if dwhy:", "the realised field is not difficulty",
                   's["terrain_tile_difficulty"] = {k: tdiff.get(k) for k in s["terrain_tiles"]["tiles"]}',
                   'out["visited_difficulty"] = vd', "vd = VC.visited_difficulty(run, tdiff, want)"):
        assert needle in src, needle
    # the realised refusal comes after the field refusal and before the first step (inside _field_realised)
    tree = ast.parse(src)
    (fr,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_field_realised"]
    seg = ast.get_source_segment(src, fr)
    assert seg.index("the video would leave the field") < seg.index("if dwhy:") < seg.index("return out, gen")
    # the probe's attribute is only read here, never written
    assert not [ln for ln in lines if "TILE_DIFFICULTY_ATTR" in ln and "setattr" in ln]


# ===================================================================================== the launch
def test_mix_seed_1514_fits_the_field_on_difficulty_1_tiles(GEN, VC, S, tmp_path):
    """The rough video's seed needs 1 row ahead + 1 column left of the spawn tile (path + 2 m), so the spawn row is
    pinned to 0..8 and the path covers 2 x 2 tiles -- every one built at 1.0 under field_max (no flat tile: FLAT_FRAC
    0.0). Same grid as field_medium (the band changes no geometry): no reseed needed."""
    vids = tmp_path / "v"
    assert GEN.main(["--plan", "mix", "--seed", str(LAUNCH_SEED), "--total-s", "180", "--out-dir", str(vids)]) == 0
    scenarios, _ = S.load_manifest(vids / "scenarios_sh_video_mix.yaml")
    (sc,) = [s for s in scenarios if s.id == "sh_mix__omni"]
    assert [h.label for h in sc.schedule] == ["fwd", "turn_l90", "turn_r90", "stand", "move_turn_l", "move_turn_r",
                                              "stand", "diag", "lat_r", "stand", "back", "lat_l", "back", "stand",
                                              "lat_l", "fwd"]
    grid = _field_grid(VC)
    ex = VC.commanded_extent(sc.schedule, prologue_s=float(sc.get_default("prologue_s")))
    assert ex["x_max"] == pytest.approx(11.024518, abs=1e-5) and ex["y_max"] == pytest.approx(5.013518, abs=1e-5)
    assert ex["x_min"] == 0.0 and ex["y_min"] == 0.0
    need = VC.field_tiles_needed(ex, grid)
    assert need == {"ahead": 1, "behind": 0, "left": 1, "right": 0, "margin_m": 2.0}
    cap = VC.field_spawn_row_cap(need, grid)
    assert cap == 8
    every = {VC.tile_key((r, c)): VC.FIELD_MAX_DIFFICULTY for r in range(10) for c in range(20)}
    assert VC.field_difficulty_reasons(every, grid, band=(1.0, 1.0)) == []
    for r in range(grid["num_rows"]):
        x0 = (r + 0.5) * 10.0 - 50.0
        y0 = 0.5 * 10.0 - 100.0
        why = VC.field_path_reasons(x0, y0, ex, grid)
        assert (why == []) == (r <= cap), (r, why)
        if r <= cap:
            assert VC.field_path_tiles(x0, y0, ex, grid) == [(r, 0), (r, 1), (r + 1, 0), (r + 1, 1)]


def test_launcher_dry_run_passes_field_max_through(TV, tmp_path):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    env, marker = TV._fake_bin(tmp_path, "/usr/bin/bash idle")
    before = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run", "--plan", "mix", "--seed", str(LAUNCH_SEED), "--total-s",
                        "180", "--stand-rule", "schedule", "--terrain", "field_max", "--prefix",
                        "sh_mix_F19_s1514_roughest", "--", "--checkpoint", str(T_STAR), "--stand-ckpt", "x.pt",
                        "--stand-mask", "1", "--deadband", "0.25"], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr
    (c,) = [ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]
    for needle in ("timeout --signal=TERM 5400 ", "--sequence sh_mix__omni", "scenarios_sh_video_mix.yaml",
                   "--prefix sh_mix_F19_s1514_roughest ", "--stand-rule schedule --terrain field_max ",
                   f"--checkpoint {T_STAR}", "--deadband 0.25", "MemoryMax=24G", "MemorySwapMax=0"):
        assert needle in c, (needle, c)
    assert " -k " not in c
    assert not marker.exists() and not (tmp_path / "out").exists()
    after = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    assert after == before                                                              # a dry run writes nothing
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run", "--terrain", "field_maximum"], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 5 and "--terrain must be flat | field_medium | field_max" in r.stderr


def test_no_bytecode_from_the_field_max_tools():
    assert _pyc_snapshot() == _PYC_BEFORE
    for pat in ("sh_video*", "sh_record*"):
        assert not list(SCRIPTS.glob(f"__pycache__/{pat}")), pat
