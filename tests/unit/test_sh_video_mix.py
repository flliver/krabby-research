"""Stand-height campaign (``2026-10-13_stand_height``): the seeded COMMAND MIX video plan and the non-S1 walker on CPU
(no Isaac / Kit / GPU; VIDEO ONLY, read by no gate). Added 2026-10-24 for the user's "a longer video (3 min) running
through various commands with the occasional stand at various heights" of the M18 Task 4 F19 deliverable (walker
T* = p3b + the stand-height net SH_T4F19_sh_s19 trained beside it).

Pinned here:

* ``sh_video_schedule.py --plan mix --seed N --total-s S``: deterministic per seed; every tour command type (the
  suite's speeds, :data:`TOUR_CMDS`) in a dealt order, the turn / move_turn pairs atomic (the facing is 0 at every
  stand), a single stand every 2-3 commands with heights spread over [h_min, h_max] (the first five stands in distinct
  bins), lengths = the tour rules + the mix's draws, never a stand last; the row passes ``sh_video_core
  .tour_row_reasons`` + the tour lengths + the harness validator, is written under ``stand_height_video`` (the launcher's
  loader refuses it: turns), re-loads through ``gait_eval.schedule.load_manifest`` and ``sh_video_core
  .load_height_row``; ``--table-start`` is refused; the launch seed 1514 is pinned;
* every pre-existing plan's YAML and JSON are BYTE-IDENTICAL to the generator before the mix (commit 36c80bc2);
* ``sh_record_video.sh --plan mix --seed N``: the dry run's command (row, manifest, default prefix, timeout), the
  refusals (mix without --seed; --seed with the other plans still exclusive), nothing written;
* the walker: ``sh_record_video.py``'s walker IS ``--checkpoint`` (cli_args); every walker identity check compares
  against ``sha256(--checkpoint)`` (the reference stamp's ``checkpoint_sha256``, the stand net's ``infos.sh_dagger
  .s1_sha256``), so a non-S1 walker records iff it is the walker the stand net was trained beside AND the stamp ran;
  on the machine's F19 files the REAL pre-Kit path (a stub ``isaaclab.app``) reaches Kit with T* + SH_T4F19 + the F19
  hraise stamp (reference checks all equal) and refuses S1 s18 against that stamp (fail closed); the F19 stand
  checkpoint's recorded walker sha and estimator are T*'s (the post-Kit pairing checks, read from the files).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
VIDEOS = CAMP / "videos"
GEN_PATH = SCRIPTS / "sh_video_schedule.py"
REC_PATH = SCRIPTS / "sh_record_video.py"
SH_PATH = SCRIPTS / "sh_record_video.sh"
GEN_REL = "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-13_stand_height/scripts/" \
          "sh_video_schedule.py"
#: the generator before the mix plan (the last commit that touched it)
PRE_MIX_COMMIT = "36c80bc2"
DS = 10
MIX_SEEDS = tuple(range(0, 40)) + (1514,)
LAUNCH_SEED = 1514
LAUNCH_LABELS = ["fwd", "turn_l90", "turn_r90", "stand", "move_turn_l", "move_turn_r", "stand", "diag", "lat_r",
                 "stand", "back", "lat_l", "back", "stand", "lat_l", "fwd"]
LAUNCH_HEIGHTS = [None, None, None, 0.73, None, None, 1.322, None, None, 0.871, None, None, None, 1.176, None, None]

T_STAR = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt"
T_STAR_SHA = "03b6009a6e952133f6c21c36c8d315b76f945d431615ef6d6798b7f4a5f59fc7"
F19_SH = REPO / "parkour/outputs/omni_plan/stand_height/runs/SH_T4F19_sh_s19/model_5000.pt"
F19_STAMPS = REPO / "parkour/outputs/omni_plan/phase2/lr/SH_T4F19_sh_s19/hraise__omni/seed001"
S1_S18 = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"


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


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def GEN():
    return _load(GEN_PATH, "_t_shvm_gen")


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
def mixes(GEN, HDR):
    return {s: GEN.generate("mix", s, 180.0, hdr=HDR) for s in MIX_SEEDS}


@pytest.fixture(scope="module")
def TV():
    """The base video test module (its fake bin for the launcher), reused."""
    return _load(Path(__file__).resolve().parent / "test_sh_video.py", "_t_shvm_base")


# ===================================================================================== the mix plan
def test_mix_is_deterministic_per_seed_and_differs_across_seeds(GEN, HDR):
    key = lambda g: [(h.label, h.hold_s, h.as_world()) for h in g["holds"]] + [g["heights"]]  # noqa: E731
    for s in (0, 7, LAUNCH_SEED):
        assert key(GEN.generate("mix", s, 180.0, hdr=HDR)) == key(GEN.generate("mix", s, 180.0, hdr=HDR))
    assert key(GEN.generate("mix", 1, 180.0, hdr=HDR)) != key(GEN.generate("mix", 2, 180.0, hdr=HDR))


def test_mix_structure_commands_pairs_and_stands(GEN, VC, CORE, mixes):
    cmds = GEN.TOUR_CMDS
    for seed, g in mixes.items():
        holds, heights, kinds = g["holds"], g["heights"], VC.hold_kinds(g["holds"])
        labels = [h.label for h in holds]
        assert labels[0] == "fwd" and kinds[-1] != VC.KIND_STAND, seed               # opener; never a stand last
        assert set(cmds) <= set(labels) and not g["summary"]["missing_command_types"], seed   # every command type
        for i, (h, hh, k) in enumerate(zip(holds, heights, kinds)):
            assert abs(h.hold_s * DS - round(h.hold_s * DS)) < 1e-9                    # whole deciseconds
            if h.label == "stand":
                assert k == VC.KIND_STAND and hh is not None and CORE.H_MIN <= hh <= CORE.H_MAX
                assert h.face == holds[i - 1].face == 0.0 and labels[i - 1] != "stand"   # single, at face 0
            else:
                assert hh is None and (h.speed, h.dir, h.face) == cmds[h.label], (seed, i, h)
            if h.label in ("turn_l90", "move_turn_l"):                                 # the pairs are atomic
                assert labels[i + 1] == {"turn_l90": "turn_r90", "move_turn_l": "move_turn_r"}[h.label], (seed, i)
            if h.label in ("turn_r90", "move_turn_r"):
                assert labels[i - 1] == {"turn_r90": "turn_l90", "move_turn_r": "move_turn_l"}[h.label], (seed, i)
        stand_at = [i for i, k in enumerate(kinds) if k == VC.KIND_STAND]
        blocks = [b - a - 1 for a, b in zip([-1] + stand_at, stand_at + [len(holds)])]
        assert stand_at and all(2 <= n <= 4 for n in blocks), (seed, blocks)          # >= U{2, 3} commands, pairs
        bins = g["summary"]["height_bins"]
        assert len(set(bins[:GEN.MIX_HEIGHT_BINS])) == min(len(bins), GEN.MIX_HEIGHT_BINS), (seed, bins)
        assert g["summary"]["video_s"] <= 180.0 + 1e-9 and g["summary"]["n_turns"] >= 2


def test_mix_lengths_are_the_tour_rules_plus_the_draws(GEN, VC, CORE, mixes):
    rate = CORE.HEIGHT_RATE_M_S
    for seed, g in mixes.items():
        holds, heights = g["holds"], g["heights"]
        for i, (h, hh) in enumerate(zip(holds, heights)):
            if hh is not None:
                rule = GEN._ceil_ds(abs(hh - CORE.H_SETTLED) / rate + GEN.TOUR_STAND_SETTLE_S + GEN.TOUR_STAND_HOLD_S)
                extra = round(h.hold_s * DS) - rule
                assert GEN.MIX_STAND_EXTRA_DS[0] <= extra <= GEN.MIX_STAND_EXTRA_DS[1], (seed, i, extra)
            else:
                prev = heights[i - 1] if i > 0 else None
                allow = 0.0 if prev is None else abs(prev - CORE.H_SETTLED) / rate + GEN.TOUR_RESTORE_EXTRA_S
                last_stretch = GEN.FINAL_STRETCH_DS if i == len(holds) - 1 else 0
                lo = GEN._ceil_ds(GEN.MIX_CMD_DS[0] / DS + allow)
                hi = GEN._ceil_ds(GEN.MIX_CMD_DS[1] / DS + allow) + last_stretch
                assert lo <= round(h.hold_s * DS) <= hi, (seed, i, h.hold_s, lo, hi)
        assert not GEN.tour_length_reasons(holds, heights)


def test_mix_rows_pass_the_tour_rules_and_the_harness_validator(GEN, VC, EC, S, HDR, mixes):
    for seed, g in mixes.items():
        hv = GEN.hold_dicts(g["holds"])
        assert VC.tour_row_reasons(hv, g["heights"]) == [], seed
        assert EC.height_row_reasons(hv, g["heights"]), seed       # turns: never a launcher row
        S.validate_scenario(GEN._scenario(S, HDR, g["holds"], g["row_id"]))
    g = mixes[0]
    bad = list(g["heights"])
    i = bad.index(next(h for h in bad if h is not None))
    bad[i] = None                                                   # a stand without a height: refused
    with pytest.raises(ValueError, match="tour rules refuse"):
        GEN.check_rules(S, HDR, g["holds"], bad, g["row_id"])


def test_mix_launch_seed_is_pinned(mixes, CORE):
    g = mixes[LAUNCH_SEED]
    assert [h.label for h in g["holds"]] == LAUNCH_LABELS
    assert g["heights"] == LAUNCH_HEIGHTS
    sm = g["summary"]
    assert sm["video_s"] == 180.0 and sm["n_stands"] == 4 and sm["height_bins"] == [0, 4, 1, 3]
    assert sm["height_min"] < CORE.H_MIN + 0.02 and sm["height_max"] > CORE.H_MAX - 0.03


def test_mix_cli_writes_the_video_block_and_refuses_a_table(GEN, VC, EC, S, tmp_path, capsys):
    out = tmp_path / "v"
    assert GEN.main(["--plan", "mix", "--seed", str(LAUNCH_SEED), "--out-dir", str(out)]) == 0
    yml = out / "scenarios_sh_video_mix.yaml"
    data = yaml.safe_load(yml.read_text())
    assert VC.VIDEO_BLOCK in data and EC.MANIFEST_BLOCK not in data
    assert data[VC.VIDEO_BLOCK]["rows"]["sh_mix__omni"]["rules"] == VC.VIDEO_RULES
    assert "--plan mix --seed 1514 --total-s 180" in yml.read_text()                 # the generator's default 180
    assert VC.video_row_names(yml) == ["sh_mix__omni"]
    row = VC.load_height_row(yml, "sh_mix__omni")
    assert row["hold_heights"] == LAUNCH_HEIGHTS and row["labels"] == LAUNCH_LABELS
    (sc,) = S.load_manifest(yml)[0]
    assert sc.id == "sh_mix__omni" and sc.command_frame == "world" and abs(sc.total_duration_s - 180.0) < 1e-9
    with pytest.raises(ValueError):
        EC.height_row(yml, "sh_mix__omni")
    js = json.loads((out / "sh_video_schedule_mix.json").read_text())
    assert js["plan"] == "mix" and js["seed"] == LAUNCH_SEED and js["summary"]["missing_command_types"] == []
    capsys.readouterr()
    assert GEN.main(["--plan", "mix", "--seed", "3", "--table-start", "0.635", "--out-dir", str(tmp_path / "t")]) == 2
    assert "must be a stand" in capsys.readouterr().err and not (tmp_path / "t").exists()
    assert GEN.main(["--plan", "mix", "--seed", "3", "--total-s", "30", "--out-dir", str(tmp_path / "s")]) == 2


def _pre_mix_generator(tmp_path):
    """The generator at :data:`PRE_MIX_COMMIT`, executed with ``__file__`` = the live path (its HERE-relative paths
    resolve to the campaign), never written next to the scripts."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_MIX_COMMIT}:{GEN_REL}"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git show {PRE_MIX_COMMIT} unavailable: {r.stderr.strip()}")
    mod = types.ModuleType("_t_shvm_gen_pre")
    mod.__file__ = str(GEN_PATH)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        exec(compile(r.stdout, str(GEN_PATH), "exec"), mod.__dict__)  # noqa: S102 - the repo's own source
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_every_pre_existing_plan_is_byte_identical(GEN, tmp_path):
    OLD = _pre_mix_generator(tmp_path)
    runs = [["--plan", p] for p in ("raise", "smoke", "tour", "tour_smoke")]
    runs += [["--plan", "tour", "--table-start", "0.635:knee_folded"]]
    runs += [["--plan", "random", "--seed", str(s), "--total-s", t] for s in (0, 1, 7, 42) for t in ("150", "180")]
    runs += [["--plan", "random", "--seed", "5"], ["--plan", "raise", "--seed", "9"]]       # the defaults
    for i, args in enumerate(runs):
        a, b = tmp_path / f"old{i}", tmp_path / f"new{i}"
        assert OLD.main(args + ["--out-dir", str(a)]) == 0 and GEN.main(args + ["--out-dir", str(b)]) == 0
        fa, fb = sorted(p.name for p in a.iterdir()), sorted(p.name for p in b.iterdir())
        assert fa == fb and len(fa) == 2, (args, fa, fb)
        for n in fa:
            assert (a / n).read_bytes() == (b / n).read_bytes(), (args, n)


# ===================================================================================== the launcher
def test_launcher_dry_run_of_the_mix(TV, tmp_path):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    env, marker = TV._fake_bin(tmp_path, "/usr/bin/bash idle")
    before = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    for extra, prefix, total in (([], f"sh_mix_s{LAUNCH_SEED}", "180"),
                                 (["--total-s", "150", "--prefix", "sh_mix_x"], "sh_mix_x", "150")):
        r = subprocess.run(["bash", str(SH_PATH), "--dry-run", "--plan", "mix", "--seed", str(LAUNCH_SEED), *extra,
                            "--stand-rule", "schedule", "--", "--checkpoint", str(T_STAR), "--stand-ckpt", "x.pt",
                            "--stand-mask", "1", "--deadband", "0.25"], env=env, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stderr
        (c,) = [ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]
        for needle in ("timeout --signal=TERM 5400 ", "--sequence sh_mix__omni", "scenarios_sh_video_mix.yaml",
                       f"--prefix {prefix} ", "--stand-rule schedule", f"--checkpoint {T_STAR}", "--deadband 0.25",
                       "MemoryMax=24G", "MemorySwapMax=0"):
            assert needle in c, (needle, c)
        assert f"plan mix seed {LAUNCH_SEED}" in r.stdout and "would write" in r.stdout
        assert f"video {total}" in r.stdout or total == "150"
    assert not marker.exists() and not (tmp_path / "out").exists()
    after = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    assert after == before                                                              # a dry run writes nothing


def test_launcher_mix_refusals(TV, tmp_path):
    env, marker = TV._fake_bin(tmp_path, "/usr/bin/bash idle")
    for args, needle in ((["--dry-run", "--plan", "mix"], "--plan mix needs --seed"),
                         (["--dry-run", "--plan", "tour", "--seed", "3"], "exclusive"),
                         (["--dry-run", "--plan", "mix", "--seed", "3", "--smoke"], "exclusive"),
                         (["--dry-run", "--plan", "mix", "--seed", "x"], "non-negative integer"),
                         (["--dry-run", "--plan", "mix", "--seed", "3", "--total-s", "9x"], "--total-s"),
                         (["--dry-run", "--plan", "mix", "--seed", "3", "--table-start", "0.635"],
                          "schedule generation failed"),
                         (["--dry-run", "--total-s", "90"], "random plan only")):
        r = subprocess.run(["bash", str(SH_PATH), *args], env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 5 and needle in r.stderr, (args, r.returncode, r.stderr)
    assert not marker.exists()


# ===================================================================================== the walker
def test_the_recorders_walker_is_checkpoint_and_every_walker_check_reads_its_sha():
    src = REC_PATH.read_text()
    assert "cli_args.add_rsl_rl_args(parser)" in src                                  # defines --checkpoint
    assert "if args_cli.checkpoint is None:\n    args_cli.checkpoint = str(DEFAULT_CKPT)" in src   # default only
    assert "S1_SHA = _sha256(Path(args_cli.checkpoint))" in src
    assert '"s1_ckpt_sha256": (RUN_META.get("checkpoint_sha256"), S1_SHA)' in src    # the stamp's walker (pre-Kit)
    assert 's1_trained = sd.get("s1_sha256")\n    if s1_trained != S1_SHA and not args_cli.allow_reference_mismatch:' \
        in src                                                                         # the stand net's walker
    assert "resume_path = retrieve_file_path(str(args_cli.checkpoint))\n    runner.load(resume_path)" in src
    assert "agent.yaml" not in src                                    # no per-walker agent cfg lookup (teacher path)
    # no other S1 pin: S1's path is the docstring's and DEFAULT_CKPT's (read once, as the default) only
    assert src.count("DEFAULT_CKPT") == 2 and src.count("model_19994") == 2


def _f19_stamp():
    if not (T_STAR.is_file() and F19_SH.is_file() and F19_STAMPS.is_dir()):
        pytest.skip("the F19 walker / stand net / SH hraise stamp is not on this machine")
    stamps = sorted(d for d in F19_STAMPS.iterdir() if (d / "p2_complete.json").is_file())
    if not stamps:
        pytest.skip("no complete F19 SH hraise stamp")
    return stamps[-1]


def test_f19_stand_checkpoint_was_trained_beside_t_star():
    """The post-Kit pairing checks, read from the files: infos.sh_dagger.s1_sha256 == sha256(T*) (the recorder's
    ``_check_stand_ckpt``), its mask / deadband, and its estimator == T*'s (the estimator identity)."""
    stamp = _f19_stamp()
    assert _sha(T_STAR) == T_STAR_SHA
    ck = torch.load(str(F19_SH), map_location="cpu", weights_only=False)
    sd = ck["infos"]["sh_dagger"]
    assert sd["s1_sha256"] == T_STAR_SHA and sd["stand_mask"] == 1
    w = torch.load(str(T_STAR), map_location="cpu", weights_only=False)
    ew, es = w["estimator_state_dict"], ck["estimator_state_dict"]
    assert ew.keys() == es.keys() and all(torch.equal(ew[k], es[k]) for k in ew)
    rm = json.loads((stamp / "run_meta.json").read_text())
    assert rm["checkpoint_sha256"] == T_STAR_SHA and rm["p2r_mode"]["specialist"]["sha256"] == _sha(F19_SH)
    assert rm["payload"] is None                                                       # the 0-kg deployment


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
    env = {k: v for k, v in os.environ.items() if not k.startswith("KRABBY_")}
    env.update(PYTHONPATH=str(stub), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", str(REC_PATH), "--headless", "--enable_cameras",
                           "--out-dir", str(tmp_path / "out"), *args], env=env, capture_output=True, text=True,
                          timeout=600, cwd=str(tmp_path))


def test_recorder_pre_kit_accepts_t_star_beside_its_stand_net_and_refuses_s1(GEN, tmp_path):
    """The REAL pre-Kit path (stub Kit) on the mix row: T* + SH_T4F19 + the F19 hraise stamp reach Kit with the
    stamp's env and every reference check equal; S1 s18 as the walker against that stamp is refused before Kit
    (``s1_ckpt_sha256``) -- a non-S1 walker records only beside the stamp it ran."""
    stamp = _f19_stamp()
    vids = tmp_path / "v"
    assert GEN.main(["--plan", "mix", "--seed", str(LAUNCH_SEED), "--out-dir", str(vids)]) == 0
    row = ["--manifest", str(vids / "scenarios_sh_video_mix.yaml"), "--sequence", "sh_mix__omni"]
    head = ["--stand-ckpt", str(F19_SH), "--stand-mask", "1", "--deadband", "0.25", "--eval-run", str(F19_STAMPS)]
    ref_env = {k: str(v) for k, v in json.loads((stamp / "run_meta.json").read_text())["env_vars"].items()}
    r = _run_recorder_pre_kit(tmp_path, row + head + ["--checkpoint", str(T_STAR), "--stand-rule", "schedule"])
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("STUB_KIT_BOOT ")]
    assert r.returncode == 42 and len(line) == 1, (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    assert json.loads(line[0][len("STUB_KIT_BOOT "):]) == ref_env
    assert "[SHV] reference checks: all equal" in r.stdout and str(stamp / "run_meta.json") in r.stdout
    assert "height row sh_mix__omni" in r.stdout
    if S1_S18.is_file():
        r = _run_recorder_pre_kit(tmp_path, row + head + ["--checkpoint", str(S1_S18)])
        assert r.returncode not in (0, 42) and "s1_ckpt_sha256" in r.stderr, (r.returncode, r.stderr[-2000:])
    r = _run_recorder_pre_kit(tmp_path, row + head + ["--checkpoint", str(T_STAR), "--deadband", "0"])
    assert r.returncode not in (0, 42) and "residual_deadband" in r.stderr                # still fail closed


def test_no_bytecode_from_the_mix_tools():
    assert _pyc_snapshot() == _PYC_BEFORE
    for pat in ("sh_video*", "sh_record*"):
        assert not list(SCRIPTS.glob(f"__pycache__/{pat}")), pat
