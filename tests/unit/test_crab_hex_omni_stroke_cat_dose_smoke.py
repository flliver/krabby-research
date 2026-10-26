"""The per-category W campaign's dose-smoke launcher (``experiments/2026-09-27_stroke_cat_W/
stroke_cat_dose_smoke.py``), on CPU, without Isaac.

Mirrors ``test_crab_hex_omni_formation_dose_smoke.py``: load the module by path, compose against the
REAL ``candidates.yaml`` this campaign ships, and drive each refusal by monkeypatching one module-
level constant at a time rather than a synthetic fake driver. The one structural difference this file
exercises is the composition path itself -- ``reference_recipe`` + ``stage_env`` on the campaign's own
``references: {S_cat: ...}`` entry, never ``compose_recipe`` on a candidate (this campaign runs no
candidate at all). Nothing here imports Isaac, starts Kit, or touches the GPU; ``--print-env`` is the
only path exercised end to end.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMPAIGN = OMNI / "experiments/2026-09-27_stroke_cat_W"
TOOLS = OMNI / "experiments/tools"
SCRIPT = CAMPAIGN / "stroke_cat_dose_smoke.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ds():
    return _load(SCRIPT, "stroke_cat_dose_smoke__test")


@pytest.fixture(scope="module")
def rc(ds):
    return _load(TOOLS / "run_candidates.py", "run_candidates_stroke_cat_dose_smoke_test")


# ============================================================================ the checkpoint / manifest
def test_stage_checkpoint_matches_the_manifest_and_the_file_on_disk(ds):
    # W left policy/ at the 2026-10-06 bake: the fork resolves through the manifest's superseded W entry to
    # its sha-pinned source run, never to the same-named stage dir of the current (side-first B) lineage
    path, sha = ds._stage_checkpoint("5_c6x_clockfree")
    assert path == ds.REPO / "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-25_14-53-27_seed3_X03_c6x_A15pB/model_11995.pt"
    assert sha == "3a527683eccdc73b66ca06bdf89bfd9d27c0f87b1f929cf25bf7fc6f01145202"
    assert ds.POLICY_DIR not in path.parents
    if not (ds.REPO / "parkour/logs").is_dir():  # a checkout without the machine-local training logs (CI)
        pytest.skip(f"{ds.REPO / 'parkour/logs'} is not on this machine: the on-disk / sha256 check of W's "
                    "source run checkpoint needs the local training logs")
    assert path.is_file(), f"{path}: W's 5_c6x_clockfree source run checkpoint is not on disk"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_stage_checkpoint_refuses_an_unknown_stage_dir(ds):
    with pytest.raises(SystemExit, match="no stage with dir"):
        ds._stage_checkpoint("not-a-real-stage")


def test_stage_checkpoint_refuses_a_missing_manifest(ds, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "MANIFEST", tmp_path / "no_such_manifest.yaml")
    with pytest.raises(SystemExit, match="not on disk"):
        ds._stage_checkpoint("5_c6x_clockfree")


# ============================================================================ compose() against the real yaml
def test_compose_reads_the_real_campaign_yaml_via_the_reference_not_a_candidate(ds, rc, tmp_path):
    comp = ds.compose(rc, tmp_path / "parity.npz", 16, 300)
    env = comp["env"]
    cands = rc.load_candidates(ds.CANDS)
    # the doses come from the reference chain's OWN stage overlay (candidates.yaml chains.c6y_scat
    # stage c6y), never a candidate's overlay -- this campaign has none (candidates: [])
    assert cands.get("candidates") in (None, [])
    recipe = rc.reference_recipe(cands, ds.REFERENCE, "expect")
    assert recipe["chain"] == ds.CHAIN
    expect_env = rc.stage_env(recipe, ds.STAGE)
    for key in ds.DOSE_KEYS:
        assert comp["doses"][key] == expect_env[key]
        assert env[key] == expect_env[key]
    # the clock-free play keys; CLOCK_HIDE already present via require_env, not set by PLAY_KEYS
    assert env["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    assert env["KRABBY_OMNI_FACING_OBS_CLIP"] == "0.6"
    assert "KRABBY_OMNI_CLOCK_HIDE" not in ds.PLAY_KEYS
    assert env["KRABBY_OMNI_CLOCK_HIDE"] == "1"
    # require_env (K + split flag), re-asserted
    assert env["KRABBY_CLOCK_W"] == "0.001"
    assert env["KRABBY_APEX_W"] == "0.001"
    assert env["KRABBY_OMNI_REWARD_SPLIT"] == "1"
    # no README eval-block keys
    assert not (set(ds.EVAL_BLOCK_ONLY) & env.keys())
    # the parity-recorder vars
    assert env["KRABBY_OMNI_STROKE_PARITY_DUMP"] == str(tmp_path / "parity.npz")
    assert env["KRABBY_OMNI_STROKE_PARITY_ENVS"] == "16"
    assert env["KRABBY_OMNI_STROKE_PARITY_STEPS"] == "300"
    assert comp["recipe_id"] == recipe["recipe_id"]


def test_compose_refuses_when_candidates_yaml_is_absent(ds, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CANDS", tmp_path / "not_written_yet.yaml")
    with pytest.raises(SystemExit, match="not written yet"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_an_unknown_reference(ds, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "REFERENCE", "NOT_A_REFERENCE")
    with pytest.raises(SystemExit, match="not found"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_reference_chain_mismatch(ds, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CHAIN", "not-the-real-chain")
    with pytest.raises(SystemExit, match="expected"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_missing_dose_key(ds, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "DOSE_KEYS", ("KRABBY_NOT_A_REAL_DOSE_KEY",))
    with pytest.raises(SystemExit, match="KRABBY_NOT_A_REAL_DOSE_KEY"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_when_clock_hide_is_not_already_one(ds, rc, monkeypatch, tmp_path):
    # simulate a require_env regression on the chain by intercepting stage_env's return value
    real_stage_env = rc.stage_env

    def _tampered(recipe, stage, extra=None):
        env = dict(real_stage_env(recipe, stage, extra))
        env["KRABBY_OMNI_CLOCK_HIDE"] = "0"
        return env

    monkeypatch.setattr(rc, "stage_env", _tampered)
    with pytest.raises(SystemExit, match="KRABBY_OMNI_CLOCK_HIDE must already be '1'"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_play_key_that_conflicts_with_the_composed_env(ds, rc, monkeypatch, tmp_path):
    # the chain's own c6y overlay already sets KRABBY_OMNI_FACING to 0.2:0.6 (candidates.yaml); asking
    # the play block to also own that key with a different value must be refused, never silently applied
    monkeypatch.setattr(ds, "PLAY_KEYS", {"KRABBY_OMNI_FACING": "9.9:9.9"})
    with pytest.raises(SystemExit, match="the composed env has"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_when_must_does_not_match(ds, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "MUST", {"KRABBY_OMNI_REWARD_SPLIT": "0"})
    with pytest.raises(SystemExit, match="must be"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_leaked_eval_block_key(ds, rc, monkeypatch, tmp_path):
    # KRABBY_OMNI_FACING really is in the composed env (the chain's c6y overlay); same trick as the
    # formation test uses to exercise the leak check without needing a synthetic env
    monkeypatch.setattr(ds, "EVAL_BLOCK_ONLY", {"KRABBY_OMNI_FACING": "0.2:0.6"})
    with pytest.raises(SystemExit, match="README eval-block keys"):
        ds.compose(rc, tmp_path / "parity.npz", 16, 300)


# ============================================================================ the CLI (--print-env only)
def test_print_env_cli_runs_on_cpu_and_prints_the_composed_env(ds, tmp_path, capsys):
    rc_code = ds.main(["--print-env", "--out", str(tmp_path)])
    assert rc_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fork"].endswith("parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-25_14-53-27_seed3_X03_c6x_A15pB/model_11995.pt")
    assert len(printed["fork_sha256_expected"]) == 64
    assert printed["extra_env"]["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    assert printed["extra_env"]["KRABBY_OMNI_CLOCK_HIDE"] == "1"
    assert printed["doses"]["KRABBY_OMNI_STROKE_W"]
    assert printed["doses"]["KRABBY_OMNI_STROKE_CAT_W"]
