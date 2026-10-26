"""The formation campaign's t3 dose-smoke launcher (``experiments/2026-09-26_stroke_formation/
formation_dose_smoke.py``), on CPU, without Isaac.

Generalised from the stroke/posture screen's ``window0_dose_smoke.py`` (no sibling test file exists for
that script; this one follows this repo's other driver-adjacent tests instead: load the module by path,
compose against the REAL ``candidates.yaml`` this campaign ships, and drive each refusal by monkeypatching
one module-level constant at a time rather than a synthetic fake driver). Nothing here imports Isaac,
starts Kit, or touches the GPU; ``--print-env`` is the only path exercised end to end.
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
CAMPAIGN = OMNI / "experiments/2026-09-26_stroke_formation"
TOOLS = OMNI / "experiments/tools"
SCRIPT = CAMPAIGN / "formation_dose_smoke.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fs():
    return _load(SCRIPT, "formation_dose_smoke__test")


@pytest.fixture(scope="module")
def rc(fs):
    return _load(TOOLS / "run_candidates.py", "run_candidates_formation_dose_smoke_test")


# ============================================================================ the checkpoint / manifest
def test_stage_checkpoint_matches_the_manifest_and_the_file_on_disk(fs):
    # W left policy/ at the 2026-10-06 bake: the fork resolves through the manifest's superseded W entry to
    # its sha-pinned source run, never to the same-named stage dir of the current (side-first B) lineage
    path, sha = fs._stage_checkpoint("1_t3")
    assert path == fs.REPO / "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_09-10-29_seed3_X08_t3_A15pB/model_2999.pt"
    assert sha == "03b87cd3432cc58e9e1cafbbbf36e3a6737034a8cc82a674e8181e9993432035"
    assert fs.POLICY_DIR not in path.parents
    if not (fs.REPO / "parkour/logs").is_dir():  # a checkout without the machine-local training logs (CI)
        pytest.skip(f"{fs.REPO / 'parkour/logs'} is not on this machine: the on-disk / sha256 check of W's "
                    "source run checkpoint needs the local training logs")
    assert path.is_file(), f"{path}: W's 1_t3 source run checkpoint is not on disk"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_stage_checkpoint_refuses_an_unknown_stage_dir(fs):
    with pytest.raises(SystemExit, match="no stage with dir"):
        fs._stage_checkpoint("not-a-real-stage")


def test_stage_checkpoint_refuses_a_missing_manifest(fs, monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "MANIFEST", tmp_path / "no_such_manifest.yaml")
    with pytest.raises(SystemExit, match="not on disk"):
        fs._stage_checkpoint("1_t3")


# ============================================================================ compose() against the real yaml
def test_compose_reads_the_real_campaign_yaml_and_builds_arm_c(fs, rc, tmp_path):
    comp = fs.compose(rc, tmp_path / "parity.npz", 16, 300)
    env = comp["env"]
    cands = rc.load_candidates(fs.CANDS)
    expect_doses = {key: str((rc.candidate_by_id(cands, cid) or {})["overlay"][fs.STAGE][key])
                   for cid, key in fs.DOSE_CANDIDATES}
    assert comp["doses"] == expect_doses
    for key, val in expect_doses.items():
        assert env[key] == val
    # the clock-held play keys, never KRABBY_OMNI_CLOCK_HIDE
    assert env["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    assert env["KRABBY_OMNI_FACING_OBS_CLIP"] == "0.6"
    assert "KRABBY_OMNI_CLOCK_HIDE" not in env
    # require_env, re-asserted
    assert env["KRABBY_OMNI_REWARD_SPLIT"] == "1"
    # no K, no README eval-block keys
    assert not (set(fs.FORBIDDEN_K) & env.keys())
    assert not (set(fs.EVAL_BLOCK_ONLY) & env.keys())
    # the parity-recorder vars
    assert env["KRABBY_OMNI_STROKE_PARITY_DUMP"] == str(tmp_path / "parity.npz")
    assert env["KRABBY_OMNI_STROKE_PARITY_ENVS"] == "16"
    assert env["KRABBY_OMNI_STROKE_PARITY_STEPS"] == "300"
    assert comp["recipe_id"]


def test_compose_refuses_when_candidates_yaml_is_absent(fs, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "CANDS", tmp_path / "not_written_yet.yaml")
    with pytest.raises(SystemExit, match="not written yet"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_an_unknown_dose_candidate(fs, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "DOSE_CANDIDATES", (("NOT_A_CANDIDATE", "KRABBY_OMNI_STROKE_W"),))
    with pytest.raises(SystemExit, match="not found"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_dose_candidate_missing_its_overlay_key(fs, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "DOSE_CANDIDATES", (("S_f", "KRABBY_NOT_A_REAL_KEY"),))
    with pytest.raises(SystemExit, match="no t3.KRABBY_NOT_A_REAL_KEY"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_play_key_that_conflicts_with_the_composed_env(fs, rc, monkeypatch, tmp_path):
    # the chain's own t3 overlay already sets KRABBY_OMNI_FACING to 0.2:0.6 (candidates.yaml); asking
    # the play block to also own that key with a different value must be refused, never silently applied
    monkeypatch.setattr(fs, "PLAY_KEYS", {"KRABBY_OMNI_FACING": "9.9:9.9"})
    with pytest.raises(SystemExit, match="the composed env has"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_when_must_does_not_match(fs, rc, monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "MUST", {"KRABBY_OMNI_REWARD_SPLIT": "0"})
    with pytest.raises(SystemExit, match="must be"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_leaked_k_key(fs, rc, monkeypatch, tmp_path):
    # KRABBY_OMNI_REWARD_SPLIT really is in the composed env (require_env); treating it as forbidden
    # exercises the leak check itself without needing a synthetic env
    monkeypatch.setattr(fs, "FORBIDDEN_K", ("KRABBY_OMNI_REWARD_SPLIT",))
    with pytest.raises(SystemExit, match="no K"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


def test_compose_refuses_a_leaked_eval_block_key(fs, rc, monkeypatch, tmp_path):
    # KRABBY_OMNI_FACING really is in the composed env (the chain's t3 overlay); same trick as above
    monkeypatch.setattr(fs, "EVAL_BLOCK_ONLY", {"KRABBY_OMNI_FACING": "0.2:0.6"})
    with pytest.raises(SystemExit, match="README eval-block keys"):
        fs.compose(rc, tmp_path / "parity.npz", 16, 300)


# ============================================================================ the CLI (--print-env only)
def test_print_env_cli_runs_on_cpu_and_prints_the_composed_env(fs, tmp_path, capsys):
    rc_code = fs.main(["--print-env", "--out", str(tmp_path)])
    assert rc_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fork"].endswith("parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_09-10-29_seed3_X08_t3_A15pB/model_2999.pt")
    assert len(printed["fork_sha256_expected"]) == 64
    assert printed["extra_env"]["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    assert "KRABBY_OMNI_CLOCK_HIDE" not in printed["extra_env"]
