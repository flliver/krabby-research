"""Layout invariants of ``crab_hex_forward_task/experiments/`` (campaign records, keep-set only).

Pure Python + git. The campaign trees live in the package whole (raw artifacts stay on disk); git
tracks only the keep-set: records, eval summaries, reference banks and ONE checkpoint of record
per campaign under ``<campaign>/head/`` (``experiments/tools/bundle_experiment.py``), plus the May-2026 stage-baseline
bundles tracked whole under ``old-runs/<stamp>/`` (checkpoint of record plus any paired USD snapshot / ONNX export).
"""
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task"
EXP = PKG / "experiments"
REL = EXP.relative_to(REPO)


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", str(REL)], cwd=REPO, capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l]


def test_no_package_marker_under_experiments():
    """An __init__.py below experiments/ would make Isaac's import_packages import every campaign
    driver at gym registration."""
    assert not list(EXP.rglob("__init__.py"))


def test_only_the_keep_set_is_tracked():
    files = _tracked()
    assert files, "experiments/ must be tracked"
    bad_pt = [f for f in files if f.endswith(".pt") and "/head/" not in f and not f.startswith(f"{REL}/old-runs/")]
    bad_ext = [f for f in files if f.endswith((".log", ".mp4", ".pid")) or "/events.out.tfevents." in f]
    bad_dirs = [f for f in files if "/metrics/" in f or "/raw/" in f or "/logs/rsl_rl/" in f and not f.endswith("curriculum_state.json")]
    assert not bad_pt, bad_pt[:5]
    assert not bad_ext, bad_ext[:5]
    assert not bad_dirs, bad_dirs[:5]


def test_bundled_heads_match_their_manifests():
    manifests = sorted(EXP.glob("*/bundle.yaml"))
    if not manifests:
        pytest.skip("no bundle.yaml yet (bundle commit pending)")
    yaml = pytest.importorskip("yaml")
    for mf in manifests:
        b = yaml.safe_load(mf.read_text())
        head = b.get("head") or {}
        if not head.get("dest"):
            continue
        f = mf.parent / head["dest"]
        assert f.exists(), f
        assert hashlib.sha256(f.read_bytes()).hexdigest() == head["sha256"], f


@pytest.mark.parametrize("driver", [
    "tools/run_phases.py",
    "2026-09-06_2130_a15b_lineage/run_lineage.py",
    "2026-09-03_1156_obstacle_exposure/run_exposure.py",
    "2026-09-04_1105_morph_x_exposure/run_morph_exposure.py",
    "2026-09-02_1446_leg_mount_morphology/run_formation_arms.py",
])
def test_driver_repo_root_constant(driver):
    """Drivers compute the repo root from their own location; the move changed the depth."""
    path = EXP / driver
    spec = importlib.util.spec_from_file_location(path.stem + "_layout_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert Path(mod.REPO).resolve() == REPO
