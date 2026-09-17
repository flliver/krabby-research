"""Layout invariants of both crab-hex tasks' ``experiments/`` (campaign records, keep-set only).

Pure Python + git. The campaign trees live in the package whole (raw artifacts stay on disk); git
tracks only the keep-set: records, eval summaries, reference banks and ONE checkpoint of record
per campaign under ``<campaign>/head/`` (``experiments/tools/bundle_experiment.py``), plus (forward
only) the May-2026 stage-baseline bundles tracked whole under ``old-runs/<stamp>/`` (checkpoint of
record plus any paired USD snapshot / ONNX export).

Parametrized over both packages: the sideways task's ``experiments/`` commits later than its
config/mdp does (Part G of the sideways plan), so the tracked-set assertion skips a package with
no tracked files yet rather than failing.
"""
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PARKOUR_TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"

#: package name -> its drivers that compute the repo root from their own file location.
PKGS = {
    "crab_hex_forward_task": [
        "tools/run_phases.py",
        "2026-09-06_2130_a15b_lineage/run_lineage.py",
        "2026-09-03_1156_obstacle_exposure/run_exposure.py",
        "2026-09-04_1105_morph_x_exposure/run_morph_exposure.py",
        "2026-09-02_1446_leg_mount_morphology/run_formation_arms.py",
    ],
    "crab_hex_sideways_task": [
        "tools/run_phases.py",
    ],
    "crab_hex_omni_task": [
        "tools/run_phases.py",
    ],
}


def _exp(pkg_name: str) -> Path:
    return PARKOUR_TASKS / pkg_name / "experiments"


def _tracked(pkg_name: str) -> list[str]:
    rel = _exp(pkg_name).relative_to(REPO)
    out = subprocess.run(["git", "ls-files", "--", str(rel)], cwd=REPO, capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l]


@pytest.mark.parametrize("pkg_name", sorted(PKGS))
def test_no_package_marker_under_experiments(pkg_name):
    """An __init__.py below experiments/ would make Isaac's import_packages import every campaign
    driver at gym registration."""
    assert not list(_exp(pkg_name).rglob("__init__.py"))


@pytest.mark.parametrize("pkg_name", sorted(PKGS))
def test_only_the_keep_set_is_tracked(pkg_name):
    files = _tracked(pkg_name)
    if not files:
        pytest.skip(f"{pkg_name}/experiments/ has no tracked files yet")
    rel = _exp(pkg_name).relative_to(REPO)
    bad_pt = [f for f in files if f.endswith(".pt") and "/head/" not in f and not f.startswith(f"{rel}/old-runs/")]
    bad_ext = [f for f in files if f.endswith((".log", ".mp4", ".pid")) or "/events.out.tfevents." in f]
    bad_dirs = [f for f in files if "/metrics/" in f or "/raw/" in f or "/logs/rsl_rl/" in f and not f.endswith("curriculum_state.json")]
    assert not bad_pt, bad_pt[:5]
    assert not bad_ext, bad_ext[:5]
    assert not bad_dirs, bad_dirs[:5]


@pytest.mark.parametrize("pkg_name", sorted(PKGS))
def test_bundled_heads_match_their_manifests(pkg_name):
    manifests = sorted(_exp(pkg_name).glob("*/bundle.yaml"))
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


@pytest.mark.parametrize("pkg_name,driver", [(p, d) for p, ds in PKGS.items() for d in ds])
def test_driver_repo_root_constant(pkg_name, driver):
    """Drivers compute the repo root from their own location; the move changed the depth."""
    path = _exp(pkg_name) / driver
    spec = importlib.util.spec_from_file_location(path.stem + "_layout_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert Path(mod.REPO).resolve() == REPO


def test_side_run_phases_is_self_relative():
    """The sideways driver: repo root, its own presets module, the student task id, and no leaked
    forward-driver leftovers (a hard-coded campaign path, or an un-inlined rx./rf. reference)."""
    path = _exp("crab_hex_sideways_task") / "tools/run_phases.py"
    spec = importlib.util.spec_from_file_location("side_run_phases_layout_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert Path(mod.REPO).resolve() == REPO
    assert "1a" in mod.ph.PHASES
    assert mod.STUDENT_TASK == "Isaac-Crab-Hex-Side-Student-v0"
    source = path.read_text()
    assert "experiments/2026-" not in source
    assert "rx." not in source
    assert "rf." not in source


def test_omni_run_phases_is_self_relative():
    """The omni driver: repo root, its own presets module, no student anywhere (this task is
    teacher-only -- no distillation task id at all), and no leaked forward/sideways-driver leftovers
    (a hard-coded campaign path, or an un-inlined rx./rf. reference)."""
    path = _exp("crab_hex_omni_task") / "tools/run_phases.py"
    spec = importlib.util.spec_from_file_location("omni_run_phases_layout_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert Path(mod.REPO).resolve() == REPO
    assert "1a" in mod.ph.PHASES
    source = path.read_text()
    assert "STUDENT_TASK" not in source
    assert "experiments/2026-" not in source
    assert "rx." not in source
    assert "rf." not in source
