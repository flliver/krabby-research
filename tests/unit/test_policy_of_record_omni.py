"""The omni-walk POLICY OF RECORD (``crab_hex_omni_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through
(1_t3 -> 2_c4 -> 3_c5 -> 4_c6 -> 5_c6x_clockfree, all phase-1a stages with per-stage overlays) plus the shared
training assets it depends on (``assets:``, the omni RSI bank). Every declared file must be present, tracked by
git, sha-pinned to the manifest, and (for stages) be the only checkpoint in its folder; the README is generated
from the manifest by ``experiments/tools/bundle_policy.py --sync``. The current head was trained with the gait
clock hidden, and its observation contract must say so.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task"
POLICY = PKG / "policy"
TOOL = PKG / "experiments" / "tools" / "bundle_policy.py"
RUN_PHASES = PKG / "experiments" / "tools" / "run_phases.py"

STAGES = ["1_t3", "2_c4", "3_c5", "4_c6", "5_c6x_clockfree"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tool():
    return _load(TOOL, "bundle_policy_omni")


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", str(POLICY.relative_to(REPO))], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return {l for l in out.stdout.splitlines() if l}


def test_manifest_and_files_are_consistent():
    tool = _tool()
    m = tool.load_manifest()
    assert m["task"] == "crab_hex_omni_task" and m["plant"] == "A15+B"
    problems = tool.check(m)
    assert not problems, problems


def test_stage_order_and_chain():
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    assert [s["dir"] for s in m["stages"]] == STAGES
    assert all(s["phase"] == "1a" for s in m["stages"])
    assert [s["stage"] for s in m["stages"]] == ["t3", "c4", "c5", "c6", "c6x"]
    assert m["stages"][0]["resume_from"] == "scratch"
    for prev, s in zip(m["stages"], m["stages"][1:]):
        assert prev["dir"] in s["resume_from"], f"{s['dir']} must resume from {prev['dir']}"
    assert m["current"] == m["stages"][-1]["dir"] == "5_c6x_clockfree"


def test_every_policy_file_is_tracked_and_nothing_else():
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    tracked = _tracked()
    rel = POLICY.relative_to(REPO)
    tool = _tool()
    expected = {f"{rel}/{f}" for f in tool.expected_files(m)}
    assert expected <= tracked, sorted(expected - tracked)
    on_disk = {str(p.relative_to(REPO)) for p in POLICY.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert on_disk == expected, sorted(on_disk ^ expected)


def test_readme_is_generated_from_the_manifest():
    tool = _tool()
    m = tool.load_manifest()
    assert (POLICY / "README.md").read_text() == tool.readme_text(m), "policy/README.md drifted: run bundle_policy.py --sync"
    text = (POLICY / "README.md").read_text()
    for s in m["stages"]:
        assert f"`{s['dir']}/`" in text and s["sha256"] in text
    for a in m.get("assets", []):
        assert f"`{a['dir']}/{a['file']}`" in text and a["sha256"] in text
    for f in m.get("followups", []):
        assert f"`{f['id']}`" in text and f["headline"] in text
    assert "Nobody actually turns during move_turn" in text
    assert "KRABBY_OMNI_CLOCK_HIDE=1" in text


def test_rsi_bank_asset_is_the_bank_the_omni_presets_use():
    """The bundled RSI bank is byte-identical to the bank every KRABBY_OMNI_PHASE preset seeds resets from."""
    sys.path.insert(0, str(PKG / "config" / "crab_hex"))
    import crab_hex_omni_phases as ph  # noqa: E402

    tool = _tool()
    m = tool.load_manifest()
    banks = [a for a in m.get("assets", []) if a["file"].startswith("rsi_bank")]
    assert banks, "policy manifest declares no RSI bank asset"
    preset_bank = Path(ph.OMNI_BANK)
    assert preset_bank.exists(), preset_bank
    assert tool.sha256_of(preset_bank) == banks[0]["sha256"]
    assert ph.OMNI_COMMON["KRABBY_RSI_BANK"] == str(preset_bank)
    for name in ph.PHASES:
        assert ph.phase_env(name).get("KRABBY_RSI_BANK") == str(preset_bank), name
    assert ph.phase_env("1a").get("KRABBY_RSI_FRAC") == "0.2"


def test_obs_contract_current_head_is_clock_hidden():
    """The current head trained with KRABBY_OMNI_CLOCK_HIDE=1; the earlier stages with the clock visible; the
    manifest's carried semantics cover exactly the driver's EVAL_CARRY set."""
    tool = _tool()
    m = tool.load_manifest()
    cur = tool.stage_by_dir(m, m["current"])
    assert str(cur["obs_contract"]["KRABBY_OMNI_CLOCK_HIDE"]) == "1"
    assert tool.eval_env(m, cur)["KRABBY_OMNI_CLOCK_HIDE"] == "1"
    assert tool.play_env(m, cur)["KRABBY_OMNI_CLOCK_HIDE"] == "1"
    for s in m["stages"][:-1]:
        assert str(s["obs_contract"]["KRABBY_OMNI_CLOCK_HIDE"]) == "0", s["dir"]
    assert m["obs_contract"]["width"] == 1149
    rp = _load(RUN_PHASES, "omni_run_phases_policy_test")
    carry = m["obs_contract"]["eval_env"]["carry"]
    assert set(carry) | {"KRABBY_OMNI_CLOCK_HIDE"} == set(rp.EVAL_CARRY)
    for k, v in rp.EVAL_CARRY_DEFAULTS.items():
        if k != "KRABBY_OMNI_CLOCK_HIDE":
            assert str(carry[k]) == v, k


def test_bundle_policy_check_is_clean():
    out = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
