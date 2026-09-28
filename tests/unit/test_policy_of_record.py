"""The forward-walk POLICY OF RECORD (``crab_hex_forward_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through
(1a -> 2a -> 2b -> 2c -> 3a) plus the shared training assets it depends on (``assets:``, e.g. the RSI bank).
Every declared file must be present, tracked by git, sha-pinned to the manifest, and (for stages) be the only
checkpoint in its folder; the README is generated from the manifest by ``experiments/tools/bundle_policy.py
--sync`` and must mention every stage and asset.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task"
POLICY = PKG / "policy"
TOOL = PKG / "experiments" / "tools" / "bundle_policy.py"
SUMMARY_TOOL = PKG / "experiments" / "tools" / "policy_summary.py"


def _tool():
    spec = importlib.util.spec_from_file_location("bundle_policy", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bundle_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _summary():
    spec = importlib.util.spec_from_file_location("policy_summary", SUMMARY_TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["policy_summary"] = mod
    spec.loader.exec_module(mod)
    return mod


def _prose_only(text: str) -> str:
    """The hand-written parts of POLICY_SUMMARY.md (generated blocks removed)."""
    import re
    return re.sub(r"<!-- generated:([a-z0-9-]+) -->.*?<!-- /generated:\1 -->", "", text, flags=re.S)


def _pair(v: str) -> tuple[float, float]:
    lo, hi = v.split(":")
    return float(lo), float(hi)


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", str(POLICY.relative_to(REPO))], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return {l for l in out.stdout.splitlines() if l}


def test_manifest_and_files_are_consistent():
    tool = _tool()
    m = tool.load_manifest()
    assert m["task"] == "crab_hex_forward_task" and m["plant"] == "A15+B"
    problems = tool.check(m)
    assert not problems, problems


def test_stage_order_and_chain():
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    phases = [s["phase"] for s in m["stages"]]
    assert phases == ["1a", "2a", "2b", "2c", "3a"], phases
    assert m["stages"][0]["resume_from"] == "scratch"
    for prev, s in zip(m["stages"], m["stages"][1:]):
        assert prev["dir"] in s["resume_from"], f"{s['dir']} must resume from {prev['dir']}"
    assert m["current"] == m["stages"][-1]["dir"]


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


def test_rsi_bank_asset_is_the_bank_the_presets_use():
    """The bundled RSI bank is byte-identical to the bank every KRABBY_PHASE preset seeds resets from."""
    sys.path.insert(0, str(PKG / "config" / "crab_hex"))
    import crab_hex_phases as ph  # noqa: E402

    tool = _tool()
    m = tool.load_manifest()
    banks = [a for a in m.get("assets", []) if a["file"].startswith("rsi_bank")]
    assert banks, "policy manifest declares no RSI bank asset"
    preset_bank = Path(ph.FORMATION["KRABBY_RSI_BANK"])
    assert preset_bank.exists(), preset_bank
    assert tool.sha256_of(preset_bank) == banks[0]["sha256"]
    for name in ("1a", "2a", "2b", "2c", "3a"):
        assert ph.PHASES[name].env.get("KRABBY_RSI_BANK") == str(preset_bank), name
        assert ph.PHASES[name].env.get("KRABBY_RSI_FRAC") == "0.2", name


# ------------------------------------------------------------------ POLICY_SUMMARY.md + mdp_pins.yaml
def test_policy_summary_blocks_are_current():
    """Every generated block of POLICY_SUMMARY.md equals a fresh render (marker ids and renderers in lock-step)."""
    ps = _summary()
    m = ps.load_manifest()
    pins = ps.load_pins()
    text = (POLICY / "POLICY_SUMMARY.md").read_text()
    assert ps.update_summary(text, ps.render_blocks(m, pins)) == text, "POLICY_SUMMARY.md blocks are stale: run bundle_policy.py --sync"
    assert "(generated)" not in text
    for s in m["stages"]:
        assert f"`{s['dir']}/`" in text
    for t in ps.reward_terms_from_cfg_source():
        assert f"`{t['name']}`" in text, t["name"]
    pure = ps.load_pure()
    for p in ps.DOCUMENTED_PHASES:
        for k in pure.ph.phase_env(p, pure.ph.MAIN_PLANT):
            assert f"`{k}`" in text or f"{k}=" in text, k
    assert "not baked" in text


def test_mdp_pins_agree_with_presets():
    """Every pin derivable from a preset knob matches phase_env(); reward weights match the class defaults + knobs."""
    import math

    ps = _summary()
    pure = ps.load_pure()
    pins = ps.load_pins()
    terms = ps.reward_terms_from_cfg_source()
    knob_map = ps.knob_overrides_from_cfg_source()
    for p, pin in pins.items():
        env = pure.ph.phase_env(p, pure.ph.MAIN_PLANT)
        if p in ps.RL_PHASES:
            assert set(pin["rewards"]) == {t["name"] for t in terms}, p
            want = ps.effective_weights(pure, p, terms, knob_map)
            got = {t: r["weight"] for t, r in pin["rewards"].items()}
            assert got == want, f"{p}: {sorted(k for k in want if want[k] != got.get(k))}"
            ramps = {t: (w0, w1) for t, w0, w1, _, _ in ps.ramps_of(pure, p)}
            cur = {c["term_name"]: (c["w0"], c["w1"]) for c in (pin["curriculum"] or {}).values()}
            assert cur == ramps, p
        else:
            assert pin["curriculum"] is None, p
        tg, pk, cm, ev = pin["terrain"], pin["parkour"], pin["commands"], pin["events"]
        assert tuple(tg["difficulty_range"]) == _pair(env["KRABBY_FLAT_TERRAIN_DIFF"]), p
        assert tg["curriculum"] == ("KRABBY_FLAT_TERRAIN_CURRICULUM" in env), p
        assert pk["freeze_terrain_levels"] == ("KRABBY_FLAT_TERRAIN_CURRICULUM" not in env), p
        assert math.isclose(tg["sub_terrains"]["parkour_flat"]["proportion"], float(env["KRABBY_FLAT_TERRAIN_FLAT_FRAC"])), p
        if "KRABBY_TERRAIN_PROMOTE" in env:
            assert (pk["move_up_frac"], pk["move_down_frac"]) == _pair(env["KRABBY_TERRAIN_PROMOTE"]), p
        if env.get("KRABBY_FLAT_TERRAIN_GEOM") == "recal2b2w":
            for sub in ("parkour_gap", "parkour_hurdle", "parkour_step"):
                assert tuple(tg["sub_terrains"][sub]["half_valid_width"]) == pure.xk.RECAL2B2W_HALF_VALID_WIDTH, (p, sub)
                assert tuple(tg["sub_terrains"][sub]["y_range"]) == pure.xk.RECAL2B2W_Y_RANGE, (p, sub)
            assert tg["sub_terrains"]["parkour"]["stone_width"] == pure.xk.RECAL2B2W_STONE_WIDTH, p
        assert pin["episode_length_s"] == float(env["KRABBY_EPISODE_S"]), p
        assert cm["stand_frac"] == float(env["KRABBY_STAND_FRAC"]), p
        assert tuple(cm["resampling_time_range"]) == _pair(env["KRABBY_RESAMPLE_S"]), p
        assert tuple(cm["ranges"]["lin_vel_x"]) == _pair(env["KRABBY_LIN_VEL_X"]), p
        assert tuple(cm["ranges"]["heading"]) == (_pair(env["KRABBY_HEADING"]) if "KRABBY_HEADING" in env else (0.0, 0.0)), p
        if "KRABBY_DR_PUSH" in env:
            v = float(env["KRABBY_DR_PUSH"])
            assert ev["push_by_setting_velocity"]["params"]["velocity_range"] == {"x": [-v, v], "y": [-v, v]}, p
        else:
            assert ev["push_by_setting_velocity"] is None, p
        if "KRABBY_DR_MASS" in env:
            assert tuple(ev["randomize_rigid_body_mass"]["params"]["mass_distribution_params"]) == _pair(env["KRABBY_DR_MASS"]), p
        else:
            assert ev["randomize_rigid_body_mass"] is None, p
        if "KRABBY_DR_COM" in env:
            v = float(env["KRABBY_DR_COM"])
            assert ev["randomize_rigid_body_com"]["params"]["com_range"] == {"x": [-v, v], "y": [-v, v], "z": [-v, v]}, p
        else:
            assert ev["randomize_rigid_body_com"] is None, p
        rsi = ev["rsi_reference_reset"]
        assert rsi["params"]["fraction"] == float(env["KRABBY_RSI_FRAC"]), p
        assert rsi["params"]["bank"]["sha256"] == ps.sha256_of(Path(env["KRABBY_RSI_BANK"])), p
        assert rsi["params"]["fix_spawn"] is ("KRABBY_RSI_SPAWN_FIX" in env), p


def test_prose_knob_values_match_presets():
    """Every `KRABBY_X=value` quoted in the hand-written prose is a value some documented phase really sets."""
    import re

    ps = _summary()
    pure = ps.load_pure()
    envs = {p: pure.ph.phase_env(p, pure.ph.MAIN_PLANT) for p in ps.DOCUMENTED_PHASES}
    known = {k for e in envs.values() for k in e}
    prose = _prose_only((POLICY / "POLICY_SUMMARY.md").read_text())
    bad = []
    for knob, value in re.findall(r"(KRABBY_[A-Z0-9_]+)=([^\s,;)`\]]+)", prose):
        if knob not in known:
            continue
        if not any(e.get(knob) == value for e in envs.values()):
            bad.append(f"{knob}={value}")
    assert not bad, bad


def test_bundle_policy_check_is_clean():
    tool = _tool()
    assert tool.check(tool.load_manifest()) == []

