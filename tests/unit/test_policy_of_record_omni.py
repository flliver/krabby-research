"""The omni-walk POLICY OF RECORD (``crab_hex_omni_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through
(since the 2026-10-06 bake the side-first B lineage: 1_fb6k (preset 1b) -> 2_c4 -> 3_c5 -> 4_c6 ->
5_c6x_clockfree (preset 1a), each with a per-stage overlay from the PPO-construction campaign's candidates.yaml)
plus the shared training assets it depends on (``assets:``, the omni RSI bank). Every declared file must be
present, tracked by git, sha-pinned to the manifest, and (for stages) be the only checkpoint in its folder; the
README is generated from the manifest by ``experiments/tools/bundle_policy.py --sync``. The current head was
trained with the gait clock hidden, and its observation contract must say so. Every stage's eval environment must
be exactly what the driver's suite eval builds from that stage's preset + overlay, and the superseded head (W)
must have left the folder while staying recoverable from git history.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task"
POLICY = PKG / "policy"
TOOL = PKG / "experiments" / "tools" / "bundle_policy.py"
RUN_PHASES = PKG / "experiments" / "tools" / "run_phases.py"
HIDE = "KRABBY_OMNI_CLOCK_HIDE"

STAGES = ["1_fb6k", "2_c4", "3_c5", "4_c6", "5_c6x_clockfree"]


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


def _overlay(ref: str) -> tuple[str, dict[str, str]]:
    """``(base preset, overlay)`` of the chain stage an ``overlay_ref`` names (tracked candidates.yaml)."""
    mo = re.fullmatch(r"(\S+/candidates\.yaml) chains\.(\S+) stage (\S+)", ref)
    assert mo, f"overlay_ref not machine-readable: {ref!r}"
    doc = yaml.safe_load((PKG / mo.group(1)).read_text())
    st = [x for x in doc["chains"][mo.group(2)]["stages"] if x["stage"] == mo.group(3)]
    assert len(st) == 1, ref
    return str(st[0]["base"]), {k: str(v) for k, v in st[0]["overlay"].items()}


def test_manifest_and_files_are_consistent():
    tool = _tool()
    m = tool.load_manifest()
    assert m["task"] == "crab_hex_omni_task" and m["plant"] == "A15+B"
    problems = tool.check(m)
    assert not problems, problems


def test_stage_order_and_chain():
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    assert [s["dir"] for s in m["stages"]] == STAGES
    assert [s["phase"] for s in m["stages"]] == ["1b", "1a", "1a", "1a", "1a"]
    assert [s["stage"] for s in m["stages"]] == ["s1", "c4", "c5", "c6", "c6x"]
    assert m["stages"][0]["resume_from"] == "scratch"
    for prev, s in zip(m["stages"], m["stages"][1:]):
        assert prev["dir"] in s["resume_from"], f"{s['dir']} must resume from {prev['dir']}"
    assert m["current"] == m["stages"][-1]["dir"] == "5_c6x_clockfree"


def test_stage_files_are_one_resume_chain():
    """Each stage file is the checkpoint the next stage resumed from, per the training campaign's own driver state
    (tracked state.json): a stage file from another lineage at the same seed -- e.g. the C0 FB6k arm X44 instead of
    the B arm X43 -- or from another seed breaks the chain even when its sha is copied consistently."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    camps = {re.match(r"(experiments/[^/]+)/", s["overlay_ref"]).group(1) for s in m["stages"]}
    assert len(camps) == 1, camps
    camp = PKG / camps.pop()
    st = json.loads((camp / "state.json").read_text())
    # the overlays every other test reads are the ones the driver ran (the live yaml, byte for byte)
    assert hashlib.sha256((camp / "candidates.yaml").read_bytes()).hexdigest() == st["ladder"]["candidates_sha256"]

    def rel(p):
        return p[p.index("parkour/logs/"):] if p else None

    entries = {rel(e["ckpt"]): e for e in st["keys"][f"seed{m['seed']}"].values()
               if isinstance(e, dict) and e.get("ckpt")}
    prev = None
    for s in m["stages"]:
        e = entries.get(s["source"])
        assert e, f"{s['dir']}: {s['source']} is not a seed{m['seed']} checkpoint of the campaign"
        assert rel(e.get("resume")) == prev, f"{s['dir']} resumed from {e.get('resume')}, not {prev}"
        assert e["env"].get("KRABBY_OMNI_PHASE") == s["phase"], s["dir"]
        prev = s["source"]


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
    # the play / eval commands must rebuild the soft-std policy, or the strict load refuses the checkpoint
    assert "KRABBY_STD_PARAM=soft" in text


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
    manifest's carried semantics cover exactly the driver's EVAL_CARRY set, and every stage's full eval env is
    EXACTLY what the driver's suite eval (run_phases.py eval_morph: keep CANARY_KEYS from ph.FORMATION) builds
    from that stage's preset + candidates.yaml overlay -- so a carry value that is not what the stage trained
    under (the ellipse, the std parameterisation, the clock hide) fails here, not silently in a hand-run play."""
    tool = _tool()
    m = tool.load_manifest()
    cur = tool.stage_by_dir(m, m["current"])
    assert str(cur["obs_contract"][HIDE]) == "1"
    assert tool.eval_env(m, cur)[HIDE] == "1"
    assert tool.play_env(m, cur)[HIDE] == "1"
    for s in m["stages"][:-1]:
        assert str(s["obs_contract"][HIDE]) == "0", s["dir"]
    assert m["obs_contract"]["width"] == 1149
    rp = _load(RUN_PHASES, "omni_run_phases_policy_test")
    carry = m["obs_contract"]["eval_env"]["carry"]
    assert set(carry) | {HIDE} == set(rp.EVAL_CARRY)
    for s in m["stages"]:
        base, ov = _overlay(s["overlay_ref"])
        assert base == s["phase"], f"{s['dir']}: chain base {base} != manifest phase {s['phase']}"
        want = rp.eval_env_for(rp.ph.PHASES[s["phase"]], m["plant"], keep=rp.CANARY_KEYS,
                               keep_source=rp.ph.FORMATION, carry_overlay=ov)
        assert tool.eval_env(m, s) == want, s["dir"]


def test_recipe_construction_is_what_every_stage_trained():
    """The manifest's runner-side construction block (and the c6x removal set) is in every stage's overlay."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    con = m["construction"]
    assert con["ppo"]["KRABBY_STD_PARAM"] == m["obs_contract"]["eval_env"]["carry"]["KRABBY_STD_PARAM"]
    assert str(con["ppo"]["KRABBY_INIT_STD"]) == str(m["obs_contract"]["eval_env"]["carry"]["KRABBY_INIT_STD"])
    for s in m["stages"]:
        _, ov = _overlay(s["overlay_ref"])
        for blk in ("ppo", "every_stage"):
            for k, v in con[blk].items():
                assert ov.get(k) == str(v), (s["dir"], k, ov.get(k), v)
        for k, v in (s.get("removal_set") or {}).items():
            assert ov.get(k) == str(v), (s["dir"], k)
        assert ov.get(HIDE, "0") == str(s["obs_contract"][HIDE]), s["dir"]


def test_superseded_heads_left_the_folder_and_stay_in_history():
    """W (baked 2026-09-25) left policy/ at the 2026-10-06 bake; every one of its files is recoverable, byte-exact,
    from the commit the manifest names."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    sups = m.get("superseded") or []
    assert [s["id"] for s in sups] == ["W"]
    for sup in sups:
        commit = sup["last_commit"]
        assert re.fullmatch(r"[0-9a-f]{40}", str(commit)), f"superseded {sup['id']}: last_commit {commit!r}"
        for st in sup["stages"]:
            rel = f"{sup['path_in_history']}/{st['dir']}/{st['file']}"
            assert not (REPO / rel).exists(), f"{rel}: a superseded file is still in the folder"
            blob = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True,
                                  check=True).stdout
            assert hashlib.sha256(blob).hexdigest() == st["sha256"], rel


def test_bundle_policy_check_is_clean():
    out = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
