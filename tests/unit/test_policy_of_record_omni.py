"""The omni-walk POLICY OF RECORD (``crab_hex_omni_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through
(the side-first B lineage: 1_fb6k (preset 1b) -> 2_c4 -> 3_c5 -> 4_c6 -> 5_c6x_clockfree (preset 1a), each with a
per-stage overlay from the PPO-construction campaign's candidates.yaml, baked 2026-10-06 as the phase-1 head; then
6_p2a_field (preset 2a + the omni phase-2 campaign's P2__S1_s18 overlay), baked 2026-10-08 as the head that ships)
plus the shared training assets it depends on (``assets:``, the omni RSI bank). Every declared file must be
present, tracked by git, sha-pinned to the manifest, and (for stages) be the only checkpoint in its folder; the
README is generated from the manifest by ``experiments/tools/bundle_policy.py --sync``. The c6x and phase-2 heads
were trained with the gait clock hidden, and their observation contract must say so. Every stage's eval environment must
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

STAGES = ["1_fb6k", "2_c4", "3_c5", "4_c6", "5_c6x_clockfree", "6_p2a_field"]
#: the stages trained with the clock observation hidden (c6x and every stage after it)
CLOCK_HIDDEN = {"5_c6x_clockfree", "6_p2a_field"}


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


def _chain_stage(ref: str) -> dict:
    """The chain stage an ``overlay_ref`` names, from its (tracked) candidates.yaml."""
    mo = re.fullmatch(r"(\S+/candidates\.yaml) chains\.(\S+) stage (\S+)", ref)
    assert mo, f"overlay_ref not machine-readable: {ref!r}"
    doc = yaml.safe_load((PKG / mo.group(1)).read_text())
    st = [x for x in doc["chains"][mo.group(2)]["stages"] if x["stage"] == mo.group(3)]
    assert len(st) == 1, ref
    return st[0]


def _overlay(ref: str) -> tuple[str, dict[str, str]]:
    """``(base preset, overlay)`` of the chain stage an ``overlay_ref`` names (tracked candidates.yaml)."""
    st = _chain_stage(ref)
    return str(st["base"]), {k: str(v) for k, v in st["overlay"].items()}


def test_manifest_and_files_are_consistent():
    tool = _tool()
    m = tool.load_manifest()
    assert m["task"] == "crab_hex_omni_task" and m["plant"] == "A15+B"
    problems = tool.check(m)
    assert not problems, problems


def test_stage_order_and_chain():
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    assert [s["dir"] for s in m["stages"]] == STAGES
    assert [s["phase"] for s in m["stages"]] == ["1b", "1a", "1a", "1a", "1a", "2a"]
    assert [s["stage"] for s in m["stages"]] == ["s1", "c4", "c5", "c6", "c6x", "p2a"]
    assert m["stages"][0]["resume_from"] == "scratch"
    for prev, s in zip(m["stages"], m["stages"][1:]):
        assert prev["dir"] in s["resume_from"], f"{s['dir']} must resume from {prev['dir']}"
    assert m["current"] == m["stages"][-1]["dir"] == "6_p2a_field"
    # the phase-1 head stays documented as such; the phase-2 head is the one that ships
    assert m["heads"] == {"phase1": "5_c6x_clockfree", "phase2": "6_p2a_field"}


def test_stage_files_are_one_resume_chain():
    """Each stage file is the checkpoint the next stage resumed from, so the folder is ONE lineage at the manifest
    seed: a stage file from another lineage at the same seed -- e.g. the C0 FB6k arm X44 instead of the B arm X43
    -- or from another seed breaks the chain even when its sha is copied consistently.

    Two sources of truth, both tracked: a stage whose chain forks by ``resume_from: {checkpoint: ...}`` (the
    phase-2 stage) must name the previous stage's source run there, byte for byte; every other stage is read from
    its campaign's own driver state (state.json), whose candidates.yaml must still be the one the driver ran."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())

    def rel(p):
        return p[p.index("parkour/logs/"):] if p else None

    def camp_of(s):
        return re.match(r"(experiments/[^/]+)/", s["overlay_ref"]).group(1)

    states = {}
    prev = None
    n_forks = n_driver = 0
    for s in m["stages"]:
        assert f"seed{m['seed']}_" in s["source"], f"{s['dir']}: {s['source']} is not a seed{m['seed']} run"
        cs = _chain_stage(s["overlay_ref"])
        ck = (cs.get("resume_from") or {}).get("checkpoint") if isinstance(cs.get("resume_from"), dict) else None
        if ck is not None:
            assert prev is not None and not cs.get("from_scratch"), s["dir"]
            assert ck == prev, f"{s['dir']}: chain forks {ck}, not the previous stage's source {prev}"
            n_forks += 1
        else:
            camp = camp_of(s)
            if camp not in states:
                st = json.loads((PKG / camp / "state.json").read_text())
                # the overlays every other test reads are the ones the driver ran (the live yaml, byte for byte)
                assert (hashlib.sha256((PKG / camp / "candidates.yaml").read_bytes()).hexdigest()
                        == st["ladder"]["candidates_sha256"]), camp
                states[camp] = {rel(e["ckpt"]): e for e in st["keys"][f"seed{m['seed']}"].values()
                                if isinstance(e, dict) and e.get("ckpt")}
            e = states[camp].get(s["source"])
            assert e, f"{s['dir']}: {s['source']} is not a seed{m['seed']} checkpoint of {camp}"
            assert rel(e.get("resume")) == prev, f"{s['dir']} resumed from {e.get('resume')}, not {prev}"
            assert e["env"].get("KRABBY_OMNI_PHASE") == s["phase"], s["dir"]
            n_driver += 1
        prev = s["source"]
    assert (n_driver, n_forks) == (5, 1)


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
    # the phase-2 head ships; the README names both heads and the phase-2 training task
    assert "policy/6_p2a_field/model_19994.pt" in text and "Isaac-Crab-Hex-Omni-Teacher-v0" in text
    assert "phase1 [`5_c6x_clockfree/`]" in text and "phase2 [`6_p2a_field/`]" in text
    assert "| `6_p2a_field/` | `field_hard` | 0.8125 | 400 |" in text


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
    """The c6x and phase-2 heads trained with KRABBY_OMNI_CLOCK_HIDE=1; the earlier stages with the clock visible; the
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
    for s in m["stages"]:
        assert str(s["obs_contract"][HIDE]) == ("1" if s["dir"] in CLOCK_HIDDEN else "0"), s["dir"]
    # phase 2 did not change the policy's inputs: the phase-2 head's full eval env is the c6x head's
    assert tool.eval_env(m, cur) == tool.eval_env(m, tool.stage_by_dir(m, "5_c6x_clockfree"))
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
    """The manifest's runner-side construction block (and each stage's removal set) is in every stage's overlay,
    except the keys a stage declares in its own ``construction_delta`` -- which must be what that overlay trained
    and never a POLICY-structure knob (the soft std / its init are carried into eval and play)."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    con = m["construction"]
    assert con["ppo"]["KRABBY_STD_PARAM"] == m["obs_contract"]["eval_env"]["carry"]["KRABBY_STD_PARAM"]
    assert str(con["ppo"]["KRABBY_INIT_STD"]) == str(m["obs_contract"]["eval_env"]["carry"]["KRABBY_INIT_STD"])
    deltas = {s["dir"]: s.get("construction_delta") or {} for s in m["stages"]}
    # the only declared delta is the phase-2 S1 sibling knob (CHARTER 8 of experiments/2026-10-07_omni_phase2)
    assert {d: v for d, v in deltas.items() if v} == {"6_p2a_field": {"KRABBY_ACTION_RATE_W": "-0.05"}}
    for s in m["stages"]:
        _, ov = _overlay(s["overlay_ref"])
        delta = {k: str(v) for k, v in deltas[s["dir"]].items()}
        assert not set(delta) & {"KRABBY_STD_PARAM", "KRABBY_INIT_STD"}, s["dir"]
        for blk in ("ppo", "every_stage"):
            for k, v in con[blk].items():
                assert ov.get(k) == delta.get(k, str(v)), (s["dir"], k, ov.get(k), v)
        for k, v in (s.get("removal_set") or {}).items():
            assert ov.get(k) == str(v), (s["dir"], k)
        assert ov.get(HIDE, "0") == str(s["obs_contract"][HIDE]), s["dir"]


def test_phase2_stage_is_the_s18_field_fork_with_its_evals_of_record():
    """6_p2a_field (baked 2026-10-08): the P2__S1_s18 p2a stage -- preset 2a on the teacher task, the wall field,
    the clock still removed -- with its obstacle rows pooled consistently, the selection recorded, and the two
    sibling seeds (forked from THEIR OWN c6x heads, not this lineage) kept as data-only replicates."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    s = next(x for x in m["stages"] if x["dir"] == "6_p2a_field")
    assert (s["task"], s["play_task"]) == ("Isaac-Crab-Hex-Omni-Teacher-v0", "Isaac-Crab-Hex-Omni-Walk-Play-v0")
    assert s["overlay_ref"] == "experiments/2026-10-07_omni_phase2/candidates.yaml chains.P2__S1_s18 stage p2a"
    _, ov = _overlay(s["overlay_ref"])
    for k, v in {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_FIELD_EDGES": "wall", "KRABBY_PHASEOUT": "",
                 "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0", "KRABBY_TERRAIN_PROMOTE": "0.7:0.4"}.items():
        assert ov.get(k) == v, (k, ov.get(k))
    for k, v in s["removal_set"].items():
        assert ov.get(k) == str(v), k
    rows = s["evals"]["rows"]
    assert set(rows) == {"field_hard", "field_light", "ring_obst", "field_turn", "field_suite"}
    for name, r in rows.items():
        per = r.get("per_env_seed")
        if per:
            assert r["n"] == 100 * len(per), name
            assert abs(sum(per) / len(per) - r["completion"]) < 1e-9, name
    assert rows["field_hard"]["completion"] == 0.8125 and rows["field_hard"]["n"] == 400
    assert (rows["field_light"]["completion"], rows["ring_obst"]["completion"]) == (0.995, 1.0)
    assert s["evals"]["lr"]["hands"] == "FAST/FAST"
    assert {"rule", "judging", "fragile", "user_decision"} <= set(s["selection"])
    reps = s["replicates"]
    assert [r["id"].split()[0] for r in reps] == ["P2__S1_s16", "P2__S1_s17"]
    shas = {r["sha256"] for r in reps} | {s["sha256"]}
    assert len(shas) == 3 and all(re.fullmatch(r"[0-9a-f]{64}", x) for x in shas)
    for r, seed in zip(reps, (16, 17)):
        assert f"seed{seed}_" in r["source"] and f"seed{seed}_" in r["stamp"], r["id"]


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
