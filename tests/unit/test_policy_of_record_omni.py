"""The omni-walk POLICY OF RECORD (``crab_hex_omni_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through
(the side-first B lineage: 1_fb6k (preset 1b) -> 2_c4 -> 3_c5 -> 4_c6 -> 5_c6x_clockfree (preset 1a), each with a
per-stage overlay from the PPO-construction campaign's candidates.yaml, baked 2026-10-06 as the phase-1 head; then
6_p2a_field (preset 2a + the omni phase-2 campaign's P2__S1_s18 overlay), baked 2026-10-08 as the head that ships)
plus the shared training assets it depends on (``assets:``, the omni RSI bank). Every declared file must be
present, tracked by git, sha-pinned to the manifest, and (for stages) be the only checkpoint in its folder; the
README is generated from the manifest by ``experiments/tools/bundle_policy.py --sync``. The c6x and phase-2 heads
were trained with the gait clock hidden, and their observation contract must say so. Every stage's eval environment must
be exactly what the driver's suite eval builds from that stage's preset + overlay. The superseded walking head W must
have left the folder while staying recoverable from git history; the superseded stand network A-mask s17 was retired
only AS THE NET OF RECORD -- its file stays in ``stand_net/`` as SH_T17's warm-start ancestor, byte-identical to the
file it was adopted with.

The STAND NETWORK OF RECORD (``stand_network:``, folder ``stand_net/``; adopted 2026-10-13 as A-mask s17, replaced
2026-10-15 by the height-commanded SH_T17 of experiments/2026-10-13_stand_height) is neither a stage nor an asset: its
file must be bundled sha-pinned (and equal its source run file); its checkpoint must be the height net trained beside
the walking stage it names (``infos.sh_dagger``: mask 1, the 0.25 deadband and the trim branch as the manifest says,
warm-started from the superseded A-mask s17; only the actor backbone differs from the walker, the estimator is the
walker's); its warm-start ancestor (``stand_network.warm_start``) must be bundled beside it, sha-pinned to the
checkpoint's recorded warm start, the superseded entry and the campaign's ``sh_core``; its contract constants must be
the campaign's ``sh_core`` and the checkpoint's; its routing env must be the walker's; its K1-K6 gate table must be the
campaign's reads (head_*.json, step4b_read.json); and adding it must leave the rest of the README untouched.
"""

import copy
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
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
#: the stand-network campaign (its sn_common.py is the stand-net input / D5 env rule both stand campaigns use)
SN_CAMPAIGN = PKG / "experiments" / "2026-10-11_stand_network"
SN_COMMON = SN_CAMPAIGN / "scripts" / "sn_common.py"
#: the stand-HEIGHT campaign (the stand network of record since 2026-10-15): its head_*.json / step4b_read.json gate
#: reads are tracked, and its sh_core.py is the ONE implementation of the height-commanded contract
SH_CAMPAIGN = PKG / "experiments" / "2026-10-13_stand_height"
SH_CORE = SH_CAMPAIGN / "scripts" / "sh_core.py"
#: the stand-height heads read against the step-4b controls (the gate table's head columns)
SH_HEADS = ("SH_T18", "SH_T16", "SH_T17")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_nobc(path: Path, name: str):
    """A private module instance by path that writes no ``__pycache__`` (the campaign scripts' own rule: their tests
    assert the scripts dirs stay bytecode-free)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
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
    # the phase-1 head stays documented as such; the phase-2 head is the one that ships; `stand` names the stand
    # network of record (adopted 2026-10-13), which is not a stage
    assert m["heads"] == {"phase1": "5_c6x_clockfree", "phase2": "6_p2a_field", "stand": "stand_net"}
    assert m["heads"]["stand"] not in STAGES


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
    assert expected <= tracked, f"policy files not in git's index (git add them): {sorted(expected - tracked)}"
    # a superseded file must leave the index too, not only the folder (--sync stages only what it writes): an
    # unstaged deletion would leave it in the committed tree with every other check green
    assert tracked <= expected, f"tracked files the manifest does not name (git rm them): {sorted(tracked - expected)}"
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


def test_superseded_heads_are_recorded_and_recoverable():
    """W (baked 2026-09-25) left policy/ at the 2026-10-06 bake; every one of its files is recoverable, byte-exact, from
    the commit the manifest names. The stand network A-mask s17 (adopted 2026-10-13) was superseded AS THE NET OF RECORD
    at the 2026-10-15 adoption of SH_T17 but its file was NOT removed: it is the bundled warm-start ancestor
    (``stand_network.warm_start``), present in stand_net/ with the recorded sha, and the very file it was adopted with."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    sn = _stand(m)
    ws = sn["warm_start"]
    sups = m.get("superseded") or []
    assert [s["id"] for s in sups] == ["W", "SN_Amask_s17"]
    w, stand = sups
    # W: its files left the folder and stay in history
    commit = w["last_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", str(commit)), f"superseded W: last_commit {commit!r}"
    for st in w["stages"]:
        rel = f"{w['path_in_history']}/{st['dir']}/{st['file']}"
        assert not (REPO / rel).exists(), f"{rel}: a superseded file is still in the folder"
        blob = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == st["sha256"], rel
    # A-mask s17: superseded as the net of record only; its file is the bundled warm start, kept in stand_net/
    assert stand["kind"] == "stand_network" and stand["id"] == ws["id"]
    assert "SH_T17" in stand["superseded"] and "2026-10-15" in stand["superseded"]
    assert "STAYS" in stand["superseded"] and "warm_start" in stand["superseded"]
    assert [(f["dir"], f["file"], f["sha256"], f["source"]) for f in stand["files"]] == [
        (sn["dir"], ws["file"], ws["sha256"], ws["source"])]
    assert ws["sha256"] != sn["sha256"]
    # no removal / restore-from-history language or keys survive on the kept file
    assert not {"last_commit", "path_in_history", "still_referenced_by"} & set(stand), sorted(stand)
    tool = _tool()
    p = tool.warm_start_path(sn)
    assert p == POLICY / stand["files"][0]["dir"] / stand["files"][0]["file"] and p.is_file(), p
    assert tool.sha256_of(p) == stand["files"][0]["sha256"]
    # ... and it is the very file A-mask s17 was adopted with (never moved)
    commit = stand["adopted_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", str(commit)) and commit[:8] in stand["adopted"], commit
    rel = str(p.relative_to(REPO))
    blob = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True, check=True).stdout
    assert hashlib.sha256(blob).hexdigest() == ws["sha256"], rel


def test_bundle_policy_check_is_clean():
    out = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


# ------------------------------------------------------------------------------ the stand network of record
def _stand(m: dict) -> dict:
    assert m.get("stand_network"), "policy manifest declares no stand_network block"
    return m["stand_network"]


def test_stand_network_file_is_bundled_with_the_manifest_sha():
    """The adopted SH_T17 file sits in its own folder (not a stage / asset dir) with the manifest sha, beside the
    UNCHANGED walking head of record it was trained and gated with; the folder holds exactly it and its warm-start
    ancestor (the superseded A-mask s17 file, kept)."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    assert (sn["dir"], sn["file"]) == ("stand_net", "model_5000.pt")
    assert (sn["id"], sn["arm"], sn["stand_mask"], sn["training_seed"], sn["iterations"]) == (
        "SH_T17", "A-mask", 1, 17, 5000)
    assert sn["dir"] not in {s["dir"] for s in m["stages"]} | {a["dir"] for a in m["assets"]}
    p = tool.stand_path(sn)
    assert p == POLICY / "stand_net" / "model_5000.pt" and p.is_file(), p
    assert tool.sha256_of(p) == sn["sha256"]
    assert sorted(q.name for q in p.parent.iterdir()) == ["model_4000.pt", "model_5000.pt"]
    assert [f["path"] for f in tool.stand_files(sn)] == [p, tool.warm_start_path(sn)] == [p, p.parent / "model_4000.pt"]
    assert m["heads"]["stand"] == sn["dir"]
    # the walking head of record is unchanged by the adoption, and the block pins exactly that walker
    assert m["current"] == sn["walker"]["stage"] == "6_p2a_field"
    assert sn["walker"]["sha256"] == tool.stage_by_dir(m, "6_p2a_field")["sha256"]
    assert '"Adopt SH_T17"' in sn["adopted"] and "2026-10-15" in sn["adopted"]
    assert sn["source"] == "parkour/outputs/omni_plan/stand_height/runs/SH_T17/model_5000.pt"


def test_expected_files_include_the_stand_network():
    tool = _tool()
    m = tool.load_manifest()
    files = tool.expected_files(m)
    # the stand network of record AND its bundled warm-start ancestor
    assert {"stand_net/model_5000.pt", "stand_net/model_4000.pt"} <= files
    no_ws = copy.deepcopy(m)
    del no_ws["stand_network"]["warm_start"]
    assert tool.expected_files(no_ws) == files - {"stand_net/model_4000.pt"}
    bare = copy.deepcopy(m)
    del bare["stand_network"]
    assert tool.expected_files(bare) == files - {"stand_net/model_5000.pt", "stand_net/model_4000.pt"}


def test_stand_network_source_is_the_bundled_file():
    """The source run file (untracked) is byte-identical to the bundled copy."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    src = REPO / sn["source"]
    if not src.exists():
        pytest.skip(f"stand-network source run file not on this machine (raw run dirs are untracked): {src}")
    assert tool.sha256_of(src) == sn["sha256"] == tool.sha256_of(tool.stand_path(sn))


def test_stand_warm_start_is_the_ancestor_sh_t17_records():
    """The bundled warm start (``stand_network.warm_start``, same folder) is the file SH_T17 was warm-started from:
    its sha256 equals SH_T17's checkpoint ``infos.sh_dagger.warm_start`` (which names this very path), the superseded
    SN_Amask_s17 entry's file, and the campaign's ``sh_core.STAND_NET_CKPT`` / ``STAND_NET_SHA256`` (the path and pin
    the stand-height trainer warm-starts from); its untracked source run file, when present, is byte-identical."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    ws = sn["warm_start"]
    assert (ws["id"], ws["file"]) == ("SN_Amask_s17", "model_4000.pt")
    assert ws["sha256"] == "769cf25331402afdfb23756100ec7c215dacf984a321e90a7b39d2969e4941ff"
    assert ws["source"] == "parkour/outputs/omni_plan/stand_network/runs/SN_Amask_s17/model_4000.pt"
    assert "NOT the stand network of record" in ws["role"] and "W[:, 15] zeroed" in ws["role"]
    p = tool.warm_start_path(sn)
    assert p == POLICY / sn["dir"] / ws["file"] and p.is_file(), p
    assert tool.sha256_of(p) == ws["sha256"]
    # the superseded entry's file is this file
    sup = next(x for x in m["superseded"] if x["id"] == "SN_Amask_s17")
    assert [f["sha256"] for f in sup["files"]] == [ws["sha256"]]
    # ... and so is the training block's warm start
    assert ws["sha256"] in sn["training"]["warm_start"]
    # the campaign's trainer pins exactly this path and sha
    core = _load_nobc(SH_CORE, "sh_core_policy_warm_test")
    assert Path(core.STAND_NET_CKPT).resolve() == p.resolve() and core.STAND_NET_SHA256 == ws["sha256"]
    # SH_T17's checkpoint records this file as its warm start
    torch = pytest.importorskip("torch")
    ck = torch.load(str(tool.stand_path(sn)), map_location="cpu", weights_only=False)
    rec = ck["infos"]["sh_dagger"]["warm_start"]
    assert rec["sha256"] == ws["sha256"] and rec["iter"] == 4000
    assert str(rec["source"]).endswith(str(p.relative_to(REPO))), rec["source"]
    src = REPO / ws["source"]
    if src.exists():
        assert tool.sha256_of(src) == ws["sha256"]


def test_stand_network_checkpoint_is_the_height_net_beside_the_walker():
    """infos.sh_dagger (and the predecessor's infos.sn_dagger block): mask 1 as the manifest says, trained beside the
    walking stage the block names (s1_sha256 == that stage's sha), with R's validated constants, the trainer flags of
    record (deadband 0.25 in the loop, the trim branch at the saturated label), warm-started from the superseded A-mask
    s17 file, under the height contract and clip the manifest's deployment.constants state; only the actor backbone
    differs from the walker's checkpoint and the estimator is the walker's, bitwise."""
    torch = pytest.importorskip("torch")
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    w = tool.stage_by_dir(m, sn["walker"]["stage"])
    c, tr = sn["deployment"]["constants"], sn["training"]
    ck = torch.load(str(tool.stand_path(sn)), map_location="cpu", weights_only=False)
    sh, snd = ck["infos"]["sh_dagger"], ck["infos"]["sn_dagger"]
    for info in (sh, snd):
        assert type(info["stand_mask"]) is int and info["stand_mask"] == sn["stand_mask"] == 1
        assert info["s1_sha256"] == w["sha256"] == sn["walker"]["sha256"]
        assert (info["seed"], info["iterations"]) == (sn["training_seed"], sn["iterations"])
        assert {"mask": "A-mask", "full": "A-full"}[info["arm"]] == sn["arm"]
    assert ck["iter"] == sn["iterations"]
    assert sh["s1"].endswith(w["source"])
    assert (sh["mode"], sh["trainable"]) == ("dagger", "actor.actor_backbone")
    # the trainer flags of record: the residual deadband in the loop, the trim branch at the saturated cam label
    assert sh["residual_deadband"] == tr["residual_deadband"] == c["residual_deadband"] == 0.25
    assert (sh["trim_branch"], sh["trim_min"]) == (tr["trim_branch"], tr["trim_min"]) == (True, 0.0999)
    assert tr["flags"].split() == ["--mode", sh["mode"], "--arm", sh["arm"], "--seed", str(sh["seed"]), "--iterations",
                                   str(sh["iterations"]), "--deadband", "0.25", "--trim-branch", "--trim-min", "0.0999"]
    assert tr["flags"] in " ".join(sn["trained_in"].split())
    sc = _load_nobc(SN_COMMON, "sn_common_policy_test")
    assert sh["r_knob"] == snd["r_knob"] == sc.RESTANCE_D3 and sh["r_knob"] in sn["trained_in"]
    # the warm start is the superseded stand net of record (A-mask s17), byte-identified, and the bundled warm_start
    sup = next(x for x in m["superseded"] if x["id"] == "SN_Amask_s17")
    assert sh["warm_start"]["sha256"] == sup["files"][0]["sha256"] and sup["files"][0]["sha256"] in tr["warm_start"]
    assert sh["warm_start"]["sha256"] == sn["warm_start"]["sha256"]
    assert sh["warm_start"]["zeroed"] == "actor.actor_backbone.0.weight[:, 15]" and sh["warm_start"]["zeroed"] in tr[
        "warm_start"]
    # the height contract and clip it was trained under are the manifest's constants
    hc, rs, clip = sh["height_command"], sh["height_command"]["restore"], sh["clip"]
    assert (hc["column"], hc["h_settled"], hc["h_range"], hc["rate_m_s"], hc["restore_zero_cols"], hc["stand_mask"]) == (
        c["height_col"], c["h_settled_m"], [c["h_min_m"], c["h_max_m"]], c["rate_m_s"], c["restore_zero_cols"], 1)
    assert hc["encoding"] == f"(h_cmd_limited - {c['h_settled_m']}) / {c['height_obs_scale']}"
    assert (rs["engage_m"], rs["rod_tol_m"], rs["rate_m_s"], rs["h_min"], rs["h_max"]) == (
        c["restore_engage_m"], c["restore_rod_tol_m"], c["rate_m_s"], c["h_min_m"], c["h_max_m"])
    assert abs(rs["timeout_s"] - c["restore_timeout_s"]) < 1e-9
    assert rs["settle_steps"] == round(c["restore_settle_s"] / rs["dt"]) == 10 and rs["timeout_steps"] == 266
    assert (clip["harness_clip_actions"], clip["env_clip_actions"], clip["config_clip_actions"], clip["s1_clamp"],
            clip["residual_clamp"], clip["stand_rod_clamp"], clip["stand_cam_clamp"], clip["residual_deadband"]) == (
        c["harness_clip_actions"], c["harness_clip_actions"], c["config_clip_actions"], c["s1_clamp"],
        c["residual_clamp"], c["stand_rod_clamp"], c["stand_cam_clamp"], c["residual_deadband"])
    assert "0.25 on hip / knee" in clip["stand_action"] and "rods +-4.8, cams +-1" in clip["stand_action"]
    # only the actor backbone differs from the walker; the estimator is the walker's, bitwise
    s1 = torch.load(str(tool.stage_path(w)), map_location="cpu", weights_only=False)
    a, b = ck["model_state_dict"], s1["model_state_dict"]
    assert set(a) == set(b)
    assert all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    changed = {k for k in a if not torch.equal(a[k], b[k])}
    assert changed and all(k.startswith("actor.actor_backbone.") for k in changed), sorted(changed)
    ea, eb = ck["estimator_state_dict"], s1["estimator_state_dict"]
    assert set(ea) == set(eb) and all(torch.equal(ea[k], eb[k]) for k in ea)
    arch = sn["architecture"]
    assert a["critic.0.weight"].shape[1] == m["obs_contract"]["width"] == arch["obs"] == 1149
    assert a["actor.actor_backbone.6.weight"].shape[0] == arch["action"] == 18
    assert "std_raw" in a and "std" not in a          # soft std, as the walker (KRABBY_STD_PARAM=soft)


def test_stand_network_contract_is_the_campaign_core():
    """Every number of deployment.constants is the stand-height campaign's sh_core (the ONE implementation the trainer,
    the launcher, the table test and the recorder import), the posture endpoints the contract states are sh_core's
    q map, and the action path behaves as stated: hard 0.25 deadband on the clamped hip / knee residual only, before
    the posture; column 15 = 0 at 1.064 (zero-height mode)."""
    torch = pytest.importorskip("torch")
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    dep = sn["deployment"]
    c = dep["constants"]
    core = _load_nobc(SH_CORE, "sh_core_policy_test")
    assert (c["h_settled_m"], c["h_min_m"], c["h_max_m"]) == (core.H_SETTLED, core.H_MIN, core.H_MAX)
    assert (c["rate_m_s"], c["height_col"], c["height_obs_scale"], c["q_grid_step_m"]) == (
        core.HEIGHT_RATE_M_S, core.HEIGHT_COL, core.HEIGHT_OBS_SCALE, core.Q_GRID_STEP_M)
    assert (c["restore_engage_m"], c["restore_rod_tol_m"], c["restore_settle_s"]) == (
        core.RESTORE_ENGAGE_M, core.RESTORE_ROD_TOL_M, core.RESTORE_SETTLE_S)
    assert abs(c["restore_timeout_s"] - core.RESTORE_TIMEOUT_S) < 1e-9
    assert abs(c["restore_timeout_s"] - ((c["h_settled_m"] - c["h_min_m"]) / c["rate_m_s"] + 1.0)) < 1e-9
    assert tuple(c["restore_zero_cols"]) == core.RESTORE_ZERO_COLS
    assert (c["residual_clamp"], c["stand_rod_clamp"], c["stand_cam_clamp"], c["s1_clamp"],
            c["harness_clip_actions"]) == (core.RESIDUAL_CLIP, core.RAW_CLIP, core.CAM_CLIP, core.S1_CLIP,
                                           core.HARNESS_CLIP)
    assert c["residual_deadband"] == core.RESIDUAL_DEADBAND_PROBE == 0.25
    cols = re.fullmatch(r"(\d+)-(\d+)", str(c["deadband_cols"]))  # an inclusive range, rendered as written
    assert cols, c["deadband_cols"]
    assert (int(cols.group(1)), int(cols.group(2)) + 1) == (core.DEADBAND_ACT.start, core.DEADBAND_ACT.stop)
    # the posture map as the contract states it: anchored at walking height, its endpoints
    hs = torch.tensor([c["h_min_m"], c["h_settled_m"], c["h_max_m"]], dtype=torch.float64)
    qh, qk = core.q_map(hs, dtype=torch.float64)
    assert qh[1].item() == 0.0 and qk[1].item() == 0.0
    for i in (0, 2):
        assert f"q({hs[i].item():g}) = ({qh[i].item():+.3f}, {qk[i].item():+.3f})" in dep["posture_map"], i
    assert core.encode_height(torch.tensor([c["h_settled_m"]], dtype=torch.float64)).item() == 0.0
    # the action path: clamp to +-1, then the HARD deadband on hip / knee (cols 6-17) only, then + q (0 at 1.064)
    a = torch.zeros(1, 18)
    a[0, 0], a[0, 6], a[0, 7], a[0, 12], a[0, 13] = 0.1, 0.2, 1.7, -0.3, -0.249
    s, q = core.stand_action(a, c["h_settled_m"], deadband=c["residual_deadband"])
    want = torch.zeros(1, 18)
    want[0, 0], want[0, 7], want[0, 12] = 0.1, 1.0, -0.3
    assert torch.equal(q, torch.zeros(1, 18)) and torch.equal(s, want), s
    # away from 1.064 the posture is added after the deadband and the rods stay inside +-4.8
    s2, q2 = core.stand_action(torch.zeros(1, 18), c["h_min_m"], deadband=c["residual_deadband"])
    assert torch.equal(s2, q2) and s2[0, 6:].abs().max().item() <= c["stand_rod_clamp"]
    for k in ("switch", "entry_exit", "input", "forward", "action", "posture_map", "height_command", "restore",
              "zero_height", "scope", "reference_implementation", "caveat_switch_keying", "caveat_height_source"):
        assert dep.get(k), k


def test_stand_network_routing_env_is_the_walkers():
    """Both networks run in one process under the walker's carry env: the block's env_required is exactly the
    walker's carry values, its env_off knobs are off there, the launcher's D5 rule (sn_common.env_reasons, which the
    stand-height launcher applies unchanged) accepts that env, the SH_T17 raise / lower video ran under exactly the
    walker's eval env, and every script the README's commands name exists; a re-run uses fresh names, the trained
    deadband, mask 1 and the bundled file."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    w = tool.stage_by_dir(m, sn["walker"]["stage"])
    carry = tool.carry_env(m, w)
    dep = sn["deployment"]
    for k, v in dep["env_required"].items():
        assert carry.get(k) == str(v), (k, carry.get(k), v)
    fixed = m["obs_contract"]["eval_env"]["fixed"]
    for k in dep["env_off"]:
        assert str(carry.get(k, fixed.get(k, ""))).lower() in ("", "0"), k
    sc = _load_nobc(SN_COMMON, "sn_common_policy_env_test")
    assert sc.env_reasons(tool.eval_env(m, w)) == []
    assert set(sc.ENV4) <= set(dep["env_required"])
    assert {sc.STAND_MASK_VAR, sc.STAND_PROXY_VAR, sc.RESTANCE_VAR} == set(dep["env_off"])
    video = json.loads((SH_CAMPAIGN / "video_sh_raise_T17_sanity.json").read_text())
    assert video["env_vars"] == tool.eval_env(m, w)
    assert (video["stand_checkpoint_sha256"], video["checkpoint_sha256"]) == (sn["sha256"], w["sha256"])
    assert (video["stand_mask"], video["residual_deadband"], video["restore"]) == (1, 0.25, "armed")
    run = sn["evaluate"]
    sdir = PKG / run["scripts"]
    assert sdir == SH_CAMPAIGN / "scripts"
    cmds = [c["run"] for c in run["commands"]]
    named = set(re.findall(r"\$S/(\S+)", "\n".join(cmds)))
    assert named == {"sh_eval_rows.sh", "sh_table_test.sh", "sh_read.py", "sh_observe_read.py", "sh_record_video.sh"}
    for f in named | {"sh_eval_crab_hex_gait.py", "sh_core.py", "sh_video_core.py", "sh_record_video.py"}:
        assert (sdir / f).is_file(), f
    assert (SH_CAMPAIGN / "head_SH_T17_spec.json").is_file()
    for cmd in cmds:
        if "$CK" in cmd:
            assert f"--deadband {sn['deployment']['constants']['residual_deadband']}" in cmd, cmd
    assert sum(bool(re.search(r"\$CK 1 ", cmd)) for cmd in cmds) == 3          # rows: the network, mask 1
    assert all("--stand-mask 1" in ln for cmd in cmds for ln in cmd.splitlines() if "sh_record_video.sh" in ln)
    taken = {h["id"] for h in sn["evals"]["heads"]}
    assert not {run["name"], run["name"] + "z"} & (taken | {h + "z" for h in taken}), "a re-run must use FRESH names"


#: gate-table row key -> (gate, kind, clause (regex for max / min / margin; '{k}' = env seed), scale, rounding tolerance)
_K_ROWS = {
    "k1_median": ("K1", "max", r"median bin\d es{k} <= 0.02", 1000.0, 0.05),
    "k1_p90": ("K1", "max", r"p90 bin\d es{k} <= 0.04", 1000.0, 0.05),
    "k2_reach": ("K2", "one", "time-to-2cm excess p90 es{k} <= 0 s", 1.0, 5e-4),
    "k2_falls": ("K2", "one", "stand falls es{k} <= max(1, R_h's)", 1.0, 0.0),
    "k3_stood": ("K3", "one", "STOOD rate tables es{k} >= R_h's - max(0.05, 2 SE)", 1.0, 5e-4),
    "k3_falls": ("K3", "one", "falls/100 tables es{k} <= max(1, R_h's)", 1.0, 5e-3),
    "k4_miss": ("K4", "one", "miss es{k} <= max(1, R_h's)", 1.0, 0.0),
    "k4_exit_falls": ("K4", "one", "exit falls es{k} <= max(1, R_h's)", 1.0, 0.0),
    "k4_onset": ("K4", "one", "onset excess p90 es{k} <= 0 s", 1.0, 5e-4),
    "k5": ("K5", "verdict", None, None, None),
    "k6_margin": ("K6", "margin", r"(homed|settled) share bin\d es{k} >= .*", 1.0, 5e-4),
    "k6_homed_min": ("K6", "min", r"homed share bin\d es{k} >= .*", 1.0, 5e-4),
    "k6_steps": ("K6", "one", "stillness steps/s pooled es{k} <= 0.5", 1.0, 5e-4),
    "k6_taps": ("K6", "one", "stillness taps/s pooled es{k} <= 1", 1.0, 5e-4),
}
_GATES = ("K1", "K2", "K3", "K4", "K5", "K6")


def _k_clauses(g: dict, kind: str, pat, k: int) -> list[dict]:
    """The clauses of env seed ``k`` a gate-table row reads: the named one, or every per-bin match of the regex."""
    if kind == "one":
        return [g["clauses"][pat.format(k=k)]]
    cl = [c for n, c in g["clauses"].items() if re.fullmatch(pat.format(k=k), n)]
    assert len(cl) in (5, 10), (pat, k, len(cl))
    return cl


def _k_cell(g: dict, kind: str, pat, scale):
    """A gate-table cell recomputed from one role's gate read: the verdict, or [es1, es2] of the row's statistic."""
    if kind == "verdict":
        return g["verdict"]
    out = []
    for k in (1, 2):
        cl = _k_clauses(g, kind, pat, k)
        if kind == "one":
            v = cl[0]["value"]
            out.append(v if isinstance(v, str) or scale == 1.0 else v * scale)
            continue
        vals = [c["value"] for c in cl]
        if kind == "max":
            out.append("inf" if "inf" in vals else max(vals) * scale if scale != 1.0 else max(vals))
        elif kind == "min":
            out.append(min(vals))
        else:
            out.append(min(c["value"] - c["line"] for c in cl))
    return out


def _k_close(got, want, tol, where):
    if isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (where, got, want)
        for g, x in zip(got, want):
            _k_close(g, x, tol, where)
    elif isinstance(want, str):
        assert got == want, (where, got, want)
    else:
        assert isinstance(got, (int, float)) and not isinstance(got, bool), (where, got)
        assert abs(float(got) - float(want)) <= tol + 1e-9, (where, got, want)


def _num_str(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def test_stand_network_gate_table_matches_the_campaign_reads():
    """Every cell of the manifest's K1-K6 table is the campaign's read at the manifest's rounding: the control columns
    from step4b_read.json (and identical in every head read), the head columns from head_SH_T{18,16,17}.json; the
    lines name the clause lines; each head read carries its own checkpoint sha, deadband 0.25 trained and evaluated,
    and S1 s18; the control table lands; the verdict row, the adoption-grade counts and the tie-break facts recompute
    from the reads."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    sn = _stand(m)
    ev = sn["evals"]
    gt = ev["gate_table"]
    step4b = json.loads((SH_CAMPAIGN / "step4b_read.json").read_text())
    heads = {h: json.loads((SH_CAMPAIGN / f"head_{h}.json").read_text()) for h in SH_HEADS}
    # the control table lands, and its cells are the same in every head read
    assert step4b["mismatches"] == [] and step4b["pending"] == []
    assert len(step4b["control_table"]) == 14 and all(r["lands"] == "yes" for r in step4b["control_table"])
    for h, d in heads.items():
        assert d["mismatches"] == [] and d["pending"] == [], h
        for K, roles in step4b["gates"].items():
            for role, g in roles.items():
                assert d["gates"][K][role]["clauses"] == g["clauses"], (h, K, role)
    controls = ("R_h", "R_old", "Amask_old", "S1", "no_restore", "posture")
    assert [c["id"] for c in gt["columns"]] == [*controls, *SH_HEADS]

    def read(col, K):
        return heads[col]["gates"][K][col] if col in heads else step4b["gates"][K].get(col)

    assert [r["key"] for r in gt["rows"]] == list(_K_ROWS)
    for r in gt["rows"]:
        gate, kind, pat, scale, tol = _K_ROWS[r["key"]]
        assert r["gate"] == gate, r["key"]
        for col in gt["columns"]:
            g = read(col["id"], gate)
            if g is None:
                assert r[col["id"]] is None, (r["key"], col["id"])
                continue
            _k_close(r[col["id"]], _k_cell(g, kind, pat, scale), tol, (r["key"], col["id"]))
        if kind in ("one", "max"):
            # the line the row states is the clauses' line (both env seeds; per bin for K1)
            g = read("SH_T17", gate)
            lines = {_num_str(c["line"] * scale) for k in (1, 2) for c in _k_clauses(g, kind, pat, k)}
            assert lines and all(x in r["line"] for x in lines), (r["key"], lines, r["line"])
    # the verdict row: every column's failing gates (known-bad controls fail exactly the gates the control table expects)
    for col in gt["columns"]:
        cid = col["id"]
        failing = [K for K in _GATES if read(cid, K) is not None and read(cid, K)["verdict"] != "PASS"]
        want = ("FAIL " + ", ".join(failing)) if failing else "PASS"
        assert gt["verdicts"][cid].startswith(want), (cid, gt["verdicts"][cid], want)
        if cid in controls:
            expected = {r["gate"] for r in step4b["control_table"] if r["role"] == cid and r["expected"] == "FAIL"}
            assert expected == set(failing), (cid, expected, failing)
    # each head read: its own checkpoint, the trained deadband, the walker of record; exactly SH_T17 passes everything
    hsha = {h["id"]: h for h in ev["heads"]}
    assert set(hsha) == set(SH_HEADS) and all(hsha[h]["column"] == h for h in SH_HEADS)
    for h, d in heads.items():
        hd = d["heads"][h]
        assert hd["stand_sha256"] == [hsha[h]["sha256"]], h
        assert (hd["residual_deadband"], hd["trained_residual_deadband"]) == (0.25, 0.25), h
        assert d["k5"]["refs_s1_sha256"] == sn["walker"]["sha256"], h
        assert hd["all_pass"] == (h == "SH_T17") and hd["failed"] == ([] if h == "SH_T17" else ["K4"]), h
    adopted = [h for h in ev["heads"] if h.get("adopted")]
    assert [h["id"] for h in adopted] == [sn["id"]] and adopted[0]["sha256"] == sn["sha256"]
    assert adopted[0]["source"] == sn["source"]
    # the SH_T17 stamps the manifest names are the ones its read used
    rd = heads["SH_T17"]["reads"]["SH_T17"]
    for k in (1, 2):
        for row in ("hwalkstop__omni", "hraise__omni"):
            assert rd[f"es{k}"][row]["stamp"].endswith(adopted[0]["stamps"][f"{row}@{k}"]), (row, k)
    # adoption grade: per-seed K4 restore-miss counts and the pooled totals the note states
    miss = {h: _k_cell(read(h, "K4"), "one", _K_ROWS["k4_miss"][2], 1.0) for h in SH_HEADS}
    assert miss == {"SH_T18": [1, 3], "SH_T16": [1, 2], "SH_T17": [0, 1]}
    pooled, rh = sum(sum(v) for v in miss.values()), sum(_k_cell(read("R_h", "K4"), "one", _K_ROWS["k4_miss"][2], 1.0))
    assert f"{pooled} / {800 * 2 * len(SH_HEADS)}" in ev["adoption_grade"] and f"{rh} / 1600" in ev["adoption_grade"]
    # the post-hoc tie-break facts: the only all-pass head, and the lowest K3 table fall rate
    k3f = {h: sum(_k_cell(read(h, "K3"), "one", _K_ROWS["k3_falls"][2], 1.0)) for h in SH_HEADS}
    assert min(k3f, key=k3f.get) == sn["id"]


def test_readme_has_the_stand_network_section():
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    text = (POLICY / "README.md").read_text()
    assert text == tool.readme_text(m)
    path = f"{sn['dir']}/{sn['file']}"
    assert "## Stand network of record" in text and "### Deployment contract (as evaluated)" in text
    assert f"**Stand network of record:** [`{path}`]({path})" in text
    assert "stand [`stand_net/`](stand_net/) (`model_5000.pt`)" in text
    # the superseded A-mask s17 file is named ONLY as the bundled warm-start ancestor, never as the net of record
    ws_lines = [line for line in text.splitlines() if "model_4000.pt" in line]
    assert ws_lines == [line for line in ws_lines
                        if line.startswith("- **Warm-start ancestor (bundled, NOT the stand network of record):** "
                                           "[`stand_net/model_4000.pt`](stand_net/model_4000.pt)")], ws_lines
    assert len(ws_lines) == 1 and sn["warm_start"]["sha256"] in ws_lines[0] and sn["warm_start"]["role"] in ws_lines[0]
    assert sn["sha256"] in text and sn["walker"]["sha256"] in text and sn["source"] in text
    for s in ("stand_net_input", "hist_encoding=True", "CAVEAT (switch keying)", "CAVEAT (height source)",
              "**Zero-height mode:**", "**Restore before hand-back:**", "**Height command:**", "**Posture map q(h):**",
              "deadband(clamp(a_net, +-1), 0.25)", "  - **residual_deadband:** 0.25", "**Adoption grade:**",
              "### Gates K1-K6 per head", "- **Training:**", "  - **trim_min:** 0.0999"):
        assert s in text, s
    assert f"CK=$PWD/$P/policy/{path}" in text
    assert "bash $S/sh_eval_rows.sh --deadband 0.25 SH_T17_re stand_net manifest armed $CK 1 \\" in text
    assert "bash $S/sh_eval_rows.sh --deadband 0.25 SH_T17_rez stand_net zero armed $CK 1 \\" in text
    assert "bash $S/sh_record_video.sh --prefix SH_T17_re_video -- --stand-ckpt $CK --stand-mask 1 --deadband 0.25" in text
    assert ("| Gate | Statistic | Line | R_h (known-good) | R, old +-1 path | A-mask s17, old +-1 path | S1 alone | "
            "no restore | posture only | SH_T18 | SH_T16 | SH_T17 (adopted) |") in text
    assert ("| K4 | restore misses (of 800 restore-required exits) | <= max(1, R_h's) = 1 | 1 / 1 | — | — | — | "
            "788 / 784 | — | 1 / 3 | 1 / 2 | 0 / 1 |") in text
    assert "| K1 | worst-bin median \\|h - h_cmd\\| (mm), 5 bins | <= 20 mm in every bin | 3.0 / 2.5 |" in text
    assert "| K2 | p90 time to 2 cm minus allowance dh / 0.08 + 1 s (s)" in text and "| inf / inf | inf / inf |" in text
    for c in sn["caveats"]:
        assert c in text
    for f in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "two_network_export", "stand_ground_getup",
              "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
              "stand_knee_hold_only"):
        assert f"### `{f}` --" in text


def test_stand_followups_track_the_new_head():
    """The stand follow-ups name SH_T17 (revised 2026-10-15) and keep the schema; the new ones come from the
    stand-height REPORT 10 and cite it."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    fu = {f["id"]: f for f in m["followups"]}
    for k in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "two_network_export"):
        assert "2026-10-15" in fu[k]["recorded"] and "SH_T17" in fu[k]["detail"], k
    for k in ("stand_ground_getup", "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
              "stand_knee_hold_only"):
        assert fu[k]["recorded"].startswith("2026-10-15"), k
    for k in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "two_network_export", "stand_ground_getup",
              "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
              "stand_knee_hold_only"):
        assert {"id", "recorded", "headline", "detail", "evidence", "levers"} <= set(fu[k]), k
        assert any("2026-10-13_stand_height" in e for e in fu[k]["evidence"]), k
        for e in fu[k]["evidence"]:
            rel = e.split(" (")[0]
            assert (PKG / rel).exists(), (k, rel)
    assert "SUPERSEDED" in fu["stand_trim_underfit"]["detail"] and "0.351" in fu["stand_trim_underfit"]["detail"]
    assert "49 / 62 %" in fu["stand_ground_getup"]["detail"] and "97 / 98 %" in fu["stand_ground_getup"]["detail"]


def test_readme_without_the_stand_network_is_unchanged():
    """The stand section is purely additive: a manifest without the block renders the same README minus the new
    section, its header line and its maintenance bullet; only the heads line changes (it gains `stand`)."""
    tool = _tool()
    m = tool.load_manifest()
    bare = copy.deepcopy(m)
    del bare["stand_network"]
    bare["heads"].pop("stand")
    with_sn = tool.readme_text(m).splitlines()
    without = tool.readme_text(bare).splitlines()
    # (follow-up text may still mention the stand network: it is manifest content, not the tool's section)
    assert "## Stand network of record" not in without
    assert not any(line.startswith("**Stand network of record:**") or "stand [`stand_net/`]" in line
                   or line.startswith("- Stand network (`stand_network:`") for line in without)
    ops = [op for op in difflib.SequenceMatcher(a=without, b=with_sn, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert ops and all(tag in ("insert", "replace") for tag, *_ in ops), ops
    removed = [line for tag, i1, i2, _, _ in ops if tag == "replace" for line in without[i1:i2]]
    heads_old = [line for line in without if line.startswith("**Head of record per phase:**")]
    assert removed == heads_old
    heads_new = [line for line in with_sn if line.startswith("**Head of record per phase:**")]
    assert heads_new == [heads_old[0][:-1] + ", stand [`stand_net/`](stand_net/) (`model_5000.pt`)."]


def _mutated_manifest(tmp_path: Path, mutate) -> Path:
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    mutate(m)
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(m, sort_keys=False))
    return p


@pytest.mark.parametrize("mutate, why", [
    (lambda m: m["stand_network"].update(dir="6_p2a_field"), "collides with a stage dir"),
    (lambda m: m["stand_network"].update(dir="rsi"), "collides with an asset dir"),
    (lambda m: m["stand_network"].pop("sha256"), "missing 'sha256'"),
    (lambda m: m["stand_network"].update(sha256="99c377a1"), "full lowercase sha256"),
    (lambda m: m["stand_network"].update(stand_mask=True), "int 0 or 1"),
    (lambda m: m["stand_network"]["walker"].update(sha256="0" * 64), "walker.sha256"),
    (lambda m: m["stand_network"]["walker"].update(stage="7_nope"), "walker.stage"),
    (lambda m: m["heads"].update(stand="nope"), "heads.stand"),
    (lambda m: m.pop("stand_network"), "heads.stand"),
    (lambda m: m["stand_network"].update(deployment="hard switch"), "deployment must be a mapping"),
    (lambda m: m["stand_network"]["deployment"].update(constants=[0.25]), "constants must be a mapping"),
    (lambda m: m["stand_network"]["deployment"]["constants"].update(residual_deadband=1.5), "residual_deadband"),
    (lambda m: m["stand_network"]["deployment"]["constants"].update(residual_deadband=True), "residual_deadband"),
    (lambda m: m["stand_network"]["deployment"]["constants"].update(h_min_m=1.1), "h_min_m < h_settled_m < h_max_m"),
    (lambda m: m["stand_network"]["deployment"]["constants"].pop("h_max_m"), "h_min_m < h_settled_m < h_max_m"),
    (lambda m: m["stand_network"].update(warm_start="model_4000.pt"), "warm_start must be a mapping"),
    (lambda m: m["stand_network"]["warm_start"].pop("source"), "warm_start missing 'source'"),
    (lambda m: m["stand_network"]["warm_start"].pop("role"), "warm_start missing 'role'"),
    (lambda m: m["stand_network"]["warm_start"].update(sha256="769cf253"), "warm_start.sha256 must be a full"),
    (lambda m: m["stand_network"]["warm_start"].update(dir="rsi"), "warm_start lives in the stand network's folder"),
    (lambda m: m["stand_network"]["warm_start"].update(file="model_5000.pt"), "different file from the stand network"),
    (lambda m: m["stand_network"]["warm_start"].update(sha256=m["stand_network"]["sha256"]),
     "different file from the stand network"),
])
def test_stand_network_block_is_validated(tmp_path, mutate, why):
    tool = _tool()
    with pytest.raises(SystemExit, match=re.escape(why)):
        tool.load_manifest(_mutated_manifest(tmp_path, mutate))
