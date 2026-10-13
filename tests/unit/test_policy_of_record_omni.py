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

The STAND NETWORK OF RECORD (``stand_network:``, adopted 2026-10-13: A-mask s17 of experiments/2026-10-11_stand_network,
folder ``stand_net/``) is neither a stage nor an asset: its file must be bundled sha-pinned (and equal its source run
file), its checkpoint must be the A-mask net trained beside the walking stage it names (``infos.sn_dagger``; only the
actor backbone differs from the walker, the estimator is the walker's), its routing env must be the walker's, its gate
table must be the campaign's reads, and adding it must leave the rest of the README untouched.
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
#: the stand-network campaign (its head_*.json gate reads are tracked)
SN_CAMPAIGN = PKG / "experiments" / "2026-10-11_stand_network"
SN_COMMON = SN_CAMPAIGN / "scripts" / "sn_common.py"


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


# ------------------------------------------------------------------------------ the stand network of record
def _stand(m: dict) -> dict:
    assert m.get("stand_network"), "policy manifest declares no stand_network block"
    return m["stand_network"]


def test_stand_network_file_is_bundled_with_the_manifest_sha():
    """The adopted A-mask s17 file sits in its own folder (not a stage / asset dir) with the manifest sha, beside the
    UNCHANGED walking head of record it was trained and gated with."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    assert (sn["dir"], sn["file"]) == ("stand_net", "model_4000.pt")
    assert (sn["id"], sn["arm"], sn["stand_mask"], sn["training_seed"], sn["iterations"]) == (
        "SN_Amask_s17", "A-mask", 1, 17, 4000)
    assert sn["dir"] not in {s["dir"] for s in m["stages"]} | {a["dir"] for a in m["assets"]}
    p = tool.stand_path(sn)
    assert p == POLICY / "stand_net" / "model_4000.pt" and p.is_file(), p
    assert tool.sha256_of(p) == sn["sha256"]
    assert m["heads"]["stand"] == sn["dir"]
    # the walking head of record is unchanged by the adoption, and the block pins exactly that walker
    assert m["current"] == sn["walker"]["stage"] == "6_p2a_field"
    assert sn["walker"]["sha256"] == tool.stage_by_dir(m, "6_p2a_field")["sha256"]
    assert "Adopt A-mask s17 as the stand network of record" in sn["adopted"]


def test_expected_files_include_the_stand_network():
    tool = _tool()
    m = tool.load_manifest()
    files = tool.expected_files(m)
    assert "stand_net/model_4000.pt" in files
    bare = copy.deepcopy(m)
    del bare["stand_network"]
    assert tool.expected_files(bare) == files - {"stand_net/model_4000.pt"}


def test_stand_network_source_is_the_bundled_file():
    """The source run file (untracked) is byte-identical to the bundled copy."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    src = REPO / sn["source"]
    if not src.exists():
        pytest.skip(f"stand-network source run file not on this machine (raw run dirs are untracked): {src}")
    assert tool.sha256_of(src) == sn["sha256"] == tool.sha256_of(tool.stand_path(sn))


def test_stand_network_checkpoint_is_the_a_mask_net_beside_the_walker():
    """infos.sn_dagger: the mask it was trained with is the manifest's (the launcher refuses a mismatch), it was
    trained beside the walking stage the block names (s1_sha256 == that stage's sha), with R's validated constants;
    only the actor backbone differs from the walker's checkpoint and the estimator is the walker's, bitwise."""
    torch = pytest.importorskip("torch")
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    w = tool.stage_by_dir(m, sn["walker"]["stage"])
    ck = torch.load(str(tool.stand_path(sn)), map_location="cpu", weights_only=False)
    info = ck["infos"]["sn_dagger"]
    assert type(info["stand_mask"]) is int and info["stand_mask"] == sn["stand_mask"]
    assert info["s1_sha256"] == w["sha256"] == sn["walker"]["sha256"]
    assert info["s1"].endswith(w["source"])
    assert (info["seed"], info["iterations"], ck["iter"]) == (sn["training_seed"], sn["iterations"], sn["iterations"])
    assert {"mask": "A-mask", "full": "A-full"}[info["arm"]] == sn["arm"]
    assert (info["mode"], info["trainable"]) == ("dagger", "actor.actor_backbone")
    sc = _load(SN_COMMON, "sn_common_policy_test")
    assert info["r_knob"] == sc.RESTANCE_D3 and info["r_knob"] in sn["trained_in"]
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


def test_stand_network_routing_env_is_the_walkers():
    """Both networks run in one process under the walker's carry env: the block's env_required is exactly the
    walker's carry values, its env_off knobs are off there, the campaign launcher's own D5 rule
    (sn_common.env_reasons) accepts that env, and every script the README names exists."""
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
    sc = _load(SN_COMMON, "sn_common_policy_env_test")
    assert sc.env_reasons(tool.eval_env(m, w)) == []
    assert set(sc.ENV4) <= set(dep["env_required"])
    assert {sc.STAND_MASK_VAR, sc.STAND_PROXY_VAR, sc.RESTANCE_VAR} == set(dep["env_off"])
    assert "caveat_switch_keying" in dep
    run = sn["evaluate"]
    sdir = PKG / run["scripts"]
    for f in [run["eval_rows"], run["video"], *run["readers"], "sn_stand_net_eval_crab_hex_gait.py", "sn_common.py"]:
        assert (sdir / f).is_file(), f
    assert set(run["readers"]) <= set(tool.STAND_READER_ARGS)
    assert run["name"] not in {h["id"] for h in sn["evals"]["heads"]}, "a re-run must use a FRESH head name"


#: gate-table row key -> (gate, sn_calib clause name: '{k}' = env seed 1 / 2; a tuple = listed clauses; ('max', ...))
_GATE_CLAUSES = {
    "g1_steps": ("G1", "steps_per_s suite es{k}"),
    "g1_taps": ("G1", "taps_per_s suite es{k}"),
    "g1_clearance": ("G1", "clearance_m suite es{k}"),
    "g1_off_ground": ("G1", "off_ground_share suite es{k}"),
    "g1_hover": ("G1", "hover_episodes_per_100 suite es{k}"),
    "g1_ks": ("G1", "ks_m_s suite es{k}"),
    "g2_homed": ("G2", "homed walkstop es{k}"),
    "g2_settled": ("G2", "settled walkstop es{k}"),
    "g2_settle_s": ("G2", "median settle walkstop es{k} <= probe + 1.0 s"),
    "g2_post_steps": ("G2", "post_settle_steps_per_s walkstop es{k}"),
    "g2_post_taps": ("G2", "post_settle_taps_per_s walkstop es{k}"),
    "g2_post_ks": ("G2", "post_settle_ks_m_s walkstop es{k}"),
    "g3_walkstop_falls": ("G3", "walkstop stand falls (count) es{k} <= max(1, Z1's count)"),
    "g3_suite_falls": ("G3", "suite stand falls es{k} == 0"),
    "g3_exit_falls": ("G3", "walkstop exit-handover falls within 2 s es{k} <= max(1, R's)"),
    "g4_field_hard": ("G4", "field_hard es1+2+3+4 completion >= REFS - max(0.05, 2 SE)"),
    "g4_field_turn": ("G4", "field_turn es1 completion >= REFS - max(0.05, band)"),
    "g4_stand_falls": ("G4", ("field_turn__omni@1 stand-fall rate <= HOLD + 2 SE(diff)",
                              "field_hard__omni@1-4 stand-fall rate <= HOLD + 2 SE(diff)")),
    "g5_onset_s": ("G5", "walkstop onset median es{k} <= REFS + max(0.3, band)"),
    "g5_stalls": ("G5", "walkstop onset stalls es{k} <= REFS + 1"),
    "g5_height_m": ("G5", ("max", "suite es1 |body height - REFS| <= 0.03", "walkstop es1 |body height - REFS| <= 0.03",
                           "suite es2 |body height - REFS| <= 0.03", "walkstop es2 |body height - REFS| <= 0.03")),
}


def _clause_value(clauses: dict, spec):
    if isinstance(spec, tuple) and spec[0] == "max":
        return max(clauses[c]["value"] for c in spec[1:])
    if isinstance(spec, tuple):
        return [clauses[c]["value"] for c in spec]
    if "{k}" in spec:
        out = []
        for k in (1, 2):
            c = clauses[spec.format(k=k)]
            out.append("n/a" if c["state"] == "na" else c["value"])
        return out
    return clauses[spec]["value"]


def _assert_close(got, want, where):
    if isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (where, got, want)
        for g, x in zip(got, want):
            _assert_close(g, x, where)
    elif want is None or isinstance(want, str):
        assert got == want, (where, got, want)
    else:
        # cells are the read rounded to 3 decimals (4 for field_hard / height), half-up as REPORT.md prints them
        assert isinstance(got, (int, float)) and abs(float(got) - float(want)) <= 5e-4 + 1e-9, (where, got, want)


def test_stand_network_gate_table_matches_the_campaign_reads():
    """Every cell of the manifest's per-seed gate table is the campaign's sn_calib read (head_SN_Amask_s{18,16,17}.json;
    the REFS / HOLD / R columns from the s17 read's reference rows, = calib.md), at the manifest's rounding; each head's
    stamps carry its own checkpoint sha and mask 1; every A-mask head passes G1-G5; the verdict row and the worst-case
    G2 margins (the post-hoc tie-break that names s17) recompute from the reads."""
    m = yaml.safe_load((POLICY / "manifest.yaml").read_text())
    sn = _stand(m)
    gt = sn["evals"]["gate_table"]
    heads = {h["column"]: h for h in sn["evals"]["heads"]}
    assert set(heads) == {"s18", "s16", "s17"}
    adopted = [h for h in heads.values() if h.get("adopted")]
    assert [h["id"] for h in adopted] == [sn["id"]] and adopted[0]["sha256"] == sn["sha256"]
    assert adopted[0]["source"] == sn["source"]
    reads = json.loads((SN_CAMPAIGN / "head_SN_Amask_s17.json").read_text())["rows"]
    rows = {c: reads[c] for c in ("REFS", "HOLD", "R")}
    for col, h in heads.items():
        d = json.loads((SN_CAMPAIGN / f"head_{h['id']}.json").read_text())["rows"][h["id"]]
        assert d["kind"] == "stand_net"
        assert {tuple(v) for v in d["stand_net_arm"].values()} == {(h["sha256"], 1)}, h["id"]
        assert {g: v["verdict"] for g, v in d["gates"].items()} == {g: "PASS" for g in ("G1", "G2", "G3", "G4", "G5")}
        fh = d["gates"]["G4"]["clauses"][_GATE_CLAUSES["g4_field_hard"][1]]["line"]
        assert abs(fh - h["field_hard_line"]) < 5e-5, h["id"]
        cl = d["gates"]["G2"]["clauses"]
        steps = 0.5 - max(cl[f"post_settle_steps_per_s walkstop es{k}"]["value"] for k in (1, 2))
        taps = 1.0 - max(cl[f"post_settle_taps_per_s walkstop es{k}"]["value"] for k in (1, 2))
        assert (h["worst_margin"]["post_settle_steps"], h["worst_margin"]["post_settle_taps"]) == (
            round(steps, 3), round(taps, 3)), h["id"]
        rows[col] = d
    assert {c["id"] for c in gt["columns"]} == set(rows)
    assert [r["key"] for r in gt["rows"]] == list(_GATE_CLAUSES)
    for r in gt["rows"]:
        gate, spec = _GATE_CLAUSES[r["key"]]
        assert r["gate"] == gate, r["key"]
        for col, d in rows.items():
            _assert_close(r[col], _clause_value(d["gates"][gate]["clauses"], spec), (r["key"], col))
    for col, d in rows.items():
        failing = [g for g, v in d["gates"].items() if v["verdict"] != "PASS"]
        want = ("FAIL " + ", ".join(failing)) if failing else "PASS"
        assert gt["verdicts"][col].startswith(want), (col, gt["verdicts"][col], want)
    # the post-hoc tie-break (widest worst-case margin on the binding G2 clauses) names the adopted head
    for k in ("post_settle_steps", "post_settle_taps"):
        assert max(heads.values(), key=lambda h: h["worst_margin"][k])["id"] == sn["id"], k


def test_readme_has_the_stand_network_section():
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    text = (POLICY / "README.md").read_text()
    assert text == tool.readme_text(m)
    path = f"{sn['dir']}/{sn['file']}"
    assert "## Stand network of record" in text and "### Deployment contract (as evaluated)" in text
    assert f"**Stand network of record:** [`{path}`]({path})" in text
    assert "stand [`stand_net/`](stand_net/) (`model_4000.pt`)" in text
    assert sn["sha256"] in text and sn["walker"]["sha256"] in text and sn["source"] in text
    assert "stand_net_input" in text and "hist_encoding=True" in text and "CAVEAT (switch keying)" in text
    assert f"CK=$PWD/$P/policy/{path}" in text
    assert f"bash $S/sn_eval_rows.sh {sn['evaluate']['name']} $CK 1 \\" in text
    assert "bash $S/sn_record_video.sh" in text
    assert ("| Gate | Statistic | Line | REFS | HOLD | R | A-mask s18 | A-mask s16 | A-mask s17 (adopted) |"
            in text)
    assert "| G2 | post-settle taps/s | <= 1 | — / — | — / — | 0 / 0 | 0.764 / 0.855 | 0.882 / 0.816 | 0.743 / 0.745 |" in text
    assert "| G4 | field_hard es1-4 pooled (n 400) |" in text and "| 0.8125 | 0.81 | 0.825 | 0.7825 | 0.81 | 0.805 |" in text
    for c in sn["caveats"]:
        assert c in text
    for f in ("stand_switch_keying", "stand_trim_underfit", "two_network_export"):
        assert f"### `{f}` --" in text


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
    assert heads_new == [heads_old[0][:-1] + ", stand [`stand_net/`](stand_net/) (`model_4000.pt`)."]


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
    (lambda m: m["stand_network"].update(sha256="769cf253"), "full lowercase sha256"),
    (lambda m: m["stand_network"].update(stand_mask=True), "int 0 or 1"),
    (lambda m: m["stand_network"]["walker"].update(sha256="0" * 64), "walker.sha256"),
    (lambda m: m["stand_network"]["walker"].update(stage="7_nope"), "walker.stage"),
    (lambda m: m["heads"].update(stand="nope"), "heads.stand"),
    (lambda m: m.pop("stand_network"), "heads.stand"),
])
def test_stand_network_block_is_validated(tmp_path, mutate, why):
    tool = _tool()
    with pytest.raises(SystemExit, match=re.escape(why)):
        tool.load_manifest(_mutated_manifest(tmp_path, mutate))
