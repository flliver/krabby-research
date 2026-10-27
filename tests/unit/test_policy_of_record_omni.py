"""The omni-walk POLICY OF RECORD (``crab_hex_omni_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through. Since the
2026-10-24 bake (user decision at the STOP-END of experiments/2026-10-16_task4_pipeline, CHARTER "2026-10-24, STOP-END:
user decisions" (3), "Supersede") that is the Task 4 deliverable run F19 at seed 19: f19_1_s1 (preset 1b) -> f19_2_c4 ->
f19_3_c5 -> f19_4_c6 -> f19_5_c6x_clockfree (preset 1a) -> f19_6_p2a_field -> f19_7_p3a_payload90 ->
f19_8_p3b_payload180 (preset 2a; = T*, current), each with its overlay from F19/candidates.yaml, then the depth student
f19_9_student (kind ``distill``; its 122 MB checkpoint is NOT bundled -- ``bundled: false``, cited by sha256 -- and its
folder holds the ONNX deploy export + io json). The shared training assets (``assets:``, the omni RSI bank) stay. Every
declared file must be present, tracked by git, sha-pinned to the manifest, and the only checkpoint in its folder; the
README is generated from the manifest by ``experiments/tools/bundle_policy.py --sync``. The c6x and later heads were
trained with the gait clock hidden, and their observation contract must say so. Every stage's eval environment must be
exactly what the driver's suite eval builds from that stage's preset + overlay, the tracked F19 env file and (when on
disk) its eval stamp's run_meta.

The STAND NETWORK OF RECORD (``stand_network:``, folder ``f19_stand_net/``) is SH_T4F19_sh_s19, trained and gated beside
T*, with SN_T4F19_sn_s19 its bundled warm-start ancestor; its contract constants must be the stand-height campaign's
``sh_core`` and the checkpoint's; its routing env and walker override must be T*'s; its K1-K6 table must be the F19 stand
reads (sh_read.json / sh_lazy_rh_read.json) and the superseded SH_T17's (head_SH_T17.json).

SUPERSEDED, KEPT: nothing was deleted at the 2026-10-24 bake. The seed-18 lineage (1_fb6k .. 6_p2a_field, S1 s18), SH_T17
(stand_net/model_5000.pt) and A-mask s17 (stand_net/model_4000.pt -- the fixed path sh_core and the stand campaign tests
load) stay in ``policy/`` as ``superseded[].files``: present, sha-pinned, byte-identical to the files they were baked
with, and their full records kept verbatim (equal to the manifest at ``record_commit``). W (superseded 2026-10-06) left
the folder and stays recoverable from git history.
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

#: the Task 4 campaign and its deliverable run F19 (every record below is tracked)
T4 = PKG / "experiments" / "2026-10-16_task4_pipeline"
F19 = T4 / "F19"
STAGES = ["f19_1_s1", "f19_2_c4", "f19_3_c5", "f19_4_c6", "f19_5_c6x_clockfree", "f19_6_p2a_field",
          "f19_7_p3a_payload90", "f19_8_p3b_payload180", "f19_9_student"]
#: the F19 pipeline's stage id of each policy stage (F19/plan.json, F19/deliverable.json stage_heads_by_sha256)
F19_ID = dict(zip(STAGES, ["s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b", "sel"]))
TSTAR = "f19_8_p3b_payload180"
STUDENT = "f19_9_student"
RL_STAGES = STAGES[:-1]
#: the stages trained with the clock observation hidden (c6x and every stage after it, the student included)
CLOCK_HIDDEN = {"f19_5_c6x_clockfree", "f19_6_p2a_field", "f19_7_p3a_payload90", "f19_8_p3b_payload180", STUDENT}
#: the superseded seed-18 lineage, kept in the folder
S18_STAGES = ["1_fb6k", "2_c4", "3_c5", "4_c6", "5_c6x_clockfree", "6_p2a_field"]
#: the stand-network campaign (its sn_common.py is the stand-net input / D5 env rule both stand campaigns use)
SN_CAMPAIGN = PKG / "experiments" / "2026-10-11_stand_network"
SN_COMMON = SN_CAMPAIGN / "scripts" / "sn_common.py"
SN_SPEC = SN_CAMPAIGN / "scripts" / "sn_dagger_spec.py"
#: the stand-HEIGHT campaign: its sh_core.py is the ONE implementation of the height-commanded contract, its
#: head_*.json / step4b_read.json are the superseded SH_T17's gate reads
SH_CAMPAIGN = PKG / "experiments" / "2026-10-13_stand_height"
SH_CORE = SH_CAMPAIGN / "scripts" / "sh_core.py"
SH_HEADS = ("SH_T18", "SH_T16", "SH_T17")
#: the fixed paths the stand campaign tests and sh_core load (they must keep resolving to the kept files)
AMASK_KEPT = POLICY / "stand_net" / "model_4000.pt"
SHT17_KEPT = POLICY / "stand_net" / "model_5000.pt"


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


def _m() -> dict:
    return yaml.safe_load((POLICY / "manifest.yaml").read_text())


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", str(POLICY.relative_to(REPO))], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return {l for l in out.stdout.splitlines() if l}


def _git_blob(commit: str, rel: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True, check=True).stdout


def _missing_commits(*commits: str) -> list[str]:
    """The named commits this clone does not hold (a shallow clone -- CI's depth-1 checkout -- has no history)."""
    return [c for c in dict.fromkeys(commits)
            if subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=REPO, capture_output=True).returncode]


#: the checkout-independent path form of the tracked plan / README-facing records (the F19 env files' KRABBY_RSI_BANK,
#: written by the pipeline's gen; the stand block's SN_WALKER_* env): ``$REPO/<repo-relative path>`` (TP
#: tp_lib.REPO_TOKEN, bundle_policy.REPO_TOKEN), resolved against the checkout that reads it
REPO_TOKEN = "$REPO"


def _at_checkout(v):
    """A ``$REPO/<path>`` value -- recursively, in a mapping -- resolved against THIS checkout; anything else as is."""
    if isinstance(v, dict):
        return {k: _at_checkout(x) for k, x in v.items()}
    if isinstance(v, str) and v.startswith(REPO_TOKEN + "/"):
        return str(REPO / v[len(REPO_TOKEN) + 1:])
    return v


def _names_recorded(portable: str, recorded: str) -> bool:
    """A ``$REPO/<rel>`` value names the path a run RECORD states (absolute, in the checkout the run ran in: records
    such as deliverable.json are history and keep the root they were written under)."""
    return portable.startswith(REPO_TOKEN + "/") and recorded.endswith("/" + portable[len(REPO_TOKEN) + 1:])


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


def _env_file(p: Path) -> dict[str, str]:
    return dict(ln.split("=", 1) for ln in p.read_text().splitlines() if ln and not ln.startswith("#"))


def _deliverable() -> dict:
    return json.loads((F19 / "deliverable.json").read_text())


def _sup(m: dict, sid: str) -> dict:
    return next(e for e in m["superseded"] if e["id"] == sid)


def _rel_logs(p):
    return p[p.index("parkour/"):] if p else None


# ------------------------------------------------------------------------------ the lineage of record (F19)
def test_manifest_and_files_are_consistent():
    tool = _tool()
    m = tool.load_manifest()
    assert m["task"] == "crab_hex_omni_task" and m["plant"] == "A15+B" and m["seed"] == 19
    problems = tool.check(m)
    assert not problems, problems


def test_stage_order_and_chain():
    m = _m()
    assert [s["dir"] for s in m["stages"]] == STAGES
    assert [s["phase"] for s in m["stages"]] == ["1b", "1a", "1a", "1a", "1a", "2a", "2a", "2a", "2a"]
    assert [s["stage"] for s in m["stages"]] == ["s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b", "st"]
    assert [s.get("kind") for s in m["stages"]] == ["rl"] * 8 + ["distill"]
    assert m["stages"][0]["resume_from"] == "scratch"
    for prev, s in zip(m["stages"], m["stages"][1:]):
        assert prev["dir"] in s["resume_from"], f"{s['dir']} must resume from {prev['dir']}"
    # current = T* = the last RL stage; the student (distill) follows it and is not the walking head
    assert m["current"] == m["stages"][-2]["dir"] == TSTAR
    assert m["heads"] == {"phase1": "f19_5_c6x_clockfree", "phase2": "f19_6_p2a_field", "payload": TSTAR,
                          "student": STUDENT, "stand": "f19_stand_net"}
    assert m["heads"]["stand"] not in STAGES
    # the new heads take folders of their own: no F19 stage reuses a superseded (kept) folder
    assert not set(STAGES) & set(S18_STAGES) and "stand_net" != m["heads"]["stand"]


def test_stage_files_are_one_resume_chain():
    """Each RL stage file is the checkpoint the next stage resumed from, read from the F19 driver's own tracked state
    (F19/state.json, whose candidates.yaml must still be the one the driver ran), at the manifest seed, with exactly the
    sha256 the deliverable record lists; the distill stage is the student the pipeline trained from T* and selected."""
    m = _m()
    dl = _deliverable()
    heads = dl["stage_heads_by_sha256"]
    st = json.loads((F19 / "state.json").read_text())
    assert (hashlib.sha256((F19 / "candidates.yaml").read_bytes()).hexdigest()
            == st["ladder"]["candidates_sha256"])
    by_ckpt = {_rel_logs(e["ckpt"]): e for e in st["keys"][f"seed{m['seed']}"].values()
               if isinstance(e, dict) and e.get("ckpt")}
    prev = None
    n_driver = n_distill = 0
    for s in m["stages"]:
        assert s["overlay_ref"].startswith("experiments/2026-10-16_task4_pipeline/F19/candidates.yaml "), s["dir"]
        assert heads[F19_ID[s["dir"]]]["sha256"] == s["sha256"], s["dir"]
        assert _rel_logs(heads[F19_ID[s["dir"]]]["ckpt"]) == s["source"], s["dir"]
        if s["kind"] == "distill":
            # the student: distilled from the previous stage (T*), selected by the sel stage, its run dir the source's
            sel = json.loads((F19 / "stage_reads/sel.json").read_text())
            sread = json.loads((F19 / "stage_reads/st.json").read_text())
            assert (sel["sha256"], _rel_logs(sel["ckpt"]), sel["tstar"]) == (s["sha256"], s["source"], "p3b")
            assert dl["student"]["sha256"] == s["sha256"] and dl["student"]["tstar"] == "p3b"
            assert _rel_logs(sread["run_dir"]) == str(Path(s["source"]).parent) and sread["tstar"] == "p3b"
            assert prev == _rel_logs(dl["tstar"]["ckpt"]), "the student must be distilled from T* = the previous stage"
            assert [c["iter"] for c in sread["candidates"]] == s["distill"]["candidates"]
            assert s["distill"]["seed"] == m["seed"]
            n_distill += 1
        else:
            assert f"seed{m['seed']}_" in s["source"], f"{s['dir']}: {s['source']} is not a seed{m['seed']} run"
            cs = _chain_stage(s["overlay_ref"])
            assert not isinstance(cs.get("resume_from"), dict) or "checkpoint" not in cs["resume_from"], s["dir"]
            e = by_ckpt.get(s["source"])
            assert e, f"{s['dir']}: {s['source']} is not a seed{m['seed']} checkpoint of F19"
            assert _rel_logs(e.get("resume")) == prev, f"{s['dir']} resumed from {e.get('resume')}, not {prev}"
            assert e["env"].get("KRABBY_OMNI_PHASE") == s["phase"] and e["phase"] == s["stage"], s["dir"]
            assert e["task"] == s["task"] and e["status"] == "ok", s["dir"]
            n_driver += 1
        prev = s["source"]
    assert (n_driver, n_distill) == (8, 1)
    assert dl["tstar"]["stage"] == "p3b" and _rel_logs(dl["tstar"]["ckpt"]) == m["stages"][7]["source"]


def test_every_policy_file_is_tracked_and_nothing_else():
    m = _m()
    tracked = _tracked()
    rel = POLICY.relative_to(REPO)
    tool = _tool()
    expected = {f"{rel}/{f}" for f in tool.expected_files(m)}
    assert expected <= tracked, f"policy files not in git's index (git add them): {sorted(expected - tracked)}"
    # a superseded file that LEFT must leave the index too, not only the folder (--sync stages only what it writes)
    assert tracked <= expected, f"tracked files the manifest does not name (git rm them): {sorted(tracked - expected)}"
    on_disk = {str(p.relative_to(REPO)) for p in POLICY.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert on_disk == expected, sorted(on_disk ^ expected)
    # the superseded-but-kept files are expected, tracked and on disk; the student checkpoint is none of them
    kept = {f"{rel}/{f['dir']}/{f['file']}" for e in m["superseded"] for f in e.get("files") or []}
    assert kept == {f"{rel}/{d}/{s['file']}" for d, s in zip(S18_STAGES, _sup(m, "S1_s18")["stages"])} | {
        f"{rel}/stand_net/model_4000.pt", f"{rel}/stand_net/model_5000.pt"}
    assert kept <= expected & tracked & on_disk
    st = next(s for s in m["stages"] if s["dir"] == STUDENT)
    assert f"{rel}/{STUDENT}/{st['file']}" not in expected | tracked | on_disk
    assert not list((POLICY / STUDENT).glob("*.pt"))


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
    assert "KRABBY_OMNI_CLOCK_HIDE=1" in text and "KRABBY_OMNI_COMMAND_FRAME=world" in text
    # the play / eval commands must rebuild the soft-std policy, or the strict load refuses the checkpoint
    assert "KRABBY_STD_PARAM=soft" in text
    # T* ships; the README names every head and the teacher task
    assert f"policy/{TSTAR}/model_25992.pt" in text and "Isaac-Crab-Hex-Omni-Teacher-v0" in text
    for k, d in (("phase1", "f19_5_c6x_clockfree"), ("phase2", "f19_6_p2a_field"), ("payload", TSTAR),
                 ("student", STUDENT), ("stand", "f19_stand_net")):
        assert f"{k} [`{d}/`]" in text, k
    assert f"| `{TSTAR}/` | `field_hard` | 0.905 | 200 |" in text
    assert "| `f19_6_p2a_field/` | `field_hard` | 0.7725 | 400 |" in text
    assert f"| `{TSTAR}/` | `0:180` | 0 yes, 45 yes, 90 yes, 135 yes, 180 NO | 135 | 180 |" in text
    # the student section: its own task and role, the cited (not bundled) checkpoint, the bundled exports
    st = tool.stage_by_dir(m, STUDENT)
    assert "## The depth student -- [`f19_9_student/`](f19_9_student/) (kind `distill`)" in text
    assert "python scripts/rsl_rl/play.py --task Isaac-Crab-Hex-Omni-Student-Play-v0 --enable_cameras \\" in text
    assert "--task Isaac-Crab-Hex-Omni-Student-Play-v0 --policy-role student \\" in text
    assert f"--checkpoint $PWD/{st['source'].removeprefix('parkour/')}" in text
    assert "**Checkpoint NOT bundled** (`bundled: false`)" in text
    for e in st["exports"]:
        assert f"[`{STUDENT}/{e['file']}`]({STUDENT}/{e['file']})" in text and e["sha256"] in text
    # the superseded files kept in the folder are listed as such, never as the policy of record
    assert "## Superseded heads kept in this folder" in text
    for e in m["superseded"]:
        for f in e.get("files") or []:
            assert f"[`{f['dir']}/{f['file']}`]({f['dir']}/{f['file']}) -- sha256 `{f['sha256']}`" in text
    assert "**Current head:** [`f19_8_p3b_payload180/model_25992.pt`]" in text


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
    # every F19 training env names the bank and turns RSI off (RSI_FRAC 0), as the asset's used_by says
    for sid in ("s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b"):
        env = _env_file(F19 / f"envs/{sid}.train.env")
        assert env["KRABBY_RSI_BANK"].startswith(REPO_TOKEN + "/"), sid           # checkout-independent as committed
        assert _at_checkout(env["KRABBY_RSI_BANK"]) == str(preset_bank) and env["KRABBY_RSI_FRAC"] == "0", sid


def test_obs_contract_current_head_is_clock_hidden():
    """c6x and every later head (the student included) trained with KRABBY_OMNI_CLOCK_HIDE=1; the earlier stages with the
    clock visible; the manifest's carried semantics cover exactly the driver's EVAL_CARRY set minus the body-only
    envelope keys (every head here is a WORLD-frame head), and every stage's full eval env is EXACTLY what the driver's
    suite eval (run_phases.eval_morph: keep CANARY_KEYS from ph.FORMATION) builds from that stage's preset + overlay --
    and exactly the tracked F19 env file the pipeline evaluated it with (the student: T*'s, its selection rows')."""
    tool = _tool()
    m = tool.load_manifest()
    cur = tool.stage_by_dir(m, m["current"])
    assert str(cur["obs_contract"][HIDE]) == "1"
    assert tool.eval_env(m, cur)[HIDE] == "1"
    assert tool.play_env(m, cur)[HIDE] == "1"
    for s in m["stages"]:
        assert str(s["obs_contract"][HIDE]) == ("1" if s["dir"] in CLOCK_HIDDEN else "0"), s["dir"]
    # phase 2, the payload stages and the student did not change the policy's inputs: their eval env is the c6x head's
    c6x = tool.eval_env(m, tool.stage_by_dir(m, "f19_5_c6x_clockfree"))
    for d in ("f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR, STUDENT):
        assert tool.eval_env(m, tool.stage_by_dir(m, d)) == c6x, d
    assert m["obs_contract"]["width"] == 1149
    rp = _load(RUN_PHASES, "omni_run_phases_policy_test")
    carry = m["obs_contract"]["eval_env"]["carry"]
    assert set(carry) | {HIDE} == set(rp.EVAL_CARRY) - set(rp.EVAL_CARRY_BODY_ONLY)
    assert carry[rp.COMMAND_FRAME_VAR] == "world"
    plan = json.loads((F19 / "plan.json").read_text())
    sel_rows = json.loads((F19 / "stage_reads/sel.rows.json").read_text())
    for s in m["stages"]:
        base, ov = _overlay(s["overlay_ref"])
        assert base == s["phase"], f"{s['dir']}: chain base {base} != manifest phase {s['phase']}"
        want = rp.eval_env_for(rp.ph.PHASES[s["phase"]], m["plant"], keep=rp.CANARY_KEYS,
                               keep_source=rp.ph.FORMATION, carry_overlay=ov)
        assert tool.eval_env(m, s) == want, s["dir"]
        # the tracked env file the pipeline evaluated the stage with (the student: the sel rows' env file)
        if s["kind"] == "distill":
            envf = {Path(v["job"]["env_file"]).name for k, v in sel_rows.items()
                    if k.startswith("c28000:suite@")}
            assert envf == {"p3b.env"}
            assert tool.eval_env(m, s) == _env_file(F19 / "envs/p3b.env")
        else:
            assert tool.eval_env(m, s) == _env_file(F19 / plan["stages"][s["stage"]]["env"]), s["dir"]


def test_eval_stamps_are_the_stage_files_under_the_manifest_env():
    """Every stage's eval stamp (untracked; skipped when not on this machine) scored exactly the stage's file, under
    exactly the manifest's eval env, with the policy width 1149; its per-category suite numbers are the manifest's."""
    tool = _tool()
    m = tool.load_manifest()
    rp = _load(RUN_PHASES, "omni_run_phases_policy_stamp_test")
    seen = 0
    for s in m["stages"]:
        st = REPO / s["evals"]["stamp"]
        if not (st / "run_meta.json").is_file():
            continue
        meta = json.loads((st / "run_meta.json").read_text())
        assert meta["checkpoint_sha256"] == s["sha256"], s["dir"]
        assert meta["env_vars"] == tool.eval_env(m, s), s["dir"]
        assert meta["obs_dim_actual"] == 1149, s["dir"]
        cat = rp.summarize(json.loads((st / "scenario_metrics.json").read_text())["aggregate"])["cat"]
        for c, e in s["evals"]["categories"].items():
            for k, v in e.items():
                got = cat[c][k]
                assert (got == v) if k == "n" else abs(round(got, 3) - v) < 1e-9, (s["dir"], c, k, got, v)
        seen += 1
    if not seen:
        pytest.skip("no F19 eval stamp on this machine (raw stamps are untracked)")


def test_recipe_construction_is_what_every_stage_trained():
    """The manifest's runner-side construction block (and each stage's removal set) is in every RL stage's overlay,
    except the keys a stage declares in its own ``construction_delta`` -- which must be what that overlay trained and
    never a POLICY-structure knob; the F19 s1 .. p2a overlays are byte-identical to the superseded seed-18 lineage's
    (same recipe, new seed); the payload knobs appear on the payload stages only; the distill stage trained under T*'s
    composed env minus exactly the student-runner-refused knobs (STD_PARAM / INIT_STD kept)."""
    m = _m()
    con = m["construction"]
    assert con["ppo"]["KRABBY_STD_PARAM"] == m["obs_contract"]["eval_env"]["carry"]["KRABBY_STD_PARAM"]
    assert str(con["ppo"]["KRABBY_INIT_STD"]) == str(m["obs_contract"]["eval_env"]["carry"]["KRABBY_INIT_STD"])
    deltas = {s["dir"]: s.get("construction_delta") or {} for s in m["stages"]}
    assert {d: v for d, v in deltas.items() if v} == {d: {"KRABBY_ACTION_RATE_W": "-0.05"} for d in (
        "f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR)}
    old = {s["stage"]: s for s in _sup(m, "S1_s18")["stages"]}
    for s in m["stages"]:
        _, ov = _overlay(s["overlay_ref"])
        if s["kind"] == "distill":
            continue
        delta = {k: str(v) for k, v in deltas[s["dir"]].items()}
        assert not set(delta) & {"KRABBY_STD_PARAM", "KRABBY_INIT_STD"}, s["dir"]
        for blk in ("ppo", "every_stage"):
            for k, v in con[blk].items():
                assert ov.get(k) == delta.get(k, str(v)), (s["dir"], k, ov.get(k), v)
        for k, v in (s.get("removal_set") or {}).items():
            assert ov.get(k) == str(v), (s["dir"], k)
        assert ov.get(HIDE, "0") == str(s["obs_contract"][HIDE]), s["dir"]
        pay = {k: v for k, v in ov.items() if k.startswith("KRABBY_PAYLOAD_")}
        assert pay == {k: str(v) for k, v in ((s.get("payload") or {}).get("knobs") or {}).items()}, s["dir"]
        if s["stage"] in old:
            _, ov_old = _overlay(old[s["stage"]]["overlay_ref"])
            assert ov == ov_old, f"{s['dir']}: the F19 overlay differs from the seed-18 lineage's"
            assert _chain_stage(s["overlay_ref"])["iters"] == _chain_stage(old[s["stage"]]["overlay_ref"])["iters"]
    # the distill stage: T*'s composed env (preset + plant + overlay) minus the student-runner-refused knobs
    st = next(s for s in m["stages"] if s["kind"] == "distill")
    tstar = next(s for s in m["stages"] if s["dir"] == st["distill"]["teacher"])
    assert st["distill"]["teacher"] == TSTAR and st["overlay_ref"] == tstar["overlay_ref"]
    _, ov = _overlay(tstar["overlay_ref"])
    denv = _env_file(PKG / st["distill"]["env_file"])
    assert denv == {"KRABBY_OMNI_PHASE": st["phase"], "KRABBY_PLANT": m["plant"], **ov}
    # the refused list is read from the student spec's source (as distill.sh does), not by importing the config module
    import ast
    src = (PKG / "config" / "crab_hex" / "omni_student_spec.py").read_text()
    refused = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED" for t in n.targets))
    assert st["distill"]["stripped"] == [k for k in refused if k in denv]
    assert {"KRABBY_STD_PARAM", "KRABBY_INIT_STD"} <= set(denv) - set(st["distill"]["stripped"])
    for k, v in st["removal_set"].items():
        assert denv.get(k) == str(v), k


def test_payload_and_field_stages_with_their_evals_of_record():
    """p2a: preset 2a on the teacher task, the wall field, the clock still removed; p3a / p3b: the same + their payload
    knobs, curves and max_stable exactly as the tracked gate records say; every obstacle row pooled consistently and
    equal to the gate's own row reads; T* = p3b by the recorded rule; every stage FAST / FAST and its gate record's
    verdict and head sha as the manifest says."""
    m = _m()
    st = {s["dir"]: s for s in m["stages"]}
    p2a = st["f19_6_p2a_field"]
    assert (p2a["task"], p2a["play_task"]) == ("Isaac-Crab-Hex-Omni-Teacher-v0", "Isaac-Crab-Hex-Omni-Walk-Play-v0")
    assert p2a["overlay_ref"] == "experiments/2026-10-16_task4_pipeline/F19/candidates.yaml chains.F19_p2a stage p2a"
    for d in ("f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR):
        _, ov = _overlay(st[d]["overlay_ref"])
        for k, v in {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_FIELD_EDGES": "wall", "KRABBY_PHASEOUT": "",
                     "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0", "KRABBY_TERRAIN_PROMOTE": "0.7:0.4"}.items():
            assert ov.get(k) == v, (d, k, ov.get(k))
        for k, v in st[d]["removal_set"].items():
            assert ov.get(k) == str(v), (d, k)
    for d in STAGES[:6]:
        assert not st[d].get("payload"), d
    assert st["f19_7_p3a_payload90"]["payload"]["knobs"]["KRABBY_PAYLOAD_KG"] == "0:90"
    assert st[TSTAR]["payload"]["knobs"]["KRABBY_PAYLOAD_KG"] == "0:180"
    dl = _deliverable()
    for s in m["stages"]:
        g = s.get("gate")
        if s["kind"] == "distill":
            sel = json.loads((F19 / "gates/sel.json").read_text())
            assert sel["verdict"] == "PASS" and sel["blocks"]["c28000"]["pass"] is True
            continue
        rec = json.loads((PKG / g["record"]).read_text())
        assert (rec["verdict"], rec["thresholds"]["version"]) == (g["verdict"], g["thresholds"]), s["dir"]
        assert rec["head"]["sha256"] == s["sha256"] and rec["stage"] == s["stage"], s["dir"]
        assert dl["status"][s["stage"]] in ("pass", "accepted", "ceiling"), s["dir"]
        side = [x for x in rec["decisive"] if "side22" in x["id"]]
        assert len(side) == 1 and side[0]["value"] == ["FAST", "FAST"] and side[0]["pass"], s["dir"]
        assert s["evals"]["lr"]["hands"] == "FAST/FAST", s["dir"]
        if s.get("payload"):
            p = s["payload"]
            assert {int(k): v for k, v in rec["curve"].items()} == {int(k): v for k, v in p["curve"].items()}, s["dir"]
            assert rec["max_stable"] == p["max_stable_kg"], s["dir"]
            assert min(int(k) for k, v in rec["curve"].items() if not v) == p["first_unstable_kg"], s["dir"]
            if rec.get("limiting_factor"):
                assert rec["limiting_factor"]["first_unstable_level"] == p["first_unstable_kg"], s["dir"]
                assert rec["limiting_factor"]["name"] in p["limiting_factor"], s["dir"]
            assert rec["zero_kg"]["verdict"] in p["zero_kg_check"] and rec["label"] == p["label"], s["dir"]
        # every obstacle row: n = 100 per env seed, the pooled completion is the mean, the per-seed values are the gate's
        for name, r in ((s.get("evals") or {}).get("rows") or {}).items():
            per = r.get("per_env_seed")
            if not per:
                continue
            assert r["n"] == 100 * len(per) == 100 * len(r["stamps"]), (s["dir"], name)
            assert abs(sum(per) / len(per) - r["completion"]) < 1e-9, (s["dir"], name)
            assert per == [rec["watch"][f"{name}@{k}"]["completion"] for k in range(1, len(per) + 1)], (s["dir"], name)
    assert st[TSTAR]["evals"]["rows"]["field_hard"]["completion"] == 0.905
    assert p2a["evals"]["rows"]["field_hard"]["completion"] == 0.7725 and p2a["evals"]["rows"]["field_hard"]["n"] == 400
    # T*: the PLAN rule's choice, the highest max_stable whose paired 0-kg check passes; p3a accepted, not chosen
    ts = dl["tstar"]
    assert ts["stage"] == "p3b" and ts["sha256"] == st[TSTAR]["sha256"] and ts["max_stable"] == 135
    by = {c["stage"]: c for c in ts["considered"]}
    assert by["p3b"]["chosen"] and by["p3b"]["zero_kg"] == "PASS" and not by["p3a"]["chosen"]
    assert by["p3a"]["zero_kg"] == "FAIL" and by["p3a"]["accepted"]
    assert "HALT" == st["f19_7_p3a_payload90"]["gate"]["verdict"] and "accepted" in st["f19_7_p3a_payload90"]["gate"]["halt"].lower()


def test_distill_stage_is_the_student_with_its_exports():
    """f19_9_student: kind distill on the Student task, NOT bundled (its .pt is over GitHub's 100 MiB limit), cited by the
    deliverable's sha256; its folder holds exactly the ONNX deploy export + io json, byte-identical to the campaign's
    tracked keep-set copies (bundle.yaml extra heads); the io json is the 1149-wide, 4-camera, clip-1.0 contract; the
    selection record is the gate's; the run file (when on disk) is that sha, over the limit, and carries T*'s weights."""
    tool = _tool()
    m = tool.load_manifest()
    s = tool.stage_by_dir(m, STUDENT)
    assert (s["kind"], s["bundled"], s["task"], s["play_task"]) == (
        "distill", False, "Isaac-Crab-Hex-Omni-Student-v0", "Isaac-Crab-Hex-Omni-Student-Play-v0")
    assert not tool.is_bundled(s) and s["file"] == "model_28000.pt"
    assert s["sha256"] == _deliverable()["student"]["sha256"] == "4fa81f0f308a4a7ac73ba8db106ab9a9c286586d05ec7a18908d776b33db9059"
    assert s["eval_args"] == ["--task", "Isaac-Crab-Hex-Omni-Student-Play-v0", "--policy-role", "student"]
    assert s["play_args"] == ["--enable_cameras"]
    files = tool.stage_files(s)
    assert [f["file"] for f in files] == ["policy.onnx", "depth_latest.onnx", "omni_student_io.json"]
    assert sorted(p.name for p in (POLICY / STUDENT).iterdir()) == sorted(f["file"] for f in files)
    bundle = yaml.safe_load((T4 / "bundle.yaml").read_text())
    extra = {Path(x["source"]).name: x for x in bundle["head"]["extra_heads"] if x["source"].endswith(
        tuple(f["file"] for f in files))}
    for f in files:
        assert tool.sha256_of(f["path"]) == f["sha256"] == extra[f["file"]]["sha256"], f["file"]
        assert f["source"] == extra[f["file"]]["source"], f["file"]
        keep = T4 / extra[f["file"]]["dest"]
        assert keep.read_bytes() == f["path"].read_bytes(), f"{f['file']} != the campaign keep-set copy {keep}"
        assert f["path"].stat().st_size <= tool.MAX_FILE_BYTES
    io = json.loads((POLICY / STUDENT / "omni_student_io.json").read_text())
    assert io["kind"] == "omni_depth_student" and io["obs"] == [1, m["obs_contract"]["width"]]
    assert len(io["camera_groups"]) == 4 and io["actions_clip"] == 1.0 and io["trained_on_zero_depth"] is False
    # the selection: the gate's paired completions and rank, the candidate the deliverable names
    sel = json.loads((F19 / "gates/sel.json").read_text())["blocks"]["c28000"]
    got = {x["id"].rsplit(".", 1)[1]: x for x in sel["decisive"] if x["id"].startswith("student.capability.completion.")}
    for row, e in s["evals"]["paired_completion"].items():
        assert abs(got[row]["value"] - e["student"]) < 1e-9 and abs(got[row]["line"] - e["line"]) < 5e-4, row
        assert got[row]["pass"] is e["pass"] is True, row
        assert abs(s["evals"]["rows"][row]["completion"] - e["student"]) < 1e-9, row
    assert s["distill"]["selected"] == "c28000" and s["distill"]["prescreen_rank"][0] == 28000
    assert "UNVERIFIED" in s["distill"]["camera_use"]
    src = REPO / s["source"]
    if not src.exists():
        pytest.skip(f"student run file not on this machine (raw run dirs are untracked): {src}")
    assert src.stat().st_size > tool.MAX_FILE_BYTES and tool.sha256_of(src) == s["sha256"]
    torch = pytest.importorskip("torch")
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    tstar = torch.load(str(tool.stage_path(tool.stage_by_dir(m, TSTAR))), map_location="cpu", weights_only=False)
    a, b = ck["model_state_dict"], tstar["model_state_dict"]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), "the student's teacher copy is T*'s"
    assert ck["iter"] == 28000 and {"depth_actor_state_dict", "depth_encoder_state_dict"} <= set(ck)


def test_bundle_policy_check_is_clean():
    out = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 cited, not bundled: f19_9_student" in out.stdout and "+ 8 kept superseded files" in out.stdout


# ------------------------------------------------------------------------------ superseded heads (kept / left)
def test_superseded_heads_are_recorded_and_recoverable():
    """W left policy/ at the 2026-10-06 bake and is recoverable, byte-exact, from the commit the manifest names. Nothing
    left at the 2026-10-24 bake: every kept file (the seed-18 lineage, SH_T17, A-mask s17) is present with its recorded
    sha, byte-identical to the file it was baked / adopted with, and its record is the manifest at record_commit
    verbatim."""
    m = _m()
    tool = _tool()
    sups = m.get("superseded") or []
    assert [s["id"] for s in sups] == ["W", "SN_Amask_s17", "S1_s18", "SH_T17"]
    w = sups[0]
    commit = w["last_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", str(commit)), f"superseded W: last_commit {commit!r}"
    assert "files" not in w
    s18 = _sup(m, "S1_s18")
    baked_at = {d: ("80e839e3a987a912222fa216a31cd192a43cf014" if d != "6_p2a_field"
                    else "dd972b6a2fc919f1ab7426fcea41a6e962577761") for d in S18_STAGES}
    # the byte-exact history reads need the commits; a shallow clone runs every other check, then skips naming them
    no_hist = _missing_commits(commit, *baked_at.values(), s18["record_commit"],
                               *(_sup(m, sid)["adopted_commit"] for sid in ("SN_Amask_s17", "SH_T17")))
    hist = not no_hist
    for st in w["stages"]:
        rel = f"{w['path_in_history']}/{st['dir']}/{st['file']}"
        assert not (REPO / rel).exists(), f"{rel}: a superseded W file is back in the folder"
        if hist:
            assert hashlib.sha256(_git_blob(commit, rel)).hexdigest() == st["sha256"], rel
    rel_pol = str(POLICY.relative_to(REPO))
    # every kept file: present, the recorded sha, not a live file, and byte-identical to its baked / adopted blob
    assert [f["dir"] for f in s18["files"]] == S18_STAGES
    assert [(f["dir"], f["file"], f["sha256"], f["source"]) for f in s18["files"]] == [
        (s["dir"], s["file"], s["sha256"], s["source"]) for s in s18["stages"]]
    for f in s18["files"]:
        p = POLICY / f["dir"] / f["file"]
        assert p.is_file() and tool.sha256_of(p) == f["sha256"], p
        if hist:
            assert hashlib.sha256(_git_blob(baked_at[f["dir"]], f"{rel_pol}/{f['dir']}/{f['file']}")).hexdigest() \
                == f["sha256"]
    for sid, path, commit_key in (("SN_Amask_s17", AMASK_KEPT, "adopted_commit"), ("SH_T17", SHT17_KEPT, "adopted_commit")):
        e = _sup(m, sid)
        assert e["kind"] == "stand_network" and "2026-10-24" in e["superseded"], sid
        assert [(f["dir"], f["file"]) for f in e["files"]] == [("stand_net", path.name)], sid
        assert path.is_file() and tool.sha256_of(path) == e["files"][0]["sha256"], sid
        c = e[commit_key]
        assert re.fullmatch(r"[0-9a-f]{40}", c) and c[:8] in e["adopted"], sid
        if hist:
            assert hashlib.sha256(_git_blob(c, str(path.relative_to(REPO)))).hexdigest() == e["files"][0]["sha256"], sid
        assert not {"last_commit", "path_in_history"} & set(e), sid
    assert "2026-10-24" in _sup(m, "SN_Amask_s17")["superseded"] and "not removed" in _sup(m, "SN_Amask_s17")["superseded"]
    sh = _sup(m, "SH_T17")
    assert sh["adopted_commit"] == s18["record_commit"]
    assert sh["block"]["sha256"] == sh["files"][0]["sha256"] and sh["block"]["walker"]["stage"] == "6_p2a_field"
    assert sh["block"]["warm_start"]["sha256"] == _sup(m, "SN_Amask_s17")["files"][0]["sha256"]
    if not hist:
        pytest.skip(f"git history not in this clone (shallow checkout): commits {no_hist} -- the byte-exact blob and "
                    "record_commit manifest checks need them")
    # the kept records are the manifest as it stood at record_commit, verbatim
    old = yaml.safe_load(_git_blob(s18["record_commit"], f"{rel_pol}/manifest.yaml"))
    assert old["current"] == s18["current_was"] == "6_p2a_field" and old["heads"] == s18["heads_were"]
    assert s18["stages"] == old["stages"] and [s["dir"] for s in old["stages"]] == S18_STAGES
    assert (s18["baked"], s18["recipe"], s18["construction"]) == (old["baked"], old["recipe"], old["construction"])
    assert sh["block"] == old["stand_network"], "the kept SH_T17 block must be the record_commit block verbatim"


def test_kept_superseded_paths_stay_valid_for_their_users():
    """The fixed paths other code names keep resolving to the same bytes: sh_core's warm-start pin (A-mask s17 at
    stand_net/model_4000.pt), the Task 4 pipeline's bundled_s1 (6_p2a_field) and its lock, p2_read's lookup of the
    seed-18 c6x categories by sha (now under superseded S1_s18), and the dose smokes' superseded-W resolution."""
    tool = _tool()
    m = tool.load_manifest()
    core = _load_nobc(SH_CORE, "sh_core_policy_kept_test")
    assert Path(core.STAND_NET_CKPT).resolve() == AMASK_KEPT.resolve()
    assert core.STAND_NET_SHA256 == tool.sha256_of(AMASK_KEPT) == _sup(m, "SN_Amask_s17")["files"][0]["sha256"]
    spec = yaml.safe_load((T4 / "pipeline.yaml").read_text())
    s1 = spec["bundled_s1"]
    assert tool.sha256_of(REPO / s1["ckpt"]) == s1["sha256"] == _sup(m, "S1_s18")["files"][-1]["sha256"]
    lock = json.loads((T4 / "pipeline.lock.json").read_text())
    locked = json.dumps(lock)
    for rel in ("6_p2a_field/model_19994.pt", "stand_net/model_4000.pt", "stand_net/model_5000.pt"):
        assert f"crab_hex_omni_task/policy/{rel}" in locked and (POLICY / rel).is_file(), rel
    p2r = _load_nobc(PKG / "experiments" / "2026-10-07_omni_phase2" / "scripts" / "p2_read.py", "p2_read_policy_kept")
    c6x = next(s for s in _sup(m, "S1_s18")["stages"] if s["dir"] == "5_c6x_clockfree")
    assert p2r._manifest_s18() == c6x["evals"]["categories"]
    w = _sup(m, "W")
    assert [st["dir"] for st in w["stages"]][-1] == "5_c6x_clockfree" and w["stages"][-1]["file"] == "model_11995.pt"


def test_superseded_sh_t17_block_still_matches_its_campaign_reads():
    """The kept SH_T17 record (superseded[SH_T17].block): its K1-K6 table is still the stand-height campaign's reads
    (head_SH_T{18,16,17}.json, step4b_read.json), its checkpoint (the kept stand_net/model_5000.pt) is the height net
    trained beside the kept S1 s18 (6_p2a_field) and warm-started from the kept A-mask s17."""
    m = _m()
    blk = _sup(m, "SH_T17")["block"]
    gt = blk["evals"]["gate_table"]
    step4b = json.loads((SH_CAMPAIGN / "step4b_read.json").read_text())
    heads = {h: json.loads((SH_CAMPAIGN / f"head_{h}.json").read_text()) for h in SH_HEADS}
    assert step4b["mismatches"] == [] and step4b["pending"] == []
    controls = ("R_h", "R_old", "Amask_old", "S1", "no_restore", "posture")
    assert [c["id"] for c in gt["columns"]] == [*controls, *SH_HEADS]

    def read(col, K):
        return heads[col]["gates"][K][col] if col in heads else step4b["gates"][K].get(col)

    assert [r["key"] for r in gt["rows"]] == list(_K_ROWS)
    for r in gt["rows"]:
        gate, kind, pat, scale, tol = _K_ROWS[r["key"]]
        for col in gt["columns"]:
            g = read(col["id"], gate)
            if g is None:
                assert r[col["id"]] is None, (r["key"], col["id"])
                continue
            _k_close(r[col["id"]], _k_cell(g, kind, pat, scale), tol, (r["key"], col["id"]))
    for col in gt["columns"]:
        cid = col["id"]
        failing = [K for K in _GATES if read(cid, K) is not None and read(cid, K)["verdict"] != "PASS"]
        assert gt["verdicts"][cid].startswith(("FAIL " + ", ".join(failing)) if failing else "PASS"), cid
    for h, d in heads.items():
        assert d["heads"][h]["stand_sha256"] == [next(x for x in blk["evals"]["heads"] if x["id"] == h)["sha256"]], h
        assert d["k5"]["refs_s1_sha256"] == blk["walker"]["sha256"], h
    torch = pytest.importorskip("torch")
    ck = torch.load(str(SHT17_KEPT), map_location="cpu", weights_only=False)
    sh = ck["infos"]["sh_dagger"]
    tool = _tool()
    assert sh["s1_sha256"] == tool.sha256_of(POLICY / "6_p2a_field" / "model_19994.pt") == blk["walker"]["sha256"]
    assert sh["warm_start"]["sha256"] == tool.sha256_of(AMASK_KEPT)
    assert (sh["seed"], sh["residual_deadband"], sh["trim_branch"]) == (17, 0.25, True)


# ------------------------------------------------------------------------------ the stand network of record
def _stand(m: dict) -> dict:
    assert m.get("stand_network"), "policy manifest declares no stand_network block"
    return m["stand_network"]


def test_stand_network_file_is_bundled_with_the_manifest_sha():
    """The adopted SH_T4F19 file sits in its own folder (not a stage / asset / kept dir) with the manifest sha, beside the
    walking head of record T* it was trained and gated with; the folder holds exactly it and its warm-start ancestor."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    assert (sn["dir"], sn["file"]) == ("f19_stand_net", "model_5000.pt")
    assert (sn["id"], sn["arm"], sn["stand_mask"], sn["training_seed"], sn["iterations"]) == (
        "SH_T4F19_sh_s19", "A-mask", 1, 19, 5000)
    kept_dirs = {f["dir"] for e in m["superseded"] for f in e.get("files") or []}
    assert sn["dir"] not in {s["dir"] for s in m["stages"]} | {a["dir"] for a in m["assets"]} | kept_dirs
    p = tool.stand_path(sn)
    assert p == POLICY / "f19_stand_net" / "model_5000.pt" and p.is_file(), p
    assert tool.sha256_of(p) == sn["sha256"] == _deliverable()["stand"]["sh"]["sha256"]
    assert sorted(q.name for q in p.parent.iterdir()) == ["model_4000.pt", "model_5000.pt"]
    assert [f["path"] for f in tool.stand_files(sn)] == [p, tool.warm_start_path(sn)] == [p, p.parent / "model_4000.pt"]
    assert m["heads"]["stand"] == sn["dir"]
    # the walking head of record, baked at the same time, and the block pins exactly that walker
    assert m["current"] == sn["walker"]["stage"] == TSTAR
    assert sn["walker"]["sha256"] == tool.stage_by_dir(m, TSTAR)["sha256"] == _deliverable()["tstar"]["sha256"]
    assert '"SH_T4F19 as stand net (SN as its warm-start ancestor)"' in sn["adopted"] and "2026-10-24" in sn["adopted"]
    assert sn["source"] == "parkour/outputs/omni_plan/stand_height/runs/SH_T4F19_sh_s19/model_5000.pt"
    # the campaign keep-set copy (tracked) is the same bytes
    assert (T4 / "head" / "model_5000.pt").read_bytes() == p.read_bytes()


def test_expected_files_include_the_stand_network():
    tool = _tool()
    m = tool.load_manifest()
    files = tool.expected_files(m)
    # the stand network of record AND its bundled warm-start ancestor; the superseded stand_net/ files stay expected
    new = {"f19_stand_net/model_5000.pt", "f19_stand_net/model_4000.pt"}
    kept = {"stand_net/model_5000.pt", "stand_net/model_4000.pt"}
    assert new | kept <= files
    no_ws = copy.deepcopy(m)
    del no_ws["stand_network"]["warm_start"]
    assert tool.expected_files(no_ws) == files - {"f19_stand_net/model_4000.pt"}
    bare = copy.deepcopy(m)
    del bare["stand_network"]
    assert tool.expected_files(bare) == files - new
    no_sup = copy.deepcopy(m)
    no_sup["superseded"] = [e for e in no_sup["superseded"] if e["id"] == "W"]
    assert tool.expected_files(no_sup) == files - kept - {f"{d}/{s['file']}" for d, s in zip(
        S18_STAGES, _sup(m, "S1_s18")["stages"])}


def test_stand_network_source_is_the_bundled_file():
    """The source run files (untracked) are byte-identical to the bundled copies."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    for f in tool.stand_files(sn):
        src = REPO / f["source"]
        if not src.exists():
            pytest.skip(f"stand-network source run file not on this machine (raw run dirs are untracked): {src}")
        assert tool.sha256_of(src) == f["sha256"] == tool.sha256_of(f["path"])


def test_stand_warm_start_is_the_ancestor_sh_t4f19_records():
    """The bundled warm start (``stand_network.warm_start``, same folder) is the file SH_T4F19 was warm-started from:
    its sha256 equals SH_T4F19's checkpoint ``infos.sh_dagger.warm_start`` / ``warm_start_cli`` (which name its source
    run), the deliverable's SN head and the campaign keep-set copy; it is itself the stand net trained beside T* (SN gate
    PASS) whose estimator is T*'s; sh_core's own pinned default warm start stays the kept A-mask s17 file."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    ws = sn["warm_start"]
    assert (ws["id"], ws["file"]) == ("SN_T4F19_sn_s19", "model_4000.pt")
    assert ws["sha256"] == "96832099e58b75d1cc86f6f9d98327c56b827c2fac1b6c61c96b07f0a8e97b73" == _deliverable()["stand"]["sn"]["sha256"]
    assert ws["source"] == "parkour/outputs/omni_plan/stand_network/runs/SN_T4F19_sn_s19/model_4000.pt"
    assert "NOT the stand network of record" in ws["role"] and "W[:, 15] zeroed" in ws["role"]
    p = tool.warm_start_path(sn)
    assert p == POLICY / sn["dir"] / ws["file"] and p.is_file(), p
    assert tool.sha256_of(p) == ws["sha256"] and (T4 / "head" / "model_4000.pt").read_bytes() == p.read_bytes()
    assert ws["sha256"] in sn["training"]["warm_start"]
    # the SN stage's gate: G1-G5 PASS with the line counts the block states
    g = json.loads((F19 / "gates/sn.json").read_text())
    assert g["verdict"] == "PASS" and g["head"]["sha256"] == ws["sha256"]
    counts = {}
    for x in g["decisive"]:
        fam = x["id"].split(".")[2]
        counts.setdefault(fam, [0, 0])
        counts[fam][0] += bool(x["pass"])
        counts[fam][1] += 1
    assert ", ".join(f"{a} / {b}" for a, b in (counts[k] for k in ("G1", "G2", "G3", "G4", "G5"))) in ws["used_by"]
    # sh_core's pinned default is the kept A-mask s17 (the trainer warm-started SH_T4F19 from SN_T4F19 by --warm-start)
    core = _load_nobc(SH_CORE, "sh_core_policy_warm_test")
    assert Path(core.STAND_NET_CKPT).resolve() == AMASK_KEPT.resolve() and core.STAND_NET_SHA256 != ws["sha256"]
    torch = pytest.importorskip("torch")
    ck = torch.load(str(tool.stand_path(sn)), map_location="cpu", weights_only=False)
    for key in ("warm_start", "warm_start_cli"):
        rec = ck["infos"]["sh_dagger"][key]
        assert rec["sha256"] == ws["sha256"], key
        assert str(rec.get("source", rec.get("path"))).endswith(ws["source"]), (key, rec)
    assert ck["infos"]["sh_dagger"]["warm_start"]["iter"] == 4000
    wck = torch.load(str(p), map_location="cpu", weights_only=False)
    snd = wck["infos"]["sn_dagger"]
    assert (snd["s1_sha256"], snd["seed"], snd["iterations"], snd["stand_mask"], snd["arm"]) == (
        sn["walker"]["sha256"], 19, 4000, 1, "mask")
    t = torch.load(str(tool.stage_path(tool.stage_by_dir(m, TSTAR))), map_location="cpu", weights_only=False)
    assert all(torch.equal(wck["estimator_state_dict"][k], t["estimator_state_dict"][k]) for k in t["estimator_state_dict"])


def test_stand_network_checkpoint_is_the_height_net_beside_the_walker():
    """infos.sh_dagger (and the predecessor's infos.sn_dagger block): mask 1 as the manifest says, trained beside the
    walking stage the block names (s1_sha256 == T*'s sha), with R's validated constants, the trainer flags of record
    (deadband 0.25 in the loop, the trim branch at the saturated label), warm-started from the bundled SN_T4F19, under
    the height contract and clip the manifest's deployment.constants state; only the actor backbone differs from T*'s
    checkpoint and the estimator is T*'s, bitwise."""
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
    assert (sh["mode"], sh["trainable"], sh["task"]) == ("dagger", "actor.actor_backbone", "Isaac-Crab-Hex-Omni-Teacher-v0")
    assert sh["residual_deadband"] == tr["residual_deadband"] == c["residual_deadband"] == 0.25
    assert (sh["trim_branch"], sh["trim_min"]) == (tr["trim_branch"], tr["trim_min"]) == (True, 0.0999)
    assert tr["flags"].split() == ["--mode", sh["mode"], "--arm", sh["arm"], "--seed", str(sh["seed"]), "--iterations",
                                   str(sh["iterations"]), "--deadband", "0.25", "--trim-branch", "--trim-min", "0.0999"]
    assert tr["flags"] in " ".join(sn["trained_in"].split())
    sc = _load_nobc(SN_COMMON, "sn_common_policy_test")
    assert sh["r_knob"] == snd["r_knob"] == sc.RESTANCE_D3 and sh["r_knob"] in " ".join(sn["trained_in"].split())
    assert sh["warm_start"]["sha256"] == sn["warm_start"]["sha256"]
    assert sh["warm_start"]["zeroed"] == "actor.actor_backbone.0.weight[:, 15]" and sh["warm_start"]["zeroed"] in tr[
        "warm_start"]
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
    the launcher, the table test and the recorder import) and identical to the superseded SH_T17 contract's (the same
    recipe on a new walker), the posture endpoints the contract states are sh_core's q map, and the action path behaves
    as stated: hard 0.25 deadband on the clamped hip / knee residual only, before the posture; column 15 = 0 at 1.064."""
    torch = pytest.importorskip("torch")
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    dep = sn["deployment"]
    c = dep["constants"]
    old = _sup(m, "SH_T17")["block"]["deployment"]
    assert c == old["constants"]
    changed = {k for k in set(dep) | set(old) if dep.get(k) != old.get(k)}
    assert changed == {"zero_height", "scope", "caveat_switch_keying"}, changed
    assert "SH_T4F19" in dep["scope"] and "SH_T17 STOOD" not in dep["scope"]
    assert "SH_T4F19_sh_s19z" in dep["zero_height"] and "43 K5 lines" in dep["zero_height"]
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
    cols = re.fullmatch(r"(\d+)-(\d+)", str(c["deadband_cols"]))
    assert cols, c["deadband_cols"]
    assert (int(cols.group(1)), int(cols.group(2)) + 1) == (core.DEADBAND_ACT.start, core.DEADBAND_ACT.stop)
    hs = torch.tensor([c["h_min_m"], c["h_settled_m"], c["h_max_m"]], dtype=torch.float64)
    qh, qk = core.q_map(hs, dtype=torch.float64)
    assert qh[1].item() == 0.0 and qk[1].item() == 0.0
    for i in (0, 2):
        assert f"q({hs[i].item():g}) = ({qh[i].item():+.3f}, {qk[i].item():+.3f})" in dep["posture_map"], i
    assert core.encode_height(torch.tensor([c["h_settled_m"]], dtype=torch.float64)).item() == 0.0
    a = torch.zeros(1, 18)
    a[0, 0], a[0, 6], a[0, 7], a[0, 12], a[0, 13] = 0.1, 0.2, 1.7, -0.3, -0.249
    s, q = core.stand_action(a, c["h_settled_m"], deadband=c["residual_deadband"])
    want = torch.zeros(1, 18)
    want[0, 0], want[0, 7], want[0, 12] = 0.1, 1.0, -0.3
    assert torch.equal(q, torch.zeros(1, 18)) and torch.equal(s, want), s
    s2, q2 = core.stand_action(torch.zeros(1, 18), c["h_min_m"], deadband=c["residual_deadband"])
    assert torch.equal(s2, q2) and s2[0, 6:].abs().max().item() <= c["stand_rod_clamp"]
    for k in ("switch", "entry_exit", "input", "forward", "action", "posture_map", "height_command", "restore",
              "zero_height", "scope", "reference_implementation", "caveat_switch_keying", "caveat_height_source"):
        assert dep.get(k), k


def test_stand_network_routing_env_is_the_walkers():
    """Both networks run in one process under T*'s carry env: the block's env_required is exactly the walker's carry
    values, its env_off knobs are off there, the launcher's D5 rule (sn_common.env_reasons) accepts that env, which is
    T*'s tracked F19 eval env (and, when on disk, the env of every SH_T4F19 height stamp and of the F19 video); the
    walker override the README exports is the pipeline's (pipeline.walker_spec_env) and parses under sn_dagger_spec;
    every script the commands name exists; a re-run uses fresh names, the trained deadband, mask 1 and the bundled file."""
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
    assert tool.eval_env(m, w) == _env_file(F19 / "envs/p3b.env")
    # the walker override: exactly what the pipeline composed for the stand stages beside T*
    run = sn["evaluate"]
    env = run["env"]
    dl = _deliverable()
    cand = yaml.safe_load((F19 / "candidates.yaml").read_text())
    spec = env["SN_WALKER_SPEC"]
    # checkout-independent as committed: every path is $REPO/<repo-relative path>, none absolute
    leaves = [env["SN_WALKER_CK"], env["SN_WALKER_ST"], *(v for v in spec.values() if isinstance(v, str))]
    assert not [v for v in leaves if v.startswith("/")], leaves
    assert (env["SN_WALKER_CK"], str(env["P2_REANCHOR"])) == (f"{REPO_TOKEN}/{w['source']}", "1")
    assert _names_recorded(env["SN_WALKER_ST"], dl["tstar"]["driver_stamp"])
    assert spec["REFS_ENV_STAMP"] == env["SN_WALKER_ST"]
    here = _at_checkout(spec)
    assert here == {"CANDIDATES": str(F19 / "candidates.yaml"), "OVERLAY_CHAIN": "F19_p2a",
                    "REFS_ENV_STAMP": _at_checkout(env["SN_WALKER_ST"]), "REFS_NAME": "REFS_T4F19",
                    "S1_AGENT_YAML": str(REPO / Path(w["source"]).parent / "params" / "agent.yaml"),
                    "S1_CKPT": str(REPO / w["source"]), "S1_SHA256": w["sha256"],
                    "S1_OVERLAY": {k: str(v) for k, v in cand["chains"]["F19_p2a"]["stages"][0]["overlay"].items()}}
    spec_mod = _load_nobc(SN_SPEC, "sn_dagger_spec_policy_test")
    parsed = spec_mod.walker_spec_override({"SN_WALKER_SPEC": json.dumps(here, sort_keys=True)})
    assert parsed["S1_SHA256"] == w["sha256"] and parsed["REFS_NAME"] == "REFS_T4F19"
    text = (POLICY / "README.md").read_text()
    # the README's exports, run as written from parkour/ (bash), give exactly the resolved values at run time -- the
    # absolute paths every consumer requires (sn_dagger_spec refuses a relative one), this checkout's, not the writer's
    exports = [ln for ln in text.splitlines() if ln.startswith(("export SN_WALKER_", "export P2_REANCHOR="))]
    assert len(exports) == 4 and not [ln for ln in exports if "/home/" in ln], exports
    got = subprocess.run(["bash", "-c", "\n".join(exports) + '\nprintf "%s\\0" "$SN_WALKER_CK" "$SN_WALKER_ST" '
                          '"$P2_REANCHOR" "$SN_WALKER_SPEC"'], cwd=REPO / "parkour", capture_output=True, text=True,
                         check=True, env={"PATH": "/usr/bin:/bin"}).stdout.split("\0")
    assert got[:3] == [str(REPO / w["source"]), _at_checkout(env["SN_WALKER_ST"]), "1"]
    assert got[3] == json.dumps(here, sort_keys=True)
    sdir = PKG / run["scripts"]
    assert sdir == SH_CAMPAIGN / "scripts"
    cmds = [c["run"] for c in run["commands"]]
    named = set(re.findall(r"\$S/(\S+)", "\n".join(cmds)))
    assert named == {"sh_eval_rows.sh", "sh_table_test.sh", "sh_read.py", "sh_record_video.sh"}
    for f in named | {"sh_eval_crab_hex_gait.py", "sh_core.py", "sh_video_core.py", "sh_record_video.py"}:
        assert (sdir / f).is_file(), f
    assert (F19 / "stand" / "sh_lazy_rh_read_spec.json").is_file()
    for cmd in cmds:
        if "$CK" in cmd:
            assert f"--deadband {sn['deployment']['constants']['residual_deadband']}" in cmd, cmd
    assert sum(bool(re.search(r"\$CK 1 ", cmd)) for cmd in cmds) == 3          # rows: the network, mask 1
    vids = [ln for cmd in cmds for ln in cmd.splitlines() if "--stand-ckpt" in ln and "sh_table_test" not in ln]
    assert vids and all("--stand-mask 1" in ln and "--checkpoint $SN_WALKER_CK" in ln for ln in vids)
    taken = {h["id"] for h in sn["evals"]["heads"]} | {"SH_T17", "SH_T18", "SH_T16"}
    assert not {run["name"], run["name"] + "z"} & (taken | {h + "z" for h in taken}), "a re-run must use FRESH names"
    # the recorded stamps and video (untracked) ran under exactly T*'s eval env, beside T*, with this stand file
    stamps = [REPO / v for k, v in sn["evals"]["heads"][0]["stamps"].items() if "@" in k]
    for st in stamps:
        if (st / "run_meta.json").is_file():
            meta = json.loads((st / "run_meta.json").read_text())
            assert meta["env_vars"] == tool.eval_env(m, w) and meta["checkpoint_sha256"] == w["sha256"], st
            assert meta["p2r_mode"]["specialist"]["sha256"] == sn["sha256"], st
    vid = REPO / "parkour/logs/videos/sh_stand_height/sh_mix_F19_s1514_rough_sequence_sanity.json"
    if vid.is_file():
        v = json.loads(vid.read_text())
        assert (v["stand_checkpoint_sha256"], v["checkpoint_sha256"]) == (sn["sha256"], w["sha256"])
        assert (v["stand_mask"], v["residual_deadband"], v["restore"], v["falls"]) == (1, 0.25, "armed", [])


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


#: the stand gate table's columns -> (read, role in that read)
_SH_COLS = {"R_h_T": ("lazy", "R_h"), "SH_T4F19": ("lazy", "SH_T4F19_sh_s19"),
            "SH_T4F19_s1era": ("orig", "SH_T4F19_sh_s19"), "SH_T17": ("T17", "SH_T17"), "R_h_S1": ("T17", "R_h")}


def _sh_reads() -> dict:
    return {"lazy": json.loads((F19 / "stand/sh_lazy_rh_read.json").read_text()),
            "orig": json.loads((F19 / "stand/sh_read.json").read_text()),
            "T17": json.loads((SH_CAMPAIGN / "head_SH_T17.json").read_text())}


def test_stand_network_gate_table_matches_the_campaign_reads():
    """Every cell of the manifest's K1-K6 table is its read at the manifest's rounding: the re-anchored and the original
    F19 reads (sh_lazy_rh_read.json / sh_read.json) and the superseded SH_T17's (head_SH_T17.json); the lines name the
    re-anchored read's clause lines; each read carries the adopted checkpoint's sha and the trained deadband; the
    verdict row names exactly each column's non-PASS gates; the pipeline gate summary is gates/sh.json's; the head's
    stamps are the ones the re-anchored read used."""
    m = _m()
    sn = _stand(m)
    ev = sn["evals"]
    gt = ev["gate_table"]
    reads = _sh_reads()
    for r in ("lazy", "orig"):
        hd = reads[r]["heads"]["SH_T4F19_sh_s19"]
        assert hd["stand_sha256"] == [sn["sha256"]] and (hd["residual_deadband"], hd["trained_residual_deadband"]) == (
            0.25, 0.25), r
    assert reads["lazy"]["mismatches"] == [] and reads["lazy"]["pending"] == []
    assert reads["lazy"]["meta"]["spec_body"]["refs"] == "REFS_T4F19"
    assert [c["id"] for c in gt["columns"]] == list(_SH_COLS)

    def read(col, K):
        rd, role = _SH_COLS[col]
        return reads[rd]["gates"][K].get(role)

    assert [r["key"] for r in gt["rows"]] == list(_K_ROWS)
    for r in gt["rows"]:
        gate, kind, pat, scale, tol = _K_ROWS[r["key"]]
        assert r["gate"] == gate, r["key"]
        for col in gt["columns"]:
            g = read(col["id"], gate)
            if g is None:
                assert r[col["id"]] is None, (r["key"], col["id"])
                continue
            _k_close(r[col["id"]], _k_cell(g, kind, pat, scale), max(tol or 0.0, 5e-5), (r["key"], col["id"]))
        if kind in ("one", "max"):
            g = read("SH_T4F19", gate)
            lines = {_num_str(c["line"] * scale) for k in (1, 2) for c in _k_clauses(g, kind, pat, k)}
            assert lines and all(x in r["line"] for x in lines), (r["key"], lines, r["line"])
    for col in gt["columns"]:
        cid = col["id"]
        failing = [f"{K} {read(cid, K)['verdict']}" for K in _GATES
                   if read(cid, K) is not None and read(cid, K)["verdict"] != "PASS"]
        assert gt["verdicts"][cid].startswith(", ".join(failing) if failing else "PASS"), (cid, gt["verdicts"][cid])
    # the re-anchored read: everything but the K5 identity passes; the original read fails K4 / K6 (the HALT)
    assert reads["lazy"]["heads"]["SH_T4F19_sh_s19"]["failed"] == ["K5"]
    assert reads["orig"]["heads"]["SH_T4F19_sh_s19"]["failed"] == ["K4", "K5", "K6"]
    # the decisive pipeline gate: one failed K6 line, the counts per family the block states
    pg = ev["pipeline_gate"]
    g = json.loads((PKG / pg["record"]).read_text())
    assert (g["verdict"], g["thresholds"]["version"]) == (pg["verdict"], pg["thresholds"]) == ("HALT", "v1.2")
    assert g["halt_lines"] == [pg["halt_line"]]
    bad = [x for x in g["decisive"] if not x["pass"]]
    assert len(bad) == 1 and (bad[0]["value"], bad[0]["line"]) == (pg["halt_value"], pg["halt_line_value"])
    counts = {}
    for x in g["decisive"]:
        K = x["id"].split(".")[2]
        counts.setdefault(K, [0, 0])
        counts[K][0] += bool(x["pass"])
        counts[K][1] += 1
    assert {K: f"{a} / {b}" for K, (a, b) in counts.items()} == pg["lines"]
    assert g["head"]["sha256"] == sn["sha256"]
    assert "sh" in {h["stage"] for h in _deliverable()["accepted_halts"]}
    adopted = [h for h in ev["heads"] if h.get("adopted")]
    assert [h["id"] for h in adopted] == [sn["id"]] and adopted[0]["sha256"] == sn["sha256"]
    assert adopted[0]["source"] == sn["source"]
    rd = reads["lazy"]["reads"]["SH_T4F19_sh_s19"]
    for k in (1, 2):
        for row in ("hwalkstop__omni", "hraise__omni"):
            assert rd[f"es{k}"][row]["stamp"].endswith(adopted[0]["stamps"][f"{row}@{k}"]), (row, k)
    # the K3 ungated cells the scope / caveat state are the read's
    rep = reads["lazy"]["gates"]["K3"]["SH_T4F19_sh_s19"]["reported"]
    for cell, txt in (("ground__knee_folded", "ground knee_folded"), ("ground__random", "ground random")):
        pair = " / ".join(f"{rep[f'es{k}']['per_cell'][cell]['stood_share']:.2f}" for k in (1, 2))
        assert f"{txt} STOOD {pair}" in sn["deployment"]["scope"] or f"{txt} {pair}" in sn["deployment"]["scope"], cell


def test_readme_has_the_stand_network_section():
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    text = (POLICY / "README.md").read_text()
    assert text == tool.readme_text(m)
    path = f"{sn['dir']}/{sn['file']}"
    assert "## Stand network of record" in text and "### Deployment contract (as evaluated)" in text
    assert f"**Stand network of record:** [`{path}`]({path})" in text
    assert "stand [`f19_stand_net/`](f19_stand_net/) (`model_5000.pt`)" in text
    # the warm start is named ONLY as the bundled ancestor, never as the net of record
    ws_lines = [line for line in text.splitlines() if "f19_stand_net/model_4000.pt" in line]
    assert len(ws_lines) == 1 and ws_lines[0].startswith(
        "- **Warm-start ancestor (bundled, NOT the stand network of record):** "
        "[`f19_stand_net/model_4000.pt`](f19_stand_net/model_4000.pt)")
    assert sn["warm_start"]["sha256"] in ws_lines[0] and sn["warm_start"]["role"] in ws_lines[0]
    # the superseded stand_net/ files appear ONLY in the superseded section
    for line in text.splitlines():
        if "stand_net/model_5000.pt" in line and "f19_stand_net" not in line:
            assert line.startswith("  - [`stand_net/model_5000.pt`]"), line
    assert sn["sha256"] in text and sn["walker"]["sha256"] in text and sn["source"] in text
    for s in ("stand_net_input", "hist_encoding=True", "CAVEAT (switch keying)", "CAVEAT (height source)",
              "**Zero-height mode:**", "**Restore before hand-back:**", "**Height command:**", "**Posture map q(h):**",
              "deadband(clamp(a_net, +-1), 0.25)", "  - **residual_deadband:** 0.25", "**Adoption grade:**",
              "### Gates K1-K6 beside T*", "- **Training:**", "  - **trim_min:** 0.0999",
              "export SN_WALKER_CK=", "export P2_REANCHOR=1"):
        assert s in text, s
    assert f"CK=$PWD/$P/policy/{path}" in text
    assert "bash $S/sh_eval_rows.sh --deadband 0.25 SH_T4F19_re stand_net manifest armed $CK 1 \\" in text
    assert "bash $S/sh_eval_rows.sh --deadband 0.25 SH_T4F19_rez stand_net zero armed $CK 1 \\" in text
    assert ("| Gate | Statistic | Line | R_h beside T* (lazy control) | SH_T4F19 re-anchored (adopted) | "
            "SH_T4F19 vs the S1-era lines | SH_T17 beside S1 (superseded) | R_h beside S1 |") in text
    assert ("| K4 | restore misses (of 800 restore-required exits) | <= max(1, R_h's) = 5 (R_h beside T*) | 5 / 5 | "
            "2 / 3 | 2 / 3 | 0 / 1 | 1 / 1 |") in text
    assert "| K5 | G1-G5 at 1.064 (sn_calib, zero-command rows) as the reader reads it |" in text
    for c in sn["caveats"]:
        assert c in text
    for f in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "two_network_export", "stand_ground_getup",
              "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
              "stand_knee_hold_only", "stand_k5_identity"):
        assert f"### `{f}` --" in text


def test_followups_track_the_new_head():
    """Every follow-up carried over is revised for the F19 bake (2026-10-24) and keeps the schema; the stand ones name
    SH_T4F19; the four new ones (camera use, sim-plant payload, the unbundled student, the K5 identity) exist and cite
    tracked evidence."""
    m = _m()
    fu = {f["id"]: f for f in m["followups"]}
    carried = ("move_turn", "c0_side_first_n1", "action_jitter", "phase2_clock_free", "stand_march",
               "stand_switch_keying", "stand_trim_underfit", "two_network_export", "stand_ground_getup",
               "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
               "stand_knee_hold_only", "p2b_negative", "back_overshoot")
    new = ("camera_use_unverified", "payload_sim_plant", "student_checkpoint_not_bundled", "stand_k5_identity")
    assert list(fu) == [*carried, *new]
    for k in carried:
        assert fu[k]["recorded"].endswith("revised 2026-10-24 (F19 bake)") and fu[k]["detail"].startswith(
            "F19 BAKE (2026-10-24)"), k
    for k in new:
        assert fu[k]["recorded"].startswith("2026-10-24"), k
    for k in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "stand_ground_getup",
              "stand_pd_torque_excess", "stand_k5_identity"):
        assert "SH_T4F19" in fu[k]["detail"], k
    for k in (*carried, *new):
        f = fu[k]
        assert {"id", "recorded", "headline", "detail"} <= set(f), k
        for e in f.get("evidence") or []:
            rel = e.split(" (")[0]
            assert (PKG / rel).exists(), (k, rel)
    assert "0.741" in fu["move_turn"]["detail"] and "0.658" in fu["move_turn"]["detail"]
    assert "RESOLVED" in fu["back_overshoot"]["detail"] and "0.993" in fu["back_overshoot"]["detail"]
    assert "0.04 / 0.06" in fu["stand_ground_getup"]["detail"]
    assert "4fa81f0f" in fu["student_checkpoint_not_bundled"]["detail"]


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
    assert "## Stand network of record" not in without
    assert not any(line.startswith("**Stand network of record:**") or "stand [`f19_stand_net/`]" in line
                   or line.startswith("- Stand network (`stand_network:`") for line in without)
    ops = [op for op in difflib.SequenceMatcher(a=without, b=with_sn, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert ops and all(tag in ("insert", "replace") for tag, *_ in ops), ops
    removed = [line for tag, i1, i2, _, _ in ops if tag == "replace" for line in without[i1:i2]]
    heads_old = [line for line in without if line.startswith("**Head of record per phase:**")]
    assert removed == heads_old
    heads_new = [line for line in with_sn if line.startswith("**Head of record per phase:**")]
    assert heads_new == [heads_old[0][:-1] + ", stand [`f19_stand_net/`](f19_stand_net/) (`model_5000.pt`)."]


def test_readme_without_the_superseded_files_is_unchanged():
    """The superseded-kept section is purely additive too: without `files:` the README loses that section only."""
    tool = _tool()
    m = tool.load_manifest()
    bare = copy.deepcopy(m)
    for e in bare["superseded"]:
        e.pop("files", None)
    with_k = tool.readme_text(m).splitlines()
    without = tool.readme_text(bare).splitlines()
    assert "## Superseded heads kept in this folder" in with_k and "## Superseded heads kept in this folder" not in without
    ops = [op for op in difflib.SequenceMatcher(a=without, b=with_k, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert ops and all(tag == "insert" for tag, *_ in ops), ops
    assert tool.expected_files(bare) == tool.expected_files(m) - {f"{f['dir']}/{f['file']}" for e in m["superseded"]
                                                                   for f in e.get("files") or []}


# ------------------------------------------------------------------------------ the tool's validation and checks
def _mutated_manifest(tmp_path: Path, mutate) -> Path:
    m = _m()
    mutate(m)
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(m, sort_keys=False))
    return p


def _st(m, d):
    return next(s for s in m["stages"] if s["dir"] == d)


@pytest.mark.parametrize("mutate, why", [
    (lambda m: m["stand_network"].update(dir="f19_8_p3b_payload180"), "collides with a stage dir"),
    (lambda m: m["stand_network"].update(dir="rsi"), "collides with an asset dir"),
    (lambda m: (m["stand_network"].update(dir="stand_net"), m["heads"].update(stand="stand_net")),
     "is a live policy file with another sha256"),
    (lambda m: m["stand_network"].pop("sha256"), "missing 'sha256'"),
    (lambda m: m["stand_network"].update(sha256="c1269234"), "full lowercase sha256"),
    (lambda m: m["stand_network"].update(stand_mask=True), "int 0 or 1"),
    (lambda m: m["stand_network"]["walker"].update(sha256="0" * 64), "walker.sha256"),
    (lambda m: m["stand_network"]["walker"].update(stage="7_nope"), "walker.stage"),
    (lambda m: m["stand_network"]["walker"].update(stage="f19_6_p2a_field"), "walker.sha256"),
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
    (lambda m: m["stand_network"]["warm_start"].update(sha256="96832099"), "warm_start.sha256 must be a full"),
    (lambda m: m["stand_network"]["warm_start"].update(dir="rsi"), "warm_start lives in the stand network's folder"),
    (lambda m: m["stand_network"]["warm_start"].update(file="model_5000.pt"), "different file from the stand network"),
    (lambda m: m["stand_network"]["warm_start"].update(sha256=m["stand_network"]["sha256"]),
     "different file from the stand network"),
    # the README-facing walker env is checkout-independent: $REPO/<path>, never an absolute path
    (lambda m: m["stand_network"]["evaluate"]["env"].update(SN_WALKER_CK="/home/x/krabby-research/parkour/m.pt"),
     "stand_network.evaluate.env.SN_WALKER_CK is an absolute path"),
    (lambda m: m["stand_network"]["evaluate"]["env"]["SN_WALKER_SPEC"].update(CANDIDATES="/r/parkour/c.yaml"),
     "stand_network.evaluate.env.SN_WALKER_SPEC.CANDIDATES is an absolute path"),
    (lambda m: m["stand_network"]["evaluate"]["env"].update(SN_WALKER_ST="$REPO/parkour/a b"), "path must match"),
    # stage kinds, the distill stage and its exports
    (lambda m: _st(m, "f19_1_s1").update(kind="ppo"), "kind 'ppo' is not one of"),
    (lambda m: _st(m, "f19_1_s1").update(bundled=False), "for kind 'distill' stages only"),
    (lambda m: _st(m, "f19_9_student").pop("exports"), "is not bundled and declares no exports"),
    (lambda m: _st(m, "f19_9_student").update(exports=[]), "exports must be a non-empty list"),
    (lambda m: _st(m, "f19_9_student").update(bundled="no"), "bundled must be true or false"),
    (lambda m: _st(m, "f19_9_student")["exports"][0].pop("sha256"), "missing 'sha256'"),
    (lambda m: _st(m, "f19_9_student")["exports"][0].update(sha256="4fb3efe1"), "must be a full lowercase sha256"),
    (lambda m: _st(m, "f19_9_student")["exports"][0].update(file="model_28000.pt"), "plain non-checkpoint file"),
    (lambda m: _st(m, "f19_9_student")["exports"][1].update(file="policy.onnx"), "must be distinct files"),
    (lambda m: m.update(current="f19_9_student"), "is a stage whose checkpoint is not bundled"),
    (lambda m: _st(m, "f19_2_c4").update(sha256="0cbab2c4"), "sha256 must be a full lowercase sha256"),
    # superseded files kept in the folder
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir="f19_2_c4"), "sits in a live stage / asset / stand folder"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir="f19_1_s1"), "is a live policy file with another sha256"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir="rsi"), "sits in a live stage / asset / stand folder"),
    (lambda m: _sup(m, "S1_s18")["files"][0].pop("sha256"), "missing 'sha256'"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(sha256="72e4b1e3"), "must be a full lowercase sha256"),
    (lambda m: _sup(m, "SH_T17")["files"].append(dict(_sup(m, "SN_Amask_s17")["files"][0])), "recorded twice"),
    (lambda m: _sup(m, "SH_T17")["files"].append({"dir": "f19_stand_net", "file": "model_4000.pt", "sha256": "0" * 64,
                                                   "source": "x"}), "is a live policy file with another sha256"),
    (lambda m: _sup(m, "SH_T17").update(files=[]), "files must be a non-empty list"),
    (lambda m: m["superseded"].append({"files": []}), "mapping with an id"),
])
def test_manifest_is_validated(tmp_path, mutate, why):
    tool = _tool()
    with pytest.raises(SystemExit, match=re.escape(why)):
        tool.load_manifest(_mutated_manifest(tmp_path, mutate))


def test_stand_env_values_render_checkout_independent():
    """``evaluate.env`` -> ``export K=<word>`` run from parkour/: a ``$REPO/`` path becomes ``"$PWD/..."``, a mapping
    its sort_keys JSON with each such path spliced in double quotes, anything else single-quoted as is."""
    tool = _tool()
    assert tool._shell_env_value("$REPO/parkour/logs/x.pt") == '"$PWD/logs/x.pt"'
    assert tool._shell_env_value("$REPO/assets/crab.usda") == '"$PWD/../assets/crab.usda"'
    assert tool._shell_env_value("1") == "1" and tool._shell_env_value("") == "''"
    assert tool._shell_env_value("a b") == "'a b'"
    word = tool._shell_env_value({"B": "it's", "A": "$REPO/parkour/c.yaml", "C": {"K": ""}})
    assert word == '\'{"A": "\'"$PWD/c.yaml"\'", "B": "it\'"\'"\'s", "C": {"K": ""}}\''
    got = subprocess.run(["bash", "-c", f"X={word}; printf %s \"$X\""], cwd=REPO / "parkour", capture_output=True,
                         text=True, check=True, env={"PATH": "/usr/bin:/bin"}).stdout
    assert json.loads(got) == {"A": str(REPO / "parkour/c.yaml"), "B": "it's", "C": {"K": ""}}
    # no path inside a mapping: exactly the old rendering (one single-quoted JSON word)
    assert tool._shell_env_value({"K": "v"}) == "'{\"K\": \"v\"}'"


def _sandbox(tmp_path: Path, tool, m: dict) -> Path:
    """A policy/ tree under ``tmp_path`` whose files are SYMLINKS to the real ones (nothing large is copied), with the
    README rendered for ``m``; points the tool's REPO / POLICY at it."""
    pol = tmp_path / "policy"
    for rel in tool.expected_files(m):
        if rel in ("manifest.yaml", "README.md"):
            continue
        dst = pol / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(POLICY / rel)
    tool.REPO, tool.POLICY = tmp_path, pol
    (pol / "README.md").write_text(tool.readme_text(m))
    return pol


def test_check_flags_problems_in_kept_and_distill_folders(tmp_path):
    """--check on a sandbox copy: clean as is; flags a missing kept file, a stray checkpoint beside kept files, the
    student checkpoint dropped into its (unbundled) folder, a wrong sha on a kept file and a stale README."""
    tool = _load(TOOL, "bundle_policy_omni_sandbox")
    m = tool.load_manifest()
    pol = _sandbox(tmp_path, tool, m)
    assert tool.check(m) == []
    (pol / "stand_net" / "model_4000.pt").unlink()
    (pol / "stand_net" / "model_9.pt").write_bytes(b"x")
    (pol / STUDENT / "model_28000.pt").write_bytes(b"x")
    bad = copy.deepcopy(m)
    _sup(bad, "S1_s18")["files"][0]["sha256"] = "0" * 64
    probs = "\n".join(tool.check(bad))
    assert "stand_net/model_4000.pt: missing" in probs
    assert "stand_net: stray checkpoints ['model_9.pt']" in probs
    assert f"{STUDENT}/model_28000.pt: stage '{STUDENT}' is `bundled: false`" in probs
    assert "1_fb6k/model_5999.pt: sha256" in probs
    assert "README.md: stale" in probs


def test_sync_refuses_a_file_over_the_per_file_limit(tmp_path):
    """--sync never bundles a file over GitHub's 100 MiB limit (a sparse 101 MiB source is refused before hashing)."""
    tool = _load(TOOL, "bundle_policy_omni_limit")
    big = tmp_path / "big.pt"
    with big.open("wb") as fh:
        fh.truncate(tool.MAX_FILE_BYTES + 1)
    tool.REPO = tmp_path
    with pytest.raises(SystemExit, match="100 MiB per-file limit"):
        tool._sync_one(tmp_path / "policy" / "x" / "big.pt", big, "0" * 64, "x", force=False)
    assert not (tmp_path / "policy").exists()


def test_sync_verifies_kept_files_and_never_touches_them(tmp_path):
    tool = _load(TOOL, "bundle_policy_omni_kept")
    m = tool.load_manifest()
    _sandbox(tmp_path, tool, m)
    kept = tool.kept_files(m)
    assert {f"{k['dir']}/{k['file']}" for k in kept} == {f"{d}/{s['file']}" for d, s in zip(
        S18_STAGES, _sup(m, "S1_s18")["stages"])} | {"stand_net/model_4000.pt", "stand_net/model_5000.pt"}
    for k in kept:
        tool._verify_kept(k)
    (tool.POLICY / "6_p2a_field" / "model_19994.pt").unlink()
    with pytest.raises(SystemExit, match="never recreates a superseded file"):
        tool._verify_kept(next(k for k in kept if k["dir"] == "6_p2a_field"))
