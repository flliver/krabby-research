"""The omni-walk POLICY OF RECORD (``crab_hex_omni_task/policy/``) stays consistent.

``policy/manifest.yaml`` declares the shipped head (``current``) and the stage heads it was trained through. Since the
2026-11-09 bake (user decision "Bake as proposed", experiments/2026-11-07_rod_speed_dr CHARTER section 6, 23:08) that is
the ROD-SPEED DR PAIRING at seed 19: the Task 4 deliverable run F19's stages f19_1_s1 (preset 1b) -> f19_2_c4 -> f19_3_c5
-> f19_4_c6 -> f19_5_c6x_clockfree (preset 1a) -> f19_6_p2a_field -> f19_7_p3a_payload90 -> f19_8_p3b_payload180 (preset
2a; = T*, still a live stage), each with its overlay from F19/candidates.yaml, then rsd_9_p3b_rodspeed_dr (= TDR, current:
T* + 3000 iterations under the training-only per-env rod-speed DR, KRABBY_DR_ROD_SPEED) and the depth student
rsd_10_student (= c31000, kind ``distill``, distilled from TDR with the DR on; its 122 MB checkpoint is NOT bundled --
``bundled: false``, cited by sha256 -- and its folder holds the ONNX deploy export + io json). The shared training assets
(``assets:``, the omni RSI bank) stay. Every declared file must be present, tracked by git, sha-pinned to the manifest,
and the only checkpoint in its folder; the README is generated from the manifest by ``experiments/tools/bundle_policy.py
--sync``. The c6x and later heads were trained with the gait clock hidden, and their observation contract must say so.
Every stage's eval environment must be exactly what the driver's suite eval builds from that stage's preset + overlay,
the tracked F19 env file and (when on disk) its eval stamp's run_meta; the DR knob never reaches it.

The STAND NETWORK OF RECORD (``stand_network:``, folder ``rsd_stand_net/``) is SH_RSD_TDR_sh_s19, trained and gated
beside TDR, with SN_RSD_TDR_sn_s19 its bundled warm-start ancestor; its contract constants must be the stand-height
campaign's ``sh_core`` and the checkpoint's; its routing env and walker override must be TDR's (the campaign's
results/stand/walker_spec.json); its K1-K6 table must be the campaign's stand reads (results/stand/sh_lazy_rh_read.json /
sh_read.json), the superseded SH_T4F19's (F19/stand/sh_lazy_rh_read.json) and R_h beside S1 (head_SH_T17.json); its
pipeline gate (results/stand/gates/sh.json) HALTed, and every failed line is re-anchored by a recorded rule
(gates/sh_reanchor.json) or accepted and disclosed by the user (gates/sh_decision.json, 18:10 "Accept and disclose").

SUPERSEDED, KEPT: nothing was deleted at the 2026-10-24 and 2026-11-09 bakes. The seed-18 lineage (1_fb6k .. 6_p2a_field,
S1 s18), SH_T17 (stand_net/model_5000.pt), A-mask s17 (stand_net/model_4000.pt -- the fixed path sh_core and the stand
campaign tests load), F19's student exports (f19_9_student/, F19_student) and F19's stand nets (f19_stand_net/,
SH_T4F19 + SN_T4F19) stay in ``policy/`` as ``superseded[].files``: present, sha-pinned, byte-identical to the files they
were baked with, and their full records kept verbatim (equal to the manifest at ``record_commit``). F19 itself is
recorded as the superseded ``current`` (its stages stay live). W (superseded 2026-10-06) left the folder and stays
recoverable from git history.

Untracked inputs (run dirs, stamps, the stand runs, the videos) are read from this checkout or, in a git worktree, from
the main checkout; every test that needs one skips when it is on neither (CI's depth-1 checkout has none, and no history:
the history checks skip naming the missing commits).
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
DR_VAR = "KRABBY_DR_ROD_SPEED"

#: the Task 4 campaign and its deliverable run F19 (every record below is tracked)
T4 = PKG / "experiments" / "2026-10-16_task4_pipeline"
F19 = T4 / "F19"
F19_STAGES = ["f19_1_s1", "f19_2_c4", "f19_3_c5", "f19_4_c6", "f19_5_c6x_clockfree", "f19_6_p2a_field",
              "f19_7_p3a_payload90", "f19_8_p3b_payload180"]
#: the F19 pipeline's stage id of each F19 policy stage (F19/plan.json, F19/deliverable.json stage_heads_by_sha256)
F19_ID = dict(zip(F19_STAGES, ["s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b"]))
TSTAR = "f19_8_p3b_payload180"
#: the rod-speed DR campaign (its results/ are tracked: runs, gates, selection, the stand re-pairing)
RSD = PKG / "experiments" / "2026-11-07_rod_speed_dr"
RES = RSD / "results"
TDR = "rsd_9_p3b_rodspeed_dr"
STUDENT = "rsd_10_student"
STAGES = [*F19_STAGES, TDR, STUDENT]
STAND_DIR = "rsd_stand_net"
#: the superseded F19 student / stand folders, kept
OLD_STUDENT = "f19_9_student"
OLD_STAND = "f19_stand_net"
#: the stages trained with the clock observation hidden (c6x and every stage after it, the student included)
CLOCK_HIDDEN = {"f19_5_c6x_clockfree", "f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR, TDR, STUDENT}
#: the superseded seed-18 lineage, kept in the folder
S18_STAGES = ["1_fb6k", "2_c4", "3_c5", "4_c6", "5_c6x_clockfree", "6_p2a_field"]
#: the commits the superseded F19 entries name: the manifest before the 2026-11-09 bake, and the F19 bake itself
F19_RECORD_COMMIT = "64f397e5621ffab2432e4a13542a34a5845f9a88"
F19_BAKE_COMMIT = "de81f98ad66df4f931fb595a7facf5743ae42eeb"
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
#: the two walker + stand mix videos of the 2026-11-09 bake (untracked; written by the main session's recorder)
RSD_VIDEOS = {"field_medium": "parkour/logs/videos/sh_stand_height/sh_mix_RSD_s1514_rough_sequence_sanity.json",
              "field_max": "parkour/logs/videos/sh_stand_height/sh_mix_RSD_s1514_roughest_sequence_sanity.json"}
#: the two reader mismatches of the stand reads beside TDR: both K5 control cells cannot be read
K5_CONTROL_INVALID = [
    "K5 NON-REGRESSION @1.064 x R_h (R_h (height-aware R)): INVALID (a listed cell cannot be read)",
    "K5 NON-REGRESSION @1.064 x Amask (A-mask s17): INVALID (a listed cell cannot be read)"]


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


def _json(p: Path) -> dict:
    return json.loads(p.read_text())


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


def _checkouts() -> list[Path]:
    """This checkout, then (in a git worktree) the main checkout: untracked run data lives in one of them."""
    roots = [REPO]
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        common = Path(out.stdout.strip())
        if common.name == ".git" and common.parent.resolve() != REPO.resolve():
            roots.append(common.parent)
    return roots


def _local(rel: str) -> Path | None:
    """An untracked repo-relative file / dir (a run file, a stamp, a video record) in this checkout or the main one."""
    for root in _checkouts():
        p = root / rel
        if p.exists():
            return p
    return None


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


def _krabby(d: dict) -> dict[str, str]:
    return {k: v for k, v in d.items() if k.startswith("KRABBY_")}


def _deliverable() -> dict:
    return _json(F19 / "deliverable.json")


def _sup(m: dict, sid: str) -> dict:
    return next(e for e in m["superseded"] if e["id"] == sid)


def _rel_logs(p):
    return p[p.index("parkour/"):] if p else None


def _stage(m: dict, d: str) -> dict:
    return next(s for s in m["stages"] if s["dir"] == d)


def _cell_kg(cell: str) -> int:
    mo = re.search(r"_L(\d{3})(?:_|$)", cell)
    assert mo, cell
    return int(mo.group(1))


#: the stress / nominal cells the TDR and student stress blocks name -> the campaign's cell ids
_STRESS_CELLS = {"x0p7_L090": "T01_rod_speed_0p7_L090", "x0p8_L090": "T01_rod_speed_0p8_L090",
                 "x0p7_L000": "T01_rod_speed_0p7_L000", "x1p35_L000": "T01_rod_speed_1p35_L000",
                 "x1p35_L090": "T01_rod_speed_1p35_L090"}


def _seed_values(cell: dict, key: str = "field_light_completion") -> list:
    return [cell["seeds"][k]["values"][key] for k in ("1", "2")]


# ------------------------------------------------------------------------------ the lineage of record
def test_manifest_and_files_are_consistent():
    tool = _tool()
    m = tool.load_manifest()
    assert m["task"] == "crab_hex_omni_task" and m["plant"] == "A15+B" and m["seed"] == 19
    problems = tool.check(m)
    assert not problems, problems


def test_stage_order_and_chain():
    m = _m()
    assert [s["dir"] for s in m["stages"]] == STAGES
    assert [s["phase"] for s in m["stages"]] == ["1b"] + ["1a"] * 4 + ["2a"] * 5
    assert [s["stage"] for s in m["stages"]] == ["s1", "c4", "c5", "c6", "c6x", "p2a", "p3a", "p3b", "p3b_dr", "st_dr"]
    assert [s.get("kind") for s in m["stages"]] == ["rl"] * 9 + ["distill"]
    assert m["stages"][0]["resume_from"] == "scratch"
    for prev, s in zip(m["stages"], m["stages"][1:]):
        assert prev["dir"] in s["resume_from"], f"{s['dir']} must resume from {prev['dir']}"
    # current = TDR = the last RL stage; the student (distill) follows it and is not the walking head
    assert m["current"] == m["stages"][-2]["dir"] == TDR
    assert m["heads"] == {"phase1": "f19_5_c6x_clockfree", "phase2": "f19_6_p2a_field", "payload": TDR,
                          "student": STUDENT, "stand": STAND_DIR}
    assert m["heads"]["stand"] not in STAGES
    # the new heads take folders of their own: no stage or the stand net reuses a superseded (kept) folder
    kept_dirs = {f["dir"] for e in m["superseded"] for f in e.get("files") or []}
    assert kept_dirs == {*S18_STAGES, "stand_net", OLD_STUDENT, OLD_STAND}
    assert not (set(STAGES) | {STAND_DIR}) & kept_dirs


def test_stage_files_are_one_resume_chain():
    """Each F19 stage file is the checkpoint the next stage resumed from, read from the F19 driver's own tracked state
    (F19/state.json, whose candidates.yaml must still be the one the driver ran), at the manifest seed, with exactly the
    sha256 the deliverable record lists; TDR is the campaign's run RSD_TDR_s19 resumed from the bundled T* file (its
    run record, the campaign's policy registry and G1 name the same file and sha); the distill stage is the student the
    campaign distilled from TDR and selected (results/sel.json, the distill run record)."""
    m = _m()
    dl = _deliverable()
    heads = dl["stage_heads_by_sha256"]
    st = _json(F19 / "state.json")
    assert (hashlib.sha256((F19 / "candidates.yaml").read_bytes()).hexdigest()
            == st["ladder"]["candidates_sha256"])
    by_ckpt = {_rel_logs(e["ckpt"]): e for e in st["keys"][f"seed{m['seed']}"].values()
               if isinstance(e, dict) and e.get("ckpt")}
    prev = None
    n_driver = 0
    for s in m["stages"][:len(F19_STAGES)]:
        assert s["dir"] in F19_STAGES and s["kind"] == "rl", s["dir"]
        assert s["overlay_ref"].startswith("experiments/2026-10-16_task4_pipeline/F19/candidates.yaml "), s["dir"]
        assert heads[F19_ID[s["dir"]]]["sha256"] == s["sha256"], s["dir"]
        assert _rel_logs(heads[F19_ID[s["dir"]]]["ckpt"]) == s["source"], s["dir"]
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
    assert n_driver == 8
    assert dl["tstar"]["stage"] == "p3b" and _rel_logs(dl["tstar"]["ckpt"]) == m["stages"][7]["source"]
    # TDR: the campaign's run RSD_TDR_s19, resumed from the BUNDLED T* file (by path, with T*'s alg_state counter)
    tstar, tdr, stu = _stage(m, TSTAR), _stage(m, TDR), _stage(m, STUDENT)
    pol = _json(RES / "policies.json")["policies"]
    run = _json(PKG / tdr["run_record"])
    assert tdr["run_record"] == "experiments/2026-11-07_rod_speed_dr/results/runs/RSD_TDR_s19.json"
    final = run["pf14"]["data"]["final"]
    assert pol["TDR"]["sha256"] == tdr["sha256"] == final["sha256"]
    assert _rel_logs(pol["TDR"]["checkpoint"]) == tdr["source"] == _rel_logs(final["path"])
    assert (run["status"], run["rc"], run["task"], run["arm"], run["name"]) == ("ok", 0, tdr["task"], "dr", "RSD_TDR_s19")
    assert final["iter"] == 28991 and tdr["file"] == f"model_{final['iter']}.pt"
    assert all(p["ok"] for p in run["preflight"].values()) and run["pf14"]["ok"]
    rel_pol = POLICY.relative_to(REPO)
    assert run["cli"]["checkpoint"].endswith(f"/{rel_pol}/{TSTAR}/{tstar['file']}")
    facts = run["preflight"]["PF3"]["data"]["facts"]
    assert (facts["sha256"], facts["iter"], facts["counter"]) == (tstar["sha256"], 25992, 27300)
    assert f"counter {facts['counter']}" in tdr["resume_from"]
    argv = run["trainer_argv"]
    assert argv[argv.index("--seed") + 1] == str(m["seed"]) and "--resume" in argv
    assert pol["TSTAR"]["sha256"] == tstar["sha256"]
    # the student: distilled from TDR, selected by the campaign's sel stage, its run dir the source's
    sel = _json(RES / "sel.json")
    pick = sel["pick"]
    assert (pick["sha256"], _rel_logs(pick["ckpt"]), pick["teacher"], pick["cand"]) == (
        stu["sha256"], stu["source"], "TDR", stu["distill"]["selected"])
    assert sel["verdict"] == "SELECTED" and stu["file"] == f"model_{pick['iter']}.pt"
    assert pol[pick["policy"]]["sha256"] == stu["sha256"] and _rel_logs(pol[pick["policy"]]["checkpoint"]) == stu["source"]
    dj = _json(RES / "runs" / "RSD_ST_s19.distill.json")
    assert dj["teacher_sha256"] == tdr["sha256"] and _rel_logs(dj["teacher"]) == tdr["source"]
    assert dj["name"] == stu["distill"]["run_name"] and Path(stu["source"]).parent.name.endswith("_" + dj["name"])
    assert dj["env_file"].endswith("/" + stu["distill"]["env_file"]) and stu["distill"]["teacher"] == TDR
    assert [c["iter"] for c in sel["candidates"]] == stu["distill"]["candidates"]
    assert sel["ranked"] == [f"c{i}" for i in stu["distill"]["prescreen_rank"]]
    assert stu["distill"]["seed"] == m["seed"]


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
    # the superseded-but-kept files are expected, tracked and on disk; neither student checkpoint is among them
    kept = {f"{rel}/{f['dir']}/{f['file']}" for e in m["superseded"] for f in e.get("files") or []}
    assert kept == {f"{rel}/{d}/{s['file']}" for d, s in zip(S18_STAGES, _sup(m, "S1_s18")["stages"])} | {
        f"{rel}/stand_net/model_4000.pt", f"{rel}/stand_net/model_5000.pt",
        *(f"{rel}/{OLD_STUDENT}/{f}" for f in ("policy.onnx", "depth_latest.onnx", "omni_student_io.json")),
        f"{rel}/{OLD_STAND}/model_5000.pt", f"{rel}/{OLD_STAND}/model_4000.pt"}
    assert kept <= expected & tracked & on_disk
    st = _stage(m, STUDENT)
    old = _sup(m, "F19_student")["block"]
    for d, s in ((STUDENT, st), (OLD_STUDENT, old)):
        assert f"{rel}/{d}/{s['file']}" not in expected | tracked | on_disk, d
        assert not list((POLICY / d).glob("*.pt")), d


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
    # TDR ships; the README names every head and the teacher task
    assert f"policy/{TDR}/model_28991.pt" in text and "Isaac-Crab-Hex-Omni-Teacher-v0" in text
    for k, d in (("phase1", "f19_5_c6x_clockfree"), ("phase2", "f19_6_p2a_field"), ("payload", TDR),
                 ("student", STUDENT), ("stand", STAND_DIR)):
        assert f"{k} [`{d}/`]" in text, k
    # T* and p2a stay live stages with their evals of record; TDR's rows / payload read are the manifest's
    assert f"| `{TSTAR}/` | `field_hard` | 0.905 | 200 |" in text
    assert "| `f19_6_p2a_field/` | `field_hard` | 0.7725 | 400 |" in text
    assert f"| `{TSTAR}/` | `0:180` | 0 yes, 45 yes, 90 yes, 135 yes, 180 NO | 135 | 180 |" in text
    tdr = tool.stage_by_dir(m, TDR)
    for row, r in tdr["evals"]["rows"].items():
        assert f"| `{TDR}/` | `{row}` | {r['completion']} | {r['n']} |" in text, row
    assert f"| `{TDR}/` | `0:180` | 0 yes, 90 yes, 135 yes | >= 135 (180 not read) | not read |" in text
    # the student section: its own task and role, the cited (not bundled) checkpoint, the bundled exports
    st = tool.stage_by_dir(m, STUDENT)
    assert f"## The depth student -- [`{STUDENT}/`]({STUDENT}/) (kind `distill`)" in text
    assert f"## The depth student -- [`{OLD_STUDENT}/`]" not in text
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
    assert f"**Current head:** [`{TDR}/model_28991.pt`]" in text


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
    # TDR's composed train env (its run record, RSD_TDR_s19.train.env) turns RSI off too (the bank comes from the
    # preset, KRABBY_OMNI_PHASE 2a); the asset's used_by says so
    tdr = tool.stage_by_dir(m, TDR)
    run_env = _json(PKG / tdr["run_record"])["env"]["krabby"]
    for env in (run_env, _env_file(PKG / tdr["env_file"])):
        assert env["KRABBY_RSI_FRAC"] == "0" and env["KRABBY_OMNI_PHASE"] == tdr["phase"]
        assert "KRABBY_RSI_BANK" not in env or env["KRABBY_RSI_BANK"].endswith("/" + str(preset_bank.relative_to(REPO)))
    assert tdr["env_file"] in banks[0]["used_by"] and "KRABBY_RSI_FRAC=0" in banks[0]["used_by"]


def test_obs_contract_current_head_is_clock_hidden():
    """c6x and every later head (the student included) trained with KRABBY_OMNI_CLOCK_HIDE=1; the earlier stages with the
    clock visible; the manifest's carried semantics cover exactly the driver's EVAL_CARRY set minus the body-only
    envelope keys (every head here is a WORLD-frame head), and every stage's full eval env is EXACTLY what the driver's
    suite eval (run_phases.eval_morph: keep CANARY_KEYS from ph.FORMATION) builds from that stage's preset + overlay --
    and exactly the tracked F19 env file the pipeline evaluated it with (TDR: T*'s; the student: its selection rows').
    The training-only rod-speed DR knob is in no eval env and is not carried."""
    tool = _tool()
    m = tool.load_manifest()
    cur = tool.stage_by_dir(m, m["current"])
    assert str(cur["obs_contract"][HIDE]) == "1"
    assert tool.eval_env(m, cur)[HIDE] == "1"
    assert tool.play_env(m, cur)[HIDE] == "1"
    for s in m["stages"]:
        assert str(s["obs_contract"][HIDE]) == ("1" if s["dir"] in CLOCK_HIDDEN else "0"), s["dir"]
    # phase 2, the payload stages, TDR and the student did not change the policy's inputs: their eval env is c6x's
    c6x = tool.eval_env(m, tool.stage_by_dir(m, "f19_5_c6x_clockfree"))
    for d in ("f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR, TDR, STUDENT):
        assert tool.eval_env(m, tool.stage_by_dir(m, d)) == c6x, d
    assert m["obs_contract"]["width"] == 1149
    rp = _load(RUN_PHASES, "omni_run_phases_policy_test")
    carry = m["obs_contract"]["eval_env"]["carry"]
    assert set(carry) | {HIDE} == set(rp.EVAL_CARRY) - set(rp.EVAL_CARRY_BODY_ONLY)
    assert carry[rp.COMMAND_FRAME_VAR] == "world"
    assert DR_VAR not in rp.EVAL_CARRY and DR_VAR not in carry and DR_VAR not in m["obs_contract"]["eval_env"]["fixed"]
    plan = _json(F19 / "plan.json")
    sel_rows = _json(RES / "sel.rows.json")
    for s in m["stages"]:
        base, ov = _overlay(s["overlay_ref"])
        assert base == s["phase"], f"{s['dir']}: chain base {base} != manifest phase {s['phase']}"
        want = rp.eval_env_for(rp.ph.PHASES[s["phase"]], m["plant"], keep=rp.CANARY_KEYS,
                               keep_source=rp.ph.FORMATION, carry_overlay=ov)
        assert tool.eval_env(m, s) == want, s["dir"]
        assert DR_VAR not in tool.eval_env(m, s) and DR_VAR not in tool.play_env(m, s), s["dir"]
        for k in s.get("overlay_delta") or {}:
            assert k not in tool.eval_env(m, s), (s["dir"], k)
        # the tracked env file the stage was evaluated with (TDR: T*'s p3b.env; the student: the sel rows' env file)
        if s["kind"] == "distill":
            envf = {Path(v["job"]["env_file"]).name for k, v in sel_rows.items()
                    if k.startswith(f"{s['distill']['selected']}:suite@")}
            assert envf == {"p3b.env"}
            assert tool.eval_env(m, s) == _env_file(F19 / "envs/p3b.env")
        elif s["dir"] == TDR:
            assert tool.eval_env(m, s) == _env_file(F19 / "envs/p3b.env")
        else:
            assert tool.eval_env(m, s) == _env_file(F19 / plan["stages"][s["stage"]]["env"]), s["dir"]


def test_eval_stamps_are_the_stage_files_under_the_manifest_env():
    """Every stage's eval stamps (untracked; this checkout or the main one, skipped when on neither) scored exactly the
    stage's file, under exactly the manifest's eval env, with the policy width 1149; the suite@1 stamp's per-category
    suite numbers are the manifest's."""
    tool = _tool()
    m = tool.load_manifest()
    rp = _load(RUN_PHASES, "omni_run_phases_policy_stamp_test")
    seen = 0
    for s in m["stages"]:
        for key in ("stamp", "es2_stamp"):
            if not s["evals"].get(key):
                continue
            st = _local(s["evals"][key])
            if st is None or not (st / "run_meta.json").is_file():
                continue
            meta = _json(st / "run_meta.json")
            assert meta["checkpoint_sha256"] == s["sha256"], (s["dir"], key)
            assert meta["env_vars"] == tool.eval_env(m, s), (s["dir"], key)
            assert meta["obs_dim_actual"] == 1149, (s["dir"], key)
            seen += 1
            if key != "stamp":
                continue
            cat = rp.summarize(_json(st / "scenario_metrics.json")["aggregate"])["cat"]
            for c, e in s["evals"]["categories"].items():
                for k, v in e.items():
                    got = cat[c][k]
                    assert (got == v) if k == "n" else abs(round(got, 3) - v) < 1e-9, (s["dir"], c, k, got, v)
    if not seen:
        pytest.skip("no eval stamp of a stage on this machine (raw stamps are untracked)")


def test_recipe_construction_is_what_every_stage_trained():
    """The manifest's runner-side construction block (and each stage's removal set) is in every RL stage's overlay,
    except the keys a stage declares in its own ``construction_delta`` -- which must be what that overlay trained and
    never a POLICY-structure knob; the F19 s1 .. p2a overlays are byte-identical to the superseded seed-18 lineage's
    (same recipe, new seed); the payload knobs appear on the payload stages only; TDR trained T*'s overlay plus exactly
    the DR knob (its recorded and tracked env); the distill stage trained under TDR's composed env plus the DR (no
    ramp) minus exactly the student-runner-refused knobs (STD_PARAM / INIT_STD kept), as F19's student did under T*'s."""
    m = _m()
    con = m["construction"]
    assert con["ppo"]["KRABBY_STD_PARAM"] == m["obs_contract"]["eval_env"]["carry"]["KRABBY_STD_PARAM"]
    assert str(con["ppo"]["KRABBY_INIT_STD"]) == str(m["obs_contract"]["eval_env"]["carry"]["KRABBY_INIT_STD"])
    deltas = {s["dir"]: s.get("construction_delta") or {} for s in m["stages"]}
    assert {d: v for d, v in deltas.items() if v} == {d: {"KRABBY_ACTION_RATE_W": "-0.05"} for d in (
        "f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR, TDR)}
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
        assert DR_VAR not in ov, s["dir"]
        if s["stage"] in old:
            _, ov_old = _overlay(old[s["stage"]]["overlay_ref"])
            assert ov == ov_old, f"{s['dir']}: the F19 overlay differs from the seed-18 lineage's"
            assert _chain_stage(s["overlay_ref"])["iters"] == _chain_stage(old[s["stage"]]["overlay_ref"])["iters"]
    # TDR: T*'s chain stage (the same overlay) + the DR knob, nothing else; recorded by its run and its env file
    tstar, tdr = _stage(m, TSTAR), _stage(m, TDR)
    assert tdr["overlay_ref"] == tstar["overlay_ref"]
    assert tdr["overlay_delta"] == {DR_VAR: "0.6:1.35:12000"} == {DR_VAR: tdr["dr"][DR_VAR]}
    _, ov = _overlay(tdr["overlay_ref"])
    composed = {"KRABBY_OMNI_PHASE": tdr["phase"], "KRABBY_PLANT": m["plant"], **ov, **tdr["overlay_delta"]}
    assert _json(PKG / tdr["run_record"])["env"]["krabby"] == composed
    assert _krabby(_env_file(PKG / tdr["env_file"])) == composed
    assert "training-only" in tdr["dr"]["what"]
    # the refused list is read from the student spec's source (as distill.sh does), not by importing the config module
    import ast
    src = (PKG / "config" / "crab_hex" / "omni_student_spec.py").read_text()
    refused = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED" for t in n.targets))
    assert DR_VAR not in refused
    # the distill stage: TDR's composed env (preset + plant + overlay) + the DR without its ramp, minus the refused
    st = next(s for s in m["stages"] if s["kind"] == "distill")
    teacher = _stage(m, st["distill"]["teacher"])
    assert st["distill"]["teacher"] == TDR and st["overlay_ref"] == teacher["overlay_ref"]
    assert st["distill"]["dr"] == {DR_VAR: "0.6:1.35"}
    denv = _env_file(PKG / st["distill"]["env_file"])
    assert denv == {"KRABBY_OMNI_PHASE": st["phase"], "KRABBY_PLANT": m["plant"], **ov, **st["distill"]["dr"]}
    assert _json(RES / "runs" / "RSD_ST_s19.distill.json")["env_file_krabby"] == _krabby(denv)
    assert st["distill"]["stripped"] == [k for k in refused if k in denv]
    assert {"KRABBY_STD_PARAM", "KRABBY_INIT_STD"} <= set(denv) - set(st["distill"]["stripped"])
    for k, v in st["removal_set"].items():
        assert denv.get(k) == str(v), k
    # the superseded F19 student: T*'s composed env, no DR, the same stripped list
    old_st = _sup(m, "F19_student")["block"]
    assert old_st["distill"]["teacher"] == TSTAR and old_st["overlay_ref"] == tstar["overlay_ref"]
    odenv = _env_file(PKG / old_st["distill"]["env_file"])
    assert odenv == {"KRABBY_OMNI_PHASE": old_st["phase"], "KRABBY_PLANT": m["plant"], **ov}
    assert old_st["distill"]["stripped"] == [k for k in refused if k in odenv] == st["distill"]["stripped"]


def test_payload_and_field_stages_with_their_evals_of_record():
    """p2a: preset 2a on the teacher task, the wall field, the clock still removed; p3a / p3b / TDR: the same + their
    payload knobs; the F19 curves and max_stable exactly as the tracked gate records say, TDR's as G1's nominal cells
    (180 kg not read); every obstacle row pooled consistently and equal to the record it was read from (F19: the gate's
    own row reads; TDR: G1 N / R and the stand re-pairing's REFS(TDR) calibration); T* = p3b by the recorded rule; every
    stage FAST / FAST and its gate record's verdict and head sha as the manifest says; the student's selection and G2."""
    m = _m()
    st = {s["dir"]: s for s in m["stages"]}
    p2a = st["f19_6_p2a_field"]
    assert (p2a["task"], p2a["play_task"]) == ("Isaac-Crab-Hex-Omni-Teacher-v0", "Isaac-Crab-Hex-Omni-Walk-Play-v0")
    assert (st[TDR]["task"], st[TDR]["play_task"]) == (p2a["task"], p2a["play_task"])
    assert p2a["overlay_ref"] == "experiments/2026-10-16_task4_pipeline/F19/candidates.yaml chains.F19_p2a stage p2a"
    for d in ("f19_6_p2a_field", "f19_7_p3a_payload90", TSTAR, TDR):
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
    assert st[TDR]["payload"]["knobs"] == st[TSTAR]["payload"]["knobs"]
    assert st[TDR]["field_env"] == st[TSTAR]["field_env"] and st[TDR]["removal_set"] == st[TSTAR]["removal_set"]
    dl = _deliverable()
    for s in m["stages"][:len(F19_STAGES)]:
        g = s.get("gate")
        rec = _json(PKG / g["record"])
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
    # TDR: the teacher gate G1 (its own record shape: S stress / N nominal / R paired 0-kg vs in-pin T*)
    tdr = st[TDR]
    g1 = _json(PKG / tdr["gate"]["record"])
    assert tdr["gate"]["record"] == "experiments/2026-11-07_rod_speed_dr/results/g1.json"
    assert g1["verdict"] == tdr["gate"]["verdict"] == "PASS" and g1["R"]["verdict"] == "PASS"
    assert g1["policies"]["tdr"]["sha256"] == tdr["sha256"] and g1["policies"]["old"]["sha256"] == st[TSTAR]["sha256"]
    assert g1["R"]["thresholds_version"] == "v1.2" and tdr["gate"]["thresholds"].startswith("v1.2 ")
    assert g1["R"]["side22"]["head"] == ["FAST", "FAST"] and tdr["evals"]["lr"]["hands"] == "FAST/FAST"
    assert "FAST / FAST" in tdr["evals"]["lr"]["source"]
    p = tdr["payload"]
    curve = {_cell_kg(c): e["status"] == "pass" and all(x["pass"] for x in e["seeds"].values())
             for c, e in g1["N"].items()}
    assert curve == {int(k): v for k, v in p["curve"].items()} == {0: True, 90: True, 135: True}
    assert 180 not in curve and p["max_stable_kg"] == f">= {max(curve)} (180 not read)"
    assert p["first_unstable_kg"] == "not read" and p["limiting_factor"].startswith("not read")
    assert {int(k): v for k, v in p["field_light_by_kg"].items()} == {
        _cell_kg(c): e["head"] for c, e in g1["N_pooled"].items()}
    assert "G1 R PASS" in p["zero_kg_check"] and p["label"] == st[TSTAR]["payload"]["label"]
    rows = tdr["evals"]["rows"]
    for name, r in rows.items():
        per = r["per_env_seed"]
        assert r["n"] == 100 * len(per) == 100 * len(r["stamps"]), name
        assert abs(sum(per) / len(per) - r["completion"]) < 1e-9, name
    rrow = {x["row"]: x for x in g1["R"]["rows"] if x["statistic"].endswith(" completion")}
    nom0 = g1["N"]["T00_nominal_L000_A"]
    assert rows["field_light"]["per_env_seed"] == _seed_values(nom0)
    assert rows["field_light"]["completion"] == g1["N_pooled"]["T00_nominal_L000_A"]["head"] == rrow["field_light"]["head"]
    assert rows["ring_obst"]["completion"] == rrow["ring_obst"]["head"]
    # the Walk-Play REFS(TDR) rows: the stand re-pairing's walker reference (walker_spec.json), read by sn_calib
    spec = _json(RES / "stand" / "walker_spec.json")
    refs = _json(RES / "stand" / "sn_calib.json")["rows"]["REFS"]["gates"]["G4"]
    assert refs["clauses"]["field_hard es1+2+3+4 completion >= REFS - max(0.05, 2 SE)"]["value"] == \
        rows["field_hard"]["completion"]
    assert refs["clauses"]["field_turn es1 completion >= REFS - max(0.05, band)"]["value"] == \
        rows["field_turn"]["completion"]
    assert refs["reported"]["z2_field_vs_hold default rows (FT es1, FH es1)"][
        "field_hard__omni@1 completion >= HOLD - 2 SE(diff)"][0] == rows["field_hard"]["per_env_seed"][0]
    for name in ("field_hard", "field_turn"):
        for k, stamp in enumerate(rows[name]["stamps"], 1):
            assert spec["refs_stamps"][f"{name}__omni@{k}"].endswith("/" + stamp), (name, k)
    assert spec["refs_stamps"]["suite__omni@1"].endswith("/" + tdr["evals"]["stamp"])
    assert spec["refs_stamps"]["suite__omni@2"].endswith("/" + tdr["evals"]["es2_stamp"])
    # the teacher-role rows (Student-Play) and in-pin T*'s, as G1's R statistic paired them
    trole = tdr["evals"]["teacher_role"]
    for name in ("field_hard", "field_turn"):
        r = trole[name]
        assert r["completion"] == rrow[name]["head"] and abs(sum(r["per_env_seed"]) / len(r["per_env_seed"])
                                                             - r["completion"]) < 1e-9, name
    assert trole["tstar_in_pin"] == {name: rrow[name]["p2a"] for name in ("field_hard", "field_light", "ring_obst",
                                                                          "field_turn")}
    # the stress block: G1 S per seed (TDR) and in-pin T* (bridge_TSTAR.json, the first read of each cell seed)
    stress = tdr["evals"]["stress"]
    assert stress["record"] == tdr["gate"]["record"]
    bridge = _json(RES / "bridge_TSTAR.json")
    assert bridge["sha256"] == st[TSTAR]["sha256"]
    for key, cell in _STRESS_CELLS.items():
        e = stress[key]
        assert e["tdr"] == _seed_values(g1["S"][cell]) and e["pass"] is (g1["S"][cell]["status"] == "pass"), key
        if "tstar" in e:
            assert e["tstar"] == [bridge["cell_seeds"][f"{cell}@es{k}"]["first"]["values"]["field_light_completion"]
                                  for k in (1, 2)], key
    # the student: the campaign's selection (the full block passes) and its student gate G2 against in-pin c28000
    stu = st[STUDENT]
    sel = _json(PKG / stu["distill"]["selection_gate"])
    assert sel["verdict"] == "SELECTED" and sel["blocks"][stu["distill"]["selected"]]["pass"] is True
    g2 = _json(RES / "g2.json")
    assert stu["distill"]["student_gate"].startswith("experiments/2026-11-07_rod_speed_dr/results/g2.json PASS")
    assert g2["verdict"] == "PASS" and g2["policies"]["pick"]["sha256"] == stu["sha256"]
    assert g2["policies"]["ref"]["sha256"] == _sup(m, "F19_student")["block"]["sha256"]
    sst = stu["evals"]["stress"]
    assert sst["record"] == "experiments/2026-11-07_rod_speed_dr/results/g2.json"
    for key in ("x0p7_L090", "x0p8_L090"):
        cell = _STRESS_CELLS[key]
        assert sst[key]["student"] == _seed_values(g2["S"][cell]), key
        assert sst[key]["pass"] is (g2["S"][cell]["status"] == "pass") is True, key
    assert sst["x0p7_L090"]["tdr"] == stress["x0p7_L090"]["tdr"]
    rows_sel = _json(RES / "sel.rows.json")
    pick = stu["distill"]["selected"]
    for name, r in stu["evals"]["rows"].items():
        per = r["per_env_seed"]
        assert r["n"] == 100 * len(per) == 100 * len(r["stamps"]), name
        assert abs(sum(per) / len(per) - r["completion"]) < 1e-9, name
        for k, stamp in enumerate(r["stamps"], 1):
            assert rows_sel[f"{pick}:{name}@{k}"]["stamp"].endswith("/" + stamp), (name, k)
    assert rows_sel[f"{pick}:suite@1"]["stamp"].endswith("/" + stu["evals"]["stamp"])
    assert rows_sel[f"{pick}:suite@2"]["stamp"].endswith("/" + stu["evals"]["es2_stamp"])
    assert rows_sel[f"{pick}:side22@1"]["stamp"].endswith("/" + stu["evals"]["lr"]["stamp"])
    assert stu["evals"]["lr"]["hands"] == "FAST/FAST"


def test_distill_stage_is_the_student_with_its_exports():
    """rsd_10_student: kind distill on the Student task, NOT bundled (its .pt is over GitHub's 100 MiB limit), cited by
    the selection record's sha256; its folder holds exactly the ONNX deploy export + io json, the parity-tested export's
    bytes (results/onnx_PICK.json); the io json is F19's byte for byte (the deploy interface is unchanged: 1149-wide,
    4-camera, clip 1.0); the paired completions are the selection gate's; the run file (when on disk) is that sha, over
    the limit, and carries TDR's weights bitwise."""
    tool = _tool()
    m = tool.load_manifest()
    s = tool.stage_by_dir(m, STUDENT)
    assert (s["kind"], s["bundled"], s["task"], s["play_task"]) == (
        "distill", False, "Isaac-Crab-Hex-Omni-Student-v0", "Isaac-Crab-Hex-Omni-Student-Play-v0")
    assert not tool.is_bundled(s) and s["file"] == "model_31000.pt"
    assert s["sha256"] == _json(RES / "sel.json")["pick"]["sha256"] == \
        "155b5981a718ec2c41460490c38e2aadb61c60c7b48dd67e3a33a45b1d5e634e"
    assert s["eval_args"] == ["--task", "Isaac-Crab-Hex-Omni-Student-Play-v0", "--policy-role", "student"]
    assert s["play_args"] == ["--enable_cameras"]
    assert "TDR" in s["play_notes"] and "c31000" in s["play_notes"] and DR_VAR in s["play_notes"]
    files = tool.stage_files(s)
    assert [f["file"] for f in files] == ["policy.onnx", "depth_latest.onnx", "omni_student_io.json"]
    assert sorted(p.name for p in (POLICY / STUDENT).iterdir()) == sorted(f["file"] for f in files)
    onnx = _json(RES / "onnx_PICK.json")
    assert onnx["parity_ok"] is True and onnx["checkpoint_sha256"] == s["sha256"]
    want = {"policy.onnx": onnx["files"]["policy_onnx"]["sha256"],
            "depth_latest.onnx": onnx["files"]["depth_onnx"]["sha256"],
            "omni_student_io.json": onnx["files"]["io"]["sha256"]}
    run_dir = str(Path(s["source"]).parent)
    for f in files:
        assert tool.sha256_of(f["path"]) == f["sha256"] == want[f["file"]], f["file"]
        assert f["source"] == f"{run_dir}/exported_deploy/{f['file']}", f["file"]
        assert f["path"].stat().st_size <= tool.MAX_FILE_BYTES
    for e in s["exports"][:2]:
        assert "results/onnx_PICK.json" in e["what"], e["file"]
    # the deploy interface is F19's: the io json is byte-identical to the kept F19 student's
    io_path = POLICY / STUDENT / "omni_student_io.json"
    assert io_path.read_bytes() == (POLICY / OLD_STUDENT / "omni_student_io.json").read_bytes()
    io = _json(io_path)
    assert io["kind"] == "omni_depth_student" and io["obs"] == [1, m["obs_contract"]["width"]]
    assert len(io["camera_groups"]) == 4 and io["actions_clip"] == 1.0 and io["trained_on_zero_depth"] is False
    # the selection: the gate's paired completions (student vs TDR in the teacher role) and rank
    sel = _json(RES / "sel.json")["blocks"]["c31000"]
    got = {x["id"].rsplit(".", 1)[1]: x for x in sel["decisive"] if x["id"].startswith("student.capability.completion.")}
    assert set(got) == set(s["evals"]["paired_completion"]) == {"field_hard", "field_light", "ring_obst", "field_turn"}
    for row, e in s["evals"]["paired_completion"].items():
        assert abs(got[row]["value"] - e["student"]) < 1e-9 and abs(got[row]["student"] - e["student"]) < 1e-9, row
        assert abs(got[row]["tstar"] - e["tdr"]) < 1e-9, row       # sel.json's key "tstar" = the teacher = TDR here
        assert abs(got[row]["line"] - e["line"]) < 5e-4, row
        assert got[row]["pass"] is e["pass"] is True, row
        assert abs(s["evals"]["rows"][row]["completion"] - e["student"]) < 1e-9, row
    assert s["distill"]["selected"] == "c31000" and s["distill"]["prescreen_rank"][0] == 31000
    assert "UNVERIFIED" in s["distill"]["camera_use"] and "not re-tested" in s["distill"]["camera_use"]
    src = _local(s["source"])
    if src is None:
        pytest.skip(f"student run file not on this machine (raw run dirs are untracked): {s['source']}")
    assert src.stat().st_size > tool.MAX_FILE_BYTES and tool.sha256_of(src) == s["sha256"]
    torch = pytest.importorskip("torch")
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    tdr = torch.load(str(tool.stage_path(tool.stage_by_dir(m, TDR))), map_location="cpu", weights_only=False)
    a, b = ck["model_state_dict"], tdr["model_state_dict"]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), "the student's teacher copy is TDR's"
    assert ck["iter"] == 31000 and {"depth_actor_state_dict", "depth_encoder_state_dict"} <= set(ck)


def test_superseded_f19_student_is_kept_with_its_exports():
    """The superseded F19 student (superseded[F19_student]): its block is the F19-era distill stage (c28000, cited by the
    deliverable's sha256, distilled from T*); its three kept files are that block's exports, byte-identical to Task 4's
    tracked keep-set copies (bundle.yaml extra heads); the io json is the 1149-wide, 4-camera, clip-1.0 contract; the
    selection record is F19's gate; the run file (when on disk) is that sha and carries T*'s weights."""
    tool = _tool()
    m = tool.load_manifest()
    e = _sup(m, "F19_student")
    s = e["block"]
    dl = _deliverable()
    assert (e["kind"], s["dir"], s["kind"], s["bundled"], s["stage"]) == ("distill", OLD_STUDENT, "distill", False, "st")
    assert s["file"] == "model_28000.pt" and s["distill"]["teacher"] == TSTAR
    assert s["sha256"] == dl["student"]["sha256"] == "4fa81f0f308a4a7ac73ba8db106ab9a9c286586d05ec7a18908d776b33db9059"
    assert dl["student"]["tstar"] == "p3b" and tool.stage_by_dir(m, TSTAR)["sha256"] in e["teacher_was"]
    assert [(f["dir"], f["file"], f["sha256"], f["source"]) for f in e["files"]] == [
        (OLD_STUDENT, x["file"], x["sha256"], x["source"]) for x in s["exports"]]
    assert sorted(p.name for p in (POLICY / OLD_STUDENT).iterdir()) == sorted(f["file"] for f in e["files"])
    bundle = yaml.safe_load((T4 / "bundle.yaml").read_text())
    extra = {Path(x["source"]).name: x for x in bundle["head"]["extra_heads"] if x["source"].endswith(
        tuple(f["file"] for f in e["files"]))}
    for f in e["files"]:
        p = POLICY / f["dir"] / f["file"]
        assert tool.sha256_of(p) == f["sha256"] == extra[f["file"]]["sha256"], f["file"]
        assert f["source"] == extra[f["file"]]["source"], f["file"]
        keep = T4 / extra[f["file"]]["dest"]
        assert keep.read_bytes() == p.read_bytes(), f"{f['file']} != the campaign keep-set copy {keep}"
    io = _json(POLICY / OLD_STUDENT / "omni_student_io.json")
    assert io["kind"] == "omni_depth_student" and io["obs"] == [1, m["obs_contract"]["width"]]
    assert len(io["camera_groups"]) == 4 and io["actions_clip"] == 1.0 and io["trained_on_zero_depth"] is False
    # the F19 selection: the gate's paired completions and rank, the candidate the deliverable names
    gsel = _json(F19 / "gates/sel.json")
    assert gsel["verdict"] == "PASS" and gsel["blocks"]["c28000"]["pass"] is True
    got = {x["id"].rsplit(".", 1)[1]: x for x in gsel["blocks"]["c28000"]["decisive"]
           if x["id"].startswith("student.capability.completion.")}
    for row, pc in s["evals"]["paired_completion"].items():
        assert abs(got[row]["value"] - pc["student"]) < 1e-9 and abs(got[row]["line"] - pc["line"]) < 5e-4, row
        assert got[row]["pass"] is pc["pass"] is True, row
        assert abs(s["evals"]["rows"][row]["completion"] - pc["student"]) < 1e-9, row
    assert s["distill"]["selected"] == "c28000" and s["distill"]["prescreen_rank"][0] == 28000
    assert "UNVERIFIED" in s["distill"]["camera_use"]
    sel = _json(F19 / "stage_reads/sel.json")
    sread = _json(F19 / "stage_reads/st.json")
    assert (sel["sha256"], _rel_logs(sel["ckpt"]), sel["tstar"]) == (s["sha256"], s["source"], "p3b")
    assert _rel_logs(sread["run_dir"]) == str(Path(s["source"]).parent) and sread["tstar"] == "p3b"
    assert [c["iter"] for c in sread["candidates"]] == s["distill"]["candidates"]
    assert _rel_logs(dl["tstar"]["ckpt"]) == tool.stage_by_dir(m, TSTAR)["source"]
    src = _local(s["source"])
    if src is None:
        pytest.skip(f"F19 student run file not on this machine (raw run dirs are untracked): {s['source']}")
    assert src.stat().st_size > tool.MAX_FILE_BYTES and tool.sha256_of(src) == s["sha256"]
    torch = pytest.importorskip("torch")
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    tstar = torch.load(str(tool.stage_path(tool.stage_by_dir(m, TSTAR))), map_location="cpu", weights_only=False)
    a, b = ck["model_state_dict"], tstar["model_state_dict"]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), "c28000's teacher copy is T*'s"
    assert ck["iter"] == 28000


def test_bundle_policy_check_is_clean():
    out = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"1 cited, not bundled: {STUDENT}" in out.stdout and "+ 13 kept superseded files" in out.stdout
    assert f"stand network {STAND_DIR}" in out.stdout and f"current {TDR}: OK" in out.stdout


# ------------------------------------------------------------------------------ superseded heads (kept / left)
def test_superseded_heads_are_recorded_and_recoverable():
    """W left policy/ at the 2026-10-06 bake and is recoverable, byte-exact, from the commit the manifest names. Nothing
    left at the 2026-10-24 or the 2026-11-09 bake: every kept file (the seed-18 lineage, SH_T17, A-mask s17, F19's
    student exports, SH_T4F19, SN_T4F19) is present with its recorded sha, byte-identical to the file it was baked /
    adopted with, and its record is the manifest at record_commit verbatim (F19: the manifest before this bake)."""
    m = _m()
    tool = _tool()
    sups = m.get("superseded") or []
    assert [s["id"] for s in sups] == ["W", "SN_Amask_s17", "S1_s18", "SH_T17", "F19", "F19_student", "SH_T4F19",
                                       "SN_T4F19"]
    w = sups[0]
    commit = w["last_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", str(commit)), f"superseded W: last_commit {commit!r}"
    assert "files" not in w
    s18 = _sup(m, "S1_s18")
    f19 = _sup(m, "F19")
    baked_at = {d: ("80e839e3a987a912222fa216a31cd192a43cf014" if d != "6_p2a_field"
                    else "dd972b6a2fc919f1ab7426fcea41a6e962577761") for d in S18_STAGES}
    # the byte-exact history reads need the commits; a shallow clone runs every other check, then skips naming them
    no_hist = _missing_commits(commit, *baked_at.values(), s18["record_commit"],
                               *(_sup(m, sid)["adopted_commit"] for sid in ("SN_Amask_s17", "SH_T17")),
                               F19_RECORD_COMMIT, F19_BAKE_COMMIT)
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
    # the 2026-11-09 entries: F19 (the superseded current, no files) and its student / stand nets (files kept)
    assert (f19["kind"], f19["current_was"], f19["record_commit"], f19["adopted_commit"]) == (
        "current", TSTAR, F19_RECORD_COMMIT, F19_BAKE_COMMIT)
    assert "files" not in f19 and "2026-11-09" in f19["superseded"] and "Bake as proposed" in f19["superseded"]
    assert f19["heads_were"] == {"phase1": "f19_5_c6x_clockfree", "phase2": "f19_6_p2a_field", "payload": TSTAR,
                                 "student": OLD_STUDENT, "stand": OLD_STAND}
    assert m["baked"].rstrip().endswith("PREVIOUSLY: F19 (2026-10-24) -- superseded: F19")
    kept_new = {"F19_student": [(OLD_STUDENT, f) for f in ("policy.onnx", "depth_latest.onnx", "omni_student_io.json")],
                "SH_T4F19": [(OLD_STAND, "model_5000.pt")], "SN_T4F19": [(OLD_STAND, "model_4000.pt")]}
    for sid, want in kept_new.items():
        e = _sup(m, sid)
        assert e["adopted_commit"] == F19_BAKE_COMMIT and F19_BAKE_COMMIT[:8] in e["adopted"], sid
        assert "2026-11-09" in e["superseded"] and "STAY" in e["superseded"], sid
        assert [(f["dir"], f["file"]) for f in e["files"]] == want, sid
        assert not {"last_commit", "path_in_history"} & set(e), sid
        for f in e["files"]:
            p = POLICY / f["dir"] / f["file"]
            assert p.is_file() and tool.sha256_of(p) == f["sha256"], p
            if hist:
                assert hashlib.sha256(_git_blob(F19_BAKE_COMMIT, f"{rel_pol}/{f['dir']}/{f['file']}")).hexdigest() \
                    == f["sha256"], p
    assert _sup(m, "SH_T4F19")["kind"] == _sup(m, "SN_T4F19")["kind"] == "stand_network"
    assert _sup(m, "SH_T4F19")["block"]["warm_start"]["sha256"] == _sup(m, "SN_T4F19")["files"][0]["sha256"]
    assert tool.stage_by_dir(m, TSTAR)["sha256"] in _sup(m, "SH_T4F19")["walker_was"]
    if not hist:
        pytest.skip(f"git history not in this clone (shallow checkout): commits {no_hist} -- the byte-exact blob and "
                    "record_commit manifest checks need them")
    # the kept records are the manifest as it stood at record_commit, verbatim
    old = yaml.safe_load(_git_blob(s18["record_commit"], f"{rel_pol}/manifest.yaml"))
    assert old["current"] == s18["current_was"] == "6_p2a_field" and old["heads"] == s18["heads_were"]
    assert s18["stages"] == old["stages"] and [s["dir"] for s in old["stages"]] == S18_STAGES
    assert (s18["baked"], s18["recipe"], s18["construction"]) == (old["baked"], old["recipe"], old["construction"])
    assert sh["block"] == old["stand_network"], "the kept SH_T17 block must be the record_commit block verbatim"
    # F19's: the manifest before this bake (64f397e5): its current / heads / baked, its student stage, its stand block
    pre = yaml.safe_load(_git_blob(F19_RECORD_COMMIT, f"{rel_pol}/manifest.yaml"))
    assert pre["current"] == f19["current_was"] and pre["heads"] == f19["heads_were"] and pre["baked"] == f19["baked"]
    assert pre["stages"][-1]["dir"] == OLD_STUDENT and _sup(m, "F19_student")["block"] == pre["stages"][-1]
    assert _sup(m, "SH_T4F19")["block"] == pre["stand_network"], "the kept SH_T4F19 block must be verbatim"
    assert m["superseded"][:4] == pre["superseded"], "the four earlier superseded entries must stay verbatim"
    # the F19 stages stay live, unchanged but for T*'s `what` / `notes` (it no longer ships)
    assert [s["dir"] for s in pre["stages"][:8]] == F19_STAGES
    for a, b in zip(m["stages"][:8], pre["stages"][:8]):
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert diff == (set() if a["dir"] != TSTAR else {"what", "notes"}), (a["dir"], diff)
    assert m["obs_contract"]["eval_env"] == pre["obs_contract"]["eval_env"]
    assert m["obs_contract"]["width"] == pre["obs_contract"]["width"]
    assert {k for k in set(m["construction"]) | set(pre["construction"])
            if m["construction"].get(k) != pre["construction"].get(k)} <= {"runner_fixed"}
    assert m["assets"][0]["sha256"] == pre["assets"][0]["sha256"]


def test_kept_superseded_paths_stay_valid_for_their_users():
    """The fixed paths other code names keep resolving to the same bytes: sh_core's warm-start pin (A-mask s17 at
    stand_net/model_4000.pt), the Task 4 pipeline's bundled_s1 (6_p2a_field) and its lock, p2_read's lookup of the
    seed-18 c6x categories by sha (now under superseded S1_s18), the dose smokes' superseded-W resolution, the DR
    sweep's ONNX parity default (the kept f19_9_student) and the rod-speed DR campaign's T* (the live f19_8 file)."""
    tool = _tool()
    m = tool.load_manifest()
    core = _load_nobc(SH_CORE, "sh_core_policy_kept_test")
    assert Path(core.STAND_NET_CKPT).resolve() == AMASK_KEPT.resolve()
    assert core.STAND_NET_SHA256 == tool.sha256_of(AMASK_KEPT) == _sup(m, "SN_Amask_s17")["files"][0]["sha256"]
    spec = yaml.safe_load((T4 / "pipeline.yaml").read_text())
    s1 = spec["bundled_s1"]
    assert tool.sha256_of(REPO / s1["ckpt"]) == s1["sha256"] == _sup(m, "S1_s18")["files"][-1]["sha256"]
    lock = _json(T4 / "pipeline.lock.json")
    locked = json.dumps(lock)
    for rel in ("6_p2a_field/model_19994.pt", "stand_net/model_4000.pt", "stand_net/model_5000.pt"):
        assert f"crab_hex_omni_task/policy/{rel}" in locked and (POLICY / rel).is_file(), rel
    p2r = _load_nobc(PKG / "experiments" / "2026-10-07_omni_phase2" / "scripts" / "p2_read.py", "p2_read_policy_kept")
    c6x = next(s for s in _sup(m, "S1_s18")["stages"] if s["dir"] == "5_c6x_clockfree")
    assert p2r._manifest_s18() == c6x["evals"]["categories"]
    w = _sup(m, "W")
    assert [st["dir"] for st in w["stages"]][-1] == "5_c6x_clockfree" and w["stages"][-1]["file"] == "model_11995.pt"
    # read as source text (both modules import torch / Task 4 tools at import time)
    parity = (PKG / "experiments" / "2026-11-06_dr_stress_sweep" / "scripts" / "onnx_parity.py").read_text()
    mo = re.search(r'^ONNX_DIR = R / "([^"]+)"$', parity, re.M)
    assert mo and (REPO / mo.group(1)).resolve() == (POLICY / OLD_STUDENT).resolve(), mo
    assert {f["file"] for f in _sup(m, "F19_student")["files"]} <= {p.name for p in (REPO / mo.group(1)).iterdir()}
    lib = (RSD / "scripts" / "rsd_lib.py").read_text()
    mo = re.search(r'^TSTAR_CKPT = OMNI / "([^"]+)"$', lib, re.M)
    assert mo and (PKG / mo.group(1)).resolve() == tool.stage_path(tool.stage_by_dir(m, TSTAR)).resolve(), mo
    pre = re.search(r'^TSTAR_SHA_PREFIX = "([0-9a-f]{8})"$', lib, re.M)
    assert pre and tool.stage_by_dir(m, TSTAR)["sha256"].startswith(pre.group(1))


def test_rsd_lib_manifest_student_sha_falls_back_to_the_superseded_f19_student_block():
    """Since this bake c28000 is no stage of the manifest: rsd_lib.manifest_student_sha reads its sha from the
    superseded F19_student block (manifest only, no run file, so this runs on CI too). rsd_lib is loaded as a private
    instance and the process state it touches (sys.dont_write_bytecode, the shared 'tp_lib' module) is restored."""
    pytest.importorskip("numpy")
    m = _m()
    assert not [s for s in m["stages"] if s["dir"] == OLD_STUDENT]
    blk = _sup(m, "F19_student")["block"]
    saved = (sys.dont_write_bytecode, sys.modules.get("tp_lib"))
    try:
        lib = _load_nobc(RSD / "scripts" / "rsd_lib.py", "rsd_lib_policy_fallback_test")
        assert lib.POLICY_MANIFEST.resolve() == (POLICY / "manifest.yaml").resolve()
        assert blk["dir"] == OLD_STUDENT and blk["source"] == lib.C28000_CKPT.relative_to(lib.MAIN).as_posix()
        assert lib.manifest_student_sha() == blk["sha256"] == \
            "4fa81f0f308a4a7ac73ba8db106ab9a9c286586d05ec7a18908d776b33db9059"
    finally:
        sys.dont_write_bytecode = saved[0]
        if saved[1] is None:
            sys.modules.pop("tp_lib", None)
        else:
            sys.modules["tp_lib"] = saved[1]


def test_superseded_sh_t17_block_still_matches_its_campaign_reads():
    """The kept SH_T17 record (superseded[SH_T17].block): its K1-K6 table is still the stand-height campaign's reads
    (head_SH_T{18,16,17}.json, step4b_read.json), its checkpoint (the kept stand_net/model_5000.pt) is the height net
    trained beside the kept S1 s18 (6_p2a_field) and warm-started from the kept A-mask s17."""
    m = _m()
    blk = _sup(m, "SH_T17")["block"]
    gt = blk["evals"]["gate_table"]
    step4b = _json(SH_CAMPAIGN / "step4b_read.json")
    heads = {h: _json(SH_CAMPAIGN / f"head_{h}.json") for h in SH_HEADS}
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


def _sn_family_counts(g: dict) -> dict:
    counts = {}
    for x in g["decisive"]:
        fam = x["id"].split(".")[2]
        counts.setdefault(fam, [0, 0])
        counts[fam][0] += bool(x["pass"])
        counts[fam][1] += 1
    return counts


def test_stand_network_file_is_bundled_with_the_manifest_sha():
    """The adopted SH_RSD_TDR file sits in its own folder (not a stage / asset / kept dir) with the manifest sha (the
    campaign's head record and gate name the same), beside the walking head of record TDR it was trained and gated
    with; the folder holds exactly it and its warm-start ancestor."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    assert (sn["dir"], sn["file"]) == (STAND_DIR, "model_5000.pt")
    assert (sn["id"], sn["arm"], sn["stand_mask"], sn["training_seed"], sn["iterations"]) == (
        "SH_RSD_TDR_sh_s19", "A-mask", 1, 19, 5000)
    kept_dirs = {f["dir"] for e in m["superseded"] for f in e.get("files") or []}
    assert sn["dir"] not in {s["dir"] for s in m["stages"]} | {a["dir"] for a in m["assets"]} | kept_dirs
    p = tool.stand_path(sn)
    assert p == POLICY / STAND_DIR / "model_5000.pt" and p.is_file(), p
    head = _json(RES / "stand" / "sh_head.json")
    assert head["name"] == sn["id"] and head["stage"] == "sh"
    assert tool.sha256_of(p) == sn["sha256"] == head["sha256"] == _json(RES / "stand/gates/sh.json")["head"]["sha256"]
    assert head["ckpt"].endswith("/" + sn["source"]) and head["status"]["iterations_run"] == sn["iterations"]
    assert sorted(q.name for q in p.parent.iterdir()) == ["model_4000.pt", "model_5000.pt"]
    assert [f["path"] for f in tool.stand_files(sn)] == [p, tool.warm_start_path(sn)] == [p, p.parent / "model_4000.pt"]
    assert m["heads"]["stand"] == sn["dir"]
    # the walking head of record, baked at the same time, and the block pins exactly that walker
    assert m["current"] == sn["walker"]["stage"] == TDR
    assert sn["walker"]["sha256"] == tool.stage_by_dir(m, TDR)["sha256"] == \
        _json(RES / "policies.json")["policies"]["TDR"]["sha256"]
    assert '"Bake as proposed"' in sn["adopted"] and "2026-11-09" in sn["adopted"]
    assert '"Accept and disclose"' in sn["adopted"] and "results/stand/gates/sh_decision.json" in sn["adopted"]
    assert sn["source"] == "parkour/outputs/omni_plan/stand_height/runs/SH_RSD_TDR_sh_s19/model_5000.pt"
    # the trainer flags the head record ran with (+ the warm start it names)
    args = head["args"]
    assert args[:len(sn["training"]["flags"].split())] == sn["training"]["flags"].split()
    assert args[args.index("--warm-sha") + 1] == sn["warm_start"]["sha256"]
    assert args[args.index("--warm-start") + 1].endswith("/" + sn["warm_start"]["source"])


def test_superseded_sh_t4f19_is_kept_with_its_ancestor():
    """The superseded F19 stand nets (superseded[SH_T4F19] / [SN_T4F19]): their kept files in f19_stand_net/ are the
    deliverable's stand heads, byte-identical to Task 4's tracked keep-set copies; SH_T4F19's block pins T* (still a
    live stage) as its walker and SN_T4F19 as its warm start, whose gate passed (F19/gates/sn.json) with the per-family
    counts its block states; its walker override is T*'s; the checkpoint (when torch is there) was trained beside T*."""
    tool = _tool()
    m = tool.load_manifest()
    sh, sn_old = _sup(m, "SH_T4F19"), _sup(m, "SN_T4F19")
    blk = sh["block"]
    dl = _deliverable()
    assert (blk["dir"], blk["file"], blk["id"]) == (OLD_STAND, "model_5000.pt", "SH_T4F19_sh_s19")
    p = POLICY / OLD_STAND / "model_5000.pt"
    assert tool.sha256_of(p) == sh["files"][0]["sha256"] == blk["sha256"] == dl["stand"]["sh"]["sha256"]
    assert sh["files"][0]["source"] == blk["source"] == "parkour/outputs/omni_plan/stand_height/runs/SH_T4F19_sh_s19/model_5000.pt"
    assert (T4 / "head" / "model_5000.pt").read_bytes() == p.read_bytes()
    assert sorted(q.name for q in p.parent.iterdir()) == ["model_4000.pt", "model_5000.pt"]
    tstar = tool.stage_by_dir(m, TSTAR)
    assert blk["walker"]["stage"] == TSTAR and blk["walker"]["sha256"] == tstar["sha256"] == dl["tstar"]["sha256"]
    ws = blk["warm_start"]
    q = POLICY / OLD_STAND / ws["file"]
    assert (ws["id"], ws["file"]) == ("SN_T4F19_sn_s19", "model_4000.pt")
    assert [(f["dir"], f["file"], f["sha256"], f["source"]) for f in sn_old["files"]] == [
        (OLD_STAND, ws["file"], ws["sha256"], ws["source"])]
    assert ws["sha256"] == "96832099e58b75d1cc86f6f9d98327c56b827c2fac1b6c61c96b07f0a8e97b73" == dl["stand"]["sn"]["sha256"]
    assert tool.sha256_of(q) == ws["sha256"] and (T4 / "head" / "model_4000.pt").read_bytes() == q.read_bytes()
    g = _json(F19 / "gates/sn.json")
    assert g["verdict"] == "PASS" and g["head"]["sha256"] == ws["sha256"]
    counts = _sn_family_counts(g)
    assert ", ".join(f"{a} / {b}" for a, b in (counts[k] for k in ("G1", "G2", "G3", "G4", "G5"))) in ws["used_by"]
    env = blk["evaluate"]["env"]
    assert env["SN_WALKER_CK"] == f"{REPO_TOKEN}/{tstar['source']}"
    assert _names_recorded(env["SN_WALKER_ST"], dl["tstar"]["driver_stamp"])
    assert env["SN_WALKER_SPEC"]["REFS_NAME"] == "REFS_T4F19" and env["SN_WALKER_SPEC"]["S1_SHA256"] == tstar["sha256"]
    torch = pytest.importorskip("torch")
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    info = ck["infos"]["sh_dagger"]
    assert info["s1_sha256"] == tstar["sha256"] and info["warm_start"]["sha256"] == ws["sha256"]
    assert (info["seed"], info["stand_mask"], info["residual_deadband"]) == (19, 1, 0.25)


def test_expected_files_include_the_stand_network():
    tool = _tool()
    m = tool.load_manifest()
    files = tool.expected_files(m)
    # the stand network of record AND its bundled warm-start ancestor; every superseded stand file stays expected
    new = {f"{STAND_DIR}/model_5000.pt", f"{STAND_DIR}/model_4000.pt"}
    kept = {"stand_net/model_5000.pt", "stand_net/model_4000.pt"}
    kept_f19 = {*(f"{OLD_STUDENT}/{f}" for f in ("policy.onnx", "depth_latest.onnx", "omni_student_io.json")),
                f"{OLD_STAND}/model_5000.pt", f"{OLD_STAND}/model_4000.pt"}
    assert new | kept | kept_f19 <= files
    no_ws = copy.deepcopy(m)
    del no_ws["stand_network"]["warm_start"]
    assert tool.expected_files(no_ws) == files - {f"{STAND_DIR}/model_4000.pt"}
    bare = copy.deepcopy(m)
    del bare["stand_network"]
    assert tool.expected_files(bare) == files - new
    no_sup = copy.deepcopy(m)
    no_sup["superseded"] = [e for e in no_sup["superseded"] if e["id"] == "W"]
    assert tool.expected_files(no_sup) == files - kept - kept_f19 - {f"{d}/{s['file']}" for d, s in zip(
        S18_STAGES, _sup(m, "S1_s18")["stages"])}


def test_stand_network_source_is_the_bundled_file():
    """The source run files (untracked; this checkout or the main one) are byte-identical to the bundled copies."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    for f in tool.stand_files(sn):
        src = _local(f["source"])
        if src is None:
            pytest.skip(f"stand-network source run file not on this machine (raw run dirs are untracked): {f['source']}")
        assert tool.sha256_of(src) == f["sha256"] == tool.sha256_of(f["path"])


def test_stand_warm_start_is_the_ancestor_sh_rsd_tdr_records():
    """The bundled warm start (``stand_network.warm_start``, same folder) is the file SH_RSD_TDR was warm-started from:
    its sha256 equals SH_RSD_TDR's checkpoint ``infos.sh_dagger.warm_start`` / ``warm_start_cli`` (which name its source
    run) and the campaign's SN head record; it is itself the stand net trained beside TDR, whose SN gate HALTed on the
    four G5 body-height lines only -- all four re-anchored on the R / HOLD controls under the user's 11:53 rule
    (gates/sn_reanchor.json) -- with the per-family line counts the block states, and whose estimator is TDR's;
    sh_core's own pinned default warm start stays the kept A-mask s17 file."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    ws = sn["warm_start"]
    head = _json(RES / "stand" / "sn_head.json")
    assert (ws["id"], ws["file"]) == ("SN_RSD_TDR_sn_s19", "model_4000.pt") and head["name"] == ws["id"]
    assert ws["sha256"] == "6cd82ff539daa3e78efa8de94fcb9fb21c8ba2f43715242a7c2e08bbb2ce534d" == head["sha256"]
    assert ws["source"] == "parkour/outputs/omni_plan/stand_network/runs/SN_RSD_TDR_sn_s19/model_4000.pt"
    assert head["ckpt"].endswith("/" + ws["source"]) and head["args"] == ["--mode", "dagger", "--arm", "mask", "--seed",
                                                                          "19", "--iterations", "4000"]
    assert "NOT the stand network of record" in ws["role"] and "W[:, 15] zeroed" in ws["role"]
    p = tool.warm_start_path(sn)
    assert p == POLICY / sn["dir"] / ws["file"] and p.is_file(), p
    assert tool.sha256_of(p) == ws["sha256"]
    assert ws["sha256"] in sn["training"]["warm_start"]
    # the SN stage's gate: HALT on exactly the four G5 body-height lines, each re-anchored; the counts the block states
    g = _json(RES / "stand" / "gates" / "sn.json")
    ra = _json(RES / "stand" / "gates" / "sn_reanchor.json")
    assert g["verdict"] == "HALT" and g["thresholds"]["version"] == "v1.2" and g["head"]["sha256"] == ws["sha256"]
    failed = [x["id"] for x in g["decisive"] if not x["pass"]]
    assert len(failed) == 4 and all(".G5." in x and "body height" in x for x in failed), failed
    assert failed == g["halt_lines"] == ra["halt_lines"]
    assert [x["id"] for x in ra["body_height"]["lines"]] == failed
    assert all(x["reanchored"] and all(c["control_fails_same_line"] and c["head_within_tol"]
                                       for c in x["controls"].values()) for x in ra["body_height"]["lines"])
    assert ra["reanchored"] is True and ra["remaining"] == [] and set(ra["body_height"]["lines"][0]["controls"]) == {
        "R", "HOLD"}
    assert "11:53" in ra["body_height"]["rule"] and "11:53" in ws["used_by"]
    counts = _sn_family_counts(g)
    assert sorted(counts) == ["G1", "G2", "G3", "G4", "G5"] and counts["G5"][1] - counts["G5"][0] == 4
    assert ", ".join(f"{a} / {b}" for a, b in (counts[k] for k in ("G1", "G2", "G3", "G4", "G5"))) in ws["used_by"]
    assert f"{sum(a for a, _ in counts.values())} / {len(g['decisive'])} decisive pass" in ws["used_by"]
    assert "results/stand/gates/sn.json" in ws["used_by"] and "results/stand/gates/sn_reanchor.json" in ws["used_by"]
    # sh_core's pinned default is the kept A-mask s17 (the trainer warm-started SH_RSD_TDR from SN_RSD_TDR by --warm-start)
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
    t = torch.load(str(tool.stage_path(tool.stage_by_dir(m, TDR))), map_location="cpu", weights_only=False)
    assert all(torch.equal(wck["estimator_state_dict"][k], t["estimator_state_dict"][k]) for k in t["estimator_state_dict"])


def test_stand_network_checkpoint_is_the_height_net_beside_the_walker():
    """infos.sh_dagger (and the predecessor's infos.sn_dagger block): mask 1 as the manifest says, trained beside the
    walking stage the block names (s1_sha256 == TDR's sha), with R's validated constants, the trainer flags of record
    (deadband 0.25 in the loop, the trim branch at the saturated label), warm-started from the bundled SN_RSD_TDR, under
    the height contract and clip the manifest's deployment.constants state; only the actor backbone differs from TDR's
    checkpoint and the estimator is TDR's, bitwise."""
    torch = pytest.importorskip("torch")
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    w = tool.stage_by_dir(m, sn["walker"]["stage"])
    assert w["dir"] == TDR
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
    the launcher, the table test and the recorder import) and identical to the superseded SH_T4F19 and SH_T17
    contracts' (the same recipe on a new walker); the contract text differs from SH_T4F19's only in the walker-specific
    keys; the posture endpoints the contract states are sh_core's q map, and the action path behaves as stated: hard
    0.25 deadband on the clamped hip / knee residual only, before the posture; column 15 = 0 at 1.064."""
    torch = pytest.importorskip("torch")
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    dep = sn["deployment"]
    c = dep["constants"]
    old = _sup(m, "SH_T4F19")["block"]["deployment"]
    assert c == old["constants"] == _sup(m, "SH_T17")["block"]["deployment"]["constants"]
    changed = {k for k in set(dep) | set(old) if dep.get(k) != old.get(k)}
    assert changed == {"zero_height", "scope", "caveat_switch_keying", "caveat_walker_height"}, changed
    assert dep["env_required"] == old["env_required"] and dep["env_off"] == old["env_off"]
    assert "SH_RSD_TDR" in dep["scope"] and "SH_T4F19 (es1" not in dep["scope"] and "5 / 6" in dep["scope"]
    assert "SH_RSD_TDR_sh_s19z" in dep["zero_height"] and "43 K5 lines" in dep["zero_height"]
    assert "REFS_RSD_TDR" in dep["zero_height"] and "0.92 / 0.91" in dep["zero_height"]
    assert "SH_T4F19" in dep["caveat_switch_keying"] and "caveat_walker_height" not in old
    assert "0.03 m taller" in dep["caveat_walker_height"] and "1.064" in dep["caveat_walker_height"]
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
              "zero_height", "scope", "reference_implementation", "caveat_switch_keying", "caveat_height_source",
              "caveat_walker_height"):
        assert dep.get(k), k


def test_stand_network_routing_env_is_the_walkers():
    """Both networks run in one process under TDR's carry env (= T*'s): the block's env_required is exactly the walker's
    carry values, its env_off knobs are off there, the launcher's D5 rule (sn_common.env_reasons) accepts that env, which
    is T*'s tracked F19 eval env (and, when on disk, the env of every SH_RSD_TDR height stamp); the walker override the
    README exports is the campaign's results/stand/walker_spec.json spec (REFS_ENV_STAMP included) and parses under
    sn_dagger_spec; every script the commands name exists; a re-run uses fresh names, the trained deadband, mask 1 and
    the bundled file."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    w = tool.stage_by_dir(m, sn["walker"]["stage"])
    assert w["dir"] == TDR
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
    # the walker override: exactly what the stand re-pairing composed beside TDR (results/stand/walker_spec.json)
    run = sn["evaluate"]
    env = run["env"]
    rec = _json(RES / "stand" / "walker_spec.json")["spec"]
    cand = yaml.safe_load((F19 / "candidates.yaml").read_text())
    spec = env["SN_WALKER_SPEC"]
    # checkout-independent as committed: every path is $REPO/<repo-relative path>, none absolute
    leaves = [env["SN_WALKER_CK"], env["SN_WALKER_ST"], *(v for v in spec.values() if isinstance(v, str))]
    assert not [v for v in leaves if v.startswith("/")], leaves
    assert (env["SN_WALKER_CK"], str(env["P2_REANCHOR"])) == (f"{REPO_TOKEN}/{w['source']}", "1")
    assert _names_recorded(env["SN_WALKER_ST"], rec["REFS_ENV_STAMP"])
    assert spec["REFS_ENV_STAMP"] == env["SN_WALKER_ST"]
    assert set(spec) == set(rec)
    for k, v in spec.items():
        if isinstance(v, str) and v.startswith(REPO_TOKEN + "/"):
            assert _names_recorded(v, rec[k]), (k, v, rec[k])
        else:
            assert v == rec[k], k
    assert env["SN_WALKER_ST"] == f"{REPO_TOKEN}/{w['evals']['stamp']}"
    here = _at_checkout(spec)
    assert here == {"CANDIDATES": str(F19 / "candidates.yaml"), "OVERLAY_CHAIN": "F19_p2a",
                    "REFS_ENV_STAMP": _at_checkout(env["SN_WALKER_ST"]), "REFS_NAME": "REFS_RSD_TDR",
                    "S1_AGENT_YAML": str(REPO / Path(w["source"]).parent / "params" / "agent.yaml"),
                    "S1_CKPT": str(REPO / w["source"]), "S1_SHA256": w["sha256"],
                    "S1_OVERLAY": {k: str(v) for k, v in cand["chains"]["F19_p2a"]["stages"][0]["overlay"].items()}}
    spec_mod = _load_nobc(SN_SPEC, "sn_dagger_spec_policy_test")
    parsed = spec_mod.walker_spec_override({"SN_WALKER_SPEC": json.dumps(here, sort_keys=True)})
    assert parsed["S1_SHA256"] == w["sha256"] and parsed["REFS_NAME"] == "REFS_RSD_TDR"
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
    assert (RES / "stand" / "sh_lazy_rh_read_spec.json").is_file()
    assert "experiments/2026-11-07_rod_speed_dr/results/stand/sh_lazy_rh_read_spec.json" in run["notes"]
    for cmd in cmds:
        if "$CK" in cmd:
            assert f"--deadband {sn['deployment']['constants']['residual_deadband']}" in cmd, cmd
    assert sum(bool(re.search(r"\$CK 1 ", cmd)) for cmd in cmds) == 3          # rows: the network, mask 1
    vids = [ln for cmd in cmds for ln in cmd.splitlines() if "--stand-ckpt" in ln and "sh_table_test" not in ln]
    assert vids and all("--stand-mask 1" in ln and "--checkpoint $SN_WALKER_CK" in ln for ln in vids)
    head = next(h for h in sn["evals"]["heads"] if h.get("adopted"))
    evr = [x for cmd in cmds for x in re.findall(r"--eval-run \$PWD/(\S+)", cmd)]
    assert evr and set(evr) == {head["stamps"]["hraise__omni@1"].removeprefix("parkour/")}
    old_run = _sup(m, "SH_T4F19")["block"]["evaluate"]["name"]
    taken = {h["id"] for h in sn["evals"]["heads"]} | {"SH_T17", "SH_T18", "SH_T16", "SH_T4F19_sh_s19", old_run,
                                                       "SH_RH_RSD_TDR"}
    assert not {run["name"], run["name"] + "z"} & (taken | {h + "z" for h in taken}), "a re-run must use FRESH names"
    # the recorded stamps (untracked) ran under exactly TDR's eval env, beside TDR, with this stand file
    for k, v in head["stamps"].items():
        if "@" not in k:
            continue
        st = _local(v)
        if st is not None and (st / "run_meta.json").is_file():
            meta = _json(st / "run_meta.json")
            assert meta["env_vars"] == tool.eval_env(m, w) and meta["checkpoint_sha256"] == w["sha256"], st
            assert meta["p2r_mode"]["specialist"]["sha256"] == sn["sha256"], st


def test_stand_network_videos_ran_the_bundled_pair():
    """The bake's walker + stand mix videos (untracked, recorded by the main session; skipped when absent): each sanity
    record names TDR and SH_RSD_TDR by sha, mask 1, the trained deadband, the restore armed, no fall, the validator's
    pass, the terrain its name says, and the adopted head's hraise@1 stamp as its reference row."""
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    w = tool.stage_by_dir(m, sn["walker"]["stage"])
    head = next(h for h in sn["evals"]["heads"] if h.get("adopted"))
    seen = 0
    for terrain, rel in RSD_VIDEOS.items():
        p = _local(rel)
        if p is None:
            continue
        v = _json(p)
        assert (v["stand_checkpoint_sha256"], v["checkpoint_sha256"]) == (sn["sha256"], w["sha256"]), rel
        assert (v["stand_mask"], v["residual_deadband"], v["restore"], v["falls"]) == (1, 0.25, "armed", []), rel
        assert v["validator"] == "pass" and v["terrain"]["terrain"] == terrain, rel
        assert v["reference_run_meta"].endswith("/" + head["stamps"]["hraise__omni@1"] + "/run_meta.json"), rel
        seen += 1
    if not seen:
        pytest.skip("the bake's stand mix videos are not on this machine (videos are untracked)")


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


#: the stand gate table's columns -> (read, role in that read): the stand net of record beside TDR
_SH_COLS = {"R_h_TDR": ("lazy", "R_h"), "SH_RSD_TDR": ("lazy", "SH_RSD_TDR_sh_s19"),
            "SH_RSD_TDR_s1era": ("orig", "SH_RSD_TDR_sh_s19"), "SH_T4F19": ("F19", "SH_T4F19_sh_s19"),
            "R_h_S1": ("T17", "R_h")}
#: ... and the superseded SH_T4F19's table (its block, kept verbatim)
_F19_COLS = {"R_h_T": ("lazy", "R_h"), "SH_T4F19": ("lazy", "SH_T4F19_sh_s19"),
             "SH_T4F19_s1era": ("orig", "SH_T4F19_sh_s19"), "SH_T17": ("T17", "SH_T17"), "R_h_S1": ("T17", "R_h")}


def _sh_reads() -> dict:
    return {"lazy": _json(RES / "stand/sh_lazy_rh_read.json"),
            "orig": _json(RES / "stand/sh_read.json"),
            "F19": _json(F19 / "stand/sh_lazy_rh_read.json"),
            "T17": _json(SH_CAMPAIGN / "head_SH_T17.json")}


def _f19_sh_reads() -> dict:
    return {"lazy": _json(F19 / "stand/sh_lazy_rh_read.json"),
            "orig": _json(F19 / "stand/sh_read.json"),
            "T17": _json(SH_CAMPAIGN / "head_SH_T17.json")}


def _check_gate_table(gt: dict, reads: dict, cols: dict, line_col: str):
    """Every cell of a K1-K6 table is its read at the manifest's rounding; the lines name ``line_col``'s read's clause
    lines; the verdict row names exactly each column's non-PASS gates."""
    assert [c["id"] for c in gt["columns"]] == list(cols)

    def read(col, K):
        rd, role = cols[col]
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
            g = read(line_col, gate)
            lines = {_num_str(c["line"] * scale) for k in (1, 2) for c in _k_clauses(g, kind, pat, k)}
            assert lines and all(x in r["line"] for x in lines), (r["key"], lines, r["line"])
    for col in gt["columns"]:
        cid = col["id"]
        failing = [f"{K} {read(cid, K)['verdict']}" for K in _GATES
                   if read(cid, K) is not None and read(cid, K)["verdict"] != "PASS"]
        assert gt["verdicts"][cid].startswith(", ".join(failing) if failing else "PASS"), (cid, gt["verdicts"][cid])


def test_stand_network_gate_table_matches_the_campaign_reads():
    """Every cell of the manifest's K1-K6 table is its read at the manifest's rounding: the re-anchored and the original
    reads beside TDR (results/stand/sh_lazy_rh_read.json / sh_read.json), the superseded SH_T4F19's beside T*
    (F19/stand/sh_lazy_rh_read.json) and R_h beside S1 (head_SH_T17.json); the lines name the re-anchored read's clause
    lines; each read carries the adopted checkpoint's sha and the trained deadband; the verdict row names exactly each
    column's non-PASS gates; both reads list exactly the two K5-control INVALID mismatches, nothing pending; the pipeline
    gate (gates/sh.json) HALTed on 12 of 101 lines, every one re-anchored by a recorded rule or accepted by the user;
    the head's stamps are the ones the re-anchored read used."""
    m = _m()
    sn = _stand(m)
    ev = sn["evals"]
    gt = ev["gate_table"]
    reads = _sh_reads()
    for r in ("lazy", "orig"):
        hd = reads[r]["heads"]["SH_RSD_TDR_sh_s19"]
        assert hd["stand_sha256"] == [sn["sha256"]] and (hd["residual_deadband"], hd["trained_residual_deadband"]) == (
            0.25, 0.25), r
        assert reads[r]["mismatches"] == K5_CONTROL_INVALID and reads[r]["pending"] == [], r
        assert reads[r]["meta"]["spec_body"]["refs"] == "REFS_RSD_TDR", r
    assert reads["F19"]["heads"]["SH_T4F19_sh_s19"]["stand_sha256"] == [_sup(m, "SH_T4F19")["block"]["sha256"]]
    assert ev["title"] == "Gates K1-K6 beside TDR"
    _check_gate_table(gt, reads, _SH_COLS, "SH_RSD_TDR")
    # the re-anchored read: K4 (restore misses) and the K5 identity fail; the original read fails K4 / K5 / K6
    assert reads["lazy"]["heads"]["SH_RSD_TDR_sh_s19"]["failed"] == ["K4", "K5"]
    assert reads["orig"]["heads"]["SH_RSD_TDR_sh_s19"]["failed"] == ["K4", "K5", "K6"]
    # the decisive pipeline gate: 12 failed lines = 10 re-anchored (4 body-height on R / HOLD, 6 K6 on the lazy R_h
    # control) + 2 K4 lines accepted and disclosed by the user; the counts per family the block states
    pg = ev["pipeline_gate"]
    g = _json(PKG / pg["record"])
    assert pg["record"] == "experiments/2026-11-07_rod_speed_dr/results/stand/gates/sh.json"
    assert (g["verdict"], g["thresholds"]["version"]) == (pg["verdict"], pg["thresholds"]) == ("HALT", "v1.2")
    assert g["halt_lines"] == pg["halt_lines"]
    bad = [x["id"] for x in g["decisive"] if not x["pass"]]
    assert bad == g["halt_lines"] and len(bad) == 12 and len(g["decisive"]) == 101
    ra = _json(PKG / pg["reanchored"]["record"])
    dec = _json(PKG / pg["accepted"]["record"])
    assert pg["reanchored"]["record"] == "experiments/2026-11-07_rod_speed_dr/results/stand/gates/sh_reanchor.json"
    assert pg["accepted"]["record"] == "experiments/2026-11-07_rod_speed_dr/results/stand/gates/sh_decision.json"
    body = [x["id"] for x in ra["body_height"]["lines"] if x["reanchored"]]
    assert len(body) == 4 and all(".K5.G5." in x and "body height" in x for x in body)
    assert all(all(c["control_fails_same_line"] and c["head_within_tol"] for c in x["controls"].values())
               for x in ra["body_height"]["lines"])
    lazy = ra["lazy_rh"]["covered"]
    assert len(lazy) == 6 and all(".K6." in x for x in lazy)
    assert [x["id"] for x in ra["lazy_rh"]["lines"]] == lazy
    assert all(x["control_fails_same_line"] and x["head_passes_reanchored"] for x in ra["lazy_rh"]["lines"])
    covered = set(body) | set(lazy)
    accepted = set(dec["accepted_lines"])
    assert len(covered) == 10 and len(accepted) == 2 and not covered & accepted
    assert set(bad) == covered | accepted and set(ra["remaining"]) == accepted == set(ra["lazy_rh"]["not_rh_derived"])
    assert all(".K4.miss " in x for x in accepted)
    vals = {x["id"]: x for x in g["decisive"]}
    for lid, e in dec["accepted_lines"].items():
        assert (vals[lid]["value"], vals[lid]["line"]) == (e["value"], e["line"]), lid
    assert pg["accepted"]["lines"] == dec["accepted_lines"]
    assert (dec["stage"], dec["by"], dec["user_words"], dec["decision"]) == ("sh", "user", "Accept and disclose",
                                                                            "accept, disclose")
    assert (pg["accepted"]["user_words"], pg["accepted"]["decided_at"], pg["accepted"]["rule"],
            pg["accepted"]["reason"], pg["accepted"]["decision"]) == (
        dec["user_words"], dec["decided_at"], dec["rule"], dec["reason"], dec["decision"])
    assert dec["decided_at"].startswith("2026-11-09T18:10") and dec["head"]["sha256"] == sn["sha256"]
    assert dec["gate"] == "results/stand/gates/sh.json" and dec["reanchor"] == "results/stand/gates/sh_reanchor.json"
    assert pg["reader_mismatches"] == K5_CONTROL_INVALID
    assert "12 of 101" in pg["resolution"] and "Accept and disclose" in pg["resolution"]
    counts = {}
    for x in g["decisive"]:
        K = x["id"].split(".")[2]
        counts.setdefault(K, [0, 0])
        counts[K][0] += bool(x["pass"])
        counts[K][1] += 1
    assert {K: f"{a} / {b}" for K, (a, b) in counts.items()} == pg["lines"]
    assert g["head"]["sha256"] == sn["sha256"]
    adopted = [h for h in ev["heads"] if h.get("adopted")]
    assert [h["id"] for h in adopted] == [sn["id"]] and adopted[0]["sha256"] == sn["sha256"]
    assert adopted[0]["source"] == sn["source"]
    rd = reads["lazy"]["reads"]["SH_RSD_TDR_sh_s19"]
    for k in (1, 2):
        for row in ("hwalkstop__omni", "hraise__omni"):
            assert rd[f"es{k}"][row]["stamp"].endswith(adopted[0]["stamps"][f"{row}@{k}"]), (row, k)
    # the K3 ungated cells the scope / caveat state are the read's
    rep = reads["lazy"]["gates"]["K3"]["SH_RSD_TDR_sh_s19"]["reported"]
    for cell, txt in (("ground__knee_folded", "ground knee_folded"), ("ground__random", "ground random")):
        pair = " / ".join(f"{rep[f'es{k}']['per_cell'][cell]['stood_share']:.2f}" for k in (1, 2))
        assert f"{txt} STOOD {pair}" in sn["deployment"]["scope"] or f"{txt} {pair}" in sn["deployment"]["scope"], cell


def test_superseded_sh_t4f19_block_still_matches_its_campaign_reads():
    """The kept SH_T4F19 record (superseded[SH_T4F19].block): its K1-K6 table is still the F19 stand reads
    (F19/stand/sh_lazy_rh_read.json / sh_read.json) and the superseded SH_T17's (head_SH_T17.json); its re-anchored read
    failed only the K5 identity, its original read K4 / K5 / K6; its pipeline gate (F19/gates/sh.json) HALTed on one K6
    line, an accepted halt of the F19 deliverable."""
    m = _m()
    blk = _sup(m, "SH_T4F19")["block"]
    ev = blk["evals"]
    reads = _f19_sh_reads()
    for r in ("lazy", "orig"):
        hd = reads[r]["heads"]["SH_T4F19_sh_s19"]
        assert hd["stand_sha256"] == [blk["sha256"]] and (hd["residual_deadband"], hd["trained_residual_deadband"]) == (
            0.25, 0.25), r
    assert reads["lazy"]["mismatches"] == [] and reads["lazy"]["pending"] == []
    assert reads["lazy"]["meta"]["spec_body"]["refs"] == "REFS_T4F19"
    _check_gate_table(ev["gate_table"], reads, _F19_COLS, "SH_T4F19")
    assert reads["lazy"]["heads"]["SH_T4F19_sh_s19"]["failed"] == ["K5"]
    assert reads["orig"]["heads"]["SH_T4F19_sh_s19"]["failed"] == ["K4", "K5", "K6"]
    pg = ev["pipeline_gate"]
    g = _json(PKG / pg["record"])
    assert (g["verdict"], g["thresholds"]["version"]) == (pg["verdict"], pg["thresholds"]) == ("HALT", "v1.2")
    assert g["halt_lines"] == [pg["halt_line"]]
    bad = [x for x in g["decisive"] if not x["pass"]]
    assert len(bad) == 1 and (bad[0]["value"], bad[0]["line"]) == (pg["halt_value"], pg["halt_line_value"])
    assert g["head"]["sha256"] == blk["sha256"]
    assert "sh" in {h["stage"] for h in _deliverable()["accepted_halts"]}


def test_readme_has_the_stand_network_section():
    tool = _tool()
    m = tool.load_manifest()
    sn = _stand(m)
    text = (POLICY / "README.md").read_text()
    assert text == tool.readme_text(m)
    path = f"{sn['dir']}/{sn['file']}"
    assert "## Stand network of record" in text and "### Deployment contract (as evaluated)" in text
    assert f"**Stand network of record:** [`{path}`]({path})" in text
    assert f"stand [`{STAND_DIR}/`]({STAND_DIR}/) (`model_5000.pt`)" in text
    # the warm start is named ONLY as the bundled ancestor, never as the net of record
    ws_lines = [line for line in text.splitlines() if f"[`{STAND_DIR}/model_4000.pt`]" in line]
    assert len(ws_lines) == 1 and ws_lines[0].startswith(
        "- **Warm-start ancestor (bundled, NOT the stand network of record):** "
        f"[`{STAND_DIR}/model_4000.pt`]({STAND_DIR}/model_4000.pt)")
    assert sn["warm_start"]["sha256"] in ws_lines[0] and sn["warm_start"]["role"] in ws_lines[0]
    # the superseded stand files appear ONLY in the superseded section
    for line in text.splitlines():
        for d in ("stand_net", OLD_STAND):
            for f in ("model_5000.pt", "model_4000.pt"):
                if f"[`{d}/{f}`]" in line:
                    assert line.startswith(f"  - [`{d}/{f}`]({d}/{f}) -- sha256 "), line
    assert sn["sha256"] in text and sn["walker"]["sha256"] in text and sn["source"] in text
    for s in ("stand_net_input", "hist_encoding=True", "CAVEAT (switch keying)", "CAVEAT (height source)",
              "**Zero-height mode:**", "**Restore before hand-back:**", "**Height command:**", "**Posture map q(h):**",
              "deadband(clamp(a_net, +-1), 0.25)", "  - **residual_deadband:** 0.25", "**Adoption grade:**",
              "### Gates K1-K6 beside TDR", "- **Training:**", "  - **trim_min:** 0.0999",
              "export SN_WALKER_CK=", "export P2_REANCHOR=1"):
        assert s in text, s
    assert "### Gates K1-K6 beside T*" not in text
    assert f"CK=$PWD/$P/policy/{path}" in text
    assert "bash $S/sh_eval_rows.sh --deadband 0.25 SH_RSD_TDR_re stand_net manifest armed $CK 1 \\" in text
    assert "bash $S/sh_eval_rows.sh --deadband 0.25 SH_RSD_TDR_rez stand_net zero armed $CK 1 \\" in text
    assert ("| Gate | Statistic | Line | R_h beside TDR (lazy control) | SH_RSD_TDR re-anchored (adopted) | "
            "SH_RSD_TDR vs the S1-era lines | SH_T4F19 beside T* (superseded) | R_h beside S1 |") in text
    assert ("| K4 | restore misses (of 800 restore-required exits) | <= max(1, R_h's) = 4 / 5 (R_h beside TDR) in this "
            "reader; the decisive pipeline gate holds D9's fixed 3 (results/stand/gates/sh.json) | 4 / 5 | 5 / 6 | "
            "5 / 6 | 2 / 3 | 1 / 1 |") in text
    assert "| K5 | G1-G5 at 1.064 (sn_calib, zero-command rows) as the reader reads it |" in text
    for c in sn["caveats"]:
        assert c in text
    for f in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "two_network_export", "stand_ground_getup",
              "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
              "stand_knee_hold_only", "stand_k5_identity", "stand_k4_restore_sag", "stand_low_bin_homing",
              "walker_stance_height_shift"):
        assert f"### `{f}` --" in text


def test_bake_disclosures_are_recorded():
    """The bake's disclosures (BAKE_PLAN section 2 item 6) are in the manifest: the stand block's caveats carry K4 5 / 6
    vs D9's 3, the K5.G2 walkstop homing, bin-1 homing, the one training seed, the uninformative exit-fall / onset
    readings, the two K5-control INVALID reader mismatches, the re-anchors and the walker's taller stance; the
    follow-ups carry the same plus the c28000 bridge class flip, camera use not re-tested for c31000 and TDR's unread
    180 kg."""
    m = _m()
    sn = _stand(m)
    cav = "\n".join(sn["caveats"])
    for s in ("5 / 6", "D9's fixed 3", "Accept and disclose", "0.92 / 0.91", "0.74 / 0.81", "one training seed",
              "0 exit falls", "-1.305", *K5_CONTROL_INVALID, "re-anchor", "0.03 m taller"):
        assert s in cav, s
    fu = {f["id"]: f for f in m["followups"]}
    assert "5 / 6" in fu["stand_k4_restore_sag"]["headline"] and "sh_decision.json" in fu["stand_k4_restore_sag"]["detail"]
    assert "0.74 / 0.81" in fu["stand_low_bin_homing"]["headline"]
    assert "one training seed" in fu["rod_speed_dr_one_seed"]["headline"]
    assert "0.03 m" in fu["walker_stance_height_shift"]["headline"]
    assert "bridge_C28000.json" in fu["eval_class_flip"]["detail"] and "class flip" in fu["eval_class_flip"]["detail"]
    assert "c31000" in fu["camera_use_unverified"]["detail"] and "not re-tested" in fu["camera_use_unverified"]["detail"]
    assert "180 kg was NEVER read for TDR" in fu["payload_sim_plant"]["detail"]
    assert all(s in fu["stand_k5_identity"]["detail"] for s in K5_CONTROL_INVALID)
    # SN's own four body-height re-anchors are disclosed on the warm start, the adoption grade names the HALT
    assert "re-anchored" in sn["warm_start"]["used_by"] and "4 / 4" in sn["warm_start"]["used_by"]
    assert "12 of 101" in sn["evals"]["adoption_grade"] and "NOT adoption-grade" in sn["evals"]["adoption_grade"]


def test_followups_track_the_new_head():
    """Every follow-up carried over is revised for the rod-speed DR bake (2026-11-09) and keeps the schema; the stand
    ones name SH_RSD_TDR; the five new ones (one seed, the K4 restore sag, low-bin homing, the walker's stance height,
    the eval class flip) exist and cite tracked evidence."""
    m = _m()
    fu = {f["id"]: f for f in m["followups"]}
    carried = ("move_turn", "c0_side_first_n1", "action_jitter", "phase2_clock_free", "stand_march",
               "stand_switch_keying", "stand_trim_underfit", "two_network_export", "stand_ground_getup",
               "stand_height_command_source", "stand_pd_torque_excess", "stand_hip_contacts_0p635",
               "stand_knee_hold_only", "p2b_negative", "back_overshoot", "camera_use_unverified", "payload_sim_plant",
               "student_checkpoint_not_bundled", "stand_k5_identity")
    new = ("rod_speed_dr_one_seed", "stand_k4_restore_sag", "stand_low_bin_homing", "walker_stance_height_shift",
           "eval_class_flip")
    assert list(fu) == [*carried, *new]
    for k in carried:
        assert fu[k]["recorded"].endswith("revised 2026-11-09 (rod-speed DR bake)") and fu[k]["detail"].startswith(
            "RSD BAKE (2026-11-09)"), k
    for k in carried[:15]:
        assert "revised 2026-10-24 (F19 bake)" in fu[k]["recorded"], k
    for k in carried[15:]:
        assert fu[k]["recorded"].startswith("2026-10-24"), k
    for k in new:
        assert fu[k]["recorded"].startswith("2026-11-09"), k
    for k in ("stand_march", "stand_switch_keying", "stand_trim_underfit", "stand_ground_getup",
              "stand_pd_torque_excess", "stand_k5_identity", "stand_k4_restore_sag", "stand_low_bin_homing"):
        assert "SH_RSD_TDR" in fu[k]["detail"], k
    for k in (*carried, *new):
        f = fu[k]
        assert {"id", "recorded", "headline", "detail"} <= set(f), k
        for e in f.get("evidence") or []:
            rel = e.split(" (")[0]
            assert (PKG / rel).exists(), (k, rel)
    for k in new:
        assert fu[k].get("evidence"), k
    assert "0.741" in fu["move_turn"]["detail"] and "0.658" in fu["move_turn"]["detail"]
    assert "RESOLVED" in fu["back_overshoot"]["detail"] and "0.993" in fu["back_overshoot"]["detail"]
    assert "0.04 / 0.06" in fu["stand_ground_getup"]["detail"]
    assert "4fa81f0f" in fu["student_checkpoint_not_bundled"]["detail"]
    assert "155b5981" in fu["student_checkpoint_not_bundled"]["detail"]


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
    assert not any(line.startswith("**Stand network of record:**") or f"stand [`{STAND_DIR}/`]" in line
                   or line.startswith("- Stand network (`stand_network:`") for line in without)
    ops = [op for op in difflib.SequenceMatcher(a=without, b=with_sn, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert ops and all(tag in ("insert", "replace") for tag, *_ in ops), ops
    removed = [line for tag, i1, i2, _, _ in ops if tag == "replace" for line in without[i1:i2]]
    heads_old = [line for line in without if line.startswith("**Head of record per phase:**")]
    assert removed == heads_old
    heads_new = [line for line in with_sn if line.startswith("**Head of record per phase:**")]
    assert heads_new == [heads_old[0][:-1] + f", stand [`{STAND_DIR}/`]({STAND_DIR}/) (`model_5000.pt`)."]


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
    (lambda m: m["stand_network"].update(dir=TDR), "collides with a stage dir"),
    (lambda m: m["stand_network"].update(dir="f19_8_p3b_payload180"), "collides with a stage dir"),
    (lambda m: m["stand_network"].update(dir="rsi"), "collides with an asset dir"),
    (lambda m: (m["stand_network"].update(dir="stand_net"), m["heads"].update(stand="stand_net")),
     "is a live policy file with another sha256"),
    (lambda m: (m["stand_network"].update(dir=OLD_STAND), m["heads"].update(stand=OLD_STAND)),
     "is a live policy file with another sha256"),
    (lambda m: m["stand_network"].pop("sha256"), "missing 'sha256'"),
    (lambda m: m["stand_network"].update(sha256="89f4428b"), "full lowercase sha256"),
    (lambda m: m["stand_network"].update(stand_mask=True), "int 0 or 1"),
    (lambda m: m["stand_network"]["walker"].update(sha256="0" * 64), "walker.sha256"),
    (lambda m: m["stand_network"]["walker"].update(stage="7_nope"), "walker.stage"),
    (lambda m: m["stand_network"]["walker"].update(stage="f19_6_p2a_field"), "walker.sha256"),
    (lambda m: m["stand_network"]["walker"].update(stage=TSTAR), "walker.sha256"),
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
    (lambda m: m["stand_network"]["warm_start"].update(sha256="6cd82ff5"), "warm_start.sha256 must be a full"),
    (lambda m: m["stand_network"]["warm_start"].update(dir="rsi"), "warm_start lives in the stand network's folder"),
    (lambda m: m["stand_network"]["warm_start"].update(file="model_5000.pt"), "different file from the stand network"),
    (lambda m: m["stand_network"]["warm_start"].update(sha256=m["stand_network"]["sha256"]),
     "different file from the stand network"),
    # the README-facing walker env is checkout-independent: $REPO/<path>, never an absolute path
    (lambda m: m["stand_network"]["evaluate"]["env"].update(SN_WALKER_CK="/home/x/krabby-research/parkour/m.pt"),
     "stand_network.evaluate.env.SN_WALKER_CK is an absolute path"),
    (lambda m: m["stand_network"]["evaluate"]["env"]["SN_WALKER_SPEC"].update(CANDIDATES="/r/parkour/c.yaml"),
     "stand_network.evaluate.env.SN_WALKER_SPEC.CANDIDATES is an absolute path"),
    (lambda m: m["stand_network"]["evaluate"]["env"]["SN_WALKER_SPEC"].update(REFS_ENV_STAMP="/r/parkour/s"),
     "stand_network.evaluate.env.SN_WALKER_SPEC.REFS_ENV_STAMP is an absolute path"),
    (lambda m: m["stand_network"]["evaluate"]["env"].update(SN_WALKER_ST="$REPO/parkour/a b"), "path must match"),
    # stage kinds, the distill stage and its exports
    (lambda m: _st(m, "f19_1_s1").update(kind="ppo"), "kind 'ppo' is not one of"),
    (lambda m: _st(m, "f19_1_s1").update(bundled=False), "for kind 'distill' stages only"),
    (lambda m: _st(m, TDR).update(bundled=False), "for kind 'distill' stages only"),
    (lambda m: _st(m, STUDENT).pop("exports"), "is not bundled and declares no exports"),
    (lambda m: _st(m, STUDENT).update(exports=[]), "exports must be a non-empty list"),
    (lambda m: _st(m, STUDENT).update(bundled="no"), "bundled must be true or false"),
    (lambda m: _st(m, STUDENT)["exports"][0].pop("sha256"), "missing 'sha256'"),
    (lambda m: _st(m, STUDENT)["exports"][0].update(sha256="ac0ea3e2"), "must be a full lowercase sha256"),
    (lambda m: _st(m, STUDENT)["exports"][0].update(file="model_31000.pt"), "plain non-checkpoint file"),
    (lambda m: _st(m, STUDENT)["exports"][1].update(file="policy.onnx"), "must be distinct files"),
    (lambda m: m.update(current=STUDENT), "is a stage whose checkpoint is not bundled"),
    (lambda m: _st(m, "f19_2_c4").update(sha256="0cbab2c4"), "sha256 must be a full lowercase sha256"),
    (lambda m: _st(m, TDR).update(sha256="03c3e472"), "sha256 must be a full lowercase sha256"),
    # superseded files kept in the folder
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir="f19_2_c4"), "sits in a live stage / asset / stand folder"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir=TDR), "sits in a live stage / asset / stand folder"),
    (lambda m: _sup(m, "F19_student")["files"][0].update(dir=STUDENT), "is a live policy file with another sha256"),
    (lambda m: _sup(m, "SH_T4F19")["files"][0].update(dir=STAND_DIR), "is a live policy file with another sha256"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir="f19_1_s1"), "is a live policy file with another sha256"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(dir="rsi"), "sits in a live stage / asset / stand folder"),
    (lambda m: _sup(m, "S1_s18")["files"][0].pop("sha256"), "missing 'sha256'"),
    (lambda m: _sup(m, "S1_s18")["files"][0].update(sha256="72e4b1e3"), "must be a full lowercase sha256"),
    (lambda m: _sup(m, "SH_T17")["files"].append(dict(_sup(m, "SN_Amask_s17")["files"][0])), "recorded twice"),
    (lambda m: _sup(m, "SH_T17")["files"].append(dict(_sup(m, "SN_T4F19")["files"][0])), "recorded twice"),
    (lambda m: _sup(m, "SH_T17")["files"].append({"dir": OLD_STAND, "file": "model_4000.pt", "sha256": "0" * 64,
                                                   "source": "x"}), "recorded twice"),
    (lambda m: _sup(m, "SH_T17")["files"].append({"dir": STAND_DIR, "file": "model_4000.pt", "sha256": "0" * 64,
                                                   "source": "x"}), "is a live policy file with another sha256"),
    (lambda m: _sup(m, "SH_T17").update(files=[]), "files must be a non-empty list"),
    (lambda m: _sup(m, "F19_student").update(files=[]), "files must be a non-empty list"),
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


def _kept_set(m: dict) -> set[str]:
    return {f"{d}/{s['file']}" for d, s in zip(S18_STAGES, _sup(m, "S1_s18")["stages"])} | {
        "stand_net/model_4000.pt", "stand_net/model_5000.pt",
        *(f"{OLD_STUDENT}/{f}" for f in ("policy.onnx", "depth_latest.onnx", "omni_student_io.json")),
        f"{OLD_STAND}/model_5000.pt", f"{OLD_STAND}/model_4000.pt"}


def test_check_flags_problems_in_kept_and_distill_folders(tmp_path):
    """--check on a sandbox copy: clean as is; flags a missing kept file, a stray checkpoint beside kept files, the
    student checkpoint dropped into its (unbundled) folder, a wrong sha on a kept file and a stale README."""
    tool = _load(TOOL, "bundle_policy_omni_sandbox")
    m = tool.load_manifest()
    pol = _sandbox(tmp_path, tool, m)
    assert tool.check(m) == []
    (pol / "stand_net" / "model_4000.pt").unlink()
    (pol / "stand_net" / "model_9.pt").write_bytes(b"x")
    (pol / STUDENT / "model_31000.pt").write_bytes(b"x")
    (pol / OLD_STAND / "model_4000.pt").unlink()
    bad = copy.deepcopy(m)
    _sup(bad, "S1_s18")["files"][0]["sha256"] = "0" * 64
    _sup(bad, "F19_student")["files"][0]["sha256"] = "0" * 64
    probs = "\n".join(tool.check(bad))
    assert "stand_net/model_4000.pt: missing" in probs
    assert f"{OLD_STAND}/model_4000.pt: missing" in probs
    assert "stand_net: stray checkpoints ['model_9.pt']" in probs
    assert f"{STUDENT}/model_31000.pt: stage '{STUDENT}' is `bundled: false`" in probs
    assert "1_fb6k/model_5999.pt: sha256" in probs
    assert f"{OLD_STUDENT}/policy.onnx: sha256" in probs
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
    assert {f"{k['dir']}/{k['file']}" for k in kept} == _kept_set(m)
    for k in kept:
        tool._verify_kept(k)
    (tool.POLICY / "6_p2a_field" / "model_19994.pt").unlink()
    with pytest.raises(SystemExit, match="never recreates a superseded file"):
        tool._verify_kept(next(k for k in kept if k["dir"] == "6_p2a_field"))
    (tool.POLICY / OLD_STAND / "model_5000.pt").unlink()
    with pytest.raises(SystemExit, match="never recreates a superseded file"):
        tool._verify_kept(next(k for k in kept if k["dir"] == OLD_STAND and k["file"] == "model_5000.pt"))
