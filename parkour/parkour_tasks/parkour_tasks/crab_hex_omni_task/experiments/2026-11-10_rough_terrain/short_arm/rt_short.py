#!/usr/bin/env python3
"""ARM step 2 (short_arm/ARM.md, Amendment A): the short arm's queue driver -- blocks strain, ssel, sconf1, sconf2.

    python -I -B rt_short.py --block strain|ssel|sconf1|sconf2 [--dry | --list] [--from L] [--only L ...]
                             [--test-pick c1|c2|c3|c4   (only with --dry; sconf1 / sconf2)]

Run it detached with ``rt_short_launch.sh BLOCK`` (setsid nohup; refuses --dry), watch the log with
``scripts/rt_wait.sh`` (a log sentinel; never pgrep). It runs the campaign's OWN queue (``rt_queue.main``: the
flock on ``.isaac.lock``, Isaac exclusivity, DDC / disk guards, pinned-build stamps, one ``RTQ <label> rc=<n>``
sentinel per step, ``RTQ_DONE <block> fail=<n>``) on plans built HERE, by patching ``rt_machinery.BLOCKS`` and
``rt_machinery.plan`` (the rt_sweep.py / rt_tdr_fresh.py pattern): no pinned or hashed campaign file changes.

Blocks (every gate row is ``M.gate_step`` into the campaign OUT/lr, a row's name ``<prefix>_<sha8>_<rid>``):

* strain (ARM step 4): S.check (build == pin), S.envcheck (RT_train.env == TDR's env + the planned changes; as
  plan_train's T.envcheck), S.train (``rt_short_train.py --name RT_s20s --wait-idle``, stop_on_fail): RT_s20s, 1010
  iterations from TDR on RT_train.env, seed 20, final model_30000.pt.
* ssel (step 5, 19 rows, prefix RTA): W.check, then c1 = R_suite es2 only (its RTSW rows hold slopes / ledge_down
  es1-2 and suite es1); c2, c3, c4 each rt_slopes_b2 es1-2 + rt_ledge_down_b2 es1-2 + R_suite es1-2 (6 rows).
* sconf1 (step 7, 12 rows, prefix RTC, THE PICK only): C1.check, C1.thresholds, then the 5 gate bands (slopes b2,
  ledge_up b2, ledge_down b2, pits b2, rubble b3 -- the thresholds' ``gated`` kinds) at es5, es6 + R_suite es5, es6.
* sconf2 (step 9, 21 rows, prefix RTC, the pick only): hills b3 + gravel b3 (the ``solved`` r_bands) at es5, es6;
  R_field_hard es5-8; R_field_light, R_ring_obst, R_field_turn, R_field_light_L090, R_field_light_L135,
  R_x07_L090_field_light at es5, es6; R_side22 es5 (exactly rows_rt's env_seeds + 4, as rt_tdr_fresh.py does for TDR).

Checkpoint identity (this file's own; never ``plan_gate`` / ``gate_ident``, which pin RT_s19 at 34990): c1 / c2 are
RT_s19 model_29500 / model_30000 and their sha256 must equal the pre-registered values (6d1f29be / a346185c) and
their ``iter`` field the filename's; c3 / c4 are RT_s20s model_29500 / model_30000: ``results/runs/RT_s20s.json``
status ok, name RT_s20s, not a smoke, cli seed 20 / iterations 1010 / TDR's checkpoint, final iter 30000 with
final.sha256 == the model_30000 file, and the two files' ``iter`` fields (torch.load on CPU, only that key) 29500 /
30000; the four shas are distinct and none is TDR. ssel refuses otherwise; under --dry / --list a c3 / c4 that
cannot be verified yet is shown as a PLACEHOLDER (name sha ``c3dry000`` / ``c4dry000``, a stand-in RT_s19 file path
so gate_row.sh's dry pass can run) and nothing about it is real.

The unblinding guard (sconf1, sconf2): they read ``results/short_arm/selection.json`` and REFUSE unless it is
committed -- ``git ls-files --error-unmatch``, ``git status --porcelain`` empty for it, ``git show HEAD:<path>`` bytes
== the file's bytes -- and its pick (``"pick": {"id": "c1".."c4", "sha256": ...}``, or ``"pick": "c2"`` with
``"pick_sha256"``) names the candidate id whose checkpoint file hashes to that sha256 (a ``path`` / ``checkpoint``
key, when present, must be that candidate's file). The ONLY bypass is ``--dry`` together with ``--test-pick cN``
(refused without --dry): the plan is then printed for that candidate and nothing runs or is written. TDR's RTF rows
(``RTF_03c3e472_*``, es5-8, PRE-REGISTERED UNREAD) are never opened or indexed here: this file reads only run_meta.json
of slots it is about to launch into (the slot-collision check below), and those names are RTA_* / RTC_*.

Slot-collision check (every plan): every launch slot's dir, when it exists, must hold only stamps of THIS row's
checkpoint (run_meta.json ``checkpoint_sha256``); a slot some other sha filled refuses the plan. Names are keyed on
the sha8 and the prefixes RTA / RTC differ from the campaign's RT6 / RT9 / RTF / RTSW.

``--dry``: nothing launches and nothing is written (rt_queue's dry pass: the build / env / thresholds checks run, the
trainer runs ``rt_short_train.py ... --dry-run``, every gate row runs gate_row.sh with TP_ROW_DRY=1).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
SCR = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/parkour_tasks/"
           "parkour_tasks/crab_hex_omni_task/experiments/2026-11-10_rough_terrain/scripts")
sys.path.insert(0, str(SCR))
import rt_machinery as M  # noqa: E402
import rt_queue as Q  # noqa: E402

NEW_BLOCKS = ("strain", "ssel", "sconf1", "sconf2")
NAME = "RT_s20s"
SHIFT = 4                                    # logical es1-4 -> fresh seeds es5-8 (as rt_tdr_fresh.py)
CIDS = ("c1", "c2", "c3", "c4")
RS19 = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/logs/rsl_rl/"
            "crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19")
#: id -> (run label, iter, pre-registered sha256 or None for the not-yet-run RT_s20s files)
POOL = {"c1": ("RT_s19", 29500, "6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1"),
        "c2": ("RT_s19", 30000, "a346185cd00119a385cbc45654d7ff9e4bf97a2bfbaaa9bfd91242ab8ec71faa"),
        "c3": (NAME, 29500, None), "c4": (NAME, 30000, None)}
THR_SHA8 = "ce45e01b"                        # thresholds_rough_v1.json, frozen (ARM Settled 1)
SEL_PATH = M.RESULTS / "short_arm" / "selection.json"
SCREEN_ROWS = (("rt_slopes_b2__omni", (1, 2)), ("rt_ledge_down_b2__omni", (1, 2)), ("R_suite", (1, 2)))
PREFIX_SEL, PREFIX_CONF = "RTA", "RTC"
EXPECT_GATED = {"rt_slopes_b2__omni", "rt_ledge_up_b2__omni", "rt_ledge_down_b2__omni", "rt_pits_b2__omni",
                "rt_rubble_b3__omni"}
EXPECT_SOLVED = {"rt_hills_b3__omni", "rt_gravel_b3__omni"}
EXPECT_SLOTS = {"strain": 0, "ssel": 19, "sconf1": 12, "sconf2": 21}
#: set by main(): --dry / --list (placeholders allowed), --dry (bypass allowed), --test-pick
MODE = {"relaxed": False, "dry": False, "test_pick": None}


# ============================================================================== checkpoint identity
def iter_field(path: Path):
    """The checkpoint's ``iter`` (torch.load on CPU; only that key is read)."""
    import torch  # noqa: PLC0415

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ck.get("iter") if isinstance(ck, dict) else None


def _file_problems(cid: str, path: Path, want_iter: int, want_sha: str | None) -> tuple[str | None, list[str]]:
    why = []
    if not path.is_file():
        return None, [f"{cid}: no checkpoint {path}"]
    sha = M.sha256_file(path)
    if want_sha is not None and sha != want_sha:
        why.append(f"{cid}: sha256 {sha[:8]} != pre-registered {want_sha[:8]}")
    it = iter_field(path)
    if it != want_iter:
        why.append(f"{cid}: {path.name} iter field {it!r} != {want_iter}")
    return sha, why


def pool_status() -> tuple[dict, dict]:
    """``(cands, problems)``: id -> {ck, sha, label} for every checkpoint that verified, id -> [why] for the rest."""
    cands, bad = {}, {}
    for cid in ("c1", "c2"):
        run, it, want = POOL[cid]
        ck = RS19 / f"model_{it}.pt"
        sha, why = _file_problems(cid, ck, it, want)
        (bad.__setitem__(cid, why) if why else cands.__setitem__(cid, {"ck": ck, "sha": sha, "label": f"{run} {it}"}))
    rec_path = M.RUNS / f"{NAME}.json"
    why = []
    run_dir = None
    try:
        rec = json.loads(rec_path.read_text())
    except (OSError, ValueError) as exc:
        rec, why = {}, [f"no readable run record {rec_path}: {exc}"]
    if rec:
        cli = rec.get("cli") or {}
        fin = rec.get("final") or {}
        if rec.get("name") != NAME or cli.get("smoke"):
            why.append(f"run record name {rec.get('name')!r} / smoke {cli.get('smoke')!r}")
        if rec.get("status") != "ok":
            why.append(f"run record status {rec.get('status')!r} != 'ok' (why: {rec.get('why')})")
        if (cli.get("seed"), cli.get("iterations"), cli.get("checkpoint")) != (20, 1010, str(M.TDR_CK)):
            why.append(f"run record cli seed/iterations/checkpoint {cli.get('seed')}/{cli.get('iterations')}/"
                       f"{cli.get('checkpoint')} != 20/1010/TDR")
        if fin.get("iter") != 30000:
            why.append(f"run record final iter {fin.get('iter')!r} != 30000")
        run_dir = Path(rec["run_dir"]) if rec.get("run_dir") else None
        if run_dir is None or not run_dir.is_dir():
            why.append(f"run record run_dir {rec.get('run_dir')!r} is not a directory")
            run_dir = None
    if run_dir is not None and not why:
        for cid in ("c3", "c4"):
            run, it, _ = POOL[cid]
            ck = run_dir / f"model_{it}.pt"
            sha, w = _file_problems(cid, ck, it, None)
            if cid == "c4" and not w and sha != (rec.get("final") or {}).get("sha256"):
                w.append(f"c4: sha256 {sha[:8]} != the run record's final {str((rec.get('final') or {}).get('sha256'))[:8]}")
            (bad.__setitem__(cid, w) if w else cands.__setitem__(cid, {"ck": ck, "sha": sha, "label": f"{run} {it}"}))
    else:
        for cid in ("c3", "c4"):
            bad[cid] = list(why)
    shas = [c["sha"] for c in cands.values()]
    if len(set(shas)) != len(shas):
        raise ValueError(f"pool checkpoints are not distinct: {[(k, v['sha'][:8]) for k, v in cands.items()]}")
    if M.TDR_SHA in shas:
        raise ValueError("a pool checkpoint IS TDR")
    return cands, bad


def candidates(need: tuple, relaxed: bool) -> dict:
    """The verified candidates in ``need``; a c3 / c4 that cannot be verified is refused, or (``relaxed``: --dry /
    --list, nothing launches) replaced by a labelled placeholder. c1 / c2 are never relaxed."""
    cands, bad = pool_status()
    out = {}
    for cid in need:
        if cid in cands:
            out[cid] = cands[cid]
        elif relaxed and cid in ("c3", "c4"):
            run, it, _ = POOL[cid]
            out[cid] = {"ck": RS19 / f"model_{it}.pt", "sha": f"{cid}dry000" + "0" * 56, "label": f"{run} {it}",
                        "placeholder": "; ".join(bad[cid])}
        else:
            raise ValueError("; ".join(bad[cid]))
    return out


# ============================================================================== the unblinding guard
def selection_problems(path: Path) -> list[str]:
    """[] iff ``path`` is a regular tracked file, unchanged in the work tree and the index, whose bytes equal HEAD's."""
    if path.is_symlink() or not path.is_file():
        return [f"no regular file {path}"]
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        rel = str(path.resolve().relative_to(Path(M.W).resolve()))
    except ValueError:
        return [f"{path} is outside the worktree {M.W}"]
    git = ["git", "-C", str(M.W)]
    why = []
    r = subprocess.run([*git, "ls-files", "--error-unmatch", "--", rel], capture_output=True, env=env)
    if r.returncode != 0:
        why.append("not tracked by git (git ls-files --error-unmatch)")
    r = subprocess.run([*git, "status", "--porcelain", "--", rel], capture_output=True, text=True, env=env)
    if r.returncode != 0 or r.stdout.strip():
        why.append(f"uncommitted changes (git status --porcelain: {r.stdout.strip()!r}, rc {r.returncode})")
    r = subprocess.run([*git, "show", f"HEAD:{rel}"], capture_output=True, env=env)
    if r.returncode != 0:
        why.append("not in HEAD (git show)")
    elif r.stdout != path.read_bytes():
        why.append("bytes differ from HEAD's")
    return why


def pick_from_selection(path: Path) -> tuple[str, str, str | None]:
    """``(id, sha256, given checkpoint path or None)`` of the committed selection's pick, or ValueError."""
    why = selection_problems(path)
    if why:
        raise ValueError(f"{path} is not committed ({'; '.join(why)}): the fresh-seed rows stay unread and "
                         f"sconf rows do not launch until it is")
    doc = json.loads(path.read_text())
    pick = doc.get("pick")
    if isinstance(pick, dict):
        cid, sha = pick.get("id"), pick.get("sha256") or pick.get("sha")
        given = pick.get("checkpoint") or pick.get("path")
    else:
        cid, sha, given = pick, doc.get("pick_sha256"), None
    if cid not in CIDS or not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"{path}: no usable pick (expected \"pick\": {{\"id\": c1..c4, \"sha256\": <64 hex>}}; "
                         f"got id {cid!r}, sha {sha!r})")
    return cid, sha, given


def pick_candidate() -> tuple[str, dict, str]:
    """``(id, candidate, how)`` of the pick the confirm rows run on."""
    if MODE["test_pick"]:
        cid = MODE["test_pick"]
        return cid, candidates((cid,), relaxed=True)[cid], f"TEST PICK {cid} (dry only; selection.json NOT read)"
    cid, sha, given = pick_from_selection(SEL_PATH)
    cand = candidates((cid,), relaxed=False)[cid]
    if cand["sha"] != sha:
        raise ValueError(f"{SEL_PATH}: pick {cid} sha256 {sha[:8]} != sha256 of the checkpoint file "
                         f"{cand['ck']} ({cand['sha'][:8]})")
    if given is not None and Path(str(given)).resolve() != Path(cand["ck"]).resolve():
        raise ValueError(f"{SEL_PATH}: pick path {given} != candidate {cid}'s file {cand['ck']}")
    return cid, cand, f"pick {cid} from the committed {SEL_PATH.name} (sha {sha[:8]})"


# ============================================================================== the plans
def _rows() -> tuple[dict, dict]:
    rows = M.rt_spec().load_rows()
    return {r["id"]: r for r in rows["kind_rows"] + rows["retention_rows"]}, rows


def _thresholds() -> dict:
    if M.sha256_file(M.THRESHOLDS)[:8] != THR_SHA8:
        raise ValueError(f"{M.THRESHOLDS} sha {M.sha256_file(M.THRESHOLDS)[:8]} != the frozen {THR_SHA8}")
    return M.load_thresholds(M.THRESHOLDS)


def _row(label, prefix, cid, cand, by_id, rid, es):
    st = M.gate_step(label, name=M.row_name(prefix, cand["sha"], rid), ck=cand["ck"], row=by_id[rid], es=es,
                     tp_out=M.OUT)
    if cand.get("placeholder"):
        st.note = (f"DRY PLACEHOLDER for {cid} ({cand['label']}): {cand['placeholder']}; stand-in checkpoint path, "
                   f"name sha {cand['sha'][:8]} -- nothing about this row is real")
    return st


def plan_strain() -> list:
    steps = M.check_steps("S")
    steps.append(M._cpu("S.envcheck", M.HERE / "rt_machinery.py", "train-env-check", stop=True, dry=True))
    steps.append(M.Step("S.train", "train", [M.PY, "-I", "-B", str(HERE / "rt_short_train.py"), "--name", NAME,
                                             "--wait-idle"], stop_on_fail=True,
                        note="resume TDR, 1010 iterations (28991 -> 30000), seed 20, RT_train.env, MemoryMax 48G, "
                             "SIGTERM-only; saves model_29500 / model_30000"))
    return steps


def plan_ssel() -> list:
    by_id, _ = _rows()
    cands = candidates(CIDS, MODE["relaxed"])
    steps = M.check_steps("W")
    for cid in CIDS:
        for rid, ess in SCREEN_ROWS:
            if cid == "c1" and rid != "R_suite":
                continue                                  # c1's slopes / ledge_down es1-2 and suite es1 are the RTSW rows
            for es in ((2,) if cid == "c1" else ess):
                steps.append(_row(f"W.{cid}.{rid}.es{es}", PREFIX_SEL, cid, cands[cid], by_id, rid, es))
    return steps


def plan_sconf(block: str) -> list:
    by_id, rows = _rows()
    thr = _thresholds()
    cid, cand, how = pick_candidate()
    gated = {M.rt_spec().row_id(k, kd["gate_band"]) for k, kd in thr["kinds"].items() if kd.get("status") == "gated"}
    solved = {M.rt_spec().row_id(k, kd["r_band"]) for k, kd in thr["kinds"].items() if kd.get("status") == "solved"}
    if gated != EXPECT_GATED or solved != EXPECT_SOLVED:
        raise ValueError(f"thresholds' gated {sorted(gated)} / solved {sorted(solved)} != the ARM's "
                         f"{sorted(EXPECT_GATED)} / {sorted(EXPECT_SOLVED)}")
    gr = [(rid, tuple(es + SHIFT for es in ess)) for rid, ess in M.gate_rows(thr)]
    ret = [(r["id"], tuple(es + SHIFT for es in r["env_seeds"])) for r in rows["retention_rows"]]
    pref = f"C{block[-1]}"
    if block == "sconf1":
        todo = [x for x in gr if x[0] in gated] + [x for x in ret if x[0] == "R_suite"]
    else:
        todo = [x for x in gr if x[0] in solved] + [x for x in ret if x[0] != "R_suite"]
    steps = M.check_steps(pref)
    steps.append(M._cpu(f"{pref}.thresholds", M.MAKE_THRESHOLDS, "--check", "--out", M.THRESHOLDS, stop=True, dry=True,
                        note="thresholds_rough_v1.json regenerates byte-equal (frozen at STOP 0)"))
    steps[-1].note += f"; {how}"
    for rid, ess in todo:
        for es in ess:
            if not 5 <= es <= 8:
                raise ValueError(f"{rid}: confirm seed es{es} outside 5-8")
            steps.append(_row(f"{pref}.{rid}.es{es}", PREFIX_CONF, cid, cand, by_id, rid, es))
    return steps


PLANS = {"strain": plan_strain, "ssel": plan_ssel, "sconf1": lambda: plan_sconf("sconf1"),
         "sconf2": lambda: plan_sconf("sconf2")}


def slot_problems(steps: list) -> list[str]:
    """Every launch slot that exists must hold only stamps of the checkpoint the step launches (run_meta.json)."""
    out = []
    for s in steps:
        if not s.slot:
            continue
        tp_out, name, sc, es = s.slot
        ck = Path(s.argv[3])
        want = M.sha256_file(ck) if ck.is_file() else None
        slot = M.slot_dir(tp_out, name, sc, es)
        if not slot.is_dir():
            continue
        if s.note.startswith("DRY PLACEHOLDER"):
            want = None                                   # a stand-in path: only its (absent) own-name slot can be checked
        for d in sorted(p for p in slot.iterdir() if p.is_dir()):
            try:
                got = json.loads((d / "run_meta.json").read_text()).get("checkpoint_sha256")
            except (OSError, ValueError):
                out.append(f"{s.label}: {d} has no readable run_meta.json")
                continue
            if want is None or got != want:
                out.append(f"{s.label}: slot {slot} holds a stamp of sha {str(got)[:8]}, not this row's "
                           f"{'placeholder' if want is None else want[:8]}")
    return out


def plan_short(block: str) -> list:
    steps = PLANS[block]()
    labels = [s.label for s in steps]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate step labels")
    slots = [s.slot for s in steps if s.slot]
    if len(slots) != len(set(slots)):
        raise RuntimeError("two gate rows share an output slot")
    if len(slots) != EXPECT_SLOTS[block]:
        raise RuntimeError(f"{block}: {len(slots)} gate rows, the ARM pre-registers {EXPECT_SLOTS[block]}")
    bad = slot_problems(steps)
    if bad:
        raise ValueError("slot collision: " + "; ".join(bad))
    return steps


def install() -> None:
    """Patch the campaign's machinery (idempotent): the four blocks and their plan dispatch."""
    if getattr(M, "_rt_short_installed", False):
        return
    orig = M.plan
    M.BLOCKS = (*M.BLOCKS, *NEW_BLOCKS)
    M.plan = lambda block, *a, **k: plan_short(block) if block in NEW_BLOCKS else orig(block, *a, **k)
    M._rt_short_installed = True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--test-pick", choices=CIDS, default=None)
    pre.add_argument("--block", default=None)
    ns, rest = pre.parse_known_args(argv)
    if ns.block is not None:
        rest = ["--block", ns.block, *rest]
    dry = "--dry" in rest
    if ns.test_pick and not dry:
        print(M.done_line(ns.block or "?", 1, "(--test-pick is refused without --dry: the pick comes only from a "
                                              "committed selection.json)"), flush=True)
        return 2
    if ns.test_pick and ns.block not in ("sconf1", "sconf2"):
        print(M.done_line(ns.block or "?", 1, "(--test-pick only applies to sconf1 / sconf2)"), flush=True)
        return 2
    MODE.update(relaxed=dry or "--list" in rest, dry=dry, test_pick=ns.test_pick)
    install()
    return Q.main(rest)


if __name__ == "__main__":
    sys.exit(main())
