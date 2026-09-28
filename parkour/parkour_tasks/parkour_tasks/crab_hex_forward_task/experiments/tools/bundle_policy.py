#!/usr/bin/env python3
"""Sync / check the crab-hex forward-walk POLICY OF RECORD: ``<task>/policy/``.

``policy/`` holds the head that ships for this task plus the stage heads it was trained through
(1a formation -> 2a -> 2b -> 2c teacher -> 3a depth student), one subfolder per stage, declared in
``policy/manifest.yaml`` (source run, sha256, campaign, evals), plus the shared training assets the
lineage depends on (``assets:`` -- e.g. the RSI reference bank every preset seeds resets from). It
changes ONLY on a bake decision, which is the user's: edit the manifest, run ``--sync``, commit.

    python3 experiments/tools/bundle_policy.py --sync     # copy missing stage files (sha-verified),
                                                          #   verify present ones, regenerate README.md,
                                                          #   git add the produced files
    python3 experiments/tools/bundle_policy.py --check    # verify every stage file + sha (test hook)

Stdlib + PyYAML; no Isaac import.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]                     # crab_hex_forward_task/
REPO = PKG.parents[3]
POLICY = PKG / "policy"
MANIFEST = POLICY / "manifest.yaml"
SUMMARY = POLICY / "POLICY_SUMMARY.md"           # hand-written prose + generated blocks (policy_summary.py)
PINS = POLICY / "mdp_pins.yaml"                   # Isaac-only numbers, extracted by --pin-mdp
GENERATED_DOCS = ("README.md", "POLICY_SUMMARY.md")
EVAL_KEYS = (("slow", "flat canary"), ("step", "step onset"), ("obst", "obstacles 0.20-0.70"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _summary():
    """The block renderer / pins extractor module (sibling file), loaded by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("policy_summary", HERE / "policy_summary.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["policy_summary"] = mod
    spec.loader.exec_module(mod)
    return mod


def expected_files(m: dict) -> set[str]:
    """Every file that may live under policy/, relative to policy/ (the layout test pins this set)."""
    files = {f"{s['dir']}/{s['file']}" for s in m["stages"]}
    files |= {f"{a['dir']}/{a['file']}" for a in m.get("assets", [])}
    files |= {"manifest.yaml", PINS.name, *GENERATED_DOCS}
    return files


def load_manifest() -> dict:
    m = yaml.safe_load(MANIFEST.read_text())
    for k in ("task", "plant", "current", "stages"):
        if k not in m:
            raise SystemExit(f"{MANIFEST}: missing key {k!r}")
    dirs = [s["dir"] for s in m["stages"]]
    if len(set(dirs)) != len(dirs):
        raise SystemExit(f"{MANIFEST}: duplicate stage dirs")
    if m["current"] not in dirs:
        raise SystemExit(f"{MANIFEST}: current={m['current']!r} is not a stage dir")
    for s in m["stages"]:
        for k in ("dir", "phase", "task", "iterations", "source", "file", "sha256"):
            if k not in s:
                raise SystemExit(f"{MANIFEST}: stage {s.get('dir')!r} missing {k!r}")
    m.setdefault("assets", [])
    for a in m["assets"]:
        for k in ("dir", "file", "source", "sha256"):
            if k not in a:
                raise SystemExit(f"{MANIFEST}: asset {a.get('file')!r} missing {k!r}")
        if a["dir"] in dirs:
            raise SystemExit(f"{MANIFEST}: asset dir {a['dir']!r} collides with a stage dir")
    return m


def stage_path(s: dict) -> Path:
    return POLICY / s["dir"] / s["file"]


def asset_path(a: dict) -> Path:
    return POLICY / a["dir"] / a["file"]


def check(m: dict) -> list[str]:
    """Problems with the tracked policy files (empty list = ok)."""
    problems = []
    for s in m["stages"]:
        p = stage_path(s)
        if not p.exists():
            problems.append(f"{p.relative_to(REPO)}: missing")
            continue
        got = sha256_of(p)
        if got != s["sha256"]:
            problems.append(f"{p.relative_to(REPO)}: sha256 {got[:12]} != manifest {s['sha256'][:12]}")
        extra = [q for q in p.parent.glob("*.pt") if q.name != s["file"]]
        if extra:
            problems.append(f"{p.parent.relative_to(REPO)}: stray checkpoints {[q.name for q in extra]}")
    for a in m["assets"]:
        p = asset_path(a)
        if not p.exists():
            problems.append(f"{p.relative_to(REPO)}: missing")
            continue
        got = sha256_of(p)
        if got != a["sha256"]:
            problems.append(f"{p.relative_to(REPO)}: sha256 {got[:12]} != manifest {a['sha256'][:12]}")
    readme = POLICY / "README.md"
    if not readme.exists() or readme.read_text() != readme_text(m):
        problems.append(f"{readme.relative_to(REPO)}: stale (bundle_policy.py --sync)")
    problems += _summary().check_summary(m)
    return problems


def _sync_one(dest: Path, src: Path, sha: str, label: str, *, force: bool) -> None:
    if dest.exists() and not force:
        got = sha256_of(dest)
        if got != sha:
            raise SystemExit(f"{dest}: sha256 {got[:12]} != manifest {sha[:12]} (use --force to recopy)")
        print(f"ok      {dest.relative_to(REPO)}")
        return
    if not src.exists():
        raise SystemExit(f"{label}: source {src} missing (raw run dirs are not tracked; recover the file first)")
    got = sha256_of(src)
    if got != sha:
        raise SystemExit(f"{label}: source {src} sha256 {got[:12]} != manifest {sha[:12]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"copied  {dest.relative_to(REPO)} ({dest.stat().st_size / 1e6:.1f} MB)")


def sync(m: dict, *, force: bool) -> list[Path]:
    produced = []
    for s in m["stages"]:
        dest = stage_path(s)
        _sync_one(dest, REPO / s["source"], s["sha256"], s["dir"], force=force)
        produced.append(dest)
    for a in m["assets"]:
        dest = asset_path(a)
        _sync_one(dest, REPO / a["source"], a["sha256"], a["file"], force=force)
        produced.append(dest)
    readme = POLICY / "README.md"
    readme.write_text(readme_text(m))
    print(f"wrote   {readme.relative_to(REPO)}")
    produced += [readme, MANIFEST]
    ps = _summary()
    if SUMMARY.exists() and PINS.exists():
        SUMMARY.write_text(ps.update_summary(SUMMARY.read_text(), ps.render_blocks(m, ps.load_pins())))
        print(f"wrote   {SUMMARY.relative_to(REPO)} (generated blocks)")
        produced += [SUMMARY, PINS]
    else:
        print(f"skip    {SUMMARY.relative_to(REPO)}: needs {PINS.name} (bundle_policy.py --pin-mdp <dump dir>) and the hand-written document")
    subprocess.run(["git", "add", "--", *[str(p) for p in produced]], cwd=REPO, check=True)
    return produced


def _ev(s: dict, key: str) -> str:
    e = (s.get("evals") or {}).get(key)
    if not e:
        return "—"
    parts = [f"{e['completion']:.2f}", f"{e['falls']} falls"]
    if e.get("tripod") is not None:
        parts.append(f"tripod {e['tripod']:.2f}")
    return " / ".join(parts)


def readme_text(m: dict) -> str:
    cur = next(s for s in m["stages"] if s["dir"] == m["current"])
    L = [
        f"# Policy of record — `{m['task']}`",
        "",
        "GENERATED by `experiments/tools/bundle_policy.py --sync` from [`manifest.yaml`](manifest.yaml) -- edit the",
        "manifest, not this file.",
        "",
        f"This folder holds the head that **ships** for the forward-walk task and the stage heads it was trained",
        f"through, one subfolder per stage. Plant: **{m['plant']}** (the main asset `assets/crab.usda`; nothing to set).",
        "It changes only on a bake decision, which is the user's: the campaign that produced a new head keeps its",
        "own sha-verified copy under `experiments/<campaign>/head/`; this folder is the shipped lineage.",
        "Shared training assets the lineage depends on (the RSI reference bank) are bundled alongside, see",
        "[Shared assets](#shared-assets).",
        "",
        f"**Current head:** [`{cur['dir']}/{cur['file']}`]({cur['dir']}/{cur['file']}) -- {cur.get('what', '')}",
        "",
        "**What it was trained on:** [`POLICY_SUMMARY.md`](POLICY_SUMMARY.md) -- per-phase goals, active rewards /",
        "terrain / other configuration, and the reward, terrain and knob catalogues (prose hand-written, tables",
        "generated from the presets, this manifest and [`mdp_pins.yaml`](mdp_pins.yaml)).",
        "",
        "| # | Stage | Preset | Iterations | Task | Resumes | File | sha256 | " + " | ".join(t for _, t in EVAL_KEYS) + " |",
        "|---|---|---|---|---|---|---|---|" + "---|" * len(EVAL_KEYS),
    ]
    for i, s in enumerate(m["stages"], 1):
        L.append(f"| {i} | [`{s['dir']}/`]({s['dir']}/) | `{s['phase']}` | {s['iterations']} | `{s['task']}` | {s.get('resume_from', '—')} | "
                 f"`{s['file']}` | `{s['sha256'][:12]}` | " + " | ".join(_ev(s, k) for k, _ in EVAL_KEYS) + " |")
    L += ["", "Evals: completion / falls out of 100 episodes / tripod score, from the producing campaign's records "
          "(flat canary = morph-manifest `slow__A15pB`; step onset = `step__A15pB`; obstacles = `flat_walk_slow_v2` on "
          "`recal2b2w` @ 0.20-0.70). Task = the preset's task. The 2a-2c files were trained as flat-walk lineage windows on "
          "`Isaac-Crab-Hex-Flat-Walk-v0` (log dir `crab_hex_flat_walk/`); the `Isaac-Crab-Hex-Teacher-v0` presets rebuild the "
          "identical MDP (pinned by `tests/integration/test_crab_hex_phase_configs.py`).", "", "## Stages", ""]
    for s in m["stages"]:
        L += [f"### `{s['dir']}/` -- {s.get('what', '')}", "",
              f"- **Preset:** `KRABBY_PHASE={s['phase']}` (`config/crab_hex/crab_hex_phases.py`; task `{s['task']}`, kind `{s.get('kind', '')}`, window {s.get('window', '—')})",
              f"- **Iterations:** {s['iterations']}; resumes {s.get('resume_from', '—')}",
              f"- **Source run (not tracked):** `{s['source']}`",
              f"- **Trained in:** {s.get('trained_in', '—')}",
              f"- **sha256:** `{s['sha256']}`"]
        if s.get("notes"):
            L.append(f"- **Notes:** {s['notes']}")
        L.append("")
    if m["assets"]:
        L += ["## Shared assets", ""]
        for a in m["assets"]:
            L += [f"### [`{a['dir']}/{a['file']}`]({a['dir']}/{a['file']}) -- {a.get('what', '')}", "",
                  f"- **Used by:** {a.get('used_by', '—')}",
                  f"- **Source (tracked):** `{a['source']}`",
                  f"- **sha256:** `{a['sha256']}`"]
            if a.get("notes"):
                L.append(f"- **Notes:** {a['notes']}")
            L.append("")
    L += [
        "## Using the heads",
        "",
        "Paths below are relative to `krabby-research/parkour/` (run from there with the Isaac venv python, headless).",
        "Task README sections: play [§4.3](../README.md#43-play-a-bundled-checkpoint) / phase-3 [§4.4](../README.md#44-student-distillation), "
        "gait harness [§4.1b](../README.md#41b-gait-metrics-eval-harness-milestone-18-task-0).",
        "",
        "```bash",
        "P=parkour_tasks/parkour_tasks/crab_hex_forward_task",
        "# flat canary of the current head, as the phase driver scores it. NOTE: the table's 3a flat number (0.79 / 21) was",
        "# recorded with the pre-2026-09-09 morph manifest, whose slow__A15pB carried an env: block; today's slow__A15pB has none,",
        "# and the harness's post-Kit env write moves the depth student between two repeatable outcomes (0.79/21 vs 0.75/25,",
        "# task README section 4.1b) -- same policy, different process state.",
        "KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_SIGMA2=0.1 KRABBY_TRACK_L1_W=-1.0 KRABBY_CLOCK_W=1.0 KRABBY_APEX_W=1.0 KRABBY_STUDENT_MDP=1 \\",
        f"  python $P/scripts/eval_crab_hex_gait.py --headless --manifest $P/experiments/eval/scenarios_morph.yaml \\",
        f"  --scenario slow__A15pB --task Isaac-Crab-Hex-Student-v0 --checkpoint $P/policy/{cur['dir']}/{cur['file']} --save-raw",
        "# a teacher-stage head (2c) on the same canary: drop KRABBY_STUDENT_MDP and the --task override",
        "KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_SIGMA2=0.1 KRABBY_TRACK_L1_W=-1.0 KRABBY_CLOCK_W=1.0 KRABBY_APEX_W=1.0 \\",
        f"  python $P/scripts/eval_crab_hex_gait.py --headless --manifest $P/experiments/eval/scenarios_morph.yaml \\",
        "  --scenario slow__A15pB --checkpoint $P/policy/2c_teacher/model_19996.pt",
        "```",
        "",
        "Resuming training from a stage head: `KRABBY_PHASE=<next preset> ... train.py --resume --checkpoint <policy path>`",
        "as in the task README §4.0 / §4.4; the presets encode each stage's MDP.",
        "",
        "## Maintenance",
        "",
        "- `python3 parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/tools/bundle_policy.py --check` (from `parkour/` like the commands above; the tool is cwd-independent) -- every stage file present with its manifest sha "
        "(run by `tests/unit/test_policy_of_record.py`).",
        "- New bake: add/replace the stage entry in `manifest.yaml` (source run, sha256, campaign, evals), set `current`, "
        "run `--sync`, commit. Old heads stay under their campaign's `head/`.",
        "- New shared asset (e.g. a re-harvested RSI bank): add an `assets:` entry (dir, file, source, sha256, what, "
        "used_by), run `--sync`, commit.",
        "- After any MDP / reward / terrain / runner change: regenerate the Isaac config dumps "
        "(`KRABBY_CFG_DUMP_DIR=<dir> RUN_CRAB_HEX_CFG_IDENTITY=1 pytest tests/integration/test_crab_hex_phase_configs.py`), "
        "run `bundle_policy.py --pin-mdp <dir>` (rewrites `mdp_pins.yaml`), then `--sync` (rewrites the generated blocks of "
        "`POLICY_SUMMARY.md`); update its prose by hand.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sync", action="store_true", help="copy missing stage files, verify, write README, git add")
    g.add_argument("--check", action="store_true", help="verify every stage file + sha, README and POLICY_SUMMARY.md blocks; exit 1 on problems")
    g.add_argument("--pin-mdp", metavar="DUMP_DIR", help="extract policy/mdp_pins.yaml from cfg_<phase>.json Isaac config dumps in DUMP_DIR")
    ap.add_argument("--force", action="store_true", help="with --sync: recopy every stage file from its source")
    a = ap.parse_args()
    m = load_manifest()
    if a.pin_mdp:
        ps = _summary()
        dumps = ps.read_dump_dir(Path(a.pin_mdp), [s["phase"] for s in m["stages"]])
        pins = ps.pins_from_dumps(dumps, m)
        ps.write_pins(pins)
        print(f"wrote   {PINS.relative_to(REPO)} ({len(pins)} phases: {', '.join(pins)}); now run --sync")
        return 0
    if a.check:
        problems = check(m)
        for p in problems:
            print("PROBLEM", p)
        print(f"policy of record: {len(m['stages'])} stages, current {m['current']}: {'OK' if not problems else 'FAILED'}")
        return 1 if problems else 0
    sync(m, force=a.force)
    total = (sum(stage_path(s).stat().st_size for s in m["stages"])
             + sum(asset_path(a).stat().st_size for a in m["assets"])) / 1e6
    print(f"synced {len(m['stages'])} stages + {len(m['assets'])} assets ({total:.1f} MB), current {m['current']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
