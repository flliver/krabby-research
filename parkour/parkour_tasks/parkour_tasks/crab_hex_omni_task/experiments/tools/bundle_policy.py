#!/usr/bin/env python3
"""Sync / check the crab-hex OMNI-walk POLICY OF RECORD: ``<task>/policy/``.

``policy/`` holds the head that ships for the omnidirectional task plus the stage heads it was
trained through (t3 formation -> c4 -> c5 -> c6 -> c6x clock-free), one subfolder per stage,
declared in ``policy/manifest.yaml`` (source run, sha256, campaign, overlay reference, per-category
suite evals, observation contract, open follow-ups), plus the shared training assets the lineage
depends on (``assets:`` -- the omni RSI reference bank every omni preset seeds resets from). It
changes ONLY on a bake decision, which is the user's: edit the manifest, run ``--sync``, commit.

    python3 experiments/tools/bundle_policy.py --sync     # copy missing stage files (sha-verified),
                                                          #   verify present ones, regenerate README.md,
                                                          #   git add the produced files
    python3 experiments/tools/bundle_policy.py --check    # verify every stage file + sha + README (test hook)

Adapted from ``crab_hex_forward_task/experiments/tools/bundle_policy.py``. Differences:

* every stage is a phase-1a stage (``KRABBY_OMNI_PHASE=1a``) plus a per-stage OVERLAY from a
  campaign's ``candidates.yaml`` (``stage`` + ``overlay_ref``) -- there are no omni presets for these
  overlays, so there is nothing to pin: no ``--pin-mdp``, no ``mdp_pins.yaml``, no
  ``POLICY_SUMMARY.md`` (the forward tool's generated-summary hooks are dropped);
* evals are the per-category suite (``suite__omni``: survival per category, speed ratio on the
  travel categories, facing_closed / yaw_progress on turn), not slow / step / obst;
* the manifest carries an ``obs_contract:`` (the clock columns and the eval-carry semantics the heads
  trained under; per-stage ``KRABBY_OMNI_CLOCK_HIDE``) and ``followups:``, both rendered in the README.

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
PKG = HERE.parents[1]                     # crab_hex_omni_task/
REPO = PKG.parents[3]
POLICY = PKG / "policy"
MANIFEST = POLICY / "manifest.yaml"
GENERATED_DOCS = ("README.md",)
HIDE_VAR = "KRABBY_OMNI_CLOCK_HIDE"
#: Suite categories in report order, and which extra per-category metrics each carries.
CATEGORIES = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
RATIO_CATS = ("fwd", "side", "back", "diag", "move_turn")
#: The harness row every stage's eval was scored on, and the scripts the README commands name.
SUITE_ROW = "suite__omni"
EVAL_SCRIPT = "parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
OMNI_MANIFEST = "parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_files(m: dict) -> set[str]:
    """Every file that may live under policy/, relative to policy/ (the layout test pins this set)."""
    files = {f"{s['dir']}/{s['file']}" for s in m["stages"]}
    files |= {f"{a['dir']}/{a['file']}" for a in m.get("assets", [])}
    files |= {"manifest.yaml", *GENERATED_DOCS}
    return files


def load_manifest() -> dict:
    m = yaml.safe_load(MANIFEST.read_text())
    for k in ("task", "plant", "current", "stages", "obs_contract"):
        if k not in m:
            raise SystemExit(f"{MANIFEST}: missing key {k!r}")
    dirs = [s["dir"] for s in m["stages"]]
    if len(set(dirs)) != len(dirs):
        raise SystemExit(f"{MANIFEST}: duplicate stage dirs")
    if m["current"] not in dirs:
        raise SystemExit(f"{MANIFEST}: current={m['current']!r} is not a stage dir")
    for s in m["stages"]:
        for k in ("dir", "phase", "stage", "task", "iterations", "source", "file", "sha256", "overlay_ref",
                  "obs_contract"):
            if k not in s:
                raise SystemExit(f"{MANIFEST}: stage {s.get('dir')!r} missing {k!r}")
        if str(s["obs_contract"].get(HIDE_VAR)) not in ("0", "1"):
            raise SystemExit(f"{MANIFEST}: stage {s['dir']!r} obs_contract.{HIDE_VAR} must be '0' or '1'")
    oc = m["obs_contract"]
    for k in ("width", "clock_columns", "eval_env"):
        if k not in oc:
            raise SystemExit(f"{MANIFEST}: obs_contract missing {k!r}")
    if HIDE_VAR in (oc["eval_env"].get("carry") or {}) or HIDE_VAR in (oc["eval_env"].get("fixed") or {}):
        raise SystemExit(f"{MANIFEST}: {HIDE_VAR} is per stage (stage obs_contract), not in obs_contract.eval_env")
    m.setdefault("assets", [])
    for a in m["assets"]:
        for k in ("dir", "file", "source", "sha256"):
            if k not in a:
                raise SystemExit(f"{MANIFEST}: asset {a.get('file')!r} missing {k!r}")
        if a["dir"] in dirs:
            raise SystemExit(f"{MANIFEST}: asset dir {a['dir']!r} collides with a stage dir")
    m.setdefault("followups", [])
    for f in m["followups"]:
        for k in ("id", "headline"):
            if k not in f:
                raise SystemExit(f"{MANIFEST}: followup {f.get('id')!r} missing {k!r}")
    return m


def stage_by_dir(m: dict, d: str) -> dict:
    return next(s for s in m["stages"] if s["dir"] == d)


def stage_path(s: dict) -> Path:
    return POLICY / s["dir"] / s["file"]


def asset_path(a: dict) -> Path:
    return POLICY / a["dir"] / a["file"]


def carry_env(m: dict, s: dict) -> dict[str, str]:
    """The observation / command semantics stage ``s`` trained under (EVAL_CARRY keys + its clock hide)."""
    ev = {k: str(v) for k, v in (m["obs_contract"]["eval_env"].get("carry") or {}).items()}
    ev[HIDE_VAR] = str(s["obs_contract"][HIDE_VAR])
    return ev


def eval_env(m: dict, s: dict) -> dict[str, str]:
    """The full environment a driver eval of stage ``s`` ran under (carry + clock + fixed eval block)."""
    ev = carry_env(m, s)
    ev.update({k: str(v) for k, v in (m["obs_contract"]["eval_env"].get("fixed") or {}).items()})
    return ev


def play_env(m: dict, s: dict) -> dict[str, str]:
    """What a viewer of stage ``s`` must export: the carried semantics, the ring geometry, the plant."""
    ev = carry_env(m, s)
    fixed = m["obs_contract"]["eval_env"].get("fixed") or {}
    for k in ("KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_PLANT"):
        if k in fixed:
            ev[k] = str(fixed[k])
    return ev


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
    subprocess.run(["git", "add", "--", *[str(p) for p in produced]], cwd=REPO, check=True)
    return produced


# ------------------------------------------------------------------------------ README rendering
def _f(v) -> str:
    """A metric at up to three decimals, never fewer than one (1.0, 0.99, 0.365)."""
    if v is None:
        return "—"
    if not isinstance(v, (int, float)):
        return str(v)
    s = f"{float(v):.3f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _cell(cats: dict, c: str) -> str:
    e = (cats or {}).get(c)
    if not e:
        return "—"
    parts = [_f(e.get("survival"))]
    if c in RATIO_CATS and e.get("ratio") is not None:
        parts.append(f"r {_f(e['ratio'])}")
    if c == "turn":
        if e.get("facing_closed") is not None:
            parts.append(f"fc {_f(e['facing_closed'])}")
        if e.get("yaw_progress") is not None:
            parts.append(f"yp {_f(e['yaw_progress'])}")
    return " / ".join(parts)


def _env_lines(env: dict[str, str], indent: str = "  ") -> list[str]:
    """``KEY=value \\`` continuation lines, one per variable, the clock-hide knob first."""
    keys = sorted(env, key=lambda k: (k != HIDE_VAR, k))
    return [f"{indent}{k}={env[k]} \\" for k in keys]


def readme_text(m: dict) -> str:
    cur = stage_by_dir(m, m["current"])
    oc = m["obs_contract"]
    hide_cur = str(cur["obs_contract"][HIDE_VAR])
    cur_path = f"{cur['dir']}/{cur['file']}"
    L = [
        f"# Policy of record — `{m['task']}`",
        "",
        "GENERATED by `experiments/tools/bundle_policy.py --sync` from [`manifest.yaml`](manifest.yaml) -- edit the",
        "manifest, not this file.",
        "",
        "This folder holds the head that **ships** for the omnidirectional walk task (phase 1) and the stage heads it",
        f"was trained through, one subfolder per stage. Plant: **{m['plant']}** (the main asset `assets/crab.usda`;",
        f"nothing to set). Seed {m.get('seed', '—')}. It changes only on a bake decision, which is the user's; the",
        "campaigns that produced these heads keep their own records under `experiments/<campaign>/`.",
        "",
        f"**Current head:** [`{cur_path}`]({cur_path}) -- {cur.get('what', '')}",
        "",
        f"**Baked:** {m.get('baked', '—')}",
        "",
        f"**Recipe:** {m.get('recipe', '—')}",
        "",
        "## Observation contract — read before playing, evaluating or exporting",
        "",
        f"- **Width:** {oc['width']}. **Clock columns:** {oc['clock_columns']}.",
        f"- **The current head was trained with `{HIDE_VAR}={hide_cur}`**"
        + (" -- the clock columns are ZERO on every frame. Any play / eval / export of it MUST set "
           f"`{HIDE_VAR}=1`." if hide_cur == "1" else "."),
        f"- **Per stage:** " + ", ".join(f"`{s['dir']}/` {HIDE_VAR}={s['obs_contract'][HIDE_VAR]}"
                                          for s in m["stages"]) + ".",
    ]
    if oc.get("hazard"):
        L.append(f"- **Hazard:** {oc['hazard']}")
    if oc.get("refs"):
        L.append(f"- **Code:** {oc['refs']}")
    L += [
        "- **Semantics the heads trained under** (`run_phases.EVAL_CARRY`; identical on every stage's eval stamp):",
        "",
        "  | variable | value |",
        "  |---|---|",
    ]
    L += [f"  | `{k}` | `{v}` |" for k, v in (oc["eval_env"].get("carry") or {}).items()]
    L += [f"  | `{HIDE_VAR}` | per stage (above) |", "",
          "- **Fixed driver eval block** (not policy inputs except the geometry and plant): "
          + ", ".join(f"`{k}={v}`" for k, v in (oc["eval_env"].get("fixed") or {}).items()) + "."]
    if oc["eval_env"].get("fixed_notes"):
        L.append(f"  {oc['eval_env']['fixed_notes']}")
    L += [
        "",
        f"Play the current head (from `krabby-research/parkour/`, Isaac venv python; `{cur.get('play_task', '—')}`):",
        "",
        "```bash",
        "P=parkour_tasks/parkour_tasks/crab_hex_omni_task",
        "env \\",
        *_env_lines(play_env(m, cur)),
        f"  python scripts/rsl_rl/play.py --task {cur.get('play_task', '—')} \\",
        f"  --num_envs 1 --real-time --checkpoint $PWD/$P/policy/{cur_path}",
        "```",
        "",
        f"Score it on the per-category suite exactly as the driver did (`{SUITE_ROW}`, 100 episodes per category):",
        "",
        "```bash",
        "P=parkour_tasks/parkour_tasks/crab_hex_omni_task",
        "env \\",
        *_env_lines(eval_env(m, cur)),
        f"  python {EVAL_SCRIPT} --headless \\",
        f"  --manifest {OMNI_MANIFEST} --scenario {SUITE_ROW} \\",
        f"  --checkpoint $P/policy/{cur_path} --save-raw --trained-categories {','.join(CATEGORIES)}",
        "```",
        "",
        f"For a clock-held stage (`{HIDE_VAR}=0`) use the same commands with `{HIDE_VAR}=0`.",
        "",
    ]
    if m.get("followups"):
        L += ["## Open follow-ups", ""]
        for f in m["followups"]:
            L += [f"### `{f['id']}` -- {f['headline']}", ""]
            if f.get("recorded"):
                L.append(f"- **Recorded:** {f['recorded']}")
            if f.get("detail"):
                L.append(f"- **Detail:** {f['detail']}")
            if f.get("evidence"):
                L.append("- **Evidence:** " + ", ".join(f"`{e}`" for e in f["evidence"]))
            if f.get("levers"):
                L.append(f"- **Candidate levers:** {f['levers']}")
            L.append("")
    L += [
        "## Stages",
        "",
        "| # | Stage | Preset + overlay | Iterations | Task | Resumes | Clock | File | sha256 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(m["stages"], 1):
        L.append(f"| {i} | [`{s['dir']}/`]({s['dir']}/) | `{s['phase']}` + `{s['stage']}` | {s['iterations']} | `{s['task']}` | "
                 f"{s.get('resume_from', '—')} | {s.get('clock', '—')} | `{s['file']}` | `{s['sha256'][:12]}` |")
    L += ["", f"Evals (`{SUITE_ROW}`): survival per category (reached the end of its hold), `r` = speed ratio, turn "
          "`fc` = facing_closed / `yp` = yaw_progress (rad/s); side / turn / move_turn pool both hands (n 200), the rest "
          "n 100. Read with `experiments/tools/run_phases.summarize(aggregate)['cat']`.", "",
          "| Stage | clock at eval | " + " | ".join(CATEGORIES) + " |",
          "|---|---|" + "---|" * len(CATEGORIES)]
    for s in m["stages"]:
        ev = s.get("evals") or {}
        L.append(f"| `{s['dir']}/` | {ev.get('clock_at_eval', '—')} | "
                 + " | ".join(_cell(ev.get("categories"), c) for c in CATEGORIES) + " |")
        tw = s.get("twin")
        if tw:
            L.append(f"| _{tw['id']} twin (not baked)_ | {tw.get('clock_at_eval', '—')} | "
                     + " | ".join(_cell(tw.get("categories"), c) for c in CATEGORIES) + " |")
    L.append("")
    for s in m["stages"]:
        ev = s.get("evals") or {}
        L += [f"### `{s['dir']}/` -- {s.get('what', '')}", "",
              f"- **Preset + overlay:** `KRABBY_OMNI_PHASE={s['phase']}` (`config/crab_hex/crab_hex_omni_phases.py`) + "
              f"stage `{s['stage']}` from {s['overlay_ref']} (task `{s['task']}`, kind `{s.get('kind', '')}`)",
              f"- **Iterations:** {s['iterations']}; resumes {s.get('resume_from', '—')}",
              f"- **Clock:** {s.get('clock', '—')}; observation contract `{HIDE_VAR}={s['obs_contract'][HIDE_VAR]}`",
              f"- **Source run (not tracked):** `{s['source']}`",
              f"- **Trained in:** {s.get('trained_in', '—')}",
              f"- **sha256:** `{s['sha256']}`"]
        if ev.get("stamp"):
            L.append(f"- **Eval stamp (not tracked):** `{ev['stamp']}`")
        if s.get("notes"):
            L.append(f"- **Notes:** {s['notes']}")
        tw = s.get("twin")
        if tw:
            L.append(f"- **Twin:** {tw['id']} -- {tw.get('what', '')}; `{tw.get('source', '—')}` "
                     f"(sha256 `{tw.get('sha256', '—')}`), eval `{tw.get('stamp', '—')}`")
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
        "## Resuming training from a stage head",
        "",
        "`KRABBY_OMNI_PHASE=1a` + the next stage's overlay (its `overlay_ref`) `... train.py --task "
        "Isaac-Crab-Hex-Omni-Walk-v0 --resume --checkpoint <policy path>`, as the campaign drivers "
        "(`experiments/tools/run_candidates.py`, `checkpoint:` resume kind) do. A fork of the current head keeps "
        f"`{HIDE_VAR}=1` and the eps clock weights unless it deliberately restores the clock.",
        "",
        "## Maintenance",
        "",
        "- `python3 parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/bundle_policy.py --check` (from "
        "`parkour/`; the tool is cwd-independent) -- every stage file present with its manifest sha, no stray "
        "checkpoints, README fresh (run by `tests/unit/test_policy_of_record_omni.py`).",
        "- New bake: add/replace the stage entry in `manifest.yaml` (source run, sha256, campaign, overlay_ref, "
        "obs_contract, evals), set `current`, run `--sync`, commit.",
        "- New shared asset (e.g. a re-harvested RSI bank): add an `assets:` entry (dir, file, source, sha256, what, "
        "used_by), run `--sync`, commit.",
        "- No MDP pins: the omni lineage's stage recipes are overlays on the 1a preset recorded in campaign "
        "`candidates.yaml` files, not presets, so there is no `--pin-mdp` / `POLICY_SUMMARY.md` here (the forward "
        "policy folder has both).",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sync", action="store_true", help="copy missing stage files, verify, write README, git add")
    g.add_argument("--check", action="store_true", help="verify every stage file + sha and the README; exit 1 on problems")
    ap.add_argument("--force", action="store_true", help="with --sync: recopy every stage file from its source")
    a = ap.parse_args()
    m = load_manifest()
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
