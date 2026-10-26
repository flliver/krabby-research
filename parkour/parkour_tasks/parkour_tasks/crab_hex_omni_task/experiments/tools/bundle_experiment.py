#!/usr/bin/env python3
"""Bundle a campaign's keep-set: its checkpoint of record, eval summaries, RSI banks and an index.

A campaign directory under ``crab_hex_omni_task/experiments/`` MAY carry a ``bundle.yaml``
manifest (so far: ``2026-09-16_1805_omni_formation``, ``2026-10-16_task4_pipeline``; the other rows of
``experiments/README.md`` are hand-maintained and must be re-added after an ``--index`` run). This tool
(stdlib + PyYAML, no Isaac) turns a manifest into the tracked bundle::

    bundle_experiment.py <campaign_dir>...      # bundle these campaigns
    bundle_experiment.py --all                  # every campaign with a bundle.yaml
    bundle_experiment.py --all --check          # re-hash heads/banks, no writes (same check tests/unit/test_experiment_layout.py runs itself)
    bundle_experiment.py --index                # regenerate experiments/README.md
    bundle_experiment.py --dry-run ...          # report what would be copied

Manifest (``bundle.yaml``)::

    campaign: 2026-09-16_1805_omni_formation
    era: B                 # A = artifacts in the campaign tree; B = under parkour/logs/rsl_rl
    dates: 2026-09-16..
    question: one line
    verdict: one line (+ pointer to the record)
    plant: A15+B | legacy_golden | per-arm variants | pre-generator snapshot | hand-authored crab_simple.usda (era A, pre-2026-08-20 rebuild)
    aliases: [sim_fine_tuning/baseline]          # pre-rename dir names seen in old records
    records: [REPORT.md, CHANGELOG.md, state.json]
    head:                                        # ONE checkpoint of record ...
      role: teacher_2b | teacher_2a | flat_walk
      task: Isaac-Crab-Hex-Omni-Teacher-v0
      source: parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-09-.../model_14997.pt   # repo-relative
                                                 # (era A: campaign-relative, e.g. logs/rsl_rl/...)
      sha256: <64 hex>                           # filled by --init when absent
      dest: head/model_14997.pt
      why: one line
      metrics: {flat_canary: "0.79 / 21 / 0.51", ...}
      env: {KRABBY_OMNI_PHASE: "2b"}
      extra_heads: [{role: teacher, source: ..., sha256: ..., dest: head/model_9998.pt}]
    # ... or  head: {none: "why there is no checkpoint of record"}
    # ... or  head: {ref: 2026-08-08_1701_stride_length_v3}   (another campaign's bake)
    banks:                                        # reference RSI banks tracked at the campaign top level
      - dest: rsi_bank_omni.npz                    # campaign-relative; the file itself is NOT copied here
        sha256: <64 hex>                           # or a TODO_* placeholder before it is pinned (--check skips it, with a note)
        why: one line (harvest recipe / provenance pointer)
    evals:                                       # era B: copy the keep-set from parkour/logs
      - {source: parkour/logs/rsl_rl/gait_eval/omni_phases, dest: evals/omni_phases}
      # era A: - {in_tree: "**/gait_eval*/**"}   (already tracked in place; counted only)
    keep_globs: [run_meta.json, scenario_metrics.json, summary.md, "gait_diagram_*.png"]
    paths_note: true                             # prepend the old->new path note to the primary record

What it guarantees: the copied head's sha256 equals the manifest (and the source), heads stay
under 64 MB, only keep-glob files are copied from eval sources, every listed bank's sha256 matches
the file already sitting at ``<campaign>/<dest>`` (unless its manifest value is a ``TODO_*``
placeholder -- not yet pinned, verification is skipped with a printed note), nothing git-ignored
and no stray ``.pt/.npz/.log/.mp4`` is staged, and ``head/README.md`` is written in the
``old-runs/`` README style.

The index's "eval runs" column counts eval STAMPS: distinct directories under the campaign that hold a
``run_meta.json`` or a ``summary.md`` (one per stamp in both the driver keep-set and the lean Task 4
keep-set, which tracks no ``run_meta.json``).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PKG = EXP.parent
REPO = PKG.parents[3]
HEAD_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_KEEP = ["run_meta.json", "scenario_metrics.json", "summary.md", "gait_diagram_*.png"]
STRAY = (".pt", ".npz", ".log", ".mp4", ".pid")
PATHS_NOTE_MARK = "<!-- paths-note -->"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(cdir: Path) -> dict:
    mf = cdir / "bundle.yaml"
    if not mf.exists():
        raise SystemExit(f"{cdir.name}: no bundle.yaml")
    b = yaml.safe_load(mf.read_text()) or {}
    for key in ("campaign", "era", "dates", "question", "verdict", "head"):
        if key not in b:
            raise SystemExit(f"{cdir.name}: bundle.yaml lacks '{key}'")
    if b["campaign"] != cdir.name:
        raise SystemExit(f"{cdir.name}: bundle.yaml campaign field is {b['campaign']!r}")
    if b["era"] not in ("A", "B"):
        raise SystemExit(f"{cdir.name}: era must be A or B")
    for rec in b.get("records") or []:
        if not (cdir / rec).exists():
            raise SystemExit(f"{cdir.name}: record {rec} missing")
    return b


def resolve_source(cdir: Path, era: str, source: str) -> Path:
    p = Path(source)
    if p.is_absolute():
        return p
    cand = (cdir / p) if era == "A" else (REPO / p)
    if not cand.exists():  # era-A manifests may still use repo-relative paths
        alt = REPO / p
        if alt.exists():
            return alt
    return cand


def git_ignored(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    out = subprocess.run(["git", "check-ignore", "--stdin"], cwd=REPO, input="\n".join(str(p) for p in paths),
                         capture_output=True, text=True)
    return [Path(l) for l in out.stdout.splitlines() if l]


def git_add(paths: list[Path]) -> None:
    if paths:
        subprocess.run(["git", "add", "--", *[str(p) for p in paths]], cwd=REPO, check=True)


def head_readme(cdir: Path, b: dict, head: dict, size: int, sha: str, extra: list[dict]) -> str:
    rel_head = f"{EXP.relative_to(REPO)}/{cdir.name}/{head['dest']}"
    task = head.get("task", "")
    env = head.get("env") or {}
    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    play_task = {"teacher": "Isaac-Crab-Hex-Omni-Teacher-Play-v0"}.get(head.get("role"), "Isaac-Crab-Hex-Omni-Walk-Play-v0")
    # the play line runs from parkour/: repo-relative env values (e.g. a variant USD path) need a ../ prefix
    play_env = " ".join(f"{k}={'../' + str(v) if (REPO / str(v)).exists() else v}" for k, v in env.items())
    lines = [
        f"# {cdir.name} — checkpoint of record",
        "",
        f"**Bundled checkpoint:** `{head['dest']}` ({size / 1e6:.1f} MB, sha256 `{sha}`)",
        "",
        f"- **Source (not tracked):** `{head['source']}`",
        f"- **Role / task:** {head.get('role', '?')} / `{task}`" if task else f"- **Role:** {head.get('role', '?')}",
        f"- **Plant:** {b.get('plant', '?')}",
        f"- **Why this checkpoint:** {head.get('why', '(see the campaign record)')}",
        f"- **Environment of record:** `{env_str}`" if env_str else "- **Environment of record:** (see the campaign record)",
    ]
    if head.get("metrics"):
        lines += ["", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in head["metrics"].items()]
    for x in extra:
        lines += ["", f"**Extra head:** `{x['dest']}` ({x.get('role', '?')}, sha256 `{x['sha256']}`) — source `{x['source']}`"]
    lines += [
        "",
        "## Play / evaluate",
        "",
        "```bash",
        "cd parkour",
        "env $(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /') OMNI_KIT_ACCEPT_EULA=yes "
        f"{play_env + ' ' if play_env else ''}/home/nickmagus/krabby/isaac_venv/bin/python scripts/rsl_rl/play.py "
        f"--task {play_task} --num_envs 1 --checkpoint ../{rel_head}",
        "```",
        "",
        "The `env -u` prefix drops every `KRABBY_*` variable inherited from the shell, so the environment of record "
        "above is the whole knob set. play.py writes an export (`exported_teacher/`, JIT + ONNX) beside the "
        "checkpoint it plays: copy the head elsewhere first to keep `head/` clean.",
    ]
    if extra:
        lines += ["", "The extra heads are not played by this line: see the campaign's own README (and, for a baked "
                  "head, `crab_hex_omni_task/policy/README.md`) for their play / eval / video commands."]
    lines += [
        "",
        "## Paths note",
        "",
        f"This campaign lives at `{EXP.relative_to(REPO)}/{cdir.name}/`. Checkpoint paths inside its records point "
        "at the raw artifacts (`parkour/logs/rsl_rl/...` for era-B campaigns, the campaign's own `logs/rsl_rl/...` "
        "tree for era A), which stay on disk but are not tracked; the tracked copy of the checkpoint of record is "
        "this `head/` directory.",
    ]
    if b.get("aliases"):
        lines.append("Older records may call this directory " + ", ".join(f"`{a}`" for a in b["aliases"]) + ".")
    lines += ["", f"_Generated by `experiments/tools/bundle_experiment.py` on {datetime.now():%Y-%m-%d}._", ""]
    return "\n".join(lines)


def paths_note(cdir: Path, b: dict) -> str:
    # The omni package was created on 2026-09-16 and its campaigns are born here, so (like the
    # sideways copy of this tool, unlike the forward one) there is no sim_fine_tuning -> experiments/
    # move to explain.
    return (
        f"{PATHS_NOTE_MARK}\n> **Paths note:** this campaign lives at "
        f"`{EXP.relative_to(REPO)}/{cdir.name}/`. Absolute paths below (`/home/.../parkour/logs/rsl_rl/...`) "
        "name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` "
        "(see `bundle.yaml`), reference RSI banks are tracked at the campaign top level (`banks:`), and the "
        "eval keep-set is under `evals/`."
        + (" Older records may call this directory " + ", ".join(f"`{a}`" for a in b["aliases"]) + "." if b.get("aliases") else "")
        + "\n\n"
    )


def primary_record(cdir: Path, b: dict) -> Path | None:
    for name in ("REPORT.md", "RESULTS.md", "CHANGELOG.md", "PLAN.md"):
        if (cdir / name).exists():
            return cdir / name
    recs = [cdir / r for r in (b.get("records") or []) if r.endswith(".md")]
    return recs[0] if recs else None


def copy_head(cdir: Path, b: dict, h: dict, *, check: bool, dry: bool, force: bool, init: bool) -> tuple[Path, str, int]:
    src = resolve_source(cdir, b["era"], h["source"])
    if not src.exists():
        raise SystemExit(f"{cdir.name}: head source missing: {src}")
    size = src.stat().st_size
    if size > HEAD_MAX_BYTES:
        raise SystemExit(f"{cdir.name}: head {src.name} is {size / 1e6:.0f} MB (> 64 MB guard)")
    sha = sha256_of(src)
    if h.get("sha256"):
        if h["sha256"] != sha:
            raise SystemExit(f"{cdir.name}: sha256 mismatch for {src}: manifest {h['sha256'][:12]} vs file {sha[:12]}")
    elif not init:
        raise SystemExit(f"{cdir.name}: head has no sha256 (run with --init to record {sha[:12]}...)")
    dest = cdir / h["dest"]
    if dest.exists():
        dsha = sha256_of(dest)
        if dsha != sha and not force:
            raise SystemExit(f"{cdir.name}: {dest} exists with a different sha ({dsha[:12]}); use --force")
    if check or dry:
        return dest, sha, size
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or sha256_of(dest) != sha:
        shutil.copy2(src, dest)
    if sha256_of(dest) != sha:
        raise SystemExit(f"{cdir.name}: copy verification failed for {dest}")
    return dest, sha, size


def check_banks(cdir: Path, b: dict) -> list[str]:
    """Verify the ``banks:`` block (D6/D12): each listed reference bank sits at ``<campaign>/<dest>``
    ALREADY (harvested straight into the campaign directory, never copied by this tool) and its
    sha256 matches the manifest -- unless the manifest still carries a ``TODO_*`` placeholder (not
    yet pinned; skipped with a note, not an error, so a pre-harvest bundle.yaml stays checkable)."""
    notes: list[str] = []
    for bank in b.get("banks") or []:
        dest = cdir / bank["dest"]
        want = str(bank.get("sha256") or "")
        if want.upper().startswith("TODO"):
            notes.append(f"  [note] bank {bank['dest']}: sha256 not yet pinned ({want}) — skipping verification")
            continue
        if not dest.exists():
            raise SystemExit(f"{cdir.name}: bank {bank['dest']} missing (manifest sha256 {want[:12]})")
        got = sha256_of(dest)
        if got != want:
            raise SystemExit(f"{cdir.name}: bank {bank['dest']} sha256 mismatch: manifest {want[:12]} vs file {got[:12]}")
        notes.append(f"  bank {bank['dest']}: sha256 OK ({got[:12]})")
    return notes


def copy_evals(cdir: Path, b: dict, *, dry: bool) -> tuple[int, int, list[Path]]:
    keep = b.get("keep_globs") or DEFAULT_KEEP
    n = 0
    total = 0
    produced: list[Path] = []
    for ev in b.get("evals") or []:
        if "in_tree" in ev:
            n += sum(1 for p in cdir.glob(ev["in_tree"]) if p.is_file() and any(fnmatch.fnmatch(p.name, g) for g in keep))
            continue
        src_root = REPO / ev["source"]
        if not src_root.exists():
            print(f"  [warn] eval source missing: {ev['source']}")
            continue
        dest_root = cdir / ev["dest"]
        for p in src_root.rglob("*"):
            if not p.is_file() or not any(fnmatch.fnmatch(p.name, g) for g in keep):
                continue
            rel = p.relative_to(src_root)
            d = dest_root / rel
            n += 1
            total += p.stat().st_size
            produced.append(d)
            if dry:
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists() or d.stat().st_size != p.stat().st_size:
                shutil.copy2(p, d)
    return n, total, produced


def bundle(cdir: Path, *, check: bool, dry: bool, force: bool, init: bool) -> dict:
    b = load_manifest(cdir)
    head = b["head"] or {}
    info = {"campaign": cdir.name, "head": None, "heads_bytes": 0, "evals": 0, "evals_bytes": 0, "bank_notes": []}
    produced: list[Path] = []
    if head.get("dest"):
        dest, sha, size = copy_head(cdir, b, head, check=check, dry=dry, force=force, init=init)
        info["head"] = dest.relative_to(cdir)
        info["heads_bytes"] += size
        extras = []
        for x in head.get("extra_heads") or []:
            xd, xsha, xsize = copy_head(cdir, b, x, check=check, dry=dry, force=force, init=init)
            extras.append({**x, "sha256": xsha})
            info["heads_bytes"] += xsize
            produced.append(xd)
        if init:
            head["sha256"] = sha
            for x, xx in zip(head.get("extra_heads") or [], extras):
                x["sha256"] = xx["sha256"]
        if not (check or dry):
            readme = dest.parent / "README.md"
            readme.write_text(head_readme(cdir, b, head, size, sha, extras))
            produced += [dest, readme]
    info["bank_notes"] = check_banks(cdir, b)
    n, total, ev_paths = copy_evals(cdir, b, dry=dry or check)
    info["evals"], info["evals_bytes"] = n, total
    produced += ev_paths
    if init and not (check or dry):
        (cdir / "bundle.yaml").write_text(yaml.safe_dump(b, sort_keys=False, allow_unicode=True))
    if b.get("paths_note", True) and not (check or dry):
        rec = primary_record(cdir, b)
        if rec is not None and PATHS_NOTE_MARK not in rec.read_text():
            rec.write_text(paths_note(cdir, b) + rec.read_text())
            produced.append(rec)
    if not (check or dry):
        ignored = git_ignored(produced)
        if ignored:
            raise SystemExit(f"{cdir.name}: produced files are git-ignored: {ignored[:3]}")
        stray = [p for p in produced if p.suffix in STRAY and not (p.parent.name == "head" and p.suffix == ".pt")]
        if stray:
            raise SystemExit(f"{cdir.name}: refusing to stage stray artifacts: {stray[:3]}")
        git_add(produced + [cdir / "bundle.yaml"])
    return info


def load_all() -> list[tuple[Path, dict]]:
    out = []
    for mf in sorted(EXP.glob("*/bundle.yaml")):
        out.append((mf.parent, yaml.safe_load(mf.read_text()) or {}))
    return out


def index_text() -> str:
    rows = []
    aliases = []
    tracked_heads = 0
    for cdir, b in load_all():
        head = b.get("head") or {}
        if head.get("dest"):
            hp = cdir / head["dest"]
            if hp.exists():
                sha = sha256_of(hp)[:8]
                hcell = f"[`{Path(head['dest']).name}`]({cdir.name}/{head['dest']}) ({hp.stat().st_size / 1e6:.1f} MB, {sha})"
                tracked_heads += 1
            else:
                hcell = f"`{head['dest']}` (not bundled yet)"
        elif head.get("ref"):
            hcell = f"→ [{head['ref']}]({head['ref']}/)"
        else:
            hcell = f"none — {head.get('none', '')}"
        n_evals = len({p.parent for name in ("run_meta.json", "summary.md") for p in cdir.rglob(name)})
        recs = ", ".join(f"[{r}]({cdir.name}/{r})" for r in (b.get("records") or [])[:4])
        rows.append(f"| [{cdir.name}]({cdir.name}/) | {b.get('dates', '')} | {b.get('era', '')} | {b.get('plant', '')} | "
                    f"{b.get('question', '')} | {b.get('verdict', '')} | {hcell} | {n_evals} | {recs} |")
        for a in b.get("aliases") or []:
            aliases.append(f"| `{a}` | `{cdir.name}` |")
    rel = EXP.relative_to(REPO)
    txt = [
        "# Experiments — crab hexapod omnidirectional-walk task",
        "",
        f"GENERATED by `{rel}/tools/bundle_experiment.py --index` — edit the `bundle.yaml` manifests, not this file.",
        "",
        "Every campaign we ran for the omnidirectional-walk task lives here **whole** (dated directories, names unchanged;",
        "the raw training/eval artifacts stay on disk, git-ignored via `experiments/.gitignore`). Git tracks only the",
        "**keep-set** per campaign: the human records (REPORT / CHANGELOG / CHARTER / RESULTS / PLAN, state.json, csv",
        "tables, driver scripts), the evaluation summaries (`run_meta.json`, `scenario_metrics.json`, `summary.md`,",
        "gait diagrams — in place for era-A campaigns, copied under `evals/` for era-B campaigns whose evals ran in",
        "`parkour/logs/`), the reference RSI banks (`banks:`, tracked at the campaign top level), and **one checkpoint",
        "of record** per campaign under `<campaign>/head/` (sha-verified copy + provenance README). Campaigns that",
        "never baked a head carry a `none` entry or a pointer to the campaign whose head they used.",
        "",
        "**Forward/sideways lineage:** reference numbers and the shared MDP live in `../../crab_hex_forward_task/` "
        "(`policy/manifest.yaml`, README) and `../../crab_hex_sideways_task/`.",
        "",
        "**Not campaigns:** [`eval/`](eval/) holds the hand-written eval manifests -- `scenarios_omni.yaml` (v2 "
        "world-frame rows, the campaign suite), `scenarios_omni_body.yaml` (v3 robot-frame rows routed through the "
        "command term, for a body-frame head) and `scenarios_gpu0_sidesweep.yaml` (v2 report-only side-speed sweep) "
        "-- and `tools/` the drivers (`tools/pipeline/` = the Task 4 pipeline); neither carries a `bundle.yaml`.",
        "",
        "| campaign | dates | era | plant | question | outcome | head of record | eval runs | records |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    if aliases:
        txt += ["## Directory aliases in older records", "", "| old name | campaign |", "|---|---|", *aliases, ""]
    txt += [f"_{tracked_heads} bundled heads; regenerate with `--index` after editing manifests._", ""]
    return "\n".join(txt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaigns", nargs="*", help="campaign directories (default: none; use --all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify only (sha256 of heads/banks); no writes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing head/ copy with a different sha")
    ap.add_argument("--init", action="store_true", help="record missing sha256 values into bundle.yaml")
    ap.add_argument("--index", action="store_true", help="(re)generate experiments/README.md")
    args = ap.parse_args()
    dirs = [Path(c).resolve() for c in args.campaigns]
    if args.all:
        dirs = [d for d, _ in load_all()]
    total_heads = total_evals = 0
    for cdir in dirs:
        info = bundle(cdir, check=args.check, dry=args.dry_run, force=args.force, init=args.init)
        total_heads += info["heads_bytes"]
        total_evals += info["evals_bytes"]
        print(f"{cdir.name}: head {info['head'] or '-'} | evals {info['evals']} files ({info['evals_bytes'] / 1e6:.1f} MB)")
        for note in info.get("bank_notes") or []:
            print(note)
    if dirs:
        print(f"total: heads {total_heads / 1e6:.1f} MB, eval summaries {total_evals / 1e6:.1f} MB")
    if args.index:
        (EXP / "README.md").write_text(index_text())
        git_add([EXP / "README.md"])
        print(f"wrote {EXP.relative_to(REPO)}/README.md")
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
