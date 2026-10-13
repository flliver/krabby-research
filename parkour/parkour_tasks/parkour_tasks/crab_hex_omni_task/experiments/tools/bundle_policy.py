#!/usr/bin/env python3
"""Sync / check the crab-hex OMNI-walk POLICY OF RECORD: ``<task>/policy/``.

``policy/`` holds the head that ships for the omnidirectional task plus the stage heads it was
trained through (in manifest order; the side-first B lineage 1_fb6k -> c4 -> c5 -> c6 -> c6x clock-free,
baked 2026-10-06 as the phase-1 head, then the phase-2 wall-field stage 6_p2a_field, baked 2026-10-08 as the
head that ships), one subfolder per stage,
declared in ``policy/manifest.yaml`` (source run, sha256, campaign, overlay reference, per-category
suite evals, observation contract, open follow-ups), plus the shared training assets the lineage
depends on (``assets:`` -- the omni RSI reference bank every omni preset seeds resets from), plus -- optionally --
the STAND NETWORK OF RECORD (``stand_network:``, adopted 2026-10-13: A-mask s17 of
``experiments/2026-10-11_stand_network``). It changes ONLY on a bake / adoption decision, which is the user's: edit
the manifest, run ``--sync``, commit.

    python3 experiments/tools/bundle_policy.py --sync     # copy missing stage / asset / stand files (sha-verified),
                                                          #   verify present ones, regenerate README.md,
                                                          #   git add the produced files
    python3 experiments/tools/bundle_policy.py --check    # verify every stage file + sha + README (test hook)

The ``stand_network:`` block is neither a stage (nothing resumes from it; no preset / overlay) nor a training asset:
it is a second policy the controller routes on stand holds beside a walking stage (``walker.stage``, whose sha256 the
block must repeat), under a routing contract (``deployment:``) recorded exactly as the campaign evaluated it. Its one
file lives in its own folder (``dir``, which must not collide with a stage or asset dir); ``--sync`` copies it from
``source`` sha-verified and ``git add``s it, ``--check`` verifies it like a stage (present, sha, no stray checkpoint
in its folder), ``expected_files`` lists it, ``heads:`` may name its dir, and the README gains a "Stand network of
record" section (what it is, file + sha, the deployment contract and env, the eval / video commands, the per-seed
gate table, the selection and the caveats) plus one header line. A manifest WITHOUT the block behaves exactly as
before (README byte-identical).

Adapted from ``crab_hex_forward_task/experiments/tools/bundle_policy.py``. Differences:

* every stage is a ``KRABBY_OMNI_PHASE`` preset stage (the stage's ``phase``) plus a per-stage OVERLAY from a
  campaign's ``candidates.yaml`` (``stage`` + ``overlay_ref``) -- there are no omni presets for these
  overlays, so there is nothing to pin: no ``--pin-mdp``, no ``mdp_pins.yaml``, no
  ``POLICY_SUMMARY.md`` (the forward tool's generated-summary hooks are dropped);
* evals are the per-category suite (``suite__omni``: survival per category, speed ratio on the
  travel categories, facing_closed / yaw_progress on turn), not slow / step / obst; a stage may also
  carry obstacle rows (``evals.rows``: schedule_completion_rate of a campaign's 8-hold rows), rendered
  as their own table;
* the manifest carries an ``obs_contract:`` (the clock columns and the eval-carry semantics the heads
  trained under; per-stage ``KRABBY_OMNI_CLOCK_HIDE``) and ``followups:``, both rendered in the README.

Stdlib + PyYAML; no Isaac import.
"""

from __future__ import annotations

import argparse
import hashlib
import re
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
#: Keys every ``stand_network:`` block must carry (the rest -- architecture, evals, selection, caveats ... -- render
#: when present).
STAND_KEYS = ("dir", "file", "source", "sha256", "adopted", "arm", "stand_mask", "walker", "deployment")
#: The stand-network readers the README's evaluate block names, with their arguments ({name} = the re-run's head name).
STAND_READER_ARGS = {
    "sn_bridge_read.py": "--head {name} --live",
    "sn_calib.py": "--head {name} --json $OUT/head_{name}.json --md $OUT/head_{name}.md",
    "sn_observe_read.py": "--head {name} --json $OUT/observe_{name}.json --md $OUT/observe_{name}.md",
}


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
    if m.get("stand_network"):
        files.add(f"{m['stand_network']['dir']}/{m['stand_network']['file']}")
    files |= {"manifest.yaml", *GENERATED_DOCS}
    return files


def load_manifest(path: Path = MANIFEST) -> dict:
    """The manifest at ``path`` (default ``policy/manifest.yaml``), validated; SystemExit names the first problem."""
    m = yaml.safe_load(Path(path).read_text())
    for k in ("task", "plant", "current", "stages", "obs_contract"):
        if k not in m:
            raise SystemExit(f"{path}: missing key {k!r}")
    dirs = [s["dir"] for s in m["stages"]]
    if len(set(dirs)) != len(dirs):
        raise SystemExit(f"{path}: duplicate stage dirs")
    if m["current"] not in dirs:
        raise SystemExit(f"{path}: current={m['current']!r} is not a stage dir")
    for s in m["stages"]:
        for k in ("dir", "phase", "stage", "task", "iterations", "source", "file", "sha256", "overlay_ref",
                  "obs_contract"):
            if k not in s:
                raise SystemExit(f"{path}: stage {s.get('dir')!r} missing {k!r}")
        if str(s["obs_contract"].get(HIDE_VAR)) not in ("0", "1"):
            raise SystemExit(f"{path}: stage {s['dir']!r} obs_contract.{HIDE_VAR} must be '0' or '1'")
    oc = m["obs_contract"]
    for k in ("width", "clock_columns", "eval_env"):
        if k not in oc:
            raise SystemExit(f"{path}: obs_contract missing {k!r}")
    if HIDE_VAR in (oc["eval_env"].get("carry") or {}) or HIDE_VAR in (oc["eval_env"].get("fixed") or {}):
        raise SystemExit(f"{path}: {HIDE_VAR} is per stage (stage obs_contract), not in obs_contract.eval_env")
    m.setdefault("assets", [])
    for a in m["assets"]:
        for k in ("dir", "file", "source", "sha256"):
            if k not in a:
                raise SystemExit(f"{path}: asset {a.get('file')!r} missing {k!r}")
        if a["dir"] in dirs:
            raise SystemExit(f"{path}: asset dir {a['dir']!r} collides with a stage dir")
    m.setdefault("followups", [])
    for f in m["followups"]:
        for k in ("id", "headline"):
            if k not in f:
                raise SystemExit(f"{path}: followup {f.get('id')!r} missing {k!r}")
    sn = m.get("stand_network")
    if sn is not None:
        _validate_stand(m, sn, dirs, path)
    for k, d in (m.get("heads") or {}).items():
        if d not in dirs and not (sn and d == sn["dir"]):
            raise SystemExit(f"{path}: heads.{k}={d!r} is neither a stage dir nor the stand_network dir")
    return m


def _validate_stand(m: dict, sn, dirs: list[str], path) -> None:
    """The optional ``stand_network:`` block: required keys, a full sha256, an int 0 / 1 mask, its own folder (no
    stage / asset dir), and a ``walker`` that is a stage of this manifest with exactly that stage's sha256."""
    if not isinstance(sn, dict):
        raise SystemExit(f"{path}: stand_network must be a mapping")
    for k in STAND_KEYS:
        if k not in sn:
            raise SystemExit(f"{path}: stand_network missing {k!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(sn["sha256"])):
        raise SystemExit(f"{path}: stand_network.sha256 must be a full lowercase sha256, got {sn['sha256']!r}")
    if sn["dir"] in dirs:
        raise SystemExit(f"{path}: stand_network dir {sn['dir']!r} collides with a stage dir")
    if sn["dir"] in {a["dir"] for a in m["assets"]}:
        raise SystemExit(f"{path}: stand_network dir {sn['dir']!r} collides with an asset dir")
    if type(sn["stand_mask"]) is not int or sn["stand_mask"] not in (0, 1):
        raise SystemExit(f"{path}: stand_network.stand_mask must be the int 0 or 1, got {sn['stand_mask']!r}")
    w = sn["walker"]
    if not isinstance(w, dict) or w.get("stage") not in dirs:
        raise SystemExit(f"{path}: stand_network.walker.stage must name a stage dir, got {w!r}")
    if w.get("sha256") != stage_by_dir(m, w["stage"])["sha256"]:
        raise SystemExit(f"{path}: stand_network.walker.sha256 is not stage {w['stage']!r}'s sha256 (the stand net "
                         "was trained and evaluated beside another walker)")


def stage_by_dir(m: dict, d: str) -> dict:
    return next(s for s in m["stages"] if s["dir"] == d)


def stage_path(s: dict) -> Path:
    return POLICY / s["dir"] / s["file"]


def asset_path(a: dict) -> Path:
    return POLICY / a["dir"] / a["file"]


def stand_path(sn: dict) -> Path:
    return POLICY / sn["dir"] / sn["file"]


def _head_file(m: dict, d: str) -> str:
    """The file a ``heads:`` entry names: its stage's, or the stand network's when ``d`` is that block's dir."""
    sn = m.get("stand_network")
    if sn and d == sn["dir"]:
        return sn["file"]
    return stage_by_dir(m, d)["file"]


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
    sn = m.get("stand_network")
    if sn:
        p = stand_path(sn)
        if not p.exists():
            problems.append(f"{p.relative_to(REPO)}: missing")
        else:
            got = sha256_of(p)
            if got != sn["sha256"]:
                problems.append(f"{p.relative_to(REPO)}: sha256 {got[:12]} != manifest {sn['sha256'][:12]}")
            extra = [q for q in p.parent.glob("*.pt") if q.name != sn["file"]]
            if extra:
                problems.append(f"{p.parent.relative_to(REPO)}: stray checkpoints {[q.name for q in extra]}")
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
    sn = m.get("stand_network")
    if sn:
        dest = stand_path(sn)
        _sync_one(dest, REPO / sn["source"], sn["sha256"], f"stand_network {sn['dir']}", force=force)
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


# ------------------------------------------------------------------------------ the stand network's README section
def _gv(v) -> str:
    """A stand-gate value: — for None; ints and strings as is; 0 as 0; below 1e-3 as 1e-05; else up to four
    decimals, never fewer than one (1.0, 0.8125, 0.0068)."""
    if v is None:
        return "—"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) < 1e-3:
        return f"{v:.0e}"
    s = f"{float(v):.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _gate_cell(v) -> str:
    """A gate-table cell: ``es1 / es2`` for a list, the value otherwise; pipes escaped for the markdown table."""
    s = " / ".join(_gv(x) for x in v) if isinstance(v, list) else _gv(v)
    return s.replace("|", "\\|")


def _md(s) -> str:
    return str(s).replace("|", "\\|")


def _flat(v) -> str:
    """A manifest value on one line: a list joined by ``; ``, a mapping as ``k v`` pairs, else the string."""
    if isinstance(v, list):
        return "; ".join(_flat(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k} {_flat(x)}" for k, x in v.items())
    return str(v)


#: deployment keys rendered as labelled bullets, in this order (env_required / env_off / caveat_* are rendered apart)
STAND_DEPLOY_LABELS = (("switch", "Switch"), ("entry_exit", "Entry / exit"), ("input", "Stand-net input"),
                       ("forward", "Forward"), ("reference_implementation", "Reference implementation"))


def _stand_section(m: dict) -> list[str]:
    """README lines for the ``stand_network:`` block (rendered only when the block exists)."""
    sn = m["stand_network"]
    w = stage_by_dir(m, sn["walker"]["stage"])
    path = f"{sn['dir']}/{sn['file']}"
    wpath = f"{w['dir']}/{w['file']}"
    dep = sn["deployment"]
    L = [
        "## Stand network of record",
        "",
        f"**[`{path}`]({path})** -- {sn.get('what', '')}",
        "",
        f"**Adopted:** {sn['adopted']}",
        "",
        "It is **not** a lineage stage and **not** a training asset: nothing resumes from it and no preset builds it. "
        f"It is a second policy that the controller routes on **stand holds** beside the walking head [`{wpath}`]"
        f"({wpath}), which acts on every other hold and is unchanged by this adoption. Played or exported alone, the "
        "walking head still marches in place at stand (follow-up `stand_march`).",
        "",
        f"- **Run:** `{sn.get('id', '—')}` -- arm {sn['arm']} (`stand_mask` {sn['stand_mask']}), training seed "
        f"{sn.get('training_seed', '—')}, {sn.get('iterations', '—')} iterations",
        f"- **Trained in:** {sn.get('trained_in', '—')}",
        f"- **Walker:** `{w['dir']}/` (sha256 `{sn['walker']['sha256']}`) -- {sn['walker'].get('note', '')}",
        f"- **Source run (not tracked):** `{sn['source']}`",
        f"- **sha256:** `{sn['sha256']}`",
    ]
    arch = sn.get("architecture") or {}
    if arch:
        L.append("- **Architecture:**")
        L += [f"  - **{k}:** {_flat(v)}" for k, v in arch.items()]
    L += ["", "### Deployment contract (as evaluated)", ""]
    known = {k for k, _ in STAND_DEPLOY_LABELS} | {"env_required", "env_off"}
    for k, label in STAND_DEPLOY_LABELS:
        if dep.get(k):
            L.append(f"- **{label}:** {dep[k]}")
    for k, v in dep.items():
        if k not in known and not k.startswith("caveat"):
            L.append(f"- **{k}:** {_flat(v)}")
    hide = str(w["obs_contract"][HIDE_VAR])
    req = {k: str(v) for k, v in (dep.get("env_required") or {}).items() if not (k == HIDE_VAR and str(v) == hide)}
    env = (f"- **Environment (both networks, one process):** the walker's carry block (Observation contract above) "
           f"with `{HIDE_VAR}={hide}`")
    if req:
        env += "; in particular " + ", ".join(f"`{k}={v}`" for k, v in req.items())
    if dep.get("env_off"):
        env += "; unset (or `''` / `0`): " + ", ".join(f"`{k}`" for k in dep["env_off"]) + " and any `P2R_*` variable"
    L.append(env + ".")
    for k, v in dep.items():
        if k.startswith("caveat"):
            L.append(f"- **CAVEAT ({k.removeprefix('caveat').strip('_').replace('_', ' ') or 'deployment'}):** {v}")
    run = sn.get("evaluate") or {}
    if run:
        name, mask, rows = run.get("name", "NAME"), sn["stand_mask"], run.get("rows") or []
        L += ["", "### Evaluate it / watch it", "",
              "Re-run the gate rows against the bundled file, read them on the CPU, or record a video of the routed "
              "pair (from `krabby-research/parkour/`; one Isaac process at a time):", "",
              "```bash",
              "P=parkour_tasks/parkour_tasks/crab_hex_omni_task",
              f"S=$PWD/$P/{run['scripts']}",
              f"CK=$PWD/$P/policy/{path}",
              f"PY={run.get('python', 'python')}",
              "OUT=$(mktemp -d)   # where the CPU reads write"]
        if run.get("eval_rows"):
            L += [f"bash $S/{run['eval_rows']} --dry-run {name} $CK {mask} {rows[0] if rows else ''}"
                  "   # print the run_arm commands, no Isaac",
                  f"bash $S/{run['eval_rows']} {name} $CK {mask} \\",
                  "  " + " ".join(rows)]
        for r in run.get("readers") or []:
            L.append(f"$PY $S/{r} " + STAND_READER_ARGS.get(r, "--head {name}").format(name=name))
        if run.get("video"):
            L += [f"bash $S/{run['video']} --dry-run --prefix {name}_video -- --stand-ckpt $CK --stand-mask {mask}"
                  "   # schedule + command only",
                  f"bash $S/{run['video']} --prefix {name}_video -- --stand-ckpt $CK --stand-mask {mask}"
                  "   # 180 s, detached, MemoryMax 24G"]
        L += ["```", ""]
        if run.get("notes"):
            L += [run["notes"], ""]
    else:
        L.append("")
    ev = sn.get("evals") or {}
    gt = ev.get("gate_table") or {}
    cols = gt.get("columns") or []
    if gt.get("rows"):
        L += ["### Gates G1-G5 per training seed", ""]
        if ev.get("what"):
            L += [ev["what"], ""]
        L += ["| Gate | Statistic | Line | " + " | ".join(_md(c["label"]) for c in cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
        for r in gt["rows"]:
            L.append(f"| {r['gate']} | {_md(r['stat'])} | {_md(r['line'])} | "
                     + " | ".join(_gate_cell(r.get(c["id"])) for c in cols) + " |")
        if gt.get("verdicts"):
            L.append("| **Verdict** | | | " + " | ".join(_md(gt["verdicts"].get(c["id"], "—")) for c in cols) + " |")
        L.append("")
        if ev.get("references"):
            L += ["References: " + "; ".join(f"**{k}** {v}" for k, v in ev["references"].items()) + ".", ""]
    if ev.get("heads"):
        L += ["Per head (eval stamps, not tracked, under `" + str(ev.get("stamps_root", "—")) + "`):", ""]
        for h in ev["heads"]:
            tag = " (**adopted**)" if h.get("adopted") else ""
            line = (f"- **`{h['id']}`**{tag} -- sha256 `{h.get('sha256', '—')}`, `{h.get('source', '—')}`; "
                    f"{h.get('gates', '—')}; field_hard line {h.get('field_hard_line', '—')}")
            if h.get("worst_margin"):
                line += "; worst G2 margins " + ", ".join(f"{k} {v}" for k, v in h["worst_margin"].items())
            line += f"; read `{h.get('read', '—')}`"
            if h.get("observe"):
                line += f", observe `{h['observe']}`"
            L.append(line)
            if h.get("detail"):
                L.append(f"  - **Detail:** {h['detail']}")
            if h.get("train_time"):
                L.append(f"  - **Train-time:** {h['train_time']}")
            if h.get("stamps"):
                L.append("  - **Stamps (not tracked):** " + ", ".join(f"{k} `{v}`" for k, v in h["stamps"].items()))
            elif h.get("stamps_root"):
                L.append(f"  - **Stamps (not tracked):** `{h['stamps_root']}`")
        for h in ev.get("not_adopted") or []:
            L.append(f"- _`{h['id']}` (not a candidate; sha256 {h.get('sha256', '—')}, `stand_mask` "
                     f"{h.get('stand_mask', '—')})_ -- {h.get('summary', '')}")
        L.append("")
    sel = sn.get("selection") or {}
    if sel:
        L += ["### Selection", ""]
        for k, label in (("rule", "Rule"), ("tie_break", "Tie-break"), ("user_decision", "User decision")):
            if sel.get(k):
                L.append(f"- **{label}:** {sel[k]}")
        L.append("")
    if sn.get("caveats"):
        L += ["### Caveats", ""]
        L += [f"- {c}" for c in sn["caveats"]]
        L.append("")
    return L


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
        "This folder holds the head that **ships** for the omnidirectional walk task and the stage heads it",
        f"was trained through, one subfolder per stage. Plant: **{m['plant']}** (the main asset `assets/crab.usda`;",
        f"nothing to set). Seed {m.get('seed', '—')}. It changes only on a bake decision, which is the user's; the",
        "campaigns that produced these heads keep their own records under `experiments/<campaign>/`.",
        "",
        f"**Current head:** [`{cur_path}`]({cur_path}) -- {cur.get('what', '')}",
        "",
    ]
    sn = m.get("stand_network")
    if sn:
        sn_path = f"{sn['dir']}/{sn['file']}"
        L += [f"**Stand network of record:** [`{sn_path}`]({sn_path}) -- a second policy (not a stage) routed on stand "
              f"holds beside `{sn['walker']['stage']}/`; contract, commands and gates in "
              "[Stand network of record](#stand-network-of-record).", ""]
    if m.get("heads"):
        L += ["**Head of record per phase:** " + ", ".join(
            f"{k} [`{d}/`]({d}/) (`{_head_file(m, d)}`)" for k, d in m["heads"].items()) + ".", ""]
    L += [
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
    if sn:
        L += _stand_section(m)
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
    row_stages = [s for s in m["stages"] if (s.get("evals") or {}).get("rows")]
    if row_stages:
        L += ["Obstacle rows (`evals.rows`: `schedule_completion_rate` of the campaign's 8-hold rows, pooled over the "
              "listed env_seeds):", "",
              "| Stage | row | completion | n | per env_seed | what |",
              "|---|---|---|---|---|---|"]
        for s in row_stages:
            for r, e in s["evals"]["rows"].items():
                per = " / ".join(_f(v) for v in e.get("per_env_seed") or []) or "—"
                comp = e.get("completion")      # pooled n=400 reads are exact quarters (0.8125): 4 decimals
                if isinstance(comp, float):
                    comp = f"{comp:.4f}".rstrip("0")
                    comp = comp + "0" if comp.endswith(".") else comp
                else:
                    comp = _f(comp)
                L.append(f"| `{s['dir']}/` | `{r}` | {comp} | {e.get('n', '—')} | {per} | "
                         f"{e.get('what', '')} |")
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
        if s.get("construction_delta"):
            L.append("- **Construction delta** (trained at a value other than `construction:`): "
                     + ", ".join(f"`{k}={v}`" for k, v in s["construction_delta"].items()))
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
    ppo = (m.get("construction") or {}).get("ppo") or {}
    construction = (
        " Every overlay here also carries the runner-side construction knobs (manifest `construction:`: "
        + ", ".join(f"`{k}={v}`" for k, v in ppo.items())
        + "), except where a stage lists a `construction_delta`; a fork must keep them -- without "
        "`KRABBY_STD_PARAM` as trained the checkpoint does not load (strict load, `std` vs `std_raw`)." if ppo else "")
    tasks = list(dict.fromkeys(f"`{s['task']}` ({s['phase']})" for s in m["stages"]))
    L += [
        "## Resuming training from a stage head",
        "",
        "`KRABBY_OMNI_PHASE=<the next stage's phase>` (the `Preset + overlay` column) + the next stage's overlay "
        "(its `overlay_ref`, every key) `... train.py --task <the next stage's task> --resume --checkpoint "
        "<policy path>` (tasks by preset: " + ", ".join(tasks) + "), as the campaign drivers "
        "(`experiments/tools/run_candidates.py`, `checkpoint:` / `reference:` resume kinds) do." + construction
        + " A fork of the current head keeps "
        f"`{HIDE_VAR}=1` and the eps clock weights unless it deliberately restores the clock.",
        "",
        "## Maintenance",
        "",
        "- `python3 parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/bundle_policy.py --check` (from "
        "`parkour/`; the tool is cwd-independent) -- every stage file present with its manifest sha, no stray "
        "checkpoints, README fresh (run by `tests/unit/test_policy_of_record_omni.py`).",
        "- New bake: add/replace the stage entry in `manifest.yaml` (source run, sha256, campaign, overlay_ref, "
        "obs_contract, evals), set `current`, `git rm` the replaced stage files (`--sync` never deletes; `--check` "
        "flags them as stray checkpoints), run `--sync`, commit.",
        "- New shared asset (e.g. a re-harvested RSI bank): add an `assets:` entry (dir, file, source, sha256, what, "
        "used_by), run `--sync`, commit.",
    ]
    if sn:
        L.append(
            f"- Stand network (`stand_network:`, folder `{sn['dir']}/`): `--sync` copies its file from `source` "
            "sha-verified, `--check` verifies it like a stage (present, sha, no stray checkpoint). Required keys: "
            + ", ".join(f"`{k}`" for k in STAND_KEYS) + "; `walker` must name a stage and repeat that stage's sha256, "
            "so re-baking the walking head forces a decision on the stand network (it was trained and gated beside "
            "that exact walker). Replace it: edit the block, `git rm` the old file, run `--sync`, commit.")
    L += [
        "- No MDP pins: the omni lineage's stage recipes are overlays on a `KRABBY_OMNI_PHASE` preset (each "
        "stage's `phase`) recorded in campaign "
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
    sn = m.get("stand_network")
    if a.check:
        problems = check(m)
        for p in problems:
            print("PROBLEM", p)
        stand = f" + stand network {sn['dir']}" if sn else ""
        print(f"policy of record: {len(m['stages'])} stages{stand}, current {m['current']}: "
              f"{'OK' if not problems else 'FAILED'}")
        return 1 if problems else 0
    sync(m, force=a.force)
    total = (sum(stage_path(s).stat().st_size for s in m["stages"])
             + sum(asset_path(a).stat().st_size for a in m["assets"])
             + (stand_path(sn).stat().st_size if sn else 0)) / 1e6
    stand = " + stand network" if sn else ""
    print(f"synced {len(m['stages'])} stages + {len(m['assets'])} assets{stand} ({total:.1f} MB), current {m['current']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
