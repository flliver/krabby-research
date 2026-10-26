#!/usr/bin/env python3
"""Sync / check the crab-hex OMNI-walk POLICY OF RECORD: ``<task>/policy/``.

``policy/`` holds the head that ships for the omnidirectional task plus the stage heads it was
trained through (in manifest order), one subfolder per stage, declared in ``policy/manifest.yaml`` (source run,
sha256, campaign, overlay reference, per-category suite evals, observation contract, open follow-ups), plus the
shared training assets the lineage depends on (``assets:`` -- the omni RSI reference bank every omni preset seeds
resets from), plus -- optionally -- the STAND NETWORK OF RECORD (``stand_network:``), plus the files of SUPERSEDED
heads that were deliberately left in the folder (``superseded[].files``). History: the side-first B lineage at seed 18
(1_fb6k -> c4 -> c5 -> c6 -> c6x clock-free, baked 2026-10-06; + the phase-2 wall-field stage 6_p2a_field, baked
2026-10-08), the stand networks A-mask s17 (2026-10-13) and SH_T17 (2026-10-15); since 2026-10-24 the Task 4
deliverable run F19 (seed 19: s1 -> c4 -> c5 -> c6 -> c6x -> p2a -> p3a -> p3b = T*, the depth student distilled from
T* as a ``kind: distill`` stage, and the stand nets SH_T4F19 / SN_T4F19 beside T*) SUPERSEDES the seed-18 lineage and
SH_T17, whose files STAY in the folder (``superseded[].files``). It changes ONLY on a bake / adoption decision, which
is the user's: edit the manifest, run ``--sync``, commit.

    python3 experiments/tools/bundle_policy.py --sync     # copy missing stage / asset / stand files (sha-verified),
                                                          #   verify present ones, regenerate README.md,
                                                          #   git add the produced files
    python3 experiments/tools/bundle_policy.py --sync --force  # recopy every stage file from its source (sha-verified)
    python3 experiments/tools/bundle_policy.py --check   # verify every stage file + sha + README (test hook)

The ``stand_network:`` block is neither a stage (nothing resumes from it; no preset / overlay) nor a training asset:
it is a second policy the controller routes on stand holds beside a walking stage (``walker.stage``, whose sha256 the
block must repeat), under a routing contract (``deployment:``) recorded exactly as the campaign evaluated it. Its one
file lives in its own folder (``dir``, which must not collide with a stage or asset dir); ``--sync`` copies it from
``source`` sha-verified and ``git add``s it, ``--check`` verifies it like a stage (present, sha, no stray checkpoint
in its folder), ``expected_files`` lists it, ``heads:`` may name its dir, and the README gains a "Stand network of
record" section (what it is, file + sha, training, the deployment contract and env -- a mapping-valued contract key
such as ``deployment.constants`` renders as nested bullets --, the eval / video commands (``evaluate.commands``), the
per-head gate table, the adoption-grade note, the selection and the caveats) plus one header line. A manifest WITHOUT
the block behaves exactly as before (README byte-identical).

The block may carry a ``warm_start:`` sub-block (``id``, ``file``, ``source``, ``sha256``, ``role``): the checkpoint the
stand network was warm-started from, bundled in the SAME folder so the adopted head stays reproducible and the
campaign trainer that names that path still runs (2026-10-15: A-mask s17 ``model_4000.pt``, SH_T17's ancestor -- the
precedent is the rsi asset, kept because the shipped recipe still names it). It is NOT the stand network of record and
is never routed by the deployment contract. Its file is handled exactly like the stand file: ``--sync`` copies it from
``source`` sha-verified and ``git add``s it, ``--check`` verifies it (present, sha; it is not a stray checkpoint),
``expected_files`` lists it, and the README's stand section renders it as one "Warm-start ancestor" bullet.

A stage's ``kind`` is ``rl`` (the default) or ``distill`` (a student distilled from the previous stage; the forward
task's ``3a_student`` is the precedent). A ``distill`` stage may declare ``bundled: false``: its checkpoint stays OUT of
``policy/`` (2026-10-24: the F19 student .pt is 122 MB, over GitHub's 100 MiB per-file limit) and is cited by
``source`` + ``sha256`` only; its folder then holds the stage's ``exports:`` (each ``file`` / ``source`` / ``sha256`` /
``what``; the ONNX deploy export + its io json), synced sha-verified and checked like a stage file, and ``--check``
flags ANY checkpoint (*.pt) in that folder. Every file ``--sync`` copies and ``--check`` verifies is held to
:data:`MAX_FILE_BYTES` (GitHub's hard per-file limit): a larger file is refused, never bundled.

``superseded:`` is data except one key: an entry's ``files:`` (each ``dir`` / ``file`` / ``sha256`` / ``source``) names
files of a superseded head that were deliberately LEFT in ``policy/`` because other sessions, tests or scripts still
use their paths (2026-10-24: the seed-18 lineage 1_fb6k .. 6_p2a_field and stand_net/). They are expected files
(``expected_files``), ``--check`` verifies them (present, sha, no stray checkpoint in their folders), ``--sync``
verifies them and NEVER copies, adds or deletes them, and the README lists them under "Superseded heads kept in this
folder". A kept file must live in a folder of its own (not a live stage / asset / stand dir) unless it IS a live file
with the same sha (one file recorded twice, e.g. a stand net's warm-start ancestor). An entry without ``files:`` (W:
its files left the folder at the 2026-10-06 bake) changes nothing.

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
import json
import re
import shlex
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
#: Keys every ``stand_network:`` block must carry (the rest -- training, architecture, evals, selection, caveats ... --
#: render when present).
STAND_KEYS = ("dir", "file", "source", "sha256", "adopted", "arm", "stand_mask", "walker", "deployment")
#: ``deployment.constants`` (optional; the height-commanded contract's numbers): the walking / min / max heights,
#: which must be ordered, and the residual deadband, which must be a number in [0, 1).
STAND_HEIGHT_KEYS = ("h_min_m", "h_settled_m", "h_max_m")
#: Keys an optional ``stand_network.warm_start:`` sub-block must carry (its file lives in the stand network's folder).
STAND_WARM_KEYS = ("id", "file", "source", "sha256", "role")
#: Stage kinds: ``rl`` (PPO on a preset + overlay; the default when ``kind`` is absent) and ``distill`` (a student
#: distilled from the previous stage, the forward task's ``3a_student`` precedent).
STAGE_KINDS = ("rl", "distill")
#: Keys every ``exports:`` entry of a distill stage carries (the files live in the stage's own folder).
EXPORT_KEYS = ("file", "source", "sha256", "what")
#: Keys every ``superseded[].files`` entry (a superseded file KEPT in ``policy/``) carries.
KEPT_KEYS = ("dir", "file", "sha256", "source")
#: GitHub refuses any file over 100 MiB: nothing larger is ever bundled (``--sync`` refuses, ``--check`` flags).
MAX_FILE_BYTES = 100 * 1024 * 1024


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_files(m: dict) -> set[str]:
    """Every file that may live under policy/, relative to policy/ (the layout test pins this set): each stage's files
    (its checkpoint unless ``bundled: false``, and its ``exports``), the assets, the stand network's files, and the
    superseded files deliberately kept in the folder (``superseded[].files``)."""
    files = {f"{s['dir']}/{f['file']}" for s in m["stages"] for f in stage_files(s)}
    files |= {f"{a['dir']}/{a['file']}" for a in m.get("assets", [])}
    if m.get("stand_network"):
        files |= {f"{m['stand_network']['dir']}/{f['file']}" for f in stand_files(m["stand_network"])}
    files |= {f"{k['dir']}/{k['file']}" for k in kept_files(m)}
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
        _validate_stage_kind(s, path)
    if stage_by_dir(m, m["current"]).get("bundled", True) is False:
        raise SystemExit(f"{path}: current={m['current']!r} is a stage whose checkpoint is not bundled")
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
    _validate_superseded(m, dirs, path)
    return m


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_stage_kind(s: dict, path) -> None:
    """``kind`` (default ``rl``) is a known stage kind; ``bundled: false`` and ``exports`` belong to ``distill`` stages
    only; a non-bundled stage must bundle its exports (one folder never holds nothing) and they are not checkpoints."""
    kind = s.get("kind", "rl")
    if kind not in STAGE_KINDS:
        raise SystemExit(f"{path}: stage {s['dir']!r} kind {kind!r} is not one of {STAGE_KINDS}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(s["sha256"])):
        raise SystemExit(f"{path}: stage {s['dir']!r} sha256 must be a full lowercase sha256, got {s['sha256']!r}")
    bundled = s.get("bundled", True)
    if bundled not in (True, False):
        raise SystemExit(f"{path}: stage {s['dir']!r} bundled must be true or false, got {bundled!r}")
    exports = s.get("exports")
    if (exports is not None or bundled is False) and kind != "distill":
        raise SystemExit(f"{path}: stage {s['dir']!r}: bundled / exports are for kind 'distill' stages only")
    if exports is None:
        if bundled is False:
            raise SystemExit(f"{path}: stage {s['dir']!r} is not bundled and declares no exports")
        return
    if not isinstance(exports, list) or not exports:
        raise SystemExit(f"{path}: stage {s['dir']!r} exports must be a non-empty list")
    names = []
    for e in exports:
        if not isinstance(e, dict):
            raise SystemExit(f"{path}: stage {s['dir']!r} exports entries must be mappings")
        for k in EXPORT_KEYS:
            if k not in e:
                raise SystemExit(f"{path}: stage {s['dir']!r} export {e.get('file')!r} missing {k!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(e["sha256"])):
            raise SystemExit(f"{path}: stage {s['dir']!r} export {e['file']!r} sha256 must be a full lowercase sha256")
        if str(e["file"]).endswith(".pt") or "/" in str(e["file"]):
            raise SystemExit(f"{path}: stage {s['dir']!r} export {e['file']!r} must be a plain non-checkpoint file "
                             "name (the checkpoint itself is the stage `file`)")
        names.append(e["file"])
    if len(set(names)) != len(names) or s["file"] in names:
        raise SystemExit(f"{path}: stage {s['dir']!r} exports must be distinct files other than the stage file")


def _live_files(m: dict) -> dict[str, str]:
    """``{"dir/file": sha256}`` of every LIVE policy file (stages, assets, the stand network's folder)."""
    live = {f"{s['dir']}/{f['file']}": f["sha256"] for s in m["stages"] for f in stage_files(s)}
    live.update({f"{a['dir']}/{a['file']}": a["sha256"] for a in m.get("assets", [])})
    sn = m.get("stand_network")
    if sn:
        live.update({f"{sn['dir']}/{f['file']}": f["sha256"] for f in stand_files(sn)})
    return live


def _validate_superseded(m: dict, dirs: list[str], path) -> None:
    """``superseded:`` is a list of mappings with an ``id``; an entry's optional ``files`` (superseded files KEPT in
    policy/) are full-sha mappings, each path recorded once across entries, each either exactly a live file (same sha:
    one file recorded twice) or in a folder no live stage / asset / stand network uses."""
    sups = m.get("superseded")
    if sups is None:
        return
    if not isinstance(sups, list):
        raise SystemExit(f"{path}: superseded must be a list")
    live = _live_files(m)
    live_dirs = set(dirs) | {a["dir"] for a in m.get("assets", [])}
    if m.get("stand_network"):
        live_dirs.add(m["stand_network"]["dir"])
    seen = set()
    for e in sups:
        if not isinstance(e, dict) or "id" not in e:
            raise SystemExit(f"{path}: every superseded entry is a mapping with an id")
        files = e.get("files")
        if files is None:
            continue
        if not isinstance(files, list) or not files:
            raise SystemExit(f"{path}: superseded {e['id']!r} files must be a non-empty list")
        for f in files:
            if not isinstance(f, dict):
                raise SystemExit(f"{path}: superseded {e['id']!r} files entries must be mappings")
            for k in KEPT_KEYS:
                if k not in f:
                    raise SystemExit(f"{path}: superseded {e['id']!r} kept file {f.get('file')!r} missing {k!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(f["sha256"])):
                raise SystemExit(f"{path}: superseded {e['id']!r} kept file {f['file']!r} sha256 must be a full "
                                 "lowercase sha256")
            rel = f"{f['dir']}/{f['file']}"
            if rel in seen:
                raise SystemExit(f"{path}: superseded kept file {rel!r} is recorded twice")
            seen.add(rel)
            if rel in live:
                if live[rel] != f["sha256"]:
                    raise SystemExit(f"{path}: superseded {e['id']!r} kept file {rel!r} is a live policy file with "
                                     "another sha256")
            elif f["dir"] in live_dirs:
                raise SystemExit(f"{path}: superseded {e['id']!r} kept file {rel!r} sits in a live stage / asset / "
                                 "stand folder (give the new head its own folder)")


def _validate_stand(m: dict, sn, dirs: list[str], path) -> None:
    """The optional ``stand_network:`` block: required keys, a full sha256, an int 0 / 1 mask, its own folder (no
    stage / asset dir), a ``walker`` that is a stage of this manifest with exactly that stage's sha256, and -- when the
    contract carries ``deployment.constants`` (a height-commanded stand net) -- a mapping whose heights are ordered
    h_min < h_settled < h_max and whose ``residual_deadband`` is a number in [0, 1)."""
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
    dep = sn["deployment"]
    if not isinstance(dep, dict):
        raise SystemExit(f"{path}: stand_network.deployment must be a mapping")
    if "constants" in dep:
        c = dep["constants"]
        if not isinstance(c, dict):
            raise SystemExit(f"{path}: stand_network.deployment.constants must be a mapping")
        hs = [c.get(k) for k in STAND_HEIGHT_KEYS]
        if any(h is not None for h in hs) and not (all(_num(h) for h in hs) and hs[0] < hs[1] < hs[2]):
            raise SystemExit(f"{path}: stand_network.deployment.constants needs numeric "
                             f"{' < '.join(STAND_HEIGHT_KEYS)}, got {hs}")
        db = c.get("residual_deadband")
        if db is not None and not (_num(db) and 0 <= db < 1):
            raise SystemExit(f"{path}: stand_network.deployment.constants.residual_deadband must be a number in "
                             f"[0, 1), got {db!r}")
    if "warm_start" in sn:
        ws = sn["warm_start"]
        if not isinstance(ws, dict):
            raise SystemExit(f"{path}: stand_network.warm_start must be a mapping")
        for k in STAND_WARM_KEYS:
            if k not in ws:
                raise SystemExit(f"{path}: stand_network.warm_start missing {k!r}")
        if "dir" in ws and ws["dir"] != sn["dir"]:
            raise SystemExit(f"{path}: stand_network.warm_start lives in the stand network's folder {sn['dir']!r}, "
                             f"got dir {ws['dir']!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(ws["sha256"])):
            raise SystemExit(f"{path}: stand_network.warm_start.sha256 must be a full lowercase sha256, got "
                             f"{ws['sha256']!r}")
        if ws["file"] == sn["file"] or ws["sha256"] == sn["sha256"]:
            raise SystemExit(f"{path}: stand_network.warm_start must be a different file from the stand network "
                             f"({ws['file']!r})")


def stage_by_dir(m: dict, d: str) -> dict:
    return next(s for s in m["stages"] if s["dir"] == d)


def stage_path(s: dict) -> Path:
    return POLICY / s["dir"] / s["file"]


def is_bundled(s: dict) -> bool:
    """False for a ``distill`` stage that keeps its checkpoint out of policy/ (``bundled: false``)."""
    return s.get("bundled", True) is not False


def stage_files(s: dict) -> list[dict]:
    """The files stage ``s``'s folder holds, each ``{label, file, source, sha256, path}``: its checkpoint (unless
    ``bundled: false``), then its ``exports`` (synced and checked identically)."""
    out = []
    if is_bundled(s):
        out.append({"label": s["dir"], "file": s["file"], "source": s["source"], "sha256": s["sha256"],
                    "path": stage_path(s)})
    for e in s.get("exports") or []:
        out.append({"label": f"{s['dir']} export {e['file']}", "file": e["file"], "source": e["source"],
                    "sha256": e["sha256"], "path": POLICY / s["dir"] / e["file"]})
    return out


def kept_files(m: dict) -> list[dict]:
    """The superseded files deliberately KEPT in policy/ (``superseded[].files``), each with its entry ``id`` and
    ``path``, minus any that are also a live file (one file recorded twice is checked once, as the live file)."""
    live = _live_files(m)
    out = []
    for e in m.get("superseded") or []:
        for f in e.get("files") or []:
            if f"{f['dir']}/{f['file']}" in live:
                continue
            out.append({**f, "id": e["id"], "path": POLICY / f["dir"] / f["file"]})
    return out


def asset_path(a: dict) -> Path:
    return POLICY / a["dir"] / a["file"]


def stand_path(sn: dict) -> Path:
    return POLICY / sn["dir"] / sn["file"]


def warm_start_path(sn: dict) -> Path | None:
    """The bundled warm-start ancestor (``stand_network.warm_start``, same folder), or None without the sub-block."""
    ws = sn.get("warm_start")
    return POLICY / sn["dir"] / ws["file"] if ws else None


def stand_files(sn: dict) -> list[dict]:
    """The files the stand network's folder holds, each ``{label, file, source, sha256, path}``: the stand network of
    record, then its warm-start ancestor when the block names one (synced and checked identically)."""
    out = [{"label": f"stand_network {sn['dir']}", "file": sn["file"], "source": sn["source"], "sha256": sn["sha256"],
            "path": stand_path(sn)}]
    ws = sn.get("warm_start")
    if ws:
        out.append({"label": f"stand_network.warm_start {ws.get('id', ws['file'])}", "file": ws["file"],
                    "source": ws["source"], "sha256": ws["sha256"], "path": warm_start_path(sn)})
    return out


def _head_file(m: dict, d: str) -> str:
    """The file a ``heads:`` entry names: its stage's, or the stand network's when ``d`` is that block's dir; a stage
    whose checkpoint is not bundled names its exports instead."""
    sn = m.get("stand_network")
    if sn and d == sn["dir"]:
        return f"`{sn['file']}`"
    s = stage_by_dir(m, d)
    if not is_bundled(s):
        return (f"`{s['file']}` not bundled; folder holds " + " + ".join(f"`{e['file']}`" for e in s["exports"]))
    return f"`{s['file']}`"


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


def _check_file(p: Path, sha: str, problems: list[str]) -> None:
    """``p`` present, at most :data:`MAX_FILE_BYTES`, with the manifest sha256 (problems appended)."""
    if not p.exists():
        problems.append(f"{p.relative_to(REPO)}: missing")
        return
    if p.stat().st_size > MAX_FILE_BYTES:
        problems.append(f"{p.relative_to(REPO)}: {p.stat().st_size / 2**20:.0f} MiB > the 100 MiB per-file limit")
    got = sha256_of(p)
    if got != sha:
        problems.append(f"{p.relative_to(REPO)}: sha256 {got[:12]} != manifest {sha[:12]}")


def _stray(folder: Path, allowed: set[str], problems: list[str]) -> None:
    extra = sorted(q.name for q in folder.glob("*.pt") if q.name not in allowed)
    if extra:
        problems.append(f"{folder.relative_to(REPO)}: stray checkpoints {extra}")


def check(m: dict) -> list[str]:
    """Problems with the tracked policy files (empty list = ok)."""
    problems = []
    for s in m["stages"]:
        files = stage_files(s)
        for f in files:
            _check_file(f["path"], f["sha256"], problems)
        folder = POLICY / s["dir"]
        if not is_bundled(s) and stage_path(s).exists():
            problems.append(f"{stage_path(s).relative_to(REPO)}: stage {s['dir']!r} is `bundled: false` -- its "
                            "checkpoint must NOT be in policy/ (cite it by source + sha256)")
        if folder.is_dir():
            _stray(folder, {f["file"] for f in files}, problems)
    for a in m["assets"]:
        _check_file(asset_path(a), a["sha256"], problems)
    sn = m.get("stand_network")
    if sn:
        files = stand_files(sn)
        for f in files:
            _check_file(f["path"], f["sha256"], problems)
        _stray(stand_path(sn).parent, {f["file"] for f in files}, problems)
    kept = kept_files(m)
    for k in kept:
        _check_file(k["path"], k["sha256"], problems)
    for folder in sorted({k["path"].parent for k in kept}):
        if folder.is_dir():
            _stray(folder, {k["file"] for k in kept if k["path"].parent == folder}, problems)
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
    if src.stat().st_size > MAX_FILE_BYTES:
        raise SystemExit(f"{label}: source {src} is {src.stat().st_size / 2**20:.0f} MiB > the 100 MiB per-file "
                         "limit -- not bundled (a distill stage declares `bundled: false` + its exports instead)")
    got = sha256_of(src)
    if got != sha:
        raise SystemExit(f"{label}: source {src} sha256 {got[:12]} != manifest {sha[:12]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"copied  {dest.relative_to(REPO)} ({dest.stat().st_size / 1e6:.1f} MB)")


def _verify_kept(k: dict) -> None:
    """A superseded file kept in policy/ is verified, never (re)copied, added or removed by ``--sync``."""
    p = k["path"]
    if not p.exists():
        raise SystemExit(f"{p}: superseded {k['id']!r} kept file is missing (restore it from git; --sync never "
                         "recreates a superseded file)")
    got = sha256_of(p)
    if got != k["sha256"]:
        raise SystemExit(f"{p}: superseded {k['id']!r} kept file sha256 {got[:12]} != manifest {k['sha256'][:12]}")
    print(f"kept    {p.relative_to(REPO)} (superseded {k['id']})")


def sync(m: dict, *, force: bool) -> list[Path]:
    produced = []
    for s in m["stages"]:
        for f in stage_files(s):
            _sync_one(f["path"], REPO / f["source"], f["sha256"], f["label"], force=force)
            produced.append(f["path"])
        if not is_bundled(s):
            print(f"cited   {s['dir']}/{s['file']} (not bundled: {s['source']}, sha256 {s['sha256'][:12]})")
    for k in kept_files(m):
        _verify_kept(k)
    for a in m["assets"]:
        dest = asset_path(a)
        _sync_one(dest, REPO / a["source"], a["sha256"], a["file"], force=force)
        produced.append(dest)
    sn = m.get("stand_network")
    if sn:
        for f in stand_files(sn):
            _sync_one(f["path"], REPO / f["source"], f["sha256"], f["label"], force=force)
            produced.append(f["path"])
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


#: deployment keys rendered as labelled bullets, in this order (env_required / env_off / caveat_* are rendered apart;
#: any other key renders under its own name, a mapping as nested bullets)
STAND_DEPLOY_LABELS = (("switch", "Switch"), ("entry_exit", "Entry / exit"), ("input", "Stand-net input"),
                       ("forward", "Forward"), ("action", "Action"), ("posture_map", "Posture map q(h)"),
                       ("height_command", "Height command"), ("restore", "Restore before hand-back"),
                       ("zero_height", "Zero-height mode"), ("scope", "Scope"),
                       ("reference_implementation", "Reference implementation"))


def _nested(label: str, d: dict) -> list[str]:
    """``- **label:**`` and one ``  - **k:** v`` sub-bullet per entry of the mapping ``d``."""
    return [f"- **{label}:**"] + [f"  - **{k}:** {_flat(v)}" for k, v in d.items()]


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
        f"({wpath}), the walker it was trained and gated beside, which acts on every other hold. Played or exported "
        "alone, the walking head still marches in place at stand (follow-up `stand_march`).",
        "",
        f"- **Run:** `{sn.get('id', '—')}` -- arm {sn['arm']} (`stand_mask` {sn['stand_mask']}), training seed "
        f"{sn.get('training_seed', '—')}, {sn.get('iterations', '—')} iterations",
        f"- **Trained in:** {sn.get('trained_in', '—')}",
        f"- **Walker:** `{w['dir']}/` (sha256 `{sn['walker']['sha256']}`) -- {sn['walker'].get('note', '')}",
        f"- **Source run (not tracked):** `{sn['source']}`",
        f"- **sha256:** `{sn['sha256']}`",
    ]
    ws = sn.get("warm_start")
    if ws:
        wsp = f"{sn['dir']}/{ws['file']}"
        L.append(f"- **Warm-start ancestor (bundled, NOT the stand network of record):** [`{wsp}`]({wsp}) -- "
                 f"`{ws['id']}`, sha256 `{ws['sha256']}`, source run (not tracked) `{ws['source']}`; {ws['role']}"
                 + (f". {ws['used_by']}" if ws.get("used_by") else ""))
    for key, label in (("training", "Training"), ("architecture", "Architecture")):
        if sn.get(key):
            L += _nested(label, sn[key])
    L += ["", "### Deployment contract (as evaluated)", ""]
    known = {k for k, _ in STAND_DEPLOY_LABELS} | {"env_required", "env_off"}
    for k, label in STAND_DEPLOY_LABELS:
        if dep.get(k):
            L.append(f"- **{label}:** {dep[k]}")
    for k, v in dep.items():
        if k not in known and not k.startswith("caveat"):
            L += _nested(k, v) if isinstance(v, dict) else [f"- **{k}:** {_flat(v)}"]
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
        fmt = {"name": run.get("name", "NAME")}
        L += ["", "### Evaluate it / watch it", "",
              "Re-run the gate rows against the bundled file, read them on the CPU, or record a video of the routed "
              "pair (from `krabby-research/parkour/`; one Isaac process at a time):", "",
              "```bash",
              "P=parkour_tasks/parkour_tasks/crab_hex_omni_task",
              f"S=$PWD/$P/{run['scripts']}",
              f"CK=$PWD/$P/policy/{path}",
              f"PY={run.get('python', 'python')}",
              "OUT=$(mktemp -d)   # where the CPU reads write"]
        if run.get("env"):
            if run.get("env_comment"):
                L.append(f"# {run['env_comment']}")
            for k, v in run["env"].items():
                val = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
                L.append(f"export {k}={shlex.quote(val)}")
        for c in run.get("commands") or []:
            if c.get("comment"):
                L.append(f"# {c['comment'].format(**fmt)}")
            L += c["run"].format(**fmt).splitlines()
        L += ["```", ""]
        if run.get("notes"):
            L += [run["notes"].format(**fmt), ""]
    else:
        L.append("")
    ev = sn.get("evals") or {}
    gt = ev.get("gate_table") or {}
    cols = gt.get("columns") or []
    if gt.get("rows"):
        L += [f"### {ev.get('title', 'Gates per head')}", ""]
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
        if ev.get("adoption_grade"):
            L += [f"**Adoption grade:** {ev['adoption_grade']}", ""]
        if ev.get("references"):
            L += ["References: " + "; ".join(f"**{k}** {v}" for k, v in ev["references"].items()) + ".", ""]
    if ev.get("heads"):
        L += ["Per head (eval stamps, not tracked, under `" + str(ev.get("stamps_root", "—")) + "`):", ""]
        for h in ev["heads"]:
            tag = " (**adopted**)" if h.get("adopted") else ""
            line = (f"- **`{h['id']}`**{tag} -- sha256 `{h.get('sha256', '—')}`, `{h.get('source', '—')}`; "
                    f"{h.get('gates', '—')}")
            if h.get("field_hard_line") is not None:
                line += f"; field_hard line {h['field_hard_line']}"
            if h.get("worst_margin"):
                line += "; worst G2 margins " + ", ".join(f"{k} {v}" for k, v in h["worst_margin"].items())
            line += f"; read `{h.get('read', '—')}`"
            if h.get("observe"):
                obs = h["observe"] if isinstance(h["observe"], list) else [h["observe"]]
                line += ", observe " + ", ".join(f"`{o}`" for o in obs)
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


def _args(xs) -> str:
    """Extra CLI args as a leading-space string (empty when there are none)."""
    return "".join(f" {x}" for x in xs or [])


def _run_path(src: str) -> str:
    """A repo-relative run path as the README commands spell it from ``parkour/`` (``$PWD/...``)."""
    return "$PWD/" + src.removeprefix("parkour/") if src.startswith("parkour/") else "$PWD/../" + src


def _distill_section(m: dict, s: dict) -> list[str]:
    """README lines for a ``kind: distill`` stage: what it is, where its checkpoint lives (cited, not bundled, when
    ``bundled: false``), its bundled exports, and its play / score commands (its own task and args, the carry env)."""
    d = s["dir"]
    ck = f"$PWD/$P/policy/{d}/{s['file']}" if is_bundled(s) else _run_path(s["source"])
    play_task = s.get("play_task", s["task"])
    L = [f"## The {s.get('role', 'student')} -- [`{d}/`]({d}/) (kind `distill`)", "",
         f"{s.get('what', '')}", ""]
    if is_bundled(s):
        L.append(f"- **Checkpoint (bundled):** [`{d}/{s['file']}`]({d}/{s['file']}), sha256 `{s['sha256']}`")
    else:
        L.append(f"- **Checkpoint NOT bundled** (`bundled: false`): `{s['file']}`, sha256 `{s['sha256']}`, source run "
                 f"(not tracked) `{s['source']}` -- {s.get('not_bundled', '')}")
    for e in s.get("exports") or []:
        L.append(f"- **Bundled export:** [`{d}/{e['file']}`]({d}/{e['file']}) -- {e['what']}; sha256 `{e['sha256']}`, "
                 f"source (not tracked) `{e['source']}`")
    if s.get("distill"):
        L += _nested("Distillation", s["distill"])
    L += ["", f"Play it (from `krabby-research/parkour/`, Isaac venv python; `{play_task}`; the checkpoint is the "
          + ("bundled file" if is_bundled(s) else "sha-pinned run file, not a policy/ path") + "):", "",
          "```bash",
          "P=parkour_tasks/parkour_tasks/crab_hex_omni_task",
          "env \\",
          *_env_lines(play_env(m, s)),
          f"  python scripts/rsl_rl/play.py --task {play_task}{_args(s.get('play_args'))} \\",
          f"  --num_envs 1 --checkpoint {ck}",
          "```", "",
          f"Score it on the per-category suite as the selection rows did (`{SUITE_ROW}`):", "",
          "```bash",
          "P=parkour_tasks/parkour_tasks/crab_hex_omni_task",
          "env \\",
          *_env_lines(eval_env(m, s)),
          f"  python {EVAL_SCRIPT} --headless \\",
          f"  --manifest {OMNI_MANIFEST} --scenario {SUITE_ROW}{_args(s.get('eval_args'))} \\",
          f"  --checkpoint {ck} --save-raw --trained-categories {','.join(CATEGORIES)}",
          "```", ""]
    if s.get("play_notes"):
        L += [s["play_notes"], ""]
    return L


def _payload_table(m: dict) -> list[str]:
    """The payload stages (``payload:``): trained range, the stable-at-L curve, max_stable and its limiting factor."""
    rows = [s for s in m["stages"] if s.get("payload")]
    if not rows:
        return []
    L = ["Payload stages (`payload:`; every number is a sim-plant number, force side not enforced):", "",
         "| Stage | trained payload (kg) | stable at L (kg) | max_stable (kg) | first unstable (kg) | limiting factor | "
         "paired 0-kg check |",
         "|---|---|---|---|---|---|---|"]
    for s in rows:
        p = s["payload"]
        curve = ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in (p.get("curve") or {}).items()) or "—"
        L.append(f"| `{s['dir']}/` | `{(p.get('knobs') or {}).get('KRABBY_PAYLOAD_KG', '—')}` | {curve} | "
                 f"{p.get('max_stable_kg', '—')} | {p.get('first_unstable_kg', '—')} | "
                 f"{_md(p.get('limiting_factor', '—'))} | {_md(p.get('zero_kg_check', '—'))} |")
    return L + [""]


def _superseded_section(m: dict) -> list[str]:
    """The superseded heads whose files stay in this folder (``superseded[].files``)."""
    ents = [e for e in m.get("superseded") or [] if e.get("files")]
    if not ents:
        return []
    L = ["## Superseded heads kept in this folder", "",
         "NOT the policy of record: these files stay only because other sessions, tests or scripts still use their "
         "paths. `--check` verifies them (present, sha, no stray checkpoint); `--sync` never copies, adds or deletes "
         "them. Their full records are the manifest's `superseded:` entries.", ""]
    for e in ents:
        L.append(f"- **`{e['id']}`**" + (f" ({e['kind']})" if e.get("kind") else "") + f" -- superseded "
                 f"{e.get('superseded', '—')}")
        for f in e["files"]:
            p = f"{f['dir']}/{f['file']}"
            L.append(f"  - [`{p}`]({p}) -- sha256 `{f['sha256']}`" + (f"; {f['note']}" if f.get("note") else ""))
    return L + [""]


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
            f"{k} [`{d}/`]({d}/) ({_head_file(m, d)})" for k, d in m["heads"].items()) + ".", ""]
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
    for s in m["stages"]:
        if s.get("kind") == "distill":
            L += _distill_section(m, s)
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
        fcell = f"`{s['file']}`" if is_bundled(s) else (f"`{s['file']}` (not bundled; exports "
                                                          + ", ".join(f"`{e['file']}`" for e in s["exports"]) + ")")
        L.append(f"| {i} | [`{s['dir']}/`]({s['dir']}/) | `{s['phase']}` + `{s['stage']}` | {s['iterations']} | `{s['task']}` | "
                 f"{s.get('resume_from', '—')} | {s.get('clock', '—')} | {fcell} | `{s['sha256'][:12]}` |")
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
    L += _payload_table(m)
    for s in m["stages"]:
        ev = s.get("evals") or {}
        L += [f"### `{s['dir']}/` -- {s.get('what', '')}", "",
              f"- **Preset + overlay:** `KRABBY_OMNI_PHASE={s['phase']}` (`config/crab_hex/crab_hex_omni_phases.py`) + "
              f"stage `{s['stage']}` from {s['overlay_ref']} (task `{s['task']}`, kind `{s.get('kind', '')}`)",
              f"- **Iterations:** {s['iterations']}; resumes {s.get('resume_from', '—')}",
              f"- **Clock:** {s.get('clock', '—')}; observation contract `{HIDE_VAR}={s['obs_contract'][HIDE_VAR]}`",
              f"- **Source run (not tracked):** `{s['source']}`"
              + ("" if is_bundled(s) else " -- the checkpoint is NOT bundled (`bundled: false`; see the student "
                 "section above)"),
              f"- **Trained in:** {s.get('trained_in', '—')}",
              f"- **sha256:** `{s['sha256']}`"]
        for e in s.get("exports") or []:
            L.append(f"- **Export:** `{e['file']}` -- sha256 `{e['sha256']}`")
        if s.get("payload"):
            L.append("- **Payload knobs (training only):** "
                     + ", ".join(f"`{k}={v}`" for k, v in (s["payload"].get("knobs") or {}).items()))
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
    L += _superseded_section(m)
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
        "`parkour/`; the tool is cwd-independent) -- every stage / export / asset / stand / kept superseded file "
        "present with its manifest sha and at most 100 MiB, no stray checkpoints, README fresh (run by "
        "`tests/unit/test_policy_of_record_omni.py`).",
        "- New bake: add/replace the stage entries in `manifest.yaml` (source run, sha256, campaign, overlay_ref, "
        "obs_contract, evals), set `current`, then EITHER `git rm` the replaced stage files (`--sync` never deletes; "
        "`--check` flags them as stray checkpoints) OR keep them in their own folders and list them under the "
        "replaced head's `superseded[].files` (dir, file, sha256, source), run `--sync`, commit. A new head never "
        "reuses a kept folder.",
        "- A `kind: distill` stage whose checkpoint exceeds the 100 MiB per-file limit declares `bundled: false` and "
        "`exports:` (file, source, sha256, what): `--sync` copies the exports, never the checkpoint, and `--check` "
        "flags any checkpoint in its folder.",
        "- New shared asset (e.g. a re-harvested RSI bank): add an `assets:` entry (dir, file, source, sha256, what, "
        "used_by), run `--sync`, commit.",
    ]
    if sn:
        L.append(
            f"- Stand network (`stand_network:`, folder `{sn['dir']}/`): `--sync` copies its file"
            + (" and its `warm_start` ancestor's" if sn.get("warm_start") else "")
            + " from `source` sha-verified, `--check` verifies "
            + ("both" if sn.get("warm_start") else "it")
            + " like a stage (present, sha, no other checkpoint in the folder). Required keys: "
            + ", ".join(f"`{k}`" for k in STAND_KEYS) + "; `walker` must name a stage and repeat that stage's sha256, "
            "so re-baking the walking head forces a decision on the stand network (it was trained and gated beside "
            "that exact walker); an optional `warm_start` (" + ", ".join(f"`{k}`" for k in STAND_WARM_KEYS)
            + ") bundles the checkpoint it was warm-started from in the same folder. Replace it: edit the block, "
            "record the replaced net under `superseded:` (its commit and file sha), then `git rm` the old file, or "
            "keep it as the new net's `warm_start`, or keep it in place under that entry's `files:` (the new net then "
            "takes a folder of its own), run `--sync`, commit.")
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
        if sn and sn.get("warm_start"):
            stand += f" (+ warm-start ancestor {sn['warm_start']['file']})"
        cited = [s["dir"] for s in m["stages"] if not is_bundled(s)]
        kept = kept_files(m)
        print(f"policy of record: {len(m['stages'])} stages"
              + (f" ({len(cited)} cited, not bundled: {', '.join(cited)})" if cited else "")
              + f"{stand}" + (f" + {len(kept)} kept superseded files" if kept else "")
              + f", current {m['current']}: {'OK' if not problems else 'FAILED'}")
        return 1 if problems else 0
    sync(m, force=a.force)
    total = (sum(f["path"].stat().st_size for s in m["stages"] for f in stage_files(s))
             + sum(asset_path(a).stat().st_size for a in m["assets"])
             + (sum(f["path"].stat().st_size for f in stand_files(sn)) if sn else 0)) / 1e6
    stand = " + stand network" if sn else ""
    print(f"synced {len(m['stages'])} stages + {len(m['assets'])} assets{stand} ({total:.1f} MB; kept superseded "
          f"files verified, untouched: {len(kept_files(m))}), current {m['current']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
