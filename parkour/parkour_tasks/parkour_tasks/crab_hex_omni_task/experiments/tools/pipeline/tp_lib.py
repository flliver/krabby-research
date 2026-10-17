#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers of the Task 4 pipeline tools (TP): paths, by-path loaders, hashing, records, the Isaac-busy check.

Pure standard library + yaml (no numpy / torch at import), so ``pipeline.py`` and the queue start fast and the CPU
tests can load every TP module by path (TP has no ``__init__.py``; nothing here is dated: every campaign path comes
from the spec ``PL/pipeline.yaml`` or the command line).

ONE ISAAC PROCESS AT A TIME. :data:`ISAAC_PAT` is the anchored pattern the campaign's launchers share (CHARTER section 2
running rules); :func:`isaac_busy` matches it against ``ps -ww -eo args`` -- never ``pgrep -f``, whose pattern would
match the watcher's own command line. No TP tool's own file name matches the pattern (pinned by the unit tests).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent


def repo_root() -> Path:
    for q in HERE.parents:
        if (q / "parkour").is_dir() and (q / ".git").exists():
            return q
    raise RuntimeError(f"repo root (a dir with parkour/ and .git) not found above {HERE}")


REPO = repo_root()
TOOLS = HERE.parent                                   # OMNI/experiments/tools
EXP = TOOLS.parent                                    # OMNI/experiments
OMNI = EXP.parent                                     # crab_hex_omni_task
FWD = OMNI.parent / "crab_hex_forward_task"
LOG_ROOT = REPO / "parkour/logs/rsl_rl"
EVAL_ROOT = LOG_ROOT / "gait_eval/omni_phases"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
HARNESS = FWD / "scripts/eval_crab_hex_gait.py"

#: the anchored Isaac-process pattern (CHARTER section 2; the prompt-pinned list + the campaign dump tools)
ISAAC_PAT = (r"^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|"
             r"play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|"
             r"bc_dump_cfgs|t4_dump_cfgs)\.py")
#: the stand tools' Kit-free modes (CPU reads, env composition) are never Isaac processes
NONKIT_PAT = (r"(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|"
              r"(/s[hn]_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|calib|common)\.py( |$))")
SENTINEL = "T4PIPE"


def isaac_lines(ps_text: str | None = None) -> list[str]:
    """The ``ps -ww -eo args`` lines that are Isaac processes (``ps_text`` injectable for the tests)."""
    if ps_text is None:
        ps_text = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout
    pat, non = re.compile(ISAAC_PAT), re.compile(NONKIT_PAT)
    return [ln for ln in ps_text.splitlines() if pat.search(ln) and not non.search(ln)]


def isaac_busy(ps_text: str | None = None) -> bool:
    return bool(isaac_lines(ps_text))


# ---------------------------------------------------------------------------------------------- loaders
def load_by_path(path: Path, name: str):
    """A module by path, no bytecode, registered in ``sys.modules`` (dataclasses / pickles need it)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = old
    return mod


_RC = None


def rc():
    """``run_candidates.py`` (the driver: code_sha / eval_sha / dirty paths, the stage env composition, the knob
    parsers, the candidates validator), loaded by path once."""
    global _RC
    if _RC is None:
        if str(TOOLS) not in sys.path:
            sys.path.insert(0, str(TOOLS))
        _RC = load_by_path(TOOLS / "run_candidates.py", "run_candidates_tp")
    return _RC


def thresholds_mod():
    return load_by_path(HERE / "thresholds.py", "tp_thresholds")


#: The dated paths the tools read (side22 classifier, sh_kin, the phase-2 scripts) -- set from the spec's ``paths``
#: block by :func:`configure` (TP's own source carries no dated path).
PATHS: dict = {}


def configure(spec_d: dict) -> None:
    """Take the spec's ``paths`` block (relative to the omni task dir)."""
    for k, v in (spec_d.get("paths") or {}).items():
        if k != "notes":
            PATHS[k] = OMNI / v


def configure_from(spec_path) -> None:
    configure(load_yaml(spec_path))


def path(key: str) -> Path:
    if key not in PATHS:
        raise RuntimeError(f"tp_lib.path({key!r}): not configured -- call tp_lib.configure(spec) (the spec's paths block)")
    return PATHS[key]


def side22_mod():
    return load_by_path(path("side22_classify"), "tp_side22_classify")


# ---------------------------------------------------------------------------------------------- hashing / io
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj) -> str:
    """Canonical JSON text: key ORDER kept (an overlay is verbatim, order included), compact separators."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def sha256_obj(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def read_json(p: Path, default=None):
    p = Path(p)
    if not p.is_file():
        return default
    return json.loads(p.read_text())


def write_text_atomic(p: Path, text: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(p: Path, obj) -> None:
    write_text_atomic(p, json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n")


def read_env_file(p: Path) -> dict:
    """``KEY=VALUE`` lines (``#`` comments and blank lines skipped); values kept verbatim (empty allowed)."""
    out = {}
    for ln in Path(p).read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        k, sep, v = ln.partition("=")
        if not sep or not re.match(r"^[A-Z][A-Z0-9_]*$", k):
            raise ValueError(f"{p}: malformed line {ln!r}")
        if k in out:
            raise ValueError(f"{p}: {k} given twice")
        out[k] = v
    return out


def env_file_text(env: dict) -> str:
    """Sorted ``KEY=VALUE`` lines -- the byte-stable form gen writes and the row runner reads."""
    for k, v in env.items():
        if "\n" in str(v) or " " in str(v):
            raise ValueError(f"env value {k}={v!r} has whitespace (the row runner word-splits the env)")
    return "".join(f"{k}={env[k]}\n" for k in sorted(env))


def rel(p) -> str:
    """Repo-relative POSIX path text (absolute input under the repo), else the path as given."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def absp(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def now_iso() -> str:
    from datetime import datetime  # noqa: PLC0415

    return datetime.now().isoformat(timespec="seconds")


def sentinel(log: Path, kind: str, text: str) -> str:
    """Append ``T4PIPE <KIND> <text> <time>`` to the queue log -- the orchestrator's hook (the watcher greps it)."""
    line = f"{SENTINEL} {kind} {text} {now_iso()}"
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")
    return line


def append_jsonl(p: Path, obj) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


def load_yaml(p: Path):
    import yaml  # noqa: PLC0415

    return yaml.safe_load(Path(p).read_text())


def dump_loader():
    """``run_candidates._dump_loader`` semantics without loading the driver: python tags read as plain data."""
    import yaml  # noqa: PLC0415

    class Loader(yaml.SafeLoader):
        pass

    def _plain(loader, _suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)
    Loader.add_multi_constructor("!", _plain)
    return Loader
