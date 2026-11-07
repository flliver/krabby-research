#!/usr/bin/env python3
"""Rod-speed DR campaign, step 4 (CPU, no Kit): the comparison helpers of ``rsd_dump_check.sh``.

Pure stdlib. The shell runs the Kit config dumps (``tests/integration/crab_hex_phase_cfg_dump.py``); this file reads
the JSON they wrote. Run it with ``python -B -I`` (no bytecode in ``experiments/``, no import-path surprises).

    rsd_dump_compare.py same REF_JSON NEW_JSON REF_ROOT NEW_ROOT
        knob-OFF proof. Prints one line ``VERDICT <IDENTICAL|ROOT_LEAK|DIFFERENT> <detail>``:
          IDENTICAL   the two dumps are byte-identical after each dump's OWN root path is replaced by ``<ROOT>``
                      (the sweep's proven method; it also implies byte-identity under the symmetric replacement of
                      both roots, longest root first)
          ROOT_LEAK   identical only under the symmetric replacement: the new dump names a path under the
                      reference checkout (a module or asset resolved to the wrong checkout)
          DIFFERENT   anything else, with the first differing line
    rsd_dump_compare.py onoff OFF_JSON ON_JSON TASK LO HI RAMP
        knob-ON proof. A structural diff of the knob-off and knob-on dumps of the same case must touch exactly
        ``<TASK>/env/actions/joint_pos/rod_speed_dr`` (added, = {lo, hi, ramp_steps}) and
        ``_environ/KRABBY_DR_ROD_SPEED`` (added, = "LO:HI:RAMP"). Prints ``VERDICT <OK|BAD> <detail>``.
    rsd_dump_compare.py refused SPEC_PY
        the tuple ``STUDENT_RUNNER_KNOBS_REFUSED`` read by AST (as ``distill.sh`` reads it), space-joined.
    rsd_dump_compare.py refusal LOG RC
        the DR + stress refusal: LOG must carry the exclusivity error and RC must be non-zero.
        Prints ``VERDICT <OK|BAD> <detail>``.
"""
from __future__ import annotations

import ast
import json
import sys

DR_VAR = "KRABBY_DR_ROD_SPEED"
STRESS_VAR = "KRABBY_STRESS_ROD_SPEED"
TOKEN = "<ROOT>"


# --------------------------------------------------------------------------------------------- same
def normalise_own(text: str, root: str) -> str:
    return text.replace(root, TOKEN)


def normalise_both(text: str, roots) -> str:
    """Every root replaced, the LONGEST first (a worktree root sits under the main root)."""
    for root in sorted({r for r in roots if r}, key=len, reverse=True):
        text = text.replace(root, TOKEN)
    return text


def first_difference(a: str, b: str) -> str:
    la, lb = a.splitlines(), b.splitlines()
    for i in range(max(len(la), len(lb))):
        x = la[i] if i < len(la) else "<missing>"
        y = lb[i] if i < len(lb) else "<missing>"
        if x != y:
            return f"line {i + 1}: ref={x.strip()[:160]!r} new={y.strip()[:160]!r}"
    return "no line difference (trailing bytes)"


def same(ref_json: str, new_json: str, ref_root: str, new_root: str) -> tuple[str, str]:
    try:
        ta = open(ref_json).read()
        tb = open(new_json).read()
    except FileNotFoundError as exc:
        return "DIFFERENT", f"missing {exc.filename}"
    if normalise_own(ta, ref_root) == normalise_own(tb, new_root):
        return "IDENTICAL", f"{len(ta)} / {len(tb)} chars"
    roots = (ref_root, new_root)
    if normalise_both(ta, roots) == normalise_both(tb, roots):
        leaks = normalise_own(tb, new_root).count(ref_root)
        return "ROOT_LEAK", f"the new dump names the reference root {leaks} time(s) outside this checkout"
    return "DIFFERENT", first_difference(normalise_own(ta, ref_root), normalise_own(tb, new_root))


# --------------------------------------------------------------------------------------------- onoff
def diff_paths(a, b, path=""):
    """Structural diff: ``{path: ("added"|"removed"|"changed", old, new)}``; an added or removed subtree is ONE path."""
    out = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}/{k}"
            if k not in a:
                out[p] = ("added", None, b[k])
            elif k not in b:
                out[p] = ("removed", a[k], None)
            else:
                out.update(diff_paths(a[k], b[k], p))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out[path or "/"] = ("changed", f"list[{len(a)}]", f"list[{len(b)}]")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.update(diff_paths(x, y, f"{path}[{i}]"))
    elif a != b or type(a) is not type(b):
        out[path or "/"] = ("changed", a, b)
    return out


def onoff(off_json: str, on_json: str, task: str, lo: float, hi: float, ramp: int) -> tuple[str, str]:
    try:
        off = json.load(open(off_json))
        on = json.load(open(on_json))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return "BAD", f"unreadable dump: {exc}"
    got = diff_paths(off, on)
    want_cfg = f"/{task}/env/actions/joint_pos/rod_speed_dr"
    want_env = f"/_environ/{DR_VAR}"
    problems = []
    if set(got) != {want_cfg, want_env}:
        extra = sorted(set(got) - {want_cfg, want_env})
        missing = sorted({want_cfg, want_env} - set(got))
        problems.append(f"touched paths differ: extra={extra[:12]} missing={missing}")
    c = got.get(want_cfg)
    if c is not None:
        if c[0] != "added" or c[2] != {"lo": lo, "hi": hi, "ramp_steps": ramp}:
            problems.append(f"{want_cfg}: {c[0]} {c[2]!r}, want added {{'lo': {lo}, 'hi': {hi}, 'ramp_steps': {ramp}}}")
    e = got.get(want_env)
    if e is not None:
        text = f"{lo:g}:{hi:g}:{ramp}"
        if e[0] != "added" or e[2] != text:
            problems.append(f"{want_env}: {e[0]} {e[2]!r}, want added {text!r}")
    if problems:
        return "BAD", "; ".join(problems)
    return "OK", f"touched exactly {sorted(got)}"


# --------------------------------------------------------------------------------------------- refused / refusal
def refused_list(spec_py: str) -> list[str]:
    for node in ast.walk(ast.parse(open(spec_py).read())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED"
                                                for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit(f"STUDENT_RUNNER_KNOBS_REFUSED not found in {spec_py}")


def refusal(log_path: str, rc: int) -> tuple[str, str]:
    try:
        text = open(log_path, errors="replace").read()
    except FileNotFoundError:
        return "BAD", f"no log {log_path}"
    # The exit code cannot tell: crab_hex_phase_cfg_dump.py catches the exception, prints it, and its
    # ``simulation_app.close()`` ends the process with 0 before ``sys.exit(1)`` (K1, 2026-11-07: rc 0 after the
    # ValueError). The proof is the error in the log, raised by the cfg build, and no "[dump] wrote" line (the caller
    # also requires that no json was written).
    hit = [ln.strip() for ln in text.splitlines()
           if DR_VAR in ln and STRESS_VAR in ln and "exclusive" in ln and ln.lstrip().startswith("ValueError")]
    if not hit:
        tail = " | ".join(text.strip().splitlines()[-3:])[:300]
        return "BAD", f"rc={rc} but no exclusivity ValueError naming both knobs in the log; tail: {tail}"
    if "[dump] wrote" in text:
        return "BAD", f"rc={rc}: the error was printed but the dump was still written"
    if "apply_dr_knobs" not in text:
        return "BAD", f"rc={rc}: the error was not raised by dr_knobs.apply_dr_knobs (no frame in the traceback)"
    note = {0: " (Kit teardown exits 0 after the printed traceback)",
            124: " (killed by the timeout at Kit teardown, after the error)"}.get(rc, "")
    return "OK", f"rc={rc}{note}; {hit[0][:200]}"


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "same" and len(argv) == 6:
        verdict, detail = same(*argv[2:6])
        print(f"VERDICT {verdict} {detail}")
        return 0
    if cmd == "onoff" and len(argv) == 8:
        verdict, detail = onoff(argv[2], argv[3], argv[4], float(argv[5]), float(argv[6]), int(argv[7]))
        print(f"VERDICT {verdict} {detail}")
        return 0
    if cmd == "refused" and len(argv) == 3:
        print(" ".join(refused_list(argv[2])))
        return 0
    if cmd == "refusal" and len(argv) == 4:
        verdict, detail = refusal(argv[2], int(argv[3]))
        print(f"VERDICT {verdict} {detail}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
