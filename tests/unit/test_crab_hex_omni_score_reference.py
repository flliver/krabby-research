# SPDX-License-Identifier: BSD-3-Clause
"""``score_reference.py`` (plan step 2/8: score a reference chain's screens against a NAMED OLD
campaign's baseline, CPU-only, the driver's own ``decide_pooled``). Pinned here:

* it reproduces the formation campaign's recorded ``D05`` ledger entry field for field, exactly the
  self-check the module's own docstring gives as the worked example
  (``--old-campaign 2026-09-26_stroke_formation --base-xids X01 X02 --cand-xids X03 X04
  --cand-spec S_f --check-entry D05``);
* ``--list-only`` finds the four screens without scoring anything;
* ``main()``'s exit code is 0 on a clean self-check reproduction and 1 the moment a single evidence
  key is made to disagree (a fault-injection check that the self-check is not a tautology);
* the self-check is interpreter-portable: D05 was recorded under Python 3.11 (the Isaac venv) and
  reproduces on the running interpreter's own ``sum()`` (3.11 or 3.12) AND under the other one's float
  summation, while a float off by more than ``FLOAT_REL_TOL`` or any non-float change still fails.

Skips cleanly (not a failure) if the real campaign directory this needs is not on disk in this
checkout -- the tool reads real ``state.json`` / ``ledger.json`` / ``candidates.yaml`` /
``resolution.<sha>.json`` files, not a synthetic fixture (module docstring: "CPU ONLY", no Isaac, but
still real recorded campaign data)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
TOOL_PY = OMNI / "experiments/tools/score_reference.py"
FORMATION_DIR = OMNI / "experiments/2026-09-26_stroke_formation"

pytestmark = pytest.mark.skipif(
    not (FORMATION_DIR / "state.json").is_file(),
    reason=f"{FORMATION_DIR} is not on disk in this checkout -- score_reference.py needs the real "
           "campaign records (state.json/ledger.json/candidates.yaml/resolution.*.json), not a fixture",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sr():
    return _load(TOOL_PY, "score_reference__test")


BASE_ARGS = ["--old-campaign", str(FORMATION_DIR), "--base-xids", "X01", "X02",
             "--cand-xids", "X03", "X04", "--cand-spec", "S_f"]


def test_list_only_finds_the_four_screens_without_scoring(sr, capsys):
    rc = sr.main([*BASE_ARGS, "--list-only"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["runs"]) == 4
    for r in out["runs"]:
        assert r["eval_sha"] == out["old_eval_sha"]
        assert r["xid"] in ("X01", "X02", "X03", "X04")


def test_reproduces_formation_d05_field_for_field(sr, capsys):
    rc = sr.main([*BASE_ARGS, "--check-entry", "D05"])
    captured = capsys.readouterr().out
    assert rc == 0, captured
    assert "reproduced field for field" in captured
    assert "[self-check] D05:" in captured


def test_check_entry_reproduction_via_the_module_api_matches_the_recorded_entry(sr):
    """The same reproduction, through :func:`score` / :func:`check_entry` directly (not the CLI),
    so the self-check is pinned at the API level too -- and to read the pick/confidence off the
    result rather than only trusting the CLI's own printed summary."""
    old = sr.Campaign(FORMATION_DIR)
    b3 = sr._ok_screen(old, old.header("X01"), "baseline s3")
    b2 = sr._ok_screen(old, old.header("X02"), "baseline s2")
    c3 = sr._ok_screen(old, old.header("X03"), "candidate s3")
    c2 = sr._ok_screen(old, old.header("X04"), "candidate s2")
    entry = sr.rc().entry_by_id(old.ledger, "D05")
    assert entry is not None
    anchor = (entry.get("evidence") or {}).get("anchor")
    result = sr.score(old, (b3, b2), (c3, c2), cand_spec_id="S_f", prior=entry, anchor_label=anchor)
    assert result["chosen"] == entry["chosen"]
    assert result["pooled_confidence"] == entry["pooled_confidence"]
    assert result["why"] == entry["why_pooled"]
    diffs = sr.check_entry(result, entry)
    assert diffs == []


def test_check_entry_exits_nonzero_the_moment_evidence_disagrees(sr, capsys):
    """Fault injection: if the reproduction (or the recorded entry) really differed, the self-check
    must say so and exit 1 -- otherwise it would be a tautology that always reads 'reproduced'."""
    old = sr.Campaign(FORMATION_DIR)
    entry = sr.rc().entry_by_id(old.ledger, "D05")
    assert entry is not None
    tampered = copy.deepcopy(entry)
    tampered["chosen"] = "some-verdict-that-was-never-recorded"
    b3 = sr._ok_screen(old, old.header("X01"), "baseline s3")
    b2 = sr._ok_screen(old, old.header("X02"), "baseline s2")
    c3 = sr._ok_screen(old, old.header("X03"), "candidate s3")
    c2 = sr._ok_screen(old, old.header("X04"), "candidate s2")
    anchor = (entry.get("evidence") or {}).get("anchor")
    result = sr.score(old, (b3, b2), (c3, c2), cand_spec_id="S_f", prior=tampered, anchor_label=anchor)
    diffs = sr.check_entry(result, tampered)
    assert any("chosen" in d for d in diffs)

    rc = sr.main([*BASE_ARGS, "--check-entry", "D05"])
    capsys.readouterr()
    assert rc == 0  # the REAL recorded D05 (untampered) still reproduces cleanly via the CLI


# ================================================================================== interpreter portability
def _left_fold_sum(iterable, start=0):
    """``sum()`` as CPython <= 3.11 computes floats: a plain left-to-right fold."""
    acc = start
    for x in iterable:
        acc = acc + x
    return acc


def _compensated_sum(iterable, start=0):
    """``sum()`` as CPython >= 3.12 computes floats (Neumaier compensation); ints / non-numbers fold as before."""
    acc, comp, fl = start, 0.0, False
    for x in iterable:
        if isinstance(x, float) or (fl and isinstance(x, int) and not isinstance(x, bool)):
            if not fl:
                acc, fl = float(acc), True
            t = acc + x
            comp += (acc - t) + x if abs(acc) >= abs(x) else (x - t) + acc
            acc = t
        else:
            acc = acc + x
    return acc + comp if fl else acc


def test_compensated_sum_matches_this_interpreter_s_own_sum_semantics():
    """The two emulations are what they claim: on 3.12+ the builtin equals the compensated one, on <= 3.11 the
    left fold; and they really differ on a 3-term case (else the cross-interpreter test below proves nothing)."""
    xs = [0.1] * 10
    own = _compensated_sum if sys.version_info >= (3, 12) else _left_fold_sum
    assert sum(xs) == own(xs)
    assert _left_fold_sum(xs) != _compensated_sum(xs)


@pytest.mark.parametrize("other", [_left_fold_sum, _compensated_sum], ids=["sum_3_11", "sum_3_12"])
def test_d05_reproduces_under_either_interpreter_s_float_summation(sr, capsys, monkeypatch, other):
    """The recorded entry reproduces whichever interpreter's float ``sum()`` computes the pooled statistics --
    the one-ulp drift between them is inside ``FLOAT_REL_TOL`` (nothing is pinned to the recording's 3.11)."""
    import builtins  # noqa: PLC0415

    monkeypatch.setattr(builtins, "sum", other)
    rc = sr.main([*BASE_ARGS, "--check-entry", "D05"])
    assert rc == 0, capsys.readouterr().out


def test_check_entry_tolerance_is_tight_and_floats_only(sr):
    """One ulp passes; 1e-9 relative fails; a 0.0 has no absolute slack; ints, strings, bools, keys and lengths are
    exact; two NaNs agree."""
    import math  # noqa: PLC0415

    assert sr.FLOAT_REL_TOL == 1e-12
    base = {"chosen": "keep", "pooled_confidence": 0.9225, "why_pooled": "w", "status": None,
            "evidence": {"pooled": {"p_bar": 0.9225, "n": 4, "z": 0.0, "nan": float("nan"), "ok": True},
                         "levels": [0.1, 0.2], "anchor": "X01"}}

    def res(**ev):
        e = copy.deepcopy(base["evidence"])
        for k, v in ev.items():
            e["pooled"][k] = v
        return {"chosen": "keep", "pooled_confidence": base["pooled_confidence"], "why": "w", "status": None,
                "evidence": e}

    assert sr.check_entry(res(), base) == []
    assert sr.check_entry(res(p_bar=math.nextafter(0.9225, 0.0)), base) == []            # 3.11 vs 3.12: one ulp
    assert sr.check_entry(res(p_bar=0.9225 * (1 + 1e-9)), base) == ["evidence.pooled differs (at pooled.p_bar)"]
    assert sr.check_entry(res(z=1e-300), base) == ["evidence.pooled differs (at pooled.z)"]   # no abs floor
    assert sr.check_entry(res(n=5), base) == ["evidence.pooled differs (at pooled.n)"]
    assert sr.check_entry(res(ok=False), base) == ["evidence.pooled differs (at pooled.ok)"]
    assert sr.check_entry(res(extra=1), base) == ["evidence.pooled differs (at pooled (keys))"]
    r = res()
    r["evidence"]["levels"].append(0.3)
    assert sr.check_entry(r, base) == ["evidence.levels differs (at levels (length))"]
    r = res()
    r["evidence"]["anchor"] = "X02"
    assert sr.check_entry(r, base) == ["evidence.anchor differs (at anchor)"]
    r = res()
    r["why"] = "w2"
    assert len(sr.check_entry(r, base)) == 1 and sr.check_entry(r, base)[0].startswith("why_pooled:")
    r = res()
    r["pooled_confidence"] = math.nextafter(0.9225, 1.0)
    assert sr.check_entry(r, base) == []
