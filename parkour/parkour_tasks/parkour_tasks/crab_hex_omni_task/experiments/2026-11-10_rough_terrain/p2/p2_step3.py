#!/usr/bin/env python3
"""P2 step 3 (p2/P2.md = PLAN_P2 step 3): the GPU smoke, the d-prime rows and the c31000 bridge, through the
rough-terrain campaign's OWN row machinery -- nothing here launches Kit itself.

    p2_step3.py --block smoke|dprime|bridge [--relaunch 1|2] [--dry | --list] [--from L] [--only L ...]
    p2_step3.py --dry                         every block's dry pass in order (smoke, dprime, bridge, bridge R1/R2)
    p2_step3.py --inputs-check --block B      (CPU) the block's pre-flight; also the first step of every block
    p2_step3.py --check-smoke                 (CPU) the smoke verdict: P2_SMOKE PASS|FAIL (exit 0 / 1)
    p2_step3.py --judge-bridge                (CPU) the bridge verdict -> results/p2/bridge_judge.json; prints
                                              P2_BRIDGE PASS | RELAUNCH | STOP | INCOMPLETE (exit 0 / 10 / 11 / 12)
    p2_step3.py --read                        (CPU) results/p2/step3.json: smoke, d-prime + trigger, bridge

HOW THE ROWS RUN (reuse, nothing copied): the blocks are installed into ``scripts/rt_machinery`` (``BLOCKS`` and
``plan`` patched, the ``short_arm/rt_short.py`` pattern) and executed by ``scripts/rt_queue.main``: the flock on the
campaign's ``.isaac.lock``, the anchored ``ps -ww`` Isaac-busy wait (never ``pgrep -f``), the DerivedDataCache /
free-disk guards before every row (``RTQ_DDC_MAX_GB`` 78, ``RTQ_FREE_MIN_GB`` 30 here, both overridable from the
caller's env), one ``RTQ <label> rc=<n>`` sentinel per step, ``RTQ_DONE <block> fail=<n>``, skip-on-complete at the
pinned build, ``STOP_QUEUE`` between steps (``results/p2/STOP_QUEUE`` OR the campaign's ``results/STOP_QUEUE``).
Every row is ``rt_machinery.gate_step`` -> ``tools/pipeline/gate_row.sh`` (env = exactly the row's env file + level
file, every parent KRABBY_* removed, PYTHONPATH = this worktree's parkour/parkour_tasks:parkour, systemd-run --user
--scope MemoryMax=48G, timeout --signal=TERM, stamp + code_state.json + tp_row.json). Output root
``$RT_TP_OUT`` = ``<MAIN>/parkour/outputs/omni_plan/rough_terrain_p2`` (set here before the machinery is imported;
a caller's different RT_TP_OUT is refused). Logs: ``<P2_OUT>/logs/<block>/<label>.log``.

The policy: c31000 = RSD_ST_s19 ``model_31000.pt`` from the MAIN checkout's logs, loaded by path and checked by
sha256 (155b5981a718...) and ``iter`` 31000 before any block plans. Role student everywhere
(``--policy-role student``); no TP_STRIP_KEYS (every env file is checked free of
``omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED``, read by AST).

Blocks (names carry the sha8; ``P2S_`` smoke, ``P2D_`` d-prime, ``P2B_`` bridge):
* smoke: rows_rt's rt_slopes_b2 / rt_ledge_down_b2 at es90, ``--episodes 4``, launched TWICE (``P2S_155b5981_<kind>``
  and ``..._L2``; PLAN Left-for-implementation 12). --check-smoke: rc 0 (tp_row), run_meta role student, sha =
  c31000, code_sha / eval_sha = results/pin.json, readout's own ``_complete`` + ``find`` (row args with
  ``--policy-role student --episodes 4``) OK and ``episode_table`` parses at the row's cross radius, raw npz count ==
  episodes; the two launches BITWISE-EQUAL or CLASS-FLIP by classflip_check.py (same pin, so its own verdict holds).
* dprime: c31000 on rt_slopes_b2 / rt_ledge_down_b2 at es1 and es2 (100 envs): rows_rt's row with ONLY the role
  swapped (env file, manifest, scenario, args otherwise identical); the inputs check proves c1's (RTSW_6d1f29be) and
  TDR's (RT9_03c3e472) recorded es1-2 stamps carry exactly that composition with role teacher.
* bridge: c31000 at es1 on L090 field_light + suite and x0.7 L090 field_light + suite, composed by
  ``rsd_lib.cell_rows`` (the RSD cell composition; cells T00_nominal_L090_Z / T01_rod_speed_0p7_L090), names
  ``P2B_155b5981_L090`` / ``P2B_155b5981_x07_L090`` (+ ``_R1`` / ``_R2`` relaunches, planned only from a FRESH
  results/p2/bridge_judge.json).

THE BRIDGE JUDGE (RSD CHARTER section 4 as amended 2026-11-08 20:12 / 23:06; classflip_check.py + wave_diff.py run
unchanged): every recorded c31000 stamp of the SAME SHAPE (rod_speed_dr/lr, keyed by content: run_meta sha256,
scenario, env seed, role, ``env_vars`` == the row's composed env, tp_row args as flag/value pairs) is compared with
each P2 launch. Path-valued run_meta strings differ across checkouts (dr-stress-sweep worktree vs MAIN logs vs this
worktree: ``checkpoint``, ``student.checkpoint_agent_yaml``, ``student.module``, ``student.zero_depth_source``,
``usd_path`` ...), so BOTH stamps are compared through a shadow dir whose run_meta.json has every string passed
through ``rt_machinery.normalise_roots`` (``<MAIN>`` and ``<MAIN>/.claude/worktrees/<name>`` -> ``<ROOT>``; all
other files symlinked); every key path the normalisation changed is recorded. Per pair:
  bitwise   = classflip verdict BITWISE-EQUAL (every raw array of every episode, bit for bit);
  signature = cfg identical (after normalisation) AND dose identical AND no structural mismatch AND (field_light)
              step-0 state identical AND earliest difference action-only at a depth-frame step, or (suite, wave 0
              per the 23:06 clarification) wave 0 bit-equal or its earliest difference action-only at a depth-frame
              step with no step-0 key but ``actions``. classflip's ``same_code`` clause is NOT used: the bridge is
              cross-pin by construction, and the amended clause replaces it by the bitwise match.
Row: PASS if ANY launch is bitwise equal to ANY recorded stamp; else, if the latest launch shows the signature
against EVERY recorded stamp and fewer than 3 launches exist, RELAUNCH_DUE (``_R1``, ``_R2``); else STOP (a code-path
difference, or no bitwise match after 2 relaunches). Every row records the step-0 action fingerprint in its class
envs (wave 0 on suite): the envs where the recorded stamps disagree at step 0 (or, with one recorded class, where the
new launch disagrees), and whether the new launch equals each recorded stamp there. Any STOP prints
``P2_BRIDGE STOP`` (notify-and-ask is the parent's job; the distill runs regardless).

THE D-PRIME READ (--read; report only, never pooled into the P2 read): success = complete AND crossed at the row's
readout cross radius (readout.episode_table), paired on (es, env_index) through ``readout.pair_tables`` (equal key
sets, equal spawns, 100 episodes per seed, one build) with c1 (6d1f29be) and TDR (03c3e472) at es1-2;
d' = TDR - c31000 and c1 - c31000 by ``readout.g_line`` (paired bootstrap SE; its gain / regress flags are not used).
Pre-registered trigger: c31000 >= c1 - 0.10 on BOTH kinds -> ``P2_DPRIME TRIGGER`` (notify and ask); on one kind ->
that kind is recorded ``cannot_gain`` (``P2_DPRIME CANNOT_GAIN <kind>``). The index is built from seed dirs 1-2 / 90
only (no es9+ stamp is ever opened here).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/p2
CAMP = HERE.parent
SCRIPTS = CAMP / "scripts"
SHORT = CAMP / "short_arm"
EXPS = CAMP.parent
RSD_SCRIPTS = EXPS / "2026-11-07_rod_speed_dr" / "scripts"
MAIN = Path("/home/nickmagus/krabby/krabby-research")
P2_OUT = MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain_p2"
RT_OUT = MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain"
RSD_LR = MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr" / "lr"

# ---- the machinery reads these at import: set them first (a different caller RT_TP_OUT is refused in main())
_CALLER_RT_TP_OUT = os.environ.get("RT_TP_OUT")
os.environ["RT_TP_OUT"] = str(P2_OUT)
os.environ.setdefault("RTQ_DDC_MAX_GB", "78")             # CHARTER 3.5 / STAND_REPAIR: <= 78 GB
os.environ.setdefault("RTQ_FREE_MIN_GB", "30")            # >= 30 GB free

sys.path.insert(0, str(SCRIPTS))
import rt_machinery as M  # noqa: E402
import rt_queue as Q  # noqa: E402
import readout  # noqa: E402  (scripts/readout.py, imported as readout_fresh.py does; never edited)
import rt_spec as S  # noqa: E402

sys.path.insert(0, str(SHORT))
import rt_short  # noqa: E402  (slot_problems, iter_field; its install() is never called here)

assert Path(M.__file__).resolve().parent == SCRIPTS and Path(readout.__file__).resolve().parent == SCRIPTS
assert Path(rt_short.M.__file__).resolve() == Path(M.__file__).resolve()
assert M.OUT == P2_OUT, f"rt_machinery.OUT {M.OUT} != {P2_OUT}"

# ============================================================================== pre-registered constants
C31000_CK = MAIN / "parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19/model_31000.pt"
C31000_SHA = "155b5981a718ec2c41460490c38e2aadb61c60c7b48dd67e3a33a45b1d5e634e"
C31000_ITER = 31000
C1_SHA = "6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1"
TDR_SHA = M.TDR_SHA
SHA8 = C31000_SHA[:8]
KINDS = {"slopes": "rt_slopes_b2__omni", "ledge_down": "rt_ledge_down_b2__omni"}
SMOKE_ES, SMOKE_EPISODES, SMOKE_LAUNCHES = 90, 4, 2
DPRIME_ES = (1, 2)
DPRIME_MARGIN = 0.10
#: the families the d-prime rows must match (name prefix by sha; the inputs check reads their stamps by content)
DPRIME_FAMILIES = {"c1": (C1_SHA, "RTSW_6d1f29be_"), "tdr": (TDR_SHA, "RT9_03c3e472_")}
BRIDGE_ES = 1
BRIDGE_CELLS = {"L090": "T00_nominal_L090_Z", "x07_L090": "T01_rod_speed_0p7_L090"}
BRIDGE_FAMS = ("field_light", "suite")
BRIDGE_MAX_LAUNCHES = 3                                   # the first launch + up to 2 relaunches
STUDENT_ROW_TIMEOUT_S = 5400                              # RSD's student-row timeout (SIGTERM only)
PERIOD = 5                                                # depth frame every 5th policy step (classflip --period)
RESULTS_P2 = CAMP / "results" / "p2"
STEP3_JSON = RESULTS_P2 / "step3.json"
JUDGE_JSON = RESULTS_P2 / "bridge_judge.json"
P2_PIN = RESULTS_P2 / "pin.json"
P2_STOP = RESULTS_P2 / "STOP_QUEUE"
CLASSFLIP = RSD_SCRIPTS / "classflip_check.py"
WAVE_DIFF = RSD_SCRIPTS / "wave_diff.py"
STUDENT_SPEC = M.OMNI / "config" / "crab_hex" / "omni_student_spec.py"
BLOCKS = {"smoke": "p2smoke", "dprime": "p2dprime", "bridge": "p2bridge", "bridge_r1": "p2bridge_r1",
          "bridge_r2": "p2bridge_r2"}
EXPECT_SLOTS = {"p2smoke": 4, "p2dprime": 4, "p2bridge": 4}
JUDGE_RC = {"PASS": 0, "RELAUNCH": 10, "STOP": 11, "INCOMPLETE": 12}
#: a block's rc for p2_queue.sh: REFUSED = it ended non-zero with NO row launched (a stop_on_fail check -- the build
#: pin, --inputs-check --, a plan refusal, the .isaac.lock refusal, the c31000 identity refusal, a STOP file, a crash);
#: HALT = rt_queue stopped on the DerivedDataCache growth bound or a guard. The chain exits on either (never on 1 = a
#: failed row, which it counts and goes on).
REFUSED_RC, HALT_RC = 20, 21
NORMALISATION_RULE = ("run_meta.json: every string value passed through rt_machinery.normalise_roots (<MAIN> and "
                      "<MAIN>/.claude/worktrees/<name> -> <ROOT>) on BOTH stamps of a pair, in a shadow dir (run_meta "
                      "rewritten, every other file symlinked); classflip_check.py / wave_diff.py run unchanged on "
                      "the shadow dirs")
MODE = {"dry": False}


def is_dry() -> bool:
    return MODE["dry"] or os.environ.get("P2_DRY") == "1"


def say(line: str) -> None:
    print(line, flush=True)


# ============================================================================== small io
def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def pin() -> dict | None:
    return M.read_pin()


def refused_keys() -> list:
    """omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED, read by AST (as distill.sh / rsd_dump_compare read it)."""
    tree = ast.parse(STUDENT_SPEC.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED"
                                                for t in node.targets):
            return [str(x) for x in ast.literal_eval(node.value)]
    raise RuntimeError(f"{STUDENT_SPEC}: no STUDENT_RUNNER_KNOBS_REFUSED")


def raw_problems(st: Path) -> list:
    """PLAN item 11: raw npz count == scenario_metrics episodes; no RAW_DELETED marker (2026-11-11 cleanup)."""
    st = Path(st)
    raw = st / "raw"
    if not raw.is_dir():
        return [f"{st}: no raw/"]
    out = []
    if any(p.name.startswith("RAW_DELETED") for p in raw.iterdir()):
        out.append(f"{st}: raw/ carries a RAW_DELETED marker")
    n_npz = len(list(raw.glob("episode_*.npz")))
    eps = (read_json(st / "scenario_metrics.json", {}) or {}).get("episodes") or []
    if n_npz != len(eps) or not eps:
        out.append(f"{st}: raw npz {n_npz} != scenario_metrics episodes {len(eps)}")
    return out


def args_pairs(args) -> list:
    """tp_row / row args as sorted (flag, value) pairs (the RSDSEL and RSDS rows order the same flags differently)."""
    args = [str(a) for a in args]
    out, i = [], 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append((args[i], args[i + 1]))
            i += 2
        else:
            out.append((args[i], None))
            i += 1
    return sorted(out, key=str)


def student_args(args) -> list:
    args = [str(a) for a in args]
    if "--policy-role" not in args or args.index("--policy-role") + 1 >= len(args):
        raise ValueError(f"row args {args} carry no --policy-role value")
    i = args.index("--policy-role")
    return [*args[:i + 1], "student", *args[i + 2:]]


def teacher_args(args) -> list:
    args = [str(a) for a in args]
    i = args.index("--policy-role")
    return [*args[:i + 1], "teacher", *args[i + 2:]]


# ============================================================================== c31000 identity
_IDENT: dict = {}


def c31000_problems(check_iter: bool = True) -> list:
    """The student checkpoint: the MAIN-logs file, sha256 == the pre-registered value, ``iter`` == 31000."""
    if "why" in _IDENT and (_IDENT.get("iter_checked") or not check_iter):
        return _IDENT["why"]
    why = []
    if not C31000_CK.is_file():
        why.append(f"no checkpoint {C31000_CK}")
    else:
        sha = M.sha256_file(C31000_CK)
        if sha != C31000_SHA:
            why.append(f"{C31000_CK} sha256 {sha[:8]} != pre-registered {C31000_SHA[:8]}")
        if check_iter and not why:
            it = rt_short.iter_field(C31000_CK)
            if it != C31000_ITER:
                why.append(f"{C31000_CK.name} iter field {it!r} != {C31000_ITER}")
    _IDENT.update(why=why, iter_checked=check_iter)
    return why


# ============================================================================== rows
def rows_by_id() -> dict:
    doc = S.load_rows()
    return {r["id"]: r for r in doc["kind_rows"] + doc["retention_rows"]}


def smoke_row(rid: str) -> dict:
    r = dict(rows_by_id()[rid])
    r["args"] = student_args(r["args"]) + ["--episodes", str(SMOKE_EPISODES)]
    return r


def dprime_row(rid: str) -> dict:
    r = dict(rows_by_id()[rid])
    r["args"] = student_args(r["args"])
    return r


def smoke_name(rid: str, launch: int) -> str:
    return M.row_name("P2S", C31000_SHA, rid) + ("" if launch == 1 else f"_L{launch}")


def dprime_name(rid: str) -> str:
    return M.row_name("P2D", C31000_SHA, rid)


def bridge_name(tag: str, k: int) -> str:
    return f"P2B_{SHA8}_{tag}" + ("" if k == 0 else f"_R{k}")


_RSD: dict = {}


def rsd_lib():
    if "lib" not in _RSD:
        sys.path.insert(0, str(RSD_SCRIPTS))
        import rsd_lib as L  # noqa: PLC0415
        assert Path(L.__file__).resolve().parent == RSD_SCRIPTS
        _RSD["lib"] = L
    return _RSD["lib"]


def bridge_rows() -> list:
    """``[{key, tag, cell, fam, row}]``: the 4 bridge rows, composed by rsd_lib.cell_rows (RSD's cell composition)."""
    L = rsd_lib()
    pol = {"name": "c31000", "role": "student", "sha8": SHA8, "checkpoint": str(C31000_CK), "sha256": C31000_SHA}
    out = []
    for tag, cid in BRIDGE_CELLS.items():
        by_fam = {r["family"]: r for r in L.cell_rows(pol, cid, BRIDGE_ES)}
        for fam in BRIDGE_FAMS:
            r = by_fam[fam]
            row = {"id": f"bridge_{tag}_{fam}", "env_file": r["env_file"], "level_file": r["level_file"],
                   "manifest": r["manifest"], "scenario": r["scenario"], "args": list(r["args"])}
            if r.get("strip"):
                raise ValueError(f"{cid} {fam}: rsd_lib composes TP_STRIP_KEYS {r['strip']} (expected none)")
            out.append({"key": f"{tag}.{fam}", "tag": tag, "cell": cid, "fam": fam, "row": row})
    return out


def _student_step(label: str, name: str, row: dict, es: int, extra=(), note: str = "") -> M.Step:
    st = M.gate_step(label, name=name, ck=C31000_CK, row=row, es=es, tp_out=P2_OUT, extra=extra)
    st.env["TP_ROW_TIMEOUT"] = str(STUDENT_ROW_TIMEOUT_S)
    tp_out, nm, sc, e = st.slot
    st.note = (f"{note + '; ' if note else ''}c31000 {SHA8} student; expected stamp "
               f"{M.slot_dir(tp_out, nm, sc, e)}/<YYYY-MM-DD_HH-MM-SS>/")
    return st


def build_check_step(prefix: str) -> M.Step:
    """``build_check.py check`` against results/p2/pin.json (PLAN step 2's pin, written with ``build_check.py pin
    --pin-path results/p2/pin.json``): results/pin.json's frozen_files predate envs/P2_distill.env, so a check against
    it now reads BAD ('frozen campaign files changed'). Without the P2 pin (a dry pass before step 2) the step checks
    results/pin.json and says so; a real run refuses in ``.inputs.check`` anyway."""
    if P2_PIN.is_file():
        return M._cpu(f"{prefix}.check", M.BUILD_CHECK, "check", "--pin-path", P2_PIN, stop=True, dry=True,
                      note=f"the live build == {P2_PIN} (code_sha / eval_sha / frozen files), clean code/eval tree")
    return M._cpu(f"{prefix}.check", M.BUILD_CHECK, "check", stop=True, dry=True,
                  note=f"NO {P2_PIN} yet: checking {M.PIN_PATH} instead (it reads BAD while envs/P2_distill.env is "
                       f"not in its frozen_files; PLAN step 2 writes the P2 pin)")


def _checks(prefix: str, block: str, relaunch: int = 0) -> list:
    steps = [build_check_step(prefix)]
    extra = ("--relaunch", relaunch) if relaunch else ()
    steps.append(M._cpu(f"{prefix}.inputs.check", HERE / "p2_step3.py", "--inputs-check", "--block", block, *extra,
                        stop=True, dry=True,
                        note="P2 pin == results/pin.json, c31000 sha/iter, no refused runner key, the recorded "
                             "families this block pairs with / bridges to"))
    return steps


def plan_smoke() -> list:
    steps = _checks("P2S", "smoke")
    for launch in range(1, SMOKE_LAUNCHES + 1):
        for kind, rid in KINDS.items():
            # ``--episodes 4`` is IN the row's args (smoke_row), so tp_row.args == the row readout.find keys on
            steps.append(_student_step(f"P2S.{kind}.L{launch}", smoke_name(rid, launch), smoke_row(rid), SMOKE_ES,
                                       note=f"smoke launch {launch} of {SMOKE_LAUNCHES} (never read)"))
    return steps


def plan_dprime() -> list:
    steps = _checks("P2D", "dprime")
    for kind, rid in KINDS.items():
        for es in DPRIME_ES:
            steps.append(_student_step(f"P2D.{kind}.es{es}", dprime_name(rid), dprime_row(rid), es,
                                       note="d-prime row (report only; paired with c1 RTSW + TDR RT9 es1-2)"))
    return steps


def plan_bridge(k: int) -> list:
    steps = _checks(f"P2B{k}", "bridge", k)
    rows = bridge_rows()
    if k == 0:
        todo, how = rows, "first launch"
    else:
        due, how = relaunch_due(k)
        if due is None:                                   # dry, no fresh judge: the conditional plan
            todo = rows
        else:
            todo = [b for b in rows if b["key"] in due]
    for b in todo:
        steps.append(_student_step(f"P2B{k}.{b['tag']}.{b['fam']}", bridge_name(b["tag"], k), b["row"], BRIDGE_ES,
                                   note=f"bridge {b['key']} launch {k + 1} of <= {BRIDGE_MAX_LAUNCHES} ({how})"))
    return steps


def relaunch_due(k: int) -> tuple:
    """``(set of row keys whose next launch is k, how)`` from a FRESH bridge_judge.json; ``(None, why)`` in a dry
    pass without one (every row shown, conditionally); a real run refuses a missing or stale judge."""
    doc = read_json(JUDGE_JSON)
    why = None
    if doc is None:
        why = f"no {JUDGE_JSON} (run --judge-bridge after the previous launch)"
    else:
        cur = {b["key"]: [str(s) for s in launch_stamps(b)] for b in bridge_rows()}
        rec = {key: r.get("launch_stamps") for key, r in (doc.get("rows") or {}).items()}
        if cur != rec:
            why = f"{JUDGE_JSON} is stale (its launch stamps differ from the slots now)"
    if why:
        if is_dry():
            return None, f"CONDITIONAL in this dry pass: {why}; a real run plans only the rows the judge marks due"
        raise ValueError(why)
    due = {d["key"] for d in doc.get("relaunch_due") or [] if int(d.get("next_k", -1)) == k}
    return due, f"due by {JUDGE_JSON.name} written {doc.get('at')}"


PLANS = {"p2smoke": plan_smoke, "p2dprime": plan_dprime, "p2bridge": lambda: plan_bridge(0),
         "p2bridge_r1": lambda: plan_bridge(1), "p2bridge_r2": lambda: plan_bridge(2)}


def plan_p2(block: str) -> list:
    steps = PLANS[block]()
    labels = [s.label for s in steps]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate step labels")
    slots = [s.slot for s in steps if s.slot]
    if len(slots) != len(set(slots)):
        raise RuntimeError("two rows share an output slot")
    if block in EXPECT_SLOTS and len(slots) != EXPECT_SLOTS[block]:
        raise RuntimeError(f"{block}: {len(slots)} rows, P2 step 3 pre-registers {EXPECT_SLOTS[block]}")
    if len(slots) > len(BRIDGE_CELLS) * len(BRIDGE_FAMS):
        raise RuntimeError(f"{block}: {len(slots)} rows")
    bad = rt_short.slot_problems(steps)
    if bad:
        raise ValueError("slot collision: " + "; ".join(bad))
    return steps


class _AnyStop:
    """``rt_queue`` reads ``M.STOP_FILE.exists()`` between steps: either STOP file stops a P2 block."""

    def __init__(self, *paths):
        self.paths = [Path(p) for p in paths]

    def exists(self) -> bool:
        return any(p.exists() for p in self.paths)

    def __str__(self) -> str:
        hit = [str(p) for p in self.paths if p.exists()]
        return " / ".join(hit or [str(p) for p in self.paths])

    __format__ = lambda self, spec: format(str(self), spec)  # noqa: E731


def install() -> None:
    """Patch the machinery (idempotent): ONLY the P2 blocks, their plan dispatch, the two STOP files."""
    if getattr(M, "_p2_installed", False):
        return
    M.BLOCKS = tuple(PLANS)
    M.plan = lambda block, *a, **k: plan_p2(block)
    M.STOP_FILE = _AnyStop(P2_STOP, M.STOP_FILE)
    # rt_queue stops a block when one Isaac step grows the DerivedDataCache by > 3 GB (the corruption signature).
    # The first launches after the step-2 move-aside rebuild a cold cache; if that warm-up is known to exceed 3 GB the
    # caller may raise the bound for this driver only (P2_DDC_GROWTH_MAX_GB); default unchanged.
    if os.environ.get("P2_DDC_GROWTH_MAX_GB"):
        M.DDC_GROWTH_MAX_GB = float(os.environ["P2_DDC_GROWTH_MAX_GB"])
    M._p2_installed = True


def print_slots(block: str) -> None:
    """The dry header: every slot, its name, checkpoint, expected stamp path and what the slot holds now."""
    steps = plan_p2(block)
    say(f"# P2 {block}: {sum(1 for s in steps if s.slot)} rows -> {P2_OUT}/lr (role student, c31000 "
        f"{C31000_SHA[:16]}..., {C31000_CK})")
    p = pin()
    for s in steps:
        if not s.slot:
            continue
        tp_out, name, sc, es = s.slot
        slot = M.slot_dir(tp_out, name, sc, es)
        done = M.complete_stamps(slot, pin=p) if p else []
        say(f"#   {s.label:<26} {name:<44} {sc:<24} es{es:<3} -> {slot}/<time>/  "
            f"[{len(done)} complete at the pin{' (skipped)' if done else ''}]")


# ============================================================================== content-keyed stamp index
def scan_seeds(roots, scenarios, seeds, shas) -> dict:
    """readout.scan restricted to the given seed dirs BEFORE any run_meta is opened (no es9+ stamp is read)."""
    idx: dict = collections.defaultdict(list)
    want = {f"seed{int(s):03d}" for s in seeds}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for nd in sorted(p for p in root.iterdir() if p.is_dir()):
            for sd in sorted(p for p in nd.iterdir() if p.is_dir()):
                if scenarios is not None and sd.name not in scenarios:
                    continue
                for ed in sorted(p for p in sd.iterdir() if p.is_dir() and p.name in want):
                    for st in sorted(p for p in ed.iterdir() if p.is_dir()):
                        if not readout._complete(st):
                            continue
                        rec = readout.stamp_record(st)
                        if shas is not None and rec["sha256"] not in shas:
                            continue
                        if rec["es"] is None or f"seed{int(rec['es']):03d}" != ed.name:
                            continue
                        idx[(rec["sha256"], rec["scenario"], int(rec["es"]))].append(rec)
    return dict(idx)


class P2Rows(readout.Rows):
    """``readout.Rows`` over a seed-restricted index whose ``found()`` reads a STUDENT sha with the row's args
    role-swapped to student (nothing else differs: env, manifest, scenario, the other args)."""

    def __init__(self, roots, seeds, shas, student_shas, dup: str = "earliest"):  # noqa: D107 (no super(): own scan)
        self.doc = S.load_rows()
        self.kind = {r["id"]: r for r in self.doc["kind_rows"]}
        self.ret = {r["id"]: r for r in self.doc["retention_rows"]}
        self.all = {**self.kind, **self.ret}
        self.dup = dup
        self.roots = [Path(r) for r in roots]
        self.seeds = {int(s) for s in seeds}
        self.student = set(student_shas)
        self.idx = scan_seeds(self.roots, {r["scenario"] for r in self.all.values()}, self.seeds, set(shas))

    def found(self, rid: str, es: int, sha: str) -> dict:
        if int(es) not in self.seeds:
            raise ValueError(f"{rid} {str(sha)[:8]}: es{es} outside this read's seeds {sorted(self.seeds)}")
        row = self.all[rid]
        if sha in self.student:
            row = dict(row, args=student_args(row["args"]))
        return readout.find(self.idx, row, es, sha, self.dup)


def stamp_build_problems(st, p=None) -> list:
    return M.build_mismatch(M.stamp_build(st), p if p is not None else pin())


# ============================================================================== --inputs-check
def p2_pin_problems() -> tuple[list, list]:
    """(refusals, warnings): results/p2/pin.json must exist (PLAN step 2) and carry results/pin.json's build."""
    base, p2 = pin(), read_json(P2_PIN)
    bad, warn = [], []
    if base is None:
        bad.append(f"no {M.PIN_PATH}")
    if p2 is None:
        (warn if is_dry() else bad).append(f"no {P2_PIN} (PLAN step 2 writes it before step 3)")
    elif base is not None:
        if p2.get("schema") != "rt_pin_v1":
            bad.append(f"{P2_PIN} is not a build_check.py pin (schema {p2.get('schema')!r}, want rt_pin_v1)")
        for k in ("code_sha", "eval_sha"):
            if p2.get(k) != base.get(k):
                bad.append(f"{P2_PIN} {k} {str(p2.get(k))[:12]} != {M.PIN_PATH.name} {str(base.get(k))[:12]}")
    if _CALLER_RT_TP_OUT and Path(_CALLER_RT_TP_OUT) != P2_OUT:
        bad.append(f"caller RT_TP_OUT {_CALLER_RT_TP_OUT} != {P2_OUT}")
    return bad, warn


def env_files_of(block: str) -> list:
    if block == "smoke" or block == "dprime":
        return sorted({str(S.OMNI / rows_by_id()[rid]["env_file"]) for rid in KINDS.values()})
    out = set()
    for b in bridge_rows():
        out.add(b["row"]["env_file"])
        if b["row"]["level_file"]:
            out.add(b["row"]["level_file"])
    return sorted(out)


def dprime_family_problems() -> tuple[list, list]:
    """c1's and TDR's recorded es1-2 stamps of each kind carry EXACTLY the d-prime row's composition with role
    teacher (tp_row env_file / level_file / manifest / scenario / env_seed / args / stripped_keys; run_meta env_vars ==
    readout's expected env; 100 episodes; build == pin; raw intact)."""
    R = P2Rows([RT_OUT / "lr"], DPRIME_ES, {C1_SHA, TDR_SHA}, set())
    bad, seen = [], []
    p = pin()
    for fam, (sha, prefix) in DPRIME_FAMILIES.items():
        for kind, rid in KINDS.items():
            planned = dprime_row(rid)
            want = {"env_file": str(S.OMNI / planned["env_file"]), "level_file": None,
                    "manifest": str(S.OMNI / planned["manifest"]), "scenario": planned["scenario"],
                    "args": teacher_args(planned["args"]), "stripped_keys": []}
            for es in DPRIME_ES:
                f = R.found(rid, es, sha)
                lab = f"{fam} {kind} es{es}"
                if f["status"] != "OK":
                    bad.append(f"{lab}: readout {f['status']} {f['problems'][:2]} {f['notes'][:1]}")
                    continue
                st = Path(f["stamp"])
                if not st.parent.parent.parent.name.startswith(prefix):
                    bad.append(f"{lab}: served from {st.parent.parent.parent.name}, not a {prefix}* row")
                tp = read_json(st / "tp_row.json", {})
                got = {k: tp.get(k) for k in want}
                got["args"] = [str(a) for a in got["args"] or []]
                if got != want:
                    bad.append(f"{lab}: tp_row composition {got} != planned (role teacher) {want}")
                if int(tp.get("env_seed", -1)) != es:
                    bad.append(f"{lab}: tp_row env_seed {tp.get('env_seed')}")
                meta = read_json(st / "run_meta.json", {})
                if int(meta.get("num_envs") or 0) != readout.KIND_ROW_EPISODES:
                    bad.append(f"{lab}: num_envs {meta.get('num_envs')} != {readout.KIND_ROW_EPISODES}")
                bad += [f"{lab}: {x}" for x in stamp_build_problems(st, p)]
                bad += [f"{lab}: {x}" for x in raw_problems(st)]
                seen.append(f"{lab}: {st}")
    return bad, seen


def inputs_check(block: str) -> int:
    bad, warn = p2_pin_problems()
    bad += c31000_problems()
    refused = set(refused_keys())
    for ef in env_files_of(block):
        keys = set(M.krabby_of(ef))
        hit = sorted(keys & refused)
        if hit:
            bad.append(f"{ef} carries student-refused runner keys {hit} (no TP_STRIP_KEYS here)")
    info = []
    if block == "dprime":
        b, info = dprime_family_problems()
        bad += b
    if block.startswith("bridge"):
        for b in bridge_rows():
            refs = recorded_stamps(b)
            if not refs:
                bad.append(f"bridge {b['key']}: no recorded c31000 stamp of this shape under {RSD_LR}")
            for r in refs:
                bad += [f"bridge {b['key']}: {x}" for x in raw_problems(r)]
            info.append(f"bridge {b['key']}: {len(refs)} recorded stamp(s): {[str(r) for r in refs]}")
    for w in warn:
        say(f"P2_INPUTS WARN {w}")
    for i in info:
        say(f"P2_INPUTS INFO {i}")
    say(f"P2_INPUTS {'OK' if not bad else 'BAD'} block={block}" + (f" -- {'; '.join(bad)}" if bad else ""))
    return 0 if not bad else 1


# ============================================================================== classflip / wave_diff on shadows
def normalise_meta(o, path: str = "", changed=None):
    changed = [] if changed is None else changed
    if isinstance(o, str):
        n = M.normalise_roots(o)
        if n != o:
            changed.append(path)
        return n, changed
    if isinstance(o, dict):
        return {k: normalise_meta(v, f"{path}.{k}" if path else str(k), changed)[0] for k, v in o.items()}, changed
    if isinstance(o, list):
        return [normalise_meta(v, f"{path}[{i}]", changed)[0] for i, v in enumerate(o)], changed
    return o, changed


def shadow(st: Path, tmp: Path, tag: str) -> tuple[Path, list]:
    d = Path(tmp) / tag
    d.mkdir(parents=True)
    meta, changed = normalise_meta(json.loads((Path(st) / "run_meta.json").read_text()))
    (d / "run_meta.json").write_text(json.dumps(meta, indent=1))
    for f in Path(st).iterdir():
        if f.name != "run_meta.json":
            (d / f.name).symlink_to(f.resolve())
    return d, changed


def run_json(script: Path, *args) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_", "KIT_"))}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    p = subprocess.run([M.PY, "-I", "-B", str(script), *map(str, args)], capture_output=True, text=True, env=env)
    last = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not last:
        raise RuntimeError(f"{script.name} {args[:2]} rc {p.returncode}: {(p.stderr or p.stdout)[-400:]}")
    return json.loads(last[-1])


def wave0_envs(st: Path) -> list | None:
    meta = json.loads((Path(st) / "run_meta.json").read_text())
    if not meta.get("suite"):
        return None
    return sorted(int(r["env_index"]) for r in meta.get("env_index_map") or [] if int(r.get("wave", 0)) == 0)


def pair_verdict(new: Path, ref: Path, tmp: Path, tag: str) -> dict:
    """classflip_check.py (and wave_diff.py on a suite row) on normalised shadows of ``new`` / ``ref``."""
    xs, cx = shadow(new, tmp, f"{tag}_x")
    ys, cy = shadow(ref, tmp, f"{tag}_y")
    cf = run_json(CLASSFLIP, xs, ys, "--period", PERIOD)
    bitwise = cf.get("verdict") == "BITWISE-EQUAL"
    common = bool(cf.get("cfg_identical") and cf.get("dose_identical") and not cf.get("structural_mismatch"))
    out = {"new": str(new), "ref": str(ref), "bitwise": bitwise, "classflip_verdict_raw": cf.get("verdict"),
           "cfg_identical": cf.get("cfg_identical"), "cfg_differs": cf.get("cfg_differs"),
           "dose_identical": cf.get("dose_identical"), "same_code": cf.get("same_code"),
           "structural_mismatch": cf.get("structural_mismatch"), "differing_episodes": cf.get("differing_episodes"),
           "earliest_step": cf.get("earliest_step"), "earliest_keys": cf.get("earliest_keys"),
           "earliest_envs": (cf.get("earliest_envs") or [])[:20], "earliest_max_dact": cf.get("earliest_max_dact"),
           "normalised_keys": {"new": sorted(set(cx)), "ref": sorted(set(cy))}}
    if bitwise:
        out["signature"] = True
        return out
    if wave0_envs(new) is not None:
        wd = run_json(WAVE_DIFF, xs, ys, "--period", PERIOD)
        w0 = (wd.get("waves") or {}).get("0") or {}
        out["wave0"] = w0
        if not w0:
            sig = False
        elif int(w0.get("differ", 0)) == 0:
            sig = True                                      # wave 0 bit-equal: every difference is a later one
        else:
            sig = (list(w0.get("earliest_keys") or []) == ["actions"] and w0.get("earliest_step") is not None
                   and int(w0["earliest_step"]) % PERIOD == 0 and set(w0.get("step0_keys") or []) <= {"actions"})
        out["judged_on"] = "wave 0 (RSD CHARTER section 4, 2026-11-08 23:06)"
    else:
        sig = bool(cf.get("step0_state_identical") and cf.get("earliest_action_only_at_depth_frame"))
        out["judged_on"] = "whole stamp (single wave)"
    out["signature"] = bool(common and sig)
    return out


def step0_actions(st: Path, envs=None) -> dict:
    import numpy as np  # noqa: PLC0415

    eps = [int(e["env_index"]) for e in json.loads((Path(st) / "scenario_metrics.json").read_text())["episodes"]]
    if envs is not None:
        eps = [e for e in eps if e in set(envs)]
    out = {}
    for e in eps:
        with np.load(Path(st) / "raw" / f"episode_{e:02d}.npz") as z:
            out[e] = np.ascontiguousarray(z["actions"][0]).tobytes()
    return out


def fp_refs(b: dict) -> list:
    """The step-0 reference set of a bridge row: every recorded c31000 stamp of the same FAMILY (scenario, es1, L090
    payload, student role) in BOTH bridge cells, as ``(stamp, cell_tag)``. Step 0 is dose-independent (the x0.7 rod
    speed acts after the first action; the recorded x0.7 stamps are bit-equal at step 0 to the nominal Z / Z_D1 /
    SEL_D1 class in all 100 wave-0 envs, RSD CHARTER 2026-11-08 23:26), so the nominal L090 stamps hold the other
    recorded class of the same policy and seed for the x0.7 rows too."""
    out = []
    for o in bridge_rows():
        if o["fam"] != b["fam"]:
            continue
        if o["row"]["scenario"] != b["row"]["scenario"]:
            raise ValueError(f"{o['key']} scenario {o['row']['scenario']} != {b['key']} {b['row']['scenario']}")
        out += [(Path(p), o["tag"]) for p in recorded_stamps(o)]
    return out


def fingerprint(new: Path, same_shape: list, refs: list) -> dict:
    """Step-0 actions of ``new`` in the class envs (wave 0 on suite) against the step-0 classes of ``refs``
    (:func:`fp_refs`: ``(stamp, cell_tag)``, both bridge cells). Class envs = the envs where the references disagree
    at step 0; the references are grouped into step-0 classes there. Per class: whether ``new`` equals it in every
    class env (``None`` -- never True -- when there are no class envs) and whether it holds a stamp of the row's own
    shape (``same_shape``). ``equals_other_recorded_class``: True / False when exactly the same-shape stamps cover
    one class (the x0.7 rows) -- does ``new`` equal a recorded class the same-shape stamps are NOT in; ``None`` when
    the same-shape stamps cover every class (the L090 rows: no 'other' class; ``equals_class`` says which one ``new``
    equals) or no step-0 class split is recorded."""
    envs = wave0_envs(new)
    a_new = step0_actions(new, envs)
    keys = sorted(a_new)
    a_ref = {str(r): step0_actions(r, envs) for r, _t in refs}
    cell_of = {str(r): t for r, t in refs}
    shape = {str(p) for p in same_shape}
    env_mismatch = sorted(r for r, a in a_ref.items() if sorted(a) != keys)
    cls = [e for e in keys if len({a.get(e) for a in a_ref.values()}) > 1]
    groups: dict = {}
    for r, a in a_ref.items():
        groups.setdefault(tuple(a.get(e) for e in cls), []).append(r)
    classes = []
    for i, (sig, members) in enumerate(groups.items()):
        eq = [sig[j] == a_new[e] for j, e in enumerate(cls)]
        classes.append({"id": i, "stamps": sorted(members), "cells": sorted({cell_of[m] for m in members}),
                        "has_same_shape_stamp": any(m in shape for m in members),
                        "new_equal_in_class_envs": (all(eq) if cls else None), "n_equal": sum(eq)})
    same_ids = [c["id"] for c in classes if c["has_same_shape_stamp"]]
    eq_ids = [c["id"] for c in classes if c["new_equal_in_class_envs"]]
    if not cls:
        other, other_why = None, "no step-0 class split recorded (every reference equal at step 0): nothing to compare"
    elif len(same_ids) == len(classes):
        other, other_why = None, "every recorded class holds a same-shape stamp (no 'other' class); see equals_class"
    else:
        other = any(c["id"] in eq_ids for c in classes if not c["has_same_shape_stamp"])
        other_why = (f"same-shape class {same_ids}; other recorded class(es) "
                     f"{[c['id'] for c in classes if not c['has_same_shape_stamp']]}")
    diff_one = None
    if len(classes) == 1:                                   # no split: where the new launch leaves the one class
        a0 = next(iter(a_ref.values()))
        diff_one = [e for e in keys if a0.get(e) != a_new[e]][:60]
    return {"class_envs": cls[:60], "n_class_envs": len(cls),
            "class_envs_rule": "envs where the step-0 references (both bridge cells, same family) disagree",
            "envs_compared": "wave 0" if envs is not None else "all", "n_envs_compared": len(keys),
            "references": {r: cell_of[r] for r in sorted(a_ref)}, "reference_env_set_mismatch": env_mismatch,
            "classes": classes, "same_shape_class": same_ids, "equals_class": (eq_ids[0] if eq_ids else None),
            "equals_recorded_class": (bool(eq_ids) if cls else None),
            "equals_other_recorded_class": other, "equals_other_why": other_why,
            "envs_new_differs_from_the_one_class": diff_one}


def fp_word(fp: dict | None) -> str:
    if not fp:
        return "n/a"
    v = fp.get("equals_other_recorded_class")
    return {True: "yes", False: "no", None: "n/a"}[v] + (f" (equals class {fp.get('equals_class')})"
                                                         if fp.get("equals_class") is not None else "")


# ============================================================================== the bridge judge
def recorded_stamps(b: dict) -> list:
    """Every complete c31000 stamp of the row's SHAPE under rod_speed_dr/lr (content-keyed; never by name)."""
    row = b["row"]
    want_env = S.expected_env(row)
    want_args = args_pairs(row["args"])
    idx = scan_seeds([RSD_LR], {row["scenario"]}, (BRIDGE_ES,), {C31000_SHA})
    out = []
    for rec in idx.get((C31000_SHA, row["scenario"], BRIDGE_ES), []):
        tp = rec["tp"]
        if (rec["role"] == "student" and rec["env_vars"] == want_env and args_pairs(tp.get("args") or []) == want_args
                and not tp.get("stripped_keys") and readout._rc_ok(tp)):
            out.append(Path(rec["stamp"]))
    return sorted(out, key=lambda p: p.name)


def launch_stamps(b: dict) -> list:
    """The complete P2 launches of a bridge row at the pinned build, in launch order (P2B_*, _R1, _R2)."""
    p = pin()
    out = []
    for k in range(BRIDGE_MAX_LAUNCHES):
        out += M.complete_stamps(M.slot_dir(P2_OUT, bridge_name(b["tag"], k), b["row"]["scenario"], BRIDGE_ES), pin=p)
    return out


def launch_problems(st: Path, b: dict) -> list:
    meta = read_json(Path(st) / "run_meta.json", {})
    tp = read_json(Path(st) / "tp_row.json", {})
    bad = []
    if meta.get("checkpoint_sha256") != C31000_SHA:
        bad.append(f"run_meta sha {str(meta.get('checkpoint_sha256'))[:8]}")
    if meta.get("role") != "student":
        bad.append(f"run_meta role {meta.get('role')!r}")
    if dict(meta.get("env_vars") or {}) != S.expected_env(b["row"]):
        bad.append(f"env_vars differ {readout.env_diff(dict(meta.get('env_vars') or {}), S.expected_env(b['row']))[:6]}")
    if args_pairs(tp.get("args") or []) != args_pairs(b["row"]["args"]):
        bad.append(f"tp_row args {tp.get('args')}")
    if tp.get("stripped_keys"):
        bad.append(f"tp_row stripped {tp.get('stripped_keys')}")
    bad += stamp_build_problems(st)
    bad += raw_problems(st)
    return bad


def judge() -> dict:
    rows = {}
    norm_keys: set = set()
    with tempfile.TemporaryDirectory(prefix="p2_bridge_norm_") as tmp:
        for b in bridge_rows():
            refs = recorded_stamps(b)
            launches = launch_stamps(b)
            r = {"tag": b["tag"], "cell": b["cell"], "fam": b["fam"], "scenario": b["row"]["scenario"],
                 "es": BRIDGE_ES, "policy_sha256": C31000_SHA, "row": b["row"],
                 "recorded": [str(x) for x in refs], "launch_stamps": [str(x) for x in launches], "launches": []}
            if not refs:
                r.update(status="STOP", why="no recorded stamp of this shape")
            elif not launches:
                r.update(status="INCOMPLETE", why="no complete launch at the pinned build")
            else:
                for i, st in enumerate(launches):
                    lp = launch_problems(st, b)
                    pairs = [] if lp else [pair_verdict(st, ref, Path(tmp), f"{b['tag']}_{b['fam']}_{i}_{j}")
                                           for j, ref in enumerate(refs)]
                    for pv in pairs:
                        norm_keys |= set(pv["normalised_keys"]["new"]) | set(pv["normalised_keys"]["ref"])
                    r["launches"].append({"stamp": str(st), "problems": lp, "pairs": pairs,
                                          "bitwise_match": [pv["ref"] for pv in pairs if pv["bitwise"]],
                                          "signature_vs_all": bool(pairs) and all(pv["signature"] for pv in pairs)})
                last = r["launches"][-1]
                hit = [(L["stamp"], m) for L in r["launches"] for m in L["bitwise_match"]]
                if any(L["problems"] for L in r["launches"]):
                    r.update(status="STOP", why=f"a launch is not this row's stamp: "
                                                f"{[L['problems'] for L in r['launches'] if L['problems']][0][:3]}")
                elif hit:
                    r.update(status="PASS", why=f"bitwise match: {hit[0][0]} == {hit[0][1]}", match=hit)
                elif last["signature_vs_all"] and len(launches) < BRIDGE_MAX_LAUNCHES:
                    r.update(status="RELAUNCH_DUE", next_k=len(launches),
                             why=f"launch {len(launches)} shows the class signature against every recorded stamp "
                                 f"and matches none bit for bit")
                elif last["signature_vs_all"]:
                    r.update(status="STOP", why=f"no bitwise match after {len(launches)} launches (class signature "
                                                f"each time)")
                else:
                    r.update(status="STOP", why="a code-path difference (no class signature against "
                                                f"{[pv['ref'] for pv in last['pairs'] if not pv['signature']]})")
                fpr = fp_refs(b)
                r["step0_references"] = {str(p): t for p, t in fpr}
                if r["status"] != "PASS" and not any(L["problems"] for L in r["launches"]):
                    r["step0_fingerprint"] = fingerprint(Path(last["stamp"]), refs, fpr)
                    r["step0_fingerprint_by_launch"] = {L["stamp"]: fingerprint(Path(L["stamp"]), refs, fpr)
                                                        for L in r["launches"]}
                    if r["status"] == "STOP":
                        fp = r["step0_fingerprint"]
                        r["equals_other_recorded_class"] = fp["equals_other_recorded_class"]
                        r["why"] += (f"; step 0 of the last launch in the class envs {fp['class_envs'][:12]}: "
                                     f"equals_other_recorded_class={fp['equals_other_recorded_class']} "
                                     f"(equals_class {fp['equals_class']}; {fp['equals_other_why']})")
                elif r["status"] == "PASS":
                    r["step0_fingerprint"] = fingerprint(Path(hit[0][0]), refs, fpr)
            rows[f"{b['tag']}.{b['fam']}"] = r
    due = [{"key": k, "next_k": r["next_k"], "name": bridge_name(r["tag"], r["next_k"])}
           for k, r in rows.items() if r["status"] == "RELAUNCH_DUE"]
    stop = [k for k, r in rows.items() if r["status"] == "STOP"]
    if any(r["status"] == "INCOMPLETE" for r in rows.values()):
        verdict = "INCOMPLETE"
    elif due:
        verdict = "RELAUNCH"
    elif stop:
        verdict = "STOP"
    else:
        verdict = "PASS"
    return {"schema": "p2_bridge_judge_v1", "at": M.now_iso(), "verdict": verdict, "stop_rows": stop,
            "relaunch_due": due, "pin": {k: (pin() or {}).get(k) for k in ("code_sha", "eval_sha")},
            "policy": {"name": "c31000", "sha256": C31000_SHA, "checkpoint": str(C31000_CK)},
            "normalisation": {"rule": NORMALISATION_RULE, "keys_changed": sorted(norm_keys)},
            "rule": "RSD CHARTER section 4 (amended 2026-11-08 20:12; wave 0 per 23:06): PASS = any launch bitwise "
                    "equal to any recorded stamp of the same shape; RELAUNCH (up to 2) on the class signature "
                    "against every recorded stamp; otherwise STOP (notify and ask; the distill runs regardless)",
            "rows": rows}


def bridge_sentinels(doc: dict) -> list:
    out = []
    if doc["stop_rows"]:
        eq = " ".join(f"{k}:equals_other_recorded_class={fp_word(doc['rows'][k].get('step0_fingerprint'))}"
                      for k in doc["stop_rows"])
        out.append(f"P2_BRIDGE STOP rows={','.join(doc['stop_rows'])} {eq} (notify and ask; the distill runs) "
                   f"judge={JUDGE_JSON}")
    if doc["verdict"] == "RELAUNCH":
        out.append("P2_BRIDGE RELAUNCH " + " ".join(f"{d['key']}->{d['name']}" for d in doc["relaunch_due"]))
    elif doc["verdict"] in ("PASS", "INCOMPLETE"):
        out.append(f"P2_BRIDGE {doc['verdict']} " + " ".join(f"{k}={r['status']}" for k, r in doc["rows"].items()))
    return out


def cmd_judge() -> int:
    doc = judge()
    M.write_json(JUDGE_JSON, doc)
    for ln in bridge_sentinels(doc):
        say(ln)
    say(f"# wrote {JUDGE_JSON}")
    if doc["verdict"] == "RELAUNCH":
        return JUDGE_RC["RELAUNCH"]
    return JUDGE_RC[doc["verdict"]]


# ============================================================================== smoke + d-prime reads
def smoke_read() -> dict:
    p = pin()
    roots = [P2_OUT / "lr"]
    idx_all = scan_seeds(roots, set(KINDS.values()), (SMOKE_ES,), {C31000_SHA})
    out = {"rows": [], "pairs": [], "status": "PASS"}
    if True:                                                # (one block, kept flat for the diff-free reading)
        for kind, rid in KINDS.items():
            row = smoke_row(rid)
            stamps = []
            for launch in range(1, SMOKE_LAUNCHES + 1):
                name = smoke_name(rid, launch)
                slot = M.slot_dir(P2_OUT, name, row["scenario"], SMOKE_ES)
                done = M.complete_stamps(slot)
                rec = {"kind": kind, "launch": launch, "name": name, "slot": str(slot), "problems": []}
                if not done:
                    rec.update(status="MISSING", problems=["no complete stamp (tp_row rc 0)"])
                    out["rows"].append(rec)
                    out["status"] = "FAIL"
                    stamps.append(None)
                    continue
                st = done[-1]
                rec["stamp"] = str(st)
                meta = read_json(st / "run_meta.json", {})
                pr = []
                if meta.get("role") != "student":
                    pr.append(f"run_meta role {meta.get('role')!r}")
                if meta.get("checkpoint_sha256") != C31000_SHA:
                    pr.append(f"sha {str(meta.get('checkpoint_sha256'))[:8]}")
                pr += stamp_build_problems(st, p)
                pr += raw_problems(st)
                if not readout._complete(st):
                    pr.append("readout._complete False")
                idx = {k: [c for c in v if Path(c["stamp"]) == st] for k, v in idx_all.items()}
                f = readout.find(idx, row, SMOKE_ES, C31000_SHA)
                rec["readout_find"] = {"status": f["status"], "problems": f["problems"], "notes": f["notes"]}
                if f["status"] != "OK":
                    pr.append(f"readout.find {f['status']} {f['problems'][:2]}")
                else:
                    try:
                        t = readout.episode_table(st, row["cross_r"])
                        rec["parse"] = {"n": len(t["completion"]), "completion": readout.share(t["completion"]),
                                        "success": readout.share(t["success"]), "cross_r": row["cross_r"]}
                        if len(t["completion"]) != SMOKE_EPISODES:
                            pr.append(f"{len(t['completion'])} episodes != {SMOKE_EPISODES}")
                    except Exception as exc:  # noqa: BLE001
                        pr.append(f"episode_table: {type(exc).__name__}: {exc}")
                cs = M.stamp_build(st) or {}
                rec.update(code_sha=cs.get("code_sha"), eval_sha=cs.get("eval_sha"), role=meta.get("role"),
                           problems=pr, status="OK" if not pr else "BAD")
                if pr:
                    out["status"] = "FAIL"
                out["rows"].append(rec)
                stamps.append(st)
            if all(stamps) and len(stamps) == 2:
                cf = run_json(CLASSFLIP, stamps[0], stamps[1], "--period", PERIOD)
                ok = cf.get("verdict") in ("BITWISE-EQUAL", "CLASS-FLIP")
                out["pairs"].append({"kind": kind, "x": str(stamps[0]), "y": str(stamps[1]),
                                     "verdict": cf.get("verdict"), "ok": ok,
                                     "cfg_differs": cf.get("cfg_differs"), "same_code": cf.get("same_code"),
                                     "differing_episodes": cf.get("differing_episodes"),
                                     "earliest_step": cf.get("earliest_step"), "earliest_keys": cf.get("earliest_keys")})
                if not ok:
                    out["status"] = "FAIL"
            else:
                out["pairs"].append({"kind": kind, "verdict": None, "ok": False, "why": "a launch is missing"})
                out["status"] = "FAIL"
    out["rule"] = ("rc 0, role student, sha c31000, build == results/pin.json, readout _complete + find + "
                   "episode_table OK, raw intact; the two launches BITWISE-EQUAL or CLASS-FLIP (classflip_check.py, "
                   "same pin)")
    return out


def cmd_check_smoke() -> int:
    s = smoke_read()
    bad = [f"{r['kind']} L{r['launch']}: {r['problems'][:2]}" for r in s["rows"] if r.get("problems")]
    bad += [f"{q['kind']} pair: {q.get('verdict') or q.get('why')}" for q in s["pairs"] if not q["ok"]]
    say(f"P2_SMOKE {s['status']} " + " ".join(f"{q['kind']}={q.get('verdict')}" for q in s["pairs"])
        + (f" -- {'; '.join(bad)}" if bad else ""))
    return 0 if s["status"] == "PASS" else 1


def dprime_read() -> dict:
    p = pin()
    R = P2Rows([RT_OUT / "lr", P2_OUT / "lr"], DPRIME_ES, {C1_SHA, TDR_SHA, C31000_SHA}, {C31000_SHA})
    kinds, sent = {}, []
    for kind, rid in KINDS.items():
        k = {"row": rid, "env_seeds": list(DPRIME_ES), "cross_r": R.all[rid]["cross_r"]}
        ok = True
        for lab, sha in (("c1", C1_SHA), ("tdr", TDR_SHA)):
            pt = readout.pair_tables(R, rid, DPRIME_ES, sha, C31000_SHA, "success")
            stamps = {side: [{"es": f["es"], "status": f["status"], "stamp": f["stamp"], "problems": f["problems"][:3]}
                             for f in pt["stamps"][side]] for side in ("new", "ref")}
            k[f"stamps_{lab}"] = stamps["new"]
            k["stamps_c31000"] = stamps["ref"]
            if pt["status"] != "OK":
                k[f"d_{lab}_minus_c31000"] = {"status": pt["status"], "why": pt.get("why")}
                ok = False
                continue
            served = [f["stamp"] for f in pt["stamps"]["new"] + pt["stamps"]["ref"]]
            off = [s for s in served if stamp_build_problems(s, p)]
            if off:
                k[f"d_{lab}_minus_c31000"] = {"status": "INVALID", "why": f"stamps off the pin: {off}"}
                ok = False
                continue
            g = readout.g_line(pt["a"], pt["b"], f"{lab} - c31000 {rid}")
            k[f"d_{lab}_minus_c31000"] = {"status": "OK", "diff": g["diff"], "se": g["se"], "ci95": g["ci95"],
                                          "n": g["n"], "le_noise": g["le_noise"],
                                          "statistic": "readout.g_line(new=" + lab + ", ref=c31000) on success; "
                                                       "gain / regress flags not used (report only)"}
            k[lab] = g["new"]
            k["c31000"] = g["tdr"]
        if ok:
            k["cannot_gain"] = bool(k["c31000"] >= k["c1"] - DPRIME_MARGIN - readout.EPS)
        k["status"] = "OK" if ok else "INCOMPLETE"
        kinds[kind] = k
    if all(k["status"] == "OK" for k in kinds.values()):
        cg = sorted(kd for kd, k in kinds.items() if k["cannot_gain"])
        if len(cg) == len(kinds):
            trig = "TRIGGER"
            sent.append(f"P2_DPRIME TRIGGER c31000 >= c1 - {DPRIME_MARGIN} on both kinds "
                        + " ".join(f"{kd}: c31000 {k['c31000']:.3f} c1 {k['c1']:.3f}" for kd, k in kinds.items())
                        + " (notify and ask; the distill runs)")
        else:
            trig = "CANNOT_GAIN" if cg else "NONE"
            for kd in cg:
                sent.append(f"P2_DPRIME CANNOT_GAIN {kd} c31000 {kinds[kd]['c31000']:.3f} >= c1 "
                            f"{kinds[kd]['c1']:.3f} - {DPRIME_MARGIN} (recorded; continue)")
            if not cg:
                sent.append("P2_DPRIME NONE " + " ".join(f"{kd}: c1-c31000 {k['d_c1_minus_c31000']['diff']:+.3f}"
                                                         for kd, k in kinds.items()))
    else:
        trig = "INCOMPLETE"
        sent.append("P2_DPRIME INCOMPLETE " + " ".join(f"{kd}={k['status']}" for kd, k in kinds.items()))
    return {"kinds": kinds, "trigger": trig, "sentinels": sent,
            "rule": f"pre-registered (P2 settled 7, Q3): c31000 >= c1 - {DPRIME_MARGIN} on BOTH kinds -> TRIGGER "
                    "(notify and ask, distill runs); on one kind -> that kind cannot gain (recorded, continue); "
                    "report only, never pooled into the read",
            "statistic": "success = complete AND crossed (readout.episode_table at rows_rt cross_r), paired on "
                         "(es, env_index) via readout.pair_tables; d' via readout.g_line (paired bootstrap SE)"}


def cmd_read() -> int:
    doc = {"schema": "p2_step3_v1", "at": M.now_iso(),
           "pin": {k: (pin() or {}).get(k) for k in ("code_sha", "eval_sha")}, "p2_pin": read_json(P2_PIN),
           "policies": {"c31000": {"sha256": C31000_SHA, "checkpoint": str(C31000_CK), "role": "student"},
                        "c1": {"sha256": C1_SHA, "role": "teacher"}, "tdr": {"sha256": TDR_SHA, "role": "teacher"}},
           "roots": {"p2": str(P2_OUT / "lr"), "rt": str(RT_OUT / "lr"), "rsd": str(RSD_LR)}}
    sent = []
    try:
        doc["smoke"] = smoke_read()
    except Exception as exc:  # noqa: BLE001
        doc["smoke"] = {"status": "ERROR", "why": f"{type(exc).__name__}: {exc}"}
    sent.append(f"P2_SMOKE {doc['smoke']['status']}")
    try:
        doc["dprime"] = dprime_read()
        sent += doc["dprime"]["sentinels"]
    except Exception as exc:  # noqa: BLE001
        doc["dprime"] = {"trigger": "ERROR", "why": f"{type(exc).__name__}: {exc}"}
        sent.append(f"P2_DPRIME ERROR {type(exc).__name__}: {exc}")
    try:
        prev = read_json(JUDGE_JSON)
        cur = {b["key"]: [str(s) for s in launch_stamps(b)] for b in bridge_rows()}
        if prev and {k: r.get("launch_stamps") for k, r in (prev.get("rows") or {}).items()} == cur:
            doc["bridge"] = dict(prev, reused_from=str(JUDGE_JSON))
        else:
            doc["bridge"] = judge()
        sent += bridge_sentinels(doc["bridge"])
    except Exception as exc:  # noqa: BLE001
        doc["bridge"] = {"verdict": "ERROR", "why": f"{type(exc).__name__}: {exc}"}
        sent.append(f"P2_BRIDGE ERROR {type(exc).__name__}: {exc}")
    doc["sentinels"] = sent
    M.write_json(STEP3_JSON, doc)
    for ln in sent:
        say(ln)
    say(f"P2_STEP3 READ wrote {STEP3_JSON}")
    return 0


# ============================================================================== main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--block", choices=("smoke", "dprime", "bridge"), default=None)
    ap.add_argument("--relaunch", type=int, choices=(1, 2), default=None, help="bridge only: relaunch R1 / R2")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="from_label", default=None)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--inputs-check", action="store_true")
    ap.add_argument("--check-smoke", action="store_true")
    ap.add_argument("--judge-bridge", action="store_true")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args(argv)
    if _CALLER_RT_TP_OUT and Path(_CALLER_RT_TP_OUT) != P2_OUT:
        say(f"P2_STEP3 REFUSED caller RT_TP_OUT={_CALLER_RT_TP_OUT} (this driver writes only {P2_OUT})")
        return 2
    if a.relaunch and a.block != "bridge":
        say("P2_STEP3 REFUSED --relaunch needs --block bridge")
        return 2
    MODE["dry"] = a.dry
    if a.dry:
        os.environ["P2_DRY"] = "1"                         # the dry pass's CPU children see it (rt_queue passes env)
    if a.inputs_check:
        blk = a.block if not a.relaunch else f"bridge_r{a.relaunch}"
        return inputs_check(blk or "smoke")
    if a.check_smoke:
        return cmd_check_smoke()
    if a.judge_bridge:
        return cmd_judge()
    if a.read:
        return cmd_read()
    if a.block is None and not a.dry:
        ap.error("one of --block / --dry / --inputs-check / --check-smoke / --judge-bridge / --read")
    install()
    blocks = ([BLOCKS["bridge"] if not a.relaunch else BLOCKS[f"bridge_r{a.relaunch}"]] if a.block == "bridge"
              else [BLOCKS[a.block]] if a.block else list(BLOCKS.values()))
    rc_all = 0
    for blk in blocks:
        rest = ["--block", blk]
        if a.dry:
            rest.append("--dry")
        if a.list:
            rest.append("--list")
        if a.from_label:
            rest += ["--from", a.from_label]
        for o in a.only or []:
            rest += ["--only", o]
        if a.dry or a.list:
            try:
                print_slots(blk)
            except (ValueError, RuntimeError, OSError) as exc:
                say(f"# P2 {blk}: plan refused: {exc}")
        else:
            why = c31000_problems()
            if why:
                say(M.done_line(blk, 1, f"(c31000 identity: {'; '.join(why)})"))
                say(f"P2_STEP3 REFUSED block={blk} rc={REFUSED_RC} (c31000 identity; no row launched)")
                return REFUSED_RC
        if a.dry or a.list:
            rc = Q.main(rest)
        else:
            rc = run_block(blk, rest)
        rc_all = rc_all or rc
    return rc_all


def run_block(blk: str, rest: list) -> int:
    """``rt_queue.main`` for one real block, watching which rows it launched (``run_child`` with a row's log) and its
    last ``RTQ_DONE`` line: rc 0 / 1 pass through when a row launched; :data:`REFUSED_RC` when it ended non-zero with no
    row launched; :data:`HALT_RC` on the DDC-growth or guard stop. Prints ``P2_STEP3 REFUSED|HALT`` at column 0."""
    try:
        row_labels = {s.label for s in plan_p2(blk) if s.slot}
    except (ValueError, RuntimeError, OSError):
        row_labels = set()                                  # rt_queue refuses the same plan and says why
    seen = {"rows": [], "done": None}
    child0, say0 = Q.run_child, Q.say

    def child(argv, env, log):
        if log is not None and Path(log).stem in row_labels:
            seen["rows"].append(Path(log).stem)
        return child0(argv, env, log)

    def say_(line: str) -> None:
        if line.startswith("RTQ_DONE "):
            seen["done"] = line
        say0(line)

    Q.run_child, Q.say = child, say_
    try:
        rc = Q.main(rest)
    finally:
        Q.run_child, Q.say = child0, say0
    if rc == 0:
        return 0
    done = seen["done"] or ""
    if "DerivedDataCache grew" in done or "guard stopped the queue" in done:
        say(f"P2_STEP3 HALT block={blk} rc={HALT_RC} rows_launched={len(seen['rows'])} -- {done}")
        return HALT_RC
    if not seen["rows"]:
        say(f"P2_STEP3 REFUSED block={blk} rc={REFUSED_RC} (rt_queue rc {rc}; no row launched) -- {done}")
        return REFUSED_RC
    return rc


if __name__ == "__main__":
    sys.exit(main())
