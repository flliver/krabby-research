"""CHARTER §4 student class-flip test (as amended 2026-11-08 20:12) for two stamps X, Y of the same row.

    classflip_check.py X Y [--known K] [--period 5]

X, Y: stamp dirs (…/<scenario>/seedNNN/<time>) holding run_meta.json, code_state.json, scenario_metrics.json, raw/.
--known K: cross-pin bridge only: X must also equal K (an earlier launch of the same policy and row) bit for bit.
Arrays are compared as raw bytes, so "equal" means bit for bit. Prints one JSON summary line last.

Steps are episode-local and the verdict treats the stamp as one unit. A multi-wave (suite) row runs its waves one after
another in one process, so a flip in wave 0 shows at every later wave's own step 0 and this verdict reads CODE-PATH.
CHARTER section 4 (clarified 2026-11-08 23:06) judges such a row on wave 0: run wave_diff.py on the pair and apply the
conditions to its wave-0 line.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# time and outcome fields: listed, not judged. n_steps_per_env = steps each env actually ran; travel_yaw_by_hold =
# circular mean of the realized per-step travel yaw per hold (eval_crab_hex_gait.py ~1997)
VOLATILE = {"started_utc", "termination_reason", "n_steps_per_env", "travel_yaw_by_hold"}


def episodes(d: Path) -> list:
    return sorted(int(e["env_index"]) for e in json.loads((d / "scenario_metrics.json").read_text())["episodes"])


def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list) and all(not isinstance(x, (dict, list)) for x in o):
        out[p] = o
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}[{i}]"))
    else:
        out[p] = o
    return out


def rows_bytes(u: np.ndarray, n: int) -> np.ndarray:
    return np.ascontiguousarray(u[:n]).view(np.uint8).reshape(n, -1)


def first_diff(job):
    """(episode, status, step, keys, max |d action| at that step, (hip, knee) max |rod_speed| for X and Y)."""
    x, y, e = job
    a, b = np.load(x / "raw" / f"episode_{e:02d}.npz"), np.load(y / "raw" / f"episode_{e:02d}.npz")
    dose = tuple((float(np.abs(z["rod_speed"][:, :6]).max()), float(np.abs(z["rod_speed"][:, 6:12]).max()))
                 for z in (a, b))
    if set(a.files) != set(b.files):
        return e, "keys", None, sorted(set(a.files) ^ set(b.files)), None, dose
    best, which = None, []
    for k in a.files:
        u, v = a[k], b[k]
        if u.ndim == 0 or k == "dt":
            if u.tobytes() != v.tobytes() or u.dtype != v.dtype:
                return e, "scalar", None, [k], None, dose
            continue
        if u.dtype != v.dtype or u.shape[1:] != v.shape[1:]:
            return e, "shape", None, [k], None, dose
        n = min(len(u), len(v))
        neq = np.any(rows_bytes(u, n) != rows_bytes(v, n), axis=1) if n else np.zeros(0, bool)
        t = int(np.argmax(neq)) if neq.any() else (n if len(u) != len(v) else None)
        if t is None:
            continue
        if best is None or t < best:
            best, which = t, [k]
        elif t == best:
            which.append(k)
    if best is None:
        return e, "equal", None, [], None, dose
    da = None
    if "actions" in which and best < min(len(a["actions"]), len(b["actions"])):
        da = float(np.max(np.abs(a["actions"][best].astype(np.float64) - b["actions"][best].astype(np.float64))))
    return e, "diff", best, sorted(which), da, dose


def compare(x: Path, y: Path, pool) -> list:
    ex, ey = episodes(x), episodes(y)
    if ex != ey:
        raise SystemExit(f"episode sets differ: {len(ex)} vs {len(ey)}")
    return pool.map(first_diff, [(x, y, e) for e in ex])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("x", type=Path)
    ap.add_argument("y", type=Path)
    ap.add_argument("--known", type=Path)
    ap.add_argument("--period", type=int, default=5)
    a = ap.parse_args()
    out = {"x": str(a.x), "y": str(a.y)}

    # 1. cfg: run_meta (the effective cfg) and code_state
    mx, my = (flat(json.loads((d / "run_meta.json").read_text())) for d in (a.x, a.y))
    dk = sorted(k for k in set(mx) | set(my) if mx.get(k, "<absent>") != my.get(k, "<absent>"))
    judged = [k for k in dk if k.split(".")[0] not in VOLATILE]
    print(f"run_meta: {len(set(mx) | set(my))} keys; differ: {dk[:30]}")
    cx, cy = (json.loads((d / "code_state.json").read_text()) if (d / "code_state.json").exists() else {}
              for d in (a.x, a.y))
    code = {k: (cx.get(k), cy.get(k)) for k in ("code_sha", "eval_sha") if cx.get(k) != cy.get(k)}
    print(f"code_state differs: {code or 'no'}")
    out["cfg_identical"] = not judged
    out["cfg_differs"] = judged
    out["same_code"] = not code

    with Pool(8) as pool:
        res = compare(a.x, a.y, pool)
        # 2. dose: per-episode max |rod_speed| (hip, knee), X vs Y
        dose_bad = [r[0] for r in res if r[5][0] != r[5][1]]
        doses = sorted({r[5][0] for r in res} | {r[5][1] for r in res})
        print(f"dose (hip, knee) values seen: {doses[:6]}; episodes where X and Y differ: {dose_bad[:20]}")
        out["dose_identical"] = not dose_bad

        # 3. differences
        bad = [r for r in res if r[1] in ("keys", "scalar", "shape")]
        diff = [r for r in res if r[1] == "diff"]
        out["episodes"] = len(res)
        out["structural_mismatch"] = [(r[0], r[1], r[3]) for r in bad]
        out["differing_episodes"] = len(diff)
        if diff:
            t0 = min(r[2] for r in diff)
            at0 = [r for r in diff if r[2] == t0]
            keys0 = sorted({k for r in at0 for k in r[3]})
            out["earliest_step"] = t0
            out["earliest_keys"] = keys0
            out["earliest_envs"] = [r[0] for r in at0]
            out["earliest_max_dact"] = max((r[4] or 0.0) for r in at0)
            out["step0_state_identical"] = not (t0 == 0 and keys0 != ["actions"])
            out["earliest_action_only_at_depth_frame"] = keys0 == ["actions"] and t0 % a.period == 0
            kinds = {}
            for r in diff:
                kind = "action-only@depth" if r[3] == ["actions"] and r[2] % a.period == 0 else "other"
                kinds[kind] = kinds.get(kind, 0) + 1
            out["first_diff_kinds"] = kinds
            out["max_first_dact"] = max((r[4] or 0.0) for r in diff)
            for r in sorted(diff, key=lambda r: (r[2], r[0]))[:12]:
                print(f"  ep {r[0]:3d} first diff step {r[2]} keys {r[3][:6]} max|dact| {r[4]}")
        else:
            out.update(earliest_step=None, step0_state_identical=True, earliest_action_only_at_depth_frame=None)

        # 4. cross-pin clause
        if a.known is not None:
            rk = compare(a.x, a.known, pool)
            nk = [r[0] for r in rk if r[1] != "equal"]
            out["known"] = str(a.known)
            out["x_equals_known"] = not nk
            print(f"X vs known: episodes not bit-equal: {len(nk)} {nk[:20]}")

    if not diff and not bad:
        verdict = "BITWISE-EQUAL"
    elif (out["cfg_identical"] and out["dose_identical"] and not bad and out["step0_state_identical"]
          and out["earliest_action_only_at_depth_frame"]
          and (a.known is None and out["same_code"] or a.known is not None and out.get("x_equals_known"))):
        verdict = "CLASS-FLIP"
    else:
        verdict = "CODE-PATH"
    out["verdict"] = verdict
    print(json.dumps(out))


if __name__ == "__main__":
    main()
