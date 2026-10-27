# SPDX-License-Identifier: BSD-3-Clause
"""Merge per-source raw RSI banks (D6) into ``rsi_bank_omni.npz``, or re-check a merged bank.

Two modes:

``merge_rsi_banks.py <bank1.npz> <bank2.npz> ... --out rsi_bank_omni.npz [--per_source 2000] [--seed 0]``
    Loads each raw bank (produced by ``harvest_omni_rsi_bank.py``), asserts they all carry the
    SAME six-array + ``source`` + ``meta`` key set and the SAME joint count. ``harvest_omni_rsi_bank.py``
    only filters on upright / the AXIS-ALIGNED speed component / |wz| / episode step -- it does not
    bound the CROSS-axis component, so a raw bank (a real rollout's natural gait-cycle wobble) is
    NOT expected to pass the angle/speed/tag check below at 100%: this script FILTERS each raw bank
    to the per-frame-passing subset first (printing the pass/fail split -- a bank whose pass rate
    is degenerate, e.g. near 0, fails loudly), THEN subsamples ``--per_source`` frames from that
    filtered subset (seeded, without replacement; a filtered subset smaller than the target is
    taken whole), concatenates evenly across sources, shuffles (seeded), re-checks the shuffled
    result (must be 100% -- a violation here means a real bug, not source noise), and writes the
    merged bank plus a ``meta`` recording both input paths + sha256s, the sampling seed/counts, git
    HEAD and date. Prints per-source and total counts and the output file's sha256.

``merge_rsi_banks.py --check <bank.npz>``
    Verification 5: standalone re-check of an already-merged (or raw) bank -- no merging, no
    write. Re-asserts, per frame, that ``atan2(vy_b, vx_b)`` lies within +/-20 deg of the axis its
    ``source`` tag implies (0 -> 0 deg, the forward/+x convention; 1 -> +90 deg, the sideways/+y,
    left-leads convention) and that ``||v_b,xy|| in [0.15, 0.5]``. Exits non-zero (via the
    assertion) on any violation -- a bad bank fails loudly, not silently.

Both modes share ``_check_angle_speed``, so the SAME assertion is applied to a raw bank right
after harvest, to the merged bank right after the shuffle, and to whatever file the campaign's
RSI reset event will actually load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_BANK_KEYS = ("joint_pos", "joint_vel", "root_quat_w", "root_z", "root_lin_vel_b", "clock_phase")
_EXPECTED_KEYS = frozenset(_BANK_KEYS) | {"source", "meta"}
_AXIS_DEG_FOR_SOURCE = {0: 0.0, 1: 90.0}  # forward (+x) / sideways (+y, left-leads)
ANGLE_TOL_DEG = 20.0
SPEED_LO, SPEED_HI = 0.15, 0.5


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        keys = set(raw.files)
        assert keys == _EXPECTED_KEYS, (
            f"{path}: key set {sorted(keys)} != expected {sorted(_EXPECTED_KEYS)} "
            "(a bank the forward RSI reset event could load carries exactly these keys)"
        )
        return {k: raw[k] for k in raw.files}


def _angle_speed_mask(bank: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (angle_ok, speed_ok, dev_deg): does ``atan2(vy_b, vx_b)`` lie within
    +/-ANGLE_TOL_DEG of the axis `source` implies, and is ``||v_b,xy|| in [SPEED_LO, SPEED_HI]``."""
    v = bank["root_lin_vel_b"]
    src = bank["source"].astype(np.int64)
    unknown = np.setdiff1d(np.unique(src), np.array(list(_AXIS_DEG_FOR_SOURCE)))
    assert unknown.size == 0, f"unknown source tag(s) {unknown.tolist()} (only 0/1 defined)"
    axis_deg = np.vectorize(_AXIS_DEG_FOR_SOURCE.get)(src).astype(np.float64)
    vxy = v[:, :2].astype(np.float64)
    speed = np.linalg.norm(vxy, axis=1)
    angle = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    dev = np.abs(((angle - axis_deg + 180.0) % 360.0) - 180.0)
    angle_ok = dev <= ANGLE_TOL_DEG
    speed_ok = (speed >= SPEED_LO) & (speed <= SPEED_HI)
    return angle_ok, speed_ok, dev, speed


def _check_angle_speed(bank: dict[str, np.ndarray]) -> dict:
    """Per-frame angle/speed/tag summary (see ``_angle_speed_mask``). Returns a summary dict;
    caller asserts ``ok`` -- used both to report a raw bank's pass rate and, where every frame
    MUST pass (the final merged bank; Verification 5's ``--check``), to gate hard."""
    angle_ok, speed_ok, dev, speed = _angle_speed_mask(bank)
    ok = angle_ok & speed_ok
    n = int(ok.size)
    return {
        "n": n,
        "n_bad_angle": int((~angle_ok).sum()),
        "n_bad_speed": int((~speed_ok).sum()),
        "n_bad": int((~ok).sum()),
        "max_angle_dev_deg": float(dev.max()) if n else float("nan"),
        "speed_min": float(speed.min()) if n else float("nan"),
        "speed_max": float(speed.max()) if n else float("nan"),
        "ok": bool((~ok).sum() == 0),
    }


def _run_check(path: Path) -> None:
    bank = _load_raw(path)
    result = _check_angle_speed(bank)
    print(f"[check] {path}: {json.dumps(result)}", flush=True)
    assert result["ok"], (
        f"[check] FAILED: {result['n_bad']}/{result['n']} frames violate the angle/speed/tag "
        f"consistency check ({result['n_bad_angle']} bad angle, {result['n_bad_speed']} bad speed) "
        f"-- {path} is not a safe bank to load"
    )
    print(f"[check] PASS -- {result['n']} frames, max angle deviation {result['max_angle_dev_deg']:.2f} deg, "
          f"speed range [{result['speed_min']:.3f}, {result['speed_max']:.3f}]", flush=True)


def _run_merge(bank_paths: list[str], out_path: str, per_source: int, seed: int) -> None:
    paths = [Path(p) for p in bank_paths]
    raws = [_load_raw(p) for p in paths]

    joint_widths = {r["joint_pos"].shape[1] for r in raws}
    assert len(joint_widths) == 1, f"joint count mismatch across banks: {joint_widths} ({paths})"

    rng = np.random.default_rng(seed)
    sub_banks: list[dict[str, np.ndarray]] = []
    sha_list: list[dict[str, str]] = []
    for path, r in zip(paths, raws):
        srcs = np.unique(r["source"])
        assert srcs.size == 1, f"{path}: mixed source tags {srcs.tolist()} in a single raw bank"
        n_raw = r["joint_pos"].shape[0]
        angle_ok, speed_ok, dev, speed = _angle_speed_mask(r)
        pass_mask = angle_ok & speed_ok
        n_pass = int(pass_mask.sum())
        pass_frac = n_pass / n_raw if n_raw else 0.0
        print(
            f"[merge] {path}: source={int(srcs[0])} raw={n_raw} pass_angle_speed={n_pass} "
            f"({pass_frac:.1%}) bad_angle={int((~angle_ok).sum())} bad_speed={int((~speed_ok).sum())} "
            f"max_angle_dev_deg={float(dev.max()):.2f}",
            flush=True,
        )
        # A degenerate pass rate means the harvest is mislabeled/broken (wrong axis, wrong sign,
        # a checkpoint that isn't actually walking on the named axis) -- fail loudly rather than
        # silently merge a near-empty or wrong-axis source. A raw rollout's natural gait-cycle
        # cross-axis wobble legitimately fails a nonzero fraction (this is why the filter runs
        # here, not just at harvest time), so the bar is "enough good frames survive", not "all of
        # them do".
        assert n_pass >= min(per_source, 100), (
            f"{path}: only {n_pass}/{n_raw} frames ({pass_frac:.1%}) pass the angle/speed/tag "
            f"check -- too few to trust this source (need >= {min(per_source, 100)})"
        )
        good_idx = np.nonzero(pass_mask)[0]
        take = min(per_source, n_pass)
        idx = rng.choice(good_idx, size=take, replace=False) if take < n_pass else good_idx
        sub = {k: r[k][idx] for k in _BANK_KEYS + ("source",)}
        sub_banks.append(sub)
        sha_list.append({"path": str(path), "sha256": _sha256_file(path)})
        print(f"[merge] {path}: sampled={take} (from {n_pass} passing frames)", flush=True)

    merged = {k: np.concatenate([b[k] for b in sub_banks]) for k in _BANK_KEYS + ("source",)}
    n_total = merged["joint_pos"].shape[0]
    perm = rng.permutation(n_total)
    merged = {k: v[perm] for k, v in merged.items()}

    result = _check_angle_speed(merged)
    assert result["ok"], f"[merge] post-shuffle check FAILED: {result}"

    meta = {
        "sources": sha_list,
        "per_source": int(per_source),
        "seed": int(seed),
        "n_total": int(n_total),
        "check": result,
        "git_head": _git_head(),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    merged["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(out_path, **merged)
    out_sha = _sha256_file(Path(out_path))
    per_source_counts = ", ".join(
        f"source={int(np.unique(sb['source'])[0])} n={sb['source'].shape[0]}" for sb in sub_banks
    )
    print(f"[merge] wrote {out_path}: {n_total} frames ({per_source_counts}); sha256={out_sha}", flush=True)
    print(f"[merge] check: {json.dumps(result)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("banks", nargs="*", help="raw per-source RSI bank .npz files to merge")
    parser.add_argument("--out", type=str, default=None, help="merged bank output path (merge mode)")
    parser.add_argument("--per_source", type=int, default=2000, help="frames sampled from EACH source bank")
    parser.add_argument("--seed", type=int, default=0, help="seed for subsampling and shuffling")
    parser.add_argument("--check", type=str, default=None, help="re-verify an existing bank; skips merging")
    args = parser.parse_args()

    if args.check:
        _run_check(Path(args.check))
        return

    assert len(args.banks) >= 1, "merge mode needs at least one raw bank path"
    assert args.out, "--out is required in merge mode"
    _run_merge(args.banks, args.out, args.per_source, args.seed)


if __name__ == "__main__":
    main()
