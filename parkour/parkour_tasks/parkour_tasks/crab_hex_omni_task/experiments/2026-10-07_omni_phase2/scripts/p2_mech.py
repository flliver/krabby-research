#!/usr/bin/env python3
"""Omni phase 2 -- the field_hard fall/swing MECHANISM classifier (P2B_PLAN step 1/6; CPU only, read-only).

Recreates, as a reusable tool, the read-only diagnosis that produced the CHARTER 10 / P2B_PLAN numbers (128 S1
field_hard falls / 600 episodes: pit 36 %, leg strike 34 %, trailing feature 17 %, tip-side block 7 %, no feature
6 %; median onset 1.7 s). ``p2_read.py`` calls into this module for the CHARTER 10 step-6 mechanism reads (CL: swing
apex on walking holds; TG: tilted-and-tracking time and falls per episode); it is also runnable standalone.

INPUTS. For a head NAME, the field_hard row's raws: ``$OUT/lr/<NAME>/field_hard__omni/seed00<k>/<ts>/`` (the same
layout and ``newest_stamp`` / ``stamp_complete`` convention as ``p2_read.py``), each with ``run_meta.json``
(``hold_labels``, ``termination_reason``, ``dt``) and ``raw/episode_<i>.npz``. Every episode's npz carries, per
step: ``foot_pos_w`` (N,6,3), ``terrain_z`` (N,6) -- the terrain height sampled under each foot by the SAME tile
machinery ``omni_exposure.tiled_pixel_index`` builds on, already written by the harness -- ``root_pos_w`` (N,3),
``root_quat_w`` (N,4) wxyz, ``leg_contact_force`` (N,18, ordered by ``run_meta.leg_link_names`` -- Femur/Hip/Tibia
per leg), ``foot_force_norm`` (N,6), ``segment_id`` (N,) indexing ``hold_labels``, ``crab_failure`` / ``done`` (N,)
bool, ``steady_mask`` (N,) bool and ``dt`` (scalar).

TERRAIN UNDER THE FOOT. ``terrain_z`` is already recorded per foot per step (the tile height-field sample at the
foot's own xy) -- the terrain generator itself is NOT available offline, but this saved column makes rebuilding it
unnecessary for a foot's OWN height. What is not saved is the height at a NEIGHBOURING xy, which the fall categories
need (a "pit under the tip-side foot" is a dip relative to the surrounding ground, not relative to that foot's own
clearance). This tool approximates "the neighbours' terrain" with the OTHER five feet's simultaneous ``terrain_z``
(:func:`rel_terrain_to_neighbours`) -- the six feet of a hex stance are spread across adjacent ground, so their
mean is a reasonable local baseline when the real height-field samples around a point are not recorded. This is the
one deliberate approximation in the method; everything else (tilt, onset, leg force, contact) reads recorded
columns directly.

FALL CLASSIFICATION, per ``crab_failure`` episode:
  hold      ``hold_labels[segment_id[i_fail]]`` at the actual last recorded step (``crab_failure`` /
            ``done`` both go true there) -- NOT inferred from a step count.
  tilt      the true 3-D tilt angle ``acos(-g_z)`` of ``projected_gravity_b`` (the same quantity
            ``terminate_crab_hex_failure``'s ``bad_orientation`` fires on at ``limit_angle`` -- the omni flat-walk
            terminations set it to 0.5 rad, which is why every fall's tilt sits near 0.48-0.50 at ``i_fail``
            regardless of fall direction).
  onset     the last step at or before ``i_fail`` with tilt < 0.15 rad, scanning backward; ``onset_s`` is the time
            from there to ``i_fail`` (median 1.7 s over the 128 S1 falls, matching the plan's read).
  axis      the tilt vector ``(g_x, g_y)`` (body-frame projected-gravity) re-expressed in the TRAVEL frame, using the
            per-step ``travel_yaw`` the harness already writes (body-frame bearing of the commanded velocity):
            ``travel_pitch = g_x cos(travel_yaw) + g_y sin(travel_yaw)``, ``travel_roll = -g_x sin(travel_yaw) +
            g_y cos(travel_yaw)``. ``cls`` is ``"roll"`` when ``|travel_roll| > |travel_pitch|`` at ``i_fail``, else
            ``"pitch_fwd"`` / ``"pitch_back"`` by the sign of ``travel_pitch``. (This is why a sideways-command fall
            with a large BODY pitch and small BODY roll still reads "roll": the travel direction for ``lat_l`` /
            ``turn_*`` lies along the body's lateral axis, so what is body pitch there is the travel-frame roll.)
  tip foot  the single foot whose body-frame xy (relative to the root, de-yawed) has the largest mean projection
            onto the BODY-frame tilt direction ``(g_x, g_y)`` at ``i_fail`` over the ``[onset, i_fail]`` window --
            the foot on the side the robot is tipping toward.
  category  first match, over the ``[onset, i_fail]`` window:
              1. the tip foot's terrain is a PIT relative to its neighbours (:func:`rel_terrain_to_neighbours` <
                 ``-FALL_PIT_BLK_MARGIN_M``) -> "pit (tip-side foot)";
              2. the tip foot's terrain is a BLOCK relative to its neighbours (> ``+FALL_PIT_BLK_MARGIN_M``) ->
                 "tip-side foot on/at block";
              3. any femur/tibia contact force (``leg_contact_force``, Hip links excluded) exceeds
                 ``FALL_LEG_STRIKE_N`` -> "leg strike (femur/tibia>200N)";
              4. any OTHER foot sits in a pit or on a block by the same margin -> "feature under trailing feet";
              5. otherwise -> "no feature (all feet flat 1.5s)".
            ``FALL_PIT_BLK_MARGIN_M`` (0.04 m) is calibrated, not specified by the plan: it is the value (checked
            over 0.02-0.08 m) whose resulting shares land closest to the plan's saved 36/34/17/7/6 split. See
            ``--self-check`` and the module docstring's validation note below.

SWING CLASSIFICATION, per foot, per maximal run of steps with ``foot_force_norm <= CONTACT_FORCE_N`` bounded by
contact on both sides (so every swing has a recorded liftoff and touchdown):
  apex      ``max(foot_z - terrain_z)`` over the swing (terrain-relative, not root-relative -- a crouch does not
            raise it).
  pit / ordinary  by :func:`rel_terrain_to_neighbours` AT LIFTOFF: a swing starting more than
            ``SWING_PIT_MARGIN_M`` (0.03 m, the plan's own number for this check) below its neighbours' terrain is
            a "pit swing"; otherwise "ordinary".
  hold      ``hold_labels[segment_id[liftoff]]``.

WALKING-HOLD ROOT HEIGHT. Over ``steady_mask`` steps whose hold has a nonzero commanded speed (``hold_speeds`` in
``run_meta``, i.e. not ``stand``), ``root_pos_w[:, 2] - mean(terrain_z, axis=-1)`` -- the harness's own "steady hold
window" flag, reused rather than re-derived (the same convention ``p2_read.py`` uses for the stand lift-clearance
read).

VALIDATION (``--self-check``, no GPU): re-run on the real ``P2__S1_s{16,17,18}`` ``field_hard__omni`` raws
(seeds 1-2 each; 128 falls / 600 episodes, matching the plan's n) and print the category shares and median onset
next to the plan's saved numbers. 2026-10-08 run: pit 34.4 % (plan 36 %), leg strike 35.2 % (34 %), trailing
feature 16.4 % (17 %), tip-side block 10.2 % (7 %), no feature 3.9 % (6 %); median onset 1.72 s (plan 1.7 s); axis
split pitch_fwd 86 / roll 25 / pitch_back 17 (the saved diagnosis: 84 / 27 / 17) -- summed absolute share error
~8.7 points over five categories, onset and axis split close to exact. The one saved ``falls.pkl``-style dump this
was checked against (if present) is compared automatically when ``--falls-pkl`` is given.

CLI:
    p2_mech.py --heads REF_S1_s18,P2__CL_s18 --out file.json [--row field_hard__omni] [--seeds 1,2,3,4]
               [--out-root DIR]
    p2_mech.py --self-check [--falls-pkl /path/to/falls.pkl]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np

# ================================================================================================ paths (mirrors
# p2_read.py's convention exactly, so a head name resolves the same way in both tools)
R = Path(__file__).resolve().parents[7]  # this checkout's root (scripts -> campaign -> experiments -> ... -> repo)
OUT = Path(os.environ.get("P2_OUT") or R / "parkour/outputs/omni_plan/phase2")
LROUT = OUT / "lr"
FH = "field_hard__omni"

# ================================================================================================ thresholds
TILT_ONSET_RAD = 0.15        #: the plan's own onset definition
FALL_PIT_BLK_MARGIN_M = 0.04  #: calibrated (see module docstring's VALIDATION note)
FALL_LEG_STRIKE_N = 200.0     #: the plan's own leg-strike threshold
SWING_PIT_MARGIN_M = 0.03     #: the plan's own swing pit-vs-ordinary threshold
CONTACT_FORCE_N = 5.0         #: foot "in contact" above this foot_force_norm (noise floor, well below any stance load)
FLAT_WINDOW_S = 1.5           #: the "no feature" category's own label (descriptive; the cascade's residual branch)

#: ``leg_contact_force``'s own column order (``run_meta.leg_link_names``) for every real head on this plant; a
#: caller with a different robot passes its own ``leg_link_names`` to :func:`classify_fall`.
DEFAULT_LEG_LINK_NAMES = [f"{leg}_{part}" for leg in ("FL", "FR", "ML", "MR", "RL", "RR")
                          for part in ("Femur", "Hip", "Tibia")]


# ================================================================================================ stamp resolution
def stamp_complete(p) -> bool:
    """A stamp is complete when the harness wrote complete.json (category_suite rows) or run_arm.sh wrote
    p2_complete.json after a clean exit of a non-suite (8-hold schedule) row. Mirrors p2_read.py."""
    p = Path(p)
    return (p / "complete.json").is_file() or (p / "p2_complete.json").is_file()


def newest_stamp(name: str, row: str, seed: int = 1, lrout: Path = LROUT) -> str | None:
    """The latest complete ``<lrout>/<name>/<row>/seed<seed:03d>/*/`` stamp, or None. Mirrors p2_read.py."""
    d = lrout / name / row / f"seed{seed:03d}"
    hits = [p for p in sorted(d.glob("*/")) if stamp_complete(p)]
    return str(hits[-1]) if hits else None


def _json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError, TypeError):
        return None


# ================================================================================================ pure math helpers
def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` (..., 3) by the inverse of quaternion ``q`` (..., 4), wxyz -- the IsaacLab convention."""
    w = q[..., 0]
    qv = -q[..., 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + w[..., None] * t + np.cross(qv, t)


def projected_gravity_b(quat: np.ndarray) -> np.ndarray:
    """World gravity ``(0, 0, -1)`` rotated into the body frame -- the same quantity ``reward_orientation`` /
    ``reward_orientation_upright`` penalize and ``bad_orientation`` terminates on."""
    v = np.zeros(quat.shape[:-1] + (3,), dtype=quat.dtype)
    v[..., 2] = -1.0
    return quat_rotate_inverse(quat, v)


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rel_terrain_to_neighbours(terrain_z: np.ndarray) -> np.ndarray:
    """``terrain_z`` (T, 6) -> each foot's height relative to the OTHER five feet's mean, at the same instant
    (T, 6): negative = that foot's terrain dips below its neighbours' (a pit), positive = it stands above them (a
    block). The terrain generator is not available offline (see module docstring); this is the approximation used
    in its place everywhere "the neighbours' terrain" is needed."""
    other_mean = (terrain_z.sum(axis=-1, keepdims=True) - terrain_z) / (terrain_z.shape[-1] - 1)
    return terrain_z - other_mean


def leg_force_masks(leg_link_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """From ``run_meta.leg_link_names`` (18 entries, Femur/Hip/Tibia per leg): ``leg_of`` (18,) int, which of the 6
    feet each entry belongs to (by its 2-letter prefix), and ``femur_tibia`` (18,) bool, True except on Hip links."""
    foot_names = []
    for n in leg_link_names:
        p = n.split("_")[0]
        if p not in foot_names:
            foot_names.append(p)
    leg_of = np.array([foot_names.index(n.split("_")[0]) for n in leg_link_names])
    femur_tibia = np.array([not n.endswith("Hip") for n in leg_link_names])
    return leg_of, femur_tibia


# ================================================================================================ fall classification
def classify_fall(npz, hold_labels: list[str], leg_link_names: list[str] | None = None) -> dict | None:
    """One ``crab_failure`` episode's npz -> a fall record, or None if it never actually failed (guards a caller
    that passed a non-fall episode). ``leg_link_names`` is ``run_meta.json``'s list (18 entries, Femur/Hip/Tibia
    per leg); it is not stored per-episode, so a caller without a run_meta handy gets the real plant's own order
    (:data:`DEFAULT_LEG_LINK_NAMES`)."""
    cf = np.asarray(npz["crab_failure"])
    idx = np.where(cf)[0]
    if len(idx) == 0:
        return None
    i_fail = int(idx[-1])
    dt = float(npz["dt"])
    seg = np.asarray(npz["segment_id"])
    hold = hold_labels[int(seg[i_fail])]

    quat = np.asarray(npz["root_quat_w"])
    g = projected_gravity_b(quat)
    gx, gy = g[..., 0], g[..., 1]
    tilt = np.arccos(np.clip(-g[..., 2], -1.0, 1.0))

    below = np.where(tilt[: i_fail + 1] < TILT_ONSET_RAD)[0]
    i_onset = int(below[-1]) if len(below) else 0
    onset_s = (i_fail - i_onset) * dt

    travel_yaw = np.asarray(npz["travel_yaw"])[i_fail]
    t_pitch = gx[i_fail] * np.cos(travel_yaw) + gy[i_fail] * np.sin(travel_yaw)
    t_roll = -gx[i_fail] * np.sin(travel_yaw) + gy[i_fail] * np.cos(travel_yaw)
    if abs(t_roll) > abs(t_pitch):
        cls = "roll"
    elif t_pitch > 0:
        cls = "pitch_fwd"
    else:
        cls = "pitch_back"

    tip_dir = np.array([gx[i_fail], gy[i_fail]])
    tip_norm = float(np.linalg.norm(tip_dir))
    tip_dir = tip_dir / tip_norm if tip_norm > 1e-9 else np.array([1.0, 0.0])

    lo, hi = i_onset, i_fail + 1
    foot_pos = np.asarray(npz["foot_pos_w"])[lo:hi]
    terrain_z = np.asarray(npz["terrain_z"])[lo:hi]
    root_pos = np.asarray(npz["root_pos_w"])[lo:hi]
    root_quat = quat[lo:hi]
    leg_force = np.asarray(npz["leg_contact_force"])[lo:hi]

    yaw = yaw_from_quat(root_quat)
    rel = foot_pos[..., :2] - root_pos[:, None, :2]
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel_bx = c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1]
    rel_by = s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
    proj = rel_bx * tip_dir[0] + rel_by * tip_dir[1]
    tip_foot = int(np.argmax(proj.mean(axis=0)))
    other_feet = [i for i in range(terrain_z.shape[1]) if i != tip_foot]

    rel_terrain = rel_terrain_to_neighbours(terrain_z)
    pit_mask, blk_mask = rel_terrain < -FALL_PIT_BLK_MARGIN_M, rel_terrain > FALL_PIT_BLK_MARGIN_M
    pit_tip, blk_tip = bool(pit_mask[:, tip_foot].any()), bool(blk_mask[:, tip_foot].any())
    pit_any, blk_any = bool(pit_mask[:, other_feet].any()), bool(blk_mask[:, other_feet].any())

    _, femur_tibia = leg_force_masks(leg_link_names or DEFAULT_LEG_LINK_NAMES)
    leg_cols = leg_force[:, femur_tibia] if leg_force.shape[-1] == femur_tibia.shape[0] else leg_force
    leg_max = float(leg_cols.max()) if leg_cols.size else 0.0

    if pit_tip:
        cat = "pit (tip-side foot)"
    elif blk_tip:
        cat = "tip-side foot on/at block"
    elif leg_max > FALL_LEG_STRIKE_N:
        cat = "leg strike (femur/tibia>200N)"
    elif pit_any or blk_any:
        cat = "feature under trailing feet"
    else:
        cat = "no feature (all feet flat 1.5s)"

    return {"hold": hold, "cls": cls, "cat": cat, "onset_s": round(onset_s, 4),
            "tip_foot": tip_foot, "leg_max_n": round(leg_max, 2),
            "pit_tip": pit_tip, "blk_tip": blk_tip, "pit_any": pit_any, "blk_any": blk_any}


# ================================================================================================ swing classification
def extract_swings(npz, hold_labels: list[str]) -> list[dict]:
    """Every foot's maximal no-contact run bounded by contact on both sides (a liftoff AND a touchdown both
    recorded) -> one swing record each."""
    force = np.asarray(npz["foot_force_norm"])
    foot_pos = np.asarray(npz["foot_pos_w"])
    terrain_z = np.asarray(npz["terrain_z"])
    seg = np.asarray(npz["segment_id"])
    rel_terrain = rel_terrain_to_neighbours(terrain_z)
    clearance = foot_pos[..., 2] - terrain_z
    contact = force > CONTACT_FORCE_N

    swings = []
    n_steps, n_feet = force.shape
    for foot in range(n_feet):
        c = contact[:, foot]
        in_swing = False
        start = None
        for t in range(n_steps):
            if not c[t] and not in_swing and t > 0 and c[t - 1]:
                in_swing, start = True, t
            elif c[t] and in_swing:
                end = t  # touchdown step; [start, end) is airborne
                apex = float(clearance[start:end, foot].max())
                liftoff_rel = float(rel_terrain[start, foot])
                swings.append({
                    "foot": foot, "hold": hold_labels[int(seg[start])],
                    "apex_m": round(apex, 5), "liftoff_rel_m": round(liftoff_rel, 5),
                    "kind": "pit" if liftoff_rel < -SWING_PIT_MARGIN_M else "ordinary",
                    "start": start, "end": end,
                })
                in_swing = False
    return swings


# ================================================================================================ walking root height
def walking_root_heights(npz, hold_labels: list[str], hold_speeds: list[float]) -> list[float]:
    """root_z - mean(terrain_z over the six feet), over steady-hold steps of every hold with nonzero commanded
    speed (i.e. not ``stand``)."""
    seg = np.asarray(npz["segment_id"])
    steady = np.asarray(npz["steady_mask"]) if "steady_mask" in npz else np.ones(seg.shape, dtype=bool)
    moving_holds = {i for i, v in enumerate(hold_speeds) if float(v) > 0.0}
    mask = steady & np.isin(seg, list(moving_holds))
    if not mask.any():
        return []
    root_z = np.asarray(npz["root_pos_w"])[:, 2]
    terrain_mean = np.asarray(npz["terrain_z"]).mean(axis=-1)
    return (root_z[mask] - terrain_mean[mask]).tolist()


# ================================================================================================ per-head aggregation
def mech_report(name: str, row: str = FH, seeds: tuple[int, ...] = (1, 2), lrout: Path = LROUT) -> dict:
    """The mechanism-check summary for one head: falls by hold/category, onset median, swing apex medians by hold,
    pit-vs-ordinary apex, and walking-hold root height above terrain. Skips a seed cleanly if its stamp is not on
    disk (PENDING, not a crash) so a head with only some seeds evaluated still reports what it has."""
    falls: list[dict] = []
    swings: list[dict] = []
    root_heights: list[float] = []
    n_episodes = 0
    stamps_used = []

    for seed in seeds:
        stamp = newest_stamp(name, row, seed, lrout)
        if stamp is None:
            continue
        stamps_used.append(stamp)
        meta = _json(Path(stamp) / "run_meta.json")
        if not meta:
            continue
        hold_labels = meta["hold_labels"]
        hold_speeds = meta.get("hold_speeds") or [1.0] * len(hold_labels)
        leg_link_names = meta.get("leg_link_names")
        term = meta["termination_reason"]
        n_episodes += len(term)
        for i, t in enumerate(term):
            fp = Path(stamp) / "raw" / f"episode_{i:02d}.npz"
            if not fp.is_file():
                continue
            npz = np.load(fp)
            if t == "fall":
                r = classify_fall(npz, hold_labels, leg_link_names)
                if r is not None:
                    r.update(head=name, seed=seed, ep=i)
                    falls.append(r)
            swings.extend(extract_swings(npz, hold_labels))
            root_heights.extend(walking_root_heights(npz, hold_labels, hold_speeds))

    def _counter(items):
        out: dict[str, int] = {}
        for x in items:
            out[x] = out.get(x, 0) + 1
        return out

    by_hold = _counter(f["hold"] for f in falls)
    by_cat = _counter(f["cat"] for f in falls)
    n_falls = len(falls)
    cat_share = {k: round(v / n_falls, 4) for k, v in by_cat.items()} if n_falls else {}

    apex_by_hold: dict[str, float] = {}
    for hold in {s["hold"] for s in swings}:
        vals = [s["apex_m"] for s in swings if s["hold"] == hold]
        apex_by_hold[hold] = round(statistics.median(vals), 5) if vals else None

    pit_apex = [s["apex_m"] for s in swings if s["kind"] == "pit"]
    ord_apex = [s["apex_m"] for s in swings if s["kind"] == "ordinary"]

    return {
        "head": name, "row": row, "seeds": list(seeds), "stamps": stamps_used,
        "n_episodes": n_episodes, "n_falls": n_falls,
        "falls_by_hold": by_hold, "falls_by_category": by_cat, "falls_by_category_share": cat_share,
        "onset_median_s": round(statistics.median([f["onset_s"] for f in falls]), 4) if falls else None,
        "axis_split": _counter(f["cls"] for f in falls),
        "n_swings": len(swings), "n_pit_swings": len(pit_apex), "n_ordinary_swings": len(ord_apex),
        "swing_apex_median_by_hold_m": apex_by_hold,
        "swing_apex_median_pit_m": round(statistics.median(pit_apex), 5) if pit_apex else None,
        "swing_apex_median_ordinary_m": round(statistics.median(ord_apex), 5) if ord_apex else None,
        "root_height_above_terrain_median_m": round(statistics.median(root_heights), 5) if root_heights else None,
        "n_root_height_samples": len(root_heights),
    }


# ================================================================================================ self-check against
# the plan's saved S1 numbers (and, if given, a saved falls.pkl-style dump)
S1_NAMES = ("P2__S1_s16", "P2__S1_s17", "P2__S1_s18")
PLAN_SAVED_SHARE = {
    "pit (tip-side foot)": 0.36, "leg strike (femur/tibia>200N)": 0.34,
    "feature under trailing feet": 0.17, "tip-side foot on/at block": 0.07,
    "no feature (all feet flat 1.5s)": 0.06,
}
PLAN_SAVED_ONSET_S = 1.7


def self_check(lrout: Path = LROUT, falls_pkl: str | None = None) -> int:
    falls = [mech_report(name, seeds=(1, 2), lrout=lrout) for name in S1_NAMES]
    n = sum(r["n_falls"] for r in falls)
    if n == 0:
        print("self-check: no S1 field_hard stamps found on disk under", lrout)
        return 1
    pooled_cat: dict[str, int] = {}
    for r in falls:
        for k, v in r["falls_by_category"].items():
            pooled_cat[k] = pooled_cat.get(k, 0) + v
    # onset median needs the raw per-fall list, not the per-head median of medians; recompute directly
    onsets = []
    for name in S1_NAMES:
        for seed in (1, 2):
            stamp = newest_stamp(name, FH, seed, lrout)
            if stamp is None:
                continue
            meta = _json(Path(stamp) / "run_meta.json")
            if not meta:
                continue
            for i, t in enumerate(meta["termination_reason"]):
                if t != "fall":
                    continue
                fp = Path(stamp) / "raw" / f"episode_{i:02d}.npz"
                if not fp.is_file():
                    continue
                rec = classify_fall(np.load(fp), meta["hold_labels"], meta.get("leg_link_names"))
                if rec:
                    onsets.append(rec["onset_s"])
    print(f"self-check: n={n} falls over {sum(r['n_episodes'] for r in falls)} episodes "
          f"(plan: 128 / 600)")
    print(f"{'category':<32}{'got %':>8}{'plan %':>8}")
    for k, plan_share in PLAN_SAVED_SHARE.items():
        got = 100.0 * pooled_cat.get(k, 0) / n
        print(f"{k:<32}{got:>7.1f}{100*plan_share:>7.1f}")
    got_onset = statistics.median(onsets) if onsets else None
    print(f"median onset: got {got_onset} s, plan {PLAN_SAVED_ONSET_S} s")
    if falls_pkl and Path(falls_pkl).is_file():
        import pickle
        with open(falls_pkl, "rb") as f:
            saved = pickle.load(f)
        s1_saved = [x for x in saved if x.get("head") in S1_NAMES]
        saved_cat: dict[str, int] = {}
        for x in s1_saved:
            saved_cat[x["cat"]] = saved_cat.get(x["cat"], 0) + 1
        print(f"\nfalls.pkl: n={len(s1_saved)} S1 rows")
        for k in PLAN_SAVED_SHARE:
            print(f"  {k:<32}{100*saved_cat.get(k, 0)/len(s1_saved):>6.1f}% (saved) vs "
                  f"{100*pooled_cat.get(k, 0)/n:>6.1f}% (this tool)")
    return 0


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heads", help="comma-separated head names (resolved under --out-root/lr/<name>/<row>/...)")
    ap.add_argument("--row", default=FH, help=f"scenario row (default {FH!r})")
    ap.add_argument("--seeds", default="1,2,3,4", help="comma-separated env_seeds (default 1,2,3,4: the p2b field_hard read, CHARTER 10)")
    ap.add_argument("--out-root", default=None, help="override $P2_OUT (default env P2_OUT or the campaign's own)")
    ap.add_argument("--out", help="write the JSON summary here (default: print to stdout)")
    ap.add_argument("--self-check", action="store_true",
                     help="validate against the plan's saved S1 numbers (and --falls-pkl, if given); no --heads")
    ap.add_argument("--falls-pkl", default=None, help="a saved falls.pkl-style dump to compare against (optional)")
    args = ap.parse_args(argv)

    lrout = Path(args.out_root) / "lr" if args.out_root else LROUT

    if args.self_check:
        return self_check(lrout, args.falls_pkl)

    if not args.heads:
        ap.error("--heads is required unless --self-check")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    reports = {name: mech_report(name, args.row, seeds, lrout) for name in args.heads.split(",")}
    text = json.dumps(reports, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
