#!/usr/bin/env python3
"""Krabby FL leg model v3: (hip counts, knee counts, yaw) -> outlines in the leg-plane frame.

MODEL ONLY. numpy (+ scipy for the fit). Nothing here opens a port or talks to the robot.

Frame (inches, degrees), as in geom3_inputs.md:
  x = horizontal distance out from the body side wall (+ outboard, - under the body),
  z = height above the belly plane (= table top; + up). Body: x < 0, 0 <= z <= 12.5.
  Table: solid for x < -13.75, z < 0. Floor: z = -34.

What drives the leg (v3 replaces geom2's linkage fit; NO linkage is forced):
  femur elevation e(hip counts)  = smooth monotone cubic, weighted LSQ through the MEASURED
                                   tape/phone/square points (FEMUR_DATA);
  knee interior theta(knee cnts) = smooth monotone cubic through the MEASURED phone readings
                                   (KNEE_DATA, 296-785) and, below 296, whatever the ruler poses
                                   A-D need (counts 0 and 157 were only ever derived from them).
  Both cubics, the toe length and the ruler yaw are fitted JOINTLY (fit()) with the measured
  points as data (+-sigma) and the pivot / toe length as priors. The knee rides on the femur,
  so theta depends on knee counts only.

Geometry: femur pivot P = (3.0, 2.5) (+- 0.25), femur 23.0 in pivot to pivot.
  Tibia direction (knee -> toe), CCW from +x in the leg plane: psi = 180 + e + theta.
  Yaw: the leg plane is turned by `yaw` about a vertical axis; heights and in-plane angles do
  not change, horizontal offsets from P shrink: x = P_x + cos(yaw) * u_rel. Two yaws:
    yaw_r  the yaw that scales the RULER side readings (fitted; ~0 => readings already
           in-plane or the plane is near square to the wall),
    yaw    the true plan-view yaw used for clearance (unknown, 0-40; ensemble). Plywood is
           1 in thick: clearance uses the worst face, x - w*sin(yaw) with w = 0.5 (femur,
           tibia) or ACT_W (knee actuator, on the femur's side face).
  Compliance: the measured femur angles are with the leg HANGING. `compliance` (deg, + = femur
  higher) shifts e when the leg is unloaded/lifted (measured 2 deg at hip 846).

Outlines (closed polygons, CAD from crab_hex_leg_profiles.py via geom2):
  femur          CAD plate, hinges 23.0 apart, 2.5 in end caps;
  tibia          CAD tapered plank (5 in at the knee, 2 in at the toe, ~4.9 deg taper; the
                 measured taper is 4.5-6), the toe side scaled so the tip is L_toe from the pin;
  toe            the last TOE_CAP in of the tibia outline (rounded end), plus the toe point;
  knee actuator  capsule (radius ACT_R, ACT_REAR behind the rear pin) from its anchor on the
                 femur (5.86 in from P, the outer hole) to the lever pin on the tibia's upper
                 extension. The lever pin (radius lever_r, offset lever_dk) is fitted so the rod
                 length is linear in knee counts at 5.33 counts/mm (fit_knee_lever()). The
                 actuator's diameter/housing are NOT measured (ACT_R/ACT_REAR are guesses).

Corridor margins (clearances(); + = inside the corridor by that many inches):
  body   distance from the body box (x <= 0, 0 <= z <= 12.5) - BODY_GAP   (BODY_GAP = 0: the
         contact facts are contact / no-contact; a planner adds its own margin)
  table  for every part below the table top (z < 0): x + INBOARD_LIMIT, INBOARD_LIMIT = 10
         (table edge 13.75 - 2 in robot off-centre - 1.75 in margin), the geom2 rule;
         table_contact = x + 13.75 (the actual solid-table edge, no margin)
  floor  z - (floor_z + FLOOR_GAP), FLOOR_GAP = 3 in (floor uneven mat over gravel, +-0.5 in
         floor height, toe rounding), the geom2 rule.

  python leg_model_v3.py            fit, write leg_model_v3_fit.json, print the validation table
  python leg_model_v3.py --ensemble N   also draw N ensemble members -> leg_model_v3_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FIT_JSON = HERE / "leg_model_v3_fit.json"

# ------------------------------------------------------------------ fixed / measured
HIP_MAX = 1069            # hip extend limit switch (counts); 0 = curl switch
KNEE_SWITCH = 785         # knee uncurl limit switch (counts); 0 = curl (mechanical) stop
COUNTS_PER_MM = 5.33
MM_PER_IN = 25.4
FEMUR_LEN = 23.0
PIVOT = (3.0, 2.5)
KNEE_ACT_ANCHOR_FROM_P = 5.86     # outer femur hole (17.14 from the knee)
TIBIA_LEN_CAD = 32.5
BODY_H = 12.5
TABLE_EDGE = -13.75
FLOOR_Z = -34.0
INBOARD_LIMIT = 10.0
FLOOR_GAP = 3.0
BODY_GAP = 0.0
PLY_W = 0.5               # half plywood thickness (out of plane)
ACT_R = 1.0               # knee actuator half-width in the leg plane (NOT measured)
ACT_REAR = 1.5            # actuator housing behind its rear pin (NOT measured)
ACT_W = 1.5               # actuator centre out of the femur mid-plane (NOT measured)
TOE_CAP = 2.0             # length of the "toe" outline at the tibia end
HUB_R = 3.0               # femur points within this radius of P (its 2.5 in end cap) are left out of
                          # the BODY rule: the hub turns in place beside the hip plate by design
CONTACT_TOL_COUNTS = 13   # 2 deg of femur at the 941-1003 slope (0.155 deg/count)

# (counts, femur elevation deg (+ = knee end up), sigma, source). Hanging leg.
FEMUR_DATA = [
    (0, 40.7, 1.0, "tape h3"), (0, 40.5, 1.0, "phone centred"),
    (478, 7.5, 1.5, "tape"), (478, 9.0, 1.5, "phone"), (482, 8.4, 1.0, "phone centred"),
    (612, -3.5, 1.0, "phone"),
    (846, -30.0, 1.0, "phone hanging"), (846, -31.4, 1.5, "tape"),
    (1069, -65.0, 2.0, "square"),
]
FEMUR_EXCLUDED = [(1069, -69.2, "tape on uneven mat floor, 6 deg/in sensitive")]
# (counts, knee interior deg, sigma, source) -- direct phone/square readings (MEAS k1)
KNEE_DATA = [
    (296, 65.6, 2.0, "square, hip 482"), (296, 69.25, 2.0, "square, hip 0"),
    (508, 100.25, 2.0, "phone"), (608, 111.25, 2.0, "phone"),
    (706, 123.25, 2.0, "phone"), (785, 132.75, 2.0, "phone, switch"),
]
KNEE_RULER_DERIVED = [(0, 19.0, "pose A"), (157, 42.0, "poses B/C, 40-44")]
KNEE_REFIT_JOB = [(0, 19.0), (157, 48.5), (300, 68.0), (500, 93.0), (600, 105.5), (782, 132.5)]
# ruler toe poses: (hip, knee, x, z), +-1 in
RULER = {"A": (0, 0, 5.0, -10.0), "B": (0, 157, 16.0, -13.0),
         "C": (475, 157, 5.0, -19.0), "D": (1069, 157, -16.5, -6.0)}
RULER_SIGMA = 1.0

# ------------------------------------------------------------------ CAD outlines (geom2)
FEMUR_OUTLINE = np.array([
    (11.5, -2.5), (12.2639, -2.4016), (12.8889, -2.1296), (13.375, -1.7188), (13.7222, -1.2037),
    (13.9306, -0.6192), (14.0, 0.0), (13.9306, 0.6192), (13.7222, 1.2037), (13.375, 1.7188),
    (12.8889, 2.1296), (12.2639, 2.4016), (11.5, 2.5), (8.625, 2.3437), (2.875, 1.6559),
    (0.0, 1.4996), (-2.875, 1.6559), (-8.625, 2.3437), (-11.5, 2.5), (-12.2639, 2.4016),
    (-12.8889, 2.1296), (-13.375, 1.7188), (-13.7222, 1.2037), (-13.9306, 0.6192), (-14.0, 0.0),
    (-13.9306, -0.6192), (-13.7222, -1.2037), (-13.375, -1.7188), (-12.8889, -2.1296),
    (-12.2639, -2.4016), (-11.5, -2.5), (-8.625, -2.3438), (-2.875, -1.6566), (0.0, -1.5004),
    (2.875, -1.6566), (8.625, -2.3438),
])
FEMUR_PIVOT_A = -11.5
TIBIA_OUTLINE = np.array([
    (16.3337, 2.5), (17.0976, 2.4016), (17.7226, 2.1296), (18.2087, 1.7187), (18.5559, 1.2037),
    (18.7643, 0.6192), (18.8337, 0.0), (18.7643, -0.6192), (18.5559, -1.2037), (18.2087, -1.7188),
    (17.7226, -2.1296), (17.0976, -2.4016), (16.3337, -2.5), (-18.6172, -1.0), (-18.9375, -0.9618),
    (-19.4032, -0.6959), (-19.5487, -0.4947), (-19.665, -0.0224), (-19.5484, 0.455),
    (-19.4028, 0.6623), (-18.9372, 0.9481), (-18.6172, 1.0), (-9.8373, 1.3543), (7.7211, 2.218),
])
TIBIA_KNEE_A = 12.835


def _densify(poly, ds):
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        m = max(1, int(math.ceil(float(np.linalg.norm(q - p)) / ds)))
        out.append(p + (np.arange(m) / m)[:, None] * (q - p))
    return np.vstack(out)


# part-local coordinates: (d along the part from its joint, b across)
FEM_LOCAL = np.c_[FEMUR_OUTLINE[:, 0] - FEMUR_PIVOT_A, FEMUR_OUTLINE[:, 1]]       # d from P
TIB_LOCAL = np.c_[TIBIA_KNEE_A - TIBIA_OUTLINE[:, 0], TIBIA_OUTLINE[:, 1]]       # d from knee, + toward toe
FEM_DENSE = np.c_[_densify(FEMUR_OUTLINE, 0.25)[:, 0] - FEMUR_PIVOT_A, _densify(FEMUR_OUTLINE, 0.25)[:, 1]]
FEM_HUB = np.hypot(FEM_DENSE[:, 0], FEM_DENSE[:, 1]) < HUB_R
_td = _densify(TIBIA_OUTLINE, 0.25)
TIB_DENSE = np.c_[TIBIA_KNEE_A - _td[:, 0], _td[:, 1]]


def _capsule(n=12):
    """Unit capsule outline in (d, b): d in [-ACT_REAR, 1]*len handled at runtime; returns
    angles for the two end caps."""
    return np.linspace(-math.pi / 2, math.pi / 2, n)


# ------------------------------------------------------------------ parameters
DEFAULT = {
    # femur cubic in s = counts/1069 (deg) and knee cubic in s = counts/785 (deg); set by fit()
    "fem_coef": [40.6, -63.73, 1.45, -43.35],
    "knee_coef": [19.0, 25.0, 0.0, 0.0],
    "L_toe": 32.5,          # knee pin -> toe tip (in)
    "yaw_r": 0.0,           # yaw scaling the ruler side readings (deg)
    "yaw": 0.0,             # true plan-view yaw for clearance (deg)
    "pivot_x": PIVOT[0], "pivot_z": PIVOT[1],
    "fem_off0": 0.0, "fem_off1": 0.0,      # femur map perturbation at hip 0 / 1069 (deg, linear)
    "knee_off0": 0.0, "knee_off1": 0.0,    # knee map perturbation at knee 0 / 785 (deg, linear)
    "compliance": 0.0,      # femur elevation added when unloaded (deg, + = up)
    "floor_z": FLOOR_Z,
    "lever_r": 3.0, "lever_dk": 0.0,       # knee actuator lever pin (fit_knee_lever)
    "act_r": ACT_R, "act_rear": ACT_REAR, "act_w": ACT_W,
}


def load_params():
    p = dict(DEFAULT)
    if FIT_JSON.exists():
        p.update(json.loads(FIT_JSON.read_text(encoding="utf-8"))["params"])
    return p


def P(p=None, **kw):
    q = load_params() if p is None else dict(p)
    q.update(kw)
    return q


# ------------------------------------------------------------------ joint maps
def _cubic(coef, s):
    a0, a1, a2, a3 = coef
    return a0 + s * (a1 + s * (a2 + s * a3))


def femur_elev_deg(hip_counts, p=None):
    """Femur elevation (deg, + = knee end above the pivot) for hip counts, incl. perturbation
    and compliance."""
    p = P(p)
    c = np.asarray(hip_counts, float)
    s = c / HIP_MAX
    return (_cubic(p["fem_coef"], s) + p["fem_off0"] + (p["fem_off1"] - p["fem_off0"]) * s
            + p["compliance"])


def knee_interior_deg(knee_counts, p=None):
    """Knee interior angle (knee->pivot vs knee->toe, deg) for knee counts."""
    p = P(p)
    c = np.asarray(knee_counts, float)
    s = c / KNEE_SWITCH
    return _cubic(p["knee_coef"], s) + p["knee_off0"] + (p["knee_off1"] - p["knee_off0"]) * s


def hip_counts_for_elev(elev_deg, p=None):
    c = np.linspace(-50, HIP_MAX + 50, 4401)
    e = femur_elev_deg(c, p)
    return np.interp(-np.asarray(elev_deg, float), -e, c)


def knee_counts_for_interior(theta_deg, p=None):
    c = np.linspace(-50, KNEE_SWITCH + 50, 3401)
    return np.interp(np.asarray(theta_deg, float), knee_interior_deg(c, p), c)


# ------------------------------------------------------------------ kinematics
def frames(hip_counts, knee_counts, p=None):
    """Broadcast counts -> in-plane frames relative to the pivot (u_rel, z), shape (M, 2)."""
    p = P(p)
    hc, kc = np.broadcast_arrays(np.asarray(hip_counts, float), np.asarray(knee_counts, float))
    e = np.radians(femur_elev_deg(hc.ravel(), p))
    th = np.radians(knee_interior_deg(kc.ravel(), p))
    ef = np.stack([np.cos(e), np.sin(e)], -1)          # pivot -> knee
    nf = np.stack([-np.sin(e), np.cos(e)], -1)         # femur +b (up at e = 0)
    psi = math.pi + e + th
    t = np.stack([np.cos(psi), np.sin(psi)], -1)       # knee -> toe
    nt = np.stack([t[:, 1], -t[:, 0]], -1)             # tibia +b (inboard at the zero pose)
    knee = FEMUR_LEN * ef
    return {"shape": hc.shape, "e": e, "th": th, "ef": ef, "nf": nf, "t": t, "nt": nt, "knee": knee}


def _place(rel, p, yaw_key="yaw"):
    """In-plane (u_rel, z_rel) from the pivot -> (x, z) in the wall frame."""
    c = math.cos(math.radians(p[yaw_key]))
    return np.stack([p["pivot_x"] + c * rel[..., 0], p["pivot_z"] + rel[..., 1]], -1)


def _tibia_local(p, dense):
    loc = (TIB_DENSE if dense else TIB_LOCAL).copy()
    k = p["L_toe"] / TIBIA_LEN_CAD
    loc[:, 0] = np.where(loc[:, 0] > 0, loc[:, 0] * k, loc[:, 0])
    return loc


def knee_actuator_rel(fr, p, n_cap=10):
    """Capsule polygons (M, 2*n_cap, 2) of the knee actuator, relative to the pivot, plus the
    anchor and lever pin (M, 2)."""
    anchor = KNEE_ACT_ANCHOR_FROM_P * fr["ef"]
    a = fr["th"] + math.radians(p["lever_dk"])          # from knee->pivot toward the -t side
    xk = -fr["ef"]
    yk = -fr["nf"]                                       # fold (toe) side of the femur
    # lever pin on the tibia's upper extension: angle theta from xk is the toe; + pi is the
    # extension. Rotating xk by (a + pi) toward the fold side:
    pin = fr["knee"] + p["lever_r"] * (np.cos(a + math.pi)[:, None] * xk
                                       + np.sin(a + math.pi)[:, None] * yk)
    d = pin - anchor
    L = np.linalg.norm(d, axis=-1, keepdims=True)
    ax = d / L
    nx = np.stack([-ax[:, 1], ax[:, 0]], -1)
    rear = anchor - p["act_rear"] * ax
    ang = _capsule(n_cap)
    R = p["act_r"]
    cap_front = pin[:, None, :] + R * (np.cos(ang)[None, :, None] * ax[:, None, :]
                                       + np.sin(ang)[None, :, None] * nx[:, None, :])
    cap_rear = rear[:, None, :] + R * (-np.cos(ang)[None, :, None] * ax[:, None, :]
                                       - np.sin(ang)[None, :, None] * nx[:, None, :])
    poly = np.concatenate([cap_front, cap_rear], 1)
    return poly, anchor, pin, L[:, 0]


def _densify_batch(poly, ds=0.25):
    """(M, N, 2) closed polygons -> (M, N*k, 2) with edges split k times (k from the longest)."""
    nxt = np.roll(poly, -1, 1)
    k = max(1, int(math.ceil(float(np.linalg.norm(nxt - poly, axis=-1).max()) / ds)))
    f = (np.arange(k) / k)[None, None, :, None]
    return (poly[:, :, None, :] + f * (nxt - poly)[:, :, None, :]).reshape(poly.shape[0], -1, 2)


def leg_outlines(hip_counts, knee_counts, p=None, dense=False, frame="wall"):
    """Outlines for broadcast count arrays. Returns a dict with (M, N, 2) polygons for 'femur',
    'tibia', 'toe', 'knee_actuator' and (M, 2) points 'pivot', 'knee', 'toe_tip',
    'act_anchor', 'lever_pin', plus 'femur_elev', 'knee_interior' (deg) and 'knee_rod_in'.
    frame='wall' -> (x, z) with the TRUE yaw p['yaw']; frame='ruler' -> yaw_r; frame='plane'
    -> in-plane (u, z) with u measured from the wall at P (no yaw scaling)."""
    p = P(p)
    fr = frames(hip_counts, knee_counts, p)
    ef, nf, t, nt, knee = fr["ef"], fr["nf"], fr["t"], fr["nt"], fr["knee"]
    fl = FEM_DENSE if dense else FEM_LOCAL
    tl = _tibia_local(p, dense)
    fem = fl[None, :, 0, None] * ef[:, None, :] + fl[None, :, 1, None] * nf[:, None, :]
    tib = knee[:, None, :] + tl[None, :, 0, None] * t[:, None, :] + tl[None, :, 1, None] * nt[:, None, :]
    toe_mask = tl[:, 0] >= p["L_toe"] - TOE_CAP
    toe = tib[:, toe_mask, :]
    tip = knee + p["L_toe"] * t
    act, anchor, pin, rod = knee_actuator_rel(fr, p)
    if dense:
        act = _densify_batch(act)
    rel = {"femur": fem, "tibia": tib, "toe": toe, "knee_actuator": act,
           "pivot": np.zeros_like(knee), "knee": knee, "toe_tip": tip,
           "act_anchor": anchor, "lever_pin": pin}
    if frame == "plane":
        out = {k: v + np.array([p["pivot_x"], p["pivot_z"]]) for k, v in rel.items()}
    else:
        out = {k: _place(v, p, "yaw" if frame == "wall" else "yaw_r") for k, v in rel.items()}
    out["femur_elev"] = np.degrees(fr["e"]).reshape(fr["shape"])
    out["knee_interior"] = np.degrees(fr["th"]).reshape(fr["shape"])
    out["knee_rod_in"] = rod.reshape(fr["shape"])
    return out


def toe_tip(hip_counts, knee_counts, p=None, frame="ruler"):
    o = leg_outlines(hip_counts, knee_counts, p, frame=frame)
    return o["toe_tip"]


# ------------------------------------------------------------------ clearance
def _body_dist(x, z, H=BODY_H):
    """Signed distance of points from the body box x <= 0, 0 <= z <= H (+ outside, - inside)."""
    inside = (x <= 0) & (z >= 0) & (z <= H)
    d = np.hypot(np.maximum(x, 0.0), np.maximum(np.maximum(-z, z - H), 0.0))
    pen = -np.minimum(np.minimum(-x, z), H - z)
    return np.where(inside, pen, d)


def clearances(hip_counts, knee_counts, p=None, parts=("tibia", "femur", "knee_actuator"),
               chunk=2000):
    """Corridor margins (inches, + = clear) for broadcast count arrays. Keys (shape of the
    broadcast): body, table, table_contact, floor, total (min of body/table/floor), and per part
    '<part>_body' etc. Worst plywood face toward the body: x - w*|sin(yaw)|."""
    p = P(p)
    hc, kc = np.broadcast_arrays(np.asarray(hip_counts, float), np.asarray(knee_counts, float))
    shp = hc.shape
    hcr, kcr = hc.ravel(), kc.ravel()
    n = hcr.size
    sy = abs(math.sin(math.radians(p["yaw"])))
    keys = [f"{pt}_{m}" for pt in parts for m in ("body", "table", "table_contact", "floor")]
    out = {k: np.empty(n) for k in keys}
    floor_lim = p["floor_z"] + FLOOR_GAP
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        o = leg_outlines(hcr[a:b], kcr[a:b], p, dense=True, frame="wall")
        for pt in parts:
            w = p["act_w"] if pt == "knee_actuator" else PLY_W
            x = o[pt][..., 0] - w * sy
            z = o[pt][..., 1]
            bd = _body_dist(x, z)
            if pt == "femur":
                bd = np.where(FEM_HUB[None, :], np.inf, bd)
            out[f"{pt}_body"][a:b] = bd.min(-1) - BODY_GAP
            xb = np.where(z < 0, x, np.inf).min(-1)
            out[f"{pt}_table"][a:b] = xb + INBOARD_LIMIT
            out[f"{pt}_table_contact"][a:b] = xb - TABLE_EDGE
            out[f"{pt}_floor"][a:b] = z.min(-1) - floor_lim
    res = {k: v.reshape(shp) for k, v in out.items()}
    for m in ("body", "table", "table_contact", "floor"):
        res[m] = np.minimum.reduce([res[f"{pt}_{m}"] for pt in parts])
    res["total"] = np.minimum.reduce([res["body"], res["table"], res["floor"]])
    return res


def first_belly_contact(knee_counts, p=None, step=1.0, parts=("tibia",)):
    """Hip count where the leg first touches the body while the hip goes 0 -> 1069 at fixed
    knee counts (linear interpolation of the gap); nan if never."""
    c = np.arange(0.0, HIP_MAX + 1e-9, step)
    g = clearances(c, knee_counts, p, parts=parts)["body"]
    hit = np.nonzero(g <= 0)[0]
    if not len(hit):
        return float("nan")
    i = hit[0]
    if i == 0:
        return 0.0
    return float(c[i - 1] + step * g[i - 1] / (g[i - 1] - g[i]))


# ------------------------------------------------------------------ knee actuator lever
def fit_knee_lever(p):
    """Fit (lever_r, lever_dk, rod length at 0) so the anchor->pin distance is linear in knee
    counts at 5.33 counts/mm over the knee map. Returns (r, dk, L0, rms residual in counts)."""
    from scipy.optimize import least_squares
    kc = np.linspace(0, KNEE_SWITCH, 40)

    def res(x):
        q = dict(p, lever_r=x[0], lever_dk=x[1])
        fr = frames(np.zeros_like(kc), kc, q)
        _, _, _, L = knee_actuator_rel(fr, q)
        return (x[2] - L) * MM_PER_IN * COUNTS_PER_MM - kc

    best = None
    for dk0 in (-60, -30, 0, 30, 60):
        r = least_squares(res, [3.0, dk0, 20.0], bounds=([1.5, -120, 5], [8.0, 120, 40]))
        if best is None or r.cost < best.cost:
            best = r
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    return float(best.x[0]), float(best.x[1]), float(best.x[2]), rms


# ------------------------------------------------------------------ joint fit
FIT_NAMES = ["f0", "f1", "f2", "f3", "k0", "k1", "k2", "k3", "L_toe", "yaw_r", "pivot_x", "pivot_z"]


def _unpack(x, base):
    q = dict(base)
    q["fem_coef"] = list(map(float, x[0:4]))
    q["knee_coef"] = list(map(float, x[4:8]))
    q["L_toe"], q["yaw_r"], q["pivot_x"], q["pivot_z"] = map(float, x[8:12])
    return q


def residuals(x, base, ruler=RULER, yaw_fixed=None, monotone=True):
    q = _unpack(x, base)
    if yaw_fixed is not None:
        q["yaw_r"] = yaw_fixed
    r = []
    for c, v, s, _ in FEMUR_DATA:
        r.append((float(femur_elev_deg(c, q)) - v) / s)
    for c, v, s, _ in KNEE_DATA:
        r.append((float(knee_interior_deg(c, q)) - v) / s)
    for k, (h, kn, xm, zm) in ruler.items():
        tp = toe_tip(h, kn, q, frame="ruler")[0]
        r += [(tp[0] - xm) / RULER_SIGMA, (tp[1] - zm) / RULER_SIGMA]
    r.append((q["pivot_x"] - PIVOT[0]) / 0.25)
    r.append((q["pivot_z"] - PIVOT[1]) / 0.25)
    r.append((q["L_toe"] - 32.0) / 1.0)          # repo 32.5, ruler poses 31-33
    if monotone:  # soft penalty if either map is not strictly monotone (>= 0.02 deg/count)
        s = np.linspace(0, 1, 41)
        df = np.polyval(np.polyder(q["fem_coef"][::-1]), s) / HIP_MAX
        dk = np.polyval(np.polyder(q["knee_coef"][::-1]), s) / KNEE_SWITCH
        r.append(10 * np.sum(np.maximum(df + 0.02, 0)))
        r.append(10 * np.sum(np.maximum(0.02 - dk, 0)))
    return np.asarray(r, float)


def fit(base=None, ruler=RULER, yaw_fixed=None):
    from scipy.optimize import least_squares
    base = dict(DEFAULT if base is None else base)
    x0 = np.array(list(DEFAULT["fem_coef"]) + [19.0, 140.0, -40.0, 14.0]
                  + [32.0, 5.0, PIVOT[0], PIVOT[1]], float)
    lo = [-np.inf] * 8 + [29.0, 0.0, 2.0, 1.5]
    hi = [np.inf] * 8 + [35.0, 40.0, 4.0, 3.5]
    best = None
    for y0 in (0.5, 10.0, 22.0, 35.0):
        x0[9] = y0
        r = least_squares(residuals, x0, bounds=(lo, hi), args=(base, ruler, yaw_fixed),
                          x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    q = _unpack(best.x, base)
    if yaw_fixed is not None:
        q["yaw_r"] = yaw_fixed
    q["yaw"] = q["yaw_r"]
    return q, float(2 * best.cost), best


# ------------------------------------------------------------------ validation
def validation(p):
    rows = []

    def add(group, item, model, meas, tol, unit):
        res = model - meas if meas is not None else None
        ok = (abs(res) <= tol) if res is not None else None
        rows.append({"group": group, "item": item, "model": model, "measured": meas,
                     "resid": res, "tol": tol, "unit": unit, "within": ok})

    for k, (h, kn, xm, zm) in RULER.items():
        tp = toe_tip(h, kn, p, frame="ruler")[0]
        add("ruler", f"{k} hip {h} knee {kn} x", float(tp[0]), xm, 1.0, "in")
        add("ruler", f"{k} hip {h} knee {kn} z", float(tp[1]), zm, 1.0, "in")
    for c, v, s, src in FEMUR_DATA:
        add("femur", f"hip {c} ({src})", float(femur_elev_deg(c, p)), v, s, "deg")
    for c, v, src in FEMUR_EXCLUDED:
        add("femur(excl)", f"hip {c} ({src})", float(femur_elev_deg(c, p)), v, 4.0, "deg")
    add("femur", "hip 846 lifted by hand (+2 compliance)", float(femur_elev_deg(846, dict(p, compliance=2.0))), -28.0, 1.0, "deg")
    for c, v, s, src in KNEE_DATA:
        add("knee", f"knee {c} ({src})", float(knee_interior_deg(c, p)), v, s, "deg")
    add("knee(ruler)", "knee 0 (pose A derived)", float(knee_interior_deg(0, p)), 19.0, 2.0, "deg")
    add("knee(ruler)", "knee 157 (B/C derived 40-44)", float(knee_interior_deg(157, p)), 42.0, 2.0, "deg")
    for c, v in KNEE_REFIT_JOB:
        add("knee(job refit)", f"knee {c}", float(knee_interior_deg(c, p)), v, 2.0, "deg")
    # contacts (tibia is the part that touches; femur beyond its hub and the knee actuator
    # are reported too)
    for kn in (0, 3, 12, 24):
        fc = first_belly_contact(kn, p)
        lo, hi = 941 - CONTACT_TOL_COUNTS, 1003 + CONTACT_TOL_COUNTS
        rows.append({"group": "contact", "item": f"first belly contact, knee {kn} (meas 941-1003, +-2 deg)",
                     "model": fc, "measured": None,
                     "resid": (fc - 941 if fc < 941 else (fc - 1003 if fc > 1003 else 0.0)),
                     "tol": float(CONTACT_TOL_COUNTS), "unit": "counts",
                     "within": bool(np.isfinite(fc) and lo <= fc <= hi)})
    hs = np.arange(0, HIP_MAX + 1, 1.0)
    for kn in (157, 166, 175):
        g = clearances(hs, kn, p)
        for pt in ("tibia", "femur", "knee_actuator"):
            v = float(g[f"{pt}_body"].min())
            rows.append({"group": "contact", "item": f"knee {kn} hip 0-1069: min {pt} body gap (meas: clear)",
                         "model": v, "measured": None, "resid": None, "tol": None, "unit": "in",
                         "within": v > 0})
    kc = np.arange(0, KNEE_SWITCH + 1, 1.0)
    g0 = clearances(0.0, kc, p)
    v = float(g0["body"].min())
    rows.append({"group": "contact", "item": "hip 0, knee 0-785: min body gap, all parts (meas: clear)",
                 "model": v, "measured": None, "resid": None, "tol": None, "unit": "in", "within": v > 0})
    o = leg_outlines(1069, 157, p, frame="wall")
    tx, tz = float(o["toe_tip"][0, 0]), float(o["toe_tip"][0, 1])
    rows.append({"group": "contact", "item": "pose D toe under the body (x < 0, z < 0), true yaw",
                 "model": tx, "measured": tz, "resid": None, "tol": None, "unit": "in (x; z in meas col)",
                 "within": tx < 0 and tz < 0})
    rows.append({"group": "contact", "item": "pose D toe inside the table footprint (x < -13.75)",
                 "model": tx - TABLE_EDGE, "measured": None, "resid": None, "tol": None, "unit": "in",
                 "within": tx < TABLE_EDGE})
    return rows


def contact_checks(p):
    """Boolean contact-fact checks for one parameter set (used on ensemble members)."""
    fc = [first_belly_contact(k, p, step=4.0) for k in (0, 24)]
    sweep = clearances(np.arange(0, HIP_MAX + 1, 4.0)[:, None], np.array([157.0, 175.0])[None, :], p,
                       parts=("tibia",))
    hip0 = clearances(0.0, np.arange(0, KNEE_SWITCH + 1, 5.0), p)
    tp = toe_tip(1069, 157, p, frame="wall")[0]
    lo, hi = 941 - CONTACT_TOL_COUNTS, 1003 + CONTACT_TOL_COUNTS
    return {
        "first_contact_k0": fc[0], "first_contact_k24": fc[1],
        "contact_in_941_1003": bool(all(np.isfinite(f) and lo <= f <= hi for f in fc)),
        "contact_before_1003": bool(all(np.isfinite(f) and f <= hi for f in fc)),
        "sweep157_175_clear": bool(sweep["body"].min() > 0),
        "hip0_clear": bool(hip0["body"].min() > 0),
        "poseD_under_body": bool(tp[0] < 0 and tp[1] < 0),
    }


def ruler_chi2(p, interpretation="a"):
    """Sum of squared ruler residuals (1 in sigma). 'a': side readings are true perpendicular
    distances (scaled by the TRUE yaw); 'b': readings were scaled by the fitted yaw_r (e.g.
    in-plane), independent of the true yaw."""
    q = dict(p)
    if interpretation == "a":
        q["yaw_r"] = q["yaw"]
    s = 0.0
    for h, kn, xm, zm in RULER.values():
        tp = toe_tip(h, kn, q, frame="ruler")[0]
        s += (tp[0] - xm) ** 2 + (tp[1] - zm) ** 2
    return float(s)


# ------------------------------------------------------------------ ensemble
ENSEMBLE_DOC = {
    "fem_off0/fem_off1": "U[-2, +2] deg at hip 0 / 1069, linear in between (femur map +-2 deg, allows tilt)",
    "knee_off0/knee_off1": "U[-2, +2] deg at knee 0 / 785, linear in between (knee map +-2 deg)",
    "compliance": "U[0, +2] deg, femur higher than the hanging readings (unloaded); the map +-2 covers lower",
    "pivot_x/pivot_z": "nominal +- U[0.25] in each",
    "L_toe": "nominal +- U[0.75] in (tibia outline toe side rescaled)",
    "yaw": "U[0, 40] deg true plan-view yaw (clearance x = P_x + cos(yaw)*u_rel, worst plywood face)",
    "floor_z": "-34 +- U[0.5] in",
}


def ensemble(n=400, seed=0, base=None, corners=True, checks=True):
    """List of member dicts {name, params, ruler_chi2_a, ruler_chi2_b, checks}. Member 0 is
    the nominal fit; with corners=True the 2^5 corners of (femur offset, knee offset, L_toe,
    yaw, compliance) at the worst-pivot/floor settings are added before the random draws."""
    base = P(base)
    rng = np.random.default_rng(seed)
    mem = [("nominal", dict(base))]
    if corners:
        for fo in (-2, 2):
            for ko in (-2, 2):
                for dl in (-0.75, 0.75):
                    for yw in (0.0, 40.0):
                        for cp in (0.0, 2.0):
                            q = dict(base, fem_off0=fo, fem_off1=fo, knee_off0=ko, knee_off1=ko,
                                     L_toe=base["L_toe"] + dl, yaw=yw, compliance=cp,
                                     pivot_x=base["pivot_x"] - 0.25, pivot_z=base["pivot_z"] - 0.25,
                                     floor_z=FLOOR_Z + 0.5)
                            mem.append((f"corner f{fo:+d} k{ko:+d} L{dl:+.2f} y{yw:.0f} c{cp:.0f}", q))
    for i in range(n):
        q = dict(base)
        q["fem_off0"], q["fem_off1"] = rng.uniform(-2, 2, 2)
        q["knee_off0"], q["knee_off1"] = rng.uniform(-2, 2, 2)
        q["compliance"] = rng.uniform(0, 2)
        q["pivot_x"] = base["pivot_x"] + rng.uniform(-0.25, 0.25)
        q["pivot_z"] = base["pivot_z"] + rng.uniform(-0.25, 0.25)
        q["L_toe"] = base["L_toe"] + rng.uniform(-0.75, 0.75)
        q["yaw"] = rng.uniform(0, 40)
        q["floor_z"] = FLOOR_Z + rng.uniform(-0.5, 0.5)
        mem.append((f"mc:{i}", {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in q.items()}))
    out = []
    for name, q in mem:
        rec = {"name": name, "params": q, "ruler_chi2_a": ruler_chi2(q, "a"),
               "ruler_chi2_b": ruler_chi2(q, "b")}
        if checks:
            rec["checks"] = contact_checks(q)
        out.append(rec)
    return out


# ------------------------------------------------------------------ CLI
def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float) and not np.isfinite(v):
        return "none"
    return f"{v:+.{nd}f}" if isinstance(v, float) else str(v)


def print_table(rows, title):
    print(f"--- validation: {title}")
    print(f"{'group':16s} {'item':62s} {'model':>9s} {'meas':>8s} {'resid':>7s} {'tol':>5s} ok")
    for row in rows:
        tol = "" if row["tol"] is None else f"{row['tol']:.1f}"
        print(f"{row['group']:16s} {row['item']:62s} {_fmt(row['model']):>9s} {_fmt(row['measured']):>8s} "
              f"{_fmt(row['resid']):>7s} {tol:>5s} {_fmt(row['within'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    q, chi2, r = fit()                       # nominal: ruler yaw free (0-40)
    q0, chi2_0, _ = fit(yaw_fixed=0.0)       # yaw-0 variant (most inboard; conservative)
    for m in (q, q0):
        r_, dk, L0, rms = fit_knee_lever(m)
        m.update(lever_r=r_, lever_dk=dk)
    prof = {}
    for y in (0, 10, 20, 30, 35, 40):
        _, c2, _ = fit(yaw_fixed=float(y))
        prof[y] = c2
    loo = {}
    for k in RULER:
        qq, c2, _ = fit(ruler={kk: v for kk, v in RULER.items() if kk != k})
        tp = toe_tip(*RULER[k][:2], qq, frame="ruler")[0]
        loo[k] = (float(tp[0] - RULER[k][2]), float(tp[1] - RULER[k][3]), qq["L_toe"], qq["yaw_r"])
    rows, rows0 = validation(q), validation(q0)
    info = {"params": q, "chi2": chi2, "n_resid": int(len(r.fun)), "params_yaw0": q0, "chi2_yaw0": chi2_0,
            "knee_lever": {"r": q["lever_r"], "dk": q["lever_dk"], "rod0_in": L0, "rms_counts": rms},
            "yaw_profile_chi2": prof, "leave_one_out": loo, "validation": rows, "validation_yaw0": rows0}
    FIT_JSON.write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")

    keys = ("fem_coef", "knee_coef", "L_toe", "yaw_r", "pivot_x", "pivot_z", "lever_r", "lever_dk")
    for name, m, c2 in (("nominal", q, chi2), ("yaw0", q0, chi2_0)):
        print(f"{name}: chi2 {c2:.2f} over {len(r.fun)} residuals;",
              {k: (np.round(m[k], 3).tolist() if isinstance(m[k], list) else round(m[k], 3)) for k in keys})
    print(f"knee lever (nominal): rod@0 {L0:.2f} in, rms {rms:.1f} counts from linear-in-counts")
    print("ruler-yaw profile (joint-fit chi2 with yaw_r fixed):", {k: round(v, 2) for k, v in prof.items()})
    print("leave-one-out ruler residuals (dx, dz, L_toe, yaw_r):",
          {k: tuple(round(x, 2) for x in v) for k, v in loo.items()})
    print()
    print_table(rows, f"nominal (yaw_r = yaw = {q['yaw_r']:.1f})")
    print()
    print_table([rw for rw in rows0 if rw["group"] in ("ruler", "contact")], "yaw0 refit (ruler + contacts)")
    print()
    print("maps (counts -> deg), nominal / yaw0:")
    for c in (0, 100, 200, 300, 400, 478, 600, 700, 800, 846, 900, 941, 1003, 1069):
        print(f"  hip {c:4d}: femur {float(femur_elev_deg(c, q)):+6.1f} / {float(femur_elev_deg(c, q0)):+6.1f}")
    for c in (0, 50, 100, 157, 200, 296, 400, 508, 608, 706, 785):
        print(f"  knee {c:3d}: interior {float(knee_interior_deg(c, q)):6.1f} / {float(knee_interior_deg(c, q0)):6.1f}")

    if args.ensemble:
        ens = ensemble(args.ensemble, args.seed, q)
        out = HERE / "leg_model_v3_ensemble.json"
        out.write_text(json.dumps({"doc": ENSEMBLE_DOC, "base": "nominal fit", "members": ens},
                                  indent=1, default=float), encoding="utf-8")
        mc = [m for m in ens if m["name"].startswith("mc:")]
        cr = [m for m in ens if m["name"].startswith("corner")]
        ck_keys = ("contact_in_941_1003", "contact_before_1003", "sweep157_175_clear", "hip0_clear",
                   "poseD_under_body")
        for label, grp in (("random", mc), ("corners", cr)):
            if not grp:
                continue
            ck = {k: round(float(np.mean([m["checks"][k] for m in grp])), 3) for k in ck_keys}
            fc0 = np.array([m["checks"]["first_contact_k0"] for m in grp])
            fc24 = np.array([m["checks"]["first_contact_k24"] for m in grp])
            print(f"ensemble {label} ({len(grp)}): pass fractions {ck}")
            print(f"   first contact knee 0 p5/p50/p95 {np.nanpercentile(fc0, [5, 50, 95]).round(0)}, "
                  f"knee 24 {np.nanpercentile(fc24, [5, 50, 95]).round(0)}; "
                  f"ruler chi2 (a) median {np.median([m['ruler_chi2_a'] for m in grp]):.1f}, "
                  f"(b) median {np.median([m['ruler_chi2_b'] for m in grp]):.1f}")
        print("wrote", out)
    print("wrote", FIT_JSON)


if __name__ == "__main__":
    main()
