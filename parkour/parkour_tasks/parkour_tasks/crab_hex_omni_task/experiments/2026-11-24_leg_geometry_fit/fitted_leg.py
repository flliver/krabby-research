"""Fitted front-left (FL) leg model: hall counts -> leg geometry, body frame, INCHES.

MODEL ONLY -- nothing here opens a port or talks to the robot. numpy only.

Fitted to the 2026-11-24 ruler measurements (poses A-D), the belly-contact fact (knee 0-24,
hip 941-1003) and the no-contact fact (knee 157-175, full hip sweep) by ``fit_leg.py`` in this
directory; the fitted numbers live in ``FITTED`` below (rewritten by ``fit_leg.py --write``).

Fixed geometry (repo: crab_hex_dimensions.py, crab_hex_foot_fk.py, crab_hex_leg_profiles.py):
  body box 28 (x) x 48 (y) x 12.5 (z) in, centre at the origin, z up, +x travel, left legs at -y;
  FL yaw axis on the side wall at x = -11.5 in (outer axis 2.5 in from the body end);
  femur pivot 2.5 in outboard of the wall, 3.25 in below the belly; femur 23.0 in hinge-hinge;
  tibia 32.5 in knee-to-toe; CAD part outlines for the femur and the tibia (rounded toe).

Hardware facts used (2026-11-24): both pitch actuators 5.33 counts/mm of rod. Hip count 0 = rod
fully retracted (curl, femur up); 1069 = fully extended (~200 mm). Knee count 0 = curl end (rod
extended, leg's own stop); 779 = rod fully retracted (146 mm less rod). The knee actuator rides on
the FEMUR, so the interior knee angle depends on knee counts only (no hip coupling).

Leg-plane frame used internally: u = distance from the side wall measured IN the (yawed) leg
plane (+ outboard), h = height above the belly plane (- below). The ruler measures are
``side = u*cos(psi)`` (horizontal distance from the side wall, + outward) and ``height = h``.

Angle conventions returned:
  hip_deg       femur angle below horizontal (+ = down, - = raised/curled), repo sim convention;
  knee_interior interior femur-tibia angle (deg), small = folded;
  psi           total leg-plane yaw (deg) = mount splay (-15) + cam yaw; only |psi| is identified
                by the side/height data (cos(psi)); the sign is taken negative like the splay.

Linkages (fit parameters, see PARAM_DOC):
  hip:  rod from an anchor fixed to the hip plate at (uA, zA) from the femur pivot to a pin on
        the femur at radius a_f, rotated dp deg from the femur axis toward its +b (upper) edge;
        pin-to-pin length Lh0 at count 0, + counts/5.33 mm.
  knee: rod from an anchor on the femur, D in from the knee toward the pivot (on the femur
        axis), to a lever pin on the tibia at radius r above the knee, rotated dk deg toward the
        tibia's +b (actuator-arm) edge. Rod length at count 0 is set by the interior stop b0;
        the rod shortens by counts/5.33 mm.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

# ------------------------------------------------------------------ fixed, measured geometry
COUNTS_PER_MM = 5.33
MM_PER_IN = 25.4
HIP_FULL_COUNTS = 1069
KNEE_FULL_COUNTS = 779

BODY_HALF = (14.0, 24.0, 6.25)          # x, y, z half extents of the body box (in)
BELLY_Z = -6.25                          # belly plane, body frame
WALL_Y = 24.0                            # side wall |y|
FL_MOUNT = (-11.5, -24.0)                # FL yaw axis (x, y) on the wall
SY = -1.0                                # left leg: outboard is -y
PIVOT_UH = (2.5, -3.25)                  # femur pivot (u, h): 2.5 in outboard, 3.25 in below the belly
FEMUR_LEN = 23.0
TIBIA_LEN = 32.5
TABLE_EDGE_INBOARD = 13.75               # table 20.5 in wide, centred: edges 13.75 in inboard of each wall

# CAD outlines (crab_hex_leg_profiles.py, inches). femur: a along the part (pivot -11.5, knee +11.5),
# b across (+b = up at the zero pose). tibia: a up the part (knee +12.835, toe -19.665), b across
# (+b = actuator-arm side, inboard at the zero pose).
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


def _densify(poly: np.ndarray, ds: float) -> np.ndarray:
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        m = max(1, int(math.ceil(float(np.linalg.norm(q - p)) / ds)))
        out.append(p + (np.arange(m) / m)[:, None] * (q - p))
    return np.vstack(out)


# densified copies (<= 1 in edges) for clearance work; the plain ones are returned as polygons
FEMUR_DENSE = _densify(FEMUR_OUTLINE, 1.0)
TIBIA_DENSE = _densify(TIBIA_OUTLINE, 1.0)

PARAM_NAMES = ("a_f", "dp", "uA", "zA", "Lh0", "r", "dk", "D", "b0", "psi", "ta", "tb")
EXTRA_PARAMS = ("pu", "ph")  # femur pivot (u, h); fixed at the repo values in the primary fit
PARAM_DOC = {
    "a_f": "hip rod pin radius on the femur from the pivot (in)",
    "dp": "hip pin angular offset from the femur axis toward its +b/upper edge (deg)",
    "uA": "hip anchor u from the pivot (in, + outboard), on the hip plate",
    "zA": "hip anchor height above the pivot (in)",
    "Lh0": "hip pin-to-pin rod length at count 0, fully retracted (in)",
    "r": "knee lever pin radius from the knee pivot (in)",
    "dk": "knee lever pin angular offset from the tibia axis toward its +b/actuator-arm edge (deg)",
    "D": "knee anchor distance from the knee toward the femur pivot, on the femur axis (in)",
    "b0": "interior femur-tibia angle at knee count 0, the curl stop (deg)",
    "psi": "total leg-plane yaw vs the side-wall normal (deg; sign not identified)",
    "ta": "measured toe reference point: distance from the knee along the tibia axis (in)",
    "tb": "measured toe reference point: offset across the tibia toward +b (in)",
    "pu": "femur pivot outboard of the side wall (in; repo 2.5, fixed in the primary fit)",
    "ph": "femur pivot height relative to the belly (in; repo -3.25, fixed in the primary fit)",
}

# >>> FITTED (written by fit_leg.py --write) >>>
FITTED = {
    "a_f": 4.42968,
    "dp": -11.0483,
    "uA": -0.328437,
    "zA": 20.7348,
    "Lh0": 17.2412,
    "r": 3.39872,
    "dk": 41.4318,
    "D": 18.3035,
    "b0": 8.04894,
    "psi": -6.43152,
    "ta": 31.3488,
    "tb": -0.145629,
    "pu": 2.5,
    "ph": -3.25,
}
# <<< FITTED <<<


def _p(p: Mapping[str, float] | None) -> dict:
    q = dict(FITTED)
    if p is not None:
        q.update(p)
    return q


# ------------------------------------------------------------------ counts -> angles
def hip_rod_in(hip_counts, p=None):
    p = _p(p)
    return p["Lh0"] + np.asarray(hip_counts, float) / COUNTS_PER_MM / MM_PER_IN


def hip_cos_arg(hip_counts, p=None):
    """Law-of-cosines argument cos(gamma); |arg| <= 1 on the operating branch."""
    p = _p(p)
    A = math.hypot(p["uA"], p["zA"])
    L = hip_rod_in(hip_counts, p)
    return (A * A + p["a_f"] ** 2 - L * L) / (2.0 * A * p["a_f"])


def hip_angle_deg(hip_counts, p=None):
    """Femur angle below horizontal (deg) for hip counts (0 = curl/retracted)."""
    p = _p(p)
    phiA = math.atan2(p["zA"], p["uA"])
    g = np.arccos(np.clip(hip_cos_arg(hip_counts, p), -1.0, 1.0))
    return np.degrees(g - phiA + math.radians(p["dp"]))


def hip_counts_for_angle(hip_deg, p=None):
    """Inverse of hip_angle_deg (may fall outside 0..1069)."""
    p = _p(p)
    A = math.hypot(p["uA"], p["zA"])
    phiA = math.atan2(p["zA"], p["uA"])
    g = np.radians(np.asarray(hip_deg, float)) + phiA - math.radians(p["dp"])
    L = np.sqrt(A * A + p["a_f"] ** 2 - 2 * A * p["a_f"] * np.cos(g))
    return (L - p["Lh0"]) * MM_PER_IN * COUNTS_PER_MM


def knee_rod0_in(p=None):
    p = _p(p)
    x0 = math.radians(p["b0"] + p["dk"])
    return math.sqrt(p["D"] ** 2 + p["r"] ** 2 + 2 * p["r"] * p["D"] * math.cos(x0))


def knee_cos_arg(knee_counts, p=None):
    p = _p(p)
    L = knee_rod0_in(p) - np.asarray(knee_counts, float) / COUNTS_PER_MM / MM_PER_IN
    return (L * L - p["D"] ** 2 - p["r"] ** 2) / (2.0 * p["r"] * p["D"])


def knee_interior_deg(knee_counts, p=None):
    """Interior femur-tibia angle (deg) for knee counts (0 = curl stop, 779 = rod retracted)."""
    p = _p(p)
    x = np.arccos(np.clip(knee_cos_arg(knee_counts, p), -1.0, 1.0))
    return np.degrees(x) - p["dk"]


def knee_counts_for_interior(interior_deg, p=None):
    p = _p(p)
    x = np.radians(np.asarray(interior_deg, float) + p["dk"])
    L = np.sqrt(p["D"] ** 2 + p["r"] ** 2 + 2 * p["r"] * p["D"] * np.cos(x))
    return (knee_rod0_in(p) - L) * MM_PER_IN * COUNTS_PER_MM


# ------------------------------------------------------------------ leg-plane geometry
def plane_geometry(hip_deg, interior_deg, p=None, dense=False, parts=("femur", "tibia")):
    """Leg-plane (u, h) geometry for angle arrays of shape (M,). Returns a dict of arrays:
    pivot (2,), knee/toe/toe_tip/lever_pin/knee_anchor/hip_pin (M, 2), hip_anchor (2,),
    femur/tibia polygons (M, N, 2)."""
    p = _p(p)
    th = np.radians(np.atleast_1d(np.asarray(hip_deg, float)))
    be = np.radians(np.broadcast_to(np.asarray(interior_deg, float), th.shape))
    piv = np.array([p.get("pu", PIVOT_UH[0]), p.get("ph", PIVOT_UH[1])], float)
    ef = np.stack([np.cos(th), -np.sin(th)], -1)          # pivot -> knee
    nf = np.stack([np.sin(th), np.cos(th)], -1)           # femur +b (up at the zero pose)
    knee = piv + FEMUR_LEN * ef
    ang_t = math.pi - th + be                             # tibia knee->toe direction (CCW from +u)
    t = np.stack([np.cos(ang_t), np.sin(ang_t)], -1)
    nt = np.stack([t[:, 1], -t[:, 0]], -1)                # tibia +b = t rotated clockwise
    toe = knee + p["ta"] * t + p["tb"] * nt
    toe_tip = knee + TIBIA_LEN * t
    xk = -ef                                              # knee frame: x toward the pivot
    yk = -nf                                              # y toward the fold (toe) side
    a = be + math.radians(p["dk"])
    lever = knee - p["r"] * (np.cos(a)[:, None] * xk + np.sin(a)[:, None] * yk)
    kanchor = knee + p["D"] * xk
    hip_pin = piv + p["a_f"] * np.stack([np.cos(-th + math.radians(p["dp"])),
                                         np.sin(-th + math.radians(p["dp"]))], -1)
    out = {"pivot": piv, "knee": knee, "toe": toe, "toe_tip": toe_tip, "lever_pin": lever,
           "knee_anchor": kanchor, "hip_pin": hip_pin, "hip_anchor": piv + np.array([p["uA"], p["zA"]])}
    if "femur" in parts:
        fo = FEMUR_DENSE if dense else FEMUR_OUTLINE
        fa = fo[:, 0] - FEMUR_PIVOT_A
        out["femur"] = piv + fa[None, :, None] * ef[:, None, :] + fo[None, :, 1, None] * nf[:, None, :]
    if "tibia" in parts:
        to = TIBIA_DENSE if dense else TIBIA_OUTLINE
        ta_ = TIBIA_KNEE_A - to[:, 0]
        out["tibia"] = (knee[:, None, :] + ta_[None, :, None] * t[:, None, :]
                        + to[None, :, 1, None] * nt[:, None, :])
    return out


def side_of(u, p=None, w=0.0):
    """Ruler 'side' (horizontal distance from the side wall, + out) of leg-plane u."""
    psi = math.radians(_p(p)["psi"])
    return np.asarray(u, float) * math.cos(psi) - w * math.sin(psi)


def to_body(uh, p=None, w=0.0):
    """Leg-plane (..., 2) (u, h) -> FL body frame (..., 3) in inches (w = out-of-plane offset)."""
    psi = math.radians(_p(p)["psi"])
    uh = np.asarray(uh, float)
    v = np.stack([np.full(uh.shape[:-1], w), SY * uh[..., 0], uh[..., 1] + BELLY_Z], -1)
    c, s = math.cos(psi), math.sin(psi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.array([FL_MOUNT[0], FL_MOUNT[1], 0.0]) + v @ R.T


def max_height_inboard(sh: np.ndarray, side_limit: float) -> np.ndarray:
    """(M, N, 2) closed polygons in (side, height): max height of the part with side <= side_limit
    (vertices plus exact edge crossings); -inf where no part is that far inboard."""
    s, h = sh[..., 0], sh[..., 1]
    hv = np.where(s <= side_limit, h, -np.inf).max(-1)
    s2, h2 = np.roll(s, -1, -1), np.roll(h, -1, -1)
    cross = (s - side_limit) * (s2 - side_limit) < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (side_limit - s) / (s2 - s)
    hc = np.where(cross, h + tt * (h2 - h), -np.inf).max(-1)
    return np.maximum(hv, hc)


def clearances(hip_counts, knee_counts, p=None, table_edge=TABLE_EDGE_INBOARD, parts=("tibia",)):
    """Vectorised clearance measures over arrays of counts (broadcast together). Returns
    belly_gap (in; vertical gap from the belly plane down to the highest femur/tibia point that
    is inboard of the side wall; negative = the leg is inside the body; +inf = nothing inboard),
    table_top_gap (same against the table top plane for the part inboard of the table edge),
    min_side (most inboard point of the femur+tibia outline; >= 0 means the whole leg is
    outside the wall), toe_side, toe_height.
    Default parts = the tibia only: the femur cannot reach inboard of the wall (its pivot-end cap is
    concentric with the pivot, radius 2.5 in = the pivot's offset from the wall, so it is tangent to
    the wall plane at 0.75-5.75 in below the belly). The CAD polygon's end cap bulges up to 0.044 in
    past that radius, which would otherwise show up as a spurious -0.04 in min_side."""
    p = _p(p)
    hc, kc = np.broadcast_arrays(np.asarray(hip_counts, float), np.asarray(knee_counts, float))
    shp = hc.shape
    g = plane_geometry(hip_angle_deg(hc.ravel(), p), knee_interior_deg(kc.ravel(), p), p, dense=True, parts=parts)
    cps = math.cos(math.radians(p["psi"]))
    poly = np.concatenate([g[k] for k in parts], 1)
    sh = np.stack([poly[..., 0] * cps, poly[..., 1]], -1)
    out = {
        "belly_gap": -max_height_inboard(sh, 0.0),
        "table_top_gap": -max_height_inboard(sh, -table_edge),
        "min_side": sh[..., 0].min(-1),
        "toe_side": g["toe"][:, 0] * cps,
        "toe_height": g["toe"][:, 1],
    }
    return {k: v.reshape(shp) for k, v in out.items()}


def leg_pose(hip_counts: float, knee_counts: float, p=None) -> dict:
    """Everything for one pose, in FL BODY coordinates (inches; origin body centre, z up, +x
    travel, left legs at -y): pivot, knee, toe (the fitted ruler reference point), toe_tip
    (outline end on the tibia axis), lever_pin, knee_anchor, hip_pin, hip_anchor, femur_outline
    (36x3) and tibia_outline (24x3) polygons (mid-plane of the 1-in plywood), plus angles and the
    ruler measures (side/height for the toe, same convention as the 2026-11-24 data) and
    clearances (belly_gap, table_top_gap, min_side)."""
    p = _p(p)
    th = float(hip_angle_deg(hip_counts, p))
    be = float(knee_interior_deg(knee_counts, p))
    g = plane_geometry(th, be, p)
    cl = clearances(hip_counts, knee_counts, p)
    fem_side = float(g["femur"][0][:, 0].min() * math.cos(math.radians(p["psi"])))
    out = {k: to_body(v[0] if v.ndim == 2 and k not in ("pivot", "hip_anchor") else v, p)
           for k, v in g.items() if k not in ("femur", "tibia")}
    out["femur_outline"] = to_body(g["femur"][0], p)
    out["tibia_outline"] = to_body(g["tibia"][0], p)
    out.update({
        "hip_deg": th, "femur_up_deg": -th, "knee_interior_deg": be,
        "hip_rod_mm": float(hip_rod_in(hip_counts, p)) * MM_PER_IN,
        "knee_rod_mm": (knee_rod0_in(p) - knee_counts / COUNTS_PER_MM / MM_PER_IN) * MM_PER_IN,
        "toe_side": float(cl["toe_side"]), "toe_height": float(cl["toe_height"]),
        "belly_gap": float(cl["belly_gap"]), "table_top_gap": float(cl["table_top_gap"]),
        "min_side": float(cl["min_side"]), "femur_min_side": fem_side,
    })
    return out


def measures(hip_counts, knee_counts, p=None):
    """(side, height) of the fitted toe reference point, comparable with the ruler data
    (floats for scalar counts, arrays for array counts)."""
    p = _p(p)
    hc, kc = np.broadcast_arrays(np.asarray(hip_counts, float), np.asarray(knee_counts, float))
    g = plane_geometry(hip_angle_deg(hc.ravel(), p), knee_interior_deg(kc.ravel(), p), p, parts=())
    side = (g["toe"][:, 0] * math.cos(math.radians(p["psi"]))).reshape(hc.shape)
    height = g["toe"][:, 1].reshape(hc.shape)
    if side.ndim == 0:
        return float(side), float(height)
    return side, height


def first_belly_contact(knee_counts: float, p=None, step: float = 2.0) -> float:
    """Hip count (interpolated) at which the leg first touches the belly/side wall while the hip
    uncurls 0 -> 1069 at fixed knee counts; nan if it never does."""
    c = np.arange(0.0, HIP_FULL_COUNTS + 1e-9, step)
    gap = clearances(c, knee_counts, p)["belly_gap"]
    hit = np.nonzero(gap <= 0.0)[0]
    if not len(hit):
        return float("nan")
    i = hit[0]
    if i == 0 or not np.isfinite(gap[i - 1]):
        return float(c[i])
    return float(c[i - 1] + step * gap[i - 1] / (gap[i - 1] - gap[i]))


MEASURED = {  # pose: (hip counts, knee counts, side, height), inches, +-1 in ruler
    "A": (0, 0, 5.0, -10.0),
    "B": (0, 157, 16.0, -13.0),
    "C": (475, 157, 5.0, -19.0),
    "D": (1069, 157, -16.5, -6.0),
}

if __name__ == "__main__":
    for k, (hc, kc, s, h) in MEASURED.items():
        ms, mh = measures(hc, kc)
        print(f"{k}: hip {hc:4d} knee {kc:3d}  model side {ms:+6.2f} height {mh:+6.2f}  "
              f"measured {s:+6.1f} {h:+6.1f}  resid {ms - s:+5.2f} {mh - h:+5.2f}")
    print("first belly contact, knee 0 / 24:", first_belly_contact(0), first_belly_contact(24))
