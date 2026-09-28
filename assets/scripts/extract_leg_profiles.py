#!/usr/bin/env python3
"""Extract the leg-part outlines from the CAD SVG into a checked-in data module.

Reads ``~/krabby/joint_specs/KrabV3-Legs.svg`` (the source that generated the CNC cut
files), samples the Hip6 / Femur6 / Tibia6 ``Body`` outlines (cubic beziers subsampled,
elliptical arcs sampled by ANGLE via the W3C endpoint->center conversion -- chord
shortcuts clip ~1.6 in off rounded end caps), maps each outline into its sim LINK frame,
decimates with Douglas-Peucker, and writes
``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/crab_hex_leg_profiles.py``.

Link frames (inches; the USD generator converts to meters):
- femur:  (a, b) = (along part toward the outboard/knee hinge, across width);
          origin at the part center; hinges at a = +-11.5.
- tibia:  (a, b) = (up the part, across width); origin at the part center of the
          AS-BUILT length (39.33 in); knee at a = +12.835, toe at a = -19.665. The SVG
          part's below-knee span (31.335 in) is stretched to the as-built 32.5 in.
- hip:    (a, b) = (up the plate, outboard across width); origin at the plate center;
          femur pivot at a = -9.5005, b = 0; top tip a = +12.8335.

Re-run whenever the CAD changes:  python3 assets/scripts/extract_leg_profiles.py
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_PATH = Path.home() / "krabby" / "joint_specs" / "KrabV3-Legs.svg"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = (
    REPO_ROOT
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "mdp"
    / "crab_hex_leg_profiles.py"
)

NS_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
MM = 1 / 25.4  # SVG user units are mm (viewBox 2438.4 for a 96 in sheet)

IDENT = (1, 0, 0, 1, 0, 0)


def mat_mul(a, b):
    return (
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    )


def mat_apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def parse_transform(s):
    m = IDENT
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", s or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "matrix":
            t = tuple(v)
        elif name == "rotate":
            a = math.radians(v[0])
            c, s_ = math.cos(a), math.sin(a)
            if len(v) > 1:
                cx, cy = v[1], v[2]
                t = mat_mul(
                    mat_mul((1, 0, 0, 1, cx, cy), (c, s_, -s_, c, 0, 0)),
                    (1, 0, 0, 1, -cx, -cy),
                )
            else:
                t = (c, s_, -s_, c, 0, 0)
        else:
            t = IDENT
        m = mat_mul(m, t)
    return m


def _sample_arc(p0, rx, ry, phi_deg, large_arc, sweep, p1, n):
    """W3C SVG elliptical-arc endpoint->center conversion, sampled by angle."""
    if rx == 0 or ry == 0 or p0 == p1:
        return [p1]
    phi = math.radians(phi_deg)
    cphi, sphi = math.cos(phi), math.sin(phi)
    dx2, dy2 = (p0[0] - p1[0]) / 2.0, (p0[1] - p1[1]) / 2.0
    x1p = cphi * dx2 + sphi * dy2
    y1p = -sphi * dx2 + cphi * dy2
    rx, ry = abs(rx), abs(ry)
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large_arc == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cphi * cxp - sphi * cyp + (p0[0] + p1[0]) / 2.0
    cy = sphi * cxp + cphi * cyp + (p0[1] + p1[1]) / 2.0

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        mag = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / mag)))
        return a if ux * vy - uy * vx >= 0 else -a

    th1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi
    out = []
    for j in range(1, n + 1):
        th = th1 + dth * j / n
        x = cphi * rx * math.cos(th) - sphi * ry * math.sin(th) + cx
        y = sphi * rx * math.cos(th) + cphi * ry * math.sin(th) + cy
        out.append((x, y))
    return out


_TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def sample_path(d: str, n_curve: int = 12):
    pts = []
    cmd = None
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    toks = []
    for c, num in _TOK.findall(d):
        toks.append(c if c else float(num))
    i = 0

    def take(k):
        nonlocal i
        v = toks[i : i + k]
        i += k
        return v

    def cubic(p0, c1, c2, p1):
        out = []
        for j in range(1, n_curve + 1):
            t = j / n_curve
            mt = 1 - t
            out.append(
                (
                    mt**3 * p0[0] + 3 * mt * mt * t * c1[0] + 3 * mt * t * t * c2[0] + t**3 * p1[0],
                    mt**3 * p0[1] + 3 * mt * mt * t * c1[1] + 3 * mt * t * t * c2[1] + t**3 * p1[1],
                )
            )
        return out

    while i < len(toks):
        t = toks[i]
        if isinstance(t, str):
            cmd = t
            i += 1
        if i > len(toks):
            break
        if cmd in "Zz":
            cur = start
            continue
        if cmd == "M":
            x, y = take(2)
            cur = (x, y)
            start = cur
            pts.append(cur)
            cmd = "L"
        elif cmd == "m":
            x, y = take(2)
            cur = (cur[0] + x, cur[1] + y)
            start = cur
            pts.append(cur)
            cmd = "l"
        elif cmd == "L":
            x, y = take(2)
            cur = (x, y)
            pts.append(cur)
        elif cmd == "l":
            x, y = take(2)
            cur = (cur[0] + x, cur[1] + y)
            pts.append(cur)
        elif cmd == "H":
            (x,) = take(1)
            cur = (x, cur[1])
            pts.append(cur)
        elif cmd == "h":
            (x,) = take(1)
            cur = (cur[0] + x, cur[1])
            pts.append(cur)
        elif cmd == "V":
            (y,) = take(1)
            cur = (cur[0], y)
            pts.append(cur)
        elif cmd == "v":
            (y,) = take(1)
            cur = (cur[0], cur[1] + y)
            pts.append(cur)
        elif cmd == "C":
            v = take(6)
            pts += cubic(cur, (v[0], v[1]), (v[2], v[3]), (v[4], v[5]))
            cur = (v[4], v[5])
        elif cmd == "c":
            v = take(6)
            c1 = (cur[0] + v[0], cur[1] + v[1])
            c2 = (cur[0] + v[2], cur[1] + v[3])
            p1 = (cur[0] + v[4], cur[1] + v[5])
            pts += cubic(cur, c1, c2, p1)
            cur = p1
        elif cmd in "Aa":
            v = take(7)
            p1 = (v[5], v[6]) if cmd == "A" else (cur[0] + v[5], cur[1] + v[6])
            pts += _sample_arc(cur, v[0], v[1], v[2], bool(v[3]), bool(v[4]), p1, n_curve)
            cur = p1
        else:
            i += 1
    return pts


def douglas_peucker(pts, eps):
    if len(pts) < 3:
        return list(pts)

    def dseg(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def rec(lo, hi):
        if hi <= lo + 1:
            return [lo, hi]
        dmax, idx = -1.0, None
        for k in range(lo + 1, hi):
            dd = dseg(pts[k], pts[lo], pts[hi])
            if dd > dmax:
                dmax, idx = dd, k
        if dmax > eps:
            left = rec(lo, idx)
            right = rec(idx, hi)
            return left[:-1] + right
        return [lo, hi]

    keep = rec(0, len(pts) - 1)
    return [pts[k] for k in keep]


def extract() -> dict[str, list[tuple[float, float]]]:
    tree = ET.parse(SVG_PATH)
    root = tree.getroot()
    found: dict[str, list[tuple[float, float]]] = {}

    targets = {
        "Layer 1/Legs/Hip/Hip6/Body": "hip",
        "Layer 1/Legs/Femur/Femur6/Body": "femur",
        "Layer 1/Legs/Tibia/Tibia6/Body": "tibia",
    }

    def walk(el, m, labels):
        m2 = mat_mul(m, parse_transform(el.get("transform")))
        lab = el.get(NS_LABEL) or ""
        chain = labels + ([lab.strip()] if lab else [])
        key = "/".join(chain)
        if el.tag.endswith("path") and key in targets:
            pts = [mat_apply(m2, x, y) for x, y in sample_path(el.get("d", ""))]
            found[targets[key]] = [(x * MM, y * MM) for x, y in pts]
        for ch in el:
            walk(ch, m2, chain)

    walk(root, IDENT, [])
    assert set(found) == {"hip", "femur", "tibia"}, f"missing outlines: {found.keys()}"
    return found


def to_link_frames(raw):
    """Map SVG-inch outlines into the link frames documented in the module header."""
    out = {}

    # --- femur: SVG x along the part, hinges (BigSockets) at x = 35.583 / 58.583 ---
    fx = [p[0] for p in raw["femur"]]
    fy = [p[1] for p in raw["femur"]]
    cx = (35.583 + 58.583) / 2.0  # midpoint between the hinges = part center
    cy = (min(fy) + max(fy)) / 2.0
    out["femur"] = [(x - cx, y - cy) for x, y in raw["femur"]]

    # --- tibia: SVG y down the part; knee socket at y = 6.319; stretch below-knee to
    #     the as-built 32.5 in (denominator = the SAMPLED outline extent, not the
    #     control-point bbox); a = up-the-part in the link frame ---
    knee_y = 6.319
    svg_below = max(p[1] for p in raw["tibia"]) - knee_y
    stretch = 32.5 / svg_below
    tx = [p[0] for p in raw["tibia"]]
    cx_t = (min(tx) + max(tx)) / 2.0
    pts = []
    for x, y in raw["tibia"]:
        d = y - knee_y  # >0 below the knee (toward the toe)
        if d > 0:
            d *= stretch
        a = 12.835 - d  # knee at +12.835 in the as-built link frame
        pts.append((a, x - cx_t))
    # Normalize orientation: the above-knee actuator arm faces +b (the generator maps
    # +b to the INBOARD side per leg).
    arm_b = [b for a, b in pts if a > 14.0]
    if arm_b and (sum(arm_b) / len(arm_b)) < 0:
        pts = [(a, -b) for a, b in pts]
    out["tibia"] = pts

    # --- hip: SVG x along the plate; femur pivot (BigSocket) at x = 42.312, top tip at
    #     the far end; a = up-the-plate, b = across the width. The pivot is the frame
    #     datum (a = -9.5005, i.e. 3.25 in below the body in the sim link frame) --
    #     the sampled plate is shorter than the control-point bbox suggested, which
    #     only shifts the free tips, never the pivot. ---
    hy = [p[1] for p in raw["hip"]]
    cy_h = (min(hy) + max(hy)) / 2.0
    hip_pts = [((x - 42.312) - 9.5005, y - cy_h) for x, y in raw["hip"]]
    # Normalize orientation: the STRAIGHT (door-hinge) edge faces +b (the generator maps
    # +b to the wall/inboard side per leg). Straightness = lower variance of the edge.
    plus = [b for a, b in hip_pts if b > 0]
    minus = [-b for a, b in hip_pts if b < 0]

    def spread(vals):
        if not vals:
            return 1e9
        mu = sum(vals) / len(vals)
        return sum((v - mu) ** 2 for v in vals) / len(vals)

    if spread(plus) > spread(minus):
        hip_pts = [(a, -b) for a, b in hip_pts]
    out["hip"] = hip_pts
    return out


def symmetry_deviation(pts):
    """Max distance between the outline and its mirror about the long-axis centerline."""
    import bisect

    mirrored = sorted((a, -b) for a, b in pts)
    xs = [p[0] for p in mirrored]
    worst = 0.0
    for a, b in pts:
        i = bisect.bisect_left(xs, a)
        best = min(
            (math.hypot(a - ma, b - mb) for ma, mb in mirrored[max(0, i - 4) : i + 4]),
            default=1e9,
        )
        worst = max(worst, best)
    return worst


def main():
    raw = extract()
    frames = to_link_frames(raw)
    lines = [
        '"""CAD leg-part outlines in link frames (inches). GENERATED -- do not hand-edit.',
        "",
        f"Source: {SVG_PATH}",
        "Regenerate with: python3 assets/scripts/extract_leg_profiles.py",
        "Frames documented in that script's header. (a, b) tuples, closed polygons",
        "(first point not repeated). Tibia below-knee region stretched to the as-built",
        "32.5 in knee-to-toe (SVG part is 31.335).",
        '"""',
        "",
    ]
    for name in ("hip", "femur", "tibia"):
        pts = frames[name]
        # close the polygon for decimation, then drop the duplicate endpoint
        dec = douglas_peucker(pts + [pts[0]], eps=0.05)[:-1]
        sym = symmetry_deviation(dec)
        span_a = max(p[0] for p in dec) - min(p[0] for p in dec)
        span_b = max(p[1] for p in dec) - min(p[1] for p in dec)
        print(
            f"{name}: {len(pts)} -> {len(dec)} pts, span {span_a:.2f} x {span_b:.2f} in,"
            f" centerline symmetry dev {sym:.3f} in"
        )
        lines.append(f"# {name}: {len(dec)} points, {span_a:.2f} x {span_b:.2f} in")
        lines.append(f"{name.upper()}_OUTLINE_IN = [")
        for a, b in dec:
            lines.append(f"    ({a:.4f}, {b:.4f}),")
        lines.append("]")
        lines.append("")
    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
