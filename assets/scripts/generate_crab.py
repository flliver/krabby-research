#!/usr/bin/env python3
"""Deterministically generate the crab-hexapod training plant from measured hardware dimensions.

Single source of truth: ``parkour_tasks/crab_hex_forward_task/mdp/crab_hex_dimensions.py``
(loaded directly by file path -- stdlib only, no Isaac/torch needed). Outputs:

    python3 assets/scripts/generate_crab.py                  # assets/crab.usda -- the MAIN asset
                                                             #   (plant of record "A15+B": 15 deg
                                                             #   outer-mount splay, outer yaw axes
                                                             #   2.5 in from the body ends)
    python3 assets/scripts/generate_crab.py --legacy-golden  # assets/variants/crab_simple__splay00_axis5p5in.usda
                                                             #   -- the 2026-08-20 build (splay 0, 5.5 in),
                                                             #   byte-pinned for the legacy heads
    python3 assets/scripts/generate_crab.py --all-variants   # assets/variants/*.usda + MANIFEST.md
    python3 assets/scripts/generate_crab.py --splay-deg 10 --outer-axis-in 5.5 --out /tmp/x.usda

The main asset and every variant are byte-pinned to this generator by
``tests/unit/test_crab_hex_usd_generation.py``: never hand-edit a generated USDA, change the
dimensions module (or this file) and regenerate. ``assets/crab_simple.usda`` is NOT generated: it is
the hand-authored Cube model of the 2026-08-09 campaign baseline, kept as a historical reference
(sha-pinned by the same test) and never written by this script.

Conventions preserved from the hand-authored asset this replaces:
- every link is a unit ``Cube`` with ``xformOp:scale`` giving its metric dimensions, so all
  ``physics:localPos0/1`` (and ``physics:centerOfMass``) are authored in PRE-SCALE units:
  divide metric offsets by the owning prim's scale;
- prim ordering (legs FL,FR,RL,RR,ML,MR; joints FL,FR,ML,MR,RL,RR), the ``Joints`` scope,
  material defs, the OmniGraph ``Position_Controller`` block, and the +Z root lift;
- right legs (FR/MR/RR) mirror via ``localRot0 = (0, 0, 1, 0)`` (180 deg about Z) on the
  Hip_Femur and Femur_Tibia joints, with knee limits mirrored accordingly.

Unlike the hand-authored file, joint anchors and cosmetic prim translates are emitted from
the same numbers, so they can never drift apart (the old asset had up to ~0.18 m of
anchor-vs-translate disagreement on the CamShaft prims).
"""

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_DIMS_PATH = (
    REPO_ROOT
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "mdp"
    / "crab_hex_dimensions.py"
)
_spec = importlib.util.spec_from_file_location("crab_hex_dimensions", _DIMS_PATH)
dims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dims)
_PROF_PATH = _DIMS_PATH.parent / "crab_hex_leg_profiles.py"
_pspec = importlib.util.spec_from_file_location("crab_hex_leg_profiles", _PROF_PATH)
profiles = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(profiles)

# Root lift: cosmetic stage-view height of /krabby (runtime spawn overrides it via
# init_state.pos). Chosen so the zero-pose toe (z = -1.0668) clears the stage ground.
ROOT_LIFT_Z = 1.1

# Leg prim order under /krabby (historical order, preserved for articulation stability).
LEG_PRIM_ORDER = ("FL", "FR", "RL", "RR", "ML", "MR")
# Joint emission order inside the Joints scope (also historical).
JOINT_LEG_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
LEFT = {"FL", "ML", "RL"}


@dataclass(frozen=True)
class MorphVariant:
    """Leg-mount geometry variant (PLAN G leg-mount morphology campaign, 2026-09-02).

    Both fields are MOUNT transforms applied to the front/rear rows only; the mid legs
    never move. The whole leg chain (hip plate, cam rotor, femur, tibia, footpad) rotates /
    translates rigidly with its mount, so joint-local anchors, link masses, inertias, the
    cam mapping and the linkage are untouched.

    - ``row_splay_deg``: outward yaw of the outer rows, symmetric about the transverse
      mid-plane (row F toes toward -x, row R toes toward +x; robot stays reversible). Hard
      cap 20 deg (user, 2026-09-02): the +-25 deg cam throw must still reach the
      perpendicular pose for sideways walking. Implemented as a rotation of the mount
      frame (``xformOp:orient`` on the leg prims + ``localRot0`` on the two Z-axis joints),
      never as a ``Body_Hip`` joint default (that would park the cam sweep off-centre --
      the 2026-08-13 perpendicular-mounts lesson).
    - ``outer_axis_from_end_in``: distance of the outer yaw axes from the body ends along
      the 28-in wall (2026-08-20 build: 5.5 in; plant of record: 2.5 in). Smaller = axes
      closer to the corners.

    Defaults are the plant of record (A15+B) and reproduce ``assets/crab.usda`` byte-for-byte;
    ``LEGACY_GOLDEN_VARIANT`` reproduces ``assets/variants/crab_simple__splay00_axis5p5in.usda``
    (the 2026-08-20 measured-hardware build, the "golden" plant of every pre-a15b head).
    """

    row_splay_deg: float = dims.OUTER_ROW_SPLAY_DEG
    outer_axis_from_end_in: float = dims.OUTER_LEG_AXIS_FROM_BODY_END_IN

    def __post_init__(self) -> None:
        if not 0.0 <= self.row_splay_deg <= 20.0:
            raise ValueError(f"row_splay_deg must be within [0, 20] deg, got {self.row_splay_deg}")
        if not 0.0 < self.outer_axis_from_end_in < dims.BODY_LENGTH_X_IN / 2.0:
            raise ValueError(
                f"outer_axis_from_end_in must lie inside the half body length, got "
                f"{self.outer_axis_from_end_in}"
            )

    @property
    def tag(self) -> str:
        return f"splay{int(round(self.row_splay_deg)):02d}_axis{self.outer_axis_from_end_in:g}in".replace(".", "p")


DEFAULT_VARIANT = MorphVariant()
LEGACY_GOLDEN_VARIANT = MorphVariant(
    row_splay_deg=dims.LEGACY_OUTER_ROW_SPLAY_DEG,
    outer_axis_from_end_in=dims.LEGACY_OUTER_LEG_AXIS_FROM_BODY_END_IN,
)
MAIN_ASSET = REPO_ROOT / "assets" / "crab.usda"
VARIANTS_DIR = REPO_ROOT / "assets" / "variants"
# Hand-authored historical asset (2026-08-09 campaign-baseline Cube model); never generated here.
HAND_AUTHORED_ASSET = REPO_ROOT / "assets" / "crab_simple.usda"

# Named plants (the training-side table in ``crab_hex_phases.PLANTS`` is locked to this one by
# a unit test). ``A15+B`` IS the main asset; its variant file is kept byte-identical so the runs
# that recorded ``assets/variants/crab_simple__splay15_axis2p5in.usda`` stay resolvable.
# ``legacy_golden`` is the 2026-08-20 measured-hardware build (splay 0, axes 5.5 in).
VARIANTS: dict[str, MorphVariant] = {
    "A15+B": DEFAULT_VARIANT,
    "legacy_golden": LEGACY_GOLDEN_VARIANT,
    "B": MorphVariant(0.0, 2.5),
    "A10": MorphVariant(10.0, 5.5),
    "A15": MorphVariant(15.0, 5.5),
    "A20": MorphVariant(20.0, 5.5),
    "A10+B": MorphVariant(10.0, 2.5),
    "A20+B": MorphVariant(20.0, 2.5),
}
VARIANT_NOTES = {
    "A15+B": "plant of record = assets/crab.usda (this file is byte-identical to it)",
    "legacy_golden": "2026-08-20 measured-hardware build (splay 0, axes 5.5 in); every head between the 2026-08-20 rebuild and the a15b lineage",
    "B": "re-hinge only", "A10": "splay only", "A15": "splay only", "A20": "splay only",
    "A10+B": "splay + re-hinge", "A20+B": "splay + re-hinge",
}


def variant_asset_path(name: str) -> Path:
    """Where the named plant's USDA lives under ``assets/variants/`` (the main plant's variant file
    is byte-identical to ``assets/crab.usda``)."""
    if name not in VARIANTS:
        raise KeyError(f"unknown plant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS_DIR / f"crab_simple__{VARIANTS[name].tag}.usda"


def variant_files() -> dict[str, Path]:
    """Every named plant's file under ``assets/variants/`` (incl. ``legacy_golden``)."""
    return {n: variant_asset_path(n) for n in VARIANTS}


def manifest_text() -> str:
    rows = ["| plant | tag | splay (deg) | outer axis (in) | file | note |", "|---|---|---|---|---|---|"]
    for name, v in VARIANTS.items():
        rel = variant_asset_path(name).relative_to(REPO_ROOT)
        rows.append(f"| `{name}` | `{v.tag}` | {v.row_splay_deg:g} | {v.outer_axis_from_end_in:g} | "
                    f"`{rel}` | {VARIANT_NOTES.get(name, '')} |")
    return (
        "# Crab-hexapod plant variants\n\n"
        "GENERATED by `assets/scripts/generate_crab.py --all-variants` -- do not hand-edit.\n\n"
        "The MAIN asset is `assets/crab.usda` (plant of record `A15+B`: 15 deg outward splay of the "
        "front/rear leg mounts, outer yaw axes 2.5 in from the body ends). `legacy_golden` is the "
        "2026-08-20 measured-hardware build (splay 0, axes 5.5 in) that every head from the 2026-08-20 "
        "rebuild up to the a15b lineage was trained on. `assets/crab_simple.usda` is NOT in this table: "
        "it is the hand-authored Cube model of the 2026-08-09 campaign baseline, kept as a historical "
        "reference and never generated. Select a plant "
        "by name with `KRABBY_PLANT=<plant>` (training) or `--plant <plant>` (gait harness); no "
        "variable is needed for the main asset. Every file is byte-pinned to the generator "
        "(`tests/unit/test_crab_hex_usd_generation.py`).\n\n" + "\n".join(rows) + "\n"
    )



def n(v: float) -> str:
    """Format a number the way the asset does: up to 10 significant digits, no -0."""
    s = f"{v:.10g}"
    return "0" if s in ("-0", "-0.0") else s


def v3(x: float, y: float, z: float) -> str:
    return f"({n(x)}, {n(y)}, {n(z)})"


# ---------------------------------------------------------------------------
# CAD-outline mesh helpers (PLAN C): the three leg links are extruded 2D
# outlines from crab_hex_leg_profiles.py, with mass properties computed from
# the polygon (shoelace area/centroid/second moments) so inertia honestly
# follows the taper. Pure python, deterministic.
# ---------------------------------------------------------------------------
def _signed_area(poly):
    a = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        a += x0 * y1 - x1 * y0
    return a / 2.0


def ensure_ccw(poly):
    return list(poly) if _signed_area(poly) > 0 else list(reversed(poly))


def ear_clip(poly):
    """Triangulate a simple CCW polygon; returns index triples into poly."""
    idx = list(range(len(poly)))
    tris = []

    def cross(o, a, b):
        return (poly[a][0] - poly[o][0]) * (poly[b][1] - poly[o][1]) - (
            poly[a][1] - poly[o][1]
        ) * (poly[b][0] - poly[o][0])

    def inside(p, a, b, c):
        (px, py), (ax, ay), (bx, by), (cx, cy) = poly[p], poly[a], poly[b], poly[c]
        d1 = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        d2 = (px - bx) * (cy - by) - (py - by) * (cx - bx)
        d3 = (px - cx) * (ay - cy) - (py - cy) * (ax - cx)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n_ = len(idx)
        clipped = False
        for k in range(n_):
            o, a, b = idx[(k - 1) % n_], idx[k], idx[(k + 1) % n_]
            if cross(o, a, b) <= 1e-12:
                continue  # reflex or degenerate
            if any(
                inside(q, o, a, b)
                for q in idx
                if q not in (o, a, b)
            ):
                continue
            tris.append((o, a, b))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break  # fall back: fan the remainder (near-degenerate leftovers)
    if len(idx) >= 3:
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


def poly_props(poly):
    """Area, centroid, and centroidal second moments (about axes p and q) of a CCW polygon."""
    A = Cx = Cy = Ixx = Iyy = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        cr = x0 * y1 - x1 * y0
        A += cr
        Cx += (x0 + x1) * cr
        Cy += (y0 + y1) * cr
        Ixx += (y0 * y0 + y0 * y1 + y1 * y1) * cr  # about the x-axis: integral of y^2
        Iyy += (x0 * x0 + x0 * x1 + x1 * x1) * cr  # about the y-axis: integral of x^2
    A /= 2.0
    Cx /= 6.0 * A
    Cy /= 6.0 * A
    Ixx = Ixx / 12.0 - A * Cy * Cy
    Iyy = Iyy / 12.0 - A * Cx * Cx
    return A, Cx, Cy, Ixx, Iyy


def mesh_geom_text(pts_2d, thickness, axis_map, indent="            "):
    """USD points/faces for an extruded outline.

    ``axis_map(p, q, x)`` maps in-plane coords (p, q) + thickness offset x to local (x, y, z).
    """
    poly = ensure_ccw(pts_2d)
    tris = ear_clip(poly)
    N = len(poly)
    half = thickness / 2.0
    points = [axis_map(p, q, half) for p, q in poly] + [axis_map(p, q, -half) for p, q in poly]
    # Mirrored axis maps flip the handedness; detect via the basis determinant and
    # reverse face windings so normals always point outward (meshes are single-sided).
    o = axis_map(0.0, 0.0, 0.0)
    e1 = [a - b for a, b in zip(axis_map(1.0, 0.0, 0.0), o)]
    e2 = [a - b for a, b in zip(axis_map(0.0, 1.0, 0.0), o)]
    e3 = [a - b for a, b in zip(axis_map(0.0, 0.0, 1.0), o)]
    det = (
        e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
        - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
        + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0])
    )
    flip = det < 0
    counts, indices = [], []
    for (i, j, k) in tris:  # cap at +thickness
        counts.append(3); indices += ([i, k, j] if flip else [i, j, k])
    for (i, j, k) in tris:  # cap at -thickness (opposite winding)
        counts.append(3); indices += ([N + i, N + j, N + k] if flip else [N + k, N + j, N + i])
    for i in range(N):  # side quads
        j = (i + 1) % N
        counts.append(4); indices += ([i, N + i, N + j, j] if flip else [i, j, N + j, N + i])
    xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
    ext = f"[({n(min(xs))}, {n(min(ys))}, {n(min(zs))}), ({n(max(xs))}, {n(max(ys))}, {n(max(zs))})]"
    pts_txt = ", ".join(f"({n(x)}, {n(y)}, {n(z)})" for x, y, z in points)
    return (
        f"{indent}float3[] extent = {ext}\n"
        f"{indent}int[] faceVertexCounts = {counts}\n"
        f"{indent}int[] faceVertexIndices = {indices}\n"
        f"{indent}point3f[] points = [{pts_txt}]"
    )


def extruded_mass_props(pts_2d, thickness, mass, axis_map):
    """CoM (local) and frame-aligned diagonal inertia of the extruded outline at ``mass``.

    In-plane axes (p, q) map through ``axis_map``; thickness runs along the local axis that
    ``axis_map`` assigns the x argument to.
    """
    poly = ensure_ccw(pts_2d)
    A, cp, cq, Ip, Iq = poly_props(poly)  # Ip = integral q^2 (about p-axis), Iq = integral p^2
    sigma = mass / A
    i_thick = sigma * (Ip + Iq)
    i_p = sigma * Ip + mass * thickness * thickness / 12.0
    i_q = sigma * Iq + mass * thickness * thickness / 12.0
    com = axis_map(cp, cq, 0.0)
    # Map the three magnitudes onto local axes: the axis that receives the thickness
    # offset gets i_thick; the axis receiving p gets the moment ABOUT p... note the
    # moment about the axis carrying p-variation is i_q etc. Resolve by probing.
    ex = axis_map(1.0, 0.0, 0.0)  # direction of p
    ey = axis_map(0.0, 1.0, 0.0)  # direction of q
    et = axis_map(0.0, 0.0, 1.0)  # direction of thickness
    origin = axis_map(0.0, 0.0, 0.0)
    def direction(v):
        d = [v[i] - origin[i] for i in range(3)]
        return max(range(3), key=lambda i: abs(d[i]))
    ax_p, ax_q, ax_t = direction(ex), direction(ey), direction(et)
    diag = [0.0, 0.0, 0.0]
    diag[ax_t] = i_thick
    # Rotation about the axis pointing in the p-direction resists q-spread (Ip = ∫q² dA)
    # plus the thickness spread; symmetrically for q. i_p/i_q include the thickness term.
    diag[ax_p] = i_p
    diag[ax_q] = i_q
    return com, tuple(diag)


class Leg:
    """All authored numbers for one leg, derived from the dimensions module."""

    def __init__(self, name: str, variant: MorphVariant = DEFAULT_VARIANT):
        IN = dims.IN_TO_M
        self.name = name
        self.variant = variant
        self.sy = -1.0 if name in LEFT else 1.0  # left legs extend toward -Y
        row = name[0]  # F / M / R
        outer_x = (dims.BODY_LENGTH_X_IN / 2.0 - variant.outer_axis_from_end_in) * IN
        self.x = {"F": -outer_x, "M": 0.0, "R": outer_x}[row]

        wall_y = dims.LEG_MOUNT_Y_M
        pivot_z = dims.FEMUR_PIVOT_Z_M
        plate_len = dims.HIP_PLATE_LENGTH_IN * IN
        plate_w = dims.HIP_PLATE_WIDTH_IN * IN
        femur_len = dims.FEMUR_PART_LENGTH_IN * IN
        tibia_len = dims.TIBIA_PART_LENGTH_IN * IN
        tibia_above = dims.TIBIA_ABOVE_KNEE_IN * IN
        thk = dims.PLY_THICKNESS_IN * IN
        plate_h = dims.HIP_PLATE_WIDTH_IN * IN  # femur/tibia plates are 5 in wide too

        sy = self.sy
        self.pivot_z = pivot_z
        self.yaw_y = sy * wall_y
        self.hip_center_y = sy * (wall_y + plate_w / 2.0)
        self.hip_center_z = dims.HIP_PLATE_CENTER_Z_M  # == 0 by the invariant
        self.pivot_y = sy * (wall_y + dims.FEMUR_PIVOT_OUTBOARD_M)
        self.femur_center_y = sy * (
            wall_y + dims.FEMUR_PIVOT_OUTBOARD_M + dims.FEMUR_HINGE_TO_HINGE_M / 2.0
        )
        self.knee_y = sy * (
            wall_y + dims.FEMUR_PIVOT_OUTBOARD_M + dims.FEMUR_HINGE_TO_HINGE_M
        )
        self.tibia_center_z = pivot_z - (tibia_len / 2.0 - tibia_above)
        self.toe_z = pivot_z - dims.TIBIA_KNEE_TO_TOE_M

        # --- PLAN G mount splay: rigid rotation of the whole leg about the vertical yaw
        # axis through (x, sy*wall_y). Outward = leading-row (R, +x) toes toward +x and
        # trailing-row (F, -x) toes toward -x on BOTH sides: alpha = -row_sign * sy * splay.
        # Mid legs never splay. alpha == 0 is the unsplayed-mount code path (mid legs, the
        # legacy asset and the splay-0 variants).
        row_sign = {"F": -1.0, "M": 0.0, "R": 1.0}[row]
        self.alpha = -row_sign * sy * math.radians(variant.row_splay_deg)
        if self.alpha == 0.0:
            self.orient = "(1, 0, 0, 0)"
        else:
            # USD quaternions are authored (w, x, y, z); rotation about +Z by alpha.
            self.orient = f"({n(math.cos(self.alpha / 2.0))}, 0, 0, {n(math.sin(self.alpha / 2.0))})"

        def place(y: float, z: float) -> tuple[float, float, float]:
            """Prim translate for a link whose unsplayed centre is (x, y, z)."""
            if self.alpha == 0.0:
                return (self.x, y, z)
            dy = y - self.yaw_y
            return (
                self.x - math.sin(self.alpha) * dy,
                self.yaw_y + math.cos(self.alpha) * dy,
                z,
            )

        self.place = place

        # Joint anchors. The yaw anchor on the (still unit-cube) body stays pre-scale;
        # anchors on the three MESH links are METRIC local offsets (mesh points are
        # authored in meters with scale (1,1,1)).
        self.yaw_pos0 = (
            self.x / dims.BODY_SIZE_M[0],
            sy * 0.5,
            self.hip_center_z / dims.BODY_SIZE_M[2],
        )
        self.yaw_pos1_y = -sy * plate_w / 2.0  # plate's inboard edge, at the wall (metric)
        self.hipfemur_pos0_z = pivot_z - self.hip_center_z  # -0.2413 on the plate
        self.hipfemur_pos1_y = -sy * dims.FEMUR_HINGE_TO_HINGE_M / 2.0
        self.kneejoint_pos0_y = sy * dims.FEMUR_HINGE_TO_HINGE_M / 2.0
        self.kneejoint_pos1_z = tibia_len / 2.0 - tibia_above  # knee above tibia center
        self.footpad_pos0_z = -tibia_len / 2.0  # toe at the as-built bottom end

        # --- PLAN C: CAD-outline meshes (crab_hex_leg_profiles.py, inches -> meters).
        # Axis maps (outline (a, b) -> prim local): thickness always local x.
        #   hip:   a = up the plate -> z; +b = wall/hinge side -> inboard (+y on left legs)
        #   femur: a = toward the knee hinge -> outboard (-y on left); b = width -> z
        #   tibia: a = up the part -> z; +b = actuator-arm side -> inboard
        def m_pts(raw):
            return [(a * IN, b * IN) for a, b in raw]

        hip_map = lambda a_, b_, xx: (xx, -sy * b_, a_)  # noqa: E731
        femur_map = lambda a_, b_, xx: (xx, sy * a_, b_)  # noqa: E731
        tibia_map = lambda a_, b_, xx: (xx, -sy * b_, a_)  # noqa: E731
        m_l = dims.LEG_LINK_MASSES_KG
        self.femur_geom = mesh_geom_text(m_pts(profiles.FEMUR_OUTLINE_IN), thk, femur_map)
        self.femur_com, self.femur_diag = extruded_mass_props(
            m_pts(profiles.FEMUR_OUTLINE_IN), thk, m_l["femur"], femur_map
        )
        self.tibia_geom = mesh_geom_text(m_pts(profiles.TIBIA_OUTLINE_IN), thk, tibia_map)
        self.tibia_com, self.tibia_diag = extruded_mass_props(
            m_pts(profiles.TIBIA_OUTLINE_IN), thk, m_l["tibia"], tibia_map
        )
        self.hip_geom = mesh_geom_text(m_pts(profiles.HIP_OUTLINE_IN), thk, hip_map)
        # Hip = composite: plywood mesh + the two actuator point masses (their (b, z)
        # plate positions come from crab_hex_dimensions). Products of inertia from the
        # off-axis point masses are dropped (diagonal approximation, documented).
        m_ply = m_l["hip"] - dims.HIP_ACT_POINT_MASS_KG - dims.KNEE_ACT_POINT_MASS_KG
        ply_com, ply_diag = extruded_mass_props(
            m_pts(profiles.HIP_OUTLINE_IN), thk, m_ply, hip_map
        )
        pts_masses = [
            (m_ply, ply_com, ply_diag),
            (
                dims.HIP_ACT_POINT_MASS_KG,
                hip_map(dims.HIP_ACT_CG_PLATE_IN[1] * IN, dims.HIP_ACT_CG_PLATE_IN[0] * IN, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                dims.KNEE_ACT_POINT_MASS_KG,
                hip_map(dims.KNEE_ACT_CG_PLATE_IN[1] * IN, dims.KNEE_ACT_CG_PLATE_IN[0] * IN, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ]
        M = sum(mm for mm, _, _ in pts_masses)
        com = tuple(sum(mm * c[i] for mm, c, _ in pts_masses) / M for i in range(3))
        diag = [0.0, 0.0, 0.0]
        for mm, c, dg in pts_masses:
            d = [c[i] - com[i] for i in range(3)]
            diag[0] += dg[0] + mm * (d[1] * d[1] + d[2] * d[2])
            diag[1] += dg[1] + mm * (d[0] * d[0] + d[2] * d[2])
            diag[2] += dg[2] + mm * (d[0] * d[0] + d[1] * d[1])
        self.hip_com, self.hip_diag = com, tuple(diag)

        self.flip = name not in LEFT  # 180-deg Z frame flip on right-leg pitch joints
        knee_lo, knee_hi = (
            dims.KNEE_SIM_LIMITS_LEFT_DEG if not self.flip else dims.KNEE_SIM_LIMITS_RIGHT_DEG
        )
        self.knee_limits = (knee_lo, knee_hi)


def leg_block(leg: Leg) -> str:
    p = leg.name
    m = dims.LEG_LINK_MASSES_KG
    return f'''    def Xform "Leg_{p}"
    {{
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        def Mesh "{p}_Femur" (
            prepend apiSchemas = ["PhysicsMassAPI", "MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            float physics:mass = {n(m["femur"])}
            point3f physics:centerOfMass = {v3(*leg.femur_com)}
            float3 physics:diagonalInertia = {v3(*leg.femur_diag)}
{leg.femur_geom}
            rel material:binding = </krabby/PlyWood> (
                bindMaterialAs = "weakerThanDescendants"
            )
            uniform token physics:approximation = "convexHull"
            bool physics:collisionEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:rigidBodyEnabled = 1
            uniform token subdivisionScheme = "none"
            quatd xformOp:orient = {leg.orient}
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = {v3(*leg.place(leg.femur_center_y, leg.pivot_z))}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}

        def Mesh "{p}_Tibia" (
            prepend apiSchemas = ["PhysicsMassAPI", "MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            float physics:mass = {n(m["tibia"])}
            point3f physics:centerOfMass = {v3(*leg.tibia_com)}
            float3 physics:diagonalInertia = {v3(*leg.tibia_diag)}
{leg.tibia_geom}
            rel material:binding = </krabby/PlyWood> (
                bindMaterialAs = "weakerThanDescendants"
            )
            uniform token physics:approximation = "convexHull"
            bool physics:collisionEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:rigidBodyEnabled = 1
            uniform token subdivisionScheme = "none"
            quatd xformOp:orient = {leg.orient}
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = {v3(*leg.place(leg.knee_y, leg.tibia_center_z))}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}

        def Cube "{p}_Footpad" (
            prepend apiSchemas = ["PhysicsMassAPI", "MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI"]
        )
        {{
            float physics:mass = {n(dims.FOOTPAD_MASS_KG)}
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            rel material:binding = </krabby/PlyWood> (
                bindMaterialAs = "weakerThanDescendants"
            )
            uniform token purpose = "guide"
            bool physics:collisionEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:rigidBodyEnabled = 1
            double size = 1
            quatd xformOp:orient = {leg.orient}
            double3 xformOp:scale = (0.06, 0.06, 0.04)
            double3 xformOp:translate = {v3(*leg.place(leg.knee_y, leg.toe_z))}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}

        def Mesh "{p}_Hip" (
            prepend apiSchemas = ["PhysicsMassAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float physics:mass = {n(m["hip"])}
            point3f physics:centerOfMass = {v3(*leg.hip_com)}
            float3 physics:diagonalInertia = {v3(*leg.hip_diag)}
{leg.hip_geom}
            rel material:binding = </krabby/PlyWood> (
                bindMaterialAs = "weakerThanDescendants"
            )
            uniform token physics:approximation = "convexHull"
            bool physics:collisionEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:rigidBodyEnabled = 1
            uniform token subdivisionScheme = "none"
            quatd xformOp:orient = {leg.orient}
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = {v3(*leg.place(leg.hip_center_y, leg.hip_center_z))}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}

        def Cube "{p}_CamShaft" (
            prepend apiSchemas = ["PhysicsMassAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        )
        {{
            float physics:mass = {n(dims.CAMSHAFT_MASS_KG)}
            float3 physics:diagonalInertia = {v3(dims.CAMSHAFT_DIAGONAL_INERTIA, dims.CAMSHAFT_DIAGONAL_INERTIA, dims.CAMSHAFT_DIAGONAL_INERTIA)}
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            bool physics:collisionEnabled = 0
            bool physics:kinematicEnabled = 0
            bool physics:rigidBodyEnabled = 1
            double size = 1
            quatd xformOp:orient = {leg.orient}
            double3 xformOp:scale = (0.02, 0.02, 0.02)
            double3 xformOp:translate = {v3(*leg.place(leg.yaw_y, leg.hip_center_z))}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}
    }}
'''


def joint_blocks(leg: Leg) -> str:
    p = leg.name
    rot0_flip = "(0, 0, 1, 0)" if leg.flip else "(1, 0, 0, 0)"
    knee_lo, knee_hi = leg.knee_limits
    hip_lo, hip_hi = dims.HIP_SIM_LIMITS_DEG
    yaw_lim = dims.YAW_HARD_LIMIT_DEG
    return f'''        def PhysicsRevoluteJoint "{p}_Body_Hip_RevoluteJoint"
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </krabby/chassis/body>
            rel physics:body1 = </krabby/Leg_{p}/{p}_Hip>
            float physics:breakForce = 3.4028235e38
            float physics:breakTorque = 3.4028235e38
            bool physics:collisionEnabled = 1
            point3f physics:localPos0 = {v3(*leg.yaw_pos0)}
            point3f physics:localPos1 = {v3(0, leg.yaw_pos1_y, 0)}
            quatf physics:localRot0 = {leg.orient}
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {n(-yaw_lim)}
            float physics:upperLimit = {n(yaw_lim)}
        }}

        def PhysicsRevoluteJoint "{p}_Body_CamShaft_RevoluteJoint" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular"]
        )
        {{
            float drive:angular:physics:damping = 50000
            float drive:angular:physics:stiffness = 800000
            uniform token physics:axis = "Z"
            rel physics:body0 = </krabby/chassis/body>
            rel physics:body1 = </krabby/Leg_{p}/{p}_CamShaft>
            float physics:breakForce = 3.4028235e38
            float physics:breakTorque = 3.4028235e38
            point3f physics:localPos0 = {v3(*leg.yaw_pos0)}
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = {leg.orient}
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}

        def PhysicsRevoluteJoint "{p}_Hip_Femur_RevoluteJoint" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular"]
        )
        {{
            float drive:angular:physics:damping = 50000
            float drive:angular:physics:stiffness = 800000
            uniform token physics:axis = "X"
            rel physics:body0 = </krabby/Leg_{p}/{p}_Hip>
            rel physics:body1 = </krabby/Leg_{p}/{p}_Femur>
            float physics:breakForce = 3.4028235e38
            float physics:breakTorque = 3.4028235e38
            point3f physics:localPos0 = {v3(0, 0, leg.hipfemur_pos0_z)}
            point3f physics:localPos1 = {v3(0, leg.hipfemur_pos1_y, 0)}
            quatf physics:localRot0 = {rot0_flip}
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {n(hip_lo)}
            float physics:upperLimit = {n(hip_hi)}
        }}

        def PhysicsRevoluteJoint "{p}_Femur_Tibia_RevoluteJoint" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular"]
        )
        {{
            float drive:angular:physics:damping = 50000
            float drive:angular:physics:stiffness = 800000
            uniform token physics:axis = "X"
            rel physics:body0 = </krabby/Leg_{p}/{p}_Femur>
            rel physics:body1 = </krabby/Leg_{p}/{p}_Tibia>
            float physics:breakForce = 3.4028235e38
            float physics:breakTorque = 3.4028235e38
            point3f physics:localPos0 = {v3(0, leg.kneejoint_pos0_y, 0)}
            point3f physics:localPos1 = {v3(0, 0, leg.kneejoint_pos1_z)}
            quatf physics:localRot0 = {rot0_flip}
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {n(knee_lo)}
            float physics:upperLimit = {n(knee_hi)}
        }}
'''


def fixed_joint_block(leg: Leg) -> str:
    p = leg.name
    return f'''        def PhysicsFixedJoint "{p}_Tibia_Footpad_FixedJoint"
        {{
            rel physics:body0 = </krabby/Leg_{p}/{p}_Tibia>
            rel physics:body1 = </krabby/Leg_{p}/{p}_Footpad>
            float physics:breakForce = 3.4028235e38
            float physics:breakTorque = 3.4028235e38
            point3f physics:localPos0 = {v3(0, 0, leg.footpad_pos0_z)}
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}
'''


HEADER = '''#usda 1.0
(
    customLayerData = {
        dictionary cameraSettings = {
            dictionary Front = {
                double3 position = (5, 0, 0)
                double radius = 5
            }
            dictionary Perspective = {
                double3 position = (7.59322740928944, -5.5307511890275105, 7.849931929740069)
                double3 target = (-0.000004844810097992536, 0.000008416391422905178, 0.000008098486583563158)
            }
            dictionary Right = {
                double3 position = (0, -5, 0)
                double radius = 5
            }
            dictionary Top = {
                double3 position = (0, 0, 5)
                double radius = 5
            }
            string boundCamera = "/OmniverseKit_Persp"
        }
        dictionary omni_layer = {
            string authoring_layer = "./crab_simple.usda"
            dictionary locked = {
            }
            dictionary muteness = {
            }
        }
        dictionary renderSettings = {
        }
    }
    defaultPrim = "krabby"
    endTimeCode = 1000000
    metersPerUnit = 1
    startTimeCode = 0
    timeCodesPerSecond = 60
    upAxis = "Z"
)

def Xform "World"
{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI"]
    )
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        uniform token physxScene:broadphaseType = "GPU"
        bool physxScene:enableGPUDynamics = 1
    }

    def Xform "GroundPlane"
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        def Mesh "CollisionMesh"
        {
            uniform bool doubleSided = 0
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
            point3f[] points = [(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)]
            color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
                interpolation = "varying"
            )
            quatf xformOp:orient = (1, 0, 0, 0)
            float3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = (0, 1.7833524788354762, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }

        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }
    }

    def SphereLight "SphereLight" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float3[] extent = [(-50, -50, -50), (50, 50, 50)]
        color3f inputs:color = (0.5, 1, 0.5)
        float inputs:intensity = 1000000
        float inputs:radius = 0.5
        float inputs:shaping:cone:angle = 45
        float inputs:shaping:cone:softness = 0.1
        float inputs:shaping:focus
        color3f inputs:shaping:focusTint
        asset inputs:shaping:ies:file
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 7)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }

    def DistantLight "defaultLight" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:angle = 1
        float inputs:colorTemperature = 6500
        float inputs:intensity = 300
        float inputs:shaping:cone:angle = 180
        float inputs:shaping:cone:softness
        float inputs:shaping:focus
        color3f inputs:shaping:focusTint
        asset inputs:shaping:ies:file
        quatd xformOp:orient = (0.6532814824381883, 0.2705980500730985, 0.27059805007309845, 0.6532814824381882)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }
}

def "Render" (
    hide_in_stage_window = true
    no_delete = true
)
{
    def "OmniverseKit"
    {
        def "HydraTextures" (
            hide_in_stage_window = true
            no_delete = true
        )
        {
            def RenderProduct "omni_kit_widget_viewport_ViewportTexture_0" (
                prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1", "OmniRtxPostColorGradingAPI_1", "OmniRtxPostChromaticAberrationAPI_1", "OmniRtxPostBloomPhysicalAPI_1", "OmniRtxPostMatteObjectAPI_1", "OmniRtxPostCompositingAPI_1", "OmniRtxPostDofAPI_1", "OmniRtxPostMotionBlurAPI_1", "OmniRtxPostTvNoiseAPI_1", "OmniRtxPostTonemapIrayReinhardAPI_1", "OmniRtxPostDebugSettingsAPI_1", "OmniRtxDebugSettingsAPI_1"]
                hide_in_stage_window = true
                no_delete = true
            )
            {
                rel camera = </OmniverseKit_Persp>
                token omni:rtx:background:source:texture:textureMode = "repeatMirrored"
                token omni:rtx:background:source:type = "domeLight"
                bool omni:rtx:dlss:frameGeneration = 0
                bool omni:rtx:material:db:nonVisualMaterialCSV:enabled = 0
                bool omni:rtx:material:db:syncLoads = 1
                token omni:rtx:post:dlss:execMode = "performance"
                bool omni:rtx:post:registeredCompositing:invertColorCorrection = 1
                bool omni:rtx:post:registeredCompositing:invertToneMap = 1
                bool omni:rtx:pt:lightcache:cached:dontResolveConflicts = 1
                int omni:rtx:pt:maxSamplesPerLaunch = 2073600
                int omni:rtx:pt:mgpu:maxPixelsPerRegionExponent = 12
                token omni:rtx:rendermode
                color3f omni:rtx:rt:ambientLight:color = (0.1, 0.1, 0.1)
                bool omni:rtx:rt:demoire = 0
                bool omni:rtx:rt:ecoMode:enabled
                bool omni:rtx:rt:lightcache:spatialCache:dontResolveConflicts = 1
                bool omni:rtx:scene:hydra:materialSyncLoads = 1
                bool omni:rtx:scene:hydra:mdlMaterialWarmup = 1
                uint omni:rtx:viewTile:limit = 4294967295
                rel orderedVars = </Render/Vars/LdrColor>
                custom bool overrideClipRange = 0
                uniform int2 resolution = (1280, 720)
            }
        }
    }

    def RenderSettings "OmniverseGlobalRenderSettings" (
        prepend apiSchemas = ["OmniRtxSettingsGlobalRtAdvancedAPI_1", "OmniRtxSettingsGlobalPtAdvancedAPI_1"]
        no_delete = true
    )
    {
        rel products = </Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0>
    }

    def "Vars"
    {
        def RenderVar "LdrColor" (
            hide_in_stage_window = true
            no_delete = true
        )
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
'''

GRAPHS_BLOCK = '''    def Scope "Graphs"
    {
        def OmniGraph "Position_Controller"
        {
            token evaluationMode = "Automatic"
            token evaluator:type = "execution"
            token fabricCacheBacking = "StageWithoutHistory"
            int2 fileFormatVersion = (1, 9)
            token pipelineStage = "pipelineStageSimulation"

            def OmniGraphNode "OnPlaybackTick"
            {
                token node:type = "omni.graph.action.OnPlaybackTick"
                int node:typeVersion = 2
                custom double outputs:deltaSeconds
                custom double outputs:frame
                custom uint outputs:tick (
                    customData = {
                        bool isExecution = 1
                    }
                )
                custom double outputs:time
            }

            def OmniGraphNode "JointCommandArray"
            {
                custom int inputs:arraySize = 18
                custom token inputs:arrayType = "double[]" (
                    allowedTokens = ["auto", "bool[]", "double[]", "float[]", "half[]", "int[]", "int64[]", "token[]", "uchar[]", "uint[]", "uint64[]", "double[2][]", "double[3][]", "double[4][]", "matrixd[2][]", "matrixd[3][]", "matrixd[4][]", "float[2][]", "float[3][]", "float[4][]", "half[2][]", "half[3][]", "half[4][]", "int[2][]", "int[3][]", "int[4][]", "timecode[]", "frame[4][]", "colord[3][]", "colorf[3][]", "colorh[3][]", "colord[4][]", "colorf[4][]", "colorh[4][]", "normald[3][]", "normalf[3][]", "normalh[3][]", "pointd[3][]", "pointf[3][]", "pointh[3][]", "quatd[4][]", "quatf[4][]", "quath[4][]", "texcoordd[2][]", "texcoordf[2][]", "texcoordh[2][]", "texcoordd[3][]", "texcoordf[3][]", "texcoordh[3][]", "vectord[3][]", "vectorf[3][]", "vectorh[3][]"]
                )
                custom token inputs:input0 (
                    customData = {
                        dictionary omni = {
                            dictionary graph = {
                                double attrValue = 0
                                string resolvedType = "double"
                            }
                        }
                    }
                )
                custom double inputs:input1 = 0
                custom double inputs:input2 = 0
                custom double inputs:input3 = 0
                custom double inputs:input4 = 0
                custom double inputs:input5 = 0
                custom double inputs:input6 = 0
                custom double inputs:input7 = 0
                custom double inputs:input8 = 0
                custom double inputs:input9 = 0
                custom double inputs:input10 = 0
                custom double inputs:input11 = 0
                custom double inputs:input12 = 0
                custom double inputs:input13 = 0
                custom double inputs:input14 = 0
                custom double inputs:input15 = 0
                custom double inputs:input16 = 0
                custom double inputs:input17 = 0
                token node:type = "omni.graph.nodes.ConstructArray"
                int node:typeVersion = 1
                custom token outputs:array
            }

            def OmniGraphNode "ArticulationController"
            {
                custom double[] inputs:effortCommand
                custom uint inputs:execIn (
                    customData = {
                        bool isExecution = 1
                    }
                )
                prepend uint inputs:execIn.connect = <../OnPlaybackTick.outputs:tick>
                custom int[] inputs:jointIndices
                custom token[] inputs:jointNames
                prepend token[] inputs:jointNames.connect = <../JointNameArray.outputs:array>
                custom double[] inputs:positionCommand
                prepend double[] inputs:positionCommand.connect = <../JointCommandArray.outputs:array>
                custom string inputs:robotPath = ""
                custom rel inputs:targetPrim = <../../..> (
                    customData = {
                        dictionary omni = {
                            dictionary graph = {
                                string relType = "target"
                            }
                        }
                    }
                )
                custom double[] inputs:velocityCommand
                token node:type = "isaacsim.core.nodes.IsaacArticulationController"
                int node:typeVersion = 1
            }

            def OmniGraphNode "JointNameArray"
            {
                custom int inputs:arraySize = 18
                custom token inputs:arrayType = "token[]" (
                    allowedTokens = ["auto", "bool[]", "double[]", "float[]", "half[]", "int[]", "int64[]", "token[]", "uchar[]", "uint[]", "uint64[]", "double[2][]", "double[3][]", "double[4][]", "matrixd[2][]", "matrixd[3][]", "matrixd[4][]", "float[2][]", "float[3][]", "float[4][]", "half[2][]", "half[3][]", "half[4][]", "int[2][]", "int[3][]", "int[4][]", "timecode[]", "frame[4][]", "colord[3][]", "colorf[3][]", "colorh[3][]", "colord[4][]", "colorf[4][]", "colorh[4][]", "normald[3][]", "normalf[3][]", "normalh[3][]", "pointd[3][]", "pointf[3][]", "pointh[3][]", "quatd[4][]", "quatf[4][]", "quath[4][]", "texcoordd[2][]", "texcoordf[2][]", "texcoordh[2][]", "texcoordd[3][]", "texcoordf[3][]", "texcoordh[3][]", "vectord[3][]", "vectorf[3][]", "vectorh[3][]"]
                )
                custom token inputs:input0 (
                    customData = {
                        dictionary omni = {
                            dictionary graph = {
                                token attrValue = "FL_Body_Hip_RevoluteJoint"
                                string resolvedType = "token"
                            }
                        }
                    }
                )
                custom token inputs:input1 = "FL_Hip_Femur_RevoluteJoint"
                custom token inputs:input2 = "FL_Femur_Tibia_RevoluteJoint"
                custom token inputs:input3 = "FR_Body_Hip_RevoluteJoint"
                custom token inputs:input4 = "FR_Hip_Femur_RevoluteJoint"
                custom token inputs:input5 = "FR_Femur_Tibia_RevoluteJoint"
                custom token inputs:input6 = "ML_Body_Hip_RevoluteJoint"
                custom token inputs:input7 = "ML_Hip_Femur_RevoluteJoint"
                custom token inputs:input8 = "ML_Femur_Tibia_RevoluteJoint"
                custom token inputs:input9 = "MR_Body_Hip_RevoluteJoint"
                custom token inputs:input10 = "MR_Hip_Femur_RevoluteJoint"
                custom token inputs:input11 = "MR_Femur_Tibia_RevoluteJoint"
                custom token inputs:input12 = "RL_Body_Hip_RevoluteJoint"
                custom token inputs:input13 = "RL_Hip_Femur_RevoluteJoint"
                custom token inputs:input14 = "RL_Femur_Tibia_RevoluteJoint"
                custom token inputs:input15 = "RR_Body_Hip_RevoluteJoint"
                custom token inputs:input16 = "RR_Hip_Femur_RevoluteJoint"
                custom token inputs:input17 = "RR_Femur_Tibia_RevoluteJoint"
                token node:type = "omni.graph.nodes.ConstructArray"
                int node:typeVersion = 1
                custom token outputs:array
            }
        }
    }
'''


def materials_block() -> str:
    return f'''    def Material "PlyWood" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = {n(dims.PLYWOOD_DYNAMIC_FRICTION)}
        float physics:restitution = {n(dims.PLYWOOD_RESTITUTION)}
        float physics:staticFriction = {n(dims.PLYWOOD_STATIC_FRICTION)}
    }}

    def Material "FootRubber" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = 0.75
        float physics:restitution = 0.15
        float physics:staticFriction = 0.9
    }}
'''


def chassis_block() -> str:
    bx, by, bz = dims.BODY_SIZE_M
    return f'''    def Xform "chassis"
    {{
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        def Cube "body" (
            prepend apiSchemas = ["PhysicsMassAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float physics:mass = {n(dims.BODY_MASS_KG)}
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            rel material:binding = </krabby/PlyWood> (
                bindMaterialAs = "weakerThanDescendants"
            )
            bool physics:collisionEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:rigidBodyEnabled = 1
            double size = 1
            quatd xformOp:orient = (1, 0, 0, 0)
            double3 xformOp:scale = {v3(bx, by, bz)}
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}
    }}
'''


def generate(variant: MorphVariant = DEFAULT_VARIANT) -> str:
    legs = {name: Leg(name, variant) for name in LEG_PRIM_ORDER}
    parts = [HEADER]
    parts.append(f'''
def Xform "krabby" (
    delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{{
    rel proxyPrim = None
    quatd xformOp:orient = (1, 0, 0, 0)
    double3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, {n(ROOT_LIFT_Z)})
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

''')
    parts.append("\n".join(leg_block(legs[name]) for name in LEG_PRIM_ORDER))
    parts.append("\n" + materials_block())
    parts.append('\n    def Scope "Joints"\n    {\n')
    parts.append("\n".join(joint_blocks(legs[name]) for name in JOINT_LEG_ORDER))
    parts.append("\n")
    parts.append("\n".join(fixed_joint_block(legs[name]) for name in JOINT_LEG_ORDER))
    parts.append("    }\n")
    parts.append("\n" + chassis_block())
    parts.append("\n" + GRAPHS_BLOCK)
    parts.append("}\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=MAIN_ASSET,
        help="output path (default: assets/crab.usda, the main asset)",
    )
    parser.add_argument(
        "--splay-deg",
        type=float,
        default=DEFAULT_VARIANT.row_splay_deg,
        help=f"outward yaw splay of the front/rear leg mounts, 0-20 deg (default: plant of record, {DEFAULT_VARIANT.row_splay_deg:g})",
    )
    parser.add_argument(
        "--outer-axis-in",
        type=float,
        default=DEFAULT_VARIANT.outer_axis_from_end_in,
        help=f"outer yaw-axis distance from the body ends in inches (default: plant of record, {DEFAULT_VARIANT.outer_axis_from_end_in:g}; 2026-08-20 build 5.5)",
    )
    parser.add_argument("--legacy-golden", action="store_true",
                        help="write the legacy golden plant assets/variants/crab_simple__splay00_axis5p5in.usda (splay 0, axes 5.5 in)")
    parser.add_argument("--all-variants", action="store_true",
                        help="write every named plant under assets/variants/ and regenerate MANIFEST.md")
    args = parser.parse_args()

    if args.all_variants:
        VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, path in variant_files().items():
            path.write_text(generate(VARIANTS[name]))
            print(f"wrote {path.relative_to(REPO_ROOT)} ({name}, {VARIANTS[name].tag})")
        (VARIANTS_DIR / "MANIFEST.md").write_text(manifest_text())
        print(f"wrote {VARIANTS_DIR.relative_to(REPO_ROOT)}/MANIFEST.md")
        return

    out = args.out.resolve()  # compare resolved paths: a relative --out must not slip past the pins
    if args.legacy_golden:
        variant = LEGACY_GOLDEN_VARIANT
        out = variant_asset_path("legacy_golden") if out == MAIN_ASSET else out
    else:
        variant = MorphVariant(row_splay_deg=args.splay_deg, outer_axis_from_end_in=args.outer_axis_in)
    if variant != DEFAULT_VARIANT and out == MAIN_ASSET:
        parser.error("assets/crab.usda is byte-pinned to the plant of record; write a variant with --out")
    if out == HAND_AUTHORED_ASSET:
        parser.error("assets/crab_simple.usda is the hand-authored historical Cube model; this generator never writes it")
    text = generate(variant)
    out.write_text(text)
    total = dims.TOTAL_MASS_KG
    print(
        f"wrote {out} ({len(text.splitlines())} lines); total robot mass {total:.2f} kg; "
        f"variant {variant.tag}"
    )


if __name__ == "__main__":
    main()
