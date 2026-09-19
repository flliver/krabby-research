"""Pin the crab-hexapod plants to their generator and the measured-dimensions module.

Guarantees, all pure-text / stdlib (no Isaac Sim):

1. The MAIN asset ``assets/crab.usda`` (plant of record A15+B) is EXACTLY what
   ``assets/scripts/generate_crab.py`` emits from ``crab_hex_dimensions.py``; the legacy golden
   ``assets/variants/crab_simple__splay00_axis5p5in.usda`` (2026-08-20 build) is exactly
   ``generate(LEGACY_GOLDEN_VARIANT)``; every ``assets/variants/*.usda`` is its named variant.
   Hand-edits to a generated USDA (the pre-2026-08-20 workflow, which accumulated
   anchor-vs-translate drift) fail here: edit the dimensions module and regenerate instead.
   ``assets/crab_simple.usda`` is the exception by design: the hand-authored Cube model of the
   2026-08-09 campaign baseline (reverted 2026-09-09, git 5ca0a8c), sha-pinned, never generated.
2. The masses in the asset add up to the measured hardware totals: 6 x 26.2 lb legs +
   350 lb body = ~230.06 kg, with the per-link split summing exactly per leg.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
USDA_PATH = REPO_ROOT / "assets" / "crab.usda"                 # main asset (A15+B)
VARIANTS_DIR = REPO_ROOT / "assets" / "variants"
LEGACY_PATH = VARIANTS_DIR / "crab_simple__splay00_axis5p5in.usda"   # legacy golden (2026-08-20 build)
HAND_AUTHORED_PATH = REPO_ROOT / "assets" / "crab_simple.usda"       # 2026-08-09 campaign-baseline Cube model
HAND_AUTHORED_SHA256 = "0a7417235af037a6a0141ff6089ced19cb52f8a018b5696278a836c2cbeb8c2a"
GENERATOR_PATH = REPO_ROOT / "assets" / "scripts" / "generate_crab.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dims = _load(
    "crab_hex_dimensions",
    REPO_ROOT
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "mdp"
    / "crab_hex_dimensions.py",
)
generator = _load("generate_crab", GENERATOR_PATH)


def test_committed_asset_matches_generator():
    generated = generator.generate()
    committed = USDA_PATH.read_text()
    assert committed == generated, (
        "assets/crab.usda differs from its generator output. Never hand-edit the "
        "asset: change crab_hex_dimensions.py (or the generator) and run "
        "python3 assets/scripts/generate_crab.py"
    )


def test_legacy_golden_variant_matches_generator():
    """The 2026-08-20 golden stays byte-pinned to the generator's LEGACY_GOLDEN_VARIANT."""
    assert LEGACY_PATH.read_text() == generator.generate(generator.LEGACY_GOLDEN_VARIANT), (
        "assets/variants/crab_simple__splay00_axis5p5in.usda (legacy golden) drifted; regenerate with "
        "python3 assets/scripts/generate_crab.py --legacy-golden"
    )
    assert generator.LEGACY_GOLDEN_VARIANT.tag == "splay00_axis5p5in"
    assert generator.variant_asset_path("legacy_golden") == LEGACY_PATH


def test_hand_authored_crab_simple_is_the_campaign_baseline_cube_model():
    """assets/crab_simple.usda was reverted (2026-09-09) to the hand-authored Cube model of the
    2026-08-09 campaign baseline (git 5ca0a8c): 31 Cube prims, cam-shaft joints, no generated Mesh
    links. A historical reference, never generated -- pinned by sha256 so it cannot drift silently."""
    import hashlib

    text = HAND_AUTHORED_PATH.read_text()
    assert hashlib.sha256(HAND_AUTHORED_PATH.read_bytes()).hexdigest() == HAND_AUTHORED_SHA256
    assert text.count("def Cube") == 31 and "CamShaft" in text
    assert text.count("def Mesh") == 1  # only the ground CollisionMesh; no generated plywood-outline link meshes
    assert "physics:centerOfMass" not in text  # the generated (measured-hardware) assets author CoM on every link


def test_main_asset_is_the_a15b_plant():
    assert generator.DEFAULT_VARIANT.tag == "splay15_axis2p5in"
    assert generator.DEFAULT_VARIANT.row_splay_deg == dims.OUTER_ROW_SPLAY_DEG == 15.0
    assert generator.DEFAULT_VARIANT.outer_axis_from_end_in == dims.OUTER_LEG_AXIS_FROM_BODY_END_IN == 2.5
    a15b = VARIANTS_DIR / "crab_simple__splay15_axis2p5in.usda"
    assert USDA_PATH.read_bytes() == a15b.read_bytes(), "the A15+B variant file must stay byte-identical to crab.usda"


def _masses_by_prim(text: str) -> dict[str, float]:
    """{prim_name: physics:mass} for every Cube/Mesh prim in the asset."""
    masses: dict[str, float] = {}
    # Tempered dot: never scan across the next prim definition (the massless ground
    # CollisionMesh would otherwise swallow the first leg's mass line).
    pattern = re.compile(
        r'def (?:Cube|Mesh) "(?P<name>\w+)"(?:(?!def ).)*?physics:mass = (?P<mass>[\d.]+)',
        re.DOTALL,
    )
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            break
        masses[m.group("name")] = float(m.group("mass"))
        pos = m.start() + 1
    return masses


@pytest.fixture(scope="module")
def prim_masses() -> dict[str, float]:
    return _masses_by_prim(USDA_PATH.read_text())


def test_per_leg_mass_sums_to_measured_leg_weight(prim_masses):
    expected_leg_kg = dims.LEG_MASS_LB * dims.LB_TO_KG
    for leg in ("FL", "FR", "ML", "MR", "RL", "RR"):
        total = sum(
            prim_masses[f"{leg}_{link}"]
            for link in ("Hip", "Femur", "Tibia", "Footpad", "CamShaft")
        )
        assert total == pytest.approx(expected_leg_kg, abs=1e-4), f"{leg} mass sum"


def test_total_mass_matches_hardware(prim_masses):
    total = sum(prim_masses.values())
    expected = dims.BODY_MASS_LB * dims.LB_TO_KG + 6.0 * dims.LEG_MASS_LB * dims.LB_TO_KG
    assert total == pytest.approx(expected, abs=1e-3)
    assert total == pytest.approx(230.06, abs=0.05)


def test_body_keeps_full_measured_mass(prim_masses):
    """Cam rotors are budgeted inside the legs' 26.2 lb, never subtracted from the body."""
    assert prim_masses["body"] == pytest.approx(
        dims.BODY_MASS_LB * dims.LB_TO_KG, abs=1e-4
    )


def test_center_of_mass_authored_on_all_leg_links():
    """Mesh-era links author CoM + diagonal inertia explicitly (hip/femur/tibia x 6)."""
    text = USDA_PATH.read_text()
    com_lines = re.findall(r"physics:centerOfMass = \((.*?)\)", text)
    assert len(com_lines) == 18, "expected authored centerOfMass on all 18 leg-link meshes"
    diag_lines = re.findall(r"float3 physics:diagonalInertia", text)
    assert len(diag_lines) == 24, "18 leg meshes + 6 camshaft rotors author diagonalInertia"


def test_mesh_ply_mass_consistent_with_outline_area():
    """Tripwire for outline/inertia regressions: the polygon-derived plywood mass of each
    part (area x 1 in thickness x ply density) must land within a loose band of the
    dimensions module's volume estimate that the link masses were normalized from."""
    import importlib.util as _ilu

    gspec = _ilu.spec_from_file_location("gen", GENERATOR_PATH)
    gen = _ilu.module_from_spec(gspec)
    gspec.loader.exec_module(gen)
    IN = dims.IN_TO_M
    t = dims.PLY_THICKNESS_IN * IN
    for name, est_in3 in (
        ("HIP_OUTLINE_IN", dims.HIP_PLATE_LENGTH_IN * dims.HIP_PLATE_WIDTH_IN),
        ("FEMUR_OUTLINE_IN", dims.FEMUR_PART_LENGTH_IN * dims.FEMUR_WIDTH_IN),
        (
            "TIBIA_OUTLINE_IN",
            dims.TIBIA_PART_LENGTH_IN * dims.TIBIA_WIDTH_IN * dims.TIBIA_TAPER_VOLUME_FACTOR,
        ),
    ):
        outline = [(a * IN, b * IN) for a, b in getattr(gen.profiles, name)]
        area = abs(gen._signed_area(gen.ensure_ccw(outline)))
        poly_kg = area * t * dims.PLY_DENSITY_KG_M3
        est_kg = est_in3 * dims.PLY_THICKNESS_IN * (IN**3) * dims.PLY_DENSITY_KG_M3
        assert 0.6 * est_kg < poly_kg < 1.15 * est_kg, (
            f"{name}: polygon ply mass {poly_kg:.3f} kg vs volume estimate {est_kg:.3f} kg"
        )


# ---------------------------------------------------------------------------
# PLAN G leg-mount morphology variants (2026-09-02)
# ---------------------------------------------------------------------------
def test_default_variant_is_byte_identical():
    assert generator.generate(generator.MorphVariant()) == generator.generate()


def test_committed_variants_match_their_regeneration():
    vdir = VARIANTS_DIR
    files = sorted(vdir.glob("crab_simple__splay*_axis*.usda"))
    assert files, "no committed variants found"
    expected = {p.name for p in generator.variant_files().values()}
    assert {f.name for f in files} == expected, "assets/variants/ must hold exactly the named plants (generate_crab.py --all-variants)"
    for f in files:
        tag = f.stem.split("__", 1)[1]
        splay = float(tag.split("_")[0].replace("splay", ""))
        axis = float(tag.split("_axis")[1].replace("in", "").replace("p", "."))
        v = generator.MorphVariant(row_splay_deg=splay, outer_axis_from_end_in=axis)
        assert f.read_text() == generator.generate(v), f"{f.name} drifted from its generator"


def test_splay_variant_touches_only_outer_leg_mount_lines():
    import difflib

    # variant-relative baseline: the unsplayed geometry at the default outer-axis position
    ax = generator.DEFAULT_VARIANT.outer_axis_from_end_in
    golden = generator.generate(generator.MorphVariant(0.0, ax)).splitlines()
    splayed = generator.generate(generator.MorphVariant(20.0, ax)).splitlines()
    added = [l for l in difflib.unified_diff(golden, splayed, lineterm="", n=0)
             if l.startswith("+") and not l.startswith("+++")]
    kinds = {"orient": 0, "translate": 0, "localRot0": 0, "scale": 0}
    for l in added:
        for k in kinds:
            if k in l:
                kinds[k] += 1
                break
        else:
            raise AssertionError(f"unexpected changed line in splay variant: {l.strip()}")
    # 4 outer legs x 5 prims orient; x 4 translated prims (the cam rotor sits ON the axis);
    # x 2 Z-axis joints localRot0; 'scale' lines are diff re-anchoring of unchanged text.
    assert kinds["orient"] == 20 and kinds["translate"] == 16 and kinds["localRot0"] == 8


def test_variant_masses_unchanged():
    text = generator.generate(generator.MorphVariant(row_splay_deg=20.0, outer_axis_from_end_in=2.5))
    masses = _masses_by_prim(text)
    total = sum(masses.values())
    assert total == pytest.approx(230.06, abs=0.05)


def test_manifest_lists_every_plant():
    text = generator.manifest_text()
    for name in generator.VARIANTS:
        assert f"`{name}`" in text
    assert "assets/crab.usda" in text and "assets/crab_simple.usda" in text


def test_variant_cap_and_range():
    with pytest.raises(ValueError):
        generator.MorphVariant(row_splay_deg=25.0)
    with pytest.raises(ValueError):
        generator.MorphVariant(outer_axis_from_end_in=0.0)
