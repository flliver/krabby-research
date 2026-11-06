"""DR stress sweep: the cell table (CHARTER §3.2) and the row composition (CHARTER §3.1). Pure, no Isaac.

One source of truth for the driver (``run_sweep.py``) and the reader (``stress_read.py``): every term, its levels,
its plausible band, the bases it runs on, the per-cell env / level files and the gate rows of a cell at a seed.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path(__file__).resolve().parents[7]
EXPERIMENTS = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments"
T4 = EXPERIMENTS / "2026-10-16_task4_pipeline"
GATE_ROW = EXPERIMENTS / "tools/pipeline/gate_row.sh"
MAIN = Path("/home/nickmagus/krabby/krabby-research")
#: ``DRS_TP_OUT`` overrides the output root (dry runs and the end-to-end smoke only).
TP_OUT = Path(os.environ.get("DRS_TP_OUT") or MAIN / "parkour/outputs/omni_plan/dr_stress")
CKPT = MAIN / "parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt"
TSTAR = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/f19_8_p3b_payload180/model_25992.pt"
ZERO_DEPTH_DIR = MAIN / "parkour/outputs/omni_plan/dr_stress/ckpt_zero_depth"
TASK = "Isaac-Crab-Hex-Omni-Student-Play-v0"
ROWS = {
    "suite": {"env": T4 / "F19/envs/p3b.env", "manifest": EXPERIMENTS / "eval/scenarios_omni.yaml",
              "scenario": "suite__omni",
              "args": ["--task", TASK, "--trained-categories", "stand,fwd,side,back,diag,turn,move_turn"]},
    "field_light": {"env": T4 / "F19/envs/p3b.field_light.env",
                    "manifest": EXPERIMENTS / "2026-10-07_omni_phase2/scenarios_p2.yaml",
                    "scenario": "field_light__omni", "args": ["--task", TASK]},
}
BASE_LEVEL = {"L000": None, "L090": T4 / "F19/envs/levels/L090.env"}
PAYLOAD_COM_Z = 0.309

#: The terms (CHARTER §3.2). ``var`` None = not a stress knob (``kind`` says how the cell is built). ``band`` None =
#: report-only / read-only (no verdict). ``nominal`` orders the levels into a low and a high side.
TERMS = [
    # Band 0.7-1.35 (CHARTER §6, 2026-11-06 amendment): the sim constant is the LOADED speed, so a lightly loaded rod
    # can run faster (~x1.37 by the 26 -> 19 mm/s droop of the review's reference actuator [7]); 1.35 added.
    {"term": "T01", "slug": "rod_speed", "var": "KRABBY_STRESS_ROD_SPEED", "kind": "stress", "nominal": 1.0,
     "levels": [0.5, 0.7, 0.8, 1.15, 1.35], "band": (0.7, 1.35), "bases": ("L000", "L090")},
    {"term": "T02", "slug": "rod_effort", "var": "KRABBY_STRESS_ROD_EFFORT", "kind": "stress", "nominal": 1.0,
     "levels": [0.5, 0.7, 0.85], "band": (0.85, 1.0), "bases": ("L000", "L090")},
    {"term": "T03", "slug": "cam", "var": "KRABBY_STRESS_CAM", "kind": "stress", "nominal": 1.0,
     "levels": [0.5, 0.8, 1.2], "band": (0.8, 1.2), "bases": ("L000",)},
    {"term": "T04", "slug": "kpkd", "var": "KRABBY_STRESS_KPKD", "kind": "stress", "nominal": 1.0,
     "levels": [0.5, 0.75, 1.25], "band": (0.75, 1.25), "bases": ("L000",)},
    {"term": "T05", "slug": "leg_mass", "var": "KRABBY_STRESS_LEG_MASS", "kind": "stress", "nominal": 1.0,
     "levels": [0.9, 1.15, 1.5], "band": (0.9, 1.15), "bases": ("L000",)},
    {"term": "T06", "slug": "mu", "var": "KRABBY_STRESS_MU", "kind": "stress", "nominal": 1.0,
     "levels": [0.3, 0.4, 0.8, 1.2], "band": (0.4, 1.2), "bases": ("L000",)},
    {"term": "T07", "slug": "act_delay", "var": "KRABBY_STRESS_ACT_DELAY", "kind": "stress", "nominal": 1,
     "levels": [2, 3, 4], "band": (1, 2), "bases": ("L000",)},
    {"term": "T08", "slug": "obs_noise", "var": "KRABBY_STRESS_OBS_NOISE", "kind": "stress", "nominal": 0.0,
     "levels": [0.5, 1.0, 2.0, 4.0], "band": (0.0, 1.0), "bases": ("L000",)},
    {"term": "T09a", "slug": "cur_scale", "var": "KRABBY_STRESS_CUR_SCALE", "kind": "stress", "nominal": 1.0,
     "levels": [0.85, 1.15], "band": (0.85, 1.15), "bases": ("L000",)},
    {"term": "T09b", "slug": "cur_noise", "var": "KRABBY_STRESS_CUR_NOISE", "kind": "stress", "nominal": 0.0,
     "levels": [0.1, 0.25], "band": (0.0, 0.1), "bases": ("L000",)},
    {"term": "T10a", "slug": "vxy_noise", "var": "KRABBY_STRESS_VXY_NOISE", "kind": "stress", "nominal": 0.0,
     "levels": [0.05, 0.1, 0.2], "band": (0.0, 0.1), "bases": ("L000",)},
    {"term": "T10b", "slug": "vxy_zero", "var": "KRABBY_STRESS_VXY_ZERO", "kind": "stress", "nominal": 0,
     "levels": [1], "band": None, "bases": ("L000",)},
    {"term": "T11", "slug": "depth_lag", "var": "KRABBY_STRESS_DEPTH_LAG", "kind": "stress", "nominal": 1,
     "levels": [2, 3], "band": (1, 2), "bases": ("L000",)},
    {"term": "T12", "slug": "zero_depth", "var": None, "kind": "zero_depth", "nominal": 0, "levels": [1],
     "band": None, "bases": ("L000",)},
    {"term": "T13", "slug": "payload_kg", "var": None, "kind": "payload_kg", "nominal": 135.0,
     "levels": [150.0, 165.0], "band": None, "bases": ("payload",)},
    {"term": "T14", "slug": "payload_com", "var": None, "kind": "payload_com", "nominal": 0.0,
     "levels": [0.2, 0.3], "band": None, "bases": ("L090",)},
    {"term": "T15", "slug": "push_dv", "var": "KRABBY_STRESS_PUSH_DV", "kind": "stress", "nominal": 0.0,
     "levels": [0.1, 0.2, 0.35, 0.5], "band": None, "bases": ("L000",)},
]
TERM = {t["term"]: t for t in TERMS}


def fmt(v) -> str:
    """``0.7 -> '0p7'``, ``2 -> '2'``, ``150.0 -> '150'`` (gate_row NAMEs are ``[A-Za-z0-9_]+``)."""
    s = f"{float(v):g}"
    return s.replace(".", "p").replace("-", "m")


def cell_id(term: dict, level, base: str) -> str:
    return f"{term['term']}_{term['slug']}_{fmt(level)}_{base}"


def nominal_id(base: str, launch: str) -> str:
    return f"T00_nominal_{base}_{launch}"


def cells() -> list[dict]:
    """Every grid cell: ``{id, term, level, base}`` in table order."""
    out = []
    for t in TERMS:
        for base in t["bases"]:
            for lv in t["levels"]:
                out.append({"id": cell_id(t, lv, base), "term": t["term"], "level": lv, "base": base})
    return out


def band_edges(term: dict) -> list:
    """The levels equal to a band end (the verdict-deciding levels), excluding the nominal."""
    if term["band"] is None:
        return []
    lo, hi = term["band"]
    return [lv for lv in term["levels"] if (abs(lv - lo) < 1e-12 or abs(lv - hi) < 1e-12)
            and abs(lv - term["nominal"]) > 1e-12]


def inside_band(term: dict, level) -> bool:
    if term["band"] is None:
        return False
    lo, hi = term["band"]
    return lo - 1e-12 <= level <= hi + 1e-12


def side(term: dict, level) -> str:
    return "low" if level < term["nominal"] else "high"


def stress_value(term: dict, level) -> str:
    if term["var"] == "KRABBY_STRESS_VXY_ZERO":
        return "1"
    if term["var"] in ("KRABBY_STRESS_ACT_DELAY", "KRABBY_STRESS_DEPTH_LAG"):
        return str(int(level))
    return f"{float(level):g}"


def level_file_for(term: dict | None, level, base: str) -> Path | None:
    """The TP_LEVEL_FILE of a cell (None = unarmed base). Terms 13 / 14 write their own (``levels/``)."""
    if term is not None and term["kind"] == "payload_kg":
        return CAMPAIGN / "levels" / f"L{int(level):03d}.env"
    if term is not None and term["kind"] == "payload_com":
        return CAMPAIGN / "levels" / f"L090_comxy{fmt(level)}.env"
    return BASE_LEVEL[base]


def level_text(term: dict, level) -> str:
    if term["kind"] == "payload_kg":
        kg, xy = float(level), 0.0
    else:
        kg, xy = 90.0, float(level)
    return (f"KRABBY_PAYLOAD_COM_XY={xy:g}\nKRABBY_PAYLOAD_COM_Z={PAYLOAD_COM_Z:g}\n"
            f"KRABBY_PAYLOAD_KG={kg:g}:{kg:g}\nKRABBY_PAYLOAD_ZERO_FRAC=0\n")


def env_file(cid: str, row: str) -> Path:
    return CAMPAIGN / "envs" / f"{cid}__{row}.env"


def env_text(term: dict | None, level, row: str) -> str:
    base = ROWS[row]["env"].read_text()
    if not base.endswith("\n"):
        base += "\n"
    if term is not None and term["kind"] == "stress":
        base += f"{term['var']}={stress_value(term, level)}\n"
    return base


def expected_env(term: dict | None, level, base: str) -> dict:
    """The ``KRABBY_*`` keys a cell's stamps MUST carry in ``run_meta.env_vars`` (the launch-error check)."""
    want = {}
    if term is not None and term["kind"] == "stress":
        want[term["var"]] = stress_value(term, level)
    lf = level_file_for(term, level, base)
    if lf is not None:
        text = lf.read_text() if lf.is_file() else level_text(term, level)
        for line in text.splitlines():
            if line.startswith("KRABBY_PAYLOAD_"):
                k, v = line.split("=", 1)
                want[k] = v
    return want


def checkpoint_for(term: dict | None, policy: str = "student") -> Path:
    if policy == "teacher":
        return TSTAR
    if term is not None and term["kind"] == "zero_depth":
        return ZERO_DEPTH_DIR / CKPT.name
    return CKPT


def gate_name(cid: str, policy: str = "student") -> str:
    return f"DRS_{cid}" if policy == "student" else f"DRST_{cid}"


def rows_for(cid: str, term: dict | None, level, base: str, es: int, policy: str = "student") -> list[dict]:
    """The two gate rows (suite, field_light) of a cell at env seed ``es``."""
    out = []
    for row, spec in ROWS.items():
        out.append({
            "cell": cid, "row": row, "es": int(es), "policy": policy,
            "name": gate_name(cid, policy), "checkpoint": str(checkpoint_for(term, policy)),
            "env_file": str(env_file(cid, row)), "manifest": str(spec["manifest"]), "scenario": spec["scenario"],
            "level_file": (None if level_file_for(term, level, base) is None
                           else str(level_file_for(term, level, base))),
            "args": [*spec["args"], "--policy-role", policy],
        })
    return out


def stamp_dir(name: str, scenario: str, es: int) -> Path | None:
    """The newest complete stamp of a row (``tp_row.json`` is written last), or None."""
    base = TP_OUT / "lr" / name / scenario / f"seed{es:03d}"
    if not base.is_dir():
        return None
    done = sorted(p for p in base.iterdir() if (p / "tp_row.json").is_file())
    return done[-1] if done else None
