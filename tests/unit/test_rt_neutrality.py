# SPDX-License-Identifier: BSD-3-Clause
"""Knobs off -> the omni knob block is main's (rough-terrain PLAN step 2: "knobs off -> config dump identical to main").

Re-uses step 1's neutrality probe: ``git archive`` of ``origin/main`` before the campaign (:data:`MAIN_SHA`, 3f944c4b)
into a scratch dir, then ONE probe program run in a subprocess against each tree (so the two trees' modules never share
a process). For every recorded omni env file (the TDR / DR-sweep / Task 4 campaigns' ``envs/**/*.env`` in this
checkout and, when present, the rod-speed DR worktree's ``experiments/**/*.env``) plus synthetic ring / field / unset /
bogus envs, the probe runs the omni terrain knob block (``install_omni_terrain_generator`` + ``apply_omni_ring_knobs``,
and on this tree also the new ``apply_rt_friction_knobs``) on a stand-in env cfg and records the cfg snapshot, stdout
and exception. It also records ``run_candidates``' knob whitelist / parser keys / dump-check keys, the dump-check
verdicts of TDR's recorded training dump, and ``omni_knobs``' public names.

Pinned: every env's snapshot, stdout and exception type equal main's; the driver tables gain exactly step 1's three
site knobs (``KRABBY_SITE_MIX`` / ``_DIFF`` / ``_LADDER``) and this step's ``KRABBY_RT_FRICTION``; TDR's dump verdicts
are unchanged; ``omni_knobs`` gains exactly step 1's and this step's re-exports. Skipped when git or the main commit is
unavailable (a shallow CI clone).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MAIN_SHA = "3f944c4b"
ARCHIVE_PATHS = (
    "parkour/parkour_isaaclab/terrains",
    "parkour/parkour_isaaclab/envs/mdp/observations.py",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/config",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp",
)
RECORDED_ENV_DIRS = ("2026-11-07_rod_speed_dr", "2026-11-06_dr_stress_sweep", "2026-10-16_task4_pipeline")
DRS_EXPERIMENTS = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/dr-stress-sweep/parkour/parkour_tasks/"
                       "parkour_tasks/crab_hex_omni_task/experiments")
TDR_DUMP = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/"
                "2026-11-07_19-49-46_RSD_TDR_s19/params/env.yaml")
TDR_ENV = "2026-11-07_rod_speed_dr/envs/RSD_TDR_s19.train.env"
#: What step 1 (6588f1c3, the site port) already added on top of main: its three site knobs and their re-exports.
STEP1_KNOBS = {"KRABBY_SITE_MIX", "KRABBY_SITE_DIFF", "KRABBY_SITE_LADDER"}
STEP1_NAMES = {"SITE_GEOM", "SITE_MIX_VAR", "SITE_DIFF_VAR", "SITE_LADDER_VAR", "SITE_KNOB_PARSERS"}

PROBE = r'''
import contextlib, copy, glob, importlib.util, io, json, os, sys, types
from pathlib import Path

root = Path(sys.argv[1]).resolve()
env_files = json.loads(Path(sys.argv[2]).read_text())
tdr_dump, tdr_env, out_path = sys.argv[3], sys.argv[4], sys.argv[5]
O = root / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
sys.path[:0] = [str(root / "parkour/parkour_tasks"), str(root / "parkour")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tk = load("crab_hex_omni_terrain_knobs", O / "mdp/omni_terrain_knobs.py")


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class RingCfg(_Obj):
    pass


class FieldCfg(_Obj):
    pass


class FlatCfg(_Obj):
    pass


def ring_gen():
    return _Obj(sub_terrains={"parkour_flat": FlatCfg(clear_radius=2.0, ring_outer_radius=4.5),
                              "omni_ring_sparse": RingCfg(n_blocks=8, clear_radius=2.0, ring_outer_radius=4.5),
                              "omni_ring_dense": RingCfg(n_blocks=16, clear_radius=2.0, ring_outer_radius=4.5)},
                difficulty_range=(0.2, 1.0), num_cols=20)


def field(n):
    return FieldCfg(n_blocks=n, clear_radius=1.7, ring_outer_radius=4.80, bar_fraction=0.4, pit_fraction=0.2,
                    vertical_edges=False, min_gap=0.0, bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32),
                    slab_side_range=(0.6, 1.2), noise_range=(0.02, 0.02))


def field_gen():
    return _Obj(sub_terrains={"parkour_flat": FlatCfg(clear_radius=1.7, ring_outer_radius=4.8),
                              "omni_field_sparse": field(14), "omni_field_dense": field(28)},
                difficulty_range=(0.2, 1.0), num_cols=20)


def material():
    return _Obj(static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="average",
                restitution_combine_mode="average")


def cfg_of():
    return _Obj(scene=_Obj(terrain=_Obj(terrain_generator=ring_gen(), physics_material=material())),
                sim=_Obj(physics_material=material()),
                parkours=_Obj(base_parkour=_Obj(clear_radius_m=2.0, ring_outer_radius_m=4.5)),
                events=_Obj(reset_root_state=_Obj(params={"offset": 0.0}),
                            physics_material=_Obj(func="randomize_rigid_body_material", mode="startup",
                                                  params={"friction_range": (0.6, 2.0), "num_buckets": 64,
                                                          "asset_cfg": "robot"})))


def snap(o, depth=0):
    if depth > 8:
        return "<deep>"
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (list, tuple)):
        return [snap(v, depth + 1) for v in o]
    if isinstance(o, dict):
        return {str(k): snap(v, depth + 1) for k, v in o.items()}
    return {"__class__": type(o).__name__, **{k: snap(v, depth + 1) for k, v in sorted(vars(o).items())}}


def read_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("KRABBY_") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


envs = {name: read_env(path) for name, path in env_files.items()}
envs["synthetic/ring"] = {"KRABBY_FLAT_TERRAIN_GEOM": "ring"}
envs["synthetic/field"] = {"KRABBY_FLAT_TERRAIN_GEOM": "field"}
envs["synthetic/unset"] = {}
envs["synthetic/field+ring_var"] = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_RING_CLEAR_R": "2.0"}
envs["synthetic/bogus"] = {"KRABBY_FLAT_TERRAIN_GEOM": "corridor"}
results = {}
for name, env in envs.items():
    cfg = cfg_of()
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            g = tk.install_omni_terrain_generator(cfg, env, field_generator=field_gen())
            tk.apply_omni_ring_knobs(cfg, env, classes=(RingCfg, FlatCfg), field_classes=(FieldCfg, FlatCfg))
            if hasattr(tk, "apply_rt_friction_knobs"):
                tk.apply_rt_friction_knobs(cfg, env, exclusive_with=("KRABBY_STRESS_MU",))
    except Exception as exc:  # noqa: BLE001
        g = None
        err = type(exc).__name__          # the type, as step 1 compared (1 refusal message names "site")
    results[name] = {"geom": g, "stdout": buf.getvalue(), "err": err, "cfg": snap(cfg)}

rc = load("run_candidates_probe", O / "experiments/tools/run_candidates.py")
rcres = {"whitelist": sorted(rc.knob_whitelist()), "parsers": sorted(rc.knob_parsers()),
         "dump_checks": sorted(rc.OVERLAY_DUMP_CHECKS)}
if Path(tdr_dump).is_file() and Path(tdr_env).is_file():
    txt = Path(tdr_dump).read_text()
    verdicts = {}
    for k, v in sorted(read_env(tdr_env).items()):
        chk = rc.OVERLAY_DUMP_CHECKS.get(k)
        if chk is None or k in rc.AGENT_DUMP_KEYS:
            continue
        try:
            verdicts[k] = bool(chk(txt, v))
        except Exception as exc:  # noqa: BLE001
            verdicts[k] = f"exc:{type(exc).__name__}"
    rcres["tdr_dump_verdicts"] = verdicts
    rcres["dump_geom"] = rc.dump_terrain_geom(txt)
ok = load("omni_knobs_probe", O / "config/crab_hex/omni_knobs.py")
json.dump({"terrain": results, "rc": rcres, "omni_knobs_dir": sorted(n for n in dir(ok) if not n.startswith("__"))},
          open(out_path, "w"), indent=1, sort_keys=True)
'''


def _git(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, timeout=300, check=True, **kw)


@pytest.fixture(scope="module")
def main_tree(tmp_path_factory):
    try:
        _git("cat-file", "-e", f"{MAIN_SHA}^{{commit}}")
        blob = _git("archive", "--format=tar", MAIN_SHA, "--", *ARCHIVE_PATHS).stdout
    except Exception as exc:  # noqa: BLE001 -- no git / shallow clone
        pytest.skip(f"git archive {MAIN_SHA} unavailable: {exc}")
    root = tmp_path_factory.mktemp("main_3f944c4b")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(root, filter="data")
    return root


def _env_files() -> dict[str, str]:
    files = {}
    exp = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments"
    for d in RECORDED_ENV_DIRS:
        for p in sorted((exp / d).rglob("*.env")):
            files[f"wt/{p.relative_to(exp)}"] = str(p)
    if DRS_EXPERIMENTS.is_dir():
        for p in sorted(DRS_EXPERIMENTS.rglob("*.env")):
            if "2026-11-10_rough_terrain" not in p.parts:
                files[f"drs/{p.relative_to(DRS_EXPERIMENTS)}"] = str(p)
    return files


def _probe(root: Path, files: dict, tmp: Path, tag: str) -> dict:
    script = tmp / "probe.py"
    script.write_text(PROBE)
    listing = tmp / "env_files.json"
    listing.write_text(json.dumps(files))
    out = tmp / f"probe_{tag}.json"
    tdr_env = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments" / TDR_ENV
    proc = subprocess.run([sys.executable, "-I", str(script), str(root), str(listing), str(TDR_DUMP), str(tdr_env),
                           str(out)], capture_output=True, text=True, timeout=900, cwd=str(tmp),
                          env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def probes(main_tree, tmp_path_factory):
    files = _env_files()
    assert len(files) >= 100, len(files)
    tmp = tmp_path_factory.mktemp("rt_neutrality")
    return _probe(main_tree, files, tmp, "main"), _probe(REPO, files, tmp, "wt"), files


def test_every_recorded_env_composes_exactly_as_on_main(probes):
    main, wt, files = probes
    assert set(main["terrain"]) == set(wt["terrain"]) == set(files) | {
        "synthetic/ring", "synthetic/field", "synthetic/unset", "synthetic/field+ring_var", "synthetic/bogus"}
    diff = [k for k in main["terrain"] if main["terrain"][k] != wt["terrain"][k]]
    assert diff == [], diff[:5]
    geoms = {v["geom"] for v in main["terrain"].values()}
    assert {"field"} <= geoms                                                   # TDR's geometry is among them
    print(f"[rt neutrality] {len(files)} recorded env files + 5 synthetic: identical snapshots / stdout / errors")


def test_the_driver_tables_gain_exactly_the_friction_knob(probes):
    main, wt, _files = probes
    for table in ("whitelist", "parsers", "dump_checks"):
        added = set(wt["rc"][table]) - set(main["rc"][table])
        removed = set(main["rc"][table]) - set(wt["rc"][table])
        assert added == STEP1_KNOBS | {"KRABBY_RT_FRICTION"} and removed == set(), (table, added, removed)
    if "tdr_dump_verdicts" in main["rc"]:
        assert wt["rc"]["tdr_dump_verdicts"] == main["rc"]["tdr_dump_verdicts"]
        assert wt["rc"]["dump_geom"] == main["rc"]["dump_geom"] == "field"
    added = set(wt["omni_knobs_dir"]) - set(main["omni_knobs_dir"])
    assert added == STEP1_NAMES | {"RT_FRICTION_VAR", "RT_KNOB_PARSERS"}, added
    assert not set(main["omni_knobs_dir"]) - set(wt["omni_knobs_dir"])
