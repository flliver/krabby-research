# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic compile-schedule snapshot generator (omni-task per-category eval plan, step 0
"reference capture").

Pure numpy + the existing ``gait_eval.schedule`` module -- no Isaac Sim import anywhere in this
file or in anything it imports, so it runs in plain ``python`` with no GPU.

Loads every committed eval-scenario manifest across the three tasks (forward v1/v2/morph,
sideways, omni) with ``gait_eval.schedule.load_manifest``, and for every scenario row in each
manifest compiles it with ``gait_eval.schedule.compile_schedule`` at the harness's own
``dt=0.02`` and ``num_envs=scenario.episodes`` -- the same values
``scripts/eval_crab_hex_gait.py`` ends up using (it builds the env with
``num_envs=scenario.episodes`` and then calls
``S.compile_schedule(scenario, dt=dt, num_envs=num_envs)`` with that same env's ``num_envs``).

For each row it records ``n_steps``, ``hold_labels``, and an identity fingerprint (dtype, shape,
sha256 of the C-contiguous bytes) of each array field on the resulting ``CompiledSchedule``:
``cmd``, ``segment_id``, ``steady_mask``, ``delta_yaw``, ``world_speed``, ``world_dir``,
``world_face`` (``null`` for any field that is ``None``, e.g. ``delta_yaw`` on a non-yaw-probe
scenario or the three ``world_*`` fields on a body-frame scenario).

Rows are keyed ``"<manifest filename>::<scenario id>"``.

Importable surface for a later test:
    ``MANIFESTS: list[Path]``
    ``build_snapshot() -> dict[str, dict]``

Run directly to (re)write the snapshot JSON:
    ``python tests/unit/fixtures/gen_gait_eval_compile_snapshot.py``
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = (
    _REPO_ROOT
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "scripts"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gait_eval import schedule as S  # noqa: E402

DT = 0.02

MANIFESTS: list[Path] = [
    _REPO_ROOT
    / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v1.yaml",
    _REPO_ROOT
    / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml",
    _REPO_ROOT
    / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_morph.yaml",
    _REPO_ROOT
    / "parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/eval/scenarios_side.yaml",
    _REPO_ROOT
    / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml",
]

FIXTURE_PATH = Path(__file__).resolve().parent / "gait_eval_compile_snapshot.json"

# The array fields on CompiledSchedule we fingerprint; None on a field that doesn't apply to a
# given row (e.g. delta_yaw outside a yaw probe, world_* on a body-frame scenario).
ARRAY_FIELDS = (
    "cmd",
    "segment_id",
    "steady_mask",
    "delta_yaw",
    "world_speed",
    "world_dir",
    "world_face",
)


def _array_identity(arr: np.ndarray | None) -> dict | None:
    if arr is None:
        return None
    contig = np.ascontiguousarray(arr)
    return {
        "dtype": str(contig.dtype),
        "shape": list(contig.shape),
        "sha256": hashlib.sha256(contig.tobytes()).hexdigest(),
    }


def build_snapshot() -> dict[str, dict]:
    """Compile every scenario in ``MANIFESTS`` and return the keyed identity snapshot."""
    snapshot: dict[str, dict] = {}
    for manifest_path in MANIFESTS:
        scenarios, _meta = S.load_manifest(manifest_path)
        for scenario in scenarios:
            compiled = S.compile_schedule(scenario, dt=DT, num_envs=scenario.episodes)
            key = f"{manifest_path.name}::{scenario.id}"
            if key in snapshot:
                raise ValueError(f"duplicate snapshot key {key!r}")
            snapshot[key] = {
                "n_steps": compiled.n_steps,
                "hold_labels": list(compiled.hold_labels),
                "arrays": {
                    field: _array_identity(getattr(compiled, field)) for field in ARRAY_FIELDS
                },
            }
    return snapshot


def main() -> None:
    snapshot = build_snapshot()
    FIXTURE_PATH.write_text(json.dumps(snapshot, sort_keys=True, indent=1) + "\n")

    counts: dict[str, int] = {}
    for key in snapshot:
        manifest_name = key.split("::", 1)[0]
        counts[manifest_name] = counts.get(manifest_name, 0) + 1
    print(f"wrote {FIXTURE_PATH} ({len(snapshot)} rows total)")
    for manifest_name in [m.name for m in MANIFESTS]:
        print(f"  {manifest_name}: {counts.get(manifest_name, 0)} rows")


if __name__ == "__main__":
    main()
