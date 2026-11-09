"""Diagnostic relaunch of G2's closing rows (2026-11-08): read by no gate, recorded in CHARTER section 6.

The closing compares the pick's cell row RSDS_155b5981_T00_nominal_L090_Z (es1) with its selection row
RSDSEL_155b5981_L090 (es1). This runs each row shape once more (suite + field_light), under `_D1` names, through
rsd_gate's own Runner: the same pin / clean-tree / disk preflight, launch, stamp checks and one retry as every gate
row, and the same lock (one driver at a time). Sentinels in results/rsd_gate.log: RSD DIAG START / END / STOP.
"""
import fcntl
import signal
import sys
from pathlib import Path

CD = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/dr-stress-sweep/parkour/parkour_tasks/"
          "parkour_tasks/crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr")
sys.path.insert(0, str(CD / "scripts"))
import rsd_lib as L  # noqa: E402
import rsd_gate as G  # noqa: E402

SUFFIX = "_D1"


def main() -> int:
    P = L.policy("PICK")
    sel = L.sel_name(P, 90) + SUFFIX
    rows = L.cell_rows(P, L.CLOSING_CELL, 1, SUFFIX) + [L.sel_row(P, "suite@1", 90, name=sel),
                                                        L.sel_row(P, "field_light@1", 90, name=sel)]
    lock = open(G.LOCK, "w")  # noqa: SIM115 -- held for the process lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("RSD another driver holds results/rsd_gate.lock", flush=True)
        return 6
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, G._on_signal)
    G.say(f"RSD DIAG START closing-class relaunch rows={len(rows)} names={rows[0]['name']},{sel} (read by no gate)")
    R = G.Runner(False, "diag")
    try:
        for r in rows:
            R.run(r)
    except L.Stop as e:
        G.say(f"RSD DIAG STOP {e}")
        return 4
    G.say(f"RSD DIAG END rows={R.n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
