# SPDX-License-Identifier: BSD-3-Clause
"""Split probe episodes by outcome: reached obstacle 1 vs not, and HOW the non-reachers ended
(fell vs timed out), plus where they fell (platform / field before obstacle 1).

Usage: python analyze_termination.py <probe_out_dir> [episode_length_s]
"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
ep_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
tl = np.load(d / "timeline_stochastic.npz")
t, x, kind, obst, env, g1, edge, failed = (tl[k] for k in ("t_s", "x_rel", "spawn_kind", "is_obst", "env", "goal1_x", "edge_x", "failed"))
n_env = int(env.max()) + 1
n_steps = len(t) // n_env
# reshape to (steps, envs): the probe appends one (n_env,) array per step
T = t.reshape(n_steps, n_env); X = x.reshape(n_steps, n_env); K = kind.reshape(n_steps, n_env)
O = obst.reshape(n_steps, n_env).astype(bool); G1 = g1.reshape(n_steps, n_env); E = edge.reshape(n_steps, n_env)
F = failed.reshape(n_steps, n_env).astype(bool)

rows = []
for e in range(n_env):
    te = T[:, e]
    # episode boundaries: t resets to ~0
    starts = [0] + [i for i in range(1, n_steps) if te[i] < te[i - 1]]
    ends = starts[1:] + [n_steps]
    for s0, s1 in zip(starts, ends):
        complete = s1 < n_steps  # a reset followed (otherwise the roll ended mid-episode)
        if not complete:
            continue
        seg = slice(s0, s1)
        xe = X[seg, e]
        fell = bool(F[s1 - 1, e])         # failure flag on the step that ended the episode
        t_end = float(te[s1 - 1]) + 0.02
        rows.append({
            "env": e, "kind": int(K[s0, e]), "obst": bool(O[s0, e]),
            "x0": float(xe[0]), "xmax": float(xe.max()), "x_end": float(xe[-1]),
            "goal1": float(G1[s0, e]), "edge": float(E[s0, e]),
            "fell": fell, "t_end": t_end, "timeout": t_end >= ep_s - 0.05,
        })
R = rows
plat_obst = [r for r in R if r["kind"] == 0 and r["obst"]]
print(f"episodes complete: {len(R)} | platform-spawned obstacle-tile: {len(plat_obst)}")
reached = [r for r in plat_obst if r["xmax"] >= r["goal1"]]
missed = [r for r in plat_obst if r["xmax"] < r["goal1"]]
print(f"reached obstacle 1: {len(reached)} ({len(reached)/max(1,len(plat_obst)):.2f}) | missed: {len(missed)}")
def pct(a, b): return f"{a}/{b} ({a/max(1,b):.2f})"
m_fell = [r for r in missed if r["fell"]]
m_to = [r for r in missed if r["timeout"] and not r["fell"]]
m_other = [r for r in missed if not r["fell"] and not r["timeout"]]
print(f"MISSED obstacle 1 -> fell: {pct(len(m_fell), len(missed))} | timed out: {pct(len(m_to), len(missed))} | other/early non-failure: {len(m_other)}")
if m_fell:
    on_plat = [r for r in m_fell if r["x_end"] <= r["edge"]]
    print(f"  fell ON the platform (before the edge): {pct(len(on_plat), len(m_fell))}; in the field before obstacle 1: {pct(len(m_fell)-len(on_plat), len(m_fell))}")
    print(f"  fall time: median {np.median([r['t_end'] for r in m_fell]):.1f} s | distance from spawn at fall: median {np.median([r['x_end']-r['x0'] for r in m_fell]):.2f} m | edge is {np.median([r['edge']-r['x0'] for r in m_fell]):.2f} m from spawn, obstacle-1 goal {np.median([r['goal1']-r['x0'] for r in m_fell]):.2f} m")
if m_to:
    print(f"  timed-out non-reachers: max x from spawn median {np.median([r['xmax']-r['x0'] for r in m_to]):.2f} m (edge {np.median([r['edge']-r['x0'] for r in m_to]):.2f}, obstacle-1 goal {np.median([r['goal1']-r['x0'] for r in m_to]):.2f}); past the edge: {pct(sum(r['xmax']>r['edge'] for r in m_to), len(m_to))}")
r_fell = [r for r in reached if r["fell"]]
print(f"REACHED obstacle 1 -> fell later: {pct(len(r_fell), len(reached))} | timed out: {pct(sum(r['timeout'] and not r['fell'] for r in reached), len(reached))}")
if r_fell:
    print(f"  fall time after reaching: median {np.median([r['t_end'] for r in r_fell]):.1f} s | x past obstacle-1 goal at fall: median {np.median([r['x_end']-r['goal1'] for r in r_fell]):.2f} m")
flat = [r for r in R if r["kind"] == 0 and not r["obst"]]
print(f"flat-tile platform episodes: {len(flat)} | fell: {pct(sum(r['fell'] for r in flat), len(flat))} (gait-only base rate)")
