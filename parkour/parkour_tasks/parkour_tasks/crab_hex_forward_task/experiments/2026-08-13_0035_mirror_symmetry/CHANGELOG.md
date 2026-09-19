# Mirror-symmetry campaign — adoption record

**User decision 2026-08-13: adopt + bake.**

## New reference flat-walk checkpoint

`fromscratch_sym_20k/logs/rsl_rl/crab_hex_flat_walk/2026-08-13_00-35-14/model_19999.pt`

Provenance: 20k iterations from scratch, 256 envs, seed 1, plain baked reward config (no v5
override), L/R mirror-symmetry loss coef 0.5 (KRABBY_SYM_LOSS_COEF), 6h14m. Supersedes
`2026-08-09_1526_gait_tuned/.../model_19999.pt` (tripod 0.401, B-handed 0.146/0.556).

Deterministic eval (flat_walk_forward, seed001): tripod **0.5170** (10 eps 0.50-0.53), duty
**0.356/0.339** (10/10 episodes balanced), pearson −0.730, 812 dominant-set swaps, completion
100%, slip 2.02%, tippy 6.6%, roll_rms 0.0498, EMA(v_z) 0.140, stride 0.148 m, signed pitch
+0.202 (lean unresolved — the isolated remaining target).

## Baked config change

`rsl_rl_ppo_cfg.py` `CrabHexFlatWalkPPORunnerCfg.__post_init__`: `KRABBY_SYM_LOSS_COEF`
default 0.0 → **0.5** (mirror loss ON by default for flat-walk training; set 0 to ablate).
Mirror maps: `mdp/crab_hex_mirror.py` (+13 unit tests). Carry into Task 4's reproducible
curriculum: from-scratch flat-walk = plain rewards + mirror loss 0.5.

## Open items

- Lean (+0.202 rad): candidate next campaign — clock reward with commanded body pitch
  (lit review §2, Walk These Ways) against the now-duty-balanced gait.
- n=1 seed: a confirmation seed (~6h) would establish recipe reliability for Task 4.
- Teacher-stack carry-up (bridge/2b1/2b2) of the symmetric policy not yet run.
