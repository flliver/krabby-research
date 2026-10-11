# Close — p2r stand regime (omni phase 2, 2026-10-09 → 2026-10-11)

**Status: CLOSED 2026-10-11 by the user. Nothing from p2r is baked.** The head of record is unchanged: the pre-Z1
S1 s18 (chain `P2__S1_s18`, run `crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt`,
sha256 `968cc550…`), shipped as `policy/6_p2a_field` since the 2026-10-08 bake (`manifest.yaml`: `current` and
`heads.phase2` = `6_p2a_field`). It still steps in place at zero command: 2.83 steps/s, 0.135 m lifts (P2R_PLAN.md
L28-30). Every p2r knob is default-off and set by no preset. With the knobs off, current code is bitwise-inert:
`inertness_bridge.json` pairs 0–6 are all `ok: true`, the last (REFH, code 14301cd2, eval 1d021625) equals HEAD's
code_sha. Z1's "stand of record" label (CHARTER L1799, L1802; "Z1 stays" at L1905) was internal to the campaign and is retired.

## The question and the answer

**Question.** Can the shared omni network get a still stand without the p2s bleed into walking (ST2: field_hard
0.8125 → 0.6175, PLAN L32-38)?

| Stage | Design | Decisive result | Verdict |
|---|---|---|---|
| Step 1 | Controller-level modes on the latched stand slot: HOLD (zero action), ST2M (ST2 specialist at stand), drift probe | HOLD: 0 steps/s, field_hard 0.810 vs 0.8125, 0 stand falls. ST2M: 16 taps/s. Drift STOP: D(C0) 1.35–1.42 > 0.5·D(ST2) = 1.14 (C:L1211-1222) | HOLD works as a fallback; G2/G1 premise fails |
| Z1 | Stand bit in col 15, statue prior + walking anchor, 5000 its | Stand 0.0 / 0.083 steps/s, field_hard 0.83, 0 / 1500 stand falls; but 55 % five-foot stands, cams unhomed (hip yaw 4.9–8.7°) (`p2r_z.json`; C:L1336-1340) | STILL, no COST, LEVER; one seed, not baked |
| Z2a | Current-sense re-stance R | t2: walkstop homed 50 %, 116/294 stands deadlocked; suite (from spawn) 37 taps/s (C:L1446-1449) | Premise STOP |
| Z2b | Cams-only kinematic R (probe) | RSTf: homed 95.7 / 94.7 %, 0 falls (C:L1561-1569) | Probe PASS (line set at 90 %) |
| Z2 | Statue-prior target = R(obs) | Walkstop homed 0 / 0 %, 7.38 / 5.04 taps/s; field_hard 0.84 (`p2r_z2_read.json`) | NO EFFECT |
| Z2c | Clean-state BC on R banks | Gate PASS (hip 0.020, cam 0.016); closed loop 50.9 / 47.7 taps/s, homed 0 / 0 % (`p2r_z2c_read.json`) | NO EFFECT |
| Z2d | Stand-row mask of last action + proxy, fine-tune | STILL, LEVER, field_hard 0.8225; walkstop homed 1.7 / 2.3 % vs mask-only 22.3 % (`p2r_z2d_read.json`, `p2r_z2d_2b.json`) | Better statue, does not home |
| Z2d rep | s16 / s17 replicates | Stopped at baselines; no training ran | Stopped |

**The arc.** Routing a separate controller at the stand slot was free for walking from the first day; every attempt
to put the stand inside the shared network was a distillation problem. Z1 learned a zero-action statue cheaply, but it
inherits HOLD's flaws (five feet, unhomed cams). The scripted re-stance R fixes those as a controller. The prior alone
(Z2) never fit R even open-loop (hip bias −0.13…−0.18 on clean HOLD rows); BC (Z2c) and mask + BC (Z2d) fit
open-loop and failed closed-loop. Fable ranked "hybrid (S1 walks, R stands)"
first (C:L1921).

## Why it closed and what comes next

The user stopped the replicate queue at 16:25 and closed p2r. The next campaign trains a task-specific stand network
separately from omni locomotion, so the two behaviours cannot interfere.

## Handoff to the stand-network campaign

**What already works end to end.** R as the stand on S1 (`P2R_RSTg_s18`, `p2r_rstg_read.json`): walkstop homed
97.0 / 97.7 %, settle median 2.92 / 2.96 s, suite STILL (KS 0.098 / 0.105), 0 stand falls, field_turn 0 falls / 1.0,
field_hard 0.81. Six-down 0.20–0.37 (suite 0.29 / 0.20, walkstop 0.37 / 0.32) -- not a policy property: five-foot stands are a
static-indeterminacy lottery (REFS 0 %, HOLD 29–35 %, R 0.20–0.37, Z1 9–24 %; C:L1337-1338), which is why the gate was
demoted to "reported". HOLD is the simpler STILL fallback (cams unhomed, yaw about 6°). Final
R constants: `lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3`
(`p2r_z2c_gate.json`). `KRABBY_RESTANCE=default` is still the 2026-10-10 seed set (up_frac 0.6, k_h 0.3, trim 0.03,
eps_step 0.1): pass the validated R above as the full string.

**Reusable assets.**
- `scripts/p2r_mode_eval_crab_hex_gait.py --p2r-mode st2 --p2r-stand-ckpt <ckpt>`: a separate checkpoint acts on
  stand rows while S1 walks, with its own obs clone and estimator output. `--p2r-mode restance` / `hold` give the two
  baselines; `--p2r-observe` gives the closed-loop view. **Not model-agnostic:** `_load_extra` (L919-924, used at L952)
  builds the specialist through the same runner factory with a deep copy of S1's `train_cfg` and `runner.load`s it, so
  the stand checkpoint must be an rsl_rl teacher checkpoint with S1's policy class, obs width 1149, the
  `num_prop/num_scan/num_priv_explicit` layout and the action dim; the factory refuses `DistillationWithExtractor`,
  `empirical_normalization` and student / `--canon` / `--sample-actions` paths (L906-917, L938-942). A stand network
  with its own observation set or architecture needs a new loader / adapter first. The only trained stand specialist
  so far, ST2, is `crab_hex_omni_teacher/2026-10-09_01-18-22_seed18_X15_p2s_A15pB/model_24993.pt`.
- `KRABBY_OMNI_STAND_OBS=clock` (stand bit + eval slot latch), `KRABBY_RESTANCE`, `KRABBY_OMNI_STAND_MASK`,
  `KRABBY_BC_*`, `KRABBY_STAND_PRIOR_*`, `KRABBY_ANCHOR_*` (`motion_knobs.py` L294-330).
- `mdp/crab_hex_omni_restance.py` (R, mirror-equivariant); `p2s_stand.py` (per-stand homed / six-down / settle / yaw);
  `p2_read.py` stand reads; `p2r_z2c_observe_read.py`.
- R-labelled stand banks `OUT/lr/P2R_RSTg_s18` (100k rows each); held-out gate bank
  `P2R_RSTg_s18/walkstop__omni/seed002/2026-10-10_22-21-17/p2r_stand_bank.npz`; observe banks `P2R_OBSZ2{c,cM,d}_s18`.
- Baselines: REFS_S1_s16 / s18 (full), P2R_HOLD_s18 / s16, C0R_s18.

**Lessons, with evidence.**
1. A routed stand controller costs walking nothing; a shared reward does (HOLD / ST2M field_hard 0.810 / 0.835 vs ST2
   in-network 0.6175). Routing does not fix a chattery objective: ST2M still tapped 16/s (C:L1212).
2. Key on the latched stand slot, not command = 0 (which fires inside settled turns), and re-derive the latch on reset
   (move_turn inherits it on about 62 % of envs) (PLAN L87-88, L327-328).
3. Closed-loop, not open-loop, is the test. Z2c and Z2d gates passed and the heads did not home; the gate's held-out
   mean was 97.7 % parked-proxy rows (C:L1793-1794). Read every stand network with an observe pass.
4. Feedback through the network's own action history and the proxy carries errors: hip bias −0.046 with the proxy
   masked, −0.088 with last action masked, +0.003 with both (C:L1785-1788). Masking both kills the stepping phase,
   which the network read from its last action (C:L1795-1798). A still stand is proxy-blind (a parked rod reads −0.5).
5. Scripted targets with discontinuities do not distil: R's LIFT onset jumps 0 → −0.5; μ_hip on LIFT rows was −0.054,
   onset rows 0.22–0.25 % of the banks (C:L1908-1911). Give the network a target it can represent, or oversample onsets.
6. Cloning only clean states leaves off-manifold behaviour unfit: Z2c fit hip 0.007 held-out, yet left the manifold
   within 3–4 steps (C:L1742-1766); the HOLD trim slope fit −0.04…−0.12 vs R's −1 because clean rows sit at
   |c| ≈ 0.0035 (C:L1912-1917). Include off-home and onset states (DAgger-style).
7. Under σ ≈ 0.8 training noise, stand rows rarely reach the clean state (Z2, C:L1652-1655).
8. Current-sense load sensing at a still stand is not viable as built (C:L1451-1461); position-"down" cannot be tighter
   than load sag; a tilt guard of 0.10 rad removed terrain tip-overs (C:L1542-1546).
9. Calibration traps: re-derive dose bands per input regime (X19 kill, C:L1855-1859); detach the priv latent under BC;
   same-seed reruns replay bit-identically; run each arm as one job (resume resets the ramp).
10. Zero-action statues never fell (HOLD 0 / 1400, Z1 0 / 1500) but neither corrects heading drift (C:L1217).
11. The mode-switch contract is fixed and verified (P2R_PLAN L83-88; C:L1196-1205): entry on the first latched-stand
    step, no debounce; exit on the step the slot changes; a turn hold stays "turn" for its whole length even once
    settled; stand-mode share 1.0 on settle / stand and 0 elsewhere in every wave; handover falls 0, walkstop onset
    0.12 s (REFS 0.14). A separate stand network inherits this contract.
12. Eval carry: the runner-side knobs (`STATUE_KNOBS`, `RESTANCE_KNOBS`, `BC_KNOBS`) are not carried into evals
    (`motion_knobs.py` L190, L205); an eval of a Z-family head needs `KRABBY_OMNI_STAND_OBS=clock` (+
    `KRABBY_OMNI_STAND_MASK=1` for Z2d heads, where `p2_read`'s masked-head INVALID rule fails closed). Post-lock obs
    knobs reach evals only via run_arm's stamp env + `P2_ENV_EXTRA` (`EVAL_CARRY_EXTERNAL`); ladder driver stamps lack
    them.

## Loose ends

| Item | State | Action |
|---|---|---|
| Chains `P2__Z2d_s16` / `_s17` (`candidates.yaml` L1443 / L1505, v19) | never launched | left as record |
| REFS_S1_s17 walkstop es1 (es2 never started) | es1 interrupted; empty dir `OUT/lr/REFS_S1_s17/walkstop__omni/seed001/2026-10-11_20-23-04/` | `rmdir` only on user approval (a newest-stamp reader would pick it) |
| DIAG_REFSAMP_s17, P2R_HOLD_s17, GRAFT_S1_s17 rows; per-seed dose files | never ran | left as record |
| Graft `S1_s16_c15` | CPU identity PASS; GPU rows run, `--p2r-graft-identity` PASS (read-only, 2026-10-11); stamps not written to the sidecar | left as record |
| Graft `S1_s17_c15` | CPU identity only | left as record |
| Z1 / Z2d replicates, any p2r bake | never ran | closed |
| Z2e, HOLD / ST2M tier 2, no-anchor control, from-scratch STAND_OBS chain | never ran | left as record |
| Read defects: `p2r_z_fidelity` HOLD None; `read_tb` null extras; sentinel truncation | recorded, unconfirmed fixed | fix if reused |
| Kit DerivedDataCache | stopped Isaac at 16:25 | check before next Isaac launch if it was SIGKILLed |
| `manifest.yaml` `followups.stand_march`, `PHASE2_REPORT.md` stand bullet | updated at the close (this commit) | done |

## Pointers

- CHARTER §12 (L1170-1185) and §12.1 log (L1186-1934).
- Plans: `P2R_PLAN.md`, `P2R_STATUE_PLAN.md`, `P2R_Z2_PLAN.md`, `P2R_Z2C_PLAN.md`, `P2R_Z2D_PLAN.md`,
  `P2R_Z2D_REP_PLAN.md`.
- Records: `p2r_step1.json`, `p2r_drift/`, `p2r_z.json`, `p2r_z2_read.json`, `p2r_rstg_read.json`, `p2r_z2c_gate.json`,
  `p2r_z2c_read.json`, `p2r_z2c_observe.json`, `p2r_z2d_2b.json`, `p2r_z2d_gate.json`, `p2r_z2d_read.json`,
  `p2r_z2d_observe.json`, `inertness_bridge.json`; also `p2r_z1_early.json`, `p2r_z2_early.json`, `p2r_z2c_early.json`,
  `p2r_z2_dose.json`, `p2r_z2d_dose.json`, `p2r_z2d_gatereport_20500.json`, `p2r_rstresg_read.json`, the three
  `*_residual.json`, `z2d_premise/`; outputs under `parkour/outputs/omni_plan/phase2/`.
