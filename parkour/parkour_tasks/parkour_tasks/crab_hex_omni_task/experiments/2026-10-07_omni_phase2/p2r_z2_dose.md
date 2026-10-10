# p2r Z2 dose (P2R_Z2_PLAN step 4, CPU)

- bank: `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/DIAG_REFSAMP_s18/suite__omni/seed001/2026-10-09_01-49-08` (100 episodes, 60000 rows, labels ['stand']); actions **sampled** (debias c = 9.0000; sigma_sample == sigma_A: True)
- sigma_A from `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt`; R = `lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3`
- rebuild cross-check: {"stamp": "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_PROBE_s18/walkstop__omni/seed001/2026-10-09_17-26-50", "bank_rows": 5000, "matched": 5000, "hip_bitwise_share": 1.0, "hip_max_abs_diff": 0.0, "hip_default_source": "linkage.hip_default_rad", "cams_bitwise": true, "tol": 1e-05, "ok": true}

| per stand row | mean | SE | n |
|---|---|---|---|
| restance | 16.7028 | 0.0418 | 60000 |
| target0 | 16.0892 | 0.04 | 60000 |
| restance_minus_target0 | 0.6137 | 0.011 | 60000 |

| share | E restance | E target-0 | band | excludes own target-0 | excludes Z1 1.99 |
|---|---|---|---|---|---|
| 0.12 | 2.0043 | 1.9307 | [1.042, 4.009] | False | False |
| 0.16 | 2.6725 | 2.5743 | [1.390, 5.345] | False | False |

- union band [1.042, 5.345]; target-0 refs inside: ['target0@0.12', 'target0@0.16', 'z1_realised']
- R branch shares (leg-rows): {"hold": 0.3093, "wait": 0.1309, "lift": 0.0107, "home": 0.17, "seat": 0.3792}
- probe branch shares: null
- R non-zero share 0.9999; legs DOWN 0.4508 / UP 0.4689 (hip default from linkage.hip_default_rad)

**rc 0: band [1.0422571847616957, 5.344908639803569] does not exclude ['target0@0.12', 'target0@0.16', 'z1_realised'] (informational: S1's mu dominates both targets); home branch present**

## BC dose (Z2c)

- fork `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt`, training path; R = `lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3`; 487500 pooled rows
- **bc_mb0_expected 7.5683** (pooled SE 0.01933908653323688); +200 band [3.935, 15.137]

| bank | rows | BC loss per row | SE | R vs stored max abs |
|---|---|---|---|---|
| `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTg_s18/walkstop__omni/seed001/2026-10-10_22-18-27/p2r_stand_bank.npz` | 100000 | 9.2287 | 0.050640057243640485 | 0.0 |
| `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTg_s18/suite__omni/seed001/2026-10-10_22-24-03/p2r_stand_bank.npz` | 100000 | 3.8108 | 0.0035183913703192337 | 0.0 |
| `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTg_s18/field_turn__omni/seed001/2026-10-10_22-38-42/p2r_stand_bank.npz` | 40000 | 20.2158 | 0.12861405513595042 | 0.0 |
| `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTg_s18/field_hard__omni/seed001/2026-10-10_22-41-00/p2r_stand_bank.npz` | 47500 | 4.4597 | 0.0035660576667944704 | 0.0 |
| `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTresg_s18/walkstop__omni/seed001/2026-10-10_22-44-47/p2r_stand_bank.npz` | 100000 | 9.4041 | 0.05129257233851035 | 0.0 |
| `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTresg_s18/suite__omni/seed001/2026-10-10_22-47-41/p2r_stand_bank.npz` | 100000 | 4.2470 | 0.0057174416534573475 | 0.0 |
