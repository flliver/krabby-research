# p2r Z2 dose (P2R_Z2_PLAN step 4, CPU)

- bank: `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/DIAG_REFSAMP_s18/suite__omni/seed001/2026-10-09_01-49-08` (100 episodes, 60000 rows, labels ['stand']); actions **sampled** (debias c = 9.0000; sigma_sample == sigma_A: True)
- sigma_A from `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt`; R = `lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=0.3,trim=0.03,eps_c=0.1,eps_air=0.05,tilt_max=0.1`
- rebuild cross-check: {"stamp": "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_PROBE_s18/walkstop__omni/seed001/2026-10-09_17-26-50", "bank_rows": 5000, "matched": 5000, "hip_bitwise_share": 1.0, "hip_max_abs_diff": 0.0, "hip_default_source": "linkage.hip_default_rad", "cams_bitwise": true, "tol": 1e-05, "ok": true}

| per stand row | mean | SE | n |
|---|---|---|---|
| restance | 16.7768 | 0.0422 | 60000 |
| target0 | 16.0892 | 0.04 | 60000 |
| restance_minus_target0 | 0.6876 | 0.0121 | 60000 |

| share | E restance | E target-0 | band | excludes own target-0 | excludes Z1 1.99 |
|---|---|---|---|---|---|
| 0.12 | 2.0132 | 1.9307 | [1.047, 4.026] | False | False |
| 0.16 | 2.6843 | 2.5743 | [1.396, 5.369] | False | False |

- union band [1.047, 5.369]; target-0 refs inside: ['target0@0.12', 'target0@0.16', 'z1_realised']
- R branch shares (leg-rows): {"hold": 0.1199, "wait": 0.3043, "lift": 0.0267, "home": 0.17, "seat": 0.3792}
- probe branch shares: null
- R non-zero share 0.9987; legs DOWN 0.4508 / UP 0.4689 (hip default from linkage.hip_default_rad)

**rc 0: band [1.046869748043747, 5.368562810480754] does not exclude ['target0@0.12', 'target0@0.16', 'z1_realised'] (informational: S1's mu dominates both targets); home branch present**
