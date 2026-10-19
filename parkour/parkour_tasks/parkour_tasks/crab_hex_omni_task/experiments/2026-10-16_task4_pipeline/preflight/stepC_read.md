# Step 8 (Phase C) read

**STOP-C: YES** -- ['C4_depth']

| check | class | verdict | first problem |
|---|---|---|---|
| C1 | stop | PASS |  |
| C3_readback | stop | PASS |  |
| C3_settle | stop | PASS |  |
| C3_S1_curve | stop | PASS |  |
| C3_pilot_live | stop | PASS |  |
| C3_np | stop | PASS |  |
| C3_gate | stop | PASS |  |
| C4_camera | stop | PASS |  |
| C4_readback | stop | PASS |  |
| C4_parity | stop | PASS |  |
| C4_timing_check | stop | PASS |  |
| C4_depth | stop | FAIL | depth - zeroed field_hard es1-2 = +0.015 < max(0.05, 2 SE) = 0.069 |
| C4_live | stop | PASS |  |
| C4_distill_payload_check | stop | PASS |  |
| C6_new | stop | PASS |  |
| C7 | stop | PASS |  |
| C2 | report | PASS |  |
| C4_det | report | REPORT |  |
| C5 | report | PASS |  |
| C6_det | report | PASS |  |

## Values for thresholds v1 / pipeline.yaml

```json
{
 "N_p": 3000,
 "x_rule": "report_only",
 "manifest_shape_split": 0.0,
 "live": {
  "student_loss_rule": "kill",
  "collapse_under_load": {
   "min_ep_len_over_kill_line": 1.7466921128957829,
   "max_crab_failure_roll": 0.36312662735581397
  }
 },
 "p3a_rates": {
  "stable90": {
   "available": true,
   "lines": {
    "p3a.stable90.survival.fwd.es1": {
     "line": 0.9,
     "mu_pilot": 0.99,
     "se": 0.009949874371066205,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.0,
     "rate": 7.459983198493968e-20
    },
    "p3a.stable90.tracking.fwd.es1": {
     "line": 0.25,
     "mu_pilot": 0.06550334884531805,
     "se": 0.00406190561008401,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.027006621667304814,
     "rate": 7.794443881837891e-07
    },
    "p3a.stable90.survival.side.es1": {
     "line": 0.9,
     "mu_pilot": 1.0,
     "se": 7.071067811865475e-05,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.0,
     "rate": 0.0
    },
    "p3a.stable90.tracking.side.es1": {
     "line": 0.25,
     "mu_pilot": 0.05446368912055344,
     "se": 0.002456353374160037,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.012970780179428432,
     "rate": 2.155033422755939e-26
    },
    "p3a.stable90.survival.back.es1": {
     "line": 0.9,
     "mu_pilot": 1.0,
     "se": 0.0001,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.0,
     "rate": 0.0
    },
    "p3a.stable90.tracking.back.es1": {
     "line": 0.25,
     "mu_pilot": 0.15980032418617074,
     "se": 0.0028791642238623328,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.04145980084581892,
     "rate": 0.062204764961931225
    },
    "p3a.stable90.survival.diag.es1": {
     "line": 0.9,
     "mu_pilot": 1.0,
     "se": 0.0001,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.0,
     "rate": 0.0
    },
    "p3a.stable90.tracking.diag.es1": {
     "line": 0.25,
     "mu_pilot": 0.006775770352925425,
     "se": 0.003475074343590541,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.026754888504485874,
     "rate": 7.705558628799053e-11
    },
    "p3a.stable90.survival.turn.es1": {
     "line": 0.885,
     "mu_pilot": 1.0,
     "se": 7.071067811865475e-05,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.0,
     "rate": 0.0
    },
    "p3a.stable90.survival.move_turn.es1": {
     "line": 0.885,
     "mu_pilot": 1.0,
     "se": 7.071067811865475e-05,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.005906674542232728,
     "rate": 2.0262859051278654e-43
    },
    "p3a.stable90.turn_fc.es1": {
     "line": 0.8,
     "mu_pilot": 0.9532445726060871,
     "se": 0.0019152086655431074,
     "se_stand_in": false,
     "sigma_hat_p2a": 0.00891714593552254,
     "rate": 1.4973636272118751e-33
    },
    "p3a.stable90.h_meas.suite.es1": {
     "line": 0.85,
     "mu_pilot": 1.1052122811476388,
     "se": null,
     "se_stand_in": true,
     "sigma_hat_p2a": 0.019659211396654615,
     "rate": 3.3135174339605455e-14
    },
    "p3a.stable90.roll_rms.suite.es1": {
     "line": 0.1,
     "mu_pilot": 0.014904559393889832,
     "se": null,
     "se_stand_in": true,
     "sigma_hat_p2a": 0.004925640436165851,
     "rate": 9.873239116347458e-24
    },
    "p3a.stable90.roll_rms.field_light.es1": {
     "line": 0.1,
     "mu_pilot": 0.02681543508202464,
     "se": null,
     "se_stand_in": true,
     "sigma_hat_p2a": 0.0019468551691453335,
     "rate": 9.553453413649643e-105
    },
    "p3a.stable90.pitch_rms.suite.es1": {
     "line": 0.1,
     "mu_pilot": 0.026902461941730793,
     "se": null,
     "se_stand_in": true,
     "sigma_hat_p2a": 0.005076124266335523,
     "rate": 4.6265641930444084e-17
    },
    "p3a.stable90.pitch_rms.field_light.es1": {
     "line": 0.1,
     "mu_pilot": 0.05155205577101463,
     "se": null,
     "se_stand_in": true,
     "sigma_hat_p2a": 0.004140325267398984,
     "rate": 7.099810287396635e-12
    },
    "p3a.stable90.h_meas.field_light.es1": {
     "line": 0.85,
     "mu_pilot": 1.10879917939504,
     "se": null,
     "se_stand_in": true,
   
```

Details per check: `/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/preflight/stepC/checks`.
