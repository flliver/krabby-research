# REPLAY_GATE -- stroke/posture plan step 7

Economy: `/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-25_14-53-27_seed3_X03_c6x_A15pB/params/env.yaml` (clock [0.001, 0.001], tibia weight -0.28). Parameters: `{"rho0": 0.5, "drho": 0.3, "power": 1.5, "c_max": 1.5, "prog_tau": 4.0, "prog_lo": 0.15, "prog_width": 0.15, "force_open": 50.0, "force_hold": 20.0, "n_off": 2, "min_loaded": 5, "sigma": 0.2, "tau": 2.0, "min_cmd": 0.12, "phi_lo": 0.1, "phi_sat": 0.25}`.

**w_S = 0.08647**, **w_P = 0.1549**

| Id | Verdict | Clauses | Deciding numbers |
|---|---|---|---|
| R1 | **FAIL** |  | min kept X0x_s2.turn 0.641 (>= 0.8) |
| R2 | **FAIL** | side_kept<=0.40: FAIL; slide_kept<=0.25: PASS; side<=0.2xsideways: PASS | max side kept X0x_s2 0.434; slide kept 0.037; max side/sideways X0x_s2/side_step 0.150 |
| R3 | **FAIL** | short<=2%: PASS; spearman<=+0.2: FAIL; v4<=healthy: N/A | max short share E1.side 0.0096; max Spearman W.side 0.293 |
| R4 | **PASS** | fwd>=5%: PASS; stand==0: PASS; turn>0: PASS; move_turn>0: PASS | w_S 0.08647; binding W:fwd; fwd share W 0.100, X0x_s3 0.086, X0x_s2 0.096, E1 0.089; sideways side share side_slow 0.146, side_step 0.124 |
| R5 | **PASS** | no_flip: PASS; term_alone<=0.5x (translation): PASS | no-flip fails: none; max term-alone (translation) E7_creeper/X0x_s3.fwd 0.143; max term-alone (rotational, reported) E7_creeper/E1.move_turn 1.910 |
| R6 | **FAIL** |  | max income shift W.move_turn 0.0219; max liftoff shift E1.side 0.0414 |
| R7 | **PASS** |  | max |clip shift| pp E1.diag -0.51; max absorbed E1.move_turn 0.109 |
| R8 | **FAIL** |  | max hand gap E1.move_turn 0.120 |
| R9 | **FAIL** |  | failing neighbours: ['force_hold=15', 'force_hold=25', 'n_off=1', 'n_off=3', 'rho0=0.4', 'rho0=0.6', 'drho=0.2', 'drho=0.4', 'power=2', 'c_max=1.25', 'c_max=1.75', 'prog_lo=0.1', 'prog_lo=0.2', 'prog_width=0.1', 'prog_width=0.2', 'prog_tau=3', 'prog_tau=5', 'force_open=40', 'force_open=60', 'min_loaded=4', 'min_loaded=6'] of ['force_hold=15', 'force_hold=25', 'n_off=1', 'n_off=3', 'rho0=0.4', 'rho0=0.6', 'drho=0.2', 'drho=0.4', 'power=2', 'c_max=1.25', 'c_max=1.75', 'prog_lo=0.1', 'prog_lo=0.2', 'prog_width=0.1', 'prog_width=0.2', 'prog_tau=3', 'prog_tau=5', 'force_open=40', 'force_open=60', 'min_loaded=4', 'min_loaded=6'] |
| P1 | **FAIL** | sideways>=0.85: FAIL; W side in [0.15,0.45]: PASS; fwd/back/stand/turn==0: PASS | sideways side {'side_slow': 0.9420073938338351, 'side_step': 0.752277317627347}; healthy side W 0.373, X0x_s3 0.345, X0x_s2 0.393, E1 0.358; max |zero row| E1.back 4.22e-08 |
| P2 | **PASS** | diag<=10%: PASS | w_P 0.1549 band [0.12390134887461453, 0.1935958576165852]; max healthy share X0x_s2.side 0.084 |
| P3 | **PASS** | no_flip: PASS; x08 side vs statue>=1.10: PASS; creeper side<=0.10x sideways: PASS | no-flip PASS; x08/statue side totals 0.875/-35.149 ratio --; creeper/sideways 0.027 |
| P4 | **PASS** |  | max |clip shift| pp E1.diag -1.73; max hand gap X0x_s3.side.hands 0.025 |

Dropped inputs (recorded, not silently skipped):
- forward-era degenerates (v1 lunge, v2 tip-rock, v3 skate, v3b drag) and v4 tippy-tap: LOCATED under crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/*/gait_eval*/ (untracked, main checkout) but NOT loadable by the tool (forward-task stamps: no suite, travel_yaw 0, no joint_pos; pre-2026-08-20 plant). Replayed term-alone by replay_gate_calib/fwd_era.py instead; results in replay_gate.json step7.forward_era_term_alone.

## Runs

| name | role | kind | N | notes |
|---|---|---|---|---|
| W | healthy | omni | 1000 |  |
| X0x_s3 | healthy | omni | 1000 |  |
| X0x_s2 | healthy | omni | 1000 |  |
| E1 | healthy | omni | 1000 |  |
| side_slow | sideways | side | 100 | sideways head: cmd rotated by travel_yaw 1.5708; cos(cmd, achieved velocity) = 1.000 |
| side_step | sideways | side | 100 | sideways head: cmd rotated by travel_yaw 1.5708; cos(cmd, achieved velocity) = 1.000 |
| Z_t3 | posture | omni | 1000 |  |
| Z_c6 | posture | omni | 1000 |  |
| A4500 | posture | omni | 1000 |  |
| E6_statue | degenerate | omni | 1000 |  |
| E7_creeper | degenerate | omni | 1000 |  |

## Numbers

### R1: FAIL

```
{
 "headline": "min kept X0x_s2.turn 0.641 (>= 0.8)",
 "kept": {
  "E1.back": 0.8417061567306519,
  "E1.fwd": 0.8487353324890137,
  "E1.turn": 0.7133184671401978,
  "W.back": 0.8417195677757263,
  "W.fwd": 0.8645230531692505,
  "W.turn": 0.7714568972587585,
  "X0x_s2.back": 0.8526707291603088,
  "X0x_s2.fwd": 0.8754606246948242,
  "X0x_s2.turn": 0.6411575675010681,
  "X0x_s3.back": 0.8370416760444641,
  "X0x_s3.fwd": 0.8511626124382019,
  "X0x_s3.turn": 0.7065048813819885,
  "side_slow.side": 0.8696284890174866,
  "side_step.side": 0.7734588384628296
 },
 "rule": "kept >= 0.8"
}
```

### R2: FAIL

```
{
 "clauses": {
  "side<=0.2xsideways": true,
  "side_kept<=0.40": false,
  "slide_kept<=0.25": true
 },
 "headline": "max side kept X0x_s2 0.434; slide kept 0.037; max side/sideways X0x_s2/side_step 0.150",
 "side_income": {
  "E1": 0.1069999025083498,
  "W": 0.1128201623723601,
  "X0x_s2": 0.1153940069680353,
  "X0x_s3": 0.11100499637467647
 },
 "side_kept": {
  "E1": 0.39817410707473755,
  "W": 0.3835206925868988,
  "X0x_s2": 0.43442001938819885,
  "X0x_s3": 0.416259765625
 },
 "side_vs_sideways": {
  "E1/side_slow": 0.09259188472205938,
  "E1/side_step": 0.13935856531100263,
  "W/side_slow": 0.09762842043608788,
  "W/side_step": 0.14693897468868772,
  "X0x_s2/side_slow": 0.09985568528875123,
  "X0x_s2/side_step": 0.15029119540831673,
  "X0x_s3/side_slow": 0.09605767469829798,
  "X0x_s3/side_step": 0.14457487039224906
 },
 "sideways_income": {
  "side_slow": 1.1556077817137014,
  "side_step": 0.7678028420396056
 },
 "slide_kept": 0.03743241177878534
}
```

### R3: FAIL

```
{
 "clauses": {
  "short<=2%": true,
  "spearman<=+0.2": false,
  "v4<=healthy": null
 },
 "headline": "max short share E1.side 0.0096; max Spearman W.side 0.293",
 "short_share": {
  "E1.back": 0.0010229656472802162,
  "E1.diag": 0.0022380638401955366,
  "E1.fwd": 0.0010546270059421659,
  "E1.move_turn": 0.0012801163829863071,
  "E1.side": 0.009635393507778645,
  "E1.turn": 0.0010691446950659156,
  "W.back": 0.0005498777609318495,
  "W.diag": 0.0014851948944851756,
  "W.fwd": 0.0004776741552632302,
  "W.move_turn": 0.0012847227044403553,
  "W.side": 0.006680465769022703,
  "W.turn": 0.0006532523548230529,
  "X0x_s2.back": 0.0007611064938828349,
  "X0x_s2.diag": 0.0016695299418643117,
  "X0x_s2.fwd": 0.0006615109741687775,
  "X0x_s2.move_turn": 0.0007766769849695265,
  "X0x_s2.side": 0.009458976797759533,
  "X0x_s2.turn": 0.0006941452156752348,
  "X0x_s3.back": 0.0010373389814049006,
  "X0x_s3.diag": 0.001698648789897561,
  "X0x_s3.fwd": 0.000975102127995342,
  "X0x_s3.move_turn": 0.0007873789872974157,
  "X0x_s3.side": 0.007899795658886433,
  "X0x_s3.turn": 0.00062461948255077,
  "side_slow.side": 0.0002873463381547481,
  "side_step.side": 0.00042001117253676057
 },
 "spearman_switch": {
  "E1.back": {
   "n": 100,
   "rho": -0.27477453723445305
  },
  "E1.diag": {
   "n": 100,
   "rho": -0.09432360088960301
  },
  "E1.fwd": {
   "n": 100,
   "rho": -0.29249833159328914
  },
  "E1.move_turn": {
   "n": 200,
   "rho": -0.19048097893883523
  },
  "E1.side": {
   "n": 200,
   "rho": -0.017246630591799792
  },
  "E1.turn": {
   "n": 200,
   "rho": 0.06224225381733299
  },
  "W.back": {
   "n": 100,
   "rho": -0.37184733973807665
  },
  "W.diag": {
   "n": 100,
   "rho": -0.00383535016508103
  },
  "W.fwd": {
   "n": 100,
   "rho": -0.1849196974795646
  },
  "W.move_turn": {
   "n": 200,
   "rho": -0.07072680649493353
  },
  "W.side": {
   "n": 200,
   "rho": 0.2933796632378114
  },
  "W.turn": {
   "n": 200,
   "rho": -0.19342985015551137
  },
  "X0x_s2.back": {
   "n": 100,
   "rho": -0.38246789092483535
  },
  "X0x_s2.diag": {
   "n": 100,
   "rho": 0.14782032106074952
  },
  "X0x_s2.fwd": {
   "n": 100,
   "rho": -0.1520159706137424
  },
  "X0x_s2.move_turn": {
   "n": 200,
   "rho": 0.06745596577155595
  },
  "X0x_s2.side": {
   "n": 200,
   "rho": 0.1556707572071818
  },
  "X0x_s2.turn": {
   "n": 200,
   "rho": -0.018855849613119913
  },
  "X0x_s3.back": {
   "n": 100,
   "rho": -0.23491517631934836
  },
  "X0x_s3.diag": {
   "n": 100,
   "rho": 0.008864109742542871
  },
  "X0x_s3.fwd": {
   "n": 100,
   "rho": -0.5215355970958073
  },
  "X0x_s3.move_turn": {
   "n": 200,
   "rho": 0.0012340733618882314
  },
  "X0x_s3.side": {
   "n": 200,
   "rho": 0.16012804581975845
  },
  "X0x_s3.turn": {
   "n": 200,
   "rho": -0.008918995081820083
  }
 },
 "v4": {}
}
```

### R4: PASS

```
{
 "clauses": {
  "fwd>=5%": true,
  "move_turn>0": true,
  "stand==0": true,
  "turn>0": true
 },
 "headline": "w_S 0.08647; binding W:fwd; fwd share W 0.100, X0x_s3 0.086, X0x_s2 0.096, E1 0.089; sideways side share side_slow 0.146, side_step 0.124",
 "note": "diag and side are reported, not gated (R4 names fwd, back, turn, move_turn); the sideways head is not priced in W's omni economy (nominal share).",
 "per_run": {
  "W": {
   "binding": "fwd",
   "shares": {
    "back": 0.07713974019977873,
    "diag": 0.053345539756553144,
    "fwd": 0.1,
    "move_turn": 0.0138001166566726,
    "side": 0.013472909565706726,
    "stand": null,
    "turn": 0.055344990198887806
   },
   "w_S": 0.08646779396708197
  }
 },
 "shares_at_w_S": {
  "E1": {
   "back": 0.07297545491203945,
   "diag": 0.04929579188279762,
   "fwd": 0.0889292107025232,
   "move_turn": 0.00961724737760776,
   "side": 0.013031101955015474,
   "turn": 0.04315155006441245
  },
  "W": {
   "back": 0.07713974019977873,
   "diag": 0.053345539756553144,
   "fwd": 0.1,
   "move_turn": 0.0138001166566726,
   "side": 0.013472909565706726,
   "turn": 0.055344990198887806
  },
  "X0x_s2": {
   "back": 0.0744808661887173,
   "diag": 0.057440046793394685,
   "fwd": 0.09583544860668092,
   "move_turn": 0.013207459951852497,
   "side": 0.01448915767588612,
   "turn": 0.034321473291361274
  },
  "X0x_s3": {
   "back": 0.07059037916547602,
   "diag": 0.05320477501199993,
   "fwd": 0.08641351463364777,
   "move_turn": 0.011792040487736995,
   "side": 0.013685663042550086,
   "turn": 0.045746942727054
  }
 },
 "sideways_side_share_nominal": {
  "side_slow": 0.14637464529593805,
  "side_step": 0.12405086696060195
 },
 "stand_pay": {
  "E1": 0.0,
  "W": 0.0,
  "X0x_s2": 0.0,
  "X0x_s3": 0.0
 },
 "w_S": 0.08646779396708197
}
```

### R5: PASS

```
{
 "clauses": {
  "no_flip": true,
  "term_alone<=0.5x (translation)": true
 },
 "headline": "no-flip fails: none; max term-alone (translation) E7_creeper/X0x_s3.fwd 0.143; max term-alone (rotational, reported) E7_creeper/E1.move_turn 1.910",
 "rg1": {
  "pass": true,
  "rows": [
   {
    "base": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.224839629235345,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.1578, free -36.1578); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.3122147467514647,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 1.031751053937961,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1490, free -35.1490); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 1.0415387825605351,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.4144079659795135,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.5869, free -35.5869); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.4848787682331683,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.8710372823320194,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.3432, free -35.3432); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.9097421969318387,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.2190878614605154,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.2200, free -35.2200); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.2679816623926956,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.3117100470691925,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.2581, free -36.2581); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.3334582514287947,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.224839629235345,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 1.6075158862538983,
    "ratio_with": 1.6990204635496873,
    "with": {
     "degenerate_total": 0.7723360459178421,
     "healthy_total": 1.3122147467514647,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3553215982913456,
     "healthy_total": 1.031751053937961,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 2.9037104946600456,
    "ratio_with": 2.9299180119107837,
    "with": {
     "degenerate_total": 0.3554839344740852,
     "healthy_total": 1.0415387825605351,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7151299339158457,
     "healthy_total": 1.4144079659795135,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 1.9778335361164685,
    "ratio_with": 2.0541398760228917,
    "with": {
     "degenerate_total": 0.7228713027606015,
     "healthy_total": 1.4848787682331683,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.5430119192276095,
     "healthy_total": 0.8710372823320194,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 1.6040850145075993,
    "ratio_with": 1.667285251531698,
    "with": {
     "degenerate_total": 0.5456428023315618,
     "healthy_total": 0.9097421969318387,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.2190878614605154,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 3.07264528758921,
    "ratio_with": 3.0952982186533036,
    "with": {
     "degenerate_total": 0.409647656807804,
     "healthy_total": 1.2679816623926956,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.3117100470691925,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -0.0228, free 0.0063); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": 0.00630840709521574,
     "healthy_total": 0.3334582514287947,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.085220068985984,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.1578, free -36.1578); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.1577954141428064,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.8738616820176787,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1490, free -35.1490); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.8834722104929092,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.1796833603997723,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.5869, free -35.5869); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.241681876198205,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.714856351412926,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.3432, free -35.3432); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.7519160223672683,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.291134827822507,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.2200, free -35.2200); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.3310101379602404,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7986707060099291,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.2581, free -36.2581); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.817770357986121,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.085220068985984,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.4242750310631278,
    "ratio_with": 1.4990824528549427,
    "with": {
     "degenerate_total": 0.7723360459178421,
     "healthy_total": 1.1577954141428064,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3553215982913456,
     "healthy_total": 0.8738616820176787,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 2.459354247588284,
    "ratio_with": 2.4852662098498137,
    "with": {
     "degenerate_total": 0.3554839344740852,
     "healthy_total": 0.8834722104929092,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7151299339158457,
     "healthy_total": 1.1796833603997723,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.6496070216780967,
    "ratio_with": 1.7177080781271816,
    "with": {
     "degenerate_total": 0.7228713027606015,
     "healthy_total": 1.241681876198205,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.5430119192276095,
     "healthy_total": 0.714856351412926,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.316465304168187,
    "ratio_with": 1.378037095246725,
    "with": {
     "degenerate_total": 0.5456428023315618,
     "healthy_total": 0.7519160223672683,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.291134827822507,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 3.2542357854324546,
    "ratio_with": 3.249158431253314,
    "with": {
     "degenerate_total": 0.409647656807804,
     "healthy_total": 1.3310101379602404,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.7986707060099291,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -0.0228, free 0.0063); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": 0.00630840709521574,
     "healthy_total": 0.817770357986121,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.149183320272118,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.1578, free -36.1578); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.2289231937449139,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.8994346630894414,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1490, free -35.1490); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.9093691629972076,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.008182731295575,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.5869, free -35.5869); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.0712632512748668,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.6580813797634171,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.3432, free -35.3432); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.699191276722096,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.16980319899595,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.2200, free -35.2200); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.1994630692704793,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7478507935470017,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.2581, free -36.2581); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7693800300756076,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.149183320272118,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.5082223006686197,
    "ratio_with": 1.5911767944023185,
    "with": {
     "degenerate_total": 0.7723360459178421,
     "healthy_total": 1.2289231937449139,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3553215982913456,
     "healthy_total": 0.8994346630894414,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.531325614357816,
    "ratio_with": 2.5581160632267634,
    "with": {
     "degenerate_total": 0.3554839344740852,
     "healthy_total": 0.9093691629972076,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7151299339158457,
     "healthy_total": 1.008182731295575,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.4097895829574025,
    "ratio_with": 1.481955705232422,
    "with": {
     "degenerate_total": 0.7228713027606015,
     "healthy_total": 1.0712632512748668,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.5430119192276095,
     "healthy_total": 0.6580813797634171,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.2119096403988416,
    "ratio_with": 1.2814084117565798,
    "with": {
     "degenerate_total": 0.5456428023315618,
     "healthy_total": 0.699191276722096,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.16980319899595,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.9484259506082418,
    "ratio_with": 2.928035958065387,
    "with": {
     "degenerate_total": 0.409647656807804,
     "healthy_total": 1.1994630692704793,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.7478507935470017,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -0.0228, free 0.0063); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
  
```

### R6: FAIL

```
{
 "headline": "max income shift W.move_turn 0.0219; max liftoff shift E1.side 0.0414",
 "pattern": "force repeats the previous reading on s % 5 == 4",
 "rows": {
  "E1.back": {
   "income": 0.748226052550376,
   "income_dup": 0.7346767197269922,
   "income_shift": 0.018108608724863307,
   "liftoff_shift": 0.017553335133675407,
   "liftoffs": 3703,
   "liftoffs_dup": 3768
  },
  "E1.diag": {
   "income": 0.40208041648704723,
   "income_dup": 0.398909964843041,
   "income_shift": 0.007885118284810533,
   "liftoff_shift": 0.031725336810082494,
   "liftoffs": 4602,
   "liftoffs_dup": 4748
  },
  "E1.fwd": {
   "income": 0.8454758237499523,
   "income_dup": 0.8331778570771972,
   "income_shift": 0.014545616003790451,
   "liftoff_shift": 0.019519519519519468,
   "liftoffs": 3996,
   "liftoffs_dup": 4074
  },
  "E1.move_turn": {
   "income": 0.17637868255012346,
   "income_dup": 0.17474053583138163,
   "income_shift": 0.009287668413535677,
   "liftoff_shift": 0.015360614424577035,
   "liftoffs": 8333,
   "liftoffs_dup": 8461
  },
  "E1.side": {
   "income": 0.1069999025083498,
   "income_dup": 0.10615782529426127,
   "income_shift": 0.007869887676045573,
   "liftoff_shift": 0.04142075329971018,
   "liftoffs": 9319,
   "liftoffs_dup": 9705
  },
  "E1.turn": {
   "income": 0.4359651732324237,
   "income_dup": 0.43222338446649444,
   "income_shift": 0.008582769899224019,
   "liftoff_shift": 0.009104704097116834,
   "liftoffs": 9226,
   "liftoffs_dup": 9310
  },
  "W.back": {
   "income": 0.8153986286946817,
   "income_dup": 0.8001125536969239,
   "income_shift": 0.018746750926265654,
   "liftoff_shift": 0.01994884910485939,
   "liftoffs": 3910,
   "liftoffs_dup": 3988
  },
  "W.diag": {
   "income": 0.4472456956221754,
   "income_dup": 0.44574938322368424,
   "income_shift": 0.0033456160967846316,
   "liftoff_shift": 0.03019607843137262,
   "liftoffs": 5100,
   "liftoffs_dup": 5254
  },
  "W.fwd": {
   "income": 1.0119824395728927,
   "income_dup": 0.9916763649915166,
   "income_shift": 0.020065639271316105,
   "liftoff_shift": 0.02037697401935823,
   "liftoffs": 3926,
   "liftoffs_dup": 4006
  },
  "W.move_turn": {
   "income": 0.25030056093954456,
   "income_dup": 0.24482741482046008,
   "income_shift": 0.021866295858627427,
   "liftoff_shift": 0.01721364221364219,
   "liftoffs": 6216,
   "liftoffs_dup": 6323
  },
  "W.side": {
   "income": 0.1128201623723601,
   "income_dup": 0.11204818074605155,
   "income_shift": 0.006842585669755041,
   "liftoff_shift": 0.03766986410871298,
   "liftoffs": 10008,
   "liftoffs_dup": 10385
  },
  "W.turn": {
   "income": 0.5691707934529899,
   "income_dup": 0.5615517828686832,
   "income_shift": 0.013386158727654363,
   "liftoff_shift": 0.015492957746478853,
   "liftoffs": 8520,
   "liftoffs_dup": 8652
  },
  "X0x_s2.back": {
   "income": 0.730907866719403,
   "income_dup": 0.7208481675669772,
   "income_shift": 0.013763293036614277,
   "liftoff_shift": 0.015041782729805053,
   "liftoffs": 3590,
   "liftoffs_dup": 3644
  },
  "X0x_s2.diag": {
   "income": 0.47778386937523354,
   "income_dup": 0.4737884413667947,
   "income_shift": 0.008362417118985999,
   "liftoff_shift": 0.02418558736426446,
   "liftoffs": 4052,
   "liftoffs_dup": 4150
  },
  "X0x_s2.fwd": {
   "income": 0.9221073799122814,
   "income_dup": 0.9043550538725311,
   "income_shift": 0.019251907561393833,
   "liftoff_shift": 0.01903628792385481,
   "liftoffs": 3362,
   "liftoffs_dup": 3426
  },
  "X0x_s2.move_turn": {
   "income": 0.24876188350455544,
   "income_dup": 0.24667837742026935,
   "income_shift": 0.008375503734469603,
   "liftoff_shift": 0.012547528517110162,
   "liftoffs": 7890,
   "liftoffs_dup": 7989
  },
  "X0x_s2.side": {
   "income": 0.1153940069680353,
   "income_dup": 0.1145610436008057,
   "income_shift": 0.007218428314568648,
   "liftoff_shift": 0.03257294429708213,
   "liftoffs": 9425,
   "liftoffs_dup": 9732
  },
  "X0x_s2.turn": {
   "income": 0.34337282972354255,
   "income_dup": 0.3393256793449602,
   "income_shift": 0.011786460745425886,
   "liftoff_shift": 0.011469788341019305,
   "liftoffs": 8457,
   "liftoffs_dup": 8554
  },
  "X0x_s3.back": {
   "income": 0.7171038828195822,
   "income_dup": 0.705427589385765,
   "income_shift": 0.016282568974396172,
   "liftoff_shift": 0.01856612396458157,
   "liftoffs": 3501,
   "liftoffs_dup": 3566
  },
  "X0x_s3.diag": {
   "income": 0.42886225587265103,
   "income_dup": 0.4296292323801679,
   "income_shift": 0.001788398249121359,
   "liftoff_shift": 0.03300110741971207,
   "liftoffs": 4515,
   "liftoffs_dup": 4664
  },
  "X0x_s3.fwd": {
   "income": 0.8392253627785985,
   "income_dup": 0.8294978886145522,
   "income_shift": 0.011591015471505206,
   "liftoff_shift": 0.017553636110337223,
   "liftoffs": 3589,
   "liftoffs_dup": 3652
  },
  "X0x_s3.move_turn": {
   "income": 0.22130187184071548,
   "income_dup": 0.22023714179627815,
   "income_shift": 0.004811211200254517,
   "liftoff_shift": 0.018770923003347706,
   "liftoffs": 8364,
   "liftoffs_dup": 8521
  },
  "X0x_s3.side": {
   "income": 0.11100499637467647,
   "income_dup": 0.10998370974049328,
   "income_shift": 0.009200366357709044,
   "liftoff_shift": 0.04079851641758481,
   "liftoffs": 9167,
   "liftoffs_dup": 9541
  },
  "X0x_s3.turn": {
   "income": 0.46469717400439847,
   "income_dup": 0.4612294756279246,
   "income_shift": 0.007462275586037959,
   "liftoff_shift": 0.014359219815723412,
   "liftoffs": 8357,
   "liftoffs_dup": 8477
  },
  "side_slow.side": {
   "income": 1.1556077817137014,
   "income_dup": 1.1366643074333433,
   "income_shift": 0.016392650326623737,
   "liftoff_shift": 0.0011569052783803047,
   "liftoffs": 6915,
   "liftoffs_dup": 6907
  },
  "side_slow.side.f32_replica": {
   "gated": false,
   "income": 1.1556077817137014,
   "income_dup": 1.1392199233340452,
   "income_shift": 0.0141811595932263,
   "stale_steps": 156
  },
  "side_step.side": {
   "income": 0.7678028420396056,
   "income_dup": 0.7552344550348764,
   "income_shift": 0.01636928950580896,
   "liftoff_shift": 0.007006456930897187,
   "liftoffs": 7279,
   "liftoffs_dup": 7330
  },
  "side_step.side.f32_replica": {
   "gated": false,
   "income": 0.7678028420396056,
   "income_dup": 0.756296115958822,
   "income_shift": 0.01498656354307959,
   "stale_steps": 156
  }
 }
}
```

### R7: PASS

```
{
 "headline": "max |clip shift| pp E1.diag -0.51; max absorbed E1.move_turn 0.109",
 "rows": {
  "E1.back": {
   "absorbed": 0.02910737301933497,
   "clip_base": 0.06867979409457106,
   "clip_with": 0.06563713952197792,
   "shift_pp": -0.30426545725931453
  },
  "E1.diag": {
   "absorbed": 0.0808294028804375,
   "clip_base": 0.1785051629557922,
   "clip_with": 0.17338254275572765,
   "shift_pp": -0.5122620200064543
  },
  "E1.fwd": {
   "absorbed": 0.028034226158347164,
   "clip_base": 0.08156170611044257,
   "clip_with": 0.0784291253605333,
   "shift_pp": -0.31325807499092695
  },
  "E1.move_turn": {
   "absorbed": 0.10863369932992817,
   "clip_base": 0.2110797533186997,
   "clip_with": 0.20894742343472353,
   "shift_pp": -0.21323298839761584
  },
  "E1.side": {
   "absorbed": 0.08877697284594877,
   "clip_base": 0.14244793825352528,
   "clip_with": 0.14133348069962146,
   "shift_pp": -0.11144575539038248
  },
  "E1.turn": {
   "absorbed": 0.04873673658414335,
   "clip_base": 0.07835611308743681,
   "clip_with": 0.07612806590526118,
   "shift_pp": -0.22280471821756326
  },
  "W.back": {
   "absorbed": 0.006288947607184792,
   "clip_base": 0.03195599381852969,
   "clip_with": 0.030907351534329237,
   "shift_pp": -0.10486422842004538
  },
  "W.diag": {
   "absorbed": 0.044616336758483865,
   "clip_base": 0.14493927125506073,
   "clip_with": 0.14149326805385556,
   "shift_pp": -0.34460032012051667
  },
  "W.fwd": {
   "absorbed": 0.008465191866182442,
   "clip_base": 0.04606479769213699,
   "clip_with": 0.044751830756712775,
   "shift_pp": -0.1312966935424216
  },
  "W.move_turn": {
   "absorbed": 0.10236649230346861,
   "clip_base": 0.24733760008291658,
   "clip_with": 0.244824190915451,
   "shift_pp": -0.2513409167465591
  },
  "W.side": {
   "absorbed": 0.05516699879252629,
   "clip_base": 0.10306193204627569,
   "clip_with": 0.10193058647494617,
   "shift_pp": -0.11313455713295173
  },
  "W.turn": {
   "absorbed": 0.02256835291878101,
   "clip_base": 0.05494758138679419,
   "clip_with": 0.05319109803200294,
   "shift_pp": -0.1756483354791248
  },
  "X0x_s2.back": {
   "absorbed": 0.04357864454866989,
   "clip_base": 0.11133332000079996,
   "clip_with": 0.10863348199108054,
   "shift_pp": -0.2699838009719416
  },
  "X0x_s2.diag": {
   "absorbed": 0.0732801331055799,
   "clip_base": 0.18536552223933328,
   "clip_with": 0.18088142041851618,
   "shift_pp": -0.4484101820817099
  },
  "X0x_s2.fwd": {
   "absorbed": 0.028897695026149806,
   "clip_base": 0.09380617056820825,
   "clip_with": 0.091057116877075,
   "shift_pp": -0.2749053691133249
  },
  "X0x_s2.move_turn": {
   "absorbed": 0.06548361876525544,
   "clip_base": 0.1504983084941026,
   "clip_with": 0.14816169374092023,
   "shift_pp": -0.23366147531823644
  },
  "X0x_s2.side": {
   "absorbed": 0.06488308858944081,
   "clip_base": 0.13003817222409342,
   "clip_with": 0.12895288349986903,
   "shift_pp": -0.10852887242243925
  },
  "X0x_s2.turn": {
   "absorbed": 0.038716613260557975,
   "clip_base": 0.060974497427852166,
   "clip_with": 0.05956072822795432,
   "shift_pp": -0.14137691998978436
  },
  "X0x_s3.back": {
   "absorbed": 0.03171676631450138,
   "clip_base": 0.08031829170024174,
   "clip_with": 0.07782030620467364,
   "shift_pp": -0.24979854955680947
  },
  "X0x_s3.diag": {
   "absorbed": 0.0712826117934624,
   "clip_base": 0.19599272164474096,
   "clip_with": 0.1910986551774622,
   "shift_pp": -0.48940664672787737
  },
  "X0x_s3.fwd": {
   "absorbed": 0.03362672359329666,
   "clip_base": 0.09132891778574467,
   "clip_with": 0.08857426424419193,
   "shift_pp": -0.2754653541552743
  },
  "X0x_s3.move_turn": {
   "absorbed": 0.05992718012523357,
   "clip_base": 0.14800373060284955,
   "clip_with": 0.14573163471841885,
   "shift_pp": -0.2272095884430697
  },
  "X0x_s3.side": {
   "absorbed": 0.07987194506955775,
   "clip_base": 0.14248474764281752,
   "clip_with": 0.1413939730079497,
   "shift_pp": -0.10907746348678138
  },
  "X0x_s3.turn": {
   "absorbed": 0.028711741924118295,
   "clip_base": 0.04759080800593032,
   "clip_with": 0.0457468495181616,
   "shift_pp": -0.1843958487768714
  }
 }
}
```

### R8: FAIL

```
{
 "headline": "max hand gap E1.move_turn 0.120",
 "rows": {
  "E1.move_turn": {
   "gap": 0.11999735659978872,
   "l": 0.1868163685856413,
   "r": 0.16566778523476053
  },
  "E1.side": {
   "gap": 0.026989807637801885,
   "l": 0.10844864466601487,
   "r": 0.10556061035317624
  },
  "E1.turn": {
   "gap": 0.053933252612280526,
   "l": 0.4476400778572993,
   "r": 0.4241313437075864
  },
  "W.move_turn": {
   "gap": 0.04553842431719527,
   "l": 0.2561506809259581,
   "r": 0.24474566573927103
  },
  "W.side": {
   "gap": 0.0003901696134603146,
   "l": 0.11279810308259494,
   "r": 0.11284212206230788
  },
  "W.turn": {
   "gap": 0.013469773686252122,
   "l": 0.5653428569834827,
   "r": 0.5730095315075723
  },
  "X0x_s2.move_turn": {
   "gap": 0.07699796242379955,
   "l": 0.25852754194866245,
   "r": 0.2393594018535701
  },
  "X0x_s2.side": {
   "gap": 0.07897733919018007,
   "l": 0.11999802249739086,
   "r": 0.11088092021226552
  },
  "X0x_s2.turn": {
   "gap": 0.02354778244448767,
   "l": 0.3393358122677511,
   "r": 0.34742161956623197
  },
  "X0x_s3.move_turn": {
   "gap": 0.03293413501914566,
   "l": 0.2249767785265443,
   "r": 0.217687397653321
  },
  "X0x_s3.side": {
   "gap": 0.08593110169662618,
   "l": 0.10624192818087312,
   "r": 0.11578127751469137
  },
  "X0x_s3.turn": {
   "gap": 0.0072513796121482415,
   "l": 0.46299970964251624,
   "r": 0.46636931343557153
  }
 }
}
```

### R9: FAIL

```
{
 "grid": {
  "c_max": [
   -0.25,
   0.25
  ],
  "drho": [
   -0.1,
   0.1
  ],
  "force_hold": [
   15.0,
   25.0
  ],
  "force_open": [
   40.0,
   60.0
  ],
  "min_loaded": [
   4,
   6
  ],
  "n_off": [
   1,
   3
  ],
  "power": [
   1.5,
   2.0
  ],
  "prog_lo": [
   -0.05,
   0.05
  ],
  "prog_tau": [
   3.0,
   5.0
  ],
  "prog_width": [
   -0.05,
   0.05
  ],
  "rho0": [
   -0.1,
   0.1
  ]
 },
 "headline": "failing neighbours: ['force_hold=15', 'force_hold=25', 'n_off=1', 'n_off=3', 'rho0=0.4', 'rho0=0.6', 'drho=0.2', 'drho=0.4', 'power=2', 'c_max=1.25', 'c_max=1.75', 'prog_lo=0.1', 'prog_lo=0.2', 'prog_width=0.1', 'prog_width=0.2', 'prog_tau=3', 'prog_tau=5', 'force_open=40', 'force_open=60', 'min_loaded=4', 'min_loaded=6'] of ['force_hold=15', 'force_hold=25', 'n_off=1', 'n_off=3', 'rho0=0.4', 'rho0=0.6', 'drho=0.2', 'drho=0.4', 'power=2', 'c_max=1.25', 'c_max=1.75', 'prog_lo=0.1', 'prog_lo=0.2', 'prog_width=0.1', 'prog_width=0.2', 'prog_tau=3', 'prog_tau=5', 'force_open=40', 'force_open=60', 'min_loaded=4', 'min_loaded=6']",
 "rows": {
  "c_max=1.25": {
   "R1": {
    "kept": {
     "E1.back": 0.8435167670249939,
     "E1.fwd": 0.8506330847740173,
     "E1.turn": 0.7133904099464417,
     "W.back": 0.8417195677757263,
     "W.fwd": 0.8659331202507019,
     "W.turn": 0.7714568972587585,
     "X0x_s2.back": 0.8532804846763611,
     "X0x_s2.fwd": 0.876201331615448,
     "X0x_s2.turn": 0.6412193179130554,
     "X0x_s3.back": 0.8377866744995117,
     "X0x_s3.fwd": 0.851988673210144,
     "X0x_s3.turn": 0.7064958214759827,
     "side_slow.side": 0.870330810546875,
     "side_step.side": 0.7735280394554138
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.39817410707473755,
     "W": 0.38352692127227783,
     "X0x_s2": 0.43442001938819885,
     "X0x_s3": 0.4163912236690521
    },
    "slide_kept": 0.03747076786097115
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "c_max=1.75": {
   "R1": {
    "kept": {
     "E1.back": 0.8415428400039673,
     "E1.fwd": 0.8482967615127563,
     "E1.turn": 0.7133184671401978,
     "W.back": 0.8417195677757263,
     "W.fwd": 0.8638823628425598,
     "W.turn": 0.7714568972587585,
     "X0x_s2.back": 0.8520767688751221,
     "X0x_s2.fwd": 0.8751816749572754,
     "X0x_s2.turn": 0.6411575675010681,
     "X0x_s3.back": 0.8368474245071411,
     "X0x_s3.fwd": 0.8503902554512024,
     "X0x_s3.turn": 0.7065048813819885,
     "side_slow.side": 0.8696284890174866,
     "side_step.side": 0.7734588384628296
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.39817410707473755,
     "W": 0.3835206925868988,
     "X0x_s2": 0.43442001938819885,
     "X0x_s3": 0.416259765625
    },
    "slide_kept": 0.03741226768725815
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "drho=0.2": {
   "R1": {
    "kept": {
     "E1.back": 0.8292954564094543,
     "E1.fwd": 0.8359543085098267,
     "E1.turn": 0.7044017314910889,
     "W.back": 0.8299577236175537,
     "W.fwd": 0.8526384234428406,
     "W.turn": 0.7568053007125854,
     "X0x_s2.back": 0.8409333825111389,
     "X0x_s2.fwd": 0.8631646633148193,
     "X0x_s2.turn": 0.6316152215003967,
     "X0x_s3.back": 0.8234662413597107,
     "X0x_s3.fwd": 0.8382286429405212,
     "X0x_s3.turn": 0.6980467438697815,
     "side_slow.side": 0.8497278094291687,
     "side_step.side": 0.7529549598693848
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": true,
     "slide_kept<=0.25": true
    },
    "pass": true,
    "side_kept": {
     "E1": 0.3616662919521332,
     "W": 0.34195980429649353,
     "X0x_s2": 0.39832985401153564,
     "X0x_s3": 0.380965918302536
    },
    "slide_kept": 0.005792440731830523
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "drho=0.4": {
   "R1": {
    "kept": {
     "E1.back": 0.8504602909088135,
     "E1.fwd": 0.8588846325874329,
     "E1.turn": 0.7199252843856812,
     "W.back": 0.8499298095703125,
     "W.fwd": 0.8732130527496338,
     "W.turn": 0.7815947532653809,
     "X0x_s2.back": 0.8613321185112,
     "X0x_s2.fwd": 0.8838445544242859,
     "X0x_s2.turn": 0.6475512385368347,
     "X0x_s3.back": 0.8465150594711304,
     "X0x_s3.fwd": 0.8603501915931702,
     "X0x_s3.turn": 0.712211549282074,
     "side_slow.side": 0.8833354711532593,
     "side_step.side": 0.7878739833831787
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.43043166399002075,
     "W": 0.42076563835144043,
     "X0x_s2": 0.4688348174095154,
     "X0x_s3": 0.44825658202171326
    },
    "slide_kept": 0.08442239228936287
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "force_hold=15": {
   "R1": {
    "kept": {
     "E1.back": 0.8408282995223999,
     "E1.fwd": 0.8473556637763977,
     "E1.turn": 0.7109430432319641,
     "W.back": 0.8404059410095215,
     "W.fwd": 0.8631572723388672,
     "W.turn": 0.7695155739784241,
     "X0x_s2.back": 0.8511379957199097,
     "X0x_s2.fwd": 0.87507164478302,
     "X0x_s2.turn": 0.6391478776931763,
     "X0x_s3.back": 0.8357512950897217,
     "X0x_s3.fwd": 0.8504160642623901,
     "X0x_s3.turn": 0.7053003311157227,
     "side_slow.side": 0.8681977987289429,
     "side_step.side": 0.7709177136421204
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.3953637480735779,
     "W": 0.3794052302837372,
     "X0x_s2": 0.4277430474758148,
     "X0x_s3": 0.4095805585384369
    },
    "slide_kept": 0.03737721045219039
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "force_hold=25": {
   "R1": {
    "kept": {
     "E1.back": 0.8426355719566345,
     "E1.fwd": 0.8501933813095093,
     "E1.turn": 0.7154578566551208,
     "W.back": 0.8428012728691101,
     "W.fwd": 0.8653568029403687,
     "W.turn": 0.7729043364524841,
     "X0x_s2.back": 0.8538661599159241,
     "X0x_s2.fwd": 0.875980794429779,
     "X0x_s2.turn": 0.6426318883895874,
     "X0x_s3.back": 0.8380730748176575,
     "X0x_s3.fwd": 0.8520926237106323,
     "X0x_s3.turn": 0.7083179354667664,
     "side_slow.side": 0.8708433508872986,
     "side_step.side": 0.7764991521835327
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.4024987518787384,
     "W": 0.38836371898651123,
     "X0x_s2": 0.4422163963317871,
     "X0x_s3": 0.41914573311805725
    },
    "slide_kept": 0.03730597791940696
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "force_open=40": {
   "R1": {
    "kept": {
     "E1.back": 0.8416956663131714,
     "E1.fwd": 0.8486944437026978,
     "E1.turn": 0.713344156742096,
     "W.back": 0.8417139649391174,
     "W.fwd": 0.8645267486572266,
     "W.turn": 0.7714514136314392,
     "X0x_s2.back": 0.8526716828346252,
     "X0x_s2.fwd": 0.8754605650901794,
     "X0x_s2.turn": 0.6411325335502625,
     "X0x_s3.back": 0.8369816541671753,
     "X0x_s3.fwd": 0.8511130213737488,
     "X0x_s3.turn": 0.7064999938011169,
     "side_slow.side": 0.8696284294128418,
     "side_step.side": 0.7734391093254089
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.39815229177474976,
     "W": 0.38351455330848694,
     "X0x_s2": 0.4343971610069275,
     "X0x_s3": 0.41625767946243286
    },
    "slide_kept": 0.03741715602068443
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "force_open=60": {
   "R1": {
    "kept": {
     "E1.back": 0.8417174816131592,
     "E1.fwd": 0.848741888999939,
     "E1.turn": 0.7132958769798279,
     "W.back": 0.8417367935180664,
     "W.fwd": 0.8645264506340027,
     "W.turn": 0.7715002298355103,
     "X0x_s2.back": 0.852698564529419,
     "X0x_s2.fwd": 0.8754424452781677,
     "X0x_s2.turn": 0.6411779522895813,
     "X0x_s3.back": 0.8370469808578491,
     "X0x_s3.fwd": 0.8511795401573181,
     "X0x_s3.turn": 0.7065138816833496,
     "side_slow.side": 0.8696511387825012,
     "side_step.side": 0.7734973430633545
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.3981846272945404,
     "W": 0.3835623860359192,
     "X0x_s2": 0.4345596730709076,
     "X0x_s3": 0.41627761721611023
    },
    "slide_kept": 0.03740782880496458
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "min_loaded=4": {
   "R1": {
    "kept": {
     "E1.back": 0.8420781493186951,
     "E1.fwd": 0.8490672707557678,
     "E1.turn": 0.7136302590370178,
     "W.back": 0.8418427109718323,
     "W.fwd": 0.8647574782371521,
     "W.turn": 0.7717905044555664,
     "X0x_s2.back": 0.8530150055885315,
     "X0x_s2.fwd": 0.8757171630859375,
     "X0x_s2.turn": 0.6415039300918579,
     "X0x_s3.back": 0.8373154997825623,
     "X0x_s3.fwd": 0.8513997793197632,
     "X0x_s3.turn": 0.7068813443183899,
     "side_slow.side": 0.8700569272041321,
     "side_step.side": 0.7739837169647217
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.3992508053779602,
     "W": 0.38441136479377747,
     "X0x_s2": 0.43508976697921753,
     "X0x_s3": 0.41711390018463135
    },
    "slide_kept": 0.037465681753539815
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "min_loaded=6": {
   "R1": {
    "kept": {
     "E1.back": 0.8415109515190125,
     "E1.fwd": 0.8483980894088745,
     "E1.turn": 0.7125896215438843,
     "W.back": 0.841448962688446,
     "W.fwd": 0.864123523235321,
     "W.turn": 0.7711854577064514,
     "X0x_s2.back": 0.8523164987564087,
     "X0x_s2.fwd": 0.8751627802848816,
     "X0x_s2.turn": 0.6407527327537537,
     "X0x_s3.back": 0.836647629737854,
     "X0x_s3.fwd": 0.8508004546165466,
     "X0x_s3.turn": 0.7061778903007507,
     "side_slow.side": 0.8690862059593201,
     "side_step.side": 0.7729062438011169
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.3969196081161499,
     "W": 0.38286370038986206,
     "X0x_s2": 0.43326881527900696,
     "X0x_s3": 0.41512054204940796
    },
    "slide_kept": 0.037388550426503214
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "n_off=1": {
   "R1": {
    "kept": {
     "E1.back": 0.8813433647155762,
     "E1.fwd": 0.892087996006012,
     "E1.turn": 0.7519592642784119,
     "W.back": 0.8775861263275146,
     "W.fwd": 0.8929787874221802,
     "W.turn": 0.8202228546142578,
     "X0x_s2.back": 0.888835072517395,
     "X0x_s2.fwd": 0.9035051465034485,
     "X0x_s2.turn": 0.68478924036026,
     "X0x_s3.back": 0.8730376362800598,
     "X0x_s3.fwd": 0.887169599533081,
     "X0x_s3.turn": 0.7468395829200745,
     "side_slow.side": 0.896106481552124,
     "side_step.side": 0.8090685606002808
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": false,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.5225545167922974,
     "W": 0.5280258059501648,
     "X0x_s2": 0.5547643899917603,
     "X0x_s3": 0.5454174280166626
    },
    "slide_kept": 0.04197055376218242
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "n_off=3": {
   "R1": {
    "kept": {
     "E1.back": 0.8103166818618774,
     "E1.fwd": 0.8219258189201355,
     "E1.turn": 0.6830852031707764,
     "W.back": 0.8136651515960693,
     "W.fwd": 0.8412292003631592,
     "W.turn": 0.7318357229232788,
     "X0x_s2.back": 0.8277243375778198,
     "X0x_s2.fwd": 0.8525508642196655,
     "X0x_s2.turn": 0.6082042455673218,
     "X0x_s3.back": 0.8126088976860046,
     "X0x_s3.fwd": 0.8256247043609619,
     "X0x_s3.turn": 0.6781333684921265,
     "side_slow.side": 0.847476065158844,
     "side_step.side": 0.7441140413284302
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": true,
     "slide_kept<=0.25": true
    },
    "pass": true,
    "side_kept": {
     "E1": 0.3357945382595062,
     "W": 0.3180728256702423,
     "X0x_s2": 0.3773285448551178,
     "X0x_s3": 0.36204415559768677
    },
    "slide_kept": 0.03617907997415228
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "power=2": {
   "R1": {
    "kept": {
     "E1.back": 0.8304909467697144,
     "E1.fwd": 0.8439816236495972,
     "E1.turn": 0.7163987159729004,
     "W.back": 0.8252260684967041,
     "W.fwd": 0.8565168380737305,
     "W.turn": 0.7610960602760315,
     "X0x_s2.back": 0.8454622030258179,
     "X0x_s2.fwd": 0.8713967204093933,
     "X0x_s2.turn": 0.6436750292778015,
     "X0x_s3.back": 0.8271595239639282,
     "X0x_s3.fwd": 0.844698965549469,
     "X0x_s3.turn": 0.7065458297729492,
     "side_slow.side": 0.8723768591880798,
     "side_step.side": 0.7876123189926147
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.4214610159397125,
     "W": 0.4032896161079407,
     "X0x_s2": 0.46463876962661743,
     "X0x_s3": 0.4488784968852997
    },
    "slide_kept": 0.03874844189407423
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "prog_lo=0.1": {
   "R1": {
    "kept": {
     "E1.back": 0.8417110443115234,
     "E1.fwd": 0.8487941026687622,
     "E1.turn": 0.7567081451416016,
     "W.back": 0.8418180346488953,
     "W.fwd": 0.8645245432853699,
     "W.turn": 0.789145827293396,
     "X0x_s2.back": 0.8529409170150757,
     "X0x_s2.fwd": 0.8757458925247192,
     "X0x_s2.turn": 0.7034571170806885,
     "X0x_s3.back": 0.8371450304985046,
     "X0x_s3.fwd": 0.8513604998588562,
     "X0x_s3.turn": 0.7485287189483643,
     "side_slow.side": 0.870219349861145,
     "side_step.side": 0.7789273858070374
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.39839357137680054,
     "W": 0.3835315704345703,
     "X0x_s2": 0.43465375900268555,
     "X0x_s3": 0.41662630438804626
    },
    "slide_kept": 0.03863286700096047
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "prog_lo=0.2": {
   "R1": {
    "kept": {
     "E1.back": 0.8416920900344849,
     "E1.fwd": 0.8483981490135193,
     "E1.turn": 0.642838716506958,
     "W.back": 0.8415340781211853,
     "W.fwd": 0.8645190000534058,
     "W.turn": 0.7263434529304504,
     "X0x_s2.back": 0.8521217107772827,
     "X0x_s2.fwd": 0.8749386072158813,
     "X0x_s2.turn": 0.5437526106834412,
     "X0x_s3.back": 0.8367919325828552,
     "X0x_s3.fwd": 0.8504233956336975,
     "X0x_s3.turn": 0.6357879042625427,
     "side_slow.side": 0.8686307668685913,
     "side_step.side": 0.7669099569320679
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.3964018225669861,
     "W": 0.38330990076065063,
     "X0x_s2": 0.43236738443374634,
     "X0x_s3": 0.4139612913131714
    },
    "slide_kept": 0.03586982415870456
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "prog_tau=3": {
   "R1": {
    "kept": {
     "E1.back": 0.841701328754425,
     "E1.fwd": 0.8487018346786499,
     "E1.turn": 0.7102012038230896,
     "W.back": 0.8417015075683594,
     "W.fwd": 0.8645211458206177,
     "W.turn": 0.7665109038352966,
     "X0x_s2.back": 0.852519690990448,
     "X0x_s2.fwd": 0.8753801584243774,
     "X0x_s2.turn": 0.6391661763191223,
     "X0x_s3.back": 0.8369759321212769,
     "X0x_s3.fwd": 0.851161539554596,
     "X0x_s3.turn": 0.7044317722320557,
     "side_slow.side": 0.8696709871292114,
     "side_step.side": 0.772671103477478
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.39779213070869446,
     "W": 0.38349488377571106,
     "X0x_s2": 0.43392041325569153,
     "X0x_s3": 0.4155426323413849
    },
    "slide_kept": 0.037291964599255795
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "prog_tau=5": {
   "R1": {
    "kept": {
     "E1.back": 0.841708242893219,
     "E1.fwd": 0.8487553596496582,
     "E1.turn": 0.7170923352241516,
     "W.back": 0.8417433500289917,
     "W.fwd": 0.8645241260528564,
     "W.turn": 0.7744653224945068,
     "X0x_s2.back": 0.8527625799179077,
     "X0x_s2.fwd": 0.875521183013916,
     "X0x_s2.turn": 0.6448546648025513,
     "X0x_s3.back": 0.8370769619941711,
     "X0x_s3.fwd": 0.8511562943458557,
     "X0x_s3.turn": 0.7094982862472534,
     "side_slow.side": 0.8696486353874207,
     "side_step.side": 0.7744702100753784
    },
    "pass": false
   },
   "R2": {
    "clauses": {
     "side<=0.2xsideways": true,
     "side_kept<=0.40": false,
     "slide_kept<=0.25": true
    },
    "pass": false,
    "side_kept": {
     "E1": 0.39828330278396606,
     "W": 0.3835262954235077,
     "X0x_s2": 0.43455174565315247,
     "X0x_s3": 0.41647499799728394
    },
    "slide_kept": 0.03758260695955182
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": false,
     "v4<=healthy": null
    },
    "pass": false
   }
  },
  "prog_width=0.1": {
   "R1": {
    "kept": {
     "E1.back": 0.8417107462882996,
     "E1.fwd": 0.8487746715545654,
     "E1.turn": 0.7412703037261963,
     "W.back": 0.8417954444885254,
     "W.fwd": 0.8645240664482117,
     "W.turn": 0.7855435609817505,
     "
```

### P1: FAIL

```
{
 "clauses": {
  "W side in [0.15,0.45]": true,
  "fwd/back/stand/turn==0": true,
  "sideways>=0.85": false
 },
 "headline": "sideways side {'side_slow': 0.9420073938338351, 'side_step': 0.752277317627347}; healthy side W 0.373, X0x_s3 0.345, X0x_s2 0.393, E1 0.358; max |zero row| E1.back 4.22e-08",
 "healthy_side": {
  "E1": 0.3579436203781097,
  "W": 0.3732090087525319,
  "X0x_s2": 0.3930104385124148,
  "X0x_s3": 0.34452390750116996
 },
 "sideways_side": {
  "side_slow": 0.9420073938338351,
  "side_step": 0.752277317627347
 },
 "zero_rows": {
  "E1.back": 4.2222159817444624e-08,
  "E1.fwd": 0.0,
  "E1.stand": 0.0,
  "E1.turn": 0.0,
  "W.back": 4.195428315596144e-08,
  "W.fwd": 0.0,
  "W.stand": 0.0,
  "W.turn": 0.0,
  "X0x_s2.back": 3.730605284878618e-08,
  "X0x_s2.fwd": 0.0,
  "X0x_s2.stand": 0.0,
  "X0x_s2.turn": 0.0,
  "X0x_s3.back": 4.0460887207586985e-08,
  "X0x_s3.fwd": 0.0,
  "X0x_s3.stand": 0.0,
  "X0x_s3.turn": 0.0
 }
}
```

### P2: PASS

```
{
 "centring_prize_per_w": 0.47393334697805917,
 "clauses": {
  "diag<=10%": true
 },
 "headline": "w_P 0.1549 band [0.12390134887461453, 0.1935958576165852]; max healthy share X0x_s2.side 0.084",
 "reported": {
  "healthy share>15% (design-panel line; not gated)": []
 },
 "rule": "w_P = w_S * (stroke prize) / (centring prize); prize = sideways-head side minus the primary omni side, per second at w = 1; the parity band is [0.8, 1.25] x",
 "shares_at_w_P": {
  "E1.diag": 0.08182483634797745,
  "E1.move_turn": 0.0,
  "E1.side": 0.07525665129866574,
  "W.diag": 0.07365166957736494,
  "W.move_turn": 0.0,
  "W.side": 0.07728441399918363,
  "X0x_s2.diag": 0.07413630066557632,
  "X0x_s2.move_turn": 0.0,
  "X0x_s2.side": 0.08422146256899792,
  "X0x_s3.diag": 0.0757600572490057,
  "X0x_s3.move_turn": 0.0,
  "X0x_s3.side": 0.07320545331944377
 },
 "stroke_prize_per_w": 0.8488851495042933,
 "w_P": 0.15487668609326816,
 "w_P_band": [
  0.12390134887461453,
  0.1935958576165852
 ]
}
```

### P3: PASS

```
{
 "clauses": {
  "creeper side<=0.10x sideways": true,
  "no_flip": true,
  "x08 side vs statue>=1.10": true
 },
 "creeper": {
  "creeper": 0.022848162277583,
  "ratio": 0.026970865195234308,
  "sideways": 0.847142355730591
 },
 "headline": "no-flip PASS; x08/statue side totals 0.875/-35.149 ratio --; creeper/sideways 0.027",
 "rg1": {
  "pass": true,
  "rows": [
   {
    "base": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.224839629235345,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.1578, free -36.1578); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.224839629235345,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 1.031751053937961,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1490, free -35.1490); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 1.0896275864911966,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.4144079659795135,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.5869, free -35.5869); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.414407972473724,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.8710372823320194,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.3432, free -35.3432); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.9283971265237132,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.2190878614605154,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.2200, free -35.2200); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.2190878614605154,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.3117100470691925,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.2581, free -36.2581); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.3117100470691925,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.224839629235345,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 1.6075158862538983,
    "ratio_with": 1.6075158862538983,
    "with": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.224839629235345,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3553215982913456,
     "healthy_total": 1.031751053937961,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 2.9037104946600456,
    "ratio_with": 3.03612125004394,
    "with": {
     "degenerate_total": 0.35888803402546166,
     "healthy_total": 1.0896275864911966,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7151299339158457,
     "healthy_total": 1.4144079659795135,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 1.9778335361164685,
    "ratio_with": 1.9778335381922296,
    "with": {
     "degenerate_total": 0.7151299364488048,
     "healthy_total": 1.414407972473724,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.5430119192276095,
     "healthy_total": 0.8710372823320194,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 1.6040850145075993,
    "ratio_with": 1.6407852936136684,
    "with": {
     "degenerate_total": 0.5658248706502054,
     "healthy_total": 0.9283971265237132,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.2190878614605154,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "W",
    "note": null,
    "pass": true,
    "ratio_base": 3.07264528758921,
    "ratio_with": 3.07264528758921,
    "with": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.2190878614605154,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.3117100470691925,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": null,
    "healthy": "W",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -0.0228, free -0.0228); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.3117100470691925,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.085220068985984,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.1578, free -36.1578); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.085220068985984,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.8738616820176787,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1490, free -35.1490); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.9273011727186514,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.1796833603997723,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.5869, free -35.5869); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.1796833666636244,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.714856351412926,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.3432, free -35.3432); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.7729402695904235,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.291134827822507,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.2200, free -35.2200); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.291134827822507,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7986707060099291,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.2581, free -36.2581); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7986707060099291,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.085220068985984,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.4242750310631278,
    "ratio_with": 1.4242750310631278,
    "with": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.085220068985984,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3553215982913456,
     "healthy_total": 0.8738616820176787,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 2.459354247588284,
    "ratio_with": 2.5838174717546116,
    "with": {
     "degenerate_total": 0.35888803402546166,
     "healthy_total": 0.9273011727186514,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7151299339158457,
     "healthy_total": 1.1796833603997723,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.6496070216780967,
    "ratio_with": 1.6496070245943009,
    "with": {
     "degenerate_total": 0.7151299364488048,
     "healthy_total": 1.1796833666636244,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.5430119192276095,
     "healthy_total": 0.714856351412926,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.316465304168187,
    "ratio_with": 1.3660415257148664,
    "with": {
     "degenerate_total": 0.5658248706502054,
     "healthy_total": 0.7729402695904235,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.291134827822507,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s3",
    "note": null,
    "pass": true,
    "ratio_base": 3.2542357854324546,
    "ratio_with": 3.2542357854324546,
    "with": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.291134827822507,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.7986707060099291,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": null,
    "healthy": "X0x_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -0.0228, free -0.0228); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.7986707060099291,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.149183320272118,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.1578, free -36.1578); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.15776614038987,
     "healthy_total": 1.149183320272118,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.8994346630894414,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1490, free -35.1490); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.149021965292434,
     "healthy_total": 0.9603609995148308,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.008182731295575,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.5869, free -35.5869); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.58685489651426,
     "healthy_total": 1.008182737070984,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.6580813797634171,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.3432, free -35.3432); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.343206803697804,
     "healthy_total": 0.7157476035060244,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.16980319899595,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.2200, free -35.2200); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.219971876022754,
     "healthy_total": 1.16980319899595,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7478507935470017,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.2581, free -36.2581); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.258082231781096,
     "healthy_total": 0.7478507935470017,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.149183320272118,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.5082223006686197,
    "ratio_with": 1.5082223006686197,
    "with": {
     "degenerate_total": 0.7619455830633628,
     "healthy_total": 1.149183320272118,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3553215982913456,
     "healthy_total": 0.8994346630894414,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.531325614357816,
    "ratio_with": 2.675934855623236,
    "with": {
     "degenerate_total": 0.35888803402546166,
     "healthy_total": 0.9603609995148308,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7151299339158457,
     "healthy_total": 1.008182731295575,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.4097895829574025,
    "ratio_with": 1.4097895860400165,
    "with": {
     "degenerate_total": 0.7151299364488048,
     "healthy_total": 1.008182737070984,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.5430119192276095,
     "healthy_total": 0.6580813797634171,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.2119096403988416,
    "ratio_with": 1.2649631372398646,
    "with": {
     "degenerate_total": 0.5658248706502054,
     "healthy_total": 0.7157476035060244,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.16980319899595,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "X0x_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.9484259506082418,
    "ratio_with": 2.9484259506082418,
    "with": {
     "degenerate_total": 0.3967551563418532,
     "healthy_total": 1.16980319899595,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -0.022799553732817168,
     "healthy_total": 0.7478507935470017,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": null,
    "healthy": "X0x_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -0.0228, free -0.0228); reported, not divided (audit item 11)
```

### P4: PASS

```
{
 "headline": "max |clip shift| pp E1.diag -1.73; max hand gap X0x_s3.side.hands 0.025",
 "rows": {
  "E1.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E1.diag": {
   "absorbed": 0.14885045699067057,
   "shift_pp": -1.7263633430138758
  },
  "E1.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E1.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E1.side": {
   "absorbed": 0.12173513789131729,
   "shift_pp": -1.1789303049561135
  },
  "E1.side.hands": {
   "gap": 0.01756636879285498
  },
  "E1.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "W.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "W.diag": {
   "absorbed": 0.11024424600042149,
   "shift_pp": -1.2748328782600515
  },
  "W.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "W.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "W.side": {
   "absorbed": 0.08527944551286137,
   "shift_pp": -0.9671179883945848
  },
  "W.side.hands": {
   "gap": 0.014030194323474764
  },
  "W.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s2.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s2.diag": {
   "absorbed": 0.1419889596964513,
   "shift_pp": -1.3520246399130353
  },
  "X0x_s2.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s2.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s2.side": {
   "absorbed": 0.10896849650800178,
   "shift_pp": -1.2069159088357484
  },
  "X0x_s2.side.hands": {
   "gap": 0.017462966609989628
  },
  "X0x_s2.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s3.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s3.diag": {
   "absorbed": 0.15505853821837623,
   "shift_pp": -1.4222073494656262
  },
  "X0x_s3.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s3.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "X0x_s3.side": {
   "absorbed": 0.1146451712265315,
   "shift_pp": -1.0491772970974282
  },
  "X0x_s3.side.hands": {
   "gap": 0.02450092496811503
  },
  "X0x_s3.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  }
 }
}
```
