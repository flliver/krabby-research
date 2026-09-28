# REPLAY_GATE -- stroke/posture plan step 7

Economy: `/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_09-10-29_seed3_X08_t3_A15pB/params/env.yaml` (clock [1.0, 1.0], tibia weight -0.28). Parameters: `{"rho0": 0.5, "drho": 0.3, "power": 1.5, "c_max": 1.5, "prog_tau": 4.0, "prog_lo": 0.15, "prog_width": 0.15, "force_open": 50.0, "force_hold": 20.0, "n_off": 2, "min_loaded": 5, "sigma": 0.2, "tau": 2.0, "min_cmd": 0.12, "phi_lo": 0.1, "phi_sat": 0.25}`.

**w_S = 0.1362**, **w_P = 0.2673**

Per-category stroke table: `KRABBY_OMNI_STROKE_W=0.1362210975498265` `KRABBY_OMNI_STROKE_CAT_W=fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0` -- effective w_S x m_c: fwd 0.1362, back 0.1362, side 0.5653, diag 0.1796, turn 0.2377, move_turn 0.1362. P1-P4 still price the stroke prize at w_S (unscaled).

| Id | Verdict | Clauses | Deciding numbers |
|---|---|---|---|
| R1 | **FAIL** |  | min kept Bf_s2.turn 0.423 (>= 0.8) |
| R2 | **FAIL** | side_kept<=0.40: FAIL; slide_kept<=0.25: PASS; side<=0.2xsideways: PASS | max side kept Bf_s3 0.527; slide kept 0.036; max side/sideways Bf_s2/side_step 0.177 |
| R3 | **PASS** | short<=2%: PASS; spearman<=+0.2: PASS; v4<=healthy: N/A | max short share Bf_s3.side 0.0106; max Spearman Bf_s3.side 0.071 |
| R4 | **PASS** | fwd>=5%: PASS; stand==0: PASS; turn>0: PASS; move_turn>0: PASS; dose targets reached: PASS | w_S 0.1362; table fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0; tie bound by {'fwd,back': 'fwd'}; shares E3 fwd 0.100, side 0.099, back 0.078, diag 0.100, turn 0.091, move_turn 0.015; Bf_s3 fwd 0.079, side 0.104, back 0.068, diag 0.090, turn 0.076, move_turn 0.010; Bf_s2 fwd 0.105, side 0.123, back 0.079, diag 0.098, turn 0.055, move_turn 0.011 |
| R5 | **FAIL** | no_flip: FAIL; term_alone<=0.5x (translation): PASS | no-flip fails: ['Bf_s3>E7_creeper move_turn']; max term-alone (translation) E7_creeper/Bf_s3.fwd 0.315; max term-alone (rotational, reported) E7_creeper/Bf_s3.move_turn 3.185 |
| R6 | **FAIL** |  | max income shift Bf_s2.side 0.0329; max liftoff shift Bf_s2.side 0.0485 |
| R7 | **PASS** |  | max |clip shift| pp Bf_s3.diag -0.41; max absorbed Bf_s3.move_turn 0.107 |
| R8 | **FAIL** |  | max hand gap E3.turn 0.214 |
| R9 | **FAIL** |  | failing neighbours: ['force_hold=15', 'force_hold=25', 'n_off=1', 'n_off=3', 'rho0=0.4', 'rho0=0.6', 'drho=0.2', 'drho=0.4', 'power=2', 'c_max=1.25', 'c_max=1.75', 'prog_lo=0.1', 'prog_lo=0.2', 'prog_width=0.1', 'prog_width=0.2', 'prog_tau=3', 'prog_tau=5', 'force_open=40', 'force_open=60', 'min_loaded=4', 'min_loaded=6'] of ['force_hold=15', 'force_hold=25', 'n_off=1', 'n_off=3', 'rho0=0.4', 'rho0=0.6', 'drho=0.2', 'drho=0.4', 'power=2', 'c_max=1.25', 'c_max=1.75', 'prog_lo=0.1', 'prog_lo=0.2', 'prog_width=0.1', 'prog_width=0.2', 'prog_tau=3', 'prog_tau=5', 'force_open=40', 'force_open=60', 'min_loaded=4', 'min_loaded=6'] |
| P1 | **FAIL** | sideways>=0.85: FAIL; E3 side in [0.15,0.45]: PASS; fwd/back/stand/turn==0: PASS | sideways side {'side_slow': 0.9420073938338351, 'side_step': 0.752277317627347}; healthy side E3 0.417, Bf_s3 0.526, Bf_s2 0.594; max |zero row| Bf_s2.back 3.69e-08 |
| P2 | **FAIL** | diag<=10%: FAIL | w_P 0.2673 band [0.21384755433835073, 0.334136803653673]; max healthy share Bf_s2.side 0.233 |
| P3 | **PASS** | no_flip: PASS; x08 side vs statue>=1.10: PASS; creeper side<=0.10x sideways: PASS | no-flip PASS; x08/statue side totals 1.403/-34.968 ratio --; creeper/sideways 0.027 |
| P4 | **PASS** |  | max |clip shift| pp Bf_s2.side -2.58; max hand gap Bf_s2.side.hands 0.033 |

Dropped inputs (recorded, not silently skipped):
- W, X0x_s3, X0x_s2, E1 (p15 healthy): not the t3 setting's heads; the t3 healthy set is E3 (X08:t3) + the formation B_f pair (all clock-held (1.0, 1.0) by stamp_clock)
- Z_t3, Z_c6, A4500 (p15 posture): clock-FREE heads (stamp_clock (0.001, 0.001)); pricing them in X08:t3's clock-held economy would mis-state their clock income, so they are dropped from the t3 R5 no-flip set
- forward-era degenerates: as in p15: not loadable by the tool (forward-task stamps, pre-2026-08-20 plant)

## Runs

| name | role | kind | N | notes |
|---|---|---|---|---|
| E3 | healthy | omni | 1000 |  |
| Bf_s3 | healthy | omni | 1000 |  |
| Bf_s2 | healthy | omni | 1000 |  |
| side_slow | sideways | side | 100 | sideways head: cmd rotated by travel_yaw 1.5708; cos(cmd, achieved velocity) = 1.000 |
| side_step | sideways | side | 100 | sideways head: cmd rotated by travel_yaw 1.5708; cos(cmd, achieved velocity) = 1.000 |
| E6_statue | degenerate | omni | 1000 |  |
| E7_creeper | degenerate | omni | 1000 |  |

## Numbers

### R1: FAIL

```
{
 "headline": "min kept Bf_s2.turn 0.423 (>= 0.8)",
 "kept": {
  "Bf_s2.back": 0.686125636100769,
  "Bf_s2.fwd": 0.7338358163833618,
  "Bf_s2.turn": 0.42334869503974915,
  "Bf_s3.back": 0.7819819450378418,
  "Bf_s3.fwd": 0.7714508175849915,
  "Bf_s3.turn": 0.6088981628417969,
  "E3.back": 0.7677779793739319,
  "E3.fwd": 0.7941877841949463,
  "E3.turn": 0.5985913276672363,
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
 "headline": "max side kept Bf_s3 0.527; slide kept 0.036; max side/sideways Bf_s2/side_step 0.177",
 "side_income": {
  "Bf_s2": 0.13591774875132318,
  "Bf_s3": 0.12857073200200972,
  "E3": 0.11853260197750105
 },
 "side_kept": {
  "Bf_s2": 0.4399391710758209,
  "Bf_s3": 0.526648223400116,
  "E3": 0.44106799364089966
 },
 "side_vs_sideways": {
  "Bf_s2/side_slow": 0.1176158129964864,
  "Bf_s2/side_step": 0.1770216796674896,
  "Bf_s3/side_slow": 0.11125810507380501,
  "Bf_s3/side_step": 0.16745279512182068,
  "E3/side_slow": 0.10257165437370443,
  "E3/side_step": 0.15437895705442933
 },
 "sideways_income": {
  "side_slow": 1.1556077817137014,
  "side_step": 0.7678028420396056
 },
 "slide_kept": 0.0356922437350623
}
```

### R3: PASS

```
{
 "clauses": {
  "short<=2%": true,
  "spearman<=+0.2": true,
  "v4<=healthy": null
 },
 "headline": "max short share Bf_s3.side 0.0106; max Spearman Bf_s3.side 0.071",
 "short_share": {
  "Bf_s2.back": 0.0013931798748672009,
  "Bf_s2.diag": 0.0017571134958416224,
  "Bf_s2.fwd": 0.0010135394986718893,
  "Bf_s2.move_turn": 0.0022702941205352545,
  "Bf_s2.side": 0.00666362838819623,
  "Bf_s2.turn": 0.0014652496902272105,
  "Bf_s3.back": 0.0031716073863208294,
  "Bf_s3.diag": 0.004084431566298008,
  "Bf_s3.fwd": 0.002985518891364336,
  "Bf_s3.move_turn": 0.0032754759304225445,
  "Bf_s3.side": 0.010607770644128323,
  "Bf_s3.turn": 0.0038208235055208206,
  "E3.back": 0.002156964736059308,
  "E3.diag": 0.002276832237839699,
  "E3.fwd": 0.0015103471232578158,
  "E3.move_turn": 0.0020086930599063635,
  "E3.side": 0.009071060456335545,
  "E3.turn": 0.0010729911737143993,
  "side_slow.side": 0.0002873463381547481,
  "side_step.side": 0.00042001117253676057
 },
 "spearman_switch": {
  "Bf_s2.back": {
   "n": 100,
   "rho": -0.08797518299337874
  },
  "Bf_s2.diag": {
   "n": 100,
   "rho": -0.35417405277536196
  },
  "Bf_s2.fwd": {
   "n": 100,
   "rho": -0.25843062072258416
  },
  "Bf_s2.move_turn": {
   "n": 200,
   "rho": -0.03763861180394828
  },
  "Bf_s2.side": {
   "n": 200,
   "rho": 0.005689433172441208
  },
  "Bf_s2.turn": {
   "n": 200,
   "rho": -0.07019989176525711
  },
  "Bf_s3.back": {
   "n": 100,
   "rho": -0.16980073397029646
  },
  "Bf_s3.diag": {
   "n": 96,
   "rho": -0.288823922598965
  },
  "Bf_s3.fwd": {
   "n": 97,
   "rho": -0.030278622550781703
  },
  "Bf_s3.move_turn": {
   "n": 199,
   "rho": 0.03833200918976266
  },
  "Bf_s3.side": {
   "n": 198,
   "rho": 0.07112261089049755
  },
  "Bf_s3.turn": {
   "n": 197,
   "rho": -0.04387271344074971
  },
  "E3.back": {
   "n": 100,
   "rho": -0.09502088832784758
  },
  "E3.diag": {
   "n": 99,
   "rho": -0.10862963662367069
  },
  "E3.fwd": {
   "n": 100,
   "rho": -0.34662914193229893
  },
  "E3.move_turn": {
   "n": 200,
   "rho": -0.07460891975295075
  },
  "E3.side": {
   "n": 199,
   "rho": 0.06313417700132817
  },
  "E3.turn": {
   "n": 200,
   "rho": -0.042391962334718154
  }
 },
 "v4": {}
}
```

### R4: PASS

```
{
 "clauses": {
  "dose targets reached": true,
  "fwd>=5%": true,
  "move_turn>0": true,
  "stand==0": true,
  "turn>0": true
 },
 "headline": "w_S 0.1362; table fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0; tie bound by {'fwd,back': 'fwd'}; shares E3 fwd 0.100, side 0.099, back 0.078, diag 0.100, turn 0.091, move_turn 0.015; Bf_s3 fwd 0.079, side 0.104, back 0.068, diag 0.090, turn 0.076, move_turn 0.010; Bf_s2 fwd 0.105, side 0.123, back 0.079, diag 0.098, turn 0.055, move_turn 0.011",
 "note": "diag and side are reported, not gated (R4 names fwd, back, turn, move_turn); the sideways head is not priced in W's omni economy (nominal share). Per-category solve: each dose_targets category bisected on its own delivered share; tie groups bound by their lower-weight member; shares_at_w_S are at the solved table.",
 "per_category_solve": {
  "hand_min": true,
  "per_run": {
   "E3": {
    "back": {
     "n_steps": 49448,
     "reached": true,
     "share": 0.1,
     "target": 0.1,
     "w": 0.17272811428263338
    },
    "diag": {
     "n_steps": 44226,
     "reached": true,
     "share": 0.1,
     "target": 0.1,
     "w": 0.17963707619400293
    },
    "fwd": {
     "n_steps": 47727,
     "reached": true,
     "share": 0.09999999999999999,
     "target": 0.1,
     "w": 0.1362210975498265
    },
    "side": {
     "bound_by_hand": "r",
     "hands": {
      "l": {
       "n_steps": 52184,
       "reached": true,
       "share": 0.1,
       "target": 0.1,
       "w": 0.5715633347460825
      },
      "r": {
       "n_steps": 53148,
       "reached": true,
       "share": 0.1,
       "target": 0.1,
       "w": 0.5653231591817289
      }
     },
     "n_steps": 53148,
     "pooled": {
      "n_steps": 105332,
      "reached": true,
      "share": 0.1,
      "target": 0.1,
      "w": 0.5684133627617872
     },
     "reached": true,
     "share": 0.1,
     "target": 0.1,
     "w": 0.5653231591817289
    },
    "turn": {
     "bound_by_hand": "r",
     "hands": {
      "l": {
       "n_steps": 53883,
       "reached": true,
       "share": 0.09999999999999998,
       "target": 0.1,
       "w": 0.28859381012404905
      },
      "r": {
       "n_steps": 54281,
       "reached": true,
       "share": 0.09999999999999999,
       "target": 0.1,
       "w": 0.23769722270275517
      }
     },
     "n_steps": 54281,
     "pooled": {
      "n_steps": 108164,
      "reached": true,
      "share": 0.09999999999999999,
      "target": 0.1,
      "w": 0.2602998261797019
     },
     "reached": true,
     "share": 0.09999999999999999,
     "target": 0.1,
     "w": 0.23769722270275517
    }
   }
  },
  "reached": {
   "back": true,
   "diag": true,
   "fwd": true,
   "side": true,
   "turn": true
  },
  "spec": "fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0",
  "table": {
   "back": 1.0,
   "diag": 1.318716993366581,
   "fwd": 1.0,
   "move_turn": 1.0,
   "side": 4.150041141570944,
   "turn": 1.7449369222400448
  },
  "targets": {
   "back": 0.1,
   "diag": 0.1,
   "fwd": 0.1,
   "side": 0.1,
   "turn": 0.1
  },
  "tie": [
   [
    "fwd",
    "back"
   ]
  ],
  "tie_bound_by": {
   "fwd,back": "fwd"
  },
  "w_S": 0.1362210975498265,
  "w_c": {
   "back": 0.1362210975498265,
   "diag": 0.17963707619400293,
   "fwd": 0.1362210975498265,
   "side": 0.5653231591817289,
   "turn": 0.23769722270275517
  },
  "w_eff": {
   "back": 0.1362210975498265,
   "diag": 0.17963707619400293,
   "fwd": 0.1362210975498265,
   "move_turn": 0.1362210975498265,
   "side": 0.5653231591817289,
   "turn": 0.23769722270275517
  }
 },
 "per_run": {
  "E3": {
   "back": 0.17272811428263338,
   "diag": 0.17963707619400293,
   "fwd": 0.1362210975498265,
   "side": 0.5653231591817289,
   "turn": 0.23769722270275517
  }
 },
 "shares_at_w_S": {
  "Bf_s2": {
   "back": 0.0793332634629169,
   "diag": 0.0981452757680026,
   "fwd": 0.10457309343599937,
   "move_turn": 0.011208370615913945,
   "side": 0.1225596937069657,
   "turn": 0.054638895033662975
  },
  "Bf_s3": {
   "back": 0.06772054970071081,
   "diag": 0.08971774580316086,
   "fwd": 0.07870025603784904,
   "move_turn": 0.01006140817400215,
   "side": 0.1041040423840837,
   "turn": 0.07590355120127577
  },
  "E3": {
   "back": 0.07832834576939601,
   "diag": 0.1,
   "fwd": 0.09999999999999999,
   "move_turn": 0.014508305427600702,
   "side": 0.09944947985240399,
   "turn": 0.09121127796812446
  }
 },
 "sideways_side_share_nominal": {
  "side_slow": 0.9569919649761357,
  "side_step": 0.8110399305862318
 },
 "stand_pay": {
  "Bf_s2": 0.0,
  "Bf_s3": 0.0,
  "E3": 0.0
 },
 "stroke_cat_w": {
  "spec": "fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0",
  "table": {
   "back": 1.0,
   "diag": 1.318716993366581,
   "fwd": 1.0,
   "move_turn": 1.0,
   "side": 4.150041141570944,
   "turn": 1.7449369222400448
  },
  "w_eff": {
   "back": 0.1362210975498265,
   "diag": 0.17963707619400293,
   "fwd": 0.1362210975498265,
   "move_turn": 0.1362210975498265,
   "side": 0.5653231591817289,
   "turn": 0.23769722270275517
  }
 },
 "w_S": 0.1362210975498265
}
```

### R5: FAIL

```
{
 "clauses": {
  "no_flip": false,
  "term_alone<=0.5x (translation)": true
 },
 "headline": "no-flip fails: ['Bf_s3>E7_creeper move_turn']; max term-alone (translation) E7_creeper/Bf_s3.fwd 0.315; max term-alone (rotational, reported) E7_creeper/Bf_s3.move_turn 3.185",
 "rg1": {
  "pass": false,
  "rows": [
   {
    "base": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.3395768502141274,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.9774, free -35.9774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.411002488718278,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.2909356647070047,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -34.9677, free -34.9677); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.358004183103306,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.3438286273129805,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.4071, free -35.4071); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.4030761857756402,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.1975194261902986,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1605, free -35.1605); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.2625617799224165,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.5815241726715061,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.0392, free -35.0392); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.6582449825633307,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0087534710568828,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.0774, free -36.0774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0299779204706394,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 1.3395768502141274,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.0213192536777291,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.0213192536777291,
    "ratio_with": 1.0625152215731333,
    "with": {
     "degenerate_total": 1.3279833173864402,
     "healthy_total": 1.411002488718278,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.8350981187954698,
     "healthy_total": 1.2909356647070047,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.5458490872533956,
    "ratio_with": 1.624097121830098,
    "with": {
     "degenerate_total": 0.8361594666044678,
     "healthy_total": 1.358004183103306,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.14219132091521,
     "healthy_total": 1.3438286273129805,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.1765354916514368,
    "ratio_with": 1.2154295963266804,
    "with": {
     "degenerate_total": 1.154387049662171,
     "healthy_total": 1.4030761857756402,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.138364596332942,
     "healthy_total": 1.1975194261902986,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.0519647484188408,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.0519647484188408,
    "ratio_with": 1.1038016921249774,
    "with": {
     "degenerate_total": 1.1438302631080435,
     "healthy_total": 1.2625617799224165,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.5815241726715061,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 2.014132456252877,
    "ratio_with": 2.0206367644338257,
    "with": {
     "degenerate_total": 0.8206546628027747,
     "healthy_total": 1.6582449825633307,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 1.0087534710568828,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 2.20151958293495,
    "ratio_with": 2.0433458738127293,
    "with": {
     "degenerate_total": 0.5040644042062141,
     "healthy_total": 1.0299779204706394,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 0.8884460816321142,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.9774, free -35.9774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 0.9403397443382089,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.1247940958621625,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -34.9677, free -34.9677); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.1976696616322748,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.1012360845703204,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.4071, free -35.4071); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.1506977505700509,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 0.7838490005825907,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1605, free -35.1605); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 0.8453755200703832,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.429995562936238,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.0392, free -35.0392); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.492675264231464,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 0.5130861986264006,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.0774, free -36.0774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 0.5275135640259487,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.8350981187954698,
     "healthy_total": 1.1247940958621625,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.346900526472919,
    "ratio_with": 1.432345993158281,
    "with": {
     "degenerate_total": 0.8361594666044678,
     "healthy_total": 1.1976696616322748,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.429995562936238,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.8211548867711924,
    "ratio_with": 1.8188835473541858,
    "with": {
     "degenerate_total": 0.8206546628027747,
     "healthy_total": 1.492675264231464,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 0.5130861986264006,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s3",
    "note": null,
    "pass": false,
    "ratio_base": 1.1197674619411315,
    "ratio_with": 1.0465201661217511,
    "with": {
     "degenerate_total": 0.5040644042062141,
     "healthy_total": 0.5275135640259487,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.5007985558378159,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.9774, free -35.9774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.5730379079452117,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.2391064218852115,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -34.9677, free -34.9677); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.3160248648941881,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.4994583773078398,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.4071, free -35.4071); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.556457533208417,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.3214437215179158,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1605, free -35.1605); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.3823887153525887,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.7332355856839796,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.0392, free -35.0392); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.7773693865215314,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0593069249987284,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.0774, free -36.0774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0744954751194378,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 1.5007985558378159,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.1442377947363593,
    "ratio_with": 1.184531377277431,
    "with": {
     "degenerate_total": 1.3279833173864402,
     "healthy_total": 1.5730379079452117,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.8350981187954698,
     "healthy_total": 1.2391064218852115,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.4837854307137894,
    "ratio_with": 1.573892202929173,
    "with": {
     "degenerate_total": 0.8361594666044678,
     "healthy_total": 1.3160248648941881,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.14219132091521,
     "healthy_total": 1.4994583773078398,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.3127909045100785,
    "ratio_with": 1.3482978119547606,
    "with": {
     "degenerate_total": 1.154387049662171,
     "healthy_total": 1.556457533208417,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.138364596332942,
     "healthy_total": 1.3214437215179158,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.1608264397669548,
    "ratio_with": 1.2085610600967387,
    "with": {
     "degenerate_total": 1.1438302631080435,
     "healthy_total": 1.3823887153525887,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.7332355856839796,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.2073428328076945,
    "ratio_with": 2.165794538291292,
    "with": {
     "degenerate_total": 0.8206546628027747,
     "healthy_total": 1.7773693865215314,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 1.0593069249987284,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.311848243039949,
    "ratio_with": 2.131663069546682,
    "with": {
     "degenerate_total": 0.5040644042062141,
     "healthy_total": 1.0744954751194378,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 0.8884460816321142,
     "ordering": false
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 0.6773684457747762,
    "healthy": "Bf_s3",
    "note": "no clock-held ordering: not an RG1 criterion",
    "pass": null,
    "ratio_base": 0.6773684457747762,
    "ratio_with": 0.7080960521317845,
    "with": {
     "degenerate_total": 1.3279833173864402,
     "healthy_total": 0.9403397443382089,
     "ordering": false
    }
   },
   {
    "base": {
     "degenerate_total": 1.14219132091521,
     "healthy_total": 1.1012360845703204,
     "ordering": false
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 0.9641432782801457,
    "healthy": "Bf_s3",
    "note": "no clock-held ordering: not an RG1 criterion",
    "pass": null,
    "ratio_base": 0.9641432782801457,
    "ratio_with": 0.9968041056132777,
    "with": {
     "degenerate_total": 1.154387049662171,
     "healthy_total": 1.1506977505700509,
     "ordering": false
    }
   },
   {
    "base": {
     "degenerate_total": 1.138364596332942,
     "healthy_total": 0.7838490005825907,
     "ordering": false
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 0.6885746474439154,
    "healthy": "Bf_s3",
    "note": "no clock-held ordering: not an RG1 criterion",
    "pass": null,
    "ratio_base": 0.6885746474439154,
    "ratio_with": 0.7390742729374097,
    "with": {
     "degenerate_total": 1.1438302631080435,
     "healthy_total": 0.8453755200703832,
     "ordering": fa
```

### R6: FAIL

```
{
 "headline": "max income shift Bf_s2.side 0.0329; max liftoff shift Bf_s2.side 0.0485",
 "pattern": "force repeats the previous reading on s % 5 == 4",
 "rows": {
  "Bf_s2.back": {
   "income": 0.4187380363786969,
   "income_dup": 0.4095671295675361,
   "income_shift": 0.02190129869851809,
   "liftoff_shift": 0.02104781514527576,
   "liftoffs": 4371,
   "liftoffs_dup": 4463
  },
  "Bf_s2.diag": {
   "income": 0.3394851918431758,
   "income_dup": 0.3330499504597445,
   "income_shift": 0.018955882430371385,
   "liftoff_shift": 0.03417062683865124,
   "liftoffs": 4419,
   "liftoffs_dup": 4570
  },
  "Bf_s2.fwd": {
   "income": 0.5312531326318074,
   "income_dup": 0.5243738429495347,
   "income_shift": 0.012949174808989716,
   "liftoff_shift": 0.013791434793128499,
   "liftoffs": 4133,
   "liftoffs_dup": 4190
  },
  "Bf_s2.move_turn": {
   "income": 0.11129459401869987,
   "income_dup": 0.11135632533157863,
   "income_shift": 0.000554665870548865,
   "liftoff_shift": 0.014976415094339668,
   "liftoffs": 8480,
   "liftoffs_dup": 8607
  },
  "Bf_s2.side": {
   "income": 0.13591774875132318,
   "income_dup": 0.14039287606192905,
   "income_shift": 0.03292526069419832,
   "liftoff_shift": 0.04854763355677427,
   "liftoffs": 9846,
   "liftoffs_dup": 10324
  },
  "Bf_s2.turn": {
   "income": 0.18561951679939548,
   "income_dup": 0.18329752898881566,
   "income_shift": 0.012509394758791759,
   "liftoff_shift": 0.016290874749144146,
   "liftoffs": 8471,
   "liftoffs_dup": 8609
  },
  "Bf_s3.back": {
   "income": 0.36202639695119504,
   "income_dup": 0.35826979110794466,
   "income_shift": 0.010376607548197136,
   "liftoff_shift": 0.014833127317676054,
   "liftoffs": 3236,
   "liftoffs_dup": 3284
  },
  "Bf_s3.diag": {
   "income": 0.3430945251176602,
   "income_dup": 0.3377358963150811,
   "income_shift": 0.015618520291868276,
   "liftoff_shift": 0.022122728469844644,
   "liftoffs": 3797,
   "liftoffs_dup": 3881
  },
  "Bf_s3.fwd": {
   "income": 0.381501591626024,
   "income_dup": 0.3751673509067576,
   "income_shift": 0.016603445066293854,
   "liftoff_shift": 0.01107351584127958,
   "liftoffs": 3251,
   "liftoffs_dup": 3287
  },
  "Bf_s3.move_turn": {
   "income": 0.10576844028251324,
   "income_dup": 0.10324725475940076,
   "income_shift": 0.023836841276833165,
   "liftoff_shift": 0.012711864406779627,
   "liftoffs": 5900,
   "liftoffs_dup": 5975
  },
  "Bf_s3.side": {
   "income": 0.12857073200200972,
   "income_dup": 0.12918347912628345,
   "income_shift": 0.0047658367867435825,
   "liftoff_shift": 0.03043130076650269,
   "liftoffs": 8741,
   "liftoffs_dup": 9007
  },
  "Bf_s3.turn": {
   "income": 0.2644148188016246,
   "income_dup": 0.25848769569374963,
   "income_shift": 0.02241600200298055,
   "liftoff_shift": 0.012234580384226534,
   "liftoffs": 9890,
   "liftoffs_dup": 10011
  },
  "E3.back": {
   "income": 0.4351197004973975,
   "income_dup": 0.43041582266463946,
   "income_shift": 0.010810537485158478,
   "liftoff_shift": 0.01681479702137878,
   "liftoffs": 4163,
   "liftoffs_dup": 4233
  },
  "E3.diag": {
   "income": 0.3642029642158494,
   "income_dup": 0.3599634801484845,
   "income_shift": 0.011640443609493345,
   "liftoff_shift": 0.029935851746258013,
   "liftoffs": 4209,
   "liftoffs_dup": 4335
  },
  "E3.fwd": {
   "income": 0.5244388604700138,
   "income_dup": 0.5172418734273347,
   "income_shift": 0.01372321463026005,
   "liftoff_shift": 0.01866404715127712,
   "liftoffs": 4072,
   "liftoffs_dup": 4148
  },
  "E3.move_turn": {
   "income": 0.15554142594041134,
   "income_dup": 0.1540989932791695,
   "income_shift": 0.009273623746990922,
   "liftoff_shift": 0.020941148152605615,
   "liftoffs": 8309,
   "liftoffs_dup": 8483
  },
  "E3.side": {
   "income": 0.11853260197750105,
   "income_dup": 0.11866068326665924,
   "income_shift": 0.0010805574755079483,
   "liftoff_shift": 0.03157158557298434,
   "liftoffs": 10009,
   "liftoffs_dup": 10325
  },
  "E3.turn": {
   "income": 0.32437544135995167,
   "income_dup": 0.31945686457420847,
   "income_shift": 0.015163221867604881,
   "liftoff_shift": 0.011862917398945516,
   "liftoffs": 9104,
   "liftoffs_dup": 9212
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
 "headline": "max |clip shift| pp Bf_s3.diag -0.41; max absorbed Bf_s3.move_turn 0.107",
 "rows": {
  "Bf_s2.back": {
   "absorbed": 0.021622267481043456,
   "clip_base": 0.08339606705469768,
   "clip_with": 0.08102748655050218,
   "shift_pp": -0.23685805041955
  },
  "Bf_s2.diag": {
   "absorbed": 0.031168715822807558,
   "clip_base": 0.11351256369706554,
   "clip_with": 0.1103887229348484,
   "shift_pp": -0.31238407622171394
  },
  "Bf_s2.fwd": {
   "absorbed": 0.03254202868890188,
   "clip_base": 0.07936901064985044,
   "clip_with": 0.07630170130884566,
   "shift_pp": -0.3067309341004784
  },
  "Bf_s2.move_turn": {
   "absorbed": 0.051133188520469774,
   "clip_base": 0.16647380231235975,
   "clip_with": 0.16495436661668963,
   "shift_pp": -0.15194356956701183
  },
  "Bf_s2.side": {
   "absorbed": 0.023285194725376712,
   "clip_base": 0.12210352718816303,
   "clip_with": 0.11973168539742247,
   "shift_pp": -0.2371841790740556
  },
  "Bf_s2.turn": {
   "absorbed": 0.016282674513507622,
   "clip_base": 0.04200522078017574,
   "clip_with": 0.04096657965366374,
   "shift_pp": -0.10386411265120021
  },
  "Bf_s3.back": {
   "absorbed": 0.0873772439118744,
   "clip_base": 0.150117634477596,
   "clip_with": 0.14669554058389478,
   "shift_pp": -0.34220938937012113
  },
  "Bf_s3.diag": {
   "absorbed": 0.07915842604754986,
   "clip_base": 0.15788567493112948,
   "clip_with": 0.15382723337268792,
   "shift_pp": -0.4058441558441567
  },
  "Bf_s3.fwd": {
   "absorbed": 0.07969085444483182,
   "clip_base": 0.18261594001666204,
   "clip_with": 0.17858928075534575,
   "shift_pp": -0.402665926131629
  },
  "Bf_s3.move_turn": {
   "absorbed": 0.10662193227305139,
   "clip_base": 0.22822605965463108,
   "clip_with": 0.2265934065934066,
   "shift_pp": -0.16326530612244816
  },
  "Bf_s3.side": {
   "absorbed": 0.04817714301727383,
   "clip_base": 0.11378584987943154,
   "clip_with": 0.11126160792160485,
   "shift_pp": -0.25242419578266895
  },
  "Bf_s3.turn": {
   "absorbed": 0.017229344334440788,
   "clip_base": 0.061617754529020345,
   "clip_with": 0.060156159852302014,
   "shift_pp": -0.14615946767183308
  },
  "E3.back": {
   "absorbed": 0.0478364236661295,
   "clip_base": 0.11193577091085585,
   "clip_with": 0.10920563015693253,
   "shift_pp": -0.27301407539233186
  },
  "E3.diag": {
   "absorbed": 0.03905116767260486,
   "clip_base": 0.11671867227422783,
   "clip_with": 0.11328177994844661,
   "shift_pp": -0.34368923257812184
  },
  "E3.fwd": {
   "absorbed": 0.04260446453575839,
   "clip_base": 0.12215307896997507,
   "clip_with": 0.11880067886102207,
   "shift_pp": -0.3352400108953005
  },
  "E3.move_turn": {
   "absorbed": 0.053652099501633854,
   "clip_base": 0.15091974147807102,
   "clip_with": 0.14934180662732638,
   "shift_pp": -0.15779348507446445
  },
  "E3.side": {
   "absorbed": 0.024701439702639494,
   "clip_base": 0.08772262940037216,
   "clip_with": 0.0860327345915771,
   "shift_pp": -0.1689894808795056
  },
  "E3.turn": {
   "absorbed": 0.018701809670286673,
   "clip_base": 0.05469472282829777,
   "clip_with": 0.05309529973003957,
   "shift_pp": -0.15994230982582028
  }
 }
}
```

### R8: FAIL

```
{
 "headline": "max hand gap E3.turn 0.214",
 "rows": {
  "Bf_s2.move_turn": {
   "gap": 0.21065333960408592,
   "l": 0.12294480852448414,
   "r": 0.09951396657501252
  },
  "Bf_s2.side": {
   "gap": 0.016721351711940877,
   "l": 0.13476720451246024,
   "r": 0.1370396938853813
  },
  "Bf_s2.turn": {
   "gap": 0.010478367766586484,
   "l": 0.18464733131841232,
   "r": 0.18659232413811952
  },
  "Bf_s3.move_turn": {
   "gap": 0.19732338725016527,
   "l": 0.09585455869667758,
   "r": 0.11683929447415675
  },
  "Bf_s3.side": {
   "gap": 0.05000034334763117,
   "l": 0.12539926201893725,
   "r": 0.1318300386963551
  },
  "Bf_s3.turn": {
   "gap": 0.08808834165884355,
   "l": 0.2758140735091988,
   "r": 0.25254302327930844
  },
  "E3.move_turn": {
   "gap": 0.1502935221544621,
   "l": 0.14331733978161804,
   "r": 0.16660716229083658
  },
  "E3.side": {
   "gap": 0.0032000946107724653,
   "l": 0.11872400192875959,
   "r": 0.1183446808217332
  },
  "E3.turn": {
   "gap": 0.214179453260184,
   "l": 0.2895241030769213,
   "r": 0.3589712983349952
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
     "Bf_s2.back": 0.687440037727356,
     "Bf_s2.fwd": 0.7352146506309509,
     "Bf_s2.turn": 0.42333847284317017,
     "Bf_s3.back": 0.7819738984107971,
     "Bf_s3.fwd": 0.7721610069274902,
     "Bf_s3.turn": 0.6088981628417969,
     "E3.back": 0.7683157920837402,
     "E3.fwd": 0.7942185401916504,
     "E3.turn": 0.5985913276672363,
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
     "Bf_s2": 0.4399391710758209,
     "Bf_s3": 0.526648223400116,
     "E3": 0.44106799364089966
    },
    "slide_kept": 0.03570928294066488
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "c_max=1.75": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.686125636100769,
     "Bf_s2.fwd": 0.7335114479064941,
     "Bf_s2.turn": 0.42334869503974915,
     "Bf_s3.back": 0.7819819450378418,
     "Bf_s3.fwd": 0.7704170346260071,
     "Bf_s3.turn": 0.6088981628417969,
     "E3.back": 0.7677779793739319,
     "E3.fwd": 0.7941877841949463,
     "E3.turn": 0.5985913276672363,
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
     "Bf_s2": 0.4399391710758209,
     "Bf_s3": 0.526648223400116,
     "E3": 0.44106799364089966
    },
    "slide_kept": 0.0356922437350623
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "drho=0.2": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6665074825286865,
     "Bf_s2.fwd": 0.7161881923675537,
     "Bf_s2.turn": 0.4104566276073456,
     "Bf_s3.back": 0.761150598526001,
     "Bf_s3.fwd": 0.749996542930603,
     "Bf_s3.turn": 0.5981816053390503,
     "E3.back": 0.7471849322319031,
     "E3.fwd": 0.7736349105834961,
     "E3.turn": 0.5848338007926941,
     "side_slow.side": 0.8497278094291687,
     "side_step.side": 0.7529549598693848
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
     "Bf_s2": 0.4024243950843811,
     "Bf_s3": 0.48716098070144653,
     "E3": 0.4065870940685272
    },
    "slide_kept": 0.005558534225857255
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "drho=0.4": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6990009546279907,
     "Bf_s2.fwd": 0.7448354959487915,
     "Bf_s2.turn": 0.43241748213768005,
     "Bf_s3.back": 0.7973523139953613,
     "Bf_s3.fwd": 0.7873112559318542,
     "Bf_s3.turn": 0.6165443658828735,
     "E3.back": 0.782558262348175,
     "E3.fwd": 0.8085887432098389,
     "E3.turn": 0.6085730791091919,
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
     "Bf_s2": 0.47318193316459656,
     "Bf_s3": 0.5604812502861023,
     "E3": 0.4726469814777374
    },
    "slide_kept": 0.07975704459540289
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "force_hold=15": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6840888261795044,
     "Bf_s2.fwd": 0.7329406142234802,
     "Bf_s2.turn": 0.422319620847702,
     "Bf_s3.back": 0.7798981666564941,
     "Bf_s3.fwd": 0.7705990076065063,
     "Bf_s3.turn": 0.6066116094589233,
     "E3.back": 0.7660897374153137,
     "E3.fwd": 0.7926174998283386,
     "E3.turn": 0.5966724157333374,
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
     "Bf_s2": 0.43650227785110474,
     "Bf_s3": 0.5213415026664734,
     "E3": 0.43866872787475586
    },
    "slide_kept": 0.03562029466026781
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "force_hold=25": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6877660155296326,
     "Bf_s2.fwd": 0.7349548935890198,
     "Bf_s2.turn": 0.4243205487728119,
     "Bf_s3.back": 0.7843128442764282,
     "Bf_s3.fwd": 0.7722157835960388,
     "Bf_s3.turn": 0.6096895337104797,
     "E3.back": 0.7690437436103821,
     "E3.fwd": 0.7962043881416321,
     "E3.turn": 0.6006465554237366,
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
     "Bf_s2": 0.44389528036117554,
     "Bf_s3": 0.533010721206665,
     "E3": 0.445061057806015
    },
    "slide_kept": 0.036012841538049054
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "force_open=40": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.686122477054596,
     "Bf_s2.fwd": 0.7338336706161499,
     "Bf_s2.turn": 0.4232921004295349,
     "Bf_s3.back": 0.7819041609764099,
     "Bf_s3.fwd": 0.7714614272117615,
     "Bf_s3.turn": 0.6088786721229553,
     "E3.back": 0.7676643133163452,
     "E3.fwd": 0.7941440939903259,
     "E3.turn": 0.5985577702522278,
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
     "Bf_s2": 0.4398084580898285,
     "Bf_s3": 0.5265644788742065,
     "E3": 0.44105106592178345
    },
    "slide_kept": 0.035698145961711905
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "force_open=60": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6861367225646973,
     "Bf_s2.fwd": 0.7338545322418213,
     "Bf_s2.turn": 0.42332273721694946,
     "Bf_s3.back": 0.7820131778717041,
     "Bf_s3.fwd": 0.7714508175849915,
     "Bf_s3.turn": 0.6089286804199219,
     "E3.back": 0.767780065536499,
     "E3.fwd": 0.794256329536438,
     "E3.turn": 0.5986197590827942,
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
     "Bf_s2": 0.43996626138687134,
     "Bf_s3": 0.5266823768615723,
     "E3": 0.4410586655139923
    },
    "slide_kept": 0.03568489440418743
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "min_loaded=4": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.686352014541626,
     "Bf_s2.fwd": 0.73404461145401,
     "Bf_s2.turn": 0.42344698309898376,
     "Bf_s3.back": 0.782276451587677,
     "Bf_s3.fwd": 0.7719359397888184,
     "Bf_s3.turn": 0.6093651652336121,
     "E3.back": 0.7681353092193604,
     "E3.fwd": 0.794567346572876,
     "E3.turn": 0.5988221764564514,
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
     "Bf_s2": 0.44048020243644714,
     "Bf_s3": 0.5276682376861572,
     "E3": 0.44189268350601196
    },
    "slide_kept": 0.0357196916586702
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "min_loaded=6": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6859581470489502,
     "Bf_s2.fwd": 0.7335165739059448,
     "Bf_s2.turn": 0.4230942726135254,
     "Bf_s3.back": 0.7813814282417297,
     "Bf_s3.fwd": 0.7704080939292908,
     "Bf_s3.turn": 0.6081692576408386,
     "E3.back": 0.7673184871673584,
     "E3.fwd": 0.7935982942581177,
     "E3.turn": 0.5981228947639465,
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
     "Bf_s2": 0.43937426805496216,
     "Bf_s3": 0.5253866910934448,
     "E3": 0.44020238518714905
    },
    "slide_kept": 0.035651932407165965
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "n_off=1": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.7595300674438477,
     "Bf_s2.fwd": 0.7905778884887695,
     "Bf_s2.turn": 0.4734131395816803,
     "Bf_s3.back": 0.8455392718315125,
     "Bf_s3.fwd": 0.8319420218467712,
     "Bf_s3.turn": 0.6377705335617065,
     "E3.back": 0.8292779922485352,
     "E3.fwd": 0.8504250049591064,
     "E3.turn": 0.6456878185272217,
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
     "Bf_s2": 0.5736214518547058,
     "Bf_s3": 0.6470504999160767,
     "E3": 0.5645094513893127
    },
    "slide_kept": 0.04036903990964506
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
     "Bf_s2.back": 0.6385793685913086,
     "Bf_s2.fwd": 0.6844789981842041,
     "Bf_s2.turn": 0.38872799277305603,
     "Bf_s3.back": 0.7334131002426147,
     "Bf_s3.fwd": 0.7245977520942688,
     "Bf_s3.turn": 0.5799551606178284,
     "E3.back": 0.7265729308128357,
     "E3.fwd": 0.7557844519615173,
     "E3.turn": 0.564071774482727,
     "side_slow.side": 0.847476065158844,
     "side_step.side": 0.7441140413284302
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
     "Bf_s2": 0.36986520886421204,
     "Bf_s3": 0.4582623839378357,
     "E3": 0.36834946274757385
    },
    "slide_kept": 0.03383683236208187
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "power=2": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6330515742301941,
     "Bf_s2.fwd": 0.6882471442222595,
     "Bf_s2.turn": 0.4166613519191742,
     "Bf_s3.back": 0.7821835279464722,
     "Bf_s3.fwd": 0.7672304511070251,
     "Bf_s3.turn": 0.6250395774841309,
     "E3.back": 0.7586115002632141,
     "E3.fwd": 0.787212073802948,
     "E3.turn": 0.6100465655326843,
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
     "Bf_s2": 0.4586493670940399,
     "Bf_s3": 0.5584197640419006,
     "E3": 0.4695635139942169
    },
    "slide_kept": 0.03593294434064192
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "prog_lo=0.1": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6889091730117798,
     "Bf_s2.fwd": 0.7351886034011841,
     "Bf_s2.turn": 0.503441572189331,
     "Bf_s3.back": 0.7846549153327942,
     "Bf_s3.fwd": 0.7760971188545227,
     "Bf_s3.turn": 0.7022430300712585,
     "E3.back": 0.7697095274925232,
     "E3.fwd": 0.7957028150558472,
     "E3.turn": 0.6603813767433167,
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
     "Bf_s2": 0.44013741612434387,
     "Bf_s3": 0.5267770290374756,
     "E3": 0.4411000907421112
    },
    "slide_kept": 0.0372102058373745
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "prog_lo=0.2": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6811267733573914,
     "Bf_s2.fwd": 0.730051577091217,
     "Bf_s2.turn": 0.3251226246356964,
     "Bf_s3.back": 0.7762865424156189,
     "Bf_s3.fwd": 0.7613009810447693,
     "Bf_s3.turn": 0.49124860763549805,
     "E3.back": 0.7630795836448669,
     "E3.fwd": 0.7898672819137573,
     "E3.turn": 0.511834442615509,
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
     "Bf_s2": 0.4390821158885956,
     "Bf_s3": 0.5251468420028687,
     "E3": 0.4404515326023102
    },
    "slide_kept": 0.03394016520639383
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "prog_tau=3": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6857604384422302,
     "Bf_s2.fwd": 0.7333795428276062,
     "Bf_s2.turn": 0.4240792691707611,
     "Bf_s3.back": 0.780875563621521,
     "Bf_s3.fwd": 0.7699494361877441,
     "Bf_s3.turn": 0.6184419393539429,
     "E3.back": 0.7673543095588684,
     "E3.fwd": 0.7935575246810913,
     "E3.turn": 0.6050551533699036,
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
     "Bf_s2": 0.439737468957901,
     "Bf_s3": 0.5263872146606445,
     "E3": 0.44096624851226807
    },
    "slide_kept": 0.03543064764972571
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "prog_tau=5": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6864691972732544,
     "Bf_s2.fwd": 0.7341423630714417,
     "Bf_s2.turn": 0.42765504121780396,
     "Bf_s3.back": 0.7826606035232544,
     "Bf_s3.fwd": 0.7723655700683594,
     "Bf_s3.turn": 0.6088295578956604,
     "E3.back": 0.768094003200531,
     "E3.fwd": 0.7945572137832642,
     "E3.turn": 0.5972272157669067,
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
     "Bf_s2": 0.44002583622932434,
     "Bf_s3": 0.5267288088798523,
     "E3": 0.4410881996154785
    },
    "slide_kept": 0.03596116469696917
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "prog_width=0.1": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6882350444793701,
     "Bf_s2.fwd": 0.7349007725715637,
     "Bf_s2.turn": 0.465975821018219,
     "Bf_s3.back": 0.7838345170021057,
     "Bf_s3.fwd": 0.7751772999763489,
     "Bf_s3.turn": 0.6605986952781677,
     "E3.back": 0.7692126035690308,
     "E3.fwd": 0.7954916954040527,
     "E3.turn": 0.6319038271903992,
     "side_slow.side": 0.8699360489845276,
     "side_step.side": 0.7755095362663269
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
     "Bf_s2": 0.4401305615901947,
     "Bf_s3": 0.5267767906188965,
     "E3": 0.44109997153282166
    },
    "slide_kept": 0.03627724019091233
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "prog_width=0.2": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.68333899974823,
     "Bf_s2.fwd": 0.7314752340316772,
     "Bf_s2.turn": 0.3811793029308319,
     "Bf_s3.back": 0.7787124514579773,
     "Bf_s3.fwd": 0.76540207862854,
     "Bf_s3.turn": 0.5572952032089233,
     "E3.back": 0.7649573683738708,
     "E3.fwd": 0.7914299368858337,
     "E3.turn": 0.5579689741134644,
     "side_slow.side": 0.8691660761833191,
     "side_step.side": 0.7709604501724243
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
     "Bf_s2": 0.43934935331344604,
     "Bf_s3": 0.5255544781684875,
     "E3": 0.440613716840744
    },
    "slide_kept": 0.0350572739408744
   },
   "R3": {
    "clauses": {
     "short<=2%": true,
     "spearman<=+0.2": true,
     "v4<=healthy": null
    },
    "pass": true
   }
  },
  "rho0=0.4": {
   "R1": {
    "kept": {
     "Bf_s2.back": 0.6531360149383545,
     "Bf_s2.fwd": 0.704230546951294,
     "Bf_s2.turn": 0.4058164656162262,
     "Bf_s3.back": 0.752997875213623,
     "Bf_s3.fwd": 0.7378774881362915,
     "Bf_s3.turn": 0.5923677682876587,
     "E3.back": 0.7364392280578613,
     "E3.fwd": 0.763577401638031,
     "E3.turn": 0.5786253213882446,
     "side_slow.side": 0.8400970697402954,
     "side_step.side": 0.7431053519248962
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
     "Bf_s2": 0.39473050832748413,
     "Bf_s3": 0.4778309464454651,
     "E3": 0.3983674943447113
    },
    "slide_kept": 0.01381491846669
```

### P1: FAIL

```
{
 "clauses": {
  "E3 side in [0.15,0.45]": true,
  "fwd/back/stand/turn==0": true,
  "sideways>=0.85": false
 },
 "headline": "sideways side {'side_slow': 0.9420073938338351, 'side_step': 0.752277317627347}; healthy side E3 0.417, Bf_s3 0.526, Bf_s2 0.594; max |zero row| Bf_s2.back 3.69e-08",
 "healthy_side": {
  "Bf_s2": 0.5939083239501011,
  "Bf_s3": 0.5258843048754873,
  "E3": 0.4174608947593977
 },
 "sideways_side": {
  "side_slow": 0.9420073938338351,
  "side_step": 0.752277317627347
 },
 "zero_rows": {
  "Bf_s2.back": 3.693961919078155e-08,
  "Bf_s2.fwd": 0.0,
  "Bf_s2.stand": 0.0,
  "Bf_s2.turn": 0.0,
  "Bf_s3.back": 3.452156570562091e-08,
  "Bf_s3.fwd": 0.0,
  "Bf_s3.stand": 0.0,
  "Bf_s3.turn": 0.0,
  "E3.back": 3.519467498000678e-08,
  "E3.fwd": 0.0,
  "E3.stand": 0.0,
  "E3.turn": 0.0
 }
}
```

### P2: FAIL

```
{
 "centring_prize_per_w": 0.42968146097119336,
 "clauses": {
  "diag<=10%": false
 },
 "headline": "w_P 0.2673 band [0.21384755433835073, 0.334136803653673]; max healthy share Bf_s2.side 0.233",
 "reported": {
  "healthy share>15% (design-panel line; not gated)": [
   "E3.side",
   "Bf_s3.side",
   "Bf_s3.diag",
   "Bf_s2.side",
   "Bf_s2.diag"
  ]
 },
 "rule": "w_P = w_S * (stroke prize) / (centring prize); prize = sideways-head side minus the primary omni side, per second at w = 1; the parity band is [0.8, 1.25] x",
 "shares_at_w_P": {
  "Bf_s2.diag": 0.19955622227236752,
  "Bf_s2.move_turn": 0.0,
  "Bf_s2.side": 0.23282738211251194,
  "Bf_s3.diag": 0.1817430478877925,
  "Bf_s3.move_turn": 0.0,
  "Bf_s3.side": 0.19101697738771267,
  "E3.diag": 0.14818579755749645,
  "E3.move_turn": 0.0,
  "E3.side": 0.15771970158787066
 },
 "stroke_prize_per_w": 0.8431727098991524,
 "w_P": 0.2673094429229384,
 "w_P_band": [
  0.21384755433835073,
  0.334136803653673
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
 "headline": "no-flip PASS; x08/statue side totals 1.403/-34.968 ratio --; creeper/sideways 0.027",
 "rg1": {
  "pass": true,
  "rows": [
   {
    "base": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.3395768502141274,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.9774, free -35.9774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.3395768502141274,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.2909356647070047,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -34.9677, free -34.9677); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.4026223764110033,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.3438286273129805,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.4071, free -35.4071); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.3438286367170695,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.1975194261902986,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1605, free -35.1605); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.2999874708261696,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.5815241726715061,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.0392, free -35.0392); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.5815241726715061,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0087534710568828,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "E3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.0774, free -36.0774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0087534710568828,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 1.3395768502141274,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.0213192536777291,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.0213192536777291,
    "ratio_with": 1.0213192536777291,
    "with": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 1.3395768502141274,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.8350981187954698,
     "healthy_total": 1.2909356647070047,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.5458490872533956,
    "ratio_with": 1.6673002779834214,
    "with": {
     "degenerate_total": 0.8412536091623863,
     "healthy_total": 1.4026223764110033,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.14219132091521,
     "healthy_total": 1.3438286273129805,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.1765354916514368,
    "ratio_with": 1.1765354953815967,
    "with": {
     "degenerate_total": 1.1421913252869715,
     "healthy_total": 1.3438286367170695,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.138364596332942,
     "healthy_total": 1.1975194261902986,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.0519647484188408,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 1.0519647484188408,
    "ratio_with": 1.1037996509969001,
    "with": {
     "degenerate_total": 1.177738613753032,
     "healthy_total": 1.2999874708261696,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.5815241726715061,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 2.014132456252877,
    "ratio_with": 2.014132456252877,
    "with": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.5815241726715061,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 1.0087534710568828,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "E3",
    "note": null,
    "pass": true,
    "ratio_base": 2.20151958293495,
    "ratio_with": 2.20151958293495,
    "with": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 1.0087534710568828,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 0.8884460816321142,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.9774, free -35.9774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 0.8884460816321142,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.1247940958621625,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -34.9677, free -34.9677); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.2654251693906458,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.1012360845703204,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.4071, free -35.4071); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.1012360937987709,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 0.7838490005825907,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1605, free -35.1605); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 0.9142348162591023,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.429995562936238,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.0392, free -35.0392); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.429995562936238,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 0.5130861986264006,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s3",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.0774, free -36.0774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 0.5130861986264006,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.8350981187954698,
     "healthy_total": 1.1247940958621625,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.346900526472919,
    "ratio_with": 1.504213658768841,
    "with": {
     "degenerate_total": 0.8412536091623863,
     "healthy_total": 1.2654251693906458,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.429995562936238,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.8211548867711924,
    "ratio_with": 1.8211548867711924,
    "with": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.429995562936238,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 0.5130861986264006,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s3",
    "note": null,
    "pass": true,
    "ratio_base": 1.1197674619411315,
    "ratio_with": 1.1197674619411315,
    "with": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 0.5130861986264006,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.5007985558378159,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.9774, free -35.9774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.977422388389044,
     "healthy_total": 1.5007985558378159,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.2391064218852115,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -34.9677, free -34.9677); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -34.967747757352484,
     "healthy_total": 1.3978389323239695,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.4994583773078398,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.4071, free -35.4071); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.407114556595715,
     "healthy_total": 1.4994583871773774,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.3214437215179158,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.1605, free -35.1605); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.160532324626026,
     "healthy_total": 1.4532366368784928,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.7332355856839796,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -35.0392, free -35.0392); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -35.039171371308385,
     "healthy_total": 1.7332355856839796,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0593069249987284,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E6_statue",
    "floor": null,
    "healthy": "Bf_s2",
    "note": "RG1 ratio undefined: degenerate total <= 0 (held -36.0774, free -36.0774); reported, not divided (audit item 11) -- judged on the clock-free ordering only",
    "pass": true,
    "ratio_base": null,
    "ratio_with": null,
    "with": {
     "degenerate_total": -36.07740629935561,
     "healthy_total": 1.0593069249987284,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 1.5007985558378159,
     "ordering": true
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.1442377947363593,
    "ratio_with": 1.1442377947363593,
    "with": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 1.5007985558378159,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.8350981187954698,
     "healthy_total": 1.2391064218852115,
     "ordering": true
    },
    "category": "side",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.4837854307137894,
    "ratio_with": 1.66161418756439,
    "with": {
     "degenerate_total": 0.8412536091623863,
     "healthy_total": 1.3978389323239695,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.14219132091521,
     "healthy_total": 1.4994583773078398,
     "ordering": true
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.3127909045100785,
    "ratio_with": 1.312790908126223,
    "with": {
     "degenerate_total": 1.1421913252869715,
     "healthy_total": 1.4994583871773774,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.138364596332942,
     "healthy_total": 1.3214437215179158,
     "ordering": true
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 1.1608264397669548,
    "ratio_with": 1.2339211943196353,
    "with": {
     "degenerate_total": 1.177738613753032,
     "healthy_total": 1.4532366368784928,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.7332355856839796,
     "ordering": true
    },
    "category": "turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.2073428328076945,
    "ratio_with": 2.2073428328076945,
    "with": {
     "degenerate_total": 0.7852135879950011,
     "healthy_total": 1.7332355856839796,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 1.0593069249987284,
     "ordering": true
    },
    "category": "move_turn",
    "degenerate": "E7_creeper",
    "floor": 1.1,
    "healthy": "Bf_s2",
    "note": null,
    "pass": true,
    "ratio_base": 2.311848243039949,
    "ratio_with": 2.311848243039949,
    "with": {
     "degenerate_total": 0.4582078119478119,
     "healthy_total": 1.0593069249987284,
     "ordering": true
    }
   },
   {
    "base": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 0.8884460816321142,
     "ordering": false
    },
    "category": "fwd",
    "degenerate": "E7_creeper",
    "floor": 0.6773684457747762,
    "healthy": "Bf_s3",
    "note": "no clock-held ordering: not an RG1 criterion",
    "pass": null,
    "ratio_base": 0.6773684457747762,
    "ratio_with": 0.6773684457747762,
    "with": {
     "degenerate_total": 1.3116142140573241,
     "healthy_total": 0.8884460816321142,
     "ordering": false
    }
   },
   {
    "base": {
     "degenerate_total": 1.14219132091521,
     "healthy_total": 1.1012360845703204,
     "ordering": false
    },
    "category": "back",
    "degenerate": "E7_creeper",
    "floor": 0.9641432782801457,
    "healthy": "Bf_s3",
    "note": "no clock-held ordering: not an RG1 criterion",
    "pass": null,
    "ratio_base": 0.9641432782801457,
    "ratio_with": 0.964143282669468,
    "with": {
     "degenerate_total": 1.1421913252869715,
     "healthy_total": 1.1012360937987709,
     "ordering": false
    }
   },
   {
    "base": {
     "degenerate_total": 1.138364596332942,
     "healthy_total": 0.7838490005825907,
     "ordering": false
    },
    "category": "diag",
    "degenerate": "E7_creeper",
    "floor": 0.6885746474439154,
    "healthy": "Bf_s3",
    "note": "no clock-held ordering: not an RG1 criterion",
    "pass": null,
    "ratio_base": 0.6885746474439154,
    "ratio_with": 0.7762629208069886,
    "with": {
     "degenerate_total": 1.177738613753032,
   
```

### P4: PASS

```
{
 "headline": "max |clip shift| pp Bf_s2.side -2.58; max hand gap Bf_s2.side.hands 0.033",
 "rows": {
  "Bf_s2.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s2.diag": {
   "absorbed": 0.08866818442064406,
   "shift_pp": -2.108592514496574
  },
  "Bf_s2.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s2.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s2.side": {
   "absorbed": 0.10196467600660053,
   "shift_pp": -2.5775249460313345
  },
  "Bf_s2.side.hands": {
   "gap": 0.03287140589936638
  },
  "Bf_s2.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s3.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s3.diag": {
   "absorbed": 0.11806603554148187,
   "shift_pp": -1.9849468713105085
  },
  "Bf_s3.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s3.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "Bf_s3.side": {
   "absorbed": 0.09698519426457275,
   "shift_pp": -1.8100661843927974
  },
  "Bf_s3.side.hands": {
   "gap": 0.012866131175169857
  },
  "Bf_s3.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E3.back": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E3.diag": {
   "absorbed": 0.09029835474283565,
   "shift_pp": -1.5511237733459957
  },
  "E3.fwd": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E3.move_turn": {
   "absorbed": null,
   "shift_pp": 0.0
  },
  "E3.side": {
   "absorbed": 0.0711929079789757,
   "shift_pp": -1.3129913036873886
  },
  "E3.side.hands": {
   "gap": 0.007195097924670528
  },
  "E3.turn": {
   "absorbed": null,
   "shift_pp": 0.0
  }
 }
}
```
