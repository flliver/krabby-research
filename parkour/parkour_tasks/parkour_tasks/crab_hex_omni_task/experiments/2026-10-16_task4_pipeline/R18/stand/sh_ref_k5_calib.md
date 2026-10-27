# Stand network -- gate calibration (PLAN.md step 2)

Generated 2026-10-19T22:48:59-04:00 by `scripts/sn_calib.py` (CPU, read-only; readers: phase-2 `p2_read.py` sha256 3f0159ad6aef, `p2s_stand.py` 80211bc5d6bf). Every number below is from an existing reader; the D5 validity rule and every other departure from p2_read's rules is listed under Deviations.

**Lands as listed: 15 / 15 expected cells yes, 0 mismatch(es), 0 pending.**

**Stamp resolution notes:**

- Z1: suite es1: lr/P2__Z1_s18/suite__omni/seed001 has no complete stamp; p2_read.z2_suite_stamp resolved the arm record's driver_stamp /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-07_omni_phase2/seed18_X16_p2r_A15pB/suite__omni/seed001/2026-10-10_00-20-08

## D6 gates: expected vs observed

Values: es1/es2 suffix = env seed; bold = a failing clause (its line in brackets); ? = pending; n/a = does not apply.

| gate | reference | expected | observed | lands as listed | values |
|---|---|---|---|---|---|
| G1 STILL | REFS | FAIL | FAIL | yes | **steps es1 2.833 (line 0.5)**; **clear es1 0.1349 (line 0.05)**; **offgnd es1 0.3382 (line 0.1)**; hover es1 0; **taps es1 2.833 (line 1)**; **KS es1 1.718 (line 0.5)**; **steps es2 2.833 (line 0.5)**; **clear es2 0.1331 (line 0.05)**; **offgnd es2 0.3342 (line 0.1)**; hover es2 0; **taps es2 2.667 (line 1)**; **KS es2 1.716 (line 0.5)** |
| G1 STILL | HOLD | PASS | PASS | yes | steps es1 0; clear es1 1.00e-05; offgnd es1 0; hover es1 0; taps es1 0; KS es1 0.09809; steps es2 0; clear es2 1.00e-05; offgnd es2 0; hover es2 0; taps es2 0; KS es2 0.1043 |
| G1 STILL | R | PASS | PASS | yes | steps es1 0; clear es1 0; offgnd es1 0; hover es1 0; taps es1 0; KS es1 0.09834; steps es2 0; clear es2 0; offgnd es2 0; hover es2 0; taps es2 0; KS es2 0.1046 |
| G1 STILL | Z1 | -- | PASS | reported | steps es1 0; clear es1 0.0013; offgnd es1 0; hover es1 0; taps es1 0.08333; KS es1 0.1803; steps es2 0.08333; clear es2 0.00123; offgnd es2 0; hover es2 0; taps es2 0.08333; KS es2 0.1778 |
| G1 STILL | Z2d | -- | PASS | reported | steps es1 0; clear es1 8.00e-05; offgnd es1 0; hover es1 0; taps es1 0; KS es1 0.1349; steps es2 0; clear es2 9.00e-05; offgnd es2 0; hover es2 0; taps es2 0; KS es2 0.1432 |
| G1 STILL | ST2M | FAIL | FAIL | yes | steps es1 0.1667; clear es1 0.00533; offgnd es1 0; hover es1 0; **taps es1 16.08 (line 1)**; KS es1 0.4046; steps es2 0.1667; clear es2 0.0054; offgnd es2 0; hover es2 0; **taps es2 15.33 (line 1)**; KS es2 0.4035 |
| G1 STILL | C0R | -- | FAIL | reported | **steps es1 2.667 (line 0.5)**; **clear es1 0.1126 (line 0.05)**; **offgnd es1 0.3409 (line 0.1)**; hover es1 0; **taps es1 2.083 (line 1)**; **KS es1 1.654 (line 0.5)**; **steps es2 2.667 (line 0.5)**; **clear es2 0.1131 (line 0.05)**; **offgnd es2 0.3386 (line 0.1)**; hover es2 0; **taps es2 2 (line 1)**; **KS es2 1.657 (line 0.5)** |
| G1 STILL | SH_T4R18_sh_ref_s17z | -- | INVALID | reported | steps es1 0; clear es1 0; offgnd es1 0; hover es1 0; taps es1 0; KS es1 0.09899; steps es2 0; clear es2 0; offgnd es2 0; hover es2 0; taps es2 0; KS es2 0.1043 |
| G2 HOMED | REFS | FAIL | FAIL | yes | **homed es1 0 (line 0.9)**; **settled es1 0 (line 0.9)**; **settle med es1 -- (line 3.92)**; **post steps es1 -- (line 0.5)**; **post taps es1 -- (line 1)**; **post KS es1 -- (line 0.5)**; **homed es2 0 (line 0.9)**; **settled es2 0 (line 0.9)**; **settle med es2 -- (line 3.96)**; **post steps es2 -- (line 0.5)**; **post taps es2 -- (line 1)**; **post KS es2 -- (line 0.5)** |
| G2 HOMED | HOLD | FAIL | FAIL | yes | **homed es1 0 (line 0.9)**; **settled es1 0 (line 0.9)**; **settle med es1 -- (line 3.92)**; **post steps es1 -- (line 0.5)**; **post taps es1 -- (line 1)**; **post KS es1 -- (line 0.5)**; **homed es2 0 (line 0.9)**; **settled es2 0 (line 0.9)**; **settle med es2 -- (line 3.96)**; **post steps es2 -- (line 0.5)**; **post taps es2 -- (line 1)**; **post KS es2 -- (line 0.5)** |
| G2 HOMED | R | PASS | PASS | yes | homed es1 0.97; settled es1 0.97; settle med es1 2.92; post steps es1 0.113; post taps es1 0; post KS es1 0.1493; homed es2 0.9767; settled es2 0.9767; settle med es2 2.96; post steps es2 0.1122; post taps es2 0; post KS es2 0.1466 |
| G2 HOMED | Z1 | FAIL | FAIL | yes | **homed es1 0 (line 0.9)**; **settled es1 0 (line 0.9)**; **settle med es1 -- (line 3.92)**; **post steps es1 -- (line 0.5)**; **post taps es1 -- (line 1)**; **post KS es1 -- (line 0.5)**; **homed es2 0 (line 0.9)**; **settled es2 0 (line 0.9)**; **settle med es2 -- (line 3.96)**; **post steps es2 -- (line 0.5)**; **post taps es2 -- (line 1)**; **post KS es2 -- (line 0.5)** |
| G2 HOMED | Z2d | FAIL | FAIL | yes | **homed es1 0.01667 (line 0.9)**; **settled es1 0.1733 (line 0.9)**; settle med es1 2.78; **post steps es1 -- (line 0.5)**; **post taps es1 -- (line 1)**; **post KS es1 -- (line 0.5)**; **homed es2 0.02333 (line 0.9)**; **settled es2 0.1867 (line 0.9)**; settle med es2 2.2; **post steps es2 -- (line 0.5)**; **post taps es2 -- (line 1)**; **post KS es2 -- (line 0.5)** |
| G2 HOMED | ST2M | -- | FAIL | reported | **homed es1 0 (line 0.9)**; **settled es1 0 (line 0.9)**; **settle med es1 -- (line 3.92)**; **post steps es1 -- (line 0.5)**; **post taps es1 -- (line 1)**; **post KS es1 -- (line 0.5)**; **homed es2 0 (line 0.9)**; **settled es2 0 (line 0.9)**; **settle med es2 -- (line 3.96)**; **post steps es2 -- (line 0.5)**; **post taps es2 -- (line 1)**; **post KS es2 -- (line 0.5)** |
| G2 HOMED | C0R | -- | FAIL | reported | **homed es1 0 (line 0.9)**; **settled es1 0 (line 0.9)**; **settle med es1 -- (line 3.92)**; **post steps es1 -- (line 0.5)**; **post taps es1 -- (line 1)**; **post KS es1 -- (line 0.5)**; **homed es2 0 (line 0.9)**; **settled es2 0 (line 0.9)**; **settle med es2 -- (line 3.96)**; **post steps es2 -- (line 0.5)**; **post taps es2 -- (line 1)**; **post KS es2 -- (line 0.5)** |
| G2 HOMED | SH_T4R18_sh_ref_s17z | -- | INVALID | reported | homed es1 1; settled es1 1; settle med es1 2.45; post steps es1 0.102; post taps es1 0; post KS es1 0.1432; homed es2 0.99; settled es2 0.99; settle med es2 2.52; post steps es2 0.0999; post taps es2 0; post KS es2 0.1489 |
| G3 FALLS | REFS | -- | PASS | reported | ws falls es1 0; suite falls es1 0; exit falls es1 n/a; ws falls es2 0; suite falls es2 0; exit falls es2 n/a |
| G3 FALLS | HOLD | PASS | PASS | yes | ws falls es1 0; suite falls es1 0; exit falls es1 0; ws falls es2 0; suite falls es2 0; exit falls es2 0 |
| G3 FALLS | R | PASS | PASS | yes | ws falls es1 0; suite falls es1 0; exit falls es1 0; ws falls es2 0; suite falls es2 0; exit falls es2 0 |
| G3 FALLS | Z1 | -- | PASS | reported | ws falls es1 0; suite falls es1 0; exit falls es1 n/a; ws falls es2 0; suite falls es2 0; exit falls es2 n/a |
| G3 FALLS | Z2d | -- | PASS | reported | ws falls es1 0; suite falls es1 0; exit falls es1 n/a; ws falls es2 0; suite falls es2 0; exit falls es2 n/a |
| G3 FALLS | ST2M | -- | PASS | reported | ws falls es1 0; suite falls es1 0; exit falls es1 0; ws falls es2 0; suite falls es2 0; exit falls es2 0 |
| G3 FALLS | C0R | -- | PASS | reported | ws falls es1 0; suite falls es1 0; exit falls es1 n/a; ws falls es2 0; suite falls es2 0; exit falls es2 n/a |
| G3 FALLS | SH_T4R18_sh_ref_s17z | -- | INVALID | reported | ws falls es1 0; suite falls es1 0; exit falls es1 0; ws falls es2 0; suite falls es2 0; exit falls es2 0 |
| G4 WALLS | REFS | -- | PASS | reported | field_hard 0.8125; field_turn es1 0.99; stand-fall FT es1 0; stand-fall FH 0 |
| G4 WALLS | HOLD | PASS | PASS | yes | field_hard 0.81; field_turn es1 0.99; stand-fall FT es1 0; stand-fall FH 0 |
| G4 WALLS | R | PASS | PASS | yes | field_hard 0.825; field_turn es1 1; stand-fall FT es1 0; stand-fall FH 0 |
| G4 WALLS | Z1 | -- | PASS | reported | field_hard 0.83; field_turn es1 1; stand-fall FT es1 0; stand-fall FH 0 |
| G4 WALLS | Z2d | -- | PASS | reported | field_hard 0.8225; field_turn es1 0.99; stand-fall FT es1 0; stand-fall FH 0 |
| G4 WALLS | ST2M | -- | PASS | reported | field_hard 0.835; field_turn es1 0.98; stand-fall FT es1 0; stand-fall FH 0 |
| G4 WALLS | C0R | -- | PASS | reported | field_hard 0.785; field_turn es1 0.99; stand-fall FT es1 0.01; stand-fall FH 0 |
| G4 WALLS | SH_T4R18_sh_ref_s17z | -- | INVALID | reported | field_hard 0.825; field_turn es1 1; stand-fall FT es1 0; stand-fall FH 0 |
| G5 HANDOVER | REFS | -- | PASS | reported | onset es1 0.14; stalls es1 0; onset es2 0.14; stalls es2 0; onset pooled 0.14; height suite es1 0; height walkstop es1 0; height suite es2 0; height walkstop es2 0 |
| G5 HANDOVER | HOLD | PASS | PASS | yes | onset es1 0.12; stalls es1 0; onset es2 0.12; stalls es2 0; onset pooled 0.12; height suite es1 0.00321; height walkstop es1 0.00458; height suite es2 0.0017; height walkstop es2 0.00408 |
| G5 HANDOVER | R | PASS | PASS | yes | onset es1 0.12; stalls es1 0; onset es2 0.12; stalls es2 0; onset pooled 0.12; height suite es1 0.00321; height walkstop es1 0.00666; height suite es2 0.0017; height walkstop es2 0.00572 |
| G5 HANDOVER | Z1 | -- | PASS | reported | onset es1 0.12; stalls es1 0; onset es2 0.12; stalls es2 0; onset pooled 0.12; height suite es1 0.00412; height walkstop es1 0.00527; height suite es2 0.00271; height walkstop es2 0.00471 |
| G5 HANDOVER | Z2d | -- | PASS | reported | onset es1 0.18; stalls es1 0; onset es2 0.18; stalls es2 0; onset pooled 0.18; height suite es1 0.00557; height walkstop es1 0.01054; height suite es2 0.00397; height walkstop es2 0.01026 |
| G5 HANDOVER | ST2M | -- | PASS | reported | onset es1 0.14; stalls es1 0; onset es2 0.14; stalls es2 0; onset pooled 0.14; height suite es1 0.00811; height walkstop es1 0.01008; height suite es2 0.00585; height walkstop es2 0.00993 |
| G5 HANDOVER | C0R | -- | PASS | reported | onset es1 0.14; stalls es1 0; onset es2 0.16; stalls es2 0; onset pooled 0.16; height suite es1 0.0051; height walkstop es1 0.00542; height suite es2 0.00503; height walkstop es2 0.00557 |
| G5 HANDOVER | SH_T4R18_sh_ref_s17z | -- | INVALID | reported | onset es1 0.12; stalls es1 0; onset es2 0.12; stalls es2 0; onset pooled 0.12; height suite es1 0.0032; height walkstop es1 0.00624; height suite es2 0.00169; height walkstop es2 0.00575 |

## Identity check: SN_RSTg_s18 vs P2R_RSTg_s18, field_hard es1, raw keys bitwise

**PASS** -- every raw key of every episode bitwise equal

- P2R_RSTg_s18: launcher b5ba0c63, restance module 7006dff1, code_sha 07e02578, eval_sha 1d021625, dirty False; R knob lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3
- SN_RSTg_s18: launcher 59539301, restance module 7a16b8d1, code_sha 14301cd2, eval_sha 1d021625, dirty False; R knob lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3
- stamps: reference `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/P2R_RSTg_s18/field_hard__omni/seed001/2026-10-10_22-41-00`, re-run `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SN_RSTg_s18/field_hard__omni/seed001/2026-10-12_03-19-22`
- stand bank (`p2r_stand_bank.npz`, not a raw key): reference 47500 rows kept of 47500 seen (logged bank_max [100000]), re-run 20000 of 47500 (logged bank_max [20000]; launcher default 20000) -- the banks DIFFER in size; the re-run's rows are a subset of the reference's by (episode, t), obs, target, branch, proxy, hold_label, category bitwise equal on every shared row. The size difference is the bank CAP, not the launcher sha: the probe ran with bank_max [100000] (P2R_BANK_MAX set for it, P2R_Z2C_PLAN.md step 3), the re-run with [20000] (sn_calib_rows.sh sets no cap: the launcher default 20000), so the re-run's bank is a reservoir sample of the same rows; the stand-bank code did not move with the launcher sha. No later step of this campaign reads a stand bank: D3 trains by online DAgger on the states the student visits (its own capped buffer), not by offline BC on p2r banks; the gates read raws only.

## R's walkstop onset (no earlier read had it)

| | es1 | es2 | pooled |
|---|---|---|---|
| R median onset (s) | 0.12 | 0.12 | 0.12 |
| censored | 0 | 0 | |
| stalled (reached, censored) / reached | 0 / 200 | 0 / 200 | |
| REFS / HOLD pooled (s) | | | 0.14 / 0.12 |

Per pair: es1 {"lat_l": {"median_s": 0.12, "n": 100}, "back": {"median_s": 0.12, "n": 100}}; es2 {"lat_l": {"median_s": 0.12, "n": 100}, "back": {"median_s": 0.12, "n": 100}}. p2s_stand.episode_onset_latency via p2_read.walkstop_read (stand -> lat_l / back, 5-step trailing mean reaches 0.5 x |cmd| and holds 0.5 s); pooled = p2_read.p2s_onset.

## Negative-control coverage (reported, no verdict changes)

A gate no reference fails is unexercised as a negative control: the calibration shows its known-good pass but never shows its line can bind, so its sensitivity rests on the line definitions, not on an observed fail. D6 lists no known-bad for G3-G5, so this is not a mismatch.

- **G1 STILL**: failed by REFS, ST2M, C0R (D6 known-bad: REFS, ST2M)
- **G2 HOMED**: failed by REFS, HOLD, Z1, Z2d, ST2M, C0R (D6 known-bad: HOLD, Z1, Z2d, REFS)
- **G3 FALLS**: **no reference fails -- unexercised as a negative control** (D6 lists no known-bad)
  - ws falls: references read 0 .. 0 over 16 clause reads; tightest pass REFS 0 vs line 1 (margin 1)
  - suite falls: references read 0 .. 0 over 16 clause reads; tightest pass REFS 0 vs line 0 (margin 0)
  - exit falls: references read 0 .. 0 over 8 clause reads; tightest pass HOLD 0 vs line 1 (margin 1)
- **G4 WALLS**: **no reference fails -- unexercised as a negative control** (D6 lists no known-bad)
  - field_hard: references read 0.785 .. 0.835 over 8 clause reads; tightest pass C0R 0.785 vs line 0.7558 (margin 0.0292)
  - field_turn: references read 0.98 .. 1 over 8 clause reads; tightest pass ST2M 0.98 vs line 0.94 (margin 0.04)
  - stand-fall FT: references read 0 .. 0.01 over 8 clause reads; tightest pass REFS 0 vs line 0 (margin 0)
  - stand-fall FH: references read 0 .. 0 over 8 clause reads; tightest pass REFS 0 vs line 0 (margin 0)
- **G5 HANDOVER**: **no reference fails -- unexercised as a negative control** (D6 lists no known-bad)
  - onset: references read 0.12 .. 0.18 over 16 clause reads; tightest pass Z2d 0.18 vs line 0.44 (margin 0.26)
  - stalls: references read 0 .. 0 over 16 clause reads; tightest pass REFS 0 vs line 1 (margin 1)
  - onset pooled: references read 0.12 .. 0.18 over 8 clause reads; tightest pass Z2d 0.18 vs line 0.44 (margin 0.26)
  - height suite: references read 0 .. 0.00811 over 16 clause reads; tightest pass ST2M 0.00811 vs line 0.03 (margin 0.0219)
  - height walkstop: references read 0 .. 0.01054 over 16 clause reads; tightest pass Z2d 0.01054 vs line 0.03 (margin 0.0195)

Margins are |line - value| on passing clauses. A margin of 0 at value 0 means different things: `suite falls` is `== 0` (one fall fails), but the G4 stand-fall lines are HOLD + 2 SE of the POOLED difference, which is 0 only while head and HOLD both read 0 falls and moves with the head's own count (one fall in 100 vs HOLD's 0 draws a line of about 0.02 and passes), so a 0 margin there is not a tight pass.

## Pending

- none

## INVALID cells

- G1 x SH_T4R18_sh_ref_s17z: [SH_T4R18_sh_ref_s17z] suite@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('cbaf67974a7ebfb644dd65584f3294e31bcaffb7ac8a04750dc55709f928e8b3', '8c8d22f078cb40c68791b96f1ee40c9d3e901593034e46ad34e537faad2129fd', False) vs ('ae892cec7f2b782b94202b31dab35816c38337e16427bca0aa9f51065a4c6214', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False)); [SH_T4R18_sh_ref_s17
- G2 x SH_T4R18_sh_ref_s17z: [SH_T4R18_sh_ref_s17z] walkstop__omni@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('cbaf67974a7ebfb644dd65584f3294e31bcaffb7ac8a04750dc55709f928e8b3', '8c8d22f078cb40c68791b96f1ee40c9d3e901593034e46ad34e537faad2129fd', False) vs ('ae892cec7f2b782b94202b31dab35816c38337e16427bca0aa9f51065a4c6214', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False)); [SH_T4R18_s
- G3 x SH_T4R18_sh_ref_s17z: [SH_T4R18_sh_ref_s17z] walkstop__omni@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('cbaf67974a7ebfb644dd65584f3294e31bcaffb7ac8a04750dc55709f928e8b3', '8c8d22f078cb40c68791b96f1ee40c9d3e901593034e46ad34e537faad2129fd', False) vs ('ae892cec7f2b782b94202b31dab35816c38337e16427bca0aa9f51065a4c6214', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False)); [SH_T4R18_s
- G4 x SH_T4R18_sh_ref_s17z: [SH_T4R18_sh_ref_s17z] field_turn__omni@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('cbaf67974a7ebfb644dd65584f3294e31bcaffb7ac8a04750dc55709f928e8b3', '8c8d22f078cb40c68791b96f1ee40c9d3e901593034e46ad34e537faad2129fd', False) vs ('ae892cec7f2b782b94202b31dab35816c38337e16427bca0aa9f51065a4c6214', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False)); [SH_T4R18
- G5 x SH_T4R18_sh_ref_s17z: [SH_T4R18_sh_ref_s17z] walkstop__omni@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('cbaf67974a7ebfb644dd65584f3294e31bcaffb7ac8a04750dc55709f928e8b3', '8c8d22f078cb40c68791b96f1ee40c9d3e901593034e46ad34e537faad2129fd', False) vs ('ae892cec7f2b782b94202b31dab35816c38337e16427bca0aa9f51065a4c6214', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False)); [SH_T4R18_s

## Plan's listed numbers vs computed (reported)

| what | listed | computed |
|---|---|---|
| REFS suite stand steps/s es1 / es2 | 2.83 | 2.833, 2.833 |
| ST2M suite stand taps/s es1 / es2 | 16 | 16.08, 15.33 |
| HOLD field_hard pooled n=400 | 0.81 | 0.81 |
| HOLD walkstop onset pooled (s) | 0.12 | 0.12 |
| REFS walkstop onset pooled (s) | 0.14 | 0.14 |
| R median-settle line es1 / es2 (s) | [3.92, 3.96] | 3.92, 3.96 |
| R walkstop homed es1 / es2 | [0.97, 0.977] | 0.97, 0.9767 |

## Reported only (not gated)

| reference | seed | six-down suite / walkstop | hip yaw mean / max walkstop (deg) | settle median walkstop (s) | KS suite / walkstop / post-settle |
|---|---|---|---|---|---|
| REFS | es1 | 0 / 0 | 3.52 / 16.6 | -- | 1.72 / 1.72 / -- |
| REFS | es2 | 0 / 0 | 3.57 / 19 | -- | 1.72 / 1.72 / -- |
| HOLD | es1 | 0.29 / 0.347 | 5.96 / 22.5 | -- | 0.0981 / 0.158 / -- |
| HOLD | es2 | 0.2 / 0.337 | 6.14 / 23.9 | -- | 0.104 / 0.164 / -- |
| R | es1 | 0.29 / 0.37 | 0.4 / 10.3 | 2.92 | 0.0983 / 0.228 / 0.149 |
| R | es2 | 0.2 / 0.32 | 0.361 / 17.2 | 2.96 | 0.105 / 0.234 / 0.147 |
| Z1 | es1 | 0.09 / 0.217 | 5.97 / 23.6 | -- | 0.18 / 0.214 / -- |
| Z1 | es2 | 0.11 / 0.243 | 5.85 / 23 | -- | 0.178 / 0.225 / -- |
| Z2d | es1 | 0.33 / 0.577 | 1.98 / 25.3 | 2.78 | 0.135 / 0.204 / -- |
| Z2d | es2 | 0.34 / 0.517 | 1.89 / 25.3 | 2.2 | 0.143 / 0.198 / -- |
| ST2M | es1 | 0 / 0 | 9.66 / 26.8 | -- | 0.405 / 0.489 / -- |
| ST2M | es2 | 0 / 0 | 9.65 / 26.2 | -- | 0.404 / 0.489 / -- |
| C0R | es1 | 0 / 0 | 3.31 / 16.4 | -- | 1.65 / 1.66 / -- |
| C0R | es2 | 0 / 0 | 3.25 / 16.2 | -- | 1.66 / 1.66 / -- |
| SH_T4R18_sh_ref_s17z | es1 | 0.29 / 0.33 | 0.501 / 7 | 2.45 | 0.099 / 0.208 / 0.143 |
| SH_T4R18_sh_ref_s17z | es2 | 0.2 / 0.417 | 0.584 / 8.33 | 2.52 | 0.104 / 0.213 / 0.149 |

## Validity

- **REFS** (REFS_S1_s18, rule `obs0`): valid on every stamp read; p2_read Z2-env reasons bypassed on 9 stamp(s) (D5), e.g. `REFS_S1_s18 suite@1: KRABBY_OMNI_STAND_OBS='0', want 'clock'`
- **HOLD** (P2R_HOLD_s18, rule `tier1`): valid on every stamp read; p2_read Z2-env reasons bypassed on 9 stamp(s) (D5), e.g. `P2R_HOLD_s18 suite@1: KRABBY_OMNI_STAND_OBS='0', want 'clock'`
- **R** (P2R_RSTg_s18 + SN_RSTg_s18 (field_hard), rule `rst`): valid on every stamp read
  - field_hard source: {"name": "SN_RSTg_s18", "seeds": [1, 2, 3, 4], "partial": false}
- **Z1** (P2__Z1_s18, rule `z`): valid on every stamp read
  - note: suite es1: lr/P2__Z1_s18/suite__omni/seed001 has no complete stamp; p2_read.z2_suite_stamp resolved the arm record's driver_stamp /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-07_omni_phase2/seed18_X16_p2r_A15pB/suite__omni/seed001/2026-10-10_00-20-08
- **Z2d** (P2__Z2dR1_s18, rule `z`): valid on every stamp read
- **ST2M** (P2R_ST2M_s18, rule `tier1`): valid on every stamp read; p2_read Z2-env reasons bypassed on 9 stamp(s) (D5), e.g. `P2R_ST2M_s18 suite@1: KRABBY_OMNI_STAND_OBS='0', want 'clock'`
- **C0R** (C0R_s18, rule `obs0`): valid on every stamp read; p2_read Z2-env reasons bypassed on 9 stamp(s) (D5), e.g. `C0R_s18 suite@1: KRABBY_OMNI_STAND_OBS='0', want 'clock'`
- **SH_T4R18_sh_ref_s17z** (SH_T4R18_sh_ref_s17z, rule `stand_net`): INVALID: {"suite@1": ["suite@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('cbaf67974a7ebfb644dd65584f3294e31bcaffb7ac8a04750dc55709f928e8b3', '8c8d22f078cb40c68791b96f1ee40c9d3e901593034e46ad34e537faad2129fd', False) vs ('ae892cec7f2b782b94202b31dab35816c38337e16427bca0aa9f51065a4c6214', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False))", "SH_T4R18_sh_ref_s17z suite@; p2_read Z2-env reasons bypassed on 9 stamp(s) (D5), e.g. `SH_T4R18_sh_ref_s17z suite@1: KRABBY_OMNI_STAND_OBS='0', want 'clock'`

## Deviations from p2_read's rules

1. VALIDITY (D5): REFS_S1_s18, P2R_HOLD_s18, P2R_ST2M_s18 and C0R_s18 run STAND_OBS=0 (col 15 = 0). p2_read's Z2 env rule (z2_env_reasons: 'STAND_OBS=clock or INVALID') would read every one of their stamps INVALID; sn_calib replaces it for them with the D5 obs-parity rule (STAND_OBS exactly '0', absent fails closed; proxy off; env-level STAND_MASK off) + p2_read.invalid_reasons vs REFS (ENV4, row env block, bridged code-state sidecar) + walkstop_flat_fails (+ p2r_mode_fails variant hold / st2 driving REFS's S1 for HOLD / ST2M). The p2_read reasons bypassed are recorded per stamp in calib.json rows.<role>.bypassed_p2_read_reasons.
2. VALIDITY (D5, campaign heads; --head): STAND_OBS exactly '0' + run_meta.p2r_mode.variant == 'stand_net' + p2r_mode.s1_checkpoint_sha256 == REFS's + proxy / env mask off + invalid_reasons vs REFS, fail closed -- in place of p2_read's Z2 'STAND_OBS=clock or INVALID'. Plus (verifier fix): every stamp names its stand checkpoint sha (p2r_mode.specialist.sha256, the p2r launcher's second-network record) and its mask flag (p2r_mode.stand_mask, 0 / 1), and ALL of a head's stamps carry ONE sha and ONE mask -- A-mask and A-full share the variant and S1's sha, so a NAME mixing both arms or a re-trained checkpoint would otherwise read valid. These two field names are a contract the step-3 launcher must write (constants STAND_NET_CKPT_KEY / STAND_NET_MASK_KEY); a launcher that names them otherwise reads INVALID on every stamp until they are aligned. No stand_net stamp exists yet: exercised by --selftest only.
3. VALIDITY kept from p2_read (not a deviation, recorded): R stamps (P2R_RSTg_s18, SN_RSTg_s18) use p2_read.rst_invalid_reasons unchanged (STAND_OBS=clock, proxy off, p2r_mode variant restance on REFS's S1, raw stand bit; PLAN 'Left for implementation': R keeps the clock rule); Z1 / Z2d use p2_read's Z2 head rule (z_clock_invalid_reasons + z2_invalid_reasons incl. the mask rule and raw stand bit) + invalid_reasons vs REFS.
4. READ PATH: REFS / HOLD (and ST2M, C0R) are calibrated by calling z2_stand_gates / still_clauses / z2_field_vs_hold / z2_height_vs_refs directly on their summaries (p2_read.z2_light_point reads), not through p2_read's --p2r-z2 path, which reads them INVALID and reads one Z head only. Z1 / Z2d are read the same way: the --p2r-z2 ladder rungs that are not D6 gates (HAND LOSS, DOSE, FLAT FAIL, attribution) are not run.
5. G2 / G3 are subsets of p2_read.z2_stand_gates' clauses (relative=True, probe P2R_RSTg_s18): its STILL clauses are G1's (cross-checked equal to still_clauses), its walkstop-falls clause is G3's, the rest is G2. Its own combined verdict (which also needs the suite Z2 read for the reported six-down) is recorded per seed.
6. G3 exit clause: p2_read.p2r_hold_reads returns n_episodes 0 (not None, contrary to its docstring) when no episode carries p2r_stand_mode; sn_calib maps n_episodes == 0 or < the raw episode count to None -> INVALID for a stamp with run_meta.p2r_mode (and for a stand_net head), and to n/a for a plain single-network eval without p2r_mode (REFS, Z1, Z2d, C0R: no stand-mode switch, no exit). R's line source must read ok.
7. G3 exit clause DENOMINATOR (verifier fix; the plan says only 'raw count per env seed (200 exits)'): a raw count is only comparable on R's set, so per env seed the head must read R's raw episode count and schedule (stand -> move boundaries per episode, from run_meta.hold_labels), R must have made every scheduled exit (R: 200 of 200), and the head may make no exit beyond the schedule (D5 switch contract) -- any of these broken -> INVALID. A scheduled exit the head's episode never made FAILS the clause closed, unless that episode fell inside a stand-mode segment (p2r_hold_reads' falls_in_stand_mode definition): a stand fall is the walkstop stand-falls clause's, which keeps its pre-registered max(1, Z1's) per 300 floor instead of being double-counted here. Exits, design exits, excused / unexcused missing exits are in calib.json rows.<role>.exit_reads.<seed>.audit.
8. G4 field_hard line: p2s_verdict / p2r_tier1_verdict's inline OBSTACLE LOSS expression restated with p2_read.se_need, P2S_OBST_FH 0.05, P2B_SE_K 2 (fh >= REFS - max(0.05, 2 SE)); pooled with p2_read.pooled_seeds.
9. G4 field_turn band: p2_read has one -- p2s_cost's inline band_ft = max(binom_se(REFS, n), COMP_BAND_FLOOR 0.04) -- restated (line REFS - max(P2S_FT_FLOOR 0.05, band)); es1 only (REFS and R have no field_turn es2).
10. G4 stand-fall clause: p2_read.z2_field_vs_hold on field_turn es1 and on a pooled field_hard es1-4 pseudo-row (n = 400 each side; R PARTIAL: es1, n = 100), not the RSTg precedent's field_hard es1 row (whose result is reported); its completion-vs-HOLD clauses are reported, not gated (D6 G4's completion lines are vs REFS).
11. G4 for R: SN_RSTg_s18 field_hard es1-4 when all four are complete, else P2R_RSTg_s18 es1 (n = 100) and the field_hard clauses PARTIAL (provisional verdict shown); suite / walkstop / field_turn stay P2R_RSTg_s18's.
12. G5 onset: read per env seed (D6 'each at es1 and es2': walkstop_read onset median vs REFS same seed) AND pooled es1+es2 (p2s_onset, p2s_cost's own statistic), line REFS + max(0.3, p2s_stand_bands onset band); all must hold. A present walkstop read whose every onset is censored (median None) fails closed.
13. G5 onset STALLS (verifier fix; not in D6 or p2s_cost): walkstop_read's median and p2s_onset drop censored onsets, so a head after which S1 never gets going on most handovers would pass on the few that do -- the 'Out: S1 sees the student's actions' failure D5 says onset measures. Added per env seed: stalled onsets (a stand -> move hold the episode reached whose p2s_stand.episode_onset_latency is None) <= REFS's same-seed count + 1 (REFS / R / HOLD: 0 of 200 each), head and REFS on the same episode count and onset pairs (else INVALID). Onsets of move holds the episode never reached (it ended first) are not stalls: that is a fall, G3's. Counting censored onsets as +inf in the median was the alternative: not adopted, it only binds once half the onsets stall. The same hole remains in p2s_cost (phase-2 code, not edited).
14. G5 body height: p2_read.z2_height_vs_refs (suite stand AND walkstop stands, per seed, 0.03 m) -- the Z2 premise version -- not p2s_cost's es1/es2-mean suite height.
15. VERDICT combination: an evaluated failing clause is decisive (FAIL) even when another clause is PENDING; p2_read's z2_stand_gates / p2r_rst_gate return PENDING in that case. A MISSING stamp is PENDING, never a FAIL. A head's COMPLETE stamp (p2_read.newest_stamp returns complete stamps only) whose read or statistic is None fails its clause closed, and a LINE source's (REFS / HOLD / R / Z1) complete stamp with no readable statistic makes the gate INVALID (the line cannot be drawn) -- on every gate (verifier fix: G4 read such a stamp PENDING). p2_read keys PENDING on the summary, not the stamp: a stamp whose summary is None reads PENDING there; here it does not, because a complete stamp with no statistic is a broken eval, not a pending one.
16. C0R_s18: every gate computed and reported (no expectation); D6 names it only as G4's attribution floor.

Full per-clause values, lines, stamp paths and code states: `calib.json`.
