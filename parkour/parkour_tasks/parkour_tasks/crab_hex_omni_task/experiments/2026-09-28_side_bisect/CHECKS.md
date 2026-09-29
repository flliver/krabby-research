# Phase B checks (plan steps 5–6), 2026-09-29 00:15–00:40

Script: `~/.claude/plans/side-speed-bisection.evidence/phaseB/checks.sh`, run from the main checkout at `aa2fed42`.
It runs everything serially, each process under a 48G memory cap.
Outputs: `parkour/outputs/omni_plan/side_bisect/` (summary in `checks_summary.txt`).
Classifier: `phaseB/side22_classify.py`. Its `hold_stats` is copied verbatim from the scoping `gate_stats.py`. On W's old
suite stamp it reproduces lat_l feet_ss 0.095 and lat_r 0.101.

**All checks PASS. No re-threshold; the pre-registered gate stands.**

## 5a. Knob smokes (`--smoke-stage`, 20 its from scratch on base 1b, scratch campaign copies)

| Smoke | Overlay | overlay_checks | Eval |
|---|---|---|---|
| C | MIRROR_IMAGES=fa, CMD_OBS=scalar, RSI_FRAC=0, SPEED_ELLIPSE=0.35:0.35:0.25 | 4/4 True | `side22_25` ran through the driver: `eval_status ok`, 300 episodes (3 waves) |
| R | CLOCK_COMBINE=sum, TRACK_SIGMA2=0.1, TRACK_L1_W=-1.0, YAW_PROGRESS_W=0 | 4/4 True | (dump only) |
| RC | C + R | 8/8 True | (dump only) |

**C's eval-path assertions**, on its `run_meta.json` (`gait_eval/omni_phases/knob_smoke_C/.../2026-09-29_04-17-17`):

| Assertion | Result |
|---|---|
| `env_vars` carries `KRABBY_OMNI_CMD_OBS=scalar` | yes |
| `KRABBY_OMNI_MIRROR_IMAGES` absent from the eval env | yes |
| `command_override_warning` present, deviation ≈ command | yes: slots 8–9 deviate by 0.25, the largest command |
| `schedule_state_max_deviation` | 0.0 (the command term itself is untouched) |
| trained categories passed | `[stand, side]` |
| obs width | 1149 |

The 20-it policy classes DEAD (feet_ss ≈ 0.00), which exercises the classifier's DEAD path.

## 5b. W harness check on `side22_25__omni` (W's recorded eval env)

| Hold | n | feet_ss median [IQR] | body_ss | lean04 | sweep |
|---|---|---|---|---|---|
| lat_l_022 | 100 | **0.096** [0.084, 0.104] | 0.095 | 0.249 | 0.034 |
| lat_l_025 | 100 | 0.103 [0.090, 0.110] | 0.100 | 0.266 | 0.035 |

**PASS:** 0.096 is inside 0.095 ± 0.01. W classes SLOW (leaned). The eval shows clock hidden, no command-override warning,
and a schedule deviation of 0.

## 5c. Sideways FAST pin on `side22_25__side` (sideways 1a, no env; pin = FAST + 0.017 = 0.172 / 0.192)

| Stamp | feet_ss 0.22 (n) | feet_ss 0.25 (n) | lean04 | sweep | Class |
|---|---|---|---|---|---|
| s3 model_3000 | **0.196** (89) | **0.193** (74) | 0.02–0.05 | 0.08 | FAST, pin PASS (thin at 0.25: +0.001) |
| s2 model_3000 | **0.201** (90) | **0.202** (78) | 0.00–0.04 | 0.09–0.10 | FAST, pin PASS |
| s3 model_4999 | 0.204 (99) | 0.206 (82) | 0.01–0.03 | 0.08–0.09 | FAST |
| s2 model_4999 | 0.218 (79) | 0.208 (65) | −0.01–0.00 | 0.11 | FAST |

The n below 100 are episodes that fell before 8 s into a hold ("short"). An episode that falls during the 0.22 hold never
reaches the 0.25 hold. On the sideways row the 0.25 hold follows the 0.22 hold without a new onset; on the omni row every
hold starts from a stand settle.

## 6. GPU-0: W on side holds at 0.15–0.35 m/s (report only; W's eval env with the lateral ellipse widened to 0.35)

| Command | feet_ss lat_l / lat_r | lean04 | lean_lvl | sweep |
|---|---|---|---|---|
| 0.15 | 0.093 / 0.089 | 0.19 | 0.20–0.21 | 0.035 |
| 0.22 | 0.097 / 0.098 | 0.25 | 0.25 | 0.033 |
| 0.30 | 0.105 / 0.106 | 0.26 | 0.26 | 0.036 |
| 0.35 | 0.110 / 0.112 | 0.28 | 0.27 | 0.038 |

The command was applied: every stamp shows `KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25`, `schedule_state_max_deviation`
0.0 and no command-override warning.

**Reading.**
- W's side speed is capped near 0.10 m/s whatever the command. It falls short even at 0.15.
- A faster command buys a deeper lean (0.19 → 0.28 m), not speed.
- The stance sweep stays at 0.03–0.04 m.

This is the attractor's signature at every speed, not a speed-dependent saturation.
