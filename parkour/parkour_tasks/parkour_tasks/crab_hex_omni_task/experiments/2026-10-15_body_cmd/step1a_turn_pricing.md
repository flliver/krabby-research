# Step 1a — body-mode turn pricing on S1's measured yaw traces

Rule: CHARTER §3 (pinned 2026-10-15 before this read). Stamp `parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24`; window 50 steps; σ 0.05 rad; weights track 0.2 / tent 1.0 (ang-vel term 0 on turning rows).

**Verdict: FAIL** — translation w_min: share_step_in_place 0.452 > B 0.4; translation w_min: share_still 0.418 > B 0.4; move_turn w_min: share_step_in_place 0.523 > B 0.4; move_turn w_min: share_still 0.494 > B 0.4

Idealised still-stander share (wobble-free tracker): w_min 0.294, 0.08 0.144, ceiling 0.019, w_max 0.007; pinned w_min bound B = 0.4.

## translation traces (n = 500, windows 225500)

1-s Δyaw: std 0.0238 rad, |·| p50 0.0146, p90 0.0390; per-step wz std 0.094 rad/s. Straight rows (c = 0): wobble income 1.1120/step vs still 1.2000.

| level | c | tracker | step-in-place | still | share SIP | share still | premium vs still /step | premium /s |
|---|---|---|---|---|---|---|---|---|
| w_min | 0.050 | 0.1760 | 0.0795 | 0.0736 | 0.452 | 0.418 | 0.1025 | 5.12 |
| 0.08 | 0.080 | 0.2060 | 0.0452 | 0.0404 | 0.219 | 0.196 | 0.1657 | 8.28 |
| ceiling | 0.167 | 0.2930 | 0.0080 | 0.0071 | 0.027 | 0.024 | 0.2860 | 14.30 |
| w_max | 0.210 | 0.3360 | 0.0034 | 0.0030 | 0.010 | 0.009 | 0.3330 | 16.65 |

## move_turn traces (n = 200, windows 90200)

1-s Δyaw: std 0.0336 rad, |·| p50 0.0226, p90 0.0554; per-step wz std 0.109 rad/s. Straight rows (c = 0): wobble income 1.0820/step vs still 1.2000.

| level | c | tracker | step-in-place | still | share SIP | share still | premium vs still /step | premium /s |
|---|---|---|---|---|---|---|---|---|
| w_min | 0.050 | 0.1488 | 0.0778 | 0.0736 | 0.523 | 0.494 | 0.0752 | 3.76 |
| 0.08 | 0.080 | 0.1788 | 0.0456 | 0.0404 | 0.255 | 0.226 | 0.1384 | 6.92 |
| ceiling | 0.167 | 0.2658 | 0.0065 | 0.0071 | 0.024 | 0.027 | 0.2587 | 12.94 |
| w_max | 0.210 | 0.3088 | 0.0016 | 0.0030 | 0.005 | 0.010 | 0.3058 | 15.29 |

