# Step 1a — body-mode turn pricing on S1's measured yaw traces

Rule: CHARTER §3 (pinned 2026-10-15 before this read). Stamp `parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24`; window 100 steps (2.0 s); σ 0.1 rad; weights track 0.2 / tent 2.0 (ang-vel term 0 on turning rows).

**Verdict: PASS**

Idealised still-stander share (wobble-free tracker): w_min 0.245, 0.08 0.112, ceiling 0.013, w_max 0.005; pinned w_min bound B = 0.4.

## translation traces (n = 500, windows 200500)

2-s Δyaw: std 0.0281 rad, |·| p50 0.0175, p90 0.0458; per-step wz std 0.094 rad/s. Straight rows (c = 0): wobble income 1.1312/step vs still 1.2000.

| level | c | tracker | step-in-place | still | share SIP | share still | premium vs still /step | premium /s |
|---|---|---|---|---|---|---|---|---|
| w_min | 0.050 | 0.2417 | 0.0765 | 0.0736 | 0.316 | 0.304 | 0.1681 | 8.41 |
| 0.08 | 0.080 | 0.3017 | 0.0420 | 0.0404 | 0.139 | 0.134 | 0.2613 | 13.07 |
| ceiling | 0.167 | 0.4757 | 0.0074 | 0.0071 | 0.016 | 0.015 | 0.4686 | 23.43 |
| w_max | 0.210 | 0.5617 | 0.0031 | 0.0030 | 0.006 | 0.005 | 0.5587 | 27.93 |

## move_turn traces (n = 200, windows 80200)

2-s Δyaw: std 0.0382 rad, |·| p50 0.0258, p90 0.0626; per-step wz std 0.109 rad/s. Straight rows (c = 0): wobble income 1.1079/step vs still 1.2000.

| level | c | tracker | step-in-place | still | share SIP | share still | premium vs still /step | premium /s |
|---|---|---|---|---|---|---|---|---|
| w_min | 0.050 | 0.2207 | 0.0723 | 0.0736 | 0.327 | 0.333 | 0.1471 | 7.35 |
| 0.08 | 0.080 | 0.2807 | 0.0383 | 0.0404 | 0.137 | 0.144 | 0.2403 | 12.01 |
| ceiling | 0.167 | 0.4547 | 0.0037 | 0.0071 | 0.008 | 0.016 | 0.4476 | 22.38 |
| w_max | 0.210 | 0.5407 | -0.0005 | 0.0030 | -0.001 | 0.006 | 0.5377 | 26.88 |

