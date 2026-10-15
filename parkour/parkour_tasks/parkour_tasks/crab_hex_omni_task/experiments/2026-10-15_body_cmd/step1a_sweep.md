# Step 1a sensitivity — non-rotating share of the yaw-term income

Share = non-rotating trace income / healthy tracker income (lower = slow turns better priced). Same S1 traces as step1a_turn_pricing.md.

| variant | c | transl. share still | transl. share SIP | move_turn share still | move_turn share SIP | transl. premium /s |
|---|---|---|---|---|---|---|
| as planned: 1 s, sigma 0.05, tent 1.0, c 0.05 | 0.050 | 0.418 | 0.452 | 0.494 | 0.523 | 5.12 |
| w_min 0.06 | 0.060 | 0.324 | 0.357 | 0.379 | 0.418 | 6.29 |
| w_min 0.07 | 0.070 | 0.252 | 0.280 | 0.292 | 0.328 | 7.34 |
| 2 s window, sigma 0.10 | 0.050 | 0.363 | 0.378 | 0.396 | 0.399 | 6.45 |
| 2 s window, sigma 0.10, at 0.08 | 0.080 | 0.174 | 0.181 | 0.187 | 0.186 | 9.61 |
| 2 s window, sigma 0.10, at ceiling | 0.167 | 0.022 | 0.023 | 0.023 | 0.018 | 15.62 |
| 1 s, sigma 0.03 | 0.050 | 0.248 | 0.313 | 0.311 | 0.403 | 5.72 |
| 1 s, tent weight 2.0 | 0.050 | 0.354 | 0.380 | 0.428 | 0.431 | 6.71 |
| 2 s window, sigma 0.10, tent 2.0 | 0.050 | 0.304 | 0.316 | 0.333 | 0.327 | 8.41 |
