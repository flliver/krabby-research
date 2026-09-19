```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_13-42-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.48440906070686696  p25=0.4179560395237857  p75=0.5292880921076784
tippy_tap_fraction median=0.20930489312842254
slip_ratio         median=0.19410436994868674
tracking_ratio     median=0.5886384580857928  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.69
terminations={'fall': 31, 'schedule_complete': 69}

by hold:
     stand: cmd 0.00 -> achieved 0.063 m/s | tripod median=0.42185982833970026 (n=100)
     creep: cmd 0.25 -> achieved 0.164 m/s | tripod median=0.547957782255232 (n=96)
       low: cmd 0.35 -> achieved 0.183 m/s | tripod median=0.5129101363512741 (n=85)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
