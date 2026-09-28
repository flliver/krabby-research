```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_21-32-46/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5393322153310163  p25=0.481945031988507  p75=0.5771508535695813
tippy_tap_fraction median=0.23127865511971474
slip_ratio         median=0.2073879637435912
tracking_ratio     median=0.6757951592891054  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}

by hold:
     stand: cmd 0.00 -> achieved 0.066 m/s | tripod median=0.4532534102062298 (n=100)
     creep: cmd 0.25 -> achieved 0.204 m/s | tripod median=0.6332270006386114 (n=96)
       low: cmd 0.35 -> achieved 0.188 m/s | tripod median=0.5949152111227155 (n=89)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
