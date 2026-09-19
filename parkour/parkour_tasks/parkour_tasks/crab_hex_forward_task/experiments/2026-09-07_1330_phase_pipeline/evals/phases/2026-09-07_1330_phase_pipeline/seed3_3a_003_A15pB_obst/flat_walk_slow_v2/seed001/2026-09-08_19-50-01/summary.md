```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4899978762080762  p25=0.4334648024200303  p75=0.5440841380877834
tippy_tap_fraction median=0.2524173357506691
slip_ratio         median=0.19289587006113407
tracking_ratio     median=0.5825015463901033  (achieved/commanded vx, walking holds; n=88)
schedule_completion_rate=0.64
terminations={'schedule_complete': 64, 'fall': 36}

by hold:
     stand: cmd 0.00 -> achieved 0.113 m/s | tripod median=0.4713725433850077 (n=100)
     creep: cmd 0.25 -> achieved 0.161 m/s | tripod median=0.49630564694343465 (n=88)
       low: cmd 0.35 -> achieved 0.175 m/s | tripod median=0.5275366742740286 (n=67)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
