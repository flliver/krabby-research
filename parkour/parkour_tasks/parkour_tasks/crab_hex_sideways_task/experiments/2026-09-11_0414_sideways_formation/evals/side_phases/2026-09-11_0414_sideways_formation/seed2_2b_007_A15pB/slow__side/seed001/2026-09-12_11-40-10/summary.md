```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.44137480350815306  p25=0.40093720285030976  p75=0.47394833652834545
tippy_tap_fraction median=0.3467367144877753
slip_ratio         median=0.42142251392873986
tracking_ratio     median=0.7480099709787811  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=87)
     creep: cmd 0.25 -> achieved 0.219 m/s | tripod median=0.6446538151958666 (n=100)
       low: cmd 0.35 -> achieved 0.223 m/s | tripod median=0.6513985038470229 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
