```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9900.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3529476916587328  p25=0.2684294882345591  p75=0.42032533901780345
tippy_tap_fraction median=0.38984674329501917
slip_ratio         median=0.448684151563763
tracking_ratio     median=0.598933547504803  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=89)
     creep: cmd 0.25 -> achieved 0.201 m/s | tripod median=0.6088737162984184 (n=97)
       low: cmd 0.35 -> achieved 0.144 m/s | tripod median=0.43164885034962686 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
