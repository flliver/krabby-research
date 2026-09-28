```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-07_23-13-07/model_24995.pt
episodes   : 100  unscored: 98
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.06536725196540048  p25=0.05641777065530845  p75=0.07431673327549253
tippy_tap_fraction median=0.5454545454545454
slip_ratio         median=0.2499103677698415
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.180 m/s | tripod median=0.06536725196540048 (n=2)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
